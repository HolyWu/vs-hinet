import math
import os

import numpy as np
import torch
import vapoursynth as vs

dirname = os.path.dirname(__file__)


def HINet(
    clip: vs.VideoNode,
    mode: int = 0,
    tile_x: int = 0,
    tile_y: int = 0,
    tile_pad: int = 16,
    device_type: str = 'cuda',
    device_index: int = 0,
    fp16: bool = False,
    trt: bool = False,
    save_trt_model: bool = False,
) -> vs.VideoNode:
    '''
    HINet: Half Instance Normalization Network for Image Restoration

    Parameters:
        clip: Clip to process. Only RGB format with float sample type of 32 bit depth is supported.

        mode: Mode of operation.
            0 = deblur, using GoPro dataset
            1 = deblur, using REDS dataset
            2 = denoise
            3 = derain

        tile_x, tile_y: Tile width and height respectively, 0 for no tiling.
            It's recommended that the input's width and height is divisible by the tile's width and height respectively.
            Set it to the maximum value that your GPU supports to reduce its impact on the output.

        tile_pad: Tile padding.

        device_type: Device type on which the tensor is allocated. Must be 'cuda' or 'cpu'.

        device_index: Device ordinal for the device type.

        fp16: fp16 mode for faster and more lightweight inference on cards with Tensor Cores.

        trt: Use TensorRT model to accelerate inference.

        save_trt_model: Save the converted TensorRT model and does no inference. One-frame evaluation is enough.
            Each model can only work with a specific dimension, hence you must save the model first for dimensions which have not been converted.
            Keep in mind that models are not portable across platforms or TensorRT versions and are specific to the exact GPU model they were built on.
    '''
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('HINet: This is not a clip')

    if clip.format.id != vs.RGBS:
        raise vs.Error('HINet: Only RGBS format is supported')

    if mode not in [0, 1, 2, 3]:
        raise vs.Error('HINet: mode must be 0, 1, 2, or 3')

    device_type = device_type.lower()

    if device_type not in ['cuda', 'cpu']:
        raise vs.Error("HINet: device_type must be 'cuda' or 'cpu'")

    if device_type == 'cuda' and not torch.cuda.is_available():
        raise vs.Error('HINet: CUDA is not available')

    if trt and save_trt_model:
        raise vs.Error('HINet: both trt and save_trt_model cannot be True at the same time')

    if (trt or save_trt_model) and device_type == 'cpu':
        raise vs.Error('HINet: TensorRT is not supported for CPU device')

    if os.path.getsize(os.path.join(dirname, 'HINet-GoPro.pth')) == 0:
        raise vs.Error("HINet: model files have not been downloaded. run 'python -m vshinet' first")

    if tile_x > 0 and tile_y > 0:
        trt_width = (min(tile_x + tile_pad, clip.width) + 15) & ~15
        trt_height = (min(tile_y + tile_pad, clip.height) + 15) & ~15
    else:
        trt_width = (clip.width + 15) & ~15
        trt_height = (clip.height + 15) & ~15

    device = torch.device(device_type, device_index)
    if device_type == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    if mode == 0:
        trt_model_name = f'HINet-GoPro_trt_{trt_width}x{trt_height}{"_fp16" if fp16 else ""}.pth'
        model_name = trt_model_name if trt else 'HINet-GoPro.pth'
    elif mode == 1:
        trt_model_name = f'HINet-REDS_trt_{trt_width}x{trt_height}{"_fp16" if fp16 else ""}.pth'
        model_name = trt_model_name if trt else 'HINet-REDS.pth'
    elif mode == 2:
        trt_model_name = f'HINet-SIDD-1x_trt_{trt_width}x{trt_height}{"_fp16" if fp16 else ""}.pth'
        model_name = trt_model_name if trt else 'HINet-SIDD-1x.pth'
    else:
        trt_model_name = f'HINet-Rain13k_trt_{trt_width}x{trt_height}{"_fp16" if fp16 else ""}.pth'
        model_name = trt_model_name if trt else 'HINet-Rain13k.pth'

    model_path = os.path.join(dirname, model_name)
    trt_model_path = os.path.join(dirname, trt_model_name)

    if trt:
        from torch2trt import TRTModule

        model = TRTModule()
    else:
        from .hinet_arch import HINet as net

        model = net(wf=64, hin_position_left=3 if mode < 2 else 0, hin_position_right=4)

    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model['params'] if 'params' in pretrained_model else pretrained_model, strict=True)
    model.eval()
    model.to(device)
    if fp16:
        model.half()

    if save_trt_model:
        with torch.inference_mode():
            from torch2trt import torch2trt

            x = torch.zeros((1, 3, trt_height, trt_width), dtype=torch.half if fp16 else torch.float, device=device)
            model_trt = torch2trt(model, [x], fp16_mode=fp16)
            torch.save(model_trt.state_dict(), trt_model_path)
            vs.core.log_message(1, f"'{trt_model_path}' saved successfully")
            return clip

    @torch.inference_mode()
    def hinet(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        img_lq = frame_to_tensor(f)
        if fp16:
            img_lq = img_lq.half()

        if tile_x > 0 and tile_y > 0:
            output = tile_process(img_lq, tile_x, tile_y, tile_pad, device, model)
        elif img_lq.size(2) % 16 == 0 and img_lq.size(3) % 16 == 0:
            output = model(img_lq.to(device))
        else:
            output = mod_pad(img_lq.to(device), 16, model)

        return tensor_to_frame(output, f.copy())

    return clip.std.ModifyFrame(clips=clip, selector=hinet)


def frame_to_tensor(f: vs.VideoFrame) -> torch.Tensor:
    arr = np.stack([np.asarray(f[plane]) for plane in range(f.format.num_planes)])
    return torch.from_numpy(arr).unsqueeze(0)


def tensor_to_frame(t: torch.Tensor, f: vs.VideoFrame) -> vs.VideoFrame:
    arr = t.squeeze(0).detach().cpu().numpy()
    for plane in range(f.format.num_planes):
        np.copyto(np.asarray(f[plane]), arr[plane, :, :])
    return f


def tile_process(img: torch.Tensor, tile_x: int, tile_y: int, tile_pad: int, device: torch.device, model: torch.nn.Module) -> torch.Tensor:
    height, width = img.shape[2:]

    # start with black image
    output = img.new_zeros(img.shape)

    tiles_x = math.ceil(width / tile_x)
    tiles_y = math.ceil(height / tile_y)

    # loop over all tiles
    for y in range(tiles_y):
        for x in range(tiles_x):
            # extract tile from input image
            ofs_x = x * tile_x
            ofs_y = y * tile_y

            # input tile area on total image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_x, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_y, height)

            # input tile area on total image with padding
            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            # input tile dimensions
            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y

            input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

            # process tile
            input_tile = input_tile.to(device)
            if input_tile.size(2) % 16 == 0 and input_tile.size(3) % 16 == 0:
                output_tile = model(input_tile)
            else:
                output_tile = mod_pad(input_tile, 16, model)

            # output tile area on total image
            output_start_x = input_start_x
            output_end_x = input_end_x
            output_start_y = input_start_y
            output_end_y = input_end_y

            # output tile area without padding
            output_start_x_tile = input_start_x - input_start_x_pad
            output_end_x_tile = output_start_x_tile + input_tile_width
            output_start_y_tile = input_start_y - input_start_y_pad
            output_end_y_tile = output_start_y_tile + input_tile_height

            # put tile into output image
            output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile[
                :, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile
            ]

    return output


def mod_pad(img: torch.Tensor, modulo: int, model: torch.nn.Module) -> torch.Tensor:
    import torch.nn.functional as F

    mod_pad_h, mod_pad_w = 0, 0
    h, w = img.shape[2:]

    if h % modulo != 0:
        mod_pad_h = modulo - h % modulo

    if w % modulo != 0:
        mod_pad_w = modulo - w % modulo

    img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    output = model(img)
    return output[:, :, :h, :w]
