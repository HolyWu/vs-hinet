# HINet
Half Instance Normalization Network for Image Restoration, based on https://github.com/JingyunLiang/SwinIR.


## Dependencies
- [NumPy](https://numpy.org/install)
- [PyTorch](https://pytorch.org/get-started), preferably with CUDA. Note that `torchvision` and `torchaudio` are not required and hence can be omitted from the command.
- [VapourSynth](http://www.vapoursynth.com/)
- (Optional) [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html). Note that `uff` and `PyCUDA` are not required and hence can be skipped from the guide.
- (Optional) [torch2trt](https://nvidia-ai-iot.github.io/torch2trt/master/getting_started.html#install-without-plugins)


## Installation
```
pip install --upgrade vshinet
python -m vshinet
```


## Usage
```python
from vshinet import HINet

ret = HINet(clip)
```

See `__init__.py` for the description of the parameters.
