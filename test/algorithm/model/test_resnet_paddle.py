import pytest
import numpy as np
try:
    import paddle
    from algorithm.model.resnet_paddle import resnet18
    PADDLE_INSTALLED = True
except ImportError:
    PADDLE_INSTALLED = False

@pytest.mark.skipif(not PADDLE_INSTALLED, reason="paddlepaddle not installed")
def test_resnet18_paddle():
    model = resnet18(class_dim=10)
    x = paddle.randn([2, 3, 224, 224])
    outputs = model(x)
    assert outputs.shape == [2, 10]