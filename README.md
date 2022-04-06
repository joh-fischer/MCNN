# MCNN - PyTorch Implementation

This is a `PyTorch` implementation of the model presented in *Multi-scale convolutional neural networks for time series classification* of Z. Cui, W. Chen, and Y. Chen (2016).

The model architecture consists of three different stages
1. Transformation stage
2. Local convolution
3. Full convolution

In the transformation stage the input time series is transformed in three different ways: identity mapping, down-sampling, and smoothing.
For the latter two several parameters are used, each resulting in a different transformed time series.
The time series as well as each of the transformed version of it is fed into a local convolutional layer, followed by a max pooling operation.
One important thing is, that the pooling size and stride of the max pooling operation are dynamically adapted to the length of the time series,
so that the output after max pooling is the same for all local convolutional branches. All outputs are then concatenated channel-wise and fed into another
convolutional layer, followed by a fully-connected and softmax layer.

## Usage

Additionally to identity mapping the original paper used two transformations: moving average and down-sampling. The window sizes for the
moving average transformations are `[3, 4, 5]` and the sampling rates for the down-sampling
operation are [`2, 3`]. Hence, the model is build with 6 branches, one identity branch, three multi-frequency branches (moving average),
and two multi-scale branches (down-sampling).
```python
from model import MCNN
from transforms import Identity, MovingAvg, Downsample

transformations = {
    'identity': {
        'class': Identity,
        'params': []
    },
    'movingAvg': {
        'class': MovingAvg,
        'params': [3,4,5]       # window sizes
    },
    'downsample': {
        'class': Downsample,
        'params': [2,3]       # sampling rates
    }
}

n_classes = 5
seq_len = 140
ts_shape = (1, seq_len)     # univariate ts should be written as multivariate ts with one channel
pool_factor = 4
kernel_size = int(seq_len) * 0.05

model = MCNN(ts_shape, n_classes, pool_factor, kernel_size, transformations)

print(model)
```

For further questions please don't hesitate to contact me.