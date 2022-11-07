# MIT License Copyright (c) 2022 joh-fischer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import torch
import torch.nn as nn
import numpy as np

if __name__ == "__main__":
    from transforms import Identity, MovingAvg, Downsample


class MCNN(nn.Module):
    def __init__(self, ts_shape: tuple, n_classes: int, pool_factor: int,
                 kernel_size: float or int, transformations: dict):
        """
        Multi-Scale Convolutional Neural Network for Time Series Classification - Cui et al. (2016).

        Args:
          ts_shape (tuple):           shape of the time series, e.g. (1, 9) for uni-variate time series
                                      with length 9, or (3, 9) for multivariate time series with length 9
                                      and three features
          n_classes (int):            number of classes
          pool_factor (int):          length of feature map after max pooling, usually in {2,3,5}
          kernel_size (int or float): filter size for convolutional layers, usually ratio in {0.05, 0.1, 0.2}
                                      times the length of time series
          transformations (dict):     dictionary with key value pairs specifying the transformations
                                      in the format 'name': {'class': <TransformClass>, 'params': <parameters>}
        """
        assert len(ts_shape) == 2, "Expecting shape in format (n_channels, seq_len)!"

        super(MCNN, self).__init__()

        self.ts_shape = ts_shape
        self.n_classes = n_classes
        self.pool_factor = pool_factor
        self.kernel_size = int(self.ts_shape[1] * kernel_size) if kernel_size < 1 else int(kernel_size)

        self.loss = nn.CrossEntropyLoss

        # layer settings
        self.local_conv_filters = 256
        self.local_conv_activation = nn.ReLU  # nn.Sigmoid in original implementation
        self.full_conv_filters = 256
        self.full_conv_activation = nn.ReLU  # nn.Sigmoid in original implementation
        self.fc_units = 256
        self.fc_activation = nn.ReLU  # nn.Sigmoid in original implementation

        # setup branches
        self.branches = self._setup_branches(transformations)
        self.n_branches = len(self.branches)

        # full convolution
        in_channels = self.local_conv_filters * self.n_branches
        # kernel shouldn't exceed the length (length is always pool factor?)
        full_conv_kernel_size = int(min(self.kernel_size, int(self.pool_factor)))
        self.full_conv = nn.Conv1d(in_channels, self.full_conv_filters,
                                   kernel_size=full_conv_kernel_size,
                                   padding='same')
        # ISSUE: In the TensorFlow implementation of https://github.com/hfawaz/dl-4-tsc they implement the pool_size
        # as follows: pool_size = int(int(full_conv.shape[1])/pool_factor).
        # However, this makes no sense here, as the output length of the timeseries after convolution (denoted as n) is
        # equal to pool_factor due to the local max pooling operation. Dividing n/pool_factor always yields a
        # max pooling size of 1, which is simply identity mapping. E.g. the output shape after local convolution is
        # (batch_size, len_ts, n_channels) and the pooling factor ensures that len_ts is equal to pool_factor. Hence,
        # the shape after local pooling is (batch_size, pool_factor, n_channels) and shape[1]/pool_factor = 1.
        pool_size = 1
        self.full_conv_pool = nn.MaxPool1d(pool_size)

        # fully-connected
        self.flatten = nn.Flatten()
        in_features = int(self.pool_factor * self.full_conv_filters)
        self.fc = nn.Linear(in_features, self.fc_units, bias=True)

        # softmax output
        self.output = nn.Linear(self.fc_units, self.n_classes, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        xs = [self.branches[idx](x) for idx in range(self.n_branches)]
        x = torch.cat(xs, dim=1)

        x = self.full_conv(x)
        x = self.full_conv_activation()(x)
        x = self.full_conv_pool(x)

        x = self.flatten(x)
        x = self.fc(x)
        x = self.fc_activation()(x)

        x = self.output(x)
        x = self.softmax(x)

        return x

    def _build_local_branch(self, name: str, transform: nn.Module, params: list):
        """
        Build transformation and local convolution branch.

        Args:
          name (str):   Name of the branch.
          transform (nn.Module):  Transformation class applied in this branch.
          params (list):   Parameters for the transformation, with the first parameter always being the input shape.
        Returns:
          branch:   Sequential model containing transform, local convolution, activation, and max pooling.
        """
        branch = nn.Sequential()
        # transformation
        branch.add_module(name + '_transform', transform(*params))
        # local convolution
        branch.add_module(name + '_conv', nn.Conv1d(self.ts_shape[0], self.local_conv_filters,
                                                    kernel_size=self.kernel_size, padding='same'))
        branch.add_module(name + '_activation', self.local_conv_activation())
        # local max pooling (ensure that outputs all have length equal to pool factor)
        pool_size = int(np.ceil(branch[0].output_shape[1] / self.pool_factor))
        assert pool_size > 1, "ATTENTION: pool_size can not be 0 or 1, as the lengths are then not equal" \
                              "for concatenation!"
        branch.add_module(name + '_pool', nn.MaxPool1d(pool_size), ceil_mode=True)  # default stride equal to pool size

        return branch

    def _setup_branches(self, transformations: dict):
        """
        Setup all branches for the local convolution.

        Args:
          transformations:  Dictionary containing the transformation classes and parameter settings.
        Returns:
          branches: List of sequential models with local convolution per branch.
        """
        branches = []
        for transform_name in transformations:
            transform_class = transformations[transform_name]['class']
            parameter_list = transformations[transform_name]['params']

            # create transform layer for each parameter configuration
            if parameter_list:
                for param in parameter_list:
                    if np.isscalar(param):
                        name = transform_name + '_' + str(param)
                        branch = self._build_local_branch(name, transform_class, [self.ts_shape, param])
                    else:
                        branch = self._build_local_branch(transform_name, transform_class,
                                                          [self.ts_shape] + list(param))
                    branches.append(branch)
            else:
                branch = self._build_local_branch(transform_name, transform_class, [self.ts_shape])
                branches.append(branch)

        return torch.nn.ModuleList(branches)


if __name__ == "__main__":
    transform_dict = {
        'identity': {
            'class': Identity,
            'params': None
        },
        'movingAvg': {
            'class': MovingAvg,
            'params': [3, 4, 5]
        },
        'downsample': {
            'class': Downsample,
            'params': [2, 3]
        }
    }

    classes = 7
    seq_len = 30
    channels = 5

    kernel_size = int(seq_len * 0.1)
    pooling_factor = 3

    print("Model with {} classes, sequence length of {}, and {} channels (multivariate)".format(
        classes, seq_len, channels
    ))

    model = MCNN((channels, seq_len), classes, pooling_factor, kernel_size, transform_dict)

    print("--- Model:\n", model)

    # forward pass
    data = torch.rand((16, channels, seq_len))
    print("--- Input shape:", data.shape)
    print("--- Model sample output:", model.forward(data).shape)
