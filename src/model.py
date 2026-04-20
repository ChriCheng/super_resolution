import mindspore.nn as nn
import mindspore.ops as ops


class ESPCN(nn.Cell):
    """A lightweight ESPCN network for x4 super-resolution.

    Input : (N, 3, H, W)
    Output: (N, 3, 4H, 4W)
    """

    def __init__(self, scale: int = 4, in_channels: int = 3, hidden1: int = 64, hidden2: int = 32):
        super().__init__()
        if scale != 4:
            raise ValueError("This homework implementation is configured for x4 only.")
        self.conv1 = nn.Conv2d(in_channels, hidden1, kernel_size=5, pad_mode="pad", padding=2, has_bias=True)
        self.conv2 = nn.Conv2d(hidden1, hidden2, kernel_size=3, pad_mode="pad", padding=1, has_bias=True)
        self.conv3 = nn.Conv2d(hidden2, in_channels * (scale ** 2), kernel_size=3, pad_mode="pad", padding=1,
                               has_bias=True)
        self.relu = nn.ReLU()
        self.shuffle = ops.DepthToSpace(scale)

    def construct(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.shuffle(x)
        return x
