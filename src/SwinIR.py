import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


def window_partition(x, window_size: int):
    """
    x: (B, H, W, C)
    return: (num_windows*B, window_size*window_size, C)
    """
    b, h, w, c = x.shape
    x = x.reshape(b, h // window_size, window_size, w // window_size, window_size, c)
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = x.reshape(-1, window_size * window_size, c)
    return windows


def window_reverse(windows, window_size: int, h: int, w: int, b: int):
    """
    windows: (num_windows*B, window_size*window_size, C)
    return: (B, H, W, C)
    """
    x = windows.reshape(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    x = x.reshape(b, h, w, -1)
    return x


class Mlp(nn.Cell):
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Dense(hidden_features, out_features)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class WindowAttention(nn.Cell):
    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.proj = nn.Dense(dim, dim)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x):
        """
        x: (B_, N, C)
        """
        b_, n, c = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(b_, n, 3, self.num_heads, c // self.num_heads)
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        attn = self.softmax(attn)

        x = ops.matmul(attn, v)
        x = ops.transpose(x, (0, 2, 1, 3))
        x = x.reshape(b_, n, c)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Cell):
    """
    简化版：保留 window attention + MLP + residual
    为了兼容你当前工程，这里不做 shifted window mask 的复杂实现。
    """
    def __init__(self, dim: int, num_heads: int, window_size: int = 8, mlp_ratio: float = 2.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.norm1 = nn.LayerNorm((dim,))
        self.attn = WindowAttention(dim=dim, window_size=window_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm((dim,))
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def construct(self, x):
        """
        x: (B, H, W, C)
        """
        b, h, w, c = x.shape
        if h % self.window_size != 0 or w % self.window_size != 0:
            raise ValueError(
                f"Feature map size {(h, w)} must be divisible by window_size={self.window_size}. "
                "Please crop input sizes accordingly."
            )

        shortcut = x
        x = self.norm1(x)

        x_windows = window_partition(x, self.window_size)
        attn_windows = self.attn(x_windows)
        x = window_reverse(attn_windows, self.window_size, h, w, b)

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class ResidualSwinBlock(nn.Cell):
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int = 8, mlp_ratio: float = 2.0):
        super().__init__()
        self.blocks = nn.CellList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, pad_mode="pad", padding=1, has_bias=True)

    def construct(self, x):
        """
        x: (B, C, H, W)
        """
        shortcut = x
        x = ops.transpose(x, (0, 2, 3, 1))  # BCHW -> BHWC
        for blk in self.blocks:
            x = blk(x)
        x = ops.transpose(x, (0, 3, 1, 2))  # BHWC -> BCHW
        x = self.conv(x)
        return shortcut + x


class SwinIR(nn.Cell):
    """
    适配当前作业的轻量版 SwinIR:
    - 输入:  (N, 3, H, W)
    - 输出:  (N, 3, 4H, 4W)
    """
    def __init__(
        self,
        scale: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths=(4, 4, 4, 4),
        num_heads=(6, 6, 6, 6),
        window_size: int = 8,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        if scale != 4:
            raise ValueError("This homework implementation is configured for x4 only.")

        self.scale = scale
        self.window_size = window_size

        self.conv_first = nn.Conv2d(in_chans, embed_dim, kernel_size=3, pad_mode="pad", padding=1, has_bias=True)

        self.layers = nn.CellList([
            ResidualSwinBlock(
                dim=embed_dim,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
            )
            for i in range(len(depths))
        ])

        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, pad_mode="pad", padding=1, has_bias=True)

        self.conv_before_upsample = nn.Conv2d(
            embed_dim, 64, kernel_size=3, pad_mode="pad", padding=1, has_bias=True
        )
        self.act = nn.LeakyReLU(alpha=0.1)

        self.conv_last = nn.Conv2d(
            64, in_chans * (scale ** 2), kernel_size=3, pad_mode="pad", padding=1, has_bias=True
        )
        self.upsample = ops.DepthToSpace(scale)

    def construct(self, x):
        """
        x: (B, 3, H, W), where H and W should be divisible by window_size
        """
        x = self.conv_first(x)
        shallow = x

        for layer in self.layers:
            x = layer(x)

        x = self.conv_after_body(x)
        x = x + shallow

        x = self.conv_before_upsample(x)
        x = self.act(x)
        x = self.conv_last(x)
        x = self.upsample(x)
        return x


class ESPCN(nn.Cell):
    """
    为了兼容你现有 train.py / eval.py 的调用方式，保留 ESPCN 这个名字，
    实际内部换成 SwinIR。
    """
    def __init__(self, scale: int = 4):
        super().__init__()
        self.model = SwinIR(scale=scale)

    def construct(self, x):
        return self.model(x)