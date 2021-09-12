import torch
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fn(x)


class GlobalAveragePooling2d(nn.Sequential):
    def __init__(self):
        super().__init__(Reduce("b c h w -> b c", "mean"))


class Classifier(nn.Sequential):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__(nn.Linear(input_dim, num_classes))


class MLP(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )


class Tokenizer(nn.Sequential):
    def __init__(self, in_channels: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__(
            nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=kernel_size, stride=2, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size, stride=2, padding=kernel_size // 2, dilation=1),
        )


class ConvBlock(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, input_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
        )


class MLPBlock(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__(
            Rearrange("b c h w -> b h w c"),
            nn.LayerNorm(input_dim),
            Residual(MLP(input_dim, hidden_dim)),
            nn.LayerNorm(input_dim),
            Rearrange("b h w c -> b c h w"),
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1, groups=input_dim, bias=False),
            Rearrange("b c h w -> b h w c"),
            nn.LayerNorm(input_dim),
            Residual(MLP(input_dim, hidden_dim)),
            Rearrange("b h w c -> b c h w"),
        )


class ConvStage(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, blocks: int):
        super().__init__(
            *[ConvBlock(input_dim, hidden_dim) for _ in range(blocks)],
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1)
        )


class ConvMLPStage(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        ratio: float,
        blocks: int,
        downsample: bool = True,
    ):
        super().__init__(
            *[MLPBlock(input_dim, int(input_dim * ratio)) for _ in range(blocks)],
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1) if downsample else nn.Identity()
        )


class ConvMLP(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        hidden_dim: int = 64,
        ratio: int = 1,
        levels: int = 3,
        c: int = 3,
        m: int = 2,
    ):
        super().__init__()
        self.embed = Tokenizer(in_channels, hidden_dim)
        self.conv_stage = ConvStage(hidden_dim, hidden_dim, hidden_dim, blocks=c)
        self.convmlp_stages = nn.Sequential(
            *[ConvMLPStage(hidden_dim, hidden_dim, ratio, blocks=m, downsample=True) for _ in range(levels - 1)],
            ConvMLPStage(hidden_dim, hidden_dim, ratio, blocks=m, downsample=False)
        )
        self.head = nn.Sequential(
            Rearrange("b c h w -> b h w c"),
            nn.LayerNorm(hidden_dim),
            Rearrange("b h w c -> b c h w"),
            GlobalAveragePooling2d(),
            Classifier(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.embed(x)
        x = self.conv_stage(x)
        x = self.convmlp_stages(x)
        x = self.head(x)
        return x
