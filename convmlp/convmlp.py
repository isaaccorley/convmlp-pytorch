import torch
import torch.nn as nn

from convmlp.blocks import ClassifierHead, ConvMLPStage, ConvStage, Tokenizer


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
        self.head = ClassifierHead(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.embed(x)
        x = self.conv_stage(x)
        x = self.convmlp_stages(x)
        x = self.head(x)
        return x
