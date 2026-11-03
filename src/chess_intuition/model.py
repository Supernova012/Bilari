"""Compact residual policy/value network. Requires the optional torch extra."""

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised in minimal installs
    raise ImportError(
        "The neural model requires PyTorch. Install project training dependencies "
        "with `python -m pip install -e '.[train]'`."
    ) from exc

from .board import ACTION_PLANES, FEATURE_CHANNELS


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.layers(x))


class ChessResNet(nn.Module):
    """A small board CNN with a 4,672-action policy and scalar value head."""

    def __init__(self, blocks: int = 2, channels: int = 16) -> None:
        super().__init__()
        if blocks < 1 or channels < 8:
            raise ValueError("Use at least one residual block and eight channels.")
        self.config = {"blocks": int(blocks), "channels": int(channels)}
        self.stem = nn.Sequential(
            nn.Conv2d(FEATURE_CHANNELS, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.policy_head = nn.Conv2d(channels, ACTION_PLANES, kernel_size=1)
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return policy logits [B,4672] and value estimates [B]."""
        trunk = self.tower(self.stem(x))
        policy = self.policy_head(trunk).permute(0, 2, 3, 1).reshape(x.shape[0], -1)
        value = self.value_head(trunk).squeeze(-1)
        return policy, value

