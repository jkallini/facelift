import torch

def get_relative_positions(seq_len: int, offset: int = 0) -> torch.tensor:
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    diff = x - y

    # Zero out the band where |i - j| < offset
    out = torch.sign(diff) * torch.clamp(diff.abs() - offset, min=0)
    return out


def get_alibi_slope(num_heads):
    x = (2 ** 8) ** (1 / num_heads)
    return (
        torch.tensor([1 / x ** (i + 1) for i in range(num_heads)])
        .unsqueeze(-1)
        .unsqueeze(-1)
    )