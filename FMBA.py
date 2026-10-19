import torch
import torch.nn as nn
import torch.nn.functional as F


class FMBA(nn.Module):
    """
    Frequency-aware Multi-branch Attention (FMBA)

    - FFT branch: band-wise 2D FFT -> magnitude -> log -> normalization -> 1x1 projection as V
    - SWT (level-1, Haar) branch: LL/LH/HL/HH via depthwise conv (no downsampling)
    - Attention: channel-wise attention scores [B, D, D] where D=embed_dim (default 32)
      Q from LL, K from {LH, HL, HH}, V from FFT.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 32,
        use_scale: bool = True,
        fusion_ratio=(1.0, 1.0, 1.0),
        eps: float = 1e-6,
        pad_mode: str = "circular",
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.embed_dim = int(embed_dim)
        self.use_scale = bool(use_scale)
        self.eps = float(eps)
        self.pad_mode = str(pad_mode)

        # Learnable projections (moved from function scope into __init__)
        self.v_proj = nn.Conv2d(self.in_channels, self.embed_dim, kernel_size=1,padding=0, bias=False)

        # Q/K projections (keep behavior: Q uses 1x1, K uses 3x3)
        # NOTE: In the original code q1_proj had padding=1 for 1x1 conv; here we use the standard padding=0.
        self.q_proj = nn.Conv2d(self.in_channels, self.embed_dim, kernel_size=1, padding=0, bias=False)
        self.k1_proj = nn.Conv2d(self.in_channels, self.embed_dim, kernel_size=3, padding=1, bias=False)
        self.k2_proj = nn.Conv2d(self.in_channels, self.embed_dim, kernel_size=3, padding=1, bias=False)
        self.k3_proj = nn.Conv2d(self.in_channels, self.embed_dim, kernel_size=3, padding=1, bias=False)

        # Fusion weights as a buffer so it moves with .to(device)
        self.register_buffer("fusion_ratio", torch.tensor(fusion_ratio, dtype=torch.float32))

        # Haar kernels (level-1) as buffers
        self.register_buffer(
            "haar_LL",
            torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32).view(1, 1, 2, 2),
        )
        self.register_buffer(
            "haar_LH",
            torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=torch.float32).view(1, 1, 2, 2),
        )
        self.register_buffer(
            "haar_HL",
            torch.tensor([[0.5, -0.5], [0.5, -0.5]], dtype=torch.float32).view(1, 1, 2, 2),
        )
        self.register_buffer(
            "haar_HH",
            torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32).view(1, 1, 2, 2),
        )

    def extract_fft_features(self, hsi_tensor: torch.Tensor) -> torch.Tensor:
        """FFT magnitude features projected to embed_dim as V: [B, D, H, W]."""
        hsi_complex = torch.fft.fft2(hsi_tensor, dim=(-2, -1))
        hsi_fft_shifted = torch.fft.fftshift(hsi_complex, dim=(-2, -1))

        # Magnitude + numeric stability
        fft_magnitude = torch.abs(hsi_fft_shifted) + self.eps

        # Log transform
        fft_magnitude = torch.log(fft_magnitude + 1.0)

        # Same normalization behavior as the original code (global mean/std)
        fft_magnitude = (fft_magnitude - fft_magnitude.mean()) / (fft_magnitude.std() + self.eps)

        # Project to embed_dim
        fft_proj = self.v_proj(fft_magnitude)  # [B, D, H, W]
        return fft_proj

    def _make_kernel(self, base_kernel: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Repeat a [1,1,2,2] kernel to depthwise conv format [C,1,2,2]."""
        return base_kernel.to(dtype=dtype, device=device).repeat(self.in_channels, 1, 1, 1)

    def _wavelet_transform(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """
        SWT-like (level-1) depthwise conv without downsampling.
        Original behavior: pad right/bottom by 1 then valid conv -> [H,W].
        """
        x_pad = F.pad(x, (0, 1, 0, 1), mode=self.pad_mode)  # [B,C,H+1,W+1]
        out = F.conv2d(x_pad, kernel, stride=1, padding=0, groups=self.in_channels)  # [B,C,H,W]
        return out

    @staticmethod
    def _channel_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, use_scale: bool) -> torch.Tensor:
        """
        Channel-wise attention:
        - Q, K: [B, D, H, W] -> mean over spatial -> [B, D]
        - scores: [B, D, D]
        - V: [B, D, H, W] -> [B, D, HW]
        - output: [B, D, H, W]
        """
        q_channel = torch.mean(Q, dim=(2, 3))  # [B,D]
        k_channel = torch.mean(K, dim=(2, 3))  # [B,D]

        scores = torch.matmul(q_channel.unsqueeze(2), k_channel.unsqueeze(1))  # [B,D,D]

        if use_scale:
            scores = scores / (Q.size(1) ** 0.5)

        attn = F.softmax(scores, dim=-1)  # [B,D,D]

        v_reshaped = V.view(V.size(0), V.size(1), -1)  # [B,D,HW]
        out = torch.matmul(attn, v_reshaped)  # [B,D,HW]
        return out.view_as(V)  # [B,D,H,W]

    def forward(self, hsi_tensor: torch.Tensor) -> torch.Tensor:
        """
        Input:  [B, C, H, W]
        Output: [B, D, H, W]
        """
        B, C, H, W = hsi_tensor.shape
        if C != self.in_channels:
            raise ValueError(f"FMBA was initialized with in_channels={self.in_channels}, but got input with C={C}.")

        # V from FFT
        fft_v = self.extract_fft_features(hsi_tensor)  # [B,D,H,W]

        # SWT subbands (level-1 Haar)
        dtype, device = hsi_tensor.dtype, hsi_tensor.device
        LL = self._wavelet_transform(hsi_tensor, self._make_kernel(self.haar_LL, dtype, device))
        LH = self._wavelet_transform(hsi_tensor, self._make_kernel(self.haar_LH, dtype, device))
        HL = self._wavelet_transform(hsi_tensor, self._make_kernel(self.haar_HL, dtype, device))
        HH = self._wavelet_transform(hsi_tensor, self._make_kernel(self.haar_HH, dtype, device))

        # Project Q/K into embed_dim
        Q = self.q_proj(LL)      # [B,D,H,W]
        K1 = self.k1_proj(LH)    # [B,D,H,W]
        K2 = self.k2_proj(HL)    # [B,D,H,W]
        K3 = self.k3_proj(HH)    # [B,D,H,W]

        # 3-branch attention, shared Q, shared V
        branch1 = self._channel_attention(Q, K1, fft_v, self.use_scale)
        branch2 = self._channel_attention(Q, K2, fft_v, self.use_scale)
        branch3 = self._channel_attention(Q, K3, fft_v, self.use_scale)

        fr = self.fusion_ratio.to(device=device, dtype=branch1.dtype)
        fused = branch1 * fr[0] + branch2 * fr[1] + branch3 * fr[2]
        fused = fused / (fr.sum() + self.eps)

        return fused


# Optional helper for quick usage (keeps a similar "function call" feel)
def extract_dwt_features(hsi_tensor: torch.Tensor, fmba: FMBA) -> torch.Tensor:
    """Backward-friendly wrapper: call FMBA forward."""
    return fmba(hsi_tensor)


def extract_fft_features(hsi_tensor: torch.Tensor, fmba: FMBA) -> torch.Tensor:
    """Backward-friendly wrapper: call FMBA FFT path only."""
    return fmba.extract_fft_features(hsi_tensor)
