import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel2d(kernel_size, sigma):
    ax = torch.arange(kernel_size) - (kernel_size - 1) / 2
    g = torch.exp(-0.5 * (ax / sigma) ** 2)
    g = g / g.sum()
    return g[:, None] @ g[None, :]


class ConvFrontEnd(nn.Module):
    """Front-end de convolucao com kernel grande antes do backbone.

    Reduz a imagem por `stride` com um filtro anti-aliasing inicializado
    como gaussiano 2D (aprendivel). Cada pixel de saida integra uma
    vizinhanca de `kernel_size`, o que preserva melhor texturas finas
    que uma bilinear direta no mesmo fator de reducao.
    """

    def __init__(self, in_channels=3, out_channels=3, kernel_size=32, stride=2):
        super().__init__()
        # padding = k/2 - 1: com H divisivel por stride, saida = H/stride exato
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=kernel_size // 2 - 1,
        )
        sigma = kernel_size / 6.0
        k = _gaussian_kernel2d(kernel_size, sigma)
        with torch.no_grad():
            w = torch.zeros(out_channels, in_channels, kernel_size, kernel_size)
            for c in range(min(in_channels, out_channels)):
                w[c, c] = k
            self.conv.weight.copy_(w)
            self.conv.bias.zero_()

    def forward(self, x):
        return self.conv(x)


def attach_frontend(model, mode="conv", target_size=512, kernel_size=32):
    """Anexa um front-end de reducao ao forward do modelo.

    Mantem os atributos do backbone intactos (Grad-CAM, unfreeze).
    O front-end fica registrado em model.conv_frontend.

    mode:
      'avg'  -> AveragePool2d(stride=2): downscale fixo, nao aprende
      'conv' -> ConvFrontEnd(k=kernel_size, s=2): filtro grande aprendido

    O forward final faz interpolate p/ target_size (cobre qualquer
    defasagem de 1 pixel do conv).
    """
    if mode == "avg":
        frontend = nn.AvgPool2d(kernel_size=2, stride=2)
    elif mode == "conv":
        frontend = ConvFrontEnd(kernel_size=kernel_size, stride=2)
    else:
        raise ValueError(f"frontend '{mode}' nao suportado (use 'avg' ou 'conv')")

    model.conv_frontend = frontend
    orig_forward = model.forward

    def forward(x):
        x = frontend(x)
        if tuple(x.shape[-2:]) != (target_size, target_size):
            x = F.interpolate(
                x, size=(target_size, target_size),
                mode="bilinear", align_corners=False,
            )
        return orig_forward(x)

    model.forward = forward
    return model
