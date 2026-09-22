import os
import sys

import pytest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from domain.FrontEnd import ConvFrontEnd, _gaussian_kernel2d, attach_frontend


class TestGaussianKernel:
    def test_sums_to_one_and_symmetric(self):
        k = _gaussian_kernel2d(8, 8 / 6.0)
        assert k.shape == (8, 8)
        assert torch.allclose(k.sum(), torch.tensor(1.0), atol=1e-5)
        assert torch.allclose(k, k.flip(0), atol=1e-6)
        assert torch.allclose(k, k.flip(1), atol=1e-6)


class TestConvFrontEnd:
    def test_output_shape_stride2(self):
        fe = ConvFrontEnd(kernel_size=32, stride=2)
        x = torch.randn(1, 3, 1024, 1024)
        out = fe(x)
        assert out.shape == (1, 3, 512, 512)

    def test_gaussian_init_no_channel_mixing(self):
        fe = ConvFrontEnd(kernel_size=32, stride=2)
        w = fe.conv.weight
        for c in range(3):
            for o in range(3):
                if c != o:
                    assert torch.allclose(w[o, c], torch.zeros_like(w[o, c]))
        # os 3 filtros sao iguais
        assert torch.allclose(w[0, 0], w[1, 1], atol=1e-6)
        assert torch.allclose(w[0, 0], w[2, 2], atol=1e-6)

    def test_kernel_covers_wide_region(self):
        # com stride 2 e kernel 32, cada pixel de saida depende de 32x32 de entrada
        fe = ConvFrontEnd(kernel_size=32, stride=2)
        x = torch.zeros(1, 3, 64, 64)
        x[0, 0, 0, 0] = 1.0
        out = fe(x)
        assert out.shape == (1, 3, 32, 32)
        # um pixel de entrada afeta um bloco ~8x8 de saida (kernel 32, stride 2)
        affected = int((out[0, 0] > 0).sum())
        assert 40 <= affected <= 100


class TestAttachFrontEnd:
    def test_conv_reduces_to_target(self):
        from domain.Vgg16 import SetupModelVgg
        model = SetupModelVgg().setup_model(torch.device("cpu"))
        attach_frontend(model, mode="conv", target_size=512)
        x = torch.randn(1, 3, 1024, 1024)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1)

    def test_avg_reduces_to_target(self):
        from domain.Vgg16 import SetupModelVgg
        model = SetupModelVgg().setup_model(torch.device("cpu"))
        attach_frontend(model, mode="avg", target_size=512)
        x = torch.randn(1, 3, 1024, 1024)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1)

    def test_frontend_in_state_dict_and_params(self):
        from domain.Vgg16 import SetupModelVgg
        model = SetupModelVgg().setup_model(torch.device("cpu"))
        attach_frontend(model, mode="conv", target_size=512)
        sd = model.state_dict()
        assert any(k.startswith("conv_frontend") for k in sd.keys())
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert any(n.startswith("conv_frontend") for n in trainable)

    def test_invalid_mode_raises(self):
        from domain.Vgg16 import SetupModelVgg
        model = SetupModelVgg().setup_model(torch.device("cpu"))
        with pytest.raises(ValueError):
            attach_frontend(model, mode="bico")
