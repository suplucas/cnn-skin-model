import os
import sys
import numpy as np
from PIL import Image
import pytest
from hypothesis import given, strategies as st, settings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from application.preprocessing.PreProcessing import OpenCVPreprocessing


class TestOpenCVPreprocessing:
    def test_output_is_pil_rgb(self):
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        result = OpenCVPreprocessing()(img)
        assert result.mode == "RGB"
        assert isinstance(result, Image.Image)

    def test_output_size_matches_input(self):
        img = Image.new("RGB", (224, 224), color=(64, 128, 192))
        result = OpenCVPreprocessing()(img)
        assert result.size == (224, 224)

    def test_grayscale_produces_identical_channels(self):
        img = Image.new("RGB", (64, 64), color=(200, 50, 50))
        result = OpenCVPreprocessing(denoise=False, grayscale=True)(img)
        arr = np.array(result)
        assert np.allclose(arr[..., 0], arr[..., 1])
        assert np.allclose(arr[..., 1], arr[..., 2])

    def test_no_grayscale_preserves_color(self):
        img = Image.new("RGB", (64, 64), color=(200, 50, 50))
        result = OpenCVPreprocessing(denoise=False, grayscale=False)(img)
        arr = np.array(result)
        assert result.mode == "RGB"
        assert not np.allclose(arr[..., 0], arr[..., 2])

    def test_denoise_disabled_does_not_crash(self):
        img = Image.new("RGB", (100, 100), color=(10, 200, 10))
        result = OpenCVPreprocessing(denoise=False, grayscale=False)(img)
        assert result.size == (100, 100)

    @given(st.integers(min_value=50, max_value=500), st.integers(min_value=50, max_value=500))
    @settings(max_examples=10, deadline=None)  # denoise e CPU-bound; deadline de 200ms e flaky
    def test_various_sizes(self, w, h):
        img = Image.new("RGB", (w, h), color=(100, 150, 200))
        result = OpenCVPreprocessing()(img)
        assert result.size == (w, h)
        assert result.mode == "RGB"


class TestTransforms:
    def test_resize_center_crop_512(self):
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
        ])
        img = Image.new("RGB", (4000, 2700), color=(255, 0, 0))
        result = transform(img)
        assert result.size == (512, 512)

    def test_resize_center_crop_square(self):
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
        ])
        img = Image.new("RGB", (1024, 1024), color=(0, 255, 0))
        result = transform(img)
        assert result.size == (512, 512)

    @given(st.integers(min_value=600, max_value=2000), st.integers(min_value=600, max_value=2000))
    @settings(max_examples=5)
    def test_various_input_sizes(self, w, h):
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
        ])
        img = Image.new("RGB", (w, h), color=(0, 0, 255))
        result = transform(img)
        assert result.size == (512, 512)


class TestImageProcessing:
    def test_train_val_transforms_exist(self):
        from application.preprocessing.PreProcessing import ImageProcessing
        ip = ImageProcessing()
        assert hasattr(ip, 'train_transforms')
        assert hasattr(ip, 'val_transforms')

    def test_val_has_no_random(self):
        from application.preprocessing.PreProcessing import ImageProcessing
        from torchvision import transforms
        ip = ImageProcessing()
        val_str = str(ip.val_transforms)
        assert "RandomRotation" not in val_str
        assert "RandomHorizontalFlip" not in val_str
        assert "RandomVerticalFlip" not in val_str
        assert "ColorJitter" not in val_str

    def test_opencv_disabled(self):
        from application.preprocessing.PreProcessing import ImageProcessing
        ip = ImageProcessing(opencv=False)
        train_str = str(ip.train_transforms)
        val_str = str(ip.val_transforms)
        assert "OpenCVPreprocessing" not in train_str
        assert "OpenCVPreprocessing" not in val_str

    def test_input_size_224(self):
        from application.preprocessing.PreProcessing import ImageProcessing
        ip = ImageProcessing(input_size=224, opencv=False)
        img = Image.new("RGB", (700, 500), color=(10, 200, 10))
        out = ip.train_transforms(img)
        assert out.shape == (3, 224, 224)
        out_val = ip.val_transforms(img)
        assert out_val.shape == (3, 224, 224)

    def test_input_size_default_512(self):
        from application.preprocessing.PreProcessing import ImageProcessing
        ip = ImageProcessing(opencv=False)
        img = Image.new("RGB", (700, 500), color=(10, 200, 10))
        out = ip.val_transforms(img)
        assert out.shape == (3, 512, 512)

    def test_random_crop_only_in_train(self):
        from application.preprocessing.PreProcessing import ImageProcessing
        ip = ImageProcessing(crop="random", opencv=False)
        assert "RandomResizedCrop" in str(ip.train_transforms)
        assert "RandomResizedCrop" not in str(ip.val_transforms)
        img = Image.new("RGB", (700, 500), color=(10, 200, 10))
        assert ip.train_transforms(img).shape == (3, 512, 512)

    def test_random_erasing_only_in_train(self):
        from application.preprocessing.PreProcessing import ImageProcessing
        ip = ImageProcessing(random_erasing=True, opencv=False)
        assert "RandomErasing" in str(ip.train_transforms)
        assert "RandomErasing" not in str(ip.val_transforms)


class TestLetterbox:
    def test_output_is_square(self):
        from application.preprocessing.PreProcessing import Letterbox
        lb = Letterbox(512)
        for w, h in [(3600, 2400), (2400, 3600), (4800, 3200), (512, 512)]:
            img = Image.new("RGB", (w, h), color=(10, 200, 10))
            assert lb(img).size == (512, 512)

    def test_aspect_preserved_no_crop(self):
        from application.preprocessing.PreProcessing import Letterbox
        import numpy as np
        lb = Letterbox(512, fill=0)
        img = Image.new("RGB", (960, 480), color=(255, 0, 0))  # 2:1
        out = np.array(lb(img))
        # lado LONGO vira 512: 960->512 (sem padding lateral), 480->256 (padding cima/baixo)
        assert out.shape == (512, 512, 3)
        # centro e vermelho; padding vertical e preto (fill=0)
        assert out[256, 256, 0] > 200
        assert out[0, 256].sum() == 0          # padding de cima
        assert out[511, 256].sum() == 0        # padding de baixo
        # altura preservada: conteudo ocupa 256 linhas no centro
        red_rows = (out[..., 0] > 200).all(axis=1)
        assert red_rows.sum() == 256

    def test_square_image_no_padding(self):
        from application.preprocessing.PreProcessing import Letterbox
        import numpy as np
        lb = Letterbox(512, fill=0)
        img = Image.new("RGB", (600, 600), color=(0, 0, 255))
        out = np.array(lb(img))
        # sem padding: todo o quadrado e azul
        assert (out[..., 2] > 200).all()


class TestOpenCVPreprocessingProperties:
    @given(st.integers(min_value=10, max_value=50))
    @settings(max_examples=5, deadline=None)  # denoise e CPU-bound; deadline de 200ms e flaky
    def test_denoise_does_not_crash(self, h):
        img = Image.new("RGB", (h * 10, h * 10), color=(200, 100, 50))
        result = OpenCVPreprocessing()(img)
        assert result.size == (h * 10, h * 10)
