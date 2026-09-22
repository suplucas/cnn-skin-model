import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SetupModelViT:
    def setup_model(self, device, dropout_prob=0.5):
        vit = models.vit_b_16(pretrained=True)

        for param in vit.parameters():
            param.requires_grad = False

        # torchvision >= 0.13 envolt o head em Sequential; antes era Linear direto
        head = vit.heads
        num_features = head.in_features if isinstance(head, nn.Linear) else head[0].in_features
        vit.heads = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.SELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(128, 1)
        )

        # torchvision >= 0.19 removeu o resize interno do ViT:
        # _process_input agora ASSERT input == image_size (224x224).
        # Redimensiona bilinearmente antes do forward original.
        orig_forward = type(vit).forward
        image_size = vit.image_size

        def forward(x):
            if tuple(x.shape[-2:]) != (image_size, image_size):
                x = F.interpolate(
                    x, size=(image_size, image_size),
                    mode="bilinear", align_corners=False,
                )
            return orig_forward(vit, x)

        vit.forward = forward

        model = vit.to(device)

        return model
