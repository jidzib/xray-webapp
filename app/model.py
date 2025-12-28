import torch
from app.multimodal import MultimodalCheXpertModel

DEVICE = "cpu"

MODEL_CONFIGS = [
    ("weights/best_model_efficientnetv2_s.pth", "efficientnet_v2_s"),
    ("weights/best_model_densenet121.pth", "densenet121"),
]

def load_models():
    models = []
    for path, backbone in MODEL_CONFIGS:
        m = MultimodalCheXpertModel(
            num_tabular_features=4,
            num_classes=14,
            backbone=backbone,
            pretrained=False
        )
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        models.append(m)
    return models


@torch.no_grad()
def predict(models, image, tabular):
    probs = []
    for m in models:
        logits = m(image, tabular)
        probs.append(torch.sigmoid(logits))
    return torch.mean(torch.stack(probs), dim=0).squeeze(0).cpu().numpy()
