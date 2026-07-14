import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
import numpy as np
import cv2

CLASS_NAMES = ["Undamaged", "Damaged"]   # confirm order apne training se
IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DamClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        base = efficientnet_b0(weights=None)
        self.backbone = base
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


_model_cache = {}

def load_classifier(weights_path="models/efficientnetB0.pt"):
    if weights_path in _model_cache:
        return _model_cache[weights_path]

    model = DamClassifier(num_classes=len(CLASS_NAMES))
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)

    model.to(DEVICE).eval()
    _model_cache[weights_path] = model
    return model


def preprocess(image: np.ndarray):
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img_norm = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor.to(DEVICE)


@torch.no_grad()
def run_classification(image: np.ndarray, weights_path="models/best_model.pth"):
    model = load_classifier(weights_path)
    tensor = preprocess(image)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_idx = int(probs.argmax())
    return {
        "label": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_probs": {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)}
    }