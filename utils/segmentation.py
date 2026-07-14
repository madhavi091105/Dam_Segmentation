import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import segmentation_models_pytorch as smp

CLASS_NAMES = ["background", "cracks", "spalling"]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 576
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = double_conv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        down = self.pool(skip)
        return down, skip

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=NUM_CLASSES):
        super().__init__()
        self.down1 = DownBlock(in_channels, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)
        self.bottleneck = double_conv(512, 1024)
        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        self.output = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1, s1 = self.down1(x)
        x2, s2 = self.down2(x1)
        x3, s3 = self.down3(x2)
        x4, s4 = self.down4(x3)
        x5 = self.bottleneck(x4)
        x = self.up1(x5, s4)
        x = self.up2(x, s3)
        x = self.up3(x, s2)
        x = self.up4(x, s1)
        return self.output(x)


_model_cache = {}

def load_segmenter(backend="segformer",
                    segformer_path="models/segformer_best.pth",
                    unet_path="models/best.pth"):
    key = backend
    if key in _model_cache:
        return _model_cache[key]

    if backend == "segformer":
        model = smp.Segformer(
            encoder_name="mit_b2",
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
        )
        state_dict = torch.load(segformer_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    elif backend == "unet":
        model = UNet(in_channels=3, num_classes=NUM_CLASSES)
        ckpt = torch.load(unet_path, map_location=DEVICE)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
    else:
        raise ValueError("backend must be 'segformer' or 'unet'")

    model.to(DEVICE).eval()
    _model_cache[key] = model
    return model


def preprocess(image: np.ndarray):
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img_norm = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor.to(DEVICE)


@torch.no_grad()
def run_segmentation(image: np.ndarray, backend="segformer", use_tta=True):
    model = load_segmenter(backend=backend)
    orig_h, orig_w = image.shape[:2]
    tensor = preprocess(image)

    if use_tta:
        probs = torch.softmax(model(tensor), dim=1)
        flip_h = torch.flip(tensor, dims=[3])
        probs += torch.flip(torch.softmax(model(flip_h), dim=1), dims=[3])
        flip_v = torch.flip(tensor, dims=[2])
        probs += torch.flip(torch.softmax(model(flip_v), dim=1), dims=[2])
        probs /= 3.0
    else:
        probs = torch.softmax(model(tensor), dim=1)

    pred = probs.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    palette = np.array([
        [0, 0, 0],
        [255, 0, 0],
        [255, 255, 0],
    ], dtype=np.uint8)
    color_mask = palette[pred]
    overlay = cv2.addWeighted(image, 0.6, color_mask, 0.4, 0)

    total_px = pred.size
    per_class_pct = {
        CLASS_NAMES[c]: round(float((pred == c).sum()) / total_px * 100, 2)
        for c in range(NUM_CLASSES)
    }

    return pred, overlay, per_class_pct