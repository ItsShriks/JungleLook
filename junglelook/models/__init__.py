from .pointnet import PointNetSeg, feature_transform_regularizer
from .pointnet2 import PointNet2Seg

__all__ = ["PointNetSeg", "PointNet2Seg", "feature_transform_regularizer", "build_model"]

MODELS = {"pointnet": PointNetSeg, "pointnet2": PointNet2Seg}


def build_model(name: str, in_channels: int, num_classes: int, **kwargs):
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](in_channels=in_channels, num_classes=num_classes, **kwargs)
