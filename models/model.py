import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models.resnet import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator

def _backbone_p2_p5(imagenet_weights=False):
    weights = "IMAGENET1K_V1" if imagenet_weights else None
    backbone = resnet50(weights=weights, norm_layer=FrozenBatchNorm2d)
    # Return C2...C5 as "0","1","2","3"
    body = IntermediateLayerGetter(backbone, return_layers={
        "layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"
    })
    in_channels_list = [256, 512, 1024, 2048]
    fpn = FeaturePyramidNetwork(in_channels_list, out_channels=256)

    class BackboneWithFPN(nn.Module):
        def __init__(self, body, fpn):
            super().__init__()
            self.body = body
            self.fpn = fpn
            self.out_channels = 256

        def forward(self, x):
            feats = self.body(x)  # keys: "0","1","2","3"
            feats = self.fpn(feats)  # same 4 keys, no extras
            # Ensure deterministic order
            return OrderedDict((k, feats[k]) for k in ["0", "1", "2", "3"])

    return BackboneWithFPN(body, fpn)

def build_model(num_classes, anchor_sizes, aspect_ratios,
                score_thresh=0.05, nms_thresh=0.45, detections_per_img=1000,
                imagenet_weights=False):
    assert len(anchor_sizes) == 4, "expect 4 size tuples for P2–P5"
    assert len(aspect_ratios) == 4, "expect 4 ratio tuples for P2–P5"

    backbone = _backbone_p2_p5(imagenet_weights=imagenet_weights)

    anchor_gen = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    model = RetinaNet(
        backbone=backbone,
        num_classes=num_classes,
        anchor_generator=anchor_gen,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
        detections_per_img=detections_per_img,
    )
    return model
