# convnext_dinov3_backbone.py
import os
from typing import Dict, List
import torch
import torch.nn as nn
import timm

from detectron2.layers import ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool


def _maybe_load_custom_weights(model: nn.Module, weight_path: str, strict: bool = False):
    """
    Load external weights (e.g., DINOv3-pretrained ConvNeXt).
    We try a few key-space fallbacks to make it robust to checkpoints with 'module.' or other prefixes.
    """
    if not weight_path:
        return

    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"MODEL.CONVNEXT.WEIGHTS not found: {weight_path}")

    ckpt = torch.load(weight_path, map_location="cpu")

    # Common patterns: a bare state_dict, or nested under 'state_dict' / 'model' / 'model_state'
    cand_keys = []
    if isinstance(ckpt, dict):
        cand_keys = [None, "state_dict", "model", "model_state", "model_state_dict"]
    else:
        cand_keys = [None]

    state_dict = None
    for k in cand_keys:
        if k is None and isinstance(ckpt, dict):
            # if it looks like a raw state dict (a lot of tensor leaves)
            if all(isinstance(v, torch.Tensor) for v in ckpt.values() if hasattr(ckpt, "values")):
                state_dict = ckpt
                break
        elif isinstance(ckpt, dict) and k in ckpt and isinstance(ckpt[k], dict):
            state_dict = ckpt[k]
            break

    if state_dict is None:
        # last resort, hope it's a raw Mapping-like thing
        state_dict = ckpt

    # Strip known prefixes (e.g., 'module.')
    def strip_prefix_if_present(sdict: Dict[str, torch.Tensor], prefix: str):
        if all(not k.startswith(prefix) for k in sdict.keys()):
            return sdict
        return {k[len(prefix):]: v for k, v in sdict.items() if k.startswith(prefix)}

    for pfx in ["module.", "backbone.", "model."]:
        state_dict = strip_prefix_if_present(state_dict, pfx)

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing or unexpected:
        print("[convnext_dinov3_backbone] load_state_dict non-strict results:")
        print("  missing:", missing[:10], "..." if len(missing) > 10 else "")
        print("  unexpected:", unexpected[:10], "..." if len(unexpected) > 10 else "")


class TimmConvNeXtBackbone(Backbone):
    """
    Bottom-up backbone that returns ConvNeXt stage maps via timm (features_only=True).
    Exposes four features: S1 (stride 4), S2 (stride 8), S3 (stride 16), S4 (stride 32).
    """

    def __init__(
        self,
        model_name: str = "convnext_base",
        pretrained: bool = False,
        out_indices: List[int] = (0, 1, 2, 3),
        weight_path: str = "",
        freeze_at: int = 0,  # 0..4 (0 = freeze nothing; 4 = freeze all stages)
        norm_eval: bool = True,
    ):
        """
        Args:
            model_name: timm model name, e.g., "convnext_tiny", "convnext_small", "convnext_base", "convnext_large"
            pretrained: if True, use timm's pretrained weights (e.g., ImageNet). You can also pass custom via weight_path.
            out_indices: stages to return. We default to all 4 stages.
            weight_path: path to an external checkpoint (e.g., DINOv3 ConvNeXt).
            freeze_at: how many early stages to freeze (0..4).
            norm_eval: set norm layers to eval() (keeps running stats fixed).
        """
        super().__init__()
        # Build timm feature extractor
        self.timm_net = timm.create_model(
            model_name,
            pretrained=pretrained and not bool(weight_path),
            features_only=True,
            out_indices=tuple(out_indices),
        )

        # Optionally load custom weights (e.g., DINOv3)
        if weight_path:
            _maybe_load_custom_weights(self.timm_net, weight_path, strict=False)

        # Channels and strides from timm's feature_info
        self._out_feature_channels = {}
        self._out_feature_strides = {}
        chs = self.timm_net.feature_info.channels()
        sts = self.timm_net.feature_info.reduction()  # reduction factor (i.e., stride)
        # timm order matches out_indices; we’ll name them S1..S4 regardless
        names = ["s1", "s2", "s3", "s4"]
        for i, (c, s) in enumerate(zip(chs, sts)):
            name = names[i]
            self._out_feature_channels[name] = c
            self._out_feature_strides[name] = s

        # Freeze early stages if requested
        # timm FeatureListNet exposes 'body' module; param names typically contain 'stages.X'
        if freeze_at > 0:
            frozen = 0
            for name, param in self.timm_net.named_parameters():
                # Rough heuristic: freeze stem + early stages by matching indices in param names
                if any(f"stages.{k}." in name for k in range(freeze_at)) or "stem." in name:
                    param.requires_grad = False
                    frozen += 1
            if frozen > 0:
                print(f"[convnext_dinov3_backbone] Froze {frozen} params up to stage {freeze_at-1}.")

        self.norm_eval = norm_eval
        if self.norm_eval:
            self._set_norm_eval(self.timm_net)

    def _set_norm_eval(self, m: nn.Module):
        for mod in m.modules():
            if isinstance(mod, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.LayerNorm)):
                mod.eval()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats: List[torch.Tensor] = self.timm_net(x)  # list of 4 tensors
        return {
            "s1": feats[0],  # stride ~4
            "s2": feats[1],  # stride ~8
            "s3": feats[2],  # stride ~16
            "s4": feats[3],  # stride ~32
        }

    def output_shape(self) -> Dict[str, ShapeSpec]:
        return {
            k: ShapeSpec(channels=v, stride=self._out_feature_strides[k])
            for k, v in self._out_feature_channels.items()
        }


@BACKBONE_REGISTRY.register()
def build_convnext_dinov3_backbone(cfg, input_shape: ShapeSpec):
    """
    Build the plain ConvNeXt bottom-up (no FPN). Suitable if your head expects C3/C4/C5 (adapt names).
    """
    name = cfg.MODEL.CONVNEXT.MODEL_NAME
    pretrained = cfg.MODEL.CONVNEXT.PRETRAINED
    weight_path = cfg.MODEL.CONVNEXT.WEIGHTS
    freeze_at = cfg.MODEL.CONVNEXT.FREEZE_AT
    norm_eval = cfg.MODEL.CONVNEXT.NORM_EVAL

    return TimmConvNeXtBackbone(
        model_name=name,
        pretrained=pretrained,
        out_indices=(0, 1, 2, 3),
        weight_path=weight_path,
        freeze_at=freeze_at,
        norm_eval=norm_eval,
    )


@BACKBONE_REGISTRY.register()
def build_convnext_dinov3_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Build ConvNeXt bottom-up + FPN that outputs P2..P5 (Detectron2 FPN default).
    """
    bottom_up = build_convnext_dinov3_backbone(cfg, input_shape)

    in_features = cfg.MODEL.FPN.IN_FEATURES
    assert len(in_features) > 0, "MODEL.FPN.IN_FEATURES must list ConvNeXt stage names (e.g., ['s1','s2','s3','s4'])."

    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    norm = cfg.MODEL.FPN.NORM
    fuse_type = cfg.MODEL.FPN.FUSE_TYPE

    fpn = FPN(
        bottom_up=bottom_up,
        in_features=in_features,       # e.g., ["s1","s2","s3","s4"]
        out_channels=out_channels,     # e.g., 256
        norm=norm,                     # "BN", "SyncBN", "GN", or ""
        top_block=LastLevelMaxPool(),  # produces P6 via maxpool on P5
        fuse_type=fuse_type,           # "sum" or "avg"
    )
    return fpn











# convnext_dinov3_config.py
from detectron2.config import CfgNode as CN

def add_convnext_config(cfg):
    _C = cfg

    _C.MODEL.CONVNEXT = CN()
    _C.MODEL.CONVNEXT.MODEL_NAME = "convnext_base"   # any timm ConvNeXt: convnext_tiny/small/base/large/xlarge, etc.
    _C.MODEL.CONVNEXT.PRETRAINED = False             # use timm’s pretrained if True (ignored if WEIGHTS is given)
    _C.MODEL.CONVNEXT.WEIGHTS = ""                   # path to external ckpt (e.g., DINOv3 ConvNeXt)
    _C.MODEL.CONVNEXT.FREEZE_AT = 0                  # 0..4 (freeze stem+first N stages)
    _C.MODEL.CONVNEXT.NORM_EVAL = True               # keep norm layers in eval mode (typical for pretrained)











"""
# Base Detectron2 config you like (you can also inherit with _BASE_)
MODEL:
  META_ARCHITECTURE: "GeneralizedRCNN"
  PIXEL_MEAN: [123.675, 116.28, 103.53]      # ImageNet means for ConvNeXt (timm default)
  PIXEL_STD:  [58.395, 57.12, 57.375]

  BACKBONE:
    NAME: "build_convnext_dinov3_fpn_backbone"

  CONVNEXT:
    MODEL_NAME: "convnext_base"
    PRETRAINED: False
    WEIGHTS: "/path/to/dinov3_convnext_base.pth"   # <<< put your DINOv3 weights here (optional)
    FREEZE_AT: 0
    NORM_EVAL: True

  FPN:
    IN_FEATURES: ["s1", "s2", "s3", "s4"]    # stages from the backbone
    OUT_CHANNELS: 256
    NORM: ""                                 # "", "GN", "BN", "SyncBN"
    FUSE_TYPE: "sum"

  ANCHOR_GENERATOR:
    SIZES: [[32], [64], [128], [256], [512]]
    ASPECT_RATIOS: [[0.5, 1.0, 2.0]]

  RPN:
    IN_FEATURES: ["p2", "p3", "p4", "p5", "p6"]
    PRE_NMS_TOPK_TRAIN: 2000
    PRE_NMS_TOPK_TEST: 1000
    POST_NMS_TOPK_TRAIN: 1000
    POST_NMS_TOPK_TEST: 1000
    BATCH_SIZE_PER_IMAGE: 256

  ROI_HEADS:
    NAME: "StandardROIHeads"
    NUM_CLASSES: 80                  # <-- change to your dataset
    IN_FEATURES: ["p2", "p3", "p4", "p5"]

  ROI_BOX_HEAD:
    NAME: "FastRCNNConvFCHead"
    NUM_FC: 2
    FC_DIM: 1024

  MASK_ON: True
  ROI_MASK_HEAD:
    NAME: "MaskRCNNConvUpsampleHead"
    NUM_CONV: 4

INPUT:
  FORMAT: "BGR"   # Detectron2 expects BGR by default; PIXEL_* above are in RGB, but D2 will handle conversion internally.

DATASETS:
  TRAIN: ("your_train",)
  TEST:  ("your_val",)

SOLVER:
  IMS_PER_BATCH: 16
  BASE_LR: 0.0005
  STEPS: [60000, 80000]
  MAX_ITER: 90000
  WARMUP_ITERS: 1000

TEST:
  EVAL_PERIOD: 5000

DATALOADER:
  NUM_WORKERS: 8


"""






