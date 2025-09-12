#!/usr/bin/env python3
import os
import argparse
import glob
import cv2
import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.utils.visualizer import Visualizer
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

# Project imports (match your training script)
from src.data.yolo_register import register_yolo
try:
    from src.data.mappers import YoloLikeMapper
    HAS_CUSTOM_MAPPER = True
except Exception:
    HAS_CUSTOM_MAPPER = False


def register_image_folder_no_labels(name: str, image_dir: str):
    """
    Minimal registration for an image folder without annotations.
    Produces dataset dicts with only file_name / height / width.
    """
    def _loader():
        entries = []
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
        paths.sort()
        for i, p in enumerate(paths):
            img = cv2.imread(p)
            if img is None:
                continue
            h, w = img.shape[:2]
            entries.append({
                "file_name": p,
                "height": h,
                "width": w,
                "image_id": i,
            })
        return entries

    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
    DatasetCatalog.register(name, _loader)
    if name in MetadataCatalog.list():
        MetadataCatalog.remove(name)
    MetadataCatalog.get(name)  # create empty metadata


def build_cfg(
    output_dir: str,
    base_lr: float,
    score_thresh: float,
    mask_on: bool,
    ims_per_batch: int,
    num_workers: int,
    custom_mapper_flag: bool,
    num_classes: int = None,
    config_file: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.MASK_ON = bool(mask_on)
    if num_classes is not None:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(num_classes)
    cfg.INPUT.CUSTOM_MAPPER = bool(custom_mapper_flag)
    cfg.OUTPUT_DIR = output_dir
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def to_rgb_uint8(img_chw: torch.Tensor, input_format: str) -> np.ndarray:
    """
    Convert CHW tensor (float or uint8) in Detectron2 sample to RGB uint8 HWC.
    """
    arr = img_chw.permute(1, 2, 0).cpu().numpy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    # Detectron2 default INPUT.FORMAT is "BGR"
    if input_format.upper() == "BGR":
        arr = arr[:, :, ::-1]
    return arr


def main():
    ap = argparse.ArgumentParser("Run inference and save visualized predictions")
    # Weights / config
    ap.add_argument("--weights", type=str, required=True, help="Path to .pth checkpoint")
    ap.add_argument("--config-file", type=str,
                    default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                    help="Model zoo config used during training")
    ap.add_argument("--output", type=str, default="./outputs/test_predictions",
                    help="Folder to save visualized predictions")
    ap.add_argument("--score-thresh-test", type=float, default=0.5)
    ap.add_argument("--mask-on", action="store_true", default=False)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2)  # affects dataloader, not model speed
    ap.add_argument("--base-lr", type=float, default=0.00025)  # irrelevant for inference, kept for parity
    ap.add_argument("--use-custom-mapper", action="store_true", default=False)

    # Option A: use an already-registered dataset (e.g., your val set)
    ap.add_argument("--dataset-name", type=str, default=None,
                    help="Name of an already-registered dataset to run on")

    # Option B: register a YOLO-style test set now (like training)
    ap.add_argument("--test-name", type=str, default=None)
    ap.add_argument("--test-images", type=str, default=None)
    ap.add_argument("--test-labels", type=str, default=None,
                    help="Optional: YOLO txt labels for test set (can be omitted)")
    ap.add_argument("--classes", type=str, nargs="*", default=None,
                    help="Class names list (required if registering a YOLO/test or folder dataset)")

    ap.add_argument("--normalized", action="store_true", default=True,
                    help="YOLO labels are normalized [0,1]")

    # Option C: raw folder without labels (auto-registers a minimal dataset)
    ap.add_argument("--input-folder", type=str, default=None,
                    help="Folder of images (no labels) — auto-registers a dataset")

    # Misc
    ap.add_argument("--limit", type=int, default=-1, help="Limit number of images processed (-1 for all)")
    ap.add_argument("--visualizer-scale", type=float, default=1.0)

    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ----- Decide dataset path -----
    dataset_name = args.dataset_name

    if dataset_name is None and args.test_images:
        # Register YOLO-style test set (labels optional)
        if not args.classes:
            raise SystemExit("--classes is required when registering a YOLO-style test set.")
        test_name = args.test_name or "my_yolo_test"
        register_yolo(
            name=test_name,
            image_dir=args.test_images,
            label_dir=args.test_labels,     # can be None
            class_names=args.classes,
            assume_normalized=args.normalized,
        )
        dataset_name = test_name

    if dataset_name is None and args.input_folder:
        # Folder-only mode (no labels)
        if not args.classes:
            print("[WARN] --classes not provided; predictions will have numeric class IDs only.")
        folder_name = args.test_name or "image_folder_test"
        register_image_folder_no_labels(folder_name, args.input_folder)
        # If classes are given, attach to metadata for nicer labels
        if args.classes:
            MetadataCatalog.get(folder_name).thing_classes = list(args.classes)
        dataset_name = folder_name

    if dataset_name is None:
        raise SystemExit(
            "No dataset specified. Use one of:\n"
            "  --dataset-name EXISTING_DATASET\n"
            "  --test-images ... [--test-labels ...] --classes ... [--test-name ...]\n"
            "  --input-folder ... [--classes ...]"
        )

    # ----- Build cfg (mirror training knobs that matter for inference) -----
    # Determine num_classes from metadata (preferred) or from --classes
    meta = MetadataCatalog.get(dataset_name)
    if getattr(meta, "thing_classes", None):
        num_classes = len(meta.thing_classes)
    else:
        num_classes = len(args.classes) if args.classes else None

    cfg = build_cfg(
        output_dir=args.output,
        base_lr=args.base_lr,
        score_thresh=args.score_thresh_test,
        mask_on=args.mask_on,
        ims_per_batch=args.batch_size,
        num_workers=args.num_workers,
        custom_mapper_flag=args.use_custom_mapper,
        num_classes=num_classes,
        config_file=args.config_file,
    )

    # Attach dataset to cfg.TEST for loader construction
    cfg.DATASETS.TEST = (dataset_name,)

    # ----- Build model & load weights -----
    model = build_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()
    DetectionCheckpointer(model).load(args.weights)

    # ----- Build test loader with (optional) custom mapper -----
    mapper = None
    if cfg.INPUT.CUSTOM_MAPPER and HAS_CUSTOM_MAPPER:
        # NOTE: custom mapper may expect annotations. If your test set has no labels,
        # consider running without --use-custom-mapper.
        mapper = YoloLikeMapper(cfg, is_train=False)

    loader = build_detection_test_loader(cfg, dataset_name=dataset_name, mapper=mapper)

    # ----- Run inference & save visualizations -----
    processed = 0
    input_format = getattr(cfg.INPUT, "FORMAT", "BGR")

    for batch in loader:
        with torch.no_grad():
            outputs = model(batch)

        for sample, output in zip(batch, outputs):
            if args.limit > 0 and processed >= args.limit:
                break

            # Visualize on the (possibly transformed) input tensor to align coords
            img_rgb = to_rgb_uint8(sample["image"], input_format)
            vis = Visualizer(img_rgb, metadata=meta, scale=args.visualizer_scale)
            vis = vis.draw_instance_predictions(output["instances"].to("cpu"))
            vis_img = vis.get_image()[:, :, ::-1]  # RGB -> BGR for cv2.imwrite

            # Derive filename
            fname = os.path.basename(sample.get("file_name", f"img_{processed}.jpg"))
            out_path = os.path.join(args.output, fname)
            cv2.imwrite(out_path, vis_img)
            print(f"Saved: {out_path}")

            processed += 1

        if args.limit > 0 and processed >= args.limit:
            break

    print(f"Done. Saved {processed} visualizations to: {args.output}")


if __name__ == "__main__":
    main()
python infer_and_save.py \
  --weights outputs/mask_rcnn_r50_fpn_3x/model_final.pth \
  --test-images dataset/test/images \
  --test-labels dataset/test/labels \
  --classes person car bike \
  --test-name my_yolo_test \
  --output outputs/test_predictions
