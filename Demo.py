#!/usr/bin/env python3
import os
import argparse
import random
import numpy as np
import torch

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.data import (
    build_detection_train_loader,
    build_detection_test_loader,
    DatasetCatalog,
    MetadataCatalog,
)
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger

# Your project imports (from previous code)
# Assumes these files exist as discussed earlier:
# src/data/yolo_register.py -> register_yolo(...)
# src/data/mappers.py -> YoloLikeMapper (optional)
from src.data.yolo_register import register_yolo
try:
    from src.data.mappers import YoloLikeMapper
    HAS_CUSTOM_MAPPER = True
except Exception:
    HAS_CUSTOM_MAPPER = False

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.INPUT.CUSTOM_MAPPER and HAS_CUSTOM_MAPPER:
            mapper = YoloLikeMapper(cfg, is_train=True)
            return build_detection_train_loader(cfg, mapper=mapper)
        return build_detection_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if cfg.INPUT.CUSTOM_MAPPER and HAS_CUSTOM_MAPPER:
            mapper = YoloLikeMapper(cfg, is_train=False)
            return build_detection_test_loader(cfg, dataset_name=dataset_name, mapper=mapper)
        return build_detection_test_loader(cfg, dataset_name=dataset_name)

def parse_args():
    p = argparse.ArgumentParser("Train Mask R-CNN R50-FPN-3x on YOLO-style data")
    # Dataset registration inputs
    p.add_argument("--train-images", type=str, default="dataset/Train/images")
    p.add_argument("--train-labels", type=str, default="dataset/Train/labels")
    p.add_argument("--val-images", type=str, default="dataset/val/images")
    p.add_argument("--val-labels", type=str, default="dataset/val/labels")
    p.add_argument("--classes", type=str, nargs="+", required=True,
                   help="List of class names in order matching YOLO class ids (zero-based)")

    p.add_argument("--train-name", type=str, default="my_yolo_train")
    p.add_argument("--val-name", type=str, default="my_yolo_val")
    p.add_argument("--normalized", action="store_true", default=True,
                   help="Labels are normalized [0,1] in YOLO txt (default True).")
    p.add_argument("--use-custom-mapper", action="store_true", default=False,
                   help="Use YoloLikeMapper from src/data/mappers.py if available.")

    # Training hyperparameters / config
    p.add_argument("--output", type=str, default="./outputs/mask_rcnn_r50_fpn_3x")
    p.add_argument("--batch-size", type=int, default=2, help="IMS_PER_BATCH")
    p.add_argument("--base-lr", type=float, default=0.00025)
    p.add_argument("--max-iter", type=int, default=90000)
    p.add_argument("--eval-period", type=int, default=5000)
    p.add_argument("--ckpt-period", type=int, default=5000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--score-thresh-test", type=float, default=0.5)
    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask-on", action="store_true", default=False,
                   help="Enable mask head; set True only if dataset provides polygon/bitmask GT.")
    return p.parse_args()

def main():
    args = parse_args()
    setup_logger()

    # 1) Register datasets (train/val) from YOLO txts
    #    This uses your earlier 'register_yolo' to produce Detectron2 dicts.
    os.makedirs(args.output, exist_ok=True)
    register_yolo(
        name=args.train_name,
        image_dir=args.train_images,
        label_dir=args.train_labels,
        class_names=args.classes,
        assume_normalized=args.normalized,
    )
    register_yolo(
        name=args.val_name,
        image_dir=args.val_images,
        label_dir=args.val_labels,
        class_names=args.classes,
        assume_normalized=args.normalized,
    )

    # 2) Build config from model zoo Mask R-CNN R50-FPN-3x
    cfg = get_cfg()
    # Merge the builtin model zoo config
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    # Override dataset names and basic training knobs
    cfg.DATASETS.TRAIN = (args.train_name,)
    cfg.DATASETS.TEST = (args.val_name,)
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = []  # adjust if using step LR schedule
    cfg.TEST.EVAL_PERIOD = args.eval_period
    cfg.SOLVER.CHECKPOINT_PERIOD = args.ckpt_period

    # Set number of classes from metadata
    num_classes = len(MetadataCatalog.get(args.train_name).thing_classes)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    # Pretrained weights from model zoo (R-50 ImageNet backbone → COCO Mask R-CNN)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )

    # Enable/disable mask branch depending on your labels
    cfg.MODEL.MASK_ON = bool(args.mask_on)

    # Score threshold for inference
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh_test

    # Custom mapper flag exposed in cfg to let Trainer read it
    cfg.INPUT.CUSTOM_MAPPER = bool(args.use_custom_mapper)

    # Output dir and seed
    cfg.OUTPUT_DIR = args.output
    seed_all(args.seed)

    # 3) Train
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()

if __name__ == "__main__":
    main()
