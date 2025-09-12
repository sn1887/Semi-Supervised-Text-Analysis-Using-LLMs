import os
import argparse
import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load config ---
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)  # e.g. detectron2 configs YAML
    cfg.MODEL.WEIGHTS = args.model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.conf_thresh  # set threshold
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0]) if len(cfg.DATASETS.TEST) > 0 else MetadataCatalog.get("__unused")

    # --- Process images ---
    for fname in os.listdir(args.input_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        fpath = os.path.join(args.input_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            print(f"[WARN] Could not read {fpath}")
            continue

        outputs = predictor(img)

        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.2)
        vis = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        result = vis.get_image()[:, :, ::-1]

        out_path = os.path.join(args.output_dir, fname)
        cv2.imwrite(out_path, result)
        print(f"[INFO] Saved {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to .pkl model")
    parser.add_argument("--config-file", required=True, help="Path to config YAML")
    parser.add_argument("--input-dir", required=True, help="Folder with test images")
    parser.add_argument("--output-dir", default="inference_results", help="Folder to save visualized predictions")
    parser.add_argument("--conf-thresh", type=float, default=0.5, help="Confidence threshold for predictions")
    args = parser.parse_args()

    main(args)
