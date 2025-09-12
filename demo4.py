import os
import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def run_inference(
    config_file: str,
    model_weights: str,
    input_folder: str,
    output_folder: str,
    dataset_name: str = None,
    score_thresh: float = 0.5,
):
    """
    Run inference on all images in `input_folder` and save visualizations to `output_folder`.

    Args:
        config_file: Path to Detectron2 config .yaml file (same one used for training).
        model_weights: Path to trained .pth checkpoint.
        input_folder: Folder containing test images.
        output_folder: Folder where predictions will be saved.
        dataset_name: Dataset metadata name for visualization (optional).
        score_thresh: Confidence threshold for predictions.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load config
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = DefaultPredictor(cfg)

    # Metadata for coloring / labels
    metadata = None
    if dataset_name and dataset_name in MetadataCatalog.list():
        metadata = MetadataCatalog.get(dataset_name)

    # Loop over all images in the folder
    for fname in os.listdir(input_folder):
        fpath = os.path.join(input_folder, fname)
        if not os.path.isfile(fpath):
            continue

        img = cv2.imread(fpath)
        if img is None:
            continue

        outputs = predictor(img)

        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        vis_img = v.get_image()[:, :, ::-1]

        out_path = os.path.join(output_folder, fname)
        cv2.imwrite(out_path, vis_img)

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    # Example usage
    run_inference(
        config_file="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        model_weights="output/model_final.pth",
        input_folder="datasets/test_images",
        output_folder="outputs/test_predictions",
        dataset_name="coco_2017_val",  # or your custom dataset name
        score_thresh=0.5,
    )
