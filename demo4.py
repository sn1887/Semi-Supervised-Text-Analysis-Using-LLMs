import os
import cv2
import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.utils.visualizer import Visualizer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

# Import your dataset register function and mapper
# from my_dataset import register_my_dataset, MyDatasetMapper

def run_inference(
    config_file: str,
    model_weights: str,
    dataset_name: str,
    output_folder: str,
    mapper=None,
    score_thresh: float = 0.5,
    num_images: int = -1,  # -1 means all
):
    """
    Run inference on a registered dataset using custom mapper and save visualizations.

    Args:
        config_file: Path to config .yaml file used for training.
        model_weights: Path to .pth checkpoint.
        dataset_name: Registered dataset name for test set.
        output_folder: Folder to save visualized predictions.
        mapper: Custom dataset mapper (callable or class).
        score_thresh: Confidence threshold for predictions.
        num_images: Limit number of images processed (-1 = all).
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load config
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(model_weights)

    # Metadata
    metadata = MetadataCatalog.get(dataset_name)

    # Build dataloader with custom mapper
    dataloader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    # Run inference
    for i, batch in enumerate(dataloader):
        if num_images > 0 and i >= num_images:
            break

        with torch.no_grad():
            outputs = model(batch)

        for sample, output in zip(batch, outputs):
            img = sample["image"].permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.2)
            v = v.draw_instance_predictions(output["instances"].to("cpu"))
            vis_img = v.get_image()[:, :, ::-1]

            # Use original filename if available
            file_name = os.path.basename(sample.get("file_name", f"image_{i}.jpg"))
            out_path = os.path.join(output_folder, file_name)
            cv2.imwrite(out_path, vis_img)

            print(f"Saved: {out_path}")


if __name__ == "__main__":
    # Example usage:
    # Make sure your dataset is registered before this call
    # register_my_dataset()

    run_inference(
        config_file="configs/custom_config.yaml",
        model_weights="output/model_final.pth",
        dataset_name="my_dataset_test",   # must be registered
        output_folder="outputs/test_predictions",
        mapper=None,  # or pass your custom mapper: MyDatasetMapper(cfg, is_train=False)
        score_thresh=0.5,
        num_images=-1,  # all images
    )
