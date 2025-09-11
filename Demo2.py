# file: src/engine/hooks_periodic_vis.py
import os, copy, cv2, torch
import mlflow
from detectron2.engine.hooks import HookBase
from detectron2.data import build_detection_test_loader
from detectron2.data import detection_utils as utils
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

class PeriodicPredictionVisualizer(HookBase):
    def __init__(
        self,
        cfg,
        dataset_name,           # e.g., your val set name
        period=2000,            # draw every N training iterations
        num_images=4,           # how many images to render each time
        save_subdir="pred_images",
        score_thresh=0.5,
        log_to_mlflow=True,
    ):
        self.cfg = copy.deepcopy(cfg)
        self.dataset_name = dataset_name
        self.period = int(period)
        self.num_images = int(num_images)
        self.save_dir = os.path.join(cfg.OUTPUT_DIR, save_subdir)
        self.score_thresh = float(score_thresh)
        self.log_to_mlflow = bool(log_to_mlflow)

        os.makedirs(self.save_dir, exist_ok=True)

    def before_train(self):
        # Build a test loader once; will be iterated and recreated on exhaustion
        self._refresh_loader()
        self.metadata = MetadataCatalog.get(self.dataset_name)
        # cache format for reading RGB/BGR correctly
        self.image_format = self.cfg.INPUT.FORMAT

    def _refresh_loader(self):
        self._loader_iter = iter(build_detection_test_loader(self.cfg, dataset_name=self.dataset_name))

    def after_step(self):
        it = self.trainer.iter + 1
        if self.period <= 0 or it % self.period != 0:
            return

        model = self.trainer.model
        was_training = model.training
        model.eval()

        saved_paths = []
        with torch.no_grad():
            samples = []
            # Collect num_images samples from the loader (recreate iterator if needed)
            for _ in range(self.num_images):
                try:
                    batch = next(self._loader_iter)   # list[dict], default batch size = 1 for test loader
                except StopIteration:
                    self._refresh_loader()
                    batch = next(self._loader_iter)
                samples.append(batch)

            # Forward pass: model expects list of dataset dicts with "image" tensor
            outputs = model(samples)

            for idx, (inp, out) in enumerate(zip(samples, outputs)):
                # Read original image for visualization
                img = utils.read_image(inp["file_name"], format=self.image_format)
                instances = out["instances"].to("cpu")
                if instances.has("scores"):
                    keep = instances.scores >= self.score_thresh
                    instances = instances[keep]

                vis = Visualizer(img[:, :, ::-1], metadata=self.metadata, scale=1.0)
                vis = vis.draw_instance_predictions(instances)
                vis_img = vis.get_image()[:, :, ::-1]  # back to BGR for cv2.imwrite

                base = os.path.basename(inp["file_name"])
                out_path = os.path.join(self.save_dir, f"iter{it:07d}_{idx}_{base}")
                cv2.imwrite(out_path, vis_img)
                saved_paths.append(out_path)

                if self.log_to_mlflow:
                    # Option A: log as an in-memory image (artifact path keeps a folder structure)
                    # mlflow.log_image expects a numpy array (RGB) or PIL.Image
                    # Convert to RGB for nicer display in UIs
                    mlflow.log_image(vis_img[:, :, ::-1], artifact_file=f"pred_images/iter{it:07d}_{idx}_{base}")
                    # Option B (alternative): mlflow.log_artifact(out_path) to upload saved file

        if was_training:
            model.train()
