# New imports
import os
import random
import torch

import mlflow
from detectron2.engine import hooks
from detectron2.utils.events import get_event_storage
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import detection_utils as dutils
import detectron2.data.transforms as T
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.utils import comm

# Add CLI flags
p.add_argument("--mlflow-uri", type=str, default=None,
               help="file:///abs/path/to/outputs/mlruns or http://localhost:5000")
p.add_argument("--mlflow-experiment", type=str, default="detectron2")
p.add_argument("--mlflow-run-name", type=str, default="mask_rcnn_r50_fpn_3x")

# New CLI flags for visualization
p.add_argument("--vis-period", type=int, default=1000,
               help="Visualize predictions every N steps (0 disables).")
p.add_argument("--vis-num-images", type=int, default=4,
               help="How many images to visualize at each interval.")
p.add_argument("--vis-dataset", type=str, default="",
               help="Dataset name to visualize (default: first TEST else TRAIN).")

# In main(), after cfg is finalized:
# 0) Configure MLflow local tracking
if args.mlflow_uri:
    mlflow.set_tracking_uri(args.mlflow_uri)
else:
    # default: local file store under project outputs/mlruns
    local_uri = "file://" + os.path.abspath(os.path.join(args.output, "..", "mlruns"))
    mlflow.set_tracking_uri(local_uri)

mlflow.set_experiment(args.mlflow_experiment)

# 1) A small MLflow hook for Detectron2 + periodic visualization
class MLflowHook(hooks.HookBase):
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.run = None

        # Visualization settings
        self.vis_period = int(getattr(args, "vis_period", 0) or 0)
        self.vis_num_images = int(getattr(args, "vis_num_images", 0) or 0)
        self.vis_dataset = getattr(args, "vis_dataset", "").strip() or None
        self.vis_samples = []
        self.vis_metadata = None

        # Preprocessing aligned with DefaultPredictor
        self.input_format = cfg.INPUT.FORMAT  # "BGR" by default
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
            cfg.INPUT.MAX_SIZE_TEST
        )

    def _prepare_vis_samples(self):
        # Pick dataset: prefer TEST, else TRAIN, unless overridden
        if self.vis_dataset:
            dataset_name = self.vis_dataset
        else:
            dataset_name = self.cfg.DATASETS.TEST if len(self.cfg.DATASETS.TEST) else self.cfg.DATASETS.TRAIN

        # Cache dataset and metadata
        dataset = DatasetCatalog.get(dataset_name)
        self.vis_metadata = MetadataCatalog.get(dataset_name)

        # Deterministic selection of N samples
        n = min(self.vis_num_images, len(dataset))
        if n <= 0:
            self.vis_samples = []
            return

        # Evenly spaced indices for stability
        step = max(1, len(dataset) // n)
        idxs = list(range(0, len(dataset), step))[:n]
        self.vis_samples = [dataset[i] for i in idxs]

    def before_train(self):
        self.run = mlflow.start_run(run_name=self.args.mlflow_run_name)
        # Log high-level params
        mlflow.log_params({
            "model": "mask_rcnn_R_50_FPN_3x",
            "ims_per_batch": self.cfg.SOLVER.IMS_PER_BATCH,
            "base_lr": self.cfg.SOLVER.BASE_LR,
            "max_iter": self.cfg.SOLVER.MAX_ITER,
            "num_workers": self.cfg.DATALOADER.NUM_WORKERS,
            "mask_on": self.cfg.MODEL.MASK_ON,
            "num_classes": self.cfg.MODEL.ROI_HEADS.NUM_CLASSES
        })
        # Save/track the full config
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        cfg_path = os.path.join(self.cfg.OUTPUT_DIR, "config.yaml")
        with open(cfg_path, "w") as f:
            f.write(self.cfg.dump())
        mlflow.log_artifact(cfg_path)

        # Prepare visualization samples once
        if self.vis_period > 0 and self.vis_num_images > 0:
            try:
                self._prepare_vis_samples()
            except Exception:
                self.vis_samples = []

    @torch.no_grad()
    def _log_visualizations(self, step: int):
        # Only the main process should log artifacts
        if not comm.is_main_process():
            return
        if not self.vis_samples or self.vis_num_images <= 0:
            return

        model = self.trainer.model
        was_training = model.training
        model.eval()

        for i, d in enumerate(self.vis_samples[:self.vis_num_images]):
            # Read original image
            im = dutils.read_image(d["file_name"], format=self.input_format)
            height, width = im.shape[:2]

            # Apply same preprocessing as DefaultPredictor
            im_aug = self.aug.get_transform(im).apply_image(im)
            tensor = torch.as_tensor(im_aug.astype("float32").transpose(2, 0, 1)).to(model.device)

            # Prepare inputs and run model
            outputs = model([{"image": tensor, "height": height, "width": width}])
            instances = outputs["instances"].to("cpu")

            # Visualize predictions on the resized image
            rgb = im_aug[:, :, ::-1] if self.input_format == "BGR" else im_aug
            v = Visualizer(rgb, metadata=self.vis_metadata, scale=0.8, instance_mode=ColorMode.IMAGE_BW)
            vis = v.draw_instance_predictions(instances).get_image()

            # Try logging directly as an image; fallback to artifact file if needed
            try:
                mlflow.log_image(vis, f"predictions/step_{step:07d}/img_{i:02d}.png")
            except Exception:
                save_dir = os.path.join(self.cfg.OUTPUT_DIR, "predictions", f"step_{step:07d}")
                os.makedirs(save_dir, exist_ok=True)
                out_path = os.path.join(save_dir, f"img_{i:02d}.png")
                from PIL import Image
                Image.fromarray(vis).save(out_path)
                mlflow.log_artifact(out_path, artifact_path=f"predictions/step_{step:07d}")

        if was_training:
            model.train()

    def after_step(self):
        # Pull the latest training scalars from EventStorage
        storage = get_event_storage()
        metrics = {}
        try:
            for k, hist in storage.histories().items():
                # log select scalars to avoid excessive volume
                if any(s in k for s in ["loss", "lr", "time", "data_time"]):
                    try:
                        # prefer latest un-smoothed value where available
                        val = hist.latest()
                    except Exception:
                        # fallback to median over recent window, if needed
                        val = getattr(hist, "median", lambda *a, **kw: None)(20)
                    if val is not None:
                        metrics[k] = float(val)
        except Exception:
            pass

        if metrics:
            mlflow.log_metrics(metrics, step=storage.iter)

        # Periodic visualization
        if self.vis_period > 0 and (storage.iter % self.vis_period == 0) and storage.iter > 0:
            try:
                self._log_visualizations(storage.iter)
            except Exception:
                pass

    def after_train(self):
        # Optionally evaluate and log final metrics
        try:
            results = self.trainer.test(self.cfg, self.trainer.model)
            # results is a list of dicts (one per dataset); flatten
            if isinstance(results, list):
                for d in results:
                    for k, v in d.items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(k, float(v))
        except Exception:
            pass

        # Log useful artifacts from OUTPUT_DIR
        out_dir = self.cfg.OUTPUT_DIR
        for fname in ["metrics.json", "last_checkpoint"]:
            fpath = os.path.join(out_dir, fname)
            if os.path.exists(fpath):
                mlflow.log_artifact(fpath)
        # Log all checkpoints directory (optional; can be large)
        ckp_dir = os.path.join(out_dir)
        # mlflow.log_artifacts(ckp_dir)  # uncomment if you want everything

        # Optionally log the model object (Detectron2 is a PyTorch nn.Module)
        try:
            import mlflow.pytorch as mlflow_pt
            mlflow_pt.log_model(self.trainer.model, artifact_path="model")
        except Exception:
            pass

        mlflow.end_run()

# Build trainer and register both your mapper and MLflow hook
trainer = Trainer(cfg)
trainer.register_hooks([MLflowHook(cfg, args)])
trainer.resume_or_load(resume=args.resume)
trainer.train()
