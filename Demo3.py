import os
import io
import random
import mlflow
import torch
from PIL import Image

from detectron2.engine import hooks
from detectron2.utils.events import get_event_storage
from detectron2.data import build_detection_test_loader, MetadataCatalog
from detectron2.utils.visualizer import Visualizer


class MLflowHook(hooks.HookBase):
    def __init__(self, cfg, vis_period=5000, num_images=5):
        """
        Args:
            cfg: detectron2 config
            vis_period: how often (in training iterations) to log test visualizations
            num_images: number of random test images to visualize each time
        """
        self.cfg = cfg
        self.run = None
        self.vis_period = vis_period
        self.num_images = num_images
        self.test_loader = None

    def before_train(self):
        self.run = mlflow.start_run(run_name=args.mlflow_run_name)
        # Log params
        mlflow.log_params({
            "model": "mask_rcnn_R_50_FPN_3x",
            "ims_per_batch": self.cfg.SOLVER.IMS_PER_BATCH,
            "base_lr": self.cfg.SOLVER.BASE_LR,
            "max_iter": self.cfg.SOLVER.MAX_ITER,
            "num_workers": self.cfg.DATALOADER.NUM_WORKERS,
            "mask_on": self.cfg.MODEL.MASK_ON,
            "num_classes": self.cfg.MODEL.ROI_HEADS.NUM_CLASSES
        })
        # Save config
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        cfg_path = os.path.join(self.cfg.OUTPUT_DIR, "config.yaml")
        with open(cfg_path, "w") as f:
            f.write(self.cfg.dump())
        mlflow.log_artifact(cfg_path)

        # Build test loader (first dataset in cfg.DATASETS.TEST)
        if len(self.cfg.DATASETS.TEST):
            self.test_dataset = self.cfg.DATASETS.TEST[0]
            self.test_loader = list(build_detection_test_loader(
                self.cfg, self.test_dataset, mapper=None
            ))

    def after_step(self):
        # Log scalars
        storage = get_event_storage()
        metrics = {}
        try:
            for k, hist in storage.histories().items():
                if any(s in k for s in ["loss", "lr", "time", "data_time"]):
                    try:
                        val = hist.latest()
                    except Exception:
                        val = getattr(hist, "median", lambda *a, **kw: None)(20)
                    if val is not None:
                        metrics[k] = float(val)
        except Exception:
            pass
        if metrics:
            mlflow.log_metrics(metrics, step=storage.iter)

        # Log visualizations periodically
        if self.test_loader and self.vis_period > 0:
            if storage.iter % self.vis_period == 0:
                self._log_test_visualizations(step=storage.iter)

    @torch.no_grad()
    def _log_test_visualizations(self, step=0):
        self.trainer.model.eval()
        metadata = MetadataCatalog.get(self.test_dataset)

        # Pick random samples from test loader
        samples = random.sample(self.test_loader,
                                min(self.num_images, len(self.test_loader)))

        for i, batch in enumerate(samples):
            outputs = self.trainer.model(batch)

            for sample, output in zip(batch, outputs):
                img = sample["image"].permute(1, 2, 0).cpu().numpy()
                img = img[:, :, ::-1]  # RGB -> BGR if needed

                vis = Visualizer(img, metadata=metadata, scale=1.2)
                vis = vis.draw_instance_predictions(output["instances"].to("cpu"))
                vis_img = Image.fromarray(vis.get_image())

                mlflow.log_image(vis_img, f"predictions/step_{step}_img_{i}.png")

        self.trainer.model.train()

    def after_train(self):
        # Evaluate & log final metrics
        try:
            results = self.trainer.test(self.cfg, self.trainer.model)
            if isinstance(results, list):
                for d in results:
                    for k, v in d.items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(k, float(v))
        except Exception:
            pass

        # Artifacts
        out_dir = self.cfg.OUTPUT_DIR
        for fname in ["metrics.json", "last_checkpoint"]:
            fpath = os.path.join(out_dir, fname)
            if os.path.exists(fpath):
                mlflow.log_artifact(fpath)

        # Save model
        try:
            import mlflow.pytorch as mlflow_pt
            mlflow_pt.log_model(self.trainer.model, artifact_path="model")
        except Exception:
            pass

        mlflow.end_run()
