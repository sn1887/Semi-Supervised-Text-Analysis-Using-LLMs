# New imports
import mlflow
from detectron2.engine import hooks
from detectron2.utils.events import get_event_storage

# Add CLI flags
p.add_argument("--mlflow-uri", type=str, default=None,
               help="file:///abs/path/to/outputs/mlruns or http://localhost:5000")
p.add_argument("--mlflow-experiment", type=str, default="detectron2")
p.add_argument("--mlflow-run-name", type=str, default="mask_rcnn_r50_fpn_3x")

# In main(), after cfg is finalized:
# 0) Configure MLflow local tracking
if args.mlflow-uri:
    mlflow.set_tracking_uri(args.mlflow-uri)
else:
    # default: local file store under project outputs/mlruns
    local_uri = "file://" + os.path.abspath(os.path.join(args.output, "..", "mlruns"))
    mlflow.set_tracking_uri(local_uri)

mlflow.set_experiment(args.mlflow_experiment)

# 1) A small MLflow hook for Detectron2
class MLflowHook(hooks.HookBase):
    def __init__(self, cfg):
        self.cfg = cfg
        self.run = None

    def before_train(self):
        self.run = mlflow.start_run(run_name=args.mlflow_run_name)
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
trainer.register_hooks([MLflowHook(cfg)])
trainer.resume_or_load(resume=args.resume)
trainer.train()
