import os
import pandas as pd
from datetime import datetime
from sklearn.metrics import confusion_matrix

class ExcelTrainingLogger:
    def __init__(self, base_dir="results", model_name="model"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_name = model_name
        self.timestamp = timestamp
        self.filename = f"{model_name}_{timestamp}.xlsx"
        self.path = os.path.join(base_dir, self.filename)
        os.makedirs(base_dir, exist_ok=True)

        self.param_table = []
        self.epoch_metrics = []
        self.batch_metrics = []
        self.meta_lrs = []
        self.confusion_matrices = {}
        self.final_confusion = None
        self.hyperparams = {}
        self.dataset_summary = {}
        self.model_path = ""

    def log_param_counts(self, model):
        total, trainable, frozen = 0, 0, 0
        for name, module in model.named_children():
            mod_total = mod_train = mod_frozen = 0
            for p in module.parameters(recurse=False):
                count = p.numel()
                mod_total += count
                if p.requires_grad:
                    mod_train += count
                else:
                    mod_frozen += count
            if mod_total > 0:
                self.param_table.append({
                    "module": name,
                    "total": mod_total,
                    "trainable": mod_train,
                    "frozen": mod_frozen
                })
                total += mod_total
                trainable += mod_train
                frozen += mod_frozen
        self.param_table.append({
            "module": "TOTAL",
            "total": total,
            "trainable": trainable,
            "frozen": frozen
        })

    def log_confusion_matrix(self, y_true, y_pred, classes, epoch=None):
        cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
        df = pd.DataFrame(cm, index=classes, columns=classes)
        if epoch is None:
            self.final_confusion = df
        else:
            self.confusion_matrices[f"Epoch {epoch}"] = df

    def log_hyperparams(self, config: dict):
        self.hyperparams = config

    def log_dataset_summary(self, summary: dict):
        self.dataset_summary = summary

    def log_model_path(self, path: str):
        self.model_path = path

    def log_epoch_metrics(self, epoch, loss, acc, time_sec, gpu_mem_bytes):
        self.epoch_metrics.append({
            "epoch": epoch,
            "loss": loss,
            "accuracy": acc,
            "time_sec": round(time_sec, 2),
            "gpu_mem_mb": round(gpu_mem_bytes / 1024 / 1024, 2)
        })

    def log_batch_metrics(self, epoch, batch_idx, loss, acc):
        self.batch_metrics.append({
            "epoch": epoch,
            "batch": batch_idx,
            "loss": loss,
            "accuracy": acc
        })

    def log_metalr_lrs(self, lrs):
        self.meta_lrs.append(lrs)

    def save(self):
        with pd.ExcelWriter(self.path) as writer:
            if self.param_table:
                pd.DataFrame(self.param_table).to_excel(writer, sheet_name="Parameters", index=False)
            if self.epoch_metrics:
                pd.DataFrame(self.epoch_metrics).to_excel(writer, sheet_name="Epoch Metrics", index=False)
            if self.batch_metrics:
                pd.DataFrame(self.batch_metrics).to_excel(writer, sheet_name="Batch Metrics", index=False)
            if self.meta_lrs:
                pd.DataFrame(self.meta_lrs).to_excel(writer, sheet_name="MetaLR LRs", index=False)
            if self.final_confusion is not None:
                self.final_confusion.to_excel(writer, sheet_name="Final Confusion")
            for epoch, df in self.confusion_matrices.items():
                df.to_excel(writer, sheet_name=f"Confusion {epoch}")
            if self.hyperparams:
                pd.DataFrame(list(self.hyperparams.items()), columns=["Hyperparameter", "Value"]).to_excel(writer, sheet_name="Hyperparameters", index=False)
            if self.dataset_summary:
                pd.DataFrame(list(self.dataset_summary.items()), columns=["Metric", "Value"]).to_excel(writer, sheet_name="Dataset Summary", index=False)
            if self.model_path:
                pd.DataFrame([{"Saved Model Path": self.model_path}]).to_excel(writer, sheet_name="Model File", index=False)

        return self.path

def make_logger(model_name, suffix="", base_dir="logs"):
    if suffix:
        model_name = f"{model_name}_{suffix}"
    return ExcelTrainingLogger(base_dir=base_dir, model_name=model_name)
