import hashlib
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import confusion_matrix

def make_logger(model_name, config, timestamp="", base_dir="logs"):
    return ExcelTrainingLogger(model_name=model_name, config=config, timestamp=timestamp, base_dir=base_dir)

class ExcelTrainingLogger:
    def __init__(self, model_name, config, timestamp, base_dir="logs"):
        self.model_name = model_name
        self.config = config
        self.config_str = json.dumps(config, sort_keys=True)
        self.config_id = hashlib.md5(self.config_str.encode()).hexdigest()[:8]

        os.makedirs(base_dir, exist_ok=True)
        self.path = os.path.join(base_dir, f"{model_name} {timestamp}.xlsx")

        # Tables to append
        self.param_table = []
        self.epoch_metrics = []
        self.batch_metrics = []
        self.meta_lrs = []
        self.confusion_matrices = []
        self.hyperparams = []
        self.dataset_summary = []
        self.model_path = None

    def log_param_counts(self, model):
        total, trainable, frozen = 0, 0, 0
        for name, module in model.named_children():
            mod_total = mod_train = mod_frozen = 0
            for p in module.parameters():
                count = p.numel()
                mod_total += count
                if p.requires_grad:
                    mod_train += count
                else:
                    mod_frozen += count
            if mod_total > 0:
                self.param_table.append({
                    "config_id": self.config_id,
                    "module": name,
                    "total": mod_total,
                    "trainable": mod_train,
                    "frozen": mod_frozen
                })
                total += mod_total
                trainable += mod_train
                frozen += mod_frozen
        self.param_table.append({
            "config_id": self.config_id,
            "module": "TOTAL",
            "total": total,
            "trainable": trainable,
            "frozen": frozen
        })

    def log_confusion_matrix(self, y_true, y_pred, classes, epoch=None):
        cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
        for i, actual_label in enumerate(classes):
            row = {
                "config_id": self.config_id,
                "epoch": epoch if epoch is not None else "final",
                "actual": actual_label
            }
            row.update({pred_label: cm[i, j] for j, pred_label in enumerate(classes)})
            self.confusion_matrices.append(row)

    def log_hyperparams(self):
        row = {"config_id": self.config_id}
        row.update(self.config)
        self.hyperparams.append(row)

    def log_dataset_summary(self, summary: dict):
        row = {"config_id": self.config_id}
        row.update(summary)
        self.dataset_summary.append(row)

    def log_model_path(self, path: str):
        self.model_path = path

    def log_epoch_metrics(self, epoch, train_loss, val_loss, acc, time_sec, gpu_mem_bytes):
        self.epoch_metrics.append({
            "config_id": self.config_id,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": acc,
            "time_sec": round(time_sec, 2),
            "gpu_mem_mb": round(gpu_mem_bytes / 1024 / 1024, 2)
        })

    def log_batch_metrics(self, epoch, batch_idx, loss, acc, meta_loss="N/A"):
        self.batch_metrics.append({
            "config_id": self.config_id,
            "epoch": epoch,
            "batch": batch_idx,
            "loss": loss,
            "accuracy": acc,
            "metaLoss": meta_loss
        })

    def log_metalr_lrs(self, epoch, batch, lrs):
        row = {"config_id": self.config_id, "epoch": epoch, "batch": batch}
        row.update({f"lr_{i}": val for i, val in enumerate(lrs)})
        self.meta_lrs.append(row)

    def save(self):
        self.compute_and_store_metrics()

        mode = "a" if os.path.exists(self.path) else "w"
        writer_args = {
            "path": self.path,
            "mode": mode,
            "engine": "openpyxl"
        }
        if mode == "a":
            writer_args["if_sheet_exists"] = "overlay"

        with pd.ExcelWriter(**writer_args) as writer:
            self._append_to_sheet(writer, "Hyperparameters", self.hyperparams)
            self._append_to_sheet(writer, "Parameters", self.param_table)
            self._append_to_sheet(writer, "Dataset Summary", self.dataset_summary)
            self._append_to_sheet(writer, "Epoch Metrics", self.epoch_metrics)
            self._append_to_sheet(writer, "Batch Metrics", self.batch_metrics)
            self._append_to_sheet(writer, "MetaLR LRs", self.meta_lrs)
            self._append_to_sheet(writer, "Metrics", self.metrics)

            if self.model_path:
                pd.DataFrame([{
                    "config_id": self.config_id,
                    "model_path": self.model_path
                }]).to_excel(writer, sheet_name="Model File", index=False)

            if self.confusion_matrices:
                df_all_cm = pd.DataFrame(self.confusion_matrices)
                try:
                    existing = pd.read_excel(self.path, sheet_name="Confusion Matrix")
                    df_combined = pd.concat([existing, df_all_cm], ignore_index=True)
                except Exception:
                    df_combined = df_all_cm
                df_combined.to_excel(writer, sheet_name="Confusion Matrix", index=False)


    def _append_to_sheet(self, writer, sheet_name, new_data):
        if not new_data:
            return
        df_new = pd.DataFrame(new_data)
        try:
            existing = pd.read_excel(self.path, sheet_name=sheet_name)
            df_combined = pd.concat([existing, df_new], ignore_index=True)
        except Exception:
            df_combined = df_new
        df_combined.to_excel(writer, sheet_name=sheet_name, index=False)

    def compute_and_store_metrics(self):
        if not self.confusion_matrices:
            return

        df_cm = pd.DataFrame(self.confusion_matrices)
        classes = [col for col in df_cm.columns if col not in ["config_id", "epoch", "actual"]]

        metrics_rows = []

        for (config_id, epoch), group in df_cm.groupby(["config_id", "epoch"]):
            # Rebuild confusion matrix
            cm = []
            for actual_class in classes:
                row = group[group["actual"] == actual_class][classes].values[0]
                cm.append(row)
            cm = np.array(cm)

            for i, cls in enumerate(classes):
                TP = cm[i, i]
                FN = cm[i, :].sum() - TP
                FP = cm[:, i].sum() - TP
                TN = cm.sum() - (TP + FP + FN)
                support = cm[i, :].sum()

                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                accuracy = (TP + TN) / cm.sum()

                metrics_rows.append({
                    "config_id": config_id,
                    "epoch": epoch,
                    "class": cls,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "accuracy": accuracy,
                    "support": support
                })

        self.metrics = metrics_rows