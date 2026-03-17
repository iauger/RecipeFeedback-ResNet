# src/trainer.py

import os
import time
import torch
import torch.nn as nn
from enum import Enum
from tqdm import tqdm
from src.config import load_settings
from src.models import AblationType, HeadType


class LossFunc(Enum):
    MSE = "mse"
    HUBER = "huber"
    LOG_COSH = "log_cosh"


class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        x = y_pred - y_true
        return torch.mean(torch.log(torch.cosh(x + 1e-12)))


class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        """
        Trainer class to encapsulate training, validation, checkpointing,
        and final evaluation for the RecipeNet model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.criterion = self._build_criterion(getattr(config, "loss_fn", LossFunc.HUBER))
        self.optimizer = self._setup_optimizer()

        self.history = {
            "model_type": None,
            "ablation_type": None,
            "loss_type": None,
            "train_loss": [],
            "val_loss": [],
            "grad_norm": [],
        }

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=5,
            factor=0.5
        )

        self.current_ablation = AblationType.ALL_FEATURES
        self.best_model_path = None

    def _build_criterion(self, loss_setting):
        val = loss_setting.value if isinstance(loss_setting, LossFunc) else loss_setting

        if val == LossFunc.MSE.value:
            return nn.MSELoss()
        elif val == LossFunc.LOG_COSH.value:
            return LogCoshLoss()
        else:
            return nn.HuberLoss()

    def _setup_optimizer(self):
        base_lr = self.config.learning_rate

        # Treat both the learned head and regressor as higher-LR components
        head_keywords = ("head", "regressor")

        base_params = [
            p for n, p in self.model.named_parameters()
            if not any(k in n for k in head_keywords)
        ]
        head_params = [
            p for n, p in self.model.named_parameters()
            if any(k in n for k in head_keywords)
        ]

        return torch.optim.AdamW(
            [
                {"params": base_params, "lr": base_lr},
                {"params": head_params, "lr": base_lr * self.config.lr_mult},
            ],
            weight_decay=self.config.weight_decay
        )

    def compute_grad_norm(self) -> float:
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        epoch_grad_norms = []

        for meta_x, tag_x, targets, ids, names in self.train_loader:
            meta_x = meta_x.to(self.device)
            tag_x = tag_x.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(meta_x, tag_x, ablation=self.current_ablation)
            loss = self.criterion(outputs, targets)

            loss.backward()

            # Optional but recommended for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            grad_norm = self.compute_grad_norm()
            epoch_grad_norms.append(grad_norm)

            self.optimizer.step()
            total_loss += loss.item()

        avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms) if epoch_grad_norms else 0.0
        self.history["grad_norm"].append(avg_grad_norm)

        avg_loss = total_loss / len(self.train_loader)
        self.history["train_loss"].append(avg_loss)
        return avg_loss

    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for meta_x, tag_x, targets, ids, names in self.val_loader:
                meta_x = meta_x.to(self.device)
                tag_x = tag_x.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(meta_x, tag_x, ablation=self.current_ablation)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        self.history["val_loss"].append(avg_loss)
        return avg_loss

    def fit(self, epochs: int, head_type: HeadType, ablation: AblationType, loss_fn: LossFunc):
        s = load_settings()

        self.criterion = self._build_criterion(loss_fn)
        self.current_ablation = ablation

        self.history["loss_type"] = loss_fn.value
        self.history["ablation_type"] = ablation.value
        self.history["model_type"] = head_type.value

        # Save best checkpoint for this experiment configuration
        model_name = os.path.join(
            s.results_dir,
            f"best_model_{head_type.value}_{ablation.value}_{loss_fn.value}.pth"
        )
        self.best_model_path = model_name

        best_val_loss = float("inf")
        patience = 20
        trigger_times = 0

        epoch_pbar = tqdm(range(1, epochs + 1), desc="Training Progress", unit="epoch")

        for epoch in epoch_pbar:
            start_time = time.time()

            train_loss = self.train_epoch()
            val_loss = self.validate()

            epoch_time = time.time() - start_time

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), model_name)
                trigger_times = 0
            else:
                trigger_times += 1

            self.scheduler.step(val_loss)

            epoch_pbar.set_postfix({
                "T-Loss": f"{train_loss:.4f}",
                "V-Loss": f"{val_loss:.4f}",
                "Sec/Epoch": f"{epoch_time:.1f}"
            })

            if trigger_times >= patience:
                print(
                    f"\nEarly stopping triggered after {epoch} epochs. "
                    f"No improvement in validation loss for {patience} consecutive epochs."
                )
                break

        # Reload best checkpoint before leaving fit
        if self.best_model_path is not None and os.path.exists(self.best_model_path):
            self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))

        return self.history

    def evaluate(self, loader, head_type: HeadType, ablation: AblationType, return_embeddings: bool = False):
        """
        Final evaluation on the unseen test set to report final model performance.
        """
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        all_embeddings = []
        all_targets = []
        all_ids = []
        all_names = []
        all_predictions = []

        print(f"Evaluating {head_type.value} model on {len(loader.dataset)} test samples:")

        with torch.no_grad():
            for meta_x, tag_x, targets, ids, names in loader:
                meta_x = meta_x.to(self.device)
                tag_x = tag_x.to(self.device)
                targets = targets.to(self.device)

                if return_embeddings:
                    outputs, embeddings = self.model(
                        meta_x,
                        tag_x,
                        return_embeddings=True,
                        ablation=ablation
                    )
                    all_embeddings.append(embeddings.cpu())
                    all_targets.append(targets.cpu())
                    all_predictions.append(outputs.cpu())
                    all_ids.extend(ids)
                    all_names.extend(names)
                else:
                    outputs = self.model(
                        meta_x,
                        tag_x,
                        return_embeddings=False,
                        ablation=ablation
                    )

                loss = torch.nn.functional.mse_loss(outputs, targets)
                mae = torch.nn.functional.l1_loss(outputs, targets)

                total_loss += loss.item()
                total_mae += mae.item()

        avg_mse = total_loss / len(loader)
        avg_rmse = avg_mse ** 0.5
        avg_mae = total_mae / len(loader)

        metrics = {
            "test_mse": avg_mse,
            "test_rmse": avg_rmse,
            "test_mae": avg_mae
        }

        if return_embeddings:
            bundle = {
                "embeddings": torch.cat(all_embeddings),
                "targets": torch.cat(all_targets),
                "predictions": torch.cat(all_predictions),
                "recipe_ids": all_ids,
                "recipe_names": all_names
            }
            return metrics, bundle

        return metrics, None