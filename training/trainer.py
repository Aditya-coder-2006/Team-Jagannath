import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import time

from utils.metrics import kendall_tau_score, pairwise_accuracy, log_metric, reset_metrics, get_avg_metric
from utils.logging_utils import Logger

class Trainer:
    def __init__(self, model, config, train_loader, val_loader, criterion):
        self.model = model
        self.full_config = config
        self.config = config.training
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=self.config.weight_decay
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.epochs)
        self.logger = Logger(config)
    
    @staticmethod
    def _format_seconds(seconds: float) -> str:
        total = max(0, int(seconds))
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def train_epoch(self, epoch: int):
        self.model.train()
        reset_metrics()
        total_batches = len(self.train_loader)
        log_interval = max(1, total_batches // 20)
        epoch_start = time.time()
        print(f"[Epoch {epoch}/{self.config.epochs}] Training started | batches={total_batches}")
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Detailed implementation heavily depends on how batch is constructed.
            # Assuming batch yields frames, flows, pairs, triplets, targets.
            
            self.optimizer.zero_grad()
            
            frames = batch["frames"].to(self.device)
            flows = batch["flows"].to(self.device)
            
            # Predict
            # 1. Base Encoding
            embeddings = self.model.encode_frames(frames, flows)
            
            # 2. Extract pairs for BCE and Triplets for classification
            # For brevity, let's assume we have index mappings in batch:
            p_i = batch["pair_i"].to(self.device)
            p_j = batch["pair_j"].to(self.device)
            pair_targets = batch["pair_targets"].to(self.device)
            
            t_a = batch["trip_a"].to(self.device)
            t_b = batch["trip_b"].to(self.device)
            t_c = batch["trip_c"].to(self.device)
            trip_targets = batch["trip_targets"].to(self.device)
            batch_indices = torch.arange(frames.size(0), device=self.device).unsqueeze(-1)
            
            # Form paired embeddings
            emb_i = embeddings[batch_indices, p_i]
            emb_j = embeddings[batch_indices, p_j]
            pair_logits = self.model.pairwise_comparator(emb_i.view(-1, emb_i.size(-1)), emb_j.view(-1, emb_j.size(-1)))
            
            emb_a = embeddings[batch_indices, t_a]
            emb_b = embeddings[batch_indices, t_b]
            emb_c = embeddings[batch_indices, t_c]
            trip_logits = self.model.triplet_module(emb_a.view(-1, emb_a.size(-1)), emb_b.view(-1, emb_b.size(-1)), emb_c.view(-1, emb_c.size(-1)))
            
            loss, loss_pair, loss_trip = self.criterion(
                pair_logits, 
                pair_targets.view(-1, 1).float(), 
                trip_logits, 
                trip_targets.view(-1)
            )
            
            loss.backward()
            self.optimizer.step()
            
            # Logging metrics
            pair_acc = pairwise_accuracy(
                torch.sigmoid(pair_logits).detach().cpu().numpy().reshape(-1),
                pair_targets.detach().cpu().numpy().reshape(-1),
            )
            
            log_metric("train_loss", loss.item())
            log_metric("train_pair_loss", loss_pair.item())
            log_metric("train_trip_loss", loss_trip.item())
            log_metric("train_acc", pair_acc)

            if (batch_idx + 1) == 1 or (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
                elapsed = time.time() - epoch_start
                done = batch_idx + 1
                pct = (done / max(1, total_batches)) * 100.0
                run_done = ((epoch - 1) * total_batches) + done
                run_total = max(1, self.config.epochs * total_batches)
                run_pct = (run_done / run_total) * 100.0
                avg_batch = elapsed / max(1, done)
                eta = avg_batch * (total_batches - done)
                print(
                    f"[Epoch {epoch}/{self.config.epochs}] "
                    f"{pct:6.2f}% ({done}/{total_batches}) "
                    f"| Run {run_pct:6.2f}% "
                    f"| ETA {self._format_seconds(eta)} "
                    f"| loss={loss.item():.4f} acc={pair_acc:.4f}"
                )
            
        return {
            "loss": get_avg_metric("train_loss"),
            "train_acc": get_avg_metric("train_acc"),
            "pair_loss": get_avg_metric("train_pair_loss"),
            "trip_loss": get_avg_metric("train_trip_loss"),
        }
        
    def validate(self):
        self.model.eval()
        reset_metrics()
        total_batches = len(self.val_loader)
        val_start = time.time()
        print(f"[Validation] Started | batches={total_batches}")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                frames = batch["frames"].to(self.device)
                flows = batch["flows"].to(self.device)
                
                # Pairwise ground truth mappings
                p_i = batch["pair_i"].to(self.device)
                p_j = batch["pair_j"].to(self.device)
                pair_targets = batch["pair_targets"].to(self.device)
                batch_indices = torch.arange(frames.size(0), device=self.device).unsqueeze(-1)
                
                embeddings = self.model.encode_frames(frames, flows)
                
                # Pair logits
                emb_i = embeddings[batch_indices, p_i]
                emb_j = embeddings[batch_indices, p_j]
                pair_logits = self.model.pairwise_comparator(emb_i.view(-1, emb_i.size(-1)), emb_j.view(-1, emb_j.size(-1)))
                
                # Calculate Val Loss (just BCE on pairs for val proxy)
                val_loss = torch.nn.functional.binary_cross_entropy_with_logits(pair_logits, pair_targets.view(-1, 1).float())
                
                # Accuracy
                val_acc = pairwise_accuracy(
                    torch.sigmoid(pair_logits).detach().cpu().numpy().reshape(-1),
                    pair_targets.detach().cpu().numpy().reshape(-1),
                )
                
                log_metric("val_loss", val_loss.item())
                log_metric("val_acc", val_acc)
                log_metric("kendall_tau", val_acc) # Temporary placeholder correlation

                if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == total_batches:
                    elapsed = time.time() - val_start
                    done = batch_idx + 1
                    pct = (done / max(1, total_batches)) * 100.0
                    avg_batch = elapsed / max(1, done)
                    eta = avg_batch * (total_batches - done)
                    print(
                        f"[Validation] {pct:6.2f}% ({done}/{total_batches}) "
                        f"| ETA {self._format_seconds(eta)} "
                        f"| val_loss={val_loss.item():.4f} val_acc={val_acc:.4f}"
                    )
                
        return {
            "val_loss": get_avg_metric("val_loss"),
            "val_acc": get_avg_metric("val_acc"),
            "kendall_tau": get_avg_metric("kendall_tau")
        }

    def run(self):
        best_tau = -1.0
        run_start = time.time()
        
        for epoch in range(1, self.config.epochs + 1):
            elapsed_total = time.time() - run_start
            completed_epochs = max(0, epoch - 1)
            if completed_epochs > 0:
                avg_epoch = elapsed_total / completed_epochs
                eta_total = avg_epoch * (self.config.epochs - completed_epochs)
                run_eta_label = self._format_seconds(eta_total)
            else:
                run_eta_label = "estimating..."
            print(
                f"\n========== Epoch {epoch}/{self.config.epochs} "
                f"| Run {(completed_epochs / max(1, self.config.epochs)) * 100.0:6.2f}% "
                f"| Run ETA {run_eta_label} =========="
            )
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            self.scheduler.step()
            
            metrics = {
                "epoch": epoch,
                "learning_rate": self.scheduler.get_last_lr()[0],
                **train_metrics,
                **val_metrics
            }
            
            self.logger.log(metrics, step=epoch)
            print(
                f"[Epoch {epoch}] done: "
                f"run={(epoch / max(1, self.config.epochs)) * 100.0:6.2f}%, "
                f"train_loss={metrics['loss']:.4f}, train_acc={metrics['train_acc']:.4f}, "
                f"val_loss={metrics['val_loss']:.4f}, val_acc={metrics['val_acc']:.4f}"
            )
            
            if val_metrics["kendall_tau"] > best_tau:
                best_tau = val_metrics["kendall_tau"]
                self.save_checkpoint("best_model.pt")
                
        self.logger.finish()
        
    def save_checkpoint(self, filename: str):
        path = f"{self.full_config.output_dir}/{filename}"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, path)
