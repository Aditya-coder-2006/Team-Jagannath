import wandb
import os
from typing import Dict, Any

class Logger:
    def __init__(self, config):
        self.use_wandb = config.use_wandb
        if self.use_wandb:
            try:
                api_key = os.environ.get("WANDB_API_KEY", "")
                if api_key:
                    wandb.login(key=api_key)
                else:
                    print("W&B API key not found in WANDB_API_KEY environment variable. If unauthenticated, it may fail.")
                    
                wandb.init(
                    project=config.wandb_project,
                    config={
                        "model": config.model.__dict__,
                        "training": config.training.__dict__
                    }
                )
            except Exception as e:
                print(f"Failed to initialize wandb: {e}")
                self.use_wandb = False
                
    def log(self, metrics: Dict[str, Any], step: int = None):
        if self.use_wandb:
            wandb.log(metrics, step=step)
            
        # Also print to console
        metrics_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
        prefix = f"Step {step} | " if step is not None else ""
        print(f"{prefix}{metrics_str}")
        
    def finish(self):
        if self.use_wandb:
            wandb.finish()
