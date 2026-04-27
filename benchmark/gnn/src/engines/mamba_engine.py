import time

import numpy as np
import torch
import wandb
from gnn.src.base.engine import BaseEngine


class MambaEngine(BaseEngine):
    def __init__(self, **args):
        super().__init__(**args)
        for p in self.model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.uniform_(p)

    def train(self):
        self._logger.info("Start training!")

        wait = 0
        min_loss = np.inf

        for epoch in range(self._max_epochs):
            t1 = time.time()
            mtrain_loss, mtrain_mape, mtrain_rmse = self.train_batch()
            t2 = time.time()

            v1 = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse = self.evaluate("val")
            v2 = time.time()

            cur_lr = self._lrate if self._lr_scheduler is None else self._lr_scheduler.get_last_lr()[0]
            if self._lr_scheduler is not None:
                self._lr_scheduler.step()

            self._logger.info(
                "Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, "
                "Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, "
                "Train Time: {:.4f}s/epoch, Valid Time: {:.4f}s, LR: {:.4e}".format(
                    epoch + 1,
                    mtrain_loss, mtrain_rmse, mtrain_mape,
                    mvalid_loss, mvalid_rmse, mvalid_mape,
                    t2 - t1, v2 - v1, cur_lr,
                )
            )

            wandb.log({
                "epoch": epoch + 1,
                "train/loss": mtrain_loss,
                "train/rmse": mtrain_rmse,
                "train/mape": mtrain_mape,
                "val/loss":   mvalid_loss,
                "val/rmse":   mvalid_rmse,
                "val/mape":   mvalid_mape,
                "lr":         cur_lr,
            })

            if mvalid_loss < min_loss:
                self.save_model(self._save_path)
                self._logger.info("Val loss decrease from {:.4f} to {:.4f}".format(min_loss, mvalid_loss))
                min_loss = mvalid_loss
                wait = 0
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info("Early stop at epoch {}, loss = {:.6f}".format(epoch + 1, min_loss))
                    break

        test_results = self.evaluate("test")
        wandb.finish()
        return test_results
