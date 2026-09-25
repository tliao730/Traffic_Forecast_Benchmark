import os
import random
import time

import h5py
import numpy as np
import torch
from gnn.src.utils.metrics import compute_all_metrics, masked_mape, masked_rmse


class BaseEngine:
    def __init__(
        self,
        device,
        model,
        dataloader,
        scaler,
        sampler,
        loss_fn,
        lrate,
        optimizer,
        scheduler,
        clip_grad_value,
        max_epochs,
        patience,
        log_dir,
        logger,
        seed,
    ):
        super().__init__()
        self._device = device
        self.model = model
        self.model.to(self._device)

        self._dataloader = dataloader
        self._scaler = scaler

        self._loss_fn = loss_fn
        self._lrate = lrate
        self._optimizer = optimizer
        self._lr_scheduler = scheduler
        self._clip_grad_value = clip_grad_value

        self._max_epochs = max_epochs
        self._patience = patience
        self._iter_cnt = 0
        self._save_path = log_dir
        self._logger = logger
        self._seed = seed

        self._logger.info("The number of parameters: {}".format(self.model.param_num()))

    def _to_device(self, tensors):
        if isinstance(tensors, list):
            return [tensor.to(self._device) for tensor in tensors]
        else:
            return tensors.to(self._device)

    def _to_numpy(self, tensors):
        if isinstance(tensors, list):
            return [tensor.detach().cpu().numpy() for tensor in tensors]
        else:
            return tensors.detach().cpu().numpy()

    def _to_tensor(self, nparray):
        if isinstance(nparray, list):
            return [torch.tensor(array, dtype=torch.float32) for array in nparray]
        else:
            return torch.tensor(nparray, dtype=torch.float32)

    def _inverse_transform(self, tensors):
        def inv(tensor):
            return self._scaler.inverse_transform(tensor)

        if isinstance(tensors, list):
            return [inv(tensor) for tensor in tensors]
        else:
            return inv(tensors)

    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = "final_model_s{}.pt".format(self._seed)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))

    def load_model(self, save_path):
        filename = "final_model_s{}.pt".format(self._seed)
        self.model.load_state_dict(
            torch.load(
                os.path.join(save_path, filename),
                map_location=self._device,
                weights_only=True,
            )
        )

    def _extra_checkpoint_state(self):
        # Curriculum-learning engines (D2STGNN, DGCRN) override this to
        # persist state that lives outside the model/optimizer.
        return {}

    def _load_extra_checkpoint_state(self, state):
        pass

    def _checkpoint_path(self):
        return os.path.join(
            self._save_path, "last_checkpoint_s{}.pt".format(self._seed)
        )

    def save_checkpoint(self, epoch, min_loss, wait, finished=False):
        if not os.path.exists(self._save_path):
            os.makedirs(self._save_path)
        ckpt = {
            "epoch": epoch,
            "min_loss": min_loss,
            "wait": wait,
            "finished": finished,
            "iter_cnt": self._iter_cnt,
            "model": self.model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "scheduler": (
                self._lr_scheduler.state_dict()
                if self._lr_scheduler is not None
                else None
            ),
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "extra": self._extra_checkpoint_state(),
        }
        path = self._checkpoint_path()
        # write-then-rename so a job killed mid-save can't corrupt the checkpoint
        torch.save(ckpt, path + ".tmp")
        os.replace(path + ".tmp", path)

    def load_checkpoint(self):
        path = self._checkpoint_path()
        if not os.path.exists(path):
            return 0, np.inf, 0, False
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        # optimizer/scheduler states are moved to the params' device automatically.
        # optimizer may be None in bootstrapped checkpoints (built from a
        # weights-only best model); training then resumes with fresh moments.
        if ckpt["optimizer"] is not None:
            self._optimizer.load_state_dict(ckpt["optimizer"])
        if self._lr_scheduler is not None and ckpt["scheduler"] is not None:
            # T_max is derived from --max_epochs, and state_dict() carries it,
            # so a checkpoint written under a different budget would silently
            # pin the old schedule -- resuming a 100-epoch cosine with
            # --max_epochs 40 stops at LR 3.4e-4 instead of annealing to
            # eta_min. The budget this run was launched with wins.
            configured_t_max = getattr(self._lr_scheduler, "T_max", None)
            self._lr_scheduler.load_state_dict(ckpt["scheduler"])
            if configured_t_max is not None:
                self._lr_scheduler.T_max = configured_t_max
        self._iter_cnt = ckpt.get("iter_cnt", 0)
        rng = ckpt.get("rng", {})
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        if rng.get("python") is not None:
            random.setstate(rng["python"])
        self._load_extra_checkpoint_state(ckpt.get("extra", {}))
        self._logger.info(
            "Resumed from checkpoint: epoch {}, min_loss {:.4f}, wait {}".format(
                ckpt["epoch"], ckpt["min_loss"], ckpt["wait"]
            )
        )
        return ckpt["epoch"], ckpt["min_loss"], ckpt["wait"], ckpt.get(
            "finished", False
        )

    def train_batch(self):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        self._dataloader["train_loader"].shuffle()
        for X, label in self._dataloader["train_loader"].get_iterator():
            self._optimizer.zero_grad()

            # X (b, t, n, f), label (b, t, n, 1)
            X, label = self._to_device(self._to_tensor([X, label]))
            pred = self.model(X, label)
            pred, label = self._inverse_transform([pred, label])

            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                print("Check mask value", mask_value)

            loss = self._loss_fn(pred, label, mask_value)
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()

            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self._clip_grad_value
                )
            self._optimizer.step()

            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)

            self._iter_cnt += 1
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)

    def train(self):
        start_epoch, min_loss, wait, finished = self.load_checkpoint()
        if finished:
            self._logger.info(
                "Training already finished according to checkpoint, "
                "running test only. Delete {} to retrain from scratch.".format(
                    self._checkpoint_path()
                )
            )
            self.evaluate("test")
            return
        if start_epoch == 0:
            self._logger.info("Start training!")

        for epoch in range(start_epoch, self._max_epochs):
            t1 = time.time()
            mtrain_loss, mtrain_mape, mtrain_rmse = self.train_batch()
            t2 = time.time()

            v1 = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse = self.evaluate("val")
            v2 = time.time()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            message = "Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Train Time: {:.4f}s/epoch, Valid Time: {:.4f}s, LR: {:.4e}"
            self._logger.info(
                message.format(
                    epoch + 1,
                    mtrain_loss,
                    mtrain_rmse,
                    mtrain_mape,
                    mvalid_loss,
                    mvalid_rmse,
                    mvalid_mape,
                    (t2 - t1),
                    (v2 - v1),
                    cur_lr,
                )
            )

            try:
                import wandb
                if getattr(wandb, 'run', None) is not None:
                    wandb.log({
                        "epoch": epoch + 1,
                        "train_loss": mtrain_loss,
                        "train_rmse": mtrain_rmse,
                        "train_mape": mtrain_mape,
                        "val_loss": mvalid_loss,
                        "val_rmse": mvalid_rmse,
                        "val_mape": mvalid_mape,
                    })
            except Exception:
                pass

            finished = False
            if mvalid_loss < min_loss:
                self.save_model(self._save_path)
                self._logger.info(
                    "Val loss decrease from {:.4f} to {:.4f}".format(
                        min_loss, mvalid_loss
                    )
                )
                min_loss = mvalid_loss
                wait = 0
            else:
                wait += 1
                if wait >= self._patience:
                    self._logger.info(
                        "Early stop at epoch {}, loss = {:.6f}".format(
                            epoch + 1, min_loss
                        )
                    )
                    finished = True

            if epoch + 1 == self._max_epochs:
                finished = True
            self.save_checkpoint(epoch + 1, min_loss, wait, finished)
            if finished:
                break

        self.evaluate("test")

    def evaluate(self, mode):
        if mode == "test":
            self.load_model(self._save_path)
        self.model.eval()

        preds = []
        labels = []
        with torch.no_grad():
            loader = self._dataloader[mode + "_loader"]
            if (
                mode == "test"
                and getattr(loader, "test_stride_mode", "fixed") == "per_horizon"
            ):
                test_mae = []
                test_mape = []
                test_rmse = []

                for i in range(self.model.horizon):
                    stride = i + 1
                    idx = loader.original_idx[::stride]
                    test_num_windows = getattr(loader, "test_num_windows", 0)
                    if test_num_windows > 0:
                        idx = idx[-test_num_windows:]

                    preds_i = []
                    labels_i = []
                    for X, label in loader.get_iterator_with_idx(idx):
                        # X (b, t, n, f), label (b, t, n, 1)
                        X, label = self._to_device(self._to_tensor([X, label]))
                        pred = self.model(X, label)
                        pred, label = self._inverse_transform([pred, label])

                        preds_i.append(pred.squeeze(-1).cpu())
                        labels_i.append(label.squeeze(-1).cpu())

                    preds_i = torch.cat(preds_i, dim=0)
                    labels_i = torch.cat(labels_i, dim=0)

                    mask_value = torch.tensor(0)
                    if labels_i.min() < 1:
                        mask_value = labels_i.min()

                    res = compute_all_metrics(
                        preds_i[:, i, :], labels_i[:, i, :], mask_value
                    )
                    log = "Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}"
                    self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                    test_mae.append(res[0])
                    test_mape.append(res[1])
                    test_rmse.append(res[2])

                log = "Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}"
                self._logger.info(
                    log.format(
                        np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape)
                    )
                )
                return
            else:
                # all the sensor values at the same time. (don't change)
                for X, label in loader.get_iterator():
                    # X (b, t, n, f), label (b, t, n, 1)
                    X, label = self._to_device(self._to_tensor([X, label]))
                    pred = self.model(X, label)
                    pred, label = self._inverse_transform([pred, label])

                    preds.append(pred.squeeze(-1).cpu())
                    labels.append(label.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        # handle the precision issue when performing inverse transform to label
        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == "val":
            mae = self._loss_fn(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return mae, mape, rmse

        elif mode == "test":
            test_mae = []
            test_mape = []
            test_rmse = []
            print("Check mask value", mask_value)
            for i in range(self.model.horizon):
                res = compute_all_metrics(preds[:, i, :], labels[:, i, :], mask_value)
                log = "Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}"
                self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])

            log = "Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}"
            self._logger.info(
                log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape))
            )

    def predict_and_save(
        self,
        mode,
        save_path,
        time_index=None,
        node_ids=None,
        pred_horizon=None,
        csv_path=None,
        ds_config=None,
    ):
        """Dump the test-split forecasts to h5 (+ CSV when csv_path is given).

        The windows are exactly the ones load_dataset built -- the gift_eval
        anchors -- so the dump lines up window for window with the FM/SSM/linear
        dumps. There is deliberately no way to trim them here.
        """
        if mode == "test":
            self.load_model(self._save_path)
        self.model.eval()

        preds = []
        labels = []
        with torch.no_grad():
            for X, label in self._dataloader[mode + "_loader"].get_iterator():
                X, label = self._to_device(self._to_tensor([X, label]))
                pred = self.model(X, label)
                pred, label = self._inverse_transform([pred, label])
                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0).numpy()
        labels = torch.cat(labels, dim=0).numpy()

        # Limit to first N prediction steps if specified
        if pred_horizon is not None and pred_horizon < preds.shape[1]:
            preds = preds[:, :pred_horizon, :]
            labels = labels[:, :pred_horizon, :]

        loader = self._dataloader[mode + "_loader"]
        sample_idx = loader.idx
        y_offsets = loader.y_offsets

        if pred_horizon is not None and pred_horizon < len(y_offsets):
            y_offsets = y_offsets[:pred_horizon]

        if node_ids is None:
            node_ids = np.arange(preds.shape[-1], dtype=np.int64)

        times = None
        if time_index is not None:
            time_index = np.asarray(time_index)
            times = time_index[sample_idx[:, None] + y_offsets[None, :]]

        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with h5py.File(save_path, "w") as f:
            f.create_dataset("pred", data=preds, compression="gzip")
            f.create_dataset("true", data=labels, compression="gzip")
            f.create_dataset("sample_idx", data=sample_idx, compression="gzip")
            f.create_dataset("y_offsets", data=y_offsets, compression="gzip")
            f.create_dataset("node_ids", data=node_ids, compression="gzip")
            if times is not None:
                # h5py cannot store datetime64; convert to int64 (ns since epoch)
                times_int = np.asarray(times).astype("int64")
                f.create_dataset("times", data=times_int, compression="gzip")

        if csv_path and ds_config and times is not None:
            self._save_predictions_csv(
                preds, labels, times, node_ids, ds_config, csv_path
            )

    def _save_predictions_csv(
        self, preds, labels, times, node_ids, ds_config, csv_path
    ):
        """Write the shared predictions-CSV schema common.save_predictions_csv uses.

        One truth row and one "mean" row per (window, node, step), in that
        nesting order. Built with numpy rather than a row loop: CA at the
        benchmark's 20 windows is already 20 x 8600 x 12 x 2 = 4.1M rows.
        """
        import pandas as pd

        num_samples, pred_len, num_nodes = preds.shape
        shape = (num_samples, num_nodes, pred_len)
        ts_str = (
            pd.DatetimeIndex(np.asarray(times).ravel())
            .strftime("%Y-%m-%d %H:%M")
            .to_numpy()
            .reshape(num_samples, pred_len)
        )
        sample = np.broadcast_to(np.arange(num_samples)[:, None, None], shape)
        item = np.broadcast_to(np.asarray(node_ids, dtype=np.int64)[None, :, None], shape)
        stamp = np.broadcast_to(ts_str[:, None, :], shape)
        # (window, step, node) -> (window, node, step), then truth/mean pairs
        value = np.stack(
            [labels.transpose(0, 2, 1), preds.transpose(0, 2, 1)], axis=-1
        ).astype(np.float64)

        df = pd.DataFrame(
            {
                "ds_config": ds_config,
                "sample_idx": np.repeat(sample.ravel(), 2),
                "item_id": np.repeat(item.ravel(), 2),
                "timestamp": np.repeat(stamp.ravel(), 2),
                "dim": 0,
                "stat": np.tile(np.array(["truth", "mean"]), sample.size),
                "value": value.ravel(),
            }
        )
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        df.to_csv(csv_path, index=False)
        self._logger.info(f"Saved predictions to {csv_path} ({len(df)} rows)")
