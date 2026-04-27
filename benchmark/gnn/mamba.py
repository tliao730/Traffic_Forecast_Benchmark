import os
import sys

import numpy as np

sys.path.append(os.path.abspath(__file__ + "/../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import wandb

torch.set_num_threads(3)

from gnn.src.engines.mamba_engine import MambaEngine
from gnn.src.models.mamba_traffic import TrafficMamba
from gnn.src.utils.args import get_public_config
from gnn.src.utils.dataloader import get_dataset_info, load_dataset
from gnn.src.utils.logging import get_logger
from gnn.src.utils.metrics import masked_mae


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()
    parser.add_argument("--d_model",    type=int,   default=64)
    parser.add_argument("--d_state",    type=int,   default=16)
    parser.add_argument("--d_conv",     type=int,   default=4)
    parser.add_argument("--expand",     type=int,   default=2)
    parser.add_argument("--num_layers", type=int,   default=2)
    parser.add_argument("--embed_dim",  type=int,   default=10)
    parser.add_argument("--lrate",      type=float, default=1e-3)
    parser.add_argument("--wdecay",     type=float, default=0.0)
    parser.add_argument("--clip_grad_value", type=float, default=5.0)
    parser.add_argument("--wandb_project", type=str, default="TrafficFM")
    parser.add_argument("--max_train_samples", type=int, default=0,
                        help="截斷 train set 樣本數（0 = 全部使用）")
    args = parser.parse_args()

    log_dir = (
        args.log_dir if args.log_dir
        else "./experiments/{}/{}/".format(args.model_name, args.dataset)
    )
    logger = get_logger(log_dir, __name__, "record_s{}.log".format(args.seed))
    logger.info(args)
    return args, log_dir, logger


def print_data_shapes(dataloader, model, logger):
    x, y = next(dataloader["train_loader"].get_iterator())
    logger.info(f"[DataLoader] x shape: {x.shape}  (B, T, N, F)")
    logger.info(f"[DataLoader] y shape: {y.shape}  (B, horizon, N, 1)")

    x_t = torch.tensor(x[:2])
    model.eval()
    with torch.no_grad():
        out = model(x_t)
    model.train()
    logger.info(f"[TrafficMamba] input:  {tuple(x_t.shape)}")
    logger.info(f"[TrafficMamba] output: {tuple(out.shape)}")


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device if args.device else "cuda")

    data_path, _, node_num = get_dataset_info(args.dataset)
    dataloader, scaler = load_dataset(data_path, args, logger)

    if args.max_train_samples > 0:
        dl = dataloader["train_loader"]
        dl.idx = dl.idx[:args.max_train_samples]
        dl.size = len(dl.idx)
        dl.num_batch = int(np.ceil(dl.size / dl.bs))
        logger.info(f"max_train_samples={args.max_train_samples}: train set 截斷為 {dl.size} 個樣本")

    model = TrafficMamba(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        d_model=args.d_model,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        num_layers=args.num_layers,
        embed_dim=args.embed_dim,
    )

    if args.mode == "train":
        print_data_shapes(dataloader, model, logger)

    wandb.init(
        project=args.wandb_project,
        name=f"mamba_{args.dataset}_{args.years}_s{args.seed}",
        config=vars(args),
        mode="online" if args.mode == "train" else "disabled",
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lrate, weight_decay=args.wdecay
    )

    engine = MambaEngine(
        device=device,
        model=model,
        dataloader=dataloader,
        scaler=scaler,
        sampler=None,
        loss_fn=masked_mae,
        lrate=args.lrate,
        optimizer=optimizer,
        scheduler=None,
        clip_grad_value=args.clip_grad_value,
        max_epochs=args.max_epochs,
        patience=args.patience,
        log_dir=log_dir,
        logger=logger,
        seed=args.seed,
    )

    if args.mode == "train":
        engine.train()
    else:
        engine.evaluate(args.mode)


if __name__ == "__main__":
    main()
