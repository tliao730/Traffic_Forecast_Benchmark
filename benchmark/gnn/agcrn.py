import argparse
import logging
import os
import sys

import numpy as np

# TODO: remove this
sys.path.append(os.path.abspath(__file__ + "/../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

torch.set_num_threads(3)

from gnn.src.engines.agcrn_engine import AGCRN_Engine
from gnn.src.models.agcrn import AGCRN
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
    parser.add_argument("--embed_dim", type=int, default=10)
    parser.add_argument("--rnn_unit", type=int, default=64)
    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument("--cheb_k", type=int, default=2)

    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=0)
    parser.add_argument("--clip_grad_value", type=float, default=0)
    parser.add_argument("--checkpoint_steps", action="store_true",
                        help="gradient-checkpoint each AGCRNCell timestep instead of "
                             "keeping all timesteps resident for BPTT; trades speed for "
                             "peak GPU memory on large-N datasets (GBA/GLA/CA).")
    args = parser.parse_args()

    log_dir = "./experiments/{}/{}/{}/".format(args.model_name, args.dataset, args.years)
    logger = get_logger(log_dir, __name__, "record_s{}.log".format(args.seed))
    logger.info(args)

    return args, log_dir, logger


def get_model_and_batches_for_eval_time(
    dataset_key, seq_len, horizon, estimation_samples
):
    """Build model and collect test batches for eval_time (same protocol as benchmark)."""
    set_seed(2023)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = argparse.Namespace(
        device=str(device),
        dataset=dataset_key,
        years="2019",
        model_name="agcrn",
        seed=2023,
        bs=1,
        seq_len=seq_len,
        horizon=horizon,
        input_dim=3,
        output_dim=1,
        mode="test",
        max_epochs=100,
        patience=30,
        embed_dim=10,
        rnn_unit=64,
        num_layer=2,
        cheb_k=2,
        lrate=1e-3,
        wdecay=0,
        clip_grad_value=0,
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)
    data_path, _, node_num = get_dataset_info(args.dataset)
    dataloader, _ = load_dataset(data_path, args, logger)
    model = AGCRN(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        embed_dim=args.embed_dim,
        rnn_unit=args.rnn_unit,
        num_layer=args.num_layer,
        cheb_k=args.cheb_k,
    )
    model.to(device)
    batches = []
    for i, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
        batches.append((x, y))
        if len(batches) >= estimation_samples:
            break
    return model, batches


def main():
    if "--eval-time" in sys.argv:
        import eval_time_common

        eval_time_common.run_eval_time_for_experiment(
            model_name="agcrn",
            model_path="experiments/agcrn",
            get_model_and_batches_fn=get_model_and_batches_for_eval_time,
            estimation_samples=10,
        )
        return

    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)

    data_path, _, node_num = get_dataset_info(args.dataset)

    dataloader, scaler = load_dataset(data_path, args, logger)

    model = AGCRN(
        node_num=node_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        embed_dim=args.embed_dim,
        rnn_unit=args.rnn_unit,
        num_layer=args.num_layer,
        cheb_k=args.cheb_k,
        checkpoint_steps=args.checkpoint_steps,
    )

    loss_fn = masked_mae
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lrate, weight_decay=args.wdecay
    )
    scheduler = None

    engine = AGCRN_Engine(
        device=device,
        model=model,
        dataloader=dataloader,
        scaler=scaler,
        sampler=None,
        loss_fn=loss_fn,
        lrate=args.lrate,
        optimizer=optimizer,
        scheduler=scheduler,
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
