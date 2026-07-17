import os
import argparse
import logging
import numpy as np

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
torch.set_num_threads(3)

from src.models.stgcn import STGCN
from src.base.engine import BaseEngine
from src.utils.args import get_public_config
from src.utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info
from src.utils.graph_algo import normalize_adj_mx
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()    
    parser.add_argument('--Kt', type=int, default=3)
    parser.add_argument('--Ks', type=int, default=3)
    parser.add_argument('--block_num', type=int, default=2)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)

    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--wdecay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--clip_grad_value', type=float, default=0)
    args = parser.parse_args()

    log_dir = './experiments/{}/{}/{}/'.format(args.model_name, args.dataset, args.years)
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)
    
    return args, log_dir, logger


def get_model_and_batches_for_eval_time(dataset_key, seq_len, horizon, estimation_samples):
    set_seed(2023)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = argparse.Namespace(
        device=str(device), dataset=dataset_key, years='2019', model_name='stgcn', seed=2023,
        bs=1, seq_len=seq_len, horizon=horizon, input_dim=3, output_dim=1, mode='test',
        max_epochs=100, patience=30, Kt=3, Ks=3, block_num=2, step_size=10, gamma=0.95,
        lrate=1e-3, wdecay=5e-4, dropout=0.5, clip_grad_value=0,
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)
    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)
    gso = normalize_adj_mx(adj_mx, 'scalap')[0]
    gso = torch.tensor(gso).to(device)
    Ko = args.seq_len - (args.Kt - 1) * 2 * args.block_num
    blocks = [[args.input_dim]] + [[64, 16, 64]] * args.block_num
    blocks.append([128] if Ko == 0 else [128, 128])
    blocks.append([args.horizon])
    dataloader, _ = load_dataset(data_path, args, logger)
    model = STGCN(node_num=node_num, input_dim=args.input_dim, output_dim=args.output_dim,
                  gso=gso, blocks=blocks, Kt=args.Kt, Ks=args.Ks, dropout=args.dropout)
    model.to(device)
    batches = []
    for i, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        batches.append((x, y))
        if len(batches) >= estimation_samples:
            break
    return model, batches


def main():
    if '--eval-time' in sys.argv:
        import eval_time_common
        eval_time_common.run_eval_time_for_experiment(
            model_name='stgcn', model_path='experiments/stgcn',
            get_model_and_batches_fn=get_model_and_batches_for_eval_time, estimation_samples=10,
        )
        return
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)
    
    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    logger.info('Adj path: ' + adj_path)
    
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(node_num)

    gso = normalize_adj_mx(adj_mx, 'scalap')[0]
    gso = torch.tensor(gso).to(device)

    Ko = args.seq_len - (args.Kt - 1) * 2 * args.block_num
    blocks = []
    blocks.append([args.input_dim])
    for l in range(args.block_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([args.horizon])

    dataloader, scaler = load_dataset(data_path, args, logger)

    model = STGCN(node_num=node_num,
                  input_dim=args.input_dim,
                  output_dim=args.output_dim,
                  gso=gso,
                  blocks=blocks,
                  Kt=args.Kt,
                  Ks=args.Ks,
                  dropout=args.dropout,
                  )
    
    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    engine = BaseEngine(device=device,
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
                        seed=args.seed
                        )

    try:
        import wandb
        wandb.init(project="TrafficFM", name=f"STGCN_SD{args.years}_s{args.seed}", config=vars(args))
    except Exception:
        pass

    if args.mode == 'train':
        engine.train()
    else:
        engine.evaluate(args.mode)

    try:
        import wandb
        if getattr(wandb, 'run', None) is not None:
            wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()