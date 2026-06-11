import os
import argparse
import logging
import numpy as np

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
torch.set_num_threads(3)

from src.models.sttn import STTN
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
    parser.add_argument('--adj_type', type=str, default='doubletransition')
    parser.add_argument('--blocks', type=int, default=2)
    parser.add_argument('--mlp_expand', type=int, default=2)
    parser.add_argument('--hid_dim', type=int, default=32)
    parser.add_argument('--end_dim', type=int, default=512)

    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--wdecay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--clip_grad_value', type=float, default=5)
    args = parser.parse_args()

    log_dir = './experiments/{}/{}/'.format(args.model_name, args.dataset)
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)
    
    return args, log_dir, logger


def get_model_and_batches_for_eval_time(dataset_key, seq_len, horizon, estimation_samples):
    set_seed(2023)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = argparse.Namespace(
        device=str(device), dataset=dataset_key, years='2019', model_name='sttn', seed=2023,
        bs=1, seq_len=seq_len, horizon=horizon, input_dim=3, output_dim=1, mode='test',
        max_epochs=100, patience=30, adj_type='doubletransition', blocks=2, mlp_expand=2,
        hid_dim=32, end_dim=512, lrate=1e-3, wdecay=1e-4, dropout=0.1, clip_grad_value=5,
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)
    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = normalize_adj_mx(adj_mx, args.adj_type)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    dataloader, _ = load_dataset(data_path, args, logger)
    model = STTN(node_num=node_num, input_dim=args.input_dim, output_dim=args.output_dim,
                device=device, supports=supports, blocks=args.blocks, mlp_expand=args.mlp_expand,
                hidden_channels=args.hid_dim, end_channels=args.end_dim, dropout=args.dropout)
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
            model_name='sttn', model_path='experiments/sttn',
            get_model_and_batches_fn=get_model_and_batches_for_eval_time, estimation_samples=10,
        )
        return
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)
    
    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    logger.info('Adj path: ' + adj_path)

    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = normalize_adj_mx(adj_mx, args.adj_type)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    
    dataloader, scaler = load_dataset(data_path, args, logger)

    model = STTN(node_num=node_num,
                 input_dim=args.input_dim,
                 output_dim=args.output_dim,
                 device=device,
                 supports=supports,
                 blocks=args.blocks,
                 mlp_expand=args.mlp_expand,
                 hidden_channels=args.hid_dim,
                 end_channels=args.end_dim,
                 dropout=args.dropout,
                 )
    
    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = None

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
        wandb.init(project="TrafficFM", name=f"STTN_SD{args.years}_s{args.seed}", config=vars(args))
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