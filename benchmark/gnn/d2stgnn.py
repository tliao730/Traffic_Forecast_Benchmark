import os
import argparse
import logging
import numpy as np

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
torch.set_num_threads(3)

from src.models.d2stgnn import D2STGNN
from src.engines.d2stgnn_engine import D2STGNN_Engine
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
    parser.add_argument('--num_feat', type=int, default=1)
    parser.add_argument('--num_hidden', type=int, default=32)
    parser.add_argument('--node_hidden', type=int, default=12)
    parser.add_argument('--time_emb_dim', type=int, default=12)
    parser.add_argument('--layer', type=int, default=5)
    parser.add_argument('--k_t', type=int, default=3)
    parser.add_argument('--k_s', type=int, default=2)
    parser.add_argument('--gap', type=int, default=3)
    parser.add_argument('--cl_epoch', type=int, default=3)
    parser.add_argument('--warm_epoch', type=int, default=30)
    parser.add_argument('--tpd', type=int, default=96)

    parser.add_argument('--lrate', type=float, default=2e-3)
    parser.add_argument('--wdecay', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--clip_grad_value', type=float, default=5)
    parser.add_argument('--gconv_chunk_size', type=int, default=0,
                        help='chunk size (over batch dim) for the graph-conv matmul in gconv(); '
                             '0=no chunking. Lower this to trade speed for peak GPU memory on '
                             'large-N datasets (GBA/GLA/CA).')
    args = parser.parse_args()

    log_dir = './experiments/{}/{}/{}/'.format(args.model_name, args.dataset, args.years)
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)
    
    return args, log_dir, logger


def get_model_and_batches_for_eval_time(dataset_key, seq_len, horizon, estimation_samples):
    set_seed(2023)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = argparse.Namespace(
        device=str(device), dataset=dataset_key, years='2019', model_name='d2stgnn', seed=2023,
        bs=1, seq_len=seq_len, horizon=horizon, input_dim=3, output_dim=1, mode='test',
        max_epochs=100, patience=30, num_feat=1, num_hidden=32, node_hidden=12, time_emb_dim=12,
        layer=5, k_t=3, k_s=2, gap=3, cl_epoch=3, warm_epoch=30, tpd=96,
        lrate=2e-3, wdecay=1e-5, dropout=0.1, clip_grad_value=5,
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)
    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = normalize_adj_mx(adj_mx, 'doubletransition')
    args.adjs = [torch.tensor(i).to(device) for i in adj_mx]
    dataloader, _ = load_dataset(data_path, args, logger)
    model = D2STGNN(node_num=node_num, input_dim=args.input_dim, output_dim=args.output_dim,
                    model_args=vars(args))
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
            model_name='d2stgnn', model_path='experiments/d2stgnn',
            get_model_and_batches_fn=get_model_and_batches_for_eval_time, estimation_samples=10,
        )
        return
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)
    
    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    logger.info('Adj path: ' + adj_path)

    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = normalize_adj_mx(adj_mx, 'doubletransition')
    args.adjs = [torch.tensor(i).to(device) for i in adj_mx]
    
    dataloader, scaler = load_dataset(data_path, args, logger)
    cl_step = args.cl_epoch * dataloader['train_loader'].num_batch
    warm_step = args.warm_epoch * dataloader['train_loader'].num_batch

    model = D2STGNN(node_num=node_num,
                    input_dim=args.input_dim,
                    output_dim=args.output_dim,
                    model_args=vars(args)
                    )
    
    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 38, 46, 54, 62, 70, 80], gamma=0.5)

    engine = D2STGNN_Engine(device=device,
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
                            cl_step=cl_step,
                            warm_step=warm_step,
                            horizon=args.horizon
                            )


    if args.mode == 'train':
        engine.train()
    else:
        engine.evaluate(args.mode)


if __name__ == "__main__":
    main()