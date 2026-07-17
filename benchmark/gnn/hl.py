import os
import argparse
import logging
import numpy as np

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
torch.set_num_threads(3)

from src.models.hl import HL
from src.engines.hl_engine import HL_Engine
from src.utils.args import get_public_config
from src.utils.dataloader import load_dataset, get_dataset_info
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
    args = parser.parse_args()

    log_dir = './experiments/{}/{}/{}/'.format(args.model_name, args.dataset, args.years)
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)
    
    return args, log_dir, logger


def get_model_and_batches_for_eval_time(dataset_key, seq_len, horizon, estimation_samples):
    set_seed(2023)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = argparse.Namespace(
        device=str(device), dataset=dataset_key, years='2019', model_name='hl', seed=2023,
        bs=1, seq_len=seq_len, horizon=horizon, input_dim=3, output_dim=1, mode='test',
        max_epochs=100, patience=30,
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)
    data_path, _, node_num = get_dataset_info(args.dataset)
    dataloader, _ = load_dataset(data_path, args, logger)
    model = HL(node_num=node_num, input_dim=args.input_dim, output_dim=args.output_dim)
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
            model_name='hl', model_path='experiments/hl',
            get_model_and_batches_fn=get_model_and_batches_for_eval_time, estimation_samples=10,
        )
        return
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)
    
    data_path, _, node_num = get_dataset_info(args.dataset)
    
    dataloader, scaler = load_dataset(data_path, args, logger)

    model = HL(node_num=node_num,
               input_dim=args.input_dim,
               output_dim=args.output_dim
               )
    
    loss_fn = masked_mae
    optimizer = None
    scheduler = None

    engine = HL_Engine(device=device,
                       model=model,
                       dataloader=dataloader,
                       scaler=scaler,
                       sampler=None,
                       loss_fn=loss_fn,
                       lrate=0,
                       optimizer=optimizer,
                       scheduler=scheduler,
                       clip_grad_value=0,
                       max_epochs=args.max_epochs,
                       patience=args.patience,
                       log_dir=log_dir,
                       logger=logger,
                       seed=args.seed
                       )

    if args.mode == 'train':
        engine.train()
    else:
        engine.evaluate(args.mode)


if __name__ == "__main__":
    main()