import argparse

def get_public_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--dataset', type=str, default='')
    # if need to use the data from multiple years, please use underline to separate them, e.g., 2018_2019
    parser.add_argument('--years', type=str, default='2019')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--seed', type=int, default=2023)

    parser.add_argument('--bs', type=int, default=64)
    # seq_len denotes input history length, horizon denotes output future length
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--output_dim', type=int, default=1)

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--log_dir', type=str, default='',
                        help='override default log/checkpoint directory')
    parser.add_argument('--pretrained_path', type=str, default='',
                        help='load model weights before training')
    # test-time window control
    parser.add_argument('--test_split', type=float, default=0,
                        help='use only last X of timeline as test (0.1=align with benchmark/gift_eval; 0=use full idx_test)')
    parser.add_argument('--test_stride', type=int, default=1,
                        help='stride for selecting test windows (set to horizon for non-overlap predictions)')
    parser.add_argument('--test_num_windows', type=int, default=0,
                        help='number of test windows taken from the end (0 = use all after stride)')
    parser.add_argument('--allow_stride_fallback', action='store_true',
                        help='evaluate on stride windows when gift_eval alignment is '
                             'unavailable, instead of failing. Off by default: stride '
                             'mode is a different test set (~7000 overlapping windows '
                             'over ~73 days vs 20 non-overlapping over ~2.4 days), so '
                             'its numbers are not comparable with the other families.')
    parser.add_argument('--test_stride_mode', type=str, default='fixed',
                        choices=['fixed', 'per_horizon'],
                        help='fixed: use test_stride for all horizons; per_horizon: stride = horizon index (1..H)')
    parser.add_argument('--save_predictions', action='store_true',
                        help='save predictions and ground truth to CSV (GNN experiments)')
    parser.add_argument('--pred_out_dir', type=str, default='',
                        help='directory for prediction CSV (default: results/ASTGCN/predictions)')
    parser.add_argument('--pred_horizon', type=int, default=0,
                        help='prediction horizon for saving (1-12, 0=use model horizon)')
    return parser