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
    parser.add_argument('--test_stride', type=int, default=1,
                        help='stride for selecting test windows (set to horizon for non-overlap predictions)')
    parser.add_argument('--test_num_windows', type=int, default=0,
                        help='number of test windows taken from the end (0 = use all after stride)')
    parser.add_argument('--test_stride_mode', type=str, default='fixed',
                        choices=['fixed', 'per_horizon'],
                        help='fixed: use test_stride for all horizons; per_horizon: stride = horizon index (1..H)')
    return parser