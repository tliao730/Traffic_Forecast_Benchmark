"""
Run LSTM SHAP experiments for all shared_shap_data_experiments datasets.

It iterates over folders like:
  sd_2019_15T_ctx48_pred12
and launches interpret/lstm_shap_inter.py with matching seq_len/horizon.
"""

import os
import re
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shared_data_dir', type=str,
                        default='/home/defu/workspace/LargeST_pretrain/shared_shap_data_experiments')
    parser.add_argument('--output_root', type=str,
                        default='/home/defu/workspace/LargeST_pretrain/interpret/lstm_shap_shared_results')
    parser.add_argument('--log_dir_template', type=str,
                        default='/home/defu/workspace/LargeST_pretrain/experiments/lstm/SD_{seq_len}_{horizon}')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--background_size', type=int, default=10)
    parser.add_argument('--horizons', type=str, default='3,6,9,12')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    if not os.path.isdir(args.shared_data_dir):
        raise FileNotFoundError(f"shared_data_dir not found: {args.shared_data_dir}")

    os.makedirs(args.output_root, exist_ok=True)

    pattern = re.compile(r'^(?P<dataset>[a-z]+)_(?P<year>\d+)_15T_ctx(?P<ctx>\d+)_pred(?P<pred>\d+)$')

    subdirs = [d for d in os.listdir(args.shared_data_dir)
               if os.path.isdir(os.path.join(args.shared_data_dir, d))]
    subdirs = sorted(d for d in subdirs if pattern.match(d))

    if not subdirs:
        raise RuntimeError('No shared data folders matched the expected pattern.')

    for name in subdirs:
        m = pattern.match(name)
        ctx = int(m.group('ctx'))
        pred = int(m.group('pred'))

        log_dir = args.log_dir_template.format(seq_len=ctx, horizon=pred)
        output_dir = os.path.join(args.output_root, name)

        cmd = [
            'python', 'interpret/lstm_shap_inter.py',
            '--device', args.device,
            '--dataset', m.group('dataset').upper(),
            '--years', m.group('year'),
            '--seq_len', str(ctx),
            '--horizon', str(pred),
            '--input_dim', str(args.input_dim),
            '--output_dim', str(args.output_dim),
            '--log_dir', log_dir,
            '--shared_data_dir', args.shared_data_dir,
            '--shared_data_name', name,
            '--output_dir', output_dir,
            '--num_samples', str(args.num_samples),
            '--background_size', str(args.background_size),
            '--horizons', args.horizons,
            '--seed', str(args.seed),
        ]

        print('Running:', ' '.join(cmd))
        if args.dry_run:
            continue

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f'Command failed for {name}')


if __name__ == '__main__':
    main()
