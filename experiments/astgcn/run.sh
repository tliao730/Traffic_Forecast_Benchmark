# python experiments/astgcn/main.py --device cuda:2 --dataset GLA --years 2019 --model_name astgcn --seed 2023 --bs 17 --wdecay 0
# python experiments/astgcn/main.py --device cuda:2 --dataset GLA --years 2019 --model_name astgcn --seed 2024 --bs 17 --wdecay 0

# python experiments/astgcn/main.py --device cuda:2 --dataset GBA --years 2019 --model_name astgcn --seed 2023 --bs 45
# python experiments/astgcn/main.py --device cuda:2 --dataset GBA --years 2019 --model_name astgcn --seed 2024 --bs 45

# python experiments/astgcn/main.py --device cuda:2 --dataset SD --years 2019 --model_name astgcn --seed 2023 --bs 64
# python experiments/astgcn/main.py --device cuda:2 --dataset SD --years 2019 --model_name astgcn --seed 2024 --bs 64

# SD test: save predictions (auto-aligns with benchmark: test_split=0.1, stride=3, 192 windows)
# python experiments/astgcn/main.py --device cuda:0 --dataset SD --years 2019 --seq_len 48 --horizon 12 --input_dim 3 --output_dim 1 --mode test --log_dir experiments/astgcn/SD_48_12 --save_predictions --test_num_windows 192 --pred_horizon 3