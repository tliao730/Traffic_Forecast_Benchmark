# first step sync uv
uv sync --project envs/moirai

# second step
cd your path/benchmark

# third step
CUDA_VISIBLE_DEVICES=1 \
MOIRAI2_L1_FINETUNE=1 \
MOIRAI2_L1_MAX_EPOCHS=3 \
MOIRAI2_L1_SERIES=512 \
uv run --project ../envs/moirai python ../Finetune/moirai2_finetune.py