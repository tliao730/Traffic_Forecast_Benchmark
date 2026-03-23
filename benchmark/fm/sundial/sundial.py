import argparse
import numpy as np
import torch
from gluonts.itertools import batcher
from gluonts.model.forecast import SampleForecast
from gluonts.transform import LastValueImputation
from tqdm.auto import tqdm

from common import eval, eval_time
from config import config as benchmark_config
from config import device

# Import transformers after benchmark config so HF cache env vars are applied
# before transformers/huggingface_hub computes dynamic module cache paths.
from transformers import AutoModelForCausalLM, set_seed

MODEL_NAME = "sundial_base_128m"
MODEL_PATH = "thuml/sundial-base-128m"
DEFAULT_BATCH_SIZE = 1024
DEFAULT_NUM_SAMPLES = 100
DEFAULT_CONTEXT_WINDOW = 2880

set_seed(1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sundial evaluation or time estimation")
    parser.add_argument(
        "--eval-time",
        action="store_true",
        help="Run eval_time (time estimation) only",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Evaluation batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Number of generated samples per series (default: {DEFAULT_NUM_SAMPLES})",
    )
    return parser


class SundialPredictor:
    def __init__(
        self,
        num_samples: int,
        prediction_length: int,
        device_map,
        batch_size: int = DEFAULT_BATCH_SIZE,
        model_path: str = MODEL_PATH,
    ):
        print("prediction_length:", prediction_length)

        # Accept both string devices and torch.device inputs.
        if isinstance(device_map, torch.device):
            self.device = device_map
        elif isinstance(device_map, str):
            self.device = torch.device(device_map)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.batch_size = batch_size

        # trust_remote_code=True is required for Sundial custom generation code.
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                cache_dir=benchmark_config.hf_home,
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
            )

        self.model.to(self.device)
        self.model.eval()

    def _left_pad_and_stack_1d(self, tensors):
        max_len = max(len(c) for c in tensors)
        padded = []
        for c in tensors:
            assert isinstance(c, torch.Tensor)
            assert c.ndim == 1
            padding = torch.full(
                size=(max_len - len(c),), fill_value=torch.nan, device=c.device
            )
            padded.append(torch.concat((padding, c), dim=-1))
        return torch.stack(padded)

    def _prepare_and_validate_context(self, context):
        if isinstance(context, list):
            context = self._left_pad_and_stack_1d(context)
        assert isinstance(context, torch.Tensor)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        assert context.ndim == 2
        return context

    @torch.no_grad()
    def predict(self, test_data_input, batch_x_shape: int = DEFAULT_CONTEXT_WINDOW):
        forecast_outputs = []

        while True:
            try:
                for batch in tqdm(batcher(test_data_input, batch_size=self.batch_size)):
                    context = [torch.tensor(entry["target"], dtype=torch.float32) for entry in batch]
                    batch_x = self._prepare_and_validate_context(context)

                    if batch_x.shape[-1] > batch_x_shape:
                        batch_x = batch_x[..., -batch_x_shape:]

                    if torch.isnan(batch_x).any():
                        bx = batch_x.cpu().numpy()
                        imputed_rows = [LastValueImputation()(bx[i]) for i in range(bx.shape[0])]
                        batch_x = torch.tensor(np.vstack(imputed_rows), dtype=torch.float32)

                    batch_x = batch_x.to(self.device)

                    use_amp = self.device.type == "cuda"
                    autocast_ctx = (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if use_amp
                        else torch.autocast(device_type="cpu", enabled=False)
                    )

                    with autocast_ctx:
                        outputs = self.model.generate(
                            batch_x,
                            max_new_tokens=self.prediction_length,
                            revin=True,
                            num_samples=self.num_samples,
                            # Disable cache to avoid DynamicCache.seen_tokens issues.
                            use_cache=False,
                        )

                    forecast_outputs.append(outputs.detach().cpu().numpy())

                forecast_outputs = np.concatenate(forecast_outputs, axis=0)
                break

            except torch.cuda.OutOfMemoryError:
                print(
                    f"OutOfMemoryError at batch_size {self.batch_size}, reducing to {self.batch_size // 2}"
                )
                self.batch_size //= 2
                if self.batch_size < 1:
                    raise RuntimeError("batch_size reduced below 1, still OOM.")

        forecasts = []
        for item, ts in zip(forecast_outputs, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            forecasts.append(SampleForecast(samples=item, start_date=forecast_start_date))

        return forecasts


def main():
    args = _build_parser().parse_args()

    def predictor_factory(dataset):
        return SundialPredictor(
            num_samples=args.num_samples,
            prediction_length=dataset.prediction_length,
            device_map=device,
            batch_size=args.batch_size,
            model_path=MODEL_PATH,
        )

    if args.eval_time:
        eval_time(MODEL_NAME, MODEL_PATH, predictor_factory)
    else:
        eval(MODEL_NAME, MODEL_PATH, predictor_factory, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
