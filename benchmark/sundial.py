import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, set_seed
from tqdm.auto import tqdm
from gluonts.itertools import batcher
from gluonts.transform import LastValueImputation
from gluonts.model.forecast import SampleForecast

from common import eval, eval_time
from config import device  # 你原来的 device

set_seed(1)


class SundialPredictor:
    def __init__(
        self,
        num_samples: int,
        prediction_length: int,
        device_map,
        batch_size: int = 1024,
        model_path: str = "thuml/sundial-base-128m",
    ):
        print("prediction_length:", prediction_length)

        # ---- 统一 device 处理：支持 "cuda"/"cpu"/torch.device ----
        if isinstance(device_map, torch.device):
            self.device = device_map
        elif isinstance(device_map, str):
            self.device = torch.device(device_map)
        else:
            # 兜底：如果 config.device 不是预期类型
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.batch_size = batch_size

        # ---- 加载模型，并移动到同一设备 ----
        # 注意：trust_remote_code=True 会用 Sundial 自定义 generate/mixin
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        self.model.to(self.device)
        self.model.eval()

    def left_pad_and_stack_1D(self, tensors):
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

    def prepare_and_validate_context(self, context):
        if isinstance(context, list):
            context = self.left_pad_and_stack_1D(context)
        assert isinstance(context, torch.Tensor)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        assert context.ndim == 2
        return context

    @torch.no_grad()
    def predict(self, test_data_input, batch_x_shape: int = 2880):
        forecast_outputs = []

        while True:
            try:
                for batch in tqdm(batcher(test_data_input, batch_size=self.batch_size)):
                    context = [torch.tensor(entry["target"], dtype=torch.float32) for entry in batch]
                    batch_x = self.prepare_and_validate_context(context)

                    if batch_x.shape[-1] > batch_x_shape:
                        batch_x = batch_x[..., -batch_x_shape:]

                    # ---- NaN impute ----
                    if torch.isnan(batch_x).any():
                        bx = batch_x.cpu().numpy()
                        imputed_rows = []
                        for i in range(bx.shape[0]):
                            imputed_rows.append(LastValueImputation()(bx[i]))
                        batch_x = torch.tensor(np.vstack(imputed_rows), dtype=torch.float32)

                    # ---- 放到同一 device ----
                    batch_x = batch_x.to(self.device)

                    # ---- autocast 只在 cuda 上用 ----
                    use_amp = (self.device.type == "cuda")
                    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else torch.cpu.amp.autocast(enabled=False)

                    with ctx:
                        outputs = self.model.generate(
                            batch_x,
                            max_new_tokens=self.prediction_length,
                            revin=True,
                            num_samples=self.num_samples,

                            # =========================
                            # 方案A关键：关闭 cache，避免 DynamicCache.seen_tokens 报错
                            # =========================
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

        # ---- 转 gluonts Forecast ----
        forecasts = []
        for item, ts in zip(forecast_outputs, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            forecasts.append(SampleForecast(samples=item, start_date=forecast_start_date))

        return forecasts


def main():
    model_name = "sundial_base_128m"
    model_path = "thuml/sundial-base-128m"
    device_map = device

    def predictor_factory(dataset):
        return SundialPredictor(
            num_samples=100,
            prediction_length=dataset.prediction_length,
            device_map=device_map,
            model_path=model_path,
        )

    parser = argparse.ArgumentParser(description="Sundial evaluation or time estimation")
    parser.add_argument("--eval-time", action="store_true", help="Run eval_time (time estimation) only")
    args = parser.parse_args()

    if args.eval_time:
        eval_time(model_name, model_path, predictor_factory)
    else:
        eval(model_name, model_path, predictor_factory, batch_size=1024)


if __name__ == "__main__":
    main()
