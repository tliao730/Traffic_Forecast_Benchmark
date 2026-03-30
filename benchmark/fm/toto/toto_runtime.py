import torch

from toto.inference.gluonts_predictor import Multivariate, TotoPredictor

DEFAULT_CONTEXT_LENGTH = 4096
DEFAULT_SUGGESTED_BATCH_SIZE = 16


def get_maximal_context_length(dataset):
    """
    Calculate the largest context window supported by the current dataset.
    """
    shortest_series_in_dataset = dataset._min_series_length
    total_prediction_windows = dataset.windows + 1
    max_context_length = (
        shortest_series_in_dataset
        - total_prediction_windows * dataset.prediction_length
    )
    return max_context_length


def get_total_gpu_memory():
    """Return total GPU VRAM capacity in MB."""
    if not torch.cuda.is_available():
        return 0
    torch.cuda.empty_cache()
    device = torch.cuda.current_device()
    return torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)


def calculate_optimal_batch_size(
    model,
    target_dim,
    prediction_length,
    context_length,
    use_kv_cache,
    num_samples,
    safety_factor=0.01,
):
    """
    Estimate a conservative batch size from the model shape and available GPU memory.
    """
    try:
        model_width = model.model.embed_dim
        model_depth = model.model.num_layers

        model_param_memory_mb = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) / (1024 * 1024)

        base_memory_per_sample = (model_width * model_depth * 4) / (1024 * 1024)
        io_memory = (target_dim * (context_length + prediction_length) * 4) / (
            1024 * 1024
        )

        kv_memory = 0
        if use_kv_cache:
            kv_memory = (model_depth * model_width * 2 * context_length * 4) / (
                1024 * 1024
            )

        mem_per_sample_mb = base_memory_per_sample + io_memory + kv_memory
        mem_per_batch_mb = mem_per_sample_mb * target_dim * num_samples

        gpu_mem = get_total_gpu_memory()
        cuda_reserved_mb = 1024

        available_memory = (
            gpu_mem - model_param_memory_mb - cuda_reserved_mb
        ) * safety_factor

        max_batch_size = max(
            1, int(available_memory / (mem_per_batch_mb / num_samples))
        )
        return min(DEFAULT_SUGGESTED_BATCH_SIZE, max_batch_size)
    except RuntimeError as e:
        print(f"Error calculating optimal batch size: {e}")
        return 1


class TOTOModelPredictorWrapper:
    """Adapt TotoPredictor and reduce samples_per_batch on CUDA OOM."""

    def __init__(
        self,
        model,
        prediction_length,
        context_length,
        mode,
        num_samples=128,
        use_kv_cache=True,
    ):
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.mode = mode
        self.num_samples = num_samples
        self.use_kv_cache = use_kv_cache
        self.samples_per_batch = num_samples
        self.model = model
        self._adjusted = False

        self._initialize_predictor()

    def _initialize_predictor(self):
        """Initialize TotoPredictor with the current samples_per_batch."""
        self.predictor = TotoPredictor.create_for_eval(
            model=self.model,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            mode=self.mode,
            samples_per_batch=self.samples_per_batch,
        )

    def predict(self, gluonts_test_data):
        """Run prediction while reducing samples_per_batch on OOM."""
        if not self._adjusted:
            print(
                "Initializing predictor with samples_per_batch =",
                self.samples_per_batch,
            )
            while self.samples_per_batch >= 1:
                try:
                    print(
                        "Attempting prediction with "
                        f"samples_per_batch = {self.samples_per_batch} "
                        f"and context_length = {self.context_length}"
                    )
                    predictions = list(
                        self.predictor.predict(
                            gluonts_test_data,
                            use_kv_cache=self.use_kv_cache,
                            num_samples=self.num_samples,
                        )
                    )
                    self._adjusted = True
                    return predictions
                except RuntimeError as e:
                    if "CUDA out of memory" not in str(e):
                        raise e

                    if self.samples_per_batch <= 1:
                        print(
                            "OOM at minimal batch size. Cannot proceed with this "
                            "context length and sample count."
                        )
                        raise e

                    print(
                        "Out of memory with "
                        f"samples_per_batch = {self.samples_per_batch}. "
                        "Reducing batch size."
                    )
                    self.samples_per_batch = self.samples_per_batch // 2
                    torch.cuda.empty_cache()
                    self._initialize_predictor()

        return self.predictor.predict(
            gluonts_test_data,
            use_kv_cache=self.use_kv_cache,
            num_samples=self.num_samples,
        )


class TotoPredictorWrapper:
    """Wrap Toto so it matches the predictor shape expected by common.eval()."""

    def __init__(
        self,
        model,
        dataset,
        prediction_length: int,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        num_samples: int = 256,
        use_kv_cache: bool = True,
        pad_short_series: bool = False,
    ):
        print("prediction_length:", prediction_length)
        self.model = model
        self.dataset = dataset
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.use_kv_cache = use_kv_cache
        self.pad_short_series = pad_short_series
        self.context_length = context_length

    def _get_context_length(self):
        """Calculate context length for the current dataset."""
        if self.pad_short_series:
            context_length = self.context_length
        else:
            max_context = get_maximal_context_length(self.dataset)
            context_length = min(self.context_length, max_context)
        return max(context_length, self.model.model.patch_embed.stride)

    def _get_suggested_batch_size(self, context_length):
        if self.dataset is None:
            return DEFAULT_SUGGESTED_BATCH_SIZE

        return calculate_optimal_batch_size(
            model=self.model,
            target_dim=self.dataset.target_dim,
            prediction_length=self.prediction_length,
            context_length=context_length,
            use_kv_cache=self.use_kv_cache,
            num_samples=self.num_samples,
        )

    def predict(self, test_data_input):
        """Create the adaptive Toto predictor for the current dataset and run it."""
        context_length = self._get_context_length()
        suggested_batch_size = self._get_suggested_batch_size(context_length)

        predictor_wrapper = TOTOModelPredictorWrapper(
            model=self.model,
            prediction_length=self.prediction_length,
            context_length=context_length,
            mode=Multivariate(batch_size=suggested_batch_size),
            num_samples=self.num_samples,
            use_kv_cache=self.use_kv_cache,
        )

        return predictor_wrapper.predict(test_data_input)
