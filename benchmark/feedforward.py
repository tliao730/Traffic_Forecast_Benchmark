import torch

# PyTorch 2.6+ defaults weights_only=True; Lightning checkpoints need False to load.
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from common import eval
from gluonts.torch.model.simple_feedforward import SimpleFeedForwardEstimator


def main():
    model_name = "feedforward"
    model_path = "gluonts.simple_feedforward"

    def predictor_factory(dataset):
        """
        Construct and return a GluonTS SimpleFeedForward predictor
        given a Dataset object created by common.eval().
        """
        estimator = SimpleFeedForwardEstimator(
            prediction_length=dataset.prediction_length,
            context_length=dataset.prediction_length,
            trainer_kwargs=dict(
                max_epochs=100,
            ),
        )

        print(
            f"Training feedforward model "
            f"(dataset={dataset.name}, prediction_length={dataset.prediction_length})..."
        )
        predictor = estimator.train(dataset.validation_dataset)
        return predictor

    # Delegate the full benchmark loop to common.eval
    # batch_size matches the original script's evaluate_model batch_size (512)
    eval(model_name, model_path, predictor_factory, batch_size=512)


if __name__ == "__main__":
    main()

