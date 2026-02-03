import argparse
from dotenv import load_dotenv

from common import eval, eval_time
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

# Load environment variables (for model caches, etc.)
load_dotenv()


# Load Moirai module once and reuse across datasets
_moirai_module = MoiraiModule.from_pretrained("Salesforce/moirai-1.0-R-small")


def main():
    model_name = "moirai_small"
    model_path = "Salesforce/moirai-1.0-R-small"

    context_length = 4000
    patch_size = 32
    num_samples = 20

    def predictor_factory(dataset):
        """
        Given a Dataset from common.eval, construct a Moirai GluonTS predictor.
        This mirrors the behavior of the old moirai.py script:
        - prediction_length is set per-term by common.eval
        - target_dim and past_feat_dynamic_real_dim come from the Dataset
        """
        model = MoiraiForecast(
            module=_moirai_module,
            prediction_length=dataset.prediction_length,
            context_length=context_length,
            patch_size=patch_size,
            num_samples=num_samples,
            target_dim=dataset.target_dim,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=dataset.past_feat_dynamic_real_dim,
        )

        # In the old script, model.create_predictor(batch_size=64) was passed directly
        predictor = model.create_predictor(batch_size=64)
        return predictor

    parser = argparse.ArgumentParser(description="Moirai evaluation or time estimation")
    parser.add_argument("--eval-time", action="store_true", help="Run eval_time (time estimation) only")
    args = parser.parse_args()

    if args.eval_time:
        eval_time(model_name, model_path, predictor_factory)
    else:
        eval(model_name, model_path, predictor_factory, batch_size=64)


if __name__ == "__main__":
    main()

