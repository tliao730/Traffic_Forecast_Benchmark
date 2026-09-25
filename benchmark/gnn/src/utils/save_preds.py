"""Dump per-timestamp GNN predictions in the shared predictions-CSV format.

Same schema as ``common.save_predictions_csv`` (the FM/SSM/linear path), so
one analysis script can read both families:

    ds_config, sample_idx, item_id, timestamp, dim, stat(truth|mean), value

Timestamps come from ``his.npz``: LargeST stores one calendar year per file at
15-minute resolution starting at Jan 1 00:00 of ``args.years``. That matches
the gift_eval arrow files -- sd/2018 test starts 2018-10-20 00:00, which is
exactly step 28032 of the 35040-step year -- so hour-of-day analysis lines up
across families.

Windows are whatever ``load_dataset`` built for the test split, i.e. the
gift_eval-aligned anchors (20 non-overlapping windows of ``args.horizon``
steps). Nothing here re-windows the test set.
"""
import os

import numpy as np
import pandas as pd

from .dataloader import _HORIZON_TO_TERM


def run_save_predictions(engine, args, data_path, logger, model_name):
    """Run one forward pass over the test split and write h5 + CSV."""
    horizon = int(getattr(args, 'pred_horizon', 0) or args.horizon)
    term = _HORIZON_TO_TERM.get(horizon, str(horizon))
    year = str(args.years)

    ptr = np.load(os.path.join(data_path, year, 'his.npz'))
    n_steps = ptr['data'].shape[0]
    time_index = pd.date_range(f'{year}-01-01', periods=n_steps, freq='15min')

    region = args.dataset.lower()
    ds_config = f'{region}/{year}/{term}'
    out_dir = getattr(args, 'pred_out_dir', '') or os.path.join(
        'results', model_name.upper(), 'predictions'
    )
    os.makedirs(out_dir, exist_ok=True)

    # Mirrors the "_prof{N}" suffix eval() puts on the FM/SSM/linear side: a
    # profiling dump spans more days than the benchmark windows and must not be
    # mistaken for one.
    profile_windows = int(getattr(args, 'profile_windows', 0) or 0)
    prof_tag = f'_prof{profile_windows}' if profile_windows > 0 else ''
    stem = f'{model_name.lower()}_{region}_{year}_{term}{prof_tag}'
    h5_path = os.path.join(out_dir, f'{stem}_pred.h5')
    # .csv.gz -- see the note in common.save_predictions_csv's caller: these
    # dumps are ~500 MB raw and /scratch runs near its project quota.
    csv_path = os.path.join(out_dir, f'{stem}_predictions.csv.gz')

    logger.info(f'Saving predictions for {ds_config} (horizon={horizon}) to {out_dir}')
    engine.predict_and_save(
        mode='test',
        save_path=h5_path,
        time_index=time_index.values,
        pred_horizon=horizon,
        csv_path=csv_path,
        ds_config=ds_config,
    )
