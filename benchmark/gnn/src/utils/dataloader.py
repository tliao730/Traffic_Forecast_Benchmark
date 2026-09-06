import math
import os
import pickle
import torch
import numpy as np
import threading
import multiprocessing as mp

class DataLoader(object):
    def __init__(self, data, idx, seq_len, horizon, bs, logger, pad_last_sample=False):
        self.original_idx = idx.copy()
        if pad_last_sample:
            num_padding = (bs - (len(idx) % bs)) % bs
            idx_padding = np.repeat(idx[-1:], num_padding, axis=0)
            idx = np.concatenate([idx, idx_padding], axis=0)
        
        self.data = data
        self.idx = idx
        self.size = len(idx)
        self.bs = bs
        self.num_batch = int(np.ceil(self.size / self.bs)) if self.bs > 0 else 0
        self.current_ind = 0
        logger.info('Sample num: ' + str(self.idx.shape[0]) + ', Batch num: ' + str(self.num_batch))
        
        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)
        self.seq_len = seq_len
        self.horizon = horizon


    def shuffle(self):
        perm = np.random.permutation(self.size)
        idx = self.idx[perm]
        self.idx = idx


    def write_to_shared_array(self, x, y, idx_ind, start_idx, end_idx):
        for i in range(start_idx, end_idx):
            x[i] = self.data[idx_ind[i] + self.x_offsets, :, :]
            y[i] = self.data[idx_ind[i] + self.y_offsets, :, :1]


    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                idx_ind = self.idx[start_ind: end_ind, ...]

                # all the sensor values at the same time. (don't change)
                x_shape = (len(idx_ind), self.seq_len, self.data.shape[1], self.data.shape[-1])
                x_shared = mp.RawArray('f', int(np.prod(x_shape)))
                x = np.frombuffer(x_shared, dtype='f').reshape(x_shape)

                y_shape = (len(idx_ind), self.horizon, self.data.shape[1], 1)
                y_shared = mp.RawArray('f', int(np.prod(y_shape)))
                y = np.frombuffer(y_shared, dtype='f').reshape(y_shape)

                array_size = len(idx_ind)
                num_threads = max(1, len(idx_ind) // 2)
                chunk_size = max(1, array_size // num_threads)
                threads = []
                for i in range(num_threads):
                    start_index = i * chunk_size
                    end_index = start_index + chunk_size if i < num_threads - 1 else array_size
                    thread = threading.Thread(target=self.write_to_shared_array, args=(x, y, idx_ind, start_index, end_index))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

                yield (x, y)
                self.current_ind += 1

        return _wrapper()


    def get_iterator_with_idx(self, idx_override):
        size = len(idx_override)
        if size == 0:
            return iter(())
        num_batch = int(np.ceil(size / self.bs)) if self.bs > 0 else 0
        current_ind = 0

        def _wrapper():
            nonlocal current_ind
            while current_ind < num_batch:
                start_ind = self.bs * current_ind
                end_ind = min(size, self.bs * (current_ind + 1))
                idx_ind = idx_override[start_ind: end_ind, ...]

                x_shape = (len(idx_ind), self.seq_len, self.data.shape[1], self.data.shape[-1])
                x_shared = mp.RawArray('f', int(np.prod(x_shape)))
                x = np.frombuffer(x_shared, dtype='f').reshape(x_shape)

                y_shape = (len(idx_ind), self.horizon, self.data.shape[1], 1)
                y_shared = mp.RawArray('f', int(np.prod(y_shape)))
                y = np.frombuffer(y_shared, dtype='f').reshape(y_shape)

                array_size = len(idx_ind)
                num_threads = max(1, len(idx_ind) // 2)
                chunk_size = max(1, array_size // num_threads)
                threads = []
                for i in range(num_threads):
                    start_index = i * chunk_size
                    end_index = start_index + chunk_size if i < num_threads - 1 else array_size
                    thread = threading.Thread(target=self.write_to_shared_array, args=(x, y, idx_ind, start_index, end_index))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

                yield (x, y)
                current_ind += 1

        return _wrapper()


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)


    def transform(self, data):
        return (data - self.mean) / self.std


    def inverse_transform(self, data):
        return (data * self.std) + self.mean


# Maps GNN horizon to gift_eval term name, matching TERM_TO_PRED_LEN in mamba.py.
_HORIZON_TO_TERM = {3: 'short', 6: 'medium', 12: 'long'}

# Mirrors common.get_prediction_length(): the eval pipeline overrides whatever
# prediction_length the raw gift_eval Dataset carries (48/480/720 at 15T).
_TERM_TO_PRED_LEN = {'short': 3, 'medium': 6, 'long': 12}

# Maps dataset arg name to gift_eval dataset prefix.
_DATASET_TO_GIFT_EVAL = {
    'SD':  'sd',
    'CA':  'ca',
    'GLA': 'gla',
    'GBA': 'gba',
}

# Path to gift_eval data, relative to this file's repo root.
_GIFT_EVAL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'dataset', 'LargeST', 'gift_eval')
)


def _gift_eval_anchors(data, args, logger):
    """
    Compute test anchor indices that exactly align with gift_eval's test windows,
    by querying the gift_eval Dataset object directly.

    gift_eval splits from the end of the series: the last window's label ends at
    T, and each window is spaced pred_len apart. Using test_stride on idx_test
    (which starts from idx_test[0]) causes an off-by-one because idx_test[0] is
    not necessarily a multiple of pred_len. This function back-calculates the
    correct anchors from the end, bypassing that issue.

    Requires args.dataset and args.horizon to be set. Falls back to None if the
    dataset or term is not recognised, or if gift_eval data is unavailable.
    """
    term = _HORIZON_TO_TERM.get(int(getattr(args, 'horizon', 0)))
    ds_key = str(getattr(args, 'dataset', '')).upper()
    ds_prefix = _DATASET_TO_GIFT_EVAL.get(ds_key)
    years = str(getattr(args, 'years', ''))

    if term is None or ds_prefix is None or not years:
        return None

    gift_eval_name = f"{ds_prefix}/{years}/15T"
    gift_eval_path = _GIFT_EVAL_PATH

    if not os.path.isdir(os.path.join(gift_eval_path, gift_eval_name)):
        logger.warning(f"gift_eval data not found at {gift_eval_path}/{gift_eval_name}, falling back to stride mode")
        return None

    try:
        os.environ['GIFT_EVAL'] = gift_eval_path
        from gift_eval.data import Dataset
        ds = Dataset(name=gift_eval_name, term=term, to_univariate=False)

        # common.py overrides prediction_length and windows right after building
        # the Dataset, so the raw attributes here (48/480/720, windows 15/2/1)
        # are not what FM/ML models are actually evaluated on. Mirror those two
        # overrides exactly, or the anchors line up with windows nobody uses.
        pred_len = _TERM_TO_PRED_LEN[term]
        min_len  = ds._min_series_length
        windows  = min(max(1, math.ceil(0.1 * min_len / pred_len)), 20)
        # Input length of the last window = min_series_length - pred_len
        # (gift_eval trims each series to min_series_length before generating windows)
        inp_len  = min_len - pred_len
    except Exception as e:
        logger.warning(f"Could not load gift_eval dataset ({e}), falling back to stride mode")
        return None

    T = data.shape[0]

    # The test sequence starts at T - (last_inp_len + pred_len).
    # Window i: anchor = seq_start + first_inp_len + i*pred_len - 1
    seq_start     = T - (inp_len + pred_len)
    first_inp_len = inp_len - (windows - 1) * pred_len
    anchors = np.array([
        seq_start + (first_inp_len + i * pred_len) - 1
        for i in range(windows)
    ])

    logger.info(
        f"gift_eval aligned anchors ({term}): [{anchors[0]}, {anchors[-1]}], "
        f"count={len(anchors)}, pred_len={pred_len}"
    )
    return anchors


def load_dataset(data_path, args, logger):
    ptr = np.load(os.path.join(data_path, args.years, 'his.npz'))
    data = ptr['data']
    logger.info('Data shape: ' + str(data.shape))

    dataloader = {}
    for cat in ['train', 'val', 'test']:
        idx = np.load(os.path.join(data_path, args.years, 'idx_' + cat + '.npy'))
        if cat == 'test':
            # Always use gift_eval-aligned anchors for test to ensure the same
            # non-overlapping windows as FM models. The stride-based fallback
            # starts from idx_test[0], which is not a multiple of pred_len and
            # causes anchors to be off by 1 vs gift_eval.
            aligned_anchors = _gift_eval_anchors(data, args, logger)
            if aligned_anchors is not None:
                idx = aligned_anchors
            else:
                # Fallback: filter by time range then subsample by stride.
                test_split = float(getattr(args, 'test_split', 0) or 0)
                if test_split > 0:
                    T = data.shape[0]
                    min_idx = int((1 - test_split) * T)
                    idx = idx[idx >= min_idx]
                    logger.info(
                        f"test_split={test_split}: filtered to last {test_split*100:.0f}% "
                        f"of timeline ({len(idx)} samples, idx>={min_idx})"
                    )

                test_stride_mode = getattr(args, 'test_stride_mode', 'fixed')
                if test_stride_mode == 'fixed':
                    # Stride defaults to 1 (every timestep is a test window).
                    # Set to horizon to get non-overlapping windows.
                    test_stride = max(1, int(getattr(args, 'test_stride', 1)))
                    if test_stride > 1:
                        idx = idx[::test_stride]

                    # Keep only the last N windows (0 = keep all).
                    test_num_windows = int(getattr(args, 'test_num_windows', 0))
                    if test_num_windows > 0:
                        idx = idx[-test_num_windows:]

        dataloader[cat + '_loader'] = DataLoader(data[..., :args.input_dim], idx,
                                                 args.seq_len, args.horizon, args.bs, logger)
        if cat == 'test':
            dataloader[cat + '_loader'].test_stride_mode = getattr(args, 'test_stride_mode', 'fixed')
            dataloader[cat + '_loader'].test_stride = int(getattr(args, 'test_stride', 1))
            dataloader[cat + '_loader'].test_num_windows = int(getattr(args, 'test_num_windows', 0))

    scaler = StandardScaler(mean=ptr['mean'], std=ptr['std'])
    return dataloader, scaler


def load_adj_from_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj_from_numpy(numpy_file):
    return np.load(numpy_file)


def get_dataset_info(dataset):
    base_dir = os.getcwd() + '/data/'
    d = {
         'CA': [base_dir+'ca', base_dir+'ca/ca_rn_adj.npy', 8600],
         'GLA': [base_dir+'gla', base_dir+'gla/gla_rn_adj.npy', 3834],
         'GBA': [base_dir+'gba', base_dir+'gba/gba_rn_adj.npy', 2352],
         'SD': [base_dir+'sd', base_dir+'sd/sd_rn_adj.npy', 716],
        }
    assert dataset in d.keys()
    return d[dataset]