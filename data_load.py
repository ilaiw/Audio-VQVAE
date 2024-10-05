import torch
import librosa
import pandas as pd
import numpy as np


def align_len(array: np.ndarray, target_length: int, axis: int = -1) -> np.ndarray:

    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array.take(indices=range(target_length), axis=axis)

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, filenames, seq_len=64000, sample_rate=16000):
        self.filenames = pd.Series(filenames)
        self.seq_len = seq_len
        self.sample_rate = sample_rate

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.filenames)

    def __getitem__(self, index):
        'Generates one sample of data'
        wav_path = self.filenames[index]
        waveform, sr = librosa.load(wav_path, mono=True, sr=self.sample_rate)
        waveform = align_len(waveform, target_length=self.seq_len)
        return waveform
