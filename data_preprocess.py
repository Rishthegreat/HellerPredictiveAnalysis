import numpy as np


class Preprocessor():
    def __init__(self, max_freq=4000, kaiser_beta=10):
        self.max_freq = max_freq
        self.kaiser_beta = kaiser_beta

    def split_in_intervals(self, mic_data, sampling_rate, data_interval):
        mic_data_size = mic_data.size
        interval = sampling_rate*data_interval
        remainder = mic_data_size % interval
        num_intervals = mic_data_size//interval
        split_mic_data = np.array(np.split(mic_data.iloc[:mic_data_size-remainder], num_intervals))
        return split_mic_data

    def get_fft(self, mic_data, sampling_rate):
        window = np.kaiser(len(mic_data), self.kaiser_beta)
        mic_data = np.multiply(mic_data, window)
        mic_fft = np.fft.rfft(mic_data, norm="forward")
        mic_freq = np.fft.rfftfreq(len(mic_data), 1/sampling_rate)
        return mic_fft, mic_freq

    def cut_freq(self, mic_freq, max_freq):
        indices = np.where(mic_freq <= max_freq)
        mic_freq = mic_freq[indices]
        return mic_freq

    def get_fft_of_split_data(self, split_mic_data, sampling_rate, split:bool=True):
        split_mic_fft_amp = []
        split_mic_freq = np.fft.rfftfreq(len(split_mic_data[0]), 1/sampling_rate)
        if split:
            split_mic_freq = self.cut_freq(split_mic_freq, self.max_freq)
        for mic_data in split_mic_data:
            mic_fft, mic_freq = self.get_fft(mic_data, sampling_rate)
            mic_fft_amplitude = np.abs(mic_fft[:len(split_mic_freq)])
            split_mic_fft_amp.append(mic_fft_amplitude)
        return split_mic_fft_amp, split_mic_freq

    def add_type_column(self, fft_data, result_type):
        if result_type == 'normal':
            result = np.zeros((len(fft_data), 1))
        else:
            result = np.ones((len(fft_data), 1))
        return np.append(fft_data, result, axis=1)