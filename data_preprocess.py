import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
class Preprocessor():
    # max_freq: The maximum frequency to keep in the fft data
    # kaiser_beta: The beta value for the kaiser window
    # zero_freqs: A list of tuples that represent the frequency range to set to zero
    # overlap: The amount of overlap between chunks, 0 means no overlap, in seconds
    def __init__(self, max_freq=1000, kaiser_beta=10, zero_freqs=None, overlap=0):
        self.max_freq = max_freq
        self.kaiser_beta = kaiser_beta
        self.zero_freqs = [] if zero_freqs is None else zero_freqs
        self.overlap = overlap

    # Get the column of a dataframe
    @staticmethod
    def c(df:pd.DataFrame, col_number):
        return df.iloc[:, col_number]

    # Split data into chunks of size data_interval, data_interval is in seconds
    def split_in_chunks(self, mic_data, sampling_rate, data_interval):
        chunk_size = int(data_interval * sampling_rate)
        overlap_size = int(self.overlap * sampling_rate)
        chunks = []
        for i in range(0, len(mic_data), chunk_size - overlap_size):
            if i + chunk_size > len(mic_data):
                break
            chunks.append(mic_data[i:i+chunk_size])
        return chunks

    # Get the fft of the mic data
    def get_fft(self, mic_data, sampling_rate):
        window = np.kaiser(len(mic_data), self.kaiser_beta)
        mic_data = np.multiply(mic_data, window)
        mic_fft = np.fft.rfft(mic_data, norm="forward")
        mic_freq = np.fft.rfftfreq(len(mic_data), 1/sampling_rate)
        return mic_freq, mic_fft

    # Limit the frequency range of the fft data
    def limit_freq(self, mic_freq, mic_fft):
        indices = np.where(mic_freq <= self.max_freq)
        mic_freq = mic_freq[indices]
        mic_fft = mic_fft[indices]
        return mic_freq, mic_fft

    # Set the fft values of the zero_freqs to zero 
    # Used to test the importance of certain frequencies for classification
    def set_freqs_to_zero(self, freq, fft):
        for zero_freq in self.zero_freqs:
            start = np.where(freq >= zero_freq[0])[0][0]
            end = np.where(freq <= zero_freq[1])[0][-1]
            fft[start:end+1] = 0
        return fft

    # Add a column to the fft data that represents the type of motor
    def add_type_column(self, fft_data, result_type):
        result_num = 0
        if result_type == 'normal' or result_type == 'n':
            result_num = 0
        elif result_type == 'imbalanced' or result_type == 'i':
            result_num = 1
        elif result_type == 'faulty' or result_type == 'f':
            result_num = 2
        result = np.full((len(fft_data), 1), result_num)
        return np.append(fft_data, result, axis=1)
    
    # Add a column to the fft data that represents the motor drive frequency
    def add_hz_column(self, fft_data, hz):
        hz = np.full((len(fft_data), 1), hz)
        return np.append(fft_data, hz, axis=1)
    
    # One hot encode the type column
    def one_hot_encode(self, data):
        ohe = OneHotEncoder(categories=[[0, 1, 2]])
        data = np.array(data).reshape(-1, 1)
        ohe.fit(data)
        return ohe.transform(data).toarray()

    # Process the fft data of a set of chunks, usually all channels from a single csv file
    def processes_chunk_set(self, split_mic_data, sampling_rate):
        fft_list = []
        for chunk in split_mic_data:
            freq, fft = self.get_fft(chunk, sampling_rate)
            freq, fft = self.limit_freq(freq, fft)
            fft_mag = np.abs(fft)
            fft_mag = self.set_freqs_to_zero(freq, fft_mag)
            fft_list.append(fft_mag)
        return fft_list

    # Preprocess the data
    def compute_preprocess(self, data_df, data_interval, shuffle=True, keep_channels_separate=False):
        data_full = []
        for df_num, df in enumerate(data_df):
            data = df[0]
            channels = df[1]
            sampling_rate = df[2]
            hz = df[3]
            types = df[4]
            processed_channels = []
            for i, channel in enumerate(channels):
                split_mic_data = self.split_in_chunks(Preprocessor.c(data, channel), sampling_rate, data_interval)
                chunk_set_fft_amp = self.processes_chunk_set(split_mic_data, sampling_rate)
                hz_ind = i
                if len(hz) == 1:
                    hz_ind = 0
                chunk_set_fft_amp = self.add_hz_column(chunk_set_fft_amp, hz[hz_ind])
                chunk_set_fft_amp = self.add_type_column(chunk_set_fft_amp, types[i])
                processed_channels.append(chunk_set_fft_amp)
            # if processed_channels is false then combine the fft data of all channels
            if keep_channels_separate:
                processed_channels = np.array(processed_channels)
            else:
                processed_channels = np.concatenate(processed_channels, axis=0)
            data_full.append(processed_channels)
            print(f"{df_num+1} of {len(data_df)} finished")
        dataset = np.concatenate(data_full, axis=0)
        if shuffle:
            rand = np.random.default_rng()
            rand.shuffle(dataset)
        return dataset
