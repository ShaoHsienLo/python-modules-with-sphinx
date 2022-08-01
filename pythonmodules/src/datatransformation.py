import pandas as pd
from numpy import ndarray
from scipy.fftpack import fft
import numpy as np
from scipy.signal import welch
from detect_peaks import detect_peaks
from typing import Tuple
import os
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


class TimeDomain:
    """


    Parameters
    ----------


    Returns
    -------
    None or Dataframe
    """

    def __init__(self, dataset: pd.DataFrame, window: int = 100, plot: bool = False,
                 time_col: str = "Timestamp"):
        self.dataset = dataset
        self.window = window
        self.plot = plot
        self.time_col = time_col

    def skew(self, x):
        x = skew(np.array(x))
        return x

    def rms(self, x):
        return np.sqrt(np.mean(np.power(x.values, 2)))

    def ptp(self, x):
        return np.ptp(x.values)

    def kur(self, x):
        return kurtosis(np.array(x.values))

    def crest(self, x):
        return max(x.values) / np.sqrt(np.mean(np.power(x.values, 2)))

    def shape(self, x):
        return np.sqrt(np.mean(np.power(x.values, 2))) / np.mean(np.abs(x.values))

    def impulse(self, x):
        return max(x.values) / np.mean(np.abs(x.values))

    def margin(self, x):
        return max(x.values) / np.power(np.mean(np.abs(x.values)), 2)

    def time_domain(self):
        agg_functions = ["mean", "std", "min", "max", "skew", "kurt", self.rms, self.ptp, self.crest,
                         self.shape, self.impulse, self.margin]
        dataframe = self.dataset.groupby([self.time_col]).agg(agg_functions)
        dataframe.columns = ["{}_{}".format(col[0], col[1]) for col in dataframe.columns]
        dataframe_ = dataframe.rolling(2).mean()
        dataframe_.columns = ["{}_rolling_avg".format(col) for col in dataframe_.columns]
        dataframe = pd.concat([dataframe, dataframe_], axis=1)
        dataframe = dataframe.fillna(method="bfill")
        dataframe.insert(0, self.time_col, dataframe.index)
        dataframe = dataframe.reset_index(drop=True)
        return dataframe


def get_values(y_values: list, f_s: int) -> Tuple[list, list]:
    """
    Generate index based on sampling frequency of signal data

    Parameters
    ----------
    y_values: list
        Signal data
    f_s: int
        Sampling frequency

    Returns
    -------
    Tuple: a tuple containing:
        - x_values (list): Index of signal data
        - y_values (list): Signal data
    """

    y_values = y_values
    x_values = [(1 / f_s) * kk for kk in range(0, len(y_values))]
    return x_values, y_values


def get_fft_values(y_values: list, T: float, N: int, plot: bool = False, path: str = None) -> Tuple[ndarray, float]:
    """
    Do Fast Fourier Transform(FFT)

    Parameters
    ----------
    y_values: list
        Signal data
    T: float
        Sampling interval
    N: int
        Number of samples
    plot: bool
        Whether to plot, the default is false
    path: str
        The location where the log files are stored, the default is None

    Returns
    -------
    Tuple: a tuple containing:
        - f_values (ndarray): Index of fast fourier transform
        - fft_values (float): Fast fourier transform signal data
    """

    f_values = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    fft_values_ = fft(y_values)
    fft_values = 2.0 / N * np.abs(fft_values_[0: N // 2])

    if plot:
        '''
        If no path is specified, a new log folder will be created in the current path, 
        and the log file will be stored in it
        '''
        directory = "fft-plot"
        if path is None:
            path = os.path.join(os.path.abspath(os.getcwd()), directory)
        else:
            path = os.path.join(path, directory)
        if not os.path.exists(path):
            os.makedirs(path)
        files_no = len(os.listdir(path))
        files_no = files_no + 1

        plt.plot(f_values, fft_values, linestyle="-", color="blue", label="signal_{}".format(files_no))
        plt.xlabel("Frequency [Hz]", fontsize=16)
        plt.ylabel("Amplitude", fontsize=16)
        plt.title("Frequency domain of the signal", fontsize=16)
        plt.legend()
        filename = "fft_{}.png".format(files_no)
        plt.savefig(os.path.join(path, filename))
        plt.close()

    return f_values, fft_values


def get_psd_values(y_values: list, f_s: int, plot: bool = False, path: str = None) -> Tuple[ndarray, ndarray]:
    """
    Do Power Spectral Density(PSD)

    Parameters
    ----------
    y_values: list
        Signal data
    f_s: int
        Sampling frequency
    plot: bool
        Whether to plot, the default is false
    path: str
        The location where the log files are stored, the default is None

    Returns
    -------
    Tuple: a tuple containing:
        - f_values (ndarray): Index of fast fourier transform
        - psd_values (ndarray): Power spectral density signal data
    """

    f_values, psd_values = welch(y_values, fs=f_s)

    if plot:
        '''
        If no path is specified, a new log folder will be created in the current path, 
        and the log file will be stored in it
        '''
        directory = "psd-plot"
        if path is None:
            path = os.path.join(os.path.abspath(os.getcwd()), directory)
        else:
            path = os.path.join(path, directory)
        if not os.path.exists(path):
            os.makedirs(path)
        files_no = len(os.listdir(path))
        files_no = files_no + 1

        plt.plot(f_values, psd_values, linestyle="-", color="blue", label="signal_{}".format(files_no))
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [V**2 / Hz]')
        plt.legend()
        filename = "psd_{}.png".format(files_no)
        plt.savefig(os.path.join(path, filename))
        plt.close()

    return f_values, psd_values


def autocorr(x: list) -> ndarray:
    """
    Autocorrelation of a 1-dimensional sequence.

    Parameters
    ----------
    x: list
        A 1-dimensional sequence

    Returns
    -------
    ndarray
        Autocorrelation of x
    """

    result = np.correlate(x, x, mode="full")
    result = result[len(result)//2:]
    return result


def get_autocorr_values(y_values: list, T: float, N: int, plot: bool = False, path: str = None) \
        -> Tuple[ndarray, ndarray]:
    """
    Get the value of the sequence autocorrelation

    Parameters
    ----------
    y_values: list
        Signal data
    T: float
        Sampling interval
    N: int
        Number of samples
    plot: bool
        Whether to plot, the default is false
    path: str
        The location where the log files are stored, the default is None

    Returns
    -------
    Tuple: a tuple containing:
        - x_values (ndarray): Index of the sequence
        - autocorr_values (ndarray): The value of the sequence autocorrelation
    """

    autocorr_values = autocorr(y_values)
    x_values = np.array([T * jj for jj in range(0, N)])

    if plot:
        '''
        If no path is specified, a new log folder will be created in the current path, 
        and the log file will be stored in it
        '''
        directory = "autocorr-plot"
        if path is None:
            path = os.path.join(os.path.abspath(os.getcwd()), directory)
        else:
            path = os.path.join(path, directory)
        if not os.path.exists(path):
            os.makedirs(path)
        files_no = len(os.listdir(path))
        files_no = files_no + 1

        plt.plot(x_values, autocorr_values, linestyle="-", color="blue", label="signal_{}".format(files_no))
        plt.xlabel('Time delay [s]')
        plt.ylabel('Autocorrelation amplitude')
        plt.legend()
        filename = "autocorr_{}.png".format(files_no)
        plt.savefig(os.path.join(path, filename))
        plt.close()

    return x_values, autocorr_values


def get_first_n_peaks(x: list, y: list, no_peaks: int = 5) -> Tuple[list, list]:
    """
    Get the indexes and values of the first N peaks

    Parameters
    ----------
    x: list
        Index of signal data
    y: list
        Signal data
    no_peaks: int
        First few peaks, the default is 5

    Returns
    -------
    Tuple: a tuple containing:
        - x [list]: Indexes of the first few peaks
        - y [list]: Values of the first few peaks
    """

    x_, y_ = list(x), list(y)
    if len(x_) >= no_peaks:
        return x_[:no_peaks], y_[:no_peaks]
    else:
        missing_no_peaks = no_peaks - len(x_)
        return x_ + [0] * missing_no_peaks, y_ + [0] * missing_no_peaks


def get_features(x_values: list, y_values: list, mph: float) -> list:
    """
    Detect peak indexes and values

    Parameters
    ----------
    x_values: list
        Index of signal data
    y_values: list
        Signal data
    mph: float
        Max peak height

    Returns
    -------
    List
        The result is the indexes and values of the first N peaks
    """

    indices_peaks = detect_peaks(y_values, mph=mph)
    peaks_x, peaks_y = get_first_n_peaks(x_values[indices_peaks], y_values[indices_peaks])
    return peaks_x + peaks_y


def extract_features_labels(dataset: ndarray, labels: ndarray, T: float, N: int, f_s: int, denominator: int) \
        -> Tuple[ndarray, ndarray]:
    """
    Get the input data of X and Y for training or testing in the machine learning model

    Parameters
    ----------
    dataset: ndarray
        3D signal data (training or testing dataest)
            - First dimension: The total number of batch data
            - Second dimension: The total number of sample data
            - Third dimension: The total number of parameters
    labels: ndarray
        1D Label data (training or testing dataest)
    T: float
        Sampling interval
    N: int
        Number of samples
    f_s: int
        Sampling frequency
    denominator: int
        Calculate the max peak height

    Returns
    -------
    Tuple: a tuple containing:
        - X [ndarray]: Input data in  the machine learning model
        - y [ndarray]: Input label in the machine learning model
    """

    percentile = 5
    list_of_features = []
    list_of_labels = []
    for signal_no in range(0, len(dataset)):
        features = []
        list_of_labels.append(labels[signal_no])
        for signal_comp in range(0, dataset.shape[2]):
            signal = dataset[signal_no, :, signal_comp]

            signal_min = np.nanpercentile(signal, percentile)
            signal_max = np.nanpercentile(signal, 100 - percentile)
            # ijk = (100 - 2*percentile)/10
            mph = signal_min + (signal_max - signal_min) / denominator

            features += get_features(*get_psd_values(signal, T, N, f_s), mph)
            features += get_features(*get_fft_values(signal, T, N, f_s), mph)
            features += get_features(*get_autocorr_values(signal, T, N, f_s), mph)
        list_of_features.append(features)
    return np.array(list_of_features), np.array(list_of_labels)
