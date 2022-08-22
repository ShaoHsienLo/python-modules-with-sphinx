import pandas as pd
from numpy import ndarray
from scipy.fftpack import fft
import numpy as np
from scipy.signal import welch
from detect_peaks import detect_peaks
from typing import Tuple
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from pandas_profiling import ProfileReport
import sweetviz as sv
import dataprep.eda as dpeda
from autoviz.AutoViz_Class import AutoViz_Class


class FeatureSelection:
    """

    """

    def __init__(self, train_dataset: pd.DataFrame, target_col_name: str, threshold: float):
        self.train_dataset = train_dataset
        self.target_col_name = target_col_name
        self.threshold = threshold

    def corr(self):
        return self.train_dataset.corr()

    def get_highly_corr_cols(self):
        return self.train_dataset.columns[self.corr()[self.target_col_name].abs() > self.threshold]


class Normalization:
    """

    """

    def __init__(self, train_dataset: pd.DataFrame):
        self.train_dataset = train_dataset

    def standard_scaler(self):
        scale = StandardScaler()
        dataframe = pd.DataFrame(scale.fit_transform(self.train_dataset), columns=self.train_dataset.columns)
        return dataframe

    def min_max_scaler(self):
        min_max = MinMaxScaler()
        dataframe = pd.DataFrame(min_max.fit_transform(self.train_dataset), columns=self.train_dataset.columns)
        return dataframe

    def row_scaler(self):
        normalize = Normalizer()
        dataframe = pd.DataFrame(normalize.fit_transform(self.train_dataset))
        dataframe = np.sqrt((dataframe ** 2).sum(axis=1)).mean()
        return dataframe


class EDA:
    """

    """

    def __init__(self, dataset: pd.DataFrame, compare_cols: list = None, filename: str = None):
        self.dataset = dataset
        self.compare_cols = compare_cols
        self.filename = filename

    def corr(self):
        feature_corr = self.dataset.corr()
        sns.heatmap(feature_corr, annot=True)
        plt.savefig("./graph/heatmap.png")
        plt.close()

    def barplot(self):
        if self.compare_cols is None or len(self.compare_cols) > 2:
            return
        sns.barplot(self.compare_cols[0], self.compare_cols[1], data=self.dataset)
        plt.savefig("./graph/barplot.png")
        plt.close()

    def pandas_profiling(self):
        report = ProfileReport(self.dataset, title="Pandas Profiling", minimal=True)
        report = ProfileReport(self.dataset, title="Pandas Profiling")
        report.to_file("./eda/Pandas-Profiling.html")

    def sweetviz(self):
        report = sv.analyze(self.dataset)
        report.show_html(filepath='./eda/Sweetviz-Profiling.html', open_browser=False)
        # compare_report = sv.compare([self.dataset, "Train"], [self.test_dataset, "Test"], "Survived")
        # compare_report.show_html("compare_report.html")

    def dataprep(self):
        report = dpeda.create_report(self.dataset, title='Sataprep Profiling')
        report.save('./eda/Dataprep-Profiling')

    def autoviz(self):
        AV = AutoViz_Class()
        df_av = AV.AutoViz(self.dataset, chart_format="svg", save_plot_dir="./eda", lowess=True, verbose=2)
        df_av = AV.AutoViz('Autoviz-Profiling.csv', chart_format="svg", save_plot_dir="./eda")


class TimeDomainAnalysis:
    """
    Define some statistical functions (mean, max, etc.) and
    average the data for each second to get the trend by sliding window

    Parameters
    ----------
    dataset: Dataframe
        data
    window: int
        number of windows (sliding window), the default is 100
    time_col: str
        The column name of the timestamp, the default is Timestamp

    Returns
    -------
    None or Dataframe
        The result after time domain analysis if calls time_domain function
    """

    def __init__(self, dataset: pd.DataFrame, window: int = 100, time_col: str = "Timestamp"):
        self.dataset = dataset
        self.window = window
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
    Tuple
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
    Tuple
        - f_values (ndarray): Index of fast fourier transform
        - fft_values (float): Fast fourier transform signal data
    """

    f_values = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    fft_values_ = fft(y_values)
    fft_values = 2.0 / N * np.abs(fft_values_[0: N // 2])

    if plot:
        # If no path is specified, a new log folder will be created in the current path,
        # and the log file will be stored in it
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
    Tuple
        - f_values (ndarray): Index of fast fourier transform
        - psd_values (ndarray): Power spectral density signal data
    """

    f_values, psd_values = welch(y_values, fs=f_s)

    if plot:
        # If no path is specified, a new log folder will be created in the current path,
        # and the log file will be stored in it
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
    result = result[len(result) // 2:]
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
    Tuple
        - x_values (ndarray): Index of the sequence
        - autocorr_values (ndarray): The value of the sequence autocorrelation
    """

    autocorr_values = autocorr(y_values)
    x_values = np.array([T * jj for jj in range(0, N)])

    if plot:
        # If no path is specified, a new log folder will be created in the current path,
        # and the log file will be stored in it
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
    Tuple
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


def extract_features_labels(dataset: ndarray, labels: ndarray, T: float, N: int, f_s: int, denominator: int = 10) \
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
        Calculate the max peak height, the default is 10

    Returns
    -------
    Tuple
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

            features += get_features(*get_psd_values(signal, f_s), mph)
            features += get_features(*get_fft_values(signal, T, N), mph)
            features += get_features(*get_autocorr_values(signal, T, N), mph)
        list_of_features.append(features)
    return np.array(list_of_features), np.array(list_of_labels)
