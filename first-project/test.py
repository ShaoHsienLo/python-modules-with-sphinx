from pythonprojects.pythonmodules.src.settings import read_config_file, set_logger
# from pythonprojects.pythonmodules.src.datastream import Connector
from pythonprojects.pythonmodules.src.datatransformation import TimeDomainAnalysis, extract_features_labels, \
    get_features, get_fft_values, get_values, get_psd_values, get_autocorr_values, get_first_n_peaks
from pythonprojects.pythonmodules.src.preprocessing import convert_input_files_to_json_format, \
    merge_json_files_to_csv_file
from pythonprojects.pythonmodules.src.decorators import deprecated
from pythonprojects.pythonmodules.src.detect_peaks import detect_peaks

# set_logger()
#
# config = read_config_file()
#
# print(config["http"]["host"])

with open("./log/2022-08-02.log", "r") as f:
    print(f.readlines())


