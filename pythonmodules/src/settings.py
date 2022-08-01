from loguru import logger
from datetime import datetime
import os
import json


def read_config_file(path: str = None, filename: str = "config.json") -> dict:
    """
    Read configuration file

    Parameters
    ----------
    path: str
        Absolute path to the configuration file
    filename: str
        Configuration file name, the default is `config.json`

    Returns
    -------
    Dict
        The result of reading the configuration file
    """

    if path is None:
        path = os.path.abspath(os.getcwd())

    logger.info("Reading configuration file ...")

    with open(os.path.join(path, filename), "r") as f:
        dict_data = json.load(f)

    return dict_data


def set_logger(path: str = None):
    """
    Configure the log file

    Parameters
    ----------
    path: str
        The location where the log files are stored, the default is None

    Returns
    -------
    None
    """

    sink = datetime.now().strftime("%Y-%m-%d")
    sink = sink + ".log"
    directory = "log"

    '''
    If no path is specified, a new log folder will be created in the current path, 
    and the log file will be stored in it
    '''
    if path is None:
        path = os.path.join(os.path.abspath(os.getcwd()), directory)
    else:
        path = os.path.join(path, directory)
    if not os.path.exists(path):
        os.makedirs(path)

    '''
    Add a log file
    '''
    logger.add(sink=os.path.join(path, sink), rotation="500MB", encoding="utf-8", enqueue=True, retention="1 day")
