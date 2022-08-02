import sys
from loguru import logger
from datetime import datetime
import configparser


def read_config_file(filename: str = "config.ini") -> configparser:
    """
    Read configuration file

    Parameters
    ----------
    filename: str
        Configuration file name, the default is config.ini

    Returns
    -------
    Configparser
        The result of reading the configuration file
    """

    logger.info("Reading configuration file ...")

    try:
        config = configparser.ConfigParser()
        config.read("./config/{}".format(filename))

        return config
    except Exception as e:
        logger.error(e)
        sys.exit(0)


def set_logger():
    """
    Configure the log file

    Parameters
    ----------

    Returns
    -------
    None
    """

    sink = datetime.now().strftime("%Y-%m-%d")
    sink = sink + ".log"

    '''
    Add a log file
    '''
    try:
        logger.add(sink="./log/{}".format(sink), rotation="500MB", encoding="utf-8", enqueue=True, retention="1 day")
    except Exception as e:
        logger.error(e)
        sys.exit(0)

