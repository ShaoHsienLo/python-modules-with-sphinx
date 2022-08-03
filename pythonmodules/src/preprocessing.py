import os
import pandas as pd
import sys
from loguru import logger
import json
from tqdm import tqdm


def convert_input_files_to_json_format(ignore_err: bool = True) -> None:
    """
    Reformat the input file to json. The original format of this input file must be

        {},

        {},

        ...

        {}

    Parameters
    ----------
    ignore_err: bool
        Whether to ignore errors for merged files, the default is true

    Returns
    -------
    None
    """

    files = os.listdir("./data/original-data")

    logger.info("Converting {} input files to json format ...".format(len(files)))

    # Check whether the file format is consistent, and re-save it into the file after reformat.
    for file in files:
        with open("./data/original-data/{}".format(file), "r") as rf:
            with open("./data/processed-data/{}".format(file), "w") as wf:
                lines = rf.readlines()
                for line in lines:
                    try:
                        line = line[:-2]
                        if json.loads(line):
                            wf.write(line)
                            wf.write("\n")
                    except Exception as e:
                        if not ignore_err:
                            logger.error(e)
                            sys.exit(0)


def merge_json_files_to_csv_file(ignore_cols: list = None, filename: str = "data.csv") -> None:
    """
    Merge json files into a csv file

    Parameters
    ----------
    ignore_cols: list
        Columns to be ignored, the default is none
    filename: str
        Save file name, the default is data.csv

    Returns
    -------
    None
    """

    if ignore_cols is None:
        ignore_cols = []

    files = os.listdir("./data/processed-data")
    df = None

    logger.info("Merging {} json files to a csv file ...".format(len(files)))

    # Add a processing bar and merge json files
    for i in tqdm(range(len(files))):
        try:
            data = pd.read_json("./data/processed-data/{}".format(files[i]), lines=True)
            data = data.drop(columns=ignore_cols)
            if df is None:
                df = data
            else:
                df = pd.concat([df, data], axis=0)
        except Exception as e:
            logger.error(e)
            sys.exit(0)

    # Output csv file
    try:
        df = df.reset_index(drop=True)
        df.to_csv("./data/model-input-data/{}".format(filename), index=False)
    except Exception as e:
        logger.error(e)
        sys.exit(0)
