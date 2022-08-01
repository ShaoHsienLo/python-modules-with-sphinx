import os
import pandas as pd
import sys
from loguru import logger
import json
from tqdm import tqdm


def convert_input_files_to_json_format(read_path: str, write_path: str) -> None:
    """
    Convert the input file to json format. The format of this input file must be

        {},

        {},

        ...

        {}

    Parameters
    ----------
    read_path : str
        Absolute path to the input file
    write_path: str
        Absolute path to the write file, it cannot be the same as the read path

    Returns
    -------
    None
    """

    '''
    Check if it is an existing absolute path
    '''
    # if not (os.path.exists(read_path) and os.path.exists(write_path) and
    #         os.path.abspath(read_path) and os.path.abspath(write_path)):
    #     logger.error("File path is not an absolute path.")
    #     sys.exit(0)

    '''
    Check if read and write paths are the same
    '''
    # if read_path == write_path:
    #     logger.error("The write path cannot be the same as the read path.")
    #     sys.exit(0)

    '''
    Check for files.
    '''
    # if len(files) < 1:
    #     logger.error("There are no files in the folder at this path.")
    #     sys.exit(0)

    files = os.listdir(read_path)

    logger.info("Converting {} input files to json format ...".format(len(files)))

    '''
    Check whether the file format is consistent, and re-save it into the file after reformat.
    '''
    for file in files:
        with open(os.path.join(read_path, file), "r") as rf:
            write_filename = file[:-5] + ".json"
            with open(os.path.join(write_path, write_filename), "w") as wf:
                lines = rf.readlines()
                for line in lines:
                    try:
                        line = line[:-2]
                        if json.loads(line):
                            wf.write(line)
                            wf.write("\n")
                    except Exception as e:
                        logger.error(e)
                        sys.exit(0)


def merge_json_files_to_csv_file(read_path: str, write_path: str = None, ignore_cols: list = None,
                                 filename: str = None, ignore_err: bool = True) -> None:
    """
    Merge json files into a csv file

    Parameters
    ----------
    read_path : str
        Absolute path to the input file
    write_path: str
        Absolute path to the write file, it cannot be the same as the read path
    ignore_cols: list
        Columns to be ignored, the default is none
    filename: str
        Save file name, the default is none
    ignore_err: bool
        Whether to ignore errors for merged files, the default is true

    Returns
    -------
    None
    """

    files = os.listdir(read_path)
    df = pd.DataFrame()

    logger.info("Merging {} json files to a csv file ...".format(len(files)))

    '''
    Add a processing bar and merge json files
    '''
    for i in tqdm(range(len(files))):
        try:
            data = pd.read_json(os.path.join(read_path, files[i]), lines=True)
            data = data.drop(columns=ignore_cols)
            if not df.empty:
                df = pd.concat([df, data], axis=0)
            else:
                df = data
        except Exception as e:
            if ignore_err:
                logger.warning("Failed to merge file: {}".format(files[i]))
            else:
                logger.error(e)
                sys.exit(0)

    '''
    Output csv file
    '''
    df = df.reset_index(drop=True)
    if write_path is None:
        write_path = os.path.abspath(os.getcwd())
    df.to_csv(os.path.join(write_path, filename), index=False)
