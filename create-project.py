import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("projectname", type=str)
args = parser.parse_args()

project_name = args.projectname

paths = [
    "/config",
    "/data",
    "/data/model-input-data",
    "/data/original-data",
    "/data/processed-data",
    "/log"
]
paths = ["./" + project_name + path for path in paths]

files = [
    "/config/config.ini",
    "/main.py"
]
files = ["./" + project_name + file for file in files]

os.mkdir(project_name)
for path in paths:
    os.mkdir(path)
for file in files:
    f = open(file, "a+")


