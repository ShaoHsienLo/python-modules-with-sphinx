from datatransformation import EDA
import os
import pandas as pd


path = r"C:\Users\samuello\Downloads\III\旺欉\code\labeled-data"
df = pd.read_csv(os.path.join(path, "1st-2nd-labeled-data.csv"))
eda = EDA(df)

eda.pandas_profiling()
eda.dataprep()
eda.sweetviz()
# eda.autoviz()













