<<<<<<< HEAD
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












=======

import pandas
print(pandas.__version__)
>>>>>>> 4e39e862cf14a9d6b1302290541b6e6979bdf6e7

