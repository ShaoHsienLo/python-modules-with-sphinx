import os
from datastream import read_file

path = r"C:\Users\samuello\Downloads\III\旺欉\code"
file = "Jul-4-data.csv"
df = read_file(os.path.join(path, file))
print(df.head(10))

