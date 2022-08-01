import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'Timestamp': ['2022-08-01 16:00:00', '2022-08-01 16:00:00', '2022-08-01 16:00:00', '2022-08-01 16:00:00',
                  '2022-08-01 16:00:00', '2022-08-01 16:00:01', '2022-08-01 16:00:01', '2022-08-01 16:00:01',
                  '2022-08-01 16:00:02', '2022-08-01 16:00:02', '2022-08-01 16:00:03', '2022-08-01 16:00:03',
                  '2022-08-01 16:00:04', '2022-08-01 16:00:04', '2022-08-01 16:00:05', '2022-08-01 16:00:05',
                  ],
    'Values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
})
fs = ["mean", "max", "min"]
df = df.groupby(["Timestamp"]).agg(fs)
df.columns = ["{}_{}".format(col[0], col[1]) for col in df.columns]
df_ = df.rolling(2).mean()
df_.columns = ["{}_rolling_avg".format(col) for col in df_.columns]
df = pd.concat([df, df_], axis=1)
df = df.fillna(method="bfill")
df.insert(0, "Timestamp", df.index)
df = df.reset_index(drop=True)
print(df)
exit(0)

sns.lineplot(x='Timestamp',
             y='Values_mean_rolling_avg',
             data=df,
             label='Values_rolling_avg')

sns.lineplot(x='Timestamp',
             y='Values_mean',
             data=df,
             label='Values_mean')

plt.xlabel('XLABEL')

# setting customized ticklabels for x axis
pos = ['2022-08-01 16:00:00', '2022-08-01 16:00:01', '2022-08-01 16:00:02', '2022-08-01 16:00:03',
       '2022-08-01 16:00:04', '2022-08-01 16:00:05']

lab = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']

plt.xticks(pos, lab)

plt.ylabel('YLABEL')

plt.show()
