from MIDICSV import midicsv
import numpy as np
from SoundMaker import soundmake
from h_q import h_q
import os
from Event import Event
import random
import pandas as pd
from scipy.fft import fft, fftfreq
from Fourier import fourier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

midipath =  "/Users/shumiyajima/EEPYTHON/MIDI/midi/BACH_SINFONIEN/2/2-3.mid"
csvfolder =  "/Users/shumiyajima/EEPYTHON/MIDI/midi/BACH_SINFONIEN/2"
filename = "Bach_Sinfonien2_3"
filename2 = "Bach-Sinfonien2-3"
logpath = "/Users/shumiyajima/EEPYTHON/MIDI/midi/log.csv"

#記録リストチェック
if os.path.exists(logpath):
    all_params_log = pd.read_csv(
        logpath,
        header=None
    ).values.tolist()
else:
    all_params_log = []

#時系列作成
d = {}
d = midicsv (midipath, csvfolder, filename, filename2)

#時系列選定
selected = np.array([int(x) for x in input("Which Part Do You Want To Analyze?").split(",")])

#1普通
c = {}

for i in selected:
    soundmake(series = d[i], csvfolder = csvfolder, filename2 = filename2, i=i, type = "normal")

    #1音源作成
    soundmake(series = d[i], csvfolder = csvfolder, filename2 = filename2, i=i, type = "normal")

    #1累積
    series = pd.DataFrame(data=d[i], columns=['time', 'freq'])
    series['freq'] = (series['freq'] - series['freq'].min()) / (series['freq'].max() - series['freq'].min())
    series['freq'] = series['freq'] - series['freq'].mean()
    series['freq'] = series['freq'].cumsum()
    series['time'] = series['time'].astype(float)
    time_index = series.set_index("time")
    #1MFDFA
    h_q(series = series, csvfolder = csvfolder, filename2 = filename2, i = i,type = "normal", log = all_params_log)

#2順番のみ
e = {}
for i in selected:
    #順番のみ 作成
    series = d[i]
    freq = Event(series[:, 1])
    x = np.linspace(1, len(freq), len(freq))
    c = np.column_stack((x, freq))  
    e[i] = c
    #2音源作成
    soundmake(series = e[i], csvfolder = csvfolder, filename2 = filename2, i=i, type = "event")
    #2累積
    series = pd.DataFrame(data=e[i], columns=['time', 'freq'])
    series['freq'] = (series['freq'] - series['freq'].min()) / (series['freq'].max() - series['freq'].min())
    series['freq'] = series['freq'] - series['freq'].mean()
    series['freq'] = series['freq'].cumsum()
    series['time'] = series['time'].astype(float)
    time_index = series.set_index("time")

    #2MFDFA    
    h_q(series = series, csvfolder = csvfolder, filename2 = filename2, i = i,type = "event", log = all_params_log)

#3Random
f = {}
for i in selected:
    #3Random作成
    series = d[i]
    freq = series[:, 1].copy()
    np.random.shuffle(freq)   # in-place、戻り値なしで使う
    x = np.linspace(1, len(freq), len(freq))
    c = np.column_stack((x, freq))   # shape (N, 2)
    f[i] = c
    #3音源作成
    soundmake(series = f[i], csvfolder = csvfolder, filename2 = filename2, i=i, type = "random")
    #3累積
    series = pd.DataFrame(data=f[i], columns=['time', 'freq'])
    series['freq'] = (series['freq'] - series['freq'].min()) / (series['freq'].max() - series['freq'].min())
    series['freq'] = series['freq'] - series['freq'].mean()
    series['freq'] = series['freq'].cumsum()
    series['time'] = series['time'].astype(float)
    time_index = series.set_index("time")

    #3MFDFA
    h_q(series = series, csvfolder = csvfolder, filename2 = filename2, i = i,type = "random", log = all_params_log)

#移動平均削除
g = {}
for i in selected: 
    #移動平均削除生成
    series = d[i]
    x = series[:,0]
    window = 10  # 好きな窓幅
    kernel = np.ones(window) / window
    mov_avg = np.convolve(series[:, 1], kernel, mode='same')
    freq = series[:, 1] - mov_avg   
    c = np.column_stack((x, freq))   # shape (N, 2)
    g[i] = c
    #4音源作成
    soundmake(series = g[i], csvfolder = csvfolder, filename2 = filename2, i=i, type = "mov_avg")
    #4累積
    series = pd.DataFrame(data=g[i], columns=['time', 'freq'])
    series['freq'] = (series['freq'] - series['freq'].min()) / (series['freq'].max() - series['freq'].min())
    series['freq'] = series['freq'] - series['freq'].mean()
    series['freq'] = series['freq'].cumsum()
    series['time'] = series['time'].astype(float)
    time_index = series.set_index("time")

    #4MFDFA
    h_q(series = series, csvfolder = csvfolder, filename2 = filename2, i = i,type = "mov_avg", log = all_params_log)

#フーリエ→逆フーリエ変換
h = {}
for i in selected: 
    #フーリエ→逆フーリエ変換作成
    series = d[i]
    h[i] = np.column_stack((series[:, 0], fourier(series=series, i=i, filename=filename2)))    #5音源作成
    series = pd.DataFrame(data=h[i], columns=['time', 'freq'])

    soundmake(series = h[i], csvfolder = csvfolder, filename2 = filename2, i=i, type = "fourier")
    #5累積
    series = pd.DataFrame(data=h[i], columns=['time', 'freq'])
    series['freq'] = (series['freq'] - series['freq'].min()) / (series['freq'].max() - series['freq'].min())
    series['freq'] = series['freq'] - series['freq'].mean()
    series['freq'] = series['freq'].cumsum()
    series['time'] = series['time'].astype(float)
    time_index = series.set_index("time")

    #5MFDFA
    h_q(series = series, csvfolder = csvfolder, filename2 = filename2, i = i,type = "fourier", log = all_params_log)
   
    #保存
    pd.DataFrame(all_params_log).to_csv(

        logpath,

        index=False,

        header=False

    )