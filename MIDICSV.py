import py_midicsv as pm
import pandas as pd
import numpy as np
from typing_extensions import NewType
from scipy import interpolate
import matplotlib.pyplot as plt
from SoundMaker import soundmake
def midicsv (midipath, csvfolder, filename, filename2):
    # Load the MIDI file and parse it into CSV format
    csv_string = pm.midi_to_csv(midipath)
    output_csv_filepath = csvfolder + '/' + filename + ".csv"
    with open(output_csv_filepath, "w") as f:
        f.writelines(csv_string)
    # Load csv file
    df = pd.read_csv( output_csv_filepath , header=None,
                 names=['Track', 'Time', 'Event', 'Channel', 'Note', 'Velocity'], # データの名前を付与
                 usecols=range(6), # 読み込む列の数を指定
                 engine='python') # 不均一なデータ構造への対応
    PPQ=int(df.iloc[0, 5])/4
    #　Channel events のみを抽出
    df_notes = df[(df['Event'].str.strip() == 'Note_on_c') | (df['Event'].str.strip() == 'Note_off_c')]
    d = {}
    tracks = sorted(df_notes['Track'].unique())
    for i in tracks:
      track_data = df_notes[df_notes['Track']==i]
      if track_data.empty:
        print("Track" + str(i) + "にはデータが存在しません")
        continue
      # トラックiの音だけを抽出
      notes = track_data['Note'].values.astype(int)
      # 周波数に変換
      freqs_old = np.ravel(440 * (2 ** ((notes - 69) / 12)))
      # トラックiの時間だけを抽出
      time = track_data['Time'].values.astype(int)
      times = np.ravel((time/PPQ)*2)
      # event だけ抽出
      events_old = track_data['Event'].values.astype(str)

      events_numeric = np.array([1 if s.strip() == 'Note_on_c' else 0 for s in events_old], dtype=int)
      # 時系列
      #データ補完
      f = interpolate.interp1d(times, freqs_old, "previous")
      times_new = np.arange(np.amin(times),np.amax(times),1)
      freqs = f(times_new)

      #データ補完2
      f = interpolate.interp1d(times, events_numeric, "previous")
      events_new = np.arange(np.amin(times),np.amax(times),1)
      events_new = f(events_new)

      #0-1に変換
      # timesとfreqsを列として結合するために、それぞれを2D配列に reshaping
      print(f"times_new shape: {times_new.shape}")
      print(f"freqs shape: {freqs.shape}")
      print(f"events_new shape: {events_new.shape}")
      series_old = np.concatenate([times_new[:, np.newaxis], freqs[:, np.newaxis], events_new[:, np.newaxis]], axis=1)
      print(f"series_old shape: {series_old.shape}")
      series_old2 = series_old[~(series_old[:, 2] == 0)]
      series = np.delete(series_old2, 2, 1)
      fig, ax = plt.subplots(figsize=(3, 2)) # You can adjust figsize as needed
      ax.plot(series[:,0], series[:,1], linewidth=0.8, color='steelblue')
      ax.set_title(f"Track {i} — Time-Freq Line Chart") # Make title track-specific
      ax.set_xlabel("Time")
      ax.set_ylabel("Freq")
      ax.grid(True, alpha=0.3)
      plt.tight_layout() # Adjust layout to prevent labels overlapping
      fig.savefig(csvfolder + "/" + filename2 + "-" + str(i)+ ".png", dpi=150, bbox_inches='tight')  # ← 先に保存
      plt.show() # Display the plot
      np.savetxt(csvfolder + "/" + filename2 + "-" + str(i)+ ".csv", series, delimiter=",")
      d[i] = series
    return d

