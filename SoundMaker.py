from scipy.io import wavfile
import numpy as np
import pandas as pd

def get_nearest_piano_key_frequency(input_freq):
    A4_freq = 440.0
    if input_freq <= 0:
        # Handle non-positive frequencies by returning a very low, audible frequency or 0
        return 27.5 # A0

    # Calculate the number of semitones from A4
    n_float = 12 * np.log2(input_freq / A4_freq)

    # Round to the nearest whole semitone
    n_nearest_int = np.round(n_float)

    # Calculate the frequency of the nearest piano key
    nearest_freq = A4_freq * (2**(n_nearest_int / 12))

    return nearest_freq

def soundmake (series, csvfolder, filename2, type, i):
    #pandasに変換
    if isinstance(series, np.ndarray):
        df_series = pd.DataFrame(series, columns=['time', 'freq'])
    else:
        df_series = series

    # --- Configuration --- #
    enable_frequency_scaling = False  # Set to True to enable frequency scaling, or False to disable it
    snap_to_piano_keys = False      # Set to True to snap frequencies to the nearest piano key frequency
    tempo_factor = 2          # Increase this value to make the audio play faster

    # Apply snapping if enabled
    if snap_to_piano_keys:
        df_series['freq'] = df_series['freq'].apply(get_nearest_piano_key_frequency)

    fs = 22050  # Sample rate
    audio_segments = []
    phase = 0.0

    # 現在の周波数範囲を取得
    min_freq_orig = df_series['freq'].min()
    max_freq_orig = df_series['freq'].max()

    # 目標の周波数範囲
    min_freq_target = 200.0
    max_freq_target = 880.0

    # Iterate through the rows to create audio segments
    for j in range(len(df_series)):
        time_start = df_series.loc[j, 'time']
        freq_orig = df_series.loc[j, 'freq']

        # Apply frequency scaling if enabled
        if enable_frequency_scaling:
            if (max_freq_orig - min_freq_orig) > 0:
                freq = (freq_orig - min_freq_orig) / (max_freq_orig - min_freq_orig) * (max_freq_target - min_freq_target) + min_freq_target
            else:
                freq = min_freq_target # 元の周波数範囲が0の場合、最小ターゲット周波数に設定
        else:
            freq = freq_orig # Use original frequency if scaling is disabled

        # Determine duration for the current note/segment
        # If it's not the last element, use the difference to the next time point
        if j < len(df_series) - 1:
            time_end = df_series.loc[j+1, 'time']
            duration = (time_end - time_start) / 4 / tempo_factor # Apply tempo factor here
        else:
            # For the last note, assume a default duration or use the previous duration
            if len(df_series) > 1:
                duration = (df_series.loc[i, 'time'] - df_series.loc[i-1, 'time']) / 4 / tempo_factor # Apply tempo factor here
            else:
                duration = 1.0 / tempo_factor # Default for a single note, apply tempo factor

        # Generate a sine wave for the current frequency and duration
        # Ensure duration is positive to avoid issues with linspace
        if duration > 0:
            t = np.linspace(0, duration, int(fs * duration), endpoint=False)

            # 前の音の続きの位相から開始
            tone = np.sin(2 * np.pi * freq * t + phase)

            # 次の音のために位相を更新
            phase += 2 * np.pi * freq * duration
            phase = np.mod(phase, 2 * np.pi)

            audio_segments.append(tone)

    # Concatenate all generated audio segments
    if audio_segments:
        audio = np.concatenate(audio_segments)
        # Normalize audio to avoid clipping
        if np.max(np.abs(audio)) > 0:
            audio /= np.max(np.abs(audio))
        else:
            audio = np.zeros_like(audio) # Handle case of all zero audio
    else:
        audio = np.array([]) # Empty array if no segments were created

    # Write the WAV file
    wavfile.write(csvfolder + "/" + filename2 + "-" + str(i) + "-" + type + ".wav", fs, (audio * 32767).astype(np.int16))