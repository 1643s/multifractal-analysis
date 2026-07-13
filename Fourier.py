from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt

def fourier(series, i, filename):
    x = series[:, 1]  # 対象の時系列(1列目)を明示的に取り出す
    N = len(x)
    dt = 1.0  # サンプリング間隔
    x_mean = np.mean(x)

    # パワースペクトル = 振幅の2乗
    yf = fft(x - x_mean)       # DC成分(平均)を除去してからFFT
    power = np.abs(yf[:N // 2])**2 / N  # パワースペクトル密度として正規化
    xf = fftfreq(N, dt)[:N // 2]

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.loglog(xf[1:], power[1:], linewidth=0.8, color='darkorange')  # f=0を除外
    title = f"{filename} - track {i} — Power Spectrum" if i is not None else "Power Spectrum"
    ax.set_title(title)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Power Spectral Density")
    ax.grid(True, alpha=0.3, which='both')  # log-logなのでminor gridも表示
    plt.tight_layout()
    plt.show()

    # ---- ここから: 位相ランダム化サロゲートの生成 ----
    amplitude = np.abs(yf)  # 元のパワースペクトルの振幅はそのまま保持

    # ランダム位相(0〜2π)を生成
    random_phase = np.random.uniform(0, 2 * np.pi, N)

    # 実数信号に戻すためエルミート対称性を課す
    # (yf[k] と yf[N-k] が複素共役になるように位相を設定)
    random_phase[0] = 0.0  # 直流成分の位相は0
    if N % 2 == 0:
        random_phase[N // 2] = 0.0  # ナイキスト成分も実数(位相0)
        random_phase[N // 2 + 1:] = -random_phase[1:N // 2][::-1]
    else:
        random_phase[(N + 1) // 2:] = -random_phase[1:(N + 1) // 2][::-1]

    surrogate_fft = amplitude * np.exp(1j * random_phase)

    # 逆フーリエ変換
    f = np.fft.ifft(surrogate_fft)

    # 実部の値のみ取り出し(理論上は虚部はほぼ0だが数値誤差が残る)
    f = f.real + x_mean  # 除いておいた平均を戻す

    return f