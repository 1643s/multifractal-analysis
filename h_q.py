import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from scipy.optimize import curve_fit

def func (q, c, A, k, q_0) :
  h_q = c - A/2 * np.tanh(k*(q-q_0))
  return h_q

def h_q (series, filename2, csvfolder, type, i, log) :
  
  error_list = []
  h_q_results = []
  #t_q_results = []
  #alpha_q = []


  for q in range (-20,20) :
    local_max_time_val = int(series["time"].max()) + 1
    F_s_list = []

    scales = np.unique(np.logspace(np.log10(55), np.log10(200), 40).astype(int))

    for s in scales:
        current_scale_fluctuations = []
        #bin_edges_f = np.arange(-1, local_max_time_val + s, s)
        #bin_edges_b = np.sort(local_max_time_val - np.arange(0, local_max_time_val + s, s))

        #for edges, is_right in [(bin_edges_f, True), (bin_edges_b, False)]:
            #temp_df = series.copy()
            #temp_df['bin_label'] = pd.cut(series["time"], edges, right=is_right, labels=False)
        data_y = series.to_numpy()[:,1]
        N_len = len(data_y)
        Ns = N_len // s
        if Ns == 0: continue

        seg_f = data_y[: Ns*s].reshape(Ns, s)
        seg_b = data_y[N_len - Ns*s :].reshape(Ns, s)
        all_seg = np.vstack([seg_f, seg_b])
        t = np.arange(s)

        for y_seg in all_seg :
            m, c = np.polyfit(t, y_seg, 1)
            ms = np.mean((y_seg - (m * t + c))**2)
            if ms <= 0 : continue
            if q==0 :
              current_scale_fluctuations.append(math.log(ms))
            else :
              current_scale_fluctuations.append(ms**(q/2))

        if current_scale_fluctuations:
            if q==0 :
              F_s = math.e**(np.mean(current_scale_fluctuations)/2)
              if F_s <0 : continue
              F_s_list.append([s, F_s])
            else :
              F_s = (np.mean(current_scale_fluctuations))**(1/q)
              if F_s <0 : continue
              F_s_list.append([s, F_s])

    if F_s_list:
        data = np.array(F_s_list)
        log_s = np.log(data[:, 0])
        log_F = np.log(data[:, 1])
        A_reg = np.vstack([log_s, np.ones(len(log_s))]).T
        h_q_val, log_C = np.linalg.lstsq(A_reg, log_F, rcond=None)[0]
        y_remain = np.sum((log_F - (h_q_val * log_s + log_C))**2)
        sse = y_remain / (np.size(log_s)-2)
        x_remain = np.sum((log_s-np.average(log_s))**2)
        t_val = q * h_q_val - 1
        error = math.sqrt(sse/x_remain)
        if q == 0:
          #t_q_results.append([q, t_val])
          h_q_results.append([q, h_q_val])
          error_list.append(error)
        elif q == 1: # Handle q=1 case to avoid ZeroDivisionError
          #t_q_results.append([q, t_val])
          h_q_results.append([q, h_q_val])
          error_list.append(error)
        else:
          #t_q_results.append([q, local_max_time_val])
          d_q_val = q*h_q_val/(1-q)
          h_q_results.append([q, h_q_val, d_q_val, t_val])
          error_list.append(error)

        plt.figure(figsize=(6, 4))
        plt.scatter(data[:, 0], data[:, 1], label='Data')
        # 回帰直線の描画 (F = C * s^h)
        fit_F = np.exp(log_C) * (data[:, 0]**h_q_val)
        log_F_pred = log_C + h_q_val * log_s
        ss_res = np.sum((log_F - log_F_pred) ** 2)
        ss_tot = np.sum((log_F - np.mean(log_F)) ** 2)
        r2_fs = 1 - ss_res / ss_tot if ss_tot != 0 else 0

        plt.plot(data[:, 0], fit_F, color='red', label=f'Fit (h={h_q_val:.2f})')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('s')
        plt.ylabel('F(s)')
        plt.title(f's-F(s): q={q},  R²={r2_fs:.4f}')
        plt.legend()
        plt.grid(True, which="both", linestyle="--")
        plt.savefig(csvfolder + "/" + filename2 + f" part{i}_q={q} - {type}.png")
        plt.show()


  # Extract q values and h(q) values from the list of lists
  q_values = np.array([item[0] for item in h_q_results]).astype(float)
  h_q_values = np.array([item[1] for item in h_q_results]).astype(float)

# inf/NaNを除外
  valid = np.isfinite(h_q_values)
  if not np.all(valid):
    removed_q = q_values[~valid]
    print(f"警告: 以下のqでh(q)がinf/NaNのため除外しました: {removed_q}")

  q_values_clean = q_values[valid]
  h_q_values_clean = h_q_values[valid]

  popt, pcov = curve_fit(func, q_values_clean, h_q_values_clean, maxfev=10000)
  print('optimized parameters=', popt)
  print('variance-covariance matrix=', pcov)
  print('error=', np.sqrt(np.diag(pcov)))
  c, A, k, q_0 = popt
  q_0_val = func(0, *popt)   # q=0 での h(q) 値が欲しいなら
  dataset_label = f"{filename2}-{i}-{type}_call_{len(log)}"
  log.append([dataset_label] + list(popt) + [q_0_val])  
  x_pred = np.linspace(np.min(q_values), np.max(q_values), 1000)

  h_q_pred = func(q_values, *popt)
  ss_res_hq = np.sum((h_q_values - h_q_pred) ** 2)
  ss_tot_hq = np.sum((h_q_values - np.mean(h_q_values)) ** 2)
  r2_hq = 1 - ss_res_hq / ss_tot_hq if ss_tot_hq != 0 else 0
  
    # Plot the h(q) data and the fitted curve
  fig = plt.figure(figsize=(6, 4))
  plt.plot(q_values, h_q_values, 'o', label='h(q) Data') # Plot the actual data points
  plt.plot(x_pred, func(x_pred, *popt), '-', label='Fitted Curve') # Plot the fitted curve
  plt.errorbar(q_values, h_q_values, yerr = error_list,ecolor='black')


  plt.xlabel('q')
  plt.ylabel('h(q)')
  plt.text(0.05, 0.95, f"h(q) = {c:.2f} - {A:.2f}/2 · tanh({k:.2f}(q-{q_0:.2f}))",
          transform=plt.gca().transAxes, fontsize=8, va='top')
  plt.gca().set_ylim(0, 3.5)
  plt.title(f'q - h(q) with Fitted Curve  R²={r2_hq:.4f}')
  plt.legend() # Display the legend with the labels
  plt.grid(True)
  fig.savefig(csvfolder + "/" + "h(q) - " + filename2 + "-" + str(i)+ "-" + type +".png", dpi=150, bbox_inches='tight')  # ← 先に保存
  plt.show()
