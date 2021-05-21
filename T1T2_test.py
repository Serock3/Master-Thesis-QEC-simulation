# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy
from qiskit.quantum_info import partial_trace
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import state_fidelity

# Our own files
from simulator_program.custom_noise_models import (thermal_relaxation_model,
                                                   thermal_relaxation_model_V2,
                                                   WACQT_target_times,
                                                   WACQT_demonstrated_times,
                                                   standard_times)
from simulator_program.data_analysis_tools import fidelity_from_scratch
from matplotlib import colors
default_colors = plt.get_cmap("tab10")

# % TEST encoding fidelities
# fid, circ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
#        reset=True, idle_noise=True, theta=np.pi, phi=0,
#        snapshot_type='dm', device='Nah', pauliop='ZZZZZ')
# print(fid)

T1 = 40e3


def monoExp(t, T, c, A):
    return (A-c) * np.exp(-t/T) + c


p0 = (T1, 0.4, 0.9)  # start with values near those we expect*
colors = plt.get_cmap("tab10")
# %% Test run
n_cycles = 12
n_shots = 1024*4

resolution = 4
T1_min = 30e3
T1_max = 200e3
T2_min = 30e3
T2_max = 200e3
T1_span = np.linspace(T1_min, T1_max, resolution)
T2_span = np.linspace(T2_min, T2_max, resolution)
fid_span = np.zeros((resolution, resolution, n_cycles+1))
time_span = np.zeros((resolution, resolution, n_cycles+1))
par_span = np.zeros((resolution, resolution, 3))

invalid_marker = float('inf')
for i in range(resolution):
    for j in range(resolution):
        if T2_span[j] > 2*T1_span[i]:
            print('invalid params T1=',T1_span[i], ' T2=', T2_span[j])
            fid_span[i][j] = invalid_marker
            time_span[i][j] = invalid_marker
            par_span[i][j] = (invalid_marker,invalid_marker,invalid_marker)
            continue
        

        fid_span[i][j], time = fidelity_from_scratch(
            n_cycles, n_shots, T1=T1_span[i], T2=T2_span[j], encoding=False, transpile=False)
        time_span[i][j] = np.array([time['dm_'+str(i)]
                                    for i in range(n_cycles+1)])
        pars, cov = scipy.optimize.curve_fit(monoExp, time_span[i][j][1:], fid_span[i][j][1:], p0)
        par_span[i][j] = pars
        plt.plot(time_span[i][j]/1000, fid_span[i][j], 'o')

        print('T1= ', T1_span[i])
        print('T2= ', T2_span[j])
        print('T=', np.round(pars[0]/1000, 1))
        print('c=', np.round(pars[1], 2))
        print('A=', np.round(pars[2], 2), '\n')
        plt.plot(time_span[i][j]/1000, monoExp(time_span[i][j], *pars), '--')
        plt.title(r'$T_1$ = ' + str(int(T1_span[i]/1000))+r' $T_2$ = ' + str(int(T2_span[j]/1000))+' \n-> T='+str(
            np.round(pars[0]/1000, 1))+' c='+str(np.round(pars[1], 2))+' A='+str(np.round(pars[2], 2)))
        plt.show()

# fid_min = np.min(fid_span[fid_span != invalid_marker])
# fid_max = np.max(fid_span[fid_span != invalid_marker])
fid_min = np.min(par_span[:][:][0][par_span[:][:][0] != invalid_marker])
fid_max = np.max(par_span[:][:][0][par_span[:][:][0] != invalid_marker])
# %%

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
norm = colors.Normalize(vmin=fid_min, vmax=fid_max)
T1_padding = (T1_max-T1_min)/(resolution*1000*2)
T2_padding = (T2_max-T2_min)/(resolution*1000*2)
aspect = (T1_max-T1_min)/(T2_max-T2_min)
HM = ax.imshow(par_span[:][:][0], extent=[T1_min/1000-T1_padding, T1_max/1000+T1_padding, T2_min /
                                   1000-T2_padding, T2_max/1000+T2_padding], aspect=aspect, cmap='magma', origin='lower')

HM.set_norm(norm)
cbar0 = fig.colorbar(HM, ax=ax, orientation='horizontal',
                     fraction=.1, pad=0.17)
cbar0.set_label(r'Encoding success probability',
                labelpad=0, y=1.20, rotation=0)

ax.set_xticks(np.round(T1_span[0:resolution:2]/1000))
ax.set_yticks(np.round(T2_span[0:resolution:2]/1000))
ax.set_xlabel(r'$T_1$ [$\mu s$]')
ax.set_ylabel(r'$T_2$ [$\mu s$]')
#%%
for i, (fid, time) in enumerate(zip(fids, times)):
    pars, cov = scipy.optimize.curve_fit(monoExp, time[1:], fid[1:], p0)
    print('t_f= ', t_fs[i])
    print('T=', np.round(pars[0]/1000, 1))
    print('c=', np.round(pars[1], 2))
    print('A=', np.round(pars[2], 2), '\n')

    plt.plot(time/1000, fid, 'o', color=colors(i), label=r'$t_f$ = ' + str(int(t_fs[i]))+' -> T='+str(
        np.round(pars[0]/1000, 1))+' c='+str(np.round(pars[1], 2))+' A='+str(np.round(pars[2], 2)))
    plt.plot(time/1000, monoExp(time, *pars), ':', color=colors(i))

plt.title(r'Curvefitting to $(A-c)e^{-t/T}+c$ for different feedback times')
plt.xlabel('Time [μs]')
plt.ylabel('Probability of remaining in initial state')
plt.legend()
plt.show()

# %%
