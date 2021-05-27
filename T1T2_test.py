# %%
import time
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
from simulator_program.data_analysis_tools import (fidelity_from_scratch,
                                                   perfect_stab_circuit)
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

# For the projected qubit we set A=1 and c=0.5
def idealExp(t,T):
    return 0.5 * np.exp(-t/T) + 0.5

p0 = (T1, 0.4, 0.9)  # start with values near those we expect*
def_colors = plt.get_cmap("tab10")

n_cycles = 15
n_shots = 1024*8

resolution = 8
T1_min = 30e3
T1_max = 100e3
T2_min = 50e3
T2_max = 120e3
T1_span = np.linspace(T1_min, T1_max, resolution)
T2_span = np.linspace(T2_min, T2_max, resolution)
fid_span = np.zeros((resolution, resolution, n_cycles+1))
P_L_span = np.zeros((resolution, resolution, n_cycles+1))
time_span = np.zeros((resolution, resolution, n_cycles+1))
par_span = np.zeros((resolution, resolution, 3))
par_L_span = np.zeros((resolution, resolution, 3))
cov_span = np.zeros((resolution, resolution, 3,3))
cov_L_span = np.zeros((resolution, resolution, 3,3))

# Define variables to save/load
function_data = [n_cycles,
                 n_shots,
                 resolution,
                 T1_min,
                 T1_max,
                 T2_min,
                 T2_max,
                 T1_span,
                 T2_span,
                 fid_span,
                 P_L_span,
                 time_span,
                 par_span,
                 par_L_span,
                 cov_span,
                 cov_L_span]

invalid_marker = float('inf')
# %% Test run
num_data_points = 0
for i in range(resolution):
    for j in range(resolution):
        if not T2_span[j] > 2*T1_span[i]:
            num_data_points += 1
print('Will take roughly ',num_data_points*10, ' minutes to run')

for i in range(resolution):
    for j in range(resolution):
        if T2_span[j] > 2*T1_span[i]:
            print('invalid params T1=', T1_span[i], ' T2=', T2_span[j])
            fid_span[i][j] = invalid_marker
            time_span[i][j] = invalid_marker
            par_span[i][j] = (invalid_marker, invalid_marker, invalid_marker)
            par_L_span[i][j] = (invalid_marker, invalid_marker, invalid_marker)
            cov_span[i][j] = invalid_marker
            cov_L_span[i][j] = invalid_marker
            continue

        seconds = time.time()

        # Using perfect decoding instead
        #fid_span[i][j], P_L_span[i][j], times = perfect_stab_circuit(
        #    n_cycles, n_shots, T1=T1_span[i], T2=T2_span[j], project=True)
        
        fid_span[i][j], P_L_span[i][j], times = fidelity_from_scratch(
            n_cycles, n_shots, T1=T1_span[i], T2=T2_span[j], encoding=False, transpile=False, project=True)
        
        time_span[i][j] = np.array([times['dm_'+str(i)]
                                    for i in range(n_cycles+1)])
                
        pars, cov = scipy.optimize.curve_fit(
            monoExp, time_span[i][j][1:], (fid_span[i][j]*P_L_span[i][j])[1:], p0)
        pars_L, cov_L = scipy.optimize.curve_fit(
            idealExp, time_span[i][j][1:]-time_span[i][j][1], fid_span[i][j][1:], min(T1_span[i],T2_span[j]))
        
        par_span[i][j] = pars
        par_L_span[i][j] = pars_L
        cov_span[i][j] = cov
        cov_L_span[i][j] = cov_L

        if not (i%2==0 and j%2==0):
            continue
        plt.plot(time_span[i][j]/1000, fid_span[i][j] *
                 P_L_span[i][j], 'o', label='F', color=def_colors(0))
        plt.plot(time_span[i][j]/1000, fid_span[i][j],
                 '^', label='F_L', color=def_colors(2))
        plt.plot(time_span[i][j]/1000, P_L_span[i][j],
                 's', label='P_L', color=def_colors(3))

        plt.plot(time_span[i][j]/1000, monoExp(time_span[i]
                                               [j], *pars), '--', color=def_colors(0))
        plt.plot(time_span[i][j]/1000, idealExp(time_span[i][j]-time_span[i][j][1], *pars_L), '--', color=def_colors(2))
        plt.plot(time_span[i][j]/1000, monoExp(time_span[i][j], *pars) /
                 idealExp(time_span[i][j]-time_span[i][j][1], *pars_L), '--', color=def_colors(3))
        plt.title(r'$T_1$ = ' + str(int(T1_span[i]/1000))+r' $T_2$ = ' + str(int(T2_span[j]/1000))+' \n-> F: T='+str(
            np.round(pars[0]/1000, 1))+' c='+str(np.round(pars[1], 2))+' A='+str(np.round(pars[2], 2)) +
            '\n F_L: '+'T='+str(np.round(pars_L[0]/1000, 1)))
        plt.legend()
        plt.show()

        print('This took ', int(time.time()-seconds), 'seconds')
        print('Time left ', int((num_data_points-(i*resolution+j+1))*(time.time()-seconds)/60),' minutes')


#%% Save
with open('data/T1T2_test_data.npy', 'wb') as f:
    for data in function_data:
        np.save(f, data)
# %% Load
with open('data/T1T2_test_data.npy', 'rb') as f:
        n_cycles = np.load(f)
        n_shots = np.load(f)
        resolution = np.load(f)
        T1_min = np.load(f)
        T1_max = np.load(f)
        T2_min = np.load(f)
        T2_max = np.load(f)
        T1_span = np.load(f)
        T2_span = np.load(f)
        fid_span = np.load(f)
        P_L_span = np.load(f)
        time_span = np.load(f)
        par_span = np.load(f)
        par_L_span = np.load(f)
        cov_span = np.load(f)
        cov_L_span = np.load(f)
# %% T1 for F
fid_min = np.min(par_span[:, :, 0][par_span[:, :, 0] != invalid_marker])
fid_max = np.max(par_span[:, :, 0][par_span[:, :, 0] != invalid_marker])
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
norm = colors.Normalize(vmin=fid_min/1000, vmax=fid_max/1000)
# norm = colors.Normalize(vmin=15, vmax=150)
T1_padding = (T1_max-T1_min)/(resolution*1000*2)
T2_padding = (T2_max-T2_min)/(resolution*1000*2)
aspect = (T1_max-T1_min)/(T2_max-T2_min)
HM = ax.imshow(par_span[:, :, 0].T/1000, extent=[T1_min/1000-T1_padding, T1_max/1000+T1_padding, T2_min /
                                                 1000-T2_padding, T2_max/1000+T2_padding], aspect=aspect, cmap='magma', origin='lower')

HM.set_norm(norm)
cbar0 = fig.colorbar(HM, ax=ax, orientation='horizontal',
                     fraction=.1, pad=0.17)
cbar0.set_label(r'$T_L$ [$\mu s$]',
                labelpad=0, y=1.20, rotation=0)

ax.set_xticks(np.round(T1_span[0:resolution:2]/1000))
ax.set_yticks(np.round(T2_span[0:resolution:2]/1000))
ax.set_xlabel(r'$T_1$ [$\mu s$]')
ax.set_ylabel(r'$T_2$ [$\mu s$]')
# %% T1 for F_L
var_to_plot = par_L_span[:, :, 0]
fid_min = np.min(var_to_plot[var_to_plot != invalid_marker])
fid_max = np.max(var_to_plot[var_to_plot != invalid_marker])
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
norm = colors.Normalize(vmin=fid_min/1000, vmax=fid_max/1000)
# norm = colors.Normalize(vmin=15, vmax=150)
T1_padding = (T1_max-T1_min)/(resolution*1000*2)
T2_padding = (T2_max-T2_min)/(resolution*1000*2)
aspect = (T1_max-T1_min)/(T2_max-T2_min)
HM = ax.imshow(var_to_plot.T/1000, extent=[T1_min/1000-T1_padding, T1_max/1000+T1_padding, T2_min /
                                                 1000-T2_padding, T2_max/1000+T2_padding], aspect=aspect, cmap='magma', origin='lower')

HM.set_norm(norm)
cbar0 = fig.colorbar(HM, ax=ax, orientation='horizontal',
                     fraction=.1, pad=0.17)
cbar0.set_label(r'$T_L$ [$\mu s$]',
                labelpad=0, y=1.20, rotation=0)

ax.set_xticks(np.round(T1_span[0:resolution]/1000))
ax.set_yticks(np.round(T2_span[0:resolution]/1000))
ax.set_xlabel(r'$T_1$ [$\mu s$]')
ax.set_ylabel(r'$T_2$ [$\mu s$]')

# %% plot c
fid_min = np.min(par_span[:, :, 1][par_span[:, :, 1] != invalid_marker])
fid_max = np.max(par_span[:, :, 1][par_span[:, :, 1] != invalid_marker])
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
norm = colors.Normalize(vmin=fid_min, vmax=fid_max)
T1_padding = (T1_max-T1_min)/(resolution*1000*2)
T2_padding = (T2_max-T2_min)/(resolution*1000*2)
aspect = (T1_max-T1_min)/(T2_max-T2_min)
HM = ax.imshow(par_span[:, :, 1].T, extent=[T1_min/1000-T1_padding, T1_max/1000+T1_padding, T2_min /
                                            1000-T2_padding, T2_max/1000+T2_padding], aspect=aspect, cmap='magma', origin='lower')

HM.set_norm(norm)
cbar0 = fig.colorbar(HM, ax=ax, orientation='horizontal',
                     fraction=.1, pad=0.17)
cbar0.set_label(r'$c$ - the maximally noisy state prob',
                labelpad=0, y=1.20, rotation=0)

ax.set_xticks(np.round(T1_span[0:resolution:2]/1000))
ax.set_yticks(np.round(T2_span[0:resolution:2]/1000))
ax.set_xlabel(r'$T_1$ [$\mu s$]')
ax.set_ylabel(r'$T_2$ [$\mu s$]')
# %% plot any
var_to_plot = par_L_span[:, :, 0]
fid_min = np.min(var_to_plot[var_to_plot != invalid_marker])
fid_max = np.max(var_to_plot[var_to_plot != invalid_marker])
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
norm = colors.Normalize(vmin=fid_min, vmax=fid_max)
T1_padding = (T1_max-T1_min)/(resolution*1000*2)
T2_padding = (T2_max-T2_min)/(resolution*1000*2)
aspect = (T1_max-T1_min)/(T2_max-T2_min)
HM = ax.imshow(var_to_plot.T, extent=[T1_min/1000-T1_padding, T1_max/1000+T1_padding, T2_min /
                                            1000-T2_padding, T2_max/1000+T2_padding], aspect=aspect, cmap='magma', origin='lower')

HM.set_norm(norm)
cbar0 = fig.colorbar(HM, ax=ax, orientation='horizontal',
                     fraction=.1, pad=0.17)
cbar0.set_label(r'$c$ - the maximally noisy state prob',
                labelpad=0, y=1.20, rotation=0)

ax.set_xticks(np.round(T1_span[0:resolution:2]/1000))
ax.set_yticks(np.round(T2_span[0:resolution:2]/1000))
ax.set_xlabel(r'$T_1$ [$\mu s$]')
ax.set_ylabel(r'$T_2$ [$\mu s$]')
# %%
