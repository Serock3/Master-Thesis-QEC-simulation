# %%
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy
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

resolution = 11
T1_min = 30e3
T1_max = 130e3
T2_min = 50e3
T2_max = 150e3
T1_span = np.linspace(T1_min, T1_max, resolution)
T2_span = np.linspace(T2_min, T2_max, resolution)
fid_span = np.zeros((resolution, resolution, n_cycles+1))
P_L_span = np.zeros((resolution, resolution, n_cycles+1))
time_span = np.zeros((resolution, resolution, n_cycles+1))
par_span = np.zeros((resolution, resolution, 3))
par_L_span = np.zeros((resolution, resolution, 3))
cov_span = np.zeros((resolution, resolution, 3,3))
cov_L_span = np.zeros((resolution, resolution, 3,3))

invalid_marker = float('inf')

#%% Expand current values
T1_span_save = T1_span
T2_span_save = T2_span
fid_span_save = fid_span
P_L_span_save = P_L_span
time_span_save = time_span
par_span_save = par_span
par_L_span_save = par_L_span
cov_span_save = cov_span
cov_L_span_save = cov_L_span
old_res = fid_span_save.shape[0]
T1_span = np.linspace(T1_min, T1_max, resolution)
T2_span = np.linspace(T2_min, T2_max, resolution)
fid_span = np.zeros((resolution, resolution, n_cycles+1))
fid_span[:old_res,:old_res,:] = fid_span_save
P_L_span = np.zeros((resolution, resolution, n_cycles+1))
P_L_span[:old_res,:old_res,:] = P_L_span_save
time_span = np.zeros((resolution, resolution, n_cycles+1))
time_span[:old_res,:old_res,:] = time_span_save
par_span = np.zeros((resolution, resolution, 3))
par_span[:old_res,:old_res,:] = par_span_save
par_L_span = np.zeros((resolution, resolution, 3))
par_L_span[:old_res,:old_res,:] = par_L_span_save
cov_span = np.zeros((resolution, resolution, 3,3))
cov_span[:old_res,:old_res,:] = cov_span_save
cov_L_span = np.zeros((resolution, resolution, 3,3))
cov_L_span[:old_res,:old_res,:] = cov_L_span_save
# %% Test run sista färdiga värdet var T1=70 T2=100
num_data_points = 0
for i in range(resolution):
    for j in range(resolution):
        if not (T2_span[j] > 2*T1_span[i] or fid_span[i][j][0] != 0):
            num_data_points += 1
print('Will take roughly ',num_data_points*10, ' minutes to run')

num_data_points_completed = 0

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
        if fid_span[i][j][0] != 0:
            print(i,j)
            continue
        # Using perfect decoding instead
        # fid_span[i][j], P_L_span[i][j], times = perfect_stab_circuit(
        #    n_cycles, n_shots, T1=T1_span[i], T2=T2_span[j], project=True)
        
        fid_span[i][j], P_L_span[i][j], times = fidelity_from_scratch(
            n_cycles, n_shots, gate_times={'delay': 5000*(T1_span[i]+T2_span[j])/100e3}, T1=T1_span[i], T2=T2_span[j], encoding=False, transpile=False, project=True)
        
        time_span[i][j] = np.array([times['dm_'+str(i)]
                                    for i in range(n_cycles+1)])
                
        pars, cov = scipy.optimize.curve_fit(
            monoExp, time_span[i][j][1:], (fid_span[i][j]*P_L_span[i][j])[1:], p0)
        # Normal decoing
        pars_L, cov_L = scipy.optimize.curve_fit(
            idealExp, time_span[i][j][1:]-time_span[i][j][1], fid_span[i][j][1:], min(T1_span[i],T2_span[j]))
        # # Perfect decoding
        # pars_L, cov_L = scipy.optimize.curve_fit(
        #     idealExp, time_span[i][j][:], fid_span[i][j][:], min(T1_span[i],T2_span[j]))

        par_span[i][j] = pars
        par_L_span[i][j] = pars_L
        cov_span[i][j] = cov
        cov_L_span[i][j] = cov_L

        # if not (i%2==0 and j%2==0):
        #     continue
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
        num_data_points_completed += 1
        print('Time left ', int((num_data_points-num_data_points_completed)*(time.time()-seconds)/60),' minutes')

        #% Save
        with open('data/T1T2_test_data_delay.npy', 'wb') as f:
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
            for data in function_data:
                np.save(f, data)
# %% Load
with open('data/T1T2_test_data_delay.npy', 'rb') as f:
        _ = np.load(f)
        _ = np.load(f)
        _ = np.load(f)
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
#%% Plot P_L
var_to_plot = np.mean(P_L_span[:,:,1:], axis = 2)
invalid_marker_tmp = 0
var_to_plot[var_to_plot == invalid_marker_tmp] = invalid_marker
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
cbar0.set_label(r'$P_L$ - overlap with codespace',
                labelpad=0, y=1.20, rotation=0)

ax.set_xticks(np.round(T1_span[0:resolution:2]/1000))
ax.set_yticks(np.round(T2_span[0:resolution:2]/1000))
ax.set_xlabel(r'$T_1$ [$\mu s$]')
ax.set_ylabel(r'$T_2$ [$\mu s$]')

# %% plot any
var_to_plot = mask_threshold
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
cbar0.set_label(r'Threshold or not',
                labelpad=0, y=1.20, rotation=0)

ax.set_xticks(np.round(T1_span[0:resolution:2]/1000))
ax.set_yticks(np.round(T2_span[0:resolution:2]/1000))
ax.set_xlabel(r'$T_1$ [$\mu s$]')
ax.set_ylabel(r'$T_2$ [$\mu s$]')
#%% Load perf stab
with open('data/T1T2_perfect_decoding_test_data.npy', 'rb') as f:
        n_cycles_perf = np.load(f)
        n_shots_perf = np.load(f)
        resolution_perf = np.load(f)
        T1_min_perf = np.load(f)
        T1_max_perf = np.load(f)
        T2_min_perf = np.load(f)
        T2_max_perf = np.load(f)
        T1_span_perf = np.load(f)
        T2_span_perf = np.load(f)
        fid_span_perf = np.load(f)
        P_L_span_perf = np.load(f)
        time_span_perf = np.load(f)
        par_span_perf = np.load(f)
        par_L_span_perf = np.load(f)
        cov_span_perf = np.load(f)
        cov_L_span_perf = np.load(f)
# %% 4.6 plot
pad = 0.12
aspect_c = 15 
fraction = 0.1
var_to_plot = par_L_span[:, :, 0]
fid_min = np.min(var_to_plot[var_to_plot != invalid_marker])
fid_max = np.max(var_to_plot[var_to_plot != invalid_marker])

fig, ax = plt.subplots(1, 3, figsize=(10, 6))
ax = ax.flatten()
norm = colors.Normalize(vmin=fid_min/1000, vmax=fid_max/1000)
# norm = colors.Normalize(vmin=15, vmax=150)
T1_padding = (T1_max-T1_min)/(resolution*1000*2)
T2_padding = (T2_max-T2_min)/(resolution*1000*2)
aspect = (T1_max-T1_min)/(T2_max-T2_min)
HM = ax[0].imshow(var_to_plot.T/1000, extent=[T1_min/1000-T1_padding, T1_max/1000+T1_padding, T2_min /
                                                 1000-T2_padding, T2_max/1000+T2_padding], aspect=aspect, cmap='magma', origin='lower')

HM.set_norm(norm)
cbar0 = fig.colorbar(HM, ax=ax[0], orientation='horizontal',
                     fraction=fraction, pad=pad,aspect = aspect_c)
cbar0.set_label(r'$T_L$ [$\mu s$]',
                labelpad=0, y=1.20, rotation=0)

ax[0].set_xticks(np.round(T1_span[0:resolution]/1000))
ax[0].set_yticks(np.round(T2_span[0:resolution]/1000))
ax[0].set_xlabel(r'$T_1$ [$\mu s$]')
ax[0].set_ylabel(r'$T_2$ [$\mu s$]')

var_to_plot = np.mean(P_L_span[:,:,1:], axis = 2)
invalid_marker_tmp = 0
var_to_plot[var_to_plot == invalid_marker_tmp] = invalid_marker
fid_min = np.min(var_to_plot[var_to_plot != invalid_marker])
fid_max = np.max(var_to_plot[var_to_plot != invalid_marker])
norm = colors.Normalize(vmin=fid_min, vmax=fid_max)
T1_padding = (T1_max-T1_min)/(resolution*1000*2)
T2_padding = (T2_max-T2_min)/(resolution*1000*2)
aspect = (T1_max-T1_min)/(T2_max-T2_min)
HM = ax[1].imshow(var_to_plot.T, extent=[T1_min/1000-T1_padding, T1_max/1000+T1_padding, T2_min /
                                            1000-T2_padding, T2_max/1000+T2_padding], aspect=aspect, cmap='magma', origin='lower')

HM.set_norm(norm)
cbar0 = fig.colorbar(HM, ax=ax[1], orientation='horizontal',
                     fraction=fraction, pad=pad, aspect = aspect_c)
cbar0.set_label(r'$P_L$ - overlap with codespace',
                labelpad=0, y=1.0, rotation=0)

ax[1].set_xticks(np.round(T1_span[0:resolution]/1000))
ax[1].set_yticks(np.round(T2_span[0:resolution]/1000))
ax[1].set_xlabel(r'$T_1$ [$\mu s$]')
ax[1].set_ylabel(r'$T_2$ [$\mu s$]')

# resolution = 9
# T1_min = 30e3
# T1_max = 110e3
# T2_min = 50e3
# T2_max = 130e3
var_to_plot = par_L_span_perf[:, :, 0]
fid_min = np.min(var_to_plot[var_to_plot != invalid_marker])
fid_max = np.max(var_to_plot[var_to_plot != invalid_marker])
norm = colors.Normalize(vmin=fid_min/1000, vmax=fid_max/1000)
# norm = colors.Normalize(vmin=15, vmax=150)
T1_padding = (T1_max-T1_min)/(resolution*1000*2)
T2_padding = (T2_max-T2_min)/(resolution*1000*2)
aspect = (T1_max-T1_min)/(T2_max-T2_min)
HM = ax[2].imshow(var_to_plot.T/1000, extent=[T1_min/1000-T1_padding, T1_max/1000+T1_padding, T2_min /
                                                 1000-T2_padding, T2_max/1000+T2_padding], aspect=aspect, cmap='magma', origin='lower')

HM.set_norm(norm)
cbar0 = fig.colorbar(HM, ax=ax[2], orientation='horizontal',
                     fraction=fraction, pad=pad, aspect = aspect_c)
cbar0.set_label(r'$T_L$ with perfect decoding [$\mu s$]',
                labelpad=0, y=1.20, rotation=0)

ax[2].set_xticks(np.round(T1_span[0:resolution]/1000))
ax[2].set_yticks(np.round(T2_span[0:resolution]/1000))
ax[2].set_xlabel(r'$T_1$ [$\mu s$]')
ax[2].set_ylabel(r'$T_2$ [$\mu s$]')
fig.tight_layout()
fig.savefig('T1T2SweepNew.pdf')
# %% Border
threshold = np.zeros((resolution, resolution))
for i in range(resolution):
    for j in range(resolution):
        if T2_span[j] > 2*T1_span[i]:
            if par_L_span[i, j, 0] == invalid_marker:
                threshold[i][j] = 0
                continue
        if par_L_span[i, j, 0] > T1_span[i] and par_L_span[i, j, 0] > T2_span[j]:
            threshold[i][j] = 1
mask_threshold = np.ma.masked_where(1-threshold,np.ones((resolution, resolution))*1000)#par_L_span[:, :, 0])

var_to_plot = par_L_span[:, :, 0]#mask_threshold #
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

var_to_plot = mask_threshold
# for i in range(resolution):
#     for j in range(resolution):
#         if T2_span[j] > 2*T1_span[i]:
#             if par_L_span[i, j, 0] == invalid_marker:
#                 continue
#         if par_L_span[i, j, 0] > T1_span[i] or par_L_span[i, j, 0] > T2_span[j]:
#             ax.scatter(T1_span[i]/1000,T2_span[j]/1000,marker = 'x',c='k')
c = 'darkviolet'
for i in range(resolution):
    for j in range(resolution):
        if threshold[i, j] == 0:
            continue
        if threshold[i, j-1] == 0:
            ax.plot([T1_span[i]/1000-T1_padding,T1_span[i]/1000+T1_padding],[T2_span[j]/1000 - T2_padding,T2_span[j]/1000 - T2_padding],c)
        if threshold[i, min(j+1,resolution-1)] == 0:
            ax.plot([T1_span[i]/1000-T1_padding,T1_span[i]/1000+T1_padding],[T2_span[j]/1000 + T2_padding+1,T2_span[j]/1000 + T2_padding+1],c)
        if threshold[min(i+1,resolution-1), j] == 0:
            ax.plot([T1_span[i]/1000+T1_padding,T1_span[i]/1000+T1_padding],[T2_span[j]/1000 - T2_padding,T2_span[j]/1000 + T2_padding+1],c)
        if threshold[max(i-1,0), j] == 0:
            ax.plot([T1_span[i]/1000-T1_padding,T1_span[i]/1000-T1_padding],[T2_span[j]/1000 - T2_padding,T2_span[j]/1000 + T2_padding+1],c)

# ax.scatter(T1_span[i]/1000,T2_span[j]/1000,marker = 'x',c='k',label = 'Threshold reached')
# HM = ax.imshow(var_to_plot.T, alpha=1, extent=[T1_min/1000-T1_padding, T1_max/1000+T1_padding, T2_min /
#                                             1000-T2_padding, T2_max/1000+T2_padding], aspect=aspect, cmap='magma', origin='lower')
# ax.legend()
# ax.set_ylim((46,145))
ax.set_xticks(np.round(T1_span[0:resolution]/1000))
ax.set_yticks(np.round(T2_span[0:resolution]/1000))
ax.set_xlabel(r'$T_1$ [$\mu s$]')
ax.set_ylabel(r'$T_2$ [$\mu s$]')

#%%
var_to_plot = threshold
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
cbar0.set_label(r'Threshold or not',
                labelpad=0, y=1.20, rotation=0)

ax.set_xticks(np.round(T1_span[0:resolution:2]/1000))
ax.set_yticks(np.round(T2_span[0:resolution:2]/1000))
ax.set_xlabel(r'$T_1$ [$\mu s$]')
ax.set_ylabel(r'$T_2$ [$\mu s$]')
# %% 4.6 including border

pad = 0.12
aspect_c = 15 
fraction = 0.1
var_to_plot = par_L_span[:, :, 0]
fid_min = np.min(var_to_plot[var_to_plot != invalid_marker])
fid_max = np.max(var_to_plot[var_to_plot != invalid_marker])

fig, ax = plt.subplots(1, 3, figsize=(10, 6))
ax = ax.flatten()
norm = colors.Normalize(vmin=fid_min/1000, vmax=fid_max/1000)
# norm = colors.Normalize(vmin=15, vmax=150)
T1_padding = (T1_max-T1_min)/(resolution*1000*2)
T2_padding = (T2_max-T2_min)/(resolution*1000*2)
aspect = (T1_max-T1_min)/(T2_max-T2_min)
HM = ax[0].imshow(var_to_plot.T/1000, extent=[T1_min/1000-T1_padding, T1_max/1000+T1_padding, T2_min /
                                                 1000-T2_padding, T2_max/1000+T2_padding], aspect=aspect, cmap='magma', origin='lower')

HM.set_norm(norm)
cbar0 = fig.colorbar(HM, ax=ax[0], orientation='horizontal',
                     fraction=fraction, pad=pad,aspect = aspect_c)
cbar0.set_label(r'$T_L$ [$\mu s$]',
                labelpad=0, y=1.20, rotation=0)

# Draw border to threshold
threshold = np.zeros((resolution, resolution))
for i in range(resolution):
    for j in range(resolution):
        if T2_span[j] > 2*T1_span[i]:
            if par_L_span[i, j, 0] == invalid_marker:
                threshold[i][j] = 0
                continue
        if par_L_span[i, j, 0] > T1_span[i] or par_L_span[i, j, 0] > T2_span[j]:
            threshold[i][j] = 1
c = 'gold'
for i in range(resolution):
    for j in range(resolution):
        if threshold[i, j] == 0:
            continue
        if threshold[i, j-1] == 0:
            ax[0].plot([T1_span[i]/1000-T1_padding,T1_span[i]/1000+T1_padding],[T2_span[j]/1000 - T2_padding,T2_span[j]/1000 - T2_padding],c)
        if threshold[i, min(j+1,resolution-1)] == 0:
            ax[0].plot([T1_span[i]/1000-T1_padding,T1_span[i]/1000+T1_padding],[T2_span[j]/1000 + T2_padding+1,T2_span[j]/1000 + T2_padding+1],c)
        if threshold[min(i+1,resolution-1), j] == 0:
            ax[0].plot([T1_span[i]/1000+T1_padding,T1_span[i]/1000+T1_padding],[T2_span[j]/1000 - T2_padding,T2_span[j]/1000 + T2_padding+1],c)
        if threshold[max(i-1,0), j] == 0:
            ax[0].plot([T1_span[i]/1000-T1_padding,T1_span[i]/1000-T1_padding],[T2_span[j]/1000 - T2_padding,T2_span[j]/1000 + T2_padding+1],c)

# Draw border to double threshold
threshold = np.zeros((resolution, resolution))
for i in range(resolution):
    for j in range(resolution):
        if T2_span[j] > 2*T1_span[i]:
            if par_L_span[i, j, 0] == invalid_marker:
                threshold[i][j] = 0
                continue
        if par_L_span[i, j, 0] > T1_span[i] and par_L_span[i, j, 0] > T2_span[j]:
            threshold[i][j] = 1
c = 'indigo'
for i in range(resolution):
    for j in range(resolution):
        if threshold[i, j] == 0:
            continue
        if threshold[i, j-1] == 0:
            ax[0].plot([T1_span[i]/1000-T1_padding,T1_span[i]/1000+T1_padding],[T2_span[j]/1000 - T2_padding,T2_span[j]/1000 - T2_padding],c)
        if threshold[i, min(j+1,resolution-1)] == 0:
            ax[0].plot([T1_span[i]/1000-T1_padding,T1_span[i]/1000+T1_padding],[T2_span[j]/1000 + T2_padding+1,T2_span[j]/1000 + T2_padding+1],c)
        if threshold[min(i+1,resolution-1), j] == 0:
            ax[0].plot([T1_span[i]/1000+T1_padding+1,T1_span[i]/1000+T1_padding+1],[T2_span[j]/1000 - T2_padding,T2_span[j]/1000 + T2_padding+1],c)
        if threshold[max(i-1,0), j] == 0:
            ax[0].plot([T1_span[i]/1000-T1_padding,T1_span[i]/1000-T1_padding],[T2_span[j]/1000 - T2_padding,T2_span[j]/1000 + T2_padding+1],c)

# ax[0].set_ylim((46,145)) # Tweak this to get the plot looking right
ax[0].set_ylim((T2_min/1000-T2_padding,T2_max/1000+T2_padding)) 

ax[0].set_xticks(np.round(T1_span[0:resolution:2]/1000))
ax[0].set_yticks(np.round(T2_span[0:resolution:2]/1000))
ax[0].set_xlabel(r'$T_1$ [$\mu s$]')
ax[0].set_ylabel(r'$T_2$ [$\mu s$]')

var_to_plot = np.mean(P_L_span[:,:,1:], axis = 2)
invalid_marker_tmp = 0
var_to_plot[var_to_plot == invalid_marker_tmp] = invalid_marker
fid_min = np.min(var_to_plot[var_to_plot != invalid_marker])
fid_max = np.max(var_to_plot[var_to_plot != invalid_marker])
norm = colors.Normalize(vmin=fid_min, vmax=fid_max)
T1_padding = (T1_max-T1_min)/(resolution*1000*2)
T2_padding = (T2_max-T2_min)/(resolution*1000*2)
aspect = (T1_max-T1_min)/(T2_max-T2_min)
HM = ax[1].imshow(var_to_plot.T, extent=[T1_min/1000-T1_padding, T1_max/1000+T1_padding, T2_min /
                                            1000-T2_padding, T2_max/1000+T2_padding], aspect=aspect, cmap='magma', origin='lower')

HM.set_norm(norm)
cbar0 = fig.colorbar(HM, ax=ax[1], orientation='horizontal',
                     fraction=fraction, pad=pad, aspect = aspect_c)
cbar0.set_label(r'$P_L$ - overlap with codespace',
                labelpad=0, y=1.0, rotation=0)

ax[1].set_xticks(np.round(T1_span[0:resolution:2]/1000))
ax[1].set_yticks(np.round(T2_span[0:resolution:2]/1000))
ax[1].set_xlabel(r'$T_1$ [$\mu s$]')
ax[1].set_ylabel(r'$T_2$ [$\mu s$]')

resolution = 10
T1_min = 30e3
T1_max = 120e3
T2_min = 50e3
T2_max = 140e3
var_to_plot = par_L_span_perf[:, :, 0]
fid_min = np.min(var_to_plot[var_to_plot != invalid_marker])
fid_max = np.max(var_to_plot[var_to_plot != invalid_marker])
norm = colors.Normalize(vmin=fid_min/1000, vmax=fid_max/1000)
# norm = colors.Normalize(vmin=15, vmax=150)
T1_padding = (T1_max-T1_min)/(resolution*1000*2)
T2_padding = (T2_max-T2_min)/(resolution*1000*2)
aspect = (T1_max-T1_min)/(T2_max-T2_min)
HM = ax[2].imshow(var_to_plot.T/1000, extent=[T1_min/1000-T1_padding, T1_max/1000+T1_padding, T2_min /
                                                 1000-T2_padding, T2_max/1000+T2_padding], aspect=aspect, cmap='magma', origin='lower')

HM.set_norm(norm)
cbar0 = fig.colorbar(HM, ax=ax[2], orientation='horizontal',
                     fraction=fraction, pad=pad, aspect = aspect_c)
cbar0.set_label(r'$T_L$ with perfect decoding [$\mu s$]',
                labelpad=0, y=1.20, rotation=0)

ax[2].set_xticks(np.round(T1_span[0:resolution:2]/1000))
ax[2].set_yticks(np.round(T2_span[0:resolution:2]/1000))
ax[2].set_xlabel(r'$T_1$ [$\mu s$]')
ax[2].set_ylabel(r'$T_2$ [$\mu s$]')
resolution = 11
T1_min = 30e3
T1_max = 130e3
T2_min = 50e3
T2_max = 150e3
fig.tight_layout()
fig.savefig('T1T2SweepNew.pdf')
# %%
