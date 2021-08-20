"""
Creates fig 4.2
"""

# Main file for all simulations of encoding fidelity
#%%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from qiskit import *

# Our own files
from simulator_program.custom_noise_models import standard_times
from simulator_program.data_analysis_tools import *

default_colors = plt.get_cmap("tab10")
# %%
resolution = 15
T1_min = 30e3
T1_max = 200e3
T2_min = 30e3
T2_max = 250e3
T1_span = np.linspace(T1_min, T1_max, resolution)
T2_span = np.linspace(T2_min, T2_max, resolution)
fid_span = np.zeros((resolution, resolution))
time_span = np.zeros((resolution, resolution))

invalid_marker = float('inf')
for i in range(resolution):
    for j in range(resolution):
        if T2_span[j] > 2*T1_span[i]:
            fid_span[i][j] = invalid_marker
            time_span[i][j] = invalid_marker
            continue
        fid_span[i][j], _, time_span[i][j] = encoding_fidelity(1, gate_times=standard_times, T1=T1_span[i], T2=T2_span[j],
                                                               theta=0, phi=0, device='WACQT')
fid_min = np.min(fid_span[fid_span != invalid_marker])
fid_max = np.max(fid_span[fid_span != invalid_marker])
# %% contourf
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
X_mesh, Y_mesh = np.meshgrid(T1_span, T2_span)
HM = ax.contourf(X_mesh/1000, Y_mesh/1000, fid_span,
                 cmap='magma', origin='lower')
cbar0 = fig.colorbar(HM, ax=ax, orientation='horizontal',
                     fraction=.1, pad=0.17)
cbar0.set_label(r'Encoding success probability',
                labelpad=0, y=1.20, rotation=0)
plt.show()
# %% imshow
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
norm = colors.Normalize(vmin=fid_min, vmax=fid_max)
T1_padding = (T1_max-T1_min)/(resolution*1000*2)
T2_padding = (T2_max-T2_min)/(resolution*1000*2)
aspect = (T1_max-T1_min)/(T2_max-T2_min)
HM = ax.imshow(fid_span.T, extent=[T1_min/1000-T1_padding, T1_max/1000+T1_padding, T2_min /
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

# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# ax.spines["left"].set_visible(False)
# ax.spines["bottom"].set_visible(False)
# ax.grid(True, alpha = 0.3)
plt.show()
plt.savefig('encode_sweep.pdf')
# %% T1 ~ T2
T1 = 40e3
T2 = 60e3
resolution = 20
scalings = np.linspace(0.5, 3, resolution)
fids = np.zeros(resolution)

for i in range(resolution):
    fids[i], _, _ = encoding_fidelity(1, gate_times=standard_times, T1=T1*scalings[i], T2=T2*scalings[i],
                                      theta=0, phi=0, device='WACQT')
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(T1*scalings/1000, fids)
ax.set_xlabel(r'$T_1=1.5 \cdot T_2$ [$\mu s$]')
ax.set_ylabel(r'Fidelity [a.u.]')
ax.set_yscale('log')
# ax.set_yticks([0.9, 0.99, 0.999])
# ax.get_yaxis().get_major_formatter().labelOnlyBase = False
plt.show()
# %% T1 ~ 1/T2
T1 = 40e3
T2 = 40e3
resolution = 20
scalings = np.linspace(np.sqrt(0.5), 2, resolution)
fids = np.zeros(resolution)

for i in range(resolution):
    fids[i], _, _ = encoding_fidelity(1, gate_times=standard_times, T1=T1*scalings[i], T2=T2/scalings[i],
                                      theta=0, phi=0, device='WACQT')
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(T1*scalings/1000, fids)
ax.set_xlabel(r'$T_1=1/ T_2$ [$\mu s$]')
ax.set_ylabel(r'Fidelity [a.u.]')
# ax.set_yscale('log')
# ax.set_yticks([0.9, 0.99, 0.999])
# ax.get_yaxis().get_major_formatter().labelOnlyBase = False
plt.show()
# %% varying T1>T2
resolution = 10
T1_min = 30e3
T1_max = 150e3
T2_min = 30e3
T2_max = 150e3
T2_span = np.linspace(T1_min, T1_max, 5)
T1_span = np.linspace(T2_min, T2_max, resolution)
scalings = np.linspace(0.5, 2, resolution)
fids = np.zeros(resolution)

fig, ax = plt.subplots(1, 1, figsize=(7, 5))

for T2 in T2_span:
    for i, T1 in enumerate(np.linspace(T2/2, T2_max, resolution)):
        fids[i], _, _ = encoding_fidelity(1, gate_times=standard_times, T1=T1, T2=T2,
                                          theta=0, phi=0, device='WACQT')
    ax.plot(np.linspace(T2/2, T2_max, resolution) /
            1000, fids, label='T2='+str(T2/1000)+'μs')

ax.set_xlabel(r'$T_1$ [$\mu s$]')
ax.set_ylabel(r'Fidelity [a.u.]')
ax.legend()
# ax.set_yscale('log')
plt.show()
# %% plot T1~T2 ~ 1/gate times, works as expected
T1 = 40e3
T2 = 60e3
resolution = 20
scalings = np.linspace(0.5, 3, resolution)
fids = np.zeros(resolution)


def scale_gates(gate_times, scale):
    new_gate_times = {}
    for key in gate_times.get_gate_times():
        new_gate_times[key] = gate_times[key]*scale
    return new_gate_times


for i in range(resolution):
    fids[i], _, _ = encoding_fidelity(1, gate_times=scale_gates(standard_times, scalings[i]), T1=T1*scalings[i], T2=T2*scalings[i],
                                      theta=0, phi=0, device='WACQT')
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(T1*scalings/1000, fids)
ax.set_xlabel(r'$T_1=1.5 \cdot T_2$ [$\mu s$]')
ax.set_ylabel(r'Fidelity [a.u.]')
# ax.set_yscale('log')
plt.show()
# %% plot gate times
T1 = 40e3
T2 = 60e3
resolution = 30


def scale_all_gates(gate_times, scale):
    new_gate_times = {}
    for key in gate_times.get_gate_times():
        new_gate_times[key] = gate_times[key]*scale
    return new_gate_times


def scale_singleqb_gates(gate_times, scale):
    new_gate_times = {}
    for key in gate_times.single_qubit_gates:
        new_gate_times[key] = gate_times[key]*scale
    return gate_times.get_gate_times(new_gate_times)


def scale_twoqb_gates(gate_times, scale):
    new_gate_times = {}
    for key in gate_times.two_qubit_gates:
        new_gate_times[key] = gate_times[key]*scale
    return gate_times.get_gate_times(new_gate_times)


fids_all = np.zeros(resolution)
cycle_durations_all = np.zeros(resolution)
scalings_all = np.geomspace(0.05, 2, resolution)
for i in range(resolution):
    fids_all[i], _, cycle_durations_all[i] = encoding_fidelity(1, gate_times=scale_all_gates(standard_times, scalings_all[i]), T1=T1, T2=T2,
                                                               theta=0, phi=0, device='WACQT')

fids_single = np.zeros(resolution)
cycle_durations_single = np.zeros(resolution)
scalings_single = np.geomspace(0.05, 5, resolution)
for i in range(resolution):
    fids_single[i], _, cycle_durations_single[i] = encoding_fidelity(1, gate_times=scale_singleqb_gates(standard_times, scalings_single[i]), T1=T1, T2=T2,
                                                                     theta=0, phi=0, device='WACQT')

fids_two = np.zeros(resolution)
cycle_durations_two = np.zeros(resolution)
scalings_two = np.geomspace(0.05, 2, resolution)
for i in range(resolution):
    fids_two[i], _, cycle_durations_two[i] = encoding_fidelity(1, gate_times=scale_twoqb_gates(standard_times, scalings_two[i]), T1=T1, T2=T2,
                                                               theta=0, phi=0, device='WACQT')

# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(cycle_durations_all, fids_all, label='All gates proportionally')
ax.plot(cycle_durations_single, fids_single, label='Only single-qubit gates')
ax.plot(cycle_durations_two, fids_two, label='Only two-qubit gates')

ax.set_xlabel(r'Total circuit duration [ns]')
ax.set_ylabel(r'Fidelity [a.u.]')
ax.legend()

plt.yticks([0.9, 0.99, 0.999], ['0.9', '0.99', '0.999'])
ax.set_yscale('logit')
# plt.yticks([])
# ax.set_yticks([0.9, 0.99,0.999])
# ax.set_yticklabels(['0.9', '0.99','0.999'])
ax.set_xscale('log')
plt.show()
# %%
