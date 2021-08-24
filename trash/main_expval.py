# %% Import modules
import matplotlib.pyplot as plt
import numpy as np
from qiskit import *


# Our own files
from simulator_program.custom_noise_models import (
    WACQT_target_times,
    WACQT_demonstrated_times,
    standard_times)
from simulator_program.data_analysis_tools import *
from simulator_program.custom_transpiler import *
from simulator_program.stabilizers import *
from simulator_program.post_select import *
from simulator_program.post_process import *
from simulator_program.idle_noise import *


#% TEST encoding fidelities
#fid, circ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
#        reset=True, idle_noise=True, theta=np.pi, phi=0,
#        snapshot_type='dm', device='Nah', pauliop='ZZZZZ')
#print(fid)


#%% Test run
n_cycles=3
n_shots=1024*4
fid_rec, time_rec = fidelity_from_scratch(n_cycles, n_shots, gate_times={'delay':1000},
    reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='dm', encoding=False, theta=0, phi=0,
    transpile=False, pauliop='ZZZZZ', device_properties=WACQT_device_properties,
    device=None, simulator_type='density_matrix',rename=True)

# %%
# Settings to used across most configurations
n_cycles = 15
n_shots = 1024*8

#%% ========== CONFIGURATIONS FOR RUNS ==========
# EXPECTATION VALUES OF |+> AND |->
exp_target_WACQT_p, time_target_WACQT_p = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
    reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=np.pi/2,
    transpile=True, pauliop='XXXXX')
#exp_target_FC_p, time_target_FC_p = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
#    reset=True, data_process_type='recovery',
#    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=np.pi/2,
#    transpile=False, pauliop='XXXXX')
exp_target_DD_p, time_target_DD_p = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
    reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=np.pi/2,
    transpile=False, pauliop='XXXXX', device='double_diamond')
print('Check!')
exp_dem_WACQT_p, time_dem_WACQT_p = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_demonstrated_times,
    reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=np.pi/2,
    transpile=True, pauliop='XXXXX')
#exp_dem_FC_p, time_dem_FC_p = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_demonstrated_times,
#    reset=True, data_process_type='recovery',
#    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=np.pi/2,
#    transpile=False, pauliop='XXXXX')
exp_dem_DD_p, time_dem_DD_p = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_demonstrated_times,
    reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=np.pi/2,
    transpile=False, pauliop='XXXXX', device='double_diamond')
print('Check!')

# |->
exp_target_WACQT_m, time_target_WACQT_m = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
    reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=-np.pi/2,
    transpile=True, pauliop='XXXXX')
#exp_target_FC_m, time_target_FC_m = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
#    reset=True, data_process_type='recovery',
#    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=-np.pi/2,
#    transpile=False, pauliop='XXXXX')
exp_target_DD_m, time_target_DD_m = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
    reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=-np.pi/2,
    transpile=False, pauliop='XXXXX', device='double_diamond')
print('Check!')
exp_dem_WACQT_m, time_dem_WACQT_m = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_demonstrated_times,
    reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=-np.pi/2,
    transpile=True, pauliop='XXXXX')
#exp_dem_FC_m, time_dem_FC_m = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_demonstrated_times,
#    reset=True, data_process_type='recovery',
#    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=-np.pi/2,
#    transpile=False, pauliop='XXXXX')
exp_dem_DD_m, time_dem_DD_m = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_demonstrated_times,
    reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=-np.pi/2,
    transpile=False, pauliop='XXXXX', device='double_diamond')
print('Check!')

#%% EXPECTATION VALUES OF |0> AND |1>
# |0>
exp_target_WACQT_0, time_target_WACQT_0 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
    reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
    transpile=True, pauliop='ZZZZZ')
#exp_target_FC_0, time_target_FC_0 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
#    reset=True, data_process_type='recovery',
#    idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
#    transpile=False, pauliop='ZZZZZ')
exp_target_DD_0, time_target_DD_0 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
    reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
    transpile=False, pauliop='ZZZZZ', device='double_diamond')
print('Check!')
exp_dem_WACQT_0, time_dem_WACQT_0 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_demonstrated_times,
    reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
    transpile=True, pauliop='ZZZZZ')
#exp_dem_FC_0, time_dem_FC_0 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_demonstrated_times,
#    reset=True, data_process_type='recovery',
#    idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
#    transpile=False, pauliop='ZZZZZ')
exp_dem_DD_0, time_dem_DD_0 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_demonstrated_times,
    reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
    transpile=False, pauliop='ZZZZZ', device='double_diamond')
print('Check!')
#%%
# |1>
exp_target_WACQT_1, time_target_WACQT_1 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
    reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi, phi=0,
    transpile=True, pauliop='ZZZZZ')
#exp_target_FC_1, time_target_FC_1 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
#    reset=True, data_process_type='recovery',
#    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi, phi=0,
#    transpile=False, pauliop='ZZZZZ')
exp_target_DD_1, time_target_DD_1 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
    reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi, phi=0,
    transpile=False, pauliop='ZZZZZ', device='double_diamond')
#%%
print('Check!')
exp_dem_WACQT_1, time_dem_WACQT_1 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_demonstrated_times,
    reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi, phi=0,
    transpile=True, pauliop='ZZZZZ')
#exp_dem_FC_1, time_dem_FC_1 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_demonstrated_times,
#    reset=True, data_process_type='recovery',
#    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi, phi=0,
#    transpile=False, pauliop='ZZZZZ')
exp_dem_DD_1, time_dem_DD_1 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_demonstrated_times,
    reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi, phi=0,
    transpile=False, pauliop='ZZZZZ', device='double_diamond')
print('Check!')

#%% FIDELITIES (Not in use)
# Fidelities of |1>
fid_target_WACQT, time_target_WACQT = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
    reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='dm', encoding=False, theta=np.pi, phi=0,
    transpile=True, pauliop='XXXXX')
#fid_target_FC, time_target_FC = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
#    reset=True, data_process_type='recovery',
#    idle_noise=True, snapshot_type='dm', encoding=False, theta=np.pi, phi=0,
#    transpile=False, pauliop='XXXXX')
fid_target_DD, time_target_DD = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
    reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='dm', encoding=False, theta=np.pi, phi=0,
    transpile=False, pauliop='XXXXX', device='double_diamond')
print('Check!')
fid_dem_WACQT, time_dem_WACQT = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_demonstrated_times,
    reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='dm', encoding=False, theta=np.pi, phi=0,
    transpile=True, pauliop='XXXXX')
#fid_dem_FC, time_dem_FC = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_demonstrated_times,
#    reset=True, data_process_type='recovery',
#    idle_noise=True, snapshot_type='dm', encoding=False, theta=np.pi, phi=0,
#    transpile=False, pauliop='XXXXX')
fid_dem_DD, time_dem_DD = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_demonstrated_times,
    reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='dm', encoding=False, theta=np.pi, phi=0,
    transpile=False, pauliop='XXXXX', device='double_diamond')
print('Check!')


#%% Post selection
n_shots=1024*16
#%% + and - (Target times)
exp_target_WACQT_p_PS, count_target_WACQT_p = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
    reset=True, data_process_type='post_select',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=np.pi/2,
    transpile=True, pauliop='XXXXX')
print('Check!')
exp_target_WACQT_m_PS, count_target_WACQT_m = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
    reset=True, data_process_type='post_select',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=-np.pi/2,
    transpile=True, pauliop='XXXXX')
print('Check!')
exp_target_DD_p_PS, count_target_DD_p = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
    reset=True, data_process_type='post_select',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=np.pi/2,
    transpile=False, pauliop='XXXXX', device='double_diamond')
print('Check!')
exp_target_DD_m_PS, count_target_DD_m = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
    reset=True, data_process_type='post_select',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=-np.pi/2,
    transpile=False, pauliop='XXXXX', device='double_diamond')
#%% + and - (Demonstrated times)
exp_dem_WACQT_p_PS, count_dem_WACQT_p = fidelity_from_scratch(7, n_shots, gate_times=WACQT_demonstrated_times,
    reset=True, data_process_type='post_select',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=np.pi/2,
    transpile=True, pauliop='XXXXX')
exp_dem_WACQT_m_PS, count_dem_WACQT_m = fidelity_from_scratch(7, n_shots, gate_times=WACQT_demonstrated_times,
    reset=True, data_process_type='post_select',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=-np.pi/2,
    transpile=True, pauliop='XXXXX')
exp_dem_DD_p_PS, count_dem_DD_p = fidelity_from_scratch(7, n_shots, gate_times=WACQT_demonstrated_times,
    reset=True, data_process_type='post_select',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=np.pi/2,
    transpile=False, pauliop='XXXXX', device='double_diamond')
exp_dem_DD_m_PS, count_dem_DD_m = fidelity_from_scratch(7, n_shots, gate_times=WACQT_demonstrated_times,
    reset=True, data_process_type='post_select',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=-np.pi/2,
    transpile=False, pauliop='XXXXX', device='double_diamond')

#%% 0 and 1 PS
n_shots=1024*16

exp_target_WACQT_0_PS, count_target_WACQT_0 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
    reset=True, data_process_type='post_select',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
    transpile=True, pauliop='ZZZZZ')
print('Check!')
exp_target_WACQT_1_PS, count_target_WACQT_1 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
    reset=True, data_process_type='post_select',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi, phi=-0,
    transpile=True, pauliop='ZZZZZ')
print('Check!')
exp_target_DD_0_PS, count_target_DD_0 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
    reset=True, data_process_type='post_select',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
    transpile=False, pauliop='ZZZZZ', device='double_diamond')
print('Check!')
exp_target_DD_1_PS, count_target_DD_1 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
    reset=True, data_process_type='post_select',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi, phi=-0,
    transpile=False, pauliop='ZZZZZ', device='double_diamond')
print('Check!')
#%% 0 and 1 (Demonstrated times)
exp_dem_WACQT_0_PS, count_dem_WACQT_0 = fidelity_from_scratch(7, n_shots, gate_times=WACQT_demonstrated_times,
    reset=True, data_process_type='post_select',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
    transpile=True, pauliop='ZZZZZ')
print('Check!')
exp_dem_WACQT_1_PS, count_dem_WACQT_1 = fidelity_from_scratch(7, n_shots, gate_times=WACQT_demonstrated_times,
    reset=True, data_process_type='post_select',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi, phi=-0,
    transpile=True, pauliop='ZZZZZ')
print('Check!')
exp_dem_DD_0_PS, count_dem_DD_0 = fidelity_from_scratch(7, n_shots, gate_times=WACQT_demonstrated_times,
    reset=True, data_process_type='post_select',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
    transpile=False, pauliop='ZZZZZ', device='double_diamond')
print('Check!')
exp_dem_DD_1_PS, count_dem_DD_1 = fidelity_from_scratch(7, n_shots, gate_times=WACQT_demonstrated_times,
    reset=True, data_process_type='post_select',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi, phi=-0,
    transpile=False, pauliop='ZZZZZ', device='double_diamond')

#%% EXPECTATION VALUES OF |0> AND |1>, NO QEC
n_shots = 1024*4
n_cycles = 15
# |0>
exp_target_WACQT_0_decay = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
    reset=True, data_process_type='empty_circuit',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
    transpile=True, pauliop='ZZZZZ')

exp_target_DD_0_decay = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
    reset=True, data_process_type='empty_circuit',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
    transpile=False, pauliop='ZZZZZ', device='double_diamond')
print('Check!')

# |1>
exp_target_WACQT_1_decay = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
    reset=True, data_process_type='empty_circuit',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi, phi=0,
    transpile=True, pauliop='ZZZZZ')
exp_target_DD_1_decay = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
    reset=True, data_process_type='empty_circuit',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi, phi=0,
    transpile=False, pauliop='ZZZZZ', device='double_diamond')
print('Check!')

#%% Testing single qubit
n_cycles = 15
n_shots = 1024*4
#%%
fid_target_single = fid_single_qubit(n_cycles, n_shots,
                                     gate_times=WACQT_target_times,
                                     snapshot_type='exp', T1=40e3, T2=10e3,
                                     theta=np.pi/2, phi=np.pi/2, pauliop='X')
#%%
exp_target_single_p = fid_single_qubit(n_cycles, n_shots,
                                     gate_times=WACQT_target_times,
                                     snapshot_type='exp', T1=40e3, T2=60e3,
                                     theta=np.pi/2, phi=np.pi/2, pauliop='X')

exp_dem_single_p = fid_single_qubit(n_cycles, n_shots,
                                     gate_times=WACQT_demonstrated_times,
                                     snapshot_type='exp', T1=40e3, T2=60e3,
                                     theta=np.pi/2, phi=np.pi/2, pauliop='X')

exp_target_single_m = fid_single_qubit(n_cycles, n_shots,
                                     gate_times=WACQT_target_times,
                                     snapshot_type='exp', T1=40e3, T2=60e3,
                                     theta=np.pi/2, phi=-np.pi/2, pauliop='X')

exp_dem_single_m = fid_single_qubit(n_cycles, n_shots,
                                     gate_times=WACQT_demonstrated_times,
                                     snapshot_type='exp', T1=40e3, T2=60e3,
                                     theta=np.pi/2, phi=-np.pi/2, pauliop='X')
#%%
fid_target_single_1 = fid_single_qubit(n_cycles, n_shots,
                                     gate_times=WACQT_target_times,
                                     snapshot_type='dm', T1=40e3, T2=60e3,
                                     theta=np.pi, phi=0, pauliop='Z')
#%%
fid_dem_single_1 = fid_single_qubit(n_cycles, n_shots,
                                     gate_times=WACQT_demonstrated_times,
                                     snapshot_type='dm', T1=40e3, T2=60e3,
                                     theta=np.pi, phi=0, pauliop='Z')
#%%
fid_target_single_p = fid_single_qubit(n_cycles, n_shots,
                                     gate_times=WACQT_target_times,
                                     snapshot_type='dm', T1=40e3, T2=60e3,
                                     theta=np.pi, phi=np.pi/2, pauliop='Z')
fid_dem_single_p = fid_single_qubit(n_cycles, n_shots,
                                     gate_times=WACQT_demonstrated_times,
                                     snapshot_type='dm', T1=40e3, T2=60e3,
                                     theta=np.pi, phi=np.pi/2, pauliop='Z')

#%% Plotting
fig, ax = plt.subplots(2,1, figsize=(10, 16))
x_dis = np.arange(0,n_cycles+1)
# Generate x-values
x_target_WACQT = []
x_target_DD = []

x_dem_WACQT = []
x_dem_DD = []

for i in range(len(time_target_WACQT_1)-1):
    x_target_WACQT.append(time_target_WACQT_1['exp_'+str(i)])
    x_target_DD.append(time_target_DD_1['exp_'+str(i)])
    #x_dem_WACQT.append(time_dem_WACQT_1['exp_'+str(i)])
    #x_dem_DD.append(time_dem_DD_1['exp_'+str(i)])

# Subplot 1: Target gate times
ax[0].plot(x_target_WACQT, exp_target_WACQT_0, '-o', color='C0', label='Hexagonal layout')
ax[0].plot(x_target_DD, exp_target_DD_0, '-o', color='C1', label='Double diamond layout')
#ax[0].plot(x_target_WACQT, exp_target_FC_0, '-o', color='C2', label='Full connectivity')
ax[0].plot(x_target_WACQT, exp_target_WACQT_0_PS, '-o', color='C2', label='Hexagonal, PS')
ax[0].plot(x_target_DD, exp_target_DD_0_PS, '-o', color='C3', label='Double diamond, PS')
ax[0].plot(x_target_WACQT, exp_target_WACQT_1, '-o', color='C0')
ax[0].plot(x_target_DD, exp_target_DD_1, '-o', color='C1')
#ax[0].plot(x_target_WACQT, exp_target_FC_1, '-o', color='C2')
ax[0].plot(x_target_WACQT, exp_target_WACQT_1_PS, '-o', color='C2')
ax[0].plot(x_target_DD, exp_target_DD_1_PS, '-o', color='C3')

ax[0].plot(x_target_WACQT[0:16], exp_target_WACQT_0_decay, '-o', color='C4', label='Decay of logical state (Hexagonal)')
ax[0].plot(x_target_WACQT[0:16], exp_target_WACQT_1_decay, '-o', color='C4')
ax[0].plot(x_target_WACQT[0:16], exp_target_DD_0_decay, '-o', color='C5', label='Decay of logical state (DD)')
ax[0].plot(x_target_WACQT[0:16], exp_target_DD_1_decay, '-o', color='C5')

ax[0].set(ylim=(-1.0, 1.0))
ax[0].plot(x_target_WACQT, fid_target_single_1, '--k', label='Single qubit')
ax[0].ticklabel_format(style='sci',scilimits=(0,0),useMathText=True)
ax[0].set_title(r'Expectation value of $|0\rangle$ and $|1\rangle$ using target gate times')
ax[0].set_xlabel('Time [ns]')
ax[0].set_ylabel(r'Expectation value of $Z_L$')
ax[0].legend()

ax0 = ax[0].twinx()
ax0.plot(x_target_WACQT, -1*np.array(fid_target_single_1), '--k')
ax0.set(ylim=(-1.0,1.0))
ax0.set_yticklabels(["1.0","0.75","0.5","0.25","0.0","0.25","0.5","0.75","1.0"]) 
ax0.set_ylabel(r'Physical $|1\rangle$ probability')
#%%

ax[1].plot(x_dem_WACQT, exp_dem_WACQT_0, '-o', color='C0', label='Hexagonal layout')
ax[1].plot(x_dem_DD, exp_dem_DD_0, '-o', color='C1', label='Double diamond layout')
#ax[1].plot(x_dem_WACQT, exp_dem_FC_0, '-o', color='C2', label='Full connectivity')
#ax[1].plot(x_dem_WACQT[0:8], exp_dem_WACQT_0_PS, '-o', color='C2', label='Hexagonal, PS')
#ax[1].plot(x_dem_DD[0:8], exp_dem_DD_0_PS, '-o', color='C2', label='Double diamond, PS')

ax[1].plot(x_dem_WACQT, exp_dem_WACQT_1, '-o', color='C0')
ax[1].plot(x_dem_DD, exp_dem_DD_1, '-o', color='C1')
#ax[1].plot(x_dem_WACQT, exp_dem_FC_1, '-o', color='C2', label='Full connectivity')
#ax[1].plot(x_dem_WACQT[0:8], exp_dem_WACQT_1_PS, '-o', color='C2', label='Hexagonal, PS')
#ax[1].plot(x_dem_DD[0:8], exp_dem_DD_1_PS, '-o', color='C2', label='Double diamond, PS')
#ax[1].plot(x_dem_WACQT, exp_dem_single_1, '--k', label='Single qubit')
ax[1].set_title(r'Expectation value of $|0\rangle$ and $|1\rangle$ using demonstrated gate times')
ax[1].set(ylim=(-1.0, 1.0))
ax[1].ticklabel_format(style='sci',scilimits=(0,0),useMathText=True)
ax[1].set_xlabel('Time [ns]')
ax[1].set_ylabel(r'Expectation value of $Z_L$')
ax[1].legend()

ax1 = ax[1].twinx()
ax[1].plot(x_dem_WACQT, fid_dem_single_1, '--k', label='Single qubit')
ax1.plot(x_dem_WACQT, -1*np.array(fid_dem_single_1), '--k')
ax1.set(ylim=(-1.0,1.0))
ax1.set_yticklabels(["1.0","0.75","0.5","0.25","0.0","0.25","0.5","0.75","1.0"]) 
ax1.set_ylabel(r'Physical $|1\rangle$ probability')


#%%
n_cycles=15
n_shots=1024*16

#%% ========== CONFIGURATIONS FOR RUNS IN REPORT ==========
# TODO: Update angles for + and - states
# EXPECTATION VALUES OF |+> AND |->
exp_hex_p, time_hex_p = fidelity_from_scratch(n_cycles, n_shots, 
    gate_times=standard_times, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=np.pi/2,
    transpile=True, pauliop='XXXXX')
exp_DD_p, time_DD_p = fidelity_from_scratch(n_cycles, n_shots, 
    gate_times=standard_times, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=np.pi/2,
    transpile=False, pauliop='XXXXX', device='double_diamond')
exp_pipe_p, time_pipe_p = fidelity_from_scratch(n_cycles, n_shots, 
    gate_times=standard_times, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=np.pi/2,
    transpile=True, pauliop='XXXXX', pipeline=True, include_barriers=False)
print(datetime.datetime.now().time())
#%%
# |->
exp_hex_m, time_hex_m = fidelity_from_scratch(n_cycles, n_shots, 
    gate_times=standard_times, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=-np.pi/2,
    transpile=True, pauliop='XXXXX')
exp_DD_m, time_DD_m = fidelity_from_scratch(n_cycles, n_shots, 
    gate_times=standard_times, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=-np.pi/2,
    transpile=False, pauliop='XXXXX', device='double_diamond')
exp_pipe_m, time_pipe_m = fidelity_from_scratch(n_cycles, n_shots, 
    gate_times=standard_times, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=-np.pi/2,
    transpile=True, pauliop='XXXXX', pipeline=True)
print(datetime.datetime.now().time())
#%%
# |0>
exp_hex_0, time_hex_0 = fidelity_from_scratch(n_cycles, n_shots, 
    gate_times=standard_times, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
    transpile=True, pauliop='ZZZZZ')
exp_DD_0, time_DD_0 = fidelity_from_scratch(n_cycles, n_shots, 
    gate_times=standard_times, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
    transpile=False, pauliop='ZZZZZ', device='double_diamond')
exp_pipe_0, time_pipe_0 = fidelity_from_scratch(n_cycles, n_shots, 
    gate_times=standard_times, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
    transpile=True, pauliop='ZZZZZ', pipeline=True)
print(datetime.datetime.now().time())
#%%
# |1>
exp_hex_1, time_hex_1 = fidelity_from_scratch(n_cycles, n_shots, 
    gate_times=standard_times, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi, phi=0,
    transpile=True, pauliop='ZZZZZ')
exp_DD_1, time_DD_1 = fidelity_from_scratch(n_cycles, n_shots, 
    gate_times=standard_times, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi, phi=0,
    transpile=False, pauliop='ZZZZZ', device='double_diamond')
exp_pipe_1, time_pipe_1 = fidelity_from_scratch(n_cycles, n_shots, 
    gate_times=standard_times, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi, phi=0,
    transpile=True, pauliop='ZZZZZ', pipeline=True)
print(datetime.datetime.now().time())
#%%
exp_pipe_p, time_pipe_p = fidelity_from_scratch(n_cycles, n_shots, 
    gate_times=standard_times, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=np.pi/2,
    transpile=True, pauliop='XXXXX', pipeline=True, include_barriers=False)
exp_pipe_m, time_pipe_m = fidelity_from_scratch(n_cycles, n_shots, 
    gate_times=standard_times, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=-np.pi/2,
    transpile=True, pauliop='XXXXX', pipeline=True, include_barriers=False)
exp_pipe_0, time_pipe_0 = fidelity_from_scratch(n_cycles, n_shots, 
    gate_times=standard_times, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
    transpile=True, pauliop='ZZZZZ', pipeline=True, include_barriers=False)
exp_pipe_1, time_pipe_1 = fidelity_from_scratch(n_cycles, n_shots, 
    gate_times=standard_times, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi, phi=0,
    transpile=True, pauliop='ZZZZZ', pipeline=True, include_barriers=False)

#%% Fidelity |0>
fid_hex_0, time_hex_0 = fidelity_from_scratch(n_cycles, n_shots, 
    gate_times=standard_times, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='dm', encoding=False, theta=0, phi=0,
    transpile=True, pauliop='ZZZZZ')
fid_DD_0, time_DD_0 = fidelity_from_scratch(n_cycles, n_shots, 
    gate_times=standard_times, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='dm', encoding=False, theta=0, phi=0,
    transpile=False, pauliop='ZZZZZ', device='double_diamond')
fid_pipe_0, time_pipe_0 = fidelity_from_scratch(n_cycles, n_shots, 
    gate_times=standard_times, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='dm', encoding=False, theta=0, phi=0,
    transpile=True, pauliop='ZZZZZ', pipeline=True, include_barriers=False)



#%% Projected fidelity |0>
n_cycles = 15
n_shots = 1024*8*3
print(datetime.datetime.now().time())
#fid_L_hex, P_L_hex, time_hex = fidelity_from_scratch(n_cycles, n_shots, 
#    gate_times=standard_times, reset=True, data_process_type='recovery',
#    idle_noise=True, snapshot_type='dm', encoding=False, theta=0, phi=0,
#    transpile=False, project=True)
#print(datetime.datetime.now().time())
fid_L_DD, P_L_DD, time_DD = fidelity_from_scratch(n_cycles, n_shots, 
    gate_times=standard_times, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='dm', encoding=False, theta=0, phi=0,
    transpile=False, project=True, device='double_diamond')
print(datetime.datetime.now().time())
fid_L_pipe, P_L_pipe, time_pipe = fidelity_from_scratch(n_cycles, n_shots, 
    gate_times=standard_times, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='dm', encoding=False, theta=0, phi=0,
    transpile=False, project=True, pipeline=True, include_barriers=False)
print(datetime.datetime.now().time())
#%% Projected fidelity |0> with delay
fid_L_hex_d, P_L_hex_d, time_hex_d = fidelity_from_scratch(n_cycles, n_shots, 
    gate_times={'delay': 5000}, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='dm', encoding=False, theta=0, phi=0,
    transpile=False, project=True)
print(datetime.datetime.now().time())
fid_L_DD_d, P_L_DD_d, time_DD_d = fidelity_from_scratch(n_cycles, n_shots, 
    gate_times={'delay': 5000}, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='dm', encoding=False, theta=0, phi=0,
    transpile=False, project=True, device='double_diamond')
print(datetime.datetime.now().time())
fid_L_pipe_d, P_L_pipe_d, time_pipe_d = fidelity_from_scratch(n_cycles, n_shots, 
    gate_times={'delay': 5000}, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='dm', encoding=False, theta=0, phi=0,
    transpile=False, project=True, pipeline=True, include_barriers=False)
print(datetime.datetime.now().time())
#%% Extra optimal run
n_cycles = 15
n_shots = 1024*8*3

fid_L_s, P_L_s, time_s = fidelity_from_scratch(n_cycles, n_shots, 
    gate_times=standard_times, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='dm', encoding=False, theta=0, phi=0,
    transpile=False, project=True)
print(datetime.datetime.now().time())
fid_L_f, P_L_f, time_f = fidelity_from_scratch(n_cycles, n_shots, 
    gate_times={'feedback': 6000}, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='dm', encoding=False, theta=0, phi=0,
    transpile=False, project=True)
print(datetime.datetime.now().time())
fid_L_d, P_L_d, time_d = fidelity_from_scratch(n_cycles, n_shots, 
    gate_times={'delay': 4000}, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='dm', encoding=False, theta=0, phi=0,
    transpile=False, project=True)
print(datetime.datetime.now().time())

#%% Calculate parameters

#%% Reformat data
# Reformat as np arrays
fid_L_standard = np.array(fid_L_s)
fid_L_feedback = np.array(fid_L_f)
fid_L_delay = np.array(fid_L_d)
P_L_standard = np.array(P_L_s)
P_L_feedback = np.array(P_L_f)
P_L_delay = np.array(P_L_d)
# Generate x-values
t_standard = np.array([time_s['dm_'+str(i)] for i in range(n_cycles+1)])
t_feedback = np.array([time_f['dm_'+str(i)] for i in range(n_cycles+1)])
t_delay = np.array([time_d['dm_'+str(i)] for i in range(n_cycles+1)])


p0 = (40e3, 0, 0.9) # start with values near those we expect
pars_s, cov_s = scipy.optimize.curve_fit(monoExp, t_standard[1:], fid_standard[1:], p0)
pars_f, cov_f = scipy.optimize.curve_fit(monoExp, t_feedback[1:], fid_feedback[1:], p0)
pars_d, cov_d = scipy.optimize.curve_fit(monoExp, t_delay[1:], fid_delay[1:], p0)

#%% Save data
# Define variables to save/load
function_data = [n_cycles,
                 n_shots,
                 fid_L_standard,
                 P_L_standard,
                 t_standard,
                 pars_s,
                 cov_s]
with open('data/standard_QEC_data.npy', 'wb') as f:
    for data in function_data:
        np.save(f, data)

function_data = [n_cycles,
                 n_shots,
                 fid_L_feedback,
                 P_L_feedback,
                 t_feedback,
                 pars_f,
                 cov_f]
with open('data/feedback_6mus_data.npy', 'wb') as f:
    for data in function_data:
        np.save(f, data)

function_data = [n_cycles,
                 n_shots,
                 fid_L_delay,
                 P_L_delay,
                 t_delay,
                 pars_d,
                 cov_d]
with open('data/delay_4mus_data.npy', 'wb') as f:
    for data in function_data:
        np.save(f, data)
#%%
fig, ax = plt.subplots(1,1, figsize=(7, 5))

fid_standard = np.array(fid_L_s)
fid_standard[1:] = fid_standard[1:]*np.mean(P_L_s[1:])
fid_feedback = np.array(fid_L_f)
fid_feedback[1:] = fid_feedback[1:]*np.mean(P_L_f[1:])
fid_delay = np.array(fid_L_d)
fid_delay[1:] = fid_delay[1:]*np.mean(P_L_d[1:])


# Generate x-values
t_standard = np.array([time_s['dm_'+str(i)] for i in range(n_cycles+1)])
t_feedback = np.array([time_f['dm_'+str(i)] for i in range(n_cycles+1)])
t_delay = np.array([time_d['dm_'+str(i)] for i in range(n_cycles+1)])

# Subplot 1: Target gate times
ax.plot(t_standard, fid_standard, 'o', color='C0', label='Standard times')
ax.plot(t_feedback, fid_feedback, '^', color='C1', label='Feedback, 6 μs')
ax.plot(t_delay, fid_delay, 'v', color='C2', label='Delay, 4 μs')

time_list = [t_standard, t_feedback, t_delay]
fid_list = [fid_standard, fid_feedback, fid_delay]
P_list = [np.mean(P_L_s[1:]), np.mean(P_L_f[1:]), np.mean(P_L_d[1:])]
color_list = ['C0', 'C1', 'C2']
lifetimes = []
errs = []
t = np.linspace(0, 140e3,100)
for i in range(len(time_list)):
    p0 = (40e3, 0, 0.9) # start with values near those we expect
    x = time_list[i]
    y = fid_list[i]
    pars, cov = scipy.optimize.curve_fit(monoExp, x[1:], y[1:], p0)
    T, c, A = pars
    print(T)
    lifetimes.append(T)
    errs.append(np.sqrt(cov[0][0]))
    ax.plot(t, monoExp(t, *pars), ':', color=color_list[i])
ax.text(65000, 0.39,rf'$T_L ={lifetimes[0]/1000:.1f} \pm {errs[0]/1000:.1f}$ μs,  $P_L={P_list[0]:.2f}$', color='C0')
ax.text(75000, 0.34,rf'$T_L ={lifetimes[1]/1000:.1f} \pm {errs[1]/1000:.1f}$ μs,  $P_L={P_list[1]:.2f}$', color='C1')
ax.text(45000, 0.56,rf'$T_L ={lifetimes[2]/1000:.1f} \pm {errs[2]/1000:.1f}$ μs,  $P_L={P_list[2]:.2f}$', color='C2')
ax.set_xticklabels(["0","0", "20", "40", "60", "80", "100", "120", "140"]) 
#ax.set(ylim=(0.0, 1.0))
#ax.ticklabel_format(style='sci',scilimits=(0,0),useMathText=True)
#ax.set_title(r'Expectation value of $|0\rangle$ and $|1\rangle$')
ax.set_xlabel('Time [μs]')
#ax.set_ylabel(r'Average state fidelity of $|0_L\rangle$')
ax.set_ylabel('Probability of remaining in the initial state')
ax.legend()
#%% Single qubit
fid_single_1, time_single_1 = fid_single_qubit(n_cycles, n_shots,
                                gate_times=standard_times,
                                snapshot_type='dm', T1=40e3, T2=60e3,
                                theta=np.pi, phi=0, pauliop='Z')
#%%
fid_single_p, time_single_p = fid_single_qubit(n_cycles, n_shots,
                                gate_times=standard_times,
                                snapshot_type='dm', T1=40e3, T2=60e3,
                                theta=np.pi/2, phi=np.pi/2, pauliop='X')

#%% Reformat data
fid_hex_0 = np.array(fid_hex_0)
P_L_hex_0 = np.array(P_L_hex_0)
P_L_hex_0_mean = np.mean(P_L_hex_0[1:])
fid_hex = fid_hex_0*P_L_hex_0_mean
fid_hex[0] = 1.0

fid_DD_0 = np.array(fid_DD_0)
P_L_DD_0 = np.array(P_L_DD_0)
P_L_DD_0_mean = np.mean(P_L_DD_0[1:])
fid_DD = fid_DD_0*P_L_DD_0_mean
fid_DD[0] = 1.0

fid_pipe_0 = np.array(fid_pipe_0)
P_L_pipe_0 = np.array(P_L_pipe_0)
P_L_pipe_0_mean = np.mean(P_L_pipe_0[1:])
fid_pipe = fid_pipe_0*P_L_pipe_0_mean
fid_pipe[0] = 1.0

fid_hex_0d = np.array(fid_hex_0d)
P_L_hex_0d = np.array(P_L_hex_0d)
P_L_hex_0d_mean = np.mean(P_L_hex_0d[1:])
fid_hexd = fid_hex_0d*P_L_hex_0d_mean
fid_hexd[0] = 1.0

fid_DD_0d = np.array(fid_DD_0d)
P_L_DD_0d = np.array(P_L_DD_0d)
P_L_DD_0d_mean = np.mean(P_L_DD_0d[1:])
fid_DDd = fid_DD_0d*P_L_DD_0d_mean
fid_DDd[0] = 1.0

fid_pipe_0d = np.array(fid_pipe_0d)
P_L_pipe_0d = np.array(P_L_pipe_0d)
P_L_pipe_0d_mean = np.mean(P_L_pipe_0d[1:])
fid_piped = fid_pipe_0d*P_L_pipe_0d_mean
fid_piped[0] = 1.0
#%% Plotting

fig, ax = plt.subplots(1,1, figsize=(7, 5))
x_dis = np.arange(0,n_cycles+1)

# Generate x-values
x_hex = []
x_DD = []
x_pipe = []
x_hexd = []
x_DDd = []
x_piped = []

for i in range(len(time_hex_0)-1):
    x_hex.append(time_hex_0['dm_'+str(i)])
    x_DD.append(time_DD_0['dm_'+str(i)])
    x_pipe.append(time_pipe_0['dm_'+str(i)])
    x_hexd.append(time_hex_0d['dm_'+str(i)])
    x_DDd.append(time_DD_0d['dm_'+str(i)])
    x_piped.append(time_pipe_0d['dm_'+str(i)])

# Subplot 1: Target gate times
ax.plot(x_hex, fid_hex, 'o', color='C0', label='Hexagonal layout', zorder=10)
ax.plot(x_DD, fid_DD, 's', color='C1', label='Square layout', zorder=5)
ax.plot(x_pipe, fid_pipe, '^', color='C2', label='Pipeline (Hexagonal)', zorder=0)

# Subplot 1: Target gate times
ax.plot(x_hexd, fid_hexd, 'o', color='C0', zorder=10)
ax.plot(x_DDd, fid_DDd, 's', color='C1', zorder=5)
ax.plot(x_piped, fid_piped, '^', color='C2', zorder=0)
 
time_list = [x_hex, x_DD, x_pipe, x_hexd, x_DDd, x_piped]
fid_list = [fid_hex, fid_DD, fid_pipe, fid_hexd, fid_DDd, fid_piped]
t = np.linspace(x_DD[0], x_DD[-1], 100)
ax.plot(t, monoExp(t,40e3,0,1), '--k', label='Single qubit', zorder=-5)
lifetimes = [40000]
for i in range(len(time_list)):
    p0 = (40e3, 0, 0.9) # start with values near those we expect
    x= time_list[i]
    y = fid_list[i]
    pars, cov = scipy.optimize.curve_fit(monoExp, x[1:], y[1:], p0)
    T, c, A = pars
    print(T)
    lifetimes.append(T)
    #ax.plot(t, monoExp(t, *pars), ':', color='C'+str(i))
#ax.text(30000, 0.35,rf'$T_L ={lifetimes[0]/1000:.0f}$ μs', color='k')
#ax.text(45000, 0.55,rf'$T_L ={lifetimes[1]/1000:.0f}$ μs', color='C0')
#ax.text(45000, 0.38,rf'$T_L ={lifetimes[2]/1000:.0f}$ μs', color='C1')
#ax.text(45000, 0.50,rf'$T_L ={lifetimes[3]/1000:.0f}$ μs', color='C2')
ax.set_xticklabels(["0","0", "10", "20", "30", "40", "50"]) 
#ax.set(ylim=(0.0, 1.0))
#ax.ticklabel_format(style='sci',scilimits=(0,0),useMathText=True)
#ax.set_title(r'Expectation value of $|0\rangle$ and $|1\rangle$')
ax.set_xlabel('Time [μs]')
#ax.set_ylabel(r'Average state fidelity of $|0_L\rangle$')
ax.set_ylabel('Probability of remaining in the initial state')
ax.legend()

#ax0 = ax.twinx()
#
#ax0.plot(x_hex, -1*np.array(fid_single_1), '--k')
#ax0.set(ylim=(1.0,0.0))
#ax0.set_yticklabels(["1.0","0.75","0.5","0.25","0.0"]) 
#ax0.set_ylabel(r'Physical $|1\rangle$ probability')

#%% Split y-axis
fig, ax = plt.subplots(1,1, figsize=(7, 5))
x_dis = np.arange(0,n_cycles+1)

ax.plot(x_hex, exp_hex_p, '-o', color='C0', label='Hexagonal layout', zorder=10)
ax.plot(x_hex, exp_hex_m, '-o', color='C0', zorder=10)
ax.plot(x_DD, exp_DD_p, '-o', color='C1', label='Double diamond layout', zorder=5)
ax.plot(x_DD, exp_DD_m, '-o', color='C1', zorder=5)
ax.plot(x_pipe, exp_pipe_p, '-o', color='C2', label='Pipeline (Hexagonal)', zorder=0)
ax.plot(x_pipe, exp_pipe_m, '-o', color='C2', zorder=0)

ax.set(ylim=(-1.0, 1.0))
ax.plot(x_hex, test, '--k', label='Single qubit')
ax.ticklabel_format(style='sci',scilimits=(0,0),useMathText=True)
ax.set_title(r'Expectation value of $|+\rangle$ and $|-\rangle$')
ax.set_xlabel('Time [ns]')
ax.set_ylabel(r'Expectation value $\langle X_L\rangle$')
ax.legend(loc="upper right")

ax1 = ax.twinx()

ax1.plot(x_hex, -1*test, '--k')
ax1.set(ylim=(-1.0,1.0))
ax1.set_yticklabels(["1.0","0.875","0.75","0.625","0.5","0.625","0.75","0.875","1.0"]) 
ax1.set_ylabel(r'Physical $|+\rangle$ probability')

#%% No-reset vs reset runs (Post-Processing)
n_cycles=12
n_shots=1024*4
fid_rec, time_rec =fidelity_from_scratch(n_cycles, n_shots, gate_times=standard_times,
    reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='dm', encoding=False, theta=0, phi=0,
    transpile=False, pauliop='ZZZZZ', device_properties=WACQT_device_properties,
    device=None, simulator_type='density_matrix',move_feedback_delay=True)
print('Check!')
fid_r, time_r =fidelity_from_scratch(n_cycles, n_shots, gate_times=standard_times,
    reset=True, data_process_type='post_process',
    idle_noise=True, snapshot_type='dm', encoding=False, theta=0, phi=0,
    transpile=False, pauliop='ZZZZZ', device_properties=WACQT_device_properties,
    device=None, simulator_type='density_matrix',move_feedback_delay=True)
print('Check!')
fid, time =fidelity_from_scratch(n_cycles, n_shots, gate_times=standard_times,
    reset=False, data_process_type='post_process',
    idle_noise=True, snapshot_type='dm', encoding=False, theta=0, phi=0,
    transpile=False, pauliop='ZZZZZ', device_properties=WACQT_device_properties,
    device=None, simulator_type='density_matrix',move_feedback_delay=True)
print('Check!')
#time = [time['dm_con_'+str(i)] for i in range(n_cycles+1)]
#%% Plots
time = [time_rec['dm_'+str(i)] for i in range(n_cycles+1)]
fig, ax = plt.subplots(1,1, figsize=(7, 5))

# Subplot 1: Target gate times
ax.plot(time, fid_rec, 'o', color='C0', label='Standard QEC (With reset)', zorder=10)
ax.plot(time, fid_r, 's', color='C1', label='With reset post-process', zorder=5)
ax.plot(time, fid, '^', color='C2', label='No-reset post-process', zorder=0)

ax.set_xlabel('Time [μs]')
#ax.set_ylabel(r'Average state fidelity of $|0_L\rangle$')
ax.set_ylabel('Probability of remaining in the initial state')
ax.legend()
 
fid_list = [fid_rec, fid_r, fid]
t = np.linspace(time[0], time[-1], 100)
lifetimes = []
for i in range(len(fid_list)):
    p0 = (40e3, 0, 0.9) # start with values near those we expect
    x= time
    y = fid_list[i]
    pars, cov = scipy.optimize.curve_fit(monoExp, x[1:], y[1:], p0)
    T, c, A = pars
    lifetimes.append(T)
    ax.plot(t, monoExp(t, *pars), ':', color='C'+str(i))
    print(T)
    print(c)
    print(np.sqrt(cov[0][0])) # One standard deviation
ax.text(20000, 0.53,rf'$T_L ={lifetimes[0]/1000:.0f} \pm 1$ μs', color='C0')
ax.text(30000, 0.6,rf'$T_L ={lifetimes[1]/1000:.0f} \pm 1$ μs', color='C1')
ax.text(32000, 0.57,rf'$T_L ={lifetimes[2]/1000:.0f} \pm 1$ μs', color='C2')
#%%
ax.set_xticklabels(["0","0", "10", "20", "30", "40", "50"]) 
#ax.set(ylim=(0.0, 1.0))
#ax.ticklabel_format(style='sci',scilimits=(0,0),useMathText=True)
#ax.set_title(r'Expectation value of $|0\rangle$ and $|1\rangle$')
ax.set_xlabel('Time [μs]')
#ax.set_ylabel(r'Average state fidelity of $|0_L\rangle$')
ax.set_ylabel('Probability of remaining in the initial state')
ax.legend()

#ax0 = ax.twinx()
#
#ax0.plot(x_hex, -1*np.array(fid_single_1), '--k')
#ax0.set(ylim=(1.0,0.0))
#ax0.set_yticklabels(["1.0","0.75","0.5","0.25","0.0"]) 
#ax0.set_ylabel(r'Physical $|1\rangle$ probability')
