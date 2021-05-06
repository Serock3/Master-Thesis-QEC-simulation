# %% Import modules
# TODO: Check which modules are actually used
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import itertools
from qiskit import *

# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import thermal_relaxation_error
from qiskit.providers.aer.library import save_density_matrix, save_expectation_value                                        

from qiskit.quantum_info import partial_trace
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import state_fidelity

# Our own files
from simulator_program.custom_noise_models import (thermal_relaxation_model,
    thermal_relaxation_model_V2,
    WACQT_target_times,
    WACQT_demonstrated_times)
from simulator_program.data_analysis_tools import *
from simulator_program.custom_transpiler import *
from simulator_program.stabilizers import *
from simulator_program.post_select import *
from simulator_program.post_process import *
from simulator_program.idle_noise import *

#%% Run parameter sweeps
T1_space = [40e3, 50e3, 60e3, 70e3, 80e3]
T2_space = [40e3, 50e3, 60e3, 70e3, 80e3]
single_qubit_gate_time_space = [20]
two_qubit_gate_time_space = [100]
measure_time_space = [300]
n_cycles = 8
n_shots = 2048

kwargs = {'reset': True, 'data_process_type': 'recovery', 'idle_noise': True, 
          'snapshot_type': 'exp', 'encoding': False, 'theta': 0, 'phi': 0,
          'transpile': False}
#%%
T1_space=[40e3, 60e3] 
T2_space=[40e3,60e3]
error_rates, MSE = sweep_parameter_space(T1_space, T2_space, 
                                         single_qubit_gate_time_space,
                                         two_qubit_gate_time_space,
                                         measure_time_space,
                                         n_cycles, n_shots,
                                         save='Testy',
                                         time_axis=True, **kwargs)
#%% Run single qubit parameter sweep
kwargs = {'data_process_type': 'recovery', 'idle_noise':True, 
          'snapshot_type': 'dm', 'encoding': False, 'theta': np.pi/2, 'phi': np.pi/2,
          'transpile': False}
error_rates_single2, MSE_single2 = sweep_parameter_space(T1_space, T2_space, 
                                         single_qubit_gate_time_space,
                                         two_qubit_gate_time_space,
                                         measure_time_space,
                                         n_cycles, n_shots, 
                                         single_qubit=True,
                                         **kwargs)
#%% Sweeps before bed
T1_space = [40e3, 50e3, 60e3, 70e3, 80e3]
T2_space = [40e3, 50e3, 60e3, 70e3, 80e3]
single_qubit_gate_time_space = [20]
two_qubit_gate_time_space = [100]
measure_time_space = [300]
n_cycles = 8
n_shots = 2048
#%%
error_rates_T1T2, MSE_T1T2 = sweep_parameter_space(T1_space, T2_space, 
                                         single_qubit_gate_time_space,
                                         two_qubit_gate_time_space,
                                         measure_time_space,
                                         n_cycles, n_shots,
                                         save='T1T2_cyc',
                                         **kwargs)
print('Whoooah, we re halfway theeeeereee')
error_rates_T2T2t, MSE_T1T2t = sweep_parameter_space(T1_space, T2_space, 
                                         single_qubit_gate_time_space,
                                         two_qubit_gate_time_space,
                                         measure_time_space,
                                         n_cycles, n_shots,
                                         save='T1T2_time',
                                         time_axis=True, **kwargs)
#%%
import datetime
T1_space = [40e3]
T2_space = [60e3]
single_qubit_gate_time_space = [20]
two_qubit_gate_time_space = [50, 60, 70, 80, 90, 100, 110]
measure_time_space = [100, 140, 180, 220, 260, 300, 340]
n_cycles = 14
n_shots = 2048*4
#%%
error_rates_tgate, MSE_tgate = sweep_parameter_space(T1_space, T2_space, 
                                         single_qubit_gate_time_space,
                                         two_qubit_gate_time_space,
                                         measure_time_space,
                                         n_cycles, n_shots,
                                         save='tgate_cyc',
                                         **kwargs)
print('LIIVING ON A PRAYERRR')
#%%
error_rates_tgatet, MSE_tgatet = sweep_parameter_space(T1_space, T2_space, 
                                         single_qubit_gate_time_space,
                                         two_qubit_gate_time_space,
                                         measure_time_space,
                                         n_cycles, n_shots,
                                         save='tgate_time_2',
                                         time_axis=True, **kwargs)
#%%
print(datetime.datetime.now().time())
#%% Run single qubit parameter sweep
kwargs = {'data_process_type': 'recovery', 'idle_noise':True, 
          'snapshot_type': 'dm', 'encoding': False, 'theta': np.pi, 'phi': 0,
          'transpile': False}
error_rates_single, MSE_single = sweep_parameter_space(T1_space, T2_space, 
                                         single_qubit_gate_time_space,
                                         two_qubit_gate_time_space,
                                         measure_time_space,
                                         n_cycles, n_shots, 
                                         single_qubit=True,
                                         **kwargs)
kwargs = {'data_process_type': 'recovery', 'idle_noise':True, 
          'snapshot_type': 'dm', 'encoding': False, 'theta': np.pi/2, 'phi': np.pi/2,
          'transpile': False}
error_rates_single2, MSE_single2 = sweep_parameter_space(T1_space, T2_space, 
                                         single_qubit_gate_time_space,
                                         two_qubit_gate_time_space,
                                         measure_time_space,
                                         n_cycles, n_shots, 
                                         single_qubit=True,
                                         **kwargs)                             

#%%
kwargs = {'data_process_type': 'recovery', 'idle_noise':True, 
          'snapshot_type': 'dm', 'encoding': False, 'theta': np.pi, 'phi': 0,
          'transpile': False}
n_cycles=8
n_shots = 2048
error_rates_single2, MSE_single2 = sweep_parameter_space(T1_space, T2_space, 
                                         single_qubit_gate_time_space,
                                         two_qubit_gate_time_space,
                                         measure_time_space,
                                         n_cycles, n_shots, 
                                         single_qubit=True,
                                         time_axis=True, **kwargs)    
#%% Load in data from comparing T1 T2s
X_mesh, Y_mesh = np.meshgrid(T1_space, T2_space)
E_mesh = -1*np.reshape(error_rates_T1T2, (5,5)).T
E_mesh_single = -1*np.reshape(error_rates_single, (5,5)).T
E_mesh_single2 = -1*np.reshape(error_rates_single2, (5,5)).T
E_list = [E_mesh, E_mesh_single, E_mesh_single2]

Nr = 1
Nc = 3
fig, ax = plt.subplots(Nr,Nc, figsize=(10,4))
maps = []
for i in range(Nc):
    HM = ax[i].contourf(X_mesh*1e-3, Y_mesh*1e-3, E_list[i])
    maps.append(HM)

ax[2].set_xlabel(r'$T_1$ [$\mu s$]')
ax[1].set_xlabel(r'$T_1$ [$\mu s$]')
ax[0].set_xlabel(r'$T_1$ [$\mu s$]')
#ax[1].set_ylabel(r'$T_2$ [$\mu s$]')
ax[0].set_ylabel(r'$T_2$ [$\mu s$]')
ax[0].set_title(r'Logical qubit $|0_L\rangle$')
ax[1].set_title(r'Single qubit ($T_1$ limited)')
ax[2].set_title(r'Single qubit ($T_2$ limited)')
E_min = np.min(np.array([np.min(E_mesh), np.min(E_mesh_single), np.min(E_mesh_single2)]))
E_max = np.max(np.array([np.max(E_mesh), np.max(E_mesh_single), np.max(E_mesh_single2)]))

norm = colors.Normalize(vmin=E_min, vmax=E_max)
for hm in maps:
    hm.set_norm(norm)
cbar = fig.colorbar(maps[0], ax=ax, orientation='horizontal', fraction=.1)
cbar.set_label(r'Error rate per cycle', labelpad=0, y=1.05, rotation=0)

#ax[0].set(ylim=(140,340))


#%% Plotting gate times
X_tmp, Y_tmp = np.meshgrid([70,80,90,100,110,120],[140,180,220,260,300,340])
X_mesh, Y_mesh = np.meshgrid(two_qubit_gate_time_space, measure_time_space)
E_mesh = -1*np.reshape(error_rates_tgate, (6,6)).T
E_mesht = -1*np.reshape(error_rates_tgatet, (7,7)).T

#E_mesh_single = -1*np.reshape(error_rates_single, (6,6)).T
#E_mesh_single2 = -1*np.reshape(error_rates_single2, (6,6)).T
#E_list = [E_mesh, E_mesh_single, E_mesh_single2]

Nr = 1
Nc = 2
fig, ax = plt.subplots(Nr,Nc, figsize=(10,6))
maps = []
#for i in range(Nc):
#    HM = ax[i].contourf(X_mesh, Y_mesh, E_list[i])
#    maps.append(HM)

HM = ax[0].contourf(X_tmp, Y_tmp, E_mesh)
HMt = ax[1].contourf(X_mesh, Y_mesh, E_mesht)
maps.append(HM)
maps.append(HMt)
# Draw constant cycle time
c_list = [400, 450, 500, 550, 600, 650, 700, 750, 800]
x = np.linspace(70,120,100)
for c in c_list:
    y = -4*x + c
    ax[0].plot(x,y,'--k',linewidth=0.8)
x = np.linspace(50,110,100)
for c in c_list:
    y = -4*x + c
    ax[1].plot(x,y,'--k',linewidth=0.8)

ax[0].set_xlabel(r'Two-qubit gate time, $t_{2qb}$ [$ns$]')
ax[1].set_xlabel(r'Two-qubit gate time, $t_{2qb}$ [$ns$]')

ax[0].set_ylabel(r'Measurement time, $t_{meas}$ [$ns$]')
ax[0].set_title(r'Error rate per cycle')
ax[1].set_title(r'Error rate per time')


E_min = np.min(E_mesh)
E_max = np.max(E_mesh)

#norm = colors.Normalize(vmin=E_min, vmax=E_max)
#for hm in maps:
#    hm.set_norm(norm)
cbar = fig.colorbar(maps[0], ax=ax[0], orientation='horizontal', fraction=.1)
cbar.set_label(r'Error rate per cycle', labelpad=-0, y=1.05, rotation=0)
cbar2 = fig.colorbar(maps[1], ax=ax[1], orientation='horizontal', fraction=.1)

ax[0].set(ylim=(140,340))
ax[1].set(ylim=(140,340))