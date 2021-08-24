# %% Import modules
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from qiskit import *


# Our own files
from simulator_program.custom_noise_models import standard_times
from simulator_program.data_analysis_tools import *
from simulator_program.custom_transpiler import *
from simulator_program.stabilizers import *
from simulator_program.post_select import *
from simulator_program.post_process import *
from simulator_program.idle_noise import *

# Temporary for showing current time
import datetime
# %% Run parameter sweeps
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
# %%

error_rates, MSE = sweep_parameter_space(T1_space, T2_space,
                                         single_qubit_gate_time_space,
                                         two_qubit_gate_time_space,
                                         measure_time_space,
                                         n_cycles, n_shots,
                                         save='Testy',
                                         time_axis=True, **kwargs)
# %% Run single qubit parameter sweep
kwargs = {'data_process_type': 'recovery', 'idle_noise': True,
          'snapshot_type': 'dm', 'encoding': False, 'theta': np.pi/2, 'phi': np.pi/2,
          'transpile': False}
error_rates_single2, MSE_single2 = sweep_parameter_space(T1_space, T2_space,
                                                         single_qubit_gate_time_space,
                                                         two_qubit_gate_time_space,
                                                         measure_time_space,
                                                         n_cycles, n_shots,
                                                         single_qubit=True,
                                                         **kwargs)
# %% Sweeps before bed
T1_space = [40e3, 50e3, 60e3, 70e3, 80e3, 90e3, 100e3]
T2_space = [40e3, 50e3, 60e3, 70e3, 80e3, 90e3, 100e3]
single_qubit_gate_time_space = [20]
two_qubit_gate_time_space = [100]
measure_time_space = [300]
feedback_time_space = [350]
n_cycles = 14
n_shots = 2048*4
kwargs = {'reset': True, 'data_process_type': 'recovery', 'idle_noise': True,
          'snapshot_type': 'exp', 'encoding': False, 'theta': 0, 'phi': 0,
          'transpile': False}
# %%
# error_rates_T1T2, MSE_T1T2 = sweep_parameter_space(T1_space, T2_space,
#                                         single_qubit_gate_time_space,
#                                         two_qubit_gate_time_space,
#                                         measure_time_space,
#                                         n_cycles, n_shots,
#                                         save='T1T2_cyc',
#                                         **kwargs)
error_rates_T1T2t, MSE_T1T2t = sweep_parameter_space(T1_space, T2_space,
                                                     single_qubit_gate_time_space,
                                                     two_qubit_gate_time_space,
                                                     measure_time_space,
                                                     n_cycles, n_shots,
                                                     save='T1T2_time_2',
                                                     time_axis=True, **kwargs)
# %% Early exploration of gate/measurement times

T1_space = [40e3]
T2_space = [60e3]
single_qubit_gate_time_space = [20]
two_qubit_gate_time_space = [50, 60, 70, 80, 90, 100, 110]
measure_time_space = [100, 140, 180, 220, 260, 300, 340]
n_cycles = 14
n_shots = 2048*2

# %% New exploration of gate/measurement times

T1_space = [40e3]
T2_space = [40e3]
single_qubit_gate_time_space = [20]
two_qubit_gate_time_space = [50, 100, 150, 200, 250, 300, 350]
measure_time_space = [100, 300, 500, 700, 900, 1100, 1300]
n_cycles = 14
n_shots = 2048*4

# %%
error_rates_tgate, MSE_tgate = sweep_parameter_space(T1_space, T2_space,
                                                     single_qubit_gate_time_space,
                                                     two_qubit_gate_time_space,
                                                     measure_time_space,
                                                     n_cycles, n_shots,
                                                     save='tgate_cyc',
                                                     **kwargs)
# %% Error rate gates 0
kwargs = {'reset': True, 'data_process_type': 'recovery', 'idle_noise': True,
          'snapshot_type': 'exp', 'encoding': False, 'theta': 0, 'phi': 0,
          'transpile': False}
error_rates_tgatet, MSE_tgatet = sweep_parameter_space(T1_space, T2_space,
                                                       single_qubit_gate_time_space,
                                                       two_qubit_gate_time_space,
                                                       measure_time_space,
                                                       n_cycles, n_shots,
                                                       save='tgate_time_3',
                                                       time_axis=True, **kwargs)

# %% Sunday 10/5 runs

T1_space = [40e3]
T2_space = [40e3]
single_qubit_gate_time_space = [20]
two_qubit_gate_time_space = [50, 100, 150, 200, 250, 300, 350]
measure_time_space = [100, 300, 500, 700, 900, 1100, 1300]
n_cycles = 14
n_shots = 2048*4
kwargs = {'reset': True, 'data_process_type': 'recovery', 'idle_noise': True,
          'snapshot_type': 'exp', 'encoding': False, 'theta': 0, 'phi': 0,
          'transpile': False}
error_rates_tgate4040, MSE_tgatet = sweep_parameter_space(T1_space, T2_space,
                                                          single_qubit_gate_time_space,
                                                          two_qubit_gate_time_space,
                                                          measure_time_space,
                                                          n_cycles, n_shots,
                                                          save='tgate_time_sun4040',
                                                          time_axis=True, **kwargs)
print('Thats one!')
T1_space = [60e3]
T2_space = [60e3]
error_rates_tgate6060, MSE_tgatet = sweep_parameter_space(T1_space, T2_space,
                                                          single_qubit_gate_time_space,
                                                          two_qubit_gate_time_space,
                                                          measure_time_space,
                                                          n_cycles, n_shots,
                                                          save='tgate_time_sun6060',
                                                          time_axis=True, **kwargs)
T1_space = [80e3]
T2_space = [80e3]
print('Thats two!')
error_rates_tgate8080, MSE_tgatet = sweep_parameter_space(T1_space, T2_space,
                                                          single_qubit_gate_time_space,
                                                          two_qubit_gate_time_space,
                                                          measure_time_space,
                                                          n_cycles, n_shots,
                                                          save='tgate_time_sun8080',
                                                          time_axis=True, **kwargs)
print('And thats three!')

# %% Single qubit parameter sweep
n_cycles = 14
n_shots = 2048*4

kwargs = {'data_process_type': 'recovery', 'idle_noise': True,
          'snapshot_type': 'dm', 'encoding': False, 'theta': np.pi, 'phi': 0,
          'transpile': False}
error_rates_single, MSE_single = sweep_parameter_space(T1_space, T2_space,
                                                       single_qubit_gate_time_space,
                                                       two_qubit_gate_time_space,
                                                       measure_time_space,
                                                       n_cycles, n_shots,
                                                       single_qubit=True,
                                                       time_axis=True, **kwargs)
kwargs = {'data_process_type': 'recovery', 'idle_noise': True,
          'snapshot_type': 'dm', 'encoding': False, 'theta': np.pi/2, 'phi': np.pi/2,
          'transpile': False}
error_rates_single2, MSE_single2 = sweep_parameter_space(T1_space, T2_space,
                                                         single_qubit_gate_time_space,
                                                         two_qubit_gate_time_space,
                                                         measure_time_space,
                                                         n_cycles, n_shots,
                                                         single_qubit=True,
                                                         time_axis=True, **kwargs)

# %%
kwargs = {'data_process_type': 'recovery', 'idle_noise': True,
          'snapshot_type': 'dm', 'encoding': False, 'theta': np.pi/2, 'phi': np.pi/2,
          'transpile': False}
n_cycles = 8
n_shots = 2048*2
error_rates_single, MSE_single = sweep_parameter_space(T1_space, T2_space,
                                                       single_qubit_gate_time_space,
                                                       two_qubit_gate_time_space,
                                                       measure_time_space,
                                                       n_cycles, n_shots,
                                                       single_qubit=True,
                                                       time_axis=True, **kwargs)
# %% 2012-05-14 perfect stab runs
rates_T1T2_feedback, MSE_T1T2_feedback = sweep_parameter_space(T1_space, T2_space,
                                                               single_qubit_gate_time_space,
                                                               two_qubit_gate_time_space,
                                                               measure_time_space, feedback_time_space,
                                                               n_cycles, n_shots,
                                                               save='T1T2_perfect_feedback',
                                                               time_axis=True, perfect_stab=True,
                                                               **kwargs)
rates_T1T2, MSE_T1T2 = sweep_parameter_space(T1_space, T2_space,
                                             single_qubit_gate_time_space,
                                             two_qubit_gate_time_space,
                                             measure_time_space, feedback_time_space,
                                             n_cycles, n_shots,
                                             save='T1T2_with_feedback',
                                             time_axis=True, **kwargs)
# %% Compare T1 and T2
X_mesh, Y_mesh = np.meshgrid(T1_space, T2_space)
E_mesh1 = -1*np.reshape(rates_T1T2, (7, 7)).T
E_mesh2 = -1*np.reshape(rates_T1T2_feedback, (7, 7)).T

# Convert to lifetime (Outdated since new sweeps does this automatically)
for i in range(len(E_mesh1)):
    for j in range(len(E_mesh1)):
        if E_mesh1[i, j] != 0.:
            E_mesh1[i, j] = E_mesh1[i, j]**-1
        if E_mesh2[i, j] != 0.:
            E_mesh2[i, j] = E_mesh2[i, j]**-1

E_list = [E_mesh1, E_mesh2]

Nr = 1
Nc = 2
fig, ax = plt.subplots(Nr, Nc, figsize=(10, 4))
maps = []
for i in range(Nc):
    HM = ax[i].imshow(E_list[i], extent=[35, 105, 35, 105],
                      cmap='magma', origin='lower')
    maps.append(HM)
    ax[i].set_xticks([40, 50, 60, 70, 80, 90, 100])
    ax[i].set_yticks([100, 90, 80, 70, 60, 50, 40])
ax[1].set_xlabel(r'$T_1$ [$\mu s$]')
ax[0].set_xlabel(r'$T_1$ [$\mu s$]')
ax[1].set_ylabel(r'$T_2$ [$\mu s$]')
ax[0].set_ylabel(r'$T_2$ [$\mu s$]')

ax[0].set_title(r'Standard QEC')
ax[1].set_title(r'Perfect syndrome extraction')

# Normalize color scale not to include the zeros at T2 > 2T1
for i in range(Nc):
    E_min = np.min(np.array([np.min(E_list[i][E_list[i] != 0])]))
    E_max = np.max(np.array([np.max(E_list[i][E_list[i] != 0])]))
    norm = colors.Normalize(vmin=E_min, vmax=E_max)
    maps[i].set_norm(norm)

cbar0 = fig.colorbar(
    maps[0], ax=ax[0], orientation='horizontal', fraction=.1, pad=0.17)
cbar0.set_label(r'Qubit lifetime $[\mu s]$', labelpad=0, y=1.20, rotation=0)

cbar1 = fig.colorbar(
    maps[1], ax=ax[1], orientation='horizontal', fraction=.1, pad=0.17)
cbar1.set_label(r'Qubit lifetime $[\mu s]$', labelpad=0, y=1.20, rotation=0)


# %% Plotting gate times
#X_tmp, Y_tmp = np.meshgrid([70,80,90,100,110,120],[140,180,220,260,300,340])
X_mesh, Y_mesh = np.meshgrid(two_qubit_gate_time_space, measure_time_space)
#E_mesh = -1*np.reshape(error_rates_tgate, (6,6)).T
E_mesht = np.reciprocal(-1*np.reshape(error_rates_tgatet, (7, 7)).T)

#E_mesh_single = -1*np.reshape(error_rates_single, (6,6)).T
#E_mesh_single2 = -1*np.reshape(error_rates_single2, (6,6)).T
#E_list = [E_mesh, E_mesh_single, E_mesh_single2]

Nr = 1
Nc = 2
fig, ax = plt.subplots(Nr, Nc, figsize=(10, 6))
maps = []
# for i in range(Nc):
#    HM = ax[i].contourf(X_mesh, Y_mesh, E_list[i])
#    maps.append(HM)

#HM = ax[0].contourf(X_tmp, Y_tmp, E_mesh)
HMt = ax[1].contourf(X_mesh, Y_mesh, E_mesht)
# maps.append(HM)
maps.append(HMt)
# Draw constant cycle time
c_list = [400, 500, 600, 700, 800, 1000, 1200, 1400]
x = np.linspace(50, 350, 100)
for c in c_list:
    y = -4*x + c
    ax[0].plot(x, y, '--k', linewidth=0.8)
x = np.linspace(50, 350, 100)
for c in c_list:
    y = -4*x + c
    ax[1].plot(x, y, '--k', linewidth=0.8)

ax[0].set_xlabel(r'Two-qubit gate time, $t_{2qb}$ [$ns$]')
ax[1].set_xlabel(r'Two-qubit gate time, $t_{2qb}$ [$ns$]')

ax[0].set_ylabel(r'Measurement time, $t_{meas}$ [$ns$]')
ax[0].set_title(r'Error rate per cycle')
ax[1].set_title(r'Error rate per time')


#E_min = np.min(E_mesh)
#E_max = np.max(E_mesh)

#norm = colors.Normalize(vmin=E_min, vmax=E_max)
# for hm in maps:
#    hm.set_norm(norm)
cbar = fig.colorbar(maps[0], ax=ax[0], orientation='horizontal', fraction=.1)
cbar.set_label(r'Error rate per cycle', labelpad=-0, y=1.05, rotation=0)
#cbar2 = fig.colorbar(maps[1], ax=ax[1], orientation='horizontal', fraction=.1)

ax[0].set(ylim=(100, 700))
ax[1].set(ylim=(100, 700))

# %% Sweeping gate times

# The scaling of gate times investigated
scalings = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
gate_times = scale_gate_times(gate_times=standard_times, scalings=scalings)
# %%
T1_space = [40e3]
T2_space = [60e3]
single_qubit_gate_time_space = [20]
two_qubit_gate_time_space = [100]
measure_time_space = [300]
feedback_time_space = [350]
n_cycles = 15
n_shots = 1024*8*6*22
kwargs = {'reset': True, 'data_process_type': 'recovery', 'idle_noise': True,
          'snapshot_type': 'dm', 'encoding': False, 'theta': 0, 'phi': 0,
          'transpile': False, 'simulator_type': 'statevector'}
# %%
# Scale all gates simultaneously
lifetime_scale_all_gates = np.zeros(len(scalings))
var_scale_measure = np.zeros(len(scalings))
for i in range(len(scalings)):
    lifetime, var = sweep_parameter_space(T1_space, T2_space,
                                          gate_times['single_qubit_gate'][i],
                                          gate_times['two_qubit_gate'][i],
                                          gate_times['measure'][i],
                                          gate_times['feedback'][i],
                                          n_cycles, n_shots,
                                          time_axis=True, **kwargs)
    lifetime_scale_all_gates[i] = lifetime
    var_scale_all_gates[i] = var
    print(datetime.datetime.now().time())
np.save('lifetime_scale_all_gates', lifetime_scale_all_gates)
np.save('var_scale_all_gates', var_scale_all_gates)
# %%
# Scale single-qubit gates, keeping others constant
scalings = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25,
            1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
gate_times = scale_gate_times(gate_times=standard_times, scalings=scalings)
lifetime_scale_single_qubit_gates = np.zeros(len(scalings))
var_scale_single_qubit_gates = np.zeros(len(scalings))
for i in range(len(scalings)):
    lifetime, var = sweep_parameter_space(T1_space, T2_space,
                                          gate_times['single_qubit_gate'][i],
                                          two_qubit_gate_time_space,
                                          measure_time_space,
                                          feedback_time_space,
                                          n_cycles, n_shots,
                                          time_axis=True, **kwargs)
    lifetime_scale_single_qubit_gates[i] = lifetime
    var_scale_single_qubit_gates[i] = var
    print(datetime.datetime.now().time())
np.save('lifetime_scale_single_qubit_gates', lifetime_scale_single_qubit_gates)
np.save('var_scale_single_qubit_gates', var_scale_single_qubit_gates)
# %%
# Scale two-qubit gates, keeping others constant
scalings = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25,
            1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
gate_times = scale_gate_times(gate_times=standard_times, scalings=scalings)
lifetime_scale_two_qubit_gates = np.zeros(len(scalings))
var_scale_two_qubit_gates = np.zeros(len(scalings))
for i in range(len(scalings)):
    lifetime, var = sweep_parameter_space(T1_space, T2_space,
                                          single_qubit_gate_time_space,
                                          gate_times['two_qubit_gate'][i],
                                          measure_time_space,
                                          feedback_time_space,
                                          n_cycles, n_shots,
                                          time_axis=True, **kwargs)
    lifetime_scale_two_qubit_gates[i] = lifetime
    var_scale_two_qubit_gates[i] = var
    print(datetime.datetime.now().time())
np.save('lifetime_scale_two_qubit_gates', lifetime_scale_two_qubit_gates)
np.save('var_scale_two_qubit_gates', var_scale_two_qubit_gates)
# %%
# Scale measure time, keeping others constant
scalings = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25,
            1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
gate_times = scale_gate_times(gate_times=standard_times, scalings=scalings)
lifetime_scale_measure = np.zeros(len(scalings))
var_scale_measure = np.zeros(len(scalings))
for i in range(len(scalings)):
    lifetime, var = sweep_parameter_space(T1_space, T2_space,
                                          single_qubit_gate_time_space,
                                          two_qubit_gate_time_space,
                                          gate_times['measure'][i],
                                          feedback_time_space,
                                          n_cycles, n_shots,
                                          time_axis=True, **kwargs)
    lifetime_scale_measure[i] = lifetime
    var_scale_measure[i] = var
    print(datetime.datetime.now().time())
np.save('lifetime_scale_measure', lifetime_scale_measure)
np.save('var_scale_measure', var_scale_measure)
# %%
# Scale feedback, keeping others constant
scalings = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25,
            1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
gate_times = scale_gate_times(gate_times=standard_times, scalings=scalings)
lifetime_scale_feedback = np.zeros(len(scalings))
var_scale_feedback = np.zeros(len(scalings))
for i in range(len(scalings)):
    lifetime, var = sweep_parameter_space(T1_space, T2_space,
                                          single_qubit_gate_time_space,
                                          two_qubit_gate_time_space,
                                          measure_time_space,
                                          gate_times['feedback'][i],
                                          n_cycles, n_shots,
                                          time_axis=True, **kwargs)
    lifetime_scale_feedback[i] = lifetime
    var_scale_feedback[i] = var
    print(datetime.datetime.now().time())
np.save('lifetime_scale_feedback', lifetime_scale_feedback)
np.save('var_scale_feedback', var_scale_feedback)

# %%
# More scalings of feedback
scalings = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
gate_times = scale_gate_times(gate_times=standard_times, scalings=scalings)
lifetime_scale_feedback_extra = np.zeros(len(scalings))
var_scale_feedback_extra = np.zeros(len(scalings))
for i in range(len(scalings)):
    lifetime, var = sweep_parameter_space(T1_space, T2_space,
                                          single_qubit_gate_time_space,
                                          two_qubit_gate_time_space,
                                          measure_time_space,
                                          gate_times['feedback'][i],
                                          n_cycles, n_shots,
                                          time_axis=True, **kwargs)
    lifetime_scale_feedback_extra[i] = lifetime
    var_scale_feedback_extra[i] = var
    print(datetime.datetime.now().time())
np.save('lifetime_scale_feedback_extra', lifetime_scale_feedback_extra)
np.save('var_scale_feedback_extra', var_scale_feedback_extra)
# %% Even more scalings of feedback
scalings = [3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.0]
gate_times = scale_gate_times(gate_times=standard_times, scalings=scalings)
lifetime_scale_feedback_extra = np.zeros(len(scalings))
var_scale_feedback_extra = np.zeros(len(scalings))
for i in range(len(scalings)):
    lifetime, var = sweep_parameter_space(T1_space, T2_space,
                                          single_qubit_gate_time_space,
                                          two_qubit_gate_time_space,
                                          measure_time_space,
                                          gate_times['feedback'][i],
                                          n_cycles, n_shots,
                                          time_axis=True, **kwargs)
    lifetime_scale_feedback_extra[i] = lifetime
    var_scale_feedback_extra[i] = var
    print(datetime.datetime.now().time())
np.save('lifetime_scale_feedback_extra2', lifetime_scale_feedback_extra)
np.save('var_scale_feedback_extra2', var_scale_feedback_extra)
# %% Final scalings
scalings = [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
gate_times = scale_gate_times(gate_times=standard_times, scalings=scalings)
lifetime_scale_feedback_extra = np.zeros(len(scalings))
var_scale_feedback_extra = np.zeros(len(scalings))
for i in range(len(scalings)):
    lifetime, var = sweep_parameter_space(T1_space, T2_space,
                                          single_qubit_gate_time_space,
                                          two_qubit_gate_time_space,
                                          measure_time_space,
                                          gate_times['feedback'][i],
                                          n_cycles, n_shots,
                                          time_axis=True, **kwargs)
    lifetime_scale_feedback_extra[i] = lifetime
    var_scale_feedback_extra[i] = var
    print(datetime.datetime.now().time())
np.save('lifetime_scale_feedback_extra', lifetime_scale_feedback_extra)
np.save('var_scale_feedback_extra', var_scale_feedback_extra)
# %% Final final scalings
scalings = [16.0, 17.0, 18.0, 19.0, 20.0, 21.0]
gate_times = scale_gate_times(gate_times=standard_times, scalings=scalings)
lifetime_scale_feedback_extra = np.zeros(len(scalings))
var_scale_feedback_extra = np.zeros(len(scalings))
for i in range(len(scalings)):
    lifetime, var = sweep_parameter_space(T1_space, T2_space,
                                          single_qubit_gate_time_space,
                                          two_qubit_gate_time_space,
                                          measure_time_space,
                                          gate_times['feedback'][i],
                                          n_cycles, n_shots,
                                          time_axis=True, **kwargs)
    lifetime_scale_feedback_extra[i] = lifetime
    var_scale_feedback_extra[i] = var
    print(datetime.datetime.now().time())
np.save('lifetime_scale_feedback_extra2', lifetime_scale_feedback_extra)
np.save('var_scale_feedback_extra2', var_scale_feedback_extra)

# %% Test with moved delay
scalings = [1.0, 4.0, 7.0, 10.0, 13.0]
kwargs = {'reset': True, 'data_process_type': 'recovery', 'idle_noise': True,
          'snapshot_type': 'dm', 'encoding': False, 'theta': 0, 'phi': 0,
          'transpile': False, 'simulator_type': 'statevector',
          'move_feedback_delay': True}
gate_times = scale_gate_times(gate_times=standard_times, scalings=scalings)
lifetime_scale_feedback_moved = np.zeros(len(scalings))
var_scale_feedback_moved = np.zeros(len(scalings))
for i in range(len(scalings)):
    lifetime, var = sweep_parameter_space(T1_space, T2_space,
                                          single_qubit_gate_time_space,
                                          two_qubit_gate_time_space,
                                          measure_time_space,
                                          gate_times['feedback'][i],
                                          n_cycles, n_shots,
                                          time_axis=True, **kwargs)
    lifetime_scale_feedback_moved[i] = lifetime
    var_scale_feedback_moved[i] = var
    print(datetime.datetime.now().time())
np.save('lifetime_scale_feedback_moved', lifetime_scale_feedback_moved)
np.save('var_scale_feedback_moved', var_scale_feedback_moved)

# %% Load data
lifetime_scale_all_gates = np.load('lifetime_scale_all_gates.npy')
lifetime_scale_single_qubit_gates = np.load(
    'lifetime_scale_single_qubit_gates.npy')
lifetime_scale_two_qubit_gates = np.load('lifetime_scale_two_qubit_gates.npy')
lifetime_scale_measure = np.load('lifetime_scale_measure.npy')
lifetime_scale_feedback = np.load('lifetime_scale_feedback.npy')
# %% Load data
# var_scale_all_gates = np.load('var_scale_all_gates.npy')
var_scale_single_qubit_gates = np.load('var_scale_single_qubit_gates.npy')
var_scale_two_qubit_gates = np.load('var_scale_two_qubit_gates.npy')
var_scale_measure = np.load('var_scale_measure.npy')
var_scale_feedback = np.load('var_scale_feedback.npy')
# %% Appending all times
feedback_start = lifetime_scale_feedback
feedback_e1 = extra1
ordering = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
feedback_list = [feedback_start, feedback_e1, extra2]
indexes = [0, 0, 0]
feedback_full = []
for i in ordering:
    feedback_full.append(feedback_list[i][indexes[i]])
    print("\n", i)
    print(indexes)
    indexes[i] += 1
# %%
full_scalings = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75,
                 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75,
                 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0,
                 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0]
lifetime_scale_feedback = np.array(feedback_full)
# %% Plot gate time scalings
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
scalings = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25,
            1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]


# Subplot 1: Target gate times
ax.plot(scalings[2:], lifetime_scale_all_gates*1e-3, '-o',
        label='All gates')  # Missing data at 0 and 0.25
# ax.plot(scalings, lifetime_scale_single_qubit_gates*1e-3, '-s', label='Single-qubit gates')
ax.plot(scalings, lifetime_scale_two_qubit_gates *
        1e-3, '-D', label='Two-qubit gates', zorder=5)
ax.plot(scalings, lifetime_scale_measure*1e-3, '-^', label='Measurements')
ax.plot(scalings, lifetime_scale_feedback*1e-3, '-v', label='Feedback')

yerr = np.sqrt(var_scale_single_qubit_gates)
ax.errorbar(scalings, lifetime_scale_single_qubit_gates*1e-3,
            yerr*1e-3, linestyle='-', label='Single-qubit gates')


ax.set_title(r'Logical Lifetime over scaling gate times')
ax.set_xlabel('Scale factor of gate times')
ax.set_ylabel(r'Logical lifetime $[\mu s]$')
ax.legend()
# %%
ax.plot(scalings, (lifetime_scale_all_gates/lifetime_scale_all_gates)**-1, '-k')
ax.plot(scalings, (lifetime_scale_single_qubit_gates/lifetime_scale_all_gates)
        ** -1, 's', color='C0', label='Single-qubit gates')
ax.plot(scalings, (lifetime_scale_two_qubit_gates/lifetime_scale_all_gates)
        ** -1, 'D', color='C1', label='Two-qubit gates', zorder=5)
ax.plot(scalings, (lifetime_scale_measure/lifetime_scale_all_gates)
        ** -1, '^', color='C2', label='Measurements')
ax.plot(scalings, (lifetime_scale_feedback/lifetime_scale_all_gates)
        ** -1, 'v', color='C3', label='Feedback')

# ax[1].set_title(r'R')
ax.set_xlabel('Scale factor of gate times')
ax.set_ylabel(r'Relative contribution to errors ')
ax.legend()

# Polyfits
gate_scalings = [lifetime_scale_single_qubit_gates, lifetime_scale_two_qubit_gates,
                 lifetime_scale_measure, lifetime_scale_feedback]
shape_list = ['o', 'D', '^', 'v']
for i in range(len(gate_scalings)):
    y0 = gate_scalings[i]*1e-3
    y1 = (gate_scalings[i]/lifetime_scale_all_gates)**-1
    x = np.array(scalings)
    # a = np.polyfit(x, y, 2) #m = slope, b = intercept.
    m, b = np.polyfit(x, y0, 1)
    #ax[0].plot(x, m*x + b, ':', color='C'+str(i))
    m, b = np.polyfit(x, y1, 1)
    #ax.plot(x, y1, shape_list[i], color='C'+str(i))
    ax.plot(x, m*x + b, ':', color='C'+str(i))
    # ax[1].plot(x, a[0]*x**2 + a[1]*x + a[2], color='C'+str(i+1)) #add line of best fit.
# %%
x = np.array([1, 3, 5, 7])
y = np.array([6, 3, 9, 5])
m, b = np.polyfit(x, y, 1)  # m = slope, b = intercept.
plt.plot(x, y, 'o')  # create scatter plot.
plt.plot(x, m*x + b)  # add line of best fit.
# %%


def get_circuit_time_formula(gate_times={}):
    tot_gate_times = standard_times.get_gate_times(gate_times)
    return (8*tot_gate_times['h']+16*tot_gate_times['cz'] + 4*tot_gate_times['measure'] +
            tot_gate_times['feedback'] + tot_gate_times['delay'])/1000


def get_scaled_circ_times(key, scales):
    if key == 'all':
        circ_times = []
        for scale in scales:
            new_gate_times = {}
            for key in standard_times.get_gate_times():
                new_gate_times[key] = standard_times[key]*scale
            circ_times.append(get_circuit_time_formula(new_gate_times))
        return circ_times
    else:
        return [get_circuit_time_formula({key: standard_times[key]*scale}) for scale in scales]


# get_scaled_circ_times('feedback',[0,1,2])
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
#scalings = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
scalings = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75,
            3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75,
            6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0,
            9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0]
times_all_gates = get_scaled_circ_times('all', scalings[2:13])
times_single_qubit_gates = get_scaled_circ_times('h', scalings[:13])
times_two_qubit_gates = get_scaled_circ_times('cz', scalings[:13])
times_measure = get_scaled_circ_times('measure', scalings[:13])
times_feedback = get_scaled_circ_times('feedback', scalings)

moved_scalings = [1.0, 4.0, 7.0, 10.0, 13.0]
times_moved = get_scaled_circ_times('feedback', moved_scalings)
# Subplot 1: Target gate times
ax.plot(times_all_gates, lifetime_scale_all_gates*1e-3, 'o',
        label='All gates proportionally')  # Missing data at 0 and 0.25
ax.plot(times_single_qubit_gates, lifetime_scale_single_qubit_gates *
        1e-3, 's', label='Single-qubit gates')
ax.plot(times_two_qubit_gates, lifetime_scale_two_qubit_gates *
        1e-3, 'D', label='Two-qubit gates', zorder=5)
ax.plot(times_measure, lifetime_scale_measure*1e-3, '^', label='Measurements')
ax.plot(times_feedback, lifetime_scale_feedback*1e-3, 'v', label='Feedback')
#ax.plot(times_moved, delay*1e-3, 's', label='Delay',zorder=100)

# Fit a 2nd degree polynomial to feedback
#c0, c1, c2 = np.polyfit(times_feedback, lifetime_scale_feedback*1e-3, 2)
#x = np.linspace(2.7, 9.0, 100)
#y = c2 + c1*x + c0*(x**2)
#ax.plot(x, y, '--', color='C4')

# yerr=np.sqrt(var_scale_single_qubit_gates)
#ax.errorbar(scalings, lifetime_scale_single_qubit_gates*1e-3, yerr*1e-3, linestyle='-',marker='s', label='Single-qubit gates')

colors_def = plt.get_cmap("tab10")
trans_offset = mtransforms.offset_copy(ax.transData, fig=fig,
                                       x=.05, y=0.15, units='inches')


ax.text(times_single_qubit_gates[-1], lifetime_scale_single_qubit_gates[-1]*1e-3,
        rf'$t_1 ={times_single_qubit_gates[-1]:.0f}$ μs', color=colors_def(1), transform=trans_offset)
ax.text(times_two_qubit_gates[-1], lifetime_scale_two_qubit_gates[-1]*1e-3,
        rf'$t_2 ={times_two_qubit_gates[-1]:.0f}$ μs', color=colors_def(2), transform=trans_offset)
ax.text(times_measure[-1], lifetime_scale_measure[-1]*1e-3,
        rf'$t_M ={times_measure [-1]:.0f}$ μs', color=colors_def(3), transform=trans_offset)
ax.text(times_feedback[-1], lifetime_scale_feedback[-1]*1e-3,
        rf'$t_f ={times_feedback[-1]:.0f}$ μs', color=colors_def(4), transform=trans_offset)

#ax.set_title(r'Logical Lifetime over scaling gate times')
ax.set_xlabel('Total stabilizer cycle duration $[\mu s]$')
ax.set_ylabel(r'Logical lifetime $[\mu s]$')
ax.legend()
# %%
