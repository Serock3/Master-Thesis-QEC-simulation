# This file contains all code used to generate and save data and figures in
# the results section "Comparing connectivity graphs" in the report.

# %% Import modules
import matplotlib.pyplot as plt
import numpy as np
from qiskit import *


# Our own files
from simulator_program.custom_noise_models import (standard_times)
from simulator_program.data_analysis_tools import *
from simulator_program.custom_transpiler import *
from simulator_program.stabilizers import *
from simulator_program.post_select import *
from simulator_program.post_process import *
from simulator_program.idle_noise import *

# %% Generate data
# With the number of cycles and shots used, each simulation takes approximately
# 30 minutes, resulting in a total of 3 hours for all six.

n_cycles = 15
n_shots = 1024*8*3

# QEC, no delay, hexagonal layout
fid_L_hex, P_L_hex, time_hex = fidelity_from_scratch(n_cycles, n_shots,
                                                     gate_times=standard_times, reset=True, data_process_type='recovery',
                                                     idle_noise=True, snapshot_type='dm', encoding=False, theta=0, phi=0,
                                                     transpile=False, project=True)

# QEC, no delay, square layout
fid_L_DD, P_L_DD, time_DD = fidelity_from_scratch(n_cycles, n_shots,
                                                  gate_times=standard_times, reset=True, data_process_type='recovery',
                                                  idle_noise=True, snapshot_type='dm', encoding=False, theta=0, phi=0,
                                                  transpile=False, project=True, device='double_diamond')

# QEC, no delay, pipelined (hexagonal layout)
fid_L_pipe, P_L_pipe, time_pipe = fidelity_from_scratch(n_cycles, n_shots,
                                                        gate_times=standard_times, reset=True, data_process_type='recovery',
                                                        idle_noise=True, snapshot_type='dm', encoding=False, theta=0, phi=0,
                                                        transpile=False, project=True, pipeline=True, include_barriers=False)

# QEC, with delay, hexagonal layout
fid_L_hex_d, P_L_hex_d, time_hex_d = fidelity_from_scratch(n_cycles, n_shots,
                                                           gate_times={'delay': 4000}, reset=True, data_process_type='recovery',
                                                           idle_noise=True, snapshot_type='dm', encoding=False, theta=0, phi=0,
                                                           transpile=False, project=True)

# QEC, with delay, square layout
fid_L_DD_d, P_L_DD_d, time_DD_d = fidelity_from_scratch(n_cycles, n_shots,
                                                        gate_times={'delay': 4000}, reset=True, data_process_type='recovery',
                                                        idle_noise=True, snapshot_type='dm', encoding=False, theta=0, phi=0,
                                                        transpile=False, project=True, device='double_diamond')

# QEC, with delay, pipelined (hexagonal layout)
fid_L_pipe_d, P_L_pipe_d, time_pipe_d = fidelity_from_scratch(n_cycles, n_shots,
                                                              gate_times={'delay': 4000}, reset=True, data_process_type='recovery',
                                                              idle_noise=True, snapshot_type='dm', encoding=False, theta=0, phi=0,
                                                              transpile=False, project=True, pipeline=True, include_barriers=False)

# %% Reformat data to numpy arrays and performe curve fits

# Reformat outputs as np arrays
fid_L_hex = np.array(fid_L_hex)
fid_L_DD = np.array(fid_L_DD)
fid_L_pipe = np.array(fid_L_pipe)
fid_L_hex_d = np.array(fid_L_hex_d)
fid_L_DD_d = np.array(fid_L_DD_d)
fid_L_pipe_d = np.array(fid_L_pipe_d)
P_L_hex = np.array(P_L_hex)
P_L_DD = np.array(P_L_DD)
P_L_pipe = np.array(P_L_pipe)
P_L_hex_d = np.array(P_L_hex_d)
P_L_DD_d = np.array(P_L_DD_d)
P_L_pipe_d = np.array(P_L_pipe_d)

# Generate x-values
t_hex = np.array([time_hex['dm_'+str(i)] for i in range(n_cycles+1)])
t_DD = np.array([time_DD['dm_'+str(i)] for i in range(n_cycles+1)])
t_pipe = np.array([time_pipe['dm_'+str(i)] for i in range(n_cycles+1)])
t_hex_d = np.array([time_hex_d['dm_'+str(i)] for i in range(n_cycles+1)])
t_DD_d = np.array([time_DD_d['dm_'+str(i)] for i in range(n_cycles+1)])
t_pipe_d = np.array([time_pipe_d['dm_'+str(i)] for i in range(n_cycles+1)])

# Perform curve fits
p0 = (40e3, 0, 0.9)  # start with values near those we expect
pars_hex, cov_hex = scipy.optimize.curve_fit(
    monoExp, t_hex[1:], fid_L_hex[1:], p0)
pars_DD, cov_DD = scipy.optimize.curve_fit(monoExp, t_DD[1:], fid_L_DD[1:], p0)
pars_pipe, cov_pipe = scipy.optimize.curve_fit(
    monoExp, t_pipe[1:], fid_L_pipe[1:], p0)
pars_hex_d, cov_hex_d = scipy.optimize.curve_fit(
    monoExp, t_hex_d[1:], fid_L_hex_d[1:], p0)
pars_DD_d, cov_DD_d = scipy.optimize.curve_fit(
    monoExp, t_DD_d[1:], fid_L_DD_d[1:], p0)
pars_pipe_d, cov_pipe_d = scipy.optimize.curve_fit(
    monoExp, t_pipe_d[1:], fid_L_pipe_d[1:], p0)

# %% Save data in .npy files
# Define variables to save/load
function_data = [n_cycles,
                 n_shots,
                 fid_L_hex,
                 P_L_hex,
                 t_hex,
                 pars_hex,
                 cov_hex]
with open('data/QEC_hexagonal_standard_times.npy', 'wb') as f:
    for data in function_data:
        np.save(f, data)

function_data = [n_cycles,
                 n_shots,
                 fid_L_DD,
                 P_L_DD,
                 t_DD,
                 pars_DD,
                 cov_DD]
with open('data/QEC_square_standard_times.npy', 'wb') as f:
    for data in function_data:
        np.save(f, data)

function_data = [n_cycles,
                 n_shots,
                 fid_L_pipe,
                 P_L_pipe,
                 t_pipe,
                 pars_pipe,
                 cov_pipe]
with open('data/QEC_pipeline_standard_times.npy', 'wb') as f:
    for data in function_data:
        np.save(f, data)

function_data = [n_cycles,
                 n_shots,
                 fid_L_hex_d,
                 P_L_hex_d,
                 t_hex_d,
                 pars_hex_d,
                 cov_hex_d]
with open('data/QEC_hexagonal_4mus_delay.npy', 'wb') as f:
    for data in function_data:
        np.save(f, data)

function_data = [n_cycles,
                 n_shots,
                 fid_L_DD_d,
                 P_L_DD_d,
                 t_DD_d,
                 pars_DD_d,
                 cov_DD_d]
with open('data/QEC_square_4mus_delay.npy', 'wb') as f:
    for data in function_data:
        np.save(f, data)

function_data = [n_cycles,
                 n_shots,
                 fid_L_pipe_d,
                 P_L_pipe_d,
                 t_pipe_d,
                 pars_pipe_d,
                 cov_pipe_d]
with open('data/QEC_pipeline_4mus_delay.npy', 'wb') as f:
    for data in function_data:
        np.save(f, data)
# %% Load data
with open('data/QEC_hexagonal_standard_times.npy', 'rb') as f:
    n_cycles = np.load(f)
    n_shots = np.load(f)
    fid_L_hex = np.load(f)
    P_L_hex = np.load(f)
    t_hex = np.load(f)
    pars_hex = np.load(f)
    cov_hex = np.load(f)

with open('data/QEC_square_standard_times.npy', 'rb') as f:
    n_cycles = np.load(f)
    n_shots = np.load(f)
    fid_L_DD = np.load(f)
    P_L_DD = np.load(f)
    t_DD = np.load(f)
    pars_DD = np.load(f)
    cov_DD = np.load(f)

with open('data/QEC_pipeline_standard_times.npy', 'rb') as f:
    n_cycles = np.load(f)
    n_shots = np.load(f)
    fid_L_pipe = np.load(f)
    P_L_pipe = np.load(f)
    t_pipe = np.load(f)
    pars_pipe = np.load(f)
    cov_pipe = np.load(f)

with open('data/QEC_hexagonal_4mus_delay.npy', 'rb') as f:
    n_cycles = np.load(f)
    n_shots = np.load(f)
    fid_L_hex_d = np.load(f)
    P_L_hex_d = np.load(f)
    t_hex_d = np.load(f)
    pars_hex_d = np.load(f)
    cov_hex_d = np.load(f)

with open('data/QEC_square_4mus_delay.npy', 'rb') as f:
    n_cycles = np.load(f)
    n_shots = np.load(f)
    fid_L_DD_d = np.load(f)
    P_L_DD_d = np.load(f)
    t_DD_d = np.load(f)
    pars_DD_d = np.load(f)
    cov_DD_d = np.load(f)

with open('data/QEC_pipeline_4mus_delay.npy', 'rb') as f:
    n_cycles = np.load(f)
    n_shots = np.load(f)
    fid_L_pipe_d = np.load(f)
    P_L_pipe_d = np.load(f)
    t_pipe_d = np.load(f)
    pars_pipe_d = np.load(f)
    cov_pipe_d = np.load(f)
# %%
with open('data/base_qec_data.npy', 'rb') as f:
    F_L, P_L, times = np.load(f)
    fidelities_perf, times_perf = np.load(f)
    fid_1_single, timespan = np.load(f)
    fid_0_encoded, timespan = np.load(f)
    fid_0_ps, counts, times_ps = np.load(f)
    fid_0_ps_perf, counts_perf, times_ps_perf = np.load(f)
    fid_p_single, timespan = np.load(f)
    fid_0_ps_L, P_0_ps_L, counts_L, times_ps_L = np.load(f)

# %% Plot figures
fig, ax = plt.subplots(1, 1, figsize=(7, 5))

# For non-projected fidelity, multiply all but 1st value with P_L.
fid = np.ones(n_cycles+1)
fid[1:] = F_L[1:]*np.mean(P_L[1:])
pars, cov = scipy.optimize.curve_fit(monoExp, times[1:], fid[1:], p0)


fid_hex = np.ones(n_cycles+1)
fid_DD = np.ones(n_cycles+1)
fid_pipe = np.ones(n_cycles+1)
fid_hex_d = np.ones(n_cycles+1)
fid_DD_d = np.ones(n_cycles+1)
fid_pipe_d = np.ones(n_cycles+1)
fid_hex[1:] = fid_L_hex[1:]*np.mean(P_L_hex[1:])
fid_DD[1:] = fid_L_DD[1:]*np.mean(P_L_DD[1:])
fid_pipe[1:] = fid_L_pipe[1:]*np.mean(P_L_pipe[1:])
fid_hex_d[1:] = fid_L_hex_d[1:]*np.mean(P_L_hex_d[1:])
fid_DD_d[1:] = fid_L_DD_d[1:]*np.mean(P_L_DD_d[1:])
fid_pipe_d[1:] = fid_L_pipe_d[1:]*np.mean(P_L_pipe_d[1:])

# No delay plots
# ax.plot(times, fid, 's', color='C0',
#        label=rf'Hexagonal, $T_L ={pars[0]/1000:.1f}$ μs')
# ax.plot(t_hex, fid_hex, 'o', color='C0',
#        label=rf'Hexagonal, $T_L ={pars_hex[0]/1000:.1f}$ μs')
# ax.plot(t_DD, fid_DD, '>', color='C1',
#        label=rf'Square, $T_L ={pars_DD[0]/1000:.1f}$ μs')
# ax.plot(t_pipe, fid_pipe, '^', color='C2',
#        label=rf'Pipeline (Hexagonal), $T_L ={pars_pipe[0]/1000:.1f}$ μs')

# No delay plots
ax.plot(t_hex_d[:9], fid_hex_d[:9], 'D', color='C0',
        label=rf'Hexagonal, 5 μs delay, $T_L ={pars_hex_d[0]/1000:.1f}$ μs')
ax.plot(t_DD_d[:9], fid_DD_d[:9], '<', color='C1',
        label=rf'Square, 5 μs delay, $T_L ={pars_DD_d[0]/1000:.1f}$ μs')
ax.plot(t_pipe_d[:9], fid_pipe_d[:9], 'v', color='C2',
        label=rf'Pipeline, 5 μs delay, $T_L ={pars_pipe_d[0]/1000:.1f}$ μs')

# Plot fitted curves
t = np.linspace(0, 70e3, 100)
#ax.plot(t, monoExp(t, *pars_hex), ':', color='C0')
#ax.plot(t, monoExp(t, *pars_DD)*np.mean(P_L_DD[1:]), ':', color='C1',zorder=-5)
#ax.plot(t, monoExp(t, *pars_pipe)*np.mean(P_L_pipe[1:]), ':', color='C2',zorder=-5)
ax.plot(t, monoExp(t, *pars_hex_d) *
        np.mean(P_L_hex_d[1:]), ':', color='C0', zorder=-5)
ax.plot(t, monoExp(t, *pars_DD_d) *
        np.mean(P_L_DD_d[1:]), ':', color='C1', zorder=-5)
ax.plot(t, monoExp(t, *pars_pipe_d) *
        np.mean(P_L_pipe_d[1:]), ':', color='C2', zorder=-5)

# Axes settings
ax.set_xticklabels(["0", "0", "20", "40", "60", "80", "100", "120", "140"])
ax.set_xlabel('Time [μs]')
ax.set_ylabel(r'Probability of remaining in the initial state $F$')
ax.legend()
