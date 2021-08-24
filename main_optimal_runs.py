# This file contains all code used to generate and save data and figures for
# the runs of optimal feedback or delay time.

# %% Import modules
import matplotlib.pyplot as plt
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

# %% Generate data
# With the number of cycles and shots used, each simulation takes approximately
# 30 minutes, resulting in a total of 1.5 hours for all three.

n_cycles = 15
n_shots = 1024*8*3*2
# %%
with open('data/base_qec_data.npy', 'rb') as f:
    fid_L_s, P_L_s, time_s = np.load(f)
    fidelities_perf, times_perf = np.load(f)
    fid_1_single, timespan = np.load(f)
    fid_0_encoded, timespan = np.load(f)
    fid_0_ps, counts, times_ps = np.load(f)
    fid_0_ps_perf, counts_perf, times_ps_perf = np.load(f)
    fid_p_single, timespan = np.load(f)
    fid_0_ps_L, P_0_ps_L, counts_L, times_ps_L = np.load(f)
# %%
fid_L_s, P_L_s, time_s = fidelity_from_scratch(n_cycles, n_shots,
                                               gate_times=standard_times, reset=True, data_process_type='recovery',
                                               idle_noise=True, snapshot_type='dm', encoding=False, theta=0, phi=0,
                                               transpile=False, project=True)
fid_L_f, P_L_f, time_f = fidelity_from_scratch(n_cycles, n_shots,
                                               gate_times={'feedback': 5000}, reset=True, data_process_type='recovery',
                                               idle_noise=True, snapshot_type='dm', encoding=False, theta=0, phi=0,
                                               transpile=False, project=True)
fid_L_d, P_L_d, time_d = fidelity_from_scratch(n_cycles, n_shots,
                                               gate_times={'delay': 5000}, reset=True, data_process_type='recovery',
                                               idle_noise=True, snapshot_type='dm', encoding=False, theta=0, phi=0,
                                               transpile=False, project=True)

# %% Reformat data
# Reformat as np arrays
fid_L_standard = np.array(fid_L_s)
fid_L_feedback = np.array(fid_L_f)
fid_L_delay = np.array(fid_L_d)
P_L_standard = np.array(P_L_s)
P_L_feedback = np.array(P_L_f)
P_L_delay = np.array(P_L_d)
# Generate x-values
t_standard = time_s
#t_standard = np.array([time_s['dm_'+str(i)] for i in range(n_cycles+1)])
t_feedback = np.array([time_f['dm_'+str(i)] for i in range(n_cycles+1)])
t_delay = np.array([time_d['dm_'+str(i)] for i in range(n_cycles+1)])
# %%
fid_standard = np.array(fid_L_s)
fid_standard[1:] = fid_standard[1:]*np.mean(P_L_s)
fid_feedback = np.array(fid_L_feedback)
fid_feedback[1:] = fid_feedback[1:]*np.mean(P_L_feedback[1:])
fid_delay = np.array(fid_L_delay)
fid_delay[1:] = fid_delay[1:]*np.mean(P_L_delay[1:])

p0 = (40e3, 0, 0.9)  # start with values near those we expect
pars_s, cov_s = scipy.optimize.curve_fit(
    monoExp, t_standard[1:], fid_standard[1:], p0)
pars_f, cov_f = scipy.optimize.curve_fit(
    monoExp, t_feedback[1:], fid_feedback[1:], p0)
pars_d, cov_d = scipy.optimize.curve_fit(
    monoExp, t_delay[1:], fid_delay[1:], p0)

# %% Save data
function_data = [n_cycles,
                 n_shots,
                 fid_L_standard,
                 P_L_standard,
                 t_standard,
                 pars_s,
                 cov_s]
with open('data/QEC_hexagonal_standard_times.npy', 'wb') as f:
    for data in function_data:
        np.save(f, data)

function_data = [n_cycles,
                 n_shots,
                 fid_L_feedback,
                 P_L_feedback,
                 t_feedback,
                 pars_f,
                 cov_f]
with open('data/feedback_5mus_data.npy', 'wb') as f:
    for data in function_data:
        np.save(f, data)

function_data = [n_cycles,
                 n_shots,
                 fid_L_delay,
                 P_L_delay,
                 t_delay,
                 pars_d,
                 cov_d]
with open('data/delay_5mus_data.npy', 'wb') as f:
    for data in function_data:
        np.save(f, data)

# %% Load data
with open('data/base_qec_data.npy', 'rb') as f:
    fid_L_s, P_L_s, time_s = np.load(f)
    fidelities_perf, times_perf = np.load(f)
    fid_1_single, timespan = np.load(f)
    fid_0_encoded, timespan = np.load(f)
    fid_0_ps, counts, times_ps = np.load(f)
    fid_0_ps_perf, counts_perf, times_ps_perf = np.load(f)
    fid_p_single, timespan = np.load(f)
    fid_0_ps_L, P_0_ps_L, counts_L, times_ps_L = np.load(f)


with open('data/feedback_5mus_data.npy', 'rb') as f:
    n_cycles = np.load(f)
    n_shots = np.load(f)
    fid_L_feedback = np.load(f)
    P_L_feedback = np.load(f)
    t_feedback = np.load(f)
    pars_f = np.load(f)
    cov_f = np.load(f)


with open('data/delay_5mus_data.npy', 'rb') as f:
    n_cycles = np.load(f)
    n_shots = np.load(f)
    fid_L_delay = np.load(f)
    P_L_delay = np.load(f)
    t_delay = np.load(f)
    pars_d = np.load(f)
    cov_d = np.load(f)
# %% Plotting
fig, ax = plt.subplots(1, 1, figsize=(7, 5))

#fid_standard = np.array(fid_L_s)
#fid_standard[1:] = fid_standard[1:]*np.mean(P_L_s[1:])
#fid_feedback = np.array(fid_L_f)
#fid_feedback[1:] = fid_feedback[1:]*np.mean(P_L_f[1:])
#fid_delay = np.array(fid_L_d)
#fid_delay[1:] = fid_delay[1:]*np.mean(P_L_d[1:])


# Plot data
ax.plot(t_standard, fid_standard, 'o', color='C0', label='Standard times')
ax.plot(t_feedback, fid_feedback, '^', color='C1', label='Feedback, 5 μs')
ax.plot(t_delay, fid_delay, 'v', color='C2', label='Delay, 5 μs')

time_list = [t_standard, t_feedback, t_delay]
fid_list = [fid_standard, fid_feedback, fid_delay]
P_list = [np.mean(P_L_s[1:]), np.mean(
    P_L_feedback[1:]), np.mean(P_L_delay[1:])]
color_list = ['C0', 'C1', 'C2']
lifetimes = []
errs = []
t = np.linspace(0, 140e3, 100)
for i in range(len(time_list)):
    p0 = (40e3, 0, 0.9)  # start with values near those we expect
    x = time_list[i]
    y = fid_list[i]
    pars, cov = scipy.optimize.curve_fit(monoExp, x[1:], y[1:], p0)
    T, c, A = pars
    print(T)
    lifetimes.append(T)
    errs.append(np.sqrt(cov[0][0]))
    ax.plot(t, monoExp(t, *pars), ':', color=color_list[i])
ax.text(65000, 0.38,
        rf'$T_L ={lifetimes[0]/1000:.1f}$ μs,  $P_L={P_list[0]:.2f}$', color='C0')
ax.text(88000, 0.34,
        rf'$T_L ={lifetimes[1]/1000:.1f}$ μs,  $P_L={P_list[1]:.2f}$', color='C1')
ax.text(45000, 0.56,
        rf'$T_L ={lifetimes[2]/1000:.1f}$ μs,  $P_L={P_list[2]:.2f}$', color='C2')
ax.set_xticklabels(["0", "0", "20", "40", "60", "80", "100", "120", "140"])
#ax.set(ylim=(0.0, 1.0))
# ax.ticklabel_format(style='sci',scilimits=(0,0),useMathText=True)
#ax.set_title(r'Expectation value of $|0\rangle$ and $|1\rangle$')
ax.set_xlabel('Time [μs]')
#ax.set_ylabel(r'Average state fidelity of $|0_L\rangle$')
ax.set_ylabel(r'Probability of remaining in the initial state, $F$')
ax.legend()
