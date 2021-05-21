# This file will make the plot that goes into the beggining of 4.3,
# which includes active error correction (on hexagonal device?)
# with and without perfect encoding, decay of single qubit/encoded qubit,
# and post selection.

# %% Imports
from matplotlib import pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from qiskit.quantum_info.states.measures import state_fidelity
import scipy
from simulator_program.idle_noise import add_idle_noise_to_circuit
from qiskit.execute_function import execute
from simulator_program.custom_transpiler import shortest_transpile_from_distribution, WACQT_device_properties
from simulator_program.stabilizers import StabilizerRegisters, get_classical_register, get_encoded_state, get_full_stabilizer_circuit
from simulator_program.custom_noise_models import thermal_relaxation_model_V2, standard_times
from qiskit import Aer, QuantumRegister, AncillaRegister, ClassicalRegister
from simulator_program import data_analysis_tools, decay

# %% Paramters
n_shots = 1024*8

n_cycles = 15
pauliop = 'ZZZZZ'
gate_times = standard_times

T1 = 40e3
T2 = 60e3

# %% Generate data

# Base QEC (hexagonal)
fidelities, time = data_analysis_tools.fidelity_from_scratch(
    n_cycles, n_shots, T1=T1, T2=T2, encoding=False)
times = np.array([time['dm_' + str(n)] for n in range(n_cycles)]+[time['end']])
print("Base QEC (hexagonal) done")

# Perfect decoding (no noise in stabilizer cycle)
fidelities_perf, time = data_analysis_tools.perfect_stab_circuit(
    n_cycles, n_shots, T1=T1, T2=T2)
times_perf = np.array([time['dm_' + str(n)] for n in range(n_cycles)]+[time['dm_' + str(n_cycles)]])
print("Perfect decoding (no noise in stabilizer cycle) done")

# Decay single qubit
n_datapoints = 100
timespan = np.linspace(0, times[-1], n_datapoints)
res_1 = decay.get_idle_single_qubit(
    timespan, snapshot_type=['dm'], theta=np.pi, T1=T1, T2=T2)
fid_1_single = [state_fidelity([0, 1], res_1.data()['dm_'+str(index)])
                for index in range(n_datapoints)]
print("Decay single qubit done")

# Decay encoded qubit
res_0 = decay.get_idle_encoded_513(
    timespan, snapshot_type=['dm'], T1=T1, T2=T2)
fid_0_encoded = [state_fidelity(get_encoded_state(0, 0, None), res_0.data()['dm_'+str(index)])
                 for index in range(n_datapoints)]
print("Decay encoded qubit done")

# Post selection
fid_0_ps, counts, time = data_analysis_tools.fidelity_from_scratch(n_cycles, n_shots,
    reset=True, data_process_type='post_select',
    idle_noise=True, snapshot_type='dm', encoding=False,
    transpile=True, pauliop='ZZZZZ')
times_ps = np.array([time['dm_con_' + str(n)] for n in range(n_cycles)]+[time['end']])
print("Post selection done")

# Post selection perf decoding 
fid_0_ps_perf, counts_perf, time = data_analysis_tools.perfect_stab_circuit(
    n_cycles, n_shots, T1=T1, T2=T2, data_process_type='post_select')
times_ps_perf = np.array([time['dm_con_' + str(n)] for n in range(n_cycles)]+[time['dm_con_' + str(n_cycles)]])
print("Post selection per decoding done")

#%% Post selection project instead of perf decoding
fid_0_ps_perf, counts_perf, time = data_analysis_tools.fidelity_from_scratch(
    n_cycles, n_shots, T1=T1, T2=T2, data_process_type='post_select', project= True)
times_ps_perf = np.array([time['dm_con_' + str(n)] for n in range(n_cycles)]+[time['dm_con_' + str(n_cycles)]])
print("Post selection per decoding done")

#%% Single qubit |+> decay
n_datapoints = 100
timespan = np.linspace(0, times[-1], n_datapoints)
res_p = decay.get_idle_single_qubit(
    timespan, snapshot_type=['dm'], theta=np.pi/2, T1=T1, T2=T2)
fid_p_single = [state_fidelity([1/np.sqrt(2), 1/np.sqrt(2)], res_p.data()['dm_'+str(index)])
                for index in range(n_datapoints)]
#%% Save
with open('data/base_qec_data.npy', 'wb') as f:
    np.save(f, (fidelities,times))
    np.save(f, (fidelities_perf,times_perf))
    np.save(f, (fid_1_single,timespan))
    np.save(f, (fid_0_encoded,timespan))
    np.save(f, (fid_0_ps,counts,times_ps))
    np.save(f, (fid_0_ps_perf,counts_perf,times_ps_perf))
#%% Load

with open('data/base_qec_data.npy', 'rb') as f:
    fidelities,times = np.load(f)
    fidelities_perf,times_perf = np.load(f)
    fid_1_single,timespan = np.load(f)
    fid_0_encoded,timespan = np.load(f)
    fid_0_ps,counts,times_ps = np.load(f)
    fid_0_ps_perf,counts_perf,times_ps_perf = np.load(f)

#%% 
# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# ## for Palatino and other serif fonts use:
# #rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)
# %% Plot
# Exp fit func
def monoExp(t, T, c, A):
    return (A-c) * np.exp(-t/T) + c

# Setup plot
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = plt.get_cmap("tab10")

# Position of fractions
trans_offset = mtransforms.offset_copy(ax.transData, fig=fig,
                                       x=0, y=-0.18, units='inches')
trans_offset_perf = mtransforms.offset_copy(ax.transData, fig=fig,
                                       x=0, y=0.08, units='inches')

trans_offset_lifetime = mtransforms.offset_copy(ax.transData, fig=fig,
                                       x=-0.05, y=0.10, units='inches')

# Plot normal QEC
ax.plot(times/1000, fidelities, '^', color=colors(0), label='QEC')
# Exp fit it
p0 = (T1, 0, 0.9)  # start with values near those we expect
pars, cov = scipy.optimize.curve_fit(monoExp, times[1:], fidelities[1:], p0)
T, c, A = pars
ax.plot(times/1000, monoExp(times, *pars), ':', color=colors(0))

ax.text(times[-3]/1000,monoExp(times[-3], *pars),rf'$T_L ={T/1000:.0f}$ μs',color=colors(0), transform=trans_offset_lifetime)
print('QEC',r'$(A-c)e^{-t/T}+c$,'+f' T={T:.0f} μs, A={A:.2f}, c={c:.3f}')

# Plot perf decoding QEC
# ax.plot(times_perf/1000, fidelities_perf, 'v', color=colors(1), label='QEC perfect decoding')
# # Exp fit it
# p0 = (T1, 0, 0.9)  # start with values near those we expect
# pars, cov = scipy.optimize.curve_fit(monoExp, times_perf, fidelities_perf, p0)
# T, c, A = pars
# ax.plot(times_perf/1000, monoExp(times_perf, *pars), ':', color=colors(1))
# trans_offset_lifetime = mtransforms.offset_copy(ax.transData, fig=fig,
#                                        x=-0.05, y=-0.25, units='inches')
# ax.text(times_perf[-3]/1000,monoExp(times_perf[-3], *pars),rf'$T_L ={T/1000:.0f}$ μs',color=colors(1), transform=trans_offset_lifetime)
# print('QEC peft decoding',r'$(A-c)e^{-t/T}+c$,'+f' T={T:.0f} ns, A={A:.2f}, c={c:.3f}')

# Plot post select
# ax.plot(times_ps/1000, fid_0_ps, '>', color=colors(4), label='Post-selection')
# # Exp fit it
# p0 = (T1, 0, 0.9)  # start with values near those we expect
# pars, cov = scipy.optimize.curve_fit(monoExp, times_ps[1:], np.array(counts)[1:]/n_shots, p0)
# T, c, A = pars
# print('Post-select',r'$(A-c)e^{-t/T}+c$,'+f' T={T:.0f} μs, A={A:.2f}, c={c:.3f}')
# # Fractions
# for i,(x, y, text) in enumerate(zip(times_ps,fid_0_ps, counts)):
#     if i == 0:
#         continue
#     if i%3==0:
#         plt.text(x/1000, y, f'{text/n_shots*100:.0f}%', size =9 ,transform=trans_offset)

# # Plot post select perf decoding
# ax.plot(times_ps_perf/1000, fid_0_ps_perf, '<', color=colors(5), label='Post-selection perfect decoding')
# # Exp fit it
# p0 = (T1, 0, 0.9)  # start with values near those we expect
# pars, cov = scipy.optimize.curve_fit(monoExp, times_ps_perf, np.array(counts_perf)/n_shots, p0)
# T, c, A = pars
# print('Post-select and project',r'$(A-c)e^{-t/T}+c$,'+f' T={T:.0f} μs, A={A:.2f}, c={c:.3f}')
# # Fractions
# for i,(x, y, text) in enumerate(zip(times_ps_perf,fid_0_ps_perf, counts_perf)):
#     if i%3==0:
#         plt.text(x/1000, y, f'{text/n_shots*100:.0f}%', size =9, transform=trans_offset_perf)

# Plot single/encoded qubit decay
ax.plot(timespan/1000, fid_1_single, '--',
        color=colors(2), label=r'Single qubit $|1\rangle$ decay')
trans_offset_lifetime = mtransforms.offset_copy(ax.transData, fig=fig,
                                       x=-0.05, y=-0.20, units='inches')
ax.text(times_perf[-3]/1000,fid_1_single[-3],rf'$T_1 ={T1/1000:.0f}$ μs',color=colors(2), transform=trans_offset_lifetime)

ax.plot(timespan/1000, fid_p_single, '--',
        color=colors(5), label=r'Single qubit $|+\rangle$ decay')
trans_offset_lifetime = mtransforms.offset_copy(ax.transData, fig=fig,
                                       x=-0.05, y=-0.20, units='inches')
ax.text(times_perf[-3]/1000,fid_p_single[-3],rf'$T_2 ={T2/1000:.0f}$ μs',color=colors(5), transform=trans_offset_lifetime)

ax.plot(timespan/1000, fid_0_encoded, '--',
        color=colors(3), label='Encoded qubit decay')
p0 = (T1, 0, 0.9) # start with values near those we expect
pars, cov = scipy.optimize.curve_fit(monoExp, timespan[:-5], fid_0_encoded[:-5], p0)
T, c, A = pars
trans_offset_lifetime = mtransforms.offset_copy(ax.transData, fig=fig,
                                       x=-0.05, y=0.20, units='inches')
ax.text(times_perf[-3]/1000,fid_0_encoded[-3],rf'$T_L ={T/1000:.0f}$ μs',color=colors(3), transform=trans_offset_lifetime)

ax.set_xlabel('Time [μs]')
ax.set_ylabel('Probability of remaining in initial state')
ax.legend()

fig.savefig('base_QEC.pdf')

# %%
