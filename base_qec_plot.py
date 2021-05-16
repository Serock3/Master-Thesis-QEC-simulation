# This file will make the plot that goes into the beggining of 4.3,
# which includes active error correction (on hexagonal device?)
# with and without perfect encoding, decay of single qubit/encoded qubit,
# and post selection.

# %% Imports
from matplotlib import pyplot as plt
import numpy as np
from qiskit.quantum_info.states.measures import state_fidelity
import scipy
from simulator_program.idle_noise import add_idle_noise_to_circuit
from qiskit.execute_function import execute
from simulator_program.custom_transpiler import shortest_transpile_from_distribution, WACQT_device_properties
from simulator_program.stabilizers import StabilizerRegisters, get_classical_register, get_encoded_state, get_full_stabilizer_circuit
from simulator_program.custom_noise_models import thermal_relaxation_model_V2, WACQT_gate_times
from qiskit import Aer, QuantumRegister, AncillaRegister, ClassicalRegister
from simulator_program import data_analysis_tools, decay

# %% Paramters
n_shots = 512

n_cycles = 5
reset = True
recovery = True
conditional = False
encoding = False
theta = 0
phi = 0
pauliop = 'ZZZZZ'
gate_times = WACQT_gate_times

T1 = 40e3
T2 = 60e3

# %% Generate data

# Base QEC (hexagonal)
fidelities, time = data_analysis_tools.fidelity_from_scratch(
    n_cycles, n_shots, T1=T1, T2=T2, encoding=False)
times = np.array([time['dm_' + str(n)] for n in range(n_cycles)]+[time['end']])

# Perfect decoding (no noise in stabilizer cycle)
fidelities_perf, time = data_analysis_tools.perfect_stab_circuit(
    n_cycles, n_shots, T1=T1, T2=T2)
times_perf = np.array([time['dm_' + str(n)] for n in range(n_cycles)]+[time['dm_' + str(n_cycles)]])

# Decay single qubit
n_datapoints = 100
timespan = np.linspace(0, times[-1], n_datapoints)
res_1 = decay.get_idle_single_qubit(
    timespan, snapshot_type=['dm'], theta=np.pi, T1=T1, T2=T2)
fid_1_single = [state_fidelity([0, 1], res_1.data()['dm_'+str(index)])
                for index in range(n_datapoints)]

# Decay encoded qubit
res_0 = decay.get_idle_encoded_513(
    timespan, snapshot_type=['dm'], T1=T1, T2=T2)
fid_0_encoded = [state_fidelity(get_encoded_state(0, 0, None), res_0.data()['dm_'+str(index)])
                 for index in range(n_datapoints)]

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
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = plt.get_cmap("tab10")

# Plot normal QEC
ax.plot(times, fidelities, '^', color=colors(0), label='FID')
# Exp fit it
p0 = (T1, 0, 0.9)  # start with values near those we expect
pars, cov = scipy.optimize.curve_fit(monoExp, times[1:], fidelities[1:], p0)
T, c, A = pars
ax.plot(times, monoExp(times, *pars), ':', color=colors(0),
        label=r'$(A-c)e^{-t/T}+c$,'+f' T={T:.0f} ns, A={A:.2f}, c={c:.3f}')

# Plot perf decoding QEC
ax.plot(times_perf, fidelities_perf, 'o', color=colors(1), label='FID perf')
# Exp fit it
p0 = (T1, 0, 0.9)  # start with values near those we expect
pars, cov = scipy.optimize.curve_fit(monoExp, times_perf, fidelities_perf, p0)
T, c, A = pars
ax.plot(times_perf, monoExp(times_perf, *pars), ':', color=colors(1),
        label=r'$(A-c)e^{-t/T}+c$,'+f' T={T:.0f} ns, A={A:.2f}, c={c:.3f}')

# Plot single/encoded qubit decay
ax.plot(timespan, fid_1_single, '--',
        color=colors(2), label='Single qubit decay')
ax.plot(timespan, fid_0_encoded, '--',
        color=colors(3), label='Encoded qubit decay')

ax.set_xlabel('Time [ns]')
ax.set_ylabel('Probability of remaining in initial state')
ax.legend()
# %% Create circuit cp/paste from fidelity_vs_exp
noise_model = thermal_relaxation_model_V2(
    T1=T1, T2=T2, gate_times=gate_times)
# Registers
qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
cr = get_classical_register(n_cycles, reset=reset,
                            recovery=recovery, flag=False)
readout = ClassicalRegister(5, 'readout')
registers = StabilizerRegisters(qb, an, cr, readout)

# Circuits
circ = get_full_stabilizer_circuit(registers, n_cycles=n_cycles, reset=reset,
                                   recovery=recovery, flag=False,
                                   snapshot_type=['dm', 'exp'],
                                   conditional=conditional,
                                   encoding=encoding, theta=theta, phi=phi,
                                   pauliop=pauliop)


circ = shortest_transpile_from_distribution(circ, print_cost=False,
                                            **WACQT_device_properties)
circ, time = add_idle_noise_to_circuit(circ, gate_times=gate_times,
                                       T1=T1, T2=T2, return_time=True)

# %% Run

results = execute(circ, Aer.get_backend('aer_simulator'),
                  noise_model=noise_model, shots=n_shots).result()


# %%


fidelities = []
for current_cycle in range(n_cycles+1):
    state = results.data()['dm_' + str(current_cycle)]
    fidelities.append(state_fidelity(
        state, get_encoded_state(theta, phi, None)))

exp = []
for current_cycle in range(n_cycles+1):
    exp.append(results.data()['exp_' + str(current_cycle)])

times = np.array([time['dm_' + str(n)] for n in range(n_cycles)]+[time['end']])
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(times, fidelities, '^', color=colors(0), label='FID')

p0 = (T1, 0, 0.9)  # start with values near those we expect
pars, cov = scipy.optimize.curve_fit(monoExp, times[1:], fidelities[1:], p0)
T, c, A = pars

ax.plot(times, monoExp(times, *pars), '--', color=colors(0),
        label=r'$(A-c)e^{-t/T}+c$,'+f' T={T:.0f} ns, A={A:.2f}, c={c:.3f}')


ax.plot(times, exp, 'o', color=colors(1), label='<0|Z|0>')

p0 = (T1, 0, 0.9)  # start with values near those we expect
pars, cov = scipy.optimize.curve_fit(monoExp, times[1:], exp[1:], p0)
T, c, A = pars

ax.plot(times, monoExp(times, *pars), '--', color=colors(1),
        label=r'$(A-c)e^{-t/T}+c$,'+f' T={T:.0f} ns, A={A:.2f}, c={c:.3f}')

ax.legend()
