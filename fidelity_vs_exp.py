# This file compares the lifetime of the logical state when using fidelity vs <Z> during active QEC
# %%
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


# %%
n_shots = 1024*1
n_cycles = 10
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
                                   snapshot_type=['dm','exp'],
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

# %% Plot
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = plt.get_cmap("tab10")

def monoExp(t, T, c, A):
    return (A-c)* np.exp(-t/T) + c

fidelities = []
for current_cycle in range(n_cycles+1):
    state = results.data()['dm_' + str(current_cycle)]
    fidelities.append(state_fidelity(state, get_encoded_state(theta,phi,None)))

exp = []
for current_cycle in range(n_cycles+1):
    exp.append(results.data()['exp_' + str(current_cycle)])

#%%
theta = np.pi
circ = get_full_stabilizer_circuit(registers, n_cycles=n_cycles, reset=reset,
                                   recovery=recovery, flag=False,
                                   snapshot_type=['dm','exp'],
                                   conditional=conditional,
                                   encoding=encoding, theta=theta, phi=phi,
                                   pauliop=pauliop)


circ = shortest_transpile_from_distribution(circ, print_cost=False,
                                            **WACQT_device_properties)
circ, time = add_idle_noise_to_circuit(circ, gate_times=gate_times,
                                         T1=T1, T2=T2, return_time=True)
results = execute(circ, Aer.get_backend('aer_simulator'),
                  noise_model=noise_model, shots=n_shots).result()
exp_1 = []
for current_cycle in range(n_cycles+1):
    exp_1.append(results.data()['exp_' + str(current_cycle)])
#%%
times = np.array([time['exp_'+ str(n)] for n in range(n_cycles)]+[time['end']])
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(times/1000, fidelities, '^',color = colors(0),label=r'$|0_L\rangle$')

p0 = (T1, 0, 0.9) # start with values near those we expect
pars, cov = scipy.optimize.curve_fit(monoExp, times[1:], fidelities[1:], p0)
T, c, A = pars

ax.plot(times/1000, monoExp(times,*pars), '--',color = colors(0),label=r'$(A-c)e^{-t/T}+c$,'+f' T={T/1000:.0f} μs, A={A:.2f}, c={c:.3f}')


ax.plot(times/1000, exp, 'o',color = colors(1) ,label=r'$|0_L\rangle$')

p0 = (T1, 0, 0.9) # start with values near those we expect
pars, cov = scipy.optimize.curve_fit(monoExp, times[1:], exp[1:], p0)
T, c, A= pars

ax.plot(times/1000, monoExp(times,*pars), '--',color = colors(1),label=r'$(A-c)e^{-t/T}+c$,'+f' T={T/1000:.0f} μs, A={A:.2f}, c={c:.3f}')

ax.plot(times/1000, exp_1, 'o',color = colors(2) ,label=r'$|1_L\rangle$')

p0 = (T1, 0, 0.9) # start with values near those we expect*
pars, cov = scipy.optimize.curve_fit(monoExp, times[1:], exp_1[1:], p0)
T, c, A= pars

ax.plot(times/1000, monoExp(times,*pars), '--',color = colors(2),label=r'$(A-c)e^{-t/T}+c$,'+f' T={T/1000:.0f} μs, A={A:.2f}, c={c:.3f}')

ax.legend()
ax.set_xlabel(r'Time $[\mu s]$')
ax.set_ylabel(r'Prob. of initial state after projection')
plt.savefig('fidvsexp.pdf')
# %%
