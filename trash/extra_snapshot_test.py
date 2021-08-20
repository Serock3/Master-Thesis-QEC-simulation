

# %% Test Int counter object
from simulator_program.idle_noise import add_idle_noise_to_circuit
from simulator_program.custom_noise_models import standard_times_delay, thermal_relaxation_model_V2
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit import (QuantumCircuit,
                    QuantumRegister,
                    ClassicalRegister,
                    AncillaRegister,
                    execute,
                    Aer
                    )
from qiskit.quantum_info.states.measures import state_fidelity
from simulator_program import decay
from simulator_program.stabilizers import StabilizerRegisters, add_delay_marker, add_snapshot_to_circuit, get_empty_stabilizer_circuit, label_counter, get_encoded_state
import matplotlib.transforms as mtransforms
import numpy as np
from simulator_program.custom_noise_models import standard_times, thermal_relaxation_model_V2
from matplotlib import pyplot as plt
from simulator_program.data_analysis_tools import fidelity_from_scratch, perfect_stab_circuit, project_dm_to_logical_subspace_V1
import scipy
from IPython.display import display
from qiskit.providers.aer import AerSimulator, QasmSimulator


class Int(object):
    def __init__(self, value):
        self.value = value

    def __add__(self, other):
        self.value += other
        return self

    def __str__(self):
        return str(self.value)

    def inc(self):
        self.value += 1


my_counter = Int(0)
# def int_counter():
#     global my_counter
#     my_counter += 1


def test(hej=my_counter):
    if hej != my_counter:
        my_counter.value = hej
    # hej = hej + 1
    print(hej)
    print(my_counter)
    my_counter.inc()


def test_2(hej=Int(0)):
    hej = hej + 1

    print(hej)
# %% Test pre-recovery snapshot
# Exp fit func


def monoExp(t, T, c, A):
    return (A-c) * np.exp(-t/T) + c


def idealExp(t, T):
    return 0.5 * np.exp(-t/T) + 0.5


n_shots = 1024*4

n_cycles = 20
pauliop = 'ZZZZZ'
gate_times = standard_times

T1 = 40e3
T2 = 60e3

# %%
F_L, P_L, time = fidelity_from_scratch(
    n_cycles, n_shots, T1=T1, T2=T2, encoding=False, project=True, gate_times={'delay': 0},
    transpile=True)
times = np.array([time['dm_' + str(n)] for n in range(label_counter.value)])
print("Base QEC (hexagonal) done")

# Decay encoded qubit
n_datapoints = 100
timespan = np.linspace(0, times[-1], n_datapoints)
res_0 = decay.get_idle_encoded_513(
    timespan, snapshot_type=['dm'], T1=T1, T2=T2)
fid_0_encoded = [state_fidelity(get_encoded_state(0, 0, None), res_0.data()['dm_'+str(index)])
                 for index in range(n_datapoints)]
fid_0_encoded_L = [state_fidelity([1, 0], project_dm_to_logical_subspace_V1(res_0.data()['dm_'+str(index)]))
                   for index in range(n_datapoints)]
print("Decay encoded qubit done")

# F_L_5, P_L_5, time = fidelity_from_scratch(
#     n_cycles, n_shots, T1=T1, T2=T2, encoding=False, project=True, gate_times={'delay': 5000},
#     transpile=False)
# times_5 = np.array([time['dm_' + str(n)] for n in range(label_counter.value)])
# print("Base QEC (hexagonal) done")

# F_L_10, P_L_10, time = fidelity_from_scratch(
#     n_cycles, n_shots, T1=T1, T2=T2, encoding=False, project=True, gate_times={'delay': 10000},
#     transpile=False)
# times_10 = np.array([time['dm_' + str(n)] for n in range(label_counter.value)])
# print("Base QEC (hexagonal) done")
# # Perfect decoding (no noise in stabilizer cycle)
# fidelities_perf, time = perfect_stab_circuit(
#     n_cycles, n_shots, T1=T1, T2=T2)
# times_perf = np.array([time['dm_' + str(n)] for n in range(n_cycles)]+[time['dm_' + str(n_cycles)]])
# print("Perfect decoding (no noise in stabilizer cycle) done")

# %%
with open('data/extra_snapshot.npy', 'wb') as f:
    np.save(f, (F_L,P_L,times))
    np.save(f, (F_L_5,P_L_5,times_5))
    np.save(f, (F_L_10,P_L_10,times_10))
    np.save(f, (fid_0_encoded,timespan))

#%%
with open('data/extra_snapshot.npy', 'rb') as f:
    F_L,P_L,times = np.load(f)
    F_L_5,P_L_5,times_5 = np.load(f)
    F_L_10,P_L_10,times_10 = np.load(f)
    fid_0_encoded,timespan = np.load(f)
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = plt.get_cmap("tab10")
color_count = 0
# Position of fractions
trans_offset = mtransforms.offset_copy(ax.transData, fig=fig,
                                       x=-0.05, y=-0.3, units='inches')
trans_offset_perf = mtransforms.offset_copy(ax.transData, fig=fig,
                                            x=0, y=0.1, units='inches')

trans_offset_lifetime = mtransforms.offset_copy(ax.transData, fig=fig,
                                                x=-0.05, y=0.15, units='inches')

# Plot normal QEC
fidelities = [F_L_i*P_L_i for F_L_i, P_L_i in zip(F_L, P_L)]
ax.plot(times/1000, fidelities, '.-', color=colors(color_count),
        label=r'$F$')
ax.plot(times/1000, F_L, '.--', color=colors(color_count),
        label=r'$F_L$')
color_count += 1

# Plot normal QEC 5 us delay
fidelities = [F_L_i*P_L_i for F_L_i, P_L_i in zip(F_L_5, P_L_5)]
ax.plot(times_5/1000, fidelities, '.-', color=colors(color_count),
        label=r'$F$, (5 $\mu s$ delay)')
ax.plot(times_5/1000, F_L_5, '.--', color=colors(color_count),
        label=r'$F_L$, (5 $\mu s$ delay)')
color_count += 1

# Plot normal QEC 10 us delay
fidelities = [F_L_i*P_L_i for F_L_i, P_L_i in zip(F_L_10, P_L_10)]
ax.plot(times_10/1000, fidelities, '.-', color=colors(color_count),
        label=r'$F$, (10 $\mu s$ delay)')
ax.plot(times_10/1000, F_L_10, '.--', color=colors(color_count),
        label=r'$F_L$, (10 $\mu s$ delay)')
color_count += 1

# Plot normal QEC projected





# # Plot P_L
# ax.plot(times/1000, P_L, '.-', color=colors(color_count), label=r'$P_L=F/F_L$')
# color_count += 1

# Plot encoded qubit decay
ax.plot(timespan/1000, fid_0_encoded, '--',
        color=colors(color_count), label='Encoded qubit decay')
# p0 = (T1, 0, 0.9) # start with values near those we expect
# pars, cov = scipy.optimize.curve_fit(monoExp, timespan[:-5], fid_0_encoded[:-5], p0)
# T, c, A = pars
# trans_offset_lifetime = mtransforms.offset_copy(ax.transData, fig=fig,
#                                        x=-0.05, y=0.20, units='inches')
# ax.text(times[-3]/1000,fid_0_encoded[-3],rf'$T_L ={T/1000:.0f}$ μs',color=colors(color_count), transform=trans_offset_lifetime)
# print('Encoded qubit',r'$(A-c)e^{-t/T}+c$,'+f' T={T/1000:.1f} ns, A={A:.2f}, c={c:.3f}')
color_count += 1

# # Plot encoded qubit decay projected
# ax.plot(timespan/1000, fid_0_encoded_L, '--',
#         color=colors(color_count), label='Encoded qubit decay projected')
# color_count += 1

ax.set_xlabel('Time [μs]')
ax.set_ylabel(r'Probability')
ax.set_ylim((0.2, 1.05))
ax.set_xlim((0, 80))
ax.legend()


# %% Testing adding idle noise
n_cycles = 2
reset = True
recovery = False
flag = False
encoding = False
# Define our registers (Maybe to be written as function?)
qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
cr = ClassicalRegister(4, 'syndrome_bit')  # The typical register
# cr = get_classical_register(n_cycles, flag) # Advanced list of registers
readout = ClassicalRegister(5, 'readout')

registers = StabilizerRegisters(qb, an, cr, readout)
circ = get_empty_stabilizer_circuit(registers)

inital_state = [0]*(2**7)
inital_state[1] = 1
circ.set_density_matrix(inital_state)
idle_snapshots = 3
add_snapshot_to_circuit(circ, snapshot_type='dm', current_cycle=0, qubits=qb,
                        conditional=False,
                        include_barriers=False)
add_delay_marker(circ, registers, idle_snapshots,
                 qubits=registers.QubitRegister, include_barriers=True)
add_snapshot_to_circuit(circ, snapshot_type='dm', current_cycle=1, qubits=qb,
                        conditional=False,
                        include_barriers=False)
add_idle_noise_to_circuit(circ)
display(circ.draw())

noise_model = thermal_relaxation_model_V2(
    T1=T1, T2=T2, gate_times=standard_times_delay)

sim_noise = QasmSimulator(noise_model=noise_model)
results = sim_noise.run(circ, shots=n_shots).result()
# results = execute(circ, Aer.get_backend('aer_simulator'), basis_gates={'x', 'y', 'z', 'h', 'sx',
#                                                                        'sz', 'sy', 's', 't', 'ry', 'u1', 'u2', 'u3', 'cx', 'cz', 'swap', 'iswap', 'measure', 'reset', 'barrier', 'set_density_matrix', 'save_density_matrix',
#                                                                        'save_expval', 'save_expval_var', 'snapshot', 'feedback', 'delay', 'unitary'},
#                   noise_model=noise_model, shots=n_shots).result()
for k in results.data():
    one = results.data()[k][1, 1]
    zero = results.data()[k][0, 0]
    print(k, '|0>=', zero, '|1>=', one)
# %%
dag = circuit_to_dag(circ)
print(dag.op_nodes()[3].name)
print(noise_model)
# %%
