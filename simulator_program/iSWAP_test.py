# %%
from custom_transpiler import WAQCT_device_properties, _add_custom_device_equivalences, shortest_transpile_from_distribution
from qiskit import *
from IPython.core.display import display
from qiskit.visualization.state_visualization import plot_bloch_multivector
from custom_transpiler import _add_custom_device_equivalences
from qiskit.quantum_info.states.measures import state_fidelity
from qiskit.providers.aer.backends.qasm_simulator import QasmSimulator
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# %%
_add_custom_device_equivalences()

#%%
circ = QuantumCircuit(3)

circ.x(1)
# circ.x(0)
circ.cz(0, 2)
# circ.swap(0,1)
circ.measure_all()

display(circ.draw())

transpiled_circ = transpile(circ, coupling_map=[[0, 1], [1, 0], [1, 2], [
                            2, 1]], **{'basis_gates': ['id', 'u1', 'u2', 'u3', 'cz', 'iswap']},
                             optimization_level=2,routing_method='lookahead',layout_method='trivial')
display(transpiled_circ.draw())

results = execute(
    circ,
    Aer.get_backend('statevector_simulator'),
    noise_model=None,
).result()
results_transpiled = execute(
    transpiled_circ,
    Aer.get_backend('statevector_simulator'),
    noise_model=None,
).result()
print(results.get_statevector())
print(results_transpiled.get_statevector())
display(plot_bloch_multivector(results.get_statevector()))  # %%
display(plot_bloch_multivector(results_transpiled.get_statevector()))  # %%

print('fidelity ', state_fidelity(
    results.get_statevector(), results_transpiled.get_statevector()))

# %%
snapshot_type = 'statevector'
circ = QuantumCircuit(3)
circ.x(1)

circ.swap(0, 1)
circ.snapshot('snap1', snapshot_type=snapshot_type)
circ.x(1)
circ.swap(0, 2)
circ.snapshot('snap2', snapshot_type=snapshot_type)
circ.swap(1, 2)
circ.snapshot('snap3', snapshot_type=snapshot_type)
circ.measure_all()

# , coupling_map=[[0, 1], [1, 0], [1, 2], [
#                             2, 1]]
transpiled_circ = transpile(
    circ,**{'basis_gates': ['id', 'u1', 'u2', 'u3', 'cz', 'iswap']}, 
                            optimization_level=2, layout_method = 'trivial'
                            ,routing_method='lookahead')

display(circ.draw())
display(transpiled_circ.draw())


results1 = execute(
    circ,
    Aer.get_backend('qasm_simulator'),
    noise_model=None,
    shots=1
).result()

results2 = execute(
    transpiled_circ,
    Aer.get_backend('qasm_simulator'),
    noise_model=None,
    shots=1
).result()

state1 = results1.data()['snapshots'][snapshot_type]
state2 = results2.data()['snapshots'][snapshot_type]

print('fidelity1 ', state_fidelity(state1['snap1'][0], state2['snap1'][0]))
print('fidelity1 ', state_fidelity(state1['snap2'][0], state2['snap2'][0]))
print('fidelity1 ', state_fidelity(state1['snap3'][0], state2['snap3'][0]))
# print('fidelity1 ',state_fidelity(state1['snap4'][0],state2['snap4'][0]))
print(np.vdot(state1['snap3'][0], state2['snap3'][0]))

# %%


def encode_input(qbReg):
    '''Encode the input into logical 0 and 1
    This assumes that the 0:th qubit is the
    original state |psi> = a|0> + b|1>'''
    encoding_circuit = QuantumCircuit(qbReg)

    encoding_circuit.h(qbReg[3])
    encoding_circuit.cz(qbReg[3], qbReg[1])
    encoding_circuit.cz(qbReg[3], qbReg[2])
    encoding_circuit.cx(qbReg[3], qbReg[0])

    encoding_circuit.h(qbReg[2])
    encoding_circuit.cx(qbReg[2], qbReg[0])
    encoding_circuit.cz(qbReg[2], qbReg[3])
    encoding_circuit.cz(qbReg[2], qbReg[4])

    encoding_circuit.h(qbReg[1])
    encoding_circuit.cz(qbReg[1], qbReg[0])
    encoding_circuit.cx(qbReg[1], qbReg[3])
    encoding_circuit.cz(qbReg[1], qbReg[4])

    encoding_circuit.h(qbReg[4])
    encoding_circuit.cz(qbReg[4], qbReg[2])
    encoding_circuit.cz(qbReg[4], qbReg[3])
    encoding_circuit.cx(qbReg[4], qbReg[1])

    return encoding_circuit


def measure_stabilizer(qbReg, anReg, clReg, i, reset=True):
    '''Function for adding stabilizer measurements to a circuit.
    Note that a measurement of X is done by using Hadamard before
    and after. Input i specifies the stabilizer to measure:
        i=0: XZZXI
        i=1: IXZZX
        i=2: XIXZZ
        i=3: ZXIXZ
    Other inputs are the circuit as well as the required registers'''

    if not isinstance(i, int):
        raise error('i must be an integer')

    stab_circuit = QuantumCircuit(qbReg, anReg, clReg)

    # Generate indexes
    index = np.mod(i + np.array([0, 1, 2, 3]), 5)

    # Measure stabilizers
    stab_circuit.h(qbReg[index[0]])
    stab_circuit.h(anReg[1])
    stab_circuit.cz(anReg[1], qbReg[index[0]])
    stab_circuit.h(qbReg[index[0]])

    stab_circuit.cz(anReg[1], qbReg[index[1]])

    stab_circuit.cz(anReg[1], qbReg[index[2]])

    stab_circuit.h(qbReg[index[3]])
    stab_circuit.cz(anReg[1], qbReg[index[3]])
    stab_circuit.h(anReg[1])
    stab_circuit.h(qbReg[index[3]])

    stab_circuit.measure(anReg[1], clReg[i])
    if reset:
        stab_circuit.reset(anReg[1])

    return stab_circuit

def run_stabilizer(qbReg, anReg, clReg, reset=True):
    stab_circuit = QuantumCircuit(qbReg, anReg, clReg)
    stab_circuit += measure_stabilizer(qbReg, anReg, clReg, 0, reset)
    stab_circuit += measure_stabilizer(qbReg, anReg, clReg, 1, reset)
    stab_circuit += measure_stabilizer(qbReg, anReg, clReg, 2, reset)
    stab_circuit += measure_stabilizer(qbReg, anReg, clReg, 3, reset)
    return stab_circuit



# Correct possible errors
def recovery_scheme(qbReg, clReg, reset=True):

    recovery_circuit = QuantumCircuit(qbReg, clReg)

    # If the ancilla is reset to |0> between measurements
    if reset:
        recovery_circuit.x(qbReg[1]).c_if(clReg, 1)
        recovery_circuit.z(qbReg[4]).c_if(clReg, 2)
        recovery_circuit.x(qbReg[2]).c_if(clReg, 3)
        recovery_circuit.z(qbReg[2]).c_if(clReg, 4)
        recovery_circuit.z(qbReg[0]).c_if(clReg, 5)
        recovery_circuit.x(qbReg[3]).c_if(clReg, 6)
        recovery_circuit.x(qbReg[2]).c_if(clReg, 7)
        recovery_circuit.z(qbReg[2]).c_if(clReg, 7)
        recovery_circuit.x(qbReg[0]).c_if(clReg, 8)
        recovery_circuit.z(qbReg[3]).c_if(clReg, 9)
        recovery_circuit.z(qbReg[1]).c_if(clReg, 10)
        recovery_circuit.x(qbReg[1]).c_if(clReg, 11)
        recovery_circuit.z(qbReg[1]).c_if(clReg, 11)
        recovery_circuit.x(qbReg[4]).c_if(clReg, 12)
        recovery_circuit.x(qbReg[0]).c_if(clReg, 13)
        recovery_circuit.z(qbReg[0]).c_if(clReg, 13)
        recovery_circuit.x(qbReg[4]).c_if(clReg, 14)
        recovery_circuit.z(qbReg[4]).c_if(clReg, 14)
        recovery_circuit.x(qbReg[3]).c_if(clReg, 15)
        recovery_circuit.z(qbReg[3]).c_if(clReg, 15)

    # If the ancilla is NOT reset between measurements
    else:
        recovery_circuit.x(qbReg[2]).c_if(clReg, 1)
        recovery_circuit.x(qbReg[3]).c_if(clReg, 2)
        recovery_circuit.z(qbReg[0]).c_if(clReg, 3)
        recovery_circuit.x(qbReg[4]).c_if(clReg, 4)
        recovery_circuit.z(qbReg[3]).c_if(clReg, 5)
        recovery_circuit.x(qbReg[3]).c_if(clReg, 5)
        recovery_circuit.z(qbReg[1]).c_if(clReg, 6)
        recovery_circuit.z(qbReg[3]).c_if(clReg, 7)
        recovery_circuit.x(qbReg[0]).c_if(clReg, 8)
        recovery_circuit.z(qbReg[1]).c_if(clReg, 9)
        recovery_circuit.x(qbReg[1]).c_if(clReg, 9)
        recovery_circuit.z(qbReg[4]).c_if(clReg, 10)
        recovery_circuit.x(qbReg[4]).c_if(clReg, 10)
        recovery_circuit.z(qbReg[0]).c_if(clReg, 11)
        recovery_circuit.x(qbReg[0]).c_if(clReg, 11)
        recovery_circuit.z(qbReg[2]).c_if(clReg, 12)
        recovery_circuit.z(qbReg[2]).c_if(clReg, 13)
        recovery_circuit.x(qbReg[2]).c_if(clReg, 13)
        recovery_circuit.z(qbReg[4]).c_if(clReg, 14)
        recovery_circuit.x(qbReg[1]).c_if(clReg, 15)

    return recovery_circuit

def define_circuit(n_cycles, snapshot_type='density_matrix'):
    '''Creates the entire circuit and returns it
    as an output. Input is the number of stabilizer
    cycles to perform'''

    # TODO: Transpile each segment seperated by a barrer or snapshop seperately and combine afterwards
    # TODO: Change the ancilla reg to use AncillaRegister()
    # Define the registers
    # The 5 qubits to encode the state in
    qb = QuantumRegister(5, 'code_qubit')
    # The two ancilla qubits (one of them is unused)
    an = QuantumRegister(2, 'ancilla_qubit')
    # Classical register for registering the syndromes
    cr = ClassicalRegister(4, 'syndrome_bit')
    # Readout of the final state at the end for statistics
    readout = ClassicalRegister(5, 'readout')

    circuit = QuantumCircuit(cr, readout, an, qb)

    # Prepare the input
    # circuit.x(qb[0])  # As an example, start in |1>

    # Encode the state
    circuit += encode_input(qb)
    circuit.snapshot('post_encoding', snapshot_type=snapshot_type)

    # Stabilizers
    for i in range(n_cycles):
        circuit += run_stabilizer(qb, an, cr, reset=False)
        circuit += recovery_scheme(qb, cr, reset=False)
        circuit.snapshot('stabilizer_' + str(i), snapshot_type=snapshot_type)

    # Readout of the encoded state
    # Measure at the end of the run
    circuit.measure(qb, readout)
    circuit.snapshot('post_measure', snapshot_type=snapshot_type)

    return circuit

def transpile_circuit(circuit, optimization_level=2, repeats=1):
    # % Transpiler
    routing_method = 'sabre'  # basic lookahead stochastic sabre
    # initial_layout = {qb[0]: 0,
    #                   qb[1]: 1,
    #                   qb[2]: 2,
    #                   qb[3]: 3,
    #                   qb[4]: 4,
    #                   an[0]: 5,
    #                   an[1]: 6}
    # initial_layout = {an[0]: 0,
    #                   an[1]: 1,
    #                   qb[0]: 2,
    #                   qb[1]: 3,
    #                   qb[2]: 4,
    #                   qb[3]: 5,
    #                   qb[4]: 6}
    initial_layout = None  # Overwriting the above layout
    layout_method = 'sabre'  # trivial 'dense', 'noise_adaptive' sabre
    translation_method = None  # 'unroller',  translator , synthesis
    transpiled_circuit = shortest_transpile_from_distribution(circuit, repeats=repeats, routing_method=routing_method, initial_layout=initial_layout,
                                                                  layout_method=layout_method, translation_method=translation_method, optimization_level=optimization_level
                                                            #   ,coupling_map = WAQCT_device_properties['coupling_map'])
                                                            #   ,**{'basis_gates': ['id', 'u1', 'u2', 'u3', 'cz','iswap']})
                                                              , **WAQCT_device_properties)

    print('Final depth = ', transpiled_circuit.depth())
    print('Final gates = ', transpiled_circuit.count_ops())
    # display(transpiled_circuit.draw(output='mpl'))

    return transpiled_circuit  # CHANGE TO TRANSPILED CIRCUIT

# %% Med stabilizer mätning
snapshot_type = 'statevector'
qb = QuantumRegister(5, 'code_qubit')
# The two ancilla qubits (one of them is unused)
an = QuantumRegister(2, 'ancilla_qubit')
# Classical register for registering the syndromes
cr = ClassicalRegister(4, 'syndrome_bit')
# Readout of the final state at the end for statistics


circuit = QuantumCircuit(cr, an, qb)

circuit += encode_input(qb)
circuit.snapshot('post_encoding', snapshot_type=snapshot_type)

# Stabilizers
cycles = 1

circuit += measure_stabilizer(qb, an, cr, 0, reset=False)
# # circuit += recovery_scheme(qb, cr, reset=False)

circuit.snapshot('stabilizer_0', snapshot_type=snapshot_type)
# circuit += measure_stabilizer(qb, an, cr, 1, reset=False)
# circuit += recovery_scheme(qb, cr, reset=False)
# circuit.snapshot('stabilizer_1', snapshot_type=snapshot_type)
optimization_level=2
routing_method = 'lookahead' # basic lookahead stochastic sabre
initial_layout = None  # Overwriting the above layout
layout_method = 'trivial'  # trivial 'dense', 'noise_adaptive' sabre
translation_method = None  # 'unroller',  translator , synthesis
transpiled_circ = transpile(circuit, optimization_level=optimization_level,initial_layout = initial_layout,
layout_method=layout_method,routing_method=routing_method,translation_method=None
# ,**{'basis_gates': ['id', 'u1', 'u2', 'u3', 'cz','iswap']})
                            , **WAQCT_device_properties)

#%%
n_shots = 100
results1 = execute(
    transpiled_circ,
    Aer.get_backend('qasm_simulator'),
    shots=n_shots
).result()

results2 = execute(
    transpiled_circ,
    Aer.get_backend('qasm_simulator'),
    shots=1
).result()

# state1 = results1.data()['snapshots'][snapshot_type]
# state2 = results2.data()['snapshots'][snapshot_type]
# print('post_encoding ', state_fidelity(
#     state1['post_encoding'][0], state2['post_encoding'][0]))
state_vectors_noisy = results1.data()['snapshots']['statevector']
state_vectors_ideal = results2.data()['snapshots']['statevector']


running_fidelity = np.zeros([n_shots, cycles+1])

for i in range(n_shots):  # n_shots
    running_fidelity[i, 0] = state_fidelity(
        state_vectors_ideal['post_encoding'][0], state_vectors_noisy['post_encoding'][i])
    for j in range(cycles):
        running_fidelity[i, j+1] = state_fidelity(
            state_vectors_ideal['stabilizer_' + str(j)][0], state_vectors_noisy['stabilizer_' + str(j)][i])

print('fidelities ', np.sum(running_fidelity, 0) / n_shots)

# print('post_measure ', state_fidelity(
#     state1['post_measure'][0], state2['post_measure'][0]))
counts = results1.get_counts()
print(counts)
counts = results2.get_counts()
print(counts)
# %%
transpiled_circ.draw()

#%% only encoding
circuit = QuantumCircuit(cr, an, qb)

circuit += encode_input(qb)
circuit.snapshot('post_encoding', snapshot_type=snapshot_type)

optimization_level=2
routing_method = 'lookahead' # basic lookahead stochastic sabre
initial_layout = None  # Overwriting the above layout
layout_method = 'trivial'  # trivial 'dense', 'noise_adaptive' sabre
translation_method = None  # 'unroller',  translator , synthesis
transpiled_circ = transpile(circuit, optimization_level=optimization_level,initial_layout = initial_layout,
layout_method=layout_method,routing_method=routing_method,translation_method=None
# ,**{'basis_gates': ['id', 'u1', 'u2', 'u3', 'cz','iswap']})
                            , **WAQCT_device_properties)
n_shots = 1
results1 = execute(
    transpiled_circ,
    Aer.get_backend('qasm_simulator'),
    shots=n_shots
).result()

results2 = execute(
    circuit,
    Aer.get_backend('qasm_simulator'),
    shots=1
).result()

state_vectors_noisy = results1.data()['snapshots']['statevector']
state_vectors_ideal = results2.data()['snapshots']['statevector']


running_fidelity = np.zeros([n_shots, 1])

for i in range(1):  # n_shots
    running_fidelity[i, 0] = state_fidelity(
        state_vectors_ideal['post_encoding'][0], state_vectors_noisy['post_encoding'][i])

print('fidelities ', np.sum(running_fidelity, 0) / n_shots)

# print('post_measure ', state_fidelity(
#     state1['post_measure'][0], state2['post_measure'][0]))
counts = results1.get_counts()
print(counts)
counts = results2.get_counts()
print(counts)
# %% Testa hela vår krets
optimization_level = 2
repeats = 1
n_cycles = 1

snapshot_type = 'statevector'  #'density_matrix'  # 
circuit = define_circuit(n_cycles, snapshot_type=snapshot_type)

routing_method = 'basic' # basic lookahead stochastic sabre
initial_layout = None  # Overwriting the above layout
layout_method = 'trivial'  # trivial 'dense', 'noise_adaptive' sabre
translation_method = None  # 'unroller',  translator , synthesis
transpiled_circuit = shortest_transpile_from_distribution(circuit, repeats=repeats, routing_method=routing_method, initial_layout=initial_layout,
                                                                layout_method=layout_method, translation_method=translation_method, optimization_level=optimization_level
                                                        #   ,coupling_map = WAQCT_device_properties['coupling_map'])
                                                        #   ,**{'basis_gates': ['id', 'u1', 'u2', 'u3', 'cz','iswap']})
                                                            , **WAQCT_device_properties)

# print('Final depth = ', transpiled_circuit.depth())
# print('Final gates = ', transpiled_circuit.count_ops())
# # transpiled_circuit = transpile_circuit(
# #     circuit, optimization_level=optimization_level, repeats=repeats)
# transpiled_circuit._layout

# %% 

# noise = pauli_noise_model(0.001, 0.00, 0.0)

n_shots = 100
results_noisy = execute(
    transpiled_circuit,
    Aer.get_backend('qasm_simulator'),
    noise_model=None,
    shots=n_shots
).result()

results_ideal = execute(
    transpiled_circuit,
    Aer.get_backend('qasm_simulator'),
    noise_model=None,
    shots=1
).result()

def _get_fidelities_vec(results_noisy, results_ideal):
    # Get the state vectors
    state_vectors_noisy = results_noisy.data()['snapshots']['statevector']
    state_vectors_ideal = results_ideal.data()['snapshots']['statevector']

    running_fidelity = np.zeros([n_shots, n_cycles+1])

    for i in range(n_shots):  # n_shots
        running_fidelity[i, 0] = state_fidelity(
            state_vectors_ideal['post_encoding'][0], state_vectors_noisy['post_encoding'][i])
        for j in range(n_cycles):
            running_fidelity[i, j+1] = state_fidelity(
                state_vectors_ideal['stabilizer_' + str(j)][0], state_vectors_noisy['stabilizer_' + str(j)][i])
    return np.sum(running_fidelity, 0) / n_shots


def get_fidelities(results_noisy, results_ideal, snapshot_type):
    # logical0 = logical_0_transp  # logical[0]  
    return _get_fidelities_vec(results_noisy, results_ideal)


logical_state = get_fidelities(
    results_noisy, results_ideal, snapshot_type=snapshot_type)

# Plotting
sns.set_context('talk', rc={"lines.linewidth": 2.5})
default_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                  'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

fig = plt.figure(figsize=[10, 6])
plt.plot(logical_state, marker='o', label=r'$p_{error}$=')
plt.xticks(ticks=range(n_cycles+1))
plt.xlabel('Number of cycles')
plt.title('Average fidelity across stabilizer cycles')
plt.legend()
plt.show()
# %%
