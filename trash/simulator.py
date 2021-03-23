"""
This file is unused at the moment
"""
# %% Import modules
import pickle
from qiskit.quantum_info import purity
from numpy.core.fromnumeric import repeat
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from qiskit import (QuantumCircuit, 
                    QuantumRegister, 
                    ClassicalRegister, 
                    AncillaRegister, 
                    execute, 
                    Aer)
from qiskit.visualization import plot_histogram
from IPython.display import display

from qiskit.quantum_info import state_fidelity
from qiskit.providers.aer.extensions.snapshot_statevector import *

# Import our own files
from custom_noise_models import pauli_noise_model, amplitude_damping_error
from custom_transpiler import (shortest_transpile_from_distribution, 
                               WAQCT_device_properties,
                               _add_custom_device_equivalences
                               )
# %% Defining useful functions

# Note that this does not consider our setup
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


def logical_states():
    logical_0 = np.zeros(2**5)
    logical_0[0b00000] = 1/4
    logical_0[0b10010] = 1/4
    logical_0[0b01001] = 1/4
    logical_0[0b10100] = 1/4
    logical_0[0b01010] = 1/4
    logical_0[0b11011] = -1/4
    logical_0[0b00110] = -1/4
    logical_0[0b11000] = -1/4
    logical_0[0b11101] = -1/4
    logical_0[0b00011] = -1/4
    logical_0[0b11110] = -1/4
    logical_0[0b01111] = -1/4
    logical_0[0b10001] = -1/4
    logical_0[0b01100] = -1/4
    logical_0[0b10111] = -1/4
    logical_0[0b00101] = 1/4

    logical_1 = np.zeros(2**5)
    logical_1[0b11111] = 1/4
    logical_1[0b01101] = 1/4
    logical_1[0b10110] = 1/4
    logical_1[0b01011] = 1/4
    logical_1[0b10101] = 1/4
    logical_1[0b00100] = -1/4
    logical_1[0b11001] = -1/4
    logical_1[0b00111] = -1/4
    logical_1[0b00010] = -1/4
    logical_1[0b11100] = -1/4
    logical_1[0b00001] = -1/4
    logical_1[0b10000] = -1/4
    logical_1[0b01110] = -1/4
    logical_1[0b10011] = -1/4
    logical_1[0b01000] = -1/4
    logical_1[0b11010] = 1/4

    # Add two ancillas in |0>
    an0 = np.zeros(2**2)
    an0[0] = 1.0

    logical_1 = np.kron(logical_1, an0)
    logical_0 = np.kron(logical_0, an0)

    #logical_0 = np.kron(an0, logical_0)
    #logical_1 = np.kron(an0, logical_1)
    return [logical_0, logical_1]


def define_circuit(n_cycles, snapshot_type='density_matrix'):
    '''Creates the entire circuit and returns it
    as an output. Input is the number of stabilizer
    cycles to perform'''

    # TODO: Transpile each segment seperated by a barrer or snapshop seperately 
    #       and combine afterwards
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


def get_logical_0(transpiled_circuit, snapshot_type):
    ''' Run the actual ciruit and get the statevector after encoding
        This is a bit problematic for several reasons so that's why I'm trying
        to avoid it
    '''
    results = execute(
        transpiled_circuit,  # NOT RUNNING THE TRANSPILED CIRCUIT AT THE MOMENT
        Aer.get_backend('qasm_simulator'),
        noise_model=None,
        shots=1
    ).result()

    # Get the state vectors
    state_vectors = results.data()['snapshots'][snapshot_type]
    sv_post_encoding = state_vectors['post_encoding'][0]
    if snapshot_type == 'density_matrix':
        sv_post_encoding = sv_post_encoding['value']
    return sv_post_encoding


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


# %% Create circuit
optimization_level = 2
repeats = 3
n_cycles = 1
snapshot_type = 'statevector'  #'density_matrix'  # 
circuit = define_circuit(n_cycles, snapshot_type=snapshot_type)

transpiled_circuit = transpile_circuit(
    circuit, optimization_level=optimization_level, repeats=repeats)
transpiled_circuit._layout
# logical = logical_states()

# logical_0_transp = get_logical_0(
#     transpiled_circuit, snapshot_type=snapshot_type)
# print('Traspiled 0 fidelity ', state_fidelity(logical_0_transp, logical[0]))

# Saves the compiled circuie
if True:
    with open('transpiled_circuit.dat', 'wb') as transpiled_circuit_file:
        pickle.dump(transpiled_circuit, transpiled_circuit_file)
# %% (Load and) run the circuit
if 'transpiled_circuit' not in locals():
    with open('transpiled_circuit.dat', 'rb') as transpiled_circuit_file:
        transpiled_circuit = pickle.load(transpiled_circuit_file)

noise = pauli_noise_model(0.001, 0.00, 0.0)

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
# Extract data from simulations

def _get_fidelities_mat(results_noisy, results_ideal):
    state_vectors_noisy = results_noisy.data()['snapshots']['density_matrix']
    state_vectors_ideal = results_ideal.data()['snapshots']['density_matrix']

    running_fidelity = np.zeros([n_cycles+1])
    running_fidelity[0] = state_fidelity(state_vectors_ideal['post_encoding'][0]['value'],
                                         state_vectors_noisy['post_encoding'][0]['value'])
    print('Purity of encoded state = ', purity(
        state_vectors_noisy['post_encoding'][0]['value']))
    for j in range(n_cycles):
        running_fidelity[j+1] = state_fidelity(state_vectors_ideal['stabilizer_' + str(j)][0]['value'],
                                               state_vectors_noisy['stabilizer_' + str(j)][0]['value'])
    return running_fidelity


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
    # logical0 = logical_0_transp  # logical[0]  #
    if snapshot_type == 'density_matrix':
        return _get_fidelities_mat(results_noisy, results_ideal)
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


# %% Demonstrate how density matrices avoid the issue of permutations
if snapshot_type == 'density_matrix':
    state_vectors_ideal = results_ideal.data()['snapshots']['density_matrix']

    running_fidelity = np.zeros([n_cycles+1])
    running_fidelity[0] = state_fidelity(logical_0_transp,
                                        state_vectors_ideal['post_encoding'][0]['value'])
    for j in range(n_cycles):
        running_fidelity[j+1] = state_fidelity(logical_0_transp,
                                            state_vectors_ideal['stabilizer_' + str(j)][0]['value'])
    print('Running fidelity without noise ', running_fidelity)
    print('It is unity since transpiled snapshots of density matrices are not permuted')
# %%
# circuit.draw(output='mpl') # If it does not work, simply remove mpl: circuit.draw()
counts = results_noisy.get_counts()
print(counts)
counts = results_ideal.get_counts()
print(counts)
# plot_histogram(counts)
# circuit.draw(output='mpl')

# %%
state1= results_noisy.data()['snapshots']['statevector']['post_encoding'][0]
print(state1[0])
state2 = results_ideal.data()['snapshots']['statevector']['post_encoding'][0]
print(state2[0])
print(np.vdot(state1,state2))
# for i in range(n_shots):
#     for j in range(n_cycles):
#         print('shot= ',i,', cycle = ',j,', norm = ',np.linalg.norm(state_vectors['stabilizer_' + str(j)][i]))

# Numpy arrays to store data in (Maybe save as file later?)

# sv_stabilizer = np.zeros([128, n_shots, n_cycles])

# A slow nested for-loop to gather all state vectors and fidelities

# Probabilities of remaining in correct state
# preserved_state_count = np.zeros(n_cycles+1)
# for i in range(n_shots):

#     state_is_preserved = True
#     if logical_state[1, i, 0] > 0.95:
#         preserved_state_count[0] += 1.
#     else:
#         state_is_preserved = False

#     for j in range(n_cycles):
#         if state_is_preserved:

#             if logical_state[1, i, j+1] > 0.95:
#                 preserved_state_count[j+1] += 1.
#             else:
#                 state_is_preserved = False

# preserved_state_count /= n_shots
# %
# fig = plt.figure(figsize=[10, 6])
# plt.plot(preserved_state_count, marker='o', label=r'$p_{error}$=')
# plt.xticks(ticks=range(n_cycles+1))
# plt.xlabel('Number of cycles')
# plt.title('Probability of remaining in original state')
# plt.legend()

# %%

# %%

# %%
