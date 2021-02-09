# %% Import modules
from numpy.core.fromnumeric import repeat
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister, execute, Aer
from qiskit.visualization import plot_histogram
from IPython.display import display

from qiskit.quantum_info import state_fidelity
from qiskit.providers.aer.extensions.snapshot_statevector import *

# Import our own files
from custom_noise_models import pauli_noise_model
from custom_transpiler import shortest_transpile_from_distribution, WAQCT_device_properties

# %% Defining useful functions

# Not that this does not consider our setup


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


# %% Define our registers and circuit
qb = QuantumRegister(5, 'code_qubit')     # The 5 qubits to encode the state in
# The two ancilla qubits (one of them is unused)
an = QuantumRegister(2, 'ancilla_qubit')
# Classical register for registering the syndromes
cr = ClassicalRegister(4, 'syndrome_bit')
# Readout of the final state at the end for statistics
readout = ClassicalRegister(5, 'readout')

# %% Running the quantum circuit


def define_circuit(n_cycles):
    '''Creates the entire circuit and returns it
    as an output. Input is the number of stabilizer
    cycles to perform'''
    # Define our registers
    # TODO: Change the ancilla reg to use AncillaRegister()
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
    circuit.snapshot_statevector('post_encoding')

    # Stabilizers
    for i in range(n_cycles):
        circuit += run_stabilizer(qb, an, cr, reset=False)
        circuit += recovery_scheme(qb, cr, reset=False)
        circuit.snapshot_statevector('stabilizer_' + str(i))

    # Readout of the encoded state
    # Measure at the end of the run
    circuit.measure(qb, readout)
    circuit.snapshot_statevector('post_measure')

    return circuit


def transpile_circuit(circuit):
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
    optimization_level = 0
    repeats = 1
    transpiled_circuit = shortest_transpile_from_distribution(circuit, repeats=repeats, routing_method=routing_method, initial_layout=initial_layout,
                                                              layout_method=layout_method, translation_method=translation_method,
                                                              optimization_level=optimization_level,
    **{'basis_gates': ['id', 'u1', 'u2', 'u3', 'swap', 'cz','CNOT']})
    # , **WAQCT_device_properties
    print('Final depth = ', transpiled_circuit.depth())
    print('Final gates = ', transpiled_circuit.count_ops())
    # display(transpiled_circuit.draw(output='mpl'))

    return transpiled_circuit  # CHANGE TO TRANSPILED CIRCUIT


# %% Create the circuit
n_cycles = 0

circuit = QuantumCircuit(cr, readout, an, qb)
circuit += encode_input(qb)
circuit.snapshot_statevector('post_encoding')

# circuit = define_circuit(0)
transpiled_circuit = transpile_circuit(circuit)
transpiled_circuit._layout

# TODO: Save compiled circuit (using e.g. pickle, and load if resired)
# import pickle
# with open('circuit.dat', 'wb') as transpiled_circuit_file:
#     pickle.dump(transpiled_circuit, transpiled_circuit_file)

# with open('circuit.dat', 'rb') as transpiled_circuit_file:
#     transpiled_circuit = pickle.load(transpiled_circuit_file)
# %% Delete this when the bug is found
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.preset_passmanagers\
    import level_0_pass_manager, level_1_pass_manager, level_2_pass_manager, level_3_pass_manager
from qiskit.transpiler import passes
[pass_ for pass_ in dir(passes) if pass_[0].isupper()]
from qiskit.transpiler.passes import BasisTranslator
from qiskit.transpiler.passes import UnrollCustomDefinitions
from IPython.display import display
from qiskit.transpiler import TransformationPass
from qiskit.converters import dag_to_circuit

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
translation_method = None
optimization_level = 0
pm = level_0_pass_manager(PassManagerConfig(routing_method=routing_method, initial_layout=initial_layout,
                                            layout_method=layout_method, translation_method=translation_method))  # the drawing functionality doesn't work if coupling_map is specified
pm.draw()

def callback_func(**kwargs):
    pass_ = kwargs['pass_']
    dag = kwargs['dag']
    time = kwargs['time']
    property_set = kwargs['property_set']
    count = kwargs['count']

    if isinstance(pass_,TransformationPass):
        circuit_tmp = dag_to_circuit(dag)
        print('pass namne: ',pass_.name())
        display(circuit_tmp.draw(output='mpl'))
print('before transpilation ')
display(circuit.draw(output='mpl'))
pm.run(circuit,callback=callback_func)
pm.passes()
# %% Extract logical 0 and 1
# Get the two logical states before transpilation (and hence permutated qubits)
logical = logical_states()


def get_logical_states_v1(transpiled_circuit):
    ''' Run the actual ciruit and get the statevector after encoding
        This is a bit problematic for several reasons so that's why I'm trying to avoid it
    '''
    results = execute(
        transpiled_circuit,  # NOT RUNNING THE TRANSPILED CIRCUIT AT THE MOMENT
        Aer.get_backend('qasm_simulator'),
        noise_model=None,
        shots=1
    ).result()

    # Get the state vectors
    state_vectors = results.data()['snapshots']['statevector']
    sv_post_encoding = state_vectors['post_encoding']
    return sv_post_encoding[0]


def get_logical_states_v2(transpiled_circuit):
    ''' Get the permutation from transpiled_circuit._layout and attempt to redo it
     after encoding using the initialize method, by permuting the qubits in the second argument
    see: https://github.com/Qiskit/qiskit-tutorials/blob/master/tutorials/circuits/3_summary_of_quantum_operations.ipynb
    '''
    circuit_log0 = QuantumCircuit(*transpiled_circuit._layout.get_registers())
    circuit_log0.initialize(
        logical[0], [transpiled_circuit._layout[i] for i in range(7)])
    results = execute(
        circuit_log0,  # NOT RUNNING THE TRANSPILED CIRCUIT AT THE MOMENT
        Aer.get_backend('statevector_simulator'),
    ).result()
    return results.get_statevector(circuit_log0)


def get_logical_states_v3(transpiled_circuit):
    ''' Same as v3 but creates new regusters instead of getting them from
    transpiled_circuit._layout.get_registers()
    '''
    # Get analytical encoded state (with permutated qubits)
    qb = QuantumRegister(5, 'code_qubit')
    # The two ancilla qubits (one of them is unused)
    an = QuantumRegister(2, 'ancilla_qubit')

    circuit_log0 = QuantumCircuit(an, qb)
    circuit_log0.initialize(
        logical[0], [transpiled_circuit._layout[i] for i in range(7)])
    results = execute(
        circuit_log0,  # NOT RUNNING THE TRANSPILED CIRCUIT AT THE MOMENT
        Aer.get_backend('statevector_simulator'),
    ).result()
    return results.get_statevector(circuit_log0)


def get_logical_states_v4(transpiled_circuit):
    ''' Get the permutation from transpiled_circuit._layout and attempt to redo 
    it after encoding using Permutation()
    '''
    from qiskit.circuit.library import Permutation
    # Get analytical encoded state (with permutated qubits)
    qb = QuantumRegister(5, 'code_qubit')
    # The two ancilla qubits (one of them is unused)
    an = QuantumRegister(2, 'ancilla_qubit')

    circuit_log0 = QuantumCircuit(an, qb)
    circuit_log0 += encode_input(qb)
    circuit_log0 += Permutation(7, [transpiled_circuit._layout[key]
                                    for key in circuit.qubits])
    results = execute(
        circuit_log0,  # NOT RUNNING THE TRANSPILED CIRCUIT AT THE MOMENT
        Aer.get_backend('statevector_simulator'),
    ).result()
    return results.get_statevector(circuit_log0)


def get_logical_states_v5(transpiled_circuit):
    ''' Run the actual ciruit and get the statevector after encoding
        This is a bit problematic for several reasons so that's why I'm trying to avoid it
    '''
    results = execute(
        transpiled_circuit,  # NOT RUNNING THE TRANSPILED CIRCUIT AT THE MOMENT
        Aer.get_backend('statevector_simulator'),
        noise_model=None,
        shots=1
    ).result()

    # Get the state vectors
    state_vectors = results.data()['snapshots']['statevector']
    sv_post_encoding = state_vectors['post_encoding']
    return results.get_statevector(transpiled_circuit)


print('unpermutated', np.where(logical[0] != 0)[0])  # unpermutated zero
log0_v1 = get_logical_states_v1(transpiled_circuit)
print('v1', np.where(log0_v1 != 0)[0])  # This should be the
log0_v5 = get_logical_states_v5(transpiled_circuit)
print('v5', np.where(log0_v5 != 0)[0])
log0_v2 = get_logical_states_v2(transpiled_circuit)
log0_v3 = get_logical_states_v3(transpiled_circuit)
log0_v4 = get_logical_states_v4(transpiled_circuit)


# Print which indices are non-zero, to quickly see if the state is correct



print('v2', np.where(log0_v2 != 0)[0])  # Doesn't give the same as v1 !?
# Gives the same as v2 (as it should), but still not right?
print('v3', np.where(log0_v3 != 0)[0])
print('v4', np.where(log0_v4 != 0)[0])  # Doesn't work at all for some reason


state_fidelity(log0_v1, log0_v2)


# %% Run the circuit
# Noise model, no input gives no noise
noise = pauli_noise_model(0.001, 0.00, 0.0)

n_shots = 2000
results = execute(
    circuit,  # NOT RUNNING THE TRANSPILED CIRCUIT AT THE MOMENT
    Aer.get_backend('qasm_simulator'),
    noise_model=noise,
    shots=n_shots
).result()


# % Extract data from simulations

counts = results.get_counts()

# Get the state vectors
state_vectors = results.data()['snapshots']['statevector']
sv_post_encoding = state_vectors['post_encoding']
sv_post_measure = state_vectors['post_measure']
# for i in range(n_shots):
#     for j in range(n_cycles):
#         print('shot= ',i,', cycle = ',j,', norm = ',np.linalg.norm(state_vectors['stabilizer_' + str(j)][i]))

# Numpy arrays to store data in (Maybe save as file later?)

logical_state = np.zeros([2, n_shots, n_cycles+1])

# sv_stabilizer = np.zeros([128, n_shots, n_cycles])
logical = [get_logical_states_v2(
    transpiled_circuit), get_logical_states_v2(transpiled_circuit)]
# A slow nested for-loop to gather all state vectors and fidelities
print('Running statistics...')
for i in range(n_shots):

    logical_state[:, i, 0] = [state_fidelity(logical[0], sv_post_encoding[i]),
                              state_fidelity(logical[1], sv_post_encoding[i])]
    for j in range(n_cycles):

        # sv_stabilizer[:, i, j] = state_vectors['stabilizer_' + str(j)][i]

        logical_state[:, i, j+1] = [state_fidelity(logical[0], state_vectors['stabilizer_' + str(j)][i]),
                                    state_fidelity(logical[1], state_vectors['stabilizer_' + str(j)][i])]


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

# %% Plotting
# For figures
sns.set_context('talk', rc={"lines.linewidth": 2.5})
default_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                  'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

x = np.sum(logical_state[0, :, :], 0) / (n_shots+1)
fig = plt.figure(figsize=[10, 6])
plt.plot(x, marker='o', label=r'$p_{error}$=')
plt.xticks(ticks=range(n_cycles+1))
plt.xlabel('Number of cycles')
plt.title('Average fidelity across stabilizer cycles')
plt.legend()
plt.show()
# %
# fig = plt.figure(figsize=[10, 6])
# plt.plot(preserved_state_count, marker='o', label=r'$p_{error}$=')
# plt.xticks(ticks=range(n_cycles+1))
# plt.xlabel('Number of cycles')
# plt.title('Probability of remaining in original state')
# plt.legend()
# %%
# circuit.draw(output='mpl') # If it does not work, simply remove mpl: circuit.draw()

print(counts)
# plot_histogram(counts)
# circuit.draw(output='mpl')
