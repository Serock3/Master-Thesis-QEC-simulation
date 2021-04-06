# %%
from qiskit.transpiler import PassManager, CouplingMap, Layout
from qiskit.transpiler.passes import UnrollCustomDefinitions
from qiskit.transpiler.passes import BasisTranslator
from qiskit.transpiler import passes
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.preset_passmanagers\
    import level_0_pass_manager, level_1_pass_manager, level_2_pass_manager, level_3_pass_manager
from qiskit.circuit import Clbit
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.transpiler import TransformationPass
from IPython.display import display
from qiskit.circuit.equivalence_library import StandardEquivalenceLibrary, SessionEquivalenceLibrary
from qiskit.circuit.library.standard_gates import iSwapGate, SwapGate, SGate, CZGate
from qiskit.circuit import EquivalenceLibrary
import logging
import warnings
from qiskit.visualization import plot_circuit_layout
from qiskit.compiler import transpile
from qiskit.ignis.verification.topological_codes import circuits
from qiskit.quantum_info.operators.symplectic import stabilizer_table
from qiskit.ignis.verification.topological_codes import fitters
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
from qiskit.providers.aer.noise import NoiseModel
from qiskit.quantum_info import state_fidelity
import math
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator, UnitarySimulator
from qiskit.tools.visualization import plot_histogram, plot_state_city, plot_state_hinton
from qiskit import Aer, execute
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from math import pi
from matplotlib import get_backend
import numpy as np
from qiskit import *
# Useful additional packages
import matplotlib.pyplot as plt
from qiskit.circuit.classicalregister import Clbit
from simulator_program.custom_noise_models import pauli_noise_model
from simulator_program.custom_transpiler import shortest_transpile_from_distribution, WACQT_device_properties
from simulator_program.simulator import *
# %matplotlib inline

#  Define our registers and circuit
qb = QuantumRegister(5, 'code_qubit')     # The 5 qubits to encode the state in
# The two ancilla qubits (one of them is unused)
an = QuantumRegister(2, 'ancilla_qubit')
# Classical register for registering the syndromes
cr = ClassicalRegister(4, 'syndrome_bit')
# Readout of the final state at the end for statistics
readout = ClassicalRegister(5, 'readout')
# %%

q = QuantumRegister(1)
c = ClassicalRegister(1)
qc = QuantumCircuit(q, c)
qc.x(q)
qc.measure(q[0], c)
qc.draw()
# %%
backend = BasicAer.get_backend('qasm_simulator')
qc = QuantumCircuit(q, c)
qc.x(q[0]).c_if(c, 0)
qc.measure(q, c)
qc.draw()
job = execute(qc, backend, shots=1024)
job.result().get_counts(qc)
# %%
# List Aer backends
Aer.backends()
# %%
# Construct quantum circuit
circ = QuantumCircuit(2, 2)
circ.h(0)
circ.cx(0, 1)
circ.measure([0, 1], [0, 1])

# Select the QasmSimulator from the Aer provider
simulator = Aer.get_backend('qasm_simulator')

# Execute and get counts
result = execute(circ, simulator).result()
counts = result.get_counts(circ)
plot_histogram(counts, title='Bell-State counts')
# %%
# Construct quantum circuit with measure
circ = QuantumCircuit(2, 2)
circ.h(0)
circ.cx(0, 1)
circ.measure([0, 1], [0, 1])

# Select the StatevectorSimulator from the Aer provider
simulator = Aer.get_backend('statevector_simulator')

# Execute and get counts
result = execute(circ, simulator).result()
statevector = result.get_statevector(circ)
# %%
circ = QuantumCircuit(2)
circ.h(0)
circ.cx(0, 1)

# Select the UnitarySimulator from the Aer provider
simulator = Aer.get_backend('unitary_simulator')

# Execute and get counts
result = execute(circ, simulator).result()
unitary = result.get_unitary(circ)
print("Circuit unitary:\n", unitary)
plot_state_city(statevector, title='Bell state post-measurement')
# %% Arbitrary initialization

desired_vector = 1.5*np.array([
    1 / math.sqrt(16) * complex(0, 1),
    1 / math.sqrt(8) * complex(1, 0),
    1 / math.sqrt(16) * complex(1, 1),
    0,
    0,
    1 / math.sqrt(8) * complex(1, 2),
    1 / math.sqrt(16) * complex(1, 0),
    0])

desired_vector /= np.linalg.norm(desired_vector)
q = QuantumRegister(3)

qc = QuantumCircuit(q)

qc.initialize(desired_vector, [q[0], q[1], q[2]])
qc.draw()

# %%
backend = BasicAer.get_backend('statevector_simulator')
job = execute(qc, backend)
qc_state = job.result().get_statevector(qc)
qc_state
# %%
state_fidelity(desired_vector, qc_state)
# %%


def get_noise(p_meas, p_gate):

    error_meas = pauli_error([('X', p_meas), ('I', 1 - p_meas)])
    error_gate1 = depolarizing_error(p_gate, 1)
    error_gate2 = error_gate1.tensor(error_gate1)

    noise_model = NoiseModel()
    # measurement error is applied to measurements
    noise_model.add_all_qubit_quantum_error(error_meas, "measure")
    # single qubit gate error is applied to x gates
    noise_model.add_all_qubit_quantum_error(error_gate1, ["x"])
    # two qubit gate error is applied to cx gates
    noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"])

    return noise_model


# %% Kolla på detta


# %% Kontrollera kod från stabilizer_code
# %%

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


def measure_stabilizer(qbReg, anReg, clReg, i):
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
    stab_circuit.h(anReg[0])
    stab_circuit.cz(anReg[0], qbReg[index[0]])
    stab_circuit.h(qbReg[index[0]])

    stab_circuit.cz(anReg[0], qbReg[index[1]])

    stab_circuit.cz(anReg[0], qbReg[index[2]])

    stab_circuit.h(qbReg[index[3]])
    stab_circuit.cz(anReg[0], qbReg[index[3]])
    stab_circuit.h(anReg[0])
    stab_circuit.h(qbReg[index[3]])

    stab_circuit.measure(anReg[0], clReg[i])
    stab_circuit.reset(anReg[0])
    return stab_circuit


def run_stabilizer(qbReg, anReg, clReg):
    stab_circuit = QuantumCircuit(qbReg, anReg, clReg)
    stab_circuit += measure_stabilizer(qbReg, anReg, clReg, 0)
    stab_circuit += measure_stabilizer(qbReg, anReg, clReg, 1)
    stab_circuit += measure_stabilizer(qbReg, anReg, clReg, 2)
    stab_circuit += measure_stabilizer(qbReg, anReg, clReg, 3)
    return stab_circuit

# Correct possible errors


def recovery_scheme(qbReg, clReg):

    recovery_circuit = QuantumCircuit(qbReg, clReg)
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
    an0[0] = 1

    logical_1 = np.kron(logical_1, an0)
    logical_0 = np.kron(logical_0, an0)
    return [logical_0, logical_1]


def noise_model():
    # Example error probabilities
    p_reset = 0.0000000
    p_meas = 0.00
    p_gate1 = 1.00

    # QuantumError objects
    error_reset = pauli_error([('X', p_reset), ('I', 1 - p_reset)])
    error_meas = pauli_error([('X', p_meas), ('I', 1 - p_meas)])
    error_gate1 = pauli_error([('X', p_gate1), ('I', 1 - p_gate1)])
    error_gate2 = error_gate1.tensor(error_gate1)

    # Add errors to noise model
    noise_bit_flip = NoiseModel()
    noise_bit_flip.add_all_qubit_quantum_error(error_reset, "reset")
    noise_bit_flip.add_all_qubit_quantum_error(error_meas, "measure")
    noise_bit_flip.add_all_qubit_quantum_error(
        error_gate1, ["u1", "u2", "u3", "x"])
#    noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ["cx"])
    return noise_bit_flip



# %% Test the encoding

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

# Test
print('<0|0>_L =', np.linalg.norm(logical_0))
print('<1|1>_L =', np.linalg.norm(logical_1))
print('<0|1>_L =', np.dot(logical_0, logical_1))


# Encode and check |0>_L
circuit = QuantumCircuit(cr, readout, qb)
encode_input(circuit, qb)
result = execute(circuit, Aer.get_backend('statevector_simulator')).result()
statevector = result.get_statevector()
print('Fidelity of encoded |0>_L', state_fidelity(logical_0, statevector))

# Encode and check |1>_L
circuit = QuantumCircuit(cr, readout, qb)
# Prepare the input
circuit.x(qb[0])  # As an example, start in |1>
encode_input(circuit, qb)
result = execute(circuit, Aer.get_backend('statevector_simulator')).result()
statevector = result.get_statevector()
print('Fidelity of encoded |1>_L', state_fidelity(logical_1, statevector))

# %% Transpiler

# %% Encoding circuit
circuit = QuantumCircuit(cr, readout, qb)
circuit += encode_input(qb)
# circuit += run_stabilizer(qb,an,cr)
print('Gates: ', circuit.count_ops())
print('Depth: ', circuit.depth())
circuit.draw(output='mpl')
# %% Full circuit
circuit = QuantumCircuit(cr, readout, an, qb)

# Prepare the input
# circuit.x( qb[0] ) # As an example, start in |1>
# circuit.snapshot_statevector('snapshot_label')

# Encode the state
circuit += encode_input(qb)
# circuit.snapshot_statevector('post_encoding')

# Add errors manually
# circuit.rx( np.pi, qb[3] )
# circuit.z( qb[3] )

# Measure stabilizers
circuit += run_stabilizer(qb, an, cr)

# Correct the error
circuit += recovery_scheme(qb, cr)

#run_stabilizer( circuit, qb, an, cr )
# Readout of the encoded state
# circuit.snapshot_statevector('pre_measure')
circuit.measure(qb, readout)
print('Gates: ', transpiled_circuit.count_ops())
print('depth: ', circuit.depth())
circuit.draw(output='mpl')
# %%
circuit = QuantumCircuit(cr, readout, an, qb)
circuit += encode_input(qb)
circuit.snapshot('post_encoding')
# %% Set up device properties
basis_gates = ['id', 'u1', 'u2', 'u3', 'iswap', 'cz']
couplinglist = [[0, 1], [0, 6], [1, 6], [2, 3],
                [2, 6], [3, 6], [4, 5], [4, 6], [5, 6]]
reverse_couplinglist = [[y, x] for [x, y] in couplinglist]
coupling_map = CouplingMap(couplinglist=couplinglist,
                           description='A hexagoal 7qb code with two ancillas')
coupling_map.draw()
layout = Layout(
    {qb[0]: 0,
     qb[1]: 1,
        qb[2]: 2,
        qb[3]: 3,
        qb[4]: 4,
        an[0]: 5,
        an[1]: 6})
optimization_level = 0
transpiled_circuit = transpile(circuit, coupling_map=coupling_map,
                               basis_gates=basis_gates, optimization_level=optimization_level)
print('Final depth: ', transpiled_circuit.depth())
print('Final num gates = ', transpiled_circuit.count_ops())
# plot_circuit_layout(transpiled_circuit,backend=coupling_map)

# %%

depth = 10000
routing_method = 'sabre'  # basic lookahead stochastic sabre
initial_layout = None
layout_method = 'sabre'  # trivial 'dense', 'noise_adaptive' sabre
translation_method = None  # 'unroller',  translator , synthesis
optimization_level = 3
for i in range(100):
    with warnings.catch_warnings():  # sabre causes deprication warning, this will ignore them
        warnings.simplefilter("ignore")
        transpiled_circuit_tmp = transpile(circuit, coupling_map=coupling_map, basis_gates=basis_gates, optimization_level=optimization_level,
                                           initial_layout=initial_layout, layout_method=layout_method, routing_method=routing_method, translation_method=translation_method)
    print('depth: ', transpiled_circuit_tmp.depth())
    if transpiled_circuit_tmp.depth() < depth:
        depth = transpiled_circuit_tmp.depth()
        transpiled_circuit = transpiled_circuit_tmp
print('Final depth: ', transpiled_circuit.depth())
print('Final gates = ', transpiled_circuit.count_ops())
transpiled_circuit.draw(output='mpl')
# %%

logging.basicConfig()
logging.getLogger('qiskit.transpiler').setLevel(logging.INFO)
logging.getLogger('qiskit.transpiler').setLevel(0)

# %% Testing EquivalenceLibrary

sel = SessionEquivalenceLibrary
sel.has_entry(iSwapGate())
# pil_img = sel.draw()
# pil_img.save('SessionEquivalenceLibrary.png')
# %% Adding custom equivalence
# SwapGate

q = QuantumRegister(2, 'q')
def_swap = QuantumCircuit(q)
for inst, qargs, cargs in [
        (iSwapGate(), [q[0], q[1]], []),
        (CZGate(), [q[0], q[1]], []),
        (SGate().inverse(), [q[1]], []),
        (SGate().inverse(), [q[0]], [])
]:
    def_swap.append(inst, qargs, cargs)
sel.add_equivalence(SwapGate(), def_swap)
sel.get_entry(SwapGate())[0].draw()

# # iSwapGate
# q = QuantumRegister(2, 'q')
# def_iswap = QuantumCircuit(q)
# for inst, qargs, cargs in [
#         (SwapGate(), [q[0], q[1]], []),
#         (CZGate(), [q[0], q[1]], []),
#         (SGate(), [q[1]], []),
#         (SGate(), [q[0]], [])
# ]:
#     def_iswap.append(inst, qargs, cargs)
# sel.add_equivalence(iSwapGate(), def_iswap)


# %%


def callback_func(**kwargs):
    pass_ = kwargs['pass_']
    dag = kwargs['dag']
    time = kwargs['time']
    property_set = kwargs['property_set']
    count = kwargs['count']

    if isinstance(pass_, TransformationPass):
        circuit_tmp = dag_to_circuit(dag)
        print('pass namne: ', pass_.name())
        print('\nqubits: ')
        for qubit in dag.qubits:
            print(qubit)
        print('\nwires: ')
        for wire in dag.wires:
            if isinstance(wire, Clbit):
                continue
            print(wire)

        display(circuit_tmp.draw(output='mpl'))


# %% Modiy and draw passmanager

# the drawing functionality doesn't work if coupling_map is specified
pm = level_0_pass_manager(PassManagerConfig())
pm.draw()

# pm.passes()
# pm.remove(1) #remove 'RemoveResetInZeroState'
# pm.draw()

# pm.replace() # Byt mot ett annat pass

# %%
[pass_ for pass_ in dir(passes) if pass_[0].isupper()]
# %%
pm = level_0_pass_manager(PassManagerConfig(
    coupling_map=coupling_map, basis_gates=basis_gates))
print(pm.passes()[-3], '\n')
pm.replace(-3, [UnrollCustomDefinitions(sel, basis_gates),
                BasisTranslator(sel, basis_gates)])
print(pm.passes()[-3])
# %%
pm = level_0_pass_manager(PassManagerConfig(coupling_map=coupling_map))
print('before transpilation ')
display(circuit.draw(output='mpl'))
transpiled_circuit = pm.run(circuit, callback=callback_func)
pm.passes()
# %%
transpiled_circuit = transpile(circuit, coupling_map=coupling_map)
results = execute(
    circuit,  # NOT RUNNING THE TRANSPILED CIRCUIT AT THE MOMENT
    Aer.get_backend('qasm_simulator'),
    noise_model=None,
    shots=1
).result()

state_vectors = results.data()['snapshots']['statevector']
sv_post_encoding_circ = state_vectors['post_encoding'][0]
results = execute(
    circuit,  # NOT RUNNING THE TRANSPILED CIRCUIT AT THE MOMENT
    Aer.get_backend('qasm_simulator'),
    noise_model=None,
    shots=1
).result()

state_vectors = results.data()['snapshots']['statevector']
sv_post_encoding_transp = state_vectors['post_encoding'][0]
state_fidelity(sv_post_encoding_circ, sv_post_encoding_transp)
# %%

# %% Things I moved out of simulator.py

# %%


def transpile_circuit(circuit, optimization_level=2, repeats=3):
    # % Transpiler
    routing_method = 'sabre'  # basic lookahead stochastic sabre
    # initial_layout = {qb[0]: 0,
    #                   qb[1]: 1,
    #                   qb[2]: 2,
    #                   qb[3]: 3,
    #                   qb[4]: 4,
    #                   an[0]: 5,
    #                   an[1]: 6}
    initial_layout = {an[0]: 0,
                      an[1]: 1,
                      qb[0]: 2,
                      qb[1]: 3,
                      qb[2]: 4,
                      qb[3]: 5,
                      qb[4]: 6}
    # initial_layout = None  # Overwriting the above layout
    layout_method = 'sabre'  # trivial 'dense', 'noise_adaptive' sabre
    translation_method = None  # 'unroller',  translator , synthesis
    transpiled_circuit = shortest_transpile_from_distribution(circuit, repeats=repeats, routing_method=routing_method, initial_layout=initial_layout,
                                                              layout_method=layout_method, translation_method=translation_method,optimization_level=optimization_level
                                                              ,coupling_map = WACQT_device_properties['coupling_map']
                                                            #   , **WAQCT_device_properties)
      ,**{'basis_gates': ['id', 'u1', 'u2', 'u3', 'cz','iswap']})

    print('Final depth = ', transpiled_circuit.depth())
    print('Final gates = ', transpiled_circuit.count_ops())
    # display(transpiled_circuit.draw(output='mpl'))

    return transpiled_circuit  # CHANGE TO TRANSPILED CIRCUIT

# %%

circuit = QuantumCircuit(cr, readout, an, qb)
circuit += encode_input(qb)
snapshot_type = 'density_matrix'  # 'statevector'
circuit.snapshot('post_encoding', snapshot_type)

# circuit = define_circuit(0)
transpiled_circuit = transpile_circuit(circuit)
transpiled_circuit._layout


def get_logical_states_v1(transpiled_circuit, snapshot_type='density_matrix'):
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
    state_vectors = results.data()['snapshots'][snapshot_type]
    sv_post_encoding = state_vectors['post_encoding'][0]
    if snapshot_type == 'density_matrix':
        sv_post_encoding = sv_post_encoding['value']
    return sv_post_encoding


def get_logical_states_v2(circuit, transpiled_circuit):
    ''' Get the permutation from transpiled_circuit._layout and attempt to redo it
     after encoding using the initialize method, by permuting the qubits in the second argument
    see: https://github.com/Qiskit/qiskit-tutorials/blob/master/tutorials/circuits/3_summary_of_quantum_operations.ipynb
    '''
    circuit_log0 = QuantumCircuit(*circuit.qregs)
    if transpiled_circuit._layout is not None:
        perm = [circuit.qubits[transpiled_circuit._layout.get_virtual_bits()[circuit.qubits[i]]]
                for i in range(7)]
        # perm = [transpiled_circuit._layout[i] for i in range(7)]
    else:
        print('NO TRANSPILED CIRCUIT')
        perm = circuit.qubits
    circuit_log0.initialize(
        logical[0], perm)
    results = execute(
        circuit_log0,
        Aer.get_backend('statevector_simulator'),
    ).result()
    return results.get_statevector(circuit_log0)


def print_state(state):
    tmp = np.where(np.round(state, decimals=2) != 0)[0]
    for x in (tmp*np.sign(state[np.round(state, decimals=2) != 0])).astype(int):
        print(x, ' ', end='')
    print('')

# Print which indices are non-zero, to quickly see if the state is correct


display(transpiled_circuit.draw(output='mpl'))
snap_circ = get_logical_states_v1(circuit, snapshot_type)
snap_transp = get_logical_states_v1(transpiled_circuit, snapshot_type)
reconstructed = get_logical_states_v2(circuit, transpiled_circuit)
# print('logical[0]')
# print_state(logical[0])

# print('snap circ')
# print_state(snap_circ)

# print('snap transp')
# print_state(snap_transp)

# print('reconstructed')
# print_state(reconstructed)
print('fidelity logical[0] snap circ', state_fidelity(logical[0], snap_circ))
print('fidelity snap transp snap circ', state_fidelity(snap_transp, snap_circ))
print('fidelity snap transp reconstructed',
      state_fidelity(snap_transp, reconstructed))
print('fidelity logical[0] reconstructed',
      state_fidelity(logical[0], reconstructed))

# %% Compare the unitaries
n_cycles = 0

circuit = QuantumCircuit(cr, readout, an, qb)
circuit += encode_input(qb)
# circuit.snapshot_statevector('post_encoding')

# circuit = define_circuit(0)
transpiled_circuit = transpile_circuit(circuit)
transpiled_circuit._layout


def compare_unitaries(circuit1, circuit2):
    simulator = Aer.get_backend('unitary_simulator')
    unitary_circuit1 = execute(
        circuit1, simulator).result().get_unitary(circuit1, decimals=3)
    unitary_circuit2 = execute(
        circuit2, simulator).result().get_unitary(circuit2, decimals=3)

    print('diff ', np.max(unitary_circuit1-unitary_circuit2))


compare_unitaries(circuit, transpiled_circuit)

# %%

def compare_unitaries_v2(qc1,qc2):
    backend_sim = Aer.get_backend('unitary_simulator')
    job_sim = execute([qc1, qc2], backend_sim)
    result_sim = job_sim.result()
    unitary1 = result_sim.get_unitary(qc1)
    unitary2 = result_sim.get_unitary(qc2)

    return np.allclose(unitary1, unitary2)

from qiskit.circuit.library import Permutation