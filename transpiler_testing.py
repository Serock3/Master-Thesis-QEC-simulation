# %% Import modules
import numpy as np
#from qiskit import(
#    QuantumCircuit,
#    execute,
#    Aer,
#    BasicAer,
#    QuantumRegister,
#    ClassicalRegister
#    )
from qiskit import *
from qiskit.visualization import plot_histogram


# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import QuantumError, ReadoutError
from qiskit.providers.aer.noise import pauli_error
from qiskit.providers.aer.noise import depolarizing_error

from qiskit.quantum_info import state_fidelity
from qiskit.providers.aer.extensions.snapshot_statevector import *
# %% Defining useful functions

# Not that this does not consider our setup
def encode_input( qbReg ):
    '''Encode the input into logical 0 and 1
    This assumes that the 0:th qubit is the
    original state |psi> = a|0> + b|1>'''
    encoding_circuit = QuantumCircuit( qbReg )

    encoding_circuit.h( qbReg[3] )
    encoding_circuit.cz( qbReg[3], qbReg[1] )
    encoding_circuit.cz( qbReg[3], qbReg[2] )
    encoding_circuit.cx( qbReg[3], qbReg[0] )   

    encoding_circuit.h( qbReg[2] )
    encoding_circuit.cx( qbReg[2], qbReg[0] )
    encoding_circuit.cz( qbReg[2], qbReg[3] )
    encoding_circuit.cz( qbReg[2], qbReg[4] )
    
    encoding_circuit.h( qbReg[1] )
    encoding_circuit.cz( qbReg[1], qbReg[0] )
    encoding_circuit.cx( qbReg[1], qbReg[3] )
    encoding_circuit.cz( qbReg[1], qbReg[4] )
    
    encoding_circuit.h( qbReg[4] )
    encoding_circuit.cz( qbReg[4], qbReg[2] )
    encoding_circuit.cz( qbReg[4], qbReg[3] )
    encoding_circuit.cx( qbReg[4], qbReg[1] )
    
    return encoding_circuit

def measure_stabilizer( stab_circuit, qbReg, anReg, clReg, i ):
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
    
    #stab_circuit = QuantumCircuit( qbReg, anReg, clReg )
    #stab_circuit = circuit

    # Generate indexes
    index = np.mod( i + np.array([0, 1, 2, 3]), 5 ) 
    
    # Measure stabilizers
    stab_circuit.h( qbReg[ index[0] ] )
    stab_circuit.h( anReg[1] )
    stab_circuit.cz( anReg[1], qbReg[ index[0] ] )
    stab_circuit.h( qbReg[ index[0] ] ) 
    
    stab_circuit.cz( anReg[1], qbReg[ index[1] ] )

    stab_circuit.cz( anReg[1], qbReg[ index[2] ] )
        
    stab_circuit.h( qbReg[ index[3] ] )
    stab_circuit.cz( anReg[1], qbReg[ index[3] ] )
    stab_circuit.h( anReg[1] )
    stab_circuit.h( qbReg[ index[3] ] ) 
        
    stab_circuit.measure( anReg[1], clReg[i] )
    stab_circuit.reset( anReg[1] )
    return #stab_circuit

def run_stabilizer( circuit, qbReg, anReg, clReg ):
#    stab_circuit = QuantumCircuit( qbReg, anReg, clReg )
    #stab_circuit = circuit
    measure_stabilizer( circuit, qbReg, anReg, clReg, 0 )
    measure_stabilizer( circuit, qbReg, anReg, clReg, 1 )
    measure_stabilizer( circuit, qbReg, anReg, clReg, 2 )
    measure_stabilizer( circuit, qbReg, anReg, clReg, 3 )
    return #stab_circuit

# Correct possible errors
def recovery_scheme( qbReg, clReg ):

    recovery_circuit = QuantumCircuit( qbReg, clReg )
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
    logical_0[0b00000]=1/4
    logical_0[0b10010]=1/4
    logical_0[0b01001]=1/4
    logical_0[0b10100]=1/4
    logical_0[0b01010]=1/4
    logical_0[0b11011]=-1/4
    logical_0[0b00110]=-1/4
    logical_0[0b11000]=-1/4
    logical_0[0b11101]=-1/4
    logical_0[0b00011]=-1/4
    logical_0[0b11110]=-1/4
    logical_0[0b01111]=-1/4
    logical_0[0b10001]=-1/4
    logical_0[0b01100]=-1/4
    logical_0[0b10111]=-1/4
    logical_0[0b00101]=1/4

    logical_1 = np.zeros(2**5)
    logical_1[0b11111]=1/4
    logical_1[0b01101]=1/4
    logical_1[0b10110]=1/4
    logical_1[0b01011]=1/4
    logical_1[0b10101]=1/4
    logical_1[0b00100]=-1/4
    logical_1[0b11001]=-1/4
    logical_1[0b00111]=-1/4
    logical_1[0b00010]=-1/4
    logical_1[0b11100]=-1/4
    logical_1[0b00001]=-1/4
    logical_1[0b10000]=-1/4
    logical_1[0b01110]=-1/4
    logical_1[0b10011]=-1/4
    logical_1[0b01000]=-1/4
    logical_1[0b11010]=1/4

    # Add two ancillas in |0>
    an0 = np.zeros(2**2)
    an0[0] = 1.0

    #logical_1 = np.kron(logical_1, an0)
    #logical_0 = np.kron(logical_0, an0)

    #logical_0 = np.kron(an0, logical_0)
    #logical_1 = np.kron(an0, logical_1)    
    return [logical_0, logical_1]

def noise_model():
    # Example error probabilities
    p_reset = 0.0000000
    p_meas = 0.00
    p_gate1 = 0.00

    # QuantumError objects
    error_reset = pauli_error([('X', p_reset), ('I', 1 - p_reset)])
    error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
    error_gate1 = pauli_error([('X',p_gate1), ('I', 1 - p_gate1)])
    error_gate2 = error_gate1.tensor(error_gate1)

    # Add errors to noise model
    noise_bit_flip = NoiseModel()
    noise_bit_flip.add_all_qubit_quantum_error(error_reset, "reset")
    noise_bit_flip.add_all_qubit_quantum_error(error_meas, "measure")
    noise_bit_flip.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3", "x"])
#    noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ["cx"])
    return noise_bit_flip

# %% Define our registers and circuit
qb = QuantumRegister(5, 'code_qubit')     # The 5 qubits to encode the state in
an = QuantumRegister(2, 'ancilla_qubit')  # The two ancilla qubits (one of them is unused)
cr = ClassicalRegister(4, 'syndrome_bit') # Classical register for registering the syndromes
readout = ClassicalRegister(5, 'readout') # Readout of the final state at the end for statistics

# %% Running the quantum circuit


circuit = QuantumCircuit( cr, readout, an, qb )

# Prepare the input
circuit.x( qb[0] ) # As an example, start in |1>
#circuit.snapshot_statevector('snapshot_label')

# Encode the state
circuit += encode_input( qb ) 
circuit.snapshot_statevector('post_encoding')

# Add errors manually
#circuit.rx( np.pi, qb[3] )
#circuit.z( qb[3] )

# Measure stabilizers
circuit += run_stabilizer( qb, an, cr )

# Correct the error
circuit += recovery_scheme( qb, cr )
circuit.reset( an[0] )
#run_stabilizer( circuit, qb, an, cr )
# Readout of the encoded state
circuit.snapshot_statevector('pre_measure')
circuit.measure( qb, readout )
circuit.snapshot_statevector('post_measure')

noise = noise_model()


# %% Testing all iterations
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager,CouplingMap,Layout
from qiskit.visualization import plot_circuit_layout
qb = QuantumRegister(7, 'code_qubit')     # The 5 qubits to encode the state in
readout = ClassicalRegister(7, 'readout') # Readout of the final state at the end for statistics

circuit = QuantumCircuit( cr, readout, qb )
circuit += encode_input( qb )
circuit.snapshot_statevector('post_encoding')

# run_stabilizer(circuit,qb,an,cr)



basis_gates = ['id', 'u1', 'u2', 'u3', 'iswap','cz', 'h']
couplinglist=[[0, 1],[0,6],[1,6],[2,3],[2,6],[3,6],[4,5],[4,6],[5,6]]
reverse_couplinglist = [[y,x] for [x,y] in couplinglist]
coupling_map = CouplingMap(couplinglist=couplinglist,description='A hexagoal 7qb code with two ancillas')
#coupling_map.draw()
#circuit.draw(output='mpl')
#%
# EXPERIMANTAL SHITSHOW:
from sympy.utilities.iterables import multiset_permutations
import numpy as np
a = np.array([0, 1, 2, 3, 4, 5, 6])
b = np.array([])
for p in multiset_permutations(a):
   b = np.append(b, p)
b = np.reshape(b, [5040,7])
logical = logical_states()
i=0
for test in b:

    layout = Layout(
    {qb[0]: int(test[0]),
     qb[1]: int(test[1]),
     qb[2]: int(test[2]),
     qb[3]: int(test[3]),
     qb[4]: int(test[4]),
     qb[5]: int(test[5]),
     qb[6]: int(test[6])})
    #print(layout)
    optimization_level=2
    transpiled_circuit = transpile(circuit,coupling_map=coupling_map,basis_gates=basis_gates,optimization_level=optimization_level,initial_layout=layout)
    #print('depth: ', transpiled_circuit.depth())
    #transpiled_circuit.reset( [5,6] )
    #transpiled_circuit.snapshot_statevector('post_encoding')

    # plot_circuit_layout(transpiled_circuit,backend=coupling_map)
    #transpiled_circuit.draw(output='mpl')

    # %
    results = execute(circuit, Aer.get_backend('qasm_simulator'), noise_model=noise, shots=1).result()
    #counts = results.get_counts()

    # Get the state vectors
    state_vectors = results.data()['snapshots']['statevector']
    sv_post_encoding = state_vectors['post_encoding'][0]
    #sv_pre_measure = state_vectors['pre_measure'][0]
    #sv_post_measure = state_vectors['post_measure'][0]

    #print(sv_post_encoding.shape)
    #print('Fidelity of encoded |1>_L',state_fidelity(logical[1],sv_post_encoding))
    #print('Fidelity of encoded |0>_L',state_fidelity(logical[0],sv_post_encoding))
    #circuit.draw(output='mpl')
    if np.linalg.norm(state_fidelity(logical[0], sv_post_encoding)-1) < 0.2:
        print('Hello there')
        print(test)
        print(state_fidelity(logical[0], sv_post_encoding))
    if np.linalg.norm(state_fidelity(logical[1], sv_post_encoding)-1) < 0.2:
        print('Hello there')
        print(test)
        print(state_fidelity(logical[1], sv_post_encoding))
    if np.mod(i, 100) == 0:
        print(i)
    i+=1
#print(counts)
#plot_histogram(counts)A
#circuit.draw(output='mpl')
#print(sv_post_encoding.shape)
#print(logical[1])

# %% Testing readouts
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager,CouplingMap,Layout
from qiskit.visualization import plot_circuit_layout
qb = QuantumRegister(7, 'code_qubit')     # The 5 qubits to encode the state in
readout = ClassicalRegister(7, 'readout') # Readout of the final state at the end for statistics

circuit = QuantumCircuit( cr, readout, qb )
circuit += encode_input( qb )
circuit.snapshot_statevector('post_encoding')
circuit.measure(qb, readout)
# run_stabilizer(circuit,qb,an,cr)



basis_gates = ['id', 'u1', 'u2', 'u3', 'iswap','cz', 'h']
couplinglist=[[0, 1],[0,6],[1,6],[2,3],[2,6],[3,6],[4,5],[4,6],[5,6]]
reverse_couplinglist = [[y,x] for [x,y] in couplinglist]
coupling_map = CouplingMap(couplinglist=couplinglist,description='A hexagoal 7qb code with two ancillas')
#coupling_map.draw()
#circuit.draw(output='mpl')
#%
# EXPERIMANTAL SHITSHOW:
test = [0,1,2,3,4,5,6]

layout = Layout(
{qb[0]: int(test[0]),
 qb[1]: int(test[1]),
 qb[2]: int(test[2]),
 qb[3]: int(test[3]),
 qb[4]: int(test[4]),
 qb[5]: int(test[5]),
 qb[6]: int(test[6])})

optimization_level=2
transpiled_circuit = transpile(circuit,coupling_map=coupling_map,basis_gates=basis_gates,optimization_level=optimization_level,initial_layout=layout)
results = execute(transpiled_circuit, Aer.get_backend('qasm_simulator'), noise_model=noise, shots=1).result()
counts = results.get_counts()

# Get the state vectors
state_vectors = results.data()['snapshots']['statevector']
sv_post_encoding = state_vectors['post_encoding'][0]
#sv_pre_measure = state_vectors['pre_measure'][0]
#sv_post_measure = state_vectors['post_measure'][0]

transpiled_circuit2 = transpile(circuit,coupling_map=coupling_map,basis_gates=basis_gates,optimization_level=optimization_level,initial_layout=layout)
results2 = execute(transpiled_circuit2, Aer.get_backend('qasm_simulator'), noise_model=noise, shots=1).result()
counts = results2.get_counts()

# Get the state vectors
state_vectors2 = results2.data()['snapshots']['statevector']
sv_post_encoding2 = state_vectors2['post_encoding'][0]

#print('Fidelity of encoded |1>_L',state_fidelity(logical[1],sv_post_encoding))
#print('Fidelity of encoded |0>_L',state_fidelity(logical[0],sv_post_encoding))
#circuit.draw(output='mpl')
if np.linalg.norm(state_fidelity(logical[0], sv_post_encoding)-1) < 0.2:
    print('Hello there')
    print(test)
    print(state_fidelity(logical[0], sv_post_encoding))
if np.linalg.norm(state_fidelity(logical[1], sv_post_encoding)-1) < 0.2:
    print('Hello there')
    print(test)
    print(state_fidelity(logical[1], sv_post_encoding))
#print(sv_post_encoding)
#print(counts)
#plot_histogram(counts)
#transpiled_circuit.draw(output='mpl')
#print(sv_post_encoding.shape)
#print(logical[1])
print(np.linalg.norm(sv_post_encoding - sv_post_encoding2) )
# %%

from qiskit.compiler import transpile
from qiskit.transpiler import PassManager,CouplingMap,Layout
from qiskit.visualization import plot_circuit_layout
from qiskit.quantum_info import partial_trace
from qiskit.quantum_info import Statevector
# Define our registers and circuit
qb = QuantumRegister(5, 'code_qubit')     # The 5 qubits to encode the state in
an = QuantumRegister(2, 'ancilla_qubit')  # The two ancilla qubits (one of them is unused)
cr = ClassicalRegister(4, 'syndrome_bit') # Classical register for registering the syndromes
readout = ClassicalRegister(5, 'readout') # Readout of the final state at the end for statistics

circuit = QuantumCircuit( cr, readout, qb, an )
circuit += encode_input( qb )
circuit.snapshot_statevector('post_encoding')
run_stabilizer( circuit, qb, an, cr )
#circuit.x([5])
#circuit.z([6])
#circuit += run_stabilizer( qb, an, cr )
circuit.measure( qb, readout )

optimization_level=2
basis_gates = ['id', 'u1', 'u2', 'u3', 'swap','cz', 'h', 'cx']
couplinglist=[[0, 1],[0,6],[1,6],[2,3],[2,6],[3,6],[4,5],[4,6],[5,6]]
reverse_couplinglist = [[y,x] for [x,y] in couplinglist]
coupling_map = CouplingMap(couplinglist=couplinglist,description='A hexagoal 7qb code with two ancillas')

layout = Layout(
{qb[0]: 0,
 qb[1]: 1,
 qb[2]: 2,
 qb[3]: 3,
 qb[4]: 4,
 an[0]: 5,
 an[1]: 6})

transpiled_circuit = transpile(circuit,coupling_map=coupling_map,basis_gates=basis_gates,optimization_level=optimization_level,initial_layout=layout)
transpiled_circuit.id([5, 6])
#run_stabilizer( transpiled_circuit, [0,1,2,3,4], [5,6], cr )
#transpiled_circuit.measure([0,1,2,3,4], readout)
results = execute(transpiled_circuit, Aer.get_backend('qasm_simulator'), shots=1000).result()
counts = results.get_counts()

# Get the state vectors
state_vectors = results.data()['snapshots']['statevector']
sv_post_encoding = state_vectors['post_encoding'][0]

sv = Statevector(sv_post_encoding)
test= partial_trace(sv, [5,6])
#test2 = test.to_statevector()
logical = logical_states()
#print(test)
#print(logical[0].shape)
print('Fidelity of encoded |0>_L',state_fidelity(logical[0],test))
print('Fidelity of encoded |1>_L',state_fidelity(logical[1],test))
print(counts)
transpiled_circuit.draw(output='mpl')
#circuit.draw(output='mpl')
#plot_histogram(counts)
cr.qasm()