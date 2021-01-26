#%%
from matplotlib import get_backend
import numpy as np
from qiskit import *
# Useful additional packages
import matplotlib.pyplot as plt
%matplotlib inline
from math import pi

#%%

q = QuantumRegister(1)
c = ClassicalRegister(1)
qc = QuantumCircuit(q,c)
qc.x(q)
qc.measure(q[0], c)
qc.draw()
# %%
backend = BasicAer.get_backend('qasm_simulator')
qc = QuantumCircuit(q, c)
qc.x(q[0]).c_if(c, 0)
qc.measure(q,c)
qc.draw()
job = execute(qc, backend, shots=1024)
job.result().get_counts(qc)
#%%
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute
from qiskit.tools.visualization import plot_histogram, plot_state_city, plot_state_hinton
# List Aer backends
Aer.backends()
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator, UnitarySimulator
#%%
# Construct quantum circuit
circ = QuantumCircuit(2, 2)
circ.h(0)
circ.cx(0, 1)
circ.measure([0,1], [0,1])

# Select the QasmSimulator from the Aer provider
simulator = Aer.get_backend('qasm_simulator')

# Execute and get counts
result = execute(circ, simulator).result()
counts = result.get_counts(circ)
plot_histogram(counts, title='Bell-State counts')
#%%
# Construct quantum circuit with measure
circ = QuantumCircuit(2, 2)
circ.h(0)
circ.cx(0, 1)
circ.measure([0,1], [0,1])

# Select the StatevectorSimulator from the Aer provider
simulator = Aer.get_backend('statevector_simulator')

# Execute and get counts
result = execute(circ, simulator).result()
statevector = result.get_statevector(circ)
plot_state_city(statevector, title='Bell state post-measurement')
# %% Arbitrary initialization
import math

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

qc.initialize(desired_vector, [q[0],q[1],q[2]])
qc.draw()

# %%
backend = BasicAer.get_backend('statevector_simulator')
job = execute(qc, backend)
qc_state = job.result().get_statevector(qc)
qc_state
# %%
from qiskit.quantum_info import state_fidelity
state_fidelity(desired_vector,qc_state)
# %%

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error

def get_noise(p_meas,p_gate):

    error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
    error_gate1 = depolarizing_error(p_gate, 1)
    error_gate2 = error_gate1.tensor(error_gate1)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure") # measurement error is applied to measurements
    noise_model.add_all_qubit_quantum_error(error_gate1, ["x"]) # single qubit gate error is applied to x gates
    noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"]) # two qubit gate error is applied to cx gates
        
    return noise_model

#%% Kolla på detta
from qiskit.ignis.verification.topological_codes import fitters
from qiskit.quantum_info.operators.symplectic import stabilizer_table
from qiskit.ignis.verification.topological_codes import circuits


#%% Kontrollera kod från stabilizer_code
#%%
# Note that some gates might be redundant, haven't checked
def encode_input( circuit, qbReg ):
    '''Encode the input into logical 0 and 1
    This assumes that the 0:th qubit is the
    original state |psi> = a|0> + b|1>'''
    
    circuit.h( qbReg[3] )
    circuit.cz( qbReg[3], qbReg[1] )
    circuit.cz( qbReg[3], qbReg[2] )
    circuit.cx( qbReg[3], qbReg[0] )   

    circuit.h( qbReg[2] )
    circuit.cx( qbReg[2], qbReg[0] )
    circuit.cz( qbReg[2], qbReg[3] )
    circuit.cz( qbReg[2], qbReg[4] )
    
    circuit.h( qbReg[1] )
    circuit.cz( qbReg[1], qbReg[0] )
    circuit.cx( qbReg[1], qbReg[3] )
    circuit.cz( qbReg[1], qbReg[4] )
    
    circuit.h( qbReg[4] )
    circuit.cz( qbReg[4], qbReg[2] )
    circuit.cz( qbReg[4], qbReg[3] )
    circuit.cx( qbReg[4], qbReg[1] )
    
    circuit.barrier( qbReg )
    return

# Define our stabilizers
def measure_stabilizer( circuit, qbReg, anReg, clReg, i ):
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
    
    # Generate indexes
    index = np.mod( i + np.array([0, 1, 2, 3]), 5 ) 
    
    # Measure stabilizers
    circuit.h( qbReg[ index[0] ] )
    circuit.h( anReg[0] )
    circuit.cz( anReg[0], qbReg[ index[0] ] )
    circuit.h( qbReg[ index[0] ] ) 
    
    circuit.cz( anReg[0], qbReg[ index[1] ] )

    circuit.cz( anReg[0], qbReg[ index[2] ] )
        
    circuit.h( qbReg[ index[3] ] )
    circuit.cz( anReg[0], qbReg[ index[3] ] )
    circuit.h( anReg[0] )
    circuit.h( qbReg[ index[3] ] ) 
        
    circuit.measure( anReg[0], clReg[i] )
    circuit.reset( anReg[0] )
    return

def run_stabilizer( circuit, qbReg, anReg, clReg ):
    measure_stabilizer( circuit, qbReg, anReg, clReg, 0 )
    measure_stabilizer( circuit, qbReg, anReg, clReg, 1 )
    measure_stabilizer( circuit, qbReg, anReg, clReg, 2 )
    measure_stabilizer( circuit, qbReg, anReg, clReg, 3 )
    return

# Correct possible errors
def recovery_scheme( circuit, qbReg, clReg ):

    circuit.x(qbReg[1]).c_if(clReg, 1)
    circuit.z(qbReg[4]).c_if(clReg, 2)
    circuit.x(qbReg[2]).c_if(clReg, 3)
    circuit.z(qbReg[2]).c_if(clReg, 4)
    circuit.z(qbReg[0]).c_if(clReg, 5)
    circuit.x(qbReg[3]).c_if(clReg, 6)
    circuit.x(qbReg[2]).c_if(clReg, 7)
    circuit.z(qbReg[2]).c_if(clReg, 7)
    circuit.x(qbReg[0]).c_if(clReg, 8)
    circuit.z(qbReg[3]).c_if(clReg, 9)
    circuit.z(qbReg[1]).c_if(clReg, 10)
    circuit.x(qbReg[1]).c_if(clReg, 11)
    circuit.z(qbReg[1]).c_if(clReg, 11)
    circuit.x(qbReg[4]).c_if(clReg, 12)
    circuit.x(qbReg[0]).c_if(clReg, 13)
    circuit.z(qbReg[0]).c_if(clReg, 13)
    circuit.x(qbReg[4]).c_if(clReg, 14)
    circuit.z(qbReg[4]).c_if(clReg, 14)
    circuit.x(qbReg[3]).c_if(clReg, 15)
    circuit.z(qbReg[3]).c_if(clReg, 15)

    return
    

# %% Define our registers and circuit
qb = QuantumRegister(5, 'code_qubit')     # The 5 qubits to encode the state in
an = QuantumRegister(2, 'ancilla_qubit')  # The two ancilla qubits (one of them is unused)
cr = ClassicalRegister(4, 'syndrome_bit') # Classical register for registering the syndromes
readout = ClassicalRegister(5, 'readout') # Readout of the final state at the end for statistics
# %% Test the encoding

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

# Test
print('<0|0>_L =',np.linalg.norm(logical_0))
print('<1|1>_L =',np.linalg.norm(logical_1))
print('<0|1>_L =',np.dot(logical_0,logical_1))

from qiskit.quantum_info import state_fidelity

# Encode and check |0>_L
circuit = QuantumCircuit( cr, readout, qb )
encode_input( circuit, qb )
result = execute(circuit, Aer.get_backend('statevector_simulator')).result()
statevector = result.get_statevector()
print('Fidelity of encoded |0>_L',state_fidelity(logical_0,statevector))

# Encode and check |1>_L
circuit = QuantumCircuit( cr, readout, qb )
# Prepare the input
circuit.x( qb[0] ) # As an example, start in |1>
encode_input( circuit, qb ) 
result = execute(circuit, Aer.get_backend('statevector_simulator')).result()
statevector = result.get_statevector()
print('Fidelity of encoded |1>_L',state_fidelity(logical_1,statevector))
#%%
# Add errors manually
# (Nothing tried here yet)

circuit = QuantumCircuit( cr, readout, qb,an)
encode_input( circuit, qb )
# Measure stabilizers
measure_stabilizer( circuit, qb, an, cr, 0 )
measure_stabilizer( circuit, qb, an, cr, 1 )
measure_stabilizer( circuit, qb, an, cr, 2 )
measure_stabilizer( circuit, qb, an, cr, 3 )

# Readout of the encoded state
circuit.measure( qb, readout )

counts = execute(circuit, Aer.get_backend('qasm_simulator')).result().get_counts()
print(counts)

circuit.draw(output='mpl') # If it does not work, simply remove mpl: circuit.draw()
plot_histogram(counts)
# All stabilizers should give 0 (meaning '0000' for all states) unless error has been added.

# %% Transpiler
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager,CouplingMap,Layout
from qiskit.visualization import plot_circuit_layout

circuit = QuantumCircuit( cr, readout, qb,an )
encode_input( circuit, qb )
# run_stabilizer(circuit,qb,an,cr)

circuit.draw(output='mpl')

basis_gates = ['id', 'u1', 'u2', 'u3', 'iswap','cz']
couplinglist=[[0, 1],[0,6],[1,6],[2,3],[2,6],[3,6],[4,5],[4,6],[5,6]]
reverse_couplinglist = [[y,x] for [x,y] in couplinglist]
coupling_map = CouplingMap(couplinglist=couplinglist,description='A hexagoal 7qb code with two ancillas')
coupling_map.draw()
#%%
layout = Layout(
{qb[0]: 0,
 qb[1]: 1,
 qb[2]: 2,
 qb[3]: 3,
 qb[4]: 4,
 an[0]: 5,
 an[1]: 6})
optimization_level=2
transpiled_circuit = transpile(circuit,coupling_map=coupling_map,basis_gates=basis_gates,optimization_level=optimization_level,initial_layout=layout)
print('depth: ', transpiled_circuit.depth())
# plot_circuit_layout(transpiled_circuit,backend=coupling_map)

# %%
depth = 100
for i in range(100):
    transpiled_circuit_tmp = transpile(circuit,coupling_map=coupling_map,basis_gates=basis_gates,optimization_level=optimization_level,initial_layout=layout)
    print('depth: ', transpiled_circuit_tmp.depth())
    if transpiled_circuit_tmp.depth()<depth:
        depth = transpiled_circuit_tmp.depth()
        transpiled_circuit = transpiled_circuit_tmp
print(depth)

transpiled_circuit.draw(output='mpl')
# %%
# %% Tesing EquivalenceLibrary
from qiskit.circuit import EquivalenceLibrary
from  qiskit.circuit.library import iswap

eq = EquivalenceLibrary()
eq.has_entry(iswap)

#%%
circ = QuantumCircuit(2)
circ.h(0)
circ.cx(0, 1)

# Select the UnitarySimulator from the Aer provider
simulator = Aer.get_backend('unitary_simulator')

# Execute and get counts
result = execute(circ, simulator).result()
unitary = result.get_unitary(circ)
print("Circuit unitary:\n", unitary)