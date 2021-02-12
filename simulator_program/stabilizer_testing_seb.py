#%%
# from stabilizers import measure_stabilizer
from qiskit import execute, Aer
from qiskit.circuit.quantumregister import QuantumRegister, AncillaRegister
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
import numpy as np
from qiskit.compiler.transpile import transpile

# %%
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
qbReg = QuantumRegister(5, 'code_qubit')
# The two ancilla qubits (one of them is unused)
anReg = QuantumRegister(1, 'ancilla_qubit')
flagQbReg = QuantumRegister(1, 'flag_qubit')
clReg = ClassicalRegister(4, 'syndrome_bit')
# Flags
flagBtReg = ClassicalRegister(1, 'flag_bit')
# Readout of the final state at the end for statistics
#%%
circ = QuantumCircuit(clReg,flagBtReg, qbReg, anReg, flagQbReg) #, flag,
reset = True
# circ += encode_input(qbReg)
# circ.snapshot_statevector('post_encoding')
circ.initialize(logical_0, qbReg)
circ.barrier()

# Noise test
# circ.z( qbReg[0] ) 

# X, a
circ.h( qbReg[0] ) # X
circ.h( anReg[0] )
circ.cz( anReg[0], qbReg[0] )
circ.h( qbReg[0] )

# Flag
circ.h( flagQbReg[0] ) # Initialize in +
circ.cz( flagQbReg[0], anReg[0] ) # Flag CX        

# Z, b
circ.cz(anReg[0], qbReg[1])

# Two qb noise test
circ.x(anReg[0])

# Z, c
circ.cz( anReg[0], qbReg[2] )

# Flag
circ.cz( flagQbReg[0], anReg[0] )   

# X, d
circ.h(qbReg[3])
circ.cz(anReg[0], qbReg[3])
circ.h(anReg[0])
circ.h(qbReg[3])

# Measure
circ.measure(anReg[0], clReg[0])
circ.h(flagQbReg[0])
circ.measure(flagQbReg[0], flagBtReg[0])
if reset:
    circ.reset(anReg[0])
    circ.reset(flagQbReg[0])
#circ += recovery_scheme(...)

#
results = execute(
    circ,
    Aer.get_backend('qasm_simulator'),
    noise_model=None,
    shots=10
).result()
print(results.get_counts())

#%%
circ.draw()
# %%
print('circ depth = ',circ.depth())
transpiled_circ = transpile(circ, optimization_level=2)
print('transpiled_circ depth = ',transpiled_circ.depth())
transpiled_circ.draw(output='mpl')
# %%
