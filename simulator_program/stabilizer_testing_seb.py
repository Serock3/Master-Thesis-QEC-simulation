#%%
from stabilizers import measure_stabilizer
from qiskit.circuit.quantumregister import QuantumRegister
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
anReg = QuantumRegister(2, 'ancilla_qubit')
clReg = ClassicalRegister(5, 'syndrome_bit')
# Flags
flag = ClassicalRegister(1, 'flag_bit')
# Readout of the final state at the end for statistics
readout = ClassicalRegister(5, 'readout')
#%%
circ = QuantumCircuit(clReg, readout, qbReg, anReg) #, flag,

# circ += encode_input(qbReg)
# circ.snapshot_statevector('post_encoding')
circ.initialize(logical_0, qbReg)
circ.barrier()
# X
circ.h( qbReg[0] ) # X
circ.h( anReg[1] )
circ.cz( anReg[1], qbReg[0] )
circ.h( qbReg[0] )

# Flag
circ.h( anReg[0] ) # Initialize in +
circ.cz( anReg[0], anReg[1] ) # Flag CX        

# Z
circ.cz(anReg[1], qbReg[1])

# Z
circ.cz( anReg[1], qbReg[2] )

# Flag
circ.cz( anReg[0], anReg[1] )   

# X
circ.h(qbReg[3])
circ.cz(anReg[1], qbReg[3])
circ.h(anReg[1])
circ.h(qbReg[3])

# Measure
circ.measure(anReg[1], clReg[0])
circ.h(anReg[0])
circ.measure(anReg[0], clReg[4])
reset = True
if reset:
    circ.reset(anReg[1])
    circ.reset(anReg[0])

# Run the stabilizers
circ += measure_stabilizer(qbReg, anReg, clReg, 0, reset=True)
circ += measure_stabilizer(qbReg, anReg, clReg, 1, reset=True)
circ += measure_stabilizer(qbReg, anReg, clReg, 2, reset=True)
circ += measure_stabilizer(qbReg, anReg, clReg, 3, reset=True)

# Temporary recovery with flag
circ.y(qbReg[2]).c_if(clReg, 1+16)
circ.x(qbReg[3]).c_if(clReg, 1+16)
circ.z(qbReg[2]).c_if(clReg, 2+16)
circ.x(qbReg[3]).c_if(clReg, 2+16)
circ.x(qbReg[1]).c_if(clReg, 3+16)
circ.z(qbReg[2]).c_if(clReg, 3+16)
circ.x(qbReg[3]).c_if(clReg, 3+16)
circ.x(qbReg[2]).c_if(clReg, 5+16)
circ.x(qbReg[3]).c_if(clReg, 5+16)
circ.x(qbReg[3]).c_if(clReg, 6+16)
circ.x(qbReg[0]).c_if(clReg, 8+16) # This seems equivalent with IZZXI
circ.y(qbReg[1]).c_if(clReg, 9+16)
circ.z(qbReg[2]).c_if(clReg, 9+16)
circ.x(qbReg[3]).c_if(clReg, 9+16)
circ.draw(output='mpl')
#circ += recovery_scheme(...)
# %%
print('circ depth = ',circ.depth())
transpiled_circ = transpile(circ, optimization_level=2)
print('transpiled_circ depth = ',transpiled_circ.depth())
transpiled_circ.draw(output='mpl')
# %%
