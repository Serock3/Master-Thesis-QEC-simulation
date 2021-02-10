# Import modules
import numpy as np
from qiskit import (QuantumCircuit,
                    QuantumRegister, 
                    ClassicalRegister, 
                    AncillaRegister, 
                    execute, 
                    Aer
                    )

# %% OLD FUNCTIONS
# Run a single stabilizer
def measure_stabilizer(qbReg, anReg, clReg, i, reset=True):
    '''OLD FUNCTION replaced by unflagged_stabilizer_XZZXI,
    unflagged_stabilizer_IXZZX, unflagged_stabilizer_XIXZZ,
    unflagged_stabilizer_ZXIXZ for the respective stabilizers.
    
    Function for adding stabilizer measurements to a circuit.
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
    stab_circuit.h(anReg[1])
    stab_circuit.h(qbReg[index[0]])
    stab_circuit.cz(anReg[1], qbReg[index[0]])
    stab_circuit.h(qbReg[index[0]])

    stab_circuit.cz(anReg[1], qbReg[index[1]])

    stab_circuit.cz(anReg[1], qbReg[index[2]])

    stab_circuit.h(qbReg[index[3]])
    stab_circuit.cz(anReg[1], qbReg[index[3]])
    stab_circuit.h(qbReg[index[3]])
    stab_circuit.h(anReg[1])

    stab_circuit.measure(anReg[1], clReg[i])
    if reset:
        stab_circuit.reset(anReg[1])

    return stab_circuit

# Old stabilizer cycle without flags
def run_stabilizer(qbReg, anReg, clReg, reset=True):
    '''OLD FUNCTION, replaced by unflagged_stabilizer_all'''
    stab_circuit = QuantumCircuit(qbReg, anReg, clReg)
    stab_circuit += measure_stabilizer(qbReg, anReg, clReg, 0, reset)
    stab_circuit += measure_stabilizer(qbReg, anReg, clReg, 1, reset)
    stab_circuit += measure_stabilizer(qbReg, anReg, clReg, 2, reset)
    stab_circuit += measure_stabilizer(qbReg, anReg, clReg, 3, reset)
    return stab_circuit

# Correct possible errors
def recovery_scheme(qbReg, clReg, reset=True):
    '''OLD FUNCTION, replaced by unflagged_recovery'''
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
# %% NEW FUNCTIONS
def unflagged_stabilizer_all(qbReg, anReg, clReg, reset=True):
    '''Runs all four stabilizers without flags'''
    circ = QuantumCircuit(qbReg, anReg, clReg)
    circ += unflagged_stabilizer_XZZXI(qbReg, anReg, clReg, reset)
    circ += unflagged_stabilizer_IXZZX(qbReg, anReg, clReg, reset)
    circ += unflagged_stabilizer_XIXZZ(qbReg, anReg, clReg, reset)
    circ += unflagged_stabilizer_ZXIXZ(qbReg, anReg, clReg, reset)
    return circ

def unflagged_recovery(qbReg, clReg, reset=True):
    '''Lookup table for recovery from a
    single qubit error on code qubits'''

    circ = QuantumCircuit(qbReg, clReg)

    # If the ancilla is reset to |0> between measurements
    if reset:
        circ.x(qbReg[1]).c_if(clReg, 1)
        circ.z(qbReg[4]).c_if(clReg, 2)
        circ.x(qbReg[2]).c_if(clReg, 3)
        circ.z(qbReg[2]).c_if(clReg, 4)
        circ.z(qbReg[0]).c_if(clReg, 5)
        circ.x(qbReg[3]).c_if(clReg, 6)
        circ.x(qbReg[2]).c_if(clReg, 7)
        circ.z(qbReg[2]).c_if(clReg, 7)
        circ.x(qbReg[0]).c_if(clReg, 8)
        circ.z(qbReg[3]).c_if(clReg, 9)
        circ.z(qbReg[1]).c_if(clReg, 10)
        circ.x(qbReg[1]).c_if(clReg, 11)
        circ.z(qbReg[1]).c_if(clReg, 11)
        circ.x(qbReg[4]).c_if(clReg, 12)
        circ.x(qbReg[0]).c_if(clReg, 13)
        circ.z(qbReg[0]).c_if(clReg, 13)
        circ.x(qbReg[4]).c_if(clReg, 14)
        circ.z(qbReg[4]).c_if(clReg, 14)
        circ.x(qbReg[3]).c_if(clReg, 15)
        circ.z(qbReg[3]).c_if(clReg, 15)

    # If the ancilla is NOT reset between measurements
    else:
        circ.x(qbReg[2]).c_if(clReg, 1)
        circ.x(qbReg[3]).c_if(clReg, 2)
        circ.z(qbReg[0]).c_if(clReg, 3)
        circ.x(qbReg[4]).c_if(clReg, 4)
        circ.z(qbReg[3]).c_if(clReg, 5)
        circ.x(qbReg[3]).c_if(clReg, 5)
        circ.z(qbReg[1]).c_if(clReg, 6)
        circ.z(qbReg[3]).c_if(clReg, 7)
        circ.x(qbReg[0]).c_if(clReg, 8)
        circ.z(qbReg[1]).c_if(clReg, 9)
        circ.x(qbReg[1]).c_if(clReg, 9)
        circ.z(qbReg[4]).c_if(clReg, 10)
        circ.x(qbReg[4]).c_if(clReg, 10)
        circ.z(qbReg[0]).c_if(clReg, 11)
        circ.x(qbReg[0]).c_if(clReg, 11)
        circ.z(qbReg[2]).c_if(clReg, 12)
        circ.z(qbReg[2]).c_if(clReg, 13)
        circ.x(qbReg[2]).c_if(clReg, 13)
        circ.z(qbReg[4]).c_if(clReg, 14)
        circ.x(qbReg[1]).c_if(clReg, 15)

    return circ

# %% New Stabilizer
def flagged_stabilizer_cycle(qbReg, anReg, clReg, reset=True):
    '''Runs the one cycle of the [[5,1,3]] code 
    with two ancillas as described in the article.
    This includes the recovery from any detected errors.
    
    Currently, it requires reset=True to appropriately
    correct errors'''

    # Define the circuit
    circ = QuantumCircuit(qbReg, anReg, clReg)

    ## === Step 1: XZZXI ===
    circ += flagged_stabilizer_XZZXI(qbReg, anReg, clReg, reset=True)
    circ += unflagged_stabilizer_all(qbReg, anReg, clReg, reset=True) # To be made conditional
    circ += full_recovery_XZZXI(qbReg, clReg)

    ## === Step 2: IXZZX ===
    circ += flagged_stabilizer_IXZZX(qbReg, anReg, clReg, reset=True)
    circ += unflagged_stabilizer_all(qbReg, anReg, clReg, reset=True) # To be made conditional
    circ += full_recovery_IXZZX( qbReg, clReg )

    ## === Step 3: XIXZZ ===
    circ += flagged_stabilizer_XIXZZ(qbReg, anReg, clReg, reset=True)
    circ += unflagged_stabilizer_all(qbReg, anReg, clReg, reset=True) # To be made conditional
    circ += full_recovery_XIXZZ( qbReg, clReg )
    
    ## === Step 4: ZXIXZ ===
    circ += flagged_stabilizer_ZXIXZ(qbReg, anReg, clReg, reset=True)
    circ += unflagged_stabilizer_all(qbReg, anReg, clReg, reset=True) # To be made conditional
    circ += full_recovery_ZXIXZ( qbReg, clReg )

    return circ


# %%

from qiskit import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import state_fidelity
from qiskit.providers.aer.extensions.snapshot_statevector import *

# Defining useful functions
# Note that this does not consider our setup
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

    return [logical_0, logical_1]

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
# %%
# Running the quantum circuit

def define_circuit(n_cycles):
    '''Creates the entire circuit and returns it
    as an output. Input is the number of stabilizer
    cycles to perform'''
    # Define our registers

    qb = QuantumRegister(5, 'code_qubit')
    an = QuantumRegister(2, 'ancilla_qubit')
    cr = ClassicalRegister(5, 'syndrome_bit')
    readout = ClassicalRegister(5, 'readout')

    circuit = QuantumCircuit(cr, readout, an, qb)

    # Prepare the input
    circuit.x(qb[0])  # As an example, start in |1>

    # Encode the state
    circuit += encode_input(qb)
    circuit.snapshot_statevector('post_encoding')

    # Random error
    #circuit.z( qb[1] )
    #circuit.x( qb[3] )

    # Stabilizers
    for i in range(n_cycles):
        circuit += flagged_stabilizer_cycle( qb, an, cr, reset=True )
    #    circuit += run_stabilizer(qb, an, cr, flag, reset=True)
    #    circuit += recovery_scheme(qb, cr, reset=True)
        circuit.snapshot_statevector('stabilizer')

    # Readout of the encoded state
    # Measure at the end of the run
    circuit.measure(qb, readout)
    circuit.snapshot_statevector('post_measure')

    return circuit



# %% All flagged stabilizers
def flagged_stabilizer_XZZXI(qbReg, anReg, clReg, reset=True):

    # Create a circuit
    circ = QuantumCircuit(qbReg, anReg, clReg)

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
    if reset:
        circ.reset(anReg[1])
        circ.reset(anReg[0])

    return circ

def flagged_stabilizer_IXZZX(qbReg, anReg, clReg, reset=True):

    # Create a circuit
    circ = QuantumCircuit(qbReg, anReg, clReg)

    # X
    circ.h( qbReg[1] ) # X
    circ.h( anReg[1] )
    circ.cz( anReg[1], qbReg[1] )
    circ.h( qbReg[1] )

    # Flag
    circ.h( anReg[0] ) # Initialize in +
    circ.cz( anReg[0], anReg[1] ) # Flag CX        

    # Z
    circ.cz( anReg[1], qbReg[2] )

    # Z
    circ.cz( anReg[1], qbReg[3] )

    # Flag
    circ.cz( anReg[0], anReg[1] )   

    # X
    circ.h(qbReg[4])
    circ.cz(anReg[1], qbReg[4])
    circ.h(anReg[1])
    circ.h(qbReg[4])

    # Measure
    circ.measure(anReg[1], clReg[1])
    circ.h(anReg[0])
    circ.measure(anReg[0], clReg[4])
    if reset:
        circ.reset(anReg[1])
        circ.reset(anReg[0])

    return circ

def flagged_stabilizer_XIXZZ(qbReg, anReg, clReg, reset=True):

    # Create a circuit
    circ = QuantumCircuit(qbReg, anReg, clReg)

    # X
    circ.h( qbReg[0] ) # X
    circ.h( anReg[1] )
    circ.cz( anReg[1], qbReg[0] )
    circ.h( qbReg[0] )

    # Flag
    circ.h( anReg[0] ) # Initialize in +
    circ.cz( anReg[0], anReg[1] ) # Flag CX        

    # X
    circ.h( qbReg[2] ) # X
    circ.cz( anReg[1], qbReg[2] )
    circ.h( qbReg[2] )

    # Z
    circ.cz( anReg[1], qbReg[3] )

    # Flag
    circ.cz( anReg[0], anReg[1] )   

    # Z
    circ.cz( anReg[1], qbReg[4] )
    circ.h(anReg[1])

    # Measure
    circ.measure(anReg[1], clReg[2])
    circ.h(anReg[0])
    circ.measure(anReg[0], clReg[4])
    if reset:
        circ.reset(anReg[1])
        circ.reset(anReg[0])

    return circ

def flagged_stabilizer_ZXIXZ(qbReg, anReg, clReg, reset=True):

    # Create a circuit
    circ = QuantumCircuit(qbReg, anReg, clReg)

    # Z
    circ.h( anReg[1] )
    circ.cz( anReg[1], qbReg[0] )

    # Flag
    circ.h( anReg[0] ) # Initialize in +
    circ.cz( anReg[0], anReg[1] ) # Flag CX  

    # X
    circ.h( qbReg[1] ) # X
    circ.cz( anReg[1], qbReg[1] )
    circ.h( qbReg[1] )

    # X
    circ.h( qbReg[3] ) # X
    circ.cz( anReg[1], qbReg[3] )
    circ.h( qbReg[3] )

    # Flag
    circ.cz( anReg[0], anReg[1] )   
    
    # Z
    circ.cz( anReg[1], qbReg[4] )
    circ.h(anReg[1])

    # Measure
    circ.measure(anReg[1], clReg[3])
    circ.h(anReg[0])
    circ.measure(anReg[0], clReg[4])
    if reset:
        circ.reset(anReg[1])
        circ.reset(anReg[0])
    
    return circ

# %% All unflagged stabilizers

def unflagged_stabilizer_XZZXI(qbReg, anReg, clReg, reset=True):

    # Create a circuit
    circ = QuantumCircuit(qbReg, anReg, clReg)

    # X
    circ.h( qbReg[0] ) # X
    circ.h( anReg[1] )
    circ.cz( anReg[1], qbReg[0] )
    circ.h( qbReg[0] )       

    # Z
    circ.cz(anReg[1], qbReg[1])

    # Z
    circ.cz( anReg[1], qbReg[2] )

    # X
    circ.h(qbReg[3])
    circ.cz(anReg[1], qbReg[3])
    circ.h(anReg[1])
    circ.h(qbReg[3])

    # Measure
    circ.measure(anReg[1], clReg[0])
    if reset:
        circ.reset(anReg[1])

    return circ

def unflagged_stabilizer_IXZZX(qbReg, anReg, clReg, reset=True):

    # Create a circuit
    circ = QuantumCircuit(qbReg, anReg, clReg)

    # X
    circ.h( qbReg[1] ) # X
    circ.h( anReg[1] )
    circ.cz( anReg[1], qbReg[1] )
    circ.h( qbReg[1] )   

    # Z
    circ.cz( anReg[1], qbReg[2] )

    # Z
    circ.cz( anReg[1], qbReg[3] ) 

    # X
    circ.h(qbReg[4])
    circ.cz(anReg[1], qbReg[4])
    circ.h(anReg[1])
    circ.h(qbReg[4])

    # Measure
    circ.measure(anReg[1], clReg[1])
    if reset:
        circ.reset(anReg[1])

    return circ

def unflagged_stabilizer_XIXZZ(qbReg, anReg, clReg, reset=True):

    # Create a circuit
    circ = QuantumCircuit(qbReg, anReg, clReg)

    # X
    circ.h( anReg[1] )
    circ.h( qbReg[0] ) # X
    circ.cz( anReg[1], qbReg[0] )
    circ.h( qbReg[0] )      

    # X
    circ.h( qbReg[2] ) # X
    circ.cz( anReg[1], qbReg[2] )
    circ.h( qbReg[2] )

    # Z
    circ.cz( anReg[1], qbReg[3] )

    # Z
    circ.cz( anReg[1], qbReg[4] )
    circ.h(anReg[1])

    # Measure
    circ.measure(anReg[1], clReg[2])
    if reset:
        circ.reset(anReg[1])


    return circ

def unflagged_stabilizer_ZXIXZ(qbReg, anReg, clReg, reset=True):

    # Create a circuit
    circ = QuantumCircuit(qbReg, anReg, clReg)

    # Z
    circ.h( anReg[1] )
    circ.cz( anReg[1], qbReg[0] )

    # X
    circ.h( qbReg[1] ) # X
    circ.cz( anReg[1], qbReg[1] )
    circ.h( qbReg[1] )

    # X
    circ.h( qbReg[3] ) # X
    circ.cz( anReg[1], qbReg[3] )
    circ.h( qbReg[3] )
    
    # Z
    circ.cz( anReg[1], qbReg[4] )
    circ.h(anReg[1])

    # Measure
    circ.measure(anReg[1], clReg[3])
    if reset:
        circ.reset(anReg[1])
    
    return circ

# %% All recoveries
def full_recovery_XZZXI( qbReg, clReg ):

    circ = QuantumCircuit(qbReg, clReg)

    # Unflagged recovery
    circ += unflagged_recovery(qbReg, clReg, reset=True)

    # Flagged recovery
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

    return circ

def full_recovery_IXZZX( qbReg, clReg ):

    circ = QuantumCircuit(qbReg, clReg)

    # Unflagged recovery
    circ += unflagged_recovery(qbReg, clReg, reset=True)

    # Flagged recovery
    circ.x( qbReg[1] ).c_if( clReg, 1+16 )
    circ.x( qbReg[1] ).c_if( clReg, 2+16 )
    circ.x( qbReg[2] ).c_if( clReg, 2+16 )
    circ.y( qbReg[3] ).c_if( clReg, 3+16 )
    circ.x( qbReg[4] ).c_if( clReg, 3+16 )
    circ.z( qbReg[3] ).c_if( clReg, 5+16 )
    circ.x( qbReg[4] ).c_if( clReg, 5+16 )
    circ.x( qbReg[0] ).c_if( clReg, 6+16 )
    circ.y( qbReg[4] ).c_if( clReg, 6+16 )
    circ.x( qbReg[3] ).c_if( clReg,10+16 )
    circ.x( qbReg[4] ).c_if( clReg,10+16 )
    circ.x( qbReg[4] ).c_if( clReg,12+16 )

    return circ

def full_recovery_XIXZZ( qbReg, clReg ):

    circ = QuantumCircuit(qbReg, clReg)

    # Unflagged recovery
    circ += unflagged_recovery(qbReg, clReg, reset=True)

    # Flagged recovery
    circ.x( qbReg[1] ).c_if( clReg, 2+16)
    circ.x( qbReg[2] ).c_if( clReg, 2+16)
    circ.x( qbReg[3] ).c_if( clReg, 4+16)
    circ.z( qbReg[4] ).c_if( clReg, 4+16)
    circ.x( qbReg[0] ).c_if( clReg, 8+16) # Seems equivalent with IZZXI
    circ.x( qbReg[0] ).c_if( clReg, 11+16)
    circ.x( qbReg[2] ).c_if( clReg, 11+16)
    circ.x( qbReg[4] ).c_if( clReg, 12+16)
    circ.z( qbReg[0] ).c_if( clReg, 13+16)
    circ.z( qbReg[2] ).c_if( clReg, 13+16)
    circ.x( qbReg[4] ).c_if( clReg, 13+16)
    circ.x( qbReg[2] ).c_if( clReg, 15+16)
    circ.x( qbReg[4] ).c_if( clReg, 15+16)

    return circ

def full_recovery_ZXIXZ( qbReg, clReg ):

    circ = QuantumCircuit(qbReg, clReg)

    # Unflagged recovery
    circ += unflagged_recovery(qbReg, clReg, reset=True)

    # Flagged recovery
    circ.x( qbReg[1] ).c_if( clReg, 2+16)
    circ.x( qbReg[2] ).c_if( clReg, 2+16)
    circ.x( qbReg[3] ).c_if( clReg, 4+16)
    circ.z( qbReg[4] ).c_if( clReg, 4+16)
    circ.x( qbReg[2] ).c_if( clReg, 5+16)
    circ.x( qbReg[3] ).c_if( clReg, 5+16)
    circ.x( qbReg[0] ).c_if( clReg, 11+16)
    circ.x( qbReg[2] ).c_if( clReg, 11+16)
    circ.z( qbReg[0] ).c_if( clReg, 13+16)
    circ.z( qbReg[2] ).c_if( clReg, 13+16)
    circ.x( qbReg[4] ).c_if( clReg, 13+16)
    circ.x( qbReg[0] ).c_if( clReg, 14+16)
    circ.z( qbReg[2] ).c_if( clReg, 14+16)
    circ.z( qbReg[4] ).c_if( clReg, 14+16)
    circ.x( qbReg[2] ).c_if( clReg, 15+16)
    circ.x( qbReg[4] ).c_if( clReg, 15+16)

    return circ

# %% TESTING THE PROCEDURE

# Create the circuit
n_cycles = 1
circuit = define_circuit( n_cycles )

n_shots = 2000
results = execute(
    circuit,  
    Aer.get_backend('qasm_simulator'),
    noise_model=None,
    shots=n_shots
).result()

# Analyze results
logical = logical_states()
sv_post_encoding = results.data()['snapshots']['statevector']['stabilizer'][0]
fid = 0
for i in range(10):
    fid += state_fidelity(logical[1], results.data()['snapshots']['statevector']['stabilizer'][i])

print('Average fidelity:')
print(fid/10)

# Plot results
counts = results.get_counts()
plot_histogram( counts )
#circuit.draw(output='mpl')