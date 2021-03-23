# %% Import modules
import numpy as np
from qiskit import (QuantumCircuit,
                    QuantumRegister, 
                    ClassicalRegister, 
                    AncillaRegister, 
                    execute, 
                    Aer
                    )
#234567890123456789012345678901234567890123456789012345678901234567890123456789
# Import our own files
from custom_noise_models import pauli_noise_model
from custom_transpiler import shortest_transpile_from_distribution, WAQCT_device_properties
#from stabilizers import (flagged_stabilizer_cycle, unflagged_stabilizer_all)

# %% NEW FUNCTIONS
def unflagged_stabilizer_all(qbReg, anReg, clReg, reset=True, current_cycle=0, current_step=0):
    '''Runs all four stabilizers without flags'''
    circ = create_empty_circuit(qbReg, anReg, clReg)
    circ += unflagged_stabilizer_XZZXI(qbReg, anReg, clReg, reset, current_cycle, current_step)
    circ += unflagged_stabilizer_IXZZX(qbReg, anReg, clReg, reset, current_cycle, current_step)
    circ += unflagged_stabilizer_XIXZZ(qbReg, anReg, clReg, reset, current_cycle, current_step)
    circ += unflagged_stabilizer_ZXIXZ(qbReg, anReg, clReg, reset, current_cycle, current_step)
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

# %% All flagged stabilizers
def flagged_stabilizer_cycle(qbReg, anReg, clReg, reset=True, recovery=True, current_cycle=0):
    '''Runs the one cycle of the [[5,1,3]] code 
    with two ancillas as described in the article.
    This includes the recovery from any detected errors.
    
    Currently, it requires reset=True to appropriately
    correct errors'''

    # Define the circuit
    circ = create_empty_circuit(qbReg, anReg, clReg)

    ## === Step 1: XZZXI ===
    circ += _flagged_stabilizer_XZZXI(qbReg, anReg, clReg, reset=reset)
    circ += unflagged_stabilizer_all(qbReg, anReg, clReg, 
        reset=reset, current_cycle=current_cycle, current_step=0) # To be made conditional
    if recovery:
        circ += full_recovery_XZZXI(qbReg, clReg)

    ## === Step 2: IXZZX ===
    circ += _flagged_stabilizer_IXZZX(qbReg, anReg, clReg, reset=reset)
    circ += unflagged_stabilizer_all(qbReg, anReg, clReg,
        reset=reset, current_cycle=current_cycle, current_step=1) # To be made conditional
    if recovery:
        circ += full_recovery_IXZZX( qbReg, clReg )

    ## === Step 3: XIXZZ ===
    circ += _flagged_stabilizer_XIXZZ(qbReg, anReg, clReg, reset=reset)
    circ += unflagged_stabilizer_all(qbReg, anReg, clReg,
        reset=reset, current_cycle=current_cycle, current_step=2) # To be made conditional
    if recovery:
        circ += full_recovery_XIXZZ( qbReg, clReg )
    
    ## === Step 4: ZXIXZ ===
    circ += _flagged_stabilizer_ZXIXZ(qbReg, anReg, clReg, reset=reset)
    circ += unflagged_stabilizer_all(qbReg, anReg, clReg,
        reset=reset, current_cycle=current_cycle, current_step=3) # To be made conditional
    if recovery:
        circ += full_recovery_ZXIXZ( qbReg, clReg )

    return circ

def _flagged_stabilizer_XZZXI(qbReg, anReg, clReg, reset=True, current_cycle=0):

    # Create a circuit
    circ = create_empty_circuit(qbReg, anReg, clReg)

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
    if isinstance(clReg, list):
        flag_register = clReg[1]
        ancilla_msmnt_register = clReg[2]

        circ.measure(anReg[1], ancilla_msmnt_register[current_cycle][0])
        circ.h(anReg[0])
        circ.measure(anReg[0], flag_register[current_cycle][0])

    else:
        circ.measure(anReg[1], clReg[0])
        circ.h(anReg[0])
        circ.measure(anReg[0], clReg[4])

    # Reset
    if reset:
        circ.reset(anReg[1])
        circ.reset(anReg[0])
    
    return circ

def _flagged_stabilizer_IXZZX(qbReg, anReg, clReg, reset=True, current_cycle=0):

    # Create a circuit
    circ = create_empty_circuit(qbReg, anReg, clReg)

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
    if isinstance(clReg, list):
        flag_register = clReg[1]
        ancilla_msmnt_register = clReg[2]

        circ.measure(anReg[1], ancilla_msmnt_register[current_cycle][1])
        circ.h(anReg[0])
        circ.measure(anReg[0], flag_register[current_cycle][1])

    else:
        circ.measure(anReg[1], clReg[1])
        circ.h(anReg[0])
        circ.measure(anReg[0], clReg[4])

    # Reset
    if reset:
        circ.reset(anReg[1])
        circ.reset(anReg[0])
    
    return circ

    return circ

def _flagged_stabilizer_XIXZZ(qbReg, anReg, clReg, reset=True, current_cycle=0):

    # Create a circuit
    circ = create_empty_circuit(qbReg, anReg, clReg)

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
    if isinstance(clReg, list):
        flag_register = clReg[1]
        ancilla_msmnt_register = clReg[2]

        circ.measure(anReg[1], ancilla_msmnt_register[current_cycle][2])
        circ.h(anReg[0])
        circ.measure(anReg[0], flag_register[current_cycle][2])

    else:
        circ.measure(anReg[1], clReg[2])
        circ.h(anReg[0])
        circ.measure(anReg[0], clReg[4])

    # Reset
    if reset:
        circ.reset(anReg[1])
        circ.reset(anReg[0])
    
    return circ

    return circ

def _flagged_stabilizer_ZXIXZ(qbReg, anReg, clReg, reset=True, current_cycle=0):

    # Create a circuit
    circ = create_empty_circuit(qbReg, anReg, clReg)

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
    if isinstance(clReg, list):
        flag_register = clReg[1]
        ancilla_msmnt_register = clReg[2]

        circ.measure(anReg[1], ancilla_msmnt_register[current_cycle][3])
        circ.h(anReg[0])
        circ.measure(anReg[0], flag_register[current_cycle][3])

    else:
        circ.measure(anReg[1], clReg[3])
        circ.h(anReg[0])
        circ.measure(anReg[0], clReg[4])

    # Reset
    if reset:
        circ.reset(anReg[1])
        circ.reset(anReg[0])
    
    return circ
    
    return circ

# %% All unflagged stabilizers
def unflagged_stabilizer_XZZXI(qbReg, anReg, clReg, reset=True, current_cycle=0, current_step=0):

    # Create a circuit
    circ = create_empty_circuit(qbReg, anReg, clReg)

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
    if isinstance(clReg, list):
        syndrome_register = clReg[0]
        circ.measure( anReg[1], syndrome_register[current_cycle][current_step][0] )
    else:
        circ.measure(anReg[1], clReg[0])
    
    # Reset
    if reset:
        circ.reset(anReg[1])

    return circ

def unflagged_stabilizer_IXZZX(qbReg, anReg, clReg, reset=True, current_cycle=0, current_step=0):

    # Create a circuit
    circ = create_empty_circuit(qbReg, anReg, clReg)

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
    if isinstance(clReg, list):
        syndrome_register = clReg[0]
        circ.measure( anReg[1], syndrome_register[current_cycle][current_step][1] )
    else:
        circ.measure(anReg[1], clReg[1])
    
    # Reset
    if reset:
        circ.reset(anReg[1])

    return circ

def unflagged_stabilizer_XIXZZ(qbReg, anReg, clReg, reset=True, current_cycle=0, current_step=0):

    # Create a circuit
    circ = create_empty_circuit(qbReg, anReg, clReg)

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
    if isinstance(clReg, list):
        syndrome_register = clReg[0]
        circ.measure( anReg[1], syndrome_register[current_cycle][current_step][2] )
    else:
        circ.measure(anReg[1], clReg[2])
    
    # Reset
    if reset:
        circ.reset(anReg[1])


    return circ

def unflagged_stabilizer_ZXIXZ(qbReg, anReg, clReg, reset=True, current_cycle=0, current_step=0):

    # Create a circuit
    circ = create_empty_circuit(qbReg, anReg, clReg)

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
    if isinstance(clReg, list):
        syndrome_register = clReg[0]
        circ.measure( anReg[1], syndrome_register[current_cycle][current_step][3] )
    else:
        circ.measure(anReg[1], clReg[3])
    
    # Reset
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


# %% Import modules for testing

from qiskit import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import state_fidelity
from qiskit.providers.aer.extensions.snapshot_statevector import *

# Import our own files
from custom_noise_models import pauli_noise_model

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

def get_classical_register(n_cycles):
    '''Generate lists of classical registers for
    storing all measurement data for the flagged
    error correction code'''


    # List of registers for each iteration of the conditional
    # step of 'all four unflagged stabilizers'
    syndrome_register = [
        [ ClassicalRegister( 4, 'syndrome_cycle_' +str(i) +'_step_' +str(j)) 
        for j in range(4) ] for i in range(n_cycles) ]

    # List of registers for each step in the flagged stabilizer cycle
    flag_register = [ 
        [ ClassicalRegister( 1, 'flag_cycle_' +str(i) +'_step_' +str(j))
        for j in range(4) ] for i in range(n_cycles) ]

    # List of registers for the single stabilizer run with flag
    ancilla_msmnt_register = [
        [ ClassicalRegister( 1, 'ancilla_cycle_' +str(i) +'_step_' +str(j))
        for j in range(4) ] for i in range(n_cycles) ]

    return [syndrome_register, flag_register, ancilla_msmnt_register]

def create_empty_circuit(qbReg, anReg, clReg):
    circ = QuantumCircuit(qbReg, anReg)
    if isinstance(clReg, list):
        circ = QuantumCircuit(qbReg, anReg)
        for reg_type in clReg:
            for reg_index in reg_type:
                for reg in reg_index:
                    circ.add_register(reg)
    else:
        circ.add_register(clReg)
    #circ.add_register(readout)

    return circ

# %% Running the quantum circuit 
def define_circuit(n_cycles, flag=True):
    '''Creates the entire circuit and returns it
    as an output. Input is the number of stabilizer
    cycles to perform'''
    # Define our registers

    qb = QuantumRegister(5, 'code_qubit')
    an = AncillaRegister(2, 'ancilla_qubit')
    #cr = ClassicalRegister(5, 'syndrome_bit')
    cr = get_classical_register(n_cycles)
    readout = ClassicalRegister(5, 'readout')

    #circuit = QuantumCircuit(cr, readout, an, qb)
    circuit = create_empty_circuit(qb, an, cr)
    circuit.add_register(readout)

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
        if flag:
            circuit += flagged_stabilizer_cycle( qb, an, cr, reset=True, recovery=False,
                current_cycle=i )
            circuit.snapshot_statevector('stabilizer_' + str(i))
        else:
            circuit += unflagged_stabilizer_all( qb, an, cr, reset=True )
            circuit += unflagged_recovery(qb, cr, reset=True)
            circuit.snapshot_statevector('stabilizer_' + str(i))
        

    # Readout of the encoded state
    # Measure at the end of the run
    circuit.measure(qb, readout)
    circuit.snapshot_statevector('post_measure')

    return circuit


# %% Simple testing with a single cycle

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
sv_post_encoding = results.data()['snapshots']['statevector']['stabilizer_0'][0]
fid = 0
for i in range(10):
    sv_post_encoding = results.data()['snapshots']['statevector']['stabilizer_0'][i]

    log1 = logical[1][np.arange(128,step=4)]
    #sv_test = sv_post_encoding[np.arange(128,step=4)]
    sv_test = sv_post_encoding[0:32]
    fid += state_fidelity(log1, sv_test)
    #fid += state_fidelity(logical[1], results.data()['snapshots']['statevector']['stabilizer_0'][i])

print('Average fidelity:')
print(fid/10)

# Plot results
counts = results.get_counts()
plot_histogram( counts )
#circuit.draw(output='mpl')

# %%
print(sv_post_encoding.shape)
sv_real = sv_post_encoding.real
print(np.linalg.norm(sv_real))
for i in range(128):
    sv_real[i] = round(sv_real[i], 2)
print(sv_real[0:3])
print(logical[1])
print(log1)
# %% Comparison between flagged and unflagged

n_cycles = 10
flagged_circuit = define_circuit( n_cycles, flag=True )
unflagged_circuit = define_circuit( n_cycles, flag=False )
noise = pauli_noise_model(0.001, 0.00, 0.0)
n_shots = 2000

results_flag_ideal = execute(
    flagged_circuit,  
    Aer.get_backend('qasm_simulator'),
    noise_model=None,
    shots=n_shots
).result()

results_flag_noise = execute(
    flagged_circuit,  
    Aer.get_backend('qasm_simulator'),
    noise_model=noise,
    shots=n_shots
).result()

results_ideal = execute(
    unflagged_circuit,  
    Aer.get_backend('qasm_simulator'),
    noise_model=None,
    shots=n_shots
).result()

results_noise = execute(
    unflagged_circuit,  
    Aer.get_backend('qasm_simulator'),
    noise_model=noise,
    shots=n_shots
).result()

logical_state = get_fidelities(
    results_noisy=results_noise, results_ideal=results_ideal, snapshot_type='statevector')
logical_state_flag = get_fidelities(
    results_noisy=results_flag_noise, results_ideal=results_flag_ideal, snapshot_type='statevector')


# Plotting
sns.set_context('talk', rc={"lines.linewidth": 2.5})
default_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                  'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

fig = plt.figure(figsize=[10, 6])
plt.plot(logical_state, marker='o', label=r'Regular')
plt.plot(logical_state_flag, marker='o', label=r'Flagged')

plt.xticks(ticks=range(n_cycles+1))
plt.xlabel('Number of cycles')
plt.title('Average fidelity across stabilizer cycles')
plt.legend()
plt.show()

# %%
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
    return np.sum(running_fidelity, 0) / (n_shots+1)


def get_fidelities(results_noisy, results_ideal, snapshot_type):
    # logical0 = logical_0_transp  # logical[0]  #
    if snapshot_type == 'density_matrix':
        return _get_fidelities_mat(results_noisy, results_ideal)
    return _get_fidelities_vec(results_noisy, results_ideal)




def get_full_stabilizer_circuit(registers, n_cycles=1,
        reset=True, recovery=False, flag=True):
    """Returns the circuit for a full stabilizer circuit, including encoding, 
    stabilizers (with conditional flags and recovery) and final measurement.
    """

    # Unpack registers
    qbReg = registers[0]
    anReg = registers[1]
    clReg = registers[2]
    readout = registers[3]

    # Define the circuit
    qb = QuantumRegister(5, 'code_qubit')
    an = AncillaRegister(2, 'ancilla_qubit')
    readout = ClassicalRegister(5, 'readout')
    if n_cycles > 1:
        cr = get_classical_register(n_cycles)
    else:
        cr = ClassicalRegister(5, 'syndrome_bit')
    circ = create_empty_circuit(qb, an, cr)
    circ.add_register(readout)

    # Encode the state
    circ += encode_input(qb)
    circ.snapshot_statevector('post_encoding')

    # Stabilizer
    for i in range(n_cycles):
        if flag is True:
            circ += flagged_stabilizer_cycle(registers,
                reset=reset,
                recovery=recovery,
                current_cycle=i,
            )
        else:
            # ADD AS CYCLE FUNCTION WHICH INCLDUES RECOVERY
            circ += unflagged_stabilizer_all(registers, reset=reset)
        circ.snapshot_statevector('stabilizer_' + str(i))

    # Final readout
    circ.measure(qb, readout)
    circ.snapshot_statevector('post_measure')

    return circuit