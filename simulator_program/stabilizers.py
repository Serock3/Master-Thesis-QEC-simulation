# %% Import modules
import numpy as np
from qiskit import (QuantumCircuit,
                    QuantumRegister, 
                    ClassicalRegister, 
                    AncillaRegister, 
                    execute, 
                    Aer
                    )
                    
# %% All flagged stabilizers
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

# %% All unflagged stabilizers
def unflagged_stabilizer_all(qbReg, anReg, clReg, reset=True):
    '''Runs all four stabilizers without flags'''
    circ = QuantumCircuit(qbReg, anReg, clReg)
    circ += unflagged_stabilizer_XZZXI(qbReg, anReg, clReg, reset)
    circ += unflagged_stabilizer_IXZZX(qbReg, anReg, clReg, reset)
    circ += unflagged_stabilizer_XIXZZ(qbReg, anReg, clReg, reset)
    circ += unflagged_stabilizer_ZXIXZ(qbReg, anReg, clReg, reset)
    return circ

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