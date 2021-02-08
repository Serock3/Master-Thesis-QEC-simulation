# Import modules
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister, execute, Aer

# Run a single stabilizer
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

    stab_circuit = QuantumCircuit(qbReg, anReg, clReg)#, flagReg)

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
def run_stabilizer(qbReg, anReg, clReg, flagReg, reset=True):
    stab_circuit = QuantumCircuit(qbReg, anReg, clReg, flagReg)
    stab_circuit += measure_stabilizer(qbReg, anReg, clReg, flagReg, 0, reset)
    stab_circuit += measure_stabilizer(qbReg, anReg, clReg, flagReg, 1, reset)
    stab_circuit += measure_stabilizer(qbReg, anReg, clReg, flagReg, 2, reset)
    stab_circuit += measure_stabilizer(qbReg, anReg, clReg, flagReg, 3, reset)
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

# %% New Stabilizer
def flagged_stabilizer_cycle(qbReg, anReg, clReg, reset=True):#, flagReg
    '''Runs the one cycle of the [[5,1,3]] code 
    with two ancillas as described in the article '''

    # Create a circuit
    circ = QuantumCircuit(qbReg, anReg, clReg)#, flagReg)

    ## ===Step 1: Run XZZXI with flag===

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
    
    # Run the stabilizers
    circ += measure_stabilizer(qbReg, anReg, clReg, 0, reset=True)
    circ += measure_stabilizer(qbReg, anReg, clReg, 1, reset=True)
    circ += measure_stabilizer(qbReg, anReg, clReg, 2, reset=True)
    circ += measure_stabilizer(qbReg, anReg, clReg, 3, reset=True)

    # Temporary recovery
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

    ## ===Step 2: Run IXZZX with flag===

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
    
    # Run the stabilizers
    circ += measure_stabilizer(qbReg, anReg, clReg, 0, reset=True)
    circ += measure_stabilizer(qbReg, anReg, clReg, 1, reset=True)
    circ += measure_stabilizer(qbReg, anReg, clReg, 2, reset=True)
    circ += measure_stabilizer(qbReg, anReg, clReg, 3, reset=True)

    # Temporary recovery 2
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

    
    ## ===Step 3: Run XIXZZ with flag===

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
    
    # Run the stabilizers
    circ += measure_stabilizer(qbReg, anReg, clReg, 0, reset=True)
    circ += measure_stabilizer(qbReg, anReg, clReg, 1, reset=True)
    circ += measure_stabilizer(qbReg, anReg, clReg, 2, reset=True)
    circ += measure_stabilizer(qbReg, anReg, clReg, 3, reset=True)

    # Temporary recovery 3
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

    ## ===Step 4: Run ZXIXZ with flag===

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
    
    # Run the stabilizers
    circ += measure_stabilizer(qbReg, anReg, clReg, 0, reset=True)
    circ += measure_stabilizer(qbReg, anReg, clReg, 1, reset=True)
    circ += measure_stabilizer(qbReg, anReg, clReg, 2, reset=True)
    circ += measure_stabilizer(qbReg, anReg, clReg, 3, reset=True)

    # Temporary recovery 4
    circ.x( qbReg[1] ).c_if( clReg, 2+16 )
    circ.x( qbReg[2] ).c_if( clReg, 2+16 )
    circ.x( qbReg[3] ).c_if( clReg, 4+16 )
    circ.z( qbReg[4] ).c_if( clReg, 4+16 )
    circ.x( qbReg[2] ).c_if( clReg, 5+16 )
    circ.x( qbReg[3] ).c_if( clReg, 5+16 )
    circ.x( qbReg[0] ).c_if( clReg, 11+16 )
    circ.x( qbReg[2] ).c_if( clReg, 11+16 )
    circ.z( qbReg[0] ).c_if( clReg, 13+16 )
    circ.z( qbReg[2] ).c_if( clReg, 13+16 )
    circ.x( qbReg[4] ).c_if( clReg, 13+16 )
    circ.x( qbReg[0] ).c_if( clReg, 14+16 )
    circ.z( qbReg[2] ).c_if( clReg, 14+16 )
    circ.z( qbReg[4] ).c_if( clReg, 14+16 )
    circ.x( qbReg[2] ).c_if( clReg, 15+16 )
    circ.x( qbReg[4] ).c_if( clReg, 15+16 )

    return circ
