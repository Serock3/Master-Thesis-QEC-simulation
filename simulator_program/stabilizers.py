# This file contains all neccessary functions for compiling and running the
# [[5,1,3]] error correction code, both with or without flagging. Most functions
# take optional arguments for whether to perform recovery, use flags, or reset
# the ancilla qubit. An example of how to use the functions is shown at the
# bottom.
#
# TO-DO: 
#   The advanced registers currently be used for recovery. Need to fix that,
#   but it has the problem of flag-measurement being in a separate register
#   and c_if can only be conditioned on one. 
#
#   Create a function for automatically generates registers for unflagged cycle
#
#   Add recovery possibilities for unflagged cycles with advanced registers
#
#   Double-check and improve all documentation

# %% Import modules
import numpy as np
from qiskit import (QuantumCircuit,
                    QuantumRegister, 
                    ClassicalRegister, 
                    AncillaRegister, 
                    execute, 
                    Aer
                    )
from qiskit.providers.aer.extensions.snapshot_statevector import *

# %% NEWLY ADDED FUNCTIONS
def get_full_stabilizer_circuit(registers, n_cycles=1,
        reset=True, recovery=False, flag=True):
    """Returns the circuit for a full stabilizer circuit, including encoding, 
    stabilizers (with conditional flags and recovery) and final measurement.
    """

    # Unpack registers
    qbReg, anReg, clReg, readout = unpack_stabilizer_registers(registers)

    # Define the circuit
    circ = get_empty_stabilizer_circuit(registers)

    # Encode the state
    circ += encode_input(registers)
    circ.snapshot_statevector('post_encoding')

    # Stabilizer
    for current_cycle in range(n_cycles):
        if flag is True:
            circ += flagged_stabilizer_cycle(registers,
                reset=reset,
                recovery=recovery,
                current_cycle=current_cycle,
            )
        else:
            circ += unflagged_stabilizer_cycle(registers,
                reset=reset,
                recovery=recovery
            )
        circ.snapshot_statevector('stabilizer_' + str(current_cycle))

    # Final readout
    circ.measure(qb, readout)
    circ.snapshot_statevector('post_measure')

    return circ

def get_empty_stabilizer_circuit(registers):
    """Create an empty qiskit circuit adapted for stabilizer circuits"""

    # Unpack registers
    qbReg, anReg, clReg, readout = unpack_stabilizer_registers(registers)

    circ = QuantumCircuit(qbReg, anReg)
    if isinstance(clReg, list):
        circ = QuantumCircuit(qbReg, anReg)
        for reg_type in clReg:
            for reg_index in reg_type:
                for reg in reg_index:
                    circ.add_register(reg)
    else:
        circ.add_register(clReg)
    circ.add_register(readout)

    return circ

def unpack_stabilizer_registers(registers):
    """Unpacks all registers packed as a list according to customs used
    in all stabilizer functions (stabilizer.py)
    """
    qbReg = registers[0]
    anReg = registers[1]
    clReg = registers[2]
    readout = registers[3]
    return qbReg, anReg, clReg, readout

def encode_input(registers):
    """Encode the input into logical 0 and 1 for the [[5,1,3]] code. This
    assumes that the 0:th qubit is the original state |psi> = a|0> + b|1>
    """

    qbReg, _, _, _ = unpack_stabilizer_registers(registers)
    circ = get_empty_stabilizer_circuit(registers)

    circ.h(qbReg[3])
    circ.cz(qbReg[3], qbReg[1])
    circ.cz(qbReg[3], qbReg[2])
    circ.cx(qbReg[3], qbReg[0])

    circ.h(qbReg[2])
    circ.cx(qbReg[2], qbReg[0])
    circ.cz(qbReg[2], qbReg[3])
    circ.cz(qbReg[2], qbReg[4])

    circ.h(qbReg[1])
    circ.cz(qbReg[1], qbReg[0])
    circ.cx(qbReg[1], qbReg[3])
    circ.cz(qbReg[1], qbReg[4])

    circ.h(qbReg[4])
    circ.cz(qbReg[4], qbReg[2])
    circ.cz(qbReg[4], qbReg[3])
    circ.cx(qbReg[4], qbReg[1])

    return circ

# %% Generate registers for storing ALL measurements
def get_classical_register(n_cycles):
    """Generate lists of classical registers for
    storing all measurement data for the flagged
    error correction code"""


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

# %% All flagged stabilizers
def flagged_stabilizer_cycle(registers, reset=True, recovery=True,
        current_cycle=0):
    '''Runs the one cycle of the [[5,1,3]] code 
    with two ancillas as described in the article.
    This includes the recovery from any detected errors.
    
    Currently, it requires reset=True to appropriately
    correct errors'''

    # Create a circuit
    qbReg, anReg, clReg, _ = unpack_stabilizer_registers(registers)
    circ = get_empty_stabilizer_circuit(registers)

    ## === Step 1: XZZXI ===
    circ += _flagged_stabilizer_XZZXI(registers, reset, current_cycle)
    circ += unflagged_stabilizer_cycle(registers, reset,
        current_cycle=current_cycle, current_step=0)
    if recovery:
        circ += full_recovery_XZZXI(registers)

    ## === Step 2: IXZZX ===
    circ += _flagged_stabilizer_IXZZX(registers, reset, current_cycle)
    circ += unflagged_stabilizer_cycle(registers, reset,
        current_cycle=current_cycle, current_step=1)
    if recovery:
        circ += full_recovery_IXZZX(registers)

    ## === Step 3: XIXZZ ===
    circ += _flagged_stabilizer_XIXZZ(registers, reset, current_cycle)
    circ += unflagged_stabilizer_cycle(registers, reset,
        current_cycle=current_cycle, current_step=2)
    if recovery:
        circ += full_recovery_XIXZZ(registers)
    
    ## === Step 4: ZXIXZ ===
    circ += _flagged_stabilizer_ZXIXZ(registers, reset, current_cycle)
    circ += unflagged_stabilizer_cycle(registers, reset,
        current_cycle=current_cycle, current_step=3)
    if recovery:
        circ += full_recovery_ZXIXZ(registers)

    return circ


def _flagged_stabilizer_XZZXI(registers, reset=True, current_cycle=0):

    # Create a circuit
    qbReg, anReg, clReg, _ = unpack_stabilizer_registers(registers)
    circ = get_empty_stabilizer_circuit(registers)

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

def _flagged_stabilizer_IXZZX(registers, reset=True, current_cycle=0):

    # Create a circuit
    qbReg, anReg, clReg, _ = unpack_stabilizer_registers(registers)
    circ = get_empty_stabilizer_circuit(registers)

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

def _flagged_stabilizer_XIXZZ(registers, reset=True, current_cycle=0):

    # Create a circuit
    qbReg, anReg, clReg, _ = unpack_stabilizer_registers(registers)
    circ = get_empty_stabilizer_circuit(registers)

    # X
    circ.h( qbReg[0] ) # X
    circ.h( anReg[1] )
    circ.cz( anReg[1], qbReg[0] )
    circ.h( qbReg[0] )

    # Flag
    circ.h( anReg[0] ) # Initialize in +
    circ.cz( anReg[0], anReg[1] ) # Flag CX        

    # X
    circ.h( qbReg[2] )
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

def _flagged_stabilizer_ZXIXZ(registers, reset=True, current_cycle=0):

    # Create a circuit
    qbReg, anReg, clReg, _ = unpack_stabilizer_registers(registers)
    circ = get_empty_stabilizer_circuit(registers)

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


# %% All unflagged stabilizers
def unflagged_stabilizer_cycle(registers, reset=True, recovery=False,
        current_cycle=0, current_step=0):
    """Run all four stabilizers without flags, as well as an optional
    recovery.
    """
    # Create circuit and run stabilizers
    circ = get_empty_stabilizer_circuit(registers)
    circ += _unflagged_stabilizer_XZZXI(registers, reset=reset,
        current_cycle=current_cycle, current_step=current_step)
    circ += _unflagged_stabilizer_IXZZX(registers, reset=reset,
        current_cycle=current_cycle, current_step=current_step)
    circ += _unflagged_stabilizer_XIXZZ(registers, reset=reset,
        current_cycle=current_cycle, current_step=current_step)
    circ += _unflagged_stabilizer_ZXIXZ(registers, reset=reset,
        current_cycle=current_cycle, current_step=current_step)

    # Recovery
    if recovery is True:
        circ += unflagged_recovery(qbReg, clReg, reset=True)
    
    return circ

def _unflagged_stabilizer_XZZXI(registers, reset=True,
        current_cycle=0, current_step=0):

    # Create a circuit
    qbReg, anReg, clReg, _ = unpack_stabilizer_registers(registers)
    circ = get_empty_stabilizer_circuit(registers)

    # X
    circ.h( qbReg[0] )
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
        syndrome_reg = clReg[0]
        circ.measure(anReg[1], syndrome_reg[current_cycle][current_step][0])
    else:
        circ.measure(anReg[1], clReg[0])
    
    # Reset
    if reset:
        circ.reset(anReg[1])

    return circ

def _unflagged_stabilizer_IXZZX(registers, reset=True,
        current_cycle=0, current_step=0):

    # Create a circuit
    qbReg, anReg, clReg, _ = unpack_stabilizer_registers(registers)
    circ = get_empty_stabilizer_circuit(registers)

    # X
    circ.h( qbReg[1] )
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
        syndrome_reg = clReg[0]
        circ.measure(anReg[1], syndrome_reg[current_cycle][current_step][1])
    else:
        circ.measure(anReg[1], clReg[1])
    
    # Reset
    if reset:
        circ.reset(anReg[1])

    return circ

def _unflagged_stabilizer_XIXZZ(registers, reset=True,
        current_cycle=0, current_step=0):

    # Create a circuit
    qbReg, anReg, clReg, _ = unpack_stabilizer_registers(registers)
    circ = get_empty_stabilizer_circuit(registers)

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
        syndrome_reg = clReg[0]
        circ.measure(anReg[1], syndrome_reg[current_cycle][current_step][2])
    else:
        circ.measure(anReg[1], clReg[2])
    
    # Reset
    if reset:
        circ.reset(anReg[1])

    return circ

def _unflagged_stabilizer_ZXIXZ(registers, reset=True,
        current_cycle=0, current_step=0):

    # Create a circuit
    qbReg, anReg, clReg, _ = unpack_stabilizer_registers(registers)
    circ = get_empty_stabilizer_circuit(registers)

    # Z
    circ.h( anReg[1] )
    circ.cz( anReg[1], qbReg[0] )

    # X
    circ.h( qbReg[1] )
    circ.cz( anReg[1], qbReg[1] )
    circ.h( qbReg[1] )

    # X
    circ.h( qbReg[3] )
    circ.cz( anReg[1], qbReg[3] )
    circ.h( qbReg[3] )
    
    # Z
    circ.cz( anReg[1], qbReg[4] )
    circ.h(anReg[1])

   # Measure
    if isinstance(clReg, list):
        syndrome_reg = clReg[0]
        circ.measure(anReg[1], syndrome_reg[current_cycle][current_step][3])
    else:
        circ.measure(anReg[1], clReg[3])
    
    # Reset
    if reset:
        circ.reset(anReg[1])
    
    return circ


# %% All recoveries
def unflagged_recovery(registers, reset=True):
    '''Lookup table for recovery from a
    single qubit error on code qubits'''

    # Create a circuit
    qbReg, anReg, clReg, _ = unpack_stabilizer_registers(registers)
    circ = get_empty_stabilizer_circuit(registers)

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

def full_recovery_XZZXI(registers, reset=True):

    # Create a circuit
    qbReg, anReg, clReg, _ = unpack_stabilizer_registers(registers)
    circ = get_empty_stabilizer_circuit(registers)

    # Unflagged recovery
    circ += unflagged_recovery(registers, reset=reset)

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

def full_recovery_IXZZX(registers, reset=True):

    # Create a circuit
    qbReg, anReg, clReg, _ = unpack_stabilizer_registers(registers)
    circ = get_empty_stabilizer_circuit(registers)

    # Unflagged recovery
    circ += unflagged_recovery(registers, reset=reset)

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

def full_recovery_XIXZZ(registers, reset=True):

    # Create a circuit
    qbReg, anReg, clReg, _ = unpack_stabilizer_registers(registers)
    circ = get_empty_stabilizer_circuit(registers)

    # Unflagged recovery
    circ += unflagged_recovery(registers, reset=reset)

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

def full_recovery_ZXIXZ(registers, reset=True):

    # Create a circuit
    qbReg, anReg, clReg, _ = unpack_stabilizer_registers(registers)
    circ = get_empty_stabilizer_circuit(registers)

    # Unflagged recovery
    circ += unflagged_recovery(registers, reset=reset)

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


# %% Testing of all functions above

n_cycles=1
# Define our registers (Maybe to be written as function?)
qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
cr = ClassicalRegister(5, 'syndrome_bit') # The typical register
#cr = get_classical_register(n_cycles) # Advanced list of registers
readout = ClassicalRegister(5, 'readout')

registers = [qb, an, cr, readout] # Pack them together
circ = get_empty_stabilizer_circuit(registers)

# Get the complete circuit
circ += get_full_stabilizer_circuit(registers,
    n_cycles=1,
    reset=True,
    recovery=True,
    flag=True,
    )

# Run it
n_shots = 2000
results = execute(
    circ,  
    Aer.get_backend('qasm_simulator'),
    noise_model=None,
    shots=n_shots
).result()

# Analyze results
from qiskit.quantum_info import state_fidelity
from qiskit.visualization import plot_histogram
logical = logical_states()
sv_post_encoding = results.data()['snapshots']['statevector']['stabilizer_0'][0]
fid = 0
for i in range(10):
    sv_post_encoding = results.data()['snapshots']['statevector']['stabilizer_0'][i]

    log0 = logical[0][np.arange(128,step=4)]
    sv_test = sv_post_encoding[0:32]
    fid += state_fidelity(log0, sv_test)

print('Average fidelity:')
print(fid/10)

# Plot results
counts = results.get_counts()
plot_histogram( counts )
#circuit.draw(output='mpl')

# %%
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