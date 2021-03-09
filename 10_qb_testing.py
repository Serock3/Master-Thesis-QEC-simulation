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

# %% General functions
def get_full_stabilizer_circuit(registers, n_cycles=1,
        reset=True, recovery=False, flag=True):
    """Returns the circuit for a full stabilizer circuit, including encoding, 
    stabilizers (with optional flags and recovery) and final measurement.
    """

    # Unpack registers
    qbReg, anReg, clReg, readout = registers
    # if not anReg.size == 2 and not anReg.size == 5:
    #     raise Exception('Ancilla register must be of size 2 or 5')

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
                recovery=recovery,
                current_cycle=current_cycle
            )
        circ.snapshot_statevector('stabilizer_' + str(current_cycle))

    # Final readout
    circ.measure(qbReg, readout)
    circ.snapshot_statevector('post_measure')

    return circ

def get_empty_stabilizer_circuit(registers):
    """Create an empty qiskit circuit adapted for stabilizer circuits"""

    # Unpack registers
    qbReg, anReg, clReg, readout = registers

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
    in all stabilizer functions (stabilizer.py). Kind of obsolete
    """
    qbReg, anReg, clReg, readout = registers
    return qbReg, anReg, clReg, readout

def encode_input(registers):
    """Encode the input into logical 0 and 1 for the [[5,1,3]] code. This
    assumes that the 0:th qubit is the original state |psi> = a|0> + b|1>
    """

    qbReg, _, _, _ = registers
    circ = get_empty_stabilizer_circuit(registers)

    circ.z(qbReg[0])
    circ.h(qbReg[1])
    circ.h(qbReg[2])
    circ.h(qbReg[3])
    circ.h(qbReg[4])

    circ.h(qbReg[0])
    circ.cz(qbReg[0], qbReg[1])
    circ.cz(qbReg[0], qbReg[2])
    circ.cz(qbReg[0], qbReg[3])
    circ.cz(qbReg[0], qbReg[4])
    circ.h(qbReg[0])
    
    circ.cz(qbReg[0], qbReg[1])
    circ.cz(qbReg[2], qbReg[3])
    circ.cz(qbReg[1], qbReg[2])
    circ.cz(qbReg[3], qbReg[4])
    circ.cz(qbReg[0], qbReg[4])

    return circ

def get_classical_register(n_cycles, flag=True):
    """Generate lists of classical registers for storing all measurement data.
    
    The function accepts the flag bool to determine whether to create 
    registers for additional flag measurements. The names are as follows:
    
    syndrome_register: The standard four unflagged stabilizer measurements.
        Without flags, its dimensions are [n_cycles][1][4] and with flags they
        are [n_cycles][current_step][4] where current step represents where
        in the full flagged stabilizer cycle measurements were made (0-3).

    flag_register: All flag measurements, only appears if flag is set to True.
        It has the dimensions [n_cycles][current_step][1].

    ancilla_msmnt_register: Contains the extra single ancilla measurement
        performed along with a flag measurement in the flagged cycle. Only
        appears if flag is set to True. Dimensions are
        [n_cycles][current_step][1]
    """

    if flag:
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
    
    else:
        # A register of four bits per cycle
        syndrome_register = [
            [ ClassicalRegister( 4, 'syndrome_cycle_' +str(i) +'_step_' +str(j)) 
            for j in range(1) ] for i in range(n_cycles) ]
        return [syndrome_register]

# %% All flagged stabilizers
def flagged_stabilizer_cycle(registers, reset=True, recovery=True,
        current_cycle=0):
    """Runs one cycle of the [[5,1,3]] code with two ancillas as described in
    the article by Chao & Reichardt (2017).
    This includes the (optional) recovery from any detected errors.
    
    Currently, it requires reset=True to appropriately
    correct errors, and cannot perform recovery with the advanced registers
    """

    # Create a circuit
    qbReg, anReg, clReg, _ = registers
    circ = get_empty_stabilizer_circuit(registers)

    # === Step 1: XZZXI ===
    circ += _flagged_stabilizer_XZZXI(registers, reset, current_cycle)
    circ += unflagged_stabilizer_cycle(registers, reset, recovery=False,
        current_cycle=current_cycle, current_step=0)
    if recovery:
        circ += full_recovery_XZZXI(registers, reset, current_cycle, 0)

    # === Step 2: IXZZX ===
    circ += _flagged_stabilizer_IXZZX(registers, reset, current_cycle)
    circ += unflagged_stabilizer_cycle(registers, reset, recovery=False,
        current_cycle=current_cycle, current_step=1)
    if recovery:
        circ += full_recovery_IXZZX(registers, reset, current_cycle, 1)

    # === Step 3: XIXZZ ===
    circ += _flagged_stabilizer_XIXZZ(registers, reset, current_cycle)
    circ += unflagged_stabilizer_cycle(registers, reset, recovery=False,
        current_cycle=current_cycle, current_step=2)
    if recovery:
        circ += full_recovery_XIXZZ(registers, reset, current_cycle, 2)
    
    # === Step 4: ZXIXZ ===
    circ += _flagged_stabilizer_ZXIXZ(registers, reset, current_cycle)
    circ += unflagged_stabilizer_cycle(registers, reset, recovery=False,
        current_cycle=current_cycle, current_step=3)
    if recovery:
        circ += full_recovery_ZXIXZ(registers, reset, current_cycle, 3)

    return circ


def _flagged_stabilizer_XZZXI(registers, reset=True, current_cycle=0):
    """Gives the circuit for running the XZZXI stabilizer with a flag ancilla,
    connected by a cz gate after the first and before the last operation.
    """

    # Create a circuit
    qbReg, anReg, clReg, _ = registers
    circ = get_empty_stabilizer_circuit(registers)

    # X
    circ.h( qbReg[0] )
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
    """Gives the circuit for running the IXZZX stabilizer with a flag ancilla,
    connected by a cz gate after the first and before the last operation.
    """

    # Create a circuit
    qbReg, anReg, clReg, _ = registers
    circ = get_empty_stabilizer_circuit(registers)

    # X
    circ.h( qbReg[1] )
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
    """Gives the circuit for running the XIXZZ stabilizer with a flag ancilla,
    connected by a cz gate after the first and before the last operation.
    """

    # Create a circuit
    qbReg, anReg, clReg, _ = registers
    circ = get_empty_stabilizer_circuit(registers)

    # X
    circ.h( qbReg[0] )
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
    """Gives the circuit for running the ZXIXZ stabilizer with a flag ancilla,
    connected by a cz gate after the first and before the last operation.
    """

    # Create a circuit
    qbReg, anReg, clReg, _ = registers
    circ = get_empty_stabilizer_circuit(registers)

    # Z
    circ.h( anReg[1] )
    circ.cz( anReg[1], qbReg[0] )

    # Flag
    circ.h( anReg[0] ) # Initialize in +
    circ.cz( anReg[0], anReg[1] ) # Flag CX  

    # X
    circ.h( qbReg[1] )
    circ.cz( anReg[1], qbReg[1] )
    circ.h( qbReg[1] )

    # X
    circ.h( qbReg[3] )
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
    recovery. The input current_step is only relevant for flagged cycles, and
    should be set to 0 otherwise.
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
        circ += unflagged_recovery(registers, reset, current_cycle)
    
    return circ

def _unflagged_stabilizer_XZZXI(registers, reset=True,
        current_cycle=0, current_step=0):
    """Gives the circuit for running the regular XZZXI stabilizer without flag.
    The current_step input should be set to zero unless running flagged cycles.
    """

    # Create a circuit
    qbReg, anReg, clReg, _ = registers
    circ = get_empty_stabilizer_circuit(registers)
    if anReg.size == 2:
        anQb = anReg[1]
    else:
        anQb = anReg[1]

    # X
    circ.h( anQb )
    circ.h( qbReg[0] )
    circ.cz( anQb, qbReg[0] )
    circ.h( qbReg[0] )       

    # Z
    circ.cz(anQb, qbReg[1])

    # Z
    circ.cz( anQb, qbReg[2] )

    # X
    circ.h(qbReg[3])
    circ.cz(anQb, qbReg[3])
    circ.h(qbReg[3])
    circ.h(anQb)

    # Measure
    if isinstance(clReg, list):
        syndrome_reg = clReg[0]
        circ.measure(anQb, syndrome_reg[current_cycle][current_step][0])
    else:
        circ.measure(anQb, clReg[0])
    
    # Reset
    if reset:
        circ.reset(anQb)

    return circ

def _unflagged_stabilizer_IXZZX(registers, reset=True,
        current_cycle=0, current_step=0):
    """Gives the circuit for running the regular IXZZX stabilizer without flag.
    The current_step input should be set to zero unless running flagged cycles.
    """

    # Create a circuit
    qbReg, anReg, clReg, _ = registers
    circ = get_empty_stabilizer_circuit(registers)
    if anReg.size == 2:
        anQb = anReg[1]
    else:
        anQb = anReg[2]

    # X
    circ.h(anQb)
    circ.h( qbReg[1] )
    circ.cz(anQb, qbReg[1] )
    circ.h( qbReg[1] )   

    # Z
    circ.cz(anQb, qbReg[2] )

    # Z
    circ.cz(anQb, qbReg[3] ) 

    # X
    circ.h(qbReg[4])
    circ.cz(anQb, qbReg[4])
    circ.h(qbReg[4])
    circ.h(anQb)

    # Measure
    if isinstance(clReg, list):
        syndrome_reg = clReg[0]
        circ.measure(anQb, syndrome_reg[current_cycle][current_step][1])
    else:
        circ.measure(anQb, clReg[1])
    
    # Reset
    if reset:
        circ.reset(anQb)

    return circ

def _unflagged_stabilizer_XIXZZ(registers, reset=True,
        current_cycle=0, current_step=0):
    """Gives the circuit for running the regular XIXZZ stabilizer without flag.
    The current_step input should be set to zero unless running flagged cycles.
    """
    # Create a circuit
    qbReg, anReg, clReg, _ = registers
    circ = get_empty_stabilizer_circuit(registers)
    if anReg.size == 2:
        anQb = anReg[1]
    else:
        anQb = anReg[3]

    # X
    circ.h(anQb)
    circ.h( qbReg[0] )
    circ.cz(anQb, qbReg[0] )
    circ.h( qbReg[0] )      

    # X
    circ.h( qbReg[2] )
    circ.cz(anQb, qbReg[2] )
    circ.h( qbReg[2] )

    # Z
    circ.cz(anQb, qbReg[3] )

    # Z
    circ.cz(anQb, qbReg[4] )
    circ.h(anQb)

    # Measure
    if isinstance(clReg, list):
        syndrome_reg = clReg[0]
        circ.measure(anQb, syndrome_reg[current_cycle][current_step][2])
    else:
        circ.measure(anQb, clReg[2])
    
    # Reset
    if reset:
        circ.reset(anQb)

    return circ

def _unflagged_stabilizer_ZXIXZ(registers, reset=True,
        current_cycle=0, current_step=0):
    """Gives the circuit for running the regular ZXIXZ stabilizer without flag.
    The current_step input should be set to zero unless running flagged cycles.
    """
    # Create a circuit
    qbReg, anReg, clReg, _ = registers
    circ = get_empty_stabilizer_circuit(registers)
    if anReg.size == 2:
        anQb = anReg[1]
    else:
        anQb = anReg[4]

    # Z
    circ.h(anQb)
    circ.cz(anQb, qbReg[0] )

    # X
    circ.h( qbReg[1] )
    circ.cz(anQb, qbReg[1] )
    circ.h( qbReg[1] )

    # X
    circ.h( qbReg[3] )
    circ.cz(anQb, qbReg[3] )
    circ.h( qbReg[3] )
    
    # Z
    circ.cz(anQb, qbReg[4] )
    circ.h(anQb)

   # Measure
    if isinstance(clReg, list):
        syndrome_reg = clReg[0]
        circ.measure(anQb, syndrome_reg[current_cycle][current_step][3])
    else:
        circ.measure(anQb, clReg[3])
    
    # Reset
    if reset:
        circ.reset(anQb)
    
    return circ


# %% All recoveries
def unflagged_recovery(registers, reset=True, current_cycle=0, current_step=0):
    """Lookup table for recovery from a
    single qubit error on code qubits"""

    # Create a circuit
    qbReg, anReg, clReg, _ = registers
    circ = get_empty_stabilizer_circuit(registers)

    # Unpack registers
    if isinstance(clReg, list):
        syndrome_reg = clReg[0][current_cycle][current_step]
    else:
        syndrome_reg = clReg

    # If the ancilla is reset to |0> between measurements
    if reset:
        circ.x(qbReg[1]).c_if(syndrome_reg, 1)
        circ.z(qbReg[4]).c_if(syndrome_reg, 2)
        circ.x(qbReg[2]).c_if(syndrome_reg, 3)
        circ.z(qbReg[2]).c_if(syndrome_reg, 4)
        circ.z(qbReg[0]).c_if(syndrome_reg, 5)
        circ.x(qbReg[3]).c_if(syndrome_reg, 6)
        circ.x(qbReg[2]).c_if(syndrome_reg, 7)
        circ.z(qbReg[2]).c_if(syndrome_reg, 7)
        circ.x(qbReg[0]).c_if(syndrome_reg, 8)
        circ.z(qbReg[3]).c_if(syndrome_reg, 9)
        circ.z(qbReg[1]).c_if(syndrome_reg, 10)
        circ.x(qbReg[1]).c_if(syndrome_reg, 11)
        circ.z(qbReg[1]).c_if(syndrome_reg, 11)
        circ.x(qbReg[4]).c_if(syndrome_reg, 12)
        circ.x(qbReg[0]).c_if(syndrome_reg, 13)
        circ.z(qbReg[0]).c_if(syndrome_reg, 13)
        circ.x(qbReg[4]).c_if(syndrome_reg, 14)
        circ.z(qbReg[4]).c_if(syndrome_reg, 14)
        circ.x(qbReg[3]).c_if(syndrome_reg, 15)
        circ.z(qbReg[3]).c_if(syndrome_reg, 15)

    # If the ancilla is NOT reset between measurements
    else:
        circ.x(qbReg[2]).c_if(syndrome_reg, 1)
        circ.x(qbReg[3]).c_if(syndrome_reg, 2)
        circ.z(qbReg[0]).c_if(syndrome_reg, 3)
        circ.x(qbReg[4]).c_if(syndrome_reg, 4)
        circ.z(qbReg[3]).c_if(syndrome_reg, 5)
        circ.x(qbReg[3]).c_if(syndrome_reg, 5)
        circ.z(qbReg[1]).c_if(syndrome_reg, 6)
        circ.z(qbReg[3]).c_if(syndrome_reg, 7)
        circ.x(qbReg[0]).c_if(syndrome_reg, 8)
        circ.z(qbReg[1]).c_if(syndrome_reg, 9)
        circ.x(qbReg[1]).c_if(syndrome_reg, 9)
        circ.z(qbReg[4]).c_if(syndrome_reg, 10)
        circ.x(qbReg[4]).c_if(syndrome_reg, 10)
        circ.z(qbReg[0]).c_if(syndrome_reg, 11)
        circ.x(qbReg[0]).c_if(syndrome_reg, 11)
        circ.z(qbReg[2]).c_if(syndrome_reg, 12)
        circ.z(qbReg[2]).c_if(syndrome_reg, 13)
        circ.x(qbReg[2]).c_if(syndrome_reg, 13)
        circ.z(qbReg[4]).c_if(syndrome_reg, 14)
        circ.x(qbReg[1]).c_if(syndrome_reg, 15)

    return circ

def full_recovery_XZZXI(registers, reset=True, current_cycle=0, current_step=0):
    """Lookup table for recovery using a flagged ancilla with the XZZXI
    stabilizer. Note that it requires a single classical register and
    reset=True to correctly recover.
    """

    # Create a circuit
    qbReg, anReg, clReg, _ = registers
    circ = get_empty_stabilizer_circuit(registers)

    # Unflagged recovery
    circ += unflagged_recovery(registers, reset, current_cycle, current_step)

    # Flagged recovery
    if isinstance(clReg, list):
        print("Classical register is a list, performing unflagged recovery")
        return circ
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

def full_recovery_IXZZX(registers, reset=True, current_cycle=0, current_step=1):
    """Lookup table for recovery using a flagged ancilla with the IXZZX
    stabilizer. Note that it requires a single classical register and
    reset=True to correctly recover.
    """

    # Create a circuit
    qbReg, anReg, clReg, _ = registers
    circ = get_empty_stabilizer_circuit(registers)

    # Unflagged recovery
    circ += unflagged_recovery(registers, reset, current_cycle, current_step)

    # Flagged recovery
    if isinstance(clReg, list):
        print("Classical register is a list, performing unflagged recovery")
        return circ
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

def full_recovery_XIXZZ(registers, reset=True, current_cycle=0, current_step=2):
    """Lookup table for recovery using a flagged ancilla with the XIXZZ
    stabilizer. Note that it requires a single classical register and
    reset=True to correctly recover.
    """

    # Create a circuit
    qbReg, anReg, clReg, _ = registers
    circ = get_empty_stabilizer_circuit(registers)

    # Unflagged recovery
    circ += unflagged_recovery(registers, reset, current_cycle, current_step)

    # Flagged recovery
    if isinstance(clReg, list):
        print("Classical register is a list, performing unflagged recovery")
        return circ
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

def full_recovery_ZXIXZ(registers, reset=True, current_cycle=0, current_step=3):
    """Lookup table for recovery using a flagged ancilla with the ZXIXZ
    stabilizer. Note that it requires a single classical register and
    reset=True to correctly recover.
    """
    # Create a circuit
    qbReg, anReg, clReg, _ = registers
    circ = get_empty_stabilizer_circuit(registers)

    # Unflagged recovery
    circ += unflagged_recovery(registers, reset, current_cycle, current_step)

    # Flagged recovery
    if isinstance(clReg, list):
        print("Classical register is a list, performing unflagged recovery")
        return circ
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


# %% Function used for internal testing
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

# %% Internal testing of functions above
from qiskit.quantum_info import state_fidelity
from qiskit.visualization import plot_histogram

if __name__ == "__main__":
    # The settings for our circuit
    n_cycles = 2
    reset = True
    recovery = False
    flag = True

    # Define our registers (Maybe to be written as function?)
    qb = QuantumRegister(5, 'code_qubit')
    an = AncillaRegister(4, 'ancilla_qubit')
    #cr = ClassicalRegister(5, 'syndrome_bit') # The typical register
    cr = get_classical_register(n_cycles, flag) # Advanced list of registers
    readout = ClassicalRegister(5, 'readout')

    registers = [qb, an, cr, readout] # Pack them together
    circ = get_empty_stabilizer_circuit(registers)

    # Get the complete circuit

    circ += get_full_stabilizer_circuit(registers,
        n_cycles=n_cycles,
        reset=reset,
        recovery=recovery,
        flag=flag,
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
    logical = logical_states()
    sv_post_encoding = results.data()['snapshots']['statevector']['stabilizer_0'][0]
    fid = 0
    for i in range(10):
        sv_post_encoding = results.data()['snapshots']['statevector']['stabilizer_0'][i]

        log0 = logical[0][np.arange(128,step=4)]
        sv_test = sv_post_encoding[0:32]
        fid += state_fidelity(log0, sv_test)

    print('Average fidelity across 10 shots:')
    print(fid/10)

    # Plot results
    counts = results.get_counts()
    plot_histogram(counts)
    

# %% 10 qb TRANSPILER testing
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager, CouplingMap, Layout
from qiskit.visualization import plot_circuit_layout
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.circuit.library.standard_gates import iSwapGate, SwapGate, SGate, CZGate
from qiskit import QuantumCircuit, QuantumRegister
import warnings

def shortest_transpile_from_distribution(circuit, repeats=40, print_depths=True, **kwargs):
    depth = 10000
    for i in range(repeats):
        with warnings.catch_warnings():  # sabre causes deprication warning, this will ignore them
            warnings.simplefilter("ignore")
            transpiled_circuit_tmp = transpile(circuit, **kwargs)
        if print_depths:
            print('depth: ', transpiled_circuit_tmp.depth())
        if transpiled_circuit_tmp.depth() < depth:
            depth = transpiled_circuit_tmp.depth()
            transpiled_circuit = transpiled_circuit_tmp
    return transpiled_circuit


basis_gates = ['id', 'u1', 'u2', 'u3', 'iswap', 'cz']
couplinglist = [[0,4], [0,1], [1,4], [1,5], [1,2], [2,5], [2,6], [2,3],
    [3,6], [4,7], [4,5], [5,7], [5,8], [5,6], [6,8], [7,8], [7,9], [8,9]]
reverse_couplinglist = [[y, x] for [x, y] in couplinglist]
coupling_map = CouplingMap(
    couplinglist=couplinglist+reverse_couplinglist, description='A hexagoal 7qb code with two ancillas')

# Dict with device properties of the WAQCT QC to be used for transpilation.
WAQCT_device_properties = {
    "basis_gates": basis_gates, "coupling_map": coupling_map}

def _add_custom_device_equivalences():
    """ Ads custom gate equivalences to the SessionEquivalenceLibrary for transpilation
    NOTE: One needs to be run once!
    """
    print('Adding custom device equivalences')
    q = QuantumRegister(2, 'q')
    def_swap = QuantumCircuit(q)
    for inst, qargs, cargs in [
            (iSwapGate(), [q[0], q[1]], []),
            (CZGate(), [q[0], q[1]], []),
            (SGate().inverse(), [q[1]], []),
            (SGate().inverse(), [q[0]], [])
    ]:
        def_swap.append(inst, qargs, cargs)
    SessionEquivalenceLibrary.add_equivalence(SwapGate(), def_swap)

# TODO: Curently not functioning? 
# This function will automatically run the first time you import this file
_add_custom_device_equivalences()


# %% Test transpiler

# The settings for our circuit
n_cycles = 1
reset = False
recovery = False
flag = False

# Define our registers (Maybe to be written as function?)
qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(5, 'ancilla_qubit')
cr = ClassicalRegister(5, 'syndrome_bit') # The typical register
#cr = get_classical_register(n_cycles, flag) # Advanced list of registers
readout = ClassicalRegister(5, 'readout')

registers = [qb, an, cr, readout] # Pack them together
circ = get_empty_stabilizer_circuit(registers)

# Get the complete circuit
circ += get_full_stabilizer_circuit(registers,
    n_cycles=n_cycles,
    reset=reset,
    recovery=recovery,
    flag=flag,
)

print('Starting transpilation')
# Transpilation
routing_method = 'sabre'  # basic lookahead stochastic sabre
initial_layout = None  # Overwriting the above layout
layout_method = 'sabre'  # trivial 'dense', 'noise_adaptive' sabre
translation_method = None  # 'unroller',  translator , synthesis
repeats = 10
optimization_level = 1
circ_t = shortest_transpile_from_distribution(
    circ,
    print_depths=True,
    repeats=repeats,
    routing_method=routing_method,
    initial_layout=initial_layout,
    layout_method=layout_method,
    translation_method=translation_method,
    optimization_level=optimization_level,
    **WAQCT_device_properties
)
print('Drawing output')
circ_t.draw(output='mpl')
# %%
print('Final depth diamond = ', circ_t.depth())
print('Final gates diamond = ', circ_t.count_ops())
print('Final depth diamond = ', circ.depth())
print('Final gates diamond = ', circ.count_ops())

# %%
# Run it
n_shots = 2000
results = execute(
    circ,  
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

    log0 = logical[0][np.arange(128,step=4)]
    sv_test = sv_post_encoding[0:32]
    fid += state_fidelity(log0, sv_test)

print('Average fidelity across 10 shots:')
print(fid/10)

# Plot results
counts = results.get_counts()
plot_histogram(counts)