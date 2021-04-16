# This file contains all necessary functions for compiling and running the
# [[5,1,3]] error correction code, both with or without flagging. Most functions
# take optional arguments for whether to perform recovery, use flags, or reset
# the ancilla qubit. An example of how to use the functions is shown at the
# bottom.
#
# TODO:
#   The advanced registers currently be used for recovery. Need to fix that,
#   but it has the problem of flag-measurement being in a separate register
#   and c_if can only be conditioned on one.

# %% Import modules
from IPython.display import display
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import (state_fidelity, Pauli)
import numpy as np
from qiskit import (QuantumCircuit,
                    QuantumRegister,
                    ClassicalRegister,
                    AncillaRegister,
                    execute,
                    Aer
                    )
from qiskit.providers.aer.extensions.snapshot_statevector import *
from qiskit.providers.aer.extensions.snapshot_density_matrix import *
from typing import List
# %% General functions


class StabilizerRegisters:

    def __init__(self,
                 qbReg=QuantumRegister(5, 'code_qubit'),
                 anReg=AncillaRegister(2, 'ancilla_qubit'),
                 clReg=ClassicalRegister(4, 'syndrome_bit'),
                 readout=ClassicalRegister(5, 'readout')
                 ):
        self.QubitRegister = qbReg
        self.AncillaRegister = anReg
        self.SyndromeRegister = clReg
        self.ReadoutRegister = readout


def get_full_stabilizer_circuit(registers, n_cycles=1,
                                reset=True, recovery=False, flag=True, **kwargs):
    """Returns the circuit for a full repeating stabilizer circuit, including encoding,
    n_cycles of repeated stabilizers (with optional flags and recovery) and final measurement.
    """

    # Unpack registers
    # qbReg, anReg, clReg, readout = registers
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    readout = registers.ReadoutRegister
    if not anReg.size == 2 and not anReg.size == 5:
        raise Exception('Ancilla register must be of size 2 or 5')

    # Define the circuit
    # circ = get_empty_stabilizer_circuit(registers)

    # Encode the state
    circ = encode_input_v2(registers)
    circ.snapshot('post_encoding', 'statevector')

    # Stabilizer
    circ += get_repeated_stabilization(registers, n_cycles=n_cycles,
                                       reset=reset, recovery=recovery, flag=flag, **kwargs)

    # Final readout
    circ.measure(qbReg, readout)
    circ.snapshot_statevector('post_measure')

    return circ


def get_repeated_stabilization(registers, n_cycles=1,
                               reset=True, recovery=False, 
                               flag=False, snapshot_type='density_matrix', 
                               include_barriers = True, **kwargs):
    """Generates a circuit for repeated stabilizers. Including recovery and
    fault tolerant flagged circuits of selected.

    Args:
        registers (Register): Register object containing all registers
        n_cycles (int, optional): Number of stabilizer circuits. Defaults to 1.
        reset (bool, optional): Whether or not to reset ancillas. Defaults to True.
        recovery (bool, optional): Whether or not to apply recovery operations. Defaults to False.
        flag (bool, optional): Whether or not to use the fault-tolerant flagged circuit. Defaults to True.
        snapshot_type (str, optional): Type of snapshot (None,'statevector' or 'density_matrix'). Defaults to 'statevector'.

    Returns:
        QuantumCircuit: The resulting circuit
    """

    circ = get_empty_stabilizer_circuit(registers)

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
                                               current_cycle=current_cycle,
                                               include_barriers=include_barriers,
                                               **kwargs
                                               )

        if snapshot_type:# TODO: Maybe a nice looking solution?
            if snapshot_type == 'density_matrix':
                if include_barriers:
                    circ.barrier()
                circ.append(Snapshot('stabilizer_' + str(current_cycle), snapshot_type, num_qubits=5), registers.QubitRegister)
                if include_barriers:
                    circ.barrier()
            elif snapshot_type == 'expectation_value':
                circ.save_expectation_value(Pauli('ZZZZZ'), registers.QubitRegister, 
                    label='exp_value_'+str(current_cycle))
            else:
                circ.snapshot('stabilizer_' + str(current_cycle), snapshot_type)

        #if snapshot_type:
            #if not isinstance(snapshot_type, list):
            #    snapshot_type = [snapshot_type]
            #        
            #add_snapshot_to_circuit(circ, snapshot_type, registers.QubitRegister)
    return circ

def add_snapshot_to_circuit(circ, snapshot_label=None, qubits=None, pauliop='ZZZZZ'):
    """Appends a snapshot to circuit, given a specific label.
    
    Args:
        circ: QuantumCircuit object to append snapshot to.
        snapshot_label (str): The label for snapshot. Can also be a list
            of strings. The label must be formatted as detailed below:
            '[type]_[conditional, optional]_[current_cycle]'
            type: The type of snapshot. Either 'exp' or 'dm'.
            conditional: If snapshot shoudl be conditional, add 'con'.
            current_cycle: The stabilizer cycle the snapshot belongs to.
            Example: 'dm_2' or 'exp_con_0'.
        qubits (list): List of qubits (or a register) to apply the snapshot to.
    """    

    if not isinstance(snapshot_label, list):
        snapshot_label = [snapshot_label]

    for snap in snapshot_label:
        # Decode the label
        label_keywords = snap.split('_')
        if label_keywords[1] == 'con':
            conditional = True
        else:
            conditional = False
        if label_keywords[0] == 'dm':
            circ.save_density_matrix(qubits, label=snap, conditional=conditional)
        elif label_keywords[0] == 'exp':
            circ.save_expectation_value(Pauli(pauliop), qubits, 
                label=snap, conditional=conditional)

    return
            



def get_empty_stabilizer_circuit(registers):
    """Create an empty qiskit circuit adapted for stabilizer circuits"""

    # Unpack registers
    # qbReg, anReg, clReg, readout = registers
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    readout = registers.ReadoutRegister

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


def encode_input(registers):
    """Encode the input into logical 0 and 1 for the [[5,1,3]] code. This
    assumes that the 0:th qubit is the original state |psi> = a|0> + b|1>
    """

    # Unpack registers
    qbReg = registers.QubitRegister
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


def encode_input_v2(registers, include_barriers = True):
    """Encode the input into logical 0 and 1 for the [[5,1,3]] code. This
    assumes that the 0:th qubit is the original state |psi> = a|0> + b|1>.

    Alternate version Basudha found on stackoverflow.
    """

    qbReg = registers.QubitRegister
    # circ = get_empty_stabilizer_circuit(registers)
    circ = QuantumCircuit(qbReg)

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
    if include_barriers:
        circ.barrier()
    return circ


def get_classical_register(n_cycles, reset=True, recovery=False, flag=True):
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
            [ClassicalRegister(4, 'syndrome_cycle_' + str(i) + '_step_' + str(j))
             for j in range(4)] for i in range(n_cycles)]

        # List of registers for each step in the flagged stabilizer cycle
        flag_register = [
            [ClassicalRegister(1, 'flag_cycle_' + str(i) + '_step_' + str(j))
             for j in range(4)] for i in range(n_cycles)]

        # List of registers for the single stabilizer run with flag
        ancilla_msmnt_register = [
            [ClassicalRegister(1, 'ancilla_cycle_' + str(i) + '_step_' + str(j))
             for j in range(4)] for i in range(n_cycles)]

        return [syndrome_register, flag_register, ancilla_msmnt_register]

    # TODO: Make this functional with flags? If necessary
    elif recovery and not reset:
        # A register of four bits per cycle
        syndrome_register = [
            [ClassicalRegister(5, 'syndrome_cycle_' + str(i) + '_step_' + str(j))
             for j in range(1)] for i in range(n_cycles)]
        return [syndrome_register]

    else:
        # A register of four bits per cycle
        syndrome_register = [
            [ClassicalRegister(4, 'syndrome_cycle_' + str(i) + '_step_' + str(j))
             for j in range(1)] for i in range(n_cycles)]
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
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)

    # X
    circ.h(qbReg[0])
    circ.h(anReg[1])
    circ.cz(anReg[1], qbReg[0])
    circ.h(qbReg[0])

    # Flag
    circ.h(anReg[0])  # Initialize in +
    circ.cz(anReg[0], anReg[1])  # Flag CX

    # Z
    circ.cz(anReg[1], qbReg[1])

    # Z
    circ.cz(anReg[1], qbReg[2])

    # Flag
    circ.cz(anReg[0], anReg[1])

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
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)

    # X
    circ.h(qbReg[1])
    circ.h(anReg[1])
    circ.cz(anReg[1], qbReg[1])
    circ.h(qbReg[1])

    # Flag
    circ.h(anReg[0])  # Initialize in +
    circ.cz(anReg[0], anReg[1])  # Flag CX

    # Z
    circ.cz(anReg[1], qbReg[2])

    # Z
    circ.cz(anReg[1], qbReg[3])

    # Flag
    circ.cz(anReg[0], anReg[1])

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
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)

    # X
    circ.h(qbReg[0])
    circ.h(anReg[1])
    circ.cz(anReg[1], qbReg[0])
    circ.h(qbReg[0])

    # Flag
    circ.h(anReg[0])  # Initialize in +
    circ.cz(anReg[0], anReg[1])  # Flag CX

    # X
    circ.h(qbReg[2])
    circ.cz(anReg[1], qbReg[2])
    circ.h(qbReg[2])

    # Z
    circ.cz(anReg[1], qbReg[3])

    # Flag
    circ.cz(anReg[0], anReg[1])

    # Z
    circ.cz(anReg[1], qbReg[4])
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
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)

    # Z
    circ.h(anReg[1])
    circ.cz(anReg[1], qbReg[0])

    # Flag
    circ.h(anReg[0])  # Initialize in +
    circ.cz(anReg[0], anReg[1])  # Flag CX

    # X
    circ.h(qbReg[1])
    circ.cz(anReg[1], qbReg[1])
    circ.h(qbReg[1])

    # X
    circ.h(qbReg[3])
    circ.cz(anReg[1], qbReg[3])
    circ.h(qbReg[3])

    # Flag
    circ.cz(anReg[0], anReg[1])

    # Z
    circ.cz(anReg[1], qbReg[4])
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
                               current_cycle=0, current_step=0, num_ancillas=None, 
                               include_barriers=True, pipeline=False):
    """Run all four stabilizers without flags, as well as an optional
    recovery. The input current_step is only relevant for flagged cycles, and
    should be set to 0 otherwise.

    NOTE: Maybe use a list if ancilla indices instead? E.g. ancillas = [1,2,1,2]
    Args:
        num_ancillas: Specifies how many ancillas to spread the measurements over
    """

    # Use the previous behavior if num_ancillas=None
    if not num_ancillas:
        if registers.AncillaRegister.size == 2:
            anQb_list = [registers.AncillaRegister[1]]*4
        elif registers.AncillaRegister.size >= 4:
            # I don't like this really, we don't use the flagged circuit anymore so it shouldn't get the 0 spot by default
            anQb_list = [registers.AncillaRegister[n] for n in [1, 2, 3, 4]]
        else:
            Warning("Ancilla reg too small (this should never happen)")
    else:
        anQb_list = [registers.AncillaRegister[n % num_ancillas]
                     for n in range(4)]

    if pipeline:
        stabilizer_list = [_pipeline_stabilizer_XZZXI,
                           _pipeline_stabilizer_IXZZX,
                           _pipeline_stabilizer_XIXZZ,
                           _pipeline_stabilizer_ZXIXZ]
    else:
        stabilizer_list = [_unflagged_stabilizer_XZZXI,
                           _unflagged_stabilizer_IXZZX,
                           _unflagged_stabilizer_XIXZZ,
                           _unflagged_stabilizer_ZXIXZ]

    # Create list of syndrome bits
    if isinstance(registers.SyndromeRegister, list):
        syn_reg = registers.SyndromeRegister[0][current_cycle][current_step]
        syn_bit_list = [syn_reg[n] for n in range(4)]
    else:
        syn_bit_list = [registers.SyndromeRegister[n] for n in range(4)]

    # Create circuit and run stabilizers
    circ = get_empty_stabilizer_circuit(registers)
    for i in range(4):
        circ += stabilizer_list[i](registers, anQb=anQb_list[i],
                                   syn_bit=syn_bit_list[i], reset=reset)
        if include_barriers:
            circ.barrier()
    # Add an extra measurement to the next syndrome register
    # TODO: Make this compatible with using more than 1 ancilla
    if recovery and not reset:
        if current_cycle < len(registers.SyndromeRegister[0])-1:
            circ.measure(anQb_list[-1],
                         registers.SyndromeRegister[0][current_cycle+1][current_step][4])
            if include_barriers:
                circ.barrier()

    # Recovery
    if recovery is True:
        circ.barrier()
        circ += unflagged_recovery(registers, reset, current_cycle)
        #if include_barriers:
        circ.barrier()
    return circ


def _unflagged_stabilizer_XZZXI(registers, anQb=None, syn_bit=None, reset=True):
    """Gives the circuit for running the regular XZZXI stabilizer without flag.
    The current_step input should be set to zero unless running flagged cycles.

    Args:
        registers (StabilizerRegister): Register object
        anQb (AncillaQubit, optional): Specifies the ancilla to use for the measurement. Defaults to None.
        reset (bool, optional): Whether to reset ancillas between measurements. Defaults to True.
        current_cycle (int, optional): [description]. Defaults to 0.
        current_step (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """

    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)
    # NOTE: Use the empty circuit below instead? It doesn't add the readout register
    # circ = QuantumCircuit(qbReg,anReg,clReg)
    if not anQb:
        if anReg.size == 2:
            anQb = anReg[1]
        else:
            anQb = anReg[1]

    # X
    circ.h(anQb)
    circ.h(qbReg[0])
    circ.cz(anQb, qbReg[0])
    circ.h(qbReg[0])

    # Z
    circ.cz(anQb, qbReg[1])

    # Z
    circ.cz(anQb, qbReg[2])

    # X
    circ.h(qbReg[3])
    circ.cz(anQb, qbReg[3])
    circ.h(qbReg[3])
    circ.h(anQb)

    # Measure
    circ.measure(anQb, syn_bit)

    # Reset
    if reset:
        circ.reset(anQb)

    return circ


def _unflagged_stabilizer_IXZZX(registers, anQb=None, syn_bit=None, reset=True):
    """Gives the circuit for running the regular IXZZX stabilizer without flag.
    The current_step input should be set to zero unless running flagged cycles.
    """

    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)
    if not anQb:
        if anReg.size == 2:
            anQb = anReg[1]
        else:
            anQb = anReg[2]

    # X
    circ.h(anQb)
    circ.h(qbReg[1])
    circ.cz(anQb, qbReg[1])
    circ.h(qbReg[1])

    # Z
    circ.cz(anQb, qbReg[2])

    # Z
    circ.cz(anQb, qbReg[3])

    # X
    circ.h(qbReg[4])
    circ.cz(anQb, qbReg[4])
    circ.h(qbReg[4])
    circ.h(anQb)

    # Measure
    circ.measure(anQb, syn_bit)

    # Reset
    if reset:
        circ.reset(anQb)

    return circ


def _unflagged_stabilizer_XIXZZ(registers, anQb=None, syn_bit=None, reset=True):
    """Gives the circuit for running the regular XIXZZ stabilizer without flag.
    The current_step input should be set to zero unless running flagged cycles.
    """
    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)

    if not anQb:
        if anReg.size == 2:
            anQb = anReg[1]
        else:
            anQb = anReg[3]

    # X
    circ.h(anQb)
    circ.h(qbReg[0])
    circ.cz(anQb, qbReg[0])
    circ.h(qbReg[0])

    # X
    circ.h(qbReg[2])
    circ.cz(anQb, qbReg[2])
    circ.h(qbReg[2])

    # Z
    circ.cz(anQb, qbReg[3])

    # Z
    circ.cz(anQb, qbReg[4])
    circ.h(anQb)

    # Measure
    circ.measure(anQb, syn_bit)

    # Reset
    if reset:
        circ.reset(anQb)

    return circ


def _unflagged_stabilizer_ZXIXZ(registers, anQb=None, syn_bit=None, reset=True):
    """Gives the circuit for running the regular ZXIXZ stabilizer without flag.
    The current_step input should be set to zero unless running flagged cycles.
    """
    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)

    if not anQb:
        if anReg.size == 2:
            anQb = anReg[1]
        else:
            anQb = anReg[4]

    # Z
    circ.h(anQb)
    circ.cz(anQb, qbReg[0])

    # X
    circ.h(qbReg[1])
    circ.cz(anQb, qbReg[1])
    circ.h(qbReg[1])

    # X
    circ.h(qbReg[3])
    circ.cz(anQb, qbReg[3])
    circ.h(qbReg[3])

    # Z
    circ.cz(anQb, qbReg[4])
    circ.h(anQb)

    # Measure
    circ.measure(anQb, syn_bit)

    # Reset
    if reset:
        circ.reset(anQb)

    return circ

# %% Pipelined stabilizers. To call these, use unflagged_stabilizer_cycle with
# pipeline=True, DO NOT use num_ancillas=2. This is due to the transpiler not
# handling it correctly, thus forcing this 'fix'.
def _pipeline_stabilizer_XZZXI(registers, anQb=None, syn_bit=None, reset=True):
    """Gives the circuit for running the regular XZZXI stabilizer in a pipelined
    scheme. Note that this assumes ancilla reset, as there currently is no
    lookup table to handle no-reset for this purpose.

    Args:
        registers (StabilizerRegister): Register object
        anQb (AncillaQubit, optional): Specifies the ancilla to use for the measurement. Defaults to None.
        reset (bool, optional): Whether to reset ancillas between measurements. Defaults to True.
        current_cycle (int, optional): [description]. Defaults to 0.
        current_step (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """

    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)
    # NOTE: Use the empty circuit below instead? It doesn't add the readout register
    # circ = QuantumCircuit(qbReg,anReg,clReg)
    if not anQb:
        if anReg.size == 2:
            anQb = anReg[1]
        else:
            anQb = anReg[1]

    # X
    circ.h(anQb)
    circ.h(qbReg[0])
    circ.cz(anQb, qbReg[0])
    circ.h(qbReg[0])

    # Z
    circ.cz(anQb, qbReg[1])

    # Z
    circ.cz(anQb, qbReg[2])

    # X
    circ.h(qbReg[3])
    circ.cz(anQb, qbReg[3])
    circ.h(qbReg[3])
    circ.h(anQb)

    # Measure and reset
    circ.swap(anReg[0], anReg[1])
    circ.measure(anReg[0], syn_bit)
    circ.reset(anReg[0])
    return circ

def _pipeline_stabilizer_IXZZX(registers, anQb=None, syn_bit=None, reset=True):
    """Gives the circuit for running the regular IXZZX stabilizer in a pipelined
    scheme. Note that this assumes ancilla reset, as there currently is no
    lookup table to handle no-reset for this purpose.
    """

    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)
    if not anQb:
        if anReg.size == 2:
            anQb = anReg[1]
        else:
            anQb = anReg[2]

    # X
    circ.h(anQb)
    circ.h(qbReg[1])
    circ.cz(anQb, qbReg[1])
    circ.h(qbReg[1])

    # Z
    circ.cz(anQb, qbReg[2])

    # Z
    circ.cz(anQb, qbReg[3])

    # X
    circ.h(qbReg[4])
    circ.cz(anQb, qbReg[4])
    circ.h(qbReg[4])
    circ.h(anQb)

    # Measure and reset
    circ.swap(anReg[0], anReg[1])
    circ.measure(anReg[0], syn_bit)
    circ.reset(anReg[0])
    return circ

def _pipeline_stabilizer_XIXZZ(registers, anQb=None, syn_bit=None, reset=True):
    """Gives the circuit for running the regular XIXZZ stabilizer in a pipelined
    scheme. Note that this assumes ancilla reset, as there currently is no
    lookup table to handle no-reset for this purpose.
    """
    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)

    if not anQb:
        if anReg.size == 2:
            anQb = anReg[1]
        else:
            anQb = anReg[3]

    # X
    circ.h(anQb)
    circ.h(qbReg[0])
    circ.cz(anQb, qbReg[0])
    circ.h(qbReg[0])

    # X
    circ.h(qbReg[2])
    circ.cz(anQb, qbReg[2])
    circ.h(qbReg[2])

    # Z
    circ.cz(anQb, qbReg[3])

    # Z
    circ.cz(anQb, qbReg[4])
    circ.h(anQb)

    # Measure and reset
    circ.swap(anReg[0], anReg[1])
    circ.measure(anReg[0], syn_bit)
    circ.reset(anReg[0])
    return circ

def _pipeline_stabilizer_ZXIXZ(registers, anQb=None, syn_bit=None, reset=True):
    """Gives the circuit for running the regular ZXIXZ stabilizer in a pipelined
    scheme. Note that this assumes ancilla reset, as there currently is no
    lookup table to handle no-reset for this purpose. 

    As this is the final stabilizer in a cycle, this one does not include a
    swap before measurements, as it is essentially useless.
    """
    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)

    if not anQb:
        if anReg.size == 2:
            anQb = anReg[1]
        else:
            anQb = anReg[4]

    # Z
    circ.h(anQb)
    circ.cz(anQb, qbReg[0])

    # X
    circ.h(qbReg[1])
    circ.cz(anQb, qbReg[1])
    circ.h(qbReg[1])

    # X
    circ.h(qbReg[3])
    circ.cz(anQb, qbReg[3])
    circ.h(qbReg[3])

    # Z
    circ.cz(anQb, qbReg[4])
    circ.h(anQb)

    # Measure and reset
    circ.measure(anReg[1], syn_bit)
    circ.reset(anReg[1])
    return circ

# %% All recoveries
def unflagged_recovery(registers, reset=True, current_cycle=0, current_step=0):
    """Lookup table for recovery from a
    single qubit error on code qubits"""

    # Create a circuit
    qbReg = registers.QubitRegister
    clReg = registers.SyndromeRegister
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

        circ.x(qbReg[2]).c_if(syndrome_reg, 16+15-1)
        circ.x(qbReg[3]).c_if(syndrome_reg, 16+15-2)
        circ.z(qbReg[0]).c_if(syndrome_reg, 16+15-3)
        circ.x(qbReg[4]).c_if(syndrome_reg, 16+15-4)
        circ.z(qbReg[3]).c_if(syndrome_reg, 16+15-5)
        circ.x(qbReg[3]).c_if(syndrome_reg, 16+15-5)
        circ.z(qbReg[1]).c_if(syndrome_reg, 16+15-6)
        circ.z(qbReg[3]).c_if(syndrome_reg, 16+15-7)
        circ.x(qbReg[0]).c_if(syndrome_reg, 16+15-8)
        circ.z(qbReg[1]).c_if(syndrome_reg, 16+15-9)
        circ.x(qbReg[1]).c_if(syndrome_reg, 16+15-9)
        circ.z(qbReg[4]).c_if(syndrome_reg, 16+15-10)
        circ.x(qbReg[4]).c_if(syndrome_reg, 16+15-10)
        circ.z(qbReg[0]).c_if(syndrome_reg, 16+15-11)
        circ.x(qbReg[0]).c_if(syndrome_reg, 16+15-11)
        circ.z(qbReg[2]).c_if(syndrome_reg, 16+15-12)
        circ.z(qbReg[2]).c_if(syndrome_reg, 16+15-13)
        circ.x(qbReg[2]).c_if(syndrome_reg, 16+15-13)
        circ.z(qbReg[4]).c_if(syndrome_reg, 16+15-14)
        circ.x(qbReg[1]).c_if(syndrome_reg, 16+15-15)

    return circ


def full_recovery_XZZXI(registers, reset=True, current_cycle=0, current_step=0):
    """Lookup table for recovery using a flagged ancilla with the XZZXI
    stabilizer. Note that it requires a single classical register and
    reset=True to correctly recover.
    """

    # Create a circuit
    qbReg = registers.QubitRegister
    clReg = registers.SyndromeRegister
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
    circ.x(qbReg[0]).c_if(clReg, 8+16)  # This seems equivalent with IZZXI
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
    qbReg = registers.QubitRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)

    # Unflagged recovery
    circ += unflagged_recovery(registers, reset, current_cycle, current_step)

    # Flagged recovery
    if isinstance(clReg, list):
        print("Classical register is a list, performing unflagged recovery")
        return circ
    circ.x(qbReg[1]).c_if(clReg, 1+16)
    circ.x(qbReg[1]).c_if(clReg, 2+16)
    circ.x(qbReg[2]).c_if(clReg, 2+16)
    circ.y(qbReg[3]).c_if(clReg, 3+16)
    circ.x(qbReg[4]).c_if(clReg, 3+16)
    circ.z(qbReg[3]).c_if(clReg, 5+16)
    circ.x(qbReg[4]).c_if(clReg, 5+16)
    circ.x(qbReg[0]).c_if(clReg, 6+16)
    circ.y(qbReg[4]).c_if(clReg, 6+16)
    circ.x(qbReg[3]).c_if(clReg, 10+16)
    circ.x(qbReg[4]).c_if(clReg, 10+16)
    circ.x(qbReg[4]).c_if(clReg, 12+16)

    return circ


def full_recovery_XIXZZ(registers, reset=True, current_cycle=0, current_step=2):
    """Lookup table for recovery using a flagged ancilla with the XIXZZ
    stabilizer. Note that it requires a single classical register and
    reset=True to correctly recover.
    """

    # Create a circuit
    qbReg = registers.QubitRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)

    # Unflagged recovery
    circ += unflagged_recovery(registers, reset, current_cycle, current_step)

    # Flagged recovery
    if isinstance(clReg, list):
        print("Classical register is a list, performing unflagged recovery")
        return circ
    circ.x(qbReg[1]).c_if(clReg, 2+16)
    circ.x(qbReg[2]).c_if(clReg, 2+16)
    circ.x(qbReg[3]).c_if(clReg, 4+16)
    circ.z(qbReg[4]).c_if(clReg, 4+16)
    circ.x(qbReg[0]).c_if(clReg, 8+16)  # Seems equivalent with IZZXI
    circ.x(qbReg[0]).c_if(clReg, 11+16)
    circ.x(qbReg[2]).c_if(clReg, 11+16)
    circ.x(qbReg[4]).c_if(clReg, 12+16)
    circ.z(qbReg[0]).c_if(clReg, 13+16)
    circ.z(qbReg[2]).c_if(clReg, 13+16)
    circ.x(qbReg[4]).c_if(clReg, 13+16)
    circ.x(qbReg[2]).c_if(clReg, 15+16)
    circ.x(qbReg[4]).c_if(clReg, 15+16)

    return circ


def full_recovery_ZXIXZ(registers, reset=True, current_cycle=0, current_step=3):
    """Lookup table for recovery using a flagged ancilla with the ZXIXZ
    stabilizer. Note that it requires a single classical register and
    reset=True to correctly recover.
    """
    # Create a circuit
    qbReg = registers.QubitRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)

    # Unflagged recovery
    circ += unflagged_recovery(registers, reset, current_cycle, current_step)

    # Flagged recovery
    if isinstance(clReg, list):
        print("Classical register is a list, performing unflagged recovery")
        return circ
    circ.x(qbReg[1]).c_if(clReg, 2+16)
    circ.x(qbReg[2]).c_if(clReg, 2+16)
    circ.x(qbReg[3]).c_if(clReg, 4+16)
    circ.z(qbReg[4]).c_if(clReg, 4+16)
    circ.x(qbReg[2]).c_if(clReg, 5+16)
    circ.x(qbReg[3]).c_if(clReg, 5+16)
    circ.x(qbReg[0]).c_if(clReg, 11+16)
    circ.x(qbReg[2]).c_if(clReg, 11+16)
    circ.z(qbReg[0]).c_if(clReg, 13+16)
    circ.z(qbReg[2]).c_if(clReg, 13+16)
    circ.x(qbReg[4]).c_if(clReg, 13+16)
    circ.x(qbReg[0]).c_if(clReg, 14+16)
    circ.z(qbReg[2]).c_if(clReg, 14+16)
    circ.z(qbReg[4]).c_if(clReg, 14+16)
    circ.x(qbReg[2]).c_if(clReg, 15+16)
    circ.x(qbReg[4]).c_if(clReg, 15+16)

    return circ

# %% Function used for internal testing


def logical_states(include_ancillas='front') -> List[List[float]]:
    """Returns the logical states for the [[5,1,3]] code.

    Args:
        include_ancillas (str/None, optional): Whether to append the ancillas by tensor product to the end. Defaults to True.

    Returns:
        List[List[float]]: List of both logical states
    """
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

    if include_ancillas:
        # Add two ancillas in |0>
        an0 = np.zeros(2**2)
        an0[0] = 1.0
        if include_ancillas == 'front':
            logical_1 = np.kron(logical_1, an0)
            logical_0 = np.kron(logical_0, an0)
        elif include_ancillas == 'back':
            logical_1 = np.kron(an0, logical_1)
            logical_0 = np.kron(an0, logical_0)

    return [logical_0, logical_1]


# %% Internal testing of functions above

if __name__ == "__main__":
    # The settings for our circuit
    n_cycles = 1
    reset = True
    recovery = False
    flag = False

    # Define our registers (Maybe to be written as function?)
    qb = QuantumRegister(5, 'code_qubit')
    an = AncillaRegister(2, 'ancilla_qubit')
    cr = ClassicalRegister(4, 'syndrome_bit')  # The typical register
    # cr = get_classical_register(n_cycles, flag) # Advanced list of registers
    readout = ClassicalRegister(5, 'readout')

    registers = StabilizerRegisters(qb, an, cr, readout)
    # registers = [qb, an, cr, readout] # Pack them together
    circ = get_empty_stabilizer_circuit(registers)

    # Get the complete circuit
    circ += get_full_stabilizer_circuit(registers,
                                        n_cycles=n_cycles,
                                        reset=reset,
                                        recovery=recovery,
                                        flag=flag,
                                        )

    display(circ.draw())
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
    sv_post_encoding = results.data(
    )['snapshots']['statevector']['stabilizer_0'][0]
    fid = 0
    for i in range(10):
        sv_post_encoding = results.data(
        )['snapshots']['statevector']['stabilizer_0'][i]

        log0 = logical[0][np.arange(128, step=4)]
        sv_test = sv_post_encoding[0:32]
        fid += state_fidelity(log0, sv_test)

    print('Average fidelity across 10 shots:')
    print(fid/10)

    # Plot results
    counts = results.get_counts()
    plot_histogram(counts)

