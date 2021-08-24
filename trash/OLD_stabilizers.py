"""This is an old version of the 'stabilizers.py' file, which contains removed
functions regarding flagged circuits."""

# This file contains all necessary functions for compiling and running the
# [[5,1,3]] error correction code, both with or without flagging. Most functions
# take optional arguments for whether to perform recovery, use flags, or reset
# the ancilla qubit. An example of how to use the functions is shown at the
# bottom.
#

# %% Import modules
from IPython.display import display
from qiskit.quantum_info import (state_fidelity, Pauli)
import numpy as np
from qiskit import (QuantumCircuit,
                    QuantumRegister,
                    ClassicalRegister,
                    AncillaRegister,
                    execute,
                    Aer
                    )
from qiskit.providers.aer.library import set_density_matrix, set_statevector
from qiskit.circuit import measure, reset
from qiskit.providers.aer.library import save_density_matrix, save_expectation_value
from typing import List
import warnings
from qiskit.circuit.library import XGate, ZGate
from qiskit.quantum_info.states.statevector import Statevector

from .splitting_circuits import add_split_marker

# Disable error which gives false positives in this file
# pylint: disable=E1101
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

def get_registers(conditional = False, final_measure = False, n_cycles = 0, reset = True, recovery = True, flag = False, include_fifth_stabilizer = False):
    qbReg = QuantumRegister(5, 'code_qubit')
    anReg = AncillaRegister(2, 'ancilla_qubit')

    if conditional:
        # Advanced list of registers
        crReg = get_classical_register(n_cycles, reset, recovery, flag, include_fifth_stabilizer)
    else:
        crReg = ClassicalRegister(
            4, 'syndrome_bit')  # The typical register
    if final_measure:
        readout = ClassicalRegister(5, 'readout')
        registers = StabilizerRegisters(qbReg, anReg, crReg, readout)
    else:
        registers = StabilizerRegisters(qbReg, anReg, crReg, None)
    return registers

def get_full_stabilizer_circuit(registers=None, n_cycles=1,
                                reset=True, recovery=False, flag=False,
                                snapshot_type='density_matrix',
                                include_barriers=True, conditional=True,
                                initial_state=0, encoding=True, theta=0, phi=0,
                                generator_snapshot=True, pauliop='ZZZZZ', device=None,
                                simulator_type='density_matrix', final_measure=True, **kwargs):
    """Returns the circuit for a full repeating stabilizer circuit, including encoding,
    n_cycles of repeated stabilizers (with optional flags and recovery) and final measurement.
    """
    
    include_fifth_stabilizer=False
    if 'include_fifth_stabilizer' in kwargs:
        include_fifth_stabilizer = kwargs['include_fifth_stabilizer']

    # TODO: Make this compatible with other codes?
    if registers is None:
        registers = get_registers(conditional, final_measure, n_cycles, reset, recovery, flag, include_fifth_stabilizer)

    if not registers.AncillaRegister.size == 2 and not registers.AncillaRegister.size == 5:
        raise Exception('Ancilla register must be of size 2 or 5')

    # Define the circuit
    circ = get_empty_stabilizer_circuit(registers)

    if encoding:
        # TODO: Using rx and rz messes with the transpiler. Make a better fix
        if initial_state != 0:
            circ.rx(theta, registers.QubitRegister[0])
            circ.rz(phi, registers.QubitRegister[0])
        circ.compose(encode_input_v2(registers), inplace=True)
    else:
        if simulator_type == 'statevector':
            circ.set_statevector(get_encoded_state(theta=theta, phi=phi))
        else:
            circ.set_density_matrix(get_encoded_state(theta=theta, phi=phi))

    add_snapshot_to_circuit(circ, snapshot_type=snapshot_type, current_cycle=0, qubits=registers.QubitRegister,
                            conditional=conditional, pauliop=pauliop,
                            include_barriers=include_barriers)

    # Stabilizer
    circ.compose(get_repeated_stabilization(registers, n_cycles=n_cycles,
                                            reset=reset, recovery=recovery, flag=flag,
                                            snapshot_type=snapshot_type,
                                            conditional=conditional,
                                            generator_snapshot=generator_snapshot,
                                            include_barriers=include_barriers,
                                            pauliop=pauliop, device=device,
                                            **kwargs), inplace=True)

    # Final readout
    if final_measure:
        circ.measure(registers.QubitRegister, registers.ReadoutRegister)
    return circ


def get_repeated_stabilization(registers, n_cycles=1,
                               reset=True, recovery=False,
                               flag=False, snapshot_type='density_matrix',
                               include_barriers=True, conditional=True, generator_snapshot=True,
                               pauliop='ZZZZZ', device=None, idle_delay='after', 
                               idle_snapshots=1, split_cycles=False,  **kwargs):
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
        if idle_delay == 'before':
            add_delay_marker(circ, registers, idle_snapshots, snapshot_type,
                             qubits=registers.QubitRegister, conditional=conditional,
                             pauliop=pauliop, include_barriers=include_barriers)
                             
        if flag is True:
            circ.compose(flagged_stabilizer_cycle(registers,
                                                  reset=reset,
                                                  recovery=recovery,
                                                  current_cycle=current_cycle,
                                                  ), inplace=True)
        elif device == 'double_diamond':
            circ.compose(transpiled_dd_cycle(registers,
                                             reset=reset,
                                             recovery=recovery,
                                             current_cycle=current_cycle,
                                             ), inplace=True)
        else:
            circ.compose(unflagged_stabilizer_cycle(registers,
                                                    reset=reset,
                                                    recovery=recovery,
                                                    current_cycle=current_cycle,
                                                    include_barriers=include_barriers,
                                                    snapshot=generator_snapshot,
                                                    snapshot_type=snapshot_type,
                                                    conditional=conditional,
                                                    pauliop=pauliop,
                                                    **kwargs
                                                    ), inplace=True)

        add_snapshot_to_circuit(circ, snapshot_type,
                                qubits=registers.QubitRegister, conditional=conditional,
                                pauliop=pauliop, include_barriers=include_barriers)

        # When splitting the circuit, each marker will add an snapshot labeled
        # 'end'
        if split_cycles:
            add_split_marker(circ)

        if idle_delay == 'after':
            add_delay_marker(circ, registers, idle_snapshots, snapshot_type,
                             qubits=registers.QubitRegister, conditional=conditional,
                             pauliop=pauliop, include_barriers=include_barriers)

    return circ

# Mutable int object to serve as a global counter for the snapshot labels

class Int(object):
    def __init__(self, value):
        self.value = value

    def __add__(self, other):
        self.value += other
        return self

    def __str__(self):
        return str(self.value)

    def inc(self):
        self.value += 1


label_counter = Int(0)

# Here, I utilize a quirk of how standard args in functions work.
# The same 'label_counter' will be used for every call of the function,
# even if it is modified. This only works for mutable objects so
# 'int' will not be shared, but 'Int' will.

def add_snapshot_to_circuit(circ, snapshot_type, current_cycle=label_counter,
                            qubits=None, conditional=False,
                            pauliop='ZZZZZ', include_barriers=True, **kwargs):
    """Appends a snapshot to circuit."""

    # Intended functionality: Set label_counter to current_cycle if given
    # if current_cycle != label_counter:
    #     label_counter.value = current_cycle

    # Temp functionality: Reset label_counter if current_cycle == 0
    if current_cycle == 0:
        label_counter.value = 0

    if not isinstance(snapshot_type, list):
        snapshot_type = [snapshot_type]
    if not isinstance(conditional, list):
        conditional = [conditional]

    # Append snapshots
    if snapshot_type:
        for snap in snapshot_type:
            for con in conditional:
                snap_label = get_snapshot_label(snap, con,
                                                label_counter)
                if include_barriers:
                    circ.barrier()
                if snap == 'dm' or snap == 'density_matrix':
                    circ.save_density_matrix(
                        qubits, label=snap_label, conditional=con)
                elif snap == 'exp' or snap == 'expectation_value':
                    circ.save_expectation_value(Pauli(pauliop), qubits,
                                                label=snap_label, conditional=con)
                elif snap == 'expvar' or snap == 'expectation_value_variance':
                    circ.save_expectation_value_variance(Pauli(pauliop), qubits,
                                                         label=snap_label, conditional=con)
                if include_barriers:
                    circ.barrier()
    label_counter.inc()

    return circ


def get_snapshot_label(snapshot_type, conditional, current_cycle):
    """Generate a label for snapshots, given its instructions"""

    # Define snapshot type
    if snapshot_type == 'dm' or snapshot_type == 'density_matrix':
        snap_label = 'dm_'
    elif snapshot_type == 'exp' or snapshot_type == 'expectation_value':
        snap_label = 'exp_'
    elif snapshot_type == 'expvar' or snapshot_type == 'expectation_value_variance':
        snap_label = 'expvar_'

    # Add conditional
    if conditional:
        snap_label += 'con_'

    # Add the current cycle
    snap_label += str(current_cycle)
    return snap_label


def get_empty_stabilizer_circuit(registers, final_measure=True):
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
    if readout is not None:
        circ.add_register(readout)

    return circ


def get_classical_register(n_cycles, reset=True, recovery=False, flag=True, include_fifth_stabilizer = False):
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
            [ClassicalRegister(4 + include_fifth_stabilizer, 'syndrome_cycle_' + str(i) + '_step_' + str(j))
             for j in range(1)] for i in range(n_cycles)]
        return [syndrome_register]


# %% Delay

def add_delay_marker(circ, registers, snapshots=0, snapshot_type='dm', qubits=None, conditional=False, pauliop='ZZZZZ', include_barriers=True):
    """Add a custom gate that does nothing but mark where delay time should be inserted, which is picked up by the noise model.
    It can also divide this idle time into partitions and put a number of snapshots after each of them.
    """

    partitions = max(1, snapshots)  # Zero snapshots still gives one partition
    if partitions > 1:
        gate_name = 'delay/'+str(partitions)
    else:
        gate_name = 'delay'

    # sub_circ = QuantumCircuit(1, name=gate_name)
    # sub_circ.id(0)
    # sub_inst = sub_circ.to_instruction()

    for _ in range(partitions):
        for qb in registers.QubitRegister:
            # circ.append(sub_inst, [qb])
            circ.unitary([[1, 0], [0, 1]], [qb], label=gate_name)
        for an in registers.AncillaRegister:
            # circ.append(sub_inst, [an])
            circ.unitary([[1, 0], [0, 1]], [an], label=gate_name)
        if snapshots:
            add_snapshot_to_circuit(circ, snapshot_type, qubits=qubits,
                                    conditional=conditional, pauliop=pauliop, include_barriers=include_barriers)
# %% ENCODING


def encode_input(registers):
    """Encode the input into logical 0 and 1 for the [[5,1,3]] code. This
    assumes that the 0:th qubit is the original state |psi> = a|0> + b|1>
    """

    warnings.warn("Use V2, it's smaller", DeprecationWarning)
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


def encode_input_v2(registers, include_barriers=True):
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


def transpiled_encoding_WACQT(registers, include_barriers=True, iswap=True):
    """Gives an encoding circuit following the connectivity of a hexagonal
    device, including swapping the ancilla into position afterwards. Note that
    this should be used with caution, as the iswaps are not 'tracked' as when
    using a transpiler, and permutations are not undone at snapshots."""

    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    circ = get_empty_stabilizer_circuit(registers)

    circ.z(qbReg[0])
    circ.h(qbReg[0])
    circ.h(qbReg[1])
    circ.h(qbReg[2])
    circ.h(qbReg[3])
    circ.h(qbReg[4])

    circ.cz(qbReg[0], qbReg[1])
    circ.cz(qbReg[0], qbReg[2])
    circ.cz(qbReg[0], qbReg[3])
    circ.cz(qbReg[0], qbReg[4])

    circ.h(qbReg[0])
    circ.cz(qbReg[0], qbReg[1])

    if iswap:
        circ.iswap(qbReg[0], qbReg[3])
        circ.cz(qbReg[0], qbReg[3])
        circ.u1(-np.pi/2, qbReg[0])
        circ.u1(-np.pi/2, qbReg[3])
    else:
        # Swap without iSwap
        #circ.u1(np.pi/2, qbReg[0])
        #circ.u1(np.pi/2, qbReg[3])
        circ.h(qbReg[0])
        circ.h(qbReg[3])
        circ.cz(qbReg[0], qbReg[3])
        circ.h(qbReg[0])
        circ.h(qbReg[3])
        circ.cz(qbReg[0], qbReg[3])
        circ.h(qbReg[0])
        circ.h(qbReg[3])
        circ.cz(qbReg[0], qbReg[3])
        #circ.u1(-np.pi/2, qbReg[0])
        #circ.u1(-np.pi/2, qbReg[3])

    circ.cz(qbReg[0], qbReg[2])
    circ.cz(qbReg[1], qbReg[2])
    circ.cz(qbReg[3], qbReg[4])
    circ.cz(qbReg[0], qbReg[4])
    if include_barriers:
        circ.barrier()

    if iswap:
        circ.iswap(qbReg[0], anReg[1])
        circ.cz(qbReg[0], anReg[1])
        circ.u1(-np.pi/2, qbReg[0])
        circ.u1(-np.pi/2, anReg[1])
    else:
        # Swap without iSwap
        #circ.u1(np.pi/2, qbReg[0])
        #circ.u1(np.pi/2, anReg[1])
        circ.h(qbReg[0])
        circ.h(anReg[1])
        circ.cz(qbReg[0], anReg[1])
        circ.h(qbReg[0])
        circ.h(anReg[1])
        circ.cz(qbReg[0], anReg[1])
        circ.h(qbReg[0])
        circ.h(anReg[1])
        circ.cz(qbReg[0], anReg[1])
        #circ.u1(-np.pi/2, qbReg[0])
        #circ.u1(-np.pi/2, anReg[1])
    if include_barriers:
        circ.barrier()
    return circ


def transpiled_encoding_DD(registers, include_barriers=True, iswap=True):
    """Gives an encoding circuit following the connectiity of a square grid
    device, including swapping the ancilla into position afterwards. Note that
    this should be used with caution, as the iswaps are not 'tracked' as when
    using a transpiler, and permutations are not undone at snapshots."""
    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    circ = get_empty_stabilizer_circuit(registers)

    circ.z(qbReg[0])
    circ.h(qbReg[0])
    circ.h(qbReg[1])
    circ.h(qbReg[2])
    circ.h(qbReg[3])
    circ.h(qbReg[4])

    circ.cz(qbReg[0], qbReg[1])
    circ.cz(qbReg[0], qbReg[2])
    circ.cz(qbReg[0], qbReg[3])

    if iswap:
        circ.iswap(qbReg[0], qbReg[2])
        circ.cz(qbReg[0], qbReg[2])
        circ.u1(-np.pi/2, qbReg[0])
        circ.u1(-np.pi/2, qbReg[2])
    else:
        # Swap without iSwap
        #circ.u1(np.pi/2, qbReg[0])
        #circ.u1(np.pi/2, qbReg[2])
        circ.h(qbReg[0])
        circ.h(qbReg[2])
        circ.cz(qbReg[0], qbReg[2])
        circ.h(qbReg[0])
        circ.h(qbReg[2])
        circ.cz(qbReg[0], qbReg[2])
        circ.h(qbReg[0])
        circ.h(qbReg[2])
        circ.cz(qbReg[0], qbReg[2])
        #circ.u1(-np.pi/2, qbReg[0])
        #circ.u1(-np.pi/2, qbReg[2])

    circ.cz(qbReg[0], qbReg[3])
    circ.cz(qbReg[2], qbReg[4])

    if iswap:
        circ.iswap(qbReg[0], qbReg[1])
        circ.cz(qbReg[0], qbReg[1])
        circ.u1(-np.pi/2, qbReg[0])
        circ.u1(-np.pi/2, qbReg[1])
    else:
        # Swap without iSwap
        #circ.u1(np.pi/2, qbReg[0])
        #circ.u1(np.pi/2, qbReg[1])
        circ.h(qbReg[0])
        circ.h(qbReg[1])
        circ.cz(qbReg[0], qbReg[1])
        circ.h(qbReg[0])
        circ.h(qbReg[1])
        circ.cz(qbReg[0], qbReg[1])
        circ.h(qbReg[0])
        circ.h(qbReg[1])
        circ.cz(qbReg[0], qbReg[1])
        #circ.u1(-np.pi/2, qbReg[0])
        #circ.u1(-np.pi/2, qbReg[1])

    circ.cz(qbReg[3], qbReg[4])
    circ.h(qbReg[2])
    circ.cz(qbReg[0], qbReg[2])
    circ.cz(qbReg[0], qbReg[1])
    circ.cz(qbReg[2], qbReg[4])
    if include_barriers:
        circ.barrier()

    if iswap:
        circ.iswap(qbReg[0], anReg[1])
        circ.cz(qbReg[0], anReg[1])
        circ.u1(-np.pi/2, qbReg[0])
        circ.u1(-np.pi/2, anReg[1])
    else:
        # Swap without iSwap
        #circ.u1(np.pi/2, qbReg[0])
        #circ.u1(np.pi/2, anReg[1])
        circ.h(qbReg[0])
        circ.h(anReg[1])
        circ.cz(qbReg[0], anReg[1])
        circ.h(qbReg[0])
        circ.h(anReg[1])
        circ.cz(qbReg[0], anReg[1])
        circ.h(qbReg[0])
        circ.h(anReg[1])
        circ.cz(qbReg[0], anReg[1])
        #circ.u1(-np.pi/2, qbReg[0])
        #circ.u1(-np.pi/2, anReg[1])
    if include_barriers:
        circ.barrier()
    return circ


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


def get_encoded_state(theta, phi, include_ancillas='back'):
    """Create the correct 7qb density matrix for an arbitary logical 5qb state.
    Angles are defined as on Bloch sphere.

    Args:
        theta (float): Zenith angle.
        phi (float): Azimuthal angle.

    Returns:
        The encoded state plus two ancilla in 0 state, as a 128x128 numpy array.

    Example: get_encoded_state(np.pi, 0) gives the |1> state.
             get_encoded_state(np.pi/2, np.pi/2) gives the |+> state.
             get_encoded_state(np.pi/2, -np.pi/2) gives the |-> state.
    """
    logical_0, logical_1 = logical_states(include_ancillas)

    return np.cos(theta/2)*logical_0 + \
        np.exp(1j*phi)*np.sin(theta/2)*logical_1

# %% Basis states outside codespace


# TODO: There is a duplicate in post_process. Where should it be?
syndrome_table = [[],
                  [(XGate, 1)],
                  [(ZGate, 4)],
                  [(XGate, 2)],
                  [(ZGate, 2)],
                  [(ZGate, 0)],
                  [(XGate, 3)],
                  [(XGate, 2), (ZGate, 2)],
                  [(XGate, 0)],
                  [(ZGate, 3)],
                  [(ZGate, 1)],
                  [(XGate, 1), (ZGate, 1)],
                  [(XGate, 4)],
                  [(XGate, 0), (ZGate, 0)],
                  [(XGate, 4), (ZGate, 4)],
                  [(XGate, 3), (ZGate, 3)]]


def get_weight_1_basis():
    """Returns a basis set for every state with a distance one from |0>_L.
    This is equivalent to a distance two from |1>_L.
    """

    logical_0 = Statevector(logical_states(None)[0])

    weight_1 = [logical_0]*15

    for syndrome in range(1, 16):  # Loop over the 15 non-trivial errors
        for correction_strategy in syndrome_table[syndrome]:
            weight_1[syndrome-1] = weight_1[syndrome -
                                            1].evolve(correction_strategy[0](), [correction_strategy[1]])

    return weight_1


def get_weight_2_basis():
    """Returns a basis set for every state with a distance one from |0>_L.
    This is equivalent to a distance two from |1>_L.
    """
    # TODO: Check the numbering here. What does it mean in terms of two-qubit errors?
    # The syndromes correspond to what single qubit recovery should be applied to go to |1>_L

    logical_1 = Statevector(logical_states(None)[1])

    weight_2 = [logical_1]*15

    for syndrome in range(1, 16):  # Loop over the 15 non-trivial errors
        for correction_strategy in syndrome_table[syndrome]:
            weight_2[syndrome-1] = weight_2[syndrome -
                                            1].evolve(correction_strategy[0](), [correction_strategy[1]])

    return weight_2
# %% [[4,2,2]] stabilizer code


def get_full_stabilizer_circuit_422(registers=None, n_cycles=1,
                                    initial_state=[1., 0., 0., 0.],
                                    encoding=True,
                                    simulator_type='density_matrix',
                                    final_measure=True, **kwargs):
    """Returns the circuit object for a full repeating stabilizer circuit, 
    including (optional) encoding, n_cycles of repeated stabilizers
    and final readout measurement.
    """

    # If no registers are defined, generate a standard set.
    if not registers:
        registers = StabilizerRegisters(qbReg=QuantumRegister(4, 'code_qubit'),
                                        anReg=AncillaRegister(
                                            1, 'ancilla_qubit'),
                                        clReg=get_classical_register_422(
                                            n_cycles),
                                        readout=ClassicalRegister(4, 'readout'))

    # Unpack registers
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    readout = registers.ReadoutRegister
    if not anReg.size == 1:
        raise Exception('Ancilla register must be of size 1')

    # Define the circuit
    circ = get_empty_stabilizer_circuit_422(registers)

    # Encode the physical qubits into logical qubits
    if encoding:
        # Prepare the two-qubit initial state
        if isinstance(initial_state, list):
            initial_state = np.array(initial_state)
        initial_state /= np.linalg.norm(initial_state)  # Normalize

        # Expand to 5 qubits (Other three in |000>)
        zero_qbs = np.zeros(2**3)
        zero_qbs[0] = 1.
        if simulator_type == 'statevector':
            circ.set_statevector(np.kron(zero_qbs, initial_state))
        else:
            circ.set_density_matrix(np.kron(zero_qbs, initial_state))

        # Encode
        circ.compose(encode_input_422(registers, **kwargs), inplace=True)
    else:
        if simulator_type == 'statevector':
            circ.set_statevector(get_encoded_state_422(initial_state))
        else:
            circ.set_density_matrix(get_encoded_state_422(initial_state))

    add_snapshot_to_circuit(circ,
                            current_cycle=0,
                            qubits=qbReg,
                            **kwargs)

    # Stabilizer
    circ.compose(get_repeated_stabilization_422(registers, n_cycles=n_cycles,
                                                # reset=reset,
                                                # snapshot_type=snapshot_type,
                                                # include_barriers=include_barriers,
                                                **kwargs), inplace=True)

    # Final readout
    if final_measure:
        circ.measure(qbReg, readout)
    return circ


def get_repeated_stabilization_422(registers, n_cycles=1, extra_snapshots=False,
                                   **kwargs):
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

    circ = get_empty_stabilizer_circuit_422(registers)

    for current_cycle in range(n_cycles):
        circ.compose(stabilizer_cycle_422(registers,
                                          current_cycle=current_cycle,
                                          extra_snapshots=extra_snapshots,
                                          **kwargs
                                          ), inplace=True)
        if not extra_snapshots: # Snapshots are instead in stabilizer_cycle_422
            add_snapshot_to_circuit(circ, qubits=registers.QubitRegister,
                                    **kwargs)

    return circ


def get_empty_stabilizer_circuit_422(registers):
    """Create an empty qiskit circuit adapted for stabilizer circuits"""

    # Unpack registers
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    clReg = registers.SyndromeRegister
    readout = registers.ReadoutRegister

    circ = QuantumCircuit(qbReg, anReg)
    if isinstance(clReg, list):
        for reg in clReg:
            circ.add_register(reg)
    else:
        circ.add_register(clReg)
    circ.add_register(readout)

    return circ


def get_classical_register_422(n_cycles):
    """Generate a list of classical registers for storing all measurement data.
    Each register contains two bits (the two syndrome measurements), each
    register in the list corresponding to one stabilizer cycle.
    """

    # A register of four bits per cycle
    syndrome_register = [ClassicalRegister(2, 'syndrome_cycle_' + str(i))
                         for i in range(n_cycles)]
    return syndrome_register


def get_encoded_state_422(initial_state, include_ancillas='back'):
    """Create the correct encoded state for the [[4,2,2]] code.

    Args:
        initial_state (list): List corresponding to the (unnormalized) initial
                              two_qubit state. 
        include_ancillas (str/None, optional): Whether to append the ancillas by
                                               tensor product to the end. 
                                               Defaults to True.

    Returns:
        The encoded state plus (optional) one ancilla in 0 state, as a 1D 
        numpy array.

    Example: get_encoded_state([1,0,0,0]) gives the |00> state.
             get_encoded_state([1,1,1,1]) gives the |++> state.
             get_encoded_state([1,-1,-1,1]) gives the |--> state.
    """

    if isinstance(initial_state, list):
        initial_state = np.array(initial_state)
    # Get the four logical states
    logical = logical_states_422(include_ancillas)

    # Map the initial state to the encoded equivalent
    if include_ancillas:
        statevec = np.zeros(2**5, dtype=complex)
    else:
        statevec = np.zeros(2**4, dtype=complex)
    for i in range(len(logical)):
        statevec += initial_state[i]*logical[i]

    # Normalize
    statevec /= np.linalg.norm(statevec)

    return statevec


def computational_state(state, threshold=1e-2):
    """Determine whether a statevector represents any of the four computational
    states of |00>, |01>, |10> or |11> and returns the corresponding key as a
    string."""
    comp_states = np.eye(4)
    fidelities = [state_fidelity(state, comp_states[i,:]) for i in range(4)]
    max_fid = max(fidelities)
    max_idx = fidelities.index(max_fid)
    if max_fid < 1-threshold:
        warnings.warn('State could not be matched with any computational state, assuming |00>')
        return '00'
    return str(bin(max_idx))[2:].zfill(2)

def encode_input_422(registers, include_barriers=True, initial_state=[1.,0.,0.,0.], 
        circuit_index=1, include_swap=False, **kwargs):
    """Encode the input into logical states for the [[4,2,2]] code. This
    assumes that the 0:th and 1:st qubit containts the original state 
    |psi> = a|00> + b|01> + c|10> + d|11>.

    Circuit for circuit_index=0 found in 
    https://www.researchgate.net/publication/330860914_Fault-tolerant_gates_via_homological_product_codes
    """

    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    circ = QuantumCircuit(qbReg)
    if include_swap:
        circ.add_register(anReg)

    # Encoding of arbitrary state
    if circuit_index == 0:
        
        circ.h(qbReg[3])
        circ.cx(qbReg[0], qbReg[2])
        circ.cx(qbReg[3], qbReg[1])
        circ.cx(qbReg[1], qbReg[2])
        circ.cx(qbReg[3], qbReg[0])
        if include_barriers:
            circ.barrier()

    # Encoding of the four computational states
    if circuit_index == 1:
        circ.h(qbReg[0])

        # Prepare any computational state
        state_str = computational_state(initial_state)
        if state_str == '01':
            #circ.x(qbReg[1])
            #circ.x(qbReg[2])
            circ.x(qbReg[2])
            circ.x(qbReg[3])
        elif state_str == '10':
            circ.x(qbReg[1])
            circ.x(qbReg[3])
        elif state_str == '11':
            circ.x(qbReg[1])
            circ.x(qbReg[2])
            #circ.x(qbReg[2])
            #circ.x(qbReg[3])

        # Encoding
        circ.cx(qbReg[0], qbReg[1])
        circ.cx(qbReg[0], qbReg[2])
        circ.cx(qbReg[0], qbReg[3])
        if include_swap:
            circ.swap(anReg[0], qbReg[0])
    return circ


def stabilizer_cycle_422(registers, reset=True, current_cycle=0,
                         include_barriers=True, extra_snapshots=False,
                         flip_stab_order=False, **kwargs):
    """Circuit for performing a full stabilizer cycle of the [[4,2,2]] code."""

    # Stabilizers in one cycle
    stabilizer_list = [stabilizer_XXXX,
                       stabilizer_ZZZZ]
    if flip_stab_order:
        stabilizer_list.reverse()

    # Create list of syndrome bits
    if isinstance(registers.SyndromeRegister, list):
        syn_reg = registers.SyndromeRegister[current_cycle]
        syn_bit_list = [syn_reg[n] for n in range(2)]
    else:
        syn_bit_list = [
            registers.SyndromeRegister[n+2*current_cycle] for n in range(2)
        ]

    # Create circuit and run stabilizers
    circ = get_empty_stabilizer_circuit_422(registers)
    for i in range(2):
        circ.compose(stabilizer_list[i](registers,
                                        syn_bit=syn_bit_list[i], reset=reset), inplace=True)
        if extra_snapshots:
            add_snapshot_to_circuit(circ, qubits=registers.QubitRegister,
                                    include_barriers=include_barriers, **kwargs)
        elif include_barriers:
            circ.barrier()

    return circ


def stabilizer_XXXX(registers, syn_bit=None, reset=True):
    """Gives the circuit for running the regular XXXX stabilizer for the
    [[4,2,2]] code.

    Args:
        registers (StabilizerRegister): Register object
        syn_bit (clbit): The classical bit or register to store measurement in.
        reset (bool, optional): Whether to reset ancillas between measurements.
                                Defaults to True.
    Returns:
        circ: QuantumCircuit object for measuring the XXXX stabilizer.
    """

    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    circ = get_empty_stabilizer_circuit_422(registers)

    # TODO: This might be redundant since we only have a single ancilla
    if not anReg.size == 1:
        warnings.warn(
            'Ancilla register has size >1, something might be wrong.')
    anQb = anReg[0]

    # Entangle ancilla
    circ.h(anQb)
    circ.cx(anQb, qbReg)
    circ.h(anQb)

    # Measure
    circ.measure(anQb, syn_bit)

    # Reset
    if reset:
        circ.reset(anQb)

    return circ


def stabilizer_ZZZZ(registers, syn_bit=None, reset=True):
    """Gives the circuit for running the regular XXXX stabilizer for the
    [[4,2,2]] code.

    Args:
        registers (StabilizerRegister): Register object
        syn_bit (clbit): The classical bit or register to store measurement in.
        reset (bool, optional): Whether to reset ancillas between measurements.
                                Defaults to True.
    Returns:
        circ: QuantumCircuit object for measuring the XXXX stabilizer.
    """

    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    circ = get_empty_stabilizer_circuit_422(registers)

    # TODO: This might be redundant since we only have a single ancilla
    if not anReg.size == 1:
        warnings.warn(
            'Ancilla register has size >1, something might be wrong.')
    anQb = anReg[0]

    # Entangle ancilla
    circ.h(anQb)
    circ.cz(anQb, qbReg)
    circ.h(anQb)

    # Measure
    circ.measure(anQb, syn_bit)

    # Reset
    if reset:
        circ.reset(anQb)

    return circ


def logical_states_422(include_ancillas='front') -> List[List[float]]:
    """Returns the logical states for the [[4,2,2]] code. This follows the
    opposite of Qiskits ordering of qubits, i.e. |10> corresponds to the
    0:th qubit (uppermost in circuits) being in the |1> and is represented by the vector
    [0, 0, 1, 0].

    Args:
        include_ancillas (str/None, optional): Whether to append the ancillas by
                                    tensor product to the end. Defaults to True.

    Returns:
        List[List[float]]: List of all four logical states
    """
    logical_00 = np.zeros(2**4)
    logical_00[0b0000] = 1/np.sqrt(2)
    logical_00[0b1111] = 1/np.sqrt(2)

    logical_01 = np.zeros(2**4)
    logical_01[0b0101] = 1/np.sqrt(2)
    logical_01[0b1010] = 1/np.sqrt(2)

    logical_10 = np.zeros(2**4)
    logical_10[0b0110] = 1/np.sqrt(2)
    logical_10[0b1001] = 1/np.sqrt(2)

    logical_11 = np.zeros(2**4)
    logical_11[0b0011] = 1/np.sqrt(2)
    logical_11[0b1100] = 1/np.sqrt(2)

    if include_ancillas:
        # Add an ancilla in |0>
        an0 = np.zeros(2)
        an0[0] = 1.0
        if include_ancillas == 'front':
            logical_00 = np.kron(logical_00, an0)
            logical_01 = np.kron(logical_01, an0)
            logical_10 = np.kron(logical_10, an0)
            logical_11 = np.kron(logical_11, an0)
        elif include_ancillas == 'back':
            logical_00 = np.kron(an0, logical_00)
            logical_01 = np.kron(an0, logical_01)
            logical_10 = np.kron(an0, logical_10)
            logical_11 = np.kron(an0, logical_11)

    #return [logical_00, logical_01, logical_10, logical_11]

    # Reordered to match some papers
    # TODO: Rename the variables instead of reordering to avoid confusion
    return [logical_00, logical_11, logical_01, logical_10]

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
                               include_barriers=True, pipeline=False, snapshot=True, 
                               snapshot_type='dm', conditional=False, pauliop='ZZZZZ',
                               include_fifth_stabilizer=False, **kwargs):
    """Run all four stabilizers without flags, as well as an optional
    recovery. The input current_step is only relevant for flagged cycles, and
    should be set to 0 otherwise.

    NOTE: Maybe use a list if ancilla indices instead? E.g. ancillas = [1,2,1,2]
    Args:
        num_ancillas: Specifies how many ancillas to spread the measurements over
    """

    # Number of stabilizer generators (default four)
    num_stabilizers = 4 if include_fifth_stabilizer is False else 5

    # Use the previous behavior if num_ancillas=None
    if not num_ancillas:
        if registers.AncillaRegister.size == 2:
            anQb_list = [registers.AncillaRegister[1]]*num_stabilizers
        elif registers.AncillaRegister.size >= 4:
            # I don't like this really, we don't use the flagged circuit anymore so it shouldn't get the 0 spot by default
            anQb_list = [registers.AncillaRegister[n] for n in np.arange(1, num_stabilizers+1)]
        else:
            Warning("Ancilla reg too small (this should never happen)")
    else:
        anQb_list = [registers.AncillaRegister[n % num_ancillas]
                     for n in range(num_stabilizers)]

    if pipeline:
        stabilizer_list = [_pipeline_stabilizer_XZZXI,
                           _pipeline_stabilizer_IXZZX,
                           _pipeline_stabilizer_XIXZZ,
                           _pipeline_stabilizer_ZXIXZ]
    elif include_fifth_stabilizer:
        # TODO: Make this work with pipeline?
        stabilizer_list = [_unflagged_stabilizer_XZZXI,
                           _unflagged_stabilizer_IXZZX,
                           _unflagged_stabilizer_XIXZZ,
                           _unflagged_stabilizer_ZXIXZ,
                           _unflagged_stabilizer_ZZXIX]
    else:
        stabilizer_list = [_unflagged_stabilizer_XZZXI,
                           _unflagged_stabilizer_IXZZX,
                           _unflagged_stabilizer_XIXZZ,
                           _unflagged_stabilizer_ZXIXZ]

    # Create list of syndrome bits
    if isinstance(registers.SyndromeRegister, list):
        syn_reg = registers.SyndromeRegister[0][current_cycle][current_step]
        syn_bit_list = [syn_reg[n] for n in range(num_stabilizers)]
    else:
        syn_bit_list = [registers.SyndromeRegister[n] for n in range(num_stabilizers)]

    # Create circuit and run stabilizers
    circ = get_empty_stabilizer_circuit(registers)
    for i in range(num_stabilizers):
        circ.compose(stabilizer_list[i](registers, anQb=anQb_list[i],
                                        syn_bit=syn_bit_list[i], reset=reset), inplace=True)

        # TODO: Make this work for all setting (e.g. conditional=True)
        if snapshot:
            add_snapshot_to_circuit(circ, snapshot_type=snapshot_type,
                                    qubits=registers.QubitRegister, conditional=conditional, include_barriers=include_barriers)

        if include_barriers and not snapshot:
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

        # TODO: Make this work for all setting (e.g. conditional=True)
        # add_snapshot_to_circuit(circ, snapshot_type='dm',
        #                         qubits=registers.QubitRegister, conditional=False, include_barriers=include_barriers, pauliop=pauliop)

        if not include_barriers:  # Always put a barrier here, so add an extra one if there wasn't one after the last stabilzier
            circ.barrier()
        circ.compose(unflagged_recovery(
            registers, reset, current_cycle), inplace=True)
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

# NOTE: This is the fifth (superfluous) stabilizer generator that is not normally used
def _unflagged_stabilizer_ZZXIX(registers, anQb=None, syn_bit=None, reset=True):
    """Gives the circuit for running the regular ZXIXZ stabilizer without flag.
    The current_step input should be set to zero unless running flagged cycles.
    """
    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    circ = get_empty_stabilizer_circuit(registers)

    if not anQb:
        if anReg.size == 2:
            anQb = anReg[1]
        else:
            anQb = anReg[4]

    # Z
    circ.h(anQb)
    circ.cz(anQb, qbReg[0])

    # Z
    circ.cz(anQb, qbReg[1])

    # X
    circ.h(qbReg[2])
    circ.cz(anQb, qbReg[2])
    circ.h(qbReg[2])

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
# %%


def transpiled_dd_cycle(registers, current_cycle=0, current_step=0, reset=True,
                        recovery=False):
    """Gives the circuit for a full stabilizer cycle following the double 
    diamond connectivity.
    """
    # Create list of syndrome bits
    if isinstance(registers.SyndromeRegister, list):
        syn_reg = registers.SyndromeRegister[0][current_cycle][current_step]
        syn_bit_list = [syn_reg[n] for n in range(4)]
    else:
        syn_bit_list = [registers.SyndromeRegister[n] for n in range(4)]

    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    circ = get_empty_stabilizer_circuit(registers)

    # XZZXI
    circ.u1(np.pi/2, qbReg[0])
    circ.sx(qbReg[0])
    circ.u1(np.pi/2, qbReg[0])
    circ.u1(np.pi/2, qbReg[3])
    circ.sx(qbReg[3])
    circ.u1(np.pi/2, qbReg[3])
    circ.u1(np.pi/2, anReg[1])
    circ.sx(anReg[1])
    circ.u1(np.pi/2, anReg[1])

    # circ.h(qbReg[0])
    # circ.h(qbReg[3])
    # circ.h(anReg[1])

    circ.cz(anReg[1], qbReg[0])
    circ.cz(anReg[1], qbReg[1])
    circ.cz(anReg[1], qbReg[2])
    circ.cz(anReg[1], qbReg[3])

    # circ.h(qbReg[0])
    # circ.h(qbReg[3])
    # circ.h(anReg[1])

    circ.u1(np.pi/2, qbReg[0])
    circ.sx(qbReg[0])
    circ.u1(np.pi/2, qbReg[0])
    circ.u1(np.pi/2, qbReg[3])
    circ.sx(qbReg[3])
    circ.u1(np.pi/2, qbReg[3])
    circ.u1(np.pi/2, anReg[1])
    circ.sx(anReg[1])
    circ.u1(np.pi/2, anReg[1])
    circ.measure(anReg[1], syn_bit_list[0])
    if reset:
        circ.reset(anReg[1])

    # IXZZX
    circ.u1(np.pi/2, qbReg[1])
    circ.sx(qbReg[1])
    circ.u1(np.pi/2, qbReg[1])
    circ.u1(np.pi/2, qbReg[4])
    circ.sx(qbReg[4])
    circ.u1(np.pi/2, qbReg[4])
    circ.u1(np.pi/2, anReg[1])
    circ.sx(anReg[1])
    circ.u1(np.pi/2, anReg[1])

    # circ.h(anReg[1])
    # circ.h(qbReg[1])
    # circ.h(qbReg[4])

    circ.cz(anReg[1], qbReg[1])

    circ.iswap(qbReg[1], qbReg[4])
    circ.cz(qbReg[1], qbReg[4])
    circ.u1(-np.pi/2, qbReg[1])
    circ.u1(-np.pi/2, qbReg[4])

    circ.cz(anReg[1], qbReg[2])
    circ.cz(anReg[1], qbReg[3])
    circ.cz(anReg[1], qbReg[1])  # 4-1

    # circ.h(anReg[1])
    # circ.h(qbReg[1])
    # circ.h(qbReg[4])

    circ.u1(np.pi/2, qbReg[1])
    circ.sx(qbReg[1])
    circ.u1(np.pi/2, qbReg[1])
    circ.u1(np.pi/2, qbReg[4])
    circ.sx(qbReg[4])
    circ.u1(np.pi/2, qbReg[4])
    circ.u1(np.pi/2, anReg[1])
    circ.sx(anReg[1])
    circ.u1(np.pi/2, anReg[1])
    circ.measure(anReg[1], syn_bit_list[1])
    if reset:
        circ.reset(anReg[1])

    # XIXZZ
    circ.u1(np.pi/2, qbReg[0])
    circ.sx(qbReg[0])
    circ.u1(np.pi/2, qbReg[0])
    circ.u1(np.pi/2, qbReg[2])
    circ.sx(qbReg[2])
    circ.u1(np.pi/2, qbReg[2])
    circ.u1(np.pi/2, anReg[1])
    circ.sx(anReg[1])
    circ.u1(np.pi/2, anReg[1])

    # circ.h(anReg[1])
    # circ.h(qbReg[0])
    # circ.h(qbReg[2])

    circ.cz(anReg[1], qbReg[0])
    circ.cz(anReg[1], qbReg[2])
    circ.cz(anReg[1], qbReg[3])
    circ.cz(anReg[1], qbReg[1])  # 4-1

    # circ.h(anReg[1])
    # circ.h(qbReg[0])
    # circ.h(qbReg[2])

    circ.u1(np.pi/2, qbReg[0])
    circ.sx(qbReg[0])
    circ.u1(np.pi/2, qbReg[0])
    circ.u1(np.pi/2, qbReg[2])
    circ.sx(qbReg[2])
    circ.u1(np.pi/2, qbReg[2])
    circ.u1(np.pi/2, anReg[1])
    circ.sx(anReg[1])
    circ.u1(np.pi/2, anReg[1])
    circ.measure(anReg[1], syn_bit_list[2])
    if reset:
        circ.reset(anReg[1])

    # ZXIXZ
    circ.iswap(qbReg[1], qbReg[4])
    circ.cz(qbReg[1], qbReg[4])
    circ.u1(-np.pi/2, qbReg[1])
    circ.u1(-np.pi/2, qbReg[4])

    circ.u1(np.pi/2, qbReg[3])
    circ.sx(qbReg[3])
    circ.u1(np.pi/2, qbReg[3])
    circ.u1(np.pi/2, qbReg[1])
    circ.sx(qbReg[1])
    circ.u1(np.pi/2, qbReg[1])
    circ.u1(np.pi/2, anReg[1])
    circ.sx(anReg[1])
    circ.u1(np.pi/2, anReg[1])

    # circ.h(anReg[1])
    # circ.h(qbReg[1])
    # circ.h(qbReg[3])

    circ.cz(anReg[1], qbReg[0])
    circ.cz(anReg[1], qbReg[1])
    circ.cz(anReg[1], qbReg[3])

    circ.u1(np.pi/2, qbReg[1])
    circ.sx(qbReg[1])
    circ.u1(np.pi/2, qbReg[1])
    # circ.h(qbReg[1])

    circ.iswap(qbReg[1], qbReg[4])
    circ.cz(qbReg[1], qbReg[4])
    circ.u1(-np.pi/2, qbReg[1])
    circ.u1(-np.pi/2, qbReg[4])

    circ.cz(anReg[1], qbReg[1])  # 4-1

    # circ.h(anReg[1])
    # circ.h(qbReg[3])

    circ.u1(np.pi/2, qbReg[3])
    circ.sx(qbReg[3])
    circ.u1(np.pi/2, qbReg[3])
    circ.u1(np.pi/2, anReg[1])
    circ.sx(anReg[1])
    circ.u1(np.pi/2, anReg[1])

    circ.iswap(qbReg[1], qbReg[4])
    circ.cz(qbReg[1], qbReg[4])
    circ.u1(-np.pi/2, qbReg[1])
    circ.u1(-np.pi/2, qbReg[4])
    circ.measure(anReg[1], syn_bit_list[3])
    if reset:
        circ.reset(anReg[1])

    # Recovery
    if recovery is True:
        circ.barrier()
        circ.compose(unflagged_recovery(
            registers, reset, current_cycle), inplace=True)
        circ.barrier()
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
    # TODO: Add delay
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


# %% Internal testing of functions above
if __name__ == "__main__":
    # The settings for our circuit
    kwargs = {
        'n_cycles': 2,
        'reset': True,
        'recovery': True,
        'flag': False,
        'encoding': False,
        'conditional': True,
        'include_barriers': True,
        'generator_snapshot': True,
        'idle_snapshots': 1,
        'final_measure': False}
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
    circ += get_full_stabilizer_circuit(registers, **kwargs)

    display(circ.draw(output='mpl'))
    # Run it
    n_shots = 2000
    results = execute(
        circ,
        Aer.get_backend('qasm_simulator'),
        noise_model=None,
        shots=n_shots
    ).result()

# %%
