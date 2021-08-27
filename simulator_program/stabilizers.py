"""This file contains functions for building the circuits necessary for running
the [[5,1,3]] QEC code. The main function here is get_full_stabilizer_circuit,
and a majority of other functions are accessed by it.
"""

# %% Import modules
from IPython.display import display
from qiskit.quantum_info import Pauli
import numpy as np
from qiskit import (QuantumCircuit,
                    QuantumRegister,
                    ClassicalRegister,
                    AncillaRegister,
                    execute,
                    Aer
                    )
# These imports need to be included even if they are unused, or you will get linter errors
from qiskit.providers.aer.library import set_density_matrix, set_statevector
from qiskit.circuit import measure, reset
from qiskit.providers.aer.library import save_density_matrix, save_expectation_value
from typing import List
import warnings
from qiskit.circuit.library import XGate, ZGate
from qiskit.quantum_info.states.statevector import Statevector

# Disable error which gives false positives in this file
# pylint: disable=E1101
# %% General functions


class StabilizerRegisters:
    """Defines a set of registers for running the [[5,1,3]] QEC circuit."""
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


def get_registers(conditional=False, final_measure=False, n_cycles=0, reset=True,
                  recovery=True, include_fifth_stabilizer=False):
    """Create a set of circuit registers based on a set of circuit parameters.
    """
    qbReg = QuantumRegister(5, 'code_qubit')
    anReg = AncillaRegister(2, 'ancilla_qubit')

    if conditional:
        # Advanced list of registers
        crReg = get_classical_register(
            n_cycles, reset, recovery, include_fifth_stabilizer)
    else:
        crReg = ClassicalRegister(4, 'syndrome_bit')  # The typical register
    if final_measure:
        readout = ClassicalRegister(5, 'readout')
        registers = StabilizerRegisters(qbReg, anReg, crReg, readout)
    else:
        registers = StabilizerRegisters(qbReg, anReg, crReg, None)
    return registers


def get_full_stabilizer_circuit(registers=None, n_cycles=1,
                                reset=True, recovery=False,
                                snapshot_type='density_matrix',
                                include_barriers=True, conditional=True,
                                encoding=True, theta=0, phi=0,
                                generator_snapshot=False, idle_snapshots=0, pauliop='ZZZZZ', device=None,
                                simulator_type='density_matrix', final_measure=True, **kwargs):
    """Returns the circuit for a full repeating stabilizer circuit, including 
    encoding, n_cycles of repeated stabilizers and final measurement. All parts
    of the cycle are optional and can be changed using different args.

    Args:
        registers: StabilizerRegisters object containing the necessary quantum-
                   and classical registers for the circuit. If left empty, it
                   defaults to standard registers based on circuit settings.
        n_cycles (int): The number of stabilizer cycles to perform. Defaults to 1.
        reset (bool): Whether to reset the ancilla qubit after measurements.
                      Defaults to True.
        recovery (bool): Whether to perform error correcion at the end of cycle,
                         based on the cycles measurement outcomes. Defaults to False.
        snapshot_type (str): The type of snapshots to save of the qubit state at
                             certain points in the cycle. Valid options are
                             'density_matrix' ('dm'), 'expectation_value ('exp')
                             or 'expectation_value_variance' ('exp_var'). For
                             the latter two, the operator to measure expectation
                             value for must be specified using the keyword 
                             pauliop. Defaults to 'density_matrix'.
        include_barriers (bool): Whether to insert barriers between sections of
                                 the circuit, preventing gates to be moved past
                                 them. This may help with robustness. Defaults
                                 to True.
        conditional (bool): Whether to condition snapshots on previous qubit
                            measurements. Defaults to True.
        encoding (bool): Whether to perform the encoding of qubits at the start.
                         If set to False, the logical state will instead be
                         perfectly initialized.
        theta (float): Zenith angle of the qubit state. Defaults to 0.
        phi (float): Azimuthal angle of the qubit state. Defaults to 0.
        generator_snapshot (bool): Whether to append a snapshot after each
                                   stabilizer measurement. If set to False, it
                                   will only take snapshots after each cycle.
        idle_snapshots (int): The number of snapshots to append during an
                              (optional) delay time between cycles. Default to 0.
        pauliop (str): Five character string corresponding to the five-qubit
                       expectation value to measure (if snapshot_type is set to
                       expectation value or expectation value variance).
                       Defaults to 'ZZZZZ'.
        device: Whether to conform the circuit to a specific device layout. 
                Available options are None or 'double_diamond'. If set to None,
                it will assume full connectivity. Note that this can also be
                taken care of through transpilation. But specifying a device
                here might help with specific circuits. Defaults to None.
        simulator_type (str): The type of simulation to run. Can be either
                              'statevector' or 'density_matrix'. Defaults to
                              'density_matrix'.
        final_measure (bool): Whether to perform a final measurement of the 
                              five-qubit state after all stabilizer cycles.
                              Defaults to True.
    Returns:
        circ: The resulting QuantumCircuit object
    """

    include_fifth_stabilizer = False
    if 'include_fifth_stabilizer' in kwargs:
        include_fifth_stabilizer = kwargs['include_fifth_stabilizer']

    # TODO: Make this compatible with other codes?
    if registers is None:
        registers = get_registers(
            conditional, final_measure, n_cycles, reset, recovery, include_fifth_stabilizer)

    if not registers.AncillaRegister.size == 2 and not registers.AncillaRegister.size == 5:
        raise Exception('Ancilla register must be of size 2 or 5')

    # Define the circuit
    circ = get_empty_stabilizer_circuit(registers)

    if encoding:
        # TODO: Using rx and rz messes with the transpiler. Make a better fix?
        if theta != 0 or phi != 0:
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
                                            reset=reset, recovery=recovery,
                                            snapshot_type=snapshot_type,
                                            conditional=conditional,
                                            generator_snapshot=generator_snapshot,
                                            idle_snapshots=idle_snapshots,
                                            include_barriers=include_barriers,
                                            pauliop=pauliop, device=device,
                                            **kwargs), inplace=True)

    # Final readout
    if final_measure:
        circ.measure(registers.QubitRegister, registers.ReadoutRegister)
    return circ


def get_repeated_stabilization(registers, n_cycles=1, reset=True, recovery=False,
                               snapshot_type='density_matrix', include_barriers=True, 
                               conditional=True, generator_snapshot=True,
                               pauliop='ZZZZZ', device=None, idle_delay='after', 
                               idle_snapshots=0,  **kwargs):
    """Generates a circuit of repeated stabilizer measurements.

    Args:
        registers: StabilizerRegisters object containing the necessary quantum-
                   and classical registers for the circuit.
        n_cycles (int): The number of stabilizer cycles to perform. Defaults to 1.
        reset (bool): Whether to reset the ancilla qubit after measurements.
                      Defaults to True.
        recovery (bool): Whether to perform error correcion at the end of cycle,
                         based on the cycles measurement outcomes. Defaults to False.
        snapshot_type (str): The type of snapshots to save of the qubit state at
                             certain points in the cycle. Valid options are
                             'density_matrix' ('dm'), 'expectation_value ('exp')
                             or 'expectation_value_variance' ('exp_var'). For
                             the latter two, the operator to measure expectation
                             value for must be specified using the keyword 
                             pauliop. Defaults to 'density_matrix'.
        include_barriers (bool): Whether to insert barriers between sections of
                                 the circuit, preventing gates to be moved past
                                 them. This may help with robustness. Defaults
                                 to True.
        conditional (bool): Whether to condition snapshots on previous qubit
                            measurements. Defaults to True.
        generator_snapshot (bool): Whether to append a snapshot after each
                                   stabilizer measurement. If set to False, it
                                   will only take snapshots after each cycle.
        idle_snapshots (int): The number of snapshots to append during an
                              (optional) delay time between cycles. Default to 0.
        pauliop (str): Five character string corresponding to the five-qubit
                       expectation value to measure (if snapshot_type is set to
                       expectation value or expectation value variance).
                       Defaults to 'ZZZZZ'.
        device: Whether to conform the circuit to a specific device layout. 
                Available options are None or 'double_diamond'. If set to None,
                it will assume full connectivity. Note that this can also be
                taken care of through transpilation. But specifying a device
                here might help with specific circuits. Defaults to None.
        idle_delay (str): Where to put (optional) idle delay in a cycle. Can be
                          either 'before' or 'after'. Defaults to 'after'. Note
                          that the delay must be introduced through a noise
                          model, and will otherwise be 0 ns.
    Returns:
        circ: The resulting QuantumCircuit object
    """

    circ = get_empty_stabilizer_circuit(registers)

    for current_cycle in range(n_cycles):
        if idle_delay == 'before':
            add_delay_marker(circ, registers, idle_snapshots, snapshot_type,
                             qubits=registers.QubitRegister, conditional=conditional,
                             pauliop=pauliop, include_barriers=include_barriers)
                             
        if device == 'double_diamond':
            circ.compose(transpiled_dd_cycle(registers,
                                             reset=reset,
                                             recovery=recovery,
                                             current_cycle=current_cycle,
                                             ), inplace=True)
        else:
            circ.compose(get_stabilizer_cycle(registers,
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
    """Create an empty Qiskit adapted for stabilizer circuits."""

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
    if readout is not None:
        circ.add_register(readout)

    return circ


def get_classical_register(n_cycles, reset=True, recovery=False, 
                           include_fifth_stabilizer=False):
    """Generate lists of classical registers for storing all measurement data
    in a full stabilizer circuit.

    Args:
        n_cycles (int): The number of stabilizer cycles in the circuit.
        reset (bool): Whether ancilla is reset between measurements. Defaults to True.
        recovery (bool): Whether error correction is performed at the end of cycle.
                         Defaults to True.
        include_fifth_stabilizer (bool): Whether the fifth (superfluous) stabilizer
                                         is in the cycle. Defaults to False.

    Returns:
        syndrome_register: By default returns a list of 4-bit syndrome registers.
                           The number of registers corresponds to the number of
                           cycles. In certain cases, registers will contains 
                           five bits instead, if the settings require so.
    """

    if recovery and not reset:  # Needs an extra bit per register
        # A register of five bits per cycle
        syndrome_register = [ClassicalRegister(5, 'syndrome_cycle_' + str(i))
                             for i in range(n_cycles)]
        return syndrome_register

    else:
        # A register of four bits per cycle (or five if include_fifth_stabilizer)
        syndrome_register = [ClassicalRegister(4 + include_fifth_stabilizer,
                                               'syndrome_cycle_' + str(i)) for i in range(n_cycles)]
        return syndrome_register


# %% Delay

def add_delay_marker(circ, registers, snapshots=0, snapshot_type='dm', 
                     qubits=None, conditional=False, pauliop='ZZZZZ', 
                     include_barriers=True):
    """Add a custom gate that does nothing but mark where delay time should be inserted, which is picked up by the noise model.
    It can also divide this idle time into partitions and put a number of snapshots after each of them.
    """

    partitions = max(1, snapshots)  # Zero snapshots still gives one partition
    if partitions > 1:
        gate_name = 'delay/'+str(partitions)
    else:
        gate_name = 'delay'


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
    using a transpiler, and permutations are not undone at snapshots.
    
    Args:
        registers: StabilizerRegisters object.
        include_barriers (bool): Whether to insert barriers between sections of
                                 the circuit, preventing gates to be moved past
                                 them. Defaults to True.
        iswap (bool): Whether to use the iSWAP gate for swapping qubit position.
                      If set to False, a series of 3 CZ gates is instead used.
    Returns:
        circ: QuantumCircuit object for the circuit.
    """

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
        circ.h(qbReg[0])
        circ.h(qbReg[3])
        circ.cz(qbReg[0], qbReg[3])
        circ.h(qbReg[0])
        circ.h(qbReg[3])
        circ.cz(qbReg[0], qbReg[3])
        circ.h(qbReg[0])
        circ.h(qbReg[3])
        circ.cz(qbReg[0], qbReg[3])

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
        circ.h(qbReg[0])
        circ.h(anReg[1])
        circ.cz(qbReg[0], anReg[1])
        circ.h(qbReg[0])
        circ.h(anReg[1])
        circ.cz(qbReg[0], anReg[1])
        circ.h(qbReg[0])
        circ.h(anReg[1])
        circ.cz(qbReg[0], anReg[1])
    if include_barriers:
        circ.barrier()
    return circ


def transpiled_encoding_DD(registers, include_barriers=True, iswap=True):
    """Gives an encoding circuit following the connectiity of a square grid
    device, including swapping the ancilla into position afterwards. Note that
    this should be used with caution, as the iswaps are not 'tracked' as when
    using a transpiler, and permutations are not undone at snapshots.
    
    Args:
        registers: StabilizerRegisters object.
        include_barriers (bool): Whether to insert barriers between sections of
                                 the circuit, preventing gates to be moved past
                                 them. Defaults to True.
        iswap (bool): Whether to use the iSWAP gate for swapping qubit position.
                      If set to False, a series of 3 CZ gates is instead used.
    Returns:
        circ: QuantumCircuit object for the circuit.
    """
    
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
        circ.h(qbReg[0])
        circ.h(qbReg[2])
        circ.cz(qbReg[0], qbReg[2])
        circ.h(qbReg[0])
        circ.h(qbReg[2])
        circ.cz(qbReg[0], qbReg[2])
        circ.h(qbReg[0])
        circ.h(qbReg[2])
        circ.cz(qbReg[0], qbReg[2])

    circ.cz(qbReg[0], qbReg[3])
    circ.cz(qbReg[2], qbReg[4])

    if iswap:
        circ.iswap(qbReg[0], qbReg[1])
        circ.cz(qbReg[0], qbReg[1])
        circ.u1(-np.pi/2, qbReg[0])
        circ.u1(-np.pi/2, qbReg[1])
    else:
        # Swap without iSwap
        circ.h(qbReg[0])
        circ.h(qbReg[1])
        circ.cz(qbReg[0], qbReg[1])
        circ.h(qbReg[0])
        circ.h(qbReg[1])
        circ.cz(qbReg[0], qbReg[1])
        circ.h(qbReg[0])
        circ.h(qbReg[1])
        circ.cz(qbReg[0], qbReg[1])

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
        circ.h(qbReg[0])
        circ.h(anReg[1])
        circ.cz(qbReg[0], anReg[1])
        circ.h(qbReg[0])
        circ.h(anReg[1])
        circ.cz(qbReg[0], anReg[1])
        circ.h(qbReg[0])
        circ.h(anReg[1])
        circ.cz(qbReg[0], anReg[1])

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
        include_ancillas (str): Where and if to include any ancillas in the
                                density matrix.

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


def get_distance_1_basis():
    """Returns a basis set for the subspace of states with a distance one from |0>_L.
    This is equivalent to a distance two from |1>_L.

    The states are eigenvectors to the stabilizers with a syndrome that is given 
    by their index+1 in binary.
    """

    logical_0 = Statevector(logical_states(None)[0])

    weight_1 = [logical_0]*15

    for syndrome in range(1, 16):  # Loop over the 15 non-trivial errors
        for correction_strategy in syndrome_table[syndrome]:
            weight_1[syndrome-1] = weight_1[syndrome -
                                            1].evolve(correction_strategy[0](), [correction_strategy[1]])

    return weight_1


def get_distance_2_basis():
    """Returns a basis set for the subspace of states with a distance one from |0>_L.
    This is equivalent to a distance two from |1>_L.

    The states are eigenvectors to the stabilizers with a syndrome that is given 
    by their index+1 in binary.
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


def get_full_syndrome_basis():
    """Returns a set of basis vectors for the entire  2^5=32 dimensional Hilbert
    space that are eigenstates to the stabilizers.  

    Index 0 gives |0>_L, 1-16 gives distance 1 states
    Returns:
        list[Statevector]: List of statevectors.
    """
    logical = logical_states(None)
    weight_1 = get_distance_1_basis()
    weight_2 = get_distance_2_basis()

    # Calculate table of how each of the 32 different basis states (labeled by syndrome plus Z_L) map onto eachother from the 16 corrections
    return [Statevector(logical[0]), *weight_1,
            Statevector(logical[1]), *weight_2]
# %% All unflagged stabilizers


def get_stabilizer_cycle(registers, reset=True, recovery=False,
                               current_cycle=0, num_ancillas=None,
                               include_barriers=True, pipeline=False, snapshot=True,
                               snapshot_type='dm', conditional=False, pauliop='ZZZZZ',
                               include_fifth_stabilizer=False, **kwargs):
    """Run all four stabilizers, as well as an optional
    recovery.

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
            anQb_list = [registers.AncillaRegister[n]
                         for n in np.arange(1, num_stabilizers+1)]
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
        stabilizer_list = [_get_stabilizer_XZZXI,
                           _get_stabilizer_IXZZX,
                           _get_stabilizer_XIXZZ,
                           _get_stabilizer_ZXIXZ,
                           _get_stabilizer_ZZXIX]
    else:
        stabilizer_list = [_get_stabilizer_XZZXI,
                           _get_stabilizer_IXZZX,
                           _get_stabilizer_XIXZZ,
                           _get_stabilizer_ZXIXZ]

    # Create list of syndrome bits
    if isinstance(registers.SyndromeRegister, list):
        syn_reg = registers.SyndromeRegister[current_cycle]
        syn_bit_list = [syn_reg[n] for n in range(num_stabilizers)]
    else:
        syn_bit_list = [registers.SyndromeRegister[n]
                        for n in range(num_stabilizers)]

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

    # Add an extra measurement to the next syndrome register. Only needed if
    # ancilla is not reset between cycles (by default it DOES reset)
    # TODO: Make this compatible with using more than 1 ancilla
    if recovery and not reset:
        if current_cycle < len(registers.SyndromeRegister)-1:
            circ.measure(anQb_list[-1],
                         registers.SyndromeRegister[current_cycle+1][4])
            if include_barriers:
                circ.barrier()

    # Recovery
    if recovery is True:

        # TODO: Make this work for all setting (e.g. conditional=True)
        # add_snapshot_to_circuit(circ, snapshot_type='dm',
        #                         qubits=registers.QubitRegister, conditional=False, include_barriers=include_barriers, pauliop=pauliop)

        if not include_barriers:  # Always put a barrier here, so add an extra one if there wasn't one after the last stabilzier
            circ.barrier()
        circ.compose(get_recovery(
            registers, reset, current_cycle), inplace=True)
    return circ


def _get_stabilizer_XZZXI(registers, anQb=None, syn_bit=None, reset=True):
    """Gives the circuit for running the regular XZZXI stabilizer.

    Args:
        registers (StabilizerRegister): Register object
        anQb (AncillaQubit, optional): Specifies the ancilla to use for the measurement. Defaults to None.
        reset (bool, optional): Whether to reset ancillas between measurements. Defaults to True.
        current_cycle (int, optional): [description]. Defaults to 0.

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


def _get_stabilizer_IXZZX(registers, anQb=None, syn_bit=None, reset=True):
    """Gives the circuit for running the regular IXZZX stabilizer.
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


def _get_stabilizer_XIXZZ(registers, anQb=None, syn_bit=None, reset=True):
    """Gives the circuit for running the regular XIXZZ stabilizer.
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


def _get_stabilizer_ZXIXZ(registers, anQb=None, syn_bit=None, reset=True):
    """Gives the circuit for running the regular ZXIXZ stabilizer.
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
def _get_stabilizer_ZZXIX(registers, anQb=None, syn_bit=None, reset=True):
    """Gives the circuit for running the regular ZXIXZ stabilizer.
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

# %% Double diamond stabilizer cycle


def transpiled_dd_cycle(registers, current_cycle=0, reset=True,
                        recovery=False):
    """Gives the circuit for a full stabilizer cycle following the double 
    diamond connectivity.
    """
    # Create list of syndrome bits
    if isinstance(registers.SyndromeRegister, list):
        syn_reg = registers.SyndromeRegister[current_cycle]
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
        circ.compose(get_recovery(
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


def get_recovery(registers, reset=True, current_cycle=0):
    """Lookup table for recovery from a
    single qubit error on code qubits"""

    # Create a circuit
    qbReg = registers.QubitRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)

    # Select the correct syndrome register
    if isinstance(clReg, list):
        syndrome_reg = clReg[current_cycle]
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


# %% Internal testing of functions above
if __name__ == "__main__":
    # The settings for our circuit
    kwargs = {
        'n_cycles': 2,
        'reset': True,
        'recovery': True,
        'encoding': False,
        'conditional': True,
        'include_barriers': True,
        'generator_snapshot': True,
        'idle_snapshots': 1,
        'final_measure': True}

    # Define our registers
    qb = QuantumRegister(5, 'code_qubit')
    an = AncillaRegister(2, 'ancilla_qubit')
    cr = ClassicalRegister(4, 'syndrome_bit')  # The typical register
    # cr = get_classical_register(n_cycles) # Advanced list of registers
    readout = ClassicalRegister(5, 'readout')

    registers = StabilizerRegisters(qb, an, cr, readout)
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
