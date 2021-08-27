""" Functions for running the [[4,2,2]] steane code, mainly on a 5qb plus-shaped
chip. Most functions here have an equivalent one in stabilizers.py, for
the [[5,1,3]] code, which has better documentation. The functions here are
primarily used in 422_code_summary.ipynb, and is referred to for a rundown of the
code.
"""


# Disable error which gives false positives in this file
# pylint: disable=E1101
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

# Local files
from .stabilizers import (StabilizerRegisters,
                          add_snapshot_to_circuit,
                          )
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
