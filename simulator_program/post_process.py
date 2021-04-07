# %% Imports
from qiskit.circuit.library import XGate, ZGate
from qiskit import QuantumCircuit, execute, Aer
import numpy as np
from .post_select import get_subsystem_counts_up_to_cycle
# %%
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

# %% Misc functions

def _filter_readout_errors_v1(syndromes, P_e, P_r):
    # TODO: Make this?
    pass

def _get_new_syndromes(syndromes):
    """Convert a list of syndromes (from cycles of stabilizer measurements)
    into the corresponding list of syndromes for the errors that were applied at every cycle
    to cause the total syndrome."""
    return [syndromes[0]] + [syndromes[i-1] ^ syndromes[i]
                             for i in range(1, len(syndromes))]

# %% Post processing statevectors
def post_process_statevec(statevector, syndromes):
    """Version two of post-processing. Takes one statevector and it corresponding syndrome,
    applies post-processing and returns a corrected statevector.

    Processes the syndromes using XOR to find where new errors are introduced.

    Args:
        statevector (list): comples statevector
        syndromes (list(int)): list of syndromes as integers ('0'th element corresponds to 'stabilizer_' etc)

    Returns:
        List[float]: the processed statevector
    """

    # Convert to only new syndromes
    syndromes = _get_new_syndromes(syndromes)

    circ = QuantumCircuit(int(np.log2(statevector.shape[0])))
    circ.initialize(statevector, [circ.qubits[i]
                                  for i in range(7)])  # [2,3,4,5,6,0,1]

    for syndrome in syndromes:
        for correction_strategy in syndrome_table[syndrome]:
            circ.append(correction_strategy[0](), [correction_strategy[1]])
    results = execute(
        circ,
        Aer.get_backend('statevector_simulator'),
    ).result()

    return results.get_statevector()


def post_process_statevec_all_shots(results, n_cycles):
    """Wrapper for post_process_statvec that processes every statevector for a repeated stabilizer cycle
    simulation with memory=True.

    Args:
        results (Results object): results from completed simulation
        n_cycles (int): number of stabilizer cycles in simulated circuit

    Returns:
        List[List[float]]: List of corrected statvectors where first index indicates cycle, second shots
    """

    # TODO: If this crashes a lot, make it yield instead of return? (Or just return fidelities instead)
    mem = results.get_memory()
    shots = len(mem)
    statevector_dim = results.data(
    )['snapshots']['statevector']['stabilizer_0'][0].shape[0]
    # TODO: Make this format same as with in other places?
    post_processed_states = np.empty(
        (n_cycles, shots, statevector_dim), dtype=np.complex_)
    for current_cycle in range(n_cycles):
        statevectors = results.data(
        )['snapshots']['statevector']['stabilizer_' + str(current_cycle)]
        assert shots == len(statevectors)
        for shot in range(shots):
            # Convert the text format to integers. The order is from right to left, and the last entry is for final measurements ans should be removed
            syndromes = [int(syn, 2) for syn in reversed(
                mem[shot].split()[-(1+current_cycle):])]
            post_processed_states[current_cycle][shot] = post_process_statevec(
                statevectors[shot], syndromes)

    return post_processed_states

# %% Post processing density matrises

def get_unitary_matrix_for_correction(syndromes, include_ancillas=False):
    """Creates the unitary matrix corresponding to the correction procedure for a syndrome.
    include_ancillas adds two additional qubits to the unitary matrix, their states are not affected.
    """
    # TODO: Using DensityMatrix.evolve(other, qargs=None) -> DensityMatrix, this func can return a circ instead
    # TODO: cache this function? Probably not needed tbh
    # TODO: Make this work for many syndromes? Or combine them afterwards?
    dim = 5  # TODO: make it work for e.g. Stean code
    if include_ancillas:
        dim = 7
    circ = QuantumCircuit(dim)
    for syndrome in syndromes:
        for correction_strategy in syndrome_table[syndrome]:
            circ.append(correction_strategy[0](), [correction_strategy[1]])

    results = execute(
        circ,
        Aer.get_backend('unitary_simulator'),
    ).result()
    return np.matrix(results.get_unitary())


def apply_unitary_to_den_mat(density_matrix, unitary):
    """ Apply the (time) evolution defined by unitary to the density matrix"""
    # NOTE: Use DensityMatrix.evolve(other, qargs=None) -> DensityMatrix instead? 
    return unitary @ density_matrix @ unitary.H


def split_mem_into_syndromes(memory, current_cycle, cl_reg_size=4):
    """Splits the memory from density matrix snapshot, written as a string of a hexadecimal number, 
    into current_cycle + 1 groups of cl_reg_size bits and convert to integers."""

    return [(int(memory, 16) >> cl_reg_size*cycle) %
            2**cl_reg_size for cycle in range(current_cycle+1)]


def get_syndromes_den_mat(memory, current_cycle):
    """Returns the integer syndromes for each new error detected a every cycle up to current_cycle.

    Args:
        memory (str): memory from a density matrix snapshot taken a current_cycle, written as a string of a hexadecimal number
        current_cycle (int): current cycle in a repeated stabilizer measurement circuit

    Returns:
        list[int]: list of syndromes corresponding to the new errors detected at every cycle
    """

    # Split the hex number into groups of cl_reg_size bit numbers and covert to int
    syndromes = split_mem_into_syndromes(memory, current_cycle)

    # Convert the (running) total syndromes to new syndromes
    syndromes = _get_new_syndromes(syndromes)
    return syndromes

    # bin(int(memory, 16))[2:].zfill(
    #                 cl_reg_size*(current_cycle+1))


def post_process_den_mat(den_mat, memory, current_cycle):
    """Returns the post-processed density matrix where errors detected 
    in the corresponding memory are corrected."""

    syndromes = get_syndromes_den_mat(memory, current_cycle)
    correction_unitary = get_unitary_matrix_for_correction(syndromes)
    return apply_unitary_to_den_mat(den_mat, correction_unitary)

def get_states_and_counts_in_cycle(results, current_cycle, post_process=True):
    """Generator for tuples of states and their counts for the density matrix snapshots at current cycle in results"""

    subsystem_counts = get_subsystem_counts_up_to_cycle(
        results.get_counts(), current_cycle)
    for selected_state in results.data()['snapshots']['density_matrix']['stabilizer_' + str(current_cycle)]:
        den_mat = selected_state['value']
        memory = selected_state['memory']
        state = post_process_den_mat(
            den_mat, memory, current_cycle) if post_process else den_mat
        yield (state, subsystem_counts[int(memory, 16)])

def get_states_and_counts(results, n_cycles, post_process=True):
    """Generator for tuples of states and their counts for the density matrix snapshots for each cycle.
    This is a double generator, use it as if it returned a 2D list where the first index specifies
    the cycle, and the resulting 'list' contains pairs of density matrices and corresponding counts"""

    # NOTE: Should this be responsible for post_process = False? Can you split it into several funcs?
    for current_cycle in range(n_cycles):
        yield get_states_and_counts_in_cycle(results, current_cycle, post_process)


# NOTE: I made the function yield instead, if it doesn't work revert to the code below, to be removed
# def get_states_and_counts(results, n_cycles, post_process=True):
#     # NOTE: Should this be responsible for post_process = False? Can you split it into several funcs?
#     # NOTE: If this becomes too heavy (since it duplicates the amount of memory needed to store the matrices)
#     # then maybe have it yield one at a time instead of returning?
#     # Or re-implement the fidelity calculation commented out below

#     # running_fid = []
#     states_and_counts = [None]*n_cycles
#     for current_cycle in range(n_cycles):
#         counts = get_subsystem_counts_up_to_cycle(
#             results.get_counts(), current_cycle)
#         # fid = 0
#         tmp = []
#         for selected_state in results.data()['snapshots']['density_matrix']['stabilizer_' + str(current_cycle)]:
#             den_mat = selected_state['value']
#             memory = selected_state['memory']
#             # fid += state_fidelity(correct_state, post_process_den_mat(den_mat, memory, current_cycle))*counts[int(memory,16)]
#             state = post_process_den_mat(
#                 den_mat, memory, current_cycle) if post_process else den_mat
#             tmp.append((state, counts[int(memory, 16)]))
#         # running_fid.append(fid)
#         states_and_counts[current_cycle] = tmp
#     return states_and_counts


# %Code to test above, to be removed
if __name__ == "__main__":
    from qiskit import QuantumRegister, AncillaRegister
    from stabilizers import *
    from custom_noise_models import thermal_relaxation_model
    from post_select import get_trivial_state
    from qiskit.quantum_info import DensityMatrix
    reset = True
    recovery = False
    flag = False
    n_cycles = 3
    qb = QuantumRegister(5, 'code_qubit')
    an = AncillaRegister(2, 'ancilla_qubit')
    # cr = ClassicalRegister(4, 'syndrome_bit') # The typical register
    cr = get_classical_register(n_cycles, reset=reset,
                                recovery=recovery, flag=False)
    readout = ClassicalRegister(5, 'readout')

    registers = StabilizerRegisters(qb, an, cr, readout)

    # circ = get_empty_stabilizer_circuit(registers)

    circ = encode_input_v2(registers)
    circ.snapshot('post_encoding', 'density_matrix')
    # Stabilizer
    # circ.x(qb[3])
    circ += unflagged_stabilizer_cycle(registers,
                                       reset=reset,
                                       recovery=recovery
                                       )
    # circ.barrier()
    # circ.append(Snapshot('stabilizer_0', 'density_matrix', num_qubits=5), qb)
    # circ.draw()
    circ += get_repeated_stabilization(registers, n_cycles=n_cycles,
                                       reset=reset, recovery=recovery, flag=flag, snapshot_type='density_matrix')

    n_shots = 10
    results = execute(
        circ,
        Aer.get_backend('qasm_simulator'),
        noise_model=thermal_relaxation_model(),
        shots=n_shots
    ).result()

    correct_state = logical_states(include_ancillas=None)[0]
    running_fid = []
    for current_cycle in range(n_cycles):
        print("\nCycle ", current_cycle)
        counts = get_subsystem_counts_up_to_cycle(
            results.get_counts(), current_cycle)
        fid = 0
        count_sum = 0
        for selected_state in results.data()['snapshots']['density_matrix']['stabilizer_' + str(current_cycle)]:
            den_mat = selected_state['value']
            memory = selected_state['memory']
            fid += state_fidelity(correct_state, post_process_den_mat(
                den_mat, memory, current_cycle))*counts[int(memory, 16)]
            count_sum += counts[int(memory, 16)]
            print(memory)
            print(counts[int(memory, 16)])
            print(state_fidelity(correct_state, post_process_den_mat(
                den_mat, memory, current_cycle)))
        running_fid.append(fid/n_shots)
    # print(results.get_counts())
    # print(results.data()['snapshots']['density_matrix']['stabilizer_0'])
    # print(state_fidelity(correct_state, den_mat))
    # print(state_fidelity(correct_state, post_process_den_mat(den_mat, memory, 0)))
    # %
    # fidelities = [state_fidelity(post_selected_state, correct_state) for post_selected_state
    #               in get_trivial_post_select_den_mat(results, n_cycles)]
    # select_counts = get_trivial_post_select_counts(
    #     results.get_counts(), n_cycles)
    from matplotlib import pylab as plt

    fig, axs = plt.subplots(1, figsize=(14, 10))
    ax1 = axs
    # ax2 = axs[1]

    ax1.plot(range(n_cycles), running_fid, 'o-', label='No transpilation')
    ax1.set_xlabel(r'Error detection cycle $n$')
    ax1.set_ylabel('Post selected count')
    ax1.legend()
    ax1.grid(linewidth=1)

    # ax2.plot(range(n_cycles), select_counts, 'o-', label='No transpilation')
    # ax2.set_xlabel(r'Error detection cycle $n$')
    # ax2.set_ylabel(r'Post select fraction')
    # ax2.legend()
    # ax2.grid(linewidth=1)
