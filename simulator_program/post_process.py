"""
File for performing post-processing of errors. Set conditional=True and recovery=False in you simulation settings.
For an example implementation see the end of this file.
"""

# %% Imports
if __package__:
    from .post_select import get_subsystem_counts_up_to_cycle
    from .stabilizers import get_snapshot_label, syndrome_table
else:
    from post_select import get_subsystem_counts_up_to_cycle
    from stabilizers import get_snapshot_label, syndrome_table

from qiskit.circuit.library import XGate, ZGate
from qiskit import QuantumCircuit, execute, Aer
import numpy as np



# %% Misc functions

def _get_new_syndromes(syndromes):
    """Convert a list of syndromes (from cycles of stabilizer measurements)
    into the corresponding list of syndromes for the errors that were applied at every cycle
    to cause the total syndrome.
    
    E.g. [1001,1111] -> [1001,0110]
    """
    return [([0]+syndromes)[i] ^ syndromes[i]
                             for i in range(0, len(syndromes))]

# Coversion of no-reset syndromes into reset equivalents
conversion_table = [0, 3, 6, 5, 12, 15, 10, 9, 8, 11, 14, 13, 4, 7, 2, 1]
def _remap_syndromes(syndromes):
    """Remaps the syndromes of no-reset onto their reset equivalence."""
    # Remap syndromes as if previous cycle always ended in |0> (Still without reset)
    for i in reversed(range(1,len(syndromes))):
        if syndromes[i-1] >= 8: # If value is >=8, the last stabilizer measured |1>
            syndromes[i] = 15-syndromes[i]

    # Remap syndromes from "no-reset" to their reset equivalents
    return [conversion_table[syndrome] for syndrome in syndromes]

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
            2**cl_reg_size for cycle in range(current_cycle)]


def get_syndromes_den_mat(memory, current_cycle, reset=True):
    """Returns the integer syndromes for each new error detected a every cycle up to current_cycle.

    Args:
        memory (str): memory from a density matrix snapshot taken a current_cycle, written as a string of a hexadecimal number
        current_cycle (int): current cycle in a repeated stabilizer measurement circuit
        reset (bool): Whether circuit is run with ancilla resets
    Returns:
        list[int]: list of syndromes corresponding to the new errors detected at every cycle
    """

    # Split the hex number into groups of cl_reg_size bit numbers and covert to int
    syndromes = split_mem_into_syndromes(memory, current_cycle)

    # Re-map the syndromes to those with resetting ancilla
    if not reset:
        syndromes = _remap_syndromes(syndromes)

    # Convert the (running) total syndromes to new syndromes
    syndromes = _get_new_syndromes(syndromes)
    return syndromes

    # bin(int(memory, 16))[2:].zfill(
    #                 cl_reg_size*(current_cycle+1))


def post_process_den_mat(den_mat, memory, current_cycle, reset=True):
    """Returns the post-processed density matrix where errors detected 
    in the corresponding memory are corrected."""

    syndromes = get_syndromes_den_mat(memory, current_cycle, reset)
    correction_unitary = get_unitary_matrix_for_correction(syndromes)
    return apply_unitary_to_den_mat(den_mat, correction_unitary)

def get_states_and_counts_in_cycle(results, current_cycle, post_process=True,
                                   reset=True):
    """Generator for tuples of states and their counts for the density matrix snapshots at current cycle in results"""

    subsystem_counts = get_subsystem_counts_up_to_cycle(
        results.get_counts(), current_cycle)
    states_cycle = results.data()[get_snapshot_label('dm',True,current_cycle)]
    for memory in states_cycle:
        den_mat = states_cycle[memory]
        state = post_process_den_mat(
            den_mat, memory, current_cycle, reset) if post_process else den_mat
        yield (state, subsystem_counts[int(memory, 16)])

def get_states_and_counts(results, n_cycles, post_process=True, reset=True):
    """Generator for tuples of states and their counts for the density matrix snapshots for each cycle.
    This is a double generator, use it as if it returned a 2D list where the first index specifies
    the cycle, and the resulting 'list' contains pairs of density matrices and corresponding counts"""

    # NOTE: Should this be responsible for post_process = False? Can you split it into several funcs?
    for current_cycle in range(n_cycles+1):
        yield get_states_and_counts_in_cycle(results, current_cycle, post_process, reset)

def get_av_fidelities(states_and_counts, correct_state, n_shots):
    av_fidelities = []
    for cycle in states_and_counts:
        fid = 0
        for state, counts in cycle:
            fid += state_fidelity(state, correct_state)*counts
        av_fidelities.append(fid/n_shots)
    return av_fidelities

#%% Verify that post-processing is equivalent to normal QEC
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from stabilizers import get_full_stabilizer_circuit, logical_states
    from idle_noise import add_idle_noise_to_circuit
    from data_analysis_tools import default_execute, fidelity_from_scratch
    from qiskit.quantum_info.states.measures import state_fidelity
    from post_select import get_trivial_post_select_den_mat, get_trivial_post_select_counts

    # Simulation settings
    reset = True
    recovery = False
    n_cycles = 4
    n_shots = 1024/2

    # Build a stabilizer circuit
    circ = get_full_stabilizer_circuit(n_cycles=n_cycles, reset=reset,
                                    recovery=recovery, 
                                    snapshot_type='dm',
                                    conditional=True,
                                    encoding=False, theta=0, phi=0)
    circ, time = add_idle_noise_to_circuit(circ, return_time=True)

    # Run simulation
    results = default_execute(
        circ, n_shots)

    # Post-process the reuslts
    correct_state = logical_states(include_ancillas=None)[0]
    fidelities_normal = get_av_fidelities(get_states_and_counts(
        results, n_cycles, post_process=False), correct_state, n_shots)
    fidelities_post_process = get_av_fidelities(get_states_and_counts(
        results, n_cycles, post_process=True), correct_state, n_shots)

    # Post-select the results
    fidelities_select = [state_fidelity(post_selected_state, correct_state) for post_selected_state
                        in get_trivial_post_select_den_mat(results, n_cycles)]
    select_counts = get_trivial_post_select_counts(
        results.get_counts(), n_cycles)

    # Rerun simulation with recovery=True to demonstrate active error correction
    recovery = True
    fidelities_QEC, times = fidelity_from_scratch(
        n_cycles, n_shots, gate_times={'feedback': 0}, encoding=False, transpile=False)

    # Plotting results
    fig, axs = plt.subplots(2, figsize=(14, 10))
    ax1 = axs[0]
    ax2 = axs[1]

    # Plot 1: Average fidelity as a function of # stabilizer cycles
    ax1.plot(range(n_cycles+1), fidelities_normal, 'o-', label='No processing')
    ax1.plot(range(n_cycles+1), fidelities_select, 'o-', label='Post select')
    ax1.plot(range(n_cycles+1), fidelities_post_process,
            'o-', label='Post process')
    ax1.plot(range(n_cycles+1), fidelities_QEC, 'o-', label='QEC')
    ax1.set_xlabel(r'Error detection cycle $n$')
    ax1.set_ylabel('Average state fidelity')

    ax1.legend()
    ax1.grid(linewidth=1)

    # Plot 2: The number of remaining runs with post-selection.
    ax2.plot(range(n_cycles+1), select_counts, 'o-', label='No transpilation')
    ax2.set_xlabel(r'Error detection cycle $n$')
    ax2.set_ylabel(r'Post select count')
    ax2.legend()
    ax2.grid(linewidth=1)
    plt.show()
# %%
