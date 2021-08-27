"""
This file contains code for investigating the decoding errors. It is showcased in decoding_errors.ipynb.
"""
# %%
import numpy as np
from matplotlib import colors as clrs  # TODO: Fix
from matplotlib import pyplot as plt
from qiskit.quantum_info.states.measures import state_fidelity

from simulator_program import post_select, post_process
from simulator_program.data_analysis_tools import (default_simulate_persist_to_file,
                                                   check_if_saved,
                                                   print_saved_runs,
                                                   overlap_with_subspace)
from simulator_program.stabilizers import (
    get_distance_1_basis,
    get_distance_2_basis,
    get_full_syndrome_basis,
    get_snapshot_label,
    logical_states,
    syndrome_table)

# %% Setup

# Calculate table of how each of the 32 different basis states (labeled by syndrome plus Z_L) map onto eachother from the 16 corrections
logical = logical_states(None)
weight_1 = get_distance_1_basis()
weight_2 = get_distance_2_basis()
basis = get_full_syndrome_basis()
# overlap = np.empty(16)


def get_basis_mapping_table():
    basis_mapping_table = np.zeros((16, 32), dtype=int)
    for basis_index in range(32):
        for correction in range(16):
            state = basis[basis_index]
            for syndrome in syndrome_table[correction]:
                state = state.evolve(syndrome[0](), [syndrome[1]])
            for i in range(32):
                if state_fidelity(state, basis[i]):
                    basis_mapping_table[correction, basis_index] = i
                    break
    return basis_mapping_table


def reduce_key(key: str, current_snapshot: int, measurements_per_snapshot=1):
    """Cuts of the initial bits of a key to get the equivalent one at earlier cycles. 

    E.g. \n
    current_cycle = 1 \n
    '0xff' -> '11111111' -> '1111' - > '0xf'  \n
    current_cycle = 3 \n
    '0xff' -> '11111111' -> '11111111' -> '0xff' \n

    Args:
        key (str): Hex key for the measurement outcome
        current_snapshot (int): Index of the snapshot
        measurements_per_snapshot (int, optional): Number of measurements (to different cl registers) that takes place in between every snapshot. Defaults to one. If only one snap is take every cycle, set this to the number of stabilizers.

    Returns:
        reduced key (str): Hex key matching the current cycle.
    """
    return hex(int(bin(int(key, 16))[2:][:measurements_per_snapshot*current_snapshot].zfill(1), 2))


def extract_syndrome(key, cycle, num_stab_gens=4):
    """Extracts the 4-bit string that corresponds to the syndrome of a specific cycle for the given key.
    That is, the one that would normally be used to perform correction in the given cycle.

    E.g. 
    cycle = 0, num_stab_gens = 5, key = 24 = int('11000',2) -> 8 = int('1000',2)
    cycle = 1, num_stab_gens = 4, key = 136 = int('11001000',2) -> 12 = int('1100',2)
    cycle = 1, num_stab_gens = 5, key = 920 = int('1110000000',2) -> 12 = int('1100',2)

    Args:
        key (int): Integer specifying measurement being conditioned on.
        cycle ([type]): [description]
        num_stab_gens ([type]): [description]
    """
    return (key % (2**(num_stab_gens*(cycle))*2**4))//2**(num_stab_gens*cycle)


""" Below are functions for computing which snapshot is which. 
Formula:
There is one snap at the start (post encoding)
One after each cycle
One after each generator measurement (num_stab_gens in tot) 
And idle_snapshots before of after the stabilizers
"""


def get_cycle_indices(n_cycles, idle_snapshots, idle_delay='before', generator_snapshot=True, include_fifth_stabilizer=False, **kwargs):
    """Returns index of the end of each cycle (final stabilizer measurement)."""
    num_stab_gens = 4 + include_fifth_stabilizer
    num_snaps_per_cycle = 1+idle_snapshots + \
        generator_snapshot*num_stab_gens
    cycle_end_indices = np.array(
        [num_snaps_per_cycle*(cycle+1) for cycle in range(n_cycles)])
    if idle_delay == 'after':
        cycle_end_indices -= idle_snapshots
    return cycle_end_indices


def get_stab_datapoints(n_cycles, idle_snapshots, generator_snapshot, idle_delay='before', include_fifth_stabilizer=False, **kwargs):
    """Returns indices of the snapshots of stabilizer generators. If these are disables, it instead
    returns the index of the end of the cycle.

    Returns:
        [list]: Indices of the stabilizer measurement snapshots.
    """
    num_stab_gens = 4 + include_fifth_stabilizer

    snaps_per_cycle = 1+idle_snapshots + \
        generator_snapshot*num_stab_gens
    if generator_snapshot:
        stab_indices_in_cycle = [i+1 for i in range(num_stab_gens)]
        # stab_indices_in_cycle = [i+1 for i in range(num_stab_gens)] if idle_delay=='after' else [i+1+idle_snapshots for i in range(num_stab_gens)]
    else:
        stab_indices_in_cycle = [1]
    if idle_delay == 'before':
        stab_indices_in_cycle = [value + idle_snapshots
                                 for value in stab_indices_in_cycle]

    stab_indices = []
    for cycle in range(0, n_cycles):
        stab_indices += [value + snaps_per_cycle *
                         (cycle) for value in stab_indices_in_cycle]
    return stab_indices


colors_def = plt.get_cmap("tab10")


def plot_colors(i):
    if i == 1:
        i = 2
    elif i == 2:
        i = 1
    return colors_def(i)
# def plot_colors(i):
#     return get_cmap('Spectral_r')(50*i+20)


def plot_overlap(results, times, n_cycles, idle_snapshots, generator_snapshot, title=None, idle_delay='before', **kwargs):

    conditional = True if 'dm_con_0' in results.data() else False
    num_snapshots = 1
    while True:
        if not get_snapshot_label('dm', conditional, num_snapshots) in results.data():
            break
        num_snapshots += 1

    time = np.empty(num_snapshots)
    P_0 = np.empty(num_snapshots)
    P_1 = np.empty(num_snapshots)
    P_w1 = np.empty(num_snapshots)
    P_w2 = np.empty(num_snapshots)
    # P = np.empty(num_snapshots)
    # order = np.empty((32,num_snapshots))
    # TODO: Make work with delay snapshots for key other than '0x0'
    # (can't accept measurements less often than once per snapshot)
    key = '0x0'
    for i in range(num_snapshots):
        label = get_snapshot_label('dm', conditional, i)
        if conditional:
            rho = results.data()[label][reduce_key(key, i)]
        else:
            rho = results.data()[label]
        time[i] = times[label]/1000

        P_0[i] = overlap_with_subspace(rho, logical[0])
        P_1[i] = overlap_with_subspace(rho, logical[1])
        P_w1[i] = overlap_with_subspace(rho, weight_1)
        P_w2[i] = overlap_with_subspace(rho, weight_2)
        # P[i] = overlap_with_subspace(rho, logical)
        # for j in range(32):
        #     order[j,i] = state_fidelity(rho,basis[j])

    order = np.array([P_0, P_w1, P_w2, P_1])

    stab_datapoints = get_stab_datapoints(
        n_cycles, idle_snapshots, generator_snapshot, idle_delay=idle_delay)
    if conditional:
        counts_at_snapshots = post_select.get_trivial_post_select_counts_V2(
            results.get_counts(), stab_datapoints, num_snapshots)
    else:
        counts_at_snapshots = np.ones(num_snapshots)

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))

    for i in range(len(order)):
        lower = order[:i].sum(axis=0)
        ax.fill_between(time, (order[i]+lower)*counts_at_snapshots, lower*counts_at_snapshots,
                        color=np.array(plot_colors(i))-np.array([0, 0, 0, 0.3]))
    for i in range(len(order)):
        lower = order[:i].sum(axis=0)
        # Picks out the data points belonging to only the generators
        ax.plot(time[stab_datapoints], ((order[i]+lower) *
                                        counts_at_snapshots)[stab_datapoints], '.', color=plot_colors(i))

    if conditional:
        ax.set_ylabel('counts')
    else:
        ax.set_ylabel(r'Overlap with subspace')
    ax.set_xlabel('Time [μs]')
    # ax.set_xticks([])
    ax.set_ylim((0.5, counts_at_snapshots[0]))
    ax.set_xlim((0, time[-1]))
    ax.legend(labels=[r'$|0_L\rangle$', 'Distance 1 states',
                      'Distance 2 states', r'$|1_L\rangle$'])
    ax.set_title(title)
    fig.tight_layout()
    plt.show()
    # fig.savefig('subspace.pdf', transparent=True)


def plot_correction_matrix(results, previous_keys=None, overlap_cycle=None, backtrack_measurement=0, check_overlap_to_distance_1=False, print_fidelity=False, include_fifth_stabilizer=False, **kwargs):
    if previous_keys is None:
        previous_keys = []

    if overlap_cycle is None:
        overlap_cycle = len(previous_keys)

    num_stab_gens = 4 + include_fifth_stabilizer
    pre_recovery_index = get_cycle_indices(
        **kwargs)[overlap_cycle]-0 - backtrack_measurement
    counts = post_select.get_subsystem_counts_up_to_bit(
        results.get_counts(), num_stab_gens*(overlap_cycle+1)-backtrack_measurement)
    label = get_snapshot_label('dm', True, pre_recovery_index)
    # len(results.data()[label])
    num_keys = 2**(num_stab_gens*(1+overlap_cycle-len(previous_keys)))
    overlaps = np.zeros((num_keys, 34))+np.infty

    # resulting (average) fidelity if applying the 'normal' lookup table definition
    total_fid_lookup = 0
    # resulting (average) fidelity if applying the best single qubit correction
    total_fid_best_single_qb = 0
    # resulting (average) fidelity if applying the best arbitrary unitary
    total_fid_best_unitary = 0

    for key in results.data()[label]:
        for i in range(len(previous_keys)):
            if extract_syndrome(int(key, 16), i, num_stab_gens) != int(previous_keys[i], 2):
                break
        else:
            # Int version of the part of the key corresponding to the overlap_cycle
            key_at_cycle = extract_syndrome(
                int(key, 16), overlap_cycle, num_stab_gens)
            # Int version of the full key up to the current cycle
            # TODO: Should be unnecessary to cut of end?
            key_up_to_cycle = int(key, 16) % (
                2**(num_stab_gens*(overlap_cycle))*2**4)
            # TODO: write
            key_segment = (int(key, 16) % (
                2**(num_stab_gens*(overlap_cycle))*2**4))//2**(num_stab_gens*len(previous_keys))
            rho = results.data()[label][key]

            overlap = np.zeros(34)
            if check_overlap_to_distance_1 == False:
                for i in range(32):
                    overlap[i] = state_fidelity(basis[i], rho)
                overlap[32] = np.linalg.eigvalsh(rho)[-1]
            elif check_overlap_to_distance_1 == True:
                for i in range(32):
                    overlap[i] = overlap_with_subspace(
                        rho, [basis[j] for j in get_basis_mapping_table()[:16, i]])
                overlap[32] = np.sum(np.linalg.eigvalsh(rho)[-16:])
            overlap[33] = counts[key_up_to_cycle]
            overlaps[key_segment, :] = overlap

            if print_fidelity:
                fid_lookup = overlap[key_at_cycle]
                fid_best_single_qb = np.max(overlap[:32])
                fid_best_arbitrary_gate = overlap[32]

                total_fid_lookup += fid_lookup*counts[key_up_to_cycle]
                total_fid_best_single_qb += fid_best_single_qb * \
                    counts[key_up_to_cycle]
                total_fid_best_unitary += fid_best_arbitrary_gate * \
                    counts[key_up_to_cycle]

            # Print the difference between the assigned correction and the theoretical max
            # if fid_lookup<fid_best_single_qb:
            #     print(bin(key_int)[2:].zfill(num_stab_gens),':', fid_lookup,' -> ',fid_best_single_qb,' -> ',fid_best_arbitrary_gate)

    total_counts = np.sum(overlaps[:, 33][overlaps[:, 33] != np.infty])
    overlaps[:, 33] /= total_counts

    if print_fidelity:
        total_fid_lookup /= total_counts
        total_fid_best_single_qb /= total_counts
        total_fid_best_unitary /= total_counts

        # TODO: these don't seem to give quite the right result when conditioning
        print('Fidelity if using standard lookup table', total_fid_lookup)
        print('Fidelity if using optimal single qubit correction',
              total_fid_best_single_qb)
        print('Fidelity if using optimal arbitrary unitary',
              total_fid_best_unitary)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    aspect = 16/num_keys
    HM = ax.imshow(overlaps, aspect=aspect, interpolation='none')
    norm = clrs.Normalize(0, 1)
    HM.set_norm(norm)
    # threshold = np.zeros(num_keys)
    # for key in range(num_keys):
    #     threshold[key] = overlaps[key,extract_syndrome(key,overlap_cycle,num_stab_gens)] < 0.5
    # c = 'red'
    # for i in range(num_keys):
    #     if threshold[i] != 0:
    #         ax.plot([(extract_syndrome(i,overlap_cycle,num_stab_gens)-0.5),(extract_syndrome(i,overlap_cycle,num_stab_gens)+0.5)],[(i - 0.5),i - 0.5],c)
    #         ax.plot([(extract_syndrome(i,overlap_cycle,num_stab_gens)-0.5),(extract_syndrome(i,overlap_cycle,num_stab_gens)+0.5)],[i + 0.5,i + 0.5],c)
    #         ax.plot([(extract_syndrome(i,overlap_cycle,num_stab_gens)+0.5),(extract_syndrome(i,overlap_cycle,num_stab_gens)+0.5)],[i - 0.5,i + 0.5],c)
    #         ax.plot([(extract_syndrome(i,overlap_cycle,num_stab_gens)-0.5),(extract_syndrome(i,overlap_cycle,num_stab_gens)-0.5)],[i - 0.5,i + 0.5],c)

    ax.set_xticks(np.arange(34))
    ax.set_xticklabels([bin(key)[2:].zfill(4) for key in range(
        16)]+[bin(key)[2:].zfill(4) for key in range(16)]+['max']+['occup.'])
    y_ticks = np.arange(0, num_keys, num_keys//16)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([bin(key)[2:].zfill(num_stab_gens) for key in y_ticks])

    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")
    cbar0 = fig.colorbar(HM, ax=ax, orientation='horizontal',
                         fraction=.06, pad=0.25)
    fig.suptitle(
        "Fidelity to the 32 basis states conditioned on stabilizer measurements\n\
        Snapshot taken after measuring the stabilizers but before correction"
        + ("\nPost-selecting previous syndrome(s) "+str(*previous_keys) if overlap_cycle > 0 else ""))
    ax.set_xlabel('Basis states, labeled by their eigenvalues to the stabilizers\n' +
                  r"Left: distance $\leq$ 1 from $|0\rangle_L$. Right: distance $\geq$ 2 from $|0\rangle_L$.")
    ax.set_ylabel("Simulation state\n conditioned on stabilizer measurements")
    fig.tight_layout()
    plt.show()


def get_best_correction(results, n_cycles, backtrack_measurement=0, check_overlap_to_distance_1=False, include_fifth_stabilizer=False, **kwargs):

    num_stab_gens = 4 + include_fifth_stabilizer

    total_fid_lookup_list = np.empty(n_cycles+1)
    total_fid_lookup_last_cycle_list = np.empty(n_cycles+1)
    total_fid_best_single_qb_list = np.empty(n_cycles+1)
    total_fid_best_unitary_list = np.empty(n_cycles+1)
    for overlap_cycle in range(n_cycles+1):
        pre_recovery_index = np.append([0], get_cycle_indices(n_cycles,
                                                              **kwargs))[overlap_cycle] - backtrack_measurement
        # counts = post_select.get_subsystem_counts_up_to_bit(
        #     results.get_counts(), num_stab_gens*(overlap_cycle)-backtrack_measurement)
        counts = post_select.get_subsystem_counts_up_to_cycle(
            results.get_counts(), overlap_cycle)
        label = get_snapshot_label('dm', True, pre_recovery_index)

        basis_mapping_table = get_basis_mapping_table()
        # resulting (average) fidelity if applying the 'normal' lookup table definition
        total_fid_lookup = 0
        # resulting (average) fidelity if applying the correction for the last syndrome only
        total_fid_lookup_last_cycle = 0
        # resulting (average) fidelity if applying the best single qubit correction
        total_fid_best_single_qb = 0
        # resulting (average) fidelity if applying the best arbitrary unitary
        total_fid_best_unitary = 0

        total_counts = 0
        for key in results.data()[label]:
            # Int version of the full key up to the current cycle
            # TODO: Should be unnecessary to cut of end?
            key_up_to_cycle = int(key, 16) % (
                2**(num_stab_gens*(overlap_cycle)))
            rho = results.data()[label][key]

            overlap = np.zeros(33)
            if check_overlap_to_distance_1 == False:
                for i in range(32):
                    overlap[i] = state_fidelity(basis[i], rho)
                overlap[32] = np.linalg.eigvalsh(rho)[-1]
            elif check_overlap_to_distance_1 == True:
                for i in range(32):
                    overlap[i] = overlap_with_subspace(
                        rho, [basis[j] for j in basis_mapping_table[:16, i]])
                overlap[32] = np.sum(np.linalg.eigvalsh(rho)[-16:])
            total_counts += counts[key_up_to_cycle]

            combined_error = 0
            for new_syndrome in post_process.get_syndromes_den_mat(key, overlap_cycle, reset=True):
                combined_error = basis_mapping_table[new_syndrome,
                                                     combined_error]
            #     print(bin(extract_syndrome(
            #         int(key, 16), cycle, num_stab_gens)))
            # print('Becomes',bin(key_at_cycle),'\n')
            fid_lookup = overlap[combined_error]
            key_at_cycle = extract_syndrome(
                int(key, 16), overlap_cycle-1, num_stab_gens) if overlap_cycle > 0 else 0
            fid_lookup_last_cycle = overlap[key_at_cycle]
            fid_best_single_qb = np.max(overlap[:32])
            fid_best_arbitrary_gate = overlap[32]

            total_fid_lookup += fid_lookup*counts[key_up_to_cycle]
            total_fid_lookup_last_cycle += fid_lookup_last_cycle * \
                counts[key_up_to_cycle]
            total_fid_best_single_qb += fid_best_single_qb * \
                counts[key_up_to_cycle]
            total_fid_best_unitary += fid_best_arbitrary_gate * \
                counts[key_up_to_cycle]

        total_fid_lookup /= total_counts
        total_fid_lookup_last_cycle /= total_counts
        total_fid_best_single_qb /= total_counts
        total_fid_best_unitary /= total_counts

        total_fid_lookup_list[overlap_cycle] = total_fid_lookup
        total_fid_lookup_last_cycle_list[overlap_cycle] = total_fid_lookup_last_cycle
        total_fid_best_single_qb_list[overlap_cycle] = total_fid_best_single_qb
        total_fid_best_unitary_list[overlap_cycle] = total_fid_best_unitary
        print('cycle', overlap_cycle, 'done')
    return total_fid_lookup_list, total_fid_lookup_last_cycle_list, total_fid_best_single_qb_list, total_fid_best_unitary_list

#  Plot how the 32 different basis states (labeled by syndrome plus Z_L) map onto eachother from the 16 corrections


def plot_error_multiplication_table(different_gradient_for_each_state=False):
    # Gives every basis state it's one gradient
    mappings = get_basis_mapping_table()
    colored_mappings = np.zeros((16, 32))
    for basis_index in range(32):
        overlap = np.empty(16)
        for correction in range(16):
            if mappings[correction, basis_index] == 0:
                overlap[correction] = 0
            elif mappings[correction, basis_index] < 16:
                overlap[correction] = 0.5 + \
                    (mappings[correction,
                              basis_index]-1)/30*different_gradient_for_each_state
            elif mappings[correction, basis_index] == 16:
                overlap[correction] = 3
            else:
                overlap[correction] = 2.5 - \
                    (mappings[correction,
                              basis_index]-17)/30*different_gradient_for_each_state
            # if state_fidelity(state, logical[0]):
            #     overlap[correction] = 0
        colored_mappings[:, basis_index] = overlap

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    HM = ax.imshow(colored_mappings, aspect=1, interpolation='none',
                   cmap='Spectral_r', alpha=1)  # tab10
    # norm = clrs.Normalize(0, 10)
    # HM.set_norm(norm)
    ax.set_xticks(np.arange(32))
    ax.set_xticklabels([bin(key)[2:].zfill(4) for key in range(
        16)]+[bin(key)[2:].zfill(4) for key in range(16)])
    ax.set_yticks(np.arange(16))
    ax.set_yticklabels([bin(key)[2:].zfill(4) for key in range(16)])
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")
    # cbar0 = fig.colorbar(HM, ax=ax, orientation='horizontal',
    #             fraction=.06, pad=0.25)
    fig.suptitle("Which subspace each basis state gets mapped to by each correction operation\n" +
                 r"Dark blue: $|0\rangle_L$"+"\nLight blue: distance-1"+"\nOrange: distance-2\n"+r"Red: $|1\rangle_L$")
    ax.set_xlabel('Basis states, labeled by their eigenvalues to the stabilizers\n' +
                  r"Left: distance $\leq$ 1 from $|0\rangle_L$. Right: distance $\geq$ 2 from $|0\rangle_L$.")
    ax.set_ylabel("Correction operations, ordered by syndrome")
    fig.tight_layout()
    plt.show()


# %%
if __name__ == '__main__':
    from simulator_program.data_analysis_tools import default_simulate_persist_to_file
    from simulator_program.custom_noise_models import standard_times, standard_times_delay, GateTimes

    file_name = 'decoding_data/cache.dat'
    print_saved_runs(file_name)
    # kwargs = {k:v for k,v in {('reset', True), ('include_barriers', True), ('recovery', False), ('conditional', True), ('include_fifth_stabilizer', False), ('idle_snapshots', 2), ('generator_snapshot', True), ('encoding', False), ('n_cycles', 3), ('idle_delay', 'before'), ('final_measure', False)}}

    kwargs = {
        'n_cycles': 12,
        'reset': True,
        'recovery': False,
        'encoding': False,
        'conditional': True,
        'include_barriers': True,
        'generator_snapshot': False,
        'idle_snapshots': 0,
        'final_measure': False,
        'idle_delay': 'before',
        'include_fifth_stabilizer': False}

    T1 = [40e3]*5+[100000000000, 100000000000]
    T2 = [60e3]*5+[100000000000, 100000000000]
    # T1 = 40e3
    # T2 = 60e3
    # gate_times = standard_times_delay
    gate_times = standard_times  # .get_gate_times({'feedback':0})
    # gate_times = GateTimes(0, 0, {'delay': 3310})

    n_shots = 1024*4

    print('Is a run with these settings saved?', check_if_saved(
        kwargs, T1, T2, gate_times, file_name=file_name))

    # Get the complete circuit
    # Whether to ignore existing simulation data and overwrite with a new simulation
    save_to_file = False
    overwrite = False

    results, times = default_simulate_persist_to_file(
        kwargs, T1, T2, gate_times, n_shots, file_name=file_name, overwrite=overwrite, save_to_file=save_to_file)
    time = np.empty(kwargs['n_cycles']+1)
    for cycle, snap_index in enumerate(np.append([0], get_cycle_indices(**kwargs))):
        time[cycle] = times[get_snapshot_label('dm', True, snap_index)]

    plot_overlap(results, times, **kwargs)

    previous_keys = []  # Post select
    backtrack_measurement = 0
    # If this is enabled, check the overlap to the distance <= 1 subspace instead of just |0_L>
    check_overlap_to_distance_1 = False
    overlap_cycle = None
    if kwargs['conditional']:
        plot_correction_matrix(
            results, previous_keys, overlap_cycle, backtrack_measurement, check_overlap_to_distance_1, print_fidelity=True, **kwargs)
        total_fid_lookup_list, total_fid_lookup_last_cycle_list, total_fid_best_single_qb_list, total_fid_best_unitary_list = get_best_correction(
            results, **kwargs)
        with open('data/decoding_strategies.npy', 'wb') as f:
            np.save(f, time)
            np.save(f, total_fid_lookup_list)
            np.save(f, total_fid_lookup_last_cycle_list)
            np.save(f, total_fid_best_single_qb_list)
            np.save(f, total_fid_best_unitary_list)
    else:
        print('Not conditional')

    # %% Look at the proportion of states that get projected to |0>_L from w1 and to |1>_L from w2, are they the same?
    conditional = True if 'dm_con_0' in results.data() else False
    num_snapshots = get_cycle_indices(**kwargs)[-1]+1
    time = np.empty(num_snapshots)
    P_0 = np.empty(num_snapshots)
    P_1 = np.empty(num_snapshots)
    P_w1 = np.empty(num_snapshots)
    P_w2 = np.empty(num_snapshots)
    # P = np.empty(num_snapshots)
    # order = np.empty((32,num_snapshots))
    # TODO: Make work with delay snapshots for key other than '0x0'
    # (can't accept measurements less often than once per snapshot)
    key = '0x0'
    for i in range(num_snapshots):
        label = get_snapshot_label('dm', conditional, i)
        if conditional:
            rho = results.data()[label][reduce_key(key, i)]
        else:
            rho = results.data()[label]
        time[i] = times[label]/1000

        P_0[i] = overlap_with_subspace(rho, logical[0])
        P_1[i] = overlap_with_subspace(rho, logical[1])
        P_w1[i] = overlap_with_subspace(rho, weight_1)
        P_w2[i] = overlap_with_subspace(rho, weight_2)
        # P[i] = overlap_with_subspace(rho, logical)
        # for j in range(32):
        #     order[j,i] = state_fidelity(rho,basis[j])
    if kwargs['recovery'] and not kwargs['conditional']:
        for cycle_num in range(kwargs['n_cycles']):
            print('At cycle', cycle_num)
            cycle_index = get_cycle_indices(**kwargs)[cycle_num]
            print(r'Fraction of increase in |0>_L to weight 1',
                  (P_0[cycle_index]-P_0[cycle_index-1])/P_w1[cycle_index-1])
            print(r'Fraction of increase in |1>_L to weight 2',
                  (P_1[cycle_index]-P_1[cycle_index-1])/P_w2[cycle_index-1])
            print('')
        print('Given no decoding errors these should be equal')

    plot_error_multiplication_table(different_gradient_for_each_state=False)

# %%
