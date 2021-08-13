# %%
from qiskit.quantum_info.states.measures import state_fidelity
from simulator_program.idle_noise import add_idle_noise_to_circuit
from simulator_program.custom_noise_models import standard_times_delay, standard_times

from simulator_program.stabilizers import get_full_stabilizer_circuit, get_weight_1_basis, get_weight_2_basis, label_counter, get_snapshot_label, logical_states
import numpy as np
from matplotlib import pyplot as plt
from simulator_program.data_analysis_tools import default_execute, overlap_with_subspace
from simulator_program import post_process, post_select
from qiskit.quantum_info import Statevector, DensityMatrix
from simulator_program.stabilizers import syndrome_table
from simulator_program.custom_noise_models import thermal_relaxation_model_V2, standard_times_delay, GateTimes
from matplotlib import colors as clrs # TODO: Fix
# %% Setup
logical = logical_states(None)
weight_1 = get_weight_1_basis()
weight_2 = get_weight_2_basis()

# Calculate table of how each of the 32 different basis states (labeled by syndrome plus Z_L) map onto eachother from the 16 corrections
basis = [Statevector(logical[0]), *weight_1,
         Statevector(logical[1]), *weight_2]
basis_mapping_table = np.zeros((16, 32), dtype=int)
for basis_index in range(32):
    overlap = np.empty(16)
    for correction in range(16):
        state = basis[basis_index]
        for syndrome in syndrome_table[correction]:
            state = state.evolve(syndrome[0](), [syndrome[1]])
        for i in range(32):
            if state_fidelity(state, basis[i]):
                basis_mapping_table[correction, basis_index] = i
                break


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
    """Extracts the 4-bit syndrome that would normally be used to perform correction
    out of the key.

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
And idle_snapshots before of after the stabilziers
"""

def get_cycle_indices():
    """Returns index of the end of each cycle (final stabilizer measurement)."""
    num_snaps_per_cycle = 1+kwargs['idle_snapshots'] + \
        kwargs['generator_snapshot']*num_stab_gens
    cycle_end_indices = np.array(
        [num_snaps_per_cycle*(cycle+1) for cycle in range(kwargs['n_cycles'])])
    if kwargs['idle_delay'] == 'after':
        cycle_end_indices -= kwargs['idle_snapshots']
    return cycle_end_indices


def get_stab_datapoints():
    """Returns indices of the snapshots of stabilizer generators. If these are disables, it instead
    returns the index of the end of the cycle.

    Returns:
        [list]: Indices of the stabilizer measurement snapshots.
    """
    
    snaps_per_cycle = 1+kwargs['idle_snapshots'] + \
        kwargs['generator_snapshot']*num_stab_gens
    if kwargs['generator_snapshot']:
        stab_indices_in_cycle = [i+1 for i in range(num_stab_gens)]
        # stab_indices_in_cycle = [i+1 for i in range(num_stab_gens)] if kwargs['idle_delay']=='after' else [i+1+kwargs['idle_snapshots'] for i in range(num_stab_gens)]
    else:
        stab_indices_in_cycle = [1]
    if kwargs['idle_delay'] == 'before':
        stab_indices_in_cycle = [value + kwargs['idle_snapshots']
                                 for value in stab_indices_in_cycle]

    stab_indices = []
    for cycle in range(0, kwargs['n_cycles']):
        stab_indices += [value + snaps_per_cycle *
                                  (cycle) for value in stab_indices_in_cycle]
    return stab_indices


colors_def = plt.get_cmap("tab10")

def colors(i):
    if i == 1:
        i = 2
    elif i == 2:
        i = 1
    return colors_def(i)
# def colors(i):
#     return get_cmap('Spectral_r')(50*i+20)


# %% Simulation settings
kwargs = {
    'n_cycles': 2,
    'reset': True,
    'recovery': True,
    'encoding': False,
    'conditional': False,
    'include_barriers': True,
    'generator_snapshot': True,
    'idle_snapshots': 2,
    'final_measure': False,
    'idle_delay': 'before',
    'include_fifth_stabilizer': False}
# Define our registers (Maybe to be written as function?)

num_stab_gens = 4 + kwargs['include_fifth_stabilizer']

T1 = [40e3]*5+[100000000000, 100000000000]
T2 = [60e3]*5+[100000000000, 100000000000]
# T1 = 40e3
# T2 = 60e3
gate_times = standard_times_delay
# gate_times = standard_times
# gate_times = GateTimes(20, 0, {'delay': 5000})
# Get the complete circuit
circ = get_full_stabilizer_circuit(**kwargs)

circ, times = add_idle_noise_to_circuit(
    circ, gate_times, T1=T1, T2=T2, return_time=True)
# display(circ.draw(output='mpl'))


# %%
# Run it
noise_model = thermal_relaxation_model_V2(T1=T1, T2=T2, gate_times=gate_times)
n_shots = 1024*1
results = default_execute(
    circ, n_shots, gate_times=gate_times, noise_model=noise_model)

# %%
time = np.empty(label_counter.value)
P_0 = np.empty(label_counter.value)
P_1 = np.empty(label_counter.value)
P_w1 = np.empty(label_counter.value)
P_w2 = np.empty(label_counter.value)
# P = np.empty(label_counter.value)
# order = np.empty((32,label_counter.value))
# TODO: Make work with delay snapshots for key other than '0x0'
# (can't accept measurements less often than once per snapshot)
key = '0x0'
for i in range(label_counter.value):
    label = get_snapshot_label('dm', kwargs['conditional'], i)
    if kwargs['conditional']:
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

stab_datapoints = get_stab_datapoints()
if kwargs['conditional']:
    counts_at_snapshots = post_select.get_trivial_post_select_counts_V2(
        results.get_counts(), stab_datapoints, label_counter.value)
else:
    counts_at_snapshots = np.ones(label_counter.value)

for i in range(len(order)):
    lower = order[:i].sum(axis=0)
    plt.fill_between(time, (order[i]+lower)*counts_at_snapshots, lower*counts_at_snapshots,
                     color=np.array(colors(i))-np.array([0, 0, 0, 0.3]))
for i in range(len(order)):
    lower = order[:i].sum(axis=0)
    # Picks out the data points belonging to only the generators
    plt.plot(time[stab_datapoints], ((order[i]+lower) *
                                     counts_at_snapshots)[stab_datapoints], '.', color=colors(i))

if kwargs['conditional']:
    plt.ylabel('counts')
else:
    plt.ylabel('Overlap')
plt.xlabel('Time [μs]')
plt.ylim((0, counts_at_snapshots[0]))
plt.xlim((0, time[-1]))
plt.legend(labels=[r'$|0\rangle_L$', 'Weight 1', 'Weight 2', r'$|1\rangle_L$'])
# %% Look at the proportion of states that get projected to |0>_L from w1 and to |1>_L from w2, are they the same?
if kwargs['recovery'] and not kwargs['conditional']:
    for cycle_num in range(kwargs['n_cycles']):
        print('At cycle',cycle_num)
        cycle_index = get_cycle_indices()[cycle_num]
        print(r'Fraction of increase in |0>_L to weight 1',(P_0[cycle_index]-P_0[cycle_index-1])/P_w1[cycle_index-1])
        print(r'Fraction of increase in |1>_L to weight 2',(P_1[cycle_index]-P_1[cycle_index-1])/P_w2[cycle_index-1])
        print('')
    print('Given no decoding errors these should be equal')
# %% Test plotting all keys in second cycle starting with a specific syndrome in the first cycle

previuous_keys = []  # Post select
overlap_cycle = len(previuous_keys)
overlap_subspace = 0
if kwargs['conditional']:
    counts = post_select.get_subsystem_counts_up_to_cycle(
        results.get_counts(), overlap_cycle+1)
    pre_recovery_index = get_cycle_indices()[overlap_cycle]-1
    label = get_snapshot_label('dm', kwargs['conditional'], pre_recovery_index)
    num_keys = 2**(num_stab_gens*(1))  # len(results.data()[label])
    overlaps = np.zeros((num_keys, 34))

    # resulting (average) fidelity if applying the 'normal' lookup table definition
    total_fid_lookup = 0
    # resulting (average) fidelity if applying the best single qubit correction
    total_fid_best_single_qb = 0
    # resulting (average) fidelity if applying the best arbitrary unitary
    total_fid_best_unitary = 0

    for key in results.data()[label]:
        for i in range(overlap_cycle):
            if extract_syndrome(int(key, 16), i, num_stab_gens) != int(previuous_keys[i], 2):
                break
        else:
            # Int version of the part of the key corresponding to the overlap_cycle
            key_int = extract_syndrome(
                int(key, 16), overlap_cycle, num_stab_gens)
            key_up_to_cycle = int(key, 16) % (
                2**(num_stab_gens*(overlap_cycle))*2**4)
            rho = results.data()[label][key]

            overlap = np.empty(34)
            if overlap_subspace == 0:
                for i in range(32):
                    overlap[i] = state_fidelity(basis[i], rho)
                overlap[32] = np.linalg.eigvalsh(rho)[-1]
            elif overlap_subspace == 1:
                for i in range(32):
                    overlap[i] = overlap_with_subspace(
                        rho, [basis[j] for j in basis_mapping_table[:16, i]])
                overlap[32] = np.sum(np.linalg.eigvalsh(rho)[-16:])
            overlap[33] = counts[key_up_to_cycle]
            overlaps[key_int, :] = overlap

            fid_lookup = overlap[extract_syndrome(
                key_int, overlap_cycle, num_stab_gens)]
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

    overlaps[:, 33] /= np.sum(overlaps[:, 33])
    total_fid_lookup /= n_shots
    total_fid_best_single_qb /= n_shots
    total_fid_best_unitary /= n_shots

    print('Fidelity if using standard lookup table', total_fid_lookup)
    print('Fidelity if using optimal single qubit correction',
          total_fid_best_single_qb)
    print('Fidelity if using optimal arbitrary unitary', total_fid_best_single_qb)

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
        "Fidelity to the 32 basis states conditioned on stabilizer measurements\n red squares show where fid<0.5 (is that meaningful?)")
    ax.set_xlabel('Basis states, labeled by their eigenvalues to the stabilizers\n' +
                  r"Left: distance $\leq$ 1 from $|0\rangle$. Right: distance $\geq$ 2 from $|0\rangle$")
    ax.set_ylabel("Simulation state\n conditioned on stabilizer measurements")
    fig.tight_layout()
    plt.show()
else:
    print('NO CONDITIONAL!')
# %% Plot how the 32 different basis states (labeled by syndrome plus Z_L) map onto eachother from the 16 corrections

mappings = np.zeros((16, 32))
for basis_index in range(32):
    overlap = np.empty(16)
    for correction in range(16):
        if basis_mapping_table[correction, basis_index] == 0:
            overlap[correction] = 0
        elif basis_mapping_table[correction, basis_index] < 16:
            overlap[correction] = 0.5 + \
                (basis_mapping_table[correction, basis_index]-1)/30
        elif basis_mapping_table[correction, basis_index] == 16:
            overlap[correction] = 3
        else:
            overlap[correction] = 2.5 - \
                (basis_mapping_table[correction, basis_index]-17)/30
        if state_fidelity(state, logical[0]):
            overlap[correction] = 0
    mappings[:, basis_index] = overlap

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
HM = ax.imshow(mappings, aspect=1, interpolation='none', cmap='Spectral_r')
ax.set_xticks(np.arange(32))
ax.set_xticklabels([bin(key)[2:].zfill(4) for key in range(
    16)]+[bin(key)[2:].zfill(4) for key in range(16)])
ax.set_yticks(np.arange(16))
ax.set_yticklabels([bin(key)[2:].zfill(4) for key in range(16)])
plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
         rotation_mode="anchor")
# cbar0 = fig.colorbar(HM, ax=ax, orientation='horizontal',
#             fraction=.06, pad=0.25)
fig.suptitle("Which subspace each basis gets mapped to by each correction operation\n" +
             r"Purple: $|0\rangle_L$"+"\nBlue: weight-1"+"\nGreen: weight-2\n"+r"Yellow: $|1\rangle_L$")
ax.set_xlabel('Basis states, labeled by their eigenvalues to the stabilizers\n' +
              r"Left: distance $\leq$ 1. Right: distance $\geq$ 2")
ax.set_ylabel("Correction operation, ordered by syndrome")
fig.tight_layout()
plt.show()

# %%
