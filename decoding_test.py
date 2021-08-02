# %%
from qiskit.quantum_info.states.measures import state_fidelity
from simulator_program.idle_noise import add_idle_noise_to_circuit
from simulator_program.custom_noise_models import standard_times_delay, standard_times

from simulator_program.stabilizers import get_full_stabilizer_circuit, get_weight_1_basis, get_weight_2_basis, label_counter, get_snapshot_label, logical_states
import numpy as np
from matplotlib import pyplot as plt
from simulator_program.data_analysis_tools import default_execute, overlap_with_subspace
from IPython.display import display
from simulator_program import post_process, post_select

# %% Set up circuit
kwargs = {
    'n_cycles': 1,
    'reset': True,
    'recovery': True,
    'encoding': False,
    'conditional': True,
    'include_barriers': True,
    'generator_snapshot': True,
    'idle_snapshots': 10,
    'final_measure': False,
    'idle_delay': 'before'}
# Define our registers (Maybe to be written as function?)


gate_times = standard_times_delay
# gate_times = standard_times
# Get the complete circuit
circ = get_full_stabilizer_circuit(**kwargs)

circ, times = add_idle_noise_to_circuit(circ, gate_times, return_time=True)
# display(circ.draw(output='mpl'))


# %%
# Run it
n_shots = 1024*8
results = default_execute(circ, n_shots, gate_times=gate_times)
# %%
logical = logical_states(None)
weight_1 = get_weight_1_basis()
weight_2 = get_weight_2_basis()

# %%


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

def get_stab_datapoints(where = True):
    snap_indices = np.arange(label_counter.value)
    snaps_per_cycle =  kwargs['idle_snapshots']+5
    stab_indices_in_cycle = [i+1 for i in range(4)] if kwargs['idle_delay']=='after' else [i+1+kwargs['idle_snapshots'] for i in range(4)]
    stab_cond = np.logical_or.reduce(np.array([(snap_indices) % (snaps_per_cycle) == i for i in stab_indices_in_cycle ]), dtype=bool)
    if where == True:
        return list(np.where(stab_cond)[0])
    else:
        return list(stab_cond)

stab_datapoints = get_stab_datapoints()

colors = plt.get_cmap("tab10")
time = np.empty(label_counter.value)
P_0 = np.empty(label_counter.value)
P_1 = np.empty(label_counter.value)
P_w1 = np.empty(label_counter.value)
P_w2 = np.empty(label_counter.value)
P = np.empty(label_counter.value)

# TODO: Make work with delay snapshots for key other than '0x0' 
# (can't accept measurements less often than once per snapshot)
key = '0x0'
for i in range(label_counter.value):
    label = get_snapshot_label('dm', kwargs['conditional'], i)
    if kwargs['conditional']:
        rho = results.data()[label][reduce_key(key, i)]
    else:
        rho = results.data()[label]
    time[i] = times[label]

    P_0[i] = overlap_with_subspace(rho, logical[0])
    P_1[i] = overlap_with_subspace(rho, logical[1])
    P_w1[i] = overlap_with_subspace(rho, weight_1)
    P_w2[i] = overlap_with_subspace(rho, weight_2)
    P[i] = overlap_with_subspace(rho, logical)


order = np.array([P_0, P_w1, P_w2, P_1])

if kwargs['conditional']:
    counts_at_snapshots = post_select.get_trivial_post_select_counts_V2(results.get_counts(),stab_datapoints,label_counter.value) 
else:
    counts_at_snapshots = np.ones(label_counter.value)

for i in range(4):
    lower = order[:i].sum(axis=0)
    plt.fill_between(time, (order[i]+lower)*counts_at_snapshots, lower*counts_at_snapshots,
                     color=np.array(colors(i))-np.array([0, 0, 0, 0.3]))
for i in range(4):
    lower = order[:i].sum(axis=0)
    # Picks out the data points belonging to only the generators
    plt.plot(time[stab_datapoints], ((order[i]+lower)*counts_at_snapshots)[stab_datapoints], '.', color=colors(i))

if kwargs['conditional']:
    plt.ylabel('counts')
else:
    plt.ylabel('Overlap')
plt.xlabel('Time [μs]')
plt.ylim((0, counts_at_snapshots[0]))
plt.xlim((0, time[-1]))
plt.legend(labels=[r'$|0\rangle_L$', 'Weight 1', 'Weight 2', r'$|1\rangle_L$'])
# %% Look at the proportion of states that get projected to |0>_L from w1 and to |1>_L from w2, are they the same?


def get_cycle_indices():
    """Returns index of the end of each cycle (final stabilizer measurement)."""
    stab_offset = 0 if kwargs['idle_delay']=='after' else kwargs['idle_snapshots'] 
    return np.where((np.arange(label_counter.value)) % (kwargs['idle_snapshots']+5) == stab_offset+4)[0]


for cycle_num in range(kwargs['n_cycles']):
    print(cycle_num)
    cycle_index = get_cycle_indices()[cycle_num]
    print((P_0[cycle_index+1]-P_0[cycle_index])/P_w1[cycle_index])
    print((P_1[cycle_index+1]-P_1[cycle_index])/P_w2[cycle_index])
    print('')


# %%
post_select.get_subsystem_counts_up_to_cycle(
    results.get_counts(), 1, hex_keys=True)
# %% Look at overlap with each state before first correction

if kwargs['conditional']:
    cycle_end = get_cycle_indices()[0]
    label = get_snapshot_label('dm', kwargs['conditional'], cycle_end)
    overlaps = np.zeros((16,32))
    for key in  results.data()[label]:
        print(bin(int(key,16))[2:].zfill(4))
        rho = results.data()[label][key]

        basis = [logical[0],*weight_1,logical[1],*weight_2]
        overlap = np.empty(32)
        for i in range(32):
            overlap[i] = state_fidelity(basis[i],rho)
        # plt.bar(range(32),overlap)
        # plt.show()
        overlaps[int(key,16),:] = overlap
    threshold = np.zeros(16)
    for key in range(16):
        threshold[key] = overlaps[key,key] < 0.5

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    HM =  ax.imshow(overlaps)

    c = 'red'
    for i in range(16):
        if threshold[i] == 0:
            continue
        else:
            ax.plot([i-0.5,i+0.5],[i - 0.5,i - 0.5],c)
            ax.plot([i-0.5,i+0.5],[i + 0.5,i + 0.5],c)
            ax.plot([i+0.5,i+0.5],[i - 0.5,i + 0.5],c)
            ax.plot([i-0.5,i-0.5],[i - 0.5,i + 0.5],c)

    ax.set_xticks(np.arange(32))
    ax.set_yticks(np.arange(16))
    ax.set_xticklabels([bin(key)[2:].zfill(4) for key in range(16)]+[bin(key)[2:].zfill(4) for key in range(16)])
    ax.set_yticklabels([bin(key)[2:].zfill(4) for key in range(16)])
    
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
         rotation_mode="anchor")
    cbar0 = fig.colorbar(HM, ax=ax, orientation='horizontal',
                     fraction=.06, pad=0.25)
    fig.suptitle("Fidelity to the 32 basis states conditioned on stabilizer measurements\n red squares show where fid<0.5 (is that meaningful?)")
    ax.set_xlabel('Basis states, labeled by their eigenvalues to the stabilizers\n'+r"Left: weight $\leq$ 1 from $|0\rangle$. Right: weight $\geq$ 2 from $|0\rangle$")
    ax.set_ylabel("Simualtion state\n conditioned on stabilizer measurements")
    fig.tight_layout()
    plt.show()
else:
    print('NO CONDITIONAL!')
# %%
