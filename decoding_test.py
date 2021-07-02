# %%
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
    'n_cycles': 4,
    'reset': True,
    'recovery': True,
    'flag': False,
    'encoding': False,
    'conditional': False,
    'include_barriers': True,
    'generator_snapshot': True,
    'idle_snapshots': 0,
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
n_shots = 1024*4
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


stab_datapoints = list(np.where(np.logical_or.reduce(np.array([(kwargs['idle_snapshots']+np.arange(label_counter.value)) % (
        kwargs['idle_snapshots']+5) == kwargs['idle_snapshots'] + i+1 for i in range(4)]), dtype=bool))[0])


colors = plt.get_cmap("tab10")
time = np.empty(label_counter.value)
P_0 = np.empty(label_counter.value)
P_1 = np.empty(label_counter.value)
P_w1 = np.empty(label_counter.value)
P_w2 = np.empty(label_counter.value)
P = np.empty(label_counter.value)

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


def get_cycle_index(cycle_num):
    """Returns index at the end of the cycle"""
    return np.where((kwargs['idle_snapshots']+np.arange(label_counter.value)) % (kwargs['idle_snapshots']+5) == kwargs['idle_snapshots']+4)[0][cycle_num]


for cycle_num in range(kwargs['n_cycles']):
    print(cycle_num)
    cycle_index = get_cycle_index(cycle_num)
    print((P_0[cycle_index+1]-P_0[cycle_index])/P_w1[cycle_index])
    print((P_1[cycle_index+1]-P_1[cycle_index])/P_w2[cycle_index])
    print('')


# %%
post_select.get_subsystem_counts_up_to_cycle(
    results.get_counts(), 1, hex_keys=True)
# %%
