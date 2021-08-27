"""This script runs simulations with 'pausing' a circuit after each stabilizer cycle,
which allows for more advanced QEC schemes to be implemented as real-time error correction.
Currently it has a few methods implemented, centered around rerunning a set of stabilizers
if their outcome is uncertain. For more details on the topic, see decoding_errors.ipynb
"""

# Disable pylinter giving false positives of error.
# pylint: disable=E1101
#
#
# TODO:
#   - Pre-define all circuits (with idle noise & feedback)
#   - use .compose(circ) for add_start_to_circuit
# %% Import modules
from matplotlib.colors import to_rgb
from matplotlib import colors as clrs  # TODO: Fix
import pickle

import matplotlib.pyplot as plt
# Standard packages
import numpy as np
import scipy
# Qiskit
from qiskit import *
from qiskit import QuantumCircuit
from qiskit.providers.aer.noise import thermal_relaxation_error
from qiskit.quantum_info import state_fidelity
from qiskit.quantum_info.states.statevector import Statevector

from simulator_program.custom_noise_models import (GateTimes,
                                                   WACQT_target_times,
                                                   standard_times,
                                                   standard_times_delay,
                                                   thermal_relaxation_model_V2)
from simulator_program.data_analysis_tools import default_execute, overlap_with_subspace
from simulator_program.idle_noise import add_idle_noise_to_circuit
from simulator_program.splitting_circuits import (add_start_to_circuit,
                                                  add_stop_to_circuit,
                                                  split_circuit)
# Local files
from simulator_program.stabilizers import (_get_stabilizer_IXZZX,
                                           _get_stabilizer_XIXZZ,
                                           _get_stabilizer_XZZXI,
                                           _get_stabilizer_ZXIXZ,
                                           _get_stabilizer_ZZXIX,
                                           get_empty_stabilizer_circuit,
                                           get_full_stabilizer_circuit,
                                           get_registers,
                                           get_repeated_stabilization,
                                           get_distance_1_basis,
                                           get_distance_2_basis, label_counter,
                                           logical_states, syndrome_table)

# %% Correction strategies


def get_partial_stab_cycle(registers=None, stabilizers=[]):
    """Circuit for rerunning a set of stabilizers

    Args: 
        stabililzers (list[int]): List of integers representing the stabilizers 
                                  to rerun.
    Returns:
        circ: QuantumCircuit object for the extra stabilizer run.
    """
    if registers is None:
        registers = get_registers()
    circ = get_empty_stabilizer_circuit(registers)

    stabilizer_list = [_get_stabilizer_XZZXI,
                       _get_stabilizer_IXZZX,
                       _get_stabilizer_XIXZZ,
                       _get_stabilizer_ZXIXZ,
                       _get_stabilizer_ZZXIX]
    syn_bit_list = [registers.SyndromeRegister[n]
                    for n in range(len(stabilizers))]

    # Append all stabilizers to perform
    for i in range(len(stabilizers)):
        stab_index = stabilizers[i]
        circ.compose(stabilizer_list[stab_index](registers, syn_bit=syn_bit_list[i]),
                     inplace=True)
        circ.barrier()
    return circ


def get_reduced_recovery(registers, syndromes_to_remove=['0100', '1000', '1100']):
    """Return the normal recovery operation but where the special syndromes are ignored."""
    qbReg = registers.QubitRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)

    # Unpack registers
    if isinstance(clReg, list):
        syndrome_reg = clReg[0][0][0]
    else:
        syndrome_reg = clReg

    circ.x(qbReg[1]).c_if(syndrome_reg, 1)
    if not '0010' in syndromes_to_remove:
        circ.z(qbReg[4]).c_if(syndrome_reg, 2)
    circ.x(qbReg[2]).c_if(syndrome_reg, 3)
    if not '0100' in syndromes_to_remove:
        circ.z(qbReg[2]).c_if(syndrome_reg, 4)
    circ.z(qbReg[0]).c_if(syndrome_reg, 5)
    if not '0110' in syndromes_to_remove:
        circ.x(qbReg[3]).c_if(syndrome_reg, 6)
    circ.x(qbReg[2]).c_if(syndrome_reg, 7)
    circ.z(qbReg[2]).c_if(syndrome_reg, 7)
    if not '1000' in syndromes_to_remove:
        circ.x(qbReg[0]).c_if(syndrome_reg, 8)
    circ.z(qbReg[3]).c_if(syndrome_reg, 9)
    if not '1010' in syndromes_to_remove:
        circ.z(qbReg[1]).c_if(syndrome_reg, 10)
    circ.x(qbReg[1]).c_if(syndrome_reg, 11)
    circ.z(qbReg[1]).c_if(syndrome_reg, 11)
    if not '1100' in syndromes_to_remove:
        circ.x(qbReg[4]).c_if(syndrome_reg, 12)
    circ.x(qbReg[0]).c_if(syndrome_reg, 13)
    circ.z(qbReg[0]).c_if(syndrome_reg, 13)
    if not '1110' in syndromes_to_remove:
        circ.x(qbReg[4]).c_if(syndrome_reg, 14)
        circ.z(qbReg[4]).c_if(syndrome_reg, 14)
    circ.x(qbReg[3]).c_if(syndrome_reg, 15)
    circ.z(qbReg[3]).c_if(syndrome_reg, 15)

    return circ


def get_partial_recovery(registers, syndrome):
    """ Returns specialized recovery schemes for certain circuits.
    """
    qbReg = registers.QubitRegister
    clReg = registers.SyndromeRegister
    circ = get_empty_stabilizer_circuit(registers)

    # Unpack registers
    if isinstance(clReg, list):
        syndrome_reg = clReg[0][0][0]
    else:
        syndrome_reg = clReg

    # If the ancilla is reset to |0> between measurements
    if syndrome == '0x2':  # 0010
        circ.z(qbReg[4]).c_if(syndrome_reg, 2-2)
        circ.x(qbReg[2]).c_if(syndrome_reg, 3-2)
    elif syndrome == '0x4':  # 0100
        circ.z(qbReg[2]).c_if(syndrome_reg, 4-4)
        circ.z(qbReg[0]).c_if(syndrome_reg, 5-4)
        circ.x(qbReg[3]).c_if(syndrome_reg, 6-4)
        circ.x(qbReg[2]).c_if(syndrome_reg, 7-4)
        circ.z(qbReg[2]).c_if(syndrome_reg, 7-4)
    elif syndrome == '0x6':  # 0110
        circ.x(qbReg[3]).c_if(syndrome_reg, 6-6)
        circ.x(qbReg[2]).c_if(syndrome_reg, 7-6)
        circ.z(qbReg[2]).c_if(syndrome_reg, 7-6)
    elif syndrome == '0x8':  # 1000
        circ.x(qbReg[0]).c_if(syndrome_reg, 8-8)
        circ.z(qbReg[3]).c_if(syndrome_reg, 9-8)
        circ.z(qbReg[1]).c_if(syndrome_reg, 10-8)
        circ.x(qbReg[1]).c_if(syndrome_reg, 11-8)
        circ.z(qbReg[1]).c_if(syndrome_reg, 11-8)
        circ.x(qbReg[4]).c_if(syndrome_reg, 12-8)
        circ.x(qbReg[0]).c_if(syndrome_reg, 13-8)
        circ.z(qbReg[0]).c_if(syndrome_reg, 13-8)
        circ.x(qbReg[4]).c_if(syndrome_reg, 14-8)
        circ.z(qbReg[4]).c_if(syndrome_reg, 14-8)
        circ.x(qbReg[3]).c_if(syndrome_reg, 15-8)
        circ.z(qbReg[3]).c_if(syndrome_reg, 15-8)
    elif syndrome == '0xa':  # 1010
        circ.z(qbReg[1]).c_if(syndrome_reg, 10-10)
        circ.x(qbReg[1]).c_if(syndrome_reg, 11-10)
        circ.z(qbReg[1]).c_if(syndrome_reg, 11-10)
    elif syndrome == '0xc':  # 1100
        circ.x(qbReg[4]).c_if(syndrome_reg, 12-12)
        circ.x(qbReg[0]).c_if(syndrome_reg, 13-12)
        circ.z(qbReg[0]).c_if(syndrome_reg, 13-12)
        circ.x(qbReg[4]).c_if(syndrome_reg, 14-12)
        circ.z(qbReg[4]).c_if(syndrome_reg, 14-12)
        circ.x(qbReg[3]).c_if(syndrome_reg, 15-12)
        circ.z(qbReg[3]).c_if(syndrome_reg, 15-12)
    elif syndrome == '0xe':  # 1110
        circ.x(qbReg[4]).c_if(syndrome_reg, 14-14)
        circ.z(qbReg[4]).c_if(syndrome_reg, 14-14)
        circ.x(qbReg[3]).c_if(syndrome_reg, 15-14)
        circ.z(qbReg[3]).c_if(syndrome_reg, 15-14)
    return circ


# %% Iteration function
logical = logical_states(None)
weight_1 = get_distance_1_basis()
weight_2 = get_distance_2_basis()
basis = [Statevector(logical[0]), *weight_1,
         Statevector(logical[1]), *weight_2]


def branching_simulation(big_dict, rho, cycle, n_cycles, start_time, circ_std_cycle, special_recoveries):
    """Recursive function to iterate through and simulate a full syndrome tree.
    """

    # Simulate stabilizer cycle
    # TODO: add T1 T2 and gate_times
    circ = add_start_to_circuit(circ_std_cycle, rho)
    res = default_execute(
        circ, shots=big_dict['counts'], gate_times=gate_times, T1=T1, T2=T2)

    # Iterate over syndromes
    for syndrome in res.data()['end']:
        if syndrome in special_recoveries:
            sub_res = default_execute(add_start_to_circuit(special_recoveries[syndrome][0],
                                                           res.data()['end'][syndrome]),
                                      gate_times=gate_times, T1=T1, T2=T2,
                                      shots=res.data()['counts'][syndrome])
            end_time = start_time + \
                times_std_cycle['end'] + special_recoveries[syndrome][1]
            rho = sub_res.data()['end']

        else:
            rho = res.data()['end'][syndrome]
            end_time = start_time + times_std_cycle['end']

        # Store some data in (dict?)
        big_dict[syndrome] = {}  # Big dict energy
        big_dict[syndrome]['counts'] = int(res.data()['counts'][syndrome])
        big_dict[syndrome]['time'] = end_time
        big_dict[syndrome]['fid'] = np.array(
            [state_fidelity(basis_vec, rho) for basis_vec in basis])

        # Run next cycle
        if cycle+1 < n_cycles:
            branching_simulation(big_dict[syndrome], rho, cycle+1, n_cycles, end_time,
                                 circ_std_cycle, special_recoveries)
    return


# %% ============== Iterate through all branches===============================
kwargs = {
    'reset': True,
    'recovery': False,  # No recovery since we wanna split instead
    'flag': False,
    'encoding': False,
    'snapshot_type': 'density_matrix',
    'conditional': True,
    'idle_delay': 'before',
    'split_cycles': False,
    'final_measure': False
}
T1 = [40e3]*5+[100000000000, 100000000000]
T2 = [60e3]*5+[100000000000, 100000000000]
# T1 = 1000000000000000 #40e3
# T2 = 1000000000000000 #60e3
#gate_times = standard_times_delay
#gate_times = WACQT_target_times
gate_times = standard_times
#gate_times = GateTimes(0, 0, {'feedback': 0, 'delay': 3310})
registers = get_registers()

# Define the extra stabilizer circuits
circ_stab_1000 = get_partial_stab_cycle(registers, [0, 1, 2])
circ_stab_1000.compose(get_partial_recovery(registers, '0x8'), qubits=circ_stab_1000.qubits,
                       clbits=circ_stab_1000.clbits, inplace=True)
circ_stab_1000.save_density_matrix(qubits=registers.QubitRegister, label='end',
                                   conditional=False)
circ_stab_1000, times_stab_1000 = add_idle_noise_to_circuit(circ_stab_1000, gate_times=gate_times,
                                                            T1=T1, T2=T2, return_time=True)


# 0100
circ_stab_0100 = get_partial_stab_cycle(registers, [0, 1])
circ_stab_0100.compose(get_partial_recovery(registers, '0x4'), qubits=circ_stab_0100.qubits,
                       clbits=circ_stab_0100.clbits, inplace=True)
circ_stab_0100.save_density_matrix(qubits=registers.QubitRegister, label='end',
                                   conditional=False)
circ_stab_0100, times_stab_0100 = add_idle_noise_to_circuit(circ_stab_0100, gate_times=gate_times,
                                                            T1=T1, T2=T2, return_time=True)

# 1100
circ_stab_1100 = get_partial_stab_cycle(registers, [0, 1])
circ_stab_1100.compose(get_partial_recovery(registers, '0xc'), qubits=circ_stab_1100.qubits,
                       clbits=circ_stab_1100.clbits, inplace=True)
circ_stab_1100.save_density_matrix(qubits=registers.QubitRegister, label='end',
                                   conditional=False)
circ_stab_1100, times_stab_1100 = add_idle_noise_to_circuit(circ_stab_1100, gate_times=gate_times,
                                                            T1=T1, T2=T2, return_time=True)

# 0110
circ_stab_0110 = get_partial_stab_cycle(registers, [0])
circ_stab_0110.compose(get_partial_recovery(registers, '0x6'), qubits=circ_stab_0110.qubits,
                       clbits=circ_stab_0110.clbits, inplace=True)
circ_stab_0110.save_density_matrix(qubits=registers.QubitRegister, label='end',
                                   conditional=False)
circ_stab_0110, times_stab_0110 = add_idle_noise_to_circuit(circ_stab_0110, gate_times=gate_times,
                                                            T1=T1, T2=T2, return_time=True)

# 1110
circ_stab_1110 = get_partial_stab_cycle(registers, [0])
circ_stab_1110.compose(get_partial_recovery(registers, '0xe'), qubits=circ_stab_1110.qubits,
                       clbits=circ_stab_1110.clbits, inplace=True)
circ_stab_1110.save_density_matrix(qubits=registers.QubitRegister, label='end',
                                   conditional=False)
circ_stab_1110, times_stab_1110 = add_idle_noise_to_circuit(circ_stab_1110, gate_times=gate_times,
                                                            T1=T1, T2=T2, return_time=True)

# 0010
circ_stab_0010 = get_partial_stab_cycle(registers, [0])
circ_stab_0010.compose(get_partial_recovery(registers, '0x2'), qubits=circ_stab_0010.qubits,
                       clbits=circ_stab_0010.clbits, inplace=True)
circ_stab_0010.save_density_matrix(qubits=registers.QubitRegister, label='end',
                                   conditional=False)
circ_stab_0010, times_stab_0010 = add_idle_noise_to_circuit(circ_stab_0010, gate_times=gate_times,
                                                            T1=T1, T2=T2, return_time=True)
# 1010
circ_stab_1010 = get_partial_stab_cycle(registers, [0])
circ_stab_1010.compose(get_partial_recovery(registers, '0xa'), qubits=circ_stab_1010.qubits,
                       clbits=circ_stab_1010.clbits, inplace=True)
circ_stab_1010.save_density_matrix(qubits=registers.QubitRegister, label='end',
                                   conditional=False)
circ_stab_1010, times_stab_1010 = add_idle_noise_to_circuit(circ_stab_1010, gate_times=gate_times,
                                                            T1=T1, T2=T2, return_time=True)


# Different recovery schemes, named as follows:
# - full: Every non-trivial syndrome starting with a 0 reruns up until first 1
# - partial: Only those which are likely to map to weight 2 are rerun
# - subpar: Only those starting with 2+ zeroes are rerun (0100, 1000, 1100)
special_recoveries_full = {'0x2': (circ_stab_0010, times_stab_0010['end']),
                           '0x4': (circ_stab_0100, times_stab_0100['end']),
                           '0x6': (circ_stab_0110, times_stab_0110['end']),
                           '0x8': (circ_stab_1000, times_stab_1000['end']),
                           '0xa': (circ_stab_1010, times_stab_1010['end']),
                           '0xc': (circ_stab_1100, times_stab_1100['end']),
                           '0xe': (circ_stab_1110, times_stab_1110['end'])}
special_recoveries_partial = {'0x4': (circ_stab_0100, times_stab_0100['end']),
                              '0x6': (circ_stab_0110, times_stab_0110['end']),
                              '0x8': (circ_stab_1000, times_stab_1000['end']),
                              '0xc': (circ_stab_1100, times_stab_1100['end']),
                              '0xe': (circ_stab_1110, times_stab_1110['end'])}
special_recoveries_subpar = {'0x4': (circ_stab_0100, times_stab_0100['end']),
                             '0x8': (circ_stab_1000, times_stab_1000['end']),
                             '0xc': (circ_stab_1100, times_stab_1100['end'])}
standard_recoveries = {}

# Circuits
label_counter.value = 0

circ_std_cycle = get_repeated_stabilization(
    registers, 1, **kwargs, generator_snapshot=False, idle_snapshots=False)

circ_std_cycle_full = circ_std_cycle.compose(get_reduced_recovery(registers, [bin(int(key, 16))[2:].zfill(4) for key in special_recoveries_full]),
                                             qubits=circ_std_cycle.qubits, clbits=circ_std_cycle.clbits)
circ_std_cycle_partial = circ_std_cycle.compose(get_reduced_recovery(registers, [bin(int(key, 16))[2:].zfill(4) for key in special_recoveries_partial]),
                                                qubits=circ_std_cycle.qubits, clbits=circ_std_cycle.clbits)
circ_std_cycle_subpar = circ_std_cycle.compose(get_reduced_recovery(registers, [bin(int(key, 16))[2:].zfill(4) for key in special_recoveries_subpar]),
                                               qubits=circ_std_cycle.qubits, clbits=circ_std_cycle.clbits)
circ_std_cycle_standard = circ_std_cycle.compose(get_reduced_recovery(registers, [bin(int(key, 16))[2:].zfill(4) for key in standard_recoveries]),
                                                 qubits=circ_std_cycle.qubits, clbits=circ_std_cycle.clbits)
circ_std_cycle_perf = circ_std_cycle.compose(get_reduced_recovery(registers, [bin(int(key, 16))[2:].zfill(4) for key in standard_recoveries]),
                                             qubits=circ_std_cycle.qubits, clbits=circ_std_cycle.clbits)

circ_std_cycle_full.save_density_matrix(qubits=registers.QubitRegister, label='end',
                                        conditional=kwargs['conditional'])
circ_std_cycle_partial.save_density_matrix(qubits=registers.QubitRegister, label='end',
                                           conditional=kwargs['conditional'])
circ_std_cycle_subpar.save_density_matrix(qubits=registers.QubitRegister, label='end',
                                          conditional=kwargs['conditional'])
circ_std_cycle_standard.save_density_matrix(qubits=registers.QubitRegister, label='end',
                                            conditional=kwargs['conditional'])
circ_std_cycle_perf.save_density_matrix(qubits=registers.QubitRegister, label='end',
                                        conditional=kwargs['conditional'])

circ_std_cycle_full, times_std_cycle = add_idle_noise_to_circuit(circ_std_cycle_full, gate_times,
                                                                 T1, T2, return_time=True)
circ_std_cycle_partial, times_std_cycle = add_idle_noise_to_circuit(circ_std_cycle_partial, gate_times,
                                                                    T1, T2, return_time=True)
circ_std_cycle_subpar, times_std_cycle = add_idle_noise_to_circuit(circ_std_cycle_subpar, gate_times,
                                                                   T1, T2, return_time=True)
circ_std_cycle_standard, times_std_cycle = add_idle_noise_to_circuit(circ_std_cycle_standard, gate_times,
                                                                     T1, T2, return_time=True)
circ_std_cycle_perf, times_std_cycle = add_idle_noise_to_circuit(circ_std_cycle_perf, GateTimes(0, 0, {'feedback': 0, 'delay': 3310}),
                                                                 T1, T2, return_time=True)

# Group the circuit and strategy
strategy_full_special = (circ_std_cycle_full, special_recoveries_full)
strategy_partial_special = (circ_std_cycle_partial, special_recoveries_partial)
strategy_subpar_special = (circ_std_cycle_subpar, special_recoveries_subpar)
strategy_standard = (circ_std_cycle_standard, standard_recoveries)
strategy_perf = (circ_std_cycle_perf, standard_recoveries)


# %% Running simulations

initial_fid = np.zeros(32)
initial_fid[0] = 1
initial_state = logical_states(include_ancillas=None)[0]
# %% Running simulations
n_cycles = 10
n_shots = int(1024/4)
names = ['Standard', "Remeasure syndromes '0100', '1000' and '1100'", 'Repeat syndromes 0100 0110 1000 1100 1110',
         "Remeasure syndromes '0010', '0100', '0110',\n '1000', '1010', '1100' and '1110'", 'Upper limit']
# %%
standard_res_dict = {'counts': n_shots, 'time': 0, 'fid': initial_fid}
branching_simulation(standard_res_dict, initial_state,
                     0, n_cycles, 0, *strategy_standard)
print('1')
subpar_res_dict = {'counts': n_shots, 'time': 0, 'fid': initial_fid}
branching_simulation(subpar_res_dict, initial_state,
                     0, n_cycles, 0, *strategy_subpar_special)
print('2')
partial_res_dict = {'counts': n_shots, 'time': 0, 'fid': initial_fid}
branching_simulation(partial_res_dict, initial_state,
                     0, n_cycles, 0, *strategy_subpar_special)
print('3')
full_res_dict = {'counts': n_shots, 'time': 0, 'fid': initial_fid}
branching_simulation(full_res_dict, initial_state,
                     0, n_cycles, 0, *strategy_subpar_special)
print('4')
perf_res_dict = {'counts': n_shots, 'time': 0, 'fid': initial_fid}
branching_simulation(perf_res_dict, initial_state,
                     0, n_cycles, 0, *strategy_perf)
print('5')
runs_to_print_together = [standard_res_dict,
                          subpar_res_dict, partial_res_dict, full_res_dict, perf_res_dict]

# %% Save
pickle.dump(runs_to_print_together, open(
    'split_data/runs_to_print_together.dat', "wb"))
# %% load
runs_to_print_together = pickle.load(
    open('split_data/runs_to_print_together.dat', "rb"))

runs_to_print_together = np.array(runs_to_print_together)[(0, 1,3, 4), ]
names = np.array(names)[(0, 1,3, 4), ]
# %% Append every shot into array


def flatten_data(big_dict):
    # TODO: fix this to not use nonlocal variables?
    n_points = (n_cycles+1)*n_shots
    times_full = np.zeros(int(n_points), dtype=int)
    fids_full = np.zeros((int(n_points), 32), dtype=float)
    cycles_full = np.zeros(int(n_points), dtype=int)
    index = 0

    def append_shots(big_dict, current_cycle):
        nonlocal index
        times_full[index:index+big_dict['counts']] = big_dict['time']
        fids_full[index:index+big_dict['counts'], :] = big_dict['fid']
        cycles_full[index:index+big_dict['counts']] = current_cycle
        index += big_dict['counts']
        for key in big_dict:
            if key == 'counts' or key == 'time' or key == 'fid':
                continue
            append_shots(big_dict[key], current_cycle+1)
    append_shots(big_dict, 0)
    return times_full, fids_full, cycles_full


flattened_data = [flatten_data(res_dict)
                  for res_dict in runs_to_print_together]

# %% Plotting functions


def plot_by_bins(ax, bins, times_full, fids_full, cycles_full, label='grouped by bins', color='b'):
    time_bins = np.linspace(0, max(times_full), bins+1)
    for i in range(bins):
        ax.plot((time_bins[i]+time_bins[i+1])/(2*1000),
                np.mean(fids_full[np.logical_and(
                    time_bins[i] < times_full, times_full < time_bins[i+1]), 0]), 'o',
                color=color, label=label)


def plot_curvefit(ax, times_full, fids_full, cycles_full, color='C1'):
    def idealExp(t, T):
        return 0.5 * np.exp(-t/T) + 0.5
    p0 = 40e3  # start with values near those we expect

    # Set up variabled defined outiside t = 0
    T = times_full[cycles_full != 0]
    F = fids_full[cycles_full != 0, 0]
    P_L = (fids_full[cycles_full != 0, 0]+fids_full[cycles_full != 0, 16])
    F_L = F/P_L

    time_after_first_cycle = np.mean(
        times_full[cycles_full == 1])  # TODO: better solution
    pars_full, cov_full = scipy.optimize.curve_fit(
        idealExp, T-time_after_first_cycle, F_L, p0)
    print('P_L =', np.mean(P_L), '+-', np.std(P_L))
    x = np.linspace(time_after_first_cycle, max(times_full), 200)
    ax.plot(x/1000, idealExp(x-time_after_first_cycle, *pars_full)
            * np.mean(P_L), ':', color=color, zorder=15)
    # label=rf'Curve fit, $T_L ={pars_full[0]/1000:.1f}$ μs')
    return pars_full, cov_full


def plot_curvefit_V2(ax, time, fid, color='C2'):
    def PLExp(t, T, P_L):
        return (0.5 * np.exp(-t/T) + 0.5)*P_L
    p0 = (40e3,0.8)  # start with values near those we expect

    # Set up variabled defined outiside t = 0

    time_after_first_cycle = time[1]
    pars_full, cov_full = scipy.optimize.curve_fit(
        PLExp, time[1:]-time_after_first_cycle, fid[1:], p0)

    x = np.linspace(time_after_first_cycle, max(time), 200)
    ax.plot(x/1000, PLExp(x-time_after_first_cycle,
                          *pars_full), ':', color=color, zorder=15)
    # label=rf'Curve fit, $T_L ={pars_full[0]/1000:.1f}$ μs')
    return pars_full, cov_full


def plot_by_cycle_errorbar(ax, times_full, fids_full, cycles_full, label='Grouped by cycle', color='C0'):
    cycles = int(max(cycles_full)+1)
    median_fid = np.zeros(cycles)
    yerr = np.zeros((2, cycles))
    xerr = np.zeros((2, cycles))
    times_cycle = np.zeros(cycles)
    for i in range(cycles):
        median_fid[i] = np.quantile(fids_full[cycles_full == i, 0], 0.5)
        yerr[:, i] = np.abs(np.quantile(
            fids_full[cycles_full == i, 0], [0.25, 0.75])-median_fid[i])
        times_cycle[i] = np.mean(times_full[cycles_full == i])/1000
        xerr[:, i] = np.abs(np.quantile(times_full[cycles_full == i], [
                            0.25, 0.75])/1000-times_cycle[i])

    ax.errorbar(times_cycle, median_fid, yerr, xerr, '.',
                color=color, label=label)


def plot_by_cycle_mean(ax, times_full, fids_full, cycles_full, label='Grouped by cycle', color='C0', marker = 'o'):
    cycles = int(max(cycles_full)+1)
    fid_cycle = np.zeros(cycles)
    times_cycle = np.zeros(cycles)
    for i in range(cycles):
        # # F_L
        # fid_cycle[i] += np.mean(fids_full[cycles_full == i,0]/(fids_full[cycles_full == i,0]+fids_full[cycles_full == i,16]))
        # F
        fid_cycle[i] += np.mean(fids_full[cycles_full == i, 0])
        times_cycle[i] += np.mean(times_full[cycles_full == i])

    ax.plot(times_cycle/1000, fid_cycle, marker,
            color=color, label=label)


# %% Load data from the script part of decoding_errors.py
with open('data/decoding_strategies.npy', 'rb') as f:
    time_decoding = np.load(f)
    total_fid_lookup_list = np.load(f)
    total_fid_lookup_last_cycle_list = np.load(f)
    total_fid_best_single_qb_list = np.load(f)
    total_fid_best_unitary_list = np.load(f)
# %% Testing plots

markers=['o', 'p', 's', '>', '<','D']

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
bins = n_cycles

for i, run in enumerate(flattened_data):
    # plot_by_bins(ax, bins, *run, label=names[i], color='C'+str(i))
    pars, cov = plot_curvefit(
        ax, *run, color='C'+str(i))  # names[i]+rf', $T_L ={pars[0]/1000:.1f}$'
    # plot_by_cycle_errorbar(ax, *run, label=None, color='C'+str(i))
    plot_by_cycle_mean(
        ax, *run, label=names[i]+rf', $T_L ={pars[0]/1000:.1f}$', color='C'+str(i), marker = markers[i])

# # Old no-splitting results


def monoExp(t, T, c, A):
    return (A-c) * np.exp(-t/T) + c

i += 1
# t = np.linspace(0, max([np.max(run[0]) for run in flattened_data]),100)
pars, cov = plot_curvefit_V2(ax, time_decoding, total_fid_best_single_qb_list, color='C'+str(i))
ax.plot(time_decoding/1000, total_fid_best_single_qb_list, markers[i], color='C'+str(i),
        label=rf'Best possible correction, $T_L ={pars[0]/1000:.1f}$ μs')
# ax.plot(time_decoding/1000, total_fid_lookup_list, 's', color='C6',
#         label=rf'standard v2, $T_L ={0/1000:.1f}$ μs')
i+=1
ax.plot(time_decoding/1000, total_fid_lookup_last_cycle_list, markers[i], color='C'+str(i),
        label=rf'Correct using only last syndrome')
# ax.plot(time_decoding/1000, total_fid_best_unitary_list, 's', color='C8',
#         label=rf'Best possible unitary, $T_L ={0/1000:.1f}$ μs')
# ax.plot(t_hex/1000, fid_hex, 's', color='C5',
#         label=rf'Hexagonal, $T_L ={pars_hex[0]/1000:.1f}$ μs')
# ax.plot(t_hex_d/1000, fid_hex_d, 'D', color='C6',
#         label=rf'Hexagonal, 5 μs delay, $T_L ={pars_hex_d[0]/1000:.1f}$ μs')
# ax.plot(t/1000, monoExp(t, *pars_hex), ':', color='C5') # This one has P_L included in pars
# ax.plot(t/1000, monoExp(t, *pars_hex_d)*np.mean(P_L_hex_d[1:]), ':', color='C6',zorder=-5) # This one doesnt


# Plot settings
ax.legend()
ax.set_ylim([0.0, 1.05])
ax.set_xlim([0.0, 36])
ax.set_xlabel('Time [μs]')
ax.set_ylabel(r'Fidelity $F$')
plt.show()
fig.savefig('repeat.png', transparent=True)
# %% Plot scatter
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.set_ylim([0.0, 1.05])
ax.set_xlim([0.0, max([np.max(run[0]) for run in flattened_data])/1000])
ax.set_xlabel('Time [μs]')
ax.set_ylabel(r'Fidelity $F$')


def reform_dict(big_dict):
    # TODO: fix this to not use nonlocal variables?
    times = []
    cycles = []
    fids = []
    counts = []

    def append_shots(big_dict, current_cycle):
        times.append(big_dict['time'])
        fids.append(big_dict['fid'])
        counts.append(big_dict['counts'])
        cycles.append(current_cycle)
        for key in big_dict:
            if key == 'counts' or key == 'time' or key == 'fid':
                continue
            append_shots(big_dict[key], current_cycle+1)
    append_shots(big_dict, 0)
    return times, fids, cycles, counts


def scatter_plot(ax, times, fids, cycles, counts, label, c='b', marker='o', alpha=0.5):
    r, g, b = to_rgb(c)
    c = [(r, g, b, alpha) for alpha in np.sqrt(np.array(counts)/n_shots)]
    ax.scatter(np.array(times)/1000, np.array(fids)
               [:, 0], label=label, s=counts, c=c, marker=marker)
    #    [:, 0], label=label, s=np.sqrt(counts)*10, c=c, marker=marker,alpha=alpha)


for i in range(len(runs_to_print_together)):
    pars, cov = plot_curvefit(ax, *flattened_data[i], color='C'+str(i))
    scatter_plot(ax, *reform_dict(runs_to_print_together[i]), label=names[i]+rf' $T_L ={pars[0]/1000:.1f}$', c='C'+str(
        i), marker=['o', 'p', 's', '>', '<'][i])
    # plot_by_cycle_errorbar(ax, *flattened_data[i], label=names[i]+rf' $T_L ={pars[0]/1000:.1f}$', color='C'+str(i))

ax.legend()
plt.show()
# %% Plot histograms
dataset = 0
plt.hist(flattened_data[dataset][1]
         [flattened_data[dataset][2] == 10, 0], bins=10)
plt.show()
# %% Old code

# Plot Many lines


def lines_plot(ax, big_dict, times, fids, current_cycle):
    times_new_branch = times
    times_new_branch[current_cycle] = big_dict['time']
    fids_new_branch = fids
    fids_new_branch[current_cycle] = np.argmax(big_dict['fid'])
    # fids_new_branch[current_cycle] = big_dict['fid'][0]
    # counts_new_branch = counts
    # counts_new_branch[current_cycle] = big_dict['counts']
    if current_cycle == n_cycles-1:
        ax.plot(times, fids, color='b', alpha=np.sqrt(big_dict['counts']/n_shots),
                linewidth=np.sqrt(big_dict['counts']))
    else:
        for key in big_dict:
            if not(key == 'counts' or key == 'time' or key == 'fid'):
                lines_plot(ax, big_dict[key], times_new_branch,
                           fids_new_branch, current_cycle + 1)


times = [0]*n_cycles
fids = [0]*n_cycles
# counts = [0]*n_cycles

dataset = 0
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
lines_plot(ax, runs_to_print_together[dataset], times, fids, 0)
plt.show()


def scatter_plot_v2(ax, big_dict, c='b', marker='o'):
    ax.scatter(big_dict['time']/1000, big_dict['fid'][0],
               s=big_dict['counts'], c=c, marker=marker, alpha=0.7)
    for key in big_dict:
        if not(key == 'counts' or key == 'time' or key == 'fid'):
            scatter_plot(ax, big_dict[key], c)
        # ax.scatter(big_dict[key]['time'],big_dict[key]['fid'], s=big_dict[key]['counts'], c='b', marker='o')


def standard_QEC(rho, syndrome, T1=40e3, T2=60e3, feedback_time=350):
    """Evolves a 7-qb density matrix in a way which corresponds to standard QEC,
    using a lookup-table and a feedback time to process results."""

    thrm_relax = thermal_relaxation_error(
        T1, T2, feedback_time).to_instruction()

    # Define circuit and idle for feedback time
    circ_correct = QuantumCircuit(7)
    circ_correct.set_density_matrix(rho)
    for qubit in circ_correct.qubits:
        circ_correct.append(thrm_relax, [qubit])

    # Apply correction
    for correction in syndrome_table[int(syndrome[:3], 16)]:
        circ_correct.append(correction[0](), [correction[1]])
    circ_correct.save_density_matrix(circ_correct.qubits, label='tmp')

    # Run the error correction
    # Note: "Noise free" since the noise is interpreted to be part of feedback time
    results_tmp = execute(circ_correct, Aer.get_backend('qasm_simulator'),
                          shots=1).result()
    return results_tmp.data()['tmp']


# %%
with open('data/QEC_hexagonal_standard_times.npy', 'rb') as f:
    _ = np.load(f)
    _ = np.load(f)
    fid_L_hex = np.load(f)
    P_L_hex = np.load(f)
    t_hex = np.load(f)
    pars_hex = np.load(f)
    cov_hex = np.load(f)

with open('data/QEC_hexagonal_4mus_delay.npy', 'rb') as f:
    _ = np.load(f)
    _ = np.load(f)
    fid_L_hex_d = np.load(f)
    P_L_hex_d = np.load(f)
    t_hex_d = np.load(f)
    pars_hex_d = np.load(f)
    cov_hex_d = np.load(f)

fid_hex = np.ones(fid_L_hex.shape)
fid_hex_d = np.ones(fid_L_hex_d.shape)
fid_hex[1:] = fid_L_hex[1:]*np.mean(P_L_hex[1:])
fid_hex_d[1:] = fid_L_hex_d[1:]*np.mean(P_L_hex_d[1:])

# %%
dataset = 3
previuous_keys = []  # Post select
state_dict = runs_to_print_together[dataset]['0x0']['0x2']
backtrack_measurement = 0

overlaps = np.zeros((16, 33))+np.infty

# resulting (average) fidelity if applying the 'normal' lookup table definition
total_fid_lookup = 0
# resulting (average) fidelity if applying the best single qubit correction
total_fid_best_single_qb = 0

for key in state_dict:
    if not(key == 'counts' or key == 'time' or key == 'fid'):
        # Int version of the part of the key corresponding to the overlap_cycle
        key_int = int(key, 16)
        # Int version of the full key up to the current cycle
        # TODO: Should be unecceray to cut of end?
        key_up_to_cycle = int(key, 16)

        overlap = np.zeros(33)
        overlap[:32] = state_dict[key]['fid']
        overlap[32] = state_dict[key]['counts']
        overlaps[key_int, :] = overlap

        fid_lookup = overlap[key_int]
        fid_best_single_qb = np.max(overlap[:32])

        total_fid_lookup += fid_lookup*state_dict[key]['counts']
        total_fid_best_single_qb += fid_best_single_qb * \
            state_dict[key]['counts']

        # Print the difference between the assigned correction and the theoretical max
        # if fid_lookup<fid_best_single_qb:
        #     print(bin(key_int)[2:].zfill(num_stab_gens),':', fid_lookup,' -> ',fid_best_single_qb,' -> ',fid_best_arbitrary_gate)

total_counts = np.sum(overlaps[:, 32][overlaps[:, 32] != np.infty])
overlaps[:, 32] /= total_counts
total_fid_lookup /= total_counts
total_fid_best_single_qb /= total_counts

# TODO: these don't seem to give quite the right result when conditioning
print('Fidelity if using standard lookup table', total_fid_lookup)
print('Fidelity if using optimal single qubit correction',
      total_fid_best_single_qb)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
HM = ax.imshow(overlaps, interpolation='none')
norm = clrs.Normalize(0, 1)
HM.set_norm(norm)
ax.set_xticks(np.arange(33))
ax.set_xticklabels([bin(key)[2:].zfill(4) for key in range(
    16)]+[bin(key)[2:].zfill(4) for key in range(16)]+['occup.'])
y_ticks = np.arange(0, 16)
ax.set_yticks(y_ticks)
ax.set_yticklabels([bin(key)[2:].zfill(4) for key in y_ticks])

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
