# Disable pylinter giving false positives of error.
# pylint: disable=E1101
#
#
# TODO:
#   - Pre-define all circuits (with idle noise & feedback)
#   - use .compose(circ) for add_start_to_circuit
# %% Import modules
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
from simulator_program.data_analysis_tools import default_execute
from simulator_program.idle_noise import add_idle_noise_to_circuit
from simulator_program.splitting_circuits import (add_start_to_circuit,
                                                  add_stop_to_circuit,
                                                  split_circuit)
# Local files
from simulator_program.stabilizers import (_unflagged_stabilizer_IXZZX,
                                           _unflagged_stabilizer_XIXZZ,
                                           _unflagged_stabilizer_XZZXI,
                                           _unflagged_stabilizer_ZXIXZ,
                                           _unflagged_stabilizer_ZZXIX,
                                           get_empty_stabilizer_circuit,
                                           get_full_stabilizer_circuit,
                                           get_registers,
                                           get_repeated_stabilization,
                                           get_weight_1_basis,
                                           get_weight_2_basis, label_counter,
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
        registers = get_registers(conditional, final_measure)

    # Define registers
    #qbReg = QuantumRegister(5, 'code_qubit')
    #anReg = AncillaRegister(2, 'ancilla_qubit')
    # crReg = ClassicalRegister(len(stabilizers), 'syndrome_bit')  # The typical register
    #registers = StabilizerRegisters(qbReg, anReg, crReg, None)
    circ = get_empty_stabilizer_circuit(registers)

    stabilizer_list = [_unflagged_stabilizer_XZZXI,
                       _unflagged_stabilizer_IXZZX,
                       _unflagged_stabilizer_XIXZZ,
                       _unflagged_stabilizer_ZXIXZ,
                       _unflagged_stabilizer_ZZXIX]
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
weight_1 = get_weight_1_basis()
weight_2 = get_weight_2_basis()
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
#gate_times = GateTimes(0, 0, {'feedback': 0, 'delay': 2000})


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

circ_std_cycle_full.save_density_matrix(qubits=registers.QubitRegister, label='end',
                                        conditional=kwargs['conditional'])
circ_std_cycle_partial.save_density_matrix(qubits=registers.QubitRegister, label='end',
                                           conditional=kwargs['conditional'])
circ_std_cycle_subpar.save_density_matrix(qubits=registers.QubitRegister, label='end',
                                          conditional=kwargs['conditional'])
circ_std_cycle_standard.save_density_matrix(qubits=registers.QubitRegister, label='end',
                                            conditional=kwargs['conditional'])

circ_std_cycle_full, times_std_cycle = add_idle_noise_to_circuit(circ_std_cycle_full, gate_times,
                                                                 T1, T2, return_time=True)
circ_std_cycle_partial, times_std_cycle = add_idle_noise_to_circuit(circ_std_cycle_partial, gate_times,
                                                                    T1, T2, return_time=True)
circ_std_cycle_subpar, times_std_cycle = add_idle_noise_to_circuit(circ_std_cycle_subpar, gate_times,
                                                                   T1, T2, return_time=True)
circ_std_cycle_standard, times_std_cycle = add_idle_noise_to_circuit(circ_std_cycle_standard, gate_times,
                                                                     T1, T2, return_time=True)

# Group the circuit and strategy
strategy_full_special = (circ_std_cycle_full, special_recoveries_full)
strategy_partial_special = (circ_std_cycle_partial, special_recoveries_partial)
strategy_subpar_special = (circ_std_cycle_subpar, special_recoveries_subpar)
strategy_standard = (circ_std_cycle_standard, standard_recoveries)


# %% Running simulations

initial_fid = np.zeros(32)
initial_fid[0] = 1
# %% Running simulations
n_cycles = 5
n_shots = int(1024/8)
initial_state = logical_states(include_ancillas=None)[0]
standard_res_dict = {'counts': n_shots, 'time': 0, 'fid': initial_fid}
branching_simulation(standard_res_dict, initial_state,
                     0, n_cycles, 0, *strategy_standard)
print('Halfway there')
special_res_dict = {'counts': n_shots, 'time': 0, 'fid': initial_fid}
branching_simulation(special_res_dict, initial_state,
                     0, n_cycles, 0, *strategy_subpar_special)
print('(Living on a prayer)')

runs_to_print_together = [standard_res_dict, special_res_dict]


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


def plot_by_bins(ax, bins, times_full, fids_full, cycles_full, c='b'):
    time_bins = np.linspace(0, max(times_full), bins+1)
    for i in range(bins):
        ax.scatter((time_bins[i]+time_bins[i+1])/2,
                   np.mean(fids_full[np.logical_and(
                       time_bins[i] < times_full, times_full < time_bins[i+1]), 0]),
                   c=c, marker='o')


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
    ax.plot(x, idealExp(x-time_after_first_cycle, *pars_full)*np.mean(P_L), '-', color=color, zorder=15,
            label=rf'Curve fit, $T_L ={pars_full[0]/1000:.1f}$ μs')
    return pars_full, cov_full


def plot_by_cycle(ax, times_full, fids_full, cycles_full, color='C0'):
    cycles = int(max(cycles_full)+1)
    fid_cycle = np.zeros(cycles)
    times_cycle = np.zeros(cycles)
    for i in range(cycles):
        # # F_L
        # fid_cycle[i] += np.mean(fids_full[cycles_full == i,0]/(fids_full[cycles_full == i,0]+fids_full[cycles_full == i,16]))
        # F
        fid_cycle[i] += np.mean(fids_full[cycles_full == i, 0])
        times_cycle[i] += np.mean(times_full[cycles_full == i])
    ax.plot(times_cycle, fid_cycle, '-o',
            color=color, label='Grouped by cycle')


# %% Testing plots


fig, ax = plt.subplots(1, 1, figsize=(7, 5))
bins = n_cycles

for i, run in enumerate(flattened_data):
    # plot_by_bins(ax, bins, *run, c='C'+str(i))
    pars_standard, cov_standard = plot_curvefit(
        ax, *run, color='C1')
    plot_by_cycle(ax, *run, color='C'+str(i))

# Old no-splitting results
# t = np.linspace(0, max(times_full_special),100)
# ax.plot(t_hex[:n_data], fid_hex, 's', color='C5',
#         label=rf'Hexagonal, $T_L ={pars_hex[0]/1000:.1f}$ μs')
# ax.plot(t_hex_d[:9], fid_hex_d[:9], 'D', color='C6',
#         label=rf'Hexagonal, 5 μs delay, $T_L ={pars_hex_d[0]/1000:.1f}$ μs')
# ax.plot(t, monoExp(t, *pars_hex), ':', color='C5') # This one has P_L included in pars
# ax.plot(t, monoExp(t, *pars_hex_d)*np.mean(P_L_hex_d[1:]), ':', color='C6',zorder=-5) # This one doesnt

# Plot settings
ax.legend()
ax.set_ylim([0.0, 1.0])
ax.set_xlim([0.0, max([np.max(run[0]) for run in flattened_data])])
plt.show()
# %% Plot Many lines


def lines_plot(ax, big_dict, times, fids, current_cycle):
    times_new_branch = times
    times_new_branch[current_cycle] = big_dict['time']
    fids_new_branch = fids
    fids_new_branch[current_cycle] = big_dict['fid']
    # counts_new_branch = counts
    # counts_new_branch[current_cycle] = big_dict['counts']
    if current_cycle == n_cycles-1:
        ax.plot(times, fids, color='b', alpha=np.sqrt(
            big_dict['counts'])/50, linewidth=np.sqrt(big_dict['counts']))
    else:
        for key in big_dict:
            if not(key == 'counts' or key == 'time' or key == 'fid'):
                lines_plot(ax, big_dict[key], times_new_branch,
                           fids_new_branch, current_cycle + 1)


times = [0]*n_cycles
fids = [0]*n_cycles
# counts = [0]*n_cycles

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
lines_plot(ax, standard_res_dict, times, fids, 0)


# %% Save
pickle.dump(standard_res_dict, open('split_data/standard_res_dict.dat', "wb"))
pickle.dump(special_res_dict, open('split_data/special_res_dict.dat', "wb"))
# %% load
standard_res_dict = pickle.load(open('split_data/standard_res_dict.dat', "rb"))
special_res_dict = pickle.load(open('split_data/special_res_dict.dat', "rb"))

# %% Plot scatter
fig, ax = plt.subplots(1, 1, figsize=(7, 5))


def scatter_plot(ax, big_dict, c='b'):
    ax.scatter(big_dict['time'], big_dict['fid'],
               s=big_dict['counts'], c=c, marker='o')

    for key in big_dict:
        if key == 'counts' or key == 'time' or key == 'fid':
            continue
        scatter_plot(ax, big_dict[key])
        # ax.scatter(big_dict[key]['time'],big_dict[key]['fid'], s=big_dict[key]['counts'], c='b', marker='o')


scatter_plot(ax, standard_res_dict)
scatter_plot(ax, special_res_dict, c='r')
ax.set_ylim([0.0, 1.0])
ax.set_xlim([0.0, n_cycles*6000.0])
plt.show()
# %% Old code
# Reform dict into arrays of each branch
times = []
cycles = []
fids = []
counts = []


def reform_dict(big_dict, current_cycle):
    times.append(big_dict['time'])
    fids.append(big_dict['fid'])
    counts.append(big_dict['counts'])
    cycles.append(current_cycle)
    for key in big_dict:
        if key == 'counts' or key == 'time' or key == 'fid':
            continue
        reform_dict(big_dict[key], current_cycle+1)


reform_dict(test_dict, 0)

times = np.array(times)
fids = np.array(fids)
counts = np.array(counts, dtype=float)
cycles = np.array(cycles)


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

# %% Run some "old" simulations


# QEC, no delay, hexagonal layout
fid_L_hex, P_L_hex, time_hex = fidelity_from_scratch(n_cycles, n_shots,
                                                     gate_times=standard_times, reset=True, data_process_type='recovery',
                                                     idle_noise=True, snapshot_type='dm', encoding=False, theta=0, phi=0,
                                                     transpile=False, project=True, generator_snapshot=False)

# QEC, with delay, hexagonal layout
fid_L_hex_d, P_L_hex_d, time_hex_d = fidelity_from_scratch(n_cycles, n_shots,
                                                           gate_times={'delay': 4000}, reset=True, data_process_type='recovery',
                                                           idle_noise=True, snapshot_type='dm', encoding=False, theta=0, phi=0,
                                                           transpile=False, project=True, generator_snapshot=False)

# %%
with open('data/QEC_hexagonal_standard_times.npy', 'rb') as f:
    n_cycles = np.load(f)
    n_shots = np.load(f)
    fid_L_hex = np.load(f)
    P_L_hex = np.load(f)
    t_hex = np.load(f)
    pars_hex = np.load(f)
    cov_hex = np.load(f)

with open('data/QEC_hexagonal_4mus_delay.npy', 'rb') as f:
    n_cycles = np.load(f)
    n_shots = np.load(f)
    fid_L_hex_d = np.load(f)
    P_L_hex_d = np.load(f)
    t_hex_d = np.load(f)
    pars_hex_d = np.load(f)
    cov_hex_d = np.load(f)
# %%
n_data = n_cycles+1
fid_hex = np.ones(n_data)
fid_hex_d = np.ones(n_data)
fid_hex[1:n_data] = fid_L_hex[1:n_data]*np.mean(P_L_hex[1:n_data])
fid_hex_d[1:n_data] = fid_L_hex_d[1:n_data]*np.mean(P_L_hex_d[1:n_data])


# %%
def testfunc(a, b):
    print(a+b)
    return


test = (1, 2)
testfunc(*test)
