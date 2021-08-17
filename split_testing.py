# Disable pylinter giving false positives of error.
# pylint: disable=E1101
#
#
# TODO:
#   - Pre-define all circuits (with idle noise & feedback)
#   - use .compose(circ) for add_start_to_circuit

#%% Import modules
# Standard packages
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pickle

# Qiskit
from qiskit import *
from qiskit import QuantumCircuit
from qiskit.quantum_info import state_fidelity
from qiskit.providers.aer.noise import thermal_relaxation_error

# Local files
from simulator_program.stabilizers import (get_full_stabilizer_circuit, 
                                           get_repeated_stabilization,
                                           get_registers,
                                           get_snapshot_label,
                                           syndrome_table,
                                           get_encoded_state,
                                           logical_states,
                                           _unflagged_stabilizer_XZZXI,
                                           _unflagged_stabilizer_IXZZX,
                                           _unflagged_stabilizer_XIXZZ,
                                           _unflagged_stabilizer_ZXIXZ,
                                           _unflagged_stabilizer_ZZXIX,
                                           get_empty_stabilizer_circuit,
                                           StabilizerRegisters,
                                           label_counter)
from simulator_program.splitting_circuits import (split_circuit, 
                                                  add_start_to_circuit,
                                                  add_stop_to_circuit)
from simulator_program.data_analysis_tools import default_execute
from simulator_program.custom_noise_models import (thermal_relaxation_model_V2, 
                                                   standard_times,
                                                   standard_times_delay,
                                                   WACQT_target_times,
                                                   GateTimes)
from simulator_program.idle_noise import add_idle_noise_to_circuit
from simulator_program.post_process import (apply_unitary_to_den_mat,
                                            get_unitary_matrix_for_correction,
                                            post_process_den_mat)
#%% Correction strategies
def standard_QEC(rho, syndrome, T1=40e3,T2=60e3, feedback_time=350):
    """Evolves a 7-qb density matrix in a way which corresponds to standard QEC,
    using a lookup-table and a feedback time to process results."""

    thrm_relax = thermal_relaxation_error(T1, T2, feedback_time).to_instruction()

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
    #crReg = ClassicalRegister(len(stabilizers), 'syndrome_bit')  # The typical register
    #registers = StabilizerRegisters(qbReg, anReg, crReg, None)
    circ = get_empty_stabilizer_circuit(registers)

    stabilizer_list = [_unflagged_stabilizer_XZZXI,
                       _unflagged_stabilizer_IXZZX,
                       _unflagged_stabilizer_XIXZZ,
                       _unflagged_stabilizer_ZXIXZ,
                       _unflagged_stabilizer_ZZXIX]
    syn_bit_list = [registers.SyndromeRegister[n] for n in range(len(stabilizers))]

    # Append all stabilizers to perform
    for i in range(len(stabilizers)):
        stab_index = stabilizers[i]
        circ.compose(stabilizer_list[stab_index](registers, syn_bit=syn_bit_list[i]), 
                     inplace=True)
        circ.barrier()
    return circ

def get_reduced_recovery(registers, sydromes_to_remove = ['0100','1000','1100']):
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
    circ.z(qbReg[4]).c_if(syndrome_reg, 2)
    circ.x(qbReg[2]).c_if(syndrome_reg, 3)
    if not '0100' in sydromes_to_remove:
        circ.z(qbReg[2]).c_if(syndrome_reg, 4)
    circ.z(qbReg[0]).c_if(syndrome_reg, 5)
    circ.x(qbReg[3]).c_if(syndrome_reg, 6)
    circ.x(qbReg[2]).c_if(syndrome_reg, 7)
    circ.z(qbReg[2]).c_if(syndrome_reg, 7)
    if not '1000' in sydromes_to_remove:
        circ.x(qbReg[0]).c_if(syndrome_reg, 8)
    circ.z(qbReg[3]).c_if(syndrome_reg, 9)
    circ.z(qbReg[1]).c_if(syndrome_reg, 10)
    circ.x(qbReg[1]).c_if(syndrome_reg, 11)
    circ.z(qbReg[1]).c_if(syndrome_reg, 11)
    if not '1100' in sydromes_to_remove:
        circ.x(qbReg[4]).c_if(syndrome_reg, 12)
    circ.x(qbReg[0]).c_if(syndrome_reg, 13)
    circ.z(qbReg[0]).c_if(syndrome_reg, 13)
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
    if syndrome == '0x4': # 0100 0101 0110 0111
        circ.z(qbReg[2]).c_if(syndrome_reg, 4-4)
        circ.z(qbReg[0]).c_if(syndrome_reg, 5-4)
        circ.x(qbReg[3]).c_if(syndrome_reg, 6-4)
        circ.x(qbReg[2]).c_if(syndrome_reg, 7-4)
        circ.z(qbReg[2]).c_if(syndrome_reg, 7-4)
    elif syndrome == '0x8': # 1000
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
    elif syndrome == '0xc': # 1100
        circ.x(qbReg[4]).c_if(syndrome_reg, 12-12)
        circ.x(qbReg[0]).c_if(syndrome_reg, 13-12)
        circ.z(qbReg[0]).c_if(syndrome_reg, 13-12)
        circ.x(qbReg[4]).c_if(syndrome_reg, 14-12)
        circ.z(qbReg[4]).c_if(syndrome_reg, 14-12)
        circ.x(qbReg[3]).c_if(syndrome_reg, 15-12)
        circ.z(qbReg[3]).c_if(syndrome_reg, 15-12)
    
    return circ

def evolve_dm_in_time(rho, time, T1=40e3, T2=60e3):
    """Evolves a state in time under amplitude and phase damping."""

    # TODO: use rho.evolve instead?
    dim = np.log2(len(rho))
    circ = QuantumCircuit(dim)
    circ.set_density_matrix(rho)
    thrm_relax = thermal_relaxation_error(T1, T2, time).to_instruction()
    for qubit in circ.qubits:
        circ.append(thrm_relax, [qubit])
    circ.save_density_matrix(circ.qubits, label='tmp')
    results_tmp = execute(circ, Aer.get_backend('qasm_simulator'),
                          shots=1).result()
    return results_tmp.data()['tmp']

def rerun_stabilizers(rho, syndrome, shots, gate_times={}, T1=40e3, T2=60e3):
    """Runs and processes a set of extra stabilizers."""

    # Get gate times missing from input
    if isinstance(gate_times, dict):
        full_gate_times = standard_times.get_gate_times(
            custom_gate_times=gate_times)
    elif isinstance(gate_times, GateTimes):
        full_gate_times = gate_times
    else:
        warnings.warn('Invalid gate times, assuming standard_times')
        full_gate_times = standard_times

    # Interpret syndrome
    # TODO: Make into separate function?
    syn_int = int(syndrome[:3], 16)
    if syn_int == 8: # 1000
        stabilizers = [0,1,2]
    elif syn_int == 4 or syn_int == 12: # 0100 or 1100
        stabilizers = [0,1]
    else:
        # TODO: "else" should maybe be no new stabilizers, just the correction?
        stabilizers = [0,1,2,3]

    # Get the circuit
    circ = get_partial_stab_cycle(stabilizers)
    circ, time = add_idle_noise_to_circuit(circ, gate_times=full_gate_times, 
                                           T1=T1, T2=T2, return_time=True)
    circ = add_start_to_circuit(circ, rho)
    add_stop_to_circuit(circ)

    # Run it
    noise_model = thermal_relaxation_model_V2(T1, T2, full_gate_times)
    res = default_execute(circ, shots=shots, noise_model=noise_model)

    # Construct full syndrome
    rho_full = np.zeros([128,128], dtype=complex)
    for new_syndrome in res.data()['end']:
        if syn_int == 8:
            # Assume 4th stabilizer would give 1
            full_syndrome = hex(8+int(new_syndrome[2:],16)) + syndrome[2]
        elif syn_int == 4:
            # Assume 3rd + 4th stabilizer would give 01
            full_syndrome = hex(4+int(new_syndrome[2:],16)) + syndrome[2]
        elif syn_int == 12:
            # Assume 3rd + 4th stabilizer would give 11
            full_syndrome = hex(12+int(new_syndrome[2:],16)) + syndrome[2]

        # Perform the correction 
        counts = res.data()['counts'][new_syndrome]
        rho = res.data()['end'][new_syndrome]
        rho_new = standard_QEC(rho, full_syndrome[:3], T1,T2, 
                               full_gate_times['feedback'])
        rho_full += (counts/shots)*rho_new

    return rho_full, time['end'] + full_gate_times['feedback']

#%% Iteration function


def not_good_with_names(big_dict, rho, cycle, n_cycles, start_time, special_recoveries):
    """Recursive function to iterate through a simulate a full syndrome tree."""

    # Simulate stabilizer cycle
    circ = add_start_to_circuit(circ_std_cycle, rho) # TODO: add T1 T2 and gate_times
    res = default_execute(circ, shots=big_dict['counts'], gate_times=gate_times, T1=T1, T2=T2)
    #return circ, res # TODO: REMOVE THIS!!!

    # Iterate over syndromes
    for syndrome in res.data()['end']:
        # TODO: Add '0xe' and '0x6' to run first stabilizer again
        #if syndrome == '0x4': # 0100
        #    sub_res = default_execute(add_start_to_circuit(circ_stab_0100, res.data()['end'][syndrome]),
        #                              gate_times=gate_times, T1=T1, T2=T2, 
        #                              shots=res.data()['counts'][syndrome])
        #    end_time = start_time + times_std_cycle['end']  + times_stab_0100['end']
        #    rho = sub_res.data()['end']
        #elif syndrome == '0x8': # 1000
        #    sub_res = default_execute(add_start_to_circuit(circ_stab_1000, res.data()['end'][syndrome]), 
        #                              gate_times=gate_times, T1=T1, T2=T2, 
        #                              shots=res.data()['counts'][syndrome])
        #    end_time = start_time + times_std_cycle['end']  +  times_stab_1000['end']
        #    rho = sub_res.data()['end']
        #elif syndrome == '0xc': # 1100
        #    sub_res = default_execute(add_start_to_circuit(circ_stab_1100, res.data()['end'][syndrome]), 
        #                              gate_times=gate_times, T1=T1, T2=T2, 
        #                              shots=res.data()['counts'][syndrome])
        #    end_time = start_time + times_std_cycle['end']  +  times_stab_1100['end']
        #    rho = sub_res.data()['end']
        if syndrome in special_recoveries:
            sub_res = default_execute(add_start_to_circuit(special_recoveries[syndrome][0], 
                                                           res.data()['end'][syndrome]),
                                      gate_times=gate_times, T1=T1, T2=T2, 
                                      shots=res.data()['counts'][syndrome])
            end_time = start_time + times_std_cycle['end']  + special_recoveries[syndrome][1]
            rho = sub_res.data()['end']
            
        else:
            rho = res.data()['end'][syndrome] 
            end_time = start_time + times_std_cycle['end']   

        # Store some data in (dict?)
        big_dict[syndrome] = {} # Big dict energy
        big_dict[syndrome]['counts'] = res.data()['counts'][syndrome]
        big_dict[syndrome]['time'] = end_time
        big_dict[syndrome]['fid'] = state_fidelity(initial_state, rho)

        # Run next cycle
        if cycle+1 < n_cycles:
            not_good_with_names(big_dict[syndrome], rho, cycle+1, n_cycles, end_time, special_recoveries)
    return
    
#%% ============== Iterate through all branches===============================
n_cycles = 9
kwargs = {
    'reset': True,
    'recovery': False, # No recovery since we wanna split instead
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
circ_stab_1000 = get_partial_stab_cycle(registers, [0,1,2])
circ_stab_1000.compose(get_partial_recovery(registers, '0x8'),qubits=circ_stab_1000.qubits,clbits=circ_stab_1000.clbits, inplace=True)
circ_stab_1000.save_density_matrix(qubits=registers.QubitRegister, label='end', 
                                   conditional=False)
circ_stab_1000, times_stab_1000 = add_idle_noise_to_circuit(circ_stab_1000, gate_times=gate_times, T1=T1, T2=T2, return_time=True)


circ_stab_0100 = get_partial_stab_cycle(registers, [0,1])
circ_stab_0100.compose(get_partial_recovery(registers, '0x4'),qubits=circ_stab_0100.qubits,clbits=circ_stab_0100.clbits, inplace=True)
circ_stab_0100.save_density_matrix(qubits=registers.QubitRegister, label='end', 
                                  conditional=False)
circ_stab_0100, times_stab_0100 = add_idle_noise_to_circuit(circ_stab_0100, gate_times=gate_times, T1=T1, T2=T2, return_time=True)

circ_stab_1100 = get_partial_stab_cycle(registers, [0,1])
circ_stab_1100.compose(get_partial_recovery(registers, '0xc'),qubits=circ_stab_1100.qubits,clbits=circ_stab_1100.clbits, inplace=True)
circ_stab_1100.save_density_matrix(qubits=registers.QubitRegister, label='end', 
                                  conditional=False)
circ_stab_1100, times_stab_1100 = add_idle_noise_to_circuit(circ_stab_1100, gate_times=gate_times, T1=T1, T2=T2, return_time=True)

# Different recovery schemes
special_recoveries = {'0x4': (circ_stab_0100,times_stab_0100['end']),
                      '0x8': (circ_stab_1000,times_stab_1000['end']),
                      '0xc': (circ_stab_1100,times_stab_1100['end'])}
standard_recoveries = {}

# Circuits
label_counter.value = 0

circ_std_cycle = get_repeated_stabilization(registers,1,**kwargs, generator_snapshot=False, idle_snapshots=False)
circ_std_cycle.compose(get_reduced_recovery(registers, [bin(int(key,16))[2:].zfill(4) for key in special_recoveries]),
                       qubits=circ_std_cycle.qubits,
                       clbits=circ_std_cycle.clbits, inplace=True)
circ_std_cycle.save_density_matrix(
                        qubits=registers.QubitRegister, label='end', conditional=kwargs['conditional'])
circ_std_cycle, times_std_cycle = add_idle_noise_to_circuit(circ_std_cycle, 
                                                            gate_times,T1,T2,
                                                            return_time=True)
                                                            

#%% Running simulations

n_shots = 1024*4
initial_state = logical_states(include_ancillas=None)[0]
standard_res_dict = {'counts': n_shots, 'time': 0, 'fid': 1}
not_good_with_names(standard_res_dict, initial_state, 0, n_cycles, 0, standard_recoveries)
print('Halfway there')
special_res_dict = {'counts': n_shots, 'time': 0, 'fid': 1}
not_good_with_names(special_res_dict, initial_state, 0, n_cycles, 0, special_recoveries)
print('(Living on a prayer)')

#%% Plot scatter
fig, ax = plt.subplots(1,1, figsize=(7, 5))

def scatter_plot(ax,big_dict, c='b'):
    ax.scatter(big_dict['time'],big_dict['fid'], s=big_dict['counts'], c=c, marker='o')
    
    for key in big_dict:
        if key == 'counts' or key == 'time' or key == 'fid':
            continue
        scatter_plot(ax,big_dict[key])
        # ax.scatter(big_dict[key]['time'],big_dict[key]['fid'], s=big_dict[key]['counts'], c='b', marker='o')

    
scatter_plot(ax,test_dict)
ax.set_ylim([0.0,1.0])
ax.set_xlim([0.0,n_cycles*6000.0])



#%% Reform dict into arrays of each branch
times = []
fids = []
counts = []
cycles = []

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

#%% Append every shot into array

def append_shots(big_dict, current_cycle,times_full, fids_full, cycles_full):
    global index
    times_full[index:index+big_dict['counts']] = big_dict['time']
    fids_full[index:index+big_dict['counts']] = big_dict['fid']
    cycles_full[index:index+big_dict['counts']] = current_cycle
    index += big_dict['counts']
    for key in big_dict:
        if key == 'counts' or key == 'time' or key == 'fid':
            continue
        append_shots(big_dict[key], current_cycle+1, times_full, fids_full, cycles_full)
            
index = 0
n_points = (n_cycles+1)*n_shots
times_full_standard = np.zeros(n_points)
fids_full_standard = np.zeros(n_points)
cycles_full_standard = np.zeros(n_points)
append_shots(standard_res_dict, 0,times_full_standard,fids_full_standard,cycles_full_standard)
     
index = 0
n_points = (n_cycles+1)*n_shots
times_full_special = np.zeros(n_points)
fids_full_special = np.zeros(n_points)
cycles_full_special = np.zeros(n_points)
append_shots(special_res_dict, 0,times_full_special,fids_full_special,cycles_full_special)

#%% Plotting functions

# Plot by bins
def plot_by_bins(ax, bins, fids_full, times_full, cycles_full,c='b'):
    time_bins = np.linspace(0,max(times_full),bins+1)
    for i in range(bins):
        ax.scatter((time_bins[i]+time_bins[i+1])/2,
                   np.mean(fids_full[np.logical_and(time_bins[i]<times_full, times_full<time_bins[i+1])]),
                   c=c, marker='o')

# bins = n_cycles+2
# time_bins = np.linspace(0,max(times),bins+1)
# for i in range(bins):
#     ax.scatter((time_bins[i]+time_bins[i+1])/2,
#                np.mean(fids_full[np.logical_and(time_bins[i]<times_full, times_full<time_bins[i+1])]),
#                c='b', marker='o')

# Plot curve fitting
def plot_curvefit(ax, fids_full, times_full, cycles_full, color='C1'):
    def monoExp(t, T, c, A):
        return (A-c) * np.exp(-t/T) + c
    p0 = (40e3, 0, 0.9) # start with values near those we expect
    pars_full, cov_full = scipy.optimize.curve_fit(monoExp, times_full, fids_full, p0)
    x = np.linspace(0,max(times_full),200)
    ax.plot(x, monoExp(x, *pars_full), '-', color=color,zorder=15,
            label=rf'Curve fit, $T_L ={pars_full[0]/1000:.1f}$ μs')
    return pars_full, cov_full

# Plot by cycle
def plot_by_cycle(ax, fids_full, times_full, cycles_full, color='C0'):
    cycles = int(max(cycles_full)+1)
    fid_cycle = np.zeros(cycles) 
    times_cycle = np.zeros(cycles)   
    for i in range(cycles):    
        fid_cycle[i] += np.mean(fids_full[cycles_full==i])
        times_cycle[i] += np.mean(times_full[cycles_full==i])
    ax.plot(times_cycle,fid_cycle, '-o', color=color, label='Grouped by cycle')





#%% Testing plots
fig, ax = plt.subplots(1,1, figsize=(7, 5))
bins = n_cycles+2

plot_by_bins(ax, bins, fids_full_standard, times_full_standard, cycles_full_standard,c='b')
pars_standard, cov_standard = plot_curvefit(ax, fids_full_standard, times_full_standard, cycles_full_standard, color='C1')
plot_by_cycle(ax, fids_full_standard, times_full_standard, cycles_full_standard, color='C0')

plot_by_bins(ax, bins, fids_full_special, times_full_special, cycles_full_special,c='C4')
pars_special, cov_special = plot_curvefit(ax, fids_full_special, times_full_special, cycles_full_special, color='C2')
plot_by_cycle(ax, fids_full_special, times_full_special, cycles_full_special, color='C3')

# Plot settings
ax.legend()
ax.set_ylim([0.0,1.0])
ax.set_xlim([0.0,max(times_full_special)])
#%% Plot Many lines

def lines_plot(ax,big_dict, times, fids, current_cycle):
    times_new_branch = times
    times_new_branch[current_cycle] = big_dict['time']
    fids_new_branch = fids
    fids_new_branch[current_cycle] = big_dict['fid']
    # counts_new_branch = counts
    # counts_new_branch[current_cycle] = big_dict['counts']
    if current_cycle == n_cycles-1:
        ax.plot(times,fids, color = 'b', alpha = np.sqrt(big_dict['counts'])/50, linewidth=np.sqrt(big_dict['counts']))
    else:
        for key in big_dict:
            if not(key == 'counts' or key == 'time' or key == 'fid'):
                lines_plot(ax,big_dict[key], times_new_branch, fids_new_branch, current_cycle + 1)

times = [0]*n_cycles
fids = [0]*n_cycles
# counts = [0]*n_cycles

fig, ax = plt.subplots(1,1, figsize=(7, 5))
lines_plot(ax,big_dict, times, fids, 0)


#%% Save
pickle.dump(standard_res_dict, open('split_data/standard_res_dict.dat', "wb" ))
pickle.dump(special_res_dict, open('split_data/special_res_dict.dat', "wb" ))
#%% load 
standard_res_dict = pickle.load( open( 'split_data/standard_res_dict.dat', "rb" ) )
special_res_dict = pickle.load( open( 'split_data/special_res_dict.dat', "rb" ) )

