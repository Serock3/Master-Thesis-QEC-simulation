# Disable pylinter giving false positives of error.
# pylint: disable=E1101
#%% Import modules
# Standard packages
import numpy as np
import matplotlib.pyplot as plt

# Qiskit
from qiskit import *
from qiskit import QuantumCircuit
from qiskit.quantum_info import state_fidelity
from qiskit.providers.aer.noise import thermal_relaxation_error

# Local files
from simulator_program.stabilizers import (get_full_stabilizer_circuit, 
                                           syndrome_table,
                                           get_encoded_state,
                                           _unflagged_stabilizer_XZZXI,
                                           _unflagged_stabilizer_IXZZX,
                                           _unflagged_stabilizer_XIXZZ,
                                           _unflagged_stabilizer_ZXIXZ,
                                           _unflagged_stabilizer_ZZXIX,
                                           get_empty_stabilizer_circuit,
                                           StabilizerRegisters)
from simulator_program.splitting_circuits import (split_circuit, 
                                                  add_start_to_circuit,
                                                  add_stop_to_circuit)
from simulator_program.data_analysis_tools import default_execute
from simulator_program.custom_noise_models import (thermal_relaxation_model_V2, 
                                                   standard_times,
                                                   GateTimes)
from simulator_program.idle_noise import add_idle_noise_to_circuit
from simulator_program.post_process import (apply_unitary_to_den_mat,
                                            get_unitary_matrix_for_correction,
                                            post_process_den_mat)
#%% Build a circuit
T1=40e3
T2=60e3
n_cycles = 7
kwargs = {

    'reset': True,
    'recovery': False, # No recovery since we wanna split instead
    'flag': False,
    'encoding': False,
    'snapshot_type': 'density_matrix',
    'conditional': True,
    'idle_delay': 'before',
    'split_cycles': True
}

circ = get_full_stabilizer_circuit(n_cycles=n_cycles, **kwargs)
#circ.draw()
#%%
# Split circuit
circ_list = split_circuit(circ)
for i in range(len(circ_list)):
    circ_list[i], time = add_idle_noise_to_circuit(circ_list[i],
                                                   standard_times,
                                                   T1=T1, T2=T2,
                                                   return_time=True)
    # Assume every regular cycle takes equal time
    if i ==0:
        cycle_time = time['end']
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


def get_rerun_circ(stabilizers):
    """Circuit for rerunning a set of stabilizers
    
    Args: 
        stabililzers (list[int]): List of integers representing the stabilizers 
                                  to rerun.
    Returns:
        circ: QuantumCircuit object for the extra stabilizer run.
    """

    # Define registers
    qbReg = QuantumRegister(5, 'code_qubit')
    anReg = AncillaRegister(2, 'ancilla_qubit')
    crReg = ClassicalRegister(len(stabilizers), 'syndrome_bit')  # The typical register
    registers = StabilizerRegisters(qbReg, anReg, crReg, None)
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


def evolve_dm_in_time(rho, time, T1=40e3, T2=60e3):
    """Evolves a state in time under amplitude and phase damping."""

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
    if syn_int == 8:
        stabilizers = [0,1,2]
    elif syn_int == 4 or syn_int == 12:
        stabilizers = [0,1]
    else:
        # TODO: "else" should maybe be no new stabilizers, just the correction?
        stabilizers = [0,1,2,3]

    # Get the circuit
    circ = get_rerun_circ(stabilizers)
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

        # Perfor the correction 
        counts = res.data()['counts'][new_syndrome]
        rho = res.data()['end'][new_syndrome]
        rho_new = standard_QEC(rho, full_syndrome[:3], T1,T2, 
                               full_gate_times['feedback'])
        rho_full += (counts/shots)*rho_new

    return rho_full, time['end'] + full_gate_times['feedback']
# %% "Full program"
trivial = get_encoded_state(0,0)
noise_model = thermal_relaxation_model_V2(T1=T1,T2=T2,gate_times=standard_times)
n_shots = 1024*6

fid_advanced = [1.0]
data_times_advanced = [0]
total_time = 0
for i in range(n_cycles):

    # Simulate the subcircuit
    results = default_execute(circ_list[i], n_shots, 
                              gate_times=standard_times, 
                              noise_model=noise_model)
    total_time += cycle_time + standard_times['feedback']

    # For each syndrome, perform QEC
    rho_full = np.zeros([128,128], dtype=complex)
    times_dic = {}
    rho_new_dic = {}
    for syndrome in results.data()['end'].keys():

        # Get the count for this syndrome
        counts = results.data()['counts'][syndrome]
        rho = results.data()['end'][syndrome]

        # Apply normal error correction (including feedback time)
        if syndrome[:3] == '0x4' or syndrome[:3] == '0x8' or syndrome[:3] == '0xc':
            # Evolve 350ns forward as "feedback time"
            rho = evolve_dm_in_time(rho, standard_times['feedback'], T1, T2)
            rho_new, time = rerun_stabilizers(rho, syndrome, n_shots,
                                              standard_times, T1, T2)
            times_dic[syndrome] = time
        else:
            rho_new = standard_QEC(rho, syndrome, T1,T2, standard_times['feedback'])
            times_dic[syndrome] = 0

        rho_new_dic[syndrome] = rho_new
        

    # Sync the runs and combine them into a single density matrix
    max_time = max(times_dic.values())
    total_time += max_time
    for syndrome in results.data()['end'].keys():
        time_diff = max_time - times_dic[syndrome]
        rho = evolve_dm_in_time(rho_new_dic[syndrome], time_diff, T1, T2)

        # Reconstruct density matrix for ALL runs
        rho_full += (results.data()['counts'][syndrome]/n_shots)*rho


    # Set the initial state for next cycle
    if i+1 < n_cycles:
        circ_list[i+1] = add_start_to_circuit(circ_list[i+1], rho_full)
    fid.append(state_fidelity(trivial, rho_full))
    data_times.append(total_time)
    print(state_fidelity(trivial, rho_full))
print('Done :)')
#%% Plot

fig, ax = plt.subplots(1,1, figsize=(7, 5))

ax.plot(data_times[0:12], fid, 'o-', color='C0',
        label=rf'Standard QEC (without delay)')
ax.plot(data_times_advanced, fid_advanced, 'o-', color='C1',
        label=rf'Repeating uncertain stabilizers')