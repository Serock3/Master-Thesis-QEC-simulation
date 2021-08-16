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

def get_reduced_recovery(registers):
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
    #circ.z(qbReg[2]).c_if(syndrome_reg, 4)
    circ.z(qbReg[0]).c_if(syndrome_reg, 5)
    circ.x(qbReg[3]).c_if(syndrome_reg, 6)
    circ.x(qbReg[2]).c_if(syndrome_reg, 7)
    circ.z(qbReg[2]).c_if(syndrome_reg, 7)
    #circ.x(qbReg[0]).c_if(syndrome_reg, 8)
    circ.z(qbReg[3]).c_if(syndrome_reg, 9)
    circ.z(qbReg[1]).c_if(syndrome_reg, 10)
    circ.x(qbReg[1]).c_if(syndrome_reg, 11)
    circ.z(qbReg[1]).c_if(syndrome_reg, 11)
    #circ.x(qbReg[4]).c_if(syndrome_reg, 12)
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
        syndrome_reg = clReg[0][current_cycle][current_step]
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

fid = [1.0] # Fidelity at start
data_times = [0]  # Time passed at start
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


#%% ============== Iterate through all branches===============================
n_cycles = 1
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
# T1 = 40e3
# T2 = 60e3
# gate_times = standard_times_delay
gate_times = standard_times
# gate_times = GateTimes(20, 0, {'delay': 5000})


# Circuits
registers = get_registers()
circ_std_cycle = get_repeated_stabilization(registers,1,**kwargs)
circ_std_cycle.compose(get_reduced_recovery(registers),qubits=circ_std_cycle.qubits,clbits=circ_std_cycle.clbits, inplace=True)
circ_std_cycle.save_density_matrix(
                        qubits=registers.QubitRegister, label='end', conditional=kwargs['conditional'])
circ_std_cycle, times_std_cycle = add_idle_noise_to_circuit(circ_std_cycle, gate_times,T1,T2, return_time=True)

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

inital_state = logical_states(include_ancillas=None)[0]

big_dict = {'counts': 10}
not_good_with_names(big_dict, inital_state, 0, n_cycles, 0)
#%% Iteration function

def not_good_with_names(big_dict, rho, cycle, n_cycles, time_passed):
    """Recursive function to iterate through a simulate a full syndrome tree."""

    # Simulate stabilizer cycle
    circ = add_start_to_circuit(circ_std_cycle, rho) # TODO: add T1 T2 and gate_times
    res = default_execute(circ, shots=big_dict['counts'])
    time_passed += times_std_cycle['end'] 

    # Iterate over syndromes
    for syndrome in res.data()['end']:
    
        if syndrome == '0x4': # 0100
            sub_res = default_execute(add_start_to_circuit(circ_stab_0100, res.data()['end'][syndrome]), 
                                  shots=res.get_counts()[syndrome])
            time_passed += times_stab_0100
        elif syndrome == '0x8': # 1000
            sub_res = default_execute(add_start_to_circuit(circ_stab_1000, res.data()['end'][syndrome]), 
                                  shots=res.get_counts()[syndrome])
            time_passed += times_stab_1000
        elif syndrome == '0xc': # 1100
            sub_res = default_execute(add_start_to_circuit(circ_stab_1100, res.data()['end'][syndrome]), 
                                  shots=res.get_counts()[syndrome])
            time_passed += times_stab_1100

        rho = sub_res.data()['end']        
        # Store some data in (dict?)
        big_dict[syndrome] = {}
        big_dict[syndrome]['counts'] = res.data()['counts'][syndrome]
        big_dict[syndrome]['time'] = time_passed
        big_dict[syndrome]['fid'] = state_fidelity(trivial, rho_new)

        # Run next cycle
        if cycle < n_cycles:
            not_good_with_names(big_dict[syndrome], rho, cycle+1, n_cycles, time_passed)
    return

# %% Post selecting a specific error and comparing =======================
trouble_syndromes = ['0x8', '0x4', '0xc']
trivial = get_encoded_state(0,0)
noise_model = thermal_relaxation_model_V2(T1=T1,T2=T2,gate_times=standard_times)
n_shots = 1024*4

fid = [1.0] # Fidelity at start
data_times = [0]  # Time passed at start
total_time = 0

syn_fid_pre_QEC = {}
syn_fid_standard_QEC = {}
syn_fid_extra_stab = {}
circ_list_1 = [circ_list[0]]
circ_list_2 = [circ_list[0]]
fid_1 = []
fid_2 = []
for syndrome in trouble_syndromes:
    syn_fid_pre_QEC[syndrome] = []
    syn_fid_standard_QEC[syndrome] = []
    syn_fid_extra_stab[syndrome] = []

for i in range(n_cycles):

    # Simulate the subcircuit
    if i == 0:
        results = default_execute(circ_list[i], n_shots, 
                                  gate_times=standard_times, 
                                  noise_model=noise_model)
    else:
        results = default_execute(circ_list_1[i], n_shots, 
                                  gate_times=standard_times, 
                                  noise_model=noise_model)
        #results_2 = default_execute(circ_list_2[i], n_shots, 
        #                          gate_times=standard_times, 
        #                          noise_model=noise_model)                  

    # For each syndrome, perform QEC
    rho_full_1 = np.zeros([128,128], dtype=complex)
    rho_full_2 = np.zeros([128,128], dtype=complex)

    for syndrome in results.data()['end'].keys():
        #if syndrome[:3] != ps_syndrome:
        #    continue

        # Get the count for this syndrome
        counts = results.data()['counts'][syndrome]
        rho = results.data()['end'][syndrome]

        # Method 1: Standard error correction
        rho_1 = standard_QEC(rho, syndrome, T1,T2, standard_times['feedback'])

        # Method 2: Repeat some stabilizers
        if syndrome[:3] in trouble_syndromes:
            rho_2 = evolve_dm_in_time(rho, standard_times['feedback'], T1, T2)
            rho_2, time = rerun_stabilizers(rho_2, syndrome, n_shots,
                                            standard_times, T1, T2)
            # Append the difference in fidelities
            syn_fid_pre_QEC[syndrome[:3]].append(state_fidelity(rho, trivial))
            syn_fid_standard_QEC[syndrome[:3]].append(state_fidelity(rho_1, trivial))
            syn_fid_extra_stab[syndrome[:3]].append(state_fidelity(rho_2, trivial))
        else:
            rho_2 = standard_QEC(rho, syndrome, T1,T2, standard_times['feedback'])

        # Reconstruct density matrix
        rho_full_1 += (counts/n_shots)*rho_1
        rho_full_2 += (counts/n_shots)*rho_2

    # Append results
    fid_1.append(state_fidelity(rho_full_1, trivial))
    fid_2.append(state_fidelity(rho_full_2, trivial))

    # Set the initial state for next cycle
    if i+1 < n_cycles:
        circ_list_1.append(add_start_to_circuit(circ_list[i+1], rho_full_1))
        circ_list_2.append(add_start_to_circuit(circ_list[i+1], rho_full_2))

print('Done :)')

#%%

qb1 = QuantumRegister(1)
qb2 = QuantumRegister(2)

an0 = np.zeros(2**2)
an0[0] = 1.0
circ1 = QuantumCircuit(qb1, qb2)
circ2 = circ1.compose( SetDensityMatrix(np.kron(an0, np.array([1,0]))), qubits = circ1.qubits)
#circ1.set_density_matrix(np.array([1,0,0,0]))
circ2.x(qb1)
circ2.draw()
#%%
circ2 = QuantumCircuit(qb1, qb2)
circ2.cz(qb1[0], qb2[0])
circ2.compose(circ1, inplace=True, front=True)
circ2.draw()

#res = default_execute(circ2)
#%%
def add_key(test, key, thresh=2):
    test[str(key)] = {}
    if key < thresh:
        add_key(test[str(key)], key+1)
    return
test = {}
add_key(test,0)
