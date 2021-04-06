# %% Import modules
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from qiskit import *
from qiskit.visualization import plot_histogram

# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import thermal_relaxation_error
                                        

from qiskit.quantum_info import partial_trace
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import state_fidelity

# Our own files
from simulator_program.custom_noise_models import (thermal_relaxation_model,
    thermal_relaxation_model_V2,
    WACQT_target_times,
    WACQT_demonstrated_times)
from simulator_program.custom_transpiler import *
from simulator_program.stabilizers import *
from simulator_program.post_select import *
from simulator_program.post_process import *
from simulator_program.idle_noise import *

# %%
def get_testing_circuit(registers, reset, recovery=True, n_cycles=15):
    circ = get_empty_stabilizer_circuit(registers)
    circ.x(0) #initialize in 1
    circ += encode_input_v2(registers)
    circ.barrier()
    circ.append(Snapshot('post_encoding', 'density_matrix', num_qubits=5), registers.QubitRegister)
    circ += get_repeated_stabilization(registers, n_cycles=n_cycles, reset=reset,
    recovery=recovery, flag=False, snapshot_type='density_matrix')
    return circ

def get_standard_transpilation(circ):
    return shortest_transpile_from_distribution(circ, print_cost=False,
        repeats=10, routing_method='sabre', initial_layout=None,
        translation_method=None, layout_method='sabre',
        optimization_level=1, **WACQT_device_properties)

def get_running_fidelity_data_den_mat_mod(results, trivial_state, n_cycles):
    """Modified version of get_running_fidelity_data_den_mat used in this file
    for correcting errors"""
    # Missing: results, 
    fidelities = []
    cl_reg_size = len(list(results.get_counts().keys())[0].split()[1])
    counts = results.get_counts()
    snapshots = reformat_density_snapshot(results)
    for current_cycle in range(n_cycles):
        fid = 0
        for key in snapshots['stabilizer_'+str(current_cycle)]:
            bin_string = bin(int(key, 16))[2:].zfill(
                cl_reg_size*(current_cycle+1))[-cl_reg_size*(current_cycle+1):]
            current_state = snapshots['stabilizer_' +
                                      str(current_cycle)][key]
            for outcome in results.get_counts():
                formated_outcome = outcome.replace(
                    ' ', '')[-cl_reg_size*(current_cycle+1):]
                if formated_outcome == bin_string:
                    fid += state_fidelity(current_state,
                                          trivial_state)*counts[outcome]
        fidelities.append(fid/n_shots)
    return fidelities

def get_running_fidelity_idle_circuit(results, trivial_state, n_cycles):
    fidelities = []
    for current_cycle in range(n_cycles):
        current_state = results.data()['snapshots']['density_matrix'][
            'stabilizer_' + str(current_cycle)][0]['value']
        fidelities.append(state_fidelity(current_state, trivial_state))
    return fidelities

def fidelity_from_scratch(n_cycles, noise_model, n_shots, gate_times={}, reset=True,
        recovery=True, post_select=False, post_process=False, idle_noise=True, 
        empty_circuit=False):


    # Get gate times missing from input
    if isinstance(gate_times, dict):
        full_gate_times = WACQT_gate_times.get_gate_times(custom_gate_times=gate_times)
    elif isinstance(gate_times, GateTimes):
        full_gate_times = gate_times
    else:
        warnings.warn('Invalid gate times, assuming WACQT_gate_times')
        full_gate_times = WACQT_gate_times

    # Registers
    qb = QuantumRegister(5, 'code_qubit')
    an = AncillaRegister(2, 'ancilla_qubit')
    cr = get_classical_register(n_cycles, reset=reset, recovery=recovery, flag=False)
    readout = ClassicalRegister(5, 'readout')
    registers = StabilizerRegisters(qb, an, cr, readout)

    # Circuits
    circ = get_testing_circuit(registers, reset, recovery, n_cycles)
    circ = get_standard_transpilation(circ)

    # Get the correct (no errors) state
    trivial = get_trivial_state(circ)

    # Create empty encoded circuit
    if empty_circuit:
        time = get_circuit_time(circ, gate_times)
        circ = get_empty_noisy_circuit_v3(circ, time, gate_times)
        # TODO: Make this part of get_empty_circuit to remove warnings
        circ = add_idle_noise_to_circuit(circ, gate_times)

        results = execute(circ, Aer.get_backend('qasm_simulator'),
            noise_model=noise_model, shots=n_shots).result()
        fid = get_running_fidelity_idle_circuit(results, trivial, n_cycles)
        return fid

    # Add idle noise (empty_circuit does this automatically)
    elif idle_noise:
        circ = add_idle_noise_to_circuit(circ, gate_times=gate_times)

    # Run the circuit
    results = execute(circ, Aer.get_backend('qasm_simulator'),
        noise_model=noise_model, shots=n_shots).result()
    if recovery:
        fid = get_running_fidelity_data_den_mat_mod(results, trivial, n_cycles)
        return fid

    if post_select:
        fidelities = [state_fidelity(post_selected_state, trivial) for post_selected_state
            in get_trivial_post_select_den_mat(results, n_cycles)]
        select_counts = get_trivial_post_select_counts(
            results.get_counts(), n_cycles)
        return fidelities, select_counts
    if post_process:

        correct_state = logical_states(include_ancillas=None)[1]
        running_fid = []
        for current_cycle in range(n_cycles):
            counts = get_subsystem_counts_up_to_cycle(
                results.get_counts(), current_cycle)
            fid = 0
            count_sum = 0
            for selected_state in results.data()['snapshots']['density_matrix']['stabilizer_' + str(current_cycle)]:
                den_mat = selected_state['value']
                memory = selected_state['memory']
                fid += state_fidelity(correct_state, post_process_den_mat(
                    den_mat, memory, current_cycle))*counts[int(memory, 16)]
                count_sum += counts[int(memory, 16)]
            running_fid.append(fid/n_shots)
        return running_fid
    else:
        fid = get_running_fidelity_data_den_mat_mod(results, trivial, n_cycles)
        return fid
    return

def get_idle_single_qubit(snapshot_times, T1=40e3, T2=60e3):
    """Generates a single qubit-circuit initialized in the |1> state with
    snapshots at given times
    """
    qb = QuantumRegister(1,'qubit')
    circ = QuantumCircuit(qb)
    circ.x(qb[0]) # Initialize in |1>
    circ.snapshot('start', 'density_matrix') # Correct state for fidelity

    time_passed = 0
    index = 0
    for key in snapshot_times:
        time_diff = snapshot_times[key]-time_passed
        if time_diff > 0:
            thrm_relax = thermal_relaxation_error(
                        T1, T2, time_diff).to_instruction()
            circ.append(thrm_relax, [qb[0]])
        circ.snapshot('snap_'+str(index),'density_matrix')
        time_passed = snapshot_times[key]
        index += 1
    return circ

def fid_single_qubit(n_cycles, n_shots, gate_times={}, T1=40e3, T2=60e3):

    # Get gate times missing from input
    if isinstance(gate_times, dict):
        full_gate_times = WACQT_gate_times.get_gate_times(custom_gate_times=gate_times)
    elif isinstance(gate_times, GateTimes):
        full_gate_times = gate_times
    else:
        warnings.warn('Invalid gate times, assuming WACQT_gate_times')
        full_gate_times = WACQT_gate_times

    # Registers
    qb = QuantumRegister(5, 'code_qubit')
    an = AncillaRegister(2, 'ancilla_qubit')
    cr = get_classical_register(n_cycles, reset=False, recovery=False, flag=False)
    readout = ClassicalRegister(5, 'readout')
    registers = StabilizerRegisters(qb, an, cr, readout)

    # Circuits
    circ = get_testing_circuit(registers, reset=False, recovery=False, n_cycles=n_cycles)
    circ = get_standard_transpilation(circ)
    circ, time = add_idle_noise_to_circuit(circ, gate_times=gate_times,return_time=True)

    circ_single = get_idle_single_qubit(time)
    results = execute(circ_single, Aer.get_backend('qasm_simulator'),
        noise_model=None, shots=n_shots).result()
    fid_single = []
    correct_state = results.data()['snapshots']['density_matrix']['start'][0]['value']
    for i in range(len(time)-2):
        current_state = results.data()['snapshots']['density_matrix'][
            'snap_'+str(i+1)][0]['value']
        fid_single.append(state_fidelity(current_state,correct_state))
    return fid_single


#%% Single qubit decay
fid_target_single = fid_single_qubit(n_cycles, n_shots, gate_times=WACQT_target_times)
fid_demonstrated_single = fid_single_qubit(n_cycles, n_shots, gate_times=WACQT_demonstrated_times)

#%%
n_cycles = 15
n_shots = 4096
target_noise = thermal_relaxation_model_V2(gate_times=WACQT_target_times)
current_noise = thermal_relaxation_model_V2(gate_times=WACQT_demonstrated_times)
fid_target_QEC = fidelity_from_scratch(n_cycles, target_noise, n_shots, 
    gate_times=WACQT_target_times, reset=True, recovery=True, post_select=False,
    post_process=False, idle_noise=True, empty_circuit=False)
print('Check!')
fid_demonstrated_QEC = fidelity_from_scratch(n_cycles, current_noise, n_shots, 
    gate_times=WACQT_demonstrated_times, reset=True, recovery=True, post_select=False,
    post_process=False, idle_noise=True, empty_circuit=False)
print('Check!')
fid_target_PS, count_target_PS = fidelity_from_scratch(n_cycles, target_noise, n_shots, 
    gate_times=WACQT_target_times, reset=True, recovery=False, post_select=True,
    post_process=False, idle_noise=True, empty_circuit=False)
print('Check!')
fid_demonstrated_PS, count_demonstrated_PS = fidelity_from_scratch(9, current_noise, 16000, 
    gate_times=WACQT_demonstrated_times, reset=True, recovery=False, post_select=True,
    post_process=False, idle_noise=True, empty_circuit=False)
print('Check!')
fid_target_PP = fidelity_from_scratch(n_cycles, target_noise, n_shots, 
    gate_times=WACQT_target_times, reset=True, recovery=False, post_select=False,
    post_process=True, idle_noise=True, empty_circuit=False)
print('Check!')
fid_demonstrated_PP = fidelity_from_scratch(n_cycles, current_noise, n_shots, 
    gate_times=WACQT_demonstrated_times, reset=True, recovery=False, post_select=False,
    post_process=True, idle_noise=True, empty_circuit=False)
print('Check!')
fid_target_stab = fidelity_from_scratch(n_cycles, target_noise, n_shots, 
    gate_times=WACQT_target_times, reset=True, recovery=False, post_select=False,
    post_process=False, idle_noise=True, empty_circuit=False)
print('Check!')
fid_demonstrated_stab = fidelity_from_scratch(n_cycles, current_noise, n_shots, 
    gate_times=WACQT_demonstrated_times, reset=True, recovery=False, post_select=False,
    post_process=False, idle_noise=True, empty_circuit=False)
print('Check!')
fid_target_empty = fidelity_from_scratch(n_cycles, target_noise, n_shots, 
    gate_times=WACQT_target_times, reset=True, recovery=True, post_select=False,
    post_process=False, idle_noise=True, empty_circuit=True)
print('Check!')
fid_demonstrated_empty = fidelity_from_scratch(n_cycles, current_noise, n_shots, 
    gate_times=WACQT_demonstrated_times, reset=True, recovery=True, post_select=False,
    post_process=False, idle_noise=True, empty_circuit=True)
print('Check!')
#%% Curve fitting with OLS
fid_list = [fid, fid_i, fid_res, fid_res_i]
theta_list = []
for fidelity in fid_list:
    x_D = np.ones((n_cycles,2))
    for i in range(n_cycles):
        x_D[i][1] += i
    y = np.log( np.reshape(np.asarray(fidelity), (n_cycles,1)) )
    theta = np.dot(np.dot(np.linalg.inv(np.dot(x_D.T, x_D)), x_D.T), y)
    theta_list.append(theta)

x = np.linspace(0,n_cycles+1,100)
y_pred_list = []
for theta in theta_list:
    y_pred = np.exp(theta[0]) * np.exp(x*theta[1])
    y_pred_list.append(y_pred)

# Calculate MSE
MSE = []
for i in range(len(fid_list)):
    err = 0.
    for cycle in range(n_cycles):
        y_pred = np.exp(theta_list[i][0]) * np.exp((cycle+1)*theta_list[i][1])
        err += (y_pred-fid_list[i][cycle])**2
    MSE.append(err)
print('MSE: ')
for i in MSE:
    print(i)
#%% Plotting
fig, ax = plt.subplots(2, figsize=(10, 10))
x_dis = np.arange(1,n_cycles+1)
ax1, ax2 = ax

ax1.plot(x_dis, fid_target_QEC, '-o', label='Error correction')
ax1.plot(x_dis, fid_target_PS, '-o', label='Post select correct states')
#ax1.plot(x_dis, fid_target_PP, '-o', label='Post processing data')
ax1.plot(x_dis, fid_target_stab, '-o', label='Only measurements, no correction')
#ax1.plot(x_dis, fid_target_empty, '-o', label='Decay of [[5,1,3]] logical state')
ax1.plot(x_dis, fid_target_single, '-o', label='Decay of single qubit in |1>')

ax2.plot(x_dis, fid_demonstrated_QEC, '-o', label='Error correction')
ax2.plot(x_dis[0:7], fid_demonstrated_PS[0:7], '-o', label='Post select correct states')
#ax2.plot(x_dis, fid_demonstrated_PP, '-o', label='Post processing data')
ax2.plot(x_dis, fid_demonstrated_stab, '-o', label='Only measurements, no correction')
#ax2.plot(x_dis, fid_demonstrated_empty, '-o', label='Decay of [[5,1,3]] logical state')
ax2.plot(x_dis, fid_demonstrated_empty, '-o', label='Decay of single qubit in |1>')

ax1.set(ylim=(0.0, 1.0))
ax2.set(ylim=(0.0, 1.0))

ax1.set_xlabel('Number of stabilizer cycles (2960 ns each)')
ax1.set_ylabel('Average fidelity')
ax1.set_title('Average fidelity of simulated [[5,1,3]] QEC code using target gate times')
ax1.legend()

ax2.set_xlabel('Number of stabilizer cycles (14160 ns each)')
ax2.set_ylabel('Average fidelity')
ax2.set_title('Average fidelity, experimentally demonstrated gate times')
ax2.legend()
#fig.savefig('fidelities.pdf')

#%% Other stuff: Checking time difference between cycles
# Registers
qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
cr = get_classical_register(n_cycles, reset=False, recovery=False, flag=False)
readout = ClassicalRegister(5, 'readout')
registers = StabilizerRegisters(qb, an, cr, readout)

# Circuits
circ = get_testing_circuit(registers, reset=False, recovery=False, n_cycles=n_cycles)
circ = get_standard_transpilation(circ)
circ, time = add_idle_noise_to_circuit(circ, gate_times=WACQT_demonstrated_times,return_time=True)

time_diff = 0
a = True
for key in time:
    current_time = time[key]
    if a:
        a = False
        previous_time = time[key]
        continue
    time_diff = current_time - previous_time
    print(time_diff)
    previous_time = time[key]



    
# %% Single qubit decay

qb = QuantumRegister(1, 'code_qubit')
circ = QuantumCircuit(qb)
circ.y(qb[0])
circ.snapshot('start','density_matrix')
circ.append(thermal_relaxation_error(t1=40e3, t2=60e3, time=50000),[0])
circ.snapshot('finish','density_matrix')

results = execute(circ, Aer.get_backend('qasm_simulator'),
    noise_model=None, shots=n_shots).result()
dm_start = results.data()['snapshots']['density_matrix']['start'][0]['value']
dm_end = results.data()['snapshots']['density_matrix']['finish'][0]['value']
print(state_fidelity(dm_start, dm_end))
print(circ)
print(dm_start)