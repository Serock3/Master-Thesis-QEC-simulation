# %% Import modules
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from qiskit import *
from qiskit.visualization import plot_histogram

# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import thermal_relaxation_error
from qiskit.providers.aer.library import save_density_matrix, save_expectation_value                                        

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

#%% Useful functions
def fidelity_from_scratch(n_cycles, noise_model, n_shots, gate_times={}, reset=True,
        data_process_type='recovery', idle_noise=True, transpile=True, 
        snapshot_type='dm', device_properties=WACQT_device_properties, **kwargs):
    """Get the fidelity of a certain setup/configuration from only its
    parameters.
    
    Args:
        n_cycles (int): The number of stabilizer cycles to be performed.
        noise_model: The noise model to be used for simulations. If no noise is
                     to be present, use noise_model=None.
        n_shots (int): The number of runs of the circuit.
        gate_times: Can be either a dict with some gate times (in ns), or a
                    GateTimes object. If it is a dict, gate times not included 
                    will be added from standard gate times.
        reset (bool): Whether or not to reset ancilla between measurements.
                      defaults to True if left empty.
        recovery (bool): Whether or not to perform error correction after each
                         stabilizer cycle. Defaults to true if left empty.
        post_select (bool): Whether or not to use post-selection after runs,
                            discarding runs which gave a -1 eigenvalue from 
                            stabilizers. Note that this will not be performed if
                            recovery=True. Defaults to False if left empty.
        post_process (bool): Whether or not to post_process the results after
                             runs, "correcting" errors as it would have been
                             done with recovery. Note that this will not be 
                             performed if recovery or post_select are set to 
                             True. Defaults to False if left empty.
        idle_noise (bool): Whether or not to add noise to idle qubits. This
                           assumes thermal relaxation with T1=40e3 and T2=60e3. 
                           Defaults to True if left empty.
        empty_circuit (bool): Whether to create an empty circuit instead,
                              essentially only containing the encoding and
                              snapshots at times matching that of a 'normal'
                              stabilizer circuit with given gate times. Defaults
                              to False if left empty.

    Returns:
        fid (list): The average fidelity after each stabilizer cycle.
        select_counts (list, optional): The remaining runs after each cycle,
            only returned if using post_select=True.
    """

    # Get gate times missing from input
    if isinstance(gate_times, dict):
        full_gate_times = WACQT_gate_times.get_gate_times(custom_gate_times=gate_times)
    elif isinstance(gate_times, GateTimes):
        full_gate_times = gate_times
    else:
        warnings.warn('Invalid gate times, assuming WACQT_gate_times')
        full_gate_times = WACQT_gate_times

    # Check the data processing method for settings
    if data_process_type == 'recovery':
        recovery = True
        conditional = False
    elif data_process_type == 'post_select':
        recovery = False
        conditional = True
    elif data_process_type == 'empty_circuit':
        recovery = False
        conditional = False
    else:
        recovery = False
        conditional = False

    # Registers
    qb = QuantumRegister(5, 'code_qubit')
    an = AncillaRegister(2, 'ancilla_qubit')
    cr = get_classical_register(n_cycles, reset=reset, recovery=recovery, flag=False)
    readout = ClassicalRegister(5, 'readout')
    registers = StabilizerRegisters(qb, an, cr, readout)

    # Circuits
    circ = get_full_stabilizer_circuit(registers, n_cycles=n_cycles, reset=reset,
                                       recovery=recovery, flag=False,
                                       snapshot_type=snapshot_type,
                                       conditional=conditional, **kwargs)

    if transpile:
        circ = shortest_transpile_from_distribution(circ, print_cost=False,
            **device_properties)

    # Get the correct (no errors) state
    trivial = logical_states(include_ancillas=None)[0]

    # Create empty encoded circuit
    if data_process_type == 'empty_circuit':

        # Prepare the circuit
        time = get_circuit_time(circ, full_gate_times)
        circ = get_empty_noisy_circuit_v3(circ, time, full_gate_times)
        results = execute(circ, Aer.get_backend('qasm_simulator'),
            noise_model=noise_model, shots=n_shots).result()

        # Calculate fidelity at each snapshot
        fidelities = []
        if snapshot_type=='dm' or snapshot_type=='density_matrix':
            for current_cycle in range(n_cycles):
                state = results.data()['dm_' + str(current_cycle+1)]
                fidelities.append(state_fidelity(state, trivial))
        elif snapshot_type=='exp' or snapshot_type=='expectation_value':
            for current_cycle in range(n_cycles):
                fidelities.append(results.data()['exp_' + str(current_cycle+1)])
        return fidelities
        

    # Add idle noise (empty_circuit does this automatically)
    if idle_noise:
        circ = add_idle_noise_to_circuit(circ, gate_times=full_gate_times)

    # Run the circuit
    results = execute(circ, Aer.get_backend('qasm_simulator'),
        noise_model=noise_model, shots=n_shots).result()


    if data_process_type == 'recovery' or data_process_type =='none':
        fidelities = []
        if snapshot_type=='dm' or snapshot_type=='density_matrix':
            for current_cycle in range(n_cycles+1):
                state = results.data()['dm_' + str(current_cycle)]
                fidelities.append(state_fidelity(state, trivial))
        elif snapshot_type=='exp' or snapshot_type=='expectation_value':
            for current_cycle in range(n_cycles+1):
                fidelities.append(results.data()['exp_' + str(current_cycle)])

        return fidelities

    elif data_process_type == 'post_select':
        # Get the fidelity for each cycle
        if snapshot_type=='dm' or snapshot_type=='density_matrix':
            fidelities = [state_fidelity(post_selected_state, trivial) for 
                post_selected_state in get_trivial_post_select_den_mat(
                results, n_cycles)]
        elif snapshot_type=='exp' or snapshot_type=='expectation_value':
            fidelities = [state_fidelity(post_selected_state, trivial) for 
                post_selected_state in get_trivial_exp_value(
                results, n_cycles)]
        
        # Get the number of remaining shot at each cycle
        select_counts = get_trivial_post_select_counts(
            results.get_counts(), n_cycles)
        return fidelities, select_counts

    elif data_process_type == 'post_process':
        print('Warning: Post-process not implemented, exiting...')
        return []

    else:
        print('Warning: No matching data_process_type')

    return []

def get_idle_single_qubit(snapshot_times, snapshot_type='dm', T1=40e3, T2=60e3):
    """Generates a single qubit-circuit initialized in the |1> state with
    snapshots at given times

    Args:
        snapshot_times (dict): The times in the circuit to add snapshots.
        T1 (float): T1 thermal relaxation, given in ns.
        T2 (float): T2 relaxation, given in ns.

    Returns:
        circ: Qiskit circuit object of a single qubit, with snapshots at given
              times and thermal relaxation in between.
    """
    qb = QuantumRegister(1,'qubit')
    circ = QuantumCircuit(qb)
    circ.x(qb[0]) # Initialize in |1>
    circ.save_density_matrix(qb, label='start')
    time_passed = 0
    index = 0
    for key in snapshot_times:
        time_diff = snapshot_times[key]-time_passed
        if time_diff > 0:
            thrm_relax = thermal_relaxation_error(
                        T1, T2, time_diff).to_instruction()
            circ.append(thrm_relax, [qb[0]])
        if snapshot_type == 'dm' or snapshot_type == 'density_matrix':
            circ.save_density_matrix(qb, label='snap_'+str(index))
        elif snapshot_type == 'exp' or snapshot_type == 'expectation_value':
            circ.save_expectation_value(Pauli('Z'), qb,label='snap_'+str(index))
        time_passed = snapshot_times[key]
        index += 1
    return circ

def fid_single_qubit(n_cycles, n_shots, gate_times={}, snapshot_type='dm',
                     T1=40e3, T2=60e3, **kwargs):
    """Calculate the fidelity of a single qubit decay at certain times in a
    circuit corresponding to the [[5,1,3]] code.
    
    Args:
        n_cycles (int): The number of corresponding stabilizer cycles. After
                        each cycle a snapshot is performed.
        n_shots (int): The number of runs for the circuit to measure over
        gate_times: Can be either a dict with some gate times (in ns), or a
                    GateTimes object. If it is a dict, gate times not included 
                    will be added from standard gate times.
        T1 (float): T1 thermal relaxation, given in ns, defaults to 40e3.
        T2 (float): T2 thermal relaxation, given in ns, defaults to 60e3.
        
    Returns:
        fid_single (list): The fidelity after each snapshot in the circuit.
    """

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
    circ = get_full_stabilizer_circuit(registers, n_cycles=n_cycles, reset=False,
                                       recovery=False, flag=False,
                                       snapshot_type=snapshot_type,
                                       conditional=False, **kwargs)
    circ = shortest_transpile_from_distribution(circ, print_cost=False)
    circ, time = add_idle_noise_to_circuit(circ, gate_times=full_gate_times,
                                           return_time=True)

    circ_single = get_idle_single_qubit(time, snapshot_type, T1, T2)
    results = execute(circ_single, Aer.get_backend('qasm_simulator'),
        noise_model=None, shots=n_shots).result()
    fidelities = []
    
    if snapshot_type == 'dm' or snapshot_type =='density_matrix':
        trivial = results.data()['start']
        for i in range(len(time)-2):
            current_state = results.data()['snap_'+str(i+1)]
            fidelities.append(state_fidelity(current_state, trivial))
    elif snapshot_type == 'exp' or snapshot_type == 'expectation_value':
        for i in range(len(time)-2):
            fidelities.append(results.data()['snap_'+str(i+1)])
    return fidelities


# %%
# Settings used across all configurations
n_cycles = 14
n_shots = 1024*8

# Noise models
target_noise = thermal_relaxation_model_V2(gate_times=WACQT_target_times)
current_noise = thermal_relaxation_model_V2(gate_times=WACQT_demonstrated_times)


# Quantum error correction for both noise models
fid_target_WACQT = fidelity_from_scratch(n_cycles, target_noise, n_shots, 
    gate_times=WACQT_target_times, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp')
fid_demonstrated_WACQT = fidelity_from_scratch(n_cycles, current_noise, n_shots, 
    gate_times=WACQT_demonstrated_times, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp')

# Double diamond QEC
fid_target_DD = fidelity_from_scratch(n_cycles, target_noise, n_shots, 
    gate_times=WACQT_target_times, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', device_properties=diamond_device_properties)
fid_demonstrated_DD = fidelity_from_scratch(n_cycles, current_noise, n_shots, 
    gate_times=WACQT_demonstrated_times, reset=True, data_process_type='recovery',
    idle_noise=True, snapshot_type='exp', device_properties=diamond_device_properties)
print('Check!')

#%%
# No transpilation QEC
fid_target_noT = fidelity_from_scratch(n_cycles, target_noise, n_shots, 
    gate_times=WACQT_target_times, reset=True, data_process_type='recovery',
    idle_noise=True, transpile=False, snapshot_type='exp')
fid_demonstrated_noT = fidelity_from_scratch(n_cycles, current_noise, n_shots, 
    gate_times=WACQT_demonstrated_times, reset=True, data_process_type='recovery',
    idle_noise=True, transpile=False, snapshot_type='exp')
#%% PS
fid_target_PS, count_target = fidelity_from_scratch(n_cycles, target_noise, n_shots, 
    gate_times=WACQT_target_times, reset=True, data_process_type='post_select',
    idle_noise=True, snapshot_type='exp')

fid_demonstrated_PS, count_demonstrated = fidelity_from_scratch(9, current_noise, 16000, 
    gate_times=WACQT_demonstrated_times, reset=True, data_process_type='post_select',
    idle_noise=True, snapshot_type='exp')
print('Check!')
#%% Empty circuit
fid_target_empty = fidelity_from_scratch(n_cycles, target_noise, n_shots, 
    gate_times=WACQT_target_times, reset=True, data_process_type='empty_circuit',
    idle_noise=True, snapshot_type='exp')
fid_demonstrated_empty = fidelity_from_scratch(n_cycles, current_noise, n_shots, 
    gate_times=WACQT_demonstrated_times, reset=True, data_process_type='empty_circuit',
    idle_noise=True, snapshot_type='exp')

#%% Only measurements
fid_target_stab = fidelity_from_scratch(n_cycles, target_noise, n_shots, 
    gate_times=WACQT_target_times, reset=True, data_process_type='none',
    idle_noise=True, snapshot_type='exp')
fid_demonstrated_stab = fidelity_from_scratch(n_cycles, current_noise, n_shots, 
    gate_times=WACQT_demonstrated_times, reset=True, data_process_type='none',
    idle_noise=True, snapshot_type='exp')
#%% Testing single qubit
n_cycles = 14
n_shots = 1024*8
fid_target_single = fid_single_qubit(n_cycles, n_shots,
                                     gate_times=WACQT_target_times,
                                     snapshot_type='exp')
fid_demonstrated_single = fid_single_qubit(n_cycles, n_shots, 
                                           gate_times=WACQT_demonstrated_times,
                                           snapshot_type='exp')


#%% Plotting
fig, ax1 = plt.subplots(1, figsize=(10, 6))
x_dis = np.arange(0,n_cycles+1)


# Subplot 1: Target gate times
ax1.plot(x_dis, fid_target_WACQT, '-o', label='WACQT, target times')
ax1.plot(x_dis, fid_target_DD, '-o', label='Double diamond, target times')
ax1.plot(x_dis, fid_target_noT, '-o', label='No transpilation, target times')
ax1.plot(x_dis[1:15], fid_target_single, '-o', label='Single qubit decay')
#ax1.plot(x_dis, fid_demonstrated_WACQT, '-o', label='WACQT, dem. times')
#ax1.plot(x_dis, fid_demonstrated_DD, '-o', label='Double diamond, dem. times')
#ax1.plot(x_dis, fid_demonstrated_noT, '-o', label='No transpilation, dem. times')

#ax1.plot(x_dis, fid_target_stab, '-o', label='Decay of logical state, target times')
#ax1.plot(x_dis, fid_demonstrated_stab, '-o', label='Decay of logical state, dem. times')


ax1.set(ylim=(-1.0, 1.0))

ax1.set_xlabel('Number of stabilizer cycles')
ax1.set_ylabel('Expectation value of ZZZZZ')
ax1.set_title('Expectation value of simulated [[5,1,3]] QEC code')
ax1.legend()