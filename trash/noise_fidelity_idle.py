# MOST OF THIS FILE IS DEPRECATED, DUE TO THE NEW RELEASE OF QISKIT, RENDERING
# FUNCTIONS OBSOLETE. FOR NEWER VERSION, REFER TO plotting_expval.py
#
# The main purpose of this script is to run a range of different simulations
# on the [[5,1,3]] code. Possible options include whether to add idle noise, if
# and how to process the data (error correction, post-selection etc), and more.
# Main function for this is 'fidelity_from_scratch', but for single qubits then
# 'fid_single_qubit' should be used instead.

# %% Import modules
import matplotlib.pyplot as plt
import numpy as np
from qiskit import *

# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import thermal_relaxation_error


from qiskit.quantum_info import state_fidelity

# Our own files
from simulator_program.custom_noise_models import (
    thermal_relaxation_model_V2,
    WACQT_target_times,
    WACQT_demonstrated_times)
from simulator_program.custom_transpiler import *
from simulator_program.stabilizers import *
from simulator_program.post_select import *
from simulator_program.post_process import *
from simulator_program.idle_noise import *

# %%


def get_testing_circuit(registers, reset, recovery=True, n_cycles=15,
                        initial_state=0, **kwargs):
    """Create a [[5,1,3]] stabilizer circuit, including encoding and snapshots.

    Args:
        registers (class): StabilizerRegisters class, can be created easily by
            calling StabilizerRegisters() from simulator_program.stabilizers.py.
        reset (bool): Option whether or not to reset ancilla between msmnts.
        recovery (bool): Option whether to perform error correction after each
            cycle. Defaults to true if left empty.
        n_cycles (int): The number of stabilizer cycles for the circuit.
        initial_state (int): The initial state to encode into 5 qubits. Can only
            be initialized to 0 or 1 currently.

    Returns:
        circ: Qiskit circuit object, containing the full stabilizer circuit.
    """
    circ = get_empty_stabilizer_circuit(registers)
    if initial_state == 1:
        circ.x(0)  # initialize in 1
    circ += encode_input_v2(registers)
    circ.barrier()
    circ.append(Snapshot('post_encoding', 'density_matrix',
                         num_qubits=5), registers.QubitRegister)
    circ += get_repeated_stabilization(registers, n_cycles=n_cycles, reset=reset,
                                       recovery=recovery, flag=False, snapshot_type='density_matrix', **kwargs)
    return circ


def get_standard_transpilation(circ):
    """
    DEPRECATED

    Transpiles a circuit using standard settings. This function should be
    removed, as these settings are now made default when using
    shortest_transpile_from_distribution.
    """
    return shortest_transpile_from_distribution(circ, print_cost=False,
                                                repeats=10, routing_method='sabre', initial_layout=None,
                                                translation_method=None, layout_method='sabre',
                                                optimization_level=1, **WACQT_device_properties)


def get_running_fidelity_data_den_mat_mod(results, trivial_state, n_cycles):
    """Modified version of get_running_fidelity_data_den_mat used in this file
    for measuring fidelity when performing quantum error correction.

    Args:
        results: result object from a qasm simulation.
        trivial_state (np.array): density matrix corresponding to the correct,
            noise free, output of simulation.
        n_cycles (int): The number of stabilizer cycles performed.

    Returns:
        fidelities (list): The average fidelity after each stabilizer cycle.
    """

    # Get the number of shots and their counts
    counts = results.get_counts()
    n_shots = 0
    for value in counts.values():
        n_shots += value

    fidelities = []
    cl_reg_size = len(list(results.get_counts().keys())[0].split()[1])
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
    """Calculates the fidelity of an idle [[5,1,3]] circuit (one without 
    measurements). Note that this assumes that snapshots are used at certain
    points in the circuit, as per default in functions from stabilizers.py.

    Args:
        results: result object from a qasm simulation.
        trivial_state (np.array): density matrix corresponding to the correct,
            noise free, output of simulation.
        n_cycles (int): The number of stabilizer cycles performed.

    Returns:
        fidelities (list): The average fidelity after each stabilizer cycle.
    """
    fidelities = []
    for current_cycle in range(n_cycles):
        current_state = results.data()['snapshots']['density_matrix'][
            'stabilizer_' + str(current_cycle)][0]['value']
        fidelities.append(state_fidelity(current_state, trivial_state))
    return fidelities


def fidelity_from_scratch(n_cycles, noise_model, n_shots, gate_times={}, reset=True,
                          recovery=True, post_select=False, post_process=False, idle_noise=True,
                          empty_circuit=False, transpile=True, **kwargs):
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
        full_gate_times = WACQT_gate_times.get_gate_times(
            custom_gate_times=gate_times)
    elif isinstance(gate_times, GateTimes):
        full_gate_times = gate_times
    else:
        warnings.warn('Invalid gate times, assuming WACQT_gate_times')
        full_gate_times = WACQT_gate_times

    # Registers
    qb = QuantumRegister(5, 'code_qubit')
    an = AncillaRegister(2, 'ancilla_qubit')
    cr = get_classical_register(
        n_cycles, reset=reset, recovery=recovery, flag=False)
    readout = ClassicalRegister(5, 'readout')
    registers = StabilizerRegisters(qb, an, cr, readout)

    # Circuits
    circ = get_testing_circuit(registers, reset, recovery, n_cycles, **kwargs)
    if transpile:
        circ = shortest_transpile_from_distribution(circ, print_cost=False)

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

    Args:
        snapshot_times (dict): The times in the circuit to add snapshots.
        T1 (float): T1 thermal relaxation, given in ns.
        T2 (float): T2 relaxation, given in ns.

    Returns:
        circ: Qiskit circuit object of a single qubit, with snapshots at given
              times and thermal relaxation in between.
    """
    qb = QuantumRegister(1, 'qubit')
    circ = QuantumCircuit(qb)
    circ.x(qb[0])  # Initialize in |1>
    circ.snapshot('start', 'density_matrix')  # Correct state for fidelity

    time_passed = 0
    index = 0
    for key in snapshot_times:
        time_diff = snapshot_times[key]-time_passed
        if time_diff > 0:
            thrm_relax = thermal_relaxation_error(
                T1, T2, time_diff).to_instruction()
            circ.append(thrm_relax, [qb[0]])
        circ.snapshot('snap_'+str(index), 'density_matrix')
        time_passed = snapshot_times[key]
        index += 1
    return circ


def fid_single_qubit(n_cycles, n_shots, gate_times={}, T1=40e3, T2=60e3):
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
        full_gate_times = WACQT_gate_times.get_gate_times(
            custom_gate_times=gate_times)
    elif isinstance(gate_times, GateTimes):
        full_gate_times = gate_times
    else:
        warnings.warn('Invalid gate times, assuming WACQT_gate_times')
        full_gate_times = WACQT_gate_times

    # Registers
    qb = QuantumRegister(5, 'code_qubit')
    an = AncillaRegister(2, 'ancilla_qubit')
    cr = get_classical_register(
        n_cycles, reset=False, recovery=False, flag=False)
    readout = ClassicalRegister(5, 'readout')
    registers = StabilizerRegisters(qb, an, cr, readout)

    # Circuits
    circ = get_testing_circuit(
        registers, reset=False, recovery=False, n_cycles=n_cycles)
    circ = get_standard_transpilation(circ)
    circ, time = add_idle_noise_to_circuit(
        circ, gate_times=gate_times, return_time=True)

    circ_single = get_idle_single_qubit(time)
    results = execute(circ_single, Aer.get_backend('qasm_simulator'),
                      noise_model=None, shots=n_shots).result()
    fid_single = []
    correct_state = results.data(
    )['snapshots']['density_matrix']['start'][0]['value']
    for i in range(len(time)-2):
        current_state = results.data()['snapshots']['density_matrix'][
            'snap_'+str(i+1)][0]['value']
        fid_single.append(state_fidelity(current_state, correct_state))
    return fid_single


# %% Test woop woop
# Settings used across all configurations
n_cycles = 10
n_shots = 2048

# Noise models
target_noise = thermal_relaxation_model_V2(gate_times=WACQT_target_times)
current_noise = thermal_relaxation_model_V2(
    gate_times=WACQT_demonstrated_times)

# Quantum error correction for both noise models
fid_standard_target_QEC = fidelity_from_scratch(n_cycles, target_noise, n_shots,
                                                gate_times=WACQT_target_times, reset=True, recovery=True, post_select=False,
                                                post_process=False, idle_noise=True, empty_circuit=False)
fid_pipeline_target_QEC = fidelity_from_scratch(n_cycles, target_noise, n_shots,
                                                gate_times=WACQT_target_times, reset=True, recovery=True, post_select=False,
                                                post_process=False, idle_noise=True, empty_circuit=False, pipeline=True)
fid_standard_dem_QEC = fidelity_from_scratch(n_cycles, current_noise, n_shots,
                                             gate_times=WACQT_demonstrated_times, reset=True, recovery=True, post_select=False,
                                             post_process=False, idle_noise=True, empty_circuit=False)
fid_pipeline_dem_QEC = fidelity_from_scratch(n_cycles, current_noise, n_shots,
                                             gate_times=WACQT_demonstrated_times, reset=True, recovery=True, post_select=False,
                                             post_process=False, idle_noise=True, empty_circuit=False, pipeline=True)

# %% Single qubit decay
fid_target_single = fid_single_qubit(
    n_cycles, n_shots, gate_times=WACQT_target_times)
fid_demonstrated_single = fid_single_qubit(
    n_cycles, n_shots, gate_times=WACQT_demonstrated_times)
# %%
# Settings used across all configurations
n_cycles = 15
n_shots = 4096

# Noise models
target_noise = thermal_relaxation_model_V2(gate_times=WACQT_target_times)
current_noise = thermal_relaxation_model_V2(
    gate_times=WACQT_demonstrated_times)

# Quantum error correction for both noise models
fid_target_QEC = fidelity_from_scratch(n_cycles, target_noise, n_shots,
                                       gate_times=WACQT_target_times, reset=True, recovery=True, post_select=False,
                                       post_process=False, idle_noise=True, empty_circuit=False)
print('Check!')
fid_demonstrated_QEC = fidelity_from_scratch(n_cycles, current_noise, n_shots,
                                             gate_times=WACQT_demonstrated_times, reset=True, recovery=True, post_select=False,
                                             post_process=False, idle_noise=True, empty_circuit=False)
print('Check!')

# Post selection for both noise models. Note that the second one has other
# settings for n_cycles and n_shots. Due to the noise model, there is little
# chance that any single shot remains at the 15th cycle, resulting in an error.
fid_target_PS, count_target_PS = fidelity_from_scratch(n_cycles, target_noise, n_shots,
                                                       gate_times=WACQT_target_times, reset=True, recovery=False, post_select=True,
                                                       post_process=False, idle_noise=True, empty_circuit=False)
print('Check!')
fid_demonstrated_PS, count_demonstrated_PS = fidelity_from_scratch(9, current_noise, 16000,
                                                                   gate_times=WACQT_demonstrated_times, reset=True, recovery=False, post_select=True,
                                                                   post_process=False, idle_noise=True, empty_circuit=False)
print('Check!')

# Post processing for both noise models.
fid_target_PP = fidelity_from_scratch(n_cycles, target_noise, n_shots,
                                      gate_times=WACQT_target_times, reset=True, recovery=False, post_select=False,
                                      post_process=True, idle_noise=True, empty_circuit=False)
print('Check!')
fid_demonstrated_PP = fidelity_from_scratch(n_cycles, current_noise, n_shots,
                                            gate_times=WACQT_demonstrated_times, reset=True, recovery=False, post_select=False,
                                            post_process=True, idle_noise=True, empty_circuit=False)
print('Check!')

# No correction, only performing stabilizer measurements for each noise model.
fid_target_stab = fidelity_from_scratch(n_cycles, target_noise, n_shots,
                                        gate_times=WACQT_target_times, reset=True, recovery=False, post_select=False,
                                        post_process=False, idle_noise=True, empty_circuit=False)
print('Check!')
fid_demonstrated_stab = fidelity_from_scratch(n_cycles, current_noise, n_shots,
                                              gate_times=WACQT_demonstrated_times, reset=True, recovery=False, post_select=False,
                                              post_process=False, idle_noise=True, empty_circuit=False)
print('Check!')

# Running 'empty' stabilizer circuits, only containing the encoding.
fid_target_empty = fidelity_from_scratch(n_cycles, target_noise, n_shots,
                                         gate_times=WACQT_target_times, reset=True, recovery=True, post_select=False,
                                         post_process=False, idle_noise=True, empty_circuit=True)
print('Check!')
fid_demonstrated_empty = fidelity_from_scratch(n_cycles, current_noise, n_shots,
                                               gate_times=WACQT_demonstrated_times, reset=True, recovery=True, post_select=False,
                                               post_process=False, idle_noise=True, empty_circuit=True)
print('Check!')

# %% Plotting
fig, ax1 = plt.subplots(1, figsize=(10, 6))
x_dis = np.arange(1, n_cycles+1)


# Subplot 1: Target gate times
ax1.plot(x_dis, fid_standard_target_QEC, '-o', label='Normal, target times')
ax1.plot(x_dis, fid_pipeline_target_QEC, '-o', label='Pipeline, target times')
ax1.plot(x_dis, fid_standard_dem_QEC, '-o', label='Normal, dem. times')
ax1.plot(x_dis, fid_pipeline_dem_QEC, '-o', label='Pipeline, dem. times')
ax1.set(ylim=(0.0, 1.0))

ax1.set_xlabel('Number of stabilizer cycles (2960 ns each)')
ax1.set_ylabel('Average fidelity')
ax1.set_title(
    'Average fidelity of simulated [[5,1,3]] QEC code using target gate times')
ax1.legend()
# %%
ax2.set_xlabel('Number of stabilizer cycles (14160 ns each)')
ax2.set_ylabel('Average fidelity')
ax2.set_title('Average fidelity, experimentally demonstrated gate times')
ax2.legend()
# fig.savefig('fidelities.pdf')

# %% Other stuff: Checking time difference between cycles
# Registers
qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
cr = get_classical_register(n_cycles, reset=False, recovery=False, flag=False)
readout = ClassicalRegister(5, 'readout')
registers = StabilizerRegisters(qb, an, cr, readout)

# Circuits
circ = get_testing_circuit(registers, reset=False,
                           recovery=False, n_cycles=n_cycles)
circ = get_standard_transpilation(circ)
circ, time = add_idle_noise_to_circuit(
    circ, gate_times=WACQT_demonstrated_times, return_time=True)

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


# %% Curve fitting with OLS (Not in use)
fid_list = [fid, fid_i, fid_res, fid_res_i]
theta_list = []
for fidelity in fid_list:
    x_D = np.ones((n_cycles, 2))
    for i in range(n_cycles):
        x_D[i][1] += i
    y = np.log(np.reshape(np.asarray(fidelity), (n_cycles, 1)))
    theta = np.dot(np.dot(np.linalg.inv(np.dot(x_D.T, x_D)), x_D.T), y)
    theta_list.append(theta)

x = np.linspace(0, n_cycles+1, 100)
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

# %% RUNS AND SETTINGS USED IN PLOTS FOR BASUDHA

# Single qubit decay
fid_target_single = fid_single_qubit(
    n_cycles, n_shots, gate_times=WACQT_target_times)
fid_demonstrated_single = fid_single_qubit(
    n_cycles, n_shots, gate_times=WACQT_demonstrated_times)

# Settings used across all configurations
n_cycles = 15
n_shots = 4096

# Noise models
target_noise = thermal_relaxation_model_V2(gate_times=WACQT_target_times)
current_noise = thermal_relaxation_model_V2(
    gate_times=WACQT_demonstrated_times)

# Quantum error correction for both noise models
fid_target_QEC = fidelity_from_scratch(n_cycles, target_noise, n_shots,
                                       gate_times=WACQT_target_times, reset=True, recovery=True, post_select=False,
                                       post_process=False, idle_noise=True, empty_circuit=False)
print('Check!')
fid_demonstrated_QEC = fidelity_from_scratch(n_cycles, current_noise, n_shots,
                                             gate_times=WACQT_demonstrated_times, reset=True, recovery=True, post_select=False,
                                             post_process=False, idle_noise=True, empty_circuit=False)
print('Check!')

# Post selection for both noise models. Note that the second one has other
# settings for n_cycles and n_shots. Due to the noise model, there is little
# chance that any single shot remains at the 15th cycle, resulting in an error.
fid_target_PS, count_target_PS = fidelity_from_scratch(n_cycles, target_noise, n_shots,
                                                       gate_times=WACQT_target_times, reset=True, recovery=False, post_select=True,
                                                       post_process=False, idle_noise=True, empty_circuit=False)
print('Check!')
fid_demonstrated_PS, count_demonstrated_PS = fidelity_from_scratch(9, current_noise, 16000,
                                                                   gate_times=WACQT_demonstrated_times, reset=True, recovery=False, post_select=True,
                                                                   post_process=False, idle_noise=True, empty_circuit=False)
print('Check!')

# Post processing for both noise models.
fid_target_PP = fidelity_from_scratch(n_cycles, target_noise, n_shots,
                                      gate_times=WACQT_target_times, reset=True, recovery=False, post_select=False,
                                      post_process=True, idle_noise=True, empty_circuit=False)
print('Check!')
fid_demonstrated_PP = fidelity_from_scratch(n_cycles, current_noise, n_shots,
                                            gate_times=WACQT_demonstrated_times, reset=True, recovery=False, post_select=False,
                                            post_process=True, idle_noise=True, empty_circuit=False)
print('Check!')

# No correction, only performing stabilizer measurements for each noise model.
fid_target_stab = fidelity_from_scratch(n_cycles, target_noise, n_shots,
                                        gate_times=WACQT_target_times, reset=True, recovery=False, post_select=False,
                                        post_process=False, idle_noise=True, empty_circuit=False)
print('Check!')
fid_demonstrated_stab = fidelity_from_scratch(n_cycles, current_noise, n_shots,
                                              gate_times=WACQT_demonstrated_times, reset=True, recovery=False, post_select=False,
                                              post_process=False, idle_noise=True, empty_circuit=False)
print('Check!')

# Running 'empty' stabilizer circuits, only containing the encoding.
fid_target_empty = fidelity_from_scratch(n_cycles, target_noise, n_shots,
                                         gate_times=WACQT_target_times, reset=True, recovery=True, post_select=False,
                                         post_process=False, idle_noise=True, empty_circuit=True)
print('Check!')
fid_demonstrated_empty = fidelity_from_scratch(n_cycles, current_noise, n_shots,
                                               gate_times=WACQT_demonstrated_times, reset=True, recovery=True, post_select=False,
                                               post_process=False, idle_noise=True, empty_circuit=True)
print('Check!')

# Plotting
fig, ax = plt.subplots(2, figsize=(10, 10))
x_dis = np.arange(1, n_cycles+1)
ax1, ax2 = ax

# Subplot 1: Target gate times
ax1.plot(x_dis, fid_target_QEC, '-o', label='Error correction')
ax1.plot(x_dis, fid_target_PS, '-o', label='Post select correct states')
#ax1.plot(x_dis, fid_target_PP, '-o', label='Post processing data')
ax1.plot(x_dis, fid_target_stab, '-o',
         label='Only measurements, no correction')
#ax1.plot(x_dis, fid_target_empty, '-o', label='Decay of [[5,1,3]] logical state')
ax1.plot(x_dis, fid_target_single, '-o', label='Decay of single qubit in |1>')

# Subplot 2: Demonstrated gate times
ax2.plot(x_dis, fid_demonstrated_QEC, '-o', label='Error correction')
ax2.plot(x_dis[0:7], fid_demonstrated_PS[0:7],
         '-o', label='Post select correct states')
#ax2.plot(x_dis, fid_demonstrated_PP, '-o', label='Post processing data')
ax2.plot(x_dis, fid_demonstrated_stab, '-o',
         label='Only measurements, no correction')
#ax2.plot(x_dis, fid_demonstrated_empty, '-o', label='Decay of [[5,1,3]] logical state')
ax2.plot(x_dis, fid_demonstrated_empty, '-o',
         label='Decay of single qubit in |1>')

ax1.set(ylim=(0.0, 1.0))
ax2.set(ylim=(0.0, 1.0))

ax1.set_xlabel('Number of stabilizer cycles (2960 ns each)')
ax1.set_ylabel('Average fidelity')
ax1.set_title(
    'Average fidelity of simulated [[5,1,3]] QEC code using target gate times')
ax1.legend()

ax2.set_xlabel('Number of stabilizer cycles (14160 ns each)')
ax2.set_ylabel('Average fidelity')
ax2.set_title('Average fidelity, experimentally demonstrated gate times')
ax2.legend()
# fig.savefig('fidelities.pdf')
