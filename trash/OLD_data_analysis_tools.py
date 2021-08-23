# This file is meant as the final version of most functions from
# 'plotting_expval.py', in order to separate different parts and make the code
# more readable.

# %% Import modules
#import seaborn as sns
#import matplotlib.pyplot as plt
import numpy as np
import scipy
import itertools
from qiskit import *
#from qiskit.visualization import plot_histogram

# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import thermal_relaxation_error
from qiskit.providers.aer.library import save_density_matrix, save_expectation_value
from qiskit.providers.aer import AerSimulator, QasmSimulator

from qiskit.quantum_info import partial_trace
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import state_fidelity

# Our own files
if __package__:
    from .custom_noise_models import (thermal_relaxation_model,
                                      thermal_relaxation_model_V2,
                                      thermal_relaxation_model_per_qb,
                                      WACQT_target_times,
                                      WACQT_demonstrated_times,
                                      standard_times)
    from .custom_transpiler import *
    from .stabilizers import *
    from .post_select import *
    from .post_process import *
    from .idle_noise import *
else:
    from custom_noise_models import (thermal_relaxation_model,
                                     thermal_relaxation_model_V2,
                                     WACQT_target_times,
                                     WACQT_demonstrated_times,
                                     standard_times)
    from custom_transpiler import *
    from stabilizers import *
    from post_select import *
    from post_process import *
    from idle_noise import *
# %%


def extend_standard_gate_times(gate_times={}):
    # TODO: move this to custom_noise_model.py?
    if isinstance(gate_times, dict):
        full_gate_times = standard_times.get_gate_times(
            custom_gate_times=gate_times)
    elif isinstance(gate_times, GateTimes):
        full_gate_times = gate_times
    else:
        warnings.warn('Invalid gate times, assuming standard_times')
        full_gate_times = standard_times
    return full_gate_times


def default_execute(circ, shots=None, noise_model=None, gate_times={}, T1=40e3, T2=60e3, simulator_name='qasm_simulator', simulator_method='density_matrix'):
    """Execute with our standard settings. Use this to make sure that the noise model i applied to delays.

    Args:
        circ (Circuit): circuit
        noise_model (NoiseMode, optional): defaults to thermal_relaxation_model_V2.
        gate_times (dict, optional)
        T1 ([type], optional): Defaults to 40e3.
        T2 ([type], optional):  Defaults to 60e3.
        simulator_type (str, optional): Simulation method. Defaults to 'density_matrix'.

    Returns:
        Results: Qiskit results object.
    """
    if noise_model is None:
        full_gate_times = extend_standard_gate_times(gate_times)
        if noise_model is None:
            noise_model = thermal_relaxation_model_V2(
                T1=T1, T2=T2, gate_times=full_gate_times)

    simulator = Aer.get_backend(simulator_name)

    simulator.set_option('noise_model', noise_model)
    try:
        simulator.set_option('method', simulator_method)
    except:
        print('Invalid simulator type, defaulting to density_matrix')
        simulator.set_option('method', 'density_matrix')

    return simulator.run(circ, shots=shots).result()


def fidelity_from_scratch(n_cycles, n_shots, gate_times={}, T1=40e3, T2=60e3,
                          reset=True, data_process_type='recovery', idle_noise=True, transpile=True,
                          snapshot_type='dm', device=None, device_properties=WACQT_device_properties,
                          encoding=True, theta=0, phi=0, pauliop='ZZZZZ', simulator_type='density_matrix',
                          project=False, **kwargs):
    """TODO: Update this description

    Get the fidelity of a certain setup/configuration from only its
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

    full_gate_times = extend_standard_gate_times(gate_times)

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
    elif data_process_type == 'post_process':
        recovery = False
        conditional = True
    else:
        recovery = False
        conditional = False

    # TODO: Delete these
    # Noise model
    # noise_model = thermal_relaxation_model_V2(
    #     T1=T1, T2=T2, gate_times=full_gate_times)
    # The code below is used to experiment with the nois model
    # noise_model = thermal_relaxation_model_per_qb(
    #     T1=[T1]*7, T2=[T2]*7, gate_times=full_gate_times)
    # This remove
    # noise_model = thermal_relaxation_model_per_qb(
    #     T1=[T1]*5+[10000000000,10000000000], T2=[T2]*5+[10000000000,10000000000], gate_times=full_gate_times)
    # noise_model = thermal_relaxation_model_per_qb(
    #     T1=[T1]*5+[1,1], T2=[T2]*5+[1,1], gate_times=full_gate_times)
    # noise_model = thermal_relaxation_model_per_qb(
    #     T1=[1,1]*7, T2=[1,1]*7, gate_times=full_gate_times)

    # Registers
    qb = QuantumRegister(5, 'code_qubit')
    an = AncillaRegister(2, 'ancilla_qubit')
    cr = get_classical_register(
        n_cycles, reset=reset, recovery=recovery, flag=False)
    readout = ClassicalRegister(5, 'readout')
    registers = StabilizerRegisters(qb, an, cr, readout)

    # Circuits
    circ = get_full_stabilizer_circuit(registers, n_cycles=n_cycles, reset=reset,
                                       recovery=recovery, flag=False,
                                       snapshot_type=snapshot_type,
                                       conditional=conditional,
                                       encoding=encoding, theta=theta, phi=phi,
                                       pauliop=pauliop, device=device,
                                       simulator_type=simulator_type, final_measure=False, **kwargs)

    if transpile:
        circ = shortest_transpile_from_distribution(circ, print_cost=False,
                                                    **device_properties)

    # Get the correct (no errors) state
    trivial = get_encoded_state(theta, phi, include_ancillas=None)

    # Create empty encoded circuit
    if data_process_type == 'empty_circuit':

        # Prepare the circuit
        time = get_circuit_time(circ, full_gate_times)
        circ = get_empty_noisy_circuit_v3(circ, time, full_gate_times,
                                          T1=T1, T2=T2)
        results = default_execute(circ, n_shots)

        # Calculate fidelity at each snapshot
        fidelities = []
        if snapshot_type == 'dm' or snapshot_type == 'density_matrix':
            for current_cycle in range(n_cycles+1):
                state = results.data()['dm_' + str(current_cycle)]
                fidelities.append(state_fidelity(state, trivial))
        elif snapshot_type == 'exp' or snapshot_type == 'expectation_value':
            for current_cycle in range(n_cycles+1):
                fidelities.append(results.data()['exp_' + str(current_cycle)])
        return fidelities, time

    # Add idle noise (empty_circuit does this automatically)
    if idle_noise:
        circ, time = add_idle_noise_to_circuit(circ, gate_times=full_gate_times,
                                               T1=T1, T2=T2, return_time=True,
                                               **kwargs)
    else:
        time = get_circuit_time(circ=circ, gate_times=full_gate_times)

    results = default_execute(circ, n_shots)
    
    if data_process_type == 'recovery' or data_process_type == 'none':
        fidelities = []  # If project = True, this contains F_L
        P_Ls = []
        if snapshot_type == 'dm' or snapshot_type == 'density_matrix':
            for current_cycle in range(label_counter.value):
                state = results.data()['dm_' + str(current_cycle)]
                if project:
                    state, P_L = project_dm_to_logical_subspace_V2(
                        state, return_P_L=True)
                    P_Ls.append(np.real(P_L))
                fidelities.append(state_fidelity(state, trivial))
            if project:
                return fidelities, P_Ls, time
        elif snapshot_type == 'exp' or snapshot_type == 'expectation_value':
            for current_cycle in range(n_cycles+1):
                fidelities.append(results.data()['exp_' + str(current_cycle)])

        return fidelities, time

    elif data_process_type == 'post_select':
        # Get the number of remaining shot at each cycle
        select_counts = get_trivial_post_select_counts(
            results.get_counts(), n_cycles)

        # Get the fidelity for each cycle
        fidelities = []  # If project = True, this contains F_L
        P_Ls = []
        if snapshot_type == 'dm' or snapshot_type == 'density_matrix':
            # TODO: Make this return F_L and P_L seperately and fix the references
            for state in get_trivial_post_select_den_mat(results, n_cycles):
                if project:
                    state, P_L = project_dm_to_logical_subspace_V2(
                        state, return_P_L=True)
                    P_Ls.append(np.real(P_L))
                fidelities.append(state_fidelity(state, trivial))
            if project:
                return fidelities, P_Ls, select_counts, time
        elif snapshot_type == 'exp' or snapshot_type == 'expectation_value':
            fidelities = [post_selected_state for
                          post_selected_state in get_trivial_exp_value(
                              results, n_cycles)]

        return fidelities, select_counts, time

    elif data_process_type == 'post_process':
        def get_av_fidelities(states_and_counts, correct_state, n_shots):
            av_fidelities = []
            for cycle in states_and_counts:
                fid = 0
                for state, counts in cycle:
                    fid += state_fidelity(state, correct_state)*counts
                av_fidelities.append(fid/n_shots)
            return av_fidelities
        fidelities = get_av_fidelities(get_states_and_counts(
            results, n_cycles, post_process=True, reset=reset), trivial, n_shots)
        # fidelities = [1.0]
        # for current_cycle in range(n_cycles):
        #     #print("\nCycle ", current_cycle)
        #     counts = get_subsystem_counts_up_to_cycle(
        #         results.get_counts(), current_cycle+1)
        #     fid = 0
        #     count_sum = 0
        #     # TODO: Retrieve the name from function instead
        #     data = results.data()['dm_con_' + str(current_cycle+1)]
        #     for memory in data.keys():
        #         den_mat = data[memory]
        #         #print("\n",memory)
        #         fid += state_fidelity(trivial, post_process_den_mat(
        #             den_mat, memory, current_cycle))*counts[int(memory, 16)]
        #         count_sum += counts[int(memory, 16)]

        #         #print(counts[int(memory, 16)])
        #         #print(state_fidelity(trivial, post_process_den_mat(
        #         #    den_mat, memory, current_cycle)))
        #     fidelities.append(fid/n_shots)
        return fidelities, time
    else:
        print('Warning: No matching data_process_type')

    return []


def get_idle_single_qubit(snapshot_times, snapshot_type='dm', T1=40e3, T2=60e3,
                          theta=0, phi=0, pauliop='Z'):
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
    circ.rx(theta, qb)
    circ.rz(phi, qb)
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
            circ.save_expectation_value(
                Pauli(pauliop), qb, label='snap_'+str(index))
        time_passed = snapshot_times[key]
        index += 1
    return circ


def fid_single_qubit(n_cycles, n_shots, gate_times={}, snapshot_type='dm',
                     T1=40e3, T2=60e3, theta=0, phi=0, pauliop='Z', **kwargs):
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
        full_gate_times = standard_times.get_gate_times(
            custom_gate_times=gate_times)
    elif isinstance(gate_times, GateTimes):
        full_gate_times = gate_times
    else:
        warnings.warn('Invalid gate times, assuming standard_times')
        full_gate_times = standard_times

    # Registers
    qb = QuantumRegister(5, 'code_qubit')
    an = AncillaRegister(2, 'ancilla_qubit')
    cr = get_classical_register(
        n_cycles, reset=False, recovery=False, flag=False)
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

    circ_single = get_idle_single_qubit(time, snapshot_type, T1, T2,
                                        theta=theta, phi=phi, pauliop=pauliop)
    results = execute(circ_single, Aer.get_backend('qasm_simulator'),
                      noise_model=None, shots=n_shots).result()
    fidelities = [1.0]  # The initial state

    if snapshot_type == 'dm' or snapshot_type == 'density_matrix':
        trivial = results.data()['start']
        for i in range(len(time)-2):
            current_state = results.data()['snap_'+str(i+1)]
            fidelities.append(state_fidelity(current_state, trivial))
    elif snapshot_type == 'exp' or snapshot_type == 'expectation_value':
        for i in range(len(time)-2):
            fidelities.append(results.data()['snap_'+str(i+1)])
    return fidelities, time


def encoding_fidelity(n_shots, gate_times={}, T1=40e3, T2=60e3,
                      idle_noise=True, theta=0., phi=0., iswap=True,
                      snapshot_type='dm', device=None, pauliop='ZZZZZ', project=False):

    # Get gate times missing from input
    if isinstance(gate_times, dict):
        full_gate_times = standard_times.get_gate_times(
            custom_gate_times=gate_times)
    elif isinstance(gate_times, GateTimes):
        full_gate_times = gate_times
    else:
        warnings.warn('Invalid gate times, assuming standard_times')
        full_gate_times = standard_times

    # Registers
    qb = QuantumRegister(5, 'code_qubit')
    an = AncillaRegister(2, 'ancilla_qubit')
    cr = get_classical_register(n_cycles=0, flag=False)
    readout = ClassicalRegister(5, 'readout')
    registers = StabilizerRegisters(qb, an, cr, readout)

    # Circuits
    circ = get_empty_stabilizer_circuit(registers)

    # Initial state
    # TODO: Better looking solution
    extra_qubits = np.zeros(2**6)
    extra_qubits[0] = 1.0
    zero_state = np.kron(extra_qubits, np.array([1, 0]))
    one_state = np.kron(extra_qubits, np.array([0, 1]))
    psi = np.cos(theta/2)*zero_state + np.exp(1j*phi)*np.sin(theta/2)*one_state
    circ.set_density_matrix(psi)
    psi = np.cos(theta/2)*logical_states(include_ancillas=None)[0] + np.exp(
        1j*phi)*np.sin(theta/2)*logical_states(include_ancillas=None)[1]

    # Encoding
    if device == 'WACQT':
        circ.compose(transpiled_encoding_WACQT(
            registers, iswap=iswap), inplace=True)
        qubits = [qb[3], qb[1], qb[2], an[1], qb[4]]  # F Qubit permutation
    elif device == 'DD':
        circ.compose(transpiled_encoding_DD(
            registers, iswap=iswap), inplace=True)
        qubits = [qb[2], an[1], qb[1], qb[3], qb[4]]  # Qubit permutation
    else:
        circ.compose(encode_input_v2(registers), inplace=True)
        qubits = qb  # Qubit permutation
    add_snapshot_to_circuit(circ, snapshot_type=snapshot_type, current_cycle=0, qubits=qubits,
                            pauliop=pauliop, include_barriers=True)

    # Trivial state
    if snapshot_type == 'dm' or snapshot_type == 'density_matrix':
        trivial_res = execute(circ, Aer.get_backend(
            'qasm_simulator'), shots=1).result()
        trivial = trivial_res.data()['dm_0']

    if idle_noise:
        circ, time = add_idle_noise_to_circuit(circ, gate_times=full_gate_times,
                                               T1=T1, T2=T2, return_time=True)
    else:
        time = {'end': None}
    # Run the circuit
    noise_model = thermal_relaxation_model_V2(
        T1=T1, T2=T2, gate_times=full_gate_times)
    # noise_model = None
    results = execute(circ, Aer.get_backend('qasm_simulator'),
                      noise_model=noise_model, shots=n_shots).result()
    if snapshot_type == 'dm' or snapshot_type == 'density_matrix':
        state = results.data()['dm_0']
        true_state = psi
        if project:
            state, P_L = project_dm_to_logical_subspace_V2(
                state, return_P_L=True)
        fidelities = state_fidelity(state, true_state)
        if project:
            return fidelities, circ, time['end'], P_L
    elif snapshot_type == 'exp' or snapshot_type == 'expectation_value':
        fidelities = results.data()['exp_0']
    # logical = logical_states(include_ancillas=None)
    # I_L = np.outer(logical[0], logical[0])+np.outer(logical[1], logical[1])
    # P_L = np.trace(state@I_L)
    return fidelities, circ, time['end']

# %%


def monoExp(t, T, c, A):
    return (A-c) * np.exp(-t/T) + c


def _get_array_indexes(index, sweep_lengths):
    """Returns a tuple of indexes for the error_array in sweep_parameter_space,
    given a single index"""
    indexes = np.zeros(len(sweep_lengths), dtype=int)
    indexes[-1] = index
    for i in reversed(range(len(sweep_lengths)-1)):
        if indexes[i+1] >= sweep_lengths[i+1]:
            indexes[i] = indexes[i+1] // sweep_lengths[i+1]
            indexes[i+1] -= indexes[i] * sweep_lengths[i+1]
    return tuple(indexes)


def get_error_rate(fidelity, time=None):
    """Calculates the logical error rate from a list of fidelities"""

    n_cycles = len(fidelity)-1
    x_D = np.ones((n_cycles, 2))
    for i in range(n_cycles):
        if time is not None:
            try:
                x_D[i][1] = time['exp_'+str(i+1)]*1e-3
            except:
                x_D[i][1] = time['dm_'+str(i+1)]*1e-3
        else:
            x_D[i][1] += i
    y = np.log(np.reshape(np.asarray(fidelity[1:]), (n_cycles, 1)))
    theta = np.dot(np.dot(np.linalg.inv(np.dot(x_D.T, x_D)), x_D.T), y)

    MSE = 0.
    for cycle in range(n_cycles):
        y_pred = np.exp(theta[0]) * np.exp((cycle+1)*theta[1])
        MSE += (y_pred-fidelity[cycle+1])**2

    # TODO: Only return theta[1] maybe?
    return theta, MSE


def sweep_parameter_space(T1, T2, single_qubit_gate_time, two_qubit_gate_time,
                          measure_time, feedback_time, n_cycles=8, n_shots=2048, single_qubit=False, save=None,
                          time_axis=False, perfect_stab=False, **kwargs):
    """
    DEPRECATED
    Use scripts/functions in T1T2_sweep.py or gate_times_test.py

    Calculate the logical error rate across a variety of parameters
    TODO: Add default values for n_cycles and n_shots that are reasonable
    """

    # Check for theta and phi in kwargs
    try:
        theta = kwargs['theta']
        phi = kwargs['phi']
    except:
        theta = 0.
        phi = 0.

    # Make every noise parameter into list (if not already)
    noise_parameters = [T1, T2, single_qubit_gate_time, two_qubit_gate_time,
                        measure_time, feedback_time]
    noise_parameters = [[param] if not isinstance(
        param, list) else param for param in noise_parameters]

    # Generate an array to store the data in
    sweep_lengths = [len(param) for param in noise_parameters]
    error_array = np.zeros(sweep_lengths)
    var_array = np.zeros(sweep_lengths)

    # Get all combinations of parameters
    index = 0
    for params in itertools.product(*noise_parameters):

        gate_times = GateTimes(params[2], params[3],
                               {'u1': 0, 'z': 0, 'measure': params[4], 'feedback': params[5]})

        # Skip cases where T2 > 2*T1
        if params[1] > 2*params[0]:
            index += 1
            continue

        if single_qubit:
            fid, time = fid_single_qubit(n_cycles, n_shots, T1=params[0],
                                         T2=params[1], gate_times=gate_times, **kwargs)

            # Normalize data if needed
            # TODO: Better solution? Now it checks if input state is |+>
            if theta == np.pi/2 and phi == np.pi/2:
                for i in range(len(fid)):
                    fid[i] = 2.*fid[i] - 1.
        elif perfect_stab:
            fid, time = perfect_stab_circuit(n_cycles, n_shots, gate_times=gate_times,
                                             T1=params[0], T2=params[1], reset=True, snapshot_type='exp')
        else:
            fid, time = fidelity_from_scratch(n_cycles, n_shots, T1=params[0],
                                              T2=params[1], gate_times=gate_times, **kwargs)

        # From fidelities, estimate lifetime
        # Old version
        # if time_axis:
        #    error_rate, MSE = get_error_rate(fid, time)
        # else:
        #    error_rate, MSE = get_error_rate(fid)
        time_list = list(time.values())[1:-1]

        p0 = (params[0], 0, 0.9)  # start with values near those we expect
        pars, cov = scipy.optimize.curve_fit(monoExp, time_list, fid[1:], p0)
        T, c, A = pars

        array_indexes = _get_array_indexes(index, sweep_lengths)
        error_array[array_indexes] = T
        #error_array[array_indexes] = error_rate[1]
        var_array[array_indexes] = cov[0][0]
        index += 1

    # Save results to file
    # TODO: Save as txt instead? Make it both readable and have the parameters used
    if save is not None:
        np.save(save, error_array)
        np.save(save+'_var', var_array)

    return error_array, var_array


def perfect_stab_circuit(n_cycles, n_shots, gate_times={}, T1=40e3, T2=60e3,
                         reset=True, data_process_type='recovery', snapshot_type='dm',
                         theta=0, phi=0, pauliop='ZZZZZ', include_barriers=True,
                         project=False):

    # Get gate times missing from input
    if isinstance(gate_times, dict):
        full_gate_times = standard_times.get_gate_times(
            custom_gate_times=gate_times)
    elif isinstance(gate_times, GateTimes):
        full_gate_times = gate_times
    else:
        warnings.warn('Invalid gate times, assuming standard_times')
        full_gate_times = standard_times

    # Extract times from GateTimes class
    two_qubit_gate_time = full_gate_times['cz']
    single_qubit_gate_time = full_gate_times['x']
    measure_time = full_gate_times['measure']
    feedback_time = full_gate_times['feedback']
    if data_process_type == 'post_select':
        feedback_time = 0

    # Calculate cycle time
    cycle_time = 8*single_qubit_gate_time + 16 * \
        two_qubit_gate_time + 4*measure_time + feedback_time
    time = {}
    for i in range(n_cycles+1):
        if data_process_type == 'recovery':
            key = snapshot_type + '_' + str(i)
        elif data_process_type == 'post_select':
            key = snapshot_type + '_con_' + str(i)
        time[key] = i*cycle_time

    # Check the data processing method for settings
    if data_process_type == 'recovery':
        recovery = True
        conditional = False
    elif data_process_type == 'post_select':
        recovery = False
        conditional = True

    # Registers
    qb = QuantumRegister(5, 'code_qubit')
    an = AncillaRegister(2, 'ancilla_qubit')
    cr = get_classical_register(
        n_cycles, reset=reset, recovery=recovery, flag=False)
    readout = ClassicalRegister(5, 'readout')
    registers = StabilizerRegisters(qb, an, cr, readout)

    # Build a custom circuit with idle noise before each cycle
    circ = get_empty_stabilizer_circuit(registers)
    circ.set_density_matrix(get_encoded_state(theta=theta, phi=phi))
    add_snapshot_to_circuit(circ, snapshot_type=snapshot_type, current_cycle=0, qubits=qb,
                            conditional=conditional, pauliop=pauliop,
                            include_barriers=include_barriers)
    thrm_relax = thermal_relaxation_error(T1, T2, cycle_time).to_instruction()
    for reg in circ.qubits:
        circ.append(thrm_relax, [reg])

    for current_cycle in range(n_cycles):
        circ.compose(unflagged_stabilizer_cycle(registers, reset=reset, recovery=recovery,
                                                current_cycle=current_cycle, current_step=0,
                                                include_barriers=include_barriers), inplace=True)
        add_snapshot_to_circuit(circ, snapshot_type=snapshot_type,
                                qubits=qb, conditional=conditional, pauliop=pauliop,
                                include_barriers=include_barriers)
        for reg in circ.qubits:
            circ.append(thrm_relax, [reg])
    circ.measure(qb, readout)

    # Run the circuit
    results = execute(circ, Aer.get_backend('qasm_simulator'),
                      noise_model=None, shots=n_shots).result()

    trivial = get_encoded_state(theta, phi, include_ancillas=None)
    fidelities = []
    P_Ls = []
    if data_process_type == 'recovery':
        if snapshot_type == 'dm' or snapshot_type == 'density_matrix':
            for current_cycle in range(n_cycles+1):
                state = results.data(circ)['dm_' + str(current_cycle)]
                if project:
                    state, P_L = project_dm_to_logical_subspace_V2(
                        state, return_P_L=True)
                    P_Ls.append(P_L)
                fidelities.append(state_fidelity(state, trivial))
            if project:
                return fidelities, P_Ls, time
        elif snapshot_type == 'exp' or snapshot_type == 'expectation_value':
            for current_cycle in range(n_cycles+1):
                fidelities.append(results.data()['exp_' + str(current_cycle)])
        return fidelities, time

    elif data_process_type == 'post_select':
        # Get the fidelity for each cycle
        if snapshot_type == 'dm' or snapshot_type == 'density_matrix':
            fidelities = [state_fidelity(post_selected_state, trivial) for
                          post_selected_state in get_trivial_post_select_den_mat(
                results, n_cycles)]
        elif snapshot_type == 'exp' or snapshot_type == 'expectation_value':
            fidelities = [post_selected_state for
                          post_selected_state in get_trivial_exp_value(
                              results, n_cycles)]

        # Get the number of remaining shot at each cycle
        select_counts = get_trivial_post_select_counts(
            results.get_counts(), n_cycles)
        return fidelities, select_counts, time
    return -1


def scale_gate_times(gate_times={}, scalings=[], return_class=False):
    """Scale up the gate times proportionally to their fraction of time in a
    full stabilizer cycle

    Args:
        gate_times - Dict or GateTimes class
        scalings [list] - The scaling factors to apply to cycle times
        return_class [bool] - Whether or not to return the scaled times as a
                              dict (False) or list of GateTimes classes

    Returns:
        scaled_times - List of GateTimes classes matching the scalings
        and gate_times input, or a dict with lists of all gate times, depending
        on the return_class input."""

    if isinstance(gate_times, dict):
        full_gate_times = standard_times.get_gate_times(
            custom_gate_times=gate_times)
    elif isinstance(gate_times, GateTimes):
        full_gate_times = gate_times
    else:
        warnings.warn('Invalid gate times, assuming standard_times')
        full_gate_times = standard_times

    # Extract the gate times from class
    # TODO: Fancier solution?
    extracted_gate_times = full_gate_times.get_gate_times()
    gate_times = {'single_qubit_gate': extracted_gate_times['x'],
                  'two_qubit_gate': extracted_gate_times['cz'],
                  'measure': extracted_gate_times['measure'],
                  'feedback': extracted_gate_times['feedback']}

    if return_class:
        scaled_times = [GateTimes(single_qubit_default=gate_times['single_qubit_gate']*scale,
                                  two_qubit_default=gate_times['two_qubit_gate']*scale,
                                  custom_gate_times={'u1': 0, 'z': 0,
                                                     'measure': gate_times['measure']*scale,
                                                     'feedback': gate_times['feedback']*scale})
                        for scale in scalings]
    else:
        scaled_times = {}
        for key in gate_times:
            scaled_times[key] = [gate_times[key]*scale for scale in scalings]
    return scaled_times


def project_dm_to_logical_subspace_V1(rho):
    """Projects a density-matrix to the logical codespace. This version 
    reduces the dimension to a 2x2 matrix.

    Args:
        rho (DensityMatrix): The 5-qubit state

    Returns:
        DensityMatrix: Projected single-qubit logical state
    """
    P_L = 0
    logical = logical_states(include_ancillas=None)
    for i in range(2):
        P_L += logical[i] @ rho @ logical[i]
    rho_L = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            rho_L[i, j] = (logical[i] @ rho @ logical[j])/P_L
    return rho_L


def project_dm_to_logical_subspace_V2(rho, return_P_L=False):
    """Projects a density-matrix to the logical codespace. This version 
    does not reduce the dimension, and returns a 2^5 x 2^5 matrix.

    F(|0_L>, rho_L) where rho_L is returned from this function is equal to
    F(|0>, rho_L) wherre rho_L is the 2x2 matrix returned from V1 above.

    Args:
        rho (DensityMatrix): The 5-qubit state

    Returns:
        DensityMatrix: Projected 5-qubit logical state
    """
    logical = logical_states(include_ancillas=None)
    # Projector to the code space
    I_L = np.outer(logical[0], logical[0])+np.outer(logical[1], logical[1])
    # Note here how the projector has to be included for this to work as expected
    logical_pauli_matrices = np.array((
        I_L,
        Pauli('XXXXX').to_matrix()@I_L,
        Pauli('YYYYY').to_matrix()@I_L,
        Pauli('ZZZZZ').to_matrix()@I_L
    ))

    P_L = np.trace(rho@logical_pauli_matrices[0])

    rho_L = np.zeros((2**5, 2**5), dtype=complex)
    for i in range(4):
        rho_L += logical_pauli_matrices[i] * \
            np.trace(rho@logical_pauli_matrices[i])/(2*P_L)
    if return_P_L:
        return rho_L, P_L
    return rho_L

def overlap_with_subspace(rho, basis):
    """Finds the overlap of rho with a subspace defined by the given basis vectors.
    This is equivalent to the proability of rho residing in the subspace.

    If basis is a single state, this is equivalent to the state_fidelity of it to rho.
    """
    basis = basis if isinstance(basis,list) else [basis]
    P = 0
    for basis_vector in basis:
        P += state_fidelity(rho, basis_vector)
    return P
# %%
