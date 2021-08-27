"""This file contains a variety of tools for running a variety of simulations and
analyzing the results."""
# %% Import modules
import warnings
import pickle
import numpy as np
from qiskit import Aer, execute, QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister

# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import thermal_relaxation_error
from qiskit.quantum_info import state_fidelity, Pauli

# Our own files
if __package__:
    from .custom_noise_models import (thermal_relaxation_model_V2,
                                      standard_times,
                                      extend_standard_gate_times, 
                                      GateTimes)
    from .stabilizers import *
    from .post_select import get_trivial_post_select_counts, get_trivial_post_select_den_mat, get_trivial_exp_value
    from .post_process import get_states_and_counts
    from .idle_noise import add_idle_noise_to_circuit, get_circuit_time
else:
    from custom_noise_models import (thermal_relaxation_model_V2,
                                     standard_times,
                                     extend_standard_gate_times, 
                                     GateTimes)
    from stabilizers import *
    from post_select import get_trivial_post_select_counts, get_trivial_post_select_den_mat, get_trivial_exp_value
    from post_process import get_states_and_counts
    from idle_noise import add_idle_noise_to_circuit, get_circuit_time
# %%


def default_execute(circ, shots=None, noise_model=None, gate_times={}, T1=40e3, T2=60e3,
                    simulator_name='qasm_simulator', simulator_method='density_matrix'):
    """Run simulation with our standard settings.

    Args:
        circ (Circuit): QuantumCircuit object to run.
        shots (int): Number of runs to perform.
        noise_model (NoiseModel, optional): defaults to thermal_relaxation_model_V2.
        gate_times (dict, optional)
        T1 ([type], optional): Defaults to 40e3.
        T2 ([type], optional):  Defaults to 60e3.
        simulator_type (str, optional): Simulation method. Defaults to 'density_matrix'.

    Returns:
        Results: Qiskit results object.
    """
    if noise_model is None:
        full_gate_times = extend_standard_gate_times(gate_times)
        noise_model = thermal_relaxation_model_V2(
            T1=T1, T2=T2, gate_times=full_gate_times)

    simulator = Aer.get_backend(simulator_name)

    simulator.set_option('noise_model', noise_model)
    try:
        simulator.set_option('method', simulator_method)
    except:
        print('Invalid simulator type, defaulting to density_matrix')
        simulator.set_option('method', 'density_matrix')

    # Run simulation
    results = simulator.run(circ, shots=shots).result()
    if results.success:
        return results
    else:
        print('WARNING: simulator.run() failed, attempting execute instead. ' \
              '\nExecute cannot handle delay markers (no noise applied) and will '\
              'automatically transpile iSWAP into four CX gates. For more info, see: '\
              '\nhttps://qiskit.org/documentation/tutorials/simulators/4_custom_gate_noise.html'
             )
        results = execute(circ, simulator, noise_model=noise_model,shots=shots).result()
        return results


def default_simulate(kwargs, T1, T2, gate_times, n_shots=1024, noise_model=None):
    """
    Creates and simulates a circuit with idle noise applied.
    """
    print('Simulating new data')
    # Create circuit
    circ = get_full_stabilizer_circuit(**kwargs)
    circ, times = add_idle_noise_to_circuit(
        circ, gate_times, T1=T1, T2=T2, return_time=True)

    # Run it
    if noise_model is None:
        noise_model = thermal_relaxation_model_V2(
            T1=T1, T2=T2, gate_times=gate_times)
    results = default_execute(
        circ, n_shots, gate_times=gate_times, noise_model=noise_model)
    return results, times


def default_simulate_persist_to_file(kwargs, T1, T2, gate_times, n_shots=1024, noise_model=None, 
                                     file_name='decoding_data/cache.dat', overwrite=False, save_to_file=True):
    """Functionally the same as default_simulate, but loads data from file if it exists.
    Otherwise, it saves it to file.

    Args:
        kwargs (dict): Dictionary of simulation options to feed to default simulate
        T1 (int): noise param
        T2 (int): noise param
        gate_times (dict): dictionary of gate (and other operations) duration
        n_shots (int, optional): simulation repetitions. Defaults to 1024.
        noise_model (NoiseModel, optional):  Defaults to None.
        file_name (str, optional): file to load from/save to. Defaults to 'decoding_data/cache.dat'.
        overwrite (bool, optional): Whether to overwrite the current data for the specified settings with a new simulation. Defaults to False.
        save_to_file (bool, optional): Whether to skip saving the data, and only loading if possible. Defaults to True.

    Returns:
        (Results, dict): Results object and times dictionary
    """
    try:
        cache = pickle.load(open(file_name, "rb"))
    except (IOError, ValueError):
        cache = {}

    key = (frozenset(kwargs.items()), str(T1), str(
        T2), frozenset(gate_times.gate_times.items()))
    if key not in cache or overwrite:
        cache[key] = default_simulate(
            kwargs, T1, T2, gate_times, n_shots=n_shots, noise_model=noise_model)
        if save_to_file:
            pickle.dump(cache, open(file_name, "wb"))
    else:
        print('Loading simulation results from file')
    return cache[key]


def check_if_saved(kwargs, T1, T2, gate_times, file_name='decoding_data/cache.dat'):
    key = (frozenset(kwargs.items()), str(T1), str(
        T2), frozenset(gate_times.gate_times.items()))
    return key in pickle.load(open(file_name, "rb"))


def print_saved_runs(file_name='decoding_data/cache.dat'):
    for key in pickle.load(open(file_name, "rb")):
        print(key[0])
        print(key[1])
        print(key[3], '\n')


def fidelity_from_scratch(n_cycles, n_shots, gate_times={}, T1=40e3, T2=60e3,
                          reset=True, data_process_type='recovery', idle_noise=True, transpile=True,
                          snapshot_type='dm', device=None, device_properties=None,
                          encoding=True, theta=0, phi=0, pauliop='ZZZZZ', simulator_type='density_matrix',
                          project=False, generator_snapshot=False, idle_snapshots=0, **kwargs):
    """Get the fidelity of a certain setup/configuration from only its
    parameters.

    Args:
        n_cycles (int): The number of stabilizer cycles to be performed.
        n_shots (int): The number of runs of the stabilizer circuit.
        gate_times (dict): The gate times for a circuit, used for thermal relaxation
                           noise. If left empty or partially filled, remaining
                           gate times default to standard_times. Can also be given
                           as a GateTimes object.
        T1 (float): T1 thermal relaxation parameter, given in ns. Defaults to 40e3.
        T2 (float): T2 thermal relaxation parameter, given in ns. Defaults to 60e3.
        reset (bool): Whether or not to reset ancilla between measurements.
                      defaults to True.
        data_process_type (str): How to process the results from simulation. Valid
                                 options are 'recovery' (Standard QEC), 'post_select',
                                 and 'post_process'.
        idle_noise (bool): Whether or not to add noise to idle qubits. This
                           assumes thermal relaxation. Defaults to True.
        transpile (bool): Whether to transpile the circuit to fit a specific
                          connectivity and gate set. Specific device properties
                          can be set with other args, otherwise if follows the
                          hexagonal layout. Defaults to True.
        snapshot_type (str): The type of snapshots to save of the qubit state at
                             certain points in the cycle. Valid options are
                             'density_matrix' ('dm'), 'expectation_value ('exp')
                             or 'expectation_value_variance' ('exp_var'). For
                             the latter two, the operator to measure expectation
                             value for must be specified using the keyword 
                             pauliop. Defaults to 'density_matrix'.
        device: Whether to conform the circuit to a specific device layout. 
                Available options are None or 'double_diamond'. If set to None,
                it will assume full connectivity. Note that this can also be
                taken care of through transpilation. But specifying a device
                here might help with specific circuits. Defaults to None.
        device_properties: Connectivity and gate set to transpile circuit to.
        encoding (bool): Whether to perform the encoding of qubits at the start.
                         If set to False, the logical state will instead be
                         perfectly initialized.
        theta (float): Zenith angle of the qubit state. Defaults to 0.
        phi (float): Azimuthal angle of the qubit state. Defaults to 0.
        pauliop (str): Five character string corresponding to the five-qubit
                       expectation value to measure (if snapshot_type is set to
                       expectation value or expectation value variance).
                       Defaults to 'ZZZZZ'.
        simulator_type (str): The type of simulation to run. Can be either
                              'statevector' or 'density_matrix'. Defaults to
                              'density_matrix'.
        project (bool): Whether to project the density matrix to the code space
                        when analyzing the results. This will split the fidelity
                        into two parameters: The logical fidelity F_L and the
                        probability of being in the code space P_L. Fidelity
                        can then be calculated as F = F_L * P_L. Defaults to False.
        generator_snapshot (bool): Whether to append a snapshot after each
                                   stabilizer measurement. If set to False, it
                                   will only take snapshots after each cycle.
        idle_snapshots (int): The number of snapshots to append during an
                              (optional) delay time between cycles. Default to 0.
                       
    Returns:
        fid (list): The average fidelity after each stabilizer cycle. If project=True,
                    this is instead the logical fidelity.
        P_L (list, optional): The probability of being in the code space. Only
                              returned if project=True.
        select_counts (list, optional): The remaining runs after each cycle,
            only returned if using data_process_type='post_select'.
        time (dict): Contains the time of each snapshot in the circuit. Keys
                     correspond to the label of its snapshot.
    """

    full_gate_times = extend_standard_gate_times(gate_times)

    # Check the data processing method for settings
    if data_process_type == 'recovery':
        recovery = True
        conditional = False
    elif data_process_type == 'post_select':
        recovery = False
        conditional = True
    elif data_process_type == 'post_process':
        recovery = False
        conditional = True
    else:
        recovery = False
        conditional = False

    # Registers
    qb = QuantumRegister(5, 'code_qubit')
    an = AncillaRegister(2, 'ancilla_qubit')
    cr = get_classical_register(
        n_cycles, reset=reset, recovery=recovery)
    readout = ClassicalRegister(5, 'readout')
    registers = StabilizerRegisters(qb, an, cr, readout)

    # Build the complete circuit
    circ = get_full_stabilizer_circuit(registers, n_cycles=n_cycles, reset=reset,
                                       recovery=recovery,
                                       snapshot_type=snapshot_type,
                                       conditional=conditional,
                                       encoding=encoding, theta=theta, phi=phi,
                                       pauliop=pauliop, device=device,
                                       generator_snapshot=generator_snapshot,
                                       idle_snapshots=idle_snapshots,
                                       simulator_type=simulator_type, final_measure=False, **kwargs)
    if transpile:
        if __package__:
            from . import custom_transpiler
        else:
            import custom_transpiler 
        if device_properties == None:
            device_properties = custom_transpiler.WACQT_device_properties
        circ = custom_transpiler.shortest_transpile_from_distribution(circ, print_cost=False,
                                                    **device_properties)
    if idle_noise:
        circ, time = add_idle_noise_to_circuit(circ, gate_times=full_gate_times,
                                               T1=T1, T2=T2, return_time=True,
                                               **kwargs)
    else:
        time = get_circuit_time(circ=circ, gate_times=full_gate_times)

    # Simulate the circuit
    trivial = get_encoded_state(theta, phi, include_ancillas=None)
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
        return fidelities, time
    else:
        print('Warning: No matching data_process_type')

    return []



def encoding_fidelity(n_shots, gate_times={}, T1=40e3, T2=60e3,
                      idle_noise=True, theta=0., phi=0., iswap=True,
                      snapshot_type='dm', device=None, pauliop='ZZZZZ', project=False):
    """Determines the circuit fidelity of an encoding scheme for the [[5,1,3]]
    code."""

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
    cr = get_classical_register(n_cycles=0)
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



def perfect_stab_circuit(n_cycles, n_shots, gate_times={}, T1=40e3, T2=60e3,
                         reset=True, data_process_type='recovery', snapshot_type='dm',
                         theta=0, phi=0, pauliop='ZZZZZ', include_barriers=True,
                         project=False):
    """Calculates the fidelity for a circuit using so-called perfect stabilizers.
    In practice, this corresponds to moving all noise in a cycle to an idle time
    before, followed by perfectly measuring them. This always gives the correct
    syndrome, and can be seen as an (unreachable) upper limit of the circuit.

    Args:
        See fidelity_from_scratch for detailed docstring.

    Returns:
        fid (list): The average fidelity after each stabilizer cycle. If project=True,
                    this is instead the logical fidelity.
        P_L (list, optional): The probability of being in the code space. Only
                              returned if project=True.
        select_counts (list, optional): The remaining runs after each cycle,
            only returned if using data_process_type='post_select'.
        time (dict): Contains the time of each snapshot in the circuit. Keys
                     correspond to the label of its snapshot.  
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
        n_cycles, reset=reset, recovery=recovery)
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
        circ.compose(get_stabilizer_cycle(registers, reset=reset, recovery=recovery,
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
    basis = basis if isinstance(basis, list) else [basis]
    P = 0
    for basis_vector in basis:
        P += state_fidelity(rho, basis_vector)
    return P
