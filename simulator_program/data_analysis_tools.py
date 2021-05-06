# This file is meant as the final version of most functions from
# 'plotting_expval.py', in order to separate different parts and make the code 
# more readable.

# %% Import modules
#import seaborn as sns
#import matplotlib.pyplot as plt
import numpy as np
import itertools
from qiskit import *
#from qiskit.visualization import plot_histogram

# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import thermal_relaxation_error
from qiskit.providers.aer.library import save_density_matrix, save_expectation_value                                        

from qiskit.quantum_info import partial_trace
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import state_fidelity

# Our own files
from .custom_noise_models import (thermal_relaxation_model,
    thermal_relaxation_model_V2,
    WACQT_target_times,
    WACQT_demonstrated_times)
from .custom_transpiler import *
from .stabilizers import *
from .post_select import *
from .post_process import *
from .idle_noise import *

#%%


def fidelity_from_scratch(n_cycles, n_shots, gate_times={}, T1=40e3, T2=60e3,
        reset=True, data_process_type='recovery', idle_noise=True, transpile=True, 
        snapshot_type='dm', device=None, device_properties=WACQT_device_properties,
        encoding=True, theta=0, phi=0, pauliop='ZZZZZ', **kwargs):

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

    # Noise model
    noise_model = thermal_relaxation_model_V2(T1=T1, T2=T2, gate_times=full_gate_times)

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
                                       conditional=conditional,
                                       encoding=encoding, theta=theta, phi=phi,
                                       pauliop=pauliop, device=device,
                                       **kwargs)

    if transpile:
        circ = shortest_transpile_from_distribution(circ, print_cost=False,
            **device_properties)

    # Get the correct (no errors) state
    trivial = logical_states(include_ancillas=None)[1]

    # Create empty encoded circuit
    if data_process_type == 'empty_circuit':

        # Prepare the circuit
        time = get_circuit_time(circ, full_gate_times)
        circ = get_empty_noisy_circuit_v3(circ, time, full_gate_times,
                                          T1=T1, T2=T2)
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
        circ, time = add_idle_noise_to_circuit(circ, gate_times=full_gate_times,
                                         T1=T1, T2=T2, return_time=True)

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

        return fidelities, time

    elif data_process_type == 'post_select':
        # Get the fidelity for each cycle
        if snapshot_type=='dm' or snapshot_type=='density_matrix':
            fidelities = [state_fidelity(post_selected_state, trivial) for 
                post_selected_state in get_trivial_post_select_den_mat(
                results, n_cycles)]
        elif snapshot_type=='exp' or snapshot_type=='expectation_value':
            fidelities = [post_selected_state for 
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
    qb = QuantumRegister(1,'qubit')
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
            circ.save_expectation_value(Pauli(pauliop), qb,label='snap_'+str(index))
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

    circ_single = get_idle_single_qubit(time, snapshot_type, T1, T2, 
                                        theta=theta, phi=phi, pauliop=pauliop)
    results = execute(circ_single, Aer.get_backend('qasm_simulator'),
        noise_model=None, shots=n_shots).result()
    fidelities = [1.0] # The initial state
    
    if snapshot_type == 'dm' or snapshot_type =='density_matrix':
        trivial = results.data()['start']
        for i in range(len(time)-2):
            current_state = results.data()['snap_'+str(i+1)]
            fidelities.append(state_fidelity(current_state, trivial))
    elif snapshot_type == 'exp' or snapshot_type == 'expectation_value':
        for i in range(len(time)-2):
            fidelities.append(results.data()['snap_'+str(i+1)])
    return fidelities, time

def encoding_fidelity(n_shots, gate_times={}, T1=40e3, T2=60e3,
        reset=True, idle_noise=True, initial_state=0,
        snapshot_type='dm', device=None, pauliop='ZZZZZ'):

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
    cr = get_classical_register(n_cycles, reset=reset, recovery=True, flag=False)
    readout = ClassicalRegister(5, 'readout')
    registers = StabilizerRegisters(qb, an, cr, readout)

    # Circuits
    circ = get_empty_stabilizer_circuit(registers)
    if initial_state == 1:
        circ.x(qb[0])
    if device == 'WACQT':
        circ.compose(transpiled_encoding_WACQT(registers), inplace=True)
    elif device == 'DD':
        circ.compose(transpiled_encoding_DD(registers), inplace=True)
    else:
        circ.compose(encode_input_v2(registers), inplace=True)
    add_snapshot_to_circuit(circ, snapshot_type=snapshot_type, current_cycle=0, qubits=qb,
                            pauliop=pauliop, include_barriers=True)

    # Trivial state
    if snapshot_type=='dm' or snapshot_type=='density_matrix':
        trivial_res = execute(circ, Aer.get_backend('qasm_simulator'), shots=1).result()
        trivial = trivial_res.data()['dm_0']

    if idle_noise:
        circ, time = add_idle_noise_to_circuit(circ, gate_times=full_gate_times,
                                         T1=T1, T2=T2, return_time=True)

    # Run the circuit
    noise_model = thermal_relaxation_model_V2(T1=T1, T2=T2, gate_times=full_gate_times)
    results = execute(circ, Aer.get_backend('qasm_simulator'),
        noise_model=noise_model, shots=n_shots).result()
    if snapshot_type=='dm' or snapshot_type=='density_matrix':
        state = results.data()['dm_0']
        fidelities = state_fidelity(state, trivial)
    elif snapshot_type=='exp' or snapshot_type=='expectation_value':
        fidelities = results.data()['exp_0']
    return fidelities, circ

#%%

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
    x_D = np.ones((n_cycles,2))
    for i in range(n_cycles):
        if time is not None:
            try:
                x_D[i][1] = time['exp_'+str(i+1)]*1e-3
            except:
                x_D[i][1] = time['dm_'+str(i+1)]*1e-3
        else:
            x_D[i][1] += i
    y = np.log( np.reshape(np.asarray(fidelity[1:]), (n_cycles,1)) )
    theta = np.dot(np.dot(np.linalg.inv(np.dot(x_D.T, x_D)), x_D.T), y)

    MSE = 0.
    for cycle in range(n_cycles):
        y_pred = np.exp(theta[0]) * np.exp((cycle+1)*theta[1])
        MSE += (y_pred-fidelity[cycle+1])**2

    # TODO: Only return theta[1] maybe?
    return theta, MSE

def sweep_parameter_space(T1, T2, single_qubit_gate_time, two_qubit_gate_time, 
        measure_time, n_cycles=8, n_shots=2048, single_qubit=False, save=None,
        time_axis=False, **kwargs):
    """Calculate the logical error rate across a variety of parameters
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
                        measure_time]
    noise_parameters = [[param] if not isinstance(param, list) else param for param in noise_parameters]
    
    # Generate an array to store the data in
    sweep_lengths = [len(param) for param in noise_parameters]
    error_array = np.zeros(sweep_lengths)
    MSE_array = np.zeros(sweep_lengths)

    # Get all combinations of parameters
    index = 0
    for params in itertools.product(*noise_parameters):
        
        gate_times = GateTimes(params[2], params[3],
                               {'u1': 0, 'z': 0, 'measure': params[4]})
        
        # Skip cases where T2 > 2*T1
        if params[1] > 2*params[0]:
            index+=1
            continue

        if single_qubit:
            fid, time = fid_single_qubit(n_cycles, n_shots, T1=params[0], 
                                T2=params[1], gate_times=gate_times, **kwargs)

            # Normalize data if needed
            # TODO: Better solution? Now it checks if input state is |+>
            if theta==np.pi/2 and phi==np.pi/2: 
                for i in range(len(fid)):
                    fid[i] = 2.*fid[i] - 1.
        else:
            fid, time = fidelity_from_scratch(n_cycles, n_shots, T1=params[0], 
                                T2=params[1], gate_times=gate_times, **kwargs)
            
        # From fidelities, do OLS
        if time_axis:
            error_rate, MSE = get_error_rate(fid, time)
        else:
            error_rate, MSE = get_error_rate(fid)

        array_indexes = _get_array_indexes(index, sweep_lengths)
        error_array[array_indexes] = error_rate[1]
        MSE_array[array_indexes] = MSE
        index += 1
 
    # Save results to file
    # TODO: Save as txt instead? Make it both readable and have the parameters used
    if save is not None:
        np.save(save, error_array)
        np.save(save+'_MSE', MSE_array)
    
    return error_array, MSE_array
