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

#%% Useful functions
def get_testing_circuit(registers, reset, recovery, n_cycles,
        snapshot_type='dm', initial_state=0, conditional=False, **kwargs):
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
        circ.x(0) #initialize in 1
    circ.compose(encode_input_v2(registers), inplace=True)
    add_snapshot_to_circuit(circ, snapshot_type, 0, registers.QubitRegister,
        conditional=conditional)
    circ.compose(get_repeated_stabilization(registers, n_cycles,
        reset, recovery, flag=False, snapshot_type=snapshot_type,
        conditional=conditional, **kwargs), inplace=True)
    return circ

def fidelity_from_scratch(n_cycles, noise_model, n_shots, gate_times={}, reset=True,
        recovery=True, post_select=False, post_process=False, idle_noise=True, 
        empty_circuit=False, transpile=True, snapshot_type='dm',
        conditional=False, **kwargs):
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

    # Registers
    qb = QuantumRegister(5, 'code_qubit')
    an = AncillaRegister(2, 'ancilla_qubit')
    cr = get_classical_register(n_cycles, reset=reset, recovery=recovery, flag=False)
    readout = ClassicalRegister(5, 'readout')
    registers = StabilizerRegisters(qb, an, cr, readout)

    # Circuits
    circ = get_testing_circuit(registers, reset, recovery, n_cycles, 
        snapshot_type, conditional=conditional, **kwargs)
    if transpile:
        circ = shortest_transpile_from_distribution(circ, print_cost=False)

    # Get the correct (no errors) state
    #trivial = get_trivial_state(circ)
    trivial = logical_states(include_ancillas=None)[0]

    # Create empty encoded circuit
    if empty_circuit:
        time = get_circuit_time(circ, gate_times)
        circ = get_empty_noisy_circuit_v3(circ, time, gate_times)
        # TODO: Make this part of get_empty_circuit to remove warnings
        #circ = add_idle_noise_to_circuit(circ, gate_times)

        results = execute(circ, Aer.get_backend('qasm_simulator'),
            noise_model=noise_model, shots=n_shots).result()


        # Calculate fidelity at each snapshot
        fidelities = []
        for current_cycle in range(n_cycles+1):
            state = results.data()['dm_' + str(current_cycle)]
            fidelities.append(state_fidelity(state, trivial))
        return fidelities
        

    # Add idle noise (empty_circuit does this automatically)
    elif idle_noise:
        circ = add_idle_noise_to_circuit(circ, gate_times=gate_times)

    # Run the circuit
    results = execute(circ, Aer.get_backend('qasm_simulator'),
        noise_model=noise_model, shots=n_shots).result()
    if recovery:
        fidelities = []
        for current_cycle in range(n_cycles+1):
            state = results.data()['dm_' + str(current_cycle)]
            fidelities.append(state_fidelity(state, trivial))

        return fidelities

    if post_select:

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

    if post_process:
        print('Warning: Post-process not implemented, exiting...')
        return []
    return

# %%
# Settings used across all configurations
n_cycles = 2
n_shots = 1024

# Noise models
target_noise = thermal_relaxation_model_V2(gate_times=WACQT_target_times)
current_noise = thermal_relaxation_model_V2(gate_times=WACQT_demonstrated_times)

# Quantum error correction for both noise models
fid = fidelity_from_scratch(n_cycles, target_noise, n_shots, 
    gate_times=WACQT_target_times, reset=True, recovery=False, post_select=True,
    post_process=False, idle_noise=True, empty_circuit=True, conditional=False,
    snapshot_type='dm')
