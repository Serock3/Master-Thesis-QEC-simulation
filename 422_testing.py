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

from qiskit.quantum_info import partial_trace
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import state_fidelity

# Our own files

from simulator_program.custom_noise_models import (thermal_relaxation_model,
                                 thermal_relaxation_model_V2,
                                 WACQT_target_times,
                                 WACQT_demonstrated_times,
                                 standard_times)
from simulator_program.custom_transpiler import *
from simulator_program.stabilizers import *
from simulator_program.post_select import *
from simulator_program.post_process import *
from simulator_program.idle_noise import *

from simulator_program.data_analysis_tools import project_dm_to_logical_subspace_V2
#%%
def fidelity_from_scratch_422(n_cycles, n_shots, gate_times={}, T1=40e3, T2=60e3,
                          idle_noise=True,
                          snapshot_type='dm',
                          transpile=True, 
                          device_properties=cross_device_properties,
                          initial_state=[1.,0.,0.,0.], 
                          simulator_type='density_matrix',
                          project=False, **kwargs):
    """Get the fidelity of a certain setup/configuration from only its
    parameters.

    Args:
        n_cycles (int): The number of stabilizer cycles to be performed.
        n_shots (int): The number of runs of the circuit.
        gate_times: Can be either a dict with some gate times (in ns), or a
                    GateTimes object. If it is a dict, gate times not included 
                    will be added from standard gate times.
        T1 (float): T1 thermal relaxation constant. (Amplitude damping)
        T2 (float): T2 thermal relaxation constant. (Phase damping)
        idle_noise (bool): Whether or not to add thermal relaxation noise to 
                           idle qubits. This follows the T1 and T2 specified. 
                           Defaults to True.
        snapshot_type (str): The type of snapshots to add to circuit. Possible
                             options are 'dm' or 'denstiy_matrix' and
                             'exp' or 'expectation_value'. Defaults to 'dm'.
        transpile (bool): Whether or not to transpile the circuit.
        device_properties (dict): The set of properties to transpile circuit to
        initial_state (list): The initial two-qubit state, given as its
                              statevector. Defaults to [1., 0., 0., 0.].
        simulator_type (str): The type of simulation to run. Possible options
                              are 'statevector' or 'density_matrix'. Defaults
                              to 'density_matrix'. 
        project (bool): Whether to project density matrix snapshots to the code
                        space. This replaces the fid output with the logical
                        fidelity and adds P_L as an extra output.

    Possible kwargs:
        conditional (bool): Whether to separate snapshots based on measurement
                            outcomes. To do post-select, this must be true.
                            Defaults to true.
        encoding (bool): Whether or not to perform the encoding circuit before
                         stabilizer cycles. If False, the logical state will
                         be perfectly initialized. Defaults to False.
        extra_snapshots (bool): Whether to include additional snapshots after
                                each stabilizer measurement. Defaults to False.
        include_barriers (bool): Whether or not to add barriers between certain
                                 segments of the circuit. This may prevent
                                 unwanted moving of gates. Defaults to True.
        pauliop (str): String corresponding to the observable whose expectation
                       value is measured in a snapshot (if snapshot type is
                       set to expectation value). Defaults to ZZZZ.
        reset (bool): Whether or not to reset ancilla between measurements.
                      defaults to True.

    Returns:
        fid (list): The average fidelity after each stabilizer cycle.
        select_counts (list): The remaining runs after each cycle.
        times: The time passed at each snapshot in the circuit
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

    # Manage kwargs used for circuit building
    default_kwargs = {'conditional': True,
                      'encoding': False,
                      'extra_snapshots': False,
                      'include_barriers': True,
                      'pauliop': 'ZZZZ',
                      'reset': True,
                      'snapshot_type': snapshot_type} # DM if nothing specified
    for key in default_kwargs:
        if key not in kwargs:
            kwargs[key] = default_kwargs[key]

    # Noise model
    noise_model = thermal_relaxation_model_V2(
        T1=T1, T2=T2, gate_times=full_gate_times)
    noise_model = None

    # Circuits
    circ = get_full_stabilizer_circuit_422(n_cycles=n_cycles, 
                                       initial_state=initial_state,
                                       simulator_type=simulator_type,
                                       **kwargs)

    if transpile:
        circ = shortest_transpile_from_distribution(circ, print_cost=False,
                                                    **device_properties)

    # Get the correct (no errors) state
    trivial = get_encoded_state_422(initial_state, include_ancillas=None)

    # Add idle noise
    if idle_noise:
        circ, time = add_idle_noise_to_circuit(circ, gate_times=full_gate_times,
                                               T1=T1, T2=T2, return_time=True,
                                               **kwargs)
    else: 
        time = get_circuit_time(circ=circ, gate_times=full_gate_times)

    # Run the circuit
    simulator = Aer.get_backend('qasm_simulator')
    try:
        simulator.set_option('method', simulator_type)
    except:
        print('Invalid simulator type, defaulting to density_matrix')
        simulator.set_option('method', 'density_matrix')
    results = execute(circ, simulator,
                      noise_model=noise_model, shots=n_shots).result()
    #return results
    # Get the number of remaining shot at each cycle
    select_counts = get_trivial_post_select_counts(results.get_counts(), n_cycles, '00')

    # Get the fidelity for each cycle
    fidelities = [] # If project = True, this contains F_L
    P_Ls = []
    
    if snapshot_type == 'dm' or snapshot_type == 'density_matrix':
        # TODO: Make this return F_L and P_L seperately and fix the references
        for state in get_trivial_post_select_den_mat(results, n_cycles, '00'):
            if project:
                state, P_L = project_dm_to_logical_subspace_V2(state, return_P_L=True)
                P_Ls.append(np.real(P_L))
            fidelities.append(state_fidelity(state, trivial))
        if project:
            return fidelities, P_Ls, select_counts, time

    elif snapshot_type == 'exp' or snapshot_type == 'expectation_value':
        fidelities = [post_selected_state for
                      post_selected_state in get_trivial_exp_value(
                          results, n_cycles)]
    return fidelities, select_counts, time

# %%
res = fidelity_from_scratch_422(0, 1024, transpile=True, encoding=True,
                                idle_noise=False, initial_state=[0., 0., 0., 1.],
                                extra_snapshots=True)
#fid, counts, time = fidelity_from_scratch_422(5, 1024, transpile=False)
print(res.get_counts())