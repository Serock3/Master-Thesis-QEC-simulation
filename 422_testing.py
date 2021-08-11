# Disable pylinter giving false positives of error.
# pylint: disable=E1101
# %% Import modules


# Standard libraries
import numpy as np
import scipy
import itertools
import warnings
import matplotlib.pyplot as plt

# Qiskit modules/packages
from qiskit import *

# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import thermal_relaxation_error
from qiskit.providers.aer.library import save_density_matrix, save_expectation_value

from qiskit.quantum_info import partial_trace
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import state_fidelity

# Local files
from simulator_program.custom_noise_models import (thermal_relaxation_model_V2,
                                                   standard_times,
                                                   standard_times_delay,
                                                   GateTimes)
from simulator_program.custom_transpiler import (cross_device_properties,
                                                 shortest_transpile_from_distribution)
from simulator_program.data_analysis_tools import (default_execute,
                                                   overlap_with_subspace)
from simulator_program.stabilizers import *
#from simulator_program.stabilizers import (get_full_stabilizer_circuit_422,
#                                           get_repeated_stabilization_422,
#                                           get_encoded_state_422,
#                                           logical_states_422,
#                                           get_classical_register_422,
#                                           StabilizerRegisters,)
from simulator_program.post_select import (get_trivial_post_select_counts_V2,
                                           get_trivial_post_select_counts,
                                           get_trivial_post_select_den_mat,
                                           get_trivial_exp_value)
#from simulator_program.post_process import *
from simulator_program.idle_noise import (add_idle_noise_to_circuit,
                                          get_circuit_time)
from simulator_program.decay import get_idle_encoded_422
from simulator_program.logical_gates_422 import *

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

    # Get the number of remaining shot at each cycle
    select_counts = get_trivial_post_select_counts(results.get_counts(), n_cycles, '00')

    # Get the fidelity for each cycle
    fidelities = [] # If project = True, this contains F_L
    P_Ls = []
    
    if snapshot_type == 'dm' or snapshot_type == 'density_matrix':
        # TODO: Make this return F_L and P_L seperately and fix the references
        for state in get_trivial_post_select_den_mat(results, n_cycles, '00'):
            if project:
                state, P_L = project_dm_to_logical_subspace_422(state, return_P_L=True)
                P_Ls.append(np.real(P_L))
                state = get_encoded_state_422(np.diag(state), include_ancillas=None)
            fidelities.append(state_fidelity(state, trivial))
        if project:
            return fidelities, P_Ls, select_counts, time

    elif snapshot_type == 'exp' or snapshot_type == 'expectation_value':
        fidelities = [post_selected_state for
                      post_selected_state in get_trivial_exp_value(
                          results, n_cycles)]
    return fidelities, select_counts, time

def project_dm_to_logical_subspace_422_V2(rho, return_P_L = False):
    """WARNING: THIS DOES NOT WORK!! Use project_dm_to_logical_subspace_422().
    
    Projects a density-matrix to the logical codespace. This is it the
    [[4,2,2]] equivalent of project_dm_to_logical_subspace_422.

    F(|0_L>, rho_L) where rho_L is returned from this function is equal to
    F(|0>, rho_L) wherre rho_L is the 2x2 matrix returned from V1 above.

    Args:
        rho (DensityMatrix): The 4-qubit state

    Returns:
        DensityMatrix: Projected 4-qubit logical state
    """
    logical = logical_states_422(include_ancillas=None)
    # Projector to the code space
    I_L = (np.outer(logical[0b00], logical[0b00]) + \
          np.outer(logical[0b01], logical[0b01]) + \
          np.outer(logical[0b10], logical[0b10]) + \
          np.outer(logical[0b11], logical[0b11]))#/np.sqrt(2)
    #I_L = Pauli('YYYY').to_matrix()
    # Note here how the projector has to be included for this to work as expected
    logical_pauli_matrices = np.array((
        I_L,
        #Pauli('XXXX').to_matrix()@I_L,
        #Pauli('YYYY').to_matrix()@I_L,
        #Pauli('ZZZZ').to_matrix()@I_L,
        Pauli('XIXI').to_matrix()@I_L,
        Pauli('XZYI').to_matrix()@I_L,
        Pauli('IZZI').to_matrix()@I_L,
        Pauli('XIIX').to_matrix()@I_L,
        Pauli('YIZX').to_matrix()@I_L,
        Pauli('ZIZI').to_matrix()@I_L,
    ))
    P_L = np.trace(rho@logical_pauli_matrices[0])

    rho_L = np.zeros((2**4, 2**4), dtype=complex)
    for i in range(len(logical_pauli_matrices)):
        rho_L += logical_pauli_matrices[i] * \
            np.trace(rho@logical_pauli_matrices[i])/(P_L) #
    rho_L /= np.linalg.norm(rho_L)
    if return_P_L:
        return rho_L, P_L
    return rho_L

def project_dm_to_logical_subspace_422(rho, return_P_L=False):
    """Project a density matrix of a 4-qubit state onto the [[4,2,2]] logical
    subspace.
    """
    P_L = 0
    logical = logical_states_422(include_ancillas=None)
    for i in range(len(logical)):
        P_L += logical[i] @ rho @ logical[i]
    rho_L = np.zeros((4, 4), dtype=complex)
    for i in range(len(logical)):
        for j in range(len(logical)):
            rho_L[i, j] = (logical[i] @ rho @ logical[j])/P_L
    if return_P_L:
        return rho_L, P_L
    return rho_L

# %% Simulations
kwargs = {
    'transpile': False,
    'project': True,
    #'extra_snapshots': True,
}

#%% Post selection
n_cycles=20
n_shots = 1024*2
kwargs = {
    'transpile': False,
    'project': True,
    #'extra_snapshots': True,
}

fid_L_00, P_L_00, counts_00, times = fidelity_from_scratch_422(n_cycles, n_shots,
                                    initial_state = [1.,0.,0.,0.], **kwargs)
fid_L_01, P_L_01, counts_01, times = fidelity_from_scratch_422(n_cycles, n_shots,
                                    initial_state = [0.,1.,0.,0.], **kwargs)
fid_L_10, P_L_10, counts_10, times = fidelity_from_scratch_422(n_cycles, n_shots,
                                    initial_state = [0.,0.,1.,0.], **kwargs)
fid_L_11, P_L_11, counts_11, times = fidelity_from_scratch_422(n_cycles, n_shots,
                                    initial_state = [0.,0.,0.,1.], **kwargs)
#%% Encoded idle decay
n_datapoints = 41
snapshot_times = np.linspace(0,1480*20, n_datapoints)
res_00 = get_idle_encoded_422(snapshot_times, initial_state=[1., 0., 0., 0.])
res_01 = get_idle_encoded_422(snapshot_times, initial_state=[0., 1., 0., 0.])
res_10 = get_idle_encoded_422(snapshot_times, initial_state=[0., 0., 1., 0.])
res_11 = get_idle_encoded_422(snapshot_times, initial_state=[0., 0., 0., 1.])

fid_00_d = [state_fidelity(logical_states_422(include_ancillas=None)[0],
            res_00.data()['dm_'+str(index)]) for index in range(n_datapoints)]
fid_01_d = [state_fidelity(logical_states_422(include_ancillas=None)[1],
            res_01.data()['dm_'+str(index)]) for index in range(n_datapoints)]
fid_10_d = [state_fidelity(logical_states_422(include_ancillas=None)[2],
            res_10.data()['dm_'+str(index)]) for index in range(n_datapoints)]
fid_11_d = [state_fidelity(logical_states_422(include_ancillas=None)[3],
            res_11.data()['dm_'+str(index)]) for index in range(n_datapoints)]

#%% Plot post-selection
fid_00 = []
fid_01 = []
fid_10 = []
fid_11 = []
for i in range(n_cycles+1):
    fid_00.append(fid_L_00[i]*P_L_00[i])
    fid_01.append(fid_L_00[i]*P_L_00[i])
    fid_10.append(fid_L_00[i]*P_L_00[i])
    fid_11.append(fid_L_00[i]*P_L_00[i])

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
t_cycles = np.array([times['dm_con_'+str(i)] for i in range(n_cycles+1)])
ax[0].plot(t_cycles, fid_00, '-o', color='C0', label=r'Fid $|00\rangle_L$')
ax[0].plot(t_cycles, fid_01, '-o', color='C1', label=r'Fid $|01\rangle_L$')
ax[0].plot(t_cycles, fid_10, '-o', color='C2', label=r'Fid $|10\rangle_L$')
ax[0].plot(t_cycles, fid_11, '-o', color='C3', label=r'Fid $|11\rangle_L$')

ax[0].plot(snapshot_times, fid_00_d, color='C0', label=r'Decay $|00\rangle_L$')
ax[0].plot(snapshot_times, fid_01_d, color='k', label=r'Decay $|01\rangle_L$, $|10\rangle_L$, $|11\rangle_L$')
#ax[0].plot(snapshot_times, fid_10_d, color='C2', label=r'$|10_L\rangle$')
#ax[0].plot(snapshot_times, fid_11_d, color='C3', label=r'$|11_L\rangle$')
ax[0].set_xlabel('Time [ns]')
ax[0].set_ylabel(r'State fidelity, $F$')
ax[0].set_title('[[4,2,2]] Post selection fidelity')
ax[0].legend()

frac_00 = []
frac_01 = []
frac_10 = []
frac_11 = []
for i in range(n_cycles+1):
    frac_00.append(counts_00[i]/n_shots)
    frac_01.append(counts_01[i]/n_shots)
    frac_10.append(counts_10[i]/n_shots)
    frac_11.append(counts_11[i]/n_shots)
ax[1].plot(t_cycles, frac_00, '-o', color='C0', label=r'Frac. runs $|00\rangle_L$')
ax[1].plot(t_cycles, frac_01, '-o', color='C1', label=r'Frac. runs $|01\rangle_L$')
ax[1].plot(t_cycles, frac_10, '-o', color='C2', label=r'Frac. runs $|10\rangle_L$')
ax[1].plot(t_cycles, frac_11, '-o', color='C3', label=r'Frac. runs $|11\rangle_L$')

ax[1].plot(snapshot_times, fid_00_d, color='C0', label=r'Decay $|00\rangle_L$')
ax[1].plot(snapshot_times, fid_01_d, color='k', label=r'Decay $|01\rangle_L$, $|10\rangle_L$, $|11\rangle_L$')


ax[1].set_xlabel('Time [ns]')
ax[1].set_ylabel(r'Fraction of runs left')
ax[1].set_title('Fraction of remaining runs')
ax[1].legend()

#%% ============================================================================
# DECAY OF THE ENCODED LOGICAL STATE
n_datapoints = 41
snapshot_times = np.linspace(0,1480*80, n_datapoints)
state_00 = get_encoded_state_422([1., 0., 0., 0.], include_ancillas=None)
state_01 = get_encoded_state_422([0., 1., 0., 0.], include_ancillas=None)
state_10 = get_encoded_state_422([0., 0., 1., 0.], include_ancillas=None)
state_11 = get_encoded_state_422([0., 0., 0., 1.], include_ancillas=None)
state_0p = get_encoded_state_422([1., 1., 0., 0.], include_ancillas=None)
state_p1 = get_encoded_state_422([0., 1., 0., 1.], include_ancillas=None)
state_1p = get_encoded_state_422([0., 0., 1., 1.], include_ancillas=None)
state_p0 = get_encoded_state_422([1., 0., 1., 0.], include_ancillas=None)
state_pp = get_encoded_state_422([1., 1., 1., 1.], include_ancillas=None)
#%%
n_datapoints = 41*2
snapshot_times = np.linspace(0,1480*80, n_datapoints)
res_00 = get_idle_encoded_422(snapshot_times, initial_state=[1., 0., 0., 0.])
res_01 = get_idle_encoded_422(snapshot_times, initial_state=[0., 1., 0., 0.])
res_10 = get_idle_encoded_422(snapshot_times, initial_state=[0., 0., 1., 0.])
res_11 = get_idle_encoded_422(snapshot_times, initial_state=[0., 0., 0., 1.])

res_0p = get_idle_encoded_422(snapshot_times, initial_state=[1., 1., 0., 0.])
res_p1 = get_idle_encoded_422(snapshot_times, initial_state=[0., 1., 0., 1.])
res_1p = get_idle_encoded_422(snapshot_times, initial_state=[0., 0., 1., 1.])
res_p0 = get_idle_encoded_422(snapshot_times, initial_state=[1., 0., 1., 0.])
res_pp = get_idle_encoded_422(snapshot_times, initial_state=[1., 1., 1., 1.])


def projected_fidelity(res, trivial, n_datapoints):
    """Calculate the fidelity of a state after projecting it to the code space.
    
    Args:
        res: Results object from a Qiskit simulation.
        trivial: The correct pure state. Can be a density matrix or statevector.
        n_datapoints (int): Number of data points (snapshots) to calculate the
                            logical fidelity for.
                            
    Returns:
        fid_L (list): Logical fidelities for each snapshot.
        P_Ls (list): Overlap with the code space for each snapshot."""
    P_Ls = []
    fid_L = []
    for state in [res.data()['dm_'+str(index)] for index in range(n_datapoints)]:
        state, P_L = project_dm_to_logical_subspace_422(state, return_P_L=True)
        P_Ls.append(np.real(P_L))
        state = get_encoded_state_422(np.diag(state), include_ancillas=None)
        fid_L.append(state_fidelity(state, trivial))
    return fid_L, P_Ls

fid_L_00_d, P_L_00_d = projected_fidelity(res_00, state_00, n_datapoints)
fid_L_01_d, P_L_01_d = projected_fidelity(res_01, state_01, n_datapoints)
fid_L_10_d, P_L_10_d = projected_fidelity(res_10, state_10, n_datapoints)
fid_L_11_d, P_L_11_d = projected_fidelity(res_11, state_11, n_datapoints)
fid_L_0p_d, P_L_0p_d = projected_fidelity(res_0p, state_0p, n_datapoints)
fid_L_p1_d, P_L_p1_d = projected_fidelity(res_p1, state_p1, n_datapoints)
fid_L_1p_d, P_L_1p_d = projected_fidelity(res_1p, state_1p, n_datapoints)
fid_L_p0_d, P_L_p0_d = projected_fidelity(res_p0, state_p0, n_datapoints)
fid_L_pp_d, P_L_pp_d = projected_fidelity(res_pp, state_pp, n_datapoints)
#%%
fid_00_d = [state_fidelity(state_00,
            res_00.data()['dm_'+str(index)]) for index in range(n_datapoints)]
fid_01_d = [state_fidelity(state_01,
            res_01.data()['dm_'+str(index)]) for index in range(n_datapoints)]
fid_10_d = [state_fidelity(state_10,
            res_10.data()['dm_'+str(index)]) for index in range(n_datapoints)]
fid_11_d = [state_fidelity(state_11,
            res_11.data()['dm_'+str(index)]) for index in range(n_datapoints)]
fid_0p_d = [state_fidelity(state_0p,
            res_0p.data()['dm_'+str(index)]) for index in range(n_datapoints)]
fid_p1_d = [state_fidelity(state_p1,
            res_p1.data()['dm_'+str(index)]) for index in range(n_datapoints)]
fid_1p_d = [state_fidelity(state_1p,
            res_1p.data()['dm_'+str(index)]) for index in range(n_datapoints)]
fid_p0_d = [state_fidelity(state_p0,
            res_p0.data()['dm_'+str(index)]) for index in range(n_datapoints)]
fid_pp_d = [state_fidelity(state_pp,
            res_pp.data()['dm_'+str(index)]) for index in range(n_datapoints)]
fig, ax = plt.subplots(1, 2, figsize=(12, 5))


ax[0].plot(snapshot_times, fid_00_d, label=r'Decay $|00\rangle_L$')
ax[0].plot(snapshot_times, fid_01_d, label=r'Decay $|01\rangle_L$, $|10\rangle_L$, $|11\rangle_L$, $|$+$1\rangle_L$, $|1$+$\rangle_L$')
ax[0].plot(snapshot_times, fid_0p_d, label=r'Decay $|0$+$\rangle_L$, $|$+$0\rangle_L$')
ax[0].plot(snapshot_times, fid_pp_d, label=r'Decay $|$++$\rangle_L$')
ax[0].set_xlabel(r'Time [ns]')
ax[0].set_ylabel(r'State fidelity, F')
ax[0].set_title(r'Decay of encoded state')
ax[0].legend()

ax[1].plot(snapshot_times, fid_L_00_d, label=r'Decay $|00\rangle_L$')
ax[1].plot(snapshot_times, fid_L_01_d, label=r'Decay $|01\rangle_L$, $|10\rangle_L$, $|11\rangle_L$, $|$+$1\rangle_L$, $|1$+$\rangle_L$')
ax[1].plot(snapshot_times, fid_L_0p_d, label=r'Decay $|0$+$\rangle_L$, $|$+$0\rangle_L$')
ax[1].plot(snapshot_times, fid_L_pp_d, label=r'Decay $|$++$\rangle_L$')
ax[1].set_xlabel(r'Time [ns]')
ax[1].set_ylabel(r'Logical state fidelity, $F_L$')
ax[1].set_title(r'Decay of encoded state, projected')
ax[1].legend(loc='lower left')



#%% ============================================================================
# ENCODING FIDELITIES OF THE [[4,2,2]] LOGICAL STATES
#
# This section calculates and presents the fidelity of the four computational
# logical states |00>_L, |01>_L, |10>_L, and |11>_L.
# Its fidelity is measured both as the state fidelity, but also the probability
# of measuring the correcting logical state, wrong logical state, or any state
# outside of the code space.

# Function for running encoding and calculating fidelity
def encoding_fidelity_422(n_shots, gate_times={}, T1=40e3, T2=60e3,
                          idle_noise=True, initial_state=[1,0,0,0],
                          include_swap=False, transpile=False,
                          snapshot_type='dm', project=False, 
                          measure_readout=True, **kwargs):

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
    qb = QuantumRegister(4, 'code_qubit')
    an = AncillaRegister(1, 'ancilla_qubit')
    cr = get_classical_register_422(0)
    readout = ClassicalRegister(4, 'readout')
    registers = StabilizerRegisters(qb, an, cr, readout)

    # Circuits
    circ = get_empty_stabilizer_circuit_422(registers)

    # Encoding
    circ.compose(encode_input_422(registers, initial_state=initial_state,
                 include_swap=include_swap, **kwargs),
                 inplace=True)
    if include_swap:
        qubits = [an[0], qb[1], qb[2], qb[3], qb[0]]  # Qubit permutation
    else:
        qubits = qb
    add_snapshot_to_circuit(circ, snapshot_type=snapshot_type, current_cycle=0,
                            qubits=qubits, **kwargs)

    # Optional measurement of output
    if measure_readout:
        if include_swap:
            circ.measure([an[0],qb[1], qb[2], qb[3]], readout)
        else:
            circ.measure(qb, readout)

    # Transpile
    if transpile:
        circ = shortest_transpile_from_distribution(circ, print_cost=False)

    # Trivial state
    if snapshot_type == 'dm' or snapshot_type == 'density_matrix':
        trivial_res = execute(circ, Aer.get_backend(
            'qasm_simulator'), shots=1).result()
        trivial = trivial_res.data()['dm_0']

    # Add idle noise
    if idle_noise:
        circ, time = add_idle_noise_to_circuit(circ, gate_times=full_gate_times,
                                               T1=T1, T2=T2, return_time=True)
    else:
        time = {'end': None}

    # Run the circuit
    noise_model = thermal_relaxation_model_V2(
        T1=T1, T2=T2, gate_times=full_gate_times)
    results = execute(circ, Aer.get_backend('qasm_simulator'),
                      noise_model=noise_model, shots=n_shots).result()

    # Analyze results
    if snapshot_type == 'dm' or snapshot_type == 'density_matrix':
        state = results.data()['dm_0']
        
        if project:
            state, P_L = project_dm_to_logical_subspace_422(
                state, return_P_L=True)
            state = get_encoded_state_422(np.diag(state), include_ancillas=None)
        fidelities = state_fidelity(state, trivial)
        if project:
            return fidelities, circ, time['end'], P_L
    elif snapshot_type == 'exp' or snapshot_type == 'expectation_value':
        fidelities = results.data()['exp_0']

    return fidelities, circ, results, time['end']

#%% Run simulations
n_shots=1e6 # Number of runs
fid_list = []
circ_list = []
res_list = []
times_list = []
state_list = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1] ]
for i in range(4):
    # Run simulations
    fid, circ, res, times = encoding_fidelity_422(n_shots, gate_times={},
                      T1=40e3, T2=60e3,
                      idle_noise=True, initial_state=state_list[i], 
                      include_swap=False,
                      transpile=True,
                      snapshot_type='dm', project=False)
    fid_list.append(fid)
    circ_list.append(circ)
    res_list.append(res)
    times_list.append(times)   

#%% Print LaTeX table of probabilitiy of
#   correctly measured & incorrect logical & syndrome measured
outcomes = [bin(i)[2:].zfill(4) for i in range(16)]
syndrome_idx = [1,2,4,7,8,11,13,14]

# Group runs based on results
for i in range(4):
    group_counts = [0,0,0]
    counts = res_list[i].get_counts()
    for key in outcomes:
        if key not in counts.keys():
            counts[key] = 0
    logical_idx = [[0,15], [6,9], [5,10],[3,12]]

    # Correct measurements
    correct_idx = logical_idx.pop(i)
    for idx in correct_idx:
        group_counts[0] += counts[outcomes[idx]]

    # Wrong logical state measured
    for entry in logical_idx:
        for idx in entry:
            group_counts[1] += counts[outcomes[idx]]

    # State outside code space measured
    for idx in syndrome_idx:
        group_counts[2] += counts[outcomes[idx]]

    # Print results as LaTeX table input
    print('State: \t&', group_counts[0]/n_shots, '\t& ', group_counts[1]/n_shots,  \
          '\t& ',group_counts[2]/n_shots, ' \\\\ \\hline')


#%% Show transpiled circuit
registers = StabilizerRegisters(qbReg=QuantumRegister(4, 'code_qubit'),
                                anReg=AncillaRegister(1, 'ancilla_qubit'),
                                clReg=get_classical_register_422(0),
                                readout=ClassicalRegister(4, 'readout'))

circ = encode_input_422(registers, initial_state=[1,0,0,0], include_swap=True)
circ = shortest_transpile_from_distribution(circ, print_cost=False, 
                                            **cross_device_properties)
#time = get_circuit_time(circ, gate_times=standard_times)
#print(time)
circ.draw(output='mpl')


#%%=============================================================================
# ERROR CLASSIFICATION
# Generate data regarding the errors present in the qubit state at different
# points of the simulation

# Set up circuit
kwargs = {
    'n_cycles': 20,
    'reset': True,
    'encoding': False,
    'conditional': True,
    'include_barriers': True,
    'initial_state': [1., 0., 0., 0.],
    'extra_snapshots': True,
    'snapshot_type': 'dm',
    'final_measure': False,
    }
gate_times = standard_times

# Get the complete circuit
circ = get_full_stabilizer_circuit_422(**kwargs)
circ, times = add_idle_noise_to_circuit(circ, gate_times, return_time=True)


#%% Run simulation
n_shots = 1024*2
results = default_execute(circ, n_shots, gate_times=gate_times)

#%% Functions necessary
def reduce_key(key: str, current_snapshot: int, measurements_per_snapshot=1):
    """Cuts of the initial bits of a key to get the equivalent one at earlier cycles. 

    E.g. \n
    current_cycle = 1 \n
    '0xff' -> '11111111' -> '1111' - > '0xf'  \n
    current_cycle = 3 \n
    '0xff' -> '11111111' -> '11111111' -> '0xff' \n

    Args:
        key (str): Hex key for the measurement outcome
        current_snapshot (int): Index of the snapshot
        measurements_per_snapshot (int, optional): Number of measurements (to different cl registers) that takes place in between every snapshot. Defaults to one. If only one snap is take every cycle, set this to the number of stabilizers.

    Returns:
        reduced key (str): Hex key matching the current cycle.
    """
    return hex(int(bin(int(key, 16))[2:][:measurements_per_snapshot*current_snapshot].zfill(1), 2))

full_syndromes = [[(XGate, 0)],
                  [(XGate, 1)],
                  [(XGate, 2)],
                  [(XGate, 3)],
                  [(ZGate, 0)],
                  [(ZGate, 1)],
                  [(ZGate, 2)],
                  [(ZGate, 3)],
                  [(XGate, 0), (ZGate, 0)],
                  [(XGate, 1), (ZGate, 1)],
                  [(XGate, 2), (ZGate, 2)],
                  [(XGate, 3), (ZGate, 3)]]

def get_syndrome_basis():
    """Returns a basis set for every state outside of the code space of the
    [[4,2,2]] QED code.
    """

    logical_00 = Statevector(logical_states_422(None)[0])

    syndromes = [logical_00]*12

    for syndrome in range(len(syndromes)):  # Loop over possible errors
        for correction_strategy in full_syndromes[syndrome]:
            syndromes[syndrome-1] = syndromes[syndrome-1].evolve(
                    correction_strategy[0](), [correction_strategy[1]])

    return syndromes

#%% Calculate density matrix overlap with basis states

time = np.empty(label_counter.value)
P_00 = np.empty(label_counter.value)
P_01 = np.empty(label_counter.value)
P_10 = np.empty(label_counter.value)
P_11 = np.empty(label_counter.value)
P_w1 = np.empty(label_counter.value)


logical = logical_states_422(include_ancillas=None)
key = '0x0'
for i in range(label_counter.value):
    label = get_snapshot_label('dm', kwargs['conditional'], i)
    if kwargs['conditional']:
        rho = results.data()[label][reduce_key(key, i)]
    else:
        rho = results.data()[label]
    time[i] = times[label]*1e-3

    P_00[i] = overlap_with_subspace(rho, logical[0b00])
    P_01[i] = overlap_with_subspace(rho, logical[0b01])
    P_10[i] = overlap_with_subspace(rho, logical[0b10])
    P_11[i] = overlap_with_subspace(rho, logical[0b11])
    P_w1[i] = overlap_with_subspace(rho, get_syndrome_basis())

#%% Plotting

# Group all other logical states
P_other = P_01 + P_10 + P_11
order = np.array([P_00, P_w1, P_other])

data_points = [i for i in range(1, label_counter.value-1)]
if kwargs['conditional']:
    counts_at_snapshots = get_trivial_post_select_counts_V2(
        results.get_counts(),
        data_points,
        label_counter.value) 
else:
    counts_at_snapshots = np.ones(label_counter.value)

colors = plt.get_cmap("tab10")
for i in range(3):
    lower = order[:i].sum(axis=0)
    plt.fill_between(time, (order[i]+lower)*counts_at_snapshots, lower*counts_at_snapshots,
                     color=np.array(colors(i))-np.array([0, 0, 0, 0.3]))
    plt.plot(time, ((order[i]+lower)*counts_at_snapshots), '.', color=colors(i))

if kwargs['conditional']:
    plt.ylabel('Counts')
else:
    plt.ylabel('Overlap')
plt.xlabel('Time [μs]')
plt.ylim((0, counts_at_snapshots[0]))
plt.xlim((0, time[-1]))
plt.legend(labels=[r'$|00\rangle_L$', 'Syndrome', 'Incorrect code word'])

#%%=============================================================================
# TRANSVERSAL GATES



initial_state = [0.,1.,0.,0.]
n_cycles = 20
kwargs = {
    'snapshot_type': 'dm',
    'conditional': True,
    'extra_snapshots': True,
    'flip_stab_order': True,
}
# Define registers, circuits and initialize the logical state
registers = StabilizerRegisters(qbReg=QuantumRegister(4, 'code_qubit'),
                                anReg=AncillaRegister(1, 'ancilla_qubit'),
                                clReg=get_classical_register_422(n_cycles),
                                readout=ClassicalRegister(4, 'readout'))
circ = get_empty_stabilizer_circuit_422(registers)
circ.set_density_matrix(get_encoded_state_422(initial_state))

logical_ops = [logical_X(0,registers.QubitRegister),
               logical_X(1,registers.QubitRegister),
               logical_CNOT(1,0,registers.QubitRegister),
               logical_SWAP(registers.QubitRegister)]

add_snapshot_to_circuit(circ, current_cycle=0, qubits=registers.QubitRegister,
                        **kwargs)
for i in range(n_cycles):
    circ.compose(stabilizer_cycle_422(registers, current_cycle=i, **kwargs), inplace=True)
    #if kwargs['flip_stab_order']:
    #    kwargs['flip_stab_order'] = False
    #else:
    #    kwargs['flip_stab_order'] = True
    
    #if i==1:
        #circ.barrier()
        #circ.compose(logical_CZ(registers.QubitRegister), inplace=True)
        #circ.compose(logical_iSWAP(registers.QubitRegister), inplace=True)
        #circ.barrier()
    #circ.compose(logical_ops[i], inplace=True)

    #circ.save_density_matrix(label='test_'+str(i))
circ.measure(registers.QubitRegister, registers.ReadoutRegister)

# Transpile
circ = shortest_transpile_from_distribution(circ, print_cost=False,
                                            optimization_level=3,
                                            **cross_device_properties)
#circ.draw(output='mpl')
#%% Run simulation

results = execute(
        circ,
        Aer.get_backend('qasm_simulator'),
        #Aer.get_backend('unitary_simulator'),
        #noise_model=None,
        noise_model=thermal_relaxation_model_V2(gate_times=standard_times),
        shots=1024*4
    ).result()

#print(results.get_counts())

#%% Post select
project=False

# Get the number of remaining shot at each cycle
select_counts = get_trivial_post_select_counts(results.get_counts(), n_cycles, '00')

# Get the fidelity for each cycle
fidelities = [] # If project = True, this contains F_L
P_Ls = []
trivial = get_encoded_state_422(initial_state, include_ancillas=None)

# TODO: Make this return F_L and P_L seperately and fix the references
for state in get_trivial_post_select_den_mat(results, n_cycles, '00'):
    if project:
        state, P_L = project_dm_to_logical_subspace_422(state, return_P_L=True)
        P_Ls.append(np.real(P_L))
        state = get_encoded_state_422(np.diag(state), include_ancillas=None)
    fidelities.append(state_fidelity(state, trivial))



# TODO: Transpilera kretsarna, ta fram circuit time
# TODO: Testa "blenda in" iSWAP i sista stabilizern. Kan transpilern lösa det
#       automatiskt kanske?
# TODO: Printa kretsar och skriv rapporten

#%%

initial_state = [0.,0.,0.,1.]
n_cycles = 20
kwargs = {
    'snapshot_type': 'dm',
    'conditional': False,
    'extra_snapshots': True,
    'flip_stab_order': True,
}
# Define registers, circuits and initialize the logical state
qb = QuantumRegister(4,'qubit')
readout = ClassicalRegister(4,'readout')
circ = QuantumCircuit(qb, readout)
#circ.set_density_matrix(get_encoded_state_422(initial_state, None))
circ.set_statevector(get_encoded_state_422(initial_state, None))
circ.save_statevector('dm_0')
circ.s(qb)
circ.compose(logical_Z(2,qb), inplace=True)
circ.save_statevector('dm_1')
#add_snapshot_to_circuit(circ, qubits=qb, **kwargs)
circ.measure(qb, readout)

# Transpile
#circ = shortest_transpile_from_distribution(circ, print_cost=False,
#                                            optimization_level=3,
#                                            **cross_device_properties)
circ.draw()
#% Run simulation

results = execute(
        circ,
        Aer.get_backend('qasm_simulator'),
        #Aer.get_backend('unitary_simulator'),
        noise_model=None,
        #noise_model=thermal_relaxation_model_V2(gate_times=standard_times),
        shots=1024*4
    ).result()

#%

#results.data()['counts']
#dm1 = project_dm_to_logical_subspace_422(results.data()['dm_0'])
np.set_printoptions(precision=2)
print(results.data()['dm_0'])
print(results.data()['dm_1'])

#%%
a = qiskit.quantum_info.random_clifford(1)
b = a.to_circuit()
b.draw()

#%% Testing part of diamond-article
qb = QuantumRegister(4, 'qubit')
an = AncillaRegister(1, 'ancilla')
cl = ClassicalRegister(1, 'syndrome')
readout = ClassicalRegister(4, 'readout')

circ = QuantumCircuit(qb, an, cl, readout)

circ.h(an)
circ.cx(an, qb[0])
circ.cx(an, qb[1])
circ.cx(an, qb[2])
circ.cx(an, qb[3])
circ.h(an)
circ.measure(an,cl)
circ.barrier()
circ.save_density_matrix(qb,'test')
circ.draw()

results = execute(
        circ,
        Aer.get_backend('qasm_simulator'),
        #Aer.get_backend('unitary_simulator'),
        noise_model=None,
        #noise_model=thermal_relaxation_model_V2(gate_times=standard_times),
        shots=1024*4
    ).result()

results.data()['test'].shape