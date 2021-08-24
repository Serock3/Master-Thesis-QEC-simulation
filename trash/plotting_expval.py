# THIS FILE IS (SOMEWHAT) DEPRECATED. All code has been moved to other files,
# which should be used instead.
# - All functions are in simulator_program.data_analysis_tools.py
# - All runs/configurations for expectation values and fidelity are in main_expval.py
# - All parameter sweep runs/configurations are main_sweep.py

# %% Import modules
import matplotlib.pyplot as plt
import numpy as np
from qiskit import *
# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import thermal_relaxation_error

from qiskit.quantum_info import state_fidelity

# Our own files
from simulator_program.data_analysis_tools import *
from simulator_program.custom_noise_models import (
    thermal_relaxation_model_V2,
    WACQT_target_times,
    WACQT_demonstrated_times,
    standard_times)
from simulator_program.custom_transpiler import *
from simulator_program.stabilizers import *
from simulator_program.post_select import *
from simulator_program.post_process import *
from simulator_program.idle_noise import *

# %% Useful functions


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
        full_gate_times = WACQT_gate_times.get_gate_times(
            custom_gate_times=gate_times)
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
    noise_model = thermal_relaxation_model_V2(
        T1=T1, T2=T2, gate_times=full_gate_times)

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
        if snapshot_type == 'dm' or snapshot_type == 'density_matrix':
            for current_cycle in range(n_cycles):
                state = results.data()['dm_' + str(current_cycle+1)]
                fidelities.append(state_fidelity(state, trivial))
        elif snapshot_type == 'exp' or snapshot_type == 'expectation_value':
            for current_cycle in range(n_cycles):
                fidelities.append(
                    results.data()['exp_' + str(current_cycle+1)])
        return fidelities

    # Add idle noise (empty_circuit does this automatically)
    if idle_noise:
        circ, time = add_idle_noise_to_circuit(circ, gate_times=full_gate_times,
                                               T1=T1, T2=T2, return_time=True)

    # Run the circuit
    results = execute(circ, Aer.get_backend('qasm_simulator'),
                      noise_model=noise_model, shots=n_shots).result()

    if data_process_type == 'recovery' or data_process_type == 'none':
        fidelities = []
        if snapshot_type == 'dm' or snapshot_type == 'density_matrix':
            for current_cycle in range(n_cycles+1):
                state = results.data()['dm_' + str(current_cycle)]
                fidelities.append(state_fidelity(state, trivial))
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
    return fidelities


def temp_dd_cycle(registers, current_cycle=0, current_step=0, reset=True,
                  recovery=False):
    """Gives the circuit for a full unflagged stabilizer cycle pre-transpiled
    onto a square grid. This function acts as a bandaid solution to perfectly
    encoding these circuits, due to errors in saving density matrix or
    expectation values after transpiling set_density_matrix.

    Args:
        registers (StabilizerRegister): Register object
        anQb (AncillaQubit, optional): Specifies the ancilla to use for the measurement. Defaults to None.
        reset (bool, optional): Whether to reset ancillas between measurements. Defaults to True.
        current_cycle (int, optional): [description]. Defaults to 0.
        current_step (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """
    # Create list of syndrome bits
    if isinstance(registers.SyndromeRegister, list):
        syn_reg = registers.SyndromeRegister[0][current_cycle][current_step]
        syn_bit_list = [syn_reg[n] for n in range(4)]
    else:
        syn_bit_list = [registers.SyndromeRegister[n] for n in range(4)]

    # Create a circuit
    qbReg = registers.QubitRegister
    anReg = registers.AncillaRegister
    circ = get_empty_stabilizer_circuit(registers)

    # XZZXI
    circ.h(qbReg[0])
    circ.h(qbReg[3])
    circ.h(anReg[1])

    circ.cz(anReg[1], qbReg[0])
    circ.cz(anReg[1], qbReg[1])
    circ.cz(anReg[1], qbReg[2])
    circ.cz(anReg[1], qbReg[3])

    circ.h(qbReg[0])
    circ.h(qbReg[3])
    circ.h(anReg[1])

    circ.measure(anReg[1], syn_bit_list[0])
    if reset:
        circ.reset(anReg[1])
    circ.barrier()

    # IXZZX
    circ.h(anReg[1])
    circ.h(qbReg[1])
    circ.h(qbReg[4])

    circ.cz(anReg[1], qbReg[1])

    circ.iswap(qbReg[1], qbReg[4])
    circ.cz(qbReg[1], qbReg[4])
    circ.u1(-np.pi/2, qbReg[1])
    circ.u1(-np.pi/2, qbReg[4])

    circ.cz(anReg[1], qbReg[2])
    circ.cz(anReg[1], qbReg[3])
    circ.cz(anReg[1], qbReg[1])  # 4-1

    circ.h(anReg[1])
    circ.h(qbReg[1])
    circ.h(qbReg[4])

    circ.measure(anReg[1], syn_bit_list[1])
    if reset:
        circ.reset(anReg[1])
    circ.barrier()
    # XIXZZ

    circ.h(anReg[1])
    circ.h(qbReg[0])
    circ.h(qbReg[2])

    circ.cz(anReg[1], qbReg[0])
    circ.cz(anReg[1], qbReg[2])
    circ.cz(anReg[1], qbReg[3])
    circ.cz(anReg[1], qbReg[1])  # 4-1

    circ.h(anReg[1])
    circ.h(qbReg[0])
    circ.h(qbReg[2])

    circ.measure(anReg[1], syn_bit_list[2])
    if reset:
        circ.reset(anReg[1])
    circ.barrier()

    # ZXIXZ
    circ.iswap(qbReg[1], qbReg[4])
    circ.cz(qbReg[1], qbReg[4])
    circ.u1(-np.pi/2, qbReg[1])
    circ.u1(-np.pi/2, qbReg[4])

    circ.h(anReg[1])
    circ.h(qbReg[1])
    circ.h(qbReg[3])

    circ.cz(anReg[1], qbReg[0])
    circ.cz(anReg[1], qbReg[1])
    circ.cz(anReg[1], qbReg[3])

    circ.h(qbReg[1])

    circ.iswap(qbReg[1], qbReg[4])
    circ.cz(qbReg[1], qbReg[4])
    circ.u1(-np.pi/2, qbReg[1])
    circ.u1(-np.pi/2, qbReg[4])

    circ.cz(anReg[1], qbReg[1])  # 4-1

    circ.h(anReg[1])
    circ.h(qbReg[3])

    circ.iswap(qbReg[1], qbReg[4])
    circ.cz(qbReg[1], qbReg[4])
    circ.u1(-np.pi/2, qbReg[1])
    circ.u1(-np.pi/2, qbReg[4])
    circ.measure(anReg[1], syn_bit_list[3])
    if reset:
        circ.reset(anReg[1])

    # Recovery
    if recovery is True:
        circ.barrier()
        circ.compose(get_recovery(
            registers, reset, current_cycle), inplace=True)
        circ.barrier()
    return circ


def encoding_fidelity(n_shots, gate_times={}, T1=40e3, T2=60e3,
                      reset=True, idle_noise=True, initial_state=0,
                      snapshot_type='dm', device=None, pauliop='ZZZZZ'):

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
        n_cycles, reset=reset, recovery=True, flag=False)
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
    if snapshot_type == 'dm' or snapshot_type == 'density_matrix':
        trivial_res = execute(circ, Aer.get_backend(
            'qasm_simulator'), shots=1).result()
        trivial = trivial_res.data()['dm_0']

    if idle_noise:
        circ, time = add_idle_noise_to_circuit(circ, gate_times=full_gate_times,
                                               T1=T1, T2=T2, return_time=True)

    # Run the circuit
    noise_model = thermal_relaxation_model_V2(
        T1=T1, T2=T2, gate_times=full_gate_times)
    results = execute(circ, Aer.get_backend('qasm_simulator'),
                      noise_model=noise_model, shots=n_shots).result()
    if snapshot_type == 'dm' or snapshot_type == 'density_matrix':
        state = results.data()['dm_0']
        fidelities = state_fidelity(state, trivial)
    elif snapshot_type == 'exp' or snapshot_type == 'expectation_value':
        fidelities = results.data()['exp_0']
    return fidelities, circ


# %% TEST encoding fidelities
fid, circ = encoding_fidelity(1024, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                              reset=True, idle_noise=True, initial_state=0,
                              snapshot_type='dm', device='WACQT', pauliop='ZZZZZ')
print(fid)
# %%
# Settings used across all configurations
n_cycles = 15
n_shots = 1024

# %% Test run
n_cycles = 15
n_shots = 1024

test_times = GateTimes(
    single_qubit_default=0, two_qubit_default=0,
    custom_gate_times={'u1': 0, 'z': 0, 'measure': 0, 'feedback': 2960})

fid2, time = fidelity_from_scratch(n_cycles, n_shots, gate_times=standard_times,
                                   T1=80e3, T2=80e3, reset=True, data_process_type='recovery',
                                   idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
                                   transpile=True, pauliop='ZZZZZ', device_properties=WACQT_device_properties, device=None)
# %%
fid2, time2 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
                                    T1=80e3, T2=80e3, reset=True, data_process_type='recovery',
                                    idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
                                    transpile=False, pauliop='ZZZZZ', device_properties=WACQT_device_properties, device=None)
# %%
# print(fid)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
x_dis = np.arange(-1, n_cycles+2)
x = np.linspace(0, 15, 100)
y = np.exp(theta[0]) * np.exp(theta[1]*x)
ax.plot(x_dis[1:17], fid, '-o', color='C0',
        label='Idle time + instant stabilizers')
ax.plot(x_dis[1:17], fid2, '-o', color='C1', label='Normal QEC')
ax.plot(x, y, color='C2')
#ax.errorbar(x_dis[1:17], fidelities, yerr=errors, color='C1', label='Normal QEC')
ax.set_title(r'QEC of $|0_L\rangle$')
ax.set_xlabel('Number of cycles')
ax.set_ylabel(r'Expectation value of $Z_L$')
ax.legend()
# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Generate x-values
x_target_WACQT = []
x_fast = []
for i in range(len(time)-1):
    x_target_WACQT.append(time['exp_'+str(i)])
    x_fast.append(time_fast['exp_'+str(i)])

ax.plot(x_target_WACQT, fid, '-o', label='Target times')
ax.plot(x_target_WACQT, fid_target_single_1[0], '-k', label='Single times')
ax.plot(x_fast, exp_fast, '-', label='Speedy times')


#x_pred = np.linspace(0, 25,100)
# for cycle in range(n_cycles):
#    y_pred = np.exp(rate[0]) * np.exp(x_pred*rate[1])

# Subplot 1: Target gate times
#ax.plot(x_target_WACQT, fid, 'o', color='C0', label='Hexagonal layout')
#ax.plot(x_pred*1e3, y_pred, '-', color='C0', label='Hexagonal layout')

# %% ========== CONFIGURATIONS FOR RUNS ==========
# EXPECTATION VALUES OF |+> AND |->
exp_target_WACQT_p, time_target_WACQT_p = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
                                                                reset=True, data_process_type='recovery',
                                                                idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=np.pi/2,
                                                                transpile=True, pauliop='XXXXX')
# exp_target_FC_p, time_target_FC_p = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
#    reset=True, data_process_type='recovery',
#    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=np.pi/2,
#    transpile=False, pauliop='XXXXX')
exp_target_DD_p, time_target_DD_p = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
                                                          reset=True, data_process_type='recovery',
                                                          idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=np.pi/2,
                                                          transpile=False, pauliop='XXXXX', device='double_diamond')
print('Check!')
exp_dem_WACQT_p, time_dem_WACQT_p = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_demonstrated_times,
                                                          reset=True, data_process_type='recovery',
                                                          idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=np.pi/2,
                                                          transpile=True, pauliop='XXXXX')
# exp_dem_FC_p, time_dem_FC_p = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_demonstrated_times,
#    reset=True, data_process_type='recovery',
#    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=np.pi/2,
#    transpile=False, pauliop='XXXXX')
exp_dem_DD_p, time_dem_DD_p = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_demonstrated_times,
                                                    reset=True, data_process_type='recovery',
                                                    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=np.pi/2,
                                                    transpile=False, pauliop='XXXXX', device='double_diamond')
print('Check!')

# |->
exp_target_WACQT_m, time_target_WACQT_m = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
                                                                reset=True, data_process_type='recovery',
                                                                idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=-np.pi/2,
                                                                transpile=True, pauliop='XXXXX')
# exp_target_FC_m, time_target_FC_m = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
#    reset=True, data_process_type='recovery',
#    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=-np.pi/2,
#    transpile=False, pauliop='XXXXX')
exp_target_DD_m, time_target_DD_m = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
                                                          reset=True, data_process_type='recovery',
                                                          idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=-np.pi/2,
                                                          transpile=False, pauliop='XXXXX', device='double_diamond')
print('Check!')
exp_dem_WACQT_m, time_dem_WACQT_m = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_demonstrated_times,
                                                          reset=True, data_process_type='recovery',
                                                          idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=-np.pi/2,
                                                          transpile=True, pauliop='XXXXX')
# exp_dem_FC_m, time_dem_FC_m = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_demonstrated_times,
#    reset=True, data_process_type='recovery',
#    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=-np.pi/2,
#    transpile=False, pauliop='XXXXX')
exp_dem_DD_m, time_dem_DD_m = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_demonstrated_times,
                                                    reset=True, data_process_type='recovery',
                                                    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=-np.pi/2,
                                                    transpile=False, pauliop='XXXXX', device='double_diamond')
print('Check!')

# %% EXPECTATION VALUES OF |0> AND |1>
# |0>
exp_target_WACQT_0, time_target_WACQT_0 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
                                                                reset=True, data_process_type='recovery',
                                                                idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
                                                                transpile=True, pauliop='ZZZZZ')
# exp_target_FC_0, time_target_FC_0 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
#    reset=True, data_process_type='recovery',
#    idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
#    transpile=False, pauliop='ZZZZZ')
exp_target_DD_0, time_target_DD_0 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
                                                          reset=True, data_process_type='recovery',
                                                          idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
                                                          transpile=False, pauliop='ZZZZZ', device='double_diamond')
print('Check!')
exp_dem_WACQT_0, time_dem_WACQT_0 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_demonstrated_times,
                                                          reset=True, data_process_type='recovery',
                                                          idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
                                                          transpile=True, pauliop='ZZZZZ')
# exp_dem_FC_0, time_dem_FC_0 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_demonstrated_times,
#    reset=True, data_process_type='recovery',
#    idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
#    transpile=False, pauliop='ZZZZZ')
exp_dem_DD_0, time_dem_DD_0 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_demonstrated_times,
                                                    reset=True, data_process_type='recovery',
                                                    idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
                                                    transpile=False, pauliop='ZZZZZ', device='double_diamond')
print('Check!')
# %%
# |1>
exp_target_WACQT_1, time_target_WACQT_1 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
                                                                reset=True, data_process_type='recovery',
                                                                idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi, phi=0,
                                                                transpile=True, pauliop='ZZZZZ')
# exp_target_FC_1, time_target_FC_1 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
#    reset=True, data_process_type='recovery',
#    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi, phi=0,
#    transpile=False, pauliop='ZZZZZ')
exp_target_DD_1, time_target_DD_1 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
                                                          reset=True, data_process_type='recovery',
                                                          idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi, phi=0,
                                                          transpile=False, pauliop='ZZZZZ', device='double_diamond')
# %%
print('Check!')
exp_dem_WACQT_1, time_dem_WACQT_1 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_demonstrated_times,
                                                          reset=True, data_process_type='recovery',
                                                          idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi, phi=0,
                                                          transpile=True, pauliop='ZZZZZ')
# exp_dem_FC_1, time_dem_FC_1 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_demonstrated_times,
#    reset=True, data_process_type='recovery',
#    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi, phi=0,
#    transpile=False, pauliop='ZZZZZ')
exp_dem_DD_1, time_dem_DD_1 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_demonstrated_times,
                                                    reset=True, data_process_type='recovery',
                                                    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi, phi=0,
                                                    transpile=False, pauliop='ZZZZZ', device='double_diamond')
print('Check!')

# %% FIDELITIES (Not in use)
# Fidelities of |1>
fid_target_WACQT, time_target_WACQT = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
                                                            reset=True, data_process_type='recovery',
                                                            idle_noise=True, snapshot_type='dm', encoding=False, theta=np.pi, phi=0,
                                                            transpile=True, pauliop='XXXXX')
# fid_target_FC, time_target_FC = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
#    reset=True, data_process_type='recovery',
#    idle_noise=True, snapshot_type='dm', encoding=False, theta=np.pi, phi=0,
#    transpile=False, pauliop='XXXXX')
fid_target_DD, time_target_DD = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
                                                      reset=True, data_process_type='recovery',
                                                      idle_noise=True, snapshot_type='dm', encoding=False, theta=np.pi, phi=0,
                                                      transpile=False, pauliop='XXXXX', device='double_diamond')
print('Check!')
fid_dem_WACQT, time_dem_WACQT = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_demonstrated_times,
                                                      reset=True, data_process_type='recovery',
                                                      idle_noise=True, snapshot_type='dm', encoding=False, theta=np.pi, phi=0,
                                                      transpile=True, pauliop='XXXXX')
# fid_dem_FC, time_dem_FC = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_demonstrated_times,
#    reset=True, data_process_type='recovery',
#    idle_noise=True, snapshot_type='dm', encoding=False, theta=np.pi, phi=0,
#    transpile=False, pauliop='XXXXX')
fid_dem_DD, time_dem_DD = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_demonstrated_times,
                                                reset=True, data_process_type='recovery',
                                                idle_noise=True, snapshot_type='dm', encoding=False, theta=np.pi, phi=0,
                                                transpile=False, pauliop='XXXXX', device='double_diamond')
print('Check!')


# %% Post selection
n_shots = 1024*16
# %% + and - (Target times)
exp_target_WACQT_p_PS, count_target_WACQT_p = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
                                                                    reset=True, data_process_type='post_select',
                                                                    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=np.pi/2,
                                                                    transpile=True, pauliop='XXXXX')
print('Check!')
exp_target_WACQT_m_PS, count_target_WACQT_m = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
                                                                    reset=True, data_process_type='post_select',
                                                                    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=-np.pi/2,
                                                                    transpile=True, pauliop='XXXXX')
print('Check!')
exp_target_DD_p_PS, count_target_DD_p = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
                                                              reset=True, data_process_type='post_select',
                                                              idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=np.pi/2,
                                                              transpile=False, pauliop='XXXXX', device='double_diamond')
print('Check!')
exp_target_DD_m_PS, count_target_DD_m = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
                                                              reset=True, data_process_type='post_select',
                                                              idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=-np.pi/2,
                                                              transpile=False, pauliop='XXXXX', device='double_diamond')
# %% + and - (Demonstrated times)
exp_dem_WACQT_p_PS, count_dem_WACQT_p = fidelity_from_scratch(7, n_shots, gate_times=WACQT_demonstrated_times,
                                                              reset=True, data_process_type='post_select',
                                                              idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=np.pi/2,
                                                              transpile=True, pauliop='XXXXX')
exp_dem_WACQT_m_PS, count_dem_WACQT_m = fidelity_from_scratch(7, n_shots, gate_times=WACQT_demonstrated_times,
                                                              reset=True, data_process_type='post_select',
                                                              idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=-np.pi/2,
                                                              transpile=True, pauliop='XXXXX')
exp_dem_DD_p_PS, count_dem_DD_p = fidelity_from_scratch(7, n_shots, gate_times=WACQT_demonstrated_times,
                                                        reset=True, data_process_type='post_select',
                                                        idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=np.pi/2,
                                                        transpile=False, pauliop='XXXXX', device='double_diamond')
exp_dem_DD_m_PS, count_dem_DD_m = fidelity_from_scratch(7, n_shots, gate_times=WACQT_demonstrated_times,
                                                        reset=True, data_process_type='post_select',
                                                        idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi/2, phi=-np.pi/2,
                                                        transpile=False, pauliop='XXXXX', device='double_diamond')

# %% 0 and 1 PS
n_shots = 1024*16

exp_target_WACQT_0_PS, count_target_WACQT_0 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
                                                                    reset=True, data_process_type='post_select',
                                                                    idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
                                                                    transpile=True, pauliop='ZZZZZ')
print('Check!')
exp_target_WACQT_1_PS, count_target_WACQT_1 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
                                                                    reset=True, data_process_type='post_select',
                                                                    idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi, phi=-0,
                                                                    transpile=True, pauliop='ZZZZZ')
print('Check!')
exp_target_DD_0_PS, count_target_DD_0 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
                                                              reset=True, data_process_type='post_select',
                                                              idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
                                                              transpile=False, pauliop='ZZZZZ', device='double_diamond')
print('Check!')
exp_target_DD_1_PS, count_target_DD_1 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
                                                              reset=True, data_process_type='post_select',
                                                              idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi, phi=-0,
                                                              transpile=False, pauliop='ZZZZZ', device='double_diamond')
print('Check!')
# %% 0 and 1 (Demonstrated times)
exp_dem_WACQT_0_PS, count_dem_WACQT_0 = fidelity_from_scratch(7, n_shots, gate_times=WACQT_demonstrated_times,
                                                              reset=True, data_process_type='post_select',
                                                              idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
                                                              transpile=True, pauliop='ZZZZZ')
print('Check!')
exp_dem_WACQT_1_PS, count_dem_WACQT_1 = fidelity_from_scratch(7, n_shots, gate_times=WACQT_demonstrated_times,
                                                              reset=True, data_process_type='post_select',
                                                              idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi, phi=-0,
                                                              transpile=True, pauliop='ZZZZZ')
print('Check!')
exp_dem_DD_0_PS, count_dem_DD_0 = fidelity_from_scratch(7, n_shots, gate_times=WACQT_demonstrated_times,
                                                        reset=True, data_process_type='post_select',
                                                        idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
                                                        transpile=False, pauliop='ZZZZZ', device='double_diamond')
print('Check!')
exp_dem_DD_1_PS, count_dem_DD_1 = fidelity_from_scratch(7, n_shots, gate_times=WACQT_demonstrated_times,
                                                        reset=True, data_process_type='post_select',
                                                        idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi, phi=-0,
                                                        transpile=False, pauliop='ZZZZZ', device='double_diamond')

# %% EXPECTATION VALUES OF |0> AND |1>, NO QEC
n_shots = 1024*4
n_cycles = 15
# |0>
exp_target_WACQT_0_decay = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
                                                 reset=True, data_process_type='empty_circuit',
                                                 idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
                                                 transpile=True, pauliop='ZZZZZ')

exp_target_DD_0_decay = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
                                              reset=True, data_process_type='empty_circuit',
                                              idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
                                              transpile=False, pauliop='ZZZZZ', device='double_diamond')
print('Check!')

# |1>
exp_target_WACQT_1_decay = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
                                                 reset=True, data_process_type='empty_circuit',
                                                 idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi, phi=0,
                                                 transpile=True, pauliop='ZZZZZ')
exp_target_DD_1_decay = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
                                              reset=True, data_process_type='empty_circuit',
                                              idle_noise=True, snapshot_type='exp', encoding=False, theta=np.pi, phi=0,
                                              transpile=False, pauliop='ZZZZZ', device='double_diamond')
print('Check!')

# %% Testing single qubit
n_cycles = 15
n_shots = 1024*4
# %%
fid_target_single, time_single = fid_single_qubit(n_cycles, n_shots,
                                                  gate_times=WACQT_target_times,
                                                  snapshot_type='exp', T1=40e3, T2=10e3,
                                                  theta=np.pi/2, phi=np.pi/2, pauliop='X')
# %%
exp_target_single_p = fid_single_qubit(n_cycles, n_shots,
                                       gate_times=WACQT_target_times,
                                       snapshot_type='exp', T1=40e3, T2=60e3,
                                       theta=np.pi/2, phi=np.pi/2, pauliop='X')

exp_dem_single_p = fid_single_qubit(n_cycles, n_shots,
                                    gate_times=WACQT_demonstrated_times,
                                    snapshot_type='exp', T1=40e3, T2=60e3,
                                    theta=np.pi/2, phi=np.pi/2, pauliop='X')

exp_target_single_m = fid_single_qubit(n_cycles, n_shots,
                                       gate_times=WACQT_target_times,
                                       snapshot_type='exp', T1=40e3, T2=60e3,
                                       theta=np.pi/2, phi=-np.pi/2, pauliop='X')

exp_dem_single_m = fid_single_qubit(n_cycles, n_shots,
                                    gate_times=WACQT_demonstrated_times,
                                    snapshot_type='exp', T1=40e3, T2=60e3,
                                    theta=np.pi/2, phi=-np.pi/2, pauliop='X')
# %%
fid_target_single_1 = fid_single_qubit(n_cycles, n_shots,
                                       gate_times=WACQT_target_times,
                                       snapshot_type='dm', T1=40e3, T2=60e3,
                                       theta=np.pi, phi=0, pauliop='Z')
# %%
fid_dem_single_1 = fid_single_qubit(n_cycles, n_shots,
                                    gate_times=WACQT_demonstrated_times,
                                    snapshot_type='dm', T1=40e3, T2=60e3,
                                    theta=np.pi, phi=0, pauliop='Z')
# %%
fid_target_single_p = fid_single_qubit(n_cycles, n_shots,
                                       gate_times=WACQT_target_times,
                                       snapshot_type='dm', T1=40e3, T2=60e3,
                                       theta=np.pi, phi=np.pi/2, pauliop='Z')
fid_dem_single_p = fid_single_qubit(n_cycles, n_shots,
                                    gate_times=WACQT_demonstrated_times,
                                    snapshot_type='dm', T1=40e3, T2=60e3,
                                    theta=np.pi, phi=np.pi/2, pauliop='Z')
# %% Fast test runs
gate_times = GateTimes(10, 50, {'u1': 0, 'z': 0, 'measure': 100})
exp_fast, time_fast = fidelity_from_scratch(n_cycles, n_shots, gate_times=gate_times,
                                            T1=80e3, T2=80e3, reset=True, data_process_type='recovery',
                                            idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
                                            transpile=False, pauliop='ZZZZZ')


# %% Test plotting
# %% Plotting
fig, ax = plt.subplots(2, 1, figsize=(10, 16))
x_dis = np.arange(0, n_cycles+1)

# Generate x-values
x_target_WACQT = []
x_target_DD = []

x_dem_WACQT = []
x_dem_DD = []

for i in range(len(time_target_WACQT_1)-1):
    x_target_WACQT.append(time_target_WACQT_1['exp_'+str(i)])
    x_target_DD.append(time_target_DD_1['exp_'+str(i)])
    # x_dem_WACQT.append(time_dem_WACQT_1['exp_'+str(i)])
    # x_dem_DD.append(time_dem_DD_1['exp_'+str(i)])

# Subplot 1: Target gate times
ax[0].plot(x_target_WACQT, exp_target_WACQT_0, '-o',
           color='C0', label='Hexagonal layout')
ax[0].plot(x_target_DD, exp_target_DD_0, '-o',
           color='C1', label='Double diamond layout')
#ax[0].plot(x_target_WACQT, exp_target_FC_0, '-o', color='C2', label='Full connectivity')
ax[0].plot(x_target_WACQT, exp_target_WACQT_0_PS,
           '-o', color='C2', label='Hexagonal, PS')
ax[0].plot(x_target_DD, exp_target_DD_0_PS, '-o',
           color='C3', label='Double diamond, PS')
ax[0].plot(x_target_WACQT, exp_target_WACQT_1, '-o', color='C0')
ax[0].plot(x_target_DD, exp_target_DD_1, '-o', color='C1')
#ax[0].plot(x_target_WACQT, exp_target_FC_1, '-o', color='C2')
ax[0].plot(x_target_WACQT, exp_target_WACQT_1_PS, '-o', color='C2')
ax[0].plot(x_target_DD, exp_target_DD_1_PS, '-o', color='C3')

ax[0].plot(x_target_WACQT[0:16], exp_target_WACQT_0_decay, '-o',
           color='C4', label='Decay of logical state (Hexagonal)')
ax[0].plot(x_target_WACQT[0:16], exp_target_WACQT_1_decay, '-o', color='C4')
ax[0].plot(x_target_WACQT[0:16], exp_target_DD_0_decay, '-o',
           color='C5', label='Decay of logical state (DD)')
ax[0].plot(x_target_WACQT[0:16], exp_target_DD_1_decay, '-o', color='C5')

ax[0].set(ylim=(-1.0, 1.0))
ax[0].plot(x_target_WACQT, fid_target_single_1, '--k', label='Single qubit')
ax[0].ticklabel_format(style='sci', scilimits=(0, 0), useMathText=True)
ax[0].set_title(
    r'Expectation value of $|0\rangle$ and $|1\rangle$ using target gate times')
ax[0].set_xlabel('Time [ns]')
ax[0].set_ylabel(r'Expectation value of $Z_L$')
ax[0].legend()

ax0 = ax[0].twinx()
ax0.plot(x_target_WACQT, -1*np.array(fid_target_single_1), '--k')
ax0.set(ylim=(-1.0, 1.0))
ax0.set_yticklabels(["1.0", "0.75", "0.5", "0.25",
                     "0.0", "0.25", "0.5", "0.75", "1.0"])
ax0.set_ylabel(r'Physical $|1\rangle$ probability')
# %%

ax[1].plot(x_dem_WACQT, exp_dem_WACQT_0, '-o',
           color='C0', label='Hexagonal layout')
ax[1].plot(x_dem_DD, exp_dem_DD_0, '-o', color='C1',
           label='Double diamond layout')
#ax[1].plot(x_dem_WACQT, exp_dem_FC_0, '-o', color='C2', label='Full connectivity')
#ax[1].plot(x_dem_WACQT[0:8], exp_dem_WACQT_0_PS, '-o', color='C2', label='Hexagonal, PS')
#ax[1].plot(x_dem_DD[0:8], exp_dem_DD_0_PS, '-o', color='C2', label='Double diamond, PS')

ax[1].plot(x_dem_WACQT, exp_dem_WACQT_1, '-o', color='C0')
ax[1].plot(x_dem_DD, exp_dem_DD_1, '-o', color='C1')
#ax[1].plot(x_dem_WACQT, exp_dem_FC_1, '-o', color='C2', label='Full connectivity')
#ax[1].plot(x_dem_WACQT[0:8], exp_dem_WACQT_1_PS, '-o', color='C2', label='Hexagonal, PS')
#ax[1].plot(x_dem_DD[0:8], exp_dem_DD_1_PS, '-o', color='C2', label='Double diamond, PS')
#ax[1].plot(x_dem_WACQT, exp_dem_single_1, '--k', label='Single qubit')
ax[1].set_title(
    r'Expectation value of $|0\rangle$ and $|1\rangle$ using demonstrated gate times')
ax[1].set(ylim=(-1.0, 1.0))
ax[1].ticklabel_format(style='sci', scilimits=(0, 0), useMathText=True)
ax[1].set_xlabel('Time [ns]')
ax[1].set_ylabel(r'Expectation value of $Z_L$')
ax[1].legend()

ax1 = ax[1].twinx()
ax[1].plot(x_dem_WACQT, fid_dem_single_1, '--k', label='Single qubit')
ax1.plot(x_dem_WACQT, -1*np.array(fid_dem_single_1), '--k')
ax1.set(ylim=(-1.0, 1.0))
ax1.set_yticklabels(["1.0", "0.75", "0.5", "0.25",
                     "0.0", "0.25", "0.5", "0.75", "1.0"])
ax1.set_ylabel(r'Physical $|1\rangle$ probability')

# %% Testing different transpilations (Not used in other parts of the script)
reset = True
recovery = False
n_cycles = 2
snapshot_type = 'dm'

# Registers
qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
cr = get_classical_register(n_cycles, reset=reset,
                            recovery=recovery, flag=False)
readout = ClassicalRegister(5, 'readout')
registers = StabilizerRegisters(qb, an, cr, readout)

circ = get_full_stabilizer_circuit(registers, n_cycles=n_cycles, reset=reset,
                                   recovery=recovery, flag=False,
                                   snapshot_type=snapshot_type,
                                   initial_state=0,
                                   encoding=True)


# WACQT 7 qb UPDATED BASIS
basis_gates = ['id', 'u1', 'x', 'y', 'z', 'sx', 'sy', 'cz', 'h',
               'save_expval', 'save_density_matrix']

#couplinglist = [[0, 2], [0, 3], [1, 3], [1, 4], [2, 5], [3, 5], [3, 6], [4, 6]]
couplinglist = [[0, 1], [0, 6], [1, 6], [2, 3],
                [2, 6], [3, 6], [4, 5], [4, 6], [5, 6]]

reverse_couplinglist = [[y, x] for [x, y] in couplinglist]

coupling_map = CouplingMap(
    couplinglist=couplinglist+reverse_couplinglist,
    description='A hexagonal 7qb code with two ancillas')
device_properties = {
    "basis_gates": basis_gates, "coupling_map": coupling_map}


circ = shortest_transpile_from_distribution(circ, print_cost=False,
                                            **device_properties)

#time = get_circuit_time(circ, gate_times=WACQT_target_times)
# print(time)
#circ, times = add_idle_noise_to_circuit(circ, WACQT_target_times, return_time=True, rename=True)
# circ.draw(output='mpl')
# %%
n_cycles = 1
n_shots = 1024*8

fid3, time3, circ3, res3 = fidelity_from_scratch(n_cycles, n_shots, gate_times=WACQT_target_times,
                                                 T1=80e3, T2=80e3, reset=True, data_process_type='recovery',
                                                 idle_noise=True, snapshot_type='exp', encoding=False, theta=0, phi=0,
                                                 transpile=False, pauliop='ZZZZZ', device_properties=WACQT_device_properties, device=None)

# %% Check the counts of 0 and 1 in each syndrome bit
counts = results.get_counts()
print(len(counts))

red_counts = {'0000': 0, '0001': 0, '0010': 0, '0011': 0, '0100': 0, '0101': 0, '0110': 0,
              '0111': 0, '1000': 0, '1001': 0, '1010': 0, '1011': 0, '1100': 0, '1101': 0,
              '1110': 0, '1111': 0}
for key in counts:
    red_counts[key[6:10]] += counts[key]
print(red_counts)

syn0 = [0, 0, 0, 0]  # Number of 0s in each bit
syn1 = [0, 0, 0, 0]  # Number of 1s in each bit
for key in red_counts:
    for i in range(len(syn0)):
        if key[i] == '0':
            syn0[i] += red_counts[key]
        elif key[i] == '1':
            syn1[i] += red_counts[key]

# %% Running one cycle of idle noise followed by perfect stabilizers

T1 = 80e3
T2 = 80e3
reset = True
recovery = True
conditional = False
snapshot_type = 'exp'
encoding = False
theta = 0
phi = 0
pauliop = 'ZZZZZ'
device = None
gate_times = test_times
n_cycles = 1
n_shots = 8192
if isinstance(gate_times, dict):
    full_gate_times = WACQT_gate_times.get_gate_times(
        custom_gate_times=gate_times)
elif isinstance(gate_times, GateTimes):
    full_gate_times = gate_times
else:
    warnings.warn('Invalid gate times, assuming WACQT_gate_times')
    full_gate_times = WACQT_gate_times

# Noise model
noise_model = thermal_relaxation_model_V2(
    T1=T1, T2=T2, gate_times=full_gate_times)

# Registers
qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
cr = get_classical_register(n_cycles, reset=reset,
                            recovery=recovery, flag=False)
readout = ClassicalRegister(5, 'readout')
registers = StabilizerRegisters(qb, an, cr, readout)

# Circuits
# circ = get_full_stabilizer_circuit(registers, n_cycles=n_cycles, reset=reset,
#                                   recovery=recovery, flag=False,
#                                   snapshot_type=snapshot_type,
#                                   conditional=conditional,
#                                   encoding=encoding, theta=theta, phi=phi,
#                                   pauliop=pauliop, device=device)


circ = get_empty_stabilizer_circuit(registers)
circ.set_density_matrix(get_encoded_state(theta=theta, phi=phi))
thrm_relax = thermal_relaxation_error(
    T1, T2, full_gate_times['feedback']).to_instruction()
for reg in circ.qubits:
    circ.append(thrm_relax, [reg])
circ.compose(get_repeated_stabilization(registers, n_cycles=n_cycles,
                                        reset=reset, recovery=recovery, flag=False,
                                        snapshot_type=snapshot_type,
                                        conditional=conditional,
                                        include_barriers=True,
                                        pauliop=pauliop, device=device,
                                        ), inplace=True)
circ.measure(qb, readout)

# Run the circuit
results = execute(circ, Aer.get_backend('qasm_simulator'),
                  noise_model=noise_model, shots=n_shots).result()


# %% Alternate version with idle noise after the snapshot of each cycle
test_times = GateTimes(
    single_qubit_default=0, two_qubit_default=0,
    custom_gate_times={'u1': 0, 'z': 0, 'measure': 0, 'feedback': 2960})
T1 = 80e3
T2 = 80e3
reset = True
recovery = True
conditional = False
snapshot_type = 'exp'
encoding = False
theta = 0
phi = 0
pauliop = 'ZZZZZ'
device = None
include_barriers = True
gate_times = test_times
n_cycles = 15
n_shots = 1024
if isinstance(gate_times, dict):
    full_gate_times = WACQT_gate_times.get_gate_times(
        custom_gate_times=gate_times)
elif isinstance(gate_times, GateTimes):
    full_gate_times = gate_times
else:
    warnings.warn('Invalid gate times, assuming WACQT_gate_times')
    full_gate_times = WACQT_gate_times

# Noise model
noise_model = thermal_relaxation_model_V2(
    T1=T1, T2=T2, gate_times=full_gate_times)

# Registers
qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
cr = get_classical_register(n_cycles, reset=reset,
                            recovery=recovery, flag=False)
readout = ClassicalRegister(5, 'readout')
registers = StabilizerRegisters(qb, an, cr, readout)

circ = get_empty_stabilizer_circuit(registers)
circ.set_density_matrix(get_encoded_state(theta=theta, phi=phi))
add_snapshot_to_circuit(circ, snapshot_type=snapshot_type, current_cycle=0, qubits=qb,
                        conditional=conditional, pauliop=pauliop,
                        include_barriers=include_barriers)
thrm_relax = thermal_relaxation_error(
    T1, T2, full_gate_times['feedback']).to_instruction()
for reg in circ.qubits:
    circ.append(thrm_relax, [reg])

for current_cycle in range(n_cycles):
    circ.compose(get_stabilizer_cycle(registers, reset=reset, recovery=recovery,
                                            current_cycle=current_cycle, current_step=0,
                                            include_barriers=include_barriers), inplace=True)
    add_snapshot_to_circuit(circ, snapshot_type=snapshot_type, current_cycle=current_cycle+1,
                            qubits=qb, conditional=conditional, pauliop=pauliop,
                            include_barriers=include_barriers)
    for reg in circ.qubits:
        circ.append(thrm_relax, [reg])
circ.measure(qb, readout)


# Run the circuit
results = execute(circ, Aer.get_backend('qasm_simulator'),
                  noise_model=noise_model, shots=n_shots).result()

fidelities = []

for current_cycle in range(n_cycles+1):
    fidelities.append(results.data()['exp_' + str(current_cycle)])
