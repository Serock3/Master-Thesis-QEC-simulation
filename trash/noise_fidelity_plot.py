# %% Import modules
from qiskit.aqua.utils import get_subsystems_counts
import matplotlib.pyplot as plt
import numpy as np
from qiskit import *

from qiskit.quantum_info import state_fidelity

# Our own files
from simulator_program.custom_noise_models import thermal_relaxation_model_V2
from simulator_program.custom_transpiler import *
from simulator_program.stabilizers import *

# %% =================  Testing noise model + stabilizer ======================
# DEFINE AND TRANSPILE THE CIRCUIT

# Define our registers
qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
cr = ClassicalRegister(5, 'syndrome_bit')  # The typical register
readout = ClassicalRegister(5, 'readout')

registers = StabilizerRegisters(qb, an, cr, readout)
circ = get_empty_stabilizer_circuit(registers)

# Settings for circuit
n_cycles = 1
reset = False
flag = False
recovery = False

# Get the circuit
# circ.x(qb[0])
# circ += get_full_stabilizer_circuit(registers,
#     n_cycles=n_cycles,
#     reset=reset,
#     recovery=recovery,
#     flag=flag,
# )
circ += encode_input_v2(registers)
circ += get_stabilizer_cycle(registers,
                                   reset=reset,
                                   recovery=recovery
                                   )
circ.barrier()
circ.append(Snapshot('stabilizer_0', 'density_matrix', num_qubits=5), qb)

# Transpilation
routing_method = 'sabre'  # basic lookahead stochastic sabre
initial_layout = None  # Overwriting the above layout
layout_method = 'sabre'  # trivial 'dense', 'noise_adaptive' sabre
translation_method = None  # 'unroller',  translator , synthesis
optimization_levels = [0, 1, 2]
repeats = 10
circuit_list = [circ]
for optimization_level in optimization_levels:
    circ_t = shortest_transpile_from_distribution(
        circ,
        print_cost=False,
        repeats=repeats,
        routing_method=routing_method,
        initial_layout=initial_layout,
        layout_method=layout_method,
        translation_method=translation_method,
        optimization_level=0,
        **WACQT_device_properties
    )
    circuit_list.append(circ_t)


# Error free run for fidelity measurements
# correct_state = []
# for circuit in circuit_list:
#     results = execute(
#         circuit,
#         Aer.get_backend('qasm_simulator'),
#         noise_model=None,
#         shots=1,
#     ).result()

#     correct_state.append(
#         results.data()['snapshots']['statevector']['stabilizer_0'][0]
#     )

# %% Functions for analyzing results
def select_no_errors(mem):
    '''Gives array of booleans corresponding to the shots without errors'''
    return np.array([int(item[6:10]) for item in mem]) == 0


def get_fidelity_data(circ, param_list, n_shots=2048,
                      post_select=True):
    '''Inputs:
    circ: The circuit to be tested
    correct_state: The correct state for comparison
    param_list: The error model parameters, currently only [T2, t_cz]
    n_shots: Number of shots to average over
    '''
    T2, t_cz = param_list

    # Get correct state
    results = execute(
        circ,
        Aer.get_backend('qasm_simulator'),
        noise_model=None,
        shots=1,
    ).result()

    # TODO: Make this work if circuit it permuted for later stabilizers
    correct_state = results.data(
    )['snapshots']['statevector']['stabilizer_0'][0]

    # Run the circuit
    results = execute(
        circ,
        Aer.get_backend('qasm_simulator'),
        noise_model=thermal_relaxation_model_V2(T2=T2, t_cz=t_cz),
        memory=True,
        shots=n_shots
    ).result()

    select_indices = select_no_errors(results.get_memory())
    select_count = np.sum(select_indices)
    select_fraction = select_count/n_shots
    data = np.zeros(n_shots)

    if post_select:
        # Post-selection
        post_selection = np.array(
            results.data()['snapshots']['statevector']['stabilizer_0']
        )[select_indices]

        # Analyze results
        for j in range(select_count):
            statevector = post_selection[j]
            data[j] = state_fidelity(statevector, correct_state)
        fid = np.sum(data)/select_count
    else:
        for j in range(n_shots):
            statevector = results.data(
            )['snapshots']['statevector']['stabilizer_0'][j]
            data[j] = state_fidelity(statevector, correct_state)
        fid = np.sum(data)/select_count
    return fid, select_fraction, data


def reformat_density_snapshot(results) -> dict:
    """Reformats the snapshot data of the results object to be a 
    dictionary with the measurement results as keys
    """
    snap_dict = {}
    for snapshot_name in results.data()['snapshots']['density_matrix']:
        res_dict = {}
        for item in results.data()[
                'snapshots']['density_matrix'][snapshot_name]:
            res_dict[item['memory']] = item['value']
        snap_dict[snapshot_name] = res_dict
    return snap_dict


def get_post_select_fraction_for_density_matrix(results, n_shots):
    syndrome_reg_counts = get_subsystems_counts(results.get_counts())[1]
    count_trivial_syndrome = 0
    for key in syndrome_reg_counts:
        if int(key) == 0:
            count_trivial_syndrome = syndrome_reg_counts[key]
    return count_trivial_syndrome/n_shots


def get_fidelity_data_den_mat(circ, param_list, n_shots=2048,
                              post_select=True):
    '''Inputs:
    circ: The circuit to be tested
    correct_state: The correct state for comparison
    param_list: The error model parameters, currently only [T2, t_cz]
    n_shots: Number of shots to average over
    '''
    T2, t_cz = param_list

    # Get correct state
    results = execute(
        circ,
        Aer.get_backend('qasm_simulator'),
        noise_model=None,
        shots=1,
    ).result()

    snapshots = reformat_density_snapshot(results)
    # TODO: Make this work if circuit it permuted for later stabilizers
    # TODO: More sophisticated key than '0x0'?
    correct_state = snapshots['stabilizer_0']['0x0']

    # Run the circuit
    results = execute(
        circ,
        Aer.get_backend('qasm_simulator'),
        noise_model=thermal_relaxation_model_V2(T2=T2, t_cz=t_cz),
        memory=True,
        shots=n_shots
    ).result()

    # TODO: make post_select=False possible
    # Post-selection
    select_fraction = get_post_select_fraction_for_density_matrix(
        results, n_shots)
    snapshots = reformat_density_snapshot(results)
    post_selection = snapshots['stabilizer_0']['0x0']

    # Analyze results
    fid = state_fidelity(post_selection, correct_state)

    return fid, select_fraction


# %% Analyze results
T2_list = np.arange(40, 81, 20)*1e3  # 40-80 mus
t_cz_list = np.arange(100, 301, 100)  # 100-300 ns
n_shots = 10  # 1024*4

fid_T2 = np.zeros(len(T2_list))
fid_t = np.zeros(len(t_cz_list))
T2_results_list = []
t_cz_results_list = []
P_T2_list = []
P_t_list = []
for index in range(len(circuit_list)):  # Loop over circuits

    # Post selection, vary T2
    t_cz = 200
    fid_T2 = np.zeros(len(T2_list))
    P_T2 = np.zeros(len(T2_list))
    for i in range(len(T2_list)):
        param_list = [T2_list[i], t_cz]
        # fid_T2[i], _, _ = get_fidelity_data(
        #     circ=circuit_list[index],
        #     # correct_state=correct_state[index],
        #     param_list=param_list,
        #     n_shots=1024,
        # )
        fid_T2[i], P_T2[i] = get_fidelity_data_den_mat(
            circ=circuit_list[index],
            # correct_state=correct_state[index],
            param_list=param_list,
            n_shots=1024,
        )
    T2_results_list.append(fid_T2)
    P_T2_list.append(P_T2)

    # Post selection, vary t_cz
    T2 = 60e3
    fid_t = np.zeros(len(t_cz_list))
    P_t = np.zeros(len(t_cz_list))
    for i in range(len(t_cz_list)):
        param_list = [T2, t_cz_list[i]]
        # fid_t[i], _, _ = get_fidelity_data(
        #     circ=circuit_list[index],
        #     # correct_state=correct_state[index],
        #     param_list=param_list,
        #     n_shots=1024,
        # )
        fid_t[i], P_t[i] = get_fidelity_data_den_mat(
            circ=circuit_list[index],
            # correct_state=correct_state[index],
            param_list=param_list,
            n_shots=1024,
        )
    t_cz_results_list.append(fid_t)
    P_t_list.append(P_t)
    print('boop')
# %% Plotting
fig, axs = plt.subplots(4, figsize=(14, 16))
ax1, ax2, ax3, ax4 = axs
for i in range(4):
    ax1.plot(T2_list*1e-3, T2_results_list[i], 'o-', label=str(i-1))
    ax2.plot(t_cz_list, t_cz_results_list[i], 'o-', label=str(i-1))
    ax3.plot(T2_list*1e-3, P_T2_list[i], 'o-', label=str(i-1))
    ax4.plot(t_cz_list, P_t_list[i], 'o-', label=str(i-1))
ax1.set_xlabel('T2 [$\mu$s]')
ax1.set_ylabel('Average fidelity')
ax1.set_title('Fidelity with varying T2, constant 2-qb gate time (200 ns)')
ax1.legend()
ax1.grid(linewidth=1)

ax2.set_xlabel('2-qb gate time [ns]')
ax2.set_ylabel('Average fidelity')
ax2.set_title('Fidelity with varying 2-qb gate time, constant T2 (60 $\mu$s)')
ax2.legend()
ax2.grid(linewidth=1)

ax3.set_xlabel('T2 [$\mu$s]')
ax3.set_ylabel('Selection fraction')
ax3.set_title(
    'Selection fraction with varying T2, constant 2-qb gate time (200 ns)')
ax3.legend()
ax3.grid(linewidth=1)

ax4.set_xlabel('2-qb gate time [ns]')
ax4.set_ylabel('Selection fraction')
ax4.set_title(
    'Selection fraction with varying 2-qb gate time, constant T2 (60 $\mu$s)')
ax4.legend()
ax4.grid(linewidth=1)


# %% Loading data files
T2_data = np.load('data/T2_data.npy')
T2_data_t0 = np.load('data/T2_data_t0.npy')
T2_data_t1 = np.load('data/T2_data_t1.npy')
T2_data_t2 = np.load('data/T2_data_t2.npy')
t_cz_data = np.load('data/t_cz_data.npy')
t_cz_data_t0 = np.load('data/t_cz_data_t0.npy')
t_cz_data_t1 = np.load('data/t_cz_data_t1.npy')
t_cz_data_t2 = np.load('data/t_cz_data_t2.npy')


# %% Plotting
fig, axs = plt.subplots(2, figsize=(14, 10))
ax1 = axs[0]
ax2 = axs[1]

# Vary T2
ax1.plot(T2_list*1e-3, fid_T2, 'o-', label='No transpilation')
ax1.plot(T2_list*1e-3, fid_T2_t0, 'o-', label='Transpiled 0')
ax1.plot(T2_list*1e-3, fid_T2_t1, 'o-', label='Transpiled 1')
#ax1.plot(T2_list*1e-3, fid_T2_t2, 'o-', label='Transpiled 2')
ax1.set_xlabel('T2 [$\mu$s]')
ax1.set_ylabel('Average fidelity')
ax1.set_title('Fidelity with varying T2, constant 2-qb gate time (200 ns)')
#ax1.set(ylim=(0.74, 0.96))
ax1.legend()
ax1.grid(linewidth=1)

# Vary t_cz
ax2.plot(t_cz_list, fid_cz, 'o-', label='No transpilation')
ax2.plot(t_cz_list, fid_cz_t0, 'o-', label='Transpiled 0')
ax2.plot(t_cz_list, fid_cz_t1, 'o-', label='Transpiled 1')
#ax2.plot(t_cz_list, fid_cz_t2, 'o-', label='Transpiled 2')
ax2.set_xlabel('2-qb gate time [ns]')
ax2.set_ylabel('Average fidelity')
ax2.set_title('Fidelity with varying 2-qb gate time, constant T2 (60 $\mu$s)')
#ax2.set(ylim=(0.74, 0.96))
ax2.legend()
ax2.grid(linewidth=1)
# %% Running fidelity (multiple cycles)
reset = False
recovery = True
flag = False
n_cycles = 15
qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
# cr = ClassicalRegister(4, 'syndrome_bit') # The typical register
cr = get_classical_register(n_cycles, reset=reset,
                            recovery=recovery, flag=False)
readout = ClassicalRegister(5, 'readout')

registers = StabilizerRegisters(qb, an, cr, readout)


# circ = get_empty_stabilizer_circuit(registers)

circ = encode_input_v2(registers)
circ.snapshot('post_encoding', 'density_matrix')
# Stabilizer
circ += get_repeated_stabilization(registers, n_cycles=n_cycles,
                                   reset=reset, recovery=recovery, flag=flag, snapshot_type='density_matrix')

# Final readout
# circ.measure(qbReg, readout)
# circ.snapshot('post_measure', 'density_matrix')


def get_running_post_select_fraction_for_density_matrix(results, n_shots, cycle):
    subsys_counts = get_subsystems_counts(results.get_counts())
    syndrome_reg_counts = subsys_counts[len(subsys_counts)-1-cycle]
    count_trivial_syndrome = 0
    for key in syndrome_reg_counts:
        if int(key) == 0:
            count_trivial_syndrome = syndrome_reg_counts[key]
    return count_trivial_syndrome/n_shots


def get_running_post_select_fraction_for_density_matrix_v2(results, n_shots, cycle):
    """Counts occurances of trivial measurements from first cycles to current cycle (given by 'cycle')

    Args:
        results (Result): result of QASM simulation
        n_shots (int): 
        cycle (int): current cycle

    Returns:
        float: [description]
    """
    split_point = len(list(results.get_counts().keys())[0].split())-1-cycle

    count_trivial_syndrome = 0
    for outcome in results.get_counts():
        formated_outcome = [int(key) for key in outcome.split()[split_point:]]
        if not any(formated_outcome):
            count_trivial_syndrome += results.get_counts()[outcome]
    return count_trivial_syndrome/n_shots


def get_running_fidelity_data_den_mat(circ, n_shots=2048,
                                      noise_model=thermal_relaxation_model_V2(), post_select=True):
    '''Inputs:
    circ: The circuit to be tested
    correct_state: The correct state for comparison
    param_list: The error model parameters, currently only [T2, t_cz]
    n_shots: Number of shots to average over
    '''

    # Get correct state
    results = execute(
        circ,
        Aer.get_backend('qasm_simulator'),
        noise_model=None,
        shots=1,
    ).result()

    snapshots = reformat_density_snapshot(results)
    # TODO: Make this work if circuit it permuted for later stabilizers
    # TODO: More sophisticated key than '0x0'?
    correct_state = snapshots['stabilizer_0'][[
        key for key in snapshots['stabilizer_0'] if int(key, 16) == 0][0]]

    # Run the circuit
    results = execute(
        circ,
        Aer.get_backend('qasm_simulator'),
        noise_model=noise_model,
        shots=n_shots
    ).result()

    # Post-selection
    fidelities = []
    snapshots = reformat_density_snapshot(results)
    select_fractions = []
    if post_select:
        for current_cycle in range(n_cycles):
            try:
                post_selection = snapshots['stabilizer_' + str(current_cycle)][[
                    key for key in snapshots['stabilizer_0'] if int(key, 16) == 0][0]]
                select_fraction = get_running_post_select_fraction_for_density_matrix_v2(
                    results, n_shots, current_cycle)
                select_fractions.append(select_fraction)
                fidelities.append(state_fidelity(
                    post_selection, correct_state))
            except:
                print("No selectable states")
                fidelities.append(-1)
                select_fractions.append(0)
        return fidelities, select_fractions

    else:
        cl_reg_size = len(list(results.get_counts().keys())[0].split()[1])
        counts = results.get_counts()
        for current_cycle in range(n_cycles):
            fid = 0
            for key in snapshots['stabilizer_'+str(current_cycle)]:
                bin_string = bin(int(key, 16))[2:].zfill(
                    cl_reg_size*(current_cycle+1))
                current_state = snapshots['stabilizer_' +
                                          str(current_cycle)][key]
                for outcome in results.get_counts():
                    formated_outcome = outcome.replace(
                        ' ', '')[-cl_reg_size*(current_cycle+1):]
                    if formated_outcome == bin_string:
                        fid += state_fidelity(current_state,
                                              correct_state)*counts[outcome]
            fidelities.append(fid/n_shots)
        return fidelities


# %% Transpilation for WACQT and diamond
routing_method = 'sabre'  # basic lookahead stochastic sabre
initial_layout = None  # Overwriting the above layout
layout_method = 'sabre'  # trivial 'dense', 'noise_adaptive' sabre
translation_method = None  # 'unroller',  translator , synthesis

repeats = 10

circ_WACQT = shortest_transpile_from_distribution(
    circ,
    print_cost=True,
    repeats=repeats,
    routing_method=routing_method,
    initial_layout=initial_layout,
    layout_method=layout_method,
    translation_method=translation_method,
    optimization_level=1,
    **WACQT_device_properties
)

print("transpilation done for circ_WACQT, dept")
circ_diamond = shortest_transpile_from_distribution(
    circ,
    print_cost=True,
    repeats=repeats,
    routing_method=routing_method,
    initial_layout=initial_layout,
    layout_method=layout_method,
    translation_method=translation_method,
    optimization_level=1,
    **diamond_device_properties
)
print("transpilation done for circ_diamond, dept")

print("circ_WACQT.depth()", circ_WACQT.depth())
print("circ_WACQT.count_ops()", circ_WACQT.count_ops())
print("circ_diamond.depth()", circ_diamond.depth())
print("circ_diamond.count_ops()", circ_diamond.count_ops())

# %% Run
n_shots = 1024*2
fidelities, select_fractions = get_running_fidelity_data_den_mat(circ, n_shots)
print(select_fractions)
fidelities_WACQT, select_fractions_WACQT = get_running_fidelity_data_den_mat(
    circ_WACQT, n_shots)
print(select_fractions_WACQT)
fidelities_diamond, select_fractions_diamond = get_running_fidelity_data_den_mat(
    circ_diamond, n_shots)
print(select_fractions_diamond)
# %% Plotting
fig, axs = plt.subplots(2, figsize=(14, 10))
ax1 = axs[0]
ax2 = axs[1]

# Vary T2
ax1.plot(range(n_cycles), fidelities, 'o-', label='No transpilation')
ax1.plot(range(n_cycles), fidelities_WACQT, 'o-', label='WACQT device')
ax1.plot(range(n_cycles), fidelities_diamond, 'o-', label='Diamond device')
ax1.set_xlabel(r'Error detection cycle $n$')
ax1.set_ylabel('Post selected fidelity')
# ax1.set_title('Fidelity of post selected states after $n$ error detection cycles')
#ax1.set(ylim=(0.74, 0.96))
ax1.legend()
ax1.grid(linewidth=1)

# Vary t_cz
ax2.plot(range(n_cycles), select_fractions, 'o-', label='No transpilation')
ax2.plot(range(n_cycles), select_fractions_WACQT, 'o-', label='WACQT device')
ax2.plot(range(n_cycles), select_fractions_diamond,
         'o-', label='Diamond device')
ax2.set_xlabel(r'Error detection cycle $n$')
ax2.set_ylabel(r'Post select fraction')
# ax2.set_title('Fidelity with varying 2-qb gate time, constant T2 (60 $\mu$s)')
#ax2.set(ylim=(0.74, 0.96))
ax2.legend()
ax2.grid(linewidth=1)

# %% TESTING WITH RECOVERY AND POST PROCESSING =================================
# Tests different combinations of reset/no reset, transpiled, post-processing
# and recovery.
#
# Took about 2h for me to run with 2048 shots per experiment

n_cycles = 15
reset = False
recovery = True
flag = False
# Registers
qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
# cr = ClassicalRegister(4, 'syndrome_bit') # The typical register
# Advanced list of registers
cr = get_classical_register(n_cycles, reset=reset,
                            recovery=recovery, flag=False)
readout = ClassicalRegister(5, 'readout')
registers = StabilizerRegisters(qb, an, cr, readout)

# Circuit without reset, without recovery
circ = encode_input_v2(registers)
circ += get_repeated_stabilization(registers, n_cycles=n_cycles,
                                   reset=False, recovery=False, flag=False, snapshot_type='density_matrix')

# Circuit without reset, with recovery
circ_rec = encode_input_v2(registers)
circ_rec += get_repeated_stabilization(registers, n_cycles=n_cycles,
                                       reset=False, recovery=True, flag=False, snapshot_type='density_matrix')

# Testing with only snapshotting the code qubits (No reset, recovery)
circ_rec_new = encode_input_v2(registers)
for current_cycle in range(n_cycles):
    circ_rec_new += get_stabilizer_cycle(registers,
                                               reset=False,
                                               recovery=True,
                                               current_cycle=current_cycle
                                               )
    circ_rec_new.barrier()
    circ_rec_new.append(Snapshot('stabilizer_'+str(current_cycle),
                                 'density_matrix', num_qubits=5), qb)

# Circuit with reset, without recovery
circ_res = encode_input_v2(registers)
circ_res += get_repeated_stabilization(registers, n_cycles=n_cycles,
                                       reset=True, recovery=False, flag=False, snapshot_type='density_matrix')

# Circuit with reset, with recovery
circ_res_rec = encode_input_v2(registers)
circ_res_rec += get_repeated_stabilization(registers, n_cycles=n_cycles,
                                           reset=True, recovery=True, flag=False, snapshot_type='density_matrix')

# Transpilation
routing_method = 'sabre'  # basic lookahead stochastic sabre
initial_layout = None  # Overwriting the above layout
layout_method = 'sabre'  # trivial 'dense', 'noise_adaptive' sabre
translation_method = None  # 'unroller',  translator , synthesis
repeats = 30
optimization_level = 1
circ_WACQT = shortest_transpile_from_distribution(circ,
                                                  print_cost=False, repeats=repeats, routing_method=routing_method,
                                                  initial_layout=initial_layout, layout_method=layout_method,
                                                  translation_method=translation_method, optimization_level=optimization_level,
                                                  **WACQT_device_properties
                                                  )
circ_rec_WACQT = shortest_transpile_from_distribution(circ_rec,
                                                      print_cost=False, repeats=repeats, routing_method=routing_method,
                                                      initial_layout=initial_layout, layout_method=layout_method,
                                                      translation_method=translation_method, optimization_level=optimization_level,
                                                      **WACQT_device_properties
                                                      )
circ_res_WACQT = shortest_transpile_from_distribution(circ_res,
                                                      print_cost=False, repeats=repeats, routing_method=routing_method,
                                                      initial_layout=initial_layout, layout_method=layout_method,
                                                      translation_method=translation_method, optimization_level=optimization_level,
                                                      **WACQT_device_properties
                                                      )
circ_res_rec_WACQT = shortest_transpile_from_distribution(circ_res_rec,
                                                          print_cost=False, repeats=repeats, routing_method=routing_method,
                                                          initial_layout=initial_layout, layout_method=layout_method,
                                                          translation_method=translation_method, optimization_level=optimization_level,
                                                          **WACQT_device_properties
                                                          )

circ_rec_new_WACQT = shortest_transpile_from_distribution(circ_rec_new,
                                                          print_cost=False, repeats=repeats, routing_method=routing_method,
                                                          initial_layout=initial_layout, layout_method=layout_method,
                                                          translation_method=translation_method, optimization_level=optimization_level,
                                                          **WACQT_device_properties
                                                          )
print('Starting to run 12 different processes')
# %% Run it
n_shots = 2048
# No processing, no reset, no transpilation
fid = get_running_fidelity_data_den_mat(circ,
                                        n_shots=n_shots,
                                        noise_model=thermal_relaxation_model_V2(),
                                        post_select=False,
                                        )
print('Check!')
# %%
# No processing, no reset, with transpilation
fid_t = get_running_fidelity_data_den_mat(circ_WACQT,
                                          n_shots=n_shots,
                                          noise_model=thermal_relaxation_model_V2(),
                                          post_select=False,
                                          )
print('Check!')
# No processing, with reset, no transpilation
# fid_res = get_running_fidelity_data_den_mat(circ_res,
#    n_shots=n_shots,
#    noise_model=thermal_relaxation_model(),
#    post_select=False,
# )
# print('Check!')
# No processing, with reset, with transpilation
fid_res_t = get_running_fidelity_data_den_mat(circ_res_WACQT,
                                              n_shots=n_shots,
                                              noise_model=thermal_relaxation_model_V2(),
                                              post_select=False,
                                              )
print('Check!')
# %%
# Post selection, no reset, no transpilation
fid_ps, frac = get_running_fidelity_data_den_mat(circ,
                                                 n_shots=n_shots,
                                                 noise_model=thermal_relaxation_model_V2(),
                                                 post_select=True,
                                                 )
print('Check!')
# Post selection, no reset, with transpilation
fid_ps_t, frac_t = get_running_fidelity_data_den_mat(circ_WACQT,
                                                     n_shots=n_shots,
                                                     noise_model=thermal_relaxation_model_V2(),
                                                     post_select=True,
                                                     )
print('Check!')
# Post selection, with reset, no transpilation
fid_ps_res, frac_res = get_running_fidelity_data_den_mat(circ_res,
                                                         n_shots=n_shots,
                                                         noise_model=thermal_relaxation_model_V2(),
                                                         post_select=True,
                                                         )
print('Check!')
# Post selection, with reset, with transpilation
fid_ps_res_t, frac_res_t = get_running_fidelity_data_den_mat(circ_res_WACQT,
                                                             n_shots=n_shots,
                                                             noise_model=thermal_relaxation_model_V2(),
                                                             post_select=True,
                                                             )
print('Check!')
# %%
# Recovery, no reset, no transpilation
fid_rec = get_running_fidelity_data_den_mat(circ_rec,
                                            n_shots=n_shots,
                                            noise_model=thermal_relaxation_model_V2(),
                                            post_select=False,
                                            )
print('Check!')
# Recovery, no reset, with transpilation
fid_rec_t = get_running_fidelity_data_den_mat(circ_rec_WACQT,
                                              n_shots=n_shots,
                                              noise_model=thermal_relaxation_model_V2(),
                                              post_select=False,
                                              )

print('Check!')
# Recovery, no reset, with transpilation
fid_rec_new_t = get_running_fidelity_data_den_mat(circ_rec_new_WACQT,
                                                  n_shots=n_shots,
                                                  noise_model=thermal_relaxation_model_V2(),
                                                  post_select=False,
                                                  )

# %%
print('Check!')
# Recovery, with reset, no transpilation
fid_rec_res = get_running_fidelity_data_den_mat(circ_res_rec,
                                                n_shots=n_shots,
                                                noise_model=thermal_relaxation_model_V2(),
                                                post_select=False,
                                                )
# %%
print('Check!')
# Recovery, with reset, with transpilation
fid_rec_res_t = get_running_fidelity_data_den_mat(circ_res_rec_WACQT,
                                                  n_shots=n_shots,
                                                  noise_model=thermal_relaxation_model_V2(),
                                                  post_select=False,
                                                  )
print('Check!')

# %%
fig, axs = plt.subplots(4, figsize=(14, 20))
ax1 = axs[0]
ax2 = axs[1]
ax3 = axs[2]
ax4 = axs[3]

# Plot 1: Reset or not (No processing)
#ax1.plot(range(n_cycles), fid, 'o-', color='blue', label='No transp, no reset')
ax1.plot(range(n_cycles), fid_t, 'o-', color='red', label='Transp, no reset')
#ax1.plot(range(n_cycles), fid_res, 'D-', color='blue', label='No transp, with reset')
ax1.plot(range(n_cycles), fid_res_t, 'D-',
         color='red', label='Transp with reset')
ax1.set_xlabel('Number of cycles')
ax1.set_ylabel('Average fidelity')
ax1.set_title('Reset vs no reset, without post selection or recovery')
#ax1.set(ylim=(0.74, 0.96))
ax1.legend()
ax1.grid(linewidth=1)
# %%
# Plot 2
ax2.plot(range(n_cycles), fid_rec, 'o-',
         color='blue', label='No transp, no reset')
ax2.plot(range(n_cycles), fid_rec_t, 'o-',
         color='red', label='Transp, no reset')
ax2.plot(range(n_cycles), fid_rec_res, 'D-',
         color='blue', label='No transp, with reset')
ax2.plot(range(n_cycles), fid_rec_res_t, 'D-',
         color='red', label='Transp with reset')
ax2.set_xlabel('Number of cycles')
ax2.set_ylabel('Average fidelity')
ax2.set_title('Reset vs no reset, with recovery')
ax2.legend()
ax2.grid(linewidth=1)
# %%
fig, axs = plt.subplots(4, figsize=(14, 20))
ax1 = axs[0]
ax2 = axs[1]
ax3 = axs[2]
ax4 = axs[3]
# Plot 3: Recovery, post-selection and nothing
#ax3.plot(range(n_cycles), fid_t, 'o-', color='blue', label='No processing')
#ax3.plot(range(n_cycles), frac_t, 'o-', color='red', label='Post-selection fraction')
ax3.plot(range(n_cycles), fid_rec, 'o-',
         color='purple', label='Nygamla otranspilerad')
ax3.plot(range(n_cycles), fid_rec_t, 'o-',
         color='red', label='Nygamla Recovery')
ax3.plot(range(n_cycles), fid_rec_res_t, 'o-',
         color='orange', label='Recovery reset')
ax3.plot(range(n_cycles), fid_rec_new_t, 'o-',
         color='blue', label='New recovery')
ax3.set_xlabel('Number of cycles')
ax3.set_ylabel('Average fidelity')
ax3.set_title('Comparing processing methods without reset')
ax3.legend()
ax3.grid(linewidth=1)
# %%
# Plot 4: Recovery, post-selection and nothing
ax4.plot(range(n_cycles), fid_res_t, 'o-', color='blue', label='No processing')
ax4.plot(range(n_cycles), frac_res_t, 'o-',
         color='red', label='Post-selected fraction')
ax4.plot(range(n_cycles), fid_rec_res_t, 'o-',
         color='orange', label='Recovery')
ax4.set_xlabel('Number of cycles')
ax4.set_ylabel('Average fidelity')
ax4.set_title('Comparing processing methods with reset')
ax4.legend()
ax4.grid(linewidth=1)

# %% Store all fidelities and fractions
with open('data/fid.txt', 'w') as f:
    for item in fid:
        f.write("%s\n" % item)

with open('data/fid_t.txt', 'w') as f:
    for item in fid_t:
        f.write("%s\n" % item)

with open('data/fid_res.txt', 'w') as f:
    for item in fid_res:
        f.write("%s\n" % item)

with open('data/fid_res_t.txt', 'w') as f:
    for item in fid_res_t:
        f.write("%s\n" % item)

with open('data/fid_rec.txt', 'w') as f:
    for item in fid_rec:
        f.write("%s\n" % item)

with open('data/fid_rec_t.txt', 'w') as f:
    for item in fid_rec_t:
        f.write("%s\n" % item)

with open('data/fid_rec_res.txt', 'w') as f:
    for item in fid_rec_res:
        f.write("%s\n" % item)

with open('data/fid_rec_res_t.txt', 'w') as f:
    for item in fid_rec_res_t:
        f.write("%s\n" % item)

with open('data/fid_ps.txt', 'w') as f:
    for item in fid_ps:
        f.write("%s\n" % item)

with open('data/fid_ps_t.txt', 'w') as f:
    for item in fid_ps_t:
        f.write("%s\n" % item)

with open('data/fid_ps_res.txt', 'w') as f:
    for item in fid_ps_res:
        f.write("%s\n" % item)

with open('data/fid_ps_res_t.txt', 'w') as f:
    for item in fid_ps_res_t:
        f.write("%s\n" % item)

with open('data/frac.txt', 'w') as f:
    for item in frac:
        f.write("%s\n" % item)

with open('data/frac_t.txt', 'w') as f:
    for item in frac_t:
        f.write("%s\n" % item)

with open('data/frac_res.txt', 'w') as f:
    for item in frac_res:
        f.write("%s\n" % item)

with open('data/frac_res_t.txt', 'w') as f:
    for item in frac_res_t:
        f.write("%s\n" % item)

# %% Load data example
fid_old_t = []
with open('data/fid_t.txt', 'r') as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string
        currentPlace = line[:-1]

        # add item to the list
        fid_old_t.append(float(currentPlace))

# print(fid_ps_res)

# %% Further testing

# Settings for circuit
n_cycles = 1
reset = False
flag = False
recovery = False


circ = encode_input_v2(registers)
circ.barrier()
circ += get_stabilizer_cycle(registers,
                                   reset=reset,
                                   recovery=recovery
                                   )
circ.barrier()
circ.append(Snapshot('stabilizer_0', 'density_matrix', num_qubits=5), qb)
circ.append(Snapshot('measure_0', 'density_matrix', num_qubits=2), an)
circ.barrier()
circ += get_stabilizer_cycle(registers,
                                   reset=reset,
                                   recovery=recovery
                                   )
circ.barrier()
circ.append(Snapshot('stabilizer_1', 'density_matrix', num_qubits=5), qb)
circ.append(Snapshot('measure_1', 'density_matrix', num_qubits=2), an)
circ.barrier()

results = execute(
    circ,
    Aer.get_backend('qasm_simulator'),
    noise_model=thermal_relaxation_model_V2(),
    shots=100
).result()

print(len(results.data()['snapshots']['density_matrix']['measure_0']))
print(len(results.data()['snapshots']['density_matrix']['measure_1']))
sv_pre = results.data()['snapshots']['density_matrix']['measure_0'][0]['value']
sv_post = results.data()[
    'snapshots']['density_matrix']['measure_1'][0]['value']

print(sv_pre)
print(sv_post)
# %%

Q = QuantumRegister(2)
R = ClassicalRegister(2)

circ = QuantumCircuit(Q, R)

circ.h(Q[0])
circ.snapshot('pre', 'density_matrix')
circ.measure(Q, R)
circ.snapshot('post', 'density_matrix')

results = execute(
    circ,
    Aer.get_backend('qasm_simulator'),
    noise_model=None,
    shots=1
).result()

sv_pre = results.data()['snapshots']['density_matrix']['pre'][0]['value']
sv_post = results.data()['snapshots']['density_matrix']['post'][0]['value']

print(sv_pre)
print(sv_post)


# %% TESTING WITH RECOVERY AND POST PROCESSING =================================
# Tests different combinations of reset/no reset, transpiled, post-processing
# and recovery.
#
# Took about 2h for me to run with 2048 shots per experiment

n_cycles = 15

# Registers
qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
# Advanced list of registers
cr = get_classical_register(n_cycles, reset=False, recovery=True, flag=False)
readout = ClassicalRegister(5, 'readout')
registers = StabilizerRegisters(qb, an, cr, readout)

# Circuit without reset, without recovery
circ = encode_input_v2(registers)
for current_cycle in range(n_cycles):
    circ += get_stabilizer_cycle(registers,
                                       reset=False,
                                       recovery=False,
                                       current_cycle=current_cycle
                                       )
    circ.barrier()
    circ.append(Snapshot('stabilizer_'+str(current_cycle),
                         'density_matrix', num_qubits=5), qb)
    circ.barrier()

# Without reset, with recovery
circ_rec = encode_input_v2(registers)
for current_cycle in range(n_cycles):
    circ_rec += get_stabilizer_cycle(registers,
                                           reset=False,
                                           recovery=True,
                                           current_cycle=current_cycle
                                           )
    circ_rec.barrier()
    circ_rec.append(Snapshot('stabilizer_'+str(current_cycle),
                             'density_matrix', num_qubits=5), qb)
    circ_rec.barrier()

# Circuit with reset, without recovery
circ_res = encode_input_v2(registers)
for current_cycle in range(n_cycles):
    circ_res += get_stabilizer_cycle(registers,
                                           reset=True,
                                           recovery=False,
                                           current_cycle=current_cycle
                                           )
    circ_res.barrier()
    circ_res.append(Snapshot('stabilizer_'+str(current_cycle),
                             'density_matrix', num_qubits=5), qb)
    circ_res.barrier()

# Circuit with reset, with recovery
circ_res_rec = encode_input_v2(registers)
for current_cycle in range(n_cycles):
    circ_res_rec += get_stabilizer_cycle(registers,
                                               reset=True,
                                               recovery=True,
                                               current_cycle=current_cycle
                                               )
    circ_res_rec.barrier()
    circ_res_rec.append(Snapshot('stabilizer_'+str(current_cycle),
                                 'density_matrix', num_qubits=5), qb)
    circ_res_rec.barrier()

# Transpilation
routing_method = 'sabre'  # basic lookahead stochastic sabre
initial_layout = None  # Overwriting the above layout
layout_method = 'sabre'  # trivial 'dense', 'noise_adaptive' sabre
translation_method = None  # 'unroller',  translator , synthesis
repeats = 30
optimization_level = 1
circ_WACQT = shortest_transpile_from_distribution(circ,
                                                  print_cost=False, repeats=repeats, routing_method=routing_method,
                                                  initial_layout=initial_layout, layout_method=layout_method,
                                                  translation_method=translation_method, optimization_level=optimization_level,
                                                  **WACQT_device_properties
                                                  )
circ_rec_WACQT = shortest_transpile_from_distribution(circ_rec,
                                                      print_cost=False, repeats=repeats, routing_method=routing_method,
                                                      initial_layout=initial_layout, layout_method=layout_method,
                                                      translation_method=translation_method, optimization_level=optimization_level,
                                                      **WACQT_device_properties
                                                      )
circ_res_WACQT = shortest_transpile_from_distribution(circ_res,
                                                      print_cost=False, repeats=repeats, routing_method=routing_method,
                                                      initial_layout=initial_layout, layout_method=layout_method,
                                                      translation_method=translation_method, optimization_level=optimization_level,
                                                      **WACQT_device_properties
                                                      )
circ_res_rec_WACQT = shortest_transpile_from_distribution(circ_res_rec,
                                                          print_cost=False, repeats=repeats, routing_method=routing_method,
                                                          initial_layout=initial_layout, layout_method=layout_method,
                                                          translation_method=translation_method, optimization_level=optimization_level,
                                                          **WACQT_device_properties
                                                          )

# %%
n_shots = 4096
# Without reset, without recovery
fid_t = get_running_fidelity_data_den_mat(circ_WACQT,
                                          n_shots=n_shots,
                                          noise_model=thermal_relaxation_model_V2(),
                                          post_select=False,
                                          )
print('Check!')
# Without reset, with recovery
fid_rec_t = get_running_fidelity_data_den_mat(circ_rec_WACQT,
                                              n_shots=n_shots,
                                              noise_model=thermal_relaxation_model_V2(),
                                              post_select=False,
                                              )
print('Check!')
# With reset, without recovery
fid_res_t = get_running_fidelity_data_den_mat(circ_res_WACQT,
                                              n_shots=n_shots,
                                              noise_model=thermal_relaxation_model_V2(),
                                              post_select=False,
                                              )
print('Check!')
# With reset, with recovery
fid_res_rec_t = get_running_fidelity_data_den_mat(circ_res_rec_WACQT,
                                                  n_shots=n_shots,
                                                  noise_model=thermal_relaxation_model_V2(),
                                                  post_select=False,
                                                  )
print('Check!')

# %%
fid_list = [fid_t, fid_rec_t, fid_res_t, fid_res_rec_t, fid_rec_old_t]
theta_list = []
for fid in fid_list:
    x_D = np.ones((n_cycles-1, 2))
    for i in range(n_cycles-1):
        x_D[i][1] += i
    y = np.log(np.reshape(np.asarray(fid), (n_cycles, 1)))[0:14]
    theta = np.dot(np.dot(np.linalg.inv(np.dot(x_D.T, x_D)), x_D.T), y)
    theta_list.append(theta)
    print(' ')

x = np.linspace(0, 15, 100)
y_pred_list = []
for theta in theta_list:
    y_pred = np.exp(theta[0]) * np.exp(x*theta[1])
    y_pred_list.append(y_pred)
# %%
fig, axs = plt.subplots(1, figsize=(10, 6))
x_dis = np.arange(1, n_cycles+1)
# Plot 3: Recovery, post-selection and nothing
axs.plot(x_dis, fid_t, 'o', color='purple', label='No reset, no recovery')
axs.plot(x_dis, fid_rec_t, 'o', color='red', label='No reset, with recovery')
axs.plot(x_dis, fid_res_t, 'o', color='orange',
         label='With reset, no recovery')
axs.plot(x_dis, fid_res_rec_t, 'o', color='blue',
         label='With reset and recovery')
axs.plot(x_dis, fid_rec_old_t, 'o', color='green',
         label='Last weeks recovery without reset')

axs.plot(x, y_pred_list[0], color='purple')
axs.plot(x, y_pred_list[1], color='red')
axs.plot(x, y_pred_list[2], color='orange')
axs.plot(x, y_pred_list[3], color='blue')
axs.plot(x, y_pred_list[4], color='green')

axs.set_xlabel('Number of cycles')
axs.set_ylabel('Average fidelity')
axs.set_title('Average fidelites with fitted curves (OLS)')
axs.legend()
axs.grid(linewidth=1)

# %% The data for this
fid_t = [0.7820266483694307, 0.6942680402113506, 0.6189581634542902,
         0.5473101432465972, 0.49189582903411766, 0.4406014871442757,
         0.3965180858132731, 0.35892438856430925, 0.3232515892060338,
         0.29521018575604546, 0.2643302101916627, 0.2432135965746566,
         0.2201951856089369, 0.20415497655150022, 0.1806484687564687]

fid_rec_t = [0.798477818390566,
             0.7491983321639275,
             0.6912507492283043,
             0.647402411328184,
             0.6061481085279817,
             0.5692872735136165,
             0.5456085538129658,
             0.50557278751874,
             0.4863295742848199,
             0.47139707390102287,
             0.45211248786089914,
             0.4367334419635363,
             0.41720460624066535,
             0.39088995869627174,
             0.5235649896825258]

fid_res_t = [0.7845025023971008,
             0.7000043612149575,
             0.6177375025741868,
             0.5531687953842557,
             0.4904526283466424,
             0.4443648730741602,
             0.39394444061392514,
             0.3515689198422542,
             0.31564453545351395,
             0.28482463935727065,
             0.26279310911880344,
             0.2354521351007122,
             0.21409207252546017,
             0.19548900335461822,
             0.17601042193263264]

fid_res_rec_t = [0.7932567010692473,
                 0.7556814792494603,
                 0.7234313525165101,
                 0.6799904055689716,
                 0.6350738978155256,
                 0.6138778889625136,
                 0.5900353078513001,
                 0.5655541807102011,
                 0.5343078026061858,
                 0.5139255686725417,
                 0.49641958452200496,
                 0.47383896955861887,
                 0.4587042687502112,
                 0.45040523409969274,
                 0.4407623852318419]

fid_rec_old_t = [0.7870037958617031,
                 0.7228019697228526,
                 0.6471260594024145,
                 0.5736778857468655,
                 0.515823071528533,
                 0.47456708456132407,
                 0.4340437032569509,
                 0.40167880095343594,
                 0.36422711302267485,
                 0.33298037969550853,
                 0.31369607750544365,
                 0.2963639572320855,
                 0.279663412338165,
                 0.2568175814295024,
                 0.24790719515050716]
