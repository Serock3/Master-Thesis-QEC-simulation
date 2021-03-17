# %%
from qiskit import QuantumRegister, AncillaRegister, ClassicalRegister
from matplotlib import pyplot as plt
from stabilizers import encode_input_v2, get_classical_register, StabilizerRegisters, get_repeated_stabilization
from qiskit import execute, Aer
from qiskit.aqua.utils import get_subsystems_counts
from custom_noise_models import thermal_relaxation_model
from qiskit.quantum_info import state_fidelity

"""
TODO: 
1. Make selection work for every syndrome
1b. Return selection fraction for every syndrome
1c. Return counts not fraction
2. Create a post_process.py file that uses this to select its states (and processes density matrices)
2b. Verify that the result is the same for matrices and statevectors with the code in post_processing_test.py
3. Give every function more well defined and encapsulated purposes
4. Make a plot with no processing, post selection and post processing for comparison

"""


def reformat_density_snapshot(results) -> dict:
    """
    DEPRICATED
    Reformats the snapshot data of the results object to be a 
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


def get_running_post_select_fraction_for_density_matrix(results, n_shots, cycle):

    subsys_counts = get_subsystems_counts(results.get_counts())
    syndrome_reg_counts = subsys_counts[len(subsys_counts)-1-cycle]
    count_trivial_syndrome = 0
    for key in syndrome_reg_counts:
        if int(key) == 0:
            count_trivial_syndrome = syndrome_reg_counts[key]
    return count_trivial_syndrome/n_shots


def get_subsystem_counts_up_to_cycle(counts, cycle, post_select_syndrome = None):
    """Get counts for the subsystem of registers up to cycle. If post_select_syndrome is 
    set, then only return the counts for that specific (subsystem) measurement.

    Args:
        counts (Dict): Counts dictionary
        cycle (int): current cycle
        post_select_syndrome (int, optional): Subsystem measurement syndrome to post select. Defaults to None.

    Returns:
        Dict: subsystem counts. If post_select_syndrome is set then and int is returned
    """

    subsys_counts = {}
    count_trivial_syndrome = 0
    for outcome in counts:
        formated_outcome = int(''.join([key for key in outcome.split()[-(1+cycle):]]))
        if formated_outcome == post_select_syndrome:
            count_trivial_syndrome += counts[outcome]
    return count_trivial_syndrome



def get_running_fidelity_data_den_mat(circ, n_cycles, n_shots=2048,
                                      noise_model=thermal_relaxation_model(), post_select=True):
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

    # snapshots = reformat_density_snapshot(results)
    # TODO: Make this work if circuit it permuted for later stabilizers
    # TODO: More sophisticated key than '0x0'?
    correct_state = [state['value'] for state in results.data(
    )['snapshots']['density_matrix']['stabilizer_0'] if int(state['memory'], 16) == 0][0]
    
    # Run the circuit
    results = execute(
        circ,
        Aer.get_backend('qasm_simulator'),
        noise_model=noise_model,
        shots=n_shots
    ).result()

    # Post-selection
    fidelities = []
    # snapshots = reformat_density_snapshot(results)
    select_fractions = []
    if post_select:
        for current_cycle in range(n_cycles):
            try:
                post_selection = [state['value'] for state in results.data(
                )['snapshots']['density_matrix']['stabilizer_' + str(current_cycle)] if int(state['memory'], 16) == 0][0]
                # post_selection = snapshots['stabilizer_' + str(current_cycle)][[
                #     key for key in snapshots['stabilizer_0'] if int(key, 16) == 0][0]]
                select_fraction = get_subsystem_counts_up_to_cycle(
                    results.get_counts(), current_cycle)
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
        snapshots = reformat_density_snapshot(results) # TODO: remove this
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


# %% Code to test above, to be removed
reset = False
recovery = True
flag = False
n_cycles = 2
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

n_shots = 10
fidelities, select_fractions = get_running_fidelity_data_den_mat(
    circ, n_cycles, n_shots)


fig, axs = plt.subplots(2, figsize=(14, 10))
ax1 = axs[0]
ax2 = axs[1]

ax1.plot(range(n_cycles), fidelities, 'o-', label='No transpilation')
ax1.set_xlabel(r'Error detection cycle $n$')
ax1.set_ylabel('Post selected fidelity')
ax1.legend()
ax1.grid(linewidth=1)


ax2.plot(range(n_cycles), select_fractions, 'o-', label='No transpilation')
ax2.set_xlabel(r'Error detection cycle $n$')
ax2.set_ylabel(r'Post select fraction')
ax2.legend()
ax2.grid(linewidth=1)

# %%
