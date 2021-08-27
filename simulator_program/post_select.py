# %%

if __package__:
    from .stabilizers import get_snapshot_label
    from .custom_noise_models import thermal_relaxation_model_V2
else:
    from stabilizers import get_snapshot_label
    from custom_noise_models import thermal_relaxation_model_V2
from qiskit import QuantumRegister, AncillaRegister, ClassicalRegister
from matplotlib import pyplot as plt
from qiskit import execute, Aer
from qiskit.aqua.utils import get_subsystems_counts
from qiskit.quantum_info import state_fidelity
import numpy as np
import warnings
# %%
"""
This file contains various functions for processing simulation data, mainly for
the purpose of post-selection (removing certain runs based on measurement outcomes).

To utilize this, set conditional=True and (optional) recovery=False in simulation.

TODO:
1. Make selection work for every syndrome
1b. Return selection fraction for every syndrome
1c. Return counts not fraction
2. Create a post_process.py file that uses this to select its states (and processes density matrices)
2b. Verify that the result is the same for matrices and statevectors with the code in post_processing_test.py
3. Give every function more well defined and encapsulated purposes
4. Make a plot with no processing, post selection and post processing for comparison

"""


def get_subsystem_counts_up_to_cycle(counts, cycle, hex_keys=False):
    """Get counts for the subsystem of registers up to cycle. From the reduced dict
    post selection counts can be extracted. Key is converted to int unless hex_keys=True.

    E.g. for cycle = 0
    {'00000 00111 00011':1, '00000 00101 00000':4, '00000 00000 00000':10} -> {0:14, 3:1}

    Args:
        counts (Dict): Counts dictionary
        cycle (int): current cycle
    Returns:
        Dict: subsystem counts. If post_select_syndrome is set then and int is returned
    """

    subsys_counts = {}
    for outcome in counts:
        formated_outcome = int(''.join([key for key in outcome.split()[
                               len(outcome.split())-(cycle):]]).zfill(1), 2)
        if hex_keys:
            formated_outcome = hex(formated_outcome)
        if formated_outcome in subsys_counts:
            subsys_counts[formated_outcome] += counts[outcome]
        else:
            subsys_counts[formated_outcome] = counts[outcome]
    return subsys_counts


def get_subsystem_counts_up_to_bit(counts, bit, hex_keys=False):
    """ Does the same as get_subsystem_counts_up_to_cycle but consideres each bit
    of the measurements individually instead of in groups of one cycle.
    """

    subsys_counts = {}
    for outcome in counts:
        trimmed_outcome = outcome.replace(" ", "")
        formated_outcome = int(
            trimmed_outcome[len(trimmed_outcome)-bit:].zfill(1), 2)
        if hex_keys:
            formated_outcome = hex(formated_outcome)
        if formated_outcome in subsys_counts:
            subsys_counts[formated_outcome] += counts[outcome]
        else:
            subsys_counts[formated_outcome] = counts[outcome]
    return subsys_counts


def _expand_key(key: str, current_cycle: int, n_cycles: int):
    """Expand key into current_cycle copies, e.g. '101'->'101101101'"""
    value = int((key*current_cycle).zfill(1), 2)
    num_measurements = (n_cycles+1)*len(key)
    # For every 64th measurement, the string will be padded with 16 additional zeros
    hex_padding = int((num_measurements)/64)*16+3
    return f"{value:#0{hex_padding}x}"


def get_trivial_post_select_counts(counts, n_cycles, trivial_key: str = '0000'):
    # TODO: this if statement is unnecessary if instead we use the default trivial_key = '0000'
    # if trivial_key is None:
    #     return [get_subsystem_counts_up_to_cycle(
    #         counts, current_cycle)[0]
    #         for current_cycle in range(n_cycles+1)]
    # else:
    trivial_key_list = [int(_expand_key(trivial_key, current_cycle, n_cycles), 16)
                        for current_cycle in range(n_cycles+1)]
    return [get_subsystem_counts_up_to_cycle(
        counts, current_cycle)[trivial_key_list[current_cycle]]
        for current_cycle in range(n_cycles+1)]


def get_trivial_post_select_counts_V2(counts, stab_datapoints, num_snapshots):
    """
    Supports up to one snapshot per measurement.
    """
    trivial_counts = np.zeros(num_snapshots)
    # TODO: [0] refers to the trivial key. Implement support for other trivial keys?
    counts_at_stab = np.array([get_subsystem_counts_up_to_bit(counts, stab)[0] for stab in range(len(stab_datapoints)+1)])
    
    current_stab = 0
    for i in range(num_snapshots):
        if i in stab_datapoints:
            current_stab += 1
        trivial_counts[i] = counts_at_stab[current_stab]

    return trivial_counts


def get_trivial_exp_value(results, n_cycles: int, trivial_key: str = '0000'):
    """Get the expectation values corresponding to only trivial stabilizer measurements
    (so far).

    Args:
        results (Results object): Simulation result
        n_cycles (int):
        trivial_key (str, optional): Measurement bitstring corresponding to what a trivial measurement
        is for the given code, e.g. '101' for the distance two code. Defaults to '0000'.

    Returns:
        List[float]: List of expectation values post selected for trivial stabilizer outcomes.
    """
    trivial_key_list = [_expand_key(trivial_key, current_cycle, n_cycles)
                        for current_cycle in range(n_cycles+1)]
    return [results.data()[get_snapshot_label(snapshot_type='exp', conditional=True, current_cycle=current_cycle)][trivial_key]
            for current_cycle, trivial_key in enumerate(trivial_key_list)]


def get_trivial_post_select_den_mat(results, n_cycles: int, trivial_key: str = '0000'):
    """Get the density matrices corresponding to only trivial stabilizer measurements 
    (so far).

    Args:
        results (Results object): Simulation result
        n_cycles (int): 
        trivial_key (str, optional): Measurement bitstring corresponding to what a trivial measurement
        is for the given code, e.g. '101' for the distance two code. Defaults to '0000'.

    Returns:
        List[np.ndarray]: List of density matrices post selected for trivial stabilizer outcomes.
    """
    trivial_key_list = [_expand_key(trivial_key, current_cycle, n_cycles)
                        for current_cycle in range(n_cycles+1)]
    return [results.data()[get_snapshot_label(snapshot_type='dm', conditional=True, current_cycle=current_cycle)][trivial_key]
            for current_cycle, trivial_key in enumerate(trivial_key_list)]

# %% Depricated functions


def reformat_density_snapshot(results) -> dict:
    """
    DEPRECATED
    Reformats the snapshot data of the results object to be a
    dictionary with the measurement results as keys
    """
    warnings.warn("DEPRECATED", DeprecationWarning)
    snap_dict = {}
    for snapshot_name in results.data()['snapshots']['density_matrix']:
        res_dict = {}
        for item in results.data()[
                'snapshots']['density_matrix'][snapshot_name]:
            res_dict[item['memory']] = item['value']
        snap_dict[snapshot_name] = res_dict
    return snap_dict


def get_trivial_subsystem_counts_at_cycle(results, n_shots, cycle):
    # Deprecated
    warnings.warn(
        "Not maintained. Only filters at the current cycle.", DeprecationWarning)
    subsys_counts = get_subsystems_counts(results.get_counts())
    syndrome_reg_counts = subsys_counts[len(subsys_counts)-1-cycle]
    count_trivial_syndrome = 0
    for key in syndrome_reg_counts:
        if int(key) == 0:
            count_trivial_syndrome = syndrome_reg_counts[key]
    return count_trivial_syndrome/n_shots


def get_trivial_state(circ):
    warnings.warn("Use logical statevec instead", DeprecationWarning)
    results = execute(
        circ,
        Aer.get_backend('qasm_simulator'),
        noise_model=None,
        shots=1,
    ).result()

    return [state['value'] for state in
            results.data()['snapshots']['density_matrix']['stabilizer_0']
            if int(state['memory'], 16) == 0][0]


# %% Code to test above, to be removed
if __name__ == "__main__":
    from stabilizers import encode_input_v2, get_repeated_stabilization, StabilizerRegisters, get_classical_register, logical_states, get_full_stabilizer_circuit
    reset = False
    recovery = True
    
    n_cycles = 15
    qb = QuantumRegister(5, 'code_qubit')
    an = AncillaRegister(2, 'ancilla_qubit')
    # cr = ClassicalRegister(4, 'syndrome_bit') # The typical register
    cr = get_classical_register(n_cycles, reset=reset,
                                recovery=recovery)
    readout = ClassicalRegister(5, 'readout')

    registers = StabilizerRegisters(qb, an, cr, readout)

    circ = get_full_stabilizer_circuit(registers, n_cycles)

    n_shots = 100
    results = execute(
        circ,
        Aer.get_backend('qasm_simulator'),
        noise_model=thermal_relaxation_model_V2(),
        shots=n_shots
    ).result()

    # correct_state = get_trivial_state(circ)
    correct_state = logical_states(None)[0]
    fidelities = [state_fidelity(post_selected_state, correct_state) for post_selected_state
                  in get_trivial_post_select_den_mat(results, n_cycles)]
    select_counts = get_trivial_post_select_counts(
        results.get_counts(), n_cycles)

    fig, axs = plt.subplots(2, figsize=(14, 10))
    ax1 = axs[0]
    ax2 = axs[1]

    ax1.plot(range(n_cycles+1), fidelities, 'o-', label='No transpilation')
    ax1.set_xlabel(r'Error detection cycle $n$')
    ax1.set_ylabel('Post selected fidelities')
    ax1.legend()
    ax1.grid(linewidth=1)

    ax2.plot(range(n_cycles+1), select_counts, 'o-', label='No transpilation')
    ax2.set_xlabel(r'Error detection cycle $n$')
    ax2.set_ylabel(r'Post select fraction')
    ax2.legend()
    ax2.grid(linewidth=1)
    plt.show()
# %%
