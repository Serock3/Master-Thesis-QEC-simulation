# File containing functions for adding noise to idle qubits and measuring
# circuit times. Also contains a dictionary of standard gate times which can
# be called for related purposes, such as noise models.
# %% Import modules
from qiskit import QuantumCircuit
from qiskit.providers.aer.noise import thermal_relaxation_error
from qiskit.converters import circuit_to_dag
import numpy as np
import warnings

if __package__:
    from .stabilizers import (encode_input_v2,
                              get_empty_stabilizer_circuit)
    from . import custom_transpiler
    from .custom_noise_models import WACQT_gate_times, GateTimes, standard_times
else:
    from stabilizers import (encode_input_v2,
                             get_empty_stabilizer_circuit)
    import custom_transpiler
    from custom_noise_models import WACQT_gate_times, GateTimes, standard_times
# %%


def add_idle_noise_to_circuit(circ, gate_times={}, T1=40e3, T2=60e3,
                              return_time=False, rename=False, **kwargs):
    """Creates a copy of a circuit with added thermal relaxation noise added
    for idle qubits.

    Args:
        circ: Qiskit circuit object to be copied
        gate_times: Dict/GateTimes object containing all gate times in ns. If left empty or 
            missing elements, standard values will be added.
        T1: T1 thermal relaxation time (ns).
        T2: T2 thermal relaxation time (ns).
        return_time: Optional boolean. If set to True, the function will return
            the total time of the circuit in addition to regular outputs.
        rename: Whether or not to replace the name 'kraus' with the 'Idle X ns' to show the idle 
        time in prints. If true, then circuit will not be runnable.

    Returns:
        new_circ: Copy of circ input, with added idle noise.
        gate_times (optional): The total time of the new circuit (ns).
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

    # TODO: Fix this hack solution? I am really ashamed of having coded this
    # The label info on the delay custom unitaries cannot be obtained from nodes,
    # so here we scan through the entire circuit just to grab this info
    delay_partitions = 1
    for dat in circ.data:
        inst = dat[0]
        if hasattr(inst, 'label'):
            if inst.label != None:
                if len(inst.label) > 6:
                    if inst.label[:5] == 'delay':
                        delay_partitions = int(inst.label[6:])
                        break


    # Convert circuit to DAG
    dag = circuit_to_dag(circ)

    # New circuit to be generated
    new_circ = QuantumCircuit()
    for reg in circ.qregs + circ.cregs:
        new_circ.add_register(reg)

    # Dictionary with the times for each snapshot
    time_at_snapshots_and_end = {}

    time_passed = {}
    for reg in circ.qubits + circ.clbits:
        time_passed[reg] = 0

    correction_step = False
    for node in dag.op_nodes():
        # Set cargs to entire classical conditional register if it exists, otherwise to the cargs
        cargs = node.condition[0] if node.condition else node.cargs

        # List of bits included in gate
        gate_args = []
        for arg in node.qargs+list(cargs):
            gate_args.append(arg)

        try:
            # Old method (Assume instant classical feedback)
            #gate_time = full_gate_times[node.name] if not node.condition else 0

            # Add idle time before any correction gates. This assumes that only
            # the correction step has condition and that all correction gates
            # in a cycle are in one "block" of nodes.
            if not node.condition:
                gate_time = full_gate_times[node.name]
                correction_step = False
                if node.name == 'unitary':
                    gate_time = full_gate_times['delay']/delay_partitions

            # First conditional gate in correction step
            elif node.condition and not correction_step:
                gate_time = 0
                correction_step = True

                # Add feedback time to all bits (Only to be applied once per cycle)
                for reg in circ.clbits:
                    time_passed[reg] += full_gate_times['feedback']
            else:
                gate_time = 0

        except KeyError as op:
            print(
                f'WARNING! No operation duration specified for {op.args}, assuming instant.')
            gate_time = 0

        latest_time = max([time_passed[gate_arg] for gate_arg in gate_args])
        # Apply idle noise to qargs
        if not isinstance(T1, list):
            for qarg in node.qargs:
                time_diff = latest_time - time_passed[qarg]
                if time_diff:
                    thrm_relax = thermal_relaxation_error(
                        T1, T2, time_diff).to_instruction()
                    if rename:
                        thrm_relax.name = f'Idle {time_diff}ns'
                    new_circ.append(thrm_relax, [qarg])
        else:
            for qarg in node.qargs:
                time_diff = latest_time - time_passed[qarg]
                if time_diff:
                    thrm_relax = thermal_relaxation_error(
                        T1[qarg.index], T2[qarg.index], time_diff).to_instruction()
                    if rename:
                        thrm_relax.name = f'Idle {time_diff}ns'
                    new_circ.append(thrm_relax, [qarg])

        # Advance the time for the qubits included in the gate
        for gate_arg in gate_args:
            time_passed[gate_arg] = latest_time + gate_time

        if node.name == 'snapshot' or node.name.split('_')[0] == 'save':
            time_at_snapshots_and_end[node.op._label] = max(
                time_passed.values())

        # Add the gate
        new_circ.append(node.op, node.qargs, node.cargs)

        # Insert moved feedback delay
        # if full_gate_times['delay'] > 0:

        #     if node.name == 'save_density_matrix' or node.name == 'save_expval':
        #         # Do not add any delay before the first cycle
        #         if not passed_first_cycle:
        #             passed_first_cycle = True
        #             continue

        #         thrm_relax = thermal_relaxation_error(
        #             T1, T2, full_gate_times['delay']).to_instruction()
        #         if rename:
        #             time_diff = full_gate_times['delay']
        #             thrm_relax.name = f'Idle {time_diff}ns'
        #         for reg in new_circ.qubits:
        #             new_circ.append(thrm_relax, [reg])
        #             time_passed[reg] += full_gate_times['delay']

    time_at_snapshots_and_end['end'] = max(time_passed.values())

    new_circ._layout = circ._layout

    if return_time:
        return new_circ, time_at_snapshots_and_end
    return new_circ


def get_circuit_time(circ, gate_times={}):
    """Returns the total run time of a circuit, given specific gate times.

    Args:
        circ: Qiskit circuit object to be measured
        gate_times: Dict containing gate times in ns. If left empty or 
            missing elements, standard values will be added.

    Returns:
        total_time: Total time in ns for running the full circuit
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

    # Covert circuit to DAG
    dag = circuit_to_dag(circ)

    # For each operation, evolve the time for each qubit by gate time and
    # possible qubit idle time.

    # Dictionary with the times for each snapshot
    time_at_snapshots_and_end = {}
    time_passed = {}
    for reg in circ.qubits + circ.clbits:
        time_passed[reg] = 0

    for node in dag.op_nodes():
        # Set cargs to entire classical conditional register if it exists, otherwise to the cargs
        cargs = node.condition[0] if node.condition else node.cargs
        gate_args = []
        for arg in node.qargs+list(cargs):
            gate_args.append(arg)

        latest_time = max([time_passed[gate_arg] for gate_arg in gate_args])

        for gate_arg in gate_args:
            time_passed[gate_arg] = latest_time + full_gate_times[node.name]

        if node.name == 'snapshot' or node.name.split('_')[0] == 'save':
            time_at_snapshots_and_end[node.op._label] = max(
                time_passed.values())

    time_at_snapshots_and_end['end'] = max(time_passed.values())

    return time_at_snapshots_and_end

# Crashes when transpile=True
# Cannot handle transpiled circuits


def get_empty_noisy_circuit(registers, snapshot_times, encode_logical=False,
                            gate_times={}, T1=40e3, T2=60e3, transpile=False):
    """
    CHECKED: NO REFERENCES
    DEPRECATED

    Returns a circuit with only idle noise and snapshots that matches the
    times from add_idle_noise_to_circuit. Assumes that all involved qubtits
    are at the same time at snapshots.
    """

    if encode_logical:
        circ = get_empty_stabilizer_circuit(registers)
        circ += encode_input_v2(registers)

    else:
        circ = get_empty_stabilizer_circuit(registers)

    # Add snapshots and idle noise
    time_passed = get_circuit_time(circ, gate_times=gate_times)['end']
    for key in snapshot_times:
        time_diff = snapshot_times[key]-time_passed
        if time_diff > 0:
            thrm_relax = thermal_relaxation_error(
                T1, T2, time_diff).to_instruction()
            for qubit in circ.qubits:
                circ.append(thrm_relax, [qubit])
        circ.append(Snapshot(key, 'density_matrix', num_qubits=5),
                    registers.QubitRegister)
        time_passed = snapshot_times[key]

    if transpile:
        return custom_transpiler.shortest_transpile_from_distribution(circ, print_cost=False,
                                                                      repeats=10, routing_method='sabre', initial_layout=None,
                                                                      translation_method=None, layout_method='sabre',
                                                                      optimization_level=1)
    return circ

# This one should work with transpilation when encode_logical=True


def get_empty_noisy_circuit_v2(circ, snapshot_times, encode_logical=False,
                               gate_times={}, T1=40e3, T2=60e3):
    """
    CHECKED: NO REFERENCES
    DEPRECATED

    Returns a circuit with only idle noise and snapshots that matches the
    times from add_idle_noise_to_circuit. Assumes that all involved qubtits
    are at the same time at snapshots.
    """

    new_circ = QuantumCircuit()
    time_passed = 0
    for reg in circ.qregs + circ.cregs:
        new_circ.add_register(reg)

    if encode_logical:
        new_circ += rebuild_circuit_up_to_barrier(circ, gate_times=gate_times)
        time_passed = snapshot_times['post_encoding']

    # Append all snapshots from the circuit
    dag = circuit_to_dag(circ)
    snapshots = []
    for node in dag.op_nodes():
        if node.name == 'snapshot':
            snapshots.append(node)

    # Add all snapshots from previous circuit
    index = 0
    for key in snapshot_times:
        if key == 'end':
            break
        time_diff = snapshot_times[key]-time_passed
        if time_diff > 0:
            thrm_relax = thermal_relaxation_error(
                T1, T2, time_diff).to_instruction()
            for qubit in new_circ.qubits:
                new_circ.append(thrm_relax, [qubit])
        new_circ.append(snapshots[index].op,
                        snapshots[index].qargs, snapshots[index].cargs)
        time_passed = snapshot_times[key]
        index += 1
    return new_circ


def get_empty_noisy_circuit_v3(circ, snapshot_times, gate_times={},
                               T1=40e3, T2=60e3):
    """
    CHECKED: ONLY REFERENCED IN fidelity_from_scratch OR TRASH/ SCRIPTS

    Creates a circuit with only idle noise and snapshots that matches the
    times from get_circuit_time. Assumes that all involved qubtits
    are at the same time at snapshots.

    Args:
        circ: Qiskit circuit object to mimic.
        snapshot_times (dict): The times for each snapshot to be added.
        gate_times: Can be either a dict with some gate times (in ns), or a
                    GateTimes object. If it is a dict, gate times not included 
                    will be added from standard gate times.
        T1 (float): T1 thermal relaxation in ns, defaults to 40e3.
        T2 (float): T2 thermal relaxation in ns, defaults to 60e3.

    Returns:
        new_circ: Qiskit circuit object containing only the encoding and snap
                  from the input circuit.
    """

    # Create the new circuit
    new_circ = QuantumCircuit()
    time_passed = 0
    for reg in circ.qregs + circ.cregs:
        new_circ.add_register(reg)

    # Encode the logical qubit
    new_circ += rebuild_circuit_up_to_encoding(circ)
    time_passed = get_circuit_time(new_circ, gate_times=gate_times)['end']
    new_circ = add_idle_noise_to_circuit(new_circ, gate_times)

    # Create a list of all snapshots
    dag = circuit_to_dag(circ)
    snapshots = []
    for node in dag.op_nodes():
        if node.name == 'snapshot' or node.name.split('_')[0] == 'save':
            snapshots.append(node)

    # Add all snapshots from previous circuit, excluding post_encoding.
    index = 0
    for key in snapshot_times:
        if key == 'end':
            break
        # TODO: Add functionality to include post_encoding by updating the time
        # after rebuild_up_to_encdoding(). Note that an iswap is moved past the
        # snapshot which messes up the permutation. Maybe some nice solution can
        # fix this?
        elif key == 'post_encoding' or key.split('_')[-1] == '0':
            index += 1
            continue  # Skip the post_encoding snapshot due to changes in encode
        time_diff = snapshot_times[key]-time_passed
        if time_diff > 0:
            thrm_relax = thermal_relaxation_error(
                T1, T2, time_diff).to_instruction()
            for qubit in new_circ.qubits:
                new_circ.append(thrm_relax, [qubit])
        elif time_diff < 0:
            print('Time difference less than zero, something might be wrong...')
        new_circ.append(snapshots[index].op,
                        snapshots[index].qargs, snapshots[index].cargs)
        time_passed = snapshot_times[key]
        index += 1
    return new_circ


def rebuild_circuit_up_to_barrier(circ, gate_times={}):
    """
    CHECKED: ONLY REFERENCED IN DEPRECATED get_empty_noisy_circuit_V2    
    DEPRECATED
    Build a copy of a circuit up until (and inculding) the first barrier."""

    # Get gate times missing from input
    if isinstance(gate_times, dict):
        full_gate_times = standard_times.get_gate_times(
            custom_gate_times=gate_times)
    elif isinstance(gate_times, GateTimes):
        full_gate_times = gate_times
    else:
        warnings.warn('Invalid gate times, assuming standard_times')
        full_gate_times = standard_times

    # Convert circuit to DAG
    dag = circuit_to_dag(circ)

    # New circuit to be generated
    new_circ = QuantumCircuit()
    for reg in circ.qregs + circ.cregs:
        new_circ.add_register(reg)

    for node in dag.op_nodes():
        new_circ.append(node.op, node.qargs, node.cargs)
        if node.name == 'barrier':
            break

    new_circ._layout = circ._layout
    return new_circ


def rebuild_circuit_up_to_encoding(circ):
    """
    CHECKED: ONLY REFERENCED IN get_empty_noise_circuit_V3
    
    Build a copy of a circuit up until (and inculding) final iSwap, plus the
    following cz and u1 gates that are part of it. This function works as a
    band-aid solution to extract only the encoding from a transpiled circuit.

    It makes a fair amount of assumptions:
    - Assume that the encoding is finished by a barrier. This barrier is moved
    to after the final iswap.
    - Assume that the second iSwap comes after
    - Assume that the iswap+cz+u1 gates that makes the swap, are all adjacent
    in the list of DAG nodes.
    - Ignores any other gates between the barrier and second iSwap.

    For a better solution, it is probably necessary to go further back and look
    at either transpiling certain sections individually, or choosing a transpied
    circuit and sticking to that (not transpiling a new one every round).
    """

    # Convert circuit to DAG
    dag = circuit_to_dag(circ)

    # New circuit to be generated
    new_circ = QuantumCircuit()
    for reg in circ.qregs + circ.cregs:
        new_circ.add_register(reg)

    # Rebuild up to the final iSwap
    nodes = dag.op_nodes()
    barrier_reached = False
    for i in range(len(nodes)):
        if nodes[i].name == 'barrier':
            barrier_reached = True
        if not barrier_reached:
            new_circ.append(nodes[i].op, nodes[i].qargs, nodes[i].cargs)

        # Find the next iSwap
        if barrier_reached and nodes[i].name == 'iswap':
            new_circ.append(nodes[i].op, nodes[i].qargs, nodes[i].cargs)
            new_circ.append(nodes[i+1].op, nodes[i+1].qargs, nodes[i+1].cargs)
            new_circ.append(nodes[i+2].op, nodes[i+2].qargs, nodes[i+2].cargs)
            new_circ.append(nodes[i+3].op, nodes[i+3].qargs, nodes[i+3].cargs)
            new_circ.barrier()
            break

    new_circ._layout = circ._layout
    return new_circ


# %% Internal testing with a standard stabilizer circuit
if __name__ == '__main__':
    from qiskit import execute, QuantumRegister, AncillaRegister, ClassicalRegister, Aer
    from stabilizers import *
    from custom_transpiler import *
    from custom_noise_models import *

    qb = QuantumRegister(3, 'code_qubit')
    an = AncillaRegister(2, 'ancilla_qubit')
    readout = ClassicalRegister(3, 'readout')

    circ = QuantumCircuit(qb, an, readout)
    circ.x(qb[0])
    circ.x(qb[1])
    circ.x(qb[1])
    circ.x(qb[1])
    circ.x(qb[1])
    circ.x(qb[1])
    circ.swap(qb[2], qb[1])
    circ.swap(qb[0], qb[1])
    circ.measure(qb[1], readout[1])
    circ.measure(qb[0], readout[0])


    new_circ, times = add_idle_noise_to_circuit(
        circ, gate_times=standard_times, return_time=True, rename=False)
    print(new_circ)

    results = execute(
        new_circ,
        Aer.get_backend('qasm_simulator'),
        noise_model=thermal_relaxation_model_V2(),
        shots=1024*8
    ).result()
    print(results.get_counts())
    # print(times)

