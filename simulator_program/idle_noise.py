"""File containing functions for adding noise to idle qubits and measuring
circuit times. 
"""
# %% Import modules
from qiskit import QuantumCircuit
from qiskit.providers.aer.noise import thermal_relaxation_error
from qiskit.converters import circuit_to_dag

if __package__:
    from .custom_noise_models import (extend_standard_gate_times)
else:
    from custom_noise_models import extend_standard_gate_times
# %%


def add_idle_noise_to_circuit(circ, gate_times={}, T1=40e3, T2=60e3,
                              return_time=False, rename=False, **kwargs):
    """Creates a copy of a circuit with added thermal relaxation noise added
    for idle qubits.

    Args:
        circ: Qiskit circuit object to be copied
        gate_times: Dict/GateTimes object containing all gate times in ns. If 
            left empty or missing elements, standard values will be added.
        T1: T1 thermal relaxation time (ns). Can be given as a list of values
            for different T1 on each qubit. Defaults to 40e3.
        T2: T2 thermal relaxation time (ns). Can be given as a list of values
            for different T2 on each qubit. Defaults to 60e3.
        return_time (bool): If set to True, the function will return
            a dict of times at each snapshot and end. Defaults to False
        rename: Whether or not to replace the name 'kraus' with the 'Idle X ns' to show the idle 
        time in prints. If true, then circuit will not be runnable.

    Returns:
        new_circ: Copy of circ input, with added idle noise.
        times_at_snapshots_and_end (optional): Dict of the circuit time at each
                                               snapshot in the circuit, as well
                                               as the total time at the end.
    """
    # Get gate times missing from input
    full_gate_times = extend_standard_gate_times(gate_times)

    # TODO: Fix this hack solution? I am really ashamed of having coded this
    # The label info on the delay custom unitaries cannot be obtained from nodes,
    # so here we scan through the entire circuit just to grab this info
    # NOTE: Maybe the custom gates made for splitting can be utilized instead?
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
        times_at_snapshots_and_end (optional): Dict of the circuit time at each
                                               snapshot in the circuit, as well
                                               as the total time at the end.
    """
    # Get gate times missing from input
    full_gate_times = extend_standard_gate_times(gate_times)

    # Covert circuit to DAG
    dag = circuit_to_dag(circ)

    # Dictionary with the times for each snapshot
    time_at_snapshots_and_end = {}
    time_passed = {}
    for reg in circ.qubits + circ.clbits:
        time_passed[reg] = 0

    # For each operation, evolve the time for each qubit by gate time and
    # possible qubit idle time.
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


