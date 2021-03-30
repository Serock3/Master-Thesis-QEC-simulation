# File containing functions for adding noise to idle qubits and measuring
# circuit times. Also contains a dictionary of standard gate times which can
# be called for related purposes, such as noise models.

# %% Import modules
from qiskit import QuantumCircuit
from qiskit.providers.aer.noise import thermal_relaxation_error
from qiskit.converters import circuit_to_dag
import numpy as np

from simulator_program.custom_noise_models import thermal_relaxation_model

# %%

def add_idle_noise_to_circuit(circ, gate_times={}, T1=40e3, T2=60e3,
                              return_time=False, rename = False):
    """Creates a copy of a circuit with added thermal relaxation noise added
    for idle qubits.

    Args:
        circ: Qiskit circuit object to be copied
        gate_times: Dict containing all gate times in ns. If left empty or 
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
    full_gate_times = add_standard_gate_times(gate_times)

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

    for node in dag.op_nodes():
        # Set cargs to entire classical conditional register if it exists, otherwise to the cargs
        cargs = node.condition[0] if node.condition else node.cargs

        # List of bits included in gate
        gate_args = []
        for arg in node.qargs+list(cargs):
            gate_args.append(arg)

        latest_time = max([time_passed[gate_arg] for gate_arg in gate_args])
        # Apply idle noise to qargs
        for qarg in node.qargs:
            time_diff = latest_time - time_passed[qarg]
            if time_diff:
                thrm_relax = thermal_relaxation_error(
                    T1, T2, time_diff).to_instruction()
                if rename:
                    thrm_relax.name = f'Idle {time_diff}ns'
                new_circ.append(thrm_relax, [qarg])

        # Assume instant if classical condition exists TODO: Better solution?

        try:
            gate_time = full_gate_times[node.name] if not node.condition else 0
        except KeyError as op:
            print(
                f'WARNING! No operation duration specified for {op.args}, assuming instant.')
            gate_time = 0

        # Advance the time for the qubits included in the gate
        for gate_arg in gate_args:
            time_passed[gate_arg] = latest_time + gate_time

        if node.name == 'snapshot':
            time_at_snapshots_and_end[node.op.label] = max(
                time_passed.values())

        # Add the gate
        new_circ.append(node.op, node.qargs, node.cargs)

    time_at_snapshots_and_end['end'] = max(time_passed.values())

    new_circ._layout = circ._layout

    if return_time:
        return new_circ, time_at_snapshots_and_end
    return new_circ


def get_circuit_time(circ, gate_times):
    """Returns the total run time of a circuit, given specific gate times.

    Args:
        circ: Qiskit circuit object to be measured
        gate_times: Dict containing all gate times in ns. If left empty or 
            missing elements, standard values will be added.

    Returns:
        total_time: Total time in ns for running the full circuit
    """
    # Get gate times missing from input
    full_gate_times = add_standard_gate_times(gate_times)

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

        latest_time = max(time_passed.values())

        for gate_arg in gate_args:
            time_passed[gate_arg] = latest_time + full_gate_times[node.name]

        if node.name == 'snapshot':
            time_at_snapshots_and_end[node.op.label] = max(
                time_passed.values())

    time_at_snapshots_and_end['end'] = max(time_passed.values())

    return time_at_snapshots_and_end

# NOTE: Kanske bör ligga i custom_transpiler? Osäker


def get_standard_gate_times():
    """Return a dict of standard gate times (ns) used for simulator purposes."""
    return {
        'x': 20, 'y': 20, 'z': 0, 'h': 20, 'u1': 0, 'u2': 20, 'u3': 20,
        'cx': 200, 'cz': 200, 'swap': 200, 'iswap': 200,
        'barrier': 0, 'measure': 500, 'snapshot': 0
    }

# TODO: Detta kan väl vara ett argumment i funktionen ovan?


def add_standard_gate_times(incomplete_gate_times={}):
    """Add the standard gate times to a dict with missing entries"""
    standard_gate_times = get_standard_gate_times()
    # TODO: Add an if-statement that checks for keys 'single' and 'double' in
    #   incomplete list. If they exist, apply its value to all single/two qubit
    #   gates instead of using standard times
    for key in standard_gate_times:
        if key not in incomplete_gate_times:
            incomplete_gate_times[key] = standard_gate_times[key]
    return incomplete_gate_times


# %% Internal testing with a standard stabilizer circuit
if __name__ == '__main__':
    from qiskit import execute
    from simulator_program.stabilizers import *
    from simulator_program.custom_transpiler import *

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
    circ.swap(qb[0],qb[1])
    circ.measure(qb[0], readout[0])
    circ.measure(qb[1], readout[1])

    # circ.cx(qb[2], qb[1])
    # circ.iswap(an[0], qb[2])
    # circ.measure(an[0], readout[0])
    # circ.measure(an[1], readout[1])
    # circ.x(qb[0]).c_if(readout, 3)
    # circ.barrier()

    # circ.z(qb[0])
    # circ.append(Snapshot('asd', "density_matrix",
    #                      num_qubits=2), [qb[2], qb[0]])
    # circ.measure(an[0], readout[0])

    # display(circ.draw())
    # # circ_t = transpile(circ, routing_method='sabre', initial_layout=None,
    # #                    translation_method=None, layout_method='sabre',
    # #                    optimization_level=1, **WAQCT_device_properties)
    # circ_t = shortest_transpile_from_distribution(circ, print_cost=False,
    #                                               repeats=1, routing_method='sabre', initial_layout=None,
    #                                               translation_method=None, layout_method='sabre',
    #                                               optimization_level=1, **WAQCT_device_properties)
    new_circ, times = add_idle_noise_to_circuit(circ, return_time=True)
    print(new_circ)

    results = execute(
        new_circ,
        Aer.get_backend('qasm_simulator'),
        noise_model=None,
        shots=1024*16
    ).result()
    print(results.get_counts())
    # print(times)

    # flag = False
    # reset = True
    # recovery = True
    # n_cycles = 2

    # # Define our registers (Maybe to be written as function?)
    # qb = QuantumRegister(5, 'code_qubit')
    # an = AncillaRegister(2, 'ancilla_qubit')
    # cr = get_classical_register(
    #     n_cycles, reset=reset, recovery=recovery, flag=flag)
    # readout = ClassicalRegister(5, 'readout')

    # registers = StabilizerRegisters(qb, an, cr, readout)
    # circ = get_empty_stabilizer_circuit(registers)

    # # Get the complete circuit
    # circ += get_full_stabilizer_circuit(registers,
    #                                     n_cycles=n_cycles,
    #                                     reset=reset,
    #                                     recovery=recovery,
    #                                     flag=flag,
    #                                     )

    # # Transpile
    # circ_t = shortest_transpile_from_distribution(circ, print_cost=False,
    #                                               repeats=10, routing_method='sabre', initial_layout=None,
    #                                               translation_method=None, layout_method='sabre',
    #                                               optimization_level=1, **WAQCT_device_properties)

    # new_circ, times = add_idle_noise_to_circuit(circ_t, return_time=True)
    # print(new_circ)
    # print(times)
    # display(new_circ.draw(output='mpl'))

# %%
