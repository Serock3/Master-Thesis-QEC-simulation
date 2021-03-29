# File containing functions for adding noise to idle qubits and measuring
# circuit times. Also contains a dictionary of standard gate times which can
# be called for related purposes, such as noise models.

# %% Import modules
from qiskit.providers.aer.noise import thermal_relaxation_error
from qiskit.converters import circuit_to_dag
import numpy as np

from simulator_program.custom_noise_models import thermal_relaxation_model

# %%
def add_idle_noise_to_circuit(circ, gate_times={}, T1=40e3, T2=60e3, 
        return_time=False):
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

    Returns:
        new_circ: Copy of circ input, with added idle noise.
        gate_times (optional): The total time of the new circuit (ns).
    """
    # Get gate times missing from input
    full_gate_times = add_standard_gate_times(gate_times)

    # Convert circuit to DAG
    dag = circuit_to_dag(circ)
    qubit_list = circ.qubits

    # New circuit to be generated
    new_circ = QuantumCircuit()
    for reg in circ.qregs + circ.cregs:
        new_circ.add_register(reg)
    

    # Build the new circuit
    time_passed = np.zeros(len(qubit_list))
    for node in dag.op_nodes():

        # Get the indexes of qubits involved in gate
        indexes = []
        for qargs in node.qargs:
            indexes.append(qubit_list.index(qargs))

        # Add idle noise if necessary
        time_diff = np.max(time_passed[indexes]) - time_passed
        for index in indexes:
            if time_diff[index] > 0.:
                new_circ.append(thermal_relaxation_error(
                    T1, T2, time_diff[index]),[index])
                time_passed[index] += time_diff[index]

        # Add the gate
        new_circ.append(node.op, node.qargs, node.cargs)
        time_passed[indexes] += full_gate_times[node.name]

    if return_time:
        return new_circ, np.max(time_passed)
    return new_circ

def get_full_circuit_time(circ, gate_times):
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
    dag = circuit_to_dag
    qubit_list = circ.qubits

    # For each operation, evolve the time for each qubit by gate time and
    # possible qubit idle time.
    time_passed = np.zeros(len(qubit_list))
    for node in dag.op_nodes():
        indexes = []
        for qargs in node.qargs:
            indexes.append(qubit_list.index(qargs))
        time_passed[indexes] = np.max(time_passed[indexes])
        time_passed[indexes] += full_gate_times[node.name]
    return np.max(time_passed)

def get_standard_gate_times():
    """Return a dict of standard gate times (ns) used for simulator purposes."""
    return {
        'x': 20, 'y': 20, 'z': 0, 'h': 20, 'u1': 0, 'u2': 20, 'u3': 20,
        'cx': 200, 'cz': 200, 'swap': 200, 'iswap': 200,
        'barrier': 0, 'measure': 500, 'snapshot': 0
    } 

def add_standard_gate_times(incomplete_gate_times={}):
    """Add the standard gate times to a dict with missing entries"""
    standard_gate_times = get_standard_gate_times()
    # TODO: Add an if-statement that checks for keys 'single' and 'double' in
    #   incomplete list. If they exist, apply its value to all single/two qubit 
    #   gates instead of using standard times
    for key in standard_gate_times:
        if key not in incomplete_gate_times:
            print(key)
            incomplete_gate_times[key] = standard_gate_times[key]
    return incomplete_gate_times

# %% Internal testing with a standard stabilizer circuit
if __name__ == '__main__':
    from qiskit import *
    from simulator_program.stabilizers import *
    from simulator_program.custom_transpiler import *
    flag = False
    reset = False
    recovery = True
    n_cycles=1
    
    # Define our registers (Maybe to be written as function?)
    qb = QuantumRegister(5, 'code_qubit')
    an = AncillaRegister(2, 'ancilla_qubit')
    cr = get_classical_register(n_cycles, reset=reset, recovery=recovery, flag=flag)
    readout = ClassicalRegister(5, 'readout')

    registers = StabilizerRegisters(qb, an, cr, readout)
    circ = get_empty_stabilizer_circuit(registers)

    # Get the complete circuit
    circ += get_full_stabilizer_circuit(registers,
                                        n_cycles=n_cycles,
                                        reset=reset,
                                        recovery=recovery,
                                        flag=flag,
                                        )

    # Transpile
    circ_t = shortest_transpile_from_distribution(circ, print_cost=False,
            repeats=10, routing_method='sabre', initial_layout=None,
            translation_method=None, layout_method='sabre',
            optimization_level=1, **WAQCT_device_properties)

    new_circ = add_idle_noise_to_circuit(circ_t)
    print(new_circ)
