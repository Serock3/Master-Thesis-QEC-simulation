# Code is Work in progress!
# Script/functions for adding thermal noise to idle qubits. The program is not
# yet finished, both in functionality but also splitting segments into 
# seperate functions for versaility.
#
# The general procedure is to create a time_schedule, which is a list of all
# qubits in the circuit. Each qubit in the list is a list on its own, with
# all gates (written as dicts) in order (including the added idle noise).
# Its elements should be able to be interpreted to create the desired circuit.
#

# %% Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import thermal_relaxation_error
from qiskit.converters import circuit_to_dag
from qiskit.converters import dag_to_circuit
import numpy as np
from qiskit import *

from simulator_program.custom_noise_models import thermal_relaxation_model
# %% Temporary circuit for testing
qb = QuantumRegister(3, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
readout = ClassicalRegister(3, 'readout')

circ = QuantumCircuit(qb, an, readout)

circ.x(qb[1])
circ.cx(qb[2],qb[1])
circ.barrier()
#circ.swap(qb[1], qb[2])
#circ.append(Snapshot('snapshot',
#            'density_matrix', num_qubits=3), qb)
circ.measure([qb[0],qb[1]], [readout[0],readout[1]])
#circ.measure(qb[0], readout[0])
#circ.measure(qb[1], readout[1])
circ.barrier()
circ.draw()

# %% Paramenters used
# TODO: Expand gate_times. Probably want to have it as optional input?
gate_times = {'x': 20, 'cx': 200, 'barrier': 0, 'measure': 1000,
    'z': 0, 'h': 20, 'cz': 200, 'iswap': 200, 'snapshot': 0, 'u1': 0,
    'u2': 20, 'u3': 20} # times in ns

# TODO: Make these available as inputs
T1 = 40e3
T2 = 60e3

# TODO: Make list size depend on number of qubits in circuit (Hardcoded to 7)
time_schedule = [ [] for _ in qubit_list ]


#%% Simpler solution than the one below
dag = circuit_to_dag(circ_t)
qubit_list = circ_t.qubits

new_circ = QuantumCircuit()
for reg in circ_t.qregs + circ_t.cregs:
    new_circ.add_register(reg)
time_passed = np.zeros(7)
for node in dag.op_nodes():

    # Get the indexes of qubits involved in gate
    indexes = []
    for qargs in node.qargs:
        try:
            indexes.append(qubit_list.index(qargs))
        except:
            raise NameError('qarg not in qubit list')

    # Add idle noise if necessary
    time_diff = np.max(time_passed[indexes]) - time_passed
    for index in indexes:
        if time_diff[index] > 0.:
            new_circ.append(thermal_relaxation_error(T1, T2, time_diff[index]),[index])
            time_passed[index] += time_diff[index]

    # Add the gate
    new_circ.append(node.op, node.qargs, node.cargs)
    time_passed[indexes] += gate_times[node.name]

print(new_circ)
#%% Slower version of the cell above
dag = circuit_to_dag(circ)
qubit_list = circ.qubits
for node in dag.op_nodes():

    # Get the indexes of qubits involved in gate
    indexes = []
    for qargs in node.qargs:
        try:
            indexes.append(qubit_list.index(qargs))
        except:
            raise NameError('qarg not in qubit list')
    
    # Check the minimum time to the gate for all qubits involved
    shortest_times = np.zeros(len(time_schedule))
    for index in indexes:
        
        if time_schedule[index]:
            for gate in time_schedule[index]:
                shortest_times[index] += gate['time']
    
    # Add the necessary time as noise on qubits
    time_diff = np.max(shortest_times) - shortest_times
    for index in indexes:
        if time_diff[index] > 0.:
            time_schedule[index].append({
                'name': 'kraus',
                'time': time_diff[index],
                'qargs': node.qargs,
                'cargs': node.cargs,
                'qubits': [index]})

    # Add the gate to time_schedule
    for index in indexes:
        time_schedule[index].append({
            'name': node.name, 
            'time': gate_times[node.name],
            'qubits': indexes,
            'qargs': node.qargs,
            'cargs': node.cargs,
            'instr': node.op})

# Reconstruct the circuit
recirc = QuantumCircuit()
for reg in circ.qregs + circ.cregs:
    recirc.add_register(reg)

depth = np.zeros(len(time_schedule), dtype=int)
time_passed = np.zeros(len(time_schedule))

while True:
    # Choose gate to add with np.argmin(time_passed)
    qubit_index = np.argmin(time_passed)

    # Extract gate name and qubits involved
    gate = time_schedule[qubit_index][depth[qubit_index]]

    # Add the gate to circuit
    if gate['name'] == 'kraus':
        recirc.append(thermal_relaxation_error(T1, T2, gate['time']),gate['qubits'])
    else:
        recirc.append(gate['instr'], gate['qargs'], gate['cargs'])

    # Update times and depth to qubits involved
    time_passed[gate['qubits']] += gate['time'] 
    depth[gate['qubits']] += 1

    # Check if all instructions have been added
    remaining_gates = [len(time_schedule[qubit])-depth[qubit] for qubit in range(len(time_schedule))]
    if not any(remaining_gates):
        break

recirc.draw()


# %% Creating stabilizer circuits for further testing

from simulator_program.stabilizers import *
from simulator_program.custom_transpiler import *
flag = False
reset=False
n_cycles=2
recovery=True

# Define our registers (Maybe to be written as function?)
qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
#cr = ClassicalRegister(4, 'syndrome_bit')  # The typical register
cr = get_classical_register(n_cycles, reset=reset, recovery=recovery, flag=flag) # Advanced list of registers
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

circ_t = shortest_transpile_from_distribution(circ, print_cost=False,
        repeats=10, routing_method='sabre', initial_layout=None,
        translation_method=None, layout_method='sabre',
        optimization_level=1, **WAQCT_device_properties)
