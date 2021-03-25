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
# TODO: The dicts in time_schedule need to also include control/target for cx,
#       classical reg for measurements, and condition for c_if. This might be
#       partly circumvented by assuming CZ and iSwap?
# TODO: An easier fix might be to not create a new circuit, but instead update
#       the old by adding noise into the DAG object. Check if this is possible
# %% Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import thermal_relaxation_error
from qiskit.converters import circuit_to_dag

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
circ.iswap(qb[2],qb[1])
circ.barrier()
circ.measure(qb[0], readout[0])
circ.draw()

# %%
# TODO: Expand gate_times. Probably want to have it as optional input?
gate_times = {'x': 20, 'cx': 200, 'barrier': 0, 'measure': 1000} # times in ns
dag = circuit_to_dag(circ)
qubit_list = circ.qubits

# TODO: Make list size depend on number of qubits in circuit (Hardcoded to 7)
time_schedule = [ [] for _ in range(7) ]

for node in dag.op_nodes():
    print(node.name)
    print(node.qargs)
    print(node.cargs)
    print(node.condition)
    print(' ')
#%%
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
        #qubit = qubit_list[index]
        
        if time_schedule[index]:
            for gate in time_schedule[index]:
                shortest_times[index] += gate['time']
    
    # Add the necessary time as noise on qubits
    time_diff = np.max(shortest_times) - shortest_times
    for index in indexes:
        if time_diff[index] > 0.:
            time_schedule[index].append(
                {'gate': 'kraus', 'time': time_diff[index]})

    # Add the gate to time_schedule
    for index in indexes:
        # TODO: Make cx/cz add target and control qubit to dict
        time_schedule[index].append(
                {'gate': node.name, 'time': gate_times[node.name]})
# %%
for i in range(len(time_schedule)):
    print('Schedule for qubit '+str(i)+':\n', time_schedule[i],'\n')
    
