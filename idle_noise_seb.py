# Code is Work in progress!
# Script/functions for adding thermal noise to idle qubits. The program is not
# yet finished, both in functionality but also splitting segments into
# separate functions for versatility.
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
from qiskit.converters import circuit_to_dag, dag_to_circuit

import numpy as np
from qiskit import *

from simulator_program.custom_noise_models import thermal_relaxation_model
# %% Temporary circuit for testing
qb = QuantumRegister(3, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
readout = ClassicalRegister(3, 'readout')

circ = QuantumCircuit(qb, an, readout)

circ.x(qb[1])
circ.cx(qb[2], qb[1])
# circ.iswap(qb[2],qb[1])
circ.barrier()
circ.measure(qb[0], readout[0])
circ.x(qb[0]).c_if(readout,1)
circ.draw()

# %%
# TODO: Expand gate_times. Probably want to have it as optional input?
gate_durations = {'x': 20, 'z':0 ,'cx': 200,'cx': 200,'iswap': 200, 'barrier': 0, 'measure': 1000}  # times in ns
dag = circuit_to_dag(circ)


# From the BasicSwap example 
from qiskit.transpiler.passes import BasicSwap
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.circuit.library.standard_gates.h import CHGate

# This is mostly a rewrite of the depth function with non-trivial time steps
reg_offset = 0
reg_map = {}
# This can be done more elegantly with the OrderedDicts
for reg in list(dag.qregs.values()) + list(dag.cregs.values()):
    reg_map[reg.name] = reg_offset
    reg_offset += reg.size

op_stack = [0] * reg_offset

new_dag = dag._copy_circuit_metadata()

for layer in dag.serial_layers():
    subdag = layer['graph']
    node = subdag.op_nodes()[0]
    print(node.name)
    print(node.qargs)
    print(node.cargs)
    print(node.condition)
    print(' ')

    kraus_layer = DAGCircuit()
    for qreg in subdag.qregs.values():
        kraus_layer.add_qreg(qreg)

    kraus_layer.apply_operation_back()

# May not work
for node in dag.op_nodes():
    print(node.name)
    print(node.qargs)
    print(node.cargs)
    print(node.condition)
    print(' ')
    
    # We will modify the (copies) of the node from op_nodes directly
    new_dag = DAGCircuit()
    qr = QuantumRegister(1)
    new_dag.add_qreg(qr)
    # TODO: FIX
    # for qreg in node.qargs:
    #     new_dag.add_qreg(qreg.register)
    # for creg in node.cargs:
    #     new_dag.add_creg(creg.register)

    new_dag._add_op_node(node.op,qr,cargs=None)
    # new_dag.apply_operation_back(node.op,node.qargs,node.cargs)
    T1=40e3
    T2=60e3
    time_to_next_op = 10

    new_circ = QuantumCircuit(node.qargs[0].register)
    new_circ.append(thermal_relaxation_error(T1, T2, time_to_next_op),[node.qargs[0]])
    new_dag.apply_operation_back(new_circ.to_instruction(), qargs=[node.qargs[0]])

    new_dag.apply_operation_back(thermal_relaxation_error(T1, T2, time_to_next_op).to_instruction(), qargs=[qr[0]])
    levels = []
    reg_ints = []

    dag_to_circuit(new_dag).draw()

    # Fix 'wires' here
    dag.substitute_node_with_dag(node=node, input_dag=new_dag, wires=(node.qargs+node.cargs))
    try:
        duration = gate_durations[node.name]
    except KeyError as op:
        print('No operation duration specified for ',op.args)
        # NOTE: here one could use time = 0 and not raise error instead
        raise

    for ind, reg in enumerate(node.qargs+node.cargs):
        reg_ints.append(reg_map[reg.register.name] + reg.index)
        levels.append(op_stack[reg_ints[ind]] + duration)
        
    # TODO: c_ifs should not have duration as normal
    # Assuming here that there is no conditional
    # snapshots or barriers ever.
    if node.condition:
        # Controls operate over all bits in the
        # classical register they use.
        # TODO: we probably don't want this.
        # Should allow e.g. measurements on each bit in a reg simultaneously 
        cint = reg_map[node.condition[0].name]
        for off in range(node.condition[0].size):
            if cint + off not in reg_ints:
                reg_ints.append(cint + off)
                levels.append(op_stack[cint + off] + duration)

    max_level = max(levels)
    for ind in reg_ints:
        idle_time = max_level - op_stack[ind]
        new_dag
        op_stack[ind] = max_level

    

#%%
qubit_list = circ.qubits

# TODO: Make list size depend on number of qubits in circuit (Hardcoded to 7)
time_schedule = [[] for _ in range(7)]
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
    # qubit = qubit_list[index]
    
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
            {'gate': node.name, 'time': gate_durations[node.name]})
    
    
# %%
for i in range(len(time_schedule)):
    print('Schedule for qubit '+str(i)+':\n', time_schedule[i],'\n')
    

# %%
