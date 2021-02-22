#%%
from qiskit import QuantumRegister,ClassicalRegister,QuantumCircuit,execute,Aer
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CommutativeCancellation
import numpy as np
from IPython.core.display import display

snapshot_type = 'statevector'
qb = QuantumRegister(5, 'code_qubit')
an = QuantumRegister(2, 'ancilla_qubit')
cr = ClassicalRegister(4, 'syndrome_bit')

circuit = QuantumCircuit(2)

circuit.u1(np.pi/2, 1)
circuit.u2(0, np.pi, 0)
circuit.cz(0, 1)

circuit.iswap(0, 1)
circuit.cz(0, 1)
circuit.u1(np.pi/2, 1)

inv = circuit.inverse()
circuit.snapshot('middle', snapshot_type=snapshot_type)
circuit += inv
circuit.snapshot('end', snapshot_type=snapshot_type)
circuit.measure_all()
display(circuit.draw(output='mpl'))

circuit_new = PassManager(CommutativeCancellation()).run(circuit)
# circuit_new = transpile(circuit,optimization_level=2)

display(circuit_new.draw(output='mpl'))

results1 = execute(circuit, Aer.get_backend(
    'qasm_simulator')).result()
results2 = execute(circuit_new, Aer.get_backend(
    'qasm_simulator')).result()

state1 = results1.data()['snapshots']['statevector']['middle'][0] 
state2 = results2.data()['snapshots']['statevector']['middle'][0]
print('state1 ',np.round(state1,decimals = 3))
print('state2 ',np.round(state2,decimals = 3))
state1 = results1.data()['snapshots']['statevector']['end'][0] 
state2 = results2.data()['snapshots']['statevector']['end'][0]
print('state1 ',np.round(state1,decimals = 3))
print('state2 ',np.round(state2,decimals = 3))


try:
    print(results1.get_counts())
    print(results2.get_counts())
except:
    pass
# %%
