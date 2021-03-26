#%%
from qiskit.providers.aer.noise import (
    pauli_error,
    depolarizing_error,
    thermal_relaxation_error, 
    amplitude_damping_error, 
    phase_damping_error , 
    phase_amplitude_damping_error,
)
from qiskit import (QuantumCircuit,
                    QuantumRegister,
                    ClassicalRegister,
                    AncillaRegister,
                    execute,
                    Aer
                    )
from qiskit.quantum_info import Operator, average_gate_fidelity
from qiskit.converters import circuit_to_dag
from qiskit.tools.visualization import dag_drawer
#%% Test that non unitary ops work
T1=40e3
T2=60e3
t_single=15

qb = QuantumRegister(2)
cl = ClassicalRegister(1)
circ = QuantumCircuit(qb,cl)
circ.x(0)
circ.append(thermal_relaxation_error(T1, T2, t_single),[qb[0]])
circ.measure(qb[0],cl)
circ.draw()
print(circuit_to_dag(circ).op_nodes())

results = execute(
    circ,
    Aer.get_backend('qasm_simulator'),
    noise_model=None,
    shots=1024*16
).result()

# Hm maybe?
results.get_counts()
#%%
qb = QuantumRegister(2)
cl = ClassicalRegister(1)
circ = QuantumCircuit(qb,cl)
circ.x(0)
circ.measure(qb[0],cl[0])
circ.x(1).c_if(cl,1)
p_error = 1
error_gate1 = pauli_error([('X',p_error), ('I', 1 - p_error)])
circ.append(error_gate1,[qb[0]]).c_if(cl,1)
circ.draw()

# %% Using pulse schedule to calculate time
from qiskit.test.mock import FakeAlmaden
from qiskit import transpile, schedule as build_schedule
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.providers.aer.noise.noise_model import NoiseModel
from IPython.core.display import display
backend = FakeAlmaden()

transpiled_circ = transpile(circ, backend)  # Undefined Hadamard is replaced by U1
schedule = build_schedule(transpiled_circ, backend)
schedule.draw(plot_range=[0, 2000])
print(schedule.duration)
# %% Evolve a density matrix with custom unitary or quantum QuantumChannel
circ = QuantumCircuit(qb,cl)
circ.x(qb[0])
circ.snapshot('asd','density_matrix')

results = execute(
    circ,
    Aer.get_backend('qasm_simulator'),
    noise_model=None,
    shots=1
).result()

den_mat = DensityMatrix(results.data(
    )['snapshots']['density_matrix']['asd'][0]['value'])

circ = QuantumCircuit(1)
circ.x(0)
# unitary = Operator(circ)
# print(unitary)
print(den_mat)
# den_mat.evolve(unitary)
print(den_mat.evolve(circ))

# %% Test if errors on iswap works
circ = QuantumCircuit(qb,cl)
circ.x(qb[0])
circ.iswap(qb[0],qb[1])
circ.snapshot('asd','statevector')

# Error that flips back qb[0] to |0>, so ends state is [1,0,0,0]
p_error = 1
error_gate1 = pauli_error([('X',p_error), ('I', 1 - p_error)])
noise_bit_flip = NoiseModel()
noise_bit_flip.add_all_qubit_quantum_error( error_gate1, ["x"] )

results = execute(
    circ,
    Aer.get_backend('qasm_simulator'),
    noise_model=noise_bit_flip,
    shots=1
).result()

den_mat = results.data(
    )['snapshots']['statevector']['asd'][0]
print(den_mat)
# %%
noise_bit_flip.basis_gates
# %%

# Build a sub-circuit
sub_q = QuantumRegister(2)
sub_circ = QuantumCircuit(sub_q, name='sub_circ')
sub_circ.h(sub_q[0])
sub_circ.crz(1, sub_q[0], sub_q[1])
sub_circ.barrier()
sub_circ.id(sub_q[1])
sub_circ.u(1, 2, -2, sub_q[0])
# sub_circ.measure_all()
# Convert to a gate and stick it into an arbitrary place in the bigger circuit
sub_inst = sub_circ.to_instruction()

qr = QuantumRegister(3, 'q')
circ = QuantumCircuit(qr)
circ.h(qr[0])
circ.cx(qr[0], qr[1])
circ.cx(qr[1], qr[2])
circ.append(sub_inst, [qr[1], qr[2]])
circ.measure_all()

display(circ.draw())

decomposed_circ = circ.decompose() # Does not modify original circuit
display(decomposed_circ.draw())

# %%
circ = QuantumCircuit(qb,cl)
circ.x(0)
circ.x(1)
circ.x(1)
circ.x(0).c_if(cl,0)
circ.data
circ.draw()
# %%
circ.x(1).instructions
# %%
circ.data.insert(4,circ.data[4])
# %%
circ.draw()
# %% C_if depth testing
cl2 = ClassicalRegister(1)
circ = QuantumCircuit(qb,cl, cl2)
# circ.x(0)
circ.measure(qb[0],cl[0])
circ.barrier()
circ.x(1).c_if(cl2,1)
display(circ.draw())
print(circ.depth())
# %%
