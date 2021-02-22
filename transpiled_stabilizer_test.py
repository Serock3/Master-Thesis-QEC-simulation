# %%
from qiskit.quantum_info import purity
from qiskit.quantum_info.states.measures import state_fidelity
from simulator_program.stabilizers import get_classical_register
from colorsys import hls_to_rgb
import pylab as plt
from numpy import pi
import numpy as np
from qiskit.tools.visualization import plot_histogram
from IPython.display import display
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister, Aer, execute
from simulator_program.custom_transpiler import *

#%%

def encode_input(registers):
    """Encode the input into logical 0 and 1 for the [[5,1,3]] code. This
    assumes that the 0:th qubit is the original state |psi> = a|0> + b|1>
    """

    qbReg, _, _, _ = registers
    circ = QuantumCircuit(qbReg)

    circ.h(qbReg[3])
    circ.cz(qbReg[3], qbReg[1])
    circ.cz(qbReg[3], qbReg[2])
    circ.cx(qbReg[3], qbReg[0])

    circ.h(qbReg[2])
    circ.cx(qbReg[2], qbReg[0])
    circ.cz(qbReg[2], qbReg[3])
    circ.cz(qbReg[2], qbReg[4])

    circ.h(qbReg[1])
    circ.cz(qbReg[1], qbReg[0])
    circ.cx(qbReg[1], qbReg[3])
    circ.cz(qbReg[1], qbReg[4])

    circ.h(qbReg[4])
    circ.cz(qbReg[4], qbReg[2])
    circ.cz(qbReg[4], qbReg[3])
    circ.cx(qbReg[4], qbReg[1])

    return circ


def encode_input_v2(registers):
    """Encode the input into logical 0 and 1 for the [[5,1,3]] code. This
    assumes that the 0:th qubit is the original state |psi> = a|0> + b|1>.

    Alternate version Basudha found on stackoverflow.
    """

    qbReg, _, _, _ = registers
    circ = QuantumCircuit(qbReg)

    circ.z(qbReg[0])
    circ.h(qbReg[1])
    circ.h(qbReg[2])
    circ.h(qbReg[3])
    circ.h(qbReg[4])

    circ.h(qbReg[0])
    circ.cz(qbReg[0], qbReg[1])
    circ.cz(qbReg[0], qbReg[2])
    circ.cz(qbReg[0], qbReg[3])
    circ.cz(qbReg[0], qbReg[4])
    circ.h(qbReg[0])

    circ.cz(qbReg[0], qbReg[1])
    circ.cz(qbReg[2], qbReg[3])
    circ.cz(qbReg[1], qbReg[2])
    circ.cz(qbReg[3], qbReg[4])
    circ.cz(qbReg[0], qbReg[4])

    return circ


n_cycles = 1
reset = True
flag = False
recovery = False

# Define our registers (Maybe to be written as function?)
qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
# cr = ClassicalRegister(5, 'syndrome_bit') # The typical register
cr = get_classical_register(n_cycles, flag)  # Advanced list of registers
readout = ClassicalRegister(5, 'readout')

registers = [qb, an, cr, readout]  # Pack them together
circ = encode_input(registers)
circ.draw()
# %%
circ2 = encode_input_v2(registers)
circ2.draw()
# %% Testa snabbt om de är samma
asd = QuantumCircuit(qb)
asd.x(0)
asd += (circ+circ2.inverse())
asd.measure_all()
results = execute(
    asd,
    Aer.get_backend('qasm_simulator'),
).result()

print(results.get_counts())

# Yes, verkar så
# %% Jämför matriser


def compare_unitaries_v2(circ, circ2):
    backend_sim = Aer.get_backend('unitary_simulator')
    job_sim = execute([circ, circ2], backend_sim)
    result_sim = job_sim.result()
    unitary1 = result_sim.get_unitary(circ)
    unitary2 = result_sim.get_unitary(circ2)

    return np.allclose(unitary1, unitary2)


compare_unitaries_v2(circ, circ2)
# %%


def colorize(z):
    r = np.abs(z)
    arg = np.angle(z)

    h = (arg + pi) / (2 * pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    c = np.vectorize(hls_to_rgb)(h, l, s)  # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0, 2)
    return c


backend_sim = Aer.get_backend('unitary_simulator')
job_sim = execute([circ, circ2], backend_sim)
result_sim = job_sim.result()
unitary1 = result_sim.get_unitary(circ)
unitary2 = result_sim.get_unitary(circ2)

display(plt.imshow(colorize(np.round(unitary1, 3))))
display(plt.imshow(colorize(np.round(unitary2, 3))))
# %%
circ = QuantumCircuit(qb)
circ = encode_input(registers)
circ.snapshot('post_encoding', 'statevector')
circ.measure_all()
results1 = execute(
    circ,
    Aer.get_backend('qasm_simulator'),
    shots=2**12
).result()

display(plot_histogram(results1.get_counts()))

circ2 = encode_input_v2(registers)
circ2.snapshot('post_encoding', 'statevector')
circ2.measure_all()

results2 = execute(
    circ2,
    Aer.get_backend('qasm_simulator'),
    shots=2**12
).result()
display(plot_histogram(results2.get_counts()))
# %% Investigate the statevectors
# Both have only real components
state1 = np.real(
    np.round(results1.data()['snapshots']['statevector']['post_encoding'][0], 3))
state2 = np.real(
    np.round(results2.data()['snapshots']['statevector']['post_encoding'][0], 3))
print(state1)
print(state2)
np.allclose(state1, state2)

# Verkar vara samma vektor (skumt? Nån symmetri som inte syns i matriserna eller någon skillnad i ancillas?)
# %% Transpilera dem


routing_method = 'sabre'  # basic lookahead stochastic sabre
# initial_layout = {qb[0]: 0,
#                   qb[1]: 1,
#                   qb[2]: 2,
#                   qb[3]: 3,
#                   qb[4]: 4,
#                   an[0]: 5,
#                   an[1]: 6}
# initial_layout = {an[0]: 0,
#                     an[1]: 1,
#                     qb[0]: 2,
#                     qb[1]: 3,
#                     qb[2]: 4,
#                     qb[3]: 5,
#                     qb[4]: 6}
initial_layout = None  # Overwriting the above layout
layout_method = 'sabre'  # trivial 'dense', 'noise_adaptive' sabre
translation_method = None  # 'unroller',  translator , synthesis
repeats = 30
optimization_level = 2
transpiled_circuit = shortest_transpile_from_distribution(circ2, repeats=repeats, routing_method=routing_method, initial_layout=initial_layout,
                                                          # ,coupling_map = WAQCT_device_properties['coupling_map']
                                                          # ,**{'basis_gates': ['id', 'u1', 'u2', 'u3', 'cz','iswap']})
                                                          layout_method=layout_method, translation_method=translation_method, optimization_level=optimization_level, **WAQCT_device_properties)


print('Final depth = ', transpiled_circuit.depth())
print('Final gates = ', transpiled_circuit.count_ops())

# %% Testa om den fortfarande är rätt


def test_encode_circ(circ1, circ2):
    results1 = execute(
        circ1,
        Aer.get_backend('qasm_simulator'),
        shots=1
    ).result()

    results2 = execute(
        circ2,
        Aer.get_backend('qasm_simulator'),
        shots=1
    ).result()

    state1 = np.real(
        np.round(results1.data()['snapshots']['statevector']['post_encoding'][0], 3))
    state2 = np.real(
        np.round(results2.data()['snapshots']['statevector']['post_encoding'][0], 3))
    print(state1)
    print(state2)
    print(np.allclose(state1, state2))


test_encode_circ(circ, transpiled_circuit)

# %%
n_cycles = 0


def _get_fidelities_mat(results_noisy, results_ideal):
    state_vectors_noisy = results_noisy.data()['snapshots']['density_matrix']
    state_vectors_ideal = results_ideal.data()['snapshots']['density_matrix']

    running_fidelity = np.zeros([n_cycles+1])
    running_fidelity[0] = state_fidelity(state_vectors_ideal['post_encoding'][0]['value'],
                                         state_vectors_noisy['post_encoding'][0]['value'])
    print('Purity of encoded state = ', purity(
        state_vectors_noisy['post_encoding'][0]['value']))
    for j in range(n_cycles):
        running_fidelity[j+1] = state_fidelity(state_vectors_ideal['stabilizer_' + str(j)][0]['value'],
                                               state_vectors_noisy['stabilizer_' + str(j)][0]['value'])
    return running_fidelity


def _get_fidelities_vec(results_noisy, results_ideal):
    # Get the state vectors
    state_vectors_noisy = results_noisy.data()['snapshots']['statevector']
    state_vectors_ideal = results_ideal.data()['snapshots']['statevector']

    running_fidelity = np.zeros([n_shots, n_cycles+1])

    for i in range(n_shots):  # n_shots
        running_fidelity[i, 0] = state_fidelity(
            state_vectors_ideal['post_encoding'][0], state_vectors_noisy['post_encoding'][i])
        for j in range(n_cycles):
            running_fidelity[i, j+1] = state_fidelity(
                state_vectors_ideal['stabilizer_' + str(j)][0], state_vectors_noisy['stabilizer_' + str(j)][i])
    return np.sum(running_fidelity, 0) / n_shots


def get_fidelities(results_noisy, results_ideal, snapshot_type):
    # logical0 = logical_0_transp  # logical[0]  #
    if snapshot_type == 'density_matrix':
        return _get_fidelities_mat(results_noisy, results_ideal)
    return _get_fidelities_vec(results_noisy, results_ideal)


def comp_states(results1, results2):
    """Compares two versions of circuits supposed to be identical. 
    Looks at snapshots and measurement counts.
    TODO: Currently does not care about shots > 1 for result2.

    Args:
        results1 (result): result() from a qasm execution
        results2 (result): result() from a qasm execution
    """
    snapshot_type = 'statevector'  # 'density_matrix'# TODO: Make this automatic
    snapshot_list1 = [(name, state) for (name, state) in results1.data()[
        'snapshots'][snapshot_type].items()]
    snapshot_list2 = [(name, state) for (name, state) in results2.data()[
        'snapshots'][snapshot_type].items()]

    n_shots1 = len(snapshot_list1[0][1])
    n_shots2 = len(snapshot_list2[0][1])

    running_fidelity = np.zeros([n_shots1])

    for i in range(len(snapshot_list1)):
        for shot in range(n_shots1):
            running_fidelity[shot] = state_fidelity(
                snapshot_list2[i][1][0], snapshot_list1[i][1][shot])
        print(snapshot_list2[i][0], np.sum(running_fidelity) / n_shots1)
    counts = results1.get_counts()
    print(counts)
    counts = results2.get_counts()
    print(counts)

# %%


# %%
circ = QuantumCircuit(qb)
circ = encode_input(registers)
circ.snapshot('post_encoding', 'density_matrix')
circ.measure_all()
results1 = execute(
    circ,
    Aer.get_backend('qasm_simulator'),
    shots=2**12
).result()


circ2 = QuantumCircuit(qb)
circ2 = encode_input_v2(registers)
circ2.snapshot('post_encoding', 'density_matrix')
circ2.measure_all()
results2 = execute(
    circ2,
    Aer.get_backend('qasm_simulator'),
    shots=2**12
).result()
# %%
_get_fidelities_mat(results1, results2)
# %%
