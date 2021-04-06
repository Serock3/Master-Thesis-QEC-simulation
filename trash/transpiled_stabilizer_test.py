# File for testing transpilation and composition of circuits
# Author: Sebastian

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
from simulator_program.custom_transpiler import _add_custom_device_equivalences
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.visualization.state_visualization import plot_state_hinton, plot_state_qsphere
from qiskit.quantum_info.states.statevector import Statevector

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

def comp_states(results1, results2):
    """Compares two versions of circuits supposed to be identical. 
    Looks at statevector snapshots and measurement counts.
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

def print_vec_diffs(state_vec1,state_vec2):
    state1 = Statevector(state_vec1)
    state2 = Statevector(state_vec2)

    if state1.dim != state2.dim:
        raise Exception("Error, dims not matching")

    dim = int(np.log2(state1.dim))

    diff = np.round(state_vec1.data-state_vec2.data,3)
    for i in np.where(diff!=0)[0]:
        print("Diff in",format(i, 'b').zfill(dim),np.round(state_vec1.data,3)[i],' -> ', np.round(state_vec2.data,3)[i])

def comp_states_mat(results1, results2):
    """Compares two versions of circuits supposed to be identical. 
    Looks at desity matrix snapshots and measurement counts.
    Works even if register sizer are different and permuted.

    TODO: Currently does not care about shots > 1 for result2.

    Args:
        results1 (result): result() from a qasm execution
        results2 (result): result() from a qasm execution
    """
    snapshot_type = 'density_matrix'  # 'density_matrix'# TODO: Make this automatic
    snapshot_list1 = [(name, state) for (name, state) in results1.data()[
        'snapshots'][snapshot_type].items()]
    snapshot_list2 = [(name, state) for (name, state) in results2.data()[
        'snapshots'][snapshot_type].items()]

    # print('Purity of encoded state = ', purity(snapshot_list2[0][1][0]['value']))

    if len(snapshot_list2[0][1]) > 1:
        print('SEVERAL MATRICES IN comp_states_mat FOR results2, SOMETHING NOT RIGHT')
    if len(snapshot_list1[0][1]) > 1:
        print('SEVERAL MATRICES IN comp_states_mat FOR results1, SOMETHING NOT RIGHT')

    for i in range(len(snapshot_list1)):
        for shot in range(1):
            fidelity = np.round(state_fidelity(
                snapshot_list2[i][1][0]['value'], snapshot_list1[i][1][0]['value']),3)

        print('Fidelity',snapshot_list2[i][0], fidelity)
        if fidelity!=1:
            print_vec_diffs(snapshot_list1[i][1][0]['value'],snapshot_list2[i][1][0]['value'])
    # TODO: Do something useful with this information
    try:
        counts = results1.get_counts()
        print('Counts 1\n',counts)
    except:
        pass
    try:
        counts = results2.get_counts()
        print('Counts 2\n',counts)
    except:
        pass

def verify_transpilation(circ,transpiled_circuit):
    results1 = execute(
        transpiled_circuit,
        Aer.get_backend('qasm_simulator'),
        shots=1
    ).result()
    results2 = execute(
        circ,
        Aer.get_backend('qasm_simulator'),
        shots=1
    ).result()

    comp_states_mat(results1,results2)

n_cycles = 0
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
circ2 = QuantumCircuit(qb)
# circ.x(0)
circ2 += encode_input_v2(registers)
circ2.snapshot('post_encoding', 'density_matrix')#statevector
# circ2.measure_all()
# results2 = execute(
#     circ2,
#     Aer.get_backend('qasm_simulator'),
#     shots=2**12
# ).result()

routing_method = 'sabre'  # basic lookahead stochastic sabre
initial_layout = {qb[0]: 0,
                  qb[1]: 1,
                  qb[2]: 2,
                  qb[3]: 3,
                  qb[4]: 4,
                  an[0]: 5,
                  an[1]: 6}
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
repeats = 10
optimization_level = 3
transpiled_circuit = shortest_transpile_from_distribution(circ2, repeats=repeats, routing_method=routing_method, initial_layout=initial_layout,
                                                          # ,coupling_map = WAQCT_device_properties['coupling_map']
                                                          # ,**{'basis_gates': ['id', 'u1', 'u2', 'u3', 'cz','iswap']})
                                                          layout_method=layout_method, translation_method=translation_method, optimization_level=optimization_level, **WACQT_device_properties)


print('Final depth = ', transpiled_circuit.depth())
print('Final gates = ', transpiled_circuit.count_ops())
verify_transpilation(circ2, transpiled_circuit)
display(transpiled_circuit.draw())

# %%



# 
# state1 = results1.data()['snapshots']['density_matrix']['post_encoding'][0]['value']
# state2 = results2.data()['snapshots']['density_matrix']['post_encoding'][0]['value']

# print(state_fidelity(state1,state2))

# %%
_add_custom_device_equivalences()
# %%
results1 = execute(
            transpiled_circuit,
            Aer.get_backend('qasm_simulator'),
            shots=1
        ).result()

state1 = results1.data()['snapshots']['density_matrix']['post_encoding'][0]['value']
state1 = DensityMatrix(state1)

results2 = execute(
            circ2,
            Aer.get_backend('qasm_simulator'),
            shots=1
        ).result()

state2 = results2.data()['snapshots']['density_matrix']['post_encoding'][0]['value']
state2 = DensityMatrix(state2)
print(np.round(state1.to_statevector().data,3))
print(np.round(state2.to_statevector().data,3))

print(np.round(state1.to_statevector().data-state2.to_statevector().data,3))
# %%
state_fidelity(state1,state2)
# %%
plot_state_qsphere(state1)


#%% https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.html#qiskit.circuit.QuantumCircuit

# %% Kod man kan använda istället för depth
circ.num_nonlocal_gates()
circ.qubit_duration()

circ.add_calibration()
#%% Kod för att hitta permutationer vid snapshot (Hade varit nice att kunna göra detta utan snapshot)
circ_t.data[-1] # om en snapshot ligger här får man typ detta

"""
circ_t.data[-1]
(<qiskit.providers.aer.extensions.snapshot.Snapshot at 0x7f1bcf7e7be0>,
 [Qubit(QuantumRegister(7, 'q'), 5),
  Qubit(QuantumRegister(7, 'q'), 2),
  Qubit(QuantumRegister(7, 'q'), 3),
  Qubit(QuantumRegister(7, 'q'), 6),
  Qubit(QuantumRegister(7, 'q'), 4),
  Qubit(QuantumRegister(7, 'q'), 0),
  Qubit(QuantumRegister(7, 'q'), 1)],
 [])
"""

#%% Kod för att konkatenera kretsar
circ.compose()

"""
            ┌───┐                   ┌─────┐                ┌───┐
lqr_1_0: ───┤ H ├───    rqr_0: ──■──┤ Tdg ├    lqr_1_0: ───┤ H ├───────────────
            ├───┤              ┌─┴─┐└─────┘                ├───┤
lqr_1_1: ───┤ X ├───    rqr_1: ┤ X ├───────    lqr_1_1: ───┤ X ├───────────────
         ┌──┴───┴──┐           └───┘                    ┌──┴───┴──┐┌───┐
lqr_1_2: ┤ U1(0.1) ├  +                     =  lqr_1_2: ┤ U1(0.1) ├┤ X ├───────
         └─────────┘                                    └─────────┘└─┬─┘┌─────┐
lqr_2_0: ─────■─────                           lqr_2_0: ─────■───────■──┤ Tdg ├
            ┌─┴─┐                                          ┌─┴─┐        └─────┘
lqr_2_1: ───┤ X ├───                           lqr_2_1: ───┤ X ├───────────────
            └───┘                                          └───┘
lcr_0: 0 ═══════════                           lcr_0: 0 ═══════════════════════

lcr_1: 0 ═══════════                           lcr_1: 0 ═══════════════════════
"""

circ.combine()
circ.extend()
circ.append()


#%% Save circuits
# Saves the compiled circuie
if True:
    with open('transpiled_circuit.dat', 'wb') as transpiled_circuit_file:
        pickle.dump(transpiled_circuit, transpiled_circuit_file)
# %% (Load and) run the circuit
if 'transpiled_circuit' not in locals():
    with open('transpiled_circuit.dat', 'rb') as transpiled_circuit_file:
        transpiled_circuit = pickle.load(transpiled_circuit_file)


import qiskit.providers.aer.extensions.snapshot_density_matrix

#%%
repeats = 5
routing_method = 'sabre'  # basic lookahead stochastic sabre
qreg_qubit = QuantumRegister(5, 'qubit')
circuit = QuantumCircuit(qreg_qubit)

circuit.z(qreg_qubit[0])
circuit.h(qreg_qubit[1])
circuit.h(qreg_qubit[2])
circuit.h(qreg_qubit[3])
circuit.h(qreg_qubit[4])
circuit.h(qreg_qubit[0])
circuit.cz(qreg_qubit[1], qreg_qubit[2])
circuit.cz(qreg_qubit[3], qreg_qubit[4])
circuit.cz(qreg_qubit[0], qreg_qubit[1])
circuit.cz(qreg_qubit[2], qreg_qubit[3])
circuit.cz(qreg_qubit[0], qreg_qubit[2])
circuit.cz(qreg_qubit[0], qreg_qubit[3])
circuit.cz(qreg_qubit[0], qreg_qubit[4])
circuit.h(qreg_qubit[0])
circuit.cz(qreg_qubit[0], qreg_qubit[4])
circuit.cz(qreg_qubit[0], qreg_qubit[1])
circuit.snapshot('post_encoding', 'density_matrix')

circuit_t = shortest_transpile_from_distribution(circuit,cost_func=depth_cost_func, repeats=repeats, routing_method=routing_method, initial_layout=initial_layout,
                                              layout_method=layout_method, translation_method=translation_method, optimization_level=optimization_level, **WACQT_device_properties)
verify_transpilation(circuit, circuit_t)
circuit_t.draw()
# %%
print(circuit_t.qasm())