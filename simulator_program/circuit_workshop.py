# Script to transpile and manage circuits
# Includes funtionality to e.g. save circuits and very transpilation

# %% Imports
from qiskit.quantum_info import purity
from qiskit.quantum_info.states.measures import state_fidelity
from stabilizers import*
from stabilizers import _unflagged_stabilizer_XZZXI
import numpy as np
from IPython.display import display
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister, Aer, execute
from custom_transpiler import *
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.quantum_info.states.statevector import Statevector


# %%
def print_vec_diffs(state_vec1, state_vec2):
    state1 = DensityMatrix(state_vec1).to_statevector()
    state2 = DensityMatrix(state_vec2).to_statevector()

    if state1.dim != state2.dim:
        raise Exception("Error, dims not matching")

    dim = int(np.log2(state1.dim))

    diff = np.round(state1.data-state2.data, 3)
    for i in np.where(diff != 0)[0]:
        print("Diff in", format(i, 'b').zfill(dim), np.round(
            state1.data, 3)[i], ' -> ', np.round(state2.data, 3)[i])


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
                snapshot_list2[i][1][0]['value'], snapshot_list1[i][1][0]['value']), 3)

        print('Fidelity', snapshot_list2[i][0], fidelity)
        if fidelity != 1:
            print_vec_diffs(
                snapshot_list1[i][1][0]['value'], snapshot_list2[i][1][0]['value'])
    # TODO: Do something useful with this information
    try:
        counts = results1.get_counts()
        print('Counts 1\n', counts)
    except:
        pass
    try:
        counts = results2.get_counts()
        print('Counts 2\n', counts)
    except:
        pass


def verify_transpilation(circ, transpiled_circuit):
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

    comp_states_mat(results1, results2)


# %% Transpile the (v2) encoding circuit for [[5,1,3]]
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

registers = StabilizerRegisters(qbReg=qb)
circ = QuantumCircuit(qb)
circ += encode_input_v2(registers)
circ.snapshot('post_encoding', 'density_matrix')
# circ += _unflagged_stabilizer_XZZXI(registers, reset=True,
#                                     current_cycle=0, current_step=0)
# circ.snapshot('post_XZZXI', 'density_matrix')

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
repeats = 10
optimization_level = 3
circ_t = shortest_transpile_from_distribution(circ, repeats=repeats, routing_method=routing_method, initial_layout=initial_layout,
                                              # ,coupling_map = WAQCT_device_properties['coupling_map']
                                              # ,**{'basis_gates': ['id', 'u1', 'u2', 'u3', 'cz','iswap']})
                                              layout_method=layout_method, translation_method=translation_method, optimization_level=optimization_level, **WAQCT_device_properties)

print('Final depth = ', circ_t.depth())
print('Final gates = ', circ_t.count_ops())
verify_transpilation(circ, circ_t)
display(circ_t.draw())
# %% print to qasm to make circuit exportable to IBM quantum experience
print(circ_t.qasm())

# %%
