# Script to transpile and manage circuits
# Includes funtionality to e.g. save circuits and very transpilation

# %% Imports
from pytket.passes import FullPeepholeOptimise, DefaultMappingPass
from pytket.predicates import GateSetPredicate, NoMidMeasurePredicate, ConnectivityPredicate
from pytket.extensions.qiskit import AerStateBackend
from pytket.passes import RebaseCustom
from pytket.routing import GraphPlacement
from pytket.passes import PlacementPass, RoutingPass, CliffordSimp
from pytket.device import Device
from pytket.routing import Architecture
from pytket import Circuit, OpType
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit, AerBackend
from qiskit.quantum_info import purity
from qiskit.quantum_info.states.measures import state_fidelity
from simulator_program.stabilizers import*
from simulator_program.stabilizers import _get_stabilizer_XZZXI
import numpy as np
from IPython.display import display

from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister, Aer, execute
from simulator_program.custom_transpiler import *
from simulator_program.custom_transpiler import WACQT_device_properties
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

    TODO: Make it work for non-trivial measurement outcomes

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

registers = StabilizerRegisters()
circ = encode_input_v2(registers)
circ.snapshot('post_encoding', 'density_matrix')
circ += get_repeated_stabilization(registers, n_cycles=n_cycles,
                                   reset=False, recovery=False, flag=False)
circ.snapshot('post_XZZXI', 'density_matrix')

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
circ_t = shortest_transpile_from_distribution(circ, cost_func=depth_cost_func,
                                              repeats=repeats, 
                                              routing_method=routing_method, 
                                              initial_layout=initial_layout,
                                              layout_method=layout_method, 
                                              translation_method=translation_method, 
                                              optimization_level=optimization_level, 
                                            #   **WAQCT_device_properties)
                                              coupling_map = WACQT_device_properties['coupling_map']
                                              ,**{'basis_gates': ['id', 'u1', 'u2', 'u3', 'cz','iswap','swap']})
print('Final depth = ', circ_t.depth())
print('Final gates = ', circ_t.count_ops())
verify_transpilation(circ, circ_t)
display(circ_t.draw())  # output='mpl'
# %% print to qasm to make circuit exportable to IBM quantum experience
print(circ_t.qasm())

# %% Try just optimizing the circuit
optimization_level = 3
circ_opt = shortest_transpile_from_distribution(circ, cost_func=depth_cost_func, repeats=repeats, routing_method=routing_method, initial_layout=initial_layout,
                                                layout_method=layout_method, translation_method=translation_method, optimization_level=optimization_level)
display(circ_opt.draw())  # output='mpl'
# %%
registers = StabilizerRegisters()
circ = encode_input_v2(registers)

circ += get_repeated_stabilization(registers, n_cycles=1,
        reset=False, recovery=False, flag=False,snapshot_type=None)

display(circ.draw())
c = qiskit_to_tk(circ)
display(tk_to_qiskit(c).draw())
topology = Architecture([[0, 1], [0, 6], [1, 6], [2, 3],
                         [2, 6], [3, 6], [4, 5], [4, 6], [5, 6]])

gateset = GateSetPredicate({OpType.noop, OpType.Reset, OpType.H, OpType.X, OpType.Z,
                            OpType.U1, OpType.U2, OpType.U3, OpType.CZ, OpType.Measure, OpType.ISWAPMax})
connectivity = ConnectivityPredicate(topology)
print(gateset.verify(c))
print(connectivity.verify(c))
# %%
def draw(tk_circ):
    qk_circ = tk_to_qiskit(tk_circ)
    display(qk_circ.draw())
    print(qk_circ.depth())
    print(qk_circ.count_ops())


dev = Device(topology)

c_copy = c.copy()
FullPeepholeOptimise().apply(c_copy)
CliffordSimp().apply(c_copy)
DefaultMappingPass(dev).apply(c_copy)

place = PlacementPass(GraphPlacement(dev))
place.apply(c_copy)
# RoutingPass(dev).apply(c_copy)
draw(c_copy)
# %%

multiq_gates = {OpType.CY, OpType.ZZPhase}
singleq_gates = {OpType.Rz, OpType.Ry}
cx_in_cy = Circuit(2)
cx_in_cy.Rz(0.5, 1).CY(0, 1).Rz(-0.5, 1)


def tk1_to_rzry(a, b, c):
    circ = Circuit(1)
    circ.Rz(c + 0.5, 0).Ry(b, 0).Rz(a - 0.5, 0)
    return circ


custom = RebaseCustom(multiq_gates, cx_in_cy, singleq_gates, tk1_to_rzry)

circ = Circuit(3)
circ.X(0).CX(0, 1).Ry(0.2, 1)
circ.add_gate(OpType.ZZPhase, -0.83, [2, 1]).Rx(0.6, 2)

custom.apply(circ)

print(circ.get_commands())
# %%
b = AerBackend()
b.compile_circuit(c)

circ_opt_tket = tk_to_qiskit(b.process_circuits(c, n_shots=100))
circ_opt_tket.draw()
# %%
sim_b = AerStateBackend()
print(sim_b.required_predicates)

#%%
from pytket import Circuit
from pytket.extensions.qiskit import IBMQBackend
from pytket.passes import DefaultMappingPass
from pytket.predicates import CompilationUnit
circ = Circuit(5, 5)
circ.CX(0, 1).CX(0, 2).CX(0, 3).CX(0, 4).measure_all()
cu = CompilationUnit(circ)
DefaultMappingPass(dev).apply(cu)
print(cu.circuit.get_commands())
print(cu.initial_map)
print(cu.final_map)
# %%
