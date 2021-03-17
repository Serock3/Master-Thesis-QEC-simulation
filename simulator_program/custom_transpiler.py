from qiskit.compiler import transpile
from qiskit.transpiler import PassManager, CouplingMap, Layout
from qiskit.visualization import plot_circuit_layout
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.circuit.library.standard_gates import iSwapGate, SwapGate, SGate, CZGate
from qiskit import QuantumCircuit, QuantumRegister
import warnings

def default_cost_func(circ,t_single=15, t_multi=300):
    num_single_qb_gates = circ.size()-circ.num_nonlocal_gates()
    num_multi_qb_gates = circ.num_nonlocal_gates()
    return num_single_qb_gates*t_single+num_multi_qb_gates*t_multi

def depth_cost_func(circ,t_single=15, t_multi=300):
    return circ.depth()

def shortest_transpile_from_distribution(circuit, repeats=40, cost_func=default_cost_func, print_cost=True, **kwargs):
    depth = 10000
    for _ in range(repeats):
        with warnings.catch_warnings():  # sabre causes deprication warning, this will ignore them
            warnings.simplefilter("ignore")
            transpiled_circuit_tmp = transpile(circuit, **kwargs)
        if print_cost:
            print('cost: ', cost_func(transpiled_circuit_tmp))
        if transpiled_circuit_tmp.depth() < depth:
            depth = transpiled_circuit_tmp.depth()
            transpiled_circuit = transpiled_circuit_tmp
    return transpiled_circuit


# WAQCT 7 qb
basis_gates = ['id', 'u1', 'u2', 'u3', 'iswap', 'cz']
couplinglist = [[0, 1], [0, 6], [1, 6], [2, 3],
                [2, 6], [3, 6], [4, 5], [4, 6], [5, 6]]
reverse_couplinglist = [[y, x] for [x, y] in couplinglist]
coupling_map = CouplingMap(
    couplinglist=couplinglist+reverse_couplinglist,
    description='A hexagoal 7qb code with two ancillas')

# Dict with device properties of the WAQCT QC to be used for transpilation.
WAQCT_device_properties = {
    "basis_gates": basis_gates, "coupling_map": coupling_map}

# Diamond 7 qb
diamond_couplinglist = [[0, 2], [0, 3], [1, 3], [1, 4], [2, 5], [3, 5],
    [3, 6], [4, 6]]
reverse_diamond_couplinglist = [[y, x] for [x, y] in diamond_couplinglist]
diamond_coupling_map = CouplingMap(
    couplinglist=diamond_couplinglist+reverse_diamond_couplinglist,
    description='A double diamond 7qb code with two ancillas')

# Dict with device properties of the "Double diamond" chip for transpilation.
diamond_device_properties = {
    "basis_gates": basis_gates, "coupling_map": diamond_coupling_map}

# Triangle 10 qb
couplinglist_triangle = [[0,4], [0,1], [1,4], [1,5], [1,2], [2,5], [2,6], [2,3],
    [3,6], [4,7], [4,5], [5,7], [5,8], [5,6], [6,8], [7,8], [7,9], [8,9]]
reverse_triangle_couplinglist = [[y, x] for [x, y] in couplinglist_triangle]
coupling_map_triangle = CouplingMap(
    couplinglist=couplinglist_triangle+reverse_triangle_couplinglist,
    description='A triangular 10qb chip')

# Dict with device properties of the WAQCT QC to be used for transpilation.
triangle_device_properties = {
    "basis_gates": basis_gates, "coupling_map": coupling_map_triangle}

def _add_custom_device_equivalences():
    """ Ads custom gate equivalences to the SessionEquivalenceLibrary for transpilation
    NOTE: One needs to be run once!
    """
    print('Adding custom device equivalences')
    q = QuantumRegister(2, 'q')
    def_swap = QuantumCircuit(q)
    for inst, qargs, cargs in [
            (iSwapGate(), [q[0], q[1]], []),
            (CZGate(), [q[0], q[1]], []),
            (SGate().inverse(), [q[1]], []),
            (SGate().inverse(), [q[0]], [])
    ]:
        def_swap.append(inst, qargs, cargs)
    SessionEquivalenceLibrary.add_equivalence(SwapGate(), def_swap)

# TODO: Curently not functioning? 
# This function will automatically run the first time you import this file
_add_custom_device_equivalences()
