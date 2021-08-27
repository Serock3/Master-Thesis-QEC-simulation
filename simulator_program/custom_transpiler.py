"""Contains the necessary functions to transpile a circuit to a given connectivity
and gate set, as well as the properties for such devices.
"""

#%%
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager, CouplingMap, Layout
from qiskit.visualization import plot_circuit_layout
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.circuit.library.standard_gates import iSwapGate, SwapGate, SGate, CZGate, CXGate
from qiskit import QuantumCircuit, QuantumRegister
if __package__:
    from . import idle_noise
else:
    import idle_noise
import warnings

#%% Device transpiling properties

# Set of basis gates us in all transpilations here
basis_gates = ['id', 'u1', 'x', 'y', 'z', 'sx', 'sy', 'iswap', 'cz',
    'save_expval', 'save_density_matrix', 'set_density_matrix', 'save_expval_var',
    'unitary', 'split']

# WACQT 7-qb hexagonal chip
couplinglist = [[0, 1], [0, 6], [1, 6], [2, 3],
                [2, 6], [3, 6], [4, 5], [4, 6], [5, 6]]
reverse_couplinglist = [[y, x] for [x, y] in couplinglist]
coupling_map = CouplingMap(
    couplinglist=couplinglist+reverse_couplinglist,
    description='A hexagonal 7qb code with two ancillas')

# Dict with device properties of the hexagonal chip for transpilation.
WACQT_device_properties = {
    "basis_gates": basis_gates, "coupling_map": coupling_map}

# Diamond 7 qb chip (Shaped as an '8' or two squares/diamonds) 
diamond_couplinglist = [[0, 2], [0, 3], [1, 3], [1, 4], [2, 5], [3, 5],
                        [3, 6], [4, 6]]
reverse_diamond_couplinglist = [[y, x] for [x, y] in diamond_couplinglist]
diamond_coupling_map = CouplingMap(
    couplinglist=diamond_couplinglist+reverse_diamond_couplinglist,
    description='A double diamond 7qb code with two ancillas')

# Dict with device properties of the "Double diamond" chip for transpilation.
diamond_device_properties = {
    "basis_gates": basis_gates, "coupling_map": diamond_coupling_map}

# Triangular 10 qb chip. Useful for [[7,1,3]] code, but difficult to realize experimentally
couplinglist_triangle = [[0, 4], [0, 1], [1, 4], [1, 5], [1, 2], [2, 5], [2, 6], [2, 3],
                         [3, 6], [4, 7], [4, 5], [5, 7], [5, 8], [5, 6], [6, 8], [7, 8], [7, 9], [8, 9]]
reverse_triangle_couplinglist = [[y, x] for [x, y] in couplinglist_triangle]
coupling_map_triangle = CouplingMap(
    couplinglist=couplinglist_triangle+reverse_triangle_couplinglist,
    description='A triangular 10qb chip')

# Dict with device properties of "triangle 10 qb" chip to be used for transpilation.
triangle_device_properties = {
    "basis_gates": basis_gates, "coupling_map": coupling_map_triangle}

# Cross-shape 5 qb chip, used for the [[4,2,2]] code.
cross_couplinglist = [[0, 4], [1, 4], [2, 4], [3, 4]]
reverse_cross_couplinglist = [[y, x] for [x, y] in cross_couplinglist]
cross_coupling_map = CouplingMap(
    couplinglist=cross_couplinglist+reverse_cross_couplinglist,
    description='A 5 qubit cross/plus with a single ancilla')

# Dict with device properties of the "Double diamond" chip for transpilation.
cross_device_properties = {
    "basis_gates": basis_gates, "coupling_map": cross_coupling_map}
#%% Functions

def weighted_gate_time_cost_fun(circ, t_single=15, t_multi=300):
    num_single_qb_gates = circ.size()-circ.num_nonlocal_gates()
    num_multi_qb_gates = circ.num_nonlocal_gates()
    return num_single_qb_gates*t_single+num_multi_qb_gates*t_multi


def depth_cost_func(circ, t_single=15, t_multi=300):
    return circ.depth()

def shortest_transpile_from_distribution(circuit,
                                         repeats=40,
                                         cost_func=lambda circ: idle_noise.get_circuit_time(circ)['end'],
                                         print_cost=True,
                                         routing_method='sabre',
                                         initial_layout=None,
                                         translation_method=None,
                                         layout_method='sabre',
                                         optimization_level=1,
                                         **kwargs):
    """Transpile 'repeats' number of times, and chose the best one according to the cost function
    (defaults to circuit time).
    """
    
    # Here we modify the kwargs dict so we can pass it to transpile() with our default values

    # Populate gate times not specified with default values
    for key in WACQT_device_properties:
        if key not in kwargs:
            kwargs[key] = WACQT_device_properties[key]

    # Add remaining settings to kwargs, so it can be passed to transpile in one dict
    kwargs['routing_method'] = routing_method
    kwargs['initial_layout'] = initial_layout
    kwargs['translation_method'] = translation_method
    kwargs['layout_method'] = layout_method
    kwargs['optimization_level'] = optimization_level

    # Here we chose the transpiling with the lower cost out of 'repeats' attempts
    cost = 1000000
    for _ in range(repeats):
        with warnings.catch_warnings():  # sabre causes deprecation warning, this will ignore them
            warnings.simplefilter("ignore")
            transpiled_circuit_tmp = transpile(circuit, **kwargs)
        new_cost = cost_func(transpiled_circuit_tmp)
        if print_cost:
            print('cost: ', new_cost)
        if new_cost < cost:
            cost = new_cost
            transpiled_circuit = transpiled_circuit_tmp
    return transpiled_circuit

def _add_custom_device_equivalences():
    """ Adds custom gate equivalences to the SessionEquivalenceLibrary for transpilation. 
    Is run automatically on running file.
    NOTE: Only needs to be run once!
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

    # Remove a superfluous decomposition of cx into iSWAPs
    # TODO: Maybe there's a solution where the transpiler instead prioritizes
    #       other decompositions rather than removing this one completely.
    cx_decompositions = SessionEquivalenceLibrary.get_entry(CXGate())
    cx_decompositions.pop(5)
    SessionEquivalenceLibrary.set_entry(CXGate(), cx_decompositions)

# This function will automatically run the first time you import this file
_add_custom_device_equivalences()
