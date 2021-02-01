from qiskit.compiler import transpile
from qiskit.transpiler import PassManager, CouplingMap, Layout
from qiskit.visualization import plot_circuit_layout
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.circuit.library.standard_gates import iSwapGate, SwapGate, SGate, CZGate
from qiskit import QuantumCircuit, QuantumRegister
import warnings


def transpile_circuit(circuit, qb, an):
    basis_gates = ['id', 'u1', 'u2', 'u3', 'iswap', 'cz']
    couplinglist = [[0, 1], [0, 6], [1, 6], [2, 3],
                    [2, 6], [3, 6], [4, 5], [4, 6], [5, 6]]
    reverse_couplinglist = [[y, x] for [x, y] in couplinglist]
    coupling_map = CouplingMap(
        couplinglist=couplinglist,
        description='A hexagoal 7qb code with two ancillas'
    )

    layout = Layout(
        {qb[0]: 0,
         qb[1]: 1,
            qb[2]: 2,
            qb[3]: 3,
            qb[4]: 4,
            an[0]: 5,
            an[1]: 6})
    optimization_level = 2

    transpiled_circuit = transpile(
        circuit,
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        optimization_level=optimization_level,
        initial_layout=layout
    )

    return transpiled_circuit


def shortest_transpile_from_distribution(circuit, repeats=40, print_depths=True, **kwargs):
    depth = 10000
    for i in range(repeats):
        with warnings.catch_warnings():  # sabre causes deprication warning, this will ignore them
            warnings.simplefilter("ignore")
            transpiled_circuit_tmp = transpile(circuit, **kwargs)
        if print_depths:
            print('depth: ', transpiled_circuit_tmp.depth())
        if transpiled_circuit_tmp.depth() < depth:
            depth = transpiled_circuit_tmp.depth()
            transpiled_circuit = transpiled_circuit_tmp
    return transpiled_circuit


basis_gates = ['id', 'u1', 'u2', 'u3', 'iswap', 'cz']
couplinglist = [[0, 1], [0, 6], [1, 6], [2, 3],
                [2, 6], [3, 6], [4, 5], [4, 6], [5, 6]]
reverse_couplinglist = [[y, x] for [x, y] in couplinglist]
coupling_map = CouplingMap(
    couplinglist=couplinglist, description='A hexagoal 7qb code with two ancillas')

# Dict with device properties of the WAQCT QC to be used for transpilation.
WAQCT_device_properties = {
    "basis_gates": basis_gates, "coupling_map": coupling_map}

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


# This function will automatically run the first time you import this file
_add_custom_device_equivalences()
