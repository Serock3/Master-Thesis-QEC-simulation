from qiskit.compiler import transpile
from qiskit.transpiler import PassManager,CouplingMap,Layout
from qiskit.visualization import plot_circuit_layout

def transpile_circuit( circuit, qb, an ):
    basis_gates = ['id', 'u1', 'u2', 'u3', 'iswap','cz']
    couplinglist=[[0, 1],[0,6],[1,6],[2,3],[2,6],[3,6],[4,5],[4,6],[5,6]]
    reverse_couplinglist = [[y,x] for [x,y] in couplinglist]
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
    optimization_level=2

    transpiled_circuit = transpile(
        circuit,
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        optimization_level=optimization_level,
        initial_layout=layout
        )

    
    return transpiled_circuit