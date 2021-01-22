# Import modules
%matplotlib inline
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller
from qiskit import(
    execute,
    Aer,
    QuantumRegister,
    ClassicalRegister
    )

def encode_input( circuit, qbReg ):
    '''Encode the input into logical 0 and 1
    This assumes that the 0:th qubit is the
    original state |psi> = a|0> + b|1>'''
    
    circuit.h( qbReg[4] )
    circuit.s( qbReg[4] )
    circuit.cz( qbReg[4], qbReg[3] )
    circuit.cz( qbReg[4], qbReg[1] )
    circuit.cy( qbReg[4], qbReg[0] )   

    circuit.h( qbReg[3] )
    circuit.cz( qbReg[3], qbReg[2] )
    circuit.cz( qbReg[3], qbReg[1] )
    circuit.cx( qbReg[3], qbReg[0] )
    
    circuit.h( qbReg[2] )
    circuit.cz( qbReg[2], qbReg[4] )
    circuit.cz( qbReg[2], qbReg[3] )
    circuit.cx( qbReg[2], qbReg[0] )
    
    circuit.h( qbReg[1] )
    circuit.s( qbReg[1] )
    circuit.cz( qbReg[1], qbReg[4] )
    circuit.cz( qbReg[1], qbReg[2] )
    circuit.cy( qbReg[1], qbReg[0] )
    
    return

q = QuantumRegister(7, 'code_qubit')
circ = QuantumCircuit(q)
encode_input( circ, q )
circ.draw(output='mpl')
pass_ = Unroller(basis_gates= ['iswap', 'cz' ,'u1', 'u2', 'u3', 'h', 'x', 'z', 'y','s','t', 'cx'],
    coupling_map = [[0,1],[0,6],[1,6],[6,2],[6,3],[2,3],[6,4],[6,5],[4,5]])
pm = PassManager(pass_)
new_circ = pm.run(circ)
new_circ.draw(output='mpl')
