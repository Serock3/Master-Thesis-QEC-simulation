#%% Import modules
import warnings
from qiskit import QuantumCircuit, QuantumRegister

#%% Functions

def logical_X(qubit: int, qbReg=QuantumRegister(4)):
    """Performs the logical X gate on a qubit encoded in the [[4,2,2]] code.
    
    Args:
        qubit (int): The logical qubit to act upon. Can be either 0, 1 or 2, 
                     where 2 corresponds to performing X on both logical qubits.
        qbReg: QuantumRegister class to build the circuit upon.
        
    Returns:
        circ: Circuit object for performing the X gate on a logical qubit.
    """
    circ = QuantumCircuit(qbReg)
    if qubit == 0:
        circ.x([qbReg[1], qbReg[3]])
    elif qubit == 1:
        circ.x([qbReg[2], qbReg[3]])
    elif qubit == 2:
        circ.x([qbReg[1], qbReg[2]])
    else:
        warnings.warn('Not a valid qubit input, returning empty circuit.')
    return circ

def logical_Z(qubit: int, qbReg=QuantumRegister(4)):
    """Performs the logical Z gate on a qubit encoded in the [[4,2,2]] code.
    
    Args:
        qubit (int): The logical qubit to act upon. Can be either 0, 1 or 2, 
                     where 2 corresponds to performing Z on both logical qubits.
        qbReg: QuantumRegister class to build the circuit upon.
        
    Returns:
        circ: Circuit object for performing the Z gate on a logical qubit.
    """
    circ = QuantumCircuit(qbReg)
    if qubit == 0:
        circ.z([qbReg[2], qbReg[3]])
    elif qubit == 1:
        circ.z([qbReg[1], qbReg[3]])
    elif qubit == 2:
        circ.z([qbReg[1], qbReg[2]])
    else:
        warnings.warn('Not a valid qubit input, returning empty circuit.')
    return circ

# TODO: Add logical Y
#def logical_Y(qubit: int, qbReg=QuantumRegister(4)):

# TODO: Add logical hadamard
#def logical_H(qubit: int, qbReg=QuantumRegister(4)):

def logical_CNOT(control_qubit, target_qubit, qbReg=QuantumRegister(4)):
    """Performs the logical CX gate on two qubits encoded in the [[4,2,2]] code.
    Both control and target (logical) qubit needs to be defined.
    
    Args:
        control_qubit (int): Control qubit for the CNOT gate, either 0 or 1.
        target_qubit (int): Target qubit for the CNOT gate, either 0 or 1.
        qbReg: QuantumRegister class to build the circuit upon.
        
    Returns:
        circ: Circuit object for performing the CNOT gate on the logical qubits.
    """
    circ = QuantumCircuit(qbReg)
    if control_qubit == 0 and target_qubit == 1:
        circ.swap(qbReg[2], qbReg[3])
    elif control_qubit == 1 and target_qubit == 0:
        circ.swap(qbReg[1], qbReg[3])
    else:
        warnings.warn('Not a valid qubit input, returning empty circuit.')

    return circ

def logical_CZ(qbReg=QuantumRegister(4)):
    """Performs the logical CZ gate on two qubits encoded in the [[4,2,2]] code.
    As the gate is symmetrical there is no need to specify target and control
    qubit since there is only one possibility.
    
    Args:
        qbReg: QuantumRegister class to build the circuit upon.
        
    Returns:
        circ: Circuit object for performing the CZ gate on the logical qubits.
    """
    circ = QuantumCircuit(qbReg)
    circ.cz(qbReg[1], qbReg[2])
    circ.cz(qbReg[0], qbReg[3])

    return circ

def logical_SWAP(qbReg=QuantumRegister(4)):
    """Performs the logical SWAP gate on qubits encoded in the [[4,2,2]] code.
    As the gate is symmetrical there is no need to specify target and control
    qubit since there is only one possibility.
    
    Args:
        qbReg: QuantumRegister class to build the circuit upon.
        
    Returns:
        circ: Circuit object for performing the SWAP gate on the logical qubits.
    """
    circ = QuantumCircuit(qbReg)
    circ.swap(qbReg[0],qbReg[3])

    return circ

def logical_iSWAP(qbReg=QuantumRegister(4)):
    """Performs the logical iSWAP gate on qubits encoded in the [[4,2,2]] code.
    As the gate is symmetrical there is no need to specify target and control
    qubit since there is only one possibility.
    
    Args:
        qbReg: QuantumRegister class to build the circuit upon.
        
    Returns:
        circ: Circuit object for performing an iSWAP gate on the logical qubits.
    """
    circ = QuantumCircuit(qbReg)
    circ.iswap(qbReg[0],qbReg[3]) 

    return circ

