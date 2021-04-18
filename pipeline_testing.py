#%% Import modules
import numpy as np
from qiskit import *
from qiskit.visualization import plot_histogram

from simulator_program.stabilizers import *
from simulator_program.custom_transpiler import *
from simulator_program.idle_noise import *
from simulator_program.custom_noise_models import (thermal_relaxation_model,
    thermal_relaxation_model_V2,
    WACQT_target_times,
    WACQT_demonstrated_times)

from qiskit.quantum_info import Pauli
#%% Compare cycle times: Pipeline vs normal
n_cycles=2
recovery=False
flag=True
reset=True
# Registers
qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
cr = get_classical_register(n_cycles, reset=reset, recovery=recovery, flag=flag)
readout = ClassicalRegister(5, 'readout')
registers = StabilizerRegisters(qb, an, cr, readout)

# Circuits
circ = get_empty_stabilizer_circuit(registers)

circ.compose(encode_input_v2(registers),inplace=True)
circ.barrier()
circ.append(Snapshot('post_encoding', 'density_matrix', num_qubits=5), registers.QubitRegister)
circ.compose(get_repeated_stabilization(registers, n_cycles=n_cycles, reset=reset, 
    recovery=recovery, flag=flag),inplace=True)


#%%
circ.compose(get_repeated_stabilization(registers, n_cycles=n_cycles, reset=reset, 
    recovery=recovery, flag=flag, include_barriers=False, pipeline=True),inplace=True)

#circ += get_repeated_stabilization(registers, n_cycles=n_cycles,
#    reset=reset, recovery=recovery, flag=flag, include_barriers=False, pipeline=True)
circ = shortest_transpile_from_distribution(circ, print_cost=False)
#circ, time = add_idle_noise_to_circuit(circ, gate_times=WACQT_demonstrated_times,
#    return_time=True, rename=False)

n_shots = 1024
noise_model = thermal_relaxation_model_V2(gate_times=WACQT_demonstrated_times)
results = execute(circ, Aer.get_backend('qasm_simulator'),
        noise_model=noise_model, shots=n_shots).result()

time_diff = 0
a = True
for key in time:
    current_time = time[key]
    if a:
        a = False
        previous_time = time[key]
        continue
    time_diff = current_time - previous_time
    print(time_diff)
    previous_time = time[key]

#%% Testing expectation values
n_cycles=9
recovery=False
flag=False
reset=True
# Registers
qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
cr = get_classical_register(n_cycles, reset=False, recovery=False, flag=False)
readout = ClassicalRegister(5, 'readout')
registers = StabilizerRegisters(qb, an, cr, readout)

# Circuits
circ = get_empty_stabilizer_circuit(registers)
#circ.x(0) #initialize in 1
#circ += encode_input_v2(registers)
circ.compose(encode_input_v2(registers),inplace=True)
circ.barrier()
#for current_cycle in range(n_cycles):
#    circ.compose(unflagged_stabilizer_cycle(registers, reset=reset,
#        current_cycle=current_cycle), inplace=True)
#    circ.save_expectation_value(Pauli('ZZZZZ'), qb, label='exp_value_'+str(current_cycle))
circ.compose(get_repeated_stabilization(registers, n_cycles=n_cycles, reset=reset, 
    recovery=recovery, flag=flag, snapshot_type='expectation_value'),inplace=True)


circ, time = add_idle_noise_to_circuit(circ, gate_times=WACQT_demonstrated_times,
    return_time=True, rename=False)
noise_model = thermal_relaxation_model_V2(gate_times=WACQT_demonstrated_times)
results = execute(circ, Aer.get_backend('qasm_simulator'),
        noise_model=noise_model, shots=2048).result()
#%circ.draw()
#print(results.data()['expectation_value'])
for current_cycle in range(n_cycles):
    print(results.data()['exp_value_'+str(current_cycle)])