# %% Import modules
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from qiskit import *
from qiskit.visualization import plot_histogram

# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import thermal_relaxation_error
                                        

from qiskit.quantum_info import partial_trace
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import state_fidelity

# Our own files
from simulator_program.custom_noise_models import thermal_relaxation_model
from simulator_program.custom_transpiler import *
from simulator_program.stabilizers import *
from simulator_program.post_select import *
from idle_noise import *

#%% Check fidelity with and without idle noise =================================
# Options
reset = True
recovery = True
flag = False
n_cycles = 15

# Registers
qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
cr = get_classical_register(n_cycles, reset=False, recovery=True, flag=False) # Advanced list of registers
readout = ClassicalRegister(5, 'readout')
registers = StabilizerRegisters(qb, an, cr, readout)

# Circuit
circ = encode_input_v2(registers)
circ += get_repeated_stabilization(registers, n_cycles=n_cycles, reset=reset,
    recovery=recovery, flag=flag, snapshot_type='density_matrix')

# Transpilation
circ_t = shortest_transpile_from_distribution(circ, print_cost=False,
    repeats=10, routing_method='sabre', initial_layout=None,
    translation_method=None, layout_method='sabre',
    optimization_level=1, **WAQCT_device_properties)

# Add idle noise
circ_i = add_idle_noise_to_circuit(circ_t)

# Run circuits
n_shots = 128
fid = get_running_fidelity_data_den_mat(circ_t, n_cycles, n_shots=n_shots,
    noise_model=thermal_relaxation_model(), post_select=False)
fid_idle = get_running_fidelity_data_den_mat(circ_i, n_cycles, n_shots=n_shots,
    noise_model=thermal_relaxation_model(), post_select=False)

#%% Plotting
fig, ax = plt.subplots(1, figsize=(10, 6))
x_dis = np.arange(1,n_cycles+1)
ax.plot(x_dis, fid, 'o', color='red', label='No idle noise')
ax.plot(x_dis, fid_idle, 'o', color='blue', label='With idle noise')
ax.legend()

