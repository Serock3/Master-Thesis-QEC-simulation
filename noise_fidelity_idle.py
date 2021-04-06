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

# %%
def get_testing_circuit(registers, reset, n_cycles):
    circ = encode_input_v2(registers)
    circ.barrier()
    circ.append(Snapshot('post_encoding', 'density_matrix', num_qubits=5), registers.QubitRegister)
    circ += get_repeated_stabilization(registers, n_cycles=n_cycles, reset=reset,
    recovery=True, flag=False, snapshot_type='density_matrix')
    return circ

def get_standard_transpilation(circ):
    return shortest_transpile_from_distribution(circ, print_cost=False,
        repeats=10, routing_method='sabre', initial_layout=None,
        translation_method=None, layout_method='sabre',
        optimization_level=1, **WACQT_device_properties)

def get_running_fidelity_data_den_mat_mod(results, trivial_state, n_cycles):
    """Modified version of get_running_fidelity_data_den_mat used in this file
    for correcting errors"""
    # Missing: results, 
    fidelities = []
    cl_reg_size = len(list(results.get_counts().keys())[0].split()[1])
    counts = results.get_counts()
    snapshots = reformat_density_snapshot(results)
    for current_cycle in range(n_cycles):
        fid = 0
        for key in snapshots['stabilizer_'+str(current_cycle)]:
            bin_string = bin(int(key, 16))[2:].zfill(
                cl_reg_size*(current_cycle+1))[-cl_reg_size*(current_cycle+1):]
            current_state = snapshots['stabilizer_' +
                                      str(current_cycle)][key]
            for outcome in results.get_counts():
                formated_outcome = outcome.replace(
                    ' ', '')[-cl_reg_size*(current_cycle+1):]
                if formated_outcome == bin_string:
                    fid += state_fidelity(current_state,
                                          trivial_state)*counts[outcome]
        fidelities.append(fid/n_shots)
    return fidelities

def get_running_fidelity_idle_circuit(results, trivial_state, n_cycles):
    fidelities = []
    for current_cycle in range(n_cycles):
        current_state = results.data()['snapshots']['density_matrix'][
            'stabilizer_' + str(current_cycle)][0]['value']
        fidelities.append(state_fidelity(current_state, trivial_state))
    return fidelities
#%% Check fidelity with and without idle noise =================================
# Options
reset = True
recovery = True
flag = False
n_cycles = 15

# Registers
qb = QuantumRegister(5, 'code_qubit')
an = AncillaRegister(2, 'ancilla_qubit')
cr = get_classical_register(n_cycles, reset=False, recovery=recovery, flag=flag)
readout = ClassicalRegister(5, 'readout')
registers = StabilizerRegisters(qb, an, cr, readout)

# Circuits
circ = get_testing_circuit(registers, reset=False, n_cycles=n_cycles)
circ_res = get_testing_circuit(registers, reset=True, n_cycles=n_cycles)

# Transpilation
circ_t = get_standard_transpilation(circ)
circ_res_t = get_standard_transpilation(circ_res)

# Add idle noise
circ_i, time = add_idle_noise_to_circuit(circ, return_time=True)
circ_res_i, time_res = add_idle_noise_to_circuit(circ_res_t, return_time=True)

# Get the correct (no errors) state
trivial = get_trivial_state(circ)
trivial_t = get_trivial_state(circ_t)
trivial_res = get_trivial_state(circ_res_t)

circ_empty = get_empty_noisy_circuit(registers, time, encode_logical=True,
    transpile=False)
#circ_empty = get_empty_noisy_circuit_v2(circ_i, time, encode_logical=True)

#%% Run circuits
noise_model = thermal_relaxation_model()
n_shots = 512
print('Running simulations')
results = execute(circ, Aer.get_backend('qasm_simulator'),
    noise_model=noise_model, shots=n_shots).result()
fid = get_running_fidelity_data_den_mat_mod(results, trivial, n_cycles)
print('Check!')
results = execute(circ_empty, Aer.get_backend('qasm_simulator'),
    noise_model=noise_model, shots=n_shots).result()
fid_empty = get_running_fidelity_idle_circuit(results, trivial, n_cycles)
print('Check!')
results = execute(circ_i, Aer.get_backend('qasm_simulator'),
    noise_model=noise_model, shots=n_shots).result()
fid_i = get_running_fidelity_data_den_mat_mod(results, trivial, n_cycles)
print('Check!')
results = execute(circ_t, Aer.get_backend('qasm_simulator'),
    noise_model=noise_model, shots=n_shots).result()
fid_t = get_running_fidelity_data_den_mat_mod(results, trivial_t, n_cycles)
#%%print('Check!')
results = execute(circ_res_t, Aer.get_backend('qasm_simulator'),
    noise_model=noise_model, shots=n_shots).result()
fid_res = get_running_fidelity_data_den_mat_mod(results, trivial_res, n_cycles)
print('Check!')
results = execute(circ_res_i, Aer.get_backend('qasm_simulator'),
    noise_model=noise_model, shots=n_shots).result()
fid_res_i = get_running_fidelity_data_den_mat_mod(results, trivial_res, n_cycles)
print('Check!')


#%% Curve fitting with OLS
fid_list = [fid, fid_i, fid_res, fid_res_i]
theta_list = []
for fidelity in fid_list:
    x_D = np.ones((n_cycles,2))
    for i in range(n_cycles):
        x_D[i][1] += i
    y = np.log( np.reshape(np.asarray(fidelity), (n_cycles,1)) )
    theta = np.dot(np.dot(np.linalg.inv(np.dot(x_D.T, x_D)), x_D.T), y)
    theta_list.append(theta)

x = np.linspace(0,n_cycles+1,100)
y_pred_list = []
for theta in theta_list:
    y_pred = np.exp(theta[0]) * np.exp(x*theta[1])
    y_pred_list.append(y_pred)

# Calculate MSE
MSE = []
for i in range(len(fid_list)):
    err = 0.
    for cycle in range(n_cycles):
        y_pred = np.exp(theta_list[i][0]) * np.exp((cycle+1)*theta_list[i][1])
        err += (y_pred-fid_list[i][cycle])**2
    MSE.append(err)
print('MSE: ')
for i in MSE:
    print(i)
#%% Plotting
fig, ax = plt.subplots(1, figsize=(10, 6))
x_dis = np.arange(1,n_cycles+1)
ax.plot(x_dis, fid, 'o', color='red', label='QEC, no idle noise')
ax.plot(x_dis, fid_i, 'o', color='blue', label='QEC with idle noise')
ax.plot(x_dis, fid_empty, 'o', color='black', label='Empty encoded circuit')
ax.plot(x_dis, fid_t, 'o', color='green', label='Transpiled QEC, no idle noise')
#ax.plot(x_dis, fid_res, 'o', color='green')
#ax.plot(x_dis, fid_res_i, 'o', color='orange')

#ax.plot(x, y_pred_list[0], color='red', label=f'No idle noise. k={theta_list[0][1]}')
#ax.plot(x, y_pred_list[1], color='blue',label=f'With idle noise. k={theta_list[1][1]}')
#ax.plot(x, y_pred_list[2], color='green', label=f'No idle noise, with reset. k={theta_list[2][1]}')
#ax.plot(x, y_pred_list[3], color='orange', label=f'With idle noise and reset. k={theta_list[3][1]}')
ax.set_xlabel('Number of cycles')
ax.set_ylabel('Average fidelity')
ax.set_title('Average fidelites with no reset, no transpilation')
ax.legend()

#%% Other stuff
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



    
# %%
