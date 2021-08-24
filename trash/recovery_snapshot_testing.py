# %% Import modules
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import itertools
from qiskit import *

# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import thermal_relaxation_error
from qiskit.providers.aer.library import save_density_matrix, save_expectation_value                                        

from qiskit.quantum_info import partial_trace
from qiskit.quantum_info import DensityMatrix
from qiskit.quantum_info import state_fidelity

# Our own files
from simulator_program.custom_noise_models import (thermal_relaxation_model_V2,
    thermal_relaxation_model_V2,
    WACQT_target_times,
    WACQT_demonstrated_times,
    standard_times)
from simulator_program.data_analysis_tools import *
from simulator_program.custom_transpiler import *
from simulator_program.stabilizers import *
from simulator_program.post_select import *
from simulator_program.post_process import *
from simulator_program.idle_noise import *

#%%
n_shots = 10
n_cycles = 2
fid, time, res = fidelity_from_scratch(n_cycles, n_shots, gate_times={},
                                      data_process_type='post_process',
                                      encoding=False,
                                      transpile=False, snapshot_type='dm',
                                      extra_snapshot=True,
                                      move_feedback_delay=False,
                                      return_res=True)

#%%
#print(res.data()['dm_con_0.5'])
registers = StabilizerRegisters()

an0 = np.zeros(2**2)
an0[0] = 1.0
an0 = DensityMatrix(an0).data
dm_with_an = np.kron(an0, dm) 

trivial = get_encoded_state(0, 0, include_ancillas=None)
for cycle in range(n_cycles):
    print('\n\n', cycle)
    snap = res.data()['dm_con_'+str(cycle+0.5)]
    for key in snap.keys():
        dm = snap[key]
        print('\n',key)
        print(state_fidelity(dm, trivial))

        circ = get_empty_stabilizer_circuit(registers)
        circ.set_density_matrix(dm_with_an)
        #circ.set_density_matrix(get_encoded_state(0,0))
        circ.compose(get_stabilizer_cycle(registers),inplace=True)
        circ.draw(output='mpl')
        results = execute(
            circ,
            Aer.get_backend('qasm_simulator'),
            noise_model=None, shots=100).result()
        print(len(results.get_counts()))
        print(results.get_counts())

#%%


registers = StabilizerRegisters()
circ = get_empty_stabilizer_circuit(registers)
circ.set_density_matrix(dm_with_an)
circ.compose(get_stabilizer_cycle(registers),inplace=True)
circ.draw(output='mpl')



results = execute(
        circ,
        Aer.get_backend('qasm_simulator'),
        noise_model=None, shots=100).result()
print(len(results.get_counts()))
# %%
