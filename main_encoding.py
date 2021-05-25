# Main file for all simulations of encoding fidelity
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
from simulator_program.custom_noise_models import (thermal_relaxation_model,
                                                   thermal_relaxation_model_V2,
                                                   WACQT_target_times,
                                                   WACQT_demonstrated_times,
                                                   standard_times)
from simulator_program.data_analysis_tools import *

# %%

# Full connectivity
fid_0_FC, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                idle_noise=True, theta=0, phi=0, iswap=True,
                                snapshot_type='dm', device=None, pauliop='ZZZZZ')
fid_1_FC, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                idle_noise=True, theta=np.pi, phi=0, iswap=True,
                                snapshot_type='dm', device=None, pauliop='ZZZZZ')
fid_p_FC, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                idle_noise=True, theta=np.pi/2, phi=0, iswap=True,
                                snapshot_type='dm', device=None, pauliop='ZZZZZ')
fid_m_FC, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                idle_noise=True, theta=np.pi/2, phi=np.pi, iswap=True,
                                snapshot_type='dm', device=None, pauliop='ZZZZZ')

# Hexagonal, with iSWAP
fid_0_hex, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                 idle_noise=True, theta=0, phi=0, iswap=True,
                                 snapshot_type='dm', device='WACQT', pauliop='ZZZZZ')
fid_1_hex, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                 idle_noise=True, theta=np.pi, phi=0, iswap=True,
                                 snapshot_type='dm', device='WACQT', pauliop='ZZZZZ')
fid_p_hex, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                 idle_noise=True, theta=np.pi/2, phi=0, iswap=True,
                                 snapshot_type='dm', device='WACQT', pauliop='ZZZZZ')
fid_m_hex, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                 idle_noise=True, theta=np.pi/2, phi=np.pi, iswap=True,
                                 snapshot_type='dm', device='WACQT', pauliop='ZZZZZ')

# Hexagonal, no iSWAP
fid_0_hex_cz, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                    idle_noise=True, theta=0, phi=0, iswap=False,
                                    snapshot_type='dm', device='WACQT', pauliop='ZZZZZ')
fid_1_hex_cz, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                    idle_noise=True, theta=np.pi, phi=0, iswap=False,
                                    snapshot_type='dm', device='WACQT', pauliop='ZZZZZ')
fid_p_hex_cz, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                    idle_noise=True, theta=np.pi/2, phi=0, iswap=False,
                                    snapshot_type='dm', device='WACQT', pauliop='ZZZZZ')
fid_m_hex_cz, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                    idle_noise=True, theta=np.pi/2, phi=np.pi, iswap=False,
                                    snapshot_type='dm', device='WACQT', pauliop='ZZZZZ')

# Double diamond, with iSWAP
fid_0_DD, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                idle_noise=True, theta=0, phi=0, iswap=True,
                                snapshot_type='dm', device='DD', pauliop='ZZZZZ')
fid_1_DD, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                idle_noise=True, theta=np.pi, phi=0, iswap=True,
                                snapshot_type='dm', device='DD', pauliop='ZZZZZ')
fid_p_DD, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                idle_noise=True, theta=np.pi/2, phi=0, iswap=True,
                                snapshot_type='dm', device='DD', pauliop='ZZZZZ')
fid_m_DD, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                idle_noise=True, theta=np.pi/2, phi=np.pi, iswap=True,
                                snapshot_type='dm', device='DD', pauliop='ZZZZZ')

# Double diamond, no iSWAP
fid_0_DD_cz, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                   idle_noise=True, theta=0, phi=0, iswap=False,
                                   snapshot_type='dm', device='DD', pauliop='ZZZZZ')
fid_1_DD_cz, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                   idle_noise=True, theta=np.pi, phi=0, iswap=False,
                                   snapshot_type='dm', device='DD', pauliop='ZZZZZ')
fid_p_DD_cz, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                   idle_noise=True, theta=np.pi/2, phi=0, iswap=False,
                                   snapshot_type='dm', device='DD', pauliop='ZZZZZ')
fid_m_DD_cz, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                   idle_noise=True, theta=np.pi/2, phi=np.pi, iswap=False,
                                   snapshot_type='dm', device='DD', pauliop='ZZZZZ')
#%%

print(*[np.round(f,3) for f in [fid_0_FC,fid_1_FC,fid_p_FC,fid_m_FC]],sep=' & ')
print(*[np.round(f,3) for f in [fid_0_hex,fid_1_hex,fid_p_hex,fid_m_hex]],sep=' & ')
print(*[np.round(f,3) for f in [fid_0_hex_cz,fid_1_hex_cz,fid_p_hex_cz,fid_m_hex_cz]],sep=' & ')
print(*[np.round(f,3) for f in [fid_0_DD,fid_1_DD,fid_p_DD,fid_m_DD]],sep=' & ')
print(*[np.round(f,3) for f in [fid_0_DD_cz,fid_1_DD_cz,fid_p_DD_cz,fid_m_DD_cz]],sep=' & ')
#%% Projected 
project = True
# Full connectivity
fid_0_FC, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                idle_noise=True, theta=0, phi=0, iswap=True,
                                snapshot_type='dm', device=None, pauliop='ZZZZZ',project=project)
fid_1_FC, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                idle_noise=True, theta=np.pi, phi=0, iswap=True,
                                snapshot_type='dm', device=None, pauliop='ZZZZZ',project=project)
fid_p_FC, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                idle_noise=True, theta=np.pi/2, phi=0, iswap=True,
                                snapshot_type='dm', device=None, pauliop='ZZZZZ',project=project)
fid_m_FC, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                idle_noise=True, theta=np.pi/2, phi=np.pi, iswap=True,
                                snapshot_type='dm', device=None, pauliop='ZZZZZ',project=project)

# Hexagonal, with iSWAP
fid_0_hex, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                 idle_noise=True, theta=0, phi=0, iswap=True,
                                 snapshot_type='dm', device='WACQT', pauliop='ZZZZZ',project=project)
fid_1_hex, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                 idle_noise=True, theta=np.pi, phi=0, iswap=True,
                                 snapshot_type='dm', device='WACQT', pauliop='ZZZZZ',project=project)
fid_p_hex, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                 idle_noise=True, theta=np.pi/2, phi=0, iswap=True,
                                 snapshot_type='dm', device='WACQT', pauliop='ZZZZZ',project=project)
fid_m_hex, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                 idle_noise=True, theta=np.pi/2, phi=np.pi, iswap=True,
                                 snapshot_type='dm', device='WACQT', pauliop='ZZZZZ',project=project)

# Hexagonal, no iSWAP
fid_0_hex_cz, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                    idle_noise=True, theta=0, phi=0, iswap=False,
                                    snapshot_type='dm', device='WACQT', pauliop='ZZZZZ',project=project)
fid_1_hex_cz, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                    idle_noise=True, theta=np.pi, phi=0, iswap=False,
                                    snapshot_type='dm', device='WACQT', pauliop='ZZZZZ',project=project)
fid_p_hex_cz, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                    idle_noise=True, theta=np.pi/2, phi=0, iswap=False,
                                    snapshot_type='dm', device='WACQT', pauliop='ZZZZZ',project=project)
fid_m_hex_cz, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                    idle_noise=True, theta=np.pi/2, phi=np.pi, iswap=False,
                                    snapshot_type='dm', device='WACQT', pauliop='ZZZZZ',project=project)

# Double diamond, with iSWAP
fid_0_DD, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                idle_noise=True, theta=0, phi=0, iswap=True,
                                snapshot_type='dm', device='DD', pauliop='ZZZZZ',project=project)
fid_1_DD, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                idle_noise=True, theta=np.pi, phi=0, iswap=True,
                                snapshot_type='dm', device='DD', pauliop='ZZZZZ',project=project)
fid_p_DD, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                idle_noise=True, theta=np.pi/2, phi=0, iswap=True,
                                snapshot_type='dm', device='DD', pauliop='ZZZZZ',project=project)
fid_m_DD, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                idle_noise=True, theta=np.pi/2, phi=np.pi, iswap=True,
                                snapshot_type='dm', device='DD', pauliop='ZZZZZ',project=project)

# Double diamond, no iSWAP
fid_0_DD_cz, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                   idle_noise=True, theta=0, phi=0, iswap=False,
                                   snapshot_type='dm', device='DD', pauliop='ZZZZZ',project=project)
fid_1_DD_cz, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                   idle_noise=True, theta=np.pi, phi=0, iswap=False,
                                   snapshot_type='dm', device='DD', pauliop='ZZZZZ',project=project)
fid_p_DD_cz, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                   idle_noise=True, theta=np.pi/2, phi=0, iswap=False,
                                   snapshot_type='dm', device='DD', pauliop='ZZZZZ',project=project)
fid_m_DD_cz, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                   idle_noise=True, theta=np.pi/2, phi=np.pi, iswap=False,
                                   snapshot_type='dm', device='DD', pauliop='ZZZZZ',project=project)

#%%

print(*[np.round(f,3) for f in [fid_0_FC,fid_1_FC,fid_p_FC,fid_m_FC]],sep=' & ')
print(*[np.round(f,3) for f in [fid_0_hex,fid_1_hex,fid_p_hex,fid_m_hex]],sep=' & ')
print(*[np.round(f,3) for f in [fid_0_hex_cz,fid_1_hex_cz,fid_p_hex_cz,fid_m_hex_cz]],sep=' & ')
print(*[np.round(f,3) for f in [fid_0_DD,fid_1_DD,fid_p_DD,fid_m_DD]],sep=' & ')
print(*[np.round(f,3) for f in [fid_0_DD_cz,fid_1_DD_cz,fid_p_DD_cz,fid_m_DD_cz]],sep=' & ')
# %%
fid_0_hex_cz, _, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                    idle_noise=True, theta=0, phi=0, iswap=False,
                                    snapshot_type='dm', device='WACQT', pauliop='ZZZZZ',project=project)
# %%
