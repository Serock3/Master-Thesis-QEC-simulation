# Main file for all simulations of encoding fidelity
# %% Import modules
from types import resolve_bases
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
resolution = 5
T2_span = np.linspace(30e3, 100e3, resolution)
T1_span = np.linspace(30e3, 100e3, resolution)
fid_span = np.zeros((resolution, resolution))
time_span = np.zeros((resolution, resolution))
for i in range(resolution):
    for j in range(resolution):
        if T2_span[j] >= T1_span[i]:
            fid_span[i][j] = -1 
            time_span[i][j] = -1
            continue
        fid_span[i][j], _, time_span[i][j] = encoding_fidelity(1, gate_times=WACQT_target_times, T1=T1_span[i], T2=T2_span[j],
                                                               theta=0, phi=0, device='WACQT')

# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
colors = plt.get_cmap("tab10")
X_mesh, Y_mesh = np.meshgrid(T1_span, T2_span)
ax.contourf(X_mesh, Y_mesh, fid_span)

#%% Compare T1 and T2
X_mesh, Y_mesh = np.meshgrid(T1_span, T2_span)
E_mesh1 = -1*np.reshape(fid_span, (resolution,resolution)).T
E_mesh2 = -1*np.reshape(rates_T1T2_feedback, (7,7)).T

# Convert to lifetime (Outdated since new sweeps does this automatically)
for i in range(len(E_mesh1)):
    for j in range(len(E_mesh1)):
        if E_mesh1[i,j] != 0.:
            E_mesh1[i,j] = E_mesh1[i,j]**-1
        if E_mesh2[i,j] != 0.:
            E_mesh2[i,j] = E_mesh2[i,j]**-1

E_list = [E_mesh1, E_mesh2]

Nr = 1
Nc = 2
fig, ax = plt.subplots(Nr,Nc, figsize=(10,4))
maps = []
for i in range(Nc):
    HM = ax[i].imshow(E_list[i], extent=[35,105,35,105], cmap='magma',origin='lower')
    maps.append(HM)
    ax[i].set_xticks([40, 50, 60, 70, 80, 90, 100])
    ax[i].set_yticks([100, 90, 80, 70, 60, 50, 40])
ax[1].set_xlabel(r'$T_1$ [$\mu s$]')
ax[0].set_xlabel(r'$T_1$ [$\mu s$]')
ax[1].set_ylabel(r'$T_2$ [$\mu s$]')
ax[0].set_ylabel(r'$T_2$ [$\mu s$]')

ax[0].set_title(r'Standard QEC')
ax[1].set_title(r'Perfect syndrome extraction')

# Normalize color scale not to include the zeros at T2 > 2T1
for i in range(Nc):
    E_min = np.min(np.array([np.min(E_list[i][E_list[i]!=0])]))
    E_max = np.max(np.array([np.max(E_list[i][E_list[i]!=0])]))
    norm = colors.Normalize(vmin=E_min, vmax=E_max)
    maps[i].set_norm(norm)

cbar0 = fig.colorbar(maps[0], ax=ax[0], orientation='horizontal', fraction=.1, pad=0.17)
cbar0.set_label(r'Qubit lifetime $[\mu s]$', labelpad=0, y=1.20, rotation=0)

cbar1 = fig.colorbar(maps[1], ax=ax[1], orientation='horizontal', fraction=.1, pad=0.17)
cbar1.set_label(r'Qubit lifetime $[\mu s]$', labelpad=0, y=1.20, rotation=0)

# %%
# Full connectivity
fid_0_FC, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                idle_noise=True, theta=0, phi=0, iswap=True,
                                snapshot_type='dm', device=None, pauliop='ZZZZZ')
fid_1_FC, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                idle_noise=True, theta=np.pi, phi=0, iswap=True,
                                snapshot_type='dm', device=None, pauliop='ZZZZZ')
fid_p_FC, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                idle_noise=True, theta=np.pi/2, phi=0, iswap=True,
                                snapshot_type='dm', device=None, pauliop='ZZZZZ')
fid_m_FC, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                idle_noise=True, theta=np.pi/2, phi=np.pi, iswap=True,
                                snapshot_type='dm', device=None, pauliop='ZZZZZ')

# Hexagonal, with iSWAP
fid_0_hex, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                 idle_noise=True, theta=0, phi=0, iswap=True,
                                 snapshot_type='dm', device='WACQT', pauliop='ZZZZZ')
fid_1_hex, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                 idle_noise=True, theta=np.pi, phi=0, iswap=True,
                                 snapshot_type='dm', device='WACQT', pauliop='ZZZZZ')
fid_p_hex, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                 idle_noise=True, theta=np.pi/2, phi=0, iswap=True,
                                 snapshot_type='dm', device='WACQT', pauliop='ZZZZZ')
fid_m_hex, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                 idle_noise=True, theta=np.pi/2, phi=np.pi, iswap=True,
                                 snapshot_type='dm', device='WACQT', pauliop='ZZZZZ')

# Hexagonal, no iSWAP
fid_0_hex_cz, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                    idle_noise=True, theta=0, phi=0, iswap=False,
                                    snapshot_type='dm', device='WACQT', pauliop='ZZZZZ')
fid_1_hex_cz, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                    idle_noise=True, theta=np.pi, phi=0, iswap=False,
                                    snapshot_type='dm', device='WACQT', pauliop='ZZZZZ')
fid_p_hex_cz, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                    idle_noise=True, theta=np.pi/2, phi=0, iswap=False,
                                    snapshot_type='dm', device='WACQT', pauliop='ZZZZZ')
fid_m_hex_cz, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                    idle_noise=True, theta=np.pi/2, phi=np.pi, iswap=False,
                                    snapshot_type='dm', device='WACQT', pauliop='ZZZZZ')

# Double diamond, with iSWAP
fid_0_DD, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                idle_noise=True, theta=0, phi=0, iswap=True,
                                snapshot_type='dm', device='DD', pauliop='ZZZZZ')
fid_1_DD, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                idle_noise=True, theta=np.pi, phi=0, iswap=True,
                                snapshot_type='dm', device='DD', pauliop='ZZZZZ')
fid_p_DD, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                idle_noise=True, theta=np.pi/2, phi=0, iswap=True,
                                snapshot_type='dm', device='DD', pauliop='ZZZZZ')
fid_m_DD, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                idle_noise=True, theta=np.pi/2, phi=np.pi, iswap=True,
                                snapshot_type='dm', device='DD', pauliop='ZZZZZ')

# Double diamond, no iSWAP
fid_0_DD_cz, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                   idle_noise=True, theta=0, phi=0, iswap=False,
                                   snapshot_type='dm', device='DD', pauliop='ZZZZZ')
fid_1_DD_cz, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                   idle_noise=True, theta=np.pi, phi=0, iswap=False,
                                   snapshot_type='dm', device='DD', pauliop='ZZZZZ')
fid_p_DD_cz, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                   idle_noise=True, theta=np.pi/2, phi=0, iswap=False,
                                   snapshot_type='dm', device='DD', pauliop='ZZZZZ')
fid_m_DD_cz, _ = encoding_fidelity(1, gate_times=WACQT_target_times, T1=40e3, T2=60e3,
                                   idle_noise=True, theta=np.pi/2, phi=np.pi, iswap=False,
                                   snapshot_type='dm', device='DD', pauliop='ZZZZZ')
