# %%
import matplotlib.pyplot as plt
import numpy as np

from simulator_program.data_analysis_tools import *


T1 = 40e3


def monoExp(t, T, c, A):
    return (A-c) * np.exp(-t/T) + c


def idealExp(t, T):
    return 0.5 * np.exp(-t/T) + 0.5


def_colors = plt.get_cmap("tab10")
p0 = (T1, 0.4, 0.9)  # start with values near those we expect*
colors = plt.get_cmap("tab10")
# %% Test run
n_curves = 4
n_cycles = 20
n_shots = 1024*8

F_Ls = []
P_Ls = []
times = []
t_fs = np.linspace(0, 6000, n_curves)
for t_f in t_fs:
    F_L, P_L, time = fidelity_from_scratch(n_cycles, n_shots, gate_times={
        'feedback': t_f}, encoding=False, transpile=False, project=True)
    time = np.array([time['dm_'+str(i)] for i in range(n_cycles+1)])
    F_L = np.array(F_L)
    P_L = np.array(P_L)
    F_Ls.append(F_L)
    P_Ls.append(P_L)
    times.append(time)

    pars, cov = scipy.optimize.curve_fit(
        monoExp, time[1:], (F_L*P_L)[1:], p0)
    pars_L, cov_L = scipy.optimize.curve_fit(
        idealExp, time[1:]-time[1], F_L[1:], 60e3)

    print('t_f= ', t_f)
    print('T=', np.round(pars[0]/1000, 1))
    print('c=', np.round(pars[1], 2))
    print('A=', np.round(pars[2], 2), '\n')
    plt.plot(time/1000, F_L *
             P_L, 'o', label='F', color=def_colors(0))
    plt.plot(time/1000, F_L,
             '^', label='F_L', color=def_colors(2))
    plt.plot(time/1000, P_L,
             's', label='P_L', color=def_colors(3))

    plt.plot(time/1000, monoExp(time, *pars), '--', color=def_colors(0))
    plt.plot(time/1000, idealExp(time -
                                 time[1], *pars_L), '--', color=def_colors(2))
    plt.plot(time/1000, monoExp(time, *pars) /
             idealExp(time-time[1], *pars_L), '--', color=def_colors(3))
    # plt.title(r'$T_1$ = ' + str(int(T1_span[i]/1000))+r' $T_2$ = ' + str(int(T2_span[j]/1000))+' \n-> F: T='+str(
    #     np.round(pars[0]/1000, 1))+' c='+str(np.round(pars[1], 2))+' A='+str(np.round(pars[2], 2)) +
    #     '\n F_L: '+'T='+str(np.round(pars_L[0]/1000, 1)))
    plt.legend()
    plt.show()
# %% 
for i, (F_L, P_L, time) in enumerate(zip(F_Ls, P_Ls, times)):
    # pars, cov = scipy.optimize.curve_fit(monoExp, time[1:], F_L[1:], p0)
    pars, cov = scipy.optimize.curve_fit(
        monoExp, time[1:], (F_L*P_L)[1:], p0)
    pars_L, cov_L = scipy.optimize.curve_fit(
        idealExp, time[1:]-time[1], F_L[1:], 60e3)

    print('t_f= ', t_fs[i])
    print('T=', np.round(pars[0]/1000, 1))
    print('c=', np.round(pars[1], 2))
    print('A=', np.round(pars[2], 2), '\n')

    plt.plot(time/1000, F_L *
             P_L, 'o', color=colors(i), label=r'$t_f$ = ' + str(int(t_fs[i]))+' -> T='+str(
                 np.round(pars[0]/1000, 1))+' c='+str(np.round(pars[1], 2))+' A='+str(np.round(pars[2], 2)))
    plt.plot(time/1000, monoExp(time, *pars), '--', color=def_colors(i))

plt.title(r'Curvefitting to $(A-c)e^{-t/T}+c$ for different feedback times')
plt.xlabel('Time [μs]')
plt.ylabel('Probability of remaining in initial state')
plt.legend()
plt.show()

# %% 
for i, (F_L, P_L, time) in enumerate(zip(F_Ls, P_Ls, times)):
    # pars, cov = scipy.optimize.curve_fit(monoExp, time[1:], F_L[1:], p0)
    pars, cov = scipy.optimize.curve_fit(
        monoExp, time[1:], (F_L*P_L)[1:], p0)
    pars_L, cov_L = scipy.optimize.curve_fit(
        idealExp, time[1:]-time[1], F_L[1:], 60e3)

    print('t_f= ', t_fs[i])
    print('T=', np.round(pars_L[0]/1000, 1), '\n')
    plt.plot(time/1000, F_L , 'o', color=colors(i), label=r'$t_f$ = ' + str(int(t_fs[i]))+' -> T='+str(
                 np.round(pars_L[0]/1000, 1)))

    plt.plot(time/1000, idealExp(time - time[1], *pars_L), '--', color=def_colors(i))
                                 

plt.title(r'Curvefitting to $(A-c)e^{-t/T}+c$ for different feedback times')
plt.xlabel('Time [μs]')
plt.ylabel('Probability of remaining in initial state')
plt.legend()
plt.show()

# %%
