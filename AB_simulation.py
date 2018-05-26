from random import randint, random, shuffle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from num_analysis import analytical, inverse
from num_integration import euler
from AB_model import Agent, Group, Pop


def basic(T, n_groups, a, n, h0):
    Agent.H = 1.0
    Agent.L = 0.0

    Agent.a = a
    Group.n = n

    Pop.h0 = h0

    n_agents = Group.n * n_groups

    alpha = 1 / (abs(1 / Group.n - Agent.a) + (Group.n - 1) / Group.n) / (Agent.H - Agent.L)

    pop = Pop(n_agents)
    groups = [Group() for _ in range(n_groups)]

    h = np.zeros(T + 1)
    marks = np.zeros((T, n_groups))
    copyh = np.zeros(T)
    copyl = np.zeros(T)

    h[0] = pop.get_h()

    for t in range(T):
        shuffle(pop.agents)
        available_groups_ind = list(range(n_groups))

        for agent in pop.agents:
            chosen_group_ind = available_groups_ind[randint(0, len(available_groups_ind) - 1)]
            group = groups[chosen_group_ind]
            group.add_agent(agent)
            if group.isfull():
                available_groups_ind.remove(chosen_group_ind)

        for j, group in enumerate(groups):
            marks[t, j] = group.give_marks()
            group.agents = []

        for agent in pop.agents:
            rand_ind = randint(0, n_agents - 1)
            while agent == pop.agents[rand_ind]:
                rand_ind = randint(0, n_agents - 1)
            reference = pop.agents[rand_ind]

            agent.copy_payoff = reference.payoff
            agent.copy_S = reference.S

        for agent in pop.agents:
            if alpha * (agent.copy_payoff - agent.payoff) > random():
                if agent.S == Agent.H and agent.copy_S == Agent.L:
                    copyl[t] += 1
                elif agent.S == Agent.L and agent.copy_S == Agent.H:
                    copyh[t] += 1
                agent.S = agent.copy_S

        h[t + 1] = pop.get_h()

    return marks, h, copyh, copyl


def plot_marks(marks, T, n_iter, ax1):
    yticks = np.arange(7)
    for i, y in enumerate(yticks):
        yticks[i] = y * Agent.H + (Group.n - y) * Agent.L
    yticks = yticks / Group.n

    a = np.zeros((n_iter, T, 7))
    for row in range(T):
        for i in range(n_iter):
            a[i, row] = np.histogram(marks[i, row, :], 7, (Agent.L, Agent.H))[0]

    a = a.mean(axis=0)

    labels = ["{:.2f}".format(ytick) for ytick in yticks]
    ax1.stackplot(range(1, T + 1), a.T, labels=labels)

    ax1.set_xlabel('t')
    ax1.set_ylabel('number of groups')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[::-1], labels[::-1], loc='lower right')


def run_simple(h0):
    T = 20
    n_groups = 20
    a = 0.5
    n = 6

    n_iter = 100
    tvec = np.arange(T + 1)
    ab = np.zeros((n_iter, T + 1))
    marks = np.zeros((n_iter, T, n_groups))
    copyh = np.zeros((n_iter, T))
    copyl = np.zeros((n_iter, T))

    # Running the agent based
    for i in range(n_iter):
        marks[i, :, :], ab[i, :], copyh[i, :], copyl[i, :] = basic(T, n_groups, a, n, h0)
    std = ab.std(axis=0)
    ab = ab.mean(axis=0)

    # Running analytical and numerical integration
    ana = analytical(tvec, h0, a, n)
    _, num = euler(T, a, n, step=1, h0=h0)

    # Plotting h
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    ax = axes[0]
    ax.errorbar(tvec, ab, yerr=std, label='agent-based')
    ax.plot(ana, label='analytical')
    ax.plot(num, label='numerical')
    ax.set(xlabel="t", ylabel="h")
    ax.legend()

    plot_marks(marks, T, n_iter, axes[1])
    fig.tight_layout()
    fig.show()

    # Plotting copy evolution
    fig1, ax = plt.subplots(figsize=(7, 3))

    ax.plot(copyh.mean(axis=0), label='H copying')
    ax.plot(copyl.mean(axis=0), label='L copying')
    ax.set(xlabel="t", ylabel="n")
    ax.legend()

    # fig1.show()

    return marks


def h0_analysis(ax, ax1):
    T = 35
    n_groups = 20
    a = 0.5
    n = 6

    K = 20
    h0vec = np.linspace(0.01, 0.95, K)

    n_iter = 100

    h8_ab = np.zeros((n_iter, K))
    h8_ana = np.zeros(K)
    h8_euler = np.zeros(K)
    teq_ab = np.zeros((n_iter, K))
    teq_ana = np.zeros(K)
    teq_euler = np.zeros(K)

    for k, h0 in enumerate(h0vec):
        # Running the agent based
        for i in range(n_iter):
            _, ab, _, _ = basic(T, n_groups, a, n, h0)
            h8_ab[i, k] = ab[8]
            for t, h in enumerate(ab):
                if (a * n > 1 and h < 0.01) or (a * n < 1 and h > 0.99):
                    teq_ab[i, k] = t
                    break

        # Running analytical
        h8_ana[k] = analytical(8, h0, a, n)
        if a * n > 1:
            teq_ana[k] = inverse(0.01, h0, a, n)
        else:
            teq_ana[k] = inverse(0.99, h0, a, n)

        # Running numerical
        _, hnum = euler(T, a, n, step=1, h0=h0)
        h8_euler[k] = hnum[8]
        for t, h in enumerate(hnum):
            if (a * n > 1 and h < 0.01) or (a * n < 1 and h > 0.99):
                teq_euler[k] = t
                break

    std8 = h8_ab.std(axis=0)
    h8_ab = h8_ab.mean(axis=0)
    std_eq = teq_ab.std(axis=0)
    teq_ab = teq_ab.mean(axis=0)
    print(std8)

    # Plotting h8 with h0
    ax.plot(h0vec, h8_ab, label='agent-based', marker="x")
    ax.plot(h0vec, h8_ana, label='analytical')
    ax.plot(h0vec, h8_euler, label='numerical', marker="x")
    ax.set(xlabel="h0", ylabel="h(t=8)", ylim=[0, 0.85])
    ax.legend()
    ax.grid()

    # Plotting t_eq with h0
    ax1.errorbar(h0vec, teq_ab, yerr=std_eq, label='agent-based', marker="x")
    ax1.plot(h0vec, teq_ana, label='analytical')
    ax1.plot(h0vec, teq_euler, label='numerical', marker="x")
    ax1.set(xlabel="h0", ylabel="t_eq", ylim=[0, 30])
    ax1.legend(loc="upper left")
    ax1.grid()

    return ax, ax1


def a_analysis(ax, ax1):
    T = 70
    n_groups = 20
    n = 6
    h0 = 0.5

    K = 20
    h0vec = np.linspace(0, 1, K)

    n_iter = 100

    h8_ab = np.zeros((n_iter, K))
    h8_ana = np.zeros(K)
    h8_euler = np.zeros(K)
    teq_ab = np.zeros((n_iter, K))
    teq_ana = np.zeros(K)
    teq_euler = np.zeros(K)

    for k, a in enumerate(h0vec):
        # Running the agent based
        for i in range(n_iter):
            _, ab, _, _ = basic(T, n_groups, a, n, h0)
            h8_ab[i, k] = ab[8]
            for t, h in enumerate(ab):
                if (a * n > 1 and h < 0.01) or (a * n < 1 and h > 0.99):
                    teq_ab[i, k] = t
                    break

        # Running analytical
        h8_ana[k] = analytical(8, h0, a, n)
        if a * n > 1:
            teq_ana[k] = inverse(0.01, h0, a, n)
        else:
            teq_ana[k] = inverse(0.99, h0, a, n)

        # Running numerical
        _, hnum = euler(T, a, n, step=1, h0=h0)
        h8_euler[k] = hnum[8]
        for t, h in enumerate(hnum):
            if (a * n > 1 and h < 0.01) or (a * n < 1 and h > 0.99):
                teq_euler[k] = t
                break

    std8 = h8_ab.std(axis=0)
    h8_ab = h8_ab.mean(axis=0)
    print(teq_ab[:, ])
    std_eq = teq_ab.std(axis=0)
    teq_ab = teq_ab.mean(axis=0)

    # Plotting h8 with h0
    ax.errorbar(h0vec, h8_ab, yerr=std8, label='agent-based', marker="x")
    ax.plot(h0vec, h8_ana, label='analytical')
    ax.plot(h0vec, h8_euler, label='numerical', marker="x")
    ax.set(xlabel="a", ylabel="h(t=8)", ylim=[0, 0.85])
    ax.legend()
    ax.grid()

    # Plotting t_eq with h0
    ax1.errorbar(h0vec, teq_ab, yerr=std_eq, label='agent-based', marker="x")
    ax1.plot(h0vec, teq_ana, label='analytical')
    ax1.plot(h0vec, teq_euler, label='numerical', marker="x")
    ax1.set(xlabel="a", ylabel="t_eq", ylim=6)
    ax1.set_yscale("log", nonposy='clip')
    ax1.legend(loc="upper right")
    ax1.grid()


def n_analysis(ax, ax1):
    T = 45
    a = 0.5
    h0 = 0.5

    h0vec = np.array([3, 4, 5, 6])
    K = len(h0vec)

    n_iter = 100

    h8_ab = np.zeros((n_iter, K))
    h8_ana = np.zeros(K)
    h8_euler = np.zeros(K)
    teq_ab = np.zeros((n_iter, K))
    teq_ana = np.zeros(K)
    teq_euler = np.zeros(K)

    copyh = np.zeros((n_iter, K, T))
    copyl = np.zeros((n_iter, K, T))

    for k, n in enumerate(h0vec):
        n_groups = int(120 / n)
        # Running the agent based
        for i in range(n_iter):
            _, ab, copyh[i, k, :], copyl[i, k, :] = basic(T, n_groups, a, n, h0)
            h8_ab[i, k] = ab[8]
            for t, h in enumerate(ab):
                if (a * n > 1 and h < 0.01) or (a * n < 1 and h > 0.99):
                    teq_ab[i, k] = t
                    break

        # Running analytical
        h8_ana[k] = analytical(8, h0, a, n)
        if a * n > 1:
            teq_ana[k] = inverse(0.01, h0, a, n)
        else:
            teq_ana[k] = inverse(0.99, h0, a, n)

        # Running numerical
        _, hnum = euler(T, a, n, step=1, h0=h0)
        h8_euler[k] = hnum[8]
        for t, h in enumerate(hnum):
            if (a * n > 1 and h < 0.01) or (a * n < 1 and h > 0.99):
                teq_euler[k] = t
                break

    std_ch = copyh.std(axis=0)
    copyh = copyh.mean(axis=0)
    std_cl = copyl.std(axis=0)
    copyl = copyl.mean(axis=0)

    std8 = h8_ab.std(axis=0)
    h8_ab = h8_ab.mean(axis=0)
    print(teq_ab[:, 0])
    std_eq = teq_ab.std(axis=0)
    print(std_eq)
    teq_ab = teq_ab.mean(axis=0)

    # Plotting h8 with h0
    ax.errorbar(h0vec, h8_ab, yerr=std8, label='agent-based', marker="x")
    ax.plot(h0vec, h8_ana, label='analytical')
    ax.plot(h0vec, h8_euler, label='numerical', marker="x")
    ax.set(xlabel="n", ylabel="h(t=8)", ylim=[0, 0.85])
    ax.legend()
    ax.grid()

    # Plotting t_eq with h0
    ax1.errorbar(h0vec, teq_ab, yerr=std_eq, label='agent-based', marker="x")
    ax1.plot(h0vec, teq_ana, label='analytical')
    ax1.plot(h0vec, teq_euler, label='numerical', marker="x")
    ax1.legend(loc="upper right")
    ax1.set(xlabel="n", ylabel="t_eq", ylim=[0, 30])
    ax1.grid()

    # Plotting copyh, copyl with n
    fig, ax2 = plt.subplots(figsize=(10, 8))
    r = 20
    for i, n in enumerate(h0vec):
        ax2.errorbar(range(r), copyh[i, :r], yerr=std_ch[i, :r], label="n = " + str(n))
        ax2.errorbar(range(r), copyl[i, :r], yerr=std_cl[i, :r], label="n = " + str(n), ls='--')
    ax2.legend()
    # fig.show()


if __name__ == "__main__":
    ### Single runs
    #basic(20, 20, 0.5, 6, h0=0.5)

    ### Plotting
    ## Simple comparison of models (for Figure 4)
    #run_simple(h0=0.5)

    ## Parameter analysis, for Figure 5
    fig, ax = plt.subplots(3, 2, figsize=(8, 10))
    h0_analysis(ax[0][0], ax[0][1])
    a_analysis(ax[1][0], ax[1][1])
    n_analysis(ax[2][0], ax[2][1])
    fig.show()
