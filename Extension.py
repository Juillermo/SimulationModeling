from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt

from AB_model import Agent, Group, Pop, Pop_Graph
from AB_simulation import plot_marks


def extension(parameters, T, folder_name=None, n_groups=20):
    Agent.H = 1.0
    Agent.L = parameters["L"]
    Agent.a = parameters["a"]
    Group.n = parameters["n"]

    Group.beta = parameters["beta"]
    Group.gamma = 0

    Pop.h0 = parameters["h0"]
    T = T

    n_agents = Group.n * n_groups
    alpha = 1 / (abs(1 / Group.n - Agent.a) + (Group.n - 1) / Group.n) / (Agent.H - Agent.L)
    Pop_Graph.alpha = alpha

    graph = Pop_Graph(n_agents, parameters)
    if folder_name:
        graph.statistics()

    h = np.zeros(T + 1)
    marks = np.zeros((T, n_groups))
    copyh = np.zeros(T)
    copyl = np.zeros(T)

    h[0] = graph.get_h()
    groups = [Group(graph.graph) for _ in range(n_groups)]

    for t in range(T):
        if folder_name:
            graph.draw_image(folder_name, t)
            print("t = "+str(t))

        for j, group in enumerate(groups):
            # Form groups
            group.add_agent(graph.pick_highest_node())
            for _ in range(Group.n - 1):
                group.add_highest_link()

            marks[t, j] = group.give_marks()
            group.rewire()

            group.agents = []
            group.links = {}

        graph.highest = 0
        graph.update_sum_links()
        copyh[t], copyl[t] = graph.copy_strategies()

        h[t + 1] = graph.get_h()

    if folder_name:
        graph.statistics()
        with open(folder_name + "/params.txt", mode='w') as a_file:
            a_file.write(str(parameters))

    return graph, h, marks, copyh, copyl


def run_simple(parameters, T, n_iter, folder_name):
    n_groups = 20

    h = np.zeros((n_iter, T + 1))
    marks = np.zeros((n_iter, T, n_groups))
    copyh = np.zeros((n_iter, T))
    copyl = np.zeros((n_iter, T))

    # Running iterations
    for i in range(n_iter):
        print(i)
        graph, h[i, :], marks[i, :, :], copyh[i, :], copyl[i, :] = extension(parameters, T=T, folder_name=folder_name)
        if i == 0:
            folder_name = None

    # Plotting
    hstd = h.std(axis=0)
    hstd[::2] = 0
    hstd[1::4] = 0

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    ax0 = ax[0][0]
    ax1 = ax[0][1]
    ax2 = ax[1][0]
    ax3 = ax[1][1]

    ax0.errorbar(range(T + 1), h.mean(axis=0), yerr=hstd)
    ax0.set(ylim=0, xlabel="t", ylabel="h(t)")

    ax1.plot(h.T, alpha=0.3, color="orange")
    ax1.set(ylim=0, xlabel="t", ylabel="h(t)")

    plot_marks(marks, T, n_iter, ax2)
    ax2.set(ylim=[15, 20])

    ax3.plot(copyh.mean(axis=0), label='H copying')
    ax3.plot(copyl.mean(axis=0), label='L copying')
    ax3.set(xlabel="t", ylabel="n", ylim=[0, 0.5])
    ax3.legend()

    fig.show()

    return h, marks, copyh, copyl


def compute_equilibrium(h):
    T = len(h) - 1

    h_buff = np.zeros(4)
    for t, ht in enumerate(h[::-1]):
        band = 0.005
        if t > len(h_buff):
            for h1, h2 in combinations(h_buff, 2):
                if abs(h1 - h2) > band:
                    teq = T - t
                    print(T - t)
                    return teq

        h_buff[1:] = h_buff[:-1]
        h_buff[0] = ht

    return 0


def plot_h_curves(h, ax, kvec):
    err = h.std(axis=0)

    for k, param in enumerate(kvec):
        err[k, k::2] = 0
        err[k, k + 1::4] = 0
        err[k, k + 3::8] = 0
        ax.errorbar(range(h.shape[2]), h[:, k, :].mean(axis=0), yerr=err[k, :], label="{:.2f}".format(param))

    ax.set(xlabel="t", ylabel="h(t)", ylim=[0, 1])
    ax.legend()
    ax.grid()


def h0_analysis(parameters, ax0, ax, ax1):
    parameters["n"] = 5

    T = 1500
    n_iter = 20

    K = 5
    h0vec = np.linspace(0.01, 0.95, K)

    h = np.zeros((n_iter, K, T + 1))
    teq = np.zeros(K)

    for k, h0 in enumerate(h0vec):
        print("h0 = " + str(h0))
        parameters["h0"] = h0

        # Run iterations
        for i in range(n_iter):
            _, h[i, k, :], _, _, _ = extension(parameters, T=T, folder_name=None)

        teq[k] = compute_equilibrium(h[:, k, :].mean(axis=0))

    # Average over iterations
    std8 = h[:, :, 8].std(axis=0)
    h8 = h[:, :, 8].mean(axis=0)
    stdeq = h[:, :, -1].std(axis=0)
    heq = h[:, :, -1].mean(axis=0)

    ## Plotting
    plot_h_curves(h, ax0, h0vec)

    # Plotting h8 and heq with h0
    ax.errorbar(h0vec, h8, yerr=std8, label="h8", marker="x")
    ax.errorbar(h0vec, heq, yerr=stdeq, label="heq", marker="x")
    ax.set(xlabel="h0", ylabel="h(t=8)", ylim=[0, 1])
    ax.legend()
    ax.grid()

    # Plotting t_eq with h0
    ax1.plot(h0vec, teq, marker="x")
    ax1.set(xlabel="h0", ylabel="t_eq")
    ax1.legend(loc="upper left")
    ax1.grid()

    return ax, ax1


def n_analysis(parameters, ax0, ax, ax1):
    T = 1500
    n_iter = 20

    K = 6
    nvec = np.arange(2, 2 + K)

    h = np.zeros((n_iter, K, T + 1))
    teq = np.zeros(K)

    for k, n in enumerate(nvec):
        print("n = " + str(n))
        parameters["n"] = n

        # Run iterations
        for i in range(n_iter):
            _, h[i, k, :], _, _, _ = extension(parameters, T=T, folder_name=None)

        # Compute equilibrium time
        teq[k] = compute_equilibrium(h[:, k, :].mean(axis=0))

    # Average over iterations
    std8 = h[:, :, 8].std(axis=0)
    h8 = h[:, :, 8].mean(axis=0)
    stdeq = h[:, :, -1].std(axis=0)
    heq = h[:, :, -1].mean(axis=0)

    ## Plotting
    plot_h_curves(h, ax0, nvec)

    # Plotting h8 and heq with n
    ax.errorbar(nvec, h8, yerr=std8, label="h8", marker="x")
    ax.errorbar(nvec, heq, yerr=stdeq, label="heq", marker="x")
    ax.set(xlabel="n", ylabel="h(t=8)", ylim=[0, 1])
    ax.legend()
    ax.grid()

    # Plotting t_eq with n
    ax1.plot(nvec, teq, marker="x")
    ax1.set(xlabel="n", ylabel="t_eq")
    ax1.legend(loc="upper left")
    ax1.grid()

    return ax, ax1


def a_analysis(parameters, ax0, ax, ax1):
    T = 1500
    n_iter = 20

    K = 5
    avec = np.linspace(0.15, 0.2, K)

    h = np.zeros((n_iter, K, T + 1))
    teq = np.zeros(K)

    for k, a in enumerate(avec):
        print("a = " + str(a))
        parameters["a"] = a

        # Run iterations
        for i in range(n_iter):
            _, h[i, k, :], _, _, _ = extension(parameters, T=T, folder_name=None)

        # Compute equilibrium time
        teq[k] = compute_equilibrium(h[:, k, :].mean(axis=0))

    # Average over iterations
    std8 = h[:, :, 8].std(axis=0)
    h8 = h[:, :, 8].mean(axis=0)
    stdeq = h[:, :, -1].std(axis=0)
    heq = h[:, :, -1].mean(axis=0)

    ## Plotting
    plot_h_curves(h, ax0, avec)

    # Plotting h8 and heq with a
    ax.errorbar(avec, h8, yerr=std8, label="h8", marker="x")
    ax.errorbar(avec, heq, yerr=stdeq, label="heq", marker="x")
    ax.set(xlabel="a", ylabel="h(t=8)", ylim=[0, 1])
    ax.legend()
    ax.grid()

    # Plotting t_eq with a
    ax1.plot(avec, teq, marker="x")
    ax1.set(xlabel="a", ylabel="t_eq")
    ax1.legend(loc="upper left")
    ax1.grid()

    return ax, ax1


if __name__ == "__main__":
    parameters = {
        "a": 0.2,
        "n": 6,
        "beta": 0.1,
        "h0": 0.5,
        "L": 0,
        "p": 0.1,
        "k": 6,
        "copying": "friendship"
    }
    ### Graph testing
    # Pop.h0 = parameters["h0"]
    # pop = Pop_Graph(120, parameters)

    ### Single run
    #graph, h, m, ch, cl = extension(parameters, T=200, folder_name=None, n_groups=20)

    ### Plotting
    ## Simple analysis of model (for Figure 6)
    # parameters["a"] = 0
    h, marks, copyh, copyl = run_simple(parameters, T=200, n_iter=5, folder_name=None)

    ## Parameter analysis
    # fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    # fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    # h0_analysis(parameters, ax[0], ax[1], ax[2])
    # h0_analysis(parameters, ax[0][0], ax[0][1], ax[0][2])
    # a_analysis(parameters, ax[0], ax[1], ax[2])
    # a_analysis
    # n_analysis(parameters, ax[0], ax[1], ax[2])
    # n_analysis(ax[2][0], ax[2][1])
    # fig.show()
