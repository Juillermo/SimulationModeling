from num_analysis import f, analytical, k
import numpy as np
import matplotlib.pyplot as plt


def euler(T, a, n, step, h0=0.5, ax=None):
    # coeff = k(a, n)
    # print("coeff = {:.3f}".format(coeff))
    # print("range delta_t*fy: start= {:.3f}, end= {:.3f}]".format(step * (1 - 2 * h0) * coeff, step * coeff))

    h = h0
    err = []

    tvec = []
    hvec = [h0]
    anvec = []

    for t in np.arange(0, T + step, step):
        tvec.append(t)

        if t != 0:
            h = h + step * f(h, a, n)
            hvec.append(h)

        ana = analytical(t, h0, a, n)
        anvec.append(ana)

        local_error = h - ana
        err.append(abs(local_error))

        # print(str(h) + " - " + str(ana) + " = " + str(local_error))

    global_err = sum(err)
    if step == 1:
        print(global_err)

    if ax:
        # ax.plot(err)
        ax.plot(tvec, hvec, tvec, anvec)
        ax.set(title='delta_t = {:.0f}, global error = {:.3f}'.format(step, global_err),
               xlabel='t', ylabel='h')

    return global_err, hvec


def euler_h(T, ax=None):
    err = []
    N_range = np.arange(1, 19)
    h_range = T / N_range

    for h in h_range:
        err.append(euler(step=h, T=T, a=0.5, n=6))

    if ax:
        ax.plot(h_range, err, marker='x')
        ax.set(xlabel='delta_t', ylabel='Global error')


def euler_N(T, ax=None):
    err = []
    N_range = np.arange(10, 300)

    for N in N_range:
        err.append(euler(step=T / N, T=T, a=0.5, n=30))

    if ax:
        ax.plot(N_range, err, marker='x')
        ax.set(xlabel='k', ylabel='Global error')


if __name__ == "__main__":
    def print_num(T):
        fig, ax = plt.subplots(1, 2, figsize=(8, 2))

        print("Normal")
        euler(step=1, T=T, a=0.5, n=6, ax=ax[0])
        print("")
        print("Unstable")
        euler(step=8, T=T, a=0.5, n=6, ax=ax[1])  # Unstable

        fig.show()


    def print_scale(T):
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))

        euler_h(5, ax[0])
        euler_N(T, ax[1])

        fig.show()

    print_num(60)
    # print_scale(50)
