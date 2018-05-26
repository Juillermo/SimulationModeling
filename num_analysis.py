import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def f(h, a, n):
    return h * (1 - h) * k(a, n)


def alpha(a, n, H=1, L=0):
    return 1 / (abs(1 / n - a) + (n - 1) / n) / (H - L)


def k(a, n, H=1, L=0):
    return alpha(a, n) * (1 / n - a) * (H - L)


def analytical(t, h0, a, n):
    return 1 / (1 + (1 - h0) / h0 * np.exp(-k(a, n) * t))


def inverse(h, h0, a, n):
    return np.log(h / (1 - h) * (1 / h0 - 1)) / k(a, n)


def print_fields(a, n):
    def E(t, h, a, n):
        dt = np.ones(t.shape)
        df = f(h, a, n)
        return dt, df

    def vec_field(a, n, ax):
        # Grid of x, y points
        nx, ny = 64, 64
        t = np.linspace(0, 20, nx)
        h = np.linspace(0, 1, ny)
        T, H = np.meshgrid(t, h)

        # Electric field vector, E=(Ex, Ey), as separate components
        vt, vh = E(t=T, h=H, a=a, n=n)

        # Plot the streamlines with an appropriate colormap and arrow style
        color = np.hypot(vt, vh)
        cmin = np.min(color)
        ax.streamplot(t, h, vt, vh, color=color, linewidth=1.5, cmap=plt.cm.magma,
                      norm=colors.Normalize(vmin=cmin,
                                            vmax=cmin + 2 * (np.max(color) - cmin)),
                      density=1, arrowstyle='->', arrowsize=1.5)

        ax.set_title("a = {:.1f}, n = {:d}, an = {:.1f}".format(a, n, a * n), fontsize=20)
        ax.set_xlabel('$t$')
        ax.set_ylabel('$h$')
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3))

    vec_field(a, n, axes[0])
    vec_field(0.1, 3, axes[1])

    plt.show()


if __name__ == "__main__":
    h0 = 0.5
    a = 0.5
    n = 6

    print_fields(a, n)

    # T = 10
    # for t in range(T):
    # print(analytical(t, 0.5, 0.5, 30))
