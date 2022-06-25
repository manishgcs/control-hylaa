from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import matplotlib.text
import numpy as np


class FarkasObject:
    def __init__(self, P1, P2, Q_set, n_vars):
        self.polytope1 = P1
        self.polytope2 = P2
        self.q_set = Q_set
        self.n_state_vars = n_vars

    def plotPolytope(self, p):
        x_index = 0
        y_index = 1
        x_min = p.intervals[x_index][0]
        x_max = p.intervals[x_index][1]
        y_min = p.intervals[y_index][0]
        y_max = p.intervals[y_index][1]
        p_verts = [
            (x_min, y_min),  # left, bottom
            (x_max, y_min),  # left, top
            (x_max, y_max),  # right, top
            (x_min, y_max),  # right, bottom
            (x_min, y_min),  # ignored
        ]

        # print(p_verts)
        codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY,
        ]

        p_path = Path(p_verts, codes)

        return p_path

    def get_rect_dims(self, poly):

        p = poly
        x_index = 0
        y_index = 1
        x_min = p.intervals[x_index][0]
        x_max = p.intervals[x_index][1]
        y_min = p.intervals[y_index][0]
        y_max = p.intervals[y_index][1]

        return x_min, x_max, y_min, y_max

    def plotPolytopes(self):

        # fig, ax = plt.subplots()
        #
        # color_id = color_id + 2
        #
        # p2_path = self.plotPolytope(self.polytope2)
        # p2_patch = patches.PathPatch(p2_path, facecolor='green', lw=1, fill=False)
        # ax.add_patch(p2_patch)
        #
        # color_id = color_id + 2
        #
        # for q_polytope in self.q_set:
        #     q_path = self.plotPolytope(q_polytope)
        #     q_patch = patches.PathPatch(q_path, facecolor=colors[color_id], lw=1, fill=False)
        #     ax.add_patch(q_patch)
        #     color_id = color_id + 2

        # colors = ['red', 'black', 'blue', 'brown', 'green']
        # plt.show()

        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0, 1, 15)]
        colors = ['red', 'black', 'blue', 'brown', 'green', 'magenta', 'pink']

        l_width = 2
        fig = plt.figure()
        ax = fig.add_subplot(111)
        color_id = 0
        x_min, x_max, y_min, y_max = self.get_rect_dims(self.polytope1)
        rect1 = matplotlib.patches.Rectangle((x_min, y_min),
                                             x_max - x_min, y_max - y_min,
                                             fc='none', ec=colors[color_id], lw=l_width, label='$P_1$')

        color_id = color_id + 1
        x_min, x_max, y_min, y_max = self.get_rect_dims(self.polytope2)
        rect2 = matplotlib.patches.Rectangle((x_min, y_min),
                                             x_max - x_min, y_max - y_min,
                                             fc='none', ec=colors[color_id], lw=l_width, label='$P\'_1$')
        ax.add_patch(rect1)
        ax.add_patch(rect2)

        color_id = color_id + 1

        poly_id = 2
        for q_polytope in self.q_set:
            poly_name = '$P_' + str(poly_id) + '$'
            x_min, x_max, y_min, y_max = self.get_rect_dims(q_polytope)
            rect_q = matplotlib.patches.Rectangle((x_min, y_min),
                                                 x_max - x_min, y_max - y_min,
                                                 fc='none', ec=colors[color_id], lw=l_width, label=poly_name)
            ax.add_patch(rect_q)
            color_id = color_id + 1
            poly_id = poly_id + 1

        plt.xlim([-2, 15])
        plt.ylim([-2, 15])

        plt.legend()
        # plt.text(2, 13, 'Sol($P_1$ & $P_1$)$ = \{P_2, P_3, P_5\}$', fontsize=10)
        # ax.text('ABC')
        plt.show()
