import numpy as np
from scipy.sparse import csr_matrix

from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.settings import HylaaSettings, PlotSettings
from hylaa.core import Core
from hylaa.stateset import StateSet
from hylaa import lputil
from farkas_central.process_stars import process_stars
from farkas_central.bdd4Ce import BDD4CE
from hylaa.timerutil import Timers
from farkas_central.control_utils import get_input
import time


def define_ha(inp=0):
    '''make the hybrid automaton'''

    ha = HybridAutomaton(discrete=False)

    a_matrix = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, -1, 0, 0, 0, 0, 0, 0],
                              [1.7152555329, 3.9705119979, -4.3600526739, -0.9999330812, -1.5731541104, 0.2669165553,
                               -0.2215507198,
                               -0.4303855023, 0.0669078193],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, -1, 0, 0, 0],
                              [0.7153224517, 2.3973578876, 0.2669165553, 1.4937048131, 3.5401264957, -4.2931448546,
                               -1.0880831031,
                               -1.7613009555, 0.2991352608],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, -1],
                              [0.493771732, 1.9669723853, 0.0669078193, 0.6271724298, 2.2092110425, 0.2991352608,
                               1.4593953061, 3.4633677762,
                               -4.2704788265]], dtype=float)
    a_csr = csr_matrix(a_matrix, dtype=float)

    mode = ha.new_mode('mode')
    mode.set_dynamics(a_csr)

    if inp > 0:
        if inp == 1:
            b_mat = [[1], [1], [1], [1], [1], [1], [1], [1], [1]]
            b_constraints = [[1], [-1]]
            b_rhs = [0.02, 0.02]
        else:
            b_mat = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
            b_constraints = [[1, 0], [-1, 0], [0, 1], [0, -1]]
            b_rhs = [0.02, 0.02, 0.02, 0.02]

        mode.set_inputs(b_mat, b_constraints, b_rhs, allow_constants=False)

    error = ha.new_mode('error')

    trans = ha.new_transition(mode, error)
    trans.set_guard([[0, 1, 0, 0, 0, 0, 0, 0, 0], ], [-0.37, ])

    print(" ****** define ha end ****** ")

    return ha


def make_init(ha):
    '''returns list of initial states'''

    mode = ha.modes['mode']
    init_lpi = lputil.from_box([[0.9, 1.1], [0.9, 1.1], [0.9, 1.1], [0.9, 1.1], [0.9, 1.1], [0.9, 1.1], [0.9, 1.1], [0.9, 1.1], [0.9, 1.1]], mode)

    init_list = [StateSet(init_lpi, mode)]

    return init_list


def define_settings():
    'get the hylaa settings object'

    step = 0.2
    max_time = 20.0
    settings = HylaaSettings(step, max_time)

    plot_settings = settings.plot
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE
    plot_settings.xdim_dir = 0
    plot_settings.ydim_dir = 1

    # plot_settings.plot_mode = PlotSettings.PLOT_VIDEO
    # plot_settings.filename = 'ha.mp4'
    # plot_settings.video_fps = 2
    # plot_settings.video_extra_frames = 10 # extra frames at the end of a video so it doesn't end so abruptly
    # plot_settings.video_pause_frames = 5 # frames to render in video whenever a 'pause' occurs

    plot_settings.label.y_label = '$y$'
    plot_settings.label.x_label = '$x$'
    plot_settings.label.title = 'Oscillating Particle'

    return settings


def run_hylaa(order=None, level_merge=None, bdd_f=None, inp=None):
    'Runs hylaa with the given settings'

    ha = define_ha(inp)
    settings = define_settings()
    init_states = make_init(ha)
    core = Core(ha, settings)
    core.run(init_states)
    error_states = core.get_error_stars()  # error states are of type StarData
    print(" no of error states: " + str(len(error_states)))
    usafeset_preds = core.get_errorset_preds()

    Timers.tic("BDD Construction")
    start_time = time.time()
    process_stars(error_states)

    bdd_ce_object = BDD4CE(error_states, usafeset_preds, smt_mip='mip')
    # #
    bdd_graphs = bdd_ce_object.create_bdd_w_level_merge(level_merge=level_merge, order=order, bdd_f=bdd_f)
    valid_exps, invalid_exps = bdd_graphs[0].generate_expressions()
    print(len(valid_exps), len(invalid_exps))
    Timers.toc("BDD Construction")
    Timers.print_stats()
    bdd_f.write("\nUnique states:" + str(len(valid_exps)))
    bdd_f.write("\nTime taken: " + str(time.time() - start_time))


if __name__ == '__main__':
    level_merges = [-1, 0]
    orders = ['default', 'mid-order', 'random']
    inputs = [0, 1, 2]

    for inp in inputs:
        bdd_f = open("vehicle-plat3-bdd-" + str(inp) + ".txt", "a+")
        for order in orders:
            for level in level_merges:
                bdd_f.write("\ninput: " + str(inp))
                bdd_f.write("\norder: " + str(order))
                bdd_f.write("\nlevel: " + str(level))
                run_hylaa(order, level, bdd_f, inp)
        bdd_f.close()
