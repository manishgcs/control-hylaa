'''
Harmonic Oscillator Example in Hylaa, demonstrating using
various approximation models for continuous-time reachability


dynamics are:
x' = y + u1
y' = -x + u2
starting from [-6, -5], [0, 1]
with u1 in [-0.5, 0.5]
'''

import math

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


def define_ha():
    '''make the hybrid automaton'''

    ha = HybridAutomaton(discrete=False)

    a_matrix = [[0, 1], [-1, 0]]

    b_mat = [[1, 0], [0, 1]]
    b_constraints = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    b_rhs = [0.2, 0.2, 0.2, 0.2]

    # if we had 3 inputs
    # b_mat = [[1, 0, 0], [0, 1, 0]]
    # b_constraints = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    # b_rhs = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    mode = ha.new_mode('mode')
    mode.set_dynamics(a_matrix)
    mode.set_inputs(b_mat, b_constraints, b_rhs)

    error = ha.new_mode('error')

    trans = ha.new_transition(mode, error)
    # y >= 6
    trans.set_guard([[0, -1], ], [-5.0, ])

    return ha


def make_init(ha):
    '''returns list of initial states'''

    mode = ha.modes['mode']
    # init states: x in [-5, -4], y in [0, 1]
    # init_lpi = lputil.from_box([[-5, -4], [0, 1]], mode)
    init_lpi = lputil.from_box([[-6, -5], [0, 1]], mode)

    init_list = [StateSet(init_lpi, mode)]

    return init_list


def define_settings():
    'get the hylaa settings object'

    step = 0.2
    max_time = 15 * step
    settings = HylaaSettings(step, max_time)

    settings.stdout = stdout = HylaaSettings.STDOUT_VERBOSE

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
    plot_settings.label.title = 'Harmonic Oscillator'

    return settings


def run_hylaa():
    'Runs hylaa with the given settings'

    ha = define_ha()
    settings = define_settings()

    tuples = []
    tuples.append((HylaaSettings.APPROX_NONE, "ha_inputs.png"))
    # tuples.append((HylaaSettings.APPROX_CHULL, "approx_chull.png"))
    # tuples.append((HylaaSettings.APPROX_LGG, "approx_lgg.png"))

    for model, filename in tuples:
        settings.approx_model, settings.plot.filename = model, filename

        init_states = make_init(ha)
        print(f"\nMaking {filename}...")
        # Core(ha, settings).run(init_states)
        core = Core(ha, settings)
        core.run(init_states)
        error_states = core.get_error_stars()  # error states are of type StarData
        print(" no of error states: " + str(len(error_states)))
        Timers.tic("BDD Construction")
        process_stars(error_states)
        usafeset_preds = core.get_errorset_preds()

        # print(guard_csr.todense())
        # self.guard_rhs

        bdd_ce_object = BDD4CE(error_states, usafeset_preds, equ_run=True, smt_mip='mip')
        # #
        bdd_graphs = bdd_ce_object.create_bdd_w_level_merge(level_merge=0, order='mid-order')
        #
        valid_exps, invalid_exps = bdd_graphs[0].generate_expressions()
        print(len(valid_exps), len(invalid_exps))
        Timers.toc("BDD Construction")
        Timers.print_stats()


if __name__ == '__main__':
    run_hylaa()
