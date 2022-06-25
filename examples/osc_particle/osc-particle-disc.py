import math

import numpy as np
from scipy.sparse import csr_matrix

from hylaa.hybrid_automaton import HybridAutomaton
from hylaa.settings import HylaaSettings, PlotSettings
from hylaa.core import Core
from hylaa.stateset import StateSet
from hylaa import lputil

def define_ha():
    '''make the hybrid automaton'''

    ha = HybridAutomaton(discrete=True)

    a_matrix = np.array([[0.722468865032875, -0.523371053120237, 0, 0], [0.785056579680355, 0.696300312376864, 0, 0.1], [0, 0, 0.930530895811206, 0.03], [0, 0, 0, 0]], dtype=float)
    a_csr = csr_matrix(a_matrix, dtype=float)
    mode = ha.new_mode('mode')
    mode.set_dynamics(a_csr)

    # b_mat = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # b_constraints = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    # b_rhs = [0.0, 0.0, 0.1, -0.1, 0.03, -0.03]

    # mode = ha.new_mode('mode')
    # mode.set_dynamics(a_csr)
    # mode.set_inputs(b_mat, b_constraints, b_rhs)

    return ha


def make_init(ha):
    '''returns list of initial states'''

    mode = ha.modes['mode']

    init_lpi = lputil.from_box([[-0.1, 0.1], [-0.8, -0.4], [-1.07, -1], [1, 1]], mode)

    init_list = [StateSet(init_lpi, mode)]

    return init_list


def define_settings():
    'get the hylaa settings object'

    step = 0.6
    max_time = 12
    settings = HylaaSettings(step, max_time)

    plot_settings = settings.plot
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE
    plot_settings.xdim_dir = 2
    plot_settings.ydim_dir = 1

    #plot_settings.plot_mode = PlotSettings.PLOT_VIDEO
    #plot_settings.filename = 'ha.mp4'
    #plot_settings.video_fps = 2
    #plot_settings.video_extra_frames = 10 # extra frames at the end of a video so it doesn't end so abruptly
    #plot_settings.video_pause_frames = 5 # frames to render in video whenever a 'pause' occurs
    
    plot_settings.label.y_label = '$y$'
    plot_settings.label.x_label = '$x$'
    plot_settings.label.title = 'Oscillating Particle'

    return settings

def run_hylaa():
    'Runs hylaa with the given settings'

    ha = define_ha()
    settings = define_settings()
    init_states = make_init(ha)

    Core(ha, settings).run(init_states)

if __name__ == '__main__':
    run_hylaa()
