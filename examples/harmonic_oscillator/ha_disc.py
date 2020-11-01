'''
Harmonic Oscillator (with time) Example in Hylaa

Very simple 2-d example:

x' == y
y' == -x
'''

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

    # dynamics: x' = y, y' = -x
    a_matrix = np.array([[0.96065997, 0.1947354], [-0.1947354, 0.96065997]], dtype=float)
    a_csr = csr_matrix(a_matrix, dtype=float)

    b_mat = [[1, 0], [0, 1]]
    b_constraints = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    b_rhs = [0.39340481, -0.39340481, -0.03933961, 0.03933961]

    mode = ha.new_mode('mode')
    mode.set_dynamics(a_csr)
    mode.set_inputs(b_mat, b_constraints, b_rhs, allow_constants=True)

    return ha

def make_init(ha):
    '''returns list of initial states'''

    mode = ha.modes['mode']
    # init states: x in [-5, -4], y in [0, 1]
    init_lpi = lputil.from_box([[-6, -5], [0, 1]], mode)

    init_list = [StateSet(init_lpi, mode)]

    return init_list

def define_settings():
    'get the hylaa settings object'

    step = 0.2
    max_time = 20
    settings = HylaaSettings(step, max_time)

    plot_settings = settings.plot
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE
    plot_settings.xdim_dir = 0
    plot_settings.ydim_dir = 1

    #plot_settings.plot_mode = PlotSettings.PLOT_VIDEO
    #plot_settings.filename = 'ha.mp4'
    #plot_settings.video_fps = 2
    #plot_settings.video_extra_frames = 10 # extra frames at the end of a video so it doesn't end so abruptly
    #plot_settings.video_pause_frames = 5 # frames to render in video whenever a 'pause' occurs
    
    plot_settings.label.y_label = '$y$'
    plot_settings.label.x_label = '$x$'
    plot_settings.label.title = 'Harmonic Oscillator'

    return settings

def run_hylaa():
    'Runs hylaa with the given settings'

    ha = define_ha()
    settings = define_settings()
    init_states = make_init(ha)

    Core(ha, settings).run(init_states)

if __name__ == '__main__':
    run_hylaa()
