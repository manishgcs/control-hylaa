[![Build Status](https://travis-ci.org/stanleybak/hylaa.svg?branch=master)](https://travis-ci.org/stanleybak/hylaa)

# Hylaa #

<p align="center"> <img src="hylaa_logo_small.png" alt="Hylaa Logo"/> </p>

Hylaa (**HY**brid **L**inear **A**utomata **A**nalyzer) is a verification tool for system models with linear ODEs, time-varying inputs, and possibly hybrid dynamics. 

The is version 2 of Hylaa, with support for resets and hybrid automata with time-varying inputs. The focus has shifted from performance to handling more general dynamics.

The latest version of Hylaa is always available on our github repository at https://github.com/stanleybak/hylaa . A website for Hylaa is maintained at http://stanleybak.com/hylaa .

The main citation to use for Hylaa is: "HyLAA: A Tool for Computing Simulation-Equivalent Reachability for Linear Systems", S. Bak, P. Duggirala, 20th International Conference on Hybrid Systems: Computation and Control (HSCC 2017)

The code was mostly written by Stanley Bak (http://stanleybak.com) with input from Parasara Sridhar Duggirala (http://engr.uconn.edu/~psd).

Hylaa is released under the GPL v3 license (see the LICENSE file). Earlier versions have been approved for public release (DISTRIBUTION A: Approved for public release; distribution unlimited #88ABW-2016-5976, 22 NOV 2016).

### Installation ###

This version of Hylaa runs in `python3` and requires a few other libraries that you can install with `pip3`, the python package manager. You must also set your `PYTHONPATH` environment variable so that it knows where the hylaa source is located. There is a `Dockerfile` in this repository which is used as part of our continuous integration framework that has step by step commands for installing the necessary packages and dependencies. This serves as the installation documentation, as it's always up to date.

* One can also create a virtual environment dedicated to running Hylaa.  To locate the list of virtualenvs, run below command.

```commandline
locate -b '\activate' | grep "/home"
```
```
virutalenv control-hylaa
source control-hylaa/bin/activate
```

The list of libraries that we needed to install is given below.

```
pip3 install numpy
pip3 install ffmpeg
pip3 install graphviz
pip3 install matplotlib
pip3 install pytest
pip3 install scipy
pip3 install swiglpk
pip3 install sympy
pip3 install termcolor
pip3 install gurobipy
pip3 install z3
```

In order to compute complete characterization of counterexamples, you would need GUROBI (https://www.gurobi.com/). Once Gurobi is install,
please update ./bashrc script. Add GUROBI_HOME to PATH variable and add GRB_LICENSE_FILE to the script. 

```
export GUROBI_HOME="$HOME/gurobi952/linux64"
export GRB_LICENSE_FILE="$HOME/gurobi.lic"
export PATH="${PATH}::${GUROBI_HOME}/bin"
export PYTHONPATH="$HOME/control-hylaa/"
```

### Getting Started + Example ###

The easiest way to get started with Hylaa is to run some of the examples. Once installed and setup, Hylaa models are just python source files you directly with `python3` in a terminal.

Go to `examples/harmonic_oscillator` and run `ha.py` from the command line (`python ha.py`). This should create `plot.png` in the same folder, which will be an 2-d plot of the reachable set.

Or,

Go to `examples/tmpc_model` and run `tmpc_model1.py` from the command line (`python3 tmpc_model1.py`). This should create `tmpc1_x_z.png` in the same folder, which will be an 2-d plot of the reachable set.  To check for the vertex containment in a reachable polytope, you will need to make changes in get_verts() in lpplot.py.

The dynamics in this example are given as x' = **A**x, where **A** is the (potentially sparse) dynamics matrix. This is defined in the `define_ha` function in the `ha.py` source file.

Initial states and unsafe states are given as conjunctions of linear constraints. These are defined in the `make_init` function.

Finally, computation settings are given in the `define_settings` function. There are lots of settings that can be adjusted, which can be found in `hylaa/settings.py`, including comments describing what each one does.

The easiest way to use Hylaa on your example is to copy an example from the examples folder and edit that. Notice that models are python code, which means you can 
create the model programatically using loops or by loading the dynamics from a .mat file.

### Pending Deprecation Warnings in Test Suite ###

The test suite produces pending deprecation warnings. This comes from scipy sparse matrices using matrix objects, particularly for computing the matrix exponential. I expect at some point scipy will make an update that will fix these, and then they'll go away. For now, we're ignoring them.

