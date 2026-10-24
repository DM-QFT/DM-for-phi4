"""Physics, noise schedule and network constants shared by all scripts. Lengths in units of r=1."""
import math

# physics. sqrt(4 D DT_PAIR) = 1 r, half a branching per particle per DT_PAIR, memory e^{-LAM DT_PAIR} = 0.61 at k -> 0.
# GAM makes the Poisson process of density 1/r^2 the exact stationary state (birth disk and merge top-hat share the radius R)
R = 1.0
D = 0.025
LAM = 0.05
GAM = 2 * LAM / math.pi
DT_PAIR = 10.0
DT_SIM = 0.05      # event-rate error of the Euler scheme ~ GAM DT_SIM pi/2 + LAM DT_SIM, below 0.5%

# noise schedule, sigma in units of r. below 1e-2 the target is pure noise, SIGMA_MAX ~ largest test box so the prior is uniform
SIGMA_MIN = 1e-2
SIGMA_MAX = 40.0

# network geometry, radii and grid spacings in units of r
R_FINE = 1.0
H_MID = 1.0
R_MID = 2.0
H_COARSE = 2.0
R_COARSE = 4.0
