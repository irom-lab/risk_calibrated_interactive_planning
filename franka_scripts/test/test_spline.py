import scipy
import numpy as np

# Time for spline 
tn = [0, 0.30, 0.10, 0.15, 0.1, 0.1]
tn = np.cumsum(tn)
t_total = tn[-1]
# ts = np.arange(0, tn[-1], self.dt)

action = np.array([0, 5, -0.05, 
                    0.6, 0.7, 0.3,
                    -0.3, -0.6, -0.6])

# Spline for x direction
xd = np.zeros((6))
xd[1:4] = action[3:6]
s_xd = scipy.interpolate.interp1d(tn, xd)

# Spline for pitch direction
pitchd = np.zeros((6))
pitchd[2:5] = action[6:9]
s_pitchd = scipy.interpolate.interp1d(tn, pitchd)

# Spline for z direction
zd = [0, 0.01, 0.0, -0.01, -0.02, -0.02]   # was -0.01 for 2nd
# s_zd = scipy.interpolate.CubicSpline(tn, zd, bc_type='clamped')
s_zd = scipy.interpolate.interp1d(tn, zd)


t_cur = 0
while t_cur <= t_total:
    print([s_xd(t_cur), 0, s_zd(t_cur), 0, s_pitchd(t_cur), 0])
    t_cur += 0.01

