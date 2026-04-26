import numpy as np

def get_pitch_x(vp, test_point):
    vp_x, vp_y = vp
    pt_x, pt_y = test_point
    if pt_y == vp_y:
        return pt_x
    return vp_x + (1000 - vp_y) * (pt_x - vp_x) / (pt_y - vp_y)

vp = (2000, -10000)
print(get_pitch_x(vp, (500, 500)))
print(get_pitch_x(vp, (600, 500)))
