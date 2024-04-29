import matplotlib as mpl
import numpy as np
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]


def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp


def add_link_colorbar(link_cmap, ax, link_vmin=None, link_vmax=None, link_colorbar_label='Link',pos=None ):
    norm = mpl.colors.Normalize(vmin=link_vmin, vmax=link_vmax)

    sm = plt.cm.ScalarMappable(cmap=link_cmap, norm=norm)
    divider = make_axes_locatable(ax)
    if pos is None:
        pos = (0.8, 0.125, 0.05, 0.75)
    clb = plt.colorbar(sm, shrink=1, cax=ax.inset_axes(pos))
    clb.ax.set_title(link_colorbar_label, fontsize=10)

import matplotlib.ticker as mticker


class MathTextSciFormatter(mticker.Formatter):
    def __init__(self, fmt="%1.2e"):
        self.fmt = fmt

    def __call__(self, x, pos=None):
        s = self.fmt % x
        decimal_point = '.'
        positive_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(decimal_point)
        sign = tup[1][0].replace(positive_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if exponent:
            exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s = r'%s{\times}%s' % (significand, exponent)
        else:
            s = r'%s%s' % (significand, exponent)
        return "${}$".format(s)


def _get_other_matrices(rotation, scaling, shearing):  # calculate cos and sin of angles

    # get the values on each axis
    r_x, r_y, r_z = rotation
    sc_x, sc_y, sc_z = scaling
    sh_x, sh_y, sh_z = shearing

    # convert degree angles to rad
    theta_rx = np.deg2rad(r_x)
    theta_ry = np.deg2rad(r_y)
    theta_rz = np.deg2rad(r_z)
    theta_shx = np.deg2rad(sh_x)
    theta_shy = np.deg2rad(sh_y)
    theta_shz = np.deg2rad(sh_z)

    sin_rx, cos_rx = np.sin(theta_rx), np.cos(theta_rx)
    sin_ry, cos_ry = np.sin(theta_ry), np.cos(theta_ry)
    sin_rz, cos_rz = np.sin(theta_rz), np.cos(theta_rz)
    # get the rotation matrix on x axis
    R_Mx = np.array([[1, 0, 0, 0],
                     [0, cos_rx, -sin_rx, 0],
                     [0, sin_rx, cos_rx, 0],
                     [0, 0, 0, 1]])
    # get the rotation matrix on y axis
    R_My = np.array([[cos_ry, 0, -sin_ry, 0],
                     [0, 1, 0, 0],
                     [sin_ry, 0, cos_ry, 0],
                     [0, 0, 0, 1]])
    # get the rotation matrix on z axis
    R_Mz = np.array([[cos_rz, -sin_rz, 0, 0],
                     [sin_rz, cos_rz, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    # compute the full rotation matrix
    R_M = np.dot(np.dot(R_Mx, R_My), R_Mz)

    # get the scaling matrix
    Sc_M = np.array([[sc_x, 0, 0, 0],
                     [0, sc_y, 0, 0],
                     [0, 0, sc_z, 0],
                     [0, 0, 0, 1]])

    # get the tan of angles
    tan_shx = np.tan(theta_shx)
    tan_shy = np.tan(theta_shy)
    tan_shz = np.tan(theta_shz)
    # get the shearing matrix on x axis
    Sh_Mx = np.array([[1, 0, 0, 0],
                      [tan_shy, 1, 0, 0],
                      [tan_shz, 0, 1, 0],
                      [0, 0, 0, 1]])
    # get the shearing matrix on y axis
    Sh_My = np.array([[1, tan_shx, 0, 0],
                      [0, 1, 0, 0],
                      [0, tan_shz, 1, 0],
                      [0, 0, 0, 1]])
    # get the shearing matrix on z axis
    Sh_Mz = np.array([[1, 0, tan_shx, 0],
                      [0, 1, tan_shy, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
    # compute the full shearing matrix
    Sh_M = np.dot(np.dot(Sh_Mx, Sh_My), Sh_Mz)
    return R_M, Sc_M, Sh_M



def coordinate_transform(points,
                         translation=(0, 0, 0),
                         rotation=(0, 0, 0),
                         scaling=(1, 1, 1),
                         shearing=(0, 0, 0)):

    # translation matrix to translate the image
    t_x, t_y, t_z = translation
    T_M = np.array([[1, 0, 0, t_x],
                    [0, 1, 0, t_y],
                    [0, 0, 1, t_z],
                    [0, 0, 0, 1]])

    R_M, Sc_M, Sh_M = _get_other_matrices(rotation, scaling, shearing)

    Identity = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    # compute the full transform matrix
    M = Identity
    M = np.dot(T_M, M)
    M = np.dot(R_M, M)
    M = np.dot(Sc_M, M)
    M = np.dot(Sh_M, M)

    # apply the transformation
    points = np.concatenate([points, np.ones((points.shape[0],1))],axis=1).T
    np.dot(M, points)
    return points
