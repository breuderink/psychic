import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle

BIOSEMI_32_LOCS = {
  'AF3': (-0.409, 0.87, 0.280), 'AF4': (0.409, 0.87, 0.280),
  'C3': (-0.719, 0.0, 0.689), 'C4': (0.719, 0.0, 0.689),
  'CP1': (-0.37, -0.37, 0.849), 'CP2': (0.37, -0.37, 0.849),
  'CP5': (-0.890, -0.340, 0.31), 'CP6': (0.890, -0.340, 0.31),
  'Cz': (-0, 0.0, 1.0), 'F3': (-0.550, 0.670, 0.5),
  'F4': (0.550, 0.670, 0.5), 'F7': (-0.810, 0.589, -0.0299),
  'F8': (0.810, 0.589, -0.0299), 'FC1': (-0.37, 0.37, 0.849),
  'FC2': (0.37, 0.37, 0.849), 'FC5': (-0.890, 0.340, 0.31),
  'FC6': (0.890, 0.340, 0.31), 'Fp1': (-0.31, 0.949, -0.0299),
  'Fp2': (0.31, 0.949, -0.0299), 'Fz': (-0, 0.719, 0.689),
  'O1': (-0.31, -0.949, -0.029), 'O2': (0.31, -0.949, -0.029),
  'Oz': (0.0, -1.0, -0.029), 'P3': (-0.550, -0.670, 0.5),
  'P4': (0.550, -0.670, 0.5), 'P7': (-0.810, -0.589, -0.029),
  'P8': (0.810, -0.589, -0.029), 'PO3': (-0.409, -0.87, 0.280),
  'PO4': (0.409, -0.87, 0.280), 'Pz': (0.0, -0.719, 0.689),
  'T7': (-1.0, 0.0, -0.029), 'T8': (1.0, 0.0, -0.0299)}

def plot_scalp(densities, sensors, sensor_locs, plot_sensors=True, 
  cmap=plt.cm.jet, clim=None):

  # add densities
  curr_sens = dict([(lab, sensor_locs[lab]) for lab in sensors]) 
  if clim == None:
    clim = [np.min(densities), np.max(densities)]
  add_density(densities, sensors, curr_sens, cmap=cmap, clim=clim)

  # setup plot
  MARGIN = 1.2
  plt.xlim(-MARGIN, MARGIN)
  plt.ylim(-MARGIN, MARGIN)
  plt.box(False)
  ax = plt.gca()
  ax.set_aspect(1.2)
  ax.yaxis.set_visible(False)
  ax.xaxis.set_visible(False)


  # add details
  add_head()
  if plot_sensors:
    add_sensors(curr_sens)
 
def add_head():
  '''Draw head outline'''
  LINEWIDTH = 1
  nose = [(Path.MOVETO, (-.1, 1.)), (Path.LINETO, (0, 1.1)),
    (Path.LINETO, (.1, 1.))]

  lear = [(Path.MOVETO, (-1, .134)), (Path.LINETO, (-1.04, 0.08)),
    (Path.LINETO, (-1.08, -0.11)), (Path.LINETO, (-1.06, -0.16)),
    (Path.LINETO, (-1.02, -0.15)), (Path.LINETO, (-1, -0.12))]

  rear = [(c, (-px, py)) for (c, (px, py)) in lear]

  # plot outline
  ax = plt.gca()
  ax.add_artist(plt.Circle((0, 0), 1, fill=False, linewidth=LINEWIDTH))

  # add nose and ears
  for p in [nose, lear, rear]:
    code, verts = zip(*p)
    ax.add_patch(PathPatch(Path(verts, code), fill=False, linewidth=LINEWIDTH))


def add_sensors(sensor_dict):
  '''Adds sensor names and markers'''
  locs = []
  for (label, coord) in sensor_dict.items():
    (x, y, z) = coord
    plt.text(x, y + .03, label, fontsize=8, ha='center')
    locs.append((x, y))
  locs = np.asarray(locs)
  plt.plot(locs[:, 0], locs[:, 1], 'ko')

def add_density(dens, labels, sensor_dict, cmap=plt.cm.jet, clim=None):
  '''
  This function draws the densities using the locations provided in
  sensor_dict. The two are connected throught the list labels.  The densities
  are inter/extrapolated on a grid slightly bigger than the head using
  scipy.interpolate.rbf. The grid is drawn using the colors provided in cmap
  and clim inside a circle. Contours are drawn on top of this grid.
  '''
  RESOLUTION = 50
  RADIUS = 1.2
  locs = [sensor_dict[l] for l in labels]
  xs, ys, zs = zip(*locs)
  extent = [-1.2, 1.2, -1.2, 1.2]
  vmin, vmax = clim

  # interpolate
  xg = np.linspace(extent[0], extent[1], RESOLUTION)
  yg = np.linspace(extent[2], extent[3], RESOLUTION)
  xg, yg = np.meshgrid(xg, yg)
  rbf = interpolate.Rbf(xs, ys, dens, function='linear', smooth=.0)
  zg = rbf(xg, yg)

  # draw contour
  plt.contour(xg, yg, np.where(xg ** 2 + yg ** 2 <= RADIUS ** 2, zg, np.nan),
    np.linspace(vmin, vmax, 13), colors='k', extent=extent, linewidths=.5)

  # draw grid, needs te be last to enable plt.colormap() to work
  im = plt.imshow(zg, origin='lower', extent=extent, vmin=vmin, vmax=vmax, 
    cmap=cmap)

  # clip grid to circle
  patch = Circle((0, 0), radius=RADIUS, facecolor='none', edgecolor='none')
  plt.gca().add_patch(patch)
  im.set_clip_path(patch)
