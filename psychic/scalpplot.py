import pylab
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import numpy as np

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

def plot_scalp(densities, sensors, sensor_locs, show_sensors=True, 
  clim=[None, None]):
  curr_sens = dict([(lab, sensor_locs[lab]) for lab in sensors]) 
  add_density(densities, sensors, curr_sens, cmap=pylab.cm.jet, clim=clim)
  add_head()
  if show_sensors:
    add_sensors(curr_sens)
 
def add_head():
  ax = pylab.gca()
  ax.set_aspect(1.2)

  LINEWIDHT = 2
  nose = [
    (Path.MOVETO, (-.1, 1.)),
    (Path.LINETO, (0, 1.1)),
    (Path.LINETO, (.1, 1.))
    ]

  lear = [
    (Path.MOVETO, (-1, .134)),
    (Path.LINETO, (-1.04, 0.08)),
    (Path.LINETO, (-1.08, -0.11)),
    (Path.LINETO, (-1.06, -0.16)),
    (Path.LINETO, (-1.02, -0.15)),
    (Path.LINETO, (-1, -0.12)),
    ]
  rear = [(c, (-px, py)) for (c, (px, py)) in lear]

  # plot outline
  c = pylab.Circle((0, 0), 1, fill=False, linewidth=LINEWIDHT)
  ax.add_artist(c)

  # add nose and ears
  for p in [nose, lear, rear]:
    code, verts = zip(*p)
    ax.add_patch(PathPatch(Path(verts, code), fill=False, linewidth=LINEWIDHT))

  # update default lims
  MARGIN = 1.2
  pylab.xlim(-MARGIN, MARGIN)
  pylab.ylim(-MARGIN, MARGIN)
  pylab.box(False)
  ax.yaxis.set_visible(False)
  ax.xaxis.set_visible(False)

def add_sensors(sensor_dict):
  locs = []
  for (label, coord) in sensor_dict.items():
    (x, y, z) = coord
    pylab.text(x, y + .03, label, fontsize=8, ha='center')
    locs.append((x, y))
  locs = np.asarray(locs)
  pylab.plot(locs[:, 0], locs[:, 1], 'ko')

def add_density(dens, labels, sensor_dict, cmap=pylab.cm.jet, 
  clim=[None, None]):
  '''
  This function uses pylab.griddata, which is known to fail in pathetic cases.
  If it does, please consult the documentation for pylab.griddata.
  '''
  locs = [sensor_dict[l] for l in labels]
  xs, ys, zs = zip(*locs)
  RESOLUTION = 50
  xg = np.linspace(-1, 1, RESOLUTION)
  yg = np.linspace(-1, 1, RESOLUTION)
  zg = pylab.griddata(xs, ys, dens, xg, yg)
  extent = [min(xg), max(xg), min(yg), max(yg)]
  pylab.contour(xg, yg, zg, colors='k', extent=extent)
  vmin, vmax = clim
  pylab.imshow(zg, origin='lower', aspect='auto', interpolation='nearest', 
    extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
