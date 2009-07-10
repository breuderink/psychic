import numpy as np

def markers_to_events(marker_array):
  '''
  Extract events from the array with markers.
  Events are encoded as TTL pulses, no event is indicated with the value 0.
  Returns (events, indices).
  '''
  markers = np.asarray(marker_array, int)
  change_ids = np.flatnonzero(np.concatenate([[1], np.diff(markers)]))
  events = markers[change_ids]
  return (events[np.nonzero(events)], change_ids[np.nonzero(events)])

def biosemi_find_ghost_markers(ys):
  '''
  Biosemi seems to decimate the status channel by taking the max of each
  window. When two makers fall in the same frame after decimation, a ghost
  marker appears, with the value of the bitwise or of the other markers.
  This function finds ghost markers using a heuristic. 
  THIS FUNCTION IS DANGEROUS!
  '''
  ys = np.asarray(ys)
  e, ei = markers_to_events(ys)
  if len(ei) < 3:
    return np.zeros(0)

  # First, find markers that are the binary OR of their neighbours
  pre_ghost_post = np.array([e[:-2], e[1:-1], e[2:]]).T
  or_matches = np.hstack([False, pre_ghost_post[:, 0] | pre_ghost_post[:,-1] \
    == pre_ghost_post[:, 1], False])

  # Now check which markers are not separated with a 0
  non_sep_matches = np.hstack(
    [False, (ys[ei[1:-1] - 1] != 0) & (ys[ei[2:] - 1] != 0), False])

  # Finally find markers that are one frame long
  one_frame = np.hstack([np.diff(ei) == 1, False])

  ghosts = or_matches & non_sep_matches & one_frame
  return ei[ghosts]

def resample_markers(markers, newlen, max_delay=0):
  '''
  Resample a marker stream without losing markers. max_delay specifies how
  many frames the markers can be delayed in *target frames*. 
  '''
  factor = float(newlen)/len(markers)
  e, ei = markers_to_events(markers)
  ei = (ei * factor).astype(int)
  old_ei = ei.copy()
  
  for i in range(1, len(ei)):
    if e[i] == e[i-1]:
      ei[i] = max(ei[i], ei[i-1] + 2)
    else:
      ei[i] = max(ei[i], ei[i-1] + 1)
      
  if len(ei) > 0:
    assert np.max(np.abs(ei - old_ei)) <= max_delay, \
      'Markers are delayed to much'
    assert max(ei) < newlen, 'Delayed markers out of bounds'
  ys = np.zeros(newlen)
  ys[ei] = e
  return ys

