# TODO:
# - add support for log-transformed channels:
#   http://www.edfplus.info/specs/edffloat.html

import re, datetime, operator, logging
import numpy as np
from golem import DataSet

class EDFEndOfData: pass

EVENT_CHANNEL = 'EDF Annotations'

def tal(tal_str):
  '''
  Return a list with (onset, duration, annotation) tuples for an EDF+ TAL
  stream.
  '''
  exp = '(?P<onset>[+\-]\d+(?:\.\d*)?)' + \
    '(?:\x15(?P<duration>\d+(?:\.\d*)))?' + \
    '(\x14(?P<annotation>[^\x00]*))?' + \
    '(?:\x14\x00)'

  def annotation_to_list(annotation):
    return unicode(annotation, 'utf-8').split('\x14') if annotation else []

  def parse(dic):
    return (
      float(dic['onset']), 
      float(dic['duration']) if dic['duration'] else 0.,
      annotation_to_list(dic['annotation']))

  return [parse(m.groupdict()) for m in re.finditer(exp, tal_str)]

def edf_header(f):
  h = {}
  assert f.tell() == 0 # check file position
  assert f.read(8) == '0       '

  # recording info)
  h['local_subject_id'] = f.read(80).strip()
  h['local_recording_id'] = f.read(80).strip()

  # parse timestamp
  (day, month, year) = [int(x) for x in re.findall('(\d+)', f.read(8))]
  (hour, minute, sec)= [int(x) for x in re.findall('(\d+)', f.read(8))]
  h['date_time'] = str(datetime.datetime(year + 2000, month, day, 
    hour, minute, sec))

  # misc
  header_nbytes = int(f.read(8))
  subtype = f.read(44)[:5]
  h['EDF+'] = subtype in ['EDF+C', 'EDF+D']
  h['contiguous'] = subtype != 'EDF+D'
  h['n_records'] = int(f.read(8))
  h['record_length'] = float(f.read(8)) # in seconds
  nchannels = h['n_channels'] = int(f.read(4))

  # read channel info
  channels = range(h['n_channels'])
  h['label'] = [f.read(16).strip() for n in channels]
  h['transducer_type'] = [f.read(80).strip() for n in channels]
  h['units'] = [f.read(8).strip() for n in channels]
  h['physical_min'] = np.asarray([float(f.read(8)) for n in channels])
  h['physical_max'] = np.asarray([float(f.read(8)) for n in channels])
  h['digital_min'] = np.asarray([float(f.read(8)) for n in channels])
  h['digital_max'] = np.asarray([float(f.read(8)) for n in channels])
  h['prefiltering'] = [f.read(80).strip() for n in channels]
  h['n_samples_per_record'] = [int(f.read(8)) for n in channels]
  f.read(32 * nchannels) # reserved
  
  assert f.tell() == header_nbytes
  return h 

class BaseEDFReader:
  def __init__(self, file):
    self.file = file

  def read_header(self):
    self.header = h = edf_header(self.file)

    # calculate ranges for rescaling
    self.dig_min = h['digital_min']
    self.phys_min = h['physical_min']
    phys_range = h['physical_max'] - h['physical_min']
    dig_range = h['digital_max'] - h['digital_min']
    assert np.all(phys_range > 0)
    assert np.all(dig_range > 0)
    self.gain = phys_range / dig_range
  
  def read_raw_record(self):
    '''
    Read a record with data and return a list containing arrays with raw
    bytes.
    '''
    result = []
    for nsamp in self.header['n_samples_per_record']:
      samples = self.file.read(nsamp * 2)
      if len(samples) != nsamp * 2:
        raise EDFEndOfData
      result.append(samples)
    return result
    
  def convert_record(self, raw_record):
    '''
    Convert a raw record to a (time, signals, events) tuple based on
    information in the header.
    '''
    h = self.header
    dig_min, phys_min, gain = self.dig_min, self.phys_min, self.gain
    time = float('nan')
    signals = []
    events = []
    for (i, samples) in enumerate(raw_record):
      if h['label'][i] == EVENT_CHANNEL:
        ann = tal(samples)
        time = ann[0][0]
        events.extend(ann[1:])
      else:
        # 2-byte little endian integers
        dig = np.fromstring(samples, '<i2').astype(float)
        signals.append((dig - dig_min[i]) * gain[i] + phys_min[i])

    return time, signals, events

  def read_record(self):
    return self.convert_record(self.read_raw_record())

  def records(self):
    '''
    Record generator.
    '''
    try:
      while True:
        yield self.read_record()
    except EDFEndOfData:
      pass

def load_edf(fname, annotation_to_marker):
  '''
  Very basic reader for EDF and EDF+ files. While BaseEDFReader does support
  exotic features like non-homogeneous sample rates and loading only parts of
  the stream, load_edf expects a single fixed sample rate for all channels and
  tries to load the whole file.

  annotation_to_marker is a dictionary containing events to keep. This is a
  bit of an hack, but we need to cast the free-form annotations of EDF+ into
  a structured event channel.
  '''
  log = logging.getLogger('psychic.utils.load_edf')
  assert 0 not in annotation_to_marker.values()
  with open(fname, 'rb') as f:
    reader = BaseEDFReader(f)
    reader.read_header()

    h = reader.header
    # get sample rate info
    nsamp = np.unique(
      [n for (l, n) in zip(h['label'], h['n_samples_per_record'])
      if l != EVENT_CHANNEL])
    assert nsamp.size == 1, 'Multiple sample rates not yet supported'
    log.info('Detected sample rate %.2f' % nsamp)
    reclen = reader.header['record_length']

    # read and sort records just to be sure
    recs = sorted(reader.records()) 

    # create timestamps
    time = [zip(*recs)[0]][0]
    rec_time = np.linspace(0, reclen, nsamp, endpoint=False)
    I = np.hstack([t + rec_time for t in time])

    # create signal matrix X
    X = np.hstack(zip(*recs)[1])

    # create label matrix Y
    annotations = reduce(operator.add, zip(*recs)[2])
    events = [(o, annotation_to_marker.get(a, 0)) 
      for (o, d, aa) in annotations for a in aa]
    Y = np.zeros(X.shape[1])
    if events:
      offsets, mi = zip(*events)
      yi = np.searchsorted(I, offsets)
      Y[yi] = mi

    # construct DataSet
    feat_lab = [lab for lab in reader.header['label'] if lab != EVENT_CHANNEL]
    return DataSet(X=X, Y=Y, I=I, feat_lab=feat_lab,
      extra={'edf+_annotations' : annotations})
