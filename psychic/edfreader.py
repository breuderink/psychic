# -*- coding: utf-8 -*-

import re, datetime, unittest, struct, collections
import numpy as np

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


#-------------------------------------------------------------------------------
import unittest, os

class TestEDFReader(unittest.TestCase):
  def test_synthetic_content(self):
    '''
    Test EDF reader using artifical EDF dataset. Note that this is not an
    EDF+ dataset and as such does not contain annotations. Annotations decoding
    is separately tested, *but not from a real file*!.
    '''
    reader = BaseEDFReader(
      open(os.path.join('data', 'sine3Hz_block0.2Hz.edf'), 'rb'))
    reader.read_header()

    h = reader.header
    # check important header fields
    self.assertEqual(h['label'], ['3Hz +5/-5 V', '0.2Hz Blk 1/0uV'])
    self.assertEqual(h['units'], ['V', 'uV'])
    self.assertEqual(h['contiguous'], True)

    fs = np.asarray(h['n_samples_per_record']) / h['record_length']

    # get records
    recs = list(reader.records())
    time = zip(*recs)[0]
    signals = zip(*recs)[1]
    annotations = list(zip(*recs)[2])

    # check EDF+ fields that are *not present in this file*
    np.testing.assert_equal(time, np.zeros(11) * np.nan)
    self.assertEqual(annotations, [[]] * 11)

    # check 3 Hz sine wave
    sine, block = [np.hstack(s) for s in zip(*signals)]
    target = 5 * np.sin(3 * np.pi * 2 * np.arange(0, sine.size) / fs[0])
    assert np.max((sine - target) ** 2) < 1e-4

    # check .2 Hz block wave
    target = np.sin(.2 * np.pi * 2 * np.arange(1, block.size + 1) / fs[1]) >= 0
    assert np.max((block - target) ** 2) < 1e-4

  def test_tal(self):
    mult_annotations = '+180\x14Lights off\x14Close door\x14\x00'
    with_duration = '+1800.2\x1525.5\x14Apnea\x14\x00'
    test_unicode = '+180\x14€\x14\x00\x00'

    # test annotation with duration
    self.assertEqual(tal(with_duration), [(1800.2, 25.5, [u'Apnea'])])

    # test multiple annotations
    self.assertEqual(tal('\x00' * 4 + with_duration * 3), 
      [(1800.2, 25.5, [u'Apnea'])] * 3)

    # test multiple annotations for one time point
    self.assertEqual(tal(mult_annotations), 
      [(180., 0., [u'Lights off', u'Close door'])])

    # test unicode support
    self.assertEqual(tal(test_unicode), [(180., 0., [u'€'])])
