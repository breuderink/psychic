import re, datetime, unittest, logging
import numpy as np

bdf_log = logging.getLogger('BDFReader')

class BDFEndOfData: pass
class BDFReader:
  '''Simple wrapper to hide the records and read specific number of frames'''
  def __init__(self, file):
    self.bdf = BaseBDFReader(file)
    self.bdf.read_header()
    self.labels = self.bdf.header['label']
    self.buff = self.bdf.read_record()

  def read_nframes_raw(self, nframes):
    while self.buff == None or nframes > self.buff.shape[0]:
      # read more data
      rec = self.bdf.read_record()
      self.buff = np.vstack([self.buff, rec])
    
    # buffer contains enough data
    result = self.buff[:nframes, :]
    self.buff = self.buff[nframes:, :]
    return result

  def read_nframes(self, nframes):
    rframes = self.read_nframes_raw(nframes)
    return rframes * self.bdf.gain

  def read_all(self):
    #records = [record for record in self.bdf.records()]
    records = [self.buff] + list(self.bdf.records())
    rframes = np.vstack(records)
    print  rframes.dtype
    return rframes * self.bdf.gain

class BaseBDFReader:
  def __init__(self, file):
    self.file = file

  def read_header(self):
    '''Read the header of the BDF-file. The header is stored in self.header.'''
    f = self.file
    h = self.header = {}
    assert(f.tell() == 0) # check file position
    assert(f.read(8) == '\xffBIOSEMI')

    # recording info
    h['local_subject_id'] = f.read(80).strip()
    h['local_recording_id'] = f.read(80).strip()

    # parse timestamp
    (day, month, year) = [int(x) for x in re.findall('(\d+)', f.read(8))]
    (hour, minute, sec)= [int(x) for x in re.findall('(\d+)', f.read(8))]
    h['date_time'] = str(datetime.datetime(year + 2000, month, day, 
      hour, minute, sec))

    # misc
    self.header_nbytes = int(f.read(8))
    format = f.read(44).strip()
    assert(format == '24BIT')
    h['n_records'] = int(f.read(8))
    h['record_length'] = int(f.read(8)) # in seconds
    self.nchannels = h['n_channels'] = int(f.read(4))

    # read channel info
    channels = range(h['n_channels'])
    h['label'] = [f.read(16).strip() for n in channels]
    h['transducer_type'] = [f.read(80).strip() for n in channels]
    h['units'] = [f.read(8).strip() for n in channels]
    h['physical_min'] = [int(f.read(8)) for n in channels]
    h['physical_max'] = [int(f.read(8)) for n in channels]
    h['digital_min'] = [int(f.read(8)) for n in channels]
    h['digital_max'] = [int(f.read(8)) for n in channels]
    h['prefiltering'] = [f.read(80).strip() for n in channels]
    h['n_samples_per_record'] = [int(f.read(8)) for n in channels]
    f.read(32 * h['n_channels']) # reserved
    
    assert(f.tell() == self.header_nbytes)

    self.gain = np.array([(h['physical_max'][n] - h['physical_min'][n]) / 
      float(h['digital_max'][n] - h['digital_min'][n]) for n in channels], 
      np.float32)
    return self.header
  
  def read_record(self):
    '''
    Read a record with data for all channels, and return an 2D array,
    sampels * channels
    '''
    h = self.header
    n_channels = h['n_channels']
    n_samp = h['n_samples_per_record']
    assert(np.min(n_samp) == np.max(n_samp))
    n_samp = np.min(n_samp)
    result = np.zeros((n_samp, n_channels), np.float32)

    for i in range(n_channels):
      bytes = self.file.read(n_samp * 3)
      if len(bytes) <> n_samp * 3:
        raise BDFEndOfData
      result[:, i] = le_to_int24(bytes)

    return result

  def records(self):
    '''
    Record generator.
    '''
    try:
      while True:
        yield self.read_record()
    except BDFEndOfData:
      pass
  
  def __str__(self):
    h = self.header
    return '%s - %s\nChannels [%s] recorded at max %dHz on %s' % \
    (\
    h['local_subject_id'], h['local_recording_id'],
    ', '.join(h['label']), max(h['n_samples_per_record']), h['date_time'],\
    )

def le_to_int24(bytes):
  '''Convert groups of 3 bytes (little endian, two's complement) to an
  iterable to a numpy array of 24-bit integers.'''
  if type(bytes) == str:
    bytes = np.fromstring(bytes, np.uint8)
  else:
    bytes = np.asarray(bytes, np.uint8)
  int_rows = bytes.reshape(-1, 3).astype(np.int32)
  ints = int_rows[:, 0] + (int_rows[:, 1] << 8) + (int_rows[:, 2] << 16)
  ints[ints >= (1 << 23)] -= (1 << 24)
  return ints
  
def int24_to_le(ints):
  '''Convert an interable with 24-bit ints to little endian, two's complement
  numpy array.'''
  uints = np.array(ints, np.int32)
  uints[ints < 0] -= (1 << 24)
  bytes = np.zeros((uints.size, 3), np.uint8)
  bytes[:, 0] = uints & 0xff
  bytes[:, 1] = (uints >> 8) & 0xff
  bytes[:, 2] = (uints >> 16) & 0xff
  return bytes.flatten()

class TestConversion(unittest.TestCase):
  def setUp(self):
    self.know_values = [
      ((0, 0, 0), 0),
      ((1, 0, 0), 1),
      ((255, 255, 127), (1 << 23) - 1),
      ((0, 0, 128), -(1 << 23)),
      ((255, 255, 255), -1)]
    self.ints = [0, 1, (1 << 23) - 1, -(1 << 23), -1]

  def test_list_conversion(self):
    bytes = list(reduce(lambda x, y: x + y, # Make flat list of bytes
      [bs for (bs, i) in self.know_values]))
    ints = [i for (bs, i) in self.know_values]

    self.assertEqual(list(le_to_int24(bytes)), ints)
    self.assertEqual(list(int24_to_le(ints)), bytes)

class TestBDFReader(unittest.TestCase):
  def setUp(self):
    self.bdf = BDFReader(open(os.path.join('data', 'sine-256Hz.bdf'), 'rb'))

  def test_read_all(self):
    b = self.bdf
    eeg = b.read_all()
    self.assertEqual(eeg.shape[0], 
      b.bdf.header['n_records'] * max(b.bdf.header['n_samples_per_record']))

def main():
  logging.basicConfig(level=logging.DEBUG)
  b = BDFReader(open(os.path.join('data', 'sine-256Hz.bdf'), 'rb'))
  eeg = b.read_nframes(256 * 3)
  eeg = eeg[:, :-1]
  pylab.plot(eeg - np.mean(eeg, axis=0) + np.arange(eeg.shape[1]) * 100)
  pylab.show()


if __name__ == '__main__':
  import os
  import pylab
  main()
  unittest.main()
