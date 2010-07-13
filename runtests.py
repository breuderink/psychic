#!/usr/bin/env python
import unittest, logging
import numpy as np
from psychic import tests

np.set_printoptions(precision=2, suppress=True)

logging.basicConfig()
logging.getLogger('psychic.preprocessing').setLevel(logging.ERROR)
suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
unittest.TextTestRunner(verbosity=1).run(suite)
