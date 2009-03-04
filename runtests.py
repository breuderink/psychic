#!/usr/bin/env python
import unittest, logging
from psychic import tests

logging.basicConfig()
logging.getLogger('psychic.preprocessing').setLevel(logging.ERROR)
suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
unittest.TextTestRunner(verbosity=1).run(suite)
