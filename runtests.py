import unittest
import tests

suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
unittest.TextTestRunner(verbosity=1).run(suite)
