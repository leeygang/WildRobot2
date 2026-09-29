import unittest

import wr2


class PackageTest(unittest.TestCase):
    def test_version_is_defined(self):
        self.assertEqual(wr2.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
