import unittest
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.path.append('src/runners')
# replaced with relative path to file for more flexibility
import os

from rawls_runner_assignment  import RawlsRunner as rr

class Rawls_Testing(unittest.TestCase):
    def test_rawls(self):
        print(os.path.dirname(__file__))
        result = rr.run_rawls(os.path.join(os.path.dirname(__file__), "rawls.dzn"))
        self.assertEqual(result, [99,30,99,99], "incorrect result")

    def test_rawls2(self):
        result = rr.run_rawls(os.path.join(os.path.dirname(__file__), "rawls2.dzn"))
        self.assertEqual(result, [98,30,98,99], "incorrect result")

    def test_rawls3(self):
        result = rr.run_rawls(os.path.join(os.path.dirname(__file__), "rawls3.dzn"))
        self.assertEqual(result, [98,30,98,99], "incorrect result")

    def test_rawls4(self):
        result = rr.run_rawls(os.path.join(os.path.dirname(__file__), "rawls4.dzn"))
        self.assertEqual(result, [0,1,0], "incorrect result")
    
    def test_rawls5(self):
        result = rr.run_rawls(os.path.join(os.path.dirname(__file__), "rawls5.dzn"))
        self.assertTrue(result == [0,1,0] or result==[0,0,1], "incorrect result")

    def test_rawls6(self):
        result = rr.run_rawls(os.path.join(os.path.dirname(__file__), "rawls6.dzn"))
        self.assertEqual(result, [1,1,1], "incorrect result")

if __name__ == '__main__':
    unittest.main()

