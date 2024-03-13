import unittest
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.path.append('src/runners')
from rawls_runner_assignment  import RawlsRunner as rr

class Rawls_Testing(unittest.TestCase):
    def test_rawls(self):
        result = rr.run_rawls("test/rawls.dzn")
        self.assertEqual(result, [99,30,99,99], "incorrect result")

    def test_rawls2(self):
        result = rr.run_rawls("test/rawls2.dzn")
        self.assertEqual(result, [98,30,98,99], "incorrect result")

    def test_rawls3(self):
        result = rr.run_rawls("test/rawls3.dzn")
        self.assertEqual(result, [98,30,98,99], "incorrect result")

    def test_rawls4(self):
        result = rr.run_rawls("test/rawls4.dzn")
        self.assertEqual(result, [0,1,0], "incorrect result")
    
    def test_rawls5(self):
        result = rr.run_rawls("test/rawls5.dzn")
        self.assertTrue(result == [0,1,0] or result==[0,0,1], "incorrect result")

    def test_rawls6(self):
        result = rr.run_rawls("test/rawls6.dzn")
        self.assertEqual(result, [1,1,1], "incorrect result")

if __name__ == '__main__':
    unittest.main()

