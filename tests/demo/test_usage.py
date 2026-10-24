"""Usage panel: parsing of nvidia-smi, ioreg and top output, and the live process sample."""
import os
import unittest

from tools import usage

GPUS = ('0, GPU-01c08ec4-a2c6-1361-6c35-b449b00f831f, NVIDIA A16, 0, 0, 15356\n'
        '4, GPU-6a490dc2-d384-9010-bc0b-b0b76076bae3, NVIDIA RTX 6000 Ada Generation, 53, 21208, 46068\n')
APPS = '96344, 21198\n12, 500\n'
IOREG = '"PerformanceStatistics" = {"Alloc system memory"=9794846720,"Device Utilization %"=38,"In use system memory"=2422243328}'


class UsageParsingTests(unittest.TestCase):
    def test_nvidia_respects_visible_devices_and_attributes_process_memory(self):
        gpus = usage.parse_nvidia(GPUS, APPS, 'GPU-6a490dc2', 96344)
        self.assertEqual([gpu['name'] for gpu in gpus], ['NVIDIA RTX 6000 Ada Generation'])
        self.assertEqual(gpus[0]['utilization_percent'], 53.0)
        self.assertEqual(gpus[0]['memory_total_bytes'], 46068 * 1024 ** 2)
        self.assertEqual(gpus[0]['process_memory_bytes'], 21198 * 1024 ** 2)
        self.assertEqual(len(usage.parse_nvidia(GPUS, APPS, None, None)), 2)
        self.assertEqual(usage.parse_nvidia(GPUS, APPS, '0', None)[0]['name'], 'NVIDIA A16')

    def test_apple_gpu_from_ioreg(self):
        (gpu,) = usage.parse_ioreg(IOREG, 'Apple M3 Max', 128 * 1024 ** 3)
        self.assertEqual((gpu['name'], gpu['kind'], gpu['utilization_percent']), ('Apple M3 Max GPU', 'mps', 38.0))
        self.assertEqual(gpu['memory_used_bytes'], 9794846720)
        self.assertEqual(usage.parse_ioreg('no accelerator', 'Apple', None), [])

    def test_top_memory_sizes(self):
        self.assertEqual(usage.parse_size('16G'), 16 * 1024 ** 3)
        self.assertEqual(usage.parse_size('512M+'), 512 * 1024 ** 2)
        self.assertEqual(usage.parse_size('1504K'), 1504 * 1024)
        self.assertIsNone(usage.parse_size('MEM'))

    def test_sample_of_this_process(self):
        snapshot = usage.sample(os.getpid())
        self.assertTrue(snapshot['host'])
        self.assertEqual(snapshot['process']['pid'], os.getpid())
        self.assertGreater(snapshot['process']['memory_bytes'], 0)
        self.assertIsNone(usage.sample()['process'])


if __name__ == '__main__':
    unittest.main()
