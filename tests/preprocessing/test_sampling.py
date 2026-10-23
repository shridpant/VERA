import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
from vera_runner.config import DEFAULTS,ROOT,load
from vera_runner.data import windows
from vera_runner.backends import prompt,intervene,strict_label
from vera_runner.refine import refine


class SamplingTests(unittest.TestCase):
    def test_upstream_sampling(self):
        # Independent literal upstream arithmetic, including fractional endpoints.
        for length in [17,300,2159,2580]:
            got=list(windows(length,DEFAULTS['sampling']))
            for row,anchor in zip(got,range(0,length,16)):
                start=max(anchor-.5*30*10,0);end=min(anchor+.5*30*10,length)
                want=np.arange(start,end,(end-start)/8).astype(int).tolist()
                self.assertEqual(row['frame_indices'],want)
