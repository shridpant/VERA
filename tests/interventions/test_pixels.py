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


class InterventionTests(unittest.TestCase):
    def test_interventions_exact(self):
        a=Image.fromarray(np.random.default_rng(0).integers(0,256,(448,448,3),dtype=np.uint8))
        iv=copy.deepcopy(DEFAULTS['intervention'])
        self.assertIs(intervene(a,iv),a)
        iv['mode']='grey';out=np.array(intervene(a,iv));self.assertTrue(np.all(out==127))
        from PIL import ImageFilter
        iv['mode']='blur';self.assertEqual(intervene(a,iv).tobytes(),a.filter(ImageFilter.GaussianBlur(17.92)).tobytes())
