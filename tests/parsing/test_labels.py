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


class ParsingTests(unittest.TestCase):
    def test_parser_does_not_invent_labels(self):
        self.assertIsNone(strict_label('Cannot tell'))
        self.assertIsNone(strict_label('Output: 0 or 1'))
        self.assertEqual(strict_label('Analysis\nOutput:\n1.'),1)
