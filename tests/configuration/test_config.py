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


class ConfigurationTests(unittest.TestCase):
    def test_unknown_setting_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'c.json';p.write_text('{"intervention":{"radius":10}}')
            with self.assertRaises(ValueError):load(p)
