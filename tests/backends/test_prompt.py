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


class PromptTests(unittest.TestCase):
    def test_prompt_matches_original_forward(self):
        # Execute the actual upstream forward with a fake model, no weights or script side effects.
        tree=ast.parse((ROOT/'generate_initial_score.py').read_text())
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='VideoAnomalyDetectionModelInference')
        fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='forward')
        ns={'torch':__import__('torch'),'generation_config':DEFAULTS['generation']}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<original forward>','exec'),ns)
        from types import SimpleNamespace
        import torch
        captured={}
        class Fake:
            def eval(self):pass
            def batch_chat(self,*args,**kw):captured.update(kw);return ['Output:\n0']
        from vera_runner.backends import upstream_questions
        obj=SimpleNamespace(model=Fake(),tokenizer=None,model_instruct=(ROOT/'VERA_learner_instruct.txt').read_text(),new_qs=upstream_questions())
        ns['forward'](obj,torch.zeros(1,8,3,2,2),[1]*8)
        self.assertEqual(prompt(DEFAULTS),captured['questions'][0])
