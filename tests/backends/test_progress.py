import unittest
import torch
from types import SimpleNamespace
from vera_runner.backends import install_progress_hooks

class ProgressHookTests(unittest.TestCase):
    def test_hooks_preserve_values_and_distinguish_forward_calls(self):
        vision=torch.nn.Linear(2,2);language=torch.nn.Linear(2,2)
        model=SimpleNamespace(vision_model=vision,language_model=language)
        x=torch.tensor([[1.,2.]])
        expected=language(vision(x));events=[]
        handles=install_progress_hooks(model,lambda stage,details:events.append((stage,dict(details))))
        actual=language(vision(x));language(actual)
        self.assertTrue(torch.equal(actual,expected))
        self.assertEqual([s for s,_ in events],['vision_forward_begin','vision_forward_end','language_forward_begin','language_forward_end','language_forward_begin','language_forward_end'])
        self.assertEqual(events[-1][1]['forward_index'],2)
        for handle in handles:handle.remove()
