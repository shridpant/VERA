import copy,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import audit3b_r1 as a
import validate3b as v

class AmendmentTests(unittest.TestCase):
    def fixture(self):
        h=['%064x'%i for i in range(8)]
        base={'window_uid':'A@0','family':'none','mask':[],'slot_tensor_sha256':h,'sequence_key':'base','slot_identity_ok':True,'cache_status':'fresh','target':v.ORIG,'_src':'baseline:1'}
        mask={**base,'family':'blur','mask':[0],'sequence_key':'blur','slot_identity_ok':False,'_src':'fixed:1'}
        inp={'video_sha256':'a'*64,'frame_sha256':['b'*64]*8,'frame_indices':list(range(8))}
        return SimpleNamespace(lines=[base,mask],inputs_by_uid={'A@0':inp})
    def check(self,run,proof=True):
        return {x.id:x for x in a.revised_checks(run,{('A@0',0):{}} if proof else {})}
    def test_historical_boundary(self):
        self.assertTrue(a.readout_ok(36.12516,36.0));self.assertNotEqual(v.bf16(36.12516),36.0)
    def test_large_error_rejected(self): self.assertFalse(a.readout_ok(36.127,36.0))
    def test_nonfinite_rejected(self): self.assertFalse(a.readout_ok(float('nan'),36.0))
    def test_non_bf16_native_rejected(self): self.assertFalse(a.readout_ok(36.01,36.01))
    def test_proved_noop(self): self.assertEqual(self.check(self.fixture())['slot_identity'].n_discrepancies,0)
    def test_unproved_noop_rejected(self): self.assertGreater(self.check(self.fixture(),False)['slot_identity'].n_discrepancies,0)
    def test_grey_noop_rejected(self):
        r=self.fixture();r.lines[1]['family']='grey';self.assertGreater(self.check(r)['slot_identity'].n_discrepancies,0)
    def test_changed_unmasked_rejected(self):
        r=self.fixture();r.lines[1]['slot_tensor_sha256']=list(r.lines[1]['slot_tensor_sha256']);r.lines[1]['slot_tensor_sha256'][1]='f'*64
        self.assertGreater(self.check(r)['slot_identity'].n_discrepancies,0)
    def test_false_legacy_flag_rejected(self):
        r=self.fixture();r.lines[1]['slot_identity_ok']=True;self.assertGreater(self.check(r)['slot_hashes_present'].n_discrepancies,0)
    def test_duplicate_source_allowed_and_changed_source_rejected(self):
        r=self.fixture();alias=copy.deepcopy(r.lines[0]);alias['window_uid']='B@0';r.lines.append(alias)
        r.inputs_by_uid['B@0']=copy.deepcopy(r.inputs_by_uid['A@0'])
        self.assertEqual(self.check(r)['mask_hash_consistency'].n_discrepancies,0)
        r.inputs_by_uid['B@0']['video_sha256']='c'*64
        self.assertGreater(self.check(r)['mask_hash_consistency'].n_discrepancies,0)
    def test_different_pixels_same_key_rejected(self):
        r=self.fixture();alias=copy.deepcopy(r.lines[0]);alias['slot_tensor_sha256'][2]='f'*64;r.lines.append(alias)
        self.assertGreater(self.check(r)['mask_hash_consistency'].n_discrepancies,0)
if __name__=='__main__':unittest.main()
