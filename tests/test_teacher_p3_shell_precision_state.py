import tempfile
import unittest
from pathlib import Path
import numpy as np
from wind3dgs.teacher.p3_shell_precision_state import (
    extended_state,split_array,join_array,save_checkpoint,load_checkpoint,
)


class Receiver:
    preserves_extended_state=True
    def state(self,*,displacement,velocity,time_s):
        return extended_state(displacement,velocity,time_s)


@unittest.skipUnless(np.finfo(np.longdouble).nmant>=63,'확장 정밀도 필요')
class PrecisionStorageTests(unittest.TestCase):
    def test_small_parts_survive_file_roundtrip_and_no_overwrite(self):
        u=np.ones((4,3),dtype=np.longdouble)+np.longdouble(2)**-60
        v=-u*np.longdouble(0.125)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'state.npz';state=extended_state(u,v,1.25)
            save_checkpoint(p,state);restored=load_checkpoint(p,Receiver())
            np.testing.assert_array_equal(restored.displacement_m,u)
            np.testing.assert_array_equal(restored.velocity_m_s,v)
            self.assertFalse(restored.displacement_m.flags.writeable)
            self.assertTrue(np.any(restored.displacement_m!=u.astype(float)))
            with self.assertRaises(FileExistsError):save_checkpoint(p,state)
            with self.assertRaises(ValueError):load_checkpoint(p,object())

    def test_reject_noncanonical_nonfinite_and_unrepresentable(self):
        with self.assertRaises(ValueError):join_array(np.array([1.]),np.array([1.]))
        with self.assertRaises(ValueError):join_array(np.array([1.]),np.array([0],dtype=np.int64))
        with self.assertRaises(ValueError):split_array(np.array([np.nan]))
        with self.assertRaises(ValueError):split_array(np.array([np.finfo(np.longdouble).max]))
        with self.assertRaises(ValueError):extended_state(np.zeros((2,3)),np.zeros((3,3)),0.)
