"""중단 시 worker 종료·다음 씬 미실행·확정 기록 보존 검사."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from wind3dgs.evaluation import teacher_cloth_gpu_sweep as m


class InterruptTests(unittest.TestCase):
    def test_owned_process_exits(self):
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
        try:
            m.terminate_worker(child)
            self.assertIsNotNone(child.poll())
        finally:
            if child.poll() is None:child.kill();child.wait()

    def test_timeout_escalates_for_owned_worker(self):
        child=Mock();child.poll.return_value=None
        child.wait.side_effect=[subprocess.TimeoutExpired('worker',2),-9]
        m.terminate_worker(child)
        child.terminate.assert_called_once();child.kill.assert_called_once()

    def test_ctrl_c_stops_sweep_and_preserves_committed_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'baseline/reference_rectangle';folder.mkdir(parents=True)
            report={'status':'running','completed_frames':120,'chunks':[{'path':'kept.npz'}]}
            (folder/'report.json').write_text(json.dumps(report))
            child=Mock();child.poll.return_value=None;child.wait.side_effect=[KeyboardInterrupt(),0]
            with patch.object(m,'verify_sweep',return_value={'cases':['baseline'],'cudss_sha256':'hash'}),patch.object(m,'digest',return_value='hash'),patch.object(m.subprocess,'Popen',return_value=child) as launch:
                self.assertEqual(m.run(root),130)
                launch.assert_called_once()
                self.assertTrue(launch.call_args.kwargs['start_new_session'])
            saved=json.loads((folder/'report.json').read_text())
            self.assertEqual(saved['status'],'interrupted')
            self.assertEqual(saved['completed_frames'],120)
            self.assertEqual(saved['chunks'],report['chunks'])

if __name__=='__main__':unittest.main()
