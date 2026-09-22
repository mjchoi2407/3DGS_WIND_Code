import csv
import json
from pathlib import Path

import tempfile
import unittest
from unittest.mock import patch

from wind3dgs.evaluation import teacher_precision_profile as profile


def kernel_csv(folder):
    path = folder/'cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv'
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Time (%)', 'Total Time (ns)', 'Instances', 'Name'])
        writer.writerows([
            [1, 20, 1, 'rare<double>'],
            [30, '3,000', 100, 'frequent<double, 3>'],
            [20, 2000, 2, 'second(float*)'],
            [10, 1000, 1, 'third'],
        ])
    return path


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def test_rank_uses_total_and_preserves_demangled_name(self):
        kernel_csv(self.folder)
        rows = profile.top_kernels(self.folder)
        self.assertEqual([row['name'] for row in rows], ['frequent<double, 3>', 'second(float*)', 'third'])
        self.assertEqual(rows[0]['instances'], 100)

    def test_unknown_units_not_guessed(self):
        path = kernel_csv(self.folder)
        path.write_text(path.read_text().replace('Total Time (ns)', 'Total Time (us)'))
        with self.assertRaisesRegex(ValueError, 'CSV'):
            profile.top_kernels(self.folder)

    def test_ncu_failure_preserves_prior_results_and_attempts_other_kernels(self):
        kernel_csv(self.folder)
        ordinary = self.folder/'ordinary.csv'
        ordinary.write_text('기존 원시값\n')
        commands = []

        def failed(out, name, command, **kwargs):
            commands.append(command)
            return {'returncode': 1}

        with patch.object(profile, 'run_command', failed):
            status = profile.compute_profiles(self.folder, 'ncu', ['python', 'worker'], {})
        self.assertEqual(status, 'partial_or_failed')
        self.assertEqual(len(commands), 3)
        self.assertEqual(len({cmd[cmd.index('--worker')+1] for cmd in commands}), 3)
        self.assertTrue(all(cmd[cmd.index('--launch-count')+1] == '1' for cmd in commands))
        self.assertEqual(ordinary.read_text(), '기존 원시값\n')
        report = json.loads((self.folder/'ncu_status.json').read_text())
        self.assertTrue(all(row['status'] == 'failed' for row in report['rows']))
