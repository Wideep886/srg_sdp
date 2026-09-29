"""Regression checks for SDPA transport of empty logical kernel blocks."""
from fractions import Fraction as F
import importlib.util
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[2] / 'code' / 'check_solver_io.py'
SPEC = importlib.util.spec_from_file_location('zero_block_export_audit', SOURCE)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class ZeroBlockTransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'input.txt'
        self.addCleanup(self.temp.cleanup)

    def write(self, text):
        self.path.write_text(text)
        return self.path

    def test_sparse_solver_ids_restore_logical_ids(self):
        dimensions = [0, 1, 0, 2, 0, 1, 0]
        source = self.write('1\n3\n1 2 1\n-2\n'
                            '0 1 1 1 -1\n1 2 1 2 0.125\n1 3 1 1 1\n')
        rhs, entries = audit.parse_dat(source, 1, dimensions)
        self.assertEqual(rhs, [F(-2)])
        self.assertEqual(entries, {(0, 2, 1, 1): F(-1),
                                   (1, 4, 1, 2): F(1, 8),
                                   (1, 6, 1, 1): F(1)})

    def test_solver_matrices_restore_empty_blocks(self):
        source = self.write('yMat = {{1.25} {{2,0.5},{0.5,3}} {{4}}}\n')
        self.assertEqual(audit.parse_solver_y_mat(source, [0, 1, 0, 2, 0, 1, 0]),
                         [[], [[F(5, 4)]], [], [[F(2), F(1, 2)], [F(1, 2), F(3)]],
                          [], [[F(4)]], []])

    def test_all_positive_transport_is_unchanged(self):
        source = self.write('1\n2\n1 1\n0\n0 1 1 1 -1\n1 2 1 1 1\n')
        _, entries = audit.parse_dat(source, 1, [1, 1])
        self.assertEqual(entries, {(0, 1, 1, 1): F(-1), (1, 2, 1, 1): F(1)})
        source = self.write('yMat = {{1},{{2}}}\n')
        self.assertEqual(audit.parse_solver_y_mat(source, [1, 1]), [[[F(1)]], [[F(2)]]])

    def test_zero_solver_dimension_is_rejected(self):
        source = self.write('1\n3\n1 0 1\n0\n')
        with self.assertRaises(ValueError):
            audit.parse_dat(source, 1, [1, 0, 1])

    def test_empty_solver_matrix_is_not_a_logical_placeholder(self):
        for value in ('yMat = {{1} {} {2}}\n', 'yMat = {{1}}\n',
                      'yMat = {{1} {2} {3}}\n'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit.parse_solver_y_mat(self.write(value), [1, 0, 1])

    def test_invalid_logical_dimensions_are_rejected(self):
        for dimensions in ([1, -1], [1, True], [0], []):
            with self.subTest(dimensions=dimensions), self.assertRaises(ValueError):
                audit.solver_block_layout(dimensions)


if __name__ == '__main__':
    unittest.main()
