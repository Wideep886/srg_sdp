"""Tests of necessary conditions, exact projections, and unsupported cases."""

from fractions import Fraction as Q
from itertools import combinations
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))
from parameters import analyze


class ParameterAnalysisTests(unittest.TestCase):
    def test_known_primitive_embeddings(self):
        examples = [
            ((10, 3, 0, 1), 's', -2, 4, '-2/3', '1/6'),  # Petersen
            ((16, 6, 2, 2), 'r', 2, 6, '-1/3', '1/3'),  # 4-by-4 rook
            ((21, 10, 5, 4), 'r', 3, 6, '-2/5', '3/10'),  # triangular T(7)
            ((9, 4, 1, 2), 'r', 1, 4, '-1/2', '1/4'),  # square discriminant conference
            ((288, 105, 52, 30), 'r', 25, 27, '-1/7', '5/21'),
        ]
        for parameters, label, eigenvalue, dimension, a, b in examples:
            with self.subTest(parameters=parameters):
                result = analyze(parameters)
                self.assertEqual(result['status'], 'supported', result['reasons'])
                self.assertEqual(result['selected_eigenspace'], label)
                self.assertEqual(result['selected_eigenvalue'], eigenvalue)
                self.assertEqual((result['n'], result['a'], result['b']), (dimension, a, b))
                self.assertTrue(all(result['exact_checks'].values()))
                self.assertEqual(result['graph_existence'], 'not_determined')
                self.assertEqual(result['core_scope']['maximum_points'], min(6, dimension))
                json.dumps(result)

    def test_explicit_eigenspaces_and_complement(self):
        petersen = analyze([10, 3, 0, 1], 'r')
        self.assertEqual((petersen['n'], petersen['a'], petersen['b']), (5, '-1/3', '1/3'))
        self.assertEqual(petersen['complement']['parameters'], [10, 6, 3, 4])
        triangular = analyze(petersen['complement']['parameters'], 's')
        self.assertEqual((triangular['n'], triangular['a'], triangular['b']),
                         (petersen['n'], petersen['a'], petersen['b']))
        self.assertEqual(triangular['selected_eigenvalue'], -1 - petersen['selected_eigenvalue'])
        first = petersen['embedding_candidates']['r']
        second = triangular['embedding_candidates']['s']
        self.assertEqual(first['inner_product_adjacent'], second['inner_product_nonadjacent'])
        self.assertEqual(first['inner_product_nonadjacent'], second['inner_product_adjacent'])

    def test_petersen_projectors_against_actual_adjacency_matrix(self):
        vertices = [frozenset(pair) for pair in combinations(range(5), 2)]
        adjacency = [[int(not x.intersection(y)) for y in vertices] for x in vertices]
        result = analyze((10, 3, 0, 1))
        for label, candidate in result['embedding_candidates'].items():
            with self.subTest(eigenspace=label):
                coefficients = {key: Q(value) for key, value in candidate['projector_coefficients'].items()}
                matrix = [[coefficients['I'] * int(i == j) + coefficients['A'] * adjacency[i][j]
                           + coefficients['J'] for j in range(10)] for i in range(10)]
                diagonal = Q(candidate['projector_diagonal'])
                for i in range(10):
                    self.assertEqual(sum(matrix[i]), 0)
                    self.assertEqual(matrix[i][i], diagonal)
                    for j in range(10):
                        self.assertEqual(sum(matrix[i][t] * matrix[t][j] for t in range(10)), matrix[i][j])
                        self.assertEqual(sum(adjacency[i][t] * matrix[t][j] for t in range(10)),
                                         candidate['eigenvalue'] * matrix[i][j])
                        if i != j:
                            expected = candidate['inner_product_adjacent' if adjacency[i][j]
                                                 else 'inner_product_nonadjacent']
                            self.assertEqual(matrix[i][j] / diagonal, Q(expected))

    def test_imprimitive_auto_avoids_collapsed_projection(self):
        cases = [
            ((9, 2, 1, 0), 's', 6, '-1/2', '0'),  # three disjoint triangles
            ((6, 1, 0, 0), 's', 3, '-1', '0'),  # three disjoint edges
            ((6, 4, 2, 4), 'r', 3, '-1', '0'),  # K(2,2,2)
            ((6, 3, 0, 3), 'r', 4, '-1/2', '0'),  # K(3,3)
            ((4, 2, 0, 2), 'r', 2, '-1', '0'),  # C4
        ]
        for parameters, chosen, n, a, b in cases:
            with self.subTest(parameters=parameters):
                result = analyze(parameters)
                self.assertEqual(result['status'], 'supported', result['reasons'])
                self.assertEqual((result['selected_eigenspace'], result['n'], result['a'], result['b']),
                                 (chosen, n, a, b))
                collapsed = 's' if chosen == 'r' else 'r'
                other = analyze(parameters, collapsed)
                self.assertEqual(other['status'], 'unsupported')
                self.assertFalse(other['embedding_candidates'][collapsed]['injective'])
                self.assertIsNone(other['n'])
                self.assertEqual(other['graph_existence'], 'not_determined')

    def test_conference_irrational_angles_are_not_rounded_or_rejected(self):
        for parameters, dimension, discriminant in [((5, 2, 0, 1), 2, 5), ((13, 6, 2, 3), 6, 13)]:
            for eigenspace in ('auto', 'r', 's'):
                with self.subTest(parameters=parameters, eigenspace=eigenspace):
                    result = analyze(parameters, eigenspace)
                    self.assertEqual(result['status'], 'unsupported')
                    self.assertEqual(result['graph_existence'], 'not_determined')
                    self.assertIsNone(result['a'])
                    self.assertIsNone(result['b'])
                    self.assertEqual(result['spectrum']['r_multiplicity'], dimension)
                    self.assertEqual(result['spectrum']['s_multiplicity'], dimension)
                    self.assertEqual(result['spectrum']['r'], '(-1 + sqrt({}))/2'.format(discriminant))
                    self.assertTrue(all(result['exact_checks'].values()))
                    self.assertFalse(result['embedding_candidates']['r']['rational'])
                    self.assertIn('quadratic irrational', result['reasons'][0])
                    json.dumps(result)

    def test_exact_obstructions(self):
        examples = [
            ((10, 3, 0, 2), 'parameter_identity'),
            ((5, 1, 0, 0), 'edge_count_integral'),
            ((10, 3, 2, 0), 'multiplicities_positive_integral'),
            ((9, 4, 0, 3), 'irrational_multiplicity_coefficient_zero'),
            ((4, 2, 1, 0), 'mu_range'),
            ((4, 4, 0, 0), 'valency_range'),
        ]
        for parameters, failed_check in examples:
            with self.subTest(parameters=parameters):
                result = analyze(parameters)
                self.assertEqual(result['status'], 'infeasible')
                self.assertFalse(result['exact_checks'][failed_check])
                self.assertEqual(result['graph_existence'], 'ruled_out_by_necessary_conditions')
                self.assertIsNone(result['n'])

    def test_degenerate_graphs_do_not_divide_by_zero(self):
        # lambda is vacuous for an empty graph; mu is vacuous for a complete graph.
        for parameters in [(1, 0, 42, 7), (5, 0, 42, 0), (5, 4, 3, 42), (2, 1, 0, 0)]:
            with self.subTest(parameters=parameters):
                result = analyze(parameters)
                self.assertEqual(result['status'], 'unsupported')
                self.assertEqual(result['graph_existence'], 'not_determined')
        for parameters in [(5, 0, 0, 1), (5, 4, 2, 0)]:
            self.assertEqual(analyze(parameters)['status'], 'infeasible')

    def test_input_validation_and_mapping(self):
        self.assertEqual(analyze({'v': '10', 'k': 3, 'lambda_': 0, 'mu': 1})['status'], 'supported')
        for parameters in [None, '10,3,0,1', [10, 3, 0], [True, 3, 0, 1], [10, 3.0, 0, 1],
                           [10, '3.0', 0, 1], [0, 0, 0, 0], [10, 3, -1, 1],
                           {'v': 10, 'valency': 3, 'k': 4, 'lambda': 0, 'mu': 1}, {'v': 10}]:
            with self.subTest(parameters=parameters):
                self.assertEqual(analyze(parameters)['status'], 'invalid')
        self.assertEqual(analyze([10, 3, 0, 1], 'minimum')['status'], 'invalid')


if __name__ == '__main__':
    unittest.main()
