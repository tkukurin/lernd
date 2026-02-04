"""Tests for lernd.generator module."""

import pytest

from lernd import classes as c
from lernd import generator as g
from lernd import util as u
from lernd.lernd_types import RuleTemplate


class TestClauseValidation:
    def test_check_clause_unsafe_safe_clauses(self):
        safe_clauses = ['p(X)<-q(X)', 'p(X)<-q(X), q(Y)']
        for safe_clause_str in safe_clauses:
            result = g.check_clause_unsafe(c.Clause.from_str(safe_clause_str))
            assert result is False

    def test_check_clause_unsafe_unsafe_clauses(self):
        unsafe_clauses = ['p(X)<-q(Y)', 'p(X)<-q(Z), q(Y)']
        for unsafe_clause_str in unsafe_clauses:
            result = g.check_clause_unsafe(c.Clause.from_str(unsafe_clause_str))
            assert result is True

    def test_check_circular(self):
        circular_clauses = ['p(X,Y)<-p(X,Y), q(Y)']
        uncircular_clauses = ['p(X,Y)<-p(Y,X), q(Y)']
        for circular_clause_str in circular_clauses:
            result = g.check_circular(c.Clause.from_str(circular_clause_str))
            assert result is True
        for uncircular_clause_str in uncircular_clauses:
            result = g.check_circular(c.Clause.from_str(uncircular_clause_str))
            assert result is False


class TestClauseGeneration:
    def test_cl_no_intensional(self):
        """Test clause generation without intensional predicates."""
        preds_ext = [u.str2pred('p/2')]
        preds_int = [u.str2pred('q/2')]
        pred = u.str2pred('q/2')
        tau = RuleTemplate(0, False)
        expected_clauses = [
            'q(A,B)<-p(A,A), p(A,B)',
            'q(A,B)<-p(A,A), p(B,A)',
            'q(A,B)<-p(A,A), p(B,B)',
            'q(A,B)<-p(A,B), p(A,B)',
            'q(A,B)<-p(A,B), p(B,A)',
            'q(A,B)<-p(A,B), p(B,B)',
            'q(A,B)<-p(B,A), p(B,A)',
            'q(A,B)<-p(B,A), p(B,B)'
        ]
        clauses = g.cl(preds_int, preds_ext, pred, tau)
        for i, clause in enumerate(clauses):
            assert str(clause) == expected_clauses[i]

    def test_cl_with_intensional(self):
        """Test clause generation with intensional predicates required."""
        preds_ext = [u.str2pred('p/2')]
        preds_int = [u.str2pred('q/2')]
        pred = u.str2pred('q/2')
        tau = RuleTemplate(1, True)
        expected_clauses = [
            'q(A,B)<-p(A,A), q(B,A)',
            'q(A,B)<-p(A,A), q(B,B)',
            'q(A,B)<-p(A,A), q(B,C)',
            'q(A,B)<-p(A,A), q(C,B)',
            'q(A,B)<-p(A,B), q(A,A)',
            'q(A,B)<-p(A,B), q(A,C)',
            'q(A,B)<-p(A,B), q(B,A)',
            'q(A,B)<-p(A,B), q(B,B)',
            'q(A,B)<-p(A,B), q(B,C)',
            'q(A,B)<-p(A,B), q(C,A)',
            'q(A,B)<-p(A,B), q(C,B)',
            'q(A,B)<-p(A,B), q(C,C)',
            'q(A,B)<-p(A,C), q(B,A)',
            'q(A,B)<-p(A,C), q(B,B)',
            'q(A,B)<-p(A,C), q(B,C)',
            'q(A,B)<-p(A,C), q(C,B)'
        ]
        expected_total = 58
        clauses = g.cl(preds_int, preds_ext, pred, tau)
        assert len(clauses) == expected_total
        for clause, expected_clause in zip(clauses, expected_clauses):
            assert str(clause) == expected_clause
