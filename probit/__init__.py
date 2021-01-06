"""
Probit.

A module for multinomial probit regression with GP priors.
"""
from .gibbs import GibbsClassifier
from .variational_bayes import VariationBayesClassifier

__all__ = [
    "GibbsClassifier",
    "VariationalBayesClassifier"
    ]
