"""Data layer: bronze->silver->gold transforms and refresh-strategy experiments.

Marks ``data`` as a regular package so ``python -m data.transforms.cli`` and
``from data.transforms import ...`` resolve without relying on implicit
namespace-package behaviour.
"""
