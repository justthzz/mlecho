"""Poison detection (the smoke alarm).

Check 1 (picture): a CNN-VAE trained only on trusted clean images; an image it
can't rebuild well is suspicious. Check 2 (label): a clean reference model;
a label it disagrees with is suspicious. Part of the guard: imports only
`mlecho.core`.
"""
