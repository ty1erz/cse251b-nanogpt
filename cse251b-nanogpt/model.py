"""Submission entry point for evaluate.py."""

# Use V5 checkpoints by default.
# from model_v5 import load_model
# For model_final checkpoints instead, comment the line above and uncomment this:
from model_final import load_model

__all__ = ["load_model"]
