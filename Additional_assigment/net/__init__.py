import numpy as np

LEARNING_PHASE = False
DTYPE = np.float32


def dtype():
  return DTYPE


def set_learning_phase(phase):
  global LEARNING_PHASE
  LEARNING_PHASE = phase


def learning_phase():
  return LEARNING_PHASE
