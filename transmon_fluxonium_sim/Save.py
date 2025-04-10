import os
import sys
import cmcrameri.cm as cmc  # scientific colormaps, perceptually uniform for all.
from cmcrameri import show_cmaps  # demonstrating all the colormaps


def save_instant():
    """
    When ever you are adjusting the different parameters,i want to be able to save the instant of the plot, such that the potential, the eigenenergies and the different energies are shown in the plot, when then additionally i want it to save a meta data file, where the informations are also saved.
    """
    curstate().save()
