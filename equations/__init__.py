from .base import PDEBase
from .heat import Heat1D
from .schrodinger import Schrodinger1D
from .kdv import KdV
from .navier_stokes import NavierStokes2D

__all__ = [
    'PDEBase',
    'Heat1D',
    'Schrodinger1D',
    'KdV',
    'NavierStokes2D',
] 