import os

import jax.numpy as jnp
import numpy as np

_Jd_np = np.load(os.path.join(os.path.dirname(__file__), "Jd.npz"))
_Jd_np = tuple(_Jd_np[key] for key in _Jd_np.files)
_Jd_jax = tuple(jnp.array(jd) for jd in _Jd_np)
