"""Standalone JAX forward model for circuit/readout identifiability spike.

NOT part of the NUDGE repo. Minimal, self-contained.

Composition observed per cell:
    u  ~ lognormal            (extrinsic input spread, latent/unobserved)
    a  = g(u; theta)          circuit map (Hill, real switch n>1)
    Lam= R(a; phi)            readout map (Hill, saturating h>=1)
    y  ~ counts(mean=Lam)     moment-matched NB (differentiable, reparam)

Only y is observed. We ask whether (theta, phi) can be separated.
"""
import jax
import jax.numpy as jnp

# ---------- maps ----------

def circuit(u, K, n, vmax, basal):
    """Circuit activity a = g(u). Hill with real ultrasensitivity (n>1)."""
    un = u ** n
    return basal + vmax * un / (K ** n + un)

def readout(a, Km, h, Vmax, b):
    """Measurement mean Lambda = R(a). Saturating Hill readout."""
    ah = a ** h
    return b + Vmax * ah / (Km ** h + ah)

# ---------- noise (moment-matched negative binomial, reparameterized) ----------

def sample_counts(key, mean, phi):
    """Moment-matched Gaussian surrogate for NB: var = mean + mean^2/phi.

    Reparameterized (mean + sd*z) so gradients flow through mean into params.
    Clipped at 0. phi -> inf recovers Poisson-like; small phi = overdispersed.
    """
    var = mean + mean ** 2 / phi
    z = jax.random.normal(key, mean.shape)
    y = mean + jnp.sqrt(var) * z
    return jnp.clip(y, 0.0)

# ---------- generative pipeline ----------

def gen_input(key, n_cells, mu_log=0.0, sd_log=0.6):
    """Extrinsic input u across cells: lognormal centered near circuit K."""
    return jnp.exp(mu_log + sd_log * jax.random.normal(key, (n_cells,)))

def forward_counts(key, u, theta, phi):
    """u -> a -> Lambda -> counts. theta packs circuit+readout+floors."""
    K, n, vmax, basal, Km, h, Vmax, b = theta
    a = circuit(u, K, n, vmax, basal)
    Lam = readout(a, Km, h, Vmax, b)
    return sample_counts(key, Lam, phi), a, Lam

# ---------- energy distance on log1p counts (1D) ----------

def _pdist(x, y):
    return jnp.abs(x[:, None] - y[None, :])

def energy_distance(x, y):
    """Cramer/energy distance between 1D samples x,y (already transformed)."""
    exy = _pdist(x, y).mean()
    exx = _pdist(x, x).mean()
    eyy = _pdist(y, y).mean()
    return 2.0 * exy - exx - eyy

def log1p(y):
    return jnp.log1p(y)
