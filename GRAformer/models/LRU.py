"""
Simplified Implementation of the Linear Recurrent Unit
------------------------------------------------------
We present here a simplified JAX implementation of the Linear Recurrent Unit (LRU). 
The state of the LRU is driven by the input $(u_k)_{k=1}^L$ of sequence length $L$ 
according to the following formula (and efficiently parallelized using an associative scan): 
$x_{k} = \Lambda x_{k-1} +\exp(\gamma^{\log})\odot (B u_{k})$, 
and the output is computed at each timestamp $k$ as follows: $y_k = C x_k + D u_k$. 
In our code, $B,C$ follow Glorot initialization, with $B$ scaled additionally by a factor 2 
to account for halving the state variance by taking the real part of the output projection.  
$D$ is random $H$-dimensional and mutiplies elementwise each $u_k$, where $k$ is the timestamp. 
$\Lambda$ is initialized with the help of Lemma, with phase potentially restricted to a thin slice
"""
import jax
import jax.numpy as jnp
import numpy as np

parallel_scan = jax.lax.associative_scan

# https://readpaper.com/pdf-annotate/note?pdfId=4733214820363419649&noteId=1787878258795028480
def forward(lru_parameters, input_sequence):
    """Forward pass of the LRU layer. Output y and input_sequence are of shape (L, H)."""

    # All LRU parameters
    nu_log, theta_log, B_re, B_im, C_re, C_im, D, gamma_log = lru_parameters

    # Materializing the diagonal of Lambda and projections
    Lambda = jnp.exp(-jnp.exp(nu_log) + 1j * jnp.exp(theta_log))
    B_norm = (B_re + 1j * B_im) * jnp.expand_dims(jnp.exp(gamma_log), axis=-1)
    C = C_re + 1j * C_im

    # Running the LRU + output projection
    # For details on parallel scan, check discussion in Smith et al (2022).
    Lambda_elements = jnp.repeat(Lambda[None, ...], input_sequence.shape[0], axis=0)
    Bu_elements = jax.vmap(lambda u: B_norm @ u)(input_sequence)
    elements = (Lambda_elements, Bu_elements)
    _, inner_states = parallel_scan(binary_operator_diag, elements)  # all x_k
    y = jax.vmap(lambda x, u: (C @ x).real + D * u)(inner_states, input_sequence)  #  @ 矩阵相乘  *  相位乘

    return y


def init_lru_parameters(N, H, r_min=0, r_max=1, max_phase=6.28):
    """Initialize parameters of the LRU layer."""

    # N: state dimension, H: model dimension
    # Initialization of Lambda is complex valued distributed uniformly on ring 
    # between r_min and r_max, with phase in [0, max_phase].
    u1 = np.random.uniform(size=(N,))
    u2 = np.random.uniform(size=(N,))
    nu_log = np.log(-0.5 * np.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
    theta_log = np.log(max_phase * u2)

    # Glorot initialized Input/Output projection matrices
    B_re = np.random.normal(size=(N, H)) / np.sqrt(2 * H)
    B_im = np.random.normal(size=(N, H)) / np.sqrt(2 * H)
    C_re = np.random.normal(size=(H, N)) / np.sqrt(N)
    C_im = np.random.normal(size=(H, N)) / np.sqrt(N)
    D = np.random.normal(size=(H,))

    # Normalization factor
    diag_lambda = np.exp(-np.exp(nu_log) + 1j * np.exp(theta_log))
    gamma_log = np.log(np.sqrt(1 - np.abs(diag_lambda) ** 2))

    return nu_log, theta_log, B_re, B_im, C_re, C_im, D, gamma_log


def binary_operator_diag(element_i, element_j):
    """Binary operator for parallel scan of linear recurrence."""
    a_i, bu_i = element_i
    a_j, bu_j = element_j
    return a_j * a_i, a_j * bu_i + bu_j