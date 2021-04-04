# Functions sourced from bebi103a.github.io
import numpy as np
import pandas as pd
import numba
import iqplot
import colorcet
import bokeh.io

@numba.njit
def draw_bootstrap_sample(data):
    """Draw a bootstrap sample from a 1D data set."""
    return np.random.choice(data, size=len(data))

def draw_bootstrap_reps(data, stat_fun, size=1):
    """Draw boostrap replicates computed with stat_fun from 1D data set."""
    return np.array([stat_fun(draw_bootstrap_sample(data)) for _ in range(size)])


@numba.njit
def draw_bootstrap_reps_mean(data, size=1):
    """Draw boostrap replicates of the mean from 1D data set."""
    out = np.empty(size)
    for i in range(size):
        out[i] = np.mean(draw_bootstrap_sample(data))
    return out


@numba.njit
def draw_bootstrap_reps_std(data, size=1):
    """Draw boostrap replicates of the standard deviation from 1D data set."""
    out = np.empty(size)
    for i in range(size):
        out[i] = np.std(draw_bootstrap_sample(data))
    return out

def bootstrap_confidence_interval(data, size=1):
    bootstrap_reps_mean =  draw_bootstrap_reps_mean(data, size)
    np.percentile(bootstrap_reps_mean, [2.5, 97.5])

@numba.njit
def draw_bootstrap_pairs(x, y):
    """Draw a pairs bootstrap sample."""
    inds = np.arange(len(x))
    bootstrap_inds = draw_bootstrap_sample(inds)

    return x[bootstrap_inds], y[bootstrap_inds]

def draw_bootstrap_pairs_reps_bivariate(x, y, size=1):
    """
    Draw bootstrap pairs replicates.
    """
    out = np.empty(size)

    for i in range(size):
        out[i] = bivariate_r(*draw_bootstrap_pairs(x, y))

    return out

@numba.njit
def bivariate_r(x, y):
    """
    Compute plug-in estimate for the bivariate correlation coefficient.
    """
    return (
        np.sum((x - np.mean(x)) * (y - np.mean(y)))
        / np.std(x)
        / np.std(y)
        / np.sqrt(len(x))
        / np.sqrt(len(y))
    )

@numba.njit
def draw_perm_sample(x, y):
    """Generate a permutation sample."""
    concat_data = np.concatenate((x, y))
    np.random.shuffle(concat_data)

    return concat_data[:len(x)], concat_data[len(x):]


def draw_perm_reps(x, y, stat_fun, size=1):
    """Generate array of permuation replicates."""
    return np.array([stat_fun(*draw_perm_sample(x, y)) for _ in range(size)])

@numba.njit
def draw_perm_reps_diff_mean(x, y, size=1):
    """Generate array of permuation replicates."""
    out = np.empty(size)
    for i in range(size):
        x_perm, y_perm = draw_perm_sample(x, y)
        out[i] = np.mean(x_perm) - np.mean(y_perm)

    return out

def perm_result(x, y, size=10000000):
    # Compute test statistic for original data set
    diff_mean = np.mean(x) - np.mean(y)

    # Draw replicates
    perm_reps = draw_perm_reps_diff_mean(x, y, size=1)

    # Compute p-value
    p_val = np.sum(perm_reps >= diff_mean) / len(perm_reps)

    print('p-value =', p_val)

def bootstrap_graph(df, q_name, cat, x_label, x, y, x_name, y_name):
    p = iqplot.ecdf(
        df,
        q=q_name,
        cats=cat,
        x_axis_label=x_label,
    )

    for _ in range(100):
        x_rep = draw_bootstrap_sample(x)
        y_rep = draw_bootstrap_sample(y)
        df_rep = pd.DataFrame(
        data={
            cat: [x_name] * len(x_rep) + [y_name] * len(y_rep),
            q_name: np.concatenate((x_rep, y_rep)),
        }
    )

        p = iqplot.ecdf(
            df_rep,
            q=q_name,
            cats=cat,
            p=p,
            marker_kwargs=dict(alpha=0.02),
        )

    bokeh.io.show(p)