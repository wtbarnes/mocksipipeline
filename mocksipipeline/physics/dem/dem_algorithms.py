"""
Implementations of DEM algorithms
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve


__all__ = ['simple_reg_dem', 'sparse_em_init', 'sparse_em_solve']


def simple_reg_dem(data, errors, exptimes, logt, tresps,
                   kmax=100, kcon=5, steps=[0.1, 0.5], drv_con=8.0, chi2_th=1.0, tol=0.1):
    """
    Computes a DEM given a set of input data, errors, exposure times, temperatures, and
    temperature response functions.

    Author: Joseph Plowman -- 09-15-2021
    See: https://ui.adsabs.harvard.edu/abs/2020ApJ...905...17P/abstract

    Parameters
    ----------
    data: image data (dimensions n_x by n_y by n_channels)
    errors: image of uncertainties in data (dimensions n_x by n_y by n_channels)
    exptimes: exposure times for each image (dimensions n_channels)
    logt: Array of temperatures (dimensions n_temperatures)
    tresps: Arrays of temperature response functions (dimensions n_temperatures by n_channels)
    kmax:
        The maximum number of iteration steps (default - 100).
    kcon:
        Initial number of steps before terminating if chi^2 never improves (default - 5).
    steps:
        Two element array containing large and small step sizes (default - 0.1 and 0.5).
    drv_con:
        The size of the derivative constraint - threshold delta_0 limiting the change in
        log DEM per unit input temperature (default - 8; e.g., per unit log_{10}(T)).
    chi2_th:
        Reduced chi^2 threshold for termination (default - 1).
    tol:
        How close to chi2_th the reduced chi^2 needs to be (default - 0.1).

    Returns
    --------
    dem: (dimensions n_x by n_y by n_temperatures).
        Dimensions depend on the units of logt (need not be log temperature, despite the name) 
        and tresps. For instance, if tresps has units cm^5 and the logt values 
        are base 10 log of temperature in Kelvin, the output DEM will be 
        in units of cm^{-5} per unit Log_{10}(T).
    chi2: Array of reduced chi squared values (dimensions n_x by n_y)
    """
    # Setup some values
    nt, nd = tresps.shape
    nt_ones = np.ones(nt)
    nx, ny, nd = data.shape
    dT = logt[1:nt] - logt[:nt-1]
    dTleft = np.diag(np.hstack((dT, 0)))
    dTright = np.diag(np.hstack((0, dT)))
    idTleft = np.diag(np.hstack((1.0/dT, 0)))
    idTright = np.diag(np.hstack((0, 1.0/dT)))
    Bij = ((dTleft+dTright)*2.0 + np.roll(dTright, -1, axis=0) + np.roll(dTleft, 1, axis=0))/6.0

    # Matrix mapping coefficents to data
    Rij = np.matmul((tresps*np.outer(nt_ones, exptimes)).T, Bij)
    Dij = idTleft+idTright - np.roll(idTright, -1, axis=0) - np.roll(idTleft, 1, axis=0)
    regmat = Dij*nd/(drv_con**2*(logt[nt-1]-logt[0]))
    rvec = np.sum(Rij, axis=1)

    # Precompute some values before looping over pixels
    dat0_mat = np.clip(data, 0.0, None)
    s_mat = np.sum(rvec * ((dat0_mat > 1.0e-2)/errors**2), axis=2) / np.sum((rvec/errors)**2, axis=2)
    s_mat = np.log(s_mat[..., np.newaxis] / nt_ones)

    dems = np.zeros((nx, ny, nt))
    chi2 = np.zeros((nx, ny)) - 1.0
    for i in range(0, nx):
        for j in range(0, ny):
            err = errors[i, j, :]
            dat0 = dat0_mat[i, j, :]
            s = s_mat[i, j, :]
            for k in range(0, kmax):
                # Correct data by f(s)-s*f'(s)
                dat = (dat0-np.matmul(Rij, ((1-s)*np.exp(s)))) / err
                mmat = Rij*np.outer(1.0/err, np.exp(s))
                amat = np.matmul(mmat.T, mmat) + regmat
                try:
                    [c, low] = cho_factor(amat)
                except (np.linalg.LinAlgError, ValueError):
                    break
                c2p = np.mean((dat0-np.dot(Rij, np.exp(s)))**2/err**2)
                deltas = cho_solve((c, low), np.dot(mmat.T, dat))-s
                deltas *= np.clip(np.max(np.abs(deltas)), None, 0.5/steps[0])/np.max(np.abs(deltas))
                ds = 1-2*(c2p < chi2_th)  # Direction sign; is chi squared too large or too small?
                c20 = np.mean((dat0-np.dot(Rij, np.exp(s+deltas*ds*steps[0])))**2.0/err**2.0)
                c21 = np.mean((dat0-np.dot(Rij, np.exp(s+deltas*ds*steps[1])))**2.0/err**2.0)
                interp_step = ((steps[0]*(c21-chi2_th)+steps[1]*(chi2_th-c20))/(c21-c20))
                s += deltas*ds*np.clip(interp_step, steps[0], steps[1])
                chi2[i, j] = np.mean((dat0-np.dot(Rij, np.exp(s)))**2/err**2)
                condition_1 = (ds*(c2p-c20)/steps[0] < tol)*(k > kcon)
                condition_2 = np.abs(chi2[i, j]-chi2_th) < tol
                if condition_1 or condition_2:
                    break
            dems[i, j, :] = np.exp(s)

    return dems, chi2


def sparse_em_init(trlogt_list, tresp_list, bases_sigmas=None, 
                   differential=False, bases_powers=[], normalize=False, use_lgtaxis=None):

    if bases_sigmas is None:
        bases_sigmas = np.array([0.0, 0.1, 0.2, 0.6])
    if use_lgtaxis is None:
        lgtaxis = trlogt_list[0]
    else:
        lgtaxis = use_lgtaxis
    nchannels = len(tresp_list)
    ntemp = lgtaxis.size
    dlgt = lgtaxis[1]-lgtaxis[0]

    nsigmas = len(bases_sigmas)
    npowers = len(bases_powers)

    nbases = (nsigmas+npowers)*ntemp
    basis_funcs = np.zeros([nbases, ntemp])
    Dict = np.zeros([nchannels, nbases])

    tresps = np.zeros([nchannels, ntemp])
    for i in range(0, nchannels): 
        tresps[i, :] = np.interp(lgtaxis, trlogt_list[i], tresp_list[i])    

    for s in range(0, nsigmas):
        if(bases_sigmas[s] == 0):
            for m in range(0, ntemp):
                basis_funcs[ntemp*s+m, m] = 1
        else:
            extended_lgtaxis = (np.arange(50)-25) * dlgt + 6.0
            line = np.exp(-((extended_lgtaxis-6.0)/bases_sigmas[s])**2.0)
            cut = line < 0.04
            line[cut] = 0.0
            norm = np.sum(line)
            for m in range(0, ntemp):
                line = np.exp(-((lgtaxis-lgtaxis[m])/bases_sigmas[s])**2.0)
                cut = line < 0.04
                line[cut] = 0.0
                if(normalize):
                    line = line/norm
                basis_funcs[ntemp*s+m, 0:ntemp] = line

    for s in range(0, npowers):
        for m in range(0, ntemp):
            if(bases_powers[s] < 0):
                basis_funcs[ntemp*(s+nsigmas)+m, m:ntemp] = np.exp(
                    (lgtaxis[m:ntemp]-lgtaxis[m])/bases_powers[s])
            if(bases_powers[s] > 0):
                basis_funcs[ntemp*(s+nsigmas)+m, 0:m+1] = np.exp(
                    (lgtaxis[0:m+1]-lgtaxis[m])/bases_powers[s])

    if(differential):
        for i in range(0, nchannels):
            for j in range(0, nbases):
                Dict[i, j] = np.trapz(tresps[i, :]*basis_funcs[j, :], lgtaxis)
    else:
        Dict = np.matmul(tresps, basis_funcs.T)

    return Dict, lgtaxis, basis_funcs, bases_sigmas


def simplex(zequation, constraint, m1, m2, m3, eps=None):
    from scipy.optimize import linprog

    b_ub = np.hstack([constraint[0, 0:m1], -constraint[0, m1:m1+m2]])
    A_ub = np.hstack([-constraint[1:, 0:m1], constraint[1:, m1:m1+m2]]).T
    b_eq = constraint[0, m1+m2:m1+m2+m3]
    A_eq = constraint[1:, m1+m2:m1+m2+m3].T

    result = linprog(-zequation,
                     A_ub=A_ub,
                     b_ub=b_ub,
                     A_eq=A_eq,
                     b_eq=b_eq,
                     options={'tol': eps, 'autoscale': True},
                     method='simplex')
    return np.hstack([result['fun'], result['x']]), result['status']


def sparse_em_solve(image, errors, exptimes, Dict,
                    zfac=[],
                    eps=1.0e-3,
                    tolfac=1.4,
                    relax=True,
                    symmbuff=1.0,
                    adaptive_tolfac=True,
                    epsfac=1.0e-10):
    dim = image.shape
    nocounts = np.where(np.sum(image, axis=2) < 10*eps)

    zmax = np.zeros([dim[0], dim[1]])
    status = np.zeros([dim[0], dim[1]])
    [nchannels, ntemp] = Dict.shape
    if(len(zfac) == ntemp):
        zequation = -zfac
    else:
        zequation = np.zeros(ntemp) - 1.0

    if(adaptive_tolfac and np.isscalar(tolfac)):
        tolfac = tolfac*np.array([1.0, 1.5, 2.25])
    if(np.isscalar(tolfac)):
        tolfac = [tolfac]
    ntol = len(tolfac)

    if(relax):
        m1 = nchannels
        m2 = nchannels
        m3 = 0
        m = m1 + m2 + m3
    constraint = np.zeros([m, ntemp+1])
    constraint[0:m1, 1:ntemp+1] = -Dict
    constraint[m1:m1+m2, 1:ntemp+1] = -Dict
    coeffs = np.zeros([image.shape[0], image.shape[1], ntemp])
    tols = np.zeros([image.shape[0], image.shape[1]])
    for i in range(0, dim[0]):
        for j in range(0, dim[1]):
            y = np.clip(image[i, j, :], 0, None)/exptimes
            for k in range(0, ntol):
                if(relax):
                    tol = tolfac[k]*errors[i, j, :]/exptimes
                    constraint[0:m1, 0] = y + tol
                    constraint[m1:m1+m2, 0] = np.clip((y-symmbuff*tol), 0.0, None)
                    [r, s] = simplex(zequation, constraint.T, m1, m2, m3,
                                     eps=eps*np.max(y)*epsfac)
                else:
                    constraint = np.hstack([y, -Dict])
                    [r, s] = simplex(zequation, constraint.T, 0, 0, nchannels)
                if s == 0:
                    break
            if np.min(r[1:ntemp+1]) < 0.0:
                coeffs[i, j, 0:ntemp] = 0.0
                s = 10
            else:
                coeffs[i, j, 0:ntemp] = r[1:ntemp+1]
            zmax[i, j] = r[0]
            status[i, j] = s
            tols[i, j] = k
    if(len(nocounts[0]) > 0):
        status[nocounts] = 11.0

    return coeffs, zmax, status
