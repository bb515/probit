"""TODO: unfinished"""
import lab as B
import numpy as np


def objective_PEP(
        N, alpha, minibatch_size,
        posterior_mean, posterior_cov, log_lik, Kuu):
    """
    Calculate fx, the variational lower bound of the log marginal
    likelihood at the EP equilibrium.

    .. math::
            \mathcal{F(\theta)} =,

        where :math:`F(\theta)` is the PEP approximation of the log
        marginal likelihood at the PEP equilibrium,
        :math:`K`. #TODO

    :arg weight: 
    :arg precision:
    :arg L_cov:
    :arg Z:

    :returns: fx
    :rtype: float
    """
    scale_logZ_tilted = N * 1.0 / minibatch_size / alpha
    log_lik = log_lik * scale_logZ_tilted
    phi_pos = _compute_phi_mvg(posterior_mean, posterior_cov)
    phi_prior = _compute_phi_mvg(
        B.zeros(posterior_mean.shape), Kuu)
    scale_post = -N * 1.0 / alpha + 1.0
    log_lik += scale_post * phi_pos - phi_prior
    return log_lik


def objective_gradient_PEP(gx, trainables, J, D, ARD):
    """
    Calculate gx, the jacobian of the variational lower bound of the
    log marginal likelihood at the EP equilibrium.

    .. math::
            \mathcal{\frac{\partial F(\theta)}{\partial \theta}}

        where :math:`F(\theta)` is the approximate log marginal likelihood
        at the EP equilibrium,
    """
    if trainables is not None:
        # Update gx
        if trainables[0]:
            gx[0] = 0
        # For gx[1] -- \b_1
        if trainables[1]:
            gx[1] = 0
        # For gx[2] -- ln\Delta^r
        for j in range(2, J):
            if trainables[j]:
                gx[j] = 0
        # For gx[J] -- variance
        if trainables[J]:
            # For gx[J] -- s
            gx[J] = 0
        # For gx[J + 1] -- theta
        if ARD:
            for d in range(D):
                if trainables[J + 1][d]:
                    gx[J + 1 + d] = 0
        else:
            if trainables[J + 1]:
                gx[J + 1] = 0
    return -gx


def compute_logZ_parallel( beta_si, gamma_si, betacavKuu,
        V, Vinv, m, var_new_parallel, mean_new_parallel, p_i, k_i, m_si_i,
        v_si_ii, logZtilted, Xbatch, ybatch, dlogZ_dmi, dlogZ_dvi, Kuu):
        # compute cavity covariance
    betacavKuu = B.einsum('abc,cd->abd', beta_si, Kuu)
    mcav = B.einsum('bc,acd->abd', Kuu, gamma_si)
    Vcav = Kuu - B.einsum('bc,acd->abd', Kuu, betacavKuu)

    signV, logdetV = B.linalg.slogdet(V)
    signKuu, logdetKuu = B.linalg.slogdet(Kuu)
    Vinvm = B.dot(Vinv, m)
    term1 = 0.5 * (logdetV - logdetKuu + B.dot(m, Vinvm))

    tn = 1.0 / var_new_parallel
    gn = mean_new_parallel
    wnVcav = B.einsum('abc,abd->adc', p_i, Vcav)
    wnVcavwn = B.einsum('abc,abd->ac', wnVcav, p_i)
    wnVcavVinvm = B.sum(wnVcav * Vinvm[:, newaxis], axis=1)
    wnV = B.einsum('abc,bd->adc', p_i, V)
    wnVwn = B.sum(wnV * p_i, axis=1)
    mwn = B.einsum('b,abc->ac', m, p_i)
    oneminuswnVwn = 1 - alpha * tn * wnVwn

    term2a = 0.5 * alpha * tn**2 * gn**2 * wnVcavwn
    term2b = - gn * tn * wnVcavVinvm
    term2c = 0.5 * tn * mwn**2 / oneminuswnVwn
    term2d = -0.5 / alpha * B.log(oneminuswnVwn)
    term2 = N / minibatch_size * B.sum(
        term2a + term2b + term2c + term2d)

    scale_logZtilted = N / minibatch_size / alpha
    term3 = scale_logZtilted * B.sum(logZtilted)

    log_lik = term1 + term2 + term3

    KuuinvMcav = B.einsum('bc,acd->abd', Kuuinv, mcav)
    dlogZt_dmiKuuinvMcav = dlogZ_dmi[:, newaxis, :] * KuuinvMcav
    dlogZt_dKuu_via_mi = -B.einsum('abc,adc->abd', dlogZt_dmiKuuinvMcav, p_i)
    
    VcavKuuinvKufi = B.einsum('abc,acd->abd', Vcav, p_i)
    KuuinvVcavKuuinvKufi = B.einsum('bc,acd->abd', Kuuinv, VcavKuuinvKufi)
    p_idlogZ_dvi = p_i * dlogZ_dvi[:, newaxis, :]
    temp1 = - B.einsum('abc,adc->abd', KuuinvVcavKuuinvKufi, p_idlogZ_dvi)
    temp2 = B.transpose(temp1, [0, 2, 1])
    temp3 = B.einsum('abc,adc->abd', p_i, p_idlogZ_dvi)
    dlogZt_dKuu_via_vi = temp1 + temp2 + temp3
    dlogZt_dKuu = B.sum(dlogZt_dKuu_via_mi + dlogZt_dKuu_via_vi, axis=0)

    dlogZt_dKfu_via_mi = dlogZt_dmiKuuinvMcav
    dlogZt_dKfu_via_vi = 2 * dlogZ_dvi[:, newaxis, :] * (-p_i + KuuinvVcavKuuinvKufi)
    dlogZt_dKfu = dlogZt_dKfu_via_mi + dlogZt_dKfu_via_vi
    dlogZt_dsf = (2*B.sum(dlogZt_dKfu * k_i) 
        + 2*B.sum(dlogZ_dvi*B.exp(2*sf)))
    ls2 = B.exp(2*ls)
    ones_M = B.ones((minibatch_size, M))
    ones_D = B.ones((minibatch_size, D))
    xi_minus_zu = B.einsum('km,kd->kmd', ones_M, Xbatch) - zu
    
    temp1 = B.einsum('kma,kd->kmd', k_i, ones_D) * 0.5 * xi_minus_zu**2
    dlogZt_dls = 2.0*B.sum(dlogZt_dKfu * temp1) / ls2
    temp2 = xi_minus_zu * B.einsum('km,d->kmd', ones_M, 1.0 / ls2 )
    dlogZt_dzu = B.sum(B.einsum('kma,kd->kmd', dlogZt_dKfu * k_i, ones_D) * temp2, axis=0)

    dlogZt_dsn = 0
    for i in range(m_si_i.shape[0]):
        dlogZt_dsn += dlogZtilted_dsn(ybatch[i], m_si_i[i], 
            v_si_ii[i], alpha)

    log_lik = log_lik

    # compute the gradients
    Vmm = V + B.outer(m, m)
    S = - Kuuinv + B.dot(
        Kuuinv, B.dot(Vmm, Kuuinv))
    S = S + 2*scale_logZtilted * dlogZt_dKuu
    dhyp = d_trace_MKzz_dhypers(2*ls, 2*sf, zu, S, Kuu)
    grads = {
        'ls': dhyp[1] + scale_logZtilted * dlogZt_dls,
        'sf': dhyp[0] + scale_logZtilted * dlogZt_dsf,
        'sn': scale_logZtilted * dlogZt_dsn, 
        'zu': dhyp[2]/2 + scale_logZtilted * dlogZt_dzu} 
    return log_lik, grads


def update_posterior_sequential_PEP(
        self, indices, beta=None, gamma=None,
        mean_EP=None, variance_EP=None, write=False):
    """
    TODO: rename this update_posterior_sequential_PEP
    Estimating the posterior means and posterior covariance (and marginal
    likelihood) via Expectation propagation iteration as written in
    Appendix B Chu, Wei & Ghahramani, Zoubin. (2005). Gaussian Processes
    for Ordinal Regression.. Journal of Machine Learning Research. 6.
    1019-1041.

    EP does not attempt to learn a posterior distribution over
    hyperparameters, but instead tries to approximate
    the joint posterior given some hyperparameters. The hyperparameters
    have to be optimized with model selection step.

    :arg indices: The set of indices of the data in this swipe.
        Could be e.g., a minibatch, the whole dataset.
    :type indices: :class:`numpy.ndarray`
    :arg posterior_mean_0: The initial state of the approximate posterior
        mean (N,). If `None` then initialised to zeros, default `None`.
    :type posterior_mean_0: :class:`numpy.ndarray`
    :arg posterior_cov_0: The initial state of the posterior covariance
        (N, N). If `None` then initialised to prior covariance,
        default `None`.
    :type posterior_cov_0: :class:`numpy.ndarray`
    :arg mean_EP_0: The initial state of the individual (site) mean (N,).
        If `None` then initialised to zeros, default `None`.
    :type mean_EP_0: :class:`numpy.ndarray`
    :arg precision_EP_0: The initial state of the individual (site)
        variance (N,). If `None` then initialised to zeros, default `None`.
    :type precision_EP_0: :class:`numpy.ndarray`
    :arg amplitude_EP_0: The initial state of the individual (site)
        amplitudes (N,). If `None` then initialised to ones, default
        `None`.
    :type amplitude_EP_0: :class:`numpy.ndarray`
    :arg bool fix_hyperparameters: Must be `True`, since the hyperparameter
        approximate posteriors are of the hyperparameters are not
        calculated in this EP approximation.
    :arg bool write: Boolean variable to compute and store the gradient
        of the tilted distribution with respect to the hyperparameters
        and the log normalizer of interest, logZ. If set to "True", the
        method will output non-empty logZtilted and dlogZ_cavity_mean_n.

        containers of evolution of the statistics over the steps.
        If set to "False", statistics will not be written and those
        containers will remain empty.
    :return: approximate posterior mean and covariances.
    :rtype: (8, ) tuple of :class:`numpy.ndarrays` of the approximate
        posterior means, other statistics and tuple of lists of per-step
        evolution of those statistics.
    """
    for index in indices:
        x_i = self.X_train[index]
        y_i = 2 * self.y_train[index] - 1
        p_i = self.KuuinvKuf[:, index]
        k_i = self.Kfu[index, :]
        K_ii = self.Kfdiag[index]
        variance_EP_n_old = variance_EP[index]
        mean_EP_n_old = mean_EP[index]
        (beta_si, gamma_si)= self._delete(
            p_i, k_i, self.alpha, beta, gamma,
            mean_EP_n_old, variance_EP_n_old)
        (h, m_si_i, v_si_ii, dlogZ_dmi, dlogZ_dmi2, beta_new,
                gamma_new) = self._project(
            y_i, p_i, k_i, self.alpha, beta_si, gamma_si, K_ii)
        mean_EP_n, variance_EP_n = self._include(
            h, m_si_i, dlogZ_dmi, dlogZ_dmi2, p_i, k_i, self.alpha,
            mean_EP_n_old, variance_EP_n_old)
        #diff = variance_EP_n - variance_EP_n_old
        if variance_EP_n > 0.0:
            # Update EP parameters
            error += (mean_EP_n - mean_EP_n_old)**2
            variance_EP[index] = variance_EP_n
            mean_EP[index] = mean_EP_n
            beta = beta_new
            gamma = gamma_new
            if write:
                logZtilted, dlogZtilted, phi_cav = self._compute_logZ(
                    p_i, self.alpha, beta_si, gamma_si,
                    x_i, y_i, k_i, m_si_i, v_si_ii, dlogZ_dmi)
                log_lik += logZtilted + phi_cav
                grads_logZtilted += dlogZtilted
        else:
            pass
            ############### Update them anyway. There is an issue with stability here possibly.
            # Update EP parameters
            # error += (diff**2
            #           + (mean_EP_n - mean_EP_n_old)**2)
            # variance_EP[index] = variance_EP_n
            # mean_EP[index] = mean_EP_n
            # beta = beta_new
            # gamma = gamma_new
            ###############
            # if variance_EP_n < 0.0:
            #     print(
            #         "Skip {}, v_new={}, v_old={}.\n".format(
            #         index, variance_EP_n, variance_EP_n_old))
    return (error, beta, gamma, mean_EP, variance_EP,
            log_lik, grads_logZtilted)


def update_posterior_parallel_PEP(
        indices, beta=None, gamma=None, mean_EP=None, variance_EP=None,
        write=False):
    """Approximate with parallel PEP.
    TODO: rename this update_posterior_parallel_PEP"""

    ## minibatch
    # Kfu = Kfu[indices, :]
    # KuuinvKuf = KuuinvKuf[:, indices]
    # Kff_diag = Kfdiag[indices]

    # Full dataset
    Kfu = Kfu
    KuuinvKuf = KuuinvKuf
    Kff_diag = Kfdiag

    # perform parallel updates
    # deletion
    p_i = KuuinvKuf[:, :, B.newaxis].transpose((1, 0, 2))
    k_i = Kfu[:, :, B.newaxis] 
    k_ii = Kff_diag[:, B.newaxis]

    gamma_new_axis = gamma[:, B.newaxis]
    h_si = p_i - B.einsum('ab,kbc->kac', beta, k_i)
    variance_i_ori = variance_EP[indices, :]
    variance_i = variance_i_ori[:, :, B.newaxis]
    mean_i_ori = mean_EP[indices, :]
    mean_i = mean_i_ori[:, :, B.newaxis]
    dlogZd_dmi2 = 1.0 / (variance_i / alpha - 
        B.sum(k_i * h_si, axis=1, keepdims=True))
    dlogZd_dmi = -dlogZd_dmi2 * (mean_i - 
        B.sum(k_i * gamma_new_axis, axis=1, keepdims=True))
    hd1 = h_si * dlogZd_dmi
    hd2h = B.einsum('abc,adc->abd', h_si, h_si) * dlogZd_dmi2
    gamma_si = gamma_new_axis + hd1
    beta_si = beta - hd2h

    # projection
    h = p_i - B.einsum('abc,acd->abd', beta_si, k_i)
    m_si_i = B.einsum('abc,abc->ac', k_i, gamma_si)
    v_si_ii = k_ii - B.einsum('abc,abd,adc->ac', k_i, beta_si, k_i)

    # ## Ordinal likelihood
    # cutpoints_tplus1s = cutpoints_tplus1s[indices, B.newaxis]
    # cutpoints_ts = cutpoints_ts[indices, B.newaxis]
    # dlogZ_dmi = ordinal_dlogZtilted_dm_vector(
    #     cutpoints_tplus1s, cutpoints_ts, noise_std,
    #     m_si_i, v_si_ii,
    #     alpha, gauss_hermite_points)
    # dlogZ_dmi2 = ordinal_dlogZtilted_dm2_vector(
    #     cutpoints_tplus1s, cutpoints_ts, noise_std,
    #     m_si_i, v_si_ii,
    #     alpha, gauss_hermite_points)
    # dlogZ_dmi = dlogZ_dmi.reshape(-1, 1)
    # dlogZ_dmi2 = dlogZ_dmi2.reshape(-1, 1)

    # Bernoulli likelihood
    ybatch = (2 * y_train[indices, B.newaxis]) - 1
    dlogZ_dmi = probit_dlogZtilted_dm_vector(ybatch, m_si_i, v_si_ii,
        alpha, gauss_hermite_points, noise_variance)
    dlogZ_dmi2 = probit_dlogZtilted_dm2_vector(ybatch, m_si_i, v_si_ii,
        alpha, gauss_hermite_points, noise_variance)
    dlogZ_dmi = dlogZ_dmi.reshape(-1, 1)
    dlogZ_dmi2 = dlogZ_dmi2.reshape(-1, 1)

    # # TODO: SS
    # ybatch = (2 * y_train[indices, B.newaxis]) - 1
    # dlogZ_dmi = B.zeros(m_si_i.shape)
    # dlogZ_dmi2 = B.zeros(m_si_i.shape)
    # #dlogZ_dvi = B.zeros(m_si_i.shape)
    # #logZtilted = B.zeros(m_si_i.shape)

    # for i in range(len(indices)):
    #     m_ii = m_si_i[i, 0]
    #     v_ii = v_si_ii[i, 0]
    #     y_ii = ybatch[i]

    #     # logZtilted[i] = probit_logZtilted(
    #     #     y_ii, m_ii, v_ii, alpha, gauss_hermite_points,
    #     #     noise_std)
    #     dlogZ_dmi[i] = probit_dlogZtilted_dm(
    #         y_ii, m_ii, v_ii, alpha, gauss_hermite_points,
    #         noise_std)
    #     dlogZ_dmi2[i] = probit_dlogZtilted_dm2(
    #         y_ii, m_ii, v_ii, alpha, gauss_hermite_points,
    #         noise_std)
    #     # dlogZ_dvi[i] = probit_dlogZtilted_dv(
    #     #     y_ii, m_ii, v_ii, alpha, gauss_hermite_points)

    var_i_new = -1.0 / dlogZ_dmi2 - B.sum(k_i * h, axis=1)
    mean_i_new = m_si_i - dlogZ_dmi / dlogZ_dmi2

    var_new_parallel = 1 / (1 / var_i_new + 1 / variance_i_ori * (
        1 - alpha))
    mean_div_var_i_new = (mean_i_new / var_i_new + 
        mean_i_ori / variance_i_ori * (1 - alpha))

    mean_new_parallel = mean_div_var_i_new * var_new_parallel

    rho = 0.5 # damped - more numerically stable in some circumstances
    # rho = 1.0  # undamped - probably best. What about underdamped tho?
    
    n1_new = 1.0 / var_new_parallel
    n2_new = mean_new_parallel / var_new_parallel

    n1_ori = 1.0 / variance_i_ori
    n2_ori = mean_i_ori / variance_i_ori

    n1_damped = rho * n1_new + (1.0 - rho) * n1_ori
    n2_damped = rho * n2_new + (1.0 - rho) * n2_ori

    var_new_parallel = 1.0 / n1_damped
    mean_new_parallel = var_new_parallel * n2_damped 

    if B.any(var_new_parallel < 0):
        # skip this update
        error = B.inf
        print("SKIP")
    else:
        # update means and variances
        mean_EP[indices, :] = mean_new_parallel
        variance_EP[indices, :] = var_new_parallel
        # update gamma and beta
        # TODO: not sure why it requires a matrix inverse at every step
        # TODO: why necessary to calculate m, V here?
        (gamma, beta, *_) = _update_posterior(mean_EP, variance_EP)
        diff_mean = mean_new_parallel[:, 0] - mean_i_ori[:, 0]
        error = (diff_mean.T @ diff_mean)**2
        if write:
            # log_lik, grads = _compute_logZ_parallel()
            grads = 0
    # except (RuntimeWarning, B.linalg.linalg.LinAlgError):
    #         print("exception: ignore this update")
    #         mean_new_parallel = mean_i_ori
    #         var_new_parallel = variance_i_ori
    #         error = True
    # need to return m, V as well. something confusing about this.
    return (error, beta, gamma, mean_EP, variance_EP, log_lik, grads)


def _update_posterior( mean_EP, variance_EP):
    means = mean_EP[:, 0]
    variances = variance_EP[:, 0]

    T2u = B.diag(1./ variances)

    # stds = B.sqrt(variances)
    # Lambda_halfinv = B.diag(1./stds)
    # Lambda_half = B.diag(stds)
    # Lambda = B.diag(variances)
    # (L, lower) = cho_factor(Lambda_halfinv @ Kuu @ Lambda_halfinv +
    #     B.eye(M))
    # LTinv = solve_triangular(L.T, B.eye(M), lower=True)
    # Ainv = solve_triangular(L, LTinv, lower=False)
    # V = Lambda - Lambda_half @ Ainv @ Lambda_half
    # cov = Lambda_halfinv @ Ainv @ Lambda_halfinv
    # T1u = means / variances
    # m = V @ T1u
    # gamma = cov @ means
    # beta = cov

    Lambda = B.diag(variances)
    (L, lower) = cho_factor(
        Kuu + Lambda)
    half_log_det_cov = -2 * B.sum(B.log(B.diag(L)))
    LTinv = solve_triangular(L.T, B.eye(M), lower=True)
    cov = solve_triangular(L, LTinv, lower=False)
    gamma = cov @ means
    beta = cov
    # V = Lambda - Lambda @ cov @ Lambda
    # T1u = means / variances
    # m = V @ T1u

    # V = matrixInverse(Vinv)
    #half_log_det_V = 0
    # # half_log_det_V = B.sum(B.log(B.diag(L_Vinv)))  # TODO: check this is the case
    # Vinv = Kuuinv + T2u
    # T2u = (KuuinvKuf / variances) @ KuuinvKuf.T
    # T1u = KuuinvKuf @ (means / variances)
    # m = V @ T1u
    # gamma = Kuuinv @ m
    # beta = Kuuinv @ (Kuu - V) @ Kuuinv
    return gamma, beta


def _update_pep_variables( Kuuinv, posterior_mean, posterior_cov):
    """TODO: collapse"""
    return Kuuinv @ posterior_mean, Kuuinv @ (Kuuinv - posterior_cov)


def _compute_posterior( Kuu, gamma, beta):
    """TODO: collapse"""
    return Kuu @ gamma, Kuu - Kuu @ (beta @ Kuu)


def _compute_phi_mvg( m, V):
    """TODO: does this need a numerically stable version."""
    (L_V, lower) = cho_factor(V + epsilon * B.eye(B.shape(V)[0]))
    half_log_det_V = - B.sum(B.log(B.diag(L_V)))
    L_VT_inv = solve_triangular(
        L_V.T, B.eye(M), lower=True)
    V_inv = solve_triangular(L_V, L_VT_inv, lower=False)
    return half_log_det_V + 0.5 * m.T @ V_inv @ m + 0.5 * M * B.log(2 * B.pi)


def _delete(
         p_i, k_i, alpha, beta, gamma, mean_i, variance_i):
    # Note h_si for the deletion uses the non \i version of beta
    h_si = p_i - B.dot(beta, k_i)
    dlogZd_dmi2 = 1.0 / (variance_i / alpha - B.dot(k_i, h_si))
    dlogZd_dmi = -dlogZd_dmi2 * (mean_i - B.dot(k_i, gamma))
    gamma_si = gamma + h_si * dlogZd_dmi
    beta_si = beta - B.outer(h_si, h_si)*dlogZd_dmi2
    return beta_si, gamma_si  # , p_i, k_i, h_si


def _project(
         y_i, p_i, k_i, alpha, beta_si, gamma_si, Kff_ii):
    h = p_i - B.dot(beta_si, k_i)
    m_si_i = B.dot(k_i, gamma_si)
    v_si_ii = Kff_ii - B.dot(B.dot(k_i, beta_si), k_i)
    dlogZ_dmi = probit_dlogZtilted_dm(
        y_i, m_si_i, v_si_ii, alpha, gauss_hermite_points,
        noise_variance)
    dlogZ_dmi2 = probit_dlogZtilted_dm2(
        y_i, m_si_i, v_si_ii, alpha, gauss_hermite_points,
        noise_variance)
    gamma_new = gamma_si + h * dlogZ_dmi
    beta_new = beta_si - B.outer(h, h) * dlogZ_dmi2
    return  h, m_si_i, v_si_ii, dlogZ_dmi, dlogZ_dmi2, beta_new, gamma_new


def _include(
         h, m_si_i, dlogZ_dmi, dlogZ_dmi2, p_i, k_i, alpha,
        mean_i, variance_i):
    var_i_new = - 1.0 / dlogZ_dmi2 - B.dot(k_i, h)
    mean_i_new = m_si_i - dlogZ_dmi / dlogZ_dmi2
    var_new = 1 / (1 / var_i_new + 1 / variance_i * (1 - alpha))
    mean_div_var_i_new = (mean_i_new / var_i_new + 
            mean_i / variance_i * (1 - alpha))
    mean_new = mean_div_var_i_new * var_new
    return mean_new, var_new


def _compute_logZ(
         p_i, alpha, beta_si, gamma_si, x_i, y_i, k_i, m_si_i,
        v_si_ii, dlogZ_dmi):
    (m_cav, V_cav) = _compute_posterior(Kuu, gamma_si, beta_si)
    phi_cav = _compute_phi_mvg(m_cav, V_cav)
    logZtilted = probit_logZtilted(
        y_i, m_si_i, v_si_ii, alpha, gauss_hermite_points)
    dlogZ_dvi = probit_dlogZtilted_dv(
        y_i, m_si_i, v_si_ii, alpha, gauss_hermite_points)
    KuuinvMcav = Kuuinv @ m_cav
    dlogZtilted_dKuu_via_mi = -B.outer(dlogZ_dmi * KuuinvMcav, p_i)
    KuuinvVcavKuuinvKufi = Kuuinv @ V_cav @ p_i
    temp1 = -B.outer(KuuinvVcavKuuinvKufi, p_i*dlogZ_dvi)
    temp2 = temp1.T
    temp3 = B.outer(p_i, p_i*dlogZ_dvi)
    dlogZtilted_dKuu_via_vi = temp1 + temp2 + temp3
    dlogZtilted_dKuu = dlogZtilted_dKuu_via_mi + dlogZtilted_dKuu_via_vi
    dlogZtilted_dKfu_via_mi = dlogZ_dmi * KuuinvMcav
    dlogZtilted_dKfu_via_vi = 2 * dlogZ_dvi * (-p_i + KuuinvVcavKuuinvKufi)
    dlogZtilted_dKfu = dlogZtilted_dKfu_via_mi + dlogZtilted_dKfu_via_vi

    # TODO: sf and ls will exist in kernel or more likely theta. This implementation assumes ARD.
    sf = B.log(variance) / 2
    ls = B.log(theta) / 2

    dlogZtilted_dsf = (2*B.sum(dlogZtilted_dKfu * k_i) 
            + 2*dlogZ_dvi*B.exp(2*sf))
    ls2 = B.exp(2*ls)  # TODO BB if ls is log ell then this is exp(ell^2)
    ones_M = B.ones((M, ))
    ones_D = B.ones((D, ))
    # TODO: I think this may be a kernel calculation.
    xi_minus_zu = B.outer(ones_M, x_i) - Z
    temp1 = B.outer(k_i, ones_D) * 0.5 * xi_minus_zu**2
    dlogZtilted_dls = 2*B.dot(dlogZtilted_dKfu, temp1) * 1.0 / ls2
    temp2 = xi_minus_zu * B.outer(ones_M, 1.0 / ls2 )
    dlogZtilted_dzu = B.outer(dlogZtilted_dKfu * k_i, ones_D) * temp2
    dlogZtilted_dsn = probit_dlogZtilted_dsn(
        y_i, m_si_i, v_si_ii, alpha, gauss_hermite_points)
    dlogZtilted = 0
    # dlogZtilted = {
    #         'ls': dlogZtilted_dls, 
    #         'sf': dlogZtilted_dsf,
    #         'sn': dlogZtilted_dsn, 
    #         'zu': dlogZtilted_dzu, 
    #         'Kuu': dlogZtilted_dKuu}
    return logZtilted, dlogZtilted, phi_cav


def d_trace_MKzz_dhypers(lls, lsf, z, M, Kzz):

    dKzz_dlsf = Kzz
    ls = B.exp(lls)

    # This is extracted from the R-code of Scalable EP for GP Classification by DHL and JMHL

    gr_lsf = B.sum(M * dKzz_dlsf)

    # This uses the vact that the distance is v^21^T - vv^T + 1v^2^T, where v is a vector with the l-dimension
    # of the inducing points. 

    Ml = 0.5 * M * Kzz
    Xl = z * B.outer(B.ones(z.shape[ 0 ]), 1.0 / B.sqrt(ls))
    gr_lls = B.dot(B.ones(Ml.shape[ 0 ]), B.dot(Ml.T, Xl**2)) + B.dot(B.ones(Ml.shape[ 0 ]), B.dot(Ml, Xl**2)) \
    - 2.0 * B.dot(B.ones(Xl.shape[ 0 ]), (Xl * B.dot(Ml, Xl)))

    Xbar = z * B.outer(B.ones(z.shape[ 0 ]), 1.0 / ls)
    Mbar1 = - M.T * Kzz
    Mbar2 = - M * Kzz
    gr_z = (Xbar * B.outer(B.dot(B.ones(Mbar1.shape[ 0 ]) , Mbar1), B.ones(Xbar.shape[ 1 ])) - B.dot(Mbar1, Xbar)) +\
        (Xbar * B.outer(B.dot(B.ones(Mbar2.shape[ 0 ]) , Mbar2), B.ones(Xbar.shape[ 1 ])) - B.dot(Mbar2, Xbar))

    # The cost of this function is dominated by five matrix multiplications with cost M^2 * D each where D is 
    # the dimensionality of the data!!!

    return gr_lsf, gr_lls, gr_z


def ordinal_logZtilted_vector(
        cutpoints_yplus1, cutpoints_y, noise_std, m, v, alpha, deg):
    gh_x, gh_w = B.polynomial.hermite.hermgauss(deg)
    gh_x = gh_x.reshape(1, -1)
    ts = gh_x * B.sqrt(2*v) + m
    pdfs = ndtr(
        (cutpoints_yplus1 - ts) / noise_std) - ndtr(
            (cutpoints_y - ts) / noise_std)
    r = B.log(pdfs**alpha @ gh_w / B.sqrt(B.pi))
    return r


def ordinal_dlogZtilted_dm_vector(
        cutpoints_yplus1, cutpoints_y, noise_std, m, v, alpha, deg):
    gh_x, gh_w = B.polynomial.hermite.hermgauss(deg) 
    gh_x = gh_x.reshape(1, -1)
    eps = 1e-8
    ts = gh_x * B.sqrt(2*v) + m
    uppers = (cutpoints_yplus1 - ts) / noise_std
    lowers = (cutpoints_y - ts) / noise_std
    pdfs = ndtr(uppers) - ndtr(lowers) + eps
    # TODO: simpler way?
    Ztilted = pdfs**alpha @ gh_w / B.sqrt(B.pi)
    dZdm = pdfs**(alpha-1.0) * (-B.exp(-uppers**2/2) + B.exp(-lowers**2/2)) @ gh_w * alpha / B.pi / B.sqrt(2) / noise_std  # TODO: should it be sqrt(2) * noise_std
    return dZdm / Ztilted


def ordinal_dlogZtilted_dm2_vector(
        cutpoints_yplus1, cutpoints_y, noise_std, m, v, alpha, deg):
    gh_x, gh_w = np.polynomial.hermite.hermgauss(deg)
    gh_x = gh_x.reshape(1, -1)
    eps = 1e-8
    ts = gh_x * B.sqrt(2*v) + m
    uppers = (cutpoints_yplus1 - ts) / noise_std
    lowers = (cutpoints_y - ts) / noise_std
    pdfs = ndtr(uppers) - ndtr(lowers) + eps
    Ztilted = pdfs**alpha @ gh_w / B.sqrt(B.pi)
    dZdv = pdfs**(alpha-1.0) * (-B.exp(-uppers**2/2) + B.exp(-lowers**2/2)) * gh_x / B.sqrt(2*v) @ gh_w * alpha / B.pi / B.sqrt(2) / noise_std
    return dZdv / Ztilted


def probit_logZtilted_vector(y, m, v, alpha, deg, noise_variance):
    if alpha == 1.0:
        t = y * m / B.sqrt(noise_variance + v)
        Z = 0.5 * (1 + B.erf(t / B.sqrt(2)))
        eps = 1e-16
        return B.log(Z + eps)
    else:
        gh_x, gh_w = np.polynomial.hermite.hermgauss(deg)
        gh_x = gh_x.reshape(1, -1)
        ts = gh_x * B.sqrt(2*v) + m
        pdfs = 0.5 * (1 + B.erf(ts * y / B.sqrt(2 * noise_variance)))
        r = B.log(pdfs**alpha @ gh_w / B.sqrt(B.pi))
        return r


def probit_dlogZtilted_dm_vector(y, m, v, alpha, deg, noise_variance):
    if alpha == 1.0:
        t = y * m / B.sqrt(noise_variance + v)
        Z = 0.5 * (1 + B.erf(t / B.sqrt(2)))
        eps = 1e-16
        Zeps = Z + eps
        beta = 1 / Zeps / B.sqrt(noise_variance + v) * 1/B.sqrt(2*B.pi) * B.exp(-t**2.0 / 2)
        return y*beta
    else:
        gh_x, gh_w = np.polynomial.hermite.hermgauss(deg) 
        gh_x = gh_x.reshape(1, -1)
        eps = 1e-8
        ts = gh_x * B.sqrt(2*v) + m
        pdfs = 0.5 * (1 + B.erf(ts * y / B.sqrt(2 * noise_variance))) + eps
        Ztilted = pdfs**alpha @ gh_w / B.sqrt(B.pi)
        dZdm =  pdfs**(alpha-1.0) * B.exp(-ts**2/(2* noise_variance)) * y @ gh_w * alpha / B.pi / B.sqrt(2 * noise_variance)  # TODO: / sqrt noise_variance
        return dZdm / Ztilted


def probit_dlogZtilted_dm2_vector(y, m, v, alpha, deg, noise_variance):
    if alpha == 1.0:
        t = y * m / B.sqrt(noise_variance + v)
        Z = 0.5 * (1 + B.erf(t / B.sqrt(2)))
        eps = 1e-16
        Zeps = Z + eps
        return - 0.5 * y * m / Zeps / (noise_variance + v)**1.5 * 1/B.sqrt(2*B.pi) * B.exp(-t**2.0 / 2)
    else:
        gh_x, gh_w = B.polynomial.hermite.hermgauss(deg)
        gh_x = gh_x.reshape(1, -1)
        eps = 1e-8    
        ts = gh_x * B.sqrt(2*v) + m
        pdfs = 0.5 * (1 + B.erf(ts * y / B.sqrt(2 * noise_variance))) + eps
        Ztilted = pdfs**alpha @ gh_w / B.sqrt(B.pi)
        dZdv = pdfs**(alpha-1.0) * B.exp(-ts**2/(2 * noise_variance)) * gh_x * y / B.sqrt(2*v) @ gh_w * alpha / B.pi / B.sqrt(2 * noise_variance)  # TODO: / sqrt noise_variance
        return dZdv / Ztilted


def probit_dlogZtilted_dsn(y_i, m_si_i, v_si_ii, alpha, deg):
    return 0


def probit_logZtilted(y, m, v, alpha, deg, noise_variance):
    if alpha == 1.0:
        t = y * m / B.sqrt(1+v)
        Z = 0.5 * (1 + B.erf(t / B.sqrt(2 * noise_variance)))
        eps = 1e-16
        return B.log(Z + eps)
    else:
        gh_x, gh_w = np.polynomial.hermite.hermgauss(deg)
        ts = gh_x * B.sqrt(2*v) + m
        pdfs = 0.5 * (1 + B.erf(y*ts / B.sqrt(2 * noise_variance)))
        return B.log(B.dot(pdfs**alpha, gh_w) / B.sqrt(B.pi)) 


def probit_dlogZtilted_dm(y, m, v, alpha, deg, noise_variance):
    if alpha == 1.0:
        t = y * m / B.sqrt(noise_variance + v)
        Z = 0.5 * (1 + B.erf(t / B.sqrt(2)))
        eps = 1e-16
        Zeps = Z + eps
        beta = 1 / Zeps / B.sqrt(noise_variance + v) * 1/B.sqrt(2*B.pi) * B.exp(-t**2.0 / 2)
        return y*beta
    else:
        gh_x, gh_w = np.polynomial.hermite.hermgauss(deg) 
        eps = 1e-8
        ts = gh_x * B.sqrt(2*v) + m
        pdfs = 0.5 * (1 + B.erf(y*ts / B.sqrt(2 * noise_variance))) + eps
        Ztilted = B.dot(pdfs**alpha, gh_w) / B.sqrt(B.pi)
        dZdm = B.dot(gh_w, pdfs**(alpha-1.0)*B.exp(-ts**2/2)) * y * alpha / B.pi / B.sqrt(2 * noise_variance)
        return dZdm / Ztilted + eps


def probit_dlogZtilted_dm2(y, m, v, alpha, deg, noise_variance):
    if alpha == 1.0:
        t = y * m / B.sqrt(noise_variance + v)
        Z = 0.5 * (1 + B.erf(t / B.sqrt(2)))
        eps = 1e-16
        Zeps = Z + eps
        return - 0.5 * y * m / Zeps / (noise_variance + v)**1.5 * 1/B.sqrt(2*B.pi) * B.exp(-t**2.0 / 2)
    else:
        gh_x, gh_w = np.polynomial.hermite.hermgauss(deg)   
        eps = 1e-8    
        ts = gh_x * B.sqrt(2*v) + m
        pdfs = 0.5 * (1 + B.erf(y*ts / B.sqrt(2 * noise_variance))) + eps
        Ztilted = B.dot(pdfs**alpha, gh_w) / B.sqrt(B.pi)
        dZdv = B.dot(gh_w, pdfs**(alpha-1.0)*B.exp(-ts**2/2) * gh_x) * y * alpha / B.pi / B.sqrt(2 * noise_variance) / B.sqrt(2*v)
        return dZdv / Ztilted + eps


def probit_dlogZtilted_dv(y, m, v, alpha, deg, noise_variance):
    if alpha == 1.0:
        t = y * m / B.sqrt(noise_variance + v)
        Z = 0.5 * (1 + B.erf(t / B.sqrt(2)))
        eps = 1e-16
        Zeps = Z + eps
        beta = 1 / Zeps / B.sqrt(noise_variance + v) * 1/B.sqrt(2*B.pi) * B.exp(-t**2.0 / 2)
        return - (beta**2 + m*y_*beta/(1+v))
    else:
        gh_x, gh_w = np.polynomial.hermite.hermgauss(deg)
        eps = 1e-8
        ts = gh_x * B.sqrt(2*v) + m
        pdfs = 0.5 * (1 + B.erf(y*ts / B.sqrt(2 * noise_variance))) + eps
        Ztilted = B.dot(pdfs**alpha, gh_w) / B.sqrt(B.pi)
        dZdm = B.dot(gh_w, pdfs**(alpha-1)*B.exp(-ts**2/2)) * y * alpha / B.pi / B.sqrt(2 * noise_variance)
        dZdm2 = B.dot(gh_w, (alpha-1)*pdfs**(alpha-2)*B.exp(-ts**2)/B.sqrt(2*B.pi)  
            - pdfs**(alpha-1) * y * ts * B.exp(-ts**2/2) ) * alpha / B.pi / B.sqrt(2 * noise_variance)
        return -dZdm**2 / Ztilted**2 + dZdm2 / Ztilted + eps
