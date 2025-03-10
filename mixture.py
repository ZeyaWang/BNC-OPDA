import sys
from sklearn.mixture import BayesianGaussianMixture
from sklearn.manifold import TSNE
import numpy as np
import plotly.express as px
from sklearn.utils import check_random_state
import warnings
from sklearn.exceptions import ConvergenceWarning
from scipy.special import logsumexp
import math
from scipy.special import betaln, digamma, gammaln
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from scipy.stats import moment
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.cluster import KMeans, kmeans_plusplus
import kmeans


def map_label(map_src, map_tar, to_be_map):
    src_indexes = np.unique(map_src, return_index=True)[1] # original array
    src_map = [map_src[index] for index in sorted(src_indexes)]
    tar_indexes = np.unique(map_tar, return_index=True)[1] # processed array
    tar_map = [map_tar[index] for index in sorted(tar_indexes)]
    mapping = {tar_map[i]:src_map[i] for i in range(len(src_map))}
    max_k = np.max(to_be_map) + 1
    print(src_map, tar_map)
    for k, v in mapping.items():
        print(k,v)
        if k != v:
            if v in to_be_map:
                to_be_map[to_be_map == v] = max_k
                max_k += 1
            to_be_map[to_be_map == k] = v
    return to_be_map


def sign_merge(embedding, labels, keep_labels):
    '''
    call sign for merging target remaining clusters
    keep_labels: the labels to be kept from merging
    '''
    ulabels = np.unique(labels).tolist()
    uln_l, ulxtx_l, ulxx_l = [], [], []
    n_keep = []
    for ui, ul in enumerate(ulabels):
        ulx = embedding[labels == ul, :]  # Nk x p
        uln = np.sum(labels == ul)  # Nk
        ulxtx = np.matmul(ulx.T, ulx)  # p x p
        ulxx = np.sum(ulx, axis=0)  # p
        uln_l.append(uln)
        ulxtx_l.append(ulxtx)
        ulxx_l.append(ulxx)
        if ul in keep_labels:
            n_keep.append(ui)

    uxx = np.stack(ulxx_l, axis=0)  # kxp
    un = np.array(uln_l)  # k
    uxtx = np.stack(ulxtx_l, axis=0).T  # p x p x k

    print('{} to be kept in the list of {}'.format(keep_labels, n_keep))

    Rest = Gibbs_DPM_Gaussian_summary_input_merge(uxtx, uxx, un, n_keep)  # mcmc

    dp_member, dp_Gammas, dp_mus = Rest['member_est'], Rest['Prec'], Rest['mu']

    labels_new = np.copy(labels)
    for u, ul in enumerate(ulabels):
        labels_new[labels == ul] = int(dp_member[u])  # order the cluster value with index

    member = np.unique(np.array([int(m) for m in dp_member]))
    member.sort()

    for mbi, mb in enumerate(member.tolist()):
        labels[labels_new == mb] = mbi

    return labels, dp_Gammas, dp_mus

def np_standardize(x):
    """standardize a numpy array.
    Args:
        x is a nxp array
    """
    stdv = (moment(x, moment=2, axis=0)) ** 0.5
    meanv = np.mean(x, axis=0)
    return (x - meanv) / stdv

def _one_hot(y, k):
    """
    y is a vector; k is the maximum number of classes
    """
    y_one_hot = np.zeros((y.size, k))
    y_one_hot[np.arange(y.size), y] = 1
    return y_one_hot

def _compute_log_det_cholesky(matrix_chol, covariance_type, n_features):
    """Compute the log-det of the cholesky decomposition of matrices.

    Parameters
    ----------
    matrix_chol : array-like
        Cholesky decompositions of the matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    n_features : int
        Number of features.

    Returns
    -------
    log_det_precision_chol : array-like of shape (n_components,)
        The determinant of the precision matrix for each component.
    """
    if covariance_type == "full":
        n_components, _, _ = matrix_chol.shape
        log_det_chol = np.sum(
            np.log(matrix_chol.reshape(n_components, -1)[:, :: n_features + 1]), 1
        )

    elif covariance_type == "tied":
        log_det_chol = np.sum(np.log(np.diag(matrix_chol)))

    elif covariance_type == "diag":
        log_det_chol = np.sum(np.log(matrix_chol), axis=1)

    else:
        log_det_chol = n_features * (np.log(matrix_chol))

    return log_det_chol

def _log_wishart_norm(degrees_of_freedom, log_det_precisions_chol, n_features):
    """Compute the log of the Wishart distribution normalization term.

    Parameters
    ----------
    degrees_of_freedom : array-like of shape (n_components,)
        The number of degrees of freedom on the covariance Wishart
        distributions.

    log_det_precision_chol : array-like of shape (n_components,)
         The determinant of the precision matrix for each component.

    n_features : int
        The number of features.

    Return
    ------
    log_wishart_norm : array-like of shape (n_components,)
        The log normalization of the Wishart distribution.
    """
    # To simplify the computation we have removed the np.log(np.pi) term
    return -(
        degrees_of_freedom * log_det_precisions_chol
        + degrees_of_freedom * n_features * 0.5 * math.log(2.0)
        + np.sum(
            gammaln(0.5 * (degrees_of_freedom - np.arange(n_features)[:, np.newaxis])),
            0,
        )
    )

class BayesianGaussianMixtureMerge(BayesianGaussianMixture):
    def __init__(
        self,
        *,
        n_components=1,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=None,
        mean_precision_prior=None,
        mean_prior=None,
        degrees_of_freedom_prior=None,
        covariance_prior=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
    ):
        super().__init__(
            n_components=n_components,
            covariance_type=covariance_type,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            weight_concentration_prior_type=weight_concentration_prior_type,
            weight_concentration_prior=weight_concentration_prior,
            mean_precision_prior=mean_precision_prior,
            mean_prior=mean_prior,
            degrees_of_freedom_prior=degrees_of_freedom_prior,
            covariance_prior=covariance_prior,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,
        )


    # def _initialize(self, X, resp):
    #     np.set_printoptions(threshold=sys.maxsize)
    #     print('======initialize=======', resp.argmax(axis=1))
    #     nk, xk, sk = _estimate_gaussian_parameters(
    #         X, resp, self.reg_covar, self.covariance_type
    #     )
    #     self._estimate_weights(nk)
    #     self._estimate_means(nk, xk)
    #     self._estimate_precisions(nk, xk, sk)


    def _initialize_parameters_v2(self, X, y, random_state):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)

        random_state : RandomState
            A random number generator instance that controls the random seed
            used for the method chosen to initialize the parameters.
        """
        X_src = X[len(y):, ]
        n_samples, _ = X_src.shape
        n_components = self.n_components - len(np.unique(y))
        if self.init_params == "kmeans":
            resp = np.zeros((n_samples, n_components))
            label = (
                KMeans(
                    n_clusters=n_components, n_init=1, random_state=random_state
                )
                .fit(X_src)
                .labels_
            )
            resp[np.arange(n_samples), label] = 1
        elif self.init_params == "random":
            resp = random_state.uniform(size=(n_samples, n_components))
            resp /= resp.sum(axis=1)[:, np.newaxis]
        elif self.init_params == "random_from_data":
            resp = np.zeros((n_samples, n_components))
            indices = random_state.choice(
                n_samples, size=n_components, replace=False
            )
            resp[indices, np.arange(n_components)] = 1
        elif self.init_params == "k-means++":
            resp = np.zeros((n_samples, n_components))
            _, indices = kmeans_plusplus(
                X_src,
                n_components,
                random_state=random_state,
            )
            resp[indices, np.arange(self.n_components)] = 1
        elif self.init_params == 'kmeans_merge':
            #print('kmeans merge')
            _, resp = kmeans.Big_KMeans(n_clusters=self.n_components, runs=5, random_state=random_state).fit_merge(X, y)
            resp = _one_hot(resp, self.n_components)

        if self.init_params != 'kmeans_merge':
            # resp: n_tar x maxk - k_tar (e.g 80)
            resp = np.concatenate([np.zeros((n_samples, self.n_components-n_components)), resp], axis=1)

        y_one_hot = _one_hot(y, self.n_components)
        # y_joke = np.array([len(np.unique(y)) for _ in range(n_samples)])
        # y_joke = _one_hot(y_joke, self.n_components)
        resp_all = np.concatenate([y_one_hot, resp], axis=0)
        #resp = np.concatenate([y_one_hot, y_joke], axis=0)
        self._initialize(X, resp_all)
        return resp

    def _initialize_parameters_true(self, X, y, yt):
        resp = _one_hot(np.concatenate([y, yt],axis=0), self.n_components)
        self._initialize(X, resp)

    def fit_merge(self, X, y, v1=True):
        """Estimate model parameters using X and predict the labels for X.

        The method fits the model n_init times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a :class:`~sklearn.exceptions.ConvergenceWarning` is
        raised. After fitting, it predicts the most probable label for the
        input data points.

        .. versionadded:: 0.20

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        X = self._validate_data(X, dtype=[np.float64, np.float32], ensure_min_samples=2)
        if X.shape[0] < self.n_components:
            raise ValueError(
                "Expected n_samples >= n_components "
                f"but got n_components = {self.n_components}, "
                f"n_samples = {X.shape[0]}"
            )
        self._check_parameters(X)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.inf
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        n_samples, _ = X.shape
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                #self._initialize_parameters(X, random_state)
                #self._initialize_parameters_true(X, y, yt)
                init_resp = self._initialize_parameters_v2(X, y, random_state)

            lower_bound = -np.inf if do_init else self.lower_bound_

            if self.max_iter == 0:
                best_params = self._get_parameters()
                best_n_iter = 0
            else:
                for n_iter in range(1, self.max_iter + 1):
                    prev_lower_bound = lower_bound

                    log_prob_norm, log_resp, fake_resp, part_resp = self._e_step_v1(X, y)
                    #print('!!!', n_iter, '!!!', log_resp.argmax(axis=1))
                    if v1:
                        self._m_step_v1(X, fake_resp, part_resp)
                    else:
                        self._m_step_v2(X, fake_resp, part_resp)

                    lower_bound = self._compute_lower_bound(log_resp, log_prob_norm)

                    change = lower_bound - prev_lower_bound
                    self._print_verbose_msg_iter_end(n_iter, change)

                    if abs(change) < self.tol:
                        self.converged_ = True
                        break

                #self._print_verbose_msg_init_end(lower_bound)

                if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                    max_lower_bound = lower_bound
                    best_params = self._get_parameters()
                    best_n_iter = n_iter
                    best_init = init_resp

        # Should only warn about convergence if max_iter > 0, otherwise
        # the user is assumed to have used 0-iters initialization
        # to get the initial means.
        if not self.converged_ and self.max_iter > 0:
            warnings.warn(
                "Initialization %d did not converge. "
                "Try different init parameters, "
                "or increase max_iter, tol "
                "or check for degenerate data." % (init + 1),
                ConvergenceWarning,
            )

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        _, log_resp, _, _ = self._e_step_v1(X, y)
        # tmp = np.exp(log_resp)
        # np.set_printoptions(threshold=sys.maxsize)
        #print(log_resp)
        return log_resp.argmax(axis=1), np.exp(log_resp), best_init


    def _e_step_v1(self, X, y):
        """E step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns        tar_data = tar_data1
        tar_label = tar_label1

        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X

        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        log_prob_norm, log_resp, fake_resp, part_resp = self._estimate_log_prob_resp_y(X, y)
        return np.mean(log_prob_norm), log_resp, fake_resp, part_resp


    def _m_step_v1(self, X, fake_resp, part_resp):
        """M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape
        nk2 = part_resp.sum(axis=0) + 10*np.finfo(part_resp.dtype).eps
        nk, xk, sk = _estimate_gaussian_parameters(
            X, fake_resp, self.reg_covar, self.covariance_type
        )
        self._estimate_weights(nk)
        self._estimate_means(nk, xk)
        self._estimate_precisions(nk, xk, sk)
    def _m_step_v2(self, X, fake_resp, part_resp):
        """M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape
        nk2 = part_resp.sum(axis=0) + 10*np.finfo(part_resp.dtype).eps
        nk, xk, sk = _estimate_gaussian_parameters(
            X, fake_resp, self.reg_covar, self.covariance_type
        )
        self._estimate_weights(nk2)
        self._estimate_means(nk, xk)
        self._estimate_precisions(nk, xk, sk)

    def _estimate_log_prob_resp_y(self, X, y):
        """Estimate log probabilities and responsibilities for each sample.

        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            log p(X)

        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under="ignore"):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        # tmp = np.exp(log_resp)
        # print('============', tmp.shape, tmp.sum(axis=1), tmp.sum(axis=1).shape)
        ## we arrange data in a way x=[source_data, target_data]; so y=[source_label] will replace
        ## log_resp for the first several rows(samples) with one-hot given from the label
        y_one_hot = _one_hot(y, self.n_components)
        resp = np.exp(log_resp)
        np.set_printoptions(threshold=sys.maxsize)
        # print('======', resp[10:12,:])
        # print('======', resp[1900:1902,:])
        # print('=====', resp.argmax(axis=1))
        # print('==y==', y)

        fake_resp = np.concatenate([y_one_hot, resp[len(y_one_hot):, ]], axis=0)
        # np.set_printoptions(threshold=sys.maxsize)
        #print('============', y_one_hot.shape, resp.shape, fake_resp.shape, fake_resp.sum(axis=1))
        assert resp.shape == fake_resp.shape
        return log_prob_norm, log_resp[len(y_one_hot):, ], fake_resp, resp[len(y_one_hot):, ]

    def post_process(self, y_tar, y_src, threshold=2):
        """
        filter out singleton cluster from y_tar with a threshold; e.g. filter out a cluster with size < 2
        only filter out these clusters that are not in y_src
        filter outed samples will be marked as the last cluster
        """
        c_src = np.unique(y_src)
        y_tar_unsrc = y_tar[~np.isin(y_tar, c_src)]
        unique_unsrc, cnt_unsrc = np.unique(y_tar_unsrc, return_counts=True)
        filter_out_idx = len(c_src) + np.sum(cnt_unsrc > threshold)
        y_tar_fout = np.copy(y_tar)
        j = 0
        for i, u in enumerate(unique_unsrc):
            if cnt_unsrc[i] > threshold:
                y_tar_fout[y_tar == u] = len(c_src) + j
                j += 1
            else:
                y_tar_fout[y_tar == u] = filter_out_idx

        return y_tar_fout, filter_out_idx



if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(description='Domain adaptation using ADDA')
    arg_parser.add_argument("--use_sign", action="store_true", help="increase output verbosity")
    arg_parser.add_argument('--max_k', type=int, default=100)
    arg_parser.add_argument('--alpha', type=float, default=1.)
    arg_parser.add_argument('--embed_dims', type=int, default=32)
    arg_parser.add_argument('--prior_sigma_scale', type=float, default=1.)
    arg_parser.add_argument('--prior', type=str, default="NIW")
    arg_parser.add_argument(
        "--prior_mu_0",
        type=str,
        default="data_mean",
    )
    arg_parser.add_argument(
        "--prior_sigma_choice",
        type=str,
        default="isotropic",
        choices=["iso_005", "iso_001", "iso_0001", "data_std"],
    )
    arg_parser.add_argument(
        "--prior_kappa",
        type=float,
        default=1.,
    )
    arg_parser.add_argument(
        "--NIW_prior_nu",
        type=float,
        default=30,
        help="Need to be at least codes_dim + 1",
    )
    args = arg_parser.parse_args()

    #smallsim=True
    smallsim=False

    if smallsim:
        num_components = 5
        num_components_cn = 3
        num_components_src = 3
        num_components_tar = num_components_src - num_components_cn
        random_state = 100
        rng = np.random.RandomState(random_state)
            # Means, standard deviations, and weights for each component
        dim = 3
        #means = rng.rand(num_components,dim)
        means = rng.multivariate_normal(np.zeros(dim), np.eye(dim)*50, num_components)
        ccov = np.absolute(rng.rand(num_components,dim))
        print(means, ccov)

        cov = np.eye(dim)
        num_samples_src = 200
        num_samples_tar = 1800
        tar_weights = rng.dirichlet([10 for i in range(num_components_tar, num_components)])
        src_label = rng.randint(num_components_src, size=num_samples_src)
        print(src_label, tar_weights)
        # Initialize an empty array to store the generated data
        src_data = np.zeros((num_samples_src, dim))
        tar_data = np.zeros((num_samples_tar, dim))
        tar_label = rng.choice(np.arange(num_components_tar, num_components), size=num_samples_tar, p=tar_weights)
        print(tar_label)
        # Generate data points
        for i in range(num_samples_src):
            sl = src_label[i]
            # Generate a random sample from the selected component
            src_data[i,:] = rng.multivariate_normal(means[sl],cov*ccov[sl])
        for i in range(num_samples_tar):
            tl = tar_label[i]
            # Generate a random sample from the selected component
            tar_data[i,:] = rng.multivariate_normal(means[tl],cov*ccov[tl])
        # print(src_data.shape)
        # print(tar_data.shape)
        # print(src_label.shape)

        # src_data = np_standardize(src_data)
        # tar_data = np_standardize(tar_data)
        data = np.concatenate([src_data, tar_data], axis=0)
        #data = np_standardize(data)
        dp = BayesianGaussianMixtureMerge(
            n_components=args.max_k,
            n_init=20,
            weight_concentration_prior=args.alpha / args.max_k,
            weight_concentration_prior_type='dirichlet_process',
            covariance_prior=dim * np.identity(dim),
            covariance_type='full')

        tar_pred, _ = dp.fit_merge(data, src_label)
        from sklearn import mixture

        # dpgmm = mixture.BayesianGaussianMixture(
        #     n_components=args.max_k,
        #     weight_concentration_prior=args.alpha / args.max_k,
        #     weight_concentration_prior_type='dirichlet_process',
        #     covariance_prior=dim * np.identity(dim),
        #     covariance_type='full').fit(data)
        # preds = dpgmm.predict(data)
        # tar_pred = preds[num_samples_src:]

        print(tar_pred)
        unique, count = np.unique(tar_pred, return_counts=True)
        print(tar_pred.shape, len(unique), count,count.sum())
        print('nmi is', nmi(tar_pred, tar_label))
        tgt_member_new = tar_pred
        tar_label_known = tar_label[tar_label < num_components_src]
        tgt_member_new_known = tgt_member_new[tar_label < num_components_src]
        tar_label_unknown = tar_label[tar_label > num_components_src-1]
        tgt_member_new_unknown = tgt_member_new[tar_label > num_components_src-1]
        print('known acc', np.sum(tar_label_known == tgt_member_new_known)/np.sum(tar_label < num_components_src), len(tar_label_known))
        print('unknown nmi', nmi(tar_label_unknown, tgt_member_new_unknown), tar_label_unknown, tgt_member_new_unknown, len(tar_label_unknown))
        # define unknown as positive
        TP = np.sum((tar_label > num_components_src-1) & (tgt_member_new > num_components_src-1))
        TN = np.sum((tar_label < num_components_src) & (tgt_member_new < num_components_src))
        FP = np.sum((tar_label < num_components_src) & (tgt_member_new > num_components_src-1))
        FN = np.sum((tar_label > num_components_src-1) & (tgt_member_new < num_components_src))
        print('recall is ', TP/(TP+FN), TP, TP+FN)
        print('precision is', TP/(TP+FP), TP, TP+FP)

        y_tar_fout, filter_out_idx = dp.post_process(tar_pred, src_label)

    elif ~smallsim and True:
        ## generate mixture of gaussian for source and target
        # Number of Gaussian components
        num_components = 30
        num_components_cn = 10
        num_components_src = 20
        num_components_tar = num_components_src - num_components_cn + 1
        random_state = 100000
        rng = np.random.RandomState(random_state)
            # Means, standard deviations, and weights for each component
        dim = args.embed_dims
        #means = rng.rand(num_components,dim)
        means = rng.multivariate_normal(np.zeros(dim), np.eye(dim), num_components)
        ccov = np.absolute(rng.rand(num_components,dim))
        cov = np.eye(dim)
        num_samples_src = 200
        num_samples_tar = 1800
        # num_samples_src = 100
        # num_samples_tar = 1800
        tar_weights = rng.dirichlet([1 for i in range(num_components_tar, num_components)])
        src_label = rng.randint(num_components_src, size=num_samples_src)
        #print(src_label)
        # Initialize an empty array to store the generated data
        src_data = np.zeros((num_samples_src, dim))
        tar_data = np.zeros((num_samples_tar, dim))
        tar_label = rng.choice(np.arange(num_components_tar, num_components), size=num_samples_tar, p=tar_weights)
        print(tar_label)
        # Generate data points
        for i in range(num_samples_src):
            sl = src_label[i]
            # Generate a random sample from the selected component
            src_data[i,:] = rng.multivariate_normal(means[sl],cov*ccov[sl])
        for i in range(num_samples_tar):
            tl = tar_label[i]
            # Generate a random sample from the selected component
            tar_data[i,:] = rng.multivariate_normal(means[tl],cov*ccov[tl])
        # print(src_data.shape)
        # print(tar_data.shape)
        # print(src_label.shape)

        # src_data = np_standardize(src_data)
        # tar_data = np_standardize(tar_data)
        data = np.concatenate([src_data, tar_data], axis=0)
        # data1 = np.concatenate([src_data, tar_data[:500]], axis=0)
        # data2 = np.concatenate([src_data, tar_data[500:1000]], axis=0)
        # data3 = np.concatenate([src_data, tar_data[1000:]], axis=0)
        # from lib import whitening
        # data = whitening(data, method='cholesky')
        #data = np_standardize(data)
        print(src_data.shape)
        import time
        t1 = time.time()
        dp = BayesianGaussianMixtureMerge(
            n_components=args.max_k,
            n_init=5,
            #random_state=2,
            init_params='kmeans_merge',
            weight_concentration_prior=args.alpha / args.max_k,
            weight_concentration_prior_type='dirichlet_process',
            covariance_prior=args.embed_dims * np.identity(args.embed_dims),
            covariance_type='full')
        t2 = time.time()
        print(t2-t1)

        tar_pred, _, _ = dp.fit_merge(data, src_label)
        # tar_pred1, _ = dp.fit_merge(data1, src_label)
        # tar_pred2, _ = dp.fit_merge(data2, src_label)
        # tar_pred3, _ = dp.fit_merge(data3, src_label)
        # tar_pred = np.concatenate([tar_pred1,tar_pred2,tar_pred3],axis=0)

        unique, count = np.unique(tar_pred, return_counts=True)
        print(tar_pred.shape, len(unique), count,count.sum())
        print('nmi is', nmi(tar_pred, tar_label))
        tgt_member_new = tar_pred
        tar_label_known = tar_label[tar_label < 20]
        tgt_member_new_known = tgt_member_new[tar_label < 20]
        tar_label_unknown = tar_label[tar_label > 19]
        tgt_member_new_unknown = tgt_member_new[tar_label > 19]
        print('known acc', np.sum(tar_label_known == tgt_member_new_known)/np.sum(tar_label < 20), len(tar_label_known))
        print('unknown nmi', nmi(tar_label_unknown, tgt_member_new_unknown), tar_label_unknown, tgt_member_new_unknown, len(tar_label_unknown))
        # define unknown as positive
        TP = np.sum((tar_label > 19) & (tgt_member_new > 19))
        TN = np.sum((tar_label < 20) & (tgt_member_new < 20))
        FP = np.sum((tar_label < 20) & (tgt_member_new > 19))
        FN = np.sum((tar_label > 19) & (tgt_member_new < 20))
        print('recall is ', TP/(TP+FN), TP, TP+FN)
        print('precision is', TP/(TP+FP), TP, TP+FP)
        y_tar_fout, filter_out_idx = dp.post_process(tar_pred, src_label)


        data = np.concatenate([src_data, tar_data], axis=0)
        # data = whitening(data, method='zca')
        y = np.concatenate([src_label, tar_label], axis=0)
        m = np.array([1 for _ in range(len(src_label))] + [5 for _ in range(len(tar_label))])
        s = np.array([5 for _ in range(len(src_label))] + [2 for _ in range(len(tar_label))])
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(data)
        tsne.kl_divergence_

        fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=y, symbol=m, size=s)
        fig.update_layout(
            title="t-SNE visualization of Custom Classification dataset",
            xaxis_title="First t-SNE",
            yaxis_title="Second t-SNE",
        )
        fig.show()

        # tar_pred, _ = dp.fit_merge(data, src_label, v1=False)
        #
        # unique, count = np.unique(tar_pred, return_counts=True)
        # print(tar_pred.shape, len(unique), count,count.sum())
        # print('nmi is', nmi(tar_pred, tar_label))
        # tgt_member_new = tar_pred
        # tar_label_known = tar_label[tar_label < 20]
        # tgt_member_new_known = tgt_member_new[tar_label < 20]
        # tar_label_unknown = tar_label[tar_label > 19]
        # tgt_member_new_unknown = tgt_member_new[tar_label > 19]
        # print('known acc', np.sum(tar_label_known == tgt_member_new_known)/np.sum(tar_label < 20), len(tar_label_known))
        # #print('unknown nmi', nmi(tar_label_unknown, tgt_member_new_unknown), tar_label_unknown, tgt_member_new_unknown, len(tar_label_unknown))
        # # define unknown as positive
        # TP = np.sum((tar_label > 19) & (tgt_member_new > 19))
        # TN = np.sum((tar_label < 20) & (tgt_member_new < 20))
        # FP = np.sum((tar_label < 20) & (tgt_member_new > 19))
        # FN = np.sum((tar_label > 19) & (tgt_member_new < 20))
        # print('recall is ', TP/(TP+FN), TP, TP+FN)
        # print('precision is', TP/(TP+FP), TP, TP+FP)
        # y_tar_fout, filter_out_idx = dp.post_process(tar_pred, src_label)

    else:
        dm = np.load('tmp.npz')
        #dm = np.load('tmp_01.npz')
        src_data = dm['src_embedding']
        tar_data = dm['tgt_embedding']
        src_label = dm['src_member']
        tar_label = dm['tgt_member']
        #src_data = src_data[src_label < 10]
        #src_label = src_label[src_label < 10]
        # sampling
        # from numpy.random import RandomState
        # rng = RandomState(123)
        # tar_data1, tar_label1 = [], []
        # for i in np.unique(tar_label):
        #     print(np.sum(tar_label == i))
        #     idx = rng.choice(np.argwhere(tar_label == i).flatten(), 5, replace=False),
        #     tar_data1.append(tar_data[idx[0]])
        #     tar_label1.append(tar_label[idx[0]])
        #
        # tar_data1 = np.concatenate(tar_data1, axis=0)
        # tar_label1 = np.concatenate(tar_label1, axis=0)

        # tar_data = tar_data1
        # tar_label = tar_label1

        data = np.concatenate([src_data, tar_data], axis=0)

        if True:
            dp = BayesianGaussianMixtureMerge(
                n_components=args.max_k,
                n_init=5,
                #random_state=2,
                init_params='kmeans_merge',
                degrees_of_freedom_prior=args.embed_dims,
                weight_concentration_prior=args.alpha / args.max_k,
                weight_concentration_prior_type='dirichlet_process',
                mean_precision_prior=10,
                #covariance_type='diag')
                #covariance_prior=0.1*args.embed_dims * np.ones((args.embed_dims)))
                # covariance_type='full',
                covariance_prior=0.1*args.embed_dims * np.identity(args.embed_dims))

            # dpgmm = BayesianGaussianMixture(
            #     n_components=dp.n_components,
            #     weight_concentration_prior=dp.weight_concentration_prior,
            #     weight_concentration_prior_type='dirichlet_process',
            #     covariance_prior=dp.covariance_prior,
            #     covariance_type='full').fit(data)
            #
            # preds = dpgmm.predict(data)
            #
            # preds = KMeans(n_clusters=100).fit(data).labels_
            # tar_pred = preds[len(src_data):]
            # src_pred = preds[:len(src_data)]
            # src_uni = np.unique(src_pred)
            # print(src_uni, np.unique(tar_pred))
            # print(np.sum(~np.isin(tar_pred, src_uni)))
            # tar_pred1 = tar_pred[~np.isin(tar_pred, src_uni)]
            # tar_l = tar_label[~np.isin(tar_pred, src_uni)]
            # tar_l1 = tar_label[np.isin(tar_pred, src_uni)]
            # print(np.sum(tar_l >= 20))
            # print(np.sum(tar_l1 < 20))

            print(src_data.shape, tar_data.shape, src_label.shape)
            import time
            t1 = time.time()

            t2 = time.time()
            print(t2-t1)

            tar_pred, tar_pred_prob, best_init = dp.fit_merge(data, src_label)
            means, precisions = dp.means_, dp.precisions_

            #_, tar_pred = kmeans.Big_KMeans(n_clusters=100, runs=10, random_state=0).fit_merge(data, src_label)
            #print(tar_pred_prob)
            # tar_pred1, _ = dp.fit_merge(data1, src_label)
            # tar_pred2, _ = dp.fit_merge(data2, src_label)
            # tar_pred3, _ = dp.fit_merge(data3, src_label)
            # tar_pred = np.concatenate([tar_pred1,tar_pred2,tar_pred3],axis=0)
            ### apply sign to merge small clusters

            # merge_pred = np.concatenate([src_label, tar_pred], axis=0)
            # keep_list = np.unique(src_label).tolist()
            # merged, _, _ = sign_merge(data, merge_pred, keep_list)
            # src_label1 = merged[:len(src_label)]
            # tar_pred = merged[len(src_label):]
            # tar_pred = map_label(src_label, src_label1, tar_pred)

            unique, count = np.unique(tar_pred, return_counts=True)
            print(tar_pred.shape, len(unique), np.stack([unique, count], axis=0),count.sum())
            print('nmi is', nmi(tar_pred, tar_label))
            tgt_member_new = tar_pred
            tar_label_known = tar_label[tar_label < 20]
            tgt_member_new_known = tgt_member_new[tar_label < 20]
            tar_label_unknown = tar_label[tar_label > 19]
            tgt_member_new_unknown = tgt_member_new[tar_label > 19]
            print('known acc', np.sum(tar_label_known == tgt_member_new_known)/np.sum(tar_label < 20), len(tar_label_known))
            print('unknown nmi', nmi(tar_label_unknown, tgt_member_new_unknown), tar_label_unknown, tgt_member_new_unknown, len(tar_label_unknown))
            # define unknown as positive
            TP = np.sum((tar_label > 19) & (tgt_member_new > 19))
            TN = np.sum((tar_label < 20) & (tgt_member_new < 20))
            FP = np.sum((tar_label < 20) & (tgt_member_new > 19))
            FN = np.sum((tar_label > 19) & (tgt_member_new < 20))
            print('recall is ', TP/(TP+FN), TP, TP+FN)
            print('precision is', TP/(TP+FP), TP, TP+FP)
            y_tar_fout, filter_out_idx = dp.post_process(tar_pred, src_label)

            best_init = best_init.argmax(axis=1)
            y0 = np.concatenate([src_label, best_init], axis=0)
            #y = np.concatenate([src_label, y_tar_fout], axis=0)
            y = np.concatenate([src_label, tar_pred], axis=0)
            m = np.array([1 for _ in range(len(src_label))] + [5 for _ in range(len(tar_label))])
            s = np.array([5 for _ in range(len(src_label))] + [2 for _ in range(len(tar_label))])
            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(data)
            tsne.kl_divergence_

            fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=y, symbol=m, size=s)
            fig.update_layout(
                title="t-SNE visualization",
                xaxis_title="First t-SNE",
                yaxis_title="Second t-SNE",
            )
            fig.show()
            fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=y0, symbol=m, size=s)
            fig.update_layout(
                title="t-SNE visualization (initialization)",
                xaxis_title="First t-SNE",
                yaxis_title="Second t-SNE",
            )
            fig.show()
        else:

            dp = BayesianGaussianMixtureMerge(
                n_components=args.max_k,
                n_init=5,
                #random_state=2,
                init_params='kmeans_merge',
                degrees_of_freedom_prior=args.embed_dims,
                weight_concentration_prior=args.alpha / args.max_k,
                weight_concentration_prior_type='dirichlet_process',
                mean_precision_prior=10,
                #covariance_type='diag')
                #covariance_prior=0.1*args.embed_dims * np.ones((args.embed_dims)))
                # covariance_type='full',
                covariance_prior=0.0002* np.identity(2))
            m = np.array([1 for _ in range(len(src_label))] + [5 for _ in range(len(tar_label))])
            s = np.array([5 for _ in range(len(src_label))] + [2 for _ in range(len(tar_label))])
            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(data)
            tsne.kl_divergence_
            tar_pred, tar_pred_prob, best_init = dp.fit_merge(X_tsne, src_label)
            unique, count = np.unique(tar_pred, return_counts=True)
            print(tar_pred.shape, len(unique), np.stack([unique, count], axis=0),count.sum())
            print('nmi is', nmi(tar_pred, tar_label))
            tgt_member_new = tar_pred
            tar_label_known = tar_label[tar_label < 20]
            tgt_member_new_known = tgt_member_new[tar_label < 20]
            tar_label_unknown = tar_label[tar_label > 19]
            tgt_member_new_unknown = tgt_member_new[tar_label > 19]
            print('known acc', np.sum(tar_label_known == tgt_member_new_known)/np.sum(tar_label < 20), len(tar_label_known))
            print('unknown nmi', nmi(tar_label_unknown, tgt_member_new_unknown), tar_label_unknown, tgt_member_new_unknown, len(tar_label_unknown))
            # define unknown as positive
            TP = np.sum((tar_label > 19) & (tgt_member_new > 19))
            TN = np.sum((tar_label < 20) & (tgt_member_new < 20))
            FP = np.sum((tar_label < 20) & (tgt_member_new > 19))
            FN = np.sum((tar_label > 19) & (tgt_member_new < 20))
            print('recall is ', TP/(TP+FN), TP, TP+FN)
            print('precision is', TP/(TP+FP), TP, TP+FP)
            y = np.concatenate([src_label, tar_pred], axis=0)
            fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=y, symbol=m, size=s)
            fig.update_layout(
                title="t-SNE visualization",
                xaxis_title="First t-SNE",
                yaxis_title="Second t-SNE",
            )
            fig.show()
            y = np.concatenate([src_label, tar_label], axis=0)
            fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=y, symbol=m, size=s)
            fig.update_layout(
                title="t-SNE visualization",
                xaxis_title="First t-SNE",
                yaxis_title="Second t-SNE",
            )
            fig.show()
