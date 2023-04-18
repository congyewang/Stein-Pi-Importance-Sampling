from stein_q_thinning.target import QTargetKGM
from stein_q_thinning.util import vkgm

import numpy as np
from scipy.stats import wishart

class TestVKGM:
    def test_vkgm(self):
        dim = np.random.randint(low=2, high=30)

        log_p = lambda x: -np.dot(x, x)/2
        grad_log_p = lambda x: -x
        hess_log_p = lambda x: -1

        linv = wishart.rvs(dim+10, np.eye(dim), 1)
        s = np.random.randint(low=3, high=10)

        x = np.random.normal(size=(10, dim))
        sx = grad_log_p(x)

        std_check = QTargetKGM(log_p, grad_log_p, hess_log_p, linv, s)
        res_check = np.array([std_check.stein_kernel(i) for i in x])

        assert (np.abs(res_check - vkgm(x, sx, linv, s)) < 1e-10).all
