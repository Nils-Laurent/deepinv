import torch

from .optim_iterator import OptimIterator, fStep, gStep
from .utils import gradient_descent_step


class GDIteration(OptimIterator):
    r"""
    Iterator for Gradient Descent.

    Class for a single iteration of the gradient descent (GD) algorithm for minimising :math:`\lambda f(x) + g(x)`.

    The iteration is given by


    .. math::
        \begin{equation*}
        \begin{aligned}
        v_{k} &= \nabla f(x_k) + \nabla g(x_k) \\
        x_{k+1} &= x_k-\gamma v_{k}
        \end{aligned}
        \end{equation*}


   where :math:`\gamma` is a stepsize.
    """

    def __init__(self, grad_fn = None, **kwargs):
        super(GDIteration, self).__init__(**kwargs)
        self.g_step = gStepGD(**kwargs)
        self.f_step = fStepGD(**kwargs)
        self.requires_grad_g = True
        self.grad_fn = grad_fn

    def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics):
        r"""
        Single gradient descent iteration on the objective :math:`f(x) + \lambda g(x)`.

        :param dict X: Dictionary containing the current iterate :math:`x_k`.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param deepinv.optim.prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :return: Dictionary `{"est": (x, ), "cost": F}` containing the updated current iterate and the estimated current cost.
        """
        x_prev = X["est"]
        if not isinstance(x_prev, torch.Tensor):
            x_prev = x_prev[0]

        grad = self.gradient(x_prev, cur_data_fidelity, cur_prior, cur_params, y, physics)
        x = gradient_descent_step(x_prev, grad)
        F = self.F_fn(x, cur_prior, cur_params, y, physics) if self.has_cost else None
        return {"est": (x,), "cost": F}

    def denoiser_MSE_step(self, x, data_fidelity, prior, y, physics, params):
        den_y = prior.denoiser(y, params['sigma_denoiser'])
        params1 = params.copy()
        params1['stepsize'] = 1.0
        grad_x = self.gradient(x, data_fidelity, prior, params1, y, physics)

        diff = torch.flatten(physics.A(x) - den_y)
        grad_vec = torch.flatten(grad_x)
        return torch.dot(diff, grad_vec) / torch.norm(grad_vec, p=2)**2

    def gradient(self, x_prev, cur_data_fidelity, cur_prior, cur_params, y, physics):
        if self.grad_fn is None:
            grad =  (
                self.g_step(x_prev, cur_prior, cur_params)
                + self.f_step(x_prev, cur_data_fidelity, cur_params, y, physics)
            )
        else:
            grad = self.grad_fn(x_prev)

        return cur_params["stepsize"] * grad


class fStepGD(fStep):
    r"""
    GD fStep module.
    """

    def __init__(self, **kwargs):
        super(fStepGD, self).__init__(**kwargs)

    def forward(self, x, cur_data_fidelity, cur_params, y, physics):
        r"""
        Single gradient descent iteration on the data fit term :math:`f`.

        :param torch.Tensor x: current iterate :math:`x_k`.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :param deepinv.physics physics: Instance of the physics modeling the data-fidelity term.
        """
        return cur_data_fidelity.grad(x, y, physics)


class gStepGD(gStep):
    r"""
    GD gStep module.
    """

    def __init__(self, **kwargs):
        super(gStepGD, self).__init__(**kwargs)

    def forward(self, x, cur_prior, cur_params):
        r"""
        Single iteration step on the prior term :math:`\lambda g`.

        :param torch.Tensor x: Current iterate :math:`x_k`.
        :param deepinv.optim.prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        """
        return cur_params["lambda"] * cur_prior.grad(x, cur_params["g_param"])
