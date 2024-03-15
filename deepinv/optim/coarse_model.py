import torch
from .info_transfer import DownsamplingTransfer


class CoarseModel(torch.nn.Module):
    def __init__(self, prior, data_fidelity, physics, params_ml, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.g = prior
        self.f = data_fidelity
        self.physics = physics
        self.cph = None
        self.cit_str = params_ml['cit']
        self.cit_op = None

    def projection(self, x):
        if self.cit_op is None:
            self.cit_op = DownsamplingTransfer(x)
            if self.cph is None:
                self.cph = self.physics.to_coarse(self.cit_op)
        x_proj = self.cit_op.projection(x)
        return x_proj

    def prolongation(self, x):
        if self.cit_op is None:
            self.cit_op = DownsamplingTransfer(x)
        x_prol = self.cit_op.prolongation(x)
        return x_prol

    def grad(self, x, y, physics, params):
        grad_f = self.f.grad(x, y, physics)

        if hasattr(self.g, 'denoiser'):
            grad_g = self.g.grad(x, sigma_denoiser=params['g_param'])
        elif hasattr(self.g, 'moreau_grad') and 'gamma_moreau' in params.keys():
            grad_g = self.g.moreau_grad(x, gamma=params['gamma_moreau'])
        else:
            grad_g = self.g.grad(x)

        return grad_f + params['lambda'] * grad_g

    def forward(self, X, y_h, params_ml_h, grad=None):
        # todo: find a better way to deal with circular imports
        from deepinv.optim import optim_builder
        from deepinv.optim.optim_iterators import GDIteration
        from deepinv.optim.optim_iterators.multi_level import MultiLevelIteration

        params_ml = params_ml_h.copy()
        params_ml['level'] = params_ml['level'] - 1

        params_h = MultiLevelIteration.get_level_params(params_ml_h)
        params = MultiLevelIteration.get_level_params(params_ml)

        # todo: compute lipschitz constant in a clever way
        if 'gamma_moreau' in params.keys():
            f_lipschitz = 1.0
            params_ml['stepsize'] = 1.0 / (f_lipschitz + params['gamma_moreau'])

        if isinstance(X['est'], torch.Tensor):
            x0_h = X['est']
        else:
            x0_h = X['est'][0]  # primal value of 'est'

        # Projection
        x0 = self.projection(x0_h)
        y = self.projection(y_h)

        if params['scale_coherent_grad'] is True:
            if grad is None:
                grad_x0 = self.grad(x0_h, y_h, self.physics, params_h)
            else:
                grad_x0 = grad(x0_h)

            v = self.projection(grad_x0)
            v -= self.grad(x0, y, self.cph, params)

            # Coarse gradient (first order coherent)
            grad_coarse = lambda x: self.grad(x, y, self.cph, params) + v
        else:
            grad_coarse = lambda x: self.grad(x, y, self.cph, params)

        level_iteration = GDIteration(has_cost=False, grad_fn=grad_coarse)
        iteration = MultiLevelIteration(level_iteration, grad_fn=grad_coarse, has_cost=False)

        # todo: verify if f_init is used correctly
        f_init = lambda def_y, def_ph: {'est': x0, 'cost': None}
        iters_vec = params['params_multilevel']['iters']
        model = optim_builder(
            iteration,
            data_fidelity=self.f,
            prior=self.g,
            custom_init=f_init,
            max_iter=iters_vec[params_ml['level'] - 1],
            params_algo=params_ml.copy(),
        )
        x_est_coarse = model(y, self.cph)

        return self.prolongation(x_est_coarse - x0)
