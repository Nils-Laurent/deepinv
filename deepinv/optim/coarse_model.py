import torch
from .info_transfer import WaveletTransfer


class CoarseModel(torch.nn.Module):
    def __init__(self, prior, data_fidelity, physics, params_ml, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.g = prior
        self.f = data_fidelity
        self.physics = physics
        self.cph = physics.to_coarse()
        self.cit = WaveletTransfer(wavelet_type=params_ml['cit'])

    @staticmethod
    def stepsize_vec_mse(iteration, x0, data_fidelity, prior, params, y, physics):
        from deepinv.optim.optim_iterators.multi_level import MultiLevelIteration
        cit = WaveletTransfer(wavelet_type=params['cit'])
        cit.build_cit_matrices(x0)
        sz_vec = []
        ph = physics
        for i in range(0, params['level']):
            level_params = MultiLevelIteration.get_level_params(params)
            sz = iteration.denoiser_MSE_step(x0, data_fidelity, prior, y, ph, level_params)
            x0 = cit.projection(x0)
            y = cit.projection(y)
            cit.build_cit_matrices(x0)
            ph = ph.to_coarse()
            sz_vec.append(sz.item())
        sz_vec.reverse()
        return sz_vec

    def grad(self, x, y, physics, params):
        grad_f = self.f.grad(x, y, physics)

        # if self.g.denoiser
        if hasattr(self.g, 'denoiser'):
            sigma_d = params['sigma_denoiser']
            # todo: verify if sigma_d is taken into account correctly
            # grad_g = self.g.grad(x, sigma_denoiser=sigma_d) / sigma_d**2
            grad_g = self.g.grad(x, sigma_denoiser=sigma_d)
        elif hasattr(self.g, 'moreau_grad') and 'gamma_moreau' in params.keys():
            grad_g = self.g.moreau_grad(x, params["g_param"] * params['gamma_moreau'])
        else:
            grad_g = self.g.grad(x)

        return grad_f + params["g_param"] * grad_g

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
        params['stepsize'] = 1.0 / (1 + params['gamma_moreau'])

        x0_h = X['est'][0] # primal value of 'est'

        # Projection
        self.cit.build_cit_matrices(x0_h)
        x0 = self.cit.projection(x0_h)
        y = self.cit.projection(y_h)

        if grad is None:
            grad_x0 = self.grad(x0_h, y_h, self.physics, params_h)
        else:
            grad_x0 = grad(x0_h)

        v = self.cit.projection(grad_x0)
        v -= self.grad(x0, y, self.cph, params)

        # Coarse gradient (first order coherent)
        grad_coarse = lambda x: self.grad(x, y, self.cph, params) + v

        level_iteration = GDIteration(has_cost=False, grad_fn=grad_coarse)
        iteration = MultiLevelIteration(level_iteration, has_cost=False)

        # todo: verify if f_init is used correctly
        f_init = lambda def_y, def_ph: {'est': x0, 'cost': None}
        iters_vec = params['params_multilevel']['iters']
        model = optim_builder(
            iteration,
            data_fidelity=self.f,
            prior=self.g,
            custom_init=f_init,
            max_iter=iters_vec[params_ml['level']-1],
            params_algo=params_ml.copy(),
        )
        x_est_coarse = model(y, self.cph)

        return self.cit.prolongation(x_est_coarse - x0)
