from .optim_iterator import OptimIterator
from deepinv.optim.coarse_model import CoarseModel


class MultiLevelIteration(OptimIterator):
    def __init__(self, fine_iteration, **kwargs):
        super(MultiLevelIteration, self).__init__(**kwargs)
        self.fine_iteration=fine_iteration
        self.ml_iter = 0

    def multilevel_step(self, X, data_fidelity, prior, params, y, physics):
        if params['level'] == 1 or self.ml_iter >= params['iml_max_iter']:
            Y = X.copy()
            return Y

        self.ml_iter += 1
        model = CoarseModel(prior, data_fidelity, physics, params)
        diff = model(X, y, params)
        step = 1.0
        if self.fine_iteration.has_cost:
            # performing backtracking if cost exists
            print("bt", params['level'])
            x0 = X['est'][0]
            while (self.fine_iteration.F_fn(x0, prior, params, y, physics) >
                self.fine_iteration.F_fn(diff, y, params)):

                step = step / 2
        x_bt = X['est'][0] + step * diff
        Y = {'est': [x_bt]}

        return Y

    def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics):
        Y = self.multilevel_step(X, cur_data_fidelity, cur_prior, cur_params, y, physics)

        cur_level_params = self.get_level_params(cur_params)
        X2 = self.fine_iteration(Y, cur_data_fidelity, cur_prior, cur_level_params, y, physics)
        return X2

    @staticmethod
    def get_level_params(params_alg):
        if not isinstance(params_alg, dict):
            raise NotImplementedError

        params_multilevel = params_alg['params_multilevel']
        level = params_alg['level']

        params_level = params_alg.copy()
        for key_ in params_multilevel.keys():
            params_level[key_] = params_multilevel[key_][level - 1]

        return params_level
