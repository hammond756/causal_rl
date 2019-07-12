import torch
import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, dim, ordered):
        super(Predictor, self).__init__()
        self.dim = dim

        # heuristic: we know the true weights are lower triangular
        # heuristic: root nodes should have self-connection of 1
        # to carry noise to prediction

        weights = torch.randn((self.dim, self.dim))
        if ordered:
            weights.tril_(-1)

        self.linear1 = nn.Parameter(weights)

    def _iterative(self, noise, intervention):
        target, value = intervention

        output = torch.zeros_like(noise)

        for i in range(self.dim):
            if i == target:
                output[:, i] = value
                continue

            output[:, i] = self.linear1[i].matmul(output.clone().t()) \
                + noise[:, i]

        return output

    def _make_model(self, noise, intervention):
        target, value = intervention

        # create dummy vector for 'hetrogeneous coordinates'
        z = torch.cat([torch.zeros(self.dim), torch.tensor([1.])])
        z.requires_grad = False

        # formulate intervention as a vector to be applied to the model matrix
        do_x = torch.tensor([0. for _ in range(self.dim)] + [value])
        do_x.requires_grad = False

        # combine matrices
        weights = torch.cat([self.linear1, noise.t()], dim=1)
        model = torch.cat([weights, z.unsqueeze(0)], dim=0)

        # perform intervention on the model
        # TODO: verify that this can be seen as a form of dropout
        model[target, :] = do_x

        return model

    def _matrix(self, noise, intervention):

        model = self._make_model(noise, intervention)

        # compute result
        z = torch.eye(self.dim + 1)[-1]  # [0, 0, 0, ..., 1]
        for i in range(self.dim - 1):
            z = model.matmul(z)

        return z[:-1]

    def _power(self, noise, intervention):
        model = self._make_model(noise, intervention)
        z = torch.eye(self.dim + 1)[-1]  # [0, 0, 0, ..., 1]
        z = model.matrix_power(self.dim - 1).matmul(z)
        return z[:-1]

    def forward(self, noise, intervention, method):
        return {
            'matrix': self._matrix(noise, intervention),
            'power': self._power(noise, intervention),
            'iterative': self._iterative(noise, intervention)
        }[method]


class Abductor(torch.nn.Module):
    def __init__(self, dim):
        super(Abductor, self).__init__()
        self.dim = dim
        self.l1 = torch.nn.Linear(dim, dim)

    def forward(self, x):
        out = self.l1(x)
        return torch.sigmoid(out)  # if self.training else (out > 0.7).float()


class TwoStepPredictor(nn.Module):
    def __init__(self, sem, ordered, method):
        super(TwoStepPredictor, self).__init__()

        self.dim = sem.dim
        self.method = method
        self.noise = None

        self.abduct = Abductor(self.dim)
        self.predict = Predictor(self.dim, ordered)

    def forward(self, observation, intervention):
        noise = self.abduct(observation)
        self.noise = noise
        prediction = self.predict(noise, intervention, self.method)
        return prediction


predictors = {
    'two_step': TwoStepPredictor
}
