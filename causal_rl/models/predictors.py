import torch
import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, sem):
        super(Predictor, self).__init__()
        self.dim = sem.dim

        # heuristic: we know the true weights are lower triangular
        # heuristic: root nodes should have self-connection of 1 to
        # carry noise to prediction
        self.linear1 = nn.Parameter(
            torch.randn((self.dim, self.dim)).tril_(-1) + sem.roots
        )

    def _mask(self, vector, intervention):
        if intervention is None:
            return

        target, value = intervention
        vector.scatter_(dim=1, index=torch.tensor([[target]]), value=value)

    def forward(self, features, intervention):
        # make a copy of the input, since _mask will modify in-place
        out = torch.tensor(features)
        self._mask(out, intervention)

        for _ in range(self.dim):
            out = out.matmul(self.linear1.t())
            self._mask(out, intervention)

        return out


class OrderedPredictor(nn.Module):
    def __init__(self, dim):
        super(OrderedPredictor, self).__init__()
        self.dim = dim

        # heuristic: we know the true weights are lower triangular
        # heuristic: root nodes should have self-connection of 1
        # to carry noise to prediction
        self.linear1 = nn.Parameter(
            torch.randn((self.dim, self.dim)).tril_(-1)
        )

    def forward(self, noise, intervention):
        target, value = intervention

        output = torch.zeros_like(noise)

        for i in range(self.dim):
            if i == target:
                output[:, i] = value
                continue

            output[:, i] = self.linear1[i].matmul(output.clone().t()) \
                + noise[:, i]

        return output


class Abductor(torch.nn.Module):
    def __init__(self, dim):
        super(Abductor, self).__init__()
        self.dim = dim
        self.l1 = torch.nn.Linear(dim, dim)

    def forward(self, x):
        out = self.l1(x)
        return torch.sigmoid(out)  # if self.training else (out > 0.7).float()


class TwoStepPredictor(nn.Module):
    def __init__(self, sem):
        super(TwoStepPredictor, self).__init__()

        self.dim = sem.dim
        self.noise = None

        self.abduct = Abductor(self.dim)
        self.predict = OrderedPredictor(self.dim)

    def forward(self, observation, intervention):
        noise = self.abduct(observation)
        self.noise = noise
        prediction = self.predict(noise, intervention)
        return prediction


class TwoStepMatrixPredictor(nn.Module):
    def __init__(self, sem):
        super(TwoStepMatrixPredictor, self).__init__()

        self.dim = sem.dim
        self.noise = None

        self.abduct = Abductor(self.dim)
        self.predict = MatrixPredictor(self.dim)

    def forward(self, observation, intervention):
        noise = self.abduct(observation)
        self.noise = noise
        prediction = self.predict(noise, intervention)
        return prediction


class MatrixPredictor(nn.Module):
    def __init__(self, dim):
        super(MatrixPredictor, self).__init__()
        self.dim = dim

        # heuristic: we know the true weights are lower triangular
        # heuristic: root nodes should have self-connection of 1
        # to carry noise to prediction
        self.linear1 = nn.Parameter(
            torch.randn((self.dim, self.dim)).tril_(-1)
        )

    def forward(self, noise, intervention):
        target, value = intervention

        # create dummy vector for 'hetrogeneous coordinates'
        z = torch.cat([torch.zeros(self.dim), torch.tensor([1.])]).unsqueeze(0)
        z.requires_grad = False

        # formulate intervention as a vector to be applied to the model matrix
        do_x = torch.tensor([0. for _ in range(self.dim)] + [value])
        do_x = do_x.unsqueeze(dim=0)
        do_x.requires_grad = False

        weights = torch.cat([self.linear1, noise.t()], dim=1)

        model = torch.cat([weights, z], dim=0)

        # perform intervention on the model
        model[target, :] = do_x

        result = z.clone().t()

        for i in range(self.dim - 1):
            result = model.matmul(result)

        return result.t()[:, :-1]


predictors = {
    'repeated': Predictor,
    'two_step': TwoStepPredictor,
    'matrix': TwoStepMatrixPredictor
}
