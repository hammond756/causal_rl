import torch
import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, dim):
        super(Predictor, self).__init__()
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
        self.predict = Predictor(self.dim)

    def forward(self, observation, intervention):
        noise = self.abduct(observation)
        self.noise = noise
        prediction = self.predict(noise, intervention)
        return prediction


predictors = {
    'two_step': TwoStepPredictor
}
