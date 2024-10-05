# EMA always in float, as accumulation needs lots of bits
class EMA:
    def __init__(self, params, mu=0.999):
        self.mu = mu
        self.state = [(p, self.get_model_state(p)) for p in params if p.requires_grad]

    def get_model_state(self, p):
        return p.data.float().detach().clone()

    def step(self):
        for p, state in self.state:
            state.mul_(self.mu).add_(p.data.float(), alpha=1-self.mu)

    def swap(self):
        # swap ema and model params
        for p, state in self.state:
            other_state = self.get_model_state(p)
            p.data.copy_(state.type_as(p.data))
            state.copy_(other_state)
