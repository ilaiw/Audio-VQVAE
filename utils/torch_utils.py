import gc
import torch as t

def freeze_model(model):
    model.eval()
    for params in model.parameters():
        params.requires_grad = False


def unfreeze_model(model):
    model.train()
    for params in model.parameters():
        params.requires_grad = True

def zero_grad(model):
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad = None

def empty_cache():
    gc.collect()
    t.cuda.empty_cache()

def assert_shape(x, exp_shape):
    assert x.shape == exp_shape, f"Expected {exp_shape} got {x.shape}"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_state(model):
    return sum(s.numel() for s in model.state_dict().values())

# My added utils:
class Metrics(dict):
    def append_metrics(self, metrics_j, suffix=''):
        for k, v in metrics_j.items():
            if type(v) is t.Tensor:
                v = v.item()
            ks = k + suffix
            if ks in self.keys():
                self[ks].append(v)
            else:
                self[ks] = [v]

    def get_avg_metrics(self):
        return {k: sum(v)/len(v) for k, v in self.items()}


def GPU_usage():
    if not t.cuda.is_available():
        return ''
    GB = 1024**3
    alctd = t.cuda.memory_allocated(0)/GB
    res = t.cuda.memory_reserved(0)/GB
    max_res = t.cuda.max_memory_reserved(0)/GB
    return f'[alc|res|max:{alctd:.2f}|{res:.2f}|{max_res:.2f} GB]'


def get_device():
    if t.cuda.is_available():
        device = t.device('cuda')
        GPU_name = t.cuda.get_device_name()
    else:
        device = t.device('cpu')
        GPU_name = ''
    return device, GPU_name
