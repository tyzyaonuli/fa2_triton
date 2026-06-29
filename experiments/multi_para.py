import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit
import torch.nn as nn
import math
from collections.abc import Callable, Iterable


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x

def log_softmax(x, dim=-1):
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x = x - x_max
    return x - torch.log(torch.sum(torch.exp(x), dim=dim, keepdim=True))

def cross_entropy(inputs, targets):
    negative_log_softmax_logits = -log_softmax(inputs)
    return torch.mean(torch.gather(negative_log_softmax_logits, -1, targets.unsqueeze(-1)))

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Can either apply weight decay here, or at the very end
                # p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")

                state = self.state[p]
                alpha = group["lr"]
                beta_1, beta_2 = group["betas"]
                eps = group["eps"]
                t = state.get("t", 1)
                prev_m_t = state.get("m", torch.zeros_like(grad))
                prev_v_t = state.get("v", torch.zeros_like(grad))

                m_t = beta_1 * prev_m_t + ((1 - beta_1) * grad)
                v_t = beta_2 * prev_v_t + ((1 - beta_2) * torch.square(grad))

                alpha_t = alpha * (math.sqrt(1 - (beta_2**t)) / (1 - (beta_1**t)))
                p.data -= alpha_t * m_t / (torch.sqrt(v_t) + eps)
                # Apply weight decay
                p.data -= alpha * group["weight_decay"] * p.data

                state["m"] = m_t
                state["v"] = v_t
                state["t"] = t + 1

        return loss

# ------------------------------------------------------------------------------
# 1. 初始化分布式（单卡模拟多卡，你 3060 直接用）
# ------------------------------------------------------------------------------
def init(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(0)  # 全部用 3060，不会报错


# ------------------------------------------------------------------------------
# 2. 三个核心操作：你作业要考的全部在这里！
# ------------------------------------------------------------------------------
def demo_operations(rank, world_size, ram: int):
    # 每个卡造自己的数据
    tensor = torch.tensor(range(rank,int(ram*1e6/4)+rank)).cuda()
    print(f"[{rank}] 原始数据: {tensor}")

    # ==========================
    # 操作 1：All-Reduce
    # ==========================
    start = timeit.default_timer()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    end = timeit.default_timer()

    print(f"[{rank}] all_reduce 后: {end-start}")
    if rank == 0:
        print("\n=== All-Reduce 结果 ==")
    dist.barrier()
    print(f"[{rank}] all_reduce 后: {tensor}")

def check_model_equal(model, rank):
    # 把整个模型的所有参数拼成一个大张量
    all_params = []
    for p in model.parameters():
        all_params.append(p.data.flatten())
    full_tensor = torch.cat(all_params)

    # 计算总和（只要总和一样，参数就一样）
    total = full_tensor.sum().item()
    print(f"[{rank}] 模型参数总和 = {total:.4f}")

def broadcast_model(model):
    for param in model.parameters():
        dist.broadcast(param.data, src = 0)

def train(rank, world_size,steps, device):
    torch.manual_seed(42)

    model = ToyModel(10, 10).to(device)
    broadcast_model(model)
    time1 = 0
    start1 = timeit.default_timer()
    for step in range(steps):
        inputs = torch.randn(
            size=(4, 10, 10),
            dtype=torch.float32,
            device=device
        )

        targets = torch.randint(
            low=0,
            high=10,
            size=(4, 10, 1),
            dtype=torch.long,  # 必须是 long !!!
            device=device
        )

        input = inputs.chunk(world_size)[rank]
        target = targets.chunk(world_size)[rank]

        adamw = AdamW(model.parameters())


        logits = model(input)
        logits = logits.view(-1,logits.size(-1))
        target = target.view(-1)

        loss = cross_entropy(logits, target)

        adamw.zero_grad()
        loss.backward()
        start2 = timeit.default_timer()

        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(world_size)

        torch.cuda.synchronize()
        end2 = timeit.default_timer()
        adamw.step()
        time1 += end2 - start2
        torch.cuda.synchronize()

    end1 = timeit.default_timer()
    time2 = end1 - start1
    print(time1 / time2, time1, time2)
    check_model_equal(model, rank)


def train_optimal(rank, world_size, steps, device):
    torch.manual_seed(42)

    model = ToyModel(10, 10).to(device)
    broadcast_model(model)
    time1 = 0
    start1 = timeit.default_timer()
    for step in range(steps):
        inputs = torch.randn(
            size=(4, 10, 10),
            dtype=torch.float32,
            device=device
        )

        targets = torch.randint(
            low=0,
            high=10,
            size=(4, 10, 1),
            dtype=torch.long,  # 必须是 long !!!
            device=device
        )

        input = inputs.chunk(world_size)[rank]
        target = targets.chunk(world_size)[rank]

        adamw = AdamW(model.parameters())

        logits = model(input)
        logits = logits.view(-1, logits.size(-1))
        target = target.view(-1)

        loss = cross_entropy(logits, target)

        adamw.zero_grad()
        loss.backward()

        start2 = timeit.default_timer()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        flat = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat)
        flat.div_(world_size)
        restored_grads = torch._utils._unflatten_dense_tensors(flat, grads)
        for g, p in zip(restored_grads, model.parameters()):
            p.grad.copy_(g)

        torch.cuda.synchronize()
        end2 = timeit.default_timer()
        adamw.step()
        time1 += end2 - start2
        torch.cuda.synchronize()

    end1 = timeit.default_timer()
    time2 =end1 - start1
    print(time1 / time2, time1, time2)
    check_model_equal(model, rank)
def run(rank, world_size, steps, device):
    init(rank, world_size)
    train(rank, world_size, steps, device)
    dist.destroy_process_group()

def run_optimal(rank, world_size, steps, device):
    init(rank, world_size)
    train_optimal(rank, world_size, steps, device)
    dist.destroy_process_group()

def data_parallel(world_size, steps, device):
    # mp.set_start_method("spawn")
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=run, args=(rank, world_size, steps, device))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
def data_parallel_optimal(world_size, steps, device):
    mp.set_start_method("spawn")
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=run_optimal, args=(rank, world_size, steps, device))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

def normal_train(steps, device):
    torch.manual_seed(42)
    model = ToyModel(10, 10).to(device)
    for step in range(steps):
        input = torch.randn(
            size=(4, 10, 10),
            dtype=torch.float32,
            device=device
        )

        target = torch.randint(
            low=0,
            high=10,
            size=(4, 10, 1),
            dtype=torch.long,  # 必须是 long !!!
            device=device
        )
        adamw = AdamW(model.parameters())

        logits = model(input)
        logits = logits.view(-1, logits.size(-1))
        target = target.view(-1)

        loss = cross_entropy(logits, target)

        adamw.zero_grad()
        loss.backward()
        adamw.step()

    check_model_equal(model, -1)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    steps = 100
    data_parallel_optimal(4, steps, device)
    data_parallel(4, steps, device)

    normal_train(steps, device)

