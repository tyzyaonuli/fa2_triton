from multi_para import ToyModel, AdamW, cross_entropy
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit
import torch.nn as nn
import math

class DDP():
    def __init__(self, model:nn.Module,device):
        self.model = model.to(device)
        self.handles = []
        for param in self.model.parameters():
            param.register_hook(self.make_grad_hook())

    def forward(self, *inputs, **kwargs):
        x,_ = inputs
        return self.model(x)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

    def make_grad_hook(self):
        def grad_hook(grad):
            handle = dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append(handle)
            return grad
        return grad_hook


class DDP_Bucketed:
    def __init__(self, model, bucket_size_mb, device, world_size):
        self.model = model.to(device)
        self.world_size = world_size
        self.buckets = []
        self.handles = []
        self.bucket_ready_counts = {}

        max_bucket_bytes = bucket_size_mb * 1e6

        current_bucket = []
        current_bytes = 0

        # -------- 分 bucket --------
        for p in model.parameters():
            if not p.requires_grad:
                continue

            param_bytes = p.numel() * 4

            if current_bytes + param_bytes > max_bucket_bytes:
                if len(current_bucket) == 0:
                    self.buckets.append([p])
                else:
                    self.buckets.append(current_bucket)
                    current_bucket = [p]
                    current_bytes = param_bytes
            else:
                current_bucket.append(p)
                current_bytes += param_bytes

        if len(current_bucket) > 0:
            self.buckets.append(current_bucket)

        # -------- 为每个 bucket 分配 buffer，并绑定 grad --------
        self.bucket_buffers = {}

        for bucket in self.buckets:
            bucket_id = id(bucket)

            total_numel = sum(p.numel() for p in bucket)
            buffer = torch.zeros(total_numel, device=device, dtype=torch.float32)

            self.bucket_buffers[bucket_id] = buffer
            self.bucket_ready_counts[bucket_id] = 0

            # 把每个 p.grad 绑定到 buffer 的一段
            offset = 0
            for p in bucket:
                numel = p.numel()
                view = buffer[offset:offset + numel].view_as(p)

                # 👇 关键：grad 直接指向 buffer
                p.grad = view

                offset += numel

        # -------- 注册 hook --------
        for bucket in self.buckets:
            for p in bucket:
                p.register_hook(self._make_hook(bucket))

    def forward(self, *inputs, **kwargs):
        x,_ = inputs
        return self.model(x)

    def _make_hook(self, bucket):
        bucket_id = id(bucket)

        def hook(grad):
            self.bucket_ready_counts[bucket_id] += 1

            if self.bucket_ready_counts[bucket_id] == len(bucket):
                buffer = self.bucket_buffers[bucket_id]

                handle = dist.all_reduce(
                    buffer,
                    op=dist.ReduceOp.SUM,
                    async_op=True
                )
                self.handles.append((handle, bucket_id))

            return grad  # 必须返回

        return hook

    def finish_gradient_synchronization(self):
        for handle, bucket_id in self.handles:
            handle.wait()

            buffer = self.bucket_buffers[bucket_id]
            buffer.div_(self.world_size)

        self.handles.clear()

        # 重置 ready 计数
        for k in self.bucket_ready_counts:
            self.bucket_ready_counts[k] = 0



class Optimizer_shard():
    def __init__(self, params, optimizer_cls: torch.optim.Optimizer, world_size, rank, device):
        self.param_idx = 0
        self.optimizer = optimizer_cls()
        self.add_param_group(params)
        self.rank = rank
        self.world_size = world_size
    def step(self, params):
        self.optimizer.step()
        i =0
        for param in params:
            dist.broadcast(param, i%self.world_size)

    def add_param_group(self, param_group):
        for param in param_group:
            if self.param_idx % self.world_size == 0:
                self.optimizer.add_param_group(param)
            self.param_idx += 1


def check_model_equal(model, rank):
    # 把整个模型的所有参数拼成一个大张量
    all_params = []
    for p in model.parameters():
        all_params.append(p.data.flatten())
    full_tensor = torch.cat(all_params)

    # 计算总和（只要总和一样，参数就一样）
    total = full_tensor.sum().item()
    print(f"[{rank}] 模型参数总和 = {total:.4f}")
def init(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(0)  # 全部用 3060，不会报错

def broadcast_model(ddp):
    for param in ddp.model.parameters():
        dist.broadcast(param.data, src = 0)

def train_optimal(rank, world_size, bucket_size, steps, device):
    torch.manual_seed(42)

    # ddp = DDP_Bucketed(ToyModel(10, 10), bucket_size ,device, world_size,)
    ddp = DDP(ToyModel(10, 10),device,)
    broadcast_model(ddp)

    model = ddp.model
    adamw = AdamW(model.parameters())
    start = timeit.default_timer()
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
        logits = model(input)
        logits = logits.view(-1, logits.size(-1))
        target = target.view(-1)

        loss = cross_entropy(logits, target)

        adamw.zero_grad()
        loss.backward()
        ddp.finish_gradient_synchronization()
        adamw.step()
        torch.cuda.synchronize()
    end = timeit.default_timer()
    print(end - start)


def run_optimal(rank, world_size, bucket_size, steps, device):
    init(rank, world_size)
    train_optimal(rank, world_size, bucket_size,steps, device)
    dist.destroy_process_group()

def data_parallel_optimal(world_size, bucket_size,steps, device):
    mp.set_start_method("spawn")
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=run_optimal, args=(rank, world_size, bucket_size, steps, device))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    steps = 100
    data_parallel_optimal(4, 0.001,steps, device)