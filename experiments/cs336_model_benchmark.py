import torch
import timeit
import hydra
from cs336_basics.model import BasicsTransformerLM,CausalSelfAttention,RotaryEmbedding
from cs336_basics.optimizer import get_cosine_lr,AdamW
from cs336_basics.nn_utils import cross_entropy,clip_gradient,softmax
import timeit
import torch
import torch.nn.functional as F
from cs336_basics.model import BasicsTransformerLM
import cs336_basics.model
import hydra
from omegaconf import DictConfig
import torch.cuda.nvtx as nvtx
from einops import rearrange, einsum
from torch import Tensor
from jaxtyping import Float, Bool, Int
import math

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))
    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension
    with nvtx.range("final matmul"):
        result = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    return result

cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
# ---------------------------
# 模型生成
# ---------------------------
def generate_model(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
):
    return BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta
    )

# ---------------------------
# 随机批次生成（支持指定 device）
# ---------------------------
def generate_random_batch(batch_size: int, seq_len: int, vocab_size: int, device):
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
        device=device  # 用传入的 device，不写死
    )
    targets = input_ids.clone()
    return input_ids, targets

def generate_random_batch_embed(batch_size: int, seq_len: int, vocab_size: int, d_model: int,device):
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len, d_model),
        dtype=torch.float32,
        device=device  # 用传入的 device，不写死
    )
    targets = input_ids.clone()
    return input_ids, targets

# ---------------------------
# Benchmark 类
# ---------------------------
class Bench:
    def __init__(
        self,
        model: BasicsTransformerLM,
        data_param: dict,
        warmup_steps: int,
        adamw_param: dict,
        device
    ):
        self.device = device
        self.model = model.to(device)
        self.data_param = data_param
        self.warmup_steps = warmup_steps
        self.adamw = AdamW(model.parameters(), **adamw_param)

    def test(self, n: int, backward: bool, use_mixed: bool):
        inputs = []
        targets = []

        # 生成数据时直接放在正确设备
        for _ in range(n + self.warmup_steps):
            input, target = generate_random_batch(**self.data_param, device=self.device)
            inputs.append(input)
            targets.append(target)

        self.model.train()
        scaler = torch.amp.GradScaler('cuda',enabled=use_mixed)
        # Warmup
        for step in range(self.warmup_steps):
            input = inputs[step]
            target = targets[step]

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_mixed):
                logits = self.model(input)
                logits = logits.view(-1, logits.size(-1))
                target = target.view(-1)
                loss = cross_entropy(logits, target)

            self.adamw.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(self.adamw)
            scaler.update()


        # 正式测试
        start = timeit.default_timer()

        for step in range(self.warmup_steps, self.warmup_steps + n):
            input = inputs[step]
            target = targets[step]
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_mixed):
                logits = self.model(input)
                if backward:
                    logits = logits.view(-1, logits.size(-1))
                    target = target.view(-1)
                    loss = cross_entropy(logits, target)
            if backward:
                self.adamw.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.adamw)
                scaler.update()


            # ✅ 题目要求：每个 step 后同步，但只有 CUDA 才调用
            if self.device.type == "cuda":
                torch.cuda.synchronize()

        end = timeit.default_timer()
        elapsed_time = (end - start) / n
        return elapsed_time

class attention_bench():
    def __init__(self,d_model:int, seq_length: int, device):
        self.d_model = d_model
        self.seq_length = seq_length
        self.Positional_encoder = RotaryEmbedding(seq_length, d_model)
        self.model = CausalSelfAttention(d_model,self.Positional_encoder).to(device)
        self.model_compiled = torch.compile(self.model)
        self.device = device
    def test(self, n:int):
        inputs = []
        targets = []
        warmup_steps = 5
        # 生成数据时直接放在正确设备

        for _ in range(n + warmup_steps):
            input, target = generate_random_batch_embed(batch_size=8, seq_len= self.seq_length,vocab_size=10000, d_model=self.d_model,device=self.device)
            inputs.append(input)
            targets.append(target)

        for i in range(warmup_steps):
            input = inputs[i]

            output = self.model(input)
            torch.cuda.synchronize()
        start_time1 = timeit.default_timer()
        for i in range(warmup_steps,warmup_steps + n):
            input = inputs[i]
            output = self.model(input)
            torch.cuda.synchronize()
        forward_time = (timeit.default_timer() - start_time1)/n
        mem = torch.cuda.max_memory_allocated() / 1e9
        start_time2 = timeit.default_timer()
        for i in range(warmup_steps,warmup_steps + n):
            input = inputs[i]
            loss = self.model(input).sum()
            loss.backward()
            torch.cuda.synchronize()
        backward_time = (timeit.default_timer()-start_time2)/n
        print(f"d_model:{self.d_model},seq_length:{self.seq_length}")
        print(f"Forward: {forward_time:.4f}s")
        print(f"Backward: {backward_time:.4f}s")
        print(f"显存: {mem:.2f}GB")

    def test_compiled(self, n: int):
        inputs = []
        targets = []
        warmup_steps = 5
        # 生成数据时直接放在正确设备

        for _ in range(n + warmup_steps):
            input, target = generate_random_batch_embed(batch_size=8, seq_len=self.seq_length, vocab_size=10000,
                                                        d_model=self.d_model, device=self.device)
            inputs.append(input)
            targets.append(target)

        for i in range(warmup_steps):
            input = inputs[i]

            output = self.model_compiled(input)
            torch.cuda.synchronize()
        start_time1 = timeit.default_timer()
        for i in range(warmup_steps, warmup_steps + n):
            input = inputs[i]
            output = self.model_compiled(input)
            torch.cuda.synchronize()
        forward_time = (timeit.default_timer() - start_time1) / n
        mem = torch.cuda.max_memory_allocated() / 1e9
        start_time2 = timeit.default_timer()
        for i in range(warmup_steps, warmup_steps + n):
            input = inputs[i]
            loss = self.model_compiled(input).sum()
            loss.backward()
            torch.cuda.synchronize()
        backward_time = (timeit.default_timer() - start_time2) / n
        print(f"d_model:{self.d_model},seq_length:{self.seq_length}")
        print(f"Forward: {forward_time:.4f}s")
        print(f"Backward: {backward_time:.4f}s")
        print(f"显存: {mem:.2f}GB")
# ---------------------------
# Hydra 入口
# ---------------------------
@hydra.main(config_path=r"C:\Users\97488\Desktop\assignment2-systems-main\assignment2-systems-main\conf", config_name="benchconfig", version_base=None)
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # model = generate_model(**cfg.model)
    # bench = Bench(
    #     model=model,
    #     data_param=cfg.data,
    #     warmup_steps=5,
    #     adamw_param=cfg.optimizer,
    #     device=device
    # )
    #
    # forward_time = bench.test(n=10, backward=True, use_mixed=True)
    # print(f"Forward time per step: {forward_time:.6f} s")
    attn_bench = attention_bench(128,4096,device)
    attn_bench.test(100)
    attn_bench.test_compiled(100)
if __name__ == "__main__":
    main()
    #print("PyTorch 版本:", torch.__version__)