import torch
import math
from einops import rearrange, einsum
# class Flashattention2_python(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, Q, K, V,is_casual:bool = False):
#
#         Br = 16
#         Bc = 16
#         Tr = Q.shape[0] // Br
#         Tc = K.shape[0] // Bc
#         Q = rearrange(Q, '... d -> (...) d')
#         K = rearrange(K, '... d -> (...) d')
#         V = rearrange(V, '... d -> (...) d')
#         O = torch.empty((Q.shape[0], Q.shape[1]))
#         L = torch.empty((Q.shape[0], Q.shape[1]))
#         for i in range(Tr):
#             Qi = Q[i * Br:(i + 1) * Br, :]
#             mi_before = torch.full((Br,),-float("inf"))
#             mi = torch.full((Br,),-float("inf"))
#             Oi = torch.zeros(Br,Q.shape[-1])
#             li= torch.zeros(Br)
#             for j in range(Tc):
#                 Kj = K[j*Bc:(j+1)*Bc,:]
#                 Vj = V[j*Bc:(j+1)*Bc,:]
#                 Sij = Qi @ Kj.T
#                 mi = torch.maximum(Sij.max(dim=-1).values, mi_before)
#                 li = torch.exp(mi_before - mi)*li+ torch.exp(Sij-mi.unsqueeze(-1)).sum(dim=-1)
#                 Pij = torch.exp(Sij-mi.unsqueeze(-1))
#                 Oi = torch.exp(mi_before-mi).unsqueeze(-1)*Oi + Pij @ Vj
#                 mi_before = mi
#             Oi = torch. reciprocal(li).unsqueeze(-1) * Oi
#             Li = mi + torch.log(li)
#             O[i*Br:(i+1)*Br, :] = Oi
#             L[i*Br:(i+1)*Br, :] = Li.unsqueeze(-1).expand(-1, Q.shape[1])
#         ctx.save_for_backward(L, Q, K, V, O)
#         return O

class Flashattention2_python(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Br = 16
        Bc = 16
        B, N, d = Q.shape
        Tr = N // Br
        Tc = K.shape[1] // Bc

        O = torch.empty_like(Q)
        L = torch.empty((B,N))

        # 缩放因子
        scale = 1.0 / math.sqrt(d)
        for b in range(B):
            Qb = Q[b]
            Kb = K[b]
            Vb = V[b]
            for i in range(Tr):
                start_i = i * Br
                end_i = start_i + Br
                Qi = Qb[start_i:end_i, :]
                # 初始化
                mi_before = torch.full((Br,), -float('inf'), device=Q.device)
                mi = torch.full((Br,), -float('inf'), device=Q.device)
                li = torch.zeros(Br, device=Q.device)
                Oi = torch.zeros_like(Qi)

                for j in range(Tc):
                    Kj = Kb[j * Bc:(j + 1) * Bc, :]
                    Vj = Vb[j * Bc:(j + 1) * Bc, :]

                    Sij = (Qi @ Kj.T) * scale

                    mi = torch.maximum(Sij.max(dim=-1).values, mi_before)
                    li = torch.exp(mi_before - mi) * li + torch.exp(Sij - mi.unsqueeze(-1)).sum(dim=-1)
                    Pij = torch.exp(Sij - mi.unsqueeze(-1))
                    Oi = torch.exp(mi_before - mi).unsqueeze(-1) * Oi + Pij @ Vj
                    mi_before = mi

                Oi = (1.0 / li).unsqueeze(-1) * Oi
                Li = mi + torch.log(li)

                O[b][start_i:end_i] = Oi
                L[b][start_i:end_i] = Li
                print(L.shape)
        ctx.save_for_backward(L, Q, K, V, O)
        return O
