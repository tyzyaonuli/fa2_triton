import triton
import triton.language as tl
import triton.testing as tt
import  torch
import math
@triton.jit
def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,  # 输入矩阵指针
        O_ptr, L_ptr,  # 输出矩阵指针
        # 各张量步长参数
        stride_qb, stride_qq, stride_qd,  # Q的批次/查询/特征维度步长
        stride_kb, stride_kk, stride_kd,  # K的批次/键/特征维度步长
        stride_vb, stride_vk, stride_vd,  # V的批次/键/特征维度步长
        stride_ob, stride_oq, stride_od,  # O的批次/查询/特征维度步长
        stride_lb, stride_lq,  # L的批次/查询维度步长
        N_QUERIES, N_KEYS,  # 查询数和键值数
        scale,  # 缩放因子1/sqrt(d)
        D: tl.constexpr,  # 特征维度（编译期常量）
        Q_TILE_SIZE: tl.constexpr,  # 查询分块尺寸B_q
        K_TILE_SIZE: tl.constexpr,  # 键分块尺寸B_k
        is_causal: tl.constexpr   # 是否进行掩码
):
    # 获取程序索引
    query_tile_index = tl.program_id(0)  # 查询区块索引
    batch_index = tl.program_id(1)  # 批次索引

    # 根据批次偏移量调整指针
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,  # 批次偏移后的指针
        shape=(N_QUERIES, D),  # 矩阵整体形状
        strides=(stride_qq, stride_qd),  # 行/列步长
        offsets=(query_tile_index * Q_TILE_SIZE, 0),  # 当前区块偏移
        block_shape=(Q_TILE_SIZE, D),  # 区块尺寸
        order=(1, 0)  # 内存布局顺序（列优先）
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,  # 批次偏移后的指针
        shape=(N_KEYS, D),  # 矩阵整体形状
        strides=(stride_kk, stride_kd),  # 行/列步长
        offsets=(0, 0),  # 当前区块偏移
        block_shape=(K_TILE_SIZE, D),  # 区块尺寸
        order=(1, 0)  # 内存布局顺序（列优先）
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,  # 批次偏移后的指针
        shape=(N_KEYS, D),  # 矩阵整体形状
        strides=(stride_vk, stride_vd),  # 行/列步长
        offsets=(0, 0),  # 当前区块偏移
        block_shape=(K_TILE_SIZE, D),  # 区块尺寸
        order=(1, 0)  # 内存布局顺序（列优先）
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,  # 批次偏移后的指针
        shape=(N_QUERIES, D),  # 矩阵整体形状
        strides=(stride_oq, stride_od),  # 行/列步长
        offsets=(query_tile_index * Q_TILE_SIZE, 0),  # 当前区块偏移
        block_shape=(Q_TILE_SIZE, D),  # 区块尺寸
        order=(1, 0)  # 内存布局顺序（列优先）
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,  # 批次偏移后的指针
        shape=(N_QUERIES,),  # 矩阵整体形状
        strides=(stride_lq,),  # 行/列步长
        offsets=(query_tile_index * Q_TILE_SIZE, ),  # 当前区块偏移
        block_shape=(Q_TILE_SIZE, ),  # 区块尺寸
        order=(0,)  # 内存布局顺序（列优先）
    )
    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    mi = tl.full((Q_TILE_SIZE,), value=-float('inf'), dtype=tl.float32)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    q_start = query_tile_index * Q_TILE_SIZE
    q_max = q_start + Q_TILE_SIZE - 1
    if is_causal:
        max_k_tile = (q_max // K_TILE_SIZE) + 1
    else:
        max_k_tile = tl.cdiv(N_KEYS, K_TILE_SIZE)

    for j in range(max_k_tile):
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Sij = tl.dot(Qi, tl.trans(Kj))*scale

        k_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        k_mask = k_offsets[None, :] < N_KEYS
        if is_causal:
            causal_mask = q_offsets[:, None] >= k_offsets[None, :]
            mask = k_mask & causal_mask
        else:
            mask = k_mask
        Sij = tl.where(mask, Sij, -float('inf'))

        mi_new = tl.maximum(mi, tl.max(Sij, axis = 1))
        alpha = tl.exp(mi-mi_new)
        beta = tl.exp(Sij - mi_new[:, None])
        li = alpha * li + tl.sum(beta, axis = 1)
        Oi = tl.dot(beta.to(Vj.dtype), Vj) + Oi * alpha[:, None]
        mi = mi_new

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    Oi = Oi/li[:, None]
    Li = mi + tl.log(li)

    tl.store(O_block_ptr, Oi.to(Qi.dtype), boundary_check=(0, 1))
    tl.store(L_block_ptr, Li, boundary_check=(0,))

class Flashattention2_triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Batch_size, N_QUERIES, D = Q.shape
        N_KEYS = K.shape[1]
        ctx.Q_TILE_SIZE = 32
        ctx.K_TILE_SIZE = 32
        O = torch.empty(Q.shape, dtype=Q.dtype, device=Q.device)
        L = torch.empty((Batch_size, N_QUERIES,), dtype=torch.float32, device=Q.device)
        scale = 1.0 / math.sqrt(D)

        flash_fwd_kernel[(math.ceil(N_QUERIES/ctx.Q_TILE_SIZE), Batch_size, )](
        Q, K, V,
        O, L,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        L.stride(0), L.stride(1),
        N_QUERIES, N_KEYS,
        scale,
        D,
        ctx.Q_TILE_SIZE,
        ctx.K_TILE_SIZE,
        is_causal
        )
        ctx.is_causal = is_causal
        ctx.save_for_backward(Q, K, V, L, O)

        return O
    @staticmethod
    def backward(ctx, GRAD_O):
        Q,K,V,L,O = ctx.saved_tensors
        Batch_size, N_QUERIES, D = Q.shape
        N_KEYS = K.shape[1]
        Q_TILE_SIZE = ctx.Q_TILE_SIZE
        K_TILE_SIZE = ctx.K_TILE_SIZE
        GRAD_Q = torch.zeros(Q.shape, dtype=Q.dtype, device=Q.device)
        GRAD_K = torch.zeros(K.shape, dtype=K.dtype, device=K.device)
        GRAD_V = torch.zeros(V.shape, dtype=V.dtype, device=V.device)
        scale = 1.0 / math.sqrt(D)
        is_causal = ctx.is_causal

        flash_bwd_KV_kernel[(math.ceil(N_KEYS/K_TILE_SIZE), Batch_size,)](
            Q, K, V, O, L,GRAD_O,
            GRAD_K, GRAD_V,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            GRAD_O.stride(0), GRAD_O.stride(1), GRAD_O.stride(2),
            GRAD_K.stride(0), GRAD_K.stride(1), GRAD_K.stride(2),
            GRAD_V.stride(0), GRAD_V.stride(1), GRAD_V.stride(2),
            scale,
            N_QUERIES, N_KEYS,
            D,
            Q_TILE_SIZE,
            K_TILE_SIZE,
            is_causal,
        )
        flash_bwd_Q_kernel[(math.ceil(N_QUERIES / Q_TILE_SIZE), Batch_size,)](
            Q, K, V, O, L, GRAD_O,
            GRAD_Q,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            GRAD_O.stride(0), GRAD_O.stride(1), GRAD_O.stride(2),
            GRAD_Q.stride(0), GRAD_Q.stride(1), GRAD_Q.stride(2),
            scale,
            N_QUERIES, N_KEYS,
            D,
            Q_TILE_SIZE,
            K_TILE_SIZE,
            is_causal,
        )
        return GRAD_Q, GRAD_K, GRAD_V, None

class Flashattention2_triton_2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Batch_size, N_QUERIES, D = Q.shape
        N_KEYS = K.shape[1]
        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16
        O = torch.empty(Q.shape, dtype=Q.dtype, device=Q.device)
        L = torch.empty((Batch_size, N_QUERIES,), dtype=torch.float32, device=Q.device)
        scale = 1.0 / math.sqrt(D)

        flash_fwd_kernel[(math.ceil(N_QUERIES/ctx.Q_TILE_SIZE), Batch_size, )](
        Q, K, V,
        O, L,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        L.stride(0), L.stride(1),
        N_QUERIES, N_KEYS,
        scale,
        D,
        ctx.Q_TILE_SIZE,
        ctx.K_TILE_SIZE,
        is_causal
        )
        ctx.is_causal = is_causal
        ctx.save_for_backward(Q, K, V, L, O)

        return O
    @staticmethod
    def backward(ctx, GRAD_O):
        Q,K,V,L,O = ctx.saved_tensors
        Batch_size, N_QUERIES, D = Q.shape
        N_KEYS = K.shape[1]
        Q_TILE_SIZE = ctx.Q_TILE_SIZE
        K_TILE_SIZE = ctx.K_TILE_SIZE
        GRAD_Q = torch.zeros(Q.shape, dtype=Q.dtype, device=Q.device)
        GRAD_K = torch.zeros(K.shape, dtype=K.dtype, device=K.device)
        GRAD_V = torch.zeros(V.shape, dtype=V.dtype, device=V.device)
        scale = 1.0 / math.sqrt(D)
        is_causal = ctx.is_causal

        flash_bwd_KV_kernel_2[(math.ceil(N_KEYS/K_TILE_SIZE), Batch_size,)](
            Q, K, V, O, L,GRAD_O,
            GRAD_K, GRAD_V,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            GRAD_O.stride(0), GRAD_O.stride(1), GRAD_O.stride(2),
            GRAD_K.stride(0), GRAD_K.stride(1), GRAD_K.stride(2),
            GRAD_V.stride(0), GRAD_V.stride(1), GRAD_V.stride(2),
            scale,
            N_QUERIES, N_KEYS,
            D,
            Q_TILE_SIZE,
            K_TILE_SIZE,
            is_causal,
        )
        flash_bwd_Q_kernel_2[(math.ceil(N_QUERIES / Q_TILE_SIZE), Batch_size,)](
            Q, K, V, O, L, GRAD_O,
            GRAD_Q,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            GRAD_O.stride(0), GRAD_O.stride(1), GRAD_O.stride(2),
            GRAD_Q.stride(0), GRAD_Q.stride(1), GRAD_Q.stride(2),
            scale,
            N_QUERIES, N_KEYS,
            D,
            Q_TILE_SIZE,
            K_TILE_SIZE,
            is_causal,
        )
        return GRAD_Q, GRAD_K, GRAD_V, None

@triton.jit
def flash_bwd_KV_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, grad_O_ptr,# 输入矩阵指针
        grad_K_ptr, grad_V_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        stride_gob, stride_goq, stride_god,
        stride_gkb, stride_gkk, stride_gkd,
        stride_gvb, stride_gvk, stride_gvd,
        scale,
        N_QUERIES, N_KEYS,  # 查询数和键值数
        D: tl.constexpr,  # 特征维度（编译期常量）
        Q_TILE_SIZE: tl.constexpr,  # 查询分块尺寸B_q
        K_TILE_SIZE: tl.constexpr,  # 键分块尺寸B_k
        is_causal:tl.constexpr ,
):
    # 获取程序索引
    key_tile_index = tl.program_id(0)  # 查询区块索引
    batch_index = tl.program_id(1)  # 批次索引


    GRAD_K_block_ptr = tl.make_block_ptr(
        grad_K_ptr + batch_index * stride_gkb,  # 批次偏移后的指针
        shape=(N_KEYS, D),  # 矩阵整体形状
        strides=(stride_gkk, stride_gkd),  # 行/列步长
        offsets=(key_tile_index * K_TILE_SIZE, 0),  # 当前区块偏移
        block_shape=(K_TILE_SIZE, D),  # 区块尺寸
        order=(1, 0)  # 内存布局顺序（列优先）
    )
    GRAD_V_block_ptr = tl.make_block_ptr(
        grad_V_ptr + batch_index * stride_gvb,  # 批次偏移后的指针
        shape=(N_KEYS, D),  # 矩阵整体形状
        strides=(stride_gvk, stride_gvd),  # 行/列步长
        offsets=(key_tile_index * K_TILE_SIZE, 0),  # 当前区块偏移
        block_shape=(K_TILE_SIZE, D),  # 区块尺寸
        order=(1, 0)  # 内存布局顺序（列优先）
    )

    Q_ptr =  Q_ptr + batch_index * stride_qb
    O_ptr = O_ptr + batch_index * stride_ob
    L_ptr = L_ptr + batch_index * stride_lb
    grad_O_ptr = grad_O_ptr + batch_index * stride_gob

    k_start =  key_tile_index * K_TILE_SIZE
    offs_n = k_start + tl.arange(0, K_TILE_SIZE)
    offs_d = tl.arange(0, D)

    K_ptrs =  K_ptr + batch_index * stride_kb + offs_n[:, None] * stride_kk + offs_d[None, :] * stride_kd
    V_ptrs =  V_ptr + batch_index * stride_vb + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vd
    Kj = tl.load(K_ptrs, mask = offs_n[:, None]<N_KEYS, other=0.)
    Vj = tl.load(V_ptrs, mask = offs_n[:, None]<N_KEYS, other=0.)

    dKj = tl.zeros((K_TILE_SIZE, D), dtype=Kj.dtype)
    dVj = tl.zeros((K_TILE_SIZE, D), dtype=Vj.dtype)



    if is_causal:
        start_q_tile = (k_start + Q_TILE_SIZE - 1) // Q_TILE_SIZE  # ceil除法
        num_q_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
    else:
        start_q_tile = 0
        num_q_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)

    offs_m = start_q_tile * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    Q_ptrs = Q_ptr + offs_m[:, None] * stride_qq + offs_d[None, :] * stride_qd
    O_ptrs = O_ptr + offs_m[:, None] * stride_oq + offs_d[None, :] * stride_od
    dO_ptrs = grad_O_ptr + offs_m[:, None] * stride_goq + offs_d[None, :] * stride_god
    L_ptrs = L_ptr + offs_m * stride_lq

    Qi = tl.load(Q_ptrs, mask=offs_m[:, None] < N_QUERIES, other=0.)
    Oi = tl.load(O_ptrs, mask=offs_m[:, None] < N_QUERIES, other=0.)
    dOi = tl.load(dO_ptrs, mask=offs_m[:, None] < N_QUERIES, other=0.)
    Li = tl.load(L_ptrs, mask=offs_m < N_QUERIES, other=0.)

    Sij = tl.dot(Qi, tl.trans(Kj)) * scale

    q_offsets = start_q_tile * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    k_offsets = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)

    if is_causal:
        mask = q_offsets[:, None] >= k_offsets[None, :]
        Sij = tl.where(mask, Sij, -float('inf'))
    beta = tl.exp(Sij - Li[:, None])

    dVj += tl.dot(tl.trans(beta.to(dOi.dtype)), dOi)
    dbeta = tl.dot(dOi, tl.trans(Vj))
    Di = tl.sum(dOi * Oi, axis=1)
    dSij = beta * (dbeta - Di[:, None]) * scale
    dKj += tl.dot(tl.trans(dSij.to(Qi.dtype)), Qi)

    for i  in range(start_q_tile+1, num_q_tiles):
        offs_m = i*Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        Q_ptrs = Q_ptr + offs_m[:, None] * stride_qq + offs_d[None, :] * stride_qd
        O_ptrs = O_ptr + offs_m[:, None] * stride_oq + offs_d[None, :] * stride_od
        dO_ptrs = grad_O_ptr + offs_m[:, None] * stride_goq+ offs_d[None, :] * stride_god
        L_ptrs = L_ptr + offs_m * stride_lq

        Qi = tl.load(Q_ptrs, mask= offs_m[:,None]< N_QUERIES, other=0.)
        Oi = tl.load(O_ptrs, mask= offs_m[:,None]< N_QUERIES, other=0.)
        dOi = tl.load(dO_ptrs, mask= offs_m[:,None]< N_QUERIES, other=0.)
        Li = tl.load(L_ptrs, mask= offs_m< N_QUERIES, other=0.)

        Sij = tl.dot(Qi, tl.trans(Kj)) * scale

        # q_offsets = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        # k_offsets = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        # if is_causal:
        #     mask = q_offsets[:, None] >= k_offsets[None, :]
        #     Sij = tl.where(mask, Sij, -float('inf'))
        beta = tl.exp(Sij - Li[:, None])


        dVj += tl.dot(tl.trans(beta.to(dOi.dtype)), dOi)
        dbeta = tl.dot(dOi, tl.trans(Vj))
        Di = tl.sum(dOi * Oi, axis = 1)
        dSij = beta*(dbeta - Di[:, None])*scale
        dKj += tl.dot(tl.trans(dSij.to(Qi.dtype)), Qi)



    tl.store(GRAD_K_block_ptr, dKj.to(Kj.dtype), boundary_check = (0, 1), )
    tl.store(GRAD_V_block_ptr, dVj.to(Vj.dtype), boundary_check = (0, 1), )
@triton.jit
def flash_bwd_KV_kernel_2(
        Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, grad_O_ptr,# 输入矩阵指针
        grad_K_ptr, grad_V_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        stride_gob, stride_goq, stride_god,
        stride_gkb, stride_gkk, stride_gkd,
        stride_gvb, stride_gvk, stride_gvd,
        scale,
        N_QUERIES, N_KEYS,  # 查询数和键值数
        D: tl.constexpr,  # 特征维度（编译期常量）
        Q_TILE_SIZE: tl.constexpr,  # 查询分块尺寸B_q
        K_TILE_SIZE: tl.constexpr,  # 键分块尺寸B_k
        is_causal:tl.constexpr ,
):
    # 获取程序索引
    key_tile_index = tl.program_id(0)  # 查询区块索引
    batch_index = tl.program_id(1)  # 批次索引


    GRAD_K_block_ptr = tl.make_block_ptr(
        grad_K_ptr + batch_index * stride_gkb,  # 批次偏移后的指针
        shape=(N_KEYS, D),  # 矩阵整体形状
        strides=(stride_gkk, stride_gkd),  # 行/列步长
        offsets=(key_tile_index * K_TILE_SIZE, 0),  # 当前区块偏移
        block_shape=(K_TILE_SIZE, D),  # 区块尺寸
        order=(1, 0)  # 内存布局顺序（列优先）
    )
    GRAD_V_block_ptr = tl.make_block_ptr(
        grad_V_ptr + batch_index * stride_gvb,  # 批次偏移后的指针
        shape=(N_KEYS, D),  # 矩阵整体形状
        strides=(stride_gvk, stride_gvd),  # 行/列步长
        offsets=(key_tile_index * K_TILE_SIZE, 0),  # 当前区块偏移
        block_shape=(K_TILE_SIZE, D),  # 区块尺寸
        order=(1, 0)  # 内存布局顺序（列优先）
    )

    Q_ptr =  Q_ptr + batch_index * stride_qb
    O_ptr = O_ptr + batch_index * stride_ob
    L_ptr = L_ptr + batch_index * stride_lb
    grad_O_ptr = grad_O_ptr + batch_index * stride_gob

    k_start =  key_tile_index * K_TILE_SIZE
    offs_n = k_start + tl.arange(0, K_TILE_SIZE)
    offs_d = tl.arange(0, D)

    K_ptrs =  K_ptr + batch_index * stride_kb + offs_n[:, None] * stride_kk + offs_d[None, :] * stride_kd
    V_ptrs =  V_ptr + batch_index * stride_vb + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vd
    Kj = tl.load(K_ptrs, mask = offs_n[:, None]<N_KEYS, other=0.)
    Vj = tl.load(V_ptrs, mask = offs_n[:, None]<N_KEYS, other=0.)

    dKj = tl.zeros((K_TILE_SIZE, D), dtype=Kj.dtype)
    dVj = tl.zeros((K_TILE_SIZE, D), dtype=Vj.dtype)



    if is_causal:
        start_q_tile = (k_start + Q_TILE_SIZE - 1) // Q_TILE_SIZE  # ceil除法
        num_q_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
    else:
        start_q_tile = 0
        num_q_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)



    for i  in range(start_q_tile, num_q_tiles):
        offs_m = i*Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        Q_ptrs = Q_ptr + offs_m[:, None] * stride_qq + offs_d[None, :] * stride_qd
        O_ptrs = O_ptr + offs_m[:, None] * stride_oq + offs_d[None, :] * stride_od
        dO_ptrs = grad_O_ptr + offs_m[:, None] * stride_goq+ offs_d[None, :] * stride_god
        L_ptrs = L_ptr + offs_m * stride_lq

        Qi = tl.load(Q_ptrs, mask= offs_m[:,None]< N_QUERIES, other=0.)
        Oi = tl.load(O_ptrs, mask= offs_m[:,None]< N_QUERIES, other=0.)
        dOi = tl.load(dO_ptrs, mask= offs_m[:,None]< N_QUERIES, other=0.)
        Li = tl.load(L_ptrs, mask= offs_m< N_QUERIES, other=0.)

        Sij = tl.dot(Qi, tl.trans(Kj)) * scale

        q_offsets = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        k_offsets = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        if is_causal:
            mask = q_offsets[:, None] >= k_offsets[None, :]
            Sij = tl.where(mask, Sij, -float('inf'))
        beta = tl.exp(Sij - Li[:, None])


        dVj += tl.dot(tl.trans(beta.to(dOi.dtype)), dOi)
        dbeta = tl.dot(dOi, tl.trans(Vj))
        Di = tl.sum(dOi * Oi, axis = 1)
        dSij = beta*(dbeta - Di[:, None])*scale
        dKj += tl.dot(tl.trans(dSij.to(Qi.dtype)), Qi)



    tl.store(GRAD_K_block_ptr, dKj.to(Kj.dtype), boundary_check = (0, 1), )
    tl.store(GRAD_V_block_ptr, dVj.to(Vj.dtype), boundary_check = (0, 1), )
@triton.jit
def flash_bwd_Q_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, grad_O_ptr,# 输入矩阵指针
        grad_Q_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        stride_gob, stride_goq, stride_god,
        stride_gqb, stride_gqq , stride_gqd,
        scale,
        N_QUERIES, N_KEYS,  # 查询数和键值数
        D: tl.constexpr,  # 特征维度（编译期常量）
        Q_TILE_SIZE: tl.constexpr,  # 查询分块尺寸B_q
        K_TILE_SIZE: tl.constexpr,  # 键分块尺寸B_k
        is_causal:tl.constexpr ,
):

    query_tile_index = tl.program_id(0)  # 查询区块索引
    batch_index = tl.program_id(1)  # 批次索引

    K_ptr = K_ptr + batch_index * stride_kb
    V_ptr = V_ptr + batch_index * stride_vb


    offs_d = tl.arange(0, D)
    offs_m = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    Q_ptrs = Q_ptr + batch_index * stride_qb + offs_m[:, None] * stride_qq + offs_d[None, :] * stride_qd
    O_ptrs = O_ptr + batch_index * stride_ob + offs_m[:, None] * stride_oq + offs_d[None, :] * stride_od
    dO_ptrs = grad_O_ptr + batch_index * stride_gob + offs_m[:, None] * stride_goq + offs_d[None, :] * stride_god
    L_ptrs = L_ptr + batch_index * stride_lb + offs_m * stride_lq

    Qi = tl.load(Q_ptrs, mask=offs_m[:, None] < N_QUERIES, other=0.)
    Oi = tl.load(O_ptrs, mask=offs_m[:, None] < N_QUERIES, other=0.)
    dOi = tl.load(dO_ptrs, mask=offs_m[:, None] < N_QUERIES, other=0.)
    Li = tl.load(L_ptrs, mask=offs_m < N_QUERIES, other=0.)

    dQi = tl.zeros((Q_TILE_SIZE, D), dtype=Qi.dtype)

    q_start = query_tile_index * Q_TILE_SIZE
    q_max = q_start + Q_TILE_SIZE - 1
    if is_causal:
        max_k_tile = (q_max // K_TILE_SIZE) + 1
    else:
        max_k_tile = tl.cdiv(N_KEYS, K_TILE_SIZE)

    for j in range(max_k_tile-1):
        offs_k = j* K_TILE_SIZE+tl.arange(0, K_TILE_SIZE)
        K_ptrs = K_ptr + offs_k[:, None] * stride_kk + offs_d[None, :] * stride_kd
        V_ptrs = V_ptr + offs_k[:, None] * stride_vk + offs_d[None, :] * stride_vd


        Kj = tl.load(K_ptrs, )
        Vj = tl.load(V_ptrs, )

        Sij = tl.dot(Qi, tl.trans(Kj)) * scale

        beta = tl.exp(Sij - Li[:, None])

        dbeta = tl.dot(dOi, tl.trans(Vj))
        Di = tl.sum(dOi * Oi, axis=1)
        dSij = beta * (dbeta - Di[:, None]) * scale

        dQi += tl.dot(dSij.to(Kj.dtype), Kj)

    offs_k = (max_k_tile-1) * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
    K_ptrs = K_ptr + offs_k[:, None] * stride_kk + offs_d[None, :] * stride_kd
    V_ptrs = V_ptr + offs_k[:, None] * stride_vk + offs_d[None, :] * stride_vd

    Kj = tl.load(K_ptrs, mask=offs_k[:, None] < N_KEYS, other=0.)
    Vj = tl.load(V_ptrs, mask=offs_k[:, None] < N_KEYS, other=0.)

    Sij = tl.dot(Qi, tl.trans(Kj)) * scale

    q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    k_offsets = (max_k_tile-1) * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)

    if is_causal:
        mask = q_offsets[:, None] >= k_offsets[None, :]
        Sij = tl.where(mask, Sij, -float('inf'))
    beta = tl.exp(Sij - Li[:, None])

    dbeta = tl.dot(dOi, tl.trans(Vj))
    Di = tl.sum(dOi * Oi, axis=1)
    dSij = beta * (dbeta - Di[:, None]) * scale

    dQi += tl.dot(dSij.to(Kj.dtype), Kj)


    dQ_ptrs = grad_Q_ptr + batch_index * stride_gqb +offs_m[:, None] * stride_gqq + offs_d[None, :] * stride_gqd
    tl.store(dQ_ptrs, dQi.to(Qi.dtype), mask=offs_m[:, None]<N_QUERIES)

@triton.jit
def flash_bwd_Q_kernel_2(
        Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, grad_O_ptr,# 输入矩阵指针
        grad_Q_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        stride_gob, stride_goq, stride_god,
        stride_gqb, stride_gqq , stride_gqd,
        scale,
        N_QUERIES, N_KEYS,  # 查询数和键值数
        D: tl.constexpr,  # 特征维度（编译期常量）
        Q_TILE_SIZE: tl.constexpr,  # 查询分块尺寸B_q
        K_TILE_SIZE: tl.constexpr,  # 键分块尺寸B_k
        is_causal:tl.constexpr ,
):

    query_tile_index = tl.program_id(0)  # 查询区块索引
    batch_index = tl.program_id(1)  # 批次索引

    K_ptr = K_ptr + batch_index * stride_kb
    V_ptr = V_ptr + batch_index * stride_vb


    offs_d = tl.arange(0, D)
    offs_m = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    Q_ptrs = Q_ptr + batch_index * stride_qb + offs_m[:, None] * stride_qq + offs_d[None, :] * stride_qd
    O_ptrs = O_ptr + batch_index * stride_ob + offs_m[:, None] * stride_oq + offs_d[None, :] * stride_od
    dO_ptrs = grad_O_ptr + batch_index * stride_gob + offs_m[:, None] * stride_goq + offs_d[None, :] * stride_god
    L_ptrs = L_ptr + batch_index * stride_lb + offs_m * stride_lq

    Qi = tl.load(Q_ptrs, mask=offs_m[:, None] < N_QUERIES, other=0.)
    Oi = tl.load(O_ptrs, mask=offs_m[:, None] < N_QUERIES, other=0.)
    dOi = tl.load(dO_ptrs, mask=offs_m[:, None] < N_QUERIES, other=0.)
    Li = tl.load(L_ptrs, mask=offs_m < N_QUERIES, other=0.)

    dQi = tl.zeros((Q_TILE_SIZE, D), dtype=Qi.dtype)

    q_start = query_tile_index * Q_TILE_SIZE
    q_max = q_start + Q_TILE_SIZE - 1
    if is_causal:
        max_k_tile = (q_max // K_TILE_SIZE) + 1
    else:
        max_k_tile = tl.cdiv(N_KEYS, K_TILE_SIZE)

    for j in range(max_k_tile):
        offs_k = j* K_TILE_SIZE+tl.arange(0, K_TILE_SIZE)
        K_ptrs = K_ptr + offs_k[:, None] * stride_kk + offs_d[None, :] * stride_kd
        V_ptrs = V_ptr + offs_k[:, None] * stride_vk + offs_d[None, :] * stride_vd

        Kj = tl.load(K_ptrs, mask=offs_k[:, None] < N_KEYS, other=0.)
        Vj = tl.load(V_ptrs, mask=offs_k[:, None] < N_KEYS, other=0.)

        Sij = tl.dot(Qi, tl.trans(Kj)) * scale
        q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        k_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)  # 修复：必须使用当前循环中的 K tile

        if is_causal:
            mask = q_offsets[:, None] >= k_offsets[None, :]
            Sij = tl.where(mask, Sij, -float('inf'))
        beta = tl.exp(Sij - Li[:, None])

        dbeta = tl.dot(dOi, tl.trans(Vj))
        Di = tl.sum(dOi * Oi, axis=1)
        dSij = beta * (dbeta - Di[:, None]) * scale

        dQi += tl.dot(dSij.to(Kj.dtype), Kj)



    dQ_ptrs = grad_Q_ptr + batch_index * stride_gqb +offs_m[:, None] * stride_gqq + offs_d[None, :] * stride_gqd
    tl.store(dQ_ptrs, dQi.to(Qi.dtype), mask=offs_m[:, None]<N_QUERIES)

@triton.jit
def flash_bwd_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, grad_O_ptr,# 输入矩阵指针
        grad_Q_ptr, grad_K_ptr, grad_V_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        stride_gob, stride_goq, stride_god,
        stride_gqb, stride_gqq , stride_gqd,
        stride_gkb, stride_gkk, stride_gkd,
        stride_gvb, stride_gvk, stride_gvd,
        scale,
        N_QUERIES, N_KEYS,  # 查询数和键值数
        D: tl.constexpr,  # 特征维度（编译期常量）
        Q_TILE_SIZE: tl.constexpr,  # 查询分块尺寸B_q
        K_TILE_SIZE: tl.constexpr,  # 键分块尺寸B_k
        is_causal:tl.constexpr ,
):
    # 获取程序索引
    key_tile_index = tl.program_id(0)  # 查询区块索引
    batch_index = tl.program_id(1)  # 批次索引


    GRAD_K_block_ptr = tl.make_block_ptr(
        grad_K_ptr + batch_index * stride_gkb,  # 批次偏移后的指针
        shape=(N_KEYS, D),  # 矩阵整体形状
        strides=(stride_gkk, stride_gkd),  # 行/列步长
        offsets=(key_tile_index * K_TILE_SIZE, 0),  # 当前区块偏移
        block_shape=(K_TILE_SIZE, D),  # 区块尺寸
        order=(1, 0)  # 内存布局顺序（列优先）
    )
    GRAD_V_block_ptr = tl.make_block_ptr(
        grad_V_ptr + batch_index * stride_gvb,  # 批次偏移后的指针
        shape=(N_KEYS, D),  # 矩阵整体形状
        strides=(stride_gvk, stride_gvd),  # 行/列步长
        offsets=(key_tile_index * K_TILE_SIZE, 0),  # 当前区块偏移
        block_shape=(K_TILE_SIZE, D),  # 区块尺寸
        order=(1, 0)  # 内存布局顺序（列优先）
    )

    Q_ptr =  Q_ptr + batch_index * stride_qb
    O_ptr = O_ptr + batch_index * stride_ob
    L_ptr = L_ptr + batch_index * stride_lb
    grad_O_ptr = grad_O_ptr + batch_index * stride_gob

    k_start =  key_tile_index * K_TILE_SIZE
    offs_n = k_start + tl.arange(0, K_TILE_SIZE)
    offs_d = tl.arange(0, D)

    K_ptrs =  K_ptr + batch_index * stride_kb + offs_n[:, None] * stride_kk + offs_d[None, :] * stride_kd
    V_ptrs =  V_ptr + batch_index * stride_vb + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vd
    Kj = tl.load(K_ptrs, mask = offs_n[:, None]<N_KEYS, other=0.)
    Vj = tl.load(V_ptrs, mask = offs_n[:, None]<N_KEYS, other=0.)

    dKj = tl.zeros((K_TILE_SIZE, D), dtype=Kj.dtype)
    dVj = tl.zeros((K_TILE_SIZE, D), dtype=Vj.dtype)



    if is_causal:
        start_q_tile = (k_start + Q_TILE_SIZE - 1) // Q_TILE_SIZE  # ceil除法
        num_q_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
    else:
        start_q_tile = 0
        num_q_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)

    for i  in range(start_q_tile, num_q_tiles):
        offs_m = i*Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        Q_ptrs = Q_ptr + offs_m[:, None] * stride_qq + offs_d[None, :] * stride_qd
        O_ptrs = O_ptr + offs_m[:, None] * stride_oq + offs_d[None, :] * stride_od
        dO_ptrs = grad_O_ptr + offs_m[:, None] * stride_goq+ offs_d[None, :] * stride_god
        L_ptrs = L_ptr + offs_m * stride_lq

        Qi = tl.load(Q_ptrs, mask= offs_m[:,None]< N_QUERIES, other=0.)
        Oi = tl.load(O_ptrs, mask= offs_m[:,None]< N_QUERIES, other=0.)
        dOi = tl.load(dO_ptrs, mask= offs_m[:,None]< N_QUERIES, other=0.)
        Li = tl.load(L_ptrs, mask= offs_m< N_QUERIES, other=0.)

        Sij = tl.dot(Qi, tl.trans(Kj)) * scale

        q_offsets = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        k_offsets = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)

        if is_causal:
            mask = q_offsets[:, None] >= k_offsets[None, :]
            Sij = tl.where(mask, Sij, -float('inf'))
        beta = tl.exp(Sij - Li[:, None])


        dVj += tl.dot(tl.trans(beta.to(dOi.dtype)), dOi)
        dbeta = tl.dot(dOi, tl.trans(Vj))
        Di = tl.sum(dOi * Oi, axis = 1)
        dSij = beta*(dbeta - Di[:, None])*scale

        dQi_local = tl.dot(dSij.to(Kj.dtype), Kj)
        offs_q = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        mask = q_offsets[:, None] < N_QUERIES
        dq_ptrs = grad_Q_ptr + batch_index * stride_gqb \
                  + offs_q[:, None] * stride_gqq \
                  + offs_d[None, :] * stride_gqd
        tl.atomic_add(dq_ptrs, dQi_local, mask=mask)

        dKj += tl.dot(tl.trans(dSij.to(Qi.dtype)), Qi)



    tl.store(GRAD_K_block_ptr, dKj.to(Kj.dtype), boundary_check = (0, 1), )
    tl.store(GRAD_V_block_ptr, dVj.to(Vj.dtype), boundary_check = (0, 1), )


# ======================== 【新增】无 atomic 的融合 backward ========================
# 设计说明：
# 1. 先用一个很小的 preprocess kernel 计算 Delta_i = sum_d(O_i * dO_i)。
# 2. 主 backward kernel 只 launch 一次，但内部包含两个阶段：
#       Phase A: 每个 program 独占一个 K/V owner tile，扫描 Q tiles，写回 dK/dV。
#       Phase B: 每个 program 独占一个 Q owner tile，扫描 K/V tiles，写回 dQ。
# 3. 每个梯度 tile 都只有一个 writer，因此不需要 tl.atomic_add。
# 4. 为了适配资源较小的 GPU，默认 OWNER_TILE=32、INNER_TILE=16。
#    可以后续尝试 OWNER_TILE=64、INNER_TILE=16/32。

@triton.jit
def flash_bwd_delta_kernel(
        O_ptr, grad_O_ptr, Delta_ptr,
        stride_ob, stride_oq, stride_od,
        stride_gob, stride_goq, stride_god,
        stride_db, stride_dq,
        N_QUERIES,
        D: tl.constexpr,
        ROW_TILE: tl.constexpr,
):
    row_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    offs_q = row_tile_index * ROW_TILE + tl.arange(0, ROW_TILE)
    offs_d = tl.arange(0, D)
    mask_q = offs_q[:, None] < N_QUERIES

    Oi = tl.load(
        O_ptr
        + batch_index * stride_ob
        + offs_q[:, None] * stride_oq
        + offs_d[None, :] * stride_od,
        mask=mask_q,
        other=0.0,
    ).to(tl.float32)

    dOi = tl.load(
        grad_O_ptr
        + batch_index * stride_gob
        + offs_q[:, None] * stride_goq
        + offs_d[None, :] * stride_god,
        mask=mask_q,
        other=0.0,
    ).to(tl.float32)

    Delta_i = tl.sum(Oi * dOi, axis=1)

    tl.store(
        Delta_ptr
        + batch_index * stride_db
        + offs_q * stride_dq,
        Delta_i,
        mask=offs_q < N_QUERIES,
    )


@triton.jit
def flash_bwd_fused_no_atomic_kernel(
        Q_ptr, K_ptr, V_ptr, L_ptr, grad_O_ptr, Delta_ptr,
        grad_Q_ptr, grad_K_ptr, grad_V_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_lb, stride_lq,
        stride_gob, stride_goq, stride_god,
        stride_db, stride_dq,
        stride_gqb, stride_gqq, stride_gqd,
        stride_gkb, stride_gkk, stride_gkd,
        stride_gvb, stride_gvk, stride_gvd,
        scale,
        N_QUERIES, N_KEYS,
        D: tl.constexpr,
        OWNER_TILE: tl.constexpr,
        INNER_TILE: tl.constexpr,
        is_causal: tl.constexpr,
):
    # 一个 program 同时负责：
    #   - 一个 OWNER_TILE 大小的 K/V tile 的 dK、dV
    #   - 一个 OWNER_TILE 大小的 Q tile 的 dQ
    # 两类梯度都独占写回，不需要 atomic。
    owner_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    tl.static_assert(OWNER_TILE % INNER_TILE == 0)

    Q_base = Q_ptr + batch_index * stride_qb
    K_base = K_ptr + batch_index * stride_kb
    V_base = V_ptr + batch_index * stride_vb
    L_base = L_ptr + batch_index * stride_lb
    dO_base = grad_O_ptr + batch_index * stride_gob
    Delta_base = Delta_ptr + batch_index * stride_db

    dQ_base = grad_Q_ptr + batch_index * stride_gqb
    dK_base = grad_K_ptr + batch_index * stride_gkb
    dV_base = grad_V_ptr + batch_index * stride_gvb

    offs_d = tl.arange(0, D)
    owner_start = owner_tile_index * OWNER_TILE

    # ------------------------------------------------------------------
    # Phase A: 固定 K/V owner tile，扫描所有相关 Q tile，计算 dK 与 dV。
    # ------------------------------------------------------------------
    offs_n_owner = owner_start + tl.arange(0, OWNER_TILE)
    valid_k_owner = offs_n_owner[:, None] < N_KEYS

    Kj_owner = tl.load(
        K_base
        + offs_n_owner[:, None] * stride_kk
        + offs_d[None, :] * stride_kd,
        mask=valid_k_owner,
        other=0.0,
    )

    Vj_owner = tl.load(
        V_base
        + offs_n_owner[:, None] * stride_vk
        + offs_d[None, :] * stride_vd,
        mask=valid_k_owner,
        other=0.0,
    )

    dKj = tl.zeros((OWNER_TILE, D), dtype=tl.float32)
    dVj = tl.zeros((OWNER_TILE, D), dtype=tl.float32)

    if is_causal:
        # 对角区域：需要 causal mask。
        # 固定 K owner tile 后，只有与它相交的 OWNER_TILE / INNER_TILE 个 Q tile
        # 需要逐元素 mask；更下方的 Q tile 一定全部合法。
        for r in range(0, OWNER_TILE // INNER_TILE):
            start_m = owner_start + r * INNER_TILE
            offs_m = start_m + tl.arange(0, INNER_TILE)

            Qi = tl.load(
                Q_base + offs_m[:, None] * stride_qq + offs_d[None, :] * stride_qd,
                mask=offs_m[:, None] < N_QUERIES,
                other=0.0,
            )
            dOi = tl.load(
                dO_base + offs_m[:, None] * stride_goq + offs_d[None, :] * stride_god,
                mask=offs_m[:, None] < N_QUERIES,
                other=0.0,
            )
            Li = tl.load(
                L_base + offs_m * stride_lq,
                mask=offs_m < N_QUERIES,
                other=0.0,
            )
            Di = tl.load(
                Delta_base + offs_m * stride_dq,
                mask=offs_m < N_QUERIES,
                other=0.0,
            )

            Sij = tl.dot(Qi, tl.trans(Kj_owner)) * scale
            valid = (offs_m[:, None] < N_QUERIES) & (offs_n_owner[None, :] < N_KEYS)
            causal_mask = offs_m[:, None] >= offs_n_owner[None, :]
            Sij = tl.where(valid & causal_mask, Sij, -float('inf'))
            Pij = tl.exp(Sij - Li[:, None])

            dVj += tl.dot(tl.trans(Pij.to(dOi.dtype)), dOi)
            dPij = tl.dot(dOi, tl.trans(Vj_owner))
            dSij = Pij * (dPij - Di[:, None]) * scale
            dKj += tl.dot(tl.trans(dSij.to(Qi.dtype)), Qi)

        # 严格位于对角线下方的区域：不再需要 causal mask，只保留边界 mask。
        for start_m in range(owner_start + OWNER_TILE, N_QUERIES, INNER_TILE):
            offs_m = start_m + tl.arange(0, INNER_TILE)

            Qi = tl.load(
                Q_base + offs_m[:, None] * stride_qq + offs_d[None, :] * stride_qd,
                mask=offs_m[:, None] < N_QUERIES,
                other=0.0,
            )
            dOi = tl.load(
                dO_base + offs_m[:, None] * stride_goq + offs_d[None, :] * stride_god,
                mask=offs_m[:, None] < N_QUERIES,
                other=0.0,
            )
            Li = tl.load(
                L_base + offs_m * stride_lq,
                mask=offs_m < N_QUERIES,
                other=0.0,
            )
            Di = tl.load(
                Delta_base + offs_m * stride_dq,
                mask=offs_m < N_QUERIES,
                other=0.0,
            )

            Sij = tl.dot(Qi, tl.trans(Kj_owner)) * scale
            valid = (offs_m[:, None] < N_QUERIES) & (offs_n_owner[None, :] < N_KEYS)
            Sij = tl.where(valid, Sij, -float('inf'))
            Pij = tl.exp(Sij - Li[:, None])

            dVj += tl.dot(tl.trans(Pij.to(dOi.dtype)), dOi)
            dPij = tl.dot(dOi, tl.trans(Vj_owner))
            dSij = Pij * (dPij - Di[:, None]) * scale
            dKj += tl.dot(tl.trans(dSij.to(Qi.dtype)), Qi)
    else:
        for start_m in range(0, N_QUERIES, INNER_TILE):
            offs_m = start_m + tl.arange(0, INNER_TILE)

            Qi = tl.load(
                Q_base + offs_m[:, None] * stride_qq + offs_d[None, :] * stride_qd,
                mask=offs_m[:, None] < N_QUERIES,
                other=0.0,
            )
            dOi = tl.load(
                dO_base + offs_m[:, None] * stride_goq + offs_d[None, :] * stride_god,
                mask=offs_m[:, None] < N_QUERIES,
                other=0.0,
            )
            Li = tl.load(
                L_base + offs_m * stride_lq,
                mask=offs_m < N_QUERIES,
                other=0.0,
            )
            Di = tl.load(
                Delta_base + offs_m * stride_dq,
                mask=offs_m < N_QUERIES,
                other=0.0,
            )

            Sij = tl.dot(Qi, tl.trans(Kj_owner)) * scale
            valid = (offs_m[:, None] < N_QUERIES) & (offs_n_owner[None, :] < N_KEYS)
            Sij = tl.where(valid, Sij, -float('inf'))
            Pij = tl.exp(Sij - Li[:, None])

            dVj += tl.dot(tl.trans(Pij.to(dOi.dtype)), dOi)
            dPij = tl.dot(dOi, tl.trans(Vj_owner))
            dSij = Pij * (dPij - Di[:, None]) * scale
            dKj += tl.dot(tl.trans(dSij.to(Qi.dtype)), Qi)

    tl.store(
        dK_base
        + offs_n_owner[:, None] * stride_gkk
        + offs_d[None, :] * stride_gkd,
        dKj.to(Kj_owner.dtype),
        mask=offs_n_owner[:, None] < N_KEYS,
    )
    tl.store(
        dV_base
        + offs_n_owner[:, None] * stride_gvk
        + offs_d[None, :] * stride_gvd,
        dVj.to(Vj_owner.dtype),
        mask=offs_n_owner[:, None] < N_KEYS,
    )

    # ------------------------------------------------------------------
    # Phase B: 固定 Q owner tile，扫描所有相关 K/V tile，计算 dQ。
    # ------------------------------------------------------------------
    offs_m_owner = owner_start + tl.arange(0, OWNER_TILE)

    Qi_owner = tl.load(
        Q_base
        + offs_m_owner[:, None] * stride_qq
        + offs_d[None, :] * stride_qd,
        mask=offs_m_owner[:, None] < N_QUERIES,
        other=0.0,
    )
    dOi_owner = tl.load(
        dO_base
        + offs_m_owner[:, None] * stride_goq
        + offs_d[None, :] * stride_god,
        mask=offs_m_owner[:, None] < N_QUERIES,
        other=0.0,
    )
    Li_owner = tl.load(
        L_base + offs_m_owner * stride_lq,
        mask=offs_m_owner < N_QUERIES,
        other=0.0,
    )
    Di_owner = tl.load(
        Delta_base + offs_m_owner * stride_dq,
        mask=offs_m_owner < N_QUERIES,
        other=0.0,
    )

    dQi = tl.zeros((OWNER_TILE, D), dtype=tl.float32)

    if is_causal:
        # 对角线左侧：全部合法，无需逐元素 causal mask。
        for start_n in range(0, owner_start, INNER_TILE):
            offs_n = start_n + tl.arange(0, INNER_TILE)
            Kj = tl.load(
                K_base + offs_n[:, None] * stride_kk + offs_d[None, :] * stride_kd,
                mask=offs_n[:, None] < N_KEYS,
                other=0.0,
            )
            Vj = tl.load(
                V_base + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vd,
                mask=offs_n[:, None] < N_KEYS,
                other=0.0,
            )

            Sij = tl.dot(Qi_owner, tl.trans(Kj)) * scale
            valid = (offs_m_owner[:, None] < N_QUERIES) & (offs_n[None, :] < N_KEYS)
            Sij = tl.where(valid, Sij, -float('inf'))
            Pij = tl.exp(Sij - Li_owner[:, None])

            dPij = tl.dot(dOi_owner, tl.trans(Vj))
            dSij = Pij * (dPij - Di_owner[:, None]) * scale
            dQi += tl.dot(dSij.to(Kj.dtype), Kj)

        # 与当前 Q owner tile 相交的对角区域：需要 causal mask。
        for r in range(0, OWNER_TILE // INNER_TILE):
            start_n = owner_start + r * INNER_TILE
            offs_n = start_n + tl.arange(0, INNER_TILE)
            Kj = tl.load(
                K_base + offs_n[:, None] * stride_kk + offs_d[None, :] * stride_kd,
                mask=offs_n[:, None] < N_KEYS,
                other=0.0,
            )
            Vj = tl.load(
                V_base + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vd,
                mask=offs_n[:, None] < N_KEYS,
                other=0.0,
            )

            Sij = tl.dot(Qi_owner, tl.trans(Kj)) * scale
            valid = (offs_m_owner[:, None] < N_QUERIES) & (offs_n[None, :] < N_KEYS)
            causal_mask = offs_m_owner[:, None] >= offs_n[None, :]
            Sij = tl.where(valid & causal_mask, Sij, -float('inf'))
            Pij = tl.exp(Sij - Li_owner[:, None])

            dPij = tl.dot(dOi_owner, tl.trans(Vj))
            dSij = Pij * (dPij - Di_owner[:, None]) * scale
            dQi += tl.dot(dSij.to(Kj.dtype), Kj)
    else:
        for start_n in range(0, N_KEYS, INNER_TILE):
            offs_n = start_n + tl.arange(0, INNER_TILE)
            Kj = tl.load(
                K_base + offs_n[:, None] * stride_kk + offs_d[None, :] * stride_kd,
                mask=offs_n[:, None] < N_KEYS,
                other=0.0,
            )
            Vj = tl.load(
                V_base + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vd,
                mask=offs_n[:, None] < N_KEYS,
                other=0.0,
            )

            Sij = tl.dot(Qi_owner, tl.trans(Kj)) * scale
            valid = (offs_m_owner[:, None] < N_QUERIES) & (offs_n[None, :] < N_KEYS)
            Sij = tl.where(valid, Sij, -float('inf'))
            Pij = tl.exp(Sij - Li_owner[:, None])

            dPij = tl.dot(dOi_owner, tl.trans(Vj))
            dSij = Pij * (dPij - Di_owner[:, None]) * scale
            dQi += tl.dot(dSij.to(Kj.dtype), Kj)

    tl.store(
        dQ_base
        + offs_m_owner[:, None] * stride_gqq
        + offs_d[None, :] * stride_gqd,
        dQi.to(Qi_owner.dtype),
        mask=offs_m_owner[:, None] < N_QUERIES,
    )


class Flashattention2_triton_fused_bwd(torch.autograd.Function):
    """
    低资源版本：forward 沿用 16x16 online-softmax kernel；
    backward 使用 Delta preprocess + 单次主 kernel launch，且不使用 atomic_add。
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Batch_size, N_QUERIES, D = Q.shape
        N_KEYS = K.shape[1]

        ctx.FWD_Q_TILE_SIZE = 16
        ctx.FWD_K_TILE_SIZE = 16
        ctx.OWNER_TILE = 32
        ctx.INNER_TILE = 16
        ctx.DELTA_TILE = 64

        O = torch.empty_like(Q)
        L = torch.empty((Batch_size, N_QUERIES), dtype=torch.float32, device=Q.device)
        scale = 1.0 / math.sqrt(D)

        flash_fwd_kernel[(math.ceil(N_QUERIES / ctx.FWD_Q_TILE_SIZE), Batch_size)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS,
            scale,
            D,
            ctx.FWD_Q_TILE_SIZE,
            ctx.FWD_K_TILE_SIZE,
            is_causal,
        )

        ctx.is_causal = is_causal
        ctx.save_for_backward(Q, K, V, L, O)
        return O

    @staticmethod
    def backward(ctx, GRAD_O):
        Q, K, V, L, O = ctx.saved_tensors
        Batch_size, N_QUERIES, D = Q.shape
        N_KEYS = K.shape[1]

        if N_QUERIES != N_KEYS:
            raise RuntimeError(
                "fused-no-atomic backward 当前针对 GPT self-attention："
                "要求 N_QUERIES == N_KEYS。"
            )

        GRAD_O = GRAD_O.contiguous()
        GRAD_Q = torch.empty_like(Q)
        GRAD_K = torch.empty_like(K)
        GRAD_V = torch.empty_like(V)
        Delta = torch.empty((Batch_size, N_QUERIES), dtype=torch.float32, device=Q.device)

        scale = 1.0 / math.sqrt(D)

        flash_bwd_delta_kernel[(math.ceil(N_QUERIES / ctx.DELTA_TILE), Batch_size)](
            O, GRAD_O, Delta,
            O.stride(0), O.stride(1), O.stride(2),
            GRAD_O.stride(0), GRAD_O.stride(1), GRAD_O.stride(2),
            Delta.stride(0), Delta.stride(1),
            N_QUERIES,
            D,
            ctx.DELTA_TILE,
            num_warps=4,
        )

        flash_bwd_fused_no_atomic_kernel[(math.ceil(N_KEYS / ctx.OWNER_TILE), Batch_size)](
            Q, K, V, L, GRAD_O, Delta,
            GRAD_Q, GRAD_K, GRAD_V,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            L.stride(0), L.stride(1),
            GRAD_O.stride(0), GRAD_O.stride(1), GRAD_O.stride(2),
            Delta.stride(0), Delta.stride(1),
            GRAD_Q.stride(0), GRAD_Q.stride(1), GRAD_Q.stride(2),
            GRAD_K.stride(0), GRAD_K.stride(1), GRAD_K.stride(2),
            GRAD_V.stride(0), GRAD_V.stride(1), GRAD_V.stride(2),
            scale,
            N_QUERIES, N_KEYS,
            D,
            ctx.OWNER_TILE,
            ctx.INNER_TILE,
            ctx.is_causal,
            num_warps=4,
            num_stages=2,
        )

        return GRAD_Q, GRAD_K, GRAD_V, None


# 包装函数，方便调用
def triton_fa2(Q, K, V, is_causal=True):
    return Flashattention2_triton.apply(Q, K, V, is_causal)

def triton_fa2_2(Q, K, V, is_causal=True):
    return Flashattention2_triton_2.apply(Q, K, V, is_causal)


def triton_fa2_fused_bwd(Q, K, V, is_causal=True):
    return Flashattention2_triton_fused_bwd.apply(Q, K, V, is_causal)

# ======================== 【2】MHA 三种 Attention Core 实现 ========================
# 说明：
# 1. 下面测试的是一个完整 MHA 层，而不是只测试 attention core。
# 2. 三个 backend 共用完全相同的 QKV projection 和 output projection 权重。
# 3. 区别只在于中间的 scaled dot-product attention core：
#       - vanilla: 显式构建注意力矩阵
#       - sdpa: torch.nn.functional.scaled_dot_product_attention
#       - triton_16 / triton_32: 你的 Triton kernel
# 4. 你的 kernel 输入为 [B*H, T, Dh]，因此在进入 kernel 前合并 batch 和 head 维度。

import argparse
from contextlib import nullcontext

import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:
    SDPBackend = None
    sdpa_kernel = None


def vanilla_attention_core(Q, K, V, is_causal=True):
    """
    Vanilla PyTorch Attention Core。

    Args:
        Q, K, V: [B * H, T, Dh]
    Returns:
        out: [B * H, T, Dh]
    """
    _, N, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    attn = (Q @ K.transpose(-2, -1)) * scale

    if is_causal:
        mask = torch.ones((N, N), device=Q.device, dtype=torch.bool).tril_()
        attn = attn.masked_fill(~mask, -torch.inf)

    prob = torch.softmax(attn, dim=-1)
    return prob @ V

import warnings
import torch


def diagnose_sdpa_flash(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = True,
) -> None:
    print("=" * 100)
    print("PyTorch SDPA Flash Backend 诊断")
    print("=" * 100)

    print("torch:", torch.__version__)
    print("cuda build:", torch.version.cuda)
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))

    print("flash enabled:", torch.backends.cuda.flash_sdp_enabled())
    print(
        "flash built / available:",
        torch.backends.cuda.is_flash_attention_available(),
    )

    print("-" * 100)

    for name, tensor in (("Q", q), ("K", k), ("V", v)):
        print(
            f"{name}: "
            f"shape={tuple(tensor.shape)}, "
            f"dtype={tensor.dtype}, "
            f"device={tensor.device}, "
            f"stride={tensor.stride()}, "
            f"contiguous={tensor.is_contiguous()}"
        )

    print("-" * 100)

    params = torch.backends.cuda.SDPAParams(
        q,
        k,
        v,
        None,       # attn_mask
        0.0,        # dropout_p
        is_causal,
        False,      # enable_gqa
    )

    # debug=True 时，PyTorch 会输出具体不满足的条件。
    with warnings.catch_warnings(record=False):
        warnings.simplefilter("always")

        usable = torch.backends.cuda.can_use_flash_attention(
            params,
            debug=True,
        )

    print("-" * 100)
    print("can_use_flash_attention:", usable)
    print("=" * 100)
def sdpa_attention_core(Q, K, V, is_causal=True, force_flash_backend=False):
    """
    PyTorch 官方 SDPA。

    Args:
        Q, K, V: [B, H, T, Dh]
        force_flash_backend:
            False: 让 PyTorch 自动选择可用 backend，兼容性更好。
            True: 强制尝试 PyTorch 官方 FLASH_ATTENTION backend。
                  通常需要 fp16 / bf16，以及支持该 backend 的 GPU。
    """
    context = nullcontext()
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    if force_flash_backend:
        diagnose_sdpa_flash(
            Q,
            K,
            V,
            is_causal=is_causal,
        )
    if force_flash_backend:
        if sdpa_kernel is None or SDPBackend is None:
            raise RuntimeError(
                "当前 PyTorch 版本不支持 torch.nn.attention.sdpa_kernel；"
                "请将 force_flash_backend=False，或升级 PyTorch。"
            )
        context = sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])

    with context:
        return F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
        )


class SingleMHA(nn.Module):
    """
    用于小样本测试的单个 Multi-Head Attention 层。

    输入:  x [B, T, d_model]
    输出:  y [B, T, d_model]

    该模块保留完整 MHA 数据流：
        x -> QKV projection -> reshape heads -> attention core
          -> concat heads -> output projection
    """

    def __init__(
        self,
        d_model=256,
        num_heads=4,
        backend="vanilla",
        force_sdpa_flash=False,
        bias=False,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能够被 num_heads 整除")

        valid_backends = {"vanilla", "sdpa", "triton_16", "triton_32", "triton_fused_bwd"}
        if backend not in valid_backends:
            raise ValueError(f"backend 必须是 {sorted(valid_backends)}，实际为 {backend!r}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.backend = backend
        self.force_sdpa_flash = force_sdpa_flash

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x):
        B, T, D = x.shape
        if D != self.d_model:
            raise ValueError(f"输入最后一维应为 {self.d_model}，实际为 {D}")

        # [B, T, 3D] -> [B, T, 3, H, Dh] -> [3, B, H, T, Dh]
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)  # 每个张量均为 [B, H, T, Dh]

        if self.backend == "sdpa":
            out = sdpa_attention_core(
                q,
                k,
                v,
                is_causal=True,
                force_flash_backend=self.force_sdpa_flash,
            )
        else:
            # 你的 Triton kernel 和 Vanilla baseline 均使用 [B*H, T, Dh]
            q = q.contiguous().view(B * self.num_heads, T, self.head_dim)
            k = k.contiguous().view(B * self.num_heads, T, self.head_dim)
            v = v.contiguous().view(B * self.num_heads, T, self.head_dim)

            if self.backend == "vanilla":
                out = vanilla_attention_core(q, k, v, is_causal=True)
            elif self.backend == "triton_16":
                out = triton_fa2_2(q, k, v, is_causal=True)
            elif self.backend == "triton_32":
                out = triton_fa2(q, k, v, is_causal=True)
            elif self.backend == "triton_fused_bwd":
                out = triton_fa2_fused_bwd(q, k, v, is_causal=True)
            else:
                raise AssertionError("不可达分支")

            out = out.view(B, self.num_heads, T, self.head_dim)

        # [B, H, T, Dh] -> [B, T, H, Dh] -> [B, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


def clear_grads(module, x):
    x.grad = None
    module.zero_grad(set_to_none=True)


def clone_module_with_backend(reference, backend, force_sdpa_flash=False):
    module = SingleMHA(
        d_model=reference.d_model,
        num_heads=reference.num_heads,
        backend=backend,
        force_sdpa_flash=force_sdpa_flash,
        bias=reference.qkv_proj.bias is not None,
    ).to(device=reference.qkv_proj.weight.device, dtype=reference.qkv_proj.weight.dtype)
    module.load_state_dict(reference.state_dict())
    return module


def check_mha_correctness(
    B=1,
    T=128,
    d_model=256,
    num_heads=4,
    dtype=torch.float32,
    include_triton_32=False,
    include_triton_fused=False,
):
    """
    小样本数值一致性测试。

    首次运行建议：
        B=1, T=128, d_model=256, num_heads=4, dtype=float32

    注意：你的 16x16 kernel 已在本文件中修复 causal backward mask 的 tile 索引。
    """
    if not torch.cuda.is_available():
        raise RuntimeError("该测试需要 CUDA GPU")

    device = "cuda"
    torch.manual_seed(42)

    x_ref = torch.randn(B, T, d_model, device=device, dtype=dtype, requires_grad=True)
    grad_out = torch.randn_like(x_ref)

    vanilla = SingleMHA(d_model, num_heads, backend="vanilla").to(device=device, dtype=dtype)
    sdpa = clone_module_with_backend(vanilla, "sdpa")
    triton16 = clone_module_with_backend(vanilla, "triton_16")

    modules = [("PyTorch SDPA", sdpa), ("Triton 16x16", triton16)]
    if include_triton_fused:
        modules.append(("Triton fused-bwd", clone_module_with_backend(vanilla, "triton_fused_bwd")))
    if include_triton_32:
        modules.append(("Triton 32x32", clone_module_with_backend(vanilla, "triton_32")))

    # Vanilla reference
    out_ref = vanilla(x_ref)
    out_ref.backward(grad_out)
    dx_ref = x_ref.grad.detach().clone()
    dqkv_ref = vanilla.qkv_proj.weight.grad.detach().clone()
    do_ref = vanilla.out_proj.weight.grad.detach().clone()

    print("=" * 96)
    print("数值一致性测试：以 Vanilla PyTorch MHA 为 reference")
    print(f"B={B}, T={T}, d_model={d_model}, num_heads={num_heads}, head_dim={d_model // num_heads}, dtype={dtype}")
    print("=" * 96)
    print(f"{'Backend':<20} {'max|Δout|':>14} {'max|Δdx|':>14} {'max|Δdqkv|':>14} {'max|ΔdWo|':>14}")

    for name, module in modules:
        x = x_ref.detach().clone().requires_grad_(True)
        out = module(x)
        out.backward(grad_out)

        max_out = (out.detach() - out_ref.detach()).abs().max().item()
        max_dx = (x.grad - dx_ref).abs().max().item()
        max_dqkv = (module.qkv_proj.weight.grad - dqkv_ref).abs().max().item()
        max_do = (module.out_proj.weight.grad - do_ref).abs().max().item()

        print(f"{name:<20} {max_out:>14.6e} {max_dx:>14.6e} {max_dqkv:>14.6e} {max_do:>14.6e}")

    print("=" * 96)


def benchmark_single_mha(
    B=1,
    T=512,
    d_model=256,
    num_heads=4,
    dtype=torch.float32,
    warmup=20,
    rep=100,
    include_triton_32=False,
    include_triton_fused=False,
    force_sdpa_flash=False,
):
    """
    对单个完整 MHA 层进行 benchmark。

    输出：
        - Forward 时间
        - Forward + Backward 时间
        - 纯 Backward 的近似值 = (Forward + Backward) - Forward
        - 一次 Forward + Backward 的峰值显存

    说明：
        纯 backward 并未被独立执行，因为 backward 依赖 forward 计算图；
        这里通过差值进行近似估算。
    """
    if not torch.cuda.is_available():
        raise RuntimeError("该测试需要 CUDA GPU")

    device = "cuda"
    torch.manual_seed(42)

    reference = SingleMHA(d_model, num_heads, backend="vanilla").to(device=device, dtype=dtype)

    modules = [
        ("Vanilla PyTorch", reference),
        ("PyTorch SDPA", clone_module_with_backend(reference, "sdpa", force_sdpa_flash)),
        ("Triton 16x16", clone_module_with_backend(reference, "triton_16")),
    ]

    if include_triton_fused:
        modules.append(("Triton fused-bwd", clone_module_with_backend(reference, "triton_fused_bwd")))

    if include_triton_32:
        modules.append(("Triton 32x32", clone_module_with_backend(reference, "triton_32")))

    x = torch.randn(B, T, d_model, device=device, dtype=dtype, requires_grad=True)
    grad_out = torch.randn_like(x)

    rows = []

    for name, module in modules:
        def run_fwd():
            return module(x)

        def run_fwd_bwd():
            clear_grads(module, x)
            out = module(x)
            out.backward(grad_out)

        # Triton/PyTorch 首次调用可能包含编译或 autotune 开销，do_bench warmup 会排除。
        t_fwd = tt.do_bench(run_fwd, warmup=warmup, rep=rep, return_mode="median")
        t_fwd_bwd = tt.do_bench(run_fwd_bwd, warmup=warmup, rep=rep, return_mode="median")
        t_bwd_est = max(t_fwd_bwd - t_fwd, 0.0)

        # 测量一次 fwd+bwd 峰值显存。
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        clear_grads(module, x)
        run_fwd_bwd()
        torch.cuda.synchronize()
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        rows.append((name, t_fwd, t_bwd_est, t_fwd_bwd, peak_mem_mb))

    vanilla_total = rows[0][3]

    print("=" * 112)
    print("单个完整 MHA Benchmark：Vanilla PyTorch vs 官方 SDPA vs 自定义 Triton")
    print(
        f"B={B}, T={T}, d_model={d_model}, num_heads={num_heads}, "
        f"head_dim={d_model // num_heads}, causal=True, dtype={dtype}"
    )
    print("注意：Backward 为差值估算；Fwd+Bwd 是直接实测。")
    print("=" * 112)
    print(
        f"{'Backend':<20} {'Forward(ms)':>14} {'Backward≈(ms)':>16} "
        f"{'Fwd+Bwd(ms)':>16} {'PeakMem(MB)':>14} {'vs Vanilla':>14}"
    )

    for name, t_fwd, t_bwd_est, t_fwd_bwd, peak_mem_mb in rows:
        speedup = vanilla_total / t_fwd_bwd
        print(
            f"{name:<20} {t_fwd:>14.3f} {t_bwd_est:>16.3f} "
            f"{t_fwd_bwd:>16.3f} {peak_mem_mb:>14.1f} {speedup:>13.2f}x"
        )

    print("=" * 112)
    print("说明：")
    print("1. Vanilla PyTorch 会显式构建 T×T 注意力矩阵。")
    print("2. PyTorch SDPA 默认自动选择当前硬件和 dtype 可用的官方 backend。")
    print("3. --force-sdpa-flash 会强制尝试 PyTorch 官方 FLASH_ATTENTION backend；不满足条件时会报错。")
    print("4. 默认只测 Triton 16x16，以降低资源压力；使用 --include-triton-fused 增加无 atomic 融合 backward。")
    print("5. 使用 --include-triton-32 可增加 32x32 版本。")
    print("=" * 112)


def parse_dtype(name):
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"不支持 dtype={name!r}，可选值为 {sorted(mapping)}")
    return mapping[name]


def main():
    parser = argparse.ArgumentParser(description="Benchmark one MHA layer with Vanilla, PyTorch SDPA, and Triton Attention")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--include-triton-32", default = True,action="store_true")
    parser.add_argument("--include-triton-fused", default = True,action="store_true")
    parser.add_argument("--force-sdpa-flash", default = True,action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args()

    dtype = parse_dtype(args.dtype)

    if not args.skip_correctness:
        # 正确性测试使用更小样本，避免 Vanilla Attention 占用过多显存。
        check_mha_correctness(
            B=1,
            T=128,
            d_model=args.d_model,
            num_heads=args.num_heads,
            dtype=dtype,
            include_triton_32=args.include_triton_32,
            include_triton_fused=args.include_triton_fused,
        )

    benchmark_single_mha(
        B=args.batch_size,
        T=args.seq_len,
        d_model=args.d_model,
        num_heads=args.num_heads,
        dtype=dtype,
        warmup=args.warmup,
        rep=args.rep,
        include_triton_32=args.include_triton_32,
        include_triton_fused=args.include_triton_fused,
        force_sdpa_flash=args.force_sdpa_flash,
    )


if __name__ == "__main__":
    main()
