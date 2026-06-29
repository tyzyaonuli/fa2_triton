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
        Oi = tl.dot(beta, Vj) + Oi*alpha[:, None]
        mi = mi_new

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    Oi = Oi/li[:, None]
    Li = mi + tl.log(li)

    tl.store(O_block_ptr, Oi, boundary_check=(0, 1))
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
        ctx.Q_TILE_SIZE = 8
        ctx.K_TILE_SIZE = 8
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

    dVj += tl.dot(tl.trans(beta), dOi)
    dbeta = tl.dot(dOi, tl.trans(Vj))
    Di = tl.sum(dOi * Oi, axis=1)
    dSij = beta * (dbeta - Di[:, None]) * scale
    dKj += tl.dot(tl.trans(dSij), Qi)

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


        dVj += tl.dot(tl.trans(beta), dOi)
        dbeta = tl.dot(dOi, tl.trans(Vj))
        Di = tl.sum(dOi * Oi, axis = 1)
        dSij = beta*(dbeta - Di[:, None])*scale
        dKj += tl.dot(tl.trans(dSij), Qi)



    tl.store(GRAD_K_block_ptr, dKj, boundary_check = (0, 1), )
    tl.store(GRAD_V_block_ptr, dVj, boundary_check = (0, 1), )
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


        dVj += tl.dot(tl.trans(beta), dOi)
        dbeta = tl.dot(dOi, tl.trans(Vj))
        Di = tl.sum(dOi * Oi, axis = 1)
        dSij = beta*(dbeta - Di[:, None])*scale
        dKj += tl.dot(tl.trans(dSij), Qi)



    tl.store(GRAD_K_block_ptr, dKj, boundary_check = (0, 1), )
    tl.store(GRAD_V_block_ptr, dVj, boundary_check = (0, 1), )
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

        dQi += tl.dot(dSij, Kj)

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

    dQi += tl.dot(dSij, Kj)


    dQ_ptrs = grad_Q_ptr + batch_index * stride_gqb +offs_m[:, None] * stride_gqq + offs_d[None, :] * stride_gqd
    tl.store(dQ_ptrs, dQi, mask=offs_m[:, None]<N_QUERIES)

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
        k_offsets = (max_k_tile - 1) * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)

        if is_causal:
            mask = q_offsets[:, None] >= k_offsets[None, :]
            Sij = tl.where(mask, Sij, -float('inf'))
        beta = tl.exp(Sij - Li[:, None])

        dbeta = tl.dot(dOi, tl.trans(Vj))
        Di = tl.sum(dOi * Oi, axis=1)
        dSij = beta * (dbeta - Di[:, None]) * scale

        dQi += tl.dot(dSij, Kj)



    dQ_ptrs = grad_Q_ptr + batch_index * stride_gqb +offs_m[:, None] * stride_gqq + offs_d[None, :] * stride_gqd
    tl.store(dQ_ptrs, dQi, mask=offs_m[:, None]<N_QUERIES)

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


        dVj += tl.dot(tl.trans(beta), dOi)
        dbeta = tl.dot(dOi, tl.trans(Vj))
        Di = tl.sum(dOi * Oi, axis = 1)
        dSij = beta*(dbeta - Di[:, None])*scale

        dQi_local = tl.dot(dSij, Kj)
        offs_q = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        mask = q_offsets[:, None] < N_QUERIES
        dq_ptrs = grad_Q_ptr + batch_index * stride_gqb \
                  + offs_q[:, None] * stride_gqq \
                  + offs_d[None, :] * stride_gqd
        tl.atomic_add(dq_ptrs, dQi_local, mask=mask)

        dKj += tl.dot(tl.trans(dSij), Qi)



    tl.store(GRAD_K_block_ptr, dKj, boundary_check = (0, 1), )
    tl.store(GRAD_V_block_ptr, dVj, boundary_check = (0, 1), )

# 包装函数，方便调用
def triton_fa2(Q, K, V, is_causal=True):
    return Flashattention2_triton.apply(Q, K, V, is_causal)

def triton_fa2_2(Q, K, V, is_causal=True):
    return Flashattention2_triton_2.apply(Q, K, V, is_causal)

# ======================== 【2】普通 PyTorch 实现（无FlashAttention） ========================
def pytorch_standard_attention(Q, K, V, is_causal=True):
    """
    标准PyTorch自注意力：matmul + softmax + matmul
    无FlashAttention，最原始实现，用于对比
    """
    B, N, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    attn = (Q @ K.transpose(-2, -1)) * scale  # Q*K^T / sqrt(d)

    # 因果掩码
    if is_causal:
        mask = torch.tril(torch.ones(N, N, device=Q.device, dtype=torch.bool))
        attn = attn.masked_fill(~mask, -torch.inf)

    attn = torch.softmax(attn, dim=-1)
    out = attn @ V
    return out


# ======================== 【3】基准测试核心函数 ========================
def benchmark_attention(B=2, N=1024, D=128, is_causal=True):
    device = "cuda"
    dtype = torch.float32  # 训练常用精度

    # 初始化 QKV（开启梯度，支持反向传播）
    Q = torch.randn(B, N, D, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(B, N, D, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(B, N, D, device=device, dtype=dtype, requires_grad=True)
    grad_out = torch.randn_like(Q)  # 反向传播用的梯度

    # ---------------- 测试函数定义 ----------------
    def triton_fwd():
        return triton_fa2(Q, K, V, is_causal)

    def triton_bwd():
        out = triton_fa2(Q, K, V, is_causal)
        out.backward(grad_out, retain_graph=True)
        Q.grad = None;
        K.grad = None;
        V.grad = None

    def triton_fwd_2():
        return triton_fa2_2(Q, K, V, is_causal)

    def triton_bwd_2():
        out = triton_fa2_2(Q, K, V, is_causal)
        out.backward(grad_out, retain_graph=True)
        Q.grad = None;
        K.grad = None;
        V.grad = None

    def torch_fwd():
        return pytorch_standard_attention(Q, K, V, is_causal)

    def torch_bwd():
        out = pytorch_standard_attention(Q, K, V, is_causal)
        out.backward(grad_out, retain_graph=True)
        Q.grad = None;
        K.grad = None;
        V.grad = None

    # ---------------- do_bench 测速 ----------------
    # Triton
    t_triton_fwd = tt.do_bench(triton_fwd, warmup=50, rep=200, return_mode="median")
    t_triton_bwd = tt.do_bench(triton_bwd, warmup=50, rep=200, return_mode="median")
    t_triton_total = t_triton_fwd + t_triton_bwd

    t_triton_fwd_2 = tt.do_bench(triton_fwd_2, warmup=50, rep=200, return_mode="median")
    t_triton_bwd_2 = tt.do_bench(triton_bwd_2, warmup=50, rep=200, return_mode="median")
    t_triton_total_2 = t_triton_fwd_2 + t_triton_bwd_2

    # PyTorch 标准
    t_torch_fwd = tt.do_bench(torch_fwd, warmup=50, rep=200, return_mode="median")
    t_torch_bwd = tt.do_bench(torch_bwd, warmup=50, rep=200, return_mode="median")
    t_torch_total = t_torch_fwd + t_torch_bwd

    # ---------------- 输出结果 ----------------
    print("=" * 80)
    print(f"  基准测试：Triton FlashAttention2 VS 标准PyTorch注意力")
    print(f"  参数：B={B}, 序列长度={N}, 头维度={D}, 因果掩码={is_causal}")
    print("=" * 80)
    print(f"{'':<10} {'前向(ms)':<12} {'反向(ms)':<12} {'总耗时(ms)':<12}")
    print(f"{'Triton':<10} {t_triton_fwd:<12.3f} {t_triton_bwd:<12.3f} {t_triton_total:<12.3f}")
    print(f"{'Triton':<10} {t_triton_fwd_2:<12.3f} {t_triton_bwd_2:<12.3f} {t_triton_total_2:<12.3f}")
    print(f"{'PyTorch':<10} {t_torch_fwd:<12.3f} {t_torch_bwd:<12.3f} {t_torch_total:<12.3f}")
    print("=" * 80)
    print(f"✅ 前向加速比: {t_torch_fwd / t_triton_fwd:.2f}x, {t_torch_fwd / t_triton_fwd_2:.2f}x")
    print(f"✅ 反向加速比: {t_torch_bwd / t_triton_bwd:.2f}x, {t_torch_bwd / t_triton_bwd_2:.2f}x")
    print(f"✅ 总加速比:   {t_torch_total / t_triton_total:.2f}x, {t_torch_total / t_triton_total_2:.2f}x")
    print("=" * 80)


# ======================== 运行测试 ========================
if __name__ == "__main__":
    # 可自由修改：批次B、序列长度N、维度D
    benchmark_attention(B=2, N=4096, D=64, is_causal=True)
