@triton.jit
def flash_bwd_KV_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, grad_O_ptr,  # 输入矩阵指针
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
        is_causal: tl.constexpr,
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

    Q_ptr = Q_ptr + batch_index * stride_qb
    O_ptr = O_ptr + batch_index * stride_ob
    L_ptr = L_ptr + batch_index * stride_lb
    grad_O_ptr = grad_O_ptr + batch_index * stride_gob

    k_start = key_tile_index * K_TILE_SIZE
    offs_n = k_start + tl.arange(0, K_TILE_SIZE)
    offs_d = tl.arange(0, D)

    K_ptrs = K_ptr + batch_index * stride_kb + offs_n[:, None] * stride_kk + offs_d[None, :] * stride_kd
    V_ptrs = V_ptr + batch_index * stride_vb + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vd
    Kj = tl.load(K_ptrs, mask=offs_n[:, None] < N_KEYS, other=0.)
    Vj = tl.load(V_ptrs, mask=offs_n[:, None] < N_KEYS, other=0.)

    dKj = tl.zeros((K_TILE_SIZE, D), dtype=Kj.dtype)
    dVj = tl.zeros((K_TILE_SIZE, D), dtype=Vj.dtype)

    if is_causal:
        start_q_tile = (k_start + Q_TILE_SIZE - 1) // Q_TILE_SIZE  # ceil除法
        num_q_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
    else:
        start_q_tile = 0
        num_q_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)

    for i in range(start_q_tile, num_q_tiles):
        offs_m = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        Q_ptrs = Q_ptr + offs_m[:, None] * stride_qq + offs_d[None, :] * stride_qd
        O_ptrs = O_ptr + offs_m[:, None] * stride_oq + offs_d[None, :] * stride_od
        dO_ptrs = grad_O_ptr + offs_m[:, None] * stride_goq + offs_d[None, :] * stride_god
        L_ptrs = L_ptr + offs_m * stride_lq

        Qi = tl.load(Q_ptrs, mask=offs_m[:, None] < N_QUERIES, other=0.)
        Oi = tl.load(O_ptrs, mask=offs_m[:, None] < N_QUERIES, other=0.)
        dOi = tl.load(dO_ptrs, mask=offs_m[:, None] < N_QUERIES, other=0.)
        Li = tl.load(L_ptrs, mask=offs_m < N_QUERIES, other=0.)

        Sij = tl.dot(Qi, tl.trans(Kj)) * scale

        q_offsets = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
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

    tl.store(GRAD_K_block_ptr, dKj, boundary_check=(0, 1), )
    tl.store(GRAD_V_block_ptr, dVj, boundary_check=(0, 1), )


@triton.jit
def flash_bwd_Q_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, grad_O_ptr,  # 输入矩阵指针
        grad_Q_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        stride_gob, stride_goq, stride_god,
        stride_gqb, stride_gqq, stride_gqd,
        scale,
        N_QUERIES, N_KEYS,  # 查询数和键值数
        D: tl.constexpr,  # 特征维度（编译期常量）
        Q_TILE_SIZE: tl.constexpr,  # 查询分块尺寸B_q
        K_TILE_SIZE: tl.constexpr,  # 键分块尺寸B_k
        is_causal: tl.constexpr,
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
        offs_k = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        K_ptrs = K_ptr + offs_k[:, None] * stride_kk + offs_d[None, :] * stride_kd
        V_ptrs = V_ptr + offs_k[:, None] * stride_vk + offs_d[None, :] * stride_vd

        Kj = tl.load(K_ptrs, mask=offs_k[:, None] < N_KEYS, other=0.)
        Vj = tl.load(V_ptrs, mask=offs_k[:, None] < N_KEYS, other=0.)

        Sij = tl.dot(Qi, tl.trans(Kj)) * scale

        q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        k_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)

        if is_causal:
            mask = q_offsets[:, None] >= k_offsets[None, :]
            Sij = tl.where(mask, Sij, -float('inf'))
        beta = tl.exp(Sij - Li[:, None])

        dbeta = tl.dot(dOi, tl.trans(Vj))
        Di = tl.sum(dOi * Oi, axis=1)
        dSij = beta * (dbeta - Di[:, None]) * scale

        dQi += tl.dot(dSij, Kj)
    dQ_ptrs = grad_Q_ptr + batch_index * stride_gqb + offs_m[:, None] * stride_gqq + offs_d[None, :] * stride_gqd
    tl.store(dQ_ptrs, dQi, mask=offs_m[:, None] < N_KEYS, other=0.)