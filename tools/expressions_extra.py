import numpy as np

COOP_T128 = "WORK=t128,REDUCE=coop"   # same as the open set's REDUCE_KNOBS (block-wide tree)
COOP_T32 = "WORK=t32,REDUCE=coop"     # one warp per row: shuffle only, no smem tree
COOP_R2 = "WORK=t32,REDUCE=coop/r2"   # warp-cooperative + 2 independent accumulators (ILP)
ILP_R4 = "REDUCE=r4"                  # per-thread reduction with 4 accumulators, no shuffles
SERIAL = "REDUCE="                    # one thread walks the whole row


def _f64(x):
    return x.astype(np.float64)


def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


OPEN = [
    # ---- category 1: pointwise (tests/compiler/cli, tests/compiler/loader) ------------------
    dict(id="e_p01_relu_8",           # smaller than one warp
         code="torch.relu(torch.randn(8))",
         ref=lambda x: [np.maximum(x, 0)], rtol=0, atol=0),
    dict(id="e_p02_relu_4",
         code="torch.relu(torch.randn(4))",
         ref=lambda x: [np.maximum(x, 0)], rtol=0, atol=0),
    dict(id="e_p03_relu_2d",          # 2D index math
         code="torch.relu(torch.randn(16,32))",
         ref=lambda x: [np.maximum(x, 0)], rtol=0, atol=0),
    dict(id="e_p04_neg",
         code="torch.neg(torch.randn(16))",
         ref=lambda x: [-x], rtol=0, atol=0),
    dict(id="e_p05_sigmoid",          # ex2 / rcp in PTX
         code="torch.sigmoid(torch.randn(16))",
         ref=lambda x: [_sigmoid(_f64(x)).astype(np.float32)], rtol=1e-5, atol=1e-6),
    dict(id="e_p06_exp_2d",
         code="torch.exp(torch.randn(8, 8))",
         ref=lambda x: [np.exp(_f64(x)).astype(np.float32)], rtol=1e-5, atol=1e-6),
    dict(id="e_p07_silu_3d",          # 3D shape, sigmoid inside
         code="torch.nn.functional.silu(torch.randn(1,32,128))",
         ref=lambda x: [(_f64(x) * _sigmoid(_f64(x))).astype(np.float32)], rtol=1e-5, atol=1e-6),

    # ---- category 2: reductions, the realization corpus schedule matrix ---------------------
    *[dict(id=f"e_r_amax_{tag}",
           code="torch.randn(4, 1024).amax(-1, keepdim=True)",
           ref=lambda x: [x.max(-1, keepdims=True)], rtol=0, atol=0, knobs=k)
      for tag, k in [("coop128", COOP_T128), ("coop32", COOP_T32), ("coop_r2", COOP_R2),
                     ("r4", ILP_R4), ("serial", SERIAL)]],
    *[dict(id=f"e_r_mean_{tag}",
           code="torch.randn(4, 1024).mean(-1, keepdim=True)",
           ref=lambda x: [_f64(x).mean(-1, keepdims=True).astype(np.float32)],
           rtol=1e-4, atol=1e-4, knobs=k)
      for tag, k in [("coop128", COOP_T128), ("coop32", COOP_T32), ("coop_r2", COOP_R2),
                     ("r4", ILP_R4), ("serial", SERIAL)]],
    dict(id="e_r_f16_sigmoid_tanh_sum",   # f16 input, pointwise prologue fused into the reduce
         code="(lambda x: (torch.sigmoid(x) * torch.tanh(x)).sum(-1))"
              "(torch.randn(4, 64, dtype=torch.float16))",
         ref=lambda x: [(_sigmoid(_f64(x)) * np.tanh(_f64(x))).sum(-1).astype(np.float16)],
         rtol=1e-2, atol=1e-2, knobs=COOP_T128),

    # ---- category 4: fused (row reductions + epilogues) --------------------------------------
    *[dict(id=f"e_f_l2_{tag}",            # tests/compiler/e2e/test_reduce_coverage.py
           code="torch.sqrt((lambda t: (t * t).sum(dim=1, keepdim=True))(torch.randn(4, 1024)))",
           ref=lambda x: [np.sqrt((_f64(x) ** 2).sum(1, keepdims=True)).astype(np.float32)],
           rtol=1e-4, atol=1e-4, knobs=k)
      for tag, k in [("coop128", COOP_T128), ("coop32", COOP_T32), ("serial", SERIAL)]],
    *[dict(id=f"e_f_softmax_{tag}",
           code="torch.softmax(torch.randn(4, 1024), dim=-1)",
           ref=lambda x: [_softmax(_f64(x)).astype(np.float32)], rtol=1e-3, atol=1e-5, knobs=k)
      for tag, k in [("coop128", COOP_T128), ("coop32", COOP_T32), ("serial", SERIAL)]],
    dict(id="e_f_softmax_3d",             # tests/compiler/cli/test_compile.py
         code="torch.nn.functional.softmax(torch.randn(1,32,128), dim=-1)",
         ref=lambda x: [_softmax(_f64(x)).astype(np.float32)], rtol=1e-3, atol=1e-5),
    dict(id="e_f_softmax_f16",            # tests/compiler/passes/test_twisted_rewrite.py
         code="torch.softmax(torch.randn(4, 8, dtype=torch.float16), dim=-1)",
         ref=lambda x: [_softmax(_f64(x)).astype(np.float16)], rtol=1e-2, atol=1e-3),
    dict(id="e_f_rmsnorm_pow",            # tests/compiler/cli/test_run.py
         code="x = torch.randn(2,64,256); "
              "torch.mul(x, torch.pow(torch.mean(torch.pow(x,2),-1,keepdim=True)+1e-6, -0.5))",
         ref=lambda x: [(_f64(x) / np.sqrt((_f64(x) ** 2).mean(-1, keepdims=True) + 1e-6))
                        .astype(np.float32)], rtol=1e-3, atol=1e-4),
    dict(id="e_f_sub_rowmax_sum_f16",     # tests/compiler/passes/test_twisted_rewrite.py
         code="(torch.randn(64,128,dtype=torch.float16) - "
              "torch.randn(64,128,dtype=torch.float16).amax(-1, keepdim=True)).sum(-1, keepdim=True)",
         ref=lambda a, b: [(_f64(a) - _f64(b).max(-1, keepdims=True)).sum(-1, keepdims=True)
                           .astype(np.float16)], rtol=1e-2, atol=5e-1),
]