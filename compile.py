"""Kernel IR -> PTX compiler (phases 1-5: pointwise, reductions, matmul, fused, attention).

    python compile.py path/to/kernel.ir.json > out.ptx

One class, `PTXEmitter`, walks the KernelOp body and has one `do_<Kind>`
method per IR statement kind, plus a recursive expression compiler.

Conventions
- Every IR value name maps to one f32 register (`self.vals`). Rebinding a
  name (an Accum, or an Assign to an existing name) writes into that same
  register, so loop-carried values and values updated inside a branch work.
- Index/axis values are s32 registers or Python ints (constant-folded).
- Registers are handed out fresh; ptxas does the real register allocation.
- Index arithmetic assumes non-negative operands for / and % (true for every
  Emmy index), so it uses unsigned div/rem, which ptxas turns into shifts and
  masks for power-of-two constants.
"""
import json
import struct
import sys
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.kernel.ir import KernelOp

PTX_HEADER = ".version 8.0\n.target sm_89\n.address_size 64\n"


def f32_hex(x):
    """PTX float literal: 0f + IEEE-754 bits in hex (exact, no rounding)."""
    return "0f%08X" % struct.unpack("<I", struct.pack("<f", float(x)))[0]


# ---------------------------------------------------------------------------
# Elementwise ops on f32: name -> (dst, [args], emitter) -> PTX lines.
# `e.tmp()` gives a fresh f32 register for ops that need intermediates.
# ---------------------------------------------------------------------------
LOG2E = f32_hex(1.4426950408889634)
LN2 = f32_hex(0.6931471805599453)
ONE, ZERO = f32_hex(1.0), f32_hex(0.0)


def _exp(d, a, e):
    t = e.tmp()
    return [f"mul.f32 {t}, {a}, {LOG2E};", f"ex2.approx.ftz.f32 {d}, {t};"]


def _sigmoid(d, a, e):  # 1 / (1 + exp(-x))
    t = e.tmp()
    return [f"neg.f32 {t}, {a};", *_exp(t, t, e),
            f"add.f32 {t}, {t}, {ONE};", f"rcp.rn.f32 {d}, {t};"]


F32_OPS = {
    "add":      lambda d, a, e: [f"add.f32 {d}, {a[0]}, {a[1]};"],
    "subtract": lambda d, a, e: [f"sub.f32 {d}, {a[0]}, {a[1]};"],
    "multiply": lambda d, a, e: [f"mul.f32 {d}, {a[0]}, {a[1]};"],
    "divide":   lambda d, a, e: [f"div.rn.f32 {d}, {a[0]}, {a[1]};"],
    "maximum":  lambda d, a, e: [f"max.f32 {d}, {a[0]}, {a[1]};"],
    "minimum":  lambda d, a, e: [f"min.f32 {d}, {a[0]}, {a[1]};"],
    "negative": lambda d, a, e: [f"neg.f32 {d}, {a[0]};"],
    "abs":      lambda d, a, e: [f"abs.f32 {d}, {a[0]};"],
    "copy":     lambda d, a, e: [f"mov.f32 {d}, {a[0]};"],
    "relu":     lambda d, a, e: [f"max.f32 {d}, {a[0]}, {ZERO};"],
    "sqrt":     lambda d, a, e: [f"sqrt.rn.f32 {d}, {a[0]};"],
    "rsqrt":    lambda d, a, e: [f"rsqrt.approx.f32 {d}, {a[0]};"],
    "reciprocal": lambda d, a, e: [f"rcp.rn.f32 {d}, {a[0]};"],
    "tanh":     lambda d, a, e: [f"tanh.approx.f32 {d}, {a[0]};"],
    "exp":      lambda d, a, e: _exp(d, a[0], e),
    "exp_fast": lambda d, a, e: _exp(d, a[0], e),
    "log":      lambda d, a, e: [f"lg2.approx.f32 {d}, {a[0]};", f"mul.f32 {d}, {d}, {LN2};"],
    "sigmoid":  lambda d, a, e: _sigmoid(d, a[0], e),
    "silu":     lambda d, a, e: [*_sigmoid(d, a[0], e), f"mul.f32 {d}, {d}, {a[0]};"],
}

# Starting value of an accumulator, per reduce op. Max/min use +-1e30 like Emmy,
# not +-inf: the online-softmax combine computes `m_a - max(m_a, m_b)`, which is
# NaN for -inf - -inf but a harmless 0 for -1e30 - -1e30.
IDENTITY = {"add": 0.0, "multiply": 1.0, "maximum": -1e30, "minimum": 1e30}

# Integer index arithmetic (s32) and comparisons (-> predicate).
INT_OPS = {"+": "add.s32", "-": "sub.s32", "*": "mul.lo.s32", "/": "div.u32", "%": "rem.u32",
           "&": "and.b32", "|": "or.b32", "<<": "shl.b32", ">>": "shr.s32"}
CMP_OPS = {"==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}
PRED_OPS = {"&&": "and.pred", "||": "or.pred", "and": "and.pred", "or": "or.pred"}
FOLD = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
        "/": lambda a, b: int(a / b), "%": lambda a, b: a % b, "&": lambda a, b: a & b,
        "|": lambda a, b: a | b, "<<": lambda a, b: a << b, ">>": lambda a, b: a >> b,
        "==": lambda a, b: a == b, "!=": lambda a, b: a != b, "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
        "&&": lambda a, b: a and b, "||": lambda a, b: a or b}


class PTXEmitter:
    def __init__(self, graph, kop, arg_order, launch_threads=None, smem_bytes=0):
        self.graph, self.kop, self.arg_order = graph, kop, arg_order
        self.launch_threads = launch_threads             # grid*block from meta.json
        self.dynamic_smem = smem_bytes > 0               # meta asks for a dynamic pool
        self.smem_offset = 0                             # next free byte in that pool
        self.lines = []                                  # kernel body
        self.globals = []                                # module-level decls (.shared)
        self.count = {"r": 0, "rd": 0, "f": 0, "p": 0}  # fresh-register counters
        self.labels = 0
        self.vals = {}                                   # IR value name -> f32 register
        self.idx = {}                                    # axis/index name -> s32 register
        self.ptr = {}                                    # buffer name -> base address (%rd)
        self.smem = {}                                   # shared array name -> extents
        self.inits = set()                               # names declared by Init (never reseeded)

    # ---------- small helpers ----------
    def reg(self, kind):
        self.count[kind] += 1
        return f"%{kind}{self.count[kind]}"

    def tmp(self):
        return self.reg("f")

    def label(self, tag):
        self.labels += 1
        return f"{tag}_{self.labels}"

    def emit(self, *lines):
        self.lines.extend(l if l.endswith(":") else "\t" + l for l in lines)

    def shape(self, buf):
        if buf in self.smem:
            return self.smem[buf]
        return tuple(int(d.value) for d in self.graph.nodes[buf].outputs[0].shape)

    def bind(self, name, value_reg):
        """name = value. Rebinding an existing name writes into its register."""
        if name in self.vals:
            self.emit(f"mov.f32 {self.vals[name]}, {value_reg};")
        else:
            self.vals[name] = value_reg

    # ---------- expressions: Var / Literal / BinaryExpr ----------
    # Returns a Python int/bool (constant), an s32 register, or a %p predicate.
    def expr(self, e):
        kind = type(e).__name__
        if kind == "Literal":
            return int(e.value)
        if kind == "Var":
            return self.idx[e.name]
        if kind == "TernaryExpr":                        # c ? a : b  ->  selp
            c = self.expr(e.cond)
            if not isinstance(c, str):
                return self.expr(e.if_true if c else e.if_false)
            a, b, d = self.expr(e.if_true), self.expr(e.if_false), self.reg("r")
            self.emit(f"selp.b32 {d}, {a}, {b}, {self.pred(c)};")
            return d
        if kind != "BinaryExpr":
            raise NotImplementedError(f"expression {kind}")
        l, r = self.expr(e.left), self.expr(e.right)
        if not isinstance(l, str) and not isinstance(r, str):
            return FOLD[e.op](l, r)                      # constant-fold
        if e.op in CMP_OPS:
            p = self.reg("p")
            self.emit(f"setp.{CMP_OPS[e.op]}.s32 {p}, {l}, {r};")
            return p
        if e.op in PRED_OPS:
            p = self.reg("p")
            self.emit(f"{PRED_OPS[e.op]} {p}, {self.pred(l)}, {self.pred(r)};")
            return p
        if e.op in ("+", "*") and r == (0 if e.op == "+" else 1):
            return l                                     # x+0, x*1
        d = self.reg("r")
        self.emit(f"{INT_OPS[e.op]} {d}, {l}, {r};")
        return d

    def pred(self, v):
        """Turn an expr() result into a predicate register."""
        if isinstance(v, str) and v.startswith("%p"):
            return v
        p = self.reg("p")
        self.emit(f"setp.ne.s32 {p}, {int(v) if not isinstance(v, str) else v}, 0;")
        return p

    # ---------- addressing ----------
    def flat_index(self, buf, index):
        """Row-major flatten of an index tuple (possibly empty) -> s32 operand."""
        if len(index) == 1:
            return self.expr(index[0])
        shape, flat = self.shape(buf), 0
        for i, ix in enumerate(index):
            stride = 1
            for s in shape[i + 1:]:
                stride *= s
            term = self.expr(ix)
            if isinstance(term, int) and isinstance(flat, int):
                flat += term * stride
                continue
            t = self.reg("r")
            if flat == 0:
                self.emit(f"mul.lo.s32 {t}, {term}, {stride};" if stride != 1 else f"mov.s32 {t}, {term};")
            elif stride == 1:
                self.emit(f"add.s32 {t}, {term}, {flat};")
            else:
                self.emit(f"mad.lo.s32 {t}, {term}, {stride}, {flat};")
            flat = t
        return flat

    def address(self, buf, index=None, flat=None):
        """buf[index] (or buf[flat]) -> ('[base+off]', state space)."""
        space = "shared" if buf in self.smem else "global"
        if flat is None:
            flat = self.flat_index(buf, index)
        if isinstance(flat, int):
            return f"[{self.ptr[buf]}+{flat * 4}]", space
        off, addr = self.reg("rd"), self.reg("rd")
        self.emit(f"mul.wide.s32 {off}, {flat}, 4;", f"add.s64 {addr}, {self.ptr[buf]}, {off};")
        return f"[{addr}]", space

    def load(self, dsts, buf, index=None, flat=None):
        addr, space = self.address(buf, index, flat)
        if len(dsts) == 1:
            self.emit(f"ld.{space}.f32 {dsts[0]}, {addr};")
        else:  # a, b, c, d = load x[i]  ->  one vector load
            self.emit(f"ld.{space}.v{len(dsts)}.f32 {{{', '.join(dsts)}}}, {addr};")

    def store(self, buf, index, vals, atomic=False, flat=None):
        addr, space = self.address(buf, index, flat)
        if atomic:  # accumulate: one red.add per element
            base, off = addr[1:-1].split("+") if "+" in addr else (addr[1:-1], "0")
            for i, v in enumerate(vals):
                self.emit(f"red.{space}.add.f32 [{base}+{int(off) + 4 * i}], {v};")
        elif len(vals) == 1:
            self.emit(f"st.{space}.f32 {addr}, {vals[0]};")
        else:  # buf[i] = (v0, v1[, v2, v3])  ->  one vector store
            self.emit(f"st.{space}.v{len(vals)}.f32 {addr}, {{{', '.join(vals)}}};")

    # ---------- statements ----------
    def body(self, stmts):
        for s in stmts or ():
            getattr(self, "do_" + type(s).__name__)(s)

    def do_Tile(self, t):
        """gid = ctaid.x * ntid.x + tid.x; guard gid < N; decode axes row-major."""
        exts = [int(a.extent.value) for a in t.axes]
        n = 1
        for x in exts:
            n *= x
        tid, gid, p = self.reg("r"), self.reg("r"), self.reg("p")
        lane, warp, bid, bdim = (self.reg("r") for _ in range(4))
        self.emit(f"mov.u32 {tid}, %tid.x;", f"mov.u32 {bid}, %ctaid.x;",
                  f"mov.u32 {bdim}, %ntid.x;", f"mad.lo.s32 {gid}, {bid}, {bdim}, {tid};",
                  f"and.b32 {lane}, {tid}, 31;", f"shr.u32 {warp}, {tid}, 5;")
        if self.launch_threads is None or self.launch_threads > n:  # extra threads: guard
            self.emit(f"setp.ge.s32 {p}, {gid}, {n};", f"@{p} bra DONE;")
        self.idx.update(tid=tid, lane=lane, warp=warp)
        rest = gid
        for axis, ext in reversed(list(zip(t.axes, exts))):  # innermost axis is last
            if ext == n:
                self.idx[axis.name] = rest
                continue
            a, q = self.reg("r"), self.reg("r")
            self.emit(f"rem.s32 {a}, {rest}, {ext};", f"div.s32 {q}, {rest}, {ext};")
            self.idx[axis.name], rest = a, q
        self.body(t.body)

    def do_Load(self, s):
        dst = [self.reg("f") for _ in s.names]
        self.load(dst, s.input, s.index)
        for n, d in zip(s.names, dst):
            self.bind(n, d)

    def do_Assign(self, s):
        d = self.reg("f")
        self.emit(*F32_OPS[s.op.name](d, [self.vals[a] for a in s.args], self))
        self.bind(s.name, d)

    def do_Let(self, s):
        d = self.reg("f")
        self.emit(f"mov.f32 {d}, {f32_hex(s.value.value)};")
        self.bind(s.name, d)

    def do_Write(self, s):
        self.store(s.output, s.index, [self.vals[v] for v in s.values], s.atomic)

    # ----- phase 2: reductions -----
    def accums(self, stmts):
        """All Accum statements under `stmts`, recursively."""
        for s in stmts or ():
            if type(s).__name__ == "Accum":
                yield s
            yield from self.accums(getattr(s, "body", None))
            yield from self.accums(getattr(s, "else_body", None))

    def seed_accums(self, loop):
        """A loop with seed=True resets, just before it starts, every accumulator
        that reduces over its axis. Nested loops therefore restart an inner
        accumulator on each outer iteration (attention's q.k dot product), while
        Init-declared carried state (matmul tiles) is never reset."""
        if not loop.seed:
            return
        for s in self.accums(loop.body):
            if loop.axis.name in s.axes and s.name not in self.inits:
                if s.name not in self.vals:
                    self.vals[s.name] = self.reg("f")
                self.emit(f"mov.f32 {self.vals[s.name]}, {f32_hex(IDENTITY[s.op.name])};")

    def do_Accum(self, s):
        """acc = op(base, value) in place; base defaults to acc itself. The online
        softmax uses base for the rescaled old sum: acc = acc*alpha + x*beta."""
        acc = self.vals[s.name]
        base = self.vals[s.base] if s.base else acc
        self.emit(*F32_OPS[s.op.name](acc, [base, self.vals[s.value]], self))

    def do_StridedLoop(self, s):
        """for (i = start; i < extent; i += step) body"""
        self.seed_accums(s)
        end = self.expr(s.end) if s.end is not None else int(s.axis.extent.value)
        i, p = self.reg("r"), self.reg("p")
        top, done = self.label("LOOP"), self.label("LOOP_END")
        self.emit(f"mov.s32 {i}, {self.expr(s.start)};", f"{top}:",
                  f"setp.ge.s32 {p}, {i}, {end};", f"@{p} bra {done};")
        self.idx[s.axis.name] = i
        self.body(s.body)
        self.emit(f"add.s32 {i}, {i}, {self.expr(s.step)};", f"bra {top};", f"{done}:")

    def do_Cond(self, s):
        c = self.expr(s.cond)
        if not isinstance(c, str):                       # constant condition
            self.body(s.body if c else s.else_body)
            return
        p = self.pred(c)
        if not s.else_body:
            end = self.label("ENDIF")
            self.emit(f"@!{p} bra {end};")
            self.body(s.body)
            self.emit(f"{end}:")
            return
        els, end = self.label("ELSE"), self.label("ENDIF")
        self.emit(f"@!{p} bra {els};")
        self.body(s.body)
        self.emit(f"bra {end};", f"{els}:")
        self.body(s.else_body)
        self.emit(f"{end}:")

    def do_Smem(self, s):
        """Static `.shared` array, or a slice of the dynamic pool when meta.json
        sets smem_bytes (needed above the 48 KB static limit, e.g. m01)."""
        n = 1
        for x in s.extents:
            n *= int(x)
        align = max(int(s.align or 0), 4)
        self.smem[s.name] = tuple(int(x) for x in s.extents)
        self.ptr[s.name] = self.reg("rd")
        if self.dynamic_smem:
            if not self.globals:
                self.globals.append(".extern .shared .align 16 .b8 smem_pool[];")
            self.smem_offset = -(-self.smem_offset // align) * align   # round up
            self.emit(f"mov.u64 {self.ptr[s.name]}, smem_pool;",
                      f"add.u64 {self.ptr[s.name]}, {self.ptr[s.name]}, {self.smem_offset};")
            self.smem_offset += 4 * n
        else:
            self.globals.append(f".shared .align {align} .f32 {s.name}[{n}];")
            self.emit(f"mov.u64 {self.ptr[s.name]}, {s.name};")

    def do_Sync(self, s):
        if s.warp:
            self.emit("bar.warp.sync -1;")
        elif s.count:
            self.emit(f"bar.sync {s.barrier_id}, {s.count};")
        else:
            self.emit(f"bar.sync {s.barrier_id};")

    def combine(self, states, states_b, others, combine_states):
        """Bind state_b names to `others`, then run the combine Assigns."""
        for nb, o in zip(states_b, others):
            self.vals[nb] = o
        self.body(combine_states)

    def do_WarpShuffle(self, s):
        """XOR butterfly over `length` lanes: offsets length/2, ..., 1."""
        off = s.length // 2
        while off >= 1:
            others = [self.reg("f") for _ in s.state]
            for st, o in zip(s.state, others):
                self.emit(f"shfl.sync.bfly.b32 {o}, {self.vals[st]}, {off}, 0x1f, 0xffffffff;")
            self.combine(s.state, s.state_b, others, s.combine_states)
            off //= 2

    def do_TreeHalve(self, s):
        """Shared-memory tree over `length` slots, then broadcast slot 0:

            for (h = length/2; h > 0; h /= 2) {
                if (tid < h) { st = buf[tid]; st_b = buf[tid+h]; combine; buf[tid] = st; }
                barrier
            }
            st = buf[0]
        """
        if s.inner:
            raise NotImplementedError("TreeHalve.inner")
        tid = self.idx[s.tid_var]
        bar = f"bar.sync {s.barrier_id}" + (f", {s.barrier_count}" if s.barrier_count else "") + ";"
        h = s.length // 2
        while h >= 1:                                    # fully unrolled: length is static
            p, partner, skip = self.reg("p"), self.reg("r"), self.label("TREE_SKIP")
            self.emit(f"setp.ge.s32 {p}, {tid}, {h};", f"@{p} bra {skip};",
                      f"add.s32 {partner}, {tid}, {h};")
            others = []
            for buf, st in zip(s.bufs, s.state):
                o = self.reg("f")
                self.load([self.vals[st]], buf, flat=tid)
                self.load([o], buf, flat=partner)
                others.append(o)
            if s.tid_var == "warp":
                # all 32 lanes share one slot: every lane must load before any lane
                # stores. Converged warps do this anyway; make it explicit so it holds
                # under independent thread scheduling too.
                self.emit("bar.warp.sync -1;")
            self.combine(s.state, s.state_b, others, s.combine_states)
            for buf, st in zip(s.bufs, s.state):
                self.store(buf, None, [self.vals[st]], flat=tid)
            self.emit(f"{skip}:", bar)
            h //= 2
        for buf, st in zip(s.bufs, s.state):              # broadcast the result
            self.load([self.vals[st]], buf, flat=0)

    # ----- phase 3: matmul -----
    def do_Init(self, s):
        """Carried state seeded with a literal (matmul accumulator tiles)."""
        d = self.reg("f")
        self.emit(f"mov.f32 {d}, {f32_hex(s.identity)};")
        self.bind(s.name, d)
        self.inits.add(s.name)

    def do_Loop(self, s):
        """for axis in 0..extent. Unrolled when the IR asks and the loop is innermost
        (matmul's k-loop): the axis becomes a Python int, so every index constant-
        folds. An outer loop stays rolled even if marked unroll: attention's key
        loop wraps four 64-step dot products, and unrolling it exhausts registers."""
        self.seed_accums(s)
        n = int(s.axis.extent.value)
        innermost = not any(type(c).__name__ in ("Loop", "StridedLoop") for c in s.body)
        if s.unroll and innermost and n <= 128:
            for i in range(n):
                self.idx[s.axis.name] = i
                self.body(s.body)
            return
        i, p = self.reg("r"), self.reg("p")
        top, done = self.label("LOOP"), self.label("LOOP_END")
        self.emit(f"mov.s32 {i}, 0;", f"{top}:", f"setp.ge.s32 {p}, {i}, {n};", f"@{p} bra {done};")
        self.idx[s.axis.name] = i
        self.body(s.body)
        self.emit(f"add.s32 {i}, {i}, 1;", f"bra {top};", f"{done}:")

    def do_CpAsyncCopy(self, s):
        """smem[smem_index + lane_index] <- src[src_index], nbytes at once.
        Both shared-side indices flatten against the smem extents and add."""
        if s.swizzle not in (None, "NONE"):
            raise NotImplementedError("CpAsyncCopy swizzle")
        flat = self.flat_index(s.smem, s.smem_index)
        if s.lane_index:
            lane = self.flat_index(s.smem, s.lane_index)
            if isinstance(flat, int) and isinstance(lane, int):
                flat += lane
            else:
                t = self.reg("r")
                self.emit(f"add.s32 {t}, {flat}, {lane};")
                flat = t
        dst, _ = self.address(s.smem, flat=flat)
        src, _ = self.address(s.src, s.src_index)
        cache = "cg" if s.nbytes == 16 else "ca"        # .cg only supports 16 bytes
        self.emit(f"cp.async.{cache}.shared.global {dst}, {src}, {s.nbytes};")

    def do_CpAsyncCommit(self, s):
        self.emit("cp.async.commit_group;")

    def do_CpAsyncWait(self, s):
        self.emit(f"cp.async.wait_group {s.group};")

    # ---------- whole kernel ----------
    def compile(self):
        params = [f"\t.param .u64 param_{n}" for n in self.arg_order]
        for n in self.arg_order:                         # prologue: param -> global ptr
            raw, g = self.reg("rd"), self.reg("rd")
            self.emit(f"ld.param.u64 {raw}, [param_{n}];", f"cvta.to.global.u64 {g}, {raw};")
            self.ptr[n] = g
        self.body(self.kop.body)
        decls = [f"\t.reg .{t} %{k}<{self.count[k] + 1}>;"
                 for k, t in (("p", "pred"), ("r", "b32"), ("rd", "b64"), ("f", "f32"))]
        return "\n".join([PTX_HEADER, *self.globals, "", f".visible .entry {self.kop.name}(",
                          ",\n".join(params), ")", "{", *decls, "", *self.lines,
                          "DONE:", "\tret;", "}", ""])


def main(path):
    path = Path(path)
    graph = Graph.from_dict(json.loads(path.read_text()))
    kop = next(n.op for n in graph.nodes.values() if isinstance(n.op, KernelOp))
    meta_file = path.parent / "meta.json"
    arg_order, launch_threads, smem_bytes = None, None, 0
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        k = next((k for k in meta["kernels"] if k["entry"] == kop.name), None)
        if k:
            arg_order, smem_bytes = k["arg_order"], k.get("smem_bytes", 0)
            launch_threads = 1
            for x in (*k["grid"], *k["block"]):
                launch_threads *= x
    if arg_order is None:  # fallback: node inputs then outputs
        node = next(n for n in graph.nodes.values() if n.op is kop)
        arg_order = list(node.inputs) + [node.id]
    sys.stdout.write(PTXEmitter(graph, kop, arg_order, launch_threads, smem_bytes).compile())


if __name__ == "__main__":
    main(sys.argv[1])