"""Kernel IR -> PTX compiler.

    python compile.py path/to/kernel.ir.json > out.ptx

How it is organised
-------------------
main() loads the IR and meta.json; PTXEmitter walks the kernel body and
appends PTX lines. There are exactly three extension points:

    IR statement  X   ->  method  do_X(self, stmt)
    IR expression X   ->  method  ex_X(self, expr)
    elementwise op    ->  one entry in F32_OPS

Adding support for something new means adding one of those; nothing else
has to change. An unknown kind raises a clear NotImplementedError.

Where values live
-----------------
    self.vals  IR value name   -> f32 register (%f)
    self.idx   axis/index name -> s32 register (%r) or a Python int (folded)
    self.ptr   buffer name     -> base address (%rd)

Registers are handed out fresh and never reused; ptxas does the real
allocation. Rebinding a name (Accum, or Assign to an existing name) writes
into the name's existing register, so loop-carried values just work.

Index arithmetic assumes non-negative operands for / and % (true for every
Emmy index), so it uses unsigned div/rem; ptxas turns power-of-two
constants into shifts and masks.
"""
import json
import math
import operator
import struct
import sys
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.kernel.ir import KernelOp

PTX_HEADER = ".version 8.0\n.target sm_89\n.address_size 64\n"


def f32_hex(x):
    """PTX float literal: 0f + the IEEE-754 bits in hex (exact, no rounding)."""
    return "0f%08X" % struct.unpack("<I", struct.pack("<f", float(x)))[0]


ZERO, ONE = f32_hex(0.0), f32_hex(1.0)
LOG2E, LN2 = f32_hex(math.log2(math.e)), f32_hex(math.log(2.0))


# =============================================================================
# Elementwise math on f32.
#   string   -> one-instruction template: {d} is the result, {0} {1} the args
#   function -> f(e, d, *args), built from other ops with e.op(...)
# =============================================================================
def _exp(e, d, x):                          # e^x = 2^(x * log2 e)
    t = e.tmp()
    e.emit(f"mul.f32 {t}, {x}, {LOG2E};", f"ex2.approx.ftz.f32 {d}, {t};")


def _log(e, d, x):                          # ln x = log2 x * ln 2
    e.emit(f"lg2.approx.f32 {d}, {x};", f"mul.f32 {d}, {d}, {LN2};")


def _sigmoid(e, d, x):                      # 1 / (1 + e^-x)
    t = e.tmp()
    e.op("negative", t, x)
    e.op("exp", t, t)
    e.op("add", t, t, ONE)
    e.op("reciprocal", d, t)


def _silu(e, d, x):                         # x * sigmoid(x)
    e.op("sigmoid", d, x)
    e.op("multiply", d, d, x)


F32_OPS = {
    "add":        "add.f32 {d}, {0}, {1};",
    "subtract":   "sub.f32 {d}, {0}, {1};",
    "multiply":   "mul.f32 {d}, {0}, {1};",
    "divide":     "div.rn.f32 {d}, {0}, {1};",
    "maximum":    "max.f32 {d}, {0}, {1};",
    "minimum":    "min.f32 {d}, {0}, {1};",
    "negative":   "neg.f32 {d}, {0};",
    "abs":        "abs.f32 {d}, {0};",
    "copy":       "mov.f32 {d}, {0};",
    "sqrt":       "sqrt.rn.f32 {d}, {0};",
    "rsqrt":      "rsqrt.approx.f32 {d}, {0};",
    "reciprocal": "rcp.rn.f32 {d}, {0};",
    "tanh":       "tanh.approx.f32 {d}, {0};",
    "relu":       lambda e, d, x: e.op("maximum", d, x, ZERO),
    "exp":        _exp,
    "exp_fast":   _exp,
    "log":        _log,
    "sigmoid":    _sigmoid,
    "silu":       _silu,
}

# Starting value of an accumulator, per reduce op. Max/min use +-1e30 like Emmy,
# not +-inf: the online-softmax combine computes `m_a - max(m_a, m_b)`, which is
# NaN for -inf - -inf but a harmless 0 for -1e30 - -1e30.
IDENTITY = {"add": 0.0, "multiply": 1.0, "maximum": -1e30, "minimum": 1e30}


# =============================================================================
# Integer index arithmetic. op -> (PTX instruction, Python constant fold)
# =============================================================================
INT_OPS = {
    "+": ("add.s32", operator.add),      "-": ("sub.s32", operator.sub),
    "*": ("mul.lo.s32", operator.mul),   "/": ("div.u32", operator.floordiv),
    "%": ("rem.u32", operator.mod),      "&": ("and.b32", operator.and_),
    "|": ("or.b32", operator.or_),       "<<": ("shl.b32", operator.lshift),
    ">>": ("shr.s32", operator.rshift),
}
CMP_OPS = {                              # -> setp.<cmp>.s32
    "==": ("eq", operator.eq), "!=": ("ne", operator.ne),
    "<": ("lt", operator.lt),  "<=": ("le", operator.le),
    ">": ("gt", operator.gt),  ">=": ("ge", operator.ge),
}
_and, _or = ("and.pred", lambda a, b: a and b), ("or.pred", lambda a, b: a or b)
LOGIC_OPS = {"&&": _and, "and": _and, "||": _or, "or": _or}
FOLD = {op: f for table in (INT_OPS, CMP_OPS, LOGIC_OPS) for op, (_, f) in table.items()}


# Small formatting helpers --------------------------------------------------
def is_reg(v):
    """expr() returns either a register name (str) or a folded Python constant."""
    return isinstance(v, str)


def mem(base, off=0):
    return f"[{base}+{off}]" if off else f"[{base}]"


def vec(regs):                              # "" or ".v2" / ".v4"
    return "" if len(regs) == 1 else f".v{len(regs)}"


def group(regs):                            # "%f1" or "{%f1, %f2}"
    return regs[0] if len(regs) == 1 else "{" + ", ".join(regs) + "}"


class PTXEmitter:
    def __init__(self, graph, kop, arg_order, launch_threads=None, smem_bytes=0):
        self.graph, self.kop, self.arg_order = graph, kop, arg_order
        self.launch_threads = launch_threads      # grid*block from meta.json
        self.dynamic_smem = smem_bytes > 0        # meta asks for a dynamic pool
        self.smem_offset = 0                      # next free byte in that pool
        self.lines = []                           # kernel body
        self.globals = []                         # module-level .shared decls
        self.count = {"p": 0, "r": 0, "rd": 0, "f": 0}
        self.labels = 0
        self.vals, self.idx, self.ptr = {}, {}, {}
        self.smem = {}                            # shared array name -> extents
        self.inits = set()                        # Init-declared names (never reseeded)

    # =========================================================================
    # Plumbing: registers, labels, output, dispatch
    # =========================================================================
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

    def dispatch(self, prefix, node):
        kind = type(node).__name__
        handler = getattr(self, prefix + kind, None)
        if handler is None:
            raise NotImplementedError(f"Kernel IR node {kind} (add {prefix}{kind})")
        return handler(node)

    def body(self, stmts):
        for s in stmts or ():
            self.dispatch("do_", s)

    def op(self, name, d, *args):
        """d = name(args) on f32 registers, using F32_OPS."""
        rule = F32_OPS[name]
        if callable(rule):
            rule(self, d, *args)
        else:
            self.emit(rule.format(*args, d=d))

    def bind(self, name, reg):
        """name = reg. Rebinding an existing name writes into its register."""
        if name in self.vals:
            self.emit(f"mov.f32 {self.vals[name]}, {reg};")
        else:
            self.vals[name] = reg

    # =========================================================================
    # Expressions -> Python int (folded), %r register, or %p predicate
    # =========================================================================
    def expr(self, e):
        return self.dispatch("ex_", e)

    def ex_Literal(self, e):
        return int(e.value)

    def ex_Var(self, e):
        return self.idx[e.name]

    def ex_TernaryExpr(self, e):                 # c ? a : b  ->  selp
        c = self.expr(e.cond)
        if not is_reg(c):
            return self.expr(e.if_true if c else e.if_false)
        a, b, d = self.expr(e.if_true), self.expr(e.if_false), self.reg("r")
        self.emit(f"selp.b32 {d}, {a}, {b}, {self.pred(c)};")
        return d

    def ex_BinaryExpr(self, e):
        l, r = self.expr(e.left), self.expr(e.right)
        if not is_reg(l) and not is_reg(r):
            return FOLD[e.op](l, r)
        if e.op in CMP_OPS:
            p = self.reg("p")
            self.emit(f"setp.{CMP_OPS[e.op][0]}.s32 {p}, {l}, {r};")
            return p
        if e.op in LOGIC_OPS:
            p = self.reg("p")
            self.emit(f"{LOGIC_OPS[e.op][0]} {p}, {self.pred(l)}, {self.pred(r)};")
            return p
        if (e.op, r) in (("+", 0), ("*", 1)):    # x+0, x*1
            return l
        d = self.reg("r")
        self.emit(f"{INT_OPS[e.op][0]} {d}, {l}, {r};")
        return d

    def pred(self, v):
        """Turn an expr() result into a predicate register."""
        if is_reg(v) and v.startswith("%p"):
            return v
        p = self.reg("p")
        self.emit(f"setp.ne.s32 {p}, {v if is_reg(v) else int(v)}, 0;")
        return p

    # =========================================================================
    # Memory: buf[index] -> address -> ld / st / red
    # =========================================================================
    def shape(self, buf):
        if buf in self.smem:
            return self.smem[buf]
        return tuple(int(d.value) for d in self.graph.nodes[buf].outputs[0].shape)

    def flat_index(self, buf, index):
        """Row-major flatten of an index tuple -> int or %r."""
        if len(index) == 1:
            return self.expr(index[0])
        shape, flat = self.shape(buf), 0
        for i, ix in enumerate(index):
            stride, term = math.prod(shape[i + 1:]), self.expr(ix)
            if not is_reg(term) and not is_reg(flat):
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
        """buf[index] (or buf[flat]) -> (state space, base register, byte offset)."""
        space = "shared" if buf in self.smem else "global"
        if flat is None:
            flat = self.flat_index(buf, index)
        if not is_reg(flat):                     # constant index: fold into the offset
            return space, self.ptr[buf], 4 * flat
        off, addr = self.reg("rd"), self.reg("rd")
        self.emit(f"mul.wide.s32 {off}, {flat}, 4;", f"add.s64 {addr}, {self.ptr[buf]}, {off};")
        return space, addr, 0

    def load(self, dsts, buf, index=None, flat=None):
        """One scalar load, or one vector load for `a, b, c, d = load x[i]`."""
        space, base, off = self.address(buf, index, flat)
        self.emit(f"ld.{space}{vec(dsts)}.f32 {group(dsts)}, {mem(base, off)};")

    def store(self, buf, index, vals, atomic=False, flat=None):
        """One scalar/vector store, or one red.add per element when atomic."""
        space, base, off = self.address(buf, index, flat)
        if atomic:
            for i, v in enumerate(vals):
                self.emit(f"red.{space}.add.f32 {mem(base, off + 4 * i)}, {v};")
        else:
            self.emit(f"st.{space}{vec(vals)}.f32 {mem(base, off)}, {group(vals)};")

    # =========================================================================
    # Statements: launch
    # =========================================================================
    def do_Tile(self, t):
        """gid = ctaid.x * ntid.x + tid.x; exit if gid >= N; decode the axes
        row-major from gid (innermost axis last)."""
        exts = [int(a.extent.value) for a in t.axes]
        n = math.prod(exts)
        tid, bid, bdim, gid, lane, warp = (self.reg("r") for _ in range(6))
        self.emit(f"mov.u32 {tid}, %tid.x;", f"mov.u32 {bid}, %ctaid.x;",
                  f"mov.u32 {bdim}, %ntid.x;", f"mad.lo.s32 {gid}, {bid}, {bdim}, {tid};",
                  f"and.b32 {lane}, {tid}, 31;", f"shr.u32 {warp}, {tid}, 5;")
        if self.launch_threads is None or self.launch_threads > n:   # surplus threads
            p = self.reg("p")
            self.emit(f"setp.ge.s32 {p}, {gid}, {n};", f"@{p} bra DONE;")
        self.idx.update(tid=tid, lane=lane, warp=warp)
        rest = gid
        for axis, ext in reversed(list(zip(t.axes, exts))):
            if ext == n:                         # the only non-trivial axis
                self.idx[axis.name] = rest
                continue
            a, q = self.reg("r"), self.reg("r")
            self.emit(f"rem.s32 {a}, {rest}, {ext};", f"div.s32 {q}, {rest}, {ext};")
            self.idx[axis.name], rest = a, q
        self.body(t.body)

    # =========================================================================
    # Statements: scalar compute
    # =========================================================================
    def do_Load(self, s):
        dsts = [self.tmp() for _ in s.names]
        self.load(dsts, s.input, s.index)
        for name, d in zip(s.names, dsts):
            self.bind(name, d)

    def do_Assign(self, s):
        d = self.tmp()
        self.op(s.op.name, d, *(self.vals[a] for a in s.args))
        self.bind(s.name, d)

    def do_Let(self, s):
        d = self.tmp()
        self.emit(f"mov.f32 {d}, {f32_hex(s.value.value)};")
        self.bind(s.name, d)

    def do_Init(self, s):
        """Carried state seeded with a literal (matmul accumulator tiles)."""
        d = self.tmp()
        self.emit(f"mov.f32 {d}, {f32_hex(s.identity)};")
        self.bind(s.name, d)
        self.inits.add(s.name)

    def do_Write(self, s):
        self.store(s.output, s.index, [self.vals[v] for v in s.values], s.atomic)

    # =========================================================================
    # Statements: control flow
    # =========================================================================
    def loop(self, axis, start, end, step, body):
        """for (axis = start; axis < end; axis += step) body"""
        i, p = self.reg("r"), self.reg("p")
        top, done = self.label("LOOP"), self.label("LOOP_END")
        self.emit(f"mov.s32 {i}, {start};", f"{top}:",
                  f"setp.ge.s32 {p}, {i}, {end};", f"@{p} bra {done};")
        self.idx[axis] = i
        self.body(body)
        self.emit(f"add.s32 {i}, {i}, {step};", f"bra {top};", f"{done}:")

    def do_StridedLoop(self, s):
        self.seed_accums(s)
        end = self.expr(s.end) if s.end is not None else int(s.axis.extent.value)
        self.loop(s.axis.name, self.expr(s.start), end, self.expr(s.step), s.body)

    def do_Loop(self, s):
        """for axis in 0..extent. Unrolled when the IR asks and the loop is
        innermost (matmul's k loop): the axis becomes a Python int and every
        index constant-folds. Outer loops stay rolled even if marked unroll:
        attention's key loop wraps four 64-step dot products, and unrolling it
        exhausts registers."""
        self.seed_accums(s)
        n = int(s.axis.extent.value)
        innermost = not any(type(c).__name__ in ("Loop", "StridedLoop") for c in s.body)
        if s.unroll and innermost and n <= 128:
            for i in range(n):
                self.idx[s.axis.name] = i
                self.body(s.body)
        else:
            self.loop(s.axis.name, 0, n, 1, s.body)

    def do_Cond(self, s):
        c = self.expr(s.cond)
        if not is_reg(c):                        # constant condition
            self.body(s.body if c else s.else_body)
            return
        end = self.label("ENDIF")
        els = self.label("ELSE") if s.else_body else end
        self.emit(f"@!{self.pred(c)} bra {els};")
        self.body(s.body)
        if s.else_body:
            self.emit(f"bra {end};", f"{els}:")
            self.body(s.else_body)
        self.emit(f"{end}:")

    # =========================================================================
    # Statements: reductions
    # =========================================================================
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
                    self.vals[s.name] = self.tmp()
                acc = self.vals[s.name]
                self.emit(f"mov.f32 {acc}, {f32_hex(IDENTITY[s.op.name])};")

    def do_Accum(self, s):
        """acc = op(base, value) in place; base defaults to acc itself. The online
        softmax uses base for the rescaled old sum: acc = acc*alpha + x*beta."""
        acc = self.vals[s.name]
        self.op(s.op.name, acc, self.vals[s.base] if s.base else acc, self.vals[s.value])

    def combine(self, s, others):
        """Bind the state_b names to `others`, then run the combine statements."""
        self.vals.update(zip(s.state_b, others))
        self.body(s.combine_states)

    def do_WarpShuffle(self, s):
        """XOR butterfly over `length` lanes: offsets length/2, ..., 1."""
        off = s.length // 2
        while off >= 1:
            others = [self.tmp() for _ in s.state]
            for st, o in zip(s.state, others):
                self.emit(f"shfl.sync.bfly.b32 {o}, {self.vals[st]}, {off}, 0x1f, 0xffffffff;")
            self.combine(s, others)
            off //= 2

    def do_TreeHalve(self, s):
        """Shared-memory tree over `length` slots, then broadcast slot 0:

            for (h = length/2; h > 0; h /= 2) {          // unrolled: length is static
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
        while h >= 1:
            p, partner, skip = self.reg("p"), self.reg("r"), self.label("TREE_SKIP")
            self.emit(f"setp.ge.s32 {p}, {tid}, {h};", f"@{p} bra {skip};",
                      f"add.s32 {partner}, {tid}, {h};")
            others = []
            for buf, st in zip(s.bufs, s.state):
                others.append(self.tmp())
                self.load([self.vals[st]], buf, flat=tid)
                self.load([others[-1]], buf, flat=partner)
            if s.tid_var == "warp":
                # All 32 lanes share one slot, so every lane must load before any
                # lane stores. Converged warps do this anyway; make it explicit so
                # it also holds under independent thread scheduling.
                self.emit("bar.warp.sync -1;")
            self.combine(s, others)
            for buf, st in zip(s.bufs, s.state):
                self.store(buf, None, [self.vals[st]], flat=tid)
            self.emit(f"{skip}:", bar)
            h //= 2
        for buf, st in zip(s.bufs, s.state):     # broadcast the result
            self.load([self.vals[st]], buf, flat=0)

    # =========================================================================
    # Statements: shared memory, barriers, async copies
    # =========================================================================
    def do_Smem(self, s):
        """Static `.shared` array, or a slice of the dynamic pool when meta.json
        sets smem_bytes (needed above the 48 KB static limit, e.g. m01)."""
        extents = tuple(int(x) for x in s.extents)
        align = max(int(s.align or 0), 4)
        self.smem[s.name] = extents
        base = self.ptr[s.name] = self.reg("rd")
        if self.dynamic_smem:
            self.smem_offset = -(-self.smem_offset // align) * align   # round up
            self.emit(f"mov.u64 {base}, smem_pool;",
                      f"add.u64 {base}, {base}, {self.smem_offset};")
            self.smem_offset += 4 * math.prod(extents)
        else:
            self.globals.append(f".shared .align {align} .f32 {s.name}[{math.prod(extents)}];")
            self.emit(f"mov.u64 {base}, {s.name};")

    def do_Sync(self, s):
        if s.warp:
            self.emit("bar.warp.sync -1;")
        elif s.count:
            self.emit(f"bar.sync {s.barrier_id}, {s.count};")
        else:
            self.emit(f"bar.sync {s.barrier_id};")

    def do_CpAsyncCopy(self, s):
        """smem[smem_index + lane_index] <- src[src_index], nbytes at once.
        Both shared-side indices flatten against the smem extents and add."""
        if s.swizzle not in (None, "NONE"):
            raise NotImplementedError("CpAsyncCopy swizzle")
        flat = self.flat_index(s.smem, s.smem_index)
        if s.lane_index:
            lane = self.flat_index(s.smem, s.lane_index)
            if not is_reg(flat) and not is_reg(lane):
                flat += lane
            else:
                t = self.reg("r")
                self.emit(f"add.s32 {t}, {flat}, {lane};")
                flat = t
        _, dbase, doff = self.address(s.smem, flat=flat)
        _, sbase, soff = self.address(s.src, s.src_index)
        cache = "cg" if s.nbytes == 16 else "ca"   # .cg only supports 16 bytes
        self.emit(f"cp.async.{cache}.shared.global {mem(dbase, doff)}, {mem(sbase, soff)}, {s.nbytes};")

    def do_CpAsyncCommit(self, s):
        self.emit("cp.async.commit_group;")

    def do_CpAsyncWait(self, s):
        self.emit(f"cp.async.wait_group {s.group};")

    # =========================================================================
    # Whole kernel
    # =========================================================================
    def compile(self):
        for n in self.arg_order:                 # params -> global pointers
            raw = self.reg("rd")
            self.ptr[n] = self.reg("rd")
            self.emit(f"ld.param.u64 {raw}, [param_{n}];",
                      f"cvta.to.global.u64 {self.ptr[n]}, {raw};")
        self.body(self.kop.body)

        if self.dynamic_smem and self.smem:
            self.globals.insert(0, ".extern .shared .align 16 .b8 smem_pool[];")
        params = ",\n".join(f"\t.param .u64 param_{n}" for n in self.arg_order)
        decls = [f"\t.reg .{t} %{k}<{self.count[k] + 1}>;"
                 for k, t in (("p", "pred"), ("r", "b32"), ("rd", "b64"), ("f", "f32"))]
        return "\n".join([PTX_HEADER, *self.globals, "",
                          f".visible .entry {self.kop.name}(", params, ")", "{",
                          *decls, "", *self.lines, "DONE:", "\tret;", "}", ""])


# =============================================================================
# Driver
# =============================================================================
def launch_info(ir_path, entry):
    """(arg_order, threads launched, dynamic smem bytes) from meta.json."""
    meta = ir_path.parent / "meta.json"
    if meta.exists():
        for k in json.loads(meta.read_text())["kernels"]:
            if k["entry"] == entry:
                threads = math.prod(k["grid"]) * math.prod(k["block"])
                return k["arg_order"], threads, k.get("smem_bytes", 0)
    return None, None, 0


def main(path):
    path = Path(path)
    graph = Graph.from_dict(json.loads(path.read_text()))
    node = next(n for n in graph.nodes.values() if isinstance(n.op, KernelOp))
    arg_order, threads, smem_bytes = launch_info(path, node.op.name)
    arg_order = arg_order or [*node.inputs, node.id]     # fallback: inputs, then output
    sys.stdout.write(PTXEmitter(graph, node.op, arg_order, threads, smem_bytes).compile())


if __name__ == "__main__":
    main(sys.argv[1])