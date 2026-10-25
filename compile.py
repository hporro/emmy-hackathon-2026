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
from __future__ import annotations

import json
import math
import operator
import struct
import sys
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence, TypeAlias, TypeGuard, Union

from emmy.compiler.graph import Graph
from emmy.compiler.ir.kernel.ir import KernelOp

# =============================================================================
# Type aliases
# =============================================================================
Reg: TypeAlias = str                     # PTX register name: "%f3", "%r7", "%rd2", "%p1"
Const: TypeAlias = Union[int, bool]      # constant-folded index value (comparisons fold to bool)
IdxValue: TypeAlias = Union[Reg, Const]  # what expr() returns: a register or a folded constant
Stmt: TypeAlias = Any                    # Kernel IR statement node (Tile, Load, Loop, ...)
Expr: TypeAlias = Any                    # Kernel IR expression node (Literal, Var, BinaryExpr, ...)
Body: TypeAlias = Optional[Sequence[Stmt]]
F32Fn: TypeAlias = Callable[..., None]   # f(emitter, d, *args) -> emits PTX
F32Rule: TypeAlias = Union[str, F32Fn]
Address: TypeAlias = tuple[str, Reg, int]  # (state space, base register, byte offset)

PTX_HEADER: str = ".version 8.0\n.target sm_89\n.address_size 64\n"


def f32_hex(x: float) -> str:
    """PTX float literal: 0f + the IEEE-754 bits in hex (exact, no rounding)."""
    return "0f%08X" % struct.unpack("<I", struct.pack("<f", float(x)))[0]


ZERO: str
ONE: str
LOG2E: str
LN2: str
ZERO, ONE = f32_hex(0.0), f32_hex(1.0)
LOG2E, LN2 = f32_hex(math.log2(math.e)), f32_hex(math.log(2.0))


# =============================================================================
# Elementwise math on f32.
#   string   -> one-instruction template: {d} is the result, {0} {1} the args
#   function -> f(e, d, *args), built from other ops with e.op(...)
# =============================================================================
def _exp(e: PTXEmitter, d: Reg, x: Reg) -> None:          # e^x = 2^(x * log2 e)
    t: Reg = e.tmp()
    e.emit(f"mul.f32 {t}, {x}, {LOG2E};", f"ex2.approx.ftz.f32 {d}, {t};")


def _log(e: PTXEmitter, d: Reg, x: Reg) -> None:          # ln x = log2 x * ln 2
    e.emit(f"lg2.approx.f32 {d}, {x};", f"mul.f32 {d}, {d}, {LN2};")


def _sigmoid(e: PTXEmitter, d: Reg, x: Reg) -> None:      # 1 / (1 + e^-x)
    t: Reg = e.tmp()
    e.op("negative", t, x)
    e.op("exp", t, t)
    e.op("add", t, t, ONE)
    e.op("reciprocal", d, t)


def _silu(e: PTXEmitter, d: Reg, x: Reg) -> None:         # x * sigmoid(x)
    e.op("sigmoid", d, x)
    e.op("multiply", d, d, x)


def _where(e: PTXEmitter, d: Reg, c: Reg, a: Reg, b: Reg) -> None:  # c ? a : b
    """Booleans live in f32 registers as 1.0 / 0.0 (what the comparison
    ops below produce), so any non-zero c counts as true."""
    p: Reg = e.reg("p")
    e.emit(f"setp.ne.f32 {p}, {c}, {ZERO};", f"selp.f32 {d}, {a}, {b}, {p};")


def _log1p(e: PTXEmitter, d: Reg, x: Reg) -> None:        # log(1 + x), accurate for tiny x
    """Goldberg's trick: u = 1 + x rounds, but log(u) / (u - 1) * x cancels
    the rounding error. When u == 1 the answer is x itself (and the division
    is 0/0, which selp throws away)."""
    u: Reg
    t: Reg
    p: Reg
    u, t, p = e.tmp(), e.tmp(), e.reg("p")
    e.op("add", u, x, ONE)
    e.emit(f"setp.eq.f32 {p}, {u}, {ONE};")
    e.op("log", t, u)
    e.op("subtract", u, u, ONE)
    e.op("divide", t, t, u)
    e.op("multiply", t, t, x)
    e.emit(f"selp.f32 {d}, {x}, {t}, {p};")


def _softplus(e: PTXEmitter, d: Reg, x: Reg) -> None:     # log(1 + e^x) = max(x, 0) + log1p(e^-|x|)
    """The rewritten form never overflows: e^-|x| is always in (0, 1]."""
    t: Reg = e.tmp()
    e.op("abs", t, x)
    e.op("negative", t, t)
    e.op("exp", t, t)
    e.op("log1p", t, t)
    e.op("relu", d, x)
    e.op("add", d, d, t)


F32_OPS: dict[str, F32Rule] = {
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

    # ========================= Operations outside the tests, but within emmy's IR
    "square":  "mul.f32 {d}, {0}, {0};",
    "fabs":    "abs.f32 {d}, {0};",
    "fmax":    "max.f32 {d}, {0}, {1};",
    "fmin":    "min.f32 {d}, {0}, {1};",
    "pad":     "mov.f32 {d}, {0};",
    "sin":     "sin.approx.f32 {d}, {0};",
    "cos":     "cos.approx.f32 {d}, {0};",

    # logical ops
    "equal":         "set.eq.f32.f32 {d}, {0}, {1};",
    "not_equal":     "set.ne.f32.f32 {d}, {0}, {1};",
    "greater":       "set.gt.f32.f32 {d}, {0}, {1};",
    "greater_equal": "set.ge.f32.f32 {d}, {0}, {1};",
    "less":          "set.lt.f32.f32 {d}, {0}, {1};",
    "less_equal":    "set.le.f32.f32 {d}, {0}, {1};",

    "log1p":   _log1p,
    "where":   _where,   # setp.ne.f32 p, c, 0;  selp.f32 d, a, b, p
    "softplus": _softplus,  # max(x, 0) + log(1 + exp(-|x|))
}

# Starting value of an accumulator, per reduce op. Max/min use +-1e30 like Emmy,
# not +-inf: the online-softmax combine computes `m_a - max(m_a, m_b)`, which is
# NaN for -inf - -inf but a harmless 0 for -1e30 - -1e30.
IDENTITY: dict[str, float] = {"add": 0.0, "multiply": 1.0, "maximum": -1e30, "minimum": 1e30}


# =============================================================================
# Integer index arithmetic. op -> (PTX instruction, Python constant fold)
# =============================================================================
INT_OPS: dict[str, tuple[str, Callable[[int, int], int]]] = {
    "+": ("add.s32", operator.add),      "-": ("sub.s32", operator.sub),
    "*": ("mul.lo.s32", operator.mul),   "/": ("div.u32", operator.floordiv),
    "%": ("rem.u32", operator.mod),      "&": ("and.b32", operator.and_),
    "|": ("or.b32", operator.or_),       "<<": ("shl.b32", operator.lshift),
    ">>": ("shr.s32", operator.rshift),
}
CMP_OPS: dict[str, tuple[str, Callable[[int, int], bool]]] = {   # -> setp.<cmp>.s32
    "==": ("eq", operator.eq), "!=": ("ne", operator.ne),
    "<": ("lt", operator.lt),  "<=": ("le", operator.le),
    ">": ("gt", operator.gt),  ">=": ("ge", operator.ge),
}
_and: tuple[str, Callable[[Const, Const], Const]] = ("and.pred", lambda a, b: a and b)
_or: tuple[str, Callable[[Const, Const], Const]] = ("or.pred", lambda a, b: a or b)
LOGIC_OPS: dict[str, tuple[str, Callable[[Const, Const], Const]]] = {
    "&&": _and, "and": _and, "||": _or, "or": _or,
}
FOLD: dict[str, Callable[[Const, Const], Const]] = {
    op: f for table in (INT_OPS, CMP_OPS, LOGIC_OPS) for op, (_, f) in table.items()
}


# Small formatting helpers --------------------------------------------------
def is_reg(v: IdxValue) -> TypeGuard[Reg]:
    """expr() returns either a register name (str) or a folded Python constant."""
    return isinstance(v, str)


def mem(base: Reg, off: int = 0) -> str:
    return f"[{base}+{off}]" if off else f"[{base}]"


def vec(regs: Sequence[Reg]) -> str:              # "" or ".v2" / ".v4"
    return "" if len(regs) == 1 else f".v{len(regs)}"


def group(regs: Sequence[Reg]) -> str:            # "%f1" or "{%f1, %f2}"
    return regs[0] if len(regs) == 1 else "{" + ", ".join(regs) + "}"


class PTXEmitter:
    graph: Graph
    kop: KernelOp
    arg_order: Sequence[str]
    launch_threads: Optional[int]
    dynamic_smem: bool
    smem_offset: int
    lines: list[str]
    globals: list[str]
    count: dict[str, int]
    labels: int
    vals: dict[str, Reg]                  # IR value name -> %f register
    idx: dict[str, IdxValue]              # axis/index name -> %r register or folded int
    ptr: dict[str, Reg]                   # buffer name -> %rd base address
    smem: dict[str, tuple[int, ...]]      # shared array name -> extents
    inits: set[str]

    def __init__(self, graph: Graph, kop: KernelOp, arg_order: Sequence[str],
                 launch_threads: Optional[int] = None, smem_bytes: int = 0) -> None:
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
    def reg(self, kind: str) -> Reg:
        self.count[kind] += 1
        return f"%{kind}{self.count[kind]}"

    def tmp(self) -> Reg:
        return self.reg("f")

    def label(self, tag: str) -> str:
        self.labels += 1
        return f"{tag}_{self.labels}"

    def emit(self, *lines: str) -> None:
        self.lines.extend(l if l.endswith(":") else "\t" + l for l in lines)

    def dispatch(self, prefix: str, node: Union[Stmt, Expr]) -> Any:
        kind: str = type(node).__name__
        handler: Optional[Callable[[Any], Any]] = getattr(self, prefix + kind, None)
        if handler is None:
            raise NotImplementedError(f"Kernel IR node {kind} (add {prefix}{kind})")
        return handler(node)

    def body(self, stmts: Body) -> None:
        for s in stmts or ():
            self.dispatch("do_", s)

    def op(self, name: str, d: Reg, *args: Reg) -> None:
        """d = name(args) on f32 registers, using F32_OPS."""
        rule: F32Rule = F32_OPS[name]
        if callable(rule):
            rule(self, d, *args)
        else:
            self.emit(rule.format(*args, d=d))

    def bind(self, name: str, reg: Reg) -> None:
        """name = reg. Rebinding an existing name writes into its register."""
        if name in self.vals:
            self.emit(f"mov.f32 {self.vals[name]}, {reg};")
        else:
            self.vals[name] = reg

    # =========================================================================
    # Expressions -> Python int (folded), %r register, or %p predicate
    # =========================================================================
    def expr(self, e: Expr) -> IdxValue:
        return self.dispatch("ex_", e)

    def ex_Literal(self, e: Expr) -> int:
        return int(e.value)

    def ex_Var(self, e: Expr) -> IdxValue:
        return self.idx[e.name]

    def ex_TernaryExpr(self, e: Expr) -> IdxValue:   # c ? a : b  ->  selp
        c: IdxValue = self.expr(e.cond)
        if not is_reg(c):
            return self.expr(e.if_true if c else e.if_false)
        a: IdxValue = self.expr(e.if_true)
        b: IdxValue = self.expr(e.if_false)
        d: Reg = self.reg("r")
        self.emit(f"selp.b32 {d}, {a}, {b}, {self.pred(c)};")
        return d

    def ex_BinaryExpr(self, e: Expr) -> IdxValue:
        l: IdxValue = self.expr(e.left)
        r: IdxValue = self.expr(e.right)
        if not is_reg(l) and not is_reg(r):
            return FOLD[e.op](l, r)
        if e.op in CMP_OPS:
            p: Reg = self.reg("p")
            self.emit(f"setp.{CMP_OPS[e.op][0]}.s32 {p}, {l}, {r};")
            return p
        if e.op in LOGIC_OPS:
            p = self.reg("p")
            self.emit(f"{LOGIC_OPS[e.op][0]} {p}, {self.pred(l)}, {self.pred(r)};")
            return p
        if (e.op, r) in (("+", 0), ("*", 1)):    # x+0, x*1
            return l
        d: Reg = self.reg("r")
        self.emit(f"{INT_OPS[e.op][0]} {d}, {l}, {r};")
        return d

    def pred(self, v: IdxValue) -> Reg:
        """Turn an expr() result into a predicate register."""
        if is_reg(v) and v.startswith("%p"):
            return v
        p: Reg = self.reg("p")
        self.emit(f"setp.ne.s32 {p}, {v if is_reg(v) else int(v)}, 0;")
        return p

    # =========================================================================
    # Memory: buf[index] -> address -> ld / st / red
    # =========================================================================
    def shape(self, buf: str) -> tuple[int, ...]:
        if buf in self.smem:
            return self.smem[buf]
        return tuple(int(d.value) for d in self.graph.nodes[buf].outputs[0].shape)

    def flat_index(self, buf: str, index: Sequence[Expr]) -> IdxValue:
        """Row-major flatten of an index tuple -> int or %r."""
        if len(index) == 1:
            return self.expr(index[0])
        shape: tuple[int, ...] = self.shape(buf)
        flat: IdxValue = 0
        for i, ix in enumerate(index):
            stride: int = math.prod(shape[i + 1:])
            term: IdxValue = self.expr(ix)
            if not is_reg(term) and not is_reg(flat):
                flat += term * stride
                continue
            t: Reg = self.reg("r")
            if flat == 0:
                self.emit(f"mul.lo.s32 {t}, {term}, {stride};" if stride != 1 else f"mov.s32 {t}, {term};")
            elif stride == 1:
                self.emit(f"add.s32 {t}, {term}, {flat};")
            else:
                self.emit(f"mad.lo.s32 {t}, {term}, {stride}, {flat};")
            flat = t
        return flat

    def address(self, buf: str, index: Optional[Sequence[Expr]] = None,
                flat: Optional[IdxValue] = None) -> Address:
        """buf[index] (or buf[flat]) -> (state space, base register, byte offset)."""
        space: str = "shared" if buf in self.smem else "global"
        if flat is None:
            flat = self.flat_index(buf, index)
        if not is_reg(flat):                     # constant index: fold into the offset
            return space, self.ptr[buf], 4 * flat
        off: Reg = self.reg("rd")
        addr: Reg = self.reg("rd")
        self.emit(f"mul.wide.s32 {off}, {flat}, 4;", f"add.s64 {addr}, {self.ptr[buf]}, {off};")
        return space, addr, 0

    def load(self, dsts: Sequence[Reg], buf: str, index: Optional[Sequence[Expr]] = None,
             flat: Optional[IdxValue] = None) -> None:
        """One scalar load, or one vector load for `a, b, c, d = load x[i]`."""
        space, base, off = self.address(buf, index, flat)
        self.emit(f"ld.{space}{vec(dsts)}.f32 {group(dsts)}, {mem(base, off)};")

    def store(self, buf: str, index: Optional[Sequence[Expr]], vals: Sequence[Reg],
              atomic: bool = False, flat: Optional[IdxValue] = None) -> None:
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
    def do_Tile(self, t: Stmt) -> None:
        """gid = ctaid.x * ntid.x + tid.x; exit if gid >= N; decode the axes
        row-major from gid (innermost axis last)."""
        exts: list[int] = [int(a.extent.value) for a in t.axes]
        n: int = math.prod(exts)
        tid, bid, bdim, gid, lane, warp = (self.reg("r") for _ in range(6))
        self.emit(f"mov.u32 {tid}, %tid.x;", f"mov.u32 {bid}, %ctaid.x;",
                  f"mov.u32 {bdim}, %ntid.x;", f"mad.lo.s32 {gid}, {bid}, {bdim}, {tid};",
                  f"and.b32 {lane}, {tid}, 31;", f"shr.u32 {warp}, {tid}, 5;")
        if self.launch_threads is None or self.launch_threads > n:   # surplus threads
            p: Reg = self.reg("p")
            self.emit(f"setp.ge.s32 {p}, {gid}, {n};", f"@{p} bra DONE;")
        self.idx.update(tid=tid, lane=lane, warp=warp)
        rest: Reg = gid
        for axis, ext in reversed(list(zip(t.axes, exts))):
            if ext == n:                         # the only non-trivial axis
                self.idx[axis.name] = rest
                continue
            a: Reg = self.reg("r")
            q: Reg = self.reg("r")
            self.emit(f"rem.s32 {a}, {rest}, {ext};", f"div.s32 {q}, {rest}, {ext};")
            self.idx[axis.name], rest = a, q
        self.body(t.body)

    # =========================================================================
    # Statements: scalar compute
    # =========================================================================
    def do_Load(self, s: Stmt) -> None:
        dsts: list[Reg] = [self.tmp() for _ in s.names]
        self.load(dsts, s.input, s.index)
        for name, d in zip(s.names, dsts):
            self.bind(name, d)

    def do_Assign(self, s: Stmt) -> None:
        d: Reg = self.tmp()
        self.op(s.op.name, d, *(self.vals[a] for a in s.args))
        self.bind(s.name, d)

    def do_Let(self, s: Stmt) -> None:
        d: Reg = self.tmp()
        self.emit(f"mov.f32 {d}, {f32_hex(s.value.value)};")
        self.bind(s.name, d)

    def do_Init(self, s: Stmt) -> None:
        """Carried state seeded with a literal (matmul accumulator tiles)."""
        d: Reg = self.tmp()
        self.emit(f"mov.f32 {d}, {f32_hex(s.identity)};")
        self.bind(s.name, d)
        self.inits.add(s.name)

    def do_Write(self, s: Stmt) -> None:
        self.store(s.output, s.index, [self.vals[v] for v in s.values], s.atomic)

    # =========================================================================
    # Statements: control flow
    # =========================================================================
    def loop(self, axis: str, start: IdxValue, end: IdxValue, step: IdxValue, body: Body) -> None:
        """for (axis = start; axis < end; axis += step) body"""
        i: Reg = self.reg("r")
        p: Reg = self.reg("p")
        top: str = self.label("LOOP")
        done: str = self.label("LOOP_END")
        self.emit(f"mov.s32 {i}, {start};", f"{top}:",
                  f"setp.ge.s32 {p}, {i}, {end};", f"@{p} bra {done};")
        self.idx[axis] = i
        self.body(body)
        self.emit(f"add.s32 {i}, {i}, {step};", f"bra {top};", f"{done}:")

    def do_StridedLoop(self, s: Stmt) -> None:
        self.seed_accums(s)
        end: IdxValue = self.expr(s.end) if s.end is not None else int(s.axis.extent.value)
        self.loop(s.axis.name, self.expr(s.start), end, self.expr(s.step), s.body)

    def do_Loop(self, s: Stmt) -> None:
        """for axis in 0..extent. Unrolled when the IR asks and the loop is
        innermost (matmul's k loop): the axis becomes a Python int and every
        index constant-folds. Outer loops stay rolled even if marked unroll:
        attention's key loop wraps four 64-step dot products, and unrolling it
        exhausts registers."""
        self.seed_accums(s)
        n: int = int(s.axis.extent.value)
        innermost: bool = not any(type(c).__name__ in ("Loop", "StridedLoop") for c in s.body)
        if s.unroll and innermost and n <= 128:
            for i in range(n):
                self.idx[s.axis.name] = i
                self.body(s.body)
        else:
            self.loop(s.axis.name, 0, n, 1, s.body)

    def do_Cond(self, s: Stmt) -> None:
        c: IdxValue = self.expr(s.cond)
        if not is_reg(c):                        # constant condition
            self.body(s.body if c else s.else_body)
            return
        end: str = self.label("ENDIF")
        els: str = self.label("ELSE") if s.else_body else end
        self.emit(f"@!{self.pred(c)} bra {els};")
        self.body(s.body)
        if s.else_body:
            self.emit(f"bra {end};", f"{els}:")
            self.body(s.else_body)
        self.emit(f"{end}:")

    # =========================================================================
    # Statements: reductions
    # =========================================================================
    def accums(self, stmts: Body) -> Iterator[Stmt]:
        """All Accum statements under `stmts`, recursively."""
        for s in stmts or ():
            if type(s).__name__ == "Accum":
                yield s
            yield from self.accums(getattr(s, "body", None))
            yield from self.accums(getattr(s, "else_body", None))

    def seed_accums(self, loop: Stmt) -> None:
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
                acc: Reg = self.vals[s.name]
                self.emit(f"mov.f32 {acc}, {f32_hex(IDENTITY[s.op.name])};")

    def do_Accum(self, s: Stmt) -> None:
        """acc = op(base, value) in place; base defaults to acc itself. The online
        softmax uses base for the rescaled old sum: acc = acc*alpha + x*beta."""
        acc: Reg = self.vals[s.name]
        self.op(s.op.name, acc, self.vals[s.base] if s.base else acc, self.vals[s.value])

    def combine(self, s: Stmt, others: Sequence[Reg]) -> None:
        """Bind the state_b names to `others`, then run the combine statements."""
        self.vals.update(zip(s.state_b, others))
        self.body(s.combine_states)

    def do_WarpShuffle(self, s: Stmt) -> None:
        """XOR butterfly over `length` lanes: offsets length/2, ..., 1."""
        off: int = s.length // 2
        while off >= 1:
            others: list[Reg] = [self.tmp() for _ in s.state]
            for st, o in zip(s.state, others):
                self.emit(f"shfl.sync.bfly.b32 {o}, {self.vals[st]}, {off}, 0x1f, 0xffffffff;")
            self.combine(s, others)
            off //= 2

    def do_TreeHalve(self, s: Stmt) -> None:
        """Shared-memory tree over `length` slots, then broadcast slot 0:

            for (h = length/2; h > 0; h /= 2) {          // unrolled: length is static
                if (tid < h) { st = buf[tid]; st_b = buf[tid+h]; combine; buf[tid] = st; }
                barrier
            }
            st = buf[0]
        """
        if s.inner:
            raise NotImplementedError("TreeHalve.inner")
        tid: IdxValue = self.idx[s.tid_var]
        bar: str = f"bar.sync {s.barrier_id}" + (f", {s.barrier_count}" if s.barrier_count else "") + ";"
        h: int = s.length // 2
        while h >= 1:
            p: Reg = self.reg("p")
            partner: Reg = self.reg("r")
            skip: str = self.label("TREE_SKIP")
            self.emit(f"setp.ge.s32 {p}, {tid}, {h};", f"@{p} bra {skip};",
                      f"add.s32 {partner}, {tid}, {h};")
            others: list[Reg] = []
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
    def do_Smem(self, s: Stmt) -> None:
        """Static `.shared` array, or a slice of the dynamic pool when meta.json
        sets smem_bytes (needed above the 48 KB static limit, e.g. m01)."""
        extents: tuple[int, ...] = tuple(int(x) for x in s.extents)
        align: int = max(int(s.align or 0), 4)
        self.smem[s.name] = extents
        base: Reg = self.reg("rd")
        self.ptr[s.name] = base
        if self.dynamic_smem:
            self.smem_offset = -(-self.smem_offset // align) * align   # round up
            self.emit(f"mov.u64 {base}, smem_pool;",
                      f"add.u64 {base}, {base}, {self.smem_offset};")
            self.smem_offset += 4 * math.prod(extents)
        else:
            self.globals.append(f".shared .align {align} .f32 {s.name}[{math.prod(extents)}];")
            self.emit(f"mov.u64 {base}, {s.name};")

    def do_Sync(self, s: Stmt) -> None:
        if s.warp:
            self.emit("bar.warp.sync -1;")
        elif s.count:
            self.emit(f"bar.sync {s.barrier_id}, {s.count};")
        else:
            self.emit(f"bar.sync {s.barrier_id};")

    def do_CpAsyncCopy(self, s: Stmt) -> None:
        """smem[smem_index + lane_index] <- src[src_index], nbytes at once.
        Both shared-side indices flatten against the smem extents and add."""
        if s.swizzle not in (None, "NONE"):
            raise NotImplementedError("CpAsyncCopy swizzle")
        flat: IdxValue = self.flat_index(s.smem, s.smem_index)
        if s.lane_index:
            lane: IdxValue = self.flat_index(s.smem, s.lane_index)
            if not is_reg(flat) and not is_reg(lane):
                flat += lane
            else:
                t: Reg = self.reg("r")
                self.emit(f"add.s32 {t}, {flat}, {lane};")
                flat = t
        dbase: Reg
        doff: int
        sbase: Reg
        soff: int
        _, dbase, doff = self.address(s.smem, flat=flat)
        _, sbase, soff = self.address(s.src, s.src_index)
        cache: str = "cg" if s.nbytes == 16 else "ca"   # .cg only supports 16 bytes
        self.emit(f"cp.async.{cache}.shared.global {mem(dbase, doff)}, {mem(sbase, soff)}, {s.nbytes};")

    def do_CpAsyncCommit(self, s: Stmt) -> None:
        self.emit("cp.async.commit_group;")

    def do_CpAsyncWait(self, s: Stmt) -> None:
        self.emit(f"cp.async.wait_group {s.group};")

    # =========================================================================
    # Whole kernel
    # =========================================================================
    def compile(self) -> str:
        for n in self.arg_order:                 # params -> global pointers
            raw: Reg = self.reg("rd")
            self.ptr[n] = self.reg("rd")
            self.emit(f"ld.param.u64 {raw}, [param_{n}];",
                      f"cvta.to.global.u64 {self.ptr[n]}, {raw};")
        self.body(self.kop.body)

        if self.dynamic_smem and self.smem:
            self.globals.insert(0, ".extern .shared .align 16 .b8 smem_pool[];")
        params: str = ",\n".join(f"\t.param .u64 param_{n}" for n in self.arg_order)
        decls: list[str] = [f"\t.reg .{t} %{k}<{self.count[k] + 1}>;"
                            for k, t in (("p", "pred"), ("r", "b32"), ("rd", "b64"), ("f", "f32"))]
        return "\n".join([PTX_HEADER, *self.globals, "",
                          f".visible .entry {self.kop.name}(", params, ")", "{",
                          *decls, "", *self.lines, "DONE:", "\tret;", "}", ""])


# =============================================================================
# Driver
# =============================================================================
def launch_info(ir_path: Path, entry: str) -> tuple[Optional[list[str]], Optional[int], int]:
    """(arg_order, threads launched, dynamic smem bytes) from meta.json."""
    meta: Path = ir_path.parent / "meta.json"
    if meta.exists():
        for k in json.loads(meta.read_text())["kernels"]:
            if k["entry"] == entry:
                threads: int = math.prod(k["grid"]) * math.prod(k["block"])
                return k["arg_order"], threads, k.get("smem_bytes", 0)
    return None, None, 0


def main(path: Union[str, Path]) -> None:
    path = Path(path)
    graph: Graph = Graph.from_dict(json.loads(path.read_text()))
    node = next(n for n in graph.nodes.values() if isinstance(n.op, KernelOp))
    arg_order, threads, smem_bytes = launch_info(path, node.op.name)
    arg_order = arg_order or [*node.inputs, node.id]     # fallback: inputs, then output
    sys.stdout.write(PTXEmitter(graph, node.op, arg_order, threads, smem_bytes).compile())


if __name__ == "__main__":
    main(sys.argv[1])