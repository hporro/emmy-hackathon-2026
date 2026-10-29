"""Kernel IR -> PTX compiler.

    python compile.py path/to/kernel.ir.json > out.ptx

How it is organised
-------------------
main() loads the IR and meta.json; PTXEmitter walks the kernel body and
appends PTX lines. Every IR node kind is lowered by its own class:

    IR statement  X   ->  class XStatement(Statement, kind="X")   with generate()
    IR expression X   ->  class XExpression(Expression, kind="X") with generate()
    elementwise op    ->  one entry in the op table of its type (F32_OPS, F16_OPS, ...)

Types
-----
Every value is a Val: a register plus its DType. The DType table below says,
for each type, which PTX suffix and register class it uses.

Each type family has its own op table: F32_OPS, F16_OPS, F64_OPS, INT_OPS,
BOOL_OPS. An op missing from a table is computed in f32 (inputs converted to
f32 first). So f16 starts out correct for every op, and adding a native f16
instruction is one line in F16_OPS.

Type changes all go through PTXEmitter.convert().

Where values live
-----------------
    em.vals  IR value name   -> Val
    em.idx   axis/index name -> s32 register (%r), predicate (%p) or a Python int
    em.ptr   buffer name     -> base address (%rd)

Registers are handed out fresh and never reused; ptxas does the real
allocation. Rebinding a name (Accum, or Assign to an existing name) writes
into the name's existing register, so loop-carried values just work.

Index arithmetic assumes non-negative operands for / and % (true for every
Emmy index), so it uses unsigned div/rem.
"""
from __future__ import annotations

import json
import math
import operator
import struct
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar, NamedTuple, Optional, Sequence, Union

from emmy.compiler.graph import Graph
from emmy.compiler.ir.kernel.ir import KernelOp
from emmy.compiler.dim import Dim
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.stmt.body import free_names
from emmy.compiler.ir.stmt.leaves import Assign


PTX_HEADER = ".version 8.0\n.target sm_89\n.address_size 64\n"


# =============================================================================
# Data types
# =============================================================================
class DType(NamedTuple):
    name: str     # Emmy's name
    t: str        # suffix for arithmetic / setp / cvt source ("f32", "s32", ...)
    mem: str      # suffix for ld / st, and for a cvt that produces an int
    reg: str      # register class
    nbytes: int

    @property
    def is_float(self) -> bool:
        return self.t[0] == "f"

    @property
    def is_int(self) -> bool:
        return self.t[0] in "su"

    @property
    def b(self) -> str:              # untyped suffix for mov / selp: "b32", "pred", ...
        return {"p": "pred", "rs": "b16", "r": "b32", "f": "b32", "rd": "b64", "fd": "b64"}[self.reg]


# PTX has no 8-bit registers or arithmetic: i8 / u8 live in 16-bit registers
# and are wrapped back to 8 bits after each op (see PTXEmitter.op).
# f16 has no ld/st suffix of its own, so it moves as raw .b16.
BOOL = DType("bool", "pred", "u8",  "p",  1)
I8   = DType("i8",   "s16",  "s8",  "rs", 1)
U8   = DType("u8",   "u16",  "u8",  "rs", 1)
I16  = DType("i16",  "s16",  "s16", "rs", 2)
U16  = DType("u16",  "u16",  "u16", "rs", 2)
I32  = DType("i32",  "s32",  "s32", "r",  4)
U32  = DType("u32",  "u32",  "u32", "r",  4)
I64  = DType("i64",  "s64",  "s64", "rd", 8)
U64  = DType("u64",  "u64",  "u64", "rd", 8)
F16  = DType("f16",  "f16",  "b16", "rs", 2)
F32  = DType("f32",  "f32",  "f32", "f",  4)
F64  = DType("f64",  "f64",  "f64", "fd", 8)

# Mixing types: the later type in this list wins (bool < ints < floats).
PROMOTION = [BOOL, U8, I8, U16, I16, U32, I32, U64, I64, F16, F32, F64]

DTYPES = {d.name: d for d in PROMOTION} | {
    # CUDA / numpy spellings used by Smem.dtype and Literal.dtype
    "float": F32, "__half": F16, "half": F16, "double": F64, "int": I32, "long long": I64,
    "unsigned char": U8, "unsigned short": U16, "short": I16, "unsigned int": U32,
    "unsigned long long": U64, "char": I8, "signed char": I8,
}


def dtype_of(x: Any, default: Optional[DType] = None) -> Optional[DType]:
    """Emmy DataType / type name / None -> DType."""
    if x is None:
        return default
    return DTYPES[getattr(x, "name", x)]


def promote(*dts: DType) -> DType:
    return max(dts, key=PROMOTION.index)


class Val(NamedTuple):
    reg: str      # register name, or an immediate like "0f3F800000"
    dt: DType

    def __str__(self) -> str:
        return self.reg


# =============================================================================
# Constants
# =============================================================================
def f32_hex(x: float) -> str:
    """PTX float literal: 0f + the IEEE-754 bits in hex (exact, no rounding)."""
    return "0f%08X" % struct.unpack("<I", struct.pack("<f", float(x)))[0]


def imm(x: Union[int, float], dt: DType) -> str:
    """Immediate operand text for a constant of type dt."""
    if dt == F32:
        return f32_hex(x)
    if dt == F64:
        return "0d%016X" % struct.unpack("<Q", struct.pack("<d", float(x)))[0]
    if dt == F16:
        if abs(x) > 65504 and not math.isinf(x):    # e.g. Emmy's -1e30 sentinel
            x = math.copysign(65504, x)
        return "0x%04X" % struct.unpack("<H", struct.pack("<e", float(x)))[0]
    if dt == BOOL:
        return "1" if x else "0"
    bits = 8 * dt.nbytes                            # ints: clamp (-1e30 -> INT_MIN)
    lo, hi = (-(1 << bits - 1), (1 << bits - 1) - 1) if dt.t[0] == "s" else (0, (1 << bits) - 1)
    return str(int(max(lo, min(hi, x))))


LOG2E, LN2 = f32_hex(math.log2(math.e)), f32_hex(math.log(2.0))

# Starting value of an accumulator, per reduce op. Max/min use +-1e30 like Emmy,
# not +-inf: the online-softmax combine computes `m_a - max(m_a, m_b)`, which is
# NaN for -inf - -inf but a harmless 0 for -1e30 - -1e30.
IDENTITY = {"add": 0.0, "multiply": 1.0, "maximum": -1e30, "minimum": 1e30}


# =============================================================================
# Elementwise ops, one table per type family.
#   string   -> one-instruction template: {d} is the result, {0} {1} {2} the
#               operands, {t} the operands' type suffix, {w} its width (b{w} = untyped)
#   function -> f(em, d, *args), built from other ops with em.op(...)
# The operands of an op always have the same type. Comparisons write a bool d.
# =============================================================================
def _exp(em: PTXEmitter, d: Val, x: Val) -> None:         # e^x = 2^(x * log2 e)
    t = em.new(F32)
    em.emit(f"mul.f32 {t}, {x}, {LOG2E};", f"ex2.approx.ftz.f32 {d}, {t};")


def _log(em: PTXEmitter, d: Val, x: Val) -> None:         # ln x = log2 x * ln 2
    em.emit(f"lg2.approx.f32 {d}, {x};", f"mul.f32 {d}, {d}, {LN2};")


def _sigmoid(em: PTXEmitter, d: Val, x: Val) -> None:     # 1 / (1 + e^-x)
    t = em.new(F32)
    em.op("negative", t, x)
    em.op("exp", t, t)
    em.op("add", t, t, em.const(1, F32))
    em.op("reciprocal", d, t)


def _silu(em: PTXEmitter, d: Val, x: Val) -> None:        # x * sigmoid(x)
    em.op("sigmoid", d, x)
    em.op("multiply", d, d, x)


def _log1p(em: PTXEmitter, d: Val, x: Val) -> None:       # log(1 + x), accurate for tiny x
    """Goldberg's trick: u = 1 + x rounds, but log(u) / (u - 1) * x cancels
    the rounding error. When u == 1 the answer is x itself (and the division
    is 0/0, which selp throws away)."""
    u, t, p = em.new(F32), em.new(F32), em.new(BOOL)
    em.op("add", u, x, em.const(1, F32))
    em.emit(f"setp.eq.f32 {p}, {u}, 0f3F800000;")
    em.op("log", t, u)
    em.op("subtract", u, u, em.const(1, F32))
    em.op("divide", t, t, u)
    em.op("multiply", t, t, x)
    em.emit(f"selp.f32 {d}, {x}, {t}, {p};")


def _softplus(em: PTXEmitter, d: Val, x: Val) -> None:    # max(x, 0) + log1p(e^-|x|)
    """The rewritten form never overflows: e^-|x| is always in (0, 1]."""
    t = em.new(F32)
    em.op("abs", t, x)
    em.op("negative", t, t)
    em.op("exp", t, t)
    em.op("log1p", t, t)
    em.op("relu", d, x)
    em.op("add", d, d, t)


# Comparisons, shared by every table: op -> setp condition.
# Float != is "neu" (unordered) so that NaN != x is true, as in C.
COMPARE = {"equal": "eq", "not_equal": "ne", "greater": "gt",
           "greater_equal": "ge", "less": "lt", "less_equal": "le"}


def compare_ops(not_equal: str = "ne") -> dict:
    ops = {op: f"setp.{c}.{{t}} {{d}}, {{0}}, {{1}};" for op, c in COMPARE.items()}
    ops["not_equal"] = f"setp.{not_equal}.{{t}} {{d}}, {{0}}, {{1}};"
    return ops


F32_OPS: dict[str, Any] = {
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
    "sin":        "sin.approx.f32 {d}, {0};",
    "cos":        "cos.approx.f32 {d}, {0};",
    "square":     "mul.f32 {d}, {0}, {0};",
    "fma":        "fma.rn.f32 {d}, {0}, {1}, {2};",
    "floor":      "cvt.rmi.f32.f32 {d}, {0};",
    "relu":       lambda em, d, x: em.op("maximum", d, x, em.const(0, F32)),
    "exp":        _exp,
    "log":        _log,
    "sigmoid":    _sigmoid,
    "silu":       _silu,
    "log1p":      _log1p,
    "softplus":   _softplus,
    **compare_ops(not_equal="neu"),
}
F32_OPS |= {"exp_fast": F32_OPS["exp"], "fabs": F32_OPS["abs"], "fmax": F32_OPS["maximum"],
            "fmin": F32_OPS["minimum"], "pad": F32_OPS["copy"]}

# Native f16 instructions (sm_53+; min/max need sm_80). Everything else runs in
# f32 and is rounded back to f16 when it is stored or bound to an f16 name.
F16_OPS: dict[str, Any] = {
    "add":      "add.rn.f16 {d}, {0}, {1};",
    "subtract": "sub.rn.f16 {d}, {0}, {1};",
    "multiply": "mul.rn.f16 {d}, {0}, {1};",
    "fma":      "fma.rn.f16 {d}, {0}, {1}, {2};",
    "maximum":  "max.f16 {d}, {0}, {1};",
    "minimum":  "min.f16 {d}, {0}, {1};",
    "negative": "neg.f16 {d}, {0};",
    "abs":      "abs.f16 {d}, {0};",
    "copy":     "mov.b16 {d}, {0};",
    **compare_ops(not_equal="neu"),
}

# f64 has add/mul/div/sqrt/rcp but no ex2/lg2/sin/cos/tanh: those run in f32.
F64_OPS: dict[str, Any] = {
    "add":        "add.rn.f64 {d}, {0}, {1};",
    "subtract":   "sub.rn.f64 {d}, {0}, {1};",
    "multiply":   "mul.rn.f64 {d}, {0}, {1};",
    "divide":     "div.rn.f64 {d}, {0}, {1};",
    "fma":        "fma.rn.f64 {d}, {0}, {1}, {2};",
    "maximum":    "max.f64 {d}, {0}, {1};",
    "minimum":    "min.f64 {d}, {0}, {1};",
    "negative":   "neg.f64 {d}, {0};",
    "abs":        "abs.f64 {d}, {0};",
    "copy":       "mov.f64 {d}, {0};",
    "sqrt":       "sqrt.rn.f64 {d}, {0};",
    "reciprocal": "rcp.rn.f64 {d}, {0};",
    **compare_ops(not_equal="neu"),
}

# Integers of every width and signedness: {t} is s32, u16, s64, ...
# Division and remainder truncate toward zero, like C.
INT_OPS: dict[str, Any] = {
    "add":         "add.{t} {d}, {0}, {1};",
    "subtract":    "sub.{t} {d}, {0}, {1};",
    "multiply":    "mul.lo.{t} {d}, {0}, {1};",
    "floor_divide": "div.{t} {d}, {0}, {1};",
    "remainder":   "rem.{t} {d}, {0}, {1};",
    "maximum":     "max.{t} {d}, {0}, {1};",
    "minimum":     "min.{t} {d}, {0}, {1};",
    "negative":    "neg.s{w} {d}, {0};",
    "abs":         lambda em, d, x: em.emit(f"{'abs' if x.dt.t[0] == 's' else 'mov'}.{x.dt.t} {d}, {x};"),
    "copy":        "mov.{t} {d}, {0};",
    "bitwise_and": "and.b{w} {d}, {0}, {1};",
    "bitwise_or":  "or.b{w} {d}, {0}, {1};",
    "bitwise_xor": "xor.b{w} {d}, {0}, {1};",
    "bitwise_not": "not.b{w} {d}, {0};",
    # shr is arithmetic for signed types; the shift amount is always a u32
    "left_shift":  lambda em, d, x, n: em.emit(f"shl.b{x.dt.t[1:]} {d}, {x}, {em.convert(n, U32)};"),
    "right_shift": lambda em, d, x, n: em.emit(f"shr.{x.dt.t} {d}, {x}, {em.convert(n, U32)};"),
    **compare_ops(),
}

BOOL_OPS: dict[str, Any] = {
    "logical_and": "and.pred {d}, {0}, {1};",
    "logical_or":  "or.pred {d}, {0}, {1};",
    "logical_xor": "xor.pred {d}, {0}, {1};",
    "logical_not": "not.pred {d}, {0};",
    "not_equal":   "xor.pred {d}, {0}, {1};",
    "maximum":     "or.pred {d}, {0}, {1};",
    "minimum":     "and.pred {d}, {0}, {1};",
    "copy":        "mov.pred {d}, {0};",
}
BOOL_OPS |= {"bitwise_and": BOOL_OPS["logical_and"], "bitwise_or": BOOL_OPS["logical_or"],
             "bitwise_xor": BOOL_OPS["logical_xor"], "bitwise_not": BOOL_OPS["logical_not"]}

LOGICAL = {"logical_and", "logical_or", "logical_xor", "logical_not"}   # operands become bool


def ops_for(dt: DType) -> dict[str, Any]:
    if dt.is_int:
        return INT_OPS
    return {"pred": BOOL_OPS, "f16": F16_OPS, "f32": F32_OPS, "f64": F64_OPS}[dt.t]


# =============================================================================
# Integer index arithmetic. op -> (PTX instruction, Python constant fold)
# =============================================================================
INT_INDEX_OPS = {
    "+": ("add.s32", operator.add),      "-": ("sub.s32", operator.sub),
    "*": ("mul.lo.s32", operator.mul),   "/": ("div.u32", operator.floordiv),
    "%": ("rem.u32", operator.mod),      "&": ("and.b32", operator.and_),
    "|": ("or.b32", operator.or_),       "^": ("xor.b32", operator.xor),
    "<<": ("shl.b32", operator.lshift),  ">>": ("shr.s32", operator.rshift),
    "//": ("div.u32", operator.floordiv),
}
CMP_OPS = {   # -> setp.<cmp>.s32
    "==": ("eq", operator.eq), "!=": ("ne", operator.ne),
    "<": ("lt", operator.lt),  "<=": ("le", operator.le),
    ">": ("gt", operator.gt),  ">=": ("ge", operator.ge),
}
_and = ("and.pred", lambda a, b: a and b)
_or = ("or.pred", lambda a, b: a or b)
LOGIC_OPS = {"&&": _and, "and": _and, "||": _or, "or": _or}
FOLD = {op: f for table in (INT_INDEX_OPS, CMP_OPS, LOGIC_OPS) for op, (_, f) in table.items()}

# Atomic Write -> red.<space>.<this> (two's complement add: i64 uses u64)
RED_ADD = {F32: "add.f32", F64: "add.f64", F16: "add.noftz.f16",
           I32: "add.s32", U32: "add.u32", I64: "add.u64", U64: "add.u64"}


def is_reg(v: Any) -> bool:
    """expr() returns either a register name (str) or a folded Python constant."""
    return isinstance(v, str)


def is_pred(v: Any) -> bool:
    return is_reg(v) and v.startswith("%p")


def mem(base: str, off: int = 0) -> str:
    return f"[{base}+{off}]" if off else f"[{base}]"


def vec(regs: Sequence) -> str:                   # "" or ".v2" / ".v4"
    return "" if len(regs) == 1 else f".v{len(regs)}"


def group(regs: Sequence) -> str:                 # "%f1" or "{%f1, %f2}"
    return str(regs[0]) if len(regs) == 1 else "{" + ", ".join(map(str, regs)) + "}"


REG_DECLS = {"p": "pred", "rs": "b16", "r": "b32", "rd": "b64", "f": "f32", "fd": "f64"}


# =============================================================================
# Emitter: shared state and plumbing. Node-specific logic lives in the
# Expression / Statement subclasses further down.
# =============================================================================
class PTXEmitter:
    def __init__(self, graph: Graph, kop: KernelOp, arg_order: Sequence[str],
                 launch_threads: Optional[int] = None, smem_bytes: int = 0) -> None:
        self.graph, self.kop, self.arg_order = graph, kop, arg_order
        self.launch_threads = launch_threads      # grid*block from meta.json
        self.dynamic_smem = smem_bytes > 0        # meta asks for a dynamic pool
        self.smem_offset = 0                      # next free byte in that pool
        self.lines: list[str] = []                # kernel body
        self.globals: list[str] = []              # module-level .shared decls
        self.count = {k: 0 for k in REG_DECLS}
        self.labels = 0
        self.vals: dict[str, Val] = {}            # IR value name -> Val
        self.named: set[str] = set()              # registers that belong to a name
        self.idx: dict[str, Any] = {}             # index name -> %r / %p / int
        self.ptr: dict[str, str] = {}             # buffer name -> %rd base address
        self.smem: dict[str, tuple[tuple[int, ...], DType]] = {}   # name -> (extents, type)
        self.inits: set[str] = set()              # Init-declared names (never reseeded)

    # =========================================================================
    # Plumbing: registers, labels, output
    # =========================================================================
    def reg(self, kind: str) -> str:
        self.count[kind] += 1
        return f"%{kind}{self.count[kind]}"

    def new(self, dt: DType) -> Val:
        return Val(self.reg(dt.reg), dt)

    def label(self, tag: str) -> str:
        self.labels += 1
        return f"{tag}_{self.labels}"

    def emit(self, *lines: str) -> None:
        self.lines.extend(l if l.endswith(":") else "\t" + l for l in lines)

    # =========================================================================
    # Typed values
    # =========================================================================
    def const(self, x: Union[int, float], dt: DType) -> Val:
        """An immediate; f16 and bool get a register (PTX has no immediates for them)."""
        if dt in (F16, BOOL):
            r = self.new(dt)
            self.emit(f"mov.{dt.b} {r}, {imm(x, dt)};")
            return r
        return Val(imm(x, dt), dt)

    def in_reg(self, v: Val) -> Val:
        """Make sure v is a register, not an immediate."""
        if v.reg.startswith("%"):
            return v
        r = self.new(v.dt)
        self.emit(f"mov.{v.dt.b} {r}, {v};")
        return r

    def convert(self, v: Val, dt: DType) -> Val:
        """v as type dt, like a C cast. Float -> int truncates, x -> bool is x != 0."""
        src = v.dt
        if src == dt:
            return v
        v = self.in_reg(v)
        if src == F16 and dt == BOOL:
            return self.convert(self.convert(v, F32), BOOL)
        d = self.new(dt)
        if dt == BOOL:
            self.emit(f"setp.{'neu' if src.is_float else 'ne'}.{src.t} {d}, {v}, {imm(0, src)};")
        elif src == BOOL:
            self.emit(f"selp.{dt.b} {d}, {imm(1, dt)}, {imm(0, dt)}, {v};")
        elif src.is_float and dt.is_float:      # round only when narrowing
            self.emit(f"cvt{'.rn' if dt.nbytes < src.nbytes else ''}.{dt.t}.{src.t} {d}, {v};")
        elif src.is_float:
            self.emit(f"cvt.rzi.{dt.mem}.{src.t} {d}, {v};")
        elif dt.is_float:
            self.emit(f"cvt.rn.{dt.t}.{src.t} {d}, {v};")
        else:
            self.emit(f"cvt.{dt.mem}.{src.t} {d}, {v};")
        return d

    def assign(self, dst: Val, src: Val) -> None:
        """dst = src (converted to dst's type)."""
        src = self.convert(src, dst.dt)
        if src.reg != dst.reg:
            self.emit(f"mov.{dst.dt.b} {dst}, {src};")

    def bind(self, name: str, v: Val) -> None:
        """name = v. Rebinding an existing name writes into its register.
        A new name gets its own register: v's, unless v is an immediate or
        already belongs to another name."""
        if name in self.vals:
            self.assign(self.vals[name], v)
            return
        if not v.reg.startswith("%") or v.reg in self.named:
            r = self.new(v.dt)
            self.assign(r, v)
            v = r
        self.vals[name] = v
        self.named.add(v.reg)

    def value(self, name: str) -> Val:
        """An IR name as a Val: a computed value, else an index variable."""
        if name in self.vals:
            return self.vals[name]
        x = self.idx[name]
        if not is_reg(x):
            return self.const(x, BOOL if isinstance(x, bool) else I32)
        return Val(x, BOOL if is_pred(x) else I32)

    def value_of(self, e: Any, dt: Optional[DType] = None) -> Val:
        """An IR expression as a Val (Let, casts)."""
        kind = type(e).__name__
        if kind == "Literal":
            v = self.const(e.value, dt or dtype_of(e.dtype, F32))
        elif kind == "Var":
            v = self.value(e.name)
        elif kind == "CastExpr":
            v = self.convert(self.value_of(e.expr), dtype_of(e.dtype))
        else:                                    # index arithmetic
            x = self.expr(e)
            v = Val(x, BOOL if is_pred(x) else I32) if is_reg(x) else self.const(x, I32)
        return self.convert(v, dt) if dt else v

    # =========================================================================
    # Elementwise ops
    # =========================================================================
    def op(self, name: str, d: Val, *args: Val) -> None:
        """d = name(args), with the rule from the operands' table. All operands
        must already have the same type; use apply() when they might not."""
        rule = ops_for(args[0].dt)[name]
        if callable(rule):
            rule(self, d, *args)
        else:
            t = args[0].dt.t
            self.emit(rule.format(*args, d=d, t=t, w=t[1:]))
        if d.dt in (I8, U8):                     # wrap the 16-bit result back to 8 bits
            self.emit(f"cvt.{d.dt.mem}.{d.dt.t} {d}, {d};")

    def apply(self, name: str, args: Sequence[Val], out: Optional[DType] = None) -> Val:
        """name(args) for operands of any types; returns a new Val.

        The op computes in the IR's declared type `out` (comparisons and
        untyped ops use the widest operand type). If that type's table has no
        rule for the op, it computes in f32. Callers convert the result."""
        if name == "where":
            return self.select(*args, out)
        if name in LOGICAL:
            ct = BOOL
        elif name in COMPARE or out is None or out == BOOL:
            ct = promote(*(a.dt for a in args))
        else:
            ct = out
        if name not in ops_for(ct):
            if name not in F32_OPS:
                raise NotImplementedError(f"elementwise op {name!r} on {ct.name}")
            ct = F32
        args = [self.convert(a, ct) for a in args]
        d = self.new(BOOL if name in COMPARE else ct)
        self.op(name, d, *args)
        return d

    def select(self, c: Val, a: Val, b: Val, out: Optional[DType] = None) -> Val:
        """c ? a : b"""
        dt = out or promote(a.dt, b.dt)
        c, a, b = self.convert(c, BOOL), self.convert(a, dt), self.convert(b, dt)
        if dt == BOOL:                           # selp has no .pred form
            a, b = self.convert(a, U16), self.convert(b, U16)
            return self.convert(self.select(c, a, b, U16), BOOL)
        d = self.new(dt)
        self.emit(f"selp.{dt.b} {d}, {a}, {b}, {c};")
        return d

    def shuffle(self, v: Val, offset: int) -> Val:
        """shfl.sync.bfly for any type. shfl moves 32 bits: 64-bit values go as
        two halves, smaller types are widened to 32 bits and back."""
        shfl = f"shfl.sync.bfly.b32 {{}}, {{}}, {offset}, 0x1f, 0xffffffff;"
        if v.dt.reg in ("p", "rs"):              # bool, 8 and 16 bit: widen first
            wide = F32 if v.dt == F16 else I32
            return self.convert(self.shuffle(self.convert(v, wide), offset), v.dt)
        v, d = self.in_reg(v), self.new(v.dt)
        if v.dt.reg in ("r", "f"):
            self.emit(shfl.format(d, v))
        else:                                    # 64 bit: two halves
            lo, hi, lo2, hi2 = (self.reg("r") for _ in range(4))
            self.emit(f"mov.b64 {{{lo}, {hi}}}, {v};", shfl.format(lo2, lo), shfl.format(hi2, hi),
                      f"mov.b64 {d}, {{{lo2}, {hi2}}};")
        return d

    # =========================================================================
    # Lowering entry points
    # =========================================================================
    def expr(self, e: Any) -> Any:
        """Lower an IR index expression -> Python int (folded), %r or %p register."""
        return Expression.lower(self, e)

    def stmt(self, s: Any) -> None:
        Statement.lower(self, s)

    def body(self, stmts: Any) -> None:
        for s in stmts or ():
            self.stmt(s)

    def pred(self, v: Any) -> str:
        """Turn an expr() result into a predicate register."""
        if is_pred(v):
            return v
        p = self.reg("p")
        self.emit(f"setp.ne.s32 {p}, {v if is_reg(v) else int(v)}, 0;")
        return p

    # =========================================================================
    # Memory: buf[index] -> address -> ld / st / red
    # =========================================================================
    def buf_dtype(self, buf: str) -> DType:
        if buf in self.smem:
            return self.smem[buf][1]
        return dtype_of(self.graph.nodes[buf].outputs[0].dtype, F32)

    def shape(self, buf: str) -> tuple[int, ...]:
        if buf in self.smem:
            return self.smem[buf][0]
        return tuple(int(d.value) for d in self.graph.nodes[buf].outputs[0].shape)

    def flat_index(self, buf: str, index: Sequence[Any]) -> Any:
        """Row-major flatten of an index tuple -> int or %r."""
        if len(index) == 1:
            return self.expr(index[0])
        shape = self.shape(buf)
        flat: Any = 0
        for i, ix in enumerate(index):
            stride = math.prod(shape[i + 1:])
            term = self.expr(ix)
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

    def address(self, buf: str, index: Any = None, flat: Any = None) -> tuple[str, str, int]:
        """buf[index] (or buf[flat]) -> (state space, base register, byte offset)."""
        space = "shared" if buf in self.smem else "global"
        size = self.buf_dtype(buf).nbytes
        if flat is None:
            flat = self.flat_index(buf, index)
        if not is_reg(flat):                     # constant index: fold into the offset
            return space, self.ptr[buf], size * flat
        off, addr = self.reg("rd"), self.reg("rd")
        self.emit(f"mul.wide.s32 {off}, {flat}, {size};", f"add.s64 {addr}, {self.ptr[buf]}, {off};")
        return space, addr, 0

    def load(self, buf: str, index: Any = None, n: int = 1, flat: Any = None) -> list[Val]:
        """n consecutive elements (one scalar or vector ld) in the buffer's type."""
        dt = self.buf_dtype(buf)
        raw = U8 if dt == BOOL else dt           # bools are bytes in memory
        space, base, off = self.address(buf, index, flat)
        regs = [self.new(raw) for _ in range(n)]
        self.emit(f"ld.{space}{vec(regs)}.{dt.mem} {group(regs)}, {mem(base, off)};")
        return [self.convert(r, dt) for r in regs]

    def store(self, buf: str, index: Any, vals: Sequence[Val], atomic: bool = False,
              flat: Any = None) -> None:
        """One scalar/vector st, or one red.add per element when atomic.
        Values are converted to the buffer's type first."""
        dt = self.buf_dtype(buf)
        raw = U8 if dt == BOOL else dt
        regs = [self.in_reg(self.convert(self.convert(v, dt), raw)) for v in vals]
        space, base, off = self.address(buf, index, flat)
        if not atomic:
            self.emit(f"st.{space}{vec(regs)}.{dt.mem} {mem(base, off)}, {group(regs)};")
        elif dt in RED_ADD:
            for i, v in enumerate(regs):
                self.emit(f"red.{space}.{RED_ADD[dt]} {mem(base, off + dt.nbytes * i)}, {v};")
        else:
            raise NotImplementedError(f"atomic add on {dt.name}")

    # =========================================================================
    # Whole kernel
    # =========================================================================
    def compile(self) -> str:
        for n in self.arg_order:                 # params -> global pointers
            raw = self.reg("rd")
            self.ptr[n] = self.reg("rd")
            self.emit(f"ld.param.u64 {raw}, [param_{n}];",
                      f"cvta.to.global.u64 {self.ptr[n]}, {raw};")
        self.body(self.kop.body)

        if self.dynamic_smem and self.smem:
            self.globals.insert(0, ".extern .shared .align 16 .b8 smem_pool[];")
        params = ",\n".join(f"\t.param .u64 param_{n}" for n in self.arg_order)
        decls = [f"\t.reg .{t} %{k}<{self.count[k] + 1}>;" for k, t in REG_DECLS.items()]
        return "\n".join([PTX_HEADER, *self.globals, "",
                          f".visible .entry {self.kop.name}(", params, ")", "{",
                          *decls, "", *self.lines, "DONE:", "\tret;", "}", ""])


# =============================================================================
# Expressions: one subclass per IR expression kind.
#   generate() -> Python int (folded), %r register, or %p predicate
# =============================================================================
class Expression:
    kinds: ClassVar[dict[str, type]] = {}

    def __init_subclass__(cls, kind: Optional[str] = None, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        if kind is not None:
            Expression.kinds[kind] = cls

    def __init__(self, em: PTXEmitter, node: Any) -> None:
        self.em, self.node = em, node

    @staticmethod
    def lower(em: PTXEmitter, node: Any) -> Any:
        kind = type(node).__name__
        if kind not in Expression.kinds:
            raise NotImplementedError(
                f"Kernel IR expression {kind} (add class {kind}Expression(Expression, kind={kind!r}))")
        return Expression.kinds[kind](em, node).generate()

    def generate(self) -> Any:
        raise NotImplementedError


def as_index(em: PTXEmitter, v: Val) -> Any:
    """A Val used in index arithmetic: bool -> %p, int -> %r (s32) or a folded int."""
    if v.dt.is_float:
        raise NotImplementedError(f"{v.dt.name} value used as an index (missing an int cast?)")
    if not v.reg.startswith("%"):                # constant
        return int(v.reg)
    if v.dt == BOOL:
        return v.reg
    return em.convert(v, I32).reg


class LiteralExpression(Expression, kind="Literal"):
    def generate(self) -> Any:
        v = self.node.value
        return v if isinstance(v, bool) else int(v)


class VarExpression(Expression, kind="Var"):
    def generate(self) -> Any:
        name = self.node.name
        if name in self.em.idx:
            return self.em.idx[name]
        return as_index(self.em, self.em.vals[name])     # a loaded value (gather)


class TernaryExpression(Expression, kind="TernaryExpr"):
    """c ? a : b  ->  selp (or folded when c is a constant)."""

    def generate(self) -> Any:
        e, em = self.node, self.em
        c = em.expr(e.cond)
        if not is_reg(c):
            return em.expr(e.if_true if c else e.if_false)
        a, b = em.expr(e.if_true), em.expr(e.if_false)
        d = em.reg("r")
        em.emit(f"selp.b32 {d}, {a}, {b}, {em.pred(c)};")
        return d


class BinaryExpression(Expression, kind="BinaryExpr"):
    def generate(self) -> Any:
        e, em = self.node, self.em
        l, r = em.expr(e.left), em.expr(e.right)
        if not is_reg(l) and not is_reg(r):
            return FOLD[e.op](l, r)
        if e.op in CMP_OPS:
            p = em.reg("p")
            em.emit(f"setp.{CMP_OPS[e.op][0]}.s32 {p}, {l}, {r};")
            return p
        if e.op in LOGIC_OPS:
            p = em.reg("p")
            em.emit(f"{LOGIC_OPS[e.op][0]} {p}, {em.pred(l)}, {em.pred(r)};")
            return p
        if (e.op, r) in (("+", 0), ("*", 1)):    # x+0, x*1
            return l
        d = em.reg("r")
        em.emit(f"{INT_INDEX_OPS[e.op][0]} {d}, {l}, {r};")
        return d


class CastExpression(Expression, kind="CastExpr"):
    """(int)x inside an index, e.g. a gather through loaded indices.
    Casts in value context ((float)x in a Let) go through em.value_of."""

    def generate(self) -> Any:
        return as_index(self.em, self.em.value_of(self.node))


# =============================================================================
# Statements: one subclass per IR statement kind. generate() emits PTX.
# =============================================================================
class Statement:
    kinds: ClassVar[dict[str, type]] = {}

    def __init_subclass__(cls, kind: Optional[str] = None, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        if kind is not None:
            Statement.kinds[kind] = cls

    def __init__(self, em: PTXEmitter, node: Any) -> None:
        self.em, self.node = em, node

    @staticmethod
    def lower(em: PTXEmitter, node: Any) -> None:
        kind = type(node).__name__
        if kind not in Statement.kinds:
            raise NotImplementedError(
                f"Kernel IR statement {kind} (add class {kind}Statement(Statement, kind={kind!r}))")
        Statement.kinds[kind](em, node).generate()

    def generate(self) -> None:
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Launch
# -----------------------------------------------------------------------------
class TileStatement(Statement, kind="Tile"):
    """gid = ctaid.x * ntid.x + tid.x; exit if gid >= N; decode the axes
    row-major from gid (innermost axis last)."""

    def generate(self) -> None:
        t, em = self.node, self.em
        exts = [int(a.extent.value) for a in t.axes]
        n = math.prod(exts)
        tid, bid, bdim, gid, lane, warp = (em.reg("r") for _ in range(6))
        em.emit(f"mov.u32 {tid}, %tid.x;", f"mov.u32 {bid}, %ctaid.x;",
                f"mov.u32 {bdim}, %ntid.x;", f"mad.lo.s32 {gid}, {bid}, {bdim}, {tid};",
                f"and.b32 {lane}, {tid}, 31;", f"shr.u32 {warp}, {tid}, 5;")
        if em.launch_threads is None or em.launch_threads > n:   # surplus threads
            p = em.reg("p")
            em.emit(f"setp.ge.s32 {p}, {gid}, {n};", f"@{p} bra DONE;")
        em.idx.update(tid=tid, lane=lane, warp=warp)
        rest = gid
        for axis, ext in reversed(list(zip(t.axes, exts))):
            if ext == n:                         # the only non-trivial axis
                em.idx[axis.name] = rest
                continue
            a, q = em.reg("r"), em.reg("r")
            em.emit(f"rem.s32 {a}, {rest}, {ext};", f"div.s32 {q}, {rest}, {ext};")
            em.idx[axis.name], rest = a, q
        em.body(t.body)


# -----------------------------------------------------------------------------
# Scalar compute
# -----------------------------------------------------------------------------
class LoadStatement(Statement, kind="Load"):
    def generate(self) -> None:
        s, em = self.node, self.em
        for name, v in zip(s.names, em.load(s.input, s.index, len(s.names))):
            em.bind(name, em.convert(v, dtype_of(s.dtype, v.dt)))


class AssignStatement(Statement, kind="Assign"):
    def generate(self) -> None:
        s, em = self.node, self.em
        out = dtype_of(s.dtype)
        v = em.apply(s.op.name, [em.value(a) for a in s.args], out)
        em.bind(s.name, em.convert(v, out or v.dt))


class LetStatement(Statement, kind="Let"):
    def generate(self) -> None:
        s, em = self.node, self.em
        em.bind(s.name, em.value_of(s.value, dtype_of(s.dtype)))


class InitStatement(Statement, kind="Init"):
    """Carried state seeded with a literal (matmul accumulator tiles)."""

    def generate(self) -> None:
        s, em = self.node, self.em
        em.bind(s.name, em.const(s.identity, dtype_of(s.dtype, F32)))
        em.inits.add(s.name)


class WriteStatement(Statement, kind="Write"):
    def generate(self) -> None:
        s, em = self.node, self.em
        em.store(s.output, s.index, [em.value(v) for v in s.values], s.atomic)


# -----------------------------------------------------------------------------
# Control flow
# -----------------------------------------------------------------------------
def iter_accums(stmts: Any):
    """All Accum statements under `stmts`, recursively."""
    for s in stmts or ():
        if type(s).__name__ == "Accum":
            yield s
        yield from iter_accums(getattr(s, "body", None))
        yield from iter_accums(getattr(s, "else_body", None))


def seed(em: PTXEmitter, s: Any) -> None:
    """Reset accumulator s to its op's identity."""
    dt = dtype_of(s.dtype, F32)
    em.bind(s.name, em.const(IDENTITY[s.op.name], dt))


class LoopStatement(Statement):
    """Shared base for Loop and StridedLoop (not registered itself)."""

    def seed_accums(self) -> None:
        """A loop with seed=True resets, just before it starts, every accumulator
        that reduces over its axis. Nested loops therefore restart an inner
        accumulator on each outer iteration (attention's q.k dot product), while
        Init-declared carried state (matmul tiles) is never reset."""
        loop, em = self.node, self.em
        if loop.seed:
            for s in iter_accums(loop.body):
                if loop.axis.name in s.axes and s.name not in em.inits:
                    seed(em, s)

    def rolled(self, start: Any, end: Any, step: Any) -> None:
        """for (axis = start; axis < end; axis += step) body"""
        em = self.em
        i, p = em.reg("r"), em.reg("p")
        top, done = em.label("LOOP"), em.label("LOOP_END")
        em.emit(f"mov.s32 {i}, {start};", f"{top}:",
                f"setp.ge.s32 {p}, {i}, {end};", f"@{p} bra {done};")
        em.idx[self.node.axis.name] = i
        em.body(self.node.body)
        em.emit(f"add.s32 {i}, {i}, {step};", f"bra {top};", f"{done}:")


class StridedLoopStatement(LoopStatement, kind="StridedLoop"):
    def generate(self) -> None:
        s, em = self.node, self.em
        self.seed_accums()
        end = em.expr(s.end) if s.end is not None else int(s.axis.extent.value)
        self.rolled(em.expr(s.start), end, em.expr(s.step))


class ForLoopStatement(LoopStatement, kind="Loop"):
    """for axis in 0..extent. Unrolled when the IR asks and the loop is
    innermost (matmul's k loop): the axis becomes a Python int and every
    index constant-folds. Outer loops stay rolled even if marked unroll:
    attention's key loop wraps four 64-step dot products, and unrolling it
    exhausts registers."""

    def generate(self) -> None:
        s, em = self.node, self.em
        self.seed_accums()
        n = int(s.axis.extent.value)
        innermost = not any(type(c).__name__ in ("Loop", "StridedLoop") for c in s.body)
        if s.unroll and innermost and n <= 128:
            for i in range(n):
                em.idx[s.axis.name] = i
                em.body(s.body)
        else:
            self.rolled(0, n, 1)


class CondStatement(Statement, kind="Cond"):
    def generate(self) -> None:
        s, em = self.node, self.em
        c = em.expr(s.cond)
        if not is_reg(c):                        # constant condition
            em.body(s.body if c else s.else_body)
            return
        end = em.label("ENDIF")
        els = em.label("ELSE") if s.else_body else end
        em.emit(f"@!{em.pred(c)} bra {els};")
        em.body(s.body)
        if s.else_body:
            em.emit(f"bra {end};", f"{els}:")
            em.body(s.else_body)
        em.emit(f"{end}:")


# -----------------------------------------------------------------------------
# Reductions
# -----------------------------------------------------------------------------
class AccumStatement(Statement, kind="Accum"):
    """acc = op(base, value) in place; base defaults to acc itself. The online
    softmax uses base for the rescaled old sum: acc = acc*alpha + x*beta."""

    def generate(self) -> None:
        s, em = self.node, self.em
        if s.name not in em.vals:                # no seeding loop above it
            seed(em, s)
        acc = em.vals[s.name]
        base = em.value(s.base) if s.base else acc
        em.assign(acc, em.apply(s.op.name, [base, em.value(s.value)], acc.dt))


class CombiningStatement(Statement):
    """Shared base for cross-thread reductions (not registered itself)."""

    def combine(self, others: Sequence[Val]) -> None:
        """Bind the state_b names to `others`, then run the combine statements."""
        s, em = self.node, self.em
        em.vals.update(zip(s.state_b, others))
        em.body(s.combine_states)


class WarpShuffleStatement(CombiningStatement, kind="WarpShuffle"):
    """XOR butterfly over `length` lanes: offsets length/2, ..., 1."""

    def generate(self) -> None:
        s, em = self.node, self.em
        off = s.length // 2
        while off >= 1:
            self.combine([em.shuffle(em.vals[st], off) for st in s.state])
            off //= 2


class TreeHalveStatement(CombiningStatement, kind="TreeHalve"):
    """Shared-memory tree over `length` slots, then broadcast slot 0:

        for (h = length/2; h > 0; h /= 2) {          // unrolled: length is static
            if (tid < h) { st = buf[tid]; st_b = buf[tid+h]; combine; buf[tid] = st; }
            barrier
        }
        st = buf[0]
    """

    def generate(self) -> None:
        s, em = self.node, self.em
        if s.inner:
            raise NotImplementedError("TreeHalve.inner")
        tid = em.idx[s.tid_var]
        bar = f"bar.sync {s.barrier_id}" + (f", {s.barrier_count}" if s.barrier_count else "") + ";"
        h = s.length // 2
        while h >= 1:
            p, partner, skip = em.reg("p"), em.reg("r"), em.label("TREE_SKIP")
            em.emit(f"setp.ge.s32 {p}, {tid}, {h};", f"@{p} bra {skip};",
                    f"add.s32 {partner}, {tid}, {h};")
            others = []
            for buf, st in zip(s.bufs, s.state):
                em.assign(em.vals[st], em.load(buf, flat=tid)[0])
                others.append(em.convert(em.load(buf, flat=partner)[0], em.vals[st].dt))
            if s.tid_var == "warp":
                # All 32 lanes share one slot, so every lane must load before any
                # lane stores. Converged warps do this anyway; make it explicit so
                # it also holds under independent thread scheduling.
                em.emit("bar.warp.sync -1;")
            self.combine(others)
            for buf, st in zip(s.bufs, s.state):
                em.store(buf, None, [em.vals[st]], flat=tid)
            em.emit(f"{skip}:", bar)
            h //= 2
        for buf, st in zip(s.bufs, s.state):     # broadcast the result
            em.assign(em.vals[st], em.load(buf, flat=0)[0])


# -----------------------------------------------------------------------------
# Shared memory, barriers, async copies
# -----------------------------------------------------------------------------
class SmemStatement(Statement, kind="Smem"):
    """Static `.shared` byte array, or a slice of the dynamic pool when
    meta.json sets smem_bytes (needed above the 48 KB static limit, e.g. m01)."""

    def generate(self) -> None:
        s, em = self.node, self.em
        dt = dtype_of(s.dtype, F32)
        extents = tuple(int(x) for x in s.extents)
        nbytes = dt.nbytes * math.prod(extents)
        align = max(int(s.align or 0), dt.nbytes)
        em.smem[s.name] = (extents, dt)
        base = em.reg("rd")
        em.ptr[s.name] = base
        if em.dynamic_smem:
            em.smem_offset = -(-em.smem_offset // align) * align   # round up
            em.emit(f"mov.u64 {base}, smem_pool;", f"add.u64 {base}, {base}, {em.smem_offset};")
            em.smem_offset += nbytes
        else:
            em.globals.append(f".shared .align {align} .b8 {s.name}[{nbytes}];")
            em.emit(f"mov.u64 {base}, {s.name};")


class SyncStatement(Statement, kind="Sync"):
    def generate(self) -> None:
        s, em = self.node, self.em
        if s.warp:
            em.emit("bar.warp.sync -1;")
        elif s.count:
            em.emit(f"bar.sync {s.barrier_id}, {s.count};")
        else:
            em.emit(f"bar.sync {s.barrier_id};")


class CpAsyncCopyStatement(Statement, kind="CpAsyncCopy"):
    """smem[smem_index + lane_index] <- src[src_index], nbytes at once.
    Both shared-side indices flatten against the smem extents and add."""

    def generate(self) -> None:
        s, em = self.node, self.em
        if s.swizzle not in (None, "NONE"):
            raise NotImplementedError("CpAsyncCopy swizzle")
        flat = em.flat_index(s.smem, s.smem_index)
        if s.lane_index:
            lane = em.flat_index(s.smem, s.lane_index)
            if not is_reg(flat) and not is_reg(lane):
                flat += lane
            else:
                t = em.reg("r")
                em.emit(f"add.s32 {t}, {flat}, {lane};")
                flat = t
        _, dbase, doff = em.address(s.smem, flat=flat)
        _, sbase, soff = em.address(s.src, s.src_index)
        cache = "cg" if s.nbytes == 16 else "ca"   # .cg only supports 16 bytes
        em.emit(f"cp.async.{cache}.shared.global {mem(dbase, doff)}, {mem(sbase, soff)}, {s.nbytes};")


class CpAsyncCommitStatement(Statement, kind="CpAsyncCommit"):
    def generate(self) -> None:
        self.em.emit("cp.async.commit_group;")


class CpAsyncWaitStatement(Statement, kind="CpAsyncWait"):
    def generate(self) -> None:
        self.em.emit(f"cp.async.wait_group {self.node.group};")


# =============================================================================
# IR optimizer (Kernel IR -> Kernel IR), run by main() before the emitter.
#
# It targets reduction loops: a Loop / StridedLoop whose body is
# Load -> pointwise ops -> Accum. A loop like
#
#     for (a = t; a < N; a += S)  { x = buf[.., a];  acc = op(acc, f(x)); }
#
# is rewritten, with vector width V, unroll factor U and A partial
# accumulators, into
#
#     for (g = t; g < N/V; g += S*U) {      // g counts groups of V elements
#         load the U groups of this iteration (V-wide when `a` is contiguous)
#         U*V copies of f; copy c accumulates into acc__p{c % A}
#     }
#     acc = op(op(acc__p0, acc__p1), ...)   // combine the partials
#
# analyze() decides whether that is legal and picks V, U and A.
# rewrite() applies the decision. The launch geometry never changes.
# Set EMMY_NO_OPT=1 to turn the pass off.
# =============================================================================

MAX_VEC_ELEMS = 4            # widest vector load: ld.v4
MAX_VEC_BYTES = 16           # ...and at most 128 bits
UNROLL_CHOICES = (8, 4, 2)   # tried largest first
MAX_BYTES_IN_FLIGHT = 256    # loaded per thread per iteration
MAX_BODY_STMTS = 512         # cap on the unrolled body size
MAX_ACCUMULATORS = 8         # more independent chains stop helping
REDUCE_OPS = {"add", "multiply", "maximum", "minimum"}   # ops IDENTITY can seed
LOOP_BODY_KINDS = {"Load", "Assign", "Let", "Accum"}


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def kind(s: Any) -> str:
    return type(s).__name__


def lit(c: int) -> Literal:
    return Literal(c, "int")


def int_value(e: Any) -> Optional[int]:
    """The value of an integer Literal, else None."""
    if kind(e) == "Literal" and not isinstance(e.value, bool):
        return int(e.value)
    return None


def mul(e: Any, c: int) -> Any:
    return e if c == 1 else BinaryExpr("*", e, lit(c))


def add(e: Any, c: int) -> Any:
    return e if c == 0 else BinaryExpr("+", e, lit(c))


def static_extent(axis: Any) -> Optional[int]:
    try:
        return int(axis.extent.value)
    except Exception:                         # symbolic extent
        return None


def children(s: Any) -> list:
    """The statement bodies nested directly inside s."""
    return [b for b in (getattr(s, "body", None), getattr(s, "else_body", None)) if b]


def walk(stmts: Any):
    for s in stmts or ():
        yield s
        for b in children(s):
            yield from walk(b)


def contains(s: Any, target: Any) -> bool:
    return s is target or any(contains(c, target) for b in children(s) for c in b)


# -----------------------------------------------------------------------------
# What the analyzer knows about the whole kernel
# -----------------------------------------------------------------------------
@dataclass
class KernelInfo:
    graph: Any
    inits: set    # Init-declared names: carried state, never split
    smem: set     # shared-memory buffers: never vector-loaded
    names: set    # every name in use, so new names don't collide

    @classmethod
    def of(cls, graph: Any, kop: Any) -> "KernelInfo":
        info = cls(graph, set(), set(), set(graph.nodes))
        for s in walk(kop.body):
            k = kind(s)
            if k == "Init":
                info.inits.add(s.name)
            if k == "Smem":
                info.smem.add(s.name)
            if k in ("Smem", "Init", "Accum"):
                info.names.add(s.name)
            info.names.update(s.defines())    # not s.name: it asserts on vector Loads
            if hasattr(s, "axis"):
                info.names.add(s.axis.name)
            for a in getattr(s, "axes", ()) or ():
                info.names.add(getattr(a, "name", a))
        return info

    def fresh(self, base: str) -> str:
        name, n = base, 0
        while name in self.names:
            n += 1
            name = f"{base}_{n}"
        self.names.add(name)
        return name

    def nbytes(self, buf: str) -> int:
        node = self.graph.nodes.get(buf)
        return int(node.outputs[0].dtype.nbytes) if node else 4

    def last_dim(self, buf: str) -> Optional[int]:
        """Size of buf's contiguous dimension, or None if buf can't be vector-loaded."""
        node = self.graph.nodes.get(buf)
        if buf in self.smem or node is None or node.outputs[0].dtype.name == "bool":
            return None
        shape = node.outputs[0].shape
        return int(shape[-1].value) if len(shape) else 1


# -----------------------------------------------------------------------------
# Analyzer
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class LoopPlan:
    vec: int                   # V: elements per group (1 = no vector loads)
    unroll: int                # U: groups per iteration
    n_acc: int                 # A: partial accumulators per Accum
    vector_loads: frozenset    # body positions of the Loads done V-wide
    elems: int                 # N: extent of the reduced axis
    step: int                  # S: thread stride (1 for Loop)
    strided: bool


def analyze(loop: Any, info: KernelInfo, after: Sequence, thread_axes: dict,
            enclosing: Sequence) -> Optional[LoopPlan]:
    """The plan for rewriting `loop`, or None to leave it alone.

    after        statements that run after the loop
    thread_axes  Tile axes in scope: name -> extent
    enclosing    bodies of the loops around this one
    """
    N = static_extent(loop.axis)
    space = iteration_space(loop, thread_axes)
    if N is None or space is None:
        return None
    S, first_start, last_start = space
    strided = kind(loop) == "StridedLoop"

    if not is_plain_reduction(loop, info):
        return None
    if not body_values_die_with_loop(loop, after, enclosing):
        return None
    if strided and N < S:      # more threads than elements: the Tile shape is the issue
        return None

    # Vector loads change which elements a thread owns, so the threads must
    # partition [0, S) exactly. Unrolling only needs every thread to start below S.
    covers_exactly = (first_start, last_start) == (0, S - 1)
    V, vector_loads = choose_vector_width(loop, info, N, covers_exactly)
    U = choose_unroll(loop, info, N // V, V, S, strided, last_start < S)
    if V * U == 1:
        return None
    return LoopPlan(V, U, min(V * U, MAX_ACCUMULATORS), vector_loads, N, S, strided)


def iteration_space(loop: Any, thread_axes: dict) -> Optional[tuple]:
    """(step, smallest start, largest start) over all threads, or None if unknown."""
    if kind(loop) == "Loop":
        return 1, 0, 0
    step = int_value(loop.step)
    if step is None or step <= 0:
        return None
    if loop.end is not None and int_value(loop.end) != static_extent(loop.axis):
        return None
    start = loop.start
    if int_value(start) is not None:
        return step, int_value(start), int_value(start)
    if kind(start) == "Var" and start.name in thread_axes:
        return step, 0, thread_axes[start.name] - 1
    return None


def is_plain_reduction(loop: Any, info: KernelInfo) -> bool:
    """Straight-line Load -> pointwise -> Accum body, where every Accum is a
    reorderable reduction over this loop and nothing else reads it."""
    body, axis = list(loop.body), loop.axis.name
    if not loop.seed or any(kind(s) not in LOOP_BODY_KINDS for s in body):
        return False

    accums = [s for s in body if kind(s) == "Accum"]
    acc_names = {a.name for a in accums}
    if not accums or len(acc_names) < len(accums):
        return False
    for a in accums:
        if (a.base is not None                        # online-softmax style recurrence
                or a.op.name not in REDUCE_OPS
                or not (a.op.associative and a.op.commutative)
                or a.name in info.inits
                or axis not in a.axes):
            return False

    defined = [n for s in body if kind(s) != "Accum" for n in s.defines()]
    if len(set(defined)) < len(defined) or set(defined) & acc_names:
        return False                                  # a name is rebound in the loop
    for s in body:
        if set(s.deps()) & acc_names:                 # an accumulator is read in the loop
            return False
        if kind(s) == "Assign" and axis in s.args:    # loop index used as a value
            return False
        if kind(s) == "Load" and any(e.free_vars() & set(defined) for e in s.index):
            return False                              # gather through a computed index
    return True


def body_values_die_with_loop(loop: Any, after: Sequence, enclosing: Sequence) -> bool:
    """The rewrite renames every value the body defines, so none may be read
    after the loop, or before it by the next iteration of an enclosing loop."""
    local = {n for s in loop.body if kind(s) != "Accum" for n in s.defines()}
    live = set(local)
    for s in after:
        if free_names(s) & live:
            return False
        live -= set(s.defines())
    return not any(free_names(s) & local
                   for body in enclosing for s in body if not contains(s, loop))


def choose_vector_width(loop: Any, info: KernelInfo, N: int, covers_exactly: bool) -> tuple:
    """(V, positions of the Loads that become V-wide). V = 1 when nothing qualifies."""
    if not covers_exactly:
        return 1, frozenset()
    axis = loop.axis.name

    def fits(s: Any, v: int) -> bool:
        """s reads `axis` as its contiguous dimension, and a v-wide load stays aligned."""
        if len(s.names) != 1 or not s.index:
            return False
        last = s.index[-1]
        if kind(last) != "Var" or last.name != axis:
            return False
        if any(axis in e.free_vars() for e in s.index[:-1]):
            return False
        dim = info.last_dim(s.input)
        return dim is not None and dim % v == 0 and v * info.nbytes(s.input) <= MAX_VEC_BYTES

    for v in (4, 2):
        if v > MAX_VEC_ELEMS or N % v:
            continue
        wide = frozenset(i for i, s in enumerate(loop.body) if kind(s) == "Load" and fits(s, v))
        if wide:
            return v, wide
    return 1, frozenset()


def choose_unroll(loop: Any, info: KernelInfo, groups: int, V: int, S: int,
                  strided: bool, starts_below_step: bool) -> int:
    """The largest U that divides every thread's trip count and stays within
    the bytes-in-flight and body-size budgets."""
    body = list(loop.body)
    bytes_per_copy = sum(V * info.nbytes(s.input) * len(s.names)
                         for s in body if kind(s) == "Load")
    for u in UNROLL_CHOICES:
        if strided:
            divides = starts_below_step and groups % (S * u) == 0
        else:
            divides = groups % u == 0
        within_budget = (u * bytes_per_copy <= MAX_BYTES_IN_FLIGHT
                         and u * V * len(body) <= MAX_BODY_STMTS)
        if divides and within_budget:
            return u
    return 1


# -----------------------------------------------------------------------------
# Rewriter
# -----------------------------------------------------------------------------
def rewrite(loop: Any, plan: LoopPlan, info: KernelInfo) -> list:
    """[new loop, combine statements...] that replace `loop`."""
    V, U, A, S = plan.vec, plan.unroll, plan.n_acc, plan.step
    axis, body = loop.axis.name, list(loop.body)
    groups = plan.elems // V

    # The new loop counts groups. Copy k of iteration i handles group
    #   i + k*S   (StridedLoop: threads interleave)   or
    #   i*U + k   (Loop: one thread does every group)
    # and element j of that group is group*V + j.
    i = Var(info.fresh(f"{axis}_v{V}u{U}"))
    if plan.strided:
        new_axis = replace(loop.axis, name=i.name, extent=Dim(groups))
        element = lambda k, j: add(mul(i, V), k * S * V + j)
    else:
        new_axis = replace(loop.axis, name=i.name, extent=Dim(groups // U))
        element = lambda k, j: add(mul(i, U * V), k * V + j)

    # Every value the body defines gets one name per copy (k, j).
    defined = sorted({n for s in body if kind(s) != "Accum" for n in s.defines()})
    copy_name = {(n, k, j): info.fresh(f"{n}__u{k}v{j}")
                 for n in defined for k in range(U) for j in range(V)}
    accums = [s for s in body if kind(s) == "Accum"]
    partial = {(a.name, c): info.fresh(f"{a.name}__p{c}") for a in accums for c in range(A)}

    def rn(name: str, k: int, j: int) -> str:
        return copy_name.get((name, k, j), name)

    # All loads first, so every load of the iteration is in flight together.
    loads = []
    for k in range(U):
        for pos, s in enumerate(body):
            if kind(s) != "Load":
                continue
            if pos in plan.vector_loads:          # one V-wide load
                loads.append(replace(
                    s, names=tuple(rn(s.names[0], k, j) for j in range(V)),
                    index=(*s.index[:-1], element(k, 0))))
            else:                                 # V scalar loads
                for j in range(V):
                    loads.append(replace(
                        s, names=tuple(rn(n, k, j) for n in s.names),
                        index=tuple(e.substitute({axis: element(k, j)}) for e in s.index)))

    # Then the compute, one copy per element; copy c feeds partial c % A.
    compute = []
    for k in range(U):
        for j in range(V):
            c = k * V + j
            for s in body:
                if kind(s) == "Let":
                    renames = {n: Var(rn(n, k, j)) for n in defined}
                    compute.append(replace(s, name=rn(s.name, k, j), value=s.value.substitute(
                        {axis: element(k, j), **renames})))
                elif kind(s) == "Assign":
                    compute.append(replace(s, name=rn(s.name, k, j),
                                           args=tuple(rn(a, k, j) for a in s.args)))
                elif kind(s) == "Accum":
                    compute.append(replace(
                        s, name=partial[(s.name, c % A)], value=rn(s.value, k, j),
                        axes=tuple(i.name if a == axis else a for a in s.axes)))

    if plan.strided:
        new_loop = replace(loop, axis=new_axis, step=lit(S * U), end=None,
                           body=(*loads, *compute))
    else:
        new_loop = replace(loop, axis=new_axis, body=(*loads, *compute))

    combine = []
    for a in accums:
        combine += combine_partials(a, [partial[(a.name, c)] for c in range(A)], info)
    return [new_loop, *combine]


def combine_partials(acc: Any, parts: list, info: KernelInfo) -> list:
    """Pairwise tree of Assigns folding `parts` into acc.name."""
    stmts = []
    while len(parts) > 1:
        merged = []
        for x, y in zip(parts[0::2], parts[1::2]):
            out = acc.name if len(parts) == 2 else info.fresh(f"{acc.name}__t")
            stmts.append(Assign(name=out, op=acc.op, args=(x, y), dtype=acc.dtype))
            merged.append(out)
        if len(parts) % 2:
            merged.append(parts[-1])
        parts = merged
    return stmts


# -----------------------------------------------------------------------------
# Pass driver
# -----------------------------------------------------------------------------
def optimize(graph: Any, kop: Any) -> Any:
    """kop with every rewritable reduction loop rewritten."""
    info = KernelInfo.of(graph, kop)

    def visit(stmts: Sequence, after: tuple, thread_axes: dict, enclosing: tuple) -> tuple:
        stmts, out = tuple(stmts or ()), []
        for n, s in enumerate(stmts):
            rest = stmts[n + 1:] + after
            k = kind(s)
            if k == "Tile":
                axes = {a.name: static_extent(a) for a in s.axes}
                out.append(replace(s, body=visit(s.body, rest, axes, enclosing)))
            elif k in ("Loop", "StridedLoop"):
                plan = analyze(s, info, rest, thread_axes, enclosing)
                if plan:
                    out += rewrite(s, plan, info)
                else:
                    out.append(replace(s, body=visit(s.body, rest, thread_axes,
                                                     enclosing + (s.body,))))
            elif k == "Cond":
                out.append(replace(s, body=visit(s.body, rest, thread_axes, enclosing),
                                   else_body=visit(s.else_body, rest, thread_axes, enclosing)))
            else:
                out.append(s)
        return tuple(out)

    return replace(kop, body=visit(kop.body, (), {}, ()))


# =============================================================================
# Driver
# =============================================================================
def launch_info(ir_path: Path, entry: str) -> tuple[Optional[list[str]], Optional[int], int]:
    """(arg_order, threads launched, dynamic smem bytes) from meta.json."""
    meta = ir_path.parent / "meta.json"
    if meta.exists():
        for k in json.loads(meta.read_text())["kernels"]:
            if k["entry"] == entry:
                threads = math.prod(k["grid"]) * math.prod(k["block"])
                return k["arg_order"], threads, k.get("smem_bytes", 0)
    return None, None, 0


def main(path: Union[str, Path]) -> None:
    path = Path(path)
    graph = Graph.from_dict(json.loads(path.read_text()))
    node = next(n for n in graph.nodes.values() if isinstance(n.op, KernelOp))
    arg_order, threads, smem_bytes = launch_info(path, node.op.name)
    arg_order = arg_order or [*node.inputs, node.id]     # fallback: inputs, then output

    kop = node.op
    kop = optimize(graph, kop)
    sys.stdout.write(PTXEmitter(graph, kop, arg_order, threads, smem_bytes).compile())


if __name__ == "__main__":
    main(sys.argv[1])