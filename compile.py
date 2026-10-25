#!/usr/bin/env python3
"""Kernel IR -> PTX compiler for the Emmy hackathon.

    python compile.py path/to/kernel.ir.json > out.ptx

Phase 1 covers pointwise kernels: Tile, Load, Assign, Let, Write.
Statements that are not implemented yet fail with a clear message on stderr;
stdout carries PTX and nothing else.

How it works (the four stages from TASK.md):
  1. load     read the IR with emmy's Graph.from_dict, read launch info from meta.json
  2. prologue .version/.target, entry signature, param loads, cvta
  3. walk     visit statements in order; `env` maps IR names to registers/constants
  4. emit     one small handler (stmt_<Kind>) per statement kind

Registers are virtual and never reused: ptxas does the real allocation.
"""
from __future__ import annotations

import json
import math
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

from emmy.compiler.graph import Graph

PTX_VERSION = "8.0"   # sm_89 needs >= 7.8
PTX_TARGET = "sm_89"  # the runner is sm_120; sm_89 PTX runs unchanged there


class CompileError(Exception):
    pass


# --------------------------------------------------------------------------
# Values: either a compile-time constant or a virtual register
# --------------------------------------------------------------------------
class Val:
    """ty is 's32', 'f32' or 'pred'. v is a Python number (constant) or a register name."""

    __slots__ = ("ty", "v")

    def __init__(self, ty: str, v):
        self.ty, self.v = ty, v

    @property
    def is_const(self) -> bool:
        return not isinstance(self.v, str)


def f32_hex(x: float) -> str:
    """PTX float immediates are written as 0f + the IEEE-754 bits in hex."""
    return "0f%08X" % struct.unpack(">I", struct.pack(">f", float(x)))[0]


def opnd(v: Val) -> str:
    if not v.is_const:
        return v.v
    return f32_hex(v.v) if v.ty == "f32" else str(int(v.v))


def ascii_only(text: str) -> str:
    """ptxas rejects non-ASCII anywhere, comments included."""
    return re.sub(r"[^\x20-\x7e]", "?", text)


# --------------------------------------------------------------------------
# PTX text builder: register pools, labels, instruction lines
# --------------------------------------------------------------------------
class Emitter:
    # type -> (register prefix, declared PTX type)
    REG = {
        "f32": ("%f", ".f32"),
        "s32": ("%r", ".b32"),
        "u64": ("%rd", ".b64"),
        "pred": ("%p", ".pred"),
    }

    def __init__(self):
        self.lines: list[str] = []
        self.count: dict[str, int] = defaultdict(int)
        self.nlabels = 0

    def new(self, ty: str) -> str:
        self.count[ty] += 1
        return f"{self.REG[ty][0]}{self.count[ty]}"

    def op(self, text: str):
        self.lines.append(f"        {text};")

    def note(self, text: str):
        self.lines.append(f"        // {ascii_only(text)}")

    def new_label(self) -> str:
        self.nlabels += 1
        return f"L{self.nlabels}"

    def place(self, label: str):
        self.lines.append(f"{label}:")

    def reg_decls(self) -> list[str]:
        out = []
        for ty, (prefix, ptx_ty) in self.REG.items():
            if self.count[ty]:
                out.append(f"        .reg {ptx_ty:<5} {prefix}<{self.count[ty] + 1}>;")
        return out


# --------------------------------------------------------------------------
# Elementwise ops: IR op name -> function(compiler, *args) -> Val
# --------------------------------------------------------------------------
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


def _exp(c, x):
    # e^x = 2^(x * log2 e)
    return c.un("ex2.approx.ftz.f32", c.bin("mul.rn.f32", x, Val("f32", LOG2E), True))


def _sigmoid(c, x):
    e = _exp(c, c.un("neg.f32", x))
    return c.un("rcp.rn.f32", c.bin("add.rn.f32", e, Val("f32", 1.0), True))


ELEMENTWISE = {
    "add": lambda c, a, b: c.bin("add.rn.f32", a, b, True),
    "subtract": lambda c, a, b: c.bin("sub.rn.f32", a, b, False),
    "multiply": lambda c, a, b: c.bin("mul.rn.f32", a, b, True),
    "divide": lambda c, a, b: c.bin("div.rn.f32", a, b, False),
    "maximum": lambda c, a, b: c.bin("max.f32", a, b, True),
    "minimum": lambda c, a, b: c.bin("min.f32", a, b, True),
    "relu": lambda c, x: c.bin("max.f32", x, Val("f32", 0.0), True),
    "copy": lambda c, x: c.un("mov.f32", x),
    "negative": lambda c, x: c.un("neg.f32", x),
    "abs": lambda c, x: c.un("abs.f32", x),
    "reciprocal": lambda c, x: c.un("rcp.rn.f32", x),
    "sqrt": lambda c, x: c.un("sqrt.rn.f32", x),
    "rsqrt": lambda c, x: c.un("rsqrt.approx.f32", x),
    "exp": _exp,
    "log2": lambda c, x: c.un("lg2.approx.f32", x),
    "log": lambda c, x: c.bin("mul.rn.f32", c.un("lg2.approx.f32", x), Val("f32", LN2), True),
    "tanh": lambda c, x: c.un("tanh.approx.f32", x),
    "sigmoid": _sigmoid,
    "silu": lambda c, x: c.bin("mul.rn.f32", x, _sigmoid(c, x), True),
    "fma": lambda c, a, b, d: c.tri("fma.rn.f32", a, b, d),
}
ALIASES = {"sub": "subtract", "true_divide": "divide", "exp_fast": "exp"}

INT_OPS = {"+": "add.s32", "-": "sub.s32", "*": "mul.lo.s32", "/": "div.s32", "%": "rem.s32"}
FLT_OPS = {"+": "add.rn.f32", "-": "sub.rn.f32", "*": "mul.rn.f32", "/": "div.rn.f32"}
COMMUTATIVE = {"+", "*"}
CMP_OPS = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}
ELEM_BYTES = {"f32": 4}


# --------------------------------------------------------------------------
# The compiler proper
# --------------------------------------------------------------------------
class KernelCompiler:
    def __init__(self, name: str, body, arg_order: list[str], shapes: dict[str, tuple]):
        self.name = name
        self.body = body
        self.arg_order = arg_order
        self.shapes = shapes          # buffer name -> tuple of ints
        self.em = Emitter()
        self.env: dict[str, Val] = {}  # IR name (axis var / SSA value) -> Val
        self.ptr: dict[str, str] = {}  # buffer name -> register holding its global address
        self.exit_label = self.em.new_label()

    # ---- stage 2: prologue --------------------------------------------------
    def prologue(self):
        self.em.note("load kernel parameters and convert to global addresses")
        for buf in self.arg_order:
            raw, glob = self.em.new("u64"), self.em.new("u64")
            self.em.op(f"ld.param.u64 {raw}, [param_{self.sym(buf)}]")
            self.em.op(f"cvta.to.global.u64 {glob}, {raw}")
            self.ptr[buf] = glob

    @staticmethod
    def sym(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]", "_", name)

    # ---- stage 3: walk ------------------------------------------------------
    def run(self) -> str:
        self.prologue()
        for st in self.body:
            self.stmt(st)
        return self.assemble()

    def stmt(self, st):
        kind = type(st).__name__
        handler = getattr(self, f"stmt_{kind}", None)
        if handler is None:
            raise CompileError(f"statement {kind} is not supported yet")
        handler(st)

    def assemble(self) -> str:
        em = self.em
        params = ",\n".join(f"        .param .u64 param_{self.sym(b)}" for b in self.arg_order)
        out = [
            f".version {PTX_VERSION}",
            f".target {PTX_TARGET}",
            ".address_size 64",
            "",
            f".visible .entry {self.name}(",
            params,
            ")",
            "{",
            *em.reg_decls(),
            "",
            *em.lines,
            f"{self.exit_label}:",
            "        ret;",
            "}",
            "",
        ]
        text = "\n".join(out)
        text.encode("ascii")  # fail loudly here rather than in ptxas
        return text

    # ---- small instruction helpers -----------------------------------------
    def reg(self, v: Val) -> str:
        """Return a register holding v (materialise constants with mov)."""
        if not v.is_const:
            return v.v
        r = self.em.new(v.ty)
        self.em.op(f"mov.{'f32' if v.ty == 'f32' else 'b32'} {r}, {opnd(v)}")
        return r

    def un(self, ins: str, a: Val) -> Val:
        d = self.em.new("f32")
        self.em.op(f"{ins} {d}, {opnd(a) if a.is_const else a.v}")
        return Val("f32", d)

    def bin(self, ins: str, a: Val, b: Val, commutative: bool, ty: str = "f32") -> Val:
        if a.is_const and not b.is_const and commutative:
            a, b = b, a
        d = self.em.new(ty)
        self.em.op(f"{ins} {d}, {self.reg(a)}, {opnd(b)}")
        return Val(ty, d)

    def tri(self, ins: str, a: Val, b: Val, c: Val) -> Val:
        d = self.em.new("f32")
        self.em.op(f"{ins} {d}, {self.reg(a)}, {self.reg(b)}, {self.reg(c)}")
        return Val("f32", d)

    # ---- expressions (indices, predicates, literals) -----------------------
    def lookup(self, name: str) -> Val:
        if name not in self.env:
            raise CompileError(f"name {name!r} used before it is defined")
        return self.env[name]

    def expr(self, e) -> Val:
        kind = type(e).__name__
        if kind == "Literal":
            return Val("f32", float(e.value)) if e.dtype == "float" else Val("s32", int(e.value))
        if kind == "Var":
            return self.lookup(e.name)
        if kind == "BinaryExpr":
            return self.binary(e.op, self.expr(e.left), self.expr(e.right))
        if kind == "TernaryExpr":
            p, a, b = self.expr(e.cond), self.expr(e.if_true), self.expr(e.if_false)
            ty = "f32" if "f32" in (a.ty, b.ty) else "s32"
            d = self.em.new(ty)
            self.em.op(f"selp.{'f32' if ty == 'f32' else 'b32'} {d}, {self.reg(a)}, {self.reg(b)}, {self.reg(p)}")
            return Val(ty, d)
        raise CompileError(f"expression {kind} is not supported yet")

    def binary(self, op: str, l: Val, r: Val) -> Val:
        if op in ("&&", "||"):
            d = self.em.new("pred")
            self.em.op(f"{'and' if op == '&&' else 'or'}.pred {d}, {self.reg(l)}, {self.reg(r)}")
            return Val("pred", d)
        is_f32 = "f32" in (l.ty, r.ty)
        if is_f32:  # int literal next to a float: promote the constant
            l = Val("f32", float(l.v)) if l.is_const else l
            r = Val("f32", float(r.v)) if r.is_const else r
        if op in CMP_OPS:
            d = self.em.new("pred")
            ty = "f32" if is_f32 else "s32"
            self.em.op(f"setp.{CMP_OPS[op]}.{ty} {d}, {self.reg(l)}, {opnd(r)}")
            return Val("pred", d)
        if op not in INT_OPS:
            raise CompileError(f"binary operator {op!r} is not supported yet")
        if is_f32:
            if op not in FLT_OPS:
                raise CompileError(f"operator {op!r} on floats is not supported")
            return self.bin(FLT_OPS[op], l, r, op in COMMUTATIVE, "f32")
        if l.is_const and r.is_const:  # fold, with C semantics (truncating / and %)
            a, b = int(l.v), int(r.v)
            if op == "+": return Val("s32", a + b)
            if op == "-": return Val("s32", a - b)
            if op == "*": return Val("s32", a * b)
            q = abs(a) // abs(b)
            q = -q if (a < 0) != (b < 0) else q
            return Val("s32", q if op == "/" else a - q * b)
        return self.bin(INT_OPS[op], l, r, op in COMMUTATIVE, "s32")

    def flat_index(self, buffer: str, index) -> Val:
        """Row-major offset (in elements) of a (possibly multi-dim) index into `buffer`."""
        idx = [self.expr(e) for e in index]
        if not idx:
            return Val("s32", 0)
        if len(idx) == 1:  # already a flat offset
            return idx[0]
        shape = self.shapes.get(buffer)
        if shape is None or len(shape) != len(idx):
            raise CompileError(f"cannot flatten {len(idx)}-d index into buffer {buffer!r} with shape {shape}")
        acc = idx[0]
        for k in range(1, len(idx)):
            acc = self.binary("+", self.binary("*", acc, Val("s32", shape[k])), idx[k])
        return acc

    def address(self, buffer: str, offset: Val, elem_bytes: int, k: int = 0) -> str:
        """Operand like [%rd5+8] for element (offset + k) of `buffer`."""
        if buffer not in self.ptr:
            raise CompileError(f"buffer {buffer!r} is not in meta.json arg_order {self.arg_order}")
        base = self.ptr[buffer]
        if offset.is_const:
            return f"[{base}+{(int(offset.v) + k) * elem_bytes}]"
        a = self.em.new("u64")
        self.em.op(f"mad.wide.s32 {a}, {offset.v}, {elem_bytes}, {base}")
        return f"[{a}+{k * elem_bytes}]" if k else f"[{a}]"

    @staticmethod
    def require_f32(dtype, what: str):
        if dtype.name != "f32":
            raise CompileError(f"{what}: dtype {dtype.name} is not supported yet (only f32)")

    # ---- stage 4: statement handlers ---------------------------------------
    def stmt_Tile(self, st):
        em = self.em
        if any(ax.window is not None for ax in st.axes):
            raise CompileError("Tile axes with a window are not supported yet")
        extents = [dim_value(ax.extent) for ax in st.axes]
        total = math.prod(extents)
        em.note(f"Tile [{', '.join(ax.name for ax in st.axes)}] N={total}")

        tid, ctaid, ntid, gid = (em.new("s32") for _ in range(4))
        em.op(f"mov.u32 {tid}, %tid.x")
        em.op(f"mov.u32 {ctaid}, %ctaid.x")
        em.op(f"mov.u32 {ntid}, %ntid.x")
        em.op(f"mad.lo.s32 {gid}, {ctaid}, {ntid}, {tid}")

        # The launch can have more threads than cells; this guard is not in the IR.
        guard = em.new("pred")
        em.op(f"setp.ge.u32 {guard}, {gid}, {total}")
        em.op(f"@{guard} bra {self.exit_label}")

        # Decode gid into one coordinate per axis, last axis fastest.
        rest = gid
        for i in reversed(range(len(st.axes))):
            if i == 0:
                coord = rest
            else:
                coord, nxt = em.new("s32"), em.new("s32")
                em.op(f"rem.s32 {coord}, {rest}, {extents[i]}")
                em.op(f"div.s32 {nxt}, {rest}, {extents[i]}")
                rest = nxt
            self.env[st.axes[i].name] = Val("s32", coord)

        for inner in st.body:
            self.stmt(inner)

    def stmt_Load(self, st):
        self.require_f32(st.dtype, "Load")
        n, eb = len(st.names), ELEM_BYTES["f32"]
        offset = self.flat_index(st.input, st.index)
        regs = [self.em.new("f32") for _ in st.names]
        self.em.note(f"load {', '.join(st.names)} = {st.input}[...]")
        if n in (2, 4):  # one vector load, e.g. ld.global.v4.f32
            self.em.op(f"ld.global.v{n}.f32 {{{', '.join(regs)}}}, {self.address(st.input, offset, eb)}")
        else:
            for k, r in enumerate(regs):
                self.em.op(f"ld.global.f32 {r}, {self.address(st.input, offset, eb, k)}")
        for name, r in zip(st.names, regs):
            self.env[name] = Val("f32", r)

    def stmt_Assign(self, st):
        self.require_f32(st.dtype, "Assign")
        name = ALIASES.get(st.op.name, st.op.name)
        fn = ELEMENTWISE.get(name)
        if fn is None:
            raise CompileError(f"elementwise op {st.op.name!r} is not supported yet")
        self.em.note(f"{st.name} = {name}({', '.join(st.args)})")
        self.env[st.name] = fn(self, *(self.lookup(a) for a in st.args))

    def stmt_Let(self, st):
        # Binds a literal / index expression; constants stay immediates until used.
        self.env[st.name] = self.expr(st.value)

    def stmt_Write(self, st):
        self.require_f32(st.value_dtype, "Write")
        n, eb = len(st.values), ELEM_BYTES["f32"]
        offset = self.flat_index(st.output, st.index)
        regs = [self.reg(self.lookup(v)) for v in st.values]
        self.em.note(f"{st.output}[...] = ({', '.join(st.values)})")
        if st.atomic:  # accumulate into global memory
            for k, r in enumerate(regs):
                self.em.op(f"red.global.add.f32 {self.address(st.output, offset, eb, k)}, {r}")
        elif n in (2, 4):  # one vector store, e.g. st.global.v2.f32
            self.em.op(f"st.global.v{n}.f32 {self.address(st.output, offset, eb)}, {{{', '.join(regs)}}}")
        else:
            for k, r in enumerate(regs):
                self.em.op(f"st.global.f32 {self.address(st.output, offset, eb, k)}, {r}")


# --------------------------------------------------------------------------
# Stage 1: load the IR and launch metadata
# --------------------------------------------------------------------------
def dim_value(d) -> int:
    if type(d.expr).__name__ != "Literal":
        raise CompileError("only static (literal) extents are supported")
    return int(d.expr.value)


def load_problem(ir_path: Path):
    graph = Graph.from_dict(json.loads(ir_path.read_text()))
    kernels = [n for n in graph.nodes.values() if type(n.op).__name__ == "KernelOp"]
    if not kernels:
        raise CompileError("no KernelOp in this IR")
    # the IR file is named <entry>.ir.json; fall back to the first kernel
    node = next((n for n in kernels if ir_path.name.startswith(n.op.name + ".")), kernels[0])

    shapes = {t.name: tuple(dim_value(d) for d in t.shape) for n in graph.nodes.values() for t in n.outputs}

    # meta.json is the contract for parameter order (not the node's inputs list)
    meta_path = ir_path.parent / "meta.json"
    arg_order = None
    if meta_path.exists():
        for k in json.loads(meta_path.read_text()).get("kernels", []):
            if k["entry"] == node.op.name:
                arg_order = list(k["arg_order"])
    if arg_order is None:
        print("warning: no meta.json entry found; assuming inputs + outputs", file=sys.stderr)
        arg_order = list(node.inputs) + [t.name for t in node.outputs]
    return node, arg_order, shapes


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python compile.py path/to/kernel.ir.json > out.ptx", file=sys.stderr)
        return 2
    try:
        node, arg_order, shapes = load_problem(Path(argv[1]))
        ptx = KernelCompiler(node.op.name, node.op.body, arg_order, shapes).run()
    except CompileError as exc:
        print(f"compile error: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(ptx)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))