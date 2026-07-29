# Design for GPU tracegen of mod-builder

```text
┌─────────────────────────────────────────────────────────────┐
│                    Mod Builder Frontend                     │
│                                                             │
│  ExprBuilder / FieldVariable / SymbolicExpr                 │
│                                                             │
│  Example:                                                   │
│      z = select(is_mul, x * y, x / y)                       │
└──────────────────────────────┬──────────────────────────────┘
                               │ finalized FieldExpr
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    Tracegen IR Compiler                     │
│                                                             │
│  compile_tracegen_ir()                                      │
│                                                             │
│  • validates CUDA capabilities                              │
│  • lowers expression trees into linear operation tapes      │
│  • derives Montgomery constants                             │
│  • calculates quotient and carry bounds                     │
│  • allocates evaluation slots and witness scratch space     │
└──────────────────────────────┬──────────────────────────────┘
                               │ Result<TracegenIr, Error>
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                        TracegenIr                           │
│                                                             │
│  Evaluation tape                    Witness tapes           │
│  ──────────                         ──────────               │
│  LoadInput / Constant               Input / Var / Constant  │
│  Add / Sub / Mul / Div              Add / Sub               │
│  IntAdd / IntMul                    Mul (limb convolution)   │
│  Select / SaveVar                   IntAdd / IntMul / Select │
│                                                             │
│  Metadata                                                   │
│  ────────                                                   │
│  • field and Montgomery constants                           │
│  • constraint tape boundaries                               │
│  • quotient/carry sizes and bounds                          │
│  • opcode-to-flag mapping                                   │
│  • scratch and trace dimensions                             │
└──────────────────────────────┬──────────────────────────────┘
                               │ encode()
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    Encoded Vec<u32> Blob                    │
│                                                             │
│  • fixed header and section offsets                         │
│  • operation tapes                                          │
│  • constants and metadata                                   │
└──────────────────────────────┬──────────────────────────────┘
                               │ upload once
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                  CUDA Tracegen Interpreter                  │
│                                                             │
│  One thread per trace row:                                  │
│                                                             │
│  1. Decode opcode/inputs                                    │
│  2. Execute evaluation tape                                 │
│  3. Execute witness tapes                                   │
│  4. Calculate q and carries                                 │
│  5. Emit range checks                                       │
│  6. Write trace columns                                     │
└──────────────────────────────┬──────────────────────────────┘
                               │ actual trace
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   Equivalence Validation                    │
│                                                             │
│  CUDA trace == production CPU tracegen                      │
│  (`FieldExpressionFiller`)                                  │
└─────────────────────────────────────────────────────────────┘
```

The encoder and CUDA decoder share a generated ABI:

```text
                        tracegen_abi.def
                               │
                            build.rs
                         ┌─────┴─────┐
                         ▼           ▼
                 Rust constants   CUDA constants
                         │           │
                         ▼           ▼
                      encode()    field_expr.cu
```
