# Design

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
│  • allocates value slots and per-constraint scratch space   │
└──────────────────────────────┬──────────────────────────────┘
                               │ Result<TracegenIr, Error>
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                        TracegenIr                           │
│                                                             │
│  Value tape                         Limb tapes              │
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
└───────────────┬───────────────────────────┬─────────────────┘
                │                           │
                │ interpret                 │ encode()
                ▼                           ▼
┌──────────────────────────────┐  ┌────────────────────────────┐
│ CPU Reference Interpreter    │  │ Encoded Vec<u32> Blob      │
│                              │  │                            │
│ Defines exact semantics and  │  │ • fixed header             │
│ produces reference rows and  │  │ • section offsets          │
│ range-check counts           │  │ • operation tapes          │
└───────────────┬──────────────┘  │ • constants and metadata   │
                │                 └──────────────┬─────────────┘
                │ expected result               │ upload once
                │                                ▼
                │                 ┌────────────────────────────┐
                │                 │ CUDA Tracegen Interpreter  │
                │                 │                            │
                │                 │ One thread per trace row:  │
                │                 │                            │
                │                 │ 1. Decode opcode/inputs    │
                │                 │ 2. Execute value tape      │
                │                 │ 3. Execute limb tapes      │
                │                 │ 4. Calculate q and carries │
                │                 │ 5. Emit range checks       │
                │                 │ 6. Write trace columns     │
                │                 └──────────────┬─────────────┘
                │                                │
                ▼                                ▼
        ┌───────────────────────────────────────────────────┐
        │                 Differential Validation           │
        │                                                   │
        │  Reference row and range counts == generate_subrow│
        │  == CUDA-generated row and range counts           │
        └───────────────────────────────────────────────────┘
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

The runtime data flow is:

```text
Dense execution record
[adapter record | opcode byte | input limbs]
                      │
                      ▼
             CUDA interprets TracegenIr
                      │
                      ▼
Proof trace row
[adapter columns | is_valid | inputs | vars | quotients | carries | flags]
```
