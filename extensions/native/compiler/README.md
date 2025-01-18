# Expression Evaluation

Currently `builder.eval` always creates a new variable. When the expression is a constant or a single variable, using
`builder.eval` to evaluate an expression causes an unnecessary variable assignment, which impacts performance
significantly.

The naming of `builder.eval` is semantically incorrect because creating a new variable is out of scope of "eval".
However, the assumption is taken too widely to fix. So we introduce a new method `builder.eval_expr` for the real
"evaluation".

`builder.eval_expr` returns a **[right value](https://www.oreilly.com/library/view/c-in-a/059600298X/ch03s01.html#:~:text=The%20term%20rvalue%20is%20a,are%20close%20to%20the%20truth.)**,
which is either a constant or a variable. An instruction can use a right value for reading a value directly.

Currently `builder.eval_expr` only supports `SymbolicVar`.

## RVar
`RVar` represents the right value of `SymbolicVar`.

# CR Variable
To unify static/dynamic programs in eDSL, we introduce a new concept, CR variable("C" for compile time and "R" for
runtime). A CR variable behaves like a normal variable in eDSL(e.g. it supports assignment), but it could be either a 
constant or a variable in runtime.

Currently, there are 2 types of CR variables: 

## Usize
`Usize` could be a constant native field or a `Var` at right time.

## Array
`Array` has 2 variants: `Fixed` and `Dyn`. 

`Fixed` is a logical array only exists in compile time. 
- A static program can only use `Fixed` to represent an array.
- `Fixed` only supports constant index access.
- Try to use `Fixed` if possible because most of its operations don't cost instructions.

`Dyn` is an array on heap.
- A static program cannot use `Dyn`.
- When initializing a `Dyn`, the length could be either a constant or a variable.
- The length of `Dyn` is fixed after initialization.
- `Dyn` supports dynamic index access.

# Branches
In **static programs**, only constant branches are allowed. 

# Loops
## Constant Loops
When both `start` and `end` of a loop are constant, the loop is a constant loop. The loop body will be unrolled. This
optimization saves 1 instruction per iteration.

In **static programs**, only constant loops are allowed.