# AxiomDB Interface

AxiomDB utilizes Apache DataFusion as its main interface. Users are able to specify SQL queries directly or utilize DataFusion's `LogicalPlanBuilder` to generate a `LogicalPlan` tree of nodes. Under the hood, AxiomDB converts the `LogicalPlan` tree to a flat execution-order vector of `AxiomDbNode`s. AxiomDB then runs keygen, prove, and verify on these nodes once the appropriate functions are called.

## Running a test

Execute a test via

```bash
cargo test --release --package axiomdb-interface --test basic_e2e -- test_basic_e2e --exact --show-output
```
