# AxiomDB Interface

AxiomDB utilizes Apache DataFusion as its main interface. Users are able to specify SQL queries directly or utilize DataFusion's `LogicalPlanBuilder` to generate a `LogicalPlan` tree of nodes. Under the hood, AxiomDB converts the `LogicalPlan` tree to a flat execution-order vector of `AxiomDbNode`s. AxiomDB then runs `keygen`, `prove`, and `verify` on these nodes once the appropriate functions are called.

## CommittedPage

Contains information in `AxiomDB`'s `Page` format, with additional required DataFusion `Schema` information for the page. This new format allows us to use AxiomDB data in DataFusion.

## AxiomDbNode

An `AxiomDbNode` contains a pointer to another `AxiomDbNode` or a DataFusion `TableSource` that is a `CommittedPage`. `AxiomDbNode`s contain both the operation to execute, a way to store the appropriate cryptographic information, and the output of the operation in the node itself. Operations must be run in the order of `execute`, `keygen`, `prove`, and then `verify`.

## AxiomDbExec

Generates a flattened `AxiomDbNode` vector from a `LogicalPlan` tree root node and DataFusion `SessionContext`.

## AxiomDbExpr

Contains a way to convert DataFusion's `Expr` into an `AxiomDbExpr`, which is a subset of DataFusion's `Expr` since we do not currently support all `Expr`s.

## Running a test

The following test runs `execute`, `keygen`, `prove`, and `verify` on a `[PageScan, Filter]` execution strategy

```bash
cargo test --release --package axiomdb-interface --test basic_e2e -- test_basic_e2e --exact --show-output
```
