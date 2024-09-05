# Database VM Interface Spec

This doc follows the [design spec](./design.md) in the design of the Database VM interface. Note that some of the terminology will likely change in the near future.

## Controller

The `DbvmController` serves to handle conversion from a `LogicalPlan` tree to `ExecutionOpNode` tree and coordinate backend and VM operations.

```rust
pub struct DbvmController<SC: StarkGenericConfig, E: StarkEngine<SC>, F: PrimeField32> {
    /// The root of the ExecutionOp tree
    pub root: ExecutionOpNode,
    /// Placeholder inputs of the query
    pub inputs: Vec<ScalarValue>,
    /// References to the leaves of the ExecutionOp tree
    pub entry_points: Vec<Arc<ExecutionOpNode>>,
    /// Execution log
    pub execution_log: Option<ExecutionLog<SC>>,
    /// STARK Config
    pub config: SC,
    /// STARK Engine
    pub engine: E,
    /// Reference to the Aggregation VM
    pub vm: Arc<VirtualMachine<F>>,
    /// The flattened program to run in the Database VM
    pub program: DbvmProgram,
}
```

## Program

The Database VM program contains a set of instructions that are generated from a LogicalPlan. The ExecutionLog is generated when the backend implementation is executed, containing the necessary Database VM opcodes and

```rust
/// A Database VM program generated from a LogicalPlan
pub struct DbvmProgram {
    pub instructions: Vec<ExecutionOpNode>
}
```

```rust
/// A single entry in the ExecutionLog. Input and output commitments point to the relevant `CommittedDataFrame`s in shared memory.
pub struct ExecutionLogEntry<SC: StarkGenericConfig> {
    pub op: ExecutionOpNode,
    pub input: PageCommitment<SC>,
    pub output: PageCommitment<SC>,
}
```

```rust
/// An ordered list of all operations executed by the physical implementation
pub struct ExecutionLog<SC: StarkGenericConfig> {
    pub entries: Vec<ExecutionLogEntry<SC>>,
}
```

## Logical Interface

The Logical interface is handled mostly by Apache DataFusion. Users are able to convert a SQL statement into a `LogicalPlan`, which can then be passed into the Database VM Controller.

### Aliases

We'll type alias some DataFusion types to more align with our higher-level model.

```rust
pub type LogicalSchema = Schema;
pub type LogicalPage = RecordBatch;
```

### Objects

```rust
/// Holds an unbounded number of rows with a fixed LogicalSchema
pub struct LogicalTable {
    pub schema: Arc<LogicalSchema>,
    pub columns: Vec<Arc<dyn Array>>,
    pub num_rows: usize,
}
```

## Physical Interface

The Physical layer represents data in its equivalent form of a trace matrix over a small prime field. Schema information is converted to match this format.

### Traits

```rust
/// Represents a physical operation in the query execution plan
#[async_trait]
pub trait ExecutionOp<SC: StarkGenericConfig> {
    /// Executes the physical operation and returns the full ExecutionLog of the operation
    async fn exec(&self) -> ExecutionLog;
    /// Generates a program to be run in the Database VM
    async fn generate_program(&self) -> ExecutionLog;
    /// Generates the vkey for the operation (overrides any currently-present vkey)
    async fn keygen(&self);
    /// Gets the input commitments
    fn input_commitments(&self) -> Option<Vec<PageCommitment<SC>>>;
    /// Gets the output commitments
    fn output_commitments(&self) -> Option<Vec<PageCommitment<SC>>>;
}
```

### Objects

```rust
/// A single database operation in the Execution tree (converted from a LogicalPlan)
pub enum ExecutionOpNode {
    TableScan(TableScanOp),
    Filter(FilterOp),
    Projection(ProjectionOp),
    //...
}
```

```rust
/// A physical Field within a PhysicalSchema. Contains the name, width (number of columns), and total number of range bits of the Field.
pub struct PhysicalField {
    pub name: String,
    pub width: usize,
    pub range_bits: usize,
}
```

```rust
/// Converted from a LogicalSchema. Contains an ordered list of PhysicalFields and the total width of the PhysicalPage.
pub struct PhysicalSchema {
    pub fields: Vec<PhysicalField>,
    pub total_width: usize,
}
```

```rust
pub struct PhysicalDataFrame {
    pub schema: PhysicalSchema,
    pub pages: Vec<PhysicalPage>,
}
```

## Cryptographic Interface

The cryptographic interface handles committing to `PhysicalPage`s and `PhysicalDataFrame`s.

### Aliases

```rust
pub type PageCommitment<SC: StarkGenericConfig> = Com<SC>;
```

### Traits

```rust
/// Is able to generate a commitment to a Page and associated Merkle tree of the LDE matrix
pub trait PageCommittable<SC: StarkGenericConfig> {
    fn page_commit(&self) -> ProverTraceData<SC>;
}
```

### Objects

```rust
/// Consists of a PhysicalPage and its asssociated commitment to the low-degree extension
pub struct CommittedPage<SC: StarkGenericConfig> {
    pub page: PhysicalPage,
    pub commitment: PageCommitment<SC>,
}
```

```rust
/// Consists of a PhysicalDataFrame and its associated commitments to the low-degree extension of each PhysicalPage, as well as the commitment to the DataFrame
pub struct CommittedDataFrame<SC: StarkGenericConfig> {
    pub df: PhysicalDataFrame,
    pub commitments: Vec<ProverTraceData<SC>>,
    pub df_commitment: PageCommitment<SC>,
}
```

## Example Execution

1. User generates a SQL query (table includes 4 columns: `col0`, `col1`, `col2`, `col3`)

```sql
SELECT * FROM table WHERE col1 = $x
```

2. DataFusion generates a `LogicalPlan` from the query
3. Pass `LogicalPlan` root node into the `AxdbController` with the input `$x = 10`
4. `DbvmController` generates a tree of `ExecutionOpNode` nodes (in this case there is no branching):
   - TableScan ["table"]
   - Filter [input = $x]
   - Predicate [`col0`, `col1`, `col2`, `col3`]
5. Run the backend/physical implementation on the generated `ExecutionOpNode` tree
   - saves the `ExecutionLog`, which includes the input/output `CommittedDataFrame`s in shared memory and the commitments themselves
6. `DbvmController` generates a `DbvmProgram` by flattening the `ExecutionOpNode` tree via depth-first search
7. `DbvmController` checks if there are already a vkey for the query
   - if not, it will generate keys for each node and the query itself based on the `PhysicalSchema` of the input(s)
8. `DbvmController` hands the `DbvmProgram` to the Aggregation VM which runs the Database VM program and generates an aggregate STARK proof of the computation
