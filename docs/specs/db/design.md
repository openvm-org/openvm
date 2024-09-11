# STARK Architecture

We describe our design for how to build a verifiable data system using STARKs.

Our data system follows a relational database model where the underlying logical data is stored
in tables with fixed schemas which determine the column structure, while the number of rows
is unbounded.

## Terminology

- **Logical Table**: The table with typed columns as appears in a normal database. The table has a fixed schema and underlying data stored in a variable unbounded number of rows.
- **Logical Page**: A logical table where the number of rows does not exceed some maximum limit set by the system (e.g., around 1 million). The columns of the table are still typed. [This corresponds to `RecordBatch` type in Datafusion.]
- **Cryptographic Page**: <!--[jpw] not sure this is a good name--> The representation of a logical page as a trace matrix over a small prime field. A typed column in logical page may be transformed to multiple columns in the trace matrix (e.g., we default to storing 2 bytes = 16 bits per column).
  The cryptographic page comes with a cryptographic schema, which is deterministically derived from the logical schema. The cryptographic schema specifies the mapping from the logical columns to the cryptographic columns in the trace matrix as well as range guarantees (in bits) for each cryptographic column. At present we will derive cryptographic schema from logical schema without additional configuration parameters. The trace matrix is resized to have height a power of 2 by padding with **unallocated** rows, always at the bottom of the matrix. We do not yet enforce that different cryptographic pages all have the same global height, but may choose to do so in the future.
- **Cryptographic Table**: An ordered list of cryptographic pages that all have the same cryptographic schema. The list may be unbounded in length. The cryptographic table is used to represent a logical table as a sequence of trace matrices over a small prime field. _Note:_ a cryptographic table is never directly materialized within a single STARK. This is an important distinction.
- **Committed Page**: A cryptographic page together with the LDE of the trace matrix and the associated Merkle tree of the LDE. When we colloquially say _cached trace_, we are referring to caching of the entire committed page. Recall that the trace commitment is the Merkle root of the Merkle tree of the LDE matrix.
  - More generally, if we have a commitment format `page_commit(cryptographic_page) -> (PageProverData, PageCommitment)`, then a committed page is a cryptographic page together with the associated `(PageProverData, PageCommitment)`. The default `page_commit` we use is the Merkle tree of the LDE matrix. The **page commitment** (alias commitment of the page) is the `PageCommitment` of the cryptographic page.
- **Committed Table**: We specify a commitment format `df_commit(page_commits: Vec<PageCommitment>) -> (DfProverData, DfCommitment)` for a vector of page commitments. A committed table is a cryptographic table together with the `df_commit(page_commits)` of the commitments of the cryptographic pages in the cryptographic table. The `page_commits` vector can be of arbitrary unbounded length. The **table commitment** is the `DfCommitment` associated to the committed table.
  - We discuss practical options for `df_commit` below, but the two main candidates are flat hash and Merkle tree root, extended by zero padding, for the vector of page commitments.
- **Page Directory**: A vector of page commitments. We will use page directory to refer to the `Vec<PageCommitment>` associated to a committed table.

## Introduction

We can now describe the functionality of a verifiable database. Data is organized into logical
tables following a relational database model, and logical tables are stored in committed tables.
The committed tables are accessible to Provers.

Provers will have the ability to prove correct execution of a **query** on a logical table.
A query is a function mapping a collection of logical tables to an output logical table, where the function is specified in a special SQL-dialect we discuss below. The query may have **placeholder** values, which are indeterminate values that are replaced by user inputs when the query is executed.
For example:

```
SELECT * FROM table WHERE col1 = $x
```

Above, the query has one placeholder value, $x.

We will now describe a framework such that given a fixed query `Q` with placeholders, one can generate
a SNARK(STARK) circuit dependent on the query _but not the input tables or query input values_
such that successful verification of a proof of the circuit is equivalent to verification of the statement

- For committed tables `t_1, ..., t_n` and query input values `x_1, ..., x_r`, execution of the query `Q(t_1, ..., t_n; x_1, ..., x_r)` results in a committed table `t_out`.

The public values of the proof consist of the table commitments of `t_1, ..., t_n, t_out` and the hash of the input values `x_1, ..., x_r`.
To clarify, query input values refers to the values to replace placeholder values in the query.

## Architecture

In a traditional database, the query is parsed and then optimized based on the logical table schema and configuration parameters. The optimized output is a logical plan in the form of a tree where each node is a logical operation (e.g., `Filter`, `Projection`, `Join`, etc.). Importantly, these are operations on logical **tables**, not pages. The root of the tree corresponds to the last operation whose output is the output of the query. The output of a node becomes one of the inputs to its parent node.

We will use existing database planners to generate the logical plan for the query, making sure that the logical plan generation does **not** depend on the concrete values in the logical input tables. Hence the logical plan should only depend on the logical table schemas and the query itself.

For each logical operation `Op_l` (of which there is a finite list), we define a **table operation** `Op_c` with the same input/output format as the logical operation but where the logical tables are replaced by page directories. The statement of `Op_c` is of the form:

- Given committed tables `t_1, ..., t_n` and `query_input_values`, there is output committed table `t_out` such that the logical tables represented by `t_out` equals `Op_l` applied to the logical tables represented by `t_1, ..., t_n`, and the application of `Op_c` to `page_dir(t_1), ..., page_dir(t_n); query_input_values` results in `page_dir(t_out)`.

We claim, with proof by construction, that each logical operation (among a list of the common logical plan operations in a typical database) has a corresponding table operation.

More specifically, we define a table operation to mean a function in a **Database VM** where the input and output page directories are read/written as arrays of page commitments in the VM's memory. These arrays are treated as having _variable length_, so they are stored on the heap in memory. The `query_input_values` is passed as a struct allocated on the memory heap.
The Database VM will be a VM created using the [Modular VM framework](../vm/README.md) with
continuations enabled.

We construct the table operation for a logical operation as follows:

The Database VM will have special opcodes, which we call page-level **execution opcodes**, which operate on committed pages. This means the opcodes take the form

- Given input committed pages `p_1, ..., p_n`, there exists output commited pages `q_1, ..., q_m` such that the opcode execution on `page_commit(p_1), ..., page_commit(p_n)` results in `page_commit(q_1), ..., page_commit(q_m)` and the cryptographic pages underlying `q_i`s equals the output of execution of a cryptographic page operation (e.g., `Filter`) on the cryptographic pages underlying `p_i`s. Note that unlike table operations, execution opcodes can have multiple output pages.

An execution opcode's spec depends on the cryptographic page operation and the cryptographic schema of each input page. The spec will _not_ depend on the concrete values in the input pages. The spec should also not depend on the height of the page, unless otherwise specified.

We claim there is a finite number of classes of cryptographic page operations such that each table
operations can be written as a function using a finite set of execution opcodes from these classes.
Here by a class of cryptographic page operations we mean an infinite collection of page operations that
can all be described with the same spec (e.g., `Filter` on predicate) but with concrete instances
of the operation depending on the cryptographic schemas of the input pages as well as other parameters (e.g., the cryptographic column indices to filter on). <!--TODO[jpw] more precise definition-->

#### Example: Filter

The table operation for filter with a fixed predicate has query input value that defines the
predicate condition, and the table operation requires two execution opcodes: `Filter` and `Merge`.

The table operation first loops through the input page directory and calls `Filter` opcode
on each page commitment in the directory. This outputs a page directory of the same length where
the output pages can be under-allocated. Then the table operation calls the `Merge` opcode
multiple times to reduce the page directory to a possibly smaller length where output pages
are near full allocation.

### Database VM Proving

We describe how to prove the execution of a table operation in such a VM using a
STARK-aggregation framework.

#### Execution Opcode STARKs

For each execution opcode, we generate a multi-trace STARK with logup interactions that proves the execution of the cryptographic page operation on materialized cryptographic pages. When `page_commit` is via the LDE Merkle tree, the cryptographic pages are materialized as partitioned traces (alias for cached traces, but we will see below we may not necessarily cache them), and the page commitments
(but not the pages themselves!) are contained in the proof of the STARK. In general, the requirement
on the STARK is that the proof of the STARK contains the page commitments of all input and output committed pages -- these may be contained in the public values or the proof commitments.

Our notion of _class_ of cryptographic page operations corresponds directly to a class of STARKs that
we can implement in a uniform way at the software engineering level.

For each concrete instance of an execution opcode, the verifying key of the STARK becomes a unique
identifier for the execution opcode. The verifying key embeds the spec of the execution opcode,
including the cryptographic schemas of the input and output pages.

#### Table Operation via Aggregation

Our Database VM will contain all opcodes needed for STARK aggregation. In other words, _the
Database VM will be an instance of an Aggregation VM with continuations enabled_.

A call to an execution opcode in the Database VM will simply be a function call to the STARK
verification function, using Aggregation VM instructions, with respect to the execution opcode's
verifying key. In this sense, executions opcodes are not actually opcodes -- we keep referring to them as such to clarify the framework, but may rename this terminology later.

The verification function will have access to the execution proof, which contains
`input_page_commits, output_page_commits, query_input_values` as in-memory values.
There are two options for how the verification functions are implemented:

1. The verification function is fully dynamic, so the Database VM program code is independent of what execution opcodes are used and the entire framework is context free. The verification function will take the execution opcode's verifying key as a dynamic input and compute `hash(vkey)`. It then either checks `hash(vkey)` is in some static list or keeps a dynamic list of all vkeys used, which will need to be exposed as a public commitment.
2. The verification function treats each execution vkey as a compile-time constant. Explicitly this means a call to `execution_opcode` in the table operation is really a call to the `verify_stark(execution_opcode_vkey, _)` function ([implementation](https://github.com/axiom-crypto/afs-prototype/blob/264d6a5b59451253ece37a8ddc0f52d1eb378cb0/recursion/src/stark/mod.rs#L128)) where `execution_opcode_vkey` is known at compile time and hardcoded into the function code. This approach likely has better performance than the fully universal approach.

Since the implementation of a table operation should be able to specify the exact
execution opcodes it will call, it is more optimal to use Option 2.

We have described how a table operation is implemented as a function in the Database VM,
with execution opcode calls being themselves function calls to STARK verification functions.
Observe that to prove the execution of this function in the Database VM, we require as input
all STARK proofs of the execution opcode STARKs called by the function. We discuss how these
are obtained in a maximally parallel fashion below.

#### Cryptographic Table Operation Unrolling

Each execution opcode should have a backend implementation that can be run separately from the STARK proving. We call this the **cryptographic implementation**. This cryptographic implementation can
be separate from any trace generation and operates on cryptographic pages.
The only functional requirement on cryptographic implementation is that it generates the output cryptographic page from the input cryptographic pages according to the cryptographic page operation spec.

Each table operation needs to have an unrolled cryptographic implementation. This is a backend
implementation with async scheduling purely designed for backend performance. It will need to
generate its own execution plan tree and async call the cryptographic implementations of the necessary execution opcodes. (We call it unrolled because while the table operation in the Database VM operates on page directories, the cryptographic implementation operates on cryptographic pages within cryptographic tables.)

The table operation's cryptographic implementation must output a log of all execution opcodes called, together with the **input and output** cryptographic pages from their cryptographic implementations.

We will collect the STARK proofs of all execution opcodes needed in a table operation in a
fully offline fashion: given the table operation, we execute its unrolled cryptographic implementation on cryptographic tables offline, ahead of any proving.
The resulting logs will contain the input and output cryptographic pages of all execution opcode cryptographic implementations.

We will generate STARK proofs of these execution opcodes fully in parallel.

1. Each execution opcode operates on committed pages. The opcode's cryptographic implementation log supplies the cryptographic page associated to the committed page. As part of proof generation, we run the `page_commit` function on the input cryptographic pages to generate the input committed pages.

The above approach results in the least scheduling complexity and best parallel proving latency
as it removes execution opcode dependency considerations from table operation. An alternative approach, which gives better overall cost (measured in terms of total serial proving time) is:

2. The table operation either: (a) specifies a special proving scheduler or (b) automatically creates the scheduling DAG from the cryptographic implementation. Using either of these approaches, the scheduler creates a DAG for proving execution opcode STARKs, where the output committed page from opcode A is directly passed as the input committed page to opcode B. The scheduler then proves the STARKs in topological order.

We default to Option 1.

### Full Query Execution

We have described how to construct table operations in a Database VM together with
unrolled cryptographic implementations. Using this framework, the full
execution of a query can be expressed as a Database VM program which makes calls to
table operations. The query execution program is a serial traversal of the logical plan tree,
from leaves to root, where table operations are called as functions.

To summarize, at the end we will have a function for query execution with inputs consisting of
in-memory page directories and query input values, and output consisting of a page directory.
To complete query execution, the query execution program must compute `df_commit(input_page_dir[i])`
for all input page directories and `df_commit(output_page_dir)` and expose these table
commitments as public values. It must also compute `hash(query_input_values)` and expose it as a public value.

- An important detail is that the explicit calls to `df_commit` are only done on the inputs and outputs of the full query. The intermediate tables that arise within the query execution are
  all handled via the VM's memory architecture.

Since the Database VM has continuations, the query execution program can have variable unbounded
number of clock cycles. After writing the query execution program, the rest of aggregation and persistent memory between segments will be fully handled by the [continuations framework](../vm/continuations.md).

The overall proving flow can be viewed as having two main parts:

1. Proving of all execution opcode STARK proofs needed
2. Proving of the query execution program in the Database VM using continuations.

## Appendix

We record below another approach to Database VM proving where the verification of the
execution opcode STARKs is separated from the Database VM.

#### Aggregating execution into shared memory

Given a collection of STARK proofs for execution opcodes (call these execution proofs),
we will generate a STARK proof that verifies all execution proofs and writes
`hash(vkey), input_page_commits, output_page_commits, query_input_values` to a persistent memory
that will be **shared** with the Database VM.

<!--TODO[jpw] query_input_values needs to be a commitment for this interface to be uniform. Spec it out -->

This will be done by aggregating all execution proofs using a generalized
tree-based aggregation strategy using a separate Aggregation VM. The aggregation strategy is not
specific to the execution opcode context, and is instead [general](../aggregation.md). The general aggregation strategy, together with some memory merging logic in the internal tree nodes,
allows the aggregation to write all verified proof outputs into persistent Aggregation VM memory
in a dedicated address space.
**Memory merging logic needs to be fleshed out.**

The commitment to this persistent memory will be a public value
of the final aggregation proof. To achieve this general goal, there are two options:

- The aggregation strategy is fully universal, where each node uses an Aggregation VM program that can verify any STARK vkey. In other words the verification program is fully dynamic. In this case the final aggregation circuit's vkey is independent of what execution opcodes are used and the entire framework is context free.
- The aggregation strategy is execution specific. This means we start with a list of enabled execution vkeys, and each node uses an Aggregation VM program that has the verification programs for
  each supported vkey compiled in. The Aggregation VM program only dynamically matches the execution vkey with the correct verification program to use. This approach likely has better performance than the fully universal approach.

We will evaluate both approaches above and choose based on performance and versatility, but the
rest of the architecture will not depend on this choice.

The Aggregation VM and Database VM share the same memory architecture, so it is possible for the
Database VM to load the **persistent shared memory** from the aggregation proof.

**Conclusion:** assuming verification of execution aggregation proof, the Database VM will have
access to a table of verified `hash(vkey), input_page_commits, output_page_commit, query_input_values` in a special address space in shared memory.
Within the Database VM, calling an execution opcode is then simply a lookup into this table.
