# STARK Architecture

We describe our design for how to build a verifiable data system using STARKs.

Our data system follows a relational database model where the underlying logical data is stored
in tables with fixed schemas which determine the column structure, while the number of rows
is unbounded.

## Terminology

- **Logical Table**: The table with typed columns as appears in a normal database. The table has a fixed schema and underlying data stored in a variable unbounded number of rows.
- **Logical Page**: A logical table where the number of rows does not exceed some maximum limit set by the system (e.g., around 1 million). The columns of the table are still typed. [This corresponds to `RecordBatch` type in Datafusion.]
- **Physical Page**: <!--[jpw] not sure this is a good name--> The representation of a logical page as a trace matrix over a small prime field. A typed column in logical page may be transformed to multiple columns in the trace matrix (e.g., we default to storing 2 bytes = 16 bits per column).
  The physical page comes with a physical schema, which is deterministically derived from the logical schema. The physical schema specifies the mapping from the logical columns to the physical columns in the trace matrix as well as range guarantees (in bits) for each physical column. At present we will derive physical schema from logical schema without additional configuration parameters. The trace matrix is resized to have height a power of 2 by padding with **unallocated** rows, always at the bottom of the matrix. We do not yet enforce that different physical pages all have the same global height, but may choose to do so in the future.
- **Physical Dataframe**: An ordered list of physical pages that all have the same physical schema. The list may be unbounded in length. The physical dataframe is used to represent a logical table as a sequence of trace matrices over a small prime field. _Note:_ a physical dataframe is never directly materialized within a single STARK. This is an important distinction.
- **Committed Page**: A physical page together with the LDE of the trace matrix and the associated Merkle tree of the LDE. When we colloquially say _cached trace_, we are referring to caching of the entire committed page. Recall that the trace commitment is the Merkle root of the Merkle tree of the LDE matrix.
  - More generally, if we have a commitment format `page_commit(physical_page) -> (PageProverData, PageCommitment)`, then a committed page is a physical page together with the associated `(PageProverData, PageCommitment)`. The default `page_commit` we use is the Merkle tree of the LDE matrix. The **page commitment** (alias commitment of the page) is the `PageCommitment` of the physical page.
- **Committed Dataframe**: We specify a commitment format `df_commit(page_commits: Vec<PageCommitment>) -> (DfProverData, DfCommitment)` for a vector of page commitments. A committed dataframe is a physical dataframe together with the `df_commit(page_commits)` of the commitments of the physical pages in the physical dataframe. The `page_commits` vector can be of arbitrary unbounded length. The **dataframe commitment** is the `DfCommitment` associated to the committed dataframe.
  - We discuss practical options for `df_commit` below, but the two main candidates are flat hash and Merkle tree root, extended by zero padding, for the vector of page commitments.
- **Page Directory**: A vector of page commitments. We will use page directory to refer to the `Vec<PageCommitment>` associated to a committed dataframe.

## Introduction

We can now describe the functionality of a verifiable database. Data is organized into logical
tables following a relational database model, and logical tables are stored in committed dataframes.
The committed dataframes are accessible to Provers.

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

- For committed dataframes `t_1, ..., t_n` and input values `x_1, ..., x_m`, execution of the query `Q(t_1, ..., t_n; x_1, ..., x_m)` results in a committed dataframe `t_out`.

The public values of the proof consist of the dataframe commitments of `t_1, ..., t_n, t_out` and the input values `x_1, ..., x_m` (or we may choose to use a commitment to the input values).
To clarify, query input values refers to the values to replace placeholder values in the query.

## Architecture

In a traditional database, the query is parsed and then optimized based on the logical table schema and configuration parameters. The optimized output is a logical plan in the form of a tree where each node is a logical operation (e.g., `Filter`, `Projection`, `Join`, etc.). Importantly, these are operations on logical **tables**, not pages. The root of the tree corresponds to the last operation whose output is the output of the query. The output of a node becomes one of the inputs to its parent node.

We will use existing database planners to generate the logical plan for the query, making sure that the logical plan generation does **not** depend on the concrete values in the logical input tables. Hence the logical plan should only depend on the logical table schemas and the query itself.

For each logical operation `Op_l` (of which there is a finite list), we define a **dataframe operation** `Op_c` with the same input/output format as the logical operation but where the logical tables are replaced by page directories. The statement of `Op_c` is of the form:

- Given committed dataframes `t_1, ..., t_n`, there is output committed dataframe `t_out` such that the logical table represented by `t_out` equals `Op_l` applied to the logical tables represented by `t_1, ..., t_n`, and the application of `Op_c` to `page_dir(t_1), ..., page_dir(t_n)` results in `page_dir(t_out)`.

We claim, with proof by construction, that each logical operation (among a list of the common logical plan operations in a typical database) has a corresponding dataframe operation.

More specifically, we define a dataframe operation to mean a function in a **Database VM** where the input and output page directories are read/written as arrays of page commitments in the VM's memory. These arrays are treated as having _variable length_, so they are stored on the heap in memory. We construct the dataframe operation for a logical operation as follows:

The Database VM will have special opcodes, which we call page-level **execution opcodes**, which operate on
committed pages. This means the opcodes take the form

- Given committed pages `p_1, ..., p_m`, there exists `p_out` such that the opcode execution on `page_commit(p_1), ..., page_commit(p_m)` results in `page_commit(p_out)` and the physical page underlying `p_out` equals the output of execution of a physical page operation (e.g., `Filter`) on the physical pages underlying `p_1, ..., p_m`.

An execution opcode's spec depends on the physical page operation and the physical schema of each input page. The spec will _not_ depend on the concrete values in the input pages. The spec should also not depend on the height of the page, unless otherwise specified.

We claim there is a finite number of classes of physical page operations such that each dataframe
operations can be written as a function using a finite set of execution opcodes from these classes.
Here by a class of physical page operations we mean an infinite collection of page operations that
can all be described with the same spec (e.g., `Filter` on predicate) but with concrete instances
of the operation depending on the physical schemas of the input pages as well as other parameters (e.g., the physical column indices to filter on). <!--TODO[jpw] more precise definition-->

### Database VM Proving

We describe how to prove the execution of a dataframe operation in such a VM using a
STARK-aggregation framework.

For each execution opcode, we generate a multi-trace STARK with logup interactions that proves the execution of the physical page operation on materialized physical pages. When `page_commit` is via the LDE Merkle tree, the physical pages are materialized as cached traces, and the page commitments
(but not the pages themselves!) are contained in the proof of the STARK. In general, the requirement
on the STARK is that the proof of the STARK contains the page commitments of all input and output committed pages -- these may be contained in the public values or the proof commitments.

Our notion of _class_ of physical page operations corresponds directly to a class of STARKs that
we can implement in a uniform way at the software engineering level.

For each concrete instance of an execution opcode, the verifying key of the STARK becomes a unique
identifier for the execution opcode. The verifying key embeds the spec of the execution opcode,
including the physical schemas of the input and output pages.

#### Aggregating execution into shared memory

Given a collection of STARK proofs for execution opcodes (call these execution proofs),
we will generate a STARK proof that verifies all execution proofs and writes
`hash(vkey), input_page_commits, output_page_commit, query_input_values` to a persistent memory
that will be **shared** with the Database VM.

<!--TODO[jpw] query_input_values needs to be a commitment for this interface to be uniform. Spec it out -->

This will be done by aggregating all execution proofs using a generalized
tree-based aggregation strategy using a separate Aggregation VM. The aggregation strategy is not
specific to the execution opcode context, and is instead [general](../aggregation.md). The general aggregation strategy, together with some memory merging logic in the internal tree nodes,
allows the aggregation to write all verified proof outputs into persistent Aggregation VM memory
in a dedicated address space. The commitment to this persistent memory will be a public value
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

#### Database VM runtime
