# STARK Architecture

We describe our design for how to build a verifiable data system using STARKs.

Our data system follows a relational database model where the underlying logical data is stored
in tables with fixed schemas which determine the column structure, while the number of rows
is unbounded.

## Terminology

- **Logical Table**: The table with typed columns as appears in a normal database. The table has a fixed schema and underlying data stored in a variable unbounded number of rows.
- **Logical Schema**: The ordered list of different types for each column in a logical table.
- **Logical Page**: A logical table where the number of rows does not exceed some maximum limit set by the system (e.g., around 1 million). The columns of the table are still typed. [This corresponds to `RecordBatch` type in Datafusion.] The number of rows is called the **logical height**.
- **Cryptographic Page**: The representation of a logical page as a trace matrix over a small prime field. A typed column in logical page may be transformed to multiple columns in the trace matrix (e.g., we default to storing 2 bytes = 16 bits per column). The trace matrix height is always a power of two, but we allow empty _unallocated_ rows, always at the bottom of the matrix. The cryptographic page includes auxiliary columns such as `is_alloc` column specifying which rows are allocated. The logical page must be recoverable from the cryptographic page. We do not yet enforce that different cryptographic pages all have the same global height, but may choose to do so in the future. The **cryptographic height** will refer to the height of the trace matrix, which is a power of two. The cryptographic page includes the cryptographic schema (see below).
- **Cryptographic Schema**: The cryptographic page comes with a cryptographic schema, which is deterministically derived from the logical schema. The cryptographic schema specifies the mapping from the logical columns to the cryptographic columns in the trace matrix as well as range guarantees (in bits) for each cryptographic column. At present we will derive cryptographic schema from logical schema without additional configuration parameters.
- **Cryptographic Table**: An ordered list of cryptographic pages that all have the same cryptographic schema. The list may be unbounded in length. The cryptographic table is used to represent a logical table as a sequence of trace matrices over a small prime field. _Note:_ a cryptographic table is never directly materialized within a single STARK. This is an important distinction. We do not enforce that any cryptographic page in a cryptographic table needs to be fully allocated.
- **Committed Page**: A cryptographic page together with the LDE of the trace matrix and the associated Merkle tree of the LDE. Recall that the trace commitment is the Merkle root of the Merkle tree of the LDE matrix.
  - Define `page_commit(cryptographic_page: CryptographicPage) -> (PageProverData, PageCommitment)`, where `PageProverData` is the type for the pair of the LDE matrix and its Merkle tree. The `PageCommitment` is the type for the Merkle root of the Merkle tree. A committed page refers to an instance of `(CryptographicPage, PageProverData, PageCommitment)`. Assuming hash collision resistance, the page commitment is a unique identifier for the committed page.
- **Page Directory**: A vector of page commitments: `Vec<PageCommitment>`.
- **Committed Table**: We define the commitment `directory_commit(page_directory: Vec<PageCommitment>) -> (TableProverData, TableCommitment)` where `TableProverData = ()` and `TableCommitment = HashDigest` to be the hash of all page commitments concatenated together. We will use poseidon2 as the hash. A committed table is a cryptographic table together with `directory_commit(page_directory)` where `page_directory` is the vector of page commitments of the cryptographic pages in the cryptographic table. The `page_directory` vector can be of arbitrary unbounded length. The **table commitment** is the `TableCommitment` associated to the committed table.
  <!-- TODO add page metadata alongside page commitment to support indexing -->
  - We leave open the possibility to switch `directory_commit` to be a Merkle tree commitment, where `TableProverData` would be the Merkle tree and `TableCommitment` is the Merkle root.

## Introduction

We can now describe the functionality of a verifiable database. Data is organized into logical
tables following a relational database model, and logical tables are stored in committed tables.
The committed tables are accessible to Provers.

Provers will have the ability to prove correct execution of a **query** on a logical table.
A query is a function mapping a collection of logical tables to an output logical table, where the function is specified in a special SQL-dialect we discuss below. The query may have **placeholder** values, which are indeterminate values that are replaced by user inputs when the query is executed. We call the specific instances of placeholder values the **query input values**.
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

## Architecture

In a traditional database, the query is parsed and then optimized based on the logical table schema and configuration parameters. The optimized output is a logical plan in the form of a tree where each node is a logical operation (e.g., `Filter`, `Projection`, `Join`, etc.). Importantly, these are operations on logical **tables**, not pages. The root of the tree corresponds to the last operation whose output is the output of the query. The output of a node becomes one of the inputs to its parent node.

We will use existing database planners to generate the logical plan for the query, making sure that the logical plan generation does **not** depend on the concrete values in the logical input tables. Hence the logical plan should only depend on the logical table schemas and the query itself.

For each logical operation (of which there is a finite list), we will define a corresponding **cryptographic table operation**.

Before we define table operations, we must first define cryptographic page operations.

### Cryptographic Page Operations

We will define a set of **cryptographic page operations**. Each page operation has two
separate components:

- Execution: a non-ZK implementation that takes input cryptographic pages `p_1, ..., p_n` and query inputs `x_1, ..., x_r` and produces output cryptographic pages `q_1, ..., q_m`. Refer to this as the Page Operation Execution.
- Verification: a multi-trace STARK with logup interactions that generates a `proof`, depending on `{p_i}, {q_i}, {x_i}`, verifying the execution of the page operation. The proof has public values `x_1, ..., x_r` and `{page_commit(p_i)}, {page_commit(q_i)}` are all separate trace commitments within the proof. Note that the page commitments of the input and output pages, but not the pages themselves, are accessible from the proof. Refer to this as the Page Operation Verification.
  - The Page Operation Verification is a STARK, which means it itself consists of the following components:
    - Keygen: proving and verifying key generation. The verifying key's dependence on IO is restricted to the cryptographic schemas of the input and output pages and the length `r`. How the query inputs are interpretted within the STARK is implementation specific.
    - Proving: given concrete input committed pages and query input values, generate the proof and output committed pages. The output committed pages may be optionally provided to the prover, in which case only the proof is generated. Refer to the proof as the Page Operation Proof.

Unless otherwise specified, when we say page operation we will mean cryptographic page operation in the future.

In formulas, let `op.execute(p_1,...,p_n,x_1,...,x_r) -> q_1,...,q_m` be the page operation execution. Then a proof of this page operation verification has the properties
that:

- There is an extractor `op.get_page_commits(proof) -> Vec<PageCommitment>`
- `verify_stark(proof, x_1,...,x_r) == true` if and only if there exists `p_1,...,p_n,q_1,...,q_m` such that `op.execute(p_1,...,p_n,x_1,...,x_r) == (q_1,...,q_m)` and `op.get_page_commits(proof) == (page_commit(p_1),...,page_commit(p_n),page_commit(q_1),...,page_commit(q_m)`.

Note that in order for verification of the `proof` to actually verify the page operation execution, we must assume the random oracle model so that the preimage of a page commitment is unique. In that case, the verifying key of the page operation verification STARK becomes a unique identifier for the page operation. We will assume the random oracle model below.

### Cryptographic Table Operations

A cryptographic table operation has two separate components:

- Execution: a non-ZK implementation that takes input cryptographic tables `t_1, ..., t_n` and query inputs `x_1, ..., x_r` and produces output cryptographic tables `t_out`. This is a backend implementation with async scheduling designed for backend performance, with the restriction that the execution should call page operation execution for any operations that need to be done at the page-level. Refer to this as the Table Operation Execution.
- Verification: We define the Table Operation Verification to be a function in a **Database IR** of the form `verify_table_op(input_page_commits, output_page_commits, input_values)` where `input_page_commits, output_page_commits` are page directories and `input_values` is a pointer to input values stored in memory.

We define the Database IR to be an intermediate representation with base instructions for memory access, control flow, and **page operation verifying keys**. The last is what distinguishes this IR from other standard ones.
Because the page operation verifying key embeds the page schemas within it, this IR has an infinite number of instructions. However any table operation verification function will only use a finite number of instructions.

#### Table Operation Verification via Aggregation

We will lower functions in the Database IR described above into program code
for an [Aggregation VM](../vm/stark.md) with continuations enabled as follows:

Given a verifying key `VKEY`, we lower it to the function in the Aggregation VM given by

```rust
fn verify_page_op(VKEY, input_page_commits, input_values) -> Vec<PageCommitments> {
    let proof = hint_proof(VKEY, input_page_commits, input_values);
    assert!(verify_stark(VKEY, proof, input_values));
    let page_commits = get_page_commits(proof);
    let output_page_commits = page_commits[input_page_commits.len()..];
    output_page_commits
}
```

We make the important assumption that the runtime of the Aggregation VM has oracle access to all necessary page operation proofs and is able to hint them into program memory. Note here the `input_page_commits, input_values` are enough to identify the proof, while the `output_page_commits` may be obtained from the proof afterwards.

We mention there are two different ways this lowering to the Aggregation VM can be done:

1. The verification function is fully dynamic, so `verify_stark = dyn_verify_stark` treats the `VKEY` as a variable input and outputs `hash(VKEY)`. It then either checks `hash(VKEY)` is in some static list or keeps a dynamic list of all vkeys used, which will need to be exposed as a public commitment.
2. The verification function `verify_stark(VKEY, _)` treats the vkey as a compile-time constant. This is what is currently [implemented](https://github.com/axiom-crypto/afs-prototype/blob/264d6a5b59451253ece37a8ddc0f52d1eb378cb0/recursion/src/stark/mod.rs#L128). This approach likely has better performance than the fully universal approach.

Since the specific verifying keys are already statically part of the IR code, it seems more optimal to use Option 2.

The verification function in the Aggregation VM must have the property that

- `verify_table_op(input_page_commits, output_page_commits, input_values) == true` if and only if there exists `input_tables, output_tables` such that `op.execute(input_tables, input_values) == output_tables` and `input_page_commits[i] = input_tables[i].map(page_commit)` and `output_page_commits[i] = output_tables[i].map(page_commit)`.

Assuming random oracle model, the `input_page_commits, output_page_commits` uniquely determine the `input_tables, output_tables` and `verify_table_op` verifies the table operation execution.

We claim, with proof by construction, that each logical operation (among a list of the common logical plan operations in a typical database) has a corresponding cryptographic table operation.

We have described how a table operation is implemented as a function in the Aggregation VM,
with page operation instructions being themselves function calls to STARK verification functions.
Observe that to prove the execution of this function in the Database VM, we require as input
all STARK proofs of the execution opcode STARKs called by the function. We discuss how these
are obtained in a maximally parallel fashion below.

#### Example: Filter

Suppose we have a cryptographic page height of 2 and we start with a 1-column table with logical pages `[[0,5],[2,6],[7]]`. Let `PAGE_FILTER` be the operation of filtering on a page for `a < $x`. Let `PAGE_COMPACT_UNIT` be the operation that takes 2 pages `(carry_over, to_compact)` as input, then output `(new_carry_over, full_page)`. This operation appends `to_compact` to `carry_over` and if the height is `>= 2`, splits the first 2 rows into `full_page`.

The table operation `FILTER` has execution defined by:

```rust
fn table_filter(input_table: CryptographicTable, x: Type) -> CryptographicTable {
    let mut filtered_pages = input_table.pages.par_iter().map(|page|
        PAGE_FILTER.execute(page, x)
    ).collect();
    let mut carry_over = filtered_pages[0];
    for page in filtered_pages.iter().skip(1) {
        let (new_carry_over, full_page) = PAGE_COMPACT_UNIT.execute(carry_over, page);
        carry_over = new_carry_over;
        if !full_page.is_empty() {
            output_pages.push(full_page);
        }
    }
    if !carry_over.is_empty() {
        output_pages.push(carry_over);
    }
    CryptographicTable { pages: output_pages }
}
```

where all `PAGE_FILTER.execute` can be parallelized.
What this does for our specific example when `x = 4` is:

- `PAGE_FILTER` each input page to get `[0],[2],[]`
- `PAGE_COMPACT_UNIT([0],[2]) = ([], [0,2])`
- `PAGE_COMPACT_UNIT([],[]) = ([], [])`

Final output table is `[[0,2]]`.

The `FILTER` table operation verification will look like the following IR code:

```rust
fn verify_table_filter(input_page_commits: Vec<PageCommits>, x: Type) -> Vec<PageCommits> {
    let mut filtered_page_commits = Vec::with_capacity(input_page_commits.len());
    for page_commit in input_page_commits {
        let filtered_page_commit = verify_page_op(PAGE_FILTER_VKEY, page_commit, x.flatten()); // x.flatten() flattens Type into vector of field elements
        filtered_page_commits.push(filtered_page_commit);
    }
    let mut carry_over_commit = filtered_page_commits[0];
    let mut output_page_commits = Vec::new();
    for filtered_page_commit in filtered_page_commits {
        let (new_carry_over_commit, full_page_commit) = verify_page_op(PAGE_COMPACT_UNIT_VKEY, carry_over_commit, filtered_page_commit);
        if full_page_commit != EMPTY_PAGE_COMMIT {
            output_page_commits.push(full_page_commit);
        }
    }
    if carry_over_commit != EMPTY_PAGE_COMMIT {
        output_page_commits.push(carry_over_commit);
    }
    output_page_commits
}
```

### Database VM Proving

We describe how to prove the execution of a table operation in an Aggregation VM using a
STARK-aggregation framework.

The table operation execution must output a log of all page operation executions, together with the **input and output** cryptographic pages of the page operations.

We will collect the STARK proofs of all execution opcodes needed in a table operation in a
fully offline fashion: given the table operation, we execute it ahead of any proving.
The resulting logs will contain the input and output cryptographic pages of all cryptographic page operations.

We will generate STARK proofs of these page operation verifications fully in parallel:

Each page operation verification operates on committed pages. The execution logs already supplies the cryptographic page associated to the committed page. As part of proof generation, we run the `page_commit` function on the cryptographic pages to generate the committed pages.

The above approach results in the least scheduling complexity and best parallel proving latency
as it removes page operation execution dependency considerations from the proof scheduling.

### Full Query Execution

We have defined both table operation execution and how to lower table operation verification to functions in an Aggregation VM. Using this framework, a full query can also be described in two components:

- Execution can be implemented in a parallelized manner with calls to table operation executions.
- Verification is the function in the Database IR obtained by composing the table operation verifications. This function can be lowered to a program in the Aggregation VM as described above.

To summarize, at the end we will have a function in the Aggregation VM for query verification with inputs consisting of
in-memory page directories and query input values, and output consisting of a page directory (at present, all queries only output a single table).
To complete query execution, the query execution program must compute `directory_commit(input_page_commits[i])`
for all input page directories and `directory_commit(output_page_commits)` and expose these table
commitments as public values. It must also compute `hash(query_input_values)` and expose it as a public value.

- Observe that the explicit calls to `directory_commit` are only done on the inputs and outputs of the full query. The intermediate functions operate directly on VM memory.

Since the Database VM has continuations, the query program can have variable unbounded
number of clock cycles. After writing the query execution program, the rest of aggregation and persistent memory between segments will be fully handled by the [continuations framework](../vm/continuations.md).

The overall proving flow can be viewed as having two main parts:

1. Proving of all page operation verification STARK proofs needed
2. Proving of the query verification program in the Aggregation VM using continuations.

#### Hinting

The query verification program is proven in the Aggregation VM using continuations.
This means that the program execution is broken up into segments, and the consistency
of memory between segments is handled by the continuations framework.
We emphasized above that in order for this Aggregation VM to support the Database IR,
it needs to be able to support the operation `hint_proof(VKEY, input_pages, input_values)`. We will implement this in conjunction with continuations:

- There will be a continuations scheduler who serially executes the runtime of the program to determine where to segment the runtime, and to snapshot the memory and program states at the segment boundaries. This scheduler prepares the necessary input data to generate a VM circuit proof for each segment.
  - The scheduler can either run the entire runtime once offline, and then define the segments and initiate each segment proof, **or** it can progressively initialize proofs as each segment becomes ready.
- We **require** that the continuations scheduler has database access to all page operation proofs necessary for the full query verification. During the runtime of the program, the scheduler can directly fetch the required proofs to hint from the database. The scheduler includes these proofs as part of the overall input data to that segment. The prover of the segment circuit will only require as input the proofs needed in that segment and not require database access to all page operation proofs.
  - There are different mechanisms to achieve this, which we will discuss elsewhere.
