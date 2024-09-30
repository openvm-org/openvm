# Continuations

Our high-level continuations framework follows previous standard designs (Starkware, Risc0), but uses a novel persistent
memory argument.

The overall runtime execution of a program is broken into **segments** (the logic of when to segment can be custom and
depend on many factors). Each segment is proven in a separate STARK VM circuit as described
in [STARK Architecture](./stark.md). The public values of the circuit must contain the pre- and post-state commitments
to the segment. The state consists of the active program counter and the full state of memory. (Recall in our
architecture that registers are part of memory, so register state is included in memory state).

While the runtime execution must be serial, we intend for the proofs of each VM segment circuit to be maximally
parallelizable. Therefore, we do **not** allow any shared randomness between different segment circuits.

## Persistent Memory

### Motivation

Inside a VM segment, we have a `PersistentBoundaryChip` chip, which verifies, with respect to the pre-state commitment,
the memory values for all addresses accessed in the segment and writes them into the `MEMORY_BUS` at timestamp 0.
Similarly, the chip verifies, with respect to the post-state commitment, the memory values for all addresses accessed
in the segment and reads them into the `MEMORY_BUS` at their final timestamps.

Thus the primary goal is an efficient commitment and verification format for the memory state. We designed our
persistent memory commitment such that the cost of verification is almost-linear <!--TODO: make this precise--> in the
number of accesses done within the segment and logarithmic in the total size of memory used across all segments. As far
as we know, all known solutions that achieve this use Merkle trees in some form.

The basic design is to represent memory as a key-value store using a binary Merkle trie. The verification of an access
requires a Merkle proof, which takes time logarithmic in the total size of the tree. Previous optimizations assume
locality of memory accesses and use higher-arity Merkle tries to emulate page tables.

We present a design which does not assume any memory access patterns while still amortizing the Merkle proof cost across
multiple accesses.

### Design

Persistent memory requires three chips: the `PersistentBoundaryChip`, the `MemoryMerkleChip`, and a chip to assist in
hashing, which is by default the `Poseidon2Chip`. For simplify the discussion, define constants `C` equal to the number
of field elements in a hash value, `L` where the addresses in an address space are $0..2^L$, `M` and `AS_OFFSET` where
the address spaces are `AS_OFFSET..AS_OFFSET + 2^M`, and `H = M + L - log2(C)`. `H` is the height of the Merkle tree in
the sense that the leaves are at distance `H` from the root. We define the following interactions:

<!--TODO: make a new diagram-->

On the <span style="color:green">MERKLE_BUS</span>, we have interactions of the form
<span style="color:green">**(expand_direction: {-1, 0, 1}, height: F, labels: (F, F), hash: [F; C])**</span>, where

- **expand_direction** represents whether **hash** is the initial (1) or final (-1) hash value of the node represented
  by **node_label**. If zero, the row is a dummy row.
- **height** indicates the height of the node represented in this interaction, i.e. `H` - its depth. `H = 0` indicates
  that a node is a leaf.
- **labels = (as_label, address_label)** are labels of the node. The root has both equal to 0; if a node has labels
  `(x, y)`, then its left child has labels `(2x, 2y)` and its right child has labels either `(2x + 1, 2y)` or
  `(2x, 2y + 1)` depending on whether the address space or address is being expanded (this is determined by the height
  of the node). Defined this way, if a leaf has labels `(x, y)`, then the address space it corresponds to is
  `(y / 2^M) + AS_OFFSET` and the addresses it corresponds to are `C * x..C * (x + 1)`
- **hash** is the hash value of the node represented by the interaction.

Rows that correspond to initial/final memory states are sent to the `MEMORY_BUS` with the corresponding timestamps and
data, as per the `MEMORY_BUS` interface.

We send the above interactions when we know the value and receive them when we would like to know the values. Below, the
frequency is 1 unless otherwise specified.

Each (IO part of a) row in the `MemoryMerkleChip` trace contains the fields
**(height, parent_labels, parent_hash, left_hash, right_hash)**
and has the following interactions:

- Send <span style="color:green">**(expand_direction, height + 1, parent_labels, parent_hash)**</span>
  on <span style="color:green">MERKLE_BUS</span> with multiplicity `expand_direction`
- Receive <span style="color:green">**(expand_direction, height, left_child_labels, left_hash)**</span>
  on <span style="color:green">MERKLE_BUS</span> with multiplicity `expand_direction`
- Receive <span style="color:green">**(expand_direction, height, right_child_labels, right_hash)**</span>
  on <span style="color:green">MERKLE_BUS</span> with multiplicity `expand_direction`

Each row corresponding to a compression in the `MemoryMerkleChip` trace
contains the following information:
`(height, parent_label, parent_hash, left_direction_change, left_node, left_hash, right_direction_change, right_node, right_hash)`
and its interactions are:

The `PersistentBoundaryChip` has rows of the form
`(expand_direction, address_space, leaf_label, values, timestamp)`
and has the following interactions on the <span style="color:green">MERKLE_BUS</span>:

- Send <span style="color:green">**(1, 0, (as - AS_OFFSET) \* 2^M, node\*label, hash_initial)**</span>
- Receive <span style="color:green">**(-1, 0, (as - AS_OFFSET) \* 2^M, node_label, hash_final)**</span>

## Aggregation

Given the execution segments of a program, we will prove each segment in a VM segment circuit in parallel. These proofs will then be aggregated in an [aggregation tree](../aggregation.md) by a segment aggregation program. This segment aggregation program will be run inside **a different VM** which **does not** have continuations turned on. The latter VM is called an **Aggregation VM**.

See [Aggregation](../aggregation.md) for more details.
