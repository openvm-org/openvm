# Continuations

Our high-level continuations framework follows previous standard designs (Starkware, Risc0), but with a novel persistent memory argument.

The overall runtime execution of a program is broken into **segments** (the logic of when to segment can be custom and depend on many factors).
Each segment is proven in a separate STARK VM circuit as described in [STARK Architecture](./stark.md). The public values of the circuit must contain the pre- and post-state commitments to the segment. The state consists of the active program counter and the full state of memory. (Recall in our architecture that registers are part of memory, so register state is included in memory state).

While the runtime execution must be serial, we intend for the proofs of
each VM segment circuit to be maximally parallelizable. Therefore we do **not** allow any shared randomness between different segment circuits.

## Persistent Memory

### Motivation

Inside a VM segment circuit which is part of continuations, we have a
MemoryInterface chip which must verify, with respect to the pre-state commitment, the memory values for all
addresses accessed in the segment and write them into the MEMORY_BUS at timestamp 0. Similarly the chip must verify, with respect to the post-state commitment, the memory values for all addresses accessed in the segment and read them into the MEMORY_BUS at their final timestamps.

Thus the primary goal is to design an efficient commitment and verification format for the memory state. We designed our persistent memory commitment such that the cost of verification is almost-linear <!--TODO: make this precise--> in the number of accesses done within the segment and logarithmic in the total size of memory used across all segments. As far as we know, all known solutions that achieve this use Merkle trees in some form.

The basic design is to store memory as a key-value store using a binary Merkle trie. A verification of an access requires a merkle proof, which is logarithmic in the total size of the tree. Previous optimizations assume locality of memory accesses and use higher-arity Merkle tries to emulate page tables.

We present a design which does not assume any memory access patterns while still amortizing the Merkle proof cost across multiple accesses.

### Design

We have three chips — the ExpandChip, the MemoryInterfaceChip, and the CompressChip. For convenience, define constants `C` equal to the number of field elements in a hash value, `L` where the addresses in an address space are $0..2^L$, `M` and `AS_OFFSET` where the address spaces are `AS_OFFSET..AS_OFFSET + 2^M`, and `H = M + L - log2(C)`. `H` is the height of the Merkle tree in the sense that the leaves are at distance `H` from the root.

We define the following interactions:

<!--TODO: make a new diagram-->

On the <span style="color:green">EXPAND_BUS</span>, we have the interaction
<span style="color:green">**(expand_direction: {-1, 1}, height: F, labels: (F, F), hash: [F; C])**</span>

- **expand_direction** represents whether **hash** is the initial (1) or final (-1) hash value of the node represented by **node_label**.
- **height** indicates the height of the node represented in this interaction, i.e. H - its depth. height = 0 indicates that a node is a leaf.
- **labels = (as_label, address_label)** are labels of the node. The root has both equal to 0; if a node has labels `(x, y)`, then its left child has labels `(2x, 2y)` and its right child has labels either `(2x + 1, 2y)` or `(2x, 2y + 1)` depending on whether the address space or address is being expanded (this is determined by the height of the node). Defined this way, if a leaf has labels `(x, y)`, then the address space it corresponds to is `(y / 2^M) + AS_OFFSET` and the addresses it corresponds to are `C * x..C * (x + 1)`
- **hash** is the hash value of the node represented by the interaction.

On the <span style="color:DodgerBlue">MEMORY_INTERFACE_BUS</span>, we have the interaction
<span style="color:DodgerBlue">**(expand_direction: {-1, 1}, as: F, address: F, value: F)**</span>

- **expand_direction** is as above, and the other fields should be straightforward.

We send the above interactions when we know the value and receive them when we would like to know the values. Below, the frequency is 1 unless otherwise specified.

Each (IO part of a) row of ExpandChip will look like
**(height, parent_labels, parent_hash, left_node, left_hash, right_node, right_hash)**
and will have the following interactions:

- Send <span style="color:green">**(1, height + 1, parent_labels, parent_hash)**</span> on <span style="color:green">EXPAND_BUS</span>
- Receive <span style="color:green">**(1, height, left_child_labels, left_hash)**</span> on <span style="color:green">EXPAND_BUS</span>
- Receive <span style="color:green">**(1, height, right_child_labels, right_hash)**</span> on <span style="color:green">EXPAND_BUS</span>

The MemoryInterfaceChip will have rows of the following form:
`(as, node*label, hash_initial, hash_final, value_matters: [bool; C], expand_direction: [bool; C])`
and will have the following interactions:

- Send <span style="color:green">**(0, 0, (as - AS_OFFSET) \* 2^M, node\*label, hash_initial)**</span> on <span style="color:green">EXPAND_BUS</span>
- Receive <span style="color:DodgerBlue">**(expand_direction[i], as, node_label + i, hash_initial[i])**</span> with frequency <span style="color:DodgerBlue">value_matters[i]</span> for each `i = 0..C` on <span style="color:DodgerBlue">MEMORY_INTERFACE_BUS</span>
- Send <span style="color:DodgerBlue">**(1, as, node_label + i, hash_final[i])**</span> for each `i = 0..C` on <span style="color:DodgerBlue">MEMORY_INTERFACE_BUS</span>
- Receive <span style="color:green">**(1, 0, (as - AS_OFFSET) \* 2^M, node_label, hash_final)**</span> on <span style="color:green">EXPAND_BUS</span>

The CompressChip is then a sort of reflection of ExpandChip. Each row of CompressChip looks like
`(as, height, parent*node, parent_hash, left_direction_change, left_node, left_hash, right_direction_change, right_node, right_hash)`
and its interactions are:

- Send <span style="color:green">**(-1 + (2 \* left_direction_change), height, left_child_labels, left_hash)**</span> on <span style="color:green">EXPAND_BUS</span>
- Send <span style="color:green">**(-1 + (2 \* right_direction_change), height, right_child_labels, right_hash)**</span> on <span style="color:green">EXPAND_BUS</span>
- Receive <span style="color:green">**(-1, height + 1, parent_labels, parent_hash)**</span> on <span style="color:green">EXPAND_BUS</span>

In this way, nodes whose hashes are not actually relevant to memory and are only used for regenerating the Merkle root are sent directly from the ExpandChip to the CompressChip on <span style="color:green">EXPAND_BUS</span>.

Furthermore, the MemoryInterfaceChip interacts with itself on <span style="color:DodgerBlue">MEMORY_INTERACTION_BUS</span> for values that are neither read nor written by memory. The **value_matters** columns are used to not receive values that are immediately overwritten by memory.

[Internal doc](https://docs.google.com/document/d/1qzMqPTn3s2DjJkk51Pn2UKnNjGXxwyctda0OIXDnFbA/edit?usp=sharing) with details on interfaces with other chips.

## Aggregation

Given the execution segments of a program, we will prove each segment in a VM segment circuit in parallel. These proofs will then be aggregated in an [aggregation tree](../aggregation.md) by a segment aggregation program. This segment aggregation program will be run inside **a different VM** which **does not** have continuations turned on. The latter VM is called an **Aggregation VM**.

See [Aggregation](../aggregation.md) for more details.
