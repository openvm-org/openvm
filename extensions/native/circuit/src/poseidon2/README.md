# Specification

This document explains the specification of the `VERIFY_BATCH` opcode and its implementation by `NativePoseidon2Chip`. `CHUNK` is 8.

## Context

We use a `VERIFY_BATCH` instruction in the following context:

There exist several trace matrices, with possibly varying heights that are all powers of 2 at most some maximum height `h_max`.
During proving, we hash these matrices to obtain a commitment `commit`.
This is done as follows:
- We first concatenate matrices of the same height, so that we then have matrices of distinct heights that are powers of 2.
- We then apply a rolling Poseidon2 hash to each row of each matrix, so that each matrix now has rows of length `CHUNK`.
- After this, we merkleize the matrices as follows.
  - We repeatedly take the largest matrix `M1` of height `h`, and compress it by hashing each row with its sibling, i.e. we apply Poseidon2 compression to the first two rows, and to the next two rows, and so on, so that its height is now `h/2`.
  - If there previously existed a matrix `M2` of height `h/2`, we combine them by applying Poseidon2 compression to each row of `M1` with the corresponding row of `M2`.
- At the end, we have a matrix of height 1 and width `CHUNK`; this is the commitment `commit`.

During verification, we would like to verify the hashing done one paths from the root of the Merkle tree to a particular leaf. We are given a specific `index` less than `h_max`, and are given access to the relevant rows of the original matrices -- these are "opened values".
We are also given the siblings encountered in the path to the root with whom we apply Poseidon2 compression. Our goal is to verify that the root computed in this way is equal to `commit`.

## Precise Specification

More precisely, we are given the following inputs. Assume that the original matrices `M_1, ..., M_n` are specified in decreasing order of height.

| Name         | Abstract type                   | Type in eDSL                                                        | Meaning                                                                                         |
|--------------|---------------------------------|---------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| `dimensions` | `Vec<F>`                        | `Array<C, Usize<C::F>>`                                             | `dimensions[i]` is the height of `M_i`                                                          |
| `opened_values` | `Vec<Vec<F>>` or `Vec<Vec<EF>>` | `Array<C, Array<C, Felt<C::F>>>` or `Array<C, Array<C, Ext<C::F>>>` | `opened_values[i]` is row `floor(index * dimensions[i]/h_max)` of `M_i`                         |
| `siblings` | `Vec<[F; CHUNK]`                | `Array<C, Array<C, Felt<C::F>>>` | The siblings with which we must apply Poseidon2 compression, in the order in which this is done |
| `index_bits` | `Vec<bit>`                      | `Array<C, Var<C::N>>` | The bits of `index` described above |
| `commit` | `[F; CHUNK]`                    | `Array<C, Felt<C::F>>` | The commitment we are verifying |

A `VERIFY_BATCH` instruction does the following:
- It maintains a node `node : [F; CHUNK]` in the Merkle tree.
- For each height `h` from `h_max` down to 1, we maintain a `proof_index`, i.e. when `h = h_max`, `proof_index = 0`, and when `h = 1`, `proof_index = lg(h_max)`.
  - If there are matrices `M_i` of height `h` (which is determined via `dimensions`), assume that those matrices are exactly `M_l, ..., M_r`.
  - It concatenates `opened_values[l], ..., opened_values[r]` into a single sequence `s`, then applies a rolling Poseidon2 hash to `s` to obtain `concat_hash : [F; CHUNK]`.
  - If this is the beginning of the algorithm, then we set `node <- concat_hash`. Otherwise, we compress as `node <- p2_compress(node, concat_hash)`.
  - Then, if `h` is not 1, we incorporate the sibling by doing `node <- p2_compress(node, siblings[proof_index])` if `index_bits[proof_index] = 0`, and `node <- p2_compress(siblings[proof_index], node)` if `index_bits[proof_index] = 1`.
- At the end, it checks that `node = commit`.

## Instruction Format

The instruction uses the following seven operands. As seven is the current maximum number of operands, the address space for all reads is fixed to be `AS::Native = 4`.

| Operand | Name | Meaning                                                                                                                                                                                                                          |
|---------|------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `a` | `dim` | Pointer to the start pointer of the `dimensions` array                                                                                                                                                                           |
| `b` | `opened_values` | Pointer to the start pointer of the `opened_values` array                                                                                                                                                                        |
| `c` | `opened_values_len` | Pointer to the length of the `opened_values` array                                                                                                                                                                               |
| `d` | `siblings` | Pointer to the start pointer of the `siblings` array                                                                                                                                                                             |
| `e` | `index_bits` | Pointer to the start pointer of the `index_bits` array                                                                                                                                                                           |
| `f` | `commit` | Pointer to the start pointer of the `commit` array                                                                                                                                                                               |
| `g` | `opened_value_size_inv` | The inverse of the size of an opened value. If the elements of the original matrices, and therefore the opened values, are field elements, then this should be 1. If they are extension field elements, then this should be 1/4. |

It is assumed that the `opened_values` array consists of elements of size 2 field elements, with the first field element being the pointer to the start of the row, and the second field element being the length of the row.

# Implementation

## Structure of Rows

The above verification process consists of two essential types of operations; as a result we handle these using two different types of rows:
- `InsideRow`: These rows are responsible for concatenating the opened values and computing the rolling hash.
- `TopLevel`: These rows are responsible for computing Poseidon2 compression. There are two subtypes:
  - `IncorporateSibling`: These rows are responsible for compressing a sibling with the current node. As part of this, they read both `sibling` and `index_bits` in order to determine whether to compute `p2_compress(sibling, node)` or `p2_compress(node, sibling)`.
  - `IncorporateRow`: These rows are responsible for compressing the rolling hash of a (concatenated) row of opened values with the current node.

As the `NativePoseidon2Chip` is also responsible for handling basic `PERM_POS2` and `COMP_POS2` instructions, these incur a third type of row:
- `SimplePoseidon`

As the three different types of rows require cells representing different things, in order to avoid having many unused cells in each row, we use the following strategy for columns.
The `NativePoseidon2Cols` struct contains cells that are used by both `InsideRow` and `TopLevel` rows (in addition to a couple cells that cannot safely be used in different ways by both), some of which are also used by `SimplePoseidon` rows.
This naturally includes the auxiliary columns used by the actual Poseidon2 hash Subair.
In addition, `NativePoseidon2Cols` contains a `specific` field whose type is simply an array of `F`.
`specific` is used in different ways by different types of rows; we accomplish this by having a different struct for each type of row:
- `TopLevelSpecificCols`
- `InsideRowSpecificCols`
- `SimplePoseidonSpecificCols`

`specific` then has length equal to the maximum width of the above three, so that it can be cast to each one.


## Layout of Rows in the Matrix

The execution of a single `VERIFY_BATCH` instruction consists of a contiguous subsegment of `TopLevel` rows in the trace matrix, with each row corresponding to a single Poseidon2 compression, except for the first one, which is necessarily an `IncorporateRow` row and simply sets the initial value of `node`.
The last of these `TopLevel` rows performs the execution interaction as well as the memory accesses to dereference operands `a` through `g`.

Each `IncorporateRow` row, in order to compute the rolling hash of a concatenation of opened values, itself incurs a contiguous subsegment of `InsideRow` rows in the trace matrix.
The last of these `InsideRow` rows interacts with the `IncorporateRow` to receive the necessary inputs and return the resulting hash; this interaction currently takes place on bus 7.

As the communication between `TopLevel` and `InsideRow` rows is done via interaction, there is no relationship between their actual positions in the trace matrix.

Each `PERM_POS2` and `COMP_POS2` instruction corresponds to a single `SimplePoseidon` row in the trace matrix.