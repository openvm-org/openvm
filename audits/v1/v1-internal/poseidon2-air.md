# Poseidon2 AIR

Author: https://github.com/MonkeyKing-1

## 1. Introduction

Scope: Plonky3 poseidon2 and poseidon2 air

Understanding how the Plonky3 poseidon2 air works, make sure it is sound.

## 2. Findings

None

## 3. Discussion

We discuss the constraints in the air. This is all the eval stuff.

We essentially start with a `state` and morph it into something else through a series of operations.
The state is an array of expressions that evaluate to the supposed state. `state` should contain degree 1 expressions when entering a sbox. A lot of columns are unneeded because we can be freely expressive with linear expressions (which makes up a lot of poseidon2).

### External Linear Layer

We use the `MdsMat4` permutation, which sets $M$ to be this:

$$
\begin{bmatrix}
2 & 3 & 1 & 1 \\
1 & 2 & 3 & 1 \\
1 & 1 & 2 & 3 \\
3 & 1 & 1 & 2
\end{bmatrix}
$$

$M$ is a matrix that we may multiply to every 4 elements later on; this is considered a "permutation." This only happens if the length of the state is a multiple of 4.

This is as opposed to the matrix used in the Horizon Labs implementation, which looks a bit different. The main difference is that this matrix uses less operations to get the result.

The external linear layer takes the current state and does some case work based on the length:

- Length 2: compute sum of elements of state and add to each element of state.
- Length 3: compute sum of elements of state and add to each element of state.
- Length is multiple of 4: Multiply every four elements by M, giving a new state. Compute the sum of elements with indices that are 0 mod 4, 1 mod 4, etc, computing four sums. Then add these sums to the elements that contributed to them. In other words, perform this multiplication:
  `[[2M M  ... M], [M  2M ... M], ..., [M  M ... 2M]]`.

### Internal Linear Layer

We can track the implementation to this:
impl InternalLayerBaseParameters<BabyBearParameters, 16> for BabyBearInternalLayerParameters
in baby-bear/src/poseidon2.rs. In particular, it is the function `generic_internal_linear_layer`. The matrix we want to multiply by is $1+diag(V)$ where $V=[-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/2^{27}, -1/2^8, -1/16, -1/2^{27}]$, and $1$ represents the matrix with all 1's. To do this, we use addition (like state[1] = sum + state[1]) for the first 3 entries since the things we multiply by are integers. To compute the first value, we use the the partial sum from state[1] and on to make sure we only need to subtract state[0] once.

### Full Round

Add round constants and then take it to some fixed power (7 in our case?). This is called sbox, and a more detailed description is written later. This results in a new state. We then run the external linear layer function on this (detailed above). This again results in a new state, with the same degree as the previous new state.

The FullRound struct contains what the end result should look like (full_round.post), and we constrain that this new state is the same as the expected result. We then make the state an expression of full_round.post, to make sure the degree is 1 again.

### Partial Round

Add round constant to state[0] then take it to some fixed power (7 in our case?). This is called sbox, and a more detailed description is written later. NOTE THAT ONLY state[0] is modified here. We constrain it to be equal to the expected result stored in partial_round.post_sbox. We then make state[0] be partial_round.post_sbox as an expression to make the degree 1 again. Then, we run the internal linear layer function to get a new state.

### SBOX

We want to calculate x^degree while using a fixed number of registers (which are extra columns in the trace). Given the number of registers, the constraint degree is implied. sbox structs contain the intermediate results.

- (3, 0) => constraint degree 3,
- (5, 0) => constraint degree 5,
- (7, 0) => constraint degree 7,
- (5, 1) => constraint degree 3,
- (7, 1) => constraint degree 3,
- (11, 2) => constraint degree 3,

### Entire Algorithm

1. Do `external_linear_layer` multiplication.
2. Do first half of the full rounds
3. Do the partial rounds
4. Do second half of the full rounds
