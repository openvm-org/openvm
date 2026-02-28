# SHA-2 VM Extension

This crate contains circuits for the SHA-2 family of hash functions.
We support constraining the block compression functions for SHA-256 and SHA-512.
It is also possible to use this crate to constrain the SHA-384 algorithm, as described in the next section.

## SHA-2 Algorithms Summary

The SHA-256, SHA-512, and SHA-384 algorithms are similar in structure.
We will first describe the SHA-256 algorithm, and then describe the differences between the three algorithms.

See the [FIPS standard](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf) for reference. In particular, sections 6.2, 6.4, and 6.5.

In short the SHA-256 algorithm works as follows.
1. Pad the message to 512 bits and split it into 512-bit 'blocks'.
2. Initialize a hash state consisting of eight 32-bit words to a specific constant value.
3. For each block, 
    1. split the message into 16 32-bit words and produce 48 more words based on them. The 16 message words together with the 48 additional words are called the 'message schedule'.
    2. apply a scrambling function 64 times to the hash state to update it based on the message schedule. We call each update a 'round'.
    3. add the previous block's final hash state to the current hash state (modulo $2^{32}$).
4. The output is the final hash state

The differences with the SHA-512 algorithm are that:
- SHA-512 uses 64-bit words, 1024-bit blocks, performs 80 rounds, and produces a 512-bit output. 
- all the arithmetic is done modulo $2^{64}$.
- the initial hash state is different.

The SHA-384 algorithm is a truncation of the SHA-512 output to 384 bits, and the only difference is that the initial hash state is different.
In particular, SHA-384 shares its compression function with SHA-512, so users may write guest code that supports SHA-384 by using this crate's SHA-512 compression function.

## Design Overview

We support the `SHA256_UPDATE` and `SHA512_UPDATE` intrinsic instructions, each of which takes three operands, `dst`, `state`, and `input`.
- `input` is a pointer to exactly one block of message bytes (the size of a block varies for each SHA-2 variant.)
- `state` is a pointer to the current hasher state (8 words in big endian). Word size varies by SHA-2 variant.
- `dst` is a pointer where the updated state should be stored. `dst` may be equal to `state`.

The `SHA256_UPDATE` and `SHA512_UPDATE` instructions write the value of the updated hasher state after consuming the input to `dst`.
Note that these instructions do not pad the input.

### Chips

The SHA-2 extension consists of 2 chips: `Sha2MainChip` and `Sha2BlockHasherChip`.  

The main chip constraints reading `state` and `input` from memory and writing the new state into `dst`.
The main chip sends the operands to the the block hasher chip via interactions.
The trace of the main chip consists of one row per instruction.

The block hasher chip constraints the SHA-2 compression algorithm for one block at a time.
More specifically, the trace of the block hasher chip consists of groups of 17 consecutive rows (21 for SHA-512) which together constrain the 64 (resp. 80) rounds of the SHA-2 compression algorithm for one block of input
Each block receives the previous hasher state and the input bytes from the main chip and sends the updated state to the main chip via interactions.
Note that the block hasher chip consists of a SubAir, which constrains all the SHA-2 logic, while the block hasher chip itself only constrains its interactions with the main chip.


### Air Design

We reuse the same AIR code to produce circuits for SHA-256 and SHA-512.
To achieve this, we parameterize the AIR by constants (such as the word size, number of rounds, and block size) that are specific to each algorithm.

The block hasher AIR consists of $R+1$ rows for each instruction, and no more rows
(for SHA-256, $R = 16$ and for SHA-512 and SHA-384, $R = 20$).
The first $R$ rows of each block are called 'round rows', and each of them constrains four rounds of the hash algorithm.
Each row constrains updates to the working variables on each round, and also constrains the message schedule words based on previous rounds.
The final row of each block is called a 'digest row' and it produces a final hash for the block, computed as the sum of the working variables and the previous block's final hash.

### Storing working variables

One optimization is that we only keep track of the `a` and `e` working variables.
It turns out that if we have their values over four consecutive rounds, we can reconstruct all eight variables at the end of the four rounds.
This is because there is overlap between the values of the working variables in adjacent rounds. 
If the state is visualized as an array, `s_0 = [a, b, c, d, e, f, g, h]`, then the new state, `s_1`, after one round is produced by a right-shift and an addition.
More formally,
```
s_1 = (s_0 >> 1) + [T_1 + T_2, 0, 0, 0, T_1, 0, 0, 0]
    = [0, a, b, c, d, e, f, g] + [T_1 + T_2, 0, 0, 0, T_1, 0, 0, 0]
    = [T_1 + T_2, a, b, c, d + T_1, e, f, g]
```
where `T_1` and `T_2` are certain functions of the working variables and message data (see the FIPS spec).
So if `a_i` and `e_i` denote the values of `a` and `e` after the `i`th round, for `0 <= i < 4`, then the state `s_3` after the fourth round can be written as `s_3 = [a_3, a_2, a_1, a_0, e_3, e_2, e_1, e_0]`.

### Message schedule constraints

The algorithm for computing the message schedule involves message schedule words from 16 rounds ago.
Since we can only constrain two rows at a time, we cannot access data from more than four rounds ago for the first round in each row.
So, we maintain intermediate values that we call `intermed_4`, `intermed_8` and `intermed_12`, where `intermed_i = w_i + sig_0(w_{i+1})` where `w_i` is the value of `w` from `i` rounds ago and `sig_0` denotes the `sigma_0` function from the FIPS spec.
Since we can reliably constrain values from four rounds ago, we can build up `intermed_16` from these values, which is needed for computing the message schedule.


### Dummy values

Some constraints have degree three, and so we cannot restrict them to particular rows due to the limitation of the maximum constraint degree.
We must enforce them on all rows, and in order to ensure they hold on the remaining rows we must fill in some cells with appropriate dummy values.
We use this trick in several places in this chip.
