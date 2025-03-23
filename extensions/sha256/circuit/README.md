# SHA256 VM Extension

This crate contains the circuit for the SHA256 VM extension.

## SHA-256 Algorithm Summary

See the [FIPS standard](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf), in particular, section 6.2 for reference.

In short the SHA-256 algorithm works as follows.
1. Pad the message to 512 bits and split it into 512-bit 'blocks'.
2. Initialize a hash state consisting of eight 32-bit words.
3. For each block, 
    1. split the message into 16 32-bit words and produce 48 more 'message schedule' words based on them.
    2. apply 64 'rounds' to update the hash state based on the message schedule.
    3. add the previous block's final hash state to the current hash state (modulo `2^32`).
4. The output is the final hash state

## Design Overview

This chip produces an AIR that consists of 17 rows for each block (512 bits) in the message, and no more rows.
The first 16 rows of each block are called 'round rows', and each of them represents four rounds of the SHA-256 algorithm.
Each row constrains updates to the working variables on each round, and it also constrains the message schedule words based on previous rounds.
The final row is called a 'digest row' and it produces a final hash for the block, computed as the sum of the working variables and the previous block's final hash.

Note that this chip only supports messages of length less than `2^29` bytes.

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

### Note about `is_last_block`

The last block of every message should have the `is_last_block` flag set to `1`.
Note that `is_last_block` is not constrained to be true for the last block of every message, instead it *defines* what the last block of a message is.
For instance, if we produce an air with 10 blocks and only the last block has `is_last_block = 1` then the constraints will interpret it as a single message of length 10 blocks.
If, however, we set `is_last_block` to true for the 6th block, the trace will be interpreted as hashing two messages, each of length 5 blocks.

Note that we do constrain, however, that the very last block of the trace has `is_last_block = 1`.

### Dummy values

Some constraints have degree three, and so we cannot restrict them to particular rows due to the limitation of the maximum constraint degree.
We must enforce them on all rows, and in order to ensure they hold on the remaining rows we must fill in some cells with appropriate dummy values.
We use this trick in several places in this chip.

### Block index counter variables

There are two "block index" counter variables in each row of the air named `global_block_idx` and `local_block_idx`.
Both of these variables take on the same value on all 17 rows in a block. 

The `global_block_idx` is the index of the block in the entire trace.
The very first 17 rows in the trace will have `global_block_idx = 1` and the counter will increment by 1 between blocks.  
The padding rows will all have `global_block_idx = 0`.
The `global_block_idx` is used in interaction constraints to constrain the value of `hash` between blocks.

The  `local_block_idx` is the index of the block in the current message.
It starts at 0 for the first block of each message and increments by 1 for each block.
The `local_block_idx` is reset to 0 after each message.
The padding rows will all have `local_block_idx = 0`.
The `local_block_idx` is used to calculate the length of the message processed so far when the first padding row is encountered.

### VM air vs SubAir

The SHA-256 VM extension chip uses the `Sha256Air` SubAir to help constrain the SHA-256 hash.
The VM extension air constrains the correctness of the SHA message padding, while the SubAir adds all other constraints related to the hash algorithm.
The VM extension air also constrains memory reads and writes.

### A gotcha about padding rows

There are two senses of the word padding used in the context of this chip and this can be confusing.
First, we use padding to refer to the extra bits added to the message that is input to the SHA-256 algorithm in order to make the input's length a multiple of 512 bits.
So, we may use the term 'padding rows' to refer to round rows that correspond to the padded bits of a message (as in `Sha256VmAir::eval_padding_row`).
Second, the dummy rows that are added to the trace to make the trace height a power of 2 are also called padding rows (see the `is_padding_row` flag).
In the SubAir, padding row probably means dummy row.
In the VM air, it probably refers to SHA-256 padding.