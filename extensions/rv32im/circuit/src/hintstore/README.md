# RV32 Hint Store Chip

The chip is an instruction executor for the HINT_STORE_RV32 and HINT_BUFFER_RV32 instructions.

Trace rows are exactly one of 3 types:
- `is_single = 1, is_buffer = 0`: to handle HINT_STORE_RV32
- `is_single = 0, is_buffer = 1`: rows for HINT_BUFFER_RV32
- `is_single = 0, is_buffer = 0`: dummy padding rows

A single HINT_BUFFER_RV32 instruction may use multiple contiguous rows. The first row,
which is also the row that will send messages to the program and execution buses with non-zero
multiplicities, is marked with `is_buffer_start = 1` (and it is the only row within the rows for that
instruction with `is_buffer_start = 1`).

On the starting row, a memory address `mem_ptr` is read from memory in the form of `4` limbs `mem_ptr_limbs`. The highest limb is range checked to be `pointer_max_bits - 24` bits, which ensures that calculating `mem_ptr` by composing the limbs `mem_ptr_limbs` will not overflow the field.

On each row in the same HINT_BUFFER_RV32 instruction, the chip does a write to `[mem_ptr:4]_2` and increments `mem_ptr += 4`.
Under the invariant that timestamp is always increasing and the memory bus does not contain any invalid writes at previous timestamps, an attempted memory write access to `mem_ptr > 2^{pointer_max_bits} - 4` will not be able to balance the memory bus: it would require a send at an earlier timestamp to an out of bounds memory address, which the invariant prevents.
Only the starting `mem_ptr` is range checked: since each row will increment `mem_ptr` by `4`, an out
of bounds memory access will occur, causing an imbalance in the memory bus, before `mem_ptr` overflows the field.

On the starting row, the `rem_words` is also read from memory in the form of `rem_words_limbs` limbs.
We also range check the highest limb to be `pointer_max_bits - 24` bits to ensure that calculating `rem_words` by composing the limbs `rem_words_limbs` will not overflow the field. Note that the bound `rem_words < 2^pointer_max_bits` is not tight, since `rem_words` refers to 4-byte words, not bytes.
On each row with `is_buffer = 1`, the `rem_words` is decremented by `1`.

Note: we constrain that when the current instruction ends then `rem_words` is 1. However, we don't constrain that when `rem_words` is 1 then we have to end the current instruction. The only way to exploit this if we to do some multiple of `p` number of additional illegal `is_buffer = 1` rows where `p` is the prime modulus of `F`. However, when doing `p` additional rows we will always reach an illegal `mem_ptr` at some point which prevents this exploit.
