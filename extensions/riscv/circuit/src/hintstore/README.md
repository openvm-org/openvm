# RV64 Hint Store Chip

The chip is an instruction executor for the HINT_STORED_RV64 and HINT_BUFFER_RV64 instructions.

Trace rows are exactly one of 3 types:
- `is_single = 1, is_buffer = 0`: to handle HINT_STORED_RV64
- `is_single = 0, is_buffer = 1`: rows for HINT_BUFFER_RV64
- `is_single = 0, is_buffer = 0`: dummy padding rows

A single HINT_BUFFER_RV64 instruction may use multiple contiguous rows. The first row,
which is also the row that will send messages to the program and execution buses with non-zero
multiplicities, is marked with `is_buffer_start = 1` (and it is the only row within the rows for that
instruction with `is_buffer_start = 1`).

On the starting row, a memory address `mem_ptr` is read from the RV64 register as `8` limbs `mem_ptr_limbs`. Since pointers are required to fit in `pointer_max_bits` (≤ 32) bits, the AIR asserts `mem_ptr_limbs[4..8]` are all zero, and range-checks the scaled `mem_ptr_limbs[3] * (1 << (32 - pointer_max_bits))` to an 8-bit lookup. The scaling constrains `mem_ptr < 2^pointer_max_bits` and requires `pointer_max_bits ∈ (24, 32]`. The preflight executor additionally asserts that the upper 32 bits of the rs1 read are zero, matching the pointer-is-u32 convention used in `Rv64LoadStoreAdapter`.

On each row in the same HINT_BUFFER_RV64 instruction, the chip does a write of 8 bytes to `[mem_ptr:8]_2` and increments `mem_ptr += 8`.
Under the invariant that timestamp is always increasing and the memory bus does not contain any invalid writes at previous timestamps, an attempted memory write access to `mem_ptr > 2^{pointer_max_bits} - 8` will not be able to balance the memory bus: it would require a send at an earlier timestamp to an out of bounds memory address, which the invariant prevents.
Only the starting `mem_ptr` is range checked: since each row will increment `mem_ptr` by `8`, an out
of bounds memory access will occur, causing an imbalance in the memory bus, before `mem_ptr` overflows the field.

On the starting row, `rem_dwords` (labelled `rem_words` in the columns struct for historical
reasons) is also read from memory as 8 limbs `rem_words_limbs`. `rem_dwords` is bounded by
`2^MAX_HINT_BUFFER_DWORDS_BITS` (= 2^10), so the AIR asserts `rem_words_limbs[2..8]` are all zero and range-checks the scaled `rem_words_limbs[1] * (1 << (16 - MAX_HINT_BUFFER_DWORDS_BITS))` to an 8-bit lookup. This constrains `rem_dwords < 2^MAX_HINT_BUFFER_DWORDS_BITS` and requires `MAX_HINT_BUFFER_DWORDS_BITS ∈ [8, 16)`.
On each row with `is_buffer = 1`, the `rem_dwords` is decremented by `1`.

Note: we constrain that when the current instruction ends then `rem_dwords` is 1. However, we don't constrain that when `rem_dwords` is 1 then we have to end the current instruction. The only way to exploit this if we to do some multiple of `p` number of additional illegal `is_buffer = 1` rows where `p` is the prime modulus of `F`. However, when doing `p` additional rows we will always reach an illegal `mem_ptr` at some point which prevents this exploit.
