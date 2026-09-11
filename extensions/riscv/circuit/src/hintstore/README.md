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

On the starting row, `mem_ptr` is read from an RV64 register. It must be an 8-byte-aligned address
in the configured memory range. The AIR range-checks its two u16 limbs and converts the byte address
to a memory-bus block index.

On each row in the same HINT_BUFFER_RV64 instruction, the chip writes 8 bytes to `[mem_ptr:8]_2` and
increments the block index. Execution rejects a buffer past the 32-bit RV64 range. Postflight
enforces the configured memory range, and the AIR enforces the configured pointer bound.

The first HINT_BUFFER_RV64 row also reads `rem_dwords`. It must fit in
`MAX_HINT_BUFFER_DWORDS_BITS` (= 10) bits, and the upper register cells must be zero.
On each row with `is_buffer = 1`, the `rem_dwords` is decremented by `1`.

The AIR requires `rem_dwords = 1` when the instruction ends, but not the converse. Exploiting this
would require a multiple of `p` extra rows, where `p` is the modulus of `F`; the configured address
bound prevents that many 8-byte increments.
