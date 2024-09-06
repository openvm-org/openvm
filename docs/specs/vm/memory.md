# Overview

Chips in the VM need to perform memory read and write operations. The goal of the memory offline checking is to ensure that memory consistency across all chips. Every memory operation consists of operation type (Read or Write), address (address_space and pointer), data, and timestamp. All memory operations across all chips should happen at distinct timestamps between 1 and 2^29. We assume that memory is initialized at timestamp 0. For simplicity, we assume that all memory operations are enabled (there is a way to disable them in the implementation).

We call an address accessed when it's initialized, finalized, read from, or written to. An address is initialized at timestamp 0 and is finalized at the same timestamp it was last read from or written to (or 0 if there were no operations involving it).

To verify memory consistency, we use MEMORY_BUS to post information about all memory accesses, done through interactions.
- To verify Read (`address`, `data`, `new_timestamp`) operations, we need to know `prev_timestamp`, the previous timestamp the address was accessed. We enforce that `prev_timestamp < new_timestamp`, and do the following interactions on MEMORY_BUS:
    - Send (`address`, `data`, `prev_timestamp`)
    - Receive (`address`, `data`, `new_timestamp`)
- To verify Write (`address, new_data, new_timestamp`) operations, we need to know `prev_timestamp` and `prev_data`, the previous timestamp the address was accessed and the data stored at the address at that time. We enforce that `prev_timestamp` < `new_timestamp`, and do the following interactions on MEMORY_BUS:
    - Send (`address`, `prev_data`, `prev_timestamp`)
    - Receive (`address`, `new_data`, `new_timestamp`)

To initialize and finalize memory, we need a Memory Interface chip. For every `address` used in the segment, suppose it's initialized with `initial_data`, `final_data` is stored at the address at the end of the segment, and `final_timestamp` is the timestamp of the last operation involving it in the segment. Then, the interface chip does the following interactions on MEMORY_BUS:
    - Send (`address`, `initial_data`, 0)
    - Receive (`address`, `final_data`, `final_timestamp`)

Note that all interactions use multiplicity 1. Crucially, the Memory Interface does exactly one such Send and Receive for every `address` used in the segment. In particular, the AIR enforces that all addresses those interactions are done on are distinct.

## Soundness proof
Assume that the MEMORY_BUS interactions and the constraints mentioned above are satisfied.

Fix any address `address` that is used in the segment. To prove memory consistency, it's enough to prove all memory operations on `address` are consistent. Let's look at all interactions done on MEMORY_BUS involving `address`.

Suppose the list of operations involving `address` *sorted* by `timestamp` is `ops_i` for `0 <= i < k`. As discussed above, for every operation `i`, we do one Receive, `r_i`, and one Send, `s_i`, on the MEMORY_BUS. Since the constraint `r_i.timestamp < s_i.timestamp` is enforced, the only way for the MEMORY_BUS interactions to balance out is through the interactions involving `address` done by the Memory Interface. This can be seen by noticing that none of the Receive interactions `r_i` can match `s_{k-1}` as it has the highest timestamp. In fact this implies that the Memory Interface has to do exactly one Receive involving `address` with the final timestamp and data, and, similarly, one Send with the initial timestamp (0) and data. Note that only one such Send and Receive are made as we enforce all addresses in Memory Interface are distinct.

Using a similar technique, by induction, we can show that `s_i.timestamp = r_{i+1}.timestamp` and `s_i.data = r_{i+1}.data` for all `0 < i < k - 1`. Since `(s_i.address, s_i.data, s_i.timestamp) = (ops_i.address, ops_i.data, ops_i.timestamp)` for all operations and `s_i.data = r_i.data` for Read operations, this proves memory consistency for `address`.

## Implementation details
In this model, there is no central memory/offline checker AIR. Every chip is responsible for doing the necessary interactions discussed above for its memory operations. To do this, every chip's AIR  to have some auxiliary columns for every memory operation. The auxiliary columns include `prev_timestamp` and, for Write operations, `prev_data`.

When we use Volatile Memory as the Memory Interface (MemoryAuditChip in the implementation), we do not, on the AIR level, constrain the initial memory. This means that if the first operation on an address is a Read, the corresponding data can be anything -- it's on the program to read from addresses that have been written to. Separately, the MemoryAuditAIR enforces that all addresses are distinct by enforcing sorting, but there are other more efficient ways to do this.

# Batch Memory Access

*Under construction: The below spec is not yet implemented.*

It is common for a chip to perform batch memory accesses. For example, the `FieldExtensionArithmeticChip` reads/writes
field extension elements which are represented by four contiguous cells.
The compression function of the `Posiedon2VmChip` reads sixteen contiguous cells and outputs eight. In this section we
outline how we support such batch accesses by composing with the memory interaction argument detailed above.
Our implementation supports reading 1, 4, or 8 contiguous cells at a time. One must always an N-cell (a list of N
contiguous cells) at a memory address whose (canonical representation) is divisible by N.

A batch access of an N-cell is specified by a tuple of the following
fields: (`address`, `data`, `timestamp`, `prev_data`, `prev_timestamps`),
where `data, prev_data: [F; N]` and `prev_timestamps: [F; N]`. The pair (`prev_data`, `prev_timestamps`) specifies the
data and last-access timestamps
of the cells `address[i], ..., address[i + N - 1]`. The field `data` specifies the new data, and `timestamp` the new
timestamp.
A batch access increments the global timestamp by one, and upon access, all last-accessed timestamps for the cells in
the batch are updated to the same new value `timestamp`.
Note that a read is simply an access where `data = prev_data`.

We now describe the offline memory checking procedure for supporting exactly two types of accesses: N-cells and 1-cells.
Every batch access of an N-cell results in sending (`address`, `data[0]`, ..., `data[N - 1]`, `timestamp`) across the
**memory bus**. To ensure consistency, we must also receive the previous data. For this there are two cases:

- If the N-cell was previously accessed as an N-cell,
  then (`address`, `prev_data[0]`, ..., `prev_data[N - 1]`, `prev_timestamps[0]`) is received.
- Otherwise, we receive the N
  tuples (`address`, `prev_data[0]`, `prev_timestamps[0]`), ..., (`address + N - 1`, `prev_data[N - 1]`, `prev_timestamps[N - 1]`).

Note: The case analysis above is where we assume N-cell accesses happen at addresses divisible by N. Otherwise the
second
case would not properly handle the situation in which some but not all of the cells were last accessed to by an N-cell
access.

For accessing 1-cells, the situation is similar. To access a 1-cell that was previously accessed in a 4-cell, we receive
(`address`, `prev_data[0]`, ..., `prev_data[N-1]`, `prev_timestamps[0]`) and send N tuples. The `i`th of the `N` tuples
is either (`address + i`, `prev_data[i]`, `timestamp`) or (`address + i`, `data[i]`, `timestamp`), where it is
the former (using `prev_data`) if this 1-cell is not being accessed and the latter otherwise.

The interactions of the first type are handled directly by the `MemoryChip`. The interactions of the second type—which
adapt between the two "word sizes"—are handled by the `MemoryAccessAdapterChip`. Each row of
the `MemoryAccessAdapterChip`'s trace specifies the data needed to send and receive the interactions.

To support 1-cell, 4-cell, 8-cell, and 16-cell batch accesses, we use three `MemoryAccessAdapterChip`s, one for each
conversion between adjacent types (1-to-4, 4-to-8, and 8-to-16). This means that to go from sixteen cells accessed as
1-cells to 16-cells, interactions of all types will be sent/received to translate.

To facilitate trace generation, the `MemoryChip` keeps trace of the last access type of each cell. If a batch access
covers cells with mixed previous-access-types, the `MemoryChip` will notify the `MemoryAccessAdapterChip`. The
`MemoryAccessAdapter` will record this memory conversion event in a log that is later used for trace generation.