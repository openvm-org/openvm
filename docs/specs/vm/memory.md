# Overview
Chips in the VM need to perform memory read and write operations. The goal of the memory offline checking is to ensure that memory consistency across all chips. Every memory operation consists of operation type (Read or Write), address_space and address, data, and timestamp. All memory operations across all chips should happen at distinct timestamps between 1 and 2^29. We assume that memory is initialized at timestamp 0. For simplicity, we assume that all memory operations are enabled (there is a way to disable them in the implementation).

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

# Soundness proof
Assume that the MEMORY_BUS interactions and the constraints mentioned above are satisfied.

Fix any address `address` that is used in the segment. To prove memory consistency, it's enough to prove all memory operations on `address` are consistent. Let's look at all interactions done on MEMORY_BUS involving `address`.

Suppose the list of operations involving `address` *sorted* by `timestamp` is `ops_i` for `0 <= i < k`. As discussed above, for every operation `i`, we do one Receive, `r_i`, and one Send, `s_i`, on the MEMORY_BUS. Since the constraint `r_i.timestamp < s_i.timestamp` is enforced, the only way for the MEMORY_BUS interactions to balance out is through the interactions involving `address` done by the Memory Interface. This can be seen by noticing that none of the Receive interactions `r_i` can match `s_{k-1}` as it has the highest timestamp. In fact this implies that the Memory Interface has to do exactly one Receive involving `address` with the final timestamp and data, and, similarly, one Send with the initial timestamp (0) and data. Note that only one such Send and Receive are made as we enforce all addresses in Memory Interface are distinct.

Using a similar technique, by induction, we can show that `s_i.timestamp = r_{i+1}.timestamp` and `s_i.data = r_{i+1}.data` for all `0 < i < k - 1`. Since `(s_i.address, s_i.data, s_i.timestamp) = (ops_i.address, ops_i.data, ops_i.timestamp)` for all operations and `s_i.data = r_i.data` for Read operations, this proves memory consistency for `address`.


# Implementation details
In this model, there is no central memory/offline checker AIR. Every chip is responsible for doing the necessary interactions discussed above for its memory operations. To do this, every chip's AIR  to have some auxiliary columns for every memory operation. The auxiliary columns include `prev_timestamp` and, for Write operations, `prev_data`.

When we use Volatile Memory as the Memory Interface (MemoryAuditChip in the implementation), we do not, on the AIR level, constrain the initial memory. This means that if the first operation on an address is a Read, the corresponding data can be anything -- it's on the program to read from addresses that have been written to. Separately, the MemoryAuditAIR enforces that all addresses are distinct by enforcing sorting, but there are other more efficient ways to do this.
