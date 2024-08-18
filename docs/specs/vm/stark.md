# STARK Architecture

We build our virtual machines in a STARK proving system with a multi-matrix commitment scheme and shared verifier randomness between AIR matrices to enable permutation arguments such as log-up.

In the following, we will refer to a circuit as a collection of AIR matrices (also referred to as chips) of possibly different heights, which may communicate with one another over buses using a log-up permutation argument. We refer to messages sent to such a bus as [interactions](../../../stark-backend/src/interaction/README.md).

Our framework is modular and allows the creation of custom VM circuits to support different instruction sets that follow our overall ISA framework.

## Motivation

We want to make the VM modular, so that adding new instructions and chips involves minimal to no changes to any centralized chip (commonly the CPU chip). We also want to avoid increasing the columns/interactions/buses of the CPU when we add new instructions/chips.

## Design

The following must exist in any VM circuit:

- The program chip
- A program bus
- An execution bus
- A memory bus

Notably, there is no CPU chip where the full transcript of executed instructions is materialized in a single trace matrix. The transcript of memory accesses is also not materialized in a single trace matrix. We discuss reasons for these choices below.

### Program Chip

We follow the Harvard architecture where the program code (ROM) is stored separately from memory. The program chip's trace matrix simply consists of the program code, one instruction per row, as a cached trace, together with interactions on the PROGRAM_BUS.

A cached trace is used so that the commitment to the program code is the proof system trace commitment. This commitment could be changed to a flat hash, likely with worse performance.

### No-CPU

The main motivation is that the existence of a CPU forces the existence of a trace matrix with rows growing with the total number of clock cycles of the program execution. We claim that the no-CPU design gives the minimum lower bound on the number of required trace cells added per opcode execution.

Traditionally, the CPU is in charge of reading/writing from memory and forwarding that information to the appropriate chip. We are changing to a model where each chip directly accesses memory itself. Traditionally this is also inefficient because the CPU uses physical general purpose registers for instruction execution, whereas in our architecture, registers are emulated as memory in a dedicated address space.

Each chip has IO columns `(timestamp, pc, instruction)` where `instruction` is `(opcode, operands)`.
The chip would receive `(pc, instruction)` on the PROGRAM_BUS to ensure it is reading the correct line of the program code.
There is a maximum length to `operands` defined by the PROGRAM_BUS, but each chip can receive only a subset of the operands (setting the rest to zero) without paying the cost for the unused operands.

Each chip receives `(timestamp, pc)` on EXECUTION_BUS and "after"
executing an instruction, sends `(new_timestamp, new_pc)` on the same bus.
The chip is in charge of constraining that `new_timestamp > timestamp`. (Here we say "after" to correspond to the temporal nature of runtime execution, but there is no before/after in the AIR matrix itself.)
The bus enforces that each timestamp transition corresponds to a particular instruction being executed.

The chip must constrain that `opcode` is one of the opcodes the chip itself owns. The chip then constrains the rest of the validity of the opcode execution, according to the opcode spec.

When there are no continuations, there will need to be another very simple source/sink chip with a 2 row trace that sends out (1, 0) on EXECUTION_BUS and receives `(final_timestamp, final_pc)` on EXECUTION_BUS. With continuations, the start and end timestamp/pc will need to be constrained with respect to the pre/post-states.

See [internal doc](https://docs.google.com/document/d/1-UkvxiW5tvYH5qw7O4t2WjMIY8v2Gso9kt_MrWW5hPg/edit?usp=sharing) for discussion of alternatives and the trace cell cost analysis.

### Offline Memory

In the no-CPU design, each chip receives the opcode instruction directly, and memory access (read or write) is constrained by the chip itself.

The VM supports a read/write memory, which is constrained via offline memory checking. We use the offline memory checking argument of [BEGKN92](https://www.cs.ubc.ca/~will/papers/memcheck.pdf).

Any offline memory checking aims to have a transcript consisting of `(a, v, t)` with address `a`, value `v`, timestamp `t` (in our ISA `a = (address_space, address)` but we omit this distinction here for brevity). The timestamp here is a single field element. As far as we know, the timestamp **must** be global and match what is used by the EXECUTION_BUS to ensure that the temporal sequencing of memory accesses matches the temporal sequencing of instruction execution.

<!--
[JPW] Lasso uses a per-address timestamp (renamed counter) but in a different setting. We did not see a way to use this argument because it did not allow constraining instruction execution matched memory access ordering.
-->

Memory aims to support two operations: read and write.

- A read of `(a, v)` at time `t` means that if we look through the transcript focusing on only entries with address `a`, the entry with timestamp immediately preceding `t` must have value also equal `v`.

- A write of `(a, v)` at time `t` means a new entry `(a, v, t)` must be introduced to the transcript.

The main distinction of [BEGKN92] is that the transcript does not need to be materialized explicitly in a single AIR matrix. The particular entries of the transcript are materialized on a per-access basis (in whatever chip needs it). This materialization is avoided by using the MEMORY_BUS and chip constraints to constrain the correctness of the transcript:

We have an offline checking MEMORY_BUS, where message fields consist of `(a, v, t)`. The bus then has two sets (send vs receive) that must be equal at the end. To match the literature, let send (resp. receive) correspond to Write (resp. Read) sets. Any memory access in a chip must add one entry into each set and constrain a relation between them:

- A read of $(a, v)$ at time $t$ must add $(a, v, t_{prev})$ to Read set and $(a, v, t)$ to Write set. It must constrain $t_{prev} < t$.
- A write of $(a, v)$ at time $t$ must add $(a,v_{prev},t_{prev})$ to Read set and $(a,v,t)$ to Write set, where $v_{prev}$ is the previous value before the write. It must constrain $t_{prev} < t$.

To balance the Read and Write sets, an additional chip must ensure that every accessed address has an initial $(a, v_{init}, 0)$ added to the Write set, and $(a, v_{final}, t_{last})$ added to Read set.

For the offline checking to be sound, it must be constrained that the list of accessed addresses in the initial Write list are all **unique**. Uniqueness of the initial address list implies, together with the bus argument, that the final address list is also unique (vice versa, uniqueness of final set implies uniqueness of initial set). The key observation of [OLB24](https://eprint.iacr.org/2024/979.pdf) is that only _uniqueness_ is necessary and not sorted-ness of the address list. The traditional approach prior to OLB24 to enforce uniqueness is to enforce the list of addresses is sorted, which uses logup lookups for range checks necessary to constrain `IsLessThan`. OLB24 shows that one can implement an AIR with in-circuit randomness to constrain that all entries in a trace column are unique (with an extension to conditional uniqueness).

The initial and final memory accesses are constrained different when the VM has continuations. See [Continuations](./continuations.md) for full details. In summary, because the initial and final memory states are committed to in a **trie**, the uniqueness of the addresses is constrained by the trie, so the arguments of the previous paragraph are not used.

### Memory Model With Variable Word Size

In traditional machine memory models, memory is stored as a sequence of cells (typically bytes), chunked into words by a fixed **word size**.
This word size then governs all memory load/store operations.
This model was governed by the constraints of physical hardware, and
we now discuss why it is unnecessary in the STARK architecture.

For efficient vectorization of memory accesses, we allow each
chip to perform "batch" read/writes of $\{(a + i, v_i, t)\}_{i \in [0,w)}$ where $w$ is any multiple of a fixed `WORD_BLOCK_SIZE` (set to `4` in practice).

The main idea is that in the offline checking memory argument [above](#offline-memory), the MEMORY_BUS can hold $(a, v, t)$ where the length of $v$ is variable. The difference in word sizes only needs to be resolved when there is a sequence of read+write or write+read involving different word sizes.

We introduce chips MemoryConverterChip[$w$] and MemoryPackChip[$w$] where

- MemoryConverterChip: read $(a, v_0 || ... || v_{B-1}, t)$ and write $(a + B \cdot i, v_i, t + 1)$ for $i \in [0,B)$ where $B$ is `WORD_BLOCK_SIZE`, or
- MemoryConverterChip: read $(a + B \cdot i, v_i, t - 1)$ for $i \in [0,B)$ and write $(a, v_0 || ... || v_{w-1}, t)$ to the memory.
- MemoryPackChip: read $(a + B \cdot i, v_i, t_{prev,i})$ for $i \in [0,B)$ and write $(a, v_0 || ... || v_{w-1}, t)$ to the memory. It must constrain $t_{prev,i} < t$. The difference between this and MemoryConverterChip is that it allows packing smaller words that may have been modified at different previous times. This packing is more expensive.

Here the length of $v_i$ is $w$.

The values of $a, v_i$ that appear in the trace of the converter chip is generated on-demand based on the needs of the runtime memory access. In other words, the converter inserts additional writes into the MEMORY_BUS when needed in order to link up accesses of different word sizes.

## Acknowledgements

- Plonky3
- Valida
- SP1
