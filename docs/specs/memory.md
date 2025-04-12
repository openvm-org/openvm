# Memory overview

- [Basic performed interactions](#basic-performed-interactions)
- [Access adapters](#access-adapters)
- [Boundary chips](#boundary-chips)
- [Invariants](#invariants)
- [Soundness proof](#soundness-proof)
  - [Time goes forward](#time-goes-forward)
  - [Memory consistency](#memory-consistency)
- [Volatile and persistent memory](#volatile-and-persistent-memory)
  - [Volatile Memory: `VolatileBoundaryChip`](#volatile-memory-volatileboundarychip)
  - [Persistent Memory: `PersistentBoundaryChip`](#persistent-memory-persistentboundarychip)
- [Implementation details](#implementation-details)
- [What to take into account when adding a new chip](#what-to-take-into-account-when-adding-a-new-chip)

---

Chips in the VM need to perform memory read and write operations. The goal of memory checking is to ensure memory consistency across all chips. Every memory operation consists of an operation type (Read or Write), address (`address_space` and `pointer`), data, and timestamp. All memory operations across all chips should happen at distinct timestamps between $1$ and $2^{29}$. We assume that memory is initialized at timestamp $0$. For simplicity, we assume that all memory operations are enabled (there is a way to disable them in the implementation).

Our memory is split into **online** and **offline** memory. Online memory provides actual data and generates access records during program execution, while offline memory uses those records to verify memory consistency. Below we describe how offline memory works. First, we look at the interactions on the memory bus and why they are sound in our ecosystem. Next, we discuss two different memory models, namely, **volatile** and **persistent** memory.

We use the offline memory checking argument of [BEGKN92](https://www.cs.ubc.ca/~will/papers/memcheck.pdf).

## Basic performed interactions

We say an address is _accessed_ when it's initialized, finalized, read from, or written to. An address is initialized at timestamp $0$ and is finalized at the same timestamp it was last read from or written to (or $0$ if there were no operations involving it).

To verify memory consistency, we use MEMORY_BUS to post information about all memory accesses through interactions. More specifically, we use a **memory bridge** that **both** checks the timestamp inequality and performs the necessary interactions:

- To verify Read (`address`, `data`, `new_timestamp`) operations, we need to know `prev_timestamp`, the previous timestamp when the address was accessed. We enforce that `prev_timestamp < new_timestamp`, and perform the following interactions on MEMORY_BUS:
  - Receive (`address`, `data`, `prev_timestamp`),
  - Send (`address`, `data`, `new_timestamp`).
- To verify Write (`address`, `new_data`, `new_timestamp`) operations, we need to know `prev_timestamp` and `prev_data`, the previous timestamp when the address was accessed and the data stored at the address at that time. We enforce that `prev_timestamp < new_timestamp`, and perform the following interactions on MEMORY_BUS:
  - Receive (`address`, `prev_data`, `prev_timestamp`),
  - Send (`address`, `new_data`, `new_timestamp`).

Here is how it's done, for example, for the write operations:

```rust
/// The max degree of constraints is:
/// eval_timestamps: deg(enabled) + max(1, deg(self.timestamp))
/// eval_bulk_access: refer to private function MemoryOfflineChecker::eval_bulk_access
impl<T: FieldAlgebra, V: Copy + Into<T>, const N: usize> MemoryWriteOperation<'_, T, V, N> {
    /// Evaluate constraints and send/receive interactions. `enabled` must be boolean.
    pub fn eval<AB>(self, builder: &mut AB, enabled: impl Into<AB::Expr>)
    where
        AB: InteractionBuilder<Var = V, Expr = T>,
    {
        let enabled = enabled.into();
        self.offline_checker.eval_timestamps(
            builder,
            self.timestamp.clone(),
            &self.aux.base,
            enabled.clone(),
        );

        self.offline_checker.eval_bulk_access(
            builder,
            self.address,
            &self.data,
            &self.aux.prev_data.map(Into::into),
            self.timestamp,
            self.aux.base.prev_timestamp,
            enabled,
        );
    }
}
```

Every instruction executor does this via the **memory controller** API. The memory controller communicates with the memory bus via the **access adapter inventory**, which consists of **access adapters**.

## Access adapters

Consider an example:

- We write `[1, 2, 3, 4]` to `[0, 4)` of address space `2`.
- We write `[5, 6, 7, 8]` to `[4, 8)` of address space `2`.
- We read `[3, 4, 5, 6]` from `[2, 6)` of address space `2`.

Using only interactions above, we won't be able to balance out the `[/* address */ 2, /* data */ 2, 3, 4, 5, 6, /* timestamp */3]` interaction. Access adapters are used to fix this.

Formally speaking, our memory model is as follows:

- Call a set of power-of-two subsegments of $[0,2^{\text{max bits}})$ _nice_. More specifically, a set $S$ of disjoint subsets of $[0,2^{\text{max bits}})$ is nice if the following holds:
  - Every element of $S$ consists of consecutive integers (or _is a subsegment_),
  - Every element of $S$ has a power-of-two size (hence nonempty).
- At any point of time, for each valid address space, we store a nice set of _memory segments_, and for each of them we remember the actual values there.
- In our memory model, the chips are allowed to perform reads and writes from/to power-of-two subsegments of memory (more specifically, subsegments with length that is a power of two not exceeding 32).
- When we want to read from/write to a subsegment, **we transform the nice set for the corresponding address space,** so that our set contains the queried subsegment, using the following operations:
  - _Create_ a new aligned subsegment of length `CHUNK` (which is a power of two). It must not overlap with any other subsegment.
  - _Split_ an existing subsegment by replacing it with its halves.
  - _Merge_ two adjacent subsegments of equal length into one.
- It is easy to see that the set remains nice after each operation.

The _split_ and _merge_ operations for a subsegment of length `N` are handled by `AccessAdapterAir<N>` (and we only have 5 of them, hence up to 32). More specifically:
- We have a `timestamp` associated with each subsegment from our nice set -- basically the timestamp of last read/write of any part in this subsegment.
- Of course, merging two halves creates a subsegment with the timestamp being the maximal of two timestamps for the halves.
- Splitting, however, just makes all child timestamps equal to the former timestamp of the segment being split.
- All these timestamp conditions are checked in the `AccessAdapterAir`.
- When merging two segments `[l, m)` and `[m, r)`, the `AccessAdapterAir` sends to **memory bus** the information about `[l, r)` and receives the information about `[l, m)` and about `[m, r)`, all with multiplicity 1. Splitting does the same, but with multiplicity -1 (or, in other words, receives about `[l, r)` and sends about `[l, m)` and `[m, r)`).
- The information about `[l, r)` sent to the bus is, in this order:
  - address space,
  - `l`,
  - the values at `[l, r)`,
  - the timestamp.

In the example above, if, say, `CHUNK = 1`, then the following interactions are performed:
- Write `[1, 2, 3, 4]` to `[0, 4)` of address space `2`.
  - We _create_ subsegments `[0, 1)`, `[1, 2)`, `[2, 3)`, `[3, 4)` at time `0`:
    - ← `[(2, 0), (0), 0]`.
    - ← `[(2, 1), (0), 0]`.
    - ← `[(2, 2), (0), 0]`.
    - ← `[(2, 3), (0), 0]`.
  - We _merge_ the first pair together and also the second pair together:
    - → `[(2, 0), (0), 0]`, → `[(2, 1), (0), 0]`, ← `[(2, 0), (0, 0), 0]`.
    - → `[(2, 2), (0), 0]`, → `[(2, 3), (0), 0]`, ← `[(2, 2), (0, 0), 0]`.
  - We _merge_ the two segments we have into one:
    - → `[(2, 0), (0, 0), 0]`, → `[(2, 2), (0, 0), 0]`, ← `[(2, 0), (0, 0, 0, 0), 0]`.
  - _Basic interaction_: we send the old data and receive the new data.
    - → `[(2, 0), (0, 0, 0, 0), 0]`.
    - ← `[(2, 0), (1, 2, 3, 4), 1]`.
- Write `[5, 6, 7, 8]` to `[4, 8)` of address space `2`.
  - We do the same steps as above:
    - ← `[(2, 4), (0), 0]`.
    - ← `[(2, 5), (0), 0]`.
    - ← `[(2, 6), (0), 0]`.
    - ← `[(2, 7), (0), 0]`.
    - → `[(2, 4), (0), 0]`, → `[(2, 5), (0), 0]`, ← `[(2, 4), (0, 0), 0]`.
    - → `[(2, 6), (0), 0]`, → `[(2, 7), (0), 0]`, ← `[(2, 6), (0, 0), 0]`.
    - → `[(2, 4), (0, 0), 0]`, → `[(2, 6), (0, 0), 0]`, ← `[(2, 4), (0, 0, 0, 0), 0]`.
  - _Basic interaction_: we send the old data and receive the new data. Notice the new timestamp.
    - → `[(2, 4), (0, 0, 0, 0), 0]`.
    - ← `[(2, 4), (5, 6, 7, 8), 2]`.
- Read `[3, 4, 5, 6]` from `[2, 6)` of address space `2`.
  - Now we have subsegments `[0, 4)` and `[4, 8)` in our nice set. We need to split each of them in two, and then merge the middle two parts.
  First we _split_ the left. Notice that the timestamps of the new parts are the same as the timestamp of the original segment:
    - → `[(2, 0), (1, 2, 3, 4), 1]`, ← `[(2, 0), (1, 2), 1]`, ← `[(2, 2), (3, 4), 1]`.
  - Then we _split_ the right:
    - → `[(2, 4), (5, 6, 7, 8), 2]`, ← `[(2, 4), (5, 6), 2]`, ← `[(2, 6), (7, 8), 2]`.
  - Finally, we _merge_ the middle two segments, making the new timestamp the maximum of the previous two:
    - → `[(2, 2), (3, 4), 1]`, → `[(2, 6), (7, 8), 2]`, ← `[(2, 2), (3, 4, 5, 6), 2]`.
  - _Basic interaction_: we send the old data and receive it again, but with the new timestamp.
    - → `[(2, 2), (3, 4, 5, 6), 2]`.
    - ← `[(2, 2), (3, 4, 5, 6), 3]`.

One can see that the splits and merges mostly balance each other out -- more formally, each subsegment in the nice set, be it born or annihilated by split or merge, corresponds to a receive (→) interaction on birth and a send (←) interaction on death.

## Boundary chips

However, the only unaccounted for interactions are the initial birth of a segment and the final memory state. We use **boundary chips** to handle them.

In **volatile** memory, we use `VolatileBoundaryChip` to handle the initial memory state. It does the following:
- The chip assumes that all initial memory is filled with zeroes.
- However, **initial memory is unconstrained** -- so the result of such program's execution is going to be something along the lines of "there is some way to fill the initial memory that this program finishes correctly".

In **persistent** memory, we use `PersistentBoundaryChip` to handle the final memory state. It does the following:
- There is some initial memory specified in the program.
- The chip _commits_ to the initial memory -- see below.

Both boundary chips perform, for every subsegment ever existed in our nice set, a receive interaction on birth and a send interaction on death.

## Invariants

The following invariants **must** be maintained by the memory architecture:
1. In the MEMORY_BUS, the `timestamp` is always in range `[0, 2^timestamp_max_bits)` where `timestamp_max_bits <= F::bits() - 2` is a configuration constant.
2. In the MEMORY_BUS, the `address_space` is always in range `[0, 1 + 2^as_height)` where `as_height` is a configuration constant satisfying `as_height < F::bits() - 2`. (Our current implementation only supports `as_height` less than the max bits supported by the VariableRangeCheckerBus).
3. In the MEMORY_BUS, the `pointer` is always in range `[0, 2^pointer_max_bits)` where `pointer_max_bits <= F::bits() - 2` is a configuration constant.

Invariant 1 is guaranteed by [time goes forward](#time-goes-forward) under the [assumption](./circuit.md#instruction-executors) that the timestamp increase during instruction execution is bounded by the number of AIR interactions.

Invariant 2 and 3 are guaranteed at timestamp `0` in the MEMORY_BUS by the boundary chips:
- VolatileBoundaryChip constrains the range checks outright.
- PersistentBoundaryChip populates the MEMORY_BUS at timestamp `0` from the initial memory state, which is committed to in the public value `initial_memory_root`. PersistentBoundaryChip upholds Invariants 2 and 3 **assuming** that the initial memory state only contains addresses in the required range. This assumption needs to be checked outside the scope of the circuit.

> [!IMPORTANT]
> At all later timestamps, it is the responsibility of each chip to ensure their memory accesses maintain Invariants 2 and 3.

We note an observation that may be useful in soundness analysis of instruction executor chips: if the `MemoryBridge` is used to add the interactions necessary for a write operation, then under the assumptions that time goes forward and that all memory accesses at previous timestamps are in valid range, any attempt to write to an address out of range will lead to an unbalanced MEMORY_BUS because it will require a send at an earlier timestamp that was also out of bounds.

## Soundness proof

Assume that the MEMORY_BUS interactions and the constraints mentioned above are satisfied.

### Time goes forward

In the connector chip, we constrain that the final timestamp is less than $`2^\text{timestamp\_max\_bits}`$ and that the start timestamp is equal to `1`. It is [guaranteed](https://github.com/openvm-org/stark-backend/blob/main/docs/interactions.md) that the total number of interaction messages is less than $p$. In our current circuit set, all chips increase timestamp [less than they do interactions](./circuit.md#inspection-of-vm-chip-timestamp-increments), which guarantees that the final timestamp cannot overflow: its actual (not mod $p$) value is less than $`2^\text{timestamp\_max\_bits}`$. Given that, our check that `timestamp - prev_timestamp - 1 < 2^timestamp_max_bits` guarantees that `prev_timestamp < timestamp` everywhere we check it.

### Memory consistency

To show that memory is consistent, consider any interactions with segments containing some `address`. They can be done by:

- Boundary chip: one send with the initial timestamp and some value at this address, one receive with the final timestamp and some value at this address. The fact that there is only one send/receive is guaranteed by the fact that we ensure in the boundary chip that all the addresses are distinct.
- Access adapter: each row corresponds to a merge or a split, and therefore for each row, if its split/merged segment contains `address`, we have exactly one send and exactly one receive corresponding to this row. They have the same value at this address, and the send timestamp is at least the receive timestamp -- and strictly greater unless these operations are within one memory access.
- Instruction executors via memory bridge: one receive with the previous timestamp and some value at this address, one send with the new timestamp and some value at this address. We also have `prev_timestamp < timestamp`.

To prove memory consistency, we need to show that, if we only consider basic interactions and boundary interactions, then accessing an address always gives us something that has the last value observed at this address (in a broad sense: for example, this includes "writing to `[3, 5)` generates a record where previous data at `4` is the last value written to or read from `4`, be it from `[4, 5)`, `[2, 10)` or any other segment containing `4`"). In other words, for every receive interaction with timestamp $t$ and value $v$ corresponding to this address, the value $v$ is the value from any _basic_ (not split/merge) send interaction with the greatest existing timestamp $t'$ less than $t$.

For every row that gets handled by a memory bridge, draw a directed edge from the previous timestamp to the new timestamp with capacity $1$. We know that in the obtained network, where the source and the sink are defined by the boundary interactions, all edges go left to right, and the maximal flow is $1$ (because there is only one edge from the source, therefore the minimal cut is $1$). Since all edges go left to right, there are no circulation in this network. Hence, all edges represent exactly one path from the source to the sink. Therefore, for every vertex (which is a timestamp), the value for the edge going to this vertex (the last receive interaction) equals the value for the edge going from this vertex (the considered send instruction).

> [!NOTE]
> Technically, it is possible to add artificial rows in the access adapter AIRs that do nothing: for example, the one corresponding to splitting a segment `[7, 11)` into two and the one merging it back, all at the same timestamp and with the same data completely unrelated to the actual execution. However, this cannot be abused by memory accesses: informally, because this kind of happens at the same time and nobody gets to read these values, and formally, because even if a flow has a circulation, it still can be decomposed into paths and a circulation, and we only consider the path in the argument above.

## Volatile and persistent memory

There are two memory interfaces: **volatile** and **persistent**. From the memory POV, the difference is:
- In volatile memory, both initial and final memory are unconstrained.
- In persistent memory, we commit to the initial memory and to the final memory. These commitments are used for establishing memory consistency between segments.

### Volatile Memory: `VolatileBoundaryChip`

- **Purpose:**
  In volatile memory, we assume that the initial memory is filled with zeros and that it is unconstrained. The `VolatileBoundaryChip` is used mainly for bookkeeping: it tracks the starting state of memory without enforcing a cryptographic commitment. This means that when a program begins execution, the chip simply "accepts" the initial (zeroed) memory without additional checks.

- **Key Points:**
  - **No Commitment:**
    The chip does not compute a Merkle tree root for the initial memory state. Instead, any memory value that gets revealed later comes directly from the execution trace.
  - **Distinctness Check:**
    It ensures that the addresses used are distinct by enforcing a sort order across boundary records.

For instance, this is how distinctness is enforced:

```rust
// Assert local addr < next addr when next.is_valid
// This ensures the addresses in non-padding rows are all sorted
let lt_io = IsLtArrayIo {
    x: [local.addr_space, local.pointer].map(Into::into),
    y: [next.addr_space, next.pointer].map(Into::into),
    out: AB::Expr::ONE,
    count: next.is_valid.into(),
};
// N.B.: this will do range checks (but not other constraints) on the last row if the first row has is_valid = 1 due to wraparound
self.addr_lt_air
    .eval(builder, (lt_io, (&local.addr_lt_aux).into()));
```

### Persistent Memory: `PersistentBoundaryChip`

- **Purpose:**
  When operating in persistent memory mode, the final state of memory must be verifiable. The `PersistentBoundaryChip` commits not only to the final memory state but also to the initial state provided by the program. These commitments become part of the public values used later in proof aggregation, see [Continuations](./continuations.md).

- **Key Points:**
  - **Commitments:**
    The chip takes the initial memory (which is provided as part of the program) and computes a commitment over it—typically by incorporating it into a Merkle tree. Later, when the segment finishes, the chip produces a final commitment (Merkle root) that reflects every change made during execution.
  - **Field: Expand Direction:**
    Each memory chunk is tagged with an `expand_direction` field:
    - `expand_direction = 1` indicates a boundary row representing the initial memory state.
    - `expand_direction = -1` indicates a boundary row representing the final memory state.
    - `expand_direction = 0` marks rows that are not relevant (for example, intermediary rows produced by splits/merges that cancel out).
  - **Multiple Buses:**
    To enforce these commitments, the PersistentBoundaryChip interacts with three buses:
    - **Merkle Bus:**
      For initial memory rows, it sends a record (for example, with a tag `0`), and for final memory rows, it receives a record (with a tag `1`). This allows the Merkle chip to build and later verify a full commitment over the memory.
    - **Compression Bus:**
      It sends the values and hash arrays for each chunk. The multiplicity of these interactions is determined by the square of the `expand_direction` (so that both initial and final rows are treated uniformly in the compression process).
    - **Memory Bus:**
      It performs the basic send/receive interactions that balance out each memory operation, similar to the ones described in the “Basic performed interactions” section.

The uniqueness of the addresses is achieved by the interactions on the merkle bus. The other chip that does interactions there is the **merkle chip**, which build the merkle tree, and to balance everything out, the boundary chip's interactions must correspond to the leaves of the merkle tree, thus, all distinct.

## Implementation details

In this model, there is no central memory/offline checker AIR. Every chip is responsible for doing the necessary interactions discussed above for its memory operations. For this, every chip's AIR must have some auxiliary columns for every memory operation. The auxiliary columns include `prev_timestamp` and, for Write operations, `prev_data`.

When we use Volatile Memory as the Memory Interface (PersistentBoundaryAir in the implementation), we do not, at the AIR level, constrain the initial memory. This means that if the first operation on an address is a Read, the corresponding data can be anything -- it's up to the program to read from addresses that have been written to.

## What to take into account when adding a new chip

Key points:
- For all memory accesses, use memory bridge. It will ensure that all memory interactions are of a certain kind, which we already established soundness for.
- Do not increase the timestamp more than necessary -- otherwise the timestamp may overflow. Ideally, the timestamp should be increased by 1 for every memory access and never otherwise.
- In general, do not communicate with system buses directly.
