# Circuit Architecture

We build our virtual machines in a STARK proving system with a multi-matrix commitment scheme and shared verifier
randomness between AIR matrices to enable permutation arguments such as log-up.

In the following, we will refer to a circuit as a collection of chips that communicate with one another over buses using a LogUp permutation argument. We refer to messages sent to such a bus as [interactions](https://github.com/openvm-org/stark-backend/blob/main/docs/interactions.md). Every _chip_ is an entity responsible for a certain operation (or set of operations), and it has an AIR to check the correctness of its execution.

> [!NOTE]
> A bus itself doesn't have any logic. It is just an index, and all related functionality is purely on the backend side.

Usually we have _bridges_, which are basically APIs for buses. Using a bridge is preferred over communicating with a bus directly since bridges may enforce some special type of communication (for example, sending messages in pairs or communicating with different buses at once, thus synchronizing these communications).

Our framework is modular and allows the creation of custom VM circuits to support different instruction sets that follow our overall ISA framework.

## Motivation

We want to make the VM modular so that adding new instructions and chips is completely isolated from the existing components.

## Design

The following must exist in any VM circuit:

- **Range checker chip** and **range checker bus**. Every time an AIR needs to constrain that some expression is less than some power of two, it communicates with the range checker bus using the range checker chip. The range checker chip keeps track of all accesses to later balance out the interactions.
- **Program chip** and **program bus**. The program chip's trace matrix simply consists of the program code, one instruction per row, as a cached trace. A cached trace is used so that the commitment to the program code is the proof system trace commitment. Every time an instruction executor (to be defined later) executes an instruction, it sends this instruction, along with the `pc`, to the program bus via the program chip. The program chip keeps track of all accesses to later balance out the interactions.
- **Connector chip**. If we only had the above interactions with the execution bus, then the initial execution state would have only been sent and the final one would have only been received. The connector chip publishes the initial and final states and balances this out (in particular, its trace is a matrix with two rows -- because it has a preprocessed trace).
- **Phantom chip**. We call an instruction _phantom_ if it doesn't mutate execution state (and, of course, the state of the memory). Phantom instructions are sent to the phantom chip.
- A set of memory-related chips and a bus (different depending on the persistence type -- see [Memory](./memory.md)),
- **Execution bus**. Every time an instruction executor executes an instruction, it sends the execution state before the instruction to the execution bus (with multiplicity $1$) and receives the execution state after the instruction from it. It has a convenient **execution bridge** that provides functions which do these two interactions at once.

Notably, there is no CPU chip where the full transcript of executed instructions is materialized in a single trace matrix. The transcript of memory accesses is also not materialized in a single trace matrix.

## Program execution

When the program is being run, in the simple scenario, the following happens at the very highest level:
- There is an _execution state_, which consists of two numbers: _timestamp_ and _program counter_ corresponding to the instruction that is currently being executed.
- While not finished:
  - The next instruction is drawn,
  - It is passed to the _instruction executor_ (which is a special kind of chip, we define it later) responsible for executing this instruction,
  - This instruction executor returns the new execution state (and maybe reports that the execution is finished). The timestamp and program counter change accordingly.

There are limitations to how many interactions/trace rows/etc. we can have in total; see [soundness criteria](https://github.com/openvm-org/stark-backend/blob/main/docs/Soundness_of_Interactions_via_LogUp.pdf). If executing the full program would lead us to overflowing these limits, the program needs to be executed in several segments. Then the process is slightly different:
- After executing an instruction, we may decide (based on `SegmentationStrategy`, which looks at the traces) to _segment_ our execution at this point. In this case, the execution will be split into several segments.
- The timestamp resets on segmentation.
- Each segment is going to be proven separately. Of course, this means that adjacent segments need to agree on some things (mainly memory state). See [Continuations](./continuations.md) for full details.

## Instruction executors

The chips that get to execute instructions are _instruction executors_. While not required, most instruction executor chips are implemented in two parts:
- **Adapter:** communicates with the program and execution buses. Also communicates with memory to read inputs and write output from/to the required locations.
- **Core:** performs chip's intended logic on the raw data. Is mostly isolated and doesn't have to bother about the other parts of the circuit, although it can if it wants, for example, to talk to the range checker.

This modularity helps to separate the functionalities, reduce space for error, and also reuse the same adapters for various chips with similar instruction signatures.
Note that technically these are parts of the same chip and therefore generate one trace, although both adapter and core have AIRs to deal with different parts of the trace.
As already mentioned, it is not required that an instruction executor must be implemented in two parts:
the entire chip with the combined functionality of the adapter and core can be implemented all at once.

> [!IMPORTANT]
> It is a responsibility of the instruction executor (more specifically, the adapter) to update the execution state. It is also its responsibility to constrain that the timestamp increases. If any of these is not done correctly, the proof of correctness will fail to be generated.

To prevent [timestamp overflow](./memory.md#time-goes-forward), we **require** that each instruction executor chip must satisfy the following condition:

> [!IMPORTANT]
> In the AIR, the amount that the timestamp is constrained to increase during execution of a single instruction is at most `num_interactions * num_rows_per_execution`.

Here `num_interactions` is the number of interactions in the AIR: this is a static property of the AIR.
The number of AIR interactions does not depend on the number of trace rows, and it doesn't depend on whether messages are actually sent or not. In general, the trace for the execution of a single instruction can
use multiple rows in the chip's trace matrix: this is represented by `num_rows_per_execution`. In summary
we bound the integer amount that the timestamp should increment in a single instruction execution based on the number of interactions and the number of rows in the trace.

We check that all VM chips contained in the OpenVM system and the standard VM extensions satisfy this condition in the section [below](#inspection-of-vm-chip-timestamp-increments).

## What to take into account when adding a new chip

- [Ensure memory consistency](./memory.md#what-to-take-into-account-when-adding-a-new-chip)
- Do not forget to constrain that `is_valid` is boolean in your core AIR.
- If your chip generates some number of trace rows, and this number is not a power of two, the trace is padded with all-zero rows. It should correspond to a legitimate operation, most likely `is_valid = 0` though. For example, if your AIR asserts that the value in the first column is 1 less than the value in the second column, you cannot just write `builder.assert_eq(local.x + 1, local.y)`, because this is not the case for the padding rows.

### Inspection of VM Chip Timestamp Increments

Below we perform a survey on all VM chips contained in the OpenVM system and the standard VM extensions
to justify that they all satisfy the condition on timestamp increments.

In all AIRs for instruction executors, the timestamp delta of a single instruction execution
is constrained via the `ExecutionBridge` as the difference between the timestamps in the two
interactions on the execution bus for the "from" and "to" states. In most AIRs, the `timestamp_delta`
is a constant which is computed by starting at `0` and incrementing by `1` on each memory access.
The memory access constraint is done via the `MemoryBridge` interface.
Any use of `read` or `write` via `MemoryBridge` uses 4 interactions: 2 on memory bus, 2 for range checks.
Therefore for chips where instruction execution uses only 1 row of the trace and timestamp increments
once per memory access as above, we actually have that `timestamp_delta <= num_interactions / 4`.
This includes all chips that use the integration API and `VmChipWrapper`.

Therefore it remains to examine:

1. All chips that compute the timestamp delta via incrementing by 1 per memory access but where single instruction execution may use multiple trace rows.
2. All cases where the timestamp delta is manually set in a custom way.

The chips that fall into these categories are:

| Name                  | timestamp_delta | # of interactions | Comment                                                                                                                  |
| --------------------- | --------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------ |
| PhantomChip           | 1               | 3 | Case 2. No memory accesses, 3 interactions from program bus and execution bus. |
| KeccakVmChip          | -               | -                 | Case 2. Special timestamp jump. |
| FriReducedOpeningChip | –               | –                 | Case 1. |
| NativePoseidon2Chip   | –               | –                 | Case 1. |
| Rv32HintStoreChip     | –               | –                 | Case 1. |
| Sha256VmChip          | –               | –                 | Case 1. |

The PhantomChip satisfies the condition because `1 < 3`.

All the chips in Case 1 can use a variable number of trace rows to execute a single instruction, but
the AIR constraints all maintain that the timestamp increments by 1 per memory access and this accounts for all increments of the timestamp. Therefore we have `timestamp_delta <= num_interactions * num_rows_per_execution / 4` in these cases.

##### KeccakVmChip

It remains to analyze KeccakVmChip. Here the `KeccakVmAir::timestamp_change` is `len + 45` where `len`
refers to the length of the input in bytes. This is an overestimate used to simplify AIR constraints
because the AIR cannot compute the non-algebraic expression `ceil(len / 136) * 20`.

In the AIR constraints :

- `constrain_absorb` adds at least `min(68, sponge.block_bytes.len())` interactions on the XOR bus.
- `eval_instruction` does an `execute_and_increment_pc` (3), 3 memory reads (12) and 2 lookups (2), giving a total of `17` interactions.
- `constrain_input_read` does 34 memory reads (136),
- `constrain_output_write` does 8 memory writes (32)

In total, there are at least 253 interactions.

A single KECCAK256_RV32 instruction uses `ceil((len + 1) / 136) * 24` rows (where `NUM_ROUNDS = 24`).
We have shown that
```
len + 45 < 253 * ceil((len + 1) / 136) * 24 <= num_interactions * num_rows_per_execution
```
