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

The chips that get to execute instructions are _instruction executors_. These chips can be split into two parts:
- **Adapter:** communicates with the program and execution buses. Also communicates with memory to read inputs and write output from/to the required locations.
- **Core:** performs chip's intended logic on the raw data. Is mostly isolated and doesn't have to bother about the other parts of the circuit, although it can if it wants, for example, to talk to the range checker.

This modularity helps to separate the functionalities, reduce space for error, and also reuse the same adapters for various chips with similar instruction signatures.
Note that technically these are parts of the same chip and therefore generate one trace, although both adapter and core have AIRs to deal with different parts of the trace.

> [!IMPORTANT]
> It is a burden of the instruction executor (more specifically, the adapter) to update the execution state. It is also its burden to constrain that the time increases. If any of these is not done correctly, the proof of correctness will fail to be generated.

## Adapter-core interface

The adapter-core interface is defined by the `VmAdapterInterface` trait.
```rust
/// The interface between primitive AIR and machine adapter AIR.
pub trait VmAdapterInterface<T> {
    /// The memory read data that should be exposed for downstream use
    type Reads;
    /// The memory write data that are expected to be provided by the integrator
    type Writes;
    /// The parts of the instruction that should be exposed to the integrator.
    /// This will typically include `is_valid`, which indicates whether the trace row
    /// is being used and `opcode` to indicate which opcode is being executed if the
    /// VmChip supports multiple opcodes.
    type ProcessedInstruction;
}
```

Here, `Reads` is some type that describes the memory reads required for the chip, and `Writes` is similar but about writes. The `VmAdapterChip` defines what the adapter needs to be able to do:

```rust
/// The adapter owns all memory accesses and timestamp changes.
/// The adapter AIR should also own `ExecutionBridge` and `MemoryBridge`.
pub trait VmAdapterChip<F> {
    /// Records generated by adapter before main instruction execution
    type ReadRecord: Send + Serialize + DeserializeOwned;
    /// Records generated by adapter after main instruction execution
    type WriteRecord: Send + Serialize + DeserializeOwned;
    /// AdapterAir should not have public values
    type Air: BaseAir<F> + Clone;

    type Interface: VmAdapterInterface<F>;

    /// Given instruction, perform memory reads and return only the read data that the integrator needs to use.
    /// This is called at the start of instruction execution.
    ///
    /// The implementer may choose to store data in the `Self::ReadRecord` struct, for example in
    /// an [Option], which will later be sent to the `postprocess` method.
    #[allow(clippy::type_complexity)]
    fn preprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )>;

    /// Given instruction and the data to write, perform memory writes and return the `(record, timestamp_delta)`
    /// of the full adapter record for this instruction. This is guaranteed to be called after `preprocess`.
    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        output: AdapterRuntimeContext<F, Self::Interface>,
        read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<u32>, Self::WriteRecord)>;

    /// Populates `row_slice` with values corresponding to `record`.
    /// The provided `row_slice` will have length equal to `self.air().width()`.
    /// This function will be called for each row in the trace which is being used, and all other
    /// rows in the trace will be filled with zeroes.
    fn generate_trace_row(
        &self,
        row_slice: &mut [F],
        read_record: Self::ReadRecord,
        write_record: Self::WriteRecord,
        memory: &OfflineMemory<F>,
    );

    fn air(&self) -> &Self::Air;
}
```

On the other hand, the `VmCoreChip` trait defines the core logic of the chip.

```rust

/// Trait to be implemented on primitive chip to integrate with the machine.
pub trait VmCoreChip<F, I: VmAdapterInterface<F>> {
    /// Minimum data that must be recorded to be able to generate trace for one row of `PrimitiveAir`.
    type Record: Send + Serialize + DeserializeOwned;
    /// The primitive AIR with main constraints that do not depend on memory and other architecture-specifics.
    type Air: BaseAirWithPublicValues<F> + Clone;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        from_pc: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)>;

    fn get_opcode_name(&self, opcode: usize) -> String;

    /// Populates `row_slice` with values corresponding to `record`.
    /// The provided `row_slice` will have length equal to `self.air().width()`.
    /// This function will be called for each row in the trace which is being used, and all other
    /// rows in the trace will be filled with zeroes.
    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record);

    /// Returns a list of public values to publish.
    fn generate_public_values(&self) -> Vec<F> {
        vec![]
    }

    fn air(&self) -> &Self::Air;

    /// Finalize the trace, especially the padded rows if the all-zero rows don't satisfy the constraints.
    /// This is done **after** records are consumed and the trace matrix is generated.
    /// Most implementations should just leave the default implementation if padding with rows of all 0s satisfies the constraints.
    fn finalize(&self, _trace: &mut RowMajorMatrix<F>, _num_records: usize) {
        // do nothing by default
    }
}
```

Here, `AdapterRuntimeContext` is a struct that contains the `to_pc` field, which is the program counter to which the instruction is being executed, and the writes to perform.

```rust
pub struct AdapterRuntimeContext<T, I: VmAdapterInterface<T>> {
    /// Leave as `None` to allow the adapter to decide the `to_pc` automatically.
    pub to_pc: Option<u32>,
    pub writes: I::Writes,
}
```

The workflow is as follows:

1. The adapter calls `preprocess` to get the reads and the read record. It then passes the reads to the core part of the chip.
2. The core calls `execute_instruction` to get the new execution state and the record.
3. The adapter calls `postprocess` on the core output to get the new execution state and the record.

The generated records are used later to fill the trace.

Here is the code that executes this mechanism:

```rust
impl<F, A, M> InstructionExecutor<F> for VmChipWrapper<F, A, M>
where
    F: PrimeField32,
    A: VmAdapterChip<F> + Send + Sync,
    M: VmCoreChip<F, A::Interface> + Send + Sync,
{
    fn execute(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
    ) -> Result<ExecutionState<u32>> {
        let (reads, read_record) = self.adapter.preprocess(memory, instruction)?;
        let (output, core_record) =
            self.core
                .execute_instruction(instruction, from_state.pc, reads)?;
        let (to_state, write_record) =
            self.adapter
                .postprocess(memory, instruction, from_state, output, &read_record)?;
        self.records.push((read_record, write_record, core_record));
        Ok(to_state)
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        self.core.get_opcode_name(opcode)
    }
}
```

> [!NOTE]
> For most of the adapters we have, the corresponding adapter AIR verifies that the timestamp changes by a constant amount. If you want to create your own adapter, take this into account -- see [Memory](./memory.md#what-to-take-into-account-when-adding-a-new-chip) for more detailed description of what to be careful about in this regard. Although we have some set of adapters that should cover most of the cases -- e.g., for execution an instruction with given number of reads and writes:

```rust
/// R reads(R<=2), W writes(W<=1).
/// Operands: b for the first read, c for the second read, a for the first write.
/// If an operand is not used, its address space and pointer should be all 0.
#[derive(Debug)]
pub struct NativeAdapterChip<F, const R: usize, const W: usize> {
    pub air: NativeAdapterAir<R, W>,
    _phantom: PhantomData<F>,
}
```

> [!WARNING]
> While the adapter uses the execution bridge and the memory bridge, the `is_valid` field that serves as the multiplicity in all the corresponding interactions is **not** constrained to be boolean by the adapter. It comes from the core, and it is responsibility of the core AIR to verify its booleanness.

## How to add a new chip to the circuit

1. **Define the chip’s functionality and constraints:**
   - Decide what operation your chip implements (arithmetic, memory, branching, etc.) and write down its semantic rules.
   - Identify the state transitions (e.g. how program counter and timestamp change) and any invariants you need to enforce.
   - Identify the memory reads and writes that your chip performs.

2. **Implement the chip’s core module:**
   Write the circuit-level implementation that enforces your constraints.

3. **Implement an adapter (if necessary):**
   If you need a new adapter, implement it based on the establidhed communication with the core and the number of reads and writes. However, we suggest you to use one of ours whenever possible, for the sake of soundness guarantees.

4. **Merge these two components using `VmChipWrapper`:**
   ```rust
   // In your chip's lib.rs
   pub type YourChip<F> = VmChipWrapper<F, YourAdapterChip<F, N, M>, YourChipCore>;
   ```

5. **Implement a Transpiler Extension:**  
   Implement a type that implements `TranspilerExtension<F>`, so that when your new instruction is encountered in the ELF, it is mapped to your chip’s operations. See [Transpiler](./transpiler.md) for more details.

### What to take into account when adding a new chip

- [Ensure memory consistency](./memory.md#what-to-take-into-account-when-adding-a-new-chip)
- Do not forget to constrain that `is_valid` is boolean in your core.
- If your chip generates some number of trace rows, and this number is not a power of two, the trace is padded with all-zero rows. It should correspond to a legitimate operation, most likely invalid though. For example, if your AIR asserts that the value in the first column is 1 less than the value in the second column, you cannot just write `builder.assert_eq(local.x + 1, local.y)`, because this is not the case for the padding rows.


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
