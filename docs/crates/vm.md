# VM Architecture and Chips

### `InstructionExecutor` Trait

We define an **instruction** to be a VM **opcode** combined with the **operands** to the opcode. Running the instrumented runtime for an opcode is encapsulated in the following trait:

```rust
pub trait InstructionExecutor<F> {
    /// Runtime execution of the instruction, if the instruction is
    /// owned by the current instance. May internally store records of
    /// this call for later trace generation.
    fn execute(
        &mut self,
        instruction: Instruction<F>,
        from_state: ExecutionState<usize>,
    ) -> Result<ExecutionState<usize>, ExecutionError>;
}
```

### Chips for Opcode Groups

We divide all opcodes in the VM into groups, each of which is handled by a single **chip**. A chip should be a struct of type `C` and associated Air of type `A` which satisfy the following trait bounds:

```rust
C: Chip<SC> + InstructionExecutor<F>
A: Air<AB> + BaseAir<F> + BaseAirWithPublicValues<F>
```

Together, these perform the following functionalities:

- **Keygen:** This is done via the `.eval()` function from `Air<AB>`
- **Trace Generation:** This is done by calling `.execute()` from `InstructionExecutor<F>` which stores execution records and then `generate_air_proof_input()` from `Chip<SC>` which generates the trace using the corresponding records.

**Todo:** make `struct AxVmOpcode(usize)` to protect the global opcode usize.

### Phantom Sub-Instructions

You can specify phantom sub-instruction executors by implementing the trait:

```rust
pub trait PhantomSubExecutor<F> {
    fn phantom_execute(
        &mut self,
        memory: &MemoryController<F>,
        streams: &mut Streams<F>,
        discriminant: PhantomDiscriminant,
        a: F,
        b: F,
        c_upper: u16,
    ) -> eyre::Result<()>;
}

pub struct PhantomDiscriminant(pub u16);
```

The `PhantomChip<F>` maintains a map `FxHashMap<PhantomDiscriminant, Box<dyn PhantomSubExecutor<F>>>` to handle different phantom sub-instructions.

### VM Configuration

**This section needs to be updated for extensions.**

Each specific instantiation of a modular VM is defined in the following structs which handle VMs with/without continuations:

```rust
pub struct VirtualMachine<F: PrimeField32> {
    pub config: VC,
    /// Streams are shared between `ExecutionSegment`s and within each
    /// segment shared with any chip(s) that handle hint opcodes
    streams: Arc<Mutex<Streams<F>>>,
    initial_memory: Option<Equipartition<F, CHUNK>>,
}

pub struct SingleSegmentVM<F: PrimeField32> {
    pub config: VC,
    _marker: PhantomData<F>,
}
```

The `Streams<F>` holds an `input_stream` and `hint_stream`:

```rust
pub struct Streams<F> {
    pub input_stream: VecDeque<Vec<F>>,
    pub hint_stream: VecDeque<F>,
}
```

Configuration of opcodes and memory is handled by:

```rust
pub struct VC {
    /// List of all executors except modular executors.
    pub executors: Vec<ExecutorName>,
    /// List of all supported modulus
    pub supported_modulus: Vec<BigUint>,

    pub poseidon2_max_constraint_degree: usize,
    pub memory_config: MemoryConfig,
    pub num_public_values: usize,
    pub max_segment_len: usize,
    pub collect_metrics: bool,
}

pub struct MemoryConfig {
    pub addr_space_max_bits: usize,
    pub pointer_max_bits: usize,
    pub clk_max_bits: usize,
    pub decomp: usize,
    pub persistence_type: PersistenceType,
}
```

### ZK Operations for the VM

#### Keygen

TODO: Update for `VmChipComplex`.

#### Trace Generation

Trace generation proceeds from:

> `VirtualMachine.execute_and_generate_with_cached_program()`

with subsets of functionality offered by `.execute()` and `execute_and_generate()`. The following struct tracks each continuation segment:

```rust
pub struct ExecutionSegment<F: PrimeField32> {
    pub config: VC,
    pub chip_set: VmChipSet<F>,

    // The streams should be mutated in serial without thread-safety,
    // but the `VmCoreChip` trait requires thread-safety.
    pub streams: Arc<Mutex<Streams<F>>>,

    pub final_memory: Option<Equipartition<F, CHUNK>>,

    pub cycle_tracker: CycleTracker,
    /// Collected metrics for this segment alone.
    /// Only collected when `config.collect_metrics` is true.
    pub(crate) collected_metrics: VmMetrics,
}
```

This will:

- Split the execution into `ExecutionSegment`s using `ExecutionSegment.execute_from_pc()`, which calls `ExecutionSegment.should_segment()` to segment online. Note that this creates a `VmChipSet` for each segment from `VmConfig.create_chip_set()`, where **each segment contains each chip**. It also passes all streams to all segments and runs the generation in serial.
- Generate traces for each segment by calling `VmChipSet.generate_proof_input()`, which iterates through all chips in order and calls `generate_proof_input()`.

#### Proof Generation

This is done by calling `StarkEngine.prove()` on `ProofInput<SC>` created from each segment in `generate_proof_input()`. There is no SDK-level API for this in `VirtualMachine` at present.

## VM Integration API

The integration API provides a way to create chips where the following conditions hold:

- a single instruction execution corresponds to a single row of the trace matrix
- rows of all 0's satisfy the constraints

Most chips in the VM satisfy this, with notable exceptions being Keccak and Poseidon2.

### Traits for Adapter and Core

- `VmAdapterInterface<T>`
- `VmAdapterChip<F>`
- `VmAdapterAir<AB>`
- `VmCoreChip<F, I: VmAdapterInterface<F>>`
- `VmCoreAir<AB, I: VmAdapterInterface<AB::Expr>>`

[!WARNING]
The word **core** will be banned from usage outside of this context.

Main idea: each VM chip will be created from an AdapterChip and a CoreChip. Analogously, the VM AIR is created from an AdapterAir and CoreAir so that the columns of the VM AIR are formed by concatenating the columns from the AdapterAir followed by the CoreAir.

The AdapterChip is responsible for all interactions with the VM system: it owns interactions with the memory bus, program bus, execution bus. It will read data from memory and expose the data (but not intermediate pointers, address spaces, etc.) to the CoreChip and then write data provided by the CoreChip back to memory.

The AdapterAir does not see the CoreAir, but the CoreAir is able to see the AdapterAir, meaning that the same AdapterAir can be used with several CoreAir's. The AdapterInterface provides a way for CoreAir to provide expressions to be included in AdapterAir constraints -- in particular AdapterAir interactions can still involve CoreAir expressions.

Traits with their associated types and functions:

```rust
/// The interface between core AIR and adapter AIR.
pub trait VmAdapterInterface<T> {
    type Reads;
    type Writes;
    type ProcessedInstruction;
}

pub trait VmAdapterChip<F: Field> {
    /// Records generated by adapter before main instruction execution
    type ReadRecord: Send;
    /// Records generated by adapter after main instruction execution
    type WriteRecord: Send;
    /// AdapterAir should not have public values
    type Air: BaseAir<F> + Clone;
    type Interface<T: AbstractField>: VmAdapterInterface<T>;

    fn preprocess(
        &mut self,
        memory: &mut MemoryChip<F>,
        instruction: &Instruction<F>,
    ) -> Result<(Reads<F, Self::Interface<F>>, Self::ReadRecord)>;

    fn postprocess(
        &mut self,
        memory: &mut MemoryChip<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<usize>,
        ctx: AdapterRuntimeContext<F, Self::Interface<F>>,
        read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<usize>, Self::WriteRecord)>;

    /// Populates `row_slice` with values corresponding to `record`.
    /// The provided `row_slice` will have length equal to `self.air().width()`.
    /// This function will be called for each row in the trace which is being used, and all other
    /// rows in the trace will be filled with zeroes.
    fn generate_trace_row(
        &self,
        row_slice: &mut [F],
        read_record: Self::ReadRecord,
        write_record: Self::WriteRecord,
    );
}

pub trait VmAdapterAir<AB: AirBuilder>: BaseAir<AB::F> {
    type Interface: VmAdapterInterface<AB::Expr>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        interface: AdapterAirContext<AB::Expr, Self::Interface>,
    );
}

pub trait VmCoreChip<F, I: VmAdapterInterface<F>> {
    type Record: Send;
    type Air: BaseAirWithPublicValues<F> + Clone;

    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        from_pc: F,
        reads: Reads<F, A::Interface<F>>,
    ) -> Result<(AdapterRuntimeContext<F, A::Interface<F>>, Self::Record)>;

    /// Populates `row_slice` with values corresponding to `record`.
    /// The provided `row_slice` will have length equal to `self.air().width()`.
    /// This function will be called for each row in the trace which is being used, and all other
    /// rows in the trace will be filled with zeroes.
    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record);
}

pub trait VmCoreAir<AB, I>: BaseAirWithPublicValues<AB::F>
where
    AB: AirBuilder,
    I: VmAdapterInterface<AB::Expr>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I>;
}

// For passing from CoreChip to AdapterChip
pub struct AdapterRuntimeContext<T, I: VmAdapterInterface<T>> {
    /// Leave as `None` to allow the adapter to decide the `to_pc` automatically.
    pub to_pc: Option<T>,
    pub writes: I::Writes,
}

// For passing from CoreAir to AdapterAir with T = AB::Expr
pub struct AdapterAirContext<T, I: VmAdapterInterface<T>> {
    /// Leave as `None` to allow the adapter to decide the `to_pc` automatically.
    pub to_pc: Option<T>,
    pub reads: I::Reads,
    pub writes: I::Writes,
    pub instruction: I::ProcessedInstruction,
}
```

[!WARNING]
You do not need to implement `Air` on the struct you implement `VmAdapterAir` or `VmCoreAir` on.

### Creating a Chip from Adapter and Core

To create a chip used to support a set of opcodes in the VM, we start with types

```rust
A: VmAdapterChip
C: VmCoreChip
A::Air: VmAdapterAir
C::Air: VmCoreAir
```

where `A::Air` and `C:Air` are implemented on all relevant `AirBuilder` required by the backend. We can then create `VmChipWrapper` and `VmAirWrapper` types below:

```rust
pub struct VmChipWrapper<F, A: VmAdapterChip<F>, C: VmCoreChip<F, A>> {
    pub adapter: A,
    pub core: C,
    pub records: Vec<(A::ReadRecord, A::WriteRecord, C::Record)>,
    // For accessing memory
    memory: MemoryChipRef<F>,
}

pub struct VmAirWrapper<A, C> {
    pub adapter: A,
    pub core: C,
}
```

They implement the following traits:

- `InstructionExecutor<F>` is implemented on `VmChipWrapper<F, A, C>`, where the `execute()` function:
  - calls `preprocess()` on `A` with `memory` and the raw `instruction`
  - calls `execute_instruction()` on `C` with the raw `instruction`, `from_pc`, and `reads` from `preprocess()`
  - calls `postprocess()` on `A` with the raw `instruction`, `from_state`, the `output: AdapterRuntimeContext` from `execute_instruction()`, and the `read_record`
  - stores the resulting `(read_record, write_record, core_record)`
- `Air<AB>`, `BaseAir<F>`, and `BaseAirWithPublicValues<F>` are implemented on `VmAirWrapper<A::Air, C::Air>`, where the `eval()` function implements constraints via:
  - calls `eval()` on `C::Air`
  - calls `eval()` on `A::Air`
- `Chip<SC>` is implemented on `VmChipWrapper<F, A, C>` with associated Air `VmAirWrapper<A::Air, C::Air>`, where `generate_air_proof_input()` iterates through all records from instruction execution and generates one row of the trace from each record. Importantly, rows which do not correspond to an instruction execution are not affected and left to be **identically zero**. Each used row in the trace is created via:
  - calls `generate_trace_row()` on `A` with the `adapter_row`, `read_record`, `write_record`
  - calls `generate_trace_row()` on `C` with the `core_row`, `core_record`

**Convention:** If you have a new `Foo` functionality you want to support, make structs `FooCoreChip, FooCoreAir`. Either use existing `BarAdapterChip, BarAdapterAir` or make your own. Then typedef

```rust
pub type FooChip<F> = VmChipWrapper<F, BarAdapterChip<F>, FooCoreChip<F>>;
pub type FooAir = VmAirWrapper<BarAdapterAir, FooCoreAir>;
```

If there is a risk of ambiguity, use name `BarFooChip` instead of just `FooChip`.

### Basic structs for shared use

```rust
pub struct BasicAdapterInterface<
    T,
    PI,
    const NUM_READS: usize,
    const NUM_WRITES: usize,
    const READ_SIZE: usize,
    const WRITE_SIZE: usize,
>(PhantomData<T, PI>);

impl<..> VmAdapterInterface for BasicAdapterInterface<..> {
    type Reads = [[T; READ_SIZE]; NUM_READS];
    type Writes = [[T; WRITE_SIZE]; NUM_WRITES];
    type ProcessedInstruction = PI;
}

pub struct MinimalInstruction<T> {
    pub is_valid: T,
    /// Absolute opcode number
    pub opcode: T,
}

pub struct ImmInstruction<T> {
    pub is_valid: T,
    /// Absolute opcode number
    pub opcode: T,
    pub imm: T
}
```