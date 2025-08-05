# VM Architecture and Chips

### `PreflightExecutor` Trait

We define an **instruction** to be an **opcode** combined with the **operands** for the opcode. Running the instrumented
runtime for an opcode is encapsulated in the following trait:

```rust
pub trait PreflightExecutor<F> {
    /// Runtime execution of the instruction, if the instruction is owned by the
    /// current instance. May internally store records of this call for later trace generation.
    fn execute(
        &mut self,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
  ) -> Result<ExecutionState<u32>>;
}
```

There is a `struct VmOpcode(usize)` to protect the global opcode `usize`, which must be globally unique for each opcode
supported in a given VM.

### Chips for Opcode Groups

Opcodes are partitioned into groups, each of which is handled by a single **chip**. A chip should be a struct of
type `C` and associated Air of type `A` which satisfy the following trait bounds:

```rust
C: Chip<SC> + PreflightExecutor<F>
A: Air<AB> + BaseAir<F> + BaseAirWithPublicValues<F>
```

Together, these provide the following functionalities:

- **Keygen:** Performed via the `Air::<AB>::eval()` function.
- **Trace Generation:** This is done by calling `PreflightExecutor::<F>::execute()` which computes and stores
  execution records and then `Chip::<SC>::generate_air_proof_input()` which generates the trace using the corresponding
  records.

### VM AIR Integration

At the AIR-level, for an AIR to integrate with the OpenVM architecture (constrain memory, read the instruction from the program, etc.), the AIR
communicates over different (virtual) buses. There are three main system buses: the memory bus, program bus, and the
execution bus. The memory bus is used to access memory, the program bus is used to read instructions from the program,
and the execution bus is used to constrain the execution flow. These buses are derivable from the `SystemPort` struct,
which is provided by the `VmInventoryBuilder`.

The buses have very low-level APIs and are not intended to be used directly. "Bridges" are provided to provide a cleaner interface for
sending interactions over the buses and enforcing additional constraints for soundness. The two system bridges are
`MemoryBridge` and `ExecutionBridge`, which should respectively be used to constrain memory accesses and execution flow.

### Phantom Sub-Instructions

You can specify phantom sub-instruction executors by implementing the trait:

```rust
pub trait PhantomSubExecutor<F> {
    fn phantom_execute(
        &mut self,
        memory: &MemoryController<F>,
        streams: &mut Streams<F>,
        rng: &mut StdRng,
        discriminant: PhantomDiscriminant,
        a: F,
        b: F,
        c_upper: u16,
    ) -> eyre::Result<()>;
}

pub struct PhantomDiscriminant(pub u16);
```

The `PhantomChip<F>` internally maintains a mapping from `PhantomDiscriminant` to `Box<dyn PhantomSubExecutor<F>>>` to
handle different phantom sub-instructions.

### VM Configuration

Each specific instantiation of a modular VM is defined by the following struct:

```rust
pub struct VirtualMachine<SC: StarkGenericConfig, E, VC> {
  pub engine: E,
  pub executor: VmExecutor<Val<SC>, VC>,
}
```

The engine type `E` should be `openvm_stark_backend::engine::StarkEngine<SC> `and the VM config type `VC` is
`openvm_circuit::arch::config::VmConfig<Val<SC>>`, shown below.

```rust
pub trait VmConfig<F: PrimeField32>: Clone + Serialize + DeserializeOwned {
  type Executor: PreflightExecutor<F> + AnyEnum + ChipUsageGetter;
  type Periphery: AnyEnum + ChipUsageGetter;

  /// Must contain system config
  fn system(&self) -> &SystemConfig;
  fn system_mut(&mut self) -> &mut SystemConfig;

  fn create_chip_complex(
    &self,
  ) -> Result<VmChipComplex<F, Self::Executor, Self::Periphery>, VmInventoryError>;
}
```

A `VmConfig` has two associated types: `Executor` and `Periphery`. The `Executor` is typically an enum over chips that
are instruction executors, while `Periphery` is an enum for the chips that are not.
See [VM Extensions](./vm-extensions.md) for more details.

### ZK Operations for the VM

#### Keygen

Key generation is computed from the `VmConfig` describing the VM. The `VmConfig` is used to create the `VmChipComplex`,
which in turn provides the list of AIRs that are used in the proving and verification process.

#### Trace Generation

Trace generation proceeds from:

> `VirtualMachine::execute_and_generate_with_cached_program()`

with subsets of functionality offered by `VirtualMachine::execute()` and `VirtualMachine::execute_and_generate()`. The
following struct tracks each continuation segment:

```rust
pub struct ExecutionSegment<F: PrimeField32, VC: VmConfig<F>> {
  pub chip_complex: VmChipComplex<F, VC::Executor, VC::Periphery>,
  pub final_memory: Option<Equipartition<F, CHUNK>>,
  pub air_names: Vec<String>,
  pub since_last_segment_check: usize,
}
```

This will:

- Split the execution into `ExecutionSegment`s using `ExecutionSegment.execute_from_pc()`, which calls
  `ExecutionSegment.should_segment()` to segment online. Note that this creates a `VmChipComplex` for each segment from
  `VmConfig.create_chip_set()`, where **each segment contains each chip**. It also passes all streams to all segments
  and runs the generation in serial.
- Generate traces for each segment by calling `VmChipSet.generate_proof_input()`, which iterates through all chips in
  order and calls `generate_proof_input()`.

#### Proof Generation

Prove generation is performed by calling `StarkEngine.prove()` on `ProofInput<SC>` created from each segment in
`generate_proof_input()`. There is no SDK-level API for this in `VirtualMachine` at present.

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

> [!WARNING]
> The word **core** will be banned from usage outside of this context.

Main idea: each VM chip is created from an `AdapterChip` and a `CoreChip`. Analogously, the VM AIR is created from an
`AdapterAir` and `CoreAir` so that the columns of the VM AIR are formed by concatenating the columns from the
`AdapterAir` followed by the `CoreAir`.

The `AdapterChip` is responsible for all interactions with the VM system: it owns interactions with the memory bus,
program bus, execution bus. It will read data from memory and expose the data (but not intermediate pointers, address
spaces, etc.) to the CoreChip and then write data provided by the CoreChip back to memory.

The `AdapterAir` does not see the `CoreAir`, but the `CoreAir` is able to see the `AdapterAir`, meaning that the same
`AdapterAir`
can be used with several `CoreAir`'s. The AdapterInterface provides a way for `CoreAir` to provide expressions to be
included in `AdapterAir` constraints -- in particular `AdapterAir` interactions can still involve `CoreAir` expressions.

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
  /// `AdapterAir` should not have public values
    type Air: BaseAir<F> + Clone;
  type Interface: VmAdapterInterface<F>;

    fn preprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
    ) -> Result<(<Self::Interface as VmAdapterInterface<F>>::Reads, Self::ReadRecord)>;

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        ctx: AdapterRuntimeContext<F, Self::Interface<F>>,
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
        aux_cols_factory: &MemoryAuxColsFactory<F>,
    );

  fn air(&self) -> &Self::Air;
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

// For passing from `CoreAir` to `AdapterAir` with T = AB::Expr
pub struct AdapterAirContext<T, I: VmAdapterInterface<T>> {
    /// Leave as `None` to allow the adapter to decide the `to_pc` automatically.
    pub to_pc: Option<T>,
    pub reads: I::Reads,
    pub writes: I::Writes,
    pub instruction: I::ProcessedInstruction,
}
```

> [!WARNING]
> You do not need to implement `Air` on the struct you implement `VmAdapterAir` or `VmCoreAir` on.

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
    offline_memory: Arc<Mutex<OfflineMemory<F>>>,
}

pub struct VmAirWrapper<A, C> {
    pub adapter: A,
    pub core: C,
}
```

They implement the following traits:

- `PreflightExecutor<F>` is implemented on `VmChipWrapper<F, A, C>`, where the `execute()` function:
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
