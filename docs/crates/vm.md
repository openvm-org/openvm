# VM Architecture and Chips

## Execution

OpenVM provides a modular interface to add VM instructions via the extension API. The `VmExecutionExtension` trait allows one to specify various execution extensions. An extension consists of executor structs that handle specific instruction opcodes and must implement the `Executor`, `MeteredExecutor`, and `PreflightExecutor` traits, corresponding to different execution modes.

We define an **instruction** to be an **opcode** combined with the **operands** for the opcode. Each opcode must be mapped to a specific executor that contains the logic for executing the instruction.
There is a `struct VmOpcode(usize)` to protect the global opcode `usize`, which must be globally unique for each opcode supported in a given VM.

### Execution Modes

#### Pure Execution

Pure execution runs the program without any overhead and is used to obtain the final VM state at termination, or after executing a fixed number of instructions.

The `Executor<F>` trait defines the interface for pure execution:

```rust
pub trait Executor<F> {
    fn pre_compute_size(&self) -> usize;

    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait;
}
```

where `ExecuteFunc<F, Ctx>` is a function pointer that contains the instruction execution logic.

```rust
pub type ExecuteFunc<F, CTX> =
    unsafe fn(pre_compute: &[u8], exec_state: &mut VmExecState<F, GuestMemory, CTX>);
```

Each executor pre-computes instruction-specific data during a preprocessing step and returns function pointers for direct instruction execution.

#### Metered Execution

Metered execution tracks the trace heights for each chip along with normal execution. This mode divides the execution into segments, where each segment consists of an instruction range and an (over)estimate of the resulting trace heights for each chip in the segment. Segmentation is done based on configurable limits like maximum trace height, maximum trace cells etc.

The `MeteredExecutor<F>` trait defines the interface for metered execution:

```rust
pub trait MeteredExecutor<F> {
    fn metered_pre_compute_size(&self) -> usize;

    fn metered_pre_compute<Ctx>(
        &self,
        air_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait;
}
```

Each executor is associated with a chip and an AIR. This mapping is defined implicitly by the VM extension. The additional `air_idx` parameter is the index of the executor's AIR in the verifying key. This is used for indexing the trace height of the chip in the `trace_heights` array contained in the `Segment` struct.

#### Preflight Execution

Preflight execution creates execution [records](records.md) of the record arena type `RA` which are needed for trace generation. Preflight execution doesn't have a precompute mechanism and uses runtime dispatch to execute each instruction.

The `PreflightExecutor<F, RA>` trait defines the interface for preflight execution:

```rust
pub trait PreflightExecutor<F, RA> {
    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError>;
}
```

### Interpreter Architecture

The `InterpretedInstance` represents the VM interpreter and handles pure and metered execution modes. More specifically, it:

- Pre-computes instruction-specific data buffers
- Generates function pointer tables for direct execution
- Supports optional tail call optimization (TCO) for improved performance

The `PreflightInterpretedInstance` handles preflight execution with:

- Runtime instruction dispatch (as opposed to the precomputed function pointers used in pure/metered execution)
- Execution record collection in record arenas `RA`
- Per-instruction frequency tracking to be used by the `ProgramChip`

### Chips for Opcode Groups

Opcodes are partitioned into groups, each of which is handled by a single executor, air and **chip**. Executor is the struct that contains logic for executing an opcode and generating records. A chip is an object that contains logic for converting execution records into a trace matrix. And AIR contains the arithmetic and lookup constraints on the trace matrix required to create a proof of execution.

```rust
pub trait Chip<R, PB: ProverBackend> {
    /// Generate all necessary context for proving a single AIR.
    fn generate_proving_ctx(&self, records: R) -> AirProvingContext<PB>;
}
```

where `PB` is either `CpuBackend` or `GpuBackend`.

As mentioned above, the executor should implement the three executor traits: `Executor`, `MeteredExecutor`, and `PreflightExecutor`.

```rust
ChipExecutor: Executor<F> + MeteredExecutor<F> + PreflightExecutor<F, RA>
```

The AIR `A` should have the following trait bounds:

```rust
A: Air<AB> + BaseAir<F> + BaseAirWithPublicValues<F>
```

where `AB` is an `AirBuilder`

Together, these provide the following functionalities:

- **Keygen:** Performed via the `Air::<AB>::eval()` function.
- **Trace Generation:** This is done by calling `PreflightExecutor::<F, RA>::execute()` which computes the execution records and then `Chip::<R, PB>::generate_proving_ctx()` which generates the trace by consuming the execution records.

### VM AIR Integration

At the AIR-level, for an AIR to integrate with the OpenVM architecture (constrain memory, read the instruction from the program, etc.), the AIR communicates over different (virtual) buses. There are three main system buses: the memory bus, program bus, and the
execution bus. The memory bus is used to access memory, the program bus is used to read instructions from the program,
and the execution bus is used to constrain the execution flow. These buses are derivable from the `SystemPort` struct,
which is provided by `AirInventory`/`SystemAirInventory`.

The buses have very low-level APIs and are not intended to be used directly. "Bridges" are provided to provide a cleaner interface for
sending interactions over the buses and enforcing additional constraints for soundness. The two system bridges are
`MemoryBridge` and `ExecutionBridge`, which should respectively be used to constrain memory accesses and execution flow.

### Phantom Sub-Instructions

Phantom sub-instructions are instructions that affect the runtime and trace matrix values but have no AIR constraints besides advancing the PC by `DEFAULT_PC_STEP`. They should not mutate memory, but they can mutate the input & hint streams.

You can specify phantom sub-instruction executors by implementing the trait:

```rust
pub trait PhantomSubExecutor<F>: Send + Sync {
    fn phantom_execute(
        &self,
        memory: &GuestMemory,
        streams: &mut Streams<F>,
        rng: &mut StdRng,
        discriminant: PhantomDiscriminant,
        a: u32,
        b: u32,
        c_upper: u16,
    ) -> eyre::Result<()>;
}

pub struct PhantomDiscriminant(pub u16);
```

The `PhantomExecutor<F>` internally maintains a mapping from `PhantomDiscriminant` to `Arc<dyn PhantomSubExecutor<F>>` to
handle different phantom sub-instructions.

### VM Configuration

Each specific instantiation of a modular VM is defined by the `VirtualMachine` struct, which contains the API to generate proofs for arbitrary programs for a fixed set of OpenVM instructions and a fixed VM circuit corresponding to those instructions. This struct represents the complete zkVM.

The `VirtualMachine` can be constructed using:

```rust
impl<E, VB> VirtualMachine<E, VB>
where
    E: StarkEngine,
    VB: VmBuilder<E>,
{
    pub fn new(
        engine: E,
        builder: VB,
        config: VB::VmConfig,
        d_pk: DeviceMultiStarkProvingKey<E::PB>,
    ) -> Result<Self, VirtualMachineError>;

    pub fn new_with_keygen(
        engine: E,
        builder: VB,
        config: VB::VmConfig,
    ) -> Result<(Self, MultiStarkProvingKey<E::SC>), VirtualMachineError>;
}
```

The engine type `E` should implement the `openvm_stark_backend::engine::StarkEngine` trait and the VM builder type `VB` implements `VmBuilder<E>`, which provides the VM configuration through `VB::VmConfig`.

```rust
pub trait VmConfig<SC>:
    Clone
    + Serialize
    + DeserializeOwned
    + InitFileGenerator
    + VmExecutionConfig<Val<SC>>
    + VmCircuitConfig<SC>
    + AsRef<SystemConfig>
    + AsMut<SystemConfig>
where
    SC: StarkGenericConfig,
{
}
```

A `VmConfig` should implement the `VmExecutionConfig` trait which provides execution configuration. The `Executor` type is typically an enum over executor structs that handle instruction execution.

```rust
pub trait VmExecutionConfig<F> {
    type Executor: AnyEnum + Send + Sync;

    fn create_executors(&self)
        -> Result<ExecutorInventory<Self::Executor>, ExecutorInventoryError>;
}
```

Finally, `VmConfig` should also implement the `VmCircuitConfig` trait which provides the AIRs for all chips in the VM. The `AirInventory` contains all AIRs required for constraining the execution trace of each chip.

```rust
pub trait VmCircuitConfig<SC: StarkGenericConfig> {
    fn create_airs(&self) -> Result<AirInventory<SC>, AirInventoryError>;
}
```

See [VM Extensions](./vm-extensions.md) for more details.

### ZK Operations for the VM

#### Keygen

Key generation is computed from the `VmConfig` describing the VM. The `VmConfig` is used to create the `AirInventory` via the `VmCircuitConfig` trait,
which in turn provides the list of AIRs that are used in the proving and verification process.

```rust
pub trait VmCircuitConfig<SC: StarkGenericConfig> {
    fn create_airs(&self) -> Result<AirInventory<SC>, AirInventoryError>;
}
```

The `AirInventory` contains a `keygen` method that generates the proving and verifying keys from the collected AIRs.

#### Trace Generation

Trace generation uses the records generated in preflight execution and proceeds from:

> `VirtualMachine::generate_proving_ctx()`

which consumes the execution records and generates the final trace matrices.

For execution with multiple segments (continuations), the trace generation process is handled by `VmInstance` and proceeds as follows:

1. **Metered Execution**: First run metered execution to determine segment boundaries using `execute_metered()` which returns a list of `Segment` structs containing:
   ```rust
   pub struct Segment {
       pub instret_start: u64,
       pub num_insns: u64,
       pub trace_heights: Vec<u32>,
   }
   ```

2. **Segment Trace Generation**: For each segment:
   - Recover the starting VM state at the beginning of the segment via pure execution from the program start (only necessary in a distributed setup)
   - Run preflight execution for the segment using `execute_preflight()` with the predetermined trace heights
   - Generate trace context from system records and record arenas via `generate_proving_ctx()`
   - Pass final state as initial state to next segment (only necessary in a local setup when proving is done on a single machine)

This approach ensures each segment has properly allocated record arenas based on metered execution estimates, and enables distributed proving where each segment can be proven independently by first recovering its starting state.

#### Proof Generation

Proof generation is performed by calling `StarkEngine.prove()` on `ProvingContext<E::PB>` created for each segment in
`generate_proving_ctx()`. For continuation proofs, each segment is proven independently using the stark engine.

## VM Integration API

The integration API provides a way to create chips where the following conditions hold:

- a single instruction execution corresponds to a single row of the trace matrix
- rows of all 0's satisfy the constraints

Most chips in the VM satisfy this, with notable exceptions being Keccak, SHA256 and Poseidon2.

### Architecture

The integration API separates chip functionality into two distinct layers:

1. **AIR**: Defines arithmetic constraints and interactions with system buses
2. **Execution/Trace generation**: Handles execution and trace generation

### AIR traits for Adapter and Core

The AIR layer consists of adapter and core components that define the constraint logic:

- `VmAdapterInterface<T>` - defines the interface between adapter and core
- `VmAdapterAir<AB>` - handles system interactions (memory, program, execution buses)
- `VmCoreAir<AB, I>` - implements instruction-specific arithmetic constraints

> [!WARNING]
> The word **core** will be banned from usage outside of this context.

Main idea: each VM chip AIR is created from an adapter and core components. The VM AIR is created from an
`AdapterAir` and `CoreAir` so that the columns of the VM AIR are formed by concatenating the columns from the
`AdapterAir` followed by the `CoreAir`.

The adapter is responsible for all interactions with the VM system: it handles interactions with the memory bus,
program bus, execution bus. It reads data from memory and exposes the data (but not intermediate pointers, address
spaces, etc.) to the core and then writes data provided by the core back to memory.

The `AdapterAir` does not see the `CoreAir`, but the `CoreAir` is able to see the `AdapterAir`, meaning that the same
`AdapterAir` can be used with several `CoreAir`s. The AdapterInterface provides a way for `CoreAir` to provide expressions to be
included in `AdapterAir` constraints -- in particular `AdapterAir` interactions can still involve `CoreAir` expressions.

AIR traits with their associated types and functions:

```rust
/// The interface between core AIR and adapter AIR.
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

pub trait VmAdapterAir<AB: AirBuilder>: BaseAir<AB::F> {
    type Interface: VmAdapterInterface<AB::Expr>;

    /// [Air](openvm_stark_backend::p3_air::Air) constraints owned by the adapter.
    /// The `interface` is given as abstract expressions so it can be directly used in other AIR
    /// constraints.
    ///
    /// Adapters should document the max constraint degree as a function of the constraint degrees
    /// of `reads, writes, instruction`.
    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        interface: AdapterAirContext<AB::Expr, Self::Interface>,
    );

    /// Return the `from_pc` expression.
    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var;
}

pub trait VmCoreAir<AB, I>: BaseAirWithPublicValues<AB::F>
where
    AB: AirBuilder,
    I: VmAdapterInterface<AB::Expr>,
{
    /// Returns `(to_pc, interface)`.
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I>;

    /// The offset the opcodes by this chip start from.
    /// This is usually just `CorrespondingOpcode::CLASS_OFFSET`,
    /// but sometimes (for modular chips, for example) it also depends on something else.
    fn start_offset(&self) -> usize;

    fn start_offset_expr(&self) -> AB::Expr {
        AB::Expr::from_canonical_usize(self.start_offset())
    }

    fn expr_to_global_expr(&self, local_expr: impl Into<AB::Expr>) -> AB::Expr {
        self.start_offset_expr() + local_expr.into()
    }

    fn opcode_to_global_expr(&self, local_opcode: impl LocalOpcode) -> AB::Expr {
        self.expr_to_global_expr(AB::Expr::from_canonical_usize(local_opcode.local_usize()))
    }
}

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

### Execution and Trace Generation Traits

The execution layer handles execution and trace generation, separate from the constraint logic:

- `AdapterTraceExecutor<F>` - handles adapter-level execution (memory accesses)
- `AdapterTraceFiller<F>` - fills adapter columns in the trace matrix
- `PreflightExecutor<F, RA>` - handles instruction execution logic and generates records
- `TraceFiller<F>` - fills complete trace rows (adapter + core)

The executor components generate the records that are later used by the trace filler to populate the trace matrix.

```rust
/// A helper trait for expressing generic state accesses within the implementation.
pub trait AdapterTraceExecutor<F>: Clone {
    const WIDTH: usize;
    type ReadData;
    type WriteData;
    type RecordMut<'a> where Self: 'a;

    fn start(pc: u32, memory: &TracingMemory, record: &mut Self::RecordMut<'_>);

    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut Self::RecordMut<'_>,
    ) -> Self::ReadData;

    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut Self::RecordMut<'_>,
    );
}

pub trait AdapterTraceFiller<F>: Send + Sync {
    const WIDTH: usize;
    /// Post-execution filling of rest of adapter row.
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, adapter_row: &mut [F]);
}

pub trait TraceFiller<F>: Send + Sync {
    /// Populates `trace`
    fn fill_trace(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        trace: &mut RowMajorMatrix<F>,
        rows_used: usize,
    ) where
        F: Send + Sync + Clone;

    /// Populates `row_slice` with values corresponding to the record.
    /// The provided `row_slice` will have length equal to the width of the AIR.
    /// This function will be called for each row in the trace which is being used.
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]);

    ...
}
```

### Creating a Chip from Adapter and Core

To create a chip used to support a set of opcodes in the VM, we start with types that implement the appropriate adapter and core traits. We then create `VmAirWrapper` and `VmChipWrapper` types:

```rust
pub struct VmAirWrapper<A, C> {
    pub adapter: A,
    pub core: C,
}

pub struct VmChipWrapper<F, FILLER> {
    pub inner: FILLER,
    pub mem_helper: SharedMemoryHelper<F>,
}
```

They implement the following traits:

- `Air<AB>`, `BaseAir<F>`, and `BaseAirWithPublicValues<F>` are implemented on `VmAirWrapper<A, C>`, where the `eval()` function implements constraints via:
  - calls `eval()` on `C::Air`
  - calls `eval()` on `A::Air`

- `TraceFiller<F>` is implemented on the inner filler, where `fill_trace()` iterates through all records from instruction execution and generates one row of the trace from each record. Rows which do not correspond to an instruction execution are left as **identically zero**. Each used row in the trace is created by calling `fill_trace_row()` with the memory helper and row slice.

- The `VmChipWrapper` provides a blanket implementation of `Chip<RA, CpuBackend<SC>>` for any struct that implements `TraceFiller<Val<SC>>`. The wrapper handles trace generation by:
  1. Instantiating a trace matrix by consuming the record arena
  2. Calling `fill_trace()` on the inner filler to populate the matrix
  3. Generating public values via `generate_public_values()`

**Convention:** If you have a new `Foo` functionality you want to support, create structs `FooExecutor`, `FooFiller`, and `FooCoreAir`. Either use existing adapter components or make your own. Then typedef:

```rust
pub type FooChip<F> = VmChipWrapper<F, FooFiller<F>>;
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
    pub imm: T,
}
```
