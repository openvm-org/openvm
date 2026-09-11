# VM Architecture and Chips

## Execution

OpenVM provides a modular interface to add VM instructions via the extension API. The
`VmExecutionExtension` trait allows one to specify various execution extensions. An extension
consists of executor structs that handle specific instruction opcodes. The same `Executor`
transition runs in pure and preflight contexts, while `MeteredExecutor` supplies segmentation
metadata.

We define an **instruction** to be an **opcode** combined with the **operands** for the opcode. Each opcode must be mapped to a specific executor that contains the logic for executing the instruction.
There is a `struct VmOpcode(usize)` to protect the global opcode `usize`, which must be globally unique for each opcode supported in a given VM.

### Execution Modes

#### Pure Execution

Pure execution runs the program without any overhead and is used to obtain the final VM state at termination, or after executing a fixed number of instructions.

The `InterpreterExecutor<F>` trait defines the interface for pure execution (aliased as `Executor<F>` via a supertrait):

```rust
pub trait InterpreterExecutor<F> {
    fn pre_compute_size(&self) -> usize;

    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait;
}
```

where `ExecuteFunc<Ctx>` is a function pointer that contains the instruction execution logic.

```rust
pub type ExecuteFunc<CTX> =
    unsafe fn(pre_compute: *const u8, exec_state: &mut VmExecState<GuestMemory, CTX>);
```

Each executor pre-computes instruction-specific data during a preprocessing step and returns function pointers for direct instruction execution.

#### Metered Execution

Metered execution tracks the trace heights for each chip along with normal execution. This mode divides the execution into segments, where each segment consists of an instruction range and an (over)estimate of the resulting trace heights for each chip in the segment. Segmentation is done based on configurable limits like maximum trace height, maximum trace cells etc.

The `InterpreterMeteredExecutor<F>` trait defines the interface for metered execution (aliased as `MeteredExecutor<F>` via a supertrait):

```rust
pub trait InterpreterMeteredExecutor<F> {
    fn metered_pre_compute_size(&self) -> usize;

    fn metered_pre_compute<Ctx>(
        &self,
        air_idx: usize,
        pc: u32,
        inst: &Instruction,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait;
}
```

Each executor is associated with a chip and an AIR. This mapping is defined implicitly by the VM extension. The additional `air_idx` parameter is the index of the executor's AIR in the verifying key. This is used for indexing the trace height of the chip in the `trace_heights` array contained in the `Segment` struct.

#### Preflight Execution

Preflight uses the same opcode executors as pure and metered execution. Its
execution context maintains read/write guest memory while appending two generic,
chip-independent logs:

- a program log containing `(timestamp, pc)` for every retired instruction and a
  final sentinel;
- a memory log containing each timed block access and its value, plus the
  first-write values required to reconstruct memory chronology.

The logs form `PreflightHistory`. `Postflight` validates that history, derives
memory predecessors, and partitions program steps by opcode. CPU and GPU trace
generators replay their opcode steps from this immutable history. Trace generation
therefore does not mutate VM memory and can process rows in parallel.

### Interpreter Architecture

The `InterpretedInstance` represents the VM interpreter and handles pure and metered execution modes. More specifically, it:

- Pre-computes instruction-specific data buffers
- Generates function pointer tables for direct execution
- Supports optional tail call optimization (TCO) for improved performance

The interpreter uses the same precomputed instruction handlers for pure, metered,
and preflight execution; the execution context selects which state and metadata
each handler maintains.

### Chips for Opcode Groups

Opcodes are partitioned into groups for AIR and trace-generation purposes, but
execution is opcode-owned rather than chip-owned. An executor implements an
opcode's state transition. An AIR constrains one or more opcodes, and a prover
extension registers a backend-specific function that replays the relevant
postflight steps into that AIR's trace.

The AIR `A` should have the following trait bounds:

```rust
A: Air<AB> + BaseAir<F> + BaseAirWithPublicValues<F>
```

where `AB` is an `AirBuilder`

Together, these provide the following functionalities:

- **Keygen:** Performed via the `Air::<AB>::eval()` function.
- **Trace Generation:** Serial preflight produces `PreflightHistory`; postflight
  validates and indexes it; backend-specific trace generators replay the
  resulting immutable steps.

### VM AIR Integration

At the AIR-level, for an AIR to integrate with the OpenVM architecture (constrain memory, read the instruction from the program, etc.), the AIR communicates over different (virtual) buses. There are three main system buses: the memory bus, program bus, and the
execution bus. The memory bus is used to access memory, the program bus is used to read instructions from the program,
and the execution bus is used to constrain the execution flow. These buses are derivable from the `SystemPort` struct,
which is provided by `AirInventory`/`SystemAirInventory`.

The program and execution buses use the program counter index `pc_idx = pc / DEFAULT_PC_STEP`. A sequential AIR
transition advances `pc_idx` by one.

Memory-bus addresses contain an address space and a block index. Each block contains
`BLOCK_FE_WIDTH` cells. In the RV64 register and memory address spaces, this is four u16 cells, or
eight guest bytes. The default RV64 configuration supports 32-bit guest addresses; adapters reject
accesses outside the configured pointer bound.

The buses have very low-level APIs and are not intended to be used directly. "Bridges" are provided to provide a cleaner interface for
sending interactions over the buses and enforcing additional constraints for soundness. The two system bridges are
`MemoryBridge` and `ExecutionBridge`, which should respectively be used to constrain memory accesses and execution flow.

### Phantom Sub-Instructions

Phantom sub-instructions are instructions that affect the runtime and trace matrix values but have no AIR constraints besides advancing `pc_idx` by one. They should not mutate memory, but they can mutate the input & hint streams.

You can specify phantom sub-instruction executors by implementing the trait:

```rust
pub trait PhantomSubExecutor: Send + Sync {
    fn phantom_execute(
        &self,
        memory: &GuestMemory,
        streams: &mut Streams,
        rng: &mut StdRng,
        discriminant: PhantomDiscriminant,
        a: u32,
        b: u32,
        c_upper: u16,
    ) -> eyre::Result<()>;
}

pub struct PhantomDiscriminant(pub u16);
```

The `PhantomExecutor` internally maintains a mapping from `PhantomDiscriminant` to `Arc<dyn PhantomSubExecutor>` to
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
    SC: StarkProtocolConfig,
{
}
```

A `VmConfig` should implement the `VmExecutionConfig` trait which provides execution configuration. The `Executor` type is typically an enum over executor structs that handle instruction execution.

```rust
pub trait VmExecutionConfig<F> {
    type Executor: AnyEnum;

    fn create_executors(&self)
        -> Result<ExecutorInventory<Self::Executor>, ExecutorInventoryError>;
}
```

Finally, `VmConfig` should also implement the `VmCircuitConfig` trait which provides the AIRs for all chips in the VM. The `AirInventory` contains all AIRs required for constraining the execution trace of each chip.

```rust
pub trait VmCircuitConfig<SC: StarkProtocolConfig> {
    fn create_airs(&self) -> Result<AirInventory<SC>, AirInventoryError>;
}
```

See [VM Extensions](./vm-extensions.md) for more details.

### ZK Operations for the VM

#### Keygen

Key generation is computed from the `VmConfig` describing the VM. The `VmConfig` is used to create the `AirInventory` via the `VmCircuitConfig` trait,
which in turn provides the list of AIRs that are used in the proving and verification process.

```rust
pub trait VmCircuitConfig<SC: StarkProtocolConfig> {
    fn create_airs(&self) -> Result<AirInventory<SC>, AirInventoryError>;
}
```

The collected `AirInventory` can be converted into AIRs with `into_airs()`, which `VirtualMachine::new_with_keygen` passes to `MultiStarkKeygenBuilder` to generate the proving and verifying keys.

#### Trace Generation

Trace generation uses the immutable history generated by preflight execution and proceeds from:

> `VirtualMachine::generate_proving_ctx()`

which derives read-only postflight indexes and generates the final trace matrices.

For execution with multiple segments (continuations), the trace generation process is handled by `VmInstance` and proceeds as follows:

1. **Metered Execution**: First run metered execution to determine segment boundaries using `execute_metered()` which returns a list of `Segment` structs containing:
   ```rust
   pub struct Segment {
       pub instret_start: u64,
       pub num_insns: u64,
       pub num_preflight_replay_values: u32,
       pub trace_heights: Vec<u32>,
   }
   ```

2. **Segment Trace Generation**: For each segment:
   - Recover the starting VM state at the beginning of the segment via pure execution from the program start (only necessary in a distributed setup)
   - Run preflight execution from the segment's starting state using the exact metered `Segment` bound
   - Derive postflight chronology and opcode indexes from the generic execution history
   - Generate the system and extension traces by replaying the indexed history
   - Pass final state as initial state to next segment (only necessary in a local setup when proving is done on a single machine)

This approach keeps segment execution independent of chip-specific witness layouts and enables
distributed proving where each segment can be proven independently after recovering its starting
state.

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

    /// `Air` constraints owned by the adapter.
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

    /// Return the `from_pc_idx` expression.
    fn get_from_pc_idx(&self, local: &[AB::Var]) -> AB::Var;
}

pub trait VmCoreAir<AB, I>: BaseAirWithPublicValues<AB::F>
where
    AB: AirBuilder,
    I: VmAdapterInterface<AB::Expr>,
{
    /// Returns `(to_pc_idx, interface)`.
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        from_pc_idx: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I>;

    /// The offset the opcodes by this chip start from.
    /// This is usually just `CorrespondingOpcode::CLASS_OFFSET`,
    /// but sometimes (for modular chips, for example) it also depends on something else.
    fn start_offset(&self) -> usize;

    fn start_offset_expr(&self) -> AB::Expr {
        AB::Expr::from_usize(self.start_offset())
    }

    fn expr_to_global_expr(&self, local_expr: impl Into<AB::Expr>) -> AB::Expr {
        self.start_offset_expr() + local_expr.into()
    }

    fn opcode_to_global_expr(&self, local_opcode: impl LocalOpcode) -> AB::Expr {
        self.expr_to_global_expr(AB::Expr::from_usize(local_opcode.local_usize()))
    }
}

pub struct AdapterAirContext<T, I: VmAdapterInterface<T>> {
    /// Leave as `None` to allow the adapter to decide the `to_pc_idx` automatically.
    pub to_pc_idx: Option<T>,
    pub reads: I::Reads,
    pub writes: I::Writes,
    pub instruction: I::ProcessedInstruction,
}
```

> [!WARNING]
> You do not need to implement `Air` on the struct you implement `VmAdapterAir` or `VmCoreAir` on.

### Execution and Trace Generation

Execution and trace generation are deliberately separate:

- `Executor<F>` provides the opcode state transition used by pure and preflight execution.
- `MeteredExecutor<F>` adds the AIR-index metadata needed for segmentation.
- A prover extension registers backend-specific postflight generators. Each generator replays the
  opcode steps assigned to its AIR from immutable preflight history.

The AIR adapter/core split remains a constraint-system abstraction; it does not own execution.

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

- `TraceFiller<F>` is implemented on the inner filler. Rows which do not correspond to an
  instruction execution are left as **identically zero**. Each used row is created by calling
  `fill_trace_row()` with the memory helper and row slice.

- CPU postflight generators use `VmChipWrapper` fillers to:
  1. Allocate the trace matrix at the derived row count
  2. Replay the relevant immutable execution-history rows into the matrix
  3. Generate public values via `generate_public_values()`

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
>(PhantomData<T>, PhantomData<PI>);

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
    pub immediate: T,
}
```
