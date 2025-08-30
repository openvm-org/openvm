# VM Extensions

The OpenVM architecture is designed for maximal composability and modularity through the VM extension framework. The `arch` module in the `openvm-circuit` crate provides the traits to build VM extensions and configure a complete VM from a collection of VM extensions.

The architecture centers on [`VmConfig`](#vmconfig), [`VmBuilder`](#vmbuilder), and the [VM extension framework](#vm-extension-framework). While `VmConfig` provides hardware-agnostic definitions for VM execution and circuit key generation, `VmBuilder` specializes these configurations to optimize performance for specific prover backends with hardware acceleration.

## VM Extension Framework
The VM extension framework provides a modular way for developers to extend the functionality of a working zkVM. A full VM extension consists of three components:
- [VmExecutionExtension](#vmexecutionextension) for extending the runtime execution handling of new instructions in custom extensions.
- [VmCircuitExtension](#vmcircuitextension) for extending the zkVM circuit with additional AIRs.
- [VmProverExtension](#vmproverextension) extending how trace generation for the additional AIRs specified by the VM circuit extension for different prover backends.

These three components are implemented via three corresponding traits `VmExecutionExtension`, `VmCircuitExtension`, and `VmProverExtension`.

### `VmExecutionExtension`

```rust
pub trait VmExecutionExtension<F> {
    /// Enum of executor variants
    type Executor: AnyEnum;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, Self::Executor>,
    ) -> Result<(), ExecutorInventoryError>;
}
```

The `VmExecutionExtension` provides a way to specify hooks for handling new instructions.
The associated type `Executor` should be an enum of all types implementing the traits
`Executor<F> + MeteredExecutor<F> + PreflightExecutor<F, RA>` for the different [execution modes](./vm.md#execution-modes) for all new instructions introduced by this VM extension. The `Executor` enum does not need to handle instructions outside of this extension. The VM execution extension is specified by registering these hooks using the `ExecutorInventoryBuilder` [API](https://docs.openvm.dev/docs/openvm/openvm_circuit/arch/struct.ExecutorInventoryBuilder.html). The main APIs are
- `inventory.add_executor(executor, opcodes)` to associate an executor with a set of opcodes.
- `inventory.add_phantom_sub_executor(sub_executor, discriminant)` to associate a phantom sub-executor with a phantom discriminant.

### `VmCircuitExtension`
```rust
pub trait VmCircuitExtension<SC: StarkGenericConfig> {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError>;
}
```
The `VmCircuitExtension` trait is the most security critical, and it should have **no** dependencies on the other two extension traits. The `VmCircuitExtension` trait is the only trait that needs to be implemented to specify the AIRs, and consequently their verifying keys, that will be added by this VM extension. The `VmCircuitExtension` should be agnostic to execution implementation details and to differences in prover backends.
The VM circuit extension is specified by adding new AIRs in order using the `AirInventory` [API](https://docs.openvm.dev/docs/openvm/openvm_circuit/arch/struct.AirInventory.html). The main APIs are
- `inventory.add_air(air)` to add a new `air`, where `air` must implement the traits
```rust
Air<AB> + BaseAirWithPublicValues<Val<SC>> + PartitionedBaseAir<Val<SC>> for AB: InteractionBuilder<F = Val<SC>>
```
(in other words, `air` is an AIR with interactions).
- `inventory.find_air::<ConcreteAir>()` returns an iterator of all preceding AIRs in the circuit which downcast to type `ConcreteAir: 'static`.

The added AIRs may have dependencies on previously added AIRs, including those that may have been added by a previous VM extension. In these cases, the `inventory.find_air()` method should be used to retrieve the dependencies.

### `VmProverExtension`
```rust
pub trait VmProverExtension<E, RA, EXT>
where
    E: StarkEngine,
    EXT: VmExecutionExtension<Val<E::SC>> + VmCircuitExtension<E::SC>,
{
    fn extend_prover(
        &self,
        extension: &EXT,
        inventory: &mut ChipInventory<E::SC, RA, E::PB>,
    ) -> Result<(), ChipInventoryError>;
}
```

The `VmProverExtension` trait is the most customizable, and hence (unfortunately) has the most generics.
The generics are `E` for [StarkEngine](https://docs.openvm.dev/docs/openvm/openvm_stark_backend/engine/trait.StarkEngine.html), `RA` for record arena, and `EXT` for execution and circuit extension. Note that the `StarkEngine` trait itself has associated types `SC: StarkGenericConfig` and `PB: ProverBackend`.
The `VmProverExtension` trait is therefore generic over the `ProverBackend` and the trait is designed to allow for different implementations of the prover extension for _the same_ execution and circuit extension `EXT` targeting different prover backends.

Since there are intended to be multiple `VmProverExtension`s for the same `EXT`, the `VmProverExtension` trait is meant to be implemented on a separate struct from `EXT` to get around Rust orphan rules. This separate struct is usually a [zero sized type](https://doc.rust-lang.org/nomicon/exotic-sizes.html#zero-sized-types-zsts) (ZST).

The VM prover extension is specified by adding new chips in order using the `ChipInventory` [API](https://docs.openvm.dev/docs/openvm/openvm_circuit/arch/struct.ChipInventory.html). The main functions are:
- `inventory.add_executor_chip(chip)` adds a chip with an associated executor. Each executor must have exactly one chip associated to it, and this is currently used to determine the record arenas that the executor writes into during preflight execution. It is **required** that the executor chips are adds in the same order as the executors were added in the `VmExecutionExtension` implementation.
- `inventory.add_periphery_chip(chip)` adds a chip without an associated executor. Not every chip needs to have a corresponding executor.
- `inventory.find_chip<ConcreteChip>()` returns an iterator of all preceding chips in the inventory, including those from other previous extensions, which downcast to type `ConcreteChip: 'static`. This may be used to obtain previously constructed data, configurations, or buffers.
- `inventory.next_air::<ConcreteAir>()` returns `Ok(&air)` if the next AIR that was added in the `VmCircuitExtension` implementation is of type `ConcreteAir` and returns error otherwise. It is used to ensure that the associated AIR to each chip is the expected one. It can also be used to obtain configuration data or bus information from the corresponding AIR.

It is **required** that the overall insertion order of the chips (both executor and periphery types) must exactly match the order of the AIRs added in the `VmCircuitExtension` implementation. There should be a 1-to-1 correspondence between AIRs and chips, and implementers should maintain a convention to call `inventory.next_air::<ConcreteAir>()` before adding each chip to clearly indicate the AIR associated with each chip.

## `VmConfig`

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
A VM configuration, represented by a struct implementing the `VmConfig` trait is the minimum serializable format to be able to create the execution
environment and circuit for a zkVM supporting a fixed set of instructions.
This trait contains the sub-traits `VmExecutionConfig` and `VmCircuitConfig`.
The `InitFileGenerator` sub-trait provides custom [build hooks](#build-hooks) to generate code for initializing some VM extensions. The `VmConfig` is expected to contain the `SystemConfig` internally.

This trait does not contain the [`VmBuilder`](#vmbuilder) trait, because a single VM configuration may
implement multiple `VmBuilder`s for different prover backends.

```rust
pub trait VmExecutionConfig<F> {
    type Executor: AnyEnum + Send + Sync;

    fn create_executors(&self)
        -> Result<ExecutorInventory<Self::Executor>, ExecutorInventoryError>;
}
```
The `VmExecutionConfig` defines the collection of `VmExecutionExtension`s that together define the VM's runtime execution environment. The implementation should use the `ExecutorInventory` [API](https://docs.openvm.dev/docs/openvm/openvm_circuit/arch/struct.ExecutorInventory.html) to define the collection of executors and the mapping from opcodes to executors. The associate type `Executor` is expected to be an enum of all executor types necessary to handle all instructions in the VM's instruction set.

Users typically should not need to implement the `VmExecutionConfig` trait directly and should instead use the [derive macro](#derive-macro).

```rust
pub trait VmCircuitConfig<SC: StarkGenericConfig> {
    fn create_airs(&self) -> Result<AirInventory<SC>, AirInventoryError>;
}
```
The `VmCircuitConfig` is the only trait necessary to generate proving and verifying keys for the zkVM circuit. The implementation should use the `AirInventory` [API](https://docs.openvm.dev/docs/openvm/openvm_circuit/arch/struct.AirInventory.html) to define the ordered collection of AIRs that make up the zkVM circuit. **Note** that the order that the AIRs are added to `AirInventory` is **not** the order they appear in the circuit's verifying key. The ordering of AIRs corresponding to the verifying key is given by the [`AirInventory::into_airs`](https://docs.openvm.dev/docs/openvm/openvm_circuit/arch/struct.AirInventory.html#method.into_airs) function: the ordering consists of the system AIRs, followed by the other AIRs in the **reverse** of the order they were added into `AirInventory`.

Users should typically not need to implement the `VmCircuitConfig` trait directly and should instead use the [derive macro](#derive-macro).

### Derive Macro
Developers are typically not expected to implement `VmConfig`, `VmExecutionConfig`, `VmCircuitConfig` directly. Instead, we provide a procedural macro `#[derive(VmConfig)]` that will automatically implement `VmConfig` on a struct that composes an existing `VmConfig` with additional VM extensions:

```rust
#[derive(VmConfig)]
pub struct Rv32IConfig {
    #[config(executor = "SystemExecutor<F>")]
    pub system: SystemConfig,
    #[extension]
    pub base: Rv32I,
    #[extension]
    pub io: Rv32Io,
}

#[derive(VmConfig)]
pub struct Rv32ImConfig {
    #[config]
    pub rv32i: Rv32IConfig,
    #[extension]
    pub mul: Rv32M,
}
```

The struct deriving `VmConfig` should have fields which are given the attribute `#[config]` or `#[extension]`. Exactly one field should have the attribute `#[config]` and its type should implement `VmConfig`. The other fields should have the attribute `#[extension]` and their types should implement `VmExecutionExtension<F>` and `VmCircuitExtension<SC>`. Each field has associated type `Executor`: the macro by default assumes the executor type name is `{FieldTypeName}Executor` without any type generics. A different executor type name can be specified using the `executor` attribute.

The macro will create a new enum named `{ConfigTypeName}Executor` with variants equal to the associated `Executor` types of each attributed field.

The macro derives `VmExecutionConfig<F>` with associated type `Executor = {ConfigTypeName}Executor` on the new config struct for all `F` where the `#[config]` field implements `VmExecutionConfig<F>` and the `#[extension]` fields all implement `VmExecutionExtension<F>`. The derived `create_executors` function adds executors in the order of the fields, first calling `create_executors` on the inner config and then calling `extend_execution` on each `#[extension]` field.

The macro derives `VmCircuitConfig<SC>` on the new config struct for all `SC` where the `#[config]` field implements `VmCircuitConfig<SC>` and the `#[extension]` fields all implement `VmCircuitExtension<SC>`. The derived `create_airs` function adds AIRs in the order of the fields, first calling `create_airs` on the inner config and then calling `extend_circuit` on each `#[extension]` field.

### Build hooks
Some of our extensions need to generate some code at build-time depending on the VM config (for example, the Algebra extension needs to call `moduli_init!` with the appropriate moduli).
To accommodate this, we support build hooks in both `cargo openvm` and the SDK.
To make use of this functionality, implement the `InitFileGenerator` trait.
The `String` returned by the `generate_init_file_contents` must be valid Rust code.
It will be written to a `openvm_init.rs` file in the package's manifest directory, and then (unhygenically) included in the guest code in place of the `openvm::init!` macro.
You can specify a custom file name at build time (by a `cargo openvm` option or an SDK method argument), in which case you must also pass it to `openvm::init!` as an argument.

## `VmBuilder`

The [`VmConfig`](#vmconfig) is independent of the prover backend and prover hardware acceleration options. The `VmBuilder` trait provides a modular way to provide different prover implementations for the same `VmConfig`. (These implementations may even be done in separate crates!)

```rust
pub trait VmBuilder<E: StarkEngine>: Sized {
    type VmConfig: VmConfig<E::SC>;
    type RecordArena: Arena;
    type SystemChipInventory: SystemChipComplex<Self::RecordArena, E::PB>;

    /// Create a [VmChipComplex] from the full [AirInventory], which should be the output of
    /// [VmCircuitConfig::create_airs].
    #[allow(clippy::type_complexity)]
    fn create_chip_complex(
        &self,
        config: &Self::VmConfig,
        airs: AirInventory<E::SC>,
    ) -> Result<
        VmChipComplex<E::SC, Self::RecordArena, E::PB, Self::SystemChipInventory>,
        ChipInventoryError,
    >;
}
```
The `VmBuilder` trait is meant to be implemented on a zero-sized type (ZST). It has an associated type for the `VmConfig`. The `VmBuilder<E>` is generic in `E: StarkEngine`, where the `StarkEngine` trait itself has associated types `SC: StarkGenericConfig` and `PB: ProverBackend`. The `StarkEngine` trait controls the backend implementation of the proof system for a specific `ProverBackend` with specialized hardware acceleration. For a given `StarkEngine`, the `VmBuilder` trait has an associated type for the `RecordArena`, which is the type of in-memory buffer to use to store records during [preflight execution](./vm.md#preflight-execution). Lastly there is an associated type for `SystemChipInventory` which implements the trace generation for the system chips. There are currently two existing choices of `SystemChipInventory` to use: [`SystemChipInventory`](https://docs.openvm.dev/docs/openvm/openvm_circuit/system/struct.SystemChipInventory.html) for CPU and [`SystemChipInventoryGPU`](../../crates/vm/src/system/cuda/mod.rs) for Nvidia GPU.

The `VmBuilder::create_chip_complex` function assumes that it is called after all AIRs have been constructed using the `VmCircuitConfig` trait on the `VmConfig`. In other words, `airs: AirInventory<E::SC>` may be assumed to be the output of `VmCircuitConfig::create_airs()`.

The implementation of `VmBuilder` should implement `create_chip_complex` by first constructing a `VmChipComplex` from a base `VmConfig` such as the `SystemConfig`. It should then mutate the `ChipInventory` contained inside `VmChipComplex` by calling `VmProverExtension::extend_prover` on the relevant prover extensions.

Currently there is no macro to derive the `VmBuilder` trait implementation, and we refer to the [examples](#examples) as a reference.

## Examples

The [`extensions/`](../../extensions/) folder contains extensions implementing all non-system functionality via custom extensions. For example, the `Rv32I`, `Rv32M`, and `Rv32Io` extensions implement `VmExecutionExtension<F>` and `VmCircuitExtension<SC>` in [`openvm-rv32im-circuit`](../../extensions/rv32im/circuit/src/extension/mod.rs) and correspond to the RISC-V 32-bit base and multiplication instruction sets and an extension for IO, respectively. The ZST `Rv32ImCpuProverExt` [implements](../../extensions/rv32im/circuit/src/extension/mod.rs) `VmProverExtension<E, RA, EXT>` for `EXT = Rv32I, Rv32M, Rv32Io`. When the `"cuda"` feature is enabled, the ZST `Rv32ImGpuProverExt` [implements](../../extensions/rv32im/circuit/src/extension/cuda.rs) `VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, EXT>` for `EXT = Rv32I, Rv32M, Rv32Io`.

The `openvm-rv32im-circuit` [crate](../../extensions/rv32im/circuit/src/lib.rs) also provides definitions for `Rv32ImConfig`, `Rv32ImCpuBuilder`, and `Rv32ImGpuBuilder`.
