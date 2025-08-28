# VM Extensions

The OpenVM architecture is designed for maximal composability and modularity through the VM extension framework. The `arch` module in the `openvm-circuit` crate provides the traits to build a VM extension.

## VM Extension Framework
The VM extension framework provides a modular way for developers to extend the functionality of a working zkVM. A full VM extension consists of three components:
- [VmExecutionExtension](#vmexecutionextension) for extending the runtime execution handling of new instructions in custom instruction set extensions.
- [VmCircuitExtension](#vmcircuitextension) extending the zkVM circuit with additional AIRs.
- [VmProverExtension](#vmproverextension) extending how trace generation for the additional AIRs specified by the VM circuit extension for different prover backends.

This three components are implemented via three corresponding traits `VmExecutionExtension`, `VmCircuitExtension`, and `VmProverExtension`.

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
`Executor<F> + MeteredExecutor<F> + PreflightExecutor<F, RA>` for the different [execution modes](./vm.md#execution-modes) for all new instructions introduced by this VM extension. The `Executor` enum does not need to handle instructions outside of this extension. The VM execution extension is specified by registering these hooks using the `ExecutorInventoryBuilder` [API](https://docs.openvm.dev/docs/openvm/openvm_circuit/arch/struct.ExecutorInventoryBuilder.html#implementations). The main APIs are
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



## Composing extensions into a VM: `VmConfig`

Once you have multiple extensions, how do you compose them into a VM?

We have trait `VmConfig`:

```rust
pub trait VmConfig<F: PrimeField32> {
    type Executor: PreflightExecutor<F> + AnyEnum + ChipUsageGetter;
    type Periphery: AnyEnum + ChipUsageGetter;

    /// Must contain system config
    fn system(&self) -> &SystemConfig;

    fn create_chip_complex(
        &self,
    ) -> Result<VmChipComplex<F, Self::Executor, Self::Periphery>, VmInventoryError>;
}
```

A `VmConfig` is a struct that is a `SystemConfig` together with a collection of extensions. From the config we should be able to **deterministically** use `create_chip_complex` to create `VmChipComplex`. The `VmConfig` macro will
automatically implement `VmConfig` using the `#[system]` and `#[extension]` attributes:

```rust
#[derive(VmConfig)]
struct MyVmConfig {
    #[system]
    system: SystemConfig,
    #[extension]
    ext1: Ext1,
    #[extension]
    ext2: Ext2
}
```

where `Ext1, Ext2` must implement `VmExtension<F>` for any `F: PrimeField32` (trait bounds can be added later).

The macro will also make two big enums: one that is an enum of the `Ext*::Executor` enums and another for the `Ext*::Periphery` enums.

The macro will then generate a `create_chip_complex` function.

For that we need to understand what `VmChipComplex` consists of:

- System chips
- `VmInventory`
  and all the methods to generate AIR proof inputs.

The macro will generate the `VmChipComplex` iteratively using the

```rust
    pub fn extend<E3, P3, Ext>(
        mut self,
        config: &Ext,
    ) -> Result<VmChipComplex<F, E3, P3>, VmInventoryError>
    where
        Ext: VmExtension<F>,
        E: Into<E3> + AnyEnum,
        P: Into<P3> + AnyEnum,
        Ext::Executor: Into<E3>,
        Ext::Periphery: Into<P3>,
```

function. What this does in words:

- Start with system chips only.
- Generate `VmInventory` for first extension, and append them to the system chip complex.
- Generate `VmInventory` for second extension, and append them to previous chip complex.

For each extension's inventory generation, the `VmInventoryBuilder` is provided with a view of all current chips already inside the running chip complex. This means the inventory generation process is sequential in the order the extensions are specified, and each extension has borrow access to all chips constructed by any extension before it.

## Build hooks
Some of our extensions need to generate some code at build-time depending on the VM config (for example, the Algebra extension needs to call `moduli_init!` with the appropriate moduli).
To accommodate this, we support build hooks in both `cargo openvm` and the SDK.
To make use of this functionality, implement the `InitFileGenerator` trait.
The `String` returned by the `generate_init_file_contents` must be valid Rust code.
It will be written to a `openvm_init.rs` file in the package's manifest directory, and then (unhygenically) included in the guest code in place of the `openvm::init!` macro.
You can specify a custom file name at build time (by a `cargo openvm` option or an SDK method argument), in which case you must also pass it to `openvm::init!` as an argument.

## Examples

The [`extensions/`](../../extensions/) folder contains extensions implementing all non-system functionality via custom extensions. For example, the `Rv32I`, `Rv32M`, and `Rv32Io` extensions implement `VmExtension<F>` in [`openvm-rv32im-circuit`](../../extensions/rv32im/circuit/) and correspond to the RISC-V 32-bit base and multiplication instruction sets and an extension for IO, respectively.

# Design Choices

Why enums and not `dyn`?

- Flexibility: when you have a concrete enum type, it is easier to introduce new traits later that the enum type could implement, whereas `dyn Trait` fully limits the functionality to the `Trait`
- Currently `Chip<SC>` is not object safe so `dyn` is not an option. Overall object safety is not always easy to guarantee.
- `dyn` has a runtime lookup which has a very marginal performance impact. This is likely not the limiting factor, so it is secondary concern.
- The opcode lookup in `VmInventory` requires more smart pointers if you use `dyn`, see below.

`VmInventory` gets rid of `Rc<RefCell<_>>` on most chips.

- We were using it just for the instruction opcode lookup even when we didn't need a shared mutable reference -- the exception is `MemoryController`, where we really do need the shared reference, and where we keep the `RefCell`.
- The internals of `VmInventory` now store all chips exactly once, and opcode lookups are true lookups by index. This should have a very small runtime improvement.
