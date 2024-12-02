# VM Extensions

```rust
pub trait VmExtension<F: PrimeField32> {
    type Executor: InstructionExecutor<F> + AnyEnum;
    type Periphery: AnyEnum;

    fn build(
        &self,
        builder: &mut VmInventoryBuilder<F>,
    ) -> Result<VmInventory<Self::Executor, Self::Periphery>, VmInventoryError>;
}
```

The `VmExtensionTrait` is a way to specify how to construct a collection of chips and all assign opcodes to be handled by them. This data is collected into a `VmInventory` struct, which is returned.

To handle previous chip dependencies necessary for chip construction and also automatic bus index management, we provide a `VmInventoryBuilder` api.

Due to strong types, we have **two** associated trait types `Executor, Periphery`. It is expected that `Executor` is an enum of all types implementing `InstructionExecutor + Chip` that this extension will construct. It is expected that `Periphery` is an enum of all types that implement `Chip` **but are not InstructionExecutor**. In general, it is always OK for the enum to have more kinds than necessary. For easy downcasting and enum wrangling, we also have an `AnyEnum` trait, which can always be derived by a macro.

### `VmInventory<Executor, Periphery>`

Think of `VmInventory<Executor, Periphery>` as the collection of all chips, which can be either `Executor` or `Periphery`. It also has a lookup from `AxVmOpcode` to `Executor` which is how runtime execution knows how to route instructions to executors.

`VmInventory` API relevant for `VmExtension`:

```rust
    pub fn add_executor(
        &mut self,
        executor: impl Into<Executor>,
        opcodes: impl IntoIterator<Item = AxVmOpcode>,
    ) -> Result<(), VmInventoryError>;

    pub fn add_periphery_chip(&mut self, periphery_chip: impl Into<Periphery>);

    pub fn add_phantom_sub_executor<F: 'static, PE: PhantomSubExecutor<F> + 'static>(
        &mut self,
        phantom_sub: PE,
        discriminant: PhantomDiscriminant,
    ) -> Result<(), VmInventoryError>;

    pub fn executors(&self) -> &[Executor] {
        &self.executors
    }

    pub fn periphery(&self) -> &[Periphery] {
        &self.periphery
    }
```

where you should specify all opcodes owned by an executor when you add it.

For runtime execution in a segment, the `VmInventory` also provides the getter functions:

```rust
    pub fn get_executor(&self, opcode: AxVmOpcode) -> Option<&Executor>;

    pub fn get_mut_executor(&mut self, opcode: &AxVmOpcode) -> Option<&mut Executor>;
```

### `VmInventoryBuilder`

Here is the API of `VmInventoryBuilder`:

```rust
impl<'a, F: PrimeField32> VmInventoryBuilder<'a, F> {
    pub fn memory_controller(&self) -> &MemoryControllerRef<F>;
    pub fn system_base(&self) -> &SystemBase<F>;
    pub fn new_bus_idx(&mut self) -> usize;
    pub fn find_chip<C: 'static>(&self) -> Vec<&C>;
    /// Shareable streams. Clone to get a shared mutable reference.
    pub fn streams(&self) -> &Arc<Mutex<Streams<F>>>;
    pub fn add_phantom_sub_executor<PE: PhantomSubExecutor<F> + 'static>(
        &self,
        phantom_sub: PE,
        discriminant: PhantomDiscriminant,
    ) -> Result<(), VmInventoryError>;
}
```

You can find the base system chips in `system_base`. If you need to generate a new bus, use `new_bus_idx`. If you want to check if a chip already exists inside of the global VM config _and not just your extension_ use `find_chip` to search for chip by type name. It will return a list of references to the chips. If you need to hold a shared reference, then the expectation is that `C = Arc<_>`.

Something the api is lacking: if you want to _change_ a previous chip (such as range tuple checker's constructor parameters) after it has been constructed, that is not currently possible. The current solution is that all those global parameters should be in the VM config (below) and you configure them in the config's constructor.

## Composing extensions into a VM: `VmConfig`

Once you have multiple extensions, how do you compose them into a VM?

We have trait `VmConfig`:

```rust
pub trait VmConfig<F: PrimeField32> {
    type Executor: InstructionExecutor<F> + AnyEnum + ChipUsageGetter;
    type Periphery: AnyEnum + ChipUsageGetter;

    /// Must contain system config
    fn system(&self) -> &SystemConfig;

    fn create_chip_complex(
        &self,
    ) -> Result<VmChipComplex<F, Self::Executor, Self::Periphery>, VmInventoryError>;
}
```

A `VmConfig` is a struct that is a `SystemConfig` together with a collection of extensions. From the config we should be able to **deterministically** use `create_chip_complex` to create `VmChipComplex`.

We will have a macro so that `VmConfig` is auto-implemented
when you do: **macro implementation in progress, do it manually for now**

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

For that we need to understand what `VmChipComplex` is: it replaces the role of the previous `VmChipSet` and consists of:

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

### `VirtualMachine`

The top level structs of `VirtualMachine`, `VmExecutor`, `SegmentExecutor` remain almost entirely the same, but now has `VmConfig` as a generic:

```rust
pub struct VirtualMachine<SC: StarkGenericConfig, E, VC> {
```

We refer to the code on usage, which should be straightforward.

## Examples

`Rv32I, Rv32M` implement `VmExtension<F>` [here](https://github.com/axiom-crypto/afs-prototype/blob/main/vm/src/extensions/rv32im.rs) corresponding to the RISC-V 32-bit base, multiplication instruction sets, respectively. There is a manual implementation of `Rv32IConfig, Rv32ImConfig: VmConfig<F>` that will be replaced by a macro. The implementation is currently in `vm/src/extensions` but will be moved into a separate crate once stable.

Toolchain tests using the new constructions [here](https://github.com/axiom-crypto/afs-prototype/blob/2cc5c3b28e3fd6d8a01c1edce76f8da3aaffbafe/toolchain/tests/src/basic_tests.rs#L21).

# Design Choices

Why enums and not `dyn`?

- Flexibility: when you have a concrete enum type, it is easier to introduce new traits later that the enum type could implement, whereas `dyn Trait` fully limits the functionality to the `Trait`
- Currently `Chip<SC>` is not object safe so `dyn` is not an option. Overall object safety is not always easy to guarantee.
- `dyn` has a runtime lookup which has a very marginal performance impact. This is likely not the limiting factor, so it is secondary concern.
- The opcode lookup in `VmInventory` requires more smart pointers if you use `dyn`, see below.

`VmInventory` gets rid of `Rc<RefCell<_>>` on most chips.

- We were using it just for the instruction opcode lookup even when we didn't need a shared mutable reference -- the exception is `MemoryController`, where we really do need the shared reference, and where we keep the `RefCell`.
- The internals of `VmInventory` now store all chips exactly once, and opcode lookups are true lookups by index. This should have a very small runtime improvement.
