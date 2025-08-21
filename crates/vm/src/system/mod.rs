use std::sync::Arc;

use derive_more::derive::From;
use openvm_circuit_derive::{AnyEnum, Executor, MeteredExecutor, PreflightExecutor};
use openvm_circuit_primitives::var_range::{
    SharedVariableRangeCheckerChip, VariableRangeCheckerAir, VariableRangeCheckerBus,
    VariableRangeCheckerChip,
};
use openvm_instructions::{
    LocalOpcode, PhantomDiscriminant, PublishOpcode, SysPhantom, SystemOpcode,
};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    engine::StarkEngine,
    interaction::{LookupBus, PermutationCheckBus},
    p3_field::{Field, PrimeField32},
    prover::{
        cpu::{CpuBackend, CpuDevice},
        hal::{MatrixDimensions, ProverBackend},
        types::{AirProvingContext, CommittedTraceData},
    },
    AirRef, Chip,
};
use rustc_hash::FxHashMap;

use self::{connector::VmConnectorAir, program::ProgramAir, public_values::PublicValuesAir};
use crate::{
    arch::{
        vm_poseidon2_config, AirInventory, AirInventoryError, BusIndexManager, ChipInventory,
        ChipInventoryError, DenseRecordArena, ExecutionBridge, ExecutionBus, ExecutionState,
        ExecutorInventory, ExecutorInventoryError, MatrixRecordArena, PhantomSubExecutor,
        RowMajorMatrixArena, SystemConfig, VmAirWrapper, VmBuilder, VmChipComplex, VmChipWrapper,
        VmCircuitConfig, VmExecutionConfig, CONNECTOR_AIR_ID, PROGRAM_AIR_ID, PUBLIC_VALUES_AIR_ID,
    },
    system::{
        connector::VmConnectorChip,
        memory::{
            interface::MemoryInterfaceAirs,
            offline_checker::{MemoryBridge, MemoryBus},
            online::GuestMemory,
            MemoryAirInventory, MemoryController, TimestampedEquipartition, CHUNK,
        },
        native_adapter::{NativeAdapterAir, NativeAdapterExecutor},
        phantom::{
            CycleEndPhantomExecutor, CycleStartPhantomExecutor, NopPhantomExecutor, PhantomAir,
            PhantomChip, PhantomExecutor, PhantomFiller,
        },
        poseidon2::{
            air::Poseidon2PeripheryAir, new_poseidon2_periphery_air, Poseidon2PeripheryChip,
        },
        program::{ProgramBus, ProgramChip},
        public_values::{
            PublicValuesChip, PublicValuesCoreAir, PublicValuesExecutor, PublicValuesFiller,
        },
    },
};

pub mod connector;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod memory;
// Necessary for the PublicValuesChip
pub mod native_adapter;
pub mod phantom;
pub mod poseidon2;
pub mod program;
pub mod public_values;

/// **If** internal poseidon2 chip exists, then its insertion index is 1.
const POSEIDON2_INSERTION_IDX: usize = 1;
/// **If** public values chip exists, then its executor index is 0.
pub(crate) const PV_EXECUTOR_IDX: usize = 0;

/// Trait for trace generation of all system AIRs. The system chip complex is special because we may
/// not exactly following the exact matching between `Air` and `Chip`. Moreover we may require more
/// flexibility than what is provided through the trait object [`AnyChip`].
///
/// The [SystemChipComplex] is meant to be constructible once the VM configuration is known, and it
/// can be loaded with arbitrary programs supported by the instruction set available to its
/// configuration. The [SystemChipComplex] is meant to persistent between instances of proof
/// generation.
pub trait SystemChipComplex<RA, PB: ProverBackend> {
    /// Loads the program in the form of a cached trace with prover data.
    fn load_program(&mut self, cached_program_trace: CommittedTraceData<PB>);

    /// Transport the initial memory state to device. This may be called before preflight execution
    /// begins and start async device processes in parallel to execution.
    fn transport_init_memory_to_device(&mut self, memory: &GuestMemory);

    /// The caller must guarantee that `record_arenas` has length equal to the number of system
    /// AIRs, although some arenas may be empty if they are unused.
    fn generate_proving_ctx(
        &mut self,
        system_records: SystemRecords<PB::Val>,
        record_arenas: Vec<RA>,
    ) -> Vec<AirProvingContext<PB>>;

    /// This function is only used for metric collection purposes and custom implementations are
    /// free to ignore it.
    ///
    /// Since system chips (primarily memory) will only have all information needed to compute the
    /// true used trace heights after `generate_proving_ctx` is called, this method will be called
    /// after `generate_proving_ctx` on the trace `heights` of all AIRs (including non-system AIRs)
    /// in the AIR ID order.
    ///
    /// The default implementation does nothing.
    #[cfg(feature = "metrics")]
    fn finalize_trace_heights(&self, _heights: &mut [usize]) {}
}

/// Trait meant to be implemented on a SystemChipComplex.
pub trait SystemWithFixedTraceHeights {
    /// `heights` will have length equal to number of system AIRs, in AIR ID order. This function
    /// must guarantee that the system trace matrices generated have the required heights.
    fn override_trace_heights(&mut self, heights: &[u32]);
}

pub struct SystemRecords<F> {
    pub from_state: ExecutionState<u32>,
    pub to_state: ExecutionState<u32>,
    pub exit_code: Option<u32>,
    /// `i` -> frequency of instruction in `i`th row of trace matrix. This requires filtering
    /// `program.instructions_and_debug_infos` to remove gaps.
    pub filtered_exec_frequencies: Vec<u32>,
    // We always use a [DenseRecordArena] here, regardless of the generic `RA` used for other
    // execution records.
    pub access_adapter_records: DenseRecordArena,
    // Perf[jpw]: this should be computed on-device and changed to just touched blocks
    pub touched_memory: TouchedMemory<F>,
    /// The public values of the [PublicValuesChip]. These should only be non-empty if
    /// continuations are disabled.
    pub public_values: Vec<F>,
}

pub enum TouchedMemory<F> {
    Persistent(TimestampedEquipartition<F, CHUNK>),
    Volatile(TimestampedEquipartition<F, 1>),
}

#[derive(Clone, AnyEnum, Executor, MeteredExecutor, PreflightExecutor, From)]
pub enum SystemExecutor<F: Field> {
    PublicValues(PublicValuesExecutor<F>),
    Phantom(PhantomExecutor<F>),
}

/// SystemPort combines system resources needed by most extensions
#[derive(Clone, Copy)]
pub struct SystemPort {
    pub execution_bus: ExecutionBus,
    pub program_bus: ProgramBus,
    pub memory_bridge: MemoryBridge,
}

#[derive(Clone)]
pub struct SystemAirInventory<SC: StarkGenericConfig> {
    pub program: ProgramAir,
    pub connector: VmConnectorAir,
    pub memory: MemoryAirInventory<SC>,
    /// Public values AIR exists if and only if continuations is disabled and `num_public_values`
    /// is greater than 0.
    pub public_values: Option<PublicValuesAir>,
}

impl<SC: StarkGenericConfig> SystemAirInventory<SC> {
    pub fn new(
        config: &SystemConfig,
        port: SystemPort,
        merkle_compression_buses: Option<(PermutationCheckBus, PermutationCheckBus)>,
    ) -> Self {
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = port;
        let range_bus = memory_bridge.range_bus();
        let program = ProgramAir::new(program_bus);
        let connector = VmConnectorAir::new(
            execution_bus,
            program_bus,
            range_bus,
            config.memory_config.timestamp_max_bits,
        );
        assert_eq!(
            config.continuation_enabled,
            merkle_compression_buses.is_some()
        );

        let memory = MemoryAirInventory::new(
            memory_bridge,
            &config.memory_config,
            range_bus,
            merkle_compression_buses,
        );

        let public_values = if config.has_public_values_chip() {
            let air = VmAirWrapper::new(
                NativeAdapterAir::new(
                    ExecutionBridge::new(execution_bus, program_bus),
                    memory_bridge,
                ),
                PublicValuesCoreAir::new(
                    config.num_public_values,
                    config.max_constraint_degree as u32 - 1,
                ),
            );
            Some(air)
        } else {
            None
        };

        Self {
            program,
            connector,
            memory,
            public_values,
        }
    }

    pub fn port(&self) -> SystemPort {
        SystemPort {
            memory_bridge: self.memory.bridge,
            program_bus: self.program.bus,
            execution_bus: self.connector.execution_bus,
        }
    }

    pub fn into_airs(self) -> Vec<AirRef<SC>> {
        let mut airs: Vec<AirRef<SC>> = Vec::new();
        airs.push(Arc::new(self.program));
        airs.push(Arc::new(self.connector));
        if let Some(public_values) = self.public_values {
            airs.push(Arc::new(public_values));
        }
        airs.extend(self.memory.into_airs());
        airs
    }
}

impl<F: PrimeField32> VmExecutionConfig<F> for SystemConfig {
    type Executor = SystemExecutor<F>;

    /// The only way to create an [ExecutorInventory] is from a [SystemConfig]. This will add an
    /// executor for [PublicValuesExecutor] if continuations is disabled. It will always add an
    /// executor for [PhantomChip], which handles all phantom sub-executors.
    fn create_executors(
        &self,
    ) -> Result<ExecutorInventory<Self::Executor>, ExecutorInventoryError> {
        let mut inventory = ExecutorInventory::new(self.clone());
        // PublicValuesChip is required when num_public_values > 0 in single segment mode.
        if self.has_public_values_chip() {
            assert_eq!(inventory.executors().len(), PV_EXECUTOR_IDX);

            let public_values = PublicValuesExecutor::new(NativeAdapterExecutor::default());
            inventory.add_executor(public_values, [PublishOpcode::PUBLISH.global_opcode()])?;
        }
        let phantom_opcode = SystemOpcode::PHANTOM.global_opcode();
        let mut phantom_executors: FxHashMap<PhantomDiscriminant, Arc<dyn PhantomSubExecutor<F>>> =
            FxHashMap::default();
        // Use NopPhantomExecutor so the discriminant is set but `DebugPanic` is handled specially.
        phantom_executors.insert(
            PhantomDiscriminant(SysPhantom::DebugPanic as u16),
            Arc::new(NopPhantomExecutor),
        );
        phantom_executors.insert(
            PhantomDiscriminant(SysPhantom::Nop as u16),
            Arc::new(NopPhantomExecutor),
        );
        phantom_executors.insert(
            PhantomDiscriminant(SysPhantom::CtStart as u16),
            Arc::new(CycleStartPhantomExecutor),
        );
        phantom_executors.insert(
            PhantomDiscriminant(SysPhantom::CtEnd as u16),
            Arc::new(CycleEndPhantomExecutor),
        );
        let phantom = PhantomExecutor::new(phantom_executors, phantom_opcode);
        inventory.add_executor(phantom, [phantom_opcode])?;

        Ok(inventory)
    }
}

impl<SC: StarkGenericConfig> VmCircuitConfig<SC> for SystemConfig {
    /// Every VM circuit within the OpenVM circuit architecture **must** be initialized from the
    /// [SystemConfig].
    fn create_airs(&self) -> Result<AirInventory<SC>, AirInventoryError> {
        let mut bus_idx_mgr = BusIndexManager::new();
        let execution_bus = ExecutionBus::new(bus_idx_mgr.new_bus_idx());
        let memory_bus = MemoryBus::new(bus_idx_mgr.new_bus_idx());
        let program_bus = ProgramBus::new(bus_idx_mgr.new_bus_idx());
        let range_bus =
            VariableRangeCheckerBus::new(bus_idx_mgr.new_bus_idx(), self.memory_config.decomp);

        let merkle_compression_buses = if self.continuation_enabled {
            let merkle_bus = PermutationCheckBus::new(bus_idx_mgr.new_bus_idx());
            let compression_bus = PermutationCheckBus::new(bus_idx_mgr.new_bus_idx());
            Some((merkle_bus, compression_bus))
        } else {
            None
        };
        let memory_bridge =
            MemoryBridge::new(memory_bus, self.memory_config.timestamp_max_bits, range_bus);
        let system_port = SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        };
        let system = SystemAirInventory::new(self, system_port, merkle_compression_buses);

        let mut inventory = AirInventory::new(self.clone(), system, bus_idx_mgr);

        let range_checker = VariableRangeCheckerAir::new(range_bus);
        // Range checker is always the first AIR in the inventory
        inventory.add_air(range_checker);

        if self.continuation_enabled {
            assert_eq!(inventory.ext_airs().len(), POSEIDON2_INSERTION_IDX);
            // Add direct poseidon2 AIR for persistent memory.
            // Currently we never use poseidon2 opcodes when continuations is enabled: we will need
            // special handling when that happens
            let (_, compression_bus) = merkle_compression_buses.unwrap();
            let direct_bus_idx = compression_bus.index;
            let air = new_poseidon2_periphery_air(
                vm_poseidon2_config(),
                LookupBus::new(direct_bus_idx),
                self.max_constraint_degree,
            );
            inventory.add_air_ref(air);
        }
        let execution_bridge = ExecutionBridge::new(execution_bus, program_bus);
        let phantom = PhantomAir {
            execution_bridge,
            phantom_opcode: SystemOpcode::PHANTOM.global_opcode(),
        };
        inventory.add_air(phantom);

        Ok(inventory)
    }
}

// =================== CPU Backend Specific System Chip Complex Constructor ==================

/// Base system chips for CPU backend. These chips must exactly correspond to the AIRs in
/// [SystemAirInventory].
pub struct SystemChipInventory<SC: StarkGenericConfig> {
    pub program_chip: ProgramChip<SC>,
    pub connector_chip: VmConnectorChip<Val<SC>>,
    /// Contains all memory chips
    pub memory_controller: MemoryController<Val<SC>>,
    pub public_values_chip: Option<PublicValuesChip<Val<SC>>>,
}

// Note[jpw]: We could get rid of the `mem_inventory` input because `MemoryController` doesn't need
// the buses for tracegen. We leave it to use old interfaces.
impl<SC: StarkGenericConfig> SystemChipInventory<SC>
where
    Val<SC>: PrimeField32,
{
    pub fn new(
        config: &SystemConfig,
        mem_inventory: &MemoryAirInventory<SC>,
        range_checker: SharedVariableRangeCheckerChip,
        hasher_chip: Option<Arc<Poseidon2PeripheryChip<Val<SC>>>>,
    ) -> Self {
        // We create an empty program chip: the program should be loaded later (and can be swapped
        // out). The execution frequencies are supplied only after execution.
        let program_chip = ProgramChip::unloaded();
        let connector_chip = VmConnectorChip::<Val<SC>>::new(
            range_checker.clone(),
            config.memory_config.timestamp_max_bits,
        );
        let memory_bus = mem_inventory.bridge.memory_bus();
        let memory_controller = match &mem_inventory.interface {
            MemoryInterfaceAirs::Persistent {
                boundary: _,
                merkle,
            } => {
                assert!(config.continuation_enabled);
                MemoryController::<Val<SC>>::with_persistent_memory(
                    memory_bus,
                    config.memory_config.clone(),
                    range_checker.clone(),
                    merkle.merkle_bus,
                    merkle.compression_bus,
                    hasher_chip.unwrap(),
                )
            }
            MemoryInterfaceAirs::Volatile { boundary: _ } => {
                assert!(!config.continuation_enabled);
                MemoryController::with_volatile_memory(
                    memory_bus,
                    config.memory_config.clone(),
                    range_checker.clone(),
                )
            }
        };

        let public_values_chip = config.has_public_values_chip().then(|| {
            VmChipWrapper::new(
                PublicValuesFiller::new(
                    NativeAdapterExecutor::default(),
                    config.num_public_values,
                    (config.max_constraint_degree as u32)
                        .checked_sub(1)
                        .unwrap(),
                ),
                memory_controller.helper(),
            )
        });

        Self {
            program_chip,
            connector_chip,
            memory_controller,
            public_values_chip,
        }
    }
}

impl<RA, SC> SystemChipComplex<RA, CpuBackend<SC>> for SystemChipInventory<SC>
where
    RA: RowMajorMatrixArena<Val<SC>>,
    SC: StarkGenericConfig,
    Val<SC>: PrimeField32,
{
    fn load_program(&mut self, cached_program_trace: CommittedTraceData<CpuBackend<SC>>) {
        let _ = self.program_chip.cached.replace(cached_program_trace);
    }

    fn transport_init_memory_to_device(&mut self, memory: &GuestMemory) {
        self.memory_controller
            .set_initial_memory(memory.memory.clone());
    }

    fn generate_proving_ctx(
        &mut self,
        system_records: SystemRecords<Val<SC>>,
        mut record_arenas: Vec<RA>,
    ) -> Vec<AirProvingContext<CpuBackend<SC>>> {
        let SystemRecords {
            from_state,
            to_state,
            exit_code,
            filtered_exec_frequencies,
            access_adapter_records,
            touched_memory,
            public_values,
        } = system_records;

        if let Some(chip) = &mut self.public_values_chip {
            chip.inner.set_public_values(public_values);
        }
        self.program_chip.filtered_exec_frequencies = filtered_exec_frequencies;
        let program_ctx = self.program_chip.generate_proving_ctx(());
        self.connector_chip.begin(from_state);
        self.connector_chip.end(to_state, exit_code);
        let connector_ctx = self.connector_chip.generate_proving_ctx(());

        let pv_ctx = self.public_values_chip.as_ref().map(|chip| {
            let arena = record_arenas.remove(PUBLIC_VALUES_AIR_ID);
            chip.generate_proving_ctx(arena)
        });

        let memory_ctxs = self
            .memory_controller
            .generate_proving_ctx(access_adapter_records, touched_memory);

        [program_ctx, connector_ctx]
            .into_iter()
            .chain(pv_ctx)
            .chain(memory_ctxs)
            .collect()
    }

    #[cfg(feature = "metrics")]
    fn finalize_trace_heights(&self, heights: &mut [usize]) {
        use openvm_stark_backend::ChipUsageGetter;

        use crate::system::memory::interface::MemoryInterface;

        let boundary_idx = PUBLIC_VALUES_AIR_ID + usize::from(self.public_values_chip.is_some());
        let mut access_adapter_offset = boundary_idx + 1;
        match &self.memory_controller.interface_chip {
            MemoryInterface::Volatile { boundary_chip } => {
                let boundary_height = boundary_chip
                    .final_memory
                    .as_ref()
                    .map(|m| m.len())
                    .unwrap_or(0);
                heights[boundary_idx] = boundary_height;
            }
            MemoryInterface::Persistent {
                boundary_chip,
                merkle_chip,
                ..
            } => {
                let boundary_height = 2 * boundary_chip.touched_labels.len();
                heights[boundary_idx] = boundary_height;
                heights[boundary_idx + 1] = merkle_chip.current_height;
                access_adapter_offset += 1;

                // Poseidon2Periphery height also varies based on memory, so set it now even though
                // it's not a system chip:
                let poseidon_chip = self.memory_controller.hasher_chip.as_ref().unwrap();
                let poseidon_height = poseidon_chip.current_trace_height();
                // We know the chip insertion index, which starts from *the end* of the the AIR
                // ordering
                let poseidon_idx = heights.len() - 1 - POSEIDON2_INSERTION_IDX;
                heights[poseidon_idx] = poseidon_height;
            }
        }
        let access_heights = &self
            .memory_controller
            .access_adapter_inventory
            .trace_heights;
        heights[access_adapter_offset..access_adapter_offset + access_heights.len()]
            .copy_from_slice(access_heights);
    }
}

#[derive(Clone)]
pub struct SystemCpuBuilder;

impl<SC, E> VmBuilder<E> for SystemCpuBuilder
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Val<SC>: PrimeField32,
{
    type VmConfig = SystemConfig;
    type RecordArena = MatrixRecordArena<Val<SC>>;
    type SystemChipInventory = SystemChipInventory<SC>;

    fn create_chip_complex(
        &self,
        config: &SystemConfig,
        airs: AirInventory<SC>,
    ) -> Result<
        VmChipComplex<SC, MatrixRecordArena<Val<SC>>, CpuBackend<SC>, SystemChipInventory<SC>>,
        ChipInventoryError,
    > {
        let range_bus = airs.range_checker().bus;
        let range_checker = Arc::new(VariableRangeCheckerChip::new(range_bus));

        let mut inventory = ChipInventory::new(airs);
        // PublicValuesChip is required when num_public_values > 0 in single segment mode.
        if config.has_public_values_chip() {
            assert_eq!(
                inventory.executor_idx_to_insertion_idx.len(),
                PV_EXECUTOR_IDX
            );
            // We set insertion_idx so that air_idx = num_airs - (insertion_idx + 1) =
            // PUBLIC_VALUES_AIR_ID in `VmChipComplex::executor_idx_to_air_idx`. We need to do this
            // because this chip is special and not part of the normal inventory.
            let insertion_idx = inventory
                .airs()
                .num_airs()
                .checked_sub(1 + PUBLIC_VALUES_AIR_ID)
                .unwrap();
            inventory.executor_idx_to_insertion_idx.push(insertion_idx);
        }
        inventory.next_air::<VariableRangeCheckerAir>()?;
        inventory.add_periphery_chip(range_checker.clone());

        let hasher_chip = if config.continuation_enabled {
            assert_eq!(inventory.chips().len(), POSEIDON2_INSERTION_IDX);
            // ATTENTION: The threshold 7 here must match the one in `new_poseidon2_periphery_air`
            let direct_bus = if config.max_constraint_degree >= 7 {
                inventory
                    .next_air::<Poseidon2PeripheryAir<Val<SC>, 0>>()?
                    .bus
            } else {
                inventory
                    .next_air::<Poseidon2PeripheryAir<Val<SC>, 1>>()?
                    .bus
            };
            let chip = Arc::new(Poseidon2PeripheryChip::new(
                vm_poseidon2_config(),
                direct_bus.index,
                config.max_constraint_degree,
            ));
            inventory.add_periphery_chip(chip.clone());
            Some(chip)
        } else {
            None
        };
        let system = SystemChipInventory::new(
            config,
            &inventory.airs().system().memory,
            range_checker,
            hasher_chip,
        );

        let phantom_chip = PhantomChip::new(PhantomFiller, system.memory_controller.helper());
        inventory.add_executor_chip(phantom_chip);

        Ok(VmChipComplex { system, inventory })
    }
}

impl<SC: StarkGenericConfig> SystemWithFixedTraceHeights for SystemChipInventory<SC>
where
    Val<SC>: PrimeField32,
{
    /// Warning: this does not set the override for the PublicValuesChip. The PublicValuesChip
    /// override must be set via the RecordArena.
    fn override_trace_heights(&mut self, heights: &[u32]) {
        assert_eq!(
            heights[PROGRAM_AIR_ID] as usize,
            self.program_chip
                .cached
                .as_ref()
                .expect("program not loaded")
                .trace
                .height()
        );
        assert_eq!(heights[CONNECTOR_AIR_ID], 2);
        let mut memory_start_idx = PUBLIC_VALUES_AIR_ID;
        if self.public_values_chip.is_some() {
            memory_start_idx += 1;
        }
        self.memory_controller
            .set_override_trace_heights(&heights[memory_start_idx..]);
    }
}
