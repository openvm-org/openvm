use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, HashMap},
    iter,
    ops::Range,
    rc::Rc,
    sync::Arc,
};

use ax_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, BitwiseOperationLookupChip},
    range_tuple::{RangeTupleCheckerBus, RangeTupleCheckerChip},
    var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip},
};
// use ax_ecc_primitives::field_expression::ExprBuilderConfig;
use ax_stark_backend::{
    config::{Domain, StarkGenericConfig},
    p3_commit::PolynomialSpace,
    prover::types::{AirProofInput, CommittedTraceData, ProofInput},
    rap::AnyRap,
    Chip, ChipUsageGetter,
};
use axvm_instructions::{program::Program, *};
use p3_baby_bear::BabyBear;
use p3_field::PrimeField32;
use p3_matrix::Matrix;
use parking_lot::Mutex;
use strum::EnumCount;

use super::{vm_poseidon2_config, Streams};
use crate::{
    arch::{AxVmChip, AxVmExecutor, ExecutionBus, ExecutorName, VmConfig},
    system::{
        connector::VmConnectorChip,
        memory::{
            merkle::{DirectCompressionBus, MemoryMerkleBus},
            offline_checker::MemoryBus,
            Equipartition, MemoryController, MemoryControllerRef, BOUNDARY_AIR_OFFSET, CHUNK,
        },
        native_adapter::NativeAdapterChip,
        phantom::PhantomChip,
        poseidon2::Poseidon2Chip,
        program::{ProgramBus, ProgramChip},
        public_values::{core::PublicValuesCoreChip, PublicValuesChip},
    },
};

pub const EXECUTION_BUS: usize = 0;
pub const MEMORY_BUS: usize = 1;
const RANGE_CHECKER_BUS: usize = 4;
pub const POSEIDON2_DIRECT_BUS: usize = 6;
pub const READ_INSTRUCTION_BUS: usize = 8;
pub const BITWISE_OP_LOOKUP_BUS: usize = 9;
pub const BYTE_XOR_BUS: usize = 10;
//pub const BYTE_XOR_BUS: XorBus = XorBus(8);
pub const RANGE_TUPLE_CHECKER_BUS: usize = 11;
pub const MEMORY_MERKLE_BUS: usize = 12;

use super::{CONNECTOR_AIR_ID, PROGRAM_AIR_ID, PUBLIC_VALUES_AIR_ID};

pub struct VmChipSet<F: PrimeField32> {
    pub executors: HashMap<usize, AxVmExecutor<F>>,

    // ATTENTION: chip destruction should follow the following field order:
    pub program_chip: ProgramChip<F>,
    pub connector_chip: VmConnectorChip<F>,
    /// PublicValuesChip is disabled when num_public_values == 0.
    pub public_values_chip: Option<Rc<RefCell<PublicValuesChip<F>>>>,
    pub chips: Vec<AxVmChip<F>>,
    pub overridden_executor_heights: Option<BTreeMap<ExecutorName, usize>>,
    pub memory_controller: MemoryControllerRef<F>,
    pub range_checker_chip: Arc<VariableRangeCheckerChip>,
}

impl<F: PrimeField32> VmChipSet<F> {
    pub(crate) fn set_program(&mut self, program: Program<F>) {
        self.program_chip.set_program(program);
    }
    pub(crate) fn set_streams(&mut self, streams: Arc<Mutex<Streams<F>>>) {
        for chip in self.chips.iter_mut() {
            if let AxVmChip::Executor(AxVmExecutor::Phantom(chip)) = chip {
                chip.borrow_mut().set_streams(streams.clone());
            }
        }
    }

    /// Returns the AIR ID of the given executor if it exists.
    pub fn get_executor_air_id(&self, executor: ExecutorName) -> Option<usize> {
        self.executor_to_air_id_mapping().get(&executor).copied()
    }
    /// Return mapping from executor name to AIR ID.
    pub fn executor_to_air_id_mapping(&self) -> BTreeMap<ExecutorName, usize> {
        let mut air_id = PUBLIC_VALUES_AIR_ID;
        if self.public_values_chip.is_some() {
            air_id += 1;
        }
        air_id += self.memory_controller.borrow().air_names().len();
        self.chips
            .iter()
            .flat_map(|chip| {
                let ret = if let AxVmChip::Executor(chip) = chip {
                    let name: ExecutorName = chip.into();
                    Some((name, air_id))
                } else {
                    None
                };
                air_id += 1;
                ret
            })
            .collect()
    }

    /// Return IDs of AIRs which heights won't during execution.
    pub(crate) fn const_height_air_ids(&self) -> Vec<usize> {
        let mut ret = vec![PROGRAM_AIR_ID, CONNECTOR_AIR_ID];
        let num_const_chip = self
            .chips
            .iter()
            .filter(|chip| !matches!(chip, AxVmChip::Executor(_)))
            .count();
        let num_air = self.num_airs();
        // Const chips are always in the end.
        // +1 is for RangeChecker.
        ret.extend((num_air - (num_const_chip + 1))..num_air);
        ret
    }
    /// Return the number of AIRs in the chip set.
    /// Careful: this costs more than O(1) due to bad implementation.
    pub(crate) fn num_airs(&self) -> usize {
        self.air_names().len()
    }
    /// Return air names of all chips in order.
    pub(crate) fn air_names(&self) -> Vec<String> {
        iter::once(self.program_chip.air_name())
            .chain([self.connector_chip.air_name()])
            .chain(self.public_values_chip.as_ref().map(|c| c.air_name()))
            .chain(self.memory_controller.borrow().air_names())
            .chain(self.chips.iter().map(|c| c.air_name()))
            .chain([self.range_checker_chip.air_name()])
            .collect()
    }
    /// Return trace heights of all chips in order.
    pub(crate) fn current_trace_heights(&self) -> Vec<usize> {
        iter::once(self.program_chip.current_trace_height())
            .chain([self.connector_chip.current_trace_height()])
            .chain(
                self.public_values_chip
                    .as_ref()
                    .map(|c| c.current_trace_height()),
            )
            .chain(self.memory_controller.borrow().current_trace_heights())
            .chain(self.chips.iter().map(|c| c.current_trace_height()))
            .chain([self.range_checker_chip.current_trace_height()])
            .collect()
    }
    /// Return trace cells of all chips in order.
    pub(crate) fn current_trace_cells(&self) -> Vec<usize> {
        iter::once(self.program_chip.current_trace_cells())
            .chain([self.connector_chip.current_trace_cells()])
            .chain(
                self.public_values_chip
                    .as_ref()
                    .map(|c| c.current_trace_cells()),
            )
            .chain(self.memory_controller.borrow().current_trace_cells())
            .chain(self.chips.iter().map(|c| c.current_trace_cells()))
            .chain([self.range_checker_chip.current_trace_cells()])
            .collect()
    }
    pub(crate) fn airs<SC: StarkGenericConfig>(&self) -> Vec<Arc<dyn AnyRap<SC>>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        // ATTENTION: The order of AIR MUST be consistent with `generate_proof_input`.
        let program_rap = Arc::new(self.program_chip.air) as Arc<dyn AnyRap<SC>>;
        let connector_rap = Arc::new(self.connector_chip.air) as Arc<dyn AnyRap<SC>>;
        [program_rap, connector_rap]
            .into_iter()
            .chain(self.public_values_chip.as_ref().map(|chip| chip.air()))
            .chain(self.memory_controller.borrow().airs())
            .chain(self.chips.iter().map(|chip| chip.air()))
            .chain(iter::once(self.range_checker_chip.air()))
            .collect()
    }

    pub(crate) fn generate_proof_input<SC: StarkGenericConfig>(
        self,
        cached_program: Option<CommittedTraceData<SC>>,
    ) -> ProofInput<SC>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        // ATTENTION: The order of AIR proof input generation MUST be consistent with `airs`.

        // Drop all strong references to chips other than self.chips, which will be consumed next.
        drop(self.executors);

        let mut pi_builder = ChipSetProofInputBuilder::new();
        // System: Program Chip
        debug_assert_eq!(pi_builder.curr_air_id, PROGRAM_AIR_ID);
        pi_builder.add_air_proof_input(self.program_chip.generate_air_proof_input(cached_program));
        // System: Connector Chip
        debug_assert_eq!(pi_builder.curr_air_id, CONNECTOR_AIR_ID);
        pi_builder.add_air_proof_input(self.connector_chip.generate_air_proof_input());
        // Kernel: PublicValues Chip
        if let Some(chip) = self.public_values_chip {
            debug_assert_eq!(pi_builder.curr_air_id, PUBLIC_VALUES_AIR_ID);
            pi_builder.add_air_proof_input(chip.generate_air_proof_input());
        }
        // Non-system chips: ONLY AirProofInput generation to release strong references.
        // Will be added after MemoryController for AIR ordering.
        let non_sys_inputs: Vec<_> =
            self.chips
                .into_iter()
                .map(|chip| {
                    if let AxVmChip::Executor(executor) = chip {
                        let height = self.overridden_executor_heights.as_ref().and_then(
                            |overridden_heights| {
                                let executor_name: ExecutorName = (&executor).into();
                                overridden_heights.get(&executor_name).copied()
                            },
                        );
                        if let Some(height) = height {
                            executor.generate_air_proof_input_with_height(height)
                        } else {
                            executor.generate_air_proof_input()
                        }
                    } else {
                        chip.generate_air_proof_input()
                    }
                })
                .collect();
        // System: Memory Controller
        {
            // memory
            let memory_controller = Rc::try_unwrap(self.memory_controller)
                .expect("other chips still hold a reference to memory chip")
                .into_inner();

            let air_proof_inputs = memory_controller.generate_air_proof_inputs();
            for air_proof_input in air_proof_inputs {
                pi_builder.add_air_proof_input(air_proof_input);
            }
        }
        // Non-system chips
        non_sys_inputs
            .into_iter()
            .for_each(|input| pi_builder.add_air_proof_input(input));
        // System: Range Checker Chip
        pi_builder.add_air_proof_input(self.range_checker_chip.generate_air_proof_input());

        pi_builder.generate_proof_input()
    }
}

impl VmConfig {
    /// Returns the AIR ID of the memory boundary AIR. Panic if the boundary AIR is not enabled.
    pub fn memory_boundary_air_id(&self) -> usize {
        assert!(
            !self.continuation_enabled,
            "Memory boundary AIR is not enabled in continuation mode"
        );
        let mut ret = PUBLIC_VALUES_AIR_ID;
        if self.num_public_values > 0 {
            ret += 1;
        }
        ret += BOUNDARY_AIR_OFFSET;
        ret
    }
    /// Return mapping from executor name to AIR ID.
    pub fn executor_to_air_id_mapping(&self) -> BTreeMap<ExecutorName, usize> {
        self.create_chip_set::<BabyBear>()
            .executor_to_air_id_mapping()
    }
    pub fn create_chip_set<F: PrimeField32>(&self) -> VmChipSet<F> {
        let execution_bus = ExecutionBus(EXECUTION_BUS);
        let program_bus = ProgramBus(READ_INSTRUCTION_BUS);
        let memory_bus = MemoryBus(MEMORY_BUS);
        let merkle_bus = MemoryMerkleBus(MEMORY_MERKLE_BUS);
        let range_bus = VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, self.memory_config.decomp);
        let range_checker = Arc::new(VariableRangeCheckerChip::new(range_bus));
        let bitwise_lookup_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
        let bitwise_lookup_chip = Arc::new(BitwiseOperationLookupChip::new(bitwise_lookup_bus));

        let memory_controller = if self.continuation_enabled {
            Rc::new(RefCell::new(MemoryController::with_persistent_memory(
                memory_bus,
                self.memory_config,
                range_checker.clone(),
                merkle_bus,
                DirectCompressionBus(POSEIDON2_DIRECT_BUS),
                Equipartition::<F, CHUNK>::new(),
                None,
            )))
        } else {
            Rc::new(RefCell::new(MemoryController::with_volatile_memory(
                memory_bus,
                self.memory_config,
                range_checker.clone(),
                None,
            )))
        };
        let program_chip = ProgramChip::new(program_bus);

        let mut executors: HashMap<usize, AxVmExecutor<F>> = HashMap::new();

        // Use BTreeSet to ensure deterministic order.
        // NOTE: The order of entries in `chips` must be a linear extension of the dependency DAG.
        // That is, if chip A holds a strong reference to chip B, then A must precede B in `required_executors`.
        let mut required_executors: BTreeSet<_> = self.executors.clone().into_iter().collect();
        let mut chips = vec![];

        // [(1 << 8), if mul_u256_enabled { 32 } else { 8 } * (1 << 8)],

        let range_tuple_bus =
            RangeTupleCheckerBus::new(RANGE_TUPLE_CHECKER_BUS, [(1 << 8), { 8 } * (1 << 8)]);
        let range_tuple_checker = Arc::new(RangeTupleCheckerChip::new(range_tuple_bus));

        // PublicValuesChip is required when num_public_values > 0 in single segment mode.
        let public_values_chip = if !self.continuation_enabled && self.num_public_values > 0 {
            let (range, offset) = default_executor_range(ExecutorName::PublicValues);
            let chip = Rc::new(RefCell::new(PublicValuesChip::new(
                NativeAdapterChip::new(execution_bus, program_bus, memory_controller.clone()),
                PublicValuesCoreChip::new(self.num_public_values, offset, 2),
                memory_controller.clone(),
            )));
            for opcode in range {
                executors.insert(opcode, chip.clone().into());
            }
            Some(chip)
        } else {
            assert!(
                !required_executors.contains(&ExecutorName::PublicValues),
                "PublicValuesChip should not be used in continuation mode."
            );
            None
        };
        // We always put Poseidon2 chips in the end. So it will be initialized separately.
        let has_poseidon_chip = required_executors.contains(&ExecutorName::Poseidon2);
        if has_poseidon_chip {
            required_executors.remove(&ExecutorName::Poseidon2);
        }
        // We may not use this chip if the memory kind is volatile and there is no executor for Poseidon2.
        let needs_poseidon_chip = has_poseidon_chip || self.continuation_enabled;

        for &executor in required_executors.iter() {
            let (range, offset) = default_executor_range(executor);
            for opcode in range.clone() {
                if executors.contains_key(&opcode) {
                    panic!("Attempting to override an executor for opcode {opcode}");
                }
            }
            match executor {
                ExecutorName::Phantom => {
                    let phantom_chip = PhantomChip::new(
                        execution_bus,
                        program_bus,
                        memory_controller.clone(),
                        offset,
                    );

                    let phantom_chip = Rc::new(RefCell::new(phantom_chip));
                    for opcode in range {
                        executors.insert(opcode, phantom_chip.clone().into());
                    }
                    chips.push(AxVmChip::Executor(phantom_chip.into()));
                }
                ExecutorName::Poseidon2 => {}
                _ => {
                    unreachable!("Unsupported executor")
                }
            }
        }

        if needs_poseidon_chip {
            let (range, offset) = default_executor_range(ExecutorName::Poseidon2);
            let poseidon_chip = Rc::new(RefCell::new(Poseidon2Chip::from_poseidon2_config(
                vm_poseidon2_config(),
                self.poseidon2_max_constraint_degree,
                execution_bus,
                program_bus,
                memory_controller.clone(),
                POSEIDON2_DIRECT_BUS,
                offset,
            )));
            for opcode in range {
                executors.insert(opcode, poseidon_chip.clone().into());
            }
            chips.push(AxVmChip::Executor(poseidon_chip.into()));
        }

        if Arc::strong_count(&bitwise_lookup_chip) > 1 {
            chips.push(AxVmChip::BitwiseOperationLookup(bitwise_lookup_chip));
        }
        if Arc::strong_count(&range_tuple_checker) > 1 {
            chips.push(AxVmChip::RangeTupleChecker(range_tuple_checker));
        }

        let connector_chip = VmConnectorChip::new(execution_bus, program_bus);

        VmChipSet {
            executors,
            program_chip,
            connector_chip,
            public_values_chip,
            chips,
            overridden_executor_heights: self.overridden_executor_heights.clone(),
            memory_controller,
            range_checker_chip: range_checker,
        }
    }
}

fn default_executor_range(executor: ExecutorName) -> (Range<usize>, usize) {
    let (start, len, offset) = match executor {
        // Terminate is not handled by executor, it is done by system (VmConnectorChip)
        ExecutorName::Phantom => (
            SystemOpcode::PHANTOM.with_default_offset(),
            1,
            SystemOpcode::default_offset(),
        ),
        ExecutorName::Poseidon2 => (
            Poseidon2Opcode::default_offset(),
            Poseidon2Opcode::COUNT,
            Poseidon2Opcode::default_offset(),
        ),
        _ => panic!("Not a default executor"),
    };
    (start..(start + len), offset)
}

struct ChipSetProofInputBuilder<SC: StarkGenericConfig> {
    curr_air_id: usize,
    proof_input_per_air: Vec<(usize, AirProofInput<SC>)>,
}

impl<SC: StarkGenericConfig> ChipSetProofInputBuilder<SC> {
    fn new() -> Self {
        Self {
            curr_air_id: 0,
            proof_input_per_air: vec![],
        }
    }
    /// Adds air proof input if one of the main trace matrices is non-empty.
    /// Always increments the internal `curr_air_id` regardless of whether a new air proof input was added or not.
    fn add_air_proof_input(&mut self, air_proof_input: AirProofInput<SC>) {
        let h = if !air_proof_input.raw.cached_mains.is_empty() {
            air_proof_input.raw.cached_mains[0].height()
        } else {
            air_proof_input
                .raw
                .common_main
                .as_ref()
                .map(|trace| trace.height())
                .unwrap()
        };
        if h > 0 {
            self.proof_input_per_air
                .push((self.curr_air_id, air_proof_input));
        }
        self.curr_air_id += 1;
    }

    fn generate_proof_input(self) -> ProofInput<SC> {
        ProofInput {
            per_air: self.proof_input_per_air,
        }
    }
}
