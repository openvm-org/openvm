use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    iter,
    ops::{Range, RangeInclusive},
    rc::Rc,
    sync::Arc,
};

use adapters::Rv32HeapAdapterChip;
use ax_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, BitwiseOperationLookupChip},
    range_tuple::{RangeTupleCheckerBus, RangeTupleCheckerChip},
    var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip},
};
use ax_ecc_primitives::field_expression::ExprBuilderConfig;
use ax_stark_backend::{
    config::{Domain, StarkGenericConfig},
    p3_commit::PolynomialSpace,
    prover::types::{AirProofInput, CommittedTraceData, ProofInput},
    rap::AnyRap,
    Chip, ChipUsageGetter,
};
use axvm_instructions::{program::Program, *};
use itertools::zip_eq;
use num_bigint_dig::BigUint;
use p3_field::PrimeField32;
use p3_matrix::Matrix;
use parking_lot::Mutex;
use program::DEFAULT_PC_STEP;
use strum::EnumCount;

use super::{vm_poseidon2_config, EcCurve, Streams};
use crate::{
    arch::{
        AxVmChip, AxVmInstructionExecutor, ExecutionBus, ExecutorName, PersistenceType, VmConfig,
    },
    intrinsics::{
        ecc::{
            pairing::{
                EcLineMul013By013Chip, EcLineMulBy01234Chip, MillerDoubleAndAddStepChip,
                MillerDoubleStepChip,
            },
            sw::{EcAddNeChip, EcDoubleChip},
        },
        hashes::{keccak::hasher::KeccakVmChip, poseidon2::Poseidon2Chip},
        int256::{
            Rv32BaseAlu256Chip, Rv32LessThan256Chip, Rv32Multiplication256Chip, Rv32Shift256Chip,
        },
        modular::{
            ModularAddSubChip, ModularAddSubCoreChip, ModularMulDivChip, ModularMulDivCoreChip,
        },
    },
    kernels::{
        adapters::{
            branch_native_adapter::BranchNativeAdapterChip, convert_adapter::ConvertAdapterChip,
            jal_native_adapter::JalNativeAdapterChip,
            loadstore_native_adapter::NativeLoadStoreAdapterChip,
            native_adapter::NativeAdapterChip, native_vec_heap_adapter::NativeVecHeapAdapterChip,
            native_vectorized_adapter::NativeVectorizedAdapterChip,
        },
        branch_eq::KernelBranchEqChip,
        castf::{CastFChip, CastFCoreChip},
        field_arithmetic::{FieldArithmeticChip, FieldArithmeticCoreChip},
        field_extension::{FieldExtensionChip, FieldExtensionCoreChip},
        fri::FriFoldChip,
        jal::{JalCoreChip, KernelJalChip},
        loadstore::{KernelLoadStoreChip, KernelLoadStoreCoreChip},
        modular::{KernelModularAddSubChip, KernelModularMulDivChip},
        public_values::{core::PublicValuesCoreChip, PublicValuesChip},
    },
    rv32im::{
        adapters::{
            Rv32BaseAluAdapterChip, Rv32BranchAdapterChip, Rv32CondRdWriteAdapterChip,
            Rv32HintStoreAdapterChip, Rv32JalrAdapterChip, Rv32LoadStoreAdapterChip,
            Rv32MultAdapterChip, Rv32RdWriteAdapterChip, Rv32VecHeapAdapterChip,
        },
        *,
    },
    system::{
        connector::VmConnectorChip,
        memory::{
            merkle::MemoryMerkleBus, offline_checker::MemoryBus, Equipartition, MemoryController,
            MemoryControllerRef, CHUNK, MERKLE_AIR_OFFSET,
        },
        phantom::PhantomChip,
        program::{ProgramBus, ProgramChip},
    },
};

pub const EXECUTION_BUS: usize = 0;
pub const MEMORY_BUS: usize = 1;
pub const RANGE_CHECKER_BUS: usize = 4;
pub const POSEIDON2_DIRECT_BUS: usize = 6;
pub const READ_INSTRUCTION_BUS: usize = 8;
pub const BITWISE_OP_LOOKUP_BUS: usize = 9;
pub const BYTE_XOR_BUS: usize = 10;
//pub const BYTE_XOR_BUS: XorBus = XorBus(8);
pub const RANGE_TUPLE_CHECKER_BUS: usize = 11;
pub const MEMORY_MERKLE_BUS: usize = 12;

pub const PROGRAM_AIR_ID: usize = 0;
/// ProgramAir is the first AIR so its cached trace should be the first main trace.
pub const PROGRAM_CACHED_TRACE_INDEX: usize = 0;
pub const CONNECTOR_AIR_ID: usize = 1;
/// If PublicValuesAir is **enabled**, its AIR ID is 2. PublicValuesAir is always disabled when
/// using persistent memory.
pub const PUBLIC_VALUES_AIR_ID: usize = 2;
/// If VM uses persistent memory, all AIRs of MemoryController are added after ConnectorChip.
/// Merkle AIR commits start/final memory states.
pub const MERKLE_AIR_ID: usize = CONNECTOR_AIR_ID + 1 + MERKLE_AIR_OFFSET;

pub struct VmChipSet<F: PrimeField32> {
    pub executors: BTreeMap<usize, AxVmInstructionExecutor<F>>,

    // ATTENTION: chip destruction should follow the following field order:
    pub program_chip: ProgramChip<F>,
    pub connector_chip: VmConnectorChip<F>,
    /// PublicValuesChip is disabled when num_public_values == 0.
    pub public_values_chip: Option<Rc<RefCell<PublicValuesChip<F>>>>,
    pub chips: Vec<AxVmChip<F>>,
    pub memory_controller: MemoryControllerRef<F>,
    pub range_checker_chip: Arc<VariableRangeCheckerChip>,
}

impl<F: PrimeField32> VmChipSet<F> {
    pub(crate) fn set_program(&mut self, program: Program<F>) {
        self.program_chip.set_program(program);
    }
    pub(crate) fn set_streams(&mut self, streams: Arc<Mutex<Streams<F>>>) {
        for chip in self.chips.iter_mut() {
            match chip {
                AxVmChip::LoadStore(chip) => chip.borrow_mut().core.set_streams(streams.clone()),
                AxVmChip::HintStoreRv32(chip) => {
                    chip.borrow_mut().core.set_streams(streams.clone())
                }
                AxVmChip::Phantom(chip) => chip.borrow_mut().set_streams(streams.clone()),
                _ => {}
            }
        }
    }
    pub(crate) fn current_trace_cells(&self) -> BTreeMap<String, usize> {
        iter::once(get_name_and_cells(&self.program_chip))
            .chain([get_name_and_cells(&self.connector_chip)])
            .chain(self.public_values_chip.as_ref().map(get_name_and_cells))
            .chain(zip_eq(
                self.memory_controller.borrow().air_names(),
                self.memory_controller.borrow().current_trace_cells(),
            ))
            .chain(self.chips.iter().map(get_name_and_cells))
            .chain([get_name_and_cells(&self.range_checker_chip)])
            .collect()
    }
    pub(crate) fn airs<SC: StarkGenericConfig>(&self) -> Vec<Arc<dyn AnyRap<SC>>>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        // ATTENTION: The order of AIR MUST be consistent with `generate_proof_input`.
        let program_rap: Arc<dyn AnyRap<SC>> = Arc::new(self.program_chip.air.clone());
        let connector_rap: Arc<dyn AnyRap<SC>> = Arc::new(self.connector_chip.air.clone());
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
        let non_sys_inputs: Vec<_> = self
            .chips
            .into_iter()
            .map(|chip| chip.generate_air_proof_input())
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
    pub fn create_chip_set<F: PrimeField32>(&self) -> VmChipSet<F> {
        let execution_bus = ExecutionBus(EXECUTION_BUS);
        let program_bus = ProgramBus(READ_INSTRUCTION_BUS);
        let memory_bus = MemoryBus(MEMORY_BUS);
        let merkle_bus = MemoryMerkleBus(MEMORY_MERKLE_BUS);
        let range_bus = VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, self.memory_config.decomp);
        let range_checker = Arc::new(VariableRangeCheckerChip::new(range_bus));
        let bitwise_lookup_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
        let bitwise_lookup_chip = Arc::new(BitwiseOperationLookupChip::new(bitwise_lookup_bus));

        let memory_controller = match self.memory_config.persistence_type {
            PersistenceType::Volatile => {
                Rc::new(RefCell::new(MemoryController::with_volatile_memory(
                    memory_bus,
                    self.memory_config,
                    range_checker.clone(),
                )))
            }
            PersistenceType::Persistent => {
                Rc::new(RefCell::new(MemoryController::with_persistent_memory(
                    memory_bus,
                    self.memory_config,
                    range_checker.clone(),
                    merkle_bus,
                    Equipartition::<F, CHUNK>::new(),
                )))
            }
        };
        let program_chip = ProgramChip::default();

        let mut executors: BTreeMap<usize, AxVmInstructionExecutor<F>> = BTreeMap::new();

        // Use BTreeSet to ensure deterministic order.
        // NOTE: The order of entries in `chips` must be a linear extension of the dependency DAG.
        // That is, if chip A holds a strong reference to chip B, then A must precede B in `required_executors`.
        let mut required_executors: BTreeSet<_> = self.executors.clone().into_iter().collect();
        let mut chips = vec![];

        let mul_u256_enabled = required_executors.contains(&ExecutorName::Multiplication256Rv32);
        let range_tuple_bus = RangeTupleCheckerBus::new(
            RANGE_TUPLE_CHECKER_BUS,
            [(1 << 8), if mul_u256_enabled { 32 } else { 8 } * (1 << 8)],
        );
        let range_tuple_checker = Arc::new(RangeTupleCheckerChip::new(range_tuple_bus));

        // PublicValuesChip is required when num_public_values > 0.
        let public_values_chip = if self.num_public_values > 0 {
            // Raw public values are not supported when continuation is enabled.
            assert!(!self.continuation_enabled());
            let (range, offset) = default_executor_range(ExecutorName::PublicValues);
            let chip = Rc::new(RefCell::new(PublicValuesChip::new(
                NativeAdapterChip::new(execution_bus, program_bus, memory_controller.clone()),
                PublicValuesCoreChip::new(self.num_public_values, offset),
                memory_controller.clone(),
            )));
            for opcode in range {
                executors.insert(opcode, chip.clone().into());
            }
            Some(chip)
        } else {
            required_executors.remove(&ExecutorName::PublicValues);
            None
        };
        // We always put Poseidon2 chips in the end. So it will be initialized separately.
        let has_poseidon_chip = required_executors.contains(&ExecutorName::Poseidon2);
        if has_poseidon_chip {
            required_executors.remove(&ExecutorName::Poseidon2);
        }
        // We may not use this chip if the memory kind is volatile and there is no executor for Poseidon2.
        let needs_poseidon_chip = has_poseidon_chip || self.continuation_enabled();

        for &executor in required_executors.iter() {
            let (range, offset) = default_executor_range(executor);
            for opcode in range.clone() {
                if executors.contains_key(&opcode) {
                    panic!("Attempting to override an executor for opcode {opcode}");
                }
            }
            match executor {
                ExecutorName::Phantom => {
                    let phantom_chip = Rc::new(RefCell::new(PhantomChip::new(
                        execution_bus,
                        program_bus,
                        memory_controller.clone(),
                        offset,
                    )));
                    for opcode in range {
                        executors.insert(opcode, phantom_chip.clone().into());
                    }
                    chips.push(AxVmChip::Phantom(phantom_chip));
                }
                ExecutorName::LoadStore => {
                    let chip = Rc::new(RefCell::new(KernelLoadStoreChip::<F, 1>::new(
                        NativeLoadStoreAdapterChip::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                            offset,
                        ),
                        KernelLoadStoreCoreChip::new(offset),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::LoadStore(chip));
                }
                ExecutorName::BranchEqual => {
                    let chip = Rc::new(RefCell::new(KernelBranchEqChip::new(
                        BranchNativeAdapterChip::<_>::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        BranchEqualCoreChip::new(offset, DEFAULT_PC_STEP),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::BranchEqual(chip));
                }
                ExecutorName::Jal => {
                    let chip = Rc::new(RefCell::new(KernelJalChip::new(
                        JalNativeAdapterChip::<_>::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        JalCoreChip::new(offset),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::Jal(chip));
                }
                ExecutorName::FieldArithmetic => {
                    let chip = Rc::new(RefCell::new(FieldArithmeticChip::new(
                        NativeAdapterChip::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        FieldArithmeticCoreChip::new(offset),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::FieldArithmetic(chip));
                }
                ExecutorName::FieldExtension => {
                    let chip = Rc::new(RefCell::new(FieldExtensionChip::new(
                        NativeVectorizedAdapterChip::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        FieldExtensionCoreChip::new(offset),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::FieldExtension(chip));
                }
                ExecutorName::PublicValues => {}
                ExecutorName::Poseidon2 => {}
                ExecutorName::Keccak256 => {
                    let chip = Rc::new(RefCell::new(KeccakVmChip::new(
                        execution_bus,
                        program_bus,
                        memory_controller.clone(),
                        bitwise_lookup_chip.clone(),
                        offset,
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::Keccak256(chip));
                }
                ExecutorName::FriFold => {
                    let chip = Rc::new(RefCell::new(FriFoldChip::new(
                        memory_controller.clone(),
                        execution_bus,
                        program_bus,
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::FriFold(chip));
                }
                ExecutorName::BaseAluRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32BaseAluChip::new(
                        Rv32BaseAluAdapterChip::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        BaseAluCoreChip::new(bitwise_lookup_chip.clone(), offset),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::BaseAluRv32(chip));
                }
                ExecutorName::LessThanRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32LessThanChip::new(
                        Rv32BaseAluAdapterChip::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        LessThanCoreChip::new(bitwise_lookup_chip.clone(), offset),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::LessThanRv32(chip));
                }
                ExecutorName::MultiplicationRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32MultiplicationChip::new(
                        Rv32MultAdapterChip::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        MultiplicationCoreChip::new(range_tuple_checker.clone(), offset),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::MultiplicationRv32(chip));
                }
                ExecutorName::MultiplicationHighRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32MulHChip::new(
                        Rv32MultAdapterChip::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        MulHCoreChip::new(
                            bitwise_lookup_chip.clone(),
                            range_tuple_checker.clone(),
                            offset,
                        ),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::MultiplicationHighRv32(chip));
                }
                ExecutorName::DivRemRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32DivRemChip::new(
                        Rv32MultAdapterChip::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        DivRemCoreChip::new(
                            bitwise_lookup_chip.clone(),
                            range_tuple_checker.clone(),
                            offset,
                        ),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::DivRemRv32(chip));
                }
                ExecutorName::ShiftRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32ShiftChip::new(
                        Rv32BaseAluAdapterChip::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        ShiftCoreChip::new(
                            bitwise_lookup_chip.clone(),
                            range_checker.clone(),
                            offset,
                        ),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::ShiftRv32(chip));
                }
                ExecutorName::LoadStoreRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32LoadStoreChip::new(
                        Rv32LoadStoreAdapterChip::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                            range_checker.clone(),
                            offset,
                        ),
                        LoadStoreCoreChip::new(offset),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::LoadStoreRv32(chip));
                }
                ExecutorName::LoadSignExtendRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32LoadSignExtendChip::new(
                        Rv32LoadStoreAdapterChip::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                            range_checker.clone(),
                            offset,
                        ),
                        LoadSignExtendCoreChip::new(range_checker.clone(), offset),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::LoadSignExtendRv32(chip));
                }
                ExecutorName::HintStoreRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32HintStoreChip::new(
                        Rv32HintStoreAdapterChip::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                            range_checker.clone(),
                        ),
                        Rv32HintStoreCoreChip::new(bitwise_lookup_chip.clone(), offset),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::HintStoreRv32(chip));
                }
                ExecutorName::BranchEqualRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32BranchEqualChip::new(
                        Rv32BranchAdapterChip::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        BranchEqualCoreChip::new(offset, DEFAULT_PC_STEP),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::BranchEqualRv32(chip));
                }
                ExecutorName::BranchLessThanRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32BranchLessThanChip::new(
                        Rv32BranchAdapterChip::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        BranchLessThanCoreChip::new(bitwise_lookup_chip.clone(), offset),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::BranchLessThanRv32(chip));
                }
                ExecutorName::JalLuiRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32JalLuiChip::new(
                        Rv32CondRdWriteAdapterChip::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        Rv32JalLuiCoreChip::new(bitwise_lookup_chip.clone(), offset),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::JalLuiRv32(chip));
                }
                ExecutorName::JalrRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32JalrChip::new(
                        Rv32JalrAdapterChip::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        Rv32JalrCoreChip::new(
                            bitwise_lookup_chip.clone(),
                            range_checker.clone(),
                            offset,
                        ),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::JalrRv32(chip));
                }
                ExecutorName::AuipcRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32AuipcChip::new(
                        Rv32RdWriteAdapterChip::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        Rv32AuipcCoreChip::new(bitwise_lookup_chip.clone(), offset),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::AuipcRv32(chip));
                }
                ExecutorName::BaseAlu256Rv32 => {
                    let chip = Rc::new(RefCell::new(Rv32BaseAlu256Chip::new(
                        Rv32HeapAdapterChip::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        BaseAluCoreChip::new(bitwise_lookup_chip.clone(), offset),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::BaseAlu256Rv32(chip));
                }
                ExecutorName::LessThan256Rv32 => {
                    let chip = Rc::new(RefCell::new(Rv32LessThan256Chip::new(
                        Rv32HeapAdapterChip::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        LessThanCoreChip::new(bitwise_lookup_chip.clone(), offset),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::LessThan256Rv32(chip));
                }
                ExecutorName::Multiplication256Rv32 => {
                    let chip = Rc::new(RefCell::new(Rv32Multiplication256Chip::new(
                        Rv32HeapAdapterChip::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        MultiplicationCoreChip::new(range_tuple_checker.clone(), offset),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::Multiplication256Rv32(chip));
                }
                ExecutorName::Shift256Rv32 => {
                    let chip = Rc::new(RefCell::new(Rv32Shift256Chip::new(
                        Rv32HeapAdapterChip::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        ShiftCoreChip::new(
                            bitwise_lookup_chip.clone(),
                            range_checker.clone(),
                            offset,
                        ),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::Shift256Rv32(chip));
                }
                ExecutorName::CastF => {
                    let chip = Rc::new(RefCell::new(CastFChip::new(
                        ConvertAdapterChip::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        CastFCoreChip::new(
                            memory_controller.borrow().range_checker.clone(),
                            offset,
                        ),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::CastF(chip));
                }
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
                offset,
            )));
            for opcode in range {
                executors.insert(opcode, poseidon_chip.clone().into());
            }
            chips.push(AxVmChip::Poseidon2(poseidon_chip));
        }

        for (local_opcode_idx, class_offset, executor, modulus) in
            gen_ec_executor_tuple(&self.supported_ec_curves)
        {
            let global_opcode_idx = local_opcode_idx + class_offset;
            if executors.contains_key(&local_opcode_idx) {
                panic!("Attempting to override an executor for opcode {global_opcode_idx}");
            }
            let config32 = ExprBuilderConfig {
                modulus: modulus.clone(),
                num_limbs: 32,
                limb_bits: 8,
            };
            let config48 = ExprBuilderConfig {
                modulus,
                num_limbs: 48,
                limb_bits: 8,
            };
            match executor {
                ExecutorName::EcAddNeRv32_2x32 => {
                    let chip = Rc::new(RefCell::new(EcAddNeChip::new(
                        Rv32VecHeapAdapterChip::<F, 2, 2, 2, 32, 32>::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        memory_controller.clone(),
                        config32,
                        class_offset,
                    )));
                    executors.insert(global_opcode_idx, chip.clone().into());
                    chips.push(AxVmChip::EcAddNeRv32_2x32(chip));
                }
                ExecutorName::EcDoubleRv32_2x32 => {
                    let chip = Rc::new(RefCell::new(EcDoubleChip::new(
                        Rv32VecHeapAdapterChip::<F, 1, 2, 2, 32, 32>::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        memory_controller.clone(),
                        config32,
                        class_offset,
                    )));
                    executors.insert(global_opcode_idx, chip.clone().into());
                    chips.push(AxVmChip::EcDoubleRv32_2x32(chip));
                }
                ExecutorName::EcAddNeRv32_6x16 => {
                    let chip = Rc::new(RefCell::new(EcAddNeChip::new(
                        Rv32VecHeapAdapterChip::<F, 2, 6, 6, 16, 16>::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        memory_controller.clone(),
                        config48,
                        class_offset,
                    )));
                    executors.insert(global_opcode_idx, chip.clone().into());
                    chips.push(AxVmChip::EcAddNeRv32_6x16(chip));
                }
                ExecutorName::EcDoubleRv32_6x16 => {
                    let chip = Rc::new(RefCell::new(EcDoubleChip::new(
                        Rv32VecHeapAdapterChip::<F, 1, 6, 6, 16, 16>::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        memory_controller.clone(),
                        config48,
                        class_offset,
                    )));
                    executors.insert(global_opcode_idx, chip.clone().into());
                    chips.push(AxVmChip::EcDoubleRv32_6x16(chip));
                }
                ExecutorName::EcLineMul013By013 => {
                    let chip = Rc::new(RefCell::new(EcLineMul013By013Chip::new(
                        Rv32VecHeapAdapterChip::<F, 2, 4, 10, 32, 32>::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        memory_controller.clone(),
                        config32,
                        class_offset,
                    )));
                    executors.insert(global_opcode_idx, chip.clone().into());
                    chips.push(AxVmChip::EcLineMul013By013(chip));
                }
                ExecutorName::EcLineMulBy01234 => {
                    let chip = Rc::new(RefCell::new(EcLineMulBy01234Chip::new(
                        Rv32VecHeapAdapterChip::<F, 2, 12, 12, 32, 32>::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        memory_controller.clone(),
                        config32,
                        class_offset,
                    )));
                    executors.insert(global_opcode_idx, chip.clone().into());
                    chips.push(AxVmChip::EcLineMulBy01234(chip));
                }
                _ => unreachable!("Unsupported executor"),
            }
        }

        for (local_opcode_idx, class_offset, executor, modulus) in
            gen_pairing_executor_tuple(&self.supported_pairing_curves)
        {
            let global_opcode_idx = local_opcode_idx + class_offset;
            if executors.contains_key(&local_opcode_idx) {
                panic!("Attempting to override an executor for opcode {global_opcode_idx}");
            }
            let config32 = ExprBuilderConfig {
                modulus: modulus.clone(),
                num_limbs: 32,
                limb_bits: 8,
            };
            let config48 = ExprBuilderConfig {
                modulus,
                num_limbs: 48,
                limb_bits: 8,
            };
            match executor {
                ExecutorName::MillerDoubleStepRv32_32 => {
                    let chip = Rc::new(RefCell::new(MillerDoubleStepChip::new(
                        Rv32VecHeapAdapterChip::<F, 1, 4, 8, 32, 32>::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        memory_controller.clone(),
                        config32,
                        class_offset,
                    )));
                    executors.insert(global_opcode_idx, chip.clone().into());
                    chips.push(AxVmChip::MillerDoubleStepRv32_32(chip));
                }
                ExecutorName::MillerDoubleStepRv32_48 => {
                    let chip = Rc::new(RefCell::new(MillerDoubleStepChip::new(
                        Rv32VecHeapAdapterChip::<F, 1, 12, 24, 16, 16>::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        memory_controller.clone(),
                        config48,
                        class_offset,
                    )));
                    executors.insert(global_opcode_idx, chip.clone().into());
                    chips.push(AxVmChip::MillerDoubleStepRv32_48(chip));
                }
                ExecutorName::MillerDoubleAndAddStepRv32_32 => {
                    let chip = Rc::new(RefCell::new(MillerDoubleAndAddStepChip::new(
                        Rv32VecHeapAdapterChip::<F, 2, 4, 12, 32, 32>::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        memory_controller.clone(),
                        config32,
                        class_offset,
                    )));
                    executors.insert(global_opcode_idx, chip.clone().into());
                    chips.push(AxVmChip::MillerDoubleAndAddStepRv32_32(chip));
                }
                ExecutorName::MillerDoubleAndAddStepRv32_48 => {
                    let chip = Rc::new(RefCell::new(MillerDoubleAndAddStepChip::new(
                        Rv32VecHeapAdapterChip::<F, 2, 12, 36, 16, 16>::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        memory_controller.clone(),
                        config48,
                        class_offset,
                    )));
                    executors.insert(global_opcode_idx, chip.clone().into());
                    chips.push(AxVmChip::MillerDoubleAndAddStepRv32_48(chip));
                }
                _ => unreachable!("Unsupported executor"),
            }
        }

        for (local_range, executor, class_offset, modulus) in
            gen_modular_executor_tuple(self.supported_modulus.clone())
        {
            let range = shift_range(*local_range.start()..*local_range.end() + 1, class_offset);
            for global_opcode_idx in range.clone() {
                if executors.contains_key(&global_opcode_idx) {
                    panic!("Attempting to override an executor for opcode {global_opcode_idx}");
                }
            }
            let config32 = ExprBuilderConfig {
                modulus: modulus.clone(),
                num_limbs: 32,
                limb_bits: 8,
            };
            let config48 = ExprBuilderConfig {
                modulus,
                num_limbs: 48,
                limb_bits: 8,
            };
            match executor {
                ExecutorName::ModularAddSub => {
                    let new_chip = Rc::new(RefCell::new(KernelModularAddSubChip::new(
                        NativeVecHeapAdapterChip::<F, 2, 1, 1, 32, 32>::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        ModularAddSubCoreChip::new(
                            config32,
                            memory_controller.borrow().range_checker.clone(),
                            class_offset,
                        ),
                        memory_controller.clone(),
                    )));
                    for global_opcode in range {
                        executors.insert(global_opcode, new_chip.clone().into());
                    }
                    chips.push(AxVmChip::ModularAddSub(new_chip.clone()));
                }
                ExecutorName::ModularMultDiv => {
                    let new_chip = Rc::new(RefCell::new(KernelModularMulDivChip::new(
                        NativeVecHeapAdapterChip::<F, 2, 1, 1, 32, 32>::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        ModularMulDivCoreChip::new(
                            config32,
                            memory_controller.borrow().range_checker.clone(),
                            class_offset,
                        ),
                        memory_controller.clone(),
                    )));
                    for global_opcode in range {
                        executors.insert(global_opcode, new_chip.clone().into());
                    }
                    chips.push(AxVmChip::ModularMultDiv(new_chip));
                }
                ExecutorName::ModularAddSubRv32_1x32 => {
                    let new_chip = Rc::new(RefCell::new(ModularAddSubChip::new(
                        Rv32VecHeapAdapterChip::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        ModularAddSubCoreChip::new(
                            config32,
                            memory_controller.borrow().range_checker.clone(),
                            class_offset,
                        ),
                        memory_controller.clone(),
                    )));
                    for global_opcode in range {
                        executors.insert(global_opcode, new_chip.clone().into());
                    }
                    chips.push(AxVmChip::ModularAddSubRv32_1x32(new_chip));
                }
                ExecutorName::ModularMulDivRv32_1x32 => {
                    let new_chip = Rc::new(RefCell::new(ModularMulDivChip::new(
                        Rv32VecHeapAdapterChip::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        ModularMulDivCoreChip::new(
                            config32,
                            memory_controller.borrow().range_checker.clone(),
                            class_offset,
                        ),
                        memory_controller.clone(),
                    )));
                    for global_opcode in range {
                        executors.insert(global_opcode, new_chip.clone().into());
                    }
                    chips.push(AxVmChip::ModularMulDivRv32_1x32(new_chip));
                }
                ExecutorName::ModularAddSubRv32_3x16 => {
                    let new_chip = Rc::new(RefCell::new(ModularAddSubChip::new(
                        Rv32VecHeapAdapterChip::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        ModularAddSubCoreChip::new(
                            config48,
                            memory_controller.borrow().range_checker.clone(),
                            class_offset,
                        ),
                        memory_controller.clone(),
                    )));
                    for global_opcode in range {
                        executors.insert(global_opcode, new_chip.clone().into());
                    }
                    chips.push(AxVmChip::ModularAddSubRv32_3x16(new_chip));
                }
                ExecutorName::ModularMulDivRv32_3x16 => {
                    let new_chip = Rc::new(RefCell::new(ModularMulDivChip::new(
                        Rv32VecHeapAdapterChip::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        ModularMulDivCoreChip::new(
                            config48,
                            memory_controller.borrow().range_checker.clone(),
                            class_offset,
                        ),
                        memory_controller.clone(),
                    )));
                    for global_opcode in range {
                        executors.insert(global_opcode, new_chip.clone().into());
                    }
                    chips.push(AxVmChip::ModularMulDivRv32_3x16(new_chip));
                }
                _ => unreachable!(
                    "modular_executors should only contain ModularAddSub and ModularMultDiv"
                ),
            }
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
            memory_controller,
            range_checker_chip: range_checker,
        }
    }
}

// Returns (local_opcode_idx, global offset, executor name, modulus)
fn gen_ec_executor_tuple(
    supported_ec_curves: &[EcCurve],
) -> Vec<(usize, usize, ExecutorName, BigUint)> {
    supported_ec_curves
        .iter()
        .enumerate()
        .flat_map(|(i, curve)| {
            let class_offset = EccOpcode::default_offset() + i * EccOpcode::COUNT;
            let bytes = curve.prime().bits().div_ceil(8);
            if bytes <= 32 {
                vec![
                    (
                        EccOpcode::EC_ADD_NE as usize,
                        class_offset,
                        ExecutorName::EcAddNeRv32_2x32,
                        curve.prime(),
                    ),
                    (
                        EccOpcode::EC_DOUBLE as usize,
                        class_offset,
                        ExecutorName::EcDoubleRv32_2x32,
                        curve.prime(),
                    ),
                ]
            } else if bytes <= 48 {
                vec![
                    (
                        EccOpcode::EC_ADD_NE as usize,
                        class_offset,
                        ExecutorName::EcAddNeRv32_6x16,
                        curve.prime(),
                    ),
                    (
                        EccOpcode::EC_DOUBLE as usize,
                        class_offset,
                        ExecutorName::EcDoubleRv32_6x16,
                        curve.prime(),
                    ),
                ]
            } else {
                panic!("curve {:?} is not supported", curve);
            }
        })
        .collect()
}

// Returns (local_opcode_idx, global offset, executor name, modulus)
fn gen_pairing_executor_tuple(
    supported_ec_curves: &[EcCurve],
) -> Vec<(usize, usize, ExecutorName, BigUint)> {
    supported_ec_curves
        .iter()
        .enumerate()
        .flat_map(|(i, curve)| {
            let class_offset = PairingOpcode::default_offset() + i * PairingOpcode::COUNT;
            let bytes = curve.prime().bits().div_ceil(8);
            if bytes <= 32 {
                vec![
                    (
                        PairingOpcode::MILLER_DOUBLE_STEP as usize,
                        class_offset,
                        ExecutorName::MillerDoubleStepRv32_32,
                        curve.prime(),
                    ),
                    (
                        PairingOpcode::MILLER_DOUBLE_AND_ADD_STEP as usize,
                        class_offset,
                        ExecutorName::MillerDoubleAndAddStepRv32_32,
                        curve.prime(),
                    ),
                ]
            } else if bytes <= 48 {
                vec![
                    (
                        PairingOpcode::MILLER_DOUBLE_STEP as usize,
                        class_offset,
                        ExecutorName::MillerDoubleStepRv32_48,
                        curve.prime(),
                    ),
                    (
                        PairingOpcode::MILLER_DOUBLE_AND_ADD_STEP as usize,
                        class_offset,
                        ExecutorName::MillerDoubleAndAddStepRv32_48,
                        curve.prime(),
                    ),
                ]
            } else {
                panic!("curve {:?} is not supported", curve);
            }
        })
        .collect()
}

fn gen_modular_executor_tuple(
    supported_modulus: Vec<BigUint>,
) -> Vec<(RangeInclusive<usize>, ExecutorName, usize, BigUint)> {
    supported_modulus
        .into_iter()
        .enumerate()
        .flat_map(|(i, modulus)| {
            // TODO[jpw]: delete the Kernel executors; for now I will always add both the kernel
            // and intrinsic executors together
            let class_offset =
                ModularArithmeticOpcode::default_offset() + i * ModularArithmeticOpcode::COUNT;
            let mut res = vec![
                (
                    ModularArithmeticOpcode::ADD as usize..=(ModularArithmeticOpcode::SUB as usize),
                    ExecutorName::ModularAddSub,
                    class_offset,
                    modulus.clone(),
                ),
                (
                    ModularArithmeticOpcode::MUL as usize..=(ModularArithmeticOpcode::DIV as usize),
                    ExecutorName::ModularMultDiv,
                    class_offset,
                    modulus.clone(),
                ),
            ];
            // determine the number of bytes needed to represent a prime field element
            let bytes = modulus.bits().div_ceil(8);
            // We want to use log_num_lanes as a const, this likely requires a macro
            let class_offset = Rv32ModularArithmeticOpcode::default_offset()
                + i * Rv32ModularArithmeticOpcode::COUNT;
            if bytes <= 32 {
                res.extend([
                    (
                        Rv32ModularArithmeticOpcode::ADD as usize
                            ..=(Rv32ModularArithmeticOpcode::SUB as usize),
                        ExecutorName::ModularAddSubRv32_1x32,
                        class_offset,
                        modulus.clone(),
                    ),
                    (
                        Rv32ModularArithmeticOpcode::MUL as usize
                            ..=(Rv32ModularArithmeticOpcode::DIV as usize),
                        ExecutorName::ModularMulDivRv32_1x32,
                        class_offset,
                        modulus,
                    ),
                ])
            } else if bytes <= 48 {
                res.extend([
                    (
                        Rv32ModularArithmeticOpcode::ADD as usize
                            ..=(Rv32ModularArithmeticOpcode::SUB as usize),
                        ExecutorName::ModularAddSubRv32_3x16,
                        class_offset,
                        modulus.clone(),
                    ),
                    (
                        Rv32ModularArithmeticOpcode::MUL as usize
                            ..=(Rv32ModularArithmeticOpcode::DIV as usize),
                        ExecutorName::ModularMulDivRv32_3x16,
                        class_offset,
                        modulus,
                    ),
                ])
            } else {
                panic!("modulus {:?} is too large", modulus);
            }

            res
        })
        .collect()
}

fn shift_range(r: Range<usize>, x: usize) -> Range<usize> {
    let start = r.start + x;
    let end = r.end + x;
    start..end
}

fn default_executor_range(executor: ExecutorName) -> (Range<usize>, usize) {
    let (start, len, offset) = match executor {
        // Terminate is not handled by executor, it is done by system (VmConnectorChip)
        ExecutorName::Phantom => (
            SystemOpcode::PHANTOM.with_default_offset(),
            1,
            SystemOpcode::default_offset(),
        ),
        ExecutorName::LoadStore => (
            NativeLoadStoreOpcode::default_offset(),
            NativeLoadStoreOpcode::COUNT,
            NativeLoadStoreOpcode::default_offset(),
        ),
        ExecutorName::BranchEqual => (
            NativeBranchEqualOpcode::default_offset(),
            BranchEqualOpcode::COUNT,
            NativeBranchEqualOpcode::default_offset(),
        ),
        ExecutorName::Jal => (
            NativeJalOpcode::default_offset(),
            NativeJalOpcode::COUNT,
            NativeJalOpcode::default_offset(),
        ),
        ExecutorName::FieldArithmetic => (
            FieldArithmeticOpcode::default_offset(),
            FieldArithmeticOpcode::COUNT,
            FieldArithmeticOpcode::default_offset(),
        ),
        ExecutorName::FieldExtension => (
            FieldExtensionOpcode::default_offset(),
            FieldExtensionOpcode::COUNT,
            FieldExtensionOpcode::default_offset(),
        ),
        ExecutorName::PublicValues => (
            PublishOpcode::default_offset(),
            PublishOpcode::COUNT,
            PublishOpcode::default_offset(),
        ),
        ExecutorName::Poseidon2 => (
            Poseidon2Opcode::default_offset(),
            Poseidon2Opcode::COUNT,
            Poseidon2Opcode::default_offset(),
        ),
        ExecutorName::Keccak256 => (
            Keccak256Opcode::default_offset(),
            Keccak256Opcode::COUNT,
            Keccak256Opcode::default_offset(),
        ),
        ExecutorName::FriFold => (
            FriFoldOpcode::default_offset(),
            FriFoldOpcode::COUNT,
            FriFoldOpcode::default_offset(),
        ),
        ExecutorName::BaseAluRv32 => (
            BaseAluOpcode::default_offset(),
            BaseAluOpcode::COUNT,
            BaseAluOpcode::default_offset(),
        ),
        ExecutorName::LoadStoreRv32 => (
            // LOADW through STOREB
            Rv32LoadStoreOpcode::default_offset(),
            Rv32LoadStoreOpcode::STOREB as usize + 1,
            Rv32LoadStoreOpcode::default_offset(),
        ),
        ExecutorName::LoadSignExtendRv32 => (
            // [LOADB, LOADH]
            Rv32LoadStoreOpcode::LOADB.with_default_offset(),
            2,
            Rv32LoadStoreOpcode::default_offset(),
        ),
        ExecutorName::HintStoreRv32 => (
            Rv32HintStoreOpcode::default_offset(),
            Rv32HintStoreOpcode::COUNT,
            Rv32HintStoreOpcode::default_offset(),
        ),
        ExecutorName::JalLuiRv32 => (
            Rv32JalLuiOpcode::default_offset(),
            Rv32JalLuiOpcode::COUNT,
            Rv32JalLuiOpcode::default_offset(),
        ),
        ExecutorName::JalrRv32 => (
            Rv32JalrOpcode::default_offset(),
            Rv32JalrOpcode::COUNT,
            Rv32JalrOpcode::default_offset(),
        ),
        ExecutorName::AuipcRv32 => (
            Rv32AuipcOpcode::default_offset(),
            Rv32AuipcOpcode::COUNT,
            Rv32AuipcOpcode::default_offset(),
        ),
        ExecutorName::BaseAlu256Rv32 => (
            Rv32BaseAlu256Opcode::default_offset(),
            BaseAluOpcode::COUNT,
            Rv32BaseAlu256Opcode::default_offset(),
        ),
        ExecutorName::LessThanRv32 => (
            LessThanOpcode::default_offset(),
            LessThanOpcode::COUNT,
            LessThanOpcode::default_offset(),
        ),
        ExecutorName::LessThan256Rv32 => (
            Rv32LessThan256Opcode::default_offset(),
            LessThanOpcode::COUNT,
            Rv32LessThan256Opcode::default_offset(),
        ),
        ExecutorName::MultiplicationRv32 => (
            MulOpcode::default_offset(),
            MulOpcode::COUNT,
            MulOpcode::default_offset(),
        ),
        ExecutorName::MultiplicationHighRv32 => (
            MulHOpcode::default_offset(),
            MulHOpcode::COUNT,
            MulHOpcode::default_offset(),
        ),
        ExecutorName::Multiplication256Rv32 => (
            Rv32Mul256Opcode::default_offset(),
            MulOpcode::COUNT,
            Rv32Mul256Opcode::default_offset(),
        ),
        ExecutorName::DivRemRv32 => (
            DivRemOpcode::default_offset(),
            DivRemOpcode::COUNT,
            DivRemOpcode::default_offset(),
        ),
        ExecutorName::ShiftRv32 => (
            ShiftOpcode::default_offset(),
            ShiftOpcode::COUNT,
            ShiftOpcode::default_offset(),
        ),
        ExecutorName::Shift256Rv32 => (
            Rv32Shift256Opcode::default_offset(),
            ShiftOpcode::COUNT,
            Rv32Shift256Opcode::default_offset(),
        ),
        ExecutorName::BranchEqualRv32 => (
            BranchEqualOpcode::default_offset(),
            BranchEqualOpcode::COUNT,
            BranchEqualOpcode::default_offset(),
        ),
        ExecutorName::BranchLessThanRv32 => (
            BranchLessThanOpcode::default_offset(),
            BranchLessThanOpcode::COUNT,
            BranchLessThanOpcode::default_offset(),
        ),
        ExecutorName::CastF => (
            CastfOpcode::default_offset(),
            CastfOpcode::COUNT,
            CastfOpcode::default_offset(),
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

fn get_name_and_cells(chip: &impl ChipUsageGetter) -> (String, usize) {
    (chip.air_name(), chip.current_trace_cells())
}
