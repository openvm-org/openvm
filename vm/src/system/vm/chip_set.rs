use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    ops::Range,
    rc::Rc,
    sync::Arc,
};

use afs_primitives::{
    range_tuple::{RangeTupleCheckerBus, RangeTupleCheckerChip},
    var_range::{bus::VariableRangeCheckerBus, VariableRangeCheckerChip},
    xor::lookup::XorLookupChip,
};
use axvm_instructions::*;
use num_bigint_dig::BigUint;
use p3_field::PrimeField32;
use poseidon2_air::poseidon2::Poseidon2Config;
use strum::EnumCount;

use crate::{
    arch::{AxVmChip, AxVmInstructionExecutor, ExecutionBus, ExecutorName},
    intrinsics::{
        ecc::{EcAddUnequalChip, EcDoubleChip},
        hashes::{keccak::hasher::KeccakVmChip, poseidon2::Poseidon2Chip},
    },
    kernels::{
        adapters::{
            convert_adapter::ConvertAdapterChip, native_adapter::NativeAdapterChip,
            native_vectorized_adapter::NativeVectorizedAdapterChip,
        },
        castf::{CastFChip, CastFCoreChip},
        core::{
            CoreChip, BYTE_XOR_BUS, RANGE_CHECKER_BUS, RANGE_TUPLE_CHECKER_BUS,
            READ_INSTRUCTION_BUS,
        },
        field_arithmetic::{FieldArithmeticChip, FieldArithmeticCoreChip},
        field_extension::{FieldExtensionChip, FieldExtensionCoreChip},
    },
    old::{
        alu::ArithmeticLogicChip, modular_addsub::ModularAddSubChip,
        modular_multdiv::ModularMultDivChip, shift::ShiftChip,
        uint_multiplication::UintMultiplicationChip,
    },
    rv32im::{
        adapters::{
            Rv32BaseAluAdapterChip, Rv32BranchAdapter, Rv32JalrAdapter, Rv32LoadStoreAdapter,
            Rv32MultAdapter, Rv32RdWriteAdapter,
        },
        base_alu::{BaseAluCoreChip, Rv32BaseAluChip},
        branch_eq::{BranchEqualCoreChip, Rv32BranchEqualChip},
        branch_lt::{BranchLessThanCoreChip, Rv32BranchLessThanChip},
        loadstore::{LoadStoreCoreChip, Rv32LoadStoreChip},
        new_divrem::{DivRemCoreChip, Rv32DivRemChip},
        new_lt::{LessThanCoreChip, Rv32LessThanChip},
        new_mul::{MultiplicationCoreChip, Rv32MultiplicationChip},
        new_mulh::{MulHCoreChip, Rv32MulHChip},
        new_shift::{Rv32ShiftChip, ShiftCoreChip},
        rv32_auipc::{Rv32AuipcChip, Rv32AuipcCoreChip},
        rv32_jal_lui::{Rv32JalLuiChip, Rv32JalLuiCoreChip},
        rv32_jalr::{Rv32JalrChip, Rv32JalrCoreChip},
    },
    system::{
        memory::{
            merkle::MemoryMerkleBus, offline_checker::MemoryBus, MemoryController,
            MemoryControllerRef, TimestampedEquipartition, CHUNK,
        },
        program::{bridge::ProgramBus, ProgramChip},
        vm::{
            config::{PersistenceType, VmConfig},
            connector::VmConnectorChip,
        },
    },
};

pub struct VmChipSet<F: PrimeField32> {
    pub program_chip: ProgramChip<F>,
    pub memory_controller: MemoryControllerRef<F>,
    pub connector_chip: VmConnectorChip<F>,

    pub executors: BTreeMap<usize, AxVmInstructionExecutor<F>>,
    pub chips: Vec<AxVmChip<F>>,
}

impl VmConfig {
    pub fn create_chip_set<F: PrimeField32>(&self) -> VmChipSet<F> {
        let execution_bus = ExecutionBus(0);
        let program_bus = ProgramBus(READ_INSTRUCTION_BUS);
        let memory_bus = MemoryBus(1);
        let merkle_bus = MemoryMerkleBus(12);
        let range_bus = VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, self.memory_config.decomp);
        let range_checker = Arc::new(VariableRangeCheckerChip::new(range_bus));
        let byte_xor_chip = Arc::new(XorLookupChip::new(BYTE_XOR_BUS));
        let range_tuple_bus =
            RangeTupleCheckerBus::new(RANGE_TUPLE_CHECKER_BUS, [(1 << 8), 32 * (1 << 8)]);
        let range_tuple_checker = Arc::new(RangeTupleCheckerChip::new(range_tuple_bus));

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
                    TimestampedEquipartition::<F, CHUNK>::new(),
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

        // CoreChip is always required even if it's not explicitly specified.
        required_executors.insert(ExecutorName::Core);
        // We always put Poseidon2 chips in the end. So it will be initialized separately.
        let has_poseidon_chip = required_executors.contains(&ExecutorName::Poseidon2);
        if has_poseidon_chip {
            required_executors.remove(&ExecutorName::Poseidon2);
        }
        // We may not use this chip if the memory kind is volatile and there is no executor for Poseidon2.
        let needs_poseidon_chip = has_poseidon_chip
            || (self.memory_config.persistence_type == PersistenceType::Persistent);

        for &executor in required_executors.iter() {
            let (range, offset) = default_executor_range(executor);
            for opcode in range.clone() {
                if executors.contains_key(&opcode) {
                    panic!("Attempting to override an executor for opcode {opcode}");
                }
            }
            match executor {
                ExecutorName::Core => {
                    let core_chip = Rc::new(RefCell::new(CoreChip::new(
                        self.core_options(),
                        execution_bus,
                        program_bus,
                        memory_controller.clone(),
                        0,
                        offset,
                    )));
                    for opcode in range {
                        executors.insert(opcode, core_chip.clone().into());
                    }
                    chips.push(AxVmChip::Core(core_chip));
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
                ExecutorName::Poseidon2 => {}
                ExecutorName::Keccak256 => {
                    let chip = Rc::new(RefCell::new(KeccakVmChip::new(
                        execution_bus,
                        program_bus,
                        memory_controller.clone(),
                        byte_xor_chip.clone(),
                        offset,
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::Keccak256(chip));
                }
                ExecutorName::ArithmeticLogicUnitRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32BaseAluChip::new(
                        Rv32BaseAluAdapterChip::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        BaseAluCoreChip::new(byte_xor_chip.clone(), offset),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::ArithmeticLogicUnitRv32(chip));
                }
                ExecutorName::ArithmeticLogicUnit256 => {
                    // We probably must include this chip if we include any modular arithmetic,
                    // not sure if we need to enforce this here.
                    let chip = Rc::new(RefCell::new(ArithmeticLogicChip::new(
                        execution_bus,
                        program_bus,
                        memory_controller.clone(),
                        byte_xor_chip.clone(),
                        offset,
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::ArithmeticLogicUnit256(chip));
                }
                ExecutorName::LessThanRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32LessThanChip::new(
                        Rv32BaseAluAdapterChip::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        LessThanCoreChip::new(byte_xor_chip.clone(), offset),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::LessThanRv32(chip));
                }
                ExecutorName::MultiplicationRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32MultiplicationChip::new(
                        Rv32MultAdapter::new(execution_bus, program_bus, memory_controller.clone()),
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
                        Rv32MultAdapter::new(execution_bus, program_bus, memory_controller.clone()),
                        MulHCoreChip::new(range_tuple_checker.clone(), offset),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::MultiplicationHighRv32(chip));
                }
                ExecutorName::U256Multiplication => {
                    let chip = Rc::new(RefCell::new(UintMultiplicationChip::new(
                        execution_bus,
                        program_bus,
                        memory_controller.clone(),
                        range_tuple_checker.clone(),
                        offset,
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::U256Multiplication(chip));
                }
                ExecutorName::DivRemRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32DivRemChip::new(
                        Rv32MultAdapter::new(execution_bus, program_bus, memory_controller.clone()),
                        DivRemCoreChip::new(range_tuple_checker.clone(), offset),
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
                        ShiftCoreChip::new(byte_xor_chip.clone(), range_checker.clone(), offset),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::ShiftRv32(chip));
                }
                ExecutorName::Shift256 => {
                    let chip = Rc::new(RefCell::new(ShiftChip::new(
                        execution_bus,
                        program_bus,
                        memory_controller.clone(),
                        byte_xor_chip.clone(),
                        offset,
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::Shift256(chip));
                }
                ExecutorName::LoadStoreRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32LoadStoreChip::new(
                        Rv32LoadStoreAdapter::new(range_checker.clone(), offset),
                        LoadStoreCoreChip::new(offset),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::LoadStoreRv32(chip));
                }
                ExecutorName::BranchEqualRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32BranchEqualChip::new(
                        Rv32BranchAdapter::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        BranchEqualCoreChip::new(offset),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::BranchEqualRv32(chip));
                }
                ExecutorName::BranchLessThanRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32BranchLessThanChip::new(
                        Rv32BranchAdapter::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        BranchLessThanCoreChip::new(byte_xor_chip.clone(), offset),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::BranchLessThanRv32(chip));
                }
                ExecutorName::JalLuiRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32JalLuiChip::new(
                        Rv32RdWriteAdapter::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        Rv32JalLuiCoreChip::new(byte_xor_chip.clone(), offset),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::JalLuiRv32(chip));
                }
                ExecutorName::JalrRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32JalrChip::new(
                        Rv32JalrAdapter::new(),
                        Rv32JalrCoreChip::new(offset),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::JalrRv32(chip));
                }
                ExecutorName::AuipcRv32 => {
                    let chip = Rc::new(RefCell::new(Rv32AuipcChip::new(
                        Rv32RdWriteAdapter::new(
                            execution_bus,
                            program_bus,
                            memory_controller.clone(),
                        ),
                        Rv32AuipcCoreChip::new(byte_xor_chip.clone(), offset),
                        memory_controller.clone(),
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::AuipcRv32(chip));
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
                // TODO: make these customizable opcode classes
                ExecutorName::Secp256k1AddUnequal => {
                    let chip = Rc::new(RefCell::new(EcAddUnequalChip::new(
                        execution_bus,
                        program_bus,
                        memory_controller.clone(),
                        offset,
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::Secp256k1AddUnequal(chip));
                }
                ExecutorName::Secp256k1Double => {
                    let chip = Rc::new(RefCell::new(EcDoubleChip::new(
                        execution_bus,
                        program_bus,
                        memory_controller.clone(),
                        offset,
                    )));
                    for opcode in range {
                        executors.insert(opcode, chip.clone().into());
                    }
                    chips.push(AxVmChip::Secp256k1Double(chip));
                }
                ExecutorName::ModularAddSub | ExecutorName::ModularMultDiv => {
                    unreachable!("Modular executors should be handled differently")
                }
            }
        }

        if needs_poseidon_chip {
            let (range, offset) = default_executor_range(ExecutorName::Poseidon2);
            let poseidon_chip = Rc::new(RefCell::new(Poseidon2Chip::from_poseidon2_config(
                Poseidon2Config::<16, F>::new_p3_baby_bear_16(),
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

        for (range, executor, offset, modulus) in
            gen_modular_executor_tuple(self.supported_modulus.clone())
        {
            for opcode in range.clone() {
                if executors.contains_key(&opcode) {
                    panic!("Attempting to override an executor for opcode {opcode}");
                }
            }
            match executor {
                ExecutorName::ModularAddSub => {
                    let new_chip = Rc::new(RefCell::new(ModularAddSubChip::new(
                        execution_bus,
                        program_bus,
                        memory_controller.clone(),
                        modulus,
                        offset,
                    )));
                    for opcode in range {
                        executors.insert(opcode, new_chip.clone().into());
                    }
                    chips.push(AxVmChip::ModularAddSub(new_chip.clone()));
                }
                ExecutorName::ModularMultDiv => {
                    let new_chip = Rc::new(RefCell::new(ModularMultDivChip::new(
                        execution_bus,
                        program_bus,
                        memory_controller.clone(),
                        modulus,
                        offset,
                    )));
                    for opcode in range {
                        executors.insert(opcode, new_chip.clone().into());
                    }
                    chips.push(AxVmChip::ModularMultDiv(new_chip.clone()));
                }
                _ => unreachable!(
                    "modular_executors should only contain ModularAddSub and ModularMultDiv"
                ),
            }
        }

        chips.push(AxVmChip::ByteXor(byte_xor_chip));
        chips.push(AxVmChip::RangeTupleChecker(range_tuple_checker));

        let connector_chip = VmConnectorChip::new(execution_bus);

        VmChipSet {
            program_chip,
            memory_controller,
            connector_chip,
            executors,
            chips,
        }
    }
}

fn gen_modular_executor_tuple(
    supported_modulus: Vec<BigUint>,
) -> Vec<(Range<usize>, ExecutorName, usize, BigUint)> {
    let num_ops_per_modulus = ModularArithmeticOpcode::COUNT;
    let add_sub_range = default_executor_range(ExecutorName::ModularAddSub);
    let mult_div_range = default_executor_range(ExecutorName::ModularMultDiv);
    supported_modulus
        .into_iter()
        .enumerate()
        .flat_map(|(i, modulus)| {
            let shift = i * num_ops_per_modulus;
            [
                (
                    shift_range(&add_sub_range.0, shift),
                    ExecutorName::ModularAddSub,
                    add_sub_range.1 + shift,
                    modulus.clone(),
                ),
                (
                    shift_range(&mult_div_range.0, shift),
                    ExecutorName::ModularMultDiv,
                    mult_div_range.1 + shift,
                    modulus,
                ),
            ]
        })
        .collect()
}

fn shift_range(r: &Range<usize>, x: usize) -> Range<usize> {
    let start = r.start + x;
    let end = r.end + x;
    start..end
}

fn default_executor_range(executor: ExecutorName) -> (Range<usize>, usize) {
    let (start, len, offset) = match executor {
        ExecutorName::Core => (
            CoreOpcode::default_offset(),
            CoreOpcode::COUNT,
            CoreOpcode::default_offset(),
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
        ExecutorName::ModularAddSub => (
            ModularArithmeticOpcode::default_offset(),
            2,
            ModularArithmeticOpcode::default_offset(),
        ),
        ExecutorName::ModularMultDiv => (
            ModularArithmeticOpcode::default_offset() + 2,
            2,
            ModularArithmeticOpcode::default_offset(),
        ),
        ExecutorName::ArithmeticLogicUnitRv32 => (
            AluOpcode::default_offset(),
            AluOpcode::COUNT,
            AluOpcode::default_offset(),
        ),
        ExecutorName::LoadStoreRv32 => (
            Rv32LoadStoreOpcode::default_offset(),
            Rv32LoadStoreOpcode::COUNT,
            Rv32LoadStoreOpcode::default_offset(),
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
        ExecutorName::ArithmeticLogicUnit256 => (
            U256Opcode::default_offset(),
            8,
            U256Opcode::default_offset(),
        ),
        ExecutorName::LessThanRv32 => (
            LessThanOpcode::default_offset(),
            LessThanOpcode::COUNT,
            LessThanOpcode::default_offset(),
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
        ExecutorName::U256Multiplication => (
            U256Opcode::default_offset() + 11,
            1,
            U256Opcode::default_offset(),
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
        ExecutorName::Shift256 => (
            U256Opcode::default_offset() + 8,
            3,
            U256Opcode::default_offset(),
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
        ExecutorName::Secp256k1AddUnequal => {
            (EccOpcode::default_offset(), 1, EccOpcode::default_offset())
        }
        ExecutorName::Secp256k1Double => (
            EccOpcode::default_offset() + 1,
            1,
            EccOpcode::default_offset(),
        ),
    };
    (start..(start + len), offset)
}
