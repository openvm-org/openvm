use std::{
    cell::RefCell,
    collections::{BTreeMap, VecDeque},
    rc::Rc,
    sync::Arc,
};

use afs_primitives::{
    modular_multiplication::bigint::air::ModularArithmeticBigIntAir,
    range_gate::RangeCheckerGateChip, xor::lookup::XorLookupChip,
};
use afs_stark_backend::rap::AnyRap;
use p3_commit::PolynomialSpace;
use p3_field::PrimeField32;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use poseidon2_air::poseidon2::Poseidon2Config;

use super::{VirtualMachineState, VmConfig, VmCycleTracker, VmMetrics};
use crate::{
    arch::{
        bus::ExecutionBus,
        chips::{InstructionExecutorVariant, MachineChip, MachineChipVariant},
        instructions::{
            Opcode, FIELD_ARITHMETIC_INSTRUCTIONS, FIELD_EXTENSION_INSTRUCTIONS,
            SECP256K1_COORD_MODULAR_ARITHMETIC_INSTRUCTIONS,
            SECP256K1_SCALAR_MODULAR_ARITHMETIC_INSTRUCTIONS,
        },
    },
    cpu::{trace::ExecutionError, CpuChip, BYTE_XOR_BUS, RANGE_CHECKER_BUS},
    field_arithmetic::FieldArithmeticChip,
    field_extension::chip::FieldExtensionArithmeticChip,
    hashes::{keccak::hasher::KeccakVmChip, poseidon2::Poseidon2Chip},
    memory::{
        manager::{MemoryChip, MemoryChipRef},
        offline_checker::bus::MemoryBus,
    },
    modular_multiplication::{air::ModularArithmeticVmAir, ModularArithmeticChip},
    program::{Program, ProgramChip},
    vm::cycle_tracker::CycleTracker,
};

pub struct ExecutionSegment<F: PrimeField32> {
    pub config: VmConfig,

    pub executors: BTreeMap<Opcode, InstructionExecutorVariant<F>>,
    pub chips: Vec<MachineChipVariant<F>>,
    pub cpu_chip: Rc<RefCell<CpuChip<F>>>,
    pub program_chip: Rc<RefCell<ProgramChip<F>>>,
    pub memory_chip: MemoryChipRef<F>,

    pub input_stream: VecDeque<Vec<F>>,
    pub hint_stream: VecDeque<F>,

    pub cycle_tracker: VmCycleTracker,
    /// Collected metrics for this segment alone.
    /// Only collected when `config.collect_metrics` is true.
    pub(crate) collected_metrics: VmMetrics,
}

pub struct SegmentResult<SC: StarkGenericConfig> {
    pub airs: Vec<Box<dyn AnyRap<SC>>>,
    pub traces: Vec<RowMajorMatrix<Val<SC>>>,
    pub public_values: Vec<Vec<Val<SC>>>,

    pub metrics: VmMetrics,
}

impl<SC: StarkGenericConfig> SegmentResult<SC> {
    pub fn max_log_degree(&self) -> usize {
        self.traces
            .iter()
            .map(RowMajorMatrix::height)
            .max()
            .unwrap()
    }
}

impl<F: PrimeField32> ExecutionSegment<F> {
    /// Creates a new execution segment from a program and initial state, using parent VM config
    pub fn new(config: VmConfig, program: Program<F>, state: VirtualMachineState<F>) -> Self {
        let execution_bus = ExecutionBus(0);
        let memory_bus = MemoryBus(1);
        let range_bus = RangeCheckBus::new(RANGE_CHECKER_BUS, 1 << config.memory_config.decomp);
        let range_checker = Arc::new(RangeCheckerGateChip::new(range_bus));

        let memory_chip = Rc::new(RefCell::new(MemoryChip::with_volatile_memory(
            memory_bus,
            config.memory_config,
            range_checker.clone(),
        )));
        let cpu_chip = Rc::new(RefCell::new(CpuChip::from_state(
            config.cpu_options(),
            execution_bus,
            memory_chip.clone(),
            state.state,
        )));
        let program_chip = Rc::new(RefCell::new(ProgramChip::new(program)));

        let mut executors = BTreeMap::new();
        macro_rules! assign {
            ($opcodes: expr, $executor: expr) => {
                for opcode in $opcodes {
                    executors.insert(opcode, $executor.clone().into());
                }
            };
        }

        // NOTE: The order of entries in `chips` must be a linear extension of the dependency DAG.
        // That is, if chip A holds a strong reference to chip B, then A must precede B in `chips`.

        let mut chips = vec![
            MachineChipVariant::Cpu(cpu_chip.clone()),
            MachineChipVariant::Program(program_chip.clone()),
        ];

        if config.field_arithmetic_enabled {
            let field_arithmetic_chip = Rc::new(RefCell::new(FieldArithmeticChip::new(
                execution_bus,
                memory_chip.clone(),
            )));
            assign!(FIELD_ARITHMETIC_INSTRUCTIONS, field_arithmetic_chip);
            chips.push(MachineChipVariant::FieldArithmetic(field_arithmetic_chip));
        }
        if config.field_extension_enabled {
            let field_extension_chip = Rc::new(RefCell::new(FieldExtensionArithmeticChip::new(
                execution_bus,
                memory_chip.clone(),
            )));
            assign!(FIELD_EXTENSION_INSTRUCTIONS, field_extension_chip);
            chips.push(MachineChipVariant::FieldExtension(field_extension_chip))
        }
        if config.perm_poseidon2_enabled || config.compress_poseidon2_enabled {
            let poseidon2_chip = Rc::new(RefCell::new(Poseidon2Chip::from_poseidon2_config(
                Poseidon2Config::<16, F>::new_p3_baby_bear_16(),
                execution_bus,
                memory_chip.clone(),
            )));
            if config.perm_poseidon2_enabled {
                assign!([Opcode::PERM_POS2], poseidon2_chip);
            }
            if config.compress_poseidon2_enabled {
                assign!([Opcode::COMP_POS2], poseidon2_chip);
            }
            chips.push(MachineChipVariant::Poseidon2(poseidon2_chip.clone()));
        }
        if config.keccak_enabled {
            let byte_xor_chip = Arc::new(XorLookupChip::new(BYTE_XOR_BUS));
            let keccak_chip = Rc::new(RefCell::new(KeccakVmChip::new(
                execution_bus,
                memory_chip.clone(),
                byte_xor_chip.clone(),
            )));
            assign!([Opcode::KECCAK256], keccak_chip);
            chips.push(MachineChipVariant::Keccak256(keccak_chip));
            chips.push(MachineChipVariant::ByteXor(byte_xor_chip));
        }
        if config.modular_multiplication_enabled {
            let airs = vec![
                (
                    ModularArithmeticChip::new(
                        ModularArithmeticVmAir {
                            air: ModularArithmeticBigIntAir::default_for_secp256k1_coord(),
                        },
                        memory_chip.clone(),
                    ),
                    ModularArithmeticBigIntAir::secp256k1_coord_prime(),
                ),
                (
                    ModularArithmeticChip::new(
                        ModularArithmeticVmAir {
                            air: ModularArithmeticBigIntAir::default_for_secp256k1_scalar(),
                        },
                        memory_chip.clone(),
                    ),
                    ModularArithmeticBigIntAir::secp256k1_scalar_prime(),
                ),
            ];
            // let mut modular_arithmetic_chips = BTreeMap::new();
            // for (air, modulus) in airs {
            //     assign!(MODULAR_ARITHMETIC_INSTRUCTIONS,);
            //     modular_arithmetic_chips.insert(modulus.clone(), air);
            // }
            assign!(
                SECP256K1_COORD_MODULAR_ARITHMETIC_INSTRUCTIONS,
                Rc::new(RefCell::new(airs[0].0.clone()))
            );
            assign!(
                SECP256K1_SCALAR_MODULAR_ARITHMETIC_INSTRUCTIONS,
                Rc::new(RefCell::new(airs[1].0.clone()))
            );
        }
        // let airs = vec![
        //     (
        //         ModularArithmeticChip::new(ModularArithmeticVmAir {
        //             air: ModularArithmeticBigIntAir::default_for_secp256k1_coord(),
        //         }),
        //         ModularArithmeticBigIntAir::secp256k1_coord_prime(),
        //     ),
        //     (
        //         ModularArithmeticChip::new(ModularArithmeticVmAir {
        //             air: ModularArithmeticBigIntAir::default_for_secp256k1_scalar(),
        //         }),
        //         ModularArithmeticBigIntAir::secp256k1_scalar_prime(),
        //     ),
        // ];
        // let mut modular_arithmetic_chips = BTreeMap::new();
        // for (air, modulus) in airs {
        //     modular_arithmetic_chips.insert(modulus.clone(), air);
        // }

        // Most chips have a reference to the memory chip, and the memory chip has a reference to
        // the range checker chip.
        chips.push(MachineChipVariant::Memory(memory_chip.clone()));
        chips.push(MachineChipVariant::RangeChecker(range_checker.clone()));

        Self {
            config,
            executors,
            chips,
            cpu_chip,
            program_chip,
            memory_chip,
            input_stream: state.input_stream,
            hint_stream: state.hint_stream,
            collected_metrics: Default::default(),
            cycle_tracker: CycleTracker::new(),
        }
    }

    /// Stopping is triggered by should_segment()
    pub fn execute(&mut self) -> Result<(), ExecutionError> {
        CpuChip::execute(self)
    }

    /// Compile the AIRs and trace generation outputs for the chips used in this segment
    /// Should be called after ::execute
    pub fn produce_result<SC: StarkGenericConfig>(self) -> SegmentResult<SC>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
    {
        let mut result = SegmentResult {
            airs: vec![],
            traces: vec![],
            public_values: vec![],
            metrics: self.collected_metrics,
        };

        // Drop all strong references to chips other than self.chips, which will be consumed next.
        drop(self.executors);
        drop(self.cpu_chip);
        drop(self.program_chip);
        drop(self.memory_chip);

        for mut chip in self.chips {
            if chip.current_trace_height() != 0 {
                result.airs.push(chip.air());
                result.public_values.push(chip.generate_public_values());
                result.traces.push(chip.generate_trace());
            }
        }

        result
    }

    /// Returns bool of whether to switch to next segment or not. This is called every clock cycle inside of CPU trace generation.
    ///
    /// Default config: switch if any runtime chip height exceeds 1<<20 - 100
    ///
    /// Used by CpuChip::execute, should be private in the future
    pub fn should_segment(&mut self) -> bool {
        self.chips
            .iter()
            .any(|chip| chip.current_trace_height() > self.config.max_segment_len)
    }

    /// Used by CpuChip::execute, should be private in the future
    pub fn current_trace_cells(&self) -> usize {
        self.chips
            .iter()
            .map(|chip| chip.current_trace_cells())
            .sum()
    }

    pub(crate) fn update_chip_metrics(&mut self) {
        self.collected_metrics.chip_metrics = self.chip_metrics();
    }

    fn chip_metrics(&self) -> BTreeMap<String, usize> {
        let mut metrics = BTreeMap::new();
        for chip in self.chips.iter() {
            let chip_name: &'static str = chip.into();
            metrics.insert(chip_name.into(), chip.current_trace_height());
        }
        metrics
    }
}
