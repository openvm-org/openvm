use std::{
    cell::RefCell,
    collections::{BTreeMap, VecDeque},
    rc::Rc,
    sync::Arc,
};

use afs_primitives::{
    modular_multiplication::bigint::air::ModularArithmeticBigIntAir,
    range_gate::RangeCheckerGateChip,
};
use afs_stark_backend::{
    config::{StarkGenericConfig, Val},
    rap::AnyRap,
};
use num_bigint_dig::BigUint;
use p3_field::PrimeField32;
use p3_matrix::dense::DenseMatrix;
use poseidon2_air::poseidon2::Poseidon2Config;

use super::{ChipType, VirtualMachineState, VmConfig};
use crate::{
    cpu::{
        trace::ExecutionError, CpuChip, CpuOptions, IS_LESS_THAN_BUS, POSEIDON2_BUS,
        RANGE_CHECKER_BUS,
    },
    field_arithmetic::FieldArithmeticChip,
    field_extension::FieldExtensionArithmeticChip,
    is_less_than::IsLessThanChip,
    memory::manager::MemoryManager,
    modular_multiplication::{air::ModularArithmeticVmAir, ModularArithmeticChip},
    poseidon2::Poseidon2Chip,
    program::{Program, ProgramChip},
    vm::{cycle_tracker::CycleTracker, metrics::VmMetrics},
};

pub struct ExecutionSegment<const NUM_WORDS: usize, const WORD_SIZE: usize, F: PrimeField32> {
    pub config: VmConfig,
    pub cpu_chip: CpuChip<WORD_SIZE, F>,
    pub program_chip: ProgramChip<F>,
    pub memory_manager: Rc<RefCell<MemoryManager<NUM_WORDS, WORD_SIZE, F>>>,
    pub field_arithmetic_chip: FieldArithmeticChip<F>,
    pub field_extension_chip: FieldExtensionArithmeticChip<NUM_WORDS, WORD_SIZE, F>,
    pub range_checker: Arc<RangeCheckerGateChip>,
    pub poseidon2_chip: Poseidon2Chip<16, NUM_WORDS, WORD_SIZE, F>,
    pub is_less_than_chip: IsLessThanChip<F>,
    pub modular_arithmetic_chips: BTreeMap<BigUint, ModularArithmeticChip<F>>,
    pub input_stream: VecDeque<Vec<F>>,
    pub hint_stream: VecDeque<F>,
    pub has_execution_happened: bool,
    pub public_values: Vec<Option<F>>,

    pub cycle_tracker: CycleTracker,
    /// Collected metrics for this segment alone.
    /// Only collected when `config.collect_metrics` is true.
    pub(crate) metrics: VmMetrics,
}

impl<const NUM_WORDS: usize, const WORD_SIZE: usize, F: PrimeField32>
    ExecutionSegment<NUM_WORDS, WORD_SIZE, F>
{
    /// Creates a new execution segment from a program and initial state, using parent VM config
    pub fn new(config: VmConfig, program: Program<F>, state: VirtualMachineState<F>) -> Self {
        let range_checker = Arc::new(RangeCheckerGateChip::new(
            RANGE_CHECKER_BUS,
            (1 << config.memory_config.decomp) as u32,
        ));
        let cpu_chip = CpuChip::from_state(config.cpu_options(), state.state, config.memory_config);
        let memory_manager = Rc::new(RefCell::new(MemoryManager::with_volatile_memory(
            config.memory_config,
            range_checker.clone(),
        )));
        let program_chip = ProgramChip::new(program);
        let field_arithmetic_chip = FieldArithmeticChip::new();
        let field_extension_chip = FieldExtensionArithmeticChip::new(
            config.memory_config,
            memory_manager.clone(),
            range_checker.clone(),
        );
        let poseidon2_chip = Poseidon2Chip::from_poseidon2_config(
            Poseidon2Config::<16, F>::new_p3_baby_bear_16(),
            config.memory_config,
            memory_manager.clone(),
            range_checker.clone(),
            POSEIDON2_BUS,
        );
        let airs = vec![
            (
                ModularArithmeticChip::new(ModularArithmeticVmAir {
                    air: ModularArithmeticBigIntAir::default_for_secp256k1_coord(),
                }),
                ModularArithmeticBigIntAir::secp256k1_coord_prime(),
            ),
            (
                ModularArithmeticChip::new(ModularArithmeticVmAir {
                    air: ModularArithmeticBigIntAir::default_for_secp256k1_scalar(),
                }),
                ModularArithmeticBigIntAir::secp256k1_scalar_prime(),
            ),
        ];
        let mut modular_arithmetic_chips = BTreeMap::new();
        for (air, modulus) in airs {
            modular_arithmetic_chips.insert(modulus.clone(), air);
        }
        let is_less_than_chip = IsLessThanChip::new(
            IS_LESS_THAN_BUS,
            30,
            config.memory_config.decomp,
            range_checker.clone(),
        );

        Self {
            config,
            has_execution_happened: false,
            public_values: vec![None; config.num_public_values],
            cpu_chip,
            memory_manager,
            program_chip,
            field_arithmetic_chip,
            field_extension_chip,
            range_checker,
            poseidon2_chip,
            modular_arithmetic_chips,
            is_less_than_chip,
            input_stream: state.input_stream,
            hint_stream: state.hint_stream,
            cycle_tracker: CycleTracker::new(),
            metrics: Default::default(),
        }
    }

    pub fn options(&self) -> CpuOptions {
        self.config.cpu_options()
    }

    /// Returns bool of whether to switch to next segment or not. This is called every clock cycle inside of CPU trace generation.
    ///
    /// Default config: switch if any runtime chip height exceeds 1<<20 - 100
    pub fn should_segment(&mut self) -> bool {
        let heights = [
            self.cpu_chip.current_height(),
            self.memory_manager.borrow().interface_chip.current_height(),
            self.field_arithmetic_chip.current_height(),
            self.field_extension_chip.current_height(),
            self.poseidon2_chip.current_height(),
        ];
        let max_height = *heights.iter().max().unwrap();
        max_height >= self.config.max_segment_len
    }

    /// Stopping is triggered by `Self::should_segment`.
    pub fn execute(&mut self) -> Result<(), ExecutionError> {
        CpuChip::execute(self)
    }

    /// Called by VM to generate traces for current segment. Includes empty traces.
    /// Should only be called after `Self::execute`.
    pub fn generate_traces(&mut self) -> Vec<DenseMatrix<F>> {
        let cpu_trace = CpuChip::generate_trace(self);
        let mut result = vec![
            cpu_trace,
            self.program_chip.generate_trace(),
            self.memory_manager
                .borrow()
                .generate_memory_interface_trace(),
        ];
        if self.config.cpu_options().field_arithmetic_enabled {
            result.push(self.field_arithmetic_chip.generate_trace());
        }
        if self.config.cpu_options().field_extension_enabled {
            result.push(self.field_extension_chip.generate_trace());
        }
        if self.config.cpu_options().poseidon2_enabled() {
            result.push(self.poseidon2_chip.generate_trace());
        }
        if self.config.cpu_options().is_less_than_enabled {
            result.push(self.is_less_than_chip.generate_trace());
        }
        // Note: range checker should be last because the chip.generate_trace() calls above
        // may influence the range checker.
        result.push(self.range_checker.generate_trace());

        result
    }

    /// Generate Merkle proof/memory diff traces, and publish public values
    ///
    /// For now, only publishes program counter public values
    pub fn generate_commitments(&mut self) -> Vec<DenseMatrix<F>> {
        // self.cpu_chip.generate_pvs();
        vec![]
    }

    pub fn get_num_chips(&self) -> usize {
        let mut result: usize = 4; // cpu, program, memory_interface, range_checker
        if self.config.cpu_options().field_arithmetic_enabled {
            result += 1;
        }
        if self.config.cpu_options().field_extension_enabled {
            result += 1;
        }
        if self.config.cpu_options().poseidon2_enabled() {
            result += 1;
        }
        if self.config.cpu_options().is_less_than_enabled {
            result += 1;
        }
        result
    }

    pub fn get_cpu_pis(&self) -> Vec<F> {
        self.cpu_chip
            .pis
            .clone()
            .into_iter()
            .chain(self.public_values.iter().map(|x| x.unwrap_or(F::zero())))
            .collect()
    }

    /// Returns public values for all chips in this segment
    pub fn get_pis(&self) -> Vec<Vec<F>> {
        let len = self.get_num_chips();
        let mut result: Vec<Vec<F>> = vec![vec![]; len];
        result[0] = self.get_cpu_pis();
        result
    }

    pub fn get_types(&self) -> Vec<ChipType> {
        let mut result: Vec<ChipType> =
            vec![ChipType::Cpu, ChipType::Program, ChipType::MemoryInterface];
        if self.config.cpu_options().field_arithmetic_enabled {
            result.push(ChipType::FieldArithmetic);
        }
        if self.config.cpu_options().field_extension_enabled {
            result.push(ChipType::FieldExtension);
        }
        if self.config.cpu_options().poseidon2_enabled() {
            result.push(ChipType::Poseidon2);
        }
        if self.config.cpu_options().is_less_than_enabled {
            result.push(ChipType::IsLessThan);
        }
        result.push(ChipType::RangeChecker);
        assert!(result.len() == self.get_num_chips());
        result
    }

    pub fn update_chip_metrics(&mut self) {
        self.metrics.chip_metrics = self.chip_metrics();
    }

    pub fn chip_metrics(&self) -> BTreeMap<String, usize> {
        let mut metrics = BTreeMap::new();
        metrics.insert("cpu_cycles".to_string(), self.cpu_chip.rows.len());
        metrics.insert("cpu_timestamp".to_string(), self.cpu_chip.state.timestamp);
        metrics.insert(
            "field_arithmetic_ops".to_string(),
            self.field_arithmetic_chip.operations.len(),
        );
        metrics.insert(
            "field_extension_ops".to_string(),
            self.field_extension_chip.operations.len(),
        );
        metrics.insert(
            "range_checker_count".to_string(),
            self.range_checker.count.len(),
        );
        metrics.insert(
            "poseidon2_chip_rows".to_string(),
            self.poseidon2_chip.rows.len(),
        );
        metrics.insert(
            "is_less_than_ops".to_string(),
            self.is_less_than_chip.rows.len(),
        );
        metrics
    }
}

/// Global function to get chips from a segment
pub fn get_chips<const NUM_WORDS: usize, const WORD_SIZE: usize, SC: StarkGenericConfig>(
    segment: ExecutionSegment<NUM_WORDS, WORD_SIZE, Val<SC>>,
    inclusion_mask: &[bool],
) -> Vec<Box<dyn AnyRap<SC>>>
where
    Val<SC>: PrimeField32,
{
    let num_chips = segment.get_num_chips();
    let mut result: Vec<Box<dyn AnyRap<SC>>> = vec![
        Box::new(segment.cpu_chip.air),
        Box::new(segment.program_chip.air),
        Box::new(segment.memory_manager.borrow().get_audit_air().clone()),
    ];
    if segment.config.cpu_options().field_arithmetic_enabled {
        result.push(Box::new(segment.field_arithmetic_chip.air));
    }
    if segment.config.cpu_options().field_extension_enabled {
        result.push(Box::new(segment.field_extension_chip.air));
    }
    if segment.config.cpu_options().poseidon2_enabled() {
        result.push(Box::new(segment.poseidon2_chip.air));
    }
    if segment.config.cpu_options().is_less_than_enabled {
        result.push(Box::new(segment.is_less_than_chip.air));
    }
    result.push(Box::new(segment.range_checker.air));

    assert_eq!(result.len(), num_chips);

    inclusion_mask
        .iter()
        .enumerate()
        .rev()
        .filter(|(_, inclusion)| !*inclusion)
        .map(|(index, _)| index)
        .for_each(|index| {
            result.remove(index);
        });

    assert_eq!(result.len(), inclusion_mask.iter().filter(|&x| *x).count());
    result
}
