use super::ChipType;
use super::VirtualMachineState;
use afs_stark_backend::config::{StarkGenericConfig, Val};
use afs_stark_backend::rap::AnyRap;
use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::sync::Arc;

use super::VmConfig;
use crate::{
    cpu::{
        trace::{ExecutionError, Instruction},
        CpuChip, CpuOptions, POSEIDON2_BUS, RANGE_CHECKER_BUS,
    },
    field_arithmetic::FieldArithmeticChip,
    field_extension::FieldExtensionArithmeticChip,
    memory::offline_checker::MemoryChip,
    poseidon2::Poseidon2Chip,
    program::ProgramChip,
};
use afs_primitives::range_gate::RangeCheckerGateChip;
use poseidon2_air::poseidon2::Poseidon2Config;

use p3_field::PrimeField32;
use p3_matrix::dense::DenseMatrix;

pub struct ExecutionSegment<const WORD_SIZE: usize, F: PrimeField32> {
    pub config: VmConfig,
    pub cpu_chip: CpuChip<WORD_SIZE, F>,
    pub program_chip: ProgramChip<F>,
    pub memory_chip: MemoryChip<WORD_SIZE, F>,
    pub field_arithmetic_chip: FieldArithmeticChip<F>,
    pub field_extension_chip: FieldExtensionArithmeticChip<WORD_SIZE, F>,
    pub range_checker: Arc<RangeCheckerGateChip>,
    pub poseidon2_chip: Poseidon2Chip<16, F>,
    pub input_stream: VecDeque<Vec<F>>,
    pub hint_stream: VecDeque<F>,
    pub has_generation_happened: bool,
    pub public_values: Vec<Option<F>>,
}

impl<const WORD_SIZE: usize, F: PrimeField32> ExecutionSegment<WORD_SIZE, F> {
    /// Creates a new execution segment from a program and initial state, using parent VM config
    pub fn new(
        config: VmConfig,
        program: Vec<Instruction<F>>,
        state: VirtualMachineState<F>,
    ) -> Self {
        let decomp = config.decomp;
        let limb_bits = config.limb_bits;

        let range_checker = Arc::new(RangeCheckerGateChip::new(RANGE_CHECKER_BUS, 1 << decomp));

        let cpu_chip = CpuChip::from_state(config.cpu_options(), state.state);
        let program_chip = ProgramChip::new(program);
        let memory_chip = MemoryChip::new(limb_bits, limb_bits, limb_bits, decomp, state.memory);
        let field_arithmetic_chip = FieldArithmeticChip::new();
        let field_extension_chip = FieldExtensionArithmeticChip::new();
        let poseidon2_chip = Poseidon2Chip::from_poseidon2_config(
            Poseidon2Config::<16, F>::new_p3_baby_bear_16(),
            POSEIDON2_BUS,
        );

        Self {
            config,
            has_generation_happened: false,
            public_values: vec![None; config.num_public_values],
            cpu_chip,
            program_chip,
            memory_chip,
            field_arithmetic_chip,
            field_extension_chip,
            range_checker,
            poseidon2_chip,
            input_stream: state.input_stream,
            hint_stream: state.hint_stream,
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
            self.memory_chip.current_height(),
            self.field_arithmetic_chip.current_height(),
            self.field_extension_chip.current_height(),
            self.poseidon2_chip.current_height(),
        ];
        let max_height = *heights.iter().max().unwrap();
        max_height >= self.config.max_segment_len
    }

    /// Called by VM to generate traces for current segment. Includes empty traces.
    ///
    /// Execution is handled by CPU trace generation. Stopping is triggered by should_segment()
    pub fn generate_traces(&mut self) -> Result<Vec<DenseMatrix<F>>, ExecutionError> {
        let cpu_trace = CpuChip::generate_trace(self)?;
        let mut result = vec![
            cpu_trace,
            self.program_chip.generate_trace(),
            if self.memory_chip.accesses.is_empty() {
                DenseMatrix::default(0, 0)
            } else {
                self.memory_chip.generate_trace(self.range_checker.clone())
            },
            self.range_checker.generate_trace(),
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
        Ok(result)
    }

    /// Generate Merkle proof/memory diff traces, and publish public values
    ///
    /// For now, only publishes program counter public values
    pub fn generate_commitments(&mut self) -> Result<Vec<DenseMatrix<F>>, ExecutionError> {
        // self.cpu_chip.generate_pvs();
        Ok(vec![])
    }

    pub fn get_num_chips(&self) -> usize {
        let mut result: usize = 4;
        if self.config.cpu_options().field_arithmetic_enabled {
            result += 1;
        }
        if self.config.cpu_options().field_extension_enabled {
            result += 1;
        }
        if self.config.cpu_options().poseidon2_enabled() {
            result += 1;
        }
        result
    }

    /// Returns public values for all chips in this segment
    pub fn get_pis(&self) -> Vec<Vec<F>> {
        let len = self.get_num_chips();
        let mut result: Vec<Vec<F>> = vec![vec![]; len];
        let mut cpu_public_values = self.cpu_chip.pis.clone();
        cpu_public_values.extend(self.public_values.iter().map(|x| x.unwrap_or(F::zero())));
        result[0].clone_from(&cpu_public_values);
        result
    }

    pub fn get_types(&self) -> Vec<ChipType> {
        let mut result: Vec<ChipType> = vec![
            ChipType::Cpu,
            ChipType::Program,
            ChipType::Memory,
            ChipType::RangeChecker,
        ];
        if self.config.cpu_options().field_arithmetic_enabled {
            result.push(ChipType::FieldArithmetic);
        }
        if self.config.cpu_options().field_extension_enabled {
            result.push(ChipType::FieldExtension);
        }
        if self.config.cpu_options().poseidon2_enabled() {
            result.push(ChipType::Poseidon2);
        }
        assert!(result.len() == self.get_num_chips());
        result
    }

    pub fn metrics(&mut self) -> BTreeMap<String, usize> {
        let mut metrics = BTreeMap::new();
        metrics.insert(
            "memory_chip_accesses".to_string(),
            self.memory_chip.accesses.len(),
        );
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
        metrics.insert("input_stream_len".to_string(), self.input_stream.len());
        metrics
    }
}

/// Global function to get chips from a segment
pub fn get_chips<const WORD_SIZE: usize, SC: StarkGenericConfig>(
    segment: ExecutionSegment<WORD_SIZE, Val<SC>>,
    inclusion_mask: &[bool],
) -> Vec<Box<dyn AnyRap<SC>>>
where
    Val<SC>: PrimeField32,
{
    let num_chips = segment.get_num_chips();
    let mut result: Vec<Box<dyn AnyRap<SC>>> = vec![
        Box::new(segment.cpu_chip.air),
        Box::new(segment.program_chip.air),
        Box::new(segment.memory_chip.air),
        Box::new(segment.range_checker.air),
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

    assert!(result.len() == num_chips);

    inclusion_mask
        .iter()
        .enumerate()
        .rev()
        .filter(|(_, inclusion)| !*inclusion)
        .map(|(index, _)| index)
        .for_each(|index| {
            result.remove(index);
        });

    assert!(result.len() == inclusion_mask.iter().filter(|&x| *x).count());
    result
}
