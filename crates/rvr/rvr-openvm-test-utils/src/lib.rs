//! Shared test helpers: compile with rvr, compare against OpenVM interpreter.

#![allow(dead_code)]

use std::{
    collections::VecDeque,
    fs,
    path::{Path, PathBuf},
};

use eyre::Result;
use openvm_circuit::arch::{
    execution_mode::{MeteredCostCtx, MeteredCtx, Segment},
    rvr as rvr_openvm, Executor, ExecutorInventory, MeteredExecutor, Streams, SystemConfig,
    VirtualMachine, VmExecutionConfig,
};
use openvm_instructions::{
    exe::VmExe,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
    LocalOpcode,
};
use openvm_platform::memory::MEM_SIZE;
use openvm_rv32im_circuit::{Rv32IConfig, Rv32ImConfig, Rv32ImCpuBuilder};
use openvm_rv32im_transpiler::{
    Rv32HintStoreOpcode, Rv32ITranspilerExtension, Rv32IoTranspilerExtension,
    Rv32MTranspilerExtension,
};
use openvm_stark_backend::{
    keygen::types::MultiStarkProvingKey, p3_field::PrimeCharacteristicRing, StarkEngine,
    StarkProtocolConfig,
};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2CpuEngine},
    p3_baby_bear::BabyBear,
};
use openvm_toolchain_tests::build_example_program_at_path_with_features;
use openvm_transpiler::{elf::Elf, transpiler::Transpiler, FromElf};
use rvr_openvm_lift::{ExtensionRegistry, RvrExtensionCtx};

pub type F = BabyBear;
type Engine = BabyBearPoseidon2CpuEngine;

pub const DATA: &str = "crates/toolchain/tests/tests/data";
pub const RVTEST: &str = "crates/toolchain/tests/rv32im-test-vectors/tests";
pub const BENCH: &str = "benchmarks/guest";

// ── Execution / comparison mode ─────────────────────────────────────────────

pub enum ExecutionMode {
    /// Compare PC + registers against OpenVM interpreter.
    Pure,
    /// Compare instret + cost against OpenVM metered-cost interpreter.
    MeteredCost,
    /// Compare segments against OpenVM metered interpreter using relaxed comparison
    /// (total instret, per-chip totals, limit checks) because block-level segmentation
    /// checks produce different boundaries than OpenVM's per-instruction checks.
    Metered,
    /// Same as Metered but with a custom max_trace_height for multi-segment testing.
    MeteredMultiseg { max_trace_height: u32 },
}

// ── Path helpers ────────────────────────────────────────────────────────────

// `rvr-openvm-test-utils` lives at `<workspace>/crates/rvr/rvr-openvm-test-utils/`,
// three levels below the workspace root.
pub fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

pub fn openvm_root() -> PathBuf {
    workspace_root()
}

pub fn rv32im_programs_dir() -> PathBuf {
    openvm_root().join("extensions/rv32im/tests/programs")
}

pub fn resolve_elf_path(elf_path: &str) -> Result<PathBuf> {
    let openvm_path = openvm_root().join(elf_path);
    if openvm_path.exists() {
        return Ok(openvm_path);
    }

    if let Ok(suffix) =
        Path::new(elf_path).strip_prefix("crates/toolchain/tests/rv32im-test-vectors/tests")
    {
        let rvr_path = workspace_root().join("rvr/bin/riscv-tests").join(suffix);
        if rvr_path.exists() {
            return Ok(rvr_path);
        }
    }

    eyre::bail!(
        "could not resolve ELF path `{elf_path}` from `{}`",
        openvm_root().display()
    )
}

// ── Transpilation ───────────────────────────────────────────────────────────

pub fn transpile(elf: Elf) -> Result<VmExe<F>> {
    Ok(VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension),
    )?)
}

/// Common input for the `read` guest program.
pub fn read_program_input() -> Vec<F> {
    let serialized_words: Vec<u32> = vec![42, 4, 0, 1, 2, 3];
    serialized_words
        .into_iter()
        .flat_map(|w| w.to_le_bytes())
        .map(F::from_u8)
        .collect()
}

// ── Shared helpers ─────────────────────────────────────────────────────────

/// Build an `RvrExtensionCtx` from an executor inventory and AIR index mapping.
pub fn rvr_extension_ctx<E>(
    inventory: &ExecutorInventory<E>,
    air_idx: &[usize],
) -> RvrExtensionCtx {
    let opcode_to_executor_idx = inventory
        .instruction_lookup
        .iter()
        .map(|(opcode, executor_idx)| (*opcode, *executor_idx as usize));
    RvrExtensionCtx::new(opcode_to_executor_idx, air_idx.to_vec())
}

// ── Generic VM test harness ─────────────────────────────────────────────────

/// Holds keygen results and an extension registry. Provides `compare` to run
/// rvr against the OpenVM interpreter for any `ExecutionMode`.
pub struct VmTestHarness<VB: openvm_circuit::arch::VmBuilder<Engine>> {
    config: VB::VmConfig,
    vm: VirtualMachine<Engine, VB>,
    pk: MultiStarkProvingKey<BabyBearPoseidon2Config>,
    air_idx: Vec<usize>,
    pub extensions: ExtensionRegistry<F>,
}

impl<VB> VmTestHarness<VB>
where
    VB: openvm_circuit::arch::VmBuilder<Engine>,
    VB::VmConfig: Clone,
    <VB::VmConfig as VmExecutionConfig<F>>::Executor: Executor<F> + MeteredExecutor<F>,
{
    pub fn new(config: VB::VmConfig, builder: VB) -> Result<Self> {
        let engine = Engine::new(openvm_stark_backend::SystemParams::new_for_testing(21));
        let (vm, pk) = VirtualMachine::new_with_keygen(engine, builder, config.clone())?;
        let air_idx = vm.executor_idx_to_air_idx();
        Ok(Self {
            config,
            vm,
            pk,
            air_idx,
            extensions: ExtensionRegistry::new(),
        })
    }

    /// Executor inventory, for building extensions that need chip mappings.
    pub fn inventory(
        &self,
    ) -> Result<ExecutorInventory<<VB::VmConfig as VmExecutionConfig<F>>::Executor>> {
        Ok(VmExecutionConfig::<F>::create_executors(&self.config)?)
    }

    /// Executor-index → AIR-index mapping, for building extensions.
    pub fn air_idx(&self) -> &[usize] {
        &self.air_idx
    }

    /// Build rvr extension context from executor inventory + AIR mapping.
    pub fn rvr_extension_ctx(&self) -> Result<RvrExtensionCtx> {
        let inventory = self.inventory()?;
        Ok(rvr_extension_ctx(&inventory, &self.air_idx))
    }

    /// Register an extension into the harness.
    pub fn register(&mut self, ext: impl rvr_openvm_lift::RvrExtension<F> + 'static) {
        self.extensions.register(ext);
    }

    /// Compile with rvr and compare against the OpenVM interpreter.
    pub fn compare(
        &self,
        label: &str,
        exe: &VmExe<F>,
        input: Vec<Vec<F>>,
        mode: ExecutionMode,
    ) -> Result<()> {
        self.compare_impl(label, exe, input, mode, false)
    }

    pub fn compare_with_full_memory(
        &self,
        label: &str,
        exe: &VmExe<F>,
        input: Vec<Vec<F>>,
        mode: ExecutionMode,
    ) -> Result<()> {
        self.compare_impl(label, exe, input, mode, true)
    }

    fn compare_impl(
        &self,
        label: &str,
        exe: &VmExe<F>,
        input: Vec<Vec<F>>,
        mode: ExecutionMode,
        compare_full_memory: bool,
    ) -> Result<()> {
        match mode {
            ExecutionMode::Pure => self.compare_pure(label, exe, input, compare_full_memory),
            ExecutionMode::MeteredCost => self.compare_metered_cost(label, exe, input),
            ExecutionMode::Metered | ExecutionMode::MeteredMultiseg { .. } => {
                self.compare_metered(label, exe, input)
            }
        }
    }

    fn make_streams(&self, input: &[Vec<F>]) -> Streams<F> {
        VecDeque::from(input.to_vec()).into()
    }

    fn compare_pure(
        &self,
        label: &str,
        exe: &VmExe<F>,
        input: Vec<Vec<F>>,
        compare_full_memory: bool,
    ) -> Result<()> {
        // OpenVM reference
        let interpreter = self.vm.executor().instance(exe)?;
        let interp_state = interpreter.execute(self.make_streams(&input), None)?;

        // rvr execution
        let input_stream = VecDeque::from(input);
        let compiled = if self.extensions.is_empty() {
            rvr_openvm::compile(exe)?
        } else {
            rvr_openvm::compile_with_extensions(exe, &self.extensions)?
        };
        let rvr_result = rvr_openvm::execute(&compiled, exe, input_stream, Default::default())?;

        // Compare PC + registers
        assert_eq!(
            rvr_result.state.pc,
            interp_state.pc(),
            "[{label}] PC mismatch: rvr={:#x}, interp={:#x}",
            rvr_result.state.pc,
            interp_state.pc()
        );
        for r in 0..32u32 {
            let rvr_val = rvr_result.state.regs[r as usize];
            let interp_u32 = u32::from_le_bytes(unsafe {
                interp_state.memory.read::<u8, 4>(RV32_REGISTER_AS, r * 4)
            });
            assert_eq!(
                rvr_val, interp_u32,
                "[{label}] register x{r} mismatch: rvr={rvr_val:#x}, interp={interp_u32:#x}",
            );
        }
        if compare_full_memory {
            assert_full_guest_memory(label, &interp_state, &rvr_result.memory);
        }
        Ok(())
    }

    fn compare_metered_cost(&self, label: &str, exe: &VmExe<F>, input: Vec<Vec<F>>) -> Result<()> {
        let ctx: MeteredCostCtx = self.vm.build_metered_cost_ctx();

        // OpenVM reference
        let instance = self
            .vm
            .executor()
            .metered_cost_instance(exe, &self.air_idx)?;
        let (ref_ctx, _) = instance.execute_metered_cost(self.make_streams(&input), ctx.clone())?;

        // rvr execution
        let inventory = self.inventory()?;
        let hint_buffer_opcode = Some(Rv32HintStoreOpcode::HINT_BUFFER.global_opcode());
        let metered_cost_config = rvr_openvm::build_metered_cost_config(
            exe,
            &inventory,
            &self.air_idx,
            &ctx.widths,
            hint_buffer_opcode,
        );
        let chips = metered_cost_config.chip_mapping();
        let compiled = if self.extensions.is_empty() {
            rvr_openvm::compile_metered_cost(exe, &chips)?
        } else {
            rvr_openvm::compile_metered_cost_with_extensions(exe, &self.extensions, &chips)?
        };
        let rvr_result = rvr_openvm::execute_metered_cost(
            &compiled,
            exe,
            VecDeque::from(input),
            metered_cost_config,
            Default::default(),
        )?;

        assert_eq!(
            rvr_result.instret, ref_ctx.instret,
            "[{label}] instret mismatch: rvr={}, openvm={}",
            rvr_result.instret, ref_ctx.instret
        );
        assert_eq!(
            rvr_result.cost, ref_ctx.cost,
            "[{label}] cost mismatch: rvr={}, openvm={}",
            rvr_result.cost, ref_ctx.cost
        );
        Ok(())
    }

    fn compare_metered(&self, label: &str, exe: &VmExe<F>, input: Vec<Vec<F>>) -> Result<()> {
        let system_config: &SystemConfig = self.config.as_ref();

        // Extract widths / interactions from PK
        let mut widths = Vec::new();
        let mut interactions = Vec::new();
        for pk in &self.pk.per_air {
            widths.push(
                pk.vk
                    .params
                    .width
                    .total_width(<Engine as StarkEngine>::SC::D_EF),
            );
            interactions.push(pk.vk.symbolic_constraints.interactions.len());
        }

        // OpenVM reference
        let metered_ctx: MeteredCtx = self.vm.build_metered_ctx(exe);
        let constant_trace_heights: Vec<Option<usize>> = metered_ctx
            .trace_heights
            .iter()
            .zip(metered_ctx.is_trace_height_constant.iter())
            .map(|(&h, &is_const)| if is_const { Some(h as usize) } else { None })
            .collect();
        let metered_interpreter = self.vm.metered_interpreter(exe)?;
        let (ref_segments, _) =
            metered_interpreter.execute_metered(self.make_streams(&input), metered_ctx)?;

        // rvr execution
        let inventory = self.inventory()?;
        let hint_buffer_opcode = Some(Rv32HintStoreOpcode::HINT_BUFFER.global_opcode());
        let trace_config = rvr_openvm::build_metered_config(
            exe,
            &inventory,
            &self.air_idx,
            &widths,
            &interactions,
            &constant_trace_heights,
            system_config,
            hint_buffer_opcode,
        );
        let chips = trace_config.chip_mapping();
        let compiled = if self.extensions.is_empty() {
            rvr_openvm::compile_metered(exe, &chips)?
        } else {
            rvr_openvm::compile_metered_with_extensions(exe, &self.extensions, &chips)?
        };
        let saved_config = trace_config.clone();
        let rvr_result = rvr_openvm::execute_metered(
            &compiled,
            exe,
            VecDeque::from(input),
            trace_config,
            Default::default(),
        )?;

        assert_segments(label, &ref_segments, &rvr_result.segments, &saved_config);
        Ok(())
    }
}

fn assert_full_guest_memory(
    label: &str,
    interp_state: &openvm_circuit::arch::VmState<F>,
    rvr_memory: &rvr_state::GuardedMemory,
) {
    for addr in 0..MEM_SIZE as u32 {
        let interp = unsafe { interp_state.memory.read::<u8, 1>(RV32_MEMORY_AS, addr) }[0];
        let rvr = unsafe { rvr_memory.read_u8(addr as usize) };
        assert_eq!(
            rvr, interp,
            "[{label}] guest memory mismatch at {addr:#x}: rvr={rvr:#04x}, interp={interp:#04x}",
        );
    }
}

/// Segment comparison for metered mode.
///
/// Because rvr checks segmentation at block granularity (not per-instruction like OpenVM),
/// segment boundaries will differ. We verify:
/// 1. Total instret matches across all segments.
/// 2. Each rvr segment is contiguous and non-empty.
/// 3. Per-chip trace height totals match for instruction-driven chips (excluding boundary, merkle
///    tree, and poseidon2 which are re-initialized at boundaries).
/// 4. No rvr segment exceeds the configured max_trace_height.
pub fn assert_segments(
    label: &str,
    ref_segments: &[Segment],
    rvr_segments: &[rvr_openvm::RvrSegment],
    config: &rvr_openvm::MeteredConfig,
) {
    // 1. Total instret
    let ref_total: u64 = ref_segments.iter().map(|s| s.num_insns).sum();
    let rvr_total: u64 = rvr_segments.iter().map(|s| s.num_insns).sum();
    assert_eq!(
        ref_total, rvr_total,
        "[{label}] total instret mismatch: openvm={ref_total}, rvr={rvr_total}"
    );

    // 2. Contiguous, non-empty segments
    let mut expected_start = 0u64;
    for (i, seg) in rvr_segments.iter().enumerate() {
        assert_eq!(
            seg.instret_start, expected_start,
            "[{label}] rvr segment {i} not contiguous: expected start={expected_start}, got={}",
            seg.instret_start
        );
        assert!(
            seg.num_insns > 0,
            "[{label}] rvr segment {i} has zero instructions"
        );
        expected_start += seg.num_insns;
    }

    // 3. Per-chip trace height totals for instruction-driven chips
    let num_chips = ref_segments[0].trace_heights.len();
    let poseidon2_idx = num_chips - 2;
    let excluded = [config.boundary_idx, config.merkle_tree_idx, poseidon2_idx];

    for chip in 0..num_chips {
        if excluded.contains(&chip) || config.is_constant[chip] {
            continue;
        }
        let ref_sum: u64 = ref_segments
            .iter()
            .map(|s| s.trace_heights[chip] as u64)
            .sum();
        let rvr_sum: u64 = rvr_segments
            .iter()
            .map(|s| s.trace_heights[chip] as u64)
            .sum();
        assert_eq!(
            ref_sum, rvr_sum,
            "[{label}] chip {chip} total trace height mismatch: openvm={ref_sum}, rvr={rvr_sum}"
        );
    }

    // 4. No segment exceeds max_trace_height
    let max_height = config.segmentation_config.limits.max_trace_height;
    for (i, seg) in rvr_segments.iter().enumerate() {
        for (chip, &h) in seg.trace_heights.iter().enumerate() {
            if !config.is_constant[chip] {
                assert!(
                    h.next_power_of_two() <= max_height,
                    "[{label}] rvr segment {i} chip {chip} trace height {h} \
                     (padded {}) exceeds max {max_height}",
                    h.next_power_of_two()
                );
            }
        }
    }
}

fn make_rv32im_config(max_trace_height: Option<u32>) -> Rv32ImConfig {
    match max_trace_height {
        None => Rv32ImConfig::default(),
        Some(h) => Rv32ImConfig {
            rv32i: Rv32IConfig {
                system: SystemConfig::default().with_max_segment_len(h as usize),
                base: Default::default(),
                io: Default::default(),
            },
            mul: Default::default(),
        },
    }
}

// ── High-level test helpers (Rv32Im convenience wrappers) ───────────────────

/// Compile with rvr and compare against the OpenVM interpreter for the given mode.
pub fn compile_and_compare(
    label: &str,
    exe: &VmExe<F>,
    input: Vec<Vec<F>>,
    mode: ExecutionMode,
) -> Result<()> {
    let config = match &mode {
        ExecutionMode::MeteredMultiseg { max_trace_height } => {
            make_rv32im_config(Some(*max_trace_height))
        }
        _ => Rv32ImConfig::default(),
    };
    let harness = VmTestHarness::new(config, Rv32ImCpuBuilder)?;
    harness.compare(label, exe, input, mode)
}

/// Load a prebuilt ELF, transpile, and compare against interpreter.
pub fn run_and_compare(elf_path: &str, mode: ExecutionMode) -> Result<()> {
    let data = fs::read(resolve_elf_path(elf_path)?)?;
    let exe = transpile(Elf::decode(&data, MEM_SIZE as u32)?)?;
    compile_and_compare(elf_path, &exe, vec![], mode)
}

/// Build a guest program from source, transpile, and compare against interpreter.
pub fn build_and_compare(
    example_name: &str,
    features: &[&str],
    input: Vec<Vec<F>>,
    mode: ExecutionMode,
) -> Result<()> {
    let elf = build_example_program_at_path_with_features(
        rv32im_programs_dir(),
        example_name,
        features.to_vec(),
        &Rv32ImConfig::default(),
    )?;
    let exe = transpile(elf)?;
    compile_and_compare(example_name, &exe, input, mode)
}
