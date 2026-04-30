#![cfg(feature = "rvr")]

//! Deferral extension integration tests.
//!
//! Builds guest programs that use deferral, transpiles them with the deferral
//! transpiler extension, compiles with rvr using the deferral rvr extension,
//! executes (pre-computed data is passed via I/O state), and compares against
//! the OpenVM interpreter.
//!
//! Test matrix ({single, multiple} × {Pure, MeteredCost, Metered}):
//!   - "single"   (1 deferral circuit, 3 calls) × Pure / MeteredCost / Metered
//!   - "multiple" (2 circuits, cross-circuit isolation) × Pure / MeteredCost / Metered

use std::{collections::VecDeque, path::PathBuf, process::Command, slice::from_ref, sync::Arc};

use eyre::Result;
use openvm_circuit::{
    arch::{
        deferral::{DeferralResult, DeferralState},
        execution_mode::{MeteredCostCtx, MeteredCtx},
        rvr as rvr_openvm, Streams, SystemConfig, VirtualMachine, VmExecutionConfig,
    },
    utils::TestStarkEngine,
};
use openvm_deferral_circuit::{
    DeferralExtension, DeferralFn, Rv32DeferralBuilder, Rv32DeferralConfig,
};
use openvm_deferral_transpiler::DeferralTranspilerExtension;
use openvm_instructions::{exe::VmExe, LocalOpcode, DEFERRAL_AS};
use openvm_platform::memory::MEM_SIZE;
use openvm_rv32im_circuit::{Rv32I, Rv32Io, Rv32M};
use openvm_rv32im_transpiler::*;
use openvm_sdk::{
    config::{AggregationSystemParams, DEFAULT_APP_L_SKIP},
    Sdk, StdIn,
};
use openvm_stark_backend::{
    codec::Encode,
    keygen::types::MultiStarkProvingKey,
    p3_field::{PrimeCharacteristicRing, PrimeField32},
    StarkEngine, StarkProtocolConfig,
};
use openvm_stark_sdk::{
    config::{
        app_params_with_100_bits_security,
        baby_bear_poseidon2::{BabyBearPoseidon2Config, DIGEST_SIZE},
    },
    p3_baby_bear::BabyBear,
};
use openvm_toolchain_tests::build_example_program_at_path;
use openvm_transpiler::{elf::Elf, transpiler::Transpiler, FromElf};
use openvm_verify_stark_circuit::extension::{
    get_deferral_state, get_raw_deferral_results, verify_stark_deferral_fn,
};
use openvm_verify_stark_host::vk::VmStarkVerifyingKey;
use rvr_openvm::DeferralData;
use rvr_openvm_ext_deferral::{DeferralCircuitInputs, DeferralRvrExtension};
use rvr_openvm_lift::ExtensionRegistry;
use rvr_openvm_test_utils::{self as utils, workspace_root};

type F = BabyBear;
type Engine = TestStarkEngine;

// ── Constants matching the guest programs ──────────────────────────────────────

const INPUT_COMMIT_0: [u8; 32] = [0x11; 32];
const INPUT_COMMIT_1: [u8; 32] = [0x22; 32];
const INPUT_COMMIT_2: [u8; 32] = [0x33; 32];

const INPUT_RAW_0: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
const INPUT_RAW_1: [u8; 8] = [8, 7, 6, 5, 4, 3, 2, 1];
const INPUT_RAW_2: [u8; 8] = [9, 9, 9, 9, 9, 9, 9, 9];

// ── Shared helpers ────────────────────────────────────────────────────────────

fn make_deferral_fn(idx: usize) -> rvr_openvm_ext_deferral::DeferralFnBox {
    Box::new(move |input_raw: &[u8]| {
        let mut prefix_sum = 0u16;
        input_raw
            .iter()
            .map(|&byte| {
                prefix_sum += byte as u16;
                (prefix_sum + idx as u16) as u8
            })
            .collect()
    })
}

fn make_deferral_fns(num_deferrals: usize) -> Vec<Arc<DeferralFn>> {
    (0..num_deferrals)
        .map(|idx| Arc::new(DeferralFn::new(make_deferral_fn(idx))))
        .collect()
}

fn make_commits(num_deferrals: usize) -> Vec<[u8; 32]> {
    (0..num_deferrals)
        .map(|_| {
            let f_commit: [F; DIGEST_SIZE] = [F::ONE; DIGEST_SIZE];
            let mut bytes = [0u8; 32];
            for (i, f) in f_commit.iter().enumerate() {
                bytes[i * 4..(i + 1) * 4].copy_from_slice(&f.to_unique_u32().to_le_bytes());
            }
            bytes
        })
        .collect()
}

fn make_deferral_extension(num_deferrals: usize) -> DeferralExtension {
    let fns = make_deferral_fns(num_deferrals);
    let commits = make_commits(num_deferrals);
    DeferralExtension::new(fns, commits)
}

fn make_config(num_deferrals: usize) -> Rv32DeferralConfig {
    let mut system = SystemConfig::default();
    system.memory_config.addr_spaces[DEFERRAL_AS as usize].num_cells = 1 << 25;
    Rv32DeferralConfig {
        system,
        rv32i: Rv32I,
        rv32m: Rv32M::default(),
        io: Rv32Io,
        deferral: make_deferral_extension(num_deferrals),
    }
}

/// Circuit inputs for the "single" guest program: 1 circuit, 3 input/output pairs.
fn single_circuit_inputs() -> Vec<DeferralCircuitInputs> {
    vec![DeferralCircuitInputs {
        func: make_deferral_fn(0),
        inputs: vec![
            (INPUT_COMMIT_0.to_vec(), INPUT_RAW_0.to_vec()),
            (INPUT_COMMIT_1.to_vec(), INPUT_RAW_1.to_vec()),
            (INPUT_COMMIT_2.to_vec(), INPUT_RAW_2.to_vec()),
        ],
    }]
}

/// Circuit inputs for the "multiple" guest program: 2 circuits.
/// Circuit 0: 3 inputs. Circuit 1: 1 input (same INPUT_COMMIT_0, different function).
fn multiple_circuit_inputs() -> Vec<DeferralCircuitInputs> {
    vec![
        DeferralCircuitInputs {
            func: make_deferral_fn(0),
            inputs: vec![
                (INPUT_COMMIT_0.to_vec(), INPUT_RAW_0.to_vec()),
                (INPUT_COMMIT_1.to_vec(), INPUT_RAW_1.to_vec()),
                (INPUT_COMMIT_2.to_vec(), INPUT_RAW_2.to_vec()),
            ],
        },
        DeferralCircuitInputs {
            func: make_deferral_fn(1),
            inputs: vec![(INPUT_COMMIT_0.to_vec(), INPUT_RAW_0.to_vec())],
        },
    ]
}

fn make_single_streams() -> Streams<F> {
    let mut state = DeferralState::new(Vec::<DeferralResult>::new());
    state.store_input(INPUT_COMMIT_0.to_vec(), INPUT_RAW_0.to_vec());
    state.store_input(INPUT_COMMIT_1.to_vec(), INPUT_RAW_1.to_vec());
    state.store_input(INPUT_COMMIT_2.to_vec(), INPUT_RAW_2.to_vec());
    Streams {
        deferrals: vec![state],
        ..Default::default()
    }
}

fn make_multiple_streams() -> Streams<F> {
    let mut state0 = DeferralState::new(Vec::<DeferralResult>::new());
    state0.store_input(INPUT_COMMIT_0.to_vec(), INPUT_RAW_0.to_vec());
    state0.store_input(INPUT_COMMIT_1.to_vec(), INPUT_RAW_1.to_vec());
    state0.store_input(INPUT_COMMIT_2.to_vec(), INPUT_RAW_2.to_vec());

    let mut state1 = DeferralState::new(Vec::<DeferralResult>::new());
    state1.store_input(INPUT_COMMIT_0.to_vec(), INPUT_RAW_0.to_vec());

    Streams {
        deferrals: vec![state0, state1],
        ..Default::default()
    }
}

fn deferral_programs_dir() -> PathBuf {
    workspace_root().join("extensions/deferral/tests/programs")
}

fn transpile_with_deferral(
    elf: openvm_transpiler::elf::Elf,
    def_circuit_commits: Vec<[u8; 32]>,
) -> Result<VmExe<F>> {
    Ok(VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension)
            .with_extension(DeferralTranspilerExtension::new(def_circuit_commits)),
    )?)
}

fn build_deferral_staticlib() -> PathBuf {
    let deferral_ffi_crate = workspace_root().join("extensions/deferral/rvr/ffi");

    let output = Command::new("cargo")
        .args(["build", "--release"])
        .current_dir(&deferral_ffi_crate)
        .output()
        .expect("failed to run cargo build for deferral-ffi extension");

    if !output.status.success() {
        panic!(
            "Failed to build deferral-ffi staticlib:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let lib_path = workspace_root().join("target/release/librvr_openvm_ext_deferral_ffi.a");
    assert!(
        lib_path.exists(),
        "Deferral FFI staticlib not found at {}",
        lib_path.display()
    );
    lib_path
}

fn build_deferral_exe(config: &Rv32DeferralConfig, example: &str) -> Result<VmExe<F>> {
    let elf = build_example_program_at_path(deferral_programs_dir(), example, config)?;
    transpile_with_deferral(elf, config.deferral.def_circuit_commits.clone())
}

// ── Deferral test harness ─────────────────────────────────────────────────────

/// Holds VM + keygen state for deferral tests. Unlike the generic VmTestHarness,
/// this handles deferral-specific streams and circuit input setup.
struct DeferralTestHarness {
    config: Rv32DeferralConfig,
    vm: VirtualMachine<Engine, Rv32DeferralBuilder>,
    pk: MultiStarkProvingKey<BabyBearPoseidon2Config>,
    air_idx: Vec<usize>,
}

impl DeferralTestHarness {
    fn new(config: Rv32DeferralConfig) -> Result<Self> {
        let engine = Engine::new(openvm_stark_backend::SystemParams::new_for_testing(21));
        let (vm, pk) = VirtualMachine::<Engine, _>::new_with_keygen(
            engine,
            Rv32DeferralBuilder,
            config.clone(),
        )?;
        let air_idx = vm.executor_idx_to_air_idx();
        Ok(Self {
            config,
            vm,
            pk,
            air_idx,
        })
    }

    fn inventory(
        &self,
    ) -> Result<
        openvm_circuit::arch::ExecutorInventory<
            <Rv32DeferralConfig as VmExecutionConfig<F>>::Executor,
        >,
    > {
        Ok(VmExecutionConfig::<F>::create_executors(&self.config)?)
    }

    fn build_ext(&self) -> DeferralRvrExtension {
        let inventory = self.inventory().unwrap();
        let ctx = utils::rvr_extension_ctx(&inventory, &self.air_idx);
        DeferralRvrExtension::new::<F>(&ctx, build_deferral_staticlib()).unwrap()
    }

    fn build_ext_pure() -> DeferralRvrExtension {
        DeferralRvrExtension::new_pure(build_deferral_staticlib())
    }

    fn make_deferral_data(circuit_inputs: &[DeferralCircuitInputs]) -> DeferralData {
        let precomputed = rvr_openvm_ext_deferral::precompute::<F>(circuit_inputs);
        DeferralData {
            call_entries: precomputed.call_entries,
            output_entries: precomputed.output_entries,
        }
    }

    /// Pure mode: compare PC + registers.
    fn compare_pure(
        &self,
        label: &str,
        exe: &VmExe<F>,
        streams: Streams<F>,
        circuit_inputs: Vec<DeferralCircuitInputs>,
    ) -> Result<()> {
        let input_stream = streams.input_stream.clone();

        // OpenVM reference
        let interpreter = self.vm.executor().instance(exe)?;
        let interp_state = interpreter.execute(streams, None)?;

        // rvr compile + execute
        let deferral = Self::make_deferral_data(&circuit_inputs);
        let ext = Self::build_ext_pure();
        let mut extensions = ExtensionRegistry::new();
        extensions.register(ext);
        let compiled = rvr_openvm::compile_with_extensions(exe, &extensions)?;

        let rvr_result = rvr_openvm::execute(&compiled, exe, input_stream, deferral)?;

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
            let interp_u32 =
                u32::from_le_bytes(unsafe { interp_state.memory.read::<u8, 4>(1, r * 4) });
            assert_eq!(
                rvr_val, interp_u32,
                "[{label}] register x{r} mismatch: rvr={rvr_val:#x}, interp={interp_u32:#x}",
            );
        }
        Ok(())
    }

    /// MeteredCost mode: compare instret + cost.
    fn compare_metered_cost(
        &self,
        label: &str,
        exe: &VmExe<F>,
        streams: Streams<F>,
        circuit_inputs: Vec<DeferralCircuitInputs>,
    ) -> Result<()> {
        let ctx: MeteredCostCtx = self.vm.build_metered_cost_ctx();

        // OpenVM reference
        let instance = self
            .vm
            .executor()
            .metered_cost_instance(exe, &self.air_idx)?;
        let (ref_ctx, _) = instance.execute_metered_cost(streams, ctx.clone())?;

        // rvr compile + execute
        let deferral = Self::make_deferral_data(&circuit_inputs);
        let ext = self.build_ext();
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
        let mut extensions = ExtensionRegistry::new();
        extensions.register(ext);
        let compiled = rvr_openvm::compile_metered_cost_with_extensions(exe, &extensions, &chips)?;

        let rvr_result = rvr_openvm::execute_metered_cost(
            &compiled,
            exe,
            VecDeque::new(),
            metered_cost_config,
            deferral,
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

    /// Metered mode: compare per-chip trace heights across segments.
    fn compare_metered(
        &self,
        label: &str,
        exe: &VmExe<F>,
        streams: Streams<F>,
        circuit_inputs: Vec<DeferralCircuitInputs>,
    ) -> Result<()> {
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
        let (ref_segments, _) = metered_interpreter.execute_metered(streams, metered_ctx)?;

        // rvr compile + execute
        let deferral = Self::make_deferral_data(&circuit_inputs);
        let ext = self.build_ext();
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
        let mut extensions = ExtensionRegistry::new();
        extensions.register(ext);
        let compiled = rvr_openvm::compile_metered_with_extensions(exe, &extensions, &chips)?;

        let saved_config = trace_config.clone();
        let rvr_result =
            rvr_openvm::execute_metered(&compiled, exe, VecDeque::new(), trace_config, deferral)?;

        utils::assert_segments(label, &ref_segments, &rvr_result.segments, &saved_config);
        Ok(())
    }
}

// ── Pure execution tests ────────────────────────────────────────────────────

/// Single deferral circuit: 3 CALL + 3 OUTPUT ops with def_idx=0.
#[test]
fn test_deferral_single_pure() -> Result<()> {
    let harness = DeferralTestHarness::new(make_config(1))?;
    let exe = build_deferral_exe(&harness.config, "single")?;
    harness.compare_pure(
        "single_pure",
        &exe,
        make_single_streams(),
        single_circuit_inputs(),
    )
}

/// Two deferral circuits: tests cross-circuit isolation. Same INPUT_COMMIT_0
/// is used with def_idx=0 and def_idx=1, producing different outputs because
/// the deferral functions differ (prefix_sum + 0 vs prefix_sum + 1).
/// Each circuit has independent accumulators in DEFERRAL_AS.
#[test]
fn test_deferral_multiple_pure() -> Result<()> {
    let harness = DeferralTestHarness::new(make_config(2))?;
    let exe = build_deferral_exe(&harness.config, "multiple")?;
    harness.compare_pure(
        "multiple_pure",
        &exe,
        make_multiple_streams(),
        multiple_circuit_inputs(),
    )
}

// ── Metered cost execution tests ────────────────────────────────────────────

/// Validates that per-chip cost accounting is correct for deferral.
/// CALL contributes: 1 row (call chip, via trace_pc) + 2 rows (poseidon2, via trace_chip).
/// OUTPUT contributes: variable rows (output chip) + variable rows (poseidon2).
#[test]
fn test_deferral_single_metered_cost() -> Result<()> {
    let harness = DeferralTestHarness::new(make_config(1))?;
    let exe = build_deferral_exe(&harness.config, "single")?;
    harness.compare_metered_cost(
        "single_metered_cost",
        &exe,
        make_single_streams(),
        single_circuit_inputs(),
    )
}

/// Two circuits metered cost: validates cost accounting with multiple def_idx
/// and independent accumulator updates.
#[test]
fn test_deferral_multiple_metered_cost() -> Result<()> {
    let harness = DeferralTestHarness::new(make_config(2))?;
    let exe = build_deferral_exe(&harness.config, "multiple")?;
    harness.compare_metered_cost(
        "multiple_metered_cost",
        &exe,
        make_multiple_streams(),
        multiple_circuit_inputs(),
    )
}

// ── Metered (segmentation) execution tests ──────────────────────────────────

/// Validates per-chip trace heights match the OpenVM interpreter, including:
/// - Variable-height OUTPUT chip (num_rows = output_len / DIGEST_SIZE + 1)
/// - Poseidon2 periphery chip heights (2 per CALL, num_rows per OUTPUT)
/// - DEFERRAL_AS memory access page tracking for segmentation
#[test]
fn test_deferral_single_metered() -> Result<()> {
    let harness = DeferralTestHarness::new(make_config(1))?;
    let exe = build_deferral_exe(&harness.config, "single")?;
    harness.compare_metered(
        "single_metered",
        &exe,
        make_single_streams(),
        single_circuit_inputs(),
    )
}

/// Two circuits metered: validates per-chip trace heights with multiple
/// def_idx, including independent DEFERRAL_AS page tracking per circuit.
#[test]
fn test_deferral_multiple_metered() -> Result<()> {
    let harness = DeferralTestHarness::new(make_config(2))?;
    let exe = build_deferral_exe(&harness.config, "multiple")?;
    harness.compare_metered(
        "multiple_metered",
        &exe,
        make_multiple_streams(),
        multiple_circuit_inputs(),
    )
}

// ── Verify-stark deferral tests ────────────────────────────────────────────
//
// Exercises the `verify_stark` guest library through the deferral extension.
// Uses a mock deferral function that produces output in the same format as
// `verify_stark_deferral_fn` (exe_commit || vk_commit || user_pvs), so the
// guest program's `verify_stark::<0>()` call succeeds.

/// Serialize a value the same way `openvm_sdk::StdIn::write` does.
fn serialize_for_stdin<T: serde::Serialize>(data: &T) -> Vec<F> {
    let words = openvm::serde::to_vec(data).unwrap();
    let bytes: Vec<u8> = words.into_iter().flat_map(|w| w.to_le_bytes()).collect();
    bytes.iter().map(|&b| F::from_u8(b)).collect()
}

fn transpile_verify_stark_elf(def_circuit_commits: Vec<[u8; 32]>) -> Result<VmExe<F>> {
    let elf = Elf::decode(
        include_bytes!("../../../sdk/programs/examples/verify-stark.elf"),
        MEM_SIZE as u32,
    )?;
    transpile_with_deferral(elf, def_circuit_commits)
}

/// End-to-end test: generates a real fibonacci proof via the SDK, computes
/// deferral state using `verify_stark_deferral_fn`, then runs the verify-stark
/// guest program (which calls `verify_stark::<0>()`) and compares RVR vs OpenVM.
#[test]
fn test_verify_stark_deferral_pure() -> Result<()> {
    // 1. Generate a real fibonacci proof via SDK
    let n_stack = 19;
    let app_params = app_params_with_100_bits_security(DEFAULT_APP_L_SKIP + n_stack);
    let agg_params = AggregationSystemParams::default();

    let fib_elf = Elf::decode(
        include_bytes!("../../../sdk/programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;

    let fib_sdk = Sdk::riscv32(app_params, agg_params);
    let fib_exe = fib_sdk.convert_to_exe(fib_elf)?;

    let mut fib_stdin = StdIn::default();
    fib_stdin.write(&100u64);

    let (fib_proof, fib_baseline) = fib_sdk.prove(fib_exe, fib_stdin, &[])?;

    // 2. Compute deferral state and stdin values from the real proof
    let fib_vk = VmStarkVerifyingKey {
        mvk: fib_sdk.agg_vk().as_ref().clone(),
        baseline: fib_baseline,
    };

    let raw_results = get_raw_deferral_results(&fib_vk, from_ref(&fib_proof))?;
    assert_eq!(raw_results.len(), 1);
    let input_commit: [u8; 32] = raw_results[0].input.clone().try_into().unwrap();
    let output_raw = &raw_results[0].output_raw;
    let app_exe_commit: [u8; 32] = output_raw[..32].try_into().unwrap();
    let app_vk_commit: [u8; 32] = output_raw[32..64].try_into().unwrap();
    let user_public_values = output_raw[64..].to_vec();

    let deferral_state = get_deferral_state(&fib_vk, from_ref(&fib_proof), 0)?;

    // 3. Build config with real verify_stark_deferral_fn
    let def_fn = Arc::new(DeferralFn::new(verify_stark_deferral_fn));
    let dummy_vk_commit = make_commits(1).pop().unwrap();
    let deferral = DeferralExtension::new(vec![def_fn], vec![dummy_vk_commit]);
    let mut system = SystemConfig::default();
    system.memory_config.addr_spaces[DEFERRAL_AS as usize].num_cells = 1 << 25;
    let config = Rv32DeferralConfig {
        system,
        rv32i: Rv32I,
        rv32m: Rv32M::default(),
        io: Rv32Io,
        deferral,
    };
    let harness = DeferralTestHarness::new(config)?;

    // 4. Build stdin and streams
    let streams = Streams {
        input_stream: VecDeque::from(vec![
            serialize_for_stdin(&app_exe_commit),
            serialize_for_stdin(&app_vk_commit),
            serialize_for_stdin(&user_public_values),
            serialize_for_stdin(&input_commit),
        ]),
        deferrals: vec![deferral_state],
        ..Default::default()
    };

    // 5. Transpile and compare — RVR uses verify_stark_deferral_fn on encoded proof bytes
    let exe = transpile_verify_stark_elf(harness.config.deferral.def_circuit_commits.clone())?;
    let encoded_proof = fib_proof.encode_to_vec().unwrap();
    let circuit_inputs = vec![DeferralCircuitInputs {
        func: Box::new(verify_stark_deferral_fn),
        inputs: vec![(input_commit.to_vec(), encoded_proof)],
    }];
    harness.compare_pure("verify_stark", &exe, streams, circuit_inputs)
}
