use ax_stark_sdk::{
    ax_stark_backend::{
        config::{Com, Domain, PcsProof, PcsProverData, StarkGenericConfig, Val},
        p3_field::PrimeField32,
    },
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    engine::StarkFriEngine,
};
use axvm_circuit::arch::{instructions::exe::AxVmExe, VirtualMachine, VmConfig};
use axvm_transpiler::{axvm_platform::memory::MEM_SIZE, elf::Elf};
use wasm_bindgen::prelude::*;
use web_sys::console::{time_end_with_label, time_with_label};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

    #[wasm_bindgen(js_namespace = console)]
    fn error(s: &str);
}

#[wasm_bindgen]
pub fn prove_fib() -> Result<(), JsValue> {
    init_panic_hook();
    log("start");
    prove().map_err(|e| e.to_string().into())
}

const ELF: &[u8] = include_bytes!("../fibonacci-elf");

fn prove() -> eyre::Result<()> {
    log("decoding elf");
    let elf = Elf::decode(ELF, MEM_SIZE as u32)?;
    let app_log_blowup = 1;
    log("create engine");
    let engine = BabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(app_log_blowup),
    );
    log("create config");
    let config = VmConfig::rv32im();
    log("start bench");
    bench_from_exe(engine, config, elf, vec![])?;
    Ok(())
}

pub fn bench_from_exe<SC, E>(
    engine: E,
    config: VmConfig,
    exe: impl Into<AxVmExe<Val<SC>>>,
    input_stream: Vec<Vec<Val<SC>>>,
) -> eyre::Result<()>
where
    SC: StarkGenericConfig,
    E: StarkFriEngine<SC>,
    Val<SC>: PrimeField32,
    Domain<SC>: Send + Sync,
    PcsProverData<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Challenge: Send + Sync,
    PcsProof<SC>: Send + Sync,
{
    let exe = exe.into();
    // 1. Generate proving key from config.
    let vm = VirtualMachine::new(engine, config);
    let pk = vm.keygen();
    log("keygen done");
    // 2. Commit to the exe by generating cached trace for program.
    let committed_exe = vm.commit_exe(exe);
    time_with_label("total_proving_time");
    // 3. Executes runtime again without metric collection and generate trace.
    time_with_label("execute_and_trace_gen_time");
    let results = vm.execute_and_generate_with_cached_program(committed_exe, input_stream)?;
    time_end_with_label("execute_and_trace_gen_time");
    // 4. Generate STARK proofs for each segment (segmentation is determined by `config`), with timer.
    // vm.prove will emit metrics for proof time of each segment
    let proofs = vm.prove(&pk, results);
    time_end_with_label("total_proving_time");

    // 6. Verify STARK proofs.
    let vk = pk.get_vk();
    vm.verify(&vk, proofs.clone()).expect("Verification failed");

    Ok(())
}

#[wasm_bindgen(js_name = initPanicHook)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}
