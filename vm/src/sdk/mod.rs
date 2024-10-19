use afs_stark_backend::config::{StarkGenericConfig, Val};
use ax_sdk::engine::ProofInputForTest;
use p3_field::PrimeField32;

use crate::system::{
    program::Program,
    vm::{config::VmConfig, VirtualMachine},
};

/// Generates the VM STARK circuit, in the form of AIRs and traces, but does not
/// do any proving. Output is the payload of everything the prover needs.
///
/// The output AIRs and traces are sorted by height in descending order.
pub fn gen_vm_program_test_proof_input<SC: StarkGenericConfig>(
    program: Program<Val<SC>>,
    input_stream: Vec<Vec<Val<SC>>>,
    config: VmConfig,
) -> ProofInputForTest<SC>
where
    Val<SC>: PrimeField32,
{
    cfg_if::cfg_if! {
        if #[cfg(feature = "bench-metrics")] {
            // Run once with metrics collection enabled, which can improve runtime performance
            let mut config = config;
            config.collect_metrics = true;
            {
                let vm = VirtualMachine::new(config.clone()).with_input_stream(input_stream.clone());
                vm.execute(program.clone()).unwrap();
            }
            // Run again with metrics collection disabled and measure trace generation time
            config.collect_metrics = false;
            let start = std::time::Instant::now();
        }
    }

    let vm = VirtualMachine::new(config).with_input_stream(input_stream);

    let mut result = vm.execute_and_generate(program).unwrap();
    assert_eq!(
        result.per_segment.len(),
        1,
        "only proving one segment for now"
    );

    let result = result.per_segment.pop().unwrap();
    #[cfg(feature = "bench-metrics")]
    {
        metrics::gauge!("trace_gen_time_ms").set(start.elapsed().as_millis() as f64);
    }

    ProofInputForTest {
        per_air: result.into_air_proof_input_vec(),
    }
}
