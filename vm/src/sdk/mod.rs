use ax_sdk::engine::StarkForTest;
use p3_field::PrimeField32;
use p3_uni_stark::{StarkGenericConfig, Val};

use crate::{
    program::Program,
    vm::{config::VmConfig, VirtualMachine},
};

/// Generates the VM STARK circuit, in the form of AIRs and traces, but does not
/// do any proving. Output is the payload of everything the prover needs.
///
/// The output AIRs and traces are sorted by height in descending order.
pub fn gen_vm_program_stark_for_test<SC: StarkGenericConfig>(
    program: Program<Val<SC>>,
    input_stream: Vec<Vec<Val<SC>>>,
    config: VmConfig,
) -> StarkForTest<SC>
where
    Val<SC>: PrimeField32,
{
    cfg_if::cfg_if! {
        if #[cfg(feature = "bench-metrics")] {
            let start = std::time::Instant::now();
            let mut config= config;
            config.collect_metrics = true;
        }
    }

    let vm = VirtualMachine::new(config, program, input_stream);

    let mut result = vm.execute_and_generate().unwrap();
    assert_eq!(
        result.segment_results.len(),
        1,
        "only proving one segment for now"
    );

    let result = result.segment_results.pop().unwrap();
    #[cfg(feature = "bench-metrics")]
    {
        result.metrics.emit();
        metrics::gauge!("trace_gen_time_ms", "stark" => "vm")
            .set(start.elapsed().as_millis() as f64);
    }

    StarkForTest {
        any_raps: result.airs.into_iter().map(|x| x.into()).collect(),
        traces: result.traces,
        pvs: result.public_values,
    }
    .sort_by_height_desc()
}
