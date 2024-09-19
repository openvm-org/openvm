use std::collections::{BTreeMap, HashMap, HashSet};

use afs_compiler::{asm::AsmBuilder, ir::Felt};
use afs_recursion::{
    halo2::testing_utils::run_evm_verifier_e2e_test,
    testing_utils::{
        gen_vm_program_stark_for_test, inner::build_verification_program, StarkForTest,
    },
};
use ax_sdk::{
    config::{
        baby_bear_poseidon2::BabyBearPoseidon2Engine,
        fri_params::fri_params_with_80_bits_of_security,
    },
    engine::StarkFriEngine,
};
use metrics::{key_var, KeyName, Label};
use metrics_tracing_context::{MetricsLayer, TracingContextLayer};
use metrics_util::{
    debugging::{DebugValue, DebuggingRecorder, Snapshot},
    layers::Layer,
    CompositeKey, MetricKind,
    MetricKind::Gauge,
};
use p3_baby_bear::BabyBear;
use p3_commit::PolynomialSpace;
use p3_field::{extension::BinomialExtensionField, AbstractField};
use p3_uni_stark::{Domain, StarkGenericConfig};
use stark_vm::{program::Program, vm::config::VmConfig};
use tracing::{info, info_span, Level};
use tracing_forest::ForestLayer;
use tracing_subscriber::{layer::SubscriberExt, EnvFilter, Registry};

fn fibonacci_program(a: u32, b: u32, n: u32) -> Program<BabyBear> {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    let mut builder = AsmBuilder::<F, EF>::default();

    let prev: Felt<_> = builder.constant(F::from_canonical_u32(a));
    let next: Felt<_> = builder.constant(F::from_canonical_u32(b));

    for _ in 2..n {
        let tmp: Felt<_> = builder.uninit();
        builder.assign(&tmp, next);
        builder.assign(&next, prev + next);
        builder.assign(&prev, tmp);
    }

    builder.halt();

    builder.compile_isa()
}

pub(crate) fn fibonacci_program_stark_for_test<SC: StarkGenericConfig>(
    a: u32,
    b: u32,
    n: u32,
) -> StarkForTest<SC>
where
    Domain<SC>: PolynomialSpace<Val = BabyBear>,
{
    let fib_program = fibonacci_program(a, b, n);

    let mut vm_config = VmConfig::core();
    vm_config.field_arithmetic_enabled = true;
    gen_vm_program_stark_for_test(fib_program, vec![], vm_config)
}

#[derive(Debug)]
struct MetricDb {
    #[allow(clippy::type_complexity)]
    metrics: HashMap<MetricKind, HashMap<KeyName, Vec<(Vec<Label>, DebugValue)>>>,
}

impl From<Snapshot> for MetricDb {
    fn from(value: Snapshot) -> Self {
        let mut metrics = HashMap::new();
        for (ckey, _, _, value) in value.into_vec() {
            let (kind, key) = ckey.into_parts();
            let (key_name, labels) = key.into_parts();
            let name_to_values = metrics.entry(kind).or_insert(HashMap::new());
            let values = name_to_values.entry(key_name).or_insert(Vec::new());
            values.push((labels, value));
        }
        Self { metrics }
    }
}

impl MetricDb {
    pub fn get_metric(&self, ckey: CompositeKey) -> Option<f64> {
        let labels_to_match: HashSet<_> = ckey.key().labels().collect();
        self.metrics.get(&ckey.kind()).and_then(|m| {
            m.get(ckey.key().name()).and_then(|v| {
                v.iter()
                    .find(|(labels, _)| {
                        let match_tot: usize = labels
                            .iter()
                            .map(|label| {
                                if labels_to_match.contains(label) {
                                    1
                                } else {
                                    0
                                }
                            })
                            .sum();
                        match_tot == labels_to_match.len()
                    })
                    .and_then(|(_, v)| match v {
                        DebugValue::Gauge(v) => Some(v.into_inner()),
                        _ => unreachable!(),
                    })
            })
        })
    }
}

macro_rules! gauge_composite_key {
    ($name:literal, $($label_key:literal => $label_value:literal),*) => {
        CompositeKey::new(
            Gauge,
            key_var!($name, $($label_key => $label_value),*).clone()
        )
    };
}

fn main() {
    let path = std::env::var("OUTPUT_PATH").unwrap();
    let file = std::fs::File::create(path).unwrap();

    // Set up tracing:
    let env_filter = EnvFilter::builder()
        .with_default_directive(Level::INFO.into())
        .from_env_lossy();
    let subscriber = Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .with(MetricsLayer::new());
    // Prepare tracing.
    tracing::subscriber::set_global_default(subscriber).unwrap();

    // Prepare metrics.
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    let recorder = TracingContextLayer::all().layer(recorder);
    // Install the registry as the global recorder
    metrics::set_global_recorder(recorder).unwrap();

    let span = info_span!("Fibonacci Program Inner", group = "fibonacci_program_inner").entered();
    let fib_program_stark = fibonacci_program_stark_for_test(0, 1, 32);
    let StarkForTest {
        any_raps,
        traces,
        pvs,
    } = fib_program_stark;
    let any_raps: Vec<_> = any_raps.iter().map(|x| x.as_ref()).collect();
    let vdata = BabyBearPoseidon2Engine::run_simple_test(&any_raps, traces, &pvs).unwrap();
    span.exit();

    let span = info_span!("Recursive Verify e2e", group = "recursive_verify_e2e").entered();
    let (program, witness_stream) = build_verification_program(pvs, vdata);
    let inner_verifier_sft = gen_vm_program_stark_for_test(
        program,
        witness_stream,
        VmConfig {
            num_public_values: 4,
            ..Default::default()
        },
    );
    run_evm_verifier_e2e_test(
        &inner_verifier_sft,
        // log_blowup = 3 because of poseidon2 chip.
        Some(fri_params_with_80_bits_of_security()[1]),
    );
    span.exit();

    info!("Snapshot: {:?}", snapshotter.snapshot().into_vec());
    let metric_db = MetricDb::from(snapshotter.snapshot());

    let inner_trace_gen_time = metric_db
        .get_metric(gauge_composite_key!("trace_gen_time_ms", "group" => "fibonacci_program_inner"))
        .unwrap();
    let inner_proof_time_ms = metric_db
        .get_metric(
            gauge_composite_key!("stark_proof_time_ms", "group" => "fibonacci_program_inner"),
        )
        .unwrap();
    let inner_total_proof_time_ms = inner_trace_gen_time + inner_proof_time_ms;
    let inner_vm_total_cell = metric_db
        .get_metric(gauge_composite_key!("vm_total_cells", "group" => "fibonacci_program_inner"))
        .unwrap();

    let outer_trace_gen_time_ms = metric_db
        .get_metric(gauge_composite_key!("trace_gen_time_ms", "group" => "recursive_verify_e2e"))
        .unwrap();
    let outer_proof_time_ms = metric_db
        .get_metric(
            gauge_composite_key!("stark_proof_time_ms", "group" => "recursive_verify_e2e", "step"=> "outer_stark_prove")
        )
        .unwrap();
    let outer_total_proof_time_ms = outer_trace_gen_time_ms + outer_proof_time_ms;
    let outer_vm_total_cell = metric_db
        .get_metric(gauge_composite_key!("vm_total_cells", "group" => "recursive_verify_e2e"))
        .unwrap();

    let static_proof_time_ms = metric_db
        .get_metric(
            gauge_composite_key!("halo2_proof_time_ms", "group" => "recursive_verify_e2e", "step"=> "static_verifier_prove")
        )
        .unwrap();
    let static_halo2_total_cell = metric_db
        .get_metric(
            gauge_composite_key!("halo2_total_cells", "group" => "recursive_verify_e2e", "step"=> "static_verifier_prove")
        )
        .unwrap();

    let static_wrapper_proof_time_ms = metric_db
        .get_metric(
            gauge_composite_key!("evm_proof_time_ms", "group" => "recursive_verify_e2e", "step"=> "static_verifier_wrapper_prove")
        )
        .unwrap();
    let static_wrapper_halo2_total_cell = metric_db
        .get_metric(
            gauge_composite_key!("halo2_total_cells", "group" => "recursive_verify_e2e", "step"=> "static_verifier_wrapper_prove")
        )
        .unwrap();

    let result = BTreeMap::from([
        ("inner_proof_time_ms", inner_total_proof_time_ms),
        ("inner_total_cell", inner_vm_total_cell),
        ("outer_proof_time_ms", outer_total_proof_time_ms),
        ("outer_total_cell", outer_vm_total_cell),
        ("static_proof_time_ms", static_proof_time_ms),
        ("static_total_cell", static_halo2_total_cell),
        ("wrapper_proof_time_ms", static_wrapper_proof_time_ms),
        ("wrapper_total_cell", static_wrapper_halo2_total_cell),
    ]);
    serde_json::to_writer_pretty(&file, &result).unwrap();
}
