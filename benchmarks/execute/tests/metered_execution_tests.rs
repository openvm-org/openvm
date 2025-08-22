use openvm_benchmarks_execute::{
    executor, load_program_executable, metered_cost_setup, metering_setup,
};
use test_case::test_case;
use tracing_subscriber::{fmt, EnvFilter};

const TOLERANCE_PERCENT: f64 = 50.0;

fn setup_logging() {
    let _ = fmt::fmt()
        .with_env_filter(EnvFilter::new("info"))
        .try_init();
}

#[test_case("fibonacci_recursive")]
#[test_case("fibonacci_iterative")]
#[test_case("quicksort")]
#[test_case("bubblesort")]
#[test_case("factorial_iterative_u256")]
// #[test_case("revm_snailtracer")]
#[test_case("keccak256")]
#[test_case("keccak256_iter")]
#[test_case("sha256")]
#[test_case("sha256_iter")]
#[test_case("revm_transfer")]
#[test_case("pairing")]
fn test_metered_vs_metered_cost_cells(program: &str) {
    setup_logging();

    let exe = load_program_executable(program)
        .unwrap_or_else(|_| panic!("Failed to load program: {}", program));

    // Metered execution
    let (metered_ctx, executor_idx_to_air_idx) = metering_setup();
    let metered_ctx_clone = metered_ctx.clone();
    let metered_interpreter = executor()
        .metered_instance(&exe, executor_idx_to_air_idx)
        .unwrap();

    let (segments, _vm_state) = metered_interpreter
        .execute_metered(vec![], metered_ctx_clone)
        .expect("Failed to execute with metered context");

    // Metered cost execution
    let (metered_cost_ctx, _) = metered_cost_setup();
    let metered_cost_ctx_clone = metered_cost_ctx.clone();
    let metered_cost_interpreter = executor()
        .metered_cost_instance(&exe, executor_idx_to_air_idx)
        .unwrap();

    let metered_cost_output = metered_cost_interpreter
        .execute_metered_cost(vec![], metered_cost_ctx_clone)
        .expect("Failed to execute with metered cost context");

    // Calculate total cells
    let total_cells: u64 = segments
        .iter()
        .map(|segment| {
            segment
                .trace_heights
                .iter()
                .zip(metered_cost_ctx.widths.iter())
                .map(|(&height, &width)| (height as u64) * (width as u64))
                .sum::<u64>()
        })
        .sum();
    let cost = metered_cost_output.cost;

    tracing::info!(
        "Program: {}, Cells: {}, Cost: {}",
        program,
        total_cells,
        cost
    );

    // Check cells >= cost
    assert!(
        total_cells >= cost,
        "Program '{}': cells ({}) >= cost ({})",
        program,
        total_cells,
        cost
    );

    let diff = total_cells - cost;
    let max_allowed_diff = (total_cells as f64 * TOLERANCE_PERCENT / 100.0) as u64;

    assert!(
        diff <= max_allowed_diff,
        "Program '{}': diff too large. cells: {}, cost: {}, diff: {}, max ({}%): {}",
        program,
        total_cells,
        cost,
        diff,
        TOLERANCE_PERCENT,
        max_allowed_diff
    );
}

#[test]
#[should_panic]
fn test_metered_cost_fails_on_keccak_bomb() {
    setup_logging();

    let exe = load_program_executable("keccak_bomb").expect("Failed to load keccak_bomb program");

    tracing::info!("Loaded keccak_bomb program");

    let (metered_cost_ctx, executor_idx_to_air_idx) = metered_cost_setup();
    let metered_cost_interpreter = executor()
        .metered_cost_instance(&exe, executor_idx_to_air_idx)
        .unwrap();

    let result = metered_cost_interpreter
        .execute_metered_cost(vec![], metered_cost_ctx.clone())
        .expect("This should have panicked due to metered cost limit");

    tracing::info!("Number of instructions executed: {}", result.instret);
    tracing::info!("Final cost: {}", result.cost);
}
#[test]
fn test_keccak_bomb_segment_count() {
    setup_logging();

    let exe = load_program_executable("keccak_bomb").expect("Failed to load keccak_bomb program");

    tracing::info!("Loaded keccak_bomb program");

    let (metered_ctx, executor_idx_to_air_idx) = metering_setup();
    let (metered_cost_ctx, _) = metered_cost_setup();
    let metered_ctx_clone = metered_ctx.clone();
    let metered_interpreter = executor()
        .metered_instance(&exe, executor_idx_to_air_idx)
        .unwrap();

    let (segments, _vm_state) = metered_interpreter
        .execute_metered(vec![], metered_ctx_clone)
        .expect("Failed to execute with metered context");

    let segment_count = segments.len();

    let total_cells_all_segments: u64 = segments
        .iter()
        .map(|segment| {
            segment
                .trace_heights
                .iter()
                .zip(metered_cost_ctx.widths.iter())
                .map(|(&height, &width)| (height as u64) * (width as u64))
                .sum::<u64>()
        })
        .sum();

    tracing::info!(
        "keccak_bomb: {} segments, {} total cells",
        segment_count,
        total_cells_all_segments
    );

    assert!(
        segment_count > 0,
        "Expected at least 1 segment, got {}",
        segment_count
    );
}
