use std::path::Path;

use openvm_stark_backend::{SystemParams, WhirProximityStrategy};
use openvm_stark_sdk::config::{params_with_100_bits_security, RECURSION_MAX_CONSTRAINT_DEGREE};
use serde::Serialize;

#[derive(Serialize, Default)]
struct ParamSet {
    app: Option<SystemParams>,
    leaf: Option<SystemParams>,
    internal: Option<SystemParams>,
    root: Option<SystemParams>,
}

fn main() {
    let output_dir = std::env::args().nth(1).unwrap_or_else(|| env!("CARGO_MANIFEST_DIR").to_string());
    generate_interesting_internal_params(&output_dir);
    generate_interesting_root_params(&output_dir);
}

fn generate_interesting_internal_params(output_dir: impl AsRef<Path>) {
    let w_stack = 512;
    let make_internal_params =
        |max_log_height, k_whir, log_blowup, l_skip, pow_bits, folding_pow_bits| -> SystemParams {
            let n_stack = max_log_height - l_skip;
            let proximity = WhirProximityStrategy::ListDecoding { m: 2 };
            params_with_100_bits_security(
                log_blowup,
                l_skip,
                n_stack,
                w_stack,
                folding_pow_bits,
                pow_bits,
                proximity,
                RECURSION_MAX_CONSTRAINT_DEGREE,
                pow_bits,
                k_whir,
            )
        };
    // Root override matching internal_sweep_root.json: fixed l_skip=2, n_stack=18, log_blowup=4,
    // k_whir=4, proximity=ListDecoding{m:1}, all pow_bits=20; only w_stack varies per entry.
    let make_root_params = |root_w_stack| -> SystemParams {
        let max_log_height = 20;
        let l_skip = 2;
        let n_stack = max_log_height - l_skip;
        let log_blowup = 4;
        let k_whir = 4;
        let proximity = WhirProximityStrategy::ListDecoding { m: 1 };
        let pow_bits = 20;
        params_with_100_bits_security(
            log_blowup,
            l_skip,
            n_stack,
            root_w_stack,
            pow_bits,
            pow_bits,
            proximity,
            RECURSION_MAX_CONSTRAINT_DEGREE,
            pow_bits,
            k_whir,
        )
    };
    let make_param =
        |max_log_height, k_whir, log_blowup, l_skip, pow_bits, folding_pow_bits| ParamSet {
            internal: Some(make_internal_params(
                max_log_height,
                k_whir,
                log_blowup,
                l_skip,
                pow_bits,
                folding_pow_bits,
            )),
            ..Default::default()
        };
    let make_param_with_root =
        |max_log_height, k_whir, log_blowup, l_skip, pow_bits, folding_pow_bits, root_w_stack| {
            ParamSet {
                internal: Some(make_internal_params(
                    max_log_height,
                    k_whir,
                    log_blowup,
                    l_skip,
                    pow_bits,
                    folding_pow_bits,
                )),
                root: Some(make_root_params(root_w_stack)),
                ..Default::default()
            }
        };
    let no_root_params = vec![
        // log_blowup=2, k_whir=3, max_log_height=19
        make_param(19, 3, 2, 1, 20, 18),
        make_param(19, 3, 2, 2, 20, 18),
        make_param(19, 3, 2, 3, 20, 18),
        make_param(19, 3, 2, 4, 20, 18),
        make_param(19, 3, 2, 5, 20, 18),
        // log_blowup=3, k_whir=3, max_log_height=19
        make_param(19, 3, 3, 1, 20, 18),
        make_param(19, 3, 3, 2, 20, 18),
        make_param(19, 3, 3, 3, 20, 18),
        make_param(19, 3, 3, 4, 20, 18),
        make_param(19, 3, 3, 5, 20, 18),
        // log_blowup=2, k_whir=4, max_log_height=19
        make_param(19, 4, 2, 1, 20, 18),
        make_param(19, 4, 2, 2, 20, 18),
        make_param(19, 4, 2, 3, 20, 18),
        make_param(19, 4, 2, 4, 20, 18),
        make_param(19, 4, 2, 5, 20, 18),
        // log_blowup=3, k_whir=4, max_log_height=19
        make_param(19, 4, 3, 1, 20, 18),
        make_param(19, 4, 3, 2, 20, 18),
        make_param(19, 4, 3, 3, 20, 18),
        make_param(19, 4, 3, 4, 20, 18),
        make_param(19, 4, 3, 5, 20, 18),
        // log_blowup=1, k_whir=4, max_log_height=20
        make_param(20, 4, 1, 1, 20, 18),
        make_param(20, 4, 1, 2, 20, 18),
        make_param(20, 4, 1, 3, 20, 18),
        make_param(20, 4, 1, 4, 20, 18),
        make_param(20, 4, 1, 5, 20, 18),
        // log_blowup=2, k_whir=4, max_log_height=20
        make_param(20, 4, 2, 1, 20, 18),
        make_param(20, 4, 2, 2, 20, 18),
        make_param(20, 4, 2, 3, 20, 18),
        make_param(20, 4, 2, 4, 20, 18),
        make_param(20, 4, 2, 5, 20, 18),
    ];
    let root_params = vec![
        // internal log_blowup=3, k_whir=4, max_log_height=19; root w_stack=18
        make_param_with_root(19, 4, 3, 1, 20, 18, 18),
        make_param_with_root(19, 4, 3, 2, 20, 18, 18),
        make_param_with_root(19, 4, 3, 3, 20, 18, 18),
        make_param_with_root(19, 4, 3, 4, 20, 18, 18),
        make_param_with_root(19, 4, 3, 5, 20, 18, 18),
    ];

    let tests_dir = output_dir.as_ref();
    let no_root_path = tests_dir.join("internal_sweep_no_root.json");
    serde_json::to_writer_pretty(
        std::fs::File::create(&no_root_path).expect("failed to create internal_sweep_no_root.json"),
        &no_root_params,
    )
    .expect("failed to write internal_sweep_no_root.json");
    println!(
        "wrote {} entries to {}",
        no_root_params.len(),
        no_root_path.display()
    );

    let root_path = tests_dir.join("internal_sweep_root.json");
    serde_json::to_writer_pretty(
        std::fs::File::create(&root_path).expect("failed to create internal_sweep_root.json"),
        &root_params,
    )
    .expect("failed to write internal_sweep_root.json");
    println!(
        "wrote {} entries to {}",
        root_params.len(),
        root_path.display()
    );
}

fn generate_interesting_root_params(output_dir: impl AsRef<Path>) {
    let max_log_height = 20;
    let w_stack = 18;
    let make_param = |k_whir, log_blowup, l_skip, pow_bits| {
        let n_stack = max_log_height - l_skip;
        let proximity = WhirProximityStrategy::ListDecoding { m: 1 };
        let root = params_with_100_bits_security(
            log_blowup,
            l_skip,
            n_stack,
            w_stack,
            pow_bits,
            pow_bits,
            proximity,
            RECURSION_MAX_CONSTRAINT_DEGREE,
            pow_bits,
            k_whir,
        );
        ParamSet {
            root: Some(root),
            ..Default::default()
        }
    };
    let good_params = vec![
        // k_whir = 4
        make_param(4, 2, 1, 20),
        make_param(4, 2, 4, 20),
        make_param(4, 2, 5, 20),
        make_param(4, 3, 1, 20),
        make_param(4, 3, 2, 20),
        make_param(4, 3, 3, 20),
        make_param(4, 3, 5, 20),
        // k_whir = 3
        make_param(3, 4, 1, 20),
        make_param(3, 4, 2, 20),
        make_param(3, 4, 4, 20),
        make_param(3, 3, 5, 20),
        make_param(3, 3, 1, 20),
        make_param(3, 3, 3, 20),
        make_param(3, 3, 5, 20),
        // k_whir = 4, pow_bits lowered
        make_param(4, 4, 2, 15),
        make_param(4, 4, 2, 18),
    ];

    let output_path = output_dir.as_ref().join("root_params.json");
    let file = std::fs::File::create(&output_path).expect("failed to create root_params.json");
    serde_json::to_writer_pretty(file, &good_params).expect("failed to write root_params.json");
    println!(
        "wrote {} good params to {}",
        good_params.len(),
        output_path.display()
    );
}
