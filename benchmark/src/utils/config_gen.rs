use itertools::iproduct;

use afs_test_utils::{
    config::{
        fri_params::{fri_params_with_100_bits_of_security, fri_params_with_80_bits_of_security},
        EngineType, FriParameters,
    },
    page_config::{
        MultitierPageConfig, MultitierPageParamsConfig, PageConfig, PageMode, PageParamsConfig,
        StarkEngineConfig, TreeParamsConfig,
    },
};

pub fn generate_configs() -> Vec<PageConfig> {
    let fri_params_vec = vec![
        fri_params_with_80_bits_of_security(),
        fri_params_with_100_bits_of_security(),
    ];
    let fri_params_vec = fri_params_vec
        .into_iter()
        .flatten()
        .collect::<Vec<FriParameters>>();
    let idx_bytes_vec = vec![32];
    let data_bytes_vec = vec![32, 256, 1024];

    // Currently we have the max_rw_ops use the height vec to reduce the number of permutations
    let height_vec = vec![65536, 262_144, 1_048_576];
    // let height_vec = vec![16, 64]; // Run a mini-benchmark for testing

    let engine_vec = vec![
        EngineType::BabyBearPoseidon2,
        EngineType::BabyBearBlake3,
        EngineType::BabyBearKeccak,
    ];

    let mut configs = Vec::new();

    for (engine, fri_params, idx_bytes, data_bytes, height) in iproduct!(
        &engine_vec,
        &fri_params_vec,
        &idx_bytes_vec,
        &data_bytes_vec,
        &height_vec
    ) {
        if (*height > 1000000 && (fri_params.log_blowup > 2 || *data_bytes > 512))
            || (*height > 500000 && fri_params.log_blowup >= 3)
        {
            continue;
        }
        let config = PageConfig {
            page: PageParamsConfig {
                index_bytes: *idx_bytes,
                data_bytes: *data_bytes,
                height: *height,
                mode: PageMode::ReadWrite,
                max_rw_ops: *height,
                bits_per_fe: 16,
            },
            fri_params: fri_params.to_owned(),
            stark_engine: StarkEngineConfig { engine: *engine },
        };
        configs.push(config);
    }

    configs
}

pub fn generate_multitier_configs() -> Vec<MultitierPageConfig> {
    let fri_params_vec = vec![
        fri_params_with_80_bits_of_security(),
        fri_params_with_100_bits_of_security(),
    ];
    let fri_params_vec = fri_params_vec
        .into_iter()
        .flatten()
        .collect::<Vec<FriParameters>>();
    let idx_bytes_vec = vec![16, 32];
    let data_bytes_vec = vec![16, 32];

    // Currently we have the max_rw_ops use the height vec to reduce the number of permutations
    let height_vec = vec![(1_048_576, 1_024), (262_144, 4_096), (32, 32)];
    // let height_vec = vec![(1_048_576, 1_024)];
    let num_ops = vec![1, 8];
    // let height_vec = vec![16, 64]; // Run a mini-benchmark for testing

    let engine_vec = vec![
        EngineType::BabyBearPoseidon2,
        // EngineType::BabyBearBlake3,
        // EngineType::BabyBearKeccak,
    ];

    let mut configs = Vec::new();

    for (engine, fri_params, idx_bytes, data_bytes, (leaf_height, internal_height), num_ops) in iproduct!(
        &engine_vec,
        &fri_params_vec,
        &idx_bytes_vec,
        &data_bytes_vec,
        &height_vec,
        &num_ops
    ) {
        if (*leaf_height > 1000000 && (fri_params.log_blowup > 2 || *data_bytes > 512))
            || (*leaf_height > 500000 && fri_params.log_blowup >= 3)
        {
            continue;
        }
        let num_ops = if *leaf_height == 1_048_576 {
            (*num_ops + 2) / 3
        } else {
            *num_ops
        };
        let config = MultitierPageConfig {
            page: MultitierPageParamsConfig {
                index_bytes: *idx_bytes,
                data_bytes: *data_bytes,
                mode: PageMode::ReadWrite,
                max_rw_ops: *leaf_height,
                bits_per_fe: 16,
                leaf_height: *leaf_height,
                internal_height: *internal_height,
            },
            fri_params: fri_params.to_owned(),
            stark_engine: StarkEngineConfig { engine: *engine },
            tree: TreeParamsConfig {
                init_leaf_cap: num_ops,
                init_internal_cap: if *leaf_height > 100 { 3 } else { num_ops * 7 },
                final_leaf_cap: 2 * num_ops,
                final_internal_cap: if *leaf_height > 100 { 3 } else { num_ops * 7 },
            },
        };
        configs.push(config);
    }

    configs
}

#[test]
#[ignore]
fn run_generate_configs() {
    let configs = generate_configs();
    let configs_len = configs.len();
    for config in configs {
        let filename = config.generate_filename();
        let filepath = format!("config/rw/{}", filename);
        println!("Saving to {}", filepath);
        config.save_to_file(&filepath);
    }
    println!("Total configs: {}", configs_len);
}
