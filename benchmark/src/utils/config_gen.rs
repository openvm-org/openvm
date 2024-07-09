use itertools::iproduct;

use afs_test_utils::{
    config::{
        fri_params::{fri_params_with_100_bits_of_security, fri_params_with_80_bits_of_security},
        EngineType, FriParameters,
    },
    page_config::{PageConfig, PageMode, PageParamsConfig, StarkEngineConfig},
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
    let height_vec = vec![65536, 262_144, 1_048_576];
    let engine_vec = vec![
        EngineType::BabyBearPoseidon2,
        // EngineType::BabyBearBlake3,
        // EngineType::BabyBearKeccak,
    ];

    let mut configs = Vec::new();

    for (fri_params, idx_bytes, data_bytes, height, engine) in iproduct!(
        &fri_params_vec,
        &idx_bytes_vec,
        &data_bytes_vec,
        &height_vec,
        &engine_vec
    ) {
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
