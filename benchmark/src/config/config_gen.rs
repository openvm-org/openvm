#[cfg(test)]
use std::fs::create_dir_all;

use itertools::iproduct;

#[allow(unused_imports)]
use afs_test_utils::{
    config::{
        fri_params::{fri_params_with_100_bits_of_security, fri_params_with_80_bits_of_security},
        EngineType,
    },
    page_config::{PageConfig, PageMode, PageParamsConfig, StarkEngineConfig},
};

use crate::commands::parse_config_folder;

pub fn get_configs(config_folder: Option<String>) -> Vec<PageConfig> {
    if let Some(config_folder) = config_folder.clone() {
        parse_config_folder(config_folder)
    } else {
        generate_configs()
    }
}

pub fn generate_configs() -> Vec<PageConfig> {
    let fri_params_vec = vec![
        // fri_params_with_80_bits_of_security()[0],
        // fri_params_with_80_bits_of_security()[1],
        fri_params_with_80_bits_of_security()[2],
        // fri_params_with_100_bits_of_security()[0],
        // fri_params_with_100_bits_of_security()[1],
    ];
    let idx_bytes_vec = vec![32];
    let data_bytes_vec = vec![32, 256, 1024];

    let height_vec = vec![65536, 262_144, 1_048_576];
    // let height_vec = vec![256, 1024]; // Run a mini-benchmark for testing

    // max_rw_ops as the number of log_2 of height
    let max_rw_ops_shift_vec = vec![0, 1, 2, 3, 4];

    let engine_vec = vec![
        EngineType::BabyBearPoseidon2,
        // EngineType::BabyBearBlake3,
        // EngineType::BabyBearKeccak,
    ];

    let mut configs = Vec::new();

    for (engine, fri_params, idx_bytes, data_bytes, height, max_rw_ops_shift) in iproduct!(
        &engine_vec,
        &fri_params_vec,
        &idx_bytes_vec,
        &data_bytes_vec,
        &height_vec,
        &max_rw_ops_shift_vec,
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
                max_rw_ops: *height >> *max_rw_ops_shift,
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
    let folder = "config/rw";
    let configs = generate_configs();
    let configs_len = configs.len();
    for config in configs {
        let filename = config.generate_filename();
        let _ = create_dir_all(folder);
        let filepath = format!("{}/{}", folder, filename);
        println!("Saving to {}", filepath);
        config.save_to_file(&filepath);
    }
    println!("Total configs: {}", configs_len);
}
