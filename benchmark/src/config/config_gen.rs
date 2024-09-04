#[cfg(test)]
use std::fs::create_dir_all;

use ax_sdk::config::FriParameters;
#[allow(unused_imports)]
use ax_sdk::{
    config::{
        fri_params::{fri_params_with_100_bits_of_security, fri_params_with_80_bits_of_security},
        EngineType,
    },
    page_config::{
        MultitierPageConfig, MultitierPageParamsConfig, PageConfig, PageMode, PageParamsConfig,
        StarkEngineConfig, TreeParamsConfig,
    },
};
use itertools::iproduct;

use crate::commands::{parse_config_folder, parse_multitier_config_folder};

pub fn get_configs(config_folder: Option<String>) -> Vec<PageConfig> {
    if let Some(config_folder) = config_folder.clone() {
        parse_config_folder(config_folder)
    } else {
        generate_configs()
    }
}

pub fn get_multitier_configs(config_folder: Option<String>) -> Vec<MultitierPageConfig> {
    if let Some(config_folder) = config_folder.clone() {
        parse_multitier_config_folder(config_folder)
    } else {
        generate_multitier_configs()
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
    // let idx_bytes_vec = vec![32];
    // let data_bytes_vec = vec![32];

    // Currently we have the max_rw_ops use the height vec to reduce the number of permutations
    let height_vec = vec![(1_048_576, 1_024), (262_144, 4_096), (32, 32)];
    // let height_vec = vec![(1_048_576, 1_024)];
    let num_ops = vec![1, 8];
    // let num_ops = vec![8];
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
        let opt_config = MultitierPageConfig {
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
                final_leaf_cap: num_ops,
                final_internal_cap: if *leaf_height > 100 { 3 } else { num_ops * 7 },
            },
        };
        let pess_config = MultitierPageConfig {
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
        configs.push(opt_config);
        configs.push(pess_config);
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
