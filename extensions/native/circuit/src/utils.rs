use std::sync::Arc;

use openvm_circuit::arch::{
    segment::DefaultSegmentationStrategy, Streams, SystemConfig, VmExecutor,
};
use openvm_instructions::program::Program;
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use crate::{Native, NativeConfig};

pub fn execute_program(program: Program<BabyBear>, input_stream: impl Into<Streams<BabyBear>>) {
    let system_config = SystemConfig::default().with_public_values(4);
    let config = NativeConfig::new(system_config, Native);
    let mut executor = VmExecutor::<BabyBear, NativeConfig>::new(config);
    executor.set_custom_segmentation_strategy(Arc::new(
        DefaultSegmentationStrategy::new_with_max_segment_len(500),
    ));

    executor.execute(program, input_stream).unwrap();
}
