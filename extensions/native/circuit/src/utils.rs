use std::array;

use openvm_circuit::arch::{
    testing::{memory::gen_pointer, VmChipTestBuilder},
    Streams, SystemConfig, VmExecutor,
};
use openvm_instructions::{instruction::Instruction, program::Program};
use openvm_native_compiler::conversion::AS;
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use rand::{distributions::Standard, prelude::Distribution, rngs::StdRng, Rng};

pub(crate) const CASTF_MAX_BITS: usize = 30;

// use crate::{Native, NativeConfig};

// pub fn execute_program(program: Program<BabyBear>, input_stream: impl Into<Streams<BabyBear>>) {
//     let system_config = SystemConfig::default()
//         .with_public_values(4)
//         .with_max_segment_len((1 << 25) - 100);
//     let config = NativeConfig::new(system_config, Native);
//     let executor = VmExecutor::<BabyBear, NativeConfig>::new(config);

//     executor.execute(program, input_stream).unwrap();
// }

pub(crate) const fn const_max(a: usize, b: usize) -> usize {
    [a, b][(a < b) as usize]
}

// If immediate, returns (value, AS::Immediate). Otherwise, writes to native memory and returns
// (ptr, AS::Native). If is_imm is None, randomizes it.
pub fn write_native_or_imm<F: PrimeField32>(
    tester: &mut VmChipTestBuilder<F>,
    rng: &mut StdRng,
    value: F,
    is_imm: Option<bool>,
) -> (F, usize) {
    let is_imm = is_imm.unwrap_or(rng.gen_bool(0.5));
    if is_imm {
        (value, AS::Immediate as usize)
    } else {
        let ptr = gen_pointer(rng, 1);
        tester.write::<1>(AS::Native as usize, ptr, [value]);
        (F::from_canonical_usize(ptr), AS::Native as usize)
    }
}

// Writes value to native memory and returns a pointer to the first element together with the value
// If `value` is None, randomizes it.
pub fn write_native_array<F: PrimeField32, const N: usize>(
    tester: &mut VmChipTestBuilder<F>,
    rng: &mut StdRng,
    value: Option<[F; N]>,
) -> ([F; N], usize)
where
    Standard: Distribution<F>, // Needed for `rng.gen`
{
    let value = value.unwrap_or(array::from_fn(|_| rng.gen()));
    let ptr = gen_pointer(rng, N);
    tester.write::<N>(AS::Native as usize, ptr, value);
    (value, ptr)
}
