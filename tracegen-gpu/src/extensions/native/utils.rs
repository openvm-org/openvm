/// Testing framework
#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils {
    use std::array;

    use openvm_circuit::arch::testing::memory::gen_pointer;
    use openvm_native_compiler::conversion::AS;
    use openvm_stark_backend::p3_field::FieldAlgebra;
    use rand::{rngs::StdRng, Rng};
    use stark_backend_gpu::types::F;

    use crate::testing::GpuChipTestBuilder;

    // If immediate, returns (value, AS::Immediate). Otherwise, writes to native memory and returns
    // (ptr, AS::Native). If is_imm is None, randomizes it.
    pub fn write_native_or_imm(
        tester: &mut GpuChipTestBuilder,
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

    // Writes value to native memory and returns a pointer to the first element together with the
    // value If `value` is None, randomizes it.
    pub fn write_native_array<const N: usize>(
        tester: &mut GpuChipTestBuilder,
        rng: &mut StdRng,
        value: Option<[F; N]>,
    ) -> ([F; N], usize) {
        let value = value.unwrap_or(array::from_fn(|_| rng.gen()));
        let ptr = gen_pointer(rng, N);
        tester.write::<N>(AS::Native as usize, ptr, value);
        (value, ptr)
    }
}
