use std::{array::from_fn, borrow::BorrowMut, mem::size_of};

use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_continuations::utils::digests_to_poseidon2_input;
use openvm_cpu_backend::CpuBackend;
use openvm_deferral_circuit::canonicity::CanonicityTraceGen;
use openvm_poseidon2_air::Permutation;
use openvm_recursion_circuit::primitives::range::RangeCheckerCpuTraceGenerator;
use openvm_stark_backend::prover::{AirProvingContext, ProverBackend};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_perm, BabyBearPoseidon2Config, DIGEST_SIZE, F,
};
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;

use crate::output::{
    DeferralOutputCommitCols, F_NUM_U16S, SPONGE_BYTES_PER_ROW, U16_BITS, VALS_IN_DIGEST,
};

pub struct DeferralOutputCtx<PB: ProverBackend> {
    pub proving_ctx: AirProvingContext<PB>,
    pub range_ctx: AirProvingContext<PB>,
    pub poseidon2_inputs: Vec<[PB::Val; POSEIDON2_WIDTH]>,
    pub output_commit: [PB::Val; DIGEST_SIZE],
}

pub fn generate_proving_ctx(
    app_exe_commit: [F; DIGEST_SIZE],
    app_vm_commit: [F; DIGEST_SIZE],
    user_pvs: Vec<F>,
    def_idx: usize,
) -> DeferralOutputCtx<CpuBackend<BabyBearPoseidon2Config>> {
    debug_assert!(DIGEST_SIZE.is_multiple_of(F_NUM_U16S));
    debug_assert!(DIGEST_SIZE.is_multiple_of(VALS_IN_DIGEST));
    debug_assert!(user_pvs.len().is_multiple_of(VALS_IN_DIGEST));

    let mut input_val_rows = values_to_rows(&app_exe_commit);
    input_val_rows.extend(values_to_rows(&app_vm_commit));
    input_val_rows.extend(values_to_rows(&user_pvs));

    let num_rows = input_val_rows.len() + 1;
    let height = num_rows.next_power_of_two();
    let width = DeferralOutputCommitCols::<u8>::width();
    let mut trace = vec![F::ZERO; height * width];
    let mut chunks = trace.chunks_exact_mut(width);

    let mut poseidon2_permute_inputs = Vec::with_capacity(num_rows);
    let range_checker = RangeCheckerCpuTraceGenerator::<U16_BITS>::default();
    let output_len = input_val_rows.len() * SPONGE_BYTES_PER_ROW;
    let mut input_capacity = [F::ZERO; DIGEST_SIZE];
    let mut output_commit = [F::ZERO; DIGEST_SIZE];
    let perm = poseidon2_perm();

    for row_idx in 0..height {
        let row = chunks.next().unwrap();
        let cols: &mut DeferralOutputCommitCols<F> = row.borrow_mut();
        cols.row_idx = F::from_usize(row_idx);
        if row_idx < num_rows {
            cols.is_valid = F::ONE;
            cols.is_first = F::from_bool(row_idx == 0);
            cols.output_len = F::from_usize(output_len);

            cols.input_vals = if row_idx == 0 {
                let mut input = [F::ZERO; DIGEST_SIZE];
                input[0] = F::from_usize(def_idx);
                input[1] = F::from_usize(output_len);
                input
            } else {
                let next_f = input_val_rows[row_idx - 1];
                let input_vals = next_f_to_digest(next_f);
                for cell in input_vals {
                    range_checker.add_count(cell.as_canonical_u32() as usize);
                }
                for (cells, aux) in input_vals
                    .chunks_exact(F_NUM_U16S)
                    .zip(cols.canonicity_aux.iter_mut())
                {
                    let x_le = from_fn(|i| cells[i]);
                    let rc = CanonicityTraceGen::generate_subrow(&x_le, aux);
                    range_checker.add_count(rc as usize);
                }
                input_vals
            };

            let perm_input = digests_to_poseidon2_input(cols.input_vals, input_capacity);
            poseidon2_permute_inputs.push(perm_input);

            let perm_output = perm.permute(perm_input);
            cols.res_left = perm_output[..DIGEST_SIZE].try_into().unwrap();
            cols.res_right = perm_output[DIGEST_SIZE..].try_into().unwrap();

            input_capacity = cols.res_right;
            output_commit = cols.res_left;
        }
    }

    DeferralOutputCtx {
        proving_ctx: AirProvingContext::simple_no_pis(RowMajorMatrix::new(trace, width)),
        range_ctx: AirProvingContext::simple_no_pis(range_checker.generate_trace_row_major()),
        poseidon2_inputs: poseidon2_permute_inputs,
        output_commit,
    }
}

fn values_to_rows(values: &[F]) -> Vec<[F; VALS_IN_DIGEST]> {
    values
        .chunks_exact(VALS_IN_DIGEST)
        .map(|chunk| chunk.try_into().unwrap())
        .collect()
}

fn next_f_to_digest(next_f: [F; VALS_IN_DIGEST]) -> [F; DIGEST_SIZE] {
    from_fn(|cell_idx| {
        let f_idx = cell_idx / F_NUM_U16S;
        let cell_in_f = cell_idx % F_NUM_U16S;
        let bytes = next_f[f_idx].as_canonical_u32().to_le_bytes();
        let offset = cell_in_f * size_of::<u16>();
        F::from_u16(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
    })
}
