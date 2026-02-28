use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::system::{connector::VmConnectorPvs, memory::merkle::MemoryMerklePvs};
use openvm_stark_backend::{
    proof::Proof,
    prover::{AirProvingContext, ColMajorMatrix, CpuBackend},
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, DIGEST_SIZE, F};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use verify_stark::pvs::{VERIFIER_PVS_AIR_ID, VM_PVS_AIR_ID};

use crate::circuit::nonroot::{app::*, verifier::VerifierCombinedPvs, vm_pvs::air::VmPvsCols};

pub fn generate_proving_ctx(
    proofs: &[Proof<BabyBearPoseidon2Config>],
    child_is_app: bool,
    deferral_enabled: bool,
) -> AirProvingContext<CpuBackend<BabyBearPoseidon2Config>> {
    debug_assert!(!proofs.is_empty());

    let mut vm_proofs = Vec::new();
    let mut has_deferral = false;

    for proof in proofs {
        if !deferral_enabled || child_is_app {
            vm_proofs.push(proof);
        } else {
            let child_pvs: &VerifierCombinedPvs<F> =
                proof.public_values[VERIFIER_PVS_AIR_ID].as_slice().borrow();
            if child_pvs.def.deferral_flag != F::ONE {
                vm_proofs.push(proof);
            }
            if child_pvs.def.deferral_flag != F::ZERO {
                has_deferral = true;
            }
        }
    }

    let deferral_flag = if !deferral_enabled || !has_deferral {
        F::ZERO
    } else if vm_proofs.is_empty() {
        F::ONE
    } else {
        F::TWO
    };

    let num_valid_rows = if deferral_flag == F::ONE {
        0
    } else if deferral_flag == F::TWO {
        1
    } else {
        vm_proofs.len()
    };
    let height = num_valid_rows.max(1).next_power_of_two();
    let base_width = VmPvsCols::<u8>::width();
    let width = base_width + deferral_enabled as usize;

    let mut trace = vec![F::ZERO; height * width];
    for (proof_idx, chunk) in trace.chunks_exact_mut(width).enumerate() {
        let (base_chunk, def_chunk) = chunk.split_at_mut(base_width);
        let cols: &mut VmPvsCols<F> = base_chunk.borrow_mut();

        if deferral_enabled {
            def_chunk[0] = deferral_flag;
        }

        if proof_idx >= num_valid_rows {
            continue;
        }
        let proof = vm_proofs[proof_idx];

        cols.proof_idx = F::from_usize(proof_idx);
        cols.is_valid = F::ONE;
        cols.is_last = F::from_bool(proof_idx + 1 == num_valid_rows);

        if child_is_app {
            cols.child_pvs.program_commit = proof.trace_vdata[PROGRAM_AIR_ID]
                .as_ref()
                .expect("program trace vdata must be present for app children")
                .cached_commitments[PROGRAM_CACHED_TRACE_INDEX];

            let &VmConnectorPvs {
                initial_pc,
                final_pc,
                exit_code,
                is_terminate,
            } = proof.public_values[CONNECTOR_AIR_ID].as_slice().borrow();
            cols.child_pvs.initial_pc = initial_pc;
            cols.child_pvs.final_pc = final_pc;
            cols.child_pvs.exit_code = exit_code;
            cols.child_pvs.is_terminate = is_terminate;

            let &MemoryMerklePvs::<_, DIGEST_SIZE> {
                initial_root,
                final_root,
            } = proof.public_values[MERKLE_AIR_ID].as_slice().borrow();
            cols.child_pvs.initial_root = initial_root;
            cols.child_pvs.final_root = final_root;
        } else {
            cols.has_verifier_pvs = F::ONE;
            let child_pvs: &verify_stark::pvs::VmPvs<F> =
                proof.public_values[VM_PVS_AIR_ID].as_slice().borrow();
            cols.child_pvs = *child_pvs;
        }
    }

    let mut public_values = vec![F::ZERO; verify_stark::pvs::VmPvs::<u8>::width()];
    let pvs: &mut verify_stark::pvs::VmPvs<F> = public_values.as_mut_slice().borrow_mut();

    if num_valid_rows > 0 {
        let first_row: &VmPvsCols<F> = trace[..base_width].borrow();
        let last_row: &VmPvsCols<F> =
            trace[(num_valid_rows - 1) * width..(num_valid_rows - 1) * width + base_width].borrow();

        pvs.program_commit = first_row.child_pvs.program_commit;
        pvs.initial_pc = first_row.child_pvs.initial_pc;
        pvs.initial_root = first_row.child_pvs.initial_root;

        pvs.final_pc = last_row.child_pvs.final_pc;
        pvs.exit_code = last_row.child_pvs.exit_code;
        pvs.is_terminate = last_row.child_pvs.is_terminate;
        pvs.final_root = last_row.child_pvs.final_root;
    }

    AirProvingContext {
        cached_mains: vec![],
        common_main: ColMajorMatrix::from_row_major(&RowMajorMatrix::new(trace, width)),
        public_values,
    }
}
