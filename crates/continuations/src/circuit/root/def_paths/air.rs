use std::borrow::Borrow;

use openvm_circuit::{
    arch::instructions::DEFERRAL_AS, system::memory::dimensions::MemoryDimensions,
};
use openvm_circuit_primitives::{
    utils::{and, assert_array_eq, not},
    SubAir,
};
use openvm_recursion_circuit::bus::Poseidon2CompressBus;
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;

use crate::{
    bn254::CommitBytes,
    circuit::{
        root::bus::{
            DeferralAccPathBus, DeferralAccPathMessage, DeferralMerkleRootsBus,
            DeferralMerkleRootsMessage,
        },
        subair::{MerklePathRowView, MerklePathSubAir, MerklePathSubAirContext},
    },
    utils::zero_hash,
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct DeferralAccMerklePathsCols<F> {
    pub is_valid: F,
    pub is_right_child: F,

    pub initial_node_commit: [F; DIGEST_SIZE],
    pub initial_sibling: [F; DIGEST_SIZE],
    pub final_node_commit: [F; DIGEST_SIZE],
    pub final_sibling: [F; DIGEST_SIZE],

    pub row_idx_exp_2: F,
    pub merkle_path_branch_bits: F,

    pub depth: F,
    pub is_skip: F,
    pub is_within_deferral_as: F,
    pub is_unset: F,
}

pub struct DeferralAccMerklePathsAir {
    pub merkle_path_subair: MerklePathSubAir,
    pub def_acc_paths_bus: DeferralAccPathBus,
    pub def_merkle_roots_bus: DeferralMerkleRootsBus,
    pub address_height: usize,
    pub zero_hash: CommitBytes,
}

impl DeferralAccMerklePathsAir {
    pub fn new(
        poseidon2_compress_bus: Poseidon2CompressBus,
        def_acc_paths_bus: DeferralAccPathBus,
        memory_merkle_roots_bus: DeferralMerkleRootsBus,
        memory_dimensions: MemoryDimensions,
    ) -> Self {
        assert!(memory_dimensions.addr_space_height > 1);
        let pv_start_idx = memory_dimensions.label_to_index((DEFERRAL_AS, 0));
        let merkle_path_branch_bits =
            u32::try_from(pv_start_idx).expect("merkle_path_branch_bits must fit in u32");
        let expected_proof_len = memory_dimensions.overall_height();
        let zero_hash = zero_hash(1).into();
        Self {
            merkle_path_subair: MerklePathSubAir::new(
                poseidon2_compress_bus,
                expected_proof_len,
                merkle_path_branch_bits,
            ),
            def_acc_paths_bus,
            def_merkle_roots_bus: memory_merkle_roots_bus,
            address_height: memory_dimensions.address_height,
            zero_hash,
        }
    }
}

impl<F> BaseAir<F> for DeferralAccMerklePathsAir {
    fn width(&self) -> usize {
        DeferralAccMerklePathsCols::<u8>::width()
    }
}
impl<F> BaseAirWithPublicValues<F> for DeferralAccMerklePathsAir {}
impl<F> PartitionedBaseAir<F> for DeferralAccMerklePathsAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for DeferralAccMerklePathsAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let local: &DeferralAccMerklePathsCols<AB::Var> = (*local).borrow();
        let next: &DeferralAccMerklePathsCols<AB::Var> = (*next).borrow();

        /*
         * Constrain that depth increases by 1 each row, and that all the is_skip flags
         * are at the beginning of the trace. The first is_skip == false row is when we
         * should receive the initial_acc_hash and final_acc_hash. If initial_acc_hash
         * and final_acc_hash were unset, the first commit should be the compression
         * of two zero digests.
         */
        builder.when_first_row().assert_zero(local.depth);
        builder
            .when_transition()
            .when(local.is_valid * next.is_valid)
            .assert_one(next.depth - local.depth);

        let is_set = not(next.is_unset);
        self.def_acc_paths_bus.receive(
            builder,
            DeferralAccPathMessage {
                initial_acc_hash: next.initial_node_commit.map(|v| v * is_set.clone()),
                final_acc_hash: next.final_node_commit.map(|v| v * is_set.clone()),
                depth: next.depth * is_set,
                is_unset: next.is_unset.into(),
            },
            (local.is_skip - next.is_skip) * (AB::Expr::TWO - next.is_valid)
                + local.is_unset
                    * local.is_valid
                    * (local.is_valid - AB::Expr::ONE)
                    * AB::F::TWO.inverse(),
        );

        assert_array_eq::<_, _, AB::Expr, _>(
            &mut builder
                .when(local.is_unset)
                .when(local.is_valid)
                .when_ne(local.is_valid, AB::F::ONE),
            local.initial_node_commit,
            self.zero_hash.into(),
        );

        /*
         * Constrain that is_unset, the flag that determines if deferrals are present, is
         * consistent over all rows.
         */
        builder.assert_bool(local.is_unset);
        builder
            .when_first_row()
            .when(local.is_unset)
            .assert_one(local.is_right_child);
        builder
            .when(and(local.is_valid, next.is_valid))
            .assert_eq(local.is_unset, next.is_unset);

        /*
         * Constrain that is_within_deferral_as is set until address_height. We constrain
         * the two paths to be equal as long as is_within_deferral_as is set, i.e. that the
         * part of DEFERRAL_AS that is not included in the Merkle root is left untouched
         * for the duration of the program execution.
         */
        builder.assert_bool(local.is_within_deferral_as);
        builder
            .when_first_row()
            .assert_one(local.is_within_deferral_as);
        builder
            .when_transition()
            .assert_bool(local.is_within_deferral_as - next.is_within_deferral_as);
        builder
            .when_transition()
            .when(local.is_within_deferral_as - next.is_within_deferral_as)
            .assert_eq(local.depth, AB::Expr::from_usize(self.address_height));

        assert_array_eq(
            &mut builder.when(local.is_within_deferral_as),
            local.initial_sibling,
            local.final_sibling,
        );

        assert_array_eq(
            &mut builder.when(and(local.is_within_deferral_as, local.is_unset)),
            local.initial_node_commit,
            local.final_node_commit,
        );

        /*
         * Call the Merkle path sub-AIR on both the initial and final paths
         */
        self.merkle_path_subair.eval(
            builder,
            (
                MerklePathSubAirContext {
                    local: MerklePathRowView {
                        is_valid: &local.is_valid,
                        is_right_child: &local.is_right_child,
                        node_commit: &local.initial_node_commit,
                        sibling: &local.initial_sibling,
                        row_idx_exp_2: &local.row_idx_exp_2,
                        merkle_path_branch_bits: &local.merkle_path_branch_bits,
                    },
                    next: MerklePathRowView {
                        is_valid: &next.is_valid,
                        is_right_child: &next.is_right_child,
                        node_commit: &next.initial_node_commit,
                        sibling: &next.initial_sibling,
                        row_idx_exp_2: &next.row_idx_exp_2,
                        merkle_path_branch_bits: &next.merkle_path_branch_bits,
                    },
                },
                local.is_skip.into(),
                next.is_skip.into(),
                local.is_unset.into(),
            ),
        );

        self.merkle_path_subair.eval(
            builder,
            (
                MerklePathSubAirContext {
                    local: MerklePathRowView {
                        is_valid: &local.is_valid,
                        is_right_child: &local.is_right_child,
                        node_commit: &local.final_node_commit,
                        sibling: &local.final_sibling,
                        row_idx_exp_2: &local.row_idx_exp_2,
                        merkle_path_branch_bits: &local.merkle_path_branch_bits,
                    },
                    next: MerklePathRowView {
                        is_valid: &next.is_valid,
                        is_right_child: &next.is_right_child,
                        node_commit: &next.final_node_commit,
                        sibling: &next.final_sibling,
                        row_idx_exp_2: &next.row_idx_exp_2,
                        merkle_path_branch_bits: &next.merkle_path_branch_bits,
                    },
                },
                local.is_skip.into(),
                next.is_skip.into(),
                local.is_unset.into(),
            ),
        );

        /*
         * Receive the final memory merkle root on the last valid row.
         */
        self.def_merkle_roots_bus.receive(
            builder,
            DeferralMerkleRootsMessage {
                initial_root: local.initial_node_commit,
                final_root: local.final_node_commit,
            },
            local.is_valid * (AB::Expr::ONE - next.is_valid).square(),
        );
    }
}
