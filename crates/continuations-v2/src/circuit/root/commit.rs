use std::borrow::{Borrow, BorrowMut};

use itertools::Itertools;
use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_circuit_primitives::{
    SubAir,
    encoder::Encoder,
    utils::{assert_array_eq, not},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use recursion_circuit::{
    bus::{Poseidon2CompressBus, Poseidon2CompressMessage},
    utils::assert_zeros,
};
use stark_backend_v2::{
    DIGEST_SIZE, F,
    poseidon2::{WIDTH, sponge::poseidon2_compress},
    prover::{AirProvingContextV2, ColMajorMatrix, CpuBackendV2},
};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::circuit::root::{
    bus::{UserPvsCommitBus, UserPvsCommitMessage, UserPvsCommitTreeBus, UserPvsCommitTreeMessage},
    digests_to_poseidon2_input,
};

const MAX_ENCODER_DEGREE: u32 = 3;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct UserPvsCommitCols<F> {
    pub row_idx: F,

    // 1 for internal, 2 for root, 0 for invalid
    pub send_type: F,
    // 1 for pvs, 2 for internal, 0 for invalid
    pub receive_type: F,

    pub parent: [F; DIGEST_SIZE],
    pub is_right_child: F,

    pub left_child: [F; DIGEST_SIZE],
    pub right_child: [F; DIGEST_SIZE],
}

/**
 * Builds a binary Merkle tree to decommit and expose the raw user public values. Constrains that:
 * - leaf nodes read single digests from exposed PVs, compress with zeros, and compute leaf hashes
 * - internal nodes receive children from an internal permutation bus
 * - root commitment is sent to `UserPvsCommitBus`
 */
pub struct UserPvsCommitAir {
    pub poseidon2_compress_bus: Poseidon2CompressBus,
    pub user_pvs_commit_bus: UserPvsCommitBus,
    pub user_pvs_commit_tree_bus: UserPvsCommitTreeBus,

    num_user_pvs: usize,
    encoder: Encoder,
}

impl UserPvsCommitAir {
    pub fn new(
        poseidon2_compress_bus: Poseidon2CompressBus,
        user_pvs_commit_bus: UserPvsCommitBus,
        user_pvs_commit_tree_bus: UserPvsCommitTreeBus,
        num_user_pvs: usize,
    ) -> Self {
        // Each leaf consumes `DIGEST_SIZE` public values, which are compressed with zeros
        // to compute the leaf hash. We require at least one leaf, and a full binary tree.
        debug_assert!(num_user_pvs >= DIGEST_SIZE);
        debug_assert!(num_user_pvs % DIGEST_SIZE == 0);
        debug_assert!((num_user_pvs / DIGEST_SIZE).is_power_of_two());

        // One selector per leaf PV chunk (each leaf consumes 1 digest).
        let encoder = Encoder::new(num_user_pvs / DIGEST_SIZE, MAX_ENCODER_DEGREE, true);
        UserPvsCommitAir {
            poseidon2_compress_bus,
            user_pvs_commit_bus,
            user_pvs_commit_tree_bus,
            num_user_pvs,
            encoder,
        }
    }
}

impl<F> BaseAir<F> for UserPvsCommitAir {
    fn width(&self) -> usize {
        UserPvsCommitCols::<u8>::width() + self.encoder.width()
    }
}
impl<F> BaseAirWithPublicValues<F> for UserPvsCommitAir {
    fn num_public_values(&self) -> usize {
        self.num_user_pvs
    }
}
impl<F> PartitionedBaseAir<F> for UserPvsCommitAir {}

impl<AB: AirBuilder + InteractionBuilder + AirBuilderWithPublicValues> Air<AB>
    for UserPvsCommitAir
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );

        let const_width = UserPvsCommitCols::<u8>::width();
        let row_idx_flags = &(*local)[const_width..];
        self.encoder.eval(builder, row_idx_flags);

        let local: &UserPvsCommitCols<AB::Var> = (*local)[..const_width].borrow();
        let next: &UserPvsCommitCols<AB::Var> = (*next)[..const_width].borrow();

        let num_rows = AB::F::from_usize(2 * self.num_user_pvs / DIGEST_SIZE);

        /*
         * Constrain that row_idx actually corresponds to the matrix row index, and
         * that there are exactly num_rows rows.
         */
        builder.when_first_row().assert_zero(local.row_idx);
        builder
            .when_transition()
            .assert_eq(local.row_idx + AB::F::ONE, next.row_idx);
        builder
            .when_last_row()
            .assert_eq(local.row_idx, num_rows - AB::F::ONE);

        /*
         * Constrain that is_right_child is correctly set. Even row_idx correspond to
         * left children, and odd row_idx correspond to right - thus we constrain that
         * the is_right_child flag alternates.
         */
        builder.when_first_row().assert_zero(local.is_right_child);
        builder.assert_eq(local.is_right_child, not(next.is_right_child));

        /*
         * Constrain that the first num_rows - 2 rows send their parent internally,
         * the second last row sends the commit, and the last row sends nothing. There
         * must be at least 2 rows, which is enforced by constraining send_type to be
         * 1 or 2 on the first row and 0 on the last.
         */
        builder.assert_tern(local.send_type);

        builder
            .when_first_row()
            .assert_bool(local.send_type - AB::F::ONE);
        builder
            .when_transition()
            .assert_bool(local.send_type - AB::F::ONE);
        builder.when_last_row().assert_zero(local.send_type);

        builder
            .when_ne(local.send_type, AB::F::TWO)
            .assert_bool(next.send_type - AB::F::ONE);
        builder
            .when_ne(local.send_type, AB::F::ZERO)
            .when_ne(local.send_type, AB::F::ONE)
            .assert_zero(next.send_type);

        /*
         * Constrain that the first num_rows / 2 rows read their left and right values
         * from pvs, the next (num_rows / 2) - 1 rows receive their values internally,
         * and that the last row receives no values. We constrain this by asserting the
         * last row's receive_type == 0 and that elsewhere the column is monotonically
         * increasing.
         */
        builder.assert_tern(local.receive_type);

        builder.when_first_row().assert_one(local.receive_type);
        builder
            .when_transition()
            .assert_bool(local.receive_type - AB::F::ONE);
        builder.when_last_row().assert_zero(local.receive_type);

        builder
            .when_ne(local.receive_type, AB::F::ZERO)
            .when_ne(local.receive_type, AB::F::ONE)
            .assert_zero(next.receive_type * (next.receive_type - AB::F::TWO));

        /*
         * Send and receive internal messages. Given a non-root node at index i, its
         * parent is at index floor((i + num_rows) / 2). Note that we send the
         * internal message when local.send_type == 1, and receive left and right
         * when internal_receive == 2.
         */
        let half = AB::F::TWO.inverse();
        let internal_send = local.send_type * (AB::Expr::TWO - local.send_type);
        let internal_receive = local.receive_type * (local.receive_type - AB::F::ONE) * half;

        self.user_pvs_commit_tree_bus.send(
            builder,
            UserPvsCommitTreeMessage {
                child_value: local.parent.map(Into::into),
                is_right_child: local.is_right_child.into(),
                parent_idx: (local.row_idx - local.is_right_child + num_rows) * half,
            },
            internal_send,
        );

        self.user_pvs_commit_tree_bus.receive(
            builder,
            UserPvsCommitTreeMessage {
                child_value: local.left_child.map(Into::into),
                is_right_child: AB::Expr::ZERO,
                parent_idx: local.row_idx.into(),
            },
            internal_receive.clone(),
        );

        self.user_pvs_commit_tree_bus.receive(
            builder,
            UserPvsCommitTreeMessage {
                child_value: local.right_child.map(Into::into),
                is_right_child: AB::Expr::ONE,
                parent_idx: local.row_idx.into(),
            },
            internal_receive,
        );

        /*
         * Send and receive external messages. At every valid node we have to receive
         * a message from the Poseidon2 peripheral AIR to constrain that parent is the
         * compression hash of left_child and right_child. At the root node we send the
         * commit to UserPvsInMemoryAir, which constrains that it is in the correct
         * position in the final memory Merkle tree. Note a row is valid iff it is not
         * the last, i.e. iff send_type (or equivalently receive_type) is non-zero.
         */
        let is_valid = local.send_type * (AB::Expr::from_u8(3) - local.send_type) * half;
        let is_root = local.send_type * (local.send_type - AB::F::ONE) * half;

        let poseidon2_input: [AB::Var; POSEIDON2_WIDTH] = local
            .left_child
            .into_iter()
            .chain(local.right_child)
            .collect_array()
            .unwrap();

        self.poseidon2_compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: poseidon2_input,
                output: local.parent,
            },
            is_valid,
        );

        self.user_pvs_commit_bus.send(
            builder,
            UserPvsCommitMessage {
                user_pvs_commit: local.parent,
            },
            is_root,
        );

        /*
         * Constrain that the left_child of each leaf node at row_idx corresponds to
         * this AIR's public values, and that right_child is all zero on these rows.
         * Leaf nodes rows should be in order of their position in the public values
         * vector. A row is a leaf node if its receive_type == 1.
         */
        let pvs = builder.public_values();
        let is_leaf = local.receive_type * (AB::Expr::TWO - local.receive_type);

        let mut pvs_digest = [AB::Expr::ZERO; DIGEST_SIZE];

        for (pv_chunk_idx, pvs_chunk) in pvs.chunks(DIGEST_SIZE).enumerate() {
            let selected = self
                .encoder
                .get_flag_expr::<AB>(pv_chunk_idx, row_idx_flags);
            for digest_idx in 0..DIGEST_SIZE {
                pvs_digest[digest_idx] += selected.clone() * pvs_chunk[digest_idx].into();
            }
        }

        builder.assert_eq(self.encoder.is_valid::<AB>(row_idx_flags), is_leaf.clone());
        assert_array_eq(
            builder,
            pvs_digest,
            local.left_child.map(|x| x * is_leaf.clone()),
        );
        assert_zeros(&mut builder.when(is_leaf), local.right_child);
    }
}

pub fn generate_proving_ctx(
    user_pvs: Vec<F>,
) -> (AirProvingContextV2<CpuBackendV2>, Vec<[F; WIDTH]>) {
    let num_user_pvs = user_pvs.len();

    // Each leaf consumes `DIGEST_SIZE` public values, which is padded and hashed before
    // being inserted into the Merkle tree. We require at least one leaf (so at least one
    // Poseidon2 hash), and a full binary tree.
    debug_assert!(num_user_pvs >= DIGEST_SIZE);
    debug_assert!(num_user_pvs % DIGEST_SIZE == 0);
    debug_assert!((num_user_pvs / DIGEST_SIZE).is_power_of_two());

    // One selector per leaf PV chunk (each leaf consumes 1 digest).
    let encoder = Encoder::new(num_user_pvs / DIGEST_SIZE, MAX_ENCODER_DEGREE, true);

    let num_pv_digests = num_user_pvs / DIGEST_SIZE;
    let const_width = UserPvsCommitCols::<u8>::width();
    let width = const_width + encoder.width();
    let mut trace = vec![F::ZERO; 2 * num_pv_digests * width];
    let mut chunks = trace.chunks_mut(width);

    let mut next_layer = Vec::with_capacity(num_pv_digests);
    let mut poseidon2_compress_inputs = Vec::with_capacity(2 * num_pv_digests - 1);

    // Write leaf nodes that read each digest child from pvs
    for pv_digest in user_pvs.chunks(DIGEST_SIZE) {
        let chunk = chunks.next().unwrap();
        let cols: &mut UserPvsCommitCols<F> = chunk[..const_width].borrow_mut();

        let row_idx = next_layer.len();
        let left: [F; DIGEST_SIZE] = pv_digest[..DIGEST_SIZE].try_into().unwrap();
        let right: [F; DIGEST_SIZE] = [F::ZERO; DIGEST_SIZE];
        let parent = poseidon2_compress(left, right);
        poseidon2_compress_inputs.push(digests_to_poseidon2_input(left, right));

        cols.row_idx = F::from_usize(row_idx);
        cols.send_type = if num_pv_digests == 1 { F::TWO } else { F::ONE };
        cols.receive_type = F::ONE;
        cols.parent = parent;
        cols.is_right_child = F::from_bool(row_idx & 1 == 1);
        cols.left_child = left;
        cols.right_child = right;

        chunk[const_width..].copy_from_slice(
            encoder
                .get_flag_pt(row_idx)
                .into_iter()
                .map(F::from_u32)
                .collect::<Vec<_>>()
                .as_slice(),
        );
        next_layer.push(parent);
    }

    let mut row_idx = next_layer.len();

    // Write internal nodes that receive left and right child from bus
    while next_layer.len() > 1 {
        let parent_layer_len = next_layer.len() >> 1;
        for parent_idx in 0..parent_layer_len {
            let chunk = chunks.next().unwrap();
            let cols: &mut UserPvsCommitCols<F> = chunk[..const_width].borrow_mut();

            let left = next_layer[2 * parent_idx];
            let right = next_layer[2 * parent_idx + 1];
            let parent = poseidon2_compress(left, right);
            poseidon2_compress_inputs.push(digests_to_poseidon2_input(left, right));

            cols.row_idx = F::from_usize(row_idx);
            cols.send_type = if parent_layer_len > 1 { F::ONE } else { F::TWO };
            cols.receive_type = F::TWO;
            cols.parent = parent;
            cols.is_right_child = F::from_bool(row_idx & 1 == 1);
            cols.left_child = left;
            cols.right_child = right;

            row_idx += 1;
            next_layer[parent_idx] = parent;
        }
        next_layer.truncate(parent_layer_len);
    }

    debug_assert_eq!(row_idx + 1, 2 * num_pv_digests);
    let last_chunk = chunks.next().unwrap();
    let last_cols: &mut UserPvsCommitCols<F> = last_chunk[..const_width].borrow_mut();
    last_cols.row_idx = F::from_usize(row_idx);
    last_cols.is_right_child = F::ONE;

    (
        AirProvingContextV2::simple(
            ColMajorMatrix::from_row_major(&RowMajorMatrix::new(trace, width)),
            user_pvs,
        ),
        poseidon2_compress_inputs,
    )
}
