use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_commit::{ExtensionMmcs, TwoAdicMultiplicativeCoset};
use p3_field::{AbstractField, Field, TwoAdicField};
use p3_field::extension::BinomialExtensionField;
use p3_fri::FriConfig;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};

use afs_compiler::asm::AsmConfig;
use afs_compiler::ir::Builder;

use crate::fri::TwoAdicMultiplicativeCosetVariable;
use crate::fri::types::FriConfigVariable;

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;
type Perm = Poseidon2<Val, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
type Hash = PaddingFreeSponge<Perm, 16, 8, 8>;
type Compress = TruncatedPermutation<Perm, 2, 8, 16>;
type RecursionConfig = AsmConfig<Val, Challenge>;
type RecursionBuilder = Builder<RecursionConfig>;
type ValMmcs =
FieldMerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, Hash, Compress, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;

pub fn const_fri_config(
    builder: &mut RecursionBuilder,
    config: &FriConfig<ChallengeMmcs>,
) -> FriConfigVariable<RecursionConfig> {
    let two_addicity = Val::TWO_ADICITY;
    let mut generators = builder.dyn_array(two_addicity);
    let mut subgroups = builder.dyn_array(two_addicity);
    for i in 0..two_addicity {
        let constant_generator = Val::two_adic_generator(i);
        builder.set(&mut generators, i, constant_generator);

        let constant_domain = TwoAdicMultiplicativeCoset {
            log_n: i,
            shift: Val::one(),
        };
        let domain_value: TwoAdicMultiplicativeCosetVariable<_> = builder.constant(constant_domain);
        builder.set(&mut subgroups, i, domain_value);
    }
    FriConfigVariable {
        log_blowup: builder.eval(BabyBear::from_canonical_usize(config.log_blowup)),
        blowup: builder.eval(BabyBear::from_canonical_usize(1 << config.log_blowup)),
        num_queries: builder.eval(BabyBear::from_canonical_usize(config.num_queries)),
        proof_of_work_bits: builder.eval(BabyBear::from_canonical_usize(config.proof_of_work_bits)),
        subgroups,
        generators,
    }
}

