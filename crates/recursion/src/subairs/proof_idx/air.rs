use openvm_circuit_primitives::SubAir;
use p3_air::AirBuilder;
use p3_field::FieldAlgebra;

#[derive(Default)]
pub struct ProofIdxSubAir;

#[derive(Clone, Copy)]
pub struct ProofIdxCols<T> {
    pub proof_idx: T,
    pub is_enabled: T,
    pub is_proof_start: T,
    pub is_proof_end: T,
}

impl<AB: AirBuilder> SubAir<AB> for ProofIdxSubAir {
    type AirContext<'a>
        = (ProofIdxCols<AB::Var>, ProofIdxCols<AB::Var>)
    where
        AB: 'a,
        AB::Var: 'a,
        AB::Expr: 'a;

    fn eval<'a>(&'a self, builder: &'a mut AB, ctx: Self::AirContext<'a>)
    where
        AB::Var: 'a,
        AB::Expr: 'a,
    {
        let (local, next) = ctx;

        let proof_idx_diff = next.proof_idx - local.proof_idx;

        // 1. Base Constraints

        // Boolean `is_enabled` flag
        builder.assert_bool(local.is_enabled);

        // `proof_idx` increments by 0 or 1
        builder
            .when_transition()
            .assert_bool(proof_idx_diff.clone());

        // 2. Boundary Constraints

        // First row enabled implies proof start
        builder
            .when_first_row()
            .when(local.is_enabled)
            .assert_one(local.is_proof_start);

        // Last row enabled implies proof end
        builder
            .when_last_row()
            .when(local.is_enabled)
            .assert_one(local.is_proof_end);

        // 3. Disabled Row Constraints

        // No proof start on disabled rows
        builder
            .when_ne(local.is_enabled, AB::Expr::ONE)
            .assert_zero(local.is_proof_start);

        // No proof end on disabled rows
        builder
            .when_ne(local.is_enabled, AB::Expr::ONE)
            .assert_zero(local.is_proof_end);

        // 4. Proof Constraints

        // 4.1. Within Proof (Δproof_idx ≠ 1)
        // When Δproof_idx ≠ 1, we are within the same proof (Δproof_idx = 0)

        // Enabled state consistency
        builder
            .when_transition()
            .when_ne(proof_idx_diff.clone(), AB::Expr::ONE)
            .assert_eq(local.is_enabled, next.is_enabled);

        // No proof end within proof
        builder
            .when_transition()
            .when_ne(proof_idx_diff.clone(), AB::Expr::ONE)
            .when(local.is_enabled)
            .assert_zero(local.is_proof_end);

        // No proof start within proof
        builder
            .when_transition()
            .when_ne(proof_idx_diff.clone(), AB::Expr::ONE)
            .when(local.is_enabled)
            .assert_zero(next.is_proof_start);

        // 4.2. At Proof Boundaries (Δproof_idx ≠ 0)

        // Enabled local row implies proof end
        builder
            .when(proof_idx_diff.clone())
            .when(local.is_enabled)
            .assert_one(local.is_proof_end);

        // Enabled next row implies proof start
        builder
            .when(proof_idx_diff)
            .when(next.is_enabled)
            .assert_one(next.is_proof_start);
    }
}
