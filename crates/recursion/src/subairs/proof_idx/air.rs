use openvm_circuit_primitives::SubAir;
use p3_air::AirBuilder;
use p3_field::FieldAlgebra;

#[derive(Default)]
pub struct ProofIdxSubAir;

#[derive(Clone, Copy)]
pub struct ProofIdxCols<T> {
    pub is_enabled: T,
    pub proof_idx: T,
    pub is_proof_start: T,
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

        // 3. Proof Constraints

        // 3.1. Within Proof (Δproof_idx ≠ 1)
        // When Δproof_idx ≠ 1, we are within the same proof (Δproof_idx = 0)

        // Enabled state consistency
        builder
            .when_transition()
            .when_ne(proof_idx_diff.clone(), AB::Expr::ONE)
            .assert_eq(local.is_enabled, next.is_enabled);

        // No proof start within proof
        builder
            .when_transition()
            .when_ne(proof_idx_diff.clone(), AB::Expr::ONE)
            .when(local.is_enabled)
            .assert_zero(next.is_proof_start);

        // 3.2. At Proof Boundaries (Δproof_idx ≠ 0)

        // Enabled next row implies proof start
        builder
            .when(proof_idx_diff)
            .when(next.is_enabled)
            .assert_one(next.is_proof_start);
    }
}

impl ProofIdxSubAir {
    /// Returns an expression for `is_transition` on enabled rows.
    ///
    /// True when:
    /// - The next row is enabled, AND
    /// - The next row is not a proof start (continuation within same proof)
    pub fn local_is_transition<FA>(next: &ProofIdxCols<FA>) -> impl Into<FA>
    where
        FA: FieldAlgebra + Copy,
    {
        next.is_enabled * (FA::ONE - next.is_proof_start)
    }

    /// Returns an expression for `is_proof_end` on enabled rows.
    ///
    /// Equivalent to `!is_transition`. True when either:
    /// - The next row is disabled, OR
    /// - The next row is enabled and is a proof start (boundary between proofs)
    pub fn local_is_proof_end<FA>(next: &ProofIdxCols<FA>) -> impl Into<FA>
    where
        FA: FieldAlgebra + Copy,
    {
        FA::ONE - Self::local_is_transition(next).into()
    }
}
