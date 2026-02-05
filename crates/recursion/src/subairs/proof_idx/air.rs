use openvm_circuit_primitives::SubAir;
use p3_air::AirBuilder;
use p3_field::PrimeCharacteristicRing;
use stark_recursion_circuit_derive::AlignedBorrow;

#[derive(Default)]
pub struct ProofIdxSubAir;

#[repr(C)]
#[derive(AlignedBorrow, Copy, Clone, Debug)]
pub struct ProofIdxIoCols<T> {
    /// Whether the current row is enabled (i.e. not padding)
    pub is_enabled: T,
    pub proof_idx: T,
}

impl<T> ProofIdxIoCols<T> {
    pub fn map_into<S>(self) -> ProofIdxIoCols<S>
    where
        T: Into<S>,
    {
        ProofIdxIoCols {
            is_enabled: self.is_enabled.into(),
            proof_idx: self.proof_idx.into(),
        }
    }
}

impl<AB: AirBuilder> SubAir<AB> for ProofIdxSubAir {
    type AirContext<'a>
        = (ProofIdxIoCols<AB::Expr>, ProofIdxIoCols<AB::Expr>)
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

        // 1. Boolean valid flag
        builder.assert_bool(local.is_enabled.clone());
        // 2. Padding rows are followed by padding rows
        builder
            .when_transition()
            .when_ne(local.is_enabled.clone(), AB::Expr::ONE)
            .assert_zero(next.is_enabled.clone());
        // 3. Proof index increments by exactly one between valid rows
        builder
            .when_transition()
            .when(next.is_enabled.clone())
            .assert_eq(next.proof_idx, local.proof_idx + AB::Expr::ONE);
    }
}
