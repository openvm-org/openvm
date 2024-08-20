use afs_compiler::ir::{Array, Builder, Config, Ext, FromConstant, RVar};
use p3_commit::{LagrangeSelectors, PolynomialSpace};

use crate::{
    challenger::ChallengerVariable,
    fri::types::{FriConfigVariable, TwoAdicPcsRoundVariable},
};

pub trait PolynomialSpaceVariable<C: Config>: Sized + FromConstant<C> {
    type Constant: PolynomialSpace<Val = C::F>;

    fn next_point(&self, builder: &mut Builder<C>, point: Ext<C::F, C::EF>) -> Ext<C::F, C::EF>;

    fn selectors_at_point(
        &self,
        builder: &mut Builder<C>,
        point: Ext<C::F, C::EF>,
    ) -> LagrangeSelectors<Ext<C::F, C::EF>>;

    fn zp_at_point(&self, builder: &mut Builder<C>, point: Ext<C::F, C::EF>) -> Ext<C::F, C::EF>;

    fn split_domains(
        &self,
        builder: &mut Builder<C>,
        log_num_chunks: impl Into<RVar<C::N>>,
        num_chunks: impl Into<RVar<C::N>>,
    ) -> Array<C, Self>;

    fn split_domains_const(&self, _: &mut Builder<C>, log_num_chunks: usize) -> Vec<Self>;

    fn create_disjoint_domain(
        &self,
        builder: &mut Builder<C>,
        log_degree: RVar<C::N>,
        config: Option<FriConfigVariable<C>>,
    ) -> Self;
}

pub trait PcsVariable<C: Config> {
    type Domain: PolynomialSpaceVariable<C>;

    type Commitment;

    type Proof;

    fn natural_domain_for_log_degree(
        &self,
        builder: &mut Builder<C>,
        log_degree: RVar<C::N>,
    ) -> Self::Domain;

    fn verify(
        &self,
        builder: &mut Builder<C>,
        rounds: Array<C, TwoAdicPcsRoundVariable<C>>,
        proof: Self::Proof,
        challenger: &mut impl ChallengerVariable<C>,
    );
}
