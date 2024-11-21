//! [StarkGenericConfig] and associated types. Originally taken from Plonky3 under MIT license.

use std::marker::PhantomData;

use p3_challenger::{CanObserve, CanSample, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{ExtensionField, Field};

/// Based on [p3_uni_stark::StarkGenericConfig].
pub trait StarkGenericConfig
where
    Domain<Self>: Send + Sync,
    Com<Self>: Send + Sync,
    PcsProof<Self>: Send + Sync,
    PcsProverData<Self>: Send + Sync,
{
    /// The PCS used to commit to trace polynomials.
    type Pcs: Pcs<Self::Challenge, Self::Challenger>;

    /// The field from which most random challenges are drawn.
    type Challenge: ExtensionField<Val<Self>> + Send + Sync;

    /// The challenger (Fiat-Shamir) implementation used.
    type Challenger: FieldChallenger<Val<Self>>
        + CanObserve<<Self::Pcs as Pcs<Self::Challenge, Self::Challenger>>::Commitment>
        + CanSample<Self::Challenge>;

    fn pcs(&self) -> &Self::Pcs;
}

pub type Val<SC> = <<<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Domain as PolynomialSpace>::Val;

pub type Com<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Commitment;

pub type PcsProverData<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::ProverData;

pub type PcsProof<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Proof;

pub type PcsError<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Error;

pub type Domain<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Domain;

pub type PackedVal<SC> = <Val<SC> as Field>::Packing;

pub type PackedChallenge<SC> =
    <<SC as StarkGenericConfig>::Challenge as ExtensionField<Val<SC>>>::ExtensionPacking;

#[derive(Debug)]
pub struct StarkConfig<Pcs, Challenge, Challenger> {
    pcs: Pcs,
    _phantom: PhantomData<(Challenge, Challenger)>,
}

impl<Pcs, Challenge, Challenger> StarkConfig<Pcs, Challenge, Challenger> {
    pub const fn new(pcs: Pcs) -> Self {
        Self {
            pcs,
            _phantom: PhantomData,
        }
    }
}

impl<Pcs, Challenge, Challenger> StarkGenericConfig for StarkConfig<Pcs, Challenge, Challenger>
where
    Challenge: ExtensionField<<Pcs::Domain as PolynomialSpace>::Val>,
    Pcs: p3_commit::Pcs<Challenge, Challenger>,
    Pcs::Domain: Send + Sync,
    Pcs::Commitment: Send + Sync,
    Pcs::ProverData: Send + Sync,
    Pcs::Proof: Send + Sync,
    Challenger: FieldChallenger<<Pcs::Domain as PolynomialSpace>::Val>
        + CanObserve<<Pcs as p3_commit::Pcs<Challenge, Challenger>>::Commitment>
        + CanSample<Challenge>,
{
    type Pcs = Pcs;
    type Challenge = Challenge;
    type Challenger = Challenger;

    fn pcs(&self) -> &Self::Pcs {
        &self.pcs
    }
}

pub struct UniStarkConfig<SC>(pub SC);

impl<SC: StarkGenericConfig> p3_uni_stark::StarkGenericConfig for UniStarkConfig<SC> {
    type Pcs = SC::Pcs;

    type Challenge = SC::Challenge;

    type Challenger = SC::Challenger;

    fn pcs(&self) -> &Self::Pcs {
        self.0.pcs()
    }
}
