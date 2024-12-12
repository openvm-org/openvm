use openvm_native_compiler::{
    ir::RVar,
    prelude::{Array, Builder, Config, Ext, Felt, Var},
};

use crate::digest::DigestVariable;

pub mod duplex;
pub mod multi_field32;

/// Reference: [p3_challenger::CanObserve].
pub trait CanObserveVariable<C: Config, V> {
    fn observe(&mut self, builder: &mut Builder<C>, value: V);

    fn observe_slice(&mut self, builder: &mut Builder<C>, values: Array<C, V>);
}

/// Reference: [p3_challenger::CanObserve].
pub trait CanObserveDigest<C: Config> {
    fn observe_digest(&mut self, builder: &mut Builder<C>, value: DigestVariable<C>);
}

pub trait CanSampleVariable<C: Config, V> {
    #[allow(dead_code)]
    fn sample(&mut self, builder: &mut Builder<C>) -> V;
}

/// Reference: [p3_challenger::FieldChallenger].
pub trait FeltChallenger<C: Config>:
    CanObserveVariable<C, Felt<C::F>> + CanSampleVariable<C, Felt<C::F>> + CanSampleBitsVariable<C>
{
    fn sample_ext(&mut self, builder: &mut Builder<C>) -> Ext<C::F, C::EF>;
}

pub trait CanSampleBitsVariable<C: Config> {
    fn sample_bits(&mut self, builder: &mut Builder<C>, nb_bits: RVar<C::N>)
        -> Array<C, Var<C::N>>;
}

pub trait CanCheckWitness<C: Config> {
    fn check_witness(&mut self, builder: &mut Builder<C>, nb_bits: usize, witness: Felt<C::F>);
}

pub trait ChallengerVariable<C: Config>:
    FeltChallenger<C> + CanObserveDigest<C> + CanCheckWitness<C>
{
    fn new(builder: &mut Builder<C>) -> Self;
}
