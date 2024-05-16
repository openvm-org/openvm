use std::{
    any::type_name,
    collections::HashMap,
    sync::{Arc, Mutex},
};

use p3_symmetric::{
    CryptographicHasher, CryptographicPermutation, Permutation, PseudoCompressionFunction,
};
use serde::{Deserialize, Serialize};

use super::FriParameters;

pub type InstrumentCounter = Arc<Mutex<HashMap<String, Vec<usize>>>>;

/// Wrapper to instrument a type to count function calls.
/// CAUTION: Performance may be impacted.
#[derive(Clone, Debug)]
pub struct Instrumented<T> {
    pub is_on: bool,
    pub inner: T,
    pub input_lens_by_type: InstrumentCounter,
}

impl<T> Instrumented<T> {
    pub fn new(inner: T) -> Self {
        Self {
            is_on: true,
            inner,
            input_lens_by_type: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn add_len_for_type<A>(&self, len: usize) {
        if !self.is_on {
            return;
        }
        self.input_lens_by_type
            .lock()
            .unwrap()
            .entry(type_name::<A>().to_string())
            .and_modify(|lens| lens.push(len))
            .or_insert(vec![len]);
    }
}

impl<T: Clone, P: Permutation<T>> Permutation<T> for Instrumented<P> {
    fn permute_mut(&self, input: &mut T) {
        self.add_len_for_type::<T>(1);
        self.inner.permute_mut(input);
    }
    fn permute(&self, input: T) -> T {
        self.add_len_for_type::<T>(1);
        self.inner.permute(input)
    }
}

impl<T: Clone, P: CryptographicPermutation<T>> CryptographicPermutation<T> for Instrumented<P> {}

// Note: this does not currently need to be used if the implemeation is derived from a CryptographicPermutation:
// we can instrument the permutation itself
impl<T, const N: usize, C: PseudoCompressionFunction<T, N>> PseudoCompressionFunction<T, N>
    for Instrumented<C>
{
    fn compress(&self, input: [T; N]) -> T {
        self.add_len_for_type::<T>(N);
        self.inner.compress(input)
    }
}

impl<Item: Clone, Out, H: CryptographicHasher<Item, Out>> CryptographicHasher<Item, Out>
    for Instrumented<H>
{
    fn hash_iter<I>(&self, input: I) -> Out
    where
        I: IntoIterator<Item = Item>,
    {
        if self.is_on {
            let input = input.into_iter().collect::<Vec<_>>();
            self.add_len_for_type::<(Item, Out)>(input.len());
            self.inner.hash_iter(input)
        } else {
            self.inner.hash_iter(input)
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HashStatistics {
    // pub cryptographic_hasher: usize,
    // pub pseudo_compression_function: usize,
    pub permutations: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StarkHashStatistics<T> {
    /// Identifier for the hash permutation
    pub name: String,
    pub stats: HashStatistics,
    pub fri_params: FriParameters,
    pub custom: T,
}
