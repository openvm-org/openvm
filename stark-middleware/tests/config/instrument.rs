use std::{
    any::type_name,
    collections::HashMap,
    sync::{Arc, Mutex},
};

use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};

pub type InstrumentCounter = Arc<Mutex<HashMap<String, Vec<usize>>>>;

/// Wrapper to instrument a type to count function calls.
/// CAUTION: Performance may be impacted.
#[derive(Clone, Debug)]
pub struct Instrumented<T> {
    pub inner: T,
    pub input_lens_by_type: InstrumentCounter,
}

impl<T> Instrumented<T> {
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            input_lens_by_type: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn add_len_for_type<A>(&self, len: usize) {
        self.input_lens_by_type
            .lock()
            .unwrap()
            .entry(type_name::<A>().to_string())
            .and_modify(|lens| lens.push(len))
            .or_insert(vec![len]);
    }
}

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
        let input = input.into_iter().collect::<Vec<_>>();
        self.add_len_for_type::<(Item, Out)>(input.len());
        self.inner.hash_iter(input)
    }
}
