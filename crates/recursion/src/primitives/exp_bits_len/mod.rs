use openvm_stark_sdk::config::baby_bear_poseidon2::F;

pub mod air;
pub mod trace;

pub use air::*;
pub use trace::*;

#[cfg(test)]
mod tests;

#[cfg(feature = "cuda")]
pub(crate) mod cuda;

#[cfg(feature = "cuda")]
pub use cuda::ExpBitsLenGpuTraceGenerator;

#[cfg(feature = "cuda")]
pub type ExpBitsLenTraceGenerator = ExpBitsLenGpuTraceGenerator;

#[cfg(not(feature = "cuda"))]
pub type ExpBitsLenTraceGenerator = ExpBitsLenCpuTraceGenerator;

/// Unified sink trait for enqueuing exp-bits-len requests.
///
/// Implemented for both [`ExpBitsLenCpuTraceGenerator`] and (when `cuda` is
/// enabled) [`ExpBitsLenTraceGenerator`], replacing the duplicated
/// `GkrExpBitsLenSink` and `WhirExpBitsLenSink` traits.
pub trait ExpBitsLenSink {
    fn add_request(&self, base: F, bit_src: F, num_bits: usize);

    fn add_requests<I>(&self, requests: I)
    where
        I: IntoIterator<Item = (F, F, usize)>;

    fn add_requests_with_shift<I>(&self, requests: I)
    where
        I: IntoIterator<Item = (F, F, usize, usize, u32)>;
}

impl ExpBitsLenSink for ExpBitsLenCpuTraceGenerator {
    fn add_request(&self, base: F, bit_src: F, num_bits: usize) {
        ExpBitsLenCpuTraceGenerator::add_request(self, base, bit_src, num_bits);
    }

    fn add_requests<I>(&self, requests: I)
    where
        I: IntoIterator<Item = (F, F, usize)>,
    {
        ExpBitsLenCpuTraceGenerator::add_requests(self, requests);
    }

    fn add_requests_with_shift<I>(&self, requests: I)
    where
        I: IntoIterator<Item = (F, F, usize, usize, u32)>,
    {
        ExpBitsLenCpuTraceGenerator::add_requests_with_shift(self, requests);
    }
}

#[cfg(feature = "cuda")]
impl ExpBitsLenSink for ExpBitsLenGpuTraceGenerator {
    fn add_request(&self, base: F, bit_src: F, num_bits: usize) {
        self.cpu.add_request(base, bit_src, num_bits);
    }

    fn add_requests<I>(&self, requests: I)
    where
        I: IntoIterator<Item = (F, F, usize)>,
    {
        self.cpu.add_requests(requests);
    }

    fn add_requests_with_shift<I>(&self, requests: I)
    where
        I: IntoIterator<Item = (F, F, usize, usize, u32)>,
    {
        self.cpu.add_requests_with_shift(requests);
    }
}
