use std::{mem::size_of, sync::Arc};

use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::InteractionBuilder,
    p3_air::{Air, BaseAir},
    p3_field::{FieldAlgebra, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::{cpu::CpuBackend, types::AirProvingContext},
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
    Chip, ChipUsageGetter,
};

use crate::system::memory::{offline_checker::MemoryBus, MemoryAddress};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DummyMemoryInteractionColsRef<'a, T> {
    pub address: MemoryAddress<&'a T, &'a T>,
    pub data: &'a [T],
    pub timestamp: &'a T,
    /// The send frequency. Send corresponds to write. To read, set to negative.
    pub count: &'a T,
}

#[repr(C)]
pub struct DummyMemoryInteractionColsMut<'a, T> {
    pub address: MemoryAddress<&'a mut T, &'a mut T>,
    pub data: &'a mut [T],
    pub timestamp: &'a mut T,
    /// The send frequency. Send corresponds to write. To read, set to negative.
    pub count: &'a mut T,
}

impl<'a, T> DummyMemoryInteractionColsRef<'a, T> {
    pub fn from_slice(slice: &'a [T]) -> Self {
        let (address, slice) = slice.split_at(size_of::<MemoryAddress<u8, u8>>());
        let (count, slice) = slice.split_last().unwrap();
        let (timestamp, data) = slice.split_last().unwrap();
        Self {
            address: MemoryAddress::new(&address[0], &address[1]),
            data,
            timestamp,
            count,
        }
    }
}

impl<'a, T> DummyMemoryInteractionColsMut<'a, T> {
    pub fn from_mut_slice(slice: &'a mut [T]) -> Self {
        let (addr_space, slice) = slice.split_first_mut().unwrap();
        let (ptr, slice) = slice.split_first_mut().unwrap();
        let (count, slice) = slice.split_last_mut().unwrap();
        let (timestamp, data) = slice.split_last_mut().unwrap();
        Self {
            address: MemoryAddress::new(addr_space, ptr),
            data,
            timestamp,
            count,
        }
    }
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct MemoryDummyAir {
    pub bus: MemoryBus,
    pub block_size: usize,
}

impl<F> BaseAirWithPublicValues<F> for MemoryDummyAir {}
impl<F> PartitionedBaseAir<F> for MemoryDummyAir {}
impl<F> BaseAir<F> for MemoryDummyAir {
    fn width(&self) -> usize {
        self.block_size + 4
    }
}

impl<AB: InteractionBuilder> Air<AB> for MemoryDummyAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local = DummyMemoryInteractionColsRef::from_slice(&local);

        self.bus
            .send(
                MemoryAddress::new(*local.address.address_space, *local.address.pointer),
                local.data.to_vec(),
                *local.timestamp,
            )
            .eval(builder, *local.count);
    }
}

#[derive(Clone)]
pub struct MemoryDummyChip<F> {
    pub air: MemoryDummyAir,
    pub trace: Vec<F>,
}

impl<F> MemoryDummyChip<F> {
    pub fn new(air: MemoryDummyAir) -> Self {
        Self {
            air,
            trace: Vec::new(),
        }
    }
}

impl<F: PrimeField32> MemoryDummyChip<F> {
    pub fn send(&mut self, addr_space: u32, ptr: u32, data: &[F], timestamp: u32) {
        self.push(addr_space, ptr, data, timestamp, F::ONE);
    }

    pub fn receive(&mut self, addr_space: u32, ptr: u32, data: &[F], timestamp: u32) {
        self.push(addr_space, ptr, data, timestamp, F::NEG_ONE);
    }

    pub fn push(&mut self, addr_space: u32, ptr: u32, data: &[F], timestamp: u32, count: F) {
        assert_eq!(data.len(), self.air.block_size);
        self.trace.push(F::from_canonical_u32(addr_space));
        self.trace.push(F::from_canonical_u32(ptr));
        self.trace.extend_from_slice(data);
        self.trace.push(F::from_canonical_u32(timestamp));
        self.trace.push(count);
    }
}

impl<SC: StarkGenericConfig, RA> Chip<RA, CpuBackend<SC>> for MemoryDummyChip<Val<SC>>
where
    Val<SC>: PrimeField32,
{
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<CpuBackend<SC>> {
        let height = self.current_trace_height().next_power_of_two();
        let width = self.trace_width();
        let mut trace = self.trace.clone();
        trace.resize(height * width, Val::<SC>::ZERO);

        let trace = Arc::new(RowMajorMatrix::new(trace, width));
        AirProvingContext::simple_no_pis(trace)
    }
}

impl<F: PrimeField32> ChipUsageGetter for MemoryDummyChip<F> {
    fn air_name(&self) -> String {
        format!("MemoryDummyAir<{}>", self.air.block_size)
    }
    fn current_trace_height(&self) -> usize {
        self.trace.len() / self.trace_width()
    }
    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}
