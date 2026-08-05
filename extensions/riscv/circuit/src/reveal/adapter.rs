use openvm_circuit::{
    arch::{
        AdapterAirContext, ExecutionBridge, Postflight, PostflightError, PostflightStep,
        VmAdapterAir, BLOCK_FE_WIDTH,
    },
    system::memory::{offline_checker::MemoryBridge, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    ColumnsAir,
};
use openvm_instructions::PUBLIC_VALUES_AS;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeField32},
};

use crate::adapters::{
    store::multi_byte::{MultiByteAdapterAir, MultiByteAdapterFiller, StoreMultiReplay},
    StoreMultiByteAdapterAirInterface, StoreMultiByteAdapterCols, DOUBLEWORD_ACCESS_WIDTH,
};

pub type RevealAdapterCols<T> = StoreMultiByteAdapterCols<T>;

#[derive(Clone, Copy, Debug, ColumnsAir)]
#[columns_via(RevealAdapterCols<u8>)]
pub struct RevealAdapterAir {
    inner: MultiByteAdapterAir,
}

impl RevealAdapterAir {
    pub fn new(
        memory_bridge: MemoryBridge,
        execution_bridge: ExecutionBridge,
        range_bus: VariableRangeCheckerBus,
        pointer_max_bits: usize,
    ) -> Self {
        Self {
            inner: MultiByteAdapterAir::new(
                memory_bridge,
                execution_bridge,
                range_bus,
                pointer_max_bits,
            ),
        }
    }
}

impl<F: Field> BaseAir<F> for RevealAdapterAir {
    fn width(&self) -> usize {
        <MultiByteAdapterAir as BaseAir<F>>::width(&self.inner)
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for RevealAdapterAir {
    type Interface = StoreMultiByteAdapterAirInterface;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        self.inner.eval(builder, local, ctx, PUBLIC_VALUES_AS);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        self.inner.get_from_pc::<AB>(local)
    }
}

#[derive(Clone)]
pub struct RevealAdapterFiller {
    inner: MultiByteAdapterFiller,
}

impl RevealAdapterFiller {
    pub fn new(
        pointer_max_bits: usize,
        range_checker_chip: SharedVariableRangeCheckerChip,
    ) -> Self {
        Self {
            inner: MultiByteAdapterFiller::new(pointer_max_bits, range_checker_chip),
        }
    }

    pub(crate) fn replay<F: PrimeField32>(
        &self,
        postflight: &Postflight<'_, F>,
        step: PostflightStep,
        mem_helper: &MemoryAuxColsFactory<F>,
        adapter_row: &mut RevealAdapterCols<F>,
        compute: impl FnOnce(
            [u16; BLOCK_FE_WIDTH],
            [[u16; BLOCK_FE_WIDTH]; 2],
            usize,
        ) -> [[u16; BLOCK_FE_WIDTH]; 2],
    ) -> Result<StoreMultiReplay, PostflightError> {
        self.inner.replay::<F, DOUBLEWORD_ACCESS_WIDTH>(
            postflight,
            step,
            mem_helper,
            adapter_row,
            compute,
            PUBLIC_VALUES_AS,
        )
    }
}
