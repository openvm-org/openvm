use openvm_circuit::arch::{AdapterAirContext, VmCoreAir};
use openvm_circuit_primitives::{bitwise_op_lookup::BitwiseOperationLookupBus, ColumnsAir};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::RevealOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder, p3_air::BaseAir, p3_field::Field, BaseAirWithPublicValues,
};

use crate::{adapters::StoreMultiByteAdapterAirInterface, store::StoreDoublewordCoreAir};

#[derive(Debug, Clone)]
pub struct RevealCoreAir {
    inner: StoreDoublewordCoreAir,
}

impl RevealCoreAir {
    pub fn new(bitwise_lookup_bus: BitwiseOperationLookupBus) -> Self {
        Self {
            inner: StoreDoublewordCoreAir::new_with_local_opcode(
                RevealOpcode::CLASS_OFFSET,
                RevealOpcode::REVEAL as usize,
                bitwise_lookup_bus,
            ),
        }
    }
}

impl<F: Field> BaseAir<F> for RevealCoreAir {
    fn width(&self) -> usize {
        <StoreDoublewordCoreAir as BaseAir<F>>::width(&self.inner)
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for RevealCoreAir {}

impl ColumnsAir for RevealCoreAir {
    fn columns(&self) -> Option<Vec<String>> {
        self.inner.columns()
    }
}

impl<AB: InteractionBuilder> VmCoreAir<AB, StoreMultiByteAdapterAirInterface>
    for RevealCoreAir
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, StoreMultiByteAdapterAirInterface> {
        self.inner.eval(builder, local_core, from_pc)
    }

    fn start_offset(&self) -> usize {
        RevealOpcode::CLASS_OFFSET
    }
}
