use std::borrow::Borrow;
use openvm_circuit::{
    arch::{ExecutionBridge, ExecutionState},
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupBus,
    utils::{not, select},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, BaseAir},
    p3_field::FieldAlgebra,
    p3_matrix::Matrix,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use crate::keccakf::columns::{KeccakfVmCols, NUM_KECCAKF_VM_COLS};

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct KeccakfVmAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    pub ptr_max_bits: usize,
    pub(super) offset: usize,
}

impl<F> BaseAirWithPublicValues<F> for KeccakfVmAir {}
impl<F> PartitionedBaseAir<F> for KeccakfVmAir {}
impl<F> BaseAir<F> for KeccakfVmAir {
    fn width(&self) -> usize {
        NUM_KECCAKF_VM_COLS
    }
}

impl<AB: InteractionBuilder> Air<AB> for KeccakfVmAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &KeccakfVmCols<_> = (*local).borrow();


    }   
}

impl KeccakfVmAir {
    #[inline]
    pub fn eval_instruction<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakfVmCols<AB::Var>,
    ) {
        let instruction = local.instruction;
        let is_enabled = instruction.is_enabled;
        builder.assert_bool(is_enabled);
        let reg_addr_sp = AB::F::ONE;


        todo!()
    }

    #[inline]
    pub fn constrain_input_read<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakfVmCols<AB::Var>,
        start_read_timestamp: AB::Expr,
    ) {
        todo!()
    }

    #[inline]
    pub fn communicate_with_keccakbus<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakfVmCols<AB::Var>,
    ) {
        todo!()
    }

    #[inline]
    pub fn constrain_output_write<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakfVmCols<AB::Var>
    ) {
        todo!()
    }

}