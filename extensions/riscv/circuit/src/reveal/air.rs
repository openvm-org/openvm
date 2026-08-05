use std::borrow::Borrow;

use openvm_circuit::{
    arch::{ExecutionBridge, ExecutionState, BLOCK_FE_WIDTH},
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadAuxCols},
            MemoryAddress,
        },
        public_values::PublicValuesBus,
    },
};
use openvm_circuit_primitives::{
    var_range::VariableRangeCheckerBus, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{program::DEFAULT_PC_STEP, riscv::REGISTER_AS, LocalOpcode};
use openvm_riscv_transpiler::RevealOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing},
    p3_matrix::Matrix,
    BaseAirWithPublicValues, PartitionedBaseAir,
};

use crate::adapters::byte_ptr_to_u16_ptr;

const REVEAL_TIMESTAMP_DELTA: usize = 1;

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct RevealCols<T> {
    /// Enables this row.
    pub is_valid: T,
    /// Execution state before this reveal.
    pub from_state: ExecutionState<T>,
    /// Byte pointer to the source-value register.
    pub src_ptr: T,
    /// Source register value as u16 limbs.
    pub src_data: [T; BLOCK_FE_WIDTH],
    /// Witness for the source-register read.
    pub src_aux: MemoryReadAuxCols<T>,
    /// Segment-local index of this reveal.
    pub ordinal: T,
    /// Indicates that the next row is another reveal.
    pub has_next: T,
    /// Low limb of the next timestamp gap minus one.
    pub timestamp_delta_low: T,
}

#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(RevealCols<u8>)]
pub struct RevealAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    pub public_values_bus: PublicValuesBus,
    pub range_bus: VariableRangeCheckerBus,
    pub timestamp_max_bits: usize,
}

impl<F: Field> BaseAir<F> for RevealAir {
    fn width(&self) -> usize {
        RevealCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for RevealAir {}
impl<F: Field> PartitionedBaseAir<F> for RevealAir {}

impl<AB: InteractionBuilder> Air<AB> for RevealAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("reveal AIR requires a local row");
        let next = main.row_slice(1).expect("reveal AIR requires a next row");
        let cols: &RevealCols<AB::Var> = (*local).borrow();
        let next: &RevealCols<AB::Var> = (*next).borrow();
        let is_valid: AB::Expr = cols.is_valid.into();
        let next_is_valid: AB::Expr = next.is_valid.into();
        let timestamp: AB::Expr = cols.from_state.timestamp.into();

        // Valid reveal rows form a contiguous prefix.
        builder.assert_bool(cols.is_valid);
        builder.assert_bool(cols.has_next);
        builder
            .when_transition()
            .assert_bool(is_valid.clone() - next_is_valid.clone());
        builder
            .when_transition()
            .assert_eq(cols.has_next, next.is_valid);
        builder.when_last_row().assert_zero(cols.has_next);

        // Ordinals start at zero and increment across reveal rows.
        builder
            .when_first_row()
            .when(is_valid.clone())
            .assert_zero(cols.ordinal);
        builder
            .when_transition()
            .when(next_is_valid.clone())
            .assert_eq(next.ordinal, cols.ordinal + AB::Expr::ONE);
        builder
            .when(AB::Expr::ONE - is_valid.clone())
            .assert_zero(cols.ordinal);

        // Read the revealed value from its source register.
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_u32(REGISTER_AS),
                    byte_ptr_to_u16_ptr::<AB>(cols.src_ptr),
                ),
                cols.src_data.map(Into::into),
                timestamp.clone(),
                &cols.src_aux,
            )
            .eval(builder, is_valid.clone());

        // Range-check gaps between consecutive reveal timestamps.
        let low_bits = self.timestamp_max_bits.min(self.range_bus.range_max_bits);
        let high_bits = self.timestamp_max_bits - low_bits;
        let limb_base = AB::F::from_usize(1 << low_bits);
        let timestamp_delta = next.from_state.timestamp - cols.from_state.timestamp - AB::Expr::ONE;
        let timestamp_delta_high =
            (timestamp_delta - cols.timestamp_delta_low) * limb_base.inverse();
        self.range_bus
            .range_check(cols.timestamp_delta_low, low_bits)
            .eval(builder, cols.has_next);
        self.range_bus
            .range_check(timestamp_delta_high, high_bits)
            .eval(builder, cols.has_next);
        builder
            .when(AB::Expr::ONE - cols.has_next)
            .assert_zero(cols.timestamp_delta_low);

        // Publish each value at its segment-local ordinal.
        self.public_values_bus
            .send(cols.ordinal, cols.src_data)
            .eval(builder, is_valid.clone());

        // Bind the row to the dedicated opcode and its register read.
        self.execution_bridge
            .execute(
                AB::Expr::from_usize(RevealOpcode::REVEAL.global_opcode().as_usize()),
                [
                    cols.src_ptr.into(),
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                ],
                cols.from_state,
                ExecutionState {
                    pc: cols.from_state.pc + AB::F::from_u32(DEFAULT_PC_STEP),
                    timestamp: timestamp + AB::F::from_usize(REVEAL_TIMESTAMP_DELTA),
                },
            )
            .eval(builder, is_valid);
    }
}
