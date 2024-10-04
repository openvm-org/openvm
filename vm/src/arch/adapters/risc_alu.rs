use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{AirBuilderWithPublicValues, PairBuilder};
use p3_field::{AbstractField, Field, PrimeField32};

use super::{Rv32RegisterAdapterAir, Rv32RegisterAdapterCols, Rv32RegisterAdapterInterface};
use crate::{
    arch::{
        ExecutionState, InstructionOutput, IntegrationInterface, MachineAdapter,
        MachineAdapterInterface, Result,
    },
    memory::{MemoryChip, MemoryReadRecord, MemoryWriteRecord},
    program::Instruction,
};

/// Reads instructions of the form OP a, b, c, d, e where [a:4]_d = [b:4]_d op [c:4]_e.
/// Operand d can only be 1, and e can be either 1 (for register reads) or 0 (when c
/// is an immediate).
pub struct Rv32ArithmeticAdapter<F: Field> {
    _marker: std::marker::PhantomData<F>,
    pub air: Rv32RegisterAdapterAir<2, 1>,
}

pub struct Rv32ArithmeticReadRecord<F: Field> {
    pub opcode: usize,
    pub address_space: F,
    pub x_read: MemoryReadRecord<F, 4>,
    pub y_read: MemoryReadRecord<F, 4>,
}

pub struct Rv32ArithmeticWriteRecord<F: Field> {
    pub from_state: ExecutionState<usize>,
    pub z_write: MemoryWriteRecord<F, 4>,
}

impl<F: PrimeField32> MachineAdapter<F> for Rv32ArithmeticAdapter<F> {
    type ReadRecord = Rv32ArithmeticReadRecord<F>;
    type WriteRecord = Rv32ArithmeticWriteRecord<F>;
    type Air = Rv32RegisterAdapterAir<2, 1>;
    type Cols<T> = Rv32RegisterAdapterCols<T, 2, 1>;
    type Interface<T: AbstractField> = Rv32RegisterAdapterInterface<T, 2, 1>;

    fn preprocess(
        &mut self,
        memory: &mut MemoryChip<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface<F> as MachineAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction {
            opcode,
            op_b: b,
            op_c: c,
            d,
            e,
            ..
        } = *instruction;

        assert_eq!(d.as_canonical_u32(), 1);

        let x_read = memory.read::<4>(d, b);
        let y_read = if (e).as_canonical_u32() == 0 {
            let c_u32 = (c).as_canonical_u32();
            let c_u8 = [
                c_u32 as u8,
                (c_u32 >> 8) as u8,
                (c_u32 >> 16) as u8,
                (c_u32 >> 16) as u8,
            ];
            MemoryReadRecord {
                address_space: F::zero(),
                pointer: F::zero(),
                timestamp: F::zero(),
                prev_timestamp: F::zero(),
                data: c_u8.map(F::from_canonical_u8),
            }
        } else {
            memory.read::<4>(e, c)
        };

        Ok((
            [x_read.data, y_read.data],
            Self::ReadRecord {
                opcode,
                address_space: e,
                x_read,
                y_read,
            },
        ))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryChip<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<usize>,
        output: InstructionOutput<F, Self::Interface<F>>,
    ) -> Result<(ExecutionState<usize>, Self::WriteRecord)> {
        debug_assert_eq!(
            from_state.timestamp,
            memory.timestamp().as_canonical_u32() as usize
        );

        let Instruction { op_a: a, d, .. } = *instruction;
        let z_write = memory.write::<4>(d, a, output.writes[0]);

        Ok((
            ExecutionState {
                pc: from_state.pc + 1,
                timestamp: memory.timestamp().as_canonical_u32() as usize,
            },
            Self::WriteRecord {
                from_state,
                z_write,
            },
        ))
    }

    fn generate_trace_row(
        &self,
        _row_slice: &mut Self::Cols<F>,
        _read_record: Self::ReadRecord,
        _write_record: Self::WriteRecord,
    ) {
        todo!();
    }

    fn eval_adapter_constraints<
        AB: InteractionBuilder<F = F> + PairBuilder + AirBuilderWithPublicValues,
    >(
        _air: &Self::Air,
        _builder: &mut AB,
        _local: &Self::Cols<AB::Var>,
        _interface: IntegrationInterface<AB::Expr, Self::Interface<AB::Expr>>,
    ) -> AB::Expr {
        todo!();
    }

    fn air(&self) -> Self::Air {
        self.air
    }
}
