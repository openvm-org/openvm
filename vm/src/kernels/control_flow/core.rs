use std::borrow::{Borrow, BorrowMut};

use afs_derive::AlignedBorrow;
use afs_primitives::{
    is_equal::{
        columns::{IsEqualAuxCols, IsEqualIoCols},
        IsEqualAir,
    },
    sub_chip::{LocalTraceInstructions, SubAir},
};
use afs_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use itertools::izip;
use p3_air::BaseAir;
use p3_field::{AbstractField, Field, PrimeField32};

use crate::{
    arch::{
        instructions::{
            ControlFlowOpcode,
            ControlFlowOpcode::{BEQ, BNE},
            UsizeOpcode,
        },
        AdapterAirContext, AdapterRuntimeContext, MinimalInstruction, Result, VmAdapterInterface,
        VmCoreAir, VmCoreChip,
    },
    system::program::Instruction,
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct ControlFlowCoreCols<T> {
    pub a: T,
    pub b: T,
    pub offset: T,

    pub is_beq: T,
    pub is_bne: T,

    pub is_equal: T,
    pub is_equal_aux: IsEqualAuxCols<T>,

    pub pc: T,
}

#[derive(Copy, Clone, Debug)]
pub struct ControlFlowCoreAir {
    offset: usize,
}

impl<F: Field> BaseAir<F> for ControlFlowCoreAir {
    fn width(&self) -> usize {
        ControlFlowCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for ControlFlowCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for ControlFlowCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; 1]; 2]>,
    I::Writes: From<()>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _local_adapter: &[AB::Var],
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &ControlFlowCoreCols<_> = local_core.borrow();

        // TODO: constrain `pc`
        // TODO: read `offset` from instruction

        let is_equal_io = IsEqualIoCols {
            x: cols.a,
            y: cols.b,
            is_equal: cols.is_equal,
        };
        SubAir::eval(&IsEqualAir, builder, is_equal_io, cols.is_equal_aux.clone());

        let is_valid = cols.is_beq + cols.is_bne;
        let expected_opcode = cols.is_beq * AB::Expr::from_canonical_usize(BEQ.as_usize())
            + cols.is_bne * AB::Expr::from_canonical_usize(BNE.as_usize());

        builder.assert_bool(is_valid.clone());
        builder.assert_bool(cols.is_beq);
        builder.assert_bool(cols.is_bne);
        let next_pc = cols.pc
            + AB::Expr::one()
            + cols.is_beq * cols.is_equal * (cols.offset - AB::Expr::one())
            + cols.is_bne * (AB::Expr::one() - cols.is_equal) * (cols.offset - AB::Expr::one());

        AdapterAirContext {
            to_pc: Some(next_pc),
            reads: [[cols.a.into()], [cols.b.into()]].into(),
            writes: ().into(),
            instruction: MinimalInstruction {
                is_valid,
                opcode: expected_opcode + AB::Expr::from_canonical_usize(self.offset),
            }
            .into(),
        }
    }
}

#[derive(Debug)]
pub struct ControlFlowRecord<F> {
    pub opcode: ControlFlowOpcode,
    pub a: F,
    pub b: F,
    pub offset: F,

    pub pc: F,
}

#[derive(Debug)]
pub struct ControlFlowCoreChip {
    pub air: ControlFlowCoreAir,
}

impl ControlFlowCoreChip {
    pub fn new(offset: usize) -> Self {
        Self {
            air: ControlFlowCoreAir { offset },
        }
    }
}

impl<F: PrimeField32, I: VmAdapterInterface<F>> VmCoreChip<F, I> for ControlFlowCoreChip
where
    I::Reads: Into<[[F; 1]; 2]>,
    I::Writes: From<()>,
{
    type Record = ControlFlowRecord<F>;
    type Air = ControlFlowCoreAir;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        from_pc: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let Instruction {
            opcode,
            op_c: offset,
            ..
        } = instruction;
        let local_opcode_index = ControlFlowOpcode::from_usize(opcode - self.air.offset);

        let data: [[F; 1]; 2] = reads.into();
        let a = data[0][0];
        let b = data[1][0];

        let to_pc = match local_opcode_index {
            BEQ => {
                if a == b {
                    Some((F::from_canonical_u32(from_pc) + *offset).as_canonical_u32())
                } else {
                    None
                }
            }
            BNE => {
                if a != b {
                    Some((F::from_canonical_u32(from_pc) + *offset).as_canonical_u32())
                } else {
                    None
                }
            }
        };

        let output: AdapterRuntimeContext<F, I> = AdapterRuntimeContext {
            to_pc,
            writes: ().into(),
        };

        let record = Self::Record {
            opcode: local_opcode_index,
            a,
            b,
            offset: *offset,
            pc: F::from_canonical_u32(from_pc),
        };

        Ok((output, record))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            ControlFlowOpcode::from_usize(opcode - self.air.offset)
        )
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        let ControlFlowRecord {
            opcode,
            a,
            b,
            offset,
            pc,
        } = record;
        let row_slice: &mut ControlFlowCoreCols<_> = row_slice.borrow_mut();
        row_slice.a = a;
        row_slice.b = b;
        row_slice.offset = offset;

        row_slice.is_beq = F::from_bool(opcode == BEQ);
        row_slice.is_bne = F::from_bool(opcode == BNE);

        row_slice.is_equal = F::from_bool(a == b);
        row_slice.is_equal_aux =
            LocalTraceInstructions::generate_trace_row(&IsEqualAir, (a, b)).aux;

        row_slice.pc = pc;
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
