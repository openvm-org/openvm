use std::{
    array,
    borrow::{Borrow, BorrowMut},
    iter::zip,
};

use openvm_circuit::{
    arch::{
        AdapterAirContext, MinimalInstruction, Result, SingleTraceStep, VmAdapterInterface,
        VmCoreAir, VmStateMut,
    },
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    utils::not,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_IMM_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_rv32im_transpiler::BaseAluOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};
use strum::IntoEnumIterator;

use crate::adapters::{tracing_read_reg, tracing_write_reg, Rv32BaseAluAdapterCols};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct BaseAluCoreCols<T, const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub a: [T; NUM_LIMBS],
    pub b: [T; NUM_LIMBS],
    pub c: [T; NUM_LIMBS],

    pub opcode_add_flag: T,
    pub opcode_sub_flag: T,
    pub opcode_xor_flag: T,
    pub opcode_or_flag: T,
    pub opcode_and_flag: T,
}

#[derive(Copy, Clone, Debug)]
pub struct BaseAluCoreAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub bus: BitwiseOperationLookupBus,
    offset: usize,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for BaseAluCoreAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        BaseAluCoreCols::<F, NUM_LIMBS, LIMB_BITS>::width()
    }
}
impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAirWithPublicValues<F>
    for BaseAluCoreAir<NUM_LIMBS, LIMB_BITS>
{
}

impl<AB, I, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, I>
    for BaseAluCoreAir<NUM_LIMBS, LIMB_BITS>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; NUM_LIMBS]; 2]>,
    I::Writes: From<[[AB::Expr; NUM_LIMBS]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &BaseAluCoreCols<_, NUM_LIMBS, LIMB_BITS> = local_core.borrow();
        let flags = [
            cols.opcode_add_flag,
            cols.opcode_sub_flag,
            cols.opcode_xor_flag,
            cols.opcode_or_flag,
            cols.opcode_and_flag,
        ];

        let is_valid = flags.iter().fold(AB::Expr::ZERO, |acc, &flag| {
            builder.assert_bool(flag);
            acc + flag.into()
        });
        builder.assert_bool(is_valid.clone());

        let a = &cols.a;
        let b = &cols.b;
        let c = &cols.c;

        // For ADD, define carry[i] = (b[i] + c[i] + carry[i - 1] - a[i]) / 2^LIMB_BITS. If
        // each carry[i] is boolean and 0 <= a[i] < 2^LIMB_BITS, it can be proven that
        // a[i] = (b[i] + c[i]) % 2^LIMB_BITS as necessary. The same holds for SUB when
        // carry[i] is (a[i] + c[i] - b[i] + carry[i - 1]) / 2^LIMB_BITS.
        let mut carry_add: [AB::Expr; NUM_LIMBS] = array::from_fn(|_| AB::Expr::ZERO);
        let mut carry_sub: [AB::Expr; NUM_LIMBS] = array::from_fn(|_| AB::Expr::ZERO);
        let carry_divide = AB::F::from_canonical_usize(1 << LIMB_BITS).inverse();

        for i in 0..NUM_LIMBS {
            // We explicitly separate the constraints for ADD and SUB in order to keep degree
            // cubic. Because we constrain that the carry (which is arbitrary) is bool, if
            // carry has degree larger than 1 the max-degree constrain could be at least 4.
            carry_add[i] = AB::Expr::from(carry_divide)
                * (b[i] + c[i] - a[i]
                    + if i > 0 {
                        carry_add[i - 1].clone()
                    } else {
                        AB::Expr::ZERO
                    });
            builder
                .when(cols.opcode_add_flag)
                .assert_bool(carry_add[i].clone());
            carry_sub[i] = AB::Expr::from(carry_divide)
                * (a[i] + c[i] - b[i]
                    + if i > 0 {
                        carry_sub[i - 1].clone()
                    } else {
                        AB::Expr::ZERO
                    });
            builder
                .when(cols.opcode_sub_flag)
                .assert_bool(carry_sub[i].clone());
        }

        // Interaction with BitwiseOperationLookup to range check a for ADD and SUB, and
        // constrain a's correctness for XOR, OR, and AND.
        let bitwise = cols.opcode_xor_flag + cols.opcode_or_flag + cols.opcode_and_flag;
        for i in 0..NUM_LIMBS {
            let x = not::<AB::Expr>(bitwise.clone()) * a[i] + bitwise.clone() * b[i];
            let y = not::<AB::Expr>(bitwise.clone()) * a[i] + bitwise.clone() * c[i];
            let x_xor_y = cols.opcode_xor_flag * a[i]
                + cols.opcode_or_flag * ((AB::Expr::from_canonical_u32(2) * a[i]) - b[i] - c[i])
                + cols.opcode_and_flag * (b[i] + c[i] - (AB::Expr::from_canonical_u32(2) * a[i]));
            self.bus
                .send_xor(x, y, x_xor_y)
                .eval(builder, is_valid.clone());
        }

        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(
            self,
            flags.iter().zip(BaseAluOpcode::iter()).fold(
                AB::Expr::ZERO,
                |acc, (flag, local_opcode)| {
                    acc + (*flag).into() * AB::Expr::from_canonical_u8(local_opcode as u8)
                },
            ),
        );

        AdapterAirContext {
            to_pc: None,
            reads: [cols.b.map(Into::into), cols.c.map(Into::into)].into(),
            writes: [cols.a.map(Into::into)].into(),
            instruction: MinimalInstruction {
                is_valid,
                opcode: expected_opcode,
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

pub struct BaseAluCoreChip<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub air: BaseAluCoreAir<NUM_LIMBS, LIMB_BITS>,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<LIMB_BITS>,
}

impl<const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAluCoreChip<NUM_LIMBS, LIMB_BITS> {
    pub fn new(
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<LIMB_BITS>,
        offset: usize,
    ) -> Self {
        Self {
            air: BaseAluCoreAir {
                bus: bitwise_lookup_chip.bus(),
                offset,
            },
            bitwise_lookup_chip,
        }
    }

    #[inline]
    pub fn core_execute<F: PrimeField32>(
        &self,
        instruction: &Instruction<F>,
        [x, y]: [[u8; NUM_LIMBS]; 2],
        core_row: &mut [F],
    ) -> [u8; NUM_LIMBS] {
        let opcode = instruction.opcode;
        let local_opcode = BaseAluOpcode::from_usize(opcode.local_opcode_idx(self.air.offset));

        let z = run_alu::<NUM_LIMBS, LIMB_BITS>(local_opcode, &x, &y);

        let core_row: &mut BaseAluCoreCols<F, NUM_LIMBS, LIMB_BITS> = core_row.borrow_mut();
        core_row.a = z.map(F::from_canonical_u8);
        core_row.b = x.map(F::from_canonical_u8);
        core_row.c = y.map(F::from_canonical_u8);
        core_row.opcode_add_flag = F::from_bool(local_opcode == BaseAluOpcode::ADD);
        core_row.opcode_sub_flag = F::from_bool(local_opcode == BaseAluOpcode::SUB);
        core_row.opcode_xor_flag = F::from_bool(local_opcode == BaseAluOpcode::XOR);
        core_row.opcode_or_flag = F::from_bool(local_opcode == BaseAluOpcode::OR);
        core_row.opcode_and_flag = F::from_bool(local_opcode == BaseAluOpcode::AND);

        z
    }

    pub fn core_fill_trace_row<F: PrimeField32>(&self, core_row: &mut [F]) {
        let core_row: &mut BaseAluCoreCols<F, NUM_LIMBS, LIMB_BITS> = core_row.borrow_mut();

        if core_row.opcode_add_flag == F::ONE || core_row.opcode_sub_flag == F::ONE {
            for a_val in core_row.a.map(|x| x.as_canonical_u32()) {
                self.bitwise_lookup_chip.request_xor(a_val, a_val);
            }
        } else {
            let b = core_row.b.map(|x| x.as_canonical_u32());
            let c = core_row.c.map(|x| x.as_canonical_u32());
            for (b_val, c_val) in zip(b, c) {
                self.bitwise_lookup_chip.request_xor(b_val, c_val);
            }
        }
    }
}

pub struct Rv32BaseAluCoreChip(pub BaseAluCoreChip<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>);

impl<F: PrimeField32, CTX> SingleTraceStep<F, CTX> for Rv32BaseAluCoreChip {
    fn execute(
        &mut self,
        state: VmStateMut<TracingMemory, CTX>,
        instruction: &Instruction<F>,
        row_slice: &mut [F],
    ) -> Result<()> {
        let (adapter_row, core_row) =
            unsafe { row_slice.split_at_mut_unchecked(Rv32BaseAluAdapterCols::<F>::width()) };
        let adapter_row: &mut Rv32BaseAluAdapterCols<F> = adapter_row.borrow_mut();

        let from_timestamp = state.memory.timestamp();
        let &Instruction { a, b, c, d, e, .. } = instruction;
        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert!(
            e.as_canonical_u32() == RV32_IMM_AS || e.as_canonical_u32() == RV32_REGISTER_AS
        );

        let rs1_idx = b.as_canonical_u32();
        let (rs1_t_prev, rs1) = tracing_read_reg(state.memory, rs1_idx);
        let (rs2_t_prev, rs2) = if e.is_zero() {
            let c_u32 = c.as_canonical_u32();
            debug_assert_eq!(c_u32 >> 24, 0);
            state.memory.increment_timestamp();
            (0, c_u32.to_le_bytes())
        } else {
            let rs2_idx = c.as_canonical_u32();
            tracing_read_reg(state.memory, rs2_idx)
        };

        let output = self.0.core_execute(instruction, [rs1, rs2], core_row);
        let rd_idx = a.as_canonical_u32();
        let (rd_t_prev, rd_prev) = tracing_write_reg(state.memory, rd_idx, &output);

        adapter_row.from_state.pc = F::from_canonical_u32(*state.pc);
        adapter_row.from_state.timestamp = F::from_canonical_u32(from_timestamp);
        adapter_row.rd_ptr = a;
        adapter_row.rs1_ptr = b;
        adapter_row.rs2 = c;
        adapter_row.rs2_as = e;
        adapter_row.reads_aux[0].set_prev(F::from_canonical_u32(rs1_t_prev));
        if !e.is_zero() {
            adapter_row.reads_aux[1].set_prev(F::from_canonical_u32(rs2_t_prev));
        }
        adapter_row.writes_aux.set_prev(
            F::from_canonical_u32(rd_t_prev),
            rd_prev.map(F::from_canonical_u8),
        );

        debug_assert_eq!(
            state.memory.timestamp(),
            from_timestamp + 3,
            "incorrect timestamp delta"
        );
        *state.pc += DEFAULT_PC_STEP;
        Ok(())
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            BaseAluOpcode::from_usize(opcode - self.0.air.offset)
        )
    }

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, core_row) =
            unsafe { row_slice.split_at_mut_unchecked(Rv32BaseAluAdapterCols::<F>::width()) };
        let adapter_row: &mut Rv32BaseAluAdapterCols<F> = adapter_row.borrow_mut();

        let mut timestamp = adapter_row.from_state.timestamp.as_canonical_u32();
        mem_helper.fill_from_prev(timestamp, adapter_row.reads_aux[0].as_mut());
        timestamp += 1;
        if !adapter_row.rs2_as.is_zero() {
            mem_helper.fill_from_prev(timestamp, adapter_row.reads_aux[1].as_mut());
        } else {
            let rs2_imm = adapter_row.rs2.as_canonical_u32();
            let mask = (1 << RV32_CELL_BITS) - 1;
            self.0
                .bitwise_lookup_chip
                .request_range(rs2_imm & mask, (rs2_imm >> 8) & mask);
        }
        timestamp += 1;
        self.0.core_fill_trace_row(core_row);
        mem_helper.fill_from_prev(timestamp, adapter_row.writes_aux.as_mut());
    }
}

pub(super) fn run_alu<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    opcode: BaseAluOpcode,
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> [u8; NUM_LIMBS] {
    debug_assert!(LIMB_BITS <= 8, "specialize for bytes");
    match opcode {
        BaseAluOpcode::ADD => run_add::<NUM_LIMBS, LIMB_BITS>(x, y),
        BaseAluOpcode::SUB => run_subtract::<NUM_LIMBS, LIMB_BITS>(x, y),
        BaseAluOpcode::XOR => run_xor::<NUM_LIMBS>(x, y),
        BaseAluOpcode::OR => run_or::<NUM_LIMBS>(x, y),
        BaseAluOpcode::AND => run_and::<NUM_LIMBS>(x, y),
    }
}

fn run_add<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> [u8; NUM_LIMBS] {
    let mut z = [0u8; NUM_LIMBS];
    let mut carry = [0u8; NUM_LIMBS];
    for i in 0..NUM_LIMBS {
        let overflow = x[i] as u16 + y[i] as u16 + if i > 0 { carry[i - 1] as u16 } else { 0 };
        carry[i] = (overflow >> LIMB_BITS) as u8;
        z[i] = overflow as u8;
    }
    z
}

fn run_subtract<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u8; NUM_LIMBS],
    y: &[u8; NUM_LIMBS],
) -> [u8; NUM_LIMBS] {
    let mut z = [0u8; NUM_LIMBS];
    let mut carry = [0u8; NUM_LIMBS];
    for i in 0..NUM_LIMBS {
        let rhs = y[i] as u16 + if i > 0 { carry[i - 1] as u16 } else { 0 };
        if x[i] as u16 >= rhs {
            z[i] = x[i] - rhs as u8;
            carry[i] = 0;
        } else {
            z[i] = (x[i] as u16 + (1u16 << LIMB_BITS) - rhs) as u8;
            carry[i] = 1;
        }
    }
    z
}

fn run_xor<const NUM_LIMBS: usize>(x: &[u8; NUM_LIMBS], y: &[u8; NUM_LIMBS]) -> [u8; NUM_LIMBS] {
    array::from_fn(|i| x[i] ^ y[i])
}

fn run_or<const NUM_LIMBS: usize>(x: &[u8; NUM_LIMBS], y: &[u8; NUM_LIMBS]) -> [u8; NUM_LIMBS] {
    array::from_fn(|i| x[i] | y[i])
}

fn run_and<const NUM_LIMBS: usize>(x: &[u8; NUM_LIMBS], y: &[u8; NUM_LIMBS]) -> [u8; NUM_LIMBS] {
    array::from_fn(|i| x[i] & y[i])
}
