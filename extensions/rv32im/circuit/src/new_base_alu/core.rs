use std::{
    array,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit::{
    arch::{
        new_integration_api::{VmAdapter, VmCoreAir, VmCoreChip},
        AirTx, AirTxMaybeRead, AirTxRead, AirTxWrite, ExecuteTx, ExecuteTxMaybeRead, ExecuteTxRead,
        ExecuteTxWrite, Result, TraceTx, TraceTxMaybeRead, TraceTxRead, TraceTxWrite,
    },
    system::memory::{MemoryAddress, MemoryController},
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    utils::not,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_rv32im_transpiler::BaseAluOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_big_array::BigArray;
use strum::IntoEnumIterator;

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

impl<AB, TX, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreAir<AB, TX>
    for BaseAluCoreAir<NUM_LIMBS, LIMB_BITS>
where
    AB: InteractionBuilder,
    TX: AirTx<AB>
        + AirTxRead<AB, [AB::Expr; NUM_LIMBS]>
        + AirTxMaybeRead<AB, [AB::Expr; NUM_LIMBS]>
        + AirTxWrite<AB, [AB::Expr; NUM_LIMBS]>,
{
    fn eval(&self, builder: &mut AB, local_core: &[AB::Var], tx: &mut TX) {
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

        tx.start(builder, is_valid.clone());
        let a = &cols.a;
        let b = &cols.b;
        let c = &cols.c;
        let rs1_addr = tx.read(builder, b.map(Into::into), is_valid.clone());
        let rs2_addr = tx.maybe_read(builder, c.map(Into::into), is_valid.clone());
        let rd_addr = tx.write(builder, a.map(Into::into), is_valid.clone());

        // For ADD, define carry[i] = (b[i] + c[i] + carry[i - 1] - a[i]) / 2^LIMB_BITS. If
        // each carry[i] is boolean and 0 <= a[i] < 2^LIMB_BITS, it can be proven that
        // a[i] = (b[i] + c[i]) % 2^LIMB_BITS as necessary. The same holds for SUB when
        // carry[i] is (a[i] + b[i] - c[i] + carry[i - 1]) / 2^LIMB_BITS.
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

        let expected_opcode = VmCoreAir::<AB, TX>::expr_to_global_expr(
            self,
            flags.iter().zip(BaseAluOpcode::iter()).fold(
                AB::Expr::ZERO,
                |acc, (flag, local_opcode)| {
                    acc + (*flag).into() * AB::Expr::from_canonical_u8(local_opcode as u8)
                },
            ),
        );

        tx.end(
            builder,
            expected_opcode,
            [
                rd_addr.pointer,
                rs1_addr.pointer,
                rs2_addr.pointer,
                rs1_addr.address_space,
                rs2_addr.address_space,
            ],
        );
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "F: Serialize + DeserializeOwned")]
pub struct BaseAluCoreRecord<F, TX, const NUM_LIMBS: usize>
where
    TX: ExecuteTxRead<F, [F; NUM_LIMBS]>
        + ExecuteTxMaybeRead<F, [F; NUM_LIMBS]>
        + ExecuteTxWrite<F, [F; NUM_LIMBS]>,
{
    pub opcode: BaseAluOpcode,
    #[serde(with = "BigArray")]
    pub a: [F; NUM_LIMBS],
    #[serde(with = "BigArray")]
    pub b: [F; NUM_LIMBS],
    #[serde(with = "BigArray")]
    pub c: [F; NUM_LIMBS],

    pub a_record: <TX as ExecuteTxWrite<F, [F; NUM_LIMBS]>>::Record,
    pub b_record: <TX as ExecuteTxRead<F, [F; NUM_LIMBS]>>::Record,
    pub c_record: <TX as ExecuteTxMaybeRead<F, [F; NUM_LIMBS]>>::Record,
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
}

impl<F, A, const NUM_LIMBS: usize, const LIMB_BITS: usize> VmCoreChip<F, A>
    for BaseAluCoreChip<NUM_LIMBS, LIMB_BITS>
where
    F: PrimeField32,
    A: VmAdapter<F>,
    A::ExecuteTx: ExecuteTx
        + ExecuteTxRead<F, [F; NUM_LIMBS]>
        + ExecuteTxMaybeRead<F, [F; NUM_LIMBS]>
        + ExecuteTxWrite<F, [F; NUM_LIMBS]>,
    for<'tx> A::TraceTx<'tx>: TraceTx<F>
        + TraceTxRead<F, Record = <A::ExecuteTx as ExecuteTxRead<F, [F; NUM_LIMBS]>>::Record>
        + TraceTxMaybeRead<
            F,
            Record = <A::ExecuteTx as ExecuteTxMaybeRead<F, [F; NUM_LIMBS]>>::Record,
        > + TraceTxWrite<F, Record = <A::ExecuteTx as ExecuteTxWrite<F, [F; NUM_LIMBS]>>::Record>,
{
    type Record = BaseAluCoreRecord<F, A::ExecuteTx, NUM_LIMBS>;
    type Air = BaseAluCoreAir<NUM_LIMBS, LIMB_BITS>;

    fn execute_instruction(
        &self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_pc: u32,
        tx: &mut A::ExecuteTx,
    ) -> Result<(u32, BaseAluCoreRecord<F, A::ExecuteTx, NUM_LIMBS>)> {
        let Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = *instruction;
        let local_opcode = BaseAluOpcode::from_usize(opcode.local_opcode_idx(self.air.offset));

        tx.start(from_pc);
        let (rs1_record, rs1_data) = tx.read(memory, MemoryAddress::new(d, b));
        let (rs2_record, rs2_data) = tx.maybe_read(memory, MemoryAddress::new(e, c));
        let b = rs1_data.map(|x| x.as_canonical_u32());
        let c = rs2_data.map(|y| y.as_canonical_u32());
        let rd_data = run_alu::<NUM_LIMBS, LIMB_BITS>(local_opcode, &b, &c);

        if local_opcode == BaseAluOpcode::ADD || local_opcode == BaseAluOpcode::SUB {
            for a_val in rd_data {
                self.bitwise_lookup_chip.request_xor(a_val, a_val);
            }
        } else {
            for (b_val, c_val) in b.iter().zip(c.iter()) {
                self.bitwise_lookup_chip.request_xor(*b_val, *c_val);
            }
        }

        let rd_data = rd_data.map(F::from_canonical_u32);
        let (rd_record, _) = tx.write(memory, MemoryAddress::new(d, a), rd_data);
        let record = BaseAluCoreRecord {
            opcode: local_opcode,
            a: rd_data,
            b: rs1_data,
            c: rs2_data,
            a_record: rd_record,
            b_record: rs1_record,
            c_record: rs2_record,
        };
        let to_pc = tx.end();

        Ok((to_pc, record))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", BaseAluOpcode::from_usize(opcode - self.air.offset))
    }

    fn generate_trace_row(
        &self,
        row_core: &mut [F],
        record: BaseAluCoreRecord<F, A::ExecuteTx, NUM_LIMBS>,
        tx: &mut A::TraceTx<'_>,
    ) {
        let buffer: &mut BaseAluCoreCols<_, NUM_LIMBS, LIMB_BITS> = row_core.borrow_mut();
        tx.start();
        tx.read(record.b_record);
        tx.maybe_read(record.c_record);
        tx.write(record.a_record);
        buffer.a = record.a;
        buffer.b = record.b;
        buffer.c = record.c;
        buffer.opcode_add_flag = F::from_bool(record.opcode == BaseAluOpcode::ADD);
        buffer.opcode_sub_flag = F::from_bool(record.opcode == BaseAluOpcode::SUB);
        buffer.opcode_xor_flag = F::from_bool(record.opcode == BaseAluOpcode::XOR);
        buffer.opcode_or_flag = F::from_bool(record.opcode == BaseAluOpcode::OR);
        buffer.opcode_and_flag = F::from_bool(record.opcode == BaseAluOpcode::AND);
        tx.end();
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

pub(super) fn run_alu<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    opcode: BaseAluOpcode,
    x: &[u32; NUM_LIMBS],
    y: &[u32; NUM_LIMBS],
) -> [u32; NUM_LIMBS] {
    match opcode {
        BaseAluOpcode::ADD => run_add::<NUM_LIMBS, LIMB_BITS>(x, y),
        BaseAluOpcode::SUB => run_subtract::<NUM_LIMBS, LIMB_BITS>(x, y),
        BaseAluOpcode::XOR => run_xor::<NUM_LIMBS, LIMB_BITS>(x, y),
        BaseAluOpcode::OR => run_or::<NUM_LIMBS, LIMB_BITS>(x, y),
        BaseAluOpcode::AND => run_and::<NUM_LIMBS, LIMB_BITS>(x, y),
    }
}

fn run_add<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u32; NUM_LIMBS],
    y: &[u32; NUM_LIMBS],
) -> [u32; NUM_LIMBS] {
    let mut z = [0u32; NUM_LIMBS];
    let mut carry = [0u32; NUM_LIMBS];
    for i in 0..NUM_LIMBS {
        z[i] = x[i] + y[i] + if i > 0 { carry[i - 1] } else { 0 };
        carry[i] = z[i] >> LIMB_BITS;
        z[i] &= (1 << LIMB_BITS) - 1;
    }
    z
}

fn run_subtract<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u32; NUM_LIMBS],
    y: &[u32; NUM_LIMBS],
) -> [u32; NUM_LIMBS] {
    let mut z = [0u32; NUM_LIMBS];
    let mut carry = [0u32; NUM_LIMBS];
    for i in 0..NUM_LIMBS {
        let rhs = y[i] + if i > 0 { carry[i - 1] } else { 0 };
        if x[i] >= rhs {
            z[i] = x[i] - rhs;
            carry[i] = 0;
        } else {
            z[i] = x[i] + (1 << LIMB_BITS) - rhs;
            carry[i] = 1;
        }
    }
    z
}

fn run_xor<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u32; NUM_LIMBS],
    y: &[u32; NUM_LIMBS],
) -> [u32; NUM_LIMBS] {
    array::from_fn(|i| x[i] ^ y[i])
}

fn run_or<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u32; NUM_LIMBS],
    y: &[u32; NUM_LIMBS],
) -> [u32; NUM_LIMBS] {
    array::from_fn(|i| x[i] | y[i])
}

fn run_and<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: &[u32; NUM_LIMBS],
    y: &[u32; NUM_LIMBS],
) -> [u32; NUM_LIMBS] {
    array::from_fn(|i| x[i] & y[i])
}
