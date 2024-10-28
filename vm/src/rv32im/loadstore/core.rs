use std::borrow::{Borrow, BorrowMut};

use ax_circuit_derive::AlignedBorrow;
use ax_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use axvm_instructions::instruction::Instruction;
use p3_air::{AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};

use crate::{
    arch::{
        instructions::{
            Rv32LoadStoreOpcode::{self, *},
            UsizeOpcode,
        },
        AdapterAirContext, AdapterRuntimeContext, Result, VmAdapterInterface, VmCoreAir,
        VmCoreChip,
    },
    rv32im::adapters::LoadStoreInstruction,
};

/// LoadStore Core Chip handles byte/halfword into word conversions and unsigned extends
/// This chip uses read_data and prev_data to constrain the write_data
/// It also handles the shifting in case of not 4 byte aligned instructions
/// This chips treats each (opcode, shift) pair as a seperate instruction
#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow)]
pub struct LoadStoreCoreCols<T, const NUM_CELLS: usize> {
    pub flags: [T; 4],
    /// we need to keep the degree of is_valid and is_load to 1
    pub is_valid: T,
    pub is_load: T,

    pub read_data: [T; NUM_CELLS],
    pub prev_data: [T; NUM_CELLS],
    /// write_data will be constrained against read_data and prev_data
    /// depending on the opcode and the shift amount
    pub write_data: [T; NUM_CELLS],
}

#[derive(Debug, Clone)]
pub struct LoadStoreCoreRecord<F, const NUM_CELLS: usize> {
    pub opcode: Rv32LoadStoreOpcode,
    pub shift: u32,
    pub read_data: [F; NUM_CELLS],
    pub prev_data: [F; NUM_CELLS],
    pub write_data: [F; NUM_CELLS],
}

#[derive(Debug, Clone)]
pub struct LoadStoreCoreAir<const NUM_CELLS: usize> {
    pub offset: usize,
}

impl<F: Field, const NUM_CELLS: usize> BaseAir<F> for LoadStoreCoreAir<NUM_CELLS> {
    fn width(&self) -> usize {
        LoadStoreCoreCols::<F, NUM_CELLS>::width()
    }
}

impl<F: Field, const NUM_CELLS: usize> BaseAirWithPublicValues<F> for LoadStoreCoreAir<NUM_CELLS> {}

impl<AB, I, const NUM_CELLS: usize> VmCoreAir<AB, I> for LoadStoreCoreAir<NUM_CELLS>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Var; NUM_CELLS]; 2]>,
    I::Writes: From<[[AB::Expr; NUM_CELLS]; 1]>,
    I::ProcessedInstruction: From<LoadStoreInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &LoadStoreCoreCols<AB::Var, NUM_CELLS> = (*local_core).borrow();
        let LoadStoreCoreCols::<AB::Var, NUM_CELLS> {
            read_data,
            prev_data,
            write_data,
            flags,
            is_valid,
            is_load,
        } = *cols;

        let get_expr_12 =
            |x: &AB::Expr| (x.clone() - AB::Expr::one()) * (x.clone() - AB::Expr::two());

        builder.assert_bool(is_valid);
        let sum = flags.iter().fold(AB::Expr::zero(), |acc, &flag| {
            builder.assert_zero(flag * get_expr_12(&flag.into()));
            acc + flag
        });
        builder.assert_zero(sum.clone() * get_expr_12(&sum));
        // when sum is 0, is_valid must be 0
        builder.when(get_expr_12(&sum)).assert_zero(is_valid);

        // We will use the following mapping for interpreting the opcodes
        // the appended digit to each opcode is the shift amount
        /*
           0: loadw0
           1: loadhu0
           2: loadhu2
           3: loadbu0
           4: loadbu1
           5: loadbu2
           6: loadbu3
           7: storew0
           8: storeh0
           9: storeh2
           10: storeb0
           11: storeb1
           12: storeb2
           13: storeb3
        */
        let inv_2 = AB::F::from_canonical_u32(2).inverse();
        let mut opcodes = vec![];
        for flag in flags {
            opcodes.push(flag * (flag - AB::F::one()) * inv_2);
        }
        for flag in flags {
            opcodes.push(flag * (sum.clone() - AB::F::two()) * AB::F::neg_one());
        }
        (0..4).for_each(|i| {
            ((i + 1)..4).for_each(|j| opcodes.push(flags[i] * flags[j]));
        });

        let opcode_when = |idxs: &[usize]| -> AB::Expr {
            idxs.iter()
                .fold(AB::Expr::zero(), |acc, &idx| acc + opcodes[idx].clone())
        };

        // constrain that is_load matches the opcode
        builder.assert_eq(is_load, opcode_when(&[0, 1, 2, 3, 4, 5, 6]));

        // there are three parts to write_data:
        // 1st limb is always read_data
        // 2nd to (NUM_CELLS/2)th limbs are read_data if loadw/loadhu/storew/storeh
        //                                  prev_data if storeb
        //                                  zero if loadbu
        // (NUM_CELLS/2 + 1)th to last limbs are read_data if loadw/storew
        //                                  prev_data if storeb/storeh
        //                                  zero if loadbu/loadhu
        // Shifting needs to be carefully handled in case by case basis
        // refer to [run_write_data] for the expected behavior in each case
        for (i, cell) in write_data.iter().enumerate() {
            // handling loads, expected_load_val = 0 if a store operation is happening
            let expected_load_val = if i == 0 {
                opcode_when(&[0, 1, 3]) * read_data[0]
                    + opcode_when(&[4]) * read_data[1]
                    + opcode_when(&[2, 5]) * read_data[2]
                    + opcode_when(&[6]) * read_data[3]
            } else if i < NUM_CELLS / 2 {
                opcode_when(&[0, 1]) * read_data[i] + opcode_when(&[2]) * read_data[i + 2]
                // + opcode_when(&[3, 4, 5, 6]) * AB::Expr::zero()
            } else {
                opcode_when(&[0]) * read_data[i] // + opcode_when(&[1, 2, 3, 4, 5, 6]) * AB::Expr::zero()
            };

            // handling stores, expected_store_val = 0 if a load operation is happening
            let expected_store_val = if i == 0 {
                opcode_when(&[7, 8, 10]) * read_data[i]
                    + opcode_when(&[9, 11, 12, 13]) * prev_data[i]
            } else if i == 1 {
                opcode_when(&[11]) * read_data[i - 1]
                    + opcode_when(&[7, 8]) * read_data[i]
                    + opcode_when(&[9, 10, 12, 13]) * prev_data[i]
            } else if i == 2 {
                opcode_when(&[9, 12]) * read_data[i - 2]
                    + opcode_when(&[7]) * read_data[i]
                    + opcode_when(&[8, 10, 11, 13]) * prev_data[i]
            } else if i == 3 {
                opcode_when(&[13]) * read_data[i - 3]
                    + opcode_when(&[9]) * read_data[i - 2]
                    + opcode_when(&[7]) * read_data[i]
                    + opcode_when(&[8, 10, 11, 12]) * prev_data[i]
            } else {
                opcode_when(&[7]) * read_data[i]
                    + opcode_when(&[10, 11, 12, 13]) * prev_data[i]
                    + opcode_when(&[8])
                        * if i < NUM_CELLS / 2 {
                            read_data[i]
                        } else {
                            prev_data[i]
                        }
                    + opcode_when(&[9])
                        * if i + 2 < NUM_CELLS / 2 {
                            read_data[i - 2]
                        } else {
                            prev_data[i]
                        }
            };
            let expected_val = expected_load_val + expected_store_val;
            builder.assert_eq(*cell, expected_val);
        }

        let expected_opcode = opcode_when(&[0]) * AB::Expr::from_canonical_u8(LOADW as u8)
            + opcode_when(&[1, 2]) * AB::Expr::from_canonical_u8(LOADHU as u8)
            + opcode_when(&[3, 4, 5, 6]) * AB::Expr::from_canonical_u8(LOADBU as u8)
            + opcode_when(&[7]) * AB::Expr::from_canonical_u8(STOREW as u8)
            + opcode_when(&[8, 9]) * AB::Expr::from_canonical_u8(STOREH as u8)
            + opcode_when(&[10, 11, 12, 13]) * AB::Expr::from_canonical_u8(STOREB as u8)
            + AB::Expr::from_canonical_usize(self.offset);

        let load_shift_amount = opcode_when(&[4]) * AB::Expr::one()
            + opcode_when(&[2, 5]) * AB::Expr::two()
            + opcode_when(&[6]) * AB::Expr::from_canonical_u32(3);

        let store_shift_amount = opcode_when(&[11]) * AB::Expr::one()
            + opcode_when(&[9, 12]) * AB::Expr::two()
            + opcode_when(&[13]) * AB::Expr::from_canonical_u32(3);

        AdapterAirContext {
            to_pc: None,
            reads: [prev_data, read_data].into(),
            writes: [write_data.map(|x| x.into())].into(),
            instruction: LoadStoreInstruction {
                is_valid: is_valid.into(),
                opcode: expected_opcode,
                is_load: is_load.into(),
                load_shift_amount,
                store_shift_amount,
            }
            .into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoadStoreCoreChip<const NUM_CELLS: usize> {
    pub air: LoadStoreCoreAir<NUM_CELLS>,
}

impl<const NUM_CELLS: usize> LoadStoreCoreChip<NUM_CELLS> {
    pub fn new(offset: usize) -> Self {
        Self {
            air: LoadStoreCoreAir::<NUM_CELLS> { offset },
        }
    }
}

impl<F: PrimeField32, I: VmAdapterInterface<F>, const NUM_CELLS: usize> VmCoreChip<F, I>
    for LoadStoreCoreChip<NUM_CELLS>
where
    I::Reads: Into<([[F; NUM_CELLS]; 2], F)>,
    I::Writes: From<[[F; NUM_CELLS]; 1]>,
{
    type Record = LoadStoreCoreRecord<F, NUM_CELLS>;
    type Air = LoadStoreCoreAir<NUM_CELLS>;

    #[allow(clippy::type_complexity)]
    fn execute_instruction(
        &self,
        instruction: &Instruction<F>,
        _from_pc: u32,
        reads: I::Reads,
    ) -> Result<(AdapterRuntimeContext<F, I>, Self::Record)> {
        let local_opcode = Rv32LoadStoreOpcode::from_usize(instruction.opcode - self.air.offset);

        let (reads, shift_amount) = reads.into();
        let shift = shift_amount.as_canonical_u32();
        let prev_data = reads[0];
        let read_data = reads[1];
        let write_data = run_write_data(local_opcode, read_data, prev_data, shift);
        let output = AdapterRuntimeContext::without_pc([write_data]);

        Ok((
            output,
            LoadStoreCoreRecord {
                opcode: local_opcode,
                shift,
                prev_data,
                read_data,
                write_data,
            },
        ))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv32LoadStoreOpcode::from_usize(opcode - self.air.offset)
        )
    }

    fn generate_trace_row(&self, row_slice: &mut [F], record: Self::Record) {
        let core_cols: &mut LoadStoreCoreCols<F, NUM_CELLS> = row_slice.borrow_mut();
        let opcode = record.opcode;
        let flags = &mut core_cols.flags;
        *flags = [F::zero(); 4];
        match (opcode, record.shift) {
            (LOADW, 0) => flags[0] = F::two(),
            (LOADHU, 0) => flags[1] = F::two(),
            (LOADHU, 2) => flags[2] = F::two(),
            (LOADBU, 0) => flags[3] = F::two(),

            (LOADBU, 1) => flags[0] = F::one(),
            (LOADBU, 2) => flags[1] = F::one(),
            (LOADBU, 3) => flags[2] = F::one(),
            (STOREW, 0) => flags[3] = F::one(),

            (STOREH, 0) => (flags[0], flags[1]) = (F::one(), F::one()),
            (STOREH, 2) => (flags[0], flags[2]) = (F::one(), F::one()),
            (STOREB, 0) => (flags[0], flags[3]) = (F::one(), F::one()),
            (STOREB, 1) => (flags[1], flags[2]) = (F::one(), F::one()),
            (STOREB, 2) => (flags[1], flags[3]) = (F::one(), F::one()),
            (STOREB, 3) => (flags[2], flags[3]) = (F::one(), F::one()),
            _ => unreachable!(),
        };
        core_cols.prev_data = record.prev_data;
        core_cols.read_data = record.read_data;
        core_cols.is_valid = F::one();
        core_cols.is_load = F::from_bool([LOADW, LOADHU, LOADBU].contains(&opcode));
        core_cols.write_data = record.write_data;
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}

pub(super) fn run_write_data<F: PrimeField32, const NUM_CELLS: usize>(
    opcode: Rv32LoadStoreOpcode,
    read_data: [F; NUM_CELLS],
    prev_data: [F; NUM_CELLS],
    shift: u32,
) -> [F; NUM_CELLS] {
    let shift = shift as usize;
    let mut write_data = read_data;
    match (opcode, shift) {
        (LOADW, 0) => (),
        (LOADBU, 0) | (LOADBU, 1) | (LOADBU, 2) | (LOADBU, 3) => {
            for cell in write_data.iter_mut().take(NUM_CELLS).skip(1) {
                *cell = F::zero();
            }
            write_data[0] = read_data[shift];
        }
        (LOADHU, 0) | (LOADHU, 2) => {
            for cell in write_data.iter_mut().take(NUM_CELLS).skip(NUM_CELLS / 2) {
                *cell = F::zero();
            }
            for (i, cell) in write_data.iter_mut().take(NUM_CELLS / 2).enumerate() {
                *cell = read_data[i + shift];
            }
        }
        (STOREW, 0) => (),
        (STOREB, 0) | (STOREB, 1) | (STOREB, 2) | (STOREB, 3) => {
            write_data = prev_data;
            write_data[shift] = read_data[0];
        }
        (STOREH, 0) | (STOREH, 2) => {
            write_data = prev_data;
            write_data[shift..(NUM_CELLS / 2 + shift)]
                .copy_from_slice(&read_data[..(NUM_CELLS / 2)]);
        }
        _ => unreachable!(),
    };
    write_data
}
