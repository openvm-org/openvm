use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{arch::*, system::memory::MemoryAuxColsFactory};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    encoder::Encoder,
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBorrow, ColumnsAir, StructReflection, StructReflectionHelper, SubAir,
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, *};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};

use crate::{
    adapters::{LoadStoreInstruction, Rv64LoadStoreAdapterFiller, RV64_BYTE_BITS},
    loadstore::common::{
        adapter_context, byte_from_cell, replace_byte, run_write_data, LoadStoreRecord, BYTE_BITS,
    },
};

const BYTE_CASES: usize = 16;
const BYTE_SELECTOR_MAX_DEGREE: u32 = 2;
pub(crate) const BYTE_SELECTOR_WIDTH: usize = 5;

#[derive(Clone, Copy)]
struct ByteCase {
    opcode: Rv64LoadStoreOpcode,
    byte_shift: usize,
}

impl ByteCase {
    fn is_load(self) -> bool {
        self.opcode == LOADBU
    }

    fn cell_shift(self) -> usize {
        self.byte_shift / 2
    }

    fn byte_idx(self) -> usize {
        self.byte_shift % 2
    }

    fn read_cell_idx(self) -> usize {
        if self.opcode == STOREB {
            0
        } else {
            self.cell_shift()
        }
    }
}

const CASES: [ByteCase; BYTE_CASES] = [
    ByteCase {
        opcode: LOADBU,
        byte_shift: 0,
    },
    ByteCase {
        opcode: LOADBU,
        byte_shift: 1,
    },
    ByteCase {
        opcode: LOADBU,
        byte_shift: 2,
    },
    ByteCase {
        opcode: LOADBU,
        byte_shift: 3,
    },
    ByteCase {
        opcode: LOADBU,
        byte_shift: 4,
    },
    ByteCase {
        opcode: LOADBU,
        byte_shift: 5,
    },
    ByteCase {
        opcode: LOADBU,
        byte_shift: 6,
    },
    ByteCase {
        opcode: LOADBU,
        byte_shift: 7,
    },
    ByteCase {
        opcode: STOREB,
        byte_shift: 0,
    },
    ByteCase {
        opcode: STOREB,
        byte_shift: 1,
    },
    ByteCase {
        opcode: STOREB,
        byte_shift: 2,
    },
    ByteCase {
        opcode: STOREB,
        byte_shift: 3,
    },
    ByteCase {
        opcode: STOREB,
        byte_shift: 4,
    },
    ByteCase {
        opcode: STOREB,
        byte_shift: 5,
    },
    ByteCase {
        opcode: STOREB,
        byte_shift: 6,
    },
    ByteCase {
        opcode: STOREB,
        byte_shift: 7,
    },
];

fn encoder() -> Encoder {
    let encoder = Encoder::new(BYTE_CASES, BYTE_SELECTOR_MAX_DEGREE, true);
    debug_assert_eq!(encoder.width(), BYTE_SELECTOR_WIDTH);
    encoder
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct LoadStoreByteCoreCols<T> {
    pub selector: [T; BYTE_SELECTOR_WIDTH],
    pub is_valid: T,
    pub is_load: T,
    pub read_cell_bytes: [T; 2],
    pub prev_cell_bytes: [T; 2],
    pub read_data: [T; BLOCK_FE_WIDTH],
    pub prev_data: [T; BLOCK_FE_WIDTH],
}

#[derive(Debug, Clone, ColumnsAir)]
#[columns_via(LoadStoreByteCoreCols<u8>)]
pub struct LoadStoreByteCoreAir {
    pub offset: usize,
    encoder: Encoder,
    bitwise_lookup_bus: BitwiseOperationLookupBus,
}

impl LoadStoreByteCoreAir {
    pub fn new(
        offset: usize,
        bitwise_lookup_bus: BitwiseOperationLookupBus,
        _range_bus: VariableRangeCheckerBus,
    ) -> Self {
        Self {
            offset,
            encoder: encoder(),
            bitwise_lookup_bus,
        }
    }
}

impl<F: Field> BaseAir<F> for LoadStoreByteCoreAir {
    fn width(&self) -> usize {
        LoadStoreByteCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for LoadStoreByteCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for LoadStoreByteCoreAir
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<([AB::Var; BLOCK_FE_WIDTH], [AB::Expr; BLOCK_FE_WIDTH])>,
    I::Writes: From<[[AB::Expr; BLOCK_FE_WIDTH]; 1]>,
    I::ProcessedInstruction: From<LoadStoreInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &LoadStoreByteCoreCols<AB::Var> = (*local_core).borrow();
        self.encoder.eval(builder, &cols.selector);
        let flags = self.encoder.flags::<AB>(&cols.selector);
        let is_valid = self.encoder.is_valid::<AB>(&cols.selector);
        let is_load = CASES
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, case)| {
                if case.is_load() {
                    acc + flags[i].clone()
                } else {
                    acc
                }
            });
        let is_store = is_valid.clone() - is_load.clone();

        builder.assert_eq(cols.is_valid, is_valid.clone());
        builder.assert_eq(cols.is_load, is_load.clone());

        self.bitwise_lookup_bus
            .send_range(cols.read_cell_bytes[0], cols.read_cell_bytes[1])
            .eval(builder, is_valid.clone());
        self.bitwise_lookup_bus
            .send_range(cols.prev_cell_bytes[0], cols.prev_cell_bytes[1])
            .eval(builder, is_store.clone());

        let read_cell =
            cols.read_cell_bytes[0] + cols.read_cell_bytes[1] * AB::Expr::from_u32(1 << BYTE_BITS);
        let expected_read_cell = CASES
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, case)| {
                acc + flags[i].clone() * cols.read_data[case.read_cell_idx()]
            });
        builder.assert_eq(read_cell.clone(), expected_read_cell);

        let prev_cell =
            cols.prev_cell_bytes[0] + cols.prev_cell_bytes[1] * AB::Expr::from_u32(1 << BYTE_BITS);
        let expected_prev_cell = CASES
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, case)| {
                if case.opcode == STOREB {
                    acc + flags[i].clone() * cols.prev_data[case.cell_shift()]
                } else {
                    acc
                }
            });
        builder.assert_eq(is_store.clone() * prev_cell, expected_prev_cell);

        let expected_opcode = CASES
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, case)| {
                acc + flags[i].clone() * AB::Expr::from_u8(case.opcode as u8)
            });
        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(self, expected_opcode);
        let load_shift_amount = CASES
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, case)| {
                if case.is_load() {
                    acc + flags[i].clone() * AB::Expr::from_usize(case.byte_shift)
                } else {
                    acc
                }
            });
        let store_shift_amount = CASES
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (i, case)| {
                if case.is_load() {
                    acc
                } else {
                    acc + flags[i].clone() * AB::Expr::from_usize(case.byte_shift)
                }
            });

        let write_data = std::array::from_fn(|i| {
            CASES
                .iter()
                .enumerate()
                .fold(AB::Expr::ZERO, |acc, (case_idx, case)| {
                    let term = match case.opcode {
                        LOADBU => {
                            let selected_case_byte = if case.byte_idx() == 0 {
                                cols.read_cell_bytes[0].into()
                            } else {
                                cols.read_cell_bytes[1].into()
                            };
                            if i == 0 {
                                selected_case_byte
                            } else {
                                AB::Expr::ZERO
                            }
                        }
                        STOREB => {
                            if i == case.cell_shift() {
                                if case.byte_idx() == 0 {
                                    cols.read_cell_bytes[0]
                                        + cols.prev_cell_bytes[1]
                                            * AB::Expr::from_u32(1 << BYTE_BITS)
                                } else {
                                    cols.prev_cell_bytes[0]
                                        + cols.read_cell_bytes[0]
                                            * AB::Expr::from_u32(1 << BYTE_BITS)
                                }
                            } else {
                                cols.prev_data[i].into()
                            }
                        }
                        _ => unreachable!(),
                    };
                    acc + flags[case_idx].clone() * term
                })
        });
        adapter_context::<AB, I>(
            cols.is_valid.into(),
            cols.is_load.into(),
            expected_opcode,
            load_shift_amount,
            store_shift_amount,
            cols.read_data,
            cols.prev_data,
            write_data,
        )
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

#[derive(Clone)]
pub struct LoadStoreByteFiller<A = Rv64LoadStoreAdapterFiller> {
    adapter: A,
    pub offset: usize,
    encoder: Encoder,
    bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
}

impl<A> LoadStoreByteFiller<A> {
    pub fn new(
        adapter: A,
        offset: usize,
        bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
        _range_checker_chip: SharedVariableRangeCheckerChip,
    ) -> Self {
        Self {
            adapter,
            offset,
            encoder: encoder(),
            bitwise_lookup_chip,
        }
    }
}

impl<F, A> TraceFiller<F> for LoadStoreByteFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);

        let record: &LoadStoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let opcode = Rv64LoadStoreOpcode::from_usize(record.local_opcode as usize);
        let shift = record.shift_amount as usize;
        let read_data = record.read_data;
        let prev_data = record.prev_data;
        let core_row: &mut LoadStoreByteCoreCols<F> = core_row.borrow_mut();
        let cell_shift = shift / 2;
        let byte_idx = shift % 2;
        let case_idx = CASES
            .iter()
            .position(|case| case.opcode == opcode && case.byte_shift == shift)
            .expect("invalid byte loadstore opcode/shift");

        let read_cell = if opcode == STOREB {
            read_data[0]
        } else {
            read_data[cell_shift]
        };
        let read_cell_bytes = [byte_from_cell(read_cell, 0), byte_from_cell(read_cell, 1)];
        self.bitwise_lookup_chip
            .request_range(read_cell_bytes[0] as u32, read_cell_bytes[1] as u32);
        core_row.read_cell_bytes = read_cell_bytes.map(F::from_u16);

        if opcode == STOREB {
            let prev_cell_bytes = [
                byte_from_cell(prev_data[cell_shift], 0),
                byte_from_cell(prev_data[cell_shift], 1),
            ];
            self.bitwise_lookup_chip
                .request_range(prev_cell_bytes[0] as u32, prev_cell_bytes[1] as u32);
            core_row.prev_cell_bytes = prev_cell_bytes.map(F::from_u16);
            debug_assert_eq!(
                run_write_data(opcode, read_data, prev_data, shift)[cell_shift],
                replace_byte(prev_data[cell_shift], byte_idx, read_cell_bytes[0])
            );
        } else {
            core_row.prev_cell_bytes = [F::ZERO; 2];
        }

        core_row.read_data = read_data.map(F::from_u16);
        core_row.prev_data = prev_data.map(F::from_u16);
        core_row.is_load = F::from_bool(opcode == LOADBU);
        core_row.is_valid = F::ONE;
        let pt: [u32; BYTE_SELECTOR_WIDTH] = self.encoder.get_flag_pt(case_idx).try_into().unwrap();
        core_row.selector = pt.map(F::from_u32);
    }
}
