use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{arch::*, system::memory::MemoryAuxColsFactory};
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerBus},
    AlignedBorrow, ColumnsAir, StructReflection, StructReflectionHelper,
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
    adapters::{LoadStoreInstruction, Rv64LoadStoreAdapterFiller},
    loadstore::common::{adapter_context, LoadStoreRecord},
};

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct LoadStoreDoublewordCoreCols<T> {
    pub is_valid: T,
    pub is_load: T,
    pub read_data: [T; BLOCK_FE_WIDTH],
    pub prev_data: [T; BLOCK_FE_WIDTH],
}

#[derive(Debug, Clone, ColumnsAir)]
#[columns_via(LoadStoreDoublewordCoreCols<u8>)]
pub struct LoadStoreDoublewordCoreAir {
    pub offset: usize,
}

impl LoadStoreDoublewordCoreAir {
    pub fn new(offset: usize, _range_bus: VariableRangeCheckerBus) -> Self {
        Self { offset }
    }
}

impl<F: Field> BaseAir<F> for LoadStoreDoublewordCoreAir {
    fn width(&self) -> usize {
        LoadStoreDoublewordCoreCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for LoadStoreDoublewordCoreAir {}

impl<AB, I> VmCoreAir<AB, I> for LoadStoreDoublewordCoreAir
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
        let cols: &LoadStoreDoublewordCoreCols<AB::Var> = (*local_core).borrow();

        builder.assert_bool(cols.is_valid);
        builder.assert_bool(cols.is_load);
        builder.assert_zero(cols.is_load * (AB::Expr::ONE - cols.is_valid));

        let is_store = cols.is_valid - cols.is_load;
        let expected_opcode = cols.is_load * AB::Expr::from_u8(LOADD as u8)
            + is_store * AB::Expr::from_u8(STORED as u8);
        let expected_opcode = VmCoreAir::<AB, I>::expr_to_global_expr(self, expected_opcode);
        let write_data = cols.read_data.map(Into::into);

        adapter_context::<AB, I>(
            cols.is_valid.into(),
            cols.is_load.into(),
            expected_opcode,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
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
pub struct LoadStoreDoublewordFiller<A = Rv64LoadStoreAdapterFiller> {
    adapter: A,
    pub offset: usize,
}

impl<A> LoadStoreDoublewordFiller<A> {
    pub fn new(
        adapter: A,
        offset: usize,
        _range_checker_chip: SharedVariableRangeCheckerChip,
    ) -> Self {
        Self { adapter, offset }
    }
}

impl<F, A> TraceFiller<F> for LoadStoreDoublewordFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);

        let record: &LoadStoreRecord = unsafe { get_record_from_slice(&mut core_row, ()) };
        let opcode = Rv64LoadStoreOpcode::from_usize(record.local_opcode as usize);
        let read_data = record.read_data;
        let prev_data = record.prev_data;
        debug_assert_eq!(record.shift_amount, 0);
        let core_row: &mut LoadStoreDoublewordCoreCols<F> = core_row.borrow_mut();

        if !matches!(opcode, LOADD | STORED) {
            unreachable!("doubleword loadstore core only handles LOADD/STORED");
        }
        core_row.is_valid = F::ONE;
        core_row.is_load = F::from_bool(opcode == LOADD);
        core_row.read_data = read_data.map(F::from_u16);
        core_row.prev_data = prev_data.map(F::from_u16);
    }
}
