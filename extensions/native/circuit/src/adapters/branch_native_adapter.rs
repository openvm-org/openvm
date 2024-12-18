use std::{
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    marker::PhantomData,
};

use openvm_circuit::{
    arch::{
        AdapterAirContext, AdapterRuntimeContext, BasicAdapterInterface, ExecutionBridge,
        ExecutionBus, ExecutionState, ImmInstruction, Result, VmAdapterAir, VmAdapterChip,
        VmAdapterInterface,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadOrImmediateAuxCols},
            MemoryAddress, MemoryAuxColsFactory, MemoryController, MemoryControllerRef,
        },
        native_adapter::NativeReadRecord,
        program::ProgramBus,
    },
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{AbstractField, Field, PrimeField32},
};

#[derive(Debug)]
pub struct BranchNativeAdapterChip<F: Field> {
    pub air: BranchNativeAdapterAir,
    _marker: PhantomData<F>,
}

impl<F: PrimeField32> BranchNativeAdapterChip<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_controller: MemoryControllerRef<F>,
    ) -> Self {
        let memory_controller = RefCell::borrow(&memory_controller);
        let memory_bridge = memory_controller.memory_bridge();
        Self {
            air: BranchNativeAdapterAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
            },
            _marker: PhantomData,
        }
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct BranchNativeAdapterReadCols<T> {
    pub address: MemoryAddress<T, T>,
    pub read_aux: MemoryReadOrImmediateAuxCols<T>,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct BranchNativeAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub reads_aux: [BranchNativeAdapterReadCols<T>; 2],
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct BranchNativeAdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F: Field> BaseAir<F> for BranchNativeAdapterAir {
    fn width(&self) -> usize {
        BranchNativeAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for BranchNativeAdapterAir {
    type Interface = BasicAdapterInterface<AB::Expr, ImmInstruction<AB::Expr>, 2, 0, 1, 1>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &BranchNativeAdapterCols<_> = local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta = 0usize;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        self.memory_bridge
            .read_or_immediate(
                cols.reads_aux[0].address,
                ctx.reads[0][0].clone(),
                timestamp_pp(),
                &cols.reads_aux[0].read_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .read_or_immediate(
                cols.reads_aux[1].address,
                ctx.reads[1][0].clone(),
                timestamp_pp(),
                &cols.reads_aux[1].read_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    cols.reads_aux[0].address.pointer.into(),
                    cols.reads_aux[1].address.pointer.into(),
                    ctx.instruction.immediate,
                    cols.reads_aux[0].address.address_space.into(),
                    cols.reads_aux[1].address.address_space.into(),
                ],
                cols.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &BranchNativeAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

impl<F: PrimeField32> VmAdapterChip<F> for BranchNativeAdapterChip<F> {
    type ReadRecord = NativeReadRecord<F, 2>;
    type WriteRecord = ExecutionState<u32>;
    type Air = BranchNativeAdapterAir;
    type Interface = BasicAdapterInterface<F, ImmInstruction<F>, 2, 0, 1, 1>;

    fn preprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
    ) -> Result<(
        <Self::Interface as VmAdapterInterface<F>>::Reads,
        Self::ReadRecord,
    )> {
        let Instruction { a, b, d, e, .. } = *instruction;

        let reads = vec![memory.read::<1>(d, a), memory.read::<1>(e, b)];
        let i_reads: [_; 2] = std::array::from_fn(|i| reads[i].data);

        Ok((
            i_reads,
            Self::ReadRecord {
                reads: reads.try_into().unwrap(),
            },
        ))
    }

    fn postprocess(
        &mut self,
        memory: &mut MemoryController<F>,
        _instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
        output: AdapterRuntimeContext<F, Self::Interface>,
        _read_record: &Self::ReadRecord,
    ) -> Result<(ExecutionState<u32>, Self::WriteRecord)> {
        Ok((
            ExecutionState {
                pc: output.to_pc.unwrap_or(from_state.pc + DEFAULT_PC_STEP),
                timestamp: memory.timestamp(),
            },
            from_state,
        ))
    }

    fn generate_trace_row(
        &self,
        row_slice: &mut [F],
        read_record: Self::ReadRecord,
        write_record: Self::WriteRecord,
        aux_cols_factory: &MemoryAuxColsFactory<F>,
    ) {
        let row_slice: &mut BranchNativeAdapterCols<_> = row_slice.borrow_mut();

        row_slice.from_state = write_record.map(F::from_canonical_u32);
        row_slice.reads_aux = read_record.reads.map(|x| {
            let address = MemoryAddress::new(x.address_space, x.pointer);
            BranchNativeAdapterReadCols {
                address,
                read_aux: aux_cols_factory.make_read_or_immediate_aux_cols(x),
            }
        });
    }

    fn air(&self) -> &Self::Air {
        &self.air
    }
}
