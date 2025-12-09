use std::{
    borrow::{Borrow, BorrowMut},
    mem::{align_of, size_of},
};

use openvm_circuit::{
    arch::*,
    system::memory::online::TracingMemory,
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::XorinVmExecutor;

#[derive(Clone, Copy)]
pub struct XorinVmMetadata {
    pub len: usize,
}

impl MultiRowMetadata for XorinVmMetadata {
    fn get_num_rows(&self) -> usize {
        0
    }
}

pub(crate) type XorinVmRecordLayout = MultiRowLayout<XorinVmMetadata>;

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct XorinVmRecordHeader {
    pub len: u32,
}

pub struct XorinVmRecordMut<'a> {
    pub inner: &'a mut XorinVmRecordHeader,
}

impl<'a> CustomBorrow<'a, XorinVmRecordMut<'a>, XorinVmRecordLayout> for [u8] {
    fn custom_borrow(&'a mut self, _layout: XorinVmRecordLayout) -> XorinVmRecordMut<'a> {
        let (_record_buf, _) = self.split_at_mut(size_of::<XorinVmRecordHeader>());
        XorinVmRecordMut {
            inner: _record_buf.borrow_mut(),
        }
    }

    unsafe fn extract_layout(&self) -> XorinVmRecordLayout {
        let header: &XorinVmRecordHeader = self.borrow();
        XorinVmRecordLayout {
            metadata: XorinVmMetadata {
                len: header.len as usize,
            },
        }
    }
}

impl SizedRecord<XorinVmRecordLayout> for XorinVmRecordMut<'_> {
    fn size(_layout: &XorinVmRecordLayout) -> usize {
        size_of::<XorinVmRecordHeader>()
    }

    fn alignment(_layout: &XorinVmRecordLayout) -> usize {
        align_of::<XorinVmRecordHeader>()
    }
}

impl<F, RA> PreflightExecutor<F, RA> for XorinVmExecutor
where
    F: PrimeField32,
    for<'buf> RA: RecordArena<'buf, XorinVmRecordLayout, XorinVmRecordMut<'buf>>,
{
    fn get_opcode_name(&self, _: usize) -> String {
        todo!()
    }

    fn execute(
        &self,
        _state: VmStateMut<F, TracingMemory, RA>,
        _instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        todo!()
    }
}