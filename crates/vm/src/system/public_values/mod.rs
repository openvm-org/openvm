use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{
    arch::{ExecutionError, InsExecutorE1, VmAirWrapper, VmChipWrapper, VmExecutionState},
    system::{
        native_adapter::{NativeAdapterAir, NativeAdapterChip},
        public_values::core::{PublicValuesCoreAir, PublicValuesCoreChip},
    },
};

use super::memory::online::GuestMemory;

mod columns;
/// Chip to publish custom public values from VM programs.
pub mod core;

#[cfg(test)]
mod tests;

pub type PublicValuesAir = VmAirWrapper<NativeAdapterAir<2, 0>, PublicValuesCoreAir>;
pub type PublicValuesChip<F> =
    VmChipWrapper<F, NativeAdapterChip<F, 2, 0>, PublicValuesCoreChip<F>>;

impl<F> InsExecutorE1<F> for PublicValuesCoreChip<F>
where
    F: PrimeField32,
{
    fn execute_e1<Mem, Ctx>(
        &mut self,
        state: &mut VmExecutionState<Mem, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError>
    where
        Mem: GuestMemory,
    {
        let Instruction { b, c, e, f, .. } = instruction;

        let [value]: [F; 1] = unsafe {
            state
                .memory
                .read(e.as_canonical_u32(), b.as_canonical_u32())
        };
        let [index]: [F; 1] = unsafe {
            state
                .memory
                .read(f.as_canonical_u32(), c.as_canonical_u32())
        };
        {
            let idx: usize = index.as_canonical_u32() as usize;
            let mut custom_pvs = self.custom_pvs.lock().unwrap();

            if custom_pvs[idx].is_none() {
                custom_pvs[idx] = Some(value);
            } else {
                panic!("Custom public value {} already set", idx);
            }
        }
        // TODO(ayush): should there be a write?

        state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}
