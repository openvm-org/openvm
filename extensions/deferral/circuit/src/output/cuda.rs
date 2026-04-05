use std::{array::from_fn, mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{
    arch::{DenseRecordArena, SizedRecord},
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU, Chip,
};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_stark_backend::{p3_field::PrimeCharacteristicRing, prover::AirProvingContext};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

use super::{
    DeferralOutputCols, DeferralOutputLayout, DeferralOutputMetadata, DeferralOutputRecordHeader,
    DeferralOutputRecordMut,
};
use crate::{
    cuda_abi::output::{self, DeferralOutputPerCall, DeferralOutputPerRow},
    poseidon2::{deferral_poseidon2_chip, DeferralPoseidon2SharedBuffer},
    utils::f_commit_to_bytes,
};

#[derive(new)]
pub struct DeferralOutputChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub address_bits: usize,
    pub timestamp_max_bits: usize,
    pub count: Arc<DeviceBuffer<u32>>,
    pub num_deferral_circuits: usize,
    pub poseidon2: DeferralPoseidon2SharedBuffer,
}

impl Chip<DenseRecordArena, GpuBackend> for DeferralOutputChipGpu {
    fn generate_proving_ctx(&self, mut arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let records = arena.allocated_mut();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }

        let poseidon2_chip = deferral_poseidon2_chip::<F>();
        let mut per_call = Vec::<DeferralOutputPerCall>::new();
        let mut per_row = Vec::<DeferralOutputPerRow>::new();

        let mut offset = 0usize;
        while offset < records.len() {
            let header_offset = offset;
            let header: &DeferralOutputRecordHeader = unsafe {
                &*(records.as_ptr().add(header_offset) as *const DeferralOutputRecordHeader)
            };

            let num_rows = header.num_rows as usize;
            let output_len = (num_rows - 1) * DIGEST_SIZE;
            let write_bytes = unsafe {
                std::slice::from_raw_parts(
                    records
                        .as_ptr()
                        .add(header_offset + size_of::<DeferralOutputRecordHeader>()),
                    output_len,
                )
            };

            let mut current_poseidon2_res = [F::ZERO; DIGEST_SIZE];
            let call_idx =
                u32::try_from(per_call.len()).expect("deferral output call index should fit u32");
            let header_offset_u32 =
                u32::try_from(header_offset).expect("record byte offset should fit u32");

            for section_idx in 0..num_rows {
                let sponge_inputs = if section_idx == 0 {
                    let mut input = [F::ZERO; DIGEST_SIZE];
                    input[0] = F::from_u32(header.deferral_idx);
                    input[1] = F::from_usize(output_len);
                    input
                } else {
                    let base = (section_idx - 1) * DIGEST_SIZE;
                    from_fn(|i| F::from_u8(write_bytes[base + i]))
                };

                current_poseidon2_res = poseidon2_chip.perm(
                    &sponge_inputs,
                    &current_poseidon2_res,
                    section_idx + 1 == num_rows,
                );

                per_row.push(DeferralOutputPerRow {
                    header_offset: header_offset_u32,
                    section_idx: section_idx as u32,
                    call_idx,
                    poseidon2_res: current_poseidon2_res,
                });
            }

            per_call.push(DeferralOutputPerCall {
                output_commit: f_commit_to_bytes(&current_poseidon2_res),
            });

            let layout = DeferralOutputLayout::new(DeferralOutputMetadata { num_rows });
            let record_size =
                <DeferralOutputRecordMut<'_> as SizedRecord<DeferralOutputLayout>>::size(&layout);
            let record_alignment = <DeferralOutputRecordMut<'_> as SizedRecord<
                DeferralOutputLayout,
            >>::alignment(&layout);
            offset += record_size.next_multiple_of(record_alignment);
        }
        debug_assert_eq!(offset, records.len());

        let rows_used = per_row.len();
        let trace_height = next_power_of_two_or_zero(rows_used);
        let trace_width = DeferralOutputCols::<F>::width();
        let ctx = &self.range_checker.ctx;
        let trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, ctx);

        let d_raw_records = records.to_device_on(ctx).unwrap();
        let d_per_call = per_call.to_device_on(ctx).unwrap();
        let d_per_row = per_row.to_device_on(ctx).unwrap();

        unsafe {
            output::tracegen(
                trace.buffer(),
                trace_height,
                trace_width,
                &d_raw_records,
                &d_per_call,
                &d_per_row,
                rows_used,
                &self.count,
                self.num_deferral_circuits,
                &self.range_checker.count,
                self.timestamp_max_bits as u32,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS as u32,
                self.address_bits,
                &self.poseidon2.records,
                &self.poseidon2.counts,
                &self.poseidon2.idx,
                // Length in F elements; the CUDA side converts to record count.
                self.poseidon2.records.len(),
                ctx.stream.as_raw(),
            )
            .expect("Failed to generate deferral output trace");
        }

        AirProvingContext::simple_no_pis(trace)
    }
}
