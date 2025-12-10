use std::sync::{Arc, Mutex};

use derive_new::new;
use openvm_circuit::{
    arch::{Arena, DenseRecordArena, MatrixRecordArena, SizedRecord},
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChip, var_range::VariableRangeCheckerChip,
};
use openvm_cuda_backend::{
    chip::{cpu_proving_ctx_to_gpu, get_empty_air_proving_ctx},
    prover_backend::GpuBackend,
};
use openvm_stark_backend::{
    config::Val, prover::types::AirProvingContext, Chip,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

use crate::{
    Sha2BlockHasherChip, Sha2Config, Sha2MainChip, Sha2Metadata, Sha2RecordLayout, Sha2RecordMut,
    Sha2SharedRecords,
};
use openvm_sha2_air::{Sha256Config, Sha512Config};

/// Generic hybrid GPU wrapper that reuses CPU tracegen and converts the proving
/// context to GPU.
#[derive(new)]
pub struct Sha2MainChipGpu<C: Sha2Config> {
    cpu_chip: Sha2MainChip<Val<BabyBearPoseidon2Config>, C>,
}

impl<C> Chip<DenseRecordArena, GpuBackend> for Sha2MainChipGpu<C>
where
    C: Sha2Config,
{
    fn generate_proving_ctx(&self, mut arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let records = arena.allocated_mut();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }

        // Move records from the dense arena into a matrix arena so the CPU chip
        // can consume them. One row per record (Sha2Metadata::get_num_rows == 1).
        let layout = Sha2RecordLayout::new(Sha2Metadata {
            variant: C::VARIANT,
        });
        let record_size = <Sha2RecordMut as SizedRecord<_>>::size(&layout);
        let record_alignment = <Sha2RecordMut as SizedRecord<_>>::alignment(&layout);
        let aligned_record_size = record_size.next_multiple_of(record_alignment);
        let num_records = records.len() / aligned_record_size;

        let mut matrix_arena = MatrixRecordArena::<Val<BabyBearPoseidon2Config>>::with_capacity(
            next_power_of_two_or_zero(num_records),
            C::MAIN_CHIP_WIDTH,
        );
        arena
            .get_record_seeker::<Sha2RecordMut, Sha2RecordLayout>()
            .transfer_to_matrix_arena(&mut matrix_arena);

        let cpu_ctx = self.cpu_chip.generate_proving_ctx(matrix_arena);
        cpu_proving_ctx_to_gpu(cpu_ctx)
    }
}

/// Generic hybrid GPU wrapper that reuses CPU block-hasher tracegen.
#[derive(new)]
pub struct Sha2BlockHasherChipGpu<C: Sha2Config> {
    cpu_chip: Sha2BlockHasherChip<Val<BabyBearPoseidon2Config>, C>,
}

impl<C> Chip<DenseRecordArena, GpuBackend> for Sha2BlockHasherChipGpu<C>
where
    C: Sha2Config,
{
    fn generate_proving_ctx(&self, _: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        // CPU block-hasher chip ignores the arena parameter.
        let cpu_ctx = self.cpu_chip.generate_proving_ctx(());
        cpu_proving_ctx_to_gpu(cpu_ctx)
    }
}

impl<C: Sha2Config> Sha2BlockHasherChipGpu<C> {
    pub fn shared_records(
        &self,
    ) -> Arc<Mutex<Option<Sha2SharedRecords<Val<BabyBearPoseidon2Config>>>>> {
        self.cpu_chip.records.clone()
    }
}

/// Helper to construct CPU chips and shared state for hybrid GPU wrappers.
pub fn make_hybrid_chips<C: Sha2Config>(
    range_checker_cpu: Arc<VariableRangeCheckerChip>,
    bitwise_cpu: Arc<BitwiseOperationLookupChip<8>>,
    pointer_max_bits: usize,
    timestamp_max_bits: usize,
) -> (
    Sha2MainChip<Val<BabyBearPoseidon2Config>, C>,
    Sha2BlockHasherChip<Val<BabyBearPoseidon2Config>, C>,
    Arc<Mutex<Option<Sha2SharedRecords<Val<BabyBearPoseidon2Config>>>>>,
) {
    // Shared records buffer between main and block-hasher chips.
    let records = Arc::new(Mutex::new(None));

    // The CPU chips expect a SharedMemoryHelper built from the CPU range
    // checker.
    let mem_helper = openvm_circuit::system::memory::SharedMemoryHelper::new(
        range_checker_cpu,
        timestamp_max_bits,
    );

    let main = Sha2MainChip::<Val<BabyBearPoseidon2Config>, C>::new(
        records.clone(),
        bitwise_cpu.clone(),
        pointer_max_bits,
        mem_helper.clone(),
    );
    let block = Sha2BlockHasherChip::<Val<BabyBearPoseidon2Config>, C>::new(
        bitwise_cpu,
        pointer_max_bits,
        mem_helper,
        records.clone(),
    );
    (main, block, records)
}

// Convenience aliases for the common BabyBear+SHA variants.
pub type Sha256VmChipGpu = Sha2MainChipGpu<Sha256Config>;
pub type Sha256BlockHasherChipGpu = Sha2BlockHasherChipGpu<Sha256Config>;
pub type Sha512VmChipGpu = Sha2MainChipGpu<Sha512Config>;
pub type Sha512BlockHasherChipGpu = Sha2BlockHasherChipGpu<Sha512Config>;
