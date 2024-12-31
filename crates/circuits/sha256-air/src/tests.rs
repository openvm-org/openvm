use std::{array, borrow::BorrowMut, cmp::max, sync::Arc};

use openvm_circuit::{
    arch::{
        instructions::riscv::RV32_CELL_BITS, testing::VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS,
    },
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, BitwiseOperationLookupChip},
    SubAir,
};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::InteractionBuilder,
    p3_air::{Air, BaseAir},
    p3_field::{AbstractField, Field, PrimeField32},
    p3_matrix::dense::RowMajorMatrix,
    p3_maybe_rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut},
    prover::types::AirProofInput,
    rap::{get_air_name, AnyRap, BaseAirWithPublicValues, PartitionedBaseAir},
    Chip, ChipUsageGetter,
};
use openvm_stark_sdk::utils::create_seeded_rng;
use rand::Rng;

use crate::{
    limbs_into_u32, Sha256Air, Sha256RoundCols, SHA256_BLOCK_U8S, SHA256_DIGEST_WIDTH, SHA256_H,
    SHA256_ROUND_WIDTH, SHA256_ROWS_PER_BLOCK, SHA256_WORD_U8S,
};

// A wrapper AIR purely for testing purposes
#[derive(Clone, Debug)]
pub struct Sha256TestAir {
    pub sub_air: Sha256Air,
}

impl<F: Field> BaseAirWithPublicValues<F> for Sha256TestAir {}
impl<F: Field> PartitionedBaseAir<F> for Sha256TestAir {}
impl<F: Field> BaseAir<F> for Sha256TestAir {
    fn width(&self) -> usize {
        <Sha256Air as BaseAir<F>>::width(&self.sub_air)
    }
}

impl<AB: InteractionBuilder> Air<AB> for Sha256TestAir {
    fn eval(&self, builder: &mut AB) {
        self.sub_air.eval(builder, 0);
    }
}

// A wrapper Chip purely for testing purposes
#[derive(Debug)]
pub struct Sha256TestChip {
    pub air: Sha256TestAir,
    pub bitwise_lookup_chip: Arc<BitwiseOperationLookupChip<8>>,
    pub records: Vec<([u8; SHA256_BLOCK_U8S], bool)>,
}

impl<SC: StarkGenericConfig> Chip<SC> for Sha256TestChip
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        Arc::new(self.air.clone())
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let air = self.air();
        let non_padded_height = self.current_trace_height();
        let height = next_power_of_two_or_zero(non_padded_height);
        let width = self.trace_width();
        let mut values = Val::<SC>::zero_vec(height * width);

        struct BlockContext {
            prev_hash: [u32; 8],
            local_block_idx: u32,
            global_block_idx: u32,
            input: [u8; SHA256_BLOCK_U8S],
            is_last_block: bool,
        }
        let mut block_ctx: Vec<BlockContext> = Vec::with_capacity(self.records.len());
        let mut prev_hash = SHA256_H;
        let mut local_block_idx = 0;
        let mut global_block_idx = 1;
        for (input, is_last_block) in self.records {
            block_ctx.push(BlockContext {
                prev_hash,
                local_block_idx,
                global_block_idx,
                input,
                is_last_block,
            });
            global_block_idx += 1;
            if is_last_block {
                local_block_idx = 0;
                prev_hash = SHA256_H;
            } else {
                local_block_idx += 1;
                prev_hash = Sha256Air::get_block_hash(&prev_hash, input);
            }
        }
        // first pass
        values
            .par_chunks_exact_mut(width * SHA256_ROWS_PER_BLOCK)
            .zip(block_ctx)
            .for_each(|(block, ctx)| {
                let BlockContext {
                    prev_hash,
                    local_block_idx,
                    global_block_idx,
                    input,
                    is_last_block,
                } = ctx;
                let input_words = array::from_fn(|i| {
                    limbs_into_u32::<SHA256_WORD_U8S>(array::from_fn(|j| {
                        input[i * SHA256_WORD_U8S + j] as u32
                    }))
                });
                self.air.sub_air.generate_block_trace(
                    block,
                    width,
                    0,
                    &input_words,
                    self.bitwise_lookup_chip.as_ref(),
                    &prev_hash,
                    is_last_block,
                    global_block_idx,
                    local_block_idx,
                    &[[Val::<SC>::ZERO; 16]; 4],
                );
            });
        // second pass: padding rows
        values[width * non_padded_height..]
            .par_chunks_mut(width)
            .for_each(|row| {
                let cols: &mut Sha256RoundCols<Val<SC>> = row.borrow_mut();
                self.air.sub_air.generate_default_row(cols);
            });
        // second pass: non-padding rows
        values[width..]
            .par_chunks_mut(width * SHA256_ROWS_PER_BLOCK)
            .take(non_padded_height / SHA256_ROWS_PER_BLOCK)
            .for_each(|chunk| {
                self.air.sub_air.generate_missing_cells(chunk, width, 0);
            });

        AirProofInput::simple(air, RowMajorMatrix::new(values, width), vec![])
    }
}

impl ChipUsageGetter for Sha256TestChip {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }
    fn current_trace_height(&self) -> usize {
        self.records.len() * SHA256_ROWS_PER_BLOCK
    }

    fn trace_width(&self) -> usize {
        max(SHA256_ROUND_WIDTH, SHA256_DIGEST_WIDTH)
    }
}

const SELF_BUS_IDX: usize = 28;
#[test]
fn rand_sha256_test() {
    let mut rng = create_seeded_rng();
    let tester = VmChipTestBuilder::default();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));
    let len = rng.gen_range(1..100);
    let random_records: Vec<_> = (0..len)
        .map(|_| (array::from_fn(|_| rng.gen::<u8>()), true))
        .collect();
    let chip = Sha256TestChip {
        air: Sha256TestAir {
            sub_air: Sha256Air::new(bitwise_bus, SELF_BUS_IDX),
        },
        bitwise_lookup_chip: bitwise_chip.clone(),
        records: random_records,
    };

    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}
