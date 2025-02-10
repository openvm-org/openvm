use std::{cmp::max, sync::Arc};

use openvm_circuit::arch::{
    instructions::riscv::RV32_CELL_BITS,
    testing::{VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip},
    SubAir,
};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::InteractionBuilder,
    p3_air::{Air, BaseAir},
    p3_field::{Field, PrimeField32},
    prover::types::AirProofInput,
    rap::{get_air_name, BaseAirWithPublicValues, PartitionedBaseAir},
    AirRef, Chip, ChipUsageGetter,
};
use openvm_stark_sdk::utils::create_seeded_rng;
use rand::Rng;

use crate::{Sha256Config, Sha512Config, ShaAir, ShaConfig, ShaPrecomputedValues};

// A wrapper AIR purely for testing purposes
#[derive(Clone, Debug)]
pub struct ShaTestAir<C: ShaConfig + ShaPrecomputedValues<C::Word>> {
    pub sub_air: ShaAir<C>,
}

impl<F: Field, C: ShaConfig + ShaPrecomputedValues<C::Word>> BaseAirWithPublicValues<F>
    for ShaTestAir<C>
{
}
impl<F: Field, C: ShaConfig + ShaPrecomputedValues<C::Word>> PartitionedBaseAir<F>
    for ShaTestAir<C>
{
}
impl<F: Field, C: ShaConfig + ShaPrecomputedValues<C::Word>> BaseAir<F> for ShaTestAir<C> {
    fn width(&self) -> usize {
        <ShaAir<C> as BaseAir<F>>::width(&self.sub_air)
    }
}

impl<AB: InteractionBuilder, C: ShaConfig + ShaPrecomputedValues<C::Word>> Air<AB>
    for ShaTestAir<C>
{
    fn eval(&self, builder: &mut AB) {
        self.sub_air.eval(builder, 0);
    }
}

// A wrapper Chip purely for testing purposes
pub struct ShaTestChip<C: ShaConfig + ShaPrecomputedValues<C::Word>> {
    pub air: ShaTestAir<C>,
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,
    pub records: Vec<(Vec<u8>, bool)>, // length of inner vec is BLOCK_U8S
}

impl<SC: StarkGenericConfig, C: ShaConfig + ShaPrecomputedValues<C::Word> + 'static> Chip<SC>
    for ShaTestChip<C>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air.clone())
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let trace = crate::generate_trace::<Val<SC>, C>(
            &self.air.sub_air,
            self.bitwise_lookup_chip.clone(),
            self.records,
        );
        AirProofInput::simple_no_pis(trace)
    }
}

impl<C: ShaConfig + ShaPrecomputedValues<C::Word>> ChipUsageGetter for ShaTestChip<C> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }
    fn current_trace_height(&self) -> usize {
        self.records.len() * C::ROWS_PER_BLOCK
    }

    fn trace_width(&self) -> usize {
        max(C::ROUND_WIDTH, C::DIGEST_WIDTH)
    }
}

const SELF_BUS_IDX: usize = 28;
fn rand_sha_test<C: ShaConfig + ShaPrecomputedValues<C::Word> + 'static>() {
    let mut rng = create_seeded_rng();
    let tester = VmChipTestBuilder::default();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);
    let len = rng.gen_range(1..100);
    let random_records: Vec<_> = (0..len)
        .map(|_| {
            (
                (0..C::BLOCK_U8S)
                    .map(|_| rng.gen::<u8>())
                    .collect::<Vec<_>>(),
                true,
            )
        })
        .collect();
    let chip = ShaTestChip {
        air: ShaTestAir {
            sub_air: ShaAir::<C>::new(bitwise_bus, SELF_BUS_IDX),
        },
        bitwise_lookup_chip: bitwise_chip.clone(),
        records: random_records,
    };

    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn rand_sha256_test() {
    rand_sha_test::<Sha256Config>();
}

#[test]
fn rand_sha512_test() {
    rand_sha_test::<Sha512Config>();
}
