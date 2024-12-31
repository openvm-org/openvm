use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    dev::MockProver,
    halo2curves::bn256::{Bn256, Fr, G1Affine},
    plonk::{
        create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Assigned, Circuit, Column,
        ConstraintSystem, Error, Fixed, Instance, ProvingKey,
    },
    poly::{
        commitment::{Params, ParamsProver},
        kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::{ProverGWC, VerifierGWC},
            strategy::AccumulatorStrategy,
        },
        Rotation, VerificationStrategy,
    },
    transcript::{TranscriptReadBuffer, TranscriptWriterBuffer},
};
use itertools::Itertools;
use openvm_algebra_circuit::{Fp2Extension, ModularExtension};
use openvm_build::{GuestOptions, TargetFilter};
use openvm_circuit::utils::air_test_with_min_segments;
use openvm_ecc_circuit::WeierstrassExtension;
use openvm_pairing_circuit::{PairingCurve, PairingExtension};
use openvm_pairing_guest::bn254::{Bn254Scalar, BN254_MODULUS, BN254_ORDER};
use openvm_sdk::{config::SdkVmConfig, Sdk, StdIn};
use openvm_snark_verifier::{loader::LOADER, KzgAccumulationScheme, PlonkVerifierContext};
use rand::{rngs::OsRng, RngCore};
use snark_verifier_sdk::snark_verifier::{
    halo2_base::halo2_proofs,
    loader::ScalarLoader,
    pcs::kzg::KzgDecidingKey,
    system::halo2::{compile, transcript::evm::EvmTranscript, Config},
};

#[derive(Clone, Copy)]
struct StandardPlonkConfig {
    a: Column<Advice>,
    b: Column<Advice>,
    c: Column<Advice>,
    q_a: Column<Fixed>,
    q_b: Column<Fixed>,
    q_c: Column<Fixed>,
    q_ab: Column<Fixed>,
    constant: Column<Fixed>,
    #[allow(dead_code)]
    instance: Column<Instance>,
}

impl StandardPlonkConfig {
    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self {
        let [a, b, c] = [(); 3].map(|_| meta.advice_column());
        let [q_a, q_b, q_c, q_ab, constant] = [(); 5].map(|_| meta.fixed_column());
        let instance = meta.instance_column();

        [a, b, c].map(|column| meta.enable_equality(column));

        meta.create_gate(
            "q_a·a + q_b·b + q_c·c + q_ab·a·b + constant + instance = 0",
            |meta| {
                let [a, b, c] = [a, b, c].map(|column| meta.query_advice(column, Rotation::cur()));
                let [q_a, q_b, q_c, q_ab, constant] = [q_a, q_b, q_c, q_ab, constant]
                    .map(|column| meta.query_fixed(column, Rotation::cur()));
                let instance = meta.query_instance(instance, Rotation::cur());
                Some(
                    q_a * a.clone()
                        + q_b * b.clone()
                        + q_c * c
                        + q_ab * a * b
                        + constant
                        + instance,
                )
            },
        );

        StandardPlonkConfig {
            a,
            b,
            c,
            q_a,
            q_b,
            q_c,
            q_ab,
            constant,
            instance,
        }
    }
}

#[derive(Clone, Default)]
struct StandardPlonk(Fr);

impl StandardPlonk {
    fn rand<R: RngCore>(mut rng: R) -> Self {
        Self(Fr::from(rng.next_u32() as u64))
    }

    fn num_instance() -> Vec<usize> {
        vec![1]
    }

    fn instances(&self) -> Vec<Vec<Fr>> {
        vec![vec![self.0]]
    }
}

impl Circuit<Fr> for StandardPlonk {
    type Config = StandardPlonkConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        meta.set_minimum_degree(4);
        StandardPlonkConfig::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || "",
            |mut region| {
                {
                    region.assign_advice(config.a, 0, Value::known(Assigned::Trivial(self.0)));
                    region.assign_fixed(config.q_a, 0, Assigned::Trivial(-Fr::one()));

                    region.assign_advice(
                        config.a,
                        1,
                        Value::known(Assigned::Trivial(-Fr::from(5u64))),
                    );
                    for (idx, column) in (1..).zip([
                        config.q_a,
                        config.q_b,
                        config.q_c,
                        config.q_ab,
                        config.constant,
                    ]) {
                        region.assign_fixed(column, 1, Assigned::Trivial(Fr::from(idx as u64)));
                    }

                    let a = region.assign_advice(
                        config.a,
                        2,
                        Value::known(Assigned::Trivial(Fr::one())),
                    );
                    a.copy_advice(&mut region, config.b, 3);
                    a.copy_advice(&mut region, config.c, 4);
                }
                Ok(())
            },
        )
    }
}

fn gen_srs(k: u32) -> ParamsKZG<Bn256> {
    ParamsKZG::<Bn256>::setup(k, OsRng)
}

fn gen_pk<C: Circuit<Fr>>(params: &ParamsKZG<Bn256>, circuit: &C) -> ProvingKey<G1Affine> {
    let vk = keygen_vk(params, circuit).unwrap();
    keygen_pk(params, vk, circuit).unwrap()
}

fn gen_proof<C: Circuit<Fr>>(
    params: &ParamsKZG<Bn256>,
    pk: &ProvingKey<G1Affine>,
    circuit: C,
    instances: Vec<Vec<Fr>>,
) -> Vec<u8> {
    MockProver::run(params.k(), &circuit, instances.clone())
        .unwrap()
        .assert_satisfied();

    let instances = instances
        .iter()
        .map(|instances| instances.as_slice())
        .collect_vec();
    let proof = {
        let mut transcript = TranscriptWriterBuffer::<_, G1Affine, _>::init(Vec::new());
        create_proof::<KZGCommitmentScheme<Bn256>, ProverGWC<_>, _, _, EvmTranscript<_, _, _, _>, _>(
            params,
            pk,
            &[circuit],
            &[instances.as_slice()],
            OsRng,
            &mut transcript,
        )
        .unwrap();
        transcript.finalize()
    };

    let accept = {
        let mut transcript = TranscriptReadBuffer::<_, G1Affine, _>::init(proof.as_slice());
        VerificationStrategy::<_, VerifierGWC<_>>::finalize(
            verify_proof::<_, VerifierGWC<_>, _, EvmTranscript<_, _, _, _>, _>(
                params.verifier_params(),
                pk.get_vk(),
                AccumulatorStrategy::new(params.verifier_params()),
                &[instances.as_slice()],
                &mut transcript,
            )
            .unwrap(),
        )
    };
    assert!(accept);

    proof
}

#[test]
fn test_plonk_zkvm() -> eyre::Result<()> {
    let params = gen_srs(8);

    let circuit = StandardPlonk::rand(OsRng);
    let pk = gen_pk(&params, &circuit);

    let proof = gen_proof(&params, &pk, circuit.clone(), circuit.instances());
    let instances = circuit.instances();
    let loader = &*LOADER;
    let public_values: Vec<Vec<Bn254Scalar>> = instances
        .into_iter()
        .map(|x| x.iter().map(|x| loader.load_const(x).0).collect())
        .collect::<Vec<_>>();
    let num_instance = StandardPlonk::num_instance();
    let protocol = compile(
        &params,
        pk.get_vk(),
        Config::kzg().with_num_instance(num_instance.clone()),
    );
    let protocol = protocol.loaded(loader);

    let dk: KzgDecidingKey<Bn256> = (params.get_g()[0], params.g2(), params.s_g2()).into();

    let ctx = PlonkVerifierContext {
        dk,
        protocol,
        proof,
        public_values,
        kzg_as: KzgAccumulationScheme::SHPLONK,
    };
    let mut stdin = StdIn::default();
    stdin.write(&ctx);

    let sdk = Sdk;
    let filter = TargetFilter {
        name: "verify".to_owned(),
        kind: "bin".to_owned(),
    };
    let elf = sdk.build(
        GuestOptions::default(),
        env!("CARGO_MANIFEST_DIR"),
        &Some(filter),
    )?;
    let config = SdkVmConfig::builder()
        .system(Default::default())
        .rv32i(Default::default())
        .rv32m(Default::default())
        .io(Default::default())
        .keccak(Default::default())
        .modular(ModularExtension::new(vec![
            BN254_MODULUS.clone(),
            BN254_ORDER.clone(),
        ]))
        .ecc(WeierstrassExtension::new(vec![
            PairingCurve::Bn254.curve_config()
        ]))
        .fp2(Fp2Extension::new(vec![BN254_MODULUS.clone()]))
        .pairing(PairingExtension::new(vec![PairingCurve::Bn254]))
        .build();
    let exe = sdk.transpile(elf, config.transpiler())?;
    // TODO: assert some parts of instances to be predefined consts (like program commitment)
    air_test_with_min_segments(config, exe, stdin, 1);
    Ok(())
}
