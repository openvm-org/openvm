use std::{borrow::BorrowMut, sync::Arc};

use ax_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, BitwiseOperationLookupChip,
};
use ax_stark_backend::{
    p3_field::AbstractField, utils::disable_debug_builder, verifier::VerificationError,
};
use ax_stark_sdk::{
    config::baby_bear_blake3::BabyBearBlake3Config, p3_baby_bear::BabyBear,
    utils::create_seeded_rng,
};
use axvm_circuit::{
    arch::{
        testing::{VmChipTestBuilder, VmChipTester},
        BITWISE_OP_LOOKUP_BUS,
    },
    intrinsics::hashes::keccak256::columns::KeccakVmCols,
};
use axvm_instructions::{instruction::Instruction, Rv32KeccakOpcode};
use hex::FromHex;
use p3_keccak_air::NUM_ROUNDS;
use rand::Rng;
use tiny_keccak::Hasher;

use super::{utils::num_keccak_f, KeccakVmChip, KECCAK_WORD_SIZE};

type F = BabyBear;
// io is vector of (input, expected_output, prank_output) where prank_output is Some if the trace
// will be replaced
#[allow(clippy::type_complexity)]
fn build_keccak256_test(
    io: Vec<(Vec<u8>, Option<[u8; 32]>, Option<[u8; 32]>)>,
) -> VmChipTester<BabyBearBlake3Config> {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<8>::new(bitwise_bus));

    let mut tester = VmChipTestBuilder::default();
    let mut chip = KeccakVmChip::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
        bitwise_chip.clone(),
        0,
    );

    let mut dst = 0;
    let src = 0;

    for (input, expected_output, prank_output) in &io {
        let [a, b, c] = [0, 4, 8]; // space apart for register limbs
        let [d, e] = [1, 2];

        tester.write(d, a, (dst as u32).to_le_bytes().map(F::from_canonical_u8));
        tester.write(d, b, (src as u32).to_le_bytes().map(F::from_canonical_u8));
        tester.write(
            d,
            c,
            (input.len() as u32).to_le_bytes().map(F::from_canonical_u8),
        );
        for (i, byte) in input.iter().enumerate() {
            tester.write_cell(e, src + i, F::from_canonical_u8(*byte));
        }

        tester.execute(
            &mut chip,
            Instruction::from_isize(
                Rv32KeccakOpcode::KECCAK256 as usize,
                a as isize,
                b as isize,
                c as isize,
                d as isize,
                e as isize,
            ),
        );
        if let Some(output) = expected_output {
            for (i, byte) in output.iter().enumerate() {
                assert_eq!(tester.read_cell(e, dst + i), F::from_canonical_u8(*byte));
            }
        }
        if let Some(output) = prank_output {
            for (i, output_byte) in output.iter().enumerate() {
                chip.records.last_mut().unwrap().digest_writes[i / KECCAK_WORD_SIZE].data
                    [i % KECCAK_WORD_SIZE] = F::from_canonical_u8(*output_byte);
            }
        }
        // shift dst to not deal with timestamps for pranking
        dst += 32;
    }
    let mut tester = tester.build().load(chip).load(bitwise_chip).finalize();

    let keccak_trace = tester.air_proof_inputs[2].raw.common_main.as_mut().unwrap();
    let mut row = 0;
    for (input, _, prank_output) in io {
        let num_blocks = num_keccak_f(input.len());
        let num_rows = NUM_ROUNDS * num_blocks;
        row += num_rows;
        if prank_output.is_none() {
            continue;
        }
        let output = prank_output.unwrap();
        let digest_row: &mut KeccakVmCols<_> = keccak_trace.row_mut(row - 1).borrow_mut();
        for i in 0..16 {
            let out_limb =
                F::from_canonical_u16(output[2 * i] as u16 + ((output[2 * i + 1] as u16) << 8));
            let x = i / 4;
            let y = 0;
            let limb = i % 4;
            if x == 0 && y == 0 {
                digest_row.inner.a_prime_prime_prime_0_0_limbs[limb] = out_limb;
            } else {
                digest_row.inner.a_prime_prime[y][x][limb] = out_limb;
            }
        }
    }

    tester
}

#[test]
fn test_keccak256_negative() {
    let mut rng = create_seeded_rng();
    let mut hasher = tiny_keccak::Keccak::v256();
    let input: Vec<_> = vec![0; 137];
    hasher.update(&input);
    let mut out = [0u8; 32];
    hasher.finalize(&mut out);
    out[0] = rng.gen();
    let tester = build_keccak256_test(vec![(input, None, Some(out))]);
    disable_debug_builder();
    assert_eq!(
        tester.simple_test().err(),
        Some(VerificationError::OodEvaluationMismatch)
    );
}

// Keccak Known Answer Test (KAT) vectors from https://keccak.team/obsolete/KeccakKAT-3.zip.
// Only selecting a small subset for now (add more later)
// KAT includes inputs at the bit level; we only include the ones that are bytes
#[test]
fn test_keccak256_positive_kat_vectors() {
    // input, output, Len in bits
    let test_vectors = vec![
        ("", "C5D2460186F7233C927E7DB2DCC703C0E500B653CA82273B7BFAD8045D85A470"), // ShortMsgKAT_256 Len = 0
        ("CC", "EEAD6DBFC7340A56CAEDC044696A168870549A6A7F6F56961E84A54BD9970B8A"), // ShortMsgKAT_256 Len = 8
        ("B55C10EAE0EC684C16D13463F29291BF26C82E2FA0422A99C71DB4AF14DD9C7F33EDA52FD73D017CC0F2DBE734D831F0D820D06D5F89DACC485739144F8CFD4799223B1AFF9031A105CB6A029BA71E6E5867D85A554991C38DF3C9EF8C1E1E9A7630BE61CAABCA69280C399C1FB7A12D12AEFC", "0347901965D3635005E75A1095695CCA050BC9ED2D440C0372A31B348514A889"), // ShortMsgKAT_256 Len = 920
        ("2EDC282FFB90B97118DD03AAA03B145F363905E3CBD2D50ECD692B37BF000185C651D3E9726C690D3773EC1E48510E42B17742B0B0377E7DE6B8F55E00A8A4DB4740CEE6DB0830529DD19617501DC1E9359AA3BCF147E0A76B3AB70C4984C13E339E6806BB35E683AF8527093670859F3D8A0FC7D493BCBA6BB12B5F65E71E705CA5D6C948D66ED3D730B26DB395B3447737C26FAD089AA0AD0E306CB28BF0ACF106F89AF3745F0EC72D534968CCA543CD2CA50C94B1456743254E358C1317C07A07BF2B0ECA438A709367FAFC89A57239028FC5FECFD53B8EF958EF10EE0608B7F5CB9923AD97058EC067700CC746C127A61EE3", "DD1D2A92B3F3F3902F064365838E1F5F3468730C343E2974E7A9ECFCD84AA6DB"), // ShortMsgKAT_256 Len = 1952,
        ("724627916C50338643E6996F07877EAFD96BDF01DA7E991D4155B9BE1295EA7D21C9391F4C4A41C75F77E5D27389253393725F1427F57914B273AB862B9E31DABCE506E558720520D33352D119F699E784F9E548FF91BC35CA147042128709820D69A8287EA3257857615EB0321270E94B84F446942765CE882B191FAEE7E1C87E0F0BD4E0CD8A927703524B559B769CA4ECE1F6DBF313FDCF67C572EC4185C1A88E86EC11B6454B371980020F19633B6B95BD280E4FBCB0161E1A82470320CEC6ECFA25AC73D09F1536F286D3F9DACAFB2CD1D0CE72D64D197F5C7520B3CCB2FD74EB72664BA93853EF41EABF52F015DD591500D018DD162815CC993595B195", "EA0E416C0F7B4F11E3F00479FDDF954F2539E5E557753BD546F69EE375A5DE29"), // LongMsgKAT_256 Len = 2048
        ("6E1CADFB2A14C5FFB1DD69919C0124ED1B9A414B2BEA1E5E422D53B022BDD13A9C88E162972EBB9852330006B13C5B2F2AFBE754AB7BACF12479D4558D19DDBB1A6289387B3AC084981DF335330D1570850B97203DBA5F20CF7FF21775367A8401B6EBE5B822ED16C39383232003ABC412B0CE0DD7C7DA064E4BB73E8C58F222A1512D5FE6D947316E02F8AA87E7AA7A3AA1C299D92E6414AE3B927DB8FF708AC86A09B24E1884743BC34067BB0412453B4A6A6509504B550F53D518E4BCC3D9C1EFDB33DA2EACCB84C9F1CAEC81057A8508F423B25DB5500E5FC86AB3B5EB10D6D0BF033A716DDE55B09FD53451BBEA644217AE1EF91FAD2B5DCC6515249C96EE7EABFD12F1EF65256BD1CFF2087DABF2F69AD1FFB9CF3BC8CA437C7F18B6095BC08D65DF99CC7F657C418D8EB109FDC91A13DC20A438941726EF24F9738B6552751A320C4EA9C8D7E8E8592A3B69D30A419C55FB6CB0850989C029AAAE66305E2C14530B39EAA86EA3BA2A7DECF4B2848B01FAA8AA91F2440B7CC4334F63061CE78AA1589BEFA38B194711697AE3AADCB15C9FBF06743315E2F97F1A8B52236ACB444069550C2345F4ED12E5B8E881CDD472E803E5DCE63AE485C2713F81BC307F25AC74D39BAF7E3BC5E7617465C2B9C309CB0AC0A570A7E46C6116B2242E1C54F456F6589E20B1C0925BF1CD5F9344E01F63B5BA9D4671ABBF920C7ED32937A074C33836F0E019DFB6B35D865312C6058DFDAFF844C8D58B75071523E79DFBAB2EA37479DF12C474584F4FF40F00F92C6BADA025CE4DF8FAF0AFB2CE75C07773907CA288167D6B011599C3DE0FFF16C1161D31DF1C1DDE217CB574ED5A33751759F8ED2B1E6979C5088B940926B9155C9D250B479948C20ACB5578DC02C97593F646CC5C558A6A0F3D8D273258887CCFF259197CB1A7380622E371FD2EB5376225EC04F9ED1D1F2F08FA2376DB5B790E73086F581064ED1C5F47E989E955D77716B50FB64B853388FBA01DAC2CEAE99642341F2DA64C56BEFC4789C051E5EB79B063F2F084DB4491C3C5AA7B4BCF7DD7A1D7CED1554FA67DCA1F9515746A237547A4A1D22ACF649FA1ED3B9BB52BDE0C6996620F8CFDB293F8BACAD02BCE428363D0BB3D391469461D212769048219220A7ED39D1F9157DFEA3B4394CA8F5F612D9AC162BF0B961BFBC157E5F863CE659EB235CF98E8444BC8C7880BDDCD0B3B389AAA89D5E05F84D0649EEBACAB4F1C75352E89F0E9D91E4ACA264493A50D2F4AED66BD13650D1F18E7199E931C78AEB763E903807499F1CD99AF81276B615BE8EC709B039584B2B57445B014F6162577F3548329FD288B0800F936FC5EA1A412E3142E609FC8E39988CA53DF4D8FB5B5FB5F42C0A01648946AC6864CFB0E92856345B08E5DF0D235261E44CFE776456B40AEF0AC1A0DFA2FE639486666C05EA196B0C1A9D346435E03965E6139B1CE10129F8A53745F80100A94AE04D996C13AC14CF2713E39DFBB19A936CF3861318BD749B1FB82F40D73D714E406CBEB3D920EA037B7DE566455CCA51980F0F53A762D5BF8A4DBB55AAC0EDDB4B1F2AED2AA3D01449D34A57FDE4329E7FF3F6BECE4456207A4225218EE9F174C2DE0FF51CEAF2A07CF84F03D1DF316331E3E725C5421356C40ED25D5ABF9D24C4570FED618CA41000455DBD759E32E2BF0B6C5E61297C20F752C3042394CE840C70943C451DD5598EB0E4953CE26E833E5AF64FC1007C04456D19F87E45636F456B7DC9D31E757622E2739573342DE75497AE181AAE7A5425756C8E2A7EEF918E5C6A968AEFE92E8B261BBFE936B19F9E69A3C90094096DAE896450E1505ED5828EE2A7F0EA3A28E6EC47C0AF711823E7689166EA07ECA00FFC493131D65F93A4E1D03E0354AFC2115CFB8D23DAE8C6F96891031B23226B8BC82F1A73DAA5BB740FC8CC36C0975BEFA0C7895A9BBC261EDB7FD384103968F7A18353D5FE56274E4515768E4353046C785267DE01E816A2873F97AAD3AB4D7234EBFD9832716F43BE8245CF0B4408BA0F0F764CE9D24947AB6ABDD9879F24FCFF10078F5894B0D64F6A8D3EA3DD92A0C38609D3C14FDC0A44064D501926BE84BF8034F1D7A8C5F382E6989BFFA2109D4FBC56D1F091E8B6FABFF04D21BB19656929D19DECB8E8291E6AE5537A169874E0FE9890DFF11FFD159AD23D749FB9E8B676E2C31313C16D1EFA06F4D7BC191280A4EE63049FCEF23042B20303AECDD412A526D7A53F760A089FBDF13F361586F0DCA76BB928EDB41931D11F679619F948A6A9E8DBA919327769006303C6EF841438A7255C806242E2E7FF4621BB0F8AFA0B4A248EAD1A1E946F3E826FBFBBF8013CE5CC814E20FEF21FA5DB19EC7FF0B06C592247B27E500EB4705E6C37D41D09E83CB0A618008CA1AAAE8A215171D817659063C2FA385CFA3C1078D5C2B28CE7312876A276773821BE145785DFF24BBB24D590678158A61EA49F2BE56FDAC8CE7F94B05D62F15ADD351E5930FD4F31B3E7401D5C0FF7FC845B165FB6ABAFD4788A8B0615FEC91092B34B710A68DA518631622BA2AAE5D19010D307E565A161E64A4319A6B261FB2F6A90533997B1AEC32EF89CF1F232696E213DAFE4DBEB1CF1D5BBD12E5FF2EBB2809184E37CD9A0E58A4E0AF099493E6D8CC98B05A2F040A7E39515038F6EE21FC25F8D459A327B83EC1A28A234237ACD52465506942646AC248EC96EBBA6E1B092475F7ADAE4D35E009FD338613C7D4C12E381847310A10E6F02C02392FC32084FBE939689BC6518BE27AF7842DEEA8043828E3DFFE3BBAC4794CA0CC78699722709F2E4B0EAE7287DEB06A27B462423EC3F0DF227ACF589043292685F2C0E73203E8588B62554FF19D6260C7FE48DF301509D33BE0D8B31D3F658C921EF7F55449FF3887D91BFB894116DF57206098E8C5835B", "3C79A3BD824542C20AF71F21D6C28DF2213A041F77DD79A328A0078123954E7B"), // LongMsgKAT_256 Len = 16664
        ("7ADC0B6693E61C269F278E6944A5A2D8300981E40022F839AC644387BFAC9086650085C2CDC585FEA47B9D2E52D65A2B29A7DC370401EF5D60DD0D21F9E2B90FAE919319B14B8C5565B0423CEFB827D5F1203302A9D01523498A4DB10374", "4CC2AFF141987F4C2E683FA2DE30042BACDCD06087D7A7B014996E9CFEAA58CE"), // ShortMsgKAT_256 Len = 752
    ];

    let mut io = vec![];
    for (input, output) in test_vectors {
        let input = Vec::from_hex(input).unwrap();
        let output = Vec::from_hex(output).unwrap();
        io.push((input, Some(output.try_into().unwrap()), None));
    }

    let tester = build_keccak256_test(io);
    tester.simple_test().expect("Verification failed");
}
