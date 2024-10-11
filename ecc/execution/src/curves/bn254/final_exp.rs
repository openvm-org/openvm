use halo2curves_axiom::{
    bn256::{Fq, Fq12, Fq2, Gt},
    ff::Field,
    pairing::MillerLoopResult,
};
use num::{BigInt, Num};

use super::Bn254;
use crate::common::{
    str_to_u64_arr, EcPoint, ExpBigInt, FeltPrint, FieldExtension, FinalExp, MultiMillerLoop,
};

#[allow(non_snake_case)]
impl FinalExp<Fq, Fq2, Fq12> for Bn254 {
    fn assert_final_exp_is_one(&self, f: Fq12, P: &[EcPoint<Fq>], Q: &[EcPoint<Fq2>]) {
        let (c, u) = self.final_exp_hint(f);
        let c_inv = c.invert().unwrap();
        c_inv.felt_print("c_inv");

        f.felt_print("f");

        // c_mul = c^{q^3 - q^2 + q}
        let c_q3 = c.frobenius_map(Some(3));
        let c_q2 = c.frobenius_map(Some(2));
        let c_q2_inv = c_q2.invert().unwrap();
        let c_q = c.frobenius_map(Some(1));
        let c_mul = c_q3 * c_q2_inv * c_q;
        let c_mul_inv = c_mul.invert().unwrap();
        c_mul.felt_print("c_mul");

        // Compute miller loop with c_inv
        let fc = self.multi_miller_loop_embedded_exp(P, Q, Some(c_inv));
        fc.felt_print("fc");

        let f_mul = fc * f;
        f_mul.felt_print("f_mul");

        // We want f_{-(6x+2)} * c^{-(q^3 - q^2 + q)} = u
        let cmp = f_mul * c_mul_inv;
        cmp.felt_print("cmp (fc * c_mul_inv)");

        // f.felt_print("f");
        // u.felt_print("u");

        // ------ native exponentiation ------
        let q = BigInt::from_str_radix(
            "21888242871839275222246405745257275088696311157297823662689037894645226208583",
            10,
        )
        .unwrap();
        let six_x_plus_2: BigInt = BigInt::from_str_radix("29793968203157093288", 10).unwrap();
        let q_pows = q.clone().pow(3) - q.clone().pow(2) + q;
        let lambda = six_x_plus_2.clone() + q_pows.clone();

        let c_to_six = c.exp(six_x_plus_2);
        let c_to_q = c.exp(q_pows);
        assert_eq!(c_mul, c_to_q);

        let c_lambda = c.exp(lambda);
        // let res = f.invert().unwrap() * c_lambda * u;
        // res.felt_print("res");
        // f.felt_print("f");
        // c_lambda.felt_print("c_lambda");

        // let res2 = f * c_lambda.invert().unwrap() * u.invert().unwrap();
        // res2.felt_print("res2");

        // ------ end ------

        let c_lambda_u = c_lambda * u;
        fc.felt_print("fc");
        c_lambda_u.felt_print("c_lambda_u");

        assert_eq!(f_mul, c_lambda_u);
        // assert_eq!(f, c_lambda);

        // assert_eq!(res, Fq12::ONE)
    }

    fn final_exp_hint(&self, f: Fq12) -> (Fq12, Fq12) {
        debug_assert_eq!(
            Gt(f).final_exponentiation(),
            Gt(Fq12::one()),
            "Trying to call final_exp_hint on {f:?} which does not final exponentiate to 1."
        );
        println!("f: {:#?}", f);
        // Residue witness inverse
        let mut c;
        // Cubic nonresidue power
        let u;

        // exp1 = (p^12 - 1) / 3
        let exp1 = BigInt::from_str_radix(
            "4030969696062745741797811005853058291874379204406359442560681893891674450106959530046539719647151210908190211459382793062006703141168852426020468083171325367934590379984666859998399967609544754664110191464072930598755441160008826659219834762354786403012110463250131961575955268597858015384895449311534622125256548620283853223733396368939858981844663598065852816056384933498610930035891058807598891752166582271931875150099691598048016175399382213304673796601585080509443902692818733420199004555566113537482054218823936116647313678747500267068559627206777530424029211671772692598157901876223857571299238046741502089890557442500582300718504160740314926185458079985126192563953772118929726791041828902047546977272656240744693339962973939047279285351052107950250121751682659529260304162131862468322644288196213423232132152125277136333208005221619443705106431645884840489295409272576227859206166894626854018093044908314720",
            10
        ).unwrap();

        // get the 27th root of unity
        let u0 = str_to_u64_arr(
            "9483667112135124394372960210728142145589475128897916459350428495526310884707",
            10,
        );
        let u1 = str_to_u64_arr(
            "4534159768373982659291990808346042891252278737770656686799127720849666919525",
            10,
        );
        let u_coeffs = Fq2::from_coeffs(&[
            Fq::from_raw([u0[0], u0[1], u0[2], u0[3]]),
            Fq::from_raw([u1[0], u1[1], u1[2], u1[3]]),
        ]);
        let unity_root_27 = Fq12::from_coeffs(&[
            Fq2::ZERO,
            Fq2::ZERO,
            u_coeffs,
            Fq2::ZERO,
            Fq2::ZERO,
            Fq2::ZERO,
        ]);
        unity_root_27.felt_print("27th root of unity");
        debug_assert_eq!(unity_root_27.pow([27]), Fq12::one());

        if f.exp(exp1.clone()) == Fq12::ONE {
            println!("f is cubic residue");
            c = f;
            u = Fq12::ONE;
        } else {
            let f_mul_unity_root_27 = f * unity_root_27;
            if f_mul_unity_root_27.exp(exp1.clone()) == Fq12::ONE {
                println!("f * omega is cubic residue");
                c = f_mul_unity_root_27;
                u = unity_root_27;
            } else {
                println!("f * omega^2 is cubic residue");
                c = f_mul_unity_root_27 * unity_root_27;
                u = unity_root_27.square();
            }
        }

        c.felt_print("c");
        u.felt_print("u");

        // 1. Compute r-th root and exponentiate to rInv where
        //   rInv = 1/r mod (p^12-1)/r
        let r_inv = BigInt::from_str_radix(
            "495819184011867778744231927046742333492451180917315223017345540833046880485481720031136878341141903241966521818658471092566752321606779256340158678675679238405722886654128392203338228575623261160538734808887996935946888297414610216445334190959815200956855428635568184508263913274453942864817234480763055154719338281461936129150171789463489422401982681230261920147923652438266934726901346095892093443898852488218812468761027620988447655860644584419583586883569984588067403598284748297179498734419889699245081714359110559679136004228878808158639412436468707589339209058958785568729925402190575720856279605832146553573981587948304340677613460685405477047119496887534881410757668344088436651291444274840864486870663164657544390995506448087189408281061890434467956047582679858345583941396130713046072603335601764495918026585155498301896749919393",
            10
        ).unwrap();
        #[cfg(debug_assertions)]
        {
            let r = BigInt::from_str_radix(
                "21888242871839275222246405745257275088548364400416034343698204186575808495617",
                10,
            )
            .unwrap();
            assert_eq!(c.exp(r_inv.clone()).exp(r), c);
        }

        c = c.exp(r_inv);
        c.felt_print("c^r_inv");

        // 2. Compute m-th root where
        //   m = (6x + 2 + q^3 - q^2 +q)/3r
        // Exponentiate to mInv where
        //   mInv = 1/m mod p^12-1
        let m_inv = BigInt::from_str_radix("17840267520054779749190587238017784600702972825655245554504342129614427201836516118803396948809179149954197175783449826546445899524065131269177708416982407215963288737761615699967145070776364294542559324079147363363059480104341231360692143673915822421222230661528586799190306058519400019024762424366780736540525310403098758015600523609594113357130678138304964034267260758692953579514899054295817541844330584721967571697039986079722203518034173581264955381924826388858518077894154909963532054519350571947910625755075099598588672669612434444513251495355121627496067454526862754597351094345783576387352673894873931328099247263766690688395096280633426669535619271711975898132416216382905928886703963310231865346128293216316379527200971959980873989485521004596686352787540034457467115536116148612884807380187255514888720048664139404687086409399", 10).unwrap();
        c = c.exp(m_inv);
        c.felt_print("c^m_inv");

        // 3. Compute cube root
        // since gcd(3, (p^12-1)/r) != 1, we use a modified Tonelli-Shanks algorithm
        // see Alg.4 of https://eprint.iacr.org/2024/640.pdf
        // Typo in the paper: p^k-1 = 3^n * s instead of p-1 = 3^r * s
        // where k=12 and n=3 here and exp2 = (s+1)/3
        let exp2 = BigInt::from_str_radix("149295173928249842288807815031594751550902933496531831205951181255247201855813315927649619246190785589192230054051214557852100116339587126889646966043382421034614458517950624444385183985538694617189266350521219651805757080000326913304438324531658755667115202342597480058368713651772519088329461085612393412046538837788290860138273939590365147475728281409846400594680923462911515927255224400281440435265428973034513894448136725853630228718495637529802733207466114092942366766400693830377740909465411612499335341437923559875826432546203713595131838044695464089778859691547136762894737106526809539677749557286722299625576201574095640767352005953344997266128077036486155280146436004404804695964512181557316554713802082990544197776406442186936269827816744738898152657469728130713344598597476387715653492155415311971560450078713968012341037230430349766855793764662401499603533676762082513303932107208402000670112774382027", 10).unwrap();
        let mut x = c.exp(exp2.clone());
        x.felt_print("c^exp2");

        // 3^t is ord(x^3 / residueWitness)
        let c_inv = c.invert().unwrap();
        let mut x3 = x.square() * x * c_inv;
        let mut t = 0;
        let mut tmp = x3.square();

        // Modified Tonelli-Shanks algorithm for computing the cube root
        fn tonelli_shanks_loop(x3: &mut Fq12, tmp: &mut Fq12, t: &mut i32) {
            while *x3 != Fq12::ONE {
                *tmp = (*x3).square();
                *x3 *= *tmp;
                *t += 1;
            }
        }

        tonelli_shanks_loop(&mut x3, &mut tmp, &mut t);

        while t != 0 {
            tmp = unity_root_27.exp(exp2.clone());
            x *= tmp;

            x3 = x.square() * x * c_inv;
            t = 0;
            tonelli_shanks_loop(&mut x3, &mut tmp, &mut t);
        }

        debug_assert_eq!(c, x * x * x);
        // x is the cube root of the residue witness c
        c = x;

        c.felt_print("c output");
        u.felt_print("u output");

        (c, u)
    }
}
