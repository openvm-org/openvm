use openvm_ecc_guest::algebra::field::FieldExtension;
use halo2curves_axiom::bn256::{Fq, Fq2};
use lazy_static::lazy_static;
use num_bigint::BigUint;
use num_traits::Num;

lazy_static! {
    pub static ref BN254_XI: Fq2 = Fq2::from_coeffs([Fq::from_raw([9, 0, 0, 0]), Fq::one()]);

    // exp1 = (p^12 - 1) / 3
    pub static ref EXP1: BigUint = BigUint::from_str_radix(
        "4030969696062745741797811005853058291874379204406359442560681893891674450106959530046539719647151210908190211459382793062006703141168852426020468083171325367934590379984666859998399967609544754664110191464072930598755441160008826659219834762354786403012110463250131961575955268597858015384895449311534622125256548620283853223733396368939858981844663598065852816056384933498610930035891058807598891752166582271931875150099691598048016175399382213304673796601585080509443902692818733420199004555566113537482054218823936116647313678747500267068559627206777530424029211671772692598157901876223857571299238046741502089890557442500582300718504160740314926185458079985126192563953772118929726791041828902047546977272656240744693339962973939047279285351052107950250121751682659529260304162131862468322644288196213423232132152125277136333208005221619443705106431645884840489295409272576227859206166894626854018093044908314720",
        10
    ).unwrap();

    // p^k-1 = 3^n * s instead of p-1 = 3^r * s
    // where k=12 and n=3 here and
    // exp2 = (s+1)/3
    pub static ref EXP2: BigUint = BigUint::from_str_radix(
        "149295173928249842288807815031594751550902933496531831205951181255247201855813315927649619246190785589192230054051214557852100116339587126889646966043382421034614458517950624444385183985538694617189266350521219651805757080000326913304438324531658755667115202342597480058368713651772519088329461085612393412046538837788290860138273939590365147475728281409846400594680923462911515927255224400281440435265428973034513894448136725853630228718495637529802733207466114092942366766400693830377740909465411612499335341437923559875826432546203713595131838044695464089778859691547136762894737106526809539677749557286722299625576201574095640767352005953344997266128077036486155280146436004404804695964512181557316554713802082990544197776406442186936269827816744738898152657469728130713344598597476387715653492155415311971560450078713968012341037230430349766855793764662401499603533676762082513303932107208402000670112774382027",
        10
    ).unwrap();

    pub static ref U27_COEFF_0: BigUint = BigUint::from_str_radix("9483667112135124394372960210728142145589475128897916459350428495526310884707",
    10).unwrap();
    pub static ref U27_COEFF_1: BigUint = BigUint::from_str_radix("4534159768373982659291990808346042891252278737770656686799127720849666919525",
    10).unwrap();

    // rInv = 1/r mod (p^12-1)/r
    pub static ref R_INV: BigUint = BigUint::from_str_radix(
        "495819184011867778744231927046742333492451180917315223017345540833046880485481720031136878341141903241966521818658471092566752321606779256340158678675679238405722886654128392203338228575623261160538734808887996935946888297414610216445334190959815200956855428635568184508263913274453942864817234480763055154719338281461936129150171789463489422401982681230261920147923652438266934726901346095892093443898852488218812468761027620988447655860644584419583586883569984588067403598284748297179498734419889699245081714359110559679136004228878808158639412436468707589339209058958785568729925402190575720856279605832146553573981587948304340677613460685405477047119496887534881410757668344088436651291444274840864486870663164657544390995506448087189408281061890434467956047582679858345583941396130713046072603335601764495918026585155498301896749919393",
        10
    ).unwrap();

    // mInv = 1/m mod p^12-1
    pub static ref M_INV: BigUint = BigUint::from_str_radix(
        "17840267520054779749190587238017784600702972825655245554504342129614427201836516118803396948809179149954197175783449826546445899524065131269177708416982407215963288737761615699967145070776364294542559324079147363363059480104341231360692143673915822421222230661528586799190306058519400019024762424366780736540525310403098758015600523609594113357130678138304964034267260758692953579514899054295817541844330584721967571697039986079722203518034173581264955381924826388858518077894154909963532054519350571947910625755075099598588672669612434444513251495355121627496067454526862754597351094345783576387352673894873931328099247263766690688395096280633426669535619271711975898132416216382905928886703963310231865346128293216316379527200971959980873989485521004596686352787540034457467115536116148612884807380187255514888720048664139404687086409399",
        10
    ).unwrap();
}

pub struct Bn254;
