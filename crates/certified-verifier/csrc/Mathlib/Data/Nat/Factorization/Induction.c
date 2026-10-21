// Lean compiler output
// Module: Mathlib.Data.Nat.Factorization.Induction
// Imports: public import Init public meta import Init public import Mathlib.Data.Nat.Factorization.Defs
#include <lean/lean.h>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-label"
#elif defined(__GNUC__) && !defined(__CLANG__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#ifdef __cplusplus
extern "C" {
#endif
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_factorization(lean_object*);
lean_object* lp_mathlib_Nat_minFac(lean_object*);
lean_object* lean_nat_pow(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
lean_object* lp_batteries_Nat_strongRec___redArg(lean_object*, lean_object*);
lean_object* l_Nat_recCompiled___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnPrimePow___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnPrimePow___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnPrimePow___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnPrimePow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnPosPrimePosCoprime___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnPosPrimePosCoprime___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnPosPrimePosCoprime(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnPrimeCoprime___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnPrimeCoprime___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnPrimeCoprime(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnMul___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnMul___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnMul___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnMul___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnMul___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnPrimePow___redArg___lam__0(lean_object* v_zero_1_, lean_object* v_one_2_, lean_object* v_prime__pow__mul_3_, lean_object* v_n_4_, lean_object* v___y_5_){
_start:
{
lean_object* v_zero_6_; uint8_t v_isZero_7_; 
v_zero_6_ = lean_unsigned_to_nat(0u);
v_isZero_7_ = lean_nat_dec_eq(v_n_4_, v_zero_6_);
if (v_isZero_7_ == 1)
{
lean_dec(v___y_5_);
lean_dec(v_prime__pow__mul_3_);
lean_inc(v_zero_1_);
return v_zero_1_;
}
else
{
lean_object* v_one_8_; lean_object* v_n_9_; uint8_t v_isZero_10_; 
v_one_8_ = lean_unsigned_to_nat(1u);
v_n_9_ = lean_nat_sub(v_n_4_, v_one_8_);
v_isZero_10_ = lean_nat_dec_eq(v_n_9_, v_zero_6_);
if (v_isZero_10_ == 1)
{
lean_dec(v_n_9_);
lean_dec(v___y_5_);
lean_dec(v_prime__pow__mul_3_);
lean_inc(v_one_2_);
return v_one_2_;
}
else
{
lean_object* v_n_11_; lean_object* v___x_12_; lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v_toFun_15_; lean_object* v___x_16_; lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v___x_21_; 
v_n_11_ = lean_nat_sub(v_n_9_, v_one_8_);
lean_dec(v_n_9_);
v___x_12_ = lean_unsigned_to_nat(2u);
v___x_13_ = lean_nat_add(v_n_11_, v___x_12_);
lean_dec(v_n_11_);
v___x_14_ = lp_mathlib_Nat_factorization(v___x_13_);
v_toFun_15_ = lean_ctor_get(v___x_14_, 1);
lean_inc(v_toFun_15_);
lean_dec_ref(v___x_14_);
v___x_16_ = lp_mathlib_Nat_minFac(v___x_13_);
lean_inc(v___x_16_);
v___x_17_ = lean_apply_1(v_toFun_15_, v___x_16_);
v___x_18_ = lean_nat_pow(v___x_16_, v___x_17_);
v___x_19_ = lean_nat_div(v___x_13_, v___x_18_);
lean_dec(v___x_18_);
lean_dec(v___x_13_);
lean_inc(v___x_19_);
v___x_20_ = lean_apply_2(v___y_5_, v___x_19_, lean_box(0));
v___x_21_ = lean_apply_7(v_prime__pow__mul_3_, v___x_19_, v___x_16_, v___x_17_, lean_box(0), lean_box(0), lean_box(0), v___x_20_);
return v___x_21_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnPrimePow___redArg___lam__0___boxed(lean_object* v_zero_22_, lean_object* v_one_23_, lean_object* v_prime__pow__mul_24_, lean_object* v_n_25_, lean_object* v___y_26_){
_start:
{
lean_object* v_res_27_; 
v_res_27_ = lp_mathlib_Nat_recOnPrimePow___redArg___lam__0(v_zero_22_, v_one_23_, v_prime__pow__mul_24_, v_n_25_, v___y_26_);
lean_dec(v_n_25_);
lean_dec(v_one_23_);
lean_dec(v_zero_22_);
return v_res_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnPrimePow___redArg(lean_object* v_zero_28_, lean_object* v_one_29_, lean_object* v_prime__pow__mul_30_, lean_object* v_t_31_){
_start:
{
lean_object* v___f_32_; lean_object* v___x_33_; 
v___f_32_ = lean_alloc_closure((void*)(lp_mathlib_Nat_recOnPrimePow___redArg___lam__0___boxed), 5, 3);
lean_closure_set(v___f_32_, 0, v_zero_28_);
lean_closure_set(v___f_32_, 1, v_one_29_);
lean_closure_set(v___f_32_, 2, v_prime__pow__mul_30_);
v___x_33_ = lp_batteries_Nat_strongRec___redArg(v___f_32_, v_t_31_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnPrimePow(lean_object* v_motive_34_, lean_object* v_zero_35_, lean_object* v_one_36_, lean_object* v_prime__pow__mul_37_, lean_object* v_t_38_){
_start:
{
lean_object* v___x_39_; 
v___x_39_ = lp_mathlib_Nat_recOnPrimePow___redArg(v_zero_35_, v_one_36_, v_prime__pow__mul_37_, v_t_38_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnPosPrimePosCoprime___redArg___lam__0(lean_object* v_prime__pow_40_, lean_object* v_coprime_41_, lean_object* v_a_42_, lean_object* v_p_43_, lean_object* v_n_44_, lean_object* v_hp_x27_45_, lean_object* v_hpa_46_, lean_object* v_hn_47_, lean_object* v_hPa_48_){
_start:
{
lean_object* v___x_49_; uint8_t v___x_50_; 
v___x_49_ = lean_unsigned_to_nat(1u);
v___x_50_ = lean_nat_dec_eq(v_a_42_, v___x_49_);
if (v___x_50_ == 0)
{
lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; 
v___x_51_ = lean_nat_pow(v_p_43_, v_n_44_);
v___x_52_ = lean_apply_4(v_prime__pow_40_, v_p_43_, v_n_44_, lean_box(0), lean_box(0));
v___x_53_ = lean_apply_7(v_coprime_41_, v___x_51_, v_a_42_, lean_box(0), lean_box(0), lean_box(0), v___x_52_, v_hPa_48_);
return v___x_53_;
}
else
{
lean_object* v___x_54_; 
lean_dec(v_hPa_48_);
lean_dec(v_a_42_);
lean_dec(v_coprime_41_);
v___x_54_ = lean_apply_4(v_prime__pow_40_, v_p_43_, v_n_44_, lean_box(0), lean_box(0));
return v___x_54_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnPosPrimePosCoprime___redArg(lean_object* v_prime__pow_55_, lean_object* v_zero_56_, lean_object* v_one_57_, lean_object* v_coprime_58_, lean_object* v_a_59_){
_start:
{
lean_object* v___f_60_; lean_object* v___x_61_; 
v___f_60_ = lean_alloc_closure((void*)(lp_mathlib_Nat_recOnPosPrimePosCoprime___redArg___lam__0), 9, 2);
lean_closure_set(v___f_60_, 0, v_prime__pow_55_);
lean_closure_set(v___f_60_, 1, v_coprime_58_);
v___x_61_ = lp_mathlib_Nat_recOnPrimePow___redArg(v_zero_56_, v_one_57_, v___f_60_, v_a_59_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnPosPrimePosCoprime(lean_object* v_motive_62_, lean_object* v_prime__pow_63_, lean_object* v_zero_64_, lean_object* v_one_65_, lean_object* v_coprime_66_, lean_object* v_a_67_){
_start:
{
lean_object* v___x_68_; 
v___x_68_ = lp_mathlib_Nat_recOnPosPrimePosCoprime___redArg(v_prime__pow_63_, v_zero_64_, v_one_65_, v_coprime_66_, v_a_67_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnPrimeCoprime___redArg___lam__0(lean_object* v_prime__pow_69_, lean_object* v_p_70_, lean_object* v_n_71_, lean_object* v_h_72_, lean_object* v_x_73_){
_start:
{
lean_object* v___x_74_; 
v___x_74_ = lean_apply_3(v_prime__pow_69_, v_p_70_, v_n_71_, lean_box(0));
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnPrimeCoprime___redArg(lean_object* v_zero_75_, lean_object* v_prime__pow_76_, lean_object* v_coprime_77_, lean_object* v_a_78_){
_start:
{
lean_object* v___f_79_; lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; 
lean_inc(v_prime__pow_76_);
v___f_79_ = lean_alloc_closure((void*)(lp_mathlib_Nat_recOnPrimeCoprime___redArg___lam__0), 5, 1);
lean_closure_set(v___f_79_, 0, v_prime__pow_76_);
v___x_80_ = lean_unsigned_to_nat(2u);
v___x_81_ = lean_unsigned_to_nat(0u);
v___x_82_ = lean_apply_3(v_prime__pow_76_, v___x_80_, v___x_81_, lean_box(0));
v___x_83_ = lp_mathlib_Nat_recOnPosPrimePosCoprime___redArg(v___f_79_, v_zero_75_, v___x_82_, v_coprime_77_, v_a_78_);
return v___x_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnPrimeCoprime(lean_object* v_motive_84_, lean_object* v_zero_85_, lean_object* v_prime__pow_86_, lean_object* v_coprime_87_, lean_object* v_a_88_){
_start:
{
lean_object* v___x_89_; 
v___x_89_ = lp_mathlib_Nat_recOnPrimeCoprime___redArg(v_zero_85_, v_prime__pow_86_, v_coprime_87_, v_a_88_);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnMul___redArg___lam__0(lean_object* v_p_90_, lean_object* v_prime_91_, lean_object* v_mul_92_, lean_object* v_x_93_, lean_object* v_ih_94_){
_start:
{
lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; 
v___x_95_ = lean_nat_pow(v_p_90_, v_x_93_);
lean_inc(v_p_90_);
v___x_96_ = lean_apply_2(v_prime_91_, v_p_90_, lean_box(0));
v___x_97_ = lean_apply_4(v_mul_92_, v___x_95_, v_p_90_, v_ih_94_, v___x_96_);
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnMul___redArg___lam__0___boxed(lean_object* v_p_98_, lean_object* v_prime_99_, lean_object* v_mul_100_, lean_object* v_x_101_, lean_object* v_ih_102_){
_start:
{
lean_object* v_res_103_; 
v_res_103_ = lp_mathlib_Nat_recOnMul___redArg___lam__0(v_p_98_, v_prime_99_, v_mul_100_, v_x_101_, v_ih_102_);
lean_dec(v_x_101_);
return v_res_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnMul___redArg___lam__1(lean_object* v_prime_104_, lean_object* v_mul_105_, lean_object* v_one_106_, lean_object* v_p_107_, lean_object* v_n_108_, lean_object* v_hp_x27_109_){
_start:
{
lean_object* v___f_110_; lean_object* v___x_111_; 
v___f_110_ = lean_alloc_closure((void*)(lp_mathlib_Nat_recOnMul___redArg___lam__0___boxed), 5, 3);
lean_closure_set(v___f_110_, 0, v_p_107_);
lean_closure_set(v___f_110_, 1, v_prime_104_);
lean_closure_set(v___f_110_, 2, v_mul_105_);
v___x_111_ = l_Nat_recCompiled___redArg(v_one_106_, v___f_110_, v_n_108_);
return v___x_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnMul___redArg___lam__1___boxed(lean_object* v_prime_112_, lean_object* v_mul_113_, lean_object* v_one_114_, lean_object* v_p_115_, lean_object* v_n_116_, lean_object* v_hp_x27_117_){
_start:
{
lean_object* v_res_118_; 
v_res_118_ = lp_mathlib_Nat_recOnMul___redArg___lam__1(v_prime_112_, v_mul_113_, v_one_114_, v_p_115_, v_n_116_, v_hp_x27_117_);
lean_dec(v_n_116_);
lean_dec(v_one_114_);
return v_res_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnMul___redArg___lam__2(lean_object* v_mul_119_, lean_object* v_a_120_, lean_object* v_b_121_, lean_object* v_x_122_, lean_object* v_x_123_, lean_object* v_x_124_, lean_object* v___y_125_, lean_object* v___y_126_){
_start:
{
lean_object* v___x_127_; 
v___x_127_ = lean_apply_4(v_mul_119_, v_a_120_, v_b_121_, v___y_125_, v___y_126_);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnMul___redArg(lean_object* v_zero_128_, lean_object* v_one_129_, lean_object* v_prime_130_, lean_object* v_mul_131_, lean_object* v_a_132_){
_start:
{
lean_object* v___f_133_; lean_object* v___f_134_; lean_object* v___x_135_; 
lean_inc(v_mul_131_);
v___f_133_ = lean_alloc_closure((void*)(lp_mathlib_Nat_recOnMul___redArg___lam__1___boxed), 6, 3);
lean_closure_set(v___f_133_, 0, v_prime_130_);
lean_closure_set(v___f_133_, 1, v_mul_131_);
lean_closure_set(v___f_133_, 2, v_one_129_);
v___f_134_ = lean_alloc_closure((void*)(lp_mathlib_Nat_recOnMul___redArg___lam__2), 8, 1);
lean_closure_set(v___f_134_, 0, v_mul_131_);
v___x_135_ = lp_mathlib_Nat_recOnPrimeCoprime___redArg(v_zero_128_, v___f_133_, v___f_134_, v_a_132_);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_recOnMul(lean_object* v_motive_136_, lean_object* v_zero_137_, lean_object* v_one_138_, lean_object* v_prime_139_, lean_object* v_mul_140_, lean_object* v_a_141_){
_start:
{
lean_object* v___x_142_; 
v___x_142_ = lp_mathlib_Nat_recOnMul___redArg(v_zero_137_, v_one_138_, v_prime_139_, v_mul_140_, v_a_141_);
return v___x_142_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Factorization_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Factorization_Induction(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Factorization_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Nat_Factorization_Induction(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Factorization_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Nat_Factorization_Induction(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Factorization_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Factorization_Induction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Nat_Factorization_Induction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Nat_Factorization_Induction(builtin);
}
#ifdef __cplusplus
}
#endif
