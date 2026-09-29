// Lean compiler output
// Module: Mathlib.Data.Nat.Prime.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Nat.Units public import Mathlib.Algebra.GroupWithZero.Nat public import Mathlib.Algebra.Prime.Defs public import Mathlib.Data.Nat.Sqrt public import Mathlib.Order.Basic
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
uint8_t l_Nat_decidable__dvd(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lp_mathlib_Nat_decidableLoHi___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Nat_decidablePrime___lam__0(lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_decidablePrime___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Nat_decidablePrime(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_decidablePrime___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_minFacAux(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_minFacAux___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_minFac(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_minFac___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Nat_decidablePrime_x27(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_decidablePrime_x27___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Nat_instDecidablePredPrime(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instDecidablePredPrime___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Nat_instDecidablePredIrreducible(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instDecidablePredIrreducible___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Nat_instDecidableEqPrimes___aux__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instDecidableEqPrimes___aux__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Nat_instDecidableEqPrimes(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instDecidableEqPrimes___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_Primes_instRepr___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_Primes_instRepr___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Nat_Primes_instRepr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Nat_Primes_instRepr___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nat_Primes_instRepr___closed__0 = (const lean_object*)&lp_mathlib_Nat_Primes_instRepr___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Nat_Primes_instRepr = (const lean_object*)&lp_mathlib_Nat_Primes_instRepr___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat_Primes_inhabitedPrimes;
LEAN_EXPORT lean_object* lp_mathlib_Nat_Primes_coeNat___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_Primes_coeNat___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Nat_Primes_coeNat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Nat_Primes_coeNat___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nat_Primes_coeNat___closed__0 = (const lean_object*)&lp_mathlib_Nat_Primes_coeNat___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Nat_Primes_coeNat = (const lean_object*)&lp_mathlib_Nat_Primes_coeNat___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat_monoid_primePow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_monoid_primePow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_monoid_primePow(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Nat_decidablePrime___lam__0(lean_object* v_p_1_, uint8_t v___x_2_, lean_object* v_a_3_){
_start:
{
uint8_t v___x_4_; 
v___x_4_ = l_Nat_decidable__dvd(v_a_3_, v_p_1_);
if (v___x_4_ == 0)
{
return v___x_2_;
}
else
{
uint8_t v___x_5_; 
v___x_5_ = 0;
return v___x_5_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_decidablePrime___lam__0___boxed(lean_object* v_p_6_, lean_object* v___x_7_, lean_object* v_a_8_){
_start:
{
uint8_t v___x_80__boxed_9_; uint8_t v_res_10_; lean_object* v_r_11_; 
v___x_80__boxed_9_ = lean_unbox(v___x_7_);
v_res_10_ = lp_mathlib_Nat_decidablePrime___lam__0(v_p_6_, v___x_80__boxed_9_, v_a_8_);
lean_dec(v_a_8_);
lean_dec(v_p_6_);
v_r_11_ = lean_box(v_res_10_);
return v_r_11_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Nat_decidablePrime(lean_object* v_p_12_){
_start:
{
lean_object* v___x_13_; uint8_t v___x_14_; 
v___x_13_ = lean_unsigned_to_nat(2u);
v___x_14_ = lean_nat_dec_le(v___x_13_, v_p_12_);
if (v___x_14_ == 0)
{
lean_dec(v_p_12_);
return v___x_14_;
}
else
{
lean_object* v___x_15_; lean_object* v___f_16_; uint8_t v___x_17_; 
v___x_15_ = lean_box(v___x_14_);
lean_inc(v_p_12_);
v___f_16_ = lean_alloc_closure((void*)(lp_mathlib_Nat_decidablePrime___lam__0___boxed), 3, 2);
lean_closure_set(v___f_16_, 0, v_p_12_);
lean_closure_set(v___f_16_, 1, v___x_15_);
v___x_17_ = lp_mathlib_Nat_decidableLoHi___redArg(v___x_13_, v_p_12_, v___f_16_);
lean_dec(v_p_12_);
return v___x_17_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_decidablePrime___boxed(lean_object* v_p_18_){
_start:
{
uint8_t v_res_19_; lean_object* v_r_20_; 
v_res_19_ = lp_mathlib_Nat_decidablePrime(v_p_18_);
v_r_20_ = lean_box(v_res_19_);
return v_r_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_minFacAux(lean_object* v_n_21_, lean_object* v_x_22_){
_start:
{
lean_object* v___x_23_; uint8_t v___x_24_; 
v___x_23_ = lean_nat_mul(v_x_22_, v_x_22_);
v___x_24_ = lean_nat_dec_lt(v_n_21_, v___x_23_);
lean_dec(v___x_23_);
if (v___x_24_ == 0)
{
uint8_t v___x_25_; 
v___x_25_ = l_Nat_decidable__dvd(v_x_22_, v_n_21_);
if (v___x_25_ == 0)
{
lean_object* v___x_26_; lean_object* v___x_27_; 
v___x_26_ = lean_unsigned_to_nat(2u);
v___x_27_ = lean_nat_add(v_x_22_, v___x_26_);
lean_dec(v_x_22_);
v_x_22_ = v___x_27_;
goto _start;
}
else
{
return v_x_22_;
}
}
else
{
lean_dec(v_x_22_);
lean_inc(v_n_21_);
return v_n_21_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_minFacAux___boxed(lean_object* v_n_29_, lean_object* v_x_30_){
_start:
{
lean_object* v_res_31_; 
v_res_31_ = lp_mathlib_Nat_minFacAux(v_n_29_, v_x_30_);
lean_dec(v_n_29_);
return v_res_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_minFac(lean_object* v_n_32_){
_start:
{
lean_object* v___x_33_; uint8_t v___x_34_; 
v___x_33_ = lean_unsigned_to_nat(2u);
v___x_34_ = l_Nat_decidable__dvd(v___x_33_, v_n_32_);
if (v___x_34_ == 0)
{
lean_object* v___x_35_; lean_object* v___x_36_; 
v___x_35_ = lean_unsigned_to_nat(3u);
v___x_36_ = lp_mathlib_Nat_minFacAux(v_n_32_, v___x_35_);
return v___x_36_;
}
else
{
return v___x_33_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_minFac___boxed(lean_object* v_n_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_mathlib_Nat_minFac(v_n_37_);
lean_dec(v_n_37_);
return v_res_38_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Nat_decidablePrime_x27(lean_object* v_p_39_){
_start:
{
lean_object* v___x_40_; uint8_t v___x_41_; 
v___x_40_ = lean_unsigned_to_nat(2u);
v___x_41_ = lean_nat_dec_le(v___x_40_, v_p_39_);
if (v___x_41_ == 0)
{
return v___x_41_;
}
else
{
lean_object* v___x_42_; uint8_t v___x_43_; 
v___x_42_ = lp_mathlib_Nat_minFac(v_p_39_);
v___x_43_ = lean_nat_dec_eq(v___x_42_, v_p_39_);
lean_dec(v___x_42_);
return v___x_43_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_decidablePrime_x27___boxed(lean_object* v_p_44_){
_start:
{
uint8_t v_res_45_; lean_object* v_r_46_; 
v_res_45_ = lp_mathlib_Nat_decidablePrime_x27(v_p_44_);
lean_dec(v_p_44_);
v_r_46_ = lean_box(v_res_45_);
return v_r_46_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Nat_instDecidablePredPrime(lean_object* v_n_47_){
_start:
{
uint8_t v___x_48_; 
v___x_48_ = lp_mathlib_Nat_decidablePrime_x27(v_n_47_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instDecidablePredPrime___boxed(lean_object* v_n_49_){
_start:
{
uint8_t v_res_50_; lean_object* v_r_51_; 
v_res_50_ = lp_mathlib_Nat_instDecidablePredPrime(v_n_49_);
lean_dec(v_n_49_);
v_r_51_ = lean_box(v_res_50_);
return v_r_51_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Nat_instDecidablePredIrreducible(lean_object* v_p_52_){
_start:
{
uint8_t v___x_53_; 
v___x_53_ = lp_mathlib_Nat_decidablePrime_x27(v_p_52_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instDecidablePredIrreducible___boxed(lean_object* v_p_54_){
_start:
{
uint8_t v_res_55_; lean_object* v_r_56_; 
v_res_55_ = lp_mathlib_Nat_instDecidablePredIrreducible(v_p_54_);
lean_dec(v_p_54_);
v_r_56_ = lean_box(v_res_55_);
return v_r_56_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Nat_instDecidableEqPrimes___aux__1(lean_object* v_a_57_, lean_object* v_b_58_){
_start:
{
uint8_t v___x_59_; 
v___x_59_ = lean_nat_dec_eq(v_a_57_, v_b_58_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instDecidableEqPrimes___aux__1___boxed(lean_object* v_a_60_, lean_object* v_b_61_){
_start:
{
uint8_t v_res_62_; lean_object* v_r_63_; 
v_res_62_ = lp_mathlib_Nat_instDecidableEqPrimes___aux__1(v_a_60_, v_b_61_);
lean_dec(v_b_61_);
lean_dec(v_a_60_);
v_r_63_ = lean_box(v_res_62_);
return v_r_63_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Nat_instDecidableEqPrimes(lean_object* v_a_64_, lean_object* v_b_65_){
_start:
{
uint8_t v___x_66_; 
v___x_66_ = lean_nat_dec_eq(v_a_64_, v_b_65_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instDecidableEqPrimes___boxed(lean_object* v_a_67_, lean_object* v_b_68_){
_start:
{
uint8_t v_res_69_; lean_object* v_r_70_; 
v_res_69_ = lp_mathlib_Nat_instDecidableEqPrimes(v_a_67_, v_b_68_);
lean_dec(v_b_68_);
lean_dec(v_a_67_);
v_r_70_ = lean_box(v_res_69_);
return v_r_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_Primes_instRepr___lam__0(lean_object* v_p_71_, lean_object* v_x_72_){
_start:
{
lean_object* v___x_73_; lean_object* v___x_74_; 
v___x_73_ = l_Nat_reprFast(v_p_71_);
v___x_74_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_74_, 0, v___x_73_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_Primes_instRepr___lam__0___boxed(lean_object* v_p_75_, lean_object* v_x_76_){
_start:
{
lean_object* v_res_77_; 
v_res_77_ = lp_mathlib_Nat_Primes_instRepr___lam__0(v_p_75_, v_x_76_);
lean_dec(v_x_76_);
return v_res_77_;
}
}
static lean_object* _init_lp_mathlib_Nat_Primes_inhabitedPrimes(void){
_start:
{
lean_object* v___x_80_; 
v___x_80_ = lean_unsigned_to_nat(2u);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_Primes_coeNat___lam__0(lean_object* v_self_81_){
_start:
{
lean_inc(v_self_81_);
return v_self_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_Primes_coeNat___lam__0___boxed(lean_object* v_self_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_mathlib_Nat_Primes_coeNat___lam__0(v_self_82_);
lean_dec(v_self_82_);
return v_res_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_monoid_primePow___redArg___lam__0(lean_object* v_toNPow_86_, lean_object* v_x_87_, lean_object* v_p_88_){
_start:
{
lean_object* v___x_89_; 
v___x_89_ = lean_apply_2(v_toNPow_86_, v_p_88_, v_x_87_);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_monoid_primePow___redArg(lean_object* v_inst_90_){
_start:
{
lean_object* v_toNPow_91_; lean_object* v___f_92_; 
v_toNPow_91_ = lean_ctor_get(v_inst_90_, 2);
lean_inc(v_toNPow_91_);
lean_dec_ref(v_inst_90_);
v___f_92_ = lean_alloc_closure((void*)(lp_mathlib_Nat_monoid_primePow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_92_, 0, v_toNPow_91_);
return v___f_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_monoid_primePow(lean_object* v_00_u03b1_93_, lean_object* v_inst_94_){
_start:
{
lean_object* v___x_95_; 
v___x_95_ = lp_mathlib_Nat_monoid_primePow___redArg(v_inst_94_);
return v___x_95_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Nat_Units(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Prime_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Sqrt(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Prime_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Nat_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Prime_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Sqrt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Nat_Primes_inhabitedPrimes = _init_lp_mathlib_Nat_Primes_inhabitedPrimes();
lean_mark_persistent(lp_mathlib_Nat_Primes_inhabitedPrimes);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Nat_Prime_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Nat_Units(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Prime_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Sqrt(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Nat_Prime_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Nat_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Prime_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Sqrt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Prime_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Nat_Prime_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Nat_Prime_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
