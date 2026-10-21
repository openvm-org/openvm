// Lean compiler output
// Module: Mathlib.Data.PNat.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.GroupWithZero.Divisibility public import Mathlib.Algebra.Order.Positive.Ring public import Mathlib.Algebra.Order.Ring.Nat public import Mathlib.Algebra.Order.Sub.Basic public import Mathlib.Data.PNat.Equiv
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
lean_object* l_Nat_add___boxed(lean_object*, lean_object*);
lean_object* lean_nat_pow(lean_object*, lean_object*);
lean_object* l_Nat_mul___boxed(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_toPNat_x27(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Nat_recCompiled___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_PNat_strongInductionOn___redArg(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
extern lean_object* lp_mathlib_Equiv_pnatEquivNat;
LEAN_EXPORT lean_object* lp_mathlib_instAddPNat___aux__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddPNat___aux__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instAddPNat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Nat_add___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instAddPNat___closed__0 = (const lean_object*)&lp_mathlib_instAddPNat___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_instAddPNat = (const lean_object*)&lp_mathlib_instAddPNat___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_instMulPNat___aux__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulPNat___aux__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instMulPNat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Nat_mul___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instMulPNat___closed__0 = (const lean_object*)&lp_mathlib_instMulPNat___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_instMulPNat = (const lean_object*)&lp_mathlib_instMulPNat___closed__0_value;
static const lean_ctor_object lp_mathlib_instDistribPNat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_instMulPNat___closed__0_value),((lean_object*)&lp_mathlib_instAddPNat___closed__0_value)}};
static const lean_object* lp_mathlib_instDistribPNat___closed__0 = (const lean_object*)&lp_mathlib_instDistribPNat___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_instDistribPNat = (const lean_object*)&lp_mathlib_instDistribPNat___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_instAddLeftCancelSemigroupPNat = (const lean_object*)&lp_mathlib_instAddPNat___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_instAddRightCancelSemigroupPNat = (const lean_object*)&lp_mathlib_instAddPNat___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_instAddCommSemigroupPNat = (const lean_object*)&lp_mathlib_instAddPNat___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_instCommMonoidPNat___aux__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCommMonoidPNat___aux__4___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCommMonoidPNat___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCommMonoidPNat___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instCommMonoidPNat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instCommMonoidPNat___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instCommMonoidPNat___closed__0 = (const lean_object*)&lp_mathlib_instCommMonoidPNat___closed__0_value;
static const lean_ctor_object lp_mathlib_instCommMonoidPNat___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_mathlib_instMulPNat___closed__0_value),((lean_object*)&lp_mathlib_instCommMonoidPNat___closed__0_value)}};
static const lean_object* lp_mathlib_instCommMonoidPNat___closed__1 = (const lean_object*)&lp_mathlib_instCommMonoidPNat___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib_instCommMonoidPNat = (const lean_object*)&lp_mathlib_instCommMonoidPNat___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib_PNat_instCancelCommMonoid = (const lean_object*)&lp_mathlib_instCommMonoidPNat___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_PNat_coeAddHom___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_coeAddHom___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_PNat_coeAddHom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_PNat_coeAddHom___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PNat_coeAddHom___closed__0 = (const lean_object*)&lp_mathlib_PNat_coeAddHom___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_PNat_coeAddHom = (const lean_object*)&lp_mathlib_PNat_coeAddHom___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_pnatIsoNat;
LEAN_EXPORT lean_object* lp_mathlib_PNat_instOrderBot;
LEAN_EXPORT lean_object* lp_mathlib_PNat_caseStrongInductionOn___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_caseStrongInductionOn___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_caseStrongInductionOn___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_caseStrongInductionOn___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_caseStrongInductionOn(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_recOn___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_recOn___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_recOn___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_PNat_recOn___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_PNat_recOn___redArg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PNat_recOn___redArg___closed__0 = (const lean_object*)&lp_mathlib_PNat_recOn___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_PNat_recOn___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_recOn___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_recOn(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_recOn___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT const lean_object* lp_mathlib_PNat_coeMonoidHom = (const lean_object*)&lp_mathlib_PNat_coeAddHom___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_PNat_instSub___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_instSub___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_PNat_instSub___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_PNat_instSub___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_PNat_instSub___closed__0 = (const lean_object*)&lp_mathlib_PNat_instSub___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_PNat_instSub = (const lean_object*)&lp_mathlib_PNat_instSub___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_instAddPNat___aux__1(lean_object* v_x_1_, lean_object* v_y_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_nat_add(v_x_1_, v_y_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddPNat___aux__1___boxed(lean_object* v_x_4_, lean_object* v_y_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_mathlib_instAddPNat___aux__1(v_x_4_, v_y_5_);
lean_dec(v_y_5_);
lean_dec(v_x_4_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulPNat___aux__1(lean_object* v_x_9_, lean_object* v_y_10_){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lean_nat_mul(v_x_9_, v_y_10_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulPNat___aux__1___boxed(lean_object* v_x_12_, lean_object* v_y_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_instMulPNat___aux__1(v_x_12_, v_y_13_);
lean_dec(v_y_13_);
lean_dec(v_x_12_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCommMonoidPNat___aux__4(lean_object* v_n_24_, lean_object* v_x_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lean_nat_pow(v_x_25_, v_n_24_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCommMonoidPNat___aux__4___boxed(lean_object* v_n_27_, lean_object* v_x_28_){
_start:
{
lean_object* v_res_29_; 
v_res_29_ = lp_mathlib_instCommMonoidPNat___aux__4(v_n_27_, v_x_28_);
lean_dec(v_x_28_);
lean_dec(v_n_27_);
return v_res_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCommMonoidPNat___lam__0(lean_object* v___y_30_, lean_object* v___y_31_){
_start:
{
lean_object* v___x_32_; 
v___x_32_ = lean_nat_pow(v___y_31_, v___y_30_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCommMonoidPNat___lam__0___boxed(lean_object* v___y_33_, lean_object* v___y_34_){
_start:
{
lean_object* v_res_35_; 
v_res_35_ = lp_mathlib_instCommMonoidPNat___lam__0(v___y_33_, v___y_34_);
lean_dec(v___y_34_);
lean_dec(v___y_33_);
return v_res_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_coeAddHom___lam__0(lean_object* v___y_43_){
_start:
{
lean_inc(v___y_43_);
return v___y_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_coeAddHom___lam__0___boxed(lean_object* v___y_44_){
_start:
{
lean_object* v_res_45_; 
v_res_45_ = lp_mathlib_PNat_coeAddHom___lam__0(v___y_44_);
lean_dec(v___y_44_);
return v_res_45_;
}
}
static lean_object* _init_lp_mathlib_OrderIso_pnatIsoNat(void){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lp_mathlib_Equiv_pnatEquivNat;
return v___x_48_;
}
}
static lean_object* _init_lp_mathlib_PNat_instOrderBot(void){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lean_unsigned_to_nat(1u);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_caseStrongInductionOn___redArg___lam__0(lean_object* v___y_50_, lean_object* v_m_51_, lean_object* v_hm_52_){
_start:
{
lean_object* v___x_53_; 
v___x_53_ = lean_apply_2(v___y_50_, v_m_51_, lean_box(0));
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_caseStrongInductionOn___redArg___lam__1(lean_object* v_hz_54_, lean_object* v_hi_55_, lean_object* v_k_56_, lean_object* v___y_57_){
_start:
{
lean_object* v_zero_58_; uint8_t v_isZero_59_; lean_object* v_one_60_; lean_object* v_n_61_; uint8_t v_isZero_62_; 
v_zero_58_ = lean_unsigned_to_nat(0u);
v_isZero_59_ = lean_nat_dec_eq(v_k_56_, v_zero_58_);
v_one_60_ = lean_unsigned_to_nat(1u);
v_n_61_ = lean_nat_sub(v_k_56_, v_one_60_);
v_isZero_62_ = lean_nat_dec_eq(v_n_61_, v_zero_58_);
if (v_isZero_62_ == 1)
{
lean_dec(v_n_61_);
lean_dec(v___y_57_);
lean_dec(v_hi_55_);
lean_inc(v_hz_54_);
return v_hz_54_;
}
else
{
lean_object* v___f_63_; lean_object* v___x_64_; 
v___f_63_ = lean_alloc_closure((void*)(lp_mathlib_PNat_caseStrongInductionOn___redArg___lam__0), 3, 1);
lean_closure_set(v___f_63_, 0, v___y_57_);
v___x_64_ = lean_apply_2(v_hi_55_, v_n_61_, v___f_63_);
return v___x_64_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_caseStrongInductionOn___redArg___lam__1___boxed(lean_object* v_hz_65_, lean_object* v_hi_66_, lean_object* v_k_67_, lean_object* v___y_68_){
_start:
{
lean_object* v_res_69_; 
v_res_69_ = lp_mathlib_PNat_caseStrongInductionOn___redArg___lam__1(v_hz_65_, v_hi_66_, v_k_67_, v___y_68_);
lean_dec(v_k_67_);
lean_dec(v_hz_65_);
return v_res_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_caseStrongInductionOn___redArg(lean_object* v_a_70_, lean_object* v_hz_71_, lean_object* v_hi_72_){
_start:
{
lean_object* v___f_73_; lean_object* v___x_74_; 
v___f_73_ = lean_alloc_closure((void*)(lp_mathlib_PNat_caseStrongInductionOn___redArg___lam__1___boxed), 4, 2);
lean_closure_set(v___f_73_, 0, v_hz_71_);
lean_closure_set(v___f_73_, 1, v_hi_72_);
v___x_74_ = lp_mathlib_PNat_strongInductionOn___redArg(v_a_70_, v___f_73_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_caseStrongInductionOn(lean_object* v_p_75_, lean_object* v_a_76_, lean_object* v_hz_77_, lean_object* v_hi_78_){
_start:
{
lean_object* v___x_79_; 
v___x_79_ = lp_mathlib_PNat_caseStrongInductionOn___redArg(v_a_76_, v_hz_77_, v_hi_78_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_recOn___redArg___lam__0(lean_object* v_h_80_){
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_recOn___redArg___lam__1(lean_object* v_one_81_, lean_object* v_succ_82_, lean_object* v_n_83_, lean_object* v_IH_84_, lean_object* v_h_85_){
_start:
{
lean_object* v_zero_86_; uint8_t v_isZero_87_; 
v_zero_86_ = lean_unsigned_to_nat(0u);
v_isZero_87_ = lean_nat_dec_eq(v_n_83_, v_zero_86_);
if (v_isZero_87_ == 1)
{
lean_dec(v_IH_84_);
lean_dec(v_succ_82_);
lean_inc(v_one_81_);
return v_one_81_;
}
else
{
lean_object* v_one_88_; lean_object* v_n_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; 
v_one_88_ = lean_unsigned_to_nat(1u);
v_n_89_ = lean_nat_sub(v_n_83_, v_one_88_);
v___x_90_ = lean_nat_add(v_n_89_, v_one_88_);
lean_dec(v_n_89_);
v___x_91_ = lean_apply_1(v_IH_84_, lean_box(0));
v___x_92_ = lean_apply_2(v_succ_82_, v___x_90_, v___x_91_);
return v___x_92_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_recOn___redArg___lam__1___boxed(lean_object* v_one_93_, lean_object* v_succ_94_, lean_object* v_n_95_, lean_object* v_IH_96_, lean_object* v_h_97_){
_start:
{
lean_object* v_res_98_; 
v_res_98_ = lp_mathlib_PNat_recOn___redArg___lam__1(v_one_93_, v_succ_94_, v_n_95_, v_IH_96_, v_h_97_);
lean_dec(v_n_95_);
lean_dec(v_one_93_);
return v_res_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_recOn___redArg(lean_object* v_n_100_, lean_object* v_one_101_, lean_object* v_succ_102_){
_start:
{
lean_object* v___f_103_; lean_object* v___f_104_; lean_object* v___x_37__overap_105_; lean_object* v___x_106_; 
v___f_103_ = ((lean_object*)(lp_mathlib_PNat_recOn___redArg___closed__0));
v___f_104_ = lean_alloc_closure((void*)(lp_mathlib_PNat_recOn___redArg___lam__1___boxed), 5, 2);
lean_closure_set(v___f_104_, 0, v_one_101_);
lean_closure_set(v___f_104_, 1, v_succ_102_);
v___x_37__overap_105_ = l_Nat_recCompiled___redArg(v___f_103_, v___f_104_, v_n_100_);
v___x_106_ = lean_apply_1(v___x_37__overap_105_, lean_box(0));
return v___x_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_recOn___redArg___boxed(lean_object* v_n_107_, lean_object* v_one_108_, lean_object* v_succ_109_){
_start:
{
lean_object* v_res_110_; 
v_res_110_ = lp_mathlib_PNat_recOn___redArg(v_n_107_, v_one_108_, v_succ_109_);
lean_dec(v_n_107_);
return v_res_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_recOn(lean_object* v_n_111_, lean_object* v_p_112_, lean_object* v_one_113_, lean_object* v_succ_114_){
_start:
{
lean_object* v___x_115_; 
v___x_115_ = lp_mathlib_PNat_recOn___redArg(v_n_111_, v_one_113_, v_succ_114_);
return v___x_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_recOn___boxed(lean_object* v_n_116_, lean_object* v_p_117_, lean_object* v_one_118_, lean_object* v_succ_119_){
_start:
{
lean_object* v_res_120_; 
v_res_120_ = lp_mathlib_PNat_recOn(v_n_116_, v_p_117_, v_one_118_, v_succ_119_);
lean_dec(v_n_116_);
return v_res_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_instSub___lam__0(lean_object* v_a_122_, lean_object* v_b_123_){
_start:
{
lean_object* v___x_124_; lean_object* v___x_125_; 
v___x_124_ = lean_nat_sub(v_a_122_, v_b_123_);
v___x_125_ = lp_mathlib_Nat_toPNat_x27(v___x_124_);
lean_dec(v___x_124_);
return v___x_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_instSub___lam__0___boxed(lean_object* v_a_126_, lean_object* v_b_127_){
_start:
{
lean_object* v_res_128_; 
v_res_128_ = lp_mathlib_PNat_instSub___lam__0(v_a_126_, v_b_127_);
lean_dec(v_b_127_);
lean_dec(v_a_126_);
return v_res_128_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Divisibility(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Positive_Ring(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Nat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Sub_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_PNat_Equiv(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_PNat_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Divisibility(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Positive_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Sub_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_PNat_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_OrderIso_pnatIsoNat = _init_lp_mathlib_OrderIso_pnatIsoNat();
lean_mark_persistent(lp_mathlib_OrderIso_pnatIsoNat);
lp_mathlib_PNat_instOrderBot = _init_lp_mathlib_PNat_instOrderBot();
lean_mark_persistent(lp_mathlib_PNat_instOrderBot);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_PNat_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Divisibility(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Positive_Ring(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Sub_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_PNat_Equiv(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_PNat_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Divisibility(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Positive_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Sub_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_PNat_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_PNat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_PNat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_PNat_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
