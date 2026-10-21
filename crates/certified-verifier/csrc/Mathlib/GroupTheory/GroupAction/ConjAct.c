// Lean compiler output
// Module: Mathlib.GroupTheory.GroupAction.ConjAct
// Imports: public import Init public meta import Init public import Mathlib.Data.Fintype.Card public import Mathlib.GroupTheory.GroupAction.Defs public import Mathlib.GroupTheory.Subgroup.Centralizer
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
lean_object* lp_mathlib_Units_instDivInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_SubgroupClass_toGroup___redArg(lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_SubmonoidClass_subtype___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Units_map___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_MonoidHom_toHomUnits___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_SubmonoidClass_toMonoid___redArg(lean_object*);
lean_object* lp_mathlib_MulDistribMulAction_toMulEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OneHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__3___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__3___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__8___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__8___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__8(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__8___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__12___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__12___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__12(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__12___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__14___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__14___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__14(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__14___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__16___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__16___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__16(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__16___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instGroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instFintype___aux__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instFintype___aux__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instFintype___aux__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instFintype___aux__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instFintype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instFintype___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instFintype(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instFintype___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instInhabited(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_ConjAct_ofConjAct___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_ConjAct_ofConjAct___closed__0 = (const lean_object*)&lp_mathlib_ConjAct_ofConjAct___closed__0_value;
static const lean_ctor_object lp_mathlib_ConjAct_ofConjAct___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_ConjAct_ofConjAct___closed__0_value),((lean_object*)&lp_mathlib_ConjAct_ofConjAct___closed__0_value)}};
static const lean_object* lp_mathlib_ConjAct_ofConjAct___closed__1 = (const lean_object*)&lp_mathlib_ConjAct_ofConjAct___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_ofConjAct(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_ofConjAct___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_toConjAct___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_toConjAct___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_toConjAct(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_toConjAct___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_rec___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_rec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_rec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instSMul___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instSMul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_unitsScalar___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_unitsScalar___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_unitsScalar___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_unitsScalar(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_unitsMulDistribMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_unitsMulDistribMulAction(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instMulDistribMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instMulDistribMulAction(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_Subgroup_conjAction___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_Subgroup_conjAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_Subgroup_conjAction(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_Subgroup_conjMulDistribMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_Subgroup_conjMulDistribMulAction(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAut_conjNormal___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulAut_conjNormal(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_unitsCentralizerEquiv___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SubmonoidClass_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_unitsCentralizerEquiv___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_unitsCentralizerEquiv___redArg___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_unitsCentralizerEquiv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_unitsCentralizerEquiv___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_unitsCentralizerEquiv___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_unitsCentralizerEquiv___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_unitsCentralizerEquiv___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_unitsCentralizerEquiv___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_unitsCentralizerEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_unitsCentralizerEquiv(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_unitsCentralizerEquiv___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__1___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v_toMonoid_2_; lean_object* v_toOne_3_; 
v_toMonoid_2_ = lean_ctor_get(v_inst_1_, 0);
v_toOne_3_ = lean_ctor_get(v_toMonoid_2_, 0);
lean_inc(v_toOne_3_);
return v_toOne_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__1___redArg___boxed(lean_object* v_inst_4_){
_start:
{
lean_object* v_res_5_; 
v_res_5_ = lp_mathlib_ConjAct_instDivInvMonoid___aux__1___redArg(v_inst_4_);
lean_dec_ref(v_inst_4_);
return v_res_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__1(lean_object* v_G_6_, lean_object* v_inst_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lp_mathlib_ConjAct_instDivInvMonoid___aux__1___redArg(v_inst_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__1___boxed(lean_object* v_G_9_, lean_object* v_inst_10_){
_start:
{
lean_object* v_res_11_; 
v_res_11_ = lp_mathlib_ConjAct_instDivInvMonoid___aux__1(v_G_9_, v_inst_10_);
lean_dec_ref(v_inst_10_);
return v_res_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__3___redArg(lean_object* v_inst_12_){
_start:
{
lean_object* v_toMonoid_13_; lean_object* v_toMul_14_; 
v_toMonoid_13_ = lean_ctor_get(v_inst_12_, 0);
v_toMul_14_ = lean_ctor_get(v_toMonoid_13_, 1);
lean_inc(v_toMul_14_);
return v_toMul_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__3___redArg___boxed(lean_object* v_inst_15_){
_start:
{
lean_object* v_res_16_; 
v_res_16_ = lp_mathlib_ConjAct_instDivInvMonoid___aux__3___redArg(v_inst_15_);
lean_dec_ref(v_inst_15_);
return v_res_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__3(lean_object* v_G_17_, lean_object* v_inst_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lp_mathlib_ConjAct_instDivInvMonoid___aux__3___redArg(v_inst_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__3___boxed(lean_object* v_G_20_, lean_object* v_inst_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_mathlib_ConjAct_instDivInvMonoid___aux__3(v_G_20_, v_inst_21_);
lean_dec_ref(v_inst_21_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__8___redArg(lean_object* v_inst_23_){
_start:
{
lean_object* v_toMonoid_24_; lean_object* v_toNPow_25_; 
v_toMonoid_24_ = lean_ctor_get(v_inst_23_, 0);
v_toNPow_25_ = lean_ctor_get(v_toMonoid_24_, 2);
lean_inc(v_toNPow_25_);
return v_toNPow_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__8___redArg___boxed(lean_object* v_inst_26_){
_start:
{
lean_object* v_res_27_; 
v_res_27_ = lp_mathlib_ConjAct_instDivInvMonoid___aux__8___redArg(v_inst_26_);
lean_dec_ref(v_inst_26_);
return v_res_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__8(lean_object* v_G_28_, lean_object* v_inst_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lp_mathlib_ConjAct_instDivInvMonoid___aux__8___redArg(v_inst_29_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__8___boxed(lean_object* v_G_31_, lean_object* v_inst_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_mathlib_ConjAct_instDivInvMonoid___aux__8(v_G_31_, v_inst_32_);
lean_dec_ref(v_inst_32_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__12___redArg(lean_object* v_inst_34_){
_start:
{
lean_object* v_toInv_35_; 
v_toInv_35_ = lean_ctor_get(v_inst_34_, 1);
lean_inc(v_toInv_35_);
return v_toInv_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__12___redArg___boxed(lean_object* v_inst_36_){
_start:
{
lean_object* v_res_37_; 
v_res_37_ = lp_mathlib_ConjAct_instDivInvMonoid___aux__12___redArg(v_inst_36_);
lean_dec_ref(v_inst_36_);
return v_res_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__12(lean_object* v_G_38_, lean_object* v_inst_39_){
_start:
{
lean_object* v_toInv_40_; 
v_toInv_40_ = lean_ctor_get(v_inst_39_, 1);
lean_inc(v_toInv_40_);
return v_toInv_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__12___boxed(lean_object* v_G_41_, lean_object* v_inst_42_){
_start:
{
lean_object* v_res_43_; 
v_res_43_ = lp_mathlib_ConjAct_instDivInvMonoid___aux__12(v_G_41_, v_inst_42_);
lean_dec_ref(v_inst_42_);
return v_res_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__14___redArg(lean_object* v_inst_44_){
_start:
{
lean_object* v_toDiv_45_; 
v_toDiv_45_ = lean_ctor_get(v_inst_44_, 2);
lean_inc(v_toDiv_45_);
return v_toDiv_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__14___redArg___boxed(lean_object* v_inst_46_){
_start:
{
lean_object* v_res_47_; 
v_res_47_ = lp_mathlib_ConjAct_instDivInvMonoid___aux__14___redArg(v_inst_46_);
lean_dec_ref(v_inst_46_);
return v_res_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__14(lean_object* v_G_48_, lean_object* v_inst_49_){
_start:
{
lean_object* v_toDiv_50_; 
v_toDiv_50_ = lean_ctor_get(v_inst_49_, 2);
lean_inc(v_toDiv_50_);
return v_toDiv_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__14___boxed(lean_object* v_G_51_, lean_object* v_inst_52_){
_start:
{
lean_object* v_res_53_; 
v_res_53_ = lp_mathlib_ConjAct_instDivInvMonoid___aux__14(v_G_51_, v_inst_52_);
lean_dec_ref(v_inst_52_);
return v_res_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__16___redArg(lean_object* v_inst_54_){
_start:
{
lean_object* v_toZPow_55_; 
v_toZPow_55_ = lean_ctor_get(v_inst_54_, 3);
lean_inc(v_toZPow_55_);
return v_toZPow_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__16___redArg___boxed(lean_object* v_inst_56_){
_start:
{
lean_object* v_res_57_; 
v_res_57_ = lp_mathlib_ConjAct_instDivInvMonoid___aux__16___redArg(v_inst_56_);
lean_dec_ref(v_inst_56_);
return v_res_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__16(lean_object* v_G_58_, lean_object* v_inst_59_){
_start:
{
lean_object* v_toZPow_60_; 
v_toZPow_60_ = lean_ctor_get(v_inst_59_, 3);
lean_inc(v_toZPow_60_);
return v_toZPow_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___aux__16___boxed(lean_object* v_G_61_, lean_object* v_inst_62_){
_start:
{
lean_object* v_res_63_; 
v_res_63_ = lp_mathlib_ConjAct_instDivInvMonoid___aux__16(v_G_61_, v_inst_62_);
lean_dec_ref(v_inst_62_);
return v_res_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid___redArg(lean_object* v_inst_64_){
_start:
{
lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v_toInv_69_; lean_object* v_toDiv_70_; lean_object* v_toZPow_71_; lean_object* v___x_73_; uint8_t v_isShared_74_; uint8_t v_isSharedCheck_78_; 
v___x_65_ = lp_mathlib_ConjAct_instDivInvMonoid___aux__1___redArg(v_inst_64_);
v___x_66_ = lp_mathlib_ConjAct_instDivInvMonoid___aux__3___redArg(v_inst_64_);
v___x_67_ = lp_mathlib_ConjAct_instDivInvMonoid___aux__8___redArg(v_inst_64_);
v___x_68_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_68_, 0, v___x_65_);
lean_ctor_set(v___x_68_, 1, v___x_66_);
lean_ctor_set(v___x_68_, 2, v___x_67_);
v_toInv_69_ = lean_ctor_get(v_inst_64_, 1);
v_toDiv_70_ = lean_ctor_get(v_inst_64_, 2);
v_toZPow_71_ = lean_ctor_get(v_inst_64_, 3);
v_isSharedCheck_78_ = !lean_is_exclusive(v_inst_64_);
if (v_isSharedCheck_78_ == 0)
{
lean_object* v_unused_79_; 
v_unused_79_ = lean_ctor_get(v_inst_64_, 0);
lean_dec(v_unused_79_);
v___x_73_ = v_inst_64_;
v_isShared_74_ = v_isSharedCheck_78_;
goto v_resetjp_72_;
}
else
{
lean_inc(v_toZPow_71_);
lean_inc(v_toDiv_70_);
lean_inc(v_toInv_69_);
lean_dec(v_inst_64_);
v___x_73_ = lean_box(0);
v_isShared_74_ = v_isSharedCheck_78_;
goto v_resetjp_72_;
}
v_resetjp_72_:
{
lean_object* v___x_76_; 
if (v_isShared_74_ == 0)
{
lean_ctor_set(v___x_73_, 0, v___x_68_);
v___x_76_ = v___x_73_;
goto v_reusejp_75_;
}
else
{
lean_object* v_reuseFailAlloc_77_; 
v_reuseFailAlloc_77_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_77_, 0, v___x_68_);
lean_ctor_set(v_reuseFailAlloc_77_, 1, v_toInv_69_);
lean_ctor_set(v_reuseFailAlloc_77_, 2, v_toDiv_70_);
lean_ctor_set(v_reuseFailAlloc_77_, 3, v_toZPow_71_);
v___x_76_ = v_reuseFailAlloc_77_;
goto v_reusejp_75_;
}
v_reusejp_75_:
{
return v___x_76_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instDivInvMonoid(lean_object* v_G_80_, lean_object* v_inst_81_){
_start:
{
lean_object* v___x_82_; 
v___x_82_ = lp_mathlib_ConjAct_instDivInvMonoid___redArg(v_inst_81_);
return v___x_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instGroup___redArg(lean_object* v_inst_83_){
_start:
{
lean_object* v___x_84_; 
v___x_84_ = lp_mathlib_ConjAct_instDivInvMonoid___redArg(v_inst_83_);
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instGroup(lean_object* v_G_85_, lean_object* v_inst_86_){
_start:
{
lean_object* v___x_87_; 
v___x_87_ = lp_mathlib_ConjAct_instDivInvMonoid___redArg(v_inst_86_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instFintype___aux__1___redArg(lean_object* v_inst_88_){
_start:
{
lean_inc(v_inst_88_);
return v_inst_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instFintype___aux__1___redArg___boxed(lean_object* v_inst_89_){
_start:
{
lean_object* v_res_90_; 
v_res_90_ = lp_mathlib_ConjAct_instFintype___aux__1___redArg(v_inst_89_);
lean_dec(v_inst_89_);
return v_res_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instFintype___aux__1(lean_object* v_G_91_, lean_object* v_inst_92_){
_start:
{
lean_inc(v_inst_92_);
return v_inst_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instFintype___aux__1___boxed(lean_object* v_G_93_, lean_object* v_inst_94_){
_start:
{
lean_object* v_res_95_; 
v_res_95_ = lp_mathlib_ConjAct_instFintype___aux__1(v_G_93_, v_inst_94_);
lean_dec(v_inst_94_);
return v_res_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instFintype___redArg(lean_object* v_inst_96_){
_start:
{
lean_inc(v_inst_96_);
return v_inst_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instFintype___redArg___boxed(lean_object* v_inst_97_){
_start:
{
lean_object* v_res_98_; 
v_res_98_ = lp_mathlib_ConjAct_instFintype___redArg(v_inst_97_);
lean_dec(v_inst_97_);
return v_res_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instFintype(lean_object* v_G_99_, lean_object* v_inst_100_){
_start:
{
lean_inc(v_inst_100_);
return v_inst_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instFintype___boxed(lean_object* v_G_101_, lean_object* v_inst_102_){
_start:
{
lean_object* v_res_103_; 
v_res_103_ = lp_mathlib_ConjAct_instFintype(v_G_101_, v_inst_102_);
lean_dec(v_inst_102_);
return v_res_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instInhabited___redArg(lean_object* v_inst_104_){
_start:
{
lean_object* v___x_105_; lean_object* v_toMonoid_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v_toOne_109_; 
v___x_105_ = lp_mathlib_ConjAct_instDivInvMonoid___redArg(v_inst_104_);
v_toMonoid_106_ = lean_ctor_get(v___x_105_, 0);
lean_inc_ref(v_toMonoid_106_);
lean_dec_ref(v___x_105_);
v___x_107_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_106_);
lean_dec_ref(v_toMonoid_106_);
v___x_108_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_107_);
v_toOne_109_ = lean_ctor_get(v___x_108_, 0);
lean_inc(v_toOne_109_);
lean_dec_ref(v___x_108_);
return v_toOne_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instInhabited(lean_object* v_G_110_, lean_object* v_inst_111_){
_start:
{
lean_object* v___x_112_; 
v___x_112_ = lp_mathlib_ConjAct_instInhabited___redArg(v_inst_111_);
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_ofConjAct(lean_object* v_G_116_, lean_object* v_inst_117_){
_start:
{
lean_object* v___x_118_; 
v___x_118_ = ((lean_object*)(lp_mathlib_ConjAct_ofConjAct___closed__1));
return v___x_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_ofConjAct___boxed(lean_object* v_G_119_, lean_object* v_inst_120_){
_start:
{
lean_object* v_res_121_; 
v_res_121_ = lp_mathlib_ConjAct_ofConjAct(v_G_119_, v_inst_120_);
lean_dec_ref(v_inst_120_);
return v_res_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_toConjAct___redArg(lean_object* v_inst_122_){
_start:
{
lean_object* v___x_123_; lean_object* v___x_124_; 
v___x_123_ = lp_mathlib_ConjAct_ofConjAct(lean_box(0), v_inst_122_);
v___x_124_ = lp_mathlib_Equiv_symm___redArg(v___x_123_);
return v___x_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_toConjAct___redArg___boxed(lean_object* v_inst_125_){
_start:
{
lean_object* v_res_126_; 
v_res_126_ = lp_mathlib_ConjAct_toConjAct___redArg(v_inst_125_);
lean_dec_ref(v_inst_125_);
return v_res_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_toConjAct(lean_object* v_G_127_, lean_object* v_inst_128_){
_start:
{
lean_object* v___x_129_; 
v___x_129_ = lp_mathlib_ConjAct_toConjAct___redArg(v_inst_128_);
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_toConjAct___boxed(lean_object* v_G_130_, lean_object* v_inst_131_){
_start:
{
lean_object* v_res_132_; 
v_res_132_ = lp_mathlib_ConjAct_toConjAct(v_G_130_, v_inst_131_);
lean_dec_ref(v_inst_131_);
return v_res_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_rec___redArg(lean_object* v_h_133_, lean_object* v_g_134_){
_start:
{
lean_object* v___x_135_; 
v___x_135_ = lean_apply_1(v_h_133_, v_g_134_);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_rec(lean_object* v_G_136_, lean_object* v_inst_137_, lean_object* v_C_138_, lean_object* v_h_139_, lean_object* v_g_140_){
_start:
{
lean_object* v___x_141_; 
v___x_141_ = lean_apply_1(v_h_139_, v_g_140_);
return v___x_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_rec___boxed(lean_object* v_G_142_, lean_object* v_inst_143_, lean_object* v_C_144_, lean_object* v_h_145_, lean_object* v_g_146_){
_start:
{
lean_object* v_res_147_; 
v_res_147_ = lp_mathlib_ConjAct_rec(v_G_142_, v_inst_143_, v_C_144_, v_h_145_, v_g_146_);
lean_dec_ref(v_inst_143_);
return v_res_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instSMul___redArg___lam__0(lean_object* v_inst_148_, lean_object* v_toMul_149_, lean_object* v_toInv_150_, lean_object* v_g_151_, lean_object* v_h_152_){
_start:
{
lean_object* v___x_153_; lean_object* v_toFun_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; 
v___x_153_ = lp_mathlib_ConjAct_ofConjAct(lean_box(0), v_inst_148_);
v_toFun_154_ = lean_ctor_get(v___x_153_, 0);
lean_inc(v_toFun_154_);
lean_dec_ref(v___x_153_);
v___x_155_ = lean_apply_1(v_toFun_154_, v_g_151_);
lean_inc(v_toMul_149_);
lean_inc(v___x_155_);
v___x_156_ = lean_apply_2(v_toMul_149_, v___x_155_, v_h_152_);
v___x_157_ = lean_apply_1(v_toInv_150_, v___x_155_);
v___x_158_ = lean_apply_2(v_toMul_149_, v___x_156_, v___x_157_);
return v___x_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instSMul___redArg___lam__0___boxed(lean_object* v_inst_159_, lean_object* v_toMul_160_, lean_object* v_toInv_161_, lean_object* v_g_162_, lean_object* v_h_163_){
_start:
{
lean_object* v_res_164_; 
v_res_164_ = lp_mathlib_ConjAct_instSMul___redArg___lam__0(v_inst_159_, v_toMul_160_, v_toInv_161_, v_g_162_, v_h_163_);
lean_dec_ref(v_inst_159_);
return v_res_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instSMul___redArg(lean_object* v_inst_165_){
_start:
{
lean_object* v_toMonoid_166_; lean_object* v_toInv_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v_toMul_170_; lean_object* v___f_171_; 
v_toMonoid_166_ = lean_ctor_get(v_inst_165_, 0);
v_toInv_167_ = lean_ctor_get(v_inst_165_, 1);
lean_inc(v_toInv_167_);
v___x_168_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_166_);
v___x_169_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_168_);
v_toMul_170_ = lean_ctor_get(v___x_169_, 1);
lean_inc(v_toMul_170_);
lean_dec_ref(v___x_169_);
v___f_171_ = lean_alloc_closure((void*)(lp_mathlib_ConjAct_instSMul___redArg___lam__0___boxed), 5, 3);
lean_closure_set(v___f_171_, 0, v_inst_165_);
lean_closure_set(v___f_171_, 1, v_toMul_170_);
lean_closure_set(v___f_171_, 2, v_toInv_167_);
return v___f_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instSMul(lean_object* v_G_172_, lean_object* v_inst_173_){
_start:
{
lean_object* v___x_174_; 
v___x_174_ = lp_mathlib_ConjAct_instSMul___redArg(v_inst_173_);
return v___x_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_unitsScalar___redArg___lam__0(lean_object* v___x_175_, lean_object* v_toMul_176_, lean_object* v_g_177_, lean_object* v_h_178_){
_start:
{
lean_object* v___x_179_; lean_object* v_toFun_180_; lean_object* v___x_181_; lean_object* v_val_182_; lean_object* v_inv_183_; lean_object* v___x_184_; lean_object* v___x_185_; 
v___x_179_ = lp_mathlib_ConjAct_ofConjAct(lean_box(0), v___x_175_);
v_toFun_180_ = lean_ctor_get(v___x_179_, 0);
lean_inc(v_toFun_180_);
lean_dec_ref(v___x_179_);
v___x_181_ = lean_apply_1(v_toFun_180_, v_g_177_);
v_val_182_ = lean_ctor_get(v___x_181_, 0);
lean_inc(v_val_182_);
v_inv_183_ = lean_ctor_get(v___x_181_, 1);
lean_inc(v_inv_183_);
lean_dec_ref(v___x_181_);
lean_inc(v_toMul_176_);
v___x_184_ = lean_apply_2(v_toMul_176_, v_val_182_, v_h_178_);
v___x_185_ = lean_apply_2(v_toMul_176_, v___x_184_, v_inv_183_);
return v___x_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_unitsScalar___redArg___lam__0___boxed(lean_object* v___x_186_, lean_object* v_toMul_187_, lean_object* v_g_188_, lean_object* v_h_189_){
_start:
{
lean_object* v_res_190_; 
v_res_190_ = lp_mathlib_ConjAct_unitsScalar___redArg___lam__0(v___x_186_, v_toMul_187_, v_g_188_, v_h_189_);
lean_dec_ref(v___x_186_);
return v_res_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_unitsScalar___redArg(lean_object* v_inst_191_){
_start:
{
lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v_toMul_194_; lean_object* v___x_195_; lean_object* v___f_196_; 
v___x_192_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_191_);
v___x_193_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_192_);
v_toMul_194_ = lean_ctor_get(v___x_193_, 1);
lean_inc(v_toMul_194_);
lean_dec_ref(v___x_193_);
v___x_195_ = lp_mathlib_Units_instDivInvMonoid___redArg(v_inst_191_);
v___f_196_ = lean_alloc_closure((void*)(lp_mathlib_ConjAct_unitsScalar___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_196_, 0, v___x_195_);
lean_closure_set(v___f_196_, 1, v_toMul_194_);
return v___f_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_unitsScalar(lean_object* v_M_197_, lean_object* v_inst_198_){
_start:
{
lean_object* v___x_199_; 
v___x_199_ = lp_mathlib_ConjAct_unitsScalar___redArg(v_inst_198_);
return v___x_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_unitsMulDistribMulAction___redArg(lean_object* v_inst_200_){
_start:
{
lean_object* v___x_201_; 
v___x_201_ = lp_mathlib_ConjAct_unitsScalar___redArg(v_inst_200_);
return v___x_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_unitsMulDistribMulAction(lean_object* v_M_202_, lean_object* v_inst_203_){
_start:
{
lean_object* v___x_204_; 
v___x_204_ = lp_mathlib_ConjAct_unitsScalar___redArg(v_inst_203_);
return v___x_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instMulDistribMulAction___redArg(lean_object* v_inst_205_){
_start:
{
lean_object* v___x_206_; 
v___x_206_ = lp_mathlib_ConjAct_instSMul___redArg(v_inst_205_);
return v___x_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_instMulDistribMulAction(lean_object* v_G_207_, lean_object* v_inst_208_){
_start:
{
lean_object* v___x_209_; 
v___x_209_ = lp_mathlib_ConjAct_instSMul___redArg(v_inst_208_);
return v___x_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_Subgroup_conjAction___redArg___lam__0(lean_object* v_inst_210_, lean_object* v_g_211_, lean_object* v_h_212_){
_start:
{
lean_object* v_toMonoid_213_; lean_object* v_toInv_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v_toMul_217_; lean_object* v___x_218_; lean_object* v_toFun_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; 
v_toMonoid_213_ = lean_ctor_get(v_inst_210_, 0);
v_toInv_214_ = lean_ctor_get(v_inst_210_, 1);
lean_inc(v_toInv_214_);
v___x_215_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_213_);
v___x_216_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_215_);
v_toMul_217_ = lean_ctor_get(v___x_216_, 1);
lean_inc_n(v_toMul_217_, 2);
lean_dec_ref(v___x_216_);
v___x_218_ = lp_mathlib_ConjAct_ofConjAct(lean_box(0), v_inst_210_);
lean_dec_ref(v_inst_210_);
v_toFun_219_ = lean_ctor_get(v___x_218_, 0);
lean_inc(v_toFun_219_);
lean_dec_ref(v___x_218_);
v___x_220_ = lean_apply_1(v_toFun_219_, v_g_211_);
lean_inc(v___x_220_);
v___x_221_ = lean_apply_2(v_toMul_217_, v___x_220_, v_h_212_);
v___x_222_ = lean_apply_1(v_toInv_214_, v___x_220_);
v___x_223_ = lean_apply_2(v_toMul_217_, v___x_221_, v___x_222_);
return v___x_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_Subgroup_conjAction___redArg(lean_object* v_inst_224_){
_start:
{
lean_object* v___f_225_; 
v___f_225_ = lean_alloc_closure((void*)(lp_mathlib_ConjAct_Subgroup_conjAction___redArg___lam__0), 3, 1);
lean_closure_set(v___f_225_, 0, v_inst_224_);
return v___f_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_Subgroup_conjAction(lean_object* v_G_226_, lean_object* v_inst_227_, lean_object* v_H_228_, lean_object* v_hH_229_){
_start:
{
lean_object* v___f_230_; 
v___f_230_ = lean_alloc_closure((void*)(lp_mathlib_ConjAct_Subgroup_conjAction___redArg___lam__0), 3, 1);
lean_closure_set(v___f_230_, 0, v_inst_227_);
return v___f_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_Subgroup_conjMulDistribMulAction___redArg(lean_object* v_inst_231_){
_start:
{
lean_object* v___f_232_; 
v___f_232_ = lean_alloc_closure((void*)(lp_mathlib_ConjAct_Subgroup_conjAction___redArg___lam__0), 3, 1);
lean_closure_set(v___f_232_, 0, v_inst_231_);
return v___f_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConjAct_Subgroup_conjMulDistribMulAction(lean_object* v_G_233_, lean_object* v_inst_234_, lean_object* v_H_235_, lean_object* v_inst_236_){
_start:
{
lean_object* v___f_237_; 
v___f_237_ = lean_alloc_closure((void*)(lp_mathlib_ConjAct_Subgroup_conjAction___redArg___lam__0), 3, 1);
lean_closure_set(v___f_237_, 0, v_inst_234_);
return v___f_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_conjNormal___redArg(lean_object* v_inst_238_){
_start:
{
lean_object* v_toMonoid_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v_toFun_243_; lean_object* v___f_244_; lean_object* v___x_245_; lean_object* v___f_246_; 
v_toMonoid_239_ = lean_ctor_get(v_inst_238_, 0);
lean_inc_ref(v_inst_238_);
v___x_240_ = lp_mathlib_ConjAct_instDivInvMonoid___redArg(v_inst_238_);
lean_inc_ref(v_toMonoid_239_);
v___x_241_ = lp_mathlib_SubmonoidClass_toMonoid___redArg(v_toMonoid_239_);
v___x_242_ = lp_mathlib_ConjAct_toConjAct___redArg(v_inst_238_);
v_toFun_243_ = lean_ctor_get(v___x_242_, 0);
lean_inc(v_toFun_243_);
lean_dec_ref(v___x_242_);
v___f_244_ = lean_alloc_closure((void*)(lp_mathlib_ConjAct_Subgroup_conjAction___redArg___lam__0), 3, 1);
lean_closure_set(v___f_244_, 0, v_inst_238_);
v___x_245_ = lean_alloc_closure((void*)(lp_mathlib_MulDistribMulAction_toMulEquiv___boxed), 6, 5);
lean_closure_set(v___x_245_, 0, lean_box(0));
lean_closure_set(v___x_245_, 1, lean_box(0));
lean_closure_set(v___x_245_, 2, v___x_240_);
lean_closure_set(v___x_245_, 3, v___x_241_);
lean_closure_set(v___x_245_, 4, v___f_244_);
v___f_246_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_246_, 0, v_toFun_243_);
lean_closure_set(v___f_246_, 1, v___x_245_);
return v___f_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulAut_conjNormal(lean_object* v_G_247_, lean_object* v_inst_248_, lean_object* v_H_249_, lean_object* v_inst_250_){
_start:
{
lean_object* v___x_251_; 
v___x_251_ = lp_mathlib_MulAut_conjNormal___redArg(v_inst_248_);
return v___x_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_unitsCentralizerEquiv___redArg___lam__0(lean_object* v___x_253_, lean_object* v_u_254_){
_start:
{
lean_object* v___x_255_; lean_object* v_toFun_256_; lean_object* v___f_257_; lean_object* v___x_258_; lean_object* v___x_259_; 
v___x_255_ = lp_mathlib_ConjAct_toConjAct___redArg(v___x_253_);
v_toFun_256_ = lean_ctor_get(v___x_255_, 0);
lean_inc(v_toFun_256_);
lean_dec_ref(v___x_255_);
v___f_257_ = ((lean_object*)(lp_mathlib_unitsCentralizerEquiv___redArg___lam__0___closed__0));
v___x_258_ = lp_mathlib_Units_map___redArg___lam__0(v___f_257_, v_u_254_);
v___x_259_ = lean_apply_1(v_toFun_256_, v___x_258_);
return v___x_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_unitsCentralizerEquiv___redArg___lam__0___boxed(lean_object* v___x_260_, lean_object* v_u_261_){
_start:
{
lean_object* v_res_262_; 
v_res_262_ = lp_mathlib_unitsCentralizerEquiv___redArg___lam__0(v___x_260_, v_u_261_);
lean_dec_ref(v___x_260_);
return v_res_262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_unitsCentralizerEquiv___redArg___lam__1(lean_object* v___x_263_, lean_object* v_u_264_){
_start:
{
lean_object* v___x_265_; lean_object* v_toFun_266_; lean_object* v___x_267_; lean_object* v_val_268_; 
v___x_265_ = lp_mathlib_ConjAct_ofConjAct(lean_box(0), v___x_263_);
v_toFun_266_ = lean_ctor_get(v___x_265_, 0);
lean_inc(v_toFun_266_);
lean_dec_ref(v___x_265_);
v___x_267_ = lean_apply_1(v_toFun_266_, v_u_264_);
v_val_268_ = lean_ctor_get(v___x_267_, 0);
lean_inc(v_val_268_);
lean_dec_ref(v___x_267_);
return v_val_268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_unitsCentralizerEquiv___redArg___lam__1___boxed(lean_object* v___x_269_, lean_object* v_u_270_){
_start:
{
lean_object* v_res_271_; 
v_res_271_ = lp_mathlib_unitsCentralizerEquiv___redArg___lam__1(v___x_269_, v_u_270_);
lean_dec_ref(v___x_269_);
return v_res_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_unitsCentralizerEquiv___redArg___lam__2(lean_object* v___x_272_, lean_object* v___f_273_, lean_object* v___y_274_){
_start:
{
lean_object* v___x_133__overap_275_; lean_object* v___x_276_; 
v___x_133__overap_275_ = lp_mathlib_MonoidHom_toHomUnits___redArg(v___x_272_, v___f_273_);
v___x_276_ = lean_apply_1(v___x_133__overap_275_, v___y_274_);
return v___x_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_unitsCentralizerEquiv___redArg___lam__2___boxed(lean_object* v___x_277_, lean_object* v___f_278_, lean_object* v___y_279_){
_start:
{
lean_object* v_res_280_; 
v_res_280_ = lp_mathlib_unitsCentralizerEquiv___redArg___lam__2(v___x_277_, v___f_278_, v___y_279_);
lean_dec_ref(v___x_277_);
return v_res_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_unitsCentralizerEquiv___redArg(lean_object* v_inst_281_){
_start:
{
lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___f_285_; lean_object* v___f_286_; lean_object* v___f_287_; lean_object* v___x_288_; lean_object* v___x_289_; 
v___x_282_ = lp_mathlib_Units_instDivInvMonoid___redArg(v_inst_281_);
lean_inc_ref_n(v___x_282_, 2);
v___x_283_ = lp_mathlib_ConjAct_instDivInvMonoid___redArg(v___x_282_);
v___x_284_ = lp_mathlib_SubgroupClass_toGroup___redArg(v___x_283_);
v___f_285_ = lean_alloc_closure((void*)(lp_mathlib_unitsCentralizerEquiv___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_285_, 0, v___x_282_);
v___f_286_ = lean_alloc_closure((void*)(lp_mathlib_unitsCentralizerEquiv___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_286_, 0, v___x_282_);
v___f_287_ = lean_alloc_closure((void*)(lp_mathlib_unitsCentralizerEquiv___redArg___lam__2___boxed), 3, 2);
lean_closure_set(v___f_287_, 0, v___x_284_);
lean_closure_set(v___f_287_, 1, v___f_286_);
v___x_288_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_288_, 0, v___f_287_);
lean_ctor_set(v___x_288_, 1, v___f_285_);
v___x_289_ = lp_mathlib_Equiv_symm___redArg(v___x_288_);
return v___x_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_unitsCentralizerEquiv(lean_object* v_M_290_, lean_object* v_inst_291_, lean_object* v_x_292_){
_start:
{
lean_object* v___x_293_; 
v___x_293_ = lp_mathlib_unitsCentralizerEquiv___redArg(v_inst_291_);
return v___x_293_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_unitsCentralizerEquiv___boxed(lean_object* v_M_294_, lean_object* v_inst_295_, lean_object* v_x_296_){
_start:
{
lean_object* v_res_297_; 
v_res_297_ = lp_mathlib_unitsCentralizerEquiv(v_M_294_, v_inst_295_, v_x_296_);
lean_dec_ref(v_x_296_);
return v_res_297_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Card(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Subgroup_Centralizer(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_ConjAct(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Subgroup_Centralizer(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_GroupTheory_GroupAction_ConjAct(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Card(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_GroupAction_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Subgroup_Centralizer(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_GroupAction_ConjAct(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_GroupAction_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Subgroup_Centralizer(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_GroupAction_ConjAct(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_GroupTheory_GroupAction_ConjAct(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_GroupTheory_GroupAction_ConjAct(builtin);
}
#ifdef __cplusplus
}
#endif
