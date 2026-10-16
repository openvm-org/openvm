// Lean compiler output
// Module: Mathlib.Algebra.Group.Subgroup.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Basic public import Mathlib.Algebra.Group.Submonoid.Defs public import Mathlib.Data.Set.Inclusion public import Mathlib.Tactic.Common public import Mathlib.Tactic.FastInstance public import Mathlib.Tactic.Attr.Core
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
lean_object* lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_ZSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_PartialOrder_ofSetLike(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubmonoidClass_toMonoid___redArg(lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
lean_object* lp_mathlib_ZPow_ofPow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddSubmonoid_zero___redArg(lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_Submonoid_one___redArg(lean_object*);
lean_object* lp_mathlib_Submonoid_mul___redArg(lean_object*);
lean_object* lp_mathlib_AddSubmonoid_add___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InvMemClass_inv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InvMemClass_inv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InvMemClass_inv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InvMemClass_inv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NegMemClass_neg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NegMemClass_neg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NegMemClass_neg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_div___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_div___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_div(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_div___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_sub___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_sub___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_sub(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_sub___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_instZPow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_instZPow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_instZPow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_instZPow___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_instZSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_instZSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_instZSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_instZSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_toGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_toGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_toGroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_toAddGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_toAddGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_toAddGroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_toCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_toCommGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_toCommGroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_toAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_toAddCommGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_toAddCommGroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_subtype___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_subtype___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_SubgroupClass_subtype___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SubgroupClass_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_SubgroupClass_subtype___closed__0 = (const lean_object*)&lp_mathlib_SubgroupClass_subtype___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_subtype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_subtype___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_subtype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_subtype___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_inclusion___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_inclusion___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_SubgroupClass_inclusion___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SubgroupClass_inclusion___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_SubgroupClass_inclusion___closed__0 = (const lean_object*)&lp_mathlib_SubgroupClass_inclusion___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_inclusion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_inclusion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_inclusion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_inclusion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instSetLike(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instSetLike___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instSetLike(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instSetLike___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Subgroup_instPartialOrder___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Subgroup_instPartialOrder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instPartialOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instPartialOrder___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_AddSubgroup_instPartialOrder___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddSubgroup_instPartialOrder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instPartialOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instPartialOrder___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_ofClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_ofClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_ofClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_ofClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_ofDiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_ofDiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_ofSub(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_ofSub___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_mul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_mul___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_mul(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_mul___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_add___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_add___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_add(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_add___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_one___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_one___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_one(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_one___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_zero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_zero___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_zero(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_zero___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_inv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_inv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_inv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_inv(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_inv___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_neg___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_neg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_neg___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_neg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_neg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_div___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_div(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_sub___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_sub(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_npow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_npow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_npow(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_nsmul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_nsmul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_nsmul(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_zpow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_zpow(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_zsmul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_zsmul(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_toGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_toGroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_toAddGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_toAddGroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_toCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_toCommGroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_toAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_toAddCommGroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_subtype(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_subtype___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_subtype(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_subtype___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_inclusion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_inclusion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_inclusion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_inclusion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_normalizer(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_normalizer___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_normalizer(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_normalizer___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_setNormalizer(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_setNormalizer___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_setNormalizer(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_setNormalizer___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InvMemClass_inv___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_a_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_apply_1(v_inst_1_, v_a_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InvMemClass_inv___redArg(lean_object* v_inst_4_){
_start:
{
lean_object* v___f_5_; 
v___f_5_ = lean_alloc_closure((void*)(lp_mathlib_InvMemClass_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_5_, 0, v_inst_4_);
return v___f_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InvMemClass_inv(lean_object* v_G_6_, lean_object* v_S_7_, lean_object* v_inst_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_H_11_){
_start:
{
lean_object* v___f_12_; 
v___f_12_ = lean_alloc_closure((void*)(lp_mathlib_InvMemClass_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_12_, 0, v_inst_8_);
return v___f_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InvMemClass_inv___boxed(lean_object* v_G_13_, lean_object* v_S_14_, lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_inst_17_, lean_object* v_H_18_){
_start:
{
lean_object* v_res_19_; 
v_res_19_ = lp_mathlib_InvMemClass_inv(v_G_13_, v_S_14_, v_inst_15_, v_inst_16_, v_inst_17_, v_H_18_);
lean_dec(v_H_18_);
return v_res_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NegMemClass_neg___redArg(lean_object* v_inst_20_){
_start:
{
lean_object* v___f_21_; 
v___f_21_ = lean_alloc_closure((void*)(lp_mathlib_InvMemClass_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_21_, 0, v_inst_20_);
return v___f_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NegMemClass_neg(lean_object* v_G_22_, lean_object* v_S_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_inst_26_, lean_object* v_H_27_){
_start:
{
lean_object* v___f_28_; 
v___f_28_ = lean_alloc_closure((void*)(lp_mathlib_InvMemClass_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_28_, 0, v_inst_24_);
return v___f_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NegMemClass_neg___boxed(lean_object* v_G_29_, lean_object* v_S_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_H_34_){
_start:
{
lean_object* v_res_35_; 
v_res_35_ = lp_mathlib_NegMemClass_neg(v_G_29_, v_S_30_, v_inst_31_, v_inst_32_, v_inst_33_, v_H_34_);
lean_dec(v_H_34_);
return v_res_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_div___redArg___lam__0(lean_object* v_toDiv_36_, lean_object* v_a_37_, lean_object* v_b_38_){
_start:
{
lean_object* v___x_39_; 
v___x_39_ = lean_apply_2(v_toDiv_36_, v_a_37_, v_b_38_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_div___redArg(lean_object* v_inst_40_){
_start:
{
lean_object* v_toDiv_41_; lean_object* v___f_42_; 
v_toDiv_41_ = lean_ctor_get(v_inst_40_, 2);
lean_inc(v_toDiv_41_);
lean_dec_ref(v_inst_40_);
v___f_42_ = lean_alloc_closure((void*)(lp_mathlib_SubgroupClass_div___redArg___lam__0), 3, 1);
lean_closure_set(v___f_42_, 0, v_toDiv_41_);
return v___f_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_div(lean_object* v_G_43_, lean_object* v_S_44_, lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_inst_47_, lean_object* v_H_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lp_mathlib_SubgroupClass_div___redArg(v_inst_45_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_div___boxed(lean_object* v_G_50_, lean_object* v_S_51_, lean_object* v_inst_52_, lean_object* v_inst_53_, lean_object* v_inst_54_, lean_object* v_H_55_){
_start:
{
lean_object* v_res_56_; 
v_res_56_ = lp_mathlib_SubgroupClass_div(v_G_50_, v_S_51_, v_inst_52_, v_inst_53_, v_inst_54_, v_H_55_);
lean_dec(v_H_55_);
return v_res_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_sub___redArg___lam__0(lean_object* v_toSub_57_, lean_object* v_a_58_, lean_object* v_b_59_){
_start:
{
lean_object* v___x_60_; 
v___x_60_ = lean_apply_2(v_toSub_57_, v_a_58_, v_b_59_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_sub___redArg(lean_object* v_inst_61_){
_start:
{
lean_object* v_toSub_62_; lean_object* v___f_63_; 
v_toSub_62_ = lean_ctor_get(v_inst_61_, 2);
lean_inc(v_toSub_62_);
lean_dec_ref(v_inst_61_);
v___f_63_ = lean_alloc_closure((void*)(lp_mathlib_AddSubgroupClass_sub___redArg___lam__0), 3, 1);
lean_closure_set(v___f_63_, 0, v_toSub_62_);
return v___f_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_sub(lean_object* v_G_64_, lean_object* v_S_65_, lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_inst_68_, lean_object* v_H_69_){
_start:
{
lean_object* v___x_70_; 
v___x_70_ = lp_mathlib_AddSubgroupClass_sub___redArg(v_inst_66_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_sub___boxed(lean_object* v_G_71_, lean_object* v_S_72_, lean_object* v_inst_73_, lean_object* v_inst_74_, lean_object* v_inst_75_, lean_object* v_H_76_){
_start:
{
lean_object* v_res_77_; 
v_res_77_ = lp_mathlib_AddSubgroupClass_sub(v_G_71_, v_S_72_, v_inst_73_, v_inst_74_, v_inst_75_, v_H_76_);
lean_dec(v_H_76_);
return v_res_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_instZPow___redArg___lam__0(lean_object* v_toZPow_78_, lean_object* v_a_79_, lean_object* v_n_80_){
_start:
{
lean_object* v___x_81_; 
v___x_81_ = lean_apply_2(v_toZPow_78_, v_n_80_, v_a_79_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_instZPow___redArg(lean_object* v_inst_82_){
_start:
{
lean_object* v_toZPow_83_; lean_object* v___f_84_; 
v_toZPow_83_ = lean_ctor_get(v_inst_82_, 3);
lean_inc(v_toZPow_83_);
lean_dec_ref(v_inst_82_);
v___f_84_ = lean_alloc_closure((void*)(lp_mathlib_SubgroupClass_instZPow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_84_, 0, v_toZPow_83_);
return v___f_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_instZPow(lean_object* v_M_85_, lean_object* v_S_86_, lean_object* v_inst_87_, lean_object* v_inst_88_, lean_object* v_inst_89_, lean_object* v_H_90_){
_start:
{
lean_object* v___x_91_; 
v___x_91_ = lp_mathlib_SubgroupClass_instZPow___redArg(v_inst_87_);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_instZPow___boxed(lean_object* v_M_92_, lean_object* v_S_93_, lean_object* v_inst_94_, lean_object* v_inst_95_, lean_object* v_inst_96_, lean_object* v_H_97_){
_start:
{
lean_object* v_res_98_; 
v_res_98_ = lp_mathlib_SubgroupClass_instZPow(v_M_92_, v_S_93_, v_inst_94_, v_inst_95_, v_inst_96_, v_H_97_);
lean_dec(v_H_97_);
return v_res_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_instZSMul___redArg___lam__0(lean_object* v_toZSMul_99_, lean_object* v_n_100_, lean_object* v_a_101_){
_start:
{
lean_object* v___x_102_; 
v___x_102_ = lean_apply_2(v_toZSMul_99_, v_n_100_, v_a_101_);
return v___x_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_instZSMul___redArg(lean_object* v_inst_103_){
_start:
{
lean_object* v_toZSMul_104_; lean_object* v___f_105_; 
v_toZSMul_104_ = lean_ctor_get(v_inst_103_, 3);
lean_inc(v_toZSMul_104_);
lean_dec_ref(v_inst_103_);
v___f_105_ = lean_alloc_closure((void*)(lp_mathlib_AddSubgroupClass_instZSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_105_, 0, v_toZSMul_104_);
return v___f_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_instZSMul(lean_object* v_M_106_, lean_object* v_S_107_, lean_object* v_inst_108_, lean_object* v_inst_109_, lean_object* v_inst_110_, lean_object* v_H_111_){
_start:
{
lean_object* v___x_112_; 
v___x_112_ = lp_mathlib_AddSubgroupClass_instZSMul___redArg(v_inst_108_);
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_instZSMul___boxed(lean_object* v_M_113_, lean_object* v_S_114_, lean_object* v_inst_115_, lean_object* v_inst_116_, lean_object* v_inst_117_, lean_object* v_H_118_){
_start:
{
lean_object* v_res_119_; 
v_res_119_ = lp_mathlib_AddSubgroupClass_instZSMul(v_M_113_, v_S_114_, v_inst_115_, v_inst_116_, v_inst_117_, v_H_118_);
lean_dec(v_H_118_);
return v_res_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_toGroup___redArg(lean_object* v_inst_120_){
_start:
{
lean_object* v_toMonoid_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v_toInv_124_; lean_object* v___f_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___f_128_; lean_object* v___x_129_; 
v_toMonoid_121_ = lean_ctor_get(v_inst_120_, 0);
lean_inc_ref(v_toMonoid_121_);
v___x_122_ = lp_mathlib_SubmonoidClass_toMonoid___redArg(v_toMonoid_121_);
v___x_123_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v_inst_120_);
v_toInv_124_ = lean_ctor_get(v___x_123_, 1);
lean_inc(v_toInv_124_);
lean_dec_ref(v___x_123_);
v___f_125_ = lean_alloc_closure((void*)(lp_mathlib_InvMemClass_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_125_, 0, v_toInv_124_);
lean_inc_ref(v_inst_120_);
v___x_126_ = lp_mathlib_SubgroupClass_div___redArg(v_inst_120_);
v___x_127_ = lp_mathlib_SubgroupClass_instZPow___redArg(v_inst_120_);
v___f_128_ = lean_alloc_closure((void*)(lp_mathlib_ZPow_ofPow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_128_, 0, v___x_127_);
v___x_129_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_129_, 0, v___x_122_);
lean_ctor_set(v___x_129_, 1, v___f_125_);
lean_ctor_set(v___x_129_, 2, v___x_126_);
lean_ctor_set(v___x_129_, 3, v___f_128_);
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_toGroup(lean_object* v_G_130_, lean_object* v_inst_131_, lean_object* v_S_132_, lean_object* v_H_133_, lean_object* v_inst_134_, lean_object* v_inst_135_){
_start:
{
lean_object* v___x_136_; 
v___x_136_ = lp_mathlib_SubgroupClass_toGroup___redArg(v_inst_131_);
return v___x_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_toGroup___boxed(lean_object* v_G_137_, lean_object* v_inst_138_, lean_object* v_S_139_, lean_object* v_H_140_, lean_object* v_inst_141_, lean_object* v_inst_142_){
_start:
{
lean_object* v_res_143_; 
v_res_143_ = lp_mathlib_SubgroupClass_toGroup(v_G_137_, v_inst_138_, v_S_139_, v_H_140_, v_inst_141_, v_inst_142_);
lean_dec(v_H_140_);
return v_res_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_toAddGroup___redArg(lean_object* v_inst_144_){
_start:
{
lean_object* v_toAddMonoid_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v_toNeg_148_; lean_object* v___f_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___f_152_; lean_object* v___x_153_; 
v_toAddMonoid_145_ = lean_ctor_get(v_inst_144_, 0);
lean_inc_ref(v_toAddMonoid_145_);
v___x_146_ = lp_mathlib_AddSubmonoidClass_toAddMonoid___redArg(v_toAddMonoid_145_);
v___x_147_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_144_);
v_toNeg_148_ = lean_ctor_get(v___x_147_, 1);
lean_inc(v_toNeg_148_);
lean_dec_ref(v___x_147_);
v___f_149_ = lean_alloc_closure((void*)(lp_mathlib_InvMemClass_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_149_, 0, v_toNeg_148_);
lean_inc_ref(v_inst_144_);
v___x_150_ = lp_mathlib_AddSubgroupClass_sub___redArg(v_inst_144_);
v___x_151_ = lp_mathlib_AddSubgroupClass_instZSMul___redArg(v_inst_144_);
v___f_152_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_152_, 0, v___x_151_);
v___x_153_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_153_, 0, v___x_146_);
lean_ctor_set(v___x_153_, 1, v___f_149_);
lean_ctor_set(v___x_153_, 2, v___x_150_);
lean_ctor_set(v___x_153_, 3, v___f_152_);
return v___x_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_toAddGroup(lean_object* v_G_154_, lean_object* v_inst_155_, lean_object* v_S_156_, lean_object* v_H_157_, lean_object* v_inst_158_, lean_object* v_inst_159_){
_start:
{
lean_object* v___x_160_; 
v___x_160_ = lp_mathlib_AddSubgroupClass_toAddGroup___redArg(v_inst_155_);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_toAddGroup___boxed(lean_object* v_G_161_, lean_object* v_inst_162_, lean_object* v_S_163_, lean_object* v_H_164_, lean_object* v_inst_165_, lean_object* v_inst_166_){
_start:
{
lean_object* v_res_167_; 
v_res_167_ = lp_mathlib_AddSubgroupClass_toAddGroup(v_G_161_, v_inst_162_, v_S_163_, v_H_164_, v_inst_165_, v_inst_166_);
lean_dec(v_H_164_);
return v_res_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_toCommGroup___redArg(lean_object* v_inst_168_){
_start:
{
lean_object* v___x_169_; 
v___x_169_ = lp_mathlib_SubgroupClass_toGroup___redArg(v_inst_168_);
return v___x_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_toCommGroup(lean_object* v_S_170_, lean_object* v_H_171_, lean_object* v_G_172_, lean_object* v_inst_173_, lean_object* v_inst_174_, lean_object* v_inst_175_){
_start:
{
lean_object* v___x_176_; 
v___x_176_ = lp_mathlib_SubgroupClass_toGroup___redArg(v_inst_173_);
return v___x_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_toCommGroup___boxed(lean_object* v_S_177_, lean_object* v_H_178_, lean_object* v_G_179_, lean_object* v_inst_180_, lean_object* v_inst_181_, lean_object* v_inst_182_){
_start:
{
lean_object* v_res_183_; 
v_res_183_ = lp_mathlib_SubgroupClass_toCommGroup(v_S_177_, v_H_178_, v_G_179_, v_inst_180_, v_inst_181_, v_inst_182_);
lean_dec(v_H_178_);
return v_res_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_toAddCommGroup___redArg(lean_object* v_inst_184_){
_start:
{
lean_object* v___x_185_; 
v___x_185_ = lp_mathlib_AddSubgroupClass_toAddGroup___redArg(v_inst_184_);
return v___x_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_toAddCommGroup(lean_object* v_S_186_, lean_object* v_H_187_, lean_object* v_G_188_, lean_object* v_inst_189_, lean_object* v_inst_190_, lean_object* v_inst_191_){
_start:
{
lean_object* v___x_192_; 
v___x_192_ = lp_mathlib_AddSubgroupClass_toAddGroup___redArg(v_inst_189_);
return v___x_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_toAddCommGroup___boxed(lean_object* v_S_193_, lean_object* v_H_194_, lean_object* v_G_195_, lean_object* v_inst_196_, lean_object* v_inst_197_, lean_object* v_inst_198_){
_start:
{
lean_object* v_res_199_; 
v_res_199_ = lp_mathlib_AddSubgroupClass_toAddCommGroup(v_S_193_, v_H_194_, v_G_195_, v_inst_196_, v_inst_197_, v_inst_198_);
lean_dec(v_H_194_);
return v_res_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_subtype___lam__0(lean_object* v_self_200_){
_start:
{
lean_inc(v_self_200_);
return v_self_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_subtype___lam__0___boxed(lean_object* v_self_201_){
_start:
{
lean_object* v_res_202_; 
v_res_202_ = lp_mathlib_SubgroupClass_subtype___lam__0(v_self_201_);
lean_dec(v_self_201_);
return v_res_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_subtype(lean_object* v_G_204_, lean_object* v_inst_205_, lean_object* v_S_206_, lean_object* v_H_207_, lean_object* v_inst_208_, lean_object* v_inst_209_){
_start:
{
lean_object* v___f_210_; 
v___f_210_ = ((lean_object*)(lp_mathlib_SubgroupClass_subtype___closed__0));
return v___f_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_subtype___boxed(lean_object* v_G_211_, lean_object* v_inst_212_, lean_object* v_S_213_, lean_object* v_H_214_, lean_object* v_inst_215_, lean_object* v_inst_216_){
_start:
{
lean_object* v_res_217_; 
v_res_217_ = lp_mathlib_SubgroupClass_subtype(v_G_211_, v_inst_212_, v_S_213_, v_H_214_, v_inst_215_, v_inst_216_);
lean_dec(v_H_214_);
lean_dec_ref(v_inst_212_);
return v_res_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_subtype(lean_object* v_G_218_, lean_object* v_inst_219_, lean_object* v_S_220_, lean_object* v_H_221_, lean_object* v_inst_222_, lean_object* v_inst_223_){
_start:
{
lean_object* v___f_224_; 
v___f_224_ = ((lean_object*)(lp_mathlib_SubgroupClass_subtype___closed__0));
return v___f_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_subtype___boxed(lean_object* v_G_225_, lean_object* v_inst_226_, lean_object* v_S_227_, lean_object* v_H_228_, lean_object* v_inst_229_, lean_object* v_inst_230_){
_start:
{
lean_object* v_res_231_; 
v_res_231_ = lp_mathlib_AddSubgroupClass_subtype(v_G_225_, v_inst_226_, v_S_227_, v_H_228_, v_inst_229_, v_inst_230_);
lean_dec(v_H_228_);
lean_dec_ref(v_inst_226_);
return v_res_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_inclusion___lam__0(lean_object* v_x_232_){
_start:
{
lean_inc(v_x_232_);
return v_x_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_inclusion___lam__0___boxed(lean_object* v_x_233_){
_start:
{
lean_object* v_res_234_; 
v_res_234_ = lp_mathlib_SubgroupClass_inclusion___lam__0(v_x_233_);
lean_dec(v_x_233_);
return v_res_234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_inclusion(lean_object* v_G_236_, lean_object* v_inst_237_, lean_object* v_S_238_, lean_object* v_inst_239_, lean_object* v_inst_240_, lean_object* v_inst_241_, lean_object* v_inst_242_, lean_object* v_H_243_, lean_object* v_K_244_, lean_object* v_h_245_){
_start:
{
lean_object* v___f_246_; 
v___f_246_ = ((lean_object*)(lp_mathlib_SubgroupClass_inclusion___closed__0));
return v___f_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SubgroupClass_inclusion___boxed(lean_object* v_G_247_, lean_object* v_inst_248_, lean_object* v_S_249_, lean_object* v_inst_250_, lean_object* v_inst_251_, lean_object* v_inst_252_, lean_object* v_inst_253_, lean_object* v_H_254_, lean_object* v_K_255_, lean_object* v_h_256_){
_start:
{
lean_object* v_res_257_; 
v_res_257_ = lp_mathlib_SubgroupClass_inclusion(v_G_247_, v_inst_248_, v_S_249_, v_inst_250_, v_inst_251_, v_inst_252_, v_inst_253_, v_H_254_, v_K_255_, v_h_256_);
lean_dec(v_K_255_);
lean_dec(v_H_254_);
lean_dec_ref(v_inst_248_);
return v_res_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_inclusion(lean_object* v_G_258_, lean_object* v_inst_259_, lean_object* v_S_260_, lean_object* v_inst_261_, lean_object* v_inst_262_, lean_object* v_inst_263_, lean_object* v_inst_264_, lean_object* v_H_265_, lean_object* v_K_266_, lean_object* v_h_267_){
_start:
{
lean_object* v___f_268_; 
v___f_268_ = ((lean_object*)(lp_mathlib_SubgroupClass_inclusion___closed__0));
return v___f_268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroupClass_inclusion___boxed(lean_object* v_G_269_, lean_object* v_inst_270_, lean_object* v_S_271_, lean_object* v_inst_272_, lean_object* v_inst_273_, lean_object* v_inst_274_, lean_object* v_inst_275_, lean_object* v_H_276_, lean_object* v_K_277_, lean_object* v_h_278_){
_start:
{
lean_object* v_res_279_; 
v_res_279_ = lp_mathlib_AddSubgroupClass_inclusion(v_G_269_, v_inst_270_, v_S_271_, v_inst_272_, v_inst_273_, v_inst_274_, v_inst_275_, v_H_276_, v_K_277_, v_h_278_);
lean_dec(v_K_277_);
lean_dec(v_H_276_);
lean_dec_ref(v_inst_270_);
return v_res_279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instSetLike(lean_object* v_G_280_, lean_object* v_inst_281_){
_start:
{
lean_object* v___x_282_; 
v___x_282_ = lean_box(0);
return v___x_282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instSetLike___boxed(lean_object* v_G_283_, lean_object* v_inst_284_){
_start:
{
lean_object* v_res_285_; 
v_res_285_ = lp_mathlib_Subgroup_instSetLike(v_G_283_, v_inst_284_);
lean_dec_ref(v_inst_284_);
return v_res_285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instSetLike(lean_object* v_G_286_, lean_object* v_inst_287_){
_start:
{
lean_object* v___x_288_; 
v___x_288_ = lean_box(0);
return v___x_288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instSetLike___boxed(lean_object* v_G_289_, lean_object* v_inst_290_){
_start:
{
lean_object* v_res_291_; 
v_res_291_ = lp_mathlib_AddSubgroup_instSetLike(v_G_289_, v_inst_290_);
lean_dec_ref(v_inst_290_);
return v_res_291_;
}
}
static lean_object* _init_lp_mathlib_Subgroup_instPartialOrder___closed__0(void){
_start:
{
lean_object* v___x_292_; lean_object* v___x_293_; 
v___x_292_ = lean_box(0);
v___x_293_ = lp_mathlib_PartialOrder_ofSetLike(lean_box(0), lean_box(0), v___x_292_);
return v___x_293_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instPartialOrder(lean_object* v_G_294_, lean_object* v_inst_295_){
_start:
{
lean_object* v___x_296_; 
v___x_296_ = lean_obj_once(&lp_mathlib_Subgroup_instPartialOrder___closed__0, &lp_mathlib_Subgroup_instPartialOrder___closed__0_once, _init_lp_mathlib_Subgroup_instPartialOrder___closed__0);
return v___x_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instPartialOrder___boxed(lean_object* v_G_297_, lean_object* v_inst_298_){
_start:
{
lean_object* v_res_299_; 
v_res_299_ = lp_mathlib_Subgroup_instPartialOrder(v_G_297_, v_inst_298_);
lean_dec_ref(v_inst_298_);
return v_res_299_;
}
}
static lean_object* _init_lp_mathlib_AddSubgroup_instPartialOrder___closed__0(void){
_start:
{
lean_object* v___x_300_; lean_object* v___x_301_; 
v___x_300_ = lean_box(0);
v___x_301_ = lp_mathlib_PartialOrder_ofSetLike(lean_box(0), lean_box(0), v___x_300_);
return v___x_301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instPartialOrder(lean_object* v_G_302_, lean_object* v_inst_303_){
_start:
{
lean_object* v___x_304_; 
v___x_304_ = lean_obj_once(&lp_mathlib_AddSubgroup_instPartialOrder___closed__0, &lp_mathlib_AddSubgroup_instPartialOrder___closed__0_once, _init_lp_mathlib_AddSubgroup_instPartialOrder___closed__0);
return v___x_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instPartialOrder___boxed(lean_object* v_G_305_, lean_object* v_inst_306_){
_start:
{
lean_object* v_res_307_; 
v_res_307_ = lp_mathlib_AddSubgroup_instPartialOrder(v_G_305_, v_inst_306_);
lean_dec_ref(v_inst_306_);
return v_res_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_ofClass(lean_object* v_S_308_, lean_object* v_G_309_, lean_object* v_inst_310_, lean_object* v_inst_311_, lean_object* v_inst_312_, lean_object* v_s_313_){
_start:
{
lean_object* v___x_314_; 
v___x_314_ = lean_box(0);
return v___x_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_ofClass___boxed(lean_object* v_S_315_, lean_object* v_G_316_, lean_object* v_inst_317_, lean_object* v_inst_318_, lean_object* v_inst_319_, lean_object* v_s_320_){
_start:
{
lean_object* v_res_321_; 
v_res_321_ = lp_mathlib_Subgroup_ofClass(v_S_315_, v_G_316_, v_inst_317_, v_inst_318_, v_inst_319_, v_s_320_);
lean_dec(v_s_320_);
lean_dec_ref(v_inst_317_);
return v_res_321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_ofClass(lean_object* v_S_322_, lean_object* v_G_323_, lean_object* v_inst_324_, lean_object* v_inst_325_, lean_object* v_inst_326_, lean_object* v_s_327_){
_start:
{
lean_object* v___x_328_; 
v___x_328_ = lean_box(0);
return v___x_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_ofClass___boxed(lean_object* v_S_329_, lean_object* v_G_330_, lean_object* v_inst_331_, lean_object* v_inst_332_, lean_object* v_inst_333_, lean_object* v_s_334_){
_start:
{
lean_object* v_res_335_; 
v_res_335_ = lp_mathlib_AddSubgroup_ofClass(v_S_329_, v_G_330_, v_inst_331_, v_inst_332_, v_inst_333_, v_s_334_);
lean_dec(v_s_334_);
lean_dec_ref(v_inst_331_);
return v_res_335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_copy(lean_object* v_G_336_, lean_object* v_inst_337_, lean_object* v_K_338_, lean_object* v_s_339_, lean_object* v_hs_340_){
_start:
{
lean_object* v___x_341_; 
v___x_341_ = lean_box(0);
return v___x_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_copy___boxed(lean_object* v_G_342_, lean_object* v_inst_343_, lean_object* v_K_344_, lean_object* v_s_345_, lean_object* v_hs_346_){
_start:
{
lean_object* v_res_347_; 
v_res_347_ = lp_mathlib_Subgroup_copy(v_G_342_, v_inst_343_, v_K_344_, v_s_345_, v_hs_346_);
lean_dec_ref(v_inst_343_);
return v_res_347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_copy(lean_object* v_G_348_, lean_object* v_inst_349_, lean_object* v_K_350_, lean_object* v_s_351_, lean_object* v_hs_352_){
_start:
{
lean_object* v___x_353_; 
v___x_353_ = lean_box(0);
return v___x_353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_copy___boxed(lean_object* v_G_354_, lean_object* v_inst_355_, lean_object* v_K_356_, lean_object* v_s_357_, lean_object* v_hs_358_){
_start:
{
lean_object* v_res_359_; 
v_res_359_ = lp_mathlib_AddSubgroup_copy(v_G_354_, v_inst_355_, v_K_356_, v_s_357_, v_hs_358_);
lean_dec_ref(v_inst_355_);
return v_res_359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_ofDiv(lean_object* v_G_360_, lean_object* v_inst_361_, lean_object* v_s_362_, lean_object* v_hsn_363_, lean_object* v_hs_364_){
_start:
{
lean_object* v___x_365_; 
v___x_365_ = lean_box(0);
return v___x_365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_ofDiv___boxed(lean_object* v_G_366_, lean_object* v_inst_367_, lean_object* v_s_368_, lean_object* v_hsn_369_, lean_object* v_hs_370_){
_start:
{
lean_object* v_res_371_; 
v_res_371_ = lp_mathlib_Subgroup_ofDiv(v_G_366_, v_inst_367_, v_s_368_, v_hsn_369_, v_hs_370_);
lean_dec_ref(v_inst_367_);
return v_res_371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_ofSub(lean_object* v_G_372_, lean_object* v_inst_373_, lean_object* v_s_374_, lean_object* v_hsn_375_, lean_object* v_hs_376_){
_start:
{
lean_object* v___x_377_; 
v___x_377_ = lean_box(0);
return v___x_377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_ofSub___boxed(lean_object* v_G_378_, lean_object* v_inst_379_, lean_object* v_s_380_, lean_object* v_hsn_381_, lean_object* v_hs_382_){
_start:
{
lean_object* v_res_383_; 
v_res_383_ = lp_mathlib_AddSubgroup_ofSub(v_G_378_, v_inst_379_, v_s_380_, v_hsn_381_, v_hs_382_);
lean_dec_ref(v_inst_379_);
return v_res_383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_mul___redArg(lean_object* v_inst_384_){
_start:
{
lean_object* v_toMonoid_385_; lean_object* v___x_386_; lean_object* v___x_387_; 
v_toMonoid_385_ = lean_ctor_get(v_inst_384_, 0);
v___x_386_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_385_);
v___x_387_ = lp_mathlib_Submonoid_mul___redArg(v___x_386_);
return v___x_387_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_mul___redArg___boxed(lean_object* v_inst_388_){
_start:
{
lean_object* v_res_389_; 
v_res_389_ = lp_mathlib_Subgroup_mul___redArg(v_inst_388_);
lean_dec_ref(v_inst_388_);
return v_res_389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_mul(lean_object* v_G_390_, lean_object* v_inst_391_, lean_object* v_H_392_){
_start:
{
lean_object* v___x_393_; 
v___x_393_ = lp_mathlib_Subgroup_mul___redArg(v_inst_391_);
return v___x_393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_mul___boxed(lean_object* v_G_394_, lean_object* v_inst_395_, lean_object* v_H_396_){
_start:
{
lean_object* v_res_397_; 
v_res_397_ = lp_mathlib_Subgroup_mul(v_G_394_, v_inst_395_, v_H_396_);
lean_dec_ref(v_inst_395_);
return v_res_397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_add___redArg(lean_object* v_inst_398_){
_start:
{
lean_object* v_toAddMonoid_399_; lean_object* v___x_400_; lean_object* v___x_401_; 
v_toAddMonoid_399_ = lean_ctor_get(v_inst_398_, 0);
v___x_400_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_399_);
v___x_401_ = lp_mathlib_AddSubmonoid_add___redArg(v___x_400_);
return v___x_401_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_add___redArg___boxed(lean_object* v_inst_402_){
_start:
{
lean_object* v_res_403_; 
v_res_403_ = lp_mathlib_AddSubgroup_add___redArg(v_inst_402_);
lean_dec_ref(v_inst_402_);
return v_res_403_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_add(lean_object* v_G_404_, lean_object* v_inst_405_, lean_object* v_H_406_){
_start:
{
lean_object* v___x_407_; 
v___x_407_ = lp_mathlib_AddSubgroup_add___redArg(v_inst_405_);
return v___x_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_add___boxed(lean_object* v_G_408_, lean_object* v_inst_409_, lean_object* v_H_410_){
_start:
{
lean_object* v_res_411_; 
v_res_411_ = lp_mathlib_AddSubgroup_add(v_G_408_, v_inst_409_, v_H_410_);
lean_dec_ref(v_inst_409_);
return v_res_411_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_one___redArg(lean_object* v_inst_412_){
_start:
{
lean_object* v_toMonoid_413_; lean_object* v___x_414_; lean_object* v___x_415_; 
v_toMonoid_413_ = lean_ctor_get(v_inst_412_, 0);
v___x_414_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_413_);
v___x_415_ = lp_mathlib_Submonoid_one___redArg(v___x_414_);
return v___x_415_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_one___redArg___boxed(lean_object* v_inst_416_){
_start:
{
lean_object* v_res_417_; 
v_res_417_ = lp_mathlib_Subgroup_one___redArg(v_inst_416_);
lean_dec_ref(v_inst_416_);
return v_res_417_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_one(lean_object* v_G_418_, lean_object* v_inst_419_, lean_object* v_H_420_){
_start:
{
lean_object* v___x_421_; 
v___x_421_ = lp_mathlib_Subgroup_one___redArg(v_inst_419_);
return v___x_421_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_one___boxed(lean_object* v_G_422_, lean_object* v_inst_423_, lean_object* v_H_424_){
_start:
{
lean_object* v_res_425_; 
v_res_425_ = lp_mathlib_Subgroup_one(v_G_422_, v_inst_423_, v_H_424_);
lean_dec_ref(v_inst_423_);
return v_res_425_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_zero___redArg(lean_object* v_inst_426_){
_start:
{
lean_object* v_toAddMonoid_427_; lean_object* v___x_428_; lean_object* v___x_429_; 
v_toAddMonoid_427_ = lean_ctor_get(v_inst_426_, 0);
v___x_428_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_427_);
v___x_429_ = lp_mathlib_AddSubmonoid_zero___redArg(v___x_428_);
return v___x_429_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_zero___redArg___boxed(lean_object* v_inst_430_){
_start:
{
lean_object* v_res_431_; 
v_res_431_ = lp_mathlib_AddSubgroup_zero___redArg(v_inst_430_);
lean_dec_ref(v_inst_430_);
return v_res_431_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_zero(lean_object* v_G_432_, lean_object* v_inst_433_, lean_object* v_H_434_){
_start:
{
lean_object* v___x_435_; 
v___x_435_ = lp_mathlib_AddSubgroup_zero___redArg(v_inst_433_);
return v___x_435_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_zero___boxed(lean_object* v_G_436_, lean_object* v_inst_437_, lean_object* v_H_438_){
_start:
{
lean_object* v_res_439_; 
v_res_439_ = lp_mathlib_AddSubgroup_zero(v_G_436_, v_inst_437_, v_H_438_);
lean_dec_ref(v_inst_437_);
return v_res_439_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_inv___redArg___lam__0(lean_object* v_toInv_440_, lean_object* v_a_441_){
_start:
{
lean_object* v___x_442_; 
v___x_442_ = lean_apply_1(v_toInv_440_, v_a_441_);
return v___x_442_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_inv___redArg(lean_object* v_inst_443_){
_start:
{
lean_object* v___x_444_; lean_object* v_toInv_445_; lean_object* v___f_446_; 
v___x_444_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v_inst_443_);
v_toInv_445_ = lean_ctor_get(v___x_444_, 1);
lean_inc(v_toInv_445_);
lean_dec_ref(v___x_444_);
v___f_446_ = lean_alloc_closure((void*)(lp_mathlib_Subgroup_inv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_446_, 0, v_toInv_445_);
return v___f_446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_inv___redArg___boxed(lean_object* v_inst_447_){
_start:
{
lean_object* v_res_448_; 
v_res_448_ = lp_mathlib_Subgroup_inv___redArg(v_inst_447_);
lean_dec_ref(v_inst_447_);
return v_res_448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_inv(lean_object* v_G_449_, lean_object* v_inst_450_, lean_object* v_H_451_){
_start:
{
lean_object* v___x_452_; 
v___x_452_ = lp_mathlib_Subgroup_inv___redArg(v_inst_450_);
return v___x_452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_inv___boxed(lean_object* v_G_453_, lean_object* v_inst_454_, lean_object* v_H_455_){
_start:
{
lean_object* v_res_456_; 
v_res_456_ = lp_mathlib_Subgroup_inv(v_G_453_, v_inst_454_, v_H_455_);
lean_dec_ref(v_inst_454_);
return v_res_456_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_neg___redArg___lam__0(lean_object* v_toNeg_457_, lean_object* v_a_458_){
_start:
{
lean_object* v___x_459_; 
v___x_459_ = lean_apply_1(v_toNeg_457_, v_a_458_);
return v___x_459_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_neg___redArg(lean_object* v_inst_460_){
_start:
{
lean_object* v___x_461_; lean_object* v_toNeg_462_; lean_object* v___f_463_; 
v___x_461_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_460_);
v_toNeg_462_ = lean_ctor_get(v___x_461_, 1);
lean_inc(v_toNeg_462_);
lean_dec_ref(v___x_461_);
v___f_463_ = lean_alloc_closure((void*)(lp_mathlib_AddSubgroup_neg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_463_, 0, v_toNeg_462_);
return v___f_463_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_neg___redArg___boxed(lean_object* v_inst_464_){
_start:
{
lean_object* v_res_465_; 
v_res_465_ = lp_mathlib_AddSubgroup_neg___redArg(v_inst_464_);
lean_dec_ref(v_inst_464_);
return v_res_465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_neg(lean_object* v_G_466_, lean_object* v_inst_467_, lean_object* v_H_468_){
_start:
{
lean_object* v___x_469_; 
v___x_469_ = lp_mathlib_AddSubgroup_neg___redArg(v_inst_467_);
return v___x_469_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_neg___boxed(lean_object* v_G_470_, lean_object* v_inst_471_, lean_object* v_H_472_){
_start:
{
lean_object* v_res_473_; 
v_res_473_ = lp_mathlib_AddSubgroup_neg(v_G_470_, v_inst_471_, v_H_472_);
lean_dec_ref(v_inst_471_);
return v_res_473_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_div___redArg(lean_object* v_inst_474_){
_start:
{
lean_object* v_toDiv_475_; lean_object* v___f_476_; 
v_toDiv_475_ = lean_ctor_get(v_inst_474_, 2);
lean_inc(v_toDiv_475_);
lean_dec_ref(v_inst_474_);
v___f_476_ = lean_alloc_closure((void*)(lp_mathlib_SubgroupClass_div___redArg___lam__0), 3, 1);
lean_closure_set(v___f_476_, 0, v_toDiv_475_);
return v___f_476_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_div(lean_object* v_G_477_, lean_object* v_inst_478_, lean_object* v_H_479_){
_start:
{
lean_object* v___x_480_; 
v___x_480_ = lp_mathlib_Subgroup_div___redArg(v_inst_478_);
return v___x_480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_sub___redArg(lean_object* v_inst_481_){
_start:
{
lean_object* v_toSub_482_; lean_object* v___f_483_; 
v_toSub_482_ = lean_ctor_get(v_inst_481_, 2);
lean_inc(v_toSub_482_);
lean_dec_ref(v_inst_481_);
v___f_483_ = lean_alloc_closure((void*)(lp_mathlib_AddSubgroupClass_sub___redArg___lam__0), 3, 1);
lean_closure_set(v___f_483_, 0, v_toSub_482_);
return v___f_483_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_sub(lean_object* v_G_484_, lean_object* v_inst_485_, lean_object* v_H_486_){
_start:
{
lean_object* v___x_487_; 
v___x_487_ = lp_mathlib_AddSubgroup_sub___redArg(v_inst_485_);
return v___x_487_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_npow___redArg___lam__0(lean_object* v_toNPow_488_, lean_object* v_a_489_, lean_object* v_n_490_){
_start:
{
lean_object* v___x_491_; 
v___x_491_ = lean_apply_2(v_toNPow_488_, v_n_490_, v_a_489_);
return v___x_491_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_npow___redArg(lean_object* v_inst_492_){
_start:
{
lean_object* v_toMonoid_493_; lean_object* v_toNPow_494_; lean_object* v___f_495_; 
v_toMonoid_493_ = lean_ctor_get(v_inst_492_, 0);
lean_inc_ref(v_toMonoid_493_);
lean_dec_ref(v_inst_492_);
v_toNPow_494_ = lean_ctor_get(v_toMonoid_493_, 2);
lean_inc(v_toNPow_494_);
lean_dec_ref(v_toMonoid_493_);
v___f_495_ = lean_alloc_closure((void*)(lp_mathlib_Subgroup_npow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_495_, 0, v_toNPow_494_);
return v___f_495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_npow(lean_object* v_G_496_, lean_object* v_inst_497_, lean_object* v_H_498_){
_start:
{
lean_object* v___x_499_; 
v___x_499_ = lp_mathlib_Subgroup_npow___redArg(v_inst_497_);
return v___x_499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_nsmul___redArg___lam__0(lean_object* v_toNSMul_500_, lean_object* v_n_501_, lean_object* v_a_502_){
_start:
{
lean_object* v___x_503_; 
v___x_503_ = lean_apply_2(v_toNSMul_500_, v_n_501_, v_a_502_);
return v___x_503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_nsmul___redArg(lean_object* v_inst_504_){
_start:
{
lean_object* v_toAddMonoid_505_; lean_object* v_toNSMul_506_; lean_object* v___f_507_; 
v_toAddMonoid_505_ = lean_ctor_get(v_inst_504_, 0);
lean_inc_ref(v_toAddMonoid_505_);
lean_dec_ref(v_inst_504_);
v_toNSMul_506_ = lean_ctor_get(v_toAddMonoid_505_, 2);
lean_inc(v_toNSMul_506_);
lean_dec_ref(v_toAddMonoid_505_);
v___f_507_ = lean_alloc_closure((void*)(lp_mathlib_AddSubgroup_nsmul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_507_, 0, v_toNSMul_506_);
return v___f_507_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_nsmul(lean_object* v_G_508_, lean_object* v_inst_509_, lean_object* v_H_510_){
_start:
{
lean_object* v___x_511_; 
v___x_511_ = lp_mathlib_AddSubgroup_nsmul___redArg(v_inst_509_);
return v___x_511_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_zpow___redArg(lean_object* v_inst_512_){
_start:
{
lean_object* v_toZPow_513_; lean_object* v___f_514_; 
v_toZPow_513_ = lean_ctor_get(v_inst_512_, 3);
lean_inc(v_toZPow_513_);
lean_dec_ref(v_inst_512_);
v___f_514_ = lean_alloc_closure((void*)(lp_mathlib_SubgroupClass_instZPow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_514_, 0, v_toZPow_513_);
return v___f_514_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_zpow(lean_object* v_G_515_, lean_object* v_inst_516_, lean_object* v_H_517_){
_start:
{
lean_object* v___x_518_; 
v___x_518_ = lp_mathlib_Subgroup_zpow___redArg(v_inst_516_);
return v___x_518_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_zsmul___redArg(lean_object* v_inst_519_){
_start:
{
lean_object* v_toZSMul_520_; lean_object* v___f_521_; 
v_toZSMul_520_ = lean_ctor_get(v_inst_519_, 3);
lean_inc(v_toZSMul_520_);
lean_dec_ref(v_inst_519_);
v___f_521_ = lean_alloc_closure((void*)(lp_mathlib_AddSubgroupClass_instZSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_521_, 0, v_toZSMul_520_);
return v___f_521_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_zsmul(lean_object* v_G_522_, lean_object* v_inst_523_, lean_object* v_H_524_){
_start:
{
lean_object* v___x_525_; 
v___x_525_ = lp_mathlib_AddSubgroup_zsmul___redArg(v_inst_523_);
return v___x_525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_toGroup___redArg(lean_object* v_inst_526_){
_start:
{
lean_object* v___x_527_; 
v___x_527_ = lp_mathlib_SubgroupClass_toGroup___redArg(v_inst_526_);
return v___x_527_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_toGroup(lean_object* v_G_528_, lean_object* v_inst_529_, lean_object* v_H_530_){
_start:
{
lean_object* v___x_531_; 
v___x_531_ = lp_mathlib_SubgroupClass_toGroup___redArg(v_inst_529_);
return v___x_531_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_toAddGroup___redArg(lean_object* v_inst_532_){
_start:
{
lean_object* v___x_533_; 
v___x_533_ = lp_mathlib_AddSubgroupClass_toAddGroup___redArg(v_inst_532_);
return v___x_533_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_toAddGroup(lean_object* v_G_534_, lean_object* v_inst_535_, lean_object* v_H_536_){
_start:
{
lean_object* v___x_537_; 
v___x_537_ = lp_mathlib_AddSubgroupClass_toAddGroup___redArg(v_inst_535_);
return v___x_537_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_toCommGroup___redArg(lean_object* v_inst_538_){
_start:
{
lean_object* v___x_539_; 
v___x_539_ = lp_mathlib_SubgroupClass_toGroup___redArg(v_inst_538_);
return v___x_539_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_toCommGroup(lean_object* v_G_540_, lean_object* v_inst_541_, lean_object* v_H_542_){
_start:
{
lean_object* v___x_543_; 
v___x_543_ = lp_mathlib_SubgroupClass_toGroup___redArg(v_inst_541_);
return v___x_543_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_toAddCommGroup___redArg(lean_object* v_inst_544_){
_start:
{
lean_object* v___x_545_; 
v___x_545_ = lp_mathlib_AddSubgroupClass_toAddGroup___redArg(v_inst_544_);
return v___x_545_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_toAddCommGroup(lean_object* v_G_546_, lean_object* v_inst_547_, lean_object* v_H_548_){
_start:
{
lean_object* v___x_549_; 
v___x_549_ = lp_mathlib_AddSubgroupClass_toAddGroup___redArg(v_inst_547_);
return v___x_549_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_subtype(lean_object* v_G_550_, lean_object* v_inst_551_, lean_object* v_H_552_){
_start:
{
lean_object* v___f_553_; 
v___f_553_ = ((lean_object*)(lp_mathlib_SubgroupClass_subtype___closed__0));
return v___f_553_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_subtype___boxed(lean_object* v_G_554_, lean_object* v_inst_555_, lean_object* v_H_556_){
_start:
{
lean_object* v_res_557_; 
v_res_557_ = lp_mathlib_Subgroup_subtype(v_G_554_, v_inst_555_, v_H_556_);
lean_dec_ref(v_inst_555_);
return v_res_557_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_subtype(lean_object* v_G_558_, lean_object* v_inst_559_, lean_object* v_H_560_){
_start:
{
lean_object* v___f_561_; 
v___f_561_ = ((lean_object*)(lp_mathlib_SubgroupClass_subtype___closed__0));
return v___f_561_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_subtype___boxed(lean_object* v_G_562_, lean_object* v_inst_563_, lean_object* v_H_564_){
_start:
{
lean_object* v_res_565_; 
v_res_565_ = lp_mathlib_AddSubgroup_subtype(v_G_562_, v_inst_563_, v_H_564_);
lean_dec_ref(v_inst_563_);
return v_res_565_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_inclusion(lean_object* v_G_566_, lean_object* v_inst_567_, lean_object* v_H_568_, lean_object* v_K_569_, lean_object* v_h_570_){
_start:
{
lean_object* v___f_571_; 
v___f_571_ = ((lean_object*)(lp_mathlib_SubgroupClass_inclusion___closed__0));
return v___f_571_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_inclusion___boxed(lean_object* v_G_572_, lean_object* v_inst_573_, lean_object* v_H_574_, lean_object* v_K_575_, lean_object* v_h_576_){
_start:
{
lean_object* v_res_577_; 
v_res_577_ = lp_mathlib_Subgroup_inclusion(v_G_572_, v_inst_573_, v_H_574_, v_K_575_, v_h_576_);
lean_dec_ref(v_inst_573_);
return v_res_577_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_inclusion(lean_object* v_G_578_, lean_object* v_inst_579_, lean_object* v_H_580_, lean_object* v_K_581_, lean_object* v_h_582_){
_start:
{
lean_object* v___f_583_; 
v___f_583_ = ((lean_object*)(lp_mathlib_SubgroupClass_inclusion___closed__0));
return v___f_583_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_inclusion___boxed(lean_object* v_G_584_, lean_object* v_inst_585_, lean_object* v_H_586_, lean_object* v_K_587_, lean_object* v_h_588_){
_start:
{
lean_object* v_res_589_; 
v_res_589_ = lp_mathlib_AddSubgroup_inclusion(v_G_584_, v_inst_585_, v_H_586_, v_K_587_, v_h_588_);
lean_dec_ref(v_inst_585_);
return v_res_589_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_normalizer(lean_object* v_G_590_, lean_object* v_inst_591_, lean_object* v_S_592_){
_start:
{
lean_object* v___x_593_; 
v___x_593_ = lean_box(0);
return v___x_593_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_normalizer___boxed(lean_object* v_G_594_, lean_object* v_inst_595_, lean_object* v_S_596_){
_start:
{
lean_object* v_res_597_; 
v_res_597_ = lp_mathlib_Subgroup_normalizer(v_G_594_, v_inst_595_, v_S_596_);
lean_dec_ref(v_inst_595_);
return v_res_597_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_normalizer(lean_object* v_G_598_, lean_object* v_inst_599_, lean_object* v_S_600_){
_start:
{
lean_object* v___x_601_; 
v___x_601_ = lean_box(0);
return v___x_601_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_normalizer___boxed(lean_object* v_G_602_, lean_object* v_inst_603_, lean_object* v_S_604_){
_start:
{
lean_object* v_res_605_; 
v_res_605_ = lp_mathlib_AddSubgroup_normalizer(v_G_602_, v_inst_603_, v_S_604_);
lean_dec_ref(v_inst_603_);
return v_res_605_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_setNormalizer(lean_object* v_G_606_, lean_object* v_inst_607_, lean_object* v_S_608_){
_start:
{
lean_object* v___x_609_; 
v___x_609_ = lean_box(0);
return v___x_609_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_setNormalizer___boxed(lean_object* v_G_610_, lean_object* v_inst_611_, lean_object* v_S_612_){
_start:
{
lean_object* v_res_613_; 
v_res_613_ = lp_mathlib_Subgroup_setNormalizer(v_G_610_, v_inst_611_, v_S_612_);
lean_dec_ref(v_inst_611_);
return v_res_613_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_setNormalizer(lean_object* v_G_614_, lean_object* v_inst_615_, lean_object* v_S_616_){
_start:
{
lean_object* v___x_617_; 
v___x_617_ = lean_box(0);
return v___x_617_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_setNormalizer___boxed(lean_object* v_G_618_, lean_object* v_inst_619_, lean_object* v_S_620_){
_start:
{
lean_object* v_res_621_; 
v_res_621_ = lp_mathlib_AddSubgroup_setNormalizer(v_G_618_, v_inst_619_, v_S_620_);
lean_dec_ref(v_inst_619_);
return v_res_621_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Inclusion(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FastInstance(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Inclusion(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FastInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Inclusion(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FastInstance(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Inclusion(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FastInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
