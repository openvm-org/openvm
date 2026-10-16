// Lean compiler output
// Module: Mathlib.Algebra.Group.Subgroup.Lattice
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Submonoid.Operations public import Mathlib.Algebra.Group.Subgroup.Defs
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
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_Additive_subNegMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Submonoid_toAddSubmonoid(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_toEmbedding___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddSubmonoid_toSubmonoid(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Subgroup_instPartialOrder(lean_object*, lean_object*);
lean_object* lp_mathlib_completeLatticeOfInf___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Submonoid_topEquiv(lean_object*, lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_AddSubmonoid_topEquiv(lean_object*, lean_object*);
lean_object* lp_mathlib_AddSubgroup_instPartialOrder(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiplicative_divInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_toAddSubgroup___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_toAddSubgroup___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_toAddSubgroup___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_toAddSubgroup___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_toAddSubgroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_toAddSubgroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_toSubgroup_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_toSubgroup_x27(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_toSubgroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_toSubgroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_toAddSubgroup_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_toAddSubgroup_x27(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instTop(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instTop___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instTop(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instTop___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_topEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_topEquiv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_topEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_topEquiv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_topEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_topEquiv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_topEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_topEquiv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instBot___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instBot___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instUniqueSubtypeMemBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instUniqueSubtypeMemBot___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instUniqueSubtypeMemBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instUniqueSubtypeMemBot___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instUniqueSubtypeMemBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instUniqueSubtypeMemBot___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instUniqueSubtypeMemBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instUniqueSubtypeMemBot___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instMin___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Subgroup_instMin___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subgroup_instMin___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subgroup_instMin___closed__0 = (const lean_object*)&lp_mathlib_Subgroup_instMin___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instMin(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instMin___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instMin___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AddSubgroup_instMin___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddSubgroup_instMin___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddSubgroup_instMin___closed__0 = (const lean_object*)&lp_mathlib_AddSubgroup_instMin___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instMin(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instMin___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instInfSet___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Subgroup_instInfSet___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subgroup_instInfSet___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subgroup_instInfSet___closed__0 = (const lean_object*)&lp_mathlib_Subgroup_instInfSet___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instInfSet(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instInfSet___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instInfSet___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_AddSubgroup_instInfSet___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddSubgroup_instInfSet___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddSubgroup_instInfSet___closed__0 = (const lean_object*)&lp_mathlib_AddSubgroup_instInfSet___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instInfSet(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instInfSet___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instCompleteLattice___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Subgroup_instCompleteLattice___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subgroup_instCompleteLattice___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subgroup_instCompleteLattice___redArg___closed__0 = (const lean_object*)&lp_mathlib_Subgroup_instCompleteLattice___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Subgroup_instCompleteLattice___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Subgroup_instCompleteLattice___redArg___closed__1 = (const lean_object*)&lp_mathlib_Subgroup_instCompleteLattice___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instCompleteLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instCompleteLattice___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instCompleteLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instCompleteLattice___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instCompleteLattice___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AddSubgroup_instCompleteLattice___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddSubgroup_instCompleteLattice___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddSubgroup_instCompleteLattice___redArg___closed__0 = (const lean_object*)&lp_mathlib_AddSubgroup_instCompleteLattice___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_AddSubgroup_instCompleteLattice___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_AddSubgroup_instCompleteLattice___redArg___closed__1 = (const lean_object*)&lp_mathlib_AddSubgroup_instCompleteLattice___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instCompleteLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instCompleteLattice___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instCompleteLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instCompleteLattice___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instUniqueOfSubsingleton(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instUniqueOfSubsingleton___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instUniqueOfSubsingleton(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instUniqueOfSubsingleton___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_closure(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_closure___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_closure(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_closure___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_gi___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Subgroup_gi___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subgroup_gi___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subgroup_gi___closed__0 = (const lean_object*)&lp_mathlib_Subgroup_gi___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_gi(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_gi___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_gi___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AddSubgroup_gi___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddSubgroup_gi___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddSubgroup_gi___closed__0 = (const lean_object*)&lp_mathlib_AddSubgroup_gi___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_gi(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_gi___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_toAddSubgroup___redArg___lam__0(lean_object* v___x_1_, lean_object* v_S_2_){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; 
v___x_3_ = lp_mathlib_Submonoid_toAddSubmonoid(lean_box(0), v___x_1_);
v___x_4_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v___x_3_, v_S_2_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_toAddSubgroup___redArg___lam__0___boxed(lean_object* v___x_5_, lean_object* v_S_6_){
_start:
{
lean_object* v_res_7_; 
v_res_7_ = lp_mathlib_Subgroup_toAddSubgroup___redArg___lam__0(v___x_5_, v_S_6_);
lean_dec_ref(v___x_5_);
return v_res_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_toAddSubgroup___redArg___lam__1(lean_object* v___x_8_, lean_object* v_S_9_){
_start:
{
lean_object* v___x_10_; lean_object* v___x_11_; 
v___x_10_ = lp_mathlib_AddSubmonoid_toSubmonoid(lean_box(0), v___x_8_);
v___x_11_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v___x_10_, v_S_9_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_toAddSubgroup___redArg___lam__1___boxed(lean_object* v___x_12_, lean_object* v_S_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_Subgroup_toAddSubgroup___redArg___lam__1(v___x_12_, v_S_13_);
lean_dec_ref(v___x_12_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_toAddSubgroup___redArg(lean_object* v_inst_15_){
_start:
{
lean_object* v_toMonoid_16_; lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v_toAddMonoid_19_; lean_object* v___f_20_; lean_object* v___x_21_; lean_object* v___f_22_; lean_object* v___x_23_; 
v_toMonoid_16_ = lean_ctor_get(v_inst_15_, 0);
v___x_17_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_16_);
v___x_18_ = lp_mathlib_Additive_subNegMonoid___redArg(v_inst_15_);
v_toAddMonoid_19_ = lean_ctor_get(v___x_18_, 0);
lean_inc_ref(v_toAddMonoid_19_);
lean_dec_ref(v___x_18_);
v___f_20_ = lean_alloc_closure((void*)(lp_mathlib_Subgroup_toAddSubgroup___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_20_, 0, v___x_17_);
v___x_21_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_19_);
lean_dec_ref(v_toAddMonoid_19_);
v___f_22_ = lean_alloc_closure((void*)(lp_mathlib_Subgroup_toAddSubgroup___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_22_, 0, v___x_21_);
v___x_23_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_23_, 0, v___f_20_);
lean_ctor_set(v___x_23_, 1, v___f_22_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_toAddSubgroup(lean_object* v_G_24_, lean_object* v_inst_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lp_mathlib_Subgroup_toAddSubgroup___redArg(v_inst_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_toSubgroup_x27___redArg(lean_object* v_inst_27_){
_start:
{
lean_object* v___x_28_; lean_object* v___x_29_; 
v___x_28_ = lp_mathlib_Subgroup_toAddSubgroup___redArg(v_inst_27_);
v___x_29_ = lp_mathlib_Equiv_symm___redArg(v___x_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_toSubgroup_x27(lean_object* v_G_30_, lean_object* v_inst_31_){
_start:
{
lean_object* v___x_32_; lean_object* v___x_33_; 
v___x_32_ = lp_mathlib_Subgroup_toAddSubgroup___redArg(v_inst_31_);
v___x_33_ = lp_mathlib_Equiv_symm___redArg(v___x_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_toSubgroup___redArg(lean_object* v_inst_34_){
_start:
{
lean_object* v_toAddMonoid_35_; lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v_toMonoid_38_; lean_object* v___f_39_; lean_object* v___x_40_; lean_object* v___f_41_; lean_object* v___x_42_; 
v_toAddMonoid_35_ = lean_ctor_get(v_inst_34_, 0);
v___x_36_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_35_);
v___x_37_ = lp_mathlib_Multiplicative_divInvMonoid___redArg(v_inst_34_);
v_toMonoid_38_ = lean_ctor_get(v___x_37_, 0);
lean_inc_ref(v_toMonoid_38_);
lean_dec_ref(v___x_37_);
v___f_39_ = lean_alloc_closure((void*)(lp_mathlib_Subgroup_toAddSubgroup___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_39_, 0, v___x_36_);
v___x_40_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_38_);
lean_dec_ref(v_toMonoid_38_);
v___f_41_ = lean_alloc_closure((void*)(lp_mathlib_Subgroup_toAddSubgroup___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_41_, 0, v___x_40_);
v___x_42_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_42_, 0, v___f_39_);
lean_ctor_set(v___x_42_, 1, v___f_41_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_toSubgroup(lean_object* v_A_43_, lean_object* v_inst_44_){
_start:
{
lean_object* v___x_45_; 
v___x_45_ = lp_mathlib_AddSubgroup_toSubgroup___redArg(v_inst_44_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_toAddSubgroup_x27___redArg(lean_object* v_inst_46_){
_start:
{
lean_object* v___x_47_; lean_object* v___x_48_; 
v___x_47_ = lp_mathlib_AddSubgroup_toSubgroup___redArg(v_inst_46_);
v___x_48_ = lp_mathlib_Equiv_symm___redArg(v___x_47_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_toAddSubgroup_x27(lean_object* v_A_49_, lean_object* v_inst_50_){
_start:
{
lean_object* v___x_51_; lean_object* v___x_52_; 
v___x_51_ = lp_mathlib_AddSubgroup_toSubgroup___redArg(v_inst_50_);
v___x_52_ = lp_mathlib_Equiv_symm___redArg(v___x_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instTop(lean_object* v_G_53_, lean_object* v_inst_54_){
_start:
{
lean_object* v___x_55_; 
v___x_55_ = lean_box(0);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instTop___boxed(lean_object* v_G_56_, lean_object* v_inst_57_){
_start:
{
lean_object* v_res_58_; 
v_res_58_ = lp_mathlib_Subgroup_instTop(v_G_56_, v_inst_57_);
lean_dec_ref(v_inst_57_);
return v_res_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instTop(lean_object* v_G_59_, lean_object* v_inst_60_){
_start:
{
lean_object* v___x_61_; 
v___x_61_ = lean_box(0);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instTop___boxed(lean_object* v_G_62_, lean_object* v_inst_63_){
_start:
{
lean_object* v_res_64_; 
v_res_64_ = lp_mathlib_AddSubgroup_instTop(v_G_62_, v_inst_63_);
lean_dec_ref(v_inst_63_);
return v_res_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_topEquiv___redArg(lean_object* v_inst_65_){
_start:
{
lean_object* v_toMonoid_66_; lean_object* v___x_67_; lean_object* v___x_68_; 
v_toMonoid_66_ = lean_ctor_get(v_inst_65_, 0);
v___x_67_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_66_);
v___x_68_ = lp_mathlib_Submonoid_topEquiv(lean_box(0), v___x_67_);
lean_dec_ref(v___x_67_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_topEquiv___redArg___boxed(lean_object* v_inst_69_){
_start:
{
lean_object* v_res_70_; 
v_res_70_ = lp_mathlib_Subgroup_topEquiv___redArg(v_inst_69_);
lean_dec_ref(v_inst_69_);
return v_res_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_topEquiv(lean_object* v_G_71_, lean_object* v_inst_72_){
_start:
{
lean_object* v___x_73_; 
v___x_73_ = lp_mathlib_Subgroup_topEquiv___redArg(v_inst_72_);
return v___x_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_topEquiv___boxed(lean_object* v_G_74_, lean_object* v_inst_75_){
_start:
{
lean_object* v_res_76_; 
v_res_76_ = lp_mathlib_Subgroup_topEquiv(v_G_74_, v_inst_75_);
lean_dec_ref(v_inst_75_);
return v_res_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_topEquiv___redArg(lean_object* v_inst_77_){
_start:
{
lean_object* v_toAddMonoid_78_; lean_object* v___x_79_; lean_object* v___x_80_; 
v_toAddMonoid_78_ = lean_ctor_get(v_inst_77_, 0);
v___x_79_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_78_);
v___x_80_ = lp_mathlib_AddSubmonoid_topEquiv(lean_box(0), v___x_79_);
lean_dec_ref(v___x_79_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_topEquiv___redArg___boxed(lean_object* v_inst_81_){
_start:
{
lean_object* v_res_82_; 
v_res_82_ = lp_mathlib_AddSubgroup_topEquiv___redArg(v_inst_81_);
lean_dec_ref(v_inst_81_);
return v_res_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_topEquiv(lean_object* v_G_83_, lean_object* v_inst_84_){
_start:
{
lean_object* v___x_85_; 
v___x_85_ = lp_mathlib_AddSubgroup_topEquiv___redArg(v_inst_84_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_topEquiv___boxed(lean_object* v_G_86_, lean_object* v_inst_87_){
_start:
{
lean_object* v_res_88_; 
v_res_88_ = lp_mathlib_AddSubgroup_topEquiv(v_G_86_, v_inst_87_);
lean_dec_ref(v_inst_87_);
return v_res_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instBot(lean_object* v_G_89_, lean_object* v_inst_90_){
_start:
{
lean_object* v___x_91_; 
v___x_91_ = lean_box(0);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instBot___boxed(lean_object* v_G_92_, lean_object* v_inst_93_){
_start:
{
lean_object* v_res_94_; 
v_res_94_ = lp_mathlib_Subgroup_instBot(v_G_92_, v_inst_93_);
lean_dec_ref(v_inst_93_);
return v_res_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instBot(lean_object* v_G_95_, lean_object* v_inst_96_){
_start:
{
lean_object* v___x_97_; 
v___x_97_ = lean_box(0);
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instBot___boxed(lean_object* v_G_98_, lean_object* v_inst_99_){
_start:
{
lean_object* v_res_100_; 
v_res_100_ = lp_mathlib_AddSubgroup_instBot(v_G_98_, v_inst_99_);
lean_dec_ref(v_inst_99_);
return v_res_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instInhabited(lean_object* v_G_101_, lean_object* v_inst_102_){
_start:
{
lean_object* v___x_103_; 
v___x_103_ = lean_box(0);
return v___x_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instInhabited___boxed(lean_object* v_G_104_, lean_object* v_inst_105_){
_start:
{
lean_object* v_res_106_; 
v_res_106_ = lp_mathlib_Subgroup_instInhabited(v_G_104_, v_inst_105_);
lean_dec_ref(v_inst_105_);
return v_res_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instInhabited(lean_object* v_G_107_, lean_object* v_inst_108_){
_start:
{
lean_object* v___x_109_; 
v___x_109_ = lean_box(0);
return v___x_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instInhabited___boxed(lean_object* v_G_110_, lean_object* v_inst_111_){
_start:
{
lean_object* v_res_112_; 
v_res_112_ = lp_mathlib_AddSubgroup_instInhabited(v_G_110_, v_inst_111_);
lean_dec_ref(v_inst_111_);
return v_res_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instUniqueSubtypeMemBot___redArg(lean_object* v_inst_113_){
_start:
{
lean_object* v_toMonoid_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v_toOne_117_; 
v_toMonoid_114_ = lean_ctor_get(v_inst_113_, 0);
v___x_115_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_114_);
v___x_116_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_115_);
v_toOne_117_ = lean_ctor_get(v___x_116_, 0);
lean_inc(v_toOne_117_);
lean_dec_ref(v___x_116_);
return v_toOne_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instUniqueSubtypeMemBot___redArg___boxed(lean_object* v_inst_118_){
_start:
{
lean_object* v_res_119_; 
v_res_119_ = lp_mathlib_Subgroup_instUniqueSubtypeMemBot___redArg(v_inst_118_);
lean_dec_ref(v_inst_118_);
return v_res_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instUniqueSubtypeMemBot(lean_object* v_G_120_, lean_object* v_inst_121_){
_start:
{
lean_object* v___x_122_; 
v___x_122_ = lp_mathlib_Subgroup_instUniqueSubtypeMemBot___redArg(v_inst_121_);
return v___x_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instUniqueSubtypeMemBot___boxed(lean_object* v_G_123_, lean_object* v_inst_124_){
_start:
{
lean_object* v_res_125_; 
v_res_125_ = lp_mathlib_Subgroup_instUniqueSubtypeMemBot(v_G_123_, v_inst_124_);
lean_dec_ref(v_inst_124_);
return v_res_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instUniqueSubtypeMemBot___redArg(lean_object* v_inst_126_){
_start:
{
lean_object* v_toAddMonoid_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v_toZero_130_; 
v_toAddMonoid_127_ = lean_ctor_get(v_inst_126_, 0);
v___x_128_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_127_);
v___x_129_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_128_);
v_toZero_130_ = lean_ctor_get(v___x_129_, 0);
lean_inc(v_toZero_130_);
lean_dec_ref(v___x_129_);
return v_toZero_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instUniqueSubtypeMemBot___redArg___boxed(lean_object* v_inst_131_){
_start:
{
lean_object* v_res_132_; 
v_res_132_ = lp_mathlib_AddSubgroup_instUniqueSubtypeMemBot___redArg(v_inst_131_);
lean_dec_ref(v_inst_131_);
return v_res_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instUniqueSubtypeMemBot(lean_object* v_G_133_, lean_object* v_inst_134_){
_start:
{
lean_object* v___x_135_; 
v___x_135_ = lp_mathlib_AddSubgroup_instUniqueSubtypeMemBot___redArg(v_inst_134_);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instUniqueSubtypeMemBot___boxed(lean_object* v_G_136_, lean_object* v_inst_137_){
_start:
{
lean_object* v_res_138_; 
v_res_138_ = lp_mathlib_AddSubgroup_instUniqueSubtypeMemBot(v_G_136_, v_inst_137_);
lean_dec_ref(v_inst_137_);
return v_res_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instMin___lam__0(lean_object* v_H_u2081_139_, lean_object* v_H_u2082_140_){
_start:
{
lean_object* v___x_141_; 
v___x_141_ = lean_box(0);
return v___x_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instMin(lean_object* v_G_143_, lean_object* v_inst_144_){
_start:
{
lean_object* v___f_145_; 
v___f_145_ = ((lean_object*)(lp_mathlib_Subgroup_instMin___closed__0));
return v___f_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instMin___boxed(lean_object* v_G_146_, lean_object* v_inst_147_){
_start:
{
lean_object* v_res_148_; 
v_res_148_ = lp_mathlib_Subgroup_instMin(v_G_146_, v_inst_147_);
lean_dec_ref(v_inst_147_);
return v_res_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instMin___lam__0(lean_object* v_H_u2081_149_, lean_object* v_H_u2082_150_){
_start:
{
lean_object* v___x_151_; 
v___x_151_ = lean_box(0);
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instMin(lean_object* v_G_153_, lean_object* v_inst_154_){
_start:
{
lean_object* v___f_155_; 
v___f_155_ = ((lean_object*)(lp_mathlib_AddSubgroup_instMin___closed__0));
return v___f_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instMin___boxed(lean_object* v_G_156_, lean_object* v_inst_157_){
_start:
{
lean_object* v_res_158_; 
v_res_158_ = lp_mathlib_AddSubgroup_instMin(v_G_156_, v_inst_157_);
lean_dec_ref(v_inst_157_);
return v_res_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instInfSet___lam__0(lean_object* v_s_159_){
_start:
{
lean_object* v___x_160_; 
v___x_160_ = lean_box(0);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instInfSet(lean_object* v_G_162_, lean_object* v_inst_163_){
_start:
{
lean_object* v___f_164_; 
v___f_164_ = ((lean_object*)(lp_mathlib_Subgroup_instInfSet___closed__0));
return v___f_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instInfSet___boxed(lean_object* v_G_165_, lean_object* v_inst_166_){
_start:
{
lean_object* v_res_167_; 
v_res_167_ = lp_mathlib_Subgroup_instInfSet(v_G_165_, v_inst_166_);
lean_dec_ref(v_inst_166_);
return v_res_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instInfSet___lam__0(lean_object* v_s_168_){
_start:
{
lean_object* v___x_169_; 
v___x_169_ = lean_box(0);
return v___x_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instInfSet(lean_object* v_G_171_, lean_object* v_inst_172_){
_start:
{
lean_object* v___f_173_; 
v___f_173_ = ((lean_object*)(lp_mathlib_AddSubgroup_instInfSet___closed__0));
return v___f_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instInfSet___boxed(lean_object* v_G_174_, lean_object* v_inst_175_){
_start:
{
lean_object* v_res_176_; 
v_res_176_ = lp_mathlib_AddSubgroup_instInfSet(v_G_174_, v_inst_175_);
lean_dec_ref(v_inst_175_);
return v_res_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instCompleteLattice___redArg___lam__0(lean_object* v_x1_177_, lean_object* v_x2_178_){
_start:
{
lean_object* v___x_179_; 
v___x_179_ = lean_box(0);
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instCompleteLattice___redArg(lean_object* v_inst_183_){
_start:
{
lean_object* v___x_184_; lean_object* v___f_185_; lean_object* v___x_186_; lean_object* v_toLattice_187_; lean_object* v_toSupSet_188_; lean_object* v_toInfSet_189_; lean_object* v___x_191_; uint8_t v_isShared_192_; uint8_t v_isSharedCheck_207_; 
v___x_184_ = lp_mathlib_Subgroup_instPartialOrder(lean_box(0), v_inst_183_);
v___f_185_ = ((lean_object*)(lp_mathlib_Subgroup_instInfSet___closed__0));
v___x_186_ = lp_mathlib_completeLatticeOfInf___redArg(v___x_184_, v___f_185_);
v_toLattice_187_ = lean_ctor_get(v___x_186_, 0);
v_toSupSet_188_ = lean_ctor_get(v___x_186_, 1);
v_toInfSet_189_ = lean_ctor_get(v___x_186_, 2);
v_isSharedCheck_207_ = !lean_is_exclusive(v___x_186_);
if (v_isSharedCheck_207_ == 0)
{
lean_object* v_unused_208_; 
v_unused_208_ = lean_ctor_get(v___x_186_, 3);
lean_dec(v_unused_208_);
v___x_191_ = v___x_186_;
v_isShared_192_ = v_isSharedCheck_207_;
goto v_resetjp_190_;
}
else
{
lean_inc(v_toInfSet_189_);
lean_inc(v_toSupSet_188_);
lean_inc(v_toLattice_187_);
lean_dec(v___x_186_);
v___x_191_ = lean_box(0);
v_isShared_192_ = v_isSharedCheck_207_;
goto v_resetjp_190_;
}
v_resetjp_190_:
{
lean_object* v_toSemilatticeSup_193_; lean_object* v___x_195_; uint8_t v_isShared_196_; uint8_t v_isSharedCheck_205_; 
v_toSemilatticeSup_193_ = lean_ctor_get(v_toLattice_187_, 0);
v_isSharedCheck_205_ = !lean_is_exclusive(v_toLattice_187_);
if (v_isSharedCheck_205_ == 0)
{
lean_object* v_unused_206_; 
v_unused_206_ = lean_ctor_get(v_toLattice_187_, 1);
lean_dec(v_unused_206_);
v___x_195_ = v_toLattice_187_;
v_isShared_196_ = v_isSharedCheck_205_;
goto v_resetjp_194_;
}
else
{
lean_inc(v_toSemilatticeSup_193_);
lean_dec(v_toLattice_187_);
v___x_195_ = lean_box(0);
v_isShared_196_ = v_isSharedCheck_205_;
goto v_resetjp_194_;
}
v_resetjp_194_:
{
lean_object* v___f_197_; lean_object* v___x_199_; 
v___f_197_ = ((lean_object*)(lp_mathlib_Subgroup_instCompleteLattice___redArg___closed__0));
if (v_isShared_196_ == 0)
{
lean_ctor_set(v___x_195_, 1, v___f_197_);
v___x_199_ = v___x_195_;
goto v_reusejp_198_;
}
else
{
lean_object* v_reuseFailAlloc_204_; 
v_reuseFailAlloc_204_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_204_, 0, v_toSemilatticeSup_193_);
lean_ctor_set(v_reuseFailAlloc_204_, 1, v___f_197_);
v___x_199_ = v_reuseFailAlloc_204_;
goto v_reusejp_198_;
}
v_reusejp_198_:
{
lean_object* v___x_200_; lean_object* v___x_202_; 
v___x_200_ = ((lean_object*)(lp_mathlib_Subgroup_instCompleteLattice___redArg___closed__1));
if (v_isShared_192_ == 0)
{
lean_ctor_set(v___x_191_, 3, v___x_200_);
lean_ctor_set(v___x_191_, 0, v___x_199_);
v___x_202_ = v___x_191_;
goto v_reusejp_201_;
}
else
{
lean_object* v_reuseFailAlloc_203_; 
v_reuseFailAlloc_203_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_203_, 0, v___x_199_);
lean_ctor_set(v_reuseFailAlloc_203_, 1, v_toSupSet_188_);
lean_ctor_set(v_reuseFailAlloc_203_, 2, v_toInfSet_189_);
lean_ctor_set(v_reuseFailAlloc_203_, 3, v___x_200_);
v___x_202_ = v_reuseFailAlloc_203_;
goto v_reusejp_201_;
}
v_reusejp_201_:
{
return v___x_202_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instCompleteLattice___redArg___boxed(lean_object* v_inst_209_){
_start:
{
lean_object* v_res_210_; 
v_res_210_ = lp_mathlib_Subgroup_instCompleteLattice___redArg(v_inst_209_);
lean_dec_ref(v_inst_209_);
return v_res_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instCompleteLattice(lean_object* v_G_211_, lean_object* v_inst_212_){
_start:
{
lean_object* v___x_213_; 
v___x_213_ = lp_mathlib_Subgroup_instCompleteLattice___redArg(v_inst_212_);
return v___x_213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instCompleteLattice___boxed(lean_object* v_G_214_, lean_object* v_inst_215_){
_start:
{
lean_object* v_res_216_; 
v_res_216_ = lp_mathlib_Subgroup_instCompleteLattice(v_G_214_, v_inst_215_);
lean_dec_ref(v_inst_215_);
return v_res_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instCompleteLattice___redArg___lam__0(lean_object* v_x1_217_, lean_object* v_x2_218_){
_start:
{
lean_object* v___x_219_; 
v___x_219_ = lean_box(0);
return v___x_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instCompleteLattice___redArg(lean_object* v_inst_223_){
_start:
{
lean_object* v___x_224_; lean_object* v___f_225_; lean_object* v___x_226_; lean_object* v_toLattice_227_; lean_object* v_toSupSet_228_; lean_object* v_toInfSet_229_; lean_object* v___x_231_; uint8_t v_isShared_232_; uint8_t v_isSharedCheck_247_; 
v___x_224_ = lp_mathlib_AddSubgroup_instPartialOrder(lean_box(0), v_inst_223_);
v___f_225_ = ((lean_object*)(lp_mathlib_AddSubgroup_instInfSet___closed__0));
v___x_226_ = lp_mathlib_completeLatticeOfInf___redArg(v___x_224_, v___f_225_);
v_toLattice_227_ = lean_ctor_get(v___x_226_, 0);
v_toSupSet_228_ = lean_ctor_get(v___x_226_, 1);
v_toInfSet_229_ = lean_ctor_get(v___x_226_, 2);
v_isSharedCheck_247_ = !lean_is_exclusive(v___x_226_);
if (v_isSharedCheck_247_ == 0)
{
lean_object* v_unused_248_; 
v_unused_248_ = lean_ctor_get(v___x_226_, 3);
lean_dec(v_unused_248_);
v___x_231_ = v___x_226_;
v_isShared_232_ = v_isSharedCheck_247_;
goto v_resetjp_230_;
}
else
{
lean_inc(v_toInfSet_229_);
lean_inc(v_toSupSet_228_);
lean_inc(v_toLattice_227_);
lean_dec(v___x_226_);
v___x_231_ = lean_box(0);
v_isShared_232_ = v_isSharedCheck_247_;
goto v_resetjp_230_;
}
v_resetjp_230_:
{
lean_object* v_toSemilatticeSup_233_; lean_object* v___x_235_; uint8_t v_isShared_236_; uint8_t v_isSharedCheck_245_; 
v_toSemilatticeSup_233_ = lean_ctor_get(v_toLattice_227_, 0);
v_isSharedCheck_245_ = !lean_is_exclusive(v_toLattice_227_);
if (v_isSharedCheck_245_ == 0)
{
lean_object* v_unused_246_; 
v_unused_246_ = lean_ctor_get(v_toLattice_227_, 1);
lean_dec(v_unused_246_);
v___x_235_ = v_toLattice_227_;
v_isShared_236_ = v_isSharedCheck_245_;
goto v_resetjp_234_;
}
else
{
lean_inc(v_toSemilatticeSup_233_);
lean_dec(v_toLattice_227_);
v___x_235_ = lean_box(0);
v_isShared_236_ = v_isSharedCheck_245_;
goto v_resetjp_234_;
}
v_resetjp_234_:
{
lean_object* v___f_237_; lean_object* v___x_239_; 
v___f_237_ = ((lean_object*)(lp_mathlib_AddSubgroup_instCompleteLattice___redArg___closed__0));
if (v_isShared_236_ == 0)
{
lean_ctor_set(v___x_235_, 1, v___f_237_);
v___x_239_ = v___x_235_;
goto v_reusejp_238_;
}
else
{
lean_object* v_reuseFailAlloc_244_; 
v_reuseFailAlloc_244_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_244_, 0, v_toSemilatticeSup_233_);
lean_ctor_set(v_reuseFailAlloc_244_, 1, v___f_237_);
v___x_239_ = v_reuseFailAlloc_244_;
goto v_reusejp_238_;
}
v_reusejp_238_:
{
lean_object* v___x_240_; lean_object* v___x_242_; 
v___x_240_ = ((lean_object*)(lp_mathlib_AddSubgroup_instCompleteLattice___redArg___closed__1));
if (v_isShared_232_ == 0)
{
lean_ctor_set(v___x_231_, 3, v___x_240_);
lean_ctor_set(v___x_231_, 0, v___x_239_);
v___x_242_ = v___x_231_;
goto v_reusejp_241_;
}
else
{
lean_object* v_reuseFailAlloc_243_; 
v_reuseFailAlloc_243_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_243_, 0, v___x_239_);
lean_ctor_set(v_reuseFailAlloc_243_, 1, v_toSupSet_228_);
lean_ctor_set(v_reuseFailAlloc_243_, 2, v_toInfSet_229_);
lean_ctor_set(v_reuseFailAlloc_243_, 3, v___x_240_);
v___x_242_ = v_reuseFailAlloc_243_;
goto v_reusejp_241_;
}
v_reusejp_241_:
{
return v___x_242_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instCompleteLattice___redArg___boxed(lean_object* v_inst_249_){
_start:
{
lean_object* v_res_250_; 
v_res_250_ = lp_mathlib_AddSubgroup_instCompleteLattice___redArg(v_inst_249_);
lean_dec_ref(v_inst_249_);
return v_res_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instCompleteLattice(lean_object* v_G_251_, lean_object* v_inst_252_){
_start:
{
lean_object* v___x_253_; 
v___x_253_ = lp_mathlib_AddSubgroup_instCompleteLattice___redArg(v_inst_252_);
return v___x_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instCompleteLattice___boxed(lean_object* v_G_254_, lean_object* v_inst_255_){
_start:
{
lean_object* v_res_256_; 
v_res_256_ = lp_mathlib_AddSubgroup_instCompleteLattice(v_G_254_, v_inst_255_);
lean_dec_ref(v_inst_255_);
return v_res_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instUniqueOfSubsingleton(lean_object* v_G_257_, lean_object* v_inst_258_, lean_object* v_inst_259_){
_start:
{
lean_object* v___x_260_; 
v___x_260_ = lean_box(0);
return v___x_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_instUniqueOfSubsingleton___boxed(lean_object* v_G_261_, lean_object* v_inst_262_, lean_object* v_inst_263_){
_start:
{
lean_object* v_res_264_; 
v_res_264_ = lp_mathlib_Subgroup_instUniqueOfSubsingleton(v_G_261_, v_inst_262_, v_inst_263_);
lean_dec_ref(v_inst_262_);
return v_res_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instUniqueOfSubsingleton(lean_object* v_G_265_, lean_object* v_inst_266_, lean_object* v_inst_267_){
_start:
{
lean_object* v___x_268_; 
v___x_268_ = lean_box(0);
return v___x_268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_instUniqueOfSubsingleton___boxed(lean_object* v_G_269_, lean_object* v_inst_270_, lean_object* v_inst_271_){
_start:
{
lean_object* v_res_272_; 
v_res_272_ = lp_mathlib_AddSubgroup_instUniqueOfSubsingleton(v_G_269_, v_inst_270_, v_inst_271_);
lean_dec_ref(v_inst_270_);
return v_res_272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_closure(lean_object* v_G_273_, lean_object* v_inst_274_, lean_object* v_k_275_){
_start:
{
lean_object* v___x_276_; 
v___x_276_ = lean_box(0);
return v___x_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_closure___boxed(lean_object* v_G_277_, lean_object* v_inst_278_, lean_object* v_k_279_){
_start:
{
lean_object* v_res_280_; 
v_res_280_ = lp_mathlib_Subgroup_closure(v_G_277_, v_inst_278_, v_k_279_);
lean_dec_ref(v_inst_278_);
return v_res_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_closure(lean_object* v_G_281_, lean_object* v_inst_282_, lean_object* v_k_283_){
_start:
{
lean_object* v___x_284_; 
v___x_284_ = lean_box(0);
return v___x_284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_closure___boxed(lean_object* v_G_285_, lean_object* v_inst_286_, lean_object* v_k_287_){
_start:
{
lean_object* v_res_288_; 
v_res_288_ = lp_mathlib_AddSubgroup_closure(v_G_285_, v_inst_286_, v_k_287_);
lean_dec_ref(v_inst_286_);
return v_res_288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_gi___lam__0(lean_object* v_s_289_, lean_object* v_x_290_){
_start:
{
lean_object* v___x_291_; 
v___x_291_ = lean_box(0);
return v___x_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_gi(lean_object* v_G_293_, lean_object* v_inst_294_){
_start:
{
lean_object* v___f_295_; 
v___f_295_ = ((lean_object*)(lp_mathlib_Subgroup_gi___closed__0));
return v___f_295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_gi___boxed(lean_object* v_G_296_, lean_object* v_inst_297_){
_start:
{
lean_object* v_res_298_; 
v_res_298_ = lp_mathlib_Subgroup_gi(v_G_296_, v_inst_297_);
lean_dec_ref(v_inst_297_);
return v_res_298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_gi___lam__0(lean_object* v_s_299_, lean_object* v_x_300_){
_start:
{
lean_object* v___x_301_; 
v___x_301_ = lean_box(0);
return v___x_301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_gi(lean_object* v_G_303_, lean_object* v_inst_304_){
_start:
{
lean_object* v___f_305_; 
v___f_305_ = ((lean_object*)(lp_mathlib_AddSubgroup_gi___closed__0));
return v___f_305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_gi___boxed(lean_object* v_G_306_, lean_object* v_inst_307_){
_start:
{
lean_object* v_res_308_; 
v_res_308_ = lp_mathlib_AddSubgroup_gi(v_G_306_, v_inst_307_);
lean_dec_ref(v_inst_307_);
return v_res_308_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Lattice(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Lattice(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Lattice(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Lattice(builtin);
}
#ifdef __cplusplus
}
#endif
