// Lean compiler output
// Module: Mathlib.Algebra.Module.Submodule.Lattice
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Subgroup.Lattice public import Mathlib.Algebra.Group.Submonoid.Membership public import Mathlib.Algebra.Group.Submonoid.BigOperators public import Mathlib.Algebra.Module.Submodule.Defs public import Mathlib.Algebra.Module.Equiv.Defs public import Mathlib.Algebra.Module.PUnit public import Mathlib.Data.Set.Subsingleton public import Mathlib.Data.Finset.Lattice.Fold public import Mathlib.Order.ConditionallyCompleteLattice.Basic
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
extern lean_object* lp_mathlib_Int_instCommRing;
lean_object* lp_mathlib_AddCommGroup_toIntModule___redArg(lean_object*);
lean_object* lp_mathlib_Submodule_toAddSubgroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Submodule_inhabited___redArg(lean_object*);
lean_object* lp_mathlib_Submodule_instPartialOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_inhabited_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_inhabited_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_uniqueBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_uniqueBot___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_uniqueBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_uniqueBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instOrderBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instOrderBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_botEquivPUnit___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_botEquivPUnit___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_botEquivPUnit___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_botEquivPUnit___redArg___lam__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Submodule_botEquivPUnit___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Submodule_botEquivPUnit___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_botEquivPUnit___redArg___closed__0 = (const lean_object*)&lp_mathlib_Submodule_botEquivPUnit___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_botEquivPUnit___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_botEquivPUnit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_botEquivPUnit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instOrderTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instOrderTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_topEquiv___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_topEquiv___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Submodule_topEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Submodule_topEquiv___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_topEquiv___closed__0 = (const lean_object*)&lp_mathlib_Submodule_topEquiv___closed__0_value;
static const lean_ctor_object lp_mathlib_Submodule_topEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Submodule_topEquiv___closed__0_value),((lean_object*)&lp_mathlib_Submodule_topEquiv___closed__0_value)}};
static const lean_object* lp_mathlib_Submodule_topEquiv___closed__1 = (const lean_object*)&lp_mathlib_Submodule_topEquiv___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_topEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_topEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instInfSet___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Submodule_instInfSet___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Submodule_instInfSet___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_instInfSet___closed__0 = (const lean_object*)&lp_mathlib_Submodule_instInfSet___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instInfSet(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instInfSet___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instMin___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Submodule_instMin___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Submodule_instMin___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_instMin___closed__0 = (const lean_object*)&lp_mathlib_Submodule_instMin___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instMin(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instMin___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_completeLattice___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_completeLattice___redArg___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Submodule_completeLattice___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Submodule_completeLattice___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_completeLattice___redArg___closed__0 = (const lean_object*)&lp_mathlib_Submodule_completeLattice___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_Submodule_completeLattice___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Submodule_completeLattice___redArg___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_completeLattice___redArg___closed__1 = (const lean_object*)&lp_mathlib_Submodule_completeLattice___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_Submodule_completeLattice___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Submodule_completeLattice___redArg___closed__2 = (const lean_object*)&lp_mathlib_Submodule_completeLattice___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_completeLattice___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_completeLattice___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_completeLattice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_completeLattice___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instUniqueOfSubsingleton(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instUniqueOfSubsingleton___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_unique_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_unique_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toNatSubmodule___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toNatSubmodule___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_AddSubmonoid_toNatSubmodule___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddSubmonoid_toNatSubmodule___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddSubmonoid_toNatSubmodule___closed__0 = (const lean_object*)&lp_mathlib_AddSubmonoid_toNatSubmodule___closed__0_value;
static const lean_closure_object lp_mathlib_AddSubmonoid_toNatSubmodule___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddSubmonoid_toNatSubmodule___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddSubmonoid_toNatSubmodule___closed__1 = (const lean_object*)&lp_mathlib_AddSubmonoid_toNatSubmodule___closed__1_value;
static const lean_ctor_object lp_mathlib_AddSubmonoid_toNatSubmodule___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_AddSubmonoid_toNatSubmodule___closed__1_value),((lean_object*)&lp_mathlib_AddSubmonoid_toNatSubmodule___closed__0_value)}};
static const lean_object* lp_mathlib_AddSubmonoid_toNatSubmodule___closed__2 = (const lean_object*)&lp_mathlib_AddSubmonoid_toNatSubmodule___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toNatSubmodule(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toNatSubmodule___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_toIntSubmodule___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_toIntSubmodule(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instBot(lean_object* v_R_1_, lean_object* v_M_2_, lean_object* v_inst_3_, lean_object* v_inst_4_, lean_object* v_inst_5_){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lean_box(0);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instBot___boxed(lean_object* v_R_7_, lean_object* v_M_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_inst_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_Submodule_instBot(v_R_7_, v_M_8_, v_inst_9_, v_inst_10_, v_inst_11_);
lean_dec(v_inst_11_);
lean_dec_ref(v_inst_10_);
lean_dec_ref(v_inst_9_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_inhabited_x27(lean_object* v_R_13_, lean_object* v_M_14_, lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_inst_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lean_box(0);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_inhabited_x27___boxed(lean_object* v_R_19_, lean_object* v_M_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_inst_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_Submodule_inhabited_x27(v_R_19_, v_M_20_, v_inst_21_, v_inst_22_, v_inst_23_);
lean_dec(v_inst_23_);
lean_dec_ref(v_inst_22_);
lean_dec_ref(v_inst_21_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_uniqueBot___redArg(lean_object* v_inst_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lp_mathlib_Submodule_inhabited___redArg(v_inst_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_uniqueBot___redArg___boxed(lean_object* v_inst_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_mathlib_Submodule_uniqueBot___redArg(v_inst_27_);
lean_dec_ref(v_inst_27_);
return v_res_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_uniqueBot(lean_object* v_R_29_, lean_object* v_M_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_inst_33_){
_start:
{
lean_object* v___x_34_; 
v___x_34_ = lp_mathlib_Submodule_inhabited___redArg(v_inst_32_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_uniqueBot___boxed(lean_object* v_R_35_, lean_object* v_M_36_, lean_object* v_inst_37_, lean_object* v_inst_38_, lean_object* v_inst_39_){
_start:
{
lean_object* v_res_40_; 
v_res_40_ = lp_mathlib_Submodule_uniqueBot(v_R_35_, v_M_36_, v_inst_37_, v_inst_38_, v_inst_39_);
lean_dec(v_inst_39_);
lean_dec_ref(v_inst_38_);
lean_dec_ref(v_inst_37_);
return v_res_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instOrderBot(lean_object* v_R_41_, lean_object* v_M_42_, lean_object* v_inst_43_, lean_object* v_inst_44_, lean_object* v_inst_45_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lean_box(0);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instOrderBot___boxed(lean_object* v_R_47_, lean_object* v_M_48_, lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_inst_51_){
_start:
{
lean_object* v_res_52_; 
v_res_52_ = lp_mathlib_Submodule_instOrderBot(v_R_47_, v_M_48_, v_inst_49_, v_inst_50_, v_inst_51_);
lean_dec(v_inst_51_);
lean_dec_ref(v_inst_50_);
lean_dec_ref(v_inst_49_);
return v_res_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_botEquivPUnit___redArg___lam__0(lean_object* v_x_53_){
_start:
{
lean_object* v___x_54_; 
v___x_54_ = lean_box(0);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_botEquivPUnit___redArg___lam__0___boxed(lean_object* v_x_55_){
_start:
{
lean_object* v_res_56_; 
v_res_56_ = lp_mathlib_Submodule_botEquivPUnit___redArg___lam__0(v_x_55_);
lean_dec(v_x_55_);
return v_res_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_botEquivPUnit___redArg___lam__1(lean_object* v_inst_57_, lean_object* v_x_58_){
_start:
{
lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v_toZero_61_; 
v___x_59_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_57_);
v___x_60_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_59_);
v_toZero_61_ = lean_ctor_get(v___x_60_, 0);
lean_inc(v_toZero_61_);
lean_dec_ref(v___x_60_);
return v_toZero_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_botEquivPUnit___redArg___lam__1___boxed(lean_object* v_inst_62_, lean_object* v_x_63_){
_start:
{
lean_object* v_res_64_; 
v_res_64_ = lp_mathlib_Submodule_botEquivPUnit___redArg___lam__1(v_inst_62_, v_x_63_);
lean_dec_ref(v_inst_62_);
return v_res_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_botEquivPUnit___redArg(lean_object* v_inst_66_){
_start:
{
lean_object* v___f_67_; lean_object* v___f_68_; lean_object* v___x_69_; 
v___f_67_ = ((lean_object*)(lp_mathlib_Submodule_botEquivPUnit___redArg___closed__0));
v___f_68_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_botEquivPUnit___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_68_, 0, v_inst_66_);
v___x_69_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_69_, 0, v___f_67_);
lean_ctor_set(v___x_69_, 1, v___f_68_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_botEquivPUnit(lean_object* v_R_70_, lean_object* v_M_71_, lean_object* v_inst_72_, lean_object* v_inst_73_, lean_object* v_inst_74_){
_start:
{
lean_object* v___x_75_; 
v___x_75_ = lp_mathlib_Submodule_botEquivPUnit___redArg(v_inst_73_);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_botEquivPUnit___boxed(lean_object* v_R_76_, lean_object* v_M_77_, lean_object* v_inst_78_, lean_object* v_inst_79_, lean_object* v_inst_80_){
_start:
{
lean_object* v_res_81_; 
v_res_81_ = lp_mathlib_Submodule_botEquivPUnit(v_R_76_, v_M_77_, v_inst_78_, v_inst_79_, v_inst_80_);
lean_dec(v_inst_80_);
lean_dec_ref(v_inst_78_);
return v_res_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instTop(lean_object* v_R_82_, lean_object* v_M_83_, lean_object* v_inst_84_, lean_object* v_inst_85_, lean_object* v_inst_86_){
_start:
{
lean_object* v___x_87_; 
v___x_87_ = lean_box(0);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instTop___boxed(lean_object* v_R_88_, lean_object* v_M_89_, lean_object* v_inst_90_, lean_object* v_inst_91_, lean_object* v_inst_92_){
_start:
{
lean_object* v_res_93_; 
v_res_93_ = lp_mathlib_Submodule_instTop(v_R_88_, v_M_89_, v_inst_90_, v_inst_91_, v_inst_92_);
lean_dec(v_inst_92_);
lean_dec_ref(v_inst_91_);
lean_dec_ref(v_inst_90_);
return v_res_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instOrderTop(lean_object* v_R_94_, lean_object* v_M_95_, lean_object* v_inst_96_, lean_object* v_inst_97_, lean_object* v_inst_98_){
_start:
{
lean_object* v___x_99_; 
v___x_99_ = lean_box(0);
return v___x_99_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instOrderTop___boxed(lean_object* v_R_100_, lean_object* v_M_101_, lean_object* v_inst_102_, lean_object* v_inst_103_, lean_object* v_inst_104_){
_start:
{
lean_object* v_res_105_; 
v_res_105_ = lp_mathlib_Submodule_instOrderTop(v_R_100_, v_M_101_, v_inst_102_, v_inst_103_, v_inst_104_);
lean_dec(v_inst_104_);
lean_dec_ref(v_inst_103_);
lean_dec_ref(v_inst_102_);
return v_res_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_topEquiv___lam__0(lean_object* v_x_106_){
_start:
{
lean_inc(v_x_106_);
return v_x_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_topEquiv___lam__0___boxed(lean_object* v_x_107_){
_start:
{
lean_object* v_res_108_; 
v_res_108_ = lp_mathlib_Submodule_topEquiv___lam__0(v_x_107_);
lean_dec(v_x_107_);
return v_res_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_topEquiv(lean_object* v_R_112_, lean_object* v_M_113_, lean_object* v_inst_114_, lean_object* v_inst_115_, lean_object* v_inst_116_){
_start:
{
lean_object* v___x_117_; 
v___x_117_ = ((lean_object*)(lp_mathlib_Submodule_topEquiv___closed__1));
return v___x_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_topEquiv___boxed(lean_object* v_R_118_, lean_object* v_M_119_, lean_object* v_inst_120_, lean_object* v_inst_121_, lean_object* v_inst_122_){
_start:
{
lean_object* v_res_123_; 
v_res_123_ = lp_mathlib_Submodule_topEquiv(v_R_118_, v_M_119_, v_inst_120_, v_inst_121_, v_inst_122_);
lean_dec(v_inst_122_);
lean_dec_ref(v_inst_121_);
lean_dec_ref(v_inst_120_);
return v_res_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instInfSet___lam__0(lean_object* v_S_124_){
_start:
{
lean_object* v___x_125_; 
v___x_125_ = lean_box(0);
return v___x_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instInfSet(lean_object* v_R_127_, lean_object* v_M_128_, lean_object* v_inst_129_, lean_object* v_inst_130_, lean_object* v_inst_131_){
_start:
{
lean_object* v___f_132_; 
v___f_132_ = ((lean_object*)(lp_mathlib_Submodule_instInfSet___closed__0));
return v___f_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instInfSet___boxed(lean_object* v_R_133_, lean_object* v_M_134_, lean_object* v_inst_135_, lean_object* v_inst_136_, lean_object* v_inst_137_){
_start:
{
lean_object* v_res_138_; 
v_res_138_ = lp_mathlib_Submodule_instInfSet(v_R_133_, v_M_134_, v_inst_135_, v_inst_136_, v_inst_137_);
lean_dec(v_inst_137_);
lean_dec_ref(v_inst_136_);
lean_dec_ref(v_inst_135_);
return v_res_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instMin___lam__0(lean_object* v_p_139_, lean_object* v_q_140_){
_start:
{
lean_object* v___x_141_; 
v___x_141_ = lean_box(0);
return v___x_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instMin(lean_object* v_R_143_, lean_object* v_M_144_, lean_object* v_inst_145_, lean_object* v_inst_146_, lean_object* v_inst_147_){
_start:
{
lean_object* v___f_148_; 
v___f_148_ = ((lean_object*)(lp_mathlib_Submodule_instMin___closed__0));
return v___f_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instMin___boxed(lean_object* v_R_149_, lean_object* v_M_150_, lean_object* v_inst_151_, lean_object* v_inst_152_, lean_object* v_inst_153_){
_start:
{
lean_object* v_res_154_; 
v_res_154_ = lp_mathlib_Submodule_instMin(v_R_149_, v_M_150_, v_inst_151_, v_inst_152_, v_inst_153_);
lean_dec(v_inst_153_);
lean_dec_ref(v_inst_152_);
lean_dec_ref(v_inst_151_);
return v_res_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_completeLattice___redArg___lam__0(lean_object* v_a_155_, lean_object* v_b_156_){
_start:
{
lean_object* v___x_157_; 
v___x_157_ = lean_box(0);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_completeLattice___redArg___lam__1(lean_object* v_x1_158_, lean_object* v_x2_159_){
_start:
{
lean_object* v___x_160_; 
v___x_160_ = lean_box(0);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_completeLattice___redArg(lean_object* v_inst_165_, lean_object* v_inst_166_, lean_object* v_inst_167_){
_start:
{
lean_object* v___f_168_; lean_object* v___f_169_; lean_object* v___f_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; 
v___f_168_ = ((lean_object*)(lp_mathlib_Submodule_completeLattice___redArg___closed__0));
v___f_169_ = ((lean_object*)(lp_mathlib_Submodule_completeLattice___redArg___closed__1));
v___f_170_ = ((lean_object*)(lp_mathlib_Submodule_instInfSet___closed__0));
v___x_171_ = lp_mathlib_Submodule_instPartialOrder(lean_box(0), lean_box(0), v_inst_165_, v_inst_166_, v_inst_167_);
v___x_172_ = ((lean_object*)(lp_mathlib_Submodule_completeLattice___redArg___closed__2));
v___x_173_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_173_, 0, v___x_171_);
lean_ctor_set(v___x_173_, 1, v___f_168_);
v___x_174_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_174_, 0, v___x_173_);
lean_ctor_set(v___x_174_, 1, v___f_169_);
v___x_175_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_175_, 0, v___x_174_);
lean_ctor_set(v___x_175_, 1, v___f_170_);
lean_ctor_set(v___x_175_, 2, v___f_170_);
lean_ctor_set(v___x_175_, 3, v___x_172_);
return v___x_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_completeLattice___redArg___boxed(lean_object* v_inst_176_, lean_object* v_inst_177_, lean_object* v_inst_178_){
_start:
{
lean_object* v_res_179_; 
v_res_179_ = lp_mathlib_Submodule_completeLattice___redArg(v_inst_176_, v_inst_177_, v_inst_178_);
lean_dec(v_inst_178_);
lean_dec_ref(v_inst_177_);
lean_dec_ref(v_inst_176_);
return v_res_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_completeLattice(lean_object* v_R_180_, lean_object* v_M_181_, lean_object* v_inst_182_, lean_object* v_inst_183_, lean_object* v_inst_184_){
_start:
{
lean_object* v___x_185_; 
v___x_185_ = lp_mathlib_Submodule_completeLattice___redArg(v_inst_182_, v_inst_183_, v_inst_184_);
return v___x_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_completeLattice___boxed(lean_object* v_R_186_, lean_object* v_M_187_, lean_object* v_inst_188_, lean_object* v_inst_189_, lean_object* v_inst_190_){
_start:
{
lean_object* v_res_191_; 
v_res_191_ = lp_mathlib_Submodule_completeLattice(v_R_186_, v_M_187_, v_inst_188_, v_inst_189_, v_inst_190_);
lean_dec(v_inst_190_);
lean_dec_ref(v_inst_189_);
lean_dec_ref(v_inst_188_);
return v_res_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instUniqueOfSubsingleton(lean_object* v_R_192_, lean_object* v_M_193_, lean_object* v_inst_194_, lean_object* v_inst_195_, lean_object* v_inst_196_, lean_object* v_inst_197_){
_start:
{
lean_object* v___x_198_; 
v___x_198_ = lean_box(0);
return v___x_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_instUniqueOfSubsingleton___boxed(lean_object* v_R_199_, lean_object* v_M_200_, lean_object* v_inst_201_, lean_object* v_inst_202_, lean_object* v_inst_203_, lean_object* v_inst_204_){
_start:
{
lean_object* v_res_205_; 
v_res_205_ = lp_mathlib_Submodule_instUniqueOfSubsingleton(v_R_199_, v_M_200_, v_inst_201_, v_inst_202_, v_inst_203_, v_inst_204_);
lean_dec(v_inst_203_);
lean_dec_ref(v_inst_202_);
lean_dec_ref(v_inst_201_);
return v_res_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_unique_x27(lean_object* v_R_206_, lean_object* v_M_207_, lean_object* v_inst_208_, lean_object* v_inst_209_, lean_object* v_inst_210_, lean_object* v_inst_211_){
_start:
{
lean_object* v___x_212_; 
v___x_212_ = lean_box(0);
return v___x_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_unique_x27___boxed(lean_object* v_R_213_, lean_object* v_M_214_, lean_object* v_inst_215_, lean_object* v_inst_216_, lean_object* v_inst_217_, lean_object* v_inst_218_){
_start:
{
lean_object* v_res_219_; 
v_res_219_ = lp_mathlib_Submodule_unique_x27(v_R_213_, v_M_214_, v_inst_215_, v_inst_216_, v_inst_217_, v_inst_218_);
lean_dec(v_inst_217_);
lean_dec_ref(v_inst_216_);
lean_dec_ref(v_inst_215_);
return v_res_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toNatSubmodule___lam__0(lean_object* v_self_220_){
_start:
{
return v_self_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toNatSubmodule___lam__1(lean_object* v_S_221_){
_start:
{
return v_S_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toNatSubmodule(lean_object* v_M_227_, lean_object* v_inst_228_){
_start:
{
lean_object* v___x_229_; 
v___x_229_ = ((lean_object*)(lp_mathlib_AddSubmonoid_toNatSubmodule___closed__2));
return v___x_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toNatSubmodule___boxed(lean_object* v_M_230_, lean_object* v_inst_231_){
_start:
{
lean_object* v_res_232_; 
v_res_232_ = lp_mathlib_AddSubmonoid_toNatSubmodule(v_M_230_, v_inst_231_);
lean_dec_ref(v_inst_231_);
return v_res_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_toIntSubmodule___redArg(lean_object* v_inst_233_){
_start:
{
lean_object* v___f_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; 
v___f_234_ = ((lean_object*)(lp_mathlib_AddSubmonoid_toNatSubmodule___closed__1));
v___x_235_ = lp_mathlib_Int_instCommRing;
lean_inc_ref(v_inst_233_);
v___x_236_ = lp_mathlib_AddCommGroup_toIntModule___redArg(v_inst_233_);
v___x_237_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_toAddSubgroup___boxed), 6, 5);
lean_closure_set(v___x_237_, 0, lean_box(0));
lean_closure_set(v___x_237_, 1, lean_box(0));
lean_closure_set(v___x_237_, 2, v___x_235_);
lean_closure_set(v___x_237_, 3, v_inst_233_);
lean_closure_set(v___x_237_, 4, v___x_236_);
v___x_238_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_238_, 0, v___f_234_);
lean_ctor_set(v___x_238_, 1, v___x_237_);
return v___x_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_toIntSubmodule(lean_object* v_M_239_, lean_object* v_inst_240_){
_start:
{
lean_object* v___x_241_; 
v___x_241_ = lp_mathlib_AddSubgroup_toIntSubmodule___redArg(v_inst_240_);
return v___x_241_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Lattice(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Membership(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_BigOperators(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_PUnit(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Subsingleton(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Lattice(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Membership(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_PUnit(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Subsingleton(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Lattice(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Lattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Membership(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_BigOperators(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_PUnit(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Subsingleton(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_Lattice(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Membership(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Submodule_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_PUnit(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Subsingleton(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Lattice_Fold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Module_Submodule_Lattice(builtin);
}
#ifdef __cplusplus
}
#endif
