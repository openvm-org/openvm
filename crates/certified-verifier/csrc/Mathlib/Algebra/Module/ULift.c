// Lean compiler output
// Module: Mathlib.Algebra.Module.ULift
// Imports: public import Init public meta import Init public import Mathlib.Algebra.GroupWithZero.ULift public import Mathlib.Algebra.Ring.ULift public import Mathlib.Algebra.Module.Equiv.Defs public import Mathlib.Data.ULift
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
lean_object* lp_mathlib_ULift_smul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_smulLeft___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_smulLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_smulLeft(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_vaddLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_vaddLeft(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulAction(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addAction(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulAction_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulAction_x27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulAction_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addAction_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addAction_x27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_addAction_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_smulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_smulZeroClass(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_smulZeroClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_smulZeroClass_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_smulZeroClass_x27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_smulZeroClass_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_distribSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_distribSMul(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_distribSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_distribSMul_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_distribSMul_x27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_distribSMul_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_distribMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_distribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_distribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_distribMulAction_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_distribMulAction_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_distribMulAction_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulDistribMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulDistribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulDistribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulDistribMulAction_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulDistribMulAction_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulDistribMulAction_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_smulWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_smulWithZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_smulWithZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_smulWithZero_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_smulWithZero_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_smulWithZero_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulActionWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulActionWithZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulActionWithZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulActionWithZero_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulActionWithZero_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulActionWithZero_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_module___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_module(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_module___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_module_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_module_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_module_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_moduleEquiv___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_moduleEquiv___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_moduleEquiv___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_moduleEquiv___lam__1___boxed(lean_object*);
static const lean_closure_object lp_mathlib_ULift_moduleEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_ULift_moduleEquiv___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_ULift_moduleEquiv___closed__0 = (const lean_object*)&lp_mathlib_ULift_moduleEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_ULift_moduleEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_ULift_moduleEquiv___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_ULift_moduleEquiv___closed__1 = (const lean_object*)&lp_mathlib_ULift_moduleEquiv___closed__1_value;
static const lean_ctor_object lp_mathlib_ULift_moduleEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_ULift_moduleEquiv___closed__0_value),((lean_object*)&lp_mathlib_ULift_moduleEquiv___closed__1_value)}};
static const lean_object* lp_mathlib_ULift_moduleEquiv___closed__2 = (const lean_object*)&lp_mathlib_ULift_moduleEquiv___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_ULift_moduleEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_moduleEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_smulLeft___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_s_2_, lean_object* v_x_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_apply_2(v_inst_1_, v_s_2_, v_x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_smulLeft___redArg(lean_object* v_inst_5_){
_start:
{
lean_object* v___f_6_; 
v___f_6_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smulLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_6_, 0, v_inst_5_);
return v___f_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_smulLeft(lean_object* v_R_7_, lean_object* v_M_8_, lean_object* v_inst_9_){
_start:
{
lean_object* v___f_10_; 
v___f_10_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smulLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_10_, 0, v_inst_9_);
return v___f_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_vaddLeft___redArg(lean_object* v_inst_11_){
_start:
{
lean_object* v___f_12_; 
v___f_12_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smulLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_12_, 0, v_inst_11_);
return v___f_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_vaddLeft(lean_object* v_R_13_, lean_object* v_M_14_, lean_object* v_inst_15_){
_start:
{
lean_object* v___f_16_; 
v___f_16_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smulLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_16_, 0, v_inst_15_);
return v___f_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulAction___redArg(lean_object* v_inst_17_){
_start:
{
lean_object* v___f_18_; 
v___f_18_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smulLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_18_, 0, v_inst_17_);
return v___f_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulAction(lean_object* v_R_19_, lean_object* v_M_20_, lean_object* v_inst_21_, lean_object* v_inst_22_){
_start:
{
lean_object* v___f_23_; 
v___f_23_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smulLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_23_, 0, v_inst_22_);
return v___f_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulAction___boxed(lean_object* v_R_24_, lean_object* v_M_25_, lean_object* v_inst_26_, lean_object* v_inst_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_mathlib_ULift_mulAction(v_R_24_, v_M_25_, v_inst_26_, v_inst_27_);
lean_dec_ref(v_inst_26_);
return v_res_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addAction___redArg(lean_object* v_inst_29_){
_start:
{
lean_object* v___f_30_; 
v___f_30_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smulLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_30_, 0, v_inst_29_);
return v___f_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addAction(lean_object* v_R_31_, lean_object* v_M_32_, lean_object* v_inst_33_, lean_object* v_inst_34_){
_start:
{
lean_object* v___f_35_; 
v___f_35_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smulLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_35_, 0, v_inst_34_);
return v___f_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addAction___boxed(lean_object* v_R_36_, lean_object* v_M_37_, lean_object* v_inst_38_, lean_object* v_inst_39_){
_start:
{
lean_object* v_res_40_; 
v_res_40_ = lp_mathlib_ULift_addAction(v_R_36_, v_M_37_, v_inst_38_, v_inst_39_);
lean_dec_ref(v_inst_38_);
return v_res_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulAction_x27___redArg(lean_object* v_inst_41_){
_start:
{
lean_object* v___f_42_; 
v___f_42_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_42_, 0, v_inst_41_);
return v___f_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulAction_x27(lean_object* v_R_43_, lean_object* v_M_44_, lean_object* v_inst_45_, lean_object* v_inst_46_){
_start:
{
lean_object* v___f_47_; 
v___f_47_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_47_, 0, v_inst_46_);
return v___f_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulAction_x27___boxed(lean_object* v_R_48_, lean_object* v_M_49_, lean_object* v_inst_50_, lean_object* v_inst_51_){
_start:
{
lean_object* v_res_52_; 
v_res_52_ = lp_mathlib_ULift_mulAction_x27(v_R_48_, v_M_49_, v_inst_50_, v_inst_51_);
lean_dec_ref(v_inst_50_);
return v_res_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addAction_x27___redArg(lean_object* v_inst_53_){
_start:
{
lean_object* v___f_54_; 
v___f_54_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_54_, 0, v_inst_53_);
return v___f_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addAction_x27(lean_object* v_R_55_, lean_object* v_M_56_, lean_object* v_inst_57_, lean_object* v_inst_58_){
_start:
{
lean_object* v___f_59_; 
v___f_59_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_59_, 0, v_inst_58_);
return v___f_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_addAction_x27___boxed(lean_object* v_R_60_, lean_object* v_M_61_, lean_object* v_inst_62_, lean_object* v_inst_63_){
_start:
{
lean_object* v_res_64_; 
v_res_64_ = lp_mathlib_ULift_addAction_x27(v_R_60_, v_M_61_, v_inst_62_, v_inst_63_);
lean_dec_ref(v_inst_62_);
return v_res_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_smulZeroClass___redArg(lean_object* v_inst_65_){
_start:
{
lean_object* v___f_66_; 
v___f_66_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smulLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_66_, 0, v_inst_65_);
return v___f_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_smulZeroClass(lean_object* v_R_67_, lean_object* v_M_68_, lean_object* v_inst_69_, lean_object* v_inst_70_){
_start:
{
lean_object* v___f_71_; 
v___f_71_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smulLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_71_, 0, v_inst_70_);
return v___f_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_smulZeroClass___boxed(lean_object* v_R_72_, lean_object* v_M_73_, lean_object* v_inst_74_, lean_object* v_inst_75_){
_start:
{
lean_object* v_res_76_; 
v_res_76_ = lp_mathlib_ULift_smulZeroClass(v_R_72_, v_M_73_, v_inst_74_, v_inst_75_);
lean_dec(v_inst_74_);
return v_res_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_smulZeroClass_x27___redArg(lean_object* v_inst_77_){
_start:
{
lean_object* v___f_78_; 
v___f_78_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_78_, 0, v_inst_77_);
return v___f_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_smulZeroClass_x27(lean_object* v_R_79_, lean_object* v_M_80_, lean_object* v_inst_81_, lean_object* v_inst_82_){
_start:
{
lean_object* v___f_83_; 
v___f_83_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_83_, 0, v_inst_82_);
return v___f_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_smulZeroClass_x27___boxed(lean_object* v_R_84_, lean_object* v_M_85_, lean_object* v_inst_86_, lean_object* v_inst_87_){
_start:
{
lean_object* v_res_88_; 
v_res_88_ = lp_mathlib_ULift_smulZeroClass_x27(v_R_84_, v_M_85_, v_inst_86_, v_inst_87_);
lean_dec(v_inst_86_);
return v_res_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_distribSMul___redArg(lean_object* v_inst_89_){
_start:
{
lean_object* v___f_90_; 
v___f_90_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smulLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_90_, 0, v_inst_89_);
return v___f_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_distribSMul(lean_object* v_R_91_, lean_object* v_M_92_, lean_object* v_inst_93_, lean_object* v_inst_94_){
_start:
{
lean_object* v___f_95_; 
v___f_95_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smulLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_95_, 0, v_inst_94_);
return v___f_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_distribSMul___boxed(lean_object* v_R_96_, lean_object* v_M_97_, lean_object* v_inst_98_, lean_object* v_inst_99_){
_start:
{
lean_object* v_res_100_; 
v_res_100_ = lp_mathlib_ULift_distribSMul(v_R_96_, v_M_97_, v_inst_98_, v_inst_99_);
lean_dec_ref(v_inst_98_);
return v_res_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_distribSMul_x27___redArg(lean_object* v_inst_101_){
_start:
{
lean_object* v___f_102_; 
v___f_102_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_102_, 0, v_inst_101_);
return v___f_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_distribSMul_x27(lean_object* v_R_103_, lean_object* v_M_104_, lean_object* v_inst_105_, lean_object* v_inst_106_){
_start:
{
lean_object* v___f_107_; 
v___f_107_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_107_, 0, v_inst_106_);
return v___f_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_distribSMul_x27___boxed(lean_object* v_R_108_, lean_object* v_M_109_, lean_object* v_inst_110_, lean_object* v_inst_111_){
_start:
{
lean_object* v_res_112_; 
v_res_112_ = lp_mathlib_ULift_distribSMul_x27(v_R_108_, v_M_109_, v_inst_110_, v_inst_111_);
lean_dec_ref(v_inst_110_);
return v_res_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_distribMulAction___redArg(lean_object* v_inst_113_){
_start:
{
lean_object* v___f_114_; 
v___f_114_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smulLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_114_, 0, v_inst_113_);
return v___f_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_distribMulAction(lean_object* v_R_115_, lean_object* v_M_116_, lean_object* v_inst_117_, lean_object* v_inst_118_, lean_object* v_inst_119_){
_start:
{
lean_object* v___f_120_; 
v___f_120_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smulLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_120_, 0, v_inst_119_);
return v___f_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_distribMulAction___boxed(lean_object* v_R_121_, lean_object* v_M_122_, lean_object* v_inst_123_, lean_object* v_inst_124_, lean_object* v_inst_125_){
_start:
{
lean_object* v_res_126_; 
v_res_126_ = lp_mathlib_ULift_distribMulAction(v_R_121_, v_M_122_, v_inst_123_, v_inst_124_, v_inst_125_);
lean_dec_ref(v_inst_124_);
lean_dec_ref(v_inst_123_);
return v_res_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_distribMulAction_x27___redArg(lean_object* v_inst_127_){
_start:
{
lean_object* v___f_128_; 
v___f_128_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_128_, 0, v_inst_127_);
return v___f_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_distribMulAction_x27(lean_object* v_R_129_, lean_object* v_M_130_, lean_object* v_inst_131_, lean_object* v_inst_132_, lean_object* v_inst_133_){
_start:
{
lean_object* v___f_134_; 
v___f_134_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_134_, 0, v_inst_133_);
return v___f_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_distribMulAction_x27___boxed(lean_object* v_R_135_, lean_object* v_M_136_, lean_object* v_inst_137_, lean_object* v_inst_138_, lean_object* v_inst_139_){
_start:
{
lean_object* v_res_140_; 
v_res_140_ = lp_mathlib_ULift_distribMulAction_x27(v_R_135_, v_M_136_, v_inst_137_, v_inst_138_, v_inst_139_);
lean_dec_ref(v_inst_138_);
lean_dec_ref(v_inst_137_);
return v_res_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulDistribMulAction___redArg(lean_object* v_inst_141_){
_start:
{
lean_object* v___f_142_; 
v___f_142_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smulLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_142_, 0, v_inst_141_);
return v___f_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulDistribMulAction(lean_object* v_R_143_, lean_object* v_M_144_, lean_object* v_inst_145_, lean_object* v_inst_146_, lean_object* v_inst_147_){
_start:
{
lean_object* v___f_148_; 
v___f_148_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smulLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_148_, 0, v_inst_147_);
return v___f_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulDistribMulAction___boxed(lean_object* v_R_149_, lean_object* v_M_150_, lean_object* v_inst_151_, lean_object* v_inst_152_, lean_object* v_inst_153_){
_start:
{
lean_object* v_res_154_; 
v_res_154_ = lp_mathlib_ULift_mulDistribMulAction(v_R_149_, v_M_150_, v_inst_151_, v_inst_152_, v_inst_153_);
lean_dec_ref(v_inst_152_);
lean_dec_ref(v_inst_151_);
return v_res_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulDistribMulAction_x27___redArg(lean_object* v_inst_155_){
_start:
{
lean_object* v___f_156_; 
v___f_156_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_156_, 0, v_inst_155_);
return v___f_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulDistribMulAction_x27(lean_object* v_R_157_, lean_object* v_M_158_, lean_object* v_inst_159_, lean_object* v_inst_160_, lean_object* v_inst_161_){
_start:
{
lean_object* v___f_162_; 
v___f_162_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_162_, 0, v_inst_161_);
return v___f_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulDistribMulAction_x27___boxed(lean_object* v_R_163_, lean_object* v_M_164_, lean_object* v_inst_165_, lean_object* v_inst_166_, lean_object* v_inst_167_){
_start:
{
lean_object* v_res_168_; 
v_res_168_ = lp_mathlib_ULift_mulDistribMulAction_x27(v_R_163_, v_M_164_, v_inst_165_, v_inst_166_, v_inst_167_);
lean_dec_ref(v_inst_166_);
lean_dec_ref(v_inst_165_);
return v_res_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_smulWithZero___redArg(lean_object* v_inst_169_){
_start:
{
lean_object* v___f_170_; 
v___f_170_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smulLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_170_, 0, v_inst_169_);
return v___f_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_smulWithZero(lean_object* v_R_171_, lean_object* v_M_172_, lean_object* v_inst_173_, lean_object* v_inst_174_, lean_object* v_inst_175_){
_start:
{
lean_object* v___f_176_; 
v___f_176_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smulLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_176_, 0, v_inst_175_);
return v___f_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_smulWithZero___boxed(lean_object* v_R_177_, lean_object* v_M_178_, lean_object* v_inst_179_, lean_object* v_inst_180_, lean_object* v_inst_181_){
_start:
{
lean_object* v_res_182_; 
v_res_182_ = lp_mathlib_ULift_smulWithZero(v_R_177_, v_M_178_, v_inst_179_, v_inst_180_, v_inst_181_);
lean_dec(v_inst_180_);
lean_dec(v_inst_179_);
return v_res_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_smulWithZero_x27___redArg(lean_object* v_inst_183_){
_start:
{
lean_object* v___f_184_; 
v___f_184_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_184_, 0, v_inst_183_);
return v___f_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_smulWithZero_x27(lean_object* v_R_185_, lean_object* v_M_186_, lean_object* v_inst_187_, lean_object* v_inst_188_, lean_object* v_inst_189_){
_start:
{
lean_object* v___f_190_; 
v___f_190_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_190_, 0, v_inst_189_);
return v___f_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_smulWithZero_x27___boxed(lean_object* v_R_191_, lean_object* v_M_192_, lean_object* v_inst_193_, lean_object* v_inst_194_, lean_object* v_inst_195_){
_start:
{
lean_object* v_res_196_; 
v_res_196_ = lp_mathlib_ULift_smulWithZero_x27(v_R_191_, v_M_192_, v_inst_193_, v_inst_194_, v_inst_195_);
lean_dec(v_inst_194_);
lean_dec(v_inst_193_);
return v_res_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulActionWithZero___redArg(lean_object* v_inst_197_){
_start:
{
lean_object* v___f_198_; 
v___f_198_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smulLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_198_, 0, v_inst_197_);
return v___f_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulActionWithZero(lean_object* v_R_199_, lean_object* v_M_200_, lean_object* v_inst_201_, lean_object* v_inst_202_, lean_object* v_inst_203_){
_start:
{
lean_object* v___f_204_; 
v___f_204_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smulLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_204_, 0, v_inst_203_);
return v___f_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulActionWithZero___boxed(lean_object* v_R_205_, lean_object* v_M_206_, lean_object* v_inst_207_, lean_object* v_inst_208_, lean_object* v_inst_209_){
_start:
{
lean_object* v_res_210_; 
v_res_210_ = lp_mathlib_ULift_mulActionWithZero(v_R_205_, v_M_206_, v_inst_207_, v_inst_208_, v_inst_209_);
lean_dec(v_inst_208_);
lean_dec_ref(v_inst_207_);
return v_res_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulActionWithZero_x27___redArg(lean_object* v_inst_211_){
_start:
{
lean_object* v___f_212_; 
v___f_212_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_212_, 0, v_inst_211_);
return v___f_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulActionWithZero_x27(lean_object* v_R_213_, lean_object* v_M_214_, lean_object* v_inst_215_, lean_object* v_inst_216_, lean_object* v_inst_217_){
_start:
{
lean_object* v___f_218_; 
v___f_218_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_218_, 0, v_inst_217_);
return v___f_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_mulActionWithZero_x27___boxed(lean_object* v_R_219_, lean_object* v_M_220_, lean_object* v_inst_221_, lean_object* v_inst_222_, lean_object* v_inst_223_){
_start:
{
lean_object* v_res_224_; 
v_res_224_ = lp_mathlib_ULift_mulActionWithZero_x27(v_R_219_, v_M_220_, v_inst_221_, v_inst_222_, v_inst_223_);
lean_dec(v_inst_222_);
lean_dec_ref(v_inst_221_);
return v_res_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_module___redArg(lean_object* v_inst_225_){
_start:
{
lean_object* v___f_226_; 
v___f_226_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smulLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_226_, 0, v_inst_225_);
return v___f_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_module(lean_object* v_R_227_, lean_object* v_M_228_, lean_object* v_inst_229_, lean_object* v_inst_230_, lean_object* v_inst_231_){
_start:
{
lean_object* v___f_232_; 
v___f_232_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smulLeft___redArg___lam__0), 3, 1);
lean_closure_set(v___f_232_, 0, v_inst_231_);
return v___f_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_module___boxed(lean_object* v_R_233_, lean_object* v_M_234_, lean_object* v_inst_235_, lean_object* v_inst_236_, lean_object* v_inst_237_){
_start:
{
lean_object* v_res_238_; 
v_res_238_ = lp_mathlib_ULift_module(v_R_233_, v_M_234_, v_inst_235_, v_inst_236_, v_inst_237_);
lean_dec_ref(v_inst_236_);
lean_dec_ref(v_inst_235_);
return v_res_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_module_x27___redArg(lean_object* v_inst_239_){
_start:
{
lean_object* v___f_240_; 
v___f_240_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_240_, 0, v_inst_239_);
return v___f_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_module_x27(lean_object* v_R_241_, lean_object* v_M_242_, lean_object* v_inst_243_, lean_object* v_inst_244_, lean_object* v_inst_245_){
_start:
{
lean_object* v___f_246_; 
v___f_246_ = lean_alloc_closure((void*)(lp_mathlib_ULift_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_246_, 0, v_inst_245_);
return v___f_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_module_x27___boxed(lean_object* v_R_247_, lean_object* v_M_248_, lean_object* v_inst_249_, lean_object* v_inst_250_, lean_object* v_inst_251_){
_start:
{
lean_object* v_res_252_; 
v_res_252_ = lp_mathlib_ULift_module_x27(v_R_247_, v_M_248_, v_inst_249_, v_inst_250_, v_inst_251_);
lean_dec_ref(v_inst_250_);
lean_dec_ref(v_inst_249_);
return v_res_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_moduleEquiv___lam__0(lean_object* v_self_253_){
_start:
{
lean_inc(v_self_253_);
return v_self_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_moduleEquiv___lam__0___boxed(lean_object* v_self_254_){
_start:
{
lean_object* v_res_255_; 
v_res_255_ = lp_mathlib_ULift_moduleEquiv___lam__0(v_self_254_);
lean_dec(v_self_254_);
return v_res_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_moduleEquiv___lam__1(lean_object* v_down_256_){
_start:
{
lean_inc(v_down_256_);
return v_down_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_moduleEquiv___lam__1___boxed(lean_object* v_down_257_){
_start:
{
lean_object* v_res_258_; 
v_res_258_ = lp_mathlib_ULift_moduleEquiv___lam__1(v_down_257_);
lean_dec(v_down_257_);
return v_res_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_moduleEquiv(lean_object* v_R_264_, lean_object* v_M_265_, lean_object* v_inst_266_, lean_object* v_inst_267_, lean_object* v_inst_268_){
_start:
{
lean_object* v___x_269_; 
v___x_269_ = ((lean_object*)(lp_mathlib_ULift_moduleEquiv___closed__2));
return v___x_269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_moduleEquiv___boxed(lean_object* v_R_270_, lean_object* v_M_271_, lean_object* v_inst_272_, lean_object* v_inst_273_, lean_object* v_inst_274_){
_start:
{
lean_object* v_res_275_; 
v_res_275_ = lp_mathlib_ULift_moduleEquiv(v_R_270_, v_M_271_, v_inst_272_, v_inst_273_, v_inst_274_);
lean_dec(v_inst_274_);
lean_dec_ref(v_inst_273_);
lean_dec_ref(v_inst_272_);
return v_res_275_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_ULift(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_ULift(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_ULift(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_ULift(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Module_ULift(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_ULift(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_ULift(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_ULift(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Module_ULift(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Module_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Module_ULift(builtin);
}
#ifdef __cplusplus
}
#endif
