// Lean compiler output
// Module: Mathlib.LinearAlgebra.Quotient.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Module.Equiv.Defs public import Mathlib.Algebra.Module.Submodule.Defs public import Mathlib.GroupTheory.QuotientGroup.Defs public import Mathlib.Logic.Small.Basic
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
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_NSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ZSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_Quot_congr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientRel(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientRel___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_hasQuotient(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_hasQuotient___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_mk___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_mk___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_mk(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_mk___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_instZeroQuotient___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_instZeroQuotient___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_instZeroQuotient(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_instZeroQuotient___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_instInhabitedQuotient___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_instInhabitedQuotient___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_instInhabitedQuotient(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_instInhabitedQuotient___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_instSMul_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_instSMul_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_instSMul_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_instSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_instSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_instSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addMonoid___aux__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addMonoid___aux__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addMonoid___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addMonoid___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addCommMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addCommGroup___aux__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addCommGroup___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addCommGroup___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addCommGroup___aux__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addCommGroup___aux__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addCommGroup___aux__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addCommGroup___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addCommGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_mulAction_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_mulAction_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_mulAction_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_mulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_mulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_mulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_smulZeroClass_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_smulZeroClass_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_smulZeroClass_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_smulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_smulZeroClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_smulZeroClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_distribSMul_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_distribSMul_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_distribSMul_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_distribSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_distribSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_distribSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_distribMulAction_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_distribMulAction_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_distribMulAction_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_distribMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_distribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_distribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_module_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_module_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_module_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_module___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_module(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_module___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mkQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mkQ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Submodule_quotEquivOfEq___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Submodule_quotEquivOfEq___closed__0;
static lean_once_cell_t lp_mathlib_Submodule_quotEquivOfEq___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Submodule_quotEquivOfEq___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotEquivOfEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotEquivOfEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientRel(lean_object* v_R_1_, lean_object* v_M_2_, lean_object* v_inst_3_, lean_object* v_inst_4_, lean_object* v_inst_5_, lean_object* v_p_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lean_box(0);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientRel___boxed(lean_object* v_R_8_, lean_object* v_M_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_p_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_Submodule_quotientRel(v_R_8_, v_M_9_, v_inst_10_, v_inst_11_, v_inst_12_, v_p_13_);
lean_dec(v_inst_12_);
lean_dec_ref(v_inst_11_);
lean_dec_ref(v_inst_10_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_hasQuotient(lean_object* v_R_15_, lean_object* v_M_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_inst_19_){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = lean_box(0);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_hasQuotient___boxed(lean_object* v_R_21_, lean_object* v_M_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_inst_25_){
_start:
{
lean_object* v_res_26_; 
v_res_26_ = lp_mathlib_Submodule_hasQuotient(v_R_21_, v_M_22_, v_inst_23_, v_inst_24_, v_inst_25_);
lean_dec(v_inst_25_);
lean_dec_ref(v_inst_24_);
lean_dec_ref(v_inst_23_);
return v_res_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_mk___redArg(lean_object* v_a_27_){
_start:
{
lean_inc(v_a_27_);
return v_a_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_mk___redArg___boxed(lean_object* v_a_28_){
_start:
{
lean_object* v_res_29_; 
v_res_29_ = lp_mathlib_Submodule_Quotient_mk___redArg(v_a_28_);
lean_dec(v_a_28_);
return v_res_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_mk(lean_object* v_R_30_, lean_object* v_M_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_inst_34_, lean_object* v_p_35_, lean_object* v_a_36_){
_start:
{
lean_inc(v_a_36_);
return v_a_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_mk___boxed(lean_object* v_R_37_, lean_object* v_M_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_inst_41_, lean_object* v_p_42_, lean_object* v_a_43_){
_start:
{
lean_object* v_res_44_; 
v_res_44_ = lp_mathlib_Submodule_Quotient_mk(v_R_37_, v_M_38_, v_inst_39_, v_inst_40_, v_inst_41_, v_p_42_, v_a_43_);
lean_dec(v_a_43_);
lean_dec(v_inst_41_);
lean_dec_ref(v_inst_40_);
lean_dec_ref(v_inst_39_);
return v_res_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_instZeroQuotient___redArg(lean_object* v_inst_45_){
_start:
{
lean_object* v___x_46_; lean_object* v_toZero_47_; 
v___x_46_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_45_);
v_toZero_47_ = lean_ctor_get(v___x_46_, 0);
lean_inc(v_toZero_47_);
lean_dec_ref(v___x_46_);
return v_toZero_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_instZeroQuotient___redArg___boxed(lean_object* v_inst_48_){
_start:
{
lean_object* v_res_49_; 
v_res_49_ = lp_mathlib_Submodule_Quotient_instZeroQuotient___redArg(v_inst_48_);
lean_dec_ref(v_inst_48_);
return v_res_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_instZeroQuotient(lean_object* v_R_50_, lean_object* v_M_51_, lean_object* v_inst_52_, lean_object* v_inst_53_, lean_object* v_inst_54_, lean_object* v_p_55_){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lp_mathlib_Submodule_Quotient_instZeroQuotient___redArg(v_inst_53_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_instZeroQuotient___boxed(lean_object* v_R_57_, lean_object* v_M_58_, lean_object* v_inst_59_, lean_object* v_inst_60_, lean_object* v_inst_61_, lean_object* v_p_62_){
_start:
{
lean_object* v_res_63_; 
v_res_63_ = lp_mathlib_Submodule_Quotient_instZeroQuotient(v_R_57_, v_M_58_, v_inst_59_, v_inst_60_, v_inst_61_, v_p_62_);
lean_dec(v_inst_61_);
lean_dec_ref(v_inst_60_);
lean_dec_ref(v_inst_59_);
return v_res_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_instInhabitedQuotient___redArg(lean_object* v_inst_64_){
_start:
{
lean_object* v___x_65_; lean_object* v_toZero_66_; 
v___x_65_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_64_);
v_toZero_66_ = lean_ctor_get(v___x_65_, 0);
lean_inc(v_toZero_66_);
lean_dec_ref(v___x_65_);
return v_toZero_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_instInhabitedQuotient___redArg___boxed(lean_object* v_inst_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_mathlib_Submodule_Quotient_instInhabitedQuotient___redArg(v_inst_67_);
lean_dec_ref(v_inst_67_);
return v_res_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_instInhabitedQuotient(lean_object* v_R_69_, lean_object* v_M_70_, lean_object* v_inst_71_, lean_object* v_inst_72_, lean_object* v_inst_73_, lean_object* v_p_74_){
_start:
{
lean_object* v___x_75_; 
v___x_75_ = lp_mathlib_Submodule_Quotient_instInhabitedQuotient___redArg(v_inst_72_);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_instInhabitedQuotient___boxed(lean_object* v_R_76_, lean_object* v_M_77_, lean_object* v_inst_78_, lean_object* v_inst_79_, lean_object* v_inst_80_, lean_object* v_p_81_){
_start:
{
lean_object* v_res_82_; 
v_res_82_ = lp_mathlib_Submodule_Quotient_instInhabitedQuotient(v_R_76_, v_M_77_, v_inst_78_, v_inst_79_, v_inst_80_, v_p_81_);
lean_dec(v_inst_80_);
lean_dec_ref(v_inst_79_);
lean_dec_ref(v_inst_78_);
return v_res_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0(lean_object* v_inst_83_, lean_object* v_a_84_, lean_object* v___y_85_){
_start:
{
lean_object* v___x_86_; 
v___x_86_ = lean_apply_2(v_inst_83_, v_a_84_, v___y_85_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_instSMul_x27___redArg(lean_object* v_inst_87_){
_start:
{
lean_object* v___f_88_; 
v___f_88_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_88_, 0, v_inst_87_);
return v___f_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_instSMul_x27(lean_object* v_R_89_, lean_object* v_M_90_, lean_object* v_inst_91_, lean_object* v_inst_92_, lean_object* v_inst_93_, lean_object* v_S_94_, lean_object* v_inst_95_, lean_object* v_inst_96_, lean_object* v_inst_97_, lean_object* v_P_98_){
_start:
{
lean_object* v___f_99_; 
v___f_99_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_99_, 0, v_inst_96_);
return v___f_99_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_instSMul_x27___boxed(lean_object* v_R_100_, lean_object* v_M_101_, lean_object* v_inst_102_, lean_object* v_inst_103_, lean_object* v_inst_104_, lean_object* v_S_105_, lean_object* v_inst_106_, lean_object* v_inst_107_, lean_object* v_inst_108_, lean_object* v_P_109_){
_start:
{
lean_object* v_res_110_; 
v_res_110_ = lp_mathlib_Submodule_Quotient_instSMul_x27(v_R_100_, v_M_101_, v_inst_102_, v_inst_103_, v_inst_104_, v_S_105_, v_inst_106_, v_inst_107_, v_inst_108_, v_P_109_);
lean_dec(v_inst_106_);
lean_dec(v_inst_104_);
lean_dec_ref(v_inst_103_);
lean_dec_ref(v_inst_102_);
return v_res_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_instSMul___redArg(lean_object* v_inst_111_){
_start:
{
lean_object* v___f_112_; 
v___f_112_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_112_, 0, v_inst_111_);
return v___f_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_instSMul(lean_object* v_R_113_, lean_object* v_M_114_, lean_object* v_inst_115_, lean_object* v_inst_116_, lean_object* v_inst_117_, lean_object* v_P_118_){
_start:
{
lean_object* v___f_119_; 
v___f_119_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_119_, 0, v_inst_117_);
return v___f_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_instSMul___boxed(lean_object* v_R_120_, lean_object* v_M_121_, lean_object* v_inst_122_, lean_object* v_inst_123_, lean_object* v_inst_124_, lean_object* v_P_125_){
_start:
{
lean_object* v_res_126_; 
v_res_126_ = lp_mathlib_Submodule_Quotient_instSMul(v_R_120_, v_M_121_, v_inst_122_, v_inst_123_, v_inst_124_, v_P_125_);
lean_dec_ref(v_inst_123_);
lean_dec_ref(v_inst_122_);
return v_res_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addMonoid___aux__1___redArg(lean_object* v_inst_127_, lean_object* v_a_128_, lean_object* v_a_129_){
_start:
{
lean_object* v_toAddMonoid_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v_toAdd_133_; lean_object* v___x_134_; 
v_toAddMonoid_130_ = lean_ctor_get(v_inst_127_, 0);
v___x_131_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_130_);
v___x_132_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_131_);
v_toAdd_133_ = lean_ctor_get(v___x_132_, 1);
lean_inc(v_toAdd_133_);
lean_dec_ref(v___x_132_);
v___x_134_ = lean_apply_2(v_toAdd_133_, v_a_128_, v_a_129_);
return v___x_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addMonoid___aux__1___redArg___boxed(lean_object* v_inst_135_, lean_object* v_a_136_, lean_object* v_a_137_){
_start:
{
lean_object* v_res_138_; 
v_res_138_ = lp_mathlib_Submodule_Quotient_addMonoid___aux__1___redArg(v_inst_135_, v_a_136_, v_a_137_);
lean_dec_ref(v_inst_135_);
return v_res_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addMonoid___aux__1(lean_object* v_R_139_, lean_object* v_M_140_, lean_object* v_inst_141_, lean_object* v_inst_142_, lean_object* v_inst_143_, lean_object* v_p_144_, lean_object* v_a_145_, lean_object* v_a_146_){
_start:
{
lean_object* v_toAddMonoid_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v_toAdd_150_; lean_object* v___x_151_; 
v_toAddMonoid_147_ = lean_ctor_get(v_inst_142_, 0);
v___x_148_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_147_);
v___x_149_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_148_);
v_toAdd_150_ = lean_ctor_get(v___x_149_, 1);
lean_inc(v_toAdd_150_);
lean_dec_ref(v___x_149_);
v___x_151_ = lean_apply_2(v_toAdd_150_, v_a_145_, v_a_146_);
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addMonoid___aux__1___boxed(lean_object* v_R_152_, lean_object* v_M_153_, lean_object* v_inst_154_, lean_object* v_inst_155_, lean_object* v_inst_156_, lean_object* v_p_157_, lean_object* v_a_158_, lean_object* v_a_159_){
_start:
{
lean_object* v_res_160_; 
v_res_160_ = lp_mathlib_Submodule_Quotient_addMonoid___aux__1(v_R_152_, v_M_153_, v_inst_154_, v_inst_155_, v_inst_156_, v_p_157_, v_a_158_, v_a_159_);
lean_dec(v_inst_156_);
lean_dec_ref(v_inst_155_);
lean_dec_ref(v_inst_154_);
return v_res_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addMonoid___redArg(lean_object* v_inst_161_, lean_object* v_inst_162_, lean_object* v_inst_163_, lean_object* v_p_164_){
_start:
{
lean_object* v_toAddMonoid_165_; lean_object* v_toNSMul_166_; lean_object* v___x_168_; uint8_t v_isShared_169_; uint8_t v_isSharedCheck_178_; 
v_toAddMonoid_165_ = lean_ctor_get(v_inst_162_, 0);
lean_inc_ref(v_toAddMonoid_165_);
v_toNSMul_166_ = lean_ctor_get(v_toAddMonoid_165_, 2);
v_isSharedCheck_178_ = !lean_is_exclusive(v_toAddMonoid_165_);
if (v_isSharedCheck_178_ == 0)
{
lean_object* v_unused_179_; lean_object* v_unused_180_; 
v_unused_179_ = lean_ctor_get(v_toAddMonoid_165_, 1);
lean_dec(v_unused_179_);
v_unused_180_ = lean_ctor_get(v_toAddMonoid_165_, 0);
lean_dec(v_unused_180_);
v___x_168_ = v_toAddMonoid_165_;
v_isShared_169_ = v_isSharedCheck_178_;
goto v_resetjp_167_;
}
else
{
lean_inc(v_toNSMul_166_);
lean_dec(v_toAddMonoid_165_);
v___x_168_ = lean_box(0);
v_isShared_169_ = v_isSharedCheck_178_;
goto v_resetjp_167_;
}
v_resetjp_167_:
{
lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___f_172_; lean_object* v___f_173_; lean_object* v___f_174_; lean_object* v___x_176_; 
v___x_170_ = lp_mathlib_Submodule_Quotient_instZeroQuotient___redArg(v_inst_162_);
v___x_171_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_addMonoid___aux__1___boxed), 8, 6);
lean_closure_set(v___x_171_, 0, lean_box(0));
lean_closure_set(v___x_171_, 1, lean_box(0));
lean_closure_set(v___x_171_, 2, v_inst_161_);
lean_closure_set(v___x_171_, 3, v_inst_162_);
lean_closure_set(v___x_171_, 4, v_inst_163_);
lean_closure_set(v___x_171_, 5, v_p_164_);
v___f_172_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_172_, 0, v_toNSMul_166_);
v___f_173_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_173_, 0, v___f_172_);
v___f_174_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_174_, 0, v___f_173_);
if (v_isShared_169_ == 0)
{
lean_ctor_set(v___x_168_, 2, v___f_174_);
lean_ctor_set(v___x_168_, 1, v___x_171_);
lean_ctor_set(v___x_168_, 0, v___x_170_);
v___x_176_ = v___x_168_;
goto v_reusejp_175_;
}
else
{
lean_object* v_reuseFailAlloc_177_; 
v_reuseFailAlloc_177_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_177_, 0, v___x_170_);
lean_ctor_set(v_reuseFailAlloc_177_, 1, v___x_171_);
lean_ctor_set(v_reuseFailAlloc_177_, 2, v___f_174_);
v___x_176_ = v_reuseFailAlloc_177_;
goto v_reusejp_175_;
}
v_reusejp_175_:
{
return v___x_176_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addMonoid(lean_object* v_R_181_, lean_object* v_M_182_, lean_object* v_inst_183_, lean_object* v_inst_184_, lean_object* v_inst_185_, lean_object* v_p_186_){
_start:
{
lean_object* v___x_187_; 
v___x_187_ = lp_mathlib_Submodule_Quotient_addMonoid___redArg(v_inst_183_, v_inst_184_, v_inst_185_, v_p_186_);
return v___x_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addCommMonoid___redArg(lean_object* v_inst_188_, lean_object* v_inst_189_, lean_object* v_inst_190_, lean_object* v_p_191_){
_start:
{
lean_object* v___x_192_; 
v___x_192_ = lp_mathlib_Submodule_Quotient_addMonoid___redArg(v_inst_188_, v_inst_189_, v_inst_190_, v_p_191_);
return v___x_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addCommMonoid(lean_object* v_R_193_, lean_object* v_M_194_, lean_object* v_inst_195_, lean_object* v_inst_196_, lean_object* v_inst_197_, lean_object* v_p_198_){
_start:
{
lean_object* v___x_199_; 
v___x_199_ = lp_mathlib_Submodule_Quotient_addMonoid___redArg(v_inst_195_, v_inst_196_, v_inst_197_, v_p_198_);
return v___x_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addCommGroup___aux__1___redArg(lean_object* v_inst_200_, lean_object* v_a_201_){
_start:
{
lean_object* v_toNeg_202_; lean_object* v___x_203_; 
v_toNeg_202_ = lean_ctor_get(v_inst_200_, 1);
lean_inc(v_toNeg_202_);
lean_dec_ref(v_inst_200_);
v___x_203_ = lean_apply_1(v_toNeg_202_, v_a_201_);
return v___x_203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addCommGroup___aux__1(lean_object* v_R_204_, lean_object* v_M_205_, lean_object* v_inst_206_, lean_object* v_inst_207_, lean_object* v_inst_208_, lean_object* v_p_209_, lean_object* v_a_210_){
_start:
{
lean_object* v_toNeg_211_; lean_object* v___x_212_; 
v_toNeg_211_ = lean_ctor_get(v_inst_207_, 1);
lean_inc(v_toNeg_211_);
lean_dec_ref(v_inst_207_);
v___x_212_ = lean_apply_1(v_toNeg_211_, v_a_210_);
return v___x_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addCommGroup___aux__1___boxed(lean_object* v_R_213_, lean_object* v_M_214_, lean_object* v_inst_215_, lean_object* v_inst_216_, lean_object* v_inst_217_, lean_object* v_p_218_, lean_object* v_a_219_){
_start:
{
lean_object* v_res_220_; 
v_res_220_ = lp_mathlib_Submodule_Quotient_addCommGroup___aux__1(v_R_213_, v_M_214_, v_inst_215_, v_inst_216_, v_inst_217_, v_p_218_, v_a_219_);
lean_dec(v_inst_217_);
lean_dec_ref(v_inst_215_);
return v_res_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addCommGroup___aux__3___redArg(lean_object* v_inst_221_, lean_object* v_a_222_, lean_object* v_a_223_){
_start:
{
lean_object* v_toSub_224_; lean_object* v___x_225_; 
v_toSub_224_ = lean_ctor_get(v_inst_221_, 2);
lean_inc(v_toSub_224_);
lean_dec_ref(v_inst_221_);
v___x_225_ = lean_apply_2(v_toSub_224_, v_a_222_, v_a_223_);
return v___x_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addCommGroup___aux__3(lean_object* v_R_226_, lean_object* v_M_227_, lean_object* v_inst_228_, lean_object* v_inst_229_, lean_object* v_inst_230_, lean_object* v_p_231_, lean_object* v_a_232_, lean_object* v_a_233_){
_start:
{
lean_object* v_toSub_234_; lean_object* v___x_235_; 
v_toSub_234_ = lean_ctor_get(v_inst_229_, 2);
lean_inc(v_toSub_234_);
lean_dec_ref(v_inst_229_);
v___x_235_ = lean_apply_2(v_toSub_234_, v_a_232_, v_a_233_);
return v___x_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addCommGroup___aux__3___boxed(lean_object* v_R_236_, lean_object* v_M_237_, lean_object* v_inst_238_, lean_object* v_inst_239_, lean_object* v_inst_240_, lean_object* v_p_241_, lean_object* v_a_242_, lean_object* v_a_243_){
_start:
{
lean_object* v_res_244_; 
v_res_244_ = lp_mathlib_Submodule_Quotient_addCommGroup___aux__3(v_R_236_, v_M_237_, v_inst_238_, v_inst_239_, v_inst_240_, v_p_241_, v_a_242_, v_a_243_);
lean_dec(v_inst_240_);
lean_dec_ref(v_inst_238_);
return v_res_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addCommGroup___redArg(lean_object* v_inst_245_, lean_object* v_inst_246_, lean_object* v_inst_247_, lean_object* v_p_248_){
_start:
{
lean_object* v___x_249_; lean_object* v_toZSMul_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___f_253_; lean_object* v___f_254_; lean_object* v___f_255_; lean_object* v___x_256_; 
lean_inc_n(v_inst_247_, 2);
lean_inc_ref_n(v_inst_246_, 2);
lean_inc_ref_n(v_inst_245_, 2);
v___x_249_ = lp_mathlib_Submodule_Quotient_addMonoid___redArg(v_inst_245_, v_inst_246_, v_inst_247_, v_p_248_);
v_toZSMul_250_ = lean_ctor_get(v_inst_246_, 3);
lean_inc(v_toZSMul_250_);
v___x_251_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_addCommGroup___aux__1___boxed), 7, 6);
lean_closure_set(v___x_251_, 0, lean_box(0));
lean_closure_set(v___x_251_, 1, lean_box(0));
lean_closure_set(v___x_251_, 2, v_inst_245_);
lean_closure_set(v___x_251_, 3, v_inst_246_);
lean_closure_set(v___x_251_, 4, v_inst_247_);
lean_closure_set(v___x_251_, 5, v_p_248_);
v___x_252_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_addCommGroup___aux__3___boxed), 8, 6);
lean_closure_set(v___x_252_, 0, lean_box(0));
lean_closure_set(v___x_252_, 1, lean_box(0));
lean_closure_set(v___x_252_, 2, v_inst_245_);
lean_closure_set(v___x_252_, 3, v_inst_246_);
lean_closure_set(v___x_252_, 4, v_inst_247_);
lean_closure_set(v___x_252_, 5, v_p_248_);
v___f_253_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_253_, 0, v_toZSMul_250_);
v___f_254_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_254_, 0, v___f_253_);
v___f_255_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_255_, 0, v___f_254_);
v___x_256_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_256_, 0, v___x_249_);
lean_ctor_set(v___x_256_, 1, v___x_251_);
lean_ctor_set(v___x_256_, 2, v___x_252_);
lean_ctor_set(v___x_256_, 3, v___f_255_);
return v___x_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_addCommGroup(lean_object* v_R_257_, lean_object* v_M_258_, lean_object* v_inst_259_, lean_object* v_inst_260_, lean_object* v_inst_261_, lean_object* v_p_262_){
_start:
{
lean_object* v___x_263_; 
v___x_263_ = lp_mathlib_Submodule_Quotient_addCommGroup___redArg(v_inst_259_, v_inst_260_, v_inst_261_, v_p_262_);
return v___x_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_mulAction_x27___redArg(lean_object* v_inst_264_){
_start:
{
lean_object* v___f_265_; 
v___f_265_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_265_, 0, v_inst_264_);
return v___f_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_mulAction_x27(lean_object* v_R_266_, lean_object* v_M_267_, lean_object* v_inst_268_, lean_object* v_inst_269_, lean_object* v_inst_270_, lean_object* v_S_271_, lean_object* v_inst_272_, lean_object* v_inst_273_, lean_object* v_inst_274_, lean_object* v_inst_275_, lean_object* v_P_276_){
_start:
{
lean_object* v___f_277_; 
v___f_277_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_277_, 0, v_inst_274_);
return v___f_277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_mulAction_x27___boxed(lean_object* v_R_278_, lean_object* v_M_279_, lean_object* v_inst_280_, lean_object* v_inst_281_, lean_object* v_inst_282_, lean_object* v_S_283_, lean_object* v_inst_284_, lean_object* v_inst_285_, lean_object* v_inst_286_, lean_object* v_inst_287_, lean_object* v_P_288_){
_start:
{
lean_object* v_res_289_; 
v_res_289_ = lp_mathlib_Submodule_Quotient_mulAction_x27(v_R_278_, v_M_279_, v_inst_280_, v_inst_281_, v_inst_282_, v_S_283_, v_inst_284_, v_inst_285_, v_inst_286_, v_inst_287_, v_P_288_);
lean_dec(v_inst_285_);
lean_dec_ref(v_inst_284_);
lean_dec(v_inst_282_);
lean_dec_ref(v_inst_281_);
lean_dec_ref(v_inst_280_);
return v_res_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_mulAction___redArg(lean_object* v_inst_290_){
_start:
{
lean_object* v___f_291_; 
v___f_291_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_291_, 0, v_inst_290_);
return v___f_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_mulAction(lean_object* v_R_292_, lean_object* v_M_293_, lean_object* v_inst_294_, lean_object* v_inst_295_, lean_object* v_inst_296_, lean_object* v_P_297_){
_start:
{
lean_object* v___f_298_; 
v___f_298_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_298_, 0, v_inst_296_);
return v___f_298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_mulAction___boxed(lean_object* v_R_299_, lean_object* v_M_300_, lean_object* v_inst_301_, lean_object* v_inst_302_, lean_object* v_inst_303_, lean_object* v_P_304_){
_start:
{
lean_object* v_res_305_; 
v_res_305_ = lp_mathlib_Submodule_Quotient_mulAction(v_R_299_, v_M_300_, v_inst_301_, v_inst_302_, v_inst_303_, v_P_304_);
lean_dec_ref(v_inst_302_);
lean_dec_ref(v_inst_301_);
return v_res_305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_smulZeroClass_x27___redArg(lean_object* v_inst_306_){
_start:
{
lean_object* v___f_307_; 
v___f_307_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_307_, 0, v_inst_306_);
return v___f_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_smulZeroClass_x27(lean_object* v_R_308_, lean_object* v_M_309_, lean_object* v_inst_310_, lean_object* v_inst_311_, lean_object* v_inst_312_, lean_object* v_S_313_, lean_object* v_inst_314_, lean_object* v_inst_315_, lean_object* v_inst_316_, lean_object* v_P_317_){
_start:
{
lean_object* v___f_318_; 
v___f_318_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_318_, 0, v_inst_315_);
return v___f_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_smulZeroClass_x27___boxed(lean_object* v_R_319_, lean_object* v_M_320_, lean_object* v_inst_321_, lean_object* v_inst_322_, lean_object* v_inst_323_, lean_object* v_S_324_, lean_object* v_inst_325_, lean_object* v_inst_326_, lean_object* v_inst_327_, lean_object* v_P_328_){
_start:
{
lean_object* v_res_329_; 
v_res_329_ = lp_mathlib_Submodule_Quotient_smulZeroClass_x27(v_R_319_, v_M_320_, v_inst_321_, v_inst_322_, v_inst_323_, v_S_324_, v_inst_325_, v_inst_326_, v_inst_327_, v_P_328_);
lean_dec(v_inst_325_);
lean_dec(v_inst_323_);
lean_dec_ref(v_inst_322_);
lean_dec_ref(v_inst_321_);
return v_res_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_smulZeroClass___redArg(lean_object* v_inst_330_){
_start:
{
lean_object* v___f_331_; 
v___f_331_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_331_, 0, v_inst_330_);
return v___f_331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_smulZeroClass(lean_object* v_R_332_, lean_object* v_M_333_, lean_object* v_inst_334_, lean_object* v_inst_335_, lean_object* v_inst_336_, lean_object* v_P_337_){
_start:
{
lean_object* v___f_338_; 
v___f_338_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_338_, 0, v_inst_336_);
return v___f_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_smulZeroClass___boxed(lean_object* v_R_339_, lean_object* v_M_340_, lean_object* v_inst_341_, lean_object* v_inst_342_, lean_object* v_inst_343_, lean_object* v_P_344_){
_start:
{
lean_object* v_res_345_; 
v_res_345_ = lp_mathlib_Submodule_Quotient_smulZeroClass(v_R_339_, v_M_340_, v_inst_341_, v_inst_342_, v_inst_343_, v_P_344_);
lean_dec_ref(v_inst_342_);
lean_dec_ref(v_inst_341_);
return v_res_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_distribSMul_x27___redArg(lean_object* v_inst_346_){
_start:
{
lean_object* v___f_347_; 
v___f_347_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_347_, 0, v_inst_346_);
return v___f_347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_distribSMul_x27(lean_object* v_R_348_, lean_object* v_M_349_, lean_object* v_inst_350_, lean_object* v_inst_351_, lean_object* v_inst_352_, lean_object* v_S_353_, lean_object* v_inst_354_, lean_object* v_inst_355_, lean_object* v_inst_356_, lean_object* v_P_357_){
_start:
{
lean_object* v___f_358_; 
v___f_358_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_358_, 0, v_inst_355_);
return v___f_358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_distribSMul_x27___boxed(lean_object* v_R_359_, lean_object* v_M_360_, lean_object* v_inst_361_, lean_object* v_inst_362_, lean_object* v_inst_363_, lean_object* v_S_364_, lean_object* v_inst_365_, lean_object* v_inst_366_, lean_object* v_inst_367_, lean_object* v_P_368_){
_start:
{
lean_object* v_res_369_; 
v_res_369_ = lp_mathlib_Submodule_Quotient_distribSMul_x27(v_R_359_, v_M_360_, v_inst_361_, v_inst_362_, v_inst_363_, v_S_364_, v_inst_365_, v_inst_366_, v_inst_367_, v_P_368_);
lean_dec(v_inst_365_);
lean_dec(v_inst_363_);
lean_dec_ref(v_inst_362_);
lean_dec_ref(v_inst_361_);
return v_res_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_distribSMul___redArg(lean_object* v_inst_370_){
_start:
{
lean_object* v___f_371_; 
v___f_371_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_371_, 0, v_inst_370_);
return v___f_371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_distribSMul(lean_object* v_R_372_, lean_object* v_M_373_, lean_object* v_inst_374_, lean_object* v_inst_375_, lean_object* v_inst_376_, lean_object* v_P_377_){
_start:
{
lean_object* v___f_378_; 
v___f_378_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_378_, 0, v_inst_376_);
return v___f_378_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_distribSMul___boxed(lean_object* v_R_379_, lean_object* v_M_380_, lean_object* v_inst_381_, lean_object* v_inst_382_, lean_object* v_inst_383_, lean_object* v_P_384_){
_start:
{
lean_object* v_res_385_; 
v_res_385_ = lp_mathlib_Submodule_Quotient_distribSMul(v_R_379_, v_M_380_, v_inst_381_, v_inst_382_, v_inst_383_, v_P_384_);
lean_dec_ref(v_inst_382_);
lean_dec_ref(v_inst_381_);
return v_res_385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_distribMulAction_x27___redArg(lean_object* v_inst_386_){
_start:
{
lean_object* v___f_387_; 
v___f_387_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_387_, 0, v_inst_386_);
return v___f_387_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_distribMulAction_x27(lean_object* v_R_388_, lean_object* v_M_389_, lean_object* v_inst_390_, lean_object* v_inst_391_, lean_object* v_inst_392_, lean_object* v_S_393_, lean_object* v_inst_394_, lean_object* v_inst_395_, lean_object* v_inst_396_, lean_object* v_inst_397_, lean_object* v_P_398_){
_start:
{
lean_object* v___f_399_; 
v___f_399_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_399_, 0, v_inst_396_);
return v___f_399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_distribMulAction_x27___boxed(lean_object* v_R_400_, lean_object* v_M_401_, lean_object* v_inst_402_, lean_object* v_inst_403_, lean_object* v_inst_404_, lean_object* v_S_405_, lean_object* v_inst_406_, lean_object* v_inst_407_, lean_object* v_inst_408_, lean_object* v_inst_409_, lean_object* v_P_410_){
_start:
{
lean_object* v_res_411_; 
v_res_411_ = lp_mathlib_Submodule_Quotient_distribMulAction_x27(v_R_400_, v_M_401_, v_inst_402_, v_inst_403_, v_inst_404_, v_S_405_, v_inst_406_, v_inst_407_, v_inst_408_, v_inst_409_, v_P_410_);
lean_dec(v_inst_407_);
lean_dec_ref(v_inst_406_);
lean_dec(v_inst_404_);
lean_dec_ref(v_inst_403_);
lean_dec_ref(v_inst_402_);
return v_res_411_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_distribMulAction___redArg(lean_object* v_inst_412_){
_start:
{
lean_object* v___f_413_; 
v___f_413_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_413_, 0, v_inst_412_);
return v___f_413_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_distribMulAction(lean_object* v_R_414_, lean_object* v_M_415_, lean_object* v_inst_416_, lean_object* v_inst_417_, lean_object* v_inst_418_, lean_object* v_P_419_){
_start:
{
lean_object* v___f_420_; 
v___f_420_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_420_, 0, v_inst_418_);
return v___f_420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_distribMulAction___boxed(lean_object* v_R_421_, lean_object* v_M_422_, lean_object* v_inst_423_, lean_object* v_inst_424_, lean_object* v_inst_425_, lean_object* v_P_426_){
_start:
{
lean_object* v_res_427_; 
v_res_427_ = lp_mathlib_Submodule_Quotient_distribMulAction(v_R_421_, v_M_422_, v_inst_423_, v_inst_424_, v_inst_425_, v_P_426_);
lean_dec_ref(v_inst_424_);
lean_dec_ref(v_inst_423_);
return v_res_427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_module_x27___redArg(lean_object* v_inst_428_){
_start:
{
lean_object* v___f_429_; 
v___f_429_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_429_, 0, v_inst_428_);
return v___f_429_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_module_x27(lean_object* v_R_430_, lean_object* v_M_431_, lean_object* v_inst_432_, lean_object* v_inst_433_, lean_object* v_inst_434_, lean_object* v_S_435_, lean_object* v_inst_436_, lean_object* v_inst_437_, lean_object* v_inst_438_, lean_object* v_inst_439_, lean_object* v_P_440_){
_start:
{
lean_object* v___f_441_; 
v___f_441_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_441_, 0, v_inst_438_);
return v___f_441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_module_x27___boxed(lean_object* v_R_442_, lean_object* v_M_443_, lean_object* v_inst_444_, lean_object* v_inst_445_, lean_object* v_inst_446_, lean_object* v_S_447_, lean_object* v_inst_448_, lean_object* v_inst_449_, lean_object* v_inst_450_, lean_object* v_inst_451_, lean_object* v_P_452_){
_start:
{
lean_object* v_res_453_; 
v_res_453_ = lp_mathlib_Submodule_Quotient_module_x27(v_R_442_, v_M_443_, v_inst_444_, v_inst_445_, v_inst_446_, v_S_447_, v_inst_448_, v_inst_449_, v_inst_450_, v_inst_451_, v_P_452_);
lean_dec(v_inst_449_);
lean_dec_ref(v_inst_448_);
lean_dec(v_inst_446_);
lean_dec_ref(v_inst_445_);
lean_dec_ref(v_inst_444_);
return v_res_453_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_module___redArg(lean_object* v_inst_454_){
_start:
{
lean_object* v___f_455_; 
v___f_455_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_455_, 0, v_inst_454_);
return v___f_455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_module(lean_object* v_R_456_, lean_object* v_M_457_, lean_object* v_inst_458_, lean_object* v_inst_459_, lean_object* v_inst_460_, lean_object* v_P_461_){
_start:
{
lean_object* v___f_462_; 
v___f_462_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_462_, 0, v_inst_460_);
return v___f_462_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_Quotient_module___boxed(lean_object* v_R_463_, lean_object* v_M_464_, lean_object* v_inst_465_, lean_object* v_inst_466_, lean_object* v_inst_467_, lean_object* v_P_468_){
_start:
{
lean_object* v_res_469_; 
v_res_469_ = lp_mathlib_Submodule_Quotient_module(v_R_463_, v_M_464_, v_inst_465_, v_inst_466_, v_inst_467_, v_P_468_);
lean_dec_ref(v_inst_466_);
lean_dec_ref(v_inst_465_);
return v_res_469_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mkQ___redArg(lean_object* v_inst_470_, lean_object* v_inst_471_, lean_object* v_inst_472_, lean_object* v_p_473_){
_start:
{
lean_object* v___x_474_; 
v___x_474_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_mk___boxed), 7, 6);
lean_closure_set(v___x_474_, 0, lean_box(0));
lean_closure_set(v___x_474_, 1, lean_box(0));
lean_closure_set(v___x_474_, 2, v_inst_470_);
lean_closure_set(v___x_474_, 3, v_inst_471_);
lean_closure_set(v___x_474_, 4, v_inst_472_);
lean_closure_set(v___x_474_, 5, v_p_473_);
return v___x_474_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mkQ(lean_object* v_R_475_, lean_object* v_M_476_, lean_object* v_inst_477_, lean_object* v_inst_478_, lean_object* v_inst_479_, lean_object* v_p_480_){
_start:
{
lean_object* v___x_481_; 
v___x_481_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_mk___boxed), 7, 6);
lean_closure_set(v___x_481_, 0, lean_box(0));
lean_closure_set(v___x_481_, 1, lean_box(0));
lean_closure_set(v___x_481_, 2, v_inst_477_);
lean_closure_set(v___x_481_, 3, v_inst_478_);
lean_closure_set(v___x_481_, 4, v_inst_479_);
lean_closure_set(v___x_481_, 5, v_p_480_);
return v___x_481_;
}
}
static lean_object* _init_lp_mathlib_Submodule_quotEquivOfEq___closed__0(void){
_start:
{
lean_object* v___x_482_; 
v___x_482_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_482_;
}
}
static lean_object* _init_lp_mathlib_Submodule_quotEquivOfEq___closed__1(void){
_start:
{
lean_object* v___x_483_; lean_object* v___x_484_; 
v___x_483_ = lean_obj_once(&lp_mathlib_Submodule_quotEquivOfEq___closed__0, &lp_mathlib_Submodule_quotEquivOfEq___closed__0_once, _init_lp_mathlib_Submodule_quotEquivOfEq___closed__0);
v___x_484_ = lp_mathlib_Quot_congr___redArg(v___x_483_);
return v___x_484_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotEquivOfEq(lean_object* v_R_485_, lean_object* v_M_486_, lean_object* v_inst_487_, lean_object* v_inst_488_, lean_object* v_inst_489_, lean_object* v_p_490_, lean_object* v_p_x27_491_, lean_object* v_h_492_){
_start:
{
lean_object* v___x_493_; lean_object* v_toFun_494_; lean_object* v_invFun_495_; lean_object* v___x_496_; 
v___x_493_ = lean_obj_once(&lp_mathlib_Submodule_quotEquivOfEq___closed__1, &lp_mathlib_Submodule_quotEquivOfEq___closed__1_once, _init_lp_mathlib_Submodule_quotEquivOfEq___closed__1);
v_toFun_494_ = lean_ctor_get(v___x_493_, 0);
v_invFun_495_ = lean_ctor_get(v___x_493_, 1);
lean_inc(v_invFun_495_);
lean_inc(v_toFun_494_);
v___x_496_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_496_, 0, v_toFun_494_);
lean_ctor_set(v___x_496_, 1, v_invFun_495_);
return v___x_496_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotEquivOfEq___boxed(lean_object* v_R_497_, lean_object* v_M_498_, lean_object* v_inst_499_, lean_object* v_inst_500_, lean_object* v_inst_501_, lean_object* v_p_502_, lean_object* v_p_x27_503_, lean_object* v_h_504_){
_start:
{
lean_object* v_res_505_; 
v_res_505_ = lp_mathlib_Submodule_quotEquivOfEq(v_R_497_, v_M_498_, v_inst_499_, v_inst_500_, v_inst_501_, v_p_502_, v_p_x27_503_, v_h_504_);
lean_dec(v_inst_501_);
lean_dec_ref(v_inst_500_);
lean_dec_ref(v_inst_499_);
return v_res_505_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_QuotientGroup_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Small_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_QuotientGroup_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Small_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_QuotientGroup_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Small_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Submodule_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_QuotientGroup_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Small_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
