// Lean compiler output
// Module: Mathlib.Algebra.Group.Action.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Commute.Defs public import Mathlib.Algebra.Notation.Defs public import Mathlib.Algebra.Opposites public import Mathlib.Logic.Function.Iterate public import Mathlib.Tactic.Spread
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
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instVAddOfAdd___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instVAddOfAdd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instVAddOfAdd(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mul_toSMulMulOpposite___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mul_toSMulMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mul_toSMulMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Add_toVAddAddOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Add_toVAddAddOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LibraryNote_bundled__maps__over__different__rings;
LEAN_EXPORT lean_object* lp_mathlib_SMul_comp_smul___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMul_comp_smul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_VAdd_comp_vadd___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_VAdd_comp_vadd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMul_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMul_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_VAdd_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_VAdd_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_mulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_mulAction___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_mulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_mulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addAction___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulAction___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addAction___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toMulAction___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toMulAction___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toMulAction(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toMulAction___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddAction___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddAction___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddAction(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddAction___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instVAddOfAdd___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_x_2_, lean_object* v_y_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_apply_2(v_inst_1_, v_x_2_, v_y_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instVAddOfAdd___redArg(lean_object* v_inst_5_){
_start:
{
lean_object* v___f_6_; 
v___f_6_ = lean_alloc_closure((void*)(lp_mathlib_instVAddOfAdd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_6_, 0, v_inst_5_);
return v___f_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instVAddOfAdd(lean_object* v_00_u03b1_7_, lean_object* v_inst_8_){
_start:
{
lean_object* v___f_9_; 
v___f_9_ = lean_alloc_closure((void*)(lp_mathlib_instVAddOfAdd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_9_, 0, v_inst_8_);
return v___f_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mul_toSMulMulOpposite___redArg___lam__0(lean_object* v_inst_10_, lean_object* v_a_11_, lean_object* v_b_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lean_apply_2(v_inst_10_, v_b_12_, v_a_11_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mul_toSMulMulOpposite___redArg(lean_object* v_inst_14_){
_start:
{
lean_object* v___f_15_; 
v___f_15_ = lean_alloc_closure((void*)(lp_mathlib_Mul_toSMulMulOpposite___redArg___lam__0), 3, 1);
lean_closure_set(v___f_15_, 0, v_inst_14_);
return v___f_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mul_toSMulMulOpposite(lean_object* v_00_u03b1_16_, lean_object* v_inst_17_){
_start:
{
lean_object* v___f_18_; 
v___f_18_ = lean_alloc_closure((void*)(lp_mathlib_Mul_toSMulMulOpposite___redArg___lam__0), 3, 1);
lean_closure_set(v___f_18_, 0, v_inst_17_);
return v___f_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Add_toVAddAddOpposite___redArg(lean_object* v_inst_19_){
_start:
{
lean_object* v___f_20_; 
v___f_20_ = lean_alloc_closure((void*)(lp_mathlib_Mul_toSMulMulOpposite___redArg___lam__0), 3, 1);
lean_closure_set(v___f_20_, 0, v_inst_19_);
return v___f_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Add_toVAddAddOpposite(lean_object* v_00_u03b1_21_, lean_object* v_inst_22_){
_start:
{
lean_object* v___f_23_; 
v___f_23_ = lean_alloc_closure((void*)(lp_mathlib_Mul_toSMulMulOpposite___redArg___lam__0), 3, 1);
lean_closure_set(v___f_23_, 0, v_inst_22_);
return v___f_23_;
}
}
static lean_object* _init_lp_mathlib_LibraryNote_bundled__maps__over__different__rings(void){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lean_box(0);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMul_comp_smul___redArg(lean_object* v_inst_25_, lean_object* v_g_26_, lean_object* v_n_27_, lean_object* v_a_28_){
_start:
{
lean_object* v___x_29_; lean_object* v___x_30_; 
v___x_29_ = lean_apply_1(v_g_26_, v_n_27_);
v___x_30_ = lean_apply_2(v_inst_25_, v___x_29_, v_a_28_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMul_comp_smul(lean_object* v_M_31_, lean_object* v_N_32_, lean_object* v_00_u03b1_33_, lean_object* v_inst_34_, lean_object* v_g_35_, lean_object* v_n_36_, lean_object* v_a_37_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = lp_mathlib_SMul_comp_smul___redArg(v_inst_34_, v_g_35_, v_n_36_, v_a_37_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_VAdd_comp_vadd___redArg(lean_object* v_inst_39_, lean_object* v_g_40_, lean_object* v_n_41_, lean_object* v_a_42_){
_start:
{
lean_object* v___x_43_; lean_object* v___x_44_; 
v___x_43_ = lean_apply_1(v_g_40_, v_n_41_);
v___x_44_ = lean_apply_2(v_inst_39_, v___x_43_, v_a_42_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_VAdd_comp_vadd(lean_object* v_M_45_, lean_object* v_N_46_, lean_object* v_00_u03b1_47_, lean_object* v_inst_48_, lean_object* v_g_49_, lean_object* v_n_50_, lean_object* v_a_51_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lp_mathlib_VAdd_comp_vadd___redArg(v_inst_48_, v_g_49_, v_n_50_, v_a_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMul_comp___redArg(lean_object* v_inst_53_, lean_object* v_g_54_){
_start:
{
lean_object* v___x_55_; 
v___x_55_ = lean_alloc_closure((void*)(lp_mathlib_SMul_comp_smul), 7, 5);
lean_closure_set(v___x_55_, 0, lean_box(0));
lean_closure_set(v___x_55_, 1, lean_box(0));
lean_closure_set(v___x_55_, 2, lean_box(0));
lean_closure_set(v___x_55_, 3, v_inst_53_);
lean_closure_set(v___x_55_, 4, v_g_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMul_comp(lean_object* v_M_56_, lean_object* v_N_57_, lean_object* v_00_u03b1_58_, lean_object* v_inst_59_, lean_object* v_g_60_){
_start:
{
lean_object* v___x_61_; 
v___x_61_ = lean_alloc_closure((void*)(lp_mathlib_SMul_comp_smul), 7, 5);
lean_closure_set(v___x_61_, 0, lean_box(0));
lean_closure_set(v___x_61_, 1, lean_box(0));
lean_closure_set(v___x_61_, 2, lean_box(0));
lean_closure_set(v___x_61_, 3, v_inst_59_);
lean_closure_set(v___x_61_, 4, v_g_60_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_VAdd_comp___redArg(lean_object* v_inst_62_, lean_object* v_g_63_){
_start:
{
lean_object* v___x_64_; 
v___x_64_ = lean_alloc_closure((void*)(lp_mathlib_VAdd_comp_vadd), 7, 5);
lean_closure_set(v___x_64_, 0, lean_box(0));
lean_closure_set(v___x_64_, 1, lean_box(0));
lean_closure_set(v___x_64_, 2, lean_box(0));
lean_closure_set(v___x_64_, 3, v_inst_62_);
lean_closure_set(v___x_64_, 4, v_g_63_);
return v___x_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_VAdd_comp(lean_object* v_M_65_, lean_object* v_N_66_, lean_object* v_00_u03b1_67_, lean_object* v_inst_68_, lean_object* v_g_69_){
_start:
{
lean_object* v___x_70_; 
v___x_70_ = lean_alloc_closure((void*)(lp_mathlib_VAdd_comp_vadd), 7, 5);
lean_closure_set(v___x_70_, 0, lean_box(0));
lean_closure_set(v___x_70_, 1, lean_box(0));
lean_closure_set(v___x_70_, 2, lean_box(0));
lean_closure_set(v___x_70_, 3, v_inst_68_);
lean_closure_set(v___x_70_, 4, v_g_69_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_mulAction___redArg(lean_object* v_inst_71_){
_start:
{
lean_inc(v_inst_71_);
return v_inst_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_mulAction___redArg___boxed(lean_object* v_inst_72_){
_start:
{
lean_object* v_res_73_; 
v_res_73_ = lp_mathlib_Function_Injective_mulAction___redArg(v_inst_72_);
lean_dec(v_inst_72_);
return v_res_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_mulAction(lean_object* v_M_74_, lean_object* v_00_u03b1_75_, lean_object* v_00_u03b2_76_, lean_object* v_inst_77_, lean_object* v_inst_78_, lean_object* v_inst_79_, lean_object* v_f_80_, lean_object* v_hf_81_, lean_object* v_smul_82_){
_start:
{
lean_inc(v_inst_79_);
return v_inst_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_mulAction___boxed(lean_object* v_M_83_, lean_object* v_00_u03b1_84_, lean_object* v_00_u03b2_85_, lean_object* v_inst_86_, lean_object* v_inst_87_, lean_object* v_inst_88_, lean_object* v_f_89_, lean_object* v_hf_90_, lean_object* v_smul_91_){
_start:
{
lean_object* v_res_92_; 
v_res_92_ = lp_mathlib_Function_Injective_mulAction(v_M_83_, v_00_u03b1_84_, v_00_u03b2_85_, v_inst_86_, v_inst_87_, v_inst_88_, v_f_89_, v_hf_90_, v_smul_91_);
lean_dec(v_f_89_);
lean_dec(v_inst_88_);
lean_dec(v_inst_87_);
lean_dec_ref(v_inst_86_);
return v_res_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addAction___redArg(lean_object* v_inst_93_){
_start:
{
lean_inc(v_inst_93_);
return v_inst_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addAction___redArg___boxed(lean_object* v_inst_94_){
_start:
{
lean_object* v_res_95_; 
v_res_95_ = lp_mathlib_Function_Injective_addAction___redArg(v_inst_94_);
lean_dec(v_inst_94_);
return v_res_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addAction(lean_object* v_M_96_, lean_object* v_00_u03b1_97_, lean_object* v_00_u03b2_98_, lean_object* v_inst_99_, lean_object* v_inst_100_, lean_object* v_inst_101_, lean_object* v_f_102_, lean_object* v_hf_103_, lean_object* v_smul_104_){
_start:
{
lean_inc(v_inst_101_);
return v_inst_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addAction___boxed(lean_object* v_M_105_, lean_object* v_00_u03b1_106_, lean_object* v_00_u03b2_107_, lean_object* v_inst_108_, lean_object* v_inst_109_, lean_object* v_inst_110_, lean_object* v_f_111_, lean_object* v_hf_112_, lean_object* v_smul_113_){
_start:
{
lean_object* v_res_114_; 
v_res_114_ = lp_mathlib_Function_Injective_addAction(v_M_105_, v_00_u03b1_106_, v_00_u03b2_107_, v_inst_108_, v_inst_109_, v_inst_110_, v_f_111_, v_hf_112_, v_smul_113_);
lean_dec(v_f_111_);
lean_dec(v_inst_110_);
lean_dec(v_inst_109_);
lean_dec_ref(v_inst_108_);
return v_res_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulAction___redArg(lean_object* v_inst_115_){
_start:
{
lean_inc(v_inst_115_);
return v_inst_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulAction___redArg___boxed(lean_object* v_inst_116_){
_start:
{
lean_object* v_res_117_; 
v_res_117_ = lp_mathlib_Function_Surjective_mulAction___redArg(v_inst_116_);
lean_dec(v_inst_116_);
return v_res_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulAction(lean_object* v_M_118_, lean_object* v_00_u03b1_119_, lean_object* v_00_u03b2_120_, lean_object* v_inst_121_, lean_object* v_inst_122_, lean_object* v_inst_123_, lean_object* v_f_124_, lean_object* v_hf_125_, lean_object* v_smul_126_){
_start:
{
lean_inc(v_inst_123_);
return v_inst_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulAction___boxed(lean_object* v_M_127_, lean_object* v_00_u03b1_128_, lean_object* v_00_u03b2_129_, lean_object* v_inst_130_, lean_object* v_inst_131_, lean_object* v_inst_132_, lean_object* v_f_133_, lean_object* v_hf_134_, lean_object* v_smul_135_){
_start:
{
lean_object* v_res_136_; 
v_res_136_ = lp_mathlib_Function_Surjective_mulAction(v_M_127_, v_00_u03b1_128_, v_00_u03b2_129_, v_inst_130_, v_inst_131_, v_inst_132_, v_f_133_, v_hf_134_, v_smul_135_);
lean_dec(v_f_133_);
lean_dec(v_inst_132_);
lean_dec(v_inst_131_);
lean_dec_ref(v_inst_130_);
return v_res_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addAction___redArg(lean_object* v_inst_137_){
_start:
{
lean_inc(v_inst_137_);
return v_inst_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addAction___redArg___boxed(lean_object* v_inst_138_){
_start:
{
lean_object* v_res_139_; 
v_res_139_ = lp_mathlib_Function_Surjective_addAction___redArg(v_inst_138_);
lean_dec(v_inst_138_);
return v_res_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addAction(lean_object* v_M_140_, lean_object* v_00_u03b1_141_, lean_object* v_00_u03b2_142_, lean_object* v_inst_143_, lean_object* v_inst_144_, lean_object* v_inst_145_, lean_object* v_f_146_, lean_object* v_hf_147_, lean_object* v_smul_148_){
_start:
{
lean_inc(v_inst_145_);
return v_inst_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addAction___boxed(lean_object* v_M_149_, lean_object* v_00_u03b1_150_, lean_object* v_00_u03b2_151_, lean_object* v_inst_152_, lean_object* v_inst_153_, lean_object* v_inst_154_, lean_object* v_f_155_, lean_object* v_hf_156_, lean_object* v_smul_157_){
_start:
{
lean_object* v_res_158_; 
v_res_158_ = lp_mathlib_Function_Surjective_addAction(v_M_149_, v_00_u03b1_150_, v_00_u03b2_151_, v_inst_152_, v_inst_153_, v_inst_154_, v_f_155_, v_hf_156_, v_smul_157_);
lean_dec(v_f_155_);
lean_dec(v_inst_154_);
lean_dec(v_inst_153_);
lean_dec_ref(v_inst_152_);
return v_res_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toMulAction___redArg___lam__0(lean_object* v_toMul_159_, lean_object* v_x1_160_, lean_object* v_x2_161_){
_start:
{
lean_object* v___x_162_; 
v___x_162_ = lean_apply_2(v_toMul_159_, v_x1_160_, v_x2_161_);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toMulAction___redArg(lean_object* v_inst_163_){
_start:
{
lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v_toMul_166_; lean_object* v___f_167_; 
v___x_164_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_163_);
v___x_165_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_164_);
v_toMul_166_ = lean_ctor_get(v___x_165_, 1);
lean_inc(v_toMul_166_);
lean_dec_ref(v___x_165_);
v___f_167_ = lean_alloc_closure((void*)(lp_mathlib_Monoid_toMulAction___redArg___lam__0), 3, 1);
lean_closure_set(v___f_167_, 0, v_toMul_166_);
return v___f_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toMulAction___redArg___boxed(lean_object* v_inst_168_){
_start:
{
lean_object* v_res_169_; 
v_res_169_ = lp_mathlib_Monoid_toMulAction___redArg(v_inst_168_);
lean_dec_ref(v_inst_168_);
return v_res_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toMulAction(lean_object* v_M_170_, lean_object* v_inst_171_){
_start:
{
lean_object* v___x_172_; 
v___x_172_ = lp_mathlib_Monoid_toMulAction___redArg(v_inst_171_);
return v___x_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Monoid_toMulAction___boxed(lean_object* v_M_173_, lean_object* v_inst_174_){
_start:
{
lean_object* v_res_175_; 
v_res_175_ = lp_mathlib_Monoid_toMulAction(v_M_173_, v_inst_174_);
lean_dec_ref(v_inst_174_);
return v_res_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddAction___redArg___lam__0(lean_object* v_toAdd_176_, lean_object* v_x1_177_, lean_object* v_x2_178_){
_start:
{
lean_object* v___x_179_; 
v___x_179_ = lean_apply_2(v_toAdd_176_, v_x1_177_, v_x2_178_);
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddAction___redArg(lean_object* v_inst_180_){
_start:
{
lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v_toAdd_183_; lean_object* v___f_184_; 
v___x_181_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_180_);
v___x_182_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_181_);
v_toAdd_183_ = lean_ctor_get(v___x_182_, 1);
lean_inc(v_toAdd_183_);
lean_dec_ref(v___x_182_);
v___f_184_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_toAddAction___redArg___lam__0), 3, 1);
lean_closure_set(v___f_184_, 0, v_toAdd_183_);
return v___f_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddAction___redArg___boxed(lean_object* v_inst_185_){
_start:
{
lean_object* v_res_186_; 
v_res_186_ = lp_mathlib_AddMonoid_toAddAction___redArg(v_inst_185_);
lean_dec_ref(v_inst_185_);
return v_res_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddAction(lean_object* v_M_187_, lean_object* v_inst_188_){
_start:
{
lean_object* v___x_189_; 
v___x_189_ = lp_mathlib_AddMonoid_toAddAction___redArg(v_inst_188_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_toAddAction___boxed(lean_object* v_M_190_, lean_object* v_inst_191_){
_start:
{
lean_object* v_res_192_; 
v_res_192_ = lp_mathlib_AddMonoid_toAddAction(v_M_190_, v_inst_191_);
lean_dec_ref(v_inst_191_);
return v_res_192_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Commute_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Notation_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Opposites(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Function_Iterate(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Spread(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Commute_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Notation_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Opposites(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Function_Iterate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Spread(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Action_Defs(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_LibraryNote_bundled__maps__over__different__rings = _init_lp_mathlib_LibraryNote_bundled__maps__over__different__rings();
lean_mark_persistent(lp_mathlib_LibraryNote_bundled__maps__over__different__rings);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Commute_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Notation_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Opposites(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Function_Iterate(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Spread(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Commute_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Notation_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Opposites(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Function_Iterate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Spread(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Action_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Action_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
