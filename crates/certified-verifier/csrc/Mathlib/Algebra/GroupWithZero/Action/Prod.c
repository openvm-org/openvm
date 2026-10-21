// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.Action.Prod
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Action.Prod public import Mathlib.Algebra.GroupWithZero.Action.End
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
lean_object* lp_mathlib_Prod_instSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_Prod_instMonoid___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_smulZeroClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_smulZeroClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_smulZeroClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_distribSMul___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_distribSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_distribSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_distribMulAction___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_distribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_distribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_mulDistribMulAction___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_mulDistribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_mulDistribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_smulWithZero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_smulWithZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_smulWithZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_mulActionWithZero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_mulActionWithZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_mulActionWithZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_prodOfSMulCommClass___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_prodOfSMulCommClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_prodOfSMulCommClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_prodOfSMulCommClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_prodEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inr___at___00DistribMulAction_prodEquiv___elam__0_spec__2___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inr___at___00DistribMulAction_prodEquiv___elam__0_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_prodEquiv___elam__0___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inl___at___00DistribMulAction_prodEquiv___elam__0_spec__0___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inl___at___00DistribMulAction_prodEquiv___elam__0_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_prodEquiv___elam__0___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMul_comp_smul___at___00DistribMulAction_prodEquiv___elam__0_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMul_comp_smul___at___00DistribMulAction_prodEquiv___elam__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMul_comp_smul___at___00DistribMulAction_prodEquiv___elam__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMul_comp_smul___at___00DistribMulAction_prodEquiv___elam__0_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMul_comp_smul___at___00DistribMulAction_prodEquiv___elam__0_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SMul_comp_smul___at___00DistribMulAction_prodEquiv___elam__0_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_prodEquiv___elam__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_prodEquiv___elam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_prodEquiv___elam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_DistribMulAction_prodEquiv___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_DistribMulAction_prodEquiv___redArg___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_DistribMulAction_prodEquiv___redArg___closed__0 = (const lean_object*)&lp_mathlib_DistribMulAction_prodEquiv___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_prodEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_prodEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inl___at___00DistribMulAction_prodEquiv___elam__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inl___at___00DistribMulAction_prodEquiv___elam__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inr___at___00DistribMulAction_prodEquiv___elam__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inr___at___00DistribMulAction_prodEquiv___elam__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_smulZeroClass___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___f_3_; 
v___f_3_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instSMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_3_, 0, v_inst_1_);
lean_closure_set(v___f_3_, 1, v_inst_2_);
return v___f_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_smulZeroClass(lean_object* v_R_4_, lean_object* v_M_5_, lean_object* v_N_6_, lean_object* v_inst_7_, lean_object* v_inst_8_, lean_object* v_inst_9_, lean_object* v_inst_10_){
_start:
{
lean_object* v___f_11_; 
v___f_11_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instSMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_11_, 0, v_inst_9_);
lean_closure_set(v___f_11_, 1, v_inst_10_);
return v___f_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_smulZeroClass___boxed(lean_object* v_R_12_, lean_object* v_M_13_, lean_object* v_N_14_, lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_inst_17_, lean_object* v_inst_18_){
_start:
{
lean_object* v_res_19_; 
v_res_19_ = lp_mathlib_Prod_smulZeroClass(v_R_12_, v_M_13_, v_N_14_, v_inst_15_, v_inst_16_, v_inst_17_, v_inst_18_);
lean_dec(v_inst_16_);
lean_dec(v_inst_15_);
return v_res_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_distribSMul___redArg(lean_object* v_inst_20_, lean_object* v_inst_21_){
_start:
{
lean_object* v___f_22_; 
v___f_22_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instSMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_22_, 0, v_inst_20_);
lean_closure_set(v___f_22_, 1, v_inst_21_);
return v___f_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_distribSMul(lean_object* v_R_23_, lean_object* v_M_24_, lean_object* v_N_25_, lean_object* v_inst_26_, lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_inst_29_){
_start:
{
lean_object* v___f_30_; 
v___f_30_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instSMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_30_, 0, v_inst_28_);
lean_closure_set(v___f_30_, 1, v_inst_29_);
return v___f_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_distribSMul___boxed(lean_object* v_R_31_, lean_object* v_M_32_, lean_object* v_N_33_, lean_object* v_inst_34_, lean_object* v_inst_35_, lean_object* v_inst_36_, lean_object* v_inst_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_mathlib_Prod_distribSMul(v_R_31_, v_M_32_, v_N_33_, v_inst_34_, v_inst_35_, v_inst_36_, v_inst_37_);
lean_dec_ref(v_inst_35_);
lean_dec_ref(v_inst_34_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_distribMulAction___redArg(lean_object* v_inst_39_, lean_object* v_inst_40_){
_start:
{
lean_object* v___f_41_; 
v___f_41_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instSMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_41_, 0, v_inst_39_);
lean_closure_set(v___f_41_, 1, v_inst_40_);
return v___f_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_distribMulAction(lean_object* v_M_42_, lean_object* v_N_43_, lean_object* v_R_44_, lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_inst_49_){
_start:
{
lean_object* v___f_50_; 
v___f_50_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instSMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_50_, 0, v_inst_48_);
lean_closure_set(v___f_50_, 1, v_inst_49_);
return v___f_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_distribMulAction___boxed(lean_object* v_M_51_, lean_object* v_N_52_, lean_object* v_R_53_, lean_object* v_inst_54_, lean_object* v_inst_55_, lean_object* v_inst_56_, lean_object* v_inst_57_, lean_object* v_inst_58_){
_start:
{
lean_object* v_res_59_; 
v_res_59_ = lp_mathlib_Prod_distribMulAction(v_M_51_, v_N_52_, v_R_53_, v_inst_54_, v_inst_55_, v_inst_56_, v_inst_57_, v_inst_58_);
lean_dec_ref(v_inst_56_);
lean_dec_ref(v_inst_55_);
lean_dec_ref(v_inst_54_);
return v_res_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_mulDistribMulAction___redArg(lean_object* v_inst_60_, lean_object* v_inst_61_){
_start:
{
lean_object* v___f_62_; 
v___f_62_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instSMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_62_, 0, v_inst_60_);
lean_closure_set(v___f_62_, 1, v_inst_61_);
return v___f_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_mulDistribMulAction(lean_object* v_M_63_, lean_object* v_N_64_, lean_object* v_R_65_, lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_inst_68_, lean_object* v_inst_69_, lean_object* v_inst_70_){
_start:
{
lean_object* v___f_71_; 
v___f_71_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instSMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_71_, 0, v_inst_69_);
lean_closure_set(v___f_71_, 1, v_inst_70_);
return v___f_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_mulDistribMulAction___boxed(lean_object* v_M_72_, lean_object* v_N_73_, lean_object* v_R_74_, lean_object* v_inst_75_, lean_object* v_inst_76_, lean_object* v_inst_77_, lean_object* v_inst_78_, lean_object* v_inst_79_){
_start:
{
lean_object* v_res_80_; 
v_res_80_ = lp_mathlib_Prod_mulDistribMulAction(v_M_72_, v_N_73_, v_R_74_, v_inst_75_, v_inst_76_, v_inst_77_, v_inst_78_, v_inst_79_);
lean_dec_ref(v_inst_77_);
lean_dec_ref(v_inst_76_);
lean_dec_ref(v_inst_75_);
return v_res_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_smulWithZero___redArg(lean_object* v_inst_81_, lean_object* v_inst_82_){
_start:
{
lean_object* v___f_83_; 
v___f_83_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instSMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_83_, 0, v_inst_81_);
lean_closure_set(v___f_83_, 1, v_inst_82_);
return v___f_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_smulWithZero(lean_object* v_M_84_, lean_object* v_N_85_, lean_object* v_R_86_, lean_object* v_inst_87_, lean_object* v_inst_88_, lean_object* v_inst_89_, lean_object* v_inst_90_, lean_object* v_inst_91_){
_start:
{
lean_object* v___f_92_; 
v___f_92_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instSMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_92_, 0, v_inst_90_);
lean_closure_set(v___f_92_, 1, v_inst_91_);
return v___f_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_smulWithZero___boxed(lean_object* v_M_93_, lean_object* v_N_94_, lean_object* v_R_95_, lean_object* v_inst_96_, lean_object* v_inst_97_, lean_object* v_inst_98_, lean_object* v_inst_99_, lean_object* v_inst_100_){
_start:
{
lean_object* v_res_101_; 
v_res_101_ = lp_mathlib_Prod_smulWithZero(v_M_93_, v_N_94_, v_R_95_, v_inst_96_, v_inst_97_, v_inst_98_, v_inst_99_, v_inst_100_);
lean_dec(v_inst_98_);
lean_dec(v_inst_97_);
lean_dec(v_inst_96_);
return v_res_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_mulActionWithZero___redArg(lean_object* v_inst_102_, lean_object* v_inst_103_){
_start:
{
lean_object* v___f_104_; 
v___f_104_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instSMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_104_, 0, v_inst_102_);
lean_closure_set(v___f_104_, 1, v_inst_103_);
return v___f_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_mulActionWithZero(lean_object* v_M_105_, lean_object* v_N_106_, lean_object* v_R_107_, lean_object* v_inst_108_, lean_object* v_inst_109_, lean_object* v_inst_110_, lean_object* v_inst_111_, lean_object* v_inst_112_){
_start:
{
lean_object* v___f_113_; 
v___f_113_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instSMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_113_, 0, v_inst_111_);
lean_closure_set(v___f_113_, 1, v_inst_112_);
return v___f_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_mulActionWithZero___boxed(lean_object* v_M_114_, lean_object* v_N_115_, lean_object* v_R_116_, lean_object* v_inst_117_, lean_object* v_inst_118_, lean_object* v_inst_119_, lean_object* v_inst_120_, lean_object* v_inst_121_){
_start:
{
lean_object* v_res_122_; 
v_res_122_ = lp_mathlib_Prod_mulActionWithZero(v_M_114_, v_N_115_, v_R_116_, v_inst_117_, v_inst_118_, v_inst_119_, v_inst_120_, v_inst_121_);
lean_dec(v_inst_119_);
lean_dec(v_inst_118_);
lean_dec_ref(v_inst_117_);
return v_res_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_prodOfSMulCommClass___redArg___lam__0(lean_object* v_inst_123_, lean_object* v_inst_124_, lean_object* v_mn_125_, lean_object* v_a_126_){
_start:
{
lean_object* v_fst_127_; lean_object* v_snd_128_; lean_object* v___x_129_; lean_object* v___x_130_; 
v_fst_127_ = lean_ctor_get(v_mn_125_, 0);
lean_inc(v_fst_127_);
v_snd_128_ = lean_ctor_get(v_mn_125_, 1);
lean_inc(v_snd_128_);
lean_dec_ref(v_mn_125_);
v___x_129_ = lean_apply_2(v_inst_123_, v_snd_128_, v_a_126_);
v___x_130_ = lean_apply_2(v_inst_124_, v_fst_127_, v___x_129_);
return v___x_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_prodOfSMulCommClass___redArg(lean_object* v_inst_131_, lean_object* v_inst_132_){
_start:
{
lean_object* v___f_133_; 
v___f_133_ = lean_alloc_closure((void*)(lp_mathlib_DistribMulAction_prodOfSMulCommClass___redArg___lam__0), 4, 2);
lean_closure_set(v___f_133_, 0, v_inst_132_);
lean_closure_set(v___f_133_, 1, v_inst_131_);
return v___f_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_prodOfSMulCommClass(lean_object* v_M_134_, lean_object* v_N_135_, lean_object* v_00_u03b1_136_, lean_object* v_inst_137_, lean_object* v_inst_138_, lean_object* v_inst_139_, lean_object* v_inst_140_, lean_object* v_inst_141_, lean_object* v_inst_142_){
_start:
{
lean_object* v___f_143_; 
v___f_143_ = lean_alloc_closure((void*)(lp_mathlib_DistribMulAction_prodOfSMulCommClass___redArg___lam__0), 4, 2);
lean_closure_set(v___f_143_, 0, v_inst_141_);
lean_closure_set(v___f_143_, 1, v_inst_140_);
return v___f_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_prodOfSMulCommClass___boxed(lean_object* v_M_144_, lean_object* v_N_145_, lean_object* v_00_u03b1_146_, lean_object* v_inst_147_, lean_object* v_inst_148_, lean_object* v_inst_149_, lean_object* v_inst_150_, lean_object* v_inst_151_, lean_object* v_inst_152_){
_start:
{
lean_object* v_res_153_; 
v_res_153_ = lp_mathlib_DistribMulAction_prodOfSMulCommClass(v_M_144_, v_N_145_, v_00_u03b1_146_, v_inst_147_, v_inst_148_, v_inst_149_, v_inst_150_, v_inst_151_, v_inst_152_);
lean_dec_ref(v_inst_149_);
lean_dec_ref(v_inst_148_);
lean_dec_ref(v_inst_147_);
return v_res_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_prodEquiv___redArg___lam__0(lean_object* v___insts_154_, lean_object* v___y_155_, lean_object* v___y_156_){
_start:
{
lean_object* v_snd_157_; lean_object* v_fst_158_; lean_object* v_fst_159_; lean_object* v_fst_160_; lean_object* v_snd_161_; lean_object* v___x_162_; lean_object* v___x_163_; 
v_snd_157_ = lean_ctor_get(v___insts_154_, 1);
lean_inc(v_snd_157_);
v_fst_158_ = lean_ctor_get(v___insts_154_, 0);
lean_inc(v_fst_158_);
lean_dec_ref(v___insts_154_);
v_fst_159_ = lean_ctor_get(v_snd_157_, 0);
lean_inc(v_fst_159_);
lean_dec(v_snd_157_);
v_fst_160_ = lean_ctor_get(v___y_155_, 0);
lean_inc(v_fst_160_);
v_snd_161_ = lean_ctor_get(v___y_155_, 1);
lean_inc(v_snd_161_);
lean_dec_ref(v___y_155_);
v___x_162_ = lean_apply_2(v_fst_159_, v_snd_161_, v___y_156_);
v___x_163_ = lean_apply_2(v_fst_158_, v_fst_160_, v___x_162_);
return v___x_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inr___at___00DistribMulAction_prodEquiv___elam__0_spec__2___redArg___lam__0(lean_object* v_toOne_164_, lean_object* v_y_165_){
_start:
{
lean_object* v___x_166_; 
v___x_166_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_166_, 0, v_toOne_164_);
lean_ctor_set(v___x_166_, 1, v_y_165_);
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inr___at___00DistribMulAction_prodEquiv___elam__0_spec__2___redArg(lean_object* v___x_167_){
_start:
{
lean_object* v___x_168_; lean_object* v_toOne_169_; lean_object* v___f_170_; 
v___x_168_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_167_);
v_toOne_169_ = lean_ctor_get(v___x_168_, 0);
lean_inc(v_toOne_169_);
lean_dec_ref(v___x_168_);
v___f_170_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_inr___at___00DistribMulAction_prodEquiv___elam__0_spec__2___redArg___lam__0), 2, 1);
lean_closure_set(v___f_170_, 0, v_toOne_169_);
return v___f_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_prodEquiv___elam__0___redArg___lam__1(lean_object* v___x_171_, lean_object* v___y_172_){
_start:
{
lean_object* v___x_304__overap_173_; lean_object* v___x_174_; 
v___x_304__overap_173_ = lp_mathlib_MonoidHom_inr___at___00DistribMulAction_prodEquiv___elam__0_spec__2___redArg(v___x_171_);
v___x_174_ = lean_apply_1(v___x_304__overap_173_, v___y_172_);
return v___x_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inl___at___00DistribMulAction_prodEquiv___elam__0_spec__0___redArg___lam__0(lean_object* v_toOne_175_, lean_object* v_x_176_){
_start:
{
lean_object* v___x_177_; 
v___x_177_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_177_, 0, v_x_176_);
lean_ctor_set(v___x_177_, 1, v_toOne_175_);
return v___x_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inl___at___00DistribMulAction_prodEquiv___elam__0_spec__0___redArg(lean_object* v___x_178_){
_start:
{
lean_object* v___x_179_; lean_object* v_toOne_180_; lean_object* v___f_181_; 
v___x_179_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_178_);
v_toOne_180_ = lean_ctor_get(v___x_179_, 0);
lean_inc(v_toOne_180_);
lean_dec_ref(v___x_179_);
v___f_181_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_inl___at___00DistribMulAction_prodEquiv___elam__0_spec__0___redArg___lam__0), 2, 1);
lean_closure_set(v___f_181_, 0, v_toOne_180_);
return v___f_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_prodEquiv___elam__0___redArg___lam__0(lean_object* v___x_182_, lean_object* v___y_183_){
_start:
{
lean_object* v___x_301__overap_184_; lean_object* v___x_185_; 
v___x_301__overap_184_ = lp_mathlib_MonoidHom_inl___at___00DistribMulAction_prodEquiv___elam__0_spec__0___redArg(v___x_182_);
v___x_185_ = lean_apply_1(v___x_301__overap_184_, v___y_183_);
return v___x_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMul_comp_smul___at___00DistribMulAction_prodEquiv___elam__0_spec__1___redArg(lean_object* v_x_186_, lean_object* v_g_187_, lean_object* v_n_188_, lean_object* v_a_189_){
_start:
{
lean_object* v___x_190_; lean_object* v___x_191_; 
v___x_190_ = lean_apply_1(v_g_187_, v_n_188_);
v___x_191_ = lean_apply_2(v_x_186_, v___x_190_, v_a_189_);
return v___x_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMul_comp_smul___at___00DistribMulAction_prodEquiv___elam__0_spec__1(lean_object* v_M_192_, lean_object* v_N_193_, lean_object* v_00_u03b1_194_, lean_object* v___x_195_, lean_object* v_inst_196_, lean_object* v_x_197_, lean_object* v_N_198_, lean_object* v_g_199_, lean_object* v_n_200_, lean_object* v_a_201_){
_start:
{
lean_object* v___x_202_; 
v___x_202_ = lp_mathlib_SMul_comp_smul___at___00DistribMulAction_prodEquiv___elam__0_spec__1___redArg(v_x_197_, v_g_199_, v_n_200_, v_a_201_);
return v___x_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMul_comp_smul___at___00DistribMulAction_prodEquiv___elam__0_spec__1___boxed(lean_object* v_M_203_, lean_object* v_N_204_, lean_object* v_00_u03b1_205_, lean_object* v___x_206_, lean_object* v_inst_207_, lean_object* v_x_208_, lean_object* v_N_209_, lean_object* v_g_210_, lean_object* v_n_211_, lean_object* v_a_212_){
_start:
{
lean_object* v_res_213_; 
v_res_213_ = lp_mathlib_SMul_comp_smul___at___00DistribMulAction_prodEquiv___elam__0_spec__1(v_M_203_, v_N_204_, v_00_u03b1_205_, v___x_206_, v_inst_207_, v_x_208_, v_N_209_, v_g_210_, v_n_211_, v_a_212_);
lean_dec_ref(v_inst_207_);
lean_dec_ref(v___x_206_);
return v_res_213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMul_comp_smul___at___00DistribMulAction_prodEquiv___elam__0_spec__3___redArg(lean_object* v_x_214_, lean_object* v_g_215_, lean_object* v_n_216_, lean_object* v_a_217_){
_start:
{
lean_object* v___x_218_; lean_object* v___x_219_; 
v___x_218_ = lean_apply_1(v_g_215_, v_n_216_);
v___x_219_ = lean_apply_2(v_x_214_, v___x_218_, v_a_217_);
return v___x_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMul_comp_smul___at___00DistribMulAction_prodEquiv___elam__0_spec__3(lean_object* v_M_220_, lean_object* v_N_221_, lean_object* v_00_u03b1_222_, lean_object* v___x_223_, lean_object* v_inst_224_, lean_object* v_x_225_, lean_object* v_N_226_, lean_object* v_g_227_, lean_object* v_n_228_, lean_object* v_a_229_){
_start:
{
lean_object* v___x_230_; 
v___x_230_ = lp_mathlib_SMul_comp_smul___at___00DistribMulAction_prodEquiv___elam__0_spec__3___redArg(v_x_225_, v_g_227_, v_n_228_, v_a_229_);
return v___x_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SMul_comp_smul___at___00DistribMulAction_prodEquiv___elam__0_spec__3___boxed(lean_object* v_M_231_, lean_object* v_N_232_, lean_object* v_00_u03b1_233_, lean_object* v___x_234_, lean_object* v_inst_235_, lean_object* v_x_236_, lean_object* v_N_237_, lean_object* v_g_238_, lean_object* v_n_239_, lean_object* v_a_240_){
_start:
{
lean_object* v_res_241_; 
v_res_241_ = lp_mathlib_SMul_comp_smul___at___00DistribMulAction_prodEquiv___elam__0_spec__3(v_M_231_, v_N_232_, v_00_u03b1_233_, v___x_234_, v_inst_235_, v_x_236_, v_N_237_, v_g_238_, v_n_239_, v_a_240_);
lean_dec_ref(v_inst_235_);
lean_dec_ref(v___x_234_);
return v_res_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_prodEquiv___elam__0___redArg(lean_object* v___x_242_, lean_object* v___x_243_, lean_object* v___x_244_, lean_object* v_inst_245_, lean_object* v_x_246_){
_start:
{
lean_object* v___f_247_; lean_object* v___f_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; 
v___f_247_ = lean_alloc_closure((void*)(lp_mathlib_DistribMulAction_prodEquiv___elam__0___redArg___lam__0), 2, 1);
lean_closure_set(v___f_247_, 0, v___x_243_);
v___f_248_ = lean_alloc_closure((void*)(lp_mathlib_DistribMulAction_prodEquiv___elam__0___redArg___lam__1), 2, 1);
lean_closure_set(v___f_248_, 0, v___x_242_);
lean_inc(v_x_246_);
lean_inc_ref(v_inst_245_);
lean_inc_ref(v___x_244_);
v___x_249_ = lean_alloc_closure((void*)(lp_mathlib_SMul_comp_smul___at___00DistribMulAction_prodEquiv___elam__0_spec__1___boxed), 10, 8);
lean_closure_set(v___x_249_, 0, lean_box(0));
lean_closure_set(v___x_249_, 1, lean_box(0));
lean_closure_set(v___x_249_, 2, lean_box(0));
lean_closure_set(v___x_249_, 3, v___x_244_);
lean_closure_set(v___x_249_, 4, v_inst_245_);
lean_closure_set(v___x_249_, 5, v_x_246_);
lean_closure_set(v___x_249_, 6, lean_box(0));
lean_closure_set(v___x_249_, 7, v___f_247_);
v___x_250_ = lean_alloc_closure((void*)(lp_mathlib_SMul_comp_smul___at___00DistribMulAction_prodEquiv___elam__0_spec__3___boxed), 10, 8);
lean_closure_set(v___x_250_, 0, lean_box(0));
lean_closure_set(v___x_250_, 1, lean_box(0));
lean_closure_set(v___x_250_, 2, lean_box(0));
lean_closure_set(v___x_250_, 3, v___x_244_);
lean_closure_set(v___x_250_, 4, v_inst_245_);
lean_closure_set(v___x_250_, 5, v_x_246_);
lean_closure_set(v___x_250_, 6, lean_box(0));
lean_closure_set(v___x_250_, 7, v___f_248_);
v___x_251_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_251_, 0, v___x_250_);
lean_ctor_set(v___x_251_, 1, lean_box(0));
v___x_252_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_252_, 0, v___x_249_);
lean_ctor_set(v___x_252_, 1, v___x_251_);
return v___x_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_prodEquiv___elam__0(lean_object* v_M_253_, lean_object* v_N_254_, lean_object* v_00_u03b1_255_, lean_object* v___x_256_, lean_object* v___x_257_, lean_object* v___x_258_, lean_object* v_inst_259_, lean_object* v_inst_260_, lean_object* v_inst_261_, lean_object* v_x_262_){
_start:
{
lean_object* v___x_263_; 
v___x_263_ = lp_mathlib_DistribMulAction_prodEquiv___elam__0___redArg(v___x_256_, v___x_257_, v___x_258_, v_inst_259_, v_x_262_);
return v___x_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_prodEquiv___elam__0___boxed(lean_object* v_M_264_, lean_object* v_N_265_, lean_object* v_00_u03b1_266_, lean_object* v___x_267_, lean_object* v___x_268_, lean_object* v___x_269_, lean_object* v_inst_270_, lean_object* v_inst_271_, lean_object* v_inst_272_, lean_object* v_x_273_){
_start:
{
lean_object* v_res_274_; 
v_res_274_ = lp_mathlib_DistribMulAction_prodEquiv___elam__0(v_M_264_, v_N_265_, v_00_u03b1_266_, v___x_267_, v___x_268_, v___x_269_, v_inst_270_, v_inst_271_, v_inst_272_, v_x_273_);
lean_dec_ref(v_inst_272_);
lean_dec_ref(v_inst_271_);
return v_res_274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_prodEquiv___redArg(lean_object* v_inst_276_, lean_object* v_inst_277_, lean_object* v_inst_278_){
_start:
{
lean_object* v___f_279_; lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___f_283_; lean_object* v___x_284_; 
v___f_279_ = ((lean_object*)(lp_mathlib_DistribMulAction_prodEquiv___redArg___closed__0));
lean_inc_ref(v_inst_277_);
lean_inc_ref(v_inst_276_);
v___x_280_ = lp_mathlib_Prod_instMonoid___redArg(v_inst_276_, v_inst_277_);
v___x_281_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_276_);
v___x_282_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_277_);
v___f_283_ = lean_alloc_closure((void*)(lp_mathlib_DistribMulAction_prodEquiv___elam__0___boxed), 10, 9);
lean_closure_set(v___f_283_, 0, lean_box(0));
lean_closure_set(v___f_283_, 1, lean_box(0));
lean_closure_set(v___f_283_, 2, lean_box(0));
lean_closure_set(v___f_283_, 3, v___x_281_);
lean_closure_set(v___f_283_, 4, v___x_282_);
lean_closure_set(v___f_283_, 5, v___x_280_);
lean_closure_set(v___f_283_, 6, v_inst_278_);
lean_closure_set(v___f_283_, 7, v_inst_276_);
lean_closure_set(v___f_283_, 8, v_inst_277_);
v___x_284_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_284_, 0, v___f_283_);
lean_ctor_set(v___x_284_, 1, v___f_279_);
return v___x_284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DistribMulAction_prodEquiv(lean_object* v_M_285_, lean_object* v_N_286_, lean_object* v_00_u03b1_287_, lean_object* v_inst_288_, lean_object* v_inst_289_, lean_object* v_inst_290_){
_start:
{
lean_object* v___x_291_; 
v___x_291_ = lp_mathlib_DistribMulAction_prodEquiv___redArg(v_inst_288_, v_inst_289_, v_inst_290_);
return v___x_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inl___at___00DistribMulAction_prodEquiv___elam__0_spec__0(lean_object* v_M_292_, lean_object* v_N_293_, lean_object* v___x_294_, lean_object* v___x_295_){
_start:
{
lean_object* v___x_296_; 
v___x_296_ = lp_mathlib_MonoidHom_inl___at___00DistribMulAction_prodEquiv___elam__0_spec__0___redArg(v___x_295_);
return v___x_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inl___at___00DistribMulAction_prodEquiv___elam__0_spec__0___boxed(lean_object* v_M_297_, lean_object* v_N_298_, lean_object* v___x_299_, lean_object* v___x_300_){
_start:
{
lean_object* v_res_301_; 
v_res_301_ = lp_mathlib_MonoidHom_inl___at___00DistribMulAction_prodEquiv___elam__0_spec__0(v_M_297_, v_N_298_, v___x_299_, v___x_300_);
lean_dec_ref(v___x_299_);
return v_res_301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inr___at___00DistribMulAction_prodEquiv___elam__0_spec__2(lean_object* v_M_302_, lean_object* v_N_303_, lean_object* v___x_304_, lean_object* v___x_305_){
_start:
{
lean_object* v___x_306_; 
v___x_306_ = lp_mathlib_MonoidHom_inr___at___00DistribMulAction_prodEquiv___elam__0_spec__2___redArg(v___x_304_);
return v___x_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inr___at___00DistribMulAction_prodEquiv___elam__0_spec__2___boxed(lean_object* v_M_307_, lean_object* v_N_308_, lean_object* v___x_309_, lean_object* v___x_310_){
_start:
{
lean_object* v_res_311_; 
v_res_311_ = lp_mathlib_MonoidHom_inr___at___00DistribMulAction_prodEquiv___elam__0_spec__2(v_M_307_, v_N_308_, v___x_309_, v___x_310_);
lean_dec_ref(v___x_310_);
return v_res_311_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_End(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Prod(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Prod(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_End(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Prod(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Prod(builtin);
}
#ifdef __cplusplus
}
#endif
