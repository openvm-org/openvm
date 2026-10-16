// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.Action.Pi
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Action.Pi public import Mathlib.Algebra.GroupWithZero.Action.Defs public import Mathlib.Algebra.GroupWithZero.Defs public import Mathlib.Algebra.GroupWithZero.Pi public import Mathlib.Tactic.Common
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
lean_object* lp_mathlib_Pi_smul_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_mulAction___redArg(lean_object*);
lean_object* lp_mathlib_Pi_instSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_mulAction_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_smulZeroClass___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_smulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_smulZeroClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_smulZeroClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_smulZeroClass_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_smulZeroClass_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_smulZeroClass_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_distribSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_distribSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_distribSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_distribSMul_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_distribSMul_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_distribSMul_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_distribMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_distribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_distribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_distribMulAction_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_distribMulAction_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_distribMulAction_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_smulWithZero___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_smulWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_smulWithZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_smulWithZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_smulWithZero_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_smulWithZero_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_smulWithZero_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulActionWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulActionWithZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulActionWithZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulActionWithZero_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulActionWithZero_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulActionWithZero_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulDistribMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulDistribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulDistribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulDistribMulAction_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulDistribMulAction_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulDistribMulAction_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_smulZeroClass___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_i_2_, lean_object* v___y_3_, lean_object* v___y_4_){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_apply_3(v_inst_1_, v_i_2_, v___y_3_, v___y_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_smulZeroClass___redArg(lean_object* v_inst_6_){
_start:
{
lean_object* v___f_7_; lean_object* v___f_8_; 
v___f_7_ = lean_alloc_closure((void*)(lp_mathlib_Pi_smulZeroClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_7_, 0, v_inst_6_);
v___f_8_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_8_, 0, v___f_7_);
return v___f_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_smulZeroClass(lean_object* v_I_9_, lean_object* v_f_10_, lean_object* v_00_u03b1_11_, lean_object* v_n_12_, lean_object* v_inst_13_){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lp_mathlib_Pi_smulZeroClass___redArg(v_inst_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_smulZeroClass___boxed(lean_object* v_I_15_, lean_object* v_f_16_, lean_object* v_00_u03b1_17_, lean_object* v_n_18_, lean_object* v_inst_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_mathlib_Pi_smulZeroClass(v_I_15_, v_f_16_, v_00_u03b1_17_, v_n_18_, v_inst_19_);
lean_dec(v_n_18_);
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_smulZeroClass_x27___redArg(lean_object* v_inst_21_){
_start:
{
lean_object* v___f_22_; lean_object* v___f_23_; 
v___f_22_ = lean_alloc_closure((void*)(lp_mathlib_Pi_smulZeroClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_22_, 0, v_inst_21_);
v___f_23_ = lean_alloc_closure((void*)(lp_mathlib_Pi_smul_x27___redArg___lam__0), 4, 1);
lean_closure_set(v___f_23_, 0, v___f_22_);
return v___f_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_smulZeroClass_x27(lean_object* v_I_24_, lean_object* v_f_25_, lean_object* v_g_26_, lean_object* v_n_27_, lean_object* v_inst_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lp_mathlib_Pi_smulZeroClass_x27___redArg(v_inst_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_smulZeroClass_x27___boxed(lean_object* v_I_30_, lean_object* v_f_31_, lean_object* v_g_32_, lean_object* v_n_33_, lean_object* v_inst_34_){
_start:
{
lean_object* v_res_35_; 
v_res_35_ = lp_mathlib_Pi_smulZeroClass_x27(v_I_30_, v_f_31_, v_g_32_, v_n_33_, v_inst_34_);
lean_dec(v_n_33_);
return v_res_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_distribSMul___redArg(lean_object* v_inst_36_){
_start:
{
lean_object* v___f_37_; lean_object* v___f_38_; 
v___f_37_ = lean_alloc_closure((void*)(lp_mathlib_Pi_smulZeroClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_37_, 0, v_inst_36_);
v___f_38_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_38_, 0, v___f_37_);
return v___f_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_distribSMul(lean_object* v_I_39_, lean_object* v_f_40_, lean_object* v_00_u03b1_41_, lean_object* v_n_42_, lean_object* v_inst_43_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = lp_mathlib_Pi_distribSMul___redArg(v_inst_43_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_distribSMul___boxed(lean_object* v_I_45_, lean_object* v_f_46_, lean_object* v_00_u03b1_47_, lean_object* v_n_48_, lean_object* v_inst_49_){
_start:
{
lean_object* v_res_50_; 
v_res_50_ = lp_mathlib_Pi_distribSMul(v_I_45_, v_f_46_, v_00_u03b1_47_, v_n_48_, v_inst_49_);
lean_dec_ref(v_n_48_);
return v_res_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_distribSMul_x27___redArg(lean_object* v_inst_51_){
_start:
{
lean_object* v___f_52_; lean_object* v___f_53_; 
v___f_52_ = lean_alloc_closure((void*)(lp_mathlib_Pi_smulZeroClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_52_, 0, v_inst_51_);
v___f_53_ = lean_alloc_closure((void*)(lp_mathlib_Pi_smul_x27___redArg___lam__0), 4, 1);
lean_closure_set(v___f_53_, 0, v___f_52_);
return v___f_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_distribSMul_x27(lean_object* v_I_54_, lean_object* v_f_55_, lean_object* v_g_56_, lean_object* v_n_57_, lean_object* v_inst_58_){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = lp_mathlib_Pi_distribSMul_x27___redArg(v_inst_58_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_distribSMul_x27___boxed(lean_object* v_I_60_, lean_object* v_f_61_, lean_object* v_g_62_, lean_object* v_n_63_, lean_object* v_inst_64_){
_start:
{
lean_object* v_res_65_; 
v_res_65_ = lp_mathlib_Pi_distribSMul_x27(v_I_60_, v_f_61_, v_g_62_, v_n_63_, v_inst_64_);
lean_dec_ref(v_n_63_);
return v_res_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_distribMulAction___redArg(lean_object* v_inst_66_){
_start:
{
lean_object* v___f_67_; lean_object* v___x_68_; 
v___f_67_ = lean_alloc_closure((void*)(lp_mathlib_Pi_smulZeroClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_67_, 0, v_inst_66_);
v___x_68_ = lp_mathlib_Pi_mulAction___redArg(v___f_67_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_distribMulAction(lean_object* v_I_69_, lean_object* v_f_70_, lean_object* v_00_u03b1_71_, lean_object* v_m_72_, lean_object* v_n_73_, lean_object* v_inst_74_){
_start:
{
lean_object* v___x_75_; 
v___x_75_ = lp_mathlib_Pi_distribMulAction___redArg(v_inst_74_);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_distribMulAction___boxed(lean_object* v_I_76_, lean_object* v_f_77_, lean_object* v_00_u03b1_78_, lean_object* v_m_79_, lean_object* v_n_80_, lean_object* v_inst_81_){
_start:
{
lean_object* v_res_82_; 
v_res_82_ = lp_mathlib_Pi_distribMulAction(v_I_76_, v_f_77_, v_00_u03b1_78_, v_m_79_, v_n_80_, v_inst_81_);
lean_dec_ref(v_n_80_);
lean_dec_ref(v_m_79_);
return v_res_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_distribMulAction_x27___redArg(lean_object* v_inst_83_){
_start:
{
lean_object* v___f_84_; lean_object* v___x_85_; 
v___f_84_ = lean_alloc_closure((void*)(lp_mathlib_Pi_smulZeroClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_84_, 0, v_inst_83_);
v___x_85_ = lp_mathlib_Pi_mulAction_x27___redArg(v___f_84_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_distribMulAction_x27(lean_object* v_I_86_, lean_object* v_f_87_, lean_object* v_g_88_, lean_object* v_m_89_, lean_object* v_n_90_, lean_object* v_inst_91_){
_start:
{
lean_object* v___x_92_; 
v___x_92_ = lp_mathlib_Pi_distribMulAction_x27___redArg(v_inst_91_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_distribMulAction_x27___boxed(lean_object* v_I_93_, lean_object* v_f_94_, lean_object* v_g_95_, lean_object* v_m_96_, lean_object* v_n_97_, lean_object* v_inst_98_){
_start:
{
lean_object* v_res_99_; 
v_res_99_ = lp_mathlib_Pi_distribMulAction_x27(v_I_93_, v_f_94_, v_g_95_, v_m_96_, v_n_97_, v_inst_98_);
lean_dec_ref(v_n_97_);
lean_dec_ref(v_m_96_);
return v_res_99_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_smulWithZero___redArg___lam__0(lean_object* v_inst_100_, lean_object* v_x_101_, lean_object* v___y_102_, lean_object* v___y_103_){
_start:
{
lean_object* v___x_104_; 
v___x_104_ = lean_apply_3(v_inst_100_, v_x_101_, v___y_102_, v___y_103_);
return v___x_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_smulWithZero___redArg(lean_object* v_inst_105_){
_start:
{
lean_object* v___f_106_; lean_object* v___f_107_; 
v___f_106_ = lean_alloc_closure((void*)(lp_mathlib_Pi_smulWithZero___redArg___lam__0), 4, 1);
lean_closure_set(v___f_106_, 0, v_inst_105_);
v___f_107_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instSMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_107_, 0, v___f_106_);
return v___f_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_smulWithZero(lean_object* v_I_108_, lean_object* v_f_109_, lean_object* v_00_u03b1_110_, lean_object* v_inst_111_, lean_object* v_inst_112_, lean_object* v_inst_113_){
_start:
{
lean_object* v___x_114_; 
v___x_114_ = lp_mathlib_Pi_smulWithZero___redArg(v_inst_113_);
return v___x_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_smulWithZero___boxed(lean_object* v_I_115_, lean_object* v_f_116_, lean_object* v_00_u03b1_117_, lean_object* v_inst_118_, lean_object* v_inst_119_, lean_object* v_inst_120_){
_start:
{
lean_object* v_res_121_; 
v_res_121_ = lp_mathlib_Pi_smulWithZero(v_I_115_, v_f_116_, v_00_u03b1_117_, v_inst_118_, v_inst_119_, v_inst_120_);
lean_dec(v_inst_119_);
lean_dec(v_inst_118_);
return v_res_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_smulWithZero_x27___redArg(lean_object* v_inst_122_){
_start:
{
lean_object* v___f_123_; lean_object* v___f_124_; 
v___f_123_ = lean_alloc_closure((void*)(lp_mathlib_Pi_smulWithZero___redArg___lam__0), 4, 1);
lean_closure_set(v___f_123_, 0, v_inst_122_);
v___f_124_ = lean_alloc_closure((void*)(lp_mathlib_Pi_smul_x27___redArg___lam__0), 4, 1);
lean_closure_set(v___f_124_, 0, v___f_123_);
return v___f_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_smulWithZero_x27(lean_object* v_I_125_, lean_object* v_f_126_, lean_object* v_g_127_, lean_object* v_inst_128_, lean_object* v_inst_129_, lean_object* v_inst_130_){
_start:
{
lean_object* v___x_131_; 
v___x_131_ = lp_mathlib_Pi_smulWithZero_x27___redArg(v_inst_130_);
return v___x_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_smulWithZero_x27___boxed(lean_object* v_I_132_, lean_object* v_f_133_, lean_object* v_g_134_, lean_object* v_inst_135_, lean_object* v_inst_136_, lean_object* v_inst_137_){
_start:
{
lean_object* v_res_138_; 
v_res_138_ = lp_mathlib_Pi_smulWithZero_x27(v_I_132_, v_f_133_, v_g_134_, v_inst_135_, v_inst_136_, v_inst_137_);
lean_dec(v_inst_136_);
lean_dec(v_inst_135_);
return v_res_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulActionWithZero___redArg(lean_object* v_inst_139_){
_start:
{
lean_object* v___f_140_; lean_object* v___x_141_; 
v___f_140_ = lean_alloc_closure((void*)(lp_mathlib_Pi_smulZeroClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_140_, 0, v_inst_139_);
v___x_141_ = lp_mathlib_Pi_mulAction___redArg(v___f_140_);
return v___x_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulActionWithZero(lean_object* v_I_142_, lean_object* v_f_143_, lean_object* v_00_u03b1_144_, lean_object* v_inst_145_, lean_object* v_inst_146_, lean_object* v_inst_147_){
_start:
{
lean_object* v___x_148_; 
v___x_148_ = lp_mathlib_Pi_mulActionWithZero___redArg(v_inst_147_);
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulActionWithZero___boxed(lean_object* v_I_149_, lean_object* v_f_150_, lean_object* v_00_u03b1_151_, lean_object* v_inst_152_, lean_object* v_inst_153_, lean_object* v_inst_154_){
_start:
{
lean_object* v_res_155_; 
v_res_155_ = lp_mathlib_Pi_mulActionWithZero(v_I_149_, v_f_150_, v_00_u03b1_151_, v_inst_152_, v_inst_153_, v_inst_154_);
lean_dec(v_inst_153_);
lean_dec_ref(v_inst_152_);
return v_res_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulActionWithZero_x27___redArg(lean_object* v_inst_156_){
_start:
{
lean_object* v___f_157_; lean_object* v___x_158_; 
v___f_157_ = lean_alloc_closure((void*)(lp_mathlib_Pi_smulZeroClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_157_, 0, v_inst_156_);
v___x_158_ = lp_mathlib_Pi_mulAction_x27___redArg(v___f_157_);
return v___x_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulActionWithZero_x27(lean_object* v_I_159_, lean_object* v_f_160_, lean_object* v_g_161_, lean_object* v_inst_162_, lean_object* v_inst_163_, lean_object* v_inst_164_){
_start:
{
lean_object* v___x_165_; 
v___x_165_ = lp_mathlib_Pi_mulActionWithZero_x27___redArg(v_inst_164_);
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulActionWithZero_x27___boxed(lean_object* v_I_166_, lean_object* v_f_167_, lean_object* v_g_168_, lean_object* v_inst_169_, lean_object* v_inst_170_, lean_object* v_inst_171_){
_start:
{
lean_object* v_res_172_; 
v_res_172_ = lp_mathlib_Pi_mulActionWithZero_x27(v_I_166_, v_f_167_, v_g_168_, v_inst_169_, v_inst_170_, v_inst_171_);
lean_dec(v_inst_170_);
lean_dec_ref(v_inst_169_);
return v_res_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulDistribMulAction___redArg(lean_object* v_inst_173_){
_start:
{
lean_object* v___f_174_; lean_object* v___x_175_; 
v___f_174_ = lean_alloc_closure((void*)(lp_mathlib_Pi_smulZeroClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_174_, 0, v_inst_173_);
v___x_175_ = lp_mathlib_Pi_mulAction___redArg(v___f_174_);
return v___x_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulDistribMulAction(lean_object* v_I_176_, lean_object* v_f_177_, lean_object* v_00_u03b1_178_, lean_object* v_m_179_, lean_object* v_n_180_, lean_object* v_inst_181_){
_start:
{
lean_object* v___x_182_; 
v___x_182_ = lp_mathlib_Pi_mulDistribMulAction___redArg(v_inst_181_);
return v___x_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulDistribMulAction___boxed(lean_object* v_I_183_, lean_object* v_f_184_, lean_object* v_00_u03b1_185_, lean_object* v_m_186_, lean_object* v_n_187_, lean_object* v_inst_188_){
_start:
{
lean_object* v_res_189_; 
v_res_189_ = lp_mathlib_Pi_mulDistribMulAction(v_I_183_, v_f_184_, v_00_u03b1_185_, v_m_186_, v_n_187_, v_inst_188_);
lean_dec_ref(v_n_187_);
lean_dec_ref(v_m_186_);
return v_res_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulDistribMulAction_x27___redArg(lean_object* v_inst_190_){
_start:
{
lean_object* v___f_191_; lean_object* v___x_192_; 
v___f_191_ = lean_alloc_closure((void*)(lp_mathlib_Pi_smulZeroClass___redArg___lam__0), 4, 1);
lean_closure_set(v___f_191_, 0, v_inst_190_);
v___x_192_ = lp_mathlib_Pi_mulAction_x27___redArg(v___f_191_);
return v___x_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulDistribMulAction_x27(lean_object* v_I_193_, lean_object* v_f_194_, lean_object* v_g_195_, lean_object* v_m_196_, lean_object* v_n_197_, lean_object* v_inst_198_){
_start:
{
lean_object* v___x_199_; 
v___x_199_ = lp_mathlib_Pi_mulDistribMulAction_x27___redArg(v_inst_198_);
return v___x_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulDistribMulAction_x27___boxed(lean_object* v_I_200_, lean_object* v_f_201_, lean_object* v_g_202_, lean_object* v_m_203_, lean_object* v_n_204_, lean_object* v_inst_205_){
_start:
{
lean_object* v_res_206_; 
v_res_206_ = lp_mathlib_Pi_mulDistribMulAction_x27(v_I_200_, v_f_201_, v_g_202_, v_m_203_, v_n_204_, v_inst_205_);
lean_dec_ref(v_n_204_);
lean_dec_ref(v_m_203_);
return v_res_206_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Pi(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Pi(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Pi(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Pi(builtin);
}
#ifdef __cplusplus
}
#endif
