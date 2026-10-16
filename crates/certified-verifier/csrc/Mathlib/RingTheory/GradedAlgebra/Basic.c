// Lean compiler output
// Module: Mathlib.RingTheory.GradedAlgebra.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.DirectSum.Algebra public import Mathlib.Algebra.DirectSum.Decomposition public import Mathlib.Algebra.DirectSum.Internal public import Mathlib.Algebra.DirectSum.Ring
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
lean_object* lp_mathlib_DirectSum_decompose___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_RingHom_codRestrict___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_DirectSum_decomposeAddEquiv___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_SMulMemClass_subtype___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_DFinsupp_lapply___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_AlgHom_toLinearMap___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubmonoidClass_subtype___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_DFinsupp_evalAddMonoidHom___redArg(lean_object*);
lean_object* lp_mathlib_OneHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decomposeRingEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decomposeRingEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decomposeRingEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_GradedRing_proj___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SubmonoidClass_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_GradedRing_proj___redArg___closed__0 = (const lean_object*)&lp_mathlib_GradedRing_proj___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_GradedRing_proj___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedRing_proj(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedRing_proj___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedAlgebra_ofAlgHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedAlgebra_ofAlgHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedAlgebra_ofAlgHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedAlgebra_ofAlgHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instGradedAlgebraRestrictScalars___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instGradedAlgebraRestrictScalars___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instGradedAlgebraRestrictScalars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instGradedAlgebraRestrictScalars___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decomposeAlgEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decomposeAlgEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decomposeAlgEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_GradedAlgebra_proj___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SMulMemClass_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_GradedAlgebra_proj___redArg___closed__0 = (const lean_object*)&lp_mathlib_GradedAlgebra_proj___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_GradedAlgebra_proj___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedAlgebra_proj(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedAlgebra_proj___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedRing_projZeroRingHom___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedRing_projZeroRingHom___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedRing_projZeroRingHom___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedRing_projZeroRingHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedRing_projZeroRingHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedRing_projZeroRingHom_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedRing_projZeroRingHom_x27___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedRing_projZeroRingHom_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GradedRing_projZeroRingHom_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decomposeRingEquiv___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_, lean_object* v_inst_3_){
_start:
{
lean_object* v_toAddCommMonoid_4_; lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; 
v_toAddCommMonoid_4_ = lean_ctor_get(v_inst_2_, 0);
lean_inc_ref(v_toAddCommMonoid_4_);
lean_dec_ref(v_inst_2_);
v___x_5_ = lp_mathlib_DirectSum_decomposeAddEquiv___redArg(v_inst_1_, v_toAddCommMonoid_4_, v_inst_3_);
v___x_6_ = lp_mathlib_Equiv_symm___redArg(v___x_5_);
v___x_7_ = lp_mathlib_Equiv_symm___redArg(v___x_6_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decomposeRingEquiv(lean_object* v_00_u03b9_8_, lean_object* v_A_9_, lean_object* v_00_u03c3_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_inst_13_, lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_00_U0001d49c_16_, lean_object* v_inst_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lp_mathlib_DirectSum_decomposeRingEquiv___redArg(v_inst_11_, v_inst_13_, v_inst_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decomposeRingEquiv___boxed(lean_object* v_00_u03b9_19_, lean_object* v_A_20_, lean_object* v_00_u03c3_21_, lean_object* v_inst_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_inst_26_, lean_object* v_00_U0001d49c_27_, lean_object* v_inst_28_){
_start:
{
lean_object* v_res_29_; 
v_res_29_ = lp_mathlib_DirectSum_decomposeRingEquiv(v_00_u03b9_19_, v_A_20_, v_00_u03c3_21_, v_inst_22_, v_inst_23_, v_inst_24_, v_inst_25_, v_inst_26_, v_00_U0001d49c_27_, v_inst_28_);
lean_dec(v_00_U0001d49c_27_);
lean_dec_ref(v_inst_23_);
return v_res_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedRing_proj___redArg(lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_i_34_){
_start:
{
lean_object* v___x_35_; lean_object* v_toFun_36_; lean_object* v___f_37_; lean_object* v___x_38_; lean_object* v___f_39_; lean_object* v___f_40_; 
v___x_35_ = lp_mathlib_DirectSum_decomposeRingEquiv___redArg(v_inst_31_, v_inst_32_, v_inst_33_);
v_toFun_36_ = lean_ctor_get(v___x_35_, 0);
lean_inc(v_toFun_36_);
lean_dec_ref(v___x_35_);
v___f_37_ = ((lean_object*)(lp_mathlib_GradedRing_proj___redArg___closed__0));
v___x_38_ = lp_mathlib_DFinsupp_evalAddMonoidHom___redArg(v_i_34_);
v___f_39_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_39_, 0, v_toFun_36_);
lean_closure_set(v___f_39_, 1, v___x_38_);
v___f_40_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_40_, 0, v___f_39_);
lean_closure_set(v___f_40_, 1, v___f_37_);
return v___f_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedRing_proj(lean_object* v_00_u03b9_41_, lean_object* v_A_42_, lean_object* v_00_u03c3_43_, lean_object* v_inst_44_, lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_00_U0001d49c_49_, lean_object* v_inst_50_, lean_object* v_i_51_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lp_mathlib_GradedRing_proj___redArg(v_inst_44_, v_inst_46_, v_inst_50_, v_i_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedRing_proj___boxed(lean_object* v_00_u03b9_53_, lean_object* v_A_54_, lean_object* v_00_u03c3_55_, lean_object* v_inst_56_, lean_object* v_inst_57_, lean_object* v_inst_58_, lean_object* v_inst_59_, lean_object* v_inst_60_, lean_object* v_00_U0001d49c_61_, lean_object* v_inst_62_, lean_object* v_i_63_){
_start:
{
lean_object* v_res_64_; 
v_res_64_ = lp_mathlib_GradedRing_proj(v_00_u03b9_53_, v_A_54_, v_00_u03c3_55_, v_inst_56_, v_inst_57_, v_inst_58_, v_inst_59_, v_inst_60_, v_00_U0001d49c_61_, v_inst_62_, v_i_63_);
lean_dec(v_00_U0001d49c_61_);
lean_dec_ref(v_inst_57_);
return v_res_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedAlgebra_ofAlgHom___redArg___lam__0(lean_object* v_decompose_65_, lean_object* v___y_66_){
_start:
{
lean_object* v___x_67_; 
v___x_67_ = lean_apply_1(v_decompose_65_, v___y_66_);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedAlgebra_ofAlgHom___redArg(lean_object* v_decompose_68_){
_start:
{
lean_object* v___f_69_; 
v___f_69_ = lean_alloc_closure((void*)(lp_mathlib_GradedAlgebra_ofAlgHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_69_, 0, v_decompose_68_);
return v___f_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedAlgebra_ofAlgHom(lean_object* v_00_u03b9_70_, lean_object* v_R_71_, lean_object* v_A_72_, lean_object* v_inst_73_, lean_object* v_inst_74_, lean_object* v_inst_75_, lean_object* v_inst_76_, lean_object* v_inst_77_, lean_object* v_00_U0001d49c_78_, lean_object* v_inst_79_, lean_object* v_decompose_80_, lean_object* v_right__inv_81_, lean_object* v_left__inv_82_){
_start:
{
lean_object* v___f_83_; 
v___f_83_ = lean_alloc_closure((void*)(lp_mathlib_GradedAlgebra_ofAlgHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_83_, 0, v_decompose_80_);
return v___f_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedAlgebra_ofAlgHom___boxed(lean_object* v_00_u03b9_84_, lean_object* v_R_85_, lean_object* v_A_86_, lean_object* v_inst_87_, lean_object* v_inst_88_, lean_object* v_inst_89_, lean_object* v_inst_90_, lean_object* v_inst_91_, lean_object* v_00_U0001d49c_92_, lean_object* v_inst_93_, lean_object* v_decompose_94_, lean_object* v_right__inv_95_, lean_object* v_left__inv_96_){
_start:
{
lean_object* v_res_97_; 
v_res_97_ = lp_mathlib_GradedAlgebra_ofAlgHom(v_00_u03b9_84_, v_R_85_, v_A_86_, v_inst_87_, v_inst_88_, v_inst_89_, v_inst_90_, v_inst_91_, v_00_U0001d49c_92_, v_inst_93_, v_decompose_94_, v_right__inv_95_, v_left__inv_96_);
lean_dec_ref(v_00_U0001d49c_92_);
lean_dec_ref(v_inst_91_);
lean_dec_ref(v_inst_90_);
lean_dec_ref(v_inst_89_);
lean_dec_ref(v_inst_88_);
lean_dec_ref(v_inst_87_);
return v_res_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instGradedAlgebraRestrictScalars___redArg(lean_object* v_i_98_){
_start:
{
lean_inc_ref(v_i_98_);
return v_i_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instGradedAlgebraRestrictScalars___redArg___boxed(lean_object* v_i_99_){
_start:
{
lean_object* v_res_100_; 
v_res_100_ = lp_mathlib_instGradedAlgebraRestrictScalars___redArg(v_i_99_);
lean_dec_ref(v_i_99_);
return v_res_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instGradedAlgebraRestrictScalars(lean_object* v_00_u03b9_101_, lean_object* v_R_102_, lean_object* v_A_103_, lean_object* v_inst_104_, lean_object* v_inst_105_, lean_object* v_inst_106_, lean_object* v_inst_107_, lean_object* v_inst_108_, lean_object* v_00_U0001d49c_109_, lean_object* v_R_u2080_110_, lean_object* v_inst_111_, lean_object* v_inst_112_, lean_object* v_inst_113_, lean_object* v_inst_114_, lean_object* v_i_115_){
_start:
{
lean_inc_ref(v_i_115_);
return v_i_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instGradedAlgebraRestrictScalars___boxed(lean_object* v_00_u03b9_116_, lean_object* v_R_117_, lean_object* v_A_118_, lean_object* v_inst_119_, lean_object* v_inst_120_, lean_object* v_inst_121_, lean_object* v_inst_122_, lean_object* v_inst_123_, lean_object* v_00_U0001d49c_124_, lean_object* v_R_u2080_125_, lean_object* v_inst_126_, lean_object* v_inst_127_, lean_object* v_inst_128_, lean_object* v_inst_129_, lean_object* v_i_130_){
_start:
{
lean_object* v_res_131_; 
v_res_131_ = lp_mathlib_instGradedAlgebraRestrictScalars(v_00_u03b9_116_, v_R_117_, v_A_118_, v_inst_119_, v_inst_120_, v_inst_121_, v_inst_122_, v_inst_123_, v_00_U0001d49c_124_, v_R_u2080_125_, v_inst_126_, v_inst_127_, v_inst_128_, v_inst_129_, v_i_130_);
lean_dec_ref(v_i_130_);
lean_dec_ref(v_inst_128_);
lean_dec_ref(v_inst_127_);
lean_dec_ref(v_inst_126_);
lean_dec_ref(v_00_U0001d49c_124_);
lean_dec_ref(v_inst_123_);
lean_dec_ref(v_inst_122_);
lean_dec_ref(v_inst_121_);
lean_dec_ref(v_inst_120_);
lean_dec_ref(v_inst_119_);
return v_res_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decomposeAlgEquiv___redArg(lean_object* v_inst_132_, lean_object* v_inst_133_, lean_object* v_inst_134_){
_start:
{
lean_object* v_toAddCommMonoid_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; 
v_toAddCommMonoid_135_ = lean_ctor_get(v_inst_133_, 0);
lean_inc_ref(v_toAddCommMonoid_135_);
lean_dec_ref(v_inst_133_);
v___x_136_ = lp_mathlib_DirectSum_decomposeAddEquiv___redArg(v_inst_132_, v_toAddCommMonoid_135_, v_inst_134_);
v___x_137_ = lp_mathlib_Equiv_symm___redArg(v___x_136_);
v___x_138_ = lp_mathlib_Equiv_symm___redArg(v___x_137_);
return v___x_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decomposeAlgEquiv(lean_object* v_00_u03b9_139_, lean_object* v_R_140_, lean_object* v_A_141_, lean_object* v_inst_142_, lean_object* v_inst_143_, lean_object* v_inst_144_, lean_object* v_inst_145_, lean_object* v_inst_146_, lean_object* v_00_U0001d49c_147_, lean_object* v_inst_148_){
_start:
{
lean_object* v___x_149_; 
v___x_149_ = lp_mathlib_DirectSum_decomposeAlgEquiv___redArg(v_inst_142_, v_inst_145_, v_inst_148_);
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_decomposeAlgEquiv___boxed(lean_object* v_00_u03b9_150_, lean_object* v_R_151_, lean_object* v_A_152_, lean_object* v_inst_153_, lean_object* v_inst_154_, lean_object* v_inst_155_, lean_object* v_inst_156_, lean_object* v_inst_157_, lean_object* v_00_U0001d49c_158_, lean_object* v_inst_159_){
_start:
{
lean_object* v_res_160_; 
v_res_160_ = lp_mathlib_DirectSum_decomposeAlgEquiv(v_00_u03b9_150_, v_R_151_, v_A_152_, v_inst_153_, v_inst_154_, v_inst_155_, v_inst_156_, v_inst_157_, v_00_U0001d49c_158_, v_inst_159_);
lean_dec_ref(v_00_U0001d49c_158_);
lean_dec_ref(v_inst_157_);
lean_dec_ref(v_inst_155_);
lean_dec_ref(v_inst_154_);
return v_res_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedAlgebra_proj___redArg(lean_object* v_inst_162_, lean_object* v_inst_163_, lean_object* v_inst_164_, lean_object* v_i_165_){
_start:
{
lean_object* v___x_166_; lean_object* v_toFun_167_; lean_object* v___f_168_; lean_object* v___f_169_; lean_object* v___f_170_; lean_object* v___f_171_; lean_object* v___f_172_; 
v___x_166_ = lp_mathlib_DirectSum_decomposeAlgEquiv___redArg(v_inst_162_, v_inst_163_, v_inst_164_);
v_toFun_167_ = lean_ctor_get(v___x_166_, 0);
lean_inc(v_toFun_167_);
lean_dec_ref(v___x_166_);
v___f_168_ = ((lean_object*)(lp_mathlib_GradedAlgebra_proj___redArg___closed__0));
v___f_169_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_lapply___redArg___lam__0), 2, 1);
lean_closure_set(v___f_169_, 0, v_i_165_);
v___f_170_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_toLinearMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_170_, 0, v_toFun_167_);
v___f_171_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_171_, 0, v___f_170_);
lean_closure_set(v___f_171_, 1, v___f_169_);
v___f_172_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_172_, 0, v___f_171_);
lean_closure_set(v___f_172_, 1, v___f_168_);
return v___f_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedAlgebra_proj(lean_object* v_00_u03b9_173_, lean_object* v_R_174_, lean_object* v_A_175_, lean_object* v_inst_176_, lean_object* v_inst_177_, lean_object* v_inst_178_, lean_object* v_inst_179_, lean_object* v_inst_180_, lean_object* v_00_U0001d49c_181_, lean_object* v_inst_182_, lean_object* v_i_183_){
_start:
{
lean_object* v___x_184_; 
v___x_184_ = lp_mathlib_GradedAlgebra_proj___redArg(v_inst_176_, v_inst_179_, v_inst_182_, v_i_183_);
return v___x_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedAlgebra_proj___boxed(lean_object* v_00_u03b9_185_, lean_object* v_R_186_, lean_object* v_A_187_, lean_object* v_inst_188_, lean_object* v_inst_189_, lean_object* v_inst_190_, lean_object* v_inst_191_, lean_object* v_inst_192_, lean_object* v_00_U0001d49c_193_, lean_object* v_inst_194_, lean_object* v_i_195_){
_start:
{
lean_object* v_res_196_; 
v_res_196_ = lp_mathlib_GradedAlgebra_proj(v_00_u03b9_185_, v_R_186_, v_A_187_, v_inst_188_, v_inst_189_, v_inst_190_, v_inst_191_, v_inst_192_, v_00_U0001d49c_193_, v_inst_194_, v_i_195_);
lean_dec_ref(v_00_U0001d49c_193_);
lean_dec_ref(v_inst_192_);
lean_dec_ref(v_inst_190_);
lean_dec_ref(v_inst_189_);
return v_res_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedRing_projZeroRingHom___redArg___lam__0(lean_object* v_inst_197_, lean_object* v_toAddCommMonoid_198_, lean_object* v_inst_199_, lean_object* v_toZero_200_, lean_object* v_a_201_){
_start:
{
lean_object* v___x_202_; lean_object* v_toFun_203_; lean_object* v___x_204_; lean_object* v_toFun_205_; lean_object* v___x_206_; 
v___x_202_ = lp_mathlib_DirectSum_decompose___redArg(v_inst_197_, v_toAddCommMonoid_198_, v_inst_199_);
v_toFun_203_ = lean_ctor_get(v___x_202_, 0);
lean_inc(v_toFun_203_);
lean_dec_ref(v___x_202_);
v___x_204_ = lean_apply_1(v_toFun_203_, v_a_201_);
v_toFun_205_ = lean_ctor_get(v___x_204_, 0);
lean_inc(v_toFun_205_);
lean_dec_ref(v___x_204_);
v___x_206_ = lean_apply_1(v_toFun_205_, v_toZero_200_);
return v___x_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedRing_projZeroRingHom___redArg(lean_object* v_inst_207_, lean_object* v_inst_208_, lean_object* v_inst_209_, lean_object* v_inst_210_){
_start:
{
lean_object* v_toAddCommMonoid_211_; lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v_toZero_214_; lean_object* v___f_215_; 
v_toAddCommMonoid_211_ = lean_ctor_get(v_inst_207_, 0);
lean_inc_ref(v_toAddCommMonoid_211_);
lean_dec_ref(v_inst_207_);
v___x_212_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_209_);
v___x_213_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_212_);
v_toZero_214_ = lean_ctor_get(v___x_213_, 0);
lean_inc(v_toZero_214_);
lean_dec_ref(v___x_213_);
v___f_215_ = lean_alloc_closure((void*)(lp_mathlib_GradedRing_projZeroRingHom___redArg___lam__0), 5, 4);
lean_closure_set(v___f_215_, 0, v_inst_208_);
lean_closure_set(v___f_215_, 1, v_toAddCommMonoid_211_);
lean_closure_set(v___f_215_, 2, v_inst_210_);
lean_closure_set(v___f_215_, 3, v_toZero_214_);
return v___f_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedRing_projZeroRingHom___redArg___boxed(lean_object* v_inst_216_, lean_object* v_inst_217_, lean_object* v_inst_218_, lean_object* v_inst_219_){
_start:
{
lean_object* v_res_220_; 
v_res_220_ = lp_mathlib_GradedRing_projZeroRingHom___redArg(v_inst_216_, v_inst_217_, v_inst_218_, v_inst_219_);
lean_dec_ref(v_inst_218_);
return v_res_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedRing_projZeroRingHom(lean_object* v_00_u03b9_221_, lean_object* v_A_222_, lean_object* v_00_u03c3_223_, lean_object* v_inst_224_, lean_object* v_inst_225_, lean_object* v_inst_226_, lean_object* v_inst_227_, lean_object* v_inst_228_, lean_object* v_inst_229_, lean_object* v_inst_230_, lean_object* v_00_U0001d49c_231_, lean_object* v_inst_232_){
_start:
{
lean_object* v___x_233_; 
v___x_233_ = lp_mathlib_GradedRing_projZeroRingHom___redArg(v_inst_224_, v_inst_225_, v_inst_226_, v_inst_232_);
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedRing_projZeroRingHom___boxed(lean_object* v_00_u03b9_234_, lean_object* v_A_235_, lean_object* v_00_u03c3_236_, lean_object* v_inst_237_, lean_object* v_inst_238_, lean_object* v_inst_239_, lean_object* v_inst_240_, lean_object* v_inst_241_, lean_object* v_inst_242_, lean_object* v_inst_243_, lean_object* v_00_U0001d49c_244_, lean_object* v_inst_245_){
_start:
{
lean_object* v_res_246_; 
v_res_246_ = lp_mathlib_GradedRing_projZeroRingHom(v_00_u03b9_234_, v_A_235_, v_00_u03c3_236_, v_inst_237_, v_inst_238_, v_inst_239_, v_inst_240_, v_inst_241_, v_inst_242_, v_inst_243_, v_00_U0001d49c_244_, v_inst_245_);
lean_dec(v_00_U0001d49c_244_);
lean_dec_ref(v_inst_240_);
lean_dec_ref(v_inst_239_);
return v_res_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedRing_projZeroRingHom_x27___redArg(lean_object* v_inst_247_, lean_object* v_inst_248_, lean_object* v_inst_249_, lean_object* v_inst_250_){
_start:
{
lean_object* v___x_251_; lean_object* v___f_252_; 
v___x_251_ = lp_mathlib_GradedRing_projZeroRingHom___redArg(v_inst_247_, v_inst_248_, v_inst_249_, v_inst_250_);
v___f_252_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_252_, 0, v___x_251_);
return v___f_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedRing_projZeroRingHom_x27___redArg___boxed(lean_object* v_inst_253_, lean_object* v_inst_254_, lean_object* v_inst_255_, lean_object* v_inst_256_){
_start:
{
lean_object* v_res_257_; 
v_res_257_ = lp_mathlib_GradedRing_projZeroRingHom_x27___redArg(v_inst_253_, v_inst_254_, v_inst_255_, v_inst_256_);
lean_dec_ref(v_inst_255_);
return v_res_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedRing_projZeroRingHom_x27(lean_object* v_00_u03b9_258_, lean_object* v_A_259_, lean_object* v_00_u03c3_260_, lean_object* v_inst_261_, lean_object* v_inst_262_, lean_object* v_inst_263_, lean_object* v_inst_264_, lean_object* v_inst_265_, lean_object* v_inst_266_, lean_object* v_inst_267_, lean_object* v_00_U0001d49c_268_, lean_object* v_inst_269_){
_start:
{
lean_object* v___x_270_; 
v___x_270_ = lp_mathlib_GradedRing_projZeroRingHom_x27___redArg(v_inst_261_, v_inst_262_, v_inst_263_, v_inst_269_);
return v___x_270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GradedRing_projZeroRingHom_x27___boxed(lean_object* v_00_u03b9_271_, lean_object* v_A_272_, lean_object* v_00_u03c3_273_, lean_object* v_inst_274_, lean_object* v_inst_275_, lean_object* v_inst_276_, lean_object* v_inst_277_, lean_object* v_inst_278_, lean_object* v_inst_279_, lean_object* v_inst_280_, lean_object* v_00_U0001d49c_281_, lean_object* v_inst_282_){
_start:
{
lean_object* v_res_283_; 
v_res_283_ = lp_mathlib_GradedRing_projZeroRingHom_x27(v_00_u03b9_271_, v_A_272_, v_00_u03c3_273_, v_inst_274_, v_inst_275_, v_inst_276_, v_inst_277_, v_inst_278_, v_inst_279_, v_inst_280_, v_00_U0001d49c_281_, v_inst_282_);
lean_dec(v_00_U0001d49c_281_);
lean_dec_ref(v_inst_277_);
lean_dec_ref(v_inst_276_);
return v_res_283_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Algebra(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Decomposition(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Internal(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Ring(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_GradedAlgebra_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Algebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Decomposition(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Internal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_RingTheory_GradedAlgebra_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_DirectSum_Algebra(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_DirectSum_Decomposition(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_DirectSum_Internal(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_DirectSum_Ring(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_GradedAlgebra_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_DirectSum_Algebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_DirectSum_Decomposition(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_DirectSum_Internal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_DirectSum_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_GradedAlgebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_RingTheory_GradedAlgebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_RingTheory_GradedAlgebra_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
