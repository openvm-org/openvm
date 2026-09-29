// Lean compiler output
// Module: Mathlib.GroupTheory.Congruence.Hom
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Hom.Defs public import Mathlib.GroupTheory.Congruence.Defs
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
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_Con_toQuotient___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_AddCon_toQuotient___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_mkMulHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_mkMulHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_mkAddHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_mkAddHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_ker(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_ker___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_ker(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_ker___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_mapGen(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_mapGen___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_mapGen(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_mapGen___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_mapOfSurjective(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_mapOfSurjective___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_mapOfSurjective(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_mapOfSurjective___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_correspondence___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Con_correspondence___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Con_correspondence___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Con_correspondence___closed__0 = (const lean_object*)&lp_mathlib_Con_correspondence___closed__0_value;
static const lean_ctor_object lp_mathlib_Con_correspondence___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Con_correspondence___closed__0_value),((lean_object*)&lp_mathlib_Con_correspondence___closed__0_value)}};
static const lean_object* lp_mathlib_Con_correspondence___closed__1 = (const lean_object*)&lp_mathlib_Con_correspondence___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Con_correspondence(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_correspondence___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_correspondence(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_correspondence___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_mk_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_mk_x27(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_mk_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_mk_x27(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_lift___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_lift___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_lift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_lift___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_lift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_kerLift___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_kerLift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_kerLift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_kerLift___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_kerLift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_kerLift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_mkMulHom___redArg(lean_object* v_inst_1_, lean_object* v_c_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_alloc_closure((void*)(lp_mathlib_Con_toQuotient___boxed), 4, 3);
lean_closure_set(v___x_3_, 0, lean_box(0));
lean_closure_set(v___x_3_, 1, v_inst_1_);
lean_closure_set(v___x_3_, 2, v_c_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_mkMulHom(lean_object* v_M_4_, lean_object* v_inst_5_, lean_object* v_c_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lean_alloc_closure((void*)(lp_mathlib_Con_toQuotient___boxed), 4, 3);
lean_closure_set(v___x_7_, 0, lean_box(0));
lean_closure_set(v___x_7_, 1, v_inst_5_);
lean_closure_set(v___x_7_, 2, v_c_6_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_mkAddHom___redArg(lean_object* v_inst_8_, lean_object* v_c_9_){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lean_alloc_closure((void*)(lp_mathlib_AddCon_toQuotient___boxed), 4, 3);
lean_closure_set(v___x_10_, 0, lean_box(0));
lean_closure_set(v___x_10_, 1, v_inst_8_);
lean_closure_set(v___x_10_, 2, v_c_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_mkAddHom(lean_object* v_M_11_, lean_object* v_inst_12_, lean_object* v_c_13_){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lean_alloc_closure((void*)(lp_mathlib_AddCon_toQuotient___boxed), 4, 3);
lean_closure_set(v___x_14_, 0, lean_box(0));
lean_closure_set(v___x_14_, 1, v_inst_12_);
lean_closure_set(v___x_14_, 2, v_c_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_ker(lean_object* v_M_15_, lean_object* v_N_16_, lean_object* v_F_17_, lean_object* v_inst_18_, lean_object* v_inst_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_f_22_){
_start:
{
lean_object* v___x_23_; 
v___x_23_ = lean_box(0);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_ker___boxed(lean_object* v_M_24_, lean_object* v_N_25_, lean_object* v_F_26_, lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_f_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_mathlib_Con_ker(v_M_24_, v_N_25_, v_F_26_, v_inst_27_, v_inst_28_, v_inst_29_, v_inst_30_, v_f_31_);
lean_dec(v_f_31_);
lean_dec(v_inst_29_);
lean_dec(v_inst_28_);
lean_dec(v_inst_27_);
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_ker(lean_object* v_M_33_, lean_object* v_N_34_, lean_object* v_F_35_, lean_object* v_inst_36_, lean_object* v_inst_37_, lean_object* v_inst_38_, lean_object* v_inst_39_, lean_object* v_f_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lean_box(0);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_ker___boxed(lean_object* v_M_42_, lean_object* v_N_43_, lean_object* v_F_44_, lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_f_49_){
_start:
{
lean_object* v_res_50_; 
v_res_50_ = lp_mathlib_AddCon_ker(v_M_42_, v_N_43_, v_F_44_, v_inst_45_, v_inst_46_, v_inst_47_, v_inst_48_, v_f_49_);
lean_dec(v_f_49_);
lean_dec(v_inst_47_);
lean_dec(v_inst_46_);
lean_dec(v_inst_45_);
return v_res_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_mapGen(lean_object* v_M_51_, lean_object* v_N_52_, lean_object* v_inst_53_, lean_object* v_inst_54_, lean_object* v_c_55_, lean_object* v_f_56_){
_start:
{
lean_object* v___x_57_; 
v___x_57_ = lean_box(0);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_mapGen___boxed(lean_object* v_M_58_, lean_object* v_N_59_, lean_object* v_inst_60_, lean_object* v_inst_61_, lean_object* v_c_62_, lean_object* v_f_63_){
_start:
{
lean_object* v_res_64_; 
v_res_64_ = lp_mathlib_Con_mapGen(v_M_58_, v_N_59_, v_inst_60_, v_inst_61_, v_c_62_, v_f_63_);
lean_dec(v_f_63_);
lean_dec(v_inst_61_);
lean_dec(v_inst_60_);
return v_res_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_mapGen(lean_object* v_M_65_, lean_object* v_N_66_, lean_object* v_inst_67_, lean_object* v_inst_68_, lean_object* v_c_69_, lean_object* v_f_70_){
_start:
{
lean_object* v___x_71_; 
v___x_71_ = lean_box(0);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_mapGen___boxed(lean_object* v_M_72_, lean_object* v_N_73_, lean_object* v_inst_74_, lean_object* v_inst_75_, lean_object* v_c_76_, lean_object* v_f_77_){
_start:
{
lean_object* v_res_78_; 
v_res_78_ = lp_mathlib_AddCon_mapGen(v_M_72_, v_N_73_, v_inst_74_, v_inst_75_, v_c_76_, v_f_77_);
lean_dec(v_f_77_);
lean_dec(v_inst_75_);
lean_dec(v_inst_74_);
return v_res_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_mapOfSurjective(lean_object* v_M_79_, lean_object* v_N_80_, lean_object* v_F_81_, lean_object* v_inst_82_, lean_object* v_inst_83_, lean_object* v_inst_84_, lean_object* v_inst_85_, lean_object* v_c_86_, lean_object* v_f_87_, lean_object* v_h_88_, lean_object* v_hf_89_){
_start:
{
lean_object* v___x_90_; 
v___x_90_ = lean_box(0);
return v___x_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_mapOfSurjective___boxed(lean_object* v_M_91_, lean_object* v_N_92_, lean_object* v_F_93_, lean_object* v_inst_94_, lean_object* v_inst_95_, lean_object* v_inst_96_, lean_object* v_inst_97_, lean_object* v_c_98_, lean_object* v_f_99_, lean_object* v_h_100_, lean_object* v_hf_101_){
_start:
{
lean_object* v_res_102_; 
v_res_102_ = lp_mathlib_Con_mapOfSurjective(v_M_91_, v_N_92_, v_F_93_, v_inst_94_, v_inst_95_, v_inst_96_, v_inst_97_, v_c_98_, v_f_99_, v_h_100_, v_hf_101_);
lean_dec(v_f_99_);
lean_dec(v_inst_96_);
lean_dec(v_inst_95_);
lean_dec(v_inst_94_);
return v_res_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_mapOfSurjective(lean_object* v_M_103_, lean_object* v_N_104_, lean_object* v_F_105_, lean_object* v_inst_106_, lean_object* v_inst_107_, lean_object* v_inst_108_, lean_object* v_inst_109_, lean_object* v_c_110_, lean_object* v_f_111_, lean_object* v_h_112_, lean_object* v_hf_113_){
_start:
{
lean_object* v___x_114_; 
v___x_114_ = lean_box(0);
return v___x_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_mapOfSurjective___boxed(lean_object* v_M_115_, lean_object* v_N_116_, lean_object* v_F_117_, lean_object* v_inst_118_, lean_object* v_inst_119_, lean_object* v_inst_120_, lean_object* v_inst_121_, lean_object* v_c_122_, lean_object* v_f_123_, lean_object* v_h_124_, lean_object* v_hf_125_){
_start:
{
lean_object* v_res_126_; 
v_res_126_ = lp_mathlib_AddCon_mapOfSurjective(v_M_115_, v_N_116_, v_F_117_, v_inst_118_, v_inst_119_, v_inst_120_, v_inst_121_, v_c_122_, v_f_123_, v_h_124_, v_hf_125_);
lean_dec(v_f_123_);
lean_dec(v_inst_120_);
lean_dec(v_inst_119_);
lean_dec(v_inst_118_);
return v_res_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_correspondence___lam__0(lean_object* v_d_127_){
_start:
{
lean_object* v___x_128_; 
v___x_128_ = lean_box(0);
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_correspondence(lean_object* v_M_132_, lean_object* v_inst_133_, lean_object* v_c_134_){
_start:
{
lean_object* v___x_135_; 
v___x_135_ = ((lean_object*)(lp_mathlib_Con_correspondence___closed__1));
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_correspondence___boxed(lean_object* v_M_136_, lean_object* v_inst_137_, lean_object* v_c_138_){
_start:
{
lean_object* v_res_139_; 
v_res_139_ = lp_mathlib_Con_correspondence(v_M_136_, v_inst_137_, v_c_138_);
lean_dec(v_inst_137_);
return v_res_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_correspondence(lean_object* v_M_140_, lean_object* v_inst_141_, lean_object* v_c_142_){
_start:
{
lean_object* v___x_143_; 
v___x_143_ = ((lean_object*)(lp_mathlib_Con_correspondence___closed__1));
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_correspondence___boxed(lean_object* v_M_144_, lean_object* v_inst_145_, lean_object* v_c_146_){
_start:
{
lean_object* v_res_147_; 
v_res_147_ = lp_mathlib_AddCon_correspondence(v_M_144_, v_inst_145_, v_c_146_);
lean_dec(v_inst_145_);
return v_res_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_mk_x27___redArg(lean_object* v_inst_148_, lean_object* v_c_149_){
_start:
{
lean_object* v___x_150_; lean_object* v_toMul_151_; lean_object* v___x_152_; 
v___x_150_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_148_);
v_toMul_151_ = lean_ctor_get(v___x_150_, 1);
lean_inc(v_toMul_151_);
lean_dec_ref(v___x_150_);
v___x_152_ = lean_alloc_closure((void*)(lp_mathlib_Con_toQuotient___boxed), 4, 3);
lean_closure_set(v___x_152_, 0, lean_box(0));
lean_closure_set(v___x_152_, 1, v_toMul_151_);
lean_closure_set(v___x_152_, 2, v_c_149_);
return v___x_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_mk_x27(lean_object* v_M_153_, lean_object* v_inst_154_, lean_object* v_c_155_){
_start:
{
lean_object* v___x_156_; 
v___x_156_ = lp_mathlib_Con_mk_x27___redArg(v_inst_154_, v_c_155_);
return v___x_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_mk_x27___redArg(lean_object* v_inst_157_, lean_object* v_c_158_){
_start:
{
lean_object* v___x_159_; lean_object* v_toAdd_160_; lean_object* v___x_161_; 
v___x_159_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_157_);
v_toAdd_160_ = lean_ctor_get(v___x_159_, 1);
lean_inc(v_toAdd_160_);
lean_dec_ref(v___x_159_);
v___x_161_ = lean_alloc_closure((void*)(lp_mathlib_AddCon_toQuotient___boxed), 4, 3);
lean_closure_set(v___x_161_, 0, lean_box(0));
lean_closure_set(v___x_161_, 1, v_toAdd_160_);
lean_closure_set(v___x_161_, 2, v_c_158_);
return v___x_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_mk_x27(lean_object* v_M_162_, lean_object* v_inst_163_, lean_object* v_c_164_){
_start:
{
lean_object* v___x_165_; 
v___x_165_ = lp_mathlib_AddCon_mk_x27___redArg(v_inst_163_, v_c_164_);
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_lift___redArg___lam__0(lean_object* v_f_166_, lean_object* v_x_167_){
_start:
{
lean_object* v___x_168_; 
v___x_168_ = lean_apply_1(v_f_166_, v_x_167_);
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_lift___redArg(lean_object* v_f_169_){
_start:
{
lean_object* v___f_170_; 
v___f_170_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___redArg___lam__0), 2, 1);
lean_closure_set(v___f_170_, 0, v_f_169_);
return v___f_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_lift(lean_object* v_M_171_, lean_object* v_P_172_, lean_object* v_inst_173_, lean_object* v_inst_174_, lean_object* v_c_175_, lean_object* v_f_176_, lean_object* v_H_177_){
_start:
{
lean_object* v___f_178_; 
v___f_178_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___redArg___lam__0), 2, 1);
lean_closure_set(v___f_178_, 0, v_f_176_);
return v___f_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_lift___boxed(lean_object* v_M_179_, lean_object* v_P_180_, lean_object* v_inst_181_, lean_object* v_inst_182_, lean_object* v_c_183_, lean_object* v_f_184_, lean_object* v_H_185_){
_start:
{
lean_object* v_res_186_; 
v_res_186_ = lp_mathlib_Con_lift(v_M_179_, v_P_180_, v_inst_181_, v_inst_182_, v_c_183_, v_f_184_, v_H_185_);
lean_dec_ref(v_inst_182_);
lean_dec_ref(v_inst_181_);
return v_res_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_lift___redArg(lean_object* v_f_187_){
_start:
{
lean_object* v___f_188_; 
v___f_188_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___redArg___lam__0), 2, 1);
lean_closure_set(v___f_188_, 0, v_f_187_);
return v___f_188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_lift(lean_object* v_M_189_, lean_object* v_P_190_, lean_object* v_inst_191_, lean_object* v_inst_192_, lean_object* v_c_193_, lean_object* v_f_194_, lean_object* v_H_195_){
_start:
{
lean_object* v___f_196_; 
v___f_196_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___redArg___lam__0), 2, 1);
lean_closure_set(v___f_196_, 0, v_f_194_);
return v___f_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_lift___boxed(lean_object* v_M_197_, lean_object* v_P_198_, lean_object* v_inst_199_, lean_object* v_inst_200_, lean_object* v_c_201_, lean_object* v_f_202_, lean_object* v_H_203_){
_start:
{
lean_object* v_res_204_; 
v_res_204_ = lp_mathlib_AddCon_lift(v_M_197_, v_P_198_, v_inst_199_, v_inst_200_, v_c_201_, v_f_202_, v_H_203_);
lean_dec_ref(v_inst_200_);
lean_dec_ref(v_inst_199_);
return v_res_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_kerLift___redArg(lean_object* v_f_205_){
_start:
{
lean_object* v___f_206_; 
v___f_206_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___redArg___lam__0), 2, 1);
lean_closure_set(v___f_206_, 0, v_f_205_);
return v___f_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_kerLift(lean_object* v_M_207_, lean_object* v_P_208_, lean_object* v_inst_209_, lean_object* v_inst_210_, lean_object* v_f_211_){
_start:
{
lean_object* v___f_212_; 
v___f_212_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___redArg___lam__0), 2, 1);
lean_closure_set(v___f_212_, 0, v_f_211_);
return v___f_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_kerLift___boxed(lean_object* v_M_213_, lean_object* v_P_214_, lean_object* v_inst_215_, lean_object* v_inst_216_, lean_object* v_f_217_){
_start:
{
lean_object* v_res_218_; 
v_res_218_ = lp_mathlib_Con_kerLift(v_M_213_, v_P_214_, v_inst_215_, v_inst_216_, v_f_217_);
lean_dec_ref(v_inst_216_);
lean_dec_ref(v_inst_215_);
return v_res_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_kerLift___redArg(lean_object* v_f_219_){
_start:
{
lean_object* v___f_220_; 
v___f_220_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___redArg___lam__0), 2, 1);
lean_closure_set(v___f_220_, 0, v_f_219_);
return v___f_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_kerLift(lean_object* v_M_221_, lean_object* v_P_222_, lean_object* v_inst_223_, lean_object* v_inst_224_, lean_object* v_f_225_){
_start:
{
lean_object* v___f_226_; 
v___f_226_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___redArg___lam__0), 2, 1);
lean_closure_set(v___f_226_, 0, v_f_225_);
return v___f_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_kerLift___boxed(lean_object* v_M_227_, lean_object* v_P_228_, lean_object* v_inst_229_, lean_object* v_inst_230_, lean_object* v_f_231_){
_start:
{
lean_object* v_res_232_; 
v_res_232_ = lp_mathlib_AddCon_kerLift(v_M_227_, v_P_228_, v_inst_229_, v_inst_230_, v_f_231_);
lean_dec_ref(v_inst_230_);
lean_dec_ref(v_inst_229_);
return v_res_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_map___redArg(lean_object* v_inst_233_, lean_object* v_d_234_){
_start:
{
lean_object* v___x_235_; lean_object* v___f_236_; 
v___x_235_ = lp_mathlib_Con_mk_x27___redArg(v_inst_233_, v_d_234_);
v___f_236_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___redArg___lam__0), 2, 1);
lean_closure_set(v___f_236_, 0, v___x_235_);
return v___f_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_map(lean_object* v_M_237_, lean_object* v_inst_238_, lean_object* v_c_239_, lean_object* v_d_240_, lean_object* v_h_241_){
_start:
{
lean_object* v___x_242_; 
v___x_242_ = lp_mathlib_Con_map___redArg(v_inst_238_, v_d_240_);
return v___x_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_map___redArg(lean_object* v_inst_243_, lean_object* v_d_244_){
_start:
{
lean_object* v___x_245_; lean_object* v___f_246_; 
v___x_245_ = lp_mathlib_AddCon_mk_x27___redArg(v_inst_243_, v_d_244_);
v___f_246_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___redArg___lam__0), 2, 1);
lean_closure_set(v___f_246_, 0, v___x_245_);
return v___f_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_map(lean_object* v_M_247_, lean_object* v_inst_248_, lean_object* v_c_249_, lean_object* v_d_250_, lean_object* v_h_251_){
_start:
{
lean_object* v___x_252_; 
v___x_252_ = lp_mathlib_AddCon_map___redArg(v_inst_248_, v_d_250_);
return v___x_252_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Congruence_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Congruence_Hom(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Congruence_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_GroupTheory_Congruence_Hom(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Congruence_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_Congruence_Hom(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Congruence_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Congruence_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_GroupTheory_Congruence_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_GroupTheory_Congruence_Hom(builtin);
}
#ifdef __cplusplus
}
#endif
