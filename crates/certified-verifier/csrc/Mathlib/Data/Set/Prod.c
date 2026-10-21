// Lean compiler output
// Module: Mathlib.Data.Set.Prod
// Imports: public import Init public meta import Init public import Mathlib.Data.Set.Image public import Mathlib.Data.SProd public import Mathlib.Data.Sum.Basic
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
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemProd___aux__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemProd___aux__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemProd___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemProd___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemProd___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemProd___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemProd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemProd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemDiagonal___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemDiagonal___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemDiagonal(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemDiagonal___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Pullback_fst___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Pullback_fst___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Pullback_fst(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Pullback_fst___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Pullback_snd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Pullback_snd___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Pullback_snd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Pullback_snd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toPullbackDiag___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toPullbackDiag(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toPullbackDiag___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_mapPullback___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_mapPullback(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_mapPullback___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_PullbackSelf_map__fst___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_PullbackSelf_map__fst(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_PullbackSelf_map__snd___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_PullbackSelf_map__snd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemProd___aux__1___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_, lean_object* v_x_3_){
_start:
{
lean_object* v_fst_4_; lean_object* v_snd_5_; lean_object* v___x_6_; uint8_t v___x_7_; 
v_fst_4_ = lean_ctor_get(v_x_3_, 0);
lean_inc(v_fst_4_);
v_snd_5_ = lean_ctor_get(v_x_3_, 1);
lean_inc(v_snd_5_);
lean_dec_ref(v_x_3_);
v___x_6_ = lean_apply_1(v_inst_1_, v_fst_4_);
v___x_7_ = lean_unbox(v___x_6_);
if (v___x_7_ == 0)
{
uint8_t v___x_8_; 
lean_dec(v_snd_5_);
lean_dec_ref(v_inst_2_);
v___x_8_ = lean_unbox(v___x_6_);
return v___x_8_;
}
else
{
lean_object* v___x_9_; uint8_t v___x_10_; 
v___x_9_ = lean_apply_1(v_inst_2_, v_snd_5_);
v___x_10_ = lean_unbox(v___x_9_);
return v___x_10_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemProd___aux__1___redArg___boxed(lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_x_13_){
_start:
{
uint8_t v_res_14_; lean_object* v_r_15_; 
v_res_14_ = lp_mathlib_Set_decidableMemProd___aux__1___redArg(v_inst_11_, v_inst_12_, v_x_13_);
v_r_15_ = lean_box(v_res_14_);
return v_r_15_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemProd___aux__1(lean_object* v_00_u03b1_16_, lean_object* v_00_u03b2_17_, lean_object* v_s_18_, lean_object* v_t_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_x_22_){
_start:
{
uint8_t v___x_23_; 
v___x_23_ = lp_mathlib_Set_decidableMemProd___aux__1___redArg(v_inst_20_, v_inst_21_, v_x_22_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemProd___aux__1___boxed(lean_object* v_00_u03b1_24_, lean_object* v_00_u03b2_25_, lean_object* v_s_26_, lean_object* v_t_27_, lean_object* v_inst_28_, lean_object* v_inst_29_, lean_object* v_x_30_){
_start:
{
uint8_t v_res_31_; lean_object* v_r_32_; 
v_res_31_ = lp_mathlib_Set_decidableMemProd___aux__1(v_00_u03b1_24_, v_00_u03b2_25_, v_s_26_, v_t_27_, v_inst_28_, v_inst_29_, v_x_30_);
v_r_32_ = lean_box(v_res_31_);
return v_r_32_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemProd___redArg(lean_object* v_inst_33_, lean_object* v_inst_34_, lean_object* v_x_35_){
_start:
{
uint8_t v___x_36_; 
v___x_36_ = lp_mathlib_Set_decidableMemProd___aux__1___redArg(v_inst_33_, v_inst_34_, v_x_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemProd___redArg___boxed(lean_object* v_inst_37_, lean_object* v_inst_38_, lean_object* v_x_39_){
_start:
{
uint8_t v_res_40_; lean_object* v_r_41_; 
v_res_40_ = lp_mathlib_Set_decidableMemProd___redArg(v_inst_37_, v_inst_38_, v_x_39_);
v_r_41_ = lean_box(v_res_40_);
return v_r_41_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemProd(lean_object* v_00_u03b1_42_, lean_object* v_00_u03b2_43_, lean_object* v_s_44_, lean_object* v_t_45_, lean_object* v_inst_46_, lean_object* v_inst_47_, lean_object* v_x_48_){
_start:
{
uint8_t v___x_49_; 
v___x_49_ = lp_mathlib_Set_decidableMemProd___aux__1___redArg(v_inst_46_, v_inst_47_, v_x_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemProd___boxed(lean_object* v_00_u03b1_50_, lean_object* v_00_u03b2_51_, lean_object* v_s_52_, lean_object* v_t_53_, lean_object* v_inst_54_, lean_object* v_inst_55_, lean_object* v_x_56_){
_start:
{
uint8_t v_res_57_; lean_object* v_r_58_; 
v_res_57_ = lp_mathlib_Set_decidableMemProd(v_00_u03b1_50_, v_00_u03b2_51_, v_s_52_, v_t_53_, v_inst_54_, v_inst_55_, v_x_56_);
v_r_58_ = lean_box(v_res_57_);
return v_r_58_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemDiagonal___redArg(lean_object* v_h_59_, lean_object* v_x_60_){
_start:
{
lean_object* v_fst_61_; lean_object* v_snd_62_; lean_object* v___x_63_; uint8_t v___x_64_; 
v_fst_61_ = lean_ctor_get(v_x_60_, 0);
lean_inc(v_fst_61_);
v_snd_62_ = lean_ctor_get(v_x_60_, 1);
lean_inc(v_snd_62_);
lean_dec_ref(v_x_60_);
v___x_63_ = lean_apply_2(v_h_59_, v_fst_61_, v_snd_62_);
v___x_64_ = lean_unbox(v___x_63_);
return v___x_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemDiagonal___redArg___boxed(lean_object* v_h_65_, lean_object* v_x_66_){
_start:
{
uint8_t v_res_67_; lean_object* v_r_68_; 
v_res_67_ = lp_mathlib_Set_decidableMemDiagonal___redArg(v_h_65_, v_x_66_);
v_r_68_ = lean_box(v_res_67_);
return v_r_68_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemDiagonal(lean_object* v_00_u03b1_69_, lean_object* v_h_70_, lean_object* v_x_71_){
_start:
{
uint8_t v___x_72_; 
v___x_72_ = lp_mathlib_Set_decidableMemDiagonal___redArg(v_h_70_, v_x_71_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemDiagonal___boxed(lean_object* v_00_u03b1_73_, lean_object* v_h_74_, lean_object* v_x_75_){
_start:
{
uint8_t v_res_76_; lean_object* v_r_77_; 
v_res_76_ = lp_mathlib_Set_decidableMemDiagonal(v_00_u03b1_73_, v_h_74_, v_x_75_);
v_r_77_ = lean_box(v_res_76_);
return v_r_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Pullback_fst___redArg(lean_object* v_p_78_){
_start:
{
lean_object* v_fst_79_; 
v_fst_79_ = lean_ctor_get(v_p_78_, 0);
lean_inc(v_fst_79_);
return v_fst_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Pullback_fst___redArg___boxed(lean_object* v_p_80_){
_start:
{
lean_object* v_res_81_; 
v_res_81_ = lp_mathlib_Function_Pullback_fst___redArg(v_p_80_);
lean_dec_ref(v_p_80_);
return v_res_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Pullback_fst(lean_object* v_X_82_, lean_object* v_Y_83_, lean_object* v_Z_84_, lean_object* v_f_85_, lean_object* v_g_86_, lean_object* v_p_87_){
_start:
{
lean_object* v_fst_88_; 
v_fst_88_ = lean_ctor_get(v_p_87_, 0);
lean_inc(v_fst_88_);
return v_fst_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Pullback_fst___boxed(lean_object* v_X_89_, lean_object* v_Y_90_, lean_object* v_Z_91_, lean_object* v_f_92_, lean_object* v_g_93_, lean_object* v_p_94_){
_start:
{
lean_object* v_res_95_; 
v_res_95_ = lp_mathlib_Function_Pullback_fst(v_X_89_, v_Y_90_, v_Z_91_, v_f_92_, v_g_93_, v_p_94_);
lean_dec_ref(v_p_94_);
lean_dec(v_g_93_);
lean_dec(v_f_92_);
return v_res_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Pullback_snd___redArg(lean_object* v_p_96_){
_start:
{
lean_object* v_snd_97_; 
v_snd_97_ = lean_ctor_get(v_p_96_, 1);
lean_inc(v_snd_97_);
return v_snd_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Pullback_snd___redArg___boxed(lean_object* v_p_98_){
_start:
{
lean_object* v_res_99_; 
v_res_99_ = lp_mathlib_Function_Pullback_snd___redArg(v_p_98_);
lean_dec_ref(v_p_98_);
return v_res_99_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Pullback_snd(lean_object* v_X_100_, lean_object* v_Y_101_, lean_object* v_Z_102_, lean_object* v_f_103_, lean_object* v_g_104_, lean_object* v_p_105_){
_start:
{
lean_object* v_snd_106_; 
v_snd_106_ = lean_ctor_get(v_p_105_, 1);
lean_inc(v_snd_106_);
return v_snd_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Pullback_snd___boxed(lean_object* v_X_107_, lean_object* v_Y_108_, lean_object* v_Z_109_, lean_object* v_f_110_, lean_object* v_g_111_, lean_object* v_p_112_){
_start:
{
lean_object* v_res_113_; 
v_res_113_ = lp_mathlib_Function_Pullback_snd(v_X_107_, v_Y_108_, v_Z_109_, v_f_110_, v_g_111_, v_p_112_);
lean_dec_ref(v_p_112_);
lean_dec(v_g_111_);
lean_dec(v_f_110_);
return v_res_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toPullbackDiag___redArg(lean_object* v_x_114_){
_start:
{
lean_object* v___x_115_; 
lean_inc(v_x_114_);
v___x_115_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_115_, 0, v_x_114_);
lean_ctor_set(v___x_115_, 1, v_x_114_);
return v___x_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toPullbackDiag(lean_object* v_X_116_, lean_object* v_Y_117_, lean_object* v_f_118_, lean_object* v_x_119_){
_start:
{
lean_object* v___x_120_; 
lean_inc(v_x_119_);
v___x_120_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_120_, 0, v_x_119_);
lean_ctor_set(v___x_120_, 1, v_x_119_);
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toPullbackDiag___boxed(lean_object* v_X_121_, lean_object* v_Y_122_, lean_object* v_f_123_, lean_object* v_x_124_){
_start:
{
lean_object* v_res_125_; 
v_res_125_ = lp_mathlib_toPullbackDiag(v_X_121_, v_Y_122_, v_f_123_, v_x_124_);
lean_dec(v_f_123_);
return v_res_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_mapPullback___redArg(lean_object* v_mapX_126_, lean_object* v_mapZ_127_, lean_object* v_p_128_){
_start:
{
lean_object* v_fst_129_; lean_object* v_snd_130_; lean_object* v___x_132_; uint8_t v_isShared_133_; uint8_t v_isSharedCheck_139_; 
v_fst_129_ = lean_ctor_get(v_p_128_, 0);
v_snd_130_ = lean_ctor_get(v_p_128_, 1);
v_isSharedCheck_139_ = !lean_is_exclusive(v_p_128_);
if (v_isSharedCheck_139_ == 0)
{
v___x_132_ = v_p_128_;
v_isShared_133_ = v_isSharedCheck_139_;
goto v_resetjp_131_;
}
else
{
lean_inc(v_snd_130_);
lean_inc(v_fst_129_);
lean_dec(v_p_128_);
v___x_132_ = lean_box(0);
v_isShared_133_ = v_isSharedCheck_139_;
goto v_resetjp_131_;
}
v_resetjp_131_:
{
lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_137_; 
v___x_134_ = lean_apply_1(v_mapX_126_, v_fst_129_);
v___x_135_ = lean_apply_1(v_mapZ_127_, v_snd_130_);
if (v_isShared_133_ == 0)
{
lean_ctor_set(v___x_132_, 1, v___x_135_);
lean_ctor_set(v___x_132_, 0, v___x_134_);
v___x_137_ = v___x_132_;
goto v_reusejp_136_;
}
else
{
lean_object* v_reuseFailAlloc_138_; 
v_reuseFailAlloc_138_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_138_, 0, v___x_134_);
lean_ctor_set(v_reuseFailAlloc_138_, 1, v___x_135_);
v___x_137_ = v_reuseFailAlloc_138_;
goto v_reusejp_136_;
}
v_reusejp_136_:
{
return v___x_137_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_mapPullback(lean_object* v_X_u2081_140_, lean_object* v_X_u2082_141_, lean_object* v_Y_u2081_142_, lean_object* v_Y_u2082_143_, lean_object* v_Z_u2081_144_, lean_object* v_Z_u2082_145_, lean_object* v_f_u2081_146_, lean_object* v_g_u2081_147_, lean_object* v_f_u2082_148_, lean_object* v_g_u2082_149_, lean_object* v_mapX_150_, lean_object* v_mapY_151_, lean_object* v_mapZ_152_, lean_object* v_commX_153_, lean_object* v_commZ_154_, lean_object* v_p_155_){
_start:
{
lean_object* v___x_156_; 
v___x_156_ = lp_mathlib_Function_mapPullback___redArg(v_mapX_150_, v_mapZ_152_, v_p_155_);
return v___x_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_mapPullback___boxed(lean_object* v_X_u2081_157_, lean_object* v_X_u2082_158_, lean_object* v_Y_u2081_159_, lean_object* v_Y_u2082_160_, lean_object* v_Z_u2081_161_, lean_object* v_Z_u2082_162_, lean_object* v_f_u2081_163_, lean_object* v_g_u2081_164_, lean_object* v_f_u2082_165_, lean_object* v_g_u2082_166_, lean_object* v_mapX_167_, lean_object* v_mapY_168_, lean_object* v_mapZ_169_, lean_object* v_commX_170_, lean_object* v_commZ_171_, lean_object* v_p_172_){
_start:
{
lean_object* v_res_173_; 
v_res_173_ = lp_mathlib_Function_mapPullback(v_X_u2081_157_, v_X_u2082_158_, v_Y_u2081_159_, v_Y_u2082_160_, v_Z_u2081_161_, v_Z_u2082_162_, v_f_u2081_163_, v_g_u2081_164_, v_f_u2082_165_, v_g_u2082_166_, v_mapX_167_, v_mapY_168_, v_mapZ_169_, v_commX_170_, v_commZ_171_, v_p_172_);
lean_dec(v_mapY_168_);
lean_dec(v_g_u2082_166_);
lean_dec(v_f_u2082_165_);
lean_dec(v_g_u2081_164_);
lean_dec(v_f_u2081_163_);
return v_res_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_PullbackSelf_map__fst___redArg(lean_object* v_f_174_, lean_object* v_g_175_, lean_object* v_p_176_){
_start:
{
lean_object* v___x_177_; lean_object* v___x_178_; 
v___x_177_ = lean_alloc_closure((void*)(lp_mathlib_Function_Pullback_fst___boxed), 6, 5);
lean_closure_set(v___x_177_, 0, lean_box(0));
lean_closure_set(v___x_177_, 1, lean_box(0));
lean_closure_set(v___x_177_, 2, lean_box(0));
lean_closure_set(v___x_177_, 3, v_f_174_);
lean_closure_set(v___x_177_, 4, v_g_175_);
lean_inc_ref(v___x_177_);
v___x_178_ = lp_mathlib_Function_mapPullback___redArg(v___x_177_, v___x_177_, v_p_176_);
return v___x_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_PullbackSelf_map__fst(lean_object* v_X_179_, lean_object* v_Y_180_, lean_object* v_Z_181_, lean_object* v_f_182_, lean_object* v_g_183_, lean_object* v_p_184_){
_start:
{
lean_object* v___x_185_; 
v___x_185_ = lp_mathlib_Function_PullbackSelf_map__fst___redArg(v_f_182_, v_g_183_, v_p_184_);
return v___x_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_PullbackSelf_map__snd___redArg(lean_object* v_f_186_, lean_object* v_g_187_, lean_object* v_p_188_){
_start:
{
lean_object* v___x_189_; lean_object* v___x_190_; 
v___x_189_ = lean_alloc_closure((void*)(lp_mathlib_Function_Pullback_snd___boxed), 6, 5);
lean_closure_set(v___x_189_, 0, lean_box(0));
lean_closure_set(v___x_189_, 1, lean_box(0));
lean_closure_set(v___x_189_, 2, lean_box(0));
lean_closure_set(v___x_189_, 3, v_f_186_);
lean_closure_set(v___x_189_, 4, v_g_187_);
lean_inc_ref(v___x_189_);
v___x_190_ = lp_mathlib_Function_mapPullback___redArg(v___x_189_, v___x_189_, v_p_188_);
return v___x_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_PullbackSelf_map__snd(lean_object* v_X_191_, lean_object* v_Y_192_, lean_object* v_Z_193_, lean_object* v_f_194_, lean_object* v_g_195_, lean_object* v_p_196_){
_start:
{
lean_object* v___x_197_; 
v___x_197_ = lp_mathlib_Function_PullbackSelf_map__snd___redArg(v_f_194_, v_g_195_, v_p_196_);
return v___x_197_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Image(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_SProd(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Sum_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Prod(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_SProd(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Sum_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Set_Prod(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Set_Image(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_SProd(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Sum_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Set_Prod(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_SProd(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Sum_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Set_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Set_Prod(builtin);
}
#ifdef __cplusplus
}
#endif
