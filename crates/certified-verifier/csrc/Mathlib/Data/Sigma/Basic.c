// Lean compiler output
// Module: Mathlib.Data.Sigma.Basic
// Imports: public import Init public meta import Init public import Mathlib.Logic.Function.Defs public import Mathlib.Logic.Function.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_Sigma_instInhabitedSigma___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sigma_instInhabitedSigma(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Sigma_instDecidableEqSigma___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sigma_instDecidableEqSigma___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Sigma_instDecidableEqSigma(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sigma_instDecidableEqSigma___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sigma_map___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sigma_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sigma_curry___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sigma_curry(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sigma_uncurry___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sigma_uncurry(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_toSigma___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_toSigma(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSigma_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSigma_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSigma_instInhabitedOfDefault__mathlib___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSigma_instInhabitedOfDefault__mathlib(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_PSigma_decidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSigma_decidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_PSigma_decidableEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSigma_decidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSigma_map___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PSigma_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sigma_instInhabitedSigma___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3_, 0, v_inst_1_);
lean_ctor_set(v___x_3_, 1, v_inst_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sigma_instInhabitedSigma(lean_object* v_00_u03b1_4_, lean_object* v_00_u03b2_5_, lean_object* v_inst_6_, lean_object* v_inst_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_8_, 0, v_inst_6_);
lean_ctor_set(v___x_8_, 1, v_inst_7_);
return v___x_8_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Sigma_instDecidableEqSigma___redArg(lean_object* v_h_u2081_9_, lean_object* v_h_u2082_10_, lean_object* v_x_11_, lean_object* v_x_12_){
_start:
{
lean_object* v_fst_13_; lean_object* v_snd_14_; lean_object* v_fst_15_; lean_object* v_snd_16_; lean_object* v___x_17_; uint8_t v___x_18_; 
v_fst_13_ = lean_ctor_get(v_x_11_, 0);
lean_inc_n(v_fst_13_, 2);
v_snd_14_ = lean_ctor_get(v_x_11_, 1);
lean_inc(v_snd_14_);
lean_dec_ref(v_x_11_);
v_fst_15_ = lean_ctor_get(v_x_12_, 0);
lean_inc(v_fst_15_);
v_snd_16_ = lean_ctor_get(v_x_12_, 1);
lean_inc(v_snd_16_);
lean_dec_ref(v_x_12_);
v___x_17_ = lean_apply_2(v_h_u2081_9_, v_fst_13_, v_fst_15_);
v___x_18_ = lean_unbox(v___x_17_);
if (v___x_18_ == 0)
{
uint8_t v___x_19_; 
lean_dec(v_snd_16_);
lean_dec(v_snd_14_);
lean_dec(v_fst_13_);
lean_dec_ref(v_h_u2082_10_);
v___x_19_ = lean_unbox(v___x_17_);
return v___x_19_;
}
else
{
lean_object* v___x_20_; uint8_t v___x_21_; 
v___x_20_ = lean_apply_3(v_h_u2082_10_, v_fst_13_, v_snd_14_, v_snd_16_);
v___x_21_ = lean_unbox(v___x_20_);
return v___x_21_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sigma_instDecidableEqSigma___redArg___boxed(lean_object* v_h_u2081_22_, lean_object* v_h_u2082_23_, lean_object* v_x_24_, lean_object* v_x_25_){
_start:
{
uint8_t v_res_26_; lean_object* v_r_27_; 
v_res_26_ = lp_mathlib_Sigma_instDecidableEqSigma___redArg(v_h_u2081_22_, v_h_u2082_23_, v_x_24_, v_x_25_);
v_r_27_ = lean_box(v_res_26_);
return v_r_27_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Sigma_instDecidableEqSigma(lean_object* v_00_u03b1_28_, lean_object* v_00_u03b2_29_, lean_object* v_h_u2081_30_, lean_object* v_h_u2082_31_, lean_object* v_x_32_, lean_object* v_x_33_){
_start:
{
uint8_t v___x_34_; 
v___x_34_ = lp_mathlib_Sigma_instDecidableEqSigma___redArg(v_h_u2081_30_, v_h_u2082_31_, v_x_32_, v_x_33_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sigma_instDecidableEqSigma___boxed(lean_object* v_00_u03b1_35_, lean_object* v_00_u03b2_36_, lean_object* v_h_u2081_37_, lean_object* v_h_u2082_38_, lean_object* v_x_39_, lean_object* v_x_40_){
_start:
{
uint8_t v_res_41_; lean_object* v_r_42_; 
v_res_41_ = lp_mathlib_Sigma_instDecidableEqSigma(v_00_u03b1_35_, v_00_u03b2_36_, v_h_u2081_37_, v_h_u2082_38_, v_x_39_, v_x_40_);
v_r_42_ = lean_box(v_res_41_);
return v_r_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sigma_map___redArg(lean_object* v_f_u2081_43_, lean_object* v_f_u2082_44_, lean_object* v_x_45_){
_start:
{
lean_object* v_fst_46_; lean_object* v_snd_47_; lean_object* v___x_49_; uint8_t v_isShared_50_; uint8_t v_isSharedCheck_56_; 
v_fst_46_ = lean_ctor_get(v_x_45_, 0);
v_snd_47_ = lean_ctor_get(v_x_45_, 1);
v_isSharedCheck_56_ = !lean_is_exclusive(v_x_45_);
if (v_isSharedCheck_56_ == 0)
{
v___x_49_ = v_x_45_;
v_isShared_50_ = v_isSharedCheck_56_;
goto v_resetjp_48_;
}
else
{
lean_inc(v_snd_47_);
lean_inc(v_fst_46_);
lean_dec(v_x_45_);
v___x_49_ = lean_box(0);
v_isShared_50_ = v_isSharedCheck_56_;
goto v_resetjp_48_;
}
v_resetjp_48_:
{
lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_54_; 
lean_inc(v_fst_46_);
v___x_51_ = lean_apply_1(v_f_u2081_43_, v_fst_46_);
v___x_52_ = lean_apply_2(v_f_u2082_44_, v_fst_46_, v_snd_47_);
if (v_isShared_50_ == 0)
{
lean_ctor_set(v___x_49_, 1, v___x_52_);
lean_ctor_set(v___x_49_, 0, v___x_51_);
v___x_54_ = v___x_49_;
goto v_reusejp_53_;
}
else
{
lean_object* v_reuseFailAlloc_55_; 
v_reuseFailAlloc_55_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_55_, 0, v___x_51_);
lean_ctor_set(v_reuseFailAlloc_55_, 1, v___x_52_);
v___x_54_ = v_reuseFailAlloc_55_;
goto v_reusejp_53_;
}
v_reusejp_53_:
{
return v___x_54_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sigma_map(lean_object* v_00_u03b1_u2081_57_, lean_object* v_00_u03b1_u2082_58_, lean_object* v_00_u03b2_u2081_59_, lean_object* v_00_u03b2_u2082_60_, lean_object* v_f_u2081_61_, lean_object* v_f_u2082_62_, lean_object* v_x_63_){
_start:
{
lean_object* v___x_64_; 
v___x_64_ = lp_mathlib_Sigma_map___redArg(v_f_u2081_61_, v_f_u2082_62_, v_x_63_);
return v___x_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sigma_curry___redArg(lean_object* v_f_65_, lean_object* v_x_66_, lean_object* v_y_67_){
_start:
{
lean_object* v___x_68_; lean_object* v___x_69_; 
v___x_68_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_68_, 0, v_x_66_);
lean_ctor_set(v___x_68_, 1, v_y_67_);
v___x_69_ = lean_apply_1(v_f_65_, v___x_68_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sigma_curry(lean_object* v_00_u03b1_70_, lean_object* v_00_u03b2_71_, lean_object* v_00_u03b3_72_, lean_object* v_f_73_, lean_object* v_x_74_, lean_object* v_y_75_){
_start:
{
lean_object* v___x_76_; 
v___x_76_ = lp_mathlib_Sigma_curry___redArg(v_f_73_, v_x_74_, v_y_75_);
return v___x_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sigma_uncurry___redArg(lean_object* v_f_77_, lean_object* v_x_78_){
_start:
{
lean_object* v_fst_79_; lean_object* v_snd_80_; lean_object* v___x_81_; 
v_fst_79_ = lean_ctor_get(v_x_78_, 0);
lean_inc(v_fst_79_);
v_snd_80_ = lean_ctor_get(v_x_78_, 1);
lean_inc(v_snd_80_);
lean_dec_ref(v_x_78_);
v___x_81_ = lean_apply_2(v_f_77_, v_fst_79_, v_snd_80_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sigma_uncurry(lean_object* v_00_u03b1_82_, lean_object* v_00_u03b2_83_, lean_object* v_00_u03b3_84_, lean_object* v_f_85_, lean_object* v_x_86_){
_start:
{
lean_object* v___x_87_; 
v___x_87_ = lp_mathlib_Sigma_uncurry___redArg(v_f_85_, v_x_86_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_toSigma___redArg(lean_object* v_p_88_){
_start:
{
lean_object* v_fst_89_; lean_object* v_snd_90_; lean_object* v___x_92_; uint8_t v_isShared_93_; uint8_t v_isSharedCheck_97_; 
v_fst_89_ = lean_ctor_get(v_p_88_, 0);
v_snd_90_ = lean_ctor_get(v_p_88_, 1);
v_isSharedCheck_97_ = !lean_is_exclusive(v_p_88_);
if (v_isSharedCheck_97_ == 0)
{
v___x_92_ = v_p_88_;
v_isShared_93_ = v_isSharedCheck_97_;
goto v_resetjp_91_;
}
else
{
lean_inc(v_snd_90_);
lean_inc(v_fst_89_);
lean_dec(v_p_88_);
v___x_92_ = lean_box(0);
v_isShared_93_ = v_isSharedCheck_97_;
goto v_resetjp_91_;
}
v_resetjp_91_:
{
lean_object* v___x_95_; 
if (v_isShared_93_ == 0)
{
v___x_95_ = v___x_92_;
goto v_reusejp_94_;
}
else
{
lean_object* v_reuseFailAlloc_96_; 
v_reuseFailAlloc_96_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_96_, 0, v_fst_89_);
lean_ctor_set(v_reuseFailAlloc_96_, 1, v_snd_90_);
v___x_95_ = v_reuseFailAlloc_96_;
goto v_reusejp_94_;
}
v_reusejp_94_:
{
return v___x_95_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_toSigma(lean_object* v_00_u03b1_98_, lean_object* v_00_u03b2_99_, lean_object* v_p_100_){
_start:
{
lean_object* v___x_101_; 
v___x_101_ = lp_mathlib_Prod_toSigma___redArg(v_p_100_);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSigma_elim___redArg(lean_object* v_f_102_, lean_object* v_a_103_){
_start:
{
lean_object* v_a_104_; lean_object* v_a_105_; lean_object* v___x_106_; 
v_a_104_ = lean_ctor_get(v_a_103_, 0);
lean_inc(v_a_104_);
v_a_105_ = lean_ctor_get(v_a_103_, 1);
lean_inc(v_a_105_);
lean_dec_ref(v_a_103_);
v___x_106_ = lean_apply_2(v_f_102_, v_a_104_, v_a_105_);
return v___x_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSigma_elim(lean_object* v_00_u03b1_107_, lean_object* v_00_u03b2_108_, lean_object* v_00_u03b3_109_, lean_object* v_f_110_, lean_object* v_a_111_){
_start:
{
lean_object* v___x_112_; 
v___x_112_ = lp_mathlib_PSigma_elim___redArg(v_f_110_, v_a_111_);
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSigma_instInhabitedOfDefault__mathlib___redArg(lean_object* v_inst_113_, lean_object* v_inst_114_){
_start:
{
lean_object* v___x_115_; 
v___x_115_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_115_, 0, v_inst_113_);
lean_ctor_set(v___x_115_, 1, v_inst_114_);
return v___x_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSigma_instInhabitedOfDefault__mathlib(lean_object* v_00_u03b1_116_, lean_object* v_00_u03b2_117_, lean_object* v_inst_118_, lean_object* v_inst_119_){
_start:
{
lean_object* v___x_120_; 
v___x_120_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_120_, 0, v_inst_118_);
lean_ctor_set(v___x_120_, 1, v_inst_119_);
return v___x_120_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_PSigma_decidableEq___redArg(lean_object* v_h_u2081_121_, lean_object* v_h_u2082_122_, lean_object* v_x_123_, lean_object* v_x_124_){
_start:
{
lean_object* v_fst_125_; lean_object* v_snd_126_; lean_object* v_fst_127_; lean_object* v_snd_128_; lean_object* v___x_129_; uint8_t v___x_130_; 
v_fst_125_ = lean_ctor_get(v_x_123_, 0);
lean_inc_n(v_fst_125_, 2);
v_snd_126_ = lean_ctor_get(v_x_123_, 1);
lean_inc(v_snd_126_);
lean_dec_ref(v_x_123_);
v_fst_127_ = lean_ctor_get(v_x_124_, 0);
lean_inc(v_fst_127_);
v_snd_128_ = lean_ctor_get(v_x_124_, 1);
lean_inc(v_snd_128_);
lean_dec_ref(v_x_124_);
v___x_129_ = lean_apply_2(v_h_u2081_121_, v_fst_125_, v_fst_127_);
v___x_130_ = lean_unbox(v___x_129_);
if (v___x_130_ == 0)
{
uint8_t v___x_131_; 
lean_dec(v_snd_128_);
lean_dec(v_snd_126_);
lean_dec(v_fst_125_);
lean_dec_ref(v_h_u2082_122_);
v___x_131_ = lean_unbox(v___x_129_);
return v___x_131_;
}
else
{
lean_object* v___x_132_; uint8_t v___x_133_; 
v___x_132_ = lean_apply_3(v_h_u2082_122_, v_fst_125_, v_snd_126_, v_snd_128_);
v___x_133_ = lean_unbox(v___x_132_);
return v___x_133_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSigma_decidableEq___redArg___boxed(lean_object* v_h_u2081_134_, lean_object* v_h_u2082_135_, lean_object* v_x_136_, lean_object* v_x_137_){
_start:
{
uint8_t v_res_138_; lean_object* v_r_139_; 
v_res_138_ = lp_mathlib_PSigma_decidableEq___redArg(v_h_u2081_134_, v_h_u2082_135_, v_x_136_, v_x_137_);
v_r_139_ = lean_box(v_res_138_);
return v_r_139_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_PSigma_decidableEq(lean_object* v_00_u03b1_140_, lean_object* v_00_u03b2_141_, lean_object* v_h_u2081_142_, lean_object* v_h_u2082_143_, lean_object* v_x_144_, lean_object* v_x_145_){
_start:
{
uint8_t v___x_146_; 
v___x_146_ = lp_mathlib_PSigma_decidableEq___redArg(v_h_u2081_142_, v_h_u2082_143_, v_x_144_, v_x_145_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSigma_decidableEq___boxed(lean_object* v_00_u03b1_147_, lean_object* v_00_u03b2_148_, lean_object* v_h_u2081_149_, lean_object* v_h_u2082_150_, lean_object* v_x_151_, lean_object* v_x_152_){
_start:
{
uint8_t v_res_153_; lean_object* v_r_154_; 
v_res_153_ = lp_mathlib_PSigma_decidableEq(v_00_u03b1_147_, v_00_u03b2_148_, v_h_u2081_149_, v_h_u2082_150_, v_x_151_, v_x_152_);
v_r_154_ = lean_box(v_res_153_);
return v_r_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSigma_map___redArg(lean_object* v_f_u2081_155_, lean_object* v_f_u2082_156_, lean_object* v_x_157_){
_start:
{
lean_object* v_fst_158_; lean_object* v_snd_159_; lean_object* v___x_161_; uint8_t v_isShared_162_; uint8_t v_isSharedCheck_168_; 
v_fst_158_ = lean_ctor_get(v_x_157_, 0);
v_snd_159_ = lean_ctor_get(v_x_157_, 1);
v_isSharedCheck_168_ = !lean_is_exclusive(v_x_157_);
if (v_isSharedCheck_168_ == 0)
{
v___x_161_ = v_x_157_;
v_isShared_162_ = v_isSharedCheck_168_;
goto v_resetjp_160_;
}
else
{
lean_inc(v_snd_159_);
lean_inc(v_fst_158_);
lean_dec(v_x_157_);
v___x_161_ = lean_box(0);
v_isShared_162_ = v_isSharedCheck_168_;
goto v_resetjp_160_;
}
v_resetjp_160_:
{
lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_166_; 
lean_inc(v_fst_158_);
v___x_163_ = lean_apply_1(v_f_u2081_155_, v_fst_158_);
v___x_164_ = lean_apply_2(v_f_u2082_156_, v_fst_158_, v_snd_159_);
if (v_isShared_162_ == 0)
{
lean_ctor_set(v___x_161_, 1, v___x_164_);
lean_ctor_set(v___x_161_, 0, v___x_163_);
v___x_166_ = v___x_161_;
goto v_reusejp_165_;
}
else
{
lean_object* v_reuseFailAlloc_167_; 
v_reuseFailAlloc_167_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_167_, 0, v___x_163_);
lean_ctor_set(v_reuseFailAlloc_167_, 1, v___x_164_);
v___x_166_ = v_reuseFailAlloc_167_;
goto v_reusejp_165_;
}
v_reusejp_165_:
{
return v___x_166_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_PSigma_map(lean_object* v_00_u03b1_u2081_169_, lean_object* v_00_u03b1_u2082_170_, lean_object* v_00_u03b2_u2081_171_, lean_object* v_00_u03b2_u2082_172_, lean_object* v_f_u2081_173_, lean_object* v_f_u2082_174_, lean_object* v_x_175_){
_start:
{
lean_object* v___x_176_; 
v___x_176_ = lp_mathlib_PSigma_map___redArg(v_f_u2081_173_, v_f_u2082_174_, v_x_175_);
return v___x_176_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Function_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Function_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Sigma_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Function_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Function_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Sigma_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Logic_Function_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Function_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Sigma_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Function_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Function_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Sigma_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Sigma_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Sigma_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
