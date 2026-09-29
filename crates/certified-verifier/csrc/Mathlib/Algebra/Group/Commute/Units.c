// Lean compiler output
// Module: Mathlib.Algebra.Group.Commute.Units
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Commute.Defs public import Mathlib.Algebra.Group.Semiconj.Units
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
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_leftOfMul___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_leftOfMul___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_leftOfMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_leftOfMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_leftOfAdd___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_leftOfAdd___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_leftOfAdd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_leftOfAdd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_rightOfMul___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_rightOfMul___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_rightOfMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_rightOfMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_rightOfAdd___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_rightOfAdd___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_rightOfAdd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_rightOfAdd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_ofPow___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_ofPow___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_ofPow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_ofPow___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_ofNSMul___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_ofNSMul___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_ofNSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_ofNSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_ofPowEqOne___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_ofPowEqOne___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_ofPowEqOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_ofPowEqOne___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_ofNSMulEqZero___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_ofNSMulEqZero___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_ofNSMulEqZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_ofNSMulEqZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_leftOfMul___redArg(lean_object* v_inst_1_, lean_object* v_u_2_, lean_object* v_a_3_, lean_object* v_b_4_){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v_toMul_7_; lean_object* v_inv_8_; lean_object* v___x_10_; uint8_t v_isShared_11_; uint8_t v_isSharedCheck_16_; 
v___x_5_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_1_);
v___x_6_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_5_);
v_toMul_7_ = lean_ctor_get(v___x_6_, 1);
lean_inc(v_toMul_7_);
lean_dec_ref(v___x_6_);
v_inv_8_ = lean_ctor_get(v_u_2_, 1);
v_isSharedCheck_16_ = !lean_is_exclusive(v_u_2_);
if (v_isSharedCheck_16_ == 0)
{
lean_object* v_unused_17_; 
v_unused_17_ = lean_ctor_get(v_u_2_, 0);
lean_dec(v_unused_17_);
v___x_10_ = v_u_2_;
v_isShared_11_ = v_isSharedCheck_16_;
goto v_resetjp_9_;
}
else
{
lean_inc(v_inv_8_);
lean_dec(v_u_2_);
v___x_10_ = lean_box(0);
v_isShared_11_ = v_isSharedCheck_16_;
goto v_resetjp_9_;
}
v_resetjp_9_:
{
lean_object* v___x_12_; lean_object* v___x_14_; 
v___x_12_ = lean_apply_2(v_toMul_7_, v_b_4_, v_inv_8_);
if (v_isShared_11_ == 0)
{
lean_ctor_set(v___x_10_, 1, v___x_12_);
lean_ctor_set(v___x_10_, 0, v_a_3_);
v___x_14_ = v___x_10_;
goto v_reusejp_13_;
}
else
{
lean_object* v_reuseFailAlloc_15_; 
v_reuseFailAlloc_15_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_15_, 0, v_a_3_);
lean_ctor_set(v_reuseFailAlloc_15_, 1, v___x_12_);
v___x_14_ = v_reuseFailAlloc_15_;
goto v_reusejp_13_;
}
v_reusejp_13_:
{
return v___x_14_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_leftOfMul___redArg___boxed(lean_object* v_inst_18_, lean_object* v_u_19_, lean_object* v_a_20_, lean_object* v_b_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_mathlib_Units_leftOfMul___redArg(v_inst_18_, v_u_19_, v_a_20_, v_b_21_);
lean_dec_ref(v_inst_18_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_leftOfMul(lean_object* v_M_23_, lean_object* v_inst_24_, lean_object* v_u_25_, lean_object* v_a_26_, lean_object* v_b_27_, lean_object* v_hu_28_, lean_object* v_hc_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lp_mathlib_Units_leftOfMul___redArg(v_inst_24_, v_u_25_, v_a_26_, v_b_27_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_leftOfMul___boxed(lean_object* v_M_31_, lean_object* v_inst_32_, lean_object* v_u_33_, lean_object* v_a_34_, lean_object* v_b_35_, lean_object* v_hu_36_, lean_object* v_hc_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_mathlib_Units_leftOfMul(v_M_31_, v_inst_32_, v_u_33_, v_a_34_, v_b_35_, v_hu_36_, v_hc_37_);
lean_dec_ref(v_inst_32_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_leftOfAdd___redArg(lean_object* v_inst_39_, lean_object* v_u_40_, lean_object* v_a_41_, lean_object* v_b_42_){
_start:
{
lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v_toAdd_45_; lean_object* v_neg_46_; lean_object* v___x_48_; uint8_t v_isShared_49_; uint8_t v_isSharedCheck_54_; 
v___x_43_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_39_);
v___x_44_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_43_);
v_toAdd_45_ = lean_ctor_get(v___x_44_, 1);
lean_inc(v_toAdd_45_);
lean_dec_ref(v___x_44_);
v_neg_46_ = lean_ctor_get(v_u_40_, 1);
v_isSharedCheck_54_ = !lean_is_exclusive(v_u_40_);
if (v_isSharedCheck_54_ == 0)
{
lean_object* v_unused_55_; 
v_unused_55_ = lean_ctor_get(v_u_40_, 0);
lean_dec(v_unused_55_);
v___x_48_ = v_u_40_;
v_isShared_49_ = v_isSharedCheck_54_;
goto v_resetjp_47_;
}
else
{
lean_inc(v_neg_46_);
lean_dec(v_u_40_);
v___x_48_ = lean_box(0);
v_isShared_49_ = v_isSharedCheck_54_;
goto v_resetjp_47_;
}
v_resetjp_47_:
{
lean_object* v___x_50_; lean_object* v___x_52_; 
v___x_50_ = lean_apply_2(v_toAdd_45_, v_b_42_, v_neg_46_);
if (v_isShared_49_ == 0)
{
lean_ctor_set(v___x_48_, 1, v___x_50_);
lean_ctor_set(v___x_48_, 0, v_a_41_);
v___x_52_ = v___x_48_;
goto v_reusejp_51_;
}
else
{
lean_object* v_reuseFailAlloc_53_; 
v_reuseFailAlloc_53_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_53_, 0, v_a_41_);
lean_ctor_set(v_reuseFailAlloc_53_, 1, v___x_50_);
v___x_52_ = v_reuseFailAlloc_53_;
goto v_reusejp_51_;
}
v_reusejp_51_:
{
return v___x_52_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_leftOfAdd___redArg___boxed(lean_object* v_inst_56_, lean_object* v_u_57_, lean_object* v_a_58_, lean_object* v_b_59_){
_start:
{
lean_object* v_res_60_; 
v_res_60_ = lp_mathlib_AddUnits_leftOfAdd___redArg(v_inst_56_, v_u_57_, v_a_58_, v_b_59_);
lean_dec_ref(v_inst_56_);
return v_res_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_leftOfAdd(lean_object* v_M_61_, lean_object* v_inst_62_, lean_object* v_u_63_, lean_object* v_a_64_, lean_object* v_b_65_, lean_object* v_hu_66_, lean_object* v_hc_67_){
_start:
{
lean_object* v___x_68_; 
v___x_68_ = lp_mathlib_AddUnits_leftOfAdd___redArg(v_inst_62_, v_u_63_, v_a_64_, v_b_65_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_leftOfAdd___boxed(lean_object* v_M_69_, lean_object* v_inst_70_, lean_object* v_u_71_, lean_object* v_a_72_, lean_object* v_b_73_, lean_object* v_hu_74_, lean_object* v_hc_75_){
_start:
{
lean_object* v_res_76_; 
v_res_76_ = lp_mathlib_AddUnits_leftOfAdd(v_M_69_, v_inst_70_, v_u_71_, v_a_72_, v_b_73_, v_hu_74_, v_hc_75_);
lean_dec_ref(v_inst_70_);
return v_res_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_rightOfMul___redArg(lean_object* v_inst_77_, lean_object* v_u_78_, lean_object* v_a_79_, lean_object* v_b_80_){
_start:
{
lean_object* v___x_81_; 
v___x_81_ = lp_mathlib_Units_leftOfMul___redArg(v_inst_77_, v_u_78_, v_b_80_, v_a_79_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_rightOfMul___redArg___boxed(lean_object* v_inst_82_, lean_object* v_u_83_, lean_object* v_a_84_, lean_object* v_b_85_){
_start:
{
lean_object* v_res_86_; 
v_res_86_ = lp_mathlib_Units_rightOfMul___redArg(v_inst_82_, v_u_83_, v_a_84_, v_b_85_);
lean_dec_ref(v_inst_82_);
return v_res_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_rightOfMul(lean_object* v_M_87_, lean_object* v_inst_88_, lean_object* v_u_89_, lean_object* v_a_90_, lean_object* v_b_91_, lean_object* v_hu_92_, lean_object* v_hc_93_){
_start:
{
lean_object* v___x_94_; 
v___x_94_ = lp_mathlib_Units_leftOfMul___redArg(v_inst_88_, v_u_89_, v_b_91_, v_a_90_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_rightOfMul___boxed(lean_object* v_M_95_, lean_object* v_inst_96_, lean_object* v_u_97_, lean_object* v_a_98_, lean_object* v_b_99_, lean_object* v_hu_100_, lean_object* v_hc_101_){
_start:
{
lean_object* v_res_102_; 
v_res_102_ = lp_mathlib_Units_rightOfMul(v_M_95_, v_inst_96_, v_u_97_, v_a_98_, v_b_99_, v_hu_100_, v_hc_101_);
lean_dec_ref(v_inst_96_);
return v_res_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_rightOfAdd___redArg(lean_object* v_inst_103_, lean_object* v_u_104_, lean_object* v_a_105_, lean_object* v_b_106_){
_start:
{
lean_object* v___x_107_; 
v___x_107_ = lp_mathlib_AddUnits_leftOfAdd___redArg(v_inst_103_, v_u_104_, v_b_106_, v_a_105_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_rightOfAdd___redArg___boxed(lean_object* v_inst_108_, lean_object* v_u_109_, lean_object* v_a_110_, lean_object* v_b_111_){
_start:
{
lean_object* v_res_112_; 
v_res_112_ = lp_mathlib_AddUnits_rightOfAdd___redArg(v_inst_108_, v_u_109_, v_a_110_, v_b_111_);
lean_dec_ref(v_inst_108_);
return v_res_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_rightOfAdd(lean_object* v_M_113_, lean_object* v_inst_114_, lean_object* v_u_115_, lean_object* v_a_116_, lean_object* v_b_117_, lean_object* v_hu_118_, lean_object* v_hc_119_){
_start:
{
lean_object* v___x_120_; 
v___x_120_ = lp_mathlib_AddUnits_leftOfAdd___redArg(v_inst_114_, v_u_115_, v_b_117_, v_a_116_);
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_rightOfAdd___boxed(lean_object* v_M_121_, lean_object* v_inst_122_, lean_object* v_u_123_, lean_object* v_a_124_, lean_object* v_b_125_, lean_object* v_hu_126_, lean_object* v_hc_127_){
_start:
{
lean_object* v_res_128_; 
v_res_128_ = lp_mathlib_AddUnits_rightOfAdd(v_M_121_, v_inst_122_, v_u_123_, v_a_124_, v_b_125_, v_hu_126_, v_hc_127_);
lean_dec_ref(v_inst_122_);
return v_res_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_ofPow___redArg(lean_object* v_inst_129_, lean_object* v_u_130_, lean_object* v_x_131_, lean_object* v_n_132_){
_start:
{
lean_object* v_toNPow_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; 
v_toNPow_133_ = lean_ctor_get(v_inst_129_, 2);
v___x_134_ = lean_unsigned_to_nat(1u);
v___x_135_ = lean_nat_sub(v_n_132_, v___x_134_);
lean_inc(v_toNPow_133_);
lean_inc(v_x_131_);
v___x_136_ = lean_apply_2(v_toNPow_133_, v___x_135_, v_x_131_);
v___x_137_ = lp_mathlib_Units_leftOfMul___redArg(v_inst_129_, v_u_130_, v_x_131_, v___x_136_);
lean_dec_ref(v_inst_129_);
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_ofPow___redArg___boxed(lean_object* v_inst_138_, lean_object* v_u_139_, lean_object* v_x_140_, lean_object* v_n_141_){
_start:
{
lean_object* v_res_142_; 
v_res_142_ = lp_mathlib_Units_ofPow___redArg(v_inst_138_, v_u_139_, v_x_140_, v_n_141_);
lean_dec(v_n_141_);
return v_res_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_ofPow(lean_object* v_M_143_, lean_object* v_inst_144_, lean_object* v_u_145_, lean_object* v_x_146_, lean_object* v_n_147_, lean_object* v_hn_148_, lean_object* v_hu_149_){
_start:
{
lean_object* v___x_150_; 
v___x_150_ = lp_mathlib_Units_ofPow___redArg(v_inst_144_, v_u_145_, v_x_146_, v_n_147_);
return v___x_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_ofPow___boxed(lean_object* v_M_151_, lean_object* v_inst_152_, lean_object* v_u_153_, lean_object* v_x_154_, lean_object* v_n_155_, lean_object* v_hn_156_, lean_object* v_hu_157_){
_start:
{
lean_object* v_res_158_; 
v_res_158_ = lp_mathlib_Units_ofPow(v_M_151_, v_inst_152_, v_u_153_, v_x_154_, v_n_155_, v_hn_156_, v_hu_157_);
lean_dec(v_n_155_);
return v_res_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_ofNSMul___redArg(lean_object* v_inst_159_, lean_object* v_u_160_, lean_object* v_x_161_, lean_object* v_n_162_){
_start:
{
lean_object* v_toNSMul_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; 
v_toNSMul_163_ = lean_ctor_get(v_inst_159_, 2);
v___x_164_ = lean_unsigned_to_nat(1u);
v___x_165_ = lean_nat_sub(v_n_162_, v___x_164_);
lean_inc(v_toNSMul_163_);
lean_inc(v_x_161_);
v___x_166_ = lean_apply_2(v_toNSMul_163_, v___x_165_, v_x_161_);
v___x_167_ = lp_mathlib_AddUnits_leftOfAdd___redArg(v_inst_159_, v_u_160_, v_x_161_, v___x_166_);
lean_dec_ref(v_inst_159_);
return v___x_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_ofNSMul___redArg___boxed(lean_object* v_inst_168_, lean_object* v_u_169_, lean_object* v_x_170_, lean_object* v_n_171_){
_start:
{
lean_object* v_res_172_; 
v_res_172_ = lp_mathlib_AddUnits_ofNSMul___redArg(v_inst_168_, v_u_169_, v_x_170_, v_n_171_);
lean_dec(v_n_171_);
return v_res_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_ofNSMul(lean_object* v_M_173_, lean_object* v_inst_174_, lean_object* v_u_175_, lean_object* v_x_176_, lean_object* v_n_177_, lean_object* v_hn_178_, lean_object* v_hu_179_){
_start:
{
lean_object* v___x_180_; 
v___x_180_ = lp_mathlib_AddUnits_ofNSMul___redArg(v_inst_174_, v_u_175_, v_x_176_, v_n_177_);
return v___x_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_ofNSMul___boxed(lean_object* v_M_181_, lean_object* v_inst_182_, lean_object* v_u_183_, lean_object* v_x_184_, lean_object* v_n_185_, lean_object* v_hn_186_, lean_object* v_hu_187_){
_start:
{
lean_object* v_res_188_; 
v_res_188_ = lp_mathlib_AddUnits_ofNSMul(v_M_181_, v_inst_182_, v_u_183_, v_x_184_, v_n_185_, v_hn_186_, v_hu_187_);
lean_dec(v_n_185_);
return v_res_188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_ofPowEqOne___redArg(lean_object* v_inst_189_, lean_object* v_a_190_, lean_object* v_n_191_){
_start:
{
lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v_toOne_194_; lean_object* v___x_196_; uint8_t v_isShared_197_; uint8_t v_isSharedCheck_202_; 
v___x_192_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_189_);
v___x_193_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_192_);
v_toOne_194_ = lean_ctor_get(v___x_193_, 0);
v_isSharedCheck_202_ = !lean_is_exclusive(v___x_193_);
if (v_isSharedCheck_202_ == 0)
{
lean_object* v_unused_203_; 
v_unused_203_ = lean_ctor_get(v___x_193_, 1);
lean_dec(v_unused_203_);
v___x_196_ = v___x_193_;
v_isShared_197_ = v_isSharedCheck_202_;
goto v_resetjp_195_;
}
else
{
lean_inc(v_toOne_194_);
lean_dec(v___x_193_);
v___x_196_ = lean_box(0);
v_isShared_197_ = v_isSharedCheck_202_;
goto v_resetjp_195_;
}
v_resetjp_195_:
{
lean_object* v___x_199_; 
lean_inc(v_toOne_194_);
if (v_isShared_197_ == 0)
{
lean_ctor_set(v___x_196_, 1, v_toOne_194_);
v___x_199_ = v___x_196_;
goto v_reusejp_198_;
}
else
{
lean_object* v_reuseFailAlloc_201_; 
v_reuseFailAlloc_201_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_201_, 0, v_toOne_194_);
lean_ctor_set(v_reuseFailAlloc_201_, 1, v_toOne_194_);
v___x_199_ = v_reuseFailAlloc_201_;
goto v_reusejp_198_;
}
v_reusejp_198_:
{
lean_object* v___x_200_; 
v___x_200_ = lp_mathlib_Units_ofPow___redArg(v_inst_189_, v___x_199_, v_a_190_, v_n_191_);
return v___x_200_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_ofPowEqOne___redArg___boxed(lean_object* v_inst_204_, lean_object* v_a_205_, lean_object* v_n_206_){
_start:
{
lean_object* v_res_207_; 
v_res_207_ = lp_mathlib_Units_ofPowEqOne___redArg(v_inst_204_, v_a_205_, v_n_206_);
lean_dec(v_n_206_);
return v_res_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_ofPowEqOne(lean_object* v_M_208_, lean_object* v_inst_209_, lean_object* v_a_210_, lean_object* v_n_211_, lean_object* v_ha_212_, lean_object* v_hn_213_){
_start:
{
lean_object* v___x_214_; 
v___x_214_ = lp_mathlib_Units_ofPowEqOne___redArg(v_inst_209_, v_a_210_, v_n_211_);
return v___x_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_ofPowEqOne___boxed(lean_object* v_M_215_, lean_object* v_inst_216_, lean_object* v_a_217_, lean_object* v_n_218_, lean_object* v_ha_219_, lean_object* v_hn_220_){
_start:
{
lean_object* v_res_221_; 
v_res_221_ = lp_mathlib_Units_ofPowEqOne(v_M_215_, v_inst_216_, v_a_217_, v_n_218_, v_ha_219_, v_hn_220_);
lean_dec(v_n_218_);
return v_res_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_ofNSMulEqZero___redArg(lean_object* v_inst_222_, lean_object* v_a_223_, lean_object* v_n_224_){
_start:
{
lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v_toZero_227_; lean_object* v___x_229_; uint8_t v_isShared_230_; uint8_t v_isSharedCheck_235_; 
v___x_225_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_222_);
v___x_226_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_225_);
v_toZero_227_ = lean_ctor_get(v___x_226_, 0);
v_isSharedCheck_235_ = !lean_is_exclusive(v___x_226_);
if (v_isSharedCheck_235_ == 0)
{
lean_object* v_unused_236_; 
v_unused_236_ = lean_ctor_get(v___x_226_, 1);
lean_dec(v_unused_236_);
v___x_229_ = v___x_226_;
v_isShared_230_ = v_isSharedCheck_235_;
goto v_resetjp_228_;
}
else
{
lean_inc(v_toZero_227_);
lean_dec(v___x_226_);
v___x_229_ = lean_box(0);
v_isShared_230_ = v_isSharedCheck_235_;
goto v_resetjp_228_;
}
v_resetjp_228_:
{
lean_object* v___x_232_; 
lean_inc(v_toZero_227_);
if (v_isShared_230_ == 0)
{
lean_ctor_set(v___x_229_, 1, v_toZero_227_);
v___x_232_ = v___x_229_;
goto v_reusejp_231_;
}
else
{
lean_object* v_reuseFailAlloc_234_; 
v_reuseFailAlloc_234_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_234_, 0, v_toZero_227_);
lean_ctor_set(v_reuseFailAlloc_234_, 1, v_toZero_227_);
v___x_232_ = v_reuseFailAlloc_234_;
goto v_reusejp_231_;
}
v_reusejp_231_:
{
lean_object* v___x_233_; 
v___x_233_ = lp_mathlib_AddUnits_ofNSMul___redArg(v_inst_222_, v___x_232_, v_a_223_, v_n_224_);
return v___x_233_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_ofNSMulEqZero___redArg___boxed(lean_object* v_inst_237_, lean_object* v_a_238_, lean_object* v_n_239_){
_start:
{
lean_object* v_res_240_; 
v_res_240_ = lp_mathlib_AddUnits_ofNSMulEqZero___redArg(v_inst_237_, v_a_238_, v_n_239_);
lean_dec(v_n_239_);
return v_res_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_ofNSMulEqZero(lean_object* v_M_241_, lean_object* v_inst_242_, lean_object* v_a_243_, lean_object* v_n_244_, lean_object* v_ha_245_, lean_object* v_hn_246_){
_start:
{
lean_object* v___x_247_; 
v___x_247_ = lp_mathlib_AddUnits_ofNSMulEqZero___redArg(v_inst_242_, v_a_243_, v_n_244_);
return v___x_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_ofNSMulEqZero___boxed(lean_object* v_M_248_, lean_object* v_inst_249_, lean_object* v_a_250_, lean_object* v_n_251_, lean_object* v_ha_252_, lean_object* v_hn_253_){
_start:
{
lean_object* v_res_254_; 
v_res_254_ = lp_mathlib_AddUnits_ofNSMulEqZero(v_M_248_, v_inst_249_, v_a_250_, v_n_251_, v_ha_252_, v_hn_253_);
lean_dec(v_n_251_);
return v_res_254_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Commute_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Semiconj_Units(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Commute_Units(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Semiconj_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Commute_Units(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Commute_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Semiconj_Units(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Commute_Units(uint8_t builtin) {
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
res = initialize_mathlib_Mathlib_Algebra_Group_Semiconj_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Commute_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Commute_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Commute_Units(builtin);
}
#ifdef __cplusplus
}
#endif
