// Lean compiler output
// Module: Mathlib.Algebra.Group.Pointwise.Set.Finite
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Pointwise.Set.Basic public import Mathlib.Algebra.Group.Pointwise.Set.Scalar public import Mathlib.Basic.Finite.Prod
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
uint8_t lp_mathlib_Finset_decidableExistsAndFinset___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* l_Nat_recCompiled___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Set_fintypeImage2___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeMul___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeAdd___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeAdd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemMul___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemMul___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemMul___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemMul___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemMul___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemAdd___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemAdd___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemAdd___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemAdd___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemAdd___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemAdd___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemAdd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemAdd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemPow___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemPow___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemPow___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemPow___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemPow___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemPow___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemPow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemPow___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemNSMul___elam__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemNSMul___elam__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemNSMul___elam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemNSMul___elam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemNSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemNSMul___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemNSMul___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemNSMul___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemNSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemNSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeDiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeDiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeSub___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeSub(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Pointwise_Set_Finite_0__AddGroup_card__nsmul__eq__card__nsmul__card__univ_match__1__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Pointwise_Set_Finite_0__AddGroup_card__nsmul__eq__card__nsmul__card__univ_match__1__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeMul___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_x1_2_, lean_object* v_x2_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_apply_2(v_inst_1_, v_x1_2_, v_x2_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeMul___redArg(lean_object* v_inst_5_, lean_object* v_inst_6_, lean_object* v_inst_7_, lean_object* v_inst_8_){
_start:
{
lean_object* v___f_9_; lean_object* v___x_10_; 
v___f_9_ = lean_alloc_closure((void*)(lp_mathlib_Set_fintypeMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_9_, 0, v_inst_5_);
v___x_10_ = lp_mathlib_Set_fintypeImage2___redArg(v_inst_6_, v___f_9_, v_inst_7_, v_inst_8_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeMul(lean_object* v_00_u03b1_11_, lean_object* v_inst_12_, lean_object* v_inst_13_, lean_object* v_s_14_, lean_object* v_t_15_, lean_object* v_inst_16_, lean_object* v_inst_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lp_mathlib_Set_fintypeMul___redArg(v_inst_12_, v_inst_13_, v_inst_16_, v_inst_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeAdd___redArg(lean_object* v_inst_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_inst_22_){
_start:
{
lean_object* v___f_23_; lean_object* v___x_24_; 
v___f_23_ = lean_alloc_closure((void*)(lp_mathlib_Set_fintypeMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_23_, 0, v_inst_19_);
v___x_24_ = lp_mathlib_Set_fintypeImage2___redArg(v_inst_20_, v___f_23_, v_inst_21_, v_inst_22_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeAdd(lean_object* v_00_u03b1_25_, lean_object* v_inst_26_, lean_object* v_inst_27_, lean_object* v_s_28_, lean_object* v_t_29_, lean_object* v_inst_30_, lean_object* v_inst_31_){
_start:
{
lean_object* v___x_32_; 
v___x_32_ = lp_mathlib_Set_fintypeAdd___redArg(v_inst_26_, v_inst_27_, v_inst_30_, v_inst_31_);
return v___x_32_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemMul___redArg___lam__0(lean_object* v_inst_33_, lean_object* v_toMul_34_, lean_object* v_a_35_, lean_object* v_inst_36_, lean_object* v_x_37_, lean_object* v_a_38_){
_start:
{
lean_object* v___x_39_; uint8_t v___x_40_; 
lean_inc(v_a_38_);
v___x_39_ = lean_apply_1(v_inst_33_, v_a_38_);
v___x_40_ = lean_unbox(v___x_39_);
if (v___x_40_ == 0)
{
uint8_t v___x_41_; 
lean_dec(v_a_38_);
lean_dec(v_x_37_);
lean_dec_ref(v_inst_36_);
lean_dec(v_a_35_);
lean_dec(v_toMul_34_);
v___x_41_ = lean_unbox(v___x_39_);
return v___x_41_;
}
else
{
lean_object* v___x_42_; lean_object* v___x_43_; uint8_t v___x_44_; 
v___x_42_ = lean_apply_2(v_toMul_34_, v_a_35_, v_a_38_);
v___x_43_ = lean_apply_2(v_inst_36_, v___x_42_, v_x_37_);
v___x_44_ = lean_unbox(v___x_43_);
return v___x_44_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemMul___redArg___lam__0___boxed(lean_object* v_inst_45_, lean_object* v_toMul_46_, lean_object* v_a_47_, lean_object* v_inst_48_, lean_object* v_x_49_, lean_object* v_a_50_){
_start:
{
uint8_t v_res_51_; lean_object* v_r_52_; 
v_res_51_ = lp_mathlib_Set_decidableMemMul___redArg___lam__0(v_inst_45_, v_toMul_46_, v_a_47_, v_inst_48_, v_x_49_, v_a_50_);
v_r_52_ = lean_box(v_res_51_);
return v_r_52_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemMul___redArg___lam__1(lean_object* v_inst_53_, lean_object* v_inst_54_, lean_object* v_toMul_55_, lean_object* v_inst_56_, lean_object* v_x_57_, lean_object* v_inst_58_, lean_object* v_a_59_){
_start:
{
lean_object* v___x_60_; uint8_t v___x_61_; 
lean_inc(v_a_59_);
v___x_60_ = lean_apply_1(v_inst_53_, v_a_59_);
v___x_61_ = lean_unbox(v___x_60_);
if (v___x_61_ == 0)
{
uint8_t v___x_62_; 
lean_dec(v_a_59_);
lean_dec(v_inst_58_);
lean_dec(v_x_57_);
lean_dec_ref(v_inst_56_);
lean_dec(v_toMul_55_);
lean_dec_ref(v_inst_54_);
v___x_62_ = lean_unbox(v___x_60_);
return v___x_62_;
}
else
{
lean_object* v___f_63_; uint8_t v___x_64_; 
v___f_63_ = lean_alloc_closure((void*)(lp_mathlib_Set_decidableMemMul___redArg___lam__0___boxed), 6, 5);
lean_closure_set(v___f_63_, 0, v_inst_54_);
lean_closure_set(v___f_63_, 1, v_toMul_55_);
lean_closure_set(v___f_63_, 2, v_a_59_);
lean_closure_set(v___f_63_, 3, v_inst_56_);
lean_closure_set(v___f_63_, 4, v_x_57_);
v___x_64_ = lp_mathlib_Finset_decidableExistsAndFinset___redArg(v_inst_58_, v___f_63_);
return v___x_64_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemMul___redArg___lam__1___boxed(lean_object* v_inst_65_, lean_object* v_inst_66_, lean_object* v_toMul_67_, lean_object* v_inst_68_, lean_object* v_x_69_, lean_object* v_inst_70_, lean_object* v_a_71_){
_start:
{
uint8_t v_res_72_; lean_object* v_r_73_; 
v_res_72_ = lp_mathlib_Set_decidableMemMul___redArg___lam__1(v_inst_65_, v_inst_66_, v_toMul_67_, v_inst_68_, v_x_69_, v_inst_70_, v_a_71_);
v_r_73_ = lean_box(v_res_72_);
return v_r_73_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemMul___redArg(lean_object* v_inst_74_, lean_object* v_inst_75_, lean_object* v_inst_76_, lean_object* v_inst_77_, lean_object* v_inst_78_, lean_object* v_x_79_){
_start:
{
lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v_toMul_82_; lean_object* v___f_83_; uint8_t v___x_84_; 
v___x_80_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_74_);
v___x_81_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_80_);
v_toMul_82_ = lean_ctor_get(v___x_81_, 1);
lean_inc(v_toMul_82_);
lean_dec_ref(v___x_81_);
lean_inc(v_inst_75_);
v___f_83_ = lean_alloc_closure((void*)(lp_mathlib_Set_decidableMemMul___redArg___lam__1___boxed), 7, 6);
lean_closure_set(v___f_83_, 0, v_inst_77_);
lean_closure_set(v___f_83_, 1, v_inst_78_);
lean_closure_set(v___f_83_, 2, v_toMul_82_);
lean_closure_set(v___f_83_, 3, v_inst_76_);
lean_closure_set(v___f_83_, 4, v_x_79_);
lean_closure_set(v___f_83_, 5, v_inst_75_);
v___x_84_ = lp_mathlib_Finset_decidableExistsAndFinset___redArg(v_inst_75_, v___f_83_);
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemMul___redArg___boxed(lean_object* v_inst_85_, lean_object* v_inst_86_, lean_object* v_inst_87_, lean_object* v_inst_88_, lean_object* v_inst_89_, lean_object* v_x_90_){
_start:
{
uint8_t v_res_91_; lean_object* v_r_92_; 
v_res_91_ = lp_mathlib_Set_decidableMemMul___redArg(v_inst_85_, v_inst_86_, v_inst_87_, v_inst_88_, v_inst_89_, v_x_90_);
lean_dec_ref(v_inst_85_);
v_r_92_ = lean_box(v_res_91_);
return v_r_92_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemMul(lean_object* v_00_u03b1_93_, lean_object* v_inst_94_, lean_object* v_s_95_, lean_object* v_t_96_, lean_object* v_inst_97_, lean_object* v_inst_98_, lean_object* v_inst_99_, lean_object* v_inst_100_, lean_object* v_x_101_){
_start:
{
uint8_t v___x_102_; 
v___x_102_ = lp_mathlib_Set_decidableMemMul___redArg(v_inst_94_, v_inst_97_, v_inst_98_, v_inst_99_, v_inst_100_, v_x_101_);
return v___x_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemMul___boxed(lean_object* v_00_u03b1_103_, lean_object* v_inst_104_, lean_object* v_s_105_, lean_object* v_t_106_, lean_object* v_inst_107_, lean_object* v_inst_108_, lean_object* v_inst_109_, lean_object* v_inst_110_, lean_object* v_x_111_){
_start:
{
uint8_t v_res_112_; lean_object* v_r_113_; 
v_res_112_ = lp_mathlib_Set_decidableMemMul(v_00_u03b1_103_, v_inst_104_, v_s_105_, v_t_106_, v_inst_107_, v_inst_108_, v_inst_109_, v_inst_110_, v_x_111_);
lean_dec_ref(v_inst_104_);
v_r_113_ = lean_box(v_res_112_);
return v_r_113_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemAdd___redArg___lam__0(lean_object* v_inst_114_, lean_object* v_toAdd_115_, lean_object* v_a_116_, lean_object* v_inst_117_, lean_object* v_x_118_, lean_object* v_a_119_){
_start:
{
lean_object* v___x_120_; uint8_t v___x_121_; 
lean_inc(v_a_119_);
v___x_120_ = lean_apply_1(v_inst_114_, v_a_119_);
v___x_121_ = lean_unbox(v___x_120_);
if (v___x_121_ == 0)
{
uint8_t v___x_122_; 
lean_dec(v_a_119_);
lean_dec(v_x_118_);
lean_dec_ref(v_inst_117_);
lean_dec(v_a_116_);
lean_dec(v_toAdd_115_);
v___x_122_ = lean_unbox(v___x_120_);
return v___x_122_;
}
else
{
lean_object* v___x_123_; lean_object* v___x_124_; uint8_t v___x_125_; 
v___x_123_ = lean_apply_2(v_toAdd_115_, v_a_116_, v_a_119_);
v___x_124_ = lean_apply_2(v_inst_117_, v___x_123_, v_x_118_);
v___x_125_ = lean_unbox(v___x_124_);
return v___x_125_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemAdd___redArg___lam__0___boxed(lean_object* v_inst_126_, lean_object* v_toAdd_127_, lean_object* v_a_128_, lean_object* v_inst_129_, lean_object* v_x_130_, lean_object* v_a_131_){
_start:
{
uint8_t v_res_132_; lean_object* v_r_133_; 
v_res_132_ = lp_mathlib_Set_decidableMemAdd___redArg___lam__0(v_inst_126_, v_toAdd_127_, v_a_128_, v_inst_129_, v_x_130_, v_a_131_);
v_r_133_ = lean_box(v_res_132_);
return v_r_133_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemAdd___redArg___lam__1(lean_object* v_inst_134_, lean_object* v_inst_135_, lean_object* v_toAdd_136_, lean_object* v_inst_137_, lean_object* v_x_138_, lean_object* v_inst_139_, lean_object* v_a_140_){
_start:
{
lean_object* v___x_141_; uint8_t v___x_142_; 
lean_inc(v_a_140_);
v___x_141_ = lean_apply_1(v_inst_134_, v_a_140_);
v___x_142_ = lean_unbox(v___x_141_);
if (v___x_142_ == 0)
{
uint8_t v___x_143_; 
lean_dec(v_a_140_);
lean_dec(v_inst_139_);
lean_dec(v_x_138_);
lean_dec_ref(v_inst_137_);
lean_dec(v_toAdd_136_);
lean_dec_ref(v_inst_135_);
v___x_143_ = lean_unbox(v___x_141_);
return v___x_143_;
}
else
{
lean_object* v___f_144_; uint8_t v___x_145_; 
v___f_144_ = lean_alloc_closure((void*)(lp_mathlib_Set_decidableMemAdd___redArg___lam__0___boxed), 6, 5);
lean_closure_set(v___f_144_, 0, v_inst_135_);
lean_closure_set(v___f_144_, 1, v_toAdd_136_);
lean_closure_set(v___f_144_, 2, v_a_140_);
lean_closure_set(v___f_144_, 3, v_inst_137_);
lean_closure_set(v___f_144_, 4, v_x_138_);
v___x_145_ = lp_mathlib_Finset_decidableExistsAndFinset___redArg(v_inst_139_, v___f_144_);
return v___x_145_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemAdd___redArg___lam__1___boxed(lean_object* v_inst_146_, lean_object* v_inst_147_, lean_object* v_toAdd_148_, lean_object* v_inst_149_, lean_object* v_x_150_, lean_object* v_inst_151_, lean_object* v_a_152_){
_start:
{
uint8_t v_res_153_; lean_object* v_r_154_; 
v_res_153_ = lp_mathlib_Set_decidableMemAdd___redArg___lam__1(v_inst_146_, v_inst_147_, v_toAdd_148_, v_inst_149_, v_x_150_, v_inst_151_, v_a_152_);
v_r_154_ = lean_box(v_res_153_);
return v_r_154_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemAdd___redArg(lean_object* v_inst_155_, lean_object* v_inst_156_, lean_object* v_inst_157_, lean_object* v_inst_158_, lean_object* v_inst_159_, lean_object* v_x_160_){
_start:
{
lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v_toAdd_163_; lean_object* v___f_164_; uint8_t v___x_165_; 
v___x_161_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_155_);
v___x_162_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_161_);
v_toAdd_163_ = lean_ctor_get(v___x_162_, 1);
lean_inc(v_toAdd_163_);
lean_dec_ref(v___x_162_);
lean_inc(v_inst_156_);
v___f_164_ = lean_alloc_closure((void*)(lp_mathlib_Set_decidableMemAdd___redArg___lam__1___boxed), 7, 6);
lean_closure_set(v___f_164_, 0, v_inst_158_);
lean_closure_set(v___f_164_, 1, v_inst_159_);
lean_closure_set(v___f_164_, 2, v_toAdd_163_);
lean_closure_set(v___f_164_, 3, v_inst_157_);
lean_closure_set(v___f_164_, 4, v_x_160_);
lean_closure_set(v___f_164_, 5, v_inst_156_);
v___x_165_ = lp_mathlib_Finset_decidableExistsAndFinset___redArg(v_inst_156_, v___f_164_);
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemAdd___redArg___boxed(lean_object* v_inst_166_, lean_object* v_inst_167_, lean_object* v_inst_168_, lean_object* v_inst_169_, lean_object* v_inst_170_, lean_object* v_x_171_){
_start:
{
uint8_t v_res_172_; lean_object* v_r_173_; 
v_res_172_ = lp_mathlib_Set_decidableMemAdd___redArg(v_inst_166_, v_inst_167_, v_inst_168_, v_inst_169_, v_inst_170_, v_x_171_);
lean_dec_ref(v_inst_166_);
v_r_173_ = lean_box(v_res_172_);
return v_r_173_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemAdd(lean_object* v_00_u03b1_174_, lean_object* v_inst_175_, lean_object* v_s_176_, lean_object* v_t_177_, lean_object* v_inst_178_, lean_object* v_inst_179_, lean_object* v_inst_180_, lean_object* v_inst_181_, lean_object* v_x_182_){
_start:
{
uint8_t v___x_183_; 
v___x_183_ = lp_mathlib_Set_decidableMemAdd___redArg(v_inst_175_, v_inst_178_, v_inst_179_, v_inst_180_, v_inst_181_, v_x_182_);
return v___x_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemAdd___boxed(lean_object* v_00_u03b1_184_, lean_object* v_inst_185_, lean_object* v_s_186_, lean_object* v_t_187_, lean_object* v_inst_188_, lean_object* v_inst_189_, lean_object* v_inst_190_, lean_object* v_inst_191_, lean_object* v_x_192_){
_start:
{
uint8_t v_res_193_; lean_object* v_r_194_; 
v_res_193_ = lp_mathlib_Set_decidableMemAdd(v_00_u03b1_184_, v_inst_185_, v_s_186_, v_t_187_, v_inst_188_, v_inst_189_, v_inst_190_, v_inst_191_, v_x_192_);
lean_dec_ref(v_inst_185_);
v_r_194_ = lean_box(v_res_193_);
return v_r_194_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemPow___redArg___lam__0(lean_object* v_inst_195_, lean_object* v_inst_196_, lean_object* v_inst_197_, lean_object* v_inst_198_, lean_object* v_n_199_, lean_object* v_ih_200_, lean_object* v___y_201_){
_start:
{
uint8_t v___x_202_; 
v___x_202_ = lp_mathlib_Set_decidableMemMul___redArg(v_inst_195_, v_inst_196_, v_inst_197_, v_ih_200_, v_inst_198_, v___y_201_);
return v___x_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemPow___redArg___lam__0___boxed(lean_object* v_inst_203_, lean_object* v_inst_204_, lean_object* v_inst_205_, lean_object* v_inst_206_, lean_object* v_n_207_, lean_object* v_ih_208_, lean_object* v___y_209_){
_start:
{
uint8_t v_res_210_; lean_object* v_r_211_; 
v_res_210_ = lp_mathlib_Set_decidableMemPow___redArg___lam__0(v_inst_203_, v_inst_204_, v_inst_205_, v_inst_206_, v_n_207_, v_ih_208_, v___y_209_);
lean_dec(v_n_207_);
lean_dec_ref(v_inst_203_);
v_r_211_ = lean_box(v_res_210_);
return v_r_211_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemPow___redArg___lam__1(lean_object* v_inst_212_, lean_object* v_toOne_213_, lean_object* v_a_214_){
_start:
{
lean_object* v___x_215_; uint8_t v___x_216_; 
v___x_215_ = lean_apply_2(v_inst_212_, v_a_214_, v_toOne_213_);
v___x_216_ = lean_unbox(v___x_215_);
return v___x_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemPow___redArg___lam__1___boxed(lean_object* v_inst_217_, lean_object* v_toOne_218_, lean_object* v_a_219_){
_start:
{
uint8_t v_res_220_; lean_object* v_r_221_; 
v_res_220_ = lp_mathlib_Set_decidableMemPow___redArg___lam__1(v_inst_217_, v_toOne_218_, v_a_219_);
v_r_221_ = lean_box(v_res_220_);
return v_r_221_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemPow___redArg(lean_object* v_inst_222_, lean_object* v_inst_223_, lean_object* v_inst_224_, lean_object* v_inst_225_, lean_object* v_n_226_, lean_object* v_a_227_){
_start:
{
lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v_toOne_230_; lean_object* v___f_231_; lean_object* v___f_232_; lean_object* v___x_26__overap_233_; lean_object* v___x_234_; uint8_t v___x_235_; 
v___x_228_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_222_);
v___x_229_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_228_);
v_toOne_230_ = lean_ctor_get(v___x_229_, 0);
lean_inc(v_toOne_230_);
lean_dec_ref(v___x_229_);
lean_inc_ref(v_inst_224_);
v___f_231_ = lean_alloc_closure((void*)(lp_mathlib_Set_decidableMemPow___redArg___lam__0___boxed), 7, 4);
lean_closure_set(v___f_231_, 0, v_inst_222_);
lean_closure_set(v___f_231_, 1, v_inst_223_);
lean_closure_set(v___f_231_, 2, v_inst_224_);
lean_closure_set(v___f_231_, 3, v_inst_225_);
v___f_232_ = lean_alloc_closure((void*)(lp_mathlib_Set_decidableMemPow___redArg___lam__1___boxed), 3, 2);
lean_closure_set(v___f_232_, 0, v_inst_224_);
lean_closure_set(v___f_232_, 1, v_toOne_230_);
v___x_26__overap_233_ = l_Nat_recCompiled___redArg(v___f_232_, v___f_231_, v_n_226_);
lean_dec_ref(v___f_232_);
v___x_234_ = lean_apply_1(v___x_26__overap_233_, v_a_227_);
v___x_235_ = lean_unbox(v___x_234_);
return v___x_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemPow___redArg___boxed(lean_object* v_inst_236_, lean_object* v_inst_237_, lean_object* v_inst_238_, lean_object* v_inst_239_, lean_object* v_n_240_, lean_object* v_a_241_){
_start:
{
uint8_t v_res_242_; lean_object* v_r_243_; 
v_res_242_ = lp_mathlib_Set_decidableMemPow___redArg(v_inst_236_, v_inst_237_, v_inst_238_, v_inst_239_, v_n_240_, v_a_241_);
lean_dec(v_n_240_);
v_r_243_ = lean_box(v_res_242_);
return v_r_243_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemPow(lean_object* v_00_u03b1_244_, lean_object* v_inst_245_, lean_object* v_s_246_, lean_object* v_inst_247_, lean_object* v_inst_248_, lean_object* v_inst_249_, lean_object* v_n_250_, lean_object* v_a_251_){
_start:
{
uint8_t v___x_252_; 
v___x_252_ = lp_mathlib_Set_decidableMemPow___redArg(v_inst_245_, v_inst_247_, v_inst_248_, v_inst_249_, v_n_250_, v_a_251_);
return v___x_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemPow___boxed(lean_object* v_00_u03b1_253_, lean_object* v_inst_254_, lean_object* v_s_255_, lean_object* v_inst_256_, lean_object* v_inst_257_, lean_object* v_inst_258_, lean_object* v_n_259_, lean_object* v_a_260_){
_start:
{
uint8_t v_res_261_; lean_object* v_r_262_; 
v_res_261_ = lp_mathlib_Set_decidableMemPow(v_00_u03b1_253_, v_inst_254_, v_s_255_, v_inst_256_, v_inst_257_, v_inst_258_, v_n_259_, v_a_260_);
lean_dec(v_n_259_);
v_r_262_ = lean_box(v_res_261_);
return v_r_262_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemNSMul___elam__0___redArg(lean_object* v_inst_263_, lean_object* v_inst_264_, lean_object* v_inst_265_, lean_object* v_inst_266_, lean_object* v_ih_267_, lean_object* v___y_268_){
_start:
{
uint8_t v___x_269_; 
v___x_269_ = lp_mathlib_Set_decidableMemAdd___redArg(v_inst_263_, v_inst_264_, v_inst_265_, v_ih_267_, v_inst_266_, v___y_268_);
return v___x_269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemNSMul___elam__0___redArg___boxed(lean_object* v_inst_270_, lean_object* v_inst_271_, lean_object* v_inst_272_, lean_object* v_inst_273_, lean_object* v_ih_274_, lean_object* v___y_275_){
_start:
{
uint8_t v_res_276_; lean_object* v_r_277_; 
v_res_276_ = lp_mathlib_Set_decidableMemNSMul___elam__0___redArg(v_inst_270_, v_inst_271_, v_inst_272_, v_inst_273_, v_ih_274_, v___y_275_);
lean_dec_ref(v_inst_270_);
v_r_277_ = lean_box(v_res_276_);
return v_r_277_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemNSMul___elam__0(lean_object* v_00_u03b1_278_, lean_object* v_inst_279_, lean_object* v_inst_280_, lean_object* v_inst_281_, lean_object* v_inst_282_, lean_object* v_n_283_, lean_object* v_ih_284_, lean_object* v___y_285_){
_start:
{
uint8_t v___x_286_; 
v___x_286_ = lp_mathlib_Set_decidableMemAdd___redArg(v_inst_279_, v_inst_280_, v_inst_281_, v_ih_284_, v_inst_282_, v___y_285_);
return v___x_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemNSMul___elam__0___boxed(lean_object* v_00_u03b1_287_, lean_object* v_inst_288_, lean_object* v_inst_289_, lean_object* v_inst_290_, lean_object* v_inst_291_, lean_object* v_n_292_, lean_object* v_ih_293_, lean_object* v___y_294_){
_start:
{
uint8_t v_res_295_; lean_object* v_r_296_; 
v_res_295_ = lp_mathlib_Set_decidableMemNSMul___elam__0(v_00_u03b1_287_, v_inst_288_, v_inst_289_, v_inst_290_, v_inst_291_, v_n_292_, v_ih_293_, v___y_294_);
lean_dec(v_n_292_);
lean_dec_ref(v_inst_288_);
v_r_296_ = lean_box(v_res_295_);
return v_r_296_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemNSMul___redArg___lam__0(lean_object* v_inst_297_, lean_object* v_toZero_298_, lean_object* v_a_299_){
_start:
{
lean_object* v___x_300_; uint8_t v___x_301_; 
v___x_300_ = lean_apply_2(v_inst_297_, v_a_299_, v_toZero_298_);
v___x_301_ = lean_unbox(v___x_300_);
return v___x_301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemNSMul___redArg___lam__0___boxed(lean_object* v_inst_302_, lean_object* v_toZero_303_, lean_object* v_a_304_){
_start:
{
uint8_t v_res_305_; lean_object* v_r_306_; 
v_res_305_ = lp_mathlib_Set_decidableMemNSMul___redArg___lam__0(v_inst_302_, v_toZero_303_, v_a_304_);
v_r_306_ = lean_box(v_res_305_);
return v_r_306_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemNSMul___redArg(lean_object* v_inst_307_, lean_object* v_inst_308_, lean_object* v_inst_309_, lean_object* v_inst_310_, lean_object* v_n_311_, lean_object* v_a_312_){
_start:
{
lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v_toZero_315_; lean_object* v___f_316_; lean_object* v___f_317_; lean_object* v___x_26__overap_318_; lean_object* v___x_319_; uint8_t v___x_320_; 
v___x_313_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_307_);
v___x_314_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_313_);
v_toZero_315_ = lean_ctor_get(v___x_314_, 0);
lean_inc(v_toZero_315_);
lean_dec_ref(v___x_314_);
lean_inc_ref(v_inst_309_);
v___f_316_ = lean_alloc_closure((void*)(lp_mathlib_Set_decidableMemNSMul___elam__0___boxed), 8, 5);
lean_closure_set(v___f_316_, 0, lean_box(0));
lean_closure_set(v___f_316_, 1, v_inst_307_);
lean_closure_set(v___f_316_, 2, v_inst_308_);
lean_closure_set(v___f_316_, 3, v_inst_309_);
lean_closure_set(v___f_316_, 4, v_inst_310_);
v___f_317_ = lean_alloc_closure((void*)(lp_mathlib_Set_decidableMemNSMul___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_317_, 0, v_inst_309_);
lean_closure_set(v___f_317_, 1, v_toZero_315_);
v___x_26__overap_318_ = l_Nat_recCompiled___redArg(v___f_317_, v___f_316_, v_n_311_);
lean_dec_ref(v___f_317_);
v___x_319_ = lean_apply_1(v___x_26__overap_318_, v_a_312_);
v___x_320_ = lean_unbox(v___x_319_);
return v___x_320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemNSMul___redArg___boxed(lean_object* v_inst_321_, lean_object* v_inst_322_, lean_object* v_inst_323_, lean_object* v_inst_324_, lean_object* v_n_325_, lean_object* v_a_326_){
_start:
{
uint8_t v_res_327_; lean_object* v_r_328_; 
v_res_327_ = lp_mathlib_Set_decidableMemNSMul___redArg(v_inst_321_, v_inst_322_, v_inst_323_, v_inst_324_, v_n_325_, v_a_326_);
lean_dec(v_n_325_);
v_r_328_ = lean_box(v_res_327_);
return v_r_328_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemNSMul(lean_object* v_00_u03b1_329_, lean_object* v_inst_330_, lean_object* v_s_331_, lean_object* v_inst_332_, lean_object* v_inst_333_, lean_object* v_inst_334_, lean_object* v_n_335_, lean_object* v_a_336_){
_start:
{
uint8_t v___x_337_; 
v___x_337_ = lp_mathlib_Set_decidableMemNSMul___redArg(v_inst_330_, v_inst_332_, v_inst_333_, v_inst_334_, v_n_335_, v_a_336_);
return v___x_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemNSMul___boxed(lean_object* v_00_u03b1_338_, lean_object* v_inst_339_, lean_object* v_s_340_, lean_object* v_inst_341_, lean_object* v_inst_342_, lean_object* v_inst_343_, lean_object* v_n_344_, lean_object* v_a_345_){
_start:
{
uint8_t v_res_346_; lean_object* v_r_347_; 
v_res_346_ = lp_mathlib_Set_decidableMemNSMul(v_00_u03b1_338_, v_inst_339_, v_s_340_, v_inst_341_, v_inst_342_, v_inst_343_, v_n_344_, v_a_345_);
lean_dec(v_n_344_);
v_r_347_ = lean_box(v_res_346_);
return v_r_347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeDiv___redArg(lean_object* v_inst_348_, lean_object* v_inst_349_, lean_object* v_inst_350_, lean_object* v_inst_351_){
_start:
{
lean_object* v___f_352_; lean_object* v___x_353_; 
v___f_352_ = lean_alloc_closure((void*)(lp_mathlib_Set_fintypeMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_352_, 0, v_inst_348_);
v___x_353_ = lp_mathlib_Set_fintypeImage2___redArg(v_inst_349_, v___f_352_, v_inst_350_, v_inst_351_);
return v___x_353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeDiv(lean_object* v_00_u03b1_354_, lean_object* v_inst_355_, lean_object* v_inst_356_, lean_object* v_s_357_, lean_object* v_t_358_, lean_object* v_inst_359_, lean_object* v_inst_360_){
_start:
{
lean_object* v___x_361_; 
v___x_361_ = lp_mathlib_Set_fintypeDiv___redArg(v_inst_355_, v_inst_356_, v_inst_359_, v_inst_360_);
return v___x_361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeSub___redArg(lean_object* v_inst_362_, lean_object* v_inst_363_, lean_object* v_inst_364_, lean_object* v_inst_365_){
_start:
{
lean_object* v___f_366_; lean_object* v___x_367_; 
v___f_366_ = lean_alloc_closure((void*)(lp_mathlib_Set_fintypeMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_366_, 0, v_inst_362_);
v___x_367_ = lp_mathlib_Set_fintypeImage2___redArg(v_inst_363_, v___f_366_, v_inst_364_, v_inst_365_);
return v___x_367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeSub(lean_object* v_00_u03b1_368_, lean_object* v_inst_369_, lean_object* v_inst_370_, lean_object* v_s_371_, lean_object* v_t_372_, lean_object* v_inst_373_, lean_object* v_inst_374_){
_start:
{
lean_object* v___x_375_; 
v___x_375_ = lp_mathlib_Set_fintypeSub___redArg(v_inst_369_, v_inst_370_, v_inst_373_, v_inst_374_);
return v___x_375_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Pointwise_Set_Finite_0__AddGroup_card__nsmul__eq__card__nsmul__card__univ_match__1__1___redArg(lean_object* v_x_376_, lean_object* v_h__1_377_){
_start:
{
lean_object* v___x_378_; 
v___x_378_ = lean_apply_2(v_h__1_377_, v_x_376_, lean_box(0));
return v___x_378_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_Pointwise_Set_Finite_0__AddGroup_card__nsmul__eq__card__nsmul__card__univ_match__1__1(lean_object* v_G_379_, lean_object* v_s_380_, lean_object* v_motive_381_, lean_object* v_x_382_, lean_object* v_h__1_383_){
_start:
{
lean_object* v___x_384_; 
v___x_384_ = lean_apply_2(v_h__1_383_, v_x_382_, lean_box(0));
return v___x_384_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Scalar(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Finite_Prod(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Finite(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Scalar(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Finite_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Finite(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Scalar(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Basic_Finite_Prod(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Finite(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Scalar(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Finite_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Finite(builtin);
}
#ifdef __cplusplus
}
#endif
