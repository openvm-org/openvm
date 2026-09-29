// Lean compiler output
// Module: Mathlib.Algebra.Group.Invertible.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Commute.Units public import Mathlib.Algebra.Group.Invertible.Defs public import Mathlib.Algebra.Group.Hom.Defs public import Mathlib.Logic.Equiv.Defs
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
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_invertibleMul___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Units_ofPowEqOne___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_unitOfInvertible___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_unitOfInvertible(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_unitOfInvertible___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_invertible___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_invertible___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_invertible(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_invertible___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfInvertibleMul___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfInvertibleMul___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfInvertibleMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfInvertibleMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfMulInvertible___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfMulInvertible___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfMulInvertible(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfMulInvertible___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___elam__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___elam__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___elam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___elam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___elam__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___elam__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___elam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___elam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulRight___elam__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulRight___elam__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulRight___elam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulRight___elam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulRight___elam__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulRight___elam__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulRight___elam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulRight___elam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulRight___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulRight(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertiblePow___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertiblePow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertiblePow___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfPowEqOne___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfPowEqOne___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfPowEqOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfPowEqOne___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_map___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_ofLeftInverse___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_ofLeftInverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Invertible_ofLeftInverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleEquivOfLeftInverse___elam__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleEquivOfLeftInverse___elam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleEquivOfLeftInverse___elam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleEquivOfLeftInverse___elam__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleEquivOfLeftInverse___elam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleEquivOfLeftInverse___elam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleEquivOfLeftInverse___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleEquivOfLeftInverse___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleEquivOfLeftInverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_invertibleEquivOfLeftInverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_unitOfInvertible___redArg(lean_object* v_a_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3_, 0, v_a_1_);
lean_ctor_set(v___x_3_, 1, v_inst_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_unitOfInvertible(lean_object* v_00_u03b1_4_, lean_object* v_inst_5_, lean_object* v_a_6_, lean_object* v_inst_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_8_, 0, v_a_6_);
lean_ctor_set(v___x_8_, 1, v_inst_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_unitOfInvertible___boxed(lean_object* v_00_u03b1_9_, lean_object* v_inst_10_, lean_object* v_a_11_, lean_object* v_inst_12_){
_start:
{
lean_object* v_res_13_; 
v_res_13_ = lp_mathlib_unitOfInvertible(v_00_u03b1_9_, v_inst_10_, v_a_11_, v_inst_12_);
lean_dec_ref(v_inst_10_);
return v_res_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_invertible___redArg(lean_object* v_u_14_){
_start:
{
lean_object* v_inv_15_; 
v_inv_15_ = lean_ctor_get(v_u_14_, 1);
lean_inc(v_inv_15_);
return v_inv_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_invertible___redArg___boxed(lean_object* v_u_16_){
_start:
{
lean_object* v_res_17_; 
v_res_17_ = lp_mathlib_Units_invertible___redArg(v_u_16_);
lean_dec_ref(v_u_16_);
return v_res_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_invertible(lean_object* v_00_u03b1_18_, lean_object* v_inst_19_, lean_object* v_u_20_){
_start:
{
lean_object* v_inv_21_; 
v_inv_21_ = lean_ctor_get(v_u_20_, 1);
lean_inc(v_inv_21_);
return v_inv_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_invertible___boxed(lean_object* v_00_u03b1_22_, lean_object* v_inst_23_, lean_object* v_u_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_mathlib_Units_invertible(v_00_u03b1_22_, v_inst_23_, v_u_24_);
lean_dec_ref(v_u_24_);
lean_dec_ref(v_inst_23_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfInvertibleMul___redArg(lean_object* v_inst_26_, lean_object* v_a_27_, lean_object* v_inst_28_){
_start:
{
lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v_toMul_31_; lean_object* v___x_32_; 
v___x_29_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_26_);
v___x_30_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_29_);
v_toMul_31_ = lean_ctor_get(v___x_30_, 1);
lean_inc(v_toMul_31_);
lean_dec_ref(v___x_30_);
v___x_32_ = lean_apply_2(v_toMul_31_, v_inst_28_, v_a_27_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfInvertibleMul___redArg___boxed(lean_object* v_inst_33_, lean_object* v_a_34_, lean_object* v_inst_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_mathlib_invertibleOfInvertibleMul___redArg(v_inst_33_, v_a_34_, v_inst_35_);
lean_dec_ref(v_inst_33_);
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfInvertibleMul(lean_object* v_00_u03b1_37_, lean_object* v_inst_38_, lean_object* v_a_39_, lean_object* v_b_40_, lean_object* v_inst_41_, lean_object* v_inst_42_){
_start:
{
lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v_toMul_45_; lean_object* v___x_46_; 
v___x_43_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_38_);
v___x_44_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_43_);
v_toMul_45_ = lean_ctor_get(v___x_44_, 1);
lean_inc(v_toMul_45_);
lean_dec_ref(v___x_44_);
v___x_46_ = lean_apply_2(v_toMul_45_, v_inst_42_, v_a_39_);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfInvertibleMul___boxed(lean_object* v_00_u03b1_47_, lean_object* v_inst_48_, lean_object* v_a_49_, lean_object* v_b_50_, lean_object* v_inst_51_, lean_object* v_inst_52_){
_start:
{
lean_object* v_res_53_; 
v_res_53_ = lp_mathlib_invertibleOfInvertibleMul(v_00_u03b1_47_, v_inst_48_, v_a_49_, v_b_50_, v_inst_51_, v_inst_52_);
lean_dec(v_inst_51_);
lean_dec(v_b_50_);
lean_dec_ref(v_inst_48_);
return v_res_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfMulInvertible___redArg(lean_object* v_inst_54_, lean_object* v_b_55_, lean_object* v_inst_56_){
_start:
{
lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v_toMul_59_; lean_object* v___x_60_; 
v___x_57_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_54_);
v___x_58_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_57_);
v_toMul_59_ = lean_ctor_get(v___x_58_, 1);
lean_inc(v_toMul_59_);
lean_dec_ref(v___x_58_);
v___x_60_ = lean_apply_2(v_toMul_59_, v_b_55_, v_inst_56_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfMulInvertible___redArg___boxed(lean_object* v_inst_61_, lean_object* v_b_62_, lean_object* v_inst_63_){
_start:
{
lean_object* v_res_64_; 
v_res_64_ = lp_mathlib_invertibleOfMulInvertible___redArg(v_inst_61_, v_b_62_, v_inst_63_);
lean_dec_ref(v_inst_61_);
return v_res_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfMulInvertible(lean_object* v_00_u03b1_65_, lean_object* v_inst_66_, lean_object* v_a_67_, lean_object* v_b_68_, lean_object* v_inst_69_, lean_object* v_inst_70_){
_start:
{
lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v_toMul_73_; lean_object* v___x_74_; 
v___x_71_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_66_);
v___x_72_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_71_);
v_toMul_73_ = lean_ctor_get(v___x_72_, 1);
lean_inc(v_toMul_73_);
lean_dec_ref(v___x_72_);
v___x_74_ = lean_apply_2(v_toMul_73_, v_b_68_, v_inst_69_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfMulInvertible___boxed(lean_object* v_00_u03b1_75_, lean_object* v_inst_76_, lean_object* v_a_77_, lean_object* v_b_78_, lean_object* v_inst_79_, lean_object* v_inst_80_){
_start:
{
lean_object* v_res_81_; 
v_res_81_ = lp_mathlib_invertibleOfMulInvertible(v_00_u03b1_75_, v_inst_76_, v_a_77_, v_b_78_, v_inst_79_, v_inst_80_);
lean_dec(v_inst_80_);
lean_dec(v_a_77_);
lean_dec_ref(v_inst_76_);
return v_res_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___elam__0___redArg(lean_object* v_inst_82_, lean_object* v_x_83_, lean_object* v_x_84_){
_start:
{
lean_object* v___x_85_; 
v___x_85_ = lp_mathlib_invertibleMul___redArg(v_inst_82_, v_x_83_, v_x_84_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___elam__0___redArg___boxed(lean_object* v_inst_86_, lean_object* v_x_87_, lean_object* v_x_88_){
_start:
{
lean_object* v_res_89_; 
v_res_89_ = lp_mathlib_Invertible_mulLeft___elam__0___redArg(v_inst_86_, v_x_87_, v_x_88_);
lean_dec_ref(v_inst_86_);
return v_res_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___elam__0(lean_object* v_00_u03b1_90_, lean_object* v_inst_91_, lean_object* v_a_92_, lean_object* v_b_93_, lean_object* v_x_94_, lean_object* v_x_95_){
_start:
{
lean_object* v___x_96_; 
v___x_96_ = lp_mathlib_invertibleMul___redArg(v_inst_91_, v_x_94_, v_x_95_);
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___elam__0___boxed(lean_object* v_00_u03b1_97_, lean_object* v_inst_98_, lean_object* v_a_99_, lean_object* v_b_100_, lean_object* v_x_101_, lean_object* v_x_102_){
_start:
{
lean_object* v_res_103_; 
v_res_103_ = lp_mathlib_Invertible_mulLeft___elam__0(v_00_u03b1_97_, v_inst_98_, v_a_99_, v_b_100_, v_x_101_, v_x_102_);
lean_dec(v_b_100_);
lean_dec(v_a_99_);
lean_dec_ref(v_inst_98_);
return v_res_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___elam__1___redArg(lean_object* v_inst_104_, lean_object* v_a_105_, lean_object* v_x_106_){
_start:
{
lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v_toMul_109_; lean_object* v___x_110_; 
v___x_107_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_104_);
v___x_108_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_107_);
v_toMul_109_ = lean_ctor_get(v___x_108_, 1);
lean_inc(v_toMul_109_);
lean_dec_ref(v___x_108_);
v___x_110_ = lean_apply_2(v_toMul_109_, v_x_106_, v_a_105_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___elam__1___redArg___boxed(lean_object* v_inst_111_, lean_object* v_a_112_, lean_object* v_x_113_){
_start:
{
lean_object* v_res_114_; 
v_res_114_ = lp_mathlib_Invertible_mulLeft___elam__1___redArg(v_inst_111_, v_a_112_, v_x_113_);
lean_dec_ref(v_inst_111_);
return v_res_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___elam__1(lean_object* v_00_u03b1_115_, lean_object* v_inst_116_, lean_object* v_a_117_, lean_object* v_b_118_, lean_object* v_x_119_){
_start:
{
lean_object* v___x_120_; 
v___x_120_ = lp_mathlib_Invertible_mulLeft___elam__1___redArg(v_inst_116_, v_a_117_, v_x_119_);
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___elam__1___boxed(lean_object* v_00_u03b1_121_, lean_object* v_inst_122_, lean_object* v_a_123_, lean_object* v_b_124_, lean_object* v_x_125_){
_start:
{
lean_object* v_res_126_; 
v_res_126_ = lp_mathlib_Invertible_mulLeft___elam__1(v_00_u03b1_121_, v_inst_122_, v_a_123_, v_b_124_, v_x_125_);
lean_dec(v_b_124_);
lean_dec_ref(v_inst_122_);
return v_res_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft___redArg(lean_object* v_inst_127_, lean_object* v_a_128_, lean_object* v_x_129_, lean_object* v_b_130_){
_start:
{
lean_object* v___f_131_; lean_object* v___f_132_; lean_object* v___x_133_; 
lean_inc(v_b_130_);
lean_inc(v_a_128_);
lean_inc_ref(v_inst_127_);
v___f_131_ = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulLeft___elam__0___boxed), 6, 5);
lean_closure_set(v___f_131_, 0, lean_box(0));
lean_closure_set(v___f_131_, 1, v_inst_127_);
lean_closure_set(v___f_131_, 2, v_a_128_);
lean_closure_set(v___f_131_, 3, v_b_130_);
lean_closure_set(v___f_131_, 4, v_x_129_);
v___f_132_ = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulLeft___elam__1___boxed), 5, 4);
lean_closure_set(v___f_132_, 0, lean_box(0));
lean_closure_set(v___f_132_, 1, v_inst_127_);
lean_closure_set(v___f_132_, 2, v_a_128_);
lean_closure_set(v___f_132_, 3, v_b_130_);
v___x_133_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_133_, 0, v___f_131_);
lean_ctor_set(v___x_133_, 1, v___f_132_);
return v___x_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulLeft(lean_object* v_00_u03b1_134_, lean_object* v_inst_135_, lean_object* v_a_136_, lean_object* v_x_137_, lean_object* v_b_138_){
_start:
{
lean_object* v___x_139_; 
v___x_139_ = lp_mathlib_Invertible_mulLeft___redArg(v_inst_135_, v_a_136_, v_x_137_, v_b_138_);
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulRight___elam__0___redArg(lean_object* v_inst_140_, lean_object* v_x_141_, lean_object* v_x_142_){
_start:
{
lean_object* v___x_143_; 
v___x_143_ = lp_mathlib_invertibleMul___redArg(v_inst_140_, v_x_142_, v_x_141_);
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulRight___elam__0___redArg___boxed(lean_object* v_inst_144_, lean_object* v_x_145_, lean_object* v_x_146_){
_start:
{
lean_object* v_res_147_; 
v_res_147_ = lp_mathlib_Invertible_mulRight___elam__0___redArg(v_inst_144_, v_x_145_, v_x_146_);
lean_dec_ref(v_inst_144_);
return v_res_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulRight___elam__0(lean_object* v_00_u03b1_148_, lean_object* v_inst_149_, lean_object* v_a_150_, lean_object* v_b_151_, lean_object* v_x_152_, lean_object* v_x_153_){
_start:
{
lean_object* v___x_154_; 
v___x_154_ = lp_mathlib_invertibleMul___redArg(v_inst_149_, v_x_153_, v_x_152_);
return v___x_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulRight___elam__0___boxed(lean_object* v_00_u03b1_155_, lean_object* v_inst_156_, lean_object* v_a_157_, lean_object* v_b_158_, lean_object* v_x_159_, lean_object* v_x_160_){
_start:
{
lean_object* v_res_161_; 
v_res_161_ = lp_mathlib_Invertible_mulRight___elam__0(v_00_u03b1_155_, v_inst_156_, v_a_157_, v_b_158_, v_x_159_, v_x_160_);
lean_dec(v_b_158_);
lean_dec(v_a_157_);
lean_dec_ref(v_inst_156_);
return v_res_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulRight___elam__1___redArg(lean_object* v_inst_162_, lean_object* v_b_163_, lean_object* v_x_164_){
_start:
{
lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v_toMul_167_; lean_object* v___x_168_; 
v___x_165_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_162_);
v___x_166_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_165_);
v_toMul_167_ = lean_ctor_get(v___x_166_, 1);
lean_inc(v_toMul_167_);
lean_dec_ref(v___x_166_);
v___x_168_ = lean_apply_2(v_toMul_167_, v_b_163_, v_x_164_);
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulRight___elam__1___redArg___boxed(lean_object* v_inst_169_, lean_object* v_b_170_, lean_object* v_x_171_){
_start:
{
lean_object* v_res_172_; 
v_res_172_ = lp_mathlib_Invertible_mulRight___elam__1___redArg(v_inst_169_, v_b_170_, v_x_171_);
lean_dec_ref(v_inst_169_);
return v_res_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulRight___elam__1(lean_object* v_00_u03b1_173_, lean_object* v_inst_174_, lean_object* v_b_175_, lean_object* v_a_176_, lean_object* v_x_177_){
_start:
{
lean_object* v___x_178_; 
v___x_178_ = lp_mathlib_Invertible_mulRight___elam__1___redArg(v_inst_174_, v_b_175_, v_x_177_);
return v___x_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulRight___elam__1___boxed(lean_object* v_00_u03b1_179_, lean_object* v_inst_180_, lean_object* v_b_181_, lean_object* v_a_182_, lean_object* v_x_183_){
_start:
{
lean_object* v_res_184_; 
v_res_184_ = lp_mathlib_Invertible_mulRight___elam__1(v_00_u03b1_179_, v_inst_180_, v_b_181_, v_a_182_, v_x_183_);
lean_dec(v_a_182_);
lean_dec_ref(v_inst_180_);
return v_res_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulRight___redArg(lean_object* v_inst_185_, lean_object* v_a_186_, lean_object* v_b_187_, lean_object* v_x_188_){
_start:
{
lean_object* v___f_189_; lean_object* v___f_190_; lean_object* v___x_191_; 
lean_inc(v_b_187_);
lean_inc(v_a_186_);
lean_inc_ref(v_inst_185_);
v___f_189_ = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulRight___elam__0___boxed), 6, 5);
lean_closure_set(v___f_189_, 0, lean_box(0));
lean_closure_set(v___f_189_, 1, v_inst_185_);
lean_closure_set(v___f_189_, 2, v_a_186_);
lean_closure_set(v___f_189_, 3, v_b_187_);
lean_closure_set(v___f_189_, 4, v_x_188_);
v___f_190_ = lean_alloc_closure((void*)(lp_mathlib_Invertible_mulRight___elam__1___boxed), 5, 4);
lean_closure_set(v___f_190_, 0, lean_box(0));
lean_closure_set(v___f_190_, 1, v_inst_185_);
lean_closure_set(v___f_190_, 2, v_b_187_);
lean_closure_set(v___f_190_, 3, v_a_186_);
v___x_191_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_191_, 0, v___f_189_);
lean_ctor_set(v___x_191_, 1, v___f_190_);
return v___x_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_mulRight(lean_object* v_00_u03b1_192_, lean_object* v_inst_193_, lean_object* v_a_194_, lean_object* v_b_195_, lean_object* v_x_196_){
_start:
{
lean_object* v___x_197_; 
v___x_197_ = lp_mathlib_Invertible_mulRight___redArg(v_inst_193_, v_a_194_, v_b_195_, v_x_196_);
return v___x_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertiblePow___redArg(lean_object* v_inst_198_, lean_object* v_inst_199_, lean_object* v_n_200_){
_start:
{
lean_object* v_toNPow_201_; lean_object* v___x_202_; 
v_toNPow_201_ = lean_ctor_get(v_inst_198_, 2);
lean_inc(v_toNPow_201_);
lean_dec_ref(v_inst_198_);
v___x_202_ = lean_apply_2(v_toNPow_201_, v_n_200_, v_inst_199_);
return v___x_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertiblePow(lean_object* v_00_u03b1_203_, lean_object* v_inst_204_, lean_object* v_m_205_, lean_object* v_inst_206_, lean_object* v_n_207_){
_start:
{
lean_object* v___x_208_; 
v___x_208_ = lp_mathlib_invertiblePow___redArg(v_inst_204_, v_inst_206_, v_n_207_);
return v___x_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertiblePow___boxed(lean_object* v_00_u03b1_209_, lean_object* v_inst_210_, lean_object* v_m_211_, lean_object* v_inst_212_, lean_object* v_n_213_){
_start:
{
lean_object* v_res_214_; 
v_res_214_ = lp_mathlib_invertiblePow(v_00_u03b1_209_, v_inst_210_, v_m_211_, v_inst_212_, v_n_213_);
lean_dec(v_m_211_);
return v_res_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfPowEqOne___redArg(lean_object* v_inst_215_, lean_object* v_x_216_, lean_object* v_n_217_){
_start:
{
lean_object* v___x_218_; lean_object* v_inv_219_; 
v___x_218_ = lp_mathlib_Units_ofPowEqOne___redArg(v_inst_215_, v_x_216_, v_n_217_);
v_inv_219_ = lean_ctor_get(v___x_218_, 1);
lean_inc(v_inv_219_);
lean_dec_ref(v___x_218_);
return v_inv_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfPowEqOne___redArg___boxed(lean_object* v_inst_220_, lean_object* v_x_221_, lean_object* v_n_222_){
_start:
{
lean_object* v_res_223_; 
v_res_223_ = lp_mathlib_invertibleOfPowEqOne___redArg(v_inst_220_, v_x_221_, v_n_222_);
lean_dec(v_n_222_);
return v_res_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfPowEqOne(lean_object* v_00_u03b1_224_, lean_object* v_inst_225_, lean_object* v_x_226_, lean_object* v_n_227_, lean_object* v_hx_228_, lean_object* v_hn_229_){
_start:
{
lean_object* v___x_230_; 
v___x_230_ = lp_mathlib_invertibleOfPowEqOne___redArg(v_inst_225_, v_x_226_, v_n_227_);
return v___x_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleOfPowEqOne___boxed(lean_object* v_00_u03b1_231_, lean_object* v_inst_232_, lean_object* v_x_233_, lean_object* v_n_234_, lean_object* v_hx_235_, lean_object* v_hn_236_){
_start:
{
lean_object* v_res_237_; 
v_res_237_ = lp_mathlib_invertibleOfPowEqOne(v_00_u03b1_231_, v_inst_232_, v_x_233_, v_n_234_, v_hx_235_, v_hn_236_);
lean_dec(v_n_234_);
return v_res_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_map___redArg(lean_object* v_inst_238_, lean_object* v_f_239_, lean_object* v_inst_240_){
_start:
{
lean_object* v___x_241_; 
v___x_241_ = lean_apply_2(v_inst_238_, v_f_239_, v_inst_240_);
return v___x_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_map(lean_object* v_R_242_, lean_object* v_S_243_, lean_object* v_F_244_, lean_object* v_inst_245_, lean_object* v_inst_246_, lean_object* v_inst_247_, lean_object* v_inst_248_, lean_object* v_f_249_, lean_object* v_r_250_, lean_object* v_inst_251_){
_start:
{
lean_object* v___x_252_; 
v___x_252_ = lean_apply_2(v_inst_247_, v_f_249_, v_inst_251_);
return v___x_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_map___boxed(lean_object* v_R_253_, lean_object* v_S_254_, lean_object* v_F_255_, lean_object* v_inst_256_, lean_object* v_inst_257_, lean_object* v_inst_258_, lean_object* v_inst_259_, lean_object* v_f_260_, lean_object* v_r_261_, lean_object* v_inst_262_){
_start:
{
lean_object* v_res_263_; 
v_res_263_ = lp_mathlib_Invertible_map(v_R_253_, v_S_254_, v_F_255_, v_inst_256_, v_inst_257_, v_inst_258_, v_inst_259_, v_f_260_, v_r_261_, v_inst_262_);
lean_dec(v_r_261_);
lean_dec_ref(v_inst_257_);
lean_dec_ref(v_inst_256_);
return v_res_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_ofLeftInverse___redArg(lean_object* v_inst_264_, lean_object* v_g_265_, lean_object* v_inst_266_){
_start:
{
lean_object* v___x_267_; 
v___x_267_ = lean_apply_2(v_inst_264_, v_g_265_, v_inst_266_);
return v___x_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_ofLeftInverse(lean_object* v_R_268_, lean_object* v_S_269_, lean_object* v_G_270_, lean_object* v_inst_271_, lean_object* v_inst_272_, lean_object* v_inst_273_, lean_object* v_inst_274_, lean_object* v_f_275_, lean_object* v_g_276_, lean_object* v_r_277_, lean_object* v_h_278_, lean_object* v_inst_279_){
_start:
{
lean_object* v___x_280_; 
v___x_280_ = lean_apply_2(v_inst_273_, v_g_276_, v_inst_279_);
return v___x_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Invertible_ofLeftInverse___boxed(lean_object* v_R_281_, lean_object* v_S_282_, lean_object* v_G_283_, lean_object* v_inst_284_, lean_object* v_inst_285_, lean_object* v_inst_286_, lean_object* v_inst_287_, lean_object* v_f_288_, lean_object* v_g_289_, lean_object* v_r_290_, lean_object* v_h_291_, lean_object* v_inst_292_){
_start:
{
lean_object* v_res_293_; 
v_res_293_ = lp_mathlib_Invertible_ofLeftInverse(v_R_281_, v_S_282_, v_G_283_, v_inst_284_, v_inst_285_, v_inst_286_, v_inst_287_, v_f_288_, v_g_289_, v_r_290_, v_h_291_, v_inst_292_);
lean_dec(v_r_290_);
lean_dec(v_f_288_);
lean_dec_ref(v_inst_285_);
lean_dec_ref(v_inst_284_);
return v_res_293_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleEquivOfLeftInverse___elam__0___redArg(lean_object* v_inst_294_, lean_object* v_f_295_, lean_object* v_x_296_){
_start:
{
lean_object* v___x_297_; 
v___x_297_ = lean_apply_2(v_inst_294_, v_f_295_, v_x_296_);
return v___x_297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleEquivOfLeftInverse___elam__0(lean_object* v_R_298_, lean_object* v_S_299_, lean_object* v_F_300_, lean_object* v___x_301_, lean_object* v___x_302_, lean_object* v_inst_303_, lean_object* v_f_304_, lean_object* v_r_305_, lean_object* v_x_306_){
_start:
{
lean_object* v___x_307_; 
v___x_307_ = lean_apply_2(v_inst_303_, v_f_304_, v_x_306_);
return v___x_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleEquivOfLeftInverse___elam__0___boxed(lean_object* v_R_308_, lean_object* v_S_309_, lean_object* v_F_310_, lean_object* v___x_311_, lean_object* v___x_312_, lean_object* v_inst_313_, lean_object* v_f_314_, lean_object* v_r_315_, lean_object* v_x_316_){
_start:
{
lean_object* v_res_317_; 
v_res_317_ = lp_mathlib_invertibleEquivOfLeftInverse___elam__0(v_R_308_, v_S_309_, v_F_310_, v___x_311_, v___x_312_, v_inst_313_, v_f_314_, v_r_315_, v_x_316_);
lean_dec(v_r_315_);
lean_dec_ref(v___x_312_);
lean_dec_ref(v___x_311_);
return v_res_317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleEquivOfLeftInverse___elam__1___redArg(lean_object* v_inst_318_, lean_object* v_g_319_, lean_object* v_x_320_){
_start:
{
lean_object* v___x_321_; 
v___x_321_ = lean_apply_2(v_inst_318_, v_g_319_, v_x_320_);
return v___x_321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleEquivOfLeftInverse___elam__1(lean_object* v_S_322_, lean_object* v_R_323_, lean_object* v_F_324_, lean_object* v_inst_325_, lean_object* v_f_326_, lean_object* v_G_327_, lean_object* v___x_328_, lean_object* v___x_329_, lean_object* v_inst_330_, lean_object* v_g_331_, lean_object* v_r_332_, lean_object* v_x_333_){
_start:
{
lean_object* v___x_334_; 
v___x_334_ = lean_apply_2(v_inst_330_, v_g_331_, v_x_333_);
return v___x_334_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleEquivOfLeftInverse___elam__1___boxed(lean_object* v_S_335_, lean_object* v_R_336_, lean_object* v_F_337_, lean_object* v_inst_338_, lean_object* v_f_339_, lean_object* v_G_340_, lean_object* v___x_341_, lean_object* v___x_342_, lean_object* v_inst_343_, lean_object* v_g_344_, lean_object* v_r_345_, lean_object* v_x_346_){
_start:
{
lean_object* v_res_347_; 
v_res_347_ = lp_mathlib_invertibleEquivOfLeftInverse___elam__1(v_S_335_, v_R_336_, v_F_337_, v_inst_338_, v_f_339_, v_G_340_, v___x_341_, v___x_342_, v_inst_343_, v_g_344_, v_r_345_, v_x_346_);
lean_dec(v_r_345_);
lean_dec_ref(v___x_342_);
lean_dec_ref(v___x_341_);
lean_dec(v_f_339_);
lean_dec(v_inst_338_);
return v_res_347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleEquivOfLeftInverse___redArg(lean_object* v_inst_348_, lean_object* v_inst_349_, lean_object* v_inst_350_, lean_object* v_inst_351_, lean_object* v_f_352_, lean_object* v_g_353_, lean_object* v_r_354_){
_start:
{
lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___f_357_; lean_object* v___f_358_; lean_object* v___x_359_; 
v___x_355_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_348_);
v___x_356_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_349_);
lean_inc(v_r_354_);
lean_inc(v_f_352_);
lean_inc(v_inst_350_);
lean_inc_ref(v___x_356_);
lean_inc_ref(v___x_355_);
v___f_357_ = lean_alloc_closure((void*)(lp_mathlib_invertibleEquivOfLeftInverse___elam__0___boxed), 9, 8);
lean_closure_set(v___f_357_, 0, lean_box(0));
lean_closure_set(v___f_357_, 1, lean_box(0));
lean_closure_set(v___f_357_, 2, lean_box(0));
lean_closure_set(v___f_357_, 3, v___x_355_);
lean_closure_set(v___f_357_, 4, v___x_356_);
lean_closure_set(v___f_357_, 5, v_inst_350_);
lean_closure_set(v___f_357_, 6, v_f_352_);
lean_closure_set(v___f_357_, 7, v_r_354_);
v___f_358_ = lean_alloc_closure((void*)(lp_mathlib_invertibleEquivOfLeftInverse___elam__1___boxed), 12, 11);
lean_closure_set(v___f_358_, 0, lean_box(0));
lean_closure_set(v___f_358_, 1, lean_box(0));
lean_closure_set(v___f_358_, 2, lean_box(0));
lean_closure_set(v___f_358_, 3, v_inst_350_);
lean_closure_set(v___f_358_, 4, v_f_352_);
lean_closure_set(v___f_358_, 5, lean_box(0));
lean_closure_set(v___f_358_, 6, v___x_355_);
lean_closure_set(v___f_358_, 7, v___x_356_);
lean_closure_set(v___f_358_, 8, v_inst_351_);
lean_closure_set(v___f_358_, 9, v_g_353_);
lean_closure_set(v___f_358_, 10, v_r_354_);
v___x_359_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_359_, 0, v___f_358_);
lean_ctor_set(v___x_359_, 1, v___f_357_);
return v___x_359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleEquivOfLeftInverse___redArg___boxed(lean_object* v_inst_360_, lean_object* v_inst_361_, lean_object* v_inst_362_, lean_object* v_inst_363_, lean_object* v_f_364_, lean_object* v_g_365_, lean_object* v_r_366_){
_start:
{
lean_object* v_res_367_; 
v_res_367_ = lp_mathlib_invertibleEquivOfLeftInverse___redArg(v_inst_360_, v_inst_361_, v_inst_362_, v_inst_363_, v_f_364_, v_g_365_, v_r_366_);
lean_dec_ref(v_inst_361_);
lean_dec_ref(v_inst_360_);
return v_res_367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleEquivOfLeftInverse(lean_object* v_R_368_, lean_object* v_S_369_, lean_object* v_F_370_, lean_object* v_G_371_, lean_object* v_inst_372_, lean_object* v_inst_373_, lean_object* v_inst_374_, lean_object* v_inst_375_, lean_object* v_inst_376_, lean_object* v_inst_377_, lean_object* v_f_378_, lean_object* v_g_379_, lean_object* v_r_380_, lean_object* v_h_381_){
_start:
{
lean_object* v___x_382_; 
v___x_382_ = lp_mathlib_invertibleEquivOfLeftInverse___redArg(v_inst_372_, v_inst_373_, v_inst_374_, v_inst_376_, v_f_378_, v_g_379_, v_r_380_);
return v___x_382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_invertibleEquivOfLeftInverse___boxed(lean_object* v_R_383_, lean_object* v_S_384_, lean_object* v_F_385_, lean_object* v_G_386_, lean_object* v_inst_387_, lean_object* v_inst_388_, lean_object* v_inst_389_, lean_object* v_inst_390_, lean_object* v_inst_391_, lean_object* v_inst_392_, lean_object* v_f_393_, lean_object* v_g_394_, lean_object* v_r_395_, lean_object* v_h_396_){
_start:
{
lean_object* v_res_397_; 
v_res_397_ = lp_mathlib_invertibleEquivOfLeftInverse(v_R_383_, v_S_384_, v_F_385_, v_G_386_, v_inst_387_, v_inst_388_, v_inst_389_, v_inst_390_, v_inst_391_, v_inst_392_, v_f_393_, v_g_394_, v_r_395_, v_h_396_);
lean_dec_ref(v_inst_388_);
lean_dec_ref(v_inst_387_);
return v_res_397_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Commute_Units(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Invertible_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Invertible_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Commute_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Invertible_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Invertible_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Commute_Units(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Invertible_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Invertible_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Commute_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Invertible_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Invertible_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Invertible_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Invertible_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
