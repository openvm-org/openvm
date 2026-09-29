// Lean compiler output
// Module: Mathlib.Algebra.Group.Pi.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Defs public import Mathlib.Algebra.Notation.Pi.Basic public import Mathlib.Basic.Unique public import Mathlib.Data.Sum.Basic public import Mathlib.Tactic.Spread
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
lean_object* lp_mathlib_Pi_instMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_Pi_instOne___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_Pi_instInv___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_commMagma___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_commMagma___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_commMagma(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addCommMagma___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addCommMagma(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_semigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_semigroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addSemigroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_commSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_commSemigroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addCommSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addCommSemigroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulOneClass___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulOneClass___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulOneClass(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addZeroClass___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addZeroClass___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addZeroClass(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_invOneClass___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_invOneClass___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_invOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_invOneClass(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_negZeroClass___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_negZeroClass___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_negZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_negZeroClass(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoid___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoid___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoid___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoid___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoid___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_commMonoid___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_commMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_commMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addCommMonoid___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addCommMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_divInvMonoid___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_divInvMonoid___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_divInvMonoid___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_divInvMonoid___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_divInvMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_divInvMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_subNegMonoid___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_subNegMonoid___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_subNegMonoid___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_subNegMonoid___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_subNegMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_subNegMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_divInvOneMonoid___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_divInvOneMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_divInvOneMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_subNegZeroMonoid___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_subNegZeroMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_subNegZeroMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_involutiveInv___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_involutiveInv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_involutiveInv(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_involutiveNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_involutiveNeg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_divisionMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_divisionMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_subtractionMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_subtractionMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_divisionCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_divisionCommMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instSubtractionCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instSubtractionCommMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_group___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_group(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addGroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_commGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_commGroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addCommGroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_leftCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_leftCancelSemigroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addLeftCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addLeftCancelSemigroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_rightCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_rightCancelSemigroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addRightCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addRightCancelSemigroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_leftCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_leftCancelMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addLeftCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addLeftCancelMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_rightCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_rightCancelMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addRightCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addRightCancelMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_cancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_cancelMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addCancelMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_cancelCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_cancelCommMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addCancelCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_addCancelCommMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSurjectiveOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSurjectiveOne___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSurjectiveOne(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSurjectiveOne___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSurjectiveZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSurjectiveZero___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSurjectiveZero(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSurjectiveZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_commMagma___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_i_2_, lean_object* v___y_3_, lean_object* v___y_4_){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_apply_3(v_inst_1_, v_i_2_, v___y_3_, v___y_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_commMagma___redArg(lean_object* v_inst_6_){
_start:
{
lean_object* v___f_7_; lean_object* v___f_8_; 
v___f_7_ = lean_alloc_closure((void*)(lp_mathlib_Pi_commMagma___redArg___lam__0), 4, 1);
lean_closure_set(v___f_7_, 0, v_inst_6_);
v___f_8_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_8_, 0, v___f_7_);
return v___f_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_commMagma(lean_object* v_I_9_, lean_object* v_f_10_, lean_object* v_inst_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lp_mathlib_Pi_commMagma___redArg(v_inst_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addCommMagma___redArg(lean_object* v_inst_13_){
_start:
{
lean_object* v___f_14_; lean_object* v___f_15_; 
v___f_14_ = lean_alloc_closure((void*)(lp_mathlib_Pi_commMagma___redArg___lam__0), 4, 1);
lean_closure_set(v___f_14_, 0, v_inst_13_);
v___f_15_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_15_, 0, v___f_14_);
return v___f_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addCommMagma(lean_object* v_I_16_, lean_object* v_f_17_, lean_object* v_inst_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lp_mathlib_Pi_addCommMagma___redArg(v_inst_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_semigroup___redArg(lean_object* v_inst_20_){
_start:
{
lean_object* v___f_21_; lean_object* v___f_22_; 
v___f_21_ = lean_alloc_closure((void*)(lp_mathlib_Pi_commMagma___redArg___lam__0), 4, 1);
lean_closure_set(v___f_21_, 0, v_inst_20_);
v___f_22_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_22_, 0, v___f_21_);
return v___f_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_semigroup(lean_object* v_I_23_, lean_object* v_f_24_, lean_object* v_inst_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lp_mathlib_Pi_semigroup___redArg(v_inst_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addSemigroup___redArg(lean_object* v_inst_27_){
_start:
{
lean_object* v___f_28_; lean_object* v___f_29_; 
v___f_28_ = lean_alloc_closure((void*)(lp_mathlib_Pi_commMagma___redArg___lam__0), 4, 1);
lean_closure_set(v___f_28_, 0, v_inst_27_);
v___f_29_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_29_, 0, v___f_28_);
return v___f_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addSemigroup(lean_object* v_I_30_, lean_object* v_f_31_, lean_object* v_inst_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lp_mathlib_Pi_addSemigroup___redArg(v_inst_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_commSemigroup___redArg(lean_object* v_inst_34_){
_start:
{
lean_object* v___f_35_; lean_object* v___x_36_; 
v___f_35_ = lean_alloc_closure((void*)(lp_mathlib_Pi_commMagma___redArg___lam__0), 4, 1);
lean_closure_set(v___f_35_, 0, v_inst_34_);
v___x_36_ = lp_mathlib_Pi_semigroup___redArg(v___f_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_commSemigroup(lean_object* v_I_37_, lean_object* v_f_38_, lean_object* v_inst_39_){
_start:
{
lean_object* v___x_40_; 
v___x_40_ = lp_mathlib_Pi_commSemigroup___redArg(v_inst_39_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addCommSemigroup___redArg(lean_object* v_inst_41_){
_start:
{
lean_object* v___f_42_; lean_object* v___x_43_; 
v___f_42_ = lean_alloc_closure((void*)(lp_mathlib_Pi_commMagma___redArg___lam__0), 4, 1);
lean_closure_set(v___f_42_, 0, v_inst_41_);
v___x_43_ = lp_mathlib_Pi_addSemigroup___redArg(v___f_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addCommSemigroup(lean_object* v_I_44_, lean_object* v_f_45_, lean_object* v_inst_46_){
_start:
{
lean_object* v___x_47_; 
v___x_47_ = lp_mathlib_Pi_addCommSemigroup___redArg(v_inst_46_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulOneClass___redArg___lam__0(lean_object* v_inst_48_, lean_object* v_i_49_){
_start:
{
lean_object* v___x_50_; lean_object* v___x_51_; lean_object* v_toOne_52_; 
v___x_50_ = lean_apply_1(v_inst_48_, v_i_49_);
v___x_51_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_50_);
v_toOne_52_ = lean_ctor_get(v___x_51_, 0);
lean_inc(v_toOne_52_);
lean_dec_ref(v___x_51_);
return v_toOne_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulOneClass___redArg___lam__1(lean_object* v_inst_53_, lean_object* v_i_54_, lean_object* v___y_55_, lean_object* v___y_56_){
_start:
{
lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v_toMul_59_; lean_object* v___x_60_; 
v___x_57_ = lean_apply_1(v_inst_53_, v_i_54_);
v___x_58_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_57_);
v_toMul_59_ = lean_ctor_get(v___x_58_, 1);
lean_inc(v_toMul_59_);
lean_dec_ref(v___x_58_);
v___x_60_ = lean_apply_2(v_toMul_59_, v___y_55_, v___y_56_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulOneClass___redArg(lean_object* v_inst_61_){
_start:
{
lean_object* v___f_62_; lean_object* v___f_63_; lean_object* v___f_64_; lean_object* v___f_65_; lean_object* v___x_66_; 
lean_inc_ref(v_inst_61_);
v___f_62_ = lean_alloc_closure((void*)(lp_mathlib_Pi_mulOneClass___redArg___lam__0), 2, 1);
lean_closure_set(v___f_62_, 0, v_inst_61_);
v___f_63_ = lean_alloc_closure((void*)(lp_mathlib_Pi_mulOneClass___redArg___lam__1), 4, 1);
lean_closure_set(v___f_63_, 0, v_inst_61_);
v___f_64_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instOne___redArg___lam__0), 2, 1);
lean_closure_set(v___f_64_, 0, v___f_62_);
v___f_65_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_65_, 0, v___f_63_);
v___x_66_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_66_, 0, v___f_64_);
lean_ctor_set(v___x_66_, 1, v___f_65_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_mulOneClass(lean_object* v_I_67_, lean_object* v_f_68_, lean_object* v_inst_69_){
_start:
{
lean_object* v___x_70_; 
v___x_70_ = lp_mathlib_Pi_mulOneClass___redArg(v_inst_69_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addZeroClass___redArg___lam__0(lean_object* v_inst_71_, lean_object* v_i_72_){
_start:
{
lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v_toZero_75_; 
v___x_73_ = lean_apply_1(v_inst_71_, v_i_72_);
v___x_74_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_73_);
v_toZero_75_ = lean_ctor_get(v___x_74_, 0);
lean_inc(v_toZero_75_);
lean_dec_ref(v___x_74_);
return v_toZero_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addZeroClass___redArg___lam__1(lean_object* v_inst_76_, lean_object* v_i_77_, lean_object* v___y_78_, lean_object* v___y_79_){
_start:
{
lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v_toAdd_82_; lean_object* v___x_83_; 
v___x_80_ = lean_apply_1(v_inst_76_, v_i_77_);
v___x_81_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_80_);
v_toAdd_82_ = lean_ctor_get(v___x_81_, 1);
lean_inc(v_toAdd_82_);
lean_dec_ref(v___x_81_);
v___x_83_ = lean_apply_2(v_toAdd_82_, v___y_78_, v___y_79_);
return v___x_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addZeroClass___redArg(lean_object* v_inst_84_){
_start:
{
lean_object* v___f_85_; lean_object* v___f_86_; lean_object* v___f_87_; lean_object* v___f_88_; lean_object* v___x_89_; 
lean_inc_ref(v_inst_84_);
v___f_85_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addZeroClass___redArg___lam__0), 2, 1);
lean_closure_set(v___f_85_, 0, v_inst_84_);
v___f_86_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addZeroClass___redArg___lam__1), 4, 1);
lean_closure_set(v___f_86_, 0, v_inst_84_);
v___f_87_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instOne___redArg___lam__0), 2, 1);
lean_closure_set(v___f_87_, 0, v___f_85_);
v___f_88_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_88_, 0, v___f_86_);
v___x_89_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_89_, 0, v___f_87_);
lean_ctor_set(v___x_89_, 1, v___f_88_);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addZeroClass(lean_object* v_I_90_, lean_object* v_f_91_, lean_object* v_inst_92_){
_start:
{
lean_object* v___x_93_; 
v___x_93_ = lp_mathlib_Pi_addZeroClass___redArg(v_inst_92_);
return v___x_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_invOneClass___redArg___lam__0(lean_object* v_inst_94_, lean_object* v_i_95_){
_start:
{
lean_object* v___x_96_; lean_object* v_toOne_97_; 
v___x_96_ = lean_apply_1(v_inst_94_, v_i_95_);
v_toOne_97_ = lean_ctor_get(v___x_96_, 0);
lean_inc(v_toOne_97_);
lean_dec_ref(v___x_96_);
return v_toOne_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_invOneClass___redArg___lam__1(lean_object* v_inst_98_, lean_object* v_i_99_, lean_object* v___y_100_){
_start:
{
lean_object* v___x_101_; lean_object* v_toInv_102_; lean_object* v___x_103_; 
v___x_101_ = lean_apply_1(v_inst_98_, v_i_99_);
v_toInv_102_ = lean_ctor_get(v___x_101_, 1);
lean_inc(v_toInv_102_);
lean_dec_ref(v___x_101_);
v___x_103_ = lean_apply_1(v_toInv_102_, v___y_100_);
return v___x_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_invOneClass___redArg(lean_object* v_inst_104_){
_start:
{
lean_object* v___f_105_; lean_object* v___f_106_; lean_object* v___f_107_; lean_object* v___f_108_; lean_object* v___x_109_; 
lean_inc_ref(v_inst_104_);
v___f_105_ = lean_alloc_closure((void*)(lp_mathlib_Pi_invOneClass___redArg___lam__0), 2, 1);
lean_closure_set(v___f_105_, 0, v_inst_104_);
v___f_106_ = lean_alloc_closure((void*)(lp_mathlib_Pi_invOneClass___redArg___lam__1), 3, 1);
lean_closure_set(v___f_106_, 0, v_inst_104_);
v___f_107_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instOne___redArg___lam__0), 2, 1);
lean_closure_set(v___f_107_, 0, v___f_105_);
v___f_108_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instInv___redArg___lam__0), 3, 1);
lean_closure_set(v___f_108_, 0, v___f_106_);
v___x_109_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_109_, 0, v___f_107_);
lean_ctor_set(v___x_109_, 1, v___f_108_);
return v___x_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_invOneClass(lean_object* v_I_110_, lean_object* v_f_111_, lean_object* v_inst_112_){
_start:
{
lean_object* v___x_113_; 
v___x_113_ = lp_mathlib_Pi_invOneClass___redArg(v_inst_112_);
return v___x_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_negZeroClass___redArg___lam__0(lean_object* v_inst_114_, lean_object* v_i_115_){
_start:
{
lean_object* v___x_116_; lean_object* v_toZero_117_; 
v___x_116_ = lean_apply_1(v_inst_114_, v_i_115_);
v_toZero_117_ = lean_ctor_get(v___x_116_, 0);
lean_inc(v_toZero_117_);
lean_dec_ref(v___x_116_);
return v_toZero_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_negZeroClass___redArg___lam__1(lean_object* v_inst_118_, lean_object* v_i_119_, lean_object* v___y_120_){
_start:
{
lean_object* v___x_121_; lean_object* v_toNeg_122_; lean_object* v___x_123_; 
v___x_121_ = lean_apply_1(v_inst_118_, v_i_119_);
v_toNeg_122_ = lean_ctor_get(v___x_121_, 1);
lean_inc(v_toNeg_122_);
lean_dec_ref(v___x_121_);
v___x_123_ = lean_apply_1(v_toNeg_122_, v___y_120_);
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_negZeroClass___redArg(lean_object* v_inst_124_){
_start:
{
lean_object* v___f_125_; lean_object* v___f_126_; lean_object* v___f_127_; lean_object* v___f_128_; lean_object* v___x_129_; 
lean_inc_ref(v_inst_124_);
v___f_125_ = lean_alloc_closure((void*)(lp_mathlib_Pi_negZeroClass___redArg___lam__0), 2, 1);
lean_closure_set(v___f_125_, 0, v_inst_124_);
v___f_126_ = lean_alloc_closure((void*)(lp_mathlib_Pi_negZeroClass___redArg___lam__1), 3, 1);
lean_closure_set(v___f_126_, 0, v_inst_124_);
v___f_127_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instOne___redArg___lam__0), 2, 1);
lean_closure_set(v___f_127_, 0, v___f_125_);
v___f_128_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instInv___redArg___lam__0), 3, 1);
lean_closure_set(v___f_128_, 0, v___f_126_);
v___x_129_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_129_, 0, v___f_127_);
lean_ctor_set(v___x_129_, 1, v___f_128_);
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_negZeroClass(lean_object* v_I_130_, lean_object* v_f_131_, lean_object* v_inst_132_){
_start:
{
lean_object* v___x_133_; 
v___x_133_ = lp_mathlib_Pi_negZeroClass___redArg(v_inst_132_);
return v___x_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoid___redArg___lam__0(lean_object* v_inst_134_, lean_object* v_i_135_, lean_object* v___y_136_, lean_object* v___y_137_){
_start:
{
lean_object* v___x_138_; lean_object* v_toMul_139_; lean_object* v___x_140_; 
v___x_138_ = lean_apply_1(v_inst_134_, v_i_135_);
v_toMul_139_ = lean_ctor_get(v___x_138_, 1);
lean_inc(v_toMul_139_);
lean_dec_ref(v___x_138_);
v___x_140_ = lean_apply_2(v_toMul_139_, v___y_136_, v___y_137_);
return v___x_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoid___redArg___lam__1(lean_object* v_inst_141_, lean_object* v_i_142_){
_start:
{
lean_object* v___x_143_; lean_object* v___x_144_; 
v___x_143_ = lean_apply_1(v_inst_141_, v_i_142_);
v___x_144_ = lp_mathlib_Monoid_toMulOneClass___redArg(v___x_143_);
lean_dec_ref(v___x_143_);
return v___x_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoid___redArg___lam__2(lean_object* v_inst_145_, lean_object* v_n_146_, lean_object* v_x_147_, lean_object* v_i_148_){
_start:
{
lean_object* v___x_149_; lean_object* v_toNPow_150_; lean_object* v___x_151_; lean_object* v___x_152_; 
lean_inc(v_i_148_);
v___x_149_ = lean_apply_1(v_inst_145_, v_i_148_);
v_toNPow_150_ = lean_ctor_get(v___x_149_, 2);
lean_inc(v_toNPow_150_);
lean_dec_ref(v___x_149_);
v___x_151_ = lean_apply_1(v_x_147_, v_i_148_);
v___x_152_ = lean_apply_2(v_toNPow_150_, v_n_146_, v___x_151_);
return v___x_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoid___redArg(lean_object* v_inst_153_){
_start:
{
lean_object* v___f_154_; lean_object* v___f_155_; lean_object* v___f_156_; lean_object* v___x_157_; lean_object* v___f_158_; lean_object* v___f_159_; lean_object* v___x_160_; 
lean_inc_ref_n(v_inst_153_, 2);
v___f_154_ = lean_alloc_closure((void*)(lp_mathlib_Pi_monoid___redArg___lam__0), 4, 1);
lean_closure_set(v___f_154_, 0, v_inst_153_);
v___f_155_ = lean_alloc_closure((void*)(lp_mathlib_Pi_monoid___redArg___lam__1), 2, 1);
lean_closure_set(v___f_155_, 0, v_inst_153_);
v___f_156_ = lean_alloc_closure((void*)(lp_mathlib_Pi_monoid___redArg___lam__2), 4, 1);
lean_closure_set(v___f_156_, 0, v_inst_153_);
v___x_157_ = lp_mathlib_Pi_semigroup___redArg(v___f_154_);
v___f_158_ = lean_alloc_closure((void*)(lp_mathlib_Pi_mulOneClass___redArg___lam__0), 2, 1);
lean_closure_set(v___f_158_, 0, v___f_155_);
v___f_159_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instOne___redArg___lam__0), 2, 1);
lean_closure_set(v___f_159_, 0, v___f_158_);
v___x_160_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_160_, 0, v___f_159_);
lean_ctor_set(v___x_160_, 1, v___x_157_);
lean_ctor_set(v___x_160_, 2, v___f_156_);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_monoid(lean_object* v_I_161_, lean_object* v_f_162_, lean_object* v_inst_163_){
_start:
{
lean_object* v___x_164_; 
v___x_164_ = lp_mathlib_Pi_monoid___redArg(v_inst_163_);
return v___x_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoid___redArg___lam__0(lean_object* v_inst_165_, lean_object* v_i_166_, lean_object* v___y_167_, lean_object* v___y_168_){
_start:
{
lean_object* v___x_169_; lean_object* v_toAdd_170_; lean_object* v___x_171_; 
v___x_169_ = lean_apply_1(v_inst_165_, v_i_166_);
v_toAdd_170_ = lean_ctor_get(v___x_169_, 1);
lean_inc(v_toAdd_170_);
lean_dec_ref(v___x_169_);
v___x_171_ = lean_apply_2(v_toAdd_170_, v___y_167_, v___y_168_);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoid___redArg___lam__1(lean_object* v_inst_172_, lean_object* v_i_173_){
_start:
{
lean_object* v___x_174_; lean_object* v___x_175_; 
v___x_174_ = lean_apply_1(v_inst_172_, v_i_173_);
v___x_175_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v___x_174_);
lean_dec_ref(v___x_174_);
return v___x_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoid___redArg___lam__2(lean_object* v_inst_176_, lean_object* v_n_177_, lean_object* v_x_178_, lean_object* v_i_179_){
_start:
{
lean_object* v___x_180_; lean_object* v_toNSMul_181_; lean_object* v___x_182_; lean_object* v___x_183_; 
lean_inc(v_i_179_);
v___x_180_ = lean_apply_1(v_inst_176_, v_i_179_);
v_toNSMul_181_ = lean_ctor_get(v___x_180_, 2);
lean_inc(v_toNSMul_181_);
lean_dec_ref(v___x_180_);
v___x_182_ = lean_apply_1(v_x_178_, v_i_179_);
v___x_183_ = lean_apply_2(v_toNSMul_181_, v_n_177_, v___x_182_);
return v___x_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoid___redArg(lean_object* v_inst_184_){
_start:
{
lean_object* v___f_185_; lean_object* v___f_186_; lean_object* v___f_187_; lean_object* v___x_188_; lean_object* v___f_189_; lean_object* v___f_190_; lean_object* v___x_191_; 
lean_inc_ref_n(v_inst_184_, 2);
v___f_185_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addMonoid___redArg___lam__0), 4, 1);
lean_closure_set(v___f_185_, 0, v_inst_184_);
v___f_186_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addMonoid___redArg___lam__1), 2, 1);
lean_closure_set(v___f_186_, 0, v_inst_184_);
v___f_187_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addMonoid___redArg___lam__2), 4, 1);
lean_closure_set(v___f_187_, 0, v_inst_184_);
v___x_188_ = lp_mathlib_Pi_addSemigroup___redArg(v___f_185_);
v___f_189_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addZeroClass___redArg___lam__0), 2, 1);
lean_closure_set(v___f_189_, 0, v___f_186_);
v___f_190_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instOne___redArg___lam__0), 2, 1);
lean_closure_set(v___f_190_, 0, v___f_189_);
v___x_191_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_191_, 0, v___f_190_);
lean_ctor_set(v___x_191_, 1, v___x_188_);
lean_ctor_set(v___x_191_, 2, v___f_187_);
return v___x_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addMonoid(lean_object* v_I_192_, lean_object* v_f_193_, lean_object* v_inst_194_){
_start:
{
lean_object* v___x_195_; 
v___x_195_ = lp_mathlib_Pi_addMonoid___redArg(v_inst_194_);
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_commMonoid___redArg___lam__0(lean_object* v_inst_196_, lean_object* v_i_197_){
_start:
{
lean_object* v___x_198_; 
v___x_198_ = lean_apply_1(v_inst_196_, v_i_197_);
return v___x_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_commMonoid___redArg(lean_object* v_inst_199_){
_start:
{
lean_object* v___f_200_; lean_object* v___x_201_; 
v___f_200_ = lean_alloc_closure((void*)(lp_mathlib_Pi_commMonoid___redArg___lam__0), 2, 1);
lean_closure_set(v___f_200_, 0, v_inst_199_);
v___x_201_ = lp_mathlib_Pi_monoid___redArg(v___f_200_);
return v___x_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_commMonoid(lean_object* v_I_202_, lean_object* v_f_203_, lean_object* v_inst_204_){
_start:
{
lean_object* v___x_205_; 
v___x_205_ = lp_mathlib_Pi_commMonoid___redArg(v_inst_204_);
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addCommMonoid___redArg___lam__0(lean_object* v_inst_206_, lean_object* v_i_207_){
_start:
{
lean_object* v___x_208_; 
v___x_208_ = lean_apply_1(v_inst_206_, v_i_207_);
return v___x_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addCommMonoid___redArg(lean_object* v_inst_209_){
_start:
{
lean_object* v___f_210_; lean_object* v___x_211_; 
v___f_210_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addCommMonoid___redArg___lam__0), 2, 1);
lean_closure_set(v___f_210_, 0, v_inst_209_);
v___x_211_ = lp_mathlib_Pi_addMonoid___redArg(v___f_210_);
return v___x_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addCommMonoid(lean_object* v_I_212_, lean_object* v_f_213_, lean_object* v_inst_214_){
_start:
{
lean_object* v___x_215_; 
v___x_215_ = lp_mathlib_Pi_addCommMonoid___redArg(v_inst_214_);
return v___x_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_divInvMonoid___redArg___lam__0(lean_object* v_inst_216_, lean_object* v_i_217_){
_start:
{
lean_object* v___x_218_; lean_object* v_toMonoid_219_; 
v___x_218_ = lean_apply_1(v_inst_216_, v_i_217_);
v_toMonoid_219_ = lean_ctor_get(v___x_218_, 0);
lean_inc_ref(v_toMonoid_219_);
lean_dec_ref(v___x_218_);
return v_toMonoid_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_divInvMonoid___redArg___lam__1(lean_object* v_inst_220_, lean_object* v_i_221_, lean_object* v___y_222_){
_start:
{
lean_object* v___x_223_; lean_object* v_toInv_224_; lean_object* v___x_225_; 
v___x_223_ = lean_apply_1(v_inst_220_, v_i_221_);
v_toInv_224_ = lean_ctor_get(v___x_223_, 1);
lean_inc(v_toInv_224_);
lean_dec_ref(v___x_223_);
v___x_225_ = lean_apply_1(v_toInv_224_, v___y_222_);
return v___x_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_divInvMonoid___redArg___lam__2(lean_object* v_inst_226_, lean_object* v_i_227_, lean_object* v___y_228_, lean_object* v___y_229_){
_start:
{
lean_object* v___x_230_; lean_object* v_toDiv_231_; lean_object* v___x_232_; 
v___x_230_ = lean_apply_1(v_inst_226_, v_i_227_);
v_toDiv_231_ = lean_ctor_get(v___x_230_, 2);
lean_inc(v_toDiv_231_);
lean_dec_ref(v___x_230_);
v___x_232_ = lean_apply_2(v_toDiv_231_, v___y_228_, v___y_229_);
return v___x_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_divInvMonoid___redArg___lam__3(lean_object* v_inst_233_, lean_object* v_z_234_, lean_object* v_x_235_, lean_object* v_i_236_){
_start:
{
lean_object* v___x_237_; lean_object* v_toZPow_238_; lean_object* v___x_239_; lean_object* v___x_240_; 
lean_inc(v_i_236_);
v___x_237_ = lean_apply_1(v_inst_233_, v_i_236_);
v_toZPow_238_ = lean_ctor_get(v___x_237_, 3);
lean_inc(v_toZPow_238_);
lean_dec_ref(v___x_237_);
v___x_239_ = lean_apply_1(v_x_235_, v_i_236_);
v___x_240_ = lean_apply_2(v_toZPow_238_, v_z_234_, v___x_239_);
return v___x_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_divInvMonoid___redArg(lean_object* v_inst_241_){
_start:
{
lean_object* v___f_242_; lean_object* v___f_243_; lean_object* v___f_244_; lean_object* v___f_245_; lean_object* v___x_246_; lean_object* v___f_247_; lean_object* v___f_248_; lean_object* v___x_249_; 
lean_inc_ref_n(v_inst_241_, 3);
v___f_242_ = lean_alloc_closure((void*)(lp_mathlib_Pi_divInvMonoid___redArg___lam__0), 2, 1);
lean_closure_set(v___f_242_, 0, v_inst_241_);
v___f_243_ = lean_alloc_closure((void*)(lp_mathlib_Pi_divInvMonoid___redArg___lam__1), 3, 1);
lean_closure_set(v___f_243_, 0, v_inst_241_);
v___f_244_ = lean_alloc_closure((void*)(lp_mathlib_Pi_divInvMonoid___redArg___lam__2), 4, 1);
lean_closure_set(v___f_244_, 0, v_inst_241_);
v___f_245_ = lean_alloc_closure((void*)(lp_mathlib_Pi_divInvMonoid___redArg___lam__3), 4, 1);
lean_closure_set(v___f_245_, 0, v_inst_241_);
v___x_246_ = lp_mathlib_Pi_monoid___redArg(v___f_242_);
v___f_247_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instInv___redArg___lam__0), 3, 1);
lean_closure_set(v___f_247_, 0, v___f_243_);
v___f_248_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_248_, 0, v___f_244_);
v___x_249_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_249_, 0, v___x_246_);
lean_ctor_set(v___x_249_, 1, v___f_247_);
lean_ctor_set(v___x_249_, 2, v___f_248_);
lean_ctor_set(v___x_249_, 3, v___f_245_);
return v___x_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_divInvMonoid(lean_object* v_I_250_, lean_object* v_f_251_, lean_object* v_inst_252_){
_start:
{
lean_object* v___x_253_; 
v___x_253_ = lp_mathlib_Pi_divInvMonoid___redArg(v_inst_252_);
return v___x_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_subNegMonoid___redArg___lam__0(lean_object* v_inst_254_, lean_object* v_i_255_){
_start:
{
lean_object* v___x_256_; lean_object* v_toAddMonoid_257_; 
v___x_256_ = lean_apply_1(v_inst_254_, v_i_255_);
v_toAddMonoid_257_ = lean_ctor_get(v___x_256_, 0);
lean_inc_ref(v_toAddMonoid_257_);
lean_dec_ref(v___x_256_);
return v_toAddMonoid_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_subNegMonoid___redArg___lam__1(lean_object* v_inst_258_, lean_object* v_i_259_, lean_object* v___y_260_){
_start:
{
lean_object* v___x_261_; lean_object* v_toNeg_262_; lean_object* v___x_263_; 
v___x_261_ = lean_apply_1(v_inst_258_, v_i_259_);
v_toNeg_262_ = lean_ctor_get(v___x_261_, 1);
lean_inc(v_toNeg_262_);
lean_dec_ref(v___x_261_);
v___x_263_ = lean_apply_1(v_toNeg_262_, v___y_260_);
return v___x_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_subNegMonoid___redArg___lam__2(lean_object* v_inst_264_, lean_object* v_i_265_, lean_object* v___y_266_, lean_object* v___y_267_){
_start:
{
lean_object* v___x_268_; lean_object* v_toSub_269_; lean_object* v___x_270_; 
v___x_268_ = lean_apply_1(v_inst_264_, v_i_265_);
v_toSub_269_ = lean_ctor_get(v___x_268_, 2);
lean_inc(v_toSub_269_);
lean_dec_ref(v___x_268_);
v___x_270_ = lean_apply_2(v_toSub_269_, v___y_266_, v___y_267_);
return v___x_270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_subNegMonoid___redArg___lam__3(lean_object* v_inst_271_, lean_object* v_z_272_, lean_object* v_x_273_, lean_object* v_i_274_){
_start:
{
lean_object* v___x_275_; lean_object* v_toZSMul_276_; lean_object* v___x_277_; lean_object* v___x_278_; 
lean_inc(v_i_274_);
v___x_275_ = lean_apply_1(v_inst_271_, v_i_274_);
v_toZSMul_276_ = lean_ctor_get(v___x_275_, 3);
lean_inc(v_toZSMul_276_);
lean_dec_ref(v___x_275_);
v___x_277_ = lean_apply_1(v_x_273_, v_i_274_);
v___x_278_ = lean_apply_2(v_toZSMul_276_, v_z_272_, v___x_277_);
return v___x_278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_subNegMonoid___redArg(lean_object* v_inst_279_){
_start:
{
lean_object* v___f_280_; lean_object* v___f_281_; lean_object* v___f_282_; lean_object* v___f_283_; lean_object* v___x_284_; lean_object* v___f_285_; lean_object* v___f_286_; lean_object* v___x_287_; 
lean_inc_ref_n(v_inst_279_, 3);
v___f_280_ = lean_alloc_closure((void*)(lp_mathlib_Pi_subNegMonoid___redArg___lam__0), 2, 1);
lean_closure_set(v___f_280_, 0, v_inst_279_);
v___f_281_ = lean_alloc_closure((void*)(lp_mathlib_Pi_subNegMonoid___redArg___lam__1), 3, 1);
lean_closure_set(v___f_281_, 0, v_inst_279_);
v___f_282_ = lean_alloc_closure((void*)(lp_mathlib_Pi_subNegMonoid___redArg___lam__2), 4, 1);
lean_closure_set(v___f_282_, 0, v_inst_279_);
v___f_283_ = lean_alloc_closure((void*)(lp_mathlib_Pi_subNegMonoid___redArg___lam__3), 4, 1);
lean_closure_set(v___f_283_, 0, v_inst_279_);
v___x_284_ = lp_mathlib_Pi_addMonoid___redArg(v___f_280_);
v___f_285_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instInv___redArg___lam__0), 3, 1);
lean_closure_set(v___f_285_, 0, v___f_281_);
v___f_286_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_286_, 0, v___f_282_);
v___x_287_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_287_, 0, v___x_284_);
lean_ctor_set(v___x_287_, 1, v___f_285_);
lean_ctor_set(v___x_287_, 2, v___f_286_);
lean_ctor_set(v___x_287_, 3, v___f_283_);
return v___x_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_subNegMonoid(lean_object* v_I_288_, lean_object* v_f_289_, lean_object* v_inst_290_){
_start:
{
lean_object* v___x_291_; 
v___x_291_ = lp_mathlib_Pi_subNegMonoid___redArg(v_inst_290_);
return v___x_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_divInvOneMonoid___redArg___lam__0(lean_object* v_inst_292_, lean_object* v_i_293_){
_start:
{
lean_object* v___x_294_; 
v___x_294_ = lean_apply_1(v_inst_292_, v_i_293_);
return v___x_294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_divInvOneMonoid___redArg(lean_object* v_inst_295_){
_start:
{
lean_object* v___f_296_; lean_object* v___x_297_; 
v___f_296_ = lean_alloc_closure((void*)(lp_mathlib_Pi_divInvOneMonoid___redArg___lam__0), 2, 1);
lean_closure_set(v___f_296_, 0, v_inst_295_);
v___x_297_ = lp_mathlib_Pi_divInvMonoid___redArg(v___f_296_);
return v___x_297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_divInvOneMonoid(lean_object* v_I_298_, lean_object* v_f_299_, lean_object* v_inst_300_){
_start:
{
lean_object* v___x_301_; 
v___x_301_ = lp_mathlib_Pi_divInvOneMonoid___redArg(v_inst_300_);
return v___x_301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_subNegZeroMonoid___redArg___lam__0(lean_object* v_inst_302_, lean_object* v_i_303_){
_start:
{
lean_object* v___x_304_; 
v___x_304_ = lean_apply_1(v_inst_302_, v_i_303_);
return v___x_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_subNegZeroMonoid___redArg(lean_object* v_inst_305_){
_start:
{
lean_object* v___f_306_; lean_object* v___x_307_; 
v___f_306_ = lean_alloc_closure((void*)(lp_mathlib_Pi_subNegZeroMonoid___redArg___lam__0), 2, 1);
lean_closure_set(v___f_306_, 0, v_inst_305_);
v___x_307_ = lp_mathlib_Pi_subNegMonoid___redArg(v___f_306_);
return v___x_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_subNegZeroMonoid(lean_object* v_I_308_, lean_object* v_f_309_, lean_object* v_inst_310_){
_start:
{
lean_object* v___x_311_; 
v___x_311_ = lp_mathlib_Pi_subNegZeroMonoid___redArg(v_inst_310_);
return v___x_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_involutiveInv___redArg___lam__0(lean_object* v_inst_312_, lean_object* v_i_313_, lean_object* v___y_314_){
_start:
{
lean_object* v___x_315_; 
v___x_315_ = lean_apply_2(v_inst_312_, v_i_313_, v___y_314_);
return v___x_315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_involutiveInv___redArg(lean_object* v_inst_316_){
_start:
{
lean_object* v___f_317_; lean_object* v___f_318_; 
v___f_317_ = lean_alloc_closure((void*)(lp_mathlib_Pi_involutiveInv___redArg___lam__0), 3, 1);
lean_closure_set(v___f_317_, 0, v_inst_316_);
v___f_318_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instInv___redArg___lam__0), 3, 1);
lean_closure_set(v___f_318_, 0, v___f_317_);
return v___f_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_involutiveInv(lean_object* v_I_319_, lean_object* v_f_320_, lean_object* v_inst_321_){
_start:
{
lean_object* v___x_322_; 
v___x_322_ = lp_mathlib_Pi_involutiveInv___redArg(v_inst_321_);
return v___x_322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_involutiveNeg___redArg(lean_object* v_inst_323_){
_start:
{
lean_object* v___f_324_; lean_object* v___f_325_; 
v___f_324_ = lean_alloc_closure((void*)(lp_mathlib_Pi_involutiveInv___redArg___lam__0), 3, 1);
lean_closure_set(v___f_324_, 0, v_inst_323_);
v___f_325_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instInv___redArg___lam__0), 3, 1);
lean_closure_set(v___f_325_, 0, v___f_324_);
return v___f_325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_involutiveNeg(lean_object* v_I_326_, lean_object* v_f_327_, lean_object* v_inst_328_){
_start:
{
lean_object* v___x_329_; 
v___x_329_ = lp_mathlib_Pi_involutiveNeg___redArg(v_inst_328_);
return v___x_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_divisionMonoid___redArg(lean_object* v_inst_330_){
_start:
{
lean_object* v___f_331_; lean_object* v___x_332_; 
v___f_331_ = lean_alloc_closure((void*)(lp_mathlib_Pi_divInvOneMonoid___redArg___lam__0), 2, 1);
lean_closure_set(v___f_331_, 0, v_inst_330_);
v___x_332_ = lp_mathlib_Pi_divInvMonoid___redArg(v___f_331_);
return v___x_332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_divisionMonoid(lean_object* v_I_333_, lean_object* v_f_334_, lean_object* v_inst_335_){
_start:
{
lean_object* v___x_336_; 
v___x_336_ = lp_mathlib_Pi_divisionMonoid___redArg(v_inst_335_);
return v___x_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_subtractionMonoid___redArg(lean_object* v_inst_337_){
_start:
{
lean_object* v___f_338_; lean_object* v___x_339_; 
v___f_338_ = lean_alloc_closure((void*)(lp_mathlib_Pi_subNegZeroMonoid___redArg___lam__0), 2, 1);
lean_closure_set(v___f_338_, 0, v_inst_337_);
v___x_339_ = lp_mathlib_Pi_subNegMonoid___redArg(v___f_338_);
return v___x_339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_subtractionMonoid(lean_object* v_I_340_, lean_object* v_f_341_, lean_object* v_inst_342_){
_start:
{
lean_object* v___x_343_; 
v___x_343_ = lp_mathlib_Pi_subtractionMonoid___redArg(v_inst_342_);
return v___x_343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_divisionCommMonoid___redArg(lean_object* v_inst_344_){
_start:
{
lean_object* v___f_345_; lean_object* v___x_346_; 
v___f_345_ = lean_alloc_closure((void*)(lp_mathlib_Pi_divInvOneMonoid___redArg___lam__0), 2, 1);
lean_closure_set(v___f_345_, 0, v_inst_344_);
v___x_346_ = lp_mathlib_Pi_divisionMonoid___redArg(v___f_345_);
return v___x_346_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_divisionCommMonoid(lean_object* v_I_347_, lean_object* v_f_348_, lean_object* v_inst_349_){
_start:
{
lean_object* v___x_350_; 
v___x_350_ = lp_mathlib_Pi_divisionCommMonoid___redArg(v_inst_349_);
return v___x_350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instSubtractionCommMonoid___redArg(lean_object* v_inst_351_){
_start:
{
lean_object* v___f_352_; lean_object* v___x_353_; 
v___f_352_ = lean_alloc_closure((void*)(lp_mathlib_Pi_subNegZeroMonoid___redArg___lam__0), 2, 1);
lean_closure_set(v___f_352_, 0, v_inst_351_);
v___x_353_ = lp_mathlib_Pi_subtractionMonoid___redArg(v___f_352_);
return v___x_353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instSubtractionCommMonoid(lean_object* v_I_354_, lean_object* v_f_355_, lean_object* v_inst_356_){
_start:
{
lean_object* v___x_357_; 
v___x_357_ = lp_mathlib_Pi_instSubtractionCommMonoid___redArg(v_inst_356_);
return v___x_357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_group___redArg(lean_object* v_inst_358_){
_start:
{
lean_object* v___f_359_; lean_object* v___x_360_; 
v___f_359_ = lean_alloc_closure((void*)(lp_mathlib_Pi_divInvOneMonoid___redArg___lam__0), 2, 1);
lean_closure_set(v___f_359_, 0, v_inst_358_);
v___x_360_ = lp_mathlib_Pi_divInvMonoid___redArg(v___f_359_);
return v___x_360_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_group(lean_object* v_I_361_, lean_object* v_f_362_, lean_object* v_inst_363_){
_start:
{
lean_object* v___x_364_; 
v___x_364_ = lp_mathlib_Pi_group___redArg(v_inst_363_);
return v___x_364_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addGroup___redArg(lean_object* v_inst_365_){
_start:
{
lean_object* v___f_366_; lean_object* v___x_367_; 
v___f_366_ = lean_alloc_closure((void*)(lp_mathlib_Pi_subNegZeroMonoid___redArg___lam__0), 2, 1);
lean_closure_set(v___f_366_, 0, v_inst_365_);
v___x_367_ = lp_mathlib_Pi_subNegMonoid___redArg(v___f_366_);
return v___x_367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addGroup(lean_object* v_I_368_, lean_object* v_f_369_, lean_object* v_inst_370_){
_start:
{
lean_object* v___x_371_; 
v___x_371_ = lp_mathlib_Pi_addGroup___redArg(v_inst_370_);
return v___x_371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_commGroup___redArg(lean_object* v_inst_372_){
_start:
{
lean_object* v___f_373_; lean_object* v___x_374_; 
v___f_373_ = lean_alloc_closure((void*)(lp_mathlib_Pi_divInvOneMonoid___redArg___lam__0), 2, 1);
lean_closure_set(v___f_373_, 0, v_inst_372_);
v___x_374_ = lp_mathlib_Pi_group___redArg(v___f_373_);
return v___x_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_commGroup(lean_object* v_I_375_, lean_object* v_f_376_, lean_object* v_inst_377_){
_start:
{
lean_object* v___x_378_; 
v___x_378_ = lp_mathlib_Pi_commGroup___redArg(v_inst_377_);
return v___x_378_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addCommGroup___redArg(lean_object* v_inst_379_){
_start:
{
lean_object* v___f_380_; lean_object* v___x_381_; 
v___f_380_ = lean_alloc_closure((void*)(lp_mathlib_Pi_subNegZeroMonoid___redArg___lam__0), 2, 1);
lean_closure_set(v___f_380_, 0, v_inst_379_);
v___x_381_ = lp_mathlib_Pi_addGroup___redArg(v___f_380_);
return v___x_381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addCommGroup(lean_object* v_I_382_, lean_object* v_f_383_, lean_object* v_inst_384_){
_start:
{
lean_object* v___x_385_; 
v___x_385_ = lp_mathlib_Pi_addCommGroup___redArg(v_inst_384_);
return v___x_385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_leftCancelSemigroup___redArg(lean_object* v_inst_386_){
_start:
{
lean_object* v___f_387_; lean_object* v___x_388_; 
v___f_387_ = lean_alloc_closure((void*)(lp_mathlib_Pi_commMagma___redArg___lam__0), 4, 1);
lean_closure_set(v___f_387_, 0, v_inst_386_);
v___x_388_ = lp_mathlib_Pi_semigroup___redArg(v___f_387_);
return v___x_388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_leftCancelSemigroup(lean_object* v_I_389_, lean_object* v_f_390_, lean_object* v_inst_391_){
_start:
{
lean_object* v___x_392_; 
v___x_392_ = lp_mathlib_Pi_leftCancelSemigroup___redArg(v_inst_391_);
return v___x_392_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addLeftCancelSemigroup___redArg(lean_object* v_inst_393_){
_start:
{
lean_object* v___f_394_; lean_object* v___x_395_; 
v___f_394_ = lean_alloc_closure((void*)(lp_mathlib_Pi_commMagma___redArg___lam__0), 4, 1);
lean_closure_set(v___f_394_, 0, v_inst_393_);
v___x_395_ = lp_mathlib_Pi_addSemigroup___redArg(v___f_394_);
return v___x_395_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addLeftCancelSemigroup(lean_object* v_I_396_, lean_object* v_f_397_, lean_object* v_inst_398_){
_start:
{
lean_object* v___x_399_; 
v___x_399_ = lp_mathlib_Pi_addLeftCancelSemigroup___redArg(v_inst_398_);
return v___x_399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_rightCancelSemigroup___redArg(lean_object* v_inst_400_){
_start:
{
lean_object* v___f_401_; lean_object* v___x_402_; 
v___f_401_ = lean_alloc_closure((void*)(lp_mathlib_Pi_commMagma___redArg___lam__0), 4, 1);
lean_closure_set(v___f_401_, 0, v_inst_400_);
v___x_402_ = lp_mathlib_Pi_semigroup___redArg(v___f_401_);
return v___x_402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_rightCancelSemigroup(lean_object* v_I_403_, lean_object* v_f_404_, lean_object* v_inst_405_){
_start:
{
lean_object* v___x_406_; 
v___x_406_ = lp_mathlib_Pi_rightCancelSemigroup___redArg(v_inst_405_);
return v___x_406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addRightCancelSemigroup___redArg(lean_object* v_inst_407_){
_start:
{
lean_object* v___f_408_; lean_object* v___x_409_; 
v___f_408_ = lean_alloc_closure((void*)(lp_mathlib_Pi_commMagma___redArg___lam__0), 4, 1);
lean_closure_set(v___f_408_, 0, v_inst_407_);
v___x_409_ = lp_mathlib_Pi_addSemigroup___redArg(v___f_408_);
return v___x_409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addRightCancelSemigroup(lean_object* v_I_410_, lean_object* v_f_411_, lean_object* v_inst_412_){
_start:
{
lean_object* v___x_413_; 
v___x_413_ = lp_mathlib_Pi_addRightCancelSemigroup___redArg(v_inst_412_);
return v___x_413_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_leftCancelMonoid___redArg(lean_object* v_inst_414_){
_start:
{
lean_object* v___f_415_; lean_object* v___f_416_; lean_object* v___x_417_; lean_object* v___f_418_; lean_object* v___f_419_; lean_object* v___f_420_; lean_object* v___f_421_; lean_object* v___x_422_; 
lean_inc_ref(v_inst_414_);
v___f_415_ = lean_alloc_closure((void*)(lp_mathlib_Pi_monoid___redArg___lam__0), 4, 1);
lean_closure_set(v___f_415_, 0, v_inst_414_);
v___f_416_ = lean_alloc_closure((void*)(lp_mathlib_Pi_commMonoid___redArg___lam__0), 2, 1);
lean_closure_set(v___f_416_, 0, v_inst_414_);
v___x_417_ = lp_mathlib_Pi_leftCancelSemigroup___redArg(v___f_415_);
lean_inc_ref(v___f_416_);
v___f_418_ = lean_alloc_closure((void*)(lp_mathlib_Pi_monoid___redArg___lam__1), 2, 1);
lean_closure_set(v___f_418_, 0, v___f_416_);
v___f_419_ = lean_alloc_closure((void*)(lp_mathlib_Pi_mulOneClass___redArg___lam__0), 2, 1);
lean_closure_set(v___f_419_, 0, v___f_418_);
v___f_420_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instOne___redArg___lam__0), 2, 1);
lean_closure_set(v___f_420_, 0, v___f_419_);
v___f_421_ = lean_alloc_closure((void*)(lp_mathlib_Pi_monoid___redArg___lam__2), 4, 1);
lean_closure_set(v___f_421_, 0, v___f_416_);
v___x_422_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_422_, 0, v___f_420_);
lean_ctor_set(v___x_422_, 1, v___x_417_);
lean_ctor_set(v___x_422_, 2, v___f_421_);
return v___x_422_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_leftCancelMonoid(lean_object* v_I_423_, lean_object* v_f_424_, lean_object* v_inst_425_){
_start:
{
lean_object* v___x_426_; 
v___x_426_ = lp_mathlib_Pi_leftCancelMonoid___redArg(v_inst_425_);
return v___x_426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addLeftCancelMonoid___redArg(lean_object* v_inst_427_){
_start:
{
lean_object* v___f_428_; lean_object* v___f_429_; lean_object* v___x_430_; lean_object* v___f_431_; lean_object* v___f_432_; lean_object* v___f_433_; lean_object* v___f_434_; lean_object* v___x_435_; 
lean_inc_ref(v_inst_427_);
v___f_428_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addMonoid___redArg___lam__0), 4, 1);
lean_closure_set(v___f_428_, 0, v_inst_427_);
v___f_429_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addCommMonoid___redArg___lam__0), 2, 1);
lean_closure_set(v___f_429_, 0, v_inst_427_);
v___x_430_ = lp_mathlib_Pi_addLeftCancelSemigroup___redArg(v___f_428_);
lean_inc_ref(v___f_429_);
v___f_431_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addMonoid___redArg___lam__1), 2, 1);
lean_closure_set(v___f_431_, 0, v___f_429_);
v___f_432_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addZeroClass___redArg___lam__0), 2, 1);
lean_closure_set(v___f_432_, 0, v___f_431_);
v___f_433_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instOne___redArg___lam__0), 2, 1);
lean_closure_set(v___f_433_, 0, v___f_432_);
v___f_434_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addMonoid___redArg___lam__2), 4, 1);
lean_closure_set(v___f_434_, 0, v___f_429_);
v___x_435_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_435_, 0, v___f_433_);
lean_ctor_set(v___x_435_, 1, v___x_430_);
lean_ctor_set(v___x_435_, 2, v___f_434_);
return v___x_435_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addLeftCancelMonoid(lean_object* v_I_436_, lean_object* v_f_437_, lean_object* v_inst_438_){
_start:
{
lean_object* v___x_439_; 
v___x_439_ = lp_mathlib_Pi_addLeftCancelMonoid___redArg(v_inst_438_);
return v___x_439_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_rightCancelMonoid___redArg(lean_object* v_inst_440_){
_start:
{
lean_object* v___f_441_; lean_object* v___f_442_; lean_object* v___x_443_; lean_object* v___f_444_; lean_object* v___f_445_; lean_object* v___f_446_; lean_object* v___f_447_; lean_object* v___x_448_; 
lean_inc_ref(v_inst_440_);
v___f_441_ = lean_alloc_closure((void*)(lp_mathlib_Pi_monoid___redArg___lam__0), 4, 1);
lean_closure_set(v___f_441_, 0, v_inst_440_);
v___f_442_ = lean_alloc_closure((void*)(lp_mathlib_Pi_commMonoid___redArg___lam__0), 2, 1);
lean_closure_set(v___f_442_, 0, v_inst_440_);
v___x_443_ = lp_mathlib_Pi_rightCancelSemigroup___redArg(v___f_441_);
lean_inc_ref(v___f_442_);
v___f_444_ = lean_alloc_closure((void*)(lp_mathlib_Pi_monoid___redArg___lam__1), 2, 1);
lean_closure_set(v___f_444_, 0, v___f_442_);
v___f_445_ = lean_alloc_closure((void*)(lp_mathlib_Pi_mulOneClass___redArg___lam__0), 2, 1);
lean_closure_set(v___f_445_, 0, v___f_444_);
v___f_446_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instOne___redArg___lam__0), 2, 1);
lean_closure_set(v___f_446_, 0, v___f_445_);
v___f_447_ = lean_alloc_closure((void*)(lp_mathlib_Pi_monoid___redArg___lam__2), 4, 1);
lean_closure_set(v___f_447_, 0, v___f_442_);
v___x_448_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_448_, 0, v___f_446_);
lean_ctor_set(v___x_448_, 1, v___x_443_);
lean_ctor_set(v___x_448_, 2, v___f_447_);
return v___x_448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_rightCancelMonoid(lean_object* v_I_449_, lean_object* v_f_450_, lean_object* v_inst_451_){
_start:
{
lean_object* v___x_452_; 
v___x_452_ = lp_mathlib_Pi_rightCancelMonoid___redArg(v_inst_451_);
return v___x_452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addRightCancelMonoid___redArg(lean_object* v_inst_453_){
_start:
{
lean_object* v___f_454_; lean_object* v___f_455_; lean_object* v___x_456_; lean_object* v___f_457_; lean_object* v___f_458_; lean_object* v___f_459_; lean_object* v___f_460_; lean_object* v___x_461_; 
lean_inc_ref(v_inst_453_);
v___f_454_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addMonoid___redArg___lam__0), 4, 1);
lean_closure_set(v___f_454_, 0, v_inst_453_);
v___f_455_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addCommMonoid___redArg___lam__0), 2, 1);
lean_closure_set(v___f_455_, 0, v_inst_453_);
v___x_456_ = lp_mathlib_Pi_addRightCancelSemigroup___redArg(v___f_454_);
lean_inc_ref(v___f_455_);
v___f_457_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addMonoid___redArg___lam__1), 2, 1);
lean_closure_set(v___f_457_, 0, v___f_455_);
v___f_458_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addZeroClass___redArg___lam__0), 2, 1);
lean_closure_set(v___f_458_, 0, v___f_457_);
v___f_459_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instOne___redArg___lam__0), 2, 1);
lean_closure_set(v___f_459_, 0, v___f_458_);
v___f_460_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addMonoid___redArg___lam__2), 4, 1);
lean_closure_set(v___f_460_, 0, v___f_455_);
v___x_461_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_461_, 0, v___f_459_);
lean_ctor_set(v___x_461_, 1, v___x_456_);
lean_ctor_set(v___x_461_, 2, v___f_460_);
return v___x_461_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addRightCancelMonoid(lean_object* v_I_462_, lean_object* v_f_463_, lean_object* v_inst_464_){
_start:
{
lean_object* v___x_465_; 
v___x_465_ = lp_mathlib_Pi_addRightCancelMonoid___redArg(v_inst_464_);
return v___x_465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_cancelMonoid___redArg(lean_object* v_inst_466_){
_start:
{
lean_object* v___f_467_; lean_object* v___x_468_; 
v___f_467_ = lean_alloc_closure((void*)(lp_mathlib_Pi_commMonoid___redArg___lam__0), 2, 1);
lean_closure_set(v___f_467_, 0, v_inst_466_);
v___x_468_ = lp_mathlib_Pi_leftCancelMonoid___redArg(v___f_467_);
return v___x_468_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_cancelMonoid(lean_object* v_I_469_, lean_object* v_f_470_, lean_object* v_inst_471_){
_start:
{
lean_object* v___x_472_; 
v___x_472_ = lp_mathlib_Pi_cancelMonoid___redArg(v_inst_471_);
return v___x_472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addCancelMonoid___redArg(lean_object* v_inst_473_){
_start:
{
lean_object* v___f_474_; lean_object* v___x_475_; 
v___f_474_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addCommMonoid___redArg___lam__0), 2, 1);
lean_closure_set(v___f_474_, 0, v_inst_473_);
v___x_475_ = lp_mathlib_Pi_addLeftCancelMonoid___redArg(v___f_474_);
return v___x_475_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addCancelMonoid(lean_object* v_I_476_, lean_object* v_f_477_, lean_object* v_inst_478_){
_start:
{
lean_object* v___x_479_; 
v___x_479_ = lp_mathlib_Pi_addCancelMonoid___redArg(v_inst_478_);
return v___x_479_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_cancelCommMonoid___redArg(lean_object* v_inst_480_){
_start:
{
lean_object* v___f_481_; lean_object* v___x_482_; 
v___f_481_ = lean_alloc_closure((void*)(lp_mathlib_Pi_commMonoid___redArg___lam__0), 2, 1);
lean_closure_set(v___f_481_, 0, v_inst_480_);
v___x_482_ = lp_mathlib_Pi_leftCancelMonoid___redArg(v___f_481_);
return v___x_482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_cancelCommMonoid(lean_object* v_I_483_, lean_object* v_f_484_, lean_object* v_inst_485_){
_start:
{
lean_object* v___x_486_; 
v___x_486_ = lp_mathlib_Pi_cancelCommMonoid___redArg(v_inst_485_);
return v___x_486_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addCancelCommMonoid___redArg(lean_object* v_inst_487_){
_start:
{
lean_object* v___f_488_; lean_object* v___x_489_; 
v___f_488_ = lean_alloc_closure((void*)(lp_mathlib_Pi_addCommMonoid___redArg___lam__0), 2, 1);
lean_closure_set(v___f_488_, 0, v_inst_487_);
v___x_489_ = lp_mathlib_Pi_addLeftCancelMonoid___redArg(v___f_488_);
return v___x_489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_addCancelCommMonoid(lean_object* v_I_490_, lean_object* v_f_491_, lean_object* v_inst_492_){
_start:
{
lean_object* v___x_493_; 
v___x_493_ = lp_mathlib_Pi_addCancelCommMonoid___redArg(v_inst_492_);
return v___x_493_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSurjectiveOne___redArg(lean_object* v_inst_494_){
_start:
{
lean_inc(v_inst_494_);
return v_inst_494_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSurjectiveOne___redArg___boxed(lean_object* v_inst_495_){
_start:
{
lean_object* v_res_496_; 
v_res_496_ = lp_mathlib_uniqueOfSurjectiveOne___redArg(v_inst_495_);
lean_dec(v_inst_495_);
return v_res_496_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSurjectiveOne(lean_object* v_00_u03b1_497_, lean_object* v_00_u03b2_498_, lean_object* v_inst_499_, lean_object* v_h_500_){
_start:
{
lean_inc(v_inst_499_);
return v_inst_499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSurjectiveOne___boxed(lean_object* v_00_u03b1_501_, lean_object* v_00_u03b2_502_, lean_object* v_inst_503_, lean_object* v_h_504_){
_start:
{
lean_object* v_res_505_; 
v_res_505_ = lp_mathlib_uniqueOfSurjectiveOne(v_00_u03b1_501_, v_00_u03b2_502_, v_inst_503_, v_h_504_);
lean_dec(v_inst_503_);
return v_res_505_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSurjectiveZero___redArg(lean_object* v_inst_506_){
_start:
{
lean_inc(v_inst_506_);
return v_inst_506_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSurjectiveZero___redArg___boxed(lean_object* v_inst_507_){
_start:
{
lean_object* v_res_508_; 
v_res_508_ = lp_mathlib_uniqueOfSurjectiveZero___redArg(v_inst_507_);
lean_dec(v_inst_507_);
return v_res_508_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSurjectiveZero(lean_object* v_00_u03b1_509_, lean_object* v_00_u03b2_510_, lean_object* v_inst_511_, lean_object* v_h_512_){
_start:
{
lean_inc(v_inst_511_);
return v_inst_511_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_uniqueOfSurjectiveZero___boxed(lean_object* v_00_u03b1_513_, lean_object* v_00_u03b2_514_, lean_object* v_inst_515_, lean_object* v_h_516_){
_start:
{
lean_object* v_res_517_; 
v_res_517_ = lp_mathlib_uniqueOfSurjectiveZero(v_00_u03b1_513_, v_00_u03b2_514_, v_inst_515_, v_h_516_);
lean_dec(v_inst_515_);
return v_res_517_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Notation_Pi_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Unique(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Sum_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Spread(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Notation_Pi_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Unique(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Sum_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Spread(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Notation_Pi_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Basic_Unique(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Sum_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Spread(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Notation_Pi_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Unique(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Sum_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Spread(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
