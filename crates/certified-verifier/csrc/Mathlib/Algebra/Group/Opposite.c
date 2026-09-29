// Lean compiler output
// Module: Mathlib.Algebra.Group.Opposite
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Commute.Defs public import Mathlib.Algebra.Group.InjSurj public import Mathlib.Algebra.Group.Torsion public import Mathlib.Algebra.Opposites public import Mathlib.Tactic.Conv
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
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_MulOpposite_instAdd___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulOpposite_instSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_addMonoid___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulOpposite_instNeg___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_ZSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_subNegMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_MulOpposite_instMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DivInvMonoid_div_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddOpposite_instMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubNegMonoid_sub_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
lean_object* lp_mathlib_AddOpposite_instInv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddLeftCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddLeftCancelSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddRightCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddRightCancelSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddCommMagma___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddCommMagma(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddCommSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddCommSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddZeroClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSubNegMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSubNegMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddGroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddCommGroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instLeftCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instLeftCancelSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddLeftCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddLeftCancelSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instRightCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instRightCancelSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddRightCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddRightCancelSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instCommSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instCommSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddCommSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddCommSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulOneClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddZeroClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instLeftCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instLeftCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddLeftCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddLeftCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instRightCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instRightCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddRightCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddRightCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddCancelMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddCancelMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instCancelCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instCancelCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddCancelCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddCancelCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDivInvMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDivInvMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDivInvMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSubNegMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSubNegMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSubNegMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDivisionMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDivisionMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSubtractionMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSubtractionMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDivisionCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDivisionCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSubtractionCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSubtractionCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instGroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddGroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instCommGroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddCommGroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instLeftCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instLeftCancelSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instRightCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instRightCancelSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instCommSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instCommSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMulOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMulOneClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_pow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_pow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_pow(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDivInvMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDivInvMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDivInvMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instGroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instCommGroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddSemigroup___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v___f_2_; 
v___f_2_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instAdd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2_, 0, v_inst_1_);
return v___f_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddSemigroup(lean_object* v_00_u03b1_3_, lean_object* v_inst_4_){
_start:
{
lean_object* v___f_5_; 
v___f_5_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instAdd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_5_, 0, v_inst_4_);
return v___f_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddLeftCancelSemigroup___redArg(lean_object* v_inst_6_){
_start:
{
lean_object* v___f_7_; 
v___f_7_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instAdd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_7_, 0, v_inst_6_);
return v___f_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddLeftCancelSemigroup(lean_object* v_00_u03b1_8_, lean_object* v_inst_9_){
_start:
{
lean_object* v___f_10_; 
v___f_10_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instAdd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_10_, 0, v_inst_9_);
return v___f_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddRightCancelSemigroup___redArg(lean_object* v_inst_11_){
_start:
{
lean_object* v___f_12_; 
v___f_12_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instAdd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_12_, 0, v_inst_11_);
return v___f_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddRightCancelSemigroup(lean_object* v_00_u03b1_13_, lean_object* v_inst_14_){
_start:
{
lean_object* v___f_15_; 
v___f_15_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instAdd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_15_, 0, v_inst_14_);
return v___f_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddCommMagma___redArg(lean_object* v_inst_16_){
_start:
{
lean_object* v___f_17_; 
v___f_17_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instAdd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_17_, 0, v_inst_16_);
return v___f_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddCommMagma(lean_object* v_00_u03b1_18_, lean_object* v_inst_19_){
_start:
{
lean_object* v___f_20_; 
v___f_20_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instAdd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_20_, 0, v_inst_19_);
return v___f_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddCommSemigroup___redArg(lean_object* v_inst_21_){
_start:
{
lean_object* v___f_22_; 
v___f_22_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instAdd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_22_, 0, v_inst_21_);
return v___f_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddCommSemigroup(lean_object* v_00_u03b1_23_, lean_object* v_inst_24_){
_start:
{
lean_object* v___f_25_; 
v___f_25_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instAdd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_25_, 0, v_inst_24_);
return v___f_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddZeroClass___redArg(lean_object* v_inst_26_){
_start:
{
lean_object* v___x_27_; lean_object* v_toZero_28_; lean_object* v_toAdd_29_; lean_object* v___x_31_; uint8_t v_isShared_32_; uint8_t v_isSharedCheck_37_; 
v___x_27_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_26_);
v_toZero_28_ = lean_ctor_get(v___x_27_, 0);
v_toAdd_29_ = lean_ctor_get(v___x_27_, 1);
v_isSharedCheck_37_ = !lean_is_exclusive(v___x_27_);
if (v_isSharedCheck_37_ == 0)
{
v___x_31_ = v___x_27_;
v_isShared_32_ = v_isSharedCheck_37_;
goto v_resetjp_30_;
}
else
{
lean_inc(v_toAdd_29_);
lean_inc(v_toZero_28_);
lean_dec(v___x_27_);
v___x_31_ = lean_box(0);
v_isShared_32_ = v_isSharedCheck_37_;
goto v_resetjp_30_;
}
v_resetjp_30_:
{
lean_object* v___f_33_; lean_object* v___x_35_; 
v___f_33_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instAdd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_33_, 0, v_toAdd_29_);
if (v_isShared_32_ == 0)
{
lean_ctor_set(v___x_31_, 1, v___f_33_);
v___x_35_ = v___x_31_;
goto v_reusejp_34_;
}
else
{
lean_object* v_reuseFailAlloc_36_; 
v_reuseFailAlloc_36_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_36_, 0, v_toZero_28_);
lean_ctor_set(v_reuseFailAlloc_36_, 1, v___f_33_);
v___x_35_ = v_reuseFailAlloc_36_;
goto v_reusejp_34_;
}
v_reusejp_34_:
{
return v___x_35_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddZeroClass(lean_object* v_00_u03b1_38_, lean_object* v_inst_39_){
_start:
{
lean_object* v___x_40_; 
v___x_40_ = lp_mathlib_MulOpposite_instAddZeroClass___redArg(v_inst_39_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddMonoid___redArg(lean_object* v_inst_41_){
_start:
{
lean_object* v_toAdd_42_; lean_object* v_toNSMul_43_; lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v_toZero_46_; lean_object* v___f_47_; lean_object* v___f_48_; lean_object* v___f_49_; lean_object* v___x_50_; 
v_toAdd_42_ = lean_ctor_get(v_inst_41_, 1);
lean_inc(v_toAdd_42_);
v_toNSMul_43_ = lean_ctor_get(v_inst_41_, 2);
lean_inc(v_toNSMul_43_);
v___x_44_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_41_);
lean_dec_ref(v_inst_41_);
v___x_45_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_44_);
v_toZero_46_ = lean_ctor_get(v___x_45_, 0);
lean_inc(v_toZero_46_);
lean_dec_ref(v___x_45_);
v___f_47_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instAdd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_47_, 0, v_toAdd_42_);
v___f_48_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_48_, 0, v_toNSMul_43_);
v___f_49_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_49_, 0, v___f_48_);
v___x_50_ = lp_mathlib_Function_Injective_addMonoid___redArg(v___f_47_, v_toZero_46_, v___f_49_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddMonoid(lean_object* v_00_u03b1_51_, lean_object* v_inst_52_){
_start:
{
lean_object* v___x_53_; 
v___x_53_ = lp_mathlib_MulOpposite_instAddMonoid___redArg(v_inst_52_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddCommMonoid___redArg(lean_object* v_inst_54_){
_start:
{
lean_object* v_toAdd_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v_toZero_58_; lean_object* v___x_59_; lean_object* v_toNSMul_60_; lean_object* v___f_61_; lean_object* v___f_62_; lean_object* v___x_63_; 
v_toAdd_55_ = lean_ctor_get(v_inst_54_, 1);
lean_inc(v_toAdd_55_);
v___x_56_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_54_);
v___x_57_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_56_);
v_toZero_58_ = lean_ctor_get(v___x_57_, 0);
lean_inc(v_toZero_58_);
lean_dec_ref(v___x_57_);
v___x_59_ = lp_mathlib_MulOpposite_instAddMonoid___redArg(v_inst_54_);
v_toNSMul_60_ = lean_ctor_get(v___x_59_, 2);
lean_inc(v_toNSMul_60_);
lean_dec_ref(v___x_59_);
v___f_61_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instAdd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_61_, 0, v_toAdd_55_);
v___f_62_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_62_, 0, v_toNSMul_60_);
v___x_63_ = lp_mathlib_Function_Injective_addMonoid___redArg(v___f_61_, v_toZero_58_, v___f_62_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddCommMonoid(lean_object* v_00_u03b1_64_, lean_object* v_inst_65_){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = lp_mathlib_MulOpposite_instAddCommMonoid___redArg(v_inst_65_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSubNegMonoid___redArg(lean_object* v_inst_67_){
_start:
{
lean_object* v_toAddMonoid_68_; lean_object* v_toNeg_69_; lean_object* v_toSub_70_; lean_object* v_toZSMul_71_; lean_object* v_toAdd_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v_toZero_75_; lean_object* v___x_76_; lean_object* v_toNSMul_77_; lean_object* v___f_78_; lean_object* v___f_79_; lean_object* v___f_80_; lean_object* v___f_81_; lean_object* v___f_82_; lean_object* v___f_83_; lean_object* v___x_84_; 
v_toAddMonoid_68_ = lean_ctor_get(v_inst_67_, 0);
lean_inc_ref(v_toAddMonoid_68_);
v_toNeg_69_ = lean_ctor_get(v_inst_67_, 1);
lean_inc(v_toNeg_69_);
v_toSub_70_ = lean_ctor_get(v_inst_67_, 2);
lean_inc(v_toSub_70_);
v_toZSMul_71_ = lean_ctor_get(v_inst_67_, 3);
lean_inc(v_toZSMul_71_);
lean_dec_ref(v_inst_67_);
v_toAdd_72_ = lean_ctor_get(v_toAddMonoid_68_, 1);
lean_inc(v_toAdd_72_);
v___x_73_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_68_);
v___x_74_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_73_);
v_toZero_75_ = lean_ctor_get(v___x_74_, 0);
lean_inc(v_toZero_75_);
lean_dec_ref(v___x_74_);
v___x_76_ = lp_mathlib_MulOpposite_instAddMonoid___redArg(v_toAddMonoid_68_);
v_toNSMul_77_ = lean_ctor_get(v___x_76_, 2);
lean_inc(v_toNSMul_77_);
lean_dec_ref(v___x_76_);
v___f_78_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instAdd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_78_, 0, v_toAdd_72_);
v___f_79_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_79_, 0, v_toNSMul_77_);
v___f_80_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNeg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_80_, 0, v_toNeg_69_);
v___f_81_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instAdd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_81_, 0, v_toSub_70_);
v___f_82_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_82_, 0, v_toZSMul_71_);
v___f_83_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_83_, 0, v___f_82_);
v___x_84_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v___f_78_, v_toZero_75_, v___f_79_, v___f_80_, v___f_81_, v___f_83_);
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSubNegMonoid(lean_object* v_00_u03b1_85_, lean_object* v_inst_86_){
_start:
{
lean_object* v___x_87_; 
v___x_87_ = lp_mathlib_MulOpposite_instSubNegMonoid___redArg(v_inst_86_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddGroup___redArg(lean_object* v_inst_88_){
_start:
{
lean_object* v_toAddMonoid_89_; lean_object* v_toSub_90_; lean_object* v_toAdd_91_; lean_object* v___x_92_; lean_object* v_toZero_93_; lean_object* v_toNeg_94_; lean_object* v___x_95_; lean_object* v_toNSMul_96_; lean_object* v___x_97_; lean_object* v_toZSMul_98_; lean_object* v___f_99_; lean_object* v___f_100_; lean_object* v___f_101_; lean_object* v___f_102_; lean_object* v___f_103_; lean_object* v___x_104_; 
v_toAddMonoid_89_ = lean_ctor_get(v_inst_88_, 0);
v_toSub_90_ = lean_ctor_get(v_inst_88_, 2);
lean_inc(v_toSub_90_);
v_toAdd_91_ = lean_ctor_get(v_toAddMonoid_89_, 1);
lean_inc(v_toAdd_91_);
v___x_92_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_88_);
v_toZero_93_ = lean_ctor_get(v___x_92_, 0);
lean_inc(v_toZero_93_);
v_toNeg_94_ = lean_ctor_get(v___x_92_, 1);
lean_inc(v_toNeg_94_);
lean_dec_ref(v___x_92_);
lean_inc_ref(v_toAddMonoid_89_);
v___x_95_ = lp_mathlib_MulOpposite_instAddMonoid___redArg(v_toAddMonoid_89_);
v_toNSMul_96_ = lean_ctor_get(v___x_95_, 2);
lean_inc(v_toNSMul_96_);
lean_dec_ref(v___x_95_);
v___x_97_ = lp_mathlib_MulOpposite_instSubNegMonoid___redArg(v_inst_88_);
v_toZSMul_98_ = lean_ctor_get(v___x_97_, 3);
lean_inc(v_toZSMul_98_);
lean_dec_ref(v___x_97_);
v___f_99_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instAdd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_99_, 0, v_toAdd_91_);
v___f_100_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_100_, 0, v_toNSMul_96_);
v___f_101_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNeg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_101_, 0, v_toNeg_94_);
v___f_102_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instAdd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_102_, 0, v_toSub_90_);
v___f_103_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_103_, 0, v_toZSMul_98_);
v___x_104_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v___f_99_, v_toZero_93_, v___f_100_, v___f_101_, v___f_102_, v___f_103_);
return v___x_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddGroup(lean_object* v_00_u03b1_105_, lean_object* v_inst_106_){
_start:
{
lean_object* v___x_107_; 
v___x_107_ = lp_mathlib_MulOpposite_instAddGroup___redArg(v_inst_106_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddCommGroup___redArg(lean_object* v_inst_108_){
_start:
{
lean_object* v_toAddMonoid_109_; lean_object* v_toSub_110_; lean_object* v_toAdd_111_; lean_object* v___x_112_; lean_object* v_toZero_113_; lean_object* v_toNeg_114_; lean_object* v___x_115_; lean_object* v_toNSMul_116_; lean_object* v___x_117_; lean_object* v_toZSMul_118_; lean_object* v___f_119_; lean_object* v___f_120_; lean_object* v___f_121_; lean_object* v___f_122_; lean_object* v___f_123_; lean_object* v___x_124_; 
v_toAddMonoid_109_ = lean_ctor_get(v_inst_108_, 0);
v_toSub_110_ = lean_ctor_get(v_inst_108_, 2);
lean_inc(v_toSub_110_);
v_toAdd_111_ = lean_ctor_get(v_toAddMonoid_109_, 1);
lean_inc(v_toAdd_111_);
v___x_112_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_108_);
v_toZero_113_ = lean_ctor_get(v___x_112_, 0);
lean_inc(v_toZero_113_);
v_toNeg_114_ = lean_ctor_get(v___x_112_, 1);
lean_inc(v_toNeg_114_);
lean_dec_ref(v___x_112_);
lean_inc_ref(v_toAddMonoid_109_);
v___x_115_ = lp_mathlib_MulOpposite_instAddMonoid___redArg(v_toAddMonoid_109_);
v_toNSMul_116_ = lean_ctor_get(v___x_115_, 2);
lean_inc(v_toNSMul_116_);
lean_dec_ref(v___x_115_);
v___x_117_ = lp_mathlib_MulOpposite_instSubNegMonoid___redArg(v_inst_108_);
v_toZSMul_118_ = lean_ctor_get(v___x_117_, 3);
lean_inc(v_toZSMul_118_);
lean_dec_ref(v___x_117_);
v___f_119_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instAdd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_119_, 0, v_toAdd_111_);
v___f_120_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_120_, 0, v_toNSMul_116_);
v___f_121_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNeg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_121_, 0, v_toNeg_114_);
v___f_122_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instAdd___redArg___lam__0), 3, 1);
lean_closure_set(v___f_122_, 0, v_toSub_110_);
v___f_123_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_123_, 0, v_toZSMul_118_);
v___x_124_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v___f_119_, v_toZero_113_, v___f_120_, v___f_121_, v___f_122_, v___f_123_);
return v___x_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAddCommGroup(lean_object* v_00_u03b1_125_, lean_object* v_inst_126_){
_start:
{
lean_object* v___x_127_; 
v___x_127_ = lp_mathlib_MulOpposite_instAddCommGroup___redArg(v_inst_126_);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSemigroup___redArg(lean_object* v_inst_128_){
_start:
{
lean_object* v___f_129_; 
v___f_129_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_129_, 0, v_inst_128_);
return v___f_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instSemigroup(lean_object* v_00_u03b1_130_, lean_object* v_inst_131_){
_start:
{
lean_object* v___f_132_; 
v___f_132_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_132_, 0, v_inst_131_);
return v___f_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddSemigroup___redArg(lean_object* v_inst_133_){
_start:
{
lean_object* v___f_134_; 
v___f_134_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_134_, 0, v_inst_133_);
return v___f_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddSemigroup(lean_object* v_00_u03b1_135_, lean_object* v_inst_136_){
_start:
{
lean_object* v___f_137_; 
v___f_137_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_137_, 0, v_inst_136_);
return v___f_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instLeftCancelSemigroup___redArg(lean_object* v_inst_138_){
_start:
{
lean_object* v___f_139_; 
v___f_139_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_139_, 0, v_inst_138_);
return v___f_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instLeftCancelSemigroup(lean_object* v_00_u03b1_140_, lean_object* v_inst_141_){
_start:
{
lean_object* v___f_142_; 
v___f_142_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_142_, 0, v_inst_141_);
return v___f_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddLeftCancelSemigroup___redArg(lean_object* v_inst_143_){
_start:
{
lean_object* v___f_144_; 
v___f_144_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_144_, 0, v_inst_143_);
return v___f_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddLeftCancelSemigroup(lean_object* v_00_u03b1_145_, lean_object* v_inst_146_){
_start:
{
lean_object* v___f_147_; 
v___f_147_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_147_, 0, v_inst_146_);
return v___f_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instRightCancelSemigroup___redArg(lean_object* v_inst_148_){
_start:
{
lean_object* v___f_149_; 
v___f_149_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_149_, 0, v_inst_148_);
return v___f_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instRightCancelSemigroup(lean_object* v_00_u03b1_150_, lean_object* v_inst_151_){
_start:
{
lean_object* v___f_152_; 
v___f_152_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_152_, 0, v_inst_151_);
return v___f_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddRightCancelSemigroup___redArg(lean_object* v_inst_153_){
_start:
{
lean_object* v___f_154_; 
v___f_154_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_154_, 0, v_inst_153_);
return v___f_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddRightCancelSemigroup(lean_object* v_00_u03b1_155_, lean_object* v_inst_156_){
_start:
{
lean_object* v___f_157_; 
v___f_157_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_157_, 0, v_inst_156_);
return v___f_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instCommSemigroup___redArg(lean_object* v_inst_158_){
_start:
{
lean_object* v___f_159_; 
v___f_159_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_159_, 0, v_inst_158_);
return v___f_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instCommSemigroup(lean_object* v_00_u03b1_160_, lean_object* v_inst_161_){
_start:
{
lean_object* v___f_162_; 
v___f_162_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_162_, 0, v_inst_161_);
return v___f_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddCommSemigroup___redArg(lean_object* v_inst_163_){
_start:
{
lean_object* v___f_164_; 
v___f_164_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_164_, 0, v_inst_163_);
return v___f_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddCommSemigroup(lean_object* v_00_u03b1_165_, lean_object* v_inst_166_){
_start:
{
lean_object* v___f_167_; 
v___f_167_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_167_, 0, v_inst_166_);
return v___f_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulOne___redArg(lean_object* v_inst_168_){
_start:
{
lean_object* v_toOne_169_; lean_object* v_toMul_170_; lean_object* v___x_172_; uint8_t v_isShared_173_; uint8_t v_isSharedCheck_178_; 
v_toOne_169_ = lean_ctor_get(v_inst_168_, 0);
v_toMul_170_ = lean_ctor_get(v_inst_168_, 1);
v_isSharedCheck_178_ = !lean_is_exclusive(v_inst_168_);
if (v_isSharedCheck_178_ == 0)
{
v___x_172_ = v_inst_168_;
v_isShared_173_ = v_isSharedCheck_178_;
goto v_resetjp_171_;
}
else
{
lean_inc(v_toMul_170_);
lean_inc(v_toOne_169_);
lean_dec(v_inst_168_);
v___x_172_ = lean_box(0);
v_isShared_173_ = v_isSharedCheck_178_;
goto v_resetjp_171_;
}
v_resetjp_171_:
{
lean_object* v___f_174_; lean_object* v___x_176_; 
v___f_174_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_174_, 0, v_toMul_170_);
if (v_isShared_173_ == 0)
{
lean_ctor_set(v___x_172_, 1, v___f_174_);
v___x_176_ = v___x_172_;
goto v_reusejp_175_;
}
else
{
lean_object* v_reuseFailAlloc_177_; 
v_reuseFailAlloc_177_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_177_, 0, v_toOne_169_);
lean_ctor_set(v_reuseFailAlloc_177_, 1, v___f_174_);
v___x_176_ = v_reuseFailAlloc_177_;
goto v_reusejp_175_;
}
v_reusejp_175_:
{
return v___x_176_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulOne(lean_object* v_00_u03b1_179_, lean_object* v_inst_180_){
_start:
{
lean_object* v___x_181_; 
v___x_181_ = lp_mathlib_MulOpposite_instMulOne___redArg(v_inst_180_);
return v___x_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddZero___redArg(lean_object* v_inst_182_){
_start:
{
lean_object* v_toZero_183_; lean_object* v_toAdd_184_; lean_object* v___x_186_; uint8_t v_isShared_187_; uint8_t v_isSharedCheck_192_; 
v_toZero_183_ = lean_ctor_get(v_inst_182_, 0);
v_toAdd_184_ = lean_ctor_get(v_inst_182_, 1);
v_isSharedCheck_192_ = !lean_is_exclusive(v_inst_182_);
if (v_isSharedCheck_192_ == 0)
{
v___x_186_ = v_inst_182_;
v_isShared_187_ = v_isSharedCheck_192_;
goto v_resetjp_185_;
}
else
{
lean_inc(v_toAdd_184_);
lean_inc(v_toZero_183_);
lean_dec(v_inst_182_);
v___x_186_ = lean_box(0);
v_isShared_187_ = v_isSharedCheck_192_;
goto v_resetjp_185_;
}
v_resetjp_185_:
{
lean_object* v___f_188_; lean_object* v___x_190_; 
v___f_188_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_188_, 0, v_toAdd_184_);
if (v_isShared_187_ == 0)
{
lean_ctor_set(v___x_186_, 1, v___f_188_);
v___x_190_ = v___x_186_;
goto v_reusejp_189_;
}
else
{
lean_object* v_reuseFailAlloc_191_; 
v_reuseFailAlloc_191_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_191_, 0, v_toZero_183_);
lean_ctor_set(v_reuseFailAlloc_191_, 1, v___f_188_);
v___x_190_ = v_reuseFailAlloc_191_;
goto v_reusejp_189_;
}
v_reusejp_189_:
{
return v___x_190_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddZero(lean_object* v_00_u03b1_193_, lean_object* v_inst_194_){
_start:
{
lean_object* v___x_195_; 
v___x_195_ = lp_mathlib_AddOpposite_instAddZero___redArg(v_inst_194_);
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulOneClass___redArg(lean_object* v_inst_196_){
_start:
{
lean_object* v___x_197_; lean_object* v_toOne_198_; lean_object* v_toMul_199_; lean_object* v___x_201_; uint8_t v_isShared_202_; uint8_t v_isSharedCheck_207_; 
v___x_197_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_196_);
v_toOne_198_ = lean_ctor_get(v___x_197_, 0);
v_toMul_199_ = lean_ctor_get(v___x_197_, 1);
v_isSharedCheck_207_ = !lean_is_exclusive(v___x_197_);
if (v_isSharedCheck_207_ == 0)
{
v___x_201_ = v___x_197_;
v_isShared_202_ = v_isSharedCheck_207_;
goto v_resetjp_200_;
}
else
{
lean_inc(v_toMul_199_);
lean_inc(v_toOne_198_);
lean_dec(v___x_197_);
v___x_201_ = lean_box(0);
v_isShared_202_ = v_isSharedCheck_207_;
goto v_resetjp_200_;
}
v_resetjp_200_:
{
lean_object* v___f_203_; lean_object* v___x_205_; 
v___f_203_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_203_, 0, v_toMul_199_);
if (v_isShared_202_ == 0)
{
lean_ctor_set(v___x_201_, 1, v___f_203_);
v___x_205_ = v___x_201_;
goto v_reusejp_204_;
}
else
{
lean_object* v_reuseFailAlloc_206_; 
v_reuseFailAlloc_206_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_206_, 0, v_toOne_198_);
lean_ctor_set(v_reuseFailAlloc_206_, 1, v___f_203_);
v___x_205_ = v_reuseFailAlloc_206_;
goto v_reusejp_204_;
}
v_reusejp_204_:
{
return v___x_205_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMulOneClass(lean_object* v_00_u03b1_208_, lean_object* v_inst_209_){
_start:
{
lean_object* v___x_210_; 
v___x_210_ = lp_mathlib_MulOpposite_instMulOneClass___redArg(v_inst_209_);
return v___x_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddZeroClass___redArg(lean_object* v_inst_211_){
_start:
{
lean_object* v___x_212_; lean_object* v_toZero_213_; lean_object* v_toAdd_214_; lean_object* v___x_216_; uint8_t v_isShared_217_; uint8_t v_isSharedCheck_222_; 
v___x_212_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_211_);
v_toZero_213_ = lean_ctor_get(v___x_212_, 0);
v_toAdd_214_ = lean_ctor_get(v___x_212_, 1);
v_isSharedCheck_222_ = !lean_is_exclusive(v___x_212_);
if (v_isSharedCheck_222_ == 0)
{
v___x_216_ = v___x_212_;
v_isShared_217_ = v_isSharedCheck_222_;
goto v_resetjp_215_;
}
else
{
lean_inc(v_toAdd_214_);
lean_inc(v_toZero_213_);
lean_dec(v___x_212_);
v___x_216_ = lean_box(0);
v_isShared_217_ = v_isSharedCheck_222_;
goto v_resetjp_215_;
}
v_resetjp_215_:
{
lean_object* v___f_218_; lean_object* v___x_220_; 
v___f_218_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_218_, 0, v_toAdd_214_);
if (v_isShared_217_ == 0)
{
lean_ctor_set(v___x_216_, 1, v___f_218_);
v___x_220_ = v___x_216_;
goto v_reusejp_219_;
}
else
{
lean_object* v_reuseFailAlloc_221_; 
v_reuseFailAlloc_221_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_221_, 0, v_toZero_213_);
lean_ctor_set(v_reuseFailAlloc_221_, 1, v___f_218_);
v___x_220_ = v_reuseFailAlloc_221_;
goto v_reusejp_219_;
}
v_reusejp_219_:
{
return v___x_220_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddZeroClass(lean_object* v_00_u03b1_223_, lean_object* v_inst_224_){
_start:
{
lean_object* v___x_225_; 
v___x_225_ = lp_mathlib_AddOpposite_instAddZeroClass___redArg(v_inst_224_);
return v___x_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMonoid___redArg___lam__0(lean_object* v_toNPow_226_, lean_object* v_n_227_, lean_object* v_a_228_){
_start:
{
lean_object* v___x_229_; 
v___x_229_ = lean_apply_2(v_toNPow_226_, v_n_227_, v_a_228_);
return v___x_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMonoid___redArg(lean_object* v_inst_230_){
_start:
{
lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v_toMul_233_; lean_object* v_toNPow_234_; lean_object* v___x_236_; uint8_t v_isShared_237_; uint8_t v_isSharedCheck_244_; 
v___x_231_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_230_);
v___x_232_ = lp_mathlib_MulOpposite_instMulOneClass___redArg(v___x_231_);
v_toMul_233_ = lean_ctor_get(v_inst_230_, 1);
v_toNPow_234_ = lean_ctor_get(v_inst_230_, 2);
v_isSharedCheck_244_ = !lean_is_exclusive(v_inst_230_);
if (v_isSharedCheck_244_ == 0)
{
lean_object* v_unused_245_; 
v_unused_245_ = lean_ctor_get(v_inst_230_, 0);
lean_dec(v_unused_245_);
v___x_236_ = v_inst_230_;
v_isShared_237_ = v_isSharedCheck_244_;
goto v_resetjp_235_;
}
else
{
lean_inc(v_toNPow_234_);
lean_inc(v_toMul_233_);
lean_dec(v_inst_230_);
v___x_236_ = lean_box(0);
v_isShared_237_ = v_isSharedCheck_244_;
goto v_resetjp_235_;
}
v_resetjp_235_:
{
lean_object* v_toOne_238_; lean_object* v___f_239_; lean_object* v___f_240_; lean_object* v___x_242_; 
v_toOne_238_ = lean_ctor_get(v___x_232_, 0);
lean_inc(v_toOne_238_);
lean_dec_ref(v___x_232_);
v___f_239_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMonoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_239_, 0, v_toNPow_234_);
v___f_240_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_240_, 0, v_toMul_233_);
if (v_isShared_237_ == 0)
{
lean_ctor_set(v___x_236_, 2, v___f_239_);
lean_ctor_set(v___x_236_, 1, v___f_240_);
lean_ctor_set(v___x_236_, 0, v_toOne_238_);
v___x_242_ = v___x_236_;
goto v_reusejp_241_;
}
else
{
lean_object* v_reuseFailAlloc_243_; 
v_reuseFailAlloc_243_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_243_, 0, v_toOne_238_);
lean_ctor_set(v_reuseFailAlloc_243_, 1, v___f_240_);
lean_ctor_set(v_reuseFailAlloc_243_, 2, v___f_239_);
v___x_242_ = v_reuseFailAlloc_243_;
goto v_reusejp_241_;
}
v_reusejp_241_:
{
return v___x_242_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instMonoid(lean_object* v_00_u03b1_246_, lean_object* v_inst_247_){
_start:
{
lean_object* v___x_248_; 
v___x_248_ = lp_mathlib_MulOpposite_instMonoid___redArg(v_inst_247_);
return v___x_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddMonoid___redArg___lam__0(lean_object* v_toNSMul_249_, lean_object* v_n_250_, lean_object* v_a_251_){
_start:
{
lean_object* v___x_252_; 
v___x_252_ = lean_apply_2(v_toNSMul_249_, v_n_250_, v_a_251_);
return v___x_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddMonoid___redArg(lean_object* v_inst_253_){
_start:
{
lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v_toAdd_256_; lean_object* v_toNSMul_257_; lean_object* v___x_259_; uint8_t v_isShared_260_; uint8_t v_isSharedCheck_267_; 
v___x_254_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_253_);
v___x_255_ = lp_mathlib_AddOpposite_instAddZeroClass___redArg(v___x_254_);
v_toAdd_256_ = lean_ctor_get(v_inst_253_, 1);
v_toNSMul_257_ = lean_ctor_get(v_inst_253_, 2);
v_isSharedCheck_267_ = !lean_is_exclusive(v_inst_253_);
if (v_isSharedCheck_267_ == 0)
{
lean_object* v_unused_268_; 
v_unused_268_ = lean_ctor_get(v_inst_253_, 0);
lean_dec(v_unused_268_);
v___x_259_ = v_inst_253_;
v_isShared_260_ = v_isSharedCheck_267_;
goto v_resetjp_258_;
}
else
{
lean_inc(v_toNSMul_257_);
lean_inc(v_toAdd_256_);
lean_dec(v_inst_253_);
v___x_259_ = lean_box(0);
v_isShared_260_ = v_isSharedCheck_267_;
goto v_resetjp_258_;
}
v_resetjp_258_:
{
lean_object* v_toZero_261_; lean_object* v___f_262_; lean_object* v___f_263_; lean_object* v___x_265_; 
v_toZero_261_ = lean_ctor_get(v___x_255_, 0);
lean_inc(v_toZero_261_);
lean_dec_ref(v___x_255_);
v___f_262_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instAddMonoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_262_, 0, v_toNSMul_257_);
v___f_263_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_263_, 0, v_toAdd_256_);
if (v_isShared_260_ == 0)
{
lean_ctor_set(v___x_259_, 2, v___f_262_);
lean_ctor_set(v___x_259_, 1, v___f_263_);
lean_ctor_set(v___x_259_, 0, v_toZero_261_);
v___x_265_ = v___x_259_;
goto v_reusejp_264_;
}
else
{
lean_object* v_reuseFailAlloc_266_; 
v_reuseFailAlloc_266_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_266_, 0, v_toZero_261_);
lean_ctor_set(v_reuseFailAlloc_266_, 1, v___f_263_);
lean_ctor_set(v_reuseFailAlloc_266_, 2, v___f_262_);
v___x_265_ = v_reuseFailAlloc_266_;
goto v_reusejp_264_;
}
v_reusejp_264_:
{
return v___x_265_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddMonoid(lean_object* v_00_u03b1_269_, lean_object* v_inst_270_){
_start:
{
lean_object* v___x_271_; 
v___x_271_ = lp_mathlib_AddOpposite_instAddMonoid___redArg(v_inst_270_);
return v___x_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instLeftCancelMonoid___redArg(lean_object* v_inst_272_){
_start:
{
lean_object* v___x_273_; 
v___x_273_ = lp_mathlib_MulOpposite_instMonoid___redArg(v_inst_272_);
return v___x_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instLeftCancelMonoid(lean_object* v_00_u03b1_274_, lean_object* v_inst_275_){
_start:
{
lean_object* v___x_276_; 
v___x_276_ = lp_mathlib_MulOpposite_instMonoid___redArg(v_inst_275_);
return v___x_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddLeftCancelMonoid___redArg(lean_object* v_inst_277_){
_start:
{
lean_object* v___x_278_; 
v___x_278_ = lp_mathlib_AddOpposite_instAddMonoid___redArg(v_inst_277_);
return v___x_278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddLeftCancelMonoid(lean_object* v_00_u03b1_279_, lean_object* v_inst_280_){
_start:
{
lean_object* v___x_281_; 
v___x_281_ = lp_mathlib_AddOpposite_instAddMonoid___redArg(v_inst_280_);
return v___x_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instRightCancelMonoid___redArg(lean_object* v_inst_282_){
_start:
{
lean_object* v___x_283_; 
v___x_283_ = lp_mathlib_MulOpposite_instMonoid___redArg(v_inst_282_);
return v___x_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instRightCancelMonoid(lean_object* v_00_u03b1_284_, lean_object* v_inst_285_){
_start:
{
lean_object* v___x_286_; 
v___x_286_ = lp_mathlib_MulOpposite_instMonoid___redArg(v_inst_285_);
return v___x_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddRightCancelMonoid___redArg(lean_object* v_inst_287_){
_start:
{
lean_object* v___x_288_; 
v___x_288_ = lp_mathlib_AddOpposite_instAddMonoid___redArg(v_inst_287_);
return v___x_288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddRightCancelMonoid(lean_object* v_00_u03b1_289_, lean_object* v_inst_290_){
_start:
{
lean_object* v___x_291_; 
v___x_291_ = lp_mathlib_AddOpposite_instAddMonoid___redArg(v_inst_290_);
return v___x_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instCancelMonoid___redArg(lean_object* v_inst_292_){
_start:
{
lean_object* v___x_293_; 
v___x_293_ = lp_mathlib_MulOpposite_instMonoid___redArg(v_inst_292_);
return v___x_293_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instCancelMonoid(lean_object* v_00_u03b1_294_, lean_object* v_inst_295_){
_start:
{
lean_object* v___x_296_; 
v___x_296_ = lp_mathlib_MulOpposite_instMonoid___redArg(v_inst_295_);
return v___x_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddCancelMonoid___redArg(lean_object* v_inst_297_){
_start:
{
lean_object* v___x_298_; 
v___x_298_ = lp_mathlib_AddOpposite_instAddMonoid___redArg(v_inst_297_);
return v___x_298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddCancelMonoid(lean_object* v_00_u03b1_299_, lean_object* v_inst_300_){
_start:
{
lean_object* v___x_301_; 
v___x_301_ = lp_mathlib_AddOpposite_instAddMonoid___redArg(v_inst_300_);
return v___x_301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instCommMonoid___redArg(lean_object* v_inst_302_){
_start:
{
lean_object* v___x_303_; 
v___x_303_ = lp_mathlib_MulOpposite_instMonoid___redArg(v_inst_302_);
return v___x_303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instCommMonoid(lean_object* v_00_u03b1_304_, lean_object* v_inst_305_){
_start:
{
lean_object* v___x_306_; 
v___x_306_ = lp_mathlib_MulOpposite_instMonoid___redArg(v_inst_305_);
return v___x_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddCommMonoid___redArg(lean_object* v_inst_307_){
_start:
{
lean_object* v___x_308_; 
v___x_308_ = lp_mathlib_AddOpposite_instAddMonoid___redArg(v_inst_307_);
return v___x_308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddCommMonoid(lean_object* v_00_u03b1_309_, lean_object* v_inst_310_){
_start:
{
lean_object* v___x_311_; 
v___x_311_ = lp_mathlib_AddOpposite_instAddMonoid___redArg(v_inst_310_);
return v___x_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instCancelCommMonoid___redArg(lean_object* v_inst_312_){
_start:
{
lean_object* v___x_313_; 
v___x_313_ = lp_mathlib_MulOpposite_instMonoid___redArg(v_inst_312_);
return v___x_313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instCancelCommMonoid(lean_object* v_00_u03b1_314_, lean_object* v_inst_315_){
_start:
{
lean_object* v___x_316_; 
v___x_316_ = lp_mathlib_MulOpposite_instMonoid___redArg(v_inst_315_);
return v___x_316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddCancelCommMonoid___redArg(lean_object* v_inst_317_){
_start:
{
lean_object* v___x_318_; 
v___x_318_ = lp_mathlib_AddOpposite_instAddMonoid___redArg(v_inst_317_);
return v___x_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddCancelCommMonoid(lean_object* v_00_u03b1_319_, lean_object* v_inst_320_){
_start:
{
lean_object* v___x_321_; 
v___x_321_ = lp_mathlib_AddOpposite_instAddMonoid___redArg(v_inst_320_);
return v___x_321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDivInvMonoid___redArg___lam__0(lean_object* v_toZPow_322_, lean_object* v_n_323_, lean_object* v_a_324_){
_start:
{
lean_object* v___x_325_; 
v___x_325_ = lean_apply_2(v_toZPow_322_, v_n_323_, v_a_324_);
return v___x_325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDivInvMonoid___redArg(lean_object* v_inst_326_){
_start:
{
lean_object* v_toMonoid_327_; lean_object* v_toInv_328_; lean_object* v_toZPow_329_; lean_object* v___x_331_; uint8_t v_isShared_332_; uint8_t v_isSharedCheck_340_; 
v_toMonoid_327_ = lean_ctor_get(v_inst_326_, 0);
v_toInv_328_ = lean_ctor_get(v_inst_326_, 1);
v_toZPow_329_ = lean_ctor_get(v_inst_326_, 3);
v_isSharedCheck_340_ = !lean_is_exclusive(v_inst_326_);
if (v_isSharedCheck_340_ == 0)
{
lean_object* v_unused_341_; 
v_unused_341_ = lean_ctor_get(v_inst_326_, 2);
lean_dec(v_unused_341_);
v___x_331_ = v_inst_326_;
v_isShared_332_ = v_isSharedCheck_340_;
goto v_resetjp_330_;
}
else
{
lean_inc(v_toZPow_329_);
lean_inc(v_toInv_328_);
lean_inc(v_toMonoid_327_);
lean_dec(v_inst_326_);
v___x_331_ = lean_box(0);
v_isShared_332_ = v_isSharedCheck_340_;
goto v_resetjp_330_;
}
v_resetjp_330_:
{
lean_object* v___f_333_; lean_object* v___x_334_; lean_object* v___f_335_; lean_object* v___x_336_; lean_object* v___x_338_; 
v___f_333_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instDivInvMonoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_333_, 0, v_toZPow_329_);
v___x_334_ = lp_mathlib_MulOpposite_instMonoid___redArg(v_toMonoid_327_);
v___f_335_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNeg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_335_, 0, v_toInv_328_);
lean_inc_ref(v___f_335_);
lean_inc_ref(v___x_334_);
v___x_336_ = lean_alloc_closure((void*)(lp_mathlib_DivInvMonoid_div_x27___boxed), 5, 3);
lean_closure_set(v___x_336_, 0, lean_box(0));
lean_closure_set(v___x_336_, 1, v___x_334_);
lean_closure_set(v___x_336_, 2, v___f_335_);
if (v_isShared_332_ == 0)
{
lean_ctor_set(v___x_331_, 3, v___f_333_);
lean_ctor_set(v___x_331_, 2, v___x_336_);
lean_ctor_set(v___x_331_, 1, v___f_335_);
lean_ctor_set(v___x_331_, 0, v___x_334_);
v___x_338_ = v___x_331_;
goto v_reusejp_337_;
}
else
{
lean_object* v_reuseFailAlloc_339_; 
v_reuseFailAlloc_339_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_339_, 0, v___x_334_);
lean_ctor_set(v_reuseFailAlloc_339_, 1, v___f_335_);
lean_ctor_set(v_reuseFailAlloc_339_, 2, v___x_336_);
lean_ctor_set(v_reuseFailAlloc_339_, 3, v___f_333_);
v___x_338_ = v_reuseFailAlloc_339_;
goto v_reusejp_337_;
}
v_reusejp_337_:
{
return v___x_338_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDivInvMonoid(lean_object* v_00_u03b1_342_, lean_object* v_inst_343_){
_start:
{
lean_object* v___x_344_; 
v___x_344_ = lp_mathlib_MulOpposite_instDivInvMonoid___redArg(v_inst_343_);
return v___x_344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSubNegMonoid___redArg___lam__0(lean_object* v_toZSMul_345_, lean_object* v_n_346_, lean_object* v_a_347_){
_start:
{
lean_object* v___x_348_; 
v___x_348_ = lean_apply_2(v_toZSMul_345_, v_n_346_, v_a_347_);
return v___x_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSubNegMonoid___redArg(lean_object* v_inst_349_){
_start:
{
lean_object* v_toAddMonoid_350_; lean_object* v_toNeg_351_; lean_object* v_toZSMul_352_; lean_object* v___x_354_; uint8_t v_isShared_355_; uint8_t v_isSharedCheck_363_; 
v_toAddMonoid_350_ = lean_ctor_get(v_inst_349_, 0);
v_toNeg_351_ = lean_ctor_get(v_inst_349_, 1);
v_toZSMul_352_ = lean_ctor_get(v_inst_349_, 3);
v_isSharedCheck_363_ = !lean_is_exclusive(v_inst_349_);
if (v_isSharedCheck_363_ == 0)
{
lean_object* v_unused_364_; 
v_unused_364_ = lean_ctor_get(v_inst_349_, 2);
lean_dec(v_unused_364_);
v___x_354_ = v_inst_349_;
v_isShared_355_ = v_isSharedCheck_363_;
goto v_resetjp_353_;
}
else
{
lean_inc(v_toZSMul_352_);
lean_inc(v_toNeg_351_);
lean_inc(v_toAddMonoid_350_);
lean_dec(v_inst_349_);
v___x_354_ = lean_box(0);
v_isShared_355_ = v_isSharedCheck_363_;
goto v_resetjp_353_;
}
v_resetjp_353_:
{
lean_object* v___f_356_; lean_object* v___x_357_; lean_object* v___f_358_; lean_object* v___x_359_; lean_object* v___x_361_; 
v___f_356_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instSubNegMonoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_356_, 0, v_toZSMul_352_);
v___x_357_ = lp_mathlib_AddOpposite_instAddMonoid___redArg(v_toAddMonoid_350_);
v___f_358_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNeg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_358_, 0, v_toNeg_351_);
lean_inc_ref(v___f_358_);
lean_inc_ref(v___x_357_);
v___x_359_ = lean_alloc_closure((void*)(lp_mathlib_SubNegMonoid_sub_x27), 5, 3);
lean_closure_set(v___x_359_, 0, lean_box(0));
lean_closure_set(v___x_359_, 1, v___x_357_);
lean_closure_set(v___x_359_, 2, v___f_358_);
if (v_isShared_355_ == 0)
{
lean_ctor_set(v___x_354_, 3, v___f_356_);
lean_ctor_set(v___x_354_, 2, v___x_359_);
lean_ctor_set(v___x_354_, 1, v___f_358_);
lean_ctor_set(v___x_354_, 0, v___x_357_);
v___x_361_ = v___x_354_;
goto v_reusejp_360_;
}
else
{
lean_object* v_reuseFailAlloc_362_; 
v_reuseFailAlloc_362_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_362_, 0, v___x_357_);
lean_ctor_set(v_reuseFailAlloc_362_, 1, v___f_358_);
lean_ctor_set(v_reuseFailAlloc_362_, 2, v___x_359_);
lean_ctor_set(v_reuseFailAlloc_362_, 3, v___f_356_);
v___x_361_ = v_reuseFailAlloc_362_;
goto v_reusejp_360_;
}
v_reusejp_360_:
{
return v___x_361_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSubNegMonoid(lean_object* v_00_u03b1_365_, lean_object* v_inst_366_){
_start:
{
lean_object* v___x_367_; 
v___x_367_ = lp_mathlib_AddOpposite_instSubNegMonoid___redArg(v_inst_366_);
return v___x_367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDivisionMonoid___redArg(lean_object* v_inst_368_){
_start:
{
lean_object* v___x_369_; 
v___x_369_ = lp_mathlib_MulOpposite_instDivInvMonoid___redArg(v_inst_368_);
return v___x_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDivisionMonoid(lean_object* v_00_u03b1_370_, lean_object* v_inst_371_){
_start:
{
lean_object* v___x_372_; 
v___x_372_ = lp_mathlib_MulOpposite_instDivInvMonoid___redArg(v_inst_371_);
return v___x_372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSubtractionMonoid___redArg(lean_object* v_inst_373_){
_start:
{
lean_object* v___x_374_; 
v___x_374_ = lp_mathlib_AddOpposite_instSubNegMonoid___redArg(v_inst_373_);
return v___x_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSubtractionMonoid(lean_object* v_00_u03b1_375_, lean_object* v_inst_376_){
_start:
{
lean_object* v___x_377_; 
v___x_377_ = lp_mathlib_AddOpposite_instSubNegMonoid___redArg(v_inst_376_);
return v___x_377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDivisionCommMonoid___redArg(lean_object* v_inst_378_){
_start:
{
lean_object* v___x_379_; 
v___x_379_ = lp_mathlib_MulOpposite_instDivInvMonoid___redArg(v_inst_378_);
return v___x_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instDivisionCommMonoid(lean_object* v_00_u03b1_380_, lean_object* v_inst_381_){
_start:
{
lean_object* v___x_382_; 
v___x_382_ = lp_mathlib_MulOpposite_instDivInvMonoid___redArg(v_inst_381_);
return v___x_382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSubtractionCommMonoid___redArg(lean_object* v_inst_383_){
_start:
{
lean_object* v___x_384_; 
v___x_384_ = lp_mathlib_AddOpposite_instSubNegMonoid___redArg(v_inst_383_);
return v___x_384_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSubtractionCommMonoid(lean_object* v_00_u03b1_385_, lean_object* v_inst_386_){
_start:
{
lean_object* v___x_387_; 
v___x_387_ = lp_mathlib_AddOpposite_instSubNegMonoid___redArg(v_inst_386_);
return v___x_387_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instGroup___redArg(lean_object* v_inst_388_){
_start:
{
lean_object* v___x_389_; 
v___x_389_ = lp_mathlib_MulOpposite_instDivInvMonoid___redArg(v_inst_388_);
return v___x_389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instGroup(lean_object* v_00_u03b1_390_, lean_object* v_inst_391_){
_start:
{
lean_object* v___x_392_; 
v___x_392_ = lp_mathlib_MulOpposite_instDivInvMonoid___redArg(v_inst_391_);
return v___x_392_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddGroup___redArg(lean_object* v_inst_393_){
_start:
{
lean_object* v___x_394_; 
v___x_394_ = lp_mathlib_AddOpposite_instSubNegMonoid___redArg(v_inst_393_);
return v___x_394_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddGroup(lean_object* v_00_u03b1_395_, lean_object* v_inst_396_){
_start:
{
lean_object* v___x_397_; 
v___x_397_ = lp_mathlib_AddOpposite_instSubNegMonoid___redArg(v_inst_396_);
return v___x_397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instCommGroup___redArg(lean_object* v_inst_398_){
_start:
{
lean_object* v___x_399_; 
v___x_399_ = lp_mathlib_MulOpposite_instDivInvMonoid___redArg(v_inst_398_);
return v___x_399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instCommGroup(lean_object* v_00_u03b1_400_, lean_object* v_inst_401_){
_start:
{
lean_object* v___x_402_; 
v___x_402_ = lp_mathlib_MulOpposite_instDivInvMonoid___redArg(v_inst_401_);
return v___x_402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddCommGroup___redArg(lean_object* v_inst_403_){
_start:
{
lean_object* v___x_404_; 
v___x_404_ = lp_mathlib_AddOpposite_instSubNegMonoid___redArg(v_inst_403_);
return v___x_404_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instAddCommGroup(lean_object* v_00_u03b1_405_, lean_object* v_inst_406_){
_start:
{
lean_object* v___x_407_; 
v___x_407_ = lp_mathlib_AddOpposite_instSubNegMonoid___redArg(v_inst_406_);
return v___x_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSemigroup___redArg(lean_object* v_inst_408_){
_start:
{
lean_object* v___f_409_; 
v___f_409_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_409_, 0, v_inst_408_);
return v___f_409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instSemigroup(lean_object* v_00_u03b1_410_, lean_object* v_inst_411_){
_start:
{
lean_object* v___f_412_; 
v___f_412_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_412_, 0, v_inst_411_);
return v___f_412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instLeftCancelSemigroup___redArg(lean_object* v_inst_413_){
_start:
{
lean_object* v___f_414_; 
v___f_414_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_414_, 0, v_inst_413_);
return v___f_414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instLeftCancelSemigroup(lean_object* v_00_u03b1_415_, lean_object* v_inst_416_){
_start:
{
lean_object* v___f_417_; 
v___f_417_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_417_, 0, v_inst_416_);
return v___f_417_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instRightCancelSemigroup___redArg(lean_object* v_inst_418_){
_start:
{
lean_object* v___f_419_; 
v___f_419_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_419_, 0, v_inst_418_);
return v___f_419_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instRightCancelSemigroup(lean_object* v_00_u03b1_420_, lean_object* v_inst_421_){
_start:
{
lean_object* v___f_422_; 
v___f_422_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_422_, 0, v_inst_421_);
return v___f_422_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instCommSemigroup___redArg(lean_object* v_inst_423_){
_start:
{
lean_object* v___f_424_; 
v___f_424_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_424_, 0, v_inst_423_);
return v___f_424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instCommSemigroup(lean_object* v_00_u03b1_425_, lean_object* v_inst_426_){
_start:
{
lean_object* v___f_427_; 
v___f_427_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_427_, 0, v_inst_426_);
return v___f_427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMulOneClass___redArg(lean_object* v_inst_428_){
_start:
{
lean_object* v___x_429_; lean_object* v_toOne_430_; lean_object* v_toMul_431_; lean_object* v___x_433_; uint8_t v_isShared_434_; uint8_t v_isSharedCheck_439_; 
v___x_429_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_428_);
v_toOne_430_ = lean_ctor_get(v___x_429_, 0);
v_toMul_431_ = lean_ctor_get(v___x_429_, 1);
v_isSharedCheck_439_ = !lean_is_exclusive(v___x_429_);
if (v_isSharedCheck_439_ == 0)
{
v___x_433_ = v___x_429_;
v_isShared_434_ = v_isSharedCheck_439_;
goto v_resetjp_432_;
}
else
{
lean_inc(v_toMul_431_);
lean_inc(v_toOne_430_);
lean_dec(v___x_429_);
v___x_433_ = lean_box(0);
v_isShared_434_ = v_isSharedCheck_439_;
goto v_resetjp_432_;
}
v_resetjp_432_:
{
lean_object* v___f_435_; lean_object* v___x_437_; 
v___f_435_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_435_, 0, v_toMul_431_);
if (v_isShared_434_ == 0)
{
lean_ctor_set(v___x_433_, 1, v___f_435_);
v___x_437_ = v___x_433_;
goto v_reusejp_436_;
}
else
{
lean_object* v_reuseFailAlloc_438_; 
v_reuseFailAlloc_438_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_438_, 0, v_toOne_430_);
lean_ctor_set(v_reuseFailAlloc_438_, 1, v___f_435_);
v___x_437_ = v_reuseFailAlloc_438_;
goto v_reusejp_436_;
}
v_reusejp_436_:
{
return v___x_437_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMulOneClass(lean_object* v_00_u03b1_440_, lean_object* v_inst_441_){
_start:
{
lean_object* v___x_442_; 
v___x_442_ = lp_mathlib_AddOpposite_instMulOneClass___redArg(v_inst_441_);
return v___x_442_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_pow___redArg___lam__0(lean_object* v_inst_443_, lean_object* v_a_444_, lean_object* v_b_445_){
_start:
{
lean_object* v___x_446_; 
v___x_446_ = lean_apply_2(v_inst_443_, v_a_444_, v_b_445_);
return v___x_446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_pow___redArg(lean_object* v_inst_447_){
_start:
{
lean_object* v___f_448_; 
v___f_448_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_pow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_448_, 0, v_inst_447_);
return v___f_448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_pow(lean_object* v_00_u03b1_449_, lean_object* v_00_u03b2_450_, lean_object* v_inst_451_){
_start:
{
lean_object* v___f_452_; 
v___f_452_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_pow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_452_, 0, v_inst_451_);
return v___f_452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMonoid___redArg___lam__0(lean_object* v_toNPow_453_, lean_object* v_n_454_, lean_object* v_x_455_){
_start:
{
lean_object* v___x_456_; 
v___x_456_ = lean_apply_2(v_toNPow_453_, v_n_454_, v_x_455_);
return v___x_456_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMonoid___redArg(lean_object* v_inst_457_){
_start:
{
lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v_toOne_460_; lean_object* v_toMul_461_; lean_object* v_toNPow_462_; lean_object* v___x_464_; uint8_t v_isShared_465_; uint8_t v_isSharedCheck_471_; 
v___x_458_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_457_);
v___x_459_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_458_);
v_toOne_460_ = lean_ctor_get(v___x_459_, 0);
lean_inc(v_toOne_460_);
v_toMul_461_ = lean_ctor_get(v___x_459_, 1);
lean_inc(v_toMul_461_);
lean_dec_ref(v___x_459_);
v_toNPow_462_ = lean_ctor_get(v_inst_457_, 2);
v_isSharedCheck_471_ = !lean_is_exclusive(v_inst_457_);
if (v_isSharedCheck_471_ == 0)
{
lean_object* v_unused_472_; lean_object* v_unused_473_; 
v_unused_472_ = lean_ctor_get(v_inst_457_, 1);
lean_dec(v_unused_472_);
v_unused_473_ = lean_ctor_get(v_inst_457_, 0);
lean_dec(v_unused_473_);
v___x_464_ = v_inst_457_;
v_isShared_465_ = v_isSharedCheck_471_;
goto v_resetjp_463_;
}
else
{
lean_inc(v_toNPow_462_);
lean_dec(v_inst_457_);
v___x_464_ = lean_box(0);
v_isShared_465_ = v_isSharedCheck_471_;
goto v_resetjp_463_;
}
v_resetjp_463_:
{
lean_object* v___f_466_; lean_object* v___f_467_; lean_object* v___x_469_; 
v___f_466_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_466_, 0, v_toMul_461_);
v___f_467_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instMonoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_467_, 0, v_toNPow_462_);
if (v_isShared_465_ == 0)
{
lean_ctor_set(v___x_464_, 2, v___f_467_);
lean_ctor_set(v___x_464_, 1, v___f_466_);
lean_ctor_set(v___x_464_, 0, v_toOne_460_);
v___x_469_ = v___x_464_;
goto v_reusejp_468_;
}
else
{
lean_object* v_reuseFailAlloc_470_; 
v_reuseFailAlloc_470_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_470_, 0, v_toOne_460_);
lean_ctor_set(v_reuseFailAlloc_470_, 1, v___f_466_);
lean_ctor_set(v_reuseFailAlloc_470_, 2, v___f_467_);
v___x_469_ = v_reuseFailAlloc_470_;
goto v_reusejp_468_;
}
v_reusejp_468_:
{
return v___x_469_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instMonoid(lean_object* v_00_u03b1_474_, lean_object* v_inst_475_){
_start:
{
lean_object* v___x_476_; 
v___x_476_ = lp_mathlib_AddOpposite_instMonoid___redArg(v_inst_475_);
return v___x_476_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instCommMonoid___redArg(lean_object* v_inst_477_){
_start:
{
lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v_toOne_480_; lean_object* v_toMul_481_; lean_object* v_toNPow_482_; lean_object* v___x_484_; uint8_t v_isShared_485_; uint8_t v_isSharedCheck_491_; 
v___x_478_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_477_);
v___x_479_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_478_);
v_toOne_480_ = lean_ctor_get(v___x_479_, 0);
lean_inc(v_toOne_480_);
v_toMul_481_ = lean_ctor_get(v___x_479_, 1);
lean_inc(v_toMul_481_);
lean_dec_ref(v___x_479_);
v_toNPow_482_ = lean_ctor_get(v_inst_477_, 2);
v_isSharedCheck_491_ = !lean_is_exclusive(v_inst_477_);
if (v_isSharedCheck_491_ == 0)
{
lean_object* v_unused_492_; lean_object* v_unused_493_; 
v_unused_492_ = lean_ctor_get(v_inst_477_, 1);
lean_dec(v_unused_492_);
v_unused_493_ = lean_ctor_get(v_inst_477_, 0);
lean_dec(v_unused_493_);
v___x_484_ = v_inst_477_;
v_isShared_485_ = v_isSharedCheck_491_;
goto v_resetjp_483_;
}
else
{
lean_inc(v_toNPow_482_);
lean_dec(v_inst_477_);
v___x_484_ = lean_box(0);
v_isShared_485_ = v_isSharedCheck_491_;
goto v_resetjp_483_;
}
v_resetjp_483_:
{
lean_object* v___f_486_; lean_object* v___f_487_; lean_object* v___x_489_; 
v___f_486_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_486_, 0, v_toMul_481_);
v___f_487_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instMonoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_487_, 0, v_toNPow_482_);
if (v_isShared_485_ == 0)
{
lean_ctor_set(v___x_484_, 2, v___f_487_);
lean_ctor_set(v___x_484_, 1, v___f_486_);
lean_ctor_set(v___x_484_, 0, v_toOne_480_);
v___x_489_ = v___x_484_;
goto v_reusejp_488_;
}
else
{
lean_object* v_reuseFailAlloc_490_; 
v_reuseFailAlloc_490_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_490_, 0, v_toOne_480_);
lean_ctor_set(v_reuseFailAlloc_490_, 1, v___f_486_);
lean_ctor_set(v_reuseFailAlloc_490_, 2, v___f_487_);
v___x_489_ = v_reuseFailAlloc_490_;
goto v_reusejp_488_;
}
v_reusejp_488_:
{
return v___x_489_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instCommMonoid(lean_object* v_00_u03b1_494_, lean_object* v_inst_495_){
_start:
{
lean_object* v___x_496_; 
v___x_496_ = lp_mathlib_AddOpposite_instCommMonoid___redArg(v_inst_495_);
return v___x_496_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDivInvMonoid___redArg___lam__0(lean_object* v_toZPow_497_, lean_object* v_n_498_, lean_object* v_x_499_){
_start:
{
lean_object* v___x_500_; 
v___x_500_ = lean_apply_2(v_toZPow_497_, v_n_498_, v_x_499_);
return v___x_500_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDivInvMonoid___redArg(lean_object* v_inst_501_){
_start:
{
lean_object* v_toMonoid_502_; lean_object* v_toInv_503_; lean_object* v_toDiv_504_; lean_object* v_toZPow_505_; lean_object* v___x_507_; uint8_t v_isShared_508_; uint8_t v_isSharedCheck_531_; 
v_toMonoid_502_ = lean_ctor_get(v_inst_501_, 0);
v_toInv_503_ = lean_ctor_get(v_inst_501_, 1);
v_toDiv_504_ = lean_ctor_get(v_inst_501_, 2);
v_toZPow_505_ = lean_ctor_get(v_inst_501_, 3);
v_isSharedCheck_531_ = !lean_is_exclusive(v_inst_501_);
if (v_isSharedCheck_531_ == 0)
{
v___x_507_ = v_inst_501_;
v_isShared_508_ = v_isSharedCheck_531_;
goto v_resetjp_506_;
}
else
{
lean_inc(v_toZPow_505_);
lean_inc(v_toDiv_504_);
lean_inc(v_toInv_503_);
lean_inc(v_toMonoid_502_);
lean_dec(v_inst_501_);
v___x_507_ = lean_box(0);
v_isShared_508_ = v_isSharedCheck_531_;
goto v_resetjp_506_;
}
v_resetjp_506_:
{
lean_object* v___x_509_; lean_object* v___x_510_; lean_object* v_toOne_511_; lean_object* v_toMul_512_; lean_object* v_toNPow_513_; lean_object* v___x_515_; uint8_t v_isShared_516_; uint8_t v_isSharedCheck_528_; 
v___x_509_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_502_);
v___x_510_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_509_);
v_toOne_511_ = lean_ctor_get(v___x_510_, 0);
lean_inc(v_toOne_511_);
v_toMul_512_ = lean_ctor_get(v___x_510_, 1);
lean_inc(v_toMul_512_);
lean_dec_ref(v___x_510_);
v_toNPow_513_ = lean_ctor_get(v_toMonoid_502_, 2);
v_isSharedCheck_528_ = !lean_is_exclusive(v_toMonoid_502_);
if (v_isSharedCheck_528_ == 0)
{
lean_object* v_unused_529_; lean_object* v_unused_530_; 
v_unused_529_ = lean_ctor_get(v_toMonoid_502_, 1);
lean_dec(v_unused_529_);
v_unused_530_ = lean_ctor_get(v_toMonoid_502_, 0);
lean_dec(v_unused_530_);
v___x_515_ = v_toMonoid_502_;
v_isShared_516_ = v_isSharedCheck_528_;
goto v_resetjp_514_;
}
else
{
lean_inc(v_toNPow_513_);
lean_dec(v_toMonoid_502_);
v___x_515_ = lean_box(0);
v_isShared_516_ = v_isSharedCheck_528_;
goto v_resetjp_514_;
}
v_resetjp_514_:
{
lean_object* v___f_517_; lean_object* v___f_518_; lean_object* v___f_519_; lean_object* v___f_520_; lean_object* v___f_521_; lean_object* v___x_523_; 
v___f_517_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instDivInvMonoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_517_, 0, v_toZPow_505_);
v___f_518_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_518_, 0, v_toMul_512_);
v___f_519_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instMonoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_519_, 0, v_toNPow_513_);
v___f_520_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instInv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_520_, 0, v_toInv_503_);
v___f_521_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_521_, 0, v_toDiv_504_);
if (v_isShared_516_ == 0)
{
lean_ctor_set(v___x_515_, 2, v___f_519_);
lean_ctor_set(v___x_515_, 1, v___f_518_);
lean_ctor_set(v___x_515_, 0, v_toOne_511_);
v___x_523_ = v___x_515_;
goto v_reusejp_522_;
}
else
{
lean_object* v_reuseFailAlloc_527_; 
v_reuseFailAlloc_527_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_527_, 0, v_toOne_511_);
lean_ctor_set(v_reuseFailAlloc_527_, 1, v___f_518_);
lean_ctor_set(v_reuseFailAlloc_527_, 2, v___f_519_);
v___x_523_ = v_reuseFailAlloc_527_;
goto v_reusejp_522_;
}
v_reusejp_522_:
{
lean_object* v___x_525_; 
if (v_isShared_508_ == 0)
{
lean_ctor_set(v___x_507_, 3, v___f_517_);
lean_ctor_set(v___x_507_, 2, v___f_521_);
lean_ctor_set(v___x_507_, 1, v___f_520_);
lean_ctor_set(v___x_507_, 0, v___x_523_);
v___x_525_ = v___x_507_;
goto v_reusejp_524_;
}
else
{
lean_object* v_reuseFailAlloc_526_; 
v_reuseFailAlloc_526_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_526_, 0, v___x_523_);
lean_ctor_set(v_reuseFailAlloc_526_, 1, v___f_520_);
lean_ctor_set(v_reuseFailAlloc_526_, 2, v___f_521_);
lean_ctor_set(v_reuseFailAlloc_526_, 3, v___f_517_);
v___x_525_ = v_reuseFailAlloc_526_;
goto v_reusejp_524_;
}
v_reusejp_524_:
{
return v___x_525_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instDivInvMonoid(lean_object* v_00_u03b1_532_, lean_object* v_inst_533_){
_start:
{
lean_object* v___x_534_; 
v___x_534_ = lp_mathlib_AddOpposite_instDivInvMonoid___redArg(v_inst_533_);
return v___x_534_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instGroup___redArg(lean_object* v_inst_535_){
_start:
{
lean_object* v_toMonoid_536_; lean_object* v_toDiv_537_; lean_object* v_toZPow_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v_toMul_541_; lean_object* v___x_542_; lean_object* v___x_544_; uint8_t v_isShared_545_; uint8_t v_isSharedCheck_566_; 
v_toMonoid_536_ = lean_ctor_get(v_inst_535_, 0);
lean_inc_ref(v_toMonoid_536_);
v_toDiv_537_ = lean_ctor_get(v_inst_535_, 2);
lean_inc(v_toDiv_537_);
v_toZPow_538_ = lean_ctor_get(v_inst_535_, 3);
lean_inc(v_toZPow_538_);
v___x_539_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_536_);
v___x_540_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_539_);
v_toMul_541_ = lean_ctor_get(v___x_540_, 1);
lean_inc(v_toMul_541_);
lean_dec_ref(v___x_540_);
v___x_542_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v_inst_535_);
v_isSharedCheck_566_ = !lean_is_exclusive(v_inst_535_);
if (v_isSharedCheck_566_ == 0)
{
lean_object* v_unused_567_; lean_object* v_unused_568_; lean_object* v_unused_569_; lean_object* v_unused_570_; 
v_unused_567_ = lean_ctor_get(v_inst_535_, 3);
lean_dec(v_unused_567_);
v_unused_568_ = lean_ctor_get(v_inst_535_, 2);
lean_dec(v_unused_568_);
v_unused_569_ = lean_ctor_get(v_inst_535_, 1);
lean_dec(v_unused_569_);
v_unused_570_ = lean_ctor_get(v_inst_535_, 0);
lean_dec(v_unused_570_);
v___x_544_ = v_inst_535_;
v_isShared_545_ = v_isSharedCheck_566_;
goto v_resetjp_543_;
}
else
{
lean_dec(v_inst_535_);
v___x_544_ = lean_box(0);
v_isShared_545_ = v_isSharedCheck_566_;
goto v_resetjp_543_;
}
v_resetjp_543_:
{
lean_object* v_toOne_546_; lean_object* v_toInv_547_; lean_object* v_toNPow_548_; lean_object* v___x_550_; uint8_t v_isShared_551_; uint8_t v_isSharedCheck_563_; 
v_toOne_546_ = lean_ctor_get(v___x_542_, 0);
lean_inc(v_toOne_546_);
v_toInv_547_ = lean_ctor_get(v___x_542_, 1);
lean_inc(v_toInv_547_);
lean_dec_ref(v___x_542_);
v_toNPow_548_ = lean_ctor_get(v_toMonoid_536_, 2);
v_isSharedCheck_563_ = !lean_is_exclusive(v_toMonoid_536_);
if (v_isSharedCheck_563_ == 0)
{
lean_object* v_unused_564_; lean_object* v_unused_565_; 
v_unused_564_ = lean_ctor_get(v_toMonoid_536_, 1);
lean_dec(v_unused_564_);
v_unused_565_ = lean_ctor_get(v_toMonoid_536_, 0);
lean_dec(v_unused_565_);
v___x_550_ = v_toMonoid_536_;
v_isShared_551_ = v_isSharedCheck_563_;
goto v_resetjp_549_;
}
else
{
lean_inc(v_toNPow_548_);
lean_dec(v_toMonoid_536_);
v___x_550_ = lean_box(0);
v_isShared_551_ = v_isSharedCheck_563_;
goto v_resetjp_549_;
}
v_resetjp_549_:
{
lean_object* v___f_552_; lean_object* v___f_553_; lean_object* v___f_554_; lean_object* v___f_555_; lean_object* v___f_556_; lean_object* v___x_558_; 
v___f_552_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instDivInvMonoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_552_, 0, v_toZPow_538_);
v___f_553_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_553_, 0, v_toMul_541_);
v___f_554_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instMonoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_554_, 0, v_toNPow_548_);
v___f_555_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instInv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_555_, 0, v_toInv_547_);
v___f_556_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_556_, 0, v_toDiv_537_);
if (v_isShared_551_ == 0)
{
lean_ctor_set(v___x_550_, 2, v___f_554_);
lean_ctor_set(v___x_550_, 1, v___f_553_);
lean_ctor_set(v___x_550_, 0, v_toOne_546_);
v___x_558_ = v___x_550_;
goto v_reusejp_557_;
}
else
{
lean_object* v_reuseFailAlloc_562_; 
v_reuseFailAlloc_562_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_562_, 0, v_toOne_546_);
lean_ctor_set(v_reuseFailAlloc_562_, 1, v___f_553_);
lean_ctor_set(v_reuseFailAlloc_562_, 2, v___f_554_);
v___x_558_ = v_reuseFailAlloc_562_;
goto v_reusejp_557_;
}
v_reusejp_557_:
{
lean_object* v___x_560_; 
if (v_isShared_545_ == 0)
{
lean_ctor_set(v___x_544_, 3, v___f_552_);
lean_ctor_set(v___x_544_, 2, v___f_556_);
lean_ctor_set(v___x_544_, 1, v___f_555_);
lean_ctor_set(v___x_544_, 0, v___x_558_);
v___x_560_ = v___x_544_;
goto v_reusejp_559_;
}
else
{
lean_object* v_reuseFailAlloc_561_; 
v_reuseFailAlloc_561_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_561_, 0, v___x_558_);
lean_ctor_set(v_reuseFailAlloc_561_, 1, v___f_555_);
lean_ctor_set(v_reuseFailAlloc_561_, 2, v___f_556_);
lean_ctor_set(v_reuseFailAlloc_561_, 3, v___f_552_);
v___x_560_ = v_reuseFailAlloc_561_;
goto v_reusejp_559_;
}
v_reusejp_559_:
{
return v___x_560_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instGroup(lean_object* v_00_u03b1_571_, lean_object* v_inst_572_){
_start:
{
lean_object* v___x_573_; 
v___x_573_ = lp_mathlib_AddOpposite_instGroup___redArg(v_inst_572_);
return v___x_573_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instCommGroup___redArg(lean_object* v_inst_574_){
_start:
{
lean_object* v_toMonoid_575_; lean_object* v_toDiv_576_; lean_object* v_toZPow_577_; lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v_toMul_580_; lean_object* v___x_581_; lean_object* v___x_583_; uint8_t v_isShared_584_; uint8_t v_isSharedCheck_605_; 
v_toMonoid_575_ = lean_ctor_get(v_inst_574_, 0);
lean_inc_ref(v_toMonoid_575_);
v_toDiv_576_ = lean_ctor_get(v_inst_574_, 2);
lean_inc(v_toDiv_576_);
v_toZPow_577_ = lean_ctor_get(v_inst_574_, 3);
lean_inc(v_toZPow_577_);
v___x_578_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_575_);
v___x_579_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_578_);
v_toMul_580_ = lean_ctor_get(v___x_579_, 1);
lean_inc(v_toMul_580_);
lean_dec_ref(v___x_579_);
v___x_581_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v_inst_574_);
v_isSharedCheck_605_ = !lean_is_exclusive(v_inst_574_);
if (v_isSharedCheck_605_ == 0)
{
lean_object* v_unused_606_; lean_object* v_unused_607_; lean_object* v_unused_608_; lean_object* v_unused_609_; 
v_unused_606_ = lean_ctor_get(v_inst_574_, 3);
lean_dec(v_unused_606_);
v_unused_607_ = lean_ctor_get(v_inst_574_, 2);
lean_dec(v_unused_607_);
v_unused_608_ = lean_ctor_get(v_inst_574_, 1);
lean_dec(v_unused_608_);
v_unused_609_ = lean_ctor_get(v_inst_574_, 0);
lean_dec(v_unused_609_);
v___x_583_ = v_inst_574_;
v_isShared_584_ = v_isSharedCheck_605_;
goto v_resetjp_582_;
}
else
{
lean_dec(v_inst_574_);
v___x_583_ = lean_box(0);
v_isShared_584_ = v_isSharedCheck_605_;
goto v_resetjp_582_;
}
v_resetjp_582_:
{
lean_object* v_toOne_585_; lean_object* v_toInv_586_; lean_object* v_toNPow_587_; lean_object* v___x_589_; uint8_t v_isShared_590_; uint8_t v_isSharedCheck_602_; 
v_toOne_585_ = lean_ctor_get(v___x_581_, 0);
lean_inc(v_toOne_585_);
v_toInv_586_ = lean_ctor_get(v___x_581_, 1);
lean_inc(v_toInv_586_);
lean_dec_ref(v___x_581_);
v_toNPow_587_ = lean_ctor_get(v_toMonoid_575_, 2);
v_isSharedCheck_602_ = !lean_is_exclusive(v_toMonoid_575_);
if (v_isSharedCheck_602_ == 0)
{
lean_object* v_unused_603_; lean_object* v_unused_604_; 
v_unused_603_ = lean_ctor_get(v_toMonoid_575_, 1);
lean_dec(v_unused_603_);
v_unused_604_ = lean_ctor_get(v_toMonoid_575_, 0);
lean_dec(v_unused_604_);
v___x_589_ = v_toMonoid_575_;
v_isShared_590_ = v_isSharedCheck_602_;
goto v_resetjp_588_;
}
else
{
lean_inc(v_toNPow_587_);
lean_dec(v_toMonoid_575_);
v___x_589_ = lean_box(0);
v_isShared_590_ = v_isSharedCheck_602_;
goto v_resetjp_588_;
}
v_resetjp_588_:
{
lean_object* v___f_591_; lean_object* v___f_592_; lean_object* v___f_593_; lean_object* v___f_594_; lean_object* v___f_595_; lean_object* v___x_597_; 
v___f_591_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instDivInvMonoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_591_, 0, v_toZPow_577_);
v___f_592_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_592_, 0, v_toMul_580_);
v___f_593_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instMonoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_593_, 0, v_toNPow_587_);
v___f_594_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instInv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_594_, 0, v_toInv_586_);
v___f_595_ = lean_alloc_closure((void*)(lp_mathlib_AddOpposite_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_595_, 0, v_toDiv_576_);
if (v_isShared_590_ == 0)
{
lean_ctor_set(v___x_589_, 2, v___f_593_);
lean_ctor_set(v___x_589_, 1, v___f_592_);
lean_ctor_set(v___x_589_, 0, v_toOne_585_);
v___x_597_ = v___x_589_;
goto v_reusejp_596_;
}
else
{
lean_object* v_reuseFailAlloc_601_; 
v_reuseFailAlloc_601_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_601_, 0, v_toOne_585_);
lean_ctor_set(v_reuseFailAlloc_601_, 1, v___f_592_);
lean_ctor_set(v_reuseFailAlloc_601_, 2, v___f_593_);
v___x_597_ = v_reuseFailAlloc_601_;
goto v_reusejp_596_;
}
v_reusejp_596_:
{
lean_object* v___x_599_; 
if (v_isShared_584_ == 0)
{
lean_ctor_set(v___x_583_, 3, v___f_591_);
lean_ctor_set(v___x_583_, 2, v___f_595_);
lean_ctor_set(v___x_583_, 1, v___f_594_);
lean_ctor_set(v___x_583_, 0, v___x_597_);
v___x_599_ = v___x_583_;
goto v_reusejp_598_;
}
else
{
lean_object* v_reuseFailAlloc_600_; 
v_reuseFailAlloc_600_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_600_, 0, v___x_597_);
lean_ctor_set(v_reuseFailAlloc_600_, 1, v___f_594_);
lean_ctor_set(v_reuseFailAlloc_600_, 2, v___f_595_);
lean_ctor_set(v_reuseFailAlloc_600_, 3, v___f_591_);
v___x_599_ = v_reuseFailAlloc_600_;
goto v_reusejp_598_;
}
v_reusejp_598_:
{
return v___x_599_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instCommGroup(lean_object* v_00_u03b1_610_, lean_object* v_inst_611_){
_start:
{
lean_object* v___x_612_; 
v___x_612_ = lp_mathlib_AddOpposite_instCommGroup___redArg(v_inst_611_);
return v___x_612_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Commute_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_InjSurj(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Torsion(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Opposites(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Conv(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Opposite(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Torsion(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Opposites(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Conv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Opposite(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_InjSurj(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Torsion(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Opposites(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Conv(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Opposite(uint8_t builtin) {
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
res = initialize_mathlib_Mathlib_Algebra_Group_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Torsion(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Opposites(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Conv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Opposite(builtin);
}
#ifdef __cplusplus
}
#endif
