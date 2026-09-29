// Lean compiler output
// Module: Mathlib.Algebra.Field.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Field.Defs public import Mathlib.Algebra.Group.SelfInv public import Mathlib.Algebra.Ring.GrindInstances public import Mathlib.Algebra.Ring.Commute public import Mathlib.Algebra.Ring.Invertible public import Mathlib.Order.OrderDual public import Mathlib.Order.Lex public import Mathlib.Algebra.Order.Ring.Synonym import Mathlib.Tactic.Tauto
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
lean_object* lp_mathlib_Function_Injective_subNegMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Int_cast(lean_object*, lean_object*, lean_object*);
lean_object* l_Nat_cast(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_addMonoid___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lex_instRing___redArg(lean_object*);
lean_object* lp_mathlib_Field_toSemifield___redArg(lean_object*);
lean_object* lp_mathlib_Semifield_toCommGroupWithZero___redArg(lean_object*);
lean_object* lp_mathlib_CommGroupWithZero_toDivisionCommMonoid___redArg(lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
lean_object* lp_mathlib_Field_toDivisionRing___redArg(lean_object*);
lean_object* lp_mathlib_DivisionRing_toDivInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Lex_instDivInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_DivisionRing_toDivisionSemiring___redArg(lean_object*);
lean_object* lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(lean_object*);
lean_object* lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Lex_instSemiring___redArg(lean_object*);
lean_object* lp_mathlib_OrderDual_instSemiring___redArg(lean_object*);
lean_object* lp_mathlib_OrderDual_instDivInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_OrderDual_instRing___redArg(lean_object*);
lean_object* lp_mathlib_Semifield_toDivisionSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toGrindRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Field_toGrindField___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Field_toGrindField___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Field_toGrindField(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionSemiring___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionRing___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionRing___redArg___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionRing___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semifield___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semifield(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semifield___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_field___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_field___redArg___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_field(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_field___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instRatCast___aux__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instRatCast___aux__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instRatCast___aux__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instRatCast___aux__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instRatCast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instRatCast___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instRatCast(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instRatCast___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instNNRatCast___aux__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instNNRatCast___aux__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instNNRatCast___aux__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instNNRatCast___aux__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instNNRatCast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instNNRatCast___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instNNRatCast(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instNNRatCast___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDivisionSemiring___aux__9___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDivisionSemiring___aux__9(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDivisionSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDivisionSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDivisionRing___aux__9___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDivisionRing___aux__9(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDivisionRing___aux__13___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDivisionRing___aux__13(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDivisionRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDivisionRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instSemifield___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instSemifield(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instField___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instField(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instRatCast___aux__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instRatCast___aux__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instRatCast___aux__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instRatCast___aux__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instRatCast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instRatCast___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instRatCast(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instRatCast___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instDivisionSemiring___aux__8___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instDivisionSemiring___aux__8___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instDivisionSemiring___aux__8(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instDivisionSemiring___aux__8___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instDivisionSemiring___aux__11___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instDivisionSemiring___aux__11(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instDivisionSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instDivisionSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instDivisionRing___aux__9___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instDivisionRing___aux__9(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instDivisionRing___aux__13___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instDivisionRing___aux__13(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instDivisionRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instDivisionRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instSemifield___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instSemifield(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instField___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lex_instField(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Field_toGrindField___redArg___lam__0(lean_object* v_toZPow_1_, lean_object* v_a_2_, lean_object* v_n_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_apply_2(v_toZPow_1_, v_n_3_, v_a_2_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Field_toGrindField___redArg(lean_object* v_inst_5_){
_start:
{
lean_object* v_toCommRing_6_; lean_object* v_toInv_7_; lean_object* v_toDiv_8_; lean_object* v___x_9_; lean_object* v___x_10_; lean_object* v___x_11_; lean_object* v_toZPow_12_; lean_object* v___x_14_; uint8_t v_isShared_15_; uint8_t v_isSharedCheck_20_; 
v_toCommRing_6_ = lean_ctor_get(v_inst_5_, 0);
v_toInv_7_ = lean_ctor_get(v_inst_5_, 1);
lean_inc(v_toInv_7_);
v_toDiv_8_ = lean_ctor_get(v_inst_5_, 2);
lean_inc(v_toDiv_8_);
lean_inc_ref(v_toCommRing_6_);
v___x_9_ = lp_mathlib_Ring_toGrindRing___redArg(v_toCommRing_6_);
v___x_10_ = lp_mathlib_Field_toDivisionRing___redArg(v_inst_5_);
v___x_11_ = lp_mathlib_DivisionRing_toDivInvMonoid___redArg(v___x_10_);
lean_dec_ref(v___x_10_);
v_toZPow_12_ = lean_ctor_get(v___x_11_, 3);
v_isSharedCheck_20_ = !lean_is_exclusive(v___x_11_);
if (v_isSharedCheck_20_ == 0)
{
lean_object* v_unused_21_; lean_object* v_unused_22_; lean_object* v_unused_23_; 
v_unused_21_ = lean_ctor_get(v___x_11_, 2);
lean_dec(v_unused_21_);
v_unused_22_ = lean_ctor_get(v___x_11_, 1);
lean_dec(v_unused_22_);
v_unused_23_ = lean_ctor_get(v___x_11_, 0);
lean_dec(v_unused_23_);
v___x_14_ = v___x_11_;
v_isShared_15_ = v_isSharedCheck_20_;
goto v_resetjp_13_;
}
else
{
lean_inc(v_toZPow_12_);
lean_dec(v___x_11_);
v___x_14_ = lean_box(0);
v_isShared_15_ = v_isSharedCheck_20_;
goto v_resetjp_13_;
}
v_resetjp_13_:
{
lean_object* v___f_16_; lean_object* v___x_18_; 
v___f_16_ = lean_alloc_closure((void*)(lp_mathlib_Field_toGrindField___redArg___lam__0), 3, 1);
lean_closure_set(v___f_16_, 0, v_toZPow_12_);
if (v_isShared_15_ == 0)
{
lean_ctor_set(v___x_14_, 3, v___f_16_);
lean_ctor_set(v___x_14_, 2, v_toDiv_8_);
lean_ctor_set(v___x_14_, 1, v_toInv_7_);
lean_ctor_set(v___x_14_, 0, v___x_9_);
v___x_18_ = v___x_14_;
goto v_reusejp_17_;
}
else
{
lean_object* v_reuseFailAlloc_19_; 
v_reuseFailAlloc_19_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_19_, 0, v___x_9_);
lean_ctor_set(v_reuseFailAlloc_19_, 1, v_toInv_7_);
lean_ctor_set(v_reuseFailAlloc_19_, 2, v_toDiv_8_);
lean_ctor_set(v_reuseFailAlloc_19_, 3, v___f_16_);
v___x_18_ = v_reuseFailAlloc_19_;
goto v_reusejp_17_;
}
v_reusejp_17_:
{
return v___x_18_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Field_toGrindField(lean_object* v_K_24_, lean_object* v_inst_25_){
_start:
{
lean_object* v___x_26_; 
v___x_26_ = lp_mathlib_Field_toGrindField___redArg(v_inst_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__0(lean_object* v_inst_27_, lean_object* v_n_28_, lean_object* v_x_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lean_apply_2(v_inst_27_, v_x_29_, v_n_28_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__1(lean_object* v_inst_31_, lean_object* v_n_32_, lean_object* v_x_33_){
_start:
{
lean_object* v___x_34_; 
v___x_34_ = lean_apply_2(v_inst_31_, v_x_33_, v_n_32_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__2(lean_object* v_inst_35_, lean_object* v_x1_36_, lean_object* v_x2_37_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = lean_apply_2(v_inst_35_, v_x1_36_, v_x2_37_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionSemiring___redArg(lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_inst_41_, lean_object* v_inst_42_, lean_object* v_inst_43_, lean_object* v_inst_44_, lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_inst_49_, lean_object* v_inst_50_){
_start:
{
lean_object* v___f_51_; lean_object* v___f_52_; lean_object* v___f_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; 
v___f_51_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_51_, 0, v_inst_48_);
v___f_52_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__1), 3, 1);
lean_closure_set(v___f_52_, 0, v_inst_47_);
v___f_53_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__2), 3, 1);
lean_closure_set(v___f_53_, 0, v_inst_46_);
v___x_54_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_54_, 0, lean_box(0));
lean_closure_set(v___x_54_, 1, v_inst_49_);
v___x_55_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_40_, v_inst_39_, v_inst_45_);
v___x_56_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_56_, 0, v_inst_41_);
lean_ctor_set(v___x_56_, 1, v_inst_42_);
lean_ctor_set(v___x_56_, 2, v___f_52_);
v___x_57_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_57_, 0, v___x_55_);
lean_ctor_set(v___x_57_, 1, v___x_56_);
lean_ctor_set(v___x_57_, 2, v___x_54_);
v___x_58_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_58_, 0, v___x_57_);
lean_ctor_set(v___x_58_, 1, v_inst_43_);
lean_ctor_set(v___x_58_, 2, v_inst_44_);
lean_ctor_set(v___x_58_, 3, v___f_51_);
lean_ctor_set(v___x_58_, 4, v_inst_50_);
lean_ctor_set(v___x_58_, 5, v___f_53_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionSemiring(lean_object* v_K_59_, lean_object* v_L_60_, lean_object* v_inst_61_, lean_object* v_inst_62_, lean_object* v_inst_63_, lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_inst_68_, lean_object* v_inst_69_, lean_object* v_inst_70_, lean_object* v_inst_71_, lean_object* v_inst_72_, lean_object* v_f_73_, lean_object* v_hf_74_, lean_object* v_inst_75_, lean_object* v_zero_76_, lean_object* v_one_77_, lean_object* v_add_78_, lean_object* v_mul_79_, lean_object* v_inv_80_, lean_object* v_div_81_, lean_object* v_nsmul_82_, lean_object* v_nnqsmul_83_, lean_object* v_npow_84_, lean_object* v_zpow_85_, lean_object* v_natCast_86_, lean_object* v_nnratCast_87_){
_start:
{
lean_object* v___f_88_; lean_object* v___f_89_; lean_object* v___f_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; 
v___f_88_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_88_, 0, v_inst_70_);
v___f_89_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__1), 3, 1);
lean_closure_set(v___f_89_, 0, v_inst_69_);
v___f_90_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__2), 3, 1);
lean_closure_set(v___f_90_, 0, v_inst_68_);
v___x_91_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_91_, 0, lean_box(0));
lean_closure_set(v___x_91_, 1, v_inst_71_);
v___x_92_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_62_, v_inst_61_, v_inst_67_);
v___x_93_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_93_, 0, v_inst_63_);
lean_ctor_set(v___x_93_, 1, v_inst_64_);
lean_ctor_set(v___x_93_, 2, v___f_89_);
v___x_94_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_94_, 0, v___x_92_);
lean_ctor_set(v___x_94_, 1, v___x_93_);
lean_ctor_set(v___x_94_, 2, v___x_91_);
v___x_95_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_95_, 0, v___x_94_);
lean_ctor_set(v___x_95_, 1, v_inst_65_);
lean_ctor_set(v___x_95_, 2, v_inst_66_);
lean_ctor_set(v___x_95_, 3, v___f_88_);
lean_ctor_set(v___x_95_, 4, v_inst_72_);
lean_ctor_set(v___x_95_, 5, v___f_90_);
return v___x_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionSemiring___boxed(lean_object** _args){
lean_object* v_K_96_ = _args[0];
lean_object* v_L_97_ = _args[1];
lean_object* v_inst_98_ = _args[2];
lean_object* v_inst_99_ = _args[3];
lean_object* v_inst_100_ = _args[4];
lean_object* v_inst_101_ = _args[5];
lean_object* v_inst_102_ = _args[6];
lean_object* v_inst_103_ = _args[7];
lean_object* v_inst_104_ = _args[8];
lean_object* v_inst_105_ = _args[9];
lean_object* v_inst_106_ = _args[10];
lean_object* v_inst_107_ = _args[11];
lean_object* v_inst_108_ = _args[12];
lean_object* v_inst_109_ = _args[13];
lean_object* v_f_110_ = _args[14];
lean_object* v_hf_111_ = _args[15];
lean_object* v_inst_112_ = _args[16];
lean_object* v_zero_113_ = _args[17];
lean_object* v_one_114_ = _args[18];
lean_object* v_add_115_ = _args[19];
lean_object* v_mul_116_ = _args[20];
lean_object* v_inv_117_ = _args[21];
lean_object* v_div_118_ = _args[22];
lean_object* v_nsmul_119_ = _args[23];
lean_object* v_nnqsmul_120_ = _args[24];
lean_object* v_npow_121_ = _args[25];
lean_object* v_zpow_122_ = _args[26];
lean_object* v_natCast_123_ = _args[27];
lean_object* v_nnratCast_124_ = _args[28];
_start:
{
lean_object* v_res_125_; 
v_res_125_ = lp_mathlib_Function_Injective_divisionSemiring(v_K_96_, v_L_97_, v_inst_98_, v_inst_99_, v_inst_100_, v_inst_101_, v_inst_102_, v_inst_103_, v_inst_104_, v_inst_105_, v_inst_106_, v_inst_107_, v_inst_108_, v_inst_109_, v_f_110_, v_hf_111_, v_inst_112_, v_zero_113_, v_one_114_, v_add_115_, v_mul_116_, v_inv_117_, v_div_118_, v_nsmul_119_, v_nnqsmul_120_, v_npow_121_, v_zpow_122_, v_natCast_123_, v_nnratCast_124_);
lean_dec_ref(v_inst_112_);
lean_dec(v_f_110_);
return v_res_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionRing___redArg___lam__3(lean_object* v_inst_126_, lean_object* v_n_127_, lean_object* v_x_128_){
_start:
{
lean_object* v___x_129_; 
v___x_129_ = lean_apply_2(v_inst_126_, v_n_127_, v_x_128_);
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionRing___redArg(lean_object* v_inst_130_, lean_object* v_inst_131_, lean_object* v_inst_132_, lean_object* v_inst_133_, lean_object* v_inst_134_, lean_object* v_inst_135_, lean_object* v_inst_136_, lean_object* v_inst_137_, lean_object* v_inst_138_, lean_object* v_inst_139_, lean_object* v_inst_140_, lean_object* v_inst_141_, lean_object* v_inst_142_, lean_object* v_inst_143_, lean_object* v_inst_144_, lean_object* v_inst_145_, lean_object* v_inst_146_, lean_object* v_inst_147_){
_start:
{
lean_object* v___x_148_; lean_object* v_toNeg_149_; lean_object* v_toSub_150_; lean_object* v___f_151_; lean_object* v___f_152_; lean_object* v___f_153_; lean_object* v___f_154_; lean_object* v___f_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; 
lean_inc(v_inst_139_);
lean_inc(v_inst_138_);
lean_inc(v_inst_130_);
lean_inc(v_inst_131_);
v___x_148_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_131_, v_inst_130_, v_inst_138_, v_inst_132_, v_inst_133_, v_inst_139_);
v_toNeg_149_ = lean_ctor_get(v___x_148_, 1);
lean_inc(v_toNeg_149_);
v_toSub_150_ = lean_ctor_get(v___x_148_, 2);
lean_inc(v_toSub_150_);
lean_dec_ref(v___x_148_);
v___f_151_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_151_, 0, v_inst_143_);
v___f_152_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__2), 3, 1);
lean_closure_set(v___f_152_, 0, v_inst_140_);
v___f_153_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__1), 3, 1);
lean_closure_set(v___f_153_, 0, v_inst_142_);
v___f_154_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionRing___redArg___lam__3), 3, 1);
lean_closure_set(v___f_154_, 0, v_inst_139_);
v___f_155_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__2), 3, 1);
lean_closure_set(v___f_155_, 0, v_inst_141_);
v___x_156_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_156_, 0, lean_box(0));
lean_closure_set(v___x_156_, 1, v_inst_145_);
v___x_157_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_157_, 0, lean_box(0));
lean_closure_set(v___x_157_, 1, v_inst_144_);
v___x_158_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_131_, v_inst_130_, v_inst_138_);
v___x_159_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_159_, 0, v_inst_134_);
lean_ctor_set(v___x_159_, 1, v_inst_135_);
lean_ctor_set(v___x_159_, 2, v___f_153_);
v___x_160_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_160_, 0, v___x_158_);
lean_ctor_set(v___x_160_, 1, v___x_159_);
lean_ctor_set(v___x_160_, 2, v___x_157_);
v___x_161_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_161_, 0, v___x_160_);
lean_ctor_set(v___x_161_, 1, v_toNeg_149_);
lean_ctor_set(v___x_161_, 2, v_toSub_150_);
lean_ctor_set(v___x_161_, 3, v___f_154_);
lean_ctor_set(v___x_161_, 4, v___x_156_);
v___x_162_ = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(v___x_162_, 0, v___x_161_);
lean_ctor_set(v___x_162_, 1, v_inst_136_);
lean_ctor_set(v___x_162_, 2, v_inst_137_);
lean_ctor_set(v___x_162_, 3, v___f_151_);
lean_ctor_set(v___x_162_, 4, v_inst_146_);
lean_ctor_set(v___x_162_, 5, v_inst_147_);
lean_ctor_set(v___x_162_, 6, v___f_152_);
lean_ctor_set(v___x_162_, 7, v___f_155_);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionRing___redArg___boxed(lean_object** _args){
lean_object* v_inst_163_ = _args[0];
lean_object* v_inst_164_ = _args[1];
lean_object* v_inst_165_ = _args[2];
lean_object* v_inst_166_ = _args[3];
lean_object* v_inst_167_ = _args[4];
lean_object* v_inst_168_ = _args[5];
lean_object* v_inst_169_ = _args[6];
lean_object* v_inst_170_ = _args[7];
lean_object* v_inst_171_ = _args[8];
lean_object* v_inst_172_ = _args[9];
lean_object* v_inst_173_ = _args[10];
lean_object* v_inst_174_ = _args[11];
lean_object* v_inst_175_ = _args[12];
lean_object* v_inst_176_ = _args[13];
lean_object* v_inst_177_ = _args[14];
lean_object* v_inst_178_ = _args[15];
lean_object* v_inst_179_ = _args[16];
lean_object* v_inst_180_ = _args[17];
_start:
{
lean_object* v_res_181_; 
v_res_181_ = lp_mathlib_Function_Injective_divisionRing___redArg(v_inst_163_, v_inst_164_, v_inst_165_, v_inst_166_, v_inst_167_, v_inst_168_, v_inst_169_, v_inst_170_, v_inst_171_, v_inst_172_, v_inst_173_, v_inst_174_, v_inst_175_, v_inst_176_, v_inst_177_, v_inst_178_, v_inst_179_, v_inst_180_);
return v_res_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionRing(lean_object* v_K_182_, lean_object* v_L_183_, lean_object* v_inst_184_, lean_object* v_inst_185_, lean_object* v_inst_186_, lean_object* v_inst_187_, lean_object* v_inst_188_, lean_object* v_inst_189_, lean_object* v_inst_190_, lean_object* v_inst_191_, lean_object* v_inst_192_, lean_object* v_inst_193_, lean_object* v_inst_194_, lean_object* v_inst_195_, lean_object* v_inst_196_, lean_object* v_inst_197_, lean_object* v_inst_198_, lean_object* v_inst_199_, lean_object* v_inst_200_, lean_object* v_inst_201_, lean_object* v_f_202_, lean_object* v_hf_203_, lean_object* v_inst_204_, lean_object* v_zero_205_, lean_object* v_one_206_, lean_object* v_add_207_, lean_object* v_mul_208_, lean_object* v_neg_209_, lean_object* v_sub_210_, lean_object* v_inv_211_, lean_object* v_div_212_, lean_object* v_nsmul_213_, lean_object* v_zsmul_214_, lean_object* v_nnqsmul_215_, lean_object* v_qsmul_216_, lean_object* v_npow_217_, lean_object* v_zpow_218_, lean_object* v_natCast_219_, lean_object* v_intCast_220_, lean_object* v_nnratCast_221_, lean_object* v_ratCast_222_){
_start:
{
lean_object* v___x_223_; lean_object* v_toNeg_224_; lean_object* v_toSub_225_; lean_object* v___f_226_; lean_object* v___f_227_; lean_object* v___f_228_; lean_object* v___f_229_; lean_object* v___f_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; 
lean_inc(v_inst_193_);
lean_inc(v_inst_192_);
lean_inc(v_inst_184_);
lean_inc(v_inst_185_);
v___x_223_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_185_, v_inst_184_, v_inst_192_, v_inst_186_, v_inst_187_, v_inst_193_);
v_toNeg_224_ = lean_ctor_get(v___x_223_, 1);
lean_inc(v_toNeg_224_);
v_toSub_225_ = lean_ctor_get(v___x_223_, 2);
lean_inc(v_toSub_225_);
lean_dec_ref(v___x_223_);
v___f_226_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_226_, 0, v_inst_197_);
v___f_227_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__2), 3, 1);
lean_closure_set(v___f_227_, 0, v_inst_194_);
v___f_228_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__1), 3, 1);
lean_closure_set(v___f_228_, 0, v_inst_196_);
v___f_229_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionRing___redArg___lam__3), 3, 1);
lean_closure_set(v___f_229_, 0, v_inst_193_);
v___f_230_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__2), 3, 1);
lean_closure_set(v___f_230_, 0, v_inst_195_);
v___x_231_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_231_, 0, lean_box(0));
lean_closure_set(v___x_231_, 1, v_inst_199_);
v___x_232_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_232_, 0, lean_box(0));
lean_closure_set(v___x_232_, 1, v_inst_198_);
v___x_233_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_185_, v_inst_184_, v_inst_192_);
v___x_234_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_234_, 0, v_inst_188_);
lean_ctor_set(v___x_234_, 1, v_inst_189_);
lean_ctor_set(v___x_234_, 2, v___f_228_);
v___x_235_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_235_, 0, v___x_233_);
lean_ctor_set(v___x_235_, 1, v___x_234_);
lean_ctor_set(v___x_235_, 2, v___x_232_);
v___x_236_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_236_, 0, v___x_235_);
lean_ctor_set(v___x_236_, 1, v_toNeg_224_);
lean_ctor_set(v___x_236_, 2, v_toSub_225_);
lean_ctor_set(v___x_236_, 3, v___f_229_);
lean_ctor_set(v___x_236_, 4, v___x_231_);
v___x_237_ = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(v___x_237_, 0, v___x_236_);
lean_ctor_set(v___x_237_, 1, v_inst_190_);
lean_ctor_set(v___x_237_, 2, v_inst_191_);
lean_ctor_set(v___x_237_, 3, v___f_226_);
lean_ctor_set(v___x_237_, 4, v_inst_200_);
lean_ctor_set(v___x_237_, 5, v_inst_201_);
lean_ctor_set(v___x_237_, 6, v___f_227_);
lean_ctor_set(v___x_237_, 7, v___f_230_);
return v___x_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionRing___boxed(lean_object** _args){
lean_object* v_K_238_ = _args[0];
lean_object* v_L_239_ = _args[1];
lean_object* v_inst_240_ = _args[2];
lean_object* v_inst_241_ = _args[3];
lean_object* v_inst_242_ = _args[4];
lean_object* v_inst_243_ = _args[5];
lean_object* v_inst_244_ = _args[6];
lean_object* v_inst_245_ = _args[7];
lean_object* v_inst_246_ = _args[8];
lean_object* v_inst_247_ = _args[9];
lean_object* v_inst_248_ = _args[10];
lean_object* v_inst_249_ = _args[11];
lean_object* v_inst_250_ = _args[12];
lean_object* v_inst_251_ = _args[13];
lean_object* v_inst_252_ = _args[14];
lean_object* v_inst_253_ = _args[15];
lean_object* v_inst_254_ = _args[16];
lean_object* v_inst_255_ = _args[17];
lean_object* v_inst_256_ = _args[18];
lean_object* v_inst_257_ = _args[19];
lean_object* v_f_258_ = _args[20];
lean_object* v_hf_259_ = _args[21];
lean_object* v_inst_260_ = _args[22];
lean_object* v_zero_261_ = _args[23];
lean_object* v_one_262_ = _args[24];
lean_object* v_add_263_ = _args[25];
lean_object* v_mul_264_ = _args[26];
lean_object* v_neg_265_ = _args[27];
lean_object* v_sub_266_ = _args[28];
lean_object* v_inv_267_ = _args[29];
lean_object* v_div_268_ = _args[30];
lean_object* v_nsmul_269_ = _args[31];
lean_object* v_zsmul_270_ = _args[32];
lean_object* v_nnqsmul_271_ = _args[33];
lean_object* v_qsmul_272_ = _args[34];
lean_object* v_npow_273_ = _args[35];
lean_object* v_zpow_274_ = _args[36];
lean_object* v_natCast_275_ = _args[37];
lean_object* v_intCast_276_ = _args[38];
lean_object* v_nnratCast_277_ = _args[39];
lean_object* v_ratCast_278_ = _args[40];
_start:
{
lean_object* v_res_279_; 
v_res_279_ = lp_mathlib_Function_Injective_divisionRing(v_K_238_, v_L_239_, v_inst_240_, v_inst_241_, v_inst_242_, v_inst_243_, v_inst_244_, v_inst_245_, v_inst_246_, v_inst_247_, v_inst_248_, v_inst_249_, v_inst_250_, v_inst_251_, v_inst_252_, v_inst_253_, v_inst_254_, v_inst_255_, v_inst_256_, v_inst_257_, v_f_258_, v_hf_259_, v_inst_260_, v_zero_261_, v_one_262_, v_add_263_, v_mul_264_, v_neg_265_, v_sub_266_, v_inv_267_, v_div_268_, v_nsmul_269_, v_zsmul_270_, v_nnqsmul_271_, v_qsmul_272_, v_npow_273_, v_zpow_274_, v_natCast_275_, v_intCast_276_, v_nnratCast_277_, v_ratCast_278_);
lean_dec_ref(v_inst_260_);
lean_dec(v_f_258_);
return v_res_279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semifield___redArg(lean_object* v_inst_280_, lean_object* v_inst_281_, lean_object* v_inst_282_, lean_object* v_inst_283_, lean_object* v_inst_284_, lean_object* v_inst_285_, lean_object* v_inst_286_, lean_object* v_inst_287_, lean_object* v_inst_288_, lean_object* v_inst_289_, lean_object* v_inst_290_, lean_object* v_inst_291_){
_start:
{
lean_object* v___f_292_; lean_object* v___f_293_; lean_object* v___f_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; 
v___f_292_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_292_, 0, v_inst_289_);
v___f_293_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__2), 3, 1);
lean_closure_set(v___f_293_, 0, v_inst_287_);
v___f_294_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__1), 3, 1);
lean_closure_set(v___f_294_, 0, v_inst_288_);
v___x_295_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_295_, 0, lean_box(0));
lean_closure_set(v___x_295_, 1, v_inst_290_);
v___x_296_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_281_, v_inst_280_, v_inst_286_);
v___x_297_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_297_, 0, v_inst_282_);
lean_ctor_set(v___x_297_, 1, v_inst_283_);
lean_ctor_set(v___x_297_, 2, v___f_294_);
v___x_298_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_298_, 0, v___x_296_);
lean_ctor_set(v___x_298_, 1, v___x_297_);
lean_ctor_set(v___x_298_, 2, v___x_295_);
v___x_299_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_299_, 0, v___x_298_);
lean_ctor_set(v___x_299_, 1, v_inst_284_);
lean_ctor_set(v___x_299_, 2, v_inst_285_);
lean_ctor_set(v___x_299_, 3, v___f_292_);
lean_ctor_set(v___x_299_, 4, v_inst_291_);
lean_ctor_set(v___x_299_, 5, v___f_293_);
return v___x_299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semifield(lean_object* v_K_300_, lean_object* v_L_301_, lean_object* v_inst_302_, lean_object* v_inst_303_, lean_object* v_inst_304_, lean_object* v_inst_305_, lean_object* v_inst_306_, lean_object* v_inst_307_, lean_object* v_inst_308_, lean_object* v_inst_309_, lean_object* v_inst_310_, lean_object* v_inst_311_, lean_object* v_inst_312_, lean_object* v_inst_313_, lean_object* v_f_314_, lean_object* v_hf_315_, lean_object* v_inst_316_, lean_object* v_zero_317_, lean_object* v_one_318_, lean_object* v_add_319_, lean_object* v_mul_320_, lean_object* v_inv_321_, lean_object* v_div_322_, lean_object* v_nsmul_323_, lean_object* v_nnqsmul_324_, lean_object* v_npow_325_, lean_object* v_zpow_326_, lean_object* v_natCast_327_, lean_object* v_nnratCast_328_){
_start:
{
lean_object* v___f_329_; lean_object* v___f_330_; lean_object* v___f_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; 
v___f_329_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_329_, 0, v_inst_311_);
v___f_330_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__2), 3, 1);
lean_closure_set(v___f_330_, 0, v_inst_309_);
v___f_331_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__1), 3, 1);
lean_closure_set(v___f_331_, 0, v_inst_310_);
v___x_332_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_332_, 0, lean_box(0));
lean_closure_set(v___x_332_, 1, v_inst_312_);
v___x_333_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_303_, v_inst_302_, v_inst_308_);
v___x_334_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_334_, 0, v_inst_304_);
lean_ctor_set(v___x_334_, 1, v_inst_305_);
lean_ctor_set(v___x_334_, 2, v___f_331_);
v___x_335_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_335_, 0, v___x_333_);
lean_ctor_set(v___x_335_, 1, v___x_334_);
lean_ctor_set(v___x_335_, 2, v___x_332_);
v___x_336_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_336_, 0, v___x_335_);
lean_ctor_set(v___x_336_, 1, v_inst_306_);
lean_ctor_set(v___x_336_, 2, v_inst_307_);
lean_ctor_set(v___x_336_, 3, v___f_329_);
lean_ctor_set(v___x_336_, 4, v_inst_313_);
lean_ctor_set(v___x_336_, 5, v___f_330_);
return v___x_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semifield___boxed(lean_object** _args){
lean_object* v_K_337_ = _args[0];
lean_object* v_L_338_ = _args[1];
lean_object* v_inst_339_ = _args[2];
lean_object* v_inst_340_ = _args[3];
lean_object* v_inst_341_ = _args[4];
lean_object* v_inst_342_ = _args[5];
lean_object* v_inst_343_ = _args[6];
lean_object* v_inst_344_ = _args[7];
lean_object* v_inst_345_ = _args[8];
lean_object* v_inst_346_ = _args[9];
lean_object* v_inst_347_ = _args[10];
lean_object* v_inst_348_ = _args[11];
lean_object* v_inst_349_ = _args[12];
lean_object* v_inst_350_ = _args[13];
lean_object* v_f_351_ = _args[14];
lean_object* v_hf_352_ = _args[15];
lean_object* v_inst_353_ = _args[16];
lean_object* v_zero_354_ = _args[17];
lean_object* v_one_355_ = _args[18];
lean_object* v_add_356_ = _args[19];
lean_object* v_mul_357_ = _args[20];
lean_object* v_inv_358_ = _args[21];
lean_object* v_div_359_ = _args[22];
lean_object* v_nsmul_360_ = _args[23];
lean_object* v_nnqsmul_361_ = _args[24];
lean_object* v_npow_362_ = _args[25];
lean_object* v_zpow_363_ = _args[26];
lean_object* v_natCast_364_ = _args[27];
lean_object* v_nnratCast_365_ = _args[28];
_start:
{
lean_object* v_res_366_; 
v_res_366_ = lp_mathlib_Function_Injective_semifield(v_K_337_, v_L_338_, v_inst_339_, v_inst_340_, v_inst_341_, v_inst_342_, v_inst_343_, v_inst_344_, v_inst_345_, v_inst_346_, v_inst_347_, v_inst_348_, v_inst_349_, v_inst_350_, v_f_351_, v_hf_352_, v_inst_353_, v_zero_354_, v_one_355_, v_add_356_, v_mul_357_, v_inv_358_, v_div_359_, v_nsmul_360_, v_nnqsmul_361_, v_npow_362_, v_zpow_363_, v_natCast_364_, v_nnratCast_365_);
lean_dec_ref(v_inst_353_);
lean_dec(v_f_351_);
return v_res_366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_field___redArg(lean_object* v_inst_367_, lean_object* v_inst_368_, lean_object* v_inst_369_, lean_object* v_inst_370_, lean_object* v_inst_371_, lean_object* v_inst_372_, lean_object* v_inst_373_, lean_object* v_inst_374_, lean_object* v_inst_375_, lean_object* v_inst_376_, lean_object* v_inst_377_, lean_object* v_inst_378_, lean_object* v_inst_379_, lean_object* v_inst_380_, lean_object* v_inst_381_, lean_object* v_inst_382_, lean_object* v_inst_383_, lean_object* v_inst_384_){
_start:
{
lean_object* v___x_385_; lean_object* v_toNeg_386_; lean_object* v_toSub_387_; lean_object* v___f_388_; lean_object* v___f_389_; lean_object* v___f_390_; lean_object* v___f_391_; lean_object* v___f_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; 
lean_inc(v_inst_376_);
lean_inc(v_inst_375_);
lean_inc(v_inst_367_);
lean_inc(v_inst_368_);
v___x_385_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_368_, v_inst_367_, v_inst_375_, v_inst_369_, v_inst_370_, v_inst_376_);
v_toNeg_386_ = lean_ctor_get(v___x_385_, 1);
lean_inc(v_toNeg_386_);
v_toSub_387_ = lean_ctor_get(v___x_385_, 2);
lean_inc(v_toSub_387_);
lean_dec_ref(v___x_385_);
v___f_388_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_388_, 0, v_inst_380_);
v___f_389_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__2), 3, 1);
lean_closure_set(v___f_389_, 0, v_inst_377_);
v___f_390_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__2), 3, 1);
lean_closure_set(v___f_390_, 0, v_inst_378_);
v___f_391_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__1), 3, 1);
lean_closure_set(v___f_391_, 0, v_inst_379_);
v___f_392_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionRing___redArg___lam__3), 3, 1);
lean_closure_set(v___f_392_, 0, v_inst_376_);
v___x_393_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_393_, 0, lean_box(0));
lean_closure_set(v___x_393_, 1, v_inst_382_);
v___x_394_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_394_, 0, lean_box(0));
lean_closure_set(v___x_394_, 1, v_inst_381_);
v___x_395_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_368_, v_inst_367_, v_inst_375_);
v___x_396_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_396_, 0, v_inst_371_);
lean_ctor_set(v___x_396_, 1, v_inst_372_);
lean_ctor_set(v___x_396_, 2, v___f_391_);
v___x_397_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_397_, 0, v___x_395_);
lean_ctor_set(v___x_397_, 1, v___x_396_);
lean_ctor_set(v___x_397_, 2, v___x_394_);
v___x_398_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_398_, 0, v___x_397_);
lean_ctor_set(v___x_398_, 1, v_toNeg_386_);
lean_ctor_set(v___x_398_, 2, v_toSub_387_);
lean_ctor_set(v___x_398_, 3, v___f_392_);
lean_ctor_set(v___x_398_, 4, v___x_393_);
v___x_399_ = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(v___x_399_, 0, v___x_398_);
lean_ctor_set(v___x_399_, 1, v_inst_373_);
lean_ctor_set(v___x_399_, 2, v_inst_374_);
lean_ctor_set(v___x_399_, 3, v___f_388_);
lean_ctor_set(v___x_399_, 4, v_inst_383_);
lean_ctor_set(v___x_399_, 5, v_inst_384_);
lean_ctor_set(v___x_399_, 6, v___f_389_);
lean_ctor_set(v___x_399_, 7, v___f_390_);
return v___x_399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_field___redArg___boxed(lean_object** _args){
lean_object* v_inst_400_ = _args[0];
lean_object* v_inst_401_ = _args[1];
lean_object* v_inst_402_ = _args[2];
lean_object* v_inst_403_ = _args[3];
lean_object* v_inst_404_ = _args[4];
lean_object* v_inst_405_ = _args[5];
lean_object* v_inst_406_ = _args[6];
lean_object* v_inst_407_ = _args[7];
lean_object* v_inst_408_ = _args[8];
lean_object* v_inst_409_ = _args[9];
lean_object* v_inst_410_ = _args[10];
lean_object* v_inst_411_ = _args[11];
lean_object* v_inst_412_ = _args[12];
lean_object* v_inst_413_ = _args[13];
lean_object* v_inst_414_ = _args[14];
lean_object* v_inst_415_ = _args[15];
lean_object* v_inst_416_ = _args[16];
lean_object* v_inst_417_ = _args[17];
_start:
{
lean_object* v_res_418_; 
v_res_418_ = lp_mathlib_Function_Injective_field___redArg(v_inst_400_, v_inst_401_, v_inst_402_, v_inst_403_, v_inst_404_, v_inst_405_, v_inst_406_, v_inst_407_, v_inst_408_, v_inst_409_, v_inst_410_, v_inst_411_, v_inst_412_, v_inst_413_, v_inst_414_, v_inst_415_, v_inst_416_, v_inst_417_);
return v_res_418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_field(lean_object* v_K_419_, lean_object* v_L_420_, lean_object* v_inst_421_, lean_object* v_inst_422_, lean_object* v_inst_423_, lean_object* v_inst_424_, lean_object* v_inst_425_, lean_object* v_inst_426_, lean_object* v_inst_427_, lean_object* v_inst_428_, lean_object* v_inst_429_, lean_object* v_inst_430_, lean_object* v_inst_431_, lean_object* v_inst_432_, lean_object* v_inst_433_, lean_object* v_inst_434_, lean_object* v_inst_435_, lean_object* v_inst_436_, lean_object* v_inst_437_, lean_object* v_inst_438_, lean_object* v_f_439_, lean_object* v_hf_440_, lean_object* v_inst_441_, lean_object* v_zero_442_, lean_object* v_one_443_, lean_object* v_add_444_, lean_object* v_mul_445_, lean_object* v_neg_446_, lean_object* v_sub_447_, lean_object* v_inv_448_, lean_object* v_div_449_, lean_object* v_nsmul_450_, lean_object* v_zsmul_451_, lean_object* v_nnqsmul_452_, lean_object* v_qsmul_453_, lean_object* v_npow_454_, lean_object* v_zpow_455_, lean_object* v_natCast_456_, lean_object* v_intCast_457_, lean_object* v_nnratCast_458_, lean_object* v_ratCast_459_){
_start:
{
lean_object* v___x_460_; lean_object* v_toNeg_461_; lean_object* v_toSub_462_; lean_object* v___f_463_; lean_object* v___f_464_; lean_object* v___f_465_; lean_object* v___f_466_; lean_object* v___f_467_; lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; 
lean_inc(v_inst_430_);
lean_inc(v_inst_429_);
lean_inc(v_inst_421_);
lean_inc(v_inst_422_);
v___x_460_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_422_, v_inst_421_, v_inst_429_, v_inst_423_, v_inst_424_, v_inst_430_);
v_toNeg_461_ = lean_ctor_get(v___x_460_, 1);
lean_inc(v_toNeg_461_);
v_toSub_462_ = lean_ctor_get(v___x_460_, 2);
lean_inc(v_toSub_462_);
lean_dec_ref(v___x_460_);
v___f_463_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_463_, 0, v_inst_434_);
v___f_464_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__2), 3, 1);
lean_closure_set(v___f_464_, 0, v_inst_431_);
v___f_465_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__2), 3, 1);
lean_closure_set(v___f_465_, 0, v_inst_432_);
v___f_466_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionSemiring___redArg___lam__1), 3, 1);
lean_closure_set(v___f_466_, 0, v_inst_433_);
v___f_467_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divisionRing___redArg___lam__3), 3, 1);
lean_closure_set(v___f_467_, 0, v_inst_430_);
v___x_468_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_468_, 0, lean_box(0));
lean_closure_set(v___x_468_, 1, v_inst_436_);
v___x_469_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_469_, 0, lean_box(0));
lean_closure_set(v___x_469_, 1, v_inst_435_);
v___x_470_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_422_, v_inst_421_, v_inst_429_);
v___x_471_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_471_, 0, v_inst_425_);
lean_ctor_set(v___x_471_, 1, v_inst_426_);
lean_ctor_set(v___x_471_, 2, v___f_466_);
v___x_472_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_472_, 0, v___x_470_);
lean_ctor_set(v___x_472_, 1, v___x_471_);
lean_ctor_set(v___x_472_, 2, v___x_469_);
v___x_473_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_473_, 0, v___x_472_);
lean_ctor_set(v___x_473_, 1, v_toNeg_461_);
lean_ctor_set(v___x_473_, 2, v_toSub_462_);
lean_ctor_set(v___x_473_, 3, v___f_467_);
lean_ctor_set(v___x_473_, 4, v___x_468_);
v___x_474_ = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(v___x_474_, 0, v___x_473_);
lean_ctor_set(v___x_474_, 1, v_inst_427_);
lean_ctor_set(v___x_474_, 2, v_inst_428_);
lean_ctor_set(v___x_474_, 3, v___f_463_);
lean_ctor_set(v___x_474_, 4, v_inst_437_);
lean_ctor_set(v___x_474_, 5, v_inst_438_);
lean_ctor_set(v___x_474_, 6, v___f_464_);
lean_ctor_set(v___x_474_, 7, v___f_465_);
return v___x_474_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_field___boxed(lean_object** _args){
lean_object* v_K_475_ = _args[0];
lean_object* v_L_476_ = _args[1];
lean_object* v_inst_477_ = _args[2];
lean_object* v_inst_478_ = _args[3];
lean_object* v_inst_479_ = _args[4];
lean_object* v_inst_480_ = _args[5];
lean_object* v_inst_481_ = _args[6];
lean_object* v_inst_482_ = _args[7];
lean_object* v_inst_483_ = _args[8];
lean_object* v_inst_484_ = _args[9];
lean_object* v_inst_485_ = _args[10];
lean_object* v_inst_486_ = _args[11];
lean_object* v_inst_487_ = _args[12];
lean_object* v_inst_488_ = _args[13];
lean_object* v_inst_489_ = _args[14];
lean_object* v_inst_490_ = _args[15];
lean_object* v_inst_491_ = _args[16];
lean_object* v_inst_492_ = _args[17];
lean_object* v_inst_493_ = _args[18];
lean_object* v_inst_494_ = _args[19];
lean_object* v_f_495_ = _args[20];
lean_object* v_hf_496_ = _args[21];
lean_object* v_inst_497_ = _args[22];
lean_object* v_zero_498_ = _args[23];
lean_object* v_one_499_ = _args[24];
lean_object* v_add_500_ = _args[25];
lean_object* v_mul_501_ = _args[26];
lean_object* v_neg_502_ = _args[27];
lean_object* v_sub_503_ = _args[28];
lean_object* v_inv_504_ = _args[29];
lean_object* v_div_505_ = _args[30];
lean_object* v_nsmul_506_ = _args[31];
lean_object* v_zsmul_507_ = _args[32];
lean_object* v_nnqsmul_508_ = _args[33];
lean_object* v_qsmul_509_ = _args[34];
lean_object* v_npow_510_ = _args[35];
lean_object* v_zpow_511_ = _args[36];
lean_object* v_natCast_512_ = _args[37];
lean_object* v_intCast_513_ = _args[38];
lean_object* v_nnratCast_514_ = _args[39];
lean_object* v_ratCast_515_ = _args[40];
_start:
{
lean_object* v_res_516_; 
v_res_516_ = lp_mathlib_Function_Injective_field(v_K_475_, v_L_476_, v_inst_477_, v_inst_478_, v_inst_479_, v_inst_480_, v_inst_481_, v_inst_482_, v_inst_483_, v_inst_484_, v_inst_485_, v_inst_486_, v_inst_487_, v_inst_488_, v_inst_489_, v_inst_490_, v_inst_491_, v_inst_492_, v_inst_493_, v_inst_494_, v_f_495_, v_hf_496_, v_inst_497_, v_zero_498_, v_one_499_, v_add_500_, v_mul_501_, v_neg_502_, v_sub_503_, v_inv_504_, v_div_505_, v_nsmul_506_, v_zsmul_507_, v_nnqsmul_508_, v_qsmul_509_, v_npow_510_, v_zpow_511_, v_natCast_512_, v_intCast_513_, v_nnratCast_514_, v_ratCast_515_);
lean_dec_ref(v_inst_497_);
lean_dec(v_f_495_);
return v_res_516_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instRatCast___aux__1___redArg(lean_object* v_inst_517_){
_start:
{
lean_inc(v_inst_517_);
return v_inst_517_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instRatCast___aux__1___redArg___boxed(lean_object* v_inst_518_){
_start:
{
lean_object* v_res_519_; 
v_res_519_ = lp_mathlib_OrderDual_instRatCast___aux__1___redArg(v_inst_518_);
lean_dec(v_inst_518_);
return v_res_519_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instRatCast___aux__1(lean_object* v_K_520_, lean_object* v_inst_521_){
_start:
{
lean_inc(v_inst_521_);
return v_inst_521_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instRatCast___aux__1___boxed(lean_object* v_K_522_, lean_object* v_inst_523_){
_start:
{
lean_object* v_res_524_; 
v_res_524_ = lp_mathlib_OrderDual_instRatCast___aux__1(v_K_522_, v_inst_523_);
lean_dec(v_inst_523_);
return v_res_524_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instRatCast___redArg(lean_object* v_inst_525_){
_start:
{
lean_inc(v_inst_525_);
return v_inst_525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instRatCast___redArg___boxed(lean_object* v_inst_526_){
_start:
{
lean_object* v_res_527_; 
v_res_527_ = lp_mathlib_OrderDual_instRatCast___redArg(v_inst_526_);
lean_dec(v_inst_526_);
return v_res_527_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instRatCast(lean_object* v_K_528_, lean_object* v_inst_529_){
_start:
{
lean_inc(v_inst_529_);
return v_inst_529_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instRatCast___boxed(lean_object* v_K_530_, lean_object* v_inst_531_){
_start:
{
lean_object* v_res_532_; 
v_res_532_ = lp_mathlib_OrderDual_instRatCast(v_K_530_, v_inst_531_);
lean_dec(v_inst_531_);
return v_res_532_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instNNRatCast___aux__1___redArg(lean_object* v_inst_533_){
_start:
{
lean_inc(v_inst_533_);
return v_inst_533_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instNNRatCast___aux__1___redArg___boxed(lean_object* v_inst_534_){
_start:
{
lean_object* v_res_535_; 
v_res_535_ = lp_mathlib_OrderDual_instNNRatCast___aux__1___redArg(v_inst_534_);
lean_dec(v_inst_534_);
return v_res_535_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instNNRatCast___aux__1(lean_object* v_K_536_, lean_object* v_inst_537_){
_start:
{
lean_inc(v_inst_537_);
return v_inst_537_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instNNRatCast___aux__1___boxed(lean_object* v_K_538_, lean_object* v_inst_539_){
_start:
{
lean_object* v_res_540_; 
v_res_540_ = lp_mathlib_OrderDual_instNNRatCast___aux__1(v_K_538_, v_inst_539_);
lean_dec(v_inst_539_);
return v_res_540_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instNNRatCast___redArg(lean_object* v_inst_541_){
_start:
{
lean_inc(v_inst_541_);
return v_inst_541_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instNNRatCast___redArg___boxed(lean_object* v_inst_542_){
_start:
{
lean_object* v_res_543_; 
v_res_543_ = lp_mathlib_OrderDual_instNNRatCast___redArg(v_inst_542_);
lean_dec(v_inst_542_);
return v_res_543_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instNNRatCast(lean_object* v_K_544_, lean_object* v_inst_545_){
_start:
{
lean_inc(v_inst_545_);
return v_inst_545_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instNNRatCast___boxed(lean_object* v_K_546_, lean_object* v_inst_547_){
_start:
{
lean_object* v_res_548_; 
v_res_548_ = lp_mathlib_OrderDual_instNNRatCast(v_K_546_, v_inst_547_);
lean_dec(v_inst_547_);
return v_res_548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDivisionSemiring___aux__9___redArg(lean_object* v_inst_549_, lean_object* v_a_550_, lean_object* v_a_551_){
_start:
{
lean_object* v_nnqsmul_552_; lean_object* v___x_553_; 
v_nnqsmul_552_ = lean_ctor_get(v_inst_549_, 5);
lean_inc(v_nnqsmul_552_);
lean_dec_ref(v_inst_549_);
v___x_553_ = lean_apply_2(v_nnqsmul_552_, v_a_550_, v_a_551_);
return v___x_553_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDivisionSemiring___aux__9(lean_object* v_K_554_, lean_object* v_inst_555_, lean_object* v_a_556_, lean_object* v_a_557_){
_start:
{
lean_object* v_nnqsmul_558_; lean_object* v___x_559_; 
v_nnqsmul_558_ = lean_ctor_get(v_inst_555_, 5);
lean_inc(v_nnqsmul_558_);
lean_dec_ref(v_inst_555_);
v___x_559_ = lean_apply_2(v_nnqsmul_558_, v_a_556_, v_a_557_);
return v___x_559_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDivisionSemiring___redArg(lean_object* v_inst_560_){
_start:
{
lean_object* v_toSemiring_561_; lean_object* v_toNNRatCast_562_; lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v_toInv_567_; lean_object* v_toDiv_568_; lean_object* v___x_569_; lean_object* v_toZPow_570_; lean_object* v___x_571_; lean_object* v___x_572_; 
v_toSemiring_561_ = lean_ctor_get(v_inst_560_, 0);
v_toNNRatCast_562_ = lean_ctor_get(v_inst_560_, 4);
lean_inc(v_toNNRatCast_562_);
lean_inc_ref(v_toSemiring_561_);
v___x_563_ = lp_mathlib_OrderDual_instSemiring___redArg(v_toSemiring_561_);
v___x_564_ = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(v_inst_560_);
v___x_565_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v___x_564_);
v___x_566_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v___x_565_);
v_toInv_567_ = lean_ctor_get(v___x_566_, 1);
lean_inc(v_toInv_567_);
lean_dec_ref(v___x_566_);
v_toDiv_568_ = lean_ctor_get(v___x_565_, 2);
lean_inc(v_toDiv_568_);
v___x_569_ = lp_mathlib_OrderDual_instDivInvMonoid___redArg(v___x_565_);
v_toZPow_570_ = lean_ctor_get(v___x_569_, 3);
lean_inc(v_toZPow_570_);
lean_dec_ref(v___x_569_);
v___x_571_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instDivisionSemiring___aux__9), 4, 2);
lean_closure_set(v___x_571_, 0, lean_box(0));
lean_closure_set(v___x_571_, 1, v_inst_560_);
v___x_572_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_572_, 0, v___x_563_);
lean_ctor_set(v___x_572_, 1, v_toInv_567_);
lean_ctor_set(v___x_572_, 2, v_toDiv_568_);
lean_ctor_set(v___x_572_, 3, v_toZPow_570_);
lean_ctor_set(v___x_572_, 4, v_toNNRatCast_562_);
lean_ctor_set(v___x_572_, 5, v___x_571_);
return v___x_572_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDivisionSemiring(lean_object* v_K_573_, lean_object* v_inst_574_){
_start:
{
lean_object* v___x_575_; 
v___x_575_ = lp_mathlib_OrderDual_instDivisionSemiring___redArg(v_inst_574_);
return v___x_575_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDivisionRing___aux__9___redArg(lean_object* v_inst_576_, lean_object* v_a_577_, lean_object* v_a_578_){
_start:
{
lean_object* v_nnqsmul_579_; lean_object* v___x_580_; 
v_nnqsmul_579_ = lean_ctor_get(v_inst_576_, 6);
lean_inc(v_nnqsmul_579_);
lean_dec_ref(v_inst_576_);
v___x_580_ = lean_apply_2(v_nnqsmul_579_, v_a_577_, v_a_578_);
return v___x_580_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDivisionRing___aux__9(lean_object* v_K_581_, lean_object* v_inst_582_, lean_object* v_a_583_, lean_object* v_a_584_){
_start:
{
lean_object* v_nnqsmul_585_; lean_object* v___x_586_; 
v_nnqsmul_585_ = lean_ctor_get(v_inst_582_, 6);
lean_inc(v_nnqsmul_585_);
lean_dec_ref(v_inst_582_);
v___x_586_ = lean_apply_2(v_nnqsmul_585_, v_a_583_, v_a_584_);
return v___x_586_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDivisionRing___aux__13___redArg(lean_object* v_inst_587_, lean_object* v_a_588_, lean_object* v_a_589_){
_start:
{
lean_object* v_qsmul_590_; lean_object* v___x_591_; 
v_qsmul_590_ = lean_ctor_get(v_inst_587_, 7);
lean_inc(v_qsmul_590_);
lean_dec_ref(v_inst_587_);
v___x_591_ = lean_apply_2(v_qsmul_590_, v_a_588_, v_a_589_);
return v___x_591_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDivisionRing___aux__13(lean_object* v_K_592_, lean_object* v_inst_593_, lean_object* v_a_594_, lean_object* v_a_595_){
_start:
{
lean_object* v_qsmul_596_; lean_object* v___x_597_; 
v_qsmul_596_ = lean_ctor_get(v_inst_593_, 7);
lean_inc(v_qsmul_596_);
lean_dec_ref(v_inst_593_);
v___x_597_ = lean_apply_2(v_qsmul_596_, v_a_594_, v_a_595_);
return v___x_597_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDivisionRing___redArg(lean_object* v_inst_598_){
_start:
{
lean_object* v_toRing_599_; lean_object* v_toNNRatCast_600_; lean_object* v_toRatCast_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_605_; lean_object* v___x_606_; lean_object* v_toInv_607_; lean_object* v___x_608_; lean_object* v_toDiv_609_; lean_object* v___x_610_; lean_object* v_toZPow_611_; lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; 
v_toRing_599_ = lean_ctor_get(v_inst_598_, 0);
v_toNNRatCast_600_ = lean_ctor_get(v_inst_598_, 4);
lean_inc(v_toNNRatCast_600_);
v_toRatCast_601_ = lean_ctor_get(v_inst_598_, 5);
lean_inc(v_toRatCast_601_);
lean_inc_ref(v_toRing_599_);
v___x_602_ = lp_mathlib_OrderDual_instRing___redArg(v_toRing_599_);
v___x_603_ = lp_mathlib_DivisionRing_toDivisionSemiring___redArg(v_inst_598_);
v___x_604_ = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(v___x_603_);
lean_dec_ref(v___x_603_);
v___x_605_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v___x_604_);
v___x_606_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v___x_605_);
lean_dec_ref(v___x_605_);
v_toInv_607_ = lean_ctor_get(v___x_606_, 1);
lean_inc(v_toInv_607_);
lean_dec_ref(v___x_606_);
v___x_608_ = lp_mathlib_DivisionRing_toDivInvMonoid___redArg(v_inst_598_);
v_toDiv_609_ = lean_ctor_get(v___x_608_, 2);
lean_inc(v_toDiv_609_);
v___x_610_ = lp_mathlib_OrderDual_instDivInvMonoid___redArg(v___x_608_);
v_toZPow_611_ = lean_ctor_get(v___x_610_, 3);
lean_inc(v_toZPow_611_);
lean_dec_ref(v___x_610_);
lean_inc_ref(v_inst_598_);
v___x_612_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instDivisionRing___aux__9), 4, 2);
lean_closure_set(v___x_612_, 0, lean_box(0));
lean_closure_set(v___x_612_, 1, v_inst_598_);
v___x_613_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instDivisionRing___aux__13), 4, 2);
lean_closure_set(v___x_613_, 0, lean_box(0));
lean_closure_set(v___x_613_, 1, v_inst_598_);
v___x_614_ = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(v___x_614_, 0, v___x_602_);
lean_ctor_set(v___x_614_, 1, v_toInv_607_);
lean_ctor_set(v___x_614_, 2, v_toDiv_609_);
lean_ctor_set(v___x_614_, 3, v_toZPow_611_);
lean_ctor_set(v___x_614_, 4, v_toNNRatCast_600_);
lean_ctor_set(v___x_614_, 5, v_toRatCast_601_);
lean_ctor_set(v___x_614_, 6, v___x_612_);
lean_ctor_set(v___x_614_, 7, v___x_613_);
return v___x_614_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instDivisionRing(lean_object* v_K_615_, lean_object* v_inst_616_){
_start:
{
lean_object* v___x_617_; 
v___x_617_ = lp_mathlib_OrderDual_instDivisionRing___redArg(v_inst_616_);
return v___x_617_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instSemifield___redArg(lean_object* v_inst_618_){
_start:
{
lean_object* v_toCommSemiring_619_; lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v_toInv_624_; lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v_toDiv_628_; lean_object* v___x_629_; lean_object* v_toZPow_630_; lean_object* v_toNNRatCast_631_; lean_object* v___x_632_; lean_object* v___x_633_; 
v_toCommSemiring_619_ = lean_ctor_get(v_inst_618_, 0);
lean_inc_ref(v_toCommSemiring_619_);
v___x_620_ = lp_mathlib_OrderDual_instSemiring___redArg(v_toCommSemiring_619_);
v___x_621_ = lp_mathlib_Semifield_toCommGroupWithZero___redArg(v_inst_618_);
v___x_622_ = lp_mathlib_CommGroupWithZero_toDivisionCommMonoid___redArg(v___x_621_);
v___x_623_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v___x_622_);
lean_dec_ref(v___x_622_);
v_toInv_624_ = lean_ctor_get(v___x_623_, 1);
lean_inc(v_toInv_624_);
lean_dec_ref(v___x_623_);
v___x_625_ = lp_mathlib_Semifield_toDivisionSemiring___redArg(v_inst_618_);
v___x_626_ = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(v___x_625_);
v___x_627_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v___x_626_);
v_toDiv_628_ = lean_ctor_get(v___x_627_, 2);
lean_inc(v_toDiv_628_);
v___x_629_ = lp_mathlib_OrderDual_instDivInvMonoid___redArg(v___x_627_);
v_toZPow_630_ = lean_ctor_get(v___x_629_, 3);
lean_inc(v_toZPow_630_);
lean_dec_ref(v___x_629_);
v_toNNRatCast_631_ = lean_ctor_get(v___x_625_, 4);
lean_inc(v_toNNRatCast_631_);
v___x_632_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instDivisionSemiring___aux__9), 4, 2);
lean_closure_set(v___x_632_, 0, lean_box(0));
lean_closure_set(v___x_632_, 1, v___x_625_);
v___x_633_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_633_, 0, v___x_620_);
lean_ctor_set(v___x_633_, 1, v_toInv_624_);
lean_ctor_set(v___x_633_, 2, v_toDiv_628_);
lean_ctor_set(v___x_633_, 3, v_toZPow_630_);
lean_ctor_set(v___x_633_, 4, v_toNNRatCast_631_);
lean_ctor_set(v___x_633_, 5, v___x_632_);
return v___x_633_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instSemifield(lean_object* v_K_634_, lean_object* v_inst_635_){
_start:
{
lean_object* v___x_636_; 
v___x_636_ = lp_mathlib_OrderDual_instSemifield___redArg(v_inst_635_);
return v___x_636_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instField___redArg(lean_object* v_inst_637_){
_start:
{
lean_object* v_toCommRing_638_; lean_object* v___x_639_; lean_object* v___x_640_; lean_object* v___x_641_; lean_object* v___x_642_; lean_object* v___x_643_; lean_object* v_toInv_644_; lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v_toDiv_647_; lean_object* v___x_648_; lean_object* v_toZPow_649_; lean_object* v_toNNRatCast_650_; lean_object* v_toRatCast_651_; lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; 
v_toCommRing_638_ = lean_ctor_get(v_inst_637_, 0);
lean_inc_ref(v_toCommRing_638_);
v___x_639_ = lp_mathlib_OrderDual_instRing___redArg(v_toCommRing_638_);
v___x_640_ = lp_mathlib_Field_toSemifield___redArg(v_inst_637_);
v___x_641_ = lp_mathlib_Semifield_toCommGroupWithZero___redArg(v___x_640_);
lean_dec_ref(v___x_640_);
v___x_642_ = lp_mathlib_CommGroupWithZero_toDivisionCommMonoid___redArg(v___x_641_);
v___x_643_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v___x_642_);
lean_dec_ref(v___x_642_);
v_toInv_644_ = lean_ctor_get(v___x_643_, 1);
lean_inc(v_toInv_644_);
lean_dec_ref(v___x_643_);
v___x_645_ = lp_mathlib_Field_toDivisionRing___redArg(v_inst_637_);
v___x_646_ = lp_mathlib_DivisionRing_toDivInvMonoid___redArg(v___x_645_);
v_toDiv_647_ = lean_ctor_get(v___x_646_, 2);
lean_inc(v_toDiv_647_);
v___x_648_ = lp_mathlib_OrderDual_instDivInvMonoid___redArg(v___x_646_);
v_toZPow_649_ = lean_ctor_get(v___x_648_, 3);
lean_inc(v_toZPow_649_);
lean_dec_ref(v___x_648_);
v_toNNRatCast_650_ = lean_ctor_get(v___x_645_, 4);
lean_inc(v_toNNRatCast_650_);
v_toRatCast_651_ = lean_ctor_get(v___x_645_, 5);
lean_inc(v_toRatCast_651_);
lean_inc_ref(v___x_645_);
v___x_652_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instDivisionRing___aux__9), 4, 2);
lean_closure_set(v___x_652_, 0, lean_box(0));
lean_closure_set(v___x_652_, 1, v___x_645_);
v___x_653_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instDivisionRing___aux__13), 4, 2);
lean_closure_set(v___x_653_, 0, lean_box(0));
lean_closure_set(v___x_653_, 1, v___x_645_);
v___x_654_ = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(v___x_654_, 0, v___x_639_);
lean_ctor_set(v___x_654_, 1, v_toInv_644_);
lean_ctor_set(v___x_654_, 2, v_toDiv_647_);
lean_ctor_set(v___x_654_, 3, v_toZPow_649_);
lean_ctor_set(v___x_654_, 4, v_toNNRatCast_650_);
lean_ctor_set(v___x_654_, 5, v_toRatCast_651_);
lean_ctor_set(v___x_654_, 6, v___x_652_);
lean_ctor_set(v___x_654_, 7, v___x_653_);
return v___x_654_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instField(lean_object* v_K_655_, lean_object* v_inst_656_){
_start:
{
lean_object* v___x_657_; 
v___x_657_ = lp_mathlib_OrderDual_instField___redArg(v_inst_656_);
return v___x_657_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instRatCast___aux__1___redArg(lean_object* v_inst_658_){
_start:
{
lean_inc(v_inst_658_);
return v_inst_658_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instRatCast___aux__1___redArg___boxed(lean_object* v_inst_659_){
_start:
{
lean_object* v_res_660_; 
v_res_660_ = lp_mathlib_Lex_instRatCast___aux__1___redArg(v_inst_659_);
lean_dec(v_inst_659_);
return v_res_660_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instRatCast___aux__1(lean_object* v_K_661_, lean_object* v_inst_662_){
_start:
{
lean_inc(v_inst_662_);
return v_inst_662_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instRatCast___aux__1___boxed(lean_object* v_K_663_, lean_object* v_inst_664_){
_start:
{
lean_object* v_res_665_; 
v_res_665_ = lp_mathlib_Lex_instRatCast___aux__1(v_K_663_, v_inst_664_);
lean_dec(v_inst_664_);
return v_res_665_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instRatCast___redArg(lean_object* v_inst_666_){
_start:
{
lean_inc(v_inst_666_);
return v_inst_666_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instRatCast___redArg___boxed(lean_object* v_inst_667_){
_start:
{
lean_object* v_res_668_; 
v_res_668_ = lp_mathlib_Lex_instRatCast___redArg(v_inst_667_);
lean_dec(v_inst_667_);
return v_res_668_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instRatCast(lean_object* v_K_669_, lean_object* v_inst_670_){
_start:
{
lean_inc(v_inst_670_);
return v_inst_670_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instRatCast___boxed(lean_object* v_K_671_, lean_object* v_inst_672_){
_start:
{
lean_object* v_res_673_; 
v_res_673_ = lp_mathlib_Lex_instRatCast(v_K_671_, v_inst_672_);
lean_dec(v_inst_672_);
return v_res_673_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instDivisionSemiring___aux__8___redArg(lean_object* v_inst_674_){
_start:
{
lean_object* v_toNNRatCast_675_; 
v_toNNRatCast_675_ = lean_ctor_get(v_inst_674_, 4);
lean_inc(v_toNNRatCast_675_);
return v_toNNRatCast_675_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instDivisionSemiring___aux__8___redArg___boxed(lean_object* v_inst_676_){
_start:
{
lean_object* v_res_677_; 
v_res_677_ = lp_mathlib_Lex_instDivisionSemiring___aux__8___redArg(v_inst_676_);
lean_dec_ref(v_inst_676_);
return v_res_677_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instDivisionSemiring___aux__8(lean_object* v_K_678_, lean_object* v_inst_679_){
_start:
{
lean_object* v_toNNRatCast_680_; 
v_toNNRatCast_680_ = lean_ctor_get(v_inst_679_, 4);
lean_inc(v_toNNRatCast_680_);
return v_toNNRatCast_680_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instDivisionSemiring___aux__8___boxed(lean_object* v_K_681_, lean_object* v_inst_682_){
_start:
{
lean_object* v_res_683_; 
v_res_683_ = lp_mathlib_Lex_instDivisionSemiring___aux__8(v_K_681_, v_inst_682_);
lean_dec_ref(v_inst_682_);
return v_res_683_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instDivisionSemiring___aux__11___redArg(lean_object* v_inst_684_, lean_object* v_a_685_, lean_object* v_a_686_){
_start:
{
lean_object* v_nnqsmul_687_; lean_object* v___x_688_; 
v_nnqsmul_687_ = lean_ctor_get(v_inst_684_, 5);
lean_inc(v_nnqsmul_687_);
lean_dec_ref(v_inst_684_);
v___x_688_ = lean_apply_2(v_nnqsmul_687_, v_a_685_, v_a_686_);
return v___x_688_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instDivisionSemiring___aux__11(lean_object* v_K_689_, lean_object* v_inst_690_, lean_object* v_a_691_, lean_object* v_a_692_){
_start:
{
lean_object* v_nnqsmul_693_; lean_object* v___x_694_; 
v_nnqsmul_693_ = lean_ctor_get(v_inst_690_, 5);
lean_inc(v_nnqsmul_693_);
lean_dec_ref(v_inst_690_);
v___x_694_ = lean_apply_2(v_nnqsmul_693_, v_a_691_, v_a_692_);
return v___x_694_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instDivisionSemiring___redArg(lean_object* v_inst_695_){
_start:
{
lean_object* v_toSemiring_696_; lean_object* v_toNNRatCast_697_; lean_object* v___x_698_; lean_object* v___x_699_; lean_object* v___x_700_; lean_object* v___x_701_; lean_object* v_toInv_702_; lean_object* v_toDiv_703_; lean_object* v___x_704_; lean_object* v_toZPow_705_; lean_object* v___x_706_; lean_object* v___x_707_; 
v_toSemiring_696_ = lean_ctor_get(v_inst_695_, 0);
v_toNNRatCast_697_ = lean_ctor_get(v_inst_695_, 4);
lean_inc(v_toNNRatCast_697_);
lean_inc_ref(v_toSemiring_696_);
v___x_698_ = lp_mathlib_Lex_instSemiring___redArg(v_toSemiring_696_);
v___x_699_ = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(v_inst_695_);
v___x_700_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v___x_699_);
v___x_701_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v___x_700_);
v_toInv_702_ = lean_ctor_get(v___x_701_, 1);
lean_inc(v_toInv_702_);
lean_dec_ref(v___x_701_);
v_toDiv_703_ = lean_ctor_get(v___x_700_, 2);
lean_inc(v_toDiv_703_);
v___x_704_ = lp_mathlib_Lex_instDivInvMonoid___redArg(v___x_700_);
v_toZPow_705_ = lean_ctor_get(v___x_704_, 3);
lean_inc(v_toZPow_705_);
lean_dec_ref(v___x_704_);
v___x_706_ = lean_alloc_closure((void*)(lp_mathlib_Lex_instDivisionSemiring___aux__11), 4, 2);
lean_closure_set(v___x_706_, 0, lean_box(0));
lean_closure_set(v___x_706_, 1, v_inst_695_);
v___x_707_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_707_, 0, v___x_698_);
lean_ctor_set(v___x_707_, 1, v_toInv_702_);
lean_ctor_set(v___x_707_, 2, v_toDiv_703_);
lean_ctor_set(v___x_707_, 3, v_toZPow_705_);
lean_ctor_set(v___x_707_, 4, v_toNNRatCast_697_);
lean_ctor_set(v___x_707_, 5, v___x_706_);
return v___x_707_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instDivisionSemiring(lean_object* v_K_708_, lean_object* v_inst_709_){
_start:
{
lean_object* v___x_710_; 
v___x_710_ = lp_mathlib_Lex_instDivisionSemiring___redArg(v_inst_709_);
return v___x_710_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instDivisionRing___aux__9___redArg(lean_object* v_inst_711_, lean_object* v_a_712_, lean_object* v_a_713_){
_start:
{
lean_object* v_nnqsmul_714_; lean_object* v___x_715_; 
v_nnqsmul_714_ = lean_ctor_get(v_inst_711_, 6);
lean_inc(v_nnqsmul_714_);
lean_dec_ref(v_inst_711_);
v___x_715_ = lean_apply_2(v_nnqsmul_714_, v_a_712_, v_a_713_);
return v___x_715_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instDivisionRing___aux__9(lean_object* v_K_716_, lean_object* v_inst_717_, lean_object* v_a_718_, lean_object* v_a_719_){
_start:
{
lean_object* v_nnqsmul_720_; lean_object* v___x_721_; 
v_nnqsmul_720_ = lean_ctor_get(v_inst_717_, 6);
lean_inc(v_nnqsmul_720_);
lean_dec_ref(v_inst_717_);
v___x_721_ = lean_apply_2(v_nnqsmul_720_, v_a_718_, v_a_719_);
return v___x_721_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instDivisionRing___aux__13___redArg(lean_object* v_inst_722_, lean_object* v_a_723_, lean_object* v_a_724_){
_start:
{
lean_object* v_qsmul_725_; lean_object* v___x_726_; 
v_qsmul_725_ = lean_ctor_get(v_inst_722_, 7);
lean_inc(v_qsmul_725_);
lean_dec_ref(v_inst_722_);
v___x_726_ = lean_apply_2(v_qsmul_725_, v_a_723_, v_a_724_);
return v___x_726_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instDivisionRing___aux__13(lean_object* v_K_727_, lean_object* v_inst_728_, lean_object* v_a_729_, lean_object* v_a_730_){
_start:
{
lean_object* v_qsmul_731_; lean_object* v___x_732_; 
v_qsmul_731_ = lean_ctor_get(v_inst_728_, 7);
lean_inc(v_qsmul_731_);
lean_dec_ref(v_inst_728_);
v___x_732_ = lean_apply_2(v_qsmul_731_, v_a_729_, v_a_730_);
return v___x_732_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instDivisionRing___redArg(lean_object* v_inst_733_){
_start:
{
lean_object* v_toRing_734_; lean_object* v_toRatCast_735_; lean_object* v___x_736_; lean_object* v___x_737_; lean_object* v___x_738_; lean_object* v___x_739_; lean_object* v___x_740_; lean_object* v_toInv_741_; lean_object* v___x_742_; lean_object* v_toDiv_743_; lean_object* v___x_744_; lean_object* v_toZPow_745_; lean_object* v___x_746_; lean_object* v_toNNRatCast_747_; lean_object* v___x_748_; lean_object* v___x_749_; lean_object* v___x_750_; 
v_toRing_734_ = lean_ctor_get(v_inst_733_, 0);
v_toRatCast_735_ = lean_ctor_get(v_inst_733_, 5);
lean_inc(v_toRatCast_735_);
lean_inc_ref(v_toRing_734_);
v___x_736_ = lp_mathlib_Lex_instRing___redArg(v_toRing_734_);
v___x_737_ = lp_mathlib_DivisionRing_toDivisionSemiring___redArg(v_inst_733_);
v___x_738_ = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(v___x_737_);
v___x_739_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v___x_738_);
v___x_740_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v___x_739_);
lean_dec_ref(v___x_739_);
v_toInv_741_ = lean_ctor_get(v___x_740_, 1);
lean_inc(v_toInv_741_);
lean_dec_ref(v___x_740_);
v___x_742_ = lp_mathlib_DivisionRing_toDivInvMonoid___redArg(v_inst_733_);
v_toDiv_743_ = lean_ctor_get(v___x_742_, 2);
lean_inc(v_toDiv_743_);
v___x_744_ = lp_mathlib_Lex_instDivInvMonoid___redArg(v___x_742_);
v_toZPow_745_ = lean_ctor_get(v___x_744_, 3);
lean_inc(v_toZPow_745_);
lean_dec_ref(v___x_744_);
v___x_746_ = lp_mathlib_Lex_instDivisionSemiring___redArg(v___x_737_);
v_toNNRatCast_747_ = lean_ctor_get(v___x_746_, 4);
lean_inc(v_toNNRatCast_747_);
lean_dec_ref(v___x_746_);
lean_inc_ref(v_inst_733_);
v___x_748_ = lean_alloc_closure((void*)(lp_mathlib_Lex_instDivisionRing___aux__9), 4, 2);
lean_closure_set(v___x_748_, 0, lean_box(0));
lean_closure_set(v___x_748_, 1, v_inst_733_);
v___x_749_ = lean_alloc_closure((void*)(lp_mathlib_Lex_instDivisionRing___aux__13), 4, 2);
lean_closure_set(v___x_749_, 0, lean_box(0));
lean_closure_set(v___x_749_, 1, v_inst_733_);
v___x_750_ = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(v___x_750_, 0, v___x_736_);
lean_ctor_set(v___x_750_, 1, v_toInv_741_);
lean_ctor_set(v___x_750_, 2, v_toDiv_743_);
lean_ctor_set(v___x_750_, 3, v_toZPow_745_);
lean_ctor_set(v___x_750_, 4, v_toNNRatCast_747_);
lean_ctor_set(v___x_750_, 5, v_toRatCast_735_);
lean_ctor_set(v___x_750_, 6, v___x_748_);
lean_ctor_set(v___x_750_, 7, v___x_749_);
return v___x_750_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instDivisionRing(lean_object* v_K_751_, lean_object* v_inst_752_){
_start:
{
lean_object* v___x_753_; 
v___x_753_ = lp_mathlib_Lex_instDivisionRing___redArg(v_inst_752_);
return v___x_753_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instSemifield___redArg(lean_object* v_inst_754_){
_start:
{
lean_object* v_toCommSemiring_755_; lean_object* v___x_756_; lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v_toInv_760_; lean_object* v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v_toDiv_764_; lean_object* v___x_765_; lean_object* v_toZPow_766_; lean_object* v___x_767_; lean_object* v_toNNRatCast_768_; lean_object* v___x_770_; uint8_t v_isShared_771_; uint8_t v_isSharedCheck_776_; 
v_toCommSemiring_755_ = lean_ctor_get(v_inst_754_, 0);
lean_inc_ref(v_toCommSemiring_755_);
v___x_756_ = lp_mathlib_Lex_instSemiring___redArg(v_toCommSemiring_755_);
v___x_757_ = lp_mathlib_Semifield_toCommGroupWithZero___redArg(v_inst_754_);
v___x_758_ = lp_mathlib_CommGroupWithZero_toDivisionCommMonoid___redArg(v___x_757_);
v___x_759_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v___x_758_);
lean_dec_ref(v___x_758_);
v_toInv_760_ = lean_ctor_get(v___x_759_, 1);
lean_inc(v_toInv_760_);
lean_dec_ref(v___x_759_);
v___x_761_ = lp_mathlib_Semifield_toDivisionSemiring___redArg(v_inst_754_);
v___x_762_ = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(v___x_761_);
v___x_763_ = lp_mathlib_GroupWithZero_toDivInvMonoid___redArg(v___x_762_);
v_toDiv_764_ = lean_ctor_get(v___x_763_, 2);
lean_inc(v_toDiv_764_);
v___x_765_ = lp_mathlib_Lex_instDivInvMonoid___redArg(v___x_763_);
v_toZPow_766_ = lean_ctor_get(v___x_765_, 3);
lean_inc(v_toZPow_766_);
lean_dec_ref(v___x_765_);
lean_inc_ref(v___x_761_);
v___x_767_ = lp_mathlib_Lex_instDivisionSemiring___redArg(v___x_761_);
v_toNNRatCast_768_ = lean_ctor_get(v___x_767_, 4);
v_isSharedCheck_776_ = !lean_is_exclusive(v___x_767_);
if (v_isSharedCheck_776_ == 0)
{
lean_object* v_unused_777_; lean_object* v_unused_778_; lean_object* v_unused_779_; lean_object* v_unused_780_; lean_object* v_unused_781_; 
v_unused_777_ = lean_ctor_get(v___x_767_, 5);
lean_dec(v_unused_777_);
v_unused_778_ = lean_ctor_get(v___x_767_, 3);
lean_dec(v_unused_778_);
v_unused_779_ = lean_ctor_get(v___x_767_, 2);
lean_dec(v_unused_779_);
v_unused_780_ = lean_ctor_get(v___x_767_, 1);
lean_dec(v_unused_780_);
v_unused_781_ = lean_ctor_get(v___x_767_, 0);
lean_dec(v_unused_781_);
v___x_770_ = v___x_767_;
v_isShared_771_ = v_isSharedCheck_776_;
goto v_resetjp_769_;
}
else
{
lean_inc(v_toNNRatCast_768_);
lean_dec(v___x_767_);
v___x_770_ = lean_box(0);
v_isShared_771_ = v_isSharedCheck_776_;
goto v_resetjp_769_;
}
v_resetjp_769_:
{
lean_object* v___x_772_; lean_object* v___x_774_; 
v___x_772_ = lean_alloc_closure((void*)(lp_mathlib_Lex_instDivisionSemiring___aux__11), 4, 2);
lean_closure_set(v___x_772_, 0, lean_box(0));
lean_closure_set(v___x_772_, 1, v___x_761_);
if (v_isShared_771_ == 0)
{
lean_ctor_set(v___x_770_, 5, v___x_772_);
lean_ctor_set(v___x_770_, 3, v_toZPow_766_);
lean_ctor_set(v___x_770_, 2, v_toDiv_764_);
lean_ctor_set(v___x_770_, 1, v_toInv_760_);
lean_ctor_set(v___x_770_, 0, v___x_756_);
v___x_774_ = v___x_770_;
goto v_reusejp_773_;
}
else
{
lean_object* v_reuseFailAlloc_775_; 
v_reuseFailAlloc_775_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_775_, 0, v___x_756_);
lean_ctor_set(v_reuseFailAlloc_775_, 1, v_toInv_760_);
lean_ctor_set(v_reuseFailAlloc_775_, 2, v_toDiv_764_);
lean_ctor_set(v_reuseFailAlloc_775_, 3, v_toZPow_766_);
lean_ctor_set(v_reuseFailAlloc_775_, 4, v_toNNRatCast_768_);
lean_ctor_set(v_reuseFailAlloc_775_, 5, v___x_772_);
v___x_774_ = v_reuseFailAlloc_775_;
goto v_reusejp_773_;
}
v_reusejp_773_:
{
return v___x_774_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instSemifield(lean_object* v_K_782_, lean_object* v_inst_783_){
_start:
{
lean_object* v___x_784_; 
v___x_784_ = lp_mathlib_Lex_instSemifield___redArg(v_inst_783_);
return v___x_784_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instField___redArg(lean_object* v_inst_785_){
_start:
{
lean_object* v_toCommRing_786_; lean_object* v___x_787_; lean_object* v___x_788_; lean_object* v___x_789_; lean_object* v___x_790_; lean_object* v___x_791_; lean_object* v_toInv_792_; lean_object* v___x_793_; lean_object* v___x_794_; lean_object* v_toDiv_795_; lean_object* v___x_796_; lean_object* v_toZPow_797_; lean_object* v___x_798_; lean_object* v_toNNRatCast_799_; lean_object* v___x_801_; uint8_t v_isShared_802_; uint8_t v_isSharedCheck_809_; 
v_toCommRing_786_ = lean_ctor_get(v_inst_785_, 0);
lean_inc_ref(v_toCommRing_786_);
v___x_787_ = lp_mathlib_Lex_instRing___redArg(v_toCommRing_786_);
v___x_788_ = lp_mathlib_Field_toSemifield___redArg(v_inst_785_);
v___x_789_ = lp_mathlib_Semifield_toCommGroupWithZero___redArg(v___x_788_);
lean_dec_ref(v___x_788_);
v___x_790_ = lp_mathlib_CommGroupWithZero_toDivisionCommMonoid___redArg(v___x_789_);
v___x_791_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v___x_790_);
lean_dec_ref(v___x_790_);
v_toInv_792_ = lean_ctor_get(v___x_791_, 1);
lean_inc(v_toInv_792_);
lean_dec_ref(v___x_791_);
v___x_793_ = lp_mathlib_Field_toDivisionRing___redArg(v_inst_785_);
v___x_794_ = lp_mathlib_DivisionRing_toDivInvMonoid___redArg(v___x_793_);
v_toDiv_795_ = lean_ctor_get(v___x_794_, 2);
lean_inc(v_toDiv_795_);
v___x_796_ = lp_mathlib_Lex_instDivInvMonoid___redArg(v___x_794_);
v_toZPow_797_ = lean_ctor_get(v___x_796_, 3);
lean_inc(v_toZPow_797_);
lean_dec_ref(v___x_796_);
lean_inc_ref(v___x_793_);
v___x_798_ = lp_mathlib_Lex_instDivisionRing___redArg(v___x_793_);
v_toNNRatCast_799_ = lean_ctor_get(v___x_798_, 4);
v_isSharedCheck_809_ = !lean_is_exclusive(v___x_798_);
if (v_isSharedCheck_809_ == 0)
{
lean_object* v_unused_810_; lean_object* v_unused_811_; lean_object* v_unused_812_; lean_object* v_unused_813_; lean_object* v_unused_814_; lean_object* v_unused_815_; lean_object* v_unused_816_; 
v_unused_810_ = lean_ctor_get(v___x_798_, 7);
lean_dec(v_unused_810_);
v_unused_811_ = lean_ctor_get(v___x_798_, 6);
lean_dec(v_unused_811_);
v_unused_812_ = lean_ctor_get(v___x_798_, 5);
lean_dec(v_unused_812_);
v_unused_813_ = lean_ctor_get(v___x_798_, 3);
lean_dec(v_unused_813_);
v_unused_814_ = lean_ctor_get(v___x_798_, 2);
lean_dec(v_unused_814_);
v_unused_815_ = lean_ctor_get(v___x_798_, 1);
lean_dec(v_unused_815_);
v_unused_816_ = lean_ctor_get(v___x_798_, 0);
lean_dec(v_unused_816_);
v___x_801_ = v___x_798_;
v_isShared_802_ = v_isSharedCheck_809_;
goto v_resetjp_800_;
}
else
{
lean_inc(v_toNNRatCast_799_);
lean_dec(v___x_798_);
v___x_801_ = lean_box(0);
v_isShared_802_ = v_isSharedCheck_809_;
goto v_resetjp_800_;
}
v_resetjp_800_:
{
lean_object* v_toRatCast_803_; lean_object* v___x_804_; lean_object* v___x_805_; lean_object* v___x_807_; 
v_toRatCast_803_ = lean_ctor_get(v___x_793_, 5);
lean_inc(v_toRatCast_803_);
lean_inc_ref(v___x_793_);
v___x_804_ = lean_alloc_closure((void*)(lp_mathlib_Lex_instDivisionRing___aux__9), 4, 2);
lean_closure_set(v___x_804_, 0, lean_box(0));
lean_closure_set(v___x_804_, 1, v___x_793_);
v___x_805_ = lean_alloc_closure((void*)(lp_mathlib_Lex_instDivisionRing___aux__13), 4, 2);
lean_closure_set(v___x_805_, 0, lean_box(0));
lean_closure_set(v___x_805_, 1, v___x_793_);
if (v_isShared_802_ == 0)
{
lean_ctor_set(v___x_801_, 7, v___x_805_);
lean_ctor_set(v___x_801_, 6, v___x_804_);
lean_ctor_set(v___x_801_, 5, v_toRatCast_803_);
lean_ctor_set(v___x_801_, 3, v_toZPow_797_);
lean_ctor_set(v___x_801_, 2, v_toDiv_795_);
lean_ctor_set(v___x_801_, 1, v_toInv_792_);
lean_ctor_set(v___x_801_, 0, v___x_787_);
v___x_807_ = v___x_801_;
goto v_reusejp_806_;
}
else
{
lean_object* v_reuseFailAlloc_808_; 
v_reuseFailAlloc_808_ = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(v_reuseFailAlloc_808_, 0, v___x_787_);
lean_ctor_set(v_reuseFailAlloc_808_, 1, v_toInv_792_);
lean_ctor_set(v_reuseFailAlloc_808_, 2, v_toDiv_795_);
lean_ctor_set(v_reuseFailAlloc_808_, 3, v_toZPow_797_);
lean_ctor_set(v_reuseFailAlloc_808_, 4, v_toNNRatCast_799_);
lean_ctor_set(v_reuseFailAlloc_808_, 5, v_toRatCast_803_);
lean_ctor_set(v_reuseFailAlloc_808_, 6, v___x_804_);
lean_ctor_set(v_reuseFailAlloc_808_, 7, v___x_805_);
v___x_807_ = v_reuseFailAlloc_808_;
goto v_reusejp_806_;
}
v_reusejp_806_:
{
return v___x_807_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lex_instField(lean_object* v_K_817_, lean_object* v_inst_818_){
_start:
{
lean_object* v___x_819_; 
v___x_819_ = lp_mathlib_Lex_instField___redArg(v_inst_818_);
return v___x_819_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_SelfInv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_GrindInstances(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Commute(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Invertible(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_OrderDual(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Lex(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Synonym(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Tauto(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Field_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_SelfInv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_GrindInstances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Commute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Invertible(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_OrderDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Lex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Synonym(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Tauto(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Field_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_SelfInv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_GrindInstances(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Commute(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Invertible(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_OrderDual(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Lex(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Synonym(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Tauto(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Field_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_SelfInv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_GrindInstances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Commute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Invertible(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_OrderDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Lex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Synonym(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Tauto(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Field_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Field_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Field_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
