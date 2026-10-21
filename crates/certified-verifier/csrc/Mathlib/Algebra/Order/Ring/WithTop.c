// Lean compiler output
// Module: Mathlib.Algebra.Order.Ring.WithTop
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.Ring.Canonical public import Mathlib.Algebra.Ring.Hom.Defs public import Mathlib.Algebra.Order.Monoid.WithTop
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
lean_object* lp_mathlib_WithTop_map(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lp_mathlib_WithBot_map(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithBot_addMonoid___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lp_mathlib_WithTop_addMonoid___redArg(lean_object*);
lean_object* lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* lp_mathlib_WithBot_addMonoidWithOne___redArg(lean_object*);
lean_object* lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toMonoidWithZero___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_SemigroupWithZero_toMulZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_WithTop_addMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instMulZeroClass___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instMulZeroClass___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instMulZeroClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instMulZeroClass(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instMulZeroClass_match__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instMulZeroClass_match__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instMulZeroClass___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instMulZeroClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instMulZeroClass(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instMulZeroOneClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instMulZeroOneClass(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instMulZeroOneClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instMulZeroOneClass(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_withTopMap___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_withTopMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_withTopMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_withTopMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_withBotMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_withBotMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_withBotMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instSemigroupWithZero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instSemigroupWithZero(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instSemigroupWithZero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instSemigroupWithZero(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Ring_WithTop_0__WithTop_instMonoidWithZero_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Ring_WithTop_0__WithTop_instMonoidWithZero_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instMonoidWithZero___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instMonoidWithZero___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instMonoidWithZero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instMonoidWithZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instMonoidWithZero_match__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instMonoidWithZero_match__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instMonoidWithZero___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instMonoidWithZero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instMonoidWithZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instCommMonoidWithZero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instCommMonoidWithZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instCommMonoidWithZero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instCommMonoidWithZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instNonUnitalNonAssocSemiring___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instNonUnitalNonAssocSemiring(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instNonUnitalNonAssocSemiring___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instNonUnitalNonAssocSemiring(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instNonAssocSemiring___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instNonAssocSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instNonAssocSemiring___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instNonAssocSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instNonUnitalSemiring___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instNonUnitalSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instNonUnitalSemiring___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instNonUnitalSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instSemiring___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instSemiring___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instCommSemiring___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instCommSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instCommSemiring___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instCommSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_withTopMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_withTopMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_withTopMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_withBotMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_withBotMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_withBotMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instMulZeroClass___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_toZero_2_, lean_object* v___x_3_, lean_object* v_toMul_4_, lean_object* v_x_5_, lean_object* v_x_6_){
_start:
{
lean_object* v_a_8_; 
if (lean_obj_tag(v_x_5_) == 0)
{
lean_dec(v_toMul_4_);
if (lean_obj_tag(v_x_6_) == 0)
{
lean_dec(v_toZero_2_);
lean_dec_ref(v_inst_1_);
lean_inc(v___x_3_);
return v___x_3_;
}
else
{
lean_object* v_val_12_; 
v_val_12_ = lean_ctor_get(v_x_6_, 0);
lean_inc(v_val_12_);
lean_dec_ref_known(v_x_6_, 1);
v_a_8_ = v_val_12_;
goto v___jp_7_;
}
}
else
{
if (lean_obj_tag(v_x_6_) == 0)
{
lean_object* v_val_13_; 
lean_dec(v_toMul_4_);
v_val_13_ = lean_ctor_get(v_x_5_, 0);
lean_inc(v_val_13_);
lean_dec_ref_known(v_x_5_, 1);
v_a_8_ = v_val_13_;
goto v___jp_7_;
}
else
{
lean_object* v_val_14_; lean_object* v_val_15_; lean_object* v___x_17_; uint8_t v_isShared_18_; uint8_t v_isSharedCheck_23_; 
lean_dec(v_toZero_2_);
lean_dec_ref(v_inst_1_);
v_val_14_ = lean_ctor_get(v_x_5_, 0);
lean_inc(v_val_14_);
lean_dec_ref_known(v_x_5_, 1);
v_val_15_ = lean_ctor_get(v_x_6_, 0);
v_isSharedCheck_23_ = !lean_is_exclusive(v_x_6_);
if (v_isSharedCheck_23_ == 0)
{
v___x_17_ = v_x_6_;
v_isShared_18_ = v_isSharedCheck_23_;
goto v_resetjp_16_;
}
else
{
lean_inc(v_val_15_);
lean_dec(v_x_6_);
v___x_17_ = lean_box(0);
v_isShared_18_ = v_isSharedCheck_23_;
goto v_resetjp_16_;
}
v_resetjp_16_:
{
lean_object* v___x_19_; lean_object* v___x_21_; 
v___x_19_ = lean_apply_2(v_toMul_4_, v_val_14_, v_val_15_);
if (v_isShared_18_ == 0)
{
lean_ctor_set(v___x_17_, 0, v___x_19_);
v___x_21_ = v___x_17_;
goto v_reusejp_20_;
}
else
{
lean_object* v_reuseFailAlloc_22_; 
v_reuseFailAlloc_22_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_22_, 0, v___x_19_);
v___x_21_ = v_reuseFailAlloc_22_;
goto v_reusejp_20_;
}
v_reusejp_20_:
{
return v___x_21_;
}
}
}
}
v___jp_7_:
{
lean_object* v___x_9_; uint8_t v___x_10_; 
lean_inc(v_toZero_2_);
v___x_9_ = lean_apply_2(v_inst_1_, v_a_8_, v_toZero_2_);
v___x_10_ = lean_unbox(v___x_9_);
if (v___x_10_ == 0)
{
lean_dec(v_toZero_2_);
lean_inc(v___x_3_);
return v___x_3_;
}
else
{
lean_object* v___x_11_; 
v___x_11_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_11_, 0, v_toZero_2_);
return v___x_11_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instMulZeroClass___redArg___lam__0___boxed(lean_object* v_inst_24_, lean_object* v_toZero_25_, lean_object* v___x_26_, lean_object* v_toMul_27_, lean_object* v_x_28_, lean_object* v_x_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_mathlib_WithTop_instMulZeroClass___redArg___lam__0(v_inst_24_, v_toZero_25_, v___x_26_, v_toMul_27_, v_x_28_, v_x_29_);
lean_dec(v___x_26_);
return v_res_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instMulZeroClass___redArg(lean_object* v_inst_31_, lean_object* v_inst_32_){
_start:
{
lean_object* v_toMul_33_; lean_object* v_toZero_34_; lean_object* v___x_36_; uint8_t v_isShared_37_; uint8_t v_isSharedCheck_44_; 
v_toMul_33_ = lean_ctor_get(v_inst_32_, 0);
v_toZero_34_ = lean_ctor_get(v_inst_32_, 1);
v_isSharedCheck_44_ = !lean_is_exclusive(v_inst_32_);
if (v_isSharedCheck_44_ == 0)
{
v___x_36_ = v_inst_32_;
v_isShared_37_ = v_isSharedCheck_44_;
goto v_resetjp_35_;
}
else
{
lean_inc(v_toZero_34_);
lean_inc(v_toMul_33_);
lean_dec(v_inst_32_);
v___x_36_ = lean_box(0);
v_isShared_37_ = v_isSharedCheck_44_;
goto v_resetjp_35_;
}
v_resetjp_35_:
{
lean_object* v___x_38_; lean_object* v___f_39_; lean_object* v___x_40_; lean_object* v___x_42_; 
v___x_38_ = lean_box(0);
lean_inc(v_toZero_34_);
v___f_39_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_instMulZeroClass___redArg___lam__0___boxed), 6, 4);
lean_closure_set(v___f_39_, 0, v_inst_31_);
lean_closure_set(v___f_39_, 1, v_toZero_34_);
lean_closure_set(v___f_39_, 2, v___x_38_);
lean_closure_set(v___f_39_, 3, v_toMul_33_);
v___x_40_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_40_, 0, v_toZero_34_);
if (v_isShared_37_ == 0)
{
lean_ctor_set(v___x_36_, 1, v___x_40_);
lean_ctor_set(v___x_36_, 0, v___f_39_);
v___x_42_ = v___x_36_;
goto v_reusejp_41_;
}
else
{
lean_object* v_reuseFailAlloc_43_; 
v_reuseFailAlloc_43_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_43_, 0, v___f_39_);
lean_ctor_set(v_reuseFailAlloc_43_, 1, v___x_40_);
v___x_42_ = v_reuseFailAlloc_43_;
goto v_reusejp_41_;
}
v_reusejp_41_:
{
return v___x_42_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instMulZeroClass(lean_object* v_00_u03b1_45_, lean_object* v_inst_46_, lean_object* v_inst_47_){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lp_mathlib_WithTop_instMulZeroClass___redArg(v_inst_46_, v_inst_47_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instMulZeroClass_match__1___redArg(lean_object* v_x_49_, lean_object* v_x_50_, lean_object* v_h__1_51_, lean_object* v_h__2_52_, lean_object* v_h__3_53_, lean_object* v_h__4_54_){
_start:
{
if (lean_obj_tag(v_x_49_) == 0)
{
lean_dec(v_h__2_52_);
lean_dec(v_h__1_51_);
if (lean_obj_tag(v_x_50_) == 0)
{
lean_object* v___x_55_; lean_object* v___x_56_; 
lean_dec(v_h__3_53_);
v___x_55_ = lean_box(0);
v___x_56_ = lean_apply_1(v_h__4_54_, v___x_55_);
return v___x_56_;
}
else
{
lean_object* v_val_57_; lean_object* v___x_58_; 
lean_dec(v_h__4_54_);
v_val_57_ = lean_ctor_get(v_x_50_, 0);
lean_inc(v_val_57_);
lean_dec_ref_known(v_x_50_, 1);
v___x_58_ = lean_apply_1(v_h__3_53_, v_val_57_);
return v___x_58_;
}
}
else
{
lean_dec(v_h__4_54_);
lean_dec(v_h__3_53_);
if (lean_obj_tag(v_x_50_) == 0)
{
lean_object* v_val_59_; lean_object* v___x_60_; 
lean_dec(v_h__1_51_);
v_val_59_ = lean_ctor_get(v_x_49_, 0);
lean_inc(v_val_59_);
lean_dec_ref_known(v_x_49_, 1);
v___x_60_ = lean_apply_1(v_h__2_52_, v_val_59_);
return v___x_60_;
}
else
{
lean_object* v_val_61_; lean_object* v_val_62_; lean_object* v___x_63_; 
lean_dec(v_h__2_52_);
v_val_61_ = lean_ctor_get(v_x_49_, 0);
lean_inc(v_val_61_);
lean_dec_ref_known(v_x_49_, 1);
v_val_62_ = lean_ctor_get(v_x_50_, 0);
lean_inc(v_val_62_);
lean_dec_ref_known(v_x_50_, 1);
v___x_63_ = lean_apply_2(v_h__1_51_, v_val_61_, v_val_62_);
return v___x_63_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instMulZeroClass_match__1(lean_object* v_00_u03b1_64_, lean_object* v_motive_65_, lean_object* v_x_66_, lean_object* v_x_67_, lean_object* v_h__1_68_, lean_object* v_h__2_69_, lean_object* v_h__3_70_, lean_object* v_h__4_71_){
_start:
{
lean_object* v___x_72_; 
v___x_72_ = lp_mathlib_WithBot_instMulZeroClass_match__1___redArg(v_x_66_, v_x_67_, v_h__1_68_, v_h__2_69_, v_h__3_70_, v_h__4_71_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instMulZeroClass___redArg___lam__0(lean_object* v_inst_73_, lean_object* v_toZero_74_, lean_object* v_toMul_75_, lean_object* v_x_76_, lean_object* v_x_77_){
_start:
{
lean_object* v_a_79_; 
if (lean_obj_tag(v_x_76_) == 0)
{
lean_dec(v_toMul_75_);
if (lean_obj_tag(v_x_77_) == 0)
{
lean_dec(v_toZero_74_);
lean_dec_ref(v_inst_73_);
return v_x_77_;
}
else
{
lean_object* v_val_84_; 
v_val_84_ = lean_ctor_get(v_x_77_, 0);
lean_inc(v_val_84_);
lean_dec_ref_known(v_x_77_, 1);
v_a_79_ = v_val_84_;
goto v___jp_78_;
}
}
else
{
if (lean_obj_tag(v_x_77_) == 0)
{
lean_object* v_val_85_; 
lean_dec(v_toMul_75_);
v_val_85_ = lean_ctor_get(v_x_76_, 0);
lean_inc(v_val_85_);
lean_dec_ref_known(v_x_76_, 1);
v_a_79_ = v_val_85_;
goto v___jp_78_;
}
else
{
lean_object* v_val_86_; lean_object* v_val_87_; lean_object* v___x_89_; uint8_t v_isShared_90_; uint8_t v_isSharedCheck_95_; 
lean_dec(v_toZero_74_);
lean_dec_ref(v_inst_73_);
v_val_86_ = lean_ctor_get(v_x_76_, 0);
lean_inc(v_val_86_);
lean_dec_ref_known(v_x_76_, 1);
v_val_87_ = lean_ctor_get(v_x_77_, 0);
v_isSharedCheck_95_ = !lean_is_exclusive(v_x_77_);
if (v_isSharedCheck_95_ == 0)
{
v___x_89_ = v_x_77_;
v_isShared_90_ = v_isSharedCheck_95_;
goto v_resetjp_88_;
}
else
{
lean_inc(v_val_87_);
lean_dec(v_x_77_);
v___x_89_ = lean_box(0);
v_isShared_90_ = v_isSharedCheck_95_;
goto v_resetjp_88_;
}
v_resetjp_88_:
{
lean_object* v___x_91_; lean_object* v___x_93_; 
v___x_91_ = lean_apply_2(v_toMul_75_, v_val_86_, v_val_87_);
if (v_isShared_90_ == 0)
{
lean_ctor_set(v___x_89_, 0, v___x_91_);
v___x_93_ = v___x_89_;
goto v_reusejp_92_;
}
else
{
lean_object* v_reuseFailAlloc_94_; 
v_reuseFailAlloc_94_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_94_, 0, v___x_91_);
v___x_93_ = v_reuseFailAlloc_94_;
goto v_reusejp_92_;
}
v_reusejp_92_:
{
return v___x_93_;
}
}
}
}
v___jp_78_:
{
lean_object* v___x_80_; uint8_t v___x_81_; 
lean_inc(v_toZero_74_);
v___x_80_ = lean_apply_2(v_inst_73_, v_a_79_, v_toZero_74_);
v___x_81_ = lean_unbox(v___x_80_);
if (v___x_81_ == 0)
{
lean_object* v___x_82_; 
lean_dec(v_toZero_74_);
v___x_82_ = lean_box(0);
return v___x_82_;
}
else
{
lean_object* v___x_83_; 
v___x_83_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_83_, 0, v_toZero_74_);
return v___x_83_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instMulZeroClass___redArg(lean_object* v_inst_96_, lean_object* v_inst_97_){
_start:
{
lean_object* v_toMul_98_; lean_object* v_toZero_99_; lean_object* v___x_101_; uint8_t v_isShared_102_; uint8_t v_isSharedCheck_108_; 
v_toMul_98_ = lean_ctor_get(v_inst_97_, 0);
v_toZero_99_ = lean_ctor_get(v_inst_97_, 1);
v_isSharedCheck_108_ = !lean_is_exclusive(v_inst_97_);
if (v_isSharedCheck_108_ == 0)
{
v___x_101_ = v_inst_97_;
v_isShared_102_ = v_isSharedCheck_108_;
goto v_resetjp_100_;
}
else
{
lean_inc(v_toZero_99_);
lean_inc(v_toMul_98_);
lean_dec(v_inst_97_);
v___x_101_ = lean_box(0);
v_isShared_102_ = v_isSharedCheck_108_;
goto v_resetjp_100_;
}
v_resetjp_100_:
{
lean_object* v___f_103_; lean_object* v___x_104_; lean_object* v___x_106_; 
lean_inc(v_toZero_99_);
v___f_103_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_instMulZeroClass___redArg___lam__0), 5, 3);
lean_closure_set(v___f_103_, 0, v_inst_96_);
lean_closure_set(v___f_103_, 1, v_toZero_99_);
lean_closure_set(v___f_103_, 2, v_toMul_98_);
v___x_104_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_104_, 0, v_toZero_99_);
if (v_isShared_102_ == 0)
{
lean_ctor_set(v___x_101_, 1, v___x_104_);
lean_ctor_set(v___x_101_, 0, v___f_103_);
v___x_106_ = v___x_101_;
goto v_reusejp_105_;
}
else
{
lean_object* v_reuseFailAlloc_107_; 
v_reuseFailAlloc_107_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_107_, 0, v___f_103_);
lean_ctor_set(v_reuseFailAlloc_107_, 1, v___x_104_);
v___x_106_ = v_reuseFailAlloc_107_;
goto v_reusejp_105_;
}
v_reusejp_105_:
{
return v___x_106_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instMulZeroClass(lean_object* v_00_u03b1_109_, lean_object* v_inst_110_, lean_object* v_inst_111_){
_start:
{
lean_object* v___x_112_; 
v___x_112_ = lp_mathlib_WithBot_instMulZeroClass___redArg(v_inst_110_, v_inst_111_);
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instMulZeroOneClass___redArg(lean_object* v_inst_113_, lean_object* v_inst_114_){
_start:
{
lean_object* v_toMulOneClass_115_; lean_object* v___x_116_; lean_object* v_toOne_117_; lean_object* v___x_119_; uint8_t v_isShared_120_; uint8_t v_isSharedCheck_136_; 
v_toMulOneClass_115_ = lean_ctor_get(v_inst_114_, 0);
lean_inc_ref(v_toMulOneClass_115_);
v___x_116_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_toMulOneClass_115_);
v_toOne_117_ = lean_ctor_get(v___x_116_, 0);
v_isSharedCheck_136_ = !lean_is_exclusive(v___x_116_);
if (v_isSharedCheck_136_ == 0)
{
lean_object* v_unused_137_; 
v_unused_137_ = lean_ctor_get(v___x_116_, 1);
lean_dec(v_unused_137_);
v___x_119_ = v___x_116_;
v_isShared_120_ = v_isSharedCheck_136_;
goto v_resetjp_118_;
}
else
{
lean_inc(v_toOne_117_);
lean_dec(v___x_116_);
v___x_119_ = lean_box(0);
v_isShared_120_ = v_isSharedCheck_136_;
goto v_resetjp_118_;
}
v_resetjp_118_:
{
lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v_toMul_123_; lean_object* v_toZero_124_; lean_object* v___x_126_; uint8_t v_isShared_127_; uint8_t v_isSharedCheck_135_; 
v___x_121_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v_inst_114_);
v___x_122_ = lp_mathlib_WithTop_instMulZeroClass___redArg(v_inst_113_, v___x_121_);
v_toMul_123_ = lean_ctor_get(v___x_122_, 0);
v_toZero_124_ = lean_ctor_get(v___x_122_, 1);
v_isSharedCheck_135_ = !lean_is_exclusive(v___x_122_);
if (v_isSharedCheck_135_ == 0)
{
v___x_126_ = v___x_122_;
v_isShared_127_ = v_isSharedCheck_135_;
goto v_resetjp_125_;
}
else
{
lean_inc(v_toZero_124_);
lean_inc(v_toMul_123_);
lean_dec(v___x_122_);
v___x_126_ = lean_box(0);
v_isShared_127_ = v_isSharedCheck_135_;
goto v_resetjp_125_;
}
v_resetjp_125_:
{
lean_object* v___x_128_; lean_object* v___x_130_; 
v___x_128_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_128_, 0, v_toOne_117_);
if (v_isShared_127_ == 0)
{
lean_ctor_set(v___x_126_, 1, v_toMul_123_);
lean_ctor_set(v___x_126_, 0, v___x_128_);
v___x_130_ = v___x_126_;
goto v_reusejp_129_;
}
else
{
lean_object* v_reuseFailAlloc_134_; 
v_reuseFailAlloc_134_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_134_, 0, v___x_128_);
lean_ctor_set(v_reuseFailAlloc_134_, 1, v_toMul_123_);
v___x_130_ = v_reuseFailAlloc_134_;
goto v_reusejp_129_;
}
v_reusejp_129_:
{
lean_object* v___x_132_; 
if (v_isShared_120_ == 0)
{
lean_ctor_set(v___x_119_, 1, v_toZero_124_);
lean_ctor_set(v___x_119_, 0, v___x_130_);
v___x_132_ = v___x_119_;
goto v_reusejp_131_;
}
else
{
lean_object* v_reuseFailAlloc_133_; 
v_reuseFailAlloc_133_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_133_, 0, v___x_130_);
lean_ctor_set(v_reuseFailAlloc_133_, 1, v_toZero_124_);
v___x_132_ = v_reuseFailAlloc_133_;
goto v_reusejp_131_;
}
v_reusejp_131_:
{
return v___x_132_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instMulZeroOneClass(lean_object* v_00_u03b1_138_, lean_object* v_inst_139_, lean_object* v_inst_140_, lean_object* v_inst_141_){
_start:
{
lean_object* v___x_142_; 
v___x_142_ = lp_mathlib_WithTop_instMulZeroOneClass___redArg(v_inst_139_, v_inst_140_);
return v___x_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instMulZeroOneClass___redArg(lean_object* v_inst_143_, lean_object* v_inst_144_){
_start:
{
lean_object* v_toMulOneClass_145_; lean_object* v___x_146_; lean_object* v_toOne_147_; lean_object* v___x_149_; uint8_t v_isShared_150_; uint8_t v_isSharedCheck_166_; 
v_toMulOneClass_145_ = lean_ctor_get(v_inst_144_, 0);
lean_inc_ref(v_toMulOneClass_145_);
v___x_146_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_toMulOneClass_145_);
v_toOne_147_ = lean_ctor_get(v___x_146_, 0);
v_isSharedCheck_166_ = !lean_is_exclusive(v___x_146_);
if (v_isSharedCheck_166_ == 0)
{
lean_object* v_unused_167_; 
v_unused_167_ = lean_ctor_get(v___x_146_, 1);
lean_dec(v_unused_167_);
v___x_149_ = v___x_146_;
v_isShared_150_ = v_isSharedCheck_166_;
goto v_resetjp_148_;
}
else
{
lean_inc(v_toOne_147_);
lean_dec(v___x_146_);
v___x_149_ = lean_box(0);
v_isShared_150_ = v_isSharedCheck_166_;
goto v_resetjp_148_;
}
v_resetjp_148_:
{
lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v_toMul_153_; lean_object* v_toZero_154_; lean_object* v___x_156_; uint8_t v_isShared_157_; uint8_t v_isSharedCheck_165_; 
v___x_151_ = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(v_inst_144_);
v___x_152_ = lp_mathlib_WithBot_instMulZeroClass___redArg(v_inst_143_, v___x_151_);
v_toMul_153_ = lean_ctor_get(v___x_152_, 0);
v_toZero_154_ = lean_ctor_get(v___x_152_, 1);
v_isSharedCheck_165_ = !lean_is_exclusive(v___x_152_);
if (v_isSharedCheck_165_ == 0)
{
v___x_156_ = v___x_152_;
v_isShared_157_ = v_isSharedCheck_165_;
goto v_resetjp_155_;
}
else
{
lean_inc(v_toZero_154_);
lean_inc(v_toMul_153_);
lean_dec(v___x_152_);
v___x_156_ = lean_box(0);
v_isShared_157_ = v_isSharedCheck_165_;
goto v_resetjp_155_;
}
v_resetjp_155_:
{
lean_object* v___x_158_; lean_object* v___x_160_; 
v___x_158_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_158_, 0, v_toOne_147_);
if (v_isShared_157_ == 0)
{
lean_ctor_set(v___x_156_, 1, v_toMul_153_);
lean_ctor_set(v___x_156_, 0, v___x_158_);
v___x_160_ = v___x_156_;
goto v_reusejp_159_;
}
else
{
lean_object* v_reuseFailAlloc_164_; 
v_reuseFailAlloc_164_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_164_, 0, v___x_158_);
lean_ctor_set(v_reuseFailAlloc_164_, 1, v_toMul_153_);
v___x_160_ = v_reuseFailAlloc_164_;
goto v_reusejp_159_;
}
v_reusejp_159_:
{
lean_object* v___x_162_; 
if (v_isShared_150_ == 0)
{
lean_ctor_set(v___x_149_, 1, v_toZero_154_);
lean_ctor_set(v___x_149_, 0, v___x_160_);
v___x_162_ = v___x_149_;
goto v_reusejp_161_;
}
else
{
lean_object* v_reuseFailAlloc_163_; 
v_reuseFailAlloc_163_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_163_, 0, v___x_160_);
lean_ctor_set(v_reuseFailAlloc_163_, 1, v_toZero_154_);
v___x_162_ = v_reuseFailAlloc_163_;
goto v_reusejp_161_;
}
v_reusejp_161_:
{
return v___x_162_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instMulZeroOneClass(lean_object* v_00_u03b1_168_, lean_object* v_inst_169_, lean_object* v_inst_170_, lean_object* v_inst_171_){
_start:
{
lean_object* v___x_172_; 
v___x_172_ = lp_mathlib_WithBot_instMulZeroOneClass___redArg(v_inst_169_, v_inst_170_);
return v___x_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_withTopMap___redArg___lam__0(lean_object* v_f_173_, lean_object* v___y_174_){
_start:
{
lean_object* v___x_175_; 
v___x_175_ = lean_apply_1(v_f_173_, v___y_174_);
return v___x_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_withTopMap___redArg(lean_object* v_f_176_){
_start:
{
lean_object* v___f_177_; lean_object* v___x_178_; 
v___f_177_ = lean_alloc_closure((void*)(lp_mathlib_MonoidWithZeroHom_withTopMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_177_, 0, v_f_176_);
v___x_178_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_map), 4, 3);
lean_closure_set(v___x_178_, 0, lean_box(0));
lean_closure_set(v___x_178_, 1, lean_box(0));
lean_closure_set(v___x_178_, 2, v___f_177_);
return v___x_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_withTopMap(lean_object* v_R_179_, lean_object* v_S_180_, lean_object* v_inst_181_, lean_object* v_inst_182_, lean_object* v_inst_183_, lean_object* v_inst_184_, lean_object* v_inst_185_, lean_object* v_inst_186_, lean_object* v_f_187_, lean_object* v_hf_188_){
_start:
{
lean_object* v___x_189_; 
v___x_189_ = lp_mathlib_MonoidWithZeroHom_withTopMap___redArg(v_f_187_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_withTopMap___boxed(lean_object* v_R_190_, lean_object* v_S_191_, lean_object* v_inst_192_, lean_object* v_inst_193_, lean_object* v_inst_194_, lean_object* v_inst_195_, lean_object* v_inst_196_, lean_object* v_inst_197_, lean_object* v_f_198_, lean_object* v_hf_199_){
_start:
{
lean_object* v_res_200_; 
v_res_200_ = lp_mathlib_MonoidWithZeroHom_withTopMap(v_R_190_, v_S_191_, v_inst_192_, v_inst_193_, v_inst_194_, v_inst_195_, v_inst_196_, v_inst_197_, v_f_198_, v_hf_199_);
lean_dec_ref(v_inst_196_);
lean_dec_ref(v_inst_195_);
lean_dec_ref(v_inst_193_);
lean_dec_ref(v_inst_192_);
return v_res_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_withBotMap___redArg(lean_object* v_f_201_){
_start:
{
lean_object* v___f_202_; lean_object* v___x_203_; 
v___f_202_ = lean_alloc_closure((void*)(lp_mathlib_MonoidWithZeroHom_withTopMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_202_, 0, v_f_201_);
v___x_203_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_map), 4, 3);
lean_closure_set(v___x_203_, 0, lean_box(0));
lean_closure_set(v___x_203_, 1, lean_box(0));
lean_closure_set(v___x_203_, 2, v___f_202_);
return v___x_203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_withBotMap(lean_object* v_R_204_, lean_object* v_S_205_, lean_object* v_inst_206_, lean_object* v_inst_207_, lean_object* v_inst_208_, lean_object* v_inst_209_, lean_object* v_inst_210_, lean_object* v_inst_211_, lean_object* v_f_212_, lean_object* v_hf_213_){
_start:
{
lean_object* v___x_214_; 
v___x_214_ = lp_mathlib_MonoidWithZeroHom_withBotMap___redArg(v_f_212_);
return v___x_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_withBotMap___boxed(lean_object* v_R_215_, lean_object* v_S_216_, lean_object* v_inst_217_, lean_object* v_inst_218_, lean_object* v_inst_219_, lean_object* v_inst_220_, lean_object* v_inst_221_, lean_object* v_inst_222_, lean_object* v_f_223_, lean_object* v_hf_224_){
_start:
{
lean_object* v_res_225_; 
v_res_225_ = lp_mathlib_MonoidWithZeroHom_withBotMap(v_R_215_, v_S_216_, v_inst_217_, v_inst_218_, v_inst_219_, v_inst_220_, v_inst_221_, v_inst_222_, v_f_223_, v_hf_224_);
lean_dec_ref(v_inst_221_);
lean_dec_ref(v_inst_220_);
lean_dec_ref(v_inst_218_);
lean_dec_ref(v_inst_217_);
return v_res_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instSemigroupWithZero___redArg(lean_object* v_inst_226_, lean_object* v_inst_227_){
_start:
{
lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v_toMul_230_; lean_object* v_toZero_231_; lean_object* v___x_233_; uint8_t v_isShared_234_; uint8_t v_isSharedCheck_238_; 
v___x_228_ = lp_mathlib_SemigroupWithZero_toMulZeroClass___redArg(v_inst_227_);
v___x_229_ = lp_mathlib_WithTop_instMulZeroClass___redArg(v_inst_226_, v___x_228_);
v_toMul_230_ = lean_ctor_get(v___x_229_, 0);
v_toZero_231_ = lean_ctor_get(v___x_229_, 1);
v_isSharedCheck_238_ = !lean_is_exclusive(v___x_229_);
if (v_isSharedCheck_238_ == 0)
{
v___x_233_ = v___x_229_;
v_isShared_234_ = v_isSharedCheck_238_;
goto v_resetjp_232_;
}
else
{
lean_inc(v_toZero_231_);
lean_inc(v_toMul_230_);
lean_dec(v___x_229_);
v___x_233_ = lean_box(0);
v_isShared_234_ = v_isSharedCheck_238_;
goto v_resetjp_232_;
}
v_resetjp_232_:
{
lean_object* v___x_236_; 
if (v_isShared_234_ == 0)
{
v___x_236_ = v___x_233_;
goto v_reusejp_235_;
}
else
{
lean_object* v_reuseFailAlloc_237_; 
v_reuseFailAlloc_237_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_237_, 0, v_toMul_230_);
lean_ctor_set(v_reuseFailAlloc_237_, 1, v_toZero_231_);
v___x_236_ = v_reuseFailAlloc_237_;
goto v_reusejp_235_;
}
v_reusejp_235_:
{
return v___x_236_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instSemigroupWithZero(lean_object* v_00_u03b1_239_, lean_object* v_inst_240_, lean_object* v_inst_241_, lean_object* v_inst_242_){
_start:
{
lean_object* v___x_243_; 
v___x_243_ = lp_mathlib_WithTop_instSemigroupWithZero___redArg(v_inst_240_, v_inst_241_);
return v___x_243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instSemigroupWithZero___redArg(lean_object* v_inst_244_, lean_object* v_inst_245_){
_start:
{
lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v_toMul_248_; lean_object* v_toZero_249_; lean_object* v___x_251_; uint8_t v_isShared_252_; uint8_t v_isSharedCheck_256_; 
v___x_246_ = lp_mathlib_SemigroupWithZero_toMulZeroClass___redArg(v_inst_245_);
v___x_247_ = lp_mathlib_WithBot_instMulZeroClass___redArg(v_inst_244_, v___x_246_);
v_toMul_248_ = lean_ctor_get(v___x_247_, 0);
v_toZero_249_ = lean_ctor_get(v___x_247_, 1);
v_isSharedCheck_256_ = !lean_is_exclusive(v___x_247_);
if (v_isSharedCheck_256_ == 0)
{
v___x_251_ = v___x_247_;
v_isShared_252_ = v_isSharedCheck_256_;
goto v_resetjp_250_;
}
else
{
lean_inc(v_toZero_249_);
lean_inc(v_toMul_248_);
lean_dec(v___x_247_);
v___x_251_ = lean_box(0);
v_isShared_252_ = v_isSharedCheck_256_;
goto v_resetjp_250_;
}
v_resetjp_250_:
{
lean_object* v___x_254_; 
if (v_isShared_252_ == 0)
{
v___x_254_ = v___x_251_;
goto v_reusejp_253_;
}
else
{
lean_object* v_reuseFailAlloc_255_; 
v_reuseFailAlloc_255_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_255_, 0, v_toMul_248_);
lean_ctor_set(v_reuseFailAlloc_255_, 1, v_toZero_249_);
v___x_254_ = v_reuseFailAlloc_255_;
goto v_reusejp_253_;
}
v_reusejp_253_:
{
return v___x_254_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instSemigroupWithZero(lean_object* v_00_u03b1_257_, lean_object* v_inst_258_, lean_object* v_inst_259_, lean_object* v_inst_260_){
_start:
{
lean_object* v___x_261_; 
v___x_261_ = lp_mathlib_WithBot_instSemigroupWithZero___redArg(v_inst_258_, v_inst_259_);
return v___x_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Ring_WithTop_0__WithTop_instMonoidWithZero_match__1_splitter___redArg(lean_object* v_a_262_, lean_object* v_n_263_, lean_object* v_h__1_264_, lean_object* v_h__2_265_, lean_object* v_h__3_266_){
_start:
{
if (lean_obj_tag(v_a_262_) == 0)
{
lean_object* v_zero_267_; uint8_t v_isZero_268_; 
lean_dec(v_h__1_264_);
v_zero_267_ = lean_unsigned_to_nat(0u);
v_isZero_268_ = lean_nat_dec_eq(v_n_263_, v_zero_267_);
if (v_isZero_268_ == 1)
{
lean_object* v___x_269_; lean_object* v___x_270_; 
lean_dec(v_h__3_266_);
lean_dec(v_n_263_);
v___x_269_ = lean_box(0);
v___x_270_ = lean_apply_1(v_h__2_265_, v___x_269_);
return v___x_270_;
}
else
{
lean_object* v_one_271_; lean_object* v_n_272_; lean_object* v___x_273_; 
lean_dec(v_h__2_265_);
v_one_271_ = lean_unsigned_to_nat(1u);
v_n_272_ = lean_nat_sub(v_n_263_, v_one_271_);
lean_dec(v_n_263_);
v___x_273_ = lean_apply_1(v_h__3_266_, v_n_272_);
return v___x_273_;
}
}
else
{
lean_object* v_val_274_; lean_object* v___x_275_; 
lean_dec(v_h__3_266_);
lean_dec(v_h__2_265_);
v_val_274_ = lean_ctor_get(v_a_262_, 0);
lean_inc(v_val_274_);
lean_dec_ref_known(v_a_262_, 1);
v___x_275_ = lean_apply_2(v_h__1_264_, v_val_274_, v_n_263_);
return v___x_275_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Order_Ring_WithTop_0__WithTop_instMonoidWithZero_match__1_splitter(lean_object* v_00_u03b1_276_, lean_object* v_motive_277_, lean_object* v_a_278_, lean_object* v_n_279_, lean_object* v_h__1_280_, lean_object* v_h__2_281_, lean_object* v_h__3_282_){
_start:
{
if (lean_obj_tag(v_a_278_) == 0)
{
lean_object* v_zero_283_; uint8_t v_isZero_284_; 
lean_dec(v_h__1_280_);
v_zero_283_ = lean_unsigned_to_nat(0u);
v_isZero_284_ = lean_nat_dec_eq(v_n_279_, v_zero_283_);
if (v_isZero_284_ == 1)
{
lean_object* v___x_285_; lean_object* v___x_286_; 
lean_dec(v_h__3_282_);
lean_dec(v_n_279_);
v___x_285_ = lean_box(0);
v___x_286_ = lean_apply_1(v_h__2_281_, v___x_285_);
return v___x_286_;
}
else
{
lean_object* v_one_287_; lean_object* v_n_288_; lean_object* v___x_289_; 
lean_dec(v_h__2_281_);
v_one_287_ = lean_unsigned_to_nat(1u);
v_n_288_ = lean_nat_sub(v_n_279_, v_one_287_);
lean_dec(v_n_279_);
v___x_289_ = lean_apply_1(v_h__3_282_, v_n_288_);
return v___x_289_;
}
}
else
{
lean_object* v_val_290_; lean_object* v___x_291_; 
lean_dec(v_h__3_282_);
lean_dec(v_h__2_281_);
v_val_290_ = lean_ctor_get(v_a_278_, 0);
lean_inc(v_val_290_);
lean_dec_ref_known(v_a_278_, 1);
v___x_291_ = lean_apply_2(v_h__1_280_, v_val_290_, v_n_279_);
return v___x_291_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instMonoidWithZero___redArg___lam__0(lean_object* v_toOne_292_, lean_object* v___x_293_, lean_object* v_toNPow_294_, lean_object* v_n_295_, lean_object* v_a_296_){
_start:
{
if (lean_obj_tag(v_a_296_) == 0)
{
lean_object* v_zero_297_; uint8_t v_isZero_298_; 
lean_dec(v_toNPow_294_);
v_zero_297_ = lean_unsigned_to_nat(0u);
v_isZero_298_ = lean_nat_dec_eq(v_n_295_, v_zero_297_);
lean_dec(v_n_295_);
if (v_isZero_298_ == 1)
{
lean_object* v___x_299_; 
v___x_299_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_299_, 0, v_toOne_292_);
return v___x_299_;
}
else
{
lean_dec(v_toOne_292_);
lean_inc(v___x_293_);
return v___x_293_;
}
}
else
{
lean_object* v_val_300_; lean_object* v___x_302_; uint8_t v_isShared_303_; uint8_t v_isSharedCheck_308_; 
lean_dec(v_toOne_292_);
v_val_300_ = lean_ctor_get(v_a_296_, 0);
v_isSharedCheck_308_ = !lean_is_exclusive(v_a_296_);
if (v_isSharedCheck_308_ == 0)
{
v___x_302_ = v_a_296_;
v_isShared_303_ = v_isSharedCheck_308_;
goto v_resetjp_301_;
}
else
{
lean_inc(v_val_300_);
lean_dec(v_a_296_);
v___x_302_ = lean_box(0);
v_isShared_303_ = v_isSharedCheck_308_;
goto v_resetjp_301_;
}
v_resetjp_301_:
{
lean_object* v___x_304_; lean_object* v___x_306_; 
v___x_304_ = lean_apply_2(v_toNPow_294_, v_n_295_, v_val_300_);
if (v_isShared_303_ == 0)
{
lean_ctor_set(v___x_302_, 0, v___x_304_);
v___x_306_ = v___x_302_;
goto v_reusejp_305_;
}
else
{
lean_object* v_reuseFailAlloc_307_; 
v_reuseFailAlloc_307_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_307_, 0, v___x_304_);
v___x_306_ = v_reuseFailAlloc_307_;
goto v_reusejp_305_;
}
v_reusejp_305_:
{
return v___x_306_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instMonoidWithZero___redArg___lam__0___boxed(lean_object* v_toOne_309_, lean_object* v___x_310_, lean_object* v_toNPow_311_, lean_object* v_n_312_, lean_object* v_a_313_){
_start:
{
lean_object* v_res_314_; 
v_res_314_ = lp_mathlib_WithTop_instMonoidWithZero___redArg___lam__0(v_toOne_309_, v___x_310_, v_toNPow_311_, v_n_312_, v_a_313_);
lean_dec(v___x_310_);
return v_res_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instMonoidWithZero___redArg(lean_object* v_inst_315_, lean_object* v_inst_316_){
_start:
{
lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v_toMulOneClass_319_; lean_object* v_toMonoid_320_; lean_object* v___x_322_; uint8_t v_isShared_323_; uint8_t v_isSharedCheck_345_; 
lean_inc_ref(v_inst_316_);
v___x_317_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v_inst_316_);
lean_inc_ref(v___x_317_);
v___x_318_ = lp_mathlib_WithTop_instMulZeroOneClass___redArg(v_inst_315_, v___x_317_);
v_toMulOneClass_319_ = lean_ctor_get(v___x_318_, 0);
lean_inc_ref(v_toMulOneClass_319_);
v_toMonoid_320_ = lean_ctor_get(v_inst_316_, 0);
v_isSharedCheck_345_ = !lean_is_exclusive(v_inst_316_);
if (v_isSharedCheck_345_ == 0)
{
lean_object* v_unused_346_; 
v_unused_346_ = lean_ctor_get(v_inst_316_, 1);
lean_dec(v_unused_346_);
v___x_322_ = v_inst_316_;
v_isShared_323_ = v_isSharedCheck_345_;
goto v_resetjp_321_;
}
else
{
lean_inc(v_toMonoid_320_);
lean_dec(v_inst_316_);
v___x_322_ = lean_box(0);
v_isShared_323_ = v_isSharedCheck_345_;
goto v_resetjp_321_;
}
v_resetjp_321_:
{
lean_object* v_toZero_324_; lean_object* v_toOne_325_; lean_object* v_toMul_326_; lean_object* v_toNPow_327_; lean_object* v___x_329_; uint8_t v_isShared_330_; uint8_t v_isSharedCheck_342_; 
v_toZero_324_ = lean_ctor_get(v___x_318_, 1);
lean_inc(v_toZero_324_);
lean_dec_ref(v___x_318_);
v_toOne_325_ = lean_ctor_get(v_toMulOneClass_319_, 0);
lean_inc(v_toOne_325_);
v_toMul_326_ = lean_ctor_get(v_toMulOneClass_319_, 1);
lean_inc(v_toMul_326_);
lean_dec_ref(v_toMulOneClass_319_);
v_toNPow_327_ = lean_ctor_get(v_toMonoid_320_, 2);
v_isSharedCheck_342_ = !lean_is_exclusive(v_toMonoid_320_);
if (v_isSharedCheck_342_ == 0)
{
lean_object* v_unused_343_; lean_object* v_unused_344_; 
v_unused_343_ = lean_ctor_get(v_toMonoid_320_, 1);
lean_dec(v_unused_343_);
v_unused_344_ = lean_ctor_get(v_toMonoid_320_, 0);
lean_dec(v_unused_344_);
v___x_329_ = v_toMonoid_320_;
v_isShared_330_ = v_isSharedCheck_342_;
goto v_resetjp_328_;
}
else
{
lean_inc(v_toNPow_327_);
lean_dec(v_toMonoid_320_);
v___x_329_ = lean_box(0);
v_isShared_330_ = v_isSharedCheck_342_;
goto v_resetjp_328_;
}
v_resetjp_328_:
{
lean_object* v_toMulOneClass_331_; lean_object* v___x_332_; lean_object* v_toOne_333_; lean_object* v___x_334_; lean_object* v___f_335_; lean_object* v___x_337_; 
v_toMulOneClass_331_ = lean_ctor_get(v___x_317_, 0);
lean_inc_ref(v_toMulOneClass_331_);
lean_dec_ref(v___x_317_);
v___x_332_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_toMulOneClass_331_);
v_toOne_333_ = lean_ctor_get(v___x_332_, 0);
lean_inc(v_toOne_333_);
lean_dec_ref(v___x_332_);
v___x_334_ = lean_box(0);
v___f_335_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_instMonoidWithZero___redArg___lam__0___boxed), 5, 3);
lean_closure_set(v___f_335_, 0, v_toOne_333_);
lean_closure_set(v___f_335_, 1, v___x_334_);
lean_closure_set(v___f_335_, 2, v_toNPow_327_);
if (v_isShared_330_ == 0)
{
lean_ctor_set(v___x_329_, 2, v___f_335_);
lean_ctor_set(v___x_329_, 1, v_toMul_326_);
lean_ctor_set(v___x_329_, 0, v_toOne_325_);
v___x_337_ = v___x_329_;
goto v_reusejp_336_;
}
else
{
lean_object* v_reuseFailAlloc_341_; 
v_reuseFailAlloc_341_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_341_, 0, v_toOne_325_);
lean_ctor_set(v_reuseFailAlloc_341_, 1, v_toMul_326_);
lean_ctor_set(v_reuseFailAlloc_341_, 2, v___f_335_);
v___x_337_ = v_reuseFailAlloc_341_;
goto v_reusejp_336_;
}
v_reusejp_336_:
{
lean_object* v___x_339_; 
if (v_isShared_323_ == 0)
{
lean_ctor_set(v___x_322_, 1, v_toZero_324_);
lean_ctor_set(v___x_322_, 0, v___x_337_);
v___x_339_ = v___x_322_;
goto v_reusejp_338_;
}
else
{
lean_object* v_reuseFailAlloc_340_; 
v_reuseFailAlloc_340_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_340_, 0, v___x_337_);
lean_ctor_set(v_reuseFailAlloc_340_, 1, v_toZero_324_);
v___x_339_ = v_reuseFailAlloc_340_;
goto v_reusejp_338_;
}
v_reusejp_338_:
{
return v___x_339_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instMonoidWithZero(lean_object* v_00_u03b1_347_, lean_object* v_inst_348_, lean_object* v_inst_349_, lean_object* v_inst_350_, lean_object* v_inst_351_){
_start:
{
lean_object* v___x_352_; 
v___x_352_ = lp_mathlib_WithTop_instMonoidWithZero___redArg(v_inst_348_, v_inst_349_);
return v___x_352_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instMonoidWithZero_match__1___redArg(lean_object* v_a_353_, lean_object* v_n_354_, lean_object* v_h__1_355_, lean_object* v_h__2_356_, lean_object* v_h__3_357_){
_start:
{
if (lean_obj_tag(v_a_353_) == 0)
{
lean_object* v_zero_358_; uint8_t v_isZero_359_; 
lean_dec(v_h__1_355_);
v_zero_358_ = lean_unsigned_to_nat(0u);
v_isZero_359_ = lean_nat_dec_eq(v_n_354_, v_zero_358_);
if (v_isZero_359_ == 1)
{
lean_object* v___x_360_; lean_object* v___x_361_; 
lean_dec(v_h__3_357_);
lean_dec(v_n_354_);
v___x_360_ = lean_box(0);
v___x_361_ = lean_apply_1(v_h__2_356_, v___x_360_);
return v___x_361_;
}
else
{
lean_object* v_one_362_; lean_object* v_n_363_; lean_object* v___x_364_; 
lean_dec(v_h__2_356_);
v_one_362_ = lean_unsigned_to_nat(1u);
v_n_363_ = lean_nat_sub(v_n_354_, v_one_362_);
lean_dec(v_n_354_);
v___x_364_ = lean_apply_1(v_h__3_357_, v_n_363_);
return v___x_364_;
}
}
else
{
lean_object* v_val_365_; lean_object* v___x_366_; 
lean_dec(v_h__3_357_);
lean_dec(v_h__2_356_);
v_val_365_ = lean_ctor_get(v_a_353_, 0);
lean_inc(v_val_365_);
lean_dec_ref_known(v_a_353_, 1);
v___x_366_ = lean_apply_2(v_h__1_355_, v_val_365_, v_n_354_);
return v___x_366_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instMonoidWithZero_match__1(lean_object* v_00_u03b1_367_, lean_object* v_motive_368_, lean_object* v_a_369_, lean_object* v_n_370_, lean_object* v_h__1_371_, lean_object* v_h__2_372_, lean_object* v_h__3_373_){
_start:
{
lean_object* v___x_374_; 
v___x_374_ = lp_mathlib_WithBot_instMonoidWithZero_match__1___redArg(v_a_369_, v_n_370_, v_h__1_371_, v_h__2_372_, v_h__3_373_);
return v___x_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instMonoidWithZero___redArg___lam__0(lean_object* v_toOne_375_, lean_object* v_toNPow_376_, lean_object* v_n_377_, lean_object* v_a_378_){
_start:
{
if (lean_obj_tag(v_a_378_) == 0)
{
lean_object* v_zero_379_; uint8_t v_isZero_380_; 
lean_dec(v_toNPow_376_);
v_zero_379_ = lean_unsigned_to_nat(0u);
v_isZero_380_ = lean_nat_dec_eq(v_n_377_, v_zero_379_);
lean_dec(v_n_377_);
if (v_isZero_380_ == 1)
{
lean_object* v___x_381_; 
v___x_381_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_381_, 0, v_toOne_375_);
return v___x_381_;
}
else
{
lean_dec(v_toOne_375_);
return v_a_378_;
}
}
else
{
lean_object* v_val_382_; lean_object* v___x_384_; uint8_t v_isShared_385_; uint8_t v_isSharedCheck_390_; 
lean_dec(v_toOne_375_);
v_val_382_ = lean_ctor_get(v_a_378_, 0);
v_isSharedCheck_390_ = !lean_is_exclusive(v_a_378_);
if (v_isSharedCheck_390_ == 0)
{
v___x_384_ = v_a_378_;
v_isShared_385_ = v_isSharedCheck_390_;
goto v_resetjp_383_;
}
else
{
lean_inc(v_val_382_);
lean_dec(v_a_378_);
v___x_384_ = lean_box(0);
v_isShared_385_ = v_isSharedCheck_390_;
goto v_resetjp_383_;
}
v_resetjp_383_:
{
lean_object* v___x_386_; lean_object* v___x_388_; 
v___x_386_ = lean_apply_2(v_toNPow_376_, v_n_377_, v_val_382_);
if (v_isShared_385_ == 0)
{
lean_ctor_set(v___x_384_, 0, v___x_386_);
v___x_388_ = v___x_384_;
goto v_reusejp_387_;
}
else
{
lean_object* v_reuseFailAlloc_389_; 
v_reuseFailAlloc_389_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_389_, 0, v___x_386_);
v___x_388_ = v_reuseFailAlloc_389_;
goto v_reusejp_387_;
}
v_reusejp_387_:
{
return v___x_388_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instMonoidWithZero___redArg(lean_object* v_inst_391_, lean_object* v_inst_392_){
_start:
{
lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v_toMulOneClass_395_; lean_object* v_toMonoid_396_; lean_object* v___x_398_; uint8_t v_isShared_399_; uint8_t v_isSharedCheck_420_; 
lean_inc_ref(v_inst_392_);
v___x_393_ = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(v_inst_392_);
lean_inc_ref(v___x_393_);
v___x_394_ = lp_mathlib_WithBot_instMulZeroOneClass___redArg(v_inst_391_, v___x_393_);
v_toMulOneClass_395_ = lean_ctor_get(v___x_394_, 0);
lean_inc_ref(v_toMulOneClass_395_);
v_toMonoid_396_ = lean_ctor_get(v_inst_392_, 0);
v_isSharedCheck_420_ = !lean_is_exclusive(v_inst_392_);
if (v_isSharedCheck_420_ == 0)
{
lean_object* v_unused_421_; 
v_unused_421_ = lean_ctor_get(v_inst_392_, 1);
lean_dec(v_unused_421_);
v___x_398_ = v_inst_392_;
v_isShared_399_ = v_isSharedCheck_420_;
goto v_resetjp_397_;
}
else
{
lean_inc(v_toMonoid_396_);
lean_dec(v_inst_392_);
v___x_398_ = lean_box(0);
v_isShared_399_ = v_isSharedCheck_420_;
goto v_resetjp_397_;
}
v_resetjp_397_:
{
lean_object* v_toZero_400_; lean_object* v_toOne_401_; lean_object* v_toMul_402_; lean_object* v_toNPow_403_; lean_object* v___x_405_; uint8_t v_isShared_406_; uint8_t v_isSharedCheck_417_; 
v_toZero_400_ = lean_ctor_get(v___x_394_, 1);
lean_inc(v_toZero_400_);
lean_dec_ref(v___x_394_);
v_toOne_401_ = lean_ctor_get(v_toMulOneClass_395_, 0);
lean_inc(v_toOne_401_);
v_toMul_402_ = lean_ctor_get(v_toMulOneClass_395_, 1);
lean_inc(v_toMul_402_);
lean_dec_ref(v_toMulOneClass_395_);
v_toNPow_403_ = lean_ctor_get(v_toMonoid_396_, 2);
v_isSharedCheck_417_ = !lean_is_exclusive(v_toMonoid_396_);
if (v_isSharedCheck_417_ == 0)
{
lean_object* v_unused_418_; lean_object* v_unused_419_; 
v_unused_418_ = lean_ctor_get(v_toMonoid_396_, 1);
lean_dec(v_unused_418_);
v_unused_419_ = lean_ctor_get(v_toMonoid_396_, 0);
lean_dec(v_unused_419_);
v___x_405_ = v_toMonoid_396_;
v_isShared_406_ = v_isSharedCheck_417_;
goto v_resetjp_404_;
}
else
{
lean_inc(v_toNPow_403_);
lean_dec(v_toMonoid_396_);
v___x_405_ = lean_box(0);
v_isShared_406_ = v_isSharedCheck_417_;
goto v_resetjp_404_;
}
v_resetjp_404_:
{
lean_object* v_toMulOneClass_407_; lean_object* v___x_408_; lean_object* v_toOne_409_; lean_object* v___f_410_; lean_object* v___x_412_; 
v_toMulOneClass_407_ = lean_ctor_get(v___x_393_, 0);
lean_inc_ref(v_toMulOneClass_407_);
lean_dec_ref(v___x_393_);
v___x_408_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_toMulOneClass_407_);
v_toOne_409_ = lean_ctor_get(v___x_408_, 0);
lean_inc(v_toOne_409_);
lean_dec_ref(v___x_408_);
v___f_410_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_instMonoidWithZero___redArg___lam__0), 4, 2);
lean_closure_set(v___f_410_, 0, v_toOne_409_);
lean_closure_set(v___f_410_, 1, v_toNPow_403_);
if (v_isShared_406_ == 0)
{
lean_ctor_set(v___x_405_, 2, v___f_410_);
lean_ctor_set(v___x_405_, 1, v_toMul_402_);
lean_ctor_set(v___x_405_, 0, v_toOne_401_);
v___x_412_ = v___x_405_;
goto v_reusejp_411_;
}
else
{
lean_object* v_reuseFailAlloc_416_; 
v_reuseFailAlloc_416_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_416_, 0, v_toOne_401_);
lean_ctor_set(v_reuseFailAlloc_416_, 1, v_toMul_402_);
lean_ctor_set(v_reuseFailAlloc_416_, 2, v___f_410_);
v___x_412_ = v_reuseFailAlloc_416_;
goto v_reusejp_411_;
}
v_reusejp_411_:
{
lean_object* v___x_414_; 
if (v_isShared_399_ == 0)
{
lean_ctor_set(v___x_398_, 1, v_toZero_400_);
lean_ctor_set(v___x_398_, 0, v___x_412_);
v___x_414_ = v___x_398_;
goto v_reusejp_413_;
}
else
{
lean_object* v_reuseFailAlloc_415_; 
v_reuseFailAlloc_415_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_415_, 0, v___x_412_);
lean_ctor_set(v_reuseFailAlloc_415_, 1, v_toZero_400_);
v___x_414_ = v_reuseFailAlloc_415_;
goto v_reusejp_413_;
}
v_reusejp_413_:
{
return v___x_414_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instMonoidWithZero(lean_object* v_00_u03b1_422_, lean_object* v_inst_423_, lean_object* v_inst_424_, lean_object* v_inst_425_, lean_object* v_inst_426_){
_start:
{
lean_object* v___x_427_; 
v___x_427_ = lp_mathlib_WithBot_instMonoidWithZero___redArg(v_inst_423_, v_inst_424_);
return v___x_427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instCommMonoidWithZero___redArg(lean_object* v_inst_428_, lean_object* v_inst_429_){
_start:
{
lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v_toMonoid_432_; lean_object* v_toZero_433_; lean_object* v___x_435_; uint8_t v_isShared_436_; uint8_t v_isSharedCheck_440_; 
v___x_430_ = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(v_inst_429_);
v___x_431_ = lp_mathlib_WithTop_instMonoidWithZero___redArg(v_inst_428_, v___x_430_);
v_toMonoid_432_ = lean_ctor_get(v___x_431_, 0);
v_toZero_433_ = lean_ctor_get(v___x_431_, 1);
v_isSharedCheck_440_ = !lean_is_exclusive(v___x_431_);
if (v_isSharedCheck_440_ == 0)
{
v___x_435_ = v___x_431_;
v_isShared_436_ = v_isSharedCheck_440_;
goto v_resetjp_434_;
}
else
{
lean_inc(v_toZero_433_);
lean_inc(v_toMonoid_432_);
lean_dec(v___x_431_);
v___x_435_ = lean_box(0);
v_isShared_436_ = v_isSharedCheck_440_;
goto v_resetjp_434_;
}
v_resetjp_434_:
{
lean_object* v___x_438_; 
if (v_isShared_436_ == 0)
{
v___x_438_ = v___x_435_;
goto v_reusejp_437_;
}
else
{
lean_object* v_reuseFailAlloc_439_; 
v_reuseFailAlloc_439_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_439_, 0, v_toMonoid_432_);
lean_ctor_set(v_reuseFailAlloc_439_, 1, v_toZero_433_);
v___x_438_ = v_reuseFailAlloc_439_;
goto v_reusejp_437_;
}
v_reusejp_437_:
{
return v___x_438_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instCommMonoidWithZero(lean_object* v_00_u03b1_441_, lean_object* v_inst_442_, lean_object* v_inst_443_, lean_object* v_inst_444_, lean_object* v_inst_445_){
_start:
{
lean_object* v___x_446_; 
v___x_446_ = lp_mathlib_WithTop_instCommMonoidWithZero___redArg(v_inst_442_, v_inst_443_);
return v___x_446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instCommMonoidWithZero___redArg(lean_object* v_inst_447_, lean_object* v_inst_448_){
_start:
{
lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v_toMonoid_451_; lean_object* v_toZero_452_; lean_object* v___x_454_; uint8_t v_isShared_455_; uint8_t v_isSharedCheck_459_; 
v___x_449_ = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(v_inst_448_);
v___x_450_ = lp_mathlib_WithBot_instMonoidWithZero___redArg(v_inst_447_, v___x_449_);
v_toMonoid_451_ = lean_ctor_get(v___x_450_, 0);
v_toZero_452_ = lean_ctor_get(v___x_450_, 1);
v_isSharedCheck_459_ = !lean_is_exclusive(v___x_450_);
if (v_isSharedCheck_459_ == 0)
{
v___x_454_ = v___x_450_;
v_isShared_455_ = v_isSharedCheck_459_;
goto v_resetjp_453_;
}
else
{
lean_inc(v_toZero_452_);
lean_inc(v_toMonoid_451_);
lean_dec(v___x_450_);
v___x_454_ = lean_box(0);
v_isShared_455_ = v_isSharedCheck_459_;
goto v_resetjp_453_;
}
v_resetjp_453_:
{
lean_object* v___x_457_; 
if (v_isShared_455_ == 0)
{
v___x_457_ = v___x_454_;
goto v_reusejp_456_;
}
else
{
lean_object* v_reuseFailAlloc_458_; 
v_reuseFailAlloc_458_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_458_, 0, v_toMonoid_451_);
lean_ctor_set(v_reuseFailAlloc_458_, 1, v_toZero_452_);
v___x_457_ = v_reuseFailAlloc_458_;
goto v_reusejp_456_;
}
v_reusejp_456_:
{
return v___x_457_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instCommMonoidWithZero(lean_object* v_00_u03b1_460_, lean_object* v_inst_461_, lean_object* v_inst_462_, lean_object* v_inst_463_, lean_object* v_inst_464_){
_start:
{
lean_object* v___x_465_; 
v___x_465_ = lp_mathlib_WithBot_instCommMonoidWithZero___redArg(v_inst_461_, v_inst_462_);
return v___x_465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instNonUnitalNonAssocSemiring___redArg(lean_object* v_inst_466_, lean_object* v_inst_467_){
_start:
{
lean_object* v_toAddCommMonoid_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v_toMul_472_; lean_object* v___x_474_; uint8_t v_isShared_475_; uint8_t v_isSharedCheck_479_; 
v_toAddCommMonoid_468_ = lean_ctor_get(v_inst_467_, 0);
lean_inc_ref(v_toAddCommMonoid_468_);
v___x_469_ = lp_mathlib_WithTop_addMonoid___redArg(v_toAddCommMonoid_468_);
v___x_470_ = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(v_inst_467_);
v___x_471_ = lp_mathlib_WithTop_instMulZeroClass___redArg(v_inst_466_, v___x_470_);
v_toMul_472_ = lean_ctor_get(v___x_471_, 0);
v_isSharedCheck_479_ = !lean_is_exclusive(v___x_471_);
if (v_isSharedCheck_479_ == 0)
{
lean_object* v_unused_480_; 
v_unused_480_ = lean_ctor_get(v___x_471_, 1);
lean_dec(v_unused_480_);
v___x_474_ = v___x_471_;
v_isShared_475_ = v_isSharedCheck_479_;
goto v_resetjp_473_;
}
else
{
lean_inc(v_toMul_472_);
lean_dec(v___x_471_);
v___x_474_ = lean_box(0);
v_isShared_475_ = v_isSharedCheck_479_;
goto v_resetjp_473_;
}
v_resetjp_473_:
{
lean_object* v___x_477_; 
if (v_isShared_475_ == 0)
{
lean_ctor_set(v___x_474_, 1, v_toMul_472_);
lean_ctor_set(v___x_474_, 0, v___x_469_);
v___x_477_ = v___x_474_;
goto v_reusejp_476_;
}
else
{
lean_object* v_reuseFailAlloc_478_; 
v_reuseFailAlloc_478_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_478_, 0, v___x_469_);
lean_ctor_set(v_reuseFailAlloc_478_, 1, v_toMul_472_);
v___x_477_ = v_reuseFailAlloc_478_;
goto v_reusejp_476_;
}
v_reusejp_476_:
{
return v___x_477_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instNonUnitalNonAssocSemiring(lean_object* v_00_u03b1_481_, lean_object* v_inst_482_, lean_object* v_inst_483_, lean_object* v_inst_484_){
_start:
{
lean_object* v___x_485_; 
v___x_485_ = lp_mathlib_WithTop_instNonUnitalNonAssocSemiring___redArg(v_inst_482_, v_inst_483_);
return v___x_485_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instNonUnitalNonAssocSemiring___redArg(lean_object* v_inst_486_, lean_object* v_inst_487_){
_start:
{
lean_object* v_toAddCommMonoid_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v_toMul_492_; lean_object* v___x_494_; uint8_t v_isShared_495_; uint8_t v_isSharedCheck_499_; 
v_toAddCommMonoid_488_ = lean_ctor_get(v_inst_487_, 0);
lean_inc_ref(v_toAddCommMonoid_488_);
v___x_489_ = lp_mathlib_WithBot_addMonoid___redArg(v_toAddCommMonoid_488_);
v___x_490_ = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(v_inst_487_);
v___x_491_ = lp_mathlib_WithBot_instMulZeroClass___redArg(v_inst_486_, v___x_490_);
v_toMul_492_ = lean_ctor_get(v___x_491_, 0);
v_isSharedCheck_499_ = !lean_is_exclusive(v___x_491_);
if (v_isSharedCheck_499_ == 0)
{
lean_object* v_unused_500_; 
v_unused_500_ = lean_ctor_get(v___x_491_, 1);
lean_dec(v_unused_500_);
v___x_494_ = v___x_491_;
v_isShared_495_ = v_isSharedCheck_499_;
goto v_resetjp_493_;
}
else
{
lean_inc(v_toMul_492_);
lean_dec(v___x_491_);
v___x_494_ = lean_box(0);
v_isShared_495_ = v_isSharedCheck_499_;
goto v_resetjp_493_;
}
v_resetjp_493_:
{
lean_object* v___x_497_; 
if (v_isShared_495_ == 0)
{
lean_ctor_set(v___x_494_, 1, v_toMul_492_);
lean_ctor_set(v___x_494_, 0, v___x_489_);
v___x_497_ = v___x_494_;
goto v_reusejp_496_;
}
else
{
lean_object* v_reuseFailAlloc_498_; 
v_reuseFailAlloc_498_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_498_, 0, v___x_489_);
lean_ctor_set(v_reuseFailAlloc_498_, 1, v_toMul_492_);
v___x_497_ = v_reuseFailAlloc_498_;
goto v_reusejp_496_;
}
v_reusejp_496_:
{
return v___x_497_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instNonUnitalNonAssocSemiring(lean_object* v_00_u03b1_501_, lean_object* v_inst_502_, lean_object* v_inst_503_, lean_object* v_inst_504_){
_start:
{
lean_object* v___x_505_; 
v___x_505_ = lp_mathlib_WithBot_instNonUnitalNonAssocSemiring___redArg(v_inst_502_, v_inst_503_);
return v___x_505_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instNonAssocSemiring___redArg(lean_object* v_inst_506_, lean_object* v_inst_507_){
_start:
{
lean_object* v_toNonUnitalNonAssocSemiring_508_; lean_object* v___x_509_; lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v_toMulOneClass_512_; lean_object* v_toOne_513_; lean_object* v___x_514_; lean_object* v___x_515_; lean_object* v_toNatCast_516_; lean_object* v___x_518_; uint8_t v_isShared_519_; uint8_t v_isSharedCheck_523_; 
v_toNonUnitalNonAssocSemiring_508_ = lean_ctor_get(v_inst_507_, 0);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_508_);
lean_inc_ref(v_inst_506_);
v___x_509_ = lp_mathlib_WithTop_instNonUnitalNonAssocSemiring___redArg(v_inst_506_, v_toNonUnitalNonAssocSemiring_508_);
lean_inc_ref(v_inst_507_);
v___x_510_ = lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(v_inst_507_);
v___x_511_ = lp_mathlib_WithTop_instMulZeroOneClass___redArg(v_inst_506_, v___x_510_);
v_toMulOneClass_512_ = lean_ctor_get(v___x_511_, 0);
lean_inc_ref(v_toMulOneClass_512_);
lean_dec_ref(v___x_511_);
v_toOne_513_ = lean_ctor_get(v_toMulOneClass_512_, 0);
lean_inc(v_toOne_513_);
lean_dec_ref(v_toMulOneClass_512_);
v___x_514_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v_inst_507_);
v___x_515_ = lp_mathlib_WithTop_addMonoidWithOne___redArg(v___x_514_);
v_toNatCast_516_ = lean_ctor_get(v___x_515_, 0);
v_isSharedCheck_523_ = !lean_is_exclusive(v___x_515_);
if (v_isSharedCheck_523_ == 0)
{
lean_object* v_unused_524_; lean_object* v_unused_525_; 
v_unused_524_ = lean_ctor_get(v___x_515_, 2);
lean_dec(v_unused_524_);
v_unused_525_ = lean_ctor_get(v___x_515_, 1);
lean_dec(v_unused_525_);
v___x_518_ = v___x_515_;
v_isShared_519_ = v_isSharedCheck_523_;
goto v_resetjp_517_;
}
else
{
lean_inc(v_toNatCast_516_);
lean_dec(v___x_515_);
v___x_518_ = lean_box(0);
v_isShared_519_ = v_isSharedCheck_523_;
goto v_resetjp_517_;
}
v_resetjp_517_:
{
lean_object* v___x_521_; 
if (v_isShared_519_ == 0)
{
lean_ctor_set(v___x_518_, 2, v_toNatCast_516_);
lean_ctor_set(v___x_518_, 1, v_toOne_513_);
lean_ctor_set(v___x_518_, 0, v___x_509_);
v___x_521_ = v___x_518_;
goto v_reusejp_520_;
}
else
{
lean_object* v_reuseFailAlloc_522_; 
v_reuseFailAlloc_522_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_522_, 0, v___x_509_);
lean_ctor_set(v_reuseFailAlloc_522_, 1, v_toOne_513_);
lean_ctor_set(v_reuseFailAlloc_522_, 2, v_toNatCast_516_);
v___x_521_ = v_reuseFailAlloc_522_;
goto v_reusejp_520_;
}
v_reusejp_520_:
{
return v___x_521_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instNonAssocSemiring(lean_object* v_00_u03b1_526_, lean_object* v_inst_527_, lean_object* v_inst_528_, lean_object* v_inst_529_, lean_object* v_inst_530_){
_start:
{
lean_object* v___x_531_; 
v___x_531_ = lp_mathlib_WithTop_instNonAssocSemiring___redArg(v_inst_527_, v_inst_528_);
return v___x_531_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instNonAssocSemiring___redArg(lean_object* v_inst_532_, lean_object* v_inst_533_){
_start:
{
lean_object* v_toNonUnitalNonAssocSemiring_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v_toMulOneClass_538_; lean_object* v_toOne_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v_toNatCast_542_; lean_object* v___x_544_; uint8_t v_isShared_545_; uint8_t v_isSharedCheck_549_; 
v_toNonUnitalNonAssocSemiring_534_ = lean_ctor_get(v_inst_533_, 0);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_534_);
lean_inc_ref(v_inst_532_);
v___x_535_ = lp_mathlib_WithBot_instNonUnitalNonAssocSemiring___redArg(v_inst_532_, v_toNonUnitalNonAssocSemiring_534_);
lean_inc_ref(v_inst_533_);
v___x_536_ = lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(v_inst_533_);
v___x_537_ = lp_mathlib_WithBot_instMulZeroOneClass___redArg(v_inst_532_, v___x_536_);
v_toMulOneClass_538_ = lean_ctor_get(v___x_537_, 0);
lean_inc_ref(v_toMulOneClass_538_);
lean_dec_ref(v___x_537_);
v_toOne_539_ = lean_ctor_get(v_toMulOneClass_538_, 0);
lean_inc(v_toOne_539_);
lean_dec_ref(v_toMulOneClass_538_);
v___x_540_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v_inst_533_);
v___x_541_ = lp_mathlib_WithBot_addMonoidWithOne___redArg(v___x_540_);
v_toNatCast_542_ = lean_ctor_get(v___x_541_, 0);
v_isSharedCheck_549_ = !lean_is_exclusive(v___x_541_);
if (v_isSharedCheck_549_ == 0)
{
lean_object* v_unused_550_; lean_object* v_unused_551_; 
v_unused_550_ = lean_ctor_get(v___x_541_, 2);
lean_dec(v_unused_550_);
v_unused_551_ = lean_ctor_get(v___x_541_, 1);
lean_dec(v_unused_551_);
v___x_544_ = v___x_541_;
v_isShared_545_ = v_isSharedCheck_549_;
goto v_resetjp_543_;
}
else
{
lean_inc(v_toNatCast_542_);
lean_dec(v___x_541_);
v___x_544_ = lean_box(0);
v_isShared_545_ = v_isSharedCheck_549_;
goto v_resetjp_543_;
}
v_resetjp_543_:
{
lean_object* v___x_547_; 
if (v_isShared_545_ == 0)
{
lean_ctor_set(v___x_544_, 2, v_toNatCast_542_);
lean_ctor_set(v___x_544_, 1, v_toOne_539_);
lean_ctor_set(v___x_544_, 0, v___x_535_);
v___x_547_ = v___x_544_;
goto v_reusejp_546_;
}
else
{
lean_object* v_reuseFailAlloc_548_; 
v_reuseFailAlloc_548_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_548_, 0, v___x_535_);
lean_ctor_set(v_reuseFailAlloc_548_, 1, v_toOne_539_);
lean_ctor_set(v_reuseFailAlloc_548_, 2, v_toNatCast_542_);
v___x_547_ = v_reuseFailAlloc_548_;
goto v_reusejp_546_;
}
v_reusejp_546_:
{
return v___x_547_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instNonAssocSemiring(lean_object* v_00_u03b1_552_, lean_object* v_inst_553_, lean_object* v_inst_554_, lean_object* v_inst_555_, lean_object* v_inst_556_){
_start:
{
lean_object* v___x_557_; 
v___x_557_ = lp_mathlib_WithBot_instNonAssocSemiring___redArg(v_inst_553_, v_inst_554_);
return v___x_557_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instNonUnitalSemiring___redArg(lean_object* v_inst_558_, lean_object* v_inst_559_){
_start:
{
lean_object* v___x_560_; 
v___x_560_ = lp_mathlib_WithTop_instNonUnitalNonAssocSemiring___redArg(v_inst_558_, v_inst_559_);
return v___x_560_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instNonUnitalSemiring(lean_object* v_00_u03b1_561_, lean_object* v_inst_562_, lean_object* v_inst_563_, lean_object* v_inst_564_, lean_object* v_inst_565_){
_start:
{
lean_object* v___x_566_; 
v___x_566_ = lp_mathlib_WithTop_instNonUnitalNonAssocSemiring___redArg(v_inst_562_, v_inst_563_);
return v___x_566_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instNonUnitalSemiring___redArg(lean_object* v_inst_567_, lean_object* v_inst_568_){
_start:
{
lean_object* v___x_569_; 
v___x_569_ = lp_mathlib_WithBot_instNonUnitalNonAssocSemiring___redArg(v_inst_567_, v_inst_568_);
return v___x_569_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instNonUnitalSemiring(lean_object* v_00_u03b1_570_, lean_object* v_inst_571_, lean_object* v_inst_572_, lean_object* v_inst_573_, lean_object* v_inst_574_){
_start:
{
lean_object* v___x_575_; 
v___x_575_ = lp_mathlib_WithBot_instNonUnitalNonAssocSemiring___redArg(v_inst_571_, v_inst_572_);
return v___x_575_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instSemiring___redArg(lean_object* v_inst_576_, lean_object* v_inst_577_){
_start:
{
lean_object* v_toAddCommMonoid_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v_toMonoid_582_; lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v_toNatCast_585_; lean_object* v___x_587_; uint8_t v_isShared_588_; uint8_t v_isSharedCheck_592_; 
v_toAddCommMonoid_578_ = lean_ctor_get(v_inst_577_, 0);
lean_inc_ref(v_toAddCommMonoid_578_);
v___x_579_ = lp_mathlib_WithTop_addMonoid___redArg(v_toAddCommMonoid_578_);
v___x_580_ = lp_mathlib_Semiring_toMonoidWithZero___redArg(v_inst_577_);
lean_inc_ref(v_inst_576_);
v___x_581_ = lp_mathlib_WithTop_instMonoidWithZero___redArg(v_inst_576_, v___x_580_);
v_toMonoid_582_ = lean_ctor_get(v___x_581_, 0);
lean_inc_ref(v_toMonoid_582_);
lean_dec_ref(v___x_581_);
v___x_583_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_577_);
v___x_584_ = lp_mathlib_WithTop_instNonAssocSemiring___redArg(v_inst_576_, v___x_583_);
v_toNatCast_585_ = lean_ctor_get(v___x_584_, 2);
v_isSharedCheck_592_ = !lean_is_exclusive(v___x_584_);
if (v_isSharedCheck_592_ == 0)
{
lean_object* v_unused_593_; lean_object* v_unused_594_; 
v_unused_593_ = lean_ctor_get(v___x_584_, 1);
lean_dec(v_unused_593_);
v_unused_594_ = lean_ctor_get(v___x_584_, 0);
lean_dec(v_unused_594_);
v___x_587_ = v___x_584_;
v_isShared_588_ = v_isSharedCheck_592_;
goto v_resetjp_586_;
}
else
{
lean_inc(v_toNatCast_585_);
lean_dec(v___x_584_);
v___x_587_ = lean_box(0);
v_isShared_588_ = v_isSharedCheck_592_;
goto v_resetjp_586_;
}
v_resetjp_586_:
{
lean_object* v___x_590_; 
if (v_isShared_588_ == 0)
{
lean_ctor_set(v___x_587_, 1, v_toMonoid_582_);
lean_ctor_set(v___x_587_, 0, v___x_579_);
v___x_590_ = v___x_587_;
goto v_reusejp_589_;
}
else
{
lean_object* v_reuseFailAlloc_591_; 
v_reuseFailAlloc_591_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_591_, 0, v___x_579_);
lean_ctor_set(v_reuseFailAlloc_591_, 1, v_toMonoid_582_);
lean_ctor_set(v_reuseFailAlloc_591_, 2, v_toNatCast_585_);
v___x_590_ = v_reuseFailAlloc_591_;
goto v_reusejp_589_;
}
v_reusejp_589_:
{
return v___x_590_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instSemiring(lean_object* v_00_u03b1_595_, lean_object* v_inst_596_, lean_object* v_inst_597_, lean_object* v_inst_598_, lean_object* v_inst_599_, lean_object* v_inst_600_){
_start:
{
lean_object* v___x_601_; 
v___x_601_ = lp_mathlib_WithTop_instSemiring___redArg(v_inst_596_, v_inst_597_);
return v___x_601_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instSemiring___redArg(lean_object* v_inst_602_, lean_object* v_inst_603_){
_start:
{
lean_object* v_toAddCommMonoid_604_; lean_object* v___x_605_; lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v_toMonoid_608_; lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v_toNatCast_611_; lean_object* v___x_613_; uint8_t v_isShared_614_; uint8_t v_isSharedCheck_618_; 
v_toAddCommMonoid_604_ = lean_ctor_get(v_inst_603_, 0);
lean_inc_ref(v_toAddCommMonoid_604_);
v___x_605_ = lp_mathlib_WithBot_addMonoid___redArg(v_toAddCommMonoid_604_);
v___x_606_ = lp_mathlib_Semiring_toMonoidWithZero___redArg(v_inst_603_);
lean_inc_ref(v_inst_602_);
v___x_607_ = lp_mathlib_WithBot_instMonoidWithZero___redArg(v_inst_602_, v___x_606_);
v_toMonoid_608_ = lean_ctor_get(v___x_607_, 0);
lean_inc_ref(v_toMonoid_608_);
lean_dec_ref(v___x_607_);
v___x_609_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_603_);
v___x_610_ = lp_mathlib_WithBot_instNonAssocSemiring___redArg(v_inst_602_, v___x_609_);
v_toNatCast_611_ = lean_ctor_get(v___x_610_, 2);
v_isSharedCheck_618_ = !lean_is_exclusive(v___x_610_);
if (v_isSharedCheck_618_ == 0)
{
lean_object* v_unused_619_; lean_object* v_unused_620_; 
v_unused_619_ = lean_ctor_get(v___x_610_, 1);
lean_dec(v_unused_619_);
v_unused_620_ = lean_ctor_get(v___x_610_, 0);
lean_dec(v_unused_620_);
v___x_613_ = v___x_610_;
v_isShared_614_ = v_isSharedCheck_618_;
goto v_resetjp_612_;
}
else
{
lean_inc(v_toNatCast_611_);
lean_dec(v___x_610_);
v___x_613_ = lean_box(0);
v_isShared_614_ = v_isSharedCheck_618_;
goto v_resetjp_612_;
}
v_resetjp_612_:
{
lean_object* v___x_616_; 
if (v_isShared_614_ == 0)
{
lean_ctor_set(v___x_613_, 1, v_toMonoid_608_);
lean_ctor_set(v___x_613_, 0, v___x_605_);
v___x_616_ = v___x_613_;
goto v_reusejp_615_;
}
else
{
lean_object* v_reuseFailAlloc_617_; 
v_reuseFailAlloc_617_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_617_, 0, v___x_605_);
lean_ctor_set(v_reuseFailAlloc_617_, 1, v_toMonoid_608_);
lean_ctor_set(v_reuseFailAlloc_617_, 2, v_toNatCast_611_);
v___x_616_ = v_reuseFailAlloc_617_;
goto v_reusejp_615_;
}
v_reusejp_615_:
{
return v___x_616_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instSemiring(lean_object* v_00_u03b1_621_, lean_object* v_inst_622_, lean_object* v_inst_623_, lean_object* v_inst_624_, lean_object* v_inst_625_, lean_object* v_inst_626_){
_start:
{
lean_object* v___x_627_; 
v___x_627_ = lp_mathlib_WithBot_instSemiring___redArg(v_inst_622_, v_inst_623_);
return v___x_627_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instCommSemiring___redArg(lean_object* v_inst_628_, lean_object* v_inst_629_){
_start:
{
lean_object* v___x_630_; 
v___x_630_ = lp_mathlib_WithTop_instSemiring___redArg(v_inst_628_, v_inst_629_);
return v___x_630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instCommSemiring(lean_object* v_00_u03b1_631_, lean_object* v_inst_632_, lean_object* v_inst_633_, lean_object* v_inst_634_, lean_object* v_inst_635_, lean_object* v_inst_636_){
_start:
{
lean_object* v___x_637_; 
v___x_637_ = lp_mathlib_WithTop_instSemiring___redArg(v_inst_632_, v_inst_633_);
return v___x_637_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instCommSemiring___redArg(lean_object* v_inst_638_, lean_object* v_inst_639_){
_start:
{
lean_object* v___x_640_; 
v___x_640_ = lp_mathlib_WithBot_instSemiring___redArg(v_inst_638_, v_inst_639_);
return v___x_640_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instCommSemiring(lean_object* v_00_u03b1_641_, lean_object* v_inst_642_, lean_object* v_inst_643_, lean_object* v_inst_644_, lean_object* v_inst_645_, lean_object* v_inst_646_){
_start:
{
lean_object* v___x_647_; 
v___x_647_ = lp_mathlib_WithBot_instSemiring___redArg(v_inst_642_, v_inst_643_);
return v___x_647_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_withTopMap___redArg(lean_object* v_f_648_){
_start:
{
lean_object* v___x_649_; 
v___x_649_ = lp_mathlib_MonoidWithZeroHom_withTopMap___redArg(v_f_648_);
return v___x_649_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_withTopMap(lean_object* v_R_650_, lean_object* v_S_651_, lean_object* v_inst_652_, lean_object* v_inst_653_, lean_object* v_inst_654_, lean_object* v_inst_655_, lean_object* v_inst_656_, lean_object* v_inst_657_, lean_object* v_inst_658_, lean_object* v_inst_659_, lean_object* v_inst_660_, lean_object* v_inst_661_, lean_object* v_f_662_, lean_object* v_hf_663_){
_start:
{
lean_object* v___x_664_; 
v___x_664_ = lp_mathlib_MonoidWithZeroHom_withTopMap___redArg(v_f_662_);
return v___x_664_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_withTopMap___boxed(lean_object* v_R_665_, lean_object* v_S_666_, lean_object* v_inst_667_, lean_object* v_inst_668_, lean_object* v_inst_669_, lean_object* v_inst_670_, lean_object* v_inst_671_, lean_object* v_inst_672_, lean_object* v_inst_673_, lean_object* v_inst_674_, lean_object* v_inst_675_, lean_object* v_inst_676_, lean_object* v_f_677_, lean_object* v_hf_678_){
_start:
{
lean_object* v_res_679_; 
v_res_679_ = lp_mathlib_RingHom_withTopMap(v_R_665_, v_S_666_, v_inst_667_, v_inst_668_, v_inst_669_, v_inst_670_, v_inst_671_, v_inst_672_, v_inst_673_, v_inst_674_, v_inst_675_, v_inst_676_, v_f_677_, v_hf_678_);
lean_dec_ref(v_inst_675_);
lean_dec_ref(v_inst_673_);
lean_dec_ref(v_inst_672_);
lean_dec_ref(v_inst_670_);
lean_dec_ref(v_inst_668_);
lean_dec_ref(v_inst_667_);
return v_res_679_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_withBotMap___redArg(lean_object* v_f_680_){
_start:
{
lean_object* v___x_681_; 
v___x_681_ = lp_mathlib_MonoidWithZeroHom_withBotMap___redArg(v_f_680_);
return v___x_681_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_withBotMap(lean_object* v_R_682_, lean_object* v_S_683_, lean_object* v_inst_684_, lean_object* v_inst_685_, lean_object* v_inst_686_, lean_object* v_inst_687_, lean_object* v_inst_688_, lean_object* v_inst_689_, lean_object* v_inst_690_, lean_object* v_inst_691_, lean_object* v_inst_692_, lean_object* v_inst_693_, lean_object* v_f_694_, lean_object* v_hf_695_){
_start:
{
lean_object* v___x_696_; 
v___x_696_ = lp_mathlib_MonoidWithZeroHom_withBotMap___redArg(v_f_694_);
return v___x_696_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_withBotMap___boxed(lean_object* v_R_697_, lean_object* v_S_698_, lean_object* v_inst_699_, lean_object* v_inst_700_, lean_object* v_inst_701_, lean_object* v_inst_702_, lean_object* v_inst_703_, lean_object* v_inst_704_, lean_object* v_inst_705_, lean_object* v_inst_706_, lean_object* v_inst_707_, lean_object* v_inst_708_, lean_object* v_f_709_, lean_object* v_hf_710_){
_start:
{
lean_object* v_res_711_; 
v_res_711_ = lp_mathlib_RingHom_withBotMap(v_R_697_, v_S_698_, v_inst_699_, v_inst_700_, v_inst_701_, v_inst_702_, v_inst_703_, v_inst_704_, v_inst_705_, v_inst_706_, v_inst_707_, v_inst_708_, v_f_709_, v_hf_710_);
lean_dec_ref(v_inst_707_);
lean_dec_ref(v_inst_705_);
lean_dec_ref(v_inst_704_);
lean_dec_ref(v_inst_702_);
lean_dec_ref(v_inst_700_);
lean_dec_ref(v_inst_699_);
return v_res_711_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Canonical(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_WithTop(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_WithTop(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Canonical(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Monoid_WithTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Order_Ring_WithTop(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Canonical(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Monoid_WithTop(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_WithTop(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Canonical(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Monoid_WithTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_WithTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Order_Ring_WithTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Order_Ring_WithTop(builtin);
}
#ifdef __cplusplus
}
#endif
