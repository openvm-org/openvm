// Lean compiler output
// Module: Mathlib.Algebra.Ring.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.IsCommutative public import Mathlib.Algebra.GroupWithZero.Defs public import Mathlib.Algebra.Notation.Defs public import Mathlib.Data.Int.Cast.Defs public import Mathlib.Tactic.CrossRefAttribute public import Mathlib.Tactic.Spread
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
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSemiring_toSemigroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSemiring_toSemigroupWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocSemiring_toMulZeroOneClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRing_toNonUnitalSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRing_toNonUnitalSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocRing_toNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocRing_toNonAssocSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocRing_toAddCommGroupWithOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toMonoidWithZero___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toMonoidWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toMonoidWithZero___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toNonUnitalSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toNonUnitalSemiring___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toNonUnitalSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toNonUnitalSemiring___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toNonAssocSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_toAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_toAddCommGroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_toAddCommGroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_toAddCommGroup___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_toAddGroupWithOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDistribOfSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDistribOfSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulZeroClassOfSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulZeroClassOfSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulZeroOneClassOfSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulZeroOneClassOfSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocCommSemiring_toCommMagma___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocCommSemiring_toCommMagma___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocCommSemiring_toCommMagma(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocCommSemiring_toCommMagma___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toCommSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toCommSemigroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toCommSemigroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toCommSemigroup___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommSemiring_toNonUnitalNonAssocCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommSemiring_toNonUnitalNonAssocCommSemiring___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommSemiring_toNonUnitalNonAssocCommSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommSemiring_toNonUnitalNonAssocCommSemiring___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toCommMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toCommMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toNonUnitalNonAssocCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toNonUnitalNonAssocCommSemiring___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toNonUnitalNonAssocCommSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toNonUnitalNonAssocCommSemiring___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toNonAssocCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toNonAssocCommSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toNonUnitalCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toNonUnitalCommSemiring___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toNonUnitalCommSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toNonUnitalCommSemiring___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toCommMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toCommMonoidWithZero___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toCommMonoidWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toCommMonoidWithZero___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulZeroClass_negZeroClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulZeroClass_negZeroClass(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocRing_toHasDistribNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocRing_toHasDistribNeg___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocRing_toHasDistribNeg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocRing_toHasDistribNeg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_toNonUnitalRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_toNonUnitalRing___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_toNonUnitalRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_toNonUnitalRing___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_toNonAssocRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_toNonAssocRing___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_toNonAssocRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_toNonAssocRing___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocCommRing_toNonUnitalNonAssocCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocCommRing_toNonUnitalNonAssocCommSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommRing_toNonUnitalNonAssocCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommRing_toNonUnitalNonAssocCommRing___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommRing_toNonUnitalNonAssocCommRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommRing_toNonUnitalNonAssocCommRing___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommRing_toNonUnitalNonAssocCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommRing_toNonUnitalNonAssocCommRing___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommRing_toNonUnitalNonAssocCommRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommRing_toNonUnitalNonAssocCommRing___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommRing_toNonAssocCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommRing_toNonAssocCommSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommRing_toNonUnitalCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommRing_toNonUnitalCommSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toNonAssocCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toNonAssocCommRing___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toNonAssocCommRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toNonAssocCommRing___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommSemiring___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommSemiring___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toNonUnitalCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toNonUnitalCommRing___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toNonUnitalCommRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toNonUnitalCommRing___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toAddCommGroupWithOne___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toAddCommGroupWithOne___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toAddCommGroupWithOne(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toAddCommGroupWithOne___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalNonAssocCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalNonAssocCommSemiring___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalNonAssocCommSemiring(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalNonAssocCommSemiring___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalCommSemiring___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalCommSemiring(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalCommSemiring___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalNonAssocCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalNonAssocCommRing___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalNonAssocCommRing(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalNonAssocCommRing___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalCommRing___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalCommRing(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalCommRing___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonAssocCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonAssocCommSemiring___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonAssocCommSemiring(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonAssocCommSemiring___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instCommSemiring___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instCommSemiring(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instCommSemiring___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonAssocCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonAssocCommRing___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonAssocCommRing(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonAssocCommRing___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instCommRing___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instCommRing(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instCommRing___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object* v_self_1_){
_start:
{
lean_object* v_toAddCommMonoid_2_; lean_object* v_toMul_3_; lean_object* v___x_5_; uint8_t v_isShared_6_; uint8_t v_isSharedCheck_11_; 
v_toAddCommMonoid_2_ = lean_ctor_get(v_self_1_, 0);
v_toMul_3_ = lean_ctor_get(v_self_1_, 1);
v_isSharedCheck_11_ = !lean_is_exclusive(v_self_1_);
if (v_isSharedCheck_11_ == 0)
{
v___x_5_ = v_self_1_;
v_isShared_6_ = v_isSharedCheck_11_;
goto v_resetjp_4_;
}
else
{
lean_inc(v_toMul_3_);
lean_inc(v_toAddCommMonoid_2_);
lean_dec(v_self_1_);
v___x_5_ = lean_box(0);
v_isShared_6_ = v_isSharedCheck_11_;
goto v_resetjp_4_;
}
v_resetjp_4_:
{
lean_object* v_toAdd_7_; lean_object* v___x_9_; 
v_toAdd_7_ = lean_ctor_get(v_toAddCommMonoid_2_, 1);
lean_inc(v_toAdd_7_);
lean_dec_ref(v_toAddCommMonoid_2_);
if (v_isShared_6_ == 0)
{
lean_ctor_set(v___x_5_, 1, v_toAdd_7_);
lean_ctor_set(v___x_5_, 0, v_toMul_3_);
v___x_9_ = v___x_5_;
goto v_reusejp_8_;
}
else
{
lean_object* v_reuseFailAlloc_10_; 
v_reuseFailAlloc_10_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_10_, 0, v_toMul_3_);
lean_ctor_set(v_reuseFailAlloc_10_, 1, v_toAdd_7_);
v___x_9_ = v_reuseFailAlloc_10_;
goto v_reusejp_8_;
}
v_reusejp_8_:
{
return v___x_9_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib(lean_object* v_00_u03b1_12_, lean_object* v_self_13_){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v_self_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(lean_object* v_self_15_){
_start:
{
lean_object* v_toAddCommMonoid_16_; lean_object* v_toMul_17_; lean_object* v___x_19_; uint8_t v_isShared_20_; uint8_t v_isSharedCheck_25_; 
v_toAddCommMonoid_16_ = lean_ctor_get(v_self_15_, 0);
v_toMul_17_ = lean_ctor_get(v_self_15_, 1);
v_isSharedCheck_25_ = !lean_is_exclusive(v_self_15_);
if (v_isSharedCheck_25_ == 0)
{
v___x_19_ = v_self_15_;
v_isShared_20_ = v_isSharedCheck_25_;
goto v_resetjp_18_;
}
else
{
lean_inc(v_toMul_17_);
lean_inc(v_toAddCommMonoid_16_);
lean_dec(v_self_15_);
v___x_19_ = lean_box(0);
v_isShared_20_ = v_isSharedCheck_25_;
goto v_resetjp_18_;
}
v_resetjp_18_:
{
lean_object* v_toZero_21_; lean_object* v___x_23_; 
v_toZero_21_ = lean_ctor_get(v_toAddCommMonoid_16_, 0);
lean_inc(v_toZero_21_);
lean_dec_ref(v_toAddCommMonoid_16_);
if (v_isShared_20_ == 0)
{
lean_ctor_set(v___x_19_, 1, v_toZero_21_);
lean_ctor_set(v___x_19_, 0, v_toMul_17_);
v___x_23_ = v___x_19_;
goto v_reusejp_22_;
}
else
{
lean_object* v_reuseFailAlloc_24_; 
v_reuseFailAlloc_24_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_24_, 0, v_toMul_17_);
lean_ctor_set(v_reuseFailAlloc_24_, 1, v_toZero_21_);
v___x_23_ = v_reuseFailAlloc_24_;
goto v_reusejp_22_;
}
v_reusejp_22_:
{
return v___x_23_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass(lean_object* v_00_u03b1_26_, lean_object* v_self_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(v_self_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSemiring_toSemigroupWithZero___redArg(lean_object* v_self_29_){
_start:
{
lean_object* v_toAddCommMonoid_30_; lean_object* v_toMul_31_; lean_object* v___x_33_; uint8_t v_isShared_34_; uint8_t v_isSharedCheck_39_; 
v_toAddCommMonoid_30_ = lean_ctor_get(v_self_29_, 0);
v_toMul_31_ = lean_ctor_get(v_self_29_, 1);
v_isSharedCheck_39_ = !lean_is_exclusive(v_self_29_);
if (v_isSharedCheck_39_ == 0)
{
v___x_33_ = v_self_29_;
v_isShared_34_ = v_isSharedCheck_39_;
goto v_resetjp_32_;
}
else
{
lean_inc(v_toMul_31_);
lean_inc(v_toAddCommMonoid_30_);
lean_dec(v_self_29_);
v___x_33_ = lean_box(0);
v_isShared_34_ = v_isSharedCheck_39_;
goto v_resetjp_32_;
}
v_resetjp_32_:
{
lean_object* v_toZero_35_; lean_object* v___x_37_; 
v_toZero_35_ = lean_ctor_get(v_toAddCommMonoid_30_, 0);
lean_inc(v_toZero_35_);
lean_dec_ref(v_toAddCommMonoid_30_);
if (v_isShared_34_ == 0)
{
lean_ctor_set(v___x_33_, 1, v_toZero_35_);
lean_ctor_set(v___x_33_, 0, v_toMul_31_);
v___x_37_ = v___x_33_;
goto v_reusejp_36_;
}
else
{
lean_object* v_reuseFailAlloc_38_; 
v_reuseFailAlloc_38_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_38_, 0, v_toMul_31_);
lean_ctor_set(v_reuseFailAlloc_38_, 1, v_toZero_35_);
v___x_37_ = v_reuseFailAlloc_38_;
goto v_reusejp_36_;
}
v_reusejp_36_:
{
return v___x_37_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalSemiring_toSemigroupWithZero(lean_object* v_00_u03b1_40_, lean_object* v_self_41_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = lp_mathlib_NonUnitalSemiring_toSemigroupWithZero___redArg(v_self_41_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(lean_object* v_self_43_){
_start:
{
lean_object* v_toNonUnitalNonAssocSemiring_44_; lean_object* v_toOne_45_; lean_object* v_toAddCommMonoid_46_; lean_object* v_toMul_47_; lean_object* v___x_49_; uint8_t v_isShared_50_; uint8_t v_isSharedCheck_56_; 
v_toNonUnitalNonAssocSemiring_44_ = lean_ctor_get(v_self_43_, 0);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_44_);
v_toOne_45_ = lean_ctor_get(v_self_43_, 1);
lean_inc(v_toOne_45_);
lean_dec_ref(v_self_43_);
v_toAddCommMonoid_46_ = lean_ctor_get(v_toNonUnitalNonAssocSemiring_44_, 0);
v_toMul_47_ = lean_ctor_get(v_toNonUnitalNonAssocSemiring_44_, 1);
v_isSharedCheck_56_ = !lean_is_exclusive(v_toNonUnitalNonAssocSemiring_44_);
if (v_isSharedCheck_56_ == 0)
{
v___x_49_ = v_toNonUnitalNonAssocSemiring_44_;
v_isShared_50_ = v_isSharedCheck_56_;
goto v_resetjp_48_;
}
else
{
lean_inc(v_toMul_47_);
lean_inc(v_toAddCommMonoid_46_);
lean_dec(v_toNonUnitalNonAssocSemiring_44_);
v___x_49_ = lean_box(0);
v_isShared_50_ = v_isSharedCheck_56_;
goto v_resetjp_48_;
}
v_resetjp_48_:
{
lean_object* v___x_52_; 
if (v_isShared_50_ == 0)
{
lean_ctor_set(v___x_49_, 0, v_toOne_45_);
v___x_52_ = v___x_49_;
goto v_reusejp_51_;
}
else
{
lean_object* v_reuseFailAlloc_55_; 
v_reuseFailAlloc_55_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_55_, 0, v_toOne_45_);
lean_ctor_set(v_reuseFailAlloc_55_, 1, v_toMul_47_);
v___x_52_ = v_reuseFailAlloc_55_;
goto v_reusejp_51_;
}
v_reusejp_51_:
{
lean_object* v_toZero_53_; lean_object* v___x_54_; 
v_toZero_53_ = lean_ctor_get(v_toAddCommMonoid_46_, 0);
lean_inc(v_toZero_53_);
lean_dec_ref(v_toAddCommMonoid_46_);
v___x_54_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_54_, 0, v___x_52_);
lean_ctor_set(v___x_54_, 1, v_toZero_53_);
return v___x_54_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocSemiring_toMulZeroOneClass(lean_object* v_00_u03b1_57_, lean_object* v_self_58_){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(v_self_58_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object* v_self_60_){
_start:
{
lean_object* v_toNonUnitalNonAssocSemiring_61_; lean_object* v_toOne_62_; lean_object* v_toNatCast_63_; lean_object* v___x_65_; uint8_t v_isShared_66_; uint8_t v_isSharedCheck_71_; 
v_toNonUnitalNonAssocSemiring_61_ = lean_ctor_get(v_self_60_, 0);
v_toOne_62_ = lean_ctor_get(v_self_60_, 1);
v_toNatCast_63_ = lean_ctor_get(v_self_60_, 2);
v_isSharedCheck_71_ = !lean_is_exclusive(v_self_60_);
if (v_isSharedCheck_71_ == 0)
{
v___x_65_ = v_self_60_;
v_isShared_66_ = v_isSharedCheck_71_;
goto v_resetjp_64_;
}
else
{
lean_inc(v_toNatCast_63_);
lean_inc(v_toOne_62_);
lean_inc(v_toNonUnitalNonAssocSemiring_61_);
lean_dec(v_self_60_);
v___x_65_ = lean_box(0);
v_isShared_66_ = v_isSharedCheck_71_;
goto v_resetjp_64_;
}
v_resetjp_64_:
{
lean_object* v_toAddCommMonoid_67_; lean_object* v___x_69_; 
v_toAddCommMonoid_67_ = lean_ctor_get(v_toNonUnitalNonAssocSemiring_61_, 0);
lean_inc_ref(v_toAddCommMonoid_67_);
lean_dec_ref(v_toNonUnitalNonAssocSemiring_61_);
if (v_isShared_66_ == 0)
{
lean_ctor_set(v___x_65_, 2, v_toOne_62_);
lean_ctor_set(v___x_65_, 1, v_toAddCommMonoid_67_);
lean_ctor_set(v___x_65_, 0, v_toNatCast_63_);
v___x_69_ = v___x_65_;
goto v_reusejp_68_;
}
else
{
lean_object* v_reuseFailAlloc_70_; 
v_reuseFailAlloc_70_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_70_, 0, v_toNatCast_63_);
lean_ctor_set(v_reuseFailAlloc_70_, 1, v_toAddCommMonoid_67_);
lean_ctor_set(v_reuseFailAlloc_70_, 2, v_toOne_62_);
v___x_69_ = v_reuseFailAlloc_70_;
goto v_reusejp_68_;
}
v_reusejp_68_:
{
return v___x_69_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne(lean_object* v_00_u03b1_72_, lean_object* v_self_73_){
_start:
{
lean_object* v___x_74_; 
v___x_74_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v_self_73_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object* v_self_75_){
_start:
{
lean_object* v_toAddCommGroup_76_; lean_object* v_toMul_77_; lean_object* v___x_79_; uint8_t v_isShared_80_; uint8_t v_isSharedCheck_85_; 
v_toAddCommGroup_76_ = lean_ctor_get(v_self_75_, 0);
v_toMul_77_ = lean_ctor_get(v_self_75_, 1);
v_isSharedCheck_85_ = !lean_is_exclusive(v_self_75_);
if (v_isSharedCheck_85_ == 0)
{
v___x_79_ = v_self_75_;
v_isShared_80_ = v_isSharedCheck_85_;
goto v_resetjp_78_;
}
else
{
lean_inc(v_toMul_77_);
lean_inc(v_toAddCommGroup_76_);
lean_dec(v_self_75_);
v___x_79_ = lean_box(0);
v_isShared_80_ = v_isSharedCheck_85_;
goto v_resetjp_78_;
}
v_resetjp_78_:
{
lean_object* v_toAddMonoid_81_; lean_object* v___x_83_; 
v_toAddMonoid_81_ = lean_ctor_get(v_toAddCommGroup_76_, 0);
lean_inc_ref(v_toAddMonoid_81_);
lean_dec_ref(v_toAddCommGroup_76_);
if (v_isShared_80_ == 0)
{
lean_ctor_set(v___x_79_, 0, v_toAddMonoid_81_);
v___x_83_ = v___x_79_;
goto v_reusejp_82_;
}
else
{
lean_object* v_reuseFailAlloc_84_; 
v_reuseFailAlloc_84_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_84_, 0, v_toAddMonoid_81_);
lean_ctor_set(v_reuseFailAlloc_84_, 1, v_toMul_77_);
v___x_83_ = v_reuseFailAlloc_84_;
goto v_reusejp_82_;
}
v_reusejp_82_:
{
return v___x_83_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring(lean_object* v_00_u03b1_86_, lean_object* v_self_87_){
_start:
{
lean_object* v___x_88_; 
v___x_88_ = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(v_self_87_);
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRing_toNonUnitalSemiring___redArg(lean_object* v_self_89_){
_start:
{
lean_object* v_toAddCommGroup_90_; lean_object* v_toMul_91_; lean_object* v___x_93_; uint8_t v_isShared_94_; uint8_t v_isSharedCheck_99_; 
v_toAddCommGroup_90_ = lean_ctor_get(v_self_89_, 0);
v_toMul_91_ = lean_ctor_get(v_self_89_, 1);
v_isSharedCheck_99_ = !lean_is_exclusive(v_self_89_);
if (v_isSharedCheck_99_ == 0)
{
v___x_93_ = v_self_89_;
v_isShared_94_ = v_isSharedCheck_99_;
goto v_resetjp_92_;
}
else
{
lean_inc(v_toMul_91_);
lean_inc(v_toAddCommGroup_90_);
lean_dec(v_self_89_);
v___x_93_ = lean_box(0);
v_isShared_94_ = v_isSharedCheck_99_;
goto v_resetjp_92_;
}
v_resetjp_92_:
{
lean_object* v_toAddMonoid_95_; lean_object* v___x_97_; 
v_toAddMonoid_95_ = lean_ctor_get(v_toAddCommGroup_90_, 0);
lean_inc_ref(v_toAddMonoid_95_);
lean_dec_ref(v_toAddCommGroup_90_);
if (v_isShared_94_ == 0)
{
lean_ctor_set(v___x_93_, 0, v_toAddMonoid_95_);
v___x_97_ = v___x_93_;
goto v_reusejp_96_;
}
else
{
lean_object* v_reuseFailAlloc_98_; 
v_reuseFailAlloc_98_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_98_, 0, v_toAddMonoid_95_);
lean_ctor_set(v_reuseFailAlloc_98_, 1, v_toMul_91_);
v___x_97_ = v_reuseFailAlloc_98_;
goto v_reusejp_96_;
}
v_reusejp_96_:
{
return v___x_97_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalRing_toNonUnitalSemiring(lean_object* v_00_u03b1_100_, lean_object* v_self_101_){
_start:
{
lean_object* v___x_102_; 
v___x_102_ = lp_mathlib_NonUnitalRing_toNonUnitalSemiring___redArg(v_self_101_);
return v___x_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocRing_toNonAssocSemiring___redArg(lean_object* v_self_103_){
_start:
{
lean_object* v_toNonUnitalNonAssocRing_104_; lean_object* v_toAddCommGroup_105_; lean_object* v_toOne_106_; lean_object* v_toNatCast_107_; lean_object* v_toMul_108_; lean_object* v___x_110_; uint8_t v_isShared_111_; uint8_t v_isSharedCheck_117_; 
v_toNonUnitalNonAssocRing_104_ = lean_ctor_get(v_self_103_, 0);
lean_inc_ref(v_toNonUnitalNonAssocRing_104_);
v_toAddCommGroup_105_ = lean_ctor_get(v_toNonUnitalNonAssocRing_104_, 0);
lean_inc_ref(v_toAddCommGroup_105_);
v_toOne_106_ = lean_ctor_get(v_self_103_, 1);
lean_inc(v_toOne_106_);
v_toNatCast_107_ = lean_ctor_get(v_self_103_, 2);
lean_inc(v_toNatCast_107_);
lean_dec_ref(v_self_103_);
v_toMul_108_ = lean_ctor_get(v_toNonUnitalNonAssocRing_104_, 1);
v_isSharedCheck_117_ = !lean_is_exclusive(v_toNonUnitalNonAssocRing_104_);
if (v_isSharedCheck_117_ == 0)
{
lean_object* v_unused_118_; 
v_unused_118_ = lean_ctor_get(v_toNonUnitalNonAssocRing_104_, 0);
lean_dec(v_unused_118_);
v___x_110_ = v_toNonUnitalNonAssocRing_104_;
v_isShared_111_ = v_isSharedCheck_117_;
goto v_resetjp_109_;
}
else
{
lean_inc(v_toMul_108_);
lean_dec(v_toNonUnitalNonAssocRing_104_);
v___x_110_ = lean_box(0);
v_isShared_111_ = v_isSharedCheck_117_;
goto v_resetjp_109_;
}
v_resetjp_109_:
{
lean_object* v_toAddMonoid_112_; lean_object* v___x_114_; 
v_toAddMonoid_112_ = lean_ctor_get(v_toAddCommGroup_105_, 0);
lean_inc_ref(v_toAddMonoid_112_);
lean_dec_ref(v_toAddCommGroup_105_);
if (v_isShared_111_ == 0)
{
lean_ctor_set(v___x_110_, 0, v_toAddMonoid_112_);
v___x_114_ = v___x_110_;
goto v_reusejp_113_;
}
else
{
lean_object* v_reuseFailAlloc_116_; 
v_reuseFailAlloc_116_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_116_, 0, v_toAddMonoid_112_);
lean_ctor_set(v_reuseFailAlloc_116_, 1, v_toMul_108_);
v___x_114_ = v_reuseFailAlloc_116_;
goto v_reusejp_113_;
}
v_reusejp_113_:
{
lean_object* v___x_115_; 
v___x_115_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_115_, 0, v___x_114_);
lean_ctor_set(v___x_115_, 1, v_toOne_106_);
lean_ctor_set(v___x_115_, 2, v_toNatCast_107_);
return v___x_115_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocRing_toNonAssocSemiring(lean_object* v_00_u03b1_119_, lean_object* v_self_120_){
_start:
{
lean_object* v___x_121_; 
v___x_121_ = lp_mathlib_NonAssocRing_toNonAssocSemiring___redArg(v_self_120_);
return v___x_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(lean_object* v_self_122_){
_start:
{
lean_object* v_toNonUnitalNonAssocRing_123_; lean_object* v_toOne_124_; lean_object* v_toNatCast_125_; lean_object* v_toIntCast_126_; lean_object* v___x_128_; uint8_t v_isShared_129_; uint8_t v_isSharedCheck_134_; 
v_toNonUnitalNonAssocRing_123_ = lean_ctor_get(v_self_122_, 0);
v_toOne_124_ = lean_ctor_get(v_self_122_, 1);
v_toNatCast_125_ = lean_ctor_get(v_self_122_, 2);
v_toIntCast_126_ = lean_ctor_get(v_self_122_, 3);
v_isSharedCheck_134_ = !lean_is_exclusive(v_self_122_);
if (v_isSharedCheck_134_ == 0)
{
v___x_128_ = v_self_122_;
v_isShared_129_ = v_isSharedCheck_134_;
goto v_resetjp_127_;
}
else
{
lean_inc(v_toIntCast_126_);
lean_inc(v_toNatCast_125_);
lean_inc(v_toOne_124_);
lean_inc(v_toNonUnitalNonAssocRing_123_);
lean_dec(v_self_122_);
v___x_128_ = lean_box(0);
v_isShared_129_ = v_isSharedCheck_134_;
goto v_resetjp_127_;
}
v_resetjp_127_:
{
lean_object* v_toAddCommGroup_130_; lean_object* v___x_132_; 
v_toAddCommGroup_130_ = lean_ctor_get(v_toNonUnitalNonAssocRing_123_, 0);
lean_inc_ref(v_toAddCommGroup_130_);
lean_dec_ref(v_toNonUnitalNonAssocRing_123_);
if (v_isShared_129_ == 0)
{
lean_ctor_set(v___x_128_, 3, v_toOne_124_);
lean_ctor_set(v___x_128_, 1, v_toIntCast_126_);
lean_ctor_set(v___x_128_, 0, v_toAddCommGroup_130_);
v___x_132_ = v___x_128_;
goto v_reusejp_131_;
}
else
{
lean_object* v_reuseFailAlloc_133_; 
v_reuseFailAlloc_133_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_133_, 0, v_toAddCommGroup_130_);
lean_ctor_set(v_reuseFailAlloc_133_, 1, v_toIntCast_126_);
lean_ctor_set(v_reuseFailAlloc_133_, 2, v_toNatCast_125_);
lean_ctor_set(v_reuseFailAlloc_133_, 3, v_toOne_124_);
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
LEAN_EXPORT lean_object* lp_mathlib_NonAssocRing_toAddCommGroupWithOne(lean_object* v_00_u03b1_135_, lean_object* v_self_136_){
_start:
{
lean_object* v___x_137_; 
v___x_137_ = lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(v_self_136_);
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toMonoidWithZero___redArg(lean_object* v_self_138_){
_start:
{
lean_object* v_toAddCommMonoid_139_; lean_object* v_toMonoid_140_; lean_object* v_toZero_141_; lean_object* v___x_142_; 
v_toAddCommMonoid_139_ = lean_ctor_get(v_self_138_, 0);
v_toMonoid_140_ = lean_ctor_get(v_self_138_, 1);
v_toZero_141_ = lean_ctor_get(v_toAddCommMonoid_139_, 0);
lean_inc(v_toZero_141_);
lean_inc_ref(v_toMonoid_140_);
v___x_142_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_142_, 0, v_toMonoid_140_);
lean_ctor_set(v___x_142_, 1, v_toZero_141_);
return v___x_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toMonoidWithZero___redArg___boxed(lean_object* v_self_143_){
_start:
{
lean_object* v_res_144_; 
v_res_144_ = lp_mathlib_Semiring_toMonoidWithZero___redArg(v_self_143_);
lean_dec_ref(v_self_143_);
return v_res_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toMonoidWithZero(lean_object* v_00_u03b1_145_, lean_object* v_self_146_){
_start:
{
lean_object* v___x_147_; 
v___x_147_ = lp_mathlib_Semiring_toMonoidWithZero___redArg(v_self_146_);
return v___x_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toMonoidWithZero___boxed(lean_object* v_00_u03b1_148_, lean_object* v_self_149_){
_start:
{
lean_object* v_res_150_; 
v_res_150_ = lp_mathlib_Semiring_toMonoidWithZero(v_00_u03b1_148_, v_self_149_);
lean_dec_ref(v_self_149_);
return v_res_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toNonUnitalSemiring___redArg(lean_object* v_self_151_){
_start:
{
lean_object* v_toMonoid_152_; lean_object* v_toAddCommMonoid_153_; lean_object* v_toMul_154_; lean_object* v___x_155_; 
v_toMonoid_152_ = lean_ctor_get(v_self_151_, 1);
v_toAddCommMonoid_153_ = lean_ctor_get(v_self_151_, 0);
v_toMul_154_ = lean_ctor_get(v_toMonoid_152_, 1);
lean_inc(v_toMul_154_);
lean_inc_ref(v_toAddCommMonoid_153_);
v___x_155_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_155_, 0, v_toAddCommMonoid_153_);
lean_ctor_set(v___x_155_, 1, v_toMul_154_);
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toNonUnitalSemiring___redArg___boxed(lean_object* v_self_156_){
_start:
{
lean_object* v_res_157_; 
v_res_157_ = lp_mathlib_Semiring_toNonUnitalSemiring___redArg(v_self_156_);
lean_dec_ref(v_self_156_);
return v_res_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toNonUnitalSemiring(lean_object* v_00_u03b1_158_, lean_object* v_self_159_){
_start:
{
lean_object* v___x_160_; 
v___x_160_ = lp_mathlib_Semiring_toNonUnitalSemiring___redArg(v_self_159_);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toNonUnitalSemiring___boxed(lean_object* v_00_u03b1_161_, lean_object* v_self_162_){
_start:
{
lean_object* v_res_163_; 
v_res_163_ = lp_mathlib_Semiring_toNonUnitalSemiring(v_00_u03b1_161_, v_self_162_);
lean_dec_ref(v_self_162_);
return v_res_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object* v_self_164_){
_start:
{
lean_object* v_toMonoid_165_; lean_object* v_toAddCommMonoid_166_; lean_object* v_toNatCast_167_; lean_object* v_toOne_168_; lean_object* v_toMul_169_; lean_object* v___x_171_; uint8_t v_isShared_172_; uint8_t v_isSharedCheck_177_; 
v_toMonoid_165_ = lean_ctor_get(v_self_164_, 1);
lean_inc_ref(v_toMonoid_165_);
v_toAddCommMonoid_166_ = lean_ctor_get(v_self_164_, 0);
lean_inc_ref(v_toAddCommMonoid_166_);
v_toNatCast_167_ = lean_ctor_get(v_self_164_, 2);
lean_inc(v_toNatCast_167_);
lean_dec_ref(v_self_164_);
v_toOne_168_ = lean_ctor_get(v_toMonoid_165_, 0);
v_toMul_169_ = lean_ctor_get(v_toMonoid_165_, 1);
v_isSharedCheck_177_ = !lean_is_exclusive(v_toMonoid_165_);
if (v_isSharedCheck_177_ == 0)
{
lean_object* v_unused_178_; 
v_unused_178_ = lean_ctor_get(v_toMonoid_165_, 2);
lean_dec(v_unused_178_);
v___x_171_ = v_toMonoid_165_;
v_isShared_172_ = v_isSharedCheck_177_;
goto v_resetjp_170_;
}
else
{
lean_inc(v_toMul_169_);
lean_inc(v_toOne_168_);
lean_dec(v_toMonoid_165_);
v___x_171_ = lean_box(0);
v_isShared_172_ = v_isSharedCheck_177_;
goto v_resetjp_170_;
}
v_resetjp_170_:
{
lean_object* v___x_173_; lean_object* v___x_175_; 
v___x_173_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_173_, 0, v_toAddCommMonoid_166_);
lean_ctor_set(v___x_173_, 1, v_toMul_169_);
if (v_isShared_172_ == 0)
{
lean_ctor_set(v___x_171_, 2, v_toNatCast_167_);
lean_ctor_set(v___x_171_, 1, v_toOne_168_);
lean_ctor_set(v___x_171_, 0, v___x_173_);
v___x_175_ = v___x_171_;
goto v_reusejp_174_;
}
else
{
lean_object* v_reuseFailAlloc_176_; 
v_reuseFailAlloc_176_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_176_, 0, v___x_173_);
lean_ctor_set(v_reuseFailAlloc_176_, 1, v_toOne_168_);
lean_ctor_set(v_reuseFailAlloc_176_, 2, v_toNatCast_167_);
v___x_175_ = v_reuseFailAlloc_176_;
goto v_reusejp_174_;
}
v_reusejp_174_:
{
return v___x_175_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_toNonAssocSemiring(lean_object* v_00_u03b1_179_, lean_object* v_self_180_){
_start:
{
lean_object* v___x_181_; 
v___x_181_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_self_180_);
return v___x_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_toAddCommGroup___redArg(lean_object* v_self_182_){
_start:
{
lean_object* v_toSemiring_183_; lean_object* v_toNeg_184_; lean_object* v_toSub_185_; lean_object* v_toZSMul_186_; lean_object* v_toAddCommMonoid_187_; lean_object* v___x_188_; 
v_toSemiring_183_ = lean_ctor_get(v_self_182_, 0);
v_toNeg_184_ = lean_ctor_get(v_self_182_, 1);
v_toSub_185_ = lean_ctor_get(v_self_182_, 2);
v_toZSMul_186_ = lean_ctor_get(v_self_182_, 3);
v_toAddCommMonoid_187_ = lean_ctor_get(v_toSemiring_183_, 0);
lean_inc(v_toZSMul_186_);
lean_inc(v_toSub_185_);
lean_inc(v_toNeg_184_);
lean_inc_ref(v_toAddCommMonoid_187_);
v___x_188_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_188_, 0, v_toAddCommMonoid_187_);
lean_ctor_set(v___x_188_, 1, v_toNeg_184_);
lean_ctor_set(v___x_188_, 2, v_toSub_185_);
lean_ctor_set(v___x_188_, 3, v_toZSMul_186_);
return v___x_188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_toAddCommGroup___redArg___boxed(lean_object* v_self_189_){
_start:
{
lean_object* v_res_190_; 
v_res_190_ = lp_mathlib_Ring_toAddCommGroup___redArg(v_self_189_);
lean_dec_ref(v_self_189_);
return v_res_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_toAddCommGroup(lean_object* v_R_191_, lean_object* v_self_192_){
_start:
{
lean_object* v___x_193_; 
v___x_193_ = lp_mathlib_Ring_toAddCommGroup___redArg(v_self_192_);
return v___x_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_toAddCommGroup___boxed(lean_object* v_R_194_, lean_object* v_self_195_){
_start:
{
lean_object* v_res_196_; 
v_res_196_ = lp_mathlib_Ring_toAddCommGroup(v_R_194_, v_self_195_);
lean_dec_ref(v_self_195_);
return v_res_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object* v_self_197_){
_start:
{
lean_object* v_toSemiring_198_; lean_object* v_toMonoid_199_; lean_object* v_toNeg_200_; lean_object* v_toSub_201_; lean_object* v_toZSMul_202_; lean_object* v_toIntCast_203_; lean_object* v___x_205_; uint8_t v_isShared_206_; uint8_t v_isSharedCheck_222_; 
v_toSemiring_198_ = lean_ctor_get(v_self_197_, 0);
lean_inc_ref(v_toSemiring_198_);
v_toMonoid_199_ = lean_ctor_get(v_toSemiring_198_, 1);
lean_inc_ref(v_toMonoid_199_);
v_toNeg_200_ = lean_ctor_get(v_self_197_, 1);
v_toSub_201_ = lean_ctor_get(v_self_197_, 2);
v_toZSMul_202_ = lean_ctor_get(v_self_197_, 3);
v_toIntCast_203_ = lean_ctor_get(v_self_197_, 4);
v_isSharedCheck_222_ = !lean_is_exclusive(v_self_197_);
if (v_isSharedCheck_222_ == 0)
{
lean_object* v_unused_223_; 
v_unused_223_ = lean_ctor_get(v_self_197_, 0);
lean_dec(v_unused_223_);
v___x_205_ = v_self_197_;
v_isShared_206_ = v_isSharedCheck_222_;
goto v_resetjp_204_;
}
else
{
lean_inc(v_toIntCast_203_);
lean_inc(v_toZSMul_202_);
lean_inc(v_toSub_201_);
lean_inc(v_toNeg_200_);
lean_dec(v_self_197_);
v___x_205_ = lean_box(0);
v_isShared_206_ = v_isSharedCheck_222_;
goto v_resetjp_204_;
}
v_resetjp_204_:
{
lean_object* v_toAddCommMonoid_207_; lean_object* v_toNatCast_208_; lean_object* v_toOne_209_; lean_object* v___x_211_; uint8_t v_isShared_212_; uint8_t v_isSharedCheck_219_; 
v_toAddCommMonoid_207_ = lean_ctor_get(v_toSemiring_198_, 0);
lean_inc_ref(v_toAddCommMonoid_207_);
v_toNatCast_208_ = lean_ctor_get(v_toSemiring_198_, 2);
lean_inc(v_toNatCast_208_);
lean_dec_ref(v_toSemiring_198_);
v_toOne_209_ = lean_ctor_get(v_toMonoid_199_, 0);
v_isSharedCheck_219_ = !lean_is_exclusive(v_toMonoid_199_);
if (v_isSharedCheck_219_ == 0)
{
lean_object* v_unused_220_; lean_object* v_unused_221_; 
v_unused_220_ = lean_ctor_get(v_toMonoid_199_, 2);
lean_dec(v_unused_220_);
v_unused_221_ = lean_ctor_get(v_toMonoid_199_, 1);
lean_dec(v_unused_221_);
v___x_211_ = v_toMonoid_199_;
v_isShared_212_ = v_isSharedCheck_219_;
goto v_resetjp_210_;
}
else
{
lean_inc(v_toOne_209_);
lean_dec(v_toMonoid_199_);
v___x_211_ = lean_box(0);
v_isShared_212_ = v_isSharedCheck_219_;
goto v_resetjp_210_;
}
v_resetjp_210_:
{
lean_object* v___x_214_; 
if (v_isShared_212_ == 0)
{
lean_ctor_set(v___x_211_, 2, v_toOne_209_);
lean_ctor_set(v___x_211_, 1, v_toAddCommMonoid_207_);
lean_ctor_set(v___x_211_, 0, v_toNatCast_208_);
v___x_214_ = v___x_211_;
goto v_reusejp_213_;
}
else
{
lean_object* v_reuseFailAlloc_218_; 
v_reuseFailAlloc_218_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_218_, 0, v_toNatCast_208_);
lean_ctor_set(v_reuseFailAlloc_218_, 1, v_toAddCommMonoid_207_);
lean_ctor_set(v_reuseFailAlloc_218_, 2, v_toOne_209_);
v___x_214_ = v_reuseFailAlloc_218_;
goto v_reusejp_213_;
}
v_reusejp_213_:
{
lean_object* v___x_216_; 
if (v_isShared_206_ == 0)
{
lean_ctor_set(v___x_205_, 4, v_toZSMul_202_);
lean_ctor_set(v___x_205_, 3, v_toSub_201_);
lean_ctor_set(v___x_205_, 2, v_toNeg_200_);
lean_ctor_set(v___x_205_, 1, v___x_214_);
lean_ctor_set(v___x_205_, 0, v_toIntCast_203_);
v___x_216_ = v___x_205_;
goto v_reusejp_215_;
}
else
{
lean_object* v_reuseFailAlloc_217_; 
v_reuseFailAlloc_217_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_217_, 0, v_toIntCast_203_);
lean_ctor_set(v_reuseFailAlloc_217_, 1, v___x_214_);
lean_ctor_set(v_reuseFailAlloc_217_, 2, v_toNeg_200_);
lean_ctor_set(v_reuseFailAlloc_217_, 3, v_toSub_201_);
lean_ctor_set(v_reuseFailAlloc_217_, 4, v_toZSMul_202_);
v___x_216_ = v_reuseFailAlloc_217_;
goto v_reusejp_215_;
}
v_reusejp_215_:
{
return v___x_216_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_toAddGroupWithOne(lean_object* v_R_224_, lean_object* v_self_225_){
_start:
{
lean_object* v___x_226_; 
v___x_226_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_self_225_);
return v___x_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDistribOfSemiring___redArg(lean_object* v_inst_227_){
_start:
{
lean_object* v___x_228_; lean_object* v_toNonUnitalNonAssocSemiring_229_; lean_object* v___x_230_; 
v___x_228_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_227_);
v_toNonUnitalNonAssocSemiring_229_ = lean_ctor_get(v___x_228_, 0);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_229_);
lean_dec_ref(v___x_228_);
v___x_230_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v_toNonUnitalNonAssocSemiring_229_);
return v___x_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDistribOfSemiring(lean_object* v_00_u03b1_231_, lean_object* v_inst_232_){
_start:
{
lean_object* v___x_233_; 
v___x_233_ = lp_mathlib_instDistribOfSemiring___redArg(v_inst_232_);
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulZeroClassOfSemiring___redArg(lean_object* v_inst_234_){
_start:
{
lean_object* v___x_235_; lean_object* v_toNonUnitalNonAssocSemiring_236_; lean_object* v___x_237_; 
v___x_235_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_234_);
v_toNonUnitalNonAssocSemiring_236_ = lean_ctor_get(v___x_235_, 0);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_236_);
lean_dec_ref(v___x_235_);
v___x_237_ = lp_mathlib_NonUnitalNonAssocSemiring_toMulZeroClass___redArg(v_toNonUnitalNonAssocSemiring_236_);
return v___x_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulZeroClassOfSemiring(lean_object* v_00_u03b1_238_, lean_object* v_inst_239_){
_start:
{
lean_object* v___x_240_; 
v___x_240_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v_inst_239_);
return v___x_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulZeroOneClassOfSemiring___redArg(lean_object* v_inst_241_){
_start:
{
lean_object* v___x_242_; lean_object* v___x_243_; 
v___x_242_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_241_);
v___x_243_ = lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(v___x_242_);
return v___x_243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulZeroOneClassOfSemiring(lean_object* v_00_u03b1_244_, lean_object* v_inst_245_){
_start:
{
lean_object* v___x_246_; 
v___x_246_ = lp_mathlib_instMulZeroOneClassOfSemiring___redArg(v_inst_245_);
return v___x_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocCommSemiring_toCommMagma___redArg(lean_object* v_self_247_){
_start:
{
lean_object* v_toMul_248_; 
v_toMul_248_ = lean_ctor_get(v_self_247_, 1);
lean_inc(v_toMul_248_);
return v_toMul_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocCommSemiring_toCommMagma___redArg___boxed(lean_object* v_self_249_){
_start:
{
lean_object* v_res_250_; 
v_res_250_ = lp_mathlib_NonUnitalNonAssocCommSemiring_toCommMagma___redArg(v_self_249_);
lean_dec_ref(v_self_249_);
return v_res_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocCommSemiring_toCommMagma(lean_object* v_00_u03b1_251_, lean_object* v_self_252_){
_start:
{
lean_object* v_toMul_253_; 
v_toMul_253_ = lean_ctor_get(v_self_252_, 1);
lean_inc(v_toMul_253_);
return v_toMul_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocCommSemiring_toCommMagma___boxed(lean_object* v_00_u03b1_254_, lean_object* v_self_255_){
_start:
{
lean_object* v_res_256_; 
v_res_256_ = lp_mathlib_NonUnitalNonAssocCommSemiring_toCommMagma(v_00_u03b1_254_, v_self_255_);
lean_dec_ref(v_self_255_);
return v_res_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toCommSemigroup___redArg(lean_object* v_self_257_){
_start:
{
lean_object* v_toMul_258_; 
v_toMul_258_ = lean_ctor_get(v_self_257_, 1);
lean_inc(v_toMul_258_);
return v_toMul_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toCommSemigroup___redArg___boxed(lean_object* v_self_259_){
_start:
{
lean_object* v_res_260_; 
v_res_260_ = lp_mathlib_NonUnitalCommSemiring_toCommSemigroup___redArg(v_self_259_);
lean_dec_ref(v_self_259_);
return v_res_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toCommSemigroup(lean_object* v_00_u03b1_261_, lean_object* v_self_262_){
_start:
{
lean_object* v_toMul_263_; 
v_toMul_263_ = lean_ctor_get(v_self_262_, 1);
lean_inc(v_toMul_263_);
return v_toMul_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toCommSemigroup___boxed(lean_object* v_00_u03b1_264_, lean_object* v_self_265_){
_start:
{
lean_object* v_res_266_; 
v_res_266_ = lp_mathlib_NonUnitalCommSemiring_toCommSemigroup(v_00_u03b1_264_, v_self_265_);
lean_dec_ref(v_self_265_);
return v_res_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommSemiring_toNonUnitalNonAssocCommSemiring___redArg(lean_object* v_self_267_){
_start:
{
lean_object* v_toNonUnitalNonAssocSemiring_268_; 
v_toNonUnitalNonAssocSemiring_268_ = lean_ctor_get(v_self_267_, 0);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_268_);
return v_toNonUnitalNonAssocSemiring_268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommSemiring_toNonUnitalNonAssocCommSemiring___redArg___boxed(lean_object* v_self_269_){
_start:
{
lean_object* v_res_270_; 
v_res_270_ = lp_mathlib_NonAssocCommSemiring_toNonUnitalNonAssocCommSemiring___redArg(v_self_269_);
lean_dec_ref(v_self_269_);
return v_res_270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommSemiring_toNonUnitalNonAssocCommSemiring(lean_object* v_00_u03b1_271_, lean_object* v_self_272_){
_start:
{
lean_object* v_toNonUnitalNonAssocSemiring_273_; 
v_toNonUnitalNonAssocSemiring_273_ = lean_ctor_get(v_self_272_, 0);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_273_);
return v_toNonUnitalNonAssocSemiring_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommSemiring_toNonUnitalNonAssocCommSemiring___boxed(lean_object* v_00_u03b1_274_, lean_object* v_self_275_){
_start:
{
lean_object* v_res_276_; 
v_res_276_ = lp_mathlib_NonAssocCommSemiring_toNonUnitalNonAssocCommSemiring(v_00_u03b1_274_, v_self_275_);
lean_dec_ref(v_self_275_);
return v_res_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toCommMonoid___redArg(lean_object* v_self_277_){
_start:
{
lean_object* v_toMonoid_278_; 
v_toMonoid_278_ = lean_ctor_get(v_self_277_, 1);
lean_inc_ref(v_toMonoid_278_);
return v_toMonoid_278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toCommMonoid___redArg___boxed(lean_object* v_self_279_){
_start:
{
lean_object* v_res_280_; 
v_res_280_ = lp_mathlib_CommSemiring_toCommMonoid___redArg(v_self_279_);
lean_dec_ref(v_self_279_);
return v_res_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toCommMonoid(lean_object* v_R_281_, lean_object* v_self_282_){
_start:
{
lean_object* v_toMonoid_283_; 
v_toMonoid_283_ = lean_ctor_get(v_self_282_, 1);
lean_inc_ref(v_toMonoid_283_);
return v_toMonoid_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toCommMonoid___boxed(lean_object* v_R_284_, lean_object* v_self_285_){
_start:
{
lean_object* v_res_286_; 
v_res_286_ = lp_mathlib_CommSemiring_toCommMonoid(v_R_284_, v_self_285_);
lean_dec_ref(v_self_285_);
return v_res_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toNonUnitalNonAssocCommSemiring___redArg(lean_object* v_inst_287_){
_start:
{
lean_inc_ref(v_inst_287_);
return v_inst_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toNonUnitalNonAssocCommSemiring___redArg___boxed(lean_object* v_inst_288_){
_start:
{
lean_object* v_res_289_; 
v_res_289_ = lp_mathlib_NonUnitalCommSemiring_toNonUnitalNonAssocCommSemiring___redArg(v_inst_288_);
lean_dec_ref(v_inst_288_);
return v_res_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toNonUnitalNonAssocCommSemiring(lean_object* v_00_u03b1_290_, lean_object* v_inst_291_){
_start:
{
lean_inc_ref(v_inst_291_);
return v_inst_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommSemiring_toNonUnitalNonAssocCommSemiring___boxed(lean_object* v_00_u03b1_292_, lean_object* v_inst_293_){
_start:
{
lean_object* v_res_294_; 
v_res_294_ = lp_mathlib_NonUnitalCommSemiring_toNonUnitalNonAssocCommSemiring(v_00_u03b1_292_, v_inst_293_);
lean_dec_ref(v_inst_293_);
return v_res_294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toNonAssocCommSemiring___redArg(lean_object* v_inst_295_){
_start:
{
lean_object* v___x_296_; 
v___x_296_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_295_);
return v___x_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toNonAssocCommSemiring(lean_object* v_00_u03b1_297_, lean_object* v_inst_298_){
_start:
{
lean_object* v___x_299_; 
v___x_299_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_298_);
return v___x_299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toNonUnitalCommSemiring___redArg(lean_object* v_inst_300_){
_start:
{
lean_object* v_toMonoid_301_; lean_object* v_toAddCommMonoid_302_; lean_object* v_toMul_303_; lean_object* v___x_304_; 
v_toMonoid_301_ = lean_ctor_get(v_inst_300_, 1);
v_toAddCommMonoid_302_ = lean_ctor_get(v_inst_300_, 0);
v_toMul_303_ = lean_ctor_get(v_toMonoid_301_, 1);
lean_inc(v_toMul_303_);
lean_inc_ref(v_toAddCommMonoid_302_);
v___x_304_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_304_, 0, v_toAddCommMonoid_302_);
lean_ctor_set(v___x_304_, 1, v_toMul_303_);
return v___x_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toNonUnitalCommSemiring___redArg___boxed(lean_object* v_inst_305_){
_start:
{
lean_object* v_res_306_; 
v_res_306_ = lp_mathlib_CommSemiring_toNonUnitalCommSemiring___redArg(v_inst_305_);
lean_dec_ref(v_inst_305_);
return v_res_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toNonUnitalCommSemiring(lean_object* v_00_u03b1_307_, lean_object* v_inst_308_){
_start:
{
lean_object* v___x_309_; 
v___x_309_ = lp_mathlib_CommSemiring_toNonUnitalCommSemiring___redArg(v_inst_308_);
return v___x_309_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toNonUnitalCommSemiring___boxed(lean_object* v_00_u03b1_310_, lean_object* v_inst_311_){
_start:
{
lean_object* v_res_312_; 
v_res_312_ = lp_mathlib_CommSemiring_toNonUnitalCommSemiring(v_00_u03b1_310_, v_inst_311_);
lean_dec_ref(v_inst_311_);
return v_res_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toCommMonoidWithZero___redArg(lean_object* v_inst_313_){
_start:
{
lean_object* v_toAddCommMonoid_314_; lean_object* v_toMonoid_315_; lean_object* v_toZero_316_; lean_object* v___x_317_; 
v_toAddCommMonoid_314_ = lean_ctor_get(v_inst_313_, 0);
v_toMonoid_315_ = lean_ctor_get(v_inst_313_, 1);
v_toZero_316_ = lean_ctor_get(v_toAddCommMonoid_314_, 0);
lean_inc(v_toZero_316_);
lean_inc_ref(v_toMonoid_315_);
v___x_317_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_317_, 0, v_toMonoid_315_);
lean_ctor_set(v___x_317_, 1, v_toZero_316_);
return v___x_317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toCommMonoidWithZero___redArg___boxed(lean_object* v_inst_318_){
_start:
{
lean_object* v_res_319_; 
v_res_319_ = lp_mathlib_CommSemiring_toCommMonoidWithZero___redArg(v_inst_318_);
lean_dec_ref(v_inst_318_);
return v_res_319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toCommMonoidWithZero(lean_object* v_00_u03b1_320_, lean_object* v_inst_321_){
_start:
{
lean_object* v___x_322_; 
v___x_322_ = lp_mathlib_CommSemiring_toCommMonoidWithZero___redArg(v_inst_321_);
return v___x_322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_toCommMonoidWithZero___boxed(lean_object* v_00_u03b1_323_, lean_object* v_inst_324_){
_start:
{
lean_object* v_res_325_; 
v_res_325_ = lp_mathlib_CommSemiring_toCommMonoidWithZero(v_00_u03b1_323_, v_inst_324_);
lean_dec_ref(v_inst_324_);
return v_res_325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulZeroClass_negZeroClass___redArg(lean_object* v_inst_326_, lean_object* v_inst_327_){
_start:
{
lean_object* v_toZero_328_; lean_object* v___x_330_; uint8_t v_isShared_331_; uint8_t v_isSharedCheck_335_; 
v_toZero_328_ = lean_ctor_get(v_inst_326_, 1);
v_isSharedCheck_335_ = !lean_is_exclusive(v_inst_326_);
if (v_isSharedCheck_335_ == 0)
{
lean_object* v_unused_336_; 
v_unused_336_ = lean_ctor_get(v_inst_326_, 0);
lean_dec(v_unused_336_);
v___x_330_ = v_inst_326_;
v_isShared_331_ = v_isSharedCheck_335_;
goto v_resetjp_329_;
}
else
{
lean_inc(v_toZero_328_);
lean_dec(v_inst_326_);
v___x_330_ = lean_box(0);
v_isShared_331_ = v_isSharedCheck_335_;
goto v_resetjp_329_;
}
v_resetjp_329_:
{
lean_object* v___x_333_; 
if (v_isShared_331_ == 0)
{
lean_ctor_set(v___x_330_, 1, v_inst_327_);
lean_ctor_set(v___x_330_, 0, v_toZero_328_);
v___x_333_ = v___x_330_;
goto v_reusejp_332_;
}
else
{
lean_object* v_reuseFailAlloc_334_; 
v_reuseFailAlloc_334_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_334_, 0, v_toZero_328_);
lean_ctor_set(v_reuseFailAlloc_334_, 1, v_inst_327_);
v___x_333_ = v_reuseFailAlloc_334_;
goto v_reusejp_332_;
}
v_reusejp_332_:
{
return v___x_333_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulZeroClass_negZeroClass(lean_object* v_00_u03b1_337_, lean_object* v_inst_338_, lean_object* v_inst_339_){
_start:
{
lean_object* v___x_340_; 
v___x_340_ = lp_mathlib_MulZeroClass_negZeroClass___redArg(v_inst_338_, v_inst_339_);
return v___x_340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocRing_toHasDistribNeg___redArg(lean_object* v_inst_341_){
_start:
{
lean_object* v_toAddCommGroup_342_; lean_object* v_toNeg_343_; 
v_toAddCommGroup_342_ = lean_ctor_get(v_inst_341_, 0);
v_toNeg_343_ = lean_ctor_get(v_toAddCommGroup_342_, 1);
lean_inc(v_toNeg_343_);
return v_toNeg_343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocRing_toHasDistribNeg___redArg___boxed(lean_object* v_inst_344_){
_start:
{
lean_object* v_res_345_; 
v_res_345_ = lp_mathlib_NonUnitalNonAssocRing_toHasDistribNeg___redArg(v_inst_344_);
lean_dec_ref(v_inst_344_);
return v_res_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocRing_toHasDistribNeg(lean_object* v_00_u03b1_346_, lean_object* v_inst_347_){
_start:
{
lean_object* v___x_348_; 
v___x_348_ = lp_mathlib_NonUnitalNonAssocRing_toHasDistribNeg___redArg(v_inst_347_);
return v___x_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocRing_toHasDistribNeg___boxed(lean_object* v_00_u03b1_349_, lean_object* v_inst_350_){
_start:
{
lean_object* v_res_351_; 
v_res_351_ = lp_mathlib_NonUnitalNonAssocRing_toHasDistribNeg(v_00_u03b1_349_, v_inst_350_);
lean_dec_ref(v_inst_350_);
return v_res_351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_toNonUnitalRing___redArg(lean_object* v_inst_352_){
_start:
{
lean_object* v_toSemiring_353_; lean_object* v_toNeg_354_; lean_object* v_toSub_355_; lean_object* v_toZSMul_356_; lean_object* v_toAddCommMonoid_357_; lean_object* v_toMonoid_358_; lean_object* v___x_359_; lean_object* v_toMul_360_; lean_object* v___x_361_; 
v_toSemiring_353_ = lean_ctor_get(v_inst_352_, 0);
v_toNeg_354_ = lean_ctor_get(v_inst_352_, 1);
v_toSub_355_ = lean_ctor_get(v_inst_352_, 2);
v_toZSMul_356_ = lean_ctor_get(v_inst_352_, 3);
v_toAddCommMonoid_357_ = lean_ctor_get(v_toSemiring_353_, 0);
v_toMonoid_358_ = lean_ctor_get(v_toSemiring_353_, 1);
lean_inc(v_toZSMul_356_);
lean_inc(v_toSub_355_);
lean_inc(v_toNeg_354_);
lean_inc_ref(v_toAddCommMonoid_357_);
v___x_359_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_359_, 0, v_toAddCommMonoid_357_);
lean_ctor_set(v___x_359_, 1, v_toNeg_354_);
lean_ctor_set(v___x_359_, 2, v_toSub_355_);
lean_ctor_set(v___x_359_, 3, v_toZSMul_356_);
v_toMul_360_ = lean_ctor_get(v_toMonoid_358_, 1);
lean_inc(v_toMul_360_);
v___x_361_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_361_, 0, v___x_359_);
lean_ctor_set(v___x_361_, 1, v_toMul_360_);
return v___x_361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_toNonUnitalRing___redArg___boxed(lean_object* v_inst_362_){
_start:
{
lean_object* v_res_363_; 
v_res_363_ = lp_mathlib_Ring_toNonUnitalRing___redArg(v_inst_362_);
lean_dec_ref(v_inst_362_);
return v_res_363_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_toNonUnitalRing(lean_object* v_00_u03b1_364_, lean_object* v_inst_365_){
_start:
{
lean_object* v___x_366_; 
v___x_366_ = lp_mathlib_Ring_toNonUnitalRing___redArg(v_inst_365_);
return v___x_366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_toNonUnitalRing___boxed(lean_object* v_00_u03b1_367_, lean_object* v_inst_368_){
_start:
{
lean_object* v_res_369_; 
v_res_369_ = lp_mathlib_Ring_toNonUnitalRing(v_00_u03b1_367_, v_inst_368_);
lean_dec_ref(v_inst_368_);
return v_res_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_toNonAssocRing___redArg(lean_object* v_inst_370_){
_start:
{
lean_object* v_toSemiring_371_; lean_object* v_toNeg_372_; lean_object* v_toSub_373_; lean_object* v_toZSMul_374_; lean_object* v_toIntCast_375_; lean_object* v_toAddCommMonoid_376_; lean_object* v_toMonoid_377_; lean_object* v_toNatCast_378_; lean_object* v___x_379_; lean_object* v_toOne_380_; lean_object* v_toMul_381_; lean_object* v___x_382_; lean_object* v___x_383_; 
v_toSemiring_371_ = lean_ctor_get(v_inst_370_, 0);
v_toNeg_372_ = lean_ctor_get(v_inst_370_, 1);
v_toSub_373_ = lean_ctor_get(v_inst_370_, 2);
v_toZSMul_374_ = lean_ctor_get(v_inst_370_, 3);
v_toIntCast_375_ = lean_ctor_get(v_inst_370_, 4);
v_toAddCommMonoid_376_ = lean_ctor_get(v_toSemiring_371_, 0);
v_toMonoid_377_ = lean_ctor_get(v_toSemiring_371_, 1);
v_toNatCast_378_ = lean_ctor_get(v_toSemiring_371_, 2);
lean_inc(v_toZSMul_374_);
lean_inc(v_toSub_373_);
lean_inc(v_toNeg_372_);
lean_inc_ref(v_toAddCommMonoid_376_);
v___x_379_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_379_, 0, v_toAddCommMonoid_376_);
lean_ctor_set(v___x_379_, 1, v_toNeg_372_);
lean_ctor_set(v___x_379_, 2, v_toSub_373_);
lean_ctor_set(v___x_379_, 3, v_toZSMul_374_);
v_toOne_380_ = lean_ctor_get(v_toMonoid_377_, 0);
v_toMul_381_ = lean_ctor_get(v_toMonoid_377_, 1);
lean_inc(v_toMul_381_);
v___x_382_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_382_, 0, v___x_379_);
lean_ctor_set(v___x_382_, 1, v_toMul_381_);
lean_inc(v_toIntCast_375_);
lean_inc(v_toNatCast_378_);
lean_inc(v_toOne_380_);
v___x_383_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_383_, 0, v___x_382_);
lean_ctor_set(v___x_383_, 1, v_toOne_380_);
lean_ctor_set(v___x_383_, 2, v_toNatCast_378_);
lean_ctor_set(v___x_383_, 3, v_toIntCast_375_);
return v___x_383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_toNonAssocRing___redArg___boxed(lean_object* v_inst_384_){
_start:
{
lean_object* v_res_385_; 
v_res_385_ = lp_mathlib_Ring_toNonAssocRing___redArg(v_inst_384_);
lean_dec_ref(v_inst_384_);
return v_res_385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_toNonAssocRing(lean_object* v_00_u03b1_386_, lean_object* v_inst_387_){
_start:
{
lean_object* v___x_388_; 
v___x_388_ = lp_mathlib_Ring_toNonAssocRing___redArg(v_inst_387_);
return v___x_388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_toNonAssocRing___boxed(lean_object* v_00_u03b1_389_, lean_object* v_inst_390_){
_start:
{
lean_object* v_res_391_; 
v_res_391_ = lp_mathlib_Ring_toNonAssocRing(v_00_u03b1_389_, v_inst_390_);
lean_dec_ref(v_inst_390_);
return v_res_391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocCommRing_toNonUnitalNonAssocCommSemiring___redArg(lean_object* v_self_392_){
_start:
{
lean_object* v_toAddCommGroup_393_; lean_object* v_toMul_394_; lean_object* v___x_396_; uint8_t v_isShared_397_; uint8_t v_isSharedCheck_402_; 
v_toAddCommGroup_393_ = lean_ctor_get(v_self_392_, 0);
v_toMul_394_ = lean_ctor_get(v_self_392_, 1);
v_isSharedCheck_402_ = !lean_is_exclusive(v_self_392_);
if (v_isSharedCheck_402_ == 0)
{
v___x_396_ = v_self_392_;
v_isShared_397_ = v_isSharedCheck_402_;
goto v_resetjp_395_;
}
else
{
lean_inc(v_toMul_394_);
lean_inc(v_toAddCommGroup_393_);
lean_dec(v_self_392_);
v___x_396_ = lean_box(0);
v_isShared_397_ = v_isSharedCheck_402_;
goto v_resetjp_395_;
}
v_resetjp_395_:
{
lean_object* v_toAddMonoid_398_; lean_object* v___x_400_; 
v_toAddMonoid_398_ = lean_ctor_get(v_toAddCommGroup_393_, 0);
lean_inc_ref(v_toAddMonoid_398_);
lean_dec_ref(v_toAddCommGroup_393_);
if (v_isShared_397_ == 0)
{
lean_ctor_set(v___x_396_, 0, v_toAddMonoid_398_);
v___x_400_ = v___x_396_;
goto v_reusejp_399_;
}
else
{
lean_object* v_reuseFailAlloc_401_; 
v_reuseFailAlloc_401_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_401_, 0, v_toAddMonoid_398_);
lean_ctor_set(v_reuseFailAlloc_401_, 1, v_toMul_394_);
v___x_400_ = v_reuseFailAlloc_401_;
goto v_reusejp_399_;
}
v_reusejp_399_:
{
return v___x_400_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocCommRing_toNonUnitalNonAssocCommSemiring(lean_object* v_00_u03b1_403_, lean_object* v_self_404_){
_start:
{
lean_object* v___x_405_; 
v___x_405_ = lp_mathlib_NonUnitalNonAssocCommRing_toNonUnitalNonAssocCommSemiring___redArg(v_self_404_);
return v___x_405_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommRing_toNonUnitalNonAssocCommRing___redArg(lean_object* v_self_406_){
_start:
{
lean_inc_ref(v_self_406_);
return v_self_406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommRing_toNonUnitalNonAssocCommRing___redArg___boxed(lean_object* v_self_407_){
_start:
{
lean_object* v_res_408_; 
v_res_408_ = lp_mathlib_NonUnitalCommRing_toNonUnitalNonAssocCommRing___redArg(v_self_407_);
lean_dec_ref(v_self_407_);
return v_res_408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommRing_toNonUnitalNonAssocCommRing(lean_object* v_00_u03b1_409_, lean_object* v_self_410_){
_start:
{
lean_inc_ref(v_self_410_);
return v_self_410_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommRing_toNonUnitalNonAssocCommRing___boxed(lean_object* v_00_u03b1_411_, lean_object* v_self_412_){
_start:
{
lean_object* v_res_413_; 
v_res_413_ = lp_mathlib_NonUnitalCommRing_toNonUnitalNonAssocCommRing(v_00_u03b1_411_, v_self_412_);
lean_dec_ref(v_self_412_);
return v_res_413_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommRing_toNonUnitalNonAssocCommRing___redArg(lean_object* v_self_414_){
_start:
{
lean_object* v_toNonUnitalNonAssocRing_415_; 
v_toNonUnitalNonAssocRing_415_ = lean_ctor_get(v_self_414_, 0);
lean_inc_ref(v_toNonUnitalNonAssocRing_415_);
return v_toNonUnitalNonAssocRing_415_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommRing_toNonUnitalNonAssocCommRing___redArg___boxed(lean_object* v_self_416_){
_start:
{
lean_object* v_res_417_; 
v_res_417_ = lp_mathlib_NonAssocCommRing_toNonUnitalNonAssocCommRing___redArg(v_self_416_);
lean_dec_ref(v_self_416_);
return v_res_417_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommRing_toNonUnitalNonAssocCommRing(lean_object* v_00_u03b1_418_, lean_object* v_self_419_){
_start:
{
lean_object* v_toNonUnitalNonAssocRing_420_; 
v_toNonUnitalNonAssocRing_420_ = lean_ctor_get(v_self_419_, 0);
lean_inc_ref(v_toNonUnitalNonAssocRing_420_);
return v_toNonUnitalNonAssocRing_420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommRing_toNonUnitalNonAssocCommRing___boxed(lean_object* v_00_u03b1_421_, lean_object* v_self_422_){
_start:
{
lean_object* v_res_423_; 
v_res_423_ = lp_mathlib_NonAssocCommRing_toNonUnitalNonAssocCommRing(v_00_u03b1_421_, v_self_422_);
lean_dec_ref(v_self_422_);
return v_res_423_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommRing_toNonAssocCommSemiring___redArg(lean_object* v_self_424_){
_start:
{
lean_object* v_toNonUnitalNonAssocRing_425_; lean_object* v_toAddCommGroup_426_; lean_object* v_toOne_427_; lean_object* v_toNatCast_428_; lean_object* v_toMul_429_; lean_object* v___x_431_; uint8_t v_isShared_432_; uint8_t v_isSharedCheck_438_; 
v_toNonUnitalNonAssocRing_425_ = lean_ctor_get(v_self_424_, 0);
lean_inc_ref(v_toNonUnitalNonAssocRing_425_);
v_toAddCommGroup_426_ = lean_ctor_get(v_toNonUnitalNonAssocRing_425_, 0);
lean_inc_ref(v_toAddCommGroup_426_);
v_toOne_427_ = lean_ctor_get(v_self_424_, 1);
lean_inc(v_toOne_427_);
v_toNatCast_428_ = lean_ctor_get(v_self_424_, 2);
lean_inc(v_toNatCast_428_);
lean_dec_ref(v_self_424_);
v_toMul_429_ = lean_ctor_get(v_toNonUnitalNonAssocRing_425_, 1);
v_isSharedCheck_438_ = !lean_is_exclusive(v_toNonUnitalNonAssocRing_425_);
if (v_isSharedCheck_438_ == 0)
{
lean_object* v_unused_439_; 
v_unused_439_ = lean_ctor_get(v_toNonUnitalNonAssocRing_425_, 0);
lean_dec(v_unused_439_);
v___x_431_ = v_toNonUnitalNonAssocRing_425_;
v_isShared_432_ = v_isSharedCheck_438_;
goto v_resetjp_430_;
}
else
{
lean_inc(v_toMul_429_);
lean_dec(v_toNonUnitalNonAssocRing_425_);
v___x_431_ = lean_box(0);
v_isShared_432_ = v_isSharedCheck_438_;
goto v_resetjp_430_;
}
v_resetjp_430_:
{
lean_object* v_toAddMonoid_433_; lean_object* v___x_435_; 
v_toAddMonoid_433_ = lean_ctor_get(v_toAddCommGroup_426_, 0);
lean_inc_ref(v_toAddMonoid_433_);
lean_dec_ref(v_toAddCommGroup_426_);
if (v_isShared_432_ == 0)
{
lean_ctor_set(v___x_431_, 0, v_toAddMonoid_433_);
v___x_435_ = v___x_431_;
goto v_reusejp_434_;
}
else
{
lean_object* v_reuseFailAlloc_437_; 
v_reuseFailAlloc_437_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_437_, 0, v_toAddMonoid_433_);
lean_ctor_set(v_reuseFailAlloc_437_, 1, v_toMul_429_);
v___x_435_ = v_reuseFailAlloc_437_;
goto v_reusejp_434_;
}
v_reusejp_434_:
{
lean_object* v___x_436_; 
v___x_436_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_436_, 0, v___x_435_);
lean_ctor_set(v___x_436_, 1, v_toOne_427_);
lean_ctor_set(v___x_436_, 2, v_toNatCast_428_);
return v___x_436_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonAssocCommRing_toNonAssocCommSemiring(lean_object* v_00_u03b1_440_, lean_object* v_self_441_){
_start:
{
lean_object* v___x_442_; 
v___x_442_ = lp_mathlib_NonAssocCommRing_toNonAssocCommSemiring___redArg(v_self_441_);
return v___x_442_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommRing_toNonUnitalCommSemiring___redArg(lean_object* v_s_443_){
_start:
{
lean_object* v_toAddCommGroup_444_; lean_object* v_toMul_445_; lean_object* v___x_447_; uint8_t v_isShared_448_; uint8_t v_isSharedCheck_453_; 
v_toAddCommGroup_444_ = lean_ctor_get(v_s_443_, 0);
v_toMul_445_ = lean_ctor_get(v_s_443_, 1);
v_isSharedCheck_453_ = !lean_is_exclusive(v_s_443_);
if (v_isSharedCheck_453_ == 0)
{
v___x_447_ = v_s_443_;
v_isShared_448_ = v_isSharedCheck_453_;
goto v_resetjp_446_;
}
else
{
lean_inc(v_toMul_445_);
lean_inc(v_toAddCommGroup_444_);
lean_dec(v_s_443_);
v___x_447_ = lean_box(0);
v_isShared_448_ = v_isSharedCheck_453_;
goto v_resetjp_446_;
}
v_resetjp_446_:
{
lean_object* v_toAddMonoid_449_; lean_object* v___x_451_; 
v_toAddMonoid_449_ = lean_ctor_get(v_toAddCommGroup_444_, 0);
lean_inc_ref(v_toAddMonoid_449_);
lean_dec_ref(v_toAddCommGroup_444_);
if (v_isShared_448_ == 0)
{
lean_ctor_set(v___x_447_, 0, v_toAddMonoid_449_);
v___x_451_ = v___x_447_;
goto v_reusejp_450_;
}
else
{
lean_object* v_reuseFailAlloc_452_; 
v_reuseFailAlloc_452_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_452_, 0, v_toAddMonoid_449_);
lean_ctor_set(v_reuseFailAlloc_452_, 1, v_toMul_445_);
v___x_451_ = v_reuseFailAlloc_452_;
goto v_reusejp_450_;
}
v_reusejp_450_:
{
return v___x_451_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalCommRing_toNonUnitalCommSemiring(lean_object* v_00_u03b1_454_, lean_object* v_s_455_){
_start:
{
lean_object* v___x_456_; 
v___x_456_ = lp_mathlib_NonUnitalCommRing_toNonUnitalCommSemiring___redArg(v_s_455_);
return v___x_456_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommMonoid___redArg(lean_object* v_self_457_){
_start:
{
lean_object* v_toSemiring_458_; lean_object* v_toMonoid_459_; 
v_toSemiring_458_ = lean_ctor_get(v_self_457_, 0);
v_toMonoid_459_ = lean_ctor_get(v_toSemiring_458_, 1);
lean_inc_ref(v_toMonoid_459_);
return v_toMonoid_459_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommMonoid___redArg___boxed(lean_object* v_self_460_){
_start:
{
lean_object* v_res_461_; 
v_res_461_ = lp_mathlib_CommRing_toCommMonoid___redArg(v_self_460_);
lean_dec_ref(v_self_460_);
return v_res_461_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommMonoid(lean_object* v_00_u03b1_462_, lean_object* v_self_463_){
_start:
{
lean_object* v___x_464_; 
v___x_464_ = lp_mathlib_CommRing_toCommMonoid___redArg(v_self_463_);
return v___x_464_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommMonoid___boxed(lean_object* v_00_u03b1_465_, lean_object* v_self_466_){
_start:
{
lean_object* v_res_467_; 
v_res_467_ = lp_mathlib_CommRing_toCommMonoid(v_00_u03b1_465_, v_self_466_);
lean_dec_ref(v_self_466_);
return v_res_467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toNonAssocCommRing___redArg(lean_object* v_inst_468_){
_start:
{
lean_object* v___x_469_; 
v___x_469_ = lp_mathlib_Ring_toNonAssocRing___redArg(v_inst_468_);
return v___x_469_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toNonAssocCommRing___redArg___boxed(lean_object* v_inst_470_){
_start:
{
lean_object* v_res_471_; 
v_res_471_ = lp_mathlib_CommRing_toNonAssocCommRing___redArg(v_inst_470_);
lean_dec_ref(v_inst_470_);
return v_res_471_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toNonAssocCommRing(lean_object* v_00_u03b1_472_, lean_object* v_inst_473_){
_start:
{
lean_object* v___x_474_; 
v___x_474_ = lp_mathlib_Ring_toNonAssocRing___redArg(v_inst_473_);
return v___x_474_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toNonAssocCommRing___boxed(lean_object* v_00_u03b1_475_, lean_object* v_inst_476_){
_start:
{
lean_object* v_res_477_; 
v_res_477_ = lp_mathlib_CommRing_toNonAssocCommRing(v_00_u03b1_475_, v_inst_476_);
lean_dec_ref(v_inst_476_);
return v_res_477_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommSemiring___redArg(lean_object* v_s_478_){
_start:
{
lean_object* v_toSemiring_479_; 
v_toSemiring_479_ = lean_ctor_get(v_s_478_, 0);
lean_inc_ref(v_toSemiring_479_);
return v_toSemiring_479_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommSemiring___redArg___boxed(lean_object* v_s_480_){
_start:
{
lean_object* v_res_481_; 
v_res_481_ = lp_mathlib_CommRing_toCommSemiring___redArg(v_s_480_);
lean_dec_ref(v_s_480_);
return v_res_481_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommSemiring(lean_object* v_00_u03b1_482_, lean_object* v_s_483_){
_start:
{
lean_object* v_toSemiring_484_; 
v_toSemiring_484_ = lean_ctor_get(v_s_483_, 0);
lean_inc_ref(v_toSemiring_484_);
return v_toSemiring_484_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toCommSemiring___boxed(lean_object* v_00_u03b1_485_, lean_object* v_s_486_){
_start:
{
lean_object* v_res_487_; 
v_res_487_ = lp_mathlib_CommRing_toCommSemiring(v_00_u03b1_485_, v_s_486_);
lean_dec_ref(v_s_486_);
return v_res_487_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toNonUnitalCommRing___redArg(lean_object* v_s_488_){
_start:
{
lean_object* v_toSemiring_489_; lean_object* v_toNeg_490_; lean_object* v_toSub_491_; lean_object* v_toZSMul_492_; lean_object* v_toAddCommMonoid_493_; lean_object* v_toMonoid_494_; lean_object* v___x_495_; lean_object* v_toMul_496_; lean_object* v___x_497_; 
v_toSemiring_489_ = lean_ctor_get(v_s_488_, 0);
v_toNeg_490_ = lean_ctor_get(v_s_488_, 1);
v_toSub_491_ = lean_ctor_get(v_s_488_, 2);
v_toZSMul_492_ = lean_ctor_get(v_s_488_, 3);
v_toAddCommMonoid_493_ = lean_ctor_get(v_toSemiring_489_, 0);
v_toMonoid_494_ = lean_ctor_get(v_toSemiring_489_, 1);
lean_inc(v_toZSMul_492_);
lean_inc(v_toSub_491_);
lean_inc(v_toNeg_490_);
lean_inc_ref(v_toAddCommMonoid_493_);
v___x_495_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_495_, 0, v_toAddCommMonoid_493_);
lean_ctor_set(v___x_495_, 1, v_toNeg_490_);
lean_ctor_set(v___x_495_, 2, v_toSub_491_);
lean_ctor_set(v___x_495_, 3, v_toZSMul_492_);
v_toMul_496_ = lean_ctor_get(v_toMonoid_494_, 1);
lean_inc(v_toMul_496_);
v___x_497_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_497_, 0, v___x_495_);
lean_ctor_set(v___x_497_, 1, v_toMul_496_);
return v___x_497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toNonUnitalCommRing___redArg___boxed(lean_object* v_s_498_){
_start:
{
lean_object* v_res_499_; 
v_res_499_ = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(v_s_498_);
lean_dec_ref(v_s_498_);
return v_res_499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toNonUnitalCommRing(lean_object* v_00_u03b1_500_, lean_object* v_s_501_){
_start:
{
lean_object* v___x_502_; 
v___x_502_ = lp_mathlib_CommRing_toNonUnitalCommRing___redArg(v_s_501_);
return v___x_502_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toNonUnitalCommRing___boxed(lean_object* v_00_u03b1_503_, lean_object* v_s_504_){
_start:
{
lean_object* v_res_505_; 
v_res_505_ = lp_mathlib_CommRing_toNonUnitalCommRing(v_00_u03b1_503_, v_s_504_);
lean_dec_ref(v_s_504_);
return v_res_505_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toAddCommGroupWithOne___redArg(lean_object* v_s_506_){
_start:
{
lean_object* v_toSemiring_507_; lean_object* v_toNeg_508_; lean_object* v_toSub_509_; lean_object* v_toZSMul_510_; lean_object* v_toIntCast_511_; lean_object* v_toAddCommMonoid_512_; lean_object* v_toMonoid_513_; lean_object* v_toNatCast_514_; lean_object* v___x_515_; lean_object* v_toOne_516_; lean_object* v___x_517_; 
v_toSemiring_507_ = lean_ctor_get(v_s_506_, 0);
v_toNeg_508_ = lean_ctor_get(v_s_506_, 1);
v_toSub_509_ = lean_ctor_get(v_s_506_, 2);
v_toZSMul_510_ = lean_ctor_get(v_s_506_, 3);
v_toIntCast_511_ = lean_ctor_get(v_s_506_, 4);
v_toAddCommMonoid_512_ = lean_ctor_get(v_toSemiring_507_, 0);
v_toMonoid_513_ = lean_ctor_get(v_toSemiring_507_, 1);
v_toNatCast_514_ = lean_ctor_get(v_toSemiring_507_, 2);
lean_inc(v_toZSMul_510_);
lean_inc(v_toSub_509_);
lean_inc(v_toNeg_508_);
lean_inc_ref(v_toAddCommMonoid_512_);
v___x_515_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_515_, 0, v_toAddCommMonoid_512_);
lean_ctor_set(v___x_515_, 1, v_toNeg_508_);
lean_ctor_set(v___x_515_, 2, v_toSub_509_);
lean_ctor_set(v___x_515_, 3, v_toZSMul_510_);
v_toOne_516_ = lean_ctor_get(v_toMonoid_513_, 0);
lean_inc(v_toOne_516_);
lean_inc(v_toNatCast_514_);
lean_inc(v_toIntCast_511_);
v___x_517_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_517_, 0, v___x_515_);
lean_ctor_set(v___x_517_, 1, v_toIntCast_511_);
lean_ctor_set(v___x_517_, 2, v_toNatCast_514_);
lean_ctor_set(v___x_517_, 3, v_toOne_516_);
return v___x_517_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toAddCommGroupWithOne___redArg___boxed(lean_object* v_s_518_){
_start:
{
lean_object* v_res_519_; 
v_res_519_ = lp_mathlib_CommRing_toAddCommGroupWithOne___redArg(v_s_518_);
lean_dec_ref(v_s_518_);
return v_res_519_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toAddCommGroupWithOne(lean_object* v_00_u03b1_520_, lean_object* v_s_521_){
_start:
{
lean_object* v___x_522_; 
v___x_522_ = lp_mathlib_CommRing_toAddCommGroupWithOne___redArg(v_s_521_);
return v___x_522_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_toAddCommGroupWithOne___boxed(lean_object* v_00_u03b1_523_, lean_object* v_s_524_){
_start:
{
lean_object* v_res_525_; 
v_res_525_ = lp_mathlib_CommRing_toAddCommGroupWithOne(v_00_u03b1_523_, v_s_524_);
lean_dec_ref(v_s_524_);
return v_res_525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalNonAssocCommSemiring___redArg(lean_object* v_inst_526_){
_start:
{
lean_inc_ref(v_inst_526_);
return v_inst_526_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalNonAssocCommSemiring___redArg___boxed(lean_object* v_inst_527_){
_start:
{
lean_object* v_res_528_; 
v_res_528_ = lp_mathlib_IsMulCommutative_instNonUnitalNonAssocCommSemiring___redArg(v_inst_527_);
lean_dec_ref(v_inst_527_);
return v_res_528_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalNonAssocCommSemiring(lean_object* v_R_529_, lean_object* v_inst_530_, lean_object* v_inst_531_){
_start:
{
lean_inc_ref(v_inst_530_);
return v_inst_530_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalNonAssocCommSemiring___boxed(lean_object* v_R_532_, lean_object* v_inst_533_, lean_object* v_inst_534_){
_start:
{
lean_object* v_res_535_; 
v_res_535_ = lp_mathlib_IsMulCommutative_instNonUnitalNonAssocCommSemiring(v_R_532_, v_inst_533_, v_inst_534_);
lean_dec_ref(v_inst_533_);
return v_res_535_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalCommSemiring___redArg(lean_object* v_inst_536_){
_start:
{
lean_inc_ref(v_inst_536_);
return v_inst_536_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalCommSemiring___redArg___boxed(lean_object* v_inst_537_){
_start:
{
lean_object* v_res_538_; 
v_res_538_ = lp_mathlib_IsMulCommutative_instNonUnitalCommSemiring___redArg(v_inst_537_);
lean_dec_ref(v_inst_537_);
return v_res_538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalCommSemiring(lean_object* v_R_539_, lean_object* v_inst_540_, lean_object* v_inst_541_){
_start:
{
lean_inc_ref(v_inst_540_);
return v_inst_540_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalCommSemiring___boxed(lean_object* v_R_542_, lean_object* v_inst_543_, lean_object* v_inst_544_){
_start:
{
lean_object* v_res_545_; 
v_res_545_ = lp_mathlib_IsMulCommutative_instNonUnitalCommSemiring(v_R_542_, v_inst_543_, v_inst_544_);
lean_dec_ref(v_inst_543_);
return v_res_545_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalNonAssocCommRing___redArg(lean_object* v_inst_546_){
_start:
{
lean_inc_ref(v_inst_546_);
return v_inst_546_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalNonAssocCommRing___redArg___boxed(lean_object* v_inst_547_){
_start:
{
lean_object* v_res_548_; 
v_res_548_ = lp_mathlib_IsMulCommutative_instNonUnitalNonAssocCommRing___redArg(v_inst_547_);
lean_dec_ref(v_inst_547_);
return v_res_548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalNonAssocCommRing(lean_object* v_R_549_, lean_object* v_inst_550_, lean_object* v_inst_551_){
_start:
{
lean_inc_ref(v_inst_550_);
return v_inst_550_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalNonAssocCommRing___boxed(lean_object* v_R_552_, lean_object* v_inst_553_, lean_object* v_inst_554_){
_start:
{
lean_object* v_res_555_; 
v_res_555_ = lp_mathlib_IsMulCommutative_instNonUnitalNonAssocCommRing(v_R_552_, v_inst_553_, v_inst_554_);
lean_dec_ref(v_inst_553_);
return v_res_555_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalCommRing___redArg(lean_object* v_inst_556_){
_start:
{
lean_inc_ref(v_inst_556_);
return v_inst_556_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalCommRing___redArg___boxed(lean_object* v_inst_557_){
_start:
{
lean_object* v_res_558_; 
v_res_558_ = lp_mathlib_IsMulCommutative_instNonUnitalCommRing___redArg(v_inst_557_);
lean_dec_ref(v_inst_557_);
return v_res_558_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalCommRing(lean_object* v_R_559_, lean_object* v_inst_560_, lean_object* v_inst_561_){
_start:
{
lean_inc_ref(v_inst_560_);
return v_inst_560_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonUnitalCommRing___boxed(lean_object* v_R_562_, lean_object* v_inst_563_, lean_object* v_inst_564_){
_start:
{
lean_object* v_res_565_; 
v_res_565_ = lp_mathlib_IsMulCommutative_instNonUnitalCommRing(v_R_562_, v_inst_563_, v_inst_564_);
lean_dec_ref(v_inst_563_);
return v_res_565_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonAssocCommSemiring___redArg(lean_object* v_inst_566_){
_start:
{
lean_inc_ref(v_inst_566_);
return v_inst_566_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonAssocCommSemiring___redArg___boxed(lean_object* v_inst_567_){
_start:
{
lean_object* v_res_568_; 
v_res_568_ = lp_mathlib_IsMulCommutative_instNonAssocCommSemiring___redArg(v_inst_567_);
lean_dec_ref(v_inst_567_);
return v_res_568_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonAssocCommSemiring(lean_object* v_R_569_, lean_object* v_inst_570_, lean_object* v_inst_571_){
_start:
{
lean_inc_ref(v_inst_570_);
return v_inst_570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonAssocCommSemiring___boxed(lean_object* v_R_572_, lean_object* v_inst_573_, lean_object* v_inst_574_){
_start:
{
lean_object* v_res_575_; 
v_res_575_ = lp_mathlib_IsMulCommutative_instNonAssocCommSemiring(v_R_572_, v_inst_573_, v_inst_574_);
lean_dec_ref(v_inst_573_);
return v_res_575_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instCommSemiring___redArg(lean_object* v_inst_576_){
_start:
{
lean_inc_ref(v_inst_576_);
return v_inst_576_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instCommSemiring___redArg___boxed(lean_object* v_inst_577_){
_start:
{
lean_object* v_res_578_; 
v_res_578_ = lp_mathlib_IsMulCommutative_instCommSemiring___redArg(v_inst_577_);
lean_dec_ref(v_inst_577_);
return v_res_578_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instCommSemiring(lean_object* v_R_579_, lean_object* v_inst_580_, lean_object* v_inst_581_){
_start:
{
lean_inc_ref(v_inst_580_);
return v_inst_580_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instCommSemiring___boxed(lean_object* v_R_582_, lean_object* v_inst_583_, lean_object* v_inst_584_){
_start:
{
lean_object* v_res_585_; 
v_res_585_ = lp_mathlib_IsMulCommutative_instCommSemiring(v_R_582_, v_inst_583_, v_inst_584_);
lean_dec_ref(v_inst_583_);
return v_res_585_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonAssocCommRing___redArg(lean_object* v_inst_586_){
_start:
{
lean_inc_ref(v_inst_586_);
return v_inst_586_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonAssocCommRing___redArg___boxed(lean_object* v_inst_587_){
_start:
{
lean_object* v_res_588_; 
v_res_588_ = lp_mathlib_IsMulCommutative_instNonAssocCommRing___redArg(v_inst_587_);
lean_dec_ref(v_inst_587_);
return v_res_588_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonAssocCommRing(lean_object* v_R_589_, lean_object* v_inst_590_, lean_object* v_inst_591_){
_start:
{
lean_inc_ref(v_inst_590_);
return v_inst_590_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instNonAssocCommRing___boxed(lean_object* v_R_592_, lean_object* v_inst_593_, lean_object* v_inst_594_){
_start:
{
lean_object* v_res_595_; 
v_res_595_ = lp_mathlib_IsMulCommutative_instNonAssocCommRing(v_R_592_, v_inst_593_, v_inst_594_);
lean_dec_ref(v_inst_593_);
return v_res_595_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instCommRing___redArg(lean_object* v_inst_596_){
_start:
{
lean_inc_ref(v_inst_596_);
return v_inst_596_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instCommRing___redArg___boxed(lean_object* v_inst_597_){
_start:
{
lean_object* v_res_598_; 
v_res_598_ = lp_mathlib_IsMulCommutative_instCommRing___redArg(v_inst_597_);
lean_dec_ref(v_inst_597_);
return v_res_598_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instCommRing(lean_object* v_R_599_, lean_object* v_inst_600_, lean_object* v_inst_601_){
_start:
{
lean_inc_ref(v_inst_600_);
return v_inst_600_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsMulCommutative_instCommRing___boxed(lean_object* v_R_602_, lean_object* v_inst_603_, lean_object* v_inst_604_){
_start:
{
lean_object* v_res_605_; 
v_res_605_ = lp_mathlib_IsMulCommutative_instCommRing(v_R_602_, v_inst_603_, v_inst_604_);
lean_dec_ref(v_inst_603_);
return v_res_605_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_IsCommutative(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Notation_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Spread(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_IsCommutative(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Notation_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Spread(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_IsCommutative(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Notation_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Cast_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Spread(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_IsCommutative(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Notation_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Cast_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Spread(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
