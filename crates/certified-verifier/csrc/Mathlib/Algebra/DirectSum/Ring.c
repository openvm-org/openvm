// Lean compiler output
// Module: Mathlib.Algebra.DirectSum.Ring
// Imports: public import Init public meta import Init public import Mathlib.Algebra.GradedMonoid public import Mathlib.Algebra.DirectSum.Basic public import Mathlib.Algebra.Ring.Associator
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
lean_object* lp_mathlib_DirectSum_of___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DirectSum_toAddMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_GradedMonoid_GradeZero_mul___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_NSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ZSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_subNegMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Monoid_gMonoid___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
lean_object* lp_mathlib_Mul_gMul___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_MonoidHom_compHom___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OneHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MonoidHom_flip___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DirectSum_instAddCommMonoid___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoidHom_instAddCommMonoid___redArg(lean_object*);
lean_object* lp_mathlib_npowBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_GradedMonoid_GradeZero_smul___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Nat_cast(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_addMonoid___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Int_cast(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DirectSum_instAddCommGroup___redArg(lean_object*);
lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GSemiring_toGMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GSemiring_toGMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GSemiring_toGMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GSemiring_toGMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommSemiring_toGCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommSemiring_toGCommMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommSemiring_toGCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommSemiring_toGCommMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommRing_toGCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommRing_toGCommSemiring___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommRing_toGCommSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommRing_toGCommSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instOne___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_gMulHom___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_gMulHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_gMulHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_gMulHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_mulHom___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_mulHom___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_mulHom___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_mulHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instMul___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNonUnitalNonAssocSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNonUnitalNonAssocSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCast___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCast___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCast___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_semiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_semiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_commSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_commSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_nonAssocRing___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_nonAssocRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_nonAssocRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_ring___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_ring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_ring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_commRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_commRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNonUnitalNonAssocSemiringOfNat___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNonUnitalNonAssocSemiringOfNat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNonUnitalNonAssocSemiringOfNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instSMulWithZeroOfNat___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instSMulWithZeroOfNat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instSMulWithZeroOfNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCastOfNat___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCastOfNat___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCastOfNat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCastOfNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instSemiringOfNat___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instSemiringOfNat___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instSemiringOfNat___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instSemiringOfNat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instSemiringOfNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_ofZeroRingHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_ofZeroRingHom___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_ofZeroRingHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_ofZeroRingHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instModuleOfNat___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instModuleOfNat___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instModuleOfNat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instModuleOfNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instCommSemiringOfNat___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instCommSemiringOfNat___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instCommSemiringOfNat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instCommSemiringOfNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNonUnitalNonAssocRingOfNat___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNonUnitalNonAssocRingOfNat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNonUnitalNonAssocRingOfNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instIntCastOfNat___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instIntCastOfNat___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instIntCastOfNat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instIntCastOfNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instRingOfNat___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instRingOfNat___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instRingOfNat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instRingOfNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instCommRingOfNat___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instCommRingOfNat___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instCommRingOfNat___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instCommRingOfNat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instCommRingOfNat___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toSemiring___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_liftRingHom___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_liftRingHom___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_liftRingHom___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_liftRingHom___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_liftRingHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_liftRingHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_liftRingHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocSemiring_directSumGNonUnitalNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocSemiring_directSumGNonUnitalNonAssocSemiring(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocSemiring_directSumGNonUnitalNonAssocSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_directSumGSemiring___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_directSumGSemiring___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_directSumGSemiring___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_directSumGSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_directSumGSemiring(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semiring_directSumGSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_directSumGRing___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_directSumGRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_directSumGRing(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ring_directSumGRing___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_directSumGCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_directSumGCommSemiring(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_directSumGCommSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_directSumGCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_directSumGCommRing(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRing_directSumGCommRing___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GSemiring_toGMonoid___redArg(lean_object* v_self_1_){
_start:
{
lean_object* v_toGNonUnitalNonAssocSemiring_2_; lean_object* v_toGOne_3_; lean_object* v_gnpow_4_; lean_object* v___x_5_; 
v_toGNonUnitalNonAssocSemiring_2_ = lean_ctor_get(v_self_1_, 0);
v_toGOne_3_ = lean_ctor_get(v_self_1_, 1);
v_gnpow_4_ = lean_ctor_get(v_self_1_, 2);
lean_inc(v_gnpow_4_);
lean_inc(v_toGOne_3_);
lean_inc(v_toGNonUnitalNonAssocSemiring_2_);
v___x_5_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_5_, 0, v_toGNonUnitalNonAssocSemiring_2_);
lean_ctor_set(v___x_5_, 1, v_toGOne_3_);
lean_ctor_set(v___x_5_, 2, v_gnpow_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GSemiring_toGMonoid___redArg___boxed(lean_object* v_self_6_){
_start:
{
lean_object* v_res_7_; 
v_res_7_ = lp_mathlib_DirectSum_GSemiring_toGMonoid___redArg(v_self_6_);
lean_dec_ref(v_self_6_);
return v_res_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GSemiring_toGMonoid(lean_object* v_00_u03b9_8_, lean_object* v_A_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_self_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lp_mathlib_DirectSum_GSemiring_toGMonoid___redArg(v_self_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GSemiring_toGMonoid___boxed(lean_object* v_00_u03b9_14_, lean_object* v_A_15_, lean_object* v_inst_16_, lean_object* v_inst_17_, lean_object* v_self_18_){
_start:
{
lean_object* v_res_19_; 
v_res_19_ = lp_mathlib_DirectSum_GSemiring_toGMonoid(v_00_u03b9_14_, v_A_15_, v_inst_16_, v_inst_17_, v_self_18_);
lean_dec_ref(v_self_18_);
lean_dec_ref(v_inst_17_);
lean_dec_ref(v_inst_16_);
return v_res_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommSemiring_toGCommMonoid___redArg(lean_object* v_self_20_){
_start:
{
lean_object* v_toGNonUnitalNonAssocSemiring_21_; lean_object* v_toGOne_22_; lean_object* v_gnpow_23_; lean_object* v___x_24_; 
v_toGNonUnitalNonAssocSemiring_21_ = lean_ctor_get(v_self_20_, 0);
v_toGOne_22_ = lean_ctor_get(v_self_20_, 1);
v_gnpow_23_ = lean_ctor_get(v_self_20_, 2);
lean_inc(v_gnpow_23_);
lean_inc(v_toGOne_22_);
lean_inc(v_toGNonUnitalNonAssocSemiring_21_);
v___x_24_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_24_, 0, v_toGNonUnitalNonAssocSemiring_21_);
lean_ctor_set(v___x_24_, 1, v_toGOne_22_);
lean_ctor_set(v___x_24_, 2, v_gnpow_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommSemiring_toGCommMonoid___redArg___boxed(lean_object* v_self_25_){
_start:
{
lean_object* v_res_26_; 
v_res_26_ = lp_mathlib_DirectSum_GCommSemiring_toGCommMonoid___redArg(v_self_25_);
lean_dec_ref(v_self_25_);
return v_res_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommSemiring_toGCommMonoid(lean_object* v_00_u03b9_27_, lean_object* v_A_28_, lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_self_31_){
_start:
{
lean_object* v___x_32_; 
v___x_32_ = lp_mathlib_DirectSum_GCommSemiring_toGCommMonoid___redArg(v_self_31_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommSemiring_toGCommMonoid___boxed(lean_object* v_00_u03b9_33_, lean_object* v_A_34_, lean_object* v_inst_35_, lean_object* v_inst_36_, lean_object* v_self_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_mathlib_DirectSum_GCommSemiring_toGCommMonoid(v_00_u03b9_33_, v_A_34_, v_inst_35_, v_inst_36_, v_self_37_);
lean_dec_ref(v_self_37_);
lean_dec_ref(v_inst_36_);
lean_dec_ref(v_inst_35_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommRing_toGCommSemiring___redArg(lean_object* v_self_39_){
_start:
{
lean_object* v_toGSemiring_40_; 
v_toGSemiring_40_ = lean_ctor_get(v_self_39_, 0);
lean_inc_ref(v_toGSemiring_40_);
return v_toGSemiring_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommRing_toGCommSemiring___redArg___boxed(lean_object* v_self_41_){
_start:
{
lean_object* v_res_42_; 
v_res_42_ = lp_mathlib_DirectSum_GCommRing_toGCommSemiring___redArg(v_self_41_);
lean_dec_ref(v_self_41_);
return v_res_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommRing_toGCommSemiring(lean_object* v_00_u03b9_43_, lean_object* v_A_44_, lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_self_47_){
_start:
{
lean_object* v_toGSemiring_48_; 
v_toGSemiring_48_ = lean_ctor_get(v_self_47_, 0);
lean_inc_ref(v_toGSemiring_48_);
return v_toGSemiring_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_GCommRing_toGCommSemiring___boxed(lean_object* v_00_u03b9_49_, lean_object* v_A_50_, lean_object* v_inst_51_, lean_object* v_inst_52_, lean_object* v_self_53_){
_start:
{
lean_object* v_res_54_; 
v_res_54_ = lp_mathlib_DirectSum_GCommRing_toGCommSemiring(v_00_u03b9_49_, v_A_50_, v_inst_51_, v_inst_52_, v_self_53_);
lean_dec_ref(v_self_53_);
lean_dec_ref(v_inst_52_);
lean_dec_ref(v_inst_51_);
return v_res_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instOne___redArg(lean_object* v_inst_55_, lean_object* v_inst_56_, lean_object* v_inst_57_, lean_object* v_inst_58_){
_start:
{
lean_object* v___x_24__overap_59_; lean_object* v___x_60_; 
v___x_24__overap_59_ = lp_mathlib_DirectSum_of___redArg(v_inst_58_, v_inst_55_, v_inst_56_);
v___x_60_ = lean_apply_1(v___x_24__overap_59_, v_inst_57_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instOne(lean_object* v_00_u03b9_61_, lean_object* v_inst_62_, lean_object* v_A_63_, lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_inst_66_){
_start:
{
lean_object* v___x_30__overap_67_; lean_object* v___x_68_; 
v___x_30__overap_67_ = lp_mathlib_DirectSum_of___redArg(v_inst_66_, v_inst_62_, v_inst_64_);
v___x_68_ = lean_apply_1(v___x_30__overap_67_, v_inst_65_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_gMulHom___redArg___lam__0(lean_object* v_inst_69_, lean_object* v_i_70_, lean_object* v_j_71_, lean_object* v_a_72_, lean_object* v___y_73_){
_start:
{
lean_object* v___x_74_; 
v___x_74_ = lean_apply_4(v_inst_69_, v_i_70_, v_j_71_, v_a_72_, v___y_73_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_gMulHom___redArg(lean_object* v_inst_75_, lean_object* v_i_76_, lean_object* v_j_77_){
_start:
{
lean_object* v___f_78_; 
v___f_78_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_gMulHom___redArg___lam__0), 5, 3);
lean_closure_set(v___f_78_, 0, v_inst_75_);
lean_closure_set(v___f_78_, 1, v_i_76_);
lean_closure_set(v___f_78_, 2, v_j_77_);
return v___f_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_gMulHom(lean_object* v_00_u03b9_79_, lean_object* v_A_80_, lean_object* v_inst_81_, lean_object* v_inst_82_, lean_object* v_inst_83_, lean_object* v_i_84_, lean_object* v_j_85_){
_start:
{
lean_object* v___f_86_; 
v___f_86_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_gMulHom___redArg___lam__0), 5, 3);
lean_closure_set(v___f_86_, 0, v_inst_83_);
lean_closure_set(v___f_86_, 1, v_i_84_);
lean_closure_set(v___f_86_, 2, v_j_85_);
return v___f_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_gMulHom___boxed(lean_object* v_00_u03b9_87_, lean_object* v_A_88_, lean_object* v_inst_89_, lean_object* v_inst_90_, lean_object* v_inst_91_, lean_object* v_i_92_, lean_object* v_j_93_){
_start:
{
lean_object* v_res_94_; 
v_res_94_ = lp_mathlib_DirectSum_gMulHom(v_00_u03b9_87_, v_A_88_, v_inst_89_, v_inst_90_, v_inst_91_, v_i_92_, v_j_93_);
lean_dec_ref(v_inst_90_);
lean_dec(v_inst_89_);
return v_res_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_mulHom___redArg___lam__0(lean_object* v_inst_95_, lean_object* v_x_96_, lean_object* v_inst_97_, lean_object* v_inst_98_, lean_object* v_inst_99_, lean_object* v_x_100_, lean_object* v___y_101_, lean_object* v___y_102_){
_start:
{
lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___f_106_; lean_object* v___f_107_; lean_object* v___x_108_; 
lean_inc(v_x_100_);
lean_inc(v_x_96_);
v___x_103_ = lean_apply_2(v_inst_95_, v_x_96_, v_x_100_);
v___x_104_ = lp_mathlib_DirectSum_of___redArg(v_inst_97_, v_inst_98_, v___x_103_);
v___x_105_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_compHom___lam__0), 3, 1);
lean_closure_set(v___x_105_, 0, v___x_104_);
v___f_106_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_gMulHom___redArg___lam__0), 5, 3);
lean_closure_set(v___f_106_, 0, v_inst_99_);
lean_closure_set(v___f_106_, 1, v_x_96_);
lean_closure_set(v___f_106_, 2, v_x_100_);
v___f_107_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_107_, 0, v___f_106_);
lean_closure_set(v___f_107_, 1, v___x_105_);
v___x_108_ = lp_mathlib_MonoidHom_flip___redArg___lam__0(v___f_107_, v___y_101_, v___y_102_);
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_mulHom___redArg___lam__1(lean_object* v_inst_109_, lean_object* v_inst_110_, lean_object* v_inst_111_, lean_object* v_inst_112_, lean_object* v___x_113_, lean_object* v_x_114_, lean_object* v___y_115_, lean_object* v___y_116_){
_start:
{
lean_object* v___f_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; 
lean_inc_ref(v_inst_111_);
lean_inc_ref(v_inst_110_);
v___f_117_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_mulHom___redArg___lam__0), 8, 5);
lean_closure_set(v___f_117_, 0, v_inst_109_);
lean_closure_set(v___f_117_, 1, v_x_114_);
lean_closure_set(v___f_117_, 2, v_inst_110_);
lean_closure_set(v___f_117_, 3, v_inst_111_);
lean_closure_set(v___f_117_, 4, v_inst_112_);
v___x_118_ = lp_mathlib_AddMonoidHom_instAddCommMonoid___redArg(v___x_113_);
v___x_119_ = lp_mathlib_DirectSum_toAddMonoid___redArg(v_inst_110_, v_inst_111_, v___x_118_, v___f_117_);
v___x_120_ = lp_mathlib_MonoidHom_flip___redArg___lam__0(v___x_119_, v___y_115_, v___y_116_);
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_mulHom___redArg(lean_object* v_inst_121_, lean_object* v_inst_122_, lean_object* v_inst_123_, lean_object* v_inst_124_){
_start:
{
lean_object* v___x_125_; lean_object* v___f_126_; lean_object* v___x_127_; lean_object* v___x_128_; 
lean_inc_ref_n(v_inst_123_, 2);
v___x_125_ = lp_mathlib_DirectSum_instAddCommMonoid___redArg(v_inst_123_);
lean_inc_ref(v___x_125_);
lean_inc_ref(v_inst_121_);
v___f_126_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_mulHom___redArg___lam__1), 8, 5);
lean_closure_set(v___f_126_, 0, v_inst_122_);
lean_closure_set(v___f_126_, 1, v_inst_123_);
lean_closure_set(v___f_126_, 2, v_inst_121_);
lean_closure_set(v___f_126_, 3, v_inst_124_);
lean_closure_set(v___f_126_, 4, v___x_125_);
v___x_127_ = lp_mathlib_AddMonoidHom_instAddCommMonoid___redArg(v___x_125_);
v___x_128_ = lp_mathlib_DirectSum_toAddMonoid___redArg(v_inst_123_, v_inst_121_, v___x_127_, v___f_126_);
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_mulHom(lean_object* v_00_u03b9_129_, lean_object* v_inst_130_, lean_object* v_A_131_, lean_object* v_inst_132_, lean_object* v_inst_133_, lean_object* v_inst_134_){
_start:
{
lean_object* v___x_135_; 
v___x_135_ = lp_mathlib_DirectSum_mulHom___redArg(v_inst_130_, v_inst_132_, v_inst_133_, v_inst_134_);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instMul___redArg___lam__0(lean_object* v_inst_136_, lean_object* v_inst_137_, lean_object* v_inst_138_, lean_object* v_inst_139_, lean_object* v_a_140_, lean_object* v_b_141_){
_start:
{
lean_object* v___x_40__overap_142_; lean_object* v___x_143_; 
v___x_40__overap_142_ = lp_mathlib_DirectSum_mulHom___redArg(v_inst_136_, v_inst_137_, v_inst_138_, v_inst_139_);
v___x_143_ = lean_apply_2(v___x_40__overap_142_, v_a_140_, v_b_141_);
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instMul___redArg(lean_object* v_inst_144_, lean_object* v_inst_145_, lean_object* v_inst_146_, lean_object* v_inst_147_){
_start:
{
lean_object* v___f_148_; 
v___f_148_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instMul___redArg___lam__0), 6, 4);
lean_closure_set(v___f_148_, 0, v_inst_144_);
lean_closure_set(v___f_148_, 1, v_inst_145_);
lean_closure_set(v___f_148_, 2, v_inst_146_);
lean_closure_set(v___f_148_, 3, v_inst_147_);
return v___f_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instMul(lean_object* v_00_u03b9_149_, lean_object* v_inst_150_, lean_object* v_A_151_, lean_object* v_inst_152_, lean_object* v_inst_153_, lean_object* v_inst_154_){
_start:
{
lean_object* v___f_155_; 
v___f_155_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instMul___redArg___lam__0), 6, 4);
lean_closure_set(v___f_155_, 0, v_inst_150_);
lean_closure_set(v___f_155_, 1, v_inst_152_);
lean_closure_set(v___f_155_, 2, v_inst_153_);
lean_closure_set(v___f_155_, 3, v_inst_154_);
return v___f_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNonUnitalNonAssocSemiring___redArg(lean_object* v_inst_156_, lean_object* v_inst_157_, lean_object* v_inst_158_, lean_object* v_inst_159_){
_start:
{
lean_object* v___x_160_; lean_object* v___f_161_; lean_object* v___x_162_; 
lean_inc_ref(v_inst_158_);
v___x_160_ = lp_mathlib_DirectSum_instAddCommMonoid___redArg(v_inst_158_);
v___f_161_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instMul___redArg___lam__0), 6, 4);
lean_closure_set(v___f_161_, 0, v_inst_156_);
lean_closure_set(v___f_161_, 1, v_inst_157_);
lean_closure_set(v___f_161_, 2, v_inst_158_);
lean_closure_set(v___f_161_, 3, v_inst_159_);
v___x_162_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_162_, 0, v___x_160_);
lean_ctor_set(v___x_162_, 1, v___f_161_);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNonUnitalNonAssocSemiring(lean_object* v_00_u03b9_163_, lean_object* v_inst_164_, lean_object* v_A_165_, lean_object* v_inst_166_, lean_object* v_inst_167_, lean_object* v_inst_168_){
_start:
{
lean_object* v___x_169_; 
v___x_169_ = lp_mathlib_DirectSum_instNonUnitalNonAssocSemiring___redArg(v_inst_164_, v_inst_166_, v_inst_167_, v_inst_168_);
return v___x_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCast___redArg___lam__0(lean_object* v_inst_170_, lean_object* v_inst_171_, lean_object* v_inst_172_, lean_object* v_toZero_173_, lean_object* v_n_174_){
_start:
{
lean_object* v_natCast_175_; lean_object* v___x_176_; lean_object* v___x_63__overap_177_; lean_object* v___x_178_; 
v_natCast_175_ = lean_ctor_get(v_inst_170_, 3);
lean_inc(v_natCast_175_);
lean_dec_ref(v_inst_170_);
v___x_176_ = lean_apply_1(v_natCast_175_, v_n_174_);
v___x_63__overap_177_ = lp_mathlib_DirectSum_of___redArg(v_inst_171_, v_inst_172_, v_toZero_173_);
v___x_178_ = lean_apply_1(v___x_63__overap_177_, v___x_176_);
return v___x_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCast___redArg(lean_object* v_inst_179_, lean_object* v_inst_180_, lean_object* v_inst_181_, lean_object* v_inst_182_){
_start:
{
lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v_toZero_185_; lean_object* v___f_186_; 
v___x_183_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_181_);
v___x_184_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_183_);
v_toZero_185_ = lean_ctor_get(v___x_184_, 0);
lean_inc(v_toZero_185_);
lean_dec_ref(v___x_184_);
v___f_186_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instNatCast___redArg___lam__0), 5, 4);
lean_closure_set(v___f_186_, 0, v_inst_182_);
lean_closure_set(v___f_186_, 1, v_inst_180_);
lean_closure_set(v___f_186_, 2, v_inst_179_);
lean_closure_set(v___f_186_, 3, v_toZero_185_);
return v___f_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCast___redArg___boxed(lean_object* v_inst_187_, lean_object* v_inst_188_, lean_object* v_inst_189_, lean_object* v_inst_190_){
_start:
{
lean_object* v_res_191_; 
v_res_191_ = lp_mathlib_DirectSum_instNatCast___redArg(v_inst_187_, v_inst_188_, v_inst_189_, v_inst_190_);
lean_dec_ref(v_inst_189_);
return v_res_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCast(lean_object* v_00_u03b9_192_, lean_object* v_inst_193_, lean_object* v_A_194_, lean_object* v_inst_195_, lean_object* v_inst_196_, lean_object* v_inst_197_){
_start:
{
lean_object* v___x_198_; 
v___x_198_ = lp_mathlib_DirectSum_instNatCast___redArg(v_inst_193_, v_inst_195_, v_inst_196_, v_inst_197_);
return v___x_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCast___boxed(lean_object* v_00_u03b9_199_, lean_object* v_inst_200_, lean_object* v_A_201_, lean_object* v_inst_202_, lean_object* v_inst_203_, lean_object* v_inst_204_){
_start:
{
lean_object* v_res_205_; 
v_res_205_ = lp_mathlib_DirectSum_instNatCast(v_00_u03b9_199_, v_inst_200_, v_A_201_, v_inst_202_, v_inst_203_, v_inst_204_);
lean_dec_ref(v_inst_203_);
return v_res_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_semiring___redArg(lean_object* v_inst_206_, lean_object* v_inst_207_, lean_object* v_inst_208_, lean_object* v_inst_209_){
_start:
{
lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v_toZero_213_; lean_object* v___x_214_; lean_object* v_toGOne_215_; lean_object* v___x_217_; uint8_t v_isShared_218_; uint8_t v_isSharedCheck_239_; 
lean_inc_ref(v_inst_207_);
v___x_210_ = lp_mathlib_DirectSum_instAddCommMonoid___redArg(v_inst_207_);
v___x_211_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_208_);
v___x_212_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_211_);
v_toZero_213_ = lean_ctor_get(v___x_212_, 0);
lean_inc(v_toZero_213_);
lean_dec_ref(v___x_212_);
v___x_214_ = lp_mathlib_DirectSum_GSemiring_toGMonoid___redArg(v_inst_209_);
v_toGOne_215_ = lean_ctor_get(v___x_214_, 1);
v_isSharedCheck_239_ = !lean_is_exclusive(v___x_214_);
if (v_isSharedCheck_239_ == 0)
{
lean_object* v_unused_240_; lean_object* v_unused_241_; 
v_unused_240_ = lean_ctor_get(v___x_214_, 2);
lean_dec(v_unused_240_);
v_unused_241_ = lean_ctor_get(v___x_214_, 0);
lean_dec(v_unused_241_);
v___x_217_ = v___x_214_;
v_isShared_218_ = v_isSharedCheck_239_;
goto v_resetjp_216_;
}
else
{
lean_inc(v_toGOne_215_);
lean_dec(v___x_214_);
v___x_217_ = lean_box(0);
v_isShared_218_ = v_isSharedCheck_239_;
goto v_resetjp_216_;
}
v_resetjp_216_:
{
lean_object* v_toAdd_219_; lean_object* v_toGNonUnitalNonAssocSemiring_220_; lean_object* v___x_221_; lean_object* v___x_223_; uint8_t v_isShared_224_; uint8_t v_isSharedCheck_235_; 
v_toAdd_219_ = lean_ctor_get(v_inst_208_, 1);
lean_inc(v_toAdd_219_);
v_toGNonUnitalNonAssocSemiring_220_ = lean_ctor_get(v_inst_209_, 0);
lean_inc(v_toGNonUnitalNonAssocSemiring_220_);
lean_inc_ref(v_inst_207_);
lean_inc_ref(v_inst_206_);
v___x_221_ = lp_mathlib_DirectSum_instNatCast___redArg(v_inst_206_, v_inst_207_, v_inst_208_, v_inst_209_);
v_isSharedCheck_235_ = !lean_is_exclusive(v_inst_208_);
if (v_isSharedCheck_235_ == 0)
{
lean_object* v_unused_236_; lean_object* v_unused_237_; lean_object* v_unused_238_; 
v_unused_236_ = lean_ctor_get(v_inst_208_, 2);
lean_dec(v_unused_236_);
v_unused_237_ = lean_ctor_get(v_inst_208_, 1);
lean_dec(v_unused_237_);
v_unused_238_ = lean_ctor_get(v_inst_208_, 0);
lean_dec(v_unused_238_);
v___x_223_ = v_inst_208_;
v_isShared_224_ = v_isSharedCheck_235_;
goto v_resetjp_222_;
}
else
{
lean_dec(v_inst_208_);
v___x_223_ = lean_box(0);
v_isShared_224_ = v_isSharedCheck_235_;
goto v_resetjp_222_;
}
v_resetjp_222_:
{
lean_object* v___x_26__overap_225_; lean_object* v___x_226_; lean_object* v___f_227_; lean_object* v___x_228_; lean_object* v___x_230_; 
lean_inc_ref(v_inst_206_);
lean_inc_ref(v_inst_207_);
v___x_26__overap_225_ = lp_mathlib_DirectSum_of___redArg(v_inst_207_, v_inst_206_, v_toZero_213_);
v___x_226_ = lean_apply_1(v___x_26__overap_225_, v_toGOne_215_);
v___f_227_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instMul___redArg___lam__0), 6, 4);
lean_closure_set(v___f_227_, 0, v_inst_206_);
lean_closure_set(v___f_227_, 1, v_toAdd_219_);
lean_closure_set(v___f_227_, 2, v_inst_207_);
lean_closure_set(v___f_227_, 3, v_toGNonUnitalNonAssocSemiring_220_);
lean_inc_ref(v___x_226_);
lean_inc_ref(v___f_227_);
v___x_228_ = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(v___x_228_, 0, lean_box(0));
lean_closure_set(v___x_228_, 1, v___f_227_);
lean_closure_set(v___x_228_, 2, v___x_226_);
if (v_isShared_224_ == 0)
{
lean_ctor_set(v___x_223_, 2, v___x_228_);
lean_ctor_set(v___x_223_, 1, v___f_227_);
lean_ctor_set(v___x_223_, 0, v___x_226_);
v___x_230_ = v___x_223_;
goto v_reusejp_229_;
}
else
{
lean_object* v_reuseFailAlloc_234_; 
v_reuseFailAlloc_234_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_234_, 0, v___x_226_);
lean_ctor_set(v_reuseFailAlloc_234_, 1, v___f_227_);
lean_ctor_set(v_reuseFailAlloc_234_, 2, v___x_228_);
v___x_230_ = v_reuseFailAlloc_234_;
goto v_reusejp_229_;
}
v_reusejp_229_:
{
lean_object* v___x_232_; 
if (v_isShared_218_ == 0)
{
lean_ctor_set(v___x_217_, 2, v___x_221_);
lean_ctor_set(v___x_217_, 1, v___x_230_);
lean_ctor_set(v___x_217_, 0, v___x_210_);
v___x_232_ = v___x_217_;
goto v_reusejp_231_;
}
else
{
lean_object* v_reuseFailAlloc_233_; 
v_reuseFailAlloc_233_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_233_, 0, v___x_210_);
lean_ctor_set(v_reuseFailAlloc_233_, 1, v___x_230_);
lean_ctor_set(v_reuseFailAlloc_233_, 2, v___x_221_);
v___x_232_ = v_reuseFailAlloc_233_;
goto v_reusejp_231_;
}
v_reusejp_231_:
{
return v___x_232_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_semiring(lean_object* v_00_u03b9_242_, lean_object* v_inst_243_, lean_object* v_A_244_, lean_object* v_inst_245_, lean_object* v_inst_246_, lean_object* v_inst_247_){
_start:
{
lean_object* v___x_248_; 
v___x_248_ = lp_mathlib_DirectSum_semiring___redArg(v_inst_243_, v_inst_245_, v_inst_246_, v_inst_247_);
return v___x_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_commSemiring___redArg(lean_object* v_inst_249_, lean_object* v_inst_250_, lean_object* v_inst_251_, lean_object* v_inst_252_){
_start:
{
lean_object* v___x_253_; 
v___x_253_ = lp_mathlib_DirectSum_semiring___redArg(v_inst_249_, v_inst_250_, v_inst_251_, v_inst_252_);
return v___x_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_commSemiring(lean_object* v_00_u03b9_254_, lean_object* v_inst_255_, lean_object* v_A_256_, lean_object* v_inst_257_, lean_object* v_inst_258_, lean_object* v_inst_259_){
_start:
{
lean_object* v___x_260_; 
v___x_260_ = lp_mathlib_DirectSum_semiring___redArg(v_inst_255_, v_inst_257_, v_inst_258_, v_inst_259_);
return v___x_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_nonAssocRing___redArg___lam__0(lean_object* v_inst_261_, lean_object* v_i_262_){
_start:
{
lean_object* v___x_263_; lean_object* v_toAddMonoid_264_; 
v___x_263_ = lean_apply_1(v_inst_261_, v_i_262_);
v_toAddMonoid_264_ = lean_ctor_get(v___x_263_, 0);
lean_inc_ref(v_toAddMonoid_264_);
lean_dec_ref(v___x_263_);
return v_toAddMonoid_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_nonAssocRing___redArg(lean_object* v_inst_265_, lean_object* v_inst_266_, lean_object* v_inst_267_, lean_object* v_inst_268_){
_start:
{
lean_object* v___f_269_; lean_object* v___x_270_; lean_object* v___f_271_; lean_object* v___x_272_; 
lean_inc_ref(v_inst_266_);
v___f_269_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_nonAssocRing___redArg___lam__0), 2, 1);
lean_closure_set(v___f_269_, 0, v_inst_266_);
v___x_270_ = lp_mathlib_DirectSum_instAddCommGroup___redArg(v_inst_266_);
v___f_271_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instMul___redArg___lam__0), 6, 4);
lean_closure_set(v___f_271_, 0, v_inst_265_);
lean_closure_set(v___f_271_, 1, v_inst_267_);
lean_closure_set(v___f_271_, 2, v___f_269_);
lean_closure_set(v___f_271_, 3, v_inst_268_);
v___x_272_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_272_, 0, v___x_270_);
lean_ctor_set(v___x_272_, 1, v___f_271_);
return v___x_272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_nonAssocRing(lean_object* v_00_u03b9_273_, lean_object* v_inst_274_, lean_object* v_A_275_, lean_object* v_inst_276_, lean_object* v_inst_277_, lean_object* v_inst_278_){
_start:
{
lean_object* v___x_279_; 
v___x_279_ = lp_mathlib_DirectSum_nonAssocRing___redArg(v_inst_274_, v_inst_276_, v_inst_277_, v_inst_278_);
return v___x_279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_ring___redArg___lam__1(lean_object* v_intCast_280_, lean_object* v___f_281_, lean_object* v_inst_282_, lean_object* v_toZero_283_, lean_object* v_z_284_){
_start:
{
lean_object* v___x_285_; lean_object* v___x_76__overap_286_; lean_object* v___x_287_; 
v___x_285_ = lean_apply_1(v_intCast_280_, v_z_284_);
v___x_76__overap_286_ = lp_mathlib_DirectSum_of___redArg(v___f_281_, v_inst_282_, v_toZero_283_);
v___x_287_ = lean_apply_1(v___x_76__overap_286_, v___x_285_);
return v___x_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_ring___redArg(lean_object* v_inst_288_, lean_object* v_inst_289_, lean_object* v_inst_290_, lean_object* v_inst_291_){
_start:
{
lean_object* v_toGSemiring_292_; lean_object* v_intCast_293_; lean_object* v___x_294_; lean_object* v_toNeg_295_; lean_object* v_toSub_296_; lean_object* v_toZSMul_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v_toZero_300_; lean_object* v___f_301_; lean_object* v___f_302_; lean_object* v___x_303_; lean_object* v___x_304_; 
v_toGSemiring_292_ = lean_ctor_get(v_inst_291_, 0);
lean_inc_ref(v_toGSemiring_292_);
v_intCast_293_ = lean_ctor_get(v_inst_291_, 1);
lean_inc(v_intCast_293_);
lean_dec_ref(v_inst_291_);
lean_inc_ref(v_inst_289_);
v___x_294_ = lp_mathlib_DirectSum_instAddCommGroup___redArg(v_inst_289_);
v_toNeg_295_ = lean_ctor_get(v___x_294_, 1);
lean_inc(v_toNeg_295_);
v_toSub_296_ = lean_ctor_get(v___x_294_, 2);
lean_inc(v_toSub_296_);
v_toZSMul_297_ = lean_ctor_get(v___x_294_, 3);
lean_inc(v_toZSMul_297_);
lean_dec_ref(v___x_294_);
v___x_298_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_290_);
v___x_299_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_298_);
v_toZero_300_ = lean_ctor_get(v___x_299_, 0);
lean_inc(v_toZero_300_);
lean_dec_ref(v___x_299_);
v___f_301_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_nonAssocRing___redArg___lam__0), 2, 1);
lean_closure_set(v___f_301_, 0, v_inst_289_);
lean_inc_ref(v_inst_288_);
lean_inc_ref(v___f_301_);
v___f_302_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_ring___redArg___lam__1), 5, 4);
lean_closure_set(v___f_302_, 0, v_intCast_293_);
lean_closure_set(v___f_302_, 1, v___f_301_);
lean_closure_set(v___f_302_, 2, v_inst_288_);
lean_closure_set(v___f_302_, 3, v_toZero_300_);
v___x_303_ = lp_mathlib_DirectSum_semiring___redArg(v_inst_288_, v___f_301_, v_inst_290_, v_toGSemiring_292_);
v___x_304_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_304_, 0, v___x_303_);
lean_ctor_set(v___x_304_, 1, v_toNeg_295_);
lean_ctor_set(v___x_304_, 2, v_toSub_296_);
lean_ctor_set(v___x_304_, 3, v_toZSMul_297_);
lean_ctor_set(v___x_304_, 4, v___f_302_);
return v___x_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_ring(lean_object* v_00_u03b9_305_, lean_object* v_inst_306_, lean_object* v_A_307_, lean_object* v_inst_308_, lean_object* v_inst_309_, lean_object* v_inst_310_){
_start:
{
lean_object* v___x_311_; 
v___x_311_ = lp_mathlib_DirectSum_ring___redArg(v_inst_306_, v_inst_308_, v_inst_309_, v_inst_310_);
return v___x_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_commRing___redArg(lean_object* v_inst_312_, lean_object* v_inst_313_, lean_object* v_inst_314_, lean_object* v_inst_315_){
_start:
{
lean_object* v___x_316_; 
v___x_316_ = lp_mathlib_DirectSum_ring___redArg(v_inst_312_, v_inst_313_, v_inst_314_, v_inst_315_);
return v___x_316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_commRing(lean_object* v_00_u03b9_317_, lean_object* v_inst_318_, lean_object* v_A_319_, lean_object* v_inst_320_, lean_object* v_inst_321_, lean_object* v_inst_322_){
_start:
{
lean_object* v___x_323_; 
v___x_323_ = lp_mathlib_DirectSum_ring___redArg(v_inst_318_, v_inst_320_, v_inst_321_, v_inst_322_);
return v___x_323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNonUnitalNonAssocSemiringOfNat___redArg(lean_object* v_inst_324_, lean_object* v_inst_325_, lean_object* v_inst_326_){
_start:
{
lean_object* v___x_327_; lean_object* v_toZero_328_; lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v_toZero_332_; lean_object* v_toAdd_333_; lean_object* v___x_335_; uint8_t v_isShared_336_; uint8_t v_isSharedCheck_344_; 
lean_inc_ref(v_inst_324_);
v___x_327_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_324_);
v_toZero_328_ = lean_ctor_get(v___x_327_, 0);
lean_inc(v_toZero_328_);
lean_dec_ref(v___x_327_);
v___x_329_ = lean_apply_1(v_inst_325_, v_toZero_328_);
v___x_330_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v___x_329_);
v___x_331_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_330_);
v_toZero_332_ = lean_ctor_get(v___x_331_, 0);
v_toAdd_333_ = lean_ctor_get(v___x_331_, 1);
v_isSharedCheck_344_ = !lean_is_exclusive(v___x_331_);
if (v_isSharedCheck_344_ == 0)
{
v___x_335_ = v___x_331_;
v_isShared_336_ = v_isSharedCheck_344_;
goto v_resetjp_334_;
}
else
{
lean_inc(v_toAdd_333_);
lean_inc(v_toZero_332_);
lean_dec(v___x_331_);
v___x_335_ = lean_box(0);
v_isShared_336_ = v_isSharedCheck_344_;
goto v_resetjp_334_;
}
v_resetjp_334_:
{
lean_object* v_toNSMul_337_; lean_object* v___x_338_; lean_object* v___f_339_; lean_object* v___x_340_; lean_object* v___x_342_; 
v_toNSMul_337_ = lean_ctor_get(v___x_329_, 2);
lean_inc(v_toNSMul_337_);
lean_dec_ref(v___x_329_);
v___x_338_ = lp_mathlib_GradedMonoid_GradeZero_mul___redArg(v_inst_324_, v_inst_326_);
v___f_339_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_339_, 0, v_toNSMul_337_);
v___x_340_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_toAdd_333_, v_toZero_332_, v___f_339_);
if (v_isShared_336_ == 0)
{
lean_ctor_set(v___x_335_, 1, v___x_338_);
lean_ctor_set(v___x_335_, 0, v___x_340_);
v___x_342_ = v___x_335_;
goto v_reusejp_341_;
}
else
{
lean_object* v_reuseFailAlloc_343_; 
v_reuseFailAlloc_343_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_343_, 0, v___x_340_);
lean_ctor_set(v_reuseFailAlloc_343_, 1, v___x_338_);
v___x_342_ = v_reuseFailAlloc_343_;
goto v_reusejp_341_;
}
v_reusejp_341_:
{
return v___x_342_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNonUnitalNonAssocSemiringOfNat(lean_object* v_00_u03b9_345_, lean_object* v_inst_346_, lean_object* v_A_347_, lean_object* v_inst_348_, lean_object* v_inst_349_, lean_object* v_inst_350_){
_start:
{
lean_object* v___x_351_; 
v___x_351_ = lp_mathlib_DirectSum_instNonUnitalNonAssocSemiringOfNat___redArg(v_inst_348_, v_inst_349_, v_inst_350_);
return v___x_351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNonUnitalNonAssocSemiringOfNat___boxed(lean_object* v_00_u03b9_352_, lean_object* v_inst_353_, lean_object* v_A_354_, lean_object* v_inst_355_, lean_object* v_inst_356_, lean_object* v_inst_357_){
_start:
{
lean_object* v_res_358_; 
v_res_358_ = lp_mathlib_DirectSum_instNonUnitalNonAssocSemiringOfNat(v_00_u03b9_352_, v_inst_353_, v_A_354_, v_inst_355_, v_inst_356_, v_inst_357_);
lean_dec_ref(v_inst_353_);
return v_res_358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instSMulWithZeroOfNat___redArg(lean_object* v_inst_359_, lean_object* v_inst_360_, lean_object* v_i_361_){
_start:
{
lean_object* v___x_362_; 
v___x_362_ = lp_mathlib_GradedMonoid_GradeZero_smul___redArg(v_inst_359_, v_inst_360_, v_i_361_);
return v___x_362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instSMulWithZeroOfNat(lean_object* v_00_u03b9_363_, lean_object* v_inst_364_, lean_object* v_A_365_, lean_object* v_inst_366_, lean_object* v_inst_367_, lean_object* v_inst_368_, lean_object* v_i_369_){
_start:
{
lean_object* v___x_370_; 
v___x_370_ = lp_mathlib_GradedMonoid_GradeZero_smul___redArg(v_inst_366_, v_inst_368_, v_i_369_);
return v___x_370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instSMulWithZeroOfNat___boxed(lean_object* v_00_u03b9_371_, lean_object* v_inst_372_, lean_object* v_A_373_, lean_object* v_inst_374_, lean_object* v_inst_375_, lean_object* v_inst_376_, lean_object* v_i_377_){
_start:
{
lean_object* v_res_378_; 
v_res_378_ = lp_mathlib_DirectSum_instSMulWithZeroOfNat(v_00_u03b9_371_, v_inst_372_, v_A_373_, v_inst_374_, v_inst_375_, v_inst_376_, v_i_377_);
lean_dec_ref(v_inst_375_);
lean_dec_ref(v_inst_372_);
return v_res_378_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCastOfNat___redArg(lean_object* v_inst_379_){
_start:
{
lean_object* v_natCast_380_; 
v_natCast_380_ = lean_ctor_get(v_inst_379_, 3);
lean_inc(v_natCast_380_);
return v_natCast_380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCastOfNat___redArg___boxed(lean_object* v_inst_381_){
_start:
{
lean_object* v_res_382_; 
v_res_382_ = lp_mathlib_DirectSum_instNatCastOfNat___redArg(v_inst_381_);
lean_dec_ref(v_inst_381_);
return v_res_382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCastOfNat(lean_object* v_00_u03b9_383_, lean_object* v_A_384_, lean_object* v_inst_385_, lean_object* v_inst_386_, lean_object* v_inst_387_){
_start:
{
lean_object* v_natCast_388_; 
v_natCast_388_ = lean_ctor_get(v_inst_387_, 3);
lean_inc(v_natCast_388_);
return v_natCast_388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNatCastOfNat___boxed(lean_object* v_00_u03b9_389_, lean_object* v_A_390_, lean_object* v_inst_391_, lean_object* v_inst_392_, lean_object* v_inst_393_){
_start:
{
lean_object* v_res_394_; 
v_res_394_ = lp_mathlib_DirectSum_instNatCastOfNat(v_00_u03b9_389_, v_A_390_, v_inst_391_, v_inst_392_, v_inst_393_);
lean_dec_ref(v_inst_393_);
lean_dec_ref(v_inst_392_);
lean_dec_ref(v_inst_391_);
return v_res_394_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instSemiringOfNat___redArg___lam__0(lean_object* v_inst_395_, lean_object* v_gnpow_396_, lean_object* v_n_397_, lean_object* v_x_398_){
_start:
{
lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v_toZero_401_; lean_object* v___x_402_; 
v___x_399_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_395_);
v___x_400_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_399_);
v_toZero_401_ = lean_ctor_get(v___x_400_, 0);
lean_inc(v_toZero_401_);
lean_dec_ref(v___x_400_);
v___x_402_ = lean_apply_3(v_gnpow_396_, v_n_397_, v_toZero_401_, v_x_398_);
return v___x_402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instSemiringOfNat___redArg___lam__0___boxed(lean_object* v_inst_403_, lean_object* v_gnpow_404_, lean_object* v_n_405_, lean_object* v_x_406_){
_start:
{
lean_object* v_res_407_; 
v_res_407_ = lp_mathlib_DirectSum_instSemiringOfNat___redArg___lam__0(v_inst_403_, v_gnpow_404_, v_n_405_, v_x_406_);
lean_dec_ref(v_inst_403_);
return v_res_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instSemiringOfNat___redArg(lean_object* v_inst_408_, lean_object* v_inst_409_, lean_object* v_inst_410_){
_start:
{
lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v_toZero_413_; lean_object* v_toGNonUnitalNonAssocSemiring_414_; lean_object* v_gnpow_415_; lean_object* v_natCast_416_; lean_object* v___x_417_; lean_object* v_toGOne_418_; lean_object* v___x_420_; uint8_t v_isShared_421_; uint8_t v_isSharedCheck_445_; 
v___x_411_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_409_);
lean_inc_ref(v___x_411_);
v___x_412_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_411_);
v_toZero_413_ = lean_ctor_get(v___x_412_, 0);
lean_inc(v_toZero_413_);
lean_dec_ref(v___x_412_);
v_toGNonUnitalNonAssocSemiring_414_ = lean_ctor_get(v_inst_410_, 0);
lean_inc(v_toGNonUnitalNonAssocSemiring_414_);
v_gnpow_415_ = lean_ctor_get(v_inst_410_, 2);
lean_inc(v_gnpow_415_);
v_natCast_416_ = lean_ctor_get(v_inst_410_, 3);
lean_inc(v_natCast_416_);
v___x_417_ = lp_mathlib_DirectSum_GSemiring_toGMonoid___redArg(v_inst_410_);
lean_dec_ref(v_inst_410_);
v_toGOne_418_ = lean_ctor_get(v___x_417_, 1);
v_isSharedCheck_445_ = !lean_is_exclusive(v___x_417_);
if (v_isSharedCheck_445_ == 0)
{
lean_object* v_unused_446_; lean_object* v_unused_447_; 
v_unused_446_ = lean_ctor_get(v___x_417_, 2);
lean_dec(v_unused_446_);
v_unused_447_ = lean_ctor_get(v___x_417_, 0);
lean_dec(v_unused_447_);
v___x_420_ = v___x_417_;
v_isShared_421_ = v_isSharedCheck_445_;
goto v_resetjp_419_;
}
else
{
lean_inc(v_toGOne_418_);
lean_dec(v___x_417_);
v___x_420_ = lean_box(0);
v_isShared_421_ = v_isSharedCheck_445_;
goto v_resetjp_419_;
}
v_resetjp_419_:
{
lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v_toZero_425_; lean_object* v_toAdd_426_; lean_object* v_toNSMul_427_; lean_object* v___x_429_; uint8_t v_isShared_430_; uint8_t v_isSharedCheck_442_; 
v___x_422_ = lean_apply_1(v_inst_408_, v_toZero_413_);
v___x_423_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v___x_422_);
v___x_424_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_423_);
v_toZero_425_ = lean_ctor_get(v___x_424_, 0);
lean_inc(v_toZero_425_);
v_toAdd_426_ = lean_ctor_get(v___x_424_, 1);
lean_inc(v_toAdd_426_);
lean_dec_ref(v___x_424_);
v_toNSMul_427_ = lean_ctor_get(v___x_422_, 2);
v_isSharedCheck_442_ = !lean_is_exclusive(v___x_422_);
if (v_isSharedCheck_442_ == 0)
{
lean_object* v_unused_443_; lean_object* v_unused_444_; 
v_unused_443_ = lean_ctor_get(v___x_422_, 1);
lean_dec(v_unused_443_);
v_unused_444_ = lean_ctor_get(v___x_422_, 0);
lean_dec(v_unused_444_);
v___x_429_ = v___x_422_;
v_isShared_430_ = v_isSharedCheck_442_;
goto v_resetjp_428_;
}
else
{
lean_inc(v_toNSMul_427_);
lean_dec(v___x_422_);
v___x_429_ = lean_box(0);
v_isShared_430_ = v_isSharedCheck_442_;
goto v_resetjp_428_;
}
v_resetjp_428_:
{
lean_object* v___f_431_; lean_object* v___x_432_; lean_object* v___f_433_; lean_object* v___x_434_; lean_object* v___x_435_; lean_object* v___x_437_; 
v___f_431_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instSemiringOfNat___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_431_, 0, v_inst_409_);
lean_closure_set(v___f_431_, 1, v_gnpow_415_);
v___x_432_ = lp_mathlib_GradedMonoid_GradeZero_mul___redArg(v___x_411_, v_toGNonUnitalNonAssocSemiring_414_);
v___f_433_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_433_, 0, v_toNSMul_427_);
v___x_434_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_434_, 0, lean_box(0));
lean_closure_set(v___x_434_, 1, v_natCast_416_);
v___x_435_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_toAdd_426_, v_toZero_425_, v___f_433_);
if (v_isShared_430_ == 0)
{
lean_ctor_set(v___x_429_, 2, v___f_431_);
lean_ctor_set(v___x_429_, 1, v___x_432_);
lean_ctor_set(v___x_429_, 0, v_toGOne_418_);
v___x_437_ = v___x_429_;
goto v_reusejp_436_;
}
else
{
lean_object* v_reuseFailAlloc_441_; 
v_reuseFailAlloc_441_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_441_, 0, v_toGOne_418_);
lean_ctor_set(v_reuseFailAlloc_441_, 1, v___x_432_);
lean_ctor_set(v_reuseFailAlloc_441_, 2, v___f_431_);
v___x_437_ = v_reuseFailAlloc_441_;
goto v_reusejp_436_;
}
v_reusejp_436_:
{
lean_object* v___x_439_; 
if (v_isShared_421_ == 0)
{
lean_ctor_set(v___x_420_, 2, v___x_434_);
lean_ctor_set(v___x_420_, 1, v___x_437_);
lean_ctor_set(v___x_420_, 0, v___x_435_);
v___x_439_ = v___x_420_;
goto v_reusejp_438_;
}
else
{
lean_object* v_reuseFailAlloc_440_; 
v_reuseFailAlloc_440_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_440_, 0, v___x_435_);
lean_ctor_set(v_reuseFailAlloc_440_, 1, v___x_437_);
lean_ctor_set(v_reuseFailAlloc_440_, 2, v___x_434_);
v___x_439_ = v_reuseFailAlloc_440_;
goto v_reusejp_438_;
}
v_reusejp_438_:
{
return v___x_439_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instSemiringOfNat(lean_object* v_00_u03b9_448_, lean_object* v_inst_449_, lean_object* v_A_450_, lean_object* v_inst_451_, lean_object* v_inst_452_, lean_object* v_inst_453_){
_start:
{
lean_object* v___x_454_; 
v___x_454_ = lp_mathlib_DirectSum_instSemiringOfNat___redArg(v_inst_451_, v_inst_452_, v_inst_453_);
return v___x_454_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instSemiringOfNat___boxed(lean_object* v_00_u03b9_455_, lean_object* v_inst_456_, lean_object* v_A_457_, lean_object* v_inst_458_, lean_object* v_inst_459_, lean_object* v_inst_460_){
_start:
{
lean_object* v_res_461_; 
v_res_461_ = lp_mathlib_DirectSum_instSemiringOfNat(v_00_u03b9_455_, v_inst_456_, v_A_457_, v_inst_458_, v_inst_459_, v_inst_460_);
lean_dec_ref(v_inst_456_);
return v_res_461_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_ofZeroRingHom___redArg(lean_object* v_inst_462_, lean_object* v_inst_463_, lean_object* v_inst_464_){
_start:
{
lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v_toZero_467_; lean_object* v___x_468_; 
v___x_465_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_464_);
v___x_466_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_465_);
v_toZero_467_ = lean_ctor_get(v___x_466_, 0);
lean_inc(v_toZero_467_);
lean_dec_ref(v___x_466_);
v___x_468_ = lp_mathlib_DirectSum_of___redArg(v_inst_463_, v_inst_462_, v_toZero_467_);
return v___x_468_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_ofZeroRingHom___redArg___boxed(lean_object* v_inst_469_, lean_object* v_inst_470_, lean_object* v_inst_471_){
_start:
{
lean_object* v_res_472_; 
v_res_472_ = lp_mathlib_DirectSum_ofZeroRingHom___redArg(v_inst_469_, v_inst_470_, v_inst_471_);
lean_dec_ref(v_inst_471_);
return v_res_472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_ofZeroRingHom(lean_object* v_00_u03b9_473_, lean_object* v_inst_474_, lean_object* v_A_475_, lean_object* v_inst_476_, lean_object* v_inst_477_, lean_object* v_inst_478_){
_start:
{
lean_object* v___x_479_; 
v___x_479_ = lp_mathlib_DirectSum_ofZeroRingHom___redArg(v_inst_474_, v_inst_476_, v_inst_477_);
return v___x_479_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_ofZeroRingHom___boxed(lean_object* v_00_u03b9_480_, lean_object* v_inst_481_, lean_object* v_A_482_, lean_object* v_inst_483_, lean_object* v_inst_484_, lean_object* v_inst_485_){
_start:
{
lean_object* v_res_486_; 
v_res_486_ = lp_mathlib_DirectSum_ofZeroRingHom(v_00_u03b9_480_, v_inst_481_, v_A_482_, v_inst_483_, v_inst_484_, v_inst_485_);
lean_dec_ref(v_inst_485_);
lean_dec_ref(v_inst_484_);
return v_res_486_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instModuleOfNat___redArg(lean_object* v_inst_487_, lean_object* v_inst_488_, lean_object* v_i_489_){
_start:
{
lean_object* v___x_490_; lean_object* v_toGNonUnitalNonAssocSemiring_491_; lean_object* v___x_492_; 
v___x_490_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_487_);
v_toGNonUnitalNonAssocSemiring_491_ = lean_ctor_get(v_inst_488_, 0);
lean_inc(v_toGNonUnitalNonAssocSemiring_491_);
lean_dec_ref(v_inst_488_);
v___x_492_ = lp_mathlib_GradedMonoid_GradeZero_smul___redArg(v___x_490_, v_toGNonUnitalNonAssocSemiring_491_, v_i_489_);
return v___x_492_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instModuleOfNat___redArg___boxed(lean_object* v_inst_493_, lean_object* v_inst_494_, lean_object* v_i_495_){
_start:
{
lean_object* v_res_496_; 
v_res_496_ = lp_mathlib_DirectSum_instModuleOfNat___redArg(v_inst_493_, v_inst_494_, v_i_495_);
lean_dec_ref(v_inst_493_);
return v_res_496_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instModuleOfNat(lean_object* v_00_u03b9_497_, lean_object* v_inst_498_, lean_object* v_A_499_, lean_object* v_inst_500_, lean_object* v_inst_501_, lean_object* v_inst_502_, lean_object* v_i_503_){
_start:
{
lean_object* v___x_504_; 
v___x_504_ = lp_mathlib_DirectSum_instModuleOfNat___redArg(v_inst_501_, v_inst_502_, v_i_503_);
return v___x_504_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instModuleOfNat___boxed(lean_object* v_00_u03b9_505_, lean_object* v_inst_506_, lean_object* v_A_507_, lean_object* v_inst_508_, lean_object* v_inst_509_, lean_object* v_inst_510_, lean_object* v_i_511_){
_start:
{
lean_object* v_res_512_; 
v_res_512_ = lp_mathlib_DirectSum_instModuleOfNat(v_00_u03b9_505_, v_inst_506_, v_A_507_, v_inst_508_, v_inst_509_, v_inst_510_, v_i_511_);
lean_dec_ref(v_inst_509_);
lean_dec_ref(v_inst_508_);
lean_dec_ref(v_inst_506_);
return v_res_512_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instCommSemiringOfNat___redArg___lam__0(lean_object* v_gnpow_513_, lean_object* v_toZero_514_, lean_object* v_n_515_, lean_object* v_x_516_){
_start:
{
lean_object* v___x_517_; 
v___x_517_ = lean_apply_3(v_gnpow_513_, v_n_515_, v_toZero_514_, v_x_516_);
return v___x_517_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instCommSemiringOfNat___redArg(lean_object* v_inst_518_, lean_object* v_inst_519_, lean_object* v_inst_520_){
_start:
{
lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v_toZero_523_; lean_object* v_toGNonUnitalNonAssocSemiring_524_; lean_object* v_gnpow_525_; lean_object* v_natCast_526_; lean_object* v___x_527_; lean_object* v_toGOne_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v_toAddMonoid_532_; lean_object* v___x_534_; uint8_t v_isShared_535_; uint8_t v_isSharedCheck_559_; 
v___x_521_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_519_);
lean_inc_ref(v___x_521_);
v___x_522_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_521_);
v_toZero_523_ = lean_ctor_get(v___x_522_, 0);
lean_inc(v_toZero_523_);
lean_dec_ref(v___x_522_);
v_toGNonUnitalNonAssocSemiring_524_ = lean_ctor_get(v_inst_520_, 0);
lean_inc(v_toGNonUnitalNonAssocSemiring_524_);
v_gnpow_525_ = lean_ctor_get(v_inst_520_, 2);
lean_inc(v_gnpow_525_);
v_natCast_526_ = lean_ctor_get(v_inst_520_, 3);
lean_inc(v_natCast_526_);
v___x_527_ = lp_mathlib_DirectSum_GSemiring_toGMonoid___redArg(v_inst_520_);
v_toGOne_528_ = lean_ctor_get(v___x_527_, 1);
lean_inc(v_toGOne_528_);
lean_dec_ref(v___x_527_);
lean_inc_ref(v_inst_518_);
v___x_529_ = lp_mathlib_DirectSum_instSemiringOfNat___redArg(v_inst_518_, v_inst_519_, v_inst_520_);
v___x_530_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v___x_529_);
v___x_531_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_530_);
v_toAddMonoid_532_ = lean_ctor_get(v___x_531_, 1);
v_isSharedCheck_559_ = !lean_is_exclusive(v___x_531_);
if (v_isSharedCheck_559_ == 0)
{
lean_object* v_unused_560_; lean_object* v_unused_561_; 
v_unused_560_ = lean_ctor_get(v___x_531_, 2);
lean_dec(v_unused_560_);
v_unused_561_ = lean_ctor_get(v___x_531_, 0);
lean_dec(v_unused_561_);
v___x_534_ = v___x_531_;
v_isShared_535_ = v_isSharedCheck_559_;
goto v_resetjp_533_;
}
else
{
lean_inc(v_toAddMonoid_532_);
lean_dec(v___x_531_);
v___x_534_ = lean_box(0);
v_isShared_535_ = v_isSharedCheck_559_;
goto v_resetjp_533_;
}
v_resetjp_533_:
{
lean_object* v_toNSMul_536_; lean_object* v___x_538_; uint8_t v_isShared_539_; uint8_t v_isSharedCheck_556_; 
v_toNSMul_536_ = lean_ctor_get(v_toAddMonoid_532_, 2);
v_isSharedCheck_556_ = !lean_is_exclusive(v_toAddMonoid_532_);
if (v_isSharedCheck_556_ == 0)
{
lean_object* v_unused_557_; lean_object* v_unused_558_; 
v_unused_557_ = lean_ctor_get(v_toAddMonoid_532_, 1);
lean_dec(v_unused_557_);
v_unused_558_ = lean_ctor_get(v_toAddMonoid_532_, 0);
lean_dec(v_unused_558_);
v___x_538_ = v_toAddMonoid_532_;
v_isShared_539_ = v_isSharedCheck_556_;
goto v_resetjp_537_;
}
else
{
lean_inc(v_toNSMul_536_);
lean_dec(v_toAddMonoid_532_);
v___x_538_ = lean_box(0);
v_isShared_539_ = v_isSharedCheck_556_;
goto v_resetjp_537_;
}
v_resetjp_537_:
{
lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v_toZero_543_; lean_object* v_toAdd_544_; lean_object* v___f_545_; lean_object* v___x_546_; lean_object* v___f_547_; lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___x_551_; 
lean_inc(v_toZero_523_);
v___x_540_ = lean_apply_1(v_inst_518_, v_toZero_523_);
v___x_541_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v___x_540_);
lean_dec_ref(v___x_540_);
v___x_542_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_541_);
v_toZero_543_ = lean_ctor_get(v___x_542_, 0);
lean_inc(v_toZero_543_);
v_toAdd_544_ = lean_ctor_get(v___x_542_, 1);
lean_inc(v_toAdd_544_);
lean_dec_ref(v___x_542_);
v___f_545_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instCommSemiringOfNat___redArg___lam__0), 4, 2);
lean_closure_set(v___f_545_, 0, v_gnpow_525_);
lean_closure_set(v___f_545_, 1, v_toZero_523_);
v___x_546_ = lp_mathlib_GradedMonoid_GradeZero_mul___redArg(v___x_521_, v_toGNonUnitalNonAssocSemiring_524_);
v___f_547_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_547_, 0, v_toNSMul_536_);
v___x_548_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_548_, 0, lean_box(0));
lean_closure_set(v___x_548_, 1, v_natCast_526_);
v___x_549_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_toAdd_544_, v_toZero_543_, v___f_547_);
if (v_isShared_539_ == 0)
{
lean_ctor_set(v___x_538_, 2, v___f_545_);
lean_ctor_set(v___x_538_, 1, v___x_546_);
lean_ctor_set(v___x_538_, 0, v_toGOne_528_);
v___x_551_ = v___x_538_;
goto v_reusejp_550_;
}
else
{
lean_object* v_reuseFailAlloc_555_; 
v_reuseFailAlloc_555_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_555_, 0, v_toGOne_528_);
lean_ctor_set(v_reuseFailAlloc_555_, 1, v___x_546_);
lean_ctor_set(v_reuseFailAlloc_555_, 2, v___f_545_);
v___x_551_ = v_reuseFailAlloc_555_;
goto v_reusejp_550_;
}
v_reusejp_550_:
{
lean_object* v___x_553_; 
if (v_isShared_535_ == 0)
{
lean_ctor_set(v___x_534_, 2, v___x_548_);
lean_ctor_set(v___x_534_, 1, v___x_551_);
lean_ctor_set(v___x_534_, 0, v___x_549_);
v___x_553_ = v___x_534_;
goto v_reusejp_552_;
}
else
{
lean_object* v_reuseFailAlloc_554_; 
v_reuseFailAlloc_554_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_554_, 0, v___x_549_);
lean_ctor_set(v_reuseFailAlloc_554_, 1, v___x_551_);
lean_ctor_set(v_reuseFailAlloc_554_, 2, v___x_548_);
v___x_553_ = v_reuseFailAlloc_554_;
goto v_reusejp_552_;
}
v_reusejp_552_:
{
return v___x_553_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instCommSemiringOfNat(lean_object* v_00_u03b9_562_, lean_object* v_inst_563_, lean_object* v_A_564_, lean_object* v_inst_565_, lean_object* v_inst_566_, lean_object* v_inst_567_){
_start:
{
lean_object* v___x_568_; 
v___x_568_ = lp_mathlib_DirectSum_instCommSemiringOfNat___redArg(v_inst_565_, v_inst_566_, v_inst_567_);
return v___x_568_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instCommSemiringOfNat___boxed(lean_object* v_00_u03b9_569_, lean_object* v_inst_570_, lean_object* v_A_571_, lean_object* v_inst_572_, lean_object* v_inst_573_, lean_object* v_inst_574_){
_start:
{
lean_object* v_res_575_; 
v_res_575_ = lp_mathlib_DirectSum_instCommSemiringOfNat(v_00_u03b9_569_, v_inst_570_, v_A_571_, v_inst_572_, v_inst_573_, v_inst_574_);
lean_dec_ref(v_inst_570_);
return v_res_575_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNonUnitalNonAssocRingOfNat___redArg(lean_object* v_inst_576_, lean_object* v_inst_577_, lean_object* v_inst_578_){
_start:
{
lean_object* v___x_579_; lean_object* v_toZero_580_; lean_object* v___x_581_; lean_object* v_toAddMonoid_582_; lean_object* v_toNeg_583_; lean_object* v_toSub_584_; lean_object* v_toZSMul_585_; lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v_toZero_588_; lean_object* v_toAdd_589_; lean_object* v___x_591_; uint8_t v_isShared_592_; uint8_t v_isSharedCheck_601_; 
lean_inc_ref(v_inst_577_);
v___x_579_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_577_);
v_toZero_580_ = lean_ctor_get(v___x_579_, 0);
lean_inc(v_toZero_580_);
lean_dec_ref(v___x_579_);
v___x_581_ = lean_apply_1(v_inst_576_, v_toZero_580_);
v_toAddMonoid_582_ = lean_ctor_get(v___x_581_, 0);
lean_inc_ref(v_toAddMonoid_582_);
v_toNeg_583_ = lean_ctor_get(v___x_581_, 1);
lean_inc(v_toNeg_583_);
v_toSub_584_ = lean_ctor_get(v___x_581_, 2);
lean_inc(v_toSub_584_);
v_toZSMul_585_ = lean_ctor_get(v___x_581_, 3);
lean_inc(v_toZSMul_585_);
lean_dec_ref(v___x_581_);
v___x_586_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_582_);
v___x_587_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_586_);
v_toZero_588_ = lean_ctor_get(v___x_587_, 0);
v_toAdd_589_ = lean_ctor_get(v___x_587_, 1);
v_isSharedCheck_601_ = !lean_is_exclusive(v___x_587_);
if (v_isSharedCheck_601_ == 0)
{
v___x_591_ = v___x_587_;
v_isShared_592_ = v_isSharedCheck_601_;
goto v_resetjp_590_;
}
else
{
lean_inc(v_toAdd_589_);
lean_inc(v_toZero_588_);
lean_dec(v___x_587_);
v___x_591_ = lean_box(0);
v_isShared_592_ = v_isSharedCheck_601_;
goto v_resetjp_590_;
}
v_resetjp_590_:
{
lean_object* v_toNSMul_593_; lean_object* v___x_594_; lean_object* v___f_595_; lean_object* v___f_596_; lean_object* v___x_597_; lean_object* v___x_599_; 
v_toNSMul_593_ = lean_ctor_get(v_toAddMonoid_582_, 2);
lean_inc(v_toNSMul_593_);
lean_dec_ref(v_toAddMonoid_582_);
v___x_594_ = lp_mathlib_GradedMonoid_GradeZero_mul___redArg(v_inst_577_, v_inst_578_);
v___f_595_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_595_, 0, v_toNSMul_593_);
v___f_596_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_596_, 0, v_toZSMul_585_);
v___x_597_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_toAdd_589_, v_toZero_588_, v___f_595_, v_toNeg_583_, v_toSub_584_, v___f_596_);
if (v_isShared_592_ == 0)
{
lean_ctor_set(v___x_591_, 1, v___x_594_);
lean_ctor_set(v___x_591_, 0, v___x_597_);
v___x_599_ = v___x_591_;
goto v_reusejp_598_;
}
else
{
lean_object* v_reuseFailAlloc_600_; 
v_reuseFailAlloc_600_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_600_, 0, v___x_597_);
lean_ctor_set(v_reuseFailAlloc_600_, 1, v___x_594_);
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
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNonUnitalNonAssocRingOfNat(lean_object* v_00_u03b9_602_, lean_object* v_inst_603_, lean_object* v_A_604_, lean_object* v_inst_605_, lean_object* v_inst_606_, lean_object* v_inst_607_){
_start:
{
lean_object* v___x_608_; 
v___x_608_ = lp_mathlib_DirectSum_instNonUnitalNonAssocRingOfNat___redArg(v_inst_605_, v_inst_606_, v_inst_607_);
return v___x_608_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instNonUnitalNonAssocRingOfNat___boxed(lean_object* v_00_u03b9_609_, lean_object* v_inst_610_, lean_object* v_A_611_, lean_object* v_inst_612_, lean_object* v_inst_613_, lean_object* v_inst_614_){
_start:
{
lean_object* v_res_615_; 
v_res_615_ = lp_mathlib_DirectSum_instNonUnitalNonAssocRingOfNat(v_00_u03b9_609_, v_inst_610_, v_A_611_, v_inst_612_, v_inst_613_, v_inst_614_);
lean_dec_ref(v_inst_610_);
return v_res_615_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instIntCastOfNat___redArg(lean_object* v_inst_616_){
_start:
{
lean_object* v_intCast_617_; 
v_intCast_617_ = lean_ctor_get(v_inst_616_, 1);
lean_inc(v_intCast_617_);
return v_intCast_617_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instIntCastOfNat___redArg___boxed(lean_object* v_inst_618_){
_start:
{
lean_object* v_res_619_; 
v_res_619_ = lp_mathlib_DirectSum_instIntCastOfNat___redArg(v_inst_618_);
lean_dec_ref(v_inst_618_);
return v_res_619_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instIntCastOfNat(lean_object* v_00_u03b9_620_, lean_object* v_A_621_, lean_object* v_inst_622_, lean_object* v_inst_623_, lean_object* v_inst_624_){
_start:
{
lean_object* v_intCast_625_; 
v_intCast_625_ = lean_ctor_get(v_inst_624_, 1);
lean_inc(v_intCast_625_);
return v_intCast_625_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instIntCastOfNat___boxed(lean_object* v_00_u03b9_626_, lean_object* v_A_627_, lean_object* v_inst_628_, lean_object* v_inst_629_, lean_object* v_inst_630_){
_start:
{
lean_object* v_res_631_; 
v_res_631_ = lp_mathlib_DirectSum_instIntCastOfNat(v_00_u03b9_626_, v_A_627_, v_inst_628_, v_inst_629_, v_inst_630_);
lean_dec_ref(v_inst_630_);
lean_dec_ref(v_inst_629_);
lean_dec_ref(v_inst_628_);
return v_res_631_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instRingOfNat___redArg___lam__2(lean_object* v_toZSMul_632_, lean_object* v_n_633_, lean_object* v_x_634_){
_start:
{
lean_object* v___x_635_; 
v___x_635_ = lean_apply_2(v_toZSMul_632_, v_n_633_, v_x_634_);
return v___x_635_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instRingOfNat___redArg(lean_object* v_inst_636_, lean_object* v_inst_637_, lean_object* v_inst_638_){
_start:
{
lean_object* v___x_639_; lean_object* v___x_640_; lean_object* v_toGSemiring_641_; lean_object* v_toZero_642_; lean_object* v_intCast_643_; lean_object* v_toGNonUnitalNonAssocSemiring_644_; lean_object* v_gnpow_645_; lean_object* v_natCast_646_; lean_object* v___x_647_; lean_object* v_toAddMonoid_648_; lean_object* v_toNeg_649_; lean_object* v_toSub_650_; lean_object* v_toZSMul_651_; lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v_toZero_654_; lean_object* v_toAdd_655_; lean_object* v___x_656_; lean_object* v_toGOne_657_; lean_object* v___f_658_; lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v___x_661_; lean_object* v_toAddMonoid_662_; lean_object* v___x_664_; uint8_t v_isShared_665_; uint8_t v_isSharedCheck_691_; 
v___x_639_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_637_);
lean_inc_ref(v___x_639_);
v___x_640_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_639_);
v_toGSemiring_641_ = lean_ctor_get(v_inst_638_, 0);
lean_inc_ref(v_toGSemiring_641_);
v_toZero_642_ = lean_ctor_get(v___x_640_, 0);
lean_inc_n(v_toZero_642_, 2);
lean_dec_ref(v___x_640_);
v_intCast_643_ = lean_ctor_get(v_inst_638_, 1);
lean_inc(v_intCast_643_);
lean_dec_ref(v_inst_638_);
v_toGNonUnitalNonAssocSemiring_644_ = lean_ctor_get(v_toGSemiring_641_, 0);
lean_inc(v_toGNonUnitalNonAssocSemiring_644_);
v_gnpow_645_ = lean_ctor_get(v_toGSemiring_641_, 2);
lean_inc(v_gnpow_645_);
v_natCast_646_ = lean_ctor_get(v_toGSemiring_641_, 3);
lean_inc(v_natCast_646_);
lean_inc_ref(v_inst_636_);
v___x_647_ = lean_apply_1(v_inst_636_, v_toZero_642_);
v_toAddMonoid_648_ = lean_ctor_get(v___x_647_, 0);
lean_inc_ref(v_toAddMonoid_648_);
v_toNeg_649_ = lean_ctor_get(v___x_647_, 1);
lean_inc(v_toNeg_649_);
v_toSub_650_ = lean_ctor_get(v___x_647_, 2);
lean_inc(v_toSub_650_);
v_toZSMul_651_ = lean_ctor_get(v___x_647_, 3);
lean_inc(v_toZSMul_651_);
lean_dec_ref(v___x_647_);
v___x_652_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_648_);
lean_dec_ref(v_toAddMonoid_648_);
v___x_653_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_652_);
v_toZero_654_ = lean_ctor_get(v___x_653_, 0);
lean_inc(v_toZero_654_);
v_toAdd_655_ = lean_ctor_get(v___x_653_, 1);
lean_inc(v_toAdd_655_);
lean_dec_ref(v___x_653_);
v___x_656_ = lp_mathlib_DirectSum_GSemiring_toGMonoid___redArg(v_toGSemiring_641_);
v_toGOne_657_ = lean_ctor_get(v___x_656_, 1);
lean_inc(v_toGOne_657_);
lean_dec_ref(v___x_656_);
v___f_658_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_nonAssocRing___redArg___lam__0), 2, 1);
lean_closure_set(v___f_658_, 0, v_inst_636_);
v___x_659_ = lp_mathlib_DirectSum_instSemiringOfNat___redArg(v___f_658_, v_inst_637_, v_toGSemiring_641_);
v___x_660_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v___x_659_);
v___x_661_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_660_);
v_toAddMonoid_662_ = lean_ctor_get(v___x_661_, 1);
v_isSharedCheck_691_ = !lean_is_exclusive(v___x_661_);
if (v_isSharedCheck_691_ == 0)
{
lean_object* v_unused_692_; lean_object* v_unused_693_; 
v_unused_692_ = lean_ctor_get(v___x_661_, 2);
lean_dec(v_unused_692_);
v_unused_693_ = lean_ctor_get(v___x_661_, 0);
lean_dec(v_unused_693_);
v___x_664_ = v___x_661_;
v_isShared_665_ = v_isSharedCheck_691_;
goto v_resetjp_663_;
}
else
{
lean_inc(v_toAddMonoid_662_);
lean_dec(v___x_661_);
v___x_664_ = lean_box(0);
v_isShared_665_ = v_isSharedCheck_691_;
goto v_resetjp_663_;
}
v_resetjp_663_:
{
lean_object* v_toNSMul_666_; lean_object* v___x_668_; uint8_t v_isShared_669_; uint8_t v_isSharedCheck_688_; 
v_toNSMul_666_ = lean_ctor_get(v_toAddMonoid_662_, 2);
v_isSharedCheck_688_ = !lean_is_exclusive(v_toAddMonoid_662_);
if (v_isSharedCheck_688_ == 0)
{
lean_object* v_unused_689_; lean_object* v_unused_690_; 
v_unused_689_ = lean_ctor_get(v_toAddMonoid_662_, 1);
lean_dec(v_unused_689_);
v_unused_690_ = lean_ctor_get(v_toAddMonoid_662_, 0);
lean_dec(v_unused_690_);
v___x_668_ = v_toAddMonoid_662_;
v_isShared_669_ = v_isSharedCheck_688_;
goto v_resetjp_667_;
}
else
{
lean_inc(v_toNSMul_666_);
lean_dec(v_toAddMonoid_662_);
v___x_668_ = lean_box(0);
v_isShared_669_ = v_isSharedCheck_688_;
goto v_resetjp_667_;
}
v_resetjp_667_:
{
lean_object* v___f_670_; lean_object* v___f_671_; lean_object* v___x_672_; lean_object* v_toNeg_673_; lean_object* v_toSub_674_; lean_object* v___f_675_; lean_object* v___x_676_; lean_object* v___f_677_; lean_object* v___x_678_; lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_682_; 
v___f_670_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_670_, 0, v_toNSMul_666_);
lean_inc(v_toZSMul_651_);
v___f_671_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_671_, 0, v_toZSMul_651_);
lean_inc_ref(v___f_670_);
lean_inc(v_toZero_654_);
lean_inc(v_toAdd_655_);
v___x_672_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_toAdd_655_, v_toZero_654_, v___f_670_, v_toNeg_649_, v_toSub_650_, v___f_671_);
v_toNeg_673_ = lean_ctor_get(v___x_672_, 1);
lean_inc(v_toNeg_673_);
v_toSub_674_ = lean_ctor_get(v___x_672_, 2);
lean_inc(v_toSub_674_);
lean_dec_ref(v___x_672_);
v___f_675_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instCommSemiringOfNat___redArg___lam__0), 4, 2);
lean_closure_set(v___f_675_, 0, v_gnpow_645_);
lean_closure_set(v___f_675_, 1, v_toZero_642_);
v___x_676_ = lp_mathlib_GradedMonoid_GradeZero_mul___redArg(v___x_639_, v_toGNonUnitalNonAssocSemiring_644_);
v___f_677_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instRingOfNat___redArg___lam__2), 3, 1);
lean_closure_set(v___f_677_, 0, v_toZSMul_651_);
v___x_678_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_678_, 0, lean_box(0));
lean_closure_set(v___x_678_, 1, v_intCast_643_);
v___x_679_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_679_, 0, lean_box(0));
lean_closure_set(v___x_679_, 1, v_natCast_646_);
v___x_680_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_toAdd_655_, v_toZero_654_, v___f_670_);
if (v_isShared_669_ == 0)
{
lean_ctor_set(v___x_668_, 2, v___f_675_);
lean_ctor_set(v___x_668_, 1, v___x_676_);
lean_ctor_set(v___x_668_, 0, v_toGOne_657_);
v___x_682_ = v___x_668_;
goto v_reusejp_681_;
}
else
{
lean_object* v_reuseFailAlloc_687_; 
v_reuseFailAlloc_687_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_687_, 0, v_toGOne_657_);
lean_ctor_set(v_reuseFailAlloc_687_, 1, v___x_676_);
lean_ctor_set(v_reuseFailAlloc_687_, 2, v___f_675_);
v___x_682_ = v_reuseFailAlloc_687_;
goto v_reusejp_681_;
}
v_reusejp_681_:
{
lean_object* v___x_684_; 
if (v_isShared_665_ == 0)
{
lean_ctor_set(v___x_664_, 2, v___x_679_);
lean_ctor_set(v___x_664_, 1, v___x_682_);
lean_ctor_set(v___x_664_, 0, v___x_680_);
v___x_684_ = v___x_664_;
goto v_reusejp_683_;
}
else
{
lean_object* v_reuseFailAlloc_686_; 
v_reuseFailAlloc_686_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_686_, 0, v___x_680_);
lean_ctor_set(v_reuseFailAlloc_686_, 1, v___x_682_);
lean_ctor_set(v_reuseFailAlloc_686_, 2, v___x_679_);
v___x_684_ = v_reuseFailAlloc_686_;
goto v_reusejp_683_;
}
v_reusejp_683_:
{
lean_object* v___x_685_; 
v___x_685_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_685_, 0, v___x_684_);
lean_ctor_set(v___x_685_, 1, v_toNeg_673_);
lean_ctor_set(v___x_685_, 2, v_toSub_674_);
lean_ctor_set(v___x_685_, 3, v___f_677_);
lean_ctor_set(v___x_685_, 4, v___x_678_);
return v___x_685_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instRingOfNat(lean_object* v_00_u03b9_694_, lean_object* v_inst_695_, lean_object* v_A_696_, lean_object* v_inst_697_, lean_object* v_inst_698_, lean_object* v_inst_699_){
_start:
{
lean_object* v___x_700_; 
v___x_700_ = lp_mathlib_DirectSum_instRingOfNat___redArg(v_inst_697_, v_inst_698_, v_inst_699_);
return v___x_700_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instRingOfNat___boxed(lean_object* v_00_u03b9_701_, lean_object* v_inst_702_, lean_object* v_A_703_, lean_object* v_inst_704_, lean_object* v_inst_705_, lean_object* v_inst_706_){
_start:
{
lean_object* v_res_707_; 
v_res_707_ = lp_mathlib_DirectSum_instRingOfNat(v_00_u03b9_701_, v_inst_702_, v_A_703_, v_inst_704_, v_inst_705_, v_inst_706_);
lean_dec_ref(v_inst_702_);
return v_res_707_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instCommRingOfNat___redArg___lam__0(lean_object* v_inst_708_, lean_object* v_inst_709_, lean_object* v_n_710_, lean_object* v_x_711_){
_start:
{
lean_object* v___x_712_; lean_object* v___x_713_; lean_object* v_toZero_714_; lean_object* v___x_715_; lean_object* v_toZSMul_716_; lean_object* v___x_717_; 
v___x_712_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_708_);
v___x_713_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_712_);
v_toZero_714_ = lean_ctor_get(v___x_713_, 0);
lean_inc(v_toZero_714_);
lean_dec_ref(v___x_713_);
v___x_715_ = lean_apply_1(v_inst_709_, v_toZero_714_);
v_toZSMul_716_ = lean_ctor_get(v___x_715_, 3);
lean_inc(v_toZSMul_716_);
lean_dec_ref(v___x_715_);
v___x_717_ = lean_apply_2(v_toZSMul_716_, v_n_710_, v_x_711_);
return v___x_717_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instCommRingOfNat___redArg___lam__0___boxed(lean_object* v_inst_718_, lean_object* v_inst_719_, lean_object* v_n_720_, lean_object* v_x_721_){
_start:
{
lean_object* v_res_722_; 
v_res_722_ = lp_mathlib_DirectSum_instCommRingOfNat___redArg___lam__0(v_inst_718_, v_inst_719_, v_n_720_, v_x_721_);
lean_dec_ref(v_inst_718_);
return v_res_722_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instCommRingOfNat___redArg(lean_object* v_inst_723_, lean_object* v_inst_724_, lean_object* v_inst_725_){
_start:
{
lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v_toGSemiring_728_; lean_object* v_toZero_729_; lean_object* v_intCast_730_; lean_object* v_toGNonUnitalNonAssocSemiring_731_; lean_object* v_gnpow_732_; lean_object* v_natCast_733_; lean_object* v___x_734_; lean_object* v___x_735_; lean_object* v___x_736_; lean_object* v_toAddMonoidWithOne_737_; lean_object* v___x_739_; uint8_t v_isShared_740_; uint8_t v_isSharedCheck_786_; 
v___x_726_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_724_);
lean_inc_ref(v___x_726_);
v___x_727_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_726_);
v_toGSemiring_728_ = lean_ctor_get(v_inst_725_, 0);
lean_inc_ref(v_toGSemiring_728_);
v_toZero_729_ = lean_ctor_get(v___x_727_, 0);
lean_inc(v_toZero_729_);
lean_dec_ref(v___x_727_);
v_intCast_730_ = lean_ctor_get(v_inst_725_, 1);
lean_inc(v_intCast_730_);
v_toGNonUnitalNonAssocSemiring_731_ = lean_ctor_get(v_toGSemiring_728_, 0);
lean_inc(v_toGNonUnitalNonAssocSemiring_731_);
v_gnpow_732_ = lean_ctor_get(v_toGSemiring_728_, 2);
lean_inc(v_gnpow_732_);
v_natCast_733_ = lean_ctor_get(v_toGSemiring_728_, 3);
lean_inc(v_natCast_733_);
lean_inc_ref(v_inst_724_);
lean_inc_ref(v_inst_723_);
v___x_734_ = lp_mathlib_DirectSum_instRingOfNat___redArg(v_inst_723_, v_inst_724_, v_inst_725_);
v___x_735_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v___x_734_);
v___x_736_ = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(v___x_735_);
v_toAddMonoidWithOne_737_ = lean_ctor_get(v___x_735_, 1);
v_isSharedCheck_786_ = !lean_is_exclusive(v___x_735_);
if (v_isSharedCheck_786_ == 0)
{
lean_object* v_unused_787_; lean_object* v_unused_788_; lean_object* v_unused_789_; lean_object* v_unused_790_; 
v_unused_787_ = lean_ctor_get(v___x_735_, 4);
lean_dec(v_unused_787_);
v_unused_788_ = lean_ctor_get(v___x_735_, 3);
lean_dec(v_unused_788_);
v_unused_789_ = lean_ctor_get(v___x_735_, 2);
lean_dec(v_unused_789_);
v_unused_790_ = lean_ctor_get(v___x_735_, 0);
lean_dec(v_unused_790_);
v___x_739_ = v___x_735_;
v_isShared_740_ = v_isSharedCheck_786_;
goto v_resetjp_738_;
}
else
{
lean_inc(v_toAddMonoidWithOne_737_);
lean_dec(v___x_735_);
v___x_739_ = lean_box(0);
v_isShared_740_ = v_isSharedCheck_786_;
goto v_resetjp_738_;
}
v_resetjp_738_:
{
lean_object* v_toAddMonoid_741_; lean_object* v_toNeg_742_; lean_object* v_toSub_743_; lean_object* v_toZSMul_744_; lean_object* v_toNSMul_745_; lean_object* v___x_747_; uint8_t v_isShared_748_; uint8_t v_isSharedCheck_783_; 
v_toAddMonoid_741_ = lean_ctor_get(v_toAddMonoidWithOne_737_, 1);
lean_inc_ref(v_toAddMonoid_741_);
lean_dec_ref(v_toAddMonoidWithOne_737_);
v_toNeg_742_ = lean_ctor_get(v___x_736_, 1);
lean_inc(v_toNeg_742_);
v_toSub_743_ = lean_ctor_get(v___x_736_, 2);
lean_inc(v_toSub_743_);
v_toZSMul_744_ = lean_ctor_get(v___x_736_, 3);
lean_inc(v_toZSMul_744_);
lean_dec_ref(v___x_736_);
v_toNSMul_745_ = lean_ctor_get(v_toAddMonoid_741_, 2);
v_isSharedCheck_783_ = !lean_is_exclusive(v_toAddMonoid_741_);
if (v_isSharedCheck_783_ == 0)
{
lean_object* v_unused_784_; lean_object* v_unused_785_; 
v_unused_784_ = lean_ctor_get(v_toAddMonoid_741_, 1);
lean_dec(v_unused_784_);
v_unused_785_ = lean_ctor_get(v_toAddMonoid_741_, 0);
lean_dec(v_unused_785_);
v___x_747_ = v_toAddMonoid_741_;
v_isShared_748_ = v_isSharedCheck_783_;
goto v_resetjp_746_;
}
else
{
lean_inc(v_toNSMul_745_);
lean_dec(v_toAddMonoid_741_);
v___x_747_ = lean_box(0);
v_isShared_748_ = v_isSharedCheck_783_;
goto v_resetjp_746_;
}
v_resetjp_746_:
{
lean_object* v___x_749_; lean_object* v_toAddMonoid_750_; lean_object* v___x_751_; lean_object* v___x_752_; lean_object* v_toZero_753_; lean_object* v_toAdd_754_; lean_object* v___x_755_; lean_object* v_toGOne_756_; lean_object* v___x_758_; uint8_t v_isShared_759_; uint8_t v_isSharedCheck_780_; 
lean_inc_ref(v_inst_723_);
lean_inc(v_toZero_729_);
v___x_749_ = lean_apply_1(v_inst_723_, v_toZero_729_);
v_toAddMonoid_750_ = lean_ctor_get(v___x_749_, 0);
lean_inc_ref(v_toAddMonoid_750_);
lean_dec_ref(v___x_749_);
v___x_751_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_750_);
lean_dec_ref(v_toAddMonoid_750_);
v___x_752_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_751_);
v_toZero_753_ = lean_ctor_get(v___x_752_, 0);
lean_inc(v_toZero_753_);
v_toAdd_754_ = lean_ctor_get(v___x_752_, 1);
lean_inc(v_toAdd_754_);
lean_dec_ref(v___x_752_);
v___x_755_ = lp_mathlib_DirectSum_GSemiring_toGMonoid___redArg(v_toGSemiring_728_);
lean_dec_ref(v_toGSemiring_728_);
v_toGOne_756_ = lean_ctor_get(v___x_755_, 1);
v_isSharedCheck_780_ = !lean_is_exclusive(v___x_755_);
if (v_isSharedCheck_780_ == 0)
{
lean_object* v_unused_781_; lean_object* v_unused_782_; 
v_unused_781_ = lean_ctor_get(v___x_755_, 2);
lean_dec(v_unused_781_);
v_unused_782_ = lean_ctor_get(v___x_755_, 0);
lean_dec(v_unused_782_);
v___x_758_ = v___x_755_;
v_isShared_759_ = v_isSharedCheck_780_;
goto v_resetjp_757_;
}
else
{
lean_inc(v_toGOne_756_);
lean_dec(v___x_755_);
v___x_758_ = lean_box(0);
v_isShared_759_ = v_isSharedCheck_780_;
goto v_resetjp_757_;
}
v_resetjp_757_:
{
lean_object* v___f_760_; lean_object* v___f_761_; lean_object* v___x_762_; lean_object* v_toNeg_763_; lean_object* v_toSub_764_; lean_object* v___f_765_; lean_object* v___f_766_; lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v___x_769_; lean_object* v___x_770_; lean_object* v___x_772_; 
v___f_760_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_760_, 0, v_toNSMul_745_);
v___f_761_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_761_, 0, v_toZSMul_744_);
lean_inc_ref(v___f_760_);
lean_inc(v_toZero_753_);
lean_inc(v_toAdd_754_);
v___x_762_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_toAdd_754_, v_toZero_753_, v___f_760_, v_toNeg_742_, v_toSub_743_, v___f_761_);
v_toNeg_763_ = lean_ctor_get(v___x_762_, 1);
lean_inc(v_toNeg_763_);
v_toSub_764_ = lean_ctor_get(v___x_762_, 2);
lean_inc(v_toSub_764_);
lean_dec_ref(v___x_762_);
v___f_765_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instCommRingOfNat___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_765_, 0, v_inst_724_);
lean_closure_set(v___f_765_, 1, v_inst_723_);
v___f_766_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_instCommSemiringOfNat___redArg___lam__0), 4, 2);
lean_closure_set(v___f_766_, 0, v_gnpow_732_);
lean_closure_set(v___f_766_, 1, v_toZero_729_);
v___x_767_ = lp_mathlib_GradedMonoid_GradeZero_mul___redArg(v___x_726_, v_toGNonUnitalNonAssocSemiring_731_);
v___x_768_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_768_, 0, lean_box(0));
lean_closure_set(v___x_768_, 1, v_intCast_730_);
v___x_769_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_769_, 0, lean_box(0));
lean_closure_set(v___x_769_, 1, v_natCast_733_);
v___x_770_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_toAdd_754_, v_toZero_753_, v___f_760_);
if (v_isShared_759_ == 0)
{
lean_ctor_set(v___x_758_, 2, v___f_766_);
lean_ctor_set(v___x_758_, 1, v___x_767_);
lean_ctor_set(v___x_758_, 0, v_toGOne_756_);
v___x_772_ = v___x_758_;
goto v_reusejp_771_;
}
else
{
lean_object* v_reuseFailAlloc_779_; 
v_reuseFailAlloc_779_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_779_, 0, v_toGOne_756_);
lean_ctor_set(v_reuseFailAlloc_779_, 1, v___x_767_);
lean_ctor_set(v_reuseFailAlloc_779_, 2, v___f_766_);
v___x_772_ = v_reuseFailAlloc_779_;
goto v_reusejp_771_;
}
v_reusejp_771_:
{
lean_object* v___x_774_; 
if (v_isShared_748_ == 0)
{
lean_ctor_set(v___x_747_, 2, v___x_769_);
lean_ctor_set(v___x_747_, 1, v___x_772_);
lean_ctor_set(v___x_747_, 0, v___x_770_);
v___x_774_ = v___x_747_;
goto v_reusejp_773_;
}
else
{
lean_object* v_reuseFailAlloc_778_; 
v_reuseFailAlloc_778_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_778_, 0, v___x_770_);
lean_ctor_set(v_reuseFailAlloc_778_, 1, v___x_772_);
lean_ctor_set(v_reuseFailAlloc_778_, 2, v___x_769_);
v___x_774_ = v_reuseFailAlloc_778_;
goto v_reusejp_773_;
}
v_reusejp_773_:
{
lean_object* v___x_776_; 
if (v_isShared_740_ == 0)
{
lean_ctor_set(v___x_739_, 4, v___x_768_);
lean_ctor_set(v___x_739_, 3, v___f_765_);
lean_ctor_set(v___x_739_, 2, v_toSub_764_);
lean_ctor_set(v___x_739_, 1, v_toNeg_763_);
lean_ctor_set(v___x_739_, 0, v___x_774_);
v___x_776_ = v___x_739_;
goto v_reusejp_775_;
}
else
{
lean_object* v_reuseFailAlloc_777_; 
v_reuseFailAlloc_777_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_777_, 0, v___x_774_);
lean_ctor_set(v_reuseFailAlloc_777_, 1, v_toNeg_763_);
lean_ctor_set(v_reuseFailAlloc_777_, 2, v_toSub_764_);
lean_ctor_set(v_reuseFailAlloc_777_, 3, v___f_765_);
lean_ctor_set(v_reuseFailAlloc_777_, 4, v___x_768_);
v___x_776_ = v_reuseFailAlloc_777_;
goto v_reusejp_775_;
}
v_reusejp_775_:
{
return v___x_776_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instCommRingOfNat(lean_object* v_00_u03b9_791_, lean_object* v_inst_792_, lean_object* v_A_793_, lean_object* v_inst_794_, lean_object* v_inst_795_, lean_object* v_inst_796_){
_start:
{
lean_object* v___x_797_; 
v___x_797_ = lp_mathlib_DirectSum_instCommRingOfNat___redArg(v_inst_794_, v_inst_795_, v_inst_796_);
return v___x_797_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_instCommRingOfNat___boxed(lean_object* v_00_u03b9_798_, lean_object* v_inst_799_, lean_object* v_A_800_, lean_object* v_inst_801_, lean_object* v_inst_802_, lean_object* v_inst_803_){
_start:
{
lean_object* v_res_804_; 
v_res_804_ = lp_mathlib_DirectSum_instCommRingOfNat(v_00_u03b9_798_, v_inst_799_, v_A_800_, v_inst_801_, v_inst_802_, v_inst_803_);
lean_dec_ref(v_inst_799_);
return v_res_804_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toSemiring___redArg___lam__0(lean_object* v_inst_805_, lean_object* v_inst_806_, lean_object* v_toAddCommMonoid_807_, lean_object* v_f_808_, lean_object* v___y_809_){
_start:
{
lean_object* v___x_47__overap_810_; lean_object* v___x_811_; 
v___x_47__overap_810_ = lp_mathlib_DirectSum_toAddMonoid___redArg(v_inst_805_, v_inst_806_, v_toAddCommMonoid_807_, v_f_808_);
v___x_811_ = lean_apply_1(v___x_47__overap_810_, v___y_809_);
return v___x_811_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toSemiring___redArg(lean_object* v_inst_812_, lean_object* v_inst_813_, lean_object* v_inst_814_, lean_object* v_f_815_){
_start:
{
lean_object* v_toAddCommMonoid_816_; lean_object* v___f_817_; 
v_toAddCommMonoid_816_ = lean_ctor_get(v_inst_814_, 0);
lean_inc_ref(v_toAddCommMonoid_816_);
lean_dec_ref(v_inst_814_);
v___f_817_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_toSemiring___redArg___lam__0), 5, 4);
lean_closure_set(v___f_817_, 0, v_inst_813_);
lean_closure_set(v___f_817_, 1, v_inst_812_);
lean_closure_set(v___f_817_, 2, v_toAddCommMonoid_816_);
lean_closure_set(v___f_817_, 3, v_f_815_);
return v___f_817_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toSemiring(lean_object* v_00_u03b9_818_, lean_object* v_inst_819_, lean_object* v_A_820_, lean_object* v_R_821_, lean_object* v_inst_822_, lean_object* v_inst_823_, lean_object* v_inst_824_, lean_object* v_inst_825_, lean_object* v_f_826_, lean_object* v_hone_827_, lean_object* v_hmul_828_){
_start:
{
lean_object* v___x_829_; 
v___x_829_ = lp_mathlib_DirectSum_toSemiring___redArg(v_inst_819_, v_inst_822_, v_inst_825_, v_f_826_);
return v___x_829_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_toSemiring___boxed(lean_object* v_00_u03b9_830_, lean_object* v_inst_831_, lean_object* v_A_832_, lean_object* v_R_833_, lean_object* v_inst_834_, lean_object* v_inst_835_, lean_object* v_inst_836_, lean_object* v_inst_837_, lean_object* v_f_838_, lean_object* v_hone_839_, lean_object* v_hmul_840_){
_start:
{
lean_object* v_res_841_; 
v_res_841_ = lp_mathlib_DirectSum_toSemiring(v_00_u03b9_830_, v_inst_831_, v_A_832_, v_R_833_, v_inst_834_, v_inst_835_, v_inst_836_, v_inst_837_, v_f_838_, v_hone_839_, v_hmul_840_);
lean_dec_ref(v_inst_836_);
lean_dec_ref(v_inst_835_);
return v_res_841_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_liftRingHom___redArg___lam__0(lean_object* v_f_842_, lean_object* v_x_843_, lean_object* v___y_844_){
_start:
{
lean_object* v___x_845_; 
v___x_845_ = lean_apply_2(v_f_842_, v_x_843_, v___y_844_);
return v___x_845_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_liftRingHom___redArg___lam__1(lean_object* v_inst_846_, lean_object* v_inst_847_, lean_object* v_inst_848_, lean_object* v_f_849_, lean_object* v___y_850_){
_start:
{
lean_object* v___f_851_; lean_object* v___x_79__overap_852_; lean_object* v___x_853_; 
v___f_851_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_liftRingHom___redArg___lam__0), 3, 1);
lean_closure_set(v___f_851_, 0, v_f_849_);
v___x_79__overap_852_ = lp_mathlib_DirectSum_toSemiring___redArg(v_inst_846_, v_inst_847_, v_inst_848_, v___f_851_);
v___x_853_ = lean_apply_1(v___x_79__overap_852_, v___y_850_);
return v___x_853_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_liftRingHom___redArg___lam__2(lean_object* v_F_854_, lean_object* v___y_855_){
_start:
{
lean_object* v___x_856_; 
v___x_856_ = lean_apply_1(v_F_854_, v___y_855_);
return v___x_856_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_liftRingHom___redArg___lam__3(lean_object* v_inst_857_, lean_object* v_inst_858_, lean_object* v_F_859_, lean_object* v___y_860_, lean_object* v___y_861_){
_start:
{
lean_object* v___f_862_; lean_object* v___x_863_; lean_object* v___x_864_; 
v___f_862_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_liftRingHom___redArg___lam__2), 2, 1);
lean_closure_set(v___f_862_, 0, v_F_859_);
v___x_863_ = lp_mathlib_DirectSum_of___redArg(v_inst_857_, v_inst_858_, v___y_860_);
v___x_864_ = lp_mathlib_OneHom_comp___redArg___lam__0(v___x_863_, v___f_862_, v___y_861_);
return v___x_864_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_liftRingHom___redArg(lean_object* v_inst_865_, lean_object* v_inst_866_, lean_object* v_inst_867_){
_start:
{
lean_object* v___f_868_; lean_object* v___f_869_; lean_object* v___x_870_; 
lean_inc_ref(v_inst_866_);
lean_inc_ref(v_inst_865_);
v___f_868_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_liftRingHom___redArg___lam__1), 5, 3);
lean_closure_set(v___f_868_, 0, v_inst_865_);
lean_closure_set(v___f_868_, 1, v_inst_866_);
lean_closure_set(v___f_868_, 2, v_inst_867_);
v___f_869_ = lean_alloc_closure((void*)(lp_mathlib_DirectSum_liftRingHom___redArg___lam__3), 5, 2);
lean_closure_set(v___f_869_, 0, v_inst_866_);
lean_closure_set(v___f_869_, 1, v_inst_865_);
v___x_870_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_870_, 0, v___f_868_);
lean_ctor_set(v___x_870_, 1, v___f_869_);
return v___x_870_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_liftRingHom(lean_object* v_00_u03b9_871_, lean_object* v_inst_872_, lean_object* v_A_873_, lean_object* v_R_874_, lean_object* v_inst_875_, lean_object* v_inst_876_, lean_object* v_inst_877_, lean_object* v_inst_878_){
_start:
{
lean_object* v___x_879_; 
v___x_879_ = lp_mathlib_DirectSum_liftRingHom___redArg(v_inst_872_, v_inst_875_, v_inst_878_);
return v___x_879_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DirectSum_liftRingHom___boxed(lean_object* v_00_u03b9_880_, lean_object* v_inst_881_, lean_object* v_A_882_, lean_object* v_R_883_, lean_object* v_inst_884_, lean_object* v_inst_885_, lean_object* v_inst_886_, lean_object* v_inst_887_){
_start:
{
lean_object* v_res_888_; 
v_res_888_ = lp_mathlib_DirectSum_liftRingHom(v_00_u03b9_880_, v_inst_881_, v_A_882_, v_R_883_, v_inst_884_, v_inst_885_, v_inst_886_, v_inst_887_);
lean_dec_ref(v_inst_886_);
lean_dec_ref(v_inst_885_);
return v_res_888_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocSemiring_directSumGNonUnitalNonAssocSemiring___redArg(lean_object* v_inst_889_){
_start:
{
lean_object* v___x_890_; lean_object* v_toMul_891_; lean_object* v___f_892_; 
v___x_890_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v_inst_889_);
v_toMul_891_ = lean_ctor_get(v___x_890_, 0);
lean_inc(v_toMul_891_);
lean_dec_ref(v___x_890_);
v___f_892_ = lean_alloc_closure((void*)(lp_mathlib_Mul_gMul___redArg___lam__0___boxed), 5, 1);
lean_closure_set(v___f_892_, 0, v_toMul_891_);
return v___f_892_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocSemiring_directSumGNonUnitalNonAssocSemiring(lean_object* v_00_u03b9_893_, lean_object* v_R_894_, lean_object* v_inst_895_, lean_object* v_inst_896_){
_start:
{
lean_object* v___x_897_; 
v___x_897_ = lp_mathlib_NonUnitalNonAssocSemiring_directSumGNonUnitalNonAssocSemiring___redArg(v_inst_896_);
return v___x_897_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NonUnitalNonAssocSemiring_directSumGNonUnitalNonAssocSemiring___boxed(lean_object* v_00_u03b9_898_, lean_object* v_R_899_, lean_object* v_inst_900_, lean_object* v_inst_901_){
_start:
{
lean_object* v_res_902_; 
v_res_902_ = lp_mathlib_NonUnitalNonAssocSemiring_directSumGNonUnitalNonAssocSemiring(v_00_u03b9_898_, v_R_899_, v_inst_900_, v_inst_901_);
lean_dec_ref(v_inst_900_);
return v_res_902_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_directSumGSemiring___redArg___lam__0(lean_object* v_toNatCast_903_, lean_object* v_n_904_){
_start:
{
lean_object* v___x_905_; 
v___x_905_ = lean_apply_1(v_toNatCast_903_, v_n_904_);
return v___x_905_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_directSumGSemiring___redArg___lam__1(lean_object* v_toNPow_906_, lean_object* v_n_907_, lean_object* v_x_908_, lean_object* v_a_909_){
_start:
{
lean_object* v___x_910_; 
v___x_910_ = lean_apply_2(v_toNPow_906_, v_n_907_, v_a_909_);
return v___x_910_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_directSumGSemiring___redArg___lam__1___boxed(lean_object* v_toNPow_911_, lean_object* v_n_912_, lean_object* v_x_913_, lean_object* v_a_914_){
_start:
{
lean_object* v_res_915_; 
v_res_915_ = lp_mathlib_Semiring_directSumGSemiring___redArg___lam__1(v_toNPow_911_, v_n_912_, v_x_913_, v_a_914_);
lean_dec(v_x_913_);
return v_res_915_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_directSumGSemiring___redArg(lean_object* v_inst_916_){
_start:
{
lean_object* v___x_917_; lean_object* v_toNonUnitalNonAssocSemiring_918_; lean_object* v_toMonoid_919_; lean_object* v___x_920_; lean_object* v_toGOne_921_; lean_object* v___x_922_; lean_object* v_toNatCast_923_; lean_object* v_toNPow_924_; lean_object* v___x_925_; lean_object* v___f_926_; lean_object* v___f_927_; lean_object* v___x_928_; 
lean_inc_ref(v_inst_916_);
v___x_917_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_916_);
v_toNonUnitalNonAssocSemiring_918_ = lean_ctor_get(v___x_917_, 0);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_918_);
v_toMonoid_919_ = lean_ctor_get(v_inst_916_, 1);
lean_inc_ref_n(v_toMonoid_919_, 2);
lean_dec_ref(v_inst_916_);
v___x_920_ = lp_mathlib_Monoid_gMonoid___redArg(v_toMonoid_919_);
v_toGOne_921_ = lean_ctor_get(v___x_920_, 1);
lean_inc(v_toGOne_921_);
lean_dec_ref(v___x_920_);
v___x_922_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_917_);
v_toNatCast_923_ = lean_ctor_get(v___x_922_, 0);
lean_inc(v_toNatCast_923_);
lean_dec_ref(v___x_922_);
v_toNPow_924_ = lean_ctor_get(v_toMonoid_919_, 2);
lean_inc(v_toNPow_924_);
lean_dec_ref(v_toMonoid_919_);
v___x_925_ = lp_mathlib_NonUnitalNonAssocSemiring_directSumGNonUnitalNonAssocSemiring___redArg(v_toNonUnitalNonAssocSemiring_918_);
v___f_926_ = lean_alloc_closure((void*)(lp_mathlib_Semiring_directSumGSemiring___redArg___lam__0), 2, 1);
lean_closure_set(v___f_926_, 0, v_toNatCast_923_);
v___f_927_ = lean_alloc_closure((void*)(lp_mathlib_Semiring_directSumGSemiring___redArg___lam__1___boxed), 4, 1);
lean_closure_set(v___f_927_, 0, v_toNPow_924_);
v___x_928_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_928_, 0, v___x_925_);
lean_ctor_set(v___x_928_, 1, v_toGOne_921_);
lean_ctor_set(v___x_928_, 2, v___f_927_);
lean_ctor_set(v___x_928_, 3, v___f_926_);
return v___x_928_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_directSumGSemiring(lean_object* v_00_u03b9_929_, lean_object* v_R_930_, lean_object* v_inst_931_, lean_object* v_inst_932_){
_start:
{
lean_object* v___x_933_; 
v___x_933_ = lp_mathlib_Semiring_directSumGSemiring___redArg(v_inst_932_);
return v___x_933_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semiring_directSumGSemiring___boxed(lean_object* v_00_u03b9_934_, lean_object* v_R_935_, lean_object* v_inst_936_, lean_object* v_inst_937_){
_start:
{
lean_object* v_res_938_; 
v_res_938_ = lp_mathlib_Semiring_directSumGSemiring(v_00_u03b9_934_, v_R_935_, v_inst_936_, v_inst_937_);
lean_dec_ref(v_inst_936_);
return v_res_938_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_directSumGRing___redArg___lam__0(lean_object* v_toIntCast_939_, lean_object* v_z_940_){
_start:
{
lean_object* v___x_941_; 
v___x_941_ = lean_apply_1(v_toIntCast_939_, v_z_940_);
return v___x_941_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_directSumGRing___redArg(lean_object* v_inst_942_){
_start:
{
lean_object* v_toSemiring_943_; lean_object* v___x_944_; lean_object* v___x_945_; lean_object* v_toIntCast_946_; lean_object* v___f_947_; lean_object* v___x_948_; 
v_toSemiring_943_ = lean_ctor_get(v_inst_942_, 0);
lean_inc_ref(v_toSemiring_943_);
v___x_944_ = lp_mathlib_Semiring_directSumGSemiring___redArg(v_toSemiring_943_);
v___x_945_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_inst_942_);
v_toIntCast_946_ = lean_ctor_get(v___x_945_, 0);
lean_inc(v_toIntCast_946_);
lean_dec_ref(v___x_945_);
v___f_947_ = lean_alloc_closure((void*)(lp_mathlib_Ring_directSumGRing___redArg___lam__0), 2, 1);
lean_closure_set(v___f_947_, 0, v_toIntCast_946_);
v___x_948_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_948_, 0, v___x_944_);
lean_ctor_set(v___x_948_, 1, v___f_947_);
return v___x_948_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_directSumGRing(lean_object* v_00_u03b9_949_, lean_object* v_R_950_, lean_object* v_inst_951_, lean_object* v_inst_952_){
_start:
{
lean_object* v___x_953_; 
v___x_953_ = lp_mathlib_Ring_directSumGRing___redArg(v_inst_952_);
return v___x_953_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ring_directSumGRing___boxed(lean_object* v_00_u03b9_954_, lean_object* v_R_955_, lean_object* v_inst_956_, lean_object* v_inst_957_){
_start:
{
lean_object* v_res_958_; 
v_res_958_ = lp_mathlib_Ring_directSumGRing(v_00_u03b9_954_, v_R_955_, v_inst_956_, v_inst_957_);
lean_dec_ref(v_inst_956_);
return v_res_958_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_directSumGCommSemiring___redArg(lean_object* v_inst_959_){
_start:
{
lean_object* v___x_960_; 
v___x_960_ = lp_mathlib_Semiring_directSumGSemiring___redArg(v_inst_959_);
return v___x_960_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_directSumGCommSemiring(lean_object* v_00_u03b9_961_, lean_object* v_R_962_, lean_object* v_inst_963_, lean_object* v_inst_964_){
_start:
{
lean_object* v___x_965_; 
v___x_965_ = lp_mathlib_Semiring_directSumGSemiring___redArg(v_inst_964_);
return v___x_965_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommSemiring_directSumGCommSemiring___boxed(lean_object* v_00_u03b9_966_, lean_object* v_R_967_, lean_object* v_inst_968_, lean_object* v_inst_969_){
_start:
{
lean_object* v_res_970_; 
v_res_970_ = lp_mathlib_CommSemiring_directSumGCommSemiring(v_00_u03b9_966_, v_R_967_, v_inst_968_, v_inst_969_);
lean_dec_ref(v_inst_968_);
return v_res_970_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_directSumGCommRing___redArg(lean_object* v_inst_971_){
_start:
{
lean_object* v___x_972_; 
v___x_972_ = lp_mathlib_Ring_directSumGRing___redArg(v_inst_971_);
return v___x_972_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_directSumGCommRing(lean_object* v_00_u03b9_973_, lean_object* v_R_974_, lean_object* v_inst_975_, lean_object* v_inst_976_){
_start:
{
lean_object* v___x_977_; 
v___x_977_ = lp_mathlib_Ring_directSumGRing___redArg(v_inst_976_);
return v___x_977_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRing_directSumGCommRing___boxed(lean_object* v_00_u03b9_978_, lean_object* v_R_979_, lean_object* v_inst_980_, lean_object* v_inst_981_){
_start:
{
lean_object* v_res_982_; 
v_res_982_ = lp_mathlib_CommRing_directSumGCommRing(v_00_u03b9_978_, v_R_979_, v_inst_980_, v_inst_981_);
lean_dec_ref(v_inst_980_);
return v_res_982_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GradedMonoid(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Associator(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Ring(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GradedMonoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Associator(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_DirectSum_Ring(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_GradedMonoid(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_DirectSum_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Associator(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_DirectSum_Ring(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GradedMonoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_DirectSum_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Associator(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_DirectSum_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_DirectSum_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_DirectSum_Ring(builtin);
}
#ifdef __cplusplus
}
#endif
