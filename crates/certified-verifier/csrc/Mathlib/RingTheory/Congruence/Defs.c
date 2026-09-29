// Lean compiler output
// Module: Mathlib.RingTheory.Congruence.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Ring.Hom.Defs public import Mathlib.Algebra.Ring.InjSurj public import Mathlib.GroupTheory.Congruence.Defs public import Mathlib.Tactic.FastInstance
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
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(lean_object*);
lean_object* lp_mathlib_ZSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_NSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocRing_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_instDistribOfSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_NPow_ofPow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_AddGroupWithOne_toAddGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_toAddCon___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_toAddCon(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_toAddCon___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ringConGen(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ringConGen___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instInhabited___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_comap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_comap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_toQuotient___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_toQuotient___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_toQuotient(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_toQuotient___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCoeTCQuotient___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCoeTCQuotient(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_RingCon_instDecidableEqQuotientOfDecidableCoeForallProp___aux__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instDecidableEqQuotientOfDecidableCoeForallProp___aux__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_RingCon_instDecidableEqQuotientOfDecidableCoeForallProp___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instDecidableEqQuotientOfDecidableCoeForallProp___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_RingCon_instDecidableEqQuotientOfDecidableCoeForallProp___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instDecidableEqQuotientOfDecidableCoeForallProp___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_RingCon_instDecidableEqQuotientOfDecidableCoeForallProp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instDecidableEqQuotientOfDecidableCoeForallProp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddQuotient___aux__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddQuotient___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddQuotient___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddQuotient___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddQuotient(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulQuotient___aux__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulQuotient___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulQuotient___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulQuotient___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulQuotient(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instZeroQuotient___aux__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instZeroQuotient___aux__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instZeroQuotient___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instZeroQuotient___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instZeroQuotient(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instZeroQuotient___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instOneQuotient___aux__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instOneQuotient___aux__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instOneQuotient___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instOneQuotient___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instOneQuotient(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instOneQuotient___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_smulAux___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_smulAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_smulAux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNegQuotient___aux__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNegQuotient___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNegQuotient___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNegQuotient___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNegQuotient(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instSubQuotient___aux__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instSubQuotient___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instSubQuotient___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instSubQuotient___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instSubQuotient(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_hasZSMul___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_hasZSMul(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_hasNSMul___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_hasNSMul(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instPowQuotientNat___aux__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instPowQuotientNat___aux__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instPowQuotientNat___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instPowQuotientNat___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instPowQuotientNat(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNatCastQuotient___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNatCastQuotient___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNatCastQuotient(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNatCastQuotient___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instIntCastQuotient___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instIntCastQuotient___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instIntCastQuotient(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instIntCastQuotient___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instInhabitedQuotient___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instInhabitedQuotient___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instInhabitedQuotient(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instInhabitedQuotient___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddZeroClassQuotient___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddZeroClassQuotient(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddSemigroupQuotient___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddSemigroupQuotient(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddCommMagmaQuotient___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddCommMagmaQuotient(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddCommSemigroupQuotient___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddCommSemigroupQuotient(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddMonoidQuotient___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddMonoidQuotient___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddMonoidQuotient(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddCommMonoidQuotient___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddCommMonoidQuotient(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddGroupQuotient___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddGroupQuotient___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddGroupQuotient(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddCommGroupQuotient___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddCommGroupQuotient(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulOneClassQuotient___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulOneClassQuotient(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instSemigroupQuotient___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instSemigroupQuotient(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCommMagmaQuotient___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCommMagmaQuotient(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCommSemigroupQuotient___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCommSemigroupQuotient(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMonoidQuotient___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMonoidQuotient(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCommMonoidQuotient___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCommMonoidQuotient(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalNonAssocSemiringQuotient___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalNonAssocSemiringQuotient(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalNonAssocCommSemiringQuotient___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalNonAssocCommSemiringQuotient(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonAssocSemiringQuotient___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonAssocSemiringQuotient(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonAssocCommSemiringQuotient___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonAssocCommSemiringQuotient(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalSemiringQuotient___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalSemiringQuotient(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalCommSemiringQuotient___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalCommSemiringQuotient(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instSemiringQuotient___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instSemiringQuotient(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCommSemiringQuotient___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCommSemiringQuotient(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalNonAssocRingQuotient___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalNonAssocRingQuotient(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalNonAssocCommRingQuotient___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalNonAssocCommRingQuotient(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonAssocRingQuotient___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonAssocRingQuotient(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonAssocCommRingQuotient___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonAssocCommRingQuotient(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalRingQuotient___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalRingQuotient(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalCommRingQuotient___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalCommRingQuotient(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instRingQuotient___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instRingQuotient(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCommRingQuotient___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCommRingQuotient(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_mk_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_mk_x27(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingCon_toAddCon___redArg(lean_object* v_self_1_){
_start:
{
return v_self_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_toAddCon(lean_object* v_R_2_, lean_object* v_inst_3_, lean_object* v_inst_4_, lean_object* v_self_5_){
_start:
{
return v_self_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_toAddCon___boxed(lean_object* v_R_6_, lean_object* v_inst_7_, lean_object* v_inst_8_, lean_object* v_self_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_mathlib_RingCon_toAddCon(v_R_6_, v_inst_7_, v_inst_8_, v_self_9_);
lean_dec(v_inst_8_);
lean_dec(v_inst_7_);
return v_res_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ringConGen(lean_object* v_R_11_, lean_object* v_inst_12_, lean_object* v_inst_13_, lean_object* v_r_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lean_box(0);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ringConGen___boxed(lean_object* v_R_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_r_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_mathlib_ringConGen(v_R_16_, v_inst_17_, v_inst_18_, v_r_19_);
lean_dec(v_inst_18_);
lean_dec(v_inst_17_);
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instInhabited(lean_object* v_R_21_, lean_object* v_inst_22_, lean_object* v_inst_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lean_box(0);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instInhabited___boxed(lean_object* v_R_25_, lean_object* v_inst_26_, lean_object* v_inst_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_mathlib_RingCon_instInhabited(v_R_25_, v_inst_26_, v_inst_27_);
lean_dec(v_inst_27_);
lean_dec(v_inst_26_);
return v_res_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_comap(lean_object* v_R_29_, lean_object* v_R_x27_30_, lean_object* v_F_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_inst_34_, lean_object* v_inst_35_, lean_object* v_inst_36_, lean_object* v_inst_37_, lean_object* v_inst_38_, lean_object* v_J_39_, lean_object* v_f_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lean_box(0);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_comap___boxed(lean_object* v_R_42_, lean_object* v_R_x27_43_, lean_object* v_F_44_, lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_inst_51_, lean_object* v_J_52_, lean_object* v_f_53_){
_start:
{
lean_object* v_res_54_; 
v_res_54_ = lp_mathlib_RingCon_comap(v_R_42_, v_R_x27_43_, v_F_44_, v_inst_45_, v_inst_46_, v_inst_47_, v_inst_48_, v_inst_49_, v_inst_50_, v_inst_51_, v_J_52_, v_f_53_);
lean_dec(v_f_53_);
lean_dec(v_inst_50_);
lean_dec(v_inst_49_);
lean_dec(v_inst_47_);
lean_dec(v_inst_46_);
lean_dec(v_inst_45_);
return v_res_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_toQuotient___redArg(lean_object* v_r_55_){
_start:
{
lean_inc(v_r_55_);
return v_r_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_toQuotient___redArg___boxed(lean_object* v_r_56_){
_start:
{
lean_object* v_res_57_; 
v_res_57_ = lp_mathlib_RingCon_toQuotient___redArg(v_r_56_);
lean_dec(v_r_56_);
return v_res_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_toQuotient(lean_object* v_R_58_, lean_object* v_inst_59_, lean_object* v_inst_60_, lean_object* v_c_61_, lean_object* v_r_62_){
_start:
{
lean_inc(v_r_62_);
return v_r_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_toQuotient___boxed(lean_object* v_R_63_, lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_c_66_, lean_object* v_r_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_mathlib_RingCon_toQuotient(v_R_63_, v_inst_64_, v_inst_65_, v_c_66_, v_r_67_);
lean_dec(v_r_67_);
lean_dec(v_inst_65_);
lean_dec(v_inst_64_);
return v_res_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCoeTCQuotient___redArg(lean_object* v_inst_69_, lean_object* v_inst_70_, lean_object* v_c_71_){
_start:
{
lean_object* v___x_72_; 
v___x_72_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_toQuotient___boxed), 5, 4);
lean_closure_set(v___x_72_, 0, lean_box(0));
lean_closure_set(v___x_72_, 1, v_inst_69_);
lean_closure_set(v___x_72_, 2, v_inst_70_);
lean_closure_set(v___x_72_, 3, v_c_71_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCoeTCQuotient(lean_object* v_R_73_, lean_object* v_inst_74_, lean_object* v_inst_75_, lean_object* v_c_76_){
_start:
{
lean_object* v___x_77_; 
v___x_77_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_toQuotient___boxed), 5, 4);
lean_closure_set(v___x_77_, 0, lean_box(0));
lean_closure_set(v___x_77_, 1, v_inst_74_);
lean_closure_set(v___x_77_, 2, v_inst_75_);
lean_closure_set(v___x_77_, 3, v_c_76_);
return v___x_77_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_RingCon_instDecidableEqQuotientOfDecidableCoeForallProp___aux__1___redArg(lean_object* v___d_78_, lean_object* v_a_79_, lean_object* v_b_80_){
_start:
{
lean_object* v___x_81_; uint8_t v___x_82_; 
v___x_81_ = lean_apply_2(v___d_78_, v_a_79_, v_b_80_);
v___x_82_ = lean_unbox(v___x_81_);
return v___x_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instDecidableEqQuotientOfDecidableCoeForallProp___aux__1___redArg___boxed(lean_object* v___d_83_, lean_object* v_a_84_, lean_object* v_b_85_){
_start:
{
uint8_t v_res_86_; lean_object* v_r_87_; 
v_res_86_ = lp_mathlib_RingCon_instDecidableEqQuotientOfDecidableCoeForallProp___aux__1___redArg(v___d_83_, v_a_84_, v_b_85_);
v_r_87_ = lean_box(v_res_86_);
return v_r_87_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_RingCon_instDecidableEqQuotientOfDecidableCoeForallProp___aux__1(lean_object* v_R_88_, lean_object* v_inst_89_, lean_object* v_inst_90_, lean_object* v_c_91_, lean_object* v___d_92_, lean_object* v_a_93_, lean_object* v_b_94_){
_start:
{
lean_object* v___x_95_; uint8_t v___x_96_; 
v___x_95_ = lean_apply_2(v___d_92_, v_a_93_, v_b_94_);
v___x_96_ = lean_unbox(v___x_95_);
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instDecidableEqQuotientOfDecidableCoeForallProp___aux__1___boxed(lean_object* v_R_97_, lean_object* v_inst_98_, lean_object* v_inst_99_, lean_object* v_c_100_, lean_object* v___d_101_, lean_object* v_a_102_, lean_object* v_b_103_){
_start:
{
uint8_t v_res_104_; lean_object* v_r_105_; 
v_res_104_ = lp_mathlib_RingCon_instDecidableEqQuotientOfDecidableCoeForallProp___aux__1(v_R_97_, v_inst_98_, v_inst_99_, v_c_100_, v___d_101_, v_a_102_, v_b_103_);
lean_dec(v_inst_99_);
lean_dec(v_inst_98_);
v_r_105_ = lean_box(v_res_104_);
return v_r_105_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_RingCon_instDecidableEqQuotientOfDecidableCoeForallProp___redArg(lean_object* v___d_106_, lean_object* v_a_107_, lean_object* v_b_108_){
_start:
{
lean_object* v___x_109_; uint8_t v___x_110_; 
v___x_109_ = lean_apply_2(v___d_106_, v_a_107_, v_b_108_);
v___x_110_ = lean_unbox(v___x_109_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instDecidableEqQuotientOfDecidableCoeForallProp___redArg___boxed(lean_object* v___d_111_, lean_object* v_a_112_, lean_object* v_b_113_){
_start:
{
uint8_t v_res_114_; lean_object* v_r_115_; 
v_res_114_ = lp_mathlib_RingCon_instDecidableEqQuotientOfDecidableCoeForallProp___redArg(v___d_111_, v_a_112_, v_b_113_);
v_r_115_ = lean_box(v_res_114_);
return v_r_115_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_RingCon_instDecidableEqQuotientOfDecidableCoeForallProp(lean_object* v_R_116_, lean_object* v_inst_117_, lean_object* v_inst_118_, lean_object* v_c_119_, lean_object* v___d_120_, lean_object* v_a_121_, lean_object* v_b_122_){
_start:
{
lean_object* v___x_123_; uint8_t v___x_124_; 
v___x_123_ = lean_apply_2(v___d_120_, v_a_121_, v_b_122_);
v___x_124_ = lean_unbox(v___x_123_);
return v___x_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instDecidableEqQuotientOfDecidableCoeForallProp___boxed(lean_object* v_R_125_, lean_object* v_inst_126_, lean_object* v_inst_127_, lean_object* v_c_128_, lean_object* v___d_129_, lean_object* v_a_130_, lean_object* v_b_131_){
_start:
{
uint8_t v_res_132_; lean_object* v_r_133_; 
v_res_132_ = lp_mathlib_RingCon_instDecidableEqQuotientOfDecidableCoeForallProp(v_R_125_, v_inst_126_, v_inst_127_, v_c_128_, v___d_129_, v_a_130_, v_b_131_);
lean_dec(v_inst_127_);
lean_dec(v_inst_126_);
v_r_133_ = lean_box(v_res_132_);
return v_r_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddQuotient___aux__1___redArg(lean_object* v_inst_134_, lean_object* v_a_135_, lean_object* v_a_136_){
_start:
{
lean_object* v___x_137_; 
v___x_137_ = lean_apply_2(v_inst_134_, v_a_135_, v_a_136_);
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddQuotient___aux__1(lean_object* v_R_138_, lean_object* v_inst_139_, lean_object* v_inst_140_, lean_object* v_c_141_, lean_object* v_a_142_, lean_object* v_a_143_){
_start:
{
lean_object* v___x_144_; 
v___x_144_ = lean_apply_2(v_inst_139_, v_a_142_, v_a_143_);
return v___x_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddQuotient___aux__1___boxed(lean_object* v_R_145_, lean_object* v_inst_146_, lean_object* v_inst_147_, lean_object* v_c_148_, lean_object* v_a_149_, lean_object* v_a_150_){
_start:
{
lean_object* v_res_151_; 
v_res_151_ = lp_mathlib_RingCon_instAddQuotient___aux__1(v_R_145_, v_inst_146_, v_inst_147_, v_c_148_, v_a_149_, v_a_150_);
lean_dec(v_inst_147_);
return v_res_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddQuotient___redArg(lean_object* v_inst_152_, lean_object* v_inst_153_, lean_object* v_c_154_){
_start:
{
lean_object* v___x_155_; 
v___x_155_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instAddQuotient___aux__1___boxed), 6, 4);
lean_closure_set(v___x_155_, 0, lean_box(0));
lean_closure_set(v___x_155_, 1, v_inst_152_);
lean_closure_set(v___x_155_, 2, v_inst_153_);
lean_closure_set(v___x_155_, 3, v_c_154_);
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddQuotient(lean_object* v_R_156_, lean_object* v_inst_157_, lean_object* v_inst_158_, lean_object* v_c_159_){
_start:
{
lean_object* v___x_160_; 
v___x_160_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instAddQuotient___aux__1___boxed), 6, 4);
lean_closure_set(v___x_160_, 0, lean_box(0));
lean_closure_set(v___x_160_, 1, v_inst_157_);
lean_closure_set(v___x_160_, 2, v_inst_158_);
lean_closure_set(v___x_160_, 3, v_c_159_);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulQuotient___aux__1___redArg(lean_object* v_inst_161_, lean_object* v_a_162_, lean_object* v_a_163_){
_start:
{
lean_object* v___x_164_; 
v___x_164_ = lean_apply_2(v_inst_161_, v_a_162_, v_a_163_);
return v___x_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulQuotient___aux__1(lean_object* v_R_165_, lean_object* v_inst_166_, lean_object* v_inst_167_, lean_object* v_c_168_, lean_object* v_a_169_, lean_object* v_a_170_){
_start:
{
lean_object* v___x_171_; 
v___x_171_ = lean_apply_2(v_inst_167_, v_a_169_, v_a_170_);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulQuotient___aux__1___boxed(lean_object* v_R_172_, lean_object* v_inst_173_, lean_object* v_inst_174_, lean_object* v_c_175_, lean_object* v_a_176_, lean_object* v_a_177_){
_start:
{
lean_object* v_res_178_; 
v_res_178_ = lp_mathlib_RingCon_instMulQuotient___aux__1(v_R_172_, v_inst_173_, v_inst_174_, v_c_175_, v_a_176_, v_a_177_);
lean_dec(v_inst_173_);
return v_res_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulQuotient___redArg(lean_object* v_inst_179_, lean_object* v_inst_180_, lean_object* v_c_181_){
_start:
{
lean_object* v___x_182_; 
v___x_182_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instMulQuotient___aux__1___boxed), 6, 4);
lean_closure_set(v___x_182_, 0, lean_box(0));
lean_closure_set(v___x_182_, 1, v_inst_179_);
lean_closure_set(v___x_182_, 2, v_inst_180_);
lean_closure_set(v___x_182_, 3, v_c_181_);
return v___x_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulQuotient(lean_object* v_R_183_, lean_object* v_inst_184_, lean_object* v_inst_185_, lean_object* v_c_186_){
_start:
{
lean_object* v___x_187_; 
v___x_187_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instMulQuotient___aux__1___boxed), 6, 4);
lean_closure_set(v___x_187_, 0, lean_box(0));
lean_closure_set(v___x_187_, 1, v_inst_184_);
lean_closure_set(v___x_187_, 2, v_inst_185_);
lean_closure_set(v___x_187_, 3, v_c_186_);
return v___x_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instZeroQuotient___aux__1___redArg(lean_object* v_inst_188_){
_start:
{
lean_object* v___x_189_; lean_object* v_toZero_190_; 
v___x_189_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_188_);
v_toZero_190_ = lean_ctor_get(v___x_189_, 0);
lean_inc(v_toZero_190_);
lean_dec_ref(v___x_189_);
return v_toZero_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instZeroQuotient___aux__1(lean_object* v_R_191_, lean_object* v_inst_192_, lean_object* v_inst_193_, lean_object* v_c_194_){
_start:
{
lean_object* v___x_195_; lean_object* v_toZero_196_; 
v___x_195_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_192_);
v_toZero_196_ = lean_ctor_get(v___x_195_, 0);
lean_inc(v_toZero_196_);
lean_dec_ref(v___x_195_);
return v_toZero_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instZeroQuotient___aux__1___boxed(lean_object* v_R_197_, lean_object* v_inst_198_, lean_object* v_inst_199_, lean_object* v_c_200_){
_start:
{
lean_object* v_res_201_; 
v_res_201_ = lp_mathlib_RingCon_instZeroQuotient___aux__1(v_R_197_, v_inst_198_, v_inst_199_, v_c_200_);
lean_dec(v_inst_199_);
return v_res_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instZeroQuotient___redArg(lean_object* v_inst_202_){
_start:
{
lean_object* v___x_203_; lean_object* v_toZero_204_; 
v___x_203_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_202_);
v_toZero_204_ = lean_ctor_get(v___x_203_, 0);
lean_inc(v_toZero_204_);
lean_dec_ref(v___x_203_);
return v_toZero_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instZeroQuotient(lean_object* v_R_205_, lean_object* v_inst_206_, lean_object* v_inst_207_, lean_object* v_c_208_){
_start:
{
lean_object* v___x_209_; 
v___x_209_ = lp_mathlib_RingCon_instZeroQuotient___redArg(v_inst_206_);
return v___x_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instZeroQuotient___boxed(lean_object* v_R_210_, lean_object* v_inst_211_, lean_object* v_inst_212_, lean_object* v_c_213_){
_start:
{
lean_object* v_res_214_; 
v_res_214_ = lp_mathlib_RingCon_instZeroQuotient(v_R_210_, v_inst_211_, v_inst_212_, v_c_213_);
lean_dec(v_inst_212_);
return v_res_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instOneQuotient___aux__1___redArg(lean_object* v_inst_215_){
_start:
{
lean_object* v___x_216_; lean_object* v_toOne_217_; 
v___x_216_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_215_);
v_toOne_217_ = lean_ctor_get(v___x_216_, 0);
lean_inc(v_toOne_217_);
lean_dec_ref(v___x_216_);
return v_toOne_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instOneQuotient___aux__1(lean_object* v_R_218_, lean_object* v_inst_219_, lean_object* v_inst_220_, lean_object* v_c_221_){
_start:
{
lean_object* v___x_222_; lean_object* v_toOne_223_; 
v___x_222_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_220_);
v_toOne_223_ = lean_ctor_get(v___x_222_, 0);
lean_inc(v_toOne_223_);
lean_dec_ref(v___x_222_);
return v_toOne_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instOneQuotient___aux__1___boxed(lean_object* v_R_224_, lean_object* v_inst_225_, lean_object* v_inst_226_, lean_object* v_c_227_){
_start:
{
lean_object* v_res_228_; 
v_res_228_ = lp_mathlib_RingCon_instOneQuotient___aux__1(v_R_224_, v_inst_225_, v_inst_226_, v_c_227_);
lean_dec(v_inst_225_);
return v_res_228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instOneQuotient___redArg(lean_object* v_inst_229_){
_start:
{
lean_object* v___x_230_; lean_object* v_toOne_231_; 
v___x_230_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_229_);
v_toOne_231_ = lean_ctor_get(v___x_230_, 0);
lean_inc(v_toOne_231_);
lean_dec_ref(v___x_230_);
return v_toOne_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instOneQuotient(lean_object* v_R_232_, lean_object* v_inst_233_, lean_object* v_inst_234_, lean_object* v_c_235_){
_start:
{
lean_object* v___x_236_; 
v___x_236_ = lp_mathlib_RingCon_instOneQuotient___redArg(v_inst_234_);
return v___x_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instOneQuotient___boxed(lean_object* v_R_237_, lean_object* v_inst_238_, lean_object* v_inst_239_, lean_object* v_c_240_){
_start:
{
lean_object* v_res_241_; 
v_res_241_ = lp_mathlib_RingCon_instOneQuotient(v_R_237_, v_inst_238_, v_inst_239_, v_c_240_);
lean_dec(v_inst_238_);
return v_res_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_smulAux___redArg(lean_object* v_inst_242_, lean_object* v_a_243_, lean_object* v_x_244_){
_start:
{
lean_object* v___x_245_; 
v___x_245_ = lean_apply_2(v_inst_242_, v_a_243_, v_x_244_);
return v___x_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_smulAux(lean_object* v_R_246_, lean_object* v_inst_247_, lean_object* v_inst_248_, lean_object* v_00_u03b1_249_, lean_object* v_inst_250_, lean_object* v_c_251_, lean_object* v_h_252_, lean_object* v_a_253_, lean_object* v_x_254_){
_start:
{
lean_object* v___x_255_; 
v___x_255_ = lean_apply_2(v_inst_250_, v_a_253_, v_x_254_);
return v___x_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_smulAux___boxed(lean_object* v_R_256_, lean_object* v_inst_257_, lean_object* v_inst_258_, lean_object* v_00_u03b1_259_, lean_object* v_inst_260_, lean_object* v_c_261_, lean_object* v_h_262_, lean_object* v_a_263_, lean_object* v_x_264_){
_start:
{
lean_object* v_res_265_; 
v_res_265_ = lp_mathlib_RingCon_smulAux(v_R_256_, v_inst_257_, v_inst_258_, v_00_u03b1_259_, v_inst_260_, v_c_261_, v_h_262_, v_a_263_, v_x_264_);
lean_dec(v_inst_258_);
lean_dec(v_inst_257_);
return v_res_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNegQuotient___aux__1___redArg(lean_object* v_inst_266_, lean_object* v_a_267_){
_start:
{
lean_object* v_toNeg_268_; lean_object* v___x_269_; 
v_toNeg_268_ = lean_ctor_get(v_inst_266_, 1);
lean_inc(v_toNeg_268_);
lean_dec_ref(v_inst_266_);
v___x_269_ = lean_apply_1(v_toNeg_268_, v_a_267_);
return v___x_269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNegQuotient___aux__1(lean_object* v_R_270_, lean_object* v_inst_271_, lean_object* v_inst_272_, lean_object* v_c_273_, lean_object* v_a_274_){
_start:
{
lean_object* v_toNeg_275_; lean_object* v___x_276_; 
v_toNeg_275_ = lean_ctor_get(v_inst_271_, 1);
lean_inc(v_toNeg_275_);
lean_dec_ref(v_inst_271_);
v___x_276_ = lean_apply_1(v_toNeg_275_, v_a_274_);
return v___x_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNegQuotient___aux__1___boxed(lean_object* v_R_277_, lean_object* v_inst_278_, lean_object* v_inst_279_, lean_object* v_c_280_, lean_object* v_a_281_){
_start:
{
lean_object* v_res_282_; 
v_res_282_ = lp_mathlib_RingCon_instNegQuotient___aux__1(v_R_277_, v_inst_278_, v_inst_279_, v_c_280_, v_a_281_);
lean_dec(v_inst_279_);
return v_res_282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNegQuotient___redArg(lean_object* v_inst_283_, lean_object* v_inst_284_, lean_object* v_c_285_){
_start:
{
lean_object* v___x_286_; 
v___x_286_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instNegQuotient___aux__1___boxed), 5, 4);
lean_closure_set(v___x_286_, 0, lean_box(0));
lean_closure_set(v___x_286_, 1, v_inst_283_);
lean_closure_set(v___x_286_, 2, v_inst_284_);
lean_closure_set(v___x_286_, 3, v_c_285_);
return v___x_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNegQuotient(lean_object* v_R_287_, lean_object* v_inst_288_, lean_object* v_inst_289_, lean_object* v_c_290_){
_start:
{
lean_object* v___x_291_; 
v___x_291_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instNegQuotient___aux__1___boxed), 5, 4);
lean_closure_set(v___x_291_, 0, lean_box(0));
lean_closure_set(v___x_291_, 1, v_inst_288_);
lean_closure_set(v___x_291_, 2, v_inst_289_);
lean_closure_set(v___x_291_, 3, v_c_290_);
return v___x_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instSubQuotient___aux__1___redArg(lean_object* v_inst_292_, lean_object* v_a_293_, lean_object* v_a_294_){
_start:
{
lean_object* v_toSub_295_; lean_object* v___x_296_; 
v_toSub_295_ = lean_ctor_get(v_inst_292_, 2);
lean_inc(v_toSub_295_);
lean_dec_ref(v_inst_292_);
v___x_296_ = lean_apply_2(v_toSub_295_, v_a_293_, v_a_294_);
return v___x_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instSubQuotient___aux__1(lean_object* v_R_297_, lean_object* v_inst_298_, lean_object* v_inst_299_, lean_object* v_c_300_, lean_object* v_a_301_, lean_object* v_a_302_){
_start:
{
lean_object* v_toSub_303_; lean_object* v___x_304_; 
v_toSub_303_ = lean_ctor_get(v_inst_298_, 2);
lean_inc(v_toSub_303_);
lean_dec_ref(v_inst_298_);
v___x_304_ = lean_apply_2(v_toSub_303_, v_a_301_, v_a_302_);
return v___x_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instSubQuotient___aux__1___boxed(lean_object* v_R_305_, lean_object* v_inst_306_, lean_object* v_inst_307_, lean_object* v_c_308_, lean_object* v_a_309_, lean_object* v_a_310_){
_start:
{
lean_object* v_res_311_; 
v_res_311_ = lp_mathlib_RingCon_instSubQuotient___aux__1(v_R_305_, v_inst_306_, v_inst_307_, v_c_308_, v_a_309_, v_a_310_);
lean_dec(v_inst_307_);
return v_res_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instSubQuotient___redArg(lean_object* v_inst_312_, lean_object* v_inst_313_, lean_object* v_c_314_){
_start:
{
lean_object* v___x_315_; 
v___x_315_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instSubQuotient___aux__1___boxed), 6, 4);
lean_closure_set(v___x_315_, 0, lean_box(0));
lean_closure_set(v___x_315_, 1, v_inst_312_);
lean_closure_set(v___x_315_, 2, v_inst_313_);
lean_closure_set(v___x_315_, 3, v_c_314_);
return v___x_315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instSubQuotient(lean_object* v_R_316_, lean_object* v_inst_317_, lean_object* v_inst_318_, lean_object* v_c_319_){
_start:
{
lean_object* v___x_320_; 
v___x_320_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instSubQuotient___aux__1___boxed), 6, 4);
lean_closure_set(v___x_320_, 0, lean_box(0));
lean_closure_set(v___x_320_, 1, v_inst_317_);
lean_closure_set(v___x_320_, 2, v_inst_318_);
lean_closure_set(v___x_320_, 3, v_c_319_);
return v___x_320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_hasZSMul___redArg(lean_object* v_inst_321_, lean_object* v_inst_322_, lean_object* v_c_323_){
_start:
{
lean_object* v_toAddMonoid_324_; lean_object* v_toZSMul_325_; lean_object* v_toAdd_326_; lean_object* v___f_327_; lean_object* v___x_328_; 
v_toAddMonoid_324_ = lean_ctor_get(v_inst_321_, 0);
lean_inc_ref(v_toAddMonoid_324_);
v_toZSMul_325_ = lean_ctor_get(v_inst_321_, 3);
lean_inc(v_toZSMul_325_);
lean_dec_ref(v_inst_321_);
v_toAdd_326_ = lean_ctor_get(v_toAddMonoid_324_, 1);
lean_inc(v_toAdd_326_);
lean_dec_ref(v_toAddMonoid_324_);
v___f_327_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_327_, 0, v_toZSMul_325_);
v___x_328_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_smulAux___boxed), 9, 7);
lean_closure_set(v___x_328_, 0, lean_box(0));
lean_closure_set(v___x_328_, 1, v_toAdd_326_);
lean_closure_set(v___x_328_, 2, v_inst_322_);
lean_closure_set(v___x_328_, 3, lean_box(0));
lean_closure_set(v___x_328_, 4, v___f_327_);
lean_closure_set(v___x_328_, 5, v_c_323_);
lean_closure_set(v___x_328_, 6, lean_box(0));
return v___x_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_hasZSMul(lean_object* v_R_329_, lean_object* v_inst_330_, lean_object* v_inst_331_, lean_object* v_c_332_){
_start:
{
lean_object* v___x_333_; 
v___x_333_ = lp_mathlib_RingCon_hasZSMul___redArg(v_inst_330_, v_inst_331_, v_c_332_);
return v___x_333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_hasNSMul___redArg(lean_object* v_inst_334_, lean_object* v_inst_335_, lean_object* v_c_336_){
_start:
{
lean_object* v_toAdd_337_; lean_object* v_toNSMul_338_; lean_object* v___f_339_; lean_object* v___x_340_; 
v_toAdd_337_ = lean_ctor_get(v_inst_334_, 1);
lean_inc(v_toAdd_337_);
v_toNSMul_338_ = lean_ctor_get(v_inst_334_, 2);
lean_inc(v_toNSMul_338_);
lean_dec_ref(v_inst_334_);
v___f_339_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_339_, 0, v_toNSMul_338_);
v___x_340_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_smulAux___boxed), 9, 7);
lean_closure_set(v___x_340_, 0, lean_box(0));
lean_closure_set(v___x_340_, 1, v_toAdd_337_);
lean_closure_set(v___x_340_, 2, v_inst_335_);
lean_closure_set(v___x_340_, 3, lean_box(0));
lean_closure_set(v___x_340_, 4, v___f_339_);
lean_closure_set(v___x_340_, 5, v_c_336_);
lean_closure_set(v___x_340_, 6, lean_box(0));
return v___x_340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_hasNSMul(lean_object* v_R_341_, lean_object* v_inst_342_, lean_object* v_inst_343_, lean_object* v_c_344_){
_start:
{
lean_object* v___x_345_; 
v___x_345_ = lp_mathlib_RingCon_hasNSMul___redArg(v_inst_342_, v_inst_343_, v_c_344_);
return v___x_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instPowQuotientNat___aux__1___redArg(lean_object* v_inst_346_, lean_object* v_x_347_, lean_object* v_n_348_){
_start:
{
lean_object* v_toNPow_349_; lean_object* v___x_350_; 
v_toNPow_349_ = lean_ctor_get(v_inst_346_, 2);
lean_inc(v_toNPow_349_);
lean_dec_ref(v_inst_346_);
v___x_350_ = lean_apply_2(v_toNPow_349_, v_n_348_, v_x_347_);
return v___x_350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instPowQuotientNat___aux__1(lean_object* v_R_351_, lean_object* v_inst_352_, lean_object* v_inst_353_, lean_object* v_c_354_, lean_object* v_x_355_, lean_object* v_n_356_){
_start:
{
lean_object* v_toNPow_357_; lean_object* v___x_358_; 
v_toNPow_357_ = lean_ctor_get(v_inst_353_, 2);
lean_inc(v_toNPow_357_);
lean_dec_ref(v_inst_353_);
v___x_358_ = lean_apply_2(v_toNPow_357_, v_n_356_, v_x_355_);
return v___x_358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instPowQuotientNat___aux__1___boxed(lean_object* v_R_359_, lean_object* v_inst_360_, lean_object* v_inst_361_, lean_object* v_c_362_, lean_object* v_x_363_, lean_object* v_n_364_){
_start:
{
lean_object* v_res_365_; 
v_res_365_ = lp_mathlib_RingCon_instPowQuotientNat___aux__1(v_R_359_, v_inst_360_, v_inst_361_, v_c_362_, v_x_363_, v_n_364_);
lean_dec(v_inst_360_);
return v_res_365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instPowQuotientNat___redArg(lean_object* v_inst_366_, lean_object* v_inst_367_, lean_object* v_c_368_){
_start:
{
lean_object* v___x_369_; 
v___x_369_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instPowQuotientNat___aux__1___boxed), 6, 4);
lean_closure_set(v___x_369_, 0, lean_box(0));
lean_closure_set(v___x_369_, 1, v_inst_366_);
lean_closure_set(v___x_369_, 2, v_inst_367_);
lean_closure_set(v___x_369_, 3, v_c_368_);
return v___x_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instPowQuotientNat(lean_object* v_R_370_, lean_object* v_inst_371_, lean_object* v_inst_372_, lean_object* v_c_373_){
_start:
{
lean_object* v___x_374_; 
v___x_374_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instPowQuotientNat___aux__1___boxed), 6, 4);
lean_closure_set(v___x_374_, 0, lean_box(0));
lean_closure_set(v___x_374_, 1, v_inst_371_);
lean_closure_set(v___x_374_, 2, v_inst_372_);
lean_closure_set(v___x_374_, 3, v_c_373_);
return v___x_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNatCastQuotient___redArg___lam__0(lean_object* v_toNatCast_375_, lean_object* v_n_376_){
_start:
{
lean_object* v___x_377_; 
v___x_377_ = lean_apply_1(v_toNatCast_375_, v_n_376_);
return v___x_377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNatCastQuotient___redArg(lean_object* v_inst_378_){
_start:
{
lean_object* v_toNatCast_379_; lean_object* v___f_380_; 
v_toNatCast_379_ = lean_ctor_get(v_inst_378_, 0);
lean_inc(v_toNatCast_379_);
lean_dec_ref(v_inst_378_);
v___f_380_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instNatCastQuotient___redArg___lam__0), 2, 1);
lean_closure_set(v___f_380_, 0, v_toNatCast_379_);
return v___f_380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNatCastQuotient(lean_object* v_R_381_, lean_object* v_inst_382_, lean_object* v_inst_383_, lean_object* v_c_384_){
_start:
{
lean_object* v___x_385_; 
v___x_385_ = lp_mathlib_RingCon_instNatCastQuotient___redArg(v_inst_382_);
return v___x_385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNatCastQuotient___boxed(lean_object* v_R_386_, lean_object* v_inst_387_, lean_object* v_inst_388_, lean_object* v_c_389_){
_start:
{
lean_object* v_res_390_; 
v_res_390_ = lp_mathlib_RingCon_instNatCastQuotient(v_R_386_, v_inst_387_, v_inst_388_, v_c_389_);
lean_dec(v_inst_388_);
return v_res_390_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instIntCastQuotient___redArg___lam__0(lean_object* v_toIntCast_391_, lean_object* v_z_392_){
_start:
{
lean_object* v___x_393_; 
v___x_393_ = lean_apply_1(v_toIntCast_391_, v_z_392_);
return v___x_393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instIntCastQuotient___redArg(lean_object* v_inst_394_){
_start:
{
lean_object* v_toIntCast_395_; lean_object* v___f_396_; 
v_toIntCast_395_ = lean_ctor_get(v_inst_394_, 0);
lean_inc(v_toIntCast_395_);
lean_dec_ref(v_inst_394_);
v___f_396_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instIntCastQuotient___redArg___lam__0), 2, 1);
lean_closure_set(v___f_396_, 0, v_toIntCast_395_);
return v___f_396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instIntCastQuotient(lean_object* v_R_397_, lean_object* v_inst_398_, lean_object* v_inst_399_, lean_object* v_c_400_){
_start:
{
lean_object* v___x_401_; 
v___x_401_ = lp_mathlib_RingCon_instIntCastQuotient___redArg(v_inst_398_);
return v___x_401_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instIntCastQuotient___boxed(lean_object* v_R_402_, lean_object* v_inst_403_, lean_object* v_inst_404_, lean_object* v_c_405_){
_start:
{
lean_object* v_res_406_; 
v_res_406_ = lp_mathlib_RingCon_instIntCastQuotient(v_R_402_, v_inst_403_, v_inst_404_, v_c_405_);
lean_dec(v_inst_404_);
return v_res_406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instInhabitedQuotient___redArg(lean_object* v_inst_407_){
_start:
{
lean_inc(v_inst_407_);
return v_inst_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instInhabitedQuotient___redArg___boxed(lean_object* v_inst_408_){
_start:
{
lean_object* v_res_409_; 
v_res_409_ = lp_mathlib_RingCon_instInhabitedQuotient___redArg(v_inst_408_);
lean_dec(v_inst_408_);
return v_res_409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instInhabitedQuotient(lean_object* v_R_410_, lean_object* v_inst_411_, lean_object* v_inst_412_, lean_object* v_inst_413_, lean_object* v_c_414_){
_start:
{
lean_inc(v_inst_411_);
return v_inst_411_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instInhabitedQuotient___boxed(lean_object* v_R_415_, lean_object* v_inst_416_, lean_object* v_inst_417_, lean_object* v_inst_418_, lean_object* v_c_419_){
_start:
{
lean_object* v_res_420_; 
v_res_420_ = lp_mathlib_RingCon_instInhabitedQuotient(v_R_415_, v_inst_416_, v_inst_417_, v_inst_418_, v_c_419_);
lean_dec(v_inst_418_);
lean_dec(v_inst_417_);
lean_dec(v_inst_416_);
return v_res_420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddZeroClassQuotient___redArg(lean_object* v_inst_421_, lean_object* v_inst_422_, lean_object* v_c_423_){
_start:
{
lean_object* v___x_424_; lean_object* v_toAdd_425_; lean_object* v___x_427_; uint8_t v_isShared_428_; uint8_t v_isSharedCheck_434_; 
lean_inc_ref(v_inst_421_);
v___x_424_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_421_);
v_toAdd_425_ = lean_ctor_get(v___x_424_, 1);
v_isSharedCheck_434_ = !lean_is_exclusive(v___x_424_);
if (v_isSharedCheck_434_ == 0)
{
lean_object* v_unused_435_; 
v_unused_435_ = lean_ctor_get(v___x_424_, 0);
lean_dec(v_unused_435_);
v___x_427_ = v___x_424_;
v_isShared_428_ = v_isSharedCheck_434_;
goto v_resetjp_426_;
}
else
{
lean_inc(v_toAdd_425_);
lean_dec(v___x_424_);
v___x_427_ = lean_box(0);
v_isShared_428_ = v_isSharedCheck_434_;
goto v_resetjp_426_;
}
v_resetjp_426_:
{
lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_432_; 
v___x_429_ = lp_mathlib_RingCon_instZeroQuotient___redArg(v_inst_421_);
v___x_430_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instAddQuotient___aux__1___boxed), 6, 4);
lean_closure_set(v___x_430_, 0, lean_box(0));
lean_closure_set(v___x_430_, 1, v_toAdd_425_);
lean_closure_set(v___x_430_, 2, v_inst_422_);
lean_closure_set(v___x_430_, 3, v_c_423_);
if (v_isShared_428_ == 0)
{
lean_ctor_set(v___x_427_, 1, v___x_430_);
lean_ctor_set(v___x_427_, 0, v___x_429_);
v___x_432_ = v___x_427_;
goto v_reusejp_431_;
}
else
{
lean_object* v_reuseFailAlloc_433_; 
v_reuseFailAlloc_433_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_433_, 0, v___x_429_);
lean_ctor_set(v_reuseFailAlloc_433_, 1, v___x_430_);
v___x_432_ = v_reuseFailAlloc_433_;
goto v_reusejp_431_;
}
v_reusejp_431_:
{
return v___x_432_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddZeroClassQuotient(lean_object* v_R_436_, lean_object* v_inst_437_, lean_object* v_inst_438_, lean_object* v_c_439_){
_start:
{
lean_object* v___x_440_; 
v___x_440_ = lp_mathlib_RingCon_instAddZeroClassQuotient___redArg(v_inst_437_, v_inst_438_, v_c_439_);
return v___x_440_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddSemigroupQuotient___redArg(lean_object* v_inst_441_, lean_object* v_inst_442_, lean_object* v_c_443_){
_start:
{
lean_object* v___x_444_; 
v___x_444_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instAddQuotient___aux__1___boxed), 6, 4);
lean_closure_set(v___x_444_, 0, lean_box(0));
lean_closure_set(v___x_444_, 1, v_inst_441_);
lean_closure_set(v___x_444_, 2, v_inst_442_);
lean_closure_set(v___x_444_, 3, v_c_443_);
return v___x_444_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddSemigroupQuotient(lean_object* v_R_445_, lean_object* v_inst_446_, lean_object* v_inst_447_, lean_object* v_c_448_){
_start:
{
lean_object* v___x_449_; 
v___x_449_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instAddQuotient___aux__1___boxed), 6, 4);
lean_closure_set(v___x_449_, 0, lean_box(0));
lean_closure_set(v___x_449_, 1, v_inst_446_);
lean_closure_set(v___x_449_, 2, v_inst_447_);
lean_closure_set(v___x_449_, 3, v_c_448_);
return v___x_449_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddCommMagmaQuotient___redArg(lean_object* v_inst_450_, lean_object* v_inst_451_, lean_object* v_c_452_){
_start:
{
lean_object* v___x_453_; 
v___x_453_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instAddQuotient___aux__1___boxed), 6, 4);
lean_closure_set(v___x_453_, 0, lean_box(0));
lean_closure_set(v___x_453_, 1, v_inst_450_);
lean_closure_set(v___x_453_, 2, v_inst_451_);
lean_closure_set(v___x_453_, 3, v_c_452_);
return v___x_453_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddCommMagmaQuotient(lean_object* v_R_454_, lean_object* v_inst_455_, lean_object* v_inst_456_, lean_object* v_c_457_){
_start:
{
lean_object* v___x_458_; 
v___x_458_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instAddQuotient___aux__1___boxed), 6, 4);
lean_closure_set(v___x_458_, 0, lean_box(0));
lean_closure_set(v___x_458_, 1, v_inst_455_);
lean_closure_set(v___x_458_, 2, v_inst_456_);
lean_closure_set(v___x_458_, 3, v_c_457_);
return v___x_458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddCommSemigroupQuotient___redArg(lean_object* v_inst_459_, lean_object* v_inst_460_, lean_object* v_c_461_){
_start:
{
lean_object* v___x_462_; 
v___x_462_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instAddQuotient___aux__1___boxed), 6, 4);
lean_closure_set(v___x_462_, 0, lean_box(0));
lean_closure_set(v___x_462_, 1, v_inst_459_);
lean_closure_set(v___x_462_, 2, v_inst_460_);
lean_closure_set(v___x_462_, 3, v_c_461_);
return v___x_462_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddCommSemigroupQuotient(lean_object* v_R_463_, lean_object* v_inst_464_, lean_object* v_inst_465_, lean_object* v_c_466_){
_start:
{
lean_object* v___x_467_; 
v___x_467_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instAddQuotient___aux__1___boxed), 6, 4);
lean_closure_set(v___x_467_, 0, lean_box(0));
lean_closure_set(v___x_467_, 1, v_inst_464_);
lean_closure_set(v___x_467_, 2, v_inst_465_);
lean_closure_set(v___x_467_, 3, v_c_466_);
return v___x_467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddMonoidQuotient___redArg___lam__0(lean_object* v_toNSMul_468_, lean_object* v_n_469_, lean_object* v_x_470_){
_start:
{
lean_object* v___x_471_; 
v___x_471_ = lp_mathlib_NSMul_toSMul___redArg___lam__0(v_toNSMul_468_, v_n_469_, v_x_470_);
return v___x_471_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddMonoidQuotient___redArg(lean_object* v_inst_472_, lean_object* v_inst_473_, lean_object* v_c_474_){
_start:
{
lean_object* v___x_475_; lean_object* v_toAdd_476_; lean_object* v_toNSMul_477_; lean_object* v___x_479_; uint8_t v_isShared_480_; uint8_t v_isSharedCheck_487_; 
v___x_475_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_472_);
v_toAdd_476_ = lean_ctor_get(v_inst_472_, 1);
v_toNSMul_477_ = lean_ctor_get(v_inst_472_, 2);
v_isSharedCheck_487_ = !lean_is_exclusive(v_inst_472_);
if (v_isSharedCheck_487_ == 0)
{
lean_object* v_unused_488_; 
v_unused_488_ = lean_ctor_get(v_inst_472_, 0);
lean_dec(v_unused_488_);
v___x_479_ = v_inst_472_;
v_isShared_480_ = v_isSharedCheck_487_;
goto v_resetjp_478_;
}
else
{
lean_inc(v_toNSMul_477_);
lean_inc(v_toAdd_476_);
lean_dec(v_inst_472_);
v___x_479_ = lean_box(0);
v_isShared_480_ = v_isSharedCheck_487_;
goto v_resetjp_478_;
}
v_resetjp_478_:
{
lean_object* v___f_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_485_; 
v___f_481_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instAddMonoidQuotient___redArg___lam__0), 3, 1);
lean_closure_set(v___f_481_, 0, v_toNSMul_477_);
v___x_482_ = lp_mathlib_RingCon_instZeroQuotient___redArg(v___x_475_);
v___x_483_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instAddQuotient___aux__1___boxed), 6, 4);
lean_closure_set(v___x_483_, 0, lean_box(0));
lean_closure_set(v___x_483_, 1, v_toAdd_476_);
lean_closure_set(v___x_483_, 2, v_inst_473_);
lean_closure_set(v___x_483_, 3, v_c_474_);
if (v_isShared_480_ == 0)
{
lean_ctor_set(v___x_479_, 2, v___f_481_);
lean_ctor_set(v___x_479_, 1, v___x_483_);
lean_ctor_set(v___x_479_, 0, v___x_482_);
v___x_485_ = v___x_479_;
goto v_reusejp_484_;
}
else
{
lean_object* v_reuseFailAlloc_486_; 
v_reuseFailAlloc_486_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_486_, 0, v___x_482_);
lean_ctor_set(v_reuseFailAlloc_486_, 1, v___x_483_);
lean_ctor_set(v_reuseFailAlloc_486_, 2, v___f_481_);
v___x_485_ = v_reuseFailAlloc_486_;
goto v_reusejp_484_;
}
v_reusejp_484_:
{
return v___x_485_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddMonoidQuotient(lean_object* v_R_489_, lean_object* v_inst_490_, lean_object* v_inst_491_, lean_object* v_c_492_){
_start:
{
lean_object* v___x_493_; 
v___x_493_ = lp_mathlib_RingCon_instAddMonoidQuotient___redArg(v_inst_490_, v_inst_491_, v_c_492_);
return v___x_493_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddCommMonoidQuotient___redArg(lean_object* v_inst_494_, lean_object* v_inst_495_, lean_object* v_c_496_){
_start:
{
lean_object* v___x_497_; 
v___x_497_ = lp_mathlib_RingCon_instAddMonoidQuotient___redArg(v_inst_494_, v_inst_495_, v_c_496_);
return v___x_497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddCommMonoidQuotient(lean_object* v_R_498_, lean_object* v_inst_499_, lean_object* v_inst_500_, lean_object* v_c_501_){
_start:
{
lean_object* v___x_502_; 
v___x_502_ = lp_mathlib_RingCon_instAddMonoidQuotient___redArg(v_inst_499_, v_inst_500_, v_c_501_);
return v___x_502_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddGroupQuotient___redArg___lam__0(lean_object* v_toZSMul_503_, lean_object* v_n_504_, lean_object* v_x_505_){
_start:
{
lean_object* v___x_506_; 
v___x_506_ = lp_mathlib_ZSMul_toSMul___redArg___lam__0(v_toZSMul_503_, v_n_504_, v_x_505_);
return v___x_506_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddGroupQuotient___redArg(lean_object* v_inst_507_, lean_object* v_inst_508_, lean_object* v_c_509_){
_start:
{
lean_object* v_toAddMonoid_510_; lean_object* v_toZSMul_511_; lean_object* v___f_512_; lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v___x_515_; lean_object* v___x_516_; 
v_toAddMonoid_510_ = lean_ctor_get(v_inst_507_, 0);
v_toZSMul_511_ = lean_ctor_get(v_inst_507_, 3);
lean_inc(v_toZSMul_511_);
v___f_512_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instAddGroupQuotient___redArg___lam__0), 3, 1);
lean_closure_set(v___f_512_, 0, v_toZSMul_511_);
lean_inc_n(v_inst_508_, 2);
lean_inc_ref(v_toAddMonoid_510_);
v___x_513_ = lp_mathlib_RingCon_instAddMonoidQuotient___redArg(v_toAddMonoid_510_, v_inst_508_, v_c_509_);
lean_inc_ref(v_inst_507_);
v___x_514_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instNegQuotient___aux__1___boxed), 5, 4);
lean_closure_set(v___x_514_, 0, lean_box(0));
lean_closure_set(v___x_514_, 1, v_inst_507_);
lean_closure_set(v___x_514_, 2, v_inst_508_);
lean_closure_set(v___x_514_, 3, v_c_509_);
v___x_515_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instSubQuotient___aux__1___boxed), 6, 4);
lean_closure_set(v___x_515_, 0, lean_box(0));
lean_closure_set(v___x_515_, 1, v_inst_507_);
lean_closure_set(v___x_515_, 2, v_inst_508_);
lean_closure_set(v___x_515_, 3, v_c_509_);
v___x_516_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_516_, 0, v___x_513_);
lean_ctor_set(v___x_516_, 1, v___x_514_);
lean_ctor_set(v___x_516_, 2, v___x_515_);
lean_ctor_set(v___x_516_, 3, v___f_512_);
return v___x_516_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddGroupQuotient(lean_object* v_R_517_, lean_object* v_inst_518_, lean_object* v_inst_519_, lean_object* v_c_520_){
_start:
{
lean_object* v___x_521_; 
v___x_521_ = lp_mathlib_RingCon_instAddGroupQuotient___redArg(v_inst_518_, v_inst_519_, v_c_520_);
return v___x_521_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddCommGroupQuotient___redArg(lean_object* v_inst_522_, lean_object* v_inst_523_, lean_object* v_c_524_){
_start:
{
lean_object* v___x_525_; 
v___x_525_ = lp_mathlib_RingCon_instAddGroupQuotient___redArg(v_inst_522_, v_inst_523_, v_c_524_);
return v___x_525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instAddCommGroupQuotient(lean_object* v_R_526_, lean_object* v_inst_527_, lean_object* v_inst_528_, lean_object* v_c_529_){
_start:
{
lean_object* v___x_530_; 
v___x_530_ = lp_mathlib_RingCon_instAddGroupQuotient___redArg(v_inst_527_, v_inst_528_, v_c_529_);
return v___x_530_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulOneClassQuotient___redArg(lean_object* v_inst_531_, lean_object* v_inst_532_, lean_object* v_c_533_){
_start:
{
lean_object* v___x_534_; lean_object* v_toMul_535_; lean_object* v___x_537_; uint8_t v_isShared_538_; uint8_t v_isSharedCheck_544_; 
lean_inc_ref(v_inst_532_);
v___x_534_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_532_);
v_toMul_535_ = lean_ctor_get(v___x_534_, 1);
v_isSharedCheck_544_ = !lean_is_exclusive(v___x_534_);
if (v_isSharedCheck_544_ == 0)
{
lean_object* v_unused_545_; 
v_unused_545_ = lean_ctor_get(v___x_534_, 0);
lean_dec(v_unused_545_);
v___x_537_ = v___x_534_;
v_isShared_538_ = v_isSharedCheck_544_;
goto v_resetjp_536_;
}
else
{
lean_inc(v_toMul_535_);
lean_dec(v___x_534_);
v___x_537_ = lean_box(0);
v_isShared_538_ = v_isSharedCheck_544_;
goto v_resetjp_536_;
}
v_resetjp_536_:
{
lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_542_; 
v___x_539_ = lp_mathlib_RingCon_instOneQuotient___redArg(v_inst_532_);
v___x_540_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instMulQuotient___aux__1___boxed), 6, 4);
lean_closure_set(v___x_540_, 0, lean_box(0));
lean_closure_set(v___x_540_, 1, v_inst_531_);
lean_closure_set(v___x_540_, 2, v_toMul_535_);
lean_closure_set(v___x_540_, 3, v_c_533_);
if (v_isShared_538_ == 0)
{
lean_ctor_set(v___x_537_, 1, v___x_540_);
lean_ctor_set(v___x_537_, 0, v___x_539_);
v___x_542_ = v___x_537_;
goto v_reusejp_541_;
}
else
{
lean_object* v_reuseFailAlloc_543_; 
v_reuseFailAlloc_543_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_543_, 0, v___x_539_);
lean_ctor_set(v_reuseFailAlloc_543_, 1, v___x_540_);
v___x_542_ = v_reuseFailAlloc_543_;
goto v_reusejp_541_;
}
v_reusejp_541_:
{
return v___x_542_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMulOneClassQuotient(lean_object* v_R_546_, lean_object* v_inst_547_, lean_object* v_inst_548_, lean_object* v_c_549_){
_start:
{
lean_object* v___x_550_; 
v___x_550_ = lp_mathlib_RingCon_instMulOneClassQuotient___redArg(v_inst_547_, v_inst_548_, v_c_549_);
return v___x_550_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instSemigroupQuotient___redArg(lean_object* v_inst_551_, lean_object* v_inst_552_, lean_object* v_c_553_){
_start:
{
lean_object* v___x_554_; 
v___x_554_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instMulQuotient___aux__1___boxed), 6, 4);
lean_closure_set(v___x_554_, 0, lean_box(0));
lean_closure_set(v___x_554_, 1, v_inst_551_);
lean_closure_set(v___x_554_, 2, v_inst_552_);
lean_closure_set(v___x_554_, 3, v_c_553_);
return v___x_554_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instSemigroupQuotient(lean_object* v_R_555_, lean_object* v_inst_556_, lean_object* v_inst_557_, lean_object* v_c_558_){
_start:
{
lean_object* v___x_559_; 
v___x_559_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instMulQuotient___aux__1___boxed), 6, 4);
lean_closure_set(v___x_559_, 0, lean_box(0));
lean_closure_set(v___x_559_, 1, v_inst_556_);
lean_closure_set(v___x_559_, 2, v_inst_557_);
lean_closure_set(v___x_559_, 3, v_c_558_);
return v___x_559_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCommMagmaQuotient___redArg(lean_object* v_inst_560_, lean_object* v_inst_561_, lean_object* v_c_562_){
_start:
{
lean_object* v___x_563_; 
v___x_563_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instMulQuotient___aux__1___boxed), 6, 4);
lean_closure_set(v___x_563_, 0, lean_box(0));
lean_closure_set(v___x_563_, 1, v_inst_560_);
lean_closure_set(v___x_563_, 2, v_inst_561_);
lean_closure_set(v___x_563_, 3, v_c_562_);
return v___x_563_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCommMagmaQuotient(lean_object* v_R_564_, lean_object* v_inst_565_, lean_object* v_inst_566_, lean_object* v_c_567_){
_start:
{
lean_object* v___x_568_; 
v___x_568_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instMulQuotient___aux__1___boxed), 6, 4);
lean_closure_set(v___x_568_, 0, lean_box(0));
lean_closure_set(v___x_568_, 1, v_inst_565_);
lean_closure_set(v___x_568_, 2, v_inst_566_);
lean_closure_set(v___x_568_, 3, v_c_567_);
return v___x_568_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCommSemigroupQuotient___redArg(lean_object* v_inst_569_, lean_object* v_inst_570_, lean_object* v_c_571_){
_start:
{
lean_object* v___x_572_; 
v___x_572_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instMulQuotient___aux__1___boxed), 6, 4);
lean_closure_set(v___x_572_, 0, lean_box(0));
lean_closure_set(v___x_572_, 1, v_inst_569_);
lean_closure_set(v___x_572_, 2, v_inst_570_);
lean_closure_set(v___x_572_, 3, v_c_571_);
return v___x_572_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCommSemigroupQuotient(lean_object* v_R_573_, lean_object* v_inst_574_, lean_object* v_inst_575_, lean_object* v_c_576_){
_start:
{
lean_object* v___x_577_; 
v___x_577_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instMulQuotient___aux__1___boxed), 6, 4);
lean_closure_set(v___x_577_, 0, lean_box(0));
lean_closure_set(v___x_577_, 1, v_inst_574_);
lean_closure_set(v___x_577_, 2, v_inst_575_);
lean_closure_set(v___x_577_, 3, v_c_576_);
return v___x_577_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMonoidQuotient___redArg(lean_object* v_inst_578_, lean_object* v_inst_579_, lean_object* v_c_580_){
_start:
{
lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v_toMul_583_; lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v___f_587_; lean_object* v___x_588_; 
v___x_581_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_579_);
lean_inc_ref(v___x_581_);
v___x_582_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_581_);
v_toMul_583_ = lean_ctor_get(v___x_582_, 1);
lean_inc(v_toMul_583_);
lean_dec_ref(v___x_582_);
v___x_584_ = lp_mathlib_RingCon_instOneQuotient___redArg(v___x_581_);
lean_inc(v_inst_578_);
v___x_585_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instMulQuotient___aux__1___boxed), 6, 4);
lean_closure_set(v___x_585_, 0, lean_box(0));
lean_closure_set(v___x_585_, 1, v_inst_578_);
lean_closure_set(v___x_585_, 2, v_toMul_583_);
lean_closure_set(v___x_585_, 3, v_c_580_);
v___x_586_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instPowQuotientNat___aux__1___boxed), 6, 4);
lean_closure_set(v___x_586_, 0, lean_box(0));
lean_closure_set(v___x_586_, 1, v_inst_578_);
lean_closure_set(v___x_586_, 2, v_inst_579_);
lean_closure_set(v___x_586_, 3, v_c_580_);
v___f_587_ = lean_alloc_closure((void*)(lp_mathlib_NPow_ofPow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_587_, 0, v___x_586_);
v___x_588_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_588_, 0, v___x_584_);
lean_ctor_set(v___x_588_, 1, v___x_585_);
lean_ctor_set(v___x_588_, 2, v___f_587_);
return v___x_588_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instMonoidQuotient(lean_object* v_R_589_, lean_object* v_inst_590_, lean_object* v_inst_591_, lean_object* v_c_592_){
_start:
{
lean_object* v___x_593_; 
v___x_593_ = lp_mathlib_RingCon_instMonoidQuotient___redArg(v_inst_590_, v_inst_591_, v_c_592_);
return v___x_593_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCommMonoidQuotient___redArg(lean_object* v_inst_594_, lean_object* v_inst_595_, lean_object* v_c_596_){
_start:
{
lean_object* v___x_597_; 
v___x_597_ = lp_mathlib_RingCon_instMonoidQuotient___redArg(v_inst_594_, v_inst_595_, v_c_596_);
return v___x_597_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCommMonoidQuotient(lean_object* v_R_598_, lean_object* v_inst_599_, lean_object* v_inst_600_, lean_object* v_c_601_){
_start:
{
lean_object* v___x_602_; 
v___x_602_ = lp_mathlib_RingCon_instMonoidQuotient___redArg(v_inst_599_, v_inst_600_, v_c_601_);
return v___x_602_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalNonAssocSemiringQuotient___redArg(lean_object* v_inst_603_, lean_object* v_c_604_){
_start:
{
lean_object* v_toAddCommMonoid_605_; lean_object* v___x_606_; lean_object* v_toMul_607_; lean_object* v_toAdd_608_; lean_object* v___x_610_; uint8_t v_isShared_611_; uint8_t v_isSharedCheck_617_; 
v_toAddCommMonoid_605_ = lean_ctor_get(v_inst_603_, 0);
lean_inc_ref(v_toAddCommMonoid_605_);
v___x_606_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v_inst_603_);
v_toMul_607_ = lean_ctor_get(v___x_606_, 0);
v_toAdd_608_ = lean_ctor_get(v___x_606_, 1);
v_isSharedCheck_617_ = !lean_is_exclusive(v___x_606_);
if (v_isSharedCheck_617_ == 0)
{
v___x_610_ = v___x_606_;
v_isShared_611_ = v_isSharedCheck_617_;
goto v_resetjp_609_;
}
else
{
lean_inc(v_toAdd_608_);
lean_inc(v_toMul_607_);
lean_dec(v___x_606_);
v___x_610_ = lean_box(0);
v_isShared_611_ = v_isSharedCheck_617_;
goto v_resetjp_609_;
}
v_resetjp_609_:
{
lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_615_; 
lean_inc(v_toMul_607_);
v___x_612_ = lp_mathlib_RingCon_instAddMonoidQuotient___redArg(v_toAddCommMonoid_605_, v_toMul_607_, v_c_604_);
v___x_613_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instMulQuotient___aux__1___boxed), 6, 4);
lean_closure_set(v___x_613_, 0, lean_box(0));
lean_closure_set(v___x_613_, 1, v_toAdd_608_);
lean_closure_set(v___x_613_, 2, v_toMul_607_);
lean_closure_set(v___x_613_, 3, v_c_604_);
if (v_isShared_611_ == 0)
{
lean_ctor_set(v___x_610_, 1, v___x_613_);
lean_ctor_set(v___x_610_, 0, v___x_612_);
v___x_615_ = v___x_610_;
goto v_reusejp_614_;
}
else
{
lean_object* v_reuseFailAlloc_616_; 
v_reuseFailAlloc_616_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_616_, 0, v___x_612_);
lean_ctor_set(v_reuseFailAlloc_616_, 1, v___x_613_);
v___x_615_ = v_reuseFailAlloc_616_;
goto v_reusejp_614_;
}
v_reusejp_614_:
{
return v___x_615_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalNonAssocSemiringQuotient(lean_object* v_R_618_, lean_object* v_inst_619_, lean_object* v_c_620_){
_start:
{
lean_object* v___x_621_; 
v___x_621_ = lp_mathlib_RingCon_instNonUnitalNonAssocSemiringQuotient___redArg(v_inst_619_, v_c_620_);
return v___x_621_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalNonAssocCommSemiringQuotient___redArg(lean_object* v_inst_622_, lean_object* v_c_623_){
_start:
{
lean_object* v___x_624_; 
v___x_624_ = lp_mathlib_RingCon_instNonUnitalNonAssocSemiringQuotient___redArg(v_inst_622_, v_c_623_);
return v___x_624_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalNonAssocCommSemiringQuotient(lean_object* v_R_625_, lean_object* v_inst_626_, lean_object* v_c_627_){
_start:
{
lean_object* v___x_628_; 
v___x_628_ = lp_mathlib_RingCon_instNonUnitalNonAssocSemiringQuotient___redArg(v_inst_626_, v_c_627_);
return v___x_628_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonAssocSemiringQuotient___redArg(lean_object* v_inst_629_, lean_object* v_c_630_){
_start:
{
lean_object* v_toNonUnitalNonAssocSemiring_631_; lean_object* v___x_632_; lean_object* v___x_633_; lean_object* v_toMulOneClass_634_; lean_object* v___x_635_; lean_object* v___x_636_; lean_object* v___x_637_; lean_object* v___x_638_; 
v_toNonUnitalNonAssocSemiring_631_ = lean_ctor_get(v_inst_629_, 0);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_631_);
v___x_632_ = lp_mathlib_RingCon_instNonUnitalNonAssocSemiringQuotient___redArg(v_toNonUnitalNonAssocSemiring_631_, v_c_630_);
lean_inc_ref(v_inst_629_);
v___x_633_ = lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(v_inst_629_);
v_toMulOneClass_634_ = lean_ctor_get(v___x_633_, 0);
lean_inc_ref(v_toMulOneClass_634_);
lean_dec_ref(v___x_633_);
v___x_635_ = lp_mathlib_RingCon_instOneQuotient___redArg(v_toMulOneClass_634_);
v___x_636_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v_inst_629_);
v___x_637_ = lp_mathlib_RingCon_instNatCastQuotient___redArg(v___x_636_);
v___x_638_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_638_, 0, v___x_632_);
lean_ctor_set(v___x_638_, 1, v___x_635_);
lean_ctor_set(v___x_638_, 2, v___x_637_);
return v___x_638_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonAssocSemiringQuotient(lean_object* v_R_639_, lean_object* v_inst_640_, lean_object* v_c_641_){
_start:
{
lean_object* v___x_642_; 
v___x_642_ = lp_mathlib_RingCon_instNonAssocSemiringQuotient___redArg(v_inst_640_, v_c_641_);
return v___x_642_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonAssocCommSemiringQuotient___redArg(lean_object* v_inst_643_, lean_object* v_c_644_){
_start:
{
lean_object* v___x_645_; 
v___x_645_ = lp_mathlib_RingCon_instNonAssocSemiringQuotient___redArg(v_inst_643_, v_c_644_);
return v___x_645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonAssocCommSemiringQuotient(lean_object* v_R_646_, lean_object* v_inst_647_, lean_object* v_c_648_){
_start:
{
lean_object* v___x_649_; 
v___x_649_ = lp_mathlib_RingCon_instNonAssocSemiringQuotient___redArg(v_inst_647_, v_c_648_);
return v___x_649_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalSemiringQuotient___redArg(lean_object* v_inst_650_, lean_object* v_c_651_){
_start:
{
lean_object* v___x_652_; 
v___x_652_ = lp_mathlib_RingCon_instNonUnitalNonAssocSemiringQuotient___redArg(v_inst_650_, v_c_651_);
return v___x_652_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalSemiringQuotient(lean_object* v_R_653_, lean_object* v_inst_654_, lean_object* v_c_655_){
_start:
{
lean_object* v___x_656_; 
v___x_656_ = lp_mathlib_RingCon_instNonUnitalNonAssocSemiringQuotient___redArg(v_inst_654_, v_c_655_);
return v___x_656_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalCommSemiringQuotient___redArg(lean_object* v_inst_657_, lean_object* v_c_658_){
_start:
{
lean_object* v___x_659_; 
v___x_659_ = lp_mathlib_RingCon_instNonUnitalNonAssocSemiringQuotient___redArg(v_inst_657_, v_c_658_);
return v___x_659_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalCommSemiringQuotient(lean_object* v_R_660_, lean_object* v_inst_661_, lean_object* v_c_662_){
_start:
{
lean_object* v___x_663_; 
v___x_663_ = lp_mathlib_RingCon_instNonUnitalNonAssocSemiringQuotient___redArg(v_inst_661_, v_c_662_);
return v___x_663_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instSemiringQuotient___redArg(lean_object* v_inst_664_, lean_object* v_c_665_){
_start:
{
lean_object* v_toAddCommMonoid_666_; lean_object* v_toMonoid_667_; lean_object* v___x_668_; lean_object* v_toMul_669_; lean_object* v_toAdd_670_; lean_object* v___x_671_; lean_object* v___x_672_; lean_object* v___x_673_; lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; 
v_toAddCommMonoid_666_ = lean_ctor_get(v_inst_664_, 0);
v_toMonoid_667_ = lean_ctor_get(v_inst_664_, 1);
lean_inc_ref(v_inst_664_);
v___x_668_ = lp_mathlib_instDistribOfSemiring___redArg(v_inst_664_);
v_toMul_669_ = lean_ctor_get(v___x_668_, 0);
lean_inc(v_toMul_669_);
v_toAdd_670_ = lean_ctor_get(v___x_668_, 1);
lean_inc(v_toAdd_670_);
lean_dec_ref(v___x_668_);
lean_inc_ref(v_toAddCommMonoid_666_);
v___x_671_ = lp_mathlib_RingCon_instAddMonoidQuotient___redArg(v_toAddCommMonoid_666_, v_toMul_669_, v_c_665_);
lean_inc_ref(v_toMonoid_667_);
v___x_672_ = lp_mathlib_RingCon_instMonoidQuotient___redArg(v_toAdd_670_, v_toMonoid_667_, v_c_665_);
v___x_673_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_664_);
v___x_674_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_673_);
v___x_675_ = lp_mathlib_RingCon_instNatCastQuotient___redArg(v___x_674_);
v___x_676_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_676_, 0, v___x_671_);
lean_ctor_set(v___x_676_, 1, v___x_672_);
lean_ctor_set(v___x_676_, 2, v___x_675_);
return v___x_676_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instSemiringQuotient(lean_object* v_R_677_, lean_object* v_inst_678_, lean_object* v_c_679_){
_start:
{
lean_object* v___x_680_; 
v___x_680_ = lp_mathlib_RingCon_instSemiringQuotient___redArg(v_inst_678_, v_c_679_);
return v___x_680_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCommSemiringQuotient___redArg(lean_object* v_inst_681_, lean_object* v_c_682_){
_start:
{
lean_object* v___x_683_; 
v___x_683_ = lp_mathlib_RingCon_instSemiringQuotient___redArg(v_inst_681_, v_c_682_);
return v___x_683_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCommSemiringQuotient(lean_object* v_R_684_, lean_object* v_inst_685_, lean_object* v_c_686_){
_start:
{
lean_object* v___x_687_; 
v___x_687_ = lp_mathlib_RingCon_instSemiringQuotient___redArg(v_inst_685_, v_c_686_);
return v___x_687_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalNonAssocRingQuotient___redArg(lean_object* v_inst_688_, lean_object* v_c_689_){
_start:
{
lean_object* v_toAddCommGroup_690_; lean_object* v___x_691_; lean_object* v___x_692_; lean_object* v_toMul_693_; lean_object* v_toAdd_694_; lean_object* v___x_696_; uint8_t v_isShared_697_; uint8_t v_isSharedCheck_703_; 
v_toAddCommGroup_690_ = lean_ctor_get(v_inst_688_, 0);
lean_inc_ref(v_toAddCommGroup_690_);
v___x_691_ = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(v_inst_688_);
v___x_692_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v___x_691_);
v_toMul_693_ = lean_ctor_get(v___x_692_, 0);
v_toAdd_694_ = lean_ctor_get(v___x_692_, 1);
v_isSharedCheck_703_ = !lean_is_exclusive(v___x_692_);
if (v_isSharedCheck_703_ == 0)
{
v___x_696_ = v___x_692_;
v_isShared_697_ = v_isSharedCheck_703_;
goto v_resetjp_695_;
}
else
{
lean_inc(v_toAdd_694_);
lean_inc(v_toMul_693_);
lean_dec(v___x_692_);
v___x_696_ = lean_box(0);
v_isShared_697_ = v_isSharedCheck_703_;
goto v_resetjp_695_;
}
v_resetjp_695_:
{
lean_object* v___x_698_; lean_object* v___x_699_; lean_object* v___x_701_; 
lean_inc(v_toMul_693_);
v___x_698_ = lp_mathlib_RingCon_instAddGroupQuotient___redArg(v_toAddCommGroup_690_, v_toMul_693_, v_c_689_);
v___x_699_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instMulQuotient___aux__1___boxed), 6, 4);
lean_closure_set(v___x_699_, 0, lean_box(0));
lean_closure_set(v___x_699_, 1, v_toAdd_694_);
lean_closure_set(v___x_699_, 2, v_toMul_693_);
lean_closure_set(v___x_699_, 3, v_c_689_);
if (v_isShared_697_ == 0)
{
lean_ctor_set(v___x_696_, 1, v___x_699_);
lean_ctor_set(v___x_696_, 0, v___x_698_);
v___x_701_ = v___x_696_;
goto v_reusejp_700_;
}
else
{
lean_object* v_reuseFailAlloc_702_; 
v_reuseFailAlloc_702_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_702_, 0, v___x_698_);
lean_ctor_set(v_reuseFailAlloc_702_, 1, v___x_699_);
v___x_701_ = v_reuseFailAlloc_702_;
goto v_reusejp_700_;
}
v_reusejp_700_:
{
return v___x_701_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalNonAssocRingQuotient(lean_object* v_R_704_, lean_object* v_inst_705_, lean_object* v_c_706_){
_start:
{
lean_object* v___x_707_; 
v___x_707_ = lp_mathlib_RingCon_instNonUnitalNonAssocRingQuotient___redArg(v_inst_705_, v_c_706_);
return v___x_707_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalNonAssocCommRingQuotient___redArg(lean_object* v_inst_708_, lean_object* v_c_709_){
_start:
{
lean_object* v___x_710_; 
v___x_710_ = lp_mathlib_RingCon_instNonUnitalNonAssocRingQuotient___redArg(v_inst_708_, v_c_709_);
return v___x_710_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalNonAssocCommRingQuotient(lean_object* v_R_711_, lean_object* v_inst_712_, lean_object* v_c_713_){
_start:
{
lean_object* v___x_714_; 
v___x_714_ = lp_mathlib_RingCon_instNonUnitalNonAssocRingQuotient___redArg(v_inst_712_, v_c_713_);
return v___x_714_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonAssocRingQuotient___redArg(lean_object* v_inst_715_, lean_object* v_c_716_){
_start:
{
lean_object* v_toNonUnitalNonAssocRing_717_; lean_object* v___x_718_; lean_object* v___x_719_; lean_object* v___x_720_; lean_object* v_toMulOneClass_721_; lean_object* v___x_722_; lean_object* v___x_723_; lean_object* v_toAddMonoidWithOne_724_; lean_object* v___x_725_; lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v___x_728_; 
v_toNonUnitalNonAssocRing_717_ = lean_ctor_get(v_inst_715_, 0);
lean_inc_ref(v_toNonUnitalNonAssocRing_717_);
v___x_718_ = lp_mathlib_RingCon_instNonUnitalNonAssocRingQuotient___redArg(v_toNonUnitalNonAssocRing_717_, v_c_716_);
lean_inc_ref(v_inst_715_);
v___x_719_ = lp_mathlib_NonAssocRing_toNonAssocSemiring___redArg(v_inst_715_);
v___x_720_ = lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(v___x_719_);
v_toMulOneClass_721_ = lean_ctor_get(v___x_720_, 0);
lean_inc_ref(v_toMulOneClass_721_);
lean_dec_ref(v___x_720_);
v___x_722_ = lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(v_inst_715_);
v___x_723_ = lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(v___x_722_);
lean_dec_ref(v___x_722_);
v_toAddMonoidWithOne_724_ = lean_ctor_get(v___x_723_, 1);
lean_inc_ref(v_toAddMonoidWithOne_724_);
v___x_725_ = lp_mathlib_RingCon_instOneQuotient___redArg(v_toMulOneClass_721_);
v___x_726_ = lp_mathlib_RingCon_instNatCastQuotient___redArg(v_toAddMonoidWithOne_724_);
v___x_727_ = lp_mathlib_RingCon_instIntCastQuotient___redArg(v___x_723_);
v___x_728_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_728_, 0, v___x_718_);
lean_ctor_set(v___x_728_, 1, v___x_725_);
lean_ctor_set(v___x_728_, 2, v___x_726_);
lean_ctor_set(v___x_728_, 3, v___x_727_);
return v___x_728_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonAssocRingQuotient(lean_object* v_R_729_, lean_object* v_inst_730_, lean_object* v_c_731_){
_start:
{
lean_object* v___x_732_; 
v___x_732_ = lp_mathlib_RingCon_instNonAssocRingQuotient___redArg(v_inst_730_, v_c_731_);
return v___x_732_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonAssocCommRingQuotient___redArg(lean_object* v_inst_733_, lean_object* v_c_734_){
_start:
{
lean_object* v___x_735_; 
v___x_735_ = lp_mathlib_RingCon_instNonAssocRingQuotient___redArg(v_inst_733_, v_c_734_);
return v___x_735_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonAssocCommRingQuotient(lean_object* v_R_736_, lean_object* v_inst_737_, lean_object* v_c_738_){
_start:
{
lean_object* v___x_739_; 
v___x_739_ = lp_mathlib_RingCon_instNonAssocRingQuotient___redArg(v_inst_737_, v_c_738_);
return v___x_739_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalRingQuotient___redArg(lean_object* v_inst_740_, lean_object* v_c_741_){
_start:
{
lean_object* v___x_742_; 
v___x_742_ = lp_mathlib_RingCon_instNonUnitalNonAssocRingQuotient___redArg(v_inst_740_, v_c_741_);
return v___x_742_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalRingQuotient(lean_object* v_R_743_, lean_object* v_inst_744_, lean_object* v_c_745_){
_start:
{
lean_object* v___x_746_; 
v___x_746_ = lp_mathlib_RingCon_instNonUnitalNonAssocRingQuotient___redArg(v_inst_744_, v_c_745_);
return v___x_746_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalCommRingQuotient___redArg(lean_object* v_inst_747_, lean_object* v_c_748_){
_start:
{
lean_object* v___x_749_; 
v___x_749_ = lp_mathlib_RingCon_instNonUnitalNonAssocRingQuotient___redArg(v_inst_747_, v_c_748_);
return v___x_749_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instNonUnitalCommRingQuotient(lean_object* v_R_750_, lean_object* v_inst_751_, lean_object* v_c_752_){
_start:
{
lean_object* v___x_753_; 
v___x_753_ = lp_mathlib_RingCon_instNonUnitalNonAssocRingQuotient___redArg(v_inst_751_, v_c_752_);
return v___x_753_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instRingQuotient___redArg(lean_object* v_inst_754_, lean_object* v_c_755_){
_start:
{
lean_object* v_toSemiring_756_; lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; lean_object* v_toMul_761_; lean_object* v___x_762_; lean_object* v_toZSMul_763_; lean_object* v___x_764_; lean_object* v___x_765_; lean_object* v___x_766_; lean_object* v___x_767_; 
v_toSemiring_756_ = lean_ctor_get(v_inst_754_, 0);
lean_inc_ref_n(v_toSemiring_756_, 2);
v___x_757_ = lp_mathlib_RingCon_instSemiringQuotient___redArg(v_toSemiring_756_, v_c_755_);
v___x_758_ = lp_mathlib_Ring_toAddGroupWithOne___redArg(v_inst_754_);
v___x_759_ = lp_mathlib_AddGroupWithOne_toAddGroup___redArg(v___x_758_);
v___x_760_ = lp_mathlib_instDistribOfSemiring___redArg(v_toSemiring_756_);
v_toMul_761_ = lean_ctor_get(v___x_760_, 0);
lean_inc_n(v_toMul_761_, 3);
lean_dec_ref(v___x_760_);
lean_inc_ref_n(v___x_759_, 2);
v___x_762_ = lp_mathlib_RingCon_instAddGroupQuotient___redArg(v___x_759_, v_toMul_761_, v_c_755_);
v_toZSMul_763_ = lean_ctor_get(v___x_762_, 3);
lean_inc(v_toZSMul_763_);
lean_dec_ref(v___x_762_);
v___x_764_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instNegQuotient___aux__1___boxed), 5, 4);
lean_closure_set(v___x_764_, 0, lean_box(0));
lean_closure_set(v___x_764_, 1, v___x_759_);
lean_closure_set(v___x_764_, 2, v_toMul_761_);
lean_closure_set(v___x_764_, 3, v_c_755_);
v___x_765_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_instSubQuotient___aux__1___boxed), 6, 4);
lean_closure_set(v___x_765_, 0, lean_box(0));
lean_closure_set(v___x_765_, 1, v___x_759_);
lean_closure_set(v___x_765_, 2, v_toMul_761_);
lean_closure_set(v___x_765_, 3, v_c_755_);
v___x_766_ = lp_mathlib_RingCon_instIntCastQuotient___redArg(v___x_758_);
v___x_767_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_767_, 0, v___x_757_);
lean_ctor_set(v___x_767_, 1, v___x_764_);
lean_ctor_set(v___x_767_, 2, v___x_765_);
lean_ctor_set(v___x_767_, 3, v_toZSMul_763_);
lean_ctor_set(v___x_767_, 4, v___x_766_);
return v___x_767_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instRingQuotient(lean_object* v_R_768_, lean_object* v_inst_769_, lean_object* v_c_770_){
_start:
{
lean_object* v___x_771_; 
v___x_771_ = lp_mathlib_RingCon_instRingQuotient___redArg(v_inst_769_, v_c_770_);
return v___x_771_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCommRingQuotient___redArg(lean_object* v_inst_772_, lean_object* v_c_773_){
_start:
{
lean_object* v___x_774_; 
v___x_774_ = lp_mathlib_RingCon_instRingQuotient___redArg(v_inst_772_, v_c_773_);
return v___x_774_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_instCommRingQuotient(lean_object* v_R_775_, lean_object* v_inst_776_, lean_object* v_c_777_){
_start:
{
lean_object* v___x_778_; 
v___x_778_ = lp_mathlib_RingCon_instRingQuotient___redArg(v_inst_776_, v_c_777_);
return v___x_778_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_mk_x27___redArg(lean_object* v_inst_779_, lean_object* v_c_780_){
_start:
{
lean_object* v_toNonUnitalNonAssocSemiring_781_; lean_object* v___x_782_; lean_object* v_toMul_783_; lean_object* v_toAdd_784_; lean_object* v___x_785_; 
v_toNonUnitalNonAssocSemiring_781_ = lean_ctor_get(v_inst_779_, 0);
lean_inc_ref(v_toNonUnitalNonAssocSemiring_781_);
lean_dec_ref(v_inst_779_);
v___x_782_ = lp_mathlib_NonUnitalNonAssocSemiring_toDistrib___redArg(v_toNonUnitalNonAssocSemiring_781_);
v_toMul_783_ = lean_ctor_get(v___x_782_, 0);
lean_inc(v_toMul_783_);
v_toAdd_784_ = lean_ctor_get(v___x_782_, 1);
lean_inc(v_toAdd_784_);
lean_dec_ref(v___x_782_);
v___x_785_ = lean_alloc_closure((void*)(lp_mathlib_RingCon_toQuotient___boxed), 5, 4);
lean_closure_set(v___x_785_, 0, lean_box(0));
lean_closure_set(v___x_785_, 1, v_toAdd_784_);
lean_closure_set(v___x_785_, 2, v_toMul_783_);
lean_closure_set(v___x_785_, 3, v_c_780_);
return v___x_785_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingCon_mk_x27(lean_object* v_R_786_, lean_object* v_inst_787_, lean_object* v_c_788_){
_start:
{
lean_object* v___x_789_; 
v___x_789_ = lp_mathlib_RingCon_mk_x27___redArg(v_inst_787_, v_c_788_);
return v___x_789_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Congruence_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FastInstance(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_Congruence_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Congruence_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FastInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_RingTheory_Congruence_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Congruence_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FastInstance(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_Congruence_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Hom_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Congruence_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FastInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_Congruence_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_RingTheory_Congruence_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_RingTheory_Congruence_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
