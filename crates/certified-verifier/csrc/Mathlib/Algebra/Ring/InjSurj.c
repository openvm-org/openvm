// Lean compiler output
// Module: Mathlib.Algebra.Ring.InjSurj
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Ring.Defs public import Mathlib.Algebra.Opposites public import Mathlib.Algebra.GroupWithZero.InjSurj public import Mathlib.Data.Int.Cast.Basic
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
lean_object* l_Nat_cast(lean_object*, lean_object*, lean_object*);
lean_object* l_Int_cast(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Surjective_subNegMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_subNegMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_addMonoid___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Surjective_addMonoid___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulOpposite_instNeg___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distrib___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distrib(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distrib___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_hasDistribNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_hasDistribNeg___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_hasDistribNeg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_hasDistribNeg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addMonoidWithOne___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addMonoidWithOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addMonoidWithOne___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommMonoidWithOne___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommMonoidWithOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommMonoidWithOne___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addGroupWithOne___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addGroupWithOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addGroupWithOne___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommGroupWithOne___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommGroupWithOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommGroupWithOne___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocSemiring___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semiring___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semiring___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocRing___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalRing___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocRing___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_ring___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_ring___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_ring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_ring___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocCommSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocCommSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocCommSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalCommSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalCommSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalCommSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocCommSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocCommSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocCommSemiring___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commSemiring___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocCommRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocCommRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocCommRing___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalCommRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalCommRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalCommRing___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocCommRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocCommRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocCommRing___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commRing___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distrib___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distrib(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distrib___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_hasDistribNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_hasDistribNeg___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_hasDistribNeg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_hasDistribNeg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addMonoidWithOne___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addMonoidWithOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addMonoidWithOne___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommMonoidWithOne___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommMonoidWithOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommMonoidWithOne___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addGroupWithOne___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addGroupWithOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addGroupWithOne___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommGroupWithOne___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommGroupWithOne(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommGroupWithOne___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocSemiring___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_semiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_semiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_semiring___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocRing___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalRing___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocRing___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_ring___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_ring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_ring___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocCommSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocCommSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocCommSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalCommSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalCommSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalCommSemiring___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocCommSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocCommSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocCommSemiring___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commSemiring___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commSemiring(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commSemiring___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocCommRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocCommRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocCommRing___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalCommRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalCommRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalCommRing___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocCommRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocCommRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocCommRing___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commRing___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instHasDistribNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instHasDistribNeg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instHasDistribNeg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distrib___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3_, 0, v_inst_2_);
lean_ctor_set(v___x_3_, 1, v_inst_1_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distrib(lean_object* v_R_4_, lean_object* v_S_5_, lean_object* v_f_6_, lean_object* v_hf_7_, lean_object* v_inst_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_add_11_, lean_object* v_mul_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_13_, 0, v_inst_9_);
lean_ctor_set(v___x_13_, 1, v_inst_8_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_distrib___boxed(lean_object* v_R_14_, lean_object* v_S_15_, lean_object* v_f_16_, lean_object* v_hf_17_, lean_object* v_inst_18_, lean_object* v_inst_19_, lean_object* v_inst_20_, lean_object* v_add_21_, lean_object* v_mul_22_){
_start:
{
lean_object* v_res_23_; 
v_res_23_ = lp_mathlib_Function_Injective_distrib(v_R_14_, v_S_15_, v_f_16_, v_hf_17_, v_inst_18_, v_inst_19_, v_inst_20_, v_add_21_, v_mul_22_);
lean_dec_ref(v_inst_20_);
lean_dec(v_f_16_);
return v_res_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_hasDistribNeg___redArg(lean_object* v_inst_24_){
_start:
{
lean_inc(v_inst_24_);
return v_inst_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_hasDistribNeg___redArg___boxed(lean_object* v_inst_25_){
_start:
{
lean_object* v_res_26_; 
v_res_26_ = lp_mathlib_Function_Injective_hasDistribNeg___redArg(v_inst_25_);
lean_dec(v_inst_25_);
return v_res_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_hasDistribNeg(lean_object* v_R_27_, lean_object* v_S_28_, lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_f_31_, lean_object* v_hf_32_, lean_object* v_inst_33_, lean_object* v_inst_34_, lean_object* v_neg_35_, lean_object* v_mul_36_){
_start:
{
lean_inc(v_inst_30_);
return v_inst_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_hasDistribNeg___boxed(lean_object* v_R_37_, lean_object* v_S_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_f_41_, lean_object* v_hf_42_, lean_object* v_inst_43_, lean_object* v_inst_44_, lean_object* v_neg_45_, lean_object* v_mul_46_){
_start:
{
lean_object* v_res_47_; 
v_res_47_ = lp_mathlib_Function_Injective_hasDistribNeg(v_R_37_, v_S_38_, v_inst_39_, v_inst_40_, v_f_41_, v_hf_42_, v_inst_43_, v_inst_44_, v_neg_45_, v_mul_46_);
lean_dec(v_inst_44_);
lean_dec(v_inst_43_);
lean_dec(v_f_41_);
lean_dec(v_inst_40_);
lean_dec(v_inst_39_);
return v_res_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addMonoidWithOne___redArg(lean_object* v_inst_48_, lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_inst_51_, lean_object* v_inst_52_){
_start:
{
lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; 
v___x_53_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_48_, v_inst_49_, v_inst_51_);
v___x_54_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_54_, 0, lean_box(0));
lean_closure_set(v___x_54_, 1, v_inst_52_);
v___x_55_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_55_, 0, v___x_54_);
lean_ctor_set(v___x_55_, 1, v___x_53_);
lean_ctor_set(v___x_55_, 2, v_inst_50_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addMonoidWithOne(lean_object* v_R_56_, lean_object* v_S_57_, lean_object* v_inst_58_, lean_object* v_inst_59_, lean_object* v_inst_60_, lean_object* v_inst_61_, lean_object* v_inst_62_, lean_object* v_inst_63_, lean_object* v_f_64_, lean_object* v_hf_65_, lean_object* v_zero_66_, lean_object* v_one_67_, lean_object* v_add_68_, lean_object* v_nsmul_69_, lean_object* v_natCast_70_){
_start:
{
lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; 
v___x_71_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_58_, v_inst_59_, v_inst_61_);
v___x_72_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_72_, 0, lean_box(0));
lean_closure_set(v___x_72_, 1, v_inst_62_);
v___x_73_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_73_, 0, v___x_72_);
lean_ctor_set(v___x_73_, 1, v___x_71_);
lean_ctor_set(v___x_73_, 2, v_inst_60_);
return v___x_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addMonoidWithOne___boxed(lean_object* v_R_74_, lean_object* v_S_75_, lean_object* v_inst_76_, lean_object* v_inst_77_, lean_object* v_inst_78_, lean_object* v_inst_79_, lean_object* v_inst_80_, lean_object* v_inst_81_, lean_object* v_f_82_, lean_object* v_hf_83_, lean_object* v_zero_84_, lean_object* v_one_85_, lean_object* v_add_86_, lean_object* v_nsmul_87_, lean_object* v_natCast_88_){
_start:
{
lean_object* v_res_89_; 
v_res_89_ = lp_mathlib_Function_Injective_addMonoidWithOne(v_R_74_, v_S_75_, v_inst_76_, v_inst_77_, v_inst_78_, v_inst_79_, v_inst_80_, v_inst_81_, v_f_82_, v_hf_83_, v_zero_84_, v_one_85_, v_add_86_, v_nsmul_87_, v_natCast_88_);
lean_dec(v_f_82_);
lean_dec_ref(v_inst_81_);
return v_res_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommMonoidWithOne___redArg(lean_object* v_inst_90_, lean_object* v_inst_91_, lean_object* v_inst_92_, lean_object* v_inst_93_, lean_object* v_inst_94_){
_start:
{
lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; 
v___x_95_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_92_, v_inst_90_, v_inst_93_);
v___x_96_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_96_, 0, lean_box(0));
lean_closure_set(v___x_96_, 1, v_inst_94_);
v___x_97_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_97_, 0, v___x_96_);
lean_ctor_set(v___x_97_, 1, v___x_95_);
lean_ctor_set(v___x_97_, 2, v_inst_91_);
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommMonoidWithOne(lean_object* v_R_98_, lean_object* v_S_99_, lean_object* v_inst_100_, lean_object* v_inst_101_, lean_object* v_inst_102_, lean_object* v_inst_103_, lean_object* v_inst_104_, lean_object* v_inst_105_, lean_object* v_f_106_, lean_object* v_hf_107_, lean_object* v_zero_108_, lean_object* v_one_109_, lean_object* v_add_110_, lean_object* v_nsmul_111_, lean_object* v_natCast_112_){
_start:
{
lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; 
v___x_113_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_102_, v_inst_100_, v_inst_103_);
v___x_114_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_114_, 0, lean_box(0));
lean_closure_set(v___x_114_, 1, v_inst_104_);
v___x_115_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_115_, 0, v___x_114_);
lean_ctor_set(v___x_115_, 1, v___x_113_);
lean_ctor_set(v___x_115_, 2, v_inst_101_);
return v___x_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommMonoidWithOne___boxed(lean_object* v_R_116_, lean_object* v_S_117_, lean_object* v_inst_118_, lean_object* v_inst_119_, lean_object* v_inst_120_, lean_object* v_inst_121_, lean_object* v_inst_122_, lean_object* v_inst_123_, lean_object* v_f_124_, lean_object* v_hf_125_, lean_object* v_zero_126_, lean_object* v_one_127_, lean_object* v_add_128_, lean_object* v_nsmul_129_, lean_object* v_natCast_130_){
_start:
{
lean_object* v_res_131_; 
v_res_131_ = lp_mathlib_Function_Injective_addCommMonoidWithOne(v_R_116_, v_S_117_, v_inst_118_, v_inst_119_, v_inst_120_, v_inst_121_, v_inst_122_, v_inst_123_, v_f_124_, v_hf_125_, v_zero_126_, v_one_127_, v_add_128_, v_nsmul_129_, v_natCast_130_);
lean_dec(v_f_124_);
lean_dec_ref(v_inst_123_);
return v_res_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addGroupWithOne___redArg(lean_object* v_inst_132_, lean_object* v_inst_133_, lean_object* v_inst_134_, lean_object* v_inst_135_, lean_object* v_inst_136_, lean_object* v_inst_137_, lean_object* v_inst_138_, lean_object* v_inst_139_, lean_object* v_inst_140_){
_start:
{
lean_object* v___x_141_; lean_object* v_toAddMonoid_142_; lean_object* v_toNeg_143_; lean_object* v_toSub_144_; lean_object* v_toZSMul_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; 
v___x_141_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_134_, v_inst_132_, v_inst_135_, v_inst_136_, v_inst_137_, v_inst_138_);
v_toAddMonoid_142_ = lean_ctor_get(v___x_141_, 0);
lean_inc_ref(v_toAddMonoid_142_);
v_toNeg_143_ = lean_ctor_get(v___x_141_, 1);
lean_inc(v_toNeg_143_);
v_toSub_144_ = lean_ctor_get(v___x_141_, 2);
lean_inc(v_toSub_144_);
v_toZSMul_145_ = lean_ctor_get(v___x_141_, 3);
lean_inc(v_toZSMul_145_);
lean_dec_ref(v___x_141_);
v___x_146_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_146_, 0, lean_box(0));
lean_closure_set(v___x_146_, 1, v_inst_139_);
v___x_147_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_147_, 0, lean_box(0));
lean_closure_set(v___x_147_, 1, v_inst_140_);
v___x_148_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_148_, 0, v___x_146_);
lean_ctor_set(v___x_148_, 1, v_toAddMonoid_142_);
lean_ctor_set(v___x_148_, 2, v_inst_133_);
v___x_149_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_149_, 0, v___x_147_);
lean_ctor_set(v___x_149_, 1, v___x_148_);
lean_ctor_set(v___x_149_, 2, v_toNeg_143_);
lean_ctor_set(v___x_149_, 3, v_toSub_144_);
lean_ctor_set(v___x_149_, 4, v_toZSMul_145_);
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addGroupWithOne(lean_object* v_R_150_, lean_object* v_S_151_, lean_object* v_inst_152_, lean_object* v_inst_153_, lean_object* v_inst_154_, lean_object* v_inst_155_, lean_object* v_inst_156_, lean_object* v_inst_157_, lean_object* v_inst_158_, lean_object* v_inst_159_, lean_object* v_inst_160_, lean_object* v_inst_161_, lean_object* v_f_162_, lean_object* v_hf_163_, lean_object* v_zero_164_, lean_object* v_one_165_, lean_object* v_add_166_, lean_object* v_neg_167_, lean_object* v_sub_168_, lean_object* v_nsmul_169_, lean_object* v_zsmul_170_, lean_object* v_natCast_171_, lean_object* v_intCast_172_){
_start:
{
lean_object* v___x_173_; lean_object* v_toAddMonoid_174_; lean_object* v_toNeg_175_; lean_object* v_toSub_176_; lean_object* v_toZSMul_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; 
v___x_173_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_154_, v_inst_152_, v_inst_155_, v_inst_156_, v_inst_157_, v_inst_158_);
v_toAddMonoid_174_ = lean_ctor_get(v___x_173_, 0);
lean_inc_ref(v_toAddMonoid_174_);
v_toNeg_175_ = lean_ctor_get(v___x_173_, 1);
lean_inc(v_toNeg_175_);
v_toSub_176_ = lean_ctor_get(v___x_173_, 2);
lean_inc(v_toSub_176_);
v_toZSMul_177_ = lean_ctor_get(v___x_173_, 3);
lean_inc(v_toZSMul_177_);
lean_dec_ref(v___x_173_);
v___x_178_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_178_, 0, lean_box(0));
lean_closure_set(v___x_178_, 1, v_inst_159_);
v___x_179_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_179_, 0, lean_box(0));
lean_closure_set(v___x_179_, 1, v_inst_160_);
v___x_180_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_180_, 0, v___x_178_);
lean_ctor_set(v___x_180_, 1, v_toAddMonoid_174_);
lean_ctor_set(v___x_180_, 2, v_inst_153_);
v___x_181_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_181_, 0, v___x_179_);
lean_ctor_set(v___x_181_, 1, v___x_180_);
lean_ctor_set(v___x_181_, 2, v_toNeg_175_);
lean_ctor_set(v___x_181_, 3, v_toSub_176_);
lean_ctor_set(v___x_181_, 4, v_toZSMul_177_);
return v___x_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addGroupWithOne___boxed(lean_object** _args){
lean_object* v_R_182_ = _args[0];
lean_object* v_S_183_ = _args[1];
lean_object* v_inst_184_ = _args[2];
lean_object* v_inst_185_ = _args[3];
lean_object* v_inst_186_ = _args[4];
lean_object* v_inst_187_ = _args[5];
lean_object* v_inst_188_ = _args[6];
lean_object* v_inst_189_ = _args[7];
lean_object* v_inst_190_ = _args[8];
lean_object* v_inst_191_ = _args[9];
lean_object* v_inst_192_ = _args[10];
lean_object* v_inst_193_ = _args[11];
lean_object* v_f_194_ = _args[12];
lean_object* v_hf_195_ = _args[13];
lean_object* v_zero_196_ = _args[14];
lean_object* v_one_197_ = _args[15];
lean_object* v_add_198_ = _args[16];
lean_object* v_neg_199_ = _args[17];
lean_object* v_sub_200_ = _args[18];
lean_object* v_nsmul_201_ = _args[19];
lean_object* v_zsmul_202_ = _args[20];
lean_object* v_natCast_203_ = _args[21];
lean_object* v_intCast_204_ = _args[22];
_start:
{
lean_object* v_res_205_; 
v_res_205_ = lp_mathlib_Function_Injective_addGroupWithOne(v_R_182_, v_S_183_, v_inst_184_, v_inst_185_, v_inst_186_, v_inst_187_, v_inst_188_, v_inst_189_, v_inst_190_, v_inst_191_, v_inst_192_, v_inst_193_, v_f_194_, v_hf_195_, v_zero_196_, v_one_197_, v_add_198_, v_neg_199_, v_sub_200_, v_nsmul_201_, v_zsmul_202_, v_natCast_203_, v_intCast_204_);
lean_dec(v_f_194_);
lean_dec_ref(v_inst_193_);
return v_res_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommGroupWithOne___redArg(lean_object* v_inst_206_, lean_object* v_inst_207_, lean_object* v_inst_208_, lean_object* v_inst_209_, lean_object* v_inst_210_, lean_object* v_inst_211_, lean_object* v_inst_212_, lean_object* v_inst_213_, lean_object* v_inst_214_){
_start:
{
lean_object* v___x_215_; lean_object* v_toAddMonoid_216_; lean_object* v_toNeg_217_; lean_object* v_toSub_218_; lean_object* v_toZSMul_219_; lean_object* v___x_221_; uint8_t v_isShared_222_; uint8_t v_isSharedCheck_229_; 
v___x_215_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_208_, v_inst_206_, v_inst_209_, v_inst_210_, v_inst_211_, v_inst_212_);
v_toAddMonoid_216_ = lean_ctor_get(v___x_215_, 0);
v_toNeg_217_ = lean_ctor_get(v___x_215_, 1);
v_toSub_218_ = lean_ctor_get(v___x_215_, 2);
v_toZSMul_219_ = lean_ctor_get(v___x_215_, 3);
v_isSharedCheck_229_ = !lean_is_exclusive(v___x_215_);
if (v_isSharedCheck_229_ == 0)
{
v___x_221_ = v___x_215_;
v_isShared_222_ = v_isSharedCheck_229_;
goto v_resetjp_220_;
}
else
{
lean_inc(v_toZSMul_219_);
lean_inc(v_toSub_218_);
lean_inc(v_toNeg_217_);
lean_inc(v_toAddMonoid_216_);
lean_dec(v___x_215_);
v___x_221_ = lean_box(0);
v_isShared_222_ = v_isSharedCheck_229_;
goto v_resetjp_220_;
}
v_resetjp_220_:
{
lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_226_; 
v___x_223_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_223_, 0, lean_box(0));
lean_closure_set(v___x_223_, 1, v_inst_213_);
v___x_224_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_224_, 0, lean_box(0));
lean_closure_set(v___x_224_, 1, v_inst_214_);
if (v_isShared_222_ == 0)
{
v___x_226_ = v___x_221_;
goto v_reusejp_225_;
}
else
{
lean_object* v_reuseFailAlloc_228_; 
v_reuseFailAlloc_228_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_228_, 0, v_toAddMonoid_216_);
lean_ctor_set(v_reuseFailAlloc_228_, 1, v_toNeg_217_);
lean_ctor_set(v_reuseFailAlloc_228_, 2, v_toSub_218_);
lean_ctor_set(v_reuseFailAlloc_228_, 3, v_toZSMul_219_);
v___x_226_ = v_reuseFailAlloc_228_;
goto v_reusejp_225_;
}
v_reusejp_225_:
{
lean_object* v___x_227_; 
v___x_227_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_227_, 0, v___x_226_);
lean_ctor_set(v___x_227_, 1, v___x_224_);
lean_ctor_set(v___x_227_, 2, v___x_223_);
lean_ctor_set(v___x_227_, 3, v_inst_207_);
return v___x_227_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommGroupWithOne(lean_object* v_R_230_, lean_object* v_S_231_, lean_object* v_inst_232_, lean_object* v_inst_233_, lean_object* v_inst_234_, lean_object* v_inst_235_, lean_object* v_inst_236_, lean_object* v_inst_237_, lean_object* v_inst_238_, lean_object* v_inst_239_, lean_object* v_inst_240_, lean_object* v_inst_241_, lean_object* v_f_242_, lean_object* v_hf_243_, lean_object* v_zero_244_, lean_object* v_one_245_, lean_object* v_add_246_, lean_object* v_neg_247_, lean_object* v_sub_248_, lean_object* v_nsmul_249_, lean_object* v_zsmul_250_, lean_object* v_natCast_251_, lean_object* v_intCast_252_){
_start:
{
lean_object* v___x_253_; lean_object* v_toAddMonoid_254_; lean_object* v_toNeg_255_; lean_object* v_toSub_256_; lean_object* v_toZSMul_257_; lean_object* v___x_259_; uint8_t v_isShared_260_; uint8_t v_isSharedCheck_267_; 
v___x_253_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_234_, v_inst_232_, v_inst_235_, v_inst_236_, v_inst_237_, v_inst_238_);
v_toAddMonoid_254_ = lean_ctor_get(v___x_253_, 0);
v_toNeg_255_ = lean_ctor_get(v___x_253_, 1);
v_toSub_256_ = lean_ctor_get(v___x_253_, 2);
v_toZSMul_257_ = lean_ctor_get(v___x_253_, 3);
v_isSharedCheck_267_ = !lean_is_exclusive(v___x_253_);
if (v_isSharedCheck_267_ == 0)
{
v___x_259_ = v___x_253_;
v_isShared_260_ = v_isSharedCheck_267_;
goto v_resetjp_258_;
}
else
{
lean_inc(v_toZSMul_257_);
lean_inc(v_toSub_256_);
lean_inc(v_toNeg_255_);
lean_inc(v_toAddMonoid_254_);
lean_dec(v___x_253_);
v___x_259_ = lean_box(0);
v_isShared_260_ = v_isSharedCheck_267_;
goto v_resetjp_258_;
}
v_resetjp_258_:
{
lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_264_; 
v___x_261_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_261_, 0, lean_box(0));
lean_closure_set(v___x_261_, 1, v_inst_239_);
v___x_262_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_262_, 0, lean_box(0));
lean_closure_set(v___x_262_, 1, v_inst_240_);
if (v_isShared_260_ == 0)
{
v___x_264_ = v___x_259_;
goto v_reusejp_263_;
}
else
{
lean_object* v_reuseFailAlloc_266_; 
v_reuseFailAlloc_266_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_266_, 0, v_toAddMonoid_254_);
lean_ctor_set(v_reuseFailAlloc_266_, 1, v_toNeg_255_);
lean_ctor_set(v_reuseFailAlloc_266_, 2, v_toSub_256_);
lean_ctor_set(v_reuseFailAlloc_266_, 3, v_toZSMul_257_);
v___x_264_ = v_reuseFailAlloc_266_;
goto v_reusejp_263_;
}
v_reusejp_263_:
{
lean_object* v___x_265_; 
v___x_265_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_265_, 0, v___x_264_);
lean_ctor_set(v___x_265_, 1, v___x_262_);
lean_ctor_set(v___x_265_, 2, v___x_261_);
lean_ctor_set(v___x_265_, 3, v_inst_233_);
return v___x_265_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommGroupWithOne___boxed(lean_object** _args){
lean_object* v_R_268_ = _args[0];
lean_object* v_S_269_ = _args[1];
lean_object* v_inst_270_ = _args[2];
lean_object* v_inst_271_ = _args[3];
lean_object* v_inst_272_ = _args[4];
lean_object* v_inst_273_ = _args[5];
lean_object* v_inst_274_ = _args[6];
lean_object* v_inst_275_ = _args[7];
lean_object* v_inst_276_ = _args[8];
lean_object* v_inst_277_ = _args[9];
lean_object* v_inst_278_ = _args[10];
lean_object* v_inst_279_ = _args[11];
lean_object* v_f_280_ = _args[12];
lean_object* v_hf_281_ = _args[13];
lean_object* v_zero_282_ = _args[14];
lean_object* v_one_283_ = _args[15];
lean_object* v_add_284_ = _args[16];
lean_object* v_neg_285_ = _args[17];
lean_object* v_sub_286_ = _args[18];
lean_object* v_nsmul_287_ = _args[19];
lean_object* v_zsmul_288_ = _args[20];
lean_object* v_natCast_289_ = _args[21];
lean_object* v_intCast_290_ = _args[22];
_start:
{
lean_object* v_res_291_; 
v_res_291_ = lp_mathlib_Function_Injective_addCommGroupWithOne(v_R_268_, v_S_269_, v_inst_270_, v_inst_271_, v_inst_272_, v_inst_273_, v_inst_274_, v_inst_275_, v_inst_276_, v_inst_277_, v_inst_278_, v_inst_279_, v_f_280_, v_hf_281_, v_zero_282_, v_one_283_, v_add_284_, v_neg_285_, v_sub_286_, v_nsmul_287_, v_zsmul_288_, v_natCast_289_, v_intCast_290_);
lean_dec(v_f_280_);
lean_dec_ref(v_inst_279_);
return v_res_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocSemiring___redArg(lean_object* v_inst_292_, lean_object* v_inst_293_, lean_object* v_inst_294_, lean_object* v_inst_295_){
_start:
{
lean_object* v___x_296_; lean_object* v___x_297_; 
v___x_296_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_292_, v_inst_294_, v_inst_295_);
v___x_297_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_297_, 0, v___x_296_);
lean_ctor_set(v___x_297_, 1, v_inst_293_);
return v___x_297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocSemiring(lean_object* v_R_298_, lean_object* v_S_299_, lean_object* v_f_300_, lean_object* v_hf_301_, lean_object* v_inst_302_, lean_object* v_inst_303_, lean_object* v_inst_304_, lean_object* v_inst_305_, lean_object* v_inst_306_, lean_object* v_zero_307_, lean_object* v_add_308_, lean_object* v_mul_309_, lean_object* v_nsmul_310_){
_start:
{
lean_object* v___x_311_; lean_object* v___x_312_; 
v___x_311_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_302_, v_inst_304_, v_inst_305_);
v___x_312_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_312_, 0, v___x_311_);
lean_ctor_set(v___x_312_, 1, v_inst_303_);
return v___x_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocSemiring___boxed(lean_object* v_R_313_, lean_object* v_S_314_, lean_object* v_f_315_, lean_object* v_hf_316_, lean_object* v_inst_317_, lean_object* v_inst_318_, lean_object* v_inst_319_, lean_object* v_inst_320_, lean_object* v_inst_321_, lean_object* v_zero_322_, lean_object* v_add_323_, lean_object* v_mul_324_, lean_object* v_nsmul_325_){
_start:
{
lean_object* v_res_326_; 
v_res_326_ = lp_mathlib_Function_Injective_nonUnitalNonAssocSemiring(v_R_313_, v_S_314_, v_f_315_, v_hf_316_, v_inst_317_, v_inst_318_, v_inst_319_, v_inst_320_, v_inst_321_, v_zero_322_, v_add_323_, v_mul_324_, v_nsmul_325_);
lean_dec_ref(v_inst_321_);
lean_dec(v_f_315_);
return v_res_326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalSemiring___redArg(lean_object* v_inst_327_, lean_object* v_inst_328_, lean_object* v_inst_329_, lean_object* v_inst_330_){
_start:
{
lean_object* v___x_331_; lean_object* v___x_332_; 
v___x_331_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_327_, v_inst_329_, v_inst_330_);
v___x_332_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_332_, 0, v___x_331_);
lean_ctor_set(v___x_332_, 1, v_inst_328_);
return v___x_332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalSemiring(lean_object* v_R_333_, lean_object* v_S_334_, lean_object* v_f_335_, lean_object* v_hf_336_, lean_object* v_inst_337_, lean_object* v_inst_338_, lean_object* v_inst_339_, lean_object* v_inst_340_, lean_object* v_inst_341_, lean_object* v_zero_342_, lean_object* v_add_343_, lean_object* v_mul_344_, lean_object* v_nsmul_345_){
_start:
{
lean_object* v___x_346_; lean_object* v___x_347_; 
v___x_346_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_337_, v_inst_339_, v_inst_340_);
v___x_347_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_347_, 0, v___x_346_);
lean_ctor_set(v___x_347_, 1, v_inst_338_);
return v___x_347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalSemiring___boxed(lean_object* v_R_348_, lean_object* v_S_349_, lean_object* v_f_350_, lean_object* v_hf_351_, lean_object* v_inst_352_, lean_object* v_inst_353_, lean_object* v_inst_354_, lean_object* v_inst_355_, lean_object* v_inst_356_, lean_object* v_zero_357_, lean_object* v_add_358_, lean_object* v_mul_359_, lean_object* v_nsmul_360_){
_start:
{
lean_object* v_res_361_; 
v_res_361_ = lp_mathlib_Function_Injective_nonUnitalSemiring(v_R_348_, v_S_349_, v_f_350_, v_hf_351_, v_inst_352_, v_inst_353_, v_inst_354_, v_inst_355_, v_inst_356_, v_zero_357_, v_add_358_, v_mul_359_, v_nsmul_360_);
lean_dec_ref(v_inst_356_);
lean_dec(v_f_350_);
return v_res_361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocSemiring___redArg(lean_object* v_inst_362_, lean_object* v_inst_363_, lean_object* v_inst_364_, lean_object* v_inst_365_, lean_object* v_inst_366_, lean_object* v_inst_367_){
_start:
{
lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; 
v___x_368_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_368_, 0, lean_box(0));
lean_closure_set(v___x_368_, 1, v_inst_367_);
v___x_369_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_362_, v_inst_364_, v_inst_366_);
v___x_370_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_370_, 0, v___x_369_);
lean_ctor_set(v___x_370_, 1, v_inst_363_);
v___x_371_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_371_, 0, v___x_370_);
lean_ctor_set(v___x_371_, 1, v_inst_365_);
lean_ctor_set(v___x_371_, 2, v___x_368_);
return v___x_371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocSemiring(lean_object* v_R_372_, lean_object* v_S_373_, lean_object* v_f_374_, lean_object* v_hf_375_, lean_object* v_inst_376_, lean_object* v_inst_377_, lean_object* v_inst_378_, lean_object* v_inst_379_, lean_object* v_inst_380_, lean_object* v_inst_381_, lean_object* v_inst_382_, lean_object* v_zero_383_, lean_object* v_one_384_, lean_object* v_add_385_, lean_object* v_mul_386_, lean_object* v_nsmul_387_, lean_object* v_natCast_388_){
_start:
{
lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; 
v___x_389_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_389_, 0, lean_box(0));
lean_closure_set(v___x_389_, 1, v_inst_381_);
v___x_390_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_376_, v_inst_378_, v_inst_380_);
v___x_391_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_391_, 0, v___x_390_);
lean_ctor_set(v___x_391_, 1, v_inst_377_);
v___x_392_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_392_, 0, v___x_391_);
lean_ctor_set(v___x_392_, 1, v_inst_379_);
lean_ctor_set(v___x_392_, 2, v___x_389_);
return v___x_392_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocSemiring___boxed(lean_object** _args){
lean_object* v_R_393_ = _args[0];
lean_object* v_S_394_ = _args[1];
lean_object* v_f_395_ = _args[2];
lean_object* v_hf_396_ = _args[3];
lean_object* v_inst_397_ = _args[4];
lean_object* v_inst_398_ = _args[5];
lean_object* v_inst_399_ = _args[6];
lean_object* v_inst_400_ = _args[7];
lean_object* v_inst_401_ = _args[8];
lean_object* v_inst_402_ = _args[9];
lean_object* v_inst_403_ = _args[10];
lean_object* v_zero_404_ = _args[11];
lean_object* v_one_405_ = _args[12];
lean_object* v_add_406_ = _args[13];
lean_object* v_mul_407_ = _args[14];
lean_object* v_nsmul_408_ = _args[15];
lean_object* v_natCast_409_ = _args[16];
_start:
{
lean_object* v_res_410_; 
v_res_410_ = lp_mathlib_Function_Injective_nonAssocSemiring(v_R_393_, v_S_394_, v_f_395_, v_hf_396_, v_inst_397_, v_inst_398_, v_inst_399_, v_inst_400_, v_inst_401_, v_inst_402_, v_inst_403_, v_zero_404_, v_one_405_, v_add_406_, v_mul_407_, v_nsmul_408_, v_natCast_409_);
lean_dec_ref(v_inst_403_);
lean_dec(v_f_395_);
return v_res_410_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semiring___redArg___lam__0(lean_object* v_inst_411_, lean_object* v_n_412_, lean_object* v_x_413_){
_start:
{
lean_object* v___x_414_; 
v___x_414_ = lean_apply_2(v_inst_411_, v_x_413_, v_n_412_);
return v___x_414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semiring___redArg(lean_object* v_inst_415_, lean_object* v_inst_416_, lean_object* v_inst_417_, lean_object* v_inst_418_, lean_object* v_inst_419_, lean_object* v_inst_420_, lean_object* v_inst_421_){
_start:
{
lean_object* v___f_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; 
v___f_422_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_422_, 0, v_inst_420_);
v___x_423_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_423_, 0, lean_box(0));
lean_closure_set(v___x_423_, 1, v_inst_421_);
v___x_424_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_415_, v_inst_417_, v_inst_419_);
v___x_425_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_425_, 0, v_inst_418_);
lean_ctor_set(v___x_425_, 1, v_inst_416_);
lean_ctor_set(v___x_425_, 2, v___f_422_);
v___x_426_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_426_, 0, v___x_424_);
lean_ctor_set(v___x_426_, 1, v___x_425_);
lean_ctor_set(v___x_426_, 2, v___x_423_);
return v___x_426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semiring(lean_object* v_R_427_, lean_object* v_S_428_, lean_object* v_f_429_, lean_object* v_hf_430_, lean_object* v_inst_431_, lean_object* v_inst_432_, lean_object* v_inst_433_, lean_object* v_inst_434_, lean_object* v_inst_435_, lean_object* v_inst_436_, lean_object* v_inst_437_, lean_object* v_inst_438_, lean_object* v_zero_439_, lean_object* v_one_440_, lean_object* v_add_441_, lean_object* v_mul_442_, lean_object* v_nsmul_443_, lean_object* v_npow_444_, lean_object* v_natCast_445_){
_start:
{
lean_object* v___f_446_; lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; 
v___f_446_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_446_, 0, v_inst_436_);
v___x_447_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_447_, 0, lean_box(0));
lean_closure_set(v___x_447_, 1, v_inst_437_);
v___x_448_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_431_, v_inst_433_, v_inst_435_);
v___x_449_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_449_, 0, v_inst_434_);
lean_ctor_set(v___x_449_, 1, v_inst_432_);
lean_ctor_set(v___x_449_, 2, v___f_446_);
v___x_450_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_450_, 0, v___x_448_);
lean_ctor_set(v___x_450_, 1, v___x_449_);
lean_ctor_set(v___x_450_, 2, v___x_447_);
return v___x_450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semiring___boxed(lean_object** _args){
lean_object* v_R_451_ = _args[0];
lean_object* v_S_452_ = _args[1];
lean_object* v_f_453_ = _args[2];
lean_object* v_hf_454_ = _args[3];
lean_object* v_inst_455_ = _args[4];
lean_object* v_inst_456_ = _args[5];
lean_object* v_inst_457_ = _args[6];
lean_object* v_inst_458_ = _args[7];
lean_object* v_inst_459_ = _args[8];
lean_object* v_inst_460_ = _args[9];
lean_object* v_inst_461_ = _args[10];
lean_object* v_inst_462_ = _args[11];
lean_object* v_zero_463_ = _args[12];
lean_object* v_one_464_ = _args[13];
lean_object* v_add_465_ = _args[14];
lean_object* v_mul_466_ = _args[15];
lean_object* v_nsmul_467_ = _args[16];
lean_object* v_npow_468_ = _args[17];
lean_object* v_natCast_469_ = _args[18];
_start:
{
lean_object* v_res_470_; 
v_res_470_ = lp_mathlib_Function_Injective_semiring(v_R_451_, v_S_452_, v_f_453_, v_hf_454_, v_inst_455_, v_inst_456_, v_inst_457_, v_inst_458_, v_inst_459_, v_inst_460_, v_inst_461_, v_inst_462_, v_zero_463_, v_one_464_, v_add_465_, v_mul_466_, v_nsmul_467_, v_npow_468_, v_natCast_469_);
lean_dec_ref(v_inst_462_);
lean_dec(v_f_453_);
return v_res_470_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocRing___redArg(lean_object* v_inst_471_, lean_object* v_inst_472_, lean_object* v_inst_473_, lean_object* v_inst_474_, lean_object* v_inst_475_, lean_object* v_inst_476_, lean_object* v_inst_477_){
_start:
{
lean_object* v___x_478_; lean_object* v___x_479_; 
v___x_478_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_471_, v_inst_473_, v_inst_476_, v_inst_474_, v_inst_475_, v_inst_477_);
v___x_479_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_479_, 0, v___x_478_);
lean_ctor_set(v___x_479_, 1, v_inst_472_);
return v___x_479_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocRing(lean_object* v_R_480_, lean_object* v_S_481_, lean_object* v_inst_482_, lean_object* v_inst_483_, lean_object* v_inst_484_, lean_object* v_inst_485_, lean_object* v_inst_486_, lean_object* v_inst_487_, lean_object* v_inst_488_, lean_object* v_inst_489_, lean_object* v_f_490_, lean_object* v_hf_491_, lean_object* v_zero_492_, lean_object* v_add_493_, lean_object* v_mul_494_, lean_object* v_neg_495_, lean_object* v_sub_496_, lean_object* v_nsmul_497_, lean_object* v_zsmul_498_){
_start:
{
lean_object* v___x_499_; lean_object* v___x_500_; 
v___x_499_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_482_, v_inst_484_, v_inst_487_, v_inst_485_, v_inst_486_, v_inst_488_);
v___x_500_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_500_, 0, v___x_499_);
lean_ctor_set(v___x_500_, 1, v_inst_483_);
return v___x_500_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocRing___boxed(lean_object** _args){
lean_object* v_R_501_ = _args[0];
lean_object* v_S_502_ = _args[1];
lean_object* v_inst_503_ = _args[2];
lean_object* v_inst_504_ = _args[3];
lean_object* v_inst_505_ = _args[4];
lean_object* v_inst_506_ = _args[5];
lean_object* v_inst_507_ = _args[6];
lean_object* v_inst_508_ = _args[7];
lean_object* v_inst_509_ = _args[8];
lean_object* v_inst_510_ = _args[9];
lean_object* v_f_511_ = _args[10];
lean_object* v_hf_512_ = _args[11];
lean_object* v_zero_513_ = _args[12];
lean_object* v_add_514_ = _args[13];
lean_object* v_mul_515_ = _args[14];
lean_object* v_neg_516_ = _args[15];
lean_object* v_sub_517_ = _args[16];
lean_object* v_nsmul_518_ = _args[17];
lean_object* v_zsmul_519_ = _args[18];
_start:
{
lean_object* v_res_520_; 
v_res_520_ = lp_mathlib_Function_Injective_nonUnitalNonAssocRing(v_R_501_, v_S_502_, v_inst_503_, v_inst_504_, v_inst_505_, v_inst_506_, v_inst_507_, v_inst_508_, v_inst_509_, v_inst_510_, v_f_511_, v_hf_512_, v_zero_513_, v_add_514_, v_mul_515_, v_neg_516_, v_sub_517_, v_nsmul_518_, v_zsmul_519_);
lean_dec(v_f_511_);
lean_dec_ref(v_inst_510_);
return v_res_520_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalRing___redArg(lean_object* v_inst_521_, lean_object* v_inst_522_, lean_object* v_inst_523_, lean_object* v_inst_524_, lean_object* v_inst_525_, lean_object* v_inst_526_, lean_object* v_inst_527_){
_start:
{
lean_object* v___x_528_; lean_object* v___x_529_; 
v___x_528_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_521_, v_inst_523_, v_inst_526_, v_inst_524_, v_inst_525_, v_inst_527_);
v___x_529_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_529_, 0, v___x_528_);
lean_ctor_set(v___x_529_, 1, v_inst_522_);
return v___x_529_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalRing(lean_object* v_R_530_, lean_object* v_S_531_, lean_object* v_f_532_, lean_object* v_hf_533_, lean_object* v_inst_534_, lean_object* v_inst_535_, lean_object* v_inst_536_, lean_object* v_inst_537_, lean_object* v_inst_538_, lean_object* v_inst_539_, lean_object* v_inst_540_, lean_object* v_inst_541_, lean_object* v_zero_542_, lean_object* v_add_543_, lean_object* v_mul_544_, lean_object* v_neg_545_, lean_object* v_sub_546_, lean_object* v_nsmul_547_, lean_object* v_zsmul_548_){
_start:
{
lean_object* v___x_549_; lean_object* v___x_550_; 
v___x_549_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_534_, v_inst_536_, v_inst_539_, v_inst_537_, v_inst_538_, v_inst_540_);
v___x_550_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_550_, 0, v___x_549_);
lean_ctor_set(v___x_550_, 1, v_inst_535_);
return v___x_550_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalRing___boxed(lean_object** _args){
lean_object* v_R_551_ = _args[0];
lean_object* v_S_552_ = _args[1];
lean_object* v_f_553_ = _args[2];
lean_object* v_hf_554_ = _args[3];
lean_object* v_inst_555_ = _args[4];
lean_object* v_inst_556_ = _args[5];
lean_object* v_inst_557_ = _args[6];
lean_object* v_inst_558_ = _args[7];
lean_object* v_inst_559_ = _args[8];
lean_object* v_inst_560_ = _args[9];
lean_object* v_inst_561_ = _args[10];
lean_object* v_inst_562_ = _args[11];
lean_object* v_zero_563_ = _args[12];
lean_object* v_add_564_ = _args[13];
lean_object* v_mul_565_ = _args[14];
lean_object* v_neg_566_ = _args[15];
lean_object* v_sub_567_ = _args[16];
lean_object* v_nsmul_568_ = _args[17];
lean_object* v_zsmul_569_ = _args[18];
_start:
{
lean_object* v_res_570_; 
v_res_570_ = lp_mathlib_Function_Injective_nonUnitalRing(v_R_551_, v_S_552_, v_f_553_, v_hf_554_, v_inst_555_, v_inst_556_, v_inst_557_, v_inst_558_, v_inst_559_, v_inst_560_, v_inst_561_, v_inst_562_, v_zero_563_, v_add_564_, v_mul_565_, v_neg_566_, v_sub_567_, v_nsmul_568_, v_zsmul_569_);
lean_dec_ref(v_inst_562_);
lean_dec(v_f_553_);
return v_res_570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocRing___redArg(lean_object* v_inst_571_, lean_object* v_inst_572_, lean_object* v_inst_573_, lean_object* v_inst_574_, lean_object* v_inst_575_, lean_object* v_inst_576_, lean_object* v_inst_577_, lean_object* v_inst_578_, lean_object* v_inst_579_, lean_object* v_inst_580_){
_start:
{
lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v___x_585_; 
v___x_581_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_581_, 0, lean_box(0));
lean_closure_set(v___x_581_, 1, v_inst_579_);
v___x_582_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_582_, 0, lean_box(0));
lean_closure_set(v___x_582_, 1, v_inst_580_);
v___x_583_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_571_, v_inst_573_, v_inst_577_, v_inst_575_, v_inst_576_, v_inst_578_);
v___x_584_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_584_, 0, v___x_583_);
lean_ctor_set(v___x_584_, 1, v_inst_572_);
v___x_585_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_585_, 0, v___x_584_);
lean_ctor_set(v___x_585_, 1, v_inst_574_);
lean_ctor_set(v___x_585_, 2, v___x_581_);
lean_ctor_set(v___x_585_, 3, v___x_582_);
return v___x_585_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocRing(lean_object* v_R_586_, lean_object* v_S_587_, lean_object* v_f_588_, lean_object* v_hf_589_, lean_object* v_inst_590_, lean_object* v_inst_591_, lean_object* v_inst_592_, lean_object* v_inst_593_, lean_object* v_inst_594_, lean_object* v_inst_595_, lean_object* v_inst_596_, lean_object* v_inst_597_, lean_object* v_inst_598_, lean_object* v_inst_599_, lean_object* v_inst_600_, lean_object* v_zero_601_, lean_object* v_one_602_, lean_object* v_add_603_, lean_object* v_mul_604_, lean_object* v_neg_605_, lean_object* v_sub_606_, lean_object* v_nsmul_607_, lean_object* v_zsmul_608_, lean_object* v_natCast_609_, lean_object* v_intCast_610_){
_start:
{
lean_object* v___x_611_; lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; 
v___x_611_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_611_, 0, lean_box(0));
lean_closure_set(v___x_611_, 1, v_inst_598_);
v___x_612_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_612_, 0, lean_box(0));
lean_closure_set(v___x_612_, 1, v_inst_599_);
v___x_613_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_590_, v_inst_592_, v_inst_596_, v_inst_594_, v_inst_595_, v_inst_597_);
v___x_614_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_614_, 0, v___x_613_);
lean_ctor_set(v___x_614_, 1, v_inst_591_);
v___x_615_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_615_, 0, v___x_614_);
lean_ctor_set(v___x_615_, 1, v_inst_593_);
lean_ctor_set(v___x_615_, 2, v___x_611_);
lean_ctor_set(v___x_615_, 3, v___x_612_);
return v___x_615_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocRing___boxed(lean_object** _args){
lean_object* v_R_616_ = _args[0];
lean_object* v_S_617_ = _args[1];
lean_object* v_f_618_ = _args[2];
lean_object* v_hf_619_ = _args[3];
lean_object* v_inst_620_ = _args[4];
lean_object* v_inst_621_ = _args[5];
lean_object* v_inst_622_ = _args[6];
lean_object* v_inst_623_ = _args[7];
lean_object* v_inst_624_ = _args[8];
lean_object* v_inst_625_ = _args[9];
lean_object* v_inst_626_ = _args[10];
lean_object* v_inst_627_ = _args[11];
lean_object* v_inst_628_ = _args[12];
lean_object* v_inst_629_ = _args[13];
lean_object* v_inst_630_ = _args[14];
lean_object* v_zero_631_ = _args[15];
lean_object* v_one_632_ = _args[16];
lean_object* v_add_633_ = _args[17];
lean_object* v_mul_634_ = _args[18];
lean_object* v_neg_635_ = _args[19];
lean_object* v_sub_636_ = _args[20];
lean_object* v_nsmul_637_ = _args[21];
lean_object* v_zsmul_638_ = _args[22];
lean_object* v_natCast_639_ = _args[23];
lean_object* v_intCast_640_ = _args[24];
_start:
{
lean_object* v_res_641_; 
v_res_641_ = lp_mathlib_Function_Injective_nonAssocRing(v_R_616_, v_S_617_, v_f_618_, v_hf_619_, v_inst_620_, v_inst_621_, v_inst_622_, v_inst_623_, v_inst_624_, v_inst_625_, v_inst_626_, v_inst_627_, v_inst_628_, v_inst_629_, v_inst_630_, v_zero_631_, v_one_632_, v_add_633_, v_mul_634_, v_neg_635_, v_sub_636_, v_nsmul_637_, v_zsmul_638_, v_natCast_639_, v_intCast_640_);
lean_dec_ref(v_inst_630_);
lean_dec(v_f_618_);
return v_res_641_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_ring___redArg___lam__1(lean_object* v_inst_642_, lean_object* v_n_643_, lean_object* v_x_644_){
_start:
{
lean_object* v___x_645_; 
v___x_645_ = lean_apply_2(v_inst_642_, v_n_643_, v_x_644_);
return v___x_645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_ring___redArg(lean_object* v_inst_646_, lean_object* v_inst_647_, lean_object* v_inst_648_, lean_object* v_inst_649_, lean_object* v_inst_650_, lean_object* v_inst_651_, lean_object* v_inst_652_, lean_object* v_inst_653_, lean_object* v_inst_654_, lean_object* v_inst_655_, lean_object* v_inst_656_){
_start:
{
lean_object* v___x_657_; lean_object* v_toNeg_658_; lean_object* v_toSub_659_; lean_object* v___f_660_; lean_object* v___f_661_; lean_object* v___x_662_; lean_object* v___x_663_; lean_object* v___x_664_; lean_object* v___x_665_; lean_object* v___x_666_; lean_object* v___x_667_; 
lean_inc(v_inst_653_);
lean_inc(v_inst_652_);
lean_inc(v_inst_648_);
lean_inc(v_inst_646_);
v___x_657_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_646_, v_inst_648_, v_inst_652_, v_inst_650_, v_inst_651_, v_inst_653_);
v_toNeg_658_ = lean_ctor_get(v___x_657_, 1);
lean_inc(v_toNeg_658_);
v_toSub_659_ = lean_ctor_get(v___x_657_, 2);
lean_inc(v_toSub_659_);
lean_dec_ref(v___x_657_);
v___f_660_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_660_, 0, v_inst_654_);
v___f_661_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_ring___redArg___lam__1), 3, 1);
lean_closure_set(v___f_661_, 0, v_inst_653_);
v___x_662_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_662_, 0, lean_box(0));
lean_closure_set(v___x_662_, 1, v_inst_656_);
v___x_663_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_663_, 0, lean_box(0));
lean_closure_set(v___x_663_, 1, v_inst_655_);
v___x_664_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_646_, v_inst_648_, v_inst_652_);
v___x_665_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_665_, 0, v_inst_649_);
lean_ctor_set(v___x_665_, 1, v_inst_647_);
lean_ctor_set(v___x_665_, 2, v___f_660_);
v___x_666_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_666_, 0, v___x_664_);
lean_ctor_set(v___x_666_, 1, v___x_665_);
lean_ctor_set(v___x_666_, 2, v___x_663_);
v___x_667_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_667_, 0, v___x_666_);
lean_ctor_set(v___x_667_, 1, v_toNeg_658_);
lean_ctor_set(v___x_667_, 2, v_toSub_659_);
lean_ctor_set(v___x_667_, 3, v___f_661_);
lean_ctor_set(v___x_667_, 4, v___x_662_);
return v___x_667_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_ring(lean_object* v_R_668_, lean_object* v_S_669_, lean_object* v_f_670_, lean_object* v_hf_671_, lean_object* v_inst_672_, lean_object* v_inst_673_, lean_object* v_inst_674_, lean_object* v_inst_675_, lean_object* v_inst_676_, lean_object* v_inst_677_, lean_object* v_inst_678_, lean_object* v_inst_679_, lean_object* v_inst_680_, lean_object* v_inst_681_, lean_object* v_inst_682_, lean_object* v_inst_683_, lean_object* v_zero_684_, lean_object* v_one_685_, lean_object* v_add_686_, lean_object* v_mul_687_, lean_object* v_neg_688_, lean_object* v_sub_689_, lean_object* v_nsmul_690_, lean_object* v_zsmul_691_, lean_object* v_npow_692_, lean_object* v_natCast_693_, lean_object* v_intCast_694_){
_start:
{
lean_object* v___x_695_; lean_object* v_toNeg_696_; lean_object* v_toSub_697_; lean_object* v___f_698_; lean_object* v___f_699_; lean_object* v___x_700_; lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; 
lean_inc(v_inst_679_);
lean_inc(v_inst_678_);
lean_inc(v_inst_674_);
lean_inc(v_inst_672_);
v___x_695_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_672_, v_inst_674_, v_inst_678_, v_inst_676_, v_inst_677_, v_inst_679_);
v_toNeg_696_ = lean_ctor_get(v___x_695_, 1);
lean_inc(v_toNeg_696_);
v_toSub_697_ = lean_ctor_get(v___x_695_, 2);
lean_inc(v_toSub_697_);
lean_dec_ref(v___x_695_);
v___f_698_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_698_, 0, v_inst_680_);
v___f_699_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_ring___redArg___lam__1), 3, 1);
lean_closure_set(v___f_699_, 0, v_inst_679_);
v___x_700_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_700_, 0, lean_box(0));
lean_closure_set(v___x_700_, 1, v_inst_682_);
v___x_701_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_701_, 0, lean_box(0));
lean_closure_set(v___x_701_, 1, v_inst_681_);
v___x_702_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_672_, v_inst_674_, v_inst_678_);
v___x_703_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_703_, 0, v_inst_675_);
lean_ctor_set(v___x_703_, 1, v_inst_673_);
lean_ctor_set(v___x_703_, 2, v___f_698_);
v___x_704_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_704_, 0, v___x_702_);
lean_ctor_set(v___x_704_, 1, v___x_703_);
lean_ctor_set(v___x_704_, 2, v___x_701_);
v___x_705_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_705_, 0, v___x_704_);
lean_ctor_set(v___x_705_, 1, v_toNeg_696_);
lean_ctor_set(v___x_705_, 2, v_toSub_697_);
lean_ctor_set(v___x_705_, 3, v___f_699_);
lean_ctor_set(v___x_705_, 4, v___x_700_);
return v___x_705_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_ring___boxed(lean_object** _args){
lean_object* v_R_706_ = _args[0];
lean_object* v_S_707_ = _args[1];
lean_object* v_f_708_ = _args[2];
lean_object* v_hf_709_ = _args[3];
lean_object* v_inst_710_ = _args[4];
lean_object* v_inst_711_ = _args[5];
lean_object* v_inst_712_ = _args[6];
lean_object* v_inst_713_ = _args[7];
lean_object* v_inst_714_ = _args[8];
lean_object* v_inst_715_ = _args[9];
lean_object* v_inst_716_ = _args[10];
lean_object* v_inst_717_ = _args[11];
lean_object* v_inst_718_ = _args[12];
lean_object* v_inst_719_ = _args[13];
lean_object* v_inst_720_ = _args[14];
lean_object* v_inst_721_ = _args[15];
lean_object* v_zero_722_ = _args[16];
lean_object* v_one_723_ = _args[17];
lean_object* v_add_724_ = _args[18];
lean_object* v_mul_725_ = _args[19];
lean_object* v_neg_726_ = _args[20];
lean_object* v_sub_727_ = _args[21];
lean_object* v_nsmul_728_ = _args[22];
lean_object* v_zsmul_729_ = _args[23];
lean_object* v_npow_730_ = _args[24];
lean_object* v_natCast_731_ = _args[25];
lean_object* v_intCast_732_ = _args[26];
_start:
{
lean_object* v_res_733_; 
v_res_733_ = lp_mathlib_Function_Injective_ring(v_R_706_, v_S_707_, v_f_708_, v_hf_709_, v_inst_710_, v_inst_711_, v_inst_712_, v_inst_713_, v_inst_714_, v_inst_715_, v_inst_716_, v_inst_717_, v_inst_718_, v_inst_719_, v_inst_720_, v_inst_721_, v_zero_722_, v_one_723_, v_add_724_, v_mul_725_, v_neg_726_, v_sub_727_, v_nsmul_728_, v_zsmul_729_, v_npow_730_, v_natCast_731_, v_intCast_732_);
lean_dec_ref(v_inst_721_);
lean_dec(v_f_708_);
return v_res_733_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocCommSemiring___redArg(lean_object* v_inst_734_, lean_object* v_inst_735_, lean_object* v_inst_736_, lean_object* v_inst_737_){
_start:
{
lean_object* v___x_738_; lean_object* v___x_739_; 
v___x_738_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_734_, v_inst_736_, v_inst_737_);
v___x_739_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_739_, 0, v___x_738_);
lean_ctor_set(v___x_739_, 1, v_inst_735_);
return v___x_739_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocCommSemiring(lean_object* v_R_740_, lean_object* v_S_741_, lean_object* v_f_742_, lean_object* v_hf_743_, lean_object* v_inst_744_, lean_object* v_inst_745_, lean_object* v_inst_746_, lean_object* v_inst_747_, lean_object* v_inst_748_, lean_object* v_zero_749_, lean_object* v_add_750_, lean_object* v_mul_751_, lean_object* v_nsmul_752_){
_start:
{
lean_object* v___x_753_; lean_object* v___x_754_; 
v___x_753_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_744_, v_inst_746_, v_inst_747_);
v___x_754_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_754_, 0, v___x_753_);
lean_ctor_set(v___x_754_, 1, v_inst_745_);
return v___x_754_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocCommSemiring___boxed(lean_object* v_R_755_, lean_object* v_S_756_, lean_object* v_f_757_, lean_object* v_hf_758_, lean_object* v_inst_759_, lean_object* v_inst_760_, lean_object* v_inst_761_, lean_object* v_inst_762_, lean_object* v_inst_763_, lean_object* v_zero_764_, lean_object* v_add_765_, lean_object* v_mul_766_, lean_object* v_nsmul_767_){
_start:
{
lean_object* v_res_768_; 
v_res_768_ = lp_mathlib_Function_Injective_nonUnitalNonAssocCommSemiring(v_R_755_, v_S_756_, v_f_757_, v_hf_758_, v_inst_759_, v_inst_760_, v_inst_761_, v_inst_762_, v_inst_763_, v_zero_764_, v_add_765_, v_mul_766_, v_nsmul_767_);
lean_dec_ref(v_inst_763_);
lean_dec(v_f_757_);
return v_res_768_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalCommSemiring___redArg(lean_object* v_inst_769_, lean_object* v_inst_770_, lean_object* v_inst_771_, lean_object* v_inst_772_){
_start:
{
lean_object* v___x_773_; lean_object* v___x_774_; 
v___x_773_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_769_, v_inst_771_, v_inst_772_);
v___x_774_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_774_, 0, v___x_773_);
lean_ctor_set(v___x_774_, 1, v_inst_770_);
return v___x_774_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalCommSemiring(lean_object* v_R_775_, lean_object* v_S_776_, lean_object* v_inst_777_, lean_object* v_inst_778_, lean_object* v_inst_779_, lean_object* v_inst_780_, lean_object* v_inst_781_, lean_object* v_f_782_, lean_object* v_hf_783_, lean_object* v_zero_784_, lean_object* v_add_785_, lean_object* v_mul_786_, lean_object* v_nsmul_787_){
_start:
{
lean_object* v___x_788_; lean_object* v___x_789_; 
v___x_788_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_777_, v_inst_779_, v_inst_780_);
v___x_789_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_789_, 0, v___x_788_);
lean_ctor_set(v___x_789_, 1, v_inst_778_);
return v___x_789_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalCommSemiring___boxed(lean_object* v_R_790_, lean_object* v_S_791_, lean_object* v_inst_792_, lean_object* v_inst_793_, lean_object* v_inst_794_, lean_object* v_inst_795_, lean_object* v_inst_796_, lean_object* v_f_797_, lean_object* v_hf_798_, lean_object* v_zero_799_, lean_object* v_add_800_, lean_object* v_mul_801_, lean_object* v_nsmul_802_){
_start:
{
lean_object* v_res_803_; 
v_res_803_ = lp_mathlib_Function_Injective_nonUnitalCommSemiring(v_R_790_, v_S_791_, v_inst_792_, v_inst_793_, v_inst_794_, v_inst_795_, v_inst_796_, v_f_797_, v_hf_798_, v_zero_799_, v_add_800_, v_mul_801_, v_nsmul_802_);
lean_dec(v_f_797_);
lean_dec_ref(v_inst_796_);
return v_res_803_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocCommSemiring___redArg(lean_object* v_inst_804_, lean_object* v_inst_805_, lean_object* v_inst_806_, lean_object* v_inst_807_, lean_object* v_inst_808_, lean_object* v_inst_809_){
_start:
{
lean_object* v___x_810_; lean_object* v___x_811_; lean_object* v___x_812_; lean_object* v___x_813_; 
v___x_810_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_810_, 0, lean_box(0));
lean_closure_set(v___x_810_, 1, v_inst_809_);
v___x_811_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_804_, v_inst_806_, v_inst_808_);
v___x_812_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_812_, 0, v___x_811_);
lean_ctor_set(v___x_812_, 1, v_inst_805_);
v___x_813_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_813_, 0, v___x_812_);
lean_ctor_set(v___x_813_, 1, v_inst_807_);
lean_ctor_set(v___x_813_, 2, v___x_810_);
return v___x_813_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocCommSemiring(lean_object* v_R_814_, lean_object* v_S_815_, lean_object* v_inst_816_, lean_object* v_inst_817_, lean_object* v_inst_818_, lean_object* v_inst_819_, lean_object* v_inst_820_, lean_object* v_inst_821_, lean_object* v_inst_822_, lean_object* v_f_823_, lean_object* v_hf_824_, lean_object* v_zero_825_, lean_object* v_one_826_, lean_object* v_add_827_, lean_object* v_mul_828_, lean_object* v_nsmul_829_, lean_object* v_natCast_830_){
_start:
{
lean_object* v___x_831_; lean_object* v___x_832_; lean_object* v___x_833_; lean_object* v___x_834_; 
v___x_831_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_831_, 0, lean_box(0));
lean_closure_set(v___x_831_, 1, v_inst_821_);
v___x_832_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_816_, v_inst_818_, v_inst_820_);
v___x_833_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_833_, 0, v___x_832_);
lean_ctor_set(v___x_833_, 1, v_inst_817_);
v___x_834_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_834_, 0, v___x_833_);
lean_ctor_set(v___x_834_, 1, v_inst_819_);
lean_ctor_set(v___x_834_, 2, v___x_831_);
return v___x_834_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocCommSemiring___boxed(lean_object** _args){
lean_object* v_R_835_ = _args[0];
lean_object* v_S_836_ = _args[1];
lean_object* v_inst_837_ = _args[2];
lean_object* v_inst_838_ = _args[3];
lean_object* v_inst_839_ = _args[4];
lean_object* v_inst_840_ = _args[5];
lean_object* v_inst_841_ = _args[6];
lean_object* v_inst_842_ = _args[7];
lean_object* v_inst_843_ = _args[8];
lean_object* v_f_844_ = _args[9];
lean_object* v_hf_845_ = _args[10];
lean_object* v_zero_846_ = _args[11];
lean_object* v_one_847_ = _args[12];
lean_object* v_add_848_ = _args[13];
lean_object* v_mul_849_ = _args[14];
lean_object* v_nsmul_850_ = _args[15];
lean_object* v_natCast_851_ = _args[16];
_start:
{
lean_object* v_res_852_; 
v_res_852_ = lp_mathlib_Function_Injective_nonAssocCommSemiring(v_R_835_, v_S_836_, v_inst_837_, v_inst_838_, v_inst_839_, v_inst_840_, v_inst_841_, v_inst_842_, v_inst_843_, v_f_844_, v_hf_845_, v_zero_846_, v_one_847_, v_add_848_, v_mul_849_, v_nsmul_850_, v_natCast_851_);
lean_dec(v_f_844_);
lean_dec_ref(v_inst_843_);
return v_res_852_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commSemiring___redArg(lean_object* v_inst_853_, lean_object* v_inst_854_, lean_object* v_inst_855_, lean_object* v_inst_856_, lean_object* v_inst_857_, lean_object* v_inst_858_, lean_object* v_inst_859_){
_start:
{
lean_object* v___f_860_; lean_object* v___x_861_; lean_object* v___x_862_; lean_object* v___x_863_; lean_object* v___x_864_; 
v___f_860_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_860_, 0, v_inst_858_);
v___x_861_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_861_, 0, lean_box(0));
lean_closure_set(v___x_861_, 1, v_inst_859_);
v___x_862_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_853_, v_inst_855_, v_inst_857_);
v___x_863_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_863_, 0, v_inst_856_);
lean_ctor_set(v___x_863_, 1, v_inst_854_);
lean_ctor_set(v___x_863_, 2, v___f_860_);
v___x_864_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_864_, 0, v___x_862_);
lean_ctor_set(v___x_864_, 1, v___x_863_);
lean_ctor_set(v___x_864_, 2, v___x_861_);
return v___x_864_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commSemiring(lean_object* v_R_865_, lean_object* v_S_866_, lean_object* v_f_867_, lean_object* v_hf_868_, lean_object* v_inst_869_, lean_object* v_inst_870_, lean_object* v_inst_871_, lean_object* v_inst_872_, lean_object* v_inst_873_, lean_object* v_inst_874_, lean_object* v_inst_875_, lean_object* v_inst_876_, lean_object* v_zero_877_, lean_object* v_one_878_, lean_object* v_add_879_, lean_object* v_mul_880_, lean_object* v_nsmul_881_, lean_object* v_npow_882_, lean_object* v_natCast_883_){
_start:
{
lean_object* v___f_884_; lean_object* v___x_885_; lean_object* v___x_886_; lean_object* v___x_887_; lean_object* v___x_888_; 
v___f_884_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_884_, 0, v_inst_874_);
v___x_885_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_885_, 0, lean_box(0));
lean_closure_set(v___x_885_, 1, v_inst_875_);
v___x_886_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_869_, v_inst_871_, v_inst_873_);
v___x_887_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_887_, 0, v_inst_872_);
lean_ctor_set(v___x_887_, 1, v_inst_870_);
lean_ctor_set(v___x_887_, 2, v___f_884_);
v___x_888_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_888_, 0, v___x_886_);
lean_ctor_set(v___x_888_, 1, v___x_887_);
lean_ctor_set(v___x_888_, 2, v___x_885_);
return v___x_888_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commSemiring___boxed(lean_object** _args){
lean_object* v_R_889_ = _args[0];
lean_object* v_S_890_ = _args[1];
lean_object* v_f_891_ = _args[2];
lean_object* v_hf_892_ = _args[3];
lean_object* v_inst_893_ = _args[4];
lean_object* v_inst_894_ = _args[5];
lean_object* v_inst_895_ = _args[6];
lean_object* v_inst_896_ = _args[7];
lean_object* v_inst_897_ = _args[8];
lean_object* v_inst_898_ = _args[9];
lean_object* v_inst_899_ = _args[10];
lean_object* v_inst_900_ = _args[11];
lean_object* v_zero_901_ = _args[12];
lean_object* v_one_902_ = _args[13];
lean_object* v_add_903_ = _args[14];
lean_object* v_mul_904_ = _args[15];
lean_object* v_nsmul_905_ = _args[16];
lean_object* v_npow_906_ = _args[17];
lean_object* v_natCast_907_ = _args[18];
_start:
{
lean_object* v_res_908_; 
v_res_908_ = lp_mathlib_Function_Injective_commSemiring(v_R_889_, v_S_890_, v_f_891_, v_hf_892_, v_inst_893_, v_inst_894_, v_inst_895_, v_inst_896_, v_inst_897_, v_inst_898_, v_inst_899_, v_inst_900_, v_zero_901_, v_one_902_, v_add_903_, v_mul_904_, v_nsmul_905_, v_npow_906_, v_natCast_907_);
lean_dec_ref(v_inst_900_);
lean_dec(v_f_891_);
return v_res_908_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocCommRing___redArg(lean_object* v_inst_909_, lean_object* v_inst_910_, lean_object* v_inst_911_, lean_object* v_inst_912_, lean_object* v_inst_913_, lean_object* v_inst_914_, lean_object* v_inst_915_){
_start:
{
lean_object* v___x_916_; lean_object* v___x_917_; 
v___x_916_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_909_, v_inst_911_, v_inst_914_, v_inst_912_, v_inst_913_, v_inst_915_);
v___x_917_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_917_, 0, v___x_916_);
lean_ctor_set(v___x_917_, 1, v_inst_910_);
return v___x_917_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocCommRing(lean_object* v_R_918_, lean_object* v_S_919_, lean_object* v_inst_920_, lean_object* v_inst_921_, lean_object* v_inst_922_, lean_object* v_inst_923_, lean_object* v_inst_924_, lean_object* v_inst_925_, lean_object* v_inst_926_, lean_object* v_inst_927_, lean_object* v_f_928_, lean_object* v_hf_929_, lean_object* v_zero_930_, lean_object* v_add_931_, lean_object* v_mul_932_, lean_object* v_neg_933_, lean_object* v_sub_934_, lean_object* v_nsmul_935_, lean_object* v_zsmul_936_){
_start:
{
lean_object* v___x_937_; lean_object* v___x_938_; 
v___x_937_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_920_, v_inst_922_, v_inst_925_, v_inst_923_, v_inst_924_, v_inst_926_);
v___x_938_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_938_, 0, v___x_937_);
lean_ctor_set(v___x_938_, 1, v_inst_921_);
return v___x_938_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalNonAssocCommRing___boxed(lean_object** _args){
lean_object* v_R_939_ = _args[0];
lean_object* v_S_940_ = _args[1];
lean_object* v_inst_941_ = _args[2];
lean_object* v_inst_942_ = _args[3];
lean_object* v_inst_943_ = _args[4];
lean_object* v_inst_944_ = _args[5];
lean_object* v_inst_945_ = _args[6];
lean_object* v_inst_946_ = _args[7];
lean_object* v_inst_947_ = _args[8];
lean_object* v_inst_948_ = _args[9];
lean_object* v_f_949_ = _args[10];
lean_object* v_hf_950_ = _args[11];
lean_object* v_zero_951_ = _args[12];
lean_object* v_add_952_ = _args[13];
lean_object* v_mul_953_ = _args[14];
lean_object* v_neg_954_ = _args[15];
lean_object* v_sub_955_ = _args[16];
lean_object* v_nsmul_956_ = _args[17];
lean_object* v_zsmul_957_ = _args[18];
_start:
{
lean_object* v_res_958_; 
v_res_958_ = lp_mathlib_Function_Injective_nonUnitalNonAssocCommRing(v_R_939_, v_S_940_, v_inst_941_, v_inst_942_, v_inst_943_, v_inst_944_, v_inst_945_, v_inst_946_, v_inst_947_, v_inst_948_, v_f_949_, v_hf_950_, v_zero_951_, v_add_952_, v_mul_953_, v_neg_954_, v_sub_955_, v_nsmul_956_, v_zsmul_957_);
lean_dec(v_f_949_);
lean_dec_ref(v_inst_948_);
return v_res_958_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalCommRing___redArg(lean_object* v_inst_959_, lean_object* v_inst_960_, lean_object* v_inst_961_, lean_object* v_inst_962_, lean_object* v_inst_963_, lean_object* v_inst_964_, lean_object* v_inst_965_){
_start:
{
lean_object* v___x_966_; lean_object* v___x_967_; 
v___x_966_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_959_, v_inst_961_, v_inst_964_, v_inst_962_, v_inst_963_, v_inst_965_);
v___x_967_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_967_, 0, v___x_966_);
lean_ctor_set(v___x_967_, 1, v_inst_960_);
return v___x_967_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalCommRing(lean_object* v_R_968_, lean_object* v_S_969_, lean_object* v_inst_970_, lean_object* v_inst_971_, lean_object* v_inst_972_, lean_object* v_inst_973_, lean_object* v_inst_974_, lean_object* v_inst_975_, lean_object* v_inst_976_, lean_object* v_inst_977_, lean_object* v_f_978_, lean_object* v_hf_979_, lean_object* v_zero_980_, lean_object* v_add_981_, lean_object* v_mul_982_, lean_object* v_neg_983_, lean_object* v_sub_984_, lean_object* v_nsmul_985_, lean_object* v_zsmul_986_){
_start:
{
lean_object* v___x_987_; lean_object* v___x_988_; 
v___x_987_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_970_, v_inst_972_, v_inst_975_, v_inst_973_, v_inst_974_, v_inst_976_);
v___x_988_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_988_, 0, v___x_987_);
lean_ctor_set(v___x_988_, 1, v_inst_971_);
return v___x_988_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonUnitalCommRing___boxed(lean_object** _args){
lean_object* v_R_989_ = _args[0];
lean_object* v_S_990_ = _args[1];
lean_object* v_inst_991_ = _args[2];
lean_object* v_inst_992_ = _args[3];
lean_object* v_inst_993_ = _args[4];
lean_object* v_inst_994_ = _args[5];
lean_object* v_inst_995_ = _args[6];
lean_object* v_inst_996_ = _args[7];
lean_object* v_inst_997_ = _args[8];
lean_object* v_inst_998_ = _args[9];
lean_object* v_f_999_ = _args[10];
lean_object* v_hf_1000_ = _args[11];
lean_object* v_zero_1001_ = _args[12];
lean_object* v_add_1002_ = _args[13];
lean_object* v_mul_1003_ = _args[14];
lean_object* v_neg_1004_ = _args[15];
lean_object* v_sub_1005_ = _args[16];
lean_object* v_nsmul_1006_ = _args[17];
lean_object* v_zsmul_1007_ = _args[18];
_start:
{
lean_object* v_res_1008_; 
v_res_1008_ = lp_mathlib_Function_Injective_nonUnitalCommRing(v_R_989_, v_S_990_, v_inst_991_, v_inst_992_, v_inst_993_, v_inst_994_, v_inst_995_, v_inst_996_, v_inst_997_, v_inst_998_, v_f_999_, v_hf_1000_, v_zero_1001_, v_add_1002_, v_mul_1003_, v_neg_1004_, v_sub_1005_, v_nsmul_1006_, v_zsmul_1007_);
lean_dec(v_f_999_);
lean_dec_ref(v_inst_998_);
return v_res_1008_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocCommRing___redArg(lean_object* v_inst_1009_, lean_object* v_inst_1010_, lean_object* v_inst_1011_, lean_object* v_inst_1012_, lean_object* v_inst_1013_, lean_object* v_inst_1014_, lean_object* v_inst_1015_, lean_object* v_inst_1016_, lean_object* v_inst_1017_, lean_object* v_inst_1018_){
_start:
{
lean_object* v___x_1019_; lean_object* v___x_1020_; lean_object* v___x_1021_; lean_object* v___x_1022_; lean_object* v___x_1023_; 
v___x_1019_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_1019_, 0, lean_box(0));
lean_closure_set(v___x_1019_, 1, v_inst_1017_);
v___x_1020_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_1020_, 0, lean_box(0));
lean_closure_set(v___x_1020_, 1, v_inst_1018_);
v___x_1021_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_1009_, v_inst_1011_, v_inst_1015_, v_inst_1013_, v_inst_1014_, v_inst_1016_);
v___x_1022_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1022_, 0, v___x_1021_);
lean_ctor_set(v___x_1022_, 1, v_inst_1010_);
v___x_1023_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1023_, 0, v___x_1022_);
lean_ctor_set(v___x_1023_, 1, v_inst_1012_);
lean_ctor_set(v___x_1023_, 2, v___x_1019_);
lean_ctor_set(v___x_1023_, 3, v___x_1020_);
return v___x_1023_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocCommRing(lean_object* v_R_1024_, lean_object* v_S_1025_, lean_object* v_inst_1026_, lean_object* v_inst_1027_, lean_object* v_inst_1028_, lean_object* v_inst_1029_, lean_object* v_inst_1030_, lean_object* v_inst_1031_, lean_object* v_inst_1032_, lean_object* v_inst_1033_, lean_object* v_inst_1034_, lean_object* v_inst_1035_, lean_object* v_inst_1036_, lean_object* v_f_1037_, lean_object* v_hf_1038_, lean_object* v_zero_1039_, lean_object* v_one_1040_, lean_object* v_add_1041_, lean_object* v_mul_1042_, lean_object* v_neg_1043_, lean_object* v_sub_1044_, lean_object* v_nsmul_1045_, lean_object* v_zsmul_1046_, lean_object* v_natCast_1047_, lean_object* v_intCast_1048_){
_start:
{
lean_object* v___x_1049_; lean_object* v___x_1050_; lean_object* v___x_1051_; lean_object* v___x_1052_; lean_object* v___x_1053_; 
v___x_1049_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_1049_, 0, lean_box(0));
lean_closure_set(v___x_1049_, 1, v_inst_1034_);
v___x_1050_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_1050_, 0, lean_box(0));
lean_closure_set(v___x_1050_, 1, v_inst_1035_);
v___x_1051_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_1026_, v_inst_1028_, v_inst_1032_, v_inst_1030_, v_inst_1031_, v_inst_1033_);
v___x_1052_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1052_, 0, v___x_1051_);
lean_ctor_set(v___x_1052_, 1, v_inst_1027_);
v___x_1053_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1053_, 0, v___x_1052_);
lean_ctor_set(v___x_1053_, 1, v_inst_1029_);
lean_ctor_set(v___x_1053_, 2, v___x_1049_);
lean_ctor_set(v___x_1053_, 3, v___x_1050_);
return v___x_1053_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_nonAssocCommRing___boxed(lean_object** _args){
lean_object* v_R_1054_ = _args[0];
lean_object* v_S_1055_ = _args[1];
lean_object* v_inst_1056_ = _args[2];
lean_object* v_inst_1057_ = _args[3];
lean_object* v_inst_1058_ = _args[4];
lean_object* v_inst_1059_ = _args[5];
lean_object* v_inst_1060_ = _args[6];
lean_object* v_inst_1061_ = _args[7];
lean_object* v_inst_1062_ = _args[8];
lean_object* v_inst_1063_ = _args[9];
lean_object* v_inst_1064_ = _args[10];
lean_object* v_inst_1065_ = _args[11];
lean_object* v_inst_1066_ = _args[12];
lean_object* v_f_1067_ = _args[13];
lean_object* v_hf_1068_ = _args[14];
lean_object* v_zero_1069_ = _args[15];
lean_object* v_one_1070_ = _args[16];
lean_object* v_add_1071_ = _args[17];
lean_object* v_mul_1072_ = _args[18];
lean_object* v_neg_1073_ = _args[19];
lean_object* v_sub_1074_ = _args[20];
lean_object* v_nsmul_1075_ = _args[21];
lean_object* v_zsmul_1076_ = _args[22];
lean_object* v_natCast_1077_ = _args[23];
lean_object* v_intCast_1078_ = _args[24];
_start:
{
lean_object* v_res_1079_; 
v_res_1079_ = lp_mathlib_Function_Injective_nonAssocCommRing(v_R_1054_, v_S_1055_, v_inst_1056_, v_inst_1057_, v_inst_1058_, v_inst_1059_, v_inst_1060_, v_inst_1061_, v_inst_1062_, v_inst_1063_, v_inst_1064_, v_inst_1065_, v_inst_1066_, v_f_1067_, v_hf_1068_, v_zero_1069_, v_one_1070_, v_add_1071_, v_mul_1072_, v_neg_1073_, v_sub_1074_, v_nsmul_1075_, v_zsmul_1076_, v_natCast_1077_, v_intCast_1078_);
lean_dec(v_f_1067_);
lean_dec_ref(v_inst_1066_);
return v_res_1079_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commRing___redArg(lean_object* v_inst_1080_, lean_object* v_inst_1081_, lean_object* v_inst_1082_, lean_object* v_inst_1083_, lean_object* v_inst_1084_, lean_object* v_inst_1085_, lean_object* v_inst_1086_, lean_object* v_inst_1087_, lean_object* v_inst_1088_, lean_object* v_inst_1089_, lean_object* v_inst_1090_){
_start:
{
lean_object* v___x_1091_; lean_object* v_toNeg_1092_; lean_object* v_toSub_1093_; lean_object* v___f_1094_; lean_object* v___f_1095_; lean_object* v___x_1096_; lean_object* v___x_1097_; lean_object* v___x_1098_; lean_object* v___x_1099_; lean_object* v___x_1100_; lean_object* v___x_1101_; 
lean_inc(v_inst_1087_);
lean_inc(v_inst_1086_);
lean_inc(v_inst_1082_);
lean_inc(v_inst_1080_);
v___x_1091_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_1080_, v_inst_1082_, v_inst_1086_, v_inst_1084_, v_inst_1085_, v_inst_1087_);
v_toNeg_1092_ = lean_ctor_get(v___x_1091_, 1);
lean_inc(v_toNeg_1092_);
v_toSub_1093_ = lean_ctor_get(v___x_1091_, 2);
lean_inc(v_toSub_1093_);
lean_dec_ref(v___x_1091_);
v___f_1094_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1094_, 0, v_inst_1088_);
v___f_1095_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_ring___redArg___lam__1), 3, 1);
lean_closure_set(v___f_1095_, 0, v_inst_1087_);
v___x_1096_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_1096_, 0, lean_box(0));
lean_closure_set(v___x_1096_, 1, v_inst_1090_);
v___x_1097_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_1097_, 0, lean_box(0));
lean_closure_set(v___x_1097_, 1, v_inst_1089_);
v___x_1098_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_1080_, v_inst_1082_, v_inst_1086_);
v___x_1099_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1099_, 0, v_inst_1083_);
lean_ctor_set(v___x_1099_, 1, v_inst_1081_);
lean_ctor_set(v___x_1099_, 2, v___f_1094_);
v___x_1100_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1100_, 0, v___x_1098_);
lean_ctor_set(v___x_1100_, 1, v___x_1099_);
lean_ctor_set(v___x_1100_, 2, v___x_1097_);
v___x_1101_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1101_, 0, v___x_1100_);
lean_ctor_set(v___x_1101_, 1, v_toNeg_1092_);
lean_ctor_set(v___x_1101_, 2, v_toSub_1093_);
lean_ctor_set(v___x_1101_, 3, v___f_1095_);
lean_ctor_set(v___x_1101_, 4, v___x_1096_);
return v___x_1101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commRing(lean_object* v_R_1102_, lean_object* v_S_1103_, lean_object* v_f_1104_, lean_object* v_hf_1105_, lean_object* v_inst_1106_, lean_object* v_inst_1107_, lean_object* v_inst_1108_, lean_object* v_inst_1109_, lean_object* v_inst_1110_, lean_object* v_inst_1111_, lean_object* v_inst_1112_, lean_object* v_inst_1113_, lean_object* v_inst_1114_, lean_object* v_inst_1115_, lean_object* v_inst_1116_, lean_object* v_inst_1117_, lean_object* v_zero_1118_, lean_object* v_one_1119_, lean_object* v_add_1120_, lean_object* v_mul_1121_, lean_object* v_neg_1122_, lean_object* v_sub_1123_, lean_object* v_nsmul_1124_, lean_object* v_zsmul_1125_, lean_object* v_npow_1126_, lean_object* v_natCast_1127_, lean_object* v_intCast_1128_){
_start:
{
lean_object* v___x_1129_; lean_object* v_toNeg_1130_; lean_object* v_toSub_1131_; lean_object* v___f_1132_; lean_object* v___f_1133_; lean_object* v___x_1134_; lean_object* v___x_1135_; lean_object* v___x_1136_; lean_object* v___x_1137_; lean_object* v___x_1138_; lean_object* v___x_1139_; 
lean_inc(v_inst_1113_);
lean_inc(v_inst_1112_);
lean_inc(v_inst_1108_);
lean_inc(v_inst_1106_);
v___x_1129_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_1106_, v_inst_1108_, v_inst_1112_, v_inst_1110_, v_inst_1111_, v_inst_1113_);
v_toNeg_1130_ = lean_ctor_get(v___x_1129_, 1);
lean_inc(v_toNeg_1130_);
v_toSub_1131_ = lean_ctor_get(v___x_1129_, 2);
lean_inc(v_toSub_1131_);
lean_dec_ref(v___x_1129_);
v___f_1132_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1132_, 0, v_inst_1114_);
v___f_1133_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_ring___redArg___lam__1), 3, 1);
lean_closure_set(v___f_1133_, 0, v_inst_1113_);
v___x_1134_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_1134_, 0, lean_box(0));
lean_closure_set(v___x_1134_, 1, v_inst_1116_);
v___x_1135_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_1135_, 0, lean_box(0));
lean_closure_set(v___x_1135_, 1, v_inst_1115_);
v___x_1136_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_1106_, v_inst_1108_, v_inst_1112_);
v___x_1137_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1137_, 0, v_inst_1109_);
lean_ctor_set(v___x_1137_, 1, v_inst_1107_);
lean_ctor_set(v___x_1137_, 2, v___f_1132_);
v___x_1138_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1138_, 0, v___x_1136_);
lean_ctor_set(v___x_1138_, 1, v___x_1137_);
lean_ctor_set(v___x_1138_, 2, v___x_1135_);
v___x_1139_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1139_, 0, v___x_1138_);
lean_ctor_set(v___x_1139_, 1, v_toNeg_1130_);
lean_ctor_set(v___x_1139_, 2, v_toSub_1131_);
lean_ctor_set(v___x_1139_, 3, v___f_1133_);
lean_ctor_set(v___x_1139_, 4, v___x_1134_);
return v___x_1139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commRing___boxed(lean_object** _args){
lean_object* v_R_1140_ = _args[0];
lean_object* v_S_1141_ = _args[1];
lean_object* v_f_1142_ = _args[2];
lean_object* v_hf_1143_ = _args[3];
lean_object* v_inst_1144_ = _args[4];
lean_object* v_inst_1145_ = _args[5];
lean_object* v_inst_1146_ = _args[6];
lean_object* v_inst_1147_ = _args[7];
lean_object* v_inst_1148_ = _args[8];
lean_object* v_inst_1149_ = _args[9];
lean_object* v_inst_1150_ = _args[10];
lean_object* v_inst_1151_ = _args[11];
lean_object* v_inst_1152_ = _args[12];
lean_object* v_inst_1153_ = _args[13];
lean_object* v_inst_1154_ = _args[14];
lean_object* v_inst_1155_ = _args[15];
lean_object* v_zero_1156_ = _args[16];
lean_object* v_one_1157_ = _args[17];
lean_object* v_add_1158_ = _args[18];
lean_object* v_mul_1159_ = _args[19];
lean_object* v_neg_1160_ = _args[20];
lean_object* v_sub_1161_ = _args[21];
lean_object* v_nsmul_1162_ = _args[22];
lean_object* v_zsmul_1163_ = _args[23];
lean_object* v_npow_1164_ = _args[24];
lean_object* v_natCast_1165_ = _args[25];
lean_object* v_intCast_1166_ = _args[26];
_start:
{
lean_object* v_res_1167_; 
v_res_1167_ = lp_mathlib_Function_Injective_commRing(v_R_1140_, v_S_1141_, v_f_1142_, v_hf_1143_, v_inst_1144_, v_inst_1145_, v_inst_1146_, v_inst_1147_, v_inst_1148_, v_inst_1149_, v_inst_1150_, v_inst_1151_, v_inst_1152_, v_inst_1153_, v_inst_1154_, v_inst_1155_, v_zero_1156_, v_one_1157_, v_add_1158_, v_mul_1159_, v_neg_1160_, v_sub_1161_, v_nsmul_1162_, v_zsmul_1163_, v_npow_1164_, v_natCast_1165_, v_intCast_1166_);
lean_dec_ref(v_inst_1155_);
lean_dec(v_f_1142_);
return v_res_1167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distrib___redArg(lean_object* v_inst_1168_, lean_object* v_inst_1169_){
_start:
{
lean_object* v___x_1170_; 
v___x_1170_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1170_, 0, v_inst_1169_);
lean_ctor_set(v___x_1170_, 1, v_inst_1168_);
return v___x_1170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distrib(lean_object* v_R_1171_, lean_object* v_S_1172_, lean_object* v_f_1173_, lean_object* v_hf_1174_, lean_object* v_inst_1175_, lean_object* v_inst_1176_, lean_object* v_inst_1177_, lean_object* v_add_1178_, lean_object* v_mul_1179_){
_start:
{
lean_object* v___x_1180_; 
v___x_1180_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1180_, 0, v_inst_1176_);
lean_ctor_set(v___x_1180_, 1, v_inst_1175_);
return v___x_1180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_distrib___boxed(lean_object* v_R_1181_, lean_object* v_S_1182_, lean_object* v_f_1183_, lean_object* v_hf_1184_, lean_object* v_inst_1185_, lean_object* v_inst_1186_, lean_object* v_inst_1187_, lean_object* v_add_1188_, lean_object* v_mul_1189_){
_start:
{
lean_object* v_res_1190_; 
v_res_1190_ = lp_mathlib_Function_Surjective_distrib(v_R_1181_, v_S_1182_, v_f_1183_, v_hf_1184_, v_inst_1185_, v_inst_1186_, v_inst_1187_, v_add_1188_, v_mul_1189_);
lean_dec_ref(v_inst_1187_);
lean_dec(v_f_1183_);
return v_res_1190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_hasDistribNeg___redArg(lean_object* v_inst_1191_){
_start:
{
lean_inc(v_inst_1191_);
return v_inst_1191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_hasDistribNeg___redArg___boxed(lean_object* v_inst_1192_){
_start:
{
lean_object* v_res_1193_; 
v_res_1193_ = lp_mathlib_Function_Surjective_hasDistribNeg___redArg(v_inst_1192_);
lean_dec(v_inst_1192_);
return v_res_1193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_hasDistribNeg(lean_object* v_R_1194_, lean_object* v_S_1195_, lean_object* v_f_1196_, lean_object* v_hf_1197_, lean_object* v_inst_1198_, lean_object* v_inst_1199_, lean_object* v_inst_1200_, lean_object* v_inst_1201_, lean_object* v_neg_1202_, lean_object* v_mul_1203_){
_start:
{
lean_inc(v_inst_1199_);
return v_inst_1199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_hasDistribNeg___boxed(lean_object* v_R_1204_, lean_object* v_S_1205_, lean_object* v_f_1206_, lean_object* v_hf_1207_, lean_object* v_inst_1208_, lean_object* v_inst_1209_, lean_object* v_inst_1210_, lean_object* v_inst_1211_, lean_object* v_neg_1212_, lean_object* v_mul_1213_){
_start:
{
lean_object* v_res_1214_; 
v_res_1214_ = lp_mathlib_Function_Surjective_hasDistribNeg(v_R_1204_, v_S_1205_, v_f_1206_, v_hf_1207_, v_inst_1208_, v_inst_1209_, v_inst_1210_, v_inst_1211_, v_neg_1212_, v_mul_1213_);
lean_dec(v_inst_1211_);
lean_dec(v_inst_1210_);
lean_dec(v_inst_1209_);
lean_dec(v_inst_1208_);
lean_dec(v_f_1206_);
return v_res_1214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addMonoidWithOne___redArg(lean_object* v_inst_1215_, lean_object* v_inst_1216_, lean_object* v_inst_1217_, lean_object* v_inst_1218_, lean_object* v_inst_1219_){
_start:
{
lean_object* v___x_1220_; lean_object* v___x_1221_; lean_object* v___x_1222_; 
v___x_1220_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_1215_, v_inst_1216_, v_inst_1218_);
v___x_1221_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_1221_, 0, lean_box(0));
lean_closure_set(v___x_1221_, 1, v_inst_1219_);
v___x_1222_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1222_, 0, v___x_1221_);
lean_ctor_set(v___x_1222_, 1, v___x_1220_);
lean_ctor_set(v___x_1222_, 2, v_inst_1217_);
return v___x_1222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addMonoidWithOne(lean_object* v_R_1223_, lean_object* v_S_1224_, lean_object* v_f_1225_, lean_object* v_hf_1226_, lean_object* v_inst_1227_, lean_object* v_inst_1228_, lean_object* v_inst_1229_, lean_object* v_inst_1230_, lean_object* v_inst_1231_, lean_object* v_inst_1232_, lean_object* v_zero_1233_, lean_object* v_one_1234_, lean_object* v_add_1235_, lean_object* v_nsmul_1236_, lean_object* v_natCast_1237_){
_start:
{
lean_object* v___x_1238_; lean_object* v___x_1239_; lean_object* v___x_1240_; 
v___x_1238_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_1227_, v_inst_1228_, v_inst_1230_);
v___x_1239_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_1239_, 0, lean_box(0));
lean_closure_set(v___x_1239_, 1, v_inst_1231_);
v___x_1240_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1240_, 0, v___x_1239_);
lean_ctor_set(v___x_1240_, 1, v___x_1238_);
lean_ctor_set(v___x_1240_, 2, v_inst_1229_);
return v___x_1240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addMonoidWithOne___boxed(lean_object* v_R_1241_, lean_object* v_S_1242_, lean_object* v_f_1243_, lean_object* v_hf_1244_, lean_object* v_inst_1245_, lean_object* v_inst_1246_, lean_object* v_inst_1247_, lean_object* v_inst_1248_, lean_object* v_inst_1249_, lean_object* v_inst_1250_, lean_object* v_zero_1251_, lean_object* v_one_1252_, lean_object* v_add_1253_, lean_object* v_nsmul_1254_, lean_object* v_natCast_1255_){
_start:
{
lean_object* v_res_1256_; 
v_res_1256_ = lp_mathlib_Function_Surjective_addMonoidWithOne(v_R_1241_, v_S_1242_, v_f_1243_, v_hf_1244_, v_inst_1245_, v_inst_1246_, v_inst_1247_, v_inst_1248_, v_inst_1249_, v_inst_1250_, v_zero_1251_, v_one_1252_, v_add_1253_, v_nsmul_1254_, v_natCast_1255_);
lean_dec_ref(v_inst_1250_);
lean_dec(v_f_1243_);
return v_res_1256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommMonoidWithOne___redArg(lean_object* v_inst_1257_, lean_object* v_inst_1258_, lean_object* v_inst_1259_, lean_object* v_inst_1260_, lean_object* v_inst_1261_){
_start:
{
lean_object* v___x_1262_; lean_object* v___x_1263_; lean_object* v___x_1264_; 
v___x_1262_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_1257_, v_inst_1258_, v_inst_1260_);
v___x_1263_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_1263_, 0, lean_box(0));
lean_closure_set(v___x_1263_, 1, v_inst_1261_);
v___x_1264_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1264_, 0, v___x_1263_);
lean_ctor_set(v___x_1264_, 1, v___x_1262_);
lean_ctor_set(v___x_1264_, 2, v_inst_1259_);
return v___x_1264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommMonoidWithOne(lean_object* v_R_1265_, lean_object* v_S_1266_, lean_object* v_f_1267_, lean_object* v_hf_1268_, lean_object* v_inst_1269_, lean_object* v_inst_1270_, lean_object* v_inst_1271_, lean_object* v_inst_1272_, lean_object* v_inst_1273_, lean_object* v_inst_1274_, lean_object* v_zero_1275_, lean_object* v_one_1276_, lean_object* v_add_1277_, lean_object* v_nsmul_1278_, lean_object* v_natCast_1279_){
_start:
{
lean_object* v___x_1280_; lean_object* v___x_1281_; lean_object* v___x_1282_; 
v___x_1280_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_1269_, v_inst_1270_, v_inst_1272_);
v___x_1281_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_1281_, 0, lean_box(0));
lean_closure_set(v___x_1281_, 1, v_inst_1273_);
v___x_1282_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1282_, 0, v___x_1281_);
lean_ctor_set(v___x_1282_, 1, v___x_1280_);
lean_ctor_set(v___x_1282_, 2, v_inst_1271_);
return v___x_1282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommMonoidWithOne___boxed(lean_object* v_R_1283_, lean_object* v_S_1284_, lean_object* v_f_1285_, lean_object* v_hf_1286_, lean_object* v_inst_1287_, lean_object* v_inst_1288_, lean_object* v_inst_1289_, lean_object* v_inst_1290_, lean_object* v_inst_1291_, lean_object* v_inst_1292_, lean_object* v_zero_1293_, lean_object* v_one_1294_, lean_object* v_add_1295_, lean_object* v_nsmul_1296_, lean_object* v_natCast_1297_){
_start:
{
lean_object* v_res_1298_; 
v_res_1298_ = lp_mathlib_Function_Surjective_addCommMonoidWithOne(v_R_1283_, v_S_1284_, v_f_1285_, v_hf_1286_, v_inst_1287_, v_inst_1288_, v_inst_1289_, v_inst_1290_, v_inst_1291_, v_inst_1292_, v_zero_1293_, v_one_1294_, v_add_1295_, v_nsmul_1296_, v_natCast_1297_);
lean_dec_ref(v_inst_1292_);
lean_dec(v_f_1285_);
return v_res_1298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addGroupWithOne___redArg(lean_object* v_inst_1299_, lean_object* v_inst_1300_, lean_object* v_inst_1301_, lean_object* v_inst_1302_, lean_object* v_inst_1303_, lean_object* v_inst_1304_, lean_object* v_inst_1305_, lean_object* v_inst_1306_, lean_object* v_inst_1307_){
_start:
{
lean_object* v___x_1308_; lean_object* v___x_1309_; lean_object* v___x_1310_; lean_object* v___x_1311_; lean_object* v_toNeg_1312_; lean_object* v_toSub_1313_; lean_object* v_toZSMul_1314_; lean_object* v___x_1315_; lean_object* v___x_1316_; 
lean_inc(v_inst_1304_);
lean_inc(v_inst_1300_);
lean_inc(v_inst_1299_);
v___x_1308_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_1299_, v_inst_1300_, v_inst_1304_);
v___x_1309_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_1309_, 0, lean_box(0));
lean_closure_set(v___x_1309_, 1, v_inst_1306_);
v___x_1310_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1310_, 0, v___x_1309_);
lean_ctor_set(v___x_1310_, 1, v___x_1308_);
lean_ctor_set(v___x_1310_, 2, v_inst_1301_);
v___x_1311_ = lp_mathlib_Function_Surjective_subNegMonoid___redArg(v_inst_1299_, v_inst_1300_, v_inst_1304_, v_inst_1302_, v_inst_1303_, v_inst_1305_);
v_toNeg_1312_ = lean_ctor_get(v___x_1311_, 1);
lean_inc(v_toNeg_1312_);
v_toSub_1313_ = lean_ctor_get(v___x_1311_, 2);
lean_inc(v_toSub_1313_);
v_toZSMul_1314_ = lean_ctor_get(v___x_1311_, 3);
lean_inc(v_toZSMul_1314_);
lean_dec_ref(v___x_1311_);
v___x_1315_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_1315_, 0, lean_box(0));
lean_closure_set(v___x_1315_, 1, v_inst_1307_);
v___x_1316_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1316_, 0, v___x_1315_);
lean_ctor_set(v___x_1316_, 1, v___x_1310_);
lean_ctor_set(v___x_1316_, 2, v_toNeg_1312_);
lean_ctor_set(v___x_1316_, 3, v_toSub_1313_);
lean_ctor_set(v___x_1316_, 4, v_toZSMul_1314_);
return v___x_1316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addGroupWithOne(lean_object* v_R_1317_, lean_object* v_S_1318_, lean_object* v_f_1319_, lean_object* v_hf_1320_, lean_object* v_inst_1321_, lean_object* v_inst_1322_, lean_object* v_inst_1323_, lean_object* v_inst_1324_, lean_object* v_inst_1325_, lean_object* v_inst_1326_, lean_object* v_inst_1327_, lean_object* v_inst_1328_, lean_object* v_inst_1329_, lean_object* v_inst_1330_, lean_object* v_zero_1331_, lean_object* v_one_1332_, lean_object* v_add_1333_, lean_object* v_neg_1334_, lean_object* v_sub_1335_, lean_object* v_nsmul_1336_, lean_object* v_zsmul_1337_, lean_object* v_natCast_1338_, lean_object* v_intCast_1339_){
_start:
{
lean_object* v___x_1340_; lean_object* v___x_1341_; lean_object* v___x_1342_; lean_object* v___x_1343_; lean_object* v_toNeg_1344_; lean_object* v_toSub_1345_; lean_object* v_toZSMul_1346_; lean_object* v___x_1347_; lean_object* v___x_1348_; 
lean_inc(v_inst_1326_);
lean_inc(v_inst_1322_);
lean_inc(v_inst_1321_);
v___x_1340_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_1321_, v_inst_1322_, v_inst_1326_);
v___x_1341_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_1341_, 0, lean_box(0));
lean_closure_set(v___x_1341_, 1, v_inst_1328_);
v___x_1342_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1342_, 0, v___x_1341_);
lean_ctor_set(v___x_1342_, 1, v___x_1340_);
lean_ctor_set(v___x_1342_, 2, v_inst_1323_);
v___x_1343_ = lp_mathlib_Function_Surjective_subNegMonoid___redArg(v_inst_1321_, v_inst_1322_, v_inst_1326_, v_inst_1324_, v_inst_1325_, v_inst_1327_);
v_toNeg_1344_ = lean_ctor_get(v___x_1343_, 1);
lean_inc(v_toNeg_1344_);
v_toSub_1345_ = lean_ctor_get(v___x_1343_, 2);
lean_inc(v_toSub_1345_);
v_toZSMul_1346_ = lean_ctor_get(v___x_1343_, 3);
lean_inc(v_toZSMul_1346_);
lean_dec_ref(v___x_1343_);
v___x_1347_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_1347_, 0, lean_box(0));
lean_closure_set(v___x_1347_, 1, v_inst_1329_);
v___x_1348_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1348_, 0, v___x_1347_);
lean_ctor_set(v___x_1348_, 1, v___x_1342_);
lean_ctor_set(v___x_1348_, 2, v_toNeg_1344_);
lean_ctor_set(v___x_1348_, 3, v_toSub_1345_);
lean_ctor_set(v___x_1348_, 4, v_toZSMul_1346_);
return v___x_1348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addGroupWithOne___boxed(lean_object** _args){
lean_object* v_R_1349_ = _args[0];
lean_object* v_S_1350_ = _args[1];
lean_object* v_f_1351_ = _args[2];
lean_object* v_hf_1352_ = _args[3];
lean_object* v_inst_1353_ = _args[4];
lean_object* v_inst_1354_ = _args[5];
lean_object* v_inst_1355_ = _args[6];
lean_object* v_inst_1356_ = _args[7];
lean_object* v_inst_1357_ = _args[8];
lean_object* v_inst_1358_ = _args[9];
lean_object* v_inst_1359_ = _args[10];
lean_object* v_inst_1360_ = _args[11];
lean_object* v_inst_1361_ = _args[12];
lean_object* v_inst_1362_ = _args[13];
lean_object* v_zero_1363_ = _args[14];
lean_object* v_one_1364_ = _args[15];
lean_object* v_add_1365_ = _args[16];
lean_object* v_neg_1366_ = _args[17];
lean_object* v_sub_1367_ = _args[18];
lean_object* v_nsmul_1368_ = _args[19];
lean_object* v_zsmul_1369_ = _args[20];
lean_object* v_natCast_1370_ = _args[21];
lean_object* v_intCast_1371_ = _args[22];
_start:
{
lean_object* v_res_1372_; 
v_res_1372_ = lp_mathlib_Function_Surjective_addGroupWithOne(v_R_1349_, v_S_1350_, v_f_1351_, v_hf_1352_, v_inst_1353_, v_inst_1354_, v_inst_1355_, v_inst_1356_, v_inst_1357_, v_inst_1358_, v_inst_1359_, v_inst_1360_, v_inst_1361_, v_inst_1362_, v_zero_1363_, v_one_1364_, v_add_1365_, v_neg_1366_, v_sub_1367_, v_nsmul_1368_, v_zsmul_1369_, v_natCast_1370_, v_intCast_1371_);
lean_dec_ref(v_inst_1362_);
lean_dec(v_f_1351_);
return v_res_1372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommGroupWithOne___redArg(lean_object* v_inst_1373_, lean_object* v_inst_1374_, lean_object* v_inst_1375_, lean_object* v_inst_1376_, lean_object* v_inst_1377_, lean_object* v_inst_1378_, lean_object* v_inst_1379_, lean_object* v_inst_1380_, lean_object* v_inst_1381_){
_start:
{
lean_object* v___x_1382_; lean_object* v___x_1383_; lean_object* v_toNeg_1384_; lean_object* v_toSub_1385_; lean_object* v_toZSMul_1386_; lean_object* v___x_1388_; uint8_t v_isShared_1389_; uint8_t v_isSharedCheck_1396_; 
lean_inc(v_inst_1378_);
lean_inc(v_inst_1374_);
lean_inc(v_inst_1373_);
v___x_1382_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_1373_, v_inst_1374_, v_inst_1378_);
v___x_1383_ = lp_mathlib_Function_Surjective_subNegMonoid___redArg(v_inst_1373_, v_inst_1374_, v_inst_1378_, v_inst_1376_, v_inst_1377_, v_inst_1379_);
v_toNeg_1384_ = lean_ctor_get(v___x_1383_, 1);
v_toSub_1385_ = lean_ctor_get(v___x_1383_, 2);
v_toZSMul_1386_ = lean_ctor_get(v___x_1383_, 3);
v_isSharedCheck_1396_ = !lean_is_exclusive(v___x_1383_);
if (v_isSharedCheck_1396_ == 0)
{
lean_object* v_unused_1397_; 
v_unused_1397_ = lean_ctor_get(v___x_1383_, 0);
lean_dec(v_unused_1397_);
v___x_1388_ = v___x_1383_;
v_isShared_1389_ = v_isSharedCheck_1396_;
goto v_resetjp_1387_;
}
else
{
lean_inc(v_toZSMul_1386_);
lean_inc(v_toSub_1385_);
lean_inc(v_toNeg_1384_);
lean_dec(v___x_1383_);
v___x_1388_ = lean_box(0);
v_isShared_1389_ = v_isSharedCheck_1396_;
goto v_resetjp_1387_;
}
v_resetjp_1387_:
{
lean_object* v___x_1390_; lean_object* v___x_1391_; lean_object* v___x_1393_; 
v___x_1390_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_1390_, 0, lean_box(0));
lean_closure_set(v___x_1390_, 1, v_inst_1380_);
v___x_1391_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_1391_, 0, lean_box(0));
lean_closure_set(v___x_1391_, 1, v_inst_1381_);
if (v_isShared_1389_ == 0)
{
lean_ctor_set(v___x_1388_, 0, v___x_1382_);
v___x_1393_ = v___x_1388_;
goto v_reusejp_1392_;
}
else
{
lean_object* v_reuseFailAlloc_1395_; 
v_reuseFailAlloc_1395_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_1395_, 0, v___x_1382_);
lean_ctor_set(v_reuseFailAlloc_1395_, 1, v_toNeg_1384_);
lean_ctor_set(v_reuseFailAlloc_1395_, 2, v_toSub_1385_);
lean_ctor_set(v_reuseFailAlloc_1395_, 3, v_toZSMul_1386_);
v___x_1393_ = v_reuseFailAlloc_1395_;
goto v_reusejp_1392_;
}
v_reusejp_1392_:
{
lean_object* v___x_1394_; 
v___x_1394_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1394_, 0, v___x_1393_);
lean_ctor_set(v___x_1394_, 1, v___x_1391_);
lean_ctor_set(v___x_1394_, 2, v___x_1390_);
lean_ctor_set(v___x_1394_, 3, v_inst_1375_);
return v___x_1394_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommGroupWithOne(lean_object* v_R_1398_, lean_object* v_S_1399_, lean_object* v_f_1400_, lean_object* v_hf_1401_, lean_object* v_inst_1402_, lean_object* v_inst_1403_, lean_object* v_inst_1404_, lean_object* v_inst_1405_, lean_object* v_inst_1406_, lean_object* v_inst_1407_, lean_object* v_inst_1408_, lean_object* v_inst_1409_, lean_object* v_inst_1410_, lean_object* v_inst_1411_, lean_object* v_zero_1412_, lean_object* v_one_1413_, lean_object* v_add_1414_, lean_object* v_neg_1415_, lean_object* v_sub_1416_, lean_object* v_nsmul_1417_, lean_object* v_zsmul_1418_, lean_object* v_natCast_1419_, lean_object* v_intCast_1420_){
_start:
{
lean_object* v___x_1421_; lean_object* v___x_1422_; lean_object* v_toNeg_1423_; lean_object* v_toSub_1424_; lean_object* v_toZSMul_1425_; lean_object* v___x_1427_; uint8_t v_isShared_1428_; uint8_t v_isSharedCheck_1435_; 
lean_inc(v_inst_1407_);
lean_inc(v_inst_1403_);
lean_inc(v_inst_1402_);
v___x_1421_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_1402_, v_inst_1403_, v_inst_1407_);
v___x_1422_ = lp_mathlib_Function_Surjective_subNegMonoid___redArg(v_inst_1402_, v_inst_1403_, v_inst_1407_, v_inst_1405_, v_inst_1406_, v_inst_1408_);
v_toNeg_1423_ = lean_ctor_get(v___x_1422_, 1);
v_toSub_1424_ = lean_ctor_get(v___x_1422_, 2);
v_toZSMul_1425_ = lean_ctor_get(v___x_1422_, 3);
v_isSharedCheck_1435_ = !lean_is_exclusive(v___x_1422_);
if (v_isSharedCheck_1435_ == 0)
{
lean_object* v_unused_1436_; 
v_unused_1436_ = lean_ctor_get(v___x_1422_, 0);
lean_dec(v_unused_1436_);
v___x_1427_ = v___x_1422_;
v_isShared_1428_ = v_isSharedCheck_1435_;
goto v_resetjp_1426_;
}
else
{
lean_inc(v_toZSMul_1425_);
lean_inc(v_toSub_1424_);
lean_inc(v_toNeg_1423_);
lean_dec(v___x_1422_);
v___x_1427_ = lean_box(0);
v_isShared_1428_ = v_isSharedCheck_1435_;
goto v_resetjp_1426_;
}
v_resetjp_1426_:
{
lean_object* v___x_1429_; lean_object* v___x_1430_; lean_object* v___x_1432_; 
v___x_1429_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_1429_, 0, lean_box(0));
lean_closure_set(v___x_1429_, 1, v_inst_1409_);
v___x_1430_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_1430_, 0, lean_box(0));
lean_closure_set(v___x_1430_, 1, v_inst_1410_);
if (v_isShared_1428_ == 0)
{
lean_ctor_set(v___x_1427_, 0, v___x_1421_);
v___x_1432_ = v___x_1427_;
goto v_reusejp_1431_;
}
else
{
lean_object* v_reuseFailAlloc_1434_; 
v_reuseFailAlloc_1434_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_1434_, 0, v___x_1421_);
lean_ctor_set(v_reuseFailAlloc_1434_, 1, v_toNeg_1423_);
lean_ctor_set(v_reuseFailAlloc_1434_, 2, v_toSub_1424_);
lean_ctor_set(v_reuseFailAlloc_1434_, 3, v_toZSMul_1425_);
v___x_1432_ = v_reuseFailAlloc_1434_;
goto v_reusejp_1431_;
}
v_reusejp_1431_:
{
lean_object* v___x_1433_; 
v___x_1433_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1433_, 0, v___x_1432_);
lean_ctor_set(v___x_1433_, 1, v___x_1430_);
lean_ctor_set(v___x_1433_, 2, v___x_1429_);
lean_ctor_set(v___x_1433_, 3, v_inst_1404_);
return v___x_1433_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommGroupWithOne___boxed(lean_object** _args){
lean_object* v_R_1437_ = _args[0];
lean_object* v_S_1438_ = _args[1];
lean_object* v_f_1439_ = _args[2];
lean_object* v_hf_1440_ = _args[3];
lean_object* v_inst_1441_ = _args[4];
lean_object* v_inst_1442_ = _args[5];
lean_object* v_inst_1443_ = _args[6];
lean_object* v_inst_1444_ = _args[7];
lean_object* v_inst_1445_ = _args[8];
lean_object* v_inst_1446_ = _args[9];
lean_object* v_inst_1447_ = _args[10];
lean_object* v_inst_1448_ = _args[11];
lean_object* v_inst_1449_ = _args[12];
lean_object* v_inst_1450_ = _args[13];
lean_object* v_zero_1451_ = _args[14];
lean_object* v_one_1452_ = _args[15];
lean_object* v_add_1453_ = _args[16];
lean_object* v_neg_1454_ = _args[17];
lean_object* v_sub_1455_ = _args[18];
lean_object* v_nsmul_1456_ = _args[19];
lean_object* v_zsmul_1457_ = _args[20];
lean_object* v_natCast_1458_ = _args[21];
lean_object* v_intCast_1459_ = _args[22];
_start:
{
lean_object* v_res_1460_; 
v_res_1460_ = lp_mathlib_Function_Surjective_addCommGroupWithOne(v_R_1437_, v_S_1438_, v_f_1439_, v_hf_1440_, v_inst_1441_, v_inst_1442_, v_inst_1443_, v_inst_1444_, v_inst_1445_, v_inst_1446_, v_inst_1447_, v_inst_1448_, v_inst_1449_, v_inst_1450_, v_zero_1451_, v_one_1452_, v_add_1453_, v_neg_1454_, v_sub_1455_, v_nsmul_1456_, v_zsmul_1457_, v_natCast_1458_, v_intCast_1459_);
lean_dec_ref(v_inst_1450_);
lean_dec(v_f_1439_);
return v_res_1460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocSemiring___redArg(lean_object* v_inst_1461_, lean_object* v_inst_1462_, lean_object* v_inst_1463_, lean_object* v_inst_1464_){
_start:
{
lean_object* v___x_1465_; lean_object* v___x_1466_; 
v___x_1465_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_1461_, v_inst_1463_, v_inst_1464_);
v___x_1466_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1466_, 0, v___x_1465_);
lean_ctor_set(v___x_1466_, 1, v_inst_1462_);
return v___x_1466_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocSemiring(lean_object* v_R_1467_, lean_object* v_S_1468_, lean_object* v_f_1469_, lean_object* v_hf_1470_, lean_object* v_inst_1471_, lean_object* v_inst_1472_, lean_object* v_inst_1473_, lean_object* v_inst_1474_, lean_object* v_inst_1475_, lean_object* v_zero_1476_, lean_object* v_add_1477_, lean_object* v_mul_1478_, lean_object* v_nsmul_1479_){
_start:
{
lean_object* v___x_1480_; lean_object* v___x_1481_; 
v___x_1480_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_1471_, v_inst_1473_, v_inst_1474_);
v___x_1481_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1481_, 0, v___x_1480_);
lean_ctor_set(v___x_1481_, 1, v_inst_1472_);
return v___x_1481_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocSemiring___boxed(lean_object* v_R_1482_, lean_object* v_S_1483_, lean_object* v_f_1484_, lean_object* v_hf_1485_, lean_object* v_inst_1486_, lean_object* v_inst_1487_, lean_object* v_inst_1488_, lean_object* v_inst_1489_, lean_object* v_inst_1490_, lean_object* v_zero_1491_, lean_object* v_add_1492_, lean_object* v_mul_1493_, lean_object* v_nsmul_1494_){
_start:
{
lean_object* v_res_1495_; 
v_res_1495_ = lp_mathlib_Function_Surjective_nonUnitalNonAssocSemiring(v_R_1482_, v_S_1483_, v_f_1484_, v_hf_1485_, v_inst_1486_, v_inst_1487_, v_inst_1488_, v_inst_1489_, v_inst_1490_, v_zero_1491_, v_add_1492_, v_mul_1493_, v_nsmul_1494_);
lean_dec_ref(v_inst_1490_);
lean_dec(v_f_1484_);
return v_res_1495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalSemiring___redArg(lean_object* v_inst_1496_, lean_object* v_inst_1497_, lean_object* v_inst_1498_, lean_object* v_inst_1499_){
_start:
{
lean_object* v___x_1500_; lean_object* v___x_1501_; 
v___x_1500_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_1496_, v_inst_1498_, v_inst_1499_);
v___x_1501_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1501_, 0, v___x_1500_);
lean_ctor_set(v___x_1501_, 1, v_inst_1497_);
return v___x_1501_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalSemiring(lean_object* v_R_1502_, lean_object* v_S_1503_, lean_object* v_f_1504_, lean_object* v_hf_1505_, lean_object* v_inst_1506_, lean_object* v_inst_1507_, lean_object* v_inst_1508_, lean_object* v_inst_1509_, lean_object* v_inst_1510_, lean_object* v_zero_1511_, lean_object* v_add_1512_, lean_object* v_mul_1513_, lean_object* v_nsmul_1514_){
_start:
{
lean_object* v___x_1515_; lean_object* v___x_1516_; 
v___x_1515_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_1506_, v_inst_1508_, v_inst_1509_);
v___x_1516_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1516_, 0, v___x_1515_);
lean_ctor_set(v___x_1516_, 1, v_inst_1507_);
return v___x_1516_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalSemiring___boxed(lean_object* v_R_1517_, lean_object* v_S_1518_, lean_object* v_f_1519_, lean_object* v_hf_1520_, lean_object* v_inst_1521_, lean_object* v_inst_1522_, lean_object* v_inst_1523_, lean_object* v_inst_1524_, lean_object* v_inst_1525_, lean_object* v_zero_1526_, lean_object* v_add_1527_, lean_object* v_mul_1528_, lean_object* v_nsmul_1529_){
_start:
{
lean_object* v_res_1530_; 
v_res_1530_ = lp_mathlib_Function_Surjective_nonUnitalSemiring(v_R_1517_, v_S_1518_, v_f_1519_, v_hf_1520_, v_inst_1521_, v_inst_1522_, v_inst_1523_, v_inst_1524_, v_inst_1525_, v_zero_1526_, v_add_1527_, v_mul_1528_, v_nsmul_1529_);
lean_dec_ref(v_inst_1525_);
lean_dec(v_f_1519_);
return v_res_1530_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocSemiring___redArg(lean_object* v_inst_1531_, lean_object* v_inst_1532_, lean_object* v_inst_1533_, lean_object* v_inst_1534_, lean_object* v_inst_1535_, lean_object* v_inst_1536_){
_start:
{
lean_object* v___x_1537_; lean_object* v___x_1538_; lean_object* v___x_1539_; lean_object* v___x_1540_; 
v___x_1537_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_1537_, 0, lean_box(0));
lean_closure_set(v___x_1537_, 1, v_inst_1536_);
v___x_1538_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_1531_, v_inst_1533_, v_inst_1535_);
v___x_1539_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1539_, 0, v___x_1538_);
lean_ctor_set(v___x_1539_, 1, v_inst_1532_);
v___x_1540_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1540_, 0, v___x_1539_);
lean_ctor_set(v___x_1540_, 1, v_inst_1534_);
lean_ctor_set(v___x_1540_, 2, v___x_1537_);
return v___x_1540_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocSemiring(lean_object* v_R_1541_, lean_object* v_S_1542_, lean_object* v_f_1543_, lean_object* v_hf_1544_, lean_object* v_inst_1545_, lean_object* v_inst_1546_, lean_object* v_inst_1547_, lean_object* v_inst_1548_, lean_object* v_inst_1549_, lean_object* v_inst_1550_, lean_object* v_inst_1551_, lean_object* v_zero_1552_, lean_object* v_one_1553_, lean_object* v_add_1554_, lean_object* v_mul_1555_, lean_object* v_nsmul_1556_, lean_object* v_natCast_1557_){
_start:
{
lean_object* v___x_1558_; lean_object* v___x_1559_; lean_object* v___x_1560_; lean_object* v___x_1561_; 
v___x_1558_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_1558_, 0, lean_box(0));
lean_closure_set(v___x_1558_, 1, v_inst_1550_);
v___x_1559_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_1545_, v_inst_1547_, v_inst_1549_);
v___x_1560_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1560_, 0, v___x_1559_);
lean_ctor_set(v___x_1560_, 1, v_inst_1546_);
v___x_1561_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1561_, 0, v___x_1560_);
lean_ctor_set(v___x_1561_, 1, v_inst_1548_);
lean_ctor_set(v___x_1561_, 2, v___x_1558_);
return v___x_1561_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocSemiring___boxed(lean_object** _args){
lean_object* v_R_1562_ = _args[0];
lean_object* v_S_1563_ = _args[1];
lean_object* v_f_1564_ = _args[2];
lean_object* v_hf_1565_ = _args[3];
lean_object* v_inst_1566_ = _args[4];
lean_object* v_inst_1567_ = _args[5];
lean_object* v_inst_1568_ = _args[6];
lean_object* v_inst_1569_ = _args[7];
lean_object* v_inst_1570_ = _args[8];
lean_object* v_inst_1571_ = _args[9];
lean_object* v_inst_1572_ = _args[10];
lean_object* v_zero_1573_ = _args[11];
lean_object* v_one_1574_ = _args[12];
lean_object* v_add_1575_ = _args[13];
lean_object* v_mul_1576_ = _args[14];
lean_object* v_nsmul_1577_ = _args[15];
lean_object* v_natCast_1578_ = _args[16];
_start:
{
lean_object* v_res_1579_; 
v_res_1579_ = lp_mathlib_Function_Surjective_nonAssocSemiring(v_R_1562_, v_S_1563_, v_f_1564_, v_hf_1565_, v_inst_1566_, v_inst_1567_, v_inst_1568_, v_inst_1569_, v_inst_1570_, v_inst_1571_, v_inst_1572_, v_zero_1573_, v_one_1574_, v_add_1575_, v_mul_1576_, v_nsmul_1577_, v_natCast_1578_);
lean_dec_ref(v_inst_1572_);
lean_dec(v_f_1564_);
return v_res_1579_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_semiring___redArg(lean_object* v_inst_1580_, lean_object* v_inst_1581_, lean_object* v_inst_1582_, lean_object* v_inst_1583_, lean_object* v_inst_1584_, lean_object* v_inst_1585_, lean_object* v_inst_1586_){
_start:
{
lean_object* v___f_1587_; lean_object* v___x_1588_; lean_object* v___x_1589_; lean_object* v___x_1590_; lean_object* v___x_1591_; 
v___f_1587_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1587_, 0, v_inst_1585_);
v___x_1588_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_1588_, 0, lean_box(0));
lean_closure_set(v___x_1588_, 1, v_inst_1586_);
v___x_1589_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_1580_, v_inst_1582_, v_inst_1584_);
v___x_1590_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1590_, 0, v_inst_1583_);
lean_ctor_set(v___x_1590_, 1, v_inst_1581_);
lean_ctor_set(v___x_1590_, 2, v___f_1587_);
v___x_1591_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1591_, 0, v___x_1589_);
lean_ctor_set(v___x_1591_, 1, v___x_1590_);
lean_ctor_set(v___x_1591_, 2, v___x_1588_);
return v___x_1591_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_semiring(lean_object* v_R_1592_, lean_object* v_S_1593_, lean_object* v_f_1594_, lean_object* v_hf_1595_, lean_object* v_inst_1596_, lean_object* v_inst_1597_, lean_object* v_inst_1598_, lean_object* v_inst_1599_, lean_object* v_inst_1600_, lean_object* v_inst_1601_, lean_object* v_inst_1602_, lean_object* v_inst_1603_, lean_object* v_zero_1604_, lean_object* v_one_1605_, lean_object* v_add_1606_, lean_object* v_mul_1607_, lean_object* v_nsmul_1608_, lean_object* v_npow_1609_, lean_object* v_natCast_1610_){
_start:
{
lean_object* v___f_1611_; lean_object* v___x_1612_; lean_object* v___x_1613_; lean_object* v___x_1614_; lean_object* v___x_1615_; 
v___f_1611_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1611_, 0, v_inst_1601_);
v___x_1612_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_1612_, 0, lean_box(0));
lean_closure_set(v___x_1612_, 1, v_inst_1602_);
v___x_1613_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_1596_, v_inst_1598_, v_inst_1600_);
v___x_1614_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1614_, 0, v_inst_1599_);
lean_ctor_set(v___x_1614_, 1, v_inst_1597_);
lean_ctor_set(v___x_1614_, 2, v___f_1611_);
v___x_1615_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1615_, 0, v___x_1613_);
lean_ctor_set(v___x_1615_, 1, v___x_1614_);
lean_ctor_set(v___x_1615_, 2, v___x_1612_);
return v___x_1615_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_semiring___boxed(lean_object** _args){
lean_object* v_R_1616_ = _args[0];
lean_object* v_S_1617_ = _args[1];
lean_object* v_f_1618_ = _args[2];
lean_object* v_hf_1619_ = _args[3];
lean_object* v_inst_1620_ = _args[4];
lean_object* v_inst_1621_ = _args[5];
lean_object* v_inst_1622_ = _args[6];
lean_object* v_inst_1623_ = _args[7];
lean_object* v_inst_1624_ = _args[8];
lean_object* v_inst_1625_ = _args[9];
lean_object* v_inst_1626_ = _args[10];
lean_object* v_inst_1627_ = _args[11];
lean_object* v_zero_1628_ = _args[12];
lean_object* v_one_1629_ = _args[13];
lean_object* v_add_1630_ = _args[14];
lean_object* v_mul_1631_ = _args[15];
lean_object* v_nsmul_1632_ = _args[16];
lean_object* v_npow_1633_ = _args[17];
lean_object* v_natCast_1634_ = _args[18];
_start:
{
lean_object* v_res_1635_; 
v_res_1635_ = lp_mathlib_Function_Surjective_semiring(v_R_1616_, v_S_1617_, v_f_1618_, v_hf_1619_, v_inst_1620_, v_inst_1621_, v_inst_1622_, v_inst_1623_, v_inst_1624_, v_inst_1625_, v_inst_1626_, v_inst_1627_, v_zero_1628_, v_one_1629_, v_add_1630_, v_mul_1631_, v_nsmul_1632_, v_npow_1633_, v_natCast_1634_);
lean_dec_ref(v_inst_1627_);
lean_dec(v_f_1618_);
return v_res_1635_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocRing___redArg(lean_object* v_inst_1636_, lean_object* v_inst_1637_, lean_object* v_inst_1638_, lean_object* v_inst_1639_, lean_object* v_inst_1640_, lean_object* v_inst_1641_, lean_object* v_inst_1642_){
_start:
{
lean_object* v___x_1643_; lean_object* v___x_1644_; 
v___x_1643_ = lp_mathlib_Function_Surjective_subNegMonoid___redArg(v_inst_1636_, v_inst_1638_, v_inst_1641_, v_inst_1639_, v_inst_1640_, v_inst_1642_);
v___x_1644_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1644_, 0, v___x_1643_);
lean_ctor_set(v___x_1644_, 1, v_inst_1637_);
return v___x_1644_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocRing(lean_object* v_R_1645_, lean_object* v_S_1646_, lean_object* v_f_1647_, lean_object* v_hf_1648_, lean_object* v_inst_1649_, lean_object* v_inst_1650_, lean_object* v_inst_1651_, lean_object* v_inst_1652_, lean_object* v_inst_1653_, lean_object* v_inst_1654_, lean_object* v_inst_1655_, lean_object* v_inst_1656_, lean_object* v_zero_1657_, lean_object* v_add_1658_, lean_object* v_mul_1659_, lean_object* v_neg_1660_, lean_object* v_sub_1661_, lean_object* v_nsmul_1662_, lean_object* v_zsmul_1663_){
_start:
{
lean_object* v___x_1664_; lean_object* v___x_1665_; 
v___x_1664_ = lp_mathlib_Function_Surjective_subNegMonoid___redArg(v_inst_1649_, v_inst_1651_, v_inst_1654_, v_inst_1652_, v_inst_1653_, v_inst_1655_);
v___x_1665_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1665_, 0, v___x_1664_);
lean_ctor_set(v___x_1665_, 1, v_inst_1650_);
return v___x_1665_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocRing___boxed(lean_object** _args){
lean_object* v_R_1666_ = _args[0];
lean_object* v_S_1667_ = _args[1];
lean_object* v_f_1668_ = _args[2];
lean_object* v_hf_1669_ = _args[3];
lean_object* v_inst_1670_ = _args[4];
lean_object* v_inst_1671_ = _args[5];
lean_object* v_inst_1672_ = _args[6];
lean_object* v_inst_1673_ = _args[7];
lean_object* v_inst_1674_ = _args[8];
lean_object* v_inst_1675_ = _args[9];
lean_object* v_inst_1676_ = _args[10];
lean_object* v_inst_1677_ = _args[11];
lean_object* v_zero_1678_ = _args[12];
lean_object* v_add_1679_ = _args[13];
lean_object* v_mul_1680_ = _args[14];
lean_object* v_neg_1681_ = _args[15];
lean_object* v_sub_1682_ = _args[16];
lean_object* v_nsmul_1683_ = _args[17];
lean_object* v_zsmul_1684_ = _args[18];
_start:
{
lean_object* v_res_1685_; 
v_res_1685_ = lp_mathlib_Function_Surjective_nonUnitalNonAssocRing(v_R_1666_, v_S_1667_, v_f_1668_, v_hf_1669_, v_inst_1670_, v_inst_1671_, v_inst_1672_, v_inst_1673_, v_inst_1674_, v_inst_1675_, v_inst_1676_, v_inst_1677_, v_zero_1678_, v_add_1679_, v_mul_1680_, v_neg_1681_, v_sub_1682_, v_nsmul_1683_, v_zsmul_1684_);
lean_dec_ref(v_inst_1677_);
lean_dec(v_f_1668_);
return v_res_1685_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalRing___redArg(lean_object* v_inst_1686_, lean_object* v_inst_1687_, lean_object* v_inst_1688_, lean_object* v_inst_1689_, lean_object* v_inst_1690_, lean_object* v_inst_1691_, lean_object* v_inst_1692_){
_start:
{
lean_object* v___x_1693_; lean_object* v___x_1694_; 
v___x_1693_ = lp_mathlib_Function_Surjective_subNegMonoid___redArg(v_inst_1686_, v_inst_1688_, v_inst_1691_, v_inst_1689_, v_inst_1690_, v_inst_1692_);
v___x_1694_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1694_, 0, v___x_1693_);
lean_ctor_set(v___x_1694_, 1, v_inst_1687_);
return v___x_1694_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalRing(lean_object* v_R_1695_, lean_object* v_S_1696_, lean_object* v_f_1697_, lean_object* v_hf_1698_, lean_object* v_inst_1699_, lean_object* v_inst_1700_, lean_object* v_inst_1701_, lean_object* v_inst_1702_, lean_object* v_inst_1703_, lean_object* v_inst_1704_, lean_object* v_inst_1705_, lean_object* v_inst_1706_, lean_object* v_zero_1707_, lean_object* v_add_1708_, lean_object* v_mul_1709_, lean_object* v_neg_1710_, lean_object* v_sub_1711_, lean_object* v_nsmul_1712_, lean_object* v_zsmul_1713_){
_start:
{
lean_object* v___x_1714_; lean_object* v___x_1715_; 
v___x_1714_ = lp_mathlib_Function_Surjective_subNegMonoid___redArg(v_inst_1699_, v_inst_1701_, v_inst_1704_, v_inst_1702_, v_inst_1703_, v_inst_1705_);
v___x_1715_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1715_, 0, v___x_1714_);
lean_ctor_set(v___x_1715_, 1, v_inst_1700_);
return v___x_1715_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalRing___boxed(lean_object** _args){
lean_object* v_R_1716_ = _args[0];
lean_object* v_S_1717_ = _args[1];
lean_object* v_f_1718_ = _args[2];
lean_object* v_hf_1719_ = _args[3];
lean_object* v_inst_1720_ = _args[4];
lean_object* v_inst_1721_ = _args[5];
lean_object* v_inst_1722_ = _args[6];
lean_object* v_inst_1723_ = _args[7];
lean_object* v_inst_1724_ = _args[8];
lean_object* v_inst_1725_ = _args[9];
lean_object* v_inst_1726_ = _args[10];
lean_object* v_inst_1727_ = _args[11];
lean_object* v_zero_1728_ = _args[12];
lean_object* v_add_1729_ = _args[13];
lean_object* v_mul_1730_ = _args[14];
lean_object* v_neg_1731_ = _args[15];
lean_object* v_sub_1732_ = _args[16];
lean_object* v_nsmul_1733_ = _args[17];
lean_object* v_zsmul_1734_ = _args[18];
_start:
{
lean_object* v_res_1735_; 
v_res_1735_ = lp_mathlib_Function_Surjective_nonUnitalRing(v_R_1716_, v_S_1717_, v_f_1718_, v_hf_1719_, v_inst_1720_, v_inst_1721_, v_inst_1722_, v_inst_1723_, v_inst_1724_, v_inst_1725_, v_inst_1726_, v_inst_1727_, v_zero_1728_, v_add_1729_, v_mul_1730_, v_neg_1731_, v_sub_1732_, v_nsmul_1733_, v_zsmul_1734_);
lean_dec_ref(v_inst_1727_);
lean_dec(v_f_1718_);
return v_res_1735_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocRing___redArg(lean_object* v_inst_1736_, lean_object* v_inst_1737_, lean_object* v_inst_1738_, lean_object* v_inst_1739_, lean_object* v_inst_1740_, lean_object* v_inst_1741_, lean_object* v_inst_1742_, lean_object* v_inst_1743_, lean_object* v_inst_1744_, lean_object* v_inst_1745_){
_start:
{
lean_object* v___x_1746_; lean_object* v___x_1747_; lean_object* v___x_1748_; lean_object* v___x_1749_; lean_object* v___x_1750_; 
v___x_1746_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_1746_, 0, lean_box(0));
lean_closure_set(v___x_1746_, 1, v_inst_1744_);
v___x_1747_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_1747_, 0, lean_box(0));
lean_closure_set(v___x_1747_, 1, v_inst_1745_);
v___x_1748_ = lp_mathlib_Function_Surjective_subNegMonoid___redArg(v_inst_1736_, v_inst_1738_, v_inst_1742_, v_inst_1740_, v_inst_1741_, v_inst_1743_);
v___x_1749_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1749_, 0, v___x_1748_);
lean_ctor_set(v___x_1749_, 1, v_inst_1737_);
v___x_1750_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1750_, 0, v___x_1749_);
lean_ctor_set(v___x_1750_, 1, v_inst_1739_);
lean_ctor_set(v___x_1750_, 2, v___x_1746_);
lean_ctor_set(v___x_1750_, 3, v___x_1747_);
return v___x_1750_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocRing(lean_object* v_R_1751_, lean_object* v_S_1752_, lean_object* v_f_1753_, lean_object* v_hf_1754_, lean_object* v_inst_1755_, lean_object* v_inst_1756_, lean_object* v_inst_1757_, lean_object* v_inst_1758_, lean_object* v_inst_1759_, lean_object* v_inst_1760_, lean_object* v_inst_1761_, lean_object* v_inst_1762_, lean_object* v_inst_1763_, lean_object* v_inst_1764_, lean_object* v_inst_1765_, lean_object* v_zero_1766_, lean_object* v_one_1767_, lean_object* v_add_1768_, lean_object* v_mul_1769_, lean_object* v_neg_1770_, lean_object* v_sub_1771_, lean_object* v_nsmul_1772_, lean_object* v_zsmul_1773_, lean_object* v_natCast_1774_, lean_object* v_intCast_1775_){
_start:
{
lean_object* v___x_1776_; lean_object* v___x_1777_; lean_object* v___x_1778_; lean_object* v___x_1779_; lean_object* v___x_1780_; 
v___x_1776_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_1776_, 0, lean_box(0));
lean_closure_set(v___x_1776_, 1, v_inst_1763_);
v___x_1777_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_1777_, 0, lean_box(0));
lean_closure_set(v___x_1777_, 1, v_inst_1764_);
v___x_1778_ = lp_mathlib_Function_Surjective_subNegMonoid___redArg(v_inst_1755_, v_inst_1757_, v_inst_1761_, v_inst_1759_, v_inst_1760_, v_inst_1762_);
v___x_1779_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1779_, 0, v___x_1778_);
lean_ctor_set(v___x_1779_, 1, v_inst_1756_);
v___x_1780_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1780_, 0, v___x_1779_);
lean_ctor_set(v___x_1780_, 1, v_inst_1758_);
lean_ctor_set(v___x_1780_, 2, v___x_1776_);
lean_ctor_set(v___x_1780_, 3, v___x_1777_);
return v___x_1780_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocRing___boxed(lean_object** _args){
lean_object* v_R_1781_ = _args[0];
lean_object* v_S_1782_ = _args[1];
lean_object* v_f_1783_ = _args[2];
lean_object* v_hf_1784_ = _args[3];
lean_object* v_inst_1785_ = _args[4];
lean_object* v_inst_1786_ = _args[5];
lean_object* v_inst_1787_ = _args[6];
lean_object* v_inst_1788_ = _args[7];
lean_object* v_inst_1789_ = _args[8];
lean_object* v_inst_1790_ = _args[9];
lean_object* v_inst_1791_ = _args[10];
lean_object* v_inst_1792_ = _args[11];
lean_object* v_inst_1793_ = _args[12];
lean_object* v_inst_1794_ = _args[13];
lean_object* v_inst_1795_ = _args[14];
lean_object* v_zero_1796_ = _args[15];
lean_object* v_one_1797_ = _args[16];
lean_object* v_add_1798_ = _args[17];
lean_object* v_mul_1799_ = _args[18];
lean_object* v_neg_1800_ = _args[19];
lean_object* v_sub_1801_ = _args[20];
lean_object* v_nsmul_1802_ = _args[21];
lean_object* v_zsmul_1803_ = _args[22];
lean_object* v_natCast_1804_ = _args[23];
lean_object* v_intCast_1805_ = _args[24];
_start:
{
lean_object* v_res_1806_; 
v_res_1806_ = lp_mathlib_Function_Surjective_nonAssocRing(v_R_1781_, v_S_1782_, v_f_1783_, v_hf_1784_, v_inst_1785_, v_inst_1786_, v_inst_1787_, v_inst_1788_, v_inst_1789_, v_inst_1790_, v_inst_1791_, v_inst_1792_, v_inst_1793_, v_inst_1794_, v_inst_1795_, v_zero_1796_, v_one_1797_, v_add_1798_, v_mul_1799_, v_neg_1800_, v_sub_1801_, v_nsmul_1802_, v_zsmul_1803_, v_natCast_1804_, v_intCast_1805_);
lean_dec_ref(v_inst_1795_);
lean_dec(v_f_1783_);
return v_res_1806_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_ring___redArg(lean_object* v_inst_1807_, lean_object* v_inst_1808_, lean_object* v_inst_1809_, lean_object* v_inst_1810_, lean_object* v_inst_1811_, lean_object* v_inst_1812_, lean_object* v_inst_1813_, lean_object* v_inst_1814_, lean_object* v_inst_1815_, lean_object* v_inst_1816_, lean_object* v_inst_1817_){
_start:
{
lean_object* v___x_1818_; lean_object* v_toNeg_1819_; lean_object* v_toSub_1820_; lean_object* v_toZSMul_1821_; lean_object* v___f_1822_; lean_object* v___x_1823_; lean_object* v___x_1824_; lean_object* v___x_1825_; lean_object* v___x_1826_; lean_object* v___x_1827_; lean_object* v___x_1828_; 
lean_inc(v_inst_1813_);
lean_inc(v_inst_1809_);
lean_inc(v_inst_1807_);
v___x_1818_ = lp_mathlib_Function_Surjective_subNegMonoid___redArg(v_inst_1807_, v_inst_1809_, v_inst_1813_, v_inst_1811_, v_inst_1812_, v_inst_1814_);
v_toNeg_1819_ = lean_ctor_get(v___x_1818_, 1);
lean_inc(v_toNeg_1819_);
v_toSub_1820_ = lean_ctor_get(v___x_1818_, 2);
lean_inc(v_toSub_1820_);
v_toZSMul_1821_ = lean_ctor_get(v___x_1818_, 3);
lean_inc(v_toZSMul_1821_);
lean_dec_ref(v___x_1818_);
v___f_1822_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1822_, 0, v_inst_1815_);
v___x_1823_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_1823_, 0, lean_box(0));
lean_closure_set(v___x_1823_, 1, v_inst_1817_);
v___x_1824_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_1824_, 0, lean_box(0));
lean_closure_set(v___x_1824_, 1, v_inst_1816_);
v___x_1825_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_1807_, v_inst_1809_, v_inst_1813_);
v___x_1826_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1826_, 0, v_inst_1810_);
lean_ctor_set(v___x_1826_, 1, v_inst_1808_);
lean_ctor_set(v___x_1826_, 2, v___f_1822_);
v___x_1827_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1827_, 0, v___x_1825_);
lean_ctor_set(v___x_1827_, 1, v___x_1826_);
lean_ctor_set(v___x_1827_, 2, v___x_1824_);
v___x_1828_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1828_, 0, v___x_1827_);
lean_ctor_set(v___x_1828_, 1, v_toNeg_1819_);
lean_ctor_set(v___x_1828_, 2, v_toSub_1820_);
lean_ctor_set(v___x_1828_, 3, v_toZSMul_1821_);
lean_ctor_set(v___x_1828_, 4, v___x_1823_);
return v___x_1828_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_ring(lean_object* v_R_1829_, lean_object* v_S_1830_, lean_object* v_f_1831_, lean_object* v_hf_1832_, lean_object* v_inst_1833_, lean_object* v_inst_1834_, lean_object* v_inst_1835_, lean_object* v_inst_1836_, lean_object* v_inst_1837_, lean_object* v_inst_1838_, lean_object* v_inst_1839_, lean_object* v_inst_1840_, lean_object* v_inst_1841_, lean_object* v_inst_1842_, lean_object* v_inst_1843_, lean_object* v_inst_1844_, lean_object* v_zero_1845_, lean_object* v_one_1846_, lean_object* v_add_1847_, lean_object* v_mul_1848_, lean_object* v_neg_1849_, lean_object* v_sub_1850_, lean_object* v_nsmul_1851_, lean_object* v_zsmul_1852_, lean_object* v_npow_1853_, lean_object* v_natCast_1854_, lean_object* v_intCast_1855_){
_start:
{
lean_object* v___x_1856_; lean_object* v_toNeg_1857_; lean_object* v_toSub_1858_; lean_object* v_toZSMul_1859_; lean_object* v___f_1860_; lean_object* v___x_1861_; lean_object* v___x_1862_; lean_object* v___x_1863_; lean_object* v___x_1864_; lean_object* v___x_1865_; lean_object* v___x_1866_; 
lean_inc(v_inst_1839_);
lean_inc(v_inst_1835_);
lean_inc(v_inst_1833_);
v___x_1856_ = lp_mathlib_Function_Surjective_subNegMonoid___redArg(v_inst_1833_, v_inst_1835_, v_inst_1839_, v_inst_1837_, v_inst_1838_, v_inst_1840_);
v_toNeg_1857_ = lean_ctor_get(v___x_1856_, 1);
lean_inc(v_toNeg_1857_);
v_toSub_1858_ = lean_ctor_get(v___x_1856_, 2);
lean_inc(v_toSub_1858_);
v_toZSMul_1859_ = lean_ctor_get(v___x_1856_, 3);
lean_inc(v_toZSMul_1859_);
lean_dec_ref(v___x_1856_);
v___f_1860_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1860_, 0, v_inst_1841_);
v___x_1861_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_1861_, 0, lean_box(0));
lean_closure_set(v___x_1861_, 1, v_inst_1843_);
v___x_1862_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_1862_, 0, lean_box(0));
lean_closure_set(v___x_1862_, 1, v_inst_1842_);
v___x_1863_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_1833_, v_inst_1835_, v_inst_1839_);
v___x_1864_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1864_, 0, v_inst_1836_);
lean_ctor_set(v___x_1864_, 1, v_inst_1834_);
lean_ctor_set(v___x_1864_, 2, v___f_1860_);
v___x_1865_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1865_, 0, v___x_1863_);
lean_ctor_set(v___x_1865_, 1, v___x_1864_);
lean_ctor_set(v___x_1865_, 2, v___x_1862_);
v___x_1866_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1866_, 0, v___x_1865_);
lean_ctor_set(v___x_1866_, 1, v_toNeg_1857_);
lean_ctor_set(v___x_1866_, 2, v_toSub_1858_);
lean_ctor_set(v___x_1866_, 3, v_toZSMul_1859_);
lean_ctor_set(v___x_1866_, 4, v___x_1861_);
return v___x_1866_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_ring___boxed(lean_object** _args){
lean_object* v_R_1867_ = _args[0];
lean_object* v_S_1868_ = _args[1];
lean_object* v_f_1869_ = _args[2];
lean_object* v_hf_1870_ = _args[3];
lean_object* v_inst_1871_ = _args[4];
lean_object* v_inst_1872_ = _args[5];
lean_object* v_inst_1873_ = _args[6];
lean_object* v_inst_1874_ = _args[7];
lean_object* v_inst_1875_ = _args[8];
lean_object* v_inst_1876_ = _args[9];
lean_object* v_inst_1877_ = _args[10];
lean_object* v_inst_1878_ = _args[11];
lean_object* v_inst_1879_ = _args[12];
lean_object* v_inst_1880_ = _args[13];
lean_object* v_inst_1881_ = _args[14];
lean_object* v_inst_1882_ = _args[15];
lean_object* v_zero_1883_ = _args[16];
lean_object* v_one_1884_ = _args[17];
lean_object* v_add_1885_ = _args[18];
lean_object* v_mul_1886_ = _args[19];
lean_object* v_neg_1887_ = _args[20];
lean_object* v_sub_1888_ = _args[21];
lean_object* v_nsmul_1889_ = _args[22];
lean_object* v_zsmul_1890_ = _args[23];
lean_object* v_npow_1891_ = _args[24];
lean_object* v_natCast_1892_ = _args[25];
lean_object* v_intCast_1893_ = _args[26];
_start:
{
lean_object* v_res_1894_; 
v_res_1894_ = lp_mathlib_Function_Surjective_ring(v_R_1867_, v_S_1868_, v_f_1869_, v_hf_1870_, v_inst_1871_, v_inst_1872_, v_inst_1873_, v_inst_1874_, v_inst_1875_, v_inst_1876_, v_inst_1877_, v_inst_1878_, v_inst_1879_, v_inst_1880_, v_inst_1881_, v_inst_1882_, v_zero_1883_, v_one_1884_, v_add_1885_, v_mul_1886_, v_neg_1887_, v_sub_1888_, v_nsmul_1889_, v_zsmul_1890_, v_npow_1891_, v_natCast_1892_, v_intCast_1893_);
lean_dec_ref(v_inst_1882_);
lean_dec(v_f_1869_);
return v_res_1894_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocCommSemiring___redArg(lean_object* v_inst_1895_, lean_object* v_inst_1896_, lean_object* v_inst_1897_, lean_object* v_inst_1898_){
_start:
{
lean_object* v___x_1899_; lean_object* v___x_1900_; 
v___x_1899_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_1895_, v_inst_1897_, v_inst_1898_);
v___x_1900_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1900_, 0, v___x_1899_);
lean_ctor_set(v___x_1900_, 1, v_inst_1896_);
return v___x_1900_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocCommSemiring(lean_object* v_R_1901_, lean_object* v_S_1902_, lean_object* v_f_1903_, lean_object* v_hf_1904_, lean_object* v_inst_1905_, lean_object* v_inst_1906_, lean_object* v_inst_1907_, lean_object* v_inst_1908_, lean_object* v_inst_1909_, lean_object* v_zero_1910_, lean_object* v_add_1911_, lean_object* v_mul_1912_, lean_object* v_nsmul_1913_){
_start:
{
lean_object* v___x_1914_; lean_object* v___x_1915_; 
v___x_1914_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_1905_, v_inst_1907_, v_inst_1908_);
v___x_1915_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1915_, 0, v___x_1914_);
lean_ctor_set(v___x_1915_, 1, v_inst_1906_);
return v___x_1915_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocCommSemiring___boxed(lean_object* v_R_1916_, lean_object* v_S_1917_, lean_object* v_f_1918_, lean_object* v_hf_1919_, lean_object* v_inst_1920_, lean_object* v_inst_1921_, lean_object* v_inst_1922_, lean_object* v_inst_1923_, lean_object* v_inst_1924_, lean_object* v_zero_1925_, lean_object* v_add_1926_, lean_object* v_mul_1927_, lean_object* v_nsmul_1928_){
_start:
{
lean_object* v_res_1929_; 
v_res_1929_ = lp_mathlib_Function_Surjective_nonUnitalNonAssocCommSemiring(v_R_1916_, v_S_1917_, v_f_1918_, v_hf_1919_, v_inst_1920_, v_inst_1921_, v_inst_1922_, v_inst_1923_, v_inst_1924_, v_zero_1925_, v_add_1926_, v_mul_1927_, v_nsmul_1928_);
lean_dec_ref(v_inst_1924_);
lean_dec(v_f_1918_);
return v_res_1929_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalCommSemiring___redArg(lean_object* v_inst_1930_, lean_object* v_inst_1931_, lean_object* v_inst_1932_, lean_object* v_inst_1933_){
_start:
{
lean_object* v___x_1934_; lean_object* v___x_1935_; 
v___x_1934_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_1930_, v_inst_1932_, v_inst_1933_);
v___x_1935_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1935_, 0, v___x_1934_);
lean_ctor_set(v___x_1935_, 1, v_inst_1931_);
return v___x_1935_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalCommSemiring(lean_object* v_R_1936_, lean_object* v_S_1937_, lean_object* v_f_1938_, lean_object* v_hf_1939_, lean_object* v_inst_1940_, lean_object* v_inst_1941_, lean_object* v_inst_1942_, lean_object* v_inst_1943_, lean_object* v_inst_1944_, lean_object* v_zero_1945_, lean_object* v_add_1946_, lean_object* v_mul_1947_, lean_object* v_nsmul_1948_){
_start:
{
lean_object* v___x_1949_; lean_object* v___x_1950_; 
v___x_1949_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_1940_, v_inst_1942_, v_inst_1943_);
v___x_1950_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1950_, 0, v___x_1949_);
lean_ctor_set(v___x_1950_, 1, v_inst_1941_);
return v___x_1950_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalCommSemiring___boxed(lean_object* v_R_1951_, lean_object* v_S_1952_, lean_object* v_f_1953_, lean_object* v_hf_1954_, lean_object* v_inst_1955_, lean_object* v_inst_1956_, lean_object* v_inst_1957_, lean_object* v_inst_1958_, lean_object* v_inst_1959_, lean_object* v_zero_1960_, lean_object* v_add_1961_, lean_object* v_mul_1962_, lean_object* v_nsmul_1963_){
_start:
{
lean_object* v_res_1964_; 
v_res_1964_ = lp_mathlib_Function_Surjective_nonUnitalCommSemiring(v_R_1951_, v_S_1952_, v_f_1953_, v_hf_1954_, v_inst_1955_, v_inst_1956_, v_inst_1957_, v_inst_1958_, v_inst_1959_, v_zero_1960_, v_add_1961_, v_mul_1962_, v_nsmul_1963_);
lean_dec_ref(v_inst_1959_);
lean_dec(v_f_1953_);
return v_res_1964_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocCommSemiring___redArg(lean_object* v_inst_1965_, lean_object* v_inst_1966_, lean_object* v_inst_1967_, lean_object* v_inst_1968_, lean_object* v_inst_1969_, lean_object* v_inst_1970_){
_start:
{
lean_object* v___x_1971_; lean_object* v___x_1972_; lean_object* v___x_1973_; lean_object* v___x_1974_; 
v___x_1971_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_1971_, 0, lean_box(0));
lean_closure_set(v___x_1971_, 1, v_inst_1970_);
v___x_1972_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_1965_, v_inst_1967_, v_inst_1969_);
v___x_1973_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1973_, 0, v___x_1972_);
lean_ctor_set(v___x_1973_, 1, v_inst_1966_);
v___x_1974_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1974_, 0, v___x_1973_);
lean_ctor_set(v___x_1974_, 1, v_inst_1968_);
lean_ctor_set(v___x_1974_, 2, v___x_1971_);
return v___x_1974_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocCommSemiring(lean_object* v_R_1975_, lean_object* v_S_1976_, lean_object* v_f_1977_, lean_object* v_hf_1978_, lean_object* v_inst_1979_, lean_object* v_inst_1980_, lean_object* v_inst_1981_, lean_object* v_inst_1982_, lean_object* v_inst_1983_, lean_object* v_inst_1984_, lean_object* v_inst_1985_, lean_object* v_zero_1986_, lean_object* v_one_1987_, lean_object* v_add_1988_, lean_object* v_mul_1989_, lean_object* v_nsmul_1990_, lean_object* v_natCast_1991_){
_start:
{
lean_object* v___x_1992_; lean_object* v___x_1993_; lean_object* v___x_1994_; lean_object* v___x_1995_; 
v___x_1992_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_1992_, 0, lean_box(0));
lean_closure_set(v___x_1992_, 1, v_inst_1984_);
v___x_1993_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_1979_, v_inst_1981_, v_inst_1983_);
v___x_1994_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1994_, 0, v___x_1993_);
lean_ctor_set(v___x_1994_, 1, v_inst_1980_);
v___x_1995_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1995_, 0, v___x_1994_);
lean_ctor_set(v___x_1995_, 1, v_inst_1982_);
lean_ctor_set(v___x_1995_, 2, v___x_1992_);
return v___x_1995_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocCommSemiring___boxed(lean_object** _args){
lean_object* v_R_1996_ = _args[0];
lean_object* v_S_1997_ = _args[1];
lean_object* v_f_1998_ = _args[2];
lean_object* v_hf_1999_ = _args[3];
lean_object* v_inst_2000_ = _args[4];
lean_object* v_inst_2001_ = _args[5];
lean_object* v_inst_2002_ = _args[6];
lean_object* v_inst_2003_ = _args[7];
lean_object* v_inst_2004_ = _args[8];
lean_object* v_inst_2005_ = _args[9];
lean_object* v_inst_2006_ = _args[10];
lean_object* v_zero_2007_ = _args[11];
lean_object* v_one_2008_ = _args[12];
lean_object* v_add_2009_ = _args[13];
lean_object* v_mul_2010_ = _args[14];
lean_object* v_nsmul_2011_ = _args[15];
lean_object* v_natCast_2012_ = _args[16];
_start:
{
lean_object* v_res_2013_; 
v_res_2013_ = lp_mathlib_Function_Surjective_nonAssocCommSemiring(v_R_1996_, v_S_1997_, v_f_1998_, v_hf_1999_, v_inst_2000_, v_inst_2001_, v_inst_2002_, v_inst_2003_, v_inst_2004_, v_inst_2005_, v_inst_2006_, v_zero_2007_, v_one_2008_, v_add_2009_, v_mul_2010_, v_nsmul_2011_, v_natCast_2012_);
lean_dec_ref(v_inst_2006_);
lean_dec(v_f_1998_);
return v_res_2013_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commSemiring___redArg(lean_object* v_inst_2014_, lean_object* v_inst_2015_, lean_object* v_inst_2016_, lean_object* v_inst_2017_, lean_object* v_inst_2018_, lean_object* v_inst_2019_, lean_object* v_inst_2020_){
_start:
{
lean_object* v___f_2021_; lean_object* v___x_2022_; lean_object* v___x_2023_; lean_object* v___x_2024_; lean_object* v___x_2025_; 
v___f_2021_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2021_, 0, v_inst_2019_);
v___x_2022_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_2022_, 0, lean_box(0));
lean_closure_set(v___x_2022_, 1, v_inst_2020_);
v___x_2023_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_2014_, v_inst_2016_, v_inst_2018_);
v___x_2024_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2024_, 0, v_inst_2017_);
lean_ctor_set(v___x_2024_, 1, v_inst_2015_);
lean_ctor_set(v___x_2024_, 2, v___f_2021_);
v___x_2025_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2025_, 0, v___x_2023_);
lean_ctor_set(v___x_2025_, 1, v___x_2024_);
lean_ctor_set(v___x_2025_, 2, v___x_2022_);
return v___x_2025_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commSemiring(lean_object* v_R_2026_, lean_object* v_S_2027_, lean_object* v_f_2028_, lean_object* v_hf_2029_, lean_object* v_inst_2030_, lean_object* v_inst_2031_, lean_object* v_inst_2032_, lean_object* v_inst_2033_, lean_object* v_inst_2034_, lean_object* v_inst_2035_, lean_object* v_inst_2036_, lean_object* v_inst_2037_, lean_object* v_zero_2038_, lean_object* v_one_2039_, lean_object* v_add_2040_, lean_object* v_mul_2041_, lean_object* v_nsmul_2042_, lean_object* v_npow_2043_, lean_object* v_natCast_2044_){
_start:
{
lean_object* v___f_2045_; lean_object* v___x_2046_; lean_object* v___x_2047_; lean_object* v___x_2048_; lean_object* v___x_2049_; 
v___f_2045_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2045_, 0, v_inst_2035_);
v___x_2046_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_2046_, 0, lean_box(0));
lean_closure_set(v___x_2046_, 1, v_inst_2036_);
v___x_2047_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_2030_, v_inst_2032_, v_inst_2034_);
v___x_2048_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2048_, 0, v_inst_2033_);
lean_ctor_set(v___x_2048_, 1, v_inst_2031_);
lean_ctor_set(v___x_2048_, 2, v___f_2045_);
v___x_2049_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2049_, 0, v___x_2047_);
lean_ctor_set(v___x_2049_, 1, v___x_2048_);
lean_ctor_set(v___x_2049_, 2, v___x_2046_);
return v___x_2049_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commSemiring___boxed(lean_object** _args){
lean_object* v_R_2050_ = _args[0];
lean_object* v_S_2051_ = _args[1];
lean_object* v_f_2052_ = _args[2];
lean_object* v_hf_2053_ = _args[3];
lean_object* v_inst_2054_ = _args[4];
lean_object* v_inst_2055_ = _args[5];
lean_object* v_inst_2056_ = _args[6];
lean_object* v_inst_2057_ = _args[7];
lean_object* v_inst_2058_ = _args[8];
lean_object* v_inst_2059_ = _args[9];
lean_object* v_inst_2060_ = _args[10];
lean_object* v_inst_2061_ = _args[11];
lean_object* v_zero_2062_ = _args[12];
lean_object* v_one_2063_ = _args[13];
lean_object* v_add_2064_ = _args[14];
lean_object* v_mul_2065_ = _args[15];
lean_object* v_nsmul_2066_ = _args[16];
lean_object* v_npow_2067_ = _args[17];
lean_object* v_natCast_2068_ = _args[18];
_start:
{
lean_object* v_res_2069_; 
v_res_2069_ = lp_mathlib_Function_Surjective_commSemiring(v_R_2050_, v_S_2051_, v_f_2052_, v_hf_2053_, v_inst_2054_, v_inst_2055_, v_inst_2056_, v_inst_2057_, v_inst_2058_, v_inst_2059_, v_inst_2060_, v_inst_2061_, v_zero_2062_, v_one_2063_, v_add_2064_, v_mul_2065_, v_nsmul_2066_, v_npow_2067_, v_natCast_2068_);
lean_dec_ref(v_inst_2061_);
lean_dec(v_f_2052_);
return v_res_2069_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocCommRing___redArg(lean_object* v_inst_2070_, lean_object* v_inst_2071_, lean_object* v_inst_2072_, lean_object* v_inst_2073_, lean_object* v_inst_2074_, lean_object* v_inst_2075_, lean_object* v_inst_2076_){
_start:
{
lean_object* v___x_2077_; lean_object* v___x_2078_; 
v___x_2077_ = lp_mathlib_Function_Surjective_subNegMonoid___redArg(v_inst_2070_, v_inst_2072_, v_inst_2075_, v_inst_2073_, v_inst_2074_, v_inst_2076_);
v___x_2078_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2078_, 0, v___x_2077_);
lean_ctor_set(v___x_2078_, 1, v_inst_2071_);
return v___x_2078_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocCommRing(lean_object* v_R_2079_, lean_object* v_S_2080_, lean_object* v_f_2081_, lean_object* v_hf_2082_, lean_object* v_inst_2083_, lean_object* v_inst_2084_, lean_object* v_inst_2085_, lean_object* v_inst_2086_, lean_object* v_inst_2087_, lean_object* v_inst_2088_, lean_object* v_inst_2089_, lean_object* v_inst_2090_, lean_object* v_zero_2091_, lean_object* v_add_2092_, lean_object* v_mul_2093_, lean_object* v_neg_2094_, lean_object* v_sub_2095_, lean_object* v_nsmul_2096_, lean_object* v_zsmul_2097_){
_start:
{
lean_object* v___x_2098_; lean_object* v___x_2099_; 
v___x_2098_ = lp_mathlib_Function_Surjective_subNegMonoid___redArg(v_inst_2083_, v_inst_2085_, v_inst_2088_, v_inst_2086_, v_inst_2087_, v_inst_2089_);
v___x_2099_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2099_, 0, v___x_2098_);
lean_ctor_set(v___x_2099_, 1, v_inst_2084_);
return v___x_2099_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalNonAssocCommRing___boxed(lean_object** _args){
lean_object* v_R_2100_ = _args[0];
lean_object* v_S_2101_ = _args[1];
lean_object* v_f_2102_ = _args[2];
lean_object* v_hf_2103_ = _args[3];
lean_object* v_inst_2104_ = _args[4];
lean_object* v_inst_2105_ = _args[5];
lean_object* v_inst_2106_ = _args[6];
lean_object* v_inst_2107_ = _args[7];
lean_object* v_inst_2108_ = _args[8];
lean_object* v_inst_2109_ = _args[9];
lean_object* v_inst_2110_ = _args[10];
lean_object* v_inst_2111_ = _args[11];
lean_object* v_zero_2112_ = _args[12];
lean_object* v_add_2113_ = _args[13];
lean_object* v_mul_2114_ = _args[14];
lean_object* v_neg_2115_ = _args[15];
lean_object* v_sub_2116_ = _args[16];
lean_object* v_nsmul_2117_ = _args[17];
lean_object* v_zsmul_2118_ = _args[18];
_start:
{
lean_object* v_res_2119_; 
v_res_2119_ = lp_mathlib_Function_Surjective_nonUnitalNonAssocCommRing(v_R_2100_, v_S_2101_, v_f_2102_, v_hf_2103_, v_inst_2104_, v_inst_2105_, v_inst_2106_, v_inst_2107_, v_inst_2108_, v_inst_2109_, v_inst_2110_, v_inst_2111_, v_zero_2112_, v_add_2113_, v_mul_2114_, v_neg_2115_, v_sub_2116_, v_nsmul_2117_, v_zsmul_2118_);
lean_dec_ref(v_inst_2111_);
lean_dec(v_f_2102_);
return v_res_2119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalCommRing___redArg(lean_object* v_inst_2120_, lean_object* v_inst_2121_, lean_object* v_inst_2122_, lean_object* v_inst_2123_, lean_object* v_inst_2124_, lean_object* v_inst_2125_, lean_object* v_inst_2126_){
_start:
{
lean_object* v___x_2127_; lean_object* v___x_2128_; 
v___x_2127_ = lp_mathlib_Function_Surjective_subNegMonoid___redArg(v_inst_2120_, v_inst_2122_, v_inst_2125_, v_inst_2123_, v_inst_2124_, v_inst_2126_);
v___x_2128_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2128_, 0, v___x_2127_);
lean_ctor_set(v___x_2128_, 1, v_inst_2121_);
return v___x_2128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalCommRing(lean_object* v_R_2129_, lean_object* v_S_2130_, lean_object* v_f_2131_, lean_object* v_hf_2132_, lean_object* v_inst_2133_, lean_object* v_inst_2134_, lean_object* v_inst_2135_, lean_object* v_inst_2136_, lean_object* v_inst_2137_, lean_object* v_inst_2138_, lean_object* v_inst_2139_, lean_object* v_inst_2140_, lean_object* v_zero_2141_, lean_object* v_add_2142_, lean_object* v_mul_2143_, lean_object* v_neg_2144_, lean_object* v_sub_2145_, lean_object* v_nsmul_2146_, lean_object* v_zsmul_2147_){
_start:
{
lean_object* v___x_2148_; lean_object* v___x_2149_; 
v___x_2148_ = lp_mathlib_Function_Surjective_subNegMonoid___redArg(v_inst_2133_, v_inst_2135_, v_inst_2138_, v_inst_2136_, v_inst_2137_, v_inst_2139_);
v___x_2149_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2149_, 0, v___x_2148_);
lean_ctor_set(v___x_2149_, 1, v_inst_2134_);
return v___x_2149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonUnitalCommRing___boxed(lean_object** _args){
lean_object* v_R_2150_ = _args[0];
lean_object* v_S_2151_ = _args[1];
lean_object* v_f_2152_ = _args[2];
lean_object* v_hf_2153_ = _args[3];
lean_object* v_inst_2154_ = _args[4];
lean_object* v_inst_2155_ = _args[5];
lean_object* v_inst_2156_ = _args[6];
lean_object* v_inst_2157_ = _args[7];
lean_object* v_inst_2158_ = _args[8];
lean_object* v_inst_2159_ = _args[9];
lean_object* v_inst_2160_ = _args[10];
lean_object* v_inst_2161_ = _args[11];
lean_object* v_zero_2162_ = _args[12];
lean_object* v_add_2163_ = _args[13];
lean_object* v_mul_2164_ = _args[14];
lean_object* v_neg_2165_ = _args[15];
lean_object* v_sub_2166_ = _args[16];
lean_object* v_nsmul_2167_ = _args[17];
lean_object* v_zsmul_2168_ = _args[18];
_start:
{
lean_object* v_res_2169_; 
v_res_2169_ = lp_mathlib_Function_Surjective_nonUnitalCommRing(v_R_2150_, v_S_2151_, v_f_2152_, v_hf_2153_, v_inst_2154_, v_inst_2155_, v_inst_2156_, v_inst_2157_, v_inst_2158_, v_inst_2159_, v_inst_2160_, v_inst_2161_, v_zero_2162_, v_add_2163_, v_mul_2164_, v_neg_2165_, v_sub_2166_, v_nsmul_2167_, v_zsmul_2168_);
lean_dec_ref(v_inst_2161_);
lean_dec(v_f_2152_);
return v_res_2169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocCommRing___redArg(lean_object* v_inst_2170_, lean_object* v_inst_2171_, lean_object* v_inst_2172_, lean_object* v_inst_2173_, lean_object* v_inst_2174_, lean_object* v_inst_2175_, lean_object* v_inst_2176_, lean_object* v_inst_2177_, lean_object* v_inst_2178_, lean_object* v_inst_2179_){
_start:
{
lean_object* v___x_2180_; lean_object* v___x_2181_; lean_object* v___x_2182_; lean_object* v___x_2183_; lean_object* v___x_2184_; 
v___x_2180_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_2180_, 0, lean_box(0));
lean_closure_set(v___x_2180_, 1, v_inst_2178_);
v___x_2181_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_2181_, 0, lean_box(0));
lean_closure_set(v___x_2181_, 1, v_inst_2179_);
v___x_2182_ = lp_mathlib_Function_Surjective_subNegMonoid___redArg(v_inst_2170_, v_inst_2172_, v_inst_2176_, v_inst_2174_, v_inst_2175_, v_inst_2177_);
v___x_2183_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2183_, 0, v___x_2182_);
lean_ctor_set(v___x_2183_, 1, v_inst_2171_);
v___x_2184_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2184_, 0, v___x_2183_);
lean_ctor_set(v___x_2184_, 1, v_inst_2173_);
lean_ctor_set(v___x_2184_, 2, v___x_2180_);
lean_ctor_set(v___x_2184_, 3, v___x_2181_);
return v___x_2184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocCommRing(lean_object* v_R_2185_, lean_object* v_S_2186_, lean_object* v_f_2187_, lean_object* v_hf_2188_, lean_object* v_inst_2189_, lean_object* v_inst_2190_, lean_object* v_inst_2191_, lean_object* v_inst_2192_, lean_object* v_inst_2193_, lean_object* v_inst_2194_, lean_object* v_inst_2195_, lean_object* v_inst_2196_, lean_object* v_inst_2197_, lean_object* v_inst_2198_, lean_object* v_inst_2199_, lean_object* v_zero_2200_, lean_object* v_one_2201_, lean_object* v_add_2202_, lean_object* v_mul_2203_, lean_object* v_neg_2204_, lean_object* v_sub_2205_, lean_object* v_nsmul_2206_, lean_object* v_zsmul_2207_, lean_object* v_natCast_2208_, lean_object* v_intCast_2209_){
_start:
{
lean_object* v___x_2210_; lean_object* v___x_2211_; lean_object* v___x_2212_; lean_object* v___x_2213_; lean_object* v___x_2214_; 
v___x_2210_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_2210_, 0, lean_box(0));
lean_closure_set(v___x_2210_, 1, v_inst_2197_);
v___x_2211_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_2211_, 0, lean_box(0));
lean_closure_set(v___x_2211_, 1, v_inst_2198_);
v___x_2212_ = lp_mathlib_Function_Surjective_subNegMonoid___redArg(v_inst_2189_, v_inst_2191_, v_inst_2195_, v_inst_2193_, v_inst_2194_, v_inst_2196_);
v___x_2213_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2213_, 0, v___x_2212_);
lean_ctor_set(v___x_2213_, 1, v_inst_2190_);
v___x_2214_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2214_, 0, v___x_2213_);
lean_ctor_set(v___x_2214_, 1, v_inst_2192_);
lean_ctor_set(v___x_2214_, 2, v___x_2210_);
lean_ctor_set(v___x_2214_, 3, v___x_2211_);
return v___x_2214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_nonAssocCommRing___boxed(lean_object** _args){
lean_object* v_R_2215_ = _args[0];
lean_object* v_S_2216_ = _args[1];
lean_object* v_f_2217_ = _args[2];
lean_object* v_hf_2218_ = _args[3];
lean_object* v_inst_2219_ = _args[4];
lean_object* v_inst_2220_ = _args[5];
lean_object* v_inst_2221_ = _args[6];
lean_object* v_inst_2222_ = _args[7];
lean_object* v_inst_2223_ = _args[8];
lean_object* v_inst_2224_ = _args[9];
lean_object* v_inst_2225_ = _args[10];
lean_object* v_inst_2226_ = _args[11];
lean_object* v_inst_2227_ = _args[12];
lean_object* v_inst_2228_ = _args[13];
lean_object* v_inst_2229_ = _args[14];
lean_object* v_zero_2230_ = _args[15];
lean_object* v_one_2231_ = _args[16];
lean_object* v_add_2232_ = _args[17];
lean_object* v_mul_2233_ = _args[18];
lean_object* v_neg_2234_ = _args[19];
lean_object* v_sub_2235_ = _args[20];
lean_object* v_nsmul_2236_ = _args[21];
lean_object* v_zsmul_2237_ = _args[22];
lean_object* v_natCast_2238_ = _args[23];
lean_object* v_intCast_2239_ = _args[24];
_start:
{
lean_object* v_res_2240_; 
v_res_2240_ = lp_mathlib_Function_Surjective_nonAssocCommRing(v_R_2215_, v_S_2216_, v_f_2217_, v_hf_2218_, v_inst_2219_, v_inst_2220_, v_inst_2221_, v_inst_2222_, v_inst_2223_, v_inst_2224_, v_inst_2225_, v_inst_2226_, v_inst_2227_, v_inst_2228_, v_inst_2229_, v_zero_2230_, v_one_2231_, v_add_2232_, v_mul_2233_, v_neg_2234_, v_sub_2235_, v_nsmul_2236_, v_zsmul_2237_, v_natCast_2238_, v_intCast_2239_);
lean_dec_ref(v_inst_2229_);
lean_dec(v_f_2217_);
return v_res_2240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commRing___redArg(lean_object* v_inst_2241_, lean_object* v_inst_2242_, lean_object* v_inst_2243_, lean_object* v_inst_2244_, lean_object* v_inst_2245_, lean_object* v_inst_2246_, lean_object* v_inst_2247_, lean_object* v_inst_2248_, lean_object* v_inst_2249_, lean_object* v_inst_2250_, lean_object* v_inst_2251_){
_start:
{
lean_object* v___x_2252_; lean_object* v_toNeg_2253_; lean_object* v_toSub_2254_; lean_object* v_toZSMul_2255_; lean_object* v___f_2256_; lean_object* v___x_2257_; lean_object* v___x_2258_; lean_object* v___x_2259_; lean_object* v___x_2260_; lean_object* v___x_2261_; lean_object* v___x_2262_; 
lean_inc(v_inst_2247_);
lean_inc(v_inst_2243_);
lean_inc(v_inst_2241_);
v___x_2252_ = lp_mathlib_Function_Surjective_subNegMonoid___redArg(v_inst_2241_, v_inst_2243_, v_inst_2247_, v_inst_2245_, v_inst_2246_, v_inst_2248_);
v_toNeg_2253_ = lean_ctor_get(v___x_2252_, 1);
lean_inc(v_toNeg_2253_);
v_toSub_2254_ = lean_ctor_get(v___x_2252_, 2);
lean_inc(v_toSub_2254_);
v_toZSMul_2255_ = lean_ctor_get(v___x_2252_, 3);
lean_inc(v_toZSMul_2255_);
lean_dec_ref(v___x_2252_);
v___f_2256_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2256_, 0, v_inst_2249_);
v___x_2257_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_2257_, 0, lean_box(0));
lean_closure_set(v___x_2257_, 1, v_inst_2251_);
v___x_2258_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_2258_, 0, lean_box(0));
lean_closure_set(v___x_2258_, 1, v_inst_2250_);
v___x_2259_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_2241_, v_inst_2243_, v_inst_2247_);
v___x_2260_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2260_, 0, v_inst_2244_);
lean_ctor_set(v___x_2260_, 1, v_inst_2242_);
lean_ctor_set(v___x_2260_, 2, v___f_2256_);
v___x_2261_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2261_, 0, v___x_2259_);
lean_ctor_set(v___x_2261_, 1, v___x_2260_);
lean_ctor_set(v___x_2261_, 2, v___x_2258_);
v___x_2262_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2262_, 0, v___x_2261_);
lean_ctor_set(v___x_2262_, 1, v_toNeg_2253_);
lean_ctor_set(v___x_2262_, 2, v_toSub_2254_);
lean_ctor_set(v___x_2262_, 3, v_toZSMul_2255_);
lean_ctor_set(v___x_2262_, 4, v___x_2257_);
return v___x_2262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commRing(lean_object* v_R_2263_, lean_object* v_S_2264_, lean_object* v_f_2265_, lean_object* v_hf_2266_, lean_object* v_inst_2267_, lean_object* v_inst_2268_, lean_object* v_inst_2269_, lean_object* v_inst_2270_, lean_object* v_inst_2271_, lean_object* v_inst_2272_, lean_object* v_inst_2273_, lean_object* v_inst_2274_, lean_object* v_inst_2275_, lean_object* v_inst_2276_, lean_object* v_inst_2277_, lean_object* v_inst_2278_, lean_object* v_zero_2279_, lean_object* v_one_2280_, lean_object* v_add_2281_, lean_object* v_mul_2282_, lean_object* v_neg_2283_, lean_object* v_sub_2284_, lean_object* v_nsmul_2285_, lean_object* v_zsmul_2286_, lean_object* v_npow_2287_, lean_object* v_natCast_2288_, lean_object* v_intCast_2289_){
_start:
{
lean_object* v___x_2290_; lean_object* v_toNeg_2291_; lean_object* v_toSub_2292_; lean_object* v_toZSMul_2293_; lean_object* v___f_2294_; lean_object* v___x_2295_; lean_object* v___x_2296_; lean_object* v___x_2297_; lean_object* v___x_2298_; lean_object* v___x_2299_; lean_object* v___x_2300_; 
lean_inc(v_inst_2273_);
lean_inc(v_inst_2269_);
lean_inc(v_inst_2267_);
v___x_2290_ = lp_mathlib_Function_Surjective_subNegMonoid___redArg(v_inst_2267_, v_inst_2269_, v_inst_2273_, v_inst_2271_, v_inst_2272_, v_inst_2274_);
v_toNeg_2291_ = lean_ctor_get(v___x_2290_, 1);
lean_inc(v_toNeg_2291_);
v_toSub_2292_ = lean_ctor_get(v___x_2290_, 2);
lean_inc(v_toSub_2292_);
v_toZSMul_2293_ = lean_ctor_get(v___x_2290_, 3);
lean_inc(v_toZSMul_2293_);
lean_dec_ref(v___x_2290_);
v___f_2294_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_semiring___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2294_, 0, v_inst_2275_);
v___x_2295_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_2295_, 0, lean_box(0));
lean_closure_set(v___x_2295_, 1, v_inst_2277_);
v___x_2296_ = lean_alloc_closure((void*)(l_Nat_cast), 3, 2);
lean_closure_set(v___x_2296_, 0, lean_box(0));
lean_closure_set(v___x_2296_, 1, v_inst_2276_);
v___x_2297_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_2267_, v_inst_2269_, v_inst_2273_);
v___x_2298_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2298_, 0, v_inst_2270_);
lean_ctor_set(v___x_2298_, 1, v_inst_2268_);
lean_ctor_set(v___x_2298_, 2, v___f_2294_);
v___x_2299_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2299_, 0, v___x_2297_);
lean_ctor_set(v___x_2299_, 1, v___x_2298_);
lean_ctor_set(v___x_2299_, 2, v___x_2296_);
v___x_2300_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2300_, 0, v___x_2299_);
lean_ctor_set(v___x_2300_, 1, v_toNeg_2291_);
lean_ctor_set(v___x_2300_, 2, v_toSub_2292_);
lean_ctor_set(v___x_2300_, 3, v_toZSMul_2293_);
lean_ctor_set(v___x_2300_, 4, v___x_2295_);
return v___x_2300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commRing___boxed(lean_object** _args){
lean_object* v_R_2301_ = _args[0];
lean_object* v_S_2302_ = _args[1];
lean_object* v_f_2303_ = _args[2];
lean_object* v_hf_2304_ = _args[3];
lean_object* v_inst_2305_ = _args[4];
lean_object* v_inst_2306_ = _args[5];
lean_object* v_inst_2307_ = _args[6];
lean_object* v_inst_2308_ = _args[7];
lean_object* v_inst_2309_ = _args[8];
lean_object* v_inst_2310_ = _args[9];
lean_object* v_inst_2311_ = _args[10];
lean_object* v_inst_2312_ = _args[11];
lean_object* v_inst_2313_ = _args[12];
lean_object* v_inst_2314_ = _args[13];
lean_object* v_inst_2315_ = _args[14];
lean_object* v_inst_2316_ = _args[15];
lean_object* v_zero_2317_ = _args[16];
lean_object* v_one_2318_ = _args[17];
lean_object* v_add_2319_ = _args[18];
lean_object* v_mul_2320_ = _args[19];
lean_object* v_neg_2321_ = _args[20];
lean_object* v_sub_2322_ = _args[21];
lean_object* v_nsmul_2323_ = _args[22];
lean_object* v_zsmul_2324_ = _args[23];
lean_object* v_npow_2325_ = _args[24];
lean_object* v_natCast_2326_ = _args[25];
lean_object* v_intCast_2327_ = _args[26];
_start:
{
lean_object* v_res_2328_; 
v_res_2328_ = lp_mathlib_Function_Surjective_commRing(v_R_2301_, v_S_2302_, v_f_2303_, v_hf_2304_, v_inst_2305_, v_inst_2306_, v_inst_2307_, v_inst_2308_, v_inst_2309_, v_inst_2310_, v_inst_2311_, v_inst_2312_, v_inst_2313_, v_inst_2314_, v_inst_2315_, v_inst_2316_, v_zero_2317_, v_one_2318_, v_add_2319_, v_mul_2320_, v_neg_2321_, v_sub_2322_, v_nsmul_2323_, v_zsmul_2324_, v_npow_2325_, v_natCast_2326_, v_intCast_2327_);
lean_dec_ref(v_inst_2316_);
lean_dec(v_f_2303_);
return v_res_2328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instHasDistribNeg___redArg(lean_object* v_inst_2329_){
_start:
{
lean_object* v___f_2330_; 
v___f_2330_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNeg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_2330_, 0, v_inst_2329_);
return v___f_2330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instHasDistribNeg(lean_object* v_R_2331_, lean_object* v_inst_2332_, lean_object* v_inst_2333_){
_start:
{
lean_object* v___f_2334_; 
v___f_2334_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instNeg___redArg___lam__0), 2, 1);
lean_closure_set(v___f_2334_, 0, v_inst_2333_);
return v___f_2334_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddOpposite_instHasDistribNeg___boxed(lean_object* v_R_2335_, lean_object* v_inst_2336_, lean_object* v_inst_2337_){
_start:
{
lean_object* v_res_2338_; 
v_res_2338_ = lp_mathlib_AddOpposite_instHasDistribNeg(v_R_2335_, v_inst_2336_, v_inst_2337_);
lean_dec(v_inst_2336_);
return v_res_2338_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Opposites(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_InjSurj(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Opposites(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Opposites(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_InjSurj(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Cast_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Opposites(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Ring_InjSurj(builtin);
}
#ifdef __cplusplus
}
#endif
