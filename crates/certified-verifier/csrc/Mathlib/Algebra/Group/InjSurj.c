// Lean compiler output
// Module: Mathlib.Algebra.Group.InjSurj
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Defs public import Mathlib.Logic.Function.Basic public import Mathlib.Tactic.Spread
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
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semigroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semigroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semigroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addSemigroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addSemigroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addSemigroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commMagma___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commMagma___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commMagma(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commMagma___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommMagma___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommMagma___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommMagma(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommMagma___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commSemigroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commSemigroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commSemigroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommSemigroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommSemigroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommSemigroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_leftCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_leftCancelSemigroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_leftCancelSemigroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_leftCancelSemigroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addLeftCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addLeftCancelSemigroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addLeftCancelSemigroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addLeftCancelSemigroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_rightCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_rightCancelSemigroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_rightCancelSemigroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_rightCancelSemigroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addRightCancelSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addRightCancelSemigroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addRightCancelSemigroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addRightCancelSemigroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_mulOneClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_mulOneClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_mulOneClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addZeroClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addZeroClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addZeroClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_monoid___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_monoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_monoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_monoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_leftCancelMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_leftCancelMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_leftCancelMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addLeftCancelMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addLeftCancelMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addLeftCancelMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_rightCancelMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_rightCancelMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_rightCancelMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addRightCancelMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addRightCancelMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addRightCancelMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_cancelMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_cancelMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_cancelMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCancelMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCancelMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCancelMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_cancelCommMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_cancelCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_cancelCommMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCancelCommMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCancelCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCancelCommMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_involutiveInv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_involutiveInv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_involutiveInv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_involutiveInv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_involutiveNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_involutiveNeg___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_involutiveNeg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_involutiveNeg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_invOneClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_invOneClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_invOneClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_negZeroClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_negZeroClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_negZeroClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divInvMonoid___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divInvMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divInvMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divInvMonoid___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_subNegMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_subNegMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_subNegMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_subNegMonoid___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divInvOneMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divInvOneMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divInvOneMonoid___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_subNegZeroMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_subNegZeroMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_subNegZeroMonoid___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionMonoid___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_subtractionMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_subtractionMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_subtractionMonoid___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionCommMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionCommMonoid___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_subtractionCommMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_subtractionCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_subtractionCommMonoid___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_group___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_group(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_group___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addGroup___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addGroup___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commGroup___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commGroup___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommGroup___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommGroup___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_semigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_semigroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_semigroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_semigroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addSemigroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addSemigroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addSemigroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commMagma___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commMagma___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commMagma(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commMagma___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommMagma___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommMagma___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommMagma(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommMagma___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commSemigroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commSemigroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commSemigroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommSemigroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommSemigroup___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommSemigroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommSemigroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulOneClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulOneClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulOneClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addZeroClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addZeroClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addZeroClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_monoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_monoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_monoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommMonoid___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_involutiveInv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_involutiveInv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_involutiveInv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_involutiveInv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_involutiveNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_involutiveNeg___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_involutiveNeg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_involutiveNeg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_divInvMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_divInvMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_divInvMonoid___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_subNegMonoid___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_subNegMonoid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_subNegMonoid___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_group___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_group(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_group___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addGroup___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addGroup___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commGroup___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commGroup___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommGroup___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommGroup___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semigroup___redArg(lean_object* v_inst_1_){
_start:
{
lean_inc(v_inst_1_);
return v_inst_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semigroup___redArg___boxed(lean_object* v_inst_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_Function_Injective_semigroup___redArg(v_inst_2_);
lean_dec(v_inst_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semigroup(lean_object* v_M_u2081_4_, lean_object* v_M_u2082_5_, lean_object* v_inst_6_, lean_object* v_inst_7_, lean_object* v_f_8_, lean_object* v_hf_9_, lean_object* v_mul_10_){
_start:
{
lean_inc(v_inst_6_);
return v_inst_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_semigroup___boxed(lean_object* v_M_u2081_11_, lean_object* v_M_u2082_12_, lean_object* v_inst_13_, lean_object* v_inst_14_, lean_object* v_f_15_, lean_object* v_hf_16_, lean_object* v_mul_17_){
_start:
{
lean_object* v_res_18_; 
v_res_18_ = lp_mathlib_Function_Injective_semigroup(v_M_u2081_11_, v_M_u2082_12_, v_inst_13_, v_inst_14_, v_f_15_, v_hf_16_, v_mul_17_);
lean_dec(v_f_15_);
lean_dec(v_inst_14_);
lean_dec(v_inst_13_);
return v_res_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addSemigroup___redArg(lean_object* v_inst_19_){
_start:
{
lean_inc(v_inst_19_);
return v_inst_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addSemigroup___redArg___boxed(lean_object* v_inst_20_){
_start:
{
lean_object* v_res_21_; 
v_res_21_ = lp_mathlib_Function_Injective_addSemigroup___redArg(v_inst_20_);
lean_dec(v_inst_20_);
return v_res_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addSemigroup(lean_object* v_M_u2081_22_, lean_object* v_M_u2082_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_f_26_, lean_object* v_hf_27_, lean_object* v_mul_28_){
_start:
{
lean_inc(v_inst_24_);
return v_inst_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addSemigroup___boxed(lean_object* v_M_u2081_29_, lean_object* v_M_u2082_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_f_33_, lean_object* v_hf_34_, lean_object* v_mul_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_mathlib_Function_Injective_addSemigroup(v_M_u2081_29_, v_M_u2082_30_, v_inst_31_, v_inst_32_, v_f_33_, v_hf_34_, v_mul_35_);
lean_dec(v_f_33_);
lean_dec(v_inst_32_);
lean_dec(v_inst_31_);
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commMagma___redArg(lean_object* v_inst_37_){
_start:
{
lean_inc(v_inst_37_);
return v_inst_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commMagma___redArg___boxed(lean_object* v_inst_38_){
_start:
{
lean_object* v_res_39_; 
v_res_39_ = lp_mathlib_Function_Injective_commMagma___redArg(v_inst_38_);
lean_dec(v_inst_38_);
return v_res_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commMagma(lean_object* v_M_u2081_40_, lean_object* v_M_u2082_41_, lean_object* v_inst_42_, lean_object* v_inst_43_, lean_object* v_f_44_, lean_object* v_hf_45_, lean_object* v_mul_46_){
_start:
{
lean_inc(v_inst_42_);
return v_inst_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commMagma___boxed(lean_object* v_M_u2081_47_, lean_object* v_M_u2082_48_, lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_f_51_, lean_object* v_hf_52_, lean_object* v_mul_53_){
_start:
{
lean_object* v_res_54_; 
v_res_54_ = lp_mathlib_Function_Injective_commMagma(v_M_u2081_47_, v_M_u2082_48_, v_inst_49_, v_inst_50_, v_f_51_, v_hf_52_, v_mul_53_);
lean_dec(v_f_51_);
lean_dec(v_inst_50_);
lean_dec(v_inst_49_);
return v_res_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommMagma___redArg(lean_object* v_inst_55_){
_start:
{
lean_inc(v_inst_55_);
return v_inst_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommMagma___redArg___boxed(lean_object* v_inst_56_){
_start:
{
lean_object* v_res_57_; 
v_res_57_ = lp_mathlib_Function_Injective_addCommMagma___redArg(v_inst_56_);
lean_dec(v_inst_56_);
return v_res_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommMagma(lean_object* v_M_u2081_58_, lean_object* v_M_u2082_59_, lean_object* v_inst_60_, lean_object* v_inst_61_, lean_object* v_f_62_, lean_object* v_hf_63_, lean_object* v_mul_64_){
_start:
{
lean_inc(v_inst_60_);
return v_inst_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommMagma___boxed(lean_object* v_M_u2081_65_, lean_object* v_M_u2082_66_, lean_object* v_inst_67_, lean_object* v_inst_68_, lean_object* v_f_69_, lean_object* v_hf_70_, lean_object* v_mul_71_){
_start:
{
lean_object* v_res_72_; 
v_res_72_ = lp_mathlib_Function_Injective_addCommMagma(v_M_u2081_65_, v_M_u2082_66_, v_inst_67_, v_inst_68_, v_f_69_, v_hf_70_, v_mul_71_);
lean_dec(v_f_69_);
lean_dec(v_inst_68_);
lean_dec(v_inst_67_);
return v_res_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commSemigroup___redArg(lean_object* v_inst_73_){
_start:
{
lean_inc(v_inst_73_);
return v_inst_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commSemigroup___redArg___boxed(lean_object* v_inst_74_){
_start:
{
lean_object* v_res_75_; 
v_res_75_ = lp_mathlib_Function_Injective_commSemigroup___redArg(v_inst_74_);
lean_dec(v_inst_74_);
return v_res_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commSemigroup(lean_object* v_M_u2081_76_, lean_object* v_M_u2082_77_, lean_object* v_inst_78_, lean_object* v_inst_79_, lean_object* v_f_80_, lean_object* v_hf_81_, lean_object* v_mul_82_){
_start:
{
lean_inc(v_inst_78_);
return v_inst_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commSemigroup___boxed(lean_object* v_M_u2081_83_, lean_object* v_M_u2082_84_, lean_object* v_inst_85_, lean_object* v_inst_86_, lean_object* v_f_87_, lean_object* v_hf_88_, lean_object* v_mul_89_){
_start:
{
lean_object* v_res_90_; 
v_res_90_ = lp_mathlib_Function_Injective_commSemigroup(v_M_u2081_83_, v_M_u2082_84_, v_inst_85_, v_inst_86_, v_f_87_, v_hf_88_, v_mul_89_);
lean_dec(v_f_87_);
lean_dec(v_inst_86_);
lean_dec(v_inst_85_);
return v_res_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommSemigroup___redArg(lean_object* v_inst_91_){
_start:
{
lean_inc(v_inst_91_);
return v_inst_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommSemigroup___redArg___boxed(lean_object* v_inst_92_){
_start:
{
lean_object* v_res_93_; 
v_res_93_ = lp_mathlib_Function_Injective_addCommSemigroup___redArg(v_inst_92_);
lean_dec(v_inst_92_);
return v_res_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommSemigroup(lean_object* v_M_u2081_94_, lean_object* v_M_u2082_95_, lean_object* v_inst_96_, lean_object* v_inst_97_, lean_object* v_f_98_, lean_object* v_hf_99_, lean_object* v_mul_100_){
_start:
{
lean_inc(v_inst_96_);
return v_inst_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommSemigroup___boxed(lean_object* v_M_u2081_101_, lean_object* v_M_u2082_102_, lean_object* v_inst_103_, lean_object* v_inst_104_, lean_object* v_f_105_, lean_object* v_hf_106_, lean_object* v_mul_107_){
_start:
{
lean_object* v_res_108_; 
v_res_108_ = lp_mathlib_Function_Injective_addCommSemigroup(v_M_u2081_101_, v_M_u2082_102_, v_inst_103_, v_inst_104_, v_f_105_, v_hf_106_, v_mul_107_);
lean_dec(v_f_105_);
lean_dec(v_inst_104_);
lean_dec(v_inst_103_);
return v_res_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_leftCancelSemigroup___redArg(lean_object* v_inst_109_){
_start:
{
lean_inc(v_inst_109_);
return v_inst_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_leftCancelSemigroup___redArg___boxed(lean_object* v_inst_110_){
_start:
{
lean_object* v_res_111_; 
v_res_111_ = lp_mathlib_Function_Injective_leftCancelSemigroup___redArg(v_inst_110_);
lean_dec(v_inst_110_);
return v_res_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_leftCancelSemigroup(lean_object* v_M_u2081_112_, lean_object* v_M_u2082_113_, lean_object* v_inst_114_, lean_object* v_inst_115_, lean_object* v_f_116_, lean_object* v_hf_117_, lean_object* v_mul_118_){
_start:
{
lean_inc(v_inst_114_);
return v_inst_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_leftCancelSemigroup___boxed(lean_object* v_M_u2081_119_, lean_object* v_M_u2082_120_, lean_object* v_inst_121_, lean_object* v_inst_122_, lean_object* v_f_123_, lean_object* v_hf_124_, lean_object* v_mul_125_){
_start:
{
lean_object* v_res_126_; 
v_res_126_ = lp_mathlib_Function_Injective_leftCancelSemigroup(v_M_u2081_119_, v_M_u2082_120_, v_inst_121_, v_inst_122_, v_f_123_, v_hf_124_, v_mul_125_);
lean_dec(v_f_123_);
lean_dec(v_inst_122_);
lean_dec(v_inst_121_);
return v_res_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addLeftCancelSemigroup___redArg(lean_object* v_inst_127_){
_start:
{
lean_inc(v_inst_127_);
return v_inst_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addLeftCancelSemigroup___redArg___boxed(lean_object* v_inst_128_){
_start:
{
lean_object* v_res_129_; 
v_res_129_ = lp_mathlib_Function_Injective_addLeftCancelSemigroup___redArg(v_inst_128_);
lean_dec(v_inst_128_);
return v_res_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addLeftCancelSemigroup(lean_object* v_M_u2081_130_, lean_object* v_M_u2082_131_, lean_object* v_inst_132_, lean_object* v_inst_133_, lean_object* v_f_134_, lean_object* v_hf_135_, lean_object* v_mul_136_){
_start:
{
lean_inc(v_inst_132_);
return v_inst_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addLeftCancelSemigroup___boxed(lean_object* v_M_u2081_137_, lean_object* v_M_u2082_138_, lean_object* v_inst_139_, lean_object* v_inst_140_, lean_object* v_f_141_, lean_object* v_hf_142_, lean_object* v_mul_143_){
_start:
{
lean_object* v_res_144_; 
v_res_144_ = lp_mathlib_Function_Injective_addLeftCancelSemigroup(v_M_u2081_137_, v_M_u2082_138_, v_inst_139_, v_inst_140_, v_f_141_, v_hf_142_, v_mul_143_);
lean_dec(v_f_141_);
lean_dec(v_inst_140_);
lean_dec(v_inst_139_);
return v_res_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_rightCancelSemigroup___redArg(lean_object* v_inst_145_){
_start:
{
lean_inc(v_inst_145_);
return v_inst_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_rightCancelSemigroup___redArg___boxed(lean_object* v_inst_146_){
_start:
{
lean_object* v_res_147_; 
v_res_147_ = lp_mathlib_Function_Injective_rightCancelSemigroup___redArg(v_inst_146_);
lean_dec(v_inst_146_);
return v_res_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_rightCancelSemigroup(lean_object* v_M_u2081_148_, lean_object* v_M_u2082_149_, lean_object* v_inst_150_, lean_object* v_inst_151_, lean_object* v_f_152_, lean_object* v_hf_153_, lean_object* v_mul_154_){
_start:
{
lean_inc(v_inst_150_);
return v_inst_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_rightCancelSemigroup___boxed(lean_object* v_M_u2081_155_, lean_object* v_M_u2082_156_, lean_object* v_inst_157_, lean_object* v_inst_158_, lean_object* v_f_159_, lean_object* v_hf_160_, lean_object* v_mul_161_){
_start:
{
lean_object* v_res_162_; 
v_res_162_ = lp_mathlib_Function_Injective_rightCancelSemigroup(v_M_u2081_155_, v_M_u2082_156_, v_inst_157_, v_inst_158_, v_f_159_, v_hf_160_, v_mul_161_);
lean_dec(v_f_159_);
lean_dec(v_inst_158_);
lean_dec(v_inst_157_);
return v_res_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addRightCancelSemigroup___redArg(lean_object* v_inst_163_){
_start:
{
lean_inc(v_inst_163_);
return v_inst_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addRightCancelSemigroup___redArg___boxed(lean_object* v_inst_164_){
_start:
{
lean_object* v_res_165_; 
v_res_165_ = lp_mathlib_Function_Injective_addRightCancelSemigroup___redArg(v_inst_164_);
lean_dec(v_inst_164_);
return v_res_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addRightCancelSemigroup(lean_object* v_M_u2081_166_, lean_object* v_M_u2082_167_, lean_object* v_inst_168_, lean_object* v_inst_169_, lean_object* v_f_170_, lean_object* v_hf_171_, lean_object* v_mul_172_){
_start:
{
lean_inc(v_inst_168_);
return v_inst_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addRightCancelSemigroup___boxed(lean_object* v_M_u2081_173_, lean_object* v_M_u2082_174_, lean_object* v_inst_175_, lean_object* v_inst_176_, lean_object* v_f_177_, lean_object* v_hf_178_, lean_object* v_mul_179_){
_start:
{
lean_object* v_res_180_; 
v_res_180_ = lp_mathlib_Function_Injective_addRightCancelSemigroup(v_M_u2081_173_, v_M_u2082_174_, v_inst_175_, v_inst_176_, v_f_177_, v_hf_178_, v_mul_179_);
lean_dec(v_f_177_);
lean_dec(v_inst_176_);
lean_dec(v_inst_175_);
return v_res_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_mulOneClass___redArg(lean_object* v_inst_181_, lean_object* v_inst_182_){
_start:
{
lean_object* v___x_183_; 
v___x_183_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_183_, 0, v_inst_182_);
lean_ctor_set(v___x_183_, 1, v_inst_181_);
return v___x_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_mulOneClass(lean_object* v_M_u2081_184_, lean_object* v_M_u2082_185_, lean_object* v_inst_186_, lean_object* v_inst_187_, lean_object* v_inst_188_, lean_object* v_f_189_, lean_object* v_hf_190_, lean_object* v_one_191_, lean_object* v_mul_192_){
_start:
{
lean_object* v___x_193_; 
v___x_193_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_193_, 0, v_inst_187_);
lean_ctor_set(v___x_193_, 1, v_inst_186_);
return v___x_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_mulOneClass___boxed(lean_object* v_M_u2081_194_, lean_object* v_M_u2082_195_, lean_object* v_inst_196_, lean_object* v_inst_197_, lean_object* v_inst_198_, lean_object* v_f_199_, lean_object* v_hf_200_, lean_object* v_one_201_, lean_object* v_mul_202_){
_start:
{
lean_object* v_res_203_; 
v_res_203_ = lp_mathlib_Function_Injective_mulOneClass(v_M_u2081_194_, v_M_u2082_195_, v_inst_196_, v_inst_197_, v_inst_198_, v_f_199_, v_hf_200_, v_one_201_, v_mul_202_);
lean_dec(v_f_199_);
lean_dec_ref(v_inst_198_);
return v_res_203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addZeroClass___redArg(lean_object* v_inst_204_, lean_object* v_inst_205_){
_start:
{
lean_object* v___x_206_; 
v___x_206_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_206_, 0, v_inst_205_);
lean_ctor_set(v___x_206_, 1, v_inst_204_);
return v___x_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addZeroClass(lean_object* v_M_u2081_207_, lean_object* v_M_u2082_208_, lean_object* v_inst_209_, lean_object* v_inst_210_, lean_object* v_inst_211_, lean_object* v_f_212_, lean_object* v_hf_213_, lean_object* v_one_214_, lean_object* v_mul_215_){
_start:
{
lean_object* v___x_216_; 
v___x_216_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_216_, 0, v_inst_210_);
lean_ctor_set(v___x_216_, 1, v_inst_209_);
return v___x_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addZeroClass___boxed(lean_object* v_M_u2081_217_, lean_object* v_M_u2082_218_, lean_object* v_inst_219_, lean_object* v_inst_220_, lean_object* v_inst_221_, lean_object* v_f_222_, lean_object* v_hf_223_, lean_object* v_one_224_, lean_object* v_mul_225_){
_start:
{
lean_object* v_res_226_; 
v_res_226_ = lp_mathlib_Function_Injective_addZeroClass(v_M_u2081_217_, v_M_u2082_218_, v_inst_219_, v_inst_220_, v_inst_221_, v_f_222_, v_hf_223_, v_one_224_, v_mul_225_);
lean_dec(v_f_222_);
lean_dec_ref(v_inst_221_);
return v_res_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_monoid___redArg___lam__0(lean_object* v_inst_227_, lean_object* v_n_228_, lean_object* v_x_229_){
_start:
{
lean_object* v___x_230_; 
v___x_230_ = lean_apply_2(v_inst_227_, v_x_229_, v_n_228_);
return v___x_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_monoid___redArg(lean_object* v_inst_231_, lean_object* v_inst_232_, lean_object* v_inst_233_){
_start:
{
lean_object* v___f_234_; lean_object* v___x_235_; 
v___f_234_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_234_, 0, v_inst_233_);
v___x_235_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_235_, 0, v_inst_232_);
lean_ctor_set(v___x_235_, 1, v_inst_231_);
lean_ctor_set(v___x_235_, 2, v___f_234_);
return v___x_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_monoid(lean_object* v_M_u2081_236_, lean_object* v_M_u2082_237_, lean_object* v_inst_238_, lean_object* v_inst_239_, lean_object* v_inst_240_, lean_object* v_inst_241_, lean_object* v_f_242_, lean_object* v_hf_243_, lean_object* v_one_244_, lean_object* v_mul_245_, lean_object* v_npow_246_){
_start:
{
lean_object* v___f_247_; lean_object* v___x_248_; 
v___f_247_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_247_, 0, v_inst_240_);
v___x_248_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_248_, 0, v_inst_239_);
lean_ctor_set(v___x_248_, 1, v_inst_238_);
lean_ctor_set(v___x_248_, 2, v___f_247_);
return v___x_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_monoid___boxed(lean_object* v_M_u2081_249_, lean_object* v_M_u2082_250_, lean_object* v_inst_251_, lean_object* v_inst_252_, lean_object* v_inst_253_, lean_object* v_inst_254_, lean_object* v_f_255_, lean_object* v_hf_256_, lean_object* v_one_257_, lean_object* v_mul_258_, lean_object* v_npow_259_){
_start:
{
lean_object* v_res_260_; 
v_res_260_ = lp_mathlib_Function_Injective_monoid(v_M_u2081_249_, v_M_u2082_250_, v_inst_251_, v_inst_252_, v_inst_253_, v_inst_254_, v_f_255_, v_hf_256_, v_one_257_, v_mul_258_, v_npow_259_);
lean_dec(v_f_255_);
lean_dec_ref(v_inst_254_);
return v_res_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addMonoid___redArg___lam__0(lean_object* v_inst_261_, lean_object* v_n_262_, lean_object* v_x_263_){
_start:
{
lean_object* v___x_264_; 
v___x_264_ = lean_apply_2(v_inst_261_, v_n_262_, v_x_263_);
return v___x_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addMonoid___redArg(lean_object* v_inst_265_, lean_object* v_inst_266_, lean_object* v_inst_267_){
_start:
{
lean_object* v___f_268_; lean_object* v___x_269_; 
v___f_268_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_addMonoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_268_, 0, v_inst_267_);
v___x_269_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_269_, 0, v_inst_266_);
lean_ctor_set(v___x_269_, 1, v_inst_265_);
lean_ctor_set(v___x_269_, 2, v___f_268_);
return v___x_269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addMonoid(lean_object* v_M_u2081_270_, lean_object* v_M_u2082_271_, lean_object* v_inst_272_, lean_object* v_inst_273_, lean_object* v_inst_274_, lean_object* v_inst_275_, lean_object* v_f_276_, lean_object* v_hf_277_, lean_object* v_one_278_, lean_object* v_mul_279_, lean_object* v_npow_280_){
_start:
{
lean_object* v___x_281_; 
v___x_281_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_272_, v_inst_273_, v_inst_274_);
return v___x_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addMonoid___boxed(lean_object* v_M_u2081_282_, lean_object* v_M_u2082_283_, lean_object* v_inst_284_, lean_object* v_inst_285_, lean_object* v_inst_286_, lean_object* v_inst_287_, lean_object* v_f_288_, lean_object* v_hf_289_, lean_object* v_one_290_, lean_object* v_mul_291_, lean_object* v_npow_292_){
_start:
{
lean_object* v_res_293_; 
v_res_293_ = lp_mathlib_Function_Injective_addMonoid(v_M_u2081_282_, v_M_u2082_283_, v_inst_284_, v_inst_285_, v_inst_286_, v_inst_287_, v_f_288_, v_hf_289_, v_one_290_, v_mul_291_, v_npow_292_);
lean_dec(v_f_288_);
lean_dec_ref(v_inst_287_);
return v_res_293_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_leftCancelMonoid___redArg(lean_object* v_inst_294_, lean_object* v_inst_295_, lean_object* v_inst_296_){
_start:
{
lean_object* v___f_297_; lean_object* v___x_298_; 
v___f_297_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_297_, 0, v_inst_296_);
v___x_298_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_298_, 0, v_inst_295_);
lean_ctor_set(v___x_298_, 1, v_inst_294_);
lean_ctor_set(v___x_298_, 2, v___f_297_);
return v___x_298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_leftCancelMonoid(lean_object* v_M_u2081_299_, lean_object* v_M_u2082_300_, lean_object* v_inst_301_, lean_object* v_inst_302_, lean_object* v_inst_303_, lean_object* v_inst_304_, lean_object* v_f_305_, lean_object* v_hf_306_, lean_object* v_one_307_, lean_object* v_mul_308_, lean_object* v_npow_309_){
_start:
{
lean_object* v___f_310_; lean_object* v___x_311_; 
v___f_310_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_310_, 0, v_inst_303_);
v___x_311_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_311_, 0, v_inst_302_);
lean_ctor_set(v___x_311_, 1, v_inst_301_);
lean_ctor_set(v___x_311_, 2, v___f_310_);
return v___x_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_leftCancelMonoid___boxed(lean_object* v_M_u2081_312_, lean_object* v_M_u2082_313_, lean_object* v_inst_314_, lean_object* v_inst_315_, lean_object* v_inst_316_, lean_object* v_inst_317_, lean_object* v_f_318_, lean_object* v_hf_319_, lean_object* v_one_320_, lean_object* v_mul_321_, lean_object* v_npow_322_){
_start:
{
lean_object* v_res_323_; 
v_res_323_ = lp_mathlib_Function_Injective_leftCancelMonoid(v_M_u2081_312_, v_M_u2082_313_, v_inst_314_, v_inst_315_, v_inst_316_, v_inst_317_, v_f_318_, v_hf_319_, v_one_320_, v_mul_321_, v_npow_322_);
lean_dec(v_f_318_);
lean_dec_ref(v_inst_317_);
return v_res_323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addLeftCancelMonoid___redArg(lean_object* v_inst_324_, lean_object* v_inst_325_, lean_object* v_inst_326_){
_start:
{
lean_object* v___x_327_; 
v___x_327_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_324_, v_inst_325_, v_inst_326_);
return v___x_327_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addLeftCancelMonoid(lean_object* v_M_u2081_328_, lean_object* v_M_u2082_329_, lean_object* v_inst_330_, lean_object* v_inst_331_, lean_object* v_inst_332_, lean_object* v_inst_333_, lean_object* v_f_334_, lean_object* v_hf_335_, lean_object* v_one_336_, lean_object* v_mul_337_, lean_object* v_npow_338_){
_start:
{
lean_object* v___x_339_; 
v___x_339_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_330_, v_inst_331_, v_inst_332_);
return v___x_339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addLeftCancelMonoid___boxed(lean_object* v_M_u2081_340_, lean_object* v_M_u2082_341_, lean_object* v_inst_342_, lean_object* v_inst_343_, lean_object* v_inst_344_, lean_object* v_inst_345_, lean_object* v_f_346_, lean_object* v_hf_347_, lean_object* v_one_348_, lean_object* v_mul_349_, lean_object* v_npow_350_){
_start:
{
lean_object* v_res_351_; 
v_res_351_ = lp_mathlib_Function_Injective_addLeftCancelMonoid(v_M_u2081_340_, v_M_u2082_341_, v_inst_342_, v_inst_343_, v_inst_344_, v_inst_345_, v_f_346_, v_hf_347_, v_one_348_, v_mul_349_, v_npow_350_);
lean_dec(v_f_346_);
lean_dec_ref(v_inst_345_);
return v_res_351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_rightCancelMonoid___redArg(lean_object* v_inst_352_, lean_object* v_inst_353_, lean_object* v_inst_354_){
_start:
{
lean_object* v___f_355_; lean_object* v___x_356_; 
v___f_355_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_355_, 0, v_inst_354_);
v___x_356_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_356_, 0, v_inst_353_);
lean_ctor_set(v___x_356_, 1, v_inst_352_);
lean_ctor_set(v___x_356_, 2, v___f_355_);
return v___x_356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_rightCancelMonoid(lean_object* v_M_u2081_357_, lean_object* v_M_u2082_358_, lean_object* v_inst_359_, lean_object* v_inst_360_, lean_object* v_inst_361_, lean_object* v_inst_362_, lean_object* v_f_363_, lean_object* v_hf_364_, lean_object* v_one_365_, lean_object* v_mul_366_, lean_object* v_npow_367_){
_start:
{
lean_object* v___f_368_; lean_object* v___x_369_; 
v___f_368_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_368_, 0, v_inst_361_);
v___x_369_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_369_, 0, v_inst_360_);
lean_ctor_set(v___x_369_, 1, v_inst_359_);
lean_ctor_set(v___x_369_, 2, v___f_368_);
return v___x_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_rightCancelMonoid___boxed(lean_object* v_M_u2081_370_, lean_object* v_M_u2082_371_, lean_object* v_inst_372_, lean_object* v_inst_373_, lean_object* v_inst_374_, lean_object* v_inst_375_, lean_object* v_f_376_, lean_object* v_hf_377_, lean_object* v_one_378_, lean_object* v_mul_379_, lean_object* v_npow_380_){
_start:
{
lean_object* v_res_381_; 
v_res_381_ = lp_mathlib_Function_Injective_rightCancelMonoid(v_M_u2081_370_, v_M_u2082_371_, v_inst_372_, v_inst_373_, v_inst_374_, v_inst_375_, v_f_376_, v_hf_377_, v_one_378_, v_mul_379_, v_npow_380_);
lean_dec(v_f_376_);
lean_dec_ref(v_inst_375_);
return v_res_381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addRightCancelMonoid___redArg(lean_object* v_inst_382_, lean_object* v_inst_383_, lean_object* v_inst_384_){
_start:
{
lean_object* v___x_385_; 
v___x_385_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_382_, v_inst_383_, v_inst_384_);
return v___x_385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addRightCancelMonoid(lean_object* v_M_u2081_386_, lean_object* v_M_u2082_387_, lean_object* v_inst_388_, lean_object* v_inst_389_, lean_object* v_inst_390_, lean_object* v_inst_391_, lean_object* v_f_392_, lean_object* v_hf_393_, lean_object* v_one_394_, lean_object* v_mul_395_, lean_object* v_npow_396_){
_start:
{
lean_object* v___x_397_; 
v___x_397_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_388_, v_inst_389_, v_inst_390_);
return v___x_397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addRightCancelMonoid___boxed(lean_object* v_M_u2081_398_, lean_object* v_M_u2082_399_, lean_object* v_inst_400_, lean_object* v_inst_401_, lean_object* v_inst_402_, lean_object* v_inst_403_, lean_object* v_f_404_, lean_object* v_hf_405_, lean_object* v_one_406_, lean_object* v_mul_407_, lean_object* v_npow_408_){
_start:
{
lean_object* v_res_409_; 
v_res_409_ = lp_mathlib_Function_Injective_addRightCancelMonoid(v_M_u2081_398_, v_M_u2082_399_, v_inst_400_, v_inst_401_, v_inst_402_, v_inst_403_, v_f_404_, v_hf_405_, v_one_406_, v_mul_407_, v_npow_408_);
lean_dec(v_f_404_);
lean_dec_ref(v_inst_403_);
return v_res_409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_cancelMonoid___redArg(lean_object* v_inst_410_, lean_object* v_inst_411_, lean_object* v_inst_412_){
_start:
{
lean_object* v___f_413_; lean_object* v___x_414_; 
v___f_413_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_413_, 0, v_inst_412_);
v___x_414_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_414_, 0, v_inst_411_);
lean_ctor_set(v___x_414_, 1, v_inst_410_);
lean_ctor_set(v___x_414_, 2, v___f_413_);
return v___x_414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_cancelMonoid(lean_object* v_M_u2081_415_, lean_object* v_M_u2082_416_, lean_object* v_inst_417_, lean_object* v_inst_418_, lean_object* v_inst_419_, lean_object* v_inst_420_, lean_object* v_f_421_, lean_object* v_hf_422_, lean_object* v_one_423_, lean_object* v_mul_424_, lean_object* v_npow_425_){
_start:
{
lean_object* v___f_426_; lean_object* v___x_427_; 
v___f_426_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_426_, 0, v_inst_419_);
v___x_427_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_427_, 0, v_inst_418_);
lean_ctor_set(v___x_427_, 1, v_inst_417_);
lean_ctor_set(v___x_427_, 2, v___f_426_);
return v___x_427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_cancelMonoid___boxed(lean_object* v_M_u2081_428_, lean_object* v_M_u2082_429_, lean_object* v_inst_430_, lean_object* v_inst_431_, lean_object* v_inst_432_, lean_object* v_inst_433_, lean_object* v_f_434_, lean_object* v_hf_435_, lean_object* v_one_436_, lean_object* v_mul_437_, lean_object* v_npow_438_){
_start:
{
lean_object* v_res_439_; 
v_res_439_ = lp_mathlib_Function_Injective_cancelMonoid(v_M_u2081_428_, v_M_u2082_429_, v_inst_430_, v_inst_431_, v_inst_432_, v_inst_433_, v_f_434_, v_hf_435_, v_one_436_, v_mul_437_, v_npow_438_);
lean_dec(v_f_434_);
lean_dec_ref(v_inst_433_);
return v_res_439_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCancelMonoid___redArg(lean_object* v_inst_440_, lean_object* v_inst_441_, lean_object* v_inst_442_){
_start:
{
lean_object* v___x_443_; 
v___x_443_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_440_, v_inst_441_, v_inst_442_);
return v___x_443_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCancelMonoid(lean_object* v_M_u2081_444_, lean_object* v_M_u2082_445_, lean_object* v_inst_446_, lean_object* v_inst_447_, lean_object* v_inst_448_, lean_object* v_inst_449_, lean_object* v_f_450_, lean_object* v_hf_451_, lean_object* v_one_452_, lean_object* v_mul_453_, lean_object* v_npow_454_){
_start:
{
lean_object* v___x_455_; 
v___x_455_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_446_, v_inst_447_, v_inst_448_);
return v___x_455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCancelMonoid___boxed(lean_object* v_M_u2081_456_, lean_object* v_M_u2082_457_, lean_object* v_inst_458_, lean_object* v_inst_459_, lean_object* v_inst_460_, lean_object* v_inst_461_, lean_object* v_f_462_, lean_object* v_hf_463_, lean_object* v_one_464_, lean_object* v_mul_465_, lean_object* v_npow_466_){
_start:
{
lean_object* v_res_467_; 
v_res_467_ = lp_mathlib_Function_Injective_addCancelMonoid(v_M_u2081_456_, v_M_u2082_457_, v_inst_458_, v_inst_459_, v_inst_460_, v_inst_461_, v_f_462_, v_hf_463_, v_one_464_, v_mul_465_, v_npow_466_);
lean_dec(v_f_462_);
lean_dec_ref(v_inst_461_);
return v_res_467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commMonoid___redArg(lean_object* v_inst_468_, lean_object* v_inst_469_, lean_object* v_inst_470_){
_start:
{
lean_object* v___f_471_; lean_object* v___x_472_; 
v___f_471_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_471_, 0, v_inst_470_);
v___x_472_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_472_, 0, v_inst_469_);
lean_ctor_set(v___x_472_, 1, v_inst_468_);
lean_ctor_set(v___x_472_, 2, v___f_471_);
return v___x_472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commMonoid(lean_object* v_M_u2081_473_, lean_object* v_M_u2082_474_, lean_object* v_inst_475_, lean_object* v_inst_476_, lean_object* v_inst_477_, lean_object* v_inst_478_, lean_object* v_f_479_, lean_object* v_hf_480_, lean_object* v_one_481_, lean_object* v_mul_482_, lean_object* v_npow_483_){
_start:
{
lean_object* v___f_484_; lean_object* v___x_485_; 
v___f_484_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_484_, 0, v_inst_477_);
v___x_485_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_485_, 0, v_inst_476_);
lean_ctor_set(v___x_485_, 1, v_inst_475_);
lean_ctor_set(v___x_485_, 2, v___f_484_);
return v___x_485_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commMonoid___boxed(lean_object* v_M_u2081_486_, lean_object* v_M_u2082_487_, lean_object* v_inst_488_, lean_object* v_inst_489_, lean_object* v_inst_490_, lean_object* v_inst_491_, lean_object* v_f_492_, lean_object* v_hf_493_, lean_object* v_one_494_, lean_object* v_mul_495_, lean_object* v_npow_496_){
_start:
{
lean_object* v_res_497_; 
v_res_497_ = lp_mathlib_Function_Injective_commMonoid(v_M_u2081_486_, v_M_u2082_487_, v_inst_488_, v_inst_489_, v_inst_490_, v_inst_491_, v_f_492_, v_hf_493_, v_one_494_, v_mul_495_, v_npow_496_);
lean_dec(v_f_492_);
lean_dec_ref(v_inst_491_);
return v_res_497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommMonoid___redArg(lean_object* v_inst_498_, lean_object* v_inst_499_, lean_object* v_inst_500_){
_start:
{
lean_object* v___x_501_; 
v___x_501_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_498_, v_inst_499_, v_inst_500_);
return v___x_501_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommMonoid(lean_object* v_M_u2081_502_, lean_object* v_M_u2082_503_, lean_object* v_inst_504_, lean_object* v_inst_505_, lean_object* v_inst_506_, lean_object* v_inst_507_, lean_object* v_f_508_, lean_object* v_hf_509_, lean_object* v_one_510_, lean_object* v_mul_511_, lean_object* v_npow_512_){
_start:
{
lean_object* v___x_513_; 
v___x_513_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_504_, v_inst_505_, v_inst_506_);
return v___x_513_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommMonoid___boxed(lean_object* v_M_u2081_514_, lean_object* v_M_u2082_515_, lean_object* v_inst_516_, lean_object* v_inst_517_, lean_object* v_inst_518_, lean_object* v_inst_519_, lean_object* v_f_520_, lean_object* v_hf_521_, lean_object* v_one_522_, lean_object* v_mul_523_, lean_object* v_npow_524_){
_start:
{
lean_object* v_res_525_; 
v_res_525_ = lp_mathlib_Function_Injective_addCommMonoid(v_M_u2081_514_, v_M_u2082_515_, v_inst_516_, v_inst_517_, v_inst_518_, v_inst_519_, v_f_520_, v_hf_521_, v_one_522_, v_mul_523_, v_npow_524_);
lean_dec(v_f_520_);
lean_dec_ref(v_inst_519_);
return v_res_525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_cancelCommMonoid___redArg(lean_object* v_inst_526_, lean_object* v_inst_527_, lean_object* v_inst_528_){
_start:
{
lean_object* v___f_529_; lean_object* v___x_530_; 
v___f_529_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_529_, 0, v_inst_528_);
v___x_530_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_530_, 0, v_inst_527_);
lean_ctor_set(v___x_530_, 1, v_inst_526_);
lean_ctor_set(v___x_530_, 2, v___f_529_);
return v___x_530_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_cancelCommMonoid(lean_object* v_M_u2081_531_, lean_object* v_M_u2082_532_, lean_object* v_inst_533_, lean_object* v_inst_534_, lean_object* v_inst_535_, lean_object* v_inst_536_, lean_object* v_f_537_, lean_object* v_hf_538_, lean_object* v_one_539_, lean_object* v_mul_540_, lean_object* v_npow_541_){
_start:
{
lean_object* v___f_542_; lean_object* v___x_543_; 
v___f_542_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_542_, 0, v_inst_535_);
v___x_543_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_543_, 0, v_inst_534_);
lean_ctor_set(v___x_543_, 1, v_inst_533_);
lean_ctor_set(v___x_543_, 2, v___f_542_);
return v___x_543_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_cancelCommMonoid___boxed(lean_object* v_M_u2081_544_, lean_object* v_M_u2082_545_, lean_object* v_inst_546_, lean_object* v_inst_547_, lean_object* v_inst_548_, lean_object* v_inst_549_, lean_object* v_f_550_, lean_object* v_hf_551_, lean_object* v_one_552_, lean_object* v_mul_553_, lean_object* v_npow_554_){
_start:
{
lean_object* v_res_555_; 
v_res_555_ = lp_mathlib_Function_Injective_cancelCommMonoid(v_M_u2081_544_, v_M_u2082_545_, v_inst_546_, v_inst_547_, v_inst_548_, v_inst_549_, v_f_550_, v_hf_551_, v_one_552_, v_mul_553_, v_npow_554_);
lean_dec(v_f_550_);
lean_dec_ref(v_inst_549_);
return v_res_555_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCancelCommMonoid___redArg(lean_object* v_inst_556_, lean_object* v_inst_557_, lean_object* v_inst_558_){
_start:
{
lean_object* v___x_559_; 
v___x_559_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_556_, v_inst_557_, v_inst_558_);
return v___x_559_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCancelCommMonoid(lean_object* v_M_u2081_560_, lean_object* v_M_u2082_561_, lean_object* v_inst_562_, lean_object* v_inst_563_, lean_object* v_inst_564_, lean_object* v_inst_565_, lean_object* v_f_566_, lean_object* v_hf_567_, lean_object* v_one_568_, lean_object* v_mul_569_, lean_object* v_npow_570_){
_start:
{
lean_object* v___x_571_; 
v___x_571_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_562_, v_inst_563_, v_inst_564_);
return v___x_571_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCancelCommMonoid___boxed(lean_object* v_M_u2081_572_, lean_object* v_M_u2082_573_, lean_object* v_inst_574_, lean_object* v_inst_575_, lean_object* v_inst_576_, lean_object* v_inst_577_, lean_object* v_f_578_, lean_object* v_hf_579_, lean_object* v_one_580_, lean_object* v_mul_581_, lean_object* v_npow_582_){
_start:
{
lean_object* v_res_583_; 
v_res_583_ = lp_mathlib_Function_Injective_addCancelCommMonoid(v_M_u2081_572_, v_M_u2082_573_, v_inst_574_, v_inst_575_, v_inst_576_, v_inst_577_, v_f_578_, v_hf_579_, v_one_580_, v_mul_581_, v_npow_582_);
lean_dec(v_f_578_);
lean_dec_ref(v_inst_577_);
return v_res_583_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_involutiveInv___redArg(lean_object* v_inst_584_){
_start:
{
lean_inc(v_inst_584_);
return v_inst_584_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_involutiveInv___redArg___boxed(lean_object* v_inst_585_){
_start:
{
lean_object* v_res_586_; 
v_res_586_ = lp_mathlib_Function_Injective_involutiveInv___redArg(v_inst_585_);
lean_dec(v_inst_585_);
return v_res_586_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_involutiveInv(lean_object* v_M_u2082_587_, lean_object* v_M_u2081_588_, lean_object* v_inst_589_, lean_object* v_inst_590_, lean_object* v_f_591_, lean_object* v_hf_592_, lean_object* v_inv_593_){
_start:
{
lean_inc(v_inst_589_);
return v_inst_589_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_involutiveInv___boxed(lean_object* v_M_u2082_594_, lean_object* v_M_u2081_595_, lean_object* v_inst_596_, lean_object* v_inst_597_, lean_object* v_f_598_, lean_object* v_hf_599_, lean_object* v_inv_600_){
_start:
{
lean_object* v_res_601_; 
v_res_601_ = lp_mathlib_Function_Injective_involutiveInv(v_M_u2082_594_, v_M_u2081_595_, v_inst_596_, v_inst_597_, v_f_598_, v_hf_599_, v_inv_600_);
lean_dec(v_f_598_);
lean_dec(v_inst_597_);
lean_dec(v_inst_596_);
return v_res_601_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_involutiveNeg___redArg(lean_object* v_inst_602_){
_start:
{
lean_inc(v_inst_602_);
return v_inst_602_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_involutiveNeg___redArg___boxed(lean_object* v_inst_603_){
_start:
{
lean_object* v_res_604_; 
v_res_604_ = lp_mathlib_Function_Injective_involutiveNeg___redArg(v_inst_603_);
lean_dec(v_inst_603_);
return v_res_604_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_involutiveNeg(lean_object* v_M_u2082_605_, lean_object* v_M_u2081_606_, lean_object* v_inst_607_, lean_object* v_inst_608_, lean_object* v_f_609_, lean_object* v_hf_610_, lean_object* v_inv_611_){
_start:
{
lean_inc(v_inst_607_);
return v_inst_607_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_involutiveNeg___boxed(lean_object* v_M_u2082_612_, lean_object* v_M_u2081_613_, lean_object* v_inst_614_, lean_object* v_inst_615_, lean_object* v_f_616_, lean_object* v_hf_617_, lean_object* v_inv_618_){
_start:
{
lean_object* v_res_619_; 
v_res_619_ = lp_mathlib_Function_Injective_involutiveNeg(v_M_u2082_612_, v_M_u2081_613_, v_inst_614_, v_inst_615_, v_f_616_, v_hf_617_, v_inv_618_);
lean_dec(v_f_616_);
lean_dec(v_inst_615_);
lean_dec(v_inst_614_);
return v_res_619_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_invOneClass___redArg(lean_object* v_inst_620_, lean_object* v_inst_621_){
_start:
{
lean_object* v___x_622_; 
v___x_622_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_622_, 0, v_inst_620_);
lean_ctor_set(v___x_622_, 1, v_inst_621_);
return v___x_622_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_invOneClass(lean_object* v_M_u2081_623_, lean_object* v_M_u2082_624_, lean_object* v_inst_625_, lean_object* v_inst_626_, lean_object* v_inst_627_, lean_object* v_f_628_, lean_object* v_hf_629_, lean_object* v_one_630_, lean_object* v_inv_631_){
_start:
{
lean_object* v___x_632_; 
v___x_632_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_632_, 0, v_inst_625_);
lean_ctor_set(v___x_632_, 1, v_inst_626_);
return v___x_632_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_invOneClass___boxed(lean_object* v_M_u2081_633_, lean_object* v_M_u2082_634_, lean_object* v_inst_635_, lean_object* v_inst_636_, lean_object* v_inst_637_, lean_object* v_f_638_, lean_object* v_hf_639_, lean_object* v_one_640_, lean_object* v_inv_641_){
_start:
{
lean_object* v_res_642_; 
v_res_642_ = lp_mathlib_Function_Injective_invOneClass(v_M_u2081_633_, v_M_u2082_634_, v_inst_635_, v_inst_636_, v_inst_637_, v_f_638_, v_hf_639_, v_one_640_, v_inv_641_);
lean_dec(v_f_638_);
lean_dec_ref(v_inst_637_);
return v_res_642_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_negZeroClass___redArg(lean_object* v_inst_643_, lean_object* v_inst_644_){
_start:
{
lean_object* v___x_645_; 
v___x_645_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_645_, 0, v_inst_643_);
lean_ctor_set(v___x_645_, 1, v_inst_644_);
return v___x_645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_negZeroClass(lean_object* v_M_u2081_646_, lean_object* v_M_u2082_647_, lean_object* v_inst_648_, lean_object* v_inst_649_, lean_object* v_inst_650_, lean_object* v_f_651_, lean_object* v_hf_652_, lean_object* v_one_653_, lean_object* v_inv_654_){
_start:
{
lean_object* v___x_655_; 
v___x_655_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_655_, 0, v_inst_648_);
lean_ctor_set(v___x_655_, 1, v_inst_649_);
return v___x_655_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_negZeroClass___boxed(lean_object* v_M_u2081_656_, lean_object* v_M_u2082_657_, lean_object* v_inst_658_, lean_object* v_inst_659_, lean_object* v_inst_660_, lean_object* v_f_661_, lean_object* v_hf_662_, lean_object* v_one_663_, lean_object* v_inv_664_){
_start:
{
lean_object* v_res_665_; 
v_res_665_ = lp_mathlib_Function_Injective_negZeroClass(v_M_u2081_656_, v_M_u2082_657_, v_inst_658_, v_inst_659_, v_inst_660_, v_f_661_, v_hf_662_, v_one_663_, v_inv_664_);
lean_dec(v_f_661_);
lean_dec_ref(v_inst_660_);
return v_res_665_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divInvMonoid___redArg___lam__1(lean_object* v_inst_666_, lean_object* v_n_667_, lean_object* v_x_668_){
_start:
{
lean_object* v___x_669_; 
v___x_669_ = lean_apply_2(v_inst_666_, v_x_668_, v_n_667_);
return v___x_669_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divInvMonoid___redArg(lean_object* v_inst_670_, lean_object* v_inst_671_, lean_object* v_inst_672_, lean_object* v_inst_673_, lean_object* v_inst_674_, lean_object* v_inst_675_){
_start:
{
lean_object* v___f_676_; lean_object* v___f_677_; lean_object* v___x_678_; lean_object* v___x_679_; 
v___f_676_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_676_, 0, v_inst_672_);
v___f_677_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divInvMonoid___redArg___lam__1), 3, 1);
lean_closure_set(v___f_677_, 0, v_inst_675_);
v___x_678_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_678_, 0, v_inst_671_);
lean_ctor_set(v___x_678_, 1, v_inst_670_);
lean_ctor_set(v___x_678_, 2, v___f_676_);
v___x_679_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_679_, 0, v___x_678_);
lean_ctor_set(v___x_679_, 1, v_inst_673_);
lean_ctor_set(v___x_679_, 2, v_inst_674_);
lean_ctor_set(v___x_679_, 3, v___f_677_);
return v___x_679_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divInvMonoid(lean_object* v_M_u2081_680_, lean_object* v_M_u2082_681_, lean_object* v_inst_682_, lean_object* v_inst_683_, lean_object* v_inst_684_, lean_object* v_inst_685_, lean_object* v_inst_686_, lean_object* v_inst_687_, lean_object* v_inst_688_, lean_object* v_f_689_, lean_object* v_hf_690_, lean_object* v_one_691_, lean_object* v_mul_692_, lean_object* v_inv_693_, lean_object* v_div_694_, lean_object* v_npow_695_, lean_object* v_zpow_696_){
_start:
{
lean_object* v___f_697_; lean_object* v___f_698_; lean_object* v___x_699_; lean_object* v___x_700_; 
v___f_697_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_697_, 0, v_inst_684_);
v___f_698_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divInvMonoid___redArg___lam__1), 3, 1);
lean_closure_set(v___f_698_, 0, v_inst_687_);
v___x_699_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_699_, 0, v_inst_683_);
lean_ctor_set(v___x_699_, 1, v_inst_682_);
lean_ctor_set(v___x_699_, 2, v___f_697_);
v___x_700_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_700_, 0, v___x_699_);
lean_ctor_set(v___x_700_, 1, v_inst_685_);
lean_ctor_set(v___x_700_, 2, v_inst_686_);
lean_ctor_set(v___x_700_, 3, v___f_698_);
return v___x_700_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divInvMonoid___boxed(lean_object** _args){
lean_object* v_M_u2081_701_ = _args[0];
lean_object* v_M_u2082_702_ = _args[1];
lean_object* v_inst_703_ = _args[2];
lean_object* v_inst_704_ = _args[3];
lean_object* v_inst_705_ = _args[4];
lean_object* v_inst_706_ = _args[5];
lean_object* v_inst_707_ = _args[6];
lean_object* v_inst_708_ = _args[7];
lean_object* v_inst_709_ = _args[8];
lean_object* v_f_710_ = _args[9];
lean_object* v_hf_711_ = _args[10];
lean_object* v_one_712_ = _args[11];
lean_object* v_mul_713_ = _args[12];
lean_object* v_inv_714_ = _args[13];
lean_object* v_div_715_ = _args[14];
lean_object* v_npow_716_ = _args[15];
lean_object* v_zpow_717_ = _args[16];
_start:
{
lean_object* v_res_718_; 
v_res_718_ = lp_mathlib_Function_Injective_divInvMonoid(v_M_u2081_701_, v_M_u2082_702_, v_inst_703_, v_inst_704_, v_inst_705_, v_inst_706_, v_inst_707_, v_inst_708_, v_inst_709_, v_f_710_, v_hf_711_, v_one_712_, v_mul_713_, v_inv_714_, v_div_715_, v_npow_716_, v_zpow_717_);
lean_dec(v_f_710_);
lean_dec_ref(v_inst_709_);
return v_res_718_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_subNegMonoid___redArg___lam__0(lean_object* v_inst_719_, lean_object* v_n_720_, lean_object* v_x_721_){
_start:
{
lean_object* v___x_722_; 
v___x_722_ = lean_apply_2(v_inst_719_, v_n_720_, v_x_721_);
return v___x_722_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_subNegMonoid___redArg(lean_object* v_inst_723_, lean_object* v_inst_724_, lean_object* v_inst_725_, lean_object* v_inst_726_, lean_object* v_inst_727_, lean_object* v_inst_728_){
_start:
{
lean_object* v___f_729_; lean_object* v___x_730_; lean_object* v___x_731_; 
v___f_729_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_subNegMonoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_729_, 0, v_inst_728_);
v___x_730_ = lp_mathlib_Function_Injective_addMonoid___redArg(v_inst_723_, v_inst_724_, v_inst_725_);
v___x_731_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_731_, 0, v___x_730_);
lean_ctor_set(v___x_731_, 1, v_inst_726_);
lean_ctor_set(v___x_731_, 2, v_inst_727_);
lean_ctor_set(v___x_731_, 3, v___f_729_);
return v___x_731_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_subNegMonoid(lean_object* v_M_u2081_732_, lean_object* v_M_u2082_733_, lean_object* v_inst_734_, lean_object* v_inst_735_, lean_object* v_inst_736_, lean_object* v_inst_737_, lean_object* v_inst_738_, lean_object* v_inst_739_, lean_object* v_inst_740_, lean_object* v_f_741_, lean_object* v_hf_742_, lean_object* v_one_743_, lean_object* v_mul_744_, lean_object* v_inv_745_, lean_object* v_div_746_, lean_object* v_npow_747_, lean_object* v_zpow_748_){
_start:
{
lean_object* v___x_749_; 
v___x_749_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_734_, v_inst_735_, v_inst_736_, v_inst_737_, v_inst_738_, v_inst_739_);
return v___x_749_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_subNegMonoid___boxed(lean_object** _args){
lean_object* v_M_u2081_750_ = _args[0];
lean_object* v_M_u2082_751_ = _args[1];
lean_object* v_inst_752_ = _args[2];
lean_object* v_inst_753_ = _args[3];
lean_object* v_inst_754_ = _args[4];
lean_object* v_inst_755_ = _args[5];
lean_object* v_inst_756_ = _args[6];
lean_object* v_inst_757_ = _args[7];
lean_object* v_inst_758_ = _args[8];
lean_object* v_f_759_ = _args[9];
lean_object* v_hf_760_ = _args[10];
lean_object* v_one_761_ = _args[11];
lean_object* v_mul_762_ = _args[12];
lean_object* v_inv_763_ = _args[13];
lean_object* v_div_764_ = _args[14];
lean_object* v_npow_765_ = _args[15];
lean_object* v_zpow_766_ = _args[16];
_start:
{
lean_object* v_res_767_; 
v_res_767_ = lp_mathlib_Function_Injective_subNegMonoid(v_M_u2081_750_, v_M_u2082_751_, v_inst_752_, v_inst_753_, v_inst_754_, v_inst_755_, v_inst_756_, v_inst_757_, v_inst_758_, v_f_759_, v_hf_760_, v_one_761_, v_mul_762_, v_inv_763_, v_div_764_, v_npow_765_, v_zpow_766_);
lean_dec(v_f_759_);
lean_dec_ref(v_inst_758_);
return v_res_767_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divInvOneMonoid___redArg(lean_object* v_inst_768_, lean_object* v_inst_769_, lean_object* v_inst_770_, lean_object* v_inst_771_, lean_object* v_inst_772_, lean_object* v_inst_773_){
_start:
{
lean_object* v___f_774_; lean_object* v___f_775_; lean_object* v___x_776_; lean_object* v___x_777_; 
v___f_774_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_774_, 0, v_inst_770_);
v___f_775_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divInvMonoid___redArg___lam__1), 3, 1);
lean_closure_set(v___f_775_, 0, v_inst_773_);
v___x_776_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_776_, 0, v_inst_769_);
lean_ctor_set(v___x_776_, 1, v_inst_768_);
lean_ctor_set(v___x_776_, 2, v___f_774_);
v___x_777_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_777_, 0, v___x_776_);
lean_ctor_set(v___x_777_, 1, v_inst_771_);
lean_ctor_set(v___x_777_, 2, v_inst_772_);
lean_ctor_set(v___x_777_, 3, v___f_775_);
return v___x_777_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divInvOneMonoid(lean_object* v_M_u2081_778_, lean_object* v_M_u2082_779_, lean_object* v_inst_780_, lean_object* v_inst_781_, lean_object* v_inst_782_, lean_object* v_inst_783_, lean_object* v_inst_784_, lean_object* v_inst_785_, lean_object* v_inst_786_, lean_object* v_f_787_, lean_object* v_hf_788_, lean_object* v_one_789_, lean_object* v_mul_790_, lean_object* v_inv_791_, lean_object* v_div_792_, lean_object* v_npow_793_, lean_object* v_zpow_794_){
_start:
{
lean_object* v___f_795_; lean_object* v___f_796_; lean_object* v___x_797_; lean_object* v___x_798_; 
v___f_795_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_795_, 0, v_inst_782_);
v___f_796_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divInvMonoid___redArg___lam__1), 3, 1);
lean_closure_set(v___f_796_, 0, v_inst_785_);
v___x_797_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_797_, 0, v_inst_781_);
lean_ctor_set(v___x_797_, 1, v_inst_780_);
lean_ctor_set(v___x_797_, 2, v___f_795_);
v___x_798_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_798_, 0, v___x_797_);
lean_ctor_set(v___x_798_, 1, v_inst_783_);
lean_ctor_set(v___x_798_, 2, v_inst_784_);
lean_ctor_set(v___x_798_, 3, v___f_796_);
return v___x_798_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divInvOneMonoid___boxed(lean_object** _args){
lean_object* v_M_u2081_799_ = _args[0];
lean_object* v_M_u2082_800_ = _args[1];
lean_object* v_inst_801_ = _args[2];
lean_object* v_inst_802_ = _args[3];
lean_object* v_inst_803_ = _args[4];
lean_object* v_inst_804_ = _args[5];
lean_object* v_inst_805_ = _args[6];
lean_object* v_inst_806_ = _args[7];
lean_object* v_inst_807_ = _args[8];
lean_object* v_f_808_ = _args[9];
lean_object* v_hf_809_ = _args[10];
lean_object* v_one_810_ = _args[11];
lean_object* v_mul_811_ = _args[12];
lean_object* v_inv_812_ = _args[13];
lean_object* v_div_813_ = _args[14];
lean_object* v_npow_814_ = _args[15];
lean_object* v_zpow_815_ = _args[16];
_start:
{
lean_object* v_res_816_; 
v_res_816_ = lp_mathlib_Function_Injective_divInvOneMonoid(v_M_u2081_799_, v_M_u2082_800_, v_inst_801_, v_inst_802_, v_inst_803_, v_inst_804_, v_inst_805_, v_inst_806_, v_inst_807_, v_f_808_, v_hf_809_, v_one_810_, v_mul_811_, v_inv_812_, v_div_813_, v_npow_814_, v_zpow_815_);
lean_dec(v_f_808_);
lean_dec_ref(v_inst_807_);
return v_res_816_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_subNegZeroMonoid___redArg(lean_object* v_inst_817_, lean_object* v_inst_818_, lean_object* v_inst_819_, lean_object* v_inst_820_, lean_object* v_inst_821_, lean_object* v_inst_822_){
_start:
{
lean_object* v___x_823_; 
v___x_823_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_817_, v_inst_818_, v_inst_819_, v_inst_820_, v_inst_821_, v_inst_822_);
return v___x_823_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_subNegZeroMonoid(lean_object* v_M_u2081_824_, lean_object* v_M_u2082_825_, lean_object* v_inst_826_, lean_object* v_inst_827_, lean_object* v_inst_828_, lean_object* v_inst_829_, lean_object* v_inst_830_, lean_object* v_inst_831_, lean_object* v_inst_832_, lean_object* v_f_833_, lean_object* v_hf_834_, lean_object* v_one_835_, lean_object* v_mul_836_, lean_object* v_inv_837_, lean_object* v_div_838_, lean_object* v_npow_839_, lean_object* v_zpow_840_){
_start:
{
lean_object* v___x_841_; 
v___x_841_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_826_, v_inst_827_, v_inst_828_, v_inst_829_, v_inst_830_, v_inst_831_);
return v___x_841_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_subNegZeroMonoid___boxed(lean_object** _args){
lean_object* v_M_u2081_842_ = _args[0];
lean_object* v_M_u2082_843_ = _args[1];
lean_object* v_inst_844_ = _args[2];
lean_object* v_inst_845_ = _args[3];
lean_object* v_inst_846_ = _args[4];
lean_object* v_inst_847_ = _args[5];
lean_object* v_inst_848_ = _args[6];
lean_object* v_inst_849_ = _args[7];
lean_object* v_inst_850_ = _args[8];
lean_object* v_f_851_ = _args[9];
lean_object* v_hf_852_ = _args[10];
lean_object* v_one_853_ = _args[11];
lean_object* v_mul_854_ = _args[12];
lean_object* v_inv_855_ = _args[13];
lean_object* v_div_856_ = _args[14];
lean_object* v_npow_857_ = _args[15];
lean_object* v_zpow_858_ = _args[16];
_start:
{
lean_object* v_res_859_; 
v_res_859_ = lp_mathlib_Function_Injective_subNegZeroMonoid(v_M_u2081_842_, v_M_u2082_843_, v_inst_844_, v_inst_845_, v_inst_846_, v_inst_847_, v_inst_848_, v_inst_849_, v_inst_850_, v_f_851_, v_hf_852_, v_one_853_, v_mul_854_, v_inv_855_, v_div_856_, v_npow_857_, v_zpow_858_);
lean_dec(v_f_851_);
lean_dec_ref(v_inst_850_);
return v_res_859_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionMonoid___redArg(lean_object* v_inst_860_, lean_object* v_inst_861_, lean_object* v_inst_862_, lean_object* v_inst_863_, lean_object* v_inst_864_, lean_object* v_inst_865_){
_start:
{
lean_object* v___f_866_; lean_object* v___f_867_; lean_object* v___x_868_; lean_object* v___x_869_; 
v___f_866_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_866_, 0, v_inst_862_);
v___f_867_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divInvMonoid___redArg___lam__1), 3, 1);
lean_closure_set(v___f_867_, 0, v_inst_865_);
v___x_868_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_868_, 0, v_inst_861_);
lean_ctor_set(v___x_868_, 1, v_inst_860_);
lean_ctor_set(v___x_868_, 2, v___f_866_);
v___x_869_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_869_, 0, v___x_868_);
lean_ctor_set(v___x_869_, 1, v_inst_863_);
lean_ctor_set(v___x_869_, 2, v_inst_864_);
lean_ctor_set(v___x_869_, 3, v___f_867_);
return v___x_869_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionMonoid(lean_object* v_M_u2081_870_, lean_object* v_M_u2082_871_, lean_object* v_inst_872_, lean_object* v_inst_873_, lean_object* v_inst_874_, lean_object* v_inst_875_, lean_object* v_inst_876_, lean_object* v_inst_877_, lean_object* v_inst_878_, lean_object* v_f_879_, lean_object* v_hf_880_, lean_object* v_one_881_, lean_object* v_mul_882_, lean_object* v_inv_883_, lean_object* v_div_884_, lean_object* v_npow_885_, lean_object* v_zpow_886_){
_start:
{
lean_object* v___f_887_; lean_object* v___f_888_; lean_object* v___x_889_; lean_object* v___x_890_; 
v___f_887_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_887_, 0, v_inst_874_);
v___f_888_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divInvMonoid___redArg___lam__1), 3, 1);
lean_closure_set(v___f_888_, 0, v_inst_877_);
v___x_889_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_889_, 0, v_inst_873_);
lean_ctor_set(v___x_889_, 1, v_inst_872_);
lean_ctor_set(v___x_889_, 2, v___f_887_);
v___x_890_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_890_, 0, v___x_889_);
lean_ctor_set(v___x_890_, 1, v_inst_875_);
lean_ctor_set(v___x_890_, 2, v_inst_876_);
lean_ctor_set(v___x_890_, 3, v___f_888_);
return v___x_890_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionMonoid___boxed(lean_object** _args){
lean_object* v_M_u2081_891_ = _args[0];
lean_object* v_M_u2082_892_ = _args[1];
lean_object* v_inst_893_ = _args[2];
lean_object* v_inst_894_ = _args[3];
lean_object* v_inst_895_ = _args[4];
lean_object* v_inst_896_ = _args[5];
lean_object* v_inst_897_ = _args[6];
lean_object* v_inst_898_ = _args[7];
lean_object* v_inst_899_ = _args[8];
lean_object* v_f_900_ = _args[9];
lean_object* v_hf_901_ = _args[10];
lean_object* v_one_902_ = _args[11];
lean_object* v_mul_903_ = _args[12];
lean_object* v_inv_904_ = _args[13];
lean_object* v_div_905_ = _args[14];
lean_object* v_npow_906_ = _args[15];
lean_object* v_zpow_907_ = _args[16];
_start:
{
lean_object* v_res_908_; 
v_res_908_ = lp_mathlib_Function_Injective_divisionMonoid(v_M_u2081_891_, v_M_u2082_892_, v_inst_893_, v_inst_894_, v_inst_895_, v_inst_896_, v_inst_897_, v_inst_898_, v_inst_899_, v_f_900_, v_hf_901_, v_one_902_, v_mul_903_, v_inv_904_, v_div_905_, v_npow_906_, v_zpow_907_);
lean_dec(v_f_900_);
lean_dec_ref(v_inst_899_);
return v_res_908_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_subtractionMonoid___redArg(lean_object* v_inst_909_, lean_object* v_inst_910_, lean_object* v_inst_911_, lean_object* v_inst_912_, lean_object* v_inst_913_, lean_object* v_inst_914_){
_start:
{
lean_object* v___x_915_; 
v___x_915_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_909_, v_inst_910_, v_inst_911_, v_inst_912_, v_inst_913_, v_inst_914_);
return v___x_915_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_subtractionMonoid(lean_object* v_M_u2081_916_, lean_object* v_M_u2082_917_, lean_object* v_inst_918_, lean_object* v_inst_919_, lean_object* v_inst_920_, lean_object* v_inst_921_, lean_object* v_inst_922_, lean_object* v_inst_923_, lean_object* v_inst_924_, lean_object* v_f_925_, lean_object* v_hf_926_, lean_object* v_one_927_, lean_object* v_mul_928_, lean_object* v_inv_929_, lean_object* v_div_930_, lean_object* v_npow_931_, lean_object* v_zpow_932_){
_start:
{
lean_object* v___x_933_; 
v___x_933_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_918_, v_inst_919_, v_inst_920_, v_inst_921_, v_inst_922_, v_inst_923_);
return v___x_933_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_subtractionMonoid___boxed(lean_object** _args){
lean_object* v_M_u2081_934_ = _args[0];
lean_object* v_M_u2082_935_ = _args[1];
lean_object* v_inst_936_ = _args[2];
lean_object* v_inst_937_ = _args[3];
lean_object* v_inst_938_ = _args[4];
lean_object* v_inst_939_ = _args[5];
lean_object* v_inst_940_ = _args[6];
lean_object* v_inst_941_ = _args[7];
lean_object* v_inst_942_ = _args[8];
lean_object* v_f_943_ = _args[9];
lean_object* v_hf_944_ = _args[10];
lean_object* v_one_945_ = _args[11];
lean_object* v_mul_946_ = _args[12];
lean_object* v_inv_947_ = _args[13];
lean_object* v_div_948_ = _args[14];
lean_object* v_npow_949_ = _args[15];
lean_object* v_zpow_950_ = _args[16];
_start:
{
lean_object* v_res_951_; 
v_res_951_ = lp_mathlib_Function_Injective_subtractionMonoid(v_M_u2081_934_, v_M_u2082_935_, v_inst_936_, v_inst_937_, v_inst_938_, v_inst_939_, v_inst_940_, v_inst_941_, v_inst_942_, v_f_943_, v_hf_944_, v_one_945_, v_mul_946_, v_inv_947_, v_div_948_, v_npow_949_, v_zpow_950_);
lean_dec(v_f_943_);
lean_dec_ref(v_inst_942_);
return v_res_951_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionCommMonoid___redArg(lean_object* v_inst_952_, lean_object* v_inst_953_, lean_object* v_inst_954_, lean_object* v_inst_955_, lean_object* v_inst_956_, lean_object* v_inst_957_){
_start:
{
lean_object* v___f_958_; lean_object* v___f_959_; lean_object* v___x_960_; lean_object* v___x_961_; 
v___f_958_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_958_, 0, v_inst_954_);
v___f_959_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divInvMonoid___redArg___lam__1), 3, 1);
lean_closure_set(v___f_959_, 0, v_inst_957_);
v___x_960_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_960_, 0, v_inst_953_);
lean_ctor_set(v___x_960_, 1, v_inst_952_);
lean_ctor_set(v___x_960_, 2, v___f_958_);
v___x_961_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_961_, 0, v___x_960_);
lean_ctor_set(v___x_961_, 1, v_inst_955_);
lean_ctor_set(v___x_961_, 2, v_inst_956_);
lean_ctor_set(v___x_961_, 3, v___f_959_);
return v___x_961_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionCommMonoid(lean_object* v_M_u2081_962_, lean_object* v_M_u2082_963_, lean_object* v_inst_964_, lean_object* v_inst_965_, lean_object* v_inst_966_, lean_object* v_inst_967_, lean_object* v_inst_968_, lean_object* v_inst_969_, lean_object* v_inst_970_, lean_object* v_f_971_, lean_object* v_hf_972_, lean_object* v_one_973_, lean_object* v_mul_974_, lean_object* v_inv_975_, lean_object* v_div_976_, lean_object* v_npow_977_, lean_object* v_zpow_978_){
_start:
{
lean_object* v___f_979_; lean_object* v___f_980_; lean_object* v___x_981_; lean_object* v___x_982_; 
v___f_979_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_979_, 0, v_inst_966_);
v___f_980_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divInvMonoid___redArg___lam__1), 3, 1);
lean_closure_set(v___f_980_, 0, v_inst_969_);
v___x_981_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_981_, 0, v_inst_965_);
lean_ctor_set(v___x_981_, 1, v_inst_964_);
lean_ctor_set(v___x_981_, 2, v___f_979_);
v___x_982_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_982_, 0, v___x_981_);
lean_ctor_set(v___x_982_, 1, v_inst_967_);
lean_ctor_set(v___x_982_, 2, v_inst_968_);
lean_ctor_set(v___x_982_, 3, v___f_980_);
return v___x_982_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_divisionCommMonoid___boxed(lean_object** _args){
lean_object* v_M_u2081_983_ = _args[0];
lean_object* v_M_u2082_984_ = _args[1];
lean_object* v_inst_985_ = _args[2];
lean_object* v_inst_986_ = _args[3];
lean_object* v_inst_987_ = _args[4];
lean_object* v_inst_988_ = _args[5];
lean_object* v_inst_989_ = _args[6];
lean_object* v_inst_990_ = _args[7];
lean_object* v_inst_991_ = _args[8];
lean_object* v_f_992_ = _args[9];
lean_object* v_hf_993_ = _args[10];
lean_object* v_one_994_ = _args[11];
lean_object* v_mul_995_ = _args[12];
lean_object* v_inv_996_ = _args[13];
lean_object* v_div_997_ = _args[14];
lean_object* v_npow_998_ = _args[15];
lean_object* v_zpow_999_ = _args[16];
_start:
{
lean_object* v_res_1000_; 
v_res_1000_ = lp_mathlib_Function_Injective_divisionCommMonoid(v_M_u2081_983_, v_M_u2082_984_, v_inst_985_, v_inst_986_, v_inst_987_, v_inst_988_, v_inst_989_, v_inst_990_, v_inst_991_, v_f_992_, v_hf_993_, v_one_994_, v_mul_995_, v_inv_996_, v_div_997_, v_npow_998_, v_zpow_999_);
lean_dec(v_f_992_);
lean_dec_ref(v_inst_991_);
return v_res_1000_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_subtractionCommMonoid___redArg(lean_object* v_inst_1001_, lean_object* v_inst_1002_, lean_object* v_inst_1003_, lean_object* v_inst_1004_, lean_object* v_inst_1005_, lean_object* v_inst_1006_){
_start:
{
lean_object* v___x_1007_; 
v___x_1007_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_1001_, v_inst_1002_, v_inst_1003_, v_inst_1004_, v_inst_1005_, v_inst_1006_);
return v___x_1007_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_subtractionCommMonoid(lean_object* v_M_u2081_1008_, lean_object* v_M_u2082_1009_, lean_object* v_inst_1010_, lean_object* v_inst_1011_, lean_object* v_inst_1012_, lean_object* v_inst_1013_, lean_object* v_inst_1014_, lean_object* v_inst_1015_, lean_object* v_inst_1016_, lean_object* v_f_1017_, lean_object* v_hf_1018_, lean_object* v_one_1019_, lean_object* v_mul_1020_, lean_object* v_inv_1021_, lean_object* v_div_1022_, lean_object* v_npow_1023_, lean_object* v_zpow_1024_){
_start:
{
lean_object* v___x_1025_; 
v___x_1025_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_1010_, v_inst_1011_, v_inst_1012_, v_inst_1013_, v_inst_1014_, v_inst_1015_);
return v___x_1025_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_subtractionCommMonoid___boxed(lean_object** _args){
lean_object* v_M_u2081_1026_ = _args[0];
lean_object* v_M_u2082_1027_ = _args[1];
lean_object* v_inst_1028_ = _args[2];
lean_object* v_inst_1029_ = _args[3];
lean_object* v_inst_1030_ = _args[4];
lean_object* v_inst_1031_ = _args[5];
lean_object* v_inst_1032_ = _args[6];
lean_object* v_inst_1033_ = _args[7];
lean_object* v_inst_1034_ = _args[8];
lean_object* v_f_1035_ = _args[9];
lean_object* v_hf_1036_ = _args[10];
lean_object* v_one_1037_ = _args[11];
lean_object* v_mul_1038_ = _args[12];
lean_object* v_inv_1039_ = _args[13];
lean_object* v_div_1040_ = _args[14];
lean_object* v_npow_1041_ = _args[15];
lean_object* v_zpow_1042_ = _args[16];
_start:
{
lean_object* v_res_1043_; 
v_res_1043_ = lp_mathlib_Function_Injective_subtractionCommMonoid(v_M_u2081_1026_, v_M_u2082_1027_, v_inst_1028_, v_inst_1029_, v_inst_1030_, v_inst_1031_, v_inst_1032_, v_inst_1033_, v_inst_1034_, v_f_1035_, v_hf_1036_, v_one_1037_, v_mul_1038_, v_inv_1039_, v_div_1040_, v_npow_1041_, v_zpow_1042_);
lean_dec(v_f_1035_);
lean_dec_ref(v_inst_1034_);
return v_res_1043_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_group___redArg(lean_object* v_inst_1044_, lean_object* v_inst_1045_, lean_object* v_inst_1046_, lean_object* v_inst_1047_, lean_object* v_inst_1048_, lean_object* v_inst_1049_){
_start:
{
lean_object* v___f_1050_; lean_object* v___f_1051_; lean_object* v___x_1052_; lean_object* v___x_1053_; 
v___f_1050_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1050_, 0, v_inst_1046_);
v___f_1051_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divInvMonoid___redArg___lam__1), 3, 1);
lean_closure_set(v___f_1051_, 0, v_inst_1049_);
v___x_1052_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1052_, 0, v_inst_1045_);
lean_ctor_set(v___x_1052_, 1, v_inst_1044_);
lean_ctor_set(v___x_1052_, 2, v___f_1050_);
v___x_1053_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1053_, 0, v___x_1052_);
lean_ctor_set(v___x_1053_, 1, v_inst_1047_);
lean_ctor_set(v___x_1053_, 2, v_inst_1048_);
lean_ctor_set(v___x_1053_, 3, v___f_1051_);
return v___x_1053_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_group(lean_object* v_M_u2081_1054_, lean_object* v_M_u2082_1055_, lean_object* v_inst_1056_, lean_object* v_inst_1057_, lean_object* v_inst_1058_, lean_object* v_inst_1059_, lean_object* v_inst_1060_, lean_object* v_inst_1061_, lean_object* v_inst_1062_, lean_object* v_f_1063_, lean_object* v_hf_1064_, lean_object* v_one_1065_, lean_object* v_mul_1066_, lean_object* v_inv_1067_, lean_object* v_div_1068_, lean_object* v_npow_1069_, lean_object* v_zpow_1070_){
_start:
{
lean_object* v___f_1071_; lean_object* v___f_1072_; lean_object* v___x_1073_; lean_object* v___x_1074_; 
v___f_1071_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1071_, 0, v_inst_1058_);
v___f_1072_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divInvMonoid___redArg___lam__1), 3, 1);
lean_closure_set(v___f_1072_, 0, v_inst_1061_);
v___x_1073_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1073_, 0, v_inst_1057_);
lean_ctor_set(v___x_1073_, 1, v_inst_1056_);
lean_ctor_set(v___x_1073_, 2, v___f_1071_);
v___x_1074_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1074_, 0, v___x_1073_);
lean_ctor_set(v___x_1074_, 1, v_inst_1059_);
lean_ctor_set(v___x_1074_, 2, v_inst_1060_);
lean_ctor_set(v___x_1074_, 3, v___f_1072_);
return v___x_1074_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_group___boxed(lean_object** _args){
lean_object* v_M_u2081_1075_ = _args[0];
lean_object* v_M_u2082_1076_ = _args[1];
lean_object* v_inst_1077_ = _args[2];
lean_object* v_inst_1078_ = _args[3];
lean_object* v_inst_1079_ = _args[4];
lean_object* v_inst_1080_ = _args[5];
lean_object* v_inst_1081_ = _args[6];
lean_object* v_inst_1082_ = _args[7];
lean_object* v_inst_1083_ = _args[8];
lean_object* v_f_1084_ = _args[9];
lean_object* v_hf_1085_ = _args[10];
lean_object* v_one_1086_ = _args[11];
lean_object* v_mul_1087_ = _args[12];
lean_object* v_inv_1088_ = _args[13];
lean_object* v_div_1089_ = _args[14];
lean_object* v_npow_1090_ = _args[15];
lean_object* v_zpow_1091_ = _args[16];
_start:
{
lean_object* v_res_1092_; 
v_res_1092_ = lp_mathlib_Function_Injective_group(v_M_u2081_1075_, v_M_u2082_1076_, v_inst_1077_, v_inst_1078_, v_inst_1079_, v_inst_1080_, v_inst_1081_, v_inst_1082_, v_inst_1083_, v_f_1084_, v_hf_1085_, v_one_1086_, v_mul_1087_, v_inv_1088_, v_div_1089_, v_npow_1090_, v_zpow_1091_);
lean_dec(v_f_1084_);
lean_dec_ref(v_inst_1083_);
return v_res_1092_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addGroup___redArg(lean_object* v_inst_1093_, lean_object* v_inst_1094_, lean_object* v_inst_1095_, lean_object* v_inst_1096_, lean_object* v_inst_1097_, lean_object* v_inst_1098_){
_start:
{
lean_object* v___x_1099_; 
v___x_1099_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_1093_, v_inst_1094_, v_inst_1095_, v_inst_1096_, v_inst_1097_, v_inst_1098_);
return v___x_1099_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addGroup(lean_object* v_M_u2081_1100_, lean_object* v_M_u2082_1101_, lean_object* v_inst_1102_, lean_object* v_inst_1103_, lean_object* v_inst_1104_, lean_object* v_inst_1105_, lean_object* v_inst_1106_, lean_object* v_inst_1107_, lean_object* v_inst_1108_, lean_object* v_f_1109_, lean_object* v_hf_1110_, lean_object* v_one_1111_, lean_object* v_mul_1112_, lean_object* v_inv_1113_, lean_object* v_div_1114_, lean_object* v_npow_1115_, lean_object* v_zpow_1116_){
_start:
{
lean_object* v___x_1117_; 
v___x_1117_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_1102_, v_inst_1103_, v_inst_1104_, v_inst_1105_, v_inst_1106_, v_inst_1107_);
return v___x_1117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addGroup___boxed(lean_object** _args){
lean_object* v_M_u2081_1118_ = _args[0];
lean_object* v_M_u2082_1119_ = _args[1];
lean_object* v_inst_1120_ = _args[2];
lean_object* v_inst_1121_ = _args[3];
lean_object* v_inst_1122_ = _args[4];
lean_object* v_inst_1123_ = _args[5];
lean_object* v_inst_1124_ = _args[6];
lean_object* v_inst_1125_ = _args[7];
lean_object* v_inst_1126_ = _args[8];
lean_object* v_f_1127_ = _args[9];
lean_object* v_hf_1128_ = _args[10];
lean_object* v_one_1129_ = _args[11];
lean_object* v_mul_1130_ = _args[12];
lean_object* v_inv_1131_ = _args[13];
lean_object* v_div_1132_ = _args[14];
lean_object* v_npow_1133_ = _args[15];
lean_object* v_zpow_1134_ = _args[16];
_start:
{
lean_object* v_res_1135_; 
v_res_1135_ = lp_mathlib_Function_Injective_addGroup(v_M_u2081_1118_, v_M_u2082_1119_, v_inst_1120_, v_inst_1121_, v_inst_1122_, v_inst_1123_, v_inst_1124_, v_inst_1125_, v_inst_1126_, v_f_1127_, v_hf_1128_, v_one_1129_, v_mul_1130_, v_inv_1131_, v_div_1132_, v_npow_1133_, v_zpow_1134_);
lean_dec(v_f_1127_);
lean_dec_ref(v_inst_1126_);
return v_res_1135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commGroup___redArg(lean_object* v_inst_1136_, lean_object* v_inst_1137_, lean_object* v_inst_1138_, lean_object* v_inst_1139_, lean_object* v_inst_1140_, lean_object* v_inst_1141_){
_start:
{
lean_object* v___f_1142_; lean_object* v___f_1143_; lean_object* v___x_1144_; lean_object* v___x_1145_; 
v___f_1142_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1142_, 0, v_inst_1138_);
v___f_1143_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divInvMonoid___redArg___lam__1), 3, 1);
lean_closure_set(v___f_1143_, 0, v_inst_1141_);
v___x_1144_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1144_, 0, v_inst_1137_);
lean_ctor_set(v___x_1144_, 1, v_inst_1136_);
lean_ctor_set(v___x_1144_, 2, v___f_1142_);
v___x_1145_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1145_, 0, v___x_1144_);
lean_ctor_set(v___x_1145_, 1, v_inst_1139_);
lean_ctor_set(v___x_1145_, 2, v_inst_1140_);
lean_ctor_set(v___x_1145_, 3, v___f_1143_);
return v___x_1145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commGroup(lean_object* v_M_u2081_1146_, lean_object* v_M_u2082_1147_, lean_object* v_inst_1148_, lean_object* v_inst_1149_, lean_object* v_inst_1150_, lean_object* v_inst_1151_, lean_object* v_inst_1152_, lean_object* v_inst_1153_, lean_object* v_inst_1154_, lean_object* v_f_1155_, lean_object* v_hf_1156_, lean_object* v_one_1157_, lean_object* v_mul_1158_, lean_object* v_inv_1159_, lean_object* v_div_1160_, lean_object* v_npow_1161_, lean_object* v_zpow_1162_){
_start:
{
lean_object* v___f_1163_; lean_object* v___f_1164_; lean_object* v___x_1165_; lean_object* v___x_1166_; 
v___f_1163_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1163_, 0, v_inst_1150_);
v___f_1164_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divInvMonoid___redArg___lam__1), 3, 1);
lean_closure_set(v___f_1164_, 0, v_inst_1153_);
v___x_1165_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1165_, 0, v_inst_1149_);
lean_ctor_set(v___x_1165_, 1, v_inst_1148_);
lean_ctor_set(v___x_1165_, 2, v___f_1163_);
v___x_1166_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1166_, 0, v___x_1165_);
lean_ctor_set(v___x_1166_, 1, v_inst_1151_);
lean_ctor_set(v___x_1166_, 2, v_inst_1152_);
lean_ctor_set(v___x_1166_, 3, v___f_1164_);
return v___x_1166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_commGroup___boxed(lean_object** _args){
lean_object* v_M_u2081_1167_ = _args[0];
lean_object* v_M_u2082_1168_ = _args[1];
lean_object* v_inst_1169_ = _args[2];
lean_object* v_inst_1170_ = _args[3];
lean_object* v_inst_1171_ = _args[4];
lean_object* v_inst_1172_ = _args[5];
lean_object* v_inst_1173_ = _args[6];
lean_object* v_inst_1174_ = _args[7];
lean_object* v_inst_1175_ = _args[8];
lean_object* v_f_1176_ = _args[9];
lean_object* v_hf_1177_ = _args[10];
lean_object* v_one_1178_ = _args[11];
lean_object* v_mul_1179_ = _args[12];
lean_object* v_inv_1180_ = _args[13];
lean_object* v_div_1181_ = _args[14];
lean_object* v_npow_1182_ = _args[15];
lean_object* v_zpow_1183_ = _args[16];
_start:
{
lean_object* v_res_1184_; 
v_res_1184_ = lp_mathlib_Function_Injective_commGroup(v_M_u2081_1167_, v_M_u2082_1168_, v_inst_1169_, v_inst_1170_, v_inst_1171_, v_inst_1172_, v_inst_1173_, v_inst_1174_, v_inst_1175_, v_f_1176_, v_hf_1177_, v_one_1178_, v_mul_1179_, v_inv_1180_, v_div_1181_, v_npow_1182_, v_zpow_1183_);
lean_dec(v_f_1176_);
lean_dec_ref(v_inst_1175_);
return v_res_1184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommGroup___redArg(lean_object* v_inst_1185_, lean_object* v_inst_1186_, lean_object* v_inst_1187_, lean_object* v_inst_1188_, lean_object* v_inst_1189_, lean_object* v_inst_1190_){
_start:
{
lean_object* v___x_1191_; 
v___x_1191_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_1185_, v_inst_1186_, v_inst_1187_, v_inst_1188_, v_inst_1189_, v_inst_1190_);
return v___x_1191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommGroup(lean_object* v_M_u2081_1192_, lean_object* v_M_u2082_1193_, lean_object* v_inst_1194_, lean_object* v_inst_1195_, lean_object* v_inst_1196_, lean_object* v_inst_1197_, lean_object* v_inst_1198_, lean_object* v_inst_1199_, lean_object* v_inst_1200_, lean_object* v_f_1201_, lean_object* v_hf_1202_, lean_object* v_one_1203_, lean_object* v_mul_1204_, lean_object* v_inv_1205_, lean_object* v_div_1206_, lean_object* v_npow_1207_, lean_object* v_zpow_1208_){
_start:
{
lean_object* v___x_1209_; 
v___x_1209_ = lp_mathlib_Function_Injective_subNegMonoid___redArg(v_inst_1194_, v_inst_1195_, v_inst_1196_, v_inst_1197_, v_inst_1198_, v_inst_1199_);
return v___x_1209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_addCommGroup___boxed(lean_object** _args){
lean_object* v_M_u2081_1210_ = _args[0];
lean_object* v_M_u2082_1211_ = _args[1];
lean_object* v_inst_1212_ = _args[2];
lean_object* v_inst_1213_ = _args[3];
lean_object* v_inst_1214_ = _args[4];
lean_object* v_inst_1215_ = _args[5];
lean_object* v_inst_1216_ = _args[6];
lean_object* v_inst_1217_ = _args[7];
lean_object* v_inst_1218_ = _args[8];
lean_object* v_f_1219_ = _args[9];
lean_object* v_hf_1220_ = _args[10];
lean_object* v_one_1221_ = _args[11];
lean_object* v_mul_1222_ = _args[12];
lean_object* v_inv_1223_ = _args[13];
lean_object* v_div_1224_ = _args[14];
lean_object* v_npow_1225_ = _args[15];
lean_object* v_zpow_1226_ = _args[16];
_start:
{
lean_object* v_res_1227_; 
v_res_1227_ = lp_mathlib_Function_Injective_addCommGroup(v_M_u2081_1210_, v_M_u2082_1211_, v_inst_1212_, v_inst_1213_, v_inst_1214_, v_inst_1215_, v_inst_1216_, v_inst_1217_, v_inst_1218_, v_f_1219_, v_hf_1220_, v_one_1221_, v_mul_1222_, v_inv_1223_, v_div_1224_, v_npow_1225_, v_zpow_1226_);
lean_dec(v_f_1219_);
lean_dec_ref(v_inst_1218_);
return v_res_1227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_semigroup___redArg(lean_object* v_inst_1228_){
_start:
{
lean_inc(v_inst_1228_);
return v_inst_1228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_semigroup___redArg___boxed(lean_object* v_inst_1229_){
_start:
{
lean_object* v_res_1230_; 
v_res_1230_ = lp_mathlib_Function_Surjective_semigroup___redArg(v_inst_1229_);
lean_dec(v_inst_1229_);
return v_res_1230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_semigroup(lean_object* v_M_u2081_1231_, lean_object* v_M_u2082_1232_, lean_object* v_inst_1233_, lean_object* v_inst_1234_, lean_object* v_f_1235_, lean_object* v_hf_1236_, lean_object* v_mul_1237_){
_start:
{
lean_inc(v_inst_1233_);
return v_inst_1233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_semigroup___boxed(lean_object* v_M_u2081_1238_, lean_object* v_M_u2082_1239_, lean_object* v_inst_1240_, lean_object* v_inst_1241_, lean_object* v_f_1242_, lean_object* v_hf_1243_, lean_object* v_mul_1244_){
_start:
{
lean_object* v_res_1245_; 
v_res_1245_ = lp_mathlib_Function_Surjective_semigroup(v_M_u2081_1238_, v_M_u2082_1239_, v_inst_1240_, v_inst_1241_, v_f_1242_, v_hf_1243_, v_mul_1244_);
lean_dec(v_f_1242_);
lean_dec(v_inst_1241_);
lean_dec(v_inst_1240_);
return v_res_1245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addSemigroup___redArg(lean_object* v_inst_1246_){
_start:
{
lean_inc(v_inst_1246_);
return v_inst_1246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addSemigroup___redArg___boxed(lean_object* v_inst_1247_){
_start:
{
lean_object* v_res_1248_; 
v_res_1248_ = lp_mathlib_Function_Surjective_addSemigroup___redArg(v_inst_1247_);
lean_dec(v_inst_1247_);
return v_res_1248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addSemigroup(lean_object* v_M_u2081_1249_, lean_object* v_M_u2082_1250_, lean_object* v_inst_1251_, lean_object* v_inst_1252_, lean_object* v_f_1253_, lean_object* v_hf_1254_, lean_object* v_mul_1255_){
_start:
{
lean_inc(v_inst_1251_);
return v_inst_1251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addSemigroup___boxed(lean_object* v_M_u2081_1256_, lean_object* v_M_u2082_1257_, lean_object* v_inst_1258_, lean_object* v_inst_1259_, lean_object* v_f_1260_, lean_object* v_hf_1261_, lean_object* v_mul_1262_){
_start:
{
lean_object* v_res_1263_; 
v_res_1263_ = lp_mathlib_Function_Surjective_addSemigroup(v_M_u2081_1256_, v_M_u2082_1257_, v_inst_1258_, v_inst_1259_, v_f_1260_, v_hf_1261_, v_mul_1262_);
lean_dec(v_f_1260_);
lean_dec(v_inst_1259_);
lean_dec(v_inst_1258_);
return v_res_1263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commMagma___redArg(lean_object* v_inst_1264_){
_start:
{
lean_inc(v_inst_1264_);
return v_inst_1264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commMagma___redArg___boxed(lean_object* v_inst_1265_){
_start:
{
lean_object* v_res_1266_; 
v_res_1266_ = lp_mathlib_Function_Surjective_commMagma___redArg(v_inst_1265_);
lean_dec(v_inst_1265_);
return v_res_1266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commMagma(lean_object* v_M_u2081_1267_, lean_object* v_M_u2082_1268_, lean_object* v_inst_1269_, lean_object* v_inst_1270_, lean_object* v_f_1271_, lean_object* v_hf_1272_, lean_object* v_mul_1273_){
_start:
{
lean_inc(v_inst_1269_);
return v_inst_1269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commMagma___boxed(lean_object* v_M_u2081_1274_, lean_object* v_M_u2082_1275_, lean_object* v_inst_1276_, lean_object* v_inst_1277_, lean_object* v_f_1278_, lean_object* v_hf_1279_, lean_object* v_mul_1280_){
_start:
{
lean_object* v_res_1281_; 
v_res_1281_ = lp_mathlib_Function_Surjective_commMagma(v_M_u2081_1274_, v_M_u2082_1275_, v_inst_1276_, v_inst_1277_, v_f_1278_, v_hf_1279_, v_mul_1280_);
lean_dec(v_f_1278_);
lean_dec(v_inst_1277_);
lean_dec(v_inst_1276_);
return v_res_1281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommMagma___redArg(lean_object* v_inst_1282_){
_start:
{
lean_inc(v_inst_1282_);
return v_inst_1282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommMagma___redArg___boxed(lean_object* v_inst_1283_){
_start:
{
lean_object* v_res_1284_; 
v_res_1284_ = lp_mathlib_Function_Surjective_addCommMagma___redArg(v_inst_1283_);
lean_dec(v_inst_1283_);
return v_res_1284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommMagma(lean_object* v_M_u2081_1285_, lean_object* v_M_u2082_1286_, lean_object* v_inst_1287_, lean_object* v_inst_1288_, lean_object* v_f_1289_, lean_object* v_hf_1290_, lean_object* v_mul_1291_){
_start:
{
lean_inc(v_inst_1287_);
return v_inst_1287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommMagma___boxed(lean_object* v_M_u2081_1292_, lean_object* v_M_u2082_1293_, lean_object* v_inst_1294_, lean_object* v_inst_1295_, lean_object* v_f_1296_, lean_object* v_hf_1297_, lean_object* v_mul_1298_){
_start:
{
lean_object* v_res_1299_; 
v_res_1299_ = lp_mathlib_Function_Surjective_addCommMagma(v_M_u2081_1292_, v_M_u2082_1293_, v_inst_1294_, v_inst_1295_, v_f_1296_, v_hf_1297_, v_mul_1298_);
lean_dec(v_f_1296_);
lean_dec(v_inst_1295_);
lean_dec(v_inst_1294_);
return v_res_1299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commSemigroup___redArg(lean_object* v_inst_1300_){
_start:
{
lean_inc(v_inst_1300_);
return v_inst_1300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commSemigroup___redArg___boxed(lean_object* v_inst_1301_){
_start:
{
lean_object* v_res_1302_; 
v_res_1302_ = lp_mathlib_Function_Surjective_commSemigroup___redArg(v_inst_1301_);
lean_dec(v_inst_1301_);
return v_res_1302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commSemigroup(lean_object* v_M_u2081_1303_, lean_object* v_M_u2082_1304_, lean_object* v_inst_1305_, lean_object* v_inst_1306_, lean_object* v_f_1307_, lean_object* v_hf_1308_, lean_object* v_mul_1309_){
_start:
{
lean_inc(v_inst_1305_);
return v_inst_1305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commSemigroup___boxed(lean_object* v_M_u2081_1310_, lean_object* v_M_u2082_1311_, lean_object* v_inst_1312_, lean_object* v_inst_1313_, lean_object* v_f_1314_, lean_object* v_hf_1315_, lean_object* v_mul_1316_){
_start:
{
lean_object* v_res_1317_; 
v_res_1317_ = lp_mathlib_Function_Surjective_commSemigroup(v_M_u2081_1310_, v_M_u2082_1311_, v_inst_1312_, v_inst_1313_, v_f_1314_, v_hf_1315_, v_mul_1316_);
lean_dec(v_f_1314_);
lean_dec(v_inst_1313_);
lean_dec(v_inst_1312_);
return v_res_1317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommSemigroup___redArg(lean_object* v_inst_1318_){
_start:
{
lean_inc(v_inst_1318_);
return v_inst_1318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommSemigroup___redArg___boxed(lean_object* v_inst_1319_){
_start:
{
lean_object* v_res_1320_; 
v_res_1320_ = lp_mathlib_Function_Surjective_addCommSemigroup___redArg(v_inst_1319_);
lean_dec(v_inst_1319_);
return v_res_1320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommSemigroup(lean_object* v_M_u2081_1321_, lean_object* v_M_u2082_1322_, lean_object* v_inst_1323_, lean_object* v_inst_1324_, lean_object* v_f_1325_, lean_object* v_hf_1326_, lean_object* v_mul_1327_){
_start:
{
lean_inc(v_inst_1323_);
return v_inst_1323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommSemigroup___boxed(lean_object* v_M_u2081_1328_, lean_object* v_M_u2082_1329_, lean_object* v_inst_1330_, lean_object* v_inst_1331_, lean_object* v_f_1332_, lean_object* v_hf_1333_, lean_object* v_mul_1334_){
_start:
{
lean_object* v_res_1335_; 
v_res_1335_ = lp_mathlib_Function_Surjective_addCommSemigroup(v_M_u2081_1328_, v_M_u2082_1329_, v_inst_1330_, v_inst_1331_, v_f_1332_, v_hf_1333_, v_mul_1334_);
lean_dec(v_f_1332_);
lean_dec(v_inst_1331_);
lean_dec(v_inst_1330_);
return v_res_1335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulOneClass___redArg(lean_object* v_inst_1336_, lean_object* v_inst_1337_){
_start:
{
lean_object* v___x_1338_; 
v___x_1338_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1338_, 0, v_inst_1337_);
lean_ctor_set(v___x_1338_, 1, v_inst_1336_);
return v___x_1338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulOneClass(lean_object* v_M_u2081_1339_, lean_object* v_M_u2082_1340_, lean_object* v_inst_1341_, lean_object* v_inst_1342_, lean_object* v_inst_1343_, lean_object* v_f_1344_, lean_object* v_hf_1345_, lean_object* v_one_1346_, lean_object* v_mul_1347_){
_start:
{
lean_object* v___x_1348_; 
v___x_1348_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1348_, 0, v_inst_1342_);
lean_ctor_set(v___x_1348_, 1, v_inst_1341_);
return v___x_1348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_mulOneClass___boxed(lean_object* v_M_u2081_1349_, lean_object* v_M_u2082_1350_, lean_object* v_inst_1351_, lean_object* v_inst_1352_, lean_object* v_inst_1353_, lean_object* v_f_1354_, lean_object* v_hf_1355_, lean_object* v_one_1356_, lean_object* v_mul_1357_){
_start:
{
lean_object* v_res_1358_; 
v_res_1358_ = lp_mathlib_Function_Surjective_mulOneClass(v_M_u2081_1349_, v_M_u2082_1350_, v_inst_1351_, v_inst_1352_, v_inst_1353_, v_f_1354_, v_hf_1355_, v_one_1356_, v_mul_1357_);
lean_dec(v_f_1354_);
lean_dec_ref(v_inst_1353_);
return v_res_1358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addZeroClass___redArg(lean_object* v_inst_1359_, lean_object* v_inst_1360_){
_start:
{
lean_object* v___x_1361_; 
v___x_1361_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1361_, 0, v_inst_1360_);
lean_ctor_set(v___x_1361_, 1, v_inst_1359_);
return v___x_1361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addZeroClass(lean_object* v_M_u2081_1362_, lean_object* v_M_u2082_1363_, lean_object* v_inst_1364_, lean_object* v_inst_1365_, lean_object* v_inst_1366_, lean_object* v_f_1367_, lean_object* v_hf_1368_, lean_object* v_one_1369_, lean_object* v_mul_1370_){
_start:
{
lean_object* v___x_1371_; 
v___x_1371_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1371_, 0, v_inst_1365_);
lean_ctor_set(v___x_1371_, 1, v_inst_1364_);
return v___x_1371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addZeroClass___boxed(lean_object* v_M_u2081_1372_, lean_object* v_M_u2082_1373_, lean_object* v_inst_1374_, lean_object* v_inst_1375_, lean_object* v_inst_1376_, lean_object* v_f_1377_, lean_object* v_hf_1378_, lean_object* v_one_1379_, lean_object* v_mul_1380_){
_start:
{
lean_object* v_res_1381_; 
v_res_1381_ = lp_mathlib_Function_Surjective_addZeroClass(v_M_u2081_1372_, v_M_u2082_1373_, v_inst_1374_, v_inst_1375_, v_inst_1376_, v_f_1377_, v_hf_1378_, v_one_1379_, v_mul_1380_);
lean_dec(v_f_1377_);
lean_dec_ref(v_inst_1376_);
return v_res_1381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_monoid___redArg(lean_object* v_inst_1382_, lean_object* v_inst_1383_, lean_object* v_inst_1384_){
_start:
{
lean_object* v___f_1385_; lean_object* v___x_1386_; 
v___f_1385_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1385_, 0, v_inst_1384_);
v___x_1386_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1386_, 0, v_inst_1383_);
lean_ctor_set(v___x_1386_, 1, v_inst_1382_);
lean_ctor_set(v___x_1386_, 2, v___f_1385_);
return v___x_1386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_monoid(lean_object* v_M_u2081_1387_, lean_object* v_M_u2082_1388_, lean_object* v_inst_1389_, lean_object* v_inst_1390_, lean_object* v_inst_1391_, lean_object* v_inst_1392_, lean_object* v_f_1393_, lean_object* v_hf_1394_, lean_object* v_one_1395_, lean_object* v_mul_1396_, lean_object* v_npow_1397_){
_start:
{
lean_object* v___f_1398_; lean_object* v___x_1399_; 
v___f_1398_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1398_, 0, v_inst_1391_);
v___x_1399_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1399_, 0, v_inst_1390_);
lean_ctor_set(v___x_1399_, 1, v_inst_1389_);
lean_ctor_set(v___x_1399_, 2, v___f_1398_);
return v___x_1399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_monoid___boxed(lean_object* v_M_u2081_1400_, lean_object* v_M_u2082_1401_, lean_object* v_inst_1402_, lean_object* v_inst_1403_, lean_object* v_inst_1404_, lean_object* v_inst_1405_, lean_object* v_f_1406_, lean_object* v_hf_1407_, lean_object* v_one_1408_, lean_object* v_mul_1409_, lean_object* v_npow_1410_){
_start:
{
lean_object* v_res_1411_; 
v_res_1411_ = lp_mathlib_Function_Surjective_monoid(v_M_u2081_1400_, v_M_u2082_1401_, v_inst_1402_, v_inst_1403_, v_inst_1404_, v_inst_1405_, v_f_1406_, v_hf_1407_, v_one_1408_, v_mul_1409_, v_npow_1410_);
lean_dec(v_f_1406_);
lean_dec_ref(v_inst_1405_);
return v_res_1411_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addMonoid___redArg(lean_object* v_inst_1412_, lean_object* v_inst_1413_, lean_object* v_inst_1414_){
_start:
{
lean_object* v___f_1415_; lean_object* v___x_1416_; 
v___f_1415_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_addMonoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1415_, 0, v_inst_1414_);
v___x_1416_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1416_, 0, v_inst_1413_);
lean_ctor_set(v___x_1416_, 1, v_inst_1412_);
lean_ctor_set(v___x_1416_, 2, v___f_1415_);
return v___x_1416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addMonoid(lean_object* v_M_u2081_1417_, lean_object* v_M_u2082_1418_, lean_object* v_inst_1419_, lean_object* v_inst_1420_, lean_object* v_inst_1421_, lean_object* v_inst_1422_, lean_object* v_f_1423_, lean_object* v_hf_1424_, lean_object* v_one_1425_, lean_object* v_mul_1426_, lean_object* v_npow_1427_){
_start:
{
lean_object* v___x_1428_; 
v___x_1428_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_1419_, v_inst_1420_, v_inst_1421_);
return v___x_1428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addMonoid___boxed(lean_object* v_M_u2081_1429_, lean_object* v_M_u2082_1430_, lean_object* v_inst_1431_, lean_object* v_inst_1432_, lean_object* v_inst_1433_, lean_object* v_inst_1434_, lean_object* v_f_1435_, lean_object* v_hf_1436_, lean_object* v_one_1437_, lean_object* v_mul_1438_, lean_object* v_npow_1439_){
_start:
{
lean_object* v_res_1440_; 
v_res_1440_ = lp_mathlib_Function_Surjective_addMonoid(v_M_u2081_1429_, v_M_u2082_1430_, v_inst_1431_, v_inst_1432_, v_inst_1433_, v_inst_1434_, v_f_1435_, v_hf_1436_, v_one_1437_, v_mul_1438_, v_npow_1439_);
lean_dec(v_f_1435_);
lean_dec_ref(v_inst_1434_);
return v_res_1440_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commMonoid___redArg(lean_object* v_inst_1441_, lean_object* v_inst_1442_, lean_object* v_inst_1443_){
_start:
{
lean_object* v___f_1444_; lean_object* v___x_1445_; 
v___f_1444_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1444_, 0, v_inst_1443_);
v___x_1445_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1445_, 0, v_inst_1442_);
lean_ctor_set(v___x_1445_, 1, v_inst_1441_);
lean_ctor_set(v___x_1445_, 2, v___f_1444_);
return v___x_1445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commMonoid(lean_object* v_M_u2081_1446_, lean_object* v_M_u2082_1447_, lean_object* v_inst_1448_, lean_object* v_inst_1449_, lean_object* v_inst_1450_, lean_object* v_inst_1451_, lean_object* v_f_1452_, lean_object* v_hf_1453_, lean_object* v_one_1454_, lean_object* v_mul_1455_, lean_object* v_npow_1456_){
_start:
{
lean_object* v___f_1457_; lean_object* v___x_1458_; 
v___f_1457_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1457_, 0, v_inst_1450_);
v___x_1458_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1458_, 0, v_inst_1449_);
lean_ctor_set(v___x_1458_, 1, v_inst_1448_);
lean_ctor_set(v___x_1458_, 2, v___f_1457_);
return v___x_1458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commMonoid___boxed(lean_object* v_M_u2081_1459_, lean_object* v_M_u2082_1460_, lean_object* v_inst_1461_, lean_object* v_inst_1462_, lean_object* v_inst_1463_, lean_object* v_inst_1464_, lean_object* v_f_1465_, lean_object* v_hf_1466_, lean_object* v_one_1467_, lean_object* v_mul_1468_, lean_object* v_npow_1469_){
_start:
{
lean_object* v_res_1470_; 
v_res_1470_ = lp_mathlib_Function_Surjective_commMonoid(v_M_u2081_1459_, v_M_u2082_1460_, v_inst_1461_, v_inst_1462_, v_inst_1463_, v_inst_1464_, v_f_1465_, v_hf_1466_, v_one_1467_, v_mul_1468_, v_npow_1469_);
lean_dec(v_f_1465_);
lean_dec_ref(v_inst_1464_);
return v_res_1470_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommMonoid___redArg(lean_object* v_inst_1471_, lean_object* v_inst_1472_, lean_object* v_inst_1473_){
_start:
{
lean_object* v___x_1474_; 
v___x_1474_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_1471_, v_inst_1472_, v_inst_1473_);
return v___x_1474_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommMonoid(lean_object* v_M_u2081_1475_, lean_object* v_M_u2082_1476_, lean_object* v_inst_1477_, lean_object* v_inst_1478_, lean_object* v_inst_1479_, lean_object* v_inst_1480_, lean_object* v_f_1481_, lean_object* v_hf_1482_, lean_object* v_one_1483_, lean_object* v_mul_1484_, lean_object* v_npow_1485_){
_start:
{
lean_object* v___x_1486_; 
v___x_1486_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_1477_, v_inst_1478_, v_inst_1479_);
return v___x_1486_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommMonoid___boxed(lean_object* v_M_u2081_1487_, lean_object* v_M_u2082_1488_, lean_object* v_inst_1489_, lean_object* v_inst_1490_, lean_object* v_inst_1491_, lean_object* v_inst_1492_, lean_object* v_f_1493_, lean_object* v_hf_1494_, lean_object* v_one_1495_, lean_object* v_mul_1496_, lean_object* v_npow_1497_){
_start:
{
lean_object* v_res_1498_; 
v_res_1498_ = lp_mathlib_Function_Surjective_addCommMonoid(v_M_u2081_1487_, v_M_u2082_1488_, v_inst_1489_, v_inst_1490_, v_inst_1491_, v_inst_1492_, v_f_1493_, v_hf_1494_, v_one_1495_, v_mul_1496_, v_npow_1497_);
lean_dec(v_f_1493_);
lean_dec_ref(v_inst_1492_);
return v_res_1498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_involutiveInv___redArg(lean_object* v_inst_1499_){
_start:
{
lean_inc(v_inst_1499_);
return v_inst_1499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_involutiveInv___redArg___boxed(lean_object* v_inst_1500_){
_start:
{
lean_object* v_res_1501_; 
v_res_1501_ = lp_mathlib_Function_Surjective_involutiveInv___redArg(v_inst_1500_);
lean_dec(v_inst_1500_);
return v_res_1501_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_involutiveInv(lean_object* v_M_u2081_1502_, lean_object* v_M_u2082_1503_, lean_object* v_inst_1504_, lean_object* v_inst_1505_, lean_object* v_f_1506_, lean_object* v_hf_1507_, lean_object* v_inv_1508_){
_start:
{
lean_inc(v_inst_1504_);
return v_inst_1504_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_involutiveInv___boxed(lean_object* v_M_u2081_1509_, lean_object* v_M_u2082_1510_, lean_object* v_inst_1511_, lean_object* v_inst_1512_, lean_object* v_f_1513_, lean_object* v_hf_1514_, lean_object* v_inv_1515_){
_start:
{
lean_object* v_res_1516_; 
v_res_1516_ = lp_mathlib_Function_Surjective_involutiveInv(v_M_u2081_1509_, v_M_u2082_1510_, v_inst_1511_, v_inst_1512_, v_f_1513_, v_hf_1514_, v_inv_1515_);
lean_dec(v_f_1513_);
lean_dec(v_inst_1512_);
lean_dec(v_inst_1511_);
return v_res_1516_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_involutiveNeg___redArg(lean_object* v_inst_1517_){
_start:
{
lean_inc(v_inst_1517_);
return v_inst_1517_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_involutiveNeg___redArg___boxed(lean_object* v_inst_1518_){
_start:
{
lean_object* v_res_1519_; 
v_res_1519_ = lp_mathlib_Function_Surjective_involutiveNeg___redArg(v_inst_1518_);
lean_dec(v_inst_1518_);
return v_res_1519_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_involutiveNeg(lean_object* v_M_u2081_1520_, lean_object* v_M_u2082_1521_, lean_object* v_inst_1522_, lean_object* v_inst_1523_, lean_object* v_f_1524_, lean_object* v_hf_1525_, lean_object* v_inv_1526_){
_start:
{
lean_inc(v_inst_1522_);
return v_inst_1522_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_involutiveNeg___boxed(lean_object* v_M_u2081_1527_, lean_object* v_M_u2082_1528_, lean_object* v_inst_1529_, lean_object* v_inst_1530_, lean_object* v_f_1531_, lean_object* v_hf_1532_, lean_object* v_inv_1533_){
_start:
{
lean_object* v_res_1534_; 
v_res_1534_ = lp_mathlib_Function_Surjective_involutiveNeg(v_M_u2081_1527_, v_M_u2082_1528_, v_inst_1529_, v_inst_1530_, v_f_1531_, v_hf_1532_, v_inv_1533_);
lean_dec(v_f_1531_);
lean_dec(v_inst_1530_);
lean_dec(v_inst_1529_);
return v_res_1534_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_divInvMonoid___redArg(lean_object* v_inst_1535_, lean_object* v_inst_1536_, lean_object* v_inst_1537_, lean_object* v_inst_1538_, lean_object* v_inst_1539_, lean_object* v_inst_1540_){
_start:
{
lean_object* v___f_1541_; lean_object* v___f_1542_; lean_object* v___x_1543_; lean_object* v___x_1544_; 
v___f_1541_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1541_, 0, v_inst_1537_);
v___f_1542_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divInvMonoid___redArg___lam__1), 3, 1);
lean_closure_set(v___f_1542_, 0, v_inst_1540_);
v___x_1543_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1543_, 0, v_inst_1536_);
lean_ctor_set(v___x_1543_, 1, v_inst_1535_);
lean_ctor_set(v___x_1543_, 2, v___f_1541_);
v___x_1544_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1544_, 0, v___x_1543_);
lean_ctor_set(v___x_1544_, 1, v_inst_1538_);
lean_ctor_set(v___x_1544_, 2, v_inst_1539_);
lean_ctor_set(v___x_1544_, 3, v___f_1542_);
return v___x_1544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_divInvMonoid(lean_object* v_M_u2081_1545_, lean_object* v_M_u2082_1546_, lean_object* v_inst_1547_, lean_object* v_inst_1548_, lean_object* v_inst_1549_, lean_object* v_inst_1550_, lean_object* v_inst_1551_, lean_object* v_inst_1552_, lean_object* v_inst_1553_, lean_object* v_f_1554_, lean_object* v_hf_1555_, lean_object* v_one_1556_, lean_object* v_mul_1557_, lean_object* v_inv_1558_, lean_object* v_div_1559_, lean_object* v_npow_1560_, lean_object* v_zpow_1561_){
_start:
{
lean_object* v___f_1562_; lean_object* v___f_1563_; lean_object* v___x_1564_; lean_object* v___x_1565_; 
v___f_1562_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1562_, 0, v_inst_1549_);
v___f_1563_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divInvMonoid___redArg___lam__1), 3, 1);
lean_closure_set(v___f_1563_, 0, v_inst_1552_);
v___x_1564_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1564_, 0, v_inst_1548_);
lean_ctor_set(v___x_1564_, 1, v_inst_1547_);
lean_ctor_set(v___x_1564_, 2, v___f_1562_);
v___x_1565_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1565_, 0, v___x_1564_);
lean_ctor_set(v___x_1565_, 1, v_inst_1550_);
lean_ctor_set(v___x_1565_, 2, v_inst_1551_);
lean_ctor_set(v___x_1565_, 3, v___f_1563_);
return v___x_1565_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_divInvMonoid___boxed(lean_object** _args){
lean_object* v_M_u2081_1566_ = _args[0];
lean_object* v_M_u2082_1567_ = _args[1];
lean_object* v_inst_1568_ = _args[2];
lean_object* v_inst_1569_ = _args[3];
lean_object* v_inst_1570_ = _args[4];
lean_object* v_inst_1571_ = _args[5];
lean_object* v_inst_1572_ = _args[6];
lean_object* v_inst_1573_ = _args[7];
lean_object* v_inst_1574_ = _args[8];
lean_object* v_f_1575_ = _args[9];
lean_object* v_hf_1576_ = _args[10];
lean_object* v_one_1577_ = _args[11];
lean_object* v_mul_1578_ = _args[12];
lean_object* v_inv_1579_ = _args[13];
lean_object* v_div_1580_ = _args[14];
lean_object* v_npow_1581_ = _args[15];
lean_object* v_zpow_1582_ = _args[16];
_start:
{
lean_object* v_res_1583_; 
v_res_1583_ = lp_mathlib_Function_Surjective_divInvMonoid(v_M_u2081_1566_, v_M_u2082_1567_, v_inst_1568_, v_inst_1569_, v_inst_1570_, v_inst_1571_, v_inst_1572_, v_inst_1573_, v_inst_1574_, v_f_1575_, v_hf_1576_, v_one_1577_, v_mul_1578_, v_inv_1579_, v_div_1580_, v_npow_1581_, v_zpow_1582_);
lean_dec(v_f_1575_);
lean_dec_ref(v_inst_1574_);
return v_res_1583_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_subNegMonoid___redArg(lean_object* v_inst_1584_, lean_object* v_inst_1585_, lean_object* v_inst_1586_, lean_object* v_inst_1587_, lean_object* v_inst_1588_, lean_object* v_inst_1589_){
_start:
{
lean_object* v___f_1590_; lean_object* v___x_1591_; lean_object* v___x_1592_; 
v___f_1590_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_subNegMonoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1590_, 0, v_inst_1589_);
v___x_1591_ = lp_mathlib_Function_Surjective_addMonoid___redArg(v_inst_1584_, v_inst_1585_, v_inst_1586_);
v___x_1592_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1592_, 0, v___x_1591_);
lean_ctor_set(v___x_1592_, 1, v_inst_1587_);
lean_ctor_set(v___x_1592_, 2, v_inst_1588_);
lean_ctor_set(v___x_1592_, 3, v___f_1590_);
return v___x_1592_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_subNegMonoid(lean_object* v_M_u2081_1593_, lean_object* v_M_u2082_1594_, lean_object* v_inst_1595_, lean_object* v_inst_1596_, lean_object* v_inst_1597_, lean_object* v_inst_1598_, lean_object* v_inst_1599_, lean_object* v_inst_1600_, lean_object* v_inst_1601_, lean_object* v_f_1602_, lean_object* v_hf_1603_, lean_object* v_one_1604_, lean_object* v_mul_1605_, lean_object* v_inv_1606_, lean_object* v_div_1607_, lean_object* v_npow_1608_, lean_object* v_zpow_1609_){
_start:
{
lean_object* v___x_1610_; 
v___x_1610_ = lp_mathlib_Function_Surjective_subNegMonoid___redArg(v_inst_1595_, v_inst_1596_, v_inst_1597_, v_inst_1598_, v_inst_1599_, v_inst_1600_);
return v___x_1610_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_subNegMonoid___boxed(lean_object** _args){
lean_object* v_M_u2081_1611_ = _args[0];
lean_object* v_M_u2082_1612_ = _args[1];
lean_object* v_inst_1613_ = _args[2];
lean_object* v_inst_1614_ = _args[3];
lean_object* v_inst_1615_ = _args[4];
lean_object* v_inst_1616_ = _args[5];
lean_object* v_inst_1617_ = _args[6];
lean_object* v_inst_1618_ = _args[7];
lean_object* v_inst_1619_ = _args[8];
lean_object* v_f_1620_ = _args[9];
lean_object* v_hf_1621_ = _args[10];
lean_object* v_one_1622_ = _args[11];
lean_object* v_mul_1623_ = _args[12];
lean_object* v_inv_1624_ = _args[13];
lean_object* v_div_1625_ = _args[14];
lean_object* v_npow_1626_ = _args[15];
lean_object* v_zpow_1627_ = _args[16];
_start:
{
lean_object* v_res_1628_; 
v_res_1628_ = lp_mathlib_Function_Surjective_subNegMonoid(v_M_u2081_1611_, v_M_u2082_1612_, v_inst_1613_, v_inst_1614_, v_inst_1615_, v_inst_1616_, v_inst_1617_, v_inst_1618_, v_inst_1619_, v_f_1620_, v_hf_1621_, v_one_1622_, v_mul_1623_, v_inv_1624_, v_div_1625_, v_npow_1626_, v_zpow_1627_);
lean_dec(v_f_1620_);
lean_dec_ref(v_inst_1619_);
return v_res_1628_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_group___redArg(lean_object* v_inst_1629_, lean_object* v_inst_1630_, lean_object* v_inst_1631_, lean_object* v_inst_1632_, lean_object* v_inst_1633_, lean_object* v_inst_1634_){
_start:
{
lean_object* v___f_1635_; lean_object* v___f_1636_; lean_object* v___x_1637_; lean_object* v___x_1638_; 
v___f_1635_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1635_, 0, v_inst_1631_);
v___f_1636_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divInvMonoid___redArg___lam__1), 3, 1);
lean_closure_set(v___f_1636_, 0, v_inst_1634_);
v___x_1637_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1637_, 0, v_inst_1630_);
lean_ctor_set(v___x_1637_, 1, v_inst_1629_);
lean_ctor_set(v___x_1637_, 2, v___f_1635_);
v___x_1638_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1638_, 0, v___x_1637_);
lean_ctor_set(v___x_1638_, 1, v_inst_1632_);
lean_ctor_set(v___x_1638_, 2, v_inst_1633_);
lean_ctor_set(v___x_1638_, 3, v___f_1636_);
return v___x_1638_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_group(lean_object* v_M_u2081_1639_, lean_object* v_M_u2082_1640_, lean_object* v_inst_1641_, lean_object* v_inst_1642_, lean_object* v_inst_1643_, lean_object* v_inst_1644_, lean_object* v_inst_1645_, lean_object* v_inst_1646_, lean_object* v_inst_1647_, lean_object* v_f_1648_, lean_object* v_hf_1649_, lean_object* v_one_1650_, lean_object* v_mul_1651_, lean_object* v_inv_1652_, lean_object* v_div_1653_, lean_object* v_npow_1654_, lean_object* v_zpow_1655_){
_start:
{
lean_object* v___f_1656_; lean_object* v___f_1657_; lean_object* v___x_1658_; lean_object* v___x_1659_; 
v___f_1656_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1656_, 0, v_inst_1643_);
v___f_1657_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divInvMonoid___redArg___lam__1), 3, 1);
lean_closure_set(v___f_1657_, 0, v_inst_1646_);
v___x_1658_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1658_, 0, v_inst_1642_);
lean_ctor_set(v___x_1658_, 1, v_inst_1641_);
lean_ctor_set(v___x_1658_, 2, v___f_1656_);
v___x_1659_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1659_, 0, v___x_1658_);
lean_ctor_set(v___x_1659_, 1, v_inst_1644_);
lean_ctor_set(v___x_1659_, 2, v_inst_1645_);
lean_ctor_set(v___x_1659_, 3, v___f_1657_);
return v___x_1659_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_group___boxed(lean_object** _args){
lean_object* v_M_u2081_1660_ = _args[0];
lean_object* v_M_u2082_1661_ = _args[1];
lean_object* v_inst_1662_ = _args[2];
lean_object* v_inst_1663_ = _args[3];
lean_object* v_inst_1664_ = _args[4];
lean_object* v_inst_1665_ = _args[5];
lean_object* v_inst_1666_ = _args[6];
lean_object* v_inst_1667_ = _args[7];
lean_object* v_inst_1668_ = _args[8];
lean_object* v_f_1669_ = _args[9];
lean_object* v_hf_1670_ = _args[10];
lean_object* v_one_1671_ = _args[11];
lean_object* v_mul_1672_ = _args[12];
lean_object* v_inv_1673_ = _args[13];
lean_object* v_div_1674_ = _args[14];
lean_object* v_npow_1675_ = _args[15];
lean_object* v_zpow_1676_ = _args[16];
_start:
{
lean_object* v_res_1677_; 
v_res_1677_ = lp_mathlib_Function_Surjective_group(v_M_u2081_1660_, v_M_u2082_1661_, v_inst_1662_, v_inst_1663_, v_inst_1664_, v_inst_1665_, v_inst_1666_, v_inst_1667_, v_inst_1668_, v_f_1669_, v_hf_1670_, v_one_1671_, v_mul_1672_, v_inv_1673_, v_div_1674_, v_npow_1675_, v_zpow_1676_);
lean_dec(v_f_1669_);
lean_dec_ref(v_inst_1668_);
return v_res_1677_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addGroup___redArg(lean_object* v_inst_1678_, lean_object* v_inst_1679_, lean_object* v_inst_1680_, lean_object* v_inst_1681_, lean_object* v_inst_1682_, lean_object* v_inst_1683_){
_start:
{
lean_object* v___x_1684_; 
v___x_1684_ = lp_mathlib_Function_Surjective_subNegMonoid___redArg(v_inst_1678_, v_inst_1679_, v_inst_1680_, v_inst_1681_, v_inst_1682_, v_inst_1683_);
return v___x_1684_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addGroup(lean_object* v_M_u2081_1685_, lean_object* v_M_u2082_1686_, lean_object* v_inst_1687_, lean_object* v_inst_1688_, lean_object* v_inst_1689_, lean_object* v_inst_1690_, lean_object* v_inst_1691_, lean_object* v_inst_1692_, lean_object* v_inst_1693_, lean_object* v_f_1694_, lean_object* v_hf_1695_, lean_object* v_one_1696_, lean_object* v_mul_1697_, lean_object* v_inv_1698_, lean_object* v_div_1699_, lean_object* v_npow_1700_, lean_object* v_zpow_1701_){
_start:
{
lean_object* v___x_1702_; 
v___x_1702_ = lp_mathlib_Function_Surjective_subNegMonoid___redArg(v_inst_1687_, v_inst_1688_, v_inst_1689_, v_inst_1690_, v_inst_1691_, v_inst_1692_);
return v___x_1702_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addGroup___boxed(lean_object** _args){
lean_object* v_M_u2081_1703_ = _args[0];
lean_object* v_M_u2082_1704_ = _args[1];
lean_object* v_inst_1705_ = _args[2];
lean_object* v_inst_1706_ = _args[3];
lean_object* v_inst_1707_ = _args[4];
lean_object* v_inst_1708_ = _args[5];
lean_object* v_inst_1709_ = _args[6];
lean_object* v_inst_1710_ = _args[7];
lean_object* v_inst_1711_ = _args[8];
lean_object* v_f_1712_ = _args[9];
lean_object* v_hf_1713_ = _args[10];
lean_object* v_one_1714_ = _args[11];
lean_object* v_mul_1715_ = _args[12];
lean_object* v_inv_1716_ = _args[13];
lean_object* v_div_1717_ = _args[14];
lean_object* v_npow_1718_ = _args[15];
lean_object* v_zpow_1719_ = _args[16];
_start:
{
lean_object* v_res_1720_; 
v_res_1720_ = lp_mathlib_Function_Surjective_addGroup(v_M_u2081_1703_, v_M_u2082_1704_, v_inst_1705_, v_inst_1706_, v_inst_1707_, v_inst_1708_, v_inst_1709_, v_inst_1710_, v_inst_1711_, v_f_1712_, v_hf_1713_, v_one_1714_, v_mul_1715_, v_inv_1716_, v_div_1717_, v_npow_1718_, v_zpow_1719_);
lean_dec(v_f_1712_);
lean_dec_ref(v_inst_1711_);
return v_res_1720_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commGroup___redArg(lean_object* v_inst_1721_, lean_object* v_inst_1722_, lean_object* v_inst_1723_, lean_object* v_inst_1724_, lean_object* v_inst_1725_, lean_object* v_inst_1726_){
_start:
{
lean_object* v___f_1727_; lean_object* v___f_1728_; lean_object* v___x_1729_; lean_object* v___x_1730_; 
v___f_1727_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1727_, 0, v_inst_1723_);
v___f_1728_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divInvMonoid___redArg___lam__1), 3, 1);
lean_closure_set(v___f_1728_, 0, v_inst_1726_);
v___x_1729_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1729_, 0, v_inst_1722_);
lean_ctor_set(v___x_1729_, 1, v_inst_1721_);
lean_ctor_set(v___x_1729_, 2, v___f_1727_);
v___x_1730_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1730_, 0, v___x_1729_);
lean_ctor_set(v___x_1730_, 1, v_inst_1724_);
lean_ctor_set(v___x_1730_, 2, v_inst_1725_);
lean_ctor_set(v___x_1730_, 3, v___f_1728_);
return v___x_1730_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commGroup(lean_object* v_M_u2081_1731_, lean_object* v_M_u2082_1732_, lean_object* v_inst_1733_, lean_object* v_inst_1734_, lean_object* v_inst_1735_, lean_object* v_inst_1736_, lean_object* v_inst_1737_, lean_object* v_inst_1738_, lean_object* v_inst_1739_, lean_object* v_f_1740_, lean_object* v_hf_1741_, lean_object* v_one_1742_, lean_object* v_mul_1743_, lean_object* v_inv_1744_, lean_object* v_div_1745_, lean_object* v_npow_1746_, lean_object* v_zpow_1747_){
_start:
{
lean_object* v___f_1748_; lean_object* v___f_1749_; lean_object* v___x_1750_; lean_object* v___x_1751_; 
v___f_1748_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_monoid___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1748_, 0, v_inst_1735_);
v___f_1749_ = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_divInvMonoid___redArg___lam__1), 3, 1);
lean_closure_set(v___f_1749_, 0, v_inst_1738_);
v___x_1750_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1750_, 0, v_inst_1734_);
lean_ctor_set(v___x_1750_, 1, v_inst_1733_);
lean_ctor_set(v___x_1750_, 2, v___f_1748_);
v___x_1751_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1751_, 0, v___x_1750_);
lean_ctor_set(v___x_1751_, 1, v_inst_1736_);
lean_ctor_set(v___x_1751_, 2, v_inst_1737_);
lean_ctor_set(v___x_1751_, 3, v___f_1749_);
return v___x_1751_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_commGroup___boxed(lean_object** _args){
lean_object* v_M_u2081_1752_ = _args[0];
lean_object* v_M_u2082_1753_ = _args[1];
lean_object* v_inst_1754_ = _args[2];
lean_object* v_inst_1755_ = _args[3];
lean_object* v_inst_1756_ = _args[4];
lean_object* v_inst_1757_ = _args[5];
lean_object* v_inst_1758_ = _args[6];
lean_object* v_inst_1759_ = _args[7];
lean_object* v_inst_1760_ = _args[8];
lean_object* v_f_1761_ = _args[9];
lean_object* v_hf_1762_ = _args[10];
lean_object* v_one_1763_ = _args[11];
lean_object* v_mul_1764_ = _args[12];
lean_object* v_inv_1765_ = _args[13];
lean_object* v_div_1766_ = _args[14];
lean_object* v_npow_1767_ = _args[15];
lean_object* v_zpow_1768_ = _args[16];
_start:
{
lean_object* v_res_1769_; 
v_res_1769_ = lp_mathlib_Function_Surjective_commGroup(v_M_u2081_1752_, v_M_u2082_1753_, v_inst_1754_, v_inst_1755_, v_inst_1756_, v_inst_1757_, v_inst_1758_, v_inst_1759_, v_inst_1760_, v_f_1761_, v_hf_1762_, v_one_1763_, v_mul_1764_, v_inv_1765_, v_div_1766_, v_npow_1767_, v_zpow_1768_);
lean_dec(v_f_1761_);
lean_dec_ref(v_inst_1760_);
return v_res_1769_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommGroup___redArg(lean_object* v_inst_1770_, lean_object* v_inst_1771_, lean_object* v_inst_1772_, lean_object* v_inst_1773_, lean_object* v_inst_1774_, lean_object* v_inst_1775_){
_start:
{
lean_object* v___x_1776_; 
v___x_1776_ = lp_mathlib_Function_Surjective_subNegMonoid___redArg(v_inst_1770_, v_inst_1771_, v_inst_1772_, v_inst_1773_, v_inst_1774_, v_inst_1775_);
return v___x_1776_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommGroup(lean_object* v_M_u2081_1777_, lean_object* v_M_u2082_1778_, lean_object* v_inst_1779_, lean_object* v_inst_1780_, lean_object* v_inst_1781_, lean_object* v_inst_1782_, lean_object* v_inst_1783_, lean_object* v_inst_1784_, lean_object* v_inst_1785_, lean_object* v_f_1786_, lean_object* v_hf_1787_, lean_object* v_one_1788_, lean_object* v_mul_1789_, lean_object* v_inv_1790_, lean_object* v_div_1791_, lean_object* v_npow_1792_, lean_object* v_zpow_1793_){
_start:
{
lean_object* v___x_1794_; 
v___x_1794_ = lp_mathlib_Function_Surjective_subNegMonoid___redArg(v_inst_1779_, v_inst_1780_, v_inst_1781_, v_inst_1782_, v_inst_1783_, v_inst_1784_);
return v___x_1794_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Surjective_addCommGroup___boxed(lean_object** _args){
lean_object* v_M_u2081_1795_ = _args[0];
lean_object* v_M_u2082_1796_ = _args[1];
lean_object* v_inst_1797_ = _args[2];
lean_object* v_inst_1798_ = _args[3];
lean_object* v_inst_1799_ = _args[4];
lean_object* v_inst_1800_ = _args[5];
lean_object* v_inst_1801_ = _args[6];
lean_object* v_inst_1802_ = _args[7];
lean_object* v_inst_1803_ = _args[8];
lean_object* v_f_1804_ = _args[9];
lean_object* v_hf_1805_ = _args[10];
lean_object* v_one_1806_ = _args[11];
lean_object* v_mul_1807_ = _args[12];
lean_object* v_inv_1808_ = _args[13];
lean_object* v_div_1809_ = _args[14];
lean_object* v_npow_1810_ = _args[15];
lean_object* v_zpow_1811_ = _args[16];
_start:
{
lean_object* v_res_1812_; 
v_res_1812_ = lp_mathlib_Function_Surjective_addCommGroup(v_M_u2081_1795_, v_M_u2082_1796_, v_inst_1797_, v_inst_1798_, v_inst_1799_, v_inst_1800_, v_inst_1801_, v_inst_1802_, v_inst_1803_, v_f_1804_, v_hf_1805_, v_one_1806_, v_mul_1807_, v_inv_1808_, v_div_1809_, v_npow_1810_, v_zpow_1811_);
lean_dec(v_f_1804_);
lean_dec_ref(v_inst_1803_);
return v_res_1812_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Function_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Spread(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_InjSurj(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Logic_Function_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Spread(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_InjSurj(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Logic_Function_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Spread(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_InjSurj(uint8_t builtin) {
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
res = initialize_mathlib_Mathlib_Logic_Function_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Spread(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_InjSurj(builtin);
}
#ifdef __cplusplus
}
#endif
