// Lean compiler output
// Module: Mathlib.Algebra.Group.Prod
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Equiv.Defs public import Mathlib.Algebra.Group.Hom.Basic public import Mathlib.Algebra.Group.Opposite public import Mathlib.Algebra.Group.SelfInv public import Mathlib.Algebra.Group.Torsion public import Mathlib.Algebra.Group.Units.Hom public import Mathlib.Algebra.Notation.Pi.Defs public import Mathlib.Algebra.Notation.Prod public import Mathlib.Logic.Equiv.Prod public import Mathlib.Tactic.TermCongr
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
lean_object* lp_mathlib_Equiv_prodAssoc(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OneHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_Prod_instMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_Function_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_instInv___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_prodCongr___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Units_map___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_prodUnique___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_uniqueProd___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_prodComm(lean_object*, lean_object*);
lean_object* lp_mathlib_AddUnits_map___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instInvolutiveInv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instInvolutiveInv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instInvolutiveNeg___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instInvolutiveNeg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_commMagma___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_commMagma(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_addCommMagma___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_addCommMagma(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSemigroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSemigroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddSemigroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddSemigroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCommSemigroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCommSemigroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddCommSemigroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddCommSemigroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMulOneClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMulOneClass(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddZeroClass___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddZeroClass(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instDivInvMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instDivInvMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instDivInvMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_subNegMonoid___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_subNegMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_subNegMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instDivisionMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instDivisionMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSubtractionMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSubtractionMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instDivisionCommMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instDivisionCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_SubtractionCommMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_SubtractionCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instGroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instGroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddGroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddGroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLeftCancelSemigroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLeftCancelSemigroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddLeftCancelSemigroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddLeftCancelSemigroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instRightCancelSemigroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instRightCancelSemigroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddRightCancelSemigroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddRightCancelSemigroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLeftCancelMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLeftCancelMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddLeftCancelMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddLeftCancelMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instRightCancelMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instRightCancelMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddRightCancelMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddRightCancelMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCancelMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCancelMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddCancelMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddCancelMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCommMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddCommMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCancelCommMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCancelCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddCancelCommMonoid___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddCancelCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCommGroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCommGroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddCommGroup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddCommGroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_fst___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_fst___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_MulHom_fst___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulHom_fst___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulHom_fst___closed__0 = (const lean_object*)&lp_mathlib_MulHom_fst___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_MulHom_fst(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_fst___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_fst(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_fst___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_snd___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_snd___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_MulHom_snd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulHom_snd___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulHom_snd___closed__0 = (const lean_object*)&lp_mathlib_MulHom_snd___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_MulHom_snd(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_snd___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_snd(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_snd___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_prod___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_prod___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_prod___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_prod___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_prodMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_prodMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_prodMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_prodMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_prodMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_prodMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_coprod___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_coprod___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_coprod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulHom_coprod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_coprod___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_coprod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_coprod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fst(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fst___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fst(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fst___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_snd(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_snd___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_snd(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_snd___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inl___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inl___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inl(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inl___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inl___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inl___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inl(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inl___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inr___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inr___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inr___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inr___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_prod___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_prod___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_prodMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_prodMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_prodMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_prodMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_prodMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_prodMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_coprod___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_coprod___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_coprod___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_coprod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_coprod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_coprod___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_coprod___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_coprod___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_coprod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_coprod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_MulEquiv_prodComm___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MulEquiv_prodComm___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodComm(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodComm___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodComm(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodComm___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_MulEquiv_prodAssoc___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MulEquiv_prodAssoc___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodAssoc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodAssoc___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodAssoc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodAssoc___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodProdProdComm___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodProdProdComm___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_MulEquiv_prodProdProdComm___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulEquiv_prodProdProdComm___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulEquiv_prodProdProdComm___closed__0 = (const lean_object*)&lp_mathlib_MulEquiv_prodProdProdComm___closed__0_value;
static const lean_closure_object lp_mathlib_MulEquiv_prodProdProdComm___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulEquiv_prodProdProdComm___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulEquiv_prodProdProdComm___closed__1 = (const lean_object*)&lp_mathlib_MulEquiv_prodProdProdComm___closed__1_value;
static const lean_ctor_object lp_mathlib_MulEquiv_prodProdProdComm___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_MulEquiv_prodProdProdComm___closed__0_value),((lean_object*)&lp_mathlib_MulEquiv_prodProdProdComm___closed__1_value)}};
static const lean_object* lp_mathlib_MulEquiv_prodProdProdComm___closed__2 = (const lean_object*)&lp_mathlib_MulEquiv_prodProdProdComm___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodProdProdComm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodProdProdComm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodProdProdComm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodProdProdComm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_uniqueProd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_uniqueProd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_uniqueProd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_uniqueProd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_uniqueProd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_uniqueProd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodUnique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodUnique(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodUnique___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodUnique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodUnique(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodUnique___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodUnits___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodUnits___lam__1(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_MulEquiv_prodUnits___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulEquiv_prodUnits___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulEquiv_prodUnits___closed__0 = (const lean_object*)&lp_mathlib_MulEquiv_prodUnits___closed__0_value;
static const lean_closure_object lp_mathlib_MulEquiv_prodUnits___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Units_map___redArg___lam__0, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_MulHom_fst___closed__0_value)} };
static const lean_object* lp_mathlib_MulEquiv_prodUnits___closed__1 = (const lean_object*)&lp_mathlib_MulEquiv_prodUnits___closed__1_value;
static const lean_closure_object lp_mathlib_MulEquiv_prodUnits___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Units_map___redArg___lam__0, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_MulHom_snd___closed__0_value)} };
static const lean_object* lp_mathlib_MulEquiv_prodUnits___closed__2 = (const lean_object*)&lp_mathlib_MulEquiv_prodUnits___closed__2_value;
static const lean_closure_object lp_mathlib_MulEquiv_prodUnits___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulEquiv_prodUnits___lam__1, .m_arity = 3, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_MulEquiv_prodUnits___closed__1_value),((lean_object*)&lp_mathlib_MulEquiv_prodUnits___closed__2_value)} };
static const lean_object* lp_mathlib_MulEquiv_prodUnits___closed__3 = (const lean_object*)&lp_mathlib_MulEquiv_prodUnits___closed__3_value;
static const lean_ctor_object lp_mathlib_MulEquiv_prodUnits___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_MulEquiv_prodUnits___closed__3_value),((lean_object*)&lp_mathlib_MulEquiv_prodUnits___closed__0_value)}};
static const lean_object* lp_mathlib_MulEquiv_prodUnits___closed__4 = (const lean_object*)&lp_mathlib_MulEquiv_prodUnits___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodUnits(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodUnits___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodAddUnits___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodAddUnits___lam__1(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AddEquiv_prodAddUnits___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddEquiv_prodAddUnits___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddEquiv_prodAddUnits___closed__0 = (const lean_object*)&lp_mathlib_AddEquiv_prodAddUnits___closed__0_value;
static const lean_closure_object lp_mathlib_AddEquiv_prodAddUnits___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddUnits_map___redArg___lam__0, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_MulHom_fst___closed__0_value)} };
static const lean_object* lp_mathlib_AddEquiv_prodAddUnits___closed__1 = (const lean_object*)&lp_mathlib_AddEquiv_prodAddUnits___closed__1_value;
static const lean_closure_object lp_mathlib_AddEquiv_prodAddUnits___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddUnits_map___redArg___lam__0, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_MulHom_snd___closed__0_value)} };
static const lean_object* lp_mathlib_AddEquiv_prodAddUnits___closed__2 = (const lean_object*)&lp_mathlib_AddEquiv_prodAddUnits___closed__2_value;
static const lean_closure_object lp_mathlib_AddEquiv_prodAddUnits___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddEquiv_prodAddUnits___lam__1, .m_arity = 3, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_AddEquiv_prodAddUnits___closed__1_value),((lean_object*)&lp_mathlib_AddEquiv_prodAddUnits___closed__2_value)} };
static const lean_object* lp_mathlib_AddEquiv_prodAddUnits___closed__3 = (const lean_object*)&lp_mathlib_AddEquiv_prodAddUnits___closed__3_value;
static const lean_ctor_object lp_mathlib_AddEquiv_prodAddUnits___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_AddEquiv_prodAddUnits___closed__3_value),((lean_object*)&lp_mathlib_AddEquiv_prodAddUnits___closed__0_value)}};
static const lean_object* lp_mathlib_AddEquiv_prodAddUnits___closed__4 = (const lean_object*)&lp_mathlib_AddEquiv_prodAddUnits___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodAddUnits(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodAddUnits___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_embedProduct___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Units_embedProduct___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Units_embedProduct___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Units_embedProduct___closed__0 = (const lean_object*)&lp_mathlib_Units_embedProduct___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Units_embedProduct(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Units_embedProduct___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_embedProduct___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_AddUnits_embedProduct___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddUnits_embedProduct___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddUnits_embedProduct___closed__0 = (const lean_object*)&lp_mathlib_AddUnits_embedProduct___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_embedProduct(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_embedProduct___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mulMulHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mulMulHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mulMulHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addAddHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addAddHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mulMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mulMonoidHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addAddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addAddMonoidHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_divMonoidHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_divMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_divMonoidHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_subAddMonoidHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_subAddMonoidHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_subAddMonoidHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instInvolutiveInv___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___f_3_; 
v___f_3_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instInv___redArg___lam__0), 3, 2);
lean_closure_set(v___f_3_, 0, v_inst_1_);
lean_closure_set(v___f_3_, 1, v_inst_2_);
return v___f_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instInvolutiveInv(lean_object* v_M_4_, lean_object* v_N_5_, lean_object* v_inst_6_, lean_object* v_inst_7_){
_start:
{
lean_object* v___f_8_; 
v___f_8_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instInv___redArg___lam__0), 3, 2);
lean_closure_set(v___f_8_, 0, v_inst_6_);
lean_closure_set(v___f_8_, 1, v_inst_7_);
return v___f_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instInvolutiveNeg___redArg(lean_object* v_inst_9_, lean_object* v_inst_10_){
_start:
{
lean_object* v___f_11_; 
v___f_11_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instInv___redArg___lam__0), 3, 2);
lean_closure_set(v___f_11_, 0, v_inst_9_);
lean_closure_set(v___f_11_, 1, v_inst_10_);
return v___f_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instInvolutiveNeg(lean_object* v_M_12_, lean_object* v_N_13_, lean_object* v_inst_14_, lean_object* v_inst_15_){
_start:
{
lean_object* v___f_16_; 
v___f_16_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instInv___redArg___lam__0), 3, 2);
lean_closure_set(v___f_16_, 0, v_inst_14_);
lean_closure_set(v___f_16_, 1, v_inst_15_);
return v___f_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_commMagma___redArg(lean_object* v_inst_17_, lean_object* v_inst_18_){
_start:
{
lean_object* v___f_19_; 
v___f_19_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_19_, 0, v_inst_17_);
lean_closure_set(v___f_19_, 1, v_inst_18_);
return v___f_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_commMagma(lean_object* v_M_20_, lean_object* v_N_21_, lean_object* v_inst_22_, lean_object* v_inst_23_){
_start:
{
lean_object* v___f_24_; 
v___f_24_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_24_, 0, v_inst_22_);
lean_closure_set(v___f_24_, 1, v_inst_23_);
return v___f_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_addCommMagma___redArg(lean_object* v_inst_25_, lean_object* v_inst_26_){
_start:
{
lean_object* v___f_27_; 
v___f_27_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_27_, 0, v_inst_25_);
lean_closure_set(v___f_27_, 1, v_inst_26_);
return v___f_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_addCommMagma(lean_object* v_M_28_, lean_object* v_N_29_, lean_object* v_inst_30_, lean_object* v_inst_31_){
_start:
{
lean_object* v___f_32_; 
v___f_32_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_32_, 0, v_inst_30_);
lean_closure_set(v___f_32_, 1, v_inst_31_);
return v___f_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSemigroup___redArg(lean_object* v_inst_33_, lean_object* v_inst_34_){
_start:
{
lean_object* v___f_35_; 
v___f_35_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_35_, 0, v_inst_33_);
lean_closure_set(v___f_35_, 1, v_inst_34_);
return v___f_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSemigroup(lean_object* v_M_36_, lean_object* v_N_37_, lean_object* v_inst_38_, lean_object* v_inst_39_){
_start:
{
lean_object* v___f_40_; 
v___f_40_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_40_, 0, v_inst_38_);
lean_closure_set(v___f_40_, 1, v_inst_39_);
return v___f_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddSemigroup___redArg(lean_object* v_inst_41_, lean_object* v_inst_42_){
_start:
{
lean_object* v___f_43_; 
v___f_43_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_43_, 0, v_inst_41_);
lean_closure_set(v___f_43_, 1, v_inst_42_);
return v___f_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddSemigroup(lean_object* v_M_44_, lean_object* v_N_45_, lean_object* v_inst_46_, lean_object* v_inst_47_){
_start:
{
lean_object* v___f_48_; 
v___f_48_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_48_, 0, v_inst_46_);
lean_closure_set(v___f_48_, 1, v_inst_47_);
return v___f_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCommSemigroup___redArg(lean_object* v_inst_49_, lean_object* v_inst_50_){
_start:
{
lean_object* v___f_51_; 
v___f_51_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_51_, 0, v_inst_49_);
lean_closure_set(v___f_51_, 1, v_inst_50_);
return v___f_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCommSemigroup(lean_object* v_G_52_, lean_object* v_H_53_, lean_object* v_inst_54_, lean_object* v_inst_55_){
_start:
{
lean_object* v___f_56_; 
v___f_56_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_56_, 0, v_inst_54_);
lean_closure_set(v___f_56_, 1, v_inst_55_);
return v___f_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddCommSemigroup___redArg(lean_object* v_inst_57_, lean_object* v_inst_58_){
_start:
{
lean_object* v___f_59_; 
v___f_59_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_59_, 0, v_inst_57_);
lean_closure_set(v___f_59_, 1, v_inst_58_);
return v___f_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddCommSemigroup(lean_object* v_G_60_, lean_object* v_H_61_, lean_object* v_inst_62_, lean_object* v_inst_63_){
_start:
{
lean_object* v___f_64_; 
v___f_64_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_64_, 0, v_inst_62_);
lean_closure_set(v___f_64_, 1, v_inst_63_);
return v___f_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMulOneClass___redArg(lean_object* v_inst_65_, lean_object* v_inst_66_){
_start:
{
lean_object* v___x_67_; lean_object* v_toOne_68_; lean_object* v_toMul_69_; lean_object* v___x_71_; uint8_t v_isShared_72_; uint8_t v_isSharedCheck_87_; 
v___x_67_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_65_);
v_toOne_68_ = lean_ctor_get(v___x_67_, 0);
v_toMul_69_ = lean_ctor_get(v___x_67_, 1);
v_isSharedCheck_87_ = !lean_is_exclusive(v___x_67_);
if (v_isSharedCheck_87_ == 0)
{
v___x_71_ = v___x_67_;
v_isShared_72_ = v_isSharedCheck_87_;
goto v_resetjp_70_;
}
else
{
lean_inc(v_toMul_69_);
lean_inc(v_toOne_68_);
lean_dec(v___x_67_);
v___x_71_ = lean_box(0);
v_isShared_72_ = v_isSharedCheck_87_;
goto v_resetjp_70_;
}
v_resetjp_70_:
{
lean_object* v___x_73_; lean_object* v_toOne_74_; lean_object* v_toMul_75_; lean_object* v___x_77_; uint8_t v_isShared_78_; uint8_t v_isSharedCheck_86_; 
v___x_73_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_66_);
v_toOne_74_ = lean_ctor_get(v___x_73_, 0);
v_toMul_75_ = lean_ctor_get(v___x_73_, 1);
v_isSharedCheck_86_ = !lean_is_exclusive(v___x_73_);
if (v_isSharedCheck_86_ == 0)
{
v___x_77_ = v___x_73_;
v_isShared_78_ = v_isSharedCheck_86_;
goto v_resetjp_76_;
}
else
{
lean_inc(v_toMul_75_);
lean_inc(v_toOne_74_);
lean_dec(v___x_73_);
v___x_77_ = lean_box(0);
v_isShared_78_ = v_isSharedCheck_86_;
goto v_resetjp_76_;
}
v_resetjp_76_:
{
lean_object* v___x_80_; 
if (v_isShared_78_ == 0)
{
lean_ctor_set(v___x_77_, 1, v_toOne_74_);
lean_ctor_set(v___x_77_, 0, v_toOne_68_);
v___x_80_ = v___x_77_;
goto v_reusejp_79_;
}
else
{
lean_object* v_reuseFailAlloc_85_; 
v_reuseFailAlloc_85_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_85_, 0, v_toOne_68_);
lean_ctor_set(v_reuseFailAlloc_85_, 1, v_toOne_74_);
v___x_80_ = v_reuseFailAlloc_85_;
goto v_reusejp_79_;
}
v_reusejp_79_:
{
lean_object* v___f_81_; lean_object* v___x_83_; 
v___f_81_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_81_, 0, v_toMul_69_);
lean_closure_set(v___f_81_, 1, v_toMul_75_);
if (v_isShared_72_ == 0)
{
lean_ctor_set(v___x_71_, 1, v___f_81_);
lean_ctor_set(v___x_71_, 0, v___x_80_);
v___x_83_ = v___x_71_;
goto v_reusejp_82_;
}
else
{
lean_object* v_reuseFailAlloc_84_; 
v_reuseFailAlloc_84_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_84_, 0, v___x_80_);
lean_ctor_set(v_reuseFailAlloc_84_, 1, v___f_81_);
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
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMulOneClass(lean_object* v_M_88_, lean_object* v_N_89_, lean_object* v_inst_90_, lean_object* v_inst_91_){
_start:
{
lean_object* v___x_92_; 
v___x_92_ = lp_mathlib_Prod_instMulOneClass___redArg(v_inst_90_, v_inst_91_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddZeroClass___redArg(lean_object* v_inst_93_, lean_object* v_inst_94_){
_start:
{
lean_object* v___x_95_; lean_object* v_toZero_96_; lean_object* v_toAdd_97_; lean_object* v___x_99_; uint8_t v_isShared_100_; uint8_t v_isSharedCheck_115_; 
v___x_95_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_93_);
v_toZero_96_ = lean_ctor_get(v___x_95_, 0);
v_toAdd_97_ = lean_ctor_get(v___x_95_, 1);
v_isSharedCheck_115_ = !lean_is_exclusive(v___x_95_);
if (v_isSharedCheck_115_ == 0)
{
v___x_99_ = v___x_95_;
v_isShared_100_ = v_isSharedCheck_115_;
goto v_resetjp_98_;
}
else
{
lean_inc(v_toAdd_97_);
lean_inc(v_toZero_96_);
lean_dec(v___x_95_);
v___x_99_ = lean_box(0);
v_isShared_100_ = v_isSharedCheck_115_;
goto v_resetjp_98_;
}
v_resetjp_98_:
{
lean_object* v___x_101_; lean_object* v_toZero_102_; lean_object* v_toAdd_103_; lean_object* v___x_105_; uint8_t v_isShared_106_; uint8_t v_isSharedCheck_114_; 
v___x_101_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_94_);
v_toZero_102_ = lean_ctor_get(v___x_101_, 0);
v_toAdd_103_ = lean_ctor_get(v___x_101_, 1);
v_isSharedCheck_114_ = !lean_is_exclusive(v___x_101_);
if (v_isSharedCheck_114_ == 0)
{
v___x_105_ = v___x_101_;
v_isShared_106_ = v_isSharedCheck_114_;
goto v_resetjp_104_;
}
else
{
lean_inc(v_toAdd_103_);
lean_inc(v_toZero_102_);
lean_dec(v___x_101_);
v___x_105_ = lean_box(0);
v_isShared_106_ = v_isSharedCheck_114_;
goto v_resetjp_104_;
}
v_resetjp_104_:
{
lean_object* v___x_108_; 
if (v_isShared_106_ == 0)
{
lean_ctor_set(v___x_105_, 1, v_toZero_102_);
lean_ctor_set(v___x_105_, 0, v_toZero_96_);
v___x_108_ = v___x_105_;
goto v_reusejp_107_;
}
else
{
lean_object* v_reuseFailAlloc_113_; 
v_reuseFailAlloc_113_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_113_, 0, v_toZero_96_);
lean_ctor_set(v_reuseFailAlloc_113_, 1, v_toZero_102_);
v___x_108_ = v_reuseFailAlloc_113_;
goto v_reusejp_107_;
}
v_reusejp_107_:
{
lean_object* v___f_109_; lean_object* v___x_111_; 
v___f_109_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_109_, 0, v_toAdd_97_);
lean_closure_set(v___f_109_, 1, v_toAdd_103_);
if (v_isShared_100_ == 0)
{
lean_ctor_set(v___x_99_, 1, v___f_109_);
lean_ctor_set(v___x_99_, 0, v___x_108_);
v___x_111_ = v___x_99_;
goto v_reusejp_110_;
}
else
{
lean_object* v_reuseFailAlloc_112_; 
v_reuseFailAlloc_112_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_112_, 0, v___x_108_);
lean_ctor_set(v_reuseFailAlloc_112_, 1, v___f_109_);
v___x_111_ = v_reuseFailAlloc_112_;
goto v_reusejp_110_;
}
v_reusejp_110_:
{
return v___x_111_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddZeroClass(lean_object* v_M_116_, lean_object* v_N_117_, lean_object* v_inst_118_, lean_object* v_inst_119_){
_start:
{
lean_object* v___x_120_; 
v___x_120_ = lp_mathlib_Prod_instAddZeroClass___redArg(v_inst_118_, v_inst_119_);
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMonoid___redArg___lam__0(lean_object* v_toNPow_121_, lean_object* v_toNPow_122_, lean_object* v_z_123_, lean_object* v_a_124_){
_start:
{
lean_object* v_fst_125_; lean_object* v_snd_126_; lean_object* v___x_128_; uint8_t v_isShared_129_; uint8_t v_isSharedCheck_135_; 
v_fst_125_ = lean_ctor_get(v_a_124_, 0);
v_snd_126_ = lean_ctor_get(v_a_124_, 1);
v_isSharedCheck_135_ = !lean_is_exclusive(v_a_124_);
if (v_isSharedCheck_135_ == 0)
{
v___x_128_ = v_a_124_;
v_isShared_129_ = v_isSharedCheck_135_;
goto v_resetjp_127_;
}
else
{
lean_inc(v_snd_126_);
lean_inc(v_fst_125_);
lean_dec(v_a_124_);
v___x_128_ = lean_box(0);
v_isShared_129_ = v_isSharedCheck_135_;
goto v_resetjp_127_;
}
v_resetjp_127_:
{
lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_133_; 
lean_inc(v_z_123_);
v___x_130_ = lean_apply_2(v_toNPow_121_, v_z_123_, v_fst_125_);
v___x_131_ = lean_apply_2(v_toNPow_122_, v_z_123_, v_snd_126_);
if (v_isShared_129_ == 0)
{
lean_ctor_set(v___x_128_, 1, v___x_131_);
lean_ctor_set(v___x_128_, 0, v___x_130_);
v___x_133_ = v___x_128_;
goto v_reusejp_132_;
}
else
{
lean_object* v_reuseFailAlloc_134_; 
v_reuseFailAlloc_134_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_134_, 0, v___x_130_);
lean_ctor_set(v_reuseFailAlloc_134_, 1, v___x_131_);
v___x_133_ = v_reuseFailAlloc_134_;
goto v_reusejp_132_;
}
v_reusejp_132_:
{
return v___x_133_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMonoid___redArg(lean_object* v_inst_136_, lean_object* v_inst_137_){
_start:
{
lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v_toOne_140_; lean_object* v_toMul_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v_toOne_144_; lean_object* v_toMul_145_; lean_object* v___x_147_; uint8_t v_isShared_148_; uint8_t v_isSharedCheck_165_; 
v___x_138_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_136_);
v___x_139_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_138_);
v_toOne_140_ = lean_ctor_get(v___x_139_, 0);
lean_inc(v_toOne_140_);
v_toMul_141_ = lean_ctor_get(v___x_139_, 1);
lean_inc(v_toMul_141_);
lean_dec_ref(v___x_139_);
v___x_142_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_137_);
v___x_143_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_142_);
v_toOne_144_ = lean_ctor_get(v___x_143_, 0);
v_toMul_145_ = lean_ctor_get(v___x_143_, 1);
v_isSharedCheck_165_ = !lean_is_exclusive(v___x_143_);
if (v_isSharedCheck_165_ == 0)
{
v___x_147_ = v___x_143_;
v_isShared_148_ = v_isSharedCheck_165_;
goto v_resetjp_146_;
}
else
{
lean_inc(v_toMul_145_);
lean_inc(v_toOne_144_);
lean_dec(v___x_143_);
v___x_147_ = lean_box(0);
v_isShared_148_ = v_isSharedCheck_165_;
goto v_resetjp_146_;
}
v_resetjp_146_:
{
lean_object* v_toNPow_149_; lean_object* v_toNPow_150_; lean_object* v___x_152_; uint8_t v_isShared_153_; uint8_t v_isSharedCheck_162_; 
v_toNPow_149_ = lean_ctor_get(v_inst_136_, 2);
lean_inc(v_toNPow_149_);
lean_dec_ref(v_inst_136_);
v_toNPow_150_ = lean_ctor_get(v_inst_137_, 2);
v_isSharedCheck_162_ = !lean_is_exclusive(v_inst_137_);
if (v_isSharedCheck_162_ == 0)
{
lean_object* v_unused_163_; lean_object* v_unused_164_; 
v_unused_163_ = lean_ctor_get(v_inst_137_, 1);
lean_dec(v_unused_163_);
v_unused_164_ = lean_ctor_get(v_inst_137_, 0);
lean_dec(v_unused_164_);
v___x_152_ = v_inst_137_;
v_isShared_153_ = v_isSharedCheck_162_;
goto v_resetjp_151_;
}
else
{
lean_inc(v_toNPow_150_);
lean_dec(v_inst_137_);
v___x_152_ = lean_box(0);
v_isShared_153_ = v_isSharedCheck_162_;
goto v_resetjp_151_;
}
v_resetjp_151_:
{
lean_object* v___x_155_; 
if (v_isShared_148_ == 0)
{
lean_ctor_set(v___x_147_, 1, v_toOne_144_);
lean_ctor_set(v___x_147_, 0, v_toOne_140_);
v___x_155_ = v___x_147_;
goto v_reusejp_154_;
}
else
{
lean_object* v_reuseFailAlloc_161_; 
v_reuseFailAlloc_161_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_161_, 0, v_toOne_140_);
lean_ctor_set(v_reuseFailAlloc_161_, 1, v_toOne_144_);
v___x_155_ = v_reuseFailAlloc_161_;
goto v_reusejp_154_;
}
v_reusejp_154_:
{
lean_object* v___f_156_; lean_object* v___f_157_; lean_object* v___x_159_; 
v___f_156_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_156_, 0, v_toMul_141_);
lean_closure_set(v___f_156_, 1, v_toMul_145_);
v___f_157_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMonoid___redArg___lam__0), 4, 2);
lean_closure_set(v___f_157_, 0, v_toNPow_149_);
lean_closure_set(v___f_157_, 1, v_toNPow_150_);
if (v_isShared_153_ == 0)
{
lean_ctor_set(v___x_152_, 2, v___f_157_);
lean_ctor_set(v___x_152_, 1, v___f_156_);
lean_ctor_set(v___x_152_, 0, v___x_155_);
v___x_159_ = v___x_152_;
goto v_reusejp_158_;
}
else
{
lean_object* v_reuseFailAlloc_160_; 
v_reuseFailAlloc_160_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_160_, 0, v___x_155_);
lean_ctor_set(v_reuseFailAlloc_160_, 1, v___f_156_);
lean_ctor_set(v_reuseFailAlloc_160_, 2, v___f_157_);
v___x_159_ = v_reuseFailAlloc_160_;
goto v_reusejp_158_;
}
v_reusejp_158_:
{
return v___x_159_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instMonoid(lean_object* v_M_166_, lean_object* v_N_167_, lean_object* v_inst_168_, lean_object* v_inst_169_){
_start:
{
lean_object* v___x_170_; 
v___x_170_ = lp_mathlib_Prod_instMonoid___redArg(v_inst_168_, v_inst_169_);
return v___x_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddMonoid___redArg___lam__0(lean_object* v_toNSMul_171_, lean_object* v_toNSMul_172_, lean_object* v_z_173_, lean_object* v_a_174_){
_start:
{
lean_object* v_fst_175_; lean_object* v_snd_176_; lean_object* v___x_178_; uint8_t v_isShared_179_; uint8_t v_isSharedCheck_185_; 
v_fst_175_ = lean_ctor_get(v_a_174_, 0);
v_snd_176_ = lean_ctor_get(v_a_174_, 1);
v_isSharedCheck_185_ = !lean_is_exclusive(v_a_174_);
if (v_isSharedCheck_185_ == 0)
{
v___x_178_ = v_a_174_;
v_isShared_179_ = v_isSharedCheck_185_;
goto v_resetjp_177_;
}
else
{
lean_inc(v_snd_176_);
lean_inc(v_fst_175_);
lean_dec(v_a_174_);
v___x_178_ = lean_box(0);
v_isShared_179_ = v_isSharedCheck_185_;
goto v_resetjp_177_;
}
v_resetjp_177_:
{
lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_183_; 
lean_inc(v_z_173_);
v___x_180_ = lean_apply_2(v_toNSMul_171_, v_z_173_, v_fst_175_);
v___x_181_ = lean_apply_2(v_toNSMul_172_, v_z_173_, v_snd_176_);
if (v_isShared_179_ == 0)
{
lean_ctor_set(v___x_178_, 1, v___x_181_);
lean_ctor_set(v___x_178_, 0, v___x_180_);
v___x_183_ = v___x_178_;
goto v_reusejp_182_;
}
else
{
lean_object* v_reuseFailAlloc_184_; 
v_reuseFailAlloc_184_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_184_, 0, v___x_180_);
lean_ctor_set(v_reuseFailAlloc_184_, 1, v___x_181_);
v___x_183_ = v_reuseFailAlloc_184_;
goto v_reusejp_182_;
}
v_reusejp_182_:
{
return v___x_183_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddMonoid___redArg(lean_object* v_inst_186_, lean_object* v_inst_187_){
_start:
{
lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v_toZero_190_; lean_object* v_toAdd_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v_toZero_194_; lean_object* v_toAdd_195_; lean_object* v___x_197_; uint8_t v_isShared_198_; uint8_t v_isSharedCheck_215_; 
v___x_188_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_186_);
v___x_189_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_188_);
v_toZero_190_ = lean_ctor_get(v___x_189_, 0);
lean_inc(v_toZero_190_);
v_toAdd_191_ = lean_ctor_get(v___x_189_, 1);
lean_inc(v_toAdd_191_);
lean_dec_ref(v___x_189_);
v___x_192_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_187_);
v___x_193_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_192_);
v_toZero_194_ = lean_ctor_get(v___x_193_, 0);
v_toAdd_195_ = lean_ctor_get(v___x_193_, 1);
v_isSharedCheck_215_ = !lean_is_exclusive(v___x_193_);
if (v_isSharedCheck_215_ == 0)
{
v___x_197_ = v___x_193_;
v_isShared_198_ = v_isSharedCheck_215_;
goto v_resetjp_196_;
}
else
{
lean_inc(v_toAdd_195_);
lean_inc(v_toZero_194_);
lean_dec(v___x_193_);
v___x_197_ = lean_box(0);
v_isShared_198_ = v_isSharedCheck_215_;
goto v_resetjp_196_;
}
v_resetjp_196_:
{
lean_object* v_toNSMul_199_; lean_object* v_toNSMul_200_; lean_object* v___x_202_; uint8_t v_isShared_203_; uint8_t v_isSharedCheck_212_; 
v_toNSMul_199_ = lean_ctor_get(v_inst_186_, 2);
lean_inc(v_toNSMul_199_);
lean_dec_ref(v_inst_186_);
v_toNSMul_200_ = lean_ctor_get(v_inst_187_, 2);
v_isSharedCheck_212_ = !lean_is_exclusive(v_inst_187_);
if (v_isSharedCheck_212_ == 0)
{
lean_object* v_unused_213_; lean_object* v_unused_214_; 
v_unused_213_ = lean_ctor_get(v_inst_187_, 1);
lean_dec(v_unused_213_);
v_unused_214_ = lean_ctor_get(v_inst_187_, 0);
lean_dec(v_unused_214_);
v___x_202_ = v_inst_187_;
v_isShared_203_ = v_isSharedCheck_212_;
goto v_resetjp_201_;
}
else
{
lean_inc(v_toNSMul_200_);
lean_dec(v_inst_187_);
v___x_202_ = lean_box(0);
v_isShared_203_ = v_isSharedCheck_212_;
goto v_resetjp_201_;
}
v_resetjp_201_:
{
lean_object* v___x_205_; 
if (v_isShared_198_ == 0)
{
lean_ctor_set(v___x_197_, 1, v_toZero_194_);
lean_ctor_set(v___x_197_, 0, v_toZero_190_);
v___x_205_ = v___x_197_;
goto v_reusejp_204_;
}
else
{
lean_object* v_reuseFailAlloc_211_; 
v_reuseFailAlloc_211_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_211_, 0, v_toZero_190_);
lean_ctor_set(v_reuseFailAlloc_211_, 1, v_toZero_194_);
v___x_205_ = v_reuseFailAlloc_211_;
goto v_reusejp_204_;
}
v_reusejp_204_:
{
lean_object* v___f_206_; lean_object* v___f_207_; lean_object* v___x_209_; 
v___f_206_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_206_, 0, v_toAdd_191_);
lean_closure_set(v___f_206_, 1, v_toAdd_195_);
v___f_207_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instAddMonoid___redArg___lam__0), 4, 2);
lean_closure_set(v___f_207_, 0, v_toNSMul_199_);
lean_closure_set(v___f_207_, 1, v_toNSMul_200_);
if (v_isShared_203_ == 0)
{
lean_ctor_set(v___x_202_, 2, v___f_207_);
lean_ctor_set(v___x_202_, 1, v___f_206_);
lean_ctor_set(v___x_202_, 0, v___x_205_);
v___x_209_ = v___x_202_;
goto v_reusejp_208_;
}
else
{
lean_object* v_reuseFailAlloc_210_; 
v_reuseFailAlloc_210_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_210_, 0, v___x_205_);
lean_ctor_set(v_reuseFailAlloc_210_, 1, v___f_206_);
lean_ctor_set(v_reuseFailAlloc_210_, 2, v___f_207_);
v___x_209_ = v_reuseFailAlloc_210_;
goto v_reusejp_208_;
}
v_reusejp_208_:
{
return v___x_209_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddMonoid(lean_object* v_M_216_, lean_object* v_N_217_, lean_object* v_inst_218_, lean_object* v_inst_219_){
_start:
{
lean_object* v___x_220_; 
v___x_220_ = lp_mathlib_Prod_instAddMonoid___redArg(v_inst_218_, v_inst_219_);
return v___x_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instDivInvMonoid___redArg___lam__0(lean_object* v_toZPow_221_, lean_object* v_toZPow_222_, lean_object* v_z_223_, lean_object* v_a_224_){
_start:
{
lean_object* v_fst_225_; lean_object* v_snd_226_; lean_object* v___x_228_; uint8_t v_isShared_229_; uint8_t v_isSharedCheck_235_; 
v_fst_225_ = lean_ctor_get(v_a_224_, 0);
v_snd_226_ = lean_ctor_get(v_a_224_, 1);
v_isSharedCheck_235_ = !lean_is_exclusive(v_a_224_);
if (v_isSharedCheck_235_ == 0)
{
v___x_228_ = v_a_224_;
v_isShared_229_ = v_isSharedCheck_235_;
goto v_resetjp_227_;
}
else
{
lean_inc(v_snd_226_);
lean_inc(v_fst_225_);
lean_dec(v_a_224_);
v___x_228_ = lean_box(0);
v_isShared_229_ = v_isSharedCheck_235_;
goto v_resetjp_227_;
}
v_resetjp_227_:
{
lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_233_; 
lean_inc(v_z_223_);
v___x_230_ = lean_apply_2(v_toZPow_221_, v_z_223_, v_fst_225_);
v___x_231_ = lean_apply_2(v_toZPow_222_, v_z_223_, v_snd_226_);
if (v_isShared_229_ == 0)
{
lean_ctor_set(v___x_228_, 1, v___x_231_);
lean_ctor_set(v___x_228_, 0, v___x_230_);
v___x_233_ = v___x_228_;
goto v_reusejp_232_;
}
else
{
lean_object* v_reuseFailAlloc_234_; 
v_reuseFailAlloc_234_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_234_, 0, v___x_230_);
lean_ctor_set(v_reuseFailAlloc_234_, 1, v___x_231_);
v___x_233_ = v_reuseFailAlloc_234_;
goto v_reusejp_232_;
}
v_reusejp_232_:
{
return v___x_233_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instDivInvMonoid___redArg(lean_object* v_inst_236_, lean_object* v_inst_237_){
_start:
{
lean_object* v_toMonoid_238_; lean_object* v_toInv_239_; lean_object* v_toDiv_240_; lean_object* v_toZPow_241_; lean_object* v_toMonoid_242_; lean_object* v_toInv_243_; lean_object* v_toDiv_244_; lean_object* v_toZPow_245_; lean_object* v___x_247_; uint8_t v_isShared_248_; uint8_t v_isSharedCheck_256_; 
v_toMonoid_238_ = lean_ctor_get(v_inst_236_, 0);
lean_inc_ref(v_toMonoid_238_);
v_toInv_239_ = lean_ctor_get(v_inst_236_, 1);
lean_inc(v_toInv_239_);
v_toDiv_240_ = lean_ctor_get(v_inst_236_, 2);
lean_inc(v_toDiv_240_);
v_toZPow_241_ = lean_ctor_get(v_inst_236_, 3);
lean_inc(v_toZPow_241_);
lean_dec_ref(v_inst_236_);
v_toMonoid_242_ = lean_ctor_get(v_inst_237_, 0);
v_toInv_243_ = lean_ctor_get(v_inst_237_, 1);
v_toDiv_244_ = lean_ctor_get(v_inst_237_, 2);
v_toZPow_245_ = lean_ctor_get(v_inst_237_, 3);
v_isSharedCheck_256_ = !lean_is_exclusive(v_inst_237_);
if (v_isSharedCheck_256_ == 0)
{
v___x_247_ = v_inst_237_;
v_isShared_248_ = v_isSharedCheck_256_;
goto v_resetjp_246_;
}
else
{
lean_inc(v_toZPow_245_);
lean_inc(v_toDiv_244_);
lean_inc(v_toInv_243_);
lean_inc(v_toMonoid_242_);
lean_dec(v_inst_237_);
v___x_247_ = lean_box(0);
v_isShared_248_ = v_isSharedCheck_256_;
goto v_resetjp_246_;
}
v_resetjp_246_:
{
lean_object* v___f_249_; lean_object* v___x_250_; lean_object* v___f_251_; lean_object* v___f_252_; lean_object* v___x_254_; 
v___f_249_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instDivInvMonoid___redArg___lam__0), 4, 2);
lean_closure_set(v___f_249_, 0, v_toZPow_241_);
lean_closure_set(v___f_249_, 1, v_toZPow_245_);
v___x_250_ = lp_mathlib_Prod_instMonoid___redArg(v_toMonoid_238_, v_toMonoid_242_);
v___f_251_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instInv___redArg___lam__0), 3, 2);
lean_closure_set(v___f_251_, 0, v_toInv_239_);
lean_closure_set(v___f_251_, 1, v_toInv_243_);
v___f_252_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_252_, 0, v_toDiv_240_);
lean_closure_set(v___f_252_, 1, v_toDiv_244_);
if (v_isShared_248_ == 0)
{
lean_ctor_set(v___x_247_, 3, v___f_249_);
lean_ctor_set(v___x_247_, 2, v___f_252_);
lean_ctor_set(v___x_247_, 1, v___f_251_);
lean_ctor_set(v___x_247_, 0, v___x_250_);
v___x_254_ = v___x_247_;
goto v_reusejp_253_;
}
else
{
lean_object* v_reuseFailAlloc_255_; 
v_reuseFailAlloc_255_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_255_, 0, v___x_250_);
lean_ctor_set(v_reuseFailAlloc_255_, 1, v___f_251_);
lean_ctor_set(v_reuseFailAlloc_255_, 2, v___f_252_);
lean_ctor_set(v_reuseFailAlloc_255_, 3, v___f_249_);
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
LEAN_EXPORT lean_object* lp_mathlib_Prod_instDivInvMonoid(lean_object* v_G_257_, lean_object* v_H_258_, lean_object* v_inst_259_, lean_object* v_inst_260_){
_start:
{
lean_object* v___x_261_; 
v___x_261_ = lp_mathlib_Prod_instDivInvMonoid___redArg(v_inst_259_, v_inst_260_);
return v___x_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_subNegMonoid___redArg___lam__0(lean_object* v_toZSMul_262_, lean_object* v_toZSMul_263_, lean_object* v_z_264_, lean_object* v_a_265_){
_start:
{
lean_object* v_fst_266_; lean_object* v_snd_267_; lean_object* v___x_269_; uint8_t v_isShared_270_; uint8_t v_isSharedCheck_276_; 
v_fst_266_ = lean_ctor_get(v_a_265_, 0);
v_snd_267_ = lean_ctor_get(v_a_265_, 1);
v_isSharedCheck_276_ = !lean_is_exclusive(v_a_265_);
if (v_isSharedCheck_276_ == 0)
{
v___x_269_ = v_a_265_;
v_isShared_270_ = v_isSharedCheck_276_;
goto v_resetjp_268_;
}
else
{
lean_inc(v_snd_267_);
lean_inc(v_fst_266_);
lean_dec(v_a_265_);
v___x_269_ = lean_box(0);
v_isShared_270_ = v_isSharedCheck_276_;
goto v_resetjp_268_;
}
v_resetjp_268_:
{
lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_274_; 
lean_inc(v_z_264_);
v___x_271_ = lean_apply_2(v_toZSMul_262_, v_z_264_, v_fst_266_);
v___x_272_ = lean_apply_2(v_toZSMul_263_, v_z_264_, v_snd_267_);
if (v_isShared_270_ == 0)
{
lean_ctor_set(v___x_269_, 1, v___x_272_);
lean_ctor_set(v___x_269_, 0, v___x_271_);
v___x_274_ = v___x_269_;
goto v_reusejp_273_;
}
else
{
lean_object* v_reuseFailAlloc_275_; 
v_reuseFailAlloc_275_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_275_, 0, v___x_271_);
lean_ctor_set(v_reuseFailAlloc_275_, 1, v___x_272_);
v___x_274_ = v_reuseFailAlloc_275_;
goto v_reusejp_273_;
}
v_reusejp_273_:
{
return v___x_274_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_subNegMonoid___redArg(lean_object* v_inst_277_, lean_object* v_inst_278_){
_start:
{
lean_object* v_toAddMonoid_279_; lean_object* v_toNeg_280_; lean_object* v_toSub_281_; lean_object* v_toZSMul_282_; lean_object* v_toAddMonoid_283_; lean_object* v_toNeg_284_; lean_object* v_toSub_285_; lean_object* v_toZSMul_286_; lean_object* v___x_288_; uint8_t v_isShared_289_; uint8_t v_isSharedCheck_297_; 
v_toAddMonoid_279_ = lean_ctor_get(v_inst_277_, 0);
lean_inc_ref(v_toAddMonoid_279_);
v_toNeg_280_ = lean_ctor_get(v_inst_277_, 1);
lean_inc(v_toNeg_280_);
v_toSub_281_ = lean_ctor_get(v_inst_277_, 2);
lean_inc(v_toSub_281_);
v_toZSMul_282_ = lean_ctor_get(v_inst_277_, 3);
lean_inc(v_toZSMul_282_);
lean_dec_ref(v_inst_277_);
v_toAddMonoid_283_ = lean_ctor_get(v_inst_278_, 0);
v_toNeg_284_ = lean_ctor_get(v_inst_278_, 1);
v_toSub_285_ = lean_ctor_get(v_inst_278_, 2);
v_toZSMul_286_ = lean_ctor_get(v_inst_278_, 3);
v_isSharedCheck_297_ = !lean_is_exclusive(v_inst_278_);
if (v_isSharedCheck_297_ == 0)
{
v___x_288_ = v_inst_278_;
v_isShared_289_ = v_isSharedCheck_297_;
goto v_resetjp_287_;
}
else
{
lean_inc(v_toZSMul_286_);
lean_inc(v_toSub_285_);
lean_inc(v_toNeg_284_);
lean_inc(v_toAddMonoid_283_);
lean_dec(v_inst_278_);
v___x_288_ = lean_box(0);
v_isShared_289_ = v_isSharedCheck_297_;
goto v_resetjp_287_;
}
v_resetjp_287_:
{
lean_object* v___f_290_; lean_object* v___x_291_; lean_object* v___f_292_; lean_object* v___f_293_; lean_object* v___x_295_; 
v___f_290_ = lean_alloc_closure((void*)(lp_mathlib_Prod_subNegMonoid___redArg___lam__0), 4, 2);
lean_closure_set(v___f_290_, 0, v_toZSMul_282_);
lean_closure_set(v___f_290_, 1, v_toZSMul_286_);
v___x_291_ = lp_mathlib_Prod_instAddMonoid___redArg(v_toAddMonoid_279_, v_toAddMonoid_283_);
v___f_292_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instInv___redArg___lam__0), 3, 2);
lean_closure_set(v___f_292_, 0, v_toNeg_280_);
lean_closure_set(v___f_292_, 1, v_toNeg_284_);
v___f_293_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_293_, 0, v_toSub_281_);
lean_closure_set(v___f_293_, 1, v_toSub_285_);
if (v_isShared_289_ == 0)
{
lean_ctor_set(v___x_288_, 3, v___f_290_);
lean_ctor_set(v___x_288_, 2, v___f_293_);
lean_ctor_set(v___x_288_, 1, v___f_292_);
lean_ctor_set(v___x_288_, 0, v___x_291_);
v___x_295_ = v___x_288_;
goto v_reusejp_294_;
}
else
{
lean_object* v_reuseFailAlloc_296_; 
v_reuseFailAlloc_296_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_296_, 0, v___x_291_);
lean_ctor_set(v_reuseFailAlloc_296_, 1, v___f_292_);
lean_ctor_set(v_reuseFailAlloc_296_, 2, v___f_293_);
lean_ctor_set(v_reuseFailAlloc_296_, 3, v___f_290_);
v___x_295_ = v_reuseFailAlloc_296_;
goto v_reusejp_294_;
}
v_reusejp_294_:
{
return v___x_295_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_subNegMonoid(lean_object* v_G_298_, lean_object* v_H_299_, lean_object* v_inst_300_, lean_object* v_inst_301_){
_start:
{
lean_object* v___x_302_; 
v___x_302_ = lp_mathlib_Prod_subNegMonoid___redArg(v_inst_300_, v_inst_301_);
return v___x_302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instDivisionMonoid___redArg(lean_object* v_inst_303_, lean_object* v_inst_304_){
_start:
{
lean_object* v___x_305_; 
v___x_305_ = lp_mathlib_Prod_instDivInvMonoid___redArg(v_inst_303_, v_inst_304_);
return v___x_305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instDivisionMonoid(lean_object* v_G_306_, lean_object* v_H_307_, lean_object* v_inst_308_, lean_object* v_inst_309_){
_start:
{
lean_object* v___x_310_; 
v___x_310_ = lp_mathlib_Prod_instDivInvMonoid___redArg(v_inst_308_, v_inst_309_);
return v___x_310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSubtractionMonoid___redArg(lean_object* v_inst_311_, lean_object* v_inst_312_){
_start:
{
lean_object* v___x_313_; 
v___x_313_ = lp_mathlib_Prod_subNegMonoid___redArg(v_inst_311_, v_inst_312_);
return v___x_313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSubtractionMonoid(lean_object* v_G_314_, lean_object* v_H_315_, lean_object* v_inst_316_, lean_object* v_inst_317_){
_start:
{
lean_object* v___x_318_; 
v___x_318_ = lp_mathlib_Prod_subNegMonoid___redArg(v_inst_316_, v_inst_317_);
return v___x_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instDivisionCommMonoid___redArg(lean_object* v_inst_319_, lean_object* v_inst_320_){
_start:
{
lean_object* v___x_321_; 
v___x_321_ = lp_mathlib_Prod_instDivInvMonoid___redArg(v_inst_319_, v_inst_320_);
return v___x_321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instDivisionCommMonoid(lean_object* v_G_322_, lean_object* v_H_323_, lean_object* v_inst_324_, lean_object* v_inst_325_){
_start:
{
lean_object* v___x_326_; 
v___x_326_ = lp_mathlib_Prod_instDivInvMonoid___redArg(v_inst_324_, v_inst_325_);
return v___x_326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_SubtractionCommMonoid___redArg(lean_object* v_inst_327_, lean_object* v_inst_328_){
_start:
{
lean_object* v___x_329_; 
v___x_329_ = lp_mathlib_Prod_subNegMonoid___redArg(v_inst_327_, v_inst_328_);
return v___x_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_SubtractionCommMonoid(lean_object* v_G_330_, lean_object* v_H_331_, lean_object* v_inst_332_, lean_object* v_inst_333_){
_start:
{
lean_object* v___x_334_; 
v___x_334_ = lp_mathlib_Prod_subNegMonoid___redArg(v_inst_332_, v_inst_333_);
return v___x_334_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instGroup___redArg(lean_object* v_inst_335_, lean_object* v_inst_336_){
_start:
{
lean_object* v___x_337_; 
v___x_337_ = lp_mathlib_Prod_instDivInvMonoid___redArg(v_inst_335_, v_inst_336_);
return v___x_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instGroup(lean_object* v_G_338_, lean_object* v_H_339_, lean_object* v_inst_340_, lean_object* v_inst_341_){
_start:
{
lean_object* v___x_342_; 
v___x_342_ = lp_mathlib_Prod_instDivInvMonoid___redArg(v_inst_340_, v_inst_341_);
return v___x_342_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddGroup___redArg(lean_object* v_inst_343_, lean_object* v_inst_344_){
_start:
{
lean_object* v___x_345_; 
v___x_345_ = lp_mathlib_Prod_subNegMonoid___redArg(v_inst_343_, v_inst_344_);
return v___x_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddGroup(lean_object* v_G_346_, lean_object* v_H_347_, lean_object* v_inst_348_, lean_object* v_inst_349_){
_start:
{
lean_object* v___x_350_; 
v___x_350_ = lp_mathlib_Prod_subNegMonoid___redArg(v_inst_348_, v_inst_349_);
return v___x_350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLeftCancelSemigroup___redArg(lean_object* v_inst_351_, lean_object* v_inst_352_){
_start:
{
lean_object* v___f_353_; 
v___f_353_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_353_, 0, v_inst_351_);
lean_closure_set(v___f_353_, 1, v_inst_352_);
return v___f_353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLeftCancelSemigroup(lean_object* v_G_354_, lean_object* v_H_355_, lean_object* v_inst_356_, lean_object* v_inst_357_){
_start:
{
lean_object* v___f_358_; 
v___f_358_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_358_, 0, v_inst_356_);
lean_closure_set(v___f_358_, 1, v_inst_357_);
return v___f_358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddLeftCancelSemigroup___redArg(lean_object* v_inst_359_, lean_object* v_inst_360_){
_start:
{
lean_object* v___f_361_; 
v___f_361_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_361_, 0, v_inst_359_);
lean_closure_set(v___f_361_, 1, v_inst_360_);
return v___f_361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddLeftCancelSemigroup(lean_object* v_G_362_, lean_object* v_H_363_, lean_object* v_inst_364_, lean_object* v_inst_365_){
_start:
{
lean_object* v___f_366_; 
v___f_366_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_366_, 0, v_inst_364_);
lean_closure_set(v___f_366_, 1, v_inst_365_);
return v___f_366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instRightCancelSemigroup___redArg(lean_object* v_inst_367_, lean_object* v_inst_368_){
_start:
{
lean_object* v___f_369_; 
v___f_369_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_369_, 0, v_inst_367_);
lean_closure_set(v___f_369_, 1, v_inst_368_);
return v___f_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instRightCancelSemigroup(lean_object* v_G_370_, lean_object* v_H_371_, lean_object* v_inst_372_, lean_object* v_inst_373_){
_start:
{
lean_object* v___f_374_; 
v___f_374_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_374_, 0, v_inst_372_);
lean_closure_set(v___f_374_, 1, v_inst_373_);
return v___f_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddRightCancelSemigroup___redArg(lean_object* v_inst_375_, lean_object* v_inst_376_){
_start:
{
lean_object* v___f_377_; 
v___f_377_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_377_, 0, v_inst_375_);
lean_closure_set(v___f_377_, 1, v_inst_376_);
return v___f_377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddRightCancelSemigroup(lean_object* v_G_378_, lean_object* v_H_379_, lean_object* v_inst_380_, lean_object* v_inst_381_){
_start:
{
lean_object* v___f_382_; 
v___f_382_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instMul___redArg___lam__0), 4, 2);
lean_closure_set(v___f_382_, 0, v_inst_380_);
lean_closure_set(v___f_382_, 1, v_inst_381_);
return v___f_382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLeftCancelMonoid___redArg(lean_object* v_inst_383_, lean_object* v_inst_384_){
_start:
{
lean_object* v___x_385_; 
v___x_385_ = lp_mathlib_Prod_instMonoid___redArg(v_inst_383_, v_inst_384_);
return v___x_385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLeftCancelMonoid(lean_object* v_M_386_, lean_object* v_N_387_, lean_object* v_inst_388_, lean_object* v_inst_389_){
_start:
{
lean_object* v___x_390_; 
v___x_390_ = lp_mathlib_Prod_instMonoid___redArg(v_inst_388_, v_inst_389_);
return v___x_390_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddLeftCancelMonoid___redArg(lean_object* v_inst_391_, lean_object* v_inst_392_){
_start:
{
lean_object* v___x_393_; 
v___x_393_ = lp_mathlib_Prod_instAddMonoid___redArg(v_inst_391_, v_inst_392_);
return v___x_393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddLeftCancelMonoid(lean_object* v_M_394_, lean_object* v_N_395_, lean_object* v_inst_396_, lean_object* v_inst_397_){
_start:
{
lean_object* v___x_398_; 
v___x_398_ = lp_mathlib_Prod_instAddMonoid___redArg(v_inst_396_, v_inst_397_);
return v___x_398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instRightCancelMonoid___redArg(lean_object* v_inst_399_, lean_object* v_inst_400_){
_start:
{
lean_object* v___x_401_; 
v___x_401_ = lp_mathlib_Prod_instMonoid___redArg(v_inst_399_, v_inst_400_);
return v___x_401_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instRightCancelMonoid(lean_object* v_M_402_, lean_object* v_N_403_, lean_object* v_inst_404_, lean_object* v_inst_405_){
_start:
{
lean_object* v___x_406_; 
v___x_406_ = lp_mathlib_Prod_instMonoid___redArg(v_inst_404_, v_inst_405_);
return v___x_406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddRightCancelMonoid___redArg(lean_object* v_inst_407_, lean_object* v_inst_408_){
_start:
{
lean_object* v___x_409_; 
v___x_409_ = lp_mathlib_Prod_instAddMonoid___redArg(v_inst_407_, v_inst_408_);
return v___x_409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddRightCancelMonoid(lean_object* v_M_410_, lean_object* v_N_411_, lean_object* v_inst_412_, lean_object* v_inst_413_){
_start:
{
lean_object* v___x_414_; 
v___x_414_ = lp_mathlib_Prod_instAddMonoid___redArg(v_inst_412_, v_inst_413_);
return v___x_414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCancelMonoid___redArg(lean_object* v_inst_415_, lean_object* v_inst_416_){
_start:
{
lean_object* v___x_417_; 
v___x_417_ = lp_mathlib_Prod_instMonoid___redArg(v_inst_415_, v_inst_416_);
return v___x_417_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCancelMonoid(lean_object* v_M_418_, lean_object* v_N_419_, lean_object* v_inst_420_, lean_object* v_inst_421_){
_start:
{
lean_object* v___x_422_; 
v___x_422_ = lp_mathlib_Prod_instMonoid___redArg(v_inst_420_, v_inst_421_);
return v___x_422_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddCancelMonoid___redArg(lean_object* v_inst_423_, lean_object* v_inst_424_){
_start:
{
lean_object* v___x_425_; 
v___x_425_ = lp_mathlib_Prod_instAddMonoid___redArg(v_inst_423_, v_inst_424_);
return v___x_425_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddCancelMonoid(lean_object* v_M_426_, lean_object* v_N_427_, lean_object* v_inst_428_, lean_object* v_inst_429_){
_start:
{
lean_object* v___x_430_; 
v___x_430_ = lp_mathlib_Prod_instAddMonoid___redArg(v_inst_428_, v_inst_429_);
return v___x_430_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCommMonoid___redArg(lean_object* v_inst_431_, lean_object* v_inst_432_){
_start:
{
lean_object* v___x_433_; 
v___x_433_ = lp_mathlib_Prod_instMonoid___redArg(v_inst_431_, v_inst_432_);
return v___x_433_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCommMonoid(lean_object* v_M_434_, lean_object* v_N_435_, lean_object* v_inst_436_, lean_object* v_inst_437_){
_start:
{
lean_object* v___x_438_; 
v___x_438_ = lp_mathlib_Prod_instMonoid___redArg(v_inst_436_, v_inst_437_);
return v___x_438_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddCommMonoid___redArg(lean_object* v_inst_439_, lean_object* v_inst_440_){
_start:
{
lean_object* v___x_441_; 
v___x_441_ = lp_mathlib_Prod_instAddMonoid___redArg(v_inst_439_, v_inst_440_);
return v___x_441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddCommMonoid(lean_object* v_M_442_, lean_object* v_N_443_, lean_object* v_inst_444_, lean_object* v_inst_445_){
_start:
{
lean_object* v___x_446_; 
v___x_446_ = lp_mathlib_Prod_instAddMonoid___redArg(v_inst_444_, v_inst_445_);
return v___x_446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCancelCommMonoid___redArg(lean_object* v_inst_447_, lean_object* v_inst_448_){
_start:
{
lean_object* v___x_449_; 
v___x_449_ = lp_mathlib_Prod_instMonoid___redArg(v_inst_447_, v_inst_448_);
return v___x_449_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCancelCommMonoid(lean_object* v_M_450_, lean_object* v_N_451_, lean_object* v_inst_452_, lean_object* v_inst_453_){
_start:
{
lean_object* v___x_454_; 
v___x_454_ = lp_mathlib_Prod_instMonoid___redArg(v_inst_452_, v_inst_453_);
return v___x_454_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddCancelCommMonoid___redArg(lean_object* v_inst_455_, lean_object* v_inst_456_){
_start:
{
lean_object* v___x_457_; 
v___x_457_ = lp_mathlib_Prod_instAddMonoid___redArg(v_inst_455_, v_inst_456_);
return v___x_457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddCancelCommMonoid(lean_object* v_M_458_, lean_object* v_N_459_, lean_object* v_inst_460_, lean_object* v_inst_461_){
_start:
{
lean_object* v___x_462_; 
v___x_462_ = lp_mathlib_Prod_instAddMonoid___redArg(v_inst_460_, v_inst_461_);
return v___x_462_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCommGroup___redArg(lean_object* v_inst_463_, lean_object* v_inst_464_){
_start:
{
lean_object* v___x_465_; 
v___x_465_ = lp_mathlib_Prod_instDivInvMonoid___redArg(v_inst_463_, v_inst_464_);
return v___x_465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCommGroup(lean_object* v_G_466_, lean_object* v_H_467_, lean_object* v_inst_468_, lean_object* v_inst_469_){
_start:
{
lean_object* v___x_470_; 
v___x_470_ = lp_mathlib_Prod_instDivInvMonoid___redArg(v_inst_468_, v_inst_469_);
return v___x_470_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddCommGroup___redArg(lean_object* v_inst_471_, lean_object* v_inst_472_){
_start:
{
lean_object* v___x_473_; 
v___x_473_ = lp_mathlib_Prod_subNegMonoid___redArg(v_inst_471_, v_inst_472_);
return v___x_473_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instAddCommGroup(lean_object* v_G_474_, lean_object* v_H_475_, lean_object* v_inst_476_, lean_object* v_inst_477_){
_start:
{
lean_object* v___x_478_; 
v___x_478_ = lp_mathlib_Prod_subNegMonoid___redArg(v_inst_476_, v_inst_477_);
return v___x_478_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_fst___lam__0(lean_object* v_self_479_){
_start:
{
lean_object* v_fst_480_; 
v_fst_480_ = lean_ctor_get(v_self_479_, 0);
lean_inc(v_fst_480_);
return v_fst_480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_fst___lam__0___boxed(lean_object* v_self_481_){
_start:
{
lean_object* v_res_482_; 
v_res_482_ = lp_mathlib_MulHom_fst___lam__0(v_self_481_);
lean_dec_ref(v_self_481_);
return v_res_482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_fst(lean_object* v_M_484_, lean_object* v_N_485_, lean_object* v_inst_486_, lean_object* v_inst_487_){
_start:
{
lean_object* v___f_488_; 
v___f_488_ = ((lean_object*)(lp_mathlib_MulHom_fst___closed__0));
return v___f_488_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_fst___boxed(lean_object* v_M_489_, lean_object* v_N_490_, lean_object* v_inst_491_, lean_object* v_inst_492_){
_start:
{
lean_object* v_res_493_; 
v_res_493_ = lp_mathlib_MulHom_fst(v_M_489_, v_N_490_, v_inst_491_, v_inst_492_);
lean_dec(v_inst_492_);
lean_dec(v_inst_491_);
return v_res_493_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_fst(lean_object* v_M_494_, lean_object* v_N_495_, lean_object* v_inst_496_, lean_object* v_inst_497_){
_start:
{
lean_object* v___f_498_; 
v___f_498_ = ((lean_object*)(lp_mathlib_MulHom_fst___closed__0));
return v___f_498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_fst___boxed(lean_object* v_M_499_, lean_object* v_N_500_, lean_object* v_inst_501_, lean_object* v_inst_502_){
_start:
{
lean_object* v_res_503_; 
v_res_503_ = lp_mathlib_AddHom_fst(v_M_499_, v_N_500_, v_inst_501_, v_inst_502_);
lean_dec(v_inst_502_);
lean_dec(v_inst_501_);
return v_res_503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_snd___lam__0(lean_object* v_self_504_){
_start:
{
lean_object* v_snd_505_; 
v_snd_505_ = lean_ctor_get(v_self_504_, 1);
lean_inc(v_snd_505_);
return v_snd_505_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_snd___lam__0___boxed(lean_object* v_self_506_){
_start:
{
lean_object* v_res_507_; 
v_res_507_ = lp_mathlib_MulHom_snd___lam__0(v_self_506_);
lean_dec_ref(v_self_506_);
return v_res_507_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_snd(lean_object* v_M_509_, lean_object* v_N_510_, lean_object* v_inst_511_, lean_object* v_inst_512_){
_start:
{
lean_object* v___f_513_; 
v___f_513_ = ((lean_object*)(lp_mathlib_MulHom_snd___closed__0));
return v___f_513_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_snd___boxed(lean_object* v_M_514_, lean_object* v_N_515_, lean_object* v_inst_516_, lean_object* v_inst_517_){
_start:
{
lean_object* v_res_518_; 
v_res_518_ = lp_mathlib_MulHom_snd(v_M_514_, v_N_515_, v_inst_516_, v_inst_517_);
lean_dec(v_inst_517_);
lean_dec(v_inst_516_);
return v_res_518_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_snd(lean_object* v_M_519_, lean_object* v_N_520_, lean_object* v_inst_521_, lean_object* v_inst_522_){
_start:
{
lean_object* v___f_523_; 
v___f_523_ = ((lean_object*)(lp_mathlib_MulHom_snd___closed__0));
return v___f_523_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_snd___boxed(lean_object* v_M_524_, lean_object* v_N_525_, lean_object* v_inst_526_, lean_object* v_inst_527_){
_start:
{
lean_object* v_res_528_; 
v_res_528_ = lp_mathlib_AddHom_snd(v_M_524_, v_N_525_, v_inst_526_, v_inst_527_);
lean_dec(v_inst_527_);
lean_dec(v_inst_526_);
return v_res_528_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_prod___redArg___lam__0(lean_object* v_f_529_, lean_object* v___y_530_){
_start:
{
lean_object* v___x_531_; 
v___x_531_ = lean_apply_1(v_f_529_, v___y_530_);
return v___x_531_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_prod___redArg___lam__1(lean_object* v_g_532_, lean_object* v___y_533_){
_start:
{
lean_object* v___x_534_; 
v___x_534_ = lean_apply_1(v_g_532_, v___y_533_);
return v___x_534_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_prod___redArg(lean_object* v_f_535_, lean_object* v_g_536_){
_start:
{
lean_object* v___f_537_; lean_object* v___f_538_; lean_object* v___x_539_; 
v___f_537_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_prod___redArg___lam__0), 2, 1);
lean_closure_set(v___f_537_, 0, v_f_535_);
v___f_538_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_prod___redArg___lam__1), 2, 1);
lean_closure_set(v___f_538_, 0, v_g_536_);
v___x_539_ = lean_alloc_closure((void*)(lp_mathlib_Function_prod), 6, 5);
lean_closure_set(v___x_539_, 0, lean_box(0));
lean_closure_set(v___x_539_, 1, lean_box(0));
lean_closure_set(v___x_539_, 2, lean_box(0));
lean_closure_set(v___x_539_, 3, v___f_537_);
lean_closure_set(v___x_539_, 4, v___f_538_);
return v___x_539_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_prod(lean_object* v_M_540_, lean_object* v_N_541_, lean_object* v_P_542_, lean_object* v_inst_543_, lean_object* v_inst_544_, lean_object* v_inst_545_, lean_object* v_f_546_, lean_object* v_g_547_){
_start:
{
lean_object* v___x_548_; 
v___x_548_ = lp_mathlib_MulHom_prod___redArg(v_f_546_, v_g_547_);
return v___x_548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_prod___boxed(lean_object* v_M_549_, lean_object* v_N_550_, lean_object* v_P_551_, lean_object* v_inst_552_, lean_object* v_inst_553_, lean_object* v_inst_554_, lean_object* v_f_555_, lean_object* v_g_556_){
_start:
{
lean_object* v_res_557_; 
v_res_557_ = lp_mathlib_MulHom_prod(v_M_549_, v_N_550_, v_P_551_, v_inst_552_, v_inst_553_, v_inst_554_, v_f_555_, v_g_556_);
lean_dec(v_inst_554_);
lean_dec(v_inst_553_);
lean_dec(v_inst_552_);
return v_res_557_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_prod___redArg(lean_object* v_f_558_, lean_object* v_g_559_){
_start:
{
lean_object* v___f_560_; lean_object* v___f_561_; lean_object* v___x_562_; 
v___f_560_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_prod___redArg___lam__0), 2, 1);
lean_closure_set(v___f_560_, 0, v_f_558_);
v___f_561_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_prod___redArg___lam__1), 2, 1);
lean_closure_set(v___f_561_, 0, v_g_559_);
v___x_562_ = lean_alloc_closure((void*)(lp_mathlib_Function_prod), 6, 5);
lean_closure_set(v___x_562_, 0, lean_box(0));
lean_closure_set(v___x_562_, 1, lean_box(0));
lean_closure_set(v___x_562_, 2, lean_box(0));
lean_closure_set(v___x_562_, 3, v___f_560_);
lean_closure_set(v___x_562_, 4, v___f_561_);
return v___x_562_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_prod(lean_object* v_M_563_, lean_object* v_N_564_, lean_object* v_P_565_, lean_object* v_inst_566_, lean_object* v_inst_567_, lean_object* v_inst_568_, lean_object* v_f_569_, lean_object* v_g_570_){
_start:
{
lean_object* v___x_571_; 
v___x_571_ = lp_mathlib_AddHom_prod___redArg(v_f_569_, v_g_570_);
return v___x_571_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_prod___boxed(lean_object* v_M_572_, lean_object* v_N_573_, lean_object* v_P_574_, lean_object* v_inst_575_, lean_object* v_inst_576_, lean_object* v_inst_577_, lean_object* v_f_578_, lean_object* v_g_579_){
_start:
{
lean_object* v_res_580_; 
v_res_580_ = lp_mathlib_AddHom_prod(v_M_572_, v_N_573_, v_P_574_, v_inst_575_, v_inst_576_, v_inst_577_, v_f_578_, v_g_579_);
lean_dec(v_inst_577_);
lean_dec(v_inst_576_);
lean_dec(v_inst_575_);
return v_res_580_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_prodMap___redArg(lean_object* v_f_581_, lean_object* v_g_582_){
_start:
{
lean_object* v___f_583_; lean_object* v___f_584_; lean_object* v___f_585_; lean_object* v___f_586_; lean_object* v___x_587_; 
v___f_583_ = ((lean_object*)(lp_mathlib_MulHom_fst___closed__0));
v___f_584_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_584_, 0, v___f_583_);
lean_closure_set(v___f_584_, 1, v_f_581_);
v___f_585_ = ((lean_object*)(lp_mathlib_MulHom_snd___closed__0));
v___f_586_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_586_, 0, v___f_585_);
lean_closure_set(v___f_586_, 1, v_g_582_);
v___x_587_ = lp_mathlib_MulHom_prod___redArg(v___f_584_, v___f_586_);
return v___x_587_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_prodMap(lean_object* v_M_588_, lean_object* v_N_589_, lean_object* v_M_x27_590_, lean_object* v_N_x27_591_, lean_object* v_inst_592_, lean_object* v_inst_593_, lean_object* v_inst_594_, lean_object* v_inst_595_, lean_object* v_f_596_, lean_object* v_g_597_){
_start:
{
lean_object* v___x_598_; 
v___x_598_ = lp_mathlib_MulHom_prodMap___redArg(v_f_596_, v_g_597_);
return v___x_598_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_prodMap___boxed(lean_object* v_M_599_, lean_object* v_N_600_, lean_object* v_M_x27_601_, lean_object* v_N_x27_602_, lean_object* v_inst_603_, lean_object* v_inst_604_, lean_object* v_inst_605_, lean_object* v_inst_606_, lean_object* v_f_607_, lean_object* v_g_608_){
_start:
{
lean_object* v_res_609_; 
v_res_609_ = lp_mathlib_MulHom_prodMap(v_M_599_, v_N_600_, v_M_x27_601_, v_N_x27_602_, v_inst_603_, v_inst_604_, v_inst_605_, v_inst_606_, v_f_607_, v_g_608_);
lean_dec(v_inst_606_);
lean_dec(v_inst_605_);
lean_dec(v_inst_604_);
lean_dec(v_inst_603_);
return v_res_609_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_prodMap___redArg(lean_object* v_f_610_, lean_object* v_g_611_){
_start:
{
lean_object* v___f_612_; lean_object* v___f_613_; lean_object* v___f_614_; lean_object* v___f_615_; lean_object* v___x_616_; 
v___f_612_ = ((lean_object*)(lp_mathlib_MulHom_fst___closed__0));
v___f_613_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_613_, 0, v___f_612_);
lean_closure_set(v___f_613_, 1, v_f_610_);
v___f_614_ = ((lean_object*)(lp_mathlib_MulHom_snd___closed__0));
v___f_615_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_615_, 0, v___f_614_);
lean_closure_set(v___f_615_, 1, v_g_611_);
v___x_616_ = lp_mathlib_AddHom_prod___redArg(v___f_613_, v___f_615_);
return v___x_616_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_prodMap(lean_object* v_M_617_, lean_object* v_N_618_, lean_object* v_M_x27_619_, lean_object* v_N_x27_620_, lean_object* v_inst_621_, lean_object* v_inst_622_, lean_object* v_inst_623_, lean_object* v_inst_624_, lean_object* v_f_625_, lean_object* v_g_626_){
_start:
{
lean_object* v___x_627_; 
v___x_627_ = lp_mathlib_AddHom_prodMap___redArg(v_f_625_, v_g_626_);
return v___x_627_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_prodMap___boxed(lean_object* v_M_628_, lean_object* v_N_629_, lean_object* v_M_x27_630_, lean_object* v_N_x27_631_, lean_object* v_inst_632_, lean_object* v_inst_633_, lean_object* v_inst_634_, lean_object* v_inst_635_, lean_object* v_f_636_, lean_object* v_g_637_){
_start:
{
lean_object* v_res_638_; 
v_res_638_ = lp_mathlib_AddHom_prodMap(v_M_628_, v_N_629_, v_M_x27_630_, v_N_x27_631_, v_inst_632_, v_inst_633_, v_inst_634_, v_inst_635_, v_f_636_, v_g_637_);
lean_dec(v_inst_635_);
lean_dec(v_inst_634_);
lean_dec(v_inst_633_);
lean_dec(v_inst_632_);
return v_res_638_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_coprod___redArg___lam__0(lean_object* v___f_639_, lean_object* v_f_640_, lean_object* v___f_641_, lean_object* v_g_642_, lean_object* v_inst_643_, lean_object* v_m_644_){
_start:
{
lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v___x_647_; 
lean_inc_ref(v_m_644_);
v___x_645_ = lp_mathlib_OneHom_comp___redArg___lam__0(v___f_639_, v_f_640_, v_m_644_);
v___x_646_ = lp_mathlib_OneHom_comp___redArg___lam__0(v___f_641_, v_g_642_, v_m_644_);
v___x_647_ = lean_apply_2(v_inst_643_, v___x_645_, v___x_646_);
return v___x_647_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_coprod___redArg(lean_object* v_inst_648_, lean_object* v_f_649_, lean_object* v_g_650_){
_start:
{
lean_object* v___f_651_; lean_object* v___f_652_; lean_object* v___f_653_; 
v___f_651_ = ((lean_object*)(lp_mathlib_MulHom_fst___closed__0));
v___f_652_ = ((lean_object*)(lp_mathlib_MulHom_snd___closed__0));
v___f_653_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_coprod___redArg___lam__0), 6, 5);
lean_closure_set(v___f_653_, 0, v___f_651_);
lean_closure_set(v___f_653_, 1, v_f_649_);
lean_closure_set(v___f_653_, 2, v___f_652_);
lean_closure_set(v___f_653_, 3, v_g_650_);
lean_closure_set(v___f_653_, 4, v_inst_648_);
return v___f_653_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_coprod(lean_object* v_M_654_, lean_object* v_N_655_, lean_object* v_P_656_, lean_object* v_inst_657_, lean_object* v_inst_658_, lean_object* v_inst_659_, lean_object* v_f_660_, lean_object* v_g_661_){
_start:
{
lean_object* v___x_662_; 
v___x_662_ = lp_mathlib_MulHom_coprod___redArg(v_inst_659_, v_f_660_, v_g_661_);
return v___x_662_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulHom_coprod___boxed(lean_object* v_M_663_, lean_object* v_N_664_, lean_object* v_P_665_, lean_object* v_inst_666_, lean_object* v_inst_667_, lean_object* v_inst_668_, lean_object* v_f_669_, lean_object* v_g_670_){
_start:
{
lean_object* v_res_671_; 
v_res_671_ = lp_mathlib_MulHom_coprod(v_M_663_, v_N_664_, v_P_665_, v_inst_666_, v_inst_667_, v_inst_668_, v_f_669_, v_g_670_);
lean_dec(v_inst_667_);
lean_dec(v_inst_666_);
return v_res_671_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_coprod___redArg(lean_object* v_inst_672_, lean_object* v_f_673_, lean_object* v_g_674_){
_start:
{
lean_object* v___f_675_; lean_object* v___f_676_; lean_object* v___f_677_; 
v___f_675_ = ((lean_object*)(lp_mathlib_MulHom_fst___closed__0));
v___f_676_ = ((lean_object*)(lp_mathlib_MulHom_snd___closed__0));
v___f_677_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_coprod___redArg___lam__0), 6, 5);
lean_closure_set(v___f_677_, 0, v___f_675_);
lean_closure_set(v___f_677_, 1, v_f_673_);
lean_closure_set(v___f_677_, 2, v___f_676_);
lean_closure_set(v___f_677_, 3, v_g_674_);
lean_closure_set(v___f_677_, 4, v_inst_672_);
return v___f_677_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_coprod(lean_object* v_M_678_, lean_object* v_N_679_, lean_object* v_P_680_, lean_object* v_inst_681_, lean_object* v_inst_682_, lean_object* v_inst_683_, lean_object* v_f_684_, lean_object* v_g_685_){
_start:
{
lean_object* v___x_686_; 
v___x_686_ = lp_mathlib_AddHom_coprod___redArg(v_inst_683_, v_f_684_, v_g_685_);
return v___x_686_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_coprod___boxed(lean_object* v_M_687_, lean_object* v_N_688_, lean_object* v_P_689_, lean_object* v_inst_690_, lean_object* v_inst_691_, lean_object* v_inst_692_, lean_object* v_f_693_, lean_object* v_g_694_){
_start:
{
lean_object* v_res_695_; 
v_res_695_ = lp_mathlib_AddHom_coprod(v_M_687_, v_N_688_, v_P_689_, v_inst_690_, v_inst_691_, v_inst_692_, v_f_693_, v_g_694_);
lean_dec(v_inst_691_);
lean_dec(v_inst_690_);
return v_res_695_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fst(lean_object* v_M_696_, lean_object* v_N_697_, lean_object* v_inst_698_, lean_object* v_inst_699_){
_start:
{
lean_object* v___f_700_; 
v___f_700_ = ((lean_object*)(lp_mathlib_MulHom_fst___closed__0));
return v___f_700_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_fst___boxed(lean_object* v_M_701_, lean_object* v_N_702_, lean_object* v_inst_703_, lean_object* v_inst_704_){
_start:
{
lean_object* v_res_705_; 
v_res_705_ = lp_mathlib_MonoidHom_fst(v_M_701_, v_N_702_, v_inst_703_, v_inst_704_);
lean_dec_ref(v_inst_704_);
lean_dec_ref(v_inst_703_);
return v_res_705_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fst(lean_object* v_M_706_, lean_object* v_N_707_, lean_object* v_inst_708_, lean_object* v_inst_709_){
_start:
{
lean_object* v___f_710_; 
v___f_710_ = ((lean_object*)(lp_mathlib_MulHom_fst___closed__0));
return v___f_710_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_fst___boxed(lean_object* v_M_711_, lean_object* v_N_712_, lean_object* v_inst_713_, lean_object* v_inst_714_){
_start:
{
lean_object* v_res_715_; 
v_res_715_ = lp_mathlib_AddMonoidHom_fst(v_M_711_, v_N_712_, v_inst_713_, v_inst_714_);
lean_dec_ref(v_inst_714_);
lean_dec_ref(v_inst_713_);
return v_res_715_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_snd(lean_object* v_M_716_, lean_object* v_N_717_, lean_object* v_inst_718_, lean_object* v_inst_719_){
_start:
{
lean_object* v___f_720_; 
v___f_720_ = ((lean_object*)(lp_mathlib_MulHom_snd___closed__0));
return v___f_720_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_snd___boxed(lean_object* v_M_721_, lean_object* v_N_722_, lean_object* v_inst_723_, lean_object* v_inst_724_){
_start:
{
lean_object* v_res_725_; 
v_res_725_ = lp_mathlib_MonoidHom_snd(v_M_721_, v_N_722_, v_inst_723_, v_inst_724_);
lean_dec_ref(v_inst_724_);
lean_dec_ref(v_inst_723_);
return v_res_725_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_snd(lean_object* v_M_726_, lean_object* v_N_727_, lean_object* v_inst_728_, lean_object* v_inst_729_){
_start:
{
lean_object* v___f_730_; 
v___f_730_ = ((lean_object*)(lp_mathlib_MulHom_snd___closed__0));
return v___f_730_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_snd___boxed(lean_object* v_M_731_, lean_object* v_N_732_, lean_object* v_inst_733_, lean_object* v_inst_734_){
_start:
{
lean_object* v_res_735_; 
v_res_735_ = lp_mathlib_AddMonoidHom_snd(v_M_731_, v_N_732_, v_inst_733_, v_inst_734_);
lean_dec_ref(v_inst_734_);
lean_dec_ref(v_inst_733_);
return v_res_735_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inl___redArg___lam__0(lean_object* v_toOne_736_, lean_object* v_x_737_){
_start:
{
lean_object* v___x_738_; 
v___x_738_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_738_, 0, v_x_737_);
lean_ctor_set(v___x_738_, 1, v_toOne_736_);
return v___x_738_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inl___redArg(lean_object* v_inst_739_){
_start:
{
lean_object* v___x_740_; lean_object* v_toOne_741_; lean_object* v___f_742_; 
v___x_740_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_739_);
v_toOne_741_ = lean_ctor_get(v___x_740_, 0);
lean_inc(v_toOne_741_);
lean_dec_ref(v___x_740_);
v___f_742_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_inl___redArg___lam__0), 2, 1);
lean_closure_set(v___f_742_, 0, v_toOne_741_);
return v___f_742_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inl(lean_object* v_M_743_, lean_object* v_N_744_, lean_object* v_inst_745_, lean_object* v_inst_746_){
_start:
{
lean_object* v___x_747_; 
v___x_747_ = lp_mathlib_MonoidHom_inl___redArg(v_inst_746_);
return v___x_747_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inl___boxed(lean_object* v_M_748_, lean_object* v_N_749_, lean_object* v_inst_750_, lean_object* v_inst_751_){
_start:
{
lean_object* v_res_752_; 
v_res_752_ = lp_mathlib_MonoidHom_inl(v_M_748_, v_N_749_, v_inst_750_, v_inst_751_);
lean_dec_ref(v_inst_750_);
return v_res_752_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inl___redArg___lam__0(lean_object* v_toZero_753_, lean_object* v_x_754_){
_start:
{
lean_object* v___x_755_; 
v___x_755_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_755_, 0, v_x_754_);
lean_ctor_set(v___x_755_, 1, v_toZero_753_);
return v___x_755_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inl___redArg(lean_object* v_inst_756_){
_start:
{
lean_object* v___x_757_; lean_object* v_toZero_758_; lean_object* v___f_759_; 
v___x_757_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_756_);
v_toZero_758_ = lean_ctor_get(v___x_757_, 0);
lean_inc(v_toZero_758_);
lean_dec_ref(v___x_757_);
v___f_759_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_inl___redArg___lam__0), 2, 1);
lean_closure_set(v___f_759_, 0, v_toZero_758_);
return v___f_759_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inl(lean_object* v_M_760_, lean_object* v_N_761_, lean_object* v_inst_762_, lean_object* v_inst_763_){
_start:
{
lean_object* v___x_764_; 
v___x_764_ = lp_mathlib_AddMonoidHom_inl___redArg(v_inst_763_);
return v___x_764_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inl___boxed(lean_object* v_M_765_, lean_object* v_N_766_, lean_object* v_inst_767_, lean_object* v_inst_768_){
_start:
{
lean_object* v_res_769_; 
v_res_769_ = lp_mathlib_AddMonoidHom_inl(v_M_765_, v_N_766_, v_inst_767_, v_inst_768_);
lean_dec_ref(v_inst_767_);
return v_res_769_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inr___redArg___lam__0(lean_object* v_toOne_770_, lean_object* v_y_771_){
_start:
{
lean_object* v___x_772_; 
v___x_772_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_772_, 0, v_toOne_770_);
lean_ctor_set(v___x_772_, 1, v_y_771_);
return v___x_772_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inr___redArg(lean_object* v_inst_773_){
_start:
{
lean_object* v___x_774_; lean_object* v_toOne_775_; lean_object* v___f_776_; 
v___x_774_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_773_);
v_toOne_775_ = lean_ctor_get(v___x_774_, 0);
lean_inc(v_toOne_775_);
lean_dec_ref(v___x_774_);
v___f_776_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_inr___redArg___lam__0), 2, 1);
lean_closure_set(v___f_776_, 0, v_toOne_775_);
return v___f_776_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inr(lean_object* v_M_777_, lean_object* v_N_778_, lean_object* v_inst_779_, lean_object* v_inst_780_){
_start:
{
lean_object* v___x_781_; 
v___x_781_ = lp_mathlib_MonoidHom_inr___redArg(v_inst_779_);
return v___x_781_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_inr___boxed(lean_object* v_M_782_, lean_object* v_N_783_, lean_object* v_inst_784_, lean_object* v_inst_785_){
_start:
{
lean_object* v_res_786_; 
v_res_786_ = lp_mathlib_MonoidHom_inr(v_M_782_, v_N_783_, v_inst_784_, v_inst_785_);
lean_dec_ref(v_inst_785_);
return v_res_786_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inr___redArg___lam__0(lean_object* v_toZero_787_, lean_object* v_y_788_){
_start:
{
lean_object* v___x_789_; 
v___x_789_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_789_, 0, v_toZero_787_);
lean_ctor_set(v___x_789_, 1, v_y_788_);
return v___x_789_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inr___redArg(lean_object* v_inst_790_){
_start:
{
lean_object* v___x_791_; lean_object* v_toZero_792_; lean_object* v___f_793_; 
v___x_791_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_790_);
v_toZero_792_ = lean_ctor_get(v___x_791_, 0);
lean_inc(v_toZero_792_);
lean_dec_ref(v___x_791_);
v___f_793_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_inr___redArg___lam__0), 2, 1);
lean_closure_set(v___f_793_, 0, v_toZero_792_);
return v___f_793_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inr(lean_object* v_M_794_, lean_object* v_N_795_, lean_object* v_inst_796_, lean_object* v_inst_797_){
_start:
{
lean_object* v___x_798_; 
v___x_798_ = lp_mathlib_AddMonoidHom_inr___redArg(v_inst_796_);
return v___x_798_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_inr___boxed(lean_object* v_M_799_, lean_object* v_N_800_, lean_object* v_inst_801_, lean_object* v_inst_802_){
_start:
{
lean_object* v_res_803_; 
v_res_803_ = lp_mathlib_AddMonoidHom_inr(v_M_799_, v_N_800_, v_inst_801_, v_inst_802_);
lean_dec_ref(v_inst_802_);
return v_res_803_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_prod___redArg(lean_object* v_f_804_, lean_object* v_g_805_){
_start:
{
lean_object* v___f_806_; lean_object* v___f_807_; lean_object* v___x_808_; 
v___f_806_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_prod___redArg___lam__0), 2, 1);
lean_closure_set(v___f_806_, 0, v_f_804_);
v___f_807_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_prod___redArg___lam__1), 2, 1);
lean_closure_set(v___f_807_, 0, v_g_805_);
v___x_808_ = lean_alloc_closure((void*)(lp_mathlib_Function_prod), 6, 5);
lean_closure_set(v___x_808_, 0, lean_box(0));
lean_closure_set(v___x_808_, 1, lean_box(0));
lean_closure_set(v___x_808_, 2, lean_box(0));
lean_closure_set(v___x_808_, 3, v___f_806_);
lean_closure_set(v___x_808_, 4, v___f_807_);
return v___x_808_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_prod(lean_object* v_M_809_, lean_object* v_N_810_, lean_object* v_P_811_, lean_object* v_inst_812_, lean_object* v_inst_813_, lean_object* v_inst_814_, lean_object* v_f_815_, lean_object* v_g_816_){
_start:
{
lean_object* v___x_817_; 
v___x_817_ = lp_mathlib_MonoidHom_prod___redArg(v_f_815_, v_g_816_);
return v___x_817_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_prod___boxed(lean_object* v_M_818_, lean_object* v_N_819_, lean_object* v_P_820_, lean_object* v_inst_821_, lean_object* v_inst_822_, lean_object* v_inst_823_, lean_object* v_f_824_, lean_object* v_g_825_){
_start:
{
lean_object* v_res_826_; 
v_res_826_ = lp_mathlib_MonoidHom_prod(v_M_818_, v_N_819_, v_P_820_, v_inst_821_, v_inst_822_, v_inst_823_, v_f_824_, v_g_825_);
lean_dec_ref(v_inst_823_);
lean_dec_ref(v_inst_822_);
lean_dec_ref(v_inst_821_);
return v_res_826_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_prod___redArg(lean_object* v_f_827_, lean_object* v_g_828_){
_start:
{
lean_object* v___f_829_; lean_object* v___f_830_; lean_object* v___x_831_; 
v___f_829_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_prod___redArg___lam__0), 2, 1);
lean_closure_set(v___f_829_, 0, v_f_827_);
v___f_830_ = lean_alloc_closure((void*)(lp_mathlib_MulHom_prod___redArg___lam__1), 2, 1);
lean_closure_set(v___f_830_, 0, v_g_828_);
v___x_831_ = lean_alloc_closure((void*)(lp_mathlib_Function_prod), 6, 5);
lean_closure_set(v___x_831_, 0, lean_box(0));
lean_closure_set(v___x_831_, 1, lean_box(0));
lean_closure_set(v___x_831_, 2, lean_box(0));
lean_closure_set(v___x_831_, 3, v___f_829_);
lean_closure_set(v___x_831_, 4, v___f_830_);
return v___x_831_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_prod(lean_object* v_M_832_, lean_object* v_N_833_, lean_object* v_P_834_, lean_object* v_inst_835_, lean_object* v_inst_836_, lean_object* v_inst_837_, lean_object* v_f_838_, lean_object* v_g_839_){
_start:
{
lean_object* v___x_840_; 
v___x_840_ = lp_mathlib_AddMonoidHom_prod___redArg(v_f_838_, v_g_839_);
return v___x_840_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_prod___boxed(lean_object* v_M_841_, lean_object* v_N_842_, lean_object* v_P_843_, lean_object* v_inst_844_, lean_object* v_inst_845_, lean_object* v_inst_846_, lean_object* v_f_847_, lean_object* v_g_848_){
_start:
{
lean_object* v_res_849_; 
v_res_849_ = lp_mathlib_AddMonoidHom_prod(v_M_841_, v_N_842_, v_P_843_, v_inst_844_, v_inst_845_, v_inst_846_, v_f_847_, v_g_848_);
lean_dec_ref(v_inst_846_);
lean_dec_ref(v_inst_845_);
lean_dec_ref(v_inst_844_);
return v_res_849_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_prodMap___redArg(lean_object* v_f_850_, lean_object* v_g_851_){
_start:
{
lean_object* v___f_852_; lean_object* v___f_853_; lean_object* v___f_854_; lean_object* v___f_855_; lean_object* v___x_856_; 
v___f_852_ = ((lean_object*)(lp_mathlib_MulHom_fst___closed__0));
v___f_853_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_853_, 0, v___f_852_);
lean_closure_set(v___f_853_, 1, v_f_850_);
v___f_854_ = ((lean_object*)(lp_mathlib_MulHom_snd___closed__0));
v___f_855_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_855_, 0, v___f_854_);
lean_closure_set(v___f_855_, 1, v_g_851_);
v___x_856_ = lp_mathlib_MonoidHom_prod___redArg(v___f_853_, v___f_855_);
return v___x_856_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_prodMap(lean_object* v_M_857_, lean_object* v_N_858_, lean_object* v_inst_859_, lean_object* v_inst_860_, lean_object* v_M_x27_861_, lean_object* v_N_x27_862_, lean_object* v_inst_863_, lean_object* v_inst_864_, lean_object* v_f_865_, lean_object* v_g_866_){
_start:
{
lean_object* v___x_867_; 
v___x_867_ = lp_mathlib_MonoidHom_prodMap___redArg(v_f_865_, v_g_866_);
return v___x_867_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_prodMap___boxed(lean_object* v_M_868_, lean_object* v_N_869_, lean_object* v_inst_870_, lean_object* v_inst_871_, lean_object* v_M_x27_872_, lean_object* v_N_x27_873_, lean_object* v_inst_874_, lean_object* v_inst_875_, lean_object* v_f_876_, lean_object* v_g_877_){
_start:
{
lean_object* v_res_878_; 
v_res_878_ = lp_mathlib_MonoidHom_prodMap(v_M_868_, v_N_869_, v_inst_870_, v_inst_871_, v_M_x27_872_, v_N_x27_873_, v_inst_874_, v_inst_875_, v_f_876_, v_g_877_);
lean_dec_ref(v_inst_875_);
lean_dec_ref(v_inst_874_);
lean_dec_ref(v_inst_871_);
lean_dec_ref(v_inst_870_);
return v_res_878_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_prodMap___redArg(lean_object* v_f_879_, lean_object* v_g_880_){
_start:
{
lean_object* v___f_881_; lean_object* v___f_882_; lean_object* v___f_883_; lean_object* v___f_884_; lean_object* v___x_885_; 
v___f_881_ = ((lean_object*)(lp_mathlib_MulHom_fst___closed__0));
v___f_882_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_882_, 0, v___f_881_);
lean_closure_set(v___f_882_, 1, v_f_879_);
v___f_883_ = ((lean_object*)(lp_mathlib_MulHom_snd___closed__0));
v___f_884_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_884_, 0, v___f_883_);
lean_closure_set(v___f_884_, 1, v_g_880_);
v___x_885_ = lp_mathlib_AddMonoidHom_prod___redArg(v___f_882_, v___f_884_);
return v___x_885_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_prodMap(lean_object* v_M_886_, lean_object* v_N_887_, lean_object* v_inst_888_, lean_object* v_inst_889_, lean_object* v_M_x27_890_, lean_object* v_N_x27_891_, lean_object* v_inst_892_, lean_object* v_inst_893_, lean_object* v_f_894_, lean_object* v_g_895_){
_start:
{
lean_object* v___x_896_; 
v___x_896_ = lp_mathlib_AddMonoidHom_prodMap___redArg(v_f_894_, v_g_895_);
return v___x_896_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_prodMap___boxed(lean_object* v_M_897_, lean_object* v_N_898_, lean_object* v_inst_899_, lean_object* v_inst_900_, lean_object* v_M_x27_901_, lean_object* v_N_x27_902_, lean_object* v_inst_903_, lean_object* v_inst_904_, lean_object* v_f_905_, lean_object* v_g_906_){
_start:
{
lean_object* v_res_907_; 
v_res_907_ = lp_mathlib_AddMonoidHom_prodMap(v_M_897_, v_N_898_, v_inst_899_, v_inst_900_, v_M_x27_901_, v_N_x27_902_, v_inst_903_, v_inst_904_, v_f_905_, v_g_906_);
lean_dec_ref(v_inst_904_);
lean_dec_ref(v_inst_903_);
lean_dec_ref(v_inst_900_);
lean_dec_ref(v_inst_899_);
return v_res_907_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_coprod___redArg___lam__0(lean_object* v___f_908_, lean_object* v_f_909_, lean_object* v___f_910_, lean_object* v_g_911_, lean_object* v_toMul_912_, lean_object* v_m_913_){
_start:
{
lean_object* v___x_914_; lean_object* v___x_915_; lean_object* v___x_916_; 
lean_inc_ref(v_m_913_);
v___x_914_ = lp_mathlib_OneHom_comp___redArg___lam__0(v___f_908_, v_f_909_, v_m_913_);
v___x_915_ = lp_mathlib_OneHom_comp___redArg___lam__0(v___f_910_, v_g_911_, v_m_913_);
v___x_916_ = lean_apply_2(v_toMul_912_, v___x_914_, v___x_915_);
return v___x_916_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_coprod___redArg(lean_object* v_inst_917_, lean_object* v_f_918_, lean_object* v_g_919_){
_start:
{
lean_object* v___x_920_; lean_object* v___x_921_; lean_object* v_toMul_922_; lean_object* v___f_923_; lean_object* v___f_924_; lean_object* v___f_925_; 
v___x_920_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_917_);
v___x_921_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_920_);
v_toMul_922_ = lean_ctor_get(v___x_921_, 1);
lean_inc(v_toMul_922_);
lean_dec_ref(v___x_921_);
v___f_923_ = ((lean_object*)(lp_mathlib_MulHom_fst___closed__0));
v___f_924_ = ((lean_object*)(lp_mathlib_MulHom_snd___closed__0));
v___f_925_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_coprod___redArg___lam__0), 6, 5);
lean_closure_set(v___f_925_, 0, v___f_923_);
lean_closure_set(v___f_925_, 1, v_f_918_);
lean_closure_set(v___f_925_, 2, v___f_924_);
lean_closure_set(v___f_925_, 3, v_g_919_);
lean_closure_set(v___f_925_, 4, v_toMul_922_);
return v___f_925_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_coprod___redArg___boxed(lean_object* v_inst_926_, lean_object* v_f_927_, lean_object* v_g_928_){
_start:
{
lean_object* v_res_929_; 
v_res_929_ = lp_mathlib_MonoidHom_coprod___redArg(v_inst_926_, v_f_927_, v_g_928_);
lean_dec_ref(v_inst_926_);
return v_res_929_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_coprod(lean_object* v_M_930_, lean_object* v_N_931_, lean_object* v_P_932_, lean_object* v_inst_933_, lean_object* v_inst_934_, lean_object* v_inst_935_, lean_object* v_f_936_, lean_object* v_g_937_){
_start:
{
lean_object* v___x_938_; 
v___x_938_ = lp_mathlib_MonoidHom_coprod___redArg(v_inst_935_, v_f_936_, v_g_937_);
return v___x_938_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_coprod___boxed(lean_object* v_M_939_, lean_object* v_N_940_, lean_object* v_P_941_, lean_object* v_inst_942_, lean_object* v_inst_943_, lean_object* v_inst_944_, lean_object* v_f_945_, lean_object* v_g_946_){
_start:
{
lean_object* v_res_947_; 
v_res_947_ = lp_mathlib_MonoidHom_coprod(v_M_939_, v_N_940_, v_P_941_, v_inst_942_, v_inst_943_, v_inst_944_, v_f_945_, v_g_946_);
lean_dec_ref(v_inst_944_);
lean_dec_ref(v_inst_943_);
lean_dec_ref(v_inst_942_);
return v_res_947_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_coprod___redArg___lam__0(lean_object* v___f_948_, lean_object* v_f_949_, lean_object* v___f_950_, lean_object* v_g_951_, lean_object* v_toAdd_952_, lean_object* v_m_953_){
_start:
{
lean_object* v___x_954_; lean_object* v___x_955_; lean_object* v___x_956_; 
lean_inc_ref(v_m_953_);
v___x_954_ = lp_mathlib_OneHom_comp___redArg___lam__0(v___f_948_, v_f_949_, v_m_953_);
v___x_955_ = lp_mathlib_OneHom_comp___redArg___lam__0(v___f_950_, v_g_951_, v_m_953_);
v___x_956_ = lean_apply_2(v_toAdd_952_, v___x_954_, v___x_955_);
return v___x_956_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_coprod___redArg(lean_object* v_inst_957_, lean_object* v_f_958_, lean_object* v_g_959_){
_start:
{
lean_object* v___x_960_; lean_object* v___x_961_; lean_object* v_toAdd_962_; lean_object* v___f_963_; lean_object* v___f_964_; lean_object* v___f_965_; 
v___x_960_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_957_);
v___x_961_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_960_);
v_toAdd_962_ = lean_ctor_get(v___x_961_, 1);
lean_inc(v_toAdd_962_);
lean_dec_ref(v___x_961_);
v___f_963_ = ((lean_object*)(lp_mathlib_MulHom_fst___closed__0));
v___f_964_ = ((lean_object*)(lp_mathlib_MulHom_snd___closed__0));
v___f_965_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_coprod___redArg___lam__0), 6, 5);
lean_closure_set(v___f_965_, 0, v___f_963_);
lean_closure_set(v___f_965_, 1, v_f_958_);
lean_closure_set(v___f_965_, 2, v___f_964_);
lean_closure_set(v___f_965_, 3, v_g_959_);
lean_closure_set(v___f_965_, 4, v_toAdd_962_);
return v___f_965_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_coprod___redArg___boxed(lean_object* v_inst_966_, lean_object* v_f_967_, lean_object* v_g_968_){
_start:
{
lean_object* v_res_969_; 
v_res_969_ = lp_mathlib_AddMonoidHom_coprod___redArg(v_inst_966_, v_f_967_, v_g_968_);
lean_dec_ref(v_inst_966_);
return v_res_969_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_coprod(lean_object* v_M_970_, lean_object* v_N_971_, lean_object* v_P_972_, lean_object* v_inst_973_, lean_object* v_inst_974_, lean_object* v_inst_975_, lean_object* v_f_976_, lean_object* v_g_977_){
_start:
{
lean_object* v___x_978_; 
v___x_978_ = lp_mathlib_AddMonoidHom_coprod___redArg(v_inst_975_, v_f_976_, v_g_977_);
return v___x_978_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_coprod___boxed(lean_object* v_M_979_, lean_object* v_N_980_, lean_object* v_P_981_, lean_object* v_inst_982_, lean_object* v_inst_983_, lean_object* v_inst_984_, lean_object* v_f_985_, lean_object* v_g_986_){
_start:
{
lean_object* v_res_987_; 
v_res_987_ = lp_mathlib_AddMonoidHom_coprod(v_M_979_, v_N_980_, v_P_981_, v_inst_982_, v_inst_983_, v_inst_984_, v_f_985_, v_g_986_);
lean_dec_ref(v_inst_984_);
lean_dec_ref(v_inst_983_);
lean_dec_ref(v_inst_982_);
return v_res_987_;
}
}
static lean_object* _init_lp_mathlib_MulEquiv_prodComm___closed__0(void){
_start:
{
lean_object* v___x_988_; 
v___x_988_ = lp_mathlib_Equiv_prodComm(lean_box(0), lean_box(0));
return v___x_988_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodComm(lean_object* v_M_989_, lean_object* v_N_990_, lean_object* v_inst_991_, lean_object* v_inst_992_){
_start:
{
lean_object* v___x_993_; 
v___x_993_ = lean_obj_once(&lp_mathlib_MulEquiv_prodComm___closed__0, &lp_mathlib_MulEquiv_prodComm___closed__0_once, _init_lp_mathlib_MulEquiv_prodComm___closed__0);
return v___x_993_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodComm___boxed(lean_object* v_M_994_, lean_object* v_N_995_, lean_object* v_inst_996_, lean_object* v_inst_997_){
_start:
{
lean_object* v_res_998_; 
v_res_998_ = lp_mathlib_MulEquiv_prodComm(v_M_994_, v_N_995_, v_inst_996_, v_inst_997_);
lean_dec_ref(v_inst_997_);
lean_dec_ref(v_inst_996_);
return v_res_998_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodComm(lean_object* v_M_999_, lean_object* v_N_1000_, lean_object* v_inst_1001_, lean_object* v_inst_1002_){
_start:
{
lean_object* v___x_1003_; 
v___x_1003_ = lean_obj_once(&lp_mathlib_MulEquiv_prodComm___closed__0, &lp_mathlib_MulEquiv_prodComm___closed__0_once, _init_lp_mathlib_MulEquiv_prodComm___closed__0);
return v___x_1003_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodComm___boxed(lean_object* v_M_1004_, lean_object* v_N_1005_, lean_object* v_inst_1006_, lean_object* v_inst_1007_){
_start:
{
lean_object* v_res_1008_; 
v_res_1008_ = lp_mathlib_AddEquiv_prodComm(v_M_1004_, v_N_1005_, v_inst_1006_, v_inst_1007_);
lean_dec_ref(v_inst_1007_);
lean_dec_ref(v_inst_1006_);
return v_res_1008_;
}
}
static lean_object* _init_lp_mathlib_MulEquiv_prodAssoc___closed__0(void){
_start:
{
lean_object* v___x_1009_; 
v___x_1009_ = lp_mathlib_Equiv_prodAssoc(lean_box(0), lean_box(0), lean_box(0));
return v___x_1009_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodAssoc(lean_object* v_M_1010_, lean_object* v_N_1011_, lean_object* v_P_1012_, lean_object* v_inst_1013_, lean_object* v_inst_1014_, lean_object* v_inst_1015_){
_start:
{
lean_object* v___x_1016_; 
v___x_1016_ = lean_obj_once(&lp_mathlib_MulEquiv_prodAssoc___closed__0, &lp_mathlib_MulEquiv_prodAssoc___closed__0_once, _init_lp_mathlib_MulEquiv_prodAssoc___closed__0);
return v___x_1016_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodAssoc___boxed(lean_object* v_M_1017_, lean_object* v_N_1018_, lean_object* v_P_1019_, lean_object* v_inst_1020_, lean_object* v_inst_1021_, lean_object* v_inst_1022_){
_start:
{
lean_object* v_res_1023_; 
v_res_1023_ = lp_mathlib_MulEquiv_prodAssoc(v_M_1017_, v_N_1018_, v_P_1019_, v_inst_1020_, v_inst_1021_, v_inst_1022_);
lean_dec_ref(v_inst_1022_);
lean_dec_ref(v_inst_1021_);
lean_dec_ref(v_inst_1020_);
return v_res_1023_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodAssoc(lean_object* v_M_1024_, lean_object* v_N_1025_, lean_object* v_P_1026_, lean_object* v_inst_1027_, lean_object* v_inst_1028_, lean_object* v_inst_1029_){
_start:
{
lean_object* v___x_1030_; 
v___x_1030_ = lean_obj_once(&lp_mathlib_MulEquiv_prodAssoc___closed__0, &lp_mathlib_MulEquiv_prodAssoc___closed__0_once, _init_lp_mathlib_MulEquiv_prodAssoc___closed__0);
return v___x_1030_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodAssoc___boxed(lean_object* v_M_1031_, lean_object* v_N_1032_, lean_object* v_P_1033_, lean_object* v_inst_1034_, lean_object* v_inst_1035_, lean_object* v_inst_1036_){
_start:
{
lean_object* v_res_1037_; 
v_res_1037_ = lp_mathlib_AddEquiv_prodAssoc(v_M_1031_, v_N_1032_, v_P_1033_, v_inst_1034_, v_inst_1035_, v_inst_1036_);
lean_dec_ref(v_inst_1036_);
lean_dec_ref(v_inst_1035_);
lean_dec_ref(v_inst_1034_);
return v_res_1037_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodProdProdComm___lam__0(lean_object* v_mnmn_1038_){
_start:
{
lean_object* v_fst_1039_; lean_object* v_snd_1040_; lean_object* v___x_1042_; uint8_t v_isShared_1043_; uint8_t v_isSharedCheck_1065_; 
v_fst_1039_ = lean_ctor_get(v_mnmn_1038_, 0);
v_snd_1040_ = lean_ctor_get(v_mnmn_1038_, 1);
v_isSharedCheck_1065_ = !lean_is_exclusive(v_mnmn_1038_);
if (v_isSharedCheck_1065_ == 0)
{
v___x_1042_ = v_mnmn_1038_;
v_isShared_1043_ = v_isSharedCheck_1065_;
goto v_resetjp_1041_;
}
else
{
lean_inc(v_snd_1040_);
lean_inc(v_fst_1039_);
lean_dec(v_mnmn_1038_);
v___x_1042_ = lean_box(0);
v_isShared_1043_ = v_isSharedCheck_1065_;
goto v_resetjp_1041_;
}
v_resetjp_1041_:
{
lean_object* v_fst_1044_; lean_object* v_snd_1045_; lean_object* v___x_1047_; uint8_t v_isShared_1048_; uint8_t v_isSharedCheck_1064_; 
v_fst_1044_ = lean_ctor_get(v_fst_1039_, 0);
v_snd_1045_ = lean_ctor_get(v_fst_1039_, 1);
v_isSharedCheck_1064_ = !lean_is_exclusive(v_fst_1039_);
if (v_isSharedCheck_1064_ == 0)
{
v___x_1047_ = v_fst_1039_;
v_isShared_1048_ = v_isSharedCheck_1064_;
goto v_resetjp_1046_;
}
else
{
lean_inc(v_snd_1045_);
lean_inc(v_fst_1044_);
lean_dec(v_fst_1039_);
v___x_1047_ = lean_box(0);
v_isShared_1048_ = v_isSharedCheck_1064_;
goto v_resetjp_1046_;
}
v_resetjp_1046_:
{
lean_object* v_fst_1049_; lean_object* v_snd_1050_; lean_object* v___x_1052_; uint8_t v_isShared_1053_; uint8_t v_isSharedCheck_1063_; 
v_fst_1049_ = lean_ctor_get(v_snd_1040_, 0);
v_snd_1050_ = lean_ctor_get(v_snd_1040_, 1);
v_isSharedCheck_1063_ = !lean_is_exclusive(v_snd_1040_);
if (v_isSharedCheck_1063_ == 0)
{
v___x_1052_ = v_snd_1040_;
v_isShared_1053_ = v_isSharedCheck_1063_;
goto v_resetjp_1051_;
}
else
{
lean_inc(v_snd_1050_);
lean_inc(v_fst_1049_);
lean_dec(v_snd_1040_);
v___x_1052_ = lean_box(0);
v_isShared_1053_ = v_isSharedCheck_1063_;
goto v_resetjp_1051_;
}
v_resetjp_1051_:
{
lean_object* v___x_1055_; 
if (v_isShared_1053_ == 0)
{
lean_ctor_set(v___x_1052_, 1, v_fst_1049_);
lean_ctor_set(v___x_1052_, 0, v_fst_1044_);
v___x_1055_ = v___x_1052_;
goto v_reusejp_1054_;
}
else
{
lean_object* v_reuseFailAlloc_1062_; 
v_reuseFailAlloc_1062_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1062_, 0, v_fst_1044_);
lean_ctor_set(v_reuseFailAlloc_1062_, 1, v_fst_1049_);
v___x_1055_ = v_reuseFailAlloc_1062_;
goto v_reusejp_1054_;
}
v_reusejp_1054_:
{
lean_object* v___x_1057_; 
if (v_isShared_1048_ == 0)
{
lean_ctor_set(v___x_1047_, 1, v_snd_1050_);
lean_ctor_set(v___x_1047_, 0, v_snd_1045_);
v___x_1057_ = v___x_1047_;
goto v_reusejp_1056_;
}
else
{
lean_object* v_reuseFailAlloc_1061_; 
v_reuseFailAlloc_1061_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1061_, 0, v_snd_1045_);
lean_ctor_set(v_reuseFailAlloc_1061_, 1, v_snd_1050_);
v___x_1057_ = v_reuseFailAlloc_1061_;
goto v_reusejp_1056_;
}
v_reusejp_1056_:
{
lean_object* v___x_1059_; 
if (v_isShared_1043_ == 0)
{
lean_ctor_set(v___x_1042_, 1, v___x_1057_);
lean_ctor_set(v___x_1042_, 0, v___x_1055_);
v___x_1059_ = v___x_1042_;
goto v_reusejp_1058_;
}
else
{
lean_object* v_reuseFailAlloc_1060_; 
v_reuseFailAlloc_1060_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1060_, 0, v___x_1055_);
lean_ctor_set(v_reuseFailAlloc_1060_, 1, v___x_1057_);
v___x_1059_ = v_reuseFailAlloc_1060_;
goto v_reusejp_1058_;
}
v_reusejp_1058_:
{
return v___x_1059_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodProdProdComm___lam__1(lean_object* v_mmnn_1066_){
_start:
{
lean_object* v_fst_1067_; lean_object* v_snd_1068_; lean_object* v___x_1070_; uint8_t v_isShared_1071_; uint8_t v_isSharedCheck_1093_; 
v_fst_1067_ = lean_ctor_get(v_mmnn_1066_, 0);
v_snd_1068_ = lean_ctor_get(v_mmnn_1066_, 1);
v_isSharedCheck_1093_ = !lean_is_exclusive(v_mmnn_1066_);
if (v_isSharedCheck_1093_ == 0)
{
v___x_1070_ = v_mmnn_1066_;
v_isShared_1071_ = v_isSharedCheck_1093_;
goto v_resetjp_1069_;
}
else
{
lean_inc(v_snd_1068_);
lean_inc(v_fst_1067_);
lean_dec(v_mmnn_1066_);
v___x_1070_ = lean_box(0);
v_isShared_1071_ = v_isSharedCheck_1093_;
goto v_resetjp_1069_;
}
v_resetjp_1069_:
{
lean_object* v_fst_1072_; lean_object* v_snd_1073_; lean_object* v___x_1075_; uint8_t v_isShared_1076_; uint8_t v_isSharedCheck_1092_; 
v_fst_1072_ = lean_ctor_get(v_fst_1067_, 0);
v_snd_1073_ = lean_ctor_get(v_fst_1067_, 1);
v_isSharedCheck_1092_ = !lean_is_exclusive(v_fst_1067_);
if (v_isSharedCheck_1092_ == 0)
{
v___x_1075_ = v_fst_1067_;
v_isShared_1076_ = v_isSharedCheck_1092_;
goto v_resetjp_1074_;
}
else
{
lean_inc(v_snd_1073_);
lean_inc(v_fst_1072_);
lean_dec(v_fst_1067_);
v___x_1075_ = lean_box(0);
v_isShared_1076_ = v_isSharedCheck_1092_;
goto v_resetjp_1074_;
}
v_resetjp_1074_:
{
lean_object* v_fst_1077_; lean_object* v_snd_1078_; lean_object* v___x_1080_; uint8_t v_isShared_1081_; uint8_t v_isSharedCheck_1091_; 
v_fst_1077_ = lean_ctor_get(v_snd_1068_, 0);
v_snd_1078_ = lean_ctor_get(v_snd_1068_, 1);
v_isSharedCheck_1091_ = !lean_is_exclusive(v_snd_1068_);
if (v_isSharedCheck_1091_ == 0)
{
v___x_1080_ = v_snd_1068_;
v_isShared_1081_ = v_isSharedCheck_1091_;
goto v_resetjp_1079_;
}
else
{
lean_inc(v_snd_1078_);
lean_inc(v_fst_1077_);
lean_dec(v_snd_1068_);
v___x_1080_ = lean_box(0);
v_isShared_1081_ = v_isSharedCheck_1091_;
goto v_resetjp_1079_;
}
v_resetjp_1079_:
{
lean_object* v___x_1083_; 
if (v_isShared_1081_ == 0)
{
lean_ctor_set(v___x_1080_, 1, v_fst_1077_);
lean_ctor_set(v___x_1080_, 0, v_fst_1072_);
v___x_1083_ = v___x_1080_;
goto v_reusejp_1082_;
}
else
{
lean_object* v_reuseFailAlloc_1090_; 
v_reuseFailAlloc_1090_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1090_, 0, v_fst_1072_);
lean_ctor_set(v_reuseFailAlloc_1090_, 1, v_fst_1077_);
v___x_1083_ = v_reuseFailAlloc_1090_;
goto v_reusejp_1082_;
}
v_reusejp_1082_:
{
lean_object* v___x_1085_; 
if (v_isShared_1076_ == 0)
{
lean_ctor_set(v___x_1075_, 1, v_snd_1078_);
lean_ctor_set(v___x_1075_, 0, v_snd_1073_);
v___x_1085_ = v___x_1075_;
goto v_reusejp_1084_;
}
else
{
lean_object* v_reuseFailAlloc_1089_; 
v_reuseFailAlloc_1089_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1089_, 0, v_snd_1073_);
lean_ctor_set(v_reuseFailAlloc_1089_, 1, v_snd_1078_);
v___x_1085_ = v_reuseFailAlloc_1089_;
goto v_reusejp_1084_;
}
v_reusejp_1084_:
{
lean_object* v___x_1087_; 
if (v_isShared_1071_ == 0)
{
lean_ctor_set(v___x_1070_, 1, v___x_1085_);
lean_ctor_set(v___x_1070_, 0, v___x_1083_);
v___x_1087_ = v___x_1070_;
goto v_reusejp_1086_;
}
else
{
lean_object* v_reuseFailAlloc_1088_; 
v_reuseFailAlloc_1088_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1088_, 0, v___x_1083_);
lean_ctor_set(v_reuseFailAlloc_1088_, 1, v___x_1085_);
v___x_1087_ = v_reuseFailAlloc_1088_;
goto v_reusejp_1086_;
}
v_reusejp_1086_:
{
return v___x_1087_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodProdProdComm(lean_object* v_M_1099_, lean_object* v_N_1100_, lean_object* v_inst_1101_, lean_object* v_inst_1102_, lean_object* v_M_x27_1103_, lean_object* v_N_x27_1104_, lean_object* v_inst_1105_, lean_object* v_inst_1106_){
_start:
{
lean_object* v___x_1107_; 
v___x_1107_ = ((lean_object*)(lp_mathlib_MulEquiv_prodProdProdComm___closed__2));
return v___x_1107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodProdProdComm___boxed(lean_object* v_M_1108_, lean_object* v_N_1109_, lean_object* v_inst_1110_, lean_object* v_inst_1111_, lean_object* v_M_x27_1112_, lean_object* v_N_x27_1113_, lean_object* v_inst_1114_, lean_object* v_inst_1115_){
_start:
{
lean_object* v_res_1116_; 
v_res_1116_ = lp_mathlib_MulEquiv_prodProdProdComm(v_M_1108_, v_N_1109_, v_inst_1110_, v_inst_1111_, v_M_x27_1112_, v_N_x27_1113_, v_inst_1114_, v_inst_1115_);
lean_dec_ref(v_inst_1115_);
lean_dec_ref(v_inst_1114_);
lean_dec_ref(v_inst_1111_);
lean_dec_ref(v_inst_1110_);
return v_res_1116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodProdProdComm(lean_object* v_M_1117_, lean_object* v_N_1118_, lean_object* v_inst_1119_, lean_object* v_inst_1120_, lean_object* v_M_x27_1121_, lean_object* v_N_x27_1122_, lean_object* v_inst_1123_, lean_object* v_inst_1124_){
_start:
{
lean_object* v___x_1125_; 
v___x_1125_ = ((lean_object*)(lp_mathlib_MulEquiv_prodProdProdComm___closed__2));
return v___x_1125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodProdProdComm___boxed(lean_object* v_M_1126_, lean_object* v_N_1127_, lean_object* v_inst_1128_, lean_object* v_inst_1129_, lean_object* v_M_x27_1130_, lean_object* v_N_x27_1131_, lean_object* v_inst_1132_, lean_object* v_inst_1133_){
_start:
{
lean_object* v_res_1134_; 
v_res_1134_ = lp_mathlib_AddEquiv_prodProdProdComm(v_M_1126_, v_N_1127_, v_inst_1128_, v_inst_1129_, v_M_x27_1130_, v_N_x27_1131_, v_inst_1132_, v_inst_1133_);
lean_dec_ref(v_inst_1133_);
lean_dec_ref(v_inst_1132_);
lean_dec_ref(v_inst_1129_);
lean_dec_ref(v_inst_1128_);
return v_res_1134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodCongr___redArg(lean_object* v_f_1135_, lean_object* v_g_1136_){
_start:
{
lean_object* v___x_1137_; 
v___x_1137_ = lp_mathlib_Equiv_prodCongr___redArg(v_f_1135_, v_g_1136_);
return v___x_1137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodCongr(lean_object* v_M_1138_, lean_object* v_N_1139_, lean_object* v_inst_1140_, lean_object* v_inst_1141_, lean_object* v_M_x27_1142_, lean_object* v_N_x27_1143_, lean_object* v_inst_1144_, lean_object* v_inst_1145_, lean_object* v_f_1146_, lean_object* v_g_1147_){
_start:
{
lean_object* v___x_1148_; 
v___x_1148_ = lp_mathlib_Equiv_prodCongr___redArg(v_f_1146_, v_g_1147_);
return v___x_1148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodCongr___boxed(lean_object* v_M_1149_, lean_object* v_N_1150_, lean_object* v_inst_1151_, lean_object* v_inst_1152_, lean_object* v_M_x27_1153_, lean_object* v_N_x27_1154_, lean_object* v_inst_1155_, lean_object* v_inst_1156_, lean_object* v_f_1157_, lean_object* v_g_1158_){
_start:
{
lean_object* v_res_1159_; 
v_res_1159_ = lp_mathlib_MulEquiv_prodCongr(v_M_1149_, v_N_1150_, v_inst_1151_, v_inst_1152_, v_M_x27_1153_, v_N_x27_1154_, v_inst_1155_, v_inst_1156_, v_f_1157_, v_g_1158_);
lean_dec_ref(v_inst_1156_);
lean_dec_ref(v_inst_1155_);
lean_dec_ref(v_inst_1152_);
lean_dec_ref(v_inst_1151_);
return v_res_1159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodCongr___redArg(lean_object* v_f_1160_, lean_object* v_g_1161_){
_start:
{
lean_object* v___x_1162_; 
v___x_1162_ = lp_mathlib_Equiv_prodCongr___redArg(v_f_1160_, v_g_1161_);
return v___x_1162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodCongr(lean_object* v_M_1163_, lean_object* v_N_1164_, lean_object* v_inst_1165_, lean_object* v_inst_1166_, lean_object* v_M_x27_1167_, lean_object* v_N_x27_1168_, lean_object* v_inst_1169_, lean_object* v_inst_1170_, lean_object* v_f_1171_, lean_object* v_g_1172_){
_start:
{
lean_object* v___x_1173_; 
v___x_1173_ = lp_mathlib_Equiv_prodCongr___redArg(v_f_1171_, v_g_1172_);
return v___x_1173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodCongr___boxed(lean_object* v_M_1174_, lean_object* v_N_1175_, lean_object* v_inst_1176_, lean_object* v_inst_1177_, lean_object* v_M_x27_1178_, lean_object* v_N_x27_1179_, lean_object* v_inst_1180_, lean_object* v_inst_1181_, lean_object* v_f_1182_, lean_object* v_g_1183_){
_start:
{
lean_object* v_res_1184_; 
v_res_1184_ = lp_mathlib_AddEquiv_prodCongr(v_M_1174_, v_N_1175_, v_inst_1176_, v_inst_1177_, v_M_x27_1178_, v_N_x27_1179_, v_inst_1180_, v_inst_1181_, v_f_1182_, v_g_1183_);
lean_dec_ref(v_inst_1181_);
lean_dec_ref(v_inst_1180_);
lean_dec_ref(v_inst_1177_);
lean_dec_ref(v_inst_1176_);
return v_res_1184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_uniqueProd___redArg(lean_object* v_inst_1185_){
_start:
{
lean_object* v___x_1186_; 
v___x_1186_ = lp_mathlib_Equiv_uniqueProd___redArg(v_inst_1185_);
return v___x_1186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_uniqueProd(lean_object* v_M_1187_, lean_object* v_N_1188_, lean_object* v_inst_1189_, lean_object* v_inst_1190_, lean_object* v_inst_1191_){
_start:
{
lean_object* v___x_1192_; 
v___x_1192_ = lp_mathlib_Equiv_uniqueProd___redArg(v_inst_1191_);
return v___x_1192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_uniqueProd___boxed(lean_object* v_M_1193_, lean_object* v_N_1194_, lean_object* v_inst_1195_, lean_object* v_inst_1196_, lean_object* v_inst_1197_){
_start:
{
lean_object* v_res_1198_; 
v_res_1198_ = lp_mathlib_MulEquiv_uniqueProd(v_M_1193_, v_N_1194_, v_inst_1195_, v_inst_1196_, v_inst_1197_);
lean_dec_ref(v_inst_1196_);
lean_dec_ref(v_inst_1195_);
return v_res_1198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_uniqueProd___redArg(lean_object* v_inst_1199_){
_start:
{
lean_object* v___x_1200_; 
v___x_1200_ = lp_mathlib_Equiv_uniqueProd___redArg(v_inst_1199_);
return v___x_1200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_uniqueProd(lean_object* v_M_1201_, lean_object* v_N_1202_, lean_object* v_inst_1203_, lean_object* v_inst_1204_, lean_object* v_inst_1205_){
_start:
{
lean_object* v___x_1206_; 
v___x_1206_ = lp_mathlib_Equiv_uniqueProd___redArg(v_inst_1205_);
return v___x_1206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_uniqueProd___boxed(lean_object* v_M_1207_, lean_object* v_N_1208_, lean_object* v_inst_1209_, lean_object* v_inst_1210_, lean_object* v_inst_1211_){
_start:
{
lean_object* v_res_1212_; 
v_res_1212_ = lp_mathlib_AddEquiv_uniqueProd(v_M_1207_, v_N_1208_, v_inst_1209_, v_inst_1210_, v_inst_1211_);
lean_dec_ref(v_inst_1210_);
lean_dec_ref(v_inst_1209_);
return v_res_1212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodUnique___redArg(lean_object* v_inst_1213_){
_start:
{
lean_object* v___x_1214_; 
v___x_1214_ = lp_mathlib_Equiv_prodUnique___redArg(v_inst_1213_);
return v___x_1214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodUnique(lean_object* v_M_1215_, lean_object* v_N_1216_, lean_object* v_inst_1217_, lean_object* v_inst_1218_, lean_object* v_inst_1219_){
_start:
{
lean_object* v___x_1220_; 
v___x_1220_ = lp_mathlib_Equiv_prodUnique___redArg(v_inst_1219_);
return v___x_1220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodUnique___boxed(lean_object* v_M_1221_, lean_object* v_N_1222_, lean_object* v_inst_1223_, lean_object* v_inst_1224_, lean_object* v_inst_1225_){
_start:
{
lean_object* v_res_1226_; 
v_res_1226_ = lp_mathlib_MulEquiv_prodUnique(v_M_1221_, v_N_1222_, v_inst_1223_, v_inst_1224_, v_inst_1225_);
lean_dec_ref(v_inst_1224_);
lean_dec_ref(v_inst_1223_);
return v_res_1226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodUnique___redArg(lean_object* v_inst_1227_){
_start:
{
lean_object* v___x_1228_; 
v___x_1228_ = lp_mathlib_Equiv_prodUnique___redArg(v_inst_1227_);
return v___x_1228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodUnique(lean_object* v_M_1229_, lean_object* v_N_1230_, lean_object* v_inst_1231_, lean_object* v_inst_1232_, lean_object* v_inst_1233_){
_start:
{
lean_object* v___x_1234_; 
v___x_1234_ = lp_mathlib_Equiv_prodUnique___redArg(v_inst_1233_);
return v___x_1234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodUnique___boxed(lean_object* v_M_1235_, lean_object* v_N_1236_, lean_object* v_inst_1237_, lean_object* v_inst_1238_, lean_object* v_inst_1239_){
_start:
{
lean_object* v_res_1240_; 
v_res_1240_ = lp_mathlib_AddEquiv_prodUnique(v_M_1235_, v_N_1236_, v_inst_1237_, v_inst_1238_, v_inst_1239_);
lean_dec_ref(v_inst_1238_);
lean_dec_ref(v_inst_1237_);
return v_res_1240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodUnits___lam__0(lean_object* v_u_1241_){
_start:
{
lean_object* v_fst_1242_; lean_object* v_snd_1243_; lean_object* v___x_1245_; uint8_t v_isShared_1246_; uint8_t v_isSharedCheck_1268_; 
v_fst_1242_ = lean_ctor_get(v_u_1241_, 0);
v_snd_1243_ = lean_ctor_get(v_u_1241_, 1);
v_isSharedCheck_1268_ = !lean_is_exclusive(v_u_1241_);
if (v_isSharedCheck_1268_ == 0)
{
v___x_1245_ = v_u_1241_;
v_isShared_1246_ = v_isSharedCheck_1268_;
goto v_resetjp_1244_;
}
else
{
lean_inc(v_snd_1243_);
lean_inc(v_fst_1242_);
lean_dec(v_u_1241_);
v___x_1245_ = lean_box(0);
v_isShared_1246_ = v_isSharedCheck_1268_;
goto v_resetjp_1244_;
}
v_resetjp_1244_:
{
lean_object* v_val_1247_; lean_object* v_inv_1248_; lean_object* v___x_1250_; uint8_t v_isShared_1251_; uint8_t v_isSharedCheck_1267_; 
v_val_1247_ = lean_ctor_get(v_fst_1242_, 0);
v_inv_1248_ = lean_ctor_get(v_fst_1242_, 1);
v_isSharedCheck_1267_ = !lean_is_exclusive(v_fst_1242_);
if (v_isSharedCheck_1267_ == 0)
{
v___x_1250_ = v_fst_1242_;
v_isShared_1251_ = v_isSharedCheck_1267_;
goto v_resetjp_1249_;
}
else
{
lean_inc(v_inv_1248_);
lean_inc(v_val_1247_);
lean_dec(v_fst_1242_);
v___x_1250_ = lean_box(0);
v_isShared_1251_ = v_isSharedCheck_1267_;
goto v_resetjp_1249_;
}
v_resetjp_1249_:
{
lean_object* v_val_1252_; lean_object* v_inv_1253_; lean_object* v___x_1255_; uint8_t v_isShared_1256_; uint8_t v_isSharedCheck_1266_; 
v_val_1252_ = lean_ctor_get(v_snd_1243_, 0);
v_inv_1253_ = lean_ctor_get(v_snd_1243_, 1);
v_isSharedCheck_1266_ = !lean_is_exclusive(v_snd_1243_);
if (v_isSharedCheck_1266_ == 0)
{
v___x_1255_ = v_snd_1243_;
v_isShared_1256_ = v_isSharedCheck_1266_;
goto v_resetjp_1254_;
}
else
{
lean_inc(v_inv_1253_);
lean_inc(v_val_1252_);
lean_dec(v_snd_1243_);
v___x_1255_ = lean_box(0);
v_isShared_1256_ = v_isSharedCheck_1266_;
goto v_resetjp_1254_;
}
v_resetjp_1254_:
{
lean_object* v___x_1258_; 
if (v_isShared_1246_ == 0)
{
lean_ctor_set(v___x_1245_, 1, v_val_1252_);
lean_ctor_set(v___x_1245_, 0, v_val_1247_);
v___x_1258_ = v___x_1245_;
goto v_reusejp_1257_;
}
else
{
lean_object* v_reuseFailAlloc_1265_; 
v_reuseFailAlloc_1265_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1265_, 0, v_val_1247_);
lean_ctor_set(v_reuseFailAlloc_1265_, 1, v_val_1252_);
v___x_1258_ = v_reuseFailAlloc_1265_;
goto v_reusejp_1257_;
}
v_reusejp_1257_:
{
lean_object* v___x_1260_; 
if (v_isShared_1251_ == 0)
{
lean_ctor_set(v___x_1250_, 1, v_inv_1253_);
lean_ctor_set(v___x_1250_, 0, v_inv_1248_);
v___x_1260_ = v___x_1250_;
goto v_reusejp_1259_;
}
else
{
lean_object* v_reuseFailAlloc_1264_; 
v_reuseFailAlloc_1264_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1264_, 0, v_inv_1248_);
lean_ctor_set(v_reuseFailAlloc_1264_, 1, v_inv_1253_);
v___x_1260_ = v_reuseFailAlloc_1264_;
goto v_reusejp_1259_;
}
v_reusejp_1259_:
{
lean_object* v___x_1262_; 
if (v_isShared_1256_ == 0)
{
lean_ctor_set(v___x_1255_, 1, v___x_1260_);
lean_ctor_set(v___x_1255_, 0, v___x_1258_);
v___x_1262_ = v___x_1255_;
goto v_reusejp_1261_;
}
else
{
lean_object* v_reuseFailAlloc_1263_; 
v_reuseFailAlloc_1263_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1263_, 0, v___x_1258_);
lean_ctor_set(v_reuseFailAlloc_1263_, 1, v___x_1260_);
v___x_1262_ = v_reuseFailAlloc_1263_;
goto v_reusejp_1261_;
}
v_reusejp_1261_:
{
return v___x_1262_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodUnits___lam__1(lean_object* v___f_1269_, lean_object* v___f_1270_, lean_object* v___y_1271_){
_start:
{
lean_object* v___x_78__overap_1272_; lean_object* v___x_1273_; 
v___x_78__overap_1272_ = lp_mathlib_MonoidHom_prod___redArg(v___f_1269_, v___f_1270_);
v___x_1273_ = lean_apply_1(v___x_78__overap_1272_, v___y_1271_);
return v___x_1273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodUnits(lean_object* v_M_1285_, lean_object* v_N_1286_, lean_object* v_inst_1287_, lean_object* v_inst_1288_){
_start:
{
lean_object* v___x_1289_; 
v___x_1289_ = ((lean_object*)(lp_mathlib_MulEquiv_prodUnits___closed__4));
return v___x_1289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodUnits___boxed(lean_object* v_M_1290_, lean_object* v_N_1291_, lean_object* v_inst_1292_, lean_object* v_inst_1293_){
_start:
{
lean_object* v_res_1294_; 
v_res_1294_ = lp_mathlib_MulEquiv_prodUnits(v_M_1290_, v_N_1291_, v_inst_1292_, v_inst_1293_);
lean_dec_ref(v_inst_1293_);
lean_dec_ref(v_inst_1292_);
return v_res_1294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodAddUnits___lam__0(lean_object* v_u_1295_){
_start:
{
lean_object* v_fst_1296_; lean_object* v_snd_1297_; lean_object* v___x_1299_; uint8_t v_isShared_1300_; uint8_t v_isSharedCheck_1322_; 
v_fst_1296_ = lean_ctor_get(v_u_1295_, 0);
v_snd_1297_ = lean_ctor_get(v_u_1295_, 1);
v_isSharedCheck_1322_ = !lean_is_exclusive(v_u_1295_);
if (v_isSharedCheck_1322_ == 0)
{
v___x_1299_ = v_u_1295_;
v_isShared_1300_ = v_isSharedCheck_1322_;
goto v_resetjp_1298_;
}
else
{
lean_inc(v_snd_1297_);
lean_inc(v_fst_1296_);
lean_dec(v_u_1295_);
v___x_1299_ = lean_box(0);
v_isShared_1300_ = v_isSharedCheck_1322_;
goto v_resetjp_1298_;
}
v_resetjp_1298_:
{
lean_object* v_val_1301_; lean_object* v_neg_1302_; lean_object* v___x_1304_; uint8_t v_isShared_1305_; uint8_t v_isSharedCheck_1321_; 
v_val_1301_ = lean_ctor_get(v_fst_1296_, 0);
v_neg_1302_ = lean_ctor_get(v_fst_1296_, 1);
v_isSharedCheck_1321_ = !lean_is_exclusive(v_fst_1296_);
if (v_isSharedCheck_1321_ == 0)
{
v___x_1304_ = v_fst_1296_;
v_isShared_1305_ = v_isSharedCheck_1321_;
goto v_resetjp_1303_;
}
else
{
lean_inc(v_neg_1302_);
lean_inc(v_val_1301_);
lean_dec(v_fst_1296_);
v___x_1304_ = lean_box(0);
v_isShared_1305_ = v_isSharedCheck_1321_;
goto v_resetjp_1303_;
}
v_resetjp_1303_:
{
lean_object* v_val_1306_; lean_object* v_neg_1307_; lean_object* v___x_1309_; uint8_t v_isShared_1310_; uint8_t v_isSharedCheck_1320_; 
v_val_1306_ = lean_ctor_get(v_snd_1297_, 0);
v_neg_1307_ = lean_ctor_get(v_snd_1297_, 1);
v_isSharedCheck_1320_ = !lean_is_exclusive(v_snd_1297_);
if (v_isSharedCheck_1320_ == 0)
{
v___x_1309_ = v_snd_1297_;
v_isShared_1310_ = v_isSharedCheck_1320_;
goto v_resetjp_1308_;
}
else
{
lean_inc(v_neg_1307_);
lean_inc(v_val_1306_);
lean_dec(v_snd_1297_);
v___x_1309_ = lean_box(0);
v_isShared_1310_ = v_isSharedCheck_1320_;
goto v_resetjp_1308_;
}
v_resetjp_1308_:
{
lean_object* v___x_1312_; 
if (v_isShared_1300_ == 0)
{
lean_ctor_set(v___x_1299_, 1, v_val_1306_);
lean_ctor_set(v___x_1299_, 0, v_val_1301_);
v___x_1312_ = v___x_1299_;
goto v_reusejp_1311_;
}
else
{
lean_object* v_reuseFailAlloc_1319_; 
v_reuseFailAlloc_1319_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1319_, 0, v_val_1301_);
lean_ctor_set(v_reuseFailAlloc_1319_, 1, v_val_1306_);
v___x_1312_ = v_reuseFailAlloc_1319_;
goto v_reusejp_1311_;
}
v_reusejp_1311_:
{
lean_object* v___x_1314_; 
if (v_isShared_1305_ == 0)
{
lean_ctor_set(v___x_1304_, 1, v_neg_1307_);
lean_ctor_set(v___x_1304_, 0, v_neg_1302_);
v___x_1314_ = v___x_1304_;
goto v_reusejp_1313_;
}
else
{
lean_object* v_reuseFailAlloc_1318_; 
v_reuseFailAlloc_1318_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1318_, 0, v_neg_1302_);
lean_ctor_set(v_reuseFailAlloc_1318_, 1, v_neg_1307_);
v___x_1314_ = v_reuseFailAlloc_1318_;
goto v_reusejp_1313_;
}
v_reusejp_1313_:
{
lean_object* v___x_1316_; 
if (v_isShared_1310_ == 0)
{
lean_ctor_set(v___x_1309_, 1, v___x_1314_);
lean_ctor_set(v___x_1309_, 0, v___x_1312_);
v___x_1316_ = v___x_1309_;
goto v_reusejp_1315_;
}
else
{
lean_object* v_reuseFailAlloc_1317_; 
v_reuseFailAlloc_1317_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1317_, 0, v___x_1312_);
lean_ctor_set(v_reuseFailAlloc_1317_, 1, v___x_1314_);
v___x_1316_ = v_reuseFailAlloc_1317_;
goto v_reusejp_1315_;
}
v_reusejp_1315_:
{
return v___x_1316_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodAddUnits___lam__1(lean_object* v___f_1323_, lean_object* v___f_1324_, lean_object* v___y_1325_){
_start:
{
lean_object* v___x_78__overap_1326_; lean_object* v___x_1327_; 
v___x_78__overap_1326_ = lp_mathlib_AddMonoidHom_prod___redArg(v___f_1323_, v___f_1324_);
v___x_1327_ = lean_apply_1(v___x_78__overap_1326_, v___y_1325_);
return v___x_1327_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodAddUnits(lean_object* v_M_1339_, lean_object* v_N_1340_, lean_object* v_inst_1341_, lean_object* v_inst_1342_){
_start:
{
lean_object* v___x_1343_; 
v___x_1343_ = ((lean_object*)(lp_mathlib_AddEquiv_prodAddUnits___closed__4));
return v___x_1343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodAddUnits___boxed(lean_object* v_M_1344_, lean_object* v_N_1345_, lean_object* v_inst_1346_, lean_object* v_inst_1347_){
_start:
{
lean_object* v_res_1348_; 
v_res_1348_ = lp_mathlib_AddEquiv_prodAddUnits(v_M_1344_, v_N_1345_, v_inst_1346_, v_inst_1347_);
lean_dec_ref(v_inst_1347_);
lean_dec_ref(v_inst_1346_);
return v_res_1348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_embedProduct___lam__0(lean_object* v_x_1349_){
_start:
{
lean_object* v_val_1350_; lean_object* v_inv_1351_; lean_object* v___x_1353_; uint8_t v_isShared_1354_; uint8_t v_isSharedCheck_1358_; 
v_val_1350_ = lean_ctor_get(v_x_1349_, 0);
v_inv_1351_ = lean_ctor_get(v_x_1349_, 1);
v_isSharedCheck_1358_ = !lean_is_exclusive(v_x_1349_);
if (v_isSharedCheck_1358_ == 0)
{
v___x_1353_ = v_x_1349_;
v_isShared_1354_ = v_isSharedCheck_1358_;
goto v_resetjp_1352_;
}
else
{
lean_inc(v_inv_1351_);
lean_inc(v_val_1350_);
lean_dec(v_x_1349_);
v___x_1353_ = lean_box(0);
v_isShared_1354_ = v_isSharedCheck_1358_;
goto v_resetjp_1352_;
}
v_resetjp_1352_:
{
lean_object* v___x_1356_; 
if (v_isShared_1354_ == 0)
{
v___x_1356_ = v___x_1353_;
goto v_reusejp_1355_;
}
else
{
lean_object* v_reuseFailAlloc_1357_; 
v_reuseFailAlloc_1357_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1357_, 0, v_val_1350_);
lean_ctor_set(v_reuseFailAlloc_1357_, 1, v_inv_1351_);
v___x_1356_ = v_reuseFailAlloc_1357_;
goto v_reusejp_1355_;
}
v_reusejp_1355_:
{
return v___x_1356_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_embedProduct(lean_object* v_00_u03b1_1360_, lean_object* v_inst_1361_){
_start:
{
lean_object* v___f_1362_; 
v___f_1362_ = ((lean_object*)(lp_mathlib_Units_embedProduct___closed__0));
return v___f_1362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Units_embedProduct___boxed(lean_object* v_00_u03b1_1363_, lean_object* v_inst_1364_){
_start:
{
lean_object* v_res_1365_; 
v_res_1365_ = lp_mathlib_Units_embedProduct(v_00_u03b1_1363_, v_inst_1364_);
lean_dec_ref(v_inst_1364_);
return v_res_1365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_embedProduct___lam__0(lean_object* v_x_1366_){
_start:
{
lean_object* v_val_1367_; lean_object* v_neg_1368_; lean_object* v___x_1370_; uint8_t v_isShared_1371_; uint8_t v_isSharedCheck_1375_; 
v_val_1367_ = lean_ctor_get(v_x_1366_, 0);
v_neg_1368_ = lean_ctor_get(v_x_1366_, 1);
v_isSharedCheck_1375_ = !lean_is_exclusive(v_x_1366_);
if (v_isSharedCheck_1375_ == 0)
{
v___x_1370_ = v_x_1366_;
v_isShared_1371_ = v_isSharedCheck_1375_;
goto v_resetjp_1369_;
}
else
{
lean_inc(v_neg_1368_);
lean_inc(v_val_1367_);
lean_dec(v_x_1366_);
v___x_1370_ = lean_box(0);
v_isShared_1371_ = v_isSharedCheck_1375_;
goto v_resetjp_1369_;
}
v_resetjp_1369_:
{
lean_object* v___x_1373_; 
if (v_isShared_1371_ == 0)
{
v___x_1373_ = v___x_1370_;
goto v_reusejp_1372_;
}
else
{
lean_object* v_reuseFailAlloc_1374_; 
v_reuseFailAlloc_1374_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1374_, 0, v_val_1367_);
lean_ctor_set(v_reuseFailAlloc_1374_, 1, v_neg_1368_);
v___x_1373_ = v_reuseFailAlloc_1374_;
goto v_reusejp_1372_;
}
v_reusejp_1372_:
{
return v___x_1373_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_embedProduct(lean_object* v_00_u03b1_1377_, lean_object* v_inst_1378_){
_start:
{
lean_object* v___f_1379_; 
v___f_1379_ = ((lean_object*)(lp_mathlib_AddUnits_embedProduct___closed__0));
return v___f_1379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddUnits_embedProduct___boxed(lean_object* v_00_u03b1_1380_, lean_object* v_inst_1381_){
_start:
{
lean_object* v_res_1382_; 
v_res_1382_ = lp_mathlib_AddUnits_embedProduct(v_00_u03b1_1380_, v_inst_1381_);
lean_dec_ref(v_inst_1381_);
return v_res_1382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mulMulHom___redArg___lam__0(lean_object* v_inst_1383_, lean_object* v_a_1384_){
_start:
{
lean_object* v_fst_1385_; lean_object* v_snd_1386_; lean_object* v___x_1387_; 
v_fst_1385_ = lean_ctor_get(v_a_1384_, 0);
lean_inc(v_fst_1385_);
v_snd_1386_ = lean_ctor_get(v_a_1384_, 1);
lean_inc(v_snd_1386_);
lean_dec_ref(v_a_1384_);
v___x_1387_ = lean_apply_2(v_inst_1383_, v_fst_1385_, v_snd_1386_);
return v___x_1387_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mulMulHom___redArg(lean_object* v_inst_1388_){
_start:
{
lean_object* v___f_1389_; 
v___f_1389_ = lean_alloc_closure((void*)(lp_mathlib_mulMulHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1389_, 0, v_inst_1388_);
return v___f_1389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mulMulHom(lean_object* v_00_u03b1_1390_, lean_object* v_inst_1391_){
_start:
{
lean_object* v___f_1392_; 
v___f_1392_ = lean_alloc_closure((void*)(lp_mathlib_mulMulHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1392_, 0, v_inst_1391_);
return v___f_1392_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addAddHom___redArg(lean_object* v_inst_1393_){
_start:
{
lean_object* v___f_1394_; 
v___f_1394_ = lean_alloc_closure((void*)(lp_mathlib_mulMulHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1394_, 0, v_inst_1393_);
return v___f_1394_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addAddHom(lean_object* v_00_u03b1_1395_, lean_object* v_inst_1396_){
_start:
{
lean_object* v___f_1397_; 
v___f_1397_ = lean_alloc_closure((void*)(lp_mathlib_mulMulHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1397_, 0, v_inst_1396_);
return v___f_1397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mulMonoidHom___redArg(lean_object* v_inst_1398_){
_start:
{
lean_object* v_toMul_1399_; lean_object* v___f_1400_; 
v_toMul_1399_ = lean_ctor_get(v_inst_1398_, 1);
lean_inc(v_toMul_1399_);
lean_dec_ref(v_inst_1398_);
v___f_1400_ = lean_alloc_closure((void*)(lp_mathlib_mulMulHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1400_, 0, v_toMul_1399_);
return v___f_1400_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mulMonoidHom(lean_object* v_00_u03b1_1401_, lean_object* v_inst_1402_){
_start:
{
lean_object* v___x_1403_; 
v___x_1403_ = lp_mathlib_mulMonoidHom___redArg(v_inst_1402_);
return v___x_1403_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addAddMonoidHom___redArg(lean_object* v_inst_1404_){
_start:
{
lean_object* v_toAdd_1405_; lean_object* v___f_1406_; 
v_toAdd_1405_ = lean_ctor_get(v_inst_1404_, 1);
lean_inc(v_toAdd_1405_);
lean_dec_ref(v_inst_1404_);
v___f_1406_ = lean_alloc_closure((void*)(lp_mathlib_mulMulHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1406_, 0, v_toAdd_1405_);
return v___f_1406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addAddMonoidHom(lean_object* v_00_u03b1_1407_, lean_object* v_inst_1408_){
_start:
{
lean_object* v___x_1409_; 
v___x_1409_ = lp_mathlib_addAddMonoidHom___redArg(v_inst_1408_);
return v___x_1409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_divMonoidHom___redArg___lam__0(lean_object* v_toDiv_1410_, lean_object* v_a_1411_){
_start:
{
lean_object* v_fst_1412_; lean_object* v_snd_1413_; lean_object* v___x_1414_; 
v_fst_1412_ = lean_ctor_get(v_a_1411_, 0);
lean_inc(v_fst_1412_);
v_snd_1413_ = lean_ctor_get(v_a_1411_, 1);
lean_inc(v_snd_1413_);
lean_dec_ref(v_a_1411_);
v___x_1414_ = lean_apply_2(v_toDiv_1410_, v_fst_1412_, v_snd_1413_);
return v___x_1414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_divMonoidHom___redArg(lean_object* v_inst_1415_){
_start:
{
lean_object* v_toDiv_1416_; lean_object* v___f_1417_; 
v_toDiv_1416_ = lean_ctor_get(v_inst_1415_, 2);
lean_inc(v_toDiv_1416_);
lean_dec_ref(v_inst_1415_);
v___f_1417_ = lean_alloc_closure((void*)(lp_mathlib_divMonoidHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1417_, 0, v_toDiv_1416_);
return v___f_1417_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_divMonoidHom(lean_object* v_00_u03b1_1418_, lean_object* v_inst_1419_){
_start:
{
lean_object* v___x_1420_; 
v___x_1420_ = lp_mathlib_divMonoidHom___redArg(v_inst_1419_);
return v___x_1420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_subAddMonoidHom___redArg___lam__0(lean_object* v_toSub_1421_, lean_object* v_a_1422_){
_start:
{
lean_object* v_fst_1423_; lean_object* v_snd_1424_; lean_object* v___x_1425_; 
v_fst_1423_ = lean_ctor_get(v_a_1422_, 0);
lean_inc(v_fst_1423_);
v_snd_1424_ = lean_ctor_get(v_a_1422_, 1);
lean_inc(v_snd_1424_);
lean_dec_ref(v_a_1422_);
v___x_1425_ = lean_apply_2(v_toSub_1421_, v_fst_1423_, v_snd_1424_);
return v___x_1425_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_subAddMonoidHom___redArg(lean_object* v_inst_1426_){
_start:
{
lean_object* v_toSub_1427_; lean_object* v___f_1428_; 
v_toSub_1427_ = lean_ctor_get(v_inst_1426_, 2);
lean_inc(v_toSub_1427_);
lean_dec_ref(v_inst_1426_);
v___f_1428_ = lean_alloc_closure((void*)(lp_mathlib_subAddMonoidHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1428_, 0, v_toSub_1427_);
return v___f_1428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_subAddMonoidHom(lean_object* v_00_u03b1_1429_, lean_object* v_inst_1430_){
_start:
{
lean_object* v___x_1431_; 
v___x_1431_ = lp_mathlib_subAddMonoidHom___redArg(v_inst_1430_);
return v___x_1431_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Opposite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_SelfInv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Torsion(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Units_Hom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Notation_Pi_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Notation_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_TermCongr(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Prod(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_SelfInv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Torsion(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Units_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Notation_Pi_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Notation_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_TermCongr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Prod(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_SelfInv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Torsion(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Units_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Notation_Pi_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Notation_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_TermCongr(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Prod(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_SelfInv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Torsion(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Units_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Notation_Pi_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Notation_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_TermCongr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Prod(builtin);
}
#ifdef __cplusplus
}
#endif
