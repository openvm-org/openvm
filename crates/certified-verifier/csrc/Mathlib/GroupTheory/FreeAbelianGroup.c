// Lean compiler output
// Module: Mathlib.GroupTheory.FreeAbelianGroup
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Module.NatInt public import Mathlib.GroupTheory.Abelianization.Defs public import Mathlib.GroupTheory.FreeGroup.Basic public import Mathlib.Control.Basic
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
lean_object* lp_mathlib_Additive_ofMul(lean_object*);
lean_object* lp_mathlib_FreeGroup_of___redArg(lean_object*);
lean_object* lp_mathlib_Additive_toMul(lean_object*);
lean_object* lp_mathlib_FreeGroup_instInv(lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* l_npowRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_zpowRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OneHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Int_instAddCommGroup;
lean_object* lp_mathlib_Multiplicative_divInvMonoid___redArg(lean_object*);
lean_object* lp_mathlib_FreeGroup_instGroup(lean_object*);
lean_object* lp_mathlib_instCommGroupAbelianization___redArg(lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_FreeGroup_lift___redArg(lean_object*);
lean_object* lp_mathlib_Abelianization_lift___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Multiplicative_mulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MonoidHom_toAdditive(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
uint8_t lean_int_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_abs(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lp_mathlib_FreeGroup_invRev___redArg(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lp_mathlib_Multiplicative_toAdd(lean_object*);
lean_object* lp_mathlib_Multiplicative_ofAdd(lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* lean_nat_land(lean_object*, lean_object*);
lean_object* lp_mathlib_npowBinRec_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_npowBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toAddCommGroup___redArg(lean_object*);
lean_object* l_Function_const___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalRing_toNonUnitalSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Nat_unaryCast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Int_castDef___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_DivInvMonoid_div_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedFreeAbelianGroup___aux__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedFreeAbelianGroup(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___lam__0(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__8___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_List_appendTR___redArg, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__8___redArg___closed__0 = (const lean_object*)&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__8___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__8___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__8___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__8(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__8___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__12___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__12___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__12___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__12(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__14___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_npowBinRecAuto___boxed, .m_arity = 5, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__8___redArg___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__14___redArg___closed__0 = (const lean_object*)&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__14___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__14___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__8___redArg___closed__0_value),((lean_object*)&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__14___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__14___redArg___closed__1 = (const lean_object*)&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__14___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__14___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__14___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__14___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__14(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__16___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_npowRec___boxed, .m_arity = 5, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__8___redArg___closed__0_value)} };
static const lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__16___redArg___closed__0 = (const lean_object*)&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__16___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__16___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__16___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__16(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__16___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_map_u2082___at___00instAddCommGroupFreeAbelianGroup_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_map_u2082___at___00instAddCommGroupFreeAbelianGroup_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_div_x27___at___00instAddCommGroupFreeAbelianGroup_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_div_x27___at___00instAddCommGroupFreeAbelianGroup_spec__2(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_zpowRec___at___00instAddCommGroupFreeAbelianGroup_spec__4___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_zpowRec___at___00instAddCommGroupFreeAbelianGroup_spec__4___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_zpowRec___at___00instAddCommGroupFreeAbelianGroup_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zpowRec___at___00instAddCommGroupFreeAbelianGroup_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zpowRec___at___00instAddCommGroupFreeAbelianGroup_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zpowRec___at___00instAddCommGroupFreeAbelianGroup_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec___at___00npowBinRec_go___at___00npowBinRec___at___00instAddCommGroupFreeAbelianGroup_spec__1_spec__1_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___at___00npowBinRec___at___00instAddCommGroupFreeAbelianGroup_spec__1_spec__1___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___at___00npowBinRec___at___00instAddCommGroupFreeAbelianGroup_spec__1_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_npowBinRec_go___at___00npowBinRec___at___00instAddCommGroupFreeAbelianGroup_spec__1_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_npowBinRec_go___at___00npowBinRec___at___00instAddCommGroupFreeAbelianGroup_spec__1_spec__1___redArg___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_npowBinRec_go___at___00npowBinRec___at___00instAddCommGroupFreeAbelianGroup_spec__1_spec__1___redArg___closed__0 = (const lean_object*)&lp_mathlib_npowBinRec_go___at___00npowBinRec___at___00instAddCommGroupFreeAbelianGroup_spec__1_spec__1___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___at___00npowBinRec___at___00instAddCommGroupFreeAbelianGroup_spec__1_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec___at___00instAddCommGroupFreeAbelianGroup_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___lam__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowRec___at___00instAddCommGroupFreeAbelianGroup_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowRec___at___00instAddCommGroupFreeAbelianGroup_spec__3___redArg___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instAddCommGroupFreeAbelianGroup___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_npowRec___at___00instAddCommGroupFreeAbelianGroup_spec__3___redArg___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___closed__0 = (const lean_object*)&lp_mathlib_instAddCommGroupFreeAbelianGroup___closed__0_value;
static const lean_closure_object lp_mathlib_instAddCommGroupFreeAbelianGroup___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___closed__1 = (const lean_object*)&lp_mathlib_instAddCommGroupFreeAbelianGroup___closed__1_value;
static const lean_closure_object lp_mathlib_instAddCommGroupFreeAbelianGroup___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instAddCommGroupFreeAbelianGroup___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___closed__2 = (const lean_object*)&lp_mathlib_instAddCommGroupFreeAbelianGroup___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec___at___00instAddCommGroupFreeAbelianGroup_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowRec___at___00instAddCommGroupFreeAbelianGroup_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowRec___at___00instAddCommGroupFreeAbelianGroup_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___at___00npowBinRec___at___00instAddCommGroupFreeAbelianGroup_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec___at___00npowBinRec_go___at___00npowBinRec___at___00instAddCommGroupFreeAbelianGroup_spec__1_spec__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_instUniqueFreeAbelianGroupOfIsEmpty___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instUniqueFreeAbelianGroupOfIsEmpty___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_instUniqueFreeAbelianGroupOfIsEmpty(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_of___at___00FreeAbelianGroup_of_spec__0___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_of___at___00FreeAbelianGroup_of_spec__0___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Abelianization_of___at___00FreeAbelianGroup_of_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Abelianization_of___at___00FreeAbelianGroup_of_spec__0___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Abelianization_of___at___00FreeAbelianGroup_of_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Abelianization_of___at___00FreeAbelianGroup_of_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_of___at___00FreeAbelianGroup_of_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_of___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_of(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_FreeAbelianGroup_lift___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeAbelianGroup_lift___redArg___closed__0;
static lean_once_cell_t lp_mathlib_FreeAbelianGroup_lift___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeAbelianGroup_lift___redArg___closed__1;
static lean_once_cell_t lp_mathlib_FreeAbelianGroup_lift___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeAbelianGroup_lift___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_lift___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_lift(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_liftAddEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_liftAddEquiv(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_liftAddGroupHom___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_liftAddGroupHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_liftAddGroupHom(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__0;
static lean_once_cell_t lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_instMonad___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_instMonad___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_instMonad___lam__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_instMonad___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_instMonad___lam__7(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_instMonad___lam__7___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_instMonad___lam__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_instMonad___lam__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_instMonad___lam__9(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_instMonad___lam__9___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_instMonad___lam__10(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_FreeAbelianGroup_instMonad___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeAbelianGroup_of___redArg, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeAbelianGroup_instMonad___closed__0 = (const lean_object*)&lp_mathlib_FreeAbelianGroup_instMonad___closed__0_value;
static const lean_closure_object lp_mathlib_FreeAbelianGroup_instMonad___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeAbelianGroup_instMonad___lam__0, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_FreeAbelianGroup_instMonad___closed__0_value)} };
static const lean_object* lp_mathlib_FreeAbelianGroup_instMonad___closed__1 = (const lean_object*)&lp_mathlib_FreeAbelianGroup_instMonad___closed__1_value;
static const lean_closure_object lp_mathlib_FreeAbelianGroup_instMonad___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeAbelianGroup_instMonad___lam__1, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_FreeAbelianGroup_instMonad___closed__0_value)} };
static const lean_object* lp_mathlib_FreeAbelianGroup_instMonad___closed__2 = (const lean_object*)&lp_mathlib_FreeAbelianGroup_instMonad___closed__2_value;
static const lean_closure_object lp_mathlib_FreeAbelianGroup_instMonad___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeAbelianGroup_instMonad___lam__2, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeAbelianGroup_instMonad___closed__3 = (const lean_object*)&lp_mathlib_FreeAbelianGroup_instMonad___closed__3_value;
static const lean_closure_object lp_mathlib_FreeAbelianGroup_instMonad___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeAbelianGroup_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeAbelianGroup_instMonad___closed__4 = (const lean_object*)&lp_mathlib_FreeAbelianGroup_instMonad___closed__4_value;
static const lean_closure_object lp_mathlib_FreeAbelianGroup_instMonad___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeAbelianGroup_instMonad___lam__5, .m_arity = 6, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_FreeAbelianGroup_instMonad___closed__0_value),((lean_object*)&lp_mathlib_FreeAbelianGroup_instMonad___closed__4_value)} };
static const lean_object* lp_mathlib_FreeAbelianGroup_instMonad___closed__5 = (const lean_object*)&lp_mathlib_FreeAbelianGroup_instMonad___closed__5_value;
static const lean_closure_object lp_mathlib_FreeAbelianGroup_instMonad___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeAbelianGroup_instMonad___lam__8, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_FreeAbelianGroup_instMonad___closed__4_value)} };
static const lean_object* lp_mathlib_FreeAbelianGroup_instMonad___closed__6 = (const lean_object*)&lp_mathlib_FreeAbelianGroup_instMonad___closed__6_value;
static const lean_closure_object lp_mathlib_FreeAbelianGroup_instMonad___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeAbelianGroup_instMonad___lam__10, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeAbelianGroup_instMonad___closed__7 = (const lean_object*)&lp_mathlib_FreeAbelianGroup_instMonad___closed__7_value;
static const lean_ctor_object lp_mathlib_FreeAbelianGroup_instMonad___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_FreeAbelianGroup_instMonad___closed__1_value),((lean_object*)&lp_mathlib_FreeAbelianGroup_instMonad___closed__2_value)}};
static const lean_object* lp_mathlib_FreeAbelianGroup_instMonad___closed__8 = (const lean_object*)&lp_mathlib_FreeAbelianGroup_instMonad___closed__8_value;
static const lean_ctor_object lp_mathlib_FreeAbelianGroup_instMonad___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_FreeAbelianGroup_instMonad___closed__8_value),((lean_object*)&lp_mathlib_FreeAbelianGroup_instMonad___closed__3_value),((lean_object*)&lp_mathlib_FreeAbelianGroup_instMonad___closed__5_value),((lean_object*)&lp_mathlib_FreeAbelianGroup_instMonad___closed__6_value),((lean_object*)&lp_mathlib_FreeAbelianGroup_instMonad___closed__7_value)}};
static const lean_object* lp_mathlib_FreeAbelianGroup_instMonad___closed__9 = (const lean_object*)&lp_mathlib_FreeAbelianGroup_instMonad___closed__9_value;
static const lean_ctor_object lp_mathlib_FreeAbelianGroup_instMonad___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_FreeAbelianGroup_instMonad___closed__9_value),((lean_object*)&lp_mathlib_FreeAbelianGroup_instMonad___closed__4_value)}};
static const lean_object* lp_mathlib_FreeAbelianGroup_instMonad___closed__10 = (const lean_object*)&lp_mathlib_FreeAbelianGroup_instMonad___closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_FreeAbelianGroup_instMonad = (const lean_object*)&lp_mathlib_FreeAbelianGroup_instMonad___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27___at___00FreeAbelianGroup_seqAddGroupHom_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27___at___00FreeAbelianGroup_seqAddGroupHom_spec__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27___at___00FreeAbelianGroup_seqAddGroupHom_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27___at___00FreeAbelianGroup_seqAddGroupHom_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_seqAddGroupHom___redArg___lam__0(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__4___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__4___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__5_spec__8___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__5_spec__8___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__5___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0___closed__0 = (const lean_object*)&lp_mathlib_FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0___closed__0_value;
static const lean_closure_object lp_mathlib_FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2___redArg, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0___closed__1 = (const lean_object*)&lp_mathlib_FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0___closed__1_value;
static const lean_ctor_object lp_mathlib_FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0___closed__1_value),((lean_object*)&lp_mathlib_FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0___closed__0_value)}};
static const lean_object* lp_mathlib_FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0___closed__2 = (const lean_object*)&lp_mathlib_FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_con___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__6_spec__13(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__6_spec__14___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__6_spec__14___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_copy___at___00Subgroup_closure___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__5_spec__11(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iInf___at___00Subgroup_closure___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__5_spec__9(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iInf___at___00Subgroup_closure___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__5_spec__9___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_closure___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__5___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_closure___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__5___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iInf___at___00Subgroup_closure___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__5_spec__10(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_iInf___at___00Subgroup_closure___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__5_spec__10___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_closure___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1___closed__0 = (const lean_object*)&lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1___closed__0_value;
static const lean_closure_object lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1___closed__1 = (const lean_object*)&lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1___closed__1_value;
static const lean_ctor_object lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1___closed__0_value),((lean_object*)&lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1___closed__1_value)}};
static const lean_object* lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1___closed__2 = (const lean_object*)&lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditive___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__2___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditive___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__2___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_MonoidHom_toAdditive___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MonoidHom_toAdditive___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__2___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MonoidHom_toAdditive___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__2___closed__0 = (const lean_object*)&lp_mathlib_MonoidHom_toAdditive___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__2___closed__0_value;
static const lean_closure_object lp_mathlib_MonoidHom_toAdditive___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MonoidHom_toAdditive___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__2___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MonoidHom_toAdditive___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__2___closed__1 = (const lean_object*)&lp_mathlib_MonoidHom_toAdditive___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__2___closed__1_value;
static const lean_ctor_object lp_mathlib_MonoidHom_toAdditive___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_MonoidHom_toAdditive___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__2___closed__0_value),((lean_object*)&lp_mathlib_MonoidHom_toAdditive___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__2___closed__1_value)}};
static const lean_object* lp_mathlib_MonoidHom_toAdditive___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__2___closed__2 = (const lean_object*)&lp_mathlib_MonoidHom_toAdditive___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__2___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditive___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__2(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0___closed__0;
static lean_once_cell_t lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0___closed__1;
static lean_once_cell_t lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0___closed__2;
static lean_once_cell_t lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0___closed__3;
static lean_once_cell_t lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_FreeAbelianGroup_seqAddGroupHom___redArg___lam__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeAbelianGroup_seqAddGroupHom___redArg___lam__1___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_seqAddGroupHom___redArg___lam__1(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_FreeAbelianGroup_seqAddGroupHom___redArg___lam__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeAbelianGroup_seqAddGroupHom___redArg___lam__2___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_seqAddGroupHom___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_seqAddGroupHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_seqAddGroupHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__3___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__3___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__6_spec__14(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__5_spec__8(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__5_spec__8___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_liftOn_x27___at___00Con_liftOn___at___00Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__6_spec__14_spec__16_spec__17___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_liftOn_x27___at___00Con_liftOn___at___00Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__6_spec__14_spec__16_spec__17(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_liftOn___at___00Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__6_spec__14_spec__16___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_liftOn___at___00Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__6_spec__14_spec__16(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_map___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__1_spec__4_spec__7___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__1_spec__4___redArg(lean_object*);
static const lean_closure_object lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__1_spec__4___redArg, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__1___closed__0 = (const lean_object*)&lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__1___closed__0_value),((lean_object*)&lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1___closed__1_value)}};
static const lean_object* lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__1___closed__1 = (const lean_object*)&lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditive___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0_spec__1_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0_spec__1___redArg, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0___closed__0 = (const lean_object*)&lp_mathlib_FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0___closed__0_value;
static const lean_ctor_object lp_mathlib_FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0___closed__0_value),((lean_object*)&lp_mathlib_FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0___closed__0_value)}};
static const lean_object* lp_mathlib_FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0___closed__1 = (const lean_object*)&lp_mathlib_FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0___closed__0;
static lean_once_cell_t lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0___closed__1;
static lean_once_cell_t lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0___closed__2;
static lean_once_cell_t lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0___closed__3;
static lean_once_cell_t lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_FreeAbelianGroup_map___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeAbelianGroup_map___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_map___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_map(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0_spec__2___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__1_spec__4_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_liftOn_x27___at___00Con_liftOn___at___00Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__1_spec__4_spec__7_spec__8_spec__9___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Quotient_liftOn_x27___at___00Con_liftOn___at___00Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__1_spec__4_spec__7_spec__8_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_liftOn___at___00Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__1_spec__4_spec__7_spec__8___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_liftOn___at___00Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__1_spec__4_spec__7_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_mul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_mul___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_mul___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_mul___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_FreeAbelianGroup_mul___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeAbelianGroup_mul___redArg___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeAbelianGroup_mul___redArg___closed__0 = (const lean_object*)&lp_mathlib_FreeAbelianGroup_mul___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_mul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_mul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_distrib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_distrib(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_nonUnitalNonAssocRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_nonUnitalNonAssocRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_one___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_one(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_nonUnitalRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_nonUnitalRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_ring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_ring(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_FreeAbelianGroup_ofMulHom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeAbelianGroup_of, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_FreeAbelianGroup_ofMulHom___closed__0 = (const lean_object*)&lp_mathlib_FreeAbelianGroup_ofMulHom___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_ofMulHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_ofMulHom___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_liftMonoid___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_liftMonoid___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_liftMonoid___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_liftMonoid___redArg___lam__3(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_FreeAbelianGroup_liftMonoid___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeAbelianGroup_liftMonoid___redArg___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeAbelianGroup_liftMonoid___redArg___closed__0 = (const lean_object*)&lp_mathlib_FreeAbelianGroup_liftMonoid___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_liftMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_liftMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_liftMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_liftMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_instCommRingOfCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_instCommRingOfCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___lam__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___lam__1___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___lam__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___lam__2(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___closed__0;
static const lean_closure_object lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___closed__1 = (const lean_object*)&lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_uniqueEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_equivOfEquiv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_equivOfEquiv___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_equivOfEquiv___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_equivOfEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_equivOfEquiv(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lp_mathlib_Additive_ofMul(lean_box(0));
return v___x_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedFreeAbelianGroup___aux__1(lean_object* v_00_u03b1_2_){
_start:
{
lean_object* v___x_3_; lean_object* v_toFun_4_; lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_3_ = lean_obj_once(&lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0, &lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0_once, _init_lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0);
v_toFun_4_ = lean_ctor_get(v___x_3_, 0);
v___x_5_ = lean_box(0);
lean_inc(v_toFun_4_);
v___x_6_ = lean_apply_1(v_toFun_4_, v___x_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedFreeAbelianGroup(lean_object* v_00_u03b1_7_){
_start:
{
lean_object* v___x_8_; lean_object* v_toFun_9_; lean_object* v___x_10_; lean_object* v___x_11_; 
v___x_8_ = lean_obj_once(&lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0, &lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0_once, _init_lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0);
v_toFun_9_ = lean_ctor_get(v___x_8_, 0);
v___x_10_ = lean_box(0);
lean_inc(v_toFun_9_);
v___x_11_ = lean_apply_1(v_toFun_9_, v___x_10_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__1(lean_object* v_00_u03b1_12_){
_start:
{
lean_object* v___x_13_; lean_object* v_toFun_14_; lean_object* v___x_15_; lean_object* v___x_16_; 
v___x_13_ = lean_obj_once(&lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0, &lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0_once, _init_lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0);
v_toFun_14_ = lean_ctor_get(v___x_13_, 0);
v___x_15_ = lean_box(0);
lean_inc(v_toFun_14_);
v___x_16_ = lean_apply_1(v_toFun_14_, v___x_15_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___lam__0(lean_object* v_self_17_, lean_object* v___y_18_){
_start:
{
lean_object* v_toFun_19_; lean_object* v___x_20_; 
v_toFun_19_ = lean_ctor_get(v_self_17_, 0);
lean_inc(v_toFun_19_);
lean_dec_ref(v_self_17_);
v___x_20_ = lean_apply_1(v_toFun_19_, v___y_18_);
return v___x_20_;
}
}
static lean_object* _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0(void){
_start:
{
lean_object* v___x_21_; 
v___x_21_ = lp_mathlib_Additive_toMul(lean_box(0));
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg(lean_object* v_x_22_, lean_object* v_y_23_){
_start:
{
lean_object* v___x_24_; lean_object* v_toFun_25_; lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v___x_28_; lean_object* v___x_29_; lean_object* v___x_30_; 
v___x_24_ = lean_obj_once(&lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0, &lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0_once, _init_lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0);
v_toFun_25_ = lean_ctor_get(v___x_24_, 0);
v___x_26_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0);
v___x_27_ = lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___lam__0(v___x_26_, v_x_22_);
v___x_28_ = lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___lam__0(v___x_26_, v_y_23_);
v___x_29_ = l_List_appendTR___redArg(v___x_27_, v___x_28_);
lean_inc(v_toFun_25_);
v___x_30_ = lean_apply_1(v_toFun_25_, v___x_29_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3(lean_object* v_00_u03b1_31_, lean_object* v_x_32_, lean_object* v_y_33_){
_start:
{
lean_object* v___x_34_; lean_object* v_toFun_35_; lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; 
v___x_34_ = lean_obj_once(&lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0, &lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0_once, _init_lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0);
v_toFun_35_ = lean_ctor_get(v___x_34_, 0);
v___x_36_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0);
v___x_37_ = lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___lam__0(v___x_36_, v_x_32_);
v___x_38_ = lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___lam__0(v___x_36_, v_y_33_);
v___x_39_ = l_List_appendTR___redArg(v___x_37_, v___x_38_);
lean_inc(v_toFun_35_);
v___x_40_ = lean_apply_1(v_toFun_35_, v___x_39_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__8___redArg(lean_object* v_n_42_, lean_object* v_a_43_){
_start:
{
lean_object* v___x_44_; lean_object* v_toFun_45_; lean_object* v___x_46_; lean_object* v_toFun_47_; lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___f_50_; lean_object* v___x_51_; lean_object* v___x_52_; 
v___x_44_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0);
v_toFun_45_ = lean_ctor_get(v___x_44_, 0);
v___x_46_ = lean_obj_once(&lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0, &lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0_once, _init_lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0);
v_toFun_47_ = lean_ctor_get(v___x_46_, 0);
lean_inc(v_toFun_45_);
v___x_48_ = lean_apply_1(v_toFun_45_, v_a_43_);
v___x_49_ = lean_box(0);
v___f_50_ = ((lean_object*)(lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__8___redArg___closed__0));
v___x_51_ = lp_mathlib_npowBinRec_go___redArg(v___f_50_, v_n_42_, v___x_49_, v___x_48_);
lean_inc(v_toFun_47_);
v___x_52_ = lean_apply_1(v_toFun_47_, v___x_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__8___redArg___boxed(lean_object* v_n_53_, lean_object* v_a_54_){
_start:
{
lean_object* v_res_55_; 
v_res_55_ = lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__8___redArg(v_n_53_, v_a_54_);
lean_dec(v_n_53_);
return v_res_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__8(lean_object* v_00_u03b1_56_, lean_object* v_n_57_, lean_object* v_a_58_){
_start:
{
lean_object* v___x_59_; lean_object* v_toFun_60_; lean_object* v___x_61_; lean_object* v_toFun_62_; lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___f_65_; lean_object* v___x_66_; lean_object* v___x_67_; 
v___x_59_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0);
v_toFun_60_ = lean_ctor_get(v___x_59_, 0);
v___x_61_ = lean_obj_once(&lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0, &lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0_once, _init_lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0);
v_toFun_62_ = lean_ctor_get(v___x_61_, 0);
lean_inc(v_toFun_60_);
v___x_63_ = lean_apply_1(v_toFun_60_, v_a_58_);
v___x_64_ = lean_box(0);
v___f_65_ = ((lean_object*)(lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__8___redArg___closed__0));
v___x_66_ = lp_mathlib_npowBinRec_go___redArg(v___f_65_, v_n_57_, v___x_64_, v___x_63_);
lean_inc(v_toFun_62_);
v___x_67_ = lean_apply_1(v_toFun_62_, v___x_66_);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__8___boxed(lean_object* v_00_u03b1_68_, lean_object* v_n_69_, lean_object* v_a_70_){
_start:
{
lean_object* v_res_71_; 
v_res_71_ = lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__8(v_00_u03b1_68_, v_n_69_, v_a_70_);
lean_dec(v_n_69_);
return v_res_71_;
}
}
static lean_object* _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__12___redArg___closed__0(void){
_start:
{
lean_object* v___x_72_; 
v___x_72_ = lp_mathlib_Multiplicative_ofAdd(lean_box(0));
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__12___redArg(lean_object* v_x_73_){
_start:
{
lean_object* v___x_74_; lean_object* v_toFun_75_; lean_object* v___x_76_; lean_object* v_toFun_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; 
v___x_74_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0);
v_toFun_75_ = lean_ctor_get(v___x_74_, 0);
v___x_76_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__12___redArg___closed__0, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__12___redArg___closed__0_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__12___redArg___closed__0);
v_toFun_77_ = lean_ctor_get(v___x_76_, 0);
lean_inc(v_toFun_75_);
v___x_78_ = lean_apply_1(v_toFun_75_, v_x_73_);
v___x_79_ = lp_mathlib_FreeGroup_invRev___redArg(v___x_78_);
lean_inc(v_toFun_77_);
v___x_80_ = lean_apply_1(v_toFun_77_, v___x_79_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__12(lean_object* v_00_u03b1_81_, lean_object* v_x_82_){
_start:
{
lean_object* v___x_83_; lean_object* v_toFun_84_; lean_object* v___x_85_; lean_object* v_toFun_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; 
v___x_83_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0);
v_toFun_84_ = lean_ctor_get(v___x_83_, 0);
v___x_85_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__12___redArg___closed__0, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__12___redArg___closed__0_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__12___redArg___closed__0);
v_toFun_86_ = lean_ctor_get(v___x_85_, 0);
lean_inc(v_toFun_84_);
v___x_87_ = lean_apply_1(v_toFun_84_, v_x_82_);
v___x_88_ = lp_mathlib_FreeGroup_invRev___redArg(v___x_87_);
lean_inc(v_toFun_86_);
v___x_89_ = lean_apply_1(v_toFun_86_, v___x_88_);
return v___x_89_;
}
}
static lean_object* _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__14___redArg___closed__2(void){
_start:
{
lean_object* v___x_97_; 
v___x_97_ = lp_mathlib_FreeGroup_instInv(lean_box(0));
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__14___redArg(lean_object* v_x_98_, lean_object* v_y_99_){
_start:
{
lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v_toFun_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; 
v___x_100_ = lean_obj_once(&lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0, &lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0_once, _init_lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0);
v___x_101_ = ((lean_object*)(lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__14___redArg___closed__1));
v_toFun_102_ = lean_ctor_get(v___x_100_, 0);
v___x_103_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0);
v___x_104_ = lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___lam__0(v___x_103_, v_x_98_);
v___x_105_ = lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___lam__0(v___x_103_, v_y_99_);
v___x_106_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__14___redArg___closed__2, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__14___redArg___closed__2_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__14___redArg___closed__2);
v___x_107_ = lp_mathlib_DivInvMonoid_div_x27___redArg(v___x_101_, v___x_106_, v___x_104_, v___x_105_);
lean_inc(v_toFun_102_);
v___x_108_ = lean_apply_1(v_toFun_102_, v___x_107_);
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__14(lean_object* v_00_u03b1_109_, lean_object* v_x_110_, lean_object* v_y_111_){
_start:
{
lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v_toFun_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; 
v___x_112_ = lean_obj_once(&lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0, &lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0_once, _init_lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0);
v___x_113_ = ((lean_object*)(lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__14___redArg___closed__1));
v_toFun_114_ = lean_ctor_get(v___x_112_, 0);
v___x_115_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0);
v___x_116_ = lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___lam__0(v___x_115_, v_x_110_);
v___x_117_ = lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___lam__0(v___x_115_, v_y_111_);
v___x_118_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__14___redArg___closed__2, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__14___redArg___closed__2_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__14___redArg___closed__2);
v___x_119_ = lp_mathlib_DivInvMonoid_div_x27___redArg(v___x_113_, v___x_118_, v___x_116_, v___x_117_);
lean_inc(v_toFun_114_);
v___x_120_ = lean_apply_1(v_toFun_114_, v___x_119_);
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__16___redArg(lean_object* v_n_124_, lean_object* v_a_125_){
_start:
{
lean_object* v___x_126_; lean_object* v_toFun_127_; lean_object* v___x_128_; lean_object* v_toFun_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; 
v___x_126_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0);
v_toFun_127_ = lean_ctor_get(v___x_126_, 0);
v___x_128_ = lean_obj_once(&lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0, &lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0_once, _init_lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0);
v_toFun_129_ = lean_ctor_get(v___x_128_, 0);
lean_inc(v_toFun_127_);
v___x_130_ = lean_apply_1(v_toFun_127_, v_a_125_);
v___x_131_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__14___redArg___closed__2, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__14___redArg___closed__2_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__14___redArg___closed__2);
v___x_132_ = ((lean_object*)(lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__16___redArg___closed__0));
v___x_133_ = lp_mathlib_zpowRec___redArg(v___x_131_, v___x_132_, v_n_124_, v___x_130_);
lean_inc(v_toFun_129_);
v___x_134_ = lean_apply_1(v_toFun_129_, v___x_133_);
return v___x_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__16___redArg___boxed(lean_object* v_n_135_, lean_object* v_a_136_){
_start:
{
lean_object* v_res_137_; 
v_res_137_ = lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__16___redArg(v_n_135_, v_a_136_);
lean_dec(v_n_135_);
return v_res_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__16(lean_object* v_00_u03b1_138_, lean_object* v_n_139_, lean_object* v_a_140_){
_start:
{
lean_object* v___x_141_; lean_object* v_toFun_142_; lean_object* v___x_143_; lean_object* v_toFun_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; 
v___x_141_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0);
v_toFun_142_ = lean_ctor_get(v___x_141_, 0);
v___x_143_ = lean_obj_once(&lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0, &lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0_once, _init_lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0);
v_toFun_144_ = lean_ctor_get(v___x_143_, 0);
lean_inc(v_toFun_142_);
v___x_145_ = lean_apply_1(v_toFun_142_, v_a_140_);
v___x_146_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__14___redArg___closed__2, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__14___redArg___closed__2_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__14___redArg___closed__2);
v___x_147_ = ((lean_object*)(lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__16___redArg___closed__0));
v___x_148_ = lp_mathlib_zpowRec___redArg(v___x_146_, v___x_147_, v_n_139_, v___x_145_);
lean_inc(v_toFun_144_);
v___x_149_ = lean_apply_1(v_toFun_144_, v___x_148_);
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__16___boxed(lean_object* v_00_u03b1_150_, lean_object* v_n_151_, lean_object* v_a_152_){
_start:
{
lean_object* v_res_153_; 
v_res_153_ = lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__16(v_00_u03b1_150_, v_n_151_, v_a_152_);
lean_dec(v_n_151_);
return v_res_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_map_u2082___at___00instAddCommGroupFreeAbelianGroup_spec__0___redArg(lean_object* v_f_154_, lean_object* v_q_u2081_155_, lean_object* v_q_u2082_156_){
_start:
{
lean_object* v___x_157_; 
v___x_157_ = lean_apply_2(v_f_154_, v_q_u2081_155_, v_q_u2082_156_);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_map_u2082___at___00instAddCommGroupFreeAbelianGroup_spec__0(lean_object* v_f_158_, lean_object* v_h_159_, lean_object* v_q_u2081_160_, lean_object* v_q_u2082_161_){
_start:
{
lean_object* v___x_162_; 
v___x_162_ = lean_apply_2(v_f_158_, v_q_u2081_160_, v_q_u2082_161_);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_div_x27___at___00instAddCommGroupFreeAbelianGroup_spec__2___redArg(lean_object* v_a_163_, lean_object* v_b_164_){
_start:
{
lean_object* v___x_165_; lean_object* v___x_166_; 
v___x_165_ = lp_mathlib_FreeGroup_invRev___redArg(v_b_164_);
v___x_166_ = l_List_appendTR___redArg(v_a_163_, v___x_165_);
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivInvMonoid_div_x27___at___00instAddCommGroupFreeAbelianGroup_spec__2(lean_object* v_00_u03b1_167_, lean_object* v_a_168_, lean_object* v_b_169_){
_start:
{
lean_object* v___x_170_; 
v___x_170_ = lp_mathlib_DivInvMonoid_div_x27___at___00instAddCommGroupFreeAbelianGroup_spec__2___redArg(v_a_168_, v_b_169_);
return v___x_170_;
}
}
static lean_object* _init_lp_mathlib_zpowRec___at___00instAddCommGroupFreeAbelianGroup_spec__4___redArg___closed__0(void){
_start:
{
lean_object* v_natZero_171_; lean_object* v_intZero_172_; 
v_natZero_171_ = lean_unsigned_to_nat(0u);
v_intZero_172_ = lean_nat_to_int(v_natZero_171_);
return v_intZero_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zpowRec___at___00instAddCommGroupFreeAbelianGroup_spec__4___redArg(lean_object* v_npow_173_, lean_object* v_x_174_, lean_object* v_x_175_){
_start:
{
lean_object* v_intZero_176_; uint8_t v_isNeg_177_; 
v_intZero_176_ = lean_obj_once(&lp_mathlib_zpowRec___at___00instAddCommGroupFreeAbelianGroup_spec__4___redArg___closed__0, &lp_mathlib_zpowRec___at___00instAddCommGroupFreeAbelianGroup_spec__4___redArg___closed__0_once, _init_lp_mathlib_zpowRec___at___00instAddCommGroupFreeAbelianGroup_spec__4___redArg___closed__0);
v_isNeg_177_ = lean_int_dec_lt(v_x_174_, v_intZero_176_);
if (v_isNeg_177_ == 0)
{
lean_object* v_a_178_; lean_object* v___x_179_; 
v_a_178_ = lean_nat_abs(v_x_174_);
v___x_179_ = lean_apply_2(v_npow_173_, v_a_178_, v_x_175_);
return v___x_179_;
}
else
{
lean_object* v_abs_180_; lean_object* v_one_181_; lean_object* v_a_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; 
v_abs_180_ = lean_nat_abs(v_x_174_);
v_one_181_ = lean_unsigned_to_nat(1u);
v_a_182_ = lean_nat_sub(v_abs_180_, v_one_181_);
lean_dec(v_abs_180_);
v___x_183_ = lean_nat_add(v_a_182_, v_one_181_);
lean_dec(v_a_182_);
v___x_184_ = lean_apply_2(v_npow_173_, v___x_183_, v_x_175_);
v___x_185_ = lp_mathlib_FreeGroup_invRev___redArg(v___x_184_);
return v___x_185_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_zpowRec___at___00instAddCommGroupFreeAbelianGroup_spec__4___redArg___boxed(lean_object* v_npow_186_, lean_object* v_x_187_, lean_object* v_x_188_){
_start:
{
lean_object* v_res_189_; 
v_res_189_ = lp_mathlib_zpowRec___at___00instAddCommGroupFreeAbelianGroup_spec__4___redArg(v_npow_186_, v_x_187_, v_x_188_);
lean_dec(v_x_187_);
return v_res_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zpowRec___at___00instAddCommGroupFreeAbelianGroup_spec__4(lean_object* v_00_u03b1_190_, lean_object* v_npow_191_, lean_object* v_x_192_, lean_object* v_x_193_){
_start:
{
lean_object* v___x_194_; 
v___x_194_ = lp_mathlib_zpowRec___at___00instAddCommGroupFreeAbelianGroup_spec__4___redArg(v_npow_191_, v_x_192_, v_x_193_);
return v___x_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zpowRec___at___00instAddCommGroupFreeAbelianGroup_spec__4___boxed(lean_object* v_00_u03b1_195_, lean_object* v_npow_196_, lean_object* v_x_197_, lean_object* v_x_198_){
_start:
{
lean_object* v_res_199_; 
v_res_199_ = lp_mathlib_zpowRec___at___00instAddCommGroupFreeAbelianGroup_spec__4(v_00_u03b1_195_, v_npow_196_, v_x_197_, v_x_198_);
lean_dec(v_x_197_);
return v_res_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___lam__1(lean_object* v___y_200_){
_start:
{
lean_object* v___x_201_; lean_object* v_toFun_202_; lean_object* v___x_203_; lean_object* v_toFun_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; 
v___x_201_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0);
v_toFun_202_ = lean_ctor_get(v___x_201_, 0);
v___x_203_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__12___redArg___closed__0, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__12___redArg___closed__0_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__12___redArg___closed__0);
v_toFun_204_ = lean_ctor_get(v___x_203_, 0);
lean_inc(v_toFun_202_);
v___x_205_ = lean_apply_1(v_toFun_202_, v___y_200_);
v___x_206_ = lp_mathlib_FreeGroup_invRev___redArg(v___x_205_);
lean_inc(v_toFun_204_);
v___x_207_ = lean_apply_1(v_toFun_204_, v___x_206_);
return v___x_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___lam__2(lean_object* v___f_208_, lean_object* v_toFun_209_, lean_object* v___y_210_, lean_object* v___y_211_){
_start:
{
lean_object* v___x_212_; lean_object* v_toFun_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; 
v___x_212_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0);
v_toFun_213_ = lean_ctor_get(v___x_212_, 0);
lean_inc(v_toFun_213_);
v___x_214_ = lean_apply_1(v_toFun_213_, v___y_211_);
v___x_215_ = lp_mathlib_zpowRec___at___00instAddCommGroupFreeAbelianGroup_spec__4___redArg(v___f_208_, v___y_210_, v___x_214_);
v___x_216_ = lean_apply_1(v_toFun_209_, v___x_215_);
return v___x_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___lam__2___boxed(lean_object* v___f_217_, lean_object* v_toFun_218_, lean_object* v___y_219_, lean_object* v___y_220_){
_start:
{
lean_object* v_res_221_; 
v_res_221_ = lp_mathlib_instAddCommGroupFreeAbelianGroup___lam__2(v___f_217_, v_toFun_218_, v___y_219_, v___y_220_);
lean_dec(v___y_219_);
return v_res_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___lam__0(lean_object* v___f_222_, lean_object* v_toFun_223_, lean_object* v___y_224_, lean_object* v___y_225_){
_start:
{
lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; 
v___x_226_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0);
lean_inc(v___f_222_);
v___x_227_ = lean_apply_2(v___f_222_, v___x_226_, v___y_224_);
v___x_228_ = lean_apply_2(v___f_222_, v___x_226_, v___y_225_);
v___x_229_ = lp_mathlib_DivInvMonoid_div_x27___at___00instAddCommGroupFreeAbelianGroup_spec__2___redArg(v___x_227_, v___x_228_);
v___x_230_ = lean_apply_1(v_toFun_223_, v___x_229_);
return v___x_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec___at___00npowBinRec_go___at___00npowBinRec___at___00instAddCommGroupFreeAbelianGroup_spec__1_spec__1_spec__4___redArg(lean_object* v_zero_231_, lean_object* v_n_232_, lean_object* v___y_233_, lean_object* v___y_234_){
_start:
{
lean_object* v___y_236_; lean_object* v___y_237_; lean_object* v___x_240_; uint8_t v___x_241_; 
v___x_240_ = lean_unsigned_to_nat(0u);
v___x_241_ = lean_nat_dec_eq(v_n_232_, v___x_240_);
if (v___x_241_ == 0)
{
lean_object* v___x_242_; lean_object* v___x_246_; uint8_t v___x_247_; 
v___x_242_ = lean_unsigned_to_nat(1u);
v___x_246_ = lean_nat_land(v___x_242_, v_n_232_);
v___x_247_ = lean_nat_dec_eq(v___x_246_, v___x_240_);
lean_dec(v___x_246_);
if (v___x_247_ == 0)
{
goto v___jp_243_;
}
else
{
if (v___x_241_ == 0)
{
lean_object* v___x_248_; 
v___x_248_ = lean_nat_shiftr(v_n_232_, v___x_242_);
lean_dec(v_n_232_);
v___y_236_ = v___x_248_;
v___y_237_ = v___y_233_;
goto v___jp_235_;
}
else
{
goto v___jp_243_;
}
}
v___jp_243_:
{
lean_object* v___x_244_; lean_object* v___x_245_; 
v___x_244_ = lean_nat_shiftr(v_n_232_, v___x_242_);
lean_dec(v_n_232_);
lean_inc(v___y_234_);
v___x_245_ = l_List_appendTR___redArg(v___y_233_, v___y_234_);
v___y_236_ = v___x_244_;
v___y_237_ = v___x_245_;
goto v___jp_235_;
}
}
else
{
lean_object* v___x_249_; 
lean_dec(v_n_232_);
v___x_249_ = lean_apply_2(v_zero_231_, v___y_233_, v___y_234_);
return v___x_249_;
}
v___jp_235_:
{
lean_object* v___x_238_; 
lean_inc(v___y_234_);
v___x_238_ = l_List_appendTR___redArg(v___y_234_, v___y_234_);
v_n_232_ = v___y_236_;
v___y_233_ = v___y_237_;
v___y_234_ = v___x_238_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___at___00npowBinRec___at___00instAddCommGroupFreeAbelianGroup_spec__1_spec__1___redArg___lam__0(lean_object* v_y_250_, lean_object* v_x_251_){
_start:
{
lean_inc(v_y_250_);
return v_y_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___at___00npowBinRec___at___00instAddCommGroupFreeAbelianGroup_spec__1_spec__1___redArg___lam__0___boxed(lean_object* v_y_252_, lean_object* v_x_253_){
_start:
{
lean_object* v_res_254_; 
v_res_254_ = lp_mathlib_npowBinRec_go___at___00npowBinRec___at___00instAddCommGroupFreeAbelianGroup_spec__1_spec__1___redArg___lam__0(v_y_252_, v_x_253_);
lean_dec(v_x_253_);
lean_dec(v_y_252_);
return v_res_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___at___00npowBinRec___at___00instAddCommGroupFreeAbelianGroup_spec__1_spec__1___redArg(lean_object* v_k_256_, lean_object* v_a_257_, lean_object* v_a_258_){
_start:
{
lean_object* v___f_259_; lean_object* v___x_260_; 
v___f_259_ = ((lean_object*)(lp_mathlib_npowBinRec_go___at___00npowBinRec___at___00instAddCommGroupFreeAbelianGroup_spec__1_spec__1___redArg___closed__0));
v___x_260_ = lp_mathlib_Nat_binaryRec___at___00npowBinRec_go___at___00npowBinRec___at___00instAddCommGroupFreeAbelianGroup_spec__1_spec__1_spec__4___redArg(v___f_259_, v_k_256_, v_a_257_, v_a_258_);
return v___x_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec___at___00instAddCommGroupFreeAbelianGroup_spec__1___redArg(lean_object* v_k_261_, lean_object* v_a_262_){
_start:
{
lean_object* v___x_263_; lean_object* v___x_264_; 
v___x_263_ = lean_box(0);
v___x_264_ = lp_mathlib_npowBinRec_go___at___00npowBinRec___at___00instAddCommGroupFreeAbelianGroup_spec__1_spec__1___redArg(v_k_261_, v___x_263_, v_a_262_);
return v___x_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___lam__3(lean_object* v_toFun_265_, lean_object* v___y_266_, lean_object* v___y_267_){
_start:
{
lean_object* v___x_268_; lean_object* v_toFun_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; 
v___x_268_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0);
v_toFun_269_ = lean_ctor_get(v___x_268_, 0);
lean_inc(v_toFun_269_);
v___x_270_ = lean_apply_1(v_toFun_269_, v___y_267_);
v___x_271_ = lp_mathlib_npowBinRec___at___00instAddCommGroupFreeAbelianGroup_spec__1___redArg(v___y_266_, v___x_270_);
v___x_272_ = lean_apply_1(v_toFun_265_, v___x_271_);
return v___x_272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup___lam__4(lean_object* v___f_273_, lean_object* v_toFun_274_, lean_object* v___y_275_, lean_object* v___y_276_){
_start:
{
lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; 
v___x_277_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0);
lean_inc(v___f_273_);
v___x_278_ = lean_apply_2(v___f_273_, v___x_277_, v___y_275_);
v___x_279_ = lean_apply_2(v___f_273_, v___x_277_, v___y_276_);
v___x_280_ = l_List_appendTR___redArg(v___x_278_, v___x_279_);
v___x_281_ = lean_apply_1(v_toFun_274_, v___x_280_);
return v___x_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowRec___at___00instAddCommGroupFreeAbelianGroup_spec__3___redArg(lean_object* v_x_282_, lean_object* v_x_283_){
_start:
{
lean_object* v_zero_284_; uint8_t v_isZero_285_; 
v_zero_284_ = lean_unsigned_to_nat(0u);
v_isZero_285_ = lean_nat_dec_eq(v_x_282_, v_zero_284_);
if (v_isZero_285_ == 1)
{
lean_object* v___x_286_; 
lean_dec(v_x_283_);
v___x_286_ = lean_box(0);
return v___x_286_;
}
else
{
lean_object* v_one_287_; lean_object* v_n_288_; lean_object* v___x_289_; lean_object* v___x_290_; 
v_one_287_ = lean_unsigned_to_nat(1u);
v_n_288_ = lean_nat_sub(v_x_282_, v_one_287_);
lean_inc(v_x_283_);
v___x_289_ = lp_mathlib_npowRec___at___00instAddCommGroupFreeAbelianGroup_spec__3___redArg(v_n_288_, v_x_283_);
lean_dec(v_n_288_);
v___x_290_ = l_List_appendTR___redArg(v___x_289_, v_x_283_);
return v___x_290_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowRec___at___00instAddCommGroupFreeAbelianGroup_spec__3___redArg___boxed(lean_object* v_x_291_, lean_object* v_x_292_){
_start:
{
lean_object* v_res_293_; 
v_res_293_ = lp_mathlib_npowRec___at___00instAddCommGroupFreeAbelianGroup_spec__3___redArg(v_x_291_, v_x_292_);
lean_dec(v_x_291_);
return v_res_293_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddCommGroupFreeAbelianGroup(lean_object* v_00_u03b1_297_){
_start:
{
lean_object* v___x_298_; lean_object* v_toFun_299_; lean_object* v___f_300_; lean_object* v___f_301_; lean_object* v___f_302_; lean_object* v___x_303_; lean_object* v___f_304_; lean_object* v___f_305_; lean_object* v___f_306_; lean_object* v___f_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; 
v___x_298_ = lean_obj_once(&lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0, &lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0_once, _init_lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0);
v_toFun_299_ = lean_ctor_get(v___x_298_, 0);
v___f_300_ = ((lean_object*)(lp_mathlib_instAddCommGroupFreeAbelianGroup___closed__0));
v___f_301_ = ((lean_object*)(lp_mathlib_instAddCommGroupFreeAbelianGroup___closed__1));
v___f_302_ = ((lean_object*)(lp_mathlib_instAddCommGroupFreeAbelianGroup___closed__2));
v___x_303_ = lean_box(0);
lean_inc_n(v_toFun_299_, 5);
v___f_304_ = lean_alloc_closure((void*)(lp_mathlib_instAddCommGroupFreeAbelianGroup___lam__2___boxed), 4, 2);
lean_closure_set(v___f_304_, 0, v___f_300_);
lean_closure_set(v___f_304_, 1, v_toFun_299_);
v___f_305_ = lean_alloc_closure((void*)(lp_mathlib_instAddCommGroupFreeAbelianGroup___lam__0), 4, 2);
lean_closure_set(v___f_305_, 0, v___f_301_);
lean_closure_set(v___f_305_, 1, v_toFun_299_);
v___f_306_ = lean_alloc_closure((void*)(lp_mathlib_instAddCommGroupFreeAbelianGroup___lam__3), 3, 1);
lean_closure_set(v___f_306_, 0, v_toFun_299_);
v___f_307_ = lean_alloc_closure((void*)(lp_mathlib_instAddCommGroupFreeAbelianGroup___lam__4), 4, 2);
lean_closure_set(v___f_307_, 0, v___f_301_);
lean_closure_set(v___f_307_, 1, v_toFun_299_);
v___x_308_ = lean_apply_1(v_toFun_299_, v___x_303_);
v___x_309_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_309_, 0, v___x_308_);
lean_ctor_set(v___x_309_, 1, v___f_307_);
lean_ctor_set(v___x_309_, 2, v___f_306_);
v___x_310_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_310_, 0, v___x_309_);
lean_ctor_set(v___x_310_, 1, v___f_302_);
lean_ctor_set(v___x_310_, 2, v___f_305_);
lean_ctor_set(v___x_310_, 3, v___f_304_);
return v___x_310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec___at___00instAddCommGroupFreeAbelianGroup_spec__1(lean_object* v_00_u03b1_311_, lean_object* v_k_312_, lean_object* v_a_313_){
_start:
{
lean_object* v___x_314_; 
v___x_314_ = lp_mathlib_npowBinRec___at___00instAddCommGroupFreeAbelianGroup_spec__1___redArg(v_k_312_, v_a_313_);
return v___x_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowRec___at___00instAddCommGroupFreeAbelianGroup_spec__3(lean_object* v_00_u03b1_315_, lean_object* v_x_316_, lean_object* v_x_317_){
_start:
{
lean_object* v___x_318_; 
v___x_318_ = lp_mathlib_npowRec___at___00instAddCommGroupFreeAbelianGroup_spec__3___redArg(v_x_316_, v_x_317_);
return v___x_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowRec___at___00instAddCommGroupFreeAbelianGroup_spec__3___boxed(lean_object* v_00_u03b1_319_, lean_object* v_x_320_, lean_object* v_x_321_){
_start:
{
lean_object* v_res_322_; 
v_res_322_ = lp_mathlib_npowRec___at___00instAddCommGroupFreeAbelianGroup_spec__3(v_00_u03b1_319_, v_x_320_, v_x_321_);
lean_dec(v_x_320_);
return v_res_322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_npowBinRec_go___at___00npowBinRec___at___00instAddCommGroupFreeAbelianGroup_spec__1_spec__1(lean_object* v_00_u03b1_323_, lean_object* v_k_324_, lean_object* v_a_325_, lean_object* v_a_326_){
_start:
{
lean_object* v___x_327_; 
v___x_327_ = lp_mathlib_npowBinRec_go___at___00npowBinRec___at___00instAddCommGroupFreeAbelianGroup_spec__1_spec__1___redArg(v_k_324_, v_a_325_, v_a_326_);
return v___x_327_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_binaryRec___at___00npowBinRec_go___at___00npowBinRec___at___00instAddCommGroupFreeAbelianGroup_spec__1_spec__1_spec__4(lean_object* v_00_u03b1_328_, lean_object* v_zero_329_, lean_object* v_n_330_, lean_object* v___y_331_, lean_object* v___y_332_){
_start:
{
lean_object* v___x_333_; 
v___x_333_ = lp_mathlib_Nat_binaryRec___at___00npowBinRec_go___at___00npowBinRec___at___00instAddCommGroupFreeAbelianGroup_spec__1_spec__1_spec__4___redArg(v_zero_329_, v_n_330_, v___y_331_, v___y_332_);
return v___x_333_;
}
}
static lean_object* _init_lp_mathlib_instUniqueFreeAbelianGroupOfIsEmpty___closed__0(void){
_start:
{
lean_object* v___x_334_; 
v___x_334_ = lp_mathlib_instInhabitedFreeAbelianGroup(lean_box(0));
return v___x_334_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instUniqueFreeAbelianGroupOfIsEmpty(lean_object* v_00_u03b1_335_, lean_object* v_inst_336_){
_start:
{
lean_object* v___x_337_; 
v___x_337_ = lean_obj_once(&lp_mathlib_instUniqueFreeAbelianGroupOfIsEmpty___closed__0, &lp_mathlib_instUniqueFreeAbelianGroupOfIsEmpty___closed__0_once, _init_lp_mathlib_instUniqueFreeAbelianGroupOfIsEmpty___closed__0);
return v___x_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_of___at___00FreeAbelianGroup_of_spec__0___lam__0(lean_object* v___y_338_){
_start:
{
lean_inc(v___y_338_);
return v___y_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_of___at___00FreeAbelianGroup_of_spec__0___lam__0___boxed(lean_object* v___y_339_){
_start:
{
lean_object* v_res_340_; 
v_res_340_ = lp_mathlib_Abelianization_of___at___00FreeAbelianGroup_of_spec__0___lam__0(v___y_339_);
lean_dec(v___y_339_);
return v_res_340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_of___at___00FreeAbelianGroup_of_spec__0(lean_object* v_00_u03b1_342_){
_start:
{
lean_object* v___f_343_; 
v___f_343_ = ((lean_object*)(lp_mathlib_Abelianization_of___at___00FreeAbelianGroup_of_spec__0___closed__0));
return v___f_343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_of___redArg(lean_object* v_x_344_){
_start:
{
lean_object* v___x_345_; lean_object* v_toFun_346_; lean_object* v___x_347_; lean_object* v___x_348_; 
v___x_345_ = lean_obj_once(&lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0, &lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0_once, _init_lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0);
v_toFun_346_ = lean_ctor_get(v___x_345_, 0);
v___x_347_ = lp_mathlib_FreeGroup_of___redArg(v_x_344_);
lean_inc(v_toFun_346_);
v___x_348_ = lean_apply_1(v_toFun_346_, v___x_347_);
return v___x_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_of(lean_object* v_00_u03b1_349_, lean_object* v_x_350_){
_start:
{
lean_object* v___x_351_; 
v___x_351_ = lp_mathlib_FreeAbelianGroup_of___redArg(v_x_350_);
return v___x_351_;
}
}
static lean_object* _init_lp_mathlib_FreeAbelianGroup_lift___redArg___closed__0(void){
_start:
{
lean_object* v___x_352_; 
v___x_352_ = lp_mathlib_FreeGroup_instGroup(lean_box(0));
return v___x_352_;
}
}
static lean_object* _init_lp_mathlib_FreeAbelianGroup_lift___redArg___closed__1(void){
_start:
{
lean_object* v___x_353_; lean_object* v___x_354_; 
v___x_353_ = lean_obj_once(&lp_mathlib_FreeAbelianGroup_lift___redArg___closed__0, &lp_mathlib_FreeAbelianGroup_lift___redArg___closed__0_once, _init_lp_mathlib_FreeAbelianGroup_lift___redArg___closed__0);
v___x_354_ = lp_mathlib_instCommGroupAbelianization___redArg(v___x_353_);
return v___x_354_;
}
}
static lean_object* _init_lp_mathlib_FreeAbelianGroup_lift___redArg___closed__2(void){
_start:
{
lean_object* v___x_355_; lean_object* v___x_356_; 
v___x_355_ = lean_obj_once(&lp_mathlib_FreeAbelianGroup_lift___redArg___closed__0, &lp_mathlib_FreeAbelianGroup_lift___redArg___closed__0_once, _init_lp_mathlib_FreeAbelianGroup_lift___redArg___closed__0);
v___x_356_ = lp_mathlib_Abelianization_lift___redArg(v___x_355_);
return v___x_356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_lift___redArg(lean_object* v_inst_357_){
_start:
{
lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v_toMonoid_360_; lean_object* v___x_361_; lean_object* v_toAddMonoid_362_; lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; 
lean_inc_ref(v_inst_357_);
v___x_358_ = lp_mathlib_Multiplicative_divInvMonoid___redArg(v_inst_357_);
v___x_359_ = lean_obj_once(&lp_mathlib_FreeAbelianGroup_lift___redArg___closed__1, &lp_mathlib_FreeAbelianGroup_lift___redArg___closed__1_once, _init_lp_mathlib_FreeAbelianGroup_lift___redArg___closed__1);
v_toMonoid_360_ = lean_ctor_get(v___x_359_, 0);
v___x_361_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_360_);
v_toAddMonoid_362_ = lean_ctor_get(v_inst_357_, 0);
lean_inc_ref(v_toAddMonoid_362_);
lean_dec_ref(v_inst_357_);
v___x_363_ = lp_mathlib_FreeGroup_lift___redArg(v___x_358_);
v___x_364_ = lean_obj_once(&lp_mathlib_FreeAbelianGroup_lift___redArg___closed__2, &lp_mathlib_FreeAbelianGroup_lift___redArg___closed__2_once, _init_lp_mathlib_FreeAbelianGroup_lift___redArg___closed__2);
v___x_365_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddMonoid_362_);
lean_dec_ref(v_toAddMonoid_362_);
v___x_366_ = lp_mathlib_Multiplicative_mulOneClass___redArg(v___x_365_);
v___x_367_ = lp_mathlib_MonoidHom_toAdditive(lean_box(0), lean_box(0), v___x_361_, v___x_366_);
lean_dec_ref(v___x_366_);
lean_dec_ref(v___x_361_);
v___x_368_ = lp_mathlib_Equiv_trans___redArg(v___x_364_, v___x_367_);
v___x_369_ = lp_mathlib_Equiv_trans___redArg(v___x_363_, v___x_368_);
return v___x_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_lift(lean_object* v_00_u03b1_370_, lean_object* v_00_u03b2_371_, lean_object* v_inst_372_){
_start:
{
lean_object* v___x_373_; 
v___x_373_ = lp_mathlib_FreeAbelianGroup_lift___redArg(v_inst_372_);
return v___x_373_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_liftAddEquiv___redArg(lean_object* v_inst_374_){
_start:
{
lean_object* v___x_375_; 
v___x_375_ = lp_mathlib_FreeAbelianGroup_lift___redArg(v_inst_374_);
return v___x_375_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_liftAddEquiv(lean_object* v_00_u03b1_376_, lean_object* v_G_377_, lean_object* v_inst_378_){
_start:
{
lean_object* v___x_379_; 
v___x_379_ = lp_mathlib_FreeAbelianGroup_lift___redArg(v_inst_378_);
return v___x_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_liftAddGroupHom___redArg___lam__0(lean_object* v_inst_380_, lean_object* v_a_381_, lean_object* v_f_382_){
_start:
{
lean_object* v___x_383_; lean_object* v_toFun_384_; lean_object* v___x_385_; 
v___x_383_ = lp_mathlib_FreeAbelianGroup_lift___redArg(v_inst_380_);
v_toFun_384_ = lean_ctor_get(v___x_383_, 0);
lean_inc(v_toFun_384_);
lean_dec_ref(v___x_383_);
v___x_385_ = lean_apply_2(v_toFun_384_, v_f_382_, v_a_381_);
return v___x_385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_liftAddGroupHom___redArg(lean_object* v_inst_386_, lean_object* v_a_387_){
_start:
{
lean_object* v___f_388_; 
v___f_388_ = lean_alloc_closure((void*)(lp_mathlib_FreeAbelianGroup_liftAddGroupHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_388_, 0, v_inst_386_);
lean_closure_set(v___f_388_, 1, v_a_387_);
return v___f_388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_liftAddGroupHom(lean_object* v_00_u03b1_389_, lean_object* v_00_u03b2_390_, lean_object* v_inst_391_, lean_object* v_a_392_){
_start:
{
lean_object* v___f_393_; 
v___f_393_ = lean_alloc_closure((void*)(lp_mathlib_FreeAbelianGroup_liftAddGroupHom___redArg___lam__0), 3, 2);
lean_closure_set(v___f_393_, 0, v_inst_391_);
lean_closure_set(v___f_393_, 1, v_a_392_);
return v___f_393_;
}
}
static lean_object* _init_lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__0(void){
_start:
{
lean_object* v___x_394_; 
v___x_394_ = lp_mathlib_instAddCommGroupFreeAbelianGroup(lean_box(0));
return v___x_394_;
}
}
static lean_object* _init_lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__1(void){
_start:
{
lean_object* v___x_395_; lean_object* v___x_396_; 
v___x_395_ = lean_obj_once(&lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__0, &lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__0_once, _init_lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__0);
v___x_396_ = lp_mathlib_FreeAbelianGroup_lift___redArg(v___x_395_);
return v___x_396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_instMonad___lam__0(lean_object* v___f_397_, lean_object* v_00_u03b1_398_, lean_object* v_00_u03b2_399_, lean_object* v_f_400_, lean_object* v_x_401_){
_start:
{
lean_object* v___x_402_; lean_object* v_toFun_403_; lean_object* v___x_404_; lean_object* v___x_405_; 
v___x_402_ = lean_obj_once(&lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__1, &lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__1_once, _init_lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__1);
v_toFun_403_ = lean_ctor_get(v___x_402_, 0);
v___x_404_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_404_, 0, lean_box(0));
lean_closure_set(v___x_404_, 1, lean_box(0));
lean_closure_set(v___x_404_, 2, lean_box(0));
lean_closure_set(v___x_404_, 3, v___f_397_);
lean_closure_set(v___x_404_, 4, v_f_400_);
lean_inc(v_toFun_403_);
v___x_405_ = lean_apply_2(v_toFun_403_, v___x_404_, v_x_401_);
return v___x_405_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_instMonad___lam__1(lean_object* v___f_406_, lean_object* v_00_u03b1_407_, lean_object* v_00_u03b2_408_, lean_object* v___y_409_, lean_object* v___y_410_){
_start:
{
lean_object* v___x_411_; lean_object* v_toFun_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; 
v___x_411_ = lean_obj_once(&lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__1, &lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__1_once, _init_lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__1);
v_toFun_412_ = lean_ctor_get(v___x_411_, 0);
v___x_413_ = lean_alloc_closure((void*)(l_Function_const___boxed), 4, 3);
lean_closure_set(v___x_413_, 0, lean_box(0));
lean_closure_set(v___x_413_, 1, lean_box(0));
lean_closure_set(v___x_413_, 2, v___y_409_);
v___x_414_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_414_, 0, lean_box(0));
lean_closure_set(v___x_414_, 1, lean_box(0));
lean_closure_set(v___x_414_, 2, lean_box(0));
lean_closure_set(v___x_414_, 3, v___f_406_);
lean_closure_set(v___x_414_, 4, v___x_413_);
lean_inc(v_toFun_412_);
v___x_415_ = lean_apply_2(v_toFun_412_, v___x_414_, v___y_410_);
return v___x_415_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_instMonad___lam__2(lean_object* v_00_u03b1_416_, lean_object* v_00_u03b1_417_){
_start:
{
lean_object* v___x_418_; 
v___x_418_ = lp_mathlib_FreeAbelianGroup_of___redArg(v_00_u03b1_417_);
return v___x_418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_instMonad___lam__3(lean_object* v_00_u03b1_419_, lean_object* v_00_u03b2_420_, lean_object* v_x_421_, lean_object* v_f_422_){
_start:
{
lean_object* v___x_423_; lean_object* v_toFun_424_; lean_object* v___x_425_; 
v___x_423_ = lean_obj_once(&lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__1, &lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__1_once, _init_lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__1);
v_toFun_424_ = lean_ctor_get(v___x_423_, 0);
lean_inc(v_toFun_424_);
v___x_425_ = lean_apply_2(v_toFun_424_, v_f_422_, v_x_421_);
return v___x_425_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_instMonad___lam__4(lean_object* v_x_426_, lean_object* v___f_427_, lean_object* v___f_428_, lean_object* v_y_429_){
_start:
{
lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; 
v___x_430_ = lean_box(0);
v___x_431_ = lean_apply_1(v_x_426_, v___x_430_);
v___x_432_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_432_, 0, lean_box(0));
lean_closure_set(v___x_432_, 1, lean_box(0));
lean_closure_set(v___x_432_, 2, lean_box(0));
lean_closure_set(v___x_432_, 3, v___f_427_);
lean_closure_set(v___x_432_, 4, v_y_429_);
v___x_433_ = lean_apply_4(v___f_428_, lean_box(0), lean_box(0), v___x_431_, v___x_432_);
return v___x_433_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_instMonad___lam__5(lean_object* v___f_434_, lean_object* v___f_435_, lean_object* v_00_u03b1_436_, lean_object* v_00_u03b2_437_, lean_object* v_f_438_, lean_object* v_x_439_){
_start:
{
lean_object* v___f_440_; lean_object* v___x_441_; 
lean_inc(v___f_435_);
v___f_440_ = lean_alloc_closure((void*)(lp_mathlib_FreeAbelianGroup_instMonad___lam__4), 4, 3);
lean_closure_set(v___f_440_, 0, v_x_439_);
lean_closure_set(v___f_440_, 1, v___f_434_);
lean_closure_set(v___f_440_, 2, v___f_435_);
v___x_441_ = lean_apply_4(v___f_435_, lean_box(0), lean_box(0), v_f_438_, v___f_440_);
return v___x_441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_instMonad___lam__7(lean_object* v_a_442_, lean_object* v_x_443_){
_start:
{
lean_object* v___x_444_; 
v___x_444_ = lp_mathlib_FreeAbelianGroup_of___redArg(v_a_442_);
return v___x_444_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_instMonad___lam__7___boxed(lean_object* v_a_445_, lean_object* v_x_446_){
_start:
{
lean_object* v_res_447_; 
v_res_447_ = lp_mathlib_FreeAbelianGroup_instMonad___lam__7(v_a_445_, v_x_446_);
lean_dec(v_x_446_);
return v_res_447_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_instMonad___lam__6(lean_object* v_y_448_, lean_object* v___f_449_, lean_object* v_a_450_){
_start:
{
lean_object* v___f_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; 
v___f_451_ = lean_alloc_closure((void*)(lp_mathlib_FreeAbelianGroup_instMonad___lam__7___boxed), 2, 1);
lean_closure_set(v___f_451_, 0, v_a_450_);
v___x_452_ = lean_box(0);
v___x_453_ = lean_apply_1(v_y_448_, v___x_452_);
v___x_454_ = lean_apply_4(v___f_449_, lean_box(0), lean_box(0), v___x_453_, v___f_451_);
return v___x_454_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_instMonad___lam__8(lean_object* v___f_455_, lean_object* v_00_u03b1_456_, lean_object* v_00_u03b2_457_, lean_object* v_x_458_, lean_object* v_y_459_){
_start:
{
lean_object* v___f_460_; lean_object* v___x_461_; 
lean_inc(v___f_455_);
v___f_460_ = lean_alloc_closure((void*)(lp_mathlib_FreeAbelianGroup_instMonad___lam__6), 3, 2);
lean_closure_set(v___f_460_, 0, v_y_459_);
lean_closure_set(v___f_460_, 1, v___f_455_);
v___x_461_ = lean_apply_4(v___f_455_, lean_box(0), lean_box(0), v_x_458_, v___f_460_);
return v___x_461_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_instMonad___lam__9(lean_object* v_y_462_, lean_object* v_x_463_){
_start:
{
lean_object* v___x_464_; lean_object* v___x_465_; 
v___x_464_ = lean_box(0);
v___x_465_ = lean_apply_1(v_y_462_, v___x_464_);
return v___x_465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_instMonad___lam__9___boxed(lean_object* v_y_466_, lean_object* v_x_467_){
_start:
{
lean_object* v_res_468_; 
v_res_468_ = lp_mathlib_FreeAbelianGroup_instMonad___lam__9(v_y_466_, v_x_467_);
lean_dec(v_x_467_);
return v_res_468_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_instMonad___lam__10(lean_object* v_00_u03b1_469_, lean_object* v_00_u03b2_470_, lean_object* v_x_471_, lean_object* v_y_472_){
_start:
{
lean_object* v___x_473_; lean_object* v_toFun_474_; lean_object* v___f_475_; lean_object* v___x_476_; 
v___x_473_ = lean_obj_once(&lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__1, &lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__1_once, _init_lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__1);
v_toFun_474_ = lean_ctor_get(v___x_473_, 0);
v___f_475_ = lean_alloc_closure((void*)(lp_mathlib_FreeAbelianGroup_instMonad___lam__9___boxed), 2, 1);
lean_closure_set(v___f_475_, 0, v_y_472_);
lean_inc(v_toFun_474_);
v___x_476_ = lean_apply_2(v_toFun_474_, v___f_475_, v_x_471_);
return v___x_476_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27___at___00FreeAbelianGroup_seqAddGroupHom_spec__1___redArg(lean_object* v_f_503_){
_start:
{
lean_inc(v_f_503_);
return v_f_503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27___at___00FreeAbelianGroup_seqAddGroupHom_spec__1___redArg___boxed(lean_object* v_f_504_){
_start:
{
lean_object* v_res_505_; 
v_res_505_ = lp_mathlib_AddMonoidHom_mk_x27___at___00FreeAbelianGroup_seqAddGroupHom_spec__1___redArg(v_f_504_);
lean_dec(v_f_504_);
return v_res_505_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27___at___00FreeAbelianGroup_seqAddGroupHom_spec__1(lean_object* v_00_u03b2_506_, lean_object* v_00_u03b1_507_, lean_object* v_f_508_, lean_object* v_map__mul_509_){
_start:
{
lean_inc(v_f_508_);
return v_f_508_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mk_x27___at___00FreeAbelianGroup_seqAddGroupHom_spec__1___boxed(lean_object* v_00_u03b2_510_, lean_object* v_00_u03b1_511_, lean_object* v_f_512_, lean_object* v_map__mul_513_){
_start:
{
lean_object* v_res_514_; 
v_res_514_ = lp_mathlib_AddMonoidHom_mk_x27___at___00FreeAbelianGroup_seqAddGroupHom_spec__1(v_00_u03b2_510_, v_00_u03b1_511_, v_f_512_, v_map__mul_513_);
lean_dec(v_f_512_);
return v_res_514_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_seqAddGroupHom___redArg___lam__0(lean_object* v_y_515_, lean_object* v___y_516_){
_start:
{
lean_object* v___x_517_; lean_object* v___x_518_; 
v___x_517_ = lean_apply_1(v_y_515_, v___y_516_);
v___x_518_ = lp_mathlib_FreeAbelianGroup_of___redArg(v___x_517_);
return v___x_518_;
}
}
static lean_object* _init_lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__4___redArg___closed__0(void){
_start:
{
lean_object* v___x_519_; 
v___x_519_ = lp_mathlib_Multiplicative_toAdd(lean_box(0));
return v___x_519_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__4___redArg(lean_object* v_f_520_, lean_object* v_a_521_, lean_object* v_a_522_){
_start:
{
if (lean_obj_tag(v_a_521_) == 0)
{
lean_object* v___x_523_; 
lean_dec(v_f_520_);
v___x_523_ = l_List_reverse___redArg(v_a_522_);
return v___x_523_;
}
else
{
lean_object* v_head_524_; lean_object* v_tail_525_; lean_object* v___x_527_; uint8_t v_isShared_528_; uint8_t v_isSharedCheck_554_; 
v_head_524_ = lean_ctor_get(v_a_521_, 0);
v_tail_525_ = lean_ctor_get(v_a_521_, 1);
v_isSharedCheck_554_ = !lean_is_exclusive(v_a_521_);
if (v_isSharedCheck_554_ == 0)
{
v___x_527_ = v_a_521_;
v_isShared_528_ = v_isSharedCheck_554_;
goto v_resetjp_526_;
}
else
{
lean_inc(v_tail_525_);
lean_inc(v_head_524_);
lean_dec(v_a_521_);
v___x_527_ = lean_box(0);
v_isShared_528_ = v_isSharedCheck_554_;
goto v_resetjp_526_;
}
v_resetjp_526_:
{
lean_object* v___y_530_; lean_object* v_snd_535_; uint8_t v___x_536_; 
v_snd_535_ = lean_ctor_get(v_head_524_, 1);
v___x_536_ = lean_unbox(v_snd_535_);
if (v___x_536_ == 0)
{
lean_object* v_fst_537_; lean_object* v___x_538_; lean_object* v_toFun_539_; lean_object* v___x_540_; lean_object* v_toFun_541_; lean_object* v___x_542_; lean_object* v_toFun_543_; lean_object* v___x_544_; lean_object* v_toFun_545_; lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; 
v_fst_537_ = lean_ctor_get(v_head_524_, 0);
lean_inc(v_fst_537_);
lean_dec(v_head_524_);
v___x_538_ = lean_obj_once(&lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__4___redArg___closed__0, &lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__4___redArg___closed__0_once, _init_lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__4___redArg___closed__0);
v_toFun_539_ = lean_ctor_get(v___x_538_, 0);
v___x_540_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0);
v_toFun_541_ = lean_ctor_get(v___x_540_, 0);
v___x_542_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__12___redArg___closed__0, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__12___redArg___closed__0_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__12___redArg___closed__0);
v_toFun_543_ = lean_ctor_get(v___x_542_, 0);
v___x_544_ = lean_obj_once(&lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0, &lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0_once, _init_lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0);
v_toFun_545_ = lean_ctor_get(v___x_544_, 0);
lean_inc(v_f_520_);
v___x_546_ = lean_apply_1(v_f_520_, v_fst_537_);
lean_inc(v_toFun_539_);
v___x_547_ = lean_apply_1(v_toFun_539_, v___x_546_);
lean_inc(v_toFun_541_);
v___x_548_ = lean_apply_1(v_toFun_541_, v___x_547_);
v___x_549_ = lp_mathlib_FreeGroup_invRev___redArg(v___x_548_);
lean_inc(v_toFun_543_);
v___x_550_ = lean_apply_1(v_toFun_543_, v___x_549_);
lean_inc(v_toFun_545_);
v___x_551_ = lean_apply_1(v_toFun_545_, v___x_550_);
v___y_530_ = v___x_551_;
goto v___jp_529_;
}
else
{
lean_object* v_fst_552_; lean_object* v___x_553_; 
v_fst_552_ = lean_ctor_get(v_head_524_, 0);
lean_inc(v_fst_552_);
lean_dec(v_head_524_);
lean_inc(v_f_520_);
v___x_553_ = lean_apply_1(v_f_520_, v_fst_552_);
v___y_530_ = v___x_553_;
goto v___jp_529_;
}
v___jp_529_:
{
lean_object* v___x_532_; 
if (v_isShared_528_ == 0)
{
lean_ctor_set(v___x_527_, 1, v_a_522_);
lean_ctor_set(v___x_527_, 0, v___y_530_);
v___x_532_ = v___x_527_;
goto v_reusejp_531_;
}
else
{
lean_object* v_reuseFailAlloc_534_; 
v_reuseFailAlloc_534_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_534_, 0, v___y_530_);
lean_ctor_set(v_reuseFailAlloc_534_, 1, v_a_522_);
v___x_532_ = v_reuseFailAlloc_534_;
goto v_reusejp_531_;
}
v_reusejp_531_:
{
v_a_521_ = v_tail_525_;
v_a_522_ = v___x_532_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__5_spec__8___redArg(lean_object* v_init_555_, lean_object* v_x_556_){
_start:
{
if (lean_obj_tag(v_x_556_) == 0)
{
lean_inc(v_init_555_);
return v_init_555_;
}
else
{
lean_object* v_head_557_; lean_object* v_tail_558_; lean_object* v___x_559_; lean_object* v_toFun_560_; lean_object* v___x_561_; lean_object* v_toFun_562_; lean_object* v___x_563_; lean_object* v_toFun_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_572_; lean_object* v___x_573_; 
v_head_557_ = lean_ctor_get(v_x_556_, 0);
lean_inc(v_head_557_);
v_tail_558_ = lean_ctor_get(v_x_556_, 1);
lean_inc(v_tail_558_);
lean_dec_ref_known(v_x_556_, 2);
v___x_559_ = lean_obj_once(&lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__4___redArg___closed__0, &lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__4___redArg___closed__0_once, _init_lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__4___redArg___closed__0);
v_toFun_560_ = lean_ctor_get(v___x_559_, 0);
v___x_561_ = lean_obj_once(&lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0, &lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0_once, _init_lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0);
v_toFun_562_ = lean_ctor_get(v___x_561_, 0);
v___x_563_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__12___redArg___closed__0, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__12___redArg___closed__0_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__12___redArg___closed__0);
v_toFun_564_ = lean_ctor_get(v___x_563_, 0);
lean_inc_n(v_toFun_560_, 2);
v___x_565_ = lean_apply_1(v_toFun_560_, v_head_557_);
v___x_566_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0);
v___x_567_ = lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___lam__0(v___x_566_, v___x_565_);
v___x_568_ = lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__5_spec__8___redArg(v_init_555_, v_tail_558_);
v___x_569_ = lean_apply_1(v_toFun_560_, v___x_568_);
v___x_570_ = lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___lam__0(v___x_566_, v___x_569_);
v___x_571_ = l_List_appendTR___redArg(v___x_567_, v___x_570_);
lean_inc(v_toFun_562_);
v___x_572_ = lean_apply_1(v_toFun_562_, v___x_571_);
lean_inc(v_toFun_564_);
v___x_573_ = lean_apply_1(v_toFun_564_, v___x_572_);
return v___x_573_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__5_spec__8___redArg___boxed(lean_object* v_init_574_, lean_object* v_x_575_){
_start:
{
lean_object* v_res_576_; 
v_res_576_ = lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__5_spec__8___redArg(v_init_574_, v_x_575_);
lean_dec(v_init_574_);
return v_res_576_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__5___redArg(lean_object* v_l_577_){
_start:
{
lean_object* v___x_578_; lean_object* v_toFun_579_; lean_object* v___x_580_; lean_object* v_toFun_581_; lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v___x_585_; 
v___x_578_ = lean_obj_once(&lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0, &lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0_once, _init_lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0);
v_toFun_579_ = lean_ctor_get(v___x_578_, 0);
v___x_580_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__12___redArg___closed__0, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__12___redArg___closed__0_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__12___redArg___closed__0);
v_toFun_581_ = lean_ctor_get(v___x_580_, 0);
v___x_582_ = lean_box(0);
lean_inc(v_toFun_579_);
v___x_583_ = lean_apply_1(v_toFun_579_, v___x_582_);
lean_inc(v_toFun_581_);
v___x_584_ = lean_apply_1(v_toFun_581_, v___x_583_);
v___x_585_ = lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__5_spec__8___redArg(v___x_584_, v_l_577_);
lean_dec(v___x_584_);
return v___x_585_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2___redArg(lean_object* v_f_586_, lean_object* v_L_587_){
_start:
{
lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; 
v___x_588_ = lean_box(0);
v___x_589_ = lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__4___redArg(v_f_586_, v_L_587_, v___x_588_);
v___x_590_ = lp_mathlib_List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__5___redArg(v___x_589_);
return v___x_590_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0___lam__0(lean_object* v_g_591_, lean_object* v___y_592_){
_start:
{
lean_object* v___x_593_; lean_object* v___x_594_; 
v___x_593_ = lp_mathlib_FreeGroup_of___redArg(v___y_592_);
v___x_594_ = lean_apply_1(v_g_591_, v___x_593_);
return v___x_594_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0(lean_object* v_00_u03b2_600_, lean_object* v_00_u03b1_601_){
_start:
{
lean_object* v___x_602_; 
v___x_602_ = ((lean_object*)(lp_mathlib_FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0___closed__2));
return v___x_602_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1___lam__1(lean_object* v_F_603_, lean_object* v___y_604_){
_start:
{
lean_object* v___f_605_; lean_object* v___x_606_; 
v___f_605_ = ((lean_object*)(lp_mathlib_Abelianization_of___at___00FreeAbelianGroup_of_spec__0___closed__0));
v___x_606_ = lp_mathlib_OneHom_comp___redArg___lam__0(v___f_605_, v_F_603_, v___y_604_);
return v___x_606_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_con___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__6_spec__13(lean_object* v_00_u03b1_607_, lean_object* v_N_608_, lean_object* v_nN_609_){
_start:
{
lean_object* v___x_610_; 
v___x_610_ = lean_box(0);
return v___x_610_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__6_spec__14___redArg___lam__0(lean_object* v_f_611_, lean_object* v_x_612_){
_start:
{
lean_object* v___x_613_; 
v___x_613_ = lean_apply_1(v_f_611_, v_x_612_);
return v___x_613_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__6_spec__14___redArg(lean_object* v_c_614_, lean_object* v_f_615_){
_start:
{
lean_object* v___f_616_; 
v___f_616_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__6_spec__14___redArg___lam__0), 2, 1);
lean_closure_set(v___f_616_, 0, v_f_615_);
return v___f_616_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__6___redArg(lean_object* v_N_617_, lean_object* v_00_u03c6_618_){
_start:
{
lean_object* v___f_619_; 
v___f_619_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__6_spec__14___redArg___lam__0), 2, 1);
lean_closure_set(v___f_619_, 0, v_00_u03c6_618_);
return v___f_619_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_copy___at___00Subgroup_closure___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__5_spec__11(lean_object* v_00_u03b1_620_, lean_object* v_S_621_, lean_object* v_s_622_, lean_object* v_hs_623_){
_start:
{
lean_object* v___x_624_; 
v___x_624_ = lean_box(0);
return v___x_624_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_iInf___at___00Subgroup_closure___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__5_spec__9(lean_object* v_00_u03b1_625_, lean_object* v_00_u03b9_626_, lean_object* v_s_627_){
_start:
{
lean_object* v___x_628_; 
v___x_628_ = lean_box(0);
return v___x_628_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_iInf___at___00Subgroup_closure___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__5_spec__9___boxed(lean_object* v_00_u03b1_629_, lean_object* v_00_u03b9_630_, lean_object* v_s_631_){
_start:
{
lean_object* v_res_632_; 
v_res_632_ = lp_mathlib_iInf___at___00Subgroup_closure___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__5_spec__9(v_00_u03b1_629_, v_00_u03b9_630_, v_s_631_);
lean_dec_ref(v_s_631_);
return v_res_632_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_closure___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__5___lam__0(lean_object* v_S_633_, lean_object* v_h_634_){
_start:
{
return v_S_633_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_closure___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__5___lam__1(lean_object* v_S_635_){
_start:
{
lean_object* v___x_636_; 
v___x_636_ = lean_box(0);
return v___x_636_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_iInf___at___00Subgroup_closure___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__5_spec__10(lean_object* v_00_u03b1_637_, lean_object* v_00_u03b9_638_, lean_object* v_s_639_){
_start:
{
lean_object* v___x_640_; 
v___x_640_ = lean_box(0);
return v___x_640_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_iInf___at___00Subgroup_closure___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__5_spec__10___boxed(lean_object* v_00_u03b1_641_, lean_object* v_00_u03b9_642_, lean_object* v_s_643_){
_start:
{
lean_object* v_res_644_; 
v_res_644_ = lp_mathlib_iInf___at___00Subgroup_closure___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__5_spec__10(v_00_u03b1_641_, v_00_u03b9_642_, v_s_643_);
lean_dec_ref(v_s_643_);
return v_res_644_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_closure___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__5(lean_object* v_00_u03b1_645_, lean_object* v_k_646_){
_start:
{
lean_object* v___x_647_; 
v___x_647_ = lean_box(0);
return v___x_647_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1___lam__0(lean_object* v_f_648_, lean_object* v___y_649_){
_start:
{
lean_object* v___x_650_; 
v___x_650_ = lean_apply_1(v_f_648_, v___y_649_);
return v___x_650_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1(lean_object* v_00_u03b1_656_, lean_object* v_00_u03b2_657_, lean_object* v_inst_658_){
_start:
{
lean_object* v___x_659_; 
v___x_659_ = ((lean_object*)(lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1___closed__2));
return v___x_659_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditive___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__2___lam__0(lean_object* v_f_660_, lean_object* v___y_661_){
_start:
{
lean_object* v___x_662_; lean_object* v_toFun_663_; lean_object* v___x_664_; lean_object* v_toFun_665_; lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v___x_668_; 
v___x_662_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0);
v_toFun_663_ = lean_ctor_get(v___x_662_, 0);
v___x_664_ = lean_obj_once(&lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0, &lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0_once, _init_lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0);
v_toFun_665_ = lean_ctor_get(v___x_664_, 0);
lean_inc(v_toFun_663_);
v___x_666_ = lean_apply_1(v_toFun_663_, v___y_661_);
v___x_667_ = lean_apply_1(v_f_660_, v___x_666_);
lean_inc(v_toFun_665_);
v___x_668_ = lean_apply_1(v_toFun_665_, v___x_667_);
return v___x_668_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditive___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__2___lam__1(lean_object* v_f_669_, lean_object* v___y_670_){
_start:
{
lean_object* v___x_671_; lean_object* v_toFun_672_; lean_object* v___x_673_; lean_object* v_toFun_674_; lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; 
v___x_671_ = lean_obj_once(&lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0, &lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0_once, _init_lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0);
v_toFun_672_ = lean_ctor_get(v___x_671_, 0);
v___x_673_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0);
v_toFun_674_ = lean_ctor_get(v___x_673_, 0);
lean_inc(v_toFun_672_);
v___x_675_ = lean_apply_1(v_toFun_672_, v___y_670_);
v___x_676_ = lean_apply_1(v_f_669_, v___x_675_);
lean_inc(v_toFun_674_);
v___x_677_ = lean_apply_1(v_toFun_674_, v___x_676_);
return v___x_677_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditive___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__2(lean_object* v_00_u03b1_683_, lean_object* v_00_u03b2_684_){
_start:
{
lean_object* v___x_685_; 
v___x_685_ = ((lean_object*)(lp_mathlib_MonoidHom_toAdditive___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__2___closed__2));
return v___x_685_;
}
}
static lean_object* _init_lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0___closed__0(void){
_start:
{
lean_object* v___x_686_; 
v___x_686_ = lp_mathlib_FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0(lean_box(0), lean_box(0));
return v___x_686_;
}
}
static lean_object* _init_lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0___closed__1(void){
_start:
{
lean_object* v___x_687_; 
v___x_687_ = lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1(lean_box(0), lean_box(0), lean_box(0));
return v___x_687_;
}
}
static lean_object* _init_lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0___closed__2(void){
_start:
{
lean_object* v___x_688_; 
v___x_688_ = lp_mathlib_MonoidHom_toAdditive___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__2(lean_box(0), lean_box(0));
return v___x_688_;
}
}
static lean_object* _init_lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0___closed__3(void){
_start:
{
lean_object* v___x_689_; lean_object* v___x_690_; lean_object* v___x_691_; 
v___x_689_ = lean_obj_once(&lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0___closed__2, &lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0___closed__2_once, _init_lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0___closed__2);
v___x_690_ = lean_obj_once(&lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0___closed__1, &lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0___closed__1_once, _init_lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0___closed__1);
v___x_691_ = lp_mathlib_Equiv_trans___redArg(v___x_690_, v___x_689_);
return v___x_691_;
}
}
static lean_object* _init_lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0___closed__4(void){
_start:
{
lean_object* v___x_692_; lean_object* v___x_693_; lean_object* v___x_694_; 
v___x_692_ = lean_obj_once(&lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0___closed__3, &lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0___closed__3_once, _init_lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0___closed__3);
v___x_693_ = lean_obj_once(&lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0___closed__0, &lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0___closed__0_once, _init_lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0___closed__0);
v___x_694_ = lp_mathlib_Equiv_trans___redArg(v___x_693_, v___x_692_);
return v___x_694_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0(lean_object* v_00_u03b2_695_, lean_object* v_00_u03b1_696_){
_start:
{
lean_object* v___x_697_; 
v___x_697_ = lean_obj_once(&lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0___closed__4, &lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0___closed__4_once, _init_lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0___closed__4);
return v___x_697_;
}
}
static lean_object* _init_lp_mathlib_FreeAbelianGroup_seqAddGroupHom___redArg___lam__1___closed__0(void){
_start:
{
lean_object* v___x_698_; 
v___x_698_ = lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0(lean_box(0), lean_box(0));
return v___x_698_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_seqAddGroupHom___redArg___lam__1(lean_object* v_x_699_, lean_object* v_y_700_){
_start:
{
lean_object* v___x_701_; lean_object* v_toFun_702_; lean_object* v___f_703_; lean_object* v___x_704_; 
v___x_701_ = lean_obj_once(&lp_mathlib_FreeAbelianGroup_seqAddGroupHom___redArg___lam__1___closed__0, &lp_mathlib_FreeAbelianGroup_seqAddGroupHom___redArg___lam__1___closed__0_once, _init_lp_mathlib_FreeAbelianGroup_seqAddGroupHom___redArg___lam__1___closed__0);
v_toFun_702_ = lean_ctor_get(v___x_701_, 0);
v___f_703_ = lean_alloc_closure((void*)(lp_mathlib_FreeAbelianGroup_seqAddGroupHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_703_, 0, v_y_700_);
lean_inc(v_toFun_702_);
v___x_704_ = lean_apply_2(v_toFun_702_, v___f_703_, v_x_699_);
return v___x_704_;
}
}
static lean_object* _init_lp_mathlib_FreeAbelianGroup_seqAddGroupHom___redArg___lam__2___closed__0(void){
_start:
{
lean_object* v___x_705_; 
v___x_705_ = lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0(lean_box(0), lean_box(0));
return v___x_705_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_seqAddGroupHom___redArg___lam__2(lean_object* v_f_706_, lean_object* v_x_707_){
_start:
{
lean_object* v___x_708_; lean_object* v_toFun_709_; lean_object* v___f_710_; lean_object* v___x_711_; 
v___x_708_ = lean_obj_once(&lp_mathlib_FreeAbelianGroup_seqAddGroupHom___redArg___lam__2___closed__0, &lp_mathlib_FreeAbelianGroup_seqAddGroupHom___redArg___lam__2___closed__0_once, _init_lp_mathlib_FreeAbelianGroup_seqAddGroupHom___redArg___lam__2___closed__0);
v_toFun_709_ = lean_ctor_get(v___x_708_, 0);
v___f_710_ = lean_alloc_closure((void*)(lp_mathlib_FreeAbelianGroup_seqAddGroupHom___redArg___lam__1), 2, 1);
lean_closure_set(v___f_710_, 0, v_x_707_);
lean_inc(v_toFun_709_);
v___x_711_ = lean_apply_2(v_toFun_709_, v___f_710_, v_f_706_);
return v___x_711_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_seqAddGroupHom___redArg(lean_object* v_f_712_){
_start:
{
lean_object* v___f_713_; 
v___f_713_ = lean_alloc_closure((void*)(lp_mathlib_FreeAbelianGroup_seqAddGroupHom___redArg___lam__2), 2, 1);
lean_closure_set(v___f_713_, 0, v_f_712_);
return v___f_713_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_seqAddGroupHom(lean_object* v_00_u03b1_714_, lean_object* v_00_u03b2_715_, lean_object* v_f_716_){
_start:
{
lean_object* v___f_717_; 
v___f_717_ = lean_alloc_closure((void*)(lp_mathlib_FreeAbelianGroup_seqAddGroupHom___redArg___lam__2), 2, 1);
lean_closure_set(v___f_717_, 0, v_f_716_);
return v___f_717_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__3___redArg(lean_object* v_f_718_){
_start:
{
lean_inc(v_f_718_);
return v_f_718_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__3___redArg___boxed(lean_object* v_f_719_){
_start:
{
lean_object* v_res_720_; 
v_res_720_ = lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__3___redArg(v_f_719_);
lean_dec(v_f_719_);
return v_res_720_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__3(lean_object* v_00_u03b2_721_, lean_object* v_00_u03b1_722_, lean_object* v_f_723_, lean_object* v_map__mul_724_){
_start:
{
lean_inc(v_f_723_);
return v_f_723_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__3___boxed(lean_object* v_00_u03b2_725_, lean_object* v_00_u03b1_726_, lean_object* v_f_727_, lean_object* v_map__mul_728_){
_start:
{
lean_object* v_res_729_; 
v_res_729_ = lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__3(v_00_u03b2_725_, v_00_u03b1_726_, v_f_727_, v_map__mul_728_);
lean_dec(v_f_727_);
return v_res_729_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2(lean_object* v_00_u03b2_730_, lean_object* v_00_u03b1_731_, lean_object* v_f_732_, lean_object* v_L_733_){
_start:
{
lean_object* v___x_734_; 
v___x_734_ = lp_mathlib_FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2___redArg(v_f_732_, v_L_733_);
return v___x_734_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__6(lean_object* v_00_u03b1_735_, lean_object* v_00_u03b2_736_, lean_object* v_N_737_, lean_object* v_nN_738_, lean_object* v_00_u03c6_739_, lean_object* v_HN_740_){
_start:
{
lean_object* v___f_741_; 
v___f_741_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__6_spec__14___redArg___lam__0), 2, 1);
lean_closure_set(v___f_741_, 0, v_00_u03c6_739_);
return v___f_741_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__4(lean_object* v_00_u03b1_742_, lean_object* v_f_743_, lean_object* v_00_u03b2_744_, lean_object* v_a_745_, lean_object* v_a_746_){
_start:
{
lean_object* v___x_747_; 
v___x_747_ = lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__4___redArg(v_f_743_, v_a_745_, v_a_746_);
return v___x_747_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__5(lean_object* v_00_u03b2_748_, lean_object* v_l_749_){
_start:
{
lean_object* v___x_750_; 
v___x_750_ = lp_mathlib_List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__5___redArg(v_l_749_);
return v___x_750_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__6_spec__14(lean_object* v_00_u03b1_751_, lean_object* v_00_u03b2_752_, lean_object* v_c_753_, lean_object* v_f_754_, lean_object* v_H_755_){
_start:
{
lean_object* v___f_756_; 
v___f_756_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__6_spec__14___redArg___lam__0), 2, 1);
lean_closure_set(v___f_756_, 0, v_f_754_);
return v___f_756_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__5_spec__8(lean_object* v_00_u03b2_757_, lean_object* v_init_758_, lean_object* v_x_759_){
_start:
{
lean_object* v___x_760_; 
v___x_760_ = lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__5_spec__8___redArg(v_init_758_, v_x_759_);
return v___x_760_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__5_spec__8___boxed(lean_object* v_00_u03b2_761_, lean_object* v_init_762_, lean_object* v_x_763_){
_start:
{
lean_object* v_res_764_; 
v_res_764_ = lp_mathlib_List_foldr___at___00List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__5_spec__8(v_00_u03b2_761_, v_init_762_, v_x_763_);
lean_dec(v_init_762_);
return v_res_764_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_liftOn_x27___at___00Con_liftOn___at___00Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__6_spec__14_spec__16_spec__17___redArg(lean_object* v_q_765_, lean_object* v_f_766_){
_start:
{
lean_object* v___x_767_; 
v___x_767_ = lean_apply_1(v_f_766_, v_q_765_);
return v___x_767_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_liftOn_x27___at___00Con_liftOn___at___00Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__6_spec__14_spec__16_spec__17(lean_object* v_c_768_, lean_object* v_00_u03c6_769_, lean_object* v_q_770_, lean_object* v_f_771_, lean_object* v_h_772_){
_start:
{
lean_object* v___x_773_; 
v___x_773_ = lean_apply_1(v_f_771_, v_q_770_);
return v___x_773_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_liftOn___at___00Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__6_spec__14_spec__16___redArg(lean_object* v_c_774_, lean_object* v_q_775_, lean_object* v_f_776_){
_start:
{
lean_object* v___x_777_; 
v___x_777_ = lean_apply_1(v_f_776_, v_q_775_);
return v___x_777_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_liftOn___at___00Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__6_spec__14_spec__16(lean_object* v_00_u03b1_778_, lean_object* v_00_u03b2_779_, lean_object* v_c_780_, lean_object* v_q_781_, lean_object* v_f_782_, lean_object* v_h_783_){
_start:
{
lean_object* v___x_784_; 
v___x_784_ = lean_apply_1(v_f_782_, v_q_781_);
return v___x_784_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_map___redArg___lam__0(lean_object* v_f_785_, lean_object* v___y_786_){
_start:
{
lean_object* v___x_787_; lean_object* v___x_788_; 
v___x_787_ = lean_apply_1(v_f_785_, v___y_786_);
v___x_788_ = lp_mathlib_FreeAbelianGroup_of___redArg(v___x_787_);
return v___x_788_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__1_spec__4_spec__7___redArg(lean_object* v_c_789_, lean_object* v_f_790_){
_start:
{
lean_object* v___f_791_; 
v___f_791_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__6_spec__14___redArg___lam__0), 2, 1);
lean_closure_set(v___f_791_, 0, v_f_790_);
return v___f_791_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__1_spec__4___redArg(lean_object* v_00_u03c6_792_){
_start:
{
lean_object* v___f_793_; 
v___f_793_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__6_spec__14___redArg___lam__0), 2, 1);
lean_closure_set(v___f_793_, 0, v_00_u03c6_792_);
return v___f_793_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__1(lean_object* v_00_u03b1_798_, lean_object* v_00_u03b2_799_, lean_object* v_inst_800_){
_start:
{
lean_object* v___x_801_; 
v___x_801_ = ((lean_object*)(lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__1___closed__1));
return v___x_801_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_toAdditive___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__2(lean_object* v_00_u03b1_802_, lean_object* v_00_u03b2_803_){
_start:
{
lean_object* v___x_804_; 
v___x_804_ = ((lean_object*)(lp_mathlib_MonoidHom_toAdditive___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__2___closed__2));
return v___x_804_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0_spec__1_spec__3___redArg(lean_object* v_f_805_, lean_object* v_a_806_, lean_object* v_a_807_){
_start:
{
if (lean_obj_tag(v_a_806_) == 0)
{
lean_object* v___x_808_; 
lean_dec(v_f_805_);
v___x_808_ = l_List_reverse___redArg(v_a_807_);
return v___x_808_;
}
else
{
lean_object* v_head_809_; lean_object* v_tail_810_; lean_object* v___x_812_; uint8_t v_isShared_813_; uint8_t v_isSharedCheck_839_; 
v_head_809_ = lean_ctor_get(v_a_806_, 0);
v_tail_810_ = lean_ctor_get(v_a_806_, 1);
v_isSharedCheck_839_ = !lean_is_exclusive(v_a_806_);
if (v_isSharedCheck_839_ == 0)
{
v___x_812_ = v_a_806_;
v_isShared_813_ = v_isSharedCheck_839_;
goto v_resetjp_811_;
}
else
{
lean_inc(v_tail_810_);
lean_inc(v_head_809_);
lean_dec(v_a_806_);
v___x_812_ = lean_box(0);
v_isShared_813_ = v_isSharedCheck_839_;
goto v_resetjp_811_;
}
v_resetjp_811_:
{
lean_object* v___y_815_; lean_object* v_snd_820_; uint8_t v___x_821_; 
v_snd_820_ = lean_ctor_get(v_head_809_, 1);
v___x_821_ = lean_unbox(v_snd_820_);
if (v___x_821_ == 0)
{
lean_object* v_fst_822_; lean_object* v___x_823_; lean_object* v_toFun_824_; lean_object* v___x_825_; lean_object* v_toFun_826_; lean_object* v___x_827_; lean_object* v_toFun_828_; lean_object* v___x_829_; lean_object* v_toFun_830_; lean_object* v___x_831_; lean_object* v___x_832_; lean_object* v___x_833_; lean_object* v___x_834_; lean_object* v___x_835_; lean_object* v___x_836_; 
v_fst_822_ = lean_ctor_get(v_head_809_, 0);
lean_inc(v_fst_822_);
lean_dec(v_head_809_);
v___x_823_ = lean_obj_once(&lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__4___redArg___closed__0, &lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__4___redArg___closed__0_once, _init_lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__4___redArg___closed__0);
v_toFun_824_ = lean_ctor_get(v___x_823_, 0);
v___x_825_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0);
v_toFun_826_ = lean_ctor_get(v___x_825_, 0);
v___x_827_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__12___redArg___closed__0, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__12___redArg___closed__0_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__12___redArg___closed__0);
v_toFun_828_ = lean_ctor_get(v___x_827_, 0);
v___x_829_ = lean_obj_once(&lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0, &lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0_once, _init_lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0);
v_toFun_830_ = lean_ctor_get(v___x_829_, 0);
lean_inc(v_f_805_);
v___x_831_ = lean_apply_1(v_f_805_, v_fst_822_);
lean_inc(v_toFun_824_);
v___x_832_ = lean_apply_1(v_toFun_824_, v___x_831_);
lean_inc(v_toFun_826_);
v___x_833_ = lean_apply_1(v_toFun_826_, v___x_832_);
v___x_834_ = lp_mathlib_FreeGroup_invRev___redArg(v___x_833_);
lean_inc(v_toFun_828_);
v___x_835_ = lean_apply_1(v_toFun_828_, v___x_834_);
lean_inc(v_toFun_830_);
v___x_836_ = lean_apply_1(v_toFun_830_, v___x_835_);
v___y_815_ = v___x_836_;
goto v___jp_814_;
}
else
{
lean_object* v_fst_837_; lean_object* v___x_838_; 
v_fst_837_ = lean_ctor_get(v_head_809_, 0);
lean_inc(v_fst_837_);
lean_dec(v_head_809_);
lean_inc(v_f_805_);
v___x_838_ = lean_apply_1(v_f_805_, v_fst_837_);
v___y_815_ = v___x_838_;
goto v___jp_814_;
}
v___jp_814_:
{
lean_object* v___x_817_; 
if (v_isShared_813_ == 0)
{
lean_ctor_set(v___x_812_, 1, v_a_807_);
lean_ctor_set(v___x_812_, 0, v___y_815_);
v___x_817_ = v___x_812_;
goto v_reusejp_816_;
}
else
{
lean_object* v_reuseFailAlloc_819_; 
v_reuseFailAlloc_819_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_819_, 0, v___y_815_);
lean_ctor_set(v_reuseFailAlloc_819_, 1, v_a_807_);
v___x_817_ = v_reuseFailAlloc_819_;
goto v_reusejp_816_;
}
v_reusejp_816_:
{
v_a_806_ = v_tail_810_;
v_a_807_ = v___x_817_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0_spec__1___redArg(lean_object* v_f_840_, lean_object* v_L_841_){
_start:
{
lean_object* v___x_842_; lean_object* v___x_843_; lean_object* v___x_844_; 
v___x_842_ = lean_box(0);
v___x_843_ = lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0_spec__1_spec__3___redArg(v_f_840_, v_L_841_, v___x_842_);
v___x_844_ = lp_mathlib_List_prod___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__0_spec__2_spec__5___redArg(v___x_843_);
return v___x_844_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0(lean_object* v_00_u03b2_849_, lean_object* v_00_u03b1_850_){
_start:
{
lean_object* v___x_851_; 
v___x_851_ = ((lean_object*)(lp_mathlib_FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0___closed__1));
return v___x_851_;
}
}
static lean_object* _init_lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0___closed__0(void){
_start:
{
lean_object* v___x_852_; 
v___x_852_ = lp_mathlib_FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0(lean_box(0), lean_box(0));
return v___x_852_;
}
}
static lean_object* _init_lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0___closed__1(void){
_start:
{
lean_object* v___x_853_; 
v___x_853_ = lp_mathlib_Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__1(lean_box(0), lean_box(0), lean_box(0));
return v___x_853_;
}
}
static lean_object* _init_lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0___closed__2(void){
_start:
{
lean_object* v___x_854_; 
v___x_854_ = lp_mathlib_MonoidHom_toAdditive___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__2(lean_box(0), lean_box(0));
return v___x_854_;
}
}
static lean_object* _init_lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0___closed__3(void){
_start:
{
lean_object* v___x_855_; lean_object* v___x_856_; lean_object* v___x_857_; 
v___x_855_ = lean_obj_once(&lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0___closed__2, &lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0___closed__2_once, _init_lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0___closed__2);
v___x_856_ = lean_obj_once(&lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0___closed__1, &lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0___closed__1_once, _init_lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0___closed__1);
v___x_857_ = lp_mathlib_Equiv_trans___redArg(v___x_856_, v___x_855_);
return v___x_857_;
}
}
static lean_object* _init_lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0___closed__4(void){
_start:
{
lean_object* v___x_858_; lean_object* v___x_859_; lean_object* v___x_860_; 
v___x_858_ = lean_obj_once(&lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0___closed__3, &lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0___closed__3_once, _init_lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0___closed__3);
v___x_859_ = lean_obj_once(&lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0___closed__0, &lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0___closed__0_once, _init_lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0___closed__0);
v___x_860_ = lp_mathlib_Equiv_trans___redArg(v___x_859_, v___x_858_);
return v___x_860_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0(lean_object* v_00_u03b2_861_, lean_object* v_00_u03b1_862_){
_start:
{
lean_object* v___x_863_; 
v___x_863_ = lean_obj_once(&lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0___closed__4, &lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0___closed__4_once, _init_lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0___closed__4);
return v___x_863_;
}
}
static lean_object* _init_lp_mathlib_FreeAbelianGroup_map___redArg___closed__0(void){
_start:
{
lean_object* v___x_864_; 
v___x_864_ = lp_mathlib_FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0(lean_box(0), lean_box(0));
return v___x_864_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_map___redArg(lean_object* v_f_865_){
_start:
{
lean_object* v___x_866_; lean_object* v_toFun_867_; lean_object* v___f_868_; lean_object* v___x_869_; 
v___x_866_ = lean_obj_once(&lp_mathlib_FreeAbelianGroup_map___redArg___closed__0, &lp_mathlib_FreeAbelianGroup_map___redArg___closed__0_once, _init_lp_mathlib_FreeAbelianGroup_map___redArg___closed__0);
v_toFun_867_ = lean_ctor_get(v___x_866_, 0);
v___f_868_ = lean_alloc_closure((void*)(lp_mathlib_FreeAbelianGroup_map___redArg___lam__0), 2, 1);
lean_closure_set(v___f_868_, 0, v_f_865_);
lean_inc(v_toFun_867_);
v___x_869_ = lean_apply_1(v_toFun_867_, v___f_868_);
return v___x_869_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_map(lean_object* v_00_u03b1_870_, lean_object* v_00_u03b2_871_, lean_object* v_f_872_){
_start:
{
lean_object* v___x_873_; 
v___x_873_ = lp_mathlib_FreeAbelianGroup_map___redArg(v_f_872_);
return v___x_873_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0_spec__2___redArg(lean_object* v_f_874_){
_start:
{
lean_inc(v_f_874_);
return v_f_874_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0_spec__2___redArg___boxed(lean_object* v_f_875_){
_start:
{
lean_object* v_res_876_; 
v_res_876_ = lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0_spec__2___redArg(v_f_875_);
lean_dec(v_f_875_);
return v_res_876_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0_spec__2(lean_object* v_00_u03b2_877_, lean_object* v_00_u03b1_878_, lean_object* v_f_879_, lean_object* v_map__mul_880_){
_start:
{
lean_inc(v_f_879_);
return v_f_879_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0_spec__2___boxed(lean_object* v_00_u03b2_881_, lean_object* v_00_u03b1_882_, lean_object* v_f_883_, lean_object* v_map__mul_884_){
_start:
{
lean_object* v_res_885_; 
v_res_885_ = lp_mathlib_MonoidHom_mk_x27___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0_spec__2(v_00_u03b2_881_, v_00_u03b1_882_, v_f_883_, v_map__mul_884_);
lean_dec(v_f_883_);
return v_res_885_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0_spec__1(lean_object* v_00_u03b2_886_, lean_object* v_00_u03b1_887_, lean_object* v_f_888_, lean_object* v_L_889_){
_start:
{
lean_object* v___x_890_; 
v___x_890_ = lp_mathlib_FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0_spec__1___redArg(v_f_888_, v_L_889_);
return v___x_890_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__1_spec__4(lean_object* v_00_u03b1_891_, lean_object* v_00_u03b2_892_, lean_object* v_N_893_, lean_object* v_nN_894_, lean_object* v_00_u03c6_895_, lean_object* v_HN_896_){
_start:
{
lean_object* v___f_897_; 
v___f_897_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__6_spec__14___redArg___lam__0), 2, 1);
lean_closure_set(v___f_897_, 0, v_00_u03c6_895_);
return v___f_897_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0_spec__1_spec__3(lean_object* v_00_u03b1_898_, lean_object* v_f_899_, lean_object* v_00_u03b2_900_, lean_object* v_a_901_, lean_object* v_a_902_){
_start:
{
lean_object* v___x_903_; 
v___x_903_ = lp_mathlib_List_mapTR_loop___at___00FreeGroup_Lift_aux___at___00FreeGroup_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__0_spec__1_spec__3___redArg(v_f_899_, v_a_901_, v_a_902_);
return v___x_903_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__1_spec__4_spec__7(lean_object* v_00_u03b1_904_, lean_object* v_00_u03b2_905_, lean_object* v_c_906_, lean_object* v_f_907_, lean_object* v_H_908_){
_start:
{
lean_object* v___f_909_; 
v___f_909_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_seqAddGroupHom_spec__0_spec__1_spec__6_spec__14___redArg___lam__0), 2, 1);
lean_closure_set(v___f_909_, 0, v_f_907_);
return v___f_909_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_liftOn_x27___at___00Con_liftOn___at___00Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__1_spec__4_spec__7_spec__8_spec__9___redArg(lean_object* v_q_910_, lean_object* v_f_911_){
_start:
{
lean_object* v___x_912_; 
v___x_912_ = lean_apply_1(v_f_911_, v_q_910_);
return v___x_912_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Quotient_liftOn_x27___at___00Con_liftOn___at___00Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__1_spec__4_spec__7_spec__8_spec__9(lean_object* v_c_913_, lean_object* v_00_u03c6_914_, lean_object* v_q_915_, lean_object* v_f_916_, lean_object* v_h_917_){
_start:
{
lean_object* v___x_918_; 
v___x_918_ = lean_apply_1(v_f_916_, v_q_915_);
return v___x_918_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_liftOn___at___00Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__1_spec__4_spec__7_spec__8___redArg(lean_object* v_c_919_, lean_object* v_q_920_, lean_object* v_f_921_){
_start:
{
lean_object* v___x_922_; 
v___x_922_ = lean_apply_1(v_f_921_, v_q_920_);
return v___x_922_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_liftOn___at___00Con_lift___at___00QuotientGroup_lift___at___00Abelianization_lift___at___00FreeAbelianGroup_lift___at___00FreeAbelianGroup_map_spec__0_spec__1_spec__4_spec__7_spec__8(lean_object* v_00_u03b1_923_, lean_object* v_00_u03b2_924_, lean_object* v_c_925_, lean_object* v_q_926_, lean_object* v_f_927_, lean_object* v_h_928_){
_start:
{
lean_object* v___x_929_; 
v___x_929_ = lean_apply_1(v_f_927_, v_q_926_);
return v___x_929_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_mul___redArg___lam__0(lean_object* v_self_930_, lean_object* v___y_931_, lean_object* v___y_932_){
_start:
{
lean_object* v_toFun_933_; lean_object* v___x_934_; 
v_toFun_933_ = lean_ctor_get(v_self_930_, 0);
lean_inc(v_toFun_933_);
lean_dec_ref(v_self_930_);
v___x_934_ = lean_apply_2(v_toFun_933_, v___y_931_, v___y_932_);
return v___x_934_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_mul___redArg___lam__1(lean_object* v_inst_935_, lean_object* v_x_u2082_936_, lean_object* v_x_u2081_937_){
_start:
{
lean_object* v___x_938_; lean_object* v___x_939_; 
v___x_938_ = lean_apply_2(v_inst_935_, v_x_u2081_937_, v_x_u2082_936_);
v___x_939_ = lp_mathlib_FreeAbelianGroup_of___redArg(v___x_938_);
return v___x_939_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_mul___redArg___lam__2(lean_object* v_inst_940_, lean_object* v___f_941_, lean_object* v___x_942_, lean_object* v_x_943_, lean_object* v_x_u2082_944_){
_start:
{
lean_object* v___f_945_; lean_object* v___x_946_; 
v___f_945_ = lean_alloc_closure((void*)(lp_mathlib_FreeAbelianGroup_mul___redArg___lam__1), 3, 2);
lean_closure_set(v___f_945_, 0, v_inst_940_);
lean_closure_set(v___f_945_, 1, v_x_u2082_944_);
v___x_946_ = lean_apply_3(v___f_941_, v___x_942_, v___f_945_, v_x_943_);
return v___x_946_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_mul___redArg___lam__3(lean_object* v___x_947_, lean_object* v_inst_948_, lean_object* v___f_949_, lean_object* v_x_950_, lean_object* v___y_951_){
_start:
{
lean_object* v___x_952_; lean_object* v___f_953_; lean_object* v___x_954_; 
v___x_952_ = lp_mathlib_FreeAbelianGroup_lift___redArg(v___x_947_);
lean_inc_ref(v___x_952_);
lean_inc(v___f_949_);
v___f_953_ = lean_alloc_closure((void*)(lp_mathlib_FreeAbelianGroup_mul___redArg___lam__2), 5, 4);
lean_closure_set(v___f_953_, 0, v_inst_948_);
lean_closure_set(v___f_953_, 1, v___f_949_);
lean_closure_set(v___f_953_, 2, v___x_952_);
lean_closure_set(v___f_953_, 3, v_x_950_);
v___x_954_ = lean_apply_3(v___f_949_, v___x_952_, v___f_953_, v___y_951_);
return v___x_954_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_mul___redArg(lean_object* v_inst_956_){
_start:
{
lean_object* v___f_957_; lean_object* v___x_958_; lean_object* v___f_959_; 
v___f_957_ = ((lean_object*)(lp_mathlib_FreeAbelianGroup_mul___redArg___closed__0));
v___x_958_ = lean_obj_once(&lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__0, &lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__0_once, _init_lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__0);
v___f_959_ = lean_alloc_closure((void*)(lp_mathlib_FreeAbelianGroup_mul___redArg___lam__3), 5, 3);
lean_closure_set(v___f_959_, 0, v___x_958_);
lean_closure_set(v___f_959_, 1, v_inst_956_);
lean_closure_set(v___f_959_, 2, v___f_957_);
return v___f_959_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_mul(lean_object* v_00_u03b1_960_, lean_object* v_inst_961_){
_start:
{
lean_object* v___x_962_; 
v___x_962_ = lp_mathlib_FreeAbelianGroup_mul___redArg(v_inst_961_);
return v___x_962_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_distrib___redArg(lean_object* v_inst_963_){
_start:
{
lean_object* v___x_964_; lean_object* v_toAddMonoid_965_; lean_object* v_toAdd_966_; lean_object* v___x_967_; lean_object* v___x_968_; 
v___x_964_ = lean_obj_once(&lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__0, &lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__0_once, _init_lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__0);
v_toAddMonoid_965_ = lean_ctor_get(v___x_964_, 0);
v_toAdd_966_ = lean_ctor_get(v_toAddMonoid_965_, 1);
v___x_967_ = lp_mathlib_FreeAbelianGroup_mul___redArg(v_inst_963_);
lean_inc(v_toAdd_966_);
v___x_968_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_968_, 0, v___x_967_);
lean_ctor_set(v___x_968_, 1, v_toAdd_966_);
return v___x_968_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_distrib(lean_object* v_00_u03b1_969_, lean_object* v_inst_970_){
_start:
{
lean_object* v___x_971_; 
v___x_971_ = lp_mathlib_FreeAbelianGroup_distrib___redArg(v_inst_970_);
return v___x_971_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_nonUnitalNonAssocRing___redArg(lean_object* v_inst_972_){
_start:
{
lean_object* v___x_973_; lean_object* v___x_974_; lean_object* v_toMul_975_; lean_object* v___x_977_; uint8_t v_isShared_978_; uint8_t v_isSharedCheck_982_; 
v___x_973_ = lean_obj_once(&lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__0, &lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__0_once, _init_lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__0);
v___x_974_ = lp_mathlib_FreeAbelianGroup_distrib___redArg(v_inst_972_);
v_toMul_975_ = lean_ctor_get(v___x_974_, 0);
v_isSharedCheck_982_ = !lean_is_exclusive(v___x_974_);
if (v_isSharedCheck_982_ == 0)
{
lean_object* v_unused_983_; 
v_unused_983_ = lean_ctor_get(v___x_974_, 1);
lean_dec(v_unused_983_);
v___x_977_ = v___x_974_;
v_isShared_978_ = v_isSharedCheck_982_;
goto v_resetjp_976_;
}
else
{
lean_inc(v_toMul_975_);
lean_dec(v___x_974_);
v___x_977_ = lean_box(0);
v_isShared_978_ = v_isSharedCheck_982_;
goto v_resetjp_976_;
}
v_resetjp_976_:
{
lean_object* v___x_980_; 
if (v_isShared_978_ == 0)
{
lean_ctor_set(v___x_977_, 1, v_toMul_975_);
lean_ctor_set(v___x_977_, 0, v___x_973_);
v___x_980_ = v___x_977_;
goto v_reusejp_979_;
}
else
{
lean_object* v_reuseFailAlloc_981_; 
v_reuseFailAlloc_981_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_981_, 0, v___x_973_);
lean_ctor_set(v_reuseFailAlloc_981_, 1, v_toMul_975_);
v___x_980_ = v_reuseFailAlloc_981_;
goto v_reusejp_979_;
}
v_reusejp_979_:
{
return v___x_980_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_nonUnitalNonAssocRing(lean_object* v_00_u03b1_984_, lean_object* v_inst_985_){
_start:
{
lean_object* v___x_986_; 
v___x_986_ = lp_mathlib_FreeAbelianGroup_nonUnitalNonAssocRing___redArg(v_inst_985_);
return v___x_986_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_one___redArg(lean_object* v_inst_987_){
_start:
{
lean_object* v___x_988_; 
v___x_988_ = lp_mathlib_FreeAbelianGroup_of___redArg(v_inst_987_);
return v___x_988_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_one(lean_object* v_00_u03b1_989_, lean_object* v_inst_990_){
_start:
{
lean_object* v___x_991_; 
v___x_991_ = lp_mathlib_FreeAbelianGroup_of___redArg(v_inst_990_);
return v___x_991_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_nonUnitalRing___redArg(lean_object* v_inst_992_){
_start:
{
lean_object* v___x_993_; 
v___x_993_ = lp_mathlib_FreeAbelianGroup_nonUnitalNonAssocRing___redArg(v_inst_992_);
return v___x_993_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_nonUnitalRing(lean_object* v_00_u03b1_994_, lean_object* v_inst_995_){
_start:
{
lean_object* v___x_996_; 
v___x_996_ = lp_mathlib_FreeAbelianGroup_nonUnitalNonAssocRing___redArg(v_inst_995_);
return v___x_996_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_ring___redArg(lean_object* v_inst_997_){
_start:
{
lean_object* v___x_998_; lean_object* v_toAddMonoid_999_; lean_object* v_toNeg_1000_; lean_object* v_toSub_1001_; lean_object* v_toZSMul_1002_; lean_object* v___x_1003_; lean_object* v___x_1004_; lean_object* v_toOne_1005_; lean_object* v_toMul_1006_; lean_object* v___x_1008_; uint8_t v_isShared_1009_; uint8_t v_isSharedCheck_1024_; 
v___x_998_ = lean_obj_once(&lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__0, &lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__0_once, _init_lp_mathlib_FreeAbelianGroup_instMonad___lam__0___closed__0);
v_toAddMonoid_999_ = lean_ctor_get(v___x_998_, 0);
v_toNeg_1000_ = lean_ctor_get(v___x_998_, 1);
v_toSub_1001_ = lean_ctor_get(v___x_998_, 2);
v_toZSMul_1002_ = lean_ctor_get(v___x_998_, 3);
v___x_1003_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_997_);
v___x_1004_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_1003_);
v_toOne_1005_ = lean_ctor_get(v___x_1004_, 0);
lean_inc(v_toOne_1005_);
lean_dec_ref(v___x_1004_);
v_toMul_1006_ = lean_ctor_get(v_inst_997_, 1);
v_isSharedCheck_1024_ = !lean_is_exclusive(v_inst_997_);
if (v_isSharedCheck_1024_ == 0)
{
lean_object* v_unused_1025_; lean_object* v_unused_1026_; 
v_unused_1025_ = lean_ctor_get(v_inst_997_, 2);
lean_dec(v_unused_1025_);
v_unused_1026_ = lean_ctor_get(v_inst_997_, 0);
lean_dec(v_unused_1026_);
v___x_1008_ = v_inst_997_;
v_isShared_1009_ = v_isSharedCheck_1024_;
goto v_resetjp_1007_;
}
else
{
lean_inc(v_toMul_1006_);
lean_dec(v_inst_997_);
v___x_1008_ = lean_box(0);
v_isShared_1009_ = v_isSharedCheck_1024_;
goto v_resetjp_1007_;
}
v_resetjp_1007_:
{
lean_object* v___x_1010_; lean_object* v___x_1011_; lean_object* v_toMul_1012_; lean_object* v___x_1013_; lean_object* v___x_1014_; lean_object* v___x_1016_; 
v___x_1010_ = lp_mathlib_FreeAbelianGroup_nonUnitalNonAssocRing___redArg(v_toMul_1006_);
v___x_1011_ = lp_mathlib_NonUnitalRing_toNonUnitalSemiring___redArg(v___x_1010_);
v_toMul_1012_ = lean_ctor_get(v___x_1011_, 1);
lean_inc_n(v_toMul_1012_, 2);
lean_dec_ref(v___x_1011_);
v___x_1013_ = lp_mathlib_FreeAbelianGroup_of___redArg(v_toOne_1005_);
lean_inc_n(v___x_1013_, 2);
v___x_1014_ = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(v___x_1014_, 0, lean_box(0));
lean_closure_set(v___x_1014_, 1, v_toMul_1012_);
lean_closure_set(v___x_1014_, 2, v___x_1013_);
if (v_isShared_1009_ == 0)
{
lean_ctor_set(v___x_1008_, 2, v___x_1014_);
lean_ctor_set(v___x_1008_, 1, v_toMul_1012_);
lean_ctor_set(v___x_1008_, 0, v___x_1013_);
v___x_1016_ = v___x_1008_;
goto v_reusejp_1015_;
}
else
{
lean_object* v_reuseFailAlloc_1023_; 
v_reuseFailAlloc_1023_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1023_, 0, v___x_1013_);
lean_ctor_set(v_reuseFailAlloc_1023_, 1, v_toMul_1012_);
lean_ctor_set(v_reuseFailAlloc_1023_, 2, v___x_1014_);
v___x_1016_ = v_reuseFailAlloc_1023_;
goto v_reusejp_1015_;
}
v_reusejp_1015_:
{
lean_object* v_toZero_1017_; lean_object* v_toAdd_1018_; lean_object* v___x_1019_; lean_object* v___x_1020_; lean_object* v___x_1021_; lean_object* v___x_1022_; 
v_toZero_1017_ = lean_ctor_get(v_toAddMonoid_999_, 0);
v_toAdd_1018_ = lean_ctor_get(v_toAddMonoid_999_, 1);
lean_inc(v_toAdd_1018_);
lean_inc(v_toZero_1017_);
v___x_1019_ = lean_alloc_closure((void*)(lp_mathlib_Nat_unaryCast___boxed), 5, 4);
lean_closure_set(v___x_1019_, 0, lean_box(0));
lean_closure_set(v___x_1019_, 1, v___x_1013_);
lean_closure_set(v___x_1019_, 2, v_toZero_1017_);
lean_closure_set(v___x_1019_, 3, v_toAdd_1018_);
lean_inc_ref(v___x_1019_);
lean_inc_ref(v_toAddMonoid_999_);
v___x_1020_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1020_, 0, v_toAddMonoid_999_);
lean_ctor_set(v___x_1020_, 1, v___x_1016_);
lean_ctor_set(v___x_1020_, 2, v___x_1019_);
lean_inc_n(v_toNeg_1000_, 2);
v___x_1021_ = lean_alloc_closure((void*)(lp_mathlib_Int_castDef___boxed), 4, 3);
lean_closure_set(v___x_1021_, 0, lean_box(0));
lean_closure_set(v___x_1021_, 1, v___x_1019_);
lean_closure_set(v___x_1021_, 2, v_toNeg_1000_);
lean_inc(v_toZSMul_1002_);
lean_inc(v_toSub_1001_);
v___x_1022_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1022_, 0, v___x_1020_);
lean_ctor_set(v___x_1022_, 1, v_toNeg_1000_);
lean_ctor_set(v___x_1022_, 2, v_toSub_1001_);
lean_ctor_set(v___x_1022_, 3, v_toZSMul_1002_);
lean_ctor_set(v___x_1022_, 4, v___x_1021_);
return v___x_1022_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_ring(lean_object* v_00_u03b1_1027_, lean_object* v_inst_1028_){
_start:
{
lean_object* v___x_1029_; 
v___x_1029_ = lp_mathlib_FreeAbelianGroup_ring___redArg(v_inst_1028_);
return v___x_1029_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_ofMulHom(lean_object* v_00_u03b1_1031_, lean_object* v_inst_1032_){
_start:
{
lean_object* v___x_1033_; 
v___x_1033_ = ((lean_object*)(lp_mathlib_FreeAbelianGroup_ofMulHom___closed__0));
return v___x_1033_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_ofMulHom___boxed(lean_object* v_00_u03b1_1034_, lean_object* v_inst_1035_){
_start:
{
lean_object* v_res_1036_; 
v_res_1036_ = lp_mathlib_FreeAbelianGroup_ofMulHom(v_00_u03b1_1034_, v_inst_1035_);
lean_dec_ref(v_inst_1035_);
return v_res_1036_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_liftMonoid___redArg___lam__0(lean_object* v_F_1037_, lean_object* v___y_1038_){
_start:
{
lean_object* v___x_1039_; 
v___x_1039_ = lean_apply_1(v_F_1037_, v___y_1038_);
return v___x_1039_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_liftMonoid___redArg___lam__1(lean_object* v_F_1040_, lean_object* v___y_1041_){
_start:
{
lean_object* v___f_1042_; lean_object* v___x_1043_; lean_object* v___x_1044_; 
v___f_1042_ = lean_alloc_closure((void*)(lp_mathlib_FreeAbelianGroup_liftMonoid___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1042_, 0, v_F_1040_);
v___x_1043_ = ((lean_object*)(lp_mathlib_FreeAbelianGroup_ofMulHom___closed__0));
v___x_1044_ = lp_mathlib_OneHom_comp___redArg___lam__0(v___x_1043_, v___f_1042_, v___y_1041_);
return v___x_1044_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_liftMonoid___redArg___lam__2(lean_object* v_f_1045_, lean_object* v___y_1046_){
_start:
{
lean_object* v___x_1047_; 
v___x_1047_ = lean_apply_1(v_f_1045_, v___y_1046_);
return v___x_1047_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_liftMonoid___redArg___lam__3(lean_object* v___x_1048_, lean_object* v_f_1049_, lean_object* v___y_1050_){
_start:
{
lean_object* v___x_1051_; lean_object* v_toFun_1052_; lean_object* v___f_1053_; lean_object* v___x_1054_; 
v___x_1051_ = lp_mathlib_FreeAbelianGroup_lift___redArg(v___x_1048_);
v_toFun_1052_ = lean_ctor_get(v___x_1051_, 0);
lean_inc(v_toFun_1052_);
lean_dec_ref(v___x_1051_);
v___f_1053_ = lean_alloc_closure((void*)(lp_mathlib_FreeAbelianGroup_liftMonoid___redArg___lam__2), 2, 1);
lean_closure_set(v___f_1053_, 0, v_f_1049_);
v___x_1054_ = lean_apply_2(v_toFun_1052_, v___f_1053_, v___y_1050_);
return v___x_1054_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_liftMonoid___redArg(lean_object* v_inst_1056_){
_start:
{
lean_object* v___f_1057_; lean_object* v___x_1058_; lean_object* v___f_1059_; lean_object* v___x_1060_; 
v___f_1057_ = ((lean_object*)(lp_mathlib_FreeAbelianGroup_liftMonoid___redArg___closed__0));
v___x_1058_ = lp_mathlib_Ring_toAddCommGroup___redArg(v_inst_1056_);
v___f_1059_ = lean_alloc_closure((void*)(lp_mathlib_FreeAbelianGroup_liftMonoid___redArg___lam__3), 3, 1);
lean_closure_set(v___f_1059_, 0, v___x_1058_);
v___x_1060_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1060_, 0, v___f_1059_);
lean_ctor_set(v___x_1060_, 1, v___f_1057_);
return v___x_1060_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_liftMonoid___redArg___boxed(lean_object* v_inst_1061_){
_start:
{
lean_object* v_res_1062_; 
v_res_1062_ = lp_mathlib_FreeAbelianGroup_liftMonoid___redArg(v_inst_1061_);
lean_dec_ref(v_inst_1061_);
return v_res_1062_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_liftMonoid(lean_object* v_00_u03b1_1063_, lean_object* v_R_1064_, lean_object* v_inst_1065_, lean_object* v_inst_1066_){
_start:
{
lean_object* v___x_1067_; 
v___x_1067_ = lp_mathlib_FreeAbelianGroup_liftMonoid___redArg(v_inst_1066_);
return v___x_1067_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_liftMonoid___boxed(lean_object* v_00_u03b1_1068_, lean_object* v_R_1069_, lean_object* v_inst_1070_, lean_object* v_inst_1071_){
_start:
{
lean_object* v_res_1072_; 
v_res_1072_ = lp_mathlib_FreeAbelianGroup_liftMonoid(v_00_u03b1_1068_, v_R_1069_, v_inst_1070_, v_inst_1071_);
lean_dec_ref(v_inst_1071_);
lean_dec_ref(v_inst_1070_);
return v_res_1072_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_instCommRingOfCommMonoid___redArg(lean_object* v_inst_1073_){
_start:
{
lean_object* v___x_1074_; 
v___x_1074_ = lp_mathlib_FreeAbelianGroup_ring___redArg(v_inst_1073_);
return v___x_1074_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_instCommRingOfCommMonoid(lean_object* v_00_u03b1_1075_, lean_object* v_inst_1076_){
_start:
{
lean_object* v___x_1077_; 
v___x_1077_ = lp_mathlib_FreeAbelianGroup_ring___redArg(v_inst_1076_);
return v___x_1077_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___lam__0(lean_object* v_inst_1078_, lean_object* v___f_1079_, lean_object* v_n_1080_){
_start:
{
lean_object* v___x_1081_; lean_object* v_toFun_1082_; lean_object* v___x_1083_; lean_object* v_toFun_1084_; lean_object* v___x_1085_; lean_object* v___x_1086_; lean_object* v___x_1087_; lean_object* v___x_1088_; 
v___x_1081_ = lean_obj_once(&lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0, &lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0_once, _init_lp_mathlib_instInhabitedFreeAbelianGroup___aux__1___closed__0);
v_toFun_1082_ = lean_ctor_get(v___x_1081_, 0);
v___x_1083_ = lean_obj_once(&lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0, &lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0_once, _init_lp_mathlib_instAddCommGroupFreeAbelianGroup___aux__3___redArg___closed__0);
v_toFun_1084_ = lean_ctor_get(v___x_1083_, 0);
v___x_1085_ = lp_mathlib_FreeAbelianGroup_of___redArg(v_inst_1078_);
lean_inc(v_toFun_1084_);
v___x_1086_ = lean_apply_1(v_toFun_1084_, v___x_1085_);
v___x_1087_ = lp_mathlib_zpowRec___at___00instAddCommGroupFreeAbelianGroup_spec__4___redArg(v___f_1079_, v_n_1080_, v___x_1086_);
lean_inc(v_toFun_1082_);
v___x_1088_ = lean_apply_1(v_toFun_1082_, v___x_1087_);
return v___x_1088_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___lam__0___boxed(lean_object* v_inst_1089_, lean_object* v___f_1090_, lean_object* v_n_1091_){
_start:
{
lean_object* v_res_1092_; 
v_res_1092_ = lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___lam__0(v_inst_1089_, v___f_1090_, v_n_1091_);
lean_dec(v_n_1091_);
return v_res_1092_;
}
}
static lean_object* _init_lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___lam__1___closed__0(void){
_start:
{
lean_object* v___x_1093_; lean_object* v___x_1094_; 
v___x_1093_ = lean_unsigned_to_nat(1u);
v___x_1094_ = lean_nat_to_int(v___x_1093_);
return v___x_1094_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___lam__1(lean_object* v_x_1095_){
_start:
{
lean_object* v___x_1096_; 
v___x_1096_ = lean_obj_once(&lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___lam__1___closed__0, &lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___lam__1___closed__0_once, _init_lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___lam__1___closed__0);
return v___x_1096_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___lam__1___boxed(lean_object* v_x_1097_){
_start:
{
lean_object* v_res_1098_; 
v_res_1098_ = lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___lam__1(v_x_1097_);
lean_dec(v_x_1097_);
return v_res_1098_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___lam__2(lean_object* v_toFun_1099_, lean_object* v___f_1100_, lean_object* v___y_1101_){
_start:
{
lean_object* v___x_1102_; 
v___x_1102_ = lean_apply_2(v_toFun_1099_, v___f_1100_, v___y_1101_);
return v___x_1102_;
}
}
static lean_object* _init_lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___closed__0(void){
_start:
{
lean_object* v___x_1103_; lean_object* v___x_1104_; 
v___x_1103_ = lp_mathlib_Int_instAddCommGroup;
v___x_1104_ = lp_mathlib_FreeAbelianGroup_lift___redArg(v___x_1103_);
return v___x_1104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg(lean_object* v_inst_1106_){
_start:
{
lean_object* v___x_1107_; lean_object* v_toFun_1108_; lean_object* v___f_1109_; lean_object* v___f_1110_; lean_object* v___f_1111_; lean_object* v___f_1112_; lean_object* v___x_1113_; 
v___x_1107_ = lean_obj_once(&lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___closed__0, &lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___closed__0_once, _init_lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___closed__0);
v_toFun_1108_ = lean_ctor_get(v___x_1107_, 0);
v___f_1109_ = ((lean_object*)(lp_mathlib_instAddCommGroupFreeAbelianGroup___closed__0));
v___f_1110_ = lean_alloc_closure((void*)(lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1110_, 0, v_inst_1106_);
lean_closure_set(v___f_1110_, 1, v___f_1109_);
v___f_1111_ = ((lean_object*)(lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___closed__1));
lean_inc(v_toFun_1108_);
v___f_1112_ = lean_alloc_closure((void*)(lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg___lam__2), 3, 2);
lean_closure_set(v___f_1112_, 0, v_toFun_1108_);
lean_closure_set(v___f_1112_, 1, v___f_1111_);
v___x_1113_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1113_, 0, v___f_1112_);
lean_ctor_set(v___x_1113_, 1, v___f_1110_);
return v___x_1113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_uniqueEquiv(lean_object* v_T_1114_, lean_object* v_inst_1115_){
_start:
{
lean_object* v___x_1116_; 
v___x_1116_ = lp_mathlib_FreeAbelianGroup_uniqueEquiv___redArg(v_inst_1115_);
return v___x_1116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_equivOfEquiv___redArg___lam__0(lean_object* v_f_1117_, lean_object* v___y_1118_){
_start:
{
lean_object* v_toFun_1119_; lean_object* v___x_1120_; 
v_toFun_1119_ = lean_ctor_get(v_f_1117_, 0);
lean_inc(v_toFun_1119_);
lean_dec_ref(v_f_1117_);
v___x_1120_ = lean_apply_1(v_toFun_1119_, v___y_1118_);
return v___x_1120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_equivOfEquiv___redArg___lam__1(lean_object* v___f_1121_, lean_object* v___y_1122_){
_start:
{
lean_object* v___x_82__overap_1123_; lean_object* v___x_1124_; 
v___x_82__overap_1123_ = lp_mathlib_FreeAbelianGroup_map___redArg(v___f_1121_);
v___x_1124_ = lean_apply_1(v___x_82__overap_1123_, v___y_1122_);
return v___x_1124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_equivOfEquiv___redArg___lam__2(lean_object* v___x_1125_, lean_object* v___y_1126_){
_start:
{
lean_object* v_toFun_1127_; lean_object* v___x_1128_; 
v_toFun_1127_ = lean_ctor_get(v___x_1125_, 0);
lean_inc(v_toFun_1127_);
lean_dec_ref(v___x_1125_);
v___x_1128_ = lean_apply_1(v_toFun_1127_, v___y_1126_);
return v___x_1128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_equivOfEquiv___redArg(lean_object* v_f_1129_){
_start:
{
lean_object* v___f_1130_; lean_object* v___f_1131_; lean_object* v___x_1132_; lean_object* v___f_1133_; lean_object* v___f_1134_; lean_object* v___x_1135_; 
lean_inc_ref(v_f_1129_);
v___f_1130_ = lean_alloc_closure((void*)(lp_mathlib_FreeAbelianGroup_equivOfEquiv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1130_, 0, v_f_1129_);
v___f_1131_ = lean_alloc_closure((void*)(lp_mathlib_FreeAbelianGroup_equivOfEquiv___redArg___lam__1), 2, 1);
lean_closure_set(v___f_1131_, 0, v___f_1130_);
v___x_1132_ = lp_mathlib_Equiv_symm___redArg(v_f_1129_);
v___f_1133_ = lean_alloc_closure((void*)(lp_mathlib_FreeAbelianGroup_equivOfEquiv___redArg___lam__2), 2, 1);
lean_closure_set(v___f_1133_, 0, v___x_1132_);
v___f_1134_ = lean_alloc_closure((void*)(lp_mathlib_FreeAbelianGroup_equivOfEquiv___redArg___lam__1), 2, 1);
lean_closure_set(v___f_1134_, 0, v___f_1133_);
v___x_1135_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1135_, 0, v___f_1131_);
lean_ctor_set(v___x_1135_, 1, v___f_1134_);
return v___x_1135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FreeAbelianGroup_equivOfEquiv(lean_object* v_00_u03b1_1136_, lean_object* v_00_u03b2_1137_, lean_object* v_f_1138_){
_start:
{
lean_object* v___x_1139_; 
v___x_1139_ = lp_mathlib_FreeAbelianGroup_equivOfEquiv___redArg(v_f_1138_);
return v___x_1139_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_NatInt(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Abelianization_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_FreeGroup_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Control_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_FreeAbelianGroup(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_NatInt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Abelianization_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_FreeGroup_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Control_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_GroupTheory_FreeAbelianGroup(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Module_NatInt(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Abelianization_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_FreeGroup_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Control_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_FreeAbelianGroup(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_NatInt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Abelianization_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_FreeGroup_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Control_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_FreeAbelianGroup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_GroupTheory_FreeAbelianGroup(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_GroupTheory_FreeAbelianGroup(builtin);
}
#ifdef __cplusplus
}
#endif
