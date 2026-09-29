// Lean compiler output
// Module: Mathlib.Order.WithBot
// Imports: public import Init public meta import Init public import Mathlib.Basic.Nontrivial.Basic public import Mathlib.Order.TypeTags public import Mathlib.Data.Option.NAry public import Mathlib.Tactic.Contrapose public import Mathlib.Tactic.Lift public import Mathlib.Data.Option.Basic public import Mathlib.Order.Lattice public import Mathlib.Order.BoundedOrder.Basic
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
uint8_t l_Option_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Option_map_u2082___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearOrder_toLattice___redArg(lean_object*);
lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
lean_object* lp_mathlib_SemilatticeInf_toMin___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SemilatticeSup_toMax___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithBot_recBotCoe___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_WithTop_recTopCoe___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instUniqueOfIsEmpty(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instUniqueOfIsEmpty(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_unbotD___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_unbotD___redArg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_WithBot_unbotD___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithBot_unbotD___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_WithBot_unbotD___redArg___closed__0 = (const lean_object*)&lp_mathlib_WithBot_unbotD___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_WithBot_unbotD___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_unbotD___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_unbotD(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_unbotD___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_untopD___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_untopD___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_untopD(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_untopD___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_map(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_map(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_map_u2082___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_map_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_map_u2082___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_map_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_unbot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_unbot___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_unbot(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_unbot___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_untop_match__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_untop_match__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_untop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_untop___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_untop(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_untop___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_WithBot_0__WithBot_unbot_match__1_splitter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_WithBot_0__WithBot_unbot_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_WithBot_0__WithTop_untop_match__1_splitter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_WithBot_0__WithTop_untop_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instTop(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_withBotSubtypeNe___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_withBotSubtypeNe___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_withBotSubtypeNe___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_withBotSubtypeNe___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_withBotSubtypeNe___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_withBotSubtypeNe___closed__0 = (const lean_object*)&lp_mathlib_Equiv_withBotSubtypeNe___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_withBotSubtypeNe___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_withBotSubtypeNe___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_withBotSubtypeNe___closed__1 = (const lean_object*)&lp_mathlib_Equiv_withBotSubtypeNe___closed__1_value;
static const lean_ctor_object lp_mathlib_Equiv_withBotSubtypeNe___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_withBotSubtypeNe___closed__0_value),((lean_object*)&lp_mathlib_Equiv_withBotSubtypeNe___closed__1_value)}};
static const lean_object* lp_mathlib_Equiv_withBotSubtypeNe___closed__2 = (const lean_object*)&lp_mathlib_Equiv_withBotSubtypeNe___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_withBotSubtypeNe(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_withTopSubtypeNe_match__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_withTopSubtypeNe_match__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_withTopSubtypeNe(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_withBotCongr___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_withBotCongr___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_withBotCongr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_withBotCongr(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_withTopCongr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_withTopCongr(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLE(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instLE(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLT(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instLT(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instOrderBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instOrderTop(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instOrderTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instOrderTop(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instOrderBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instOrderBot(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instBoundedOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instBoundedOrder(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instBoundedOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instBoundedOrder(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_WithBot_instPreorder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_WithBot_instPreorder___closed__0 = (const lean_object*)&lp_mathlib_WithBot_instPreorder___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instPreorder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instPreorder___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instPreorder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instPreorder___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instPartialOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instPartialOrder___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instPartialOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instPartialOrder___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instPartialOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instPartialOrder___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instPartialOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instPartialOrder___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_semilatticeSup___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_semilatticeSup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_semilatticeSup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_semilatticeInf___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_semilatticeInf___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_semilatticeInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_semilatticeInf(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_semilatticeInf___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_semilatticeInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_semilatticeInf(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_semilatticeSup___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_semilatticeSup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_semilatticeSup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_lattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_lattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_lattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_lattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_distribLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_distribLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_distribLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_distribLattice(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithBot_decidableEq___aux__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_decidableEq___aux__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithBot_decidableEq___aux__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_decidableEq___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithBot_decidableEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_decidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithBot_decidableEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_decidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithTop_decidableEq___aux__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_decidableEq___aux__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithTop_decidableEq___aux__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_decidableEq___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithTop_decidableEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_decidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithTop_decidableEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_decidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithBot_decidableLE___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_decidableLE___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithBot_decidableLE(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_decidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_decidableLE_match__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_decidableLE_match__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithTop_decidableLE___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_decidableLE___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithTop_decidableLE(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_decidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_WithBot_0__WithBot_decidableLE_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_WithBot_0__WithBot_decidableLE_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_WithBot_0__WithTop_decidableLE_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_WithBot_0__WithTop_decidableLE_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithBot_decidableLT___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_decidableLT___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithBot_decidableLT(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_decidableLT___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_decidableLT_match__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_decidableLT_match__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithTop_decidableLT___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_decidableLT___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithTop_decidableLT(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_decidableLT___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_WithBot_0__WithBot_decidableLT_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_WithBot_0__WithBot_decidableLT_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_WithBot_0__WithTop_decidableLT_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_WithBot_0__WithTop_decidableLT_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithBot_linearOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_linearOrder___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithBot_linearOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_linearOrder___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithBot_linearOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_linearOrder___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithBot_linearOrder___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_linearOrder___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithBot_linearOrder___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_linearOrder___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_linearOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_linearOrder(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithTop_linearOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_linearOrder___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_WithTop_linearOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_linearOrder___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_linearOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_linearOrder(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_WithBot_toDual___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_WithBot_toDual___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_WithBot_toDual(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_toDual(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_ofDual(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_ofDual(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instUniqueOfIsEmpty(lean_object* v_00_u03b1_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_box(0);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instUniqueOfIsEmpty(lean_object* v_00_u03b1_4_, lean_object* v_inst_5_){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lean_box(0);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_unbotD___redArg___lam__0(lean_object* v___y_7_){
_start:
{
lean_inc(v___y_7_);
return v___y_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_unbotD___redArg___lam__0___boxed(lean_object* v___y_8_){
_start:
{
lean_object* v_res_9_; 
v_res_9_ = lp_mathlib_WithBot_unbotD___redArg___lam__0(v___y_8_);
lean_dec(v___y_8_);
return v_res_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_unbotD___redArg(lean_object* v_d_11_, lean_object* v_x_12_){
_start:
{
lean_object* v___f_13_; lean_object* v___x_14_; 
v___f_13_ = ((lean_object*)(lp_mathlib_WithBot_unbotD___redArg___closed__0));
v___x_14_ = lp_mathlib_WithBot_recBotCoe___redArg(v_d_11_, v___f_13_, v_x_12_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_unbotD___redArg___boxed(lean_object* v_d_15_, lean_object* v_x_16_){
_start:
{
lean_object* v_res_17_; 
v_res_17_ = lp_mathlib_WithBot_unbotD___redArg(v_d_15_, v_x_16_);
lean_dec(v_d_15_);
return v_res_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_unbotD(lean_object* v_00_u03b1_18_, lean_object* v_d_19_, lean_object* v_x_20_){
_start:
{
lean_object* v___x_21_; 
v___x_21_ = lp_mathlib_WithBot_unbotD___redArg(v_d_19_, v_x_20_);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_unbotD___boxed(lean_object* v_00_u03b1_22_, lean_object* v_d_23_, lean_object* v_x_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_mathlib_WithBot_unbotD(v_00_u03b1_22_, v_d_23_, v_x_24_);
lean_dec(v_d_23_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_untopD___redArg(lean_object* v_d_26_, lean_object* v_x_27_){
_start:
{
lean_object* v___f_28_; lean_object* v___x_29_; 
v___f_28_ = ((lean_object*)(lp_mathlib_WithBot_unbotD___redArg___closed__0));
v___x_29_ = lp_mathlib_WithTop_recTopCoe___redArg(v_d_26_, v___f_28_, v_x_27_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_untopD___redArg___boxed(lean_object* v_d_30_, lean_object* v_x_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_mathlib_WithTop_untopD___redArg(v_d_30_, v_x_31_);
lean_dec(v_d_30_);
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_untopD(lean_object* v_00_u03b1_33_, lean_object* v_d_34_, lean_object* v_x_35_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_mathlib_WithTop_untopD___redArg(v_d_34_, v_x_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_untopD___boxed(lean_object* v_00_u03b1_37_, lean_object* v_d_38_, lean_object* v_x_39_){
_start:
{
lean_object* v_res_40_; 
v_res_40_ = lp_mathlib_WithTop_untopD(v_00_u03b1_37_, v_d_38_, v_x_39_);
lean_dec(v_d_38_);
return v_res_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_map___redArg(lean_object* v_f_41_, lean_object* v_a_42_){
_start:
{
if (lean_obj_tag(v_a_42_) == 0)
{
lean_object* v___x_43_; 
lean_dec(v_f_41_);
v___x_43_ = lean_box(0);
return v___x_43_;
}
else
{
lean_object* v_val_44_; lean_object* v___x_46_; uint8_t v_isShared_47_; uint8_t v_isSharedCheck_52_; 
v_val_44_ = lean_ctor_get(v_a_42_, 0);
v_isSharedCheck_52_ = !lean_is_exclusive(v_a_42_);
if (v_isSharedCheck_52_ == 0)
{
v___x_46_ = v_a_42_;
v_isShared_47_ = v_isSharedCheck_52_;
goto v_resetjp_45_;
}
else
{
lean_inc(v_val_44_);
lean_dec(v_a_42_);
v___x_46_ = lean_box(0);
v_isShared_47_ = v_isSharedCheck_52_;
goto v_resetjp_45_;
}
v_resetjp_45_:
{
lean_object* v___x_48_; lean_object* v___x_50_; 
v___x_48_ = lean_apply_1(v_f_41_, v_val_44_);
if (v_isShared_47_ == 0)
{
lean_ctor_set(v___x_46_, 0, v___x_48_);
v___x_50_ = v___x_46_;
goto v_reusejp_49_;
}
else
{
lean_object* v_reuseFailAlloc_51_; 
v_reuseFailAlloc_51_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_51_, 0, v___x_48_);
v___x_50_ = v_reuseFailAlloc_51_;
goto v_reusejp_49_;
}
v_reusejp_49_:
{
return v___x_50_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_map(lean_object* v_00_u03b1_53_, lean_object* v_00_u03b2_54_, lean_object* v_f_55_, lean_object* v_a_56_){
_start:
{
lean_object* v___x_57_; 
v___x_57_ = lp_mathlib_WithBot_map___redArg(v_f_55_, v_a_56_);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_map___redArg(lean_object* v_f_58_, lean_object* v_a_59_){
_start:
{
if (lean_obj_tag(v_a_59_) == 0)
{
lean_object* v___x_60_; 
lean_dec(v_f_58_);
v___x_60_ = lean_box(0);
return v___x_60_;
}
else
{
lean_object* v_val_61_; lean_object* v___x_63_; uint8_t v_isShared_64_; uint8_t v_isSharedCheck_69_; 
v_val_61_ = lean_ctor_get(v_a_59_, 0);
v_isSharedCheck_69_ = !lean_is_exclusive(v_a_59_);
if (v_isSharedCheck_69_ == 0)
{
v___x_63_ = v_a_59_;
v_isShared_64_ = v_isSharedCheck_69_;
goto v_resetjp_62_;
}
else
{
lean_inc(v_val_61_);
lean_dec(v_a_59_);
v___x_63_ = lean_box(0);
v_isShared_64_ = v_isSharedCheck_69_;
goto v_resetjp_62_;
}
v_resetjp_62_:
{
lean_object* v___x_65_; lean_object* v___x_67_; 
v___x_65_ = lean_apply_1(v_f_58_, v_val_61_);
if (v_isShared_64_ == 0)
{
lean_ctor_set(v___x_63_, 0, v___x_65_);
v___x_67_ = v___x_63_;
goto v_reusejp_66_;
}
else
{
lean_object* v_reuseFailAlloc_68_; 
v_reuseFailAlloc_68_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_68_, 0, v___x_65_);
v___x_67_ = v_reuseFailAlloc_68_;
goto v_reusejp_66_;
}
v_reusejp_66_:
{
return v___x_67_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_map(lean_object* v_00_u03b1_70_, lean_object* v_00_u03b2_71_, lean_object* v_f_72_, lean_object* v_a_73_){
_start:
{
lean_object* v___x_74_; 
v___x_74_ = lp_mathlib_WithTop_map___redArg(v_f_72_, v_a_73_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_map_u2082___redArg(lean_object* v_f_75_, lean_object* v_a_76_, lean_object* v_b_77_){
_start:
{
lean_object* v___x_78_; 
v___x_78_ = lp_mathlib_Option_map_u2082___redArg(v_f_75_, v_a_76_, v_b_77_);
return v___x_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_map_u2082(lean_object* v_00_u03b1_79_, lean_object* v_00_u03b2_80_, lean_object* v_00_u03b3_81_, lean_object* v_f_82_, lean_object* v_a_83_, lean_object* v_b_84_){
_start:
{
lean_object* v___x_85_; 
v___x_85_ = lp_mathlib_Option_map_u2082___redArg(v_f_82_, v_a_83_, v_b_84_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_map_u2082___redArg(lean_object* v_f_86_, lean_object* v_a_87_, lean_object* v_b_88_){
_start:
{
lean_object* v___x_89_; 
v___x_89_ = lp_mathlib_Option_map_u2082___redArg(v_f_86_, v_a_87_, v_b_88_);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_map_u2082(lean_object* v_00_u03b1_90_, lean_object* v_00_u03b2_91_, lean_object* v_00_u03b3_92_, lean_object* v_f_93_, lean_object* v_a_94_, lean_object* v_b_95_){
_start:
{
lean_object* v___x_96_; 
v___x_96_ = lp_mathlib_Option_map_u2082___redArg(v_f_93_, v_a_94_, v_b_95_);
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_unbot___redArg(lean_object* v_x_97_){
_start:
{
lean_object* v_val_98_; 
v_val_98_ = lean_ctor_get(v_x_97_, 0);
lean_inc(v_val_98_);
return v_val_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_unbot___redArg___boxed(lean_object* v_x_99_){
_start:
{
lean_object* v_res_100_; 
v_res_100_ = lp_mathlib_WithBot_unbot___redArg(v_x_99_);
lean_dec(v_x_99_);
return v_res_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_unbot(lean_object* v_00_u03b1_101_, lean_object* v_x_102_, lean_object* v_x_103_){
_start:
{
lean_object* v_val_104_; 
v_val_104_ = lean_ctor_get(v_x_102_, 0);
lean_inc(v_val_104_);
return v_val_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_unbot___boxed(lean_object* v_00_u03b1_105_, lean_object* v_x_106_, lean_object* v_x_107_){
_start:
{
lean_object* v_res_108_; 
v_res_108_ = lp_mathlib_WithBot_unbot(v_00_u03b1_105_, v_x_106_, v_x_107_);
lean_dec(v_x_106_);
return v_res_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_untop_match__1___redArg(lean_object* v_x_109_, lean_object* v_h__1_110_){
_start:
{
lean_object* v_val_111_; lean_object* v___x_112_; 
v_val_111_ = lean_ctor_get(v_x_109_, 0);
lean_inc(v_val_111_);
lean_dec(v_x_109_);
v___x_112_ = lean_apply_2(v_h__1_110_, v_val_111_, lean_box(0));
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_untop_match__1(lean_object* v_00_u03b1_113_, lean_object* v_motive_114_, lean_object* v_x_115_, lean_object* v_x_116_, lean_object* v_h__1_117_){
_start:
{
lean_object* v___x_118_; 
v___x_118_ = lp_mathlib_WithTop_untop_match__1___redArg(v_x_115_, v_h__1_117_);
return v___x_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_untop___redArg(lean_object* v_x_119_){
_start:
{
lean_object* v_val_120_; 
v_val_120_ = lean_ctor_get(v_x_119_, 0);
lean_inc(v_val_120_);
return v_val_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_untop___redArg___boxed(lean_object* v_x_121_){
_start:
{
lean_object* v_res_122_; 
v_res_122_ = lp_mathlib_WithTop_untop___redArg(v_x_121_);
lean_dec(v_x_121_);
return v_res_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_untop(lean_object* v_00_u03b1_123_, lean_object* v_x_124_, lean_object* v_x_125_){
_start:
{
lean_object* v_val_126_; 
v_val_126_ = lean_ctor_get(v_x_124_, 0);
lean_inc(v_val_126_);
return v_val_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_untop___boxed(lean_object* v_00_u03b1_127_, lean_object* v_x_128_, lean_object* v_x_129_){
_start:
{
lean_object* v_res_130_; 
v_res_130_ = lp_mathlib_WithTop_untop(v_00_u03b1_127_, v_x_128_, v_x_129_);
lean_dec(v_x_128_);
return v_res_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_WithBot_0__WithBot_unbot_match__1_splitter___redArg(lean_object* v_x_131_, lean_object* v_h__1_132_){
_start:
{
lean_object* v_val_133_; lean_object* v___x_134_; 
v_val_133_ = lean_ctor_get(v_x_131_, 0);
lean_inc(v_val_133_);
lean_dec(v_x_131_);
v___x_134_ = lean_apply_2(v_h__1_132_, v_val_133_, lean_box(0));
return v___x_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_WithBot_0__WithBot_unbot_match__1_splitter(lean_object* v_00_u03b1_135_, lean_object* v_motive_136_, lean_object* v_x_137_, lean_object* v_x_138_, lean_object* v_h__1_139_){
_start:
{
lean_object* v_val_140_; lean_object* v___x_141_; 
v_val_140_ = lean_ctor_get(v_x_137_, 0);
lean_inc(v_val_140_);
lean_dec(v_x_137_);
v___x_141_ = lean_apply_2(v_h__1_139_, v_val_140_, lean_box(0));
return v___x_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_WithBot_0__WithTop_untop_match__1_splitter___redArg(lean_object* v_x_142_, lean_object* v_h__1_143_){
_start:
{
lean_object* v_val_144_; lean_object* v___x_145_; 
v_val_144_ = lean_ctor_get(v_x_142_, 0);
lean_inc(v_val_144_);
lean_dec(v_x_142_);
v___x_145_ = lean_apply_2(v_h__1_143_, v_val_144_, lean_box(0));
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_WithBot_0__WithTop_untop_match__1_splitter(lean_object* v_00_u03b1_146_, lean_object* v_motive_147_, lean_object* v_x_148_, lean_object* v_x_149_, lean_object* v_h__1_150_){
_start:
{
lean_object* v_val_151_; lean_object* v___x_152_; 
v_val_151_ = lean_ctor_get(v_x_148_, 0);
lean_inc(v_val_151_);
lean_dec(v_x_148_);
v___x_152_ = lean_apply_2(v_h__1_150_, v_val_151_, lean_box(0));
return v___x_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instTop___redArg(lean_object* v_inst_153_){
_start:
{
lean_object* v___x_154_; 
v___x_154_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_154_, 0, v_inst_153_);
return v___x_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instTop(lean_object* v_00_u03b1_155_, lean_object* v_inst_156_){
_start:
{
lean_object* v___x_157_; 
v___x_157_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_157_, 0, v_inst_156_);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instBot___redArg(lean_object* v_inst_158_){
_start:
{
lean_object* v___x_159_; 
v___x_159_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_159_, 0, v_inst_158_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instBot(lean_object* v_00_u03b1_160_, lean_object* v_inst_161_){
_start:
{
lean_object* v___x_162_; 
v___x_162_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_162_, 0, v_inst_161_);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_withBotSubtypeNe___lam__0(lean_object* v_x_163_){
_start:
{
lean_object* v_val_164_; 
v_val_164_ = lean_ctor_get(v_x_163_, 0);
lean_inc(v_val_164_);
return v_val_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_withBotSubtypeNe___lam__0___boxed(lean_object* v_x_165_){
_start:
{
lean_object* v_res_166_; 
v_res_166_ = lp_mathlib_Equiv_withBotSubtypeNe___lam__0(v_x_165_);
lean_dec(v_x_165_);
return v_res_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_withBotSubtypeNe___lam__1(lean_object* v_x_167_){
_start:
{
lean_object* v___x_168_; 
v___x_168_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_168_, 0, v_x_167_);
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_withBotSubtypeNe(lean_object* v_00_u03b1_174_){
_start:
{
lean_object* v___x_175_; 
v___x_175_ = ((lean_object*)(lp_mathlib_Equiv_withBotSubtypeNe___closed__2));
return v___x_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_withTopSubtypeNe_match__1___redArg(lean_object* v_x_176_, lean_object* v_h__1_177_){
_start:
{
lean_object* v___x_178_; 
v___x_178_ = lean_apply_2(v_h__1_177_, v_x_176_, lean_box(0));
return v___x_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_withTopSubtypeNe_match__1(lean_object* v_00_u03b1_179_, lean_object* v_motive_180_, lean_object* v_x_181_, lean_object* v_h__1_182_){
_start:
{
lean_object* v___x_183_; 
v___x_183_ = lean_apply_2(v_h__1_182_, v_x_181_, lean_box(0));
return v___x_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_withTopSubtypeNe(lean_object* v_00_u03b1_184_){
_start:
{
lean_object* v___x_185_; 
v___x_185_ = ((lean_object*)(lp_mathlib_Equiv_withBotSubtypeNe___closed__2));
return v___x_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_withBotCongr___redArg___lam__0(lean_object* v_e_186_, lean_object* v___y_187_){
_start:
{
lean_object* v_toFun_188_; lean_object* v___x_189_; 
v_toFun_188_ = lean_ctor_get(v_e_186_, 0);
lean_inc(v_toFun_188_);
lean_dec_ref(v_e_186_);
v___x_189_ = lean_apply_1(v_toFun_188_, v___y_187_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_withBotCongr___redArg___lam__1(lean_object* v___x_190_, lean_object* v___y_191_){
_start:
{
lean_object* v_toFun_192_; lean_object* v___x_193_; 
v_toFun_192_ = lean_ctor_get(v___x_190_, 0);
lean_inc(v_toFun_192_);
lean_dec_ref(v___x_190_);
v___x_193_ = lean_apply_1(v_toFun_192_, v___y_191_);
return v___x_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_withBotCongr___redArg(lean_object* v_e_194_){
_start:
{
lean_object* v___f_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___f_198_; lean_object* v___x_199_; lean_object* v___x_200_; 
lean_inc_ref(v_e_194_);
v___f_195_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_withBotCongr___redArg___lam__0), 2, 1);
lean_closure_set(v___f_195_, 0, v_e_194_);
v___x_196_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_map), 4, 3);
lean_closure_set(v___x_196_, 0, lean_box(0));
lean_closure_set(v___x_196_, 1, lean_box(0));
lean_closure_set(v___x_196_, 2, v___f_195_);
v___x_197_ = lp_mathlib_Equiv_symm___redArg(v_e_194_);
v___f_198_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_withBotCongr___redArg___lam__1), 2, 1);
lean_closure_set(v___f_198_, 0, v___x_197_);
v___x_199_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_map), 4, 3);
lean_closure_set(v___x_199_, 0, lean_box(0));
lean_closure_set(v___x_199_, 1, lean_box(0));
lean_closure_set(v___x_199_, 2, v___f_198_);
v___x_200_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_200_, 0, v___x_196_);
lean_ctor_set(v___x_200_, 1, v___x_199_);
return v___x_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_withBotCongr(lean_object* v_00_u03b1_201_, lean_object* v_00_u03b2_202_, lean_object* v_e_203_){
_start:
{
lean_object* v___x_204_; 
v___x_204_ = lp_mathlib_Equiv_withBotCongr___redArg(v_e_203_);
return v___x_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_withTopCongr___redArg(lean_object* v_e_205_){
_start:
{
lean_object* v___f_206_; lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___f_209_; lean_object* v___x_210_; lean_object* v___x_211_; 
lean_inc_ref(v_e_205_);
v___f_206_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_withBotCongr___redArg___lam__0), 2, 1);
lean_closure_set(v___f_206_, 0, v_e_205_);
v___x_207_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_map), 4, 3);
lean_closure_set(v___x_207_, 0, lean_box(0));
lean_closure_set(v___x_207_, 1, lean_box(0));
lean_closure_set(v___x_207_, 2, v___f_206_);
v___x_208_ = lp_mathlib_Equiv_symm___redArg(v_e_205_);
v___f_209_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_withBotCongr___redArg___lam__1), 2, 1);
lean_closure_set(v___f_209_, 0, v___x_208_);
v___x_210_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_map), 4, 3);
lean_closure_set(v___x_210_, 0, lean_box(0));
lean_closure_set(v___x_210_, 1, lean_box(0));
lean_closure_set(v___x_210_, 2, v___f_209_);
v___x_211_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_211_, 0, v___x_207_);
lean_ctor_set(v___x_211_, 1, v___x_210_);
return v___x_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_withTopCongr(lean_object* v_00_u03b1_212_, lean_object* v_00_u03b2_213_, lean_object* v_e_214_){
_start:
{
lean_object* v___x_215_; 
v___x_215_ = lp_mathlib_Equiv_withTopCongr___redArg(v_e_214_);
return v___x_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLE(lean_object* v_00_u03b1_216_, lean_object* v_inst_217_){
_start:
{
lean_object* v___x_218_; 
v___x_218_ = lean_box(0);
return v___x_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instLE(lean_object* v_00_u03b1_219_, lean_object* v_inst_220_){
_start:
{
lean_object* v___x_221_; 
v___x_221_ = lean_box(0);
return v___x_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLT(lean_object* v_00_u03b1_222_, lean_object* v_inst_223_){
_start:
{
lean_object* v___x_224_; 
v___x_224_ = lean_box(0);
return v___x_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instLT(lean_object* v_00_u03b1_225_, lean_object* v_inst_226_){
_start:
{
lean_object* v___x_227_; 
v___x_227_ = lean_box(0);
return v___x_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instOrderBot(lean_object* v_00_u03b1_228_, lean_object* v_inst_229_){
_start:
{
lean_object* v___x_230_; 
v___x_230_ = lean_box(0);
return v___x_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instOrderTop(lean_object* v_00_u03b1_231_, lean_object* v_inst_232_){
_start:
{
lean_object* v___x_233_; 
v___x_233_ = lean_box(0);
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instOrderTop___redArg(lean_object* v_inst_234_){
_start:
{
lean_object* v___x_235_; 
v___x_235_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_235_, 0, v_inst_234_);
return v___x_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instOrderTop(lean_object* v_00_u03b1_236_, lean_object* v_inst_237_, lean_object* v_inst_238_){
_start:
{
lean_object* v___x_239_; 
v___x_239_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_239_, 0, v_inst_238_);
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instOrderBot___redArg(lean_object* v_inst_240_){
_start:
{
lean_object* v___x_241_; 
v___x_241_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_241_, 0, v_inst_240_);
return v___x_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instOrderBot(lean_object* v_00_u03b1_242_, lean_object* v_inst_243_, lean_object* v_inst_244_){
_start:
{
lean_object* v___x_245_; 
v___x_245_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_245_, 0, v_inst_244_);
return v___x_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instBoundedOrder___redArg(lean_object* v_inst_246_){
_start:
{
lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; 
v___x_247_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_247_, 0, v_inst_246_);
v___x_248_ = lean_box(0);
v___x_249_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_249_, 0, v___x_247_);
lean_ctor_set(v___x_249_, 1, v___x_248_);
return v___x_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instBoundedOrder(lean_object* v_00_u03b1_250_, lean_object* v_inst_251_, lean_object* v_inst_252_){
_start:
{
lean_object* v___x_253_; 
v___x_253_ = lp_mathlib_WithBot_instBoundedOrder___redArg(v_inst_252_);
return v___x_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instBoundedOrder___redArg(lean_object* v_inst_254_){
_start:
{
lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; 
v___x_255_ = lean_box(0);
v___x_256_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_256_, 0, v_inst_254_);
v___x_257_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_257_, 0, v___x_255_);
lean_ctor_set(v___x_257_, 1, v___x_256_);
return v___x_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instBoundedOrder(lean_object* v_00_u03b1_258_, lean_object* v_inst_259_, lean_object* v_inst_260_){
_start:
{
lean_object* v___x_261_; 
v___x_261_ = lp_mathlib_WithTop_instBoundedOrder___redArg(v_inst_260_);
return v___x_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instPreorder(lean_object* v_00_u03b1_265_, lean_object* v_inst_266_){
_start:
{
lean_object* v___x_267_; 
v___x_267_ = ((lean_object*)(lp_mathlib_WithBot_instPreorder___closed__0));
return v___x_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instPreorder___boxed(lean_object* v_00_u03b1_268_, lean_object* v_inst_269_){
_start:
{
lean_object* v_res_270_; 
v_res_270_ = lp_mathlib_WithBot_instPreorder(v_00_u03b1_268_, v_inst_269_);
lean_dec_ref(v_inst_269_);
return v_res_270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instPreorder(lean_object* v_00_u03b1_271_, lean_object* v_inst_272_){
_start:
{
lean_object* v___x_273_; 
v___x_273_ = ((lean_object*)(lp_mathlib_WithBot_instPreorder___closed__0));
return v___x_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instPreorder___boxed(lean_object* v_00_u03b1_274_, lean_object* v_inst_275_){
_start:
{
lean_object* v_res_276_; 
v_res_276_ = lp_mathlib_WithTop_instPreorder(v_00_u03b1_274_, v_inst_275_);
lean_dec_ref(v_inst_275_);
return v_res_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instPartialOrder___redArg(lean_object* v_inst_277_){
_start:
{
lean_object* v___x_278_; 
v___x_278_ = lp_mathlib_WithBot_instPreorder(lean_box(0), v_inst_277_);
return v___x_278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instPartialOrder___redArg___boxed(lean_object* v_inst_279_){
_start:
{
lean_object* v_res_280_; 
v_res_280_ = lp_mathlib_WithBot_instPartialOrder___redArg(v_inst_279_);
lean_dec_ref(v_inst_279_);
return v_res_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instPartialOrder(lean_object* v_00_u03b1_281_, lean_object* v_inst_282_){
_start:
{
lean_object* v___x_283_; 
v___x_283_ = lp_mathlib_WithBot_instPreorder(lean_box(0), v_inst_282_);
return v___x_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instPartialOrder___boxed(lean_object* v_00_u03b1_284_, lean_object* v_inst_285_){
_start:
{
lean_object* v_res_286_; 
v_res_286_ = lp_mathlib_WithBot_instPartialOrder(v_00_u03b1_284_, v_inst_285_);
lean_dec_ref(v_inst_285_);
return v_res_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instPartialOrder___redArg(lean_object* v_inst_287_){
_start:
{
lean_object* v___x_288_; 
v___x_288_ = lp_mathlib_WithTop_instPreorder(lean_box(0), v_inst_287_);
return v___x_288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instPartialOrder___redArg___boxed(lean_object* v_inst_289_){
_start:
{
lean_object* v_res_290_; 
v_res_290_ = lp_mathlib_WithTop_instPartialOrder___redArg(v_inst_289_);
lean_dec_ref(v_inst_289_);
return v_res_290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instPartialOrder(lean_object* v_00_u03b1_291_, lean_object* v_inst_292_){
_start:
{
lean_object* v___x_293_; 
v___x_293_ = lp_mathlib_WithTop_instPreorder(lean_box(0), v_inst_292_);
return v___x_293_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instPartialOrder___boxed(lean_object* v_00_u03b1_294_, lean_object* v_inst_295_){
_start:
{
lean_object* v_res_296_; 
v_res_296_ = lp_mathlib_WithTop_instPartialOrder(v_00_u03b1_294_, v_inst_295_);
lean_dec_ref(v_inst_295_);
return v_res_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_semilatticeSup___redArg___lam__0(lean_object* v_sup_297_, lean_object* v_x_298_, lean_object* v_x_299_){
_start:
{
if (lean_obj_tag(v_x_298_) == 0)
{
lean_dec(v_sup_297_);
return v_x_299_;
}
else
{
if (lean_obj_tag(v_x_299_) == 0)
{
lean_dec(v_sup_297_);
return v_x_298_;
}
else
{
lean_object* v_val_300_; lean_object* v_val_301_; lean_object* v___x_303_; uint8_t v_isShared_304_; uint8_t v_isSharedCheck_309_; 
v_val_300_ = lean_ctor_get(v_x_298_, 0);
lean_inc(v_val_300_);
lean_dec_ref_known(v_x_298_, 1);
v_val_301_ = lean_ctor_get(v_x_299_, 0);
v_isSharedCheck_309_ = !lean_is_exclusive(v_x_299_);
if (v_isSharedCheck_309_ == 0)
{
v___x_303_ = v_x_299_;
v_isShared_304_ = v_isSharedCheck_309_;
goto v_resetjp_302_;
}
else
{
lean_inc(v_val_301_);
lean_dec(v_x_299_);
v___x_303_ = lean_box(0);
v_isShared_304_ = v_isSharedCheck_309_;
goto v_resetjp_302_;
}
v_resetjp_302_:
{
lean_object* v___x_305_; lean_object* v___x_307_; 
v___x_305_ = lean_apply_2(v_sup_297_, v_val_300_, v_val_301_);
if (v_isShared_304_ == 0)
{
lean_ctor_set(v___x_303_, 0, v___x_305_);
v___x_307_ = v___x_303_;
goto v_reusejp_306_;
}
else
{
lean_object* v_reuseFailAlloc_308_; 
v_reuseFailAlloc_308_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_308_, 0, v___x_305_);
v___x_307_ = v_reuseFailAlloc_308_;
goto v_reusejp_306_;
}
v_reusejp_306_:
{
return v___x_307_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_semilatticeSup___redArg(lean_object* v_inst_310_){
_start:
{
lean_object* v_toPartialOrder_311_; lean_object* v_sup_312_; lean_object* v___x_314_; uint8_t v_isShared_315_; uint8_t v_isSharedCheck_321_; 
v_toPartialOrder_311_ = lean_ctor_get(v_inst_310_, 0);
v_sup_312_ = lean_ctor_get(v_inst_310_, 1);
v_isSharedCheck_321_ = !lean_is_exclusive(v_inst_310_);
if (v_isSharedCheck_321_ == 0)
{
v___x_314_ = v_inst_310_;
v_isShared_315_ = v_isSharedCheck_321_;
goto v_resetjp_313_;
}
else
{
lean_inc(v_sup_312_);
lean_inc(v_toPartialOrder_311_);
lean_dec(v_inst_310_);
v___x_314_ = lean_box(0);
v_isShared_315_ = v_isSharedCheck_321_;
goto v_resetjp_313_;
}
v_resetjp_313_:
{
lean_object* v___f_316_; lean_object* v___x_317_; lean_object* v___x_319_; 
v___f_316_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_semilatticeSup___redArg___lam__0), 3, 1);
lean_closure_set(v___f_316_, 0, v_sup_312_);
v___x_317_ = lp_mathlib_WithBot_instPreorder(lean_box(0), v_toPartialOrder_311_);
lean_dec_ref(v_toPartialOrder_311_);
if (v_isShared_315_ == 0)
{
lean_ctor_set(v___x_314_, 1, v___f_316_);
lean_ctor_set(v___x_314_, 0, v___x_317_);
v___x_319_ = v___x_314_;
goto v_reusejp_318_;
}
else
{
lean_object* v_reuseFailAlloc_320_; 
v_reuseFailAlloc_320_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_320_, 0, v___x_317_);
lean_ctor_set(v_reuseFailAlloc_320_, 1, v___f_316_);
v___x_319_ = v_reuseFailAlloc_320_;
goto v_reusejp_318_;
}
v_reusejp_318_:
{
return v___x_319_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_semilatticeSup(lean_object* v_00_u03b1_322_, lean_object* v_inst_323_){
_start:
{
lean_object* v___x_324_; 
v___x_324_ = lp_mathlib_WithBot_semilatticeSup___redArg(v_inst_323_);
return v___x_324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_semilatticeInf___redArg___lam__0(lean_object* v___x_325_, lean_object* v_inf_326_, lean_object* v_x_327_, lean_object* v_x_328_){
_start:
{
if (lean_obj_tag(v_x_327_) == 0)
{
lean_dec(v_inf_326_);
if (lean_obj_tag(v_x_328_) == 0)
{
lean_inc(v___x_325_);
return v___x_325_;
}
else
{
return v_x_328_;
}
}
else
{
if (lean_obj_tag(v_x_328_) == 0)
{
lean_dec(v_inf_326_);
return v_x_327_;
}
else
{
lean_object* v_val_329_; lean_object* v_val_330_; lean_object* v___x_332_; uint8_t v_isShared_333_; uint8_t v_isSharedCheck_338_; 
v_val_329_ = lean_ctor_get(v_x_327_, 0);
lean_inc(v_val_329_);
lean_dec_ref_known(v_x_327_, 1);
v_val_330_ = lean_ctor_get(v_x_328_, 0);
v_isSharedCheck_338_ = !lean_is_exclusive(v_x_328_);
if (v_isSharedCheck_338_ == 0)
{
v___x_332_ = v_x_328_;
v_isShared_333_ = v_isSharedCheck_338_;
goto v_resetjp_331_;
}
else
{
lean_inc(v_val_330_);
lean_dec(v_x_328_);
v___x_332_ = lean_box(0);
v_isShared_333_ = v_isSharedCheck_338_;
goto v_resetjp_331_;
}
v_resetjp_331_:
{
lean_object* v___x_334_; lean_object* v___x_336_; 
v___x_334_ = lean_apply_2(v_inf_326_, v_val_329_, v_val_330_);
if (v_isShared_333_ == 0)
{
lean_ctor_set(v___x_332_, 0, v___x_334_);
v___x_336_ = v___x_332_;
goto v_reusejp_335_;
}
else
{
lean_object* v_reuseFailAlloc_337_; 
v_reuseFailAlloc_337_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_337_, 0, v___x_334_);
v___x_336_ = v_reuseFailAlloc_337_;
goto v_reusejp_335_;
}
v_reusejp_335_:
{
return v___x_336_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_semilatticeInf___redArg___lam__0___boxed(lean_object* v___x_339_, lean_object* v_inf_340_, lean_object* v_x_341_, lean_object* v_x_342_){
_start:
{
lean_object* v_res_343_; 
v_res_343_ = lp_mathlib_WithTop_semilatticeInf___redArg___lam__0(v___x_339_, v_inf_340_, v_x_341_, v_x_342_);
lean_dec(v___x_339_);
return v_res_343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_semilatticeInf___redArg(lean_object* v_inst_344_){
_start:
{
lean_object* v_toPartialOrder_345_; lean_object* v_inf_346_; lean_object* v___x_348_; uint8_t v_isShared_349_; uint8_t v_isSharedCheck_356_; 
v_toPartialOrder_345_ = lean_ctor_get(v_inst_344_, 0);
v_inf_346_ = lean_ctor_get(v_inst_344_, 1);
v_isSharedCheck_356_ = !lean_is_exclusive(v_inst_344_);
if (v_isSharedCheck_356_ == 0)
{
v___x_348_ = v_inst_344_;
v_isShared_349_ = v_isSharedCheck_356_;
goto v_resetjp_347_;
}
else
{
lean_inc(v_inf_346_);
lean_inc(v_toPartialOrder_345_);
lean_dec(v_inst_344_);
v___x_348_ = lean_box(0);
v_isShared_349_ = v_isSharedCheck_356_;
goto v_resetjp_347_;
}
v_resetjp_347_:
{
lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___f_352_; lean_object* v___x_354_; 
v___x_350_ = lp_mathlib_WithTop_instPreorder(lean_box(0), v_toPartialOrder_345_);
lean_dec_ref(v_toPartialOrder_345_);
v___x_351_ = lean_box(0);
v___f_352_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_semilatticeInf___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_352_, 0, v___x_351_);
lean_closure_set(v___f_352_, 1, v_inf_346_);
if (v_isShared_349_ == 0)
{
lean_ctor_set(v___x_348_, 1, v___f_352_);
lean_ctor_set(v___x_348_, 0, v___x_350_);
v___x_354_ = v___x_348_;
goto v_reusejp_353_;
}
else
{
lean_object* v_reuseFailAlloc_355_; 
v_reuseFailAlloc_355_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_355_, 0, v___x_350_);
lean_ctor_set(v_reuseFailAlloc_355_, 1, v___f_352_);
v___x_354_ = v_reuseFailAlloc_355_;
goto v_reusejp_353_;
}
v_reusejp_353_:
{
return v___x_354_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_semilatticeInf(lean_object* v_00_u03b1_357_, lean_object* v_inst_358_){
_start:
{
lean_object* v___x_359_; 
v___x_359_ = lp_mathlib_WithTop_semilatticeInf___redArg(v_inst_358_);
return v___x_359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_semilatticeInf___redArg___lam__0(lean_object* v_inf_360_, lean_object* v_x1_361_, lean_object* v_x2_362_){
_start:
{
lean_object* v___x_363_; 
v___x_363_ = lean_apply_2(v_inf_360_, v_x1_361_, v_x2_362_);
return v___x_363_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_semilatticeInf___redArg(lean_object* v_inst_364_){
_start:
{
lean_object* v_toPartialOrder_365_; lean_object* v_inf_366_; lean_object* v___x_368_; uint8_t v_isShared_369_; uint8_t v_isSharedCheck_376_; 
v_toPartialOrder_365_ = lean_ctor_get(v_inst_364_, 0);
v_inf_366_ = lean_ctor_get(v_inst_364_, 1);
v_isSharedCheck_376_ = !lean_is_exclusive(v_inst_364_);
if (v_isSharedCheck_376_ == 0)
{
v___x_368_ = v_inst_364_;
v_isShared_369_ = v_isSharedCheck_376_;
goto v_resetjp_367_;
}
else
{
lean_inc(v_inf_366_);
lean_inc(v_toPartialOrder_365_);
lean_dec(v_inst_364_);
v___x_368_ = lean_box(0);
v_isShared_369_ = v_isSharedCheck_376_;
goto v_resetjp_367_;
}
v_resetjp_367_:
{
lean_object* v___f_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_374_; 
v___f_370_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_semilatticeInf___redArg___lam__0), 3, 1);
lean_closure_set(v___f_370_, 0, v_inf_366_);
v___x_371_ = lp_mathlib_WithBot_instPreorder(lean_box(0), v_toPartialOrder_365_);
lean_dec_ref(v_toPartialOrder_365_);
v___x_372_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_map_u2082), 6, 4);
lean_closure_set(v___x_372_, 0, lean_box(0));
lean_closure_set(v___x_372_, 1, lean_box(0));
lean_closure_set(v___x_372_, 2, lean_box(0));
lean_closure_set(v___x_372_, 3, v___f_370_);
if (v_isShared_369_ == 0)
{
lean_ctor_set(v___x_368_, 1, v___x_372_);
lean_ctor_set(v___x_368_, 0, v___x_371_);
v___x_374_ = v___x_368_;
goto v_reusejp_373_;
}
else
{
lean_object* v_reuseFailAlloc_375_; 
v_reuseFailAlloc_375_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_375_, 0, v___x_371_);
lean_ctor_set(v_reuseFailAlloc_375_, 1, v___x_372_);
v___x_374_ = v_reuseFailAlloc_375_;
goto v_reusejp_373_;
}
v_reusejp_373_:
{
return v___x_374_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_semilatticeInf(lean_object* v_00_u03b1_377_, lean_object* v_inst_378_){
_start:
{
lean_object* v___x_379_; 
v___x_379_ = lp_mathlib_WithBot_semilatticeInf___redArg(v_inst_378_);
return v___x_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_semilatticeSup___redArg___lam__0(lean_object* v_sup_380_, lean_object* v_x1_381_, lean_object* v_x2_382_){
_start:
{
lean_object* v___x_383_; 
v___x_383_ = lean_apply_2(v_sup_380_, v_x1_381_, v_x2_382_);
return v___x_383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_semilatticeSup___redArg(lean_object* v_inst_384_){
_start:
{
lean_object* v_toPartialOrder_385_; lean_object* v_sup_386_; lean_object* v___x_388_; uint8_t v_isShared_389_; uint8_t v_isSharedCheck_396_; 
v_toPartialOrder_385_ = lean_ctor_get(v_inst_384_, 0);
v_sup_386_ = lean_ctor_get(v_inst_384_, 1);
v_isSharedCheck_396_ = !lean_is_exclusive(v_inst_384_);
if (v_isSharedCheck_396_ == 0)
{
v___x_388_ = v_inst_384_;
v_isShared_389_ = v_isSharedCheck_396_;
goto v_resetjp_387_;
}
else
{
lean_inc(v_sup_386_);
lean_inc(v_toPartialOrder_385_);
lean_dec(v_inst_384_);
v___x_388_ = lean_box(0);
v_isShared_389_ = v_isSharedCheck_396_;
goto v_resetjp_387_;
}
v_resetjp_387_:
{
lean_object* v___f_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_394_; 
v___f_390_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_semilatticeSup___redArg___lam__0), 3, 1);
lean_closure_set(v___f_390_, 0, v_sup_386_);
v___x_391_ = lp_mathlib_WithTop_instPreorder(lean_box(0), v_toPartialOrder_385_);
lean_dec_ref(v_toPartialOrder_385_);
v___x_392_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_map_u2082), 6, 4);
lean_closure_set(v___x_392_, 0, lean_box(0));
lean_closure_set(v___x_392_, 1, lean_box(0));
lean_closure_set(v___x_392_, 2, lean_box(0));
lean_closure_set(v___x_392_, 3, v___f_390_);
if (v_isShared_389_ == 0)
{
lean_ctor_set(v___x_388_, 1, v___x_392_);
lean_ctor_set(v___x_388_, 0, v___x_391_);
v___x_394_ = v___x_388_;
goto v_reusejp_393_;
}
else
{
lean_object* v_reuseFailAlloc_395_; 
v_reuseFailAlloc_395_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_395_, 0, v___x_391_);
lean_ctor_set(v_reuseFailAlloc_395_, 1, v___x_392_);
v___x_394_ = v_reuseFailAlloc_395_;
goto v_reusejp_393_;
}
v_reusejp_393_:
{
return v___x_394_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_semilatticeSup(lean_object* v_00_u03b1_397_, lean_object* v_inst_398_){
_start:
{
lean_object* v___x_399_; 
v___x_399_ = lp_mathlib_WithTop_semilatticeSup___redArg(v_inst_398_);
return v___x_399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_lattice___redArg(lean_object* v_inst_400_){
_start:
{
lean_object* v_toSemilatticeSup_401_; lean_object* v_inf_402_; lean_object* v___x_404_; uint8_t v_isShared_405_; uint8_t v_isSharedCheck_412_; 
v_toSemilatticeSup_401_ = lean_ctor_get(v_inst_400_, 0);
v_inf_402_ = lean_ctor_get(v_inst_400_, 1);
v_isSharedCheck_412_ = !lean_is_exclusive(v_inst_400_);
if (v_isSharedCheck_412_ == 0)
{
v___x_404_ = v_inst_400_;
v_isShared_405_ = v_isSharedCheck_412_;
goto v_resetjp_403_;
}
else
{
lean_inc(v_inf_402_);
lean_inc(v_toSemilatticeSup_401_);
lean_dec(v_inst_400_);
v___x_404_ = lean_box(0);
v_isShared_405_ = v_isSharedCheck_412_;
goto v_resetjp_403_;
}
v_resetjp_403_:
{
lean_object* v___f_406_; lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_410_; 
v___f_406_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_semilatticeInf___redArg___lam__0), 3, 1);
lean_closure_set(v___f_406_, 0, v_inf_402_);
v___x_407_ = lp_mathlib_WithBot_semilatticeSup___redArg(v_toSemilatticeSup_401_);
v___x_408_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_map_u2082), 6, 4);
lean_closure_set(v___x_408_, 0, lean_box(0));
lean_closure_set(v___x_408_, 1, lean_box(0));
lean_closure_set(v___x_408_, 2, lean_box(0));
lean_closure_set(v___x_408_, 3, v___f_406_);
if (v_isShared_405_ == 0)
{
lean_ctor_set(v___x_404_, 1, v___x_408_);
lean_ctor_set(v___x_404_, 0, v___x_407_);
v___x_410_ = v___x_404_;
goto v_reusejp_409_;
}
else
{
lean_object* v_reuseFailAlloc_411_; 
v_reuseFailAlloc_411_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_411_, 0, v___x_407_);
lean_ctor_set(v_reuseFailAlloc_411_, 1, v___x_408_);
v___x_410_ = v_reuseFailAlloc_411_;
goto v_reusejp_409_;
}
v_reusejp_409_:
{
return v___x_410_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_lattice(lean_object* v_00_u03b1_413_, lean_object* v_inst_414_){
_start:
{
lean_object* v___x_415_; 
v___x_415_ = lp_mathlib_WithBot_lattice___redArg(v_inst_414_);
return v___x_415_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_lattice___redArg(lean_object* v_inst_416_){
_start:
{
lean_object* v_toSemilatticeSup_417_; lean_object* v_inf_418_; lean_object* v___x_420_; uint8_t v_isShared_421_; uint8_t v_isSharedCheck_428_; 
v_toSemilatticeSup_417_ = lean_ctor_get(v_inst_416_, 0);
v_inf_418_ = lean_ctor_get(v_inst_416_, 1);
v_isSharedCheck_428_ = !lean_is_exclusive(v_inst_416_);
if (v_isSharedCheck_428_ == 0)
{
v___x_420_ = v_inst_416_;
v_isShared_421_ = v_isSharedCheck_428_;
goto v_resetjp_419_;
}
else
{
lean_inc(v_inf_418_);
lean_inc(v_toSemilatticeSup_417_);
lean_dec(v_inst_416_);
v___x_420_ = lean_box(0);
v_isShared_421_ = v_isSharedCheck_428_;
goto v_resetjp_419_;
}
v_resetjp_419_:
{
lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___f_424_; lean_object* v___x_426_; 
v___x_422_ = lp_mathlib_WithTop_semilatticeSup___redArg(v_toSemilatticeSup_417_);
v___x_423_ = lean_box(0);
v___f_424_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_semilatticeInf___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_424_, 0, v___x_423_);
lean_closure_set(v___f_424_, 1, v_inf_418_);
if (v_isShared_421_ == 0)
{
lean_ctor_set(v___x_420_, 1, v___f_424_);
lean_ctor_set(v___x_420_, 0, v___x_422_);
v___x_426_ = v___x_420_;
goto v_reusejp_425_;
}
else
{
lean_object* v_reuseFailAlloc_427_; 
v_reuseFailAlloc_427_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_427_, 0, v___x_422_);
lean_ctor_set(v_reuseFailAlloc_427_, 1, v___f_424_);
v___x_426_ = v_reuseFailAlloc_427_;
goto v_reusejp_425_;
}
v_reusejp_425_:
{
return v___x_426_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_lattice(lean_object* v_00_u03b1_429_, lean_object* v_inst_430_){
_start:
{
lean_object* v___x_431_; 
v___x_431_ = lp_mathlib_WithTop_lattice___redArg(v_inst_430_);
return v___x_431_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_distribLattice___redArg(lean_object* v_inst_432_){
_start:
{
lean_object* v___x_433_; 
v___x_433_ = lp_mathlib_WithBot_lattice___redArg(v_inst_432_);
return v___x_433_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_distribLattice(lean_object* v_00_u03b1_434_, lean_object* v_inst_435_){
_start:
{
lean_object* v___x_436_; 
v___x_436_ = lp_mathlib_WithBot_lattice___redArg(v_inst_435_);
return v___x_436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_distribLattice___redArg(lean_object* v_inst_437_){
_start:
{
lean_object* v___x_438_; 
v___x_438_ = lp_mathlib_WithTop_lattice___redArg(v_inst_437_);
return v___x_438_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_distribLattice(lean_object* v_00_u03b1_439_, lean_object* v_inst_440_){
_start:
{
lean_object* v___x_441_; 
v___x_441_ = lp_mathlib_WithTop_lattice___redArg(v_inst_440_);
return v___x_441_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithBot_decidableEq___aux__1___redArg(lean_object* v_inst_442_, lean_object* v_a_443_, lean_object* v_b_444_){
_start:
{
uint8_t v___x_445_; 
v___x_445_ = l_Option_instDecidableEq___redArg(v_inst_442_, v_a_443_, v_b_444_);
return v___x_445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_decidableEq___aux__1___redArg___boxed(lean_object* v_inst_446_, lean_object* v_a_447_, lean_object* v_b_448_){
_start:
{
uint8_t v_res_449_; lean_object* v_r_450_; 
v_res_449_ = lp_mathlib_WithBot_decidableEq___aux__1___redArg(v_inst_446_, v_a_447_, v_b_448_);
v_r_450_ = lean_box(v_res_449_);
return v_r_450_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithBot_decidableEq___aux__1(lean_object* v_00_u03b1_451_, lean_object* v_inst_452_, lean_object* v_a_453_, lean_object* v_b_454_){
_start:
{
uint8_t v___x_455_; 
v___x_455_ = l_Option_instDecidableEq___redArg(v_inst_452_, v_a_453_, v_b_454_);
return v___x_455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_decidableEq___aux__1___boxed(lean_object* v_00_u03b1_456_, lean_object* v_inst_457_, lean_object* v_a_458_, lean_object* v_b_459_){
_start:
{
uint8_t v_res_460_; lean_object* v_r_461_; 
v_res_460_ = lp_mathlib_WithBot_decidableEq___aux__1(v_00_u03b1_456_, v_inst_457_, v_a_458_, v_b_459_);
v_r_461_ = lean_box(v_res_460_);
return v_r_461_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithBot_decidableEq___redArg(lean_object* v_inst_462_, lean_object* v_a_463_, lean_object* v_b_464_){
_start:
{
uint8_t v___x_465_; 
v___x_465_ = l_Option_instDecidableEq___redArg(v_inst_462_, v_a_463_, v_b_464_);
return v___x_465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_decidableEq___redArg___boxed(lean_object* v_inst_466_, lean_object* v_a_467_, lean_object* v_b_468_){
_start:
{
uint8_t v_res_469_; lean_object* v_r_470_; 
v_res_469_ = lp_mathlib_WithBot_decidableEq___redArg(v_inst_466_, v_a_467_, v_b_468_);
v_r_470_ = lean_box(v_res_469_);
return v_r_470_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithBot_decidableEq(lean_object* v_00_u03b1_471_, lean_object* v_inst_472_, lean_object* v_a_473_, lean_object* v_b_474_){
_start:
{
uint8_t v___x_475_; 
v___x_475_ = l_Option_instDecidableEq___redArg(v_inst_472_, v_a_473_, v_b_474_);
return v___x_475_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_decidableEq___boxed(lean_object* v_00_u03b1_476_, lean_object* v_inst_477_, lean_object* v_a_478_, lean_object* v_b_479_){
_start:
{
uint8_t v_res_480_; lean_object* v_r_481_; 
v_res_480_ = lp_mathlib_WithBot_decidableEq(v_00_u03b1_476_, v_inst_477_, v_a_478_, v_b_479_);
v_r_481_ = lean_box(v_res_480_);
return v_r_481_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithTop_decidableEq___aux__1___redArg(lean_object* v_inst_482_, lean_object* v_a_483_, lean_object* v_b_484_){
_start:
{
uint8_t v___x_485_; 
v___x_485_ = l_Option_instDecidableEq___redArg(v_inst_482_, v_a_483_, v_b_484_);
return v___x_485_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_decidableEq___aux__1___redArg___boxed(lean_object* v_inst_486_, lean_object* v_a_487_, lean_object* v_b_488_){
_start:
{
uint8_t v_res_489_; lean_object* v_r_490_; 
v_res_489_ = lp_mathlib_WithTop_decidableEq___aux__1___redArg(v_inst_486_, v_a_487_, v_b_488_);
v_r_490_ = lean_box(v_res_489_);
return v_r_490_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithTop_decidableEq___aux__1(lean_object* v_00_u03b1_491_, lean_object* v_inst_492_, lean_object* v_a_493_, lean_object* v_b_494_){
_start:
{
uint8_t v___x_495_; 
v___x_495_ = l_Option_instDecidableEq___redArg(v_inst_492_, v_a_493_, v_b_494_);
return v___x_495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_decidableEq___aux__1___boxed(lean_object* v_00_u03b1_496_, lean_object* v_inst_497_, lean_object* v_a_498_, lean_object* v_b_499_){
_start:
{
uint8_t v_res_500_; lean_object* v_r_501_; 
v_res_500_ = lp_mathlib_WithTop_decidableEq___aux__1(v_00_u03b1_496_, v_inst_497_, v_a_498_, v_b_499_);
v_r_501_ = lean_box(v_res_500_);
return v_r_501_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithTop_decidableEq___redArg(lean_object* v_inst_502_, lean_object* v_a_503_, lean_object* v_b_504_){
_start:
{
uint8_t v___x_505_; 
v___x_505_ = l_Option_instDecidableEq___redArg(v_inst_502_, v_a_503_, v_b_504_);
return v___x_505_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_decidableEq___redArg___boxed(lean_object* v_inst_506_, lean_object* v_a_507_, lean_object* v_b_508_){
_start:
{
uint8_t v_res_509_; lean_object* v_r_510_; 
v_res_509_ = lp_mathlib_WithTop_decidableEq___redArg(v_inst_506_, v_a_507_, v_b_508_);
v_r_510_ = lean_box(v_res_509_);
return v_r_510_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithTop_decidableEq(lean_object* v_00_u03b1_511_, lean_object* v_inst_512_, lean_object* v_a_513_, lean_object* v_b_514_){
_start:
{
uint8_t v___x_515_; 
v___x_515_ = l_Option_instDecidableEq___redArg(v_inst_512_, v_a_513_, v_b_514_);
return v___x_515_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_decidableEq___boxed(lean_object* v_00_u03b1_516_, lean_object* v_inst_517_, lean_object* v_a_518_, lean_object* v_b_519_){
_start:
{
uint8_t v_res_520_; lean_object* v_r_521_; 
v_res_520_ = lp_mathlib_WithTop_decidableEq(v_00_u03b1_516_, v_inst_517_, v_a_518_, v_b_519_);
v_r_521_ = lean_box(v_res_520_);
return v_r_521_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithBot_decidableLE___redArg(lean_object* v_inst_522_, lean_object* v_x_523_, lean_object* v_x_524_){
_start:
{
uint8_t v___x_525_; 
v___x_525_ = 1;
if (lean_obj_tag(v_x_523_) == 0)
{
lean_dec(v_x_524_);
lean_dec_ref(v_inst_522_);
return v___x_525_;
}
else
{
lean_object* v_val_526_; uint8_t v___x_527_; 
v_val_526_ = lean_ctor_get(v_x_523_, 0);
lean_inc(v_val_526_);
lean_dec_ref_known(v_x_523_, 1);
v___x_527_ = 0;
if (lean_obj_tag(v_x_524_) == 0)
{
lean_dec(v_val_526_);
lean_dec_ref(v_inst_522_);
return v___x_527_;
}
else
{
lean_object* v_val_528_; lean_object* v___x_529_; uint8_t v___x_530_; 
v_val_528_ = lean_ctor_get(v_x_524_, 0);
lean_inc(v_val_528_);
lean_dec_ref_known(v_x_524_, 1);
v___x_529_ = lean_apply_2(v_inst_522_, v_val_526_, v_val_528_);
v___x_530_ = lean_unbox(v___x_529_);
if (v___x_530_ == 0)
{
return v___x_527_;
}
else
{
return v___x_525_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_decidableLE___redArg___boxed(lean_object* v_inst_531_, lean_object* v_x_532_, lean_object* v_x_533_){
_start:
{
uint8_t v_res_534_; lean_object* v_r_535_; 
v_res_534_ = lp_mathlib_WithBot_decidableLE___redArg(v_inst_531_, v_x_532_, v_x_533_);
v_r_535_ = lean_box(v_res_534_);
return v_r_535_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithBot_decidableLE(lean_object* v_00_u03b1_536_, lean_object* v_inst_537_, lean_object* v_inst_538_, lean_object* v_x_539_, lean_object* v_x_540_){
_start:
{
uint8_t v___x_541_; 
v___x_541_ = lp_mathlib_WithBot_decidableLE___redArg(v_inst_538_, v_x_539_, v_x_540_);
return v___x_541_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_decidableLE___boxed(lean_object* v_00_u03b1_542_, lean_object* v_inst_543_, lean_object* v_inst_544_, lean_object* v_x_545_, lean_object* v_x_546_){
_start:
{
uint8_t v_res_547_; lean_object* v_r_548_; 
v_res_547_ = lp_mathlib_WithBot_decidableLE(v_00_u03b1_542_, v_inst_543_, v_inst_544_, v_x_545_, v_x_546_);
v_r_548_ = lean_box(v_res_547_);
return v_r_548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_decidableLE_match__1___redArg(lean_object* v_x_549_, lean_object* v_x_550_, lean_object* v_h__1_551_, lean_object* v_h__2_552_, lean_object* v_h__3_553_){
_start:
{
if (lean_obj_tag(v_x_549_) == 0)
{
lean_object* v___x_554_; 
lean_dec(v_h__3_553_);
lean_dec(v_h__2_552_);
v___x_554_ = lean_apply_1(v_h__1_551_, v_x_550_);
return v___x_554_;
}
else
{
lean_dec(v_h__1_551_);
if (lean_obj_tag(v_x_550_) == 0)
{
lean_object* v_val_555_; lean_object* v___x_556_; 
lean_dec(v_h__3_553_);
v_val_555_ = lean_ctor_get(v_x_549_, 0);
lean_inc(v_val_555_);
lean_dec_ref_known(v_x_549_, 1);
v___x_556_ = lean_apply_1(v_h__2_552_, v_val_555_);
return v___x_556_;
}
else
{
lean_object* v_val_557_; lean_object* v_val_558_; lean_object* v___x_559_; 
lean_dec(v_h__2_552_);
v_val_557_ = lean_ctor_get(v_x_549_, 0);
lean_inc(v_val_557_);
lean_dec_ref_known(v_x_549_, 1);
v_val_558_ = lean_ctor_get(v_x_550_, 0);
lean_inc(v_val_558_);
lean_dec_ref_known(v_x_550_, 1);
v___x_559_ = lean_apply_2(v_h__3_553_, v_val_557_, v_val_558_);
return v___x_559_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_decidableLE_match__1(lean_object* v_00_u03b1_560_, lean_object* v_motive_561_, lean_object* v_x_562_, lean_object* v_x_563_, lean_object* v_h__1_564_, lean_object* v_h__2_565_, lean_object* v_h__3_566_){
_start:
{
lean_object* v___x_567_; 
v___x_567_ = lp_mathlib_WithTop_decidableLE_match__1___redArg(v_x_562_, v_x_563_, v_h__1_564_, v_h__2_565_, v_h__3_566_);
return v___x_567_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithTop_decidableLE___redArg(lean_object* v_inst_568_, lean_object* v_a_569_, lean_object* v_b_570_){
_start:
{
uint8_t v___x_571_; 
v___x_571_ = 1;
if (lean_obj_tag(v_b_570_) == 0)
{
lean_dec(v_a_569_);
lean_dec_ref(v_inst_568_);
return v___x_571_;
}
else
{
lean_object* v_val_572_; uint8_t v___x_573_; 
v_val_572_ = lean_ctor_get(v_b_570_, 0);
lean_inc(v_val_572_);
lean_dec_ref_known(v_b_570_, 1);
v___x_573_ = 0;
if (lean_obj_tag(v_a_569_) == 0)
{
lean_dec(v_val_572_);
lean_dec_ref(v_inst_568_);
return v___x_573_;
}
else
{
lean_object* v_val_574_; lean_object* v___x_575_; uint8_t v___x_576_; 
v_val_574_ = lean_ctor_get(v_a_569_, 0);
lean_inc(v_val_574_);
lean_dec_ref_known(v_a_569_, 1);
v___x_575_ = lean_apply_2(v_inst_568_, v_val_574_, v_val_572_);
v___x_576_ = lean_unbox(v___x_575_);
if (v___x_576_ == 0)
{
return v___x_573_;
}
else
{
return v___x_571_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_decidableLE___redArg___boxed(lean_object* v_inst_577_, lean_object* v_a_578_, lean_object* v_b_579_){
_start:
{
uint8_t v_res_580_; lean_object* v_r_581_; 
v_res_580_ = lp_mathlib_WithTop_decidableLE___redArg(v_inst_577_, v_a_578_, v_b_579_);
v_r_581_ = lean_box(v_res_580_);
return v_r_581_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithTop_decidableLE(lean_object* v_00_u03b1_582_, lean_object* v_inst_583_, lean_object* v_inst_584_, lean_object* v_a_585_, lean_object* v_b_586_){
_start:
{
uint8_t v___x_587_; 
v___x_587_ = lp_mathlib_WithTop_decidableLE___redArg(v_inst_584_, v_a_585_, v_b_586_);
return v___x_587_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_decidableLE___boxed(lean_object* v_00_u03b1_588_, lean_object* v_inst_589_, lean_object* v_inst_590_, lean_object* v_a_591_, lean_object* v_b_592_){
_start:
{
uint8_t v_res_593_; lean_object* v_r_594_; 
v_res_593_ = lp_mathlib_WithTop_decidableLE(v_00_u03b1_588_, v_inst_589_, v_inst_590_, v_a_591_, v_b_592_);
v_r_594_ = lean_box(v_res_593_);
return v_r_594_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_WithBot_0__WithBot_decidableLE_match__1_splitter___redArg(lean_object* v_x_595_, lean_object* v_x_596_, lean_object* v_h__1_597_, lean_object* v_h__2_598_, lean_object* v_h__3_599_){
_start:
{
if (lean_obj_tag(v_x_595_) == 0)
{
lean_object* v___x_600_; 
lean_dec(v_h__3_599_);
lean_dec(v_h__2_598_);
v___x_600_ = lean_apply_1(v_h__1_597_, v_x_596_);
return v___x_600_;
}
else
{
lean_dec(v_h__1_597_);
if (lean_obj_tag(v_x_596_) == 0)
{
lean_object* v_val_601_; lean_object* v___x_602_; 
lean_dec(v_h__3_599_);
v_val_601_ = lean_ctor_get(v_x_595_, 0);
lean_inc(v_val_601_);
lean_dec_ref_known(v_x_595_, 1);
v___x_602_ = lean_apply_1(v_h__2_598_, v_val_601_);
return v___x_602_;
}
else
{
lean_object* v_val_603_; lean_object* v_val_604_; lean_object* v___x_605_; 
lean_dec(v_h__2_598_);
v_val_603_ = lean_ctor_get(v_x_595_, 0);
lean_inc(v_val_603_);
lean_dec_ref_known(v_x_595_, 1);
v_val_604_ = lean_ctor_get(v_x_596_, 0);
lean_inc(v_val_604_);
lean_dec_ref_known(v_x_596_, 1);
v___x_605_ = lean_apply_2(v_h__3_599_, v_val_603_, v_val_604_);
return v___x_605_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_WithBot_0__WithBot_decidableLE_match__1_splitter(lean_object* v_00_u03b1_606_, lean_object* v_motive_607_, lean_object* v_x_608_, lean_object* v_x_609_, lean_object* v_h__1_610_, lean_object* v_h__2_611_, lean_object* v_h__3_612_){
_start:
{
if (lean_obj_tag(v_x_608_) == 0)
{
lean_object* v___x_613_; 
lean_dec(v_h__3_612_);
lean_dec(v_h__2_611_);
v___x_613_ = lean_apply_1(v_h__1_610_, v_x_609_);
return v___x_613_;
}
else
{
lean_dec(v_h__1_610_);
if (lean_obj_tag(v_x_609_) == 0)
{
lean_object* v_val_614_; lean_object* v___x_615_; 
lean_dec(v_h__3_612_);
v_val_614_ = lean_ctor_get(v_x_608_, 0);
lean_inc(v_val_614_);
lean_dec_ref_known(v_x_608_, 1);
v___x_615_ = lean_apply_1(v_h__2_611_, v_val_614_);
return v___x_615_;
}
else
{
lean_object* v_val_616_; lean_object* v_val_617_; lean_object* v___x_618_; 
lean_dec(v_h__2_611_);
v_val_616_ = lean_ctor_get(v_x_608_, 0);
lean_inc(v_val_616_);
lean_dec_ref_known(v_x_608_, 1);
v_val_617_ = lean_ctor_get(v_x_609_, 0);
lean_inc(v_val_617_);
lean_dec_ref_known(v_x_609_, 1);
v___x_618_ = lean_apply_2(v_h__3_612_, v_val_616_, v_val_617_);
return v___x_618_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_WithBot_0__WithTop_decidableLE_match__1_splitter___redArg(lean_object* v_x_619_, lean_object* v_x_620_, lean_object* v_h__1_621_, lean_object* v_h__2_622_, lean_object* v_h__3_623_){
_start:
{
if (lean_obj_tag(v_x_619_) == 0)
{
lean_object* v___x_624_; 
lean_dec(v_h__3_623_);
lean_dec(v_h__2_622_);
v___x_624_ = lean_apply_1(v_h__1_621_, v_x_620_);
return v___x_624_;
}
else
{
lean_dec(v_h__1_621_);
if (lean_obj_tag(v_x_620_) == 0)
{
lean_object* v_val_625_; lean_object* v___x_626_; 
lean_dec(v_h__3_623_);
v_val_625_ = lean_ctor_get(v_x_619_, 0);
lean_inc(v_val_625_);
lean_dec_ref_known(v_x_619_, 1);
v___x_626_ = lean_apply_1(v_h__2_622_, v_val_625_);
return v___x_626_;
}
else
{
lean_object* v_val_627_; lean_object* v_val_628_; lean_object* v___x_629_; 
lean_dec(v_h__2_622_);
v_val_627_ = lean_ctor_get(v_x_619_, 0);
lean_inc(v_val_627_);
lean_dec_ref_known(v_x_619_, 1);
v_val_628_ = lean_ctor_get(v_x_620_, 0);
lean_inc(v_val_628_);
lean_dec_ref_known(v_x_620_, 1);
v___x_629_ = lean_apply_2(v_h__3_623_, v_val_627_, v_val_628_);
return v___x_629_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_WithBot_0__WithTop_decidableLE_match__1_splitter(lean_object* v_00_u03b1_630_, lean_object* v_motive_631_, lean_object* v_x_632_, lean_object* v_x_633_, lean_object* v_h__1_634_, lean_object* v_h__2_635_, lean_object* v_h__3_636_){
_start:
{
if (lean_obj_tag(v_x_632_) == 0)
{
lean_object* v___x_637_; 
lean_dec(v_h__3_636_);
lean_dec(v_h__2_635_);
v___x_637_ = lean_apply_1(v_h__1_634_, v_x_633_);
return v___x_637_;
}
else
{
lean_dec(v_h__1_634_);
if (lean_obj_tag(v_x_633_) == 0)
{
lean_object* v_val_638_; lean_object* v___x_639_; 
lean_dec(v_h__3_636_);
v_val_638_ = lean_ctor_get(v_x_632_, 0);
lean_inc(v_val_638_);
lean_dec_ref_known(v_x_632_, 1);
v___x_639_ = lean_apply_1(v_h__2_635_, v_val_638_);
return v___x_639_;
}
else
{
lean_object* v_val_640_; lean_object* v_val_641_; lean_object* v___x_642_; 
lean_dec(v_h__2_635_);
v_val_640_ = lean_ctor_get(v_x_632_, 0);
lean_inc(v_val_640_);
lean_dec_ref_known(v_x_632_, 1);
v_val_641_ = lean_ctor_get(v_x_633_, 0);
lean_inc(v_val_641_);
lean_dec_ref_known(v_x_633_, 1);
v___x_642_ = lean_apply_2(v_h__3_636_, v_val_640_, v_val_641_);
return v___x_642_;
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithBot_decidableLT___redArg(lean_object* v_inst_643_, lean_object* v_x_644_, lean_object* v_x_645_){
_start:
{
uint8_t v___x_646_; 
v___x_646_ = 0;
if (lean_obj_tag(v_x_645_) == 0)
{
lean_dec(v_x_644_);
lean_dec_ref(v_inst_643_);
return v___x_646_;
}
else
{
lean_object* v_val_647_; uint8_t v___x_648_; 
v_val_647_ = lean_ctor_get(v_x_645_, 0);
lean_inc(v_val_647_);
lean_dec_ref_known(v_x_645_, 1);
v___x_648_ = 1;
if (lean_obj_tag(v_x_644_) == 0)
{
lean_dec(v_val_647_);
lean_dec_ref(v_inst_643_);
return v___x_648_;
}
else
{
lean_object* v_val_649_; lean_object* v___x_650_; uint8_t v___x_651_; 
v_val_649_ = lean_ctor_get(v_x_644_, 0);
lean_inc(v_val_649_);
lean_dec_ref_known(v_x_644_, 1);
v___x_650_ = lean_apply_2(v_inst_643_, v_val_649_, v_val_647_);
v___x_651_ = lean_unbox(v___x_650_);
if (v___x_651_ == 0)
{
return v___x_646_;
}
else
{
return v___x_648_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_decidableLT___redArg___boxed(lean_object* v_inst_652_, lean_object* v_x_653_, lean_object* v_x_654_){
_start:
{
uint8_t v_res_655_; lean_object* v_r_656_; 
v_res_655_ = lp_mathlib_WithBot_decidableLT___redArg(v_inst_652_, v_x_653_, v_x_654_);
v_r_656_ = lean_box(v_res_655_);
return v_r_656_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithBot_decidableLT(lean_object* v_00_u03b1_657_, lean_object* v_inst_658_, lean_object* v_inst_659_, lean_object* v_x_660_, lean_object* v_x_661_){
_start:
{
uint8_t v___x_662_; 
v___x_662_ = lp_mathlib_WithBot_decidableLT___redArg(v_inst_659_, v_x_660_, v_x_661_);
return v___x_662_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_decidableLT___boxed(lean_object* v_00_u03b1_663_, lean_object* v_inst_664_, lean_object* v_inst_665_, lean_object* v_x_666_, lean_object* v_x_667_){
_start:
{
uint8_t v_res_668_; lean_object* v_r_669_; 
v_res_668_ = lp_mathlib_WithBot_decidableLT(v_00_u03b1_663_, v_inst_664_, v_inst_665_, v_x_666_, v_x_667_);
v_r_669_ = lean_box(v_res_668_);
return v_r_669_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_decidableLT_match__1___redArg(lean_object* v_x_670_, lean_object* v_x_671_, lean_object* v_h__1_672_, lean_object* v_h__2_673_, lean_object* v_h__3_674_){
_start:
{
if (lean_obj_tag(v_x_671_) == 0)
{
lean_object* v___x_675_; 
lean_dec(v_h__3_674_);
lean_dec(v_h__2_673_);
v___x_675_ = lean_apply_1(v_h__1_672_, v_x_670_);
return v___x_675_;
}
else
{
lean_dec(v_h__1_672_);
if (lean_obj_tag(v_x_670_) == 0)
{
lean_object* v_val_676_; lean_object* v___x_677_; 
lean_dec(v_h__3_674_);
v_val_676_ = lean_ctor_get(v_x_671_, 0);
lean_inc(v_val_676_);
lean_dec_ref_known(v_x_671_, 1);
v___x_677_ = lean_apply_1(v_h__2_673_, v_val_676_);
return v___x_677_;
}
else
{
lean_object* v_val_678_; lean_object* v_val_679_; lean_object* v___x_680_; 
lean_dec(v_h__2_673_);
v_val_678_ = lean_ctor_get(v_x_671_, 0);
lean_inc(v_val_678_);
lean_dec_ref_known(v_x_671_, 1);
v_val_679_ = lean_ctor_get(v_x_670_, 0);
lean_inc(v_val_679_);
lean_dec_ref_known(v_x_670_, 1);
v___x_680_ = lean_apply_2(v_h__3_674_, v_val_679_, v_val_678_);
return v___x_680_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_decidableLT_match__1(lean_object* v_00_u03b1_681_, lean_object* v_motive_682_, lean_object* v_x_683_, lean_object* v_x_684_, lean_object* v_h__1_685_, lean_object* v_h__2_686_, lean_object* v_h__3_687_){
_start:
{
lean_object* v___x_688_; 
v___x_688_ = lp_mathlib_WithTop_decidableLT_match__1___redArg(v_x_683_, v_x_684_, v_h__1_685_, v_h__2_686_, v_h__3_687_);
return v___x_688_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithTop_decidableLT___redArg(lean_object* v_inst_689_, lean_object* v_a_690_, lean_object* v_b_691_){
_start:
{
uint8_t v___x_692_; 
v___x_692_ = 0;
if (lean_obj_tag(v_a_690_) == 0)
{
lean_dec(v_b_691_);
lean_dec_ref(v_inst_689_);
return v___x_692_;
}
else
{
lean_object* v_val_693_; uint8_t v___x_694_; 
v_val_693_ = lean_ctor_get(v_a_690_, 0);
lean_inc(v_val_693_);
lean_dec_ref_known(v_a_690_, 1);
v___x_694_ = 1;
if (lean_obj_tag(v_b_691_) == 0)
{
lean_dec(v_val_693_);
lean_dec_ref(v_inst_689_);
return v___x_694_;
}
else
{
lean_object* v_val_695_; lean_object* v___x_696_; uint8_t v___x_697_; 
v_val_695_ = lean_ctor_get(v_b_691_, 0);
lean_inc(v_val_695_);
lean_dec_ref_known(v_b_691_, 1);
v___x_696_ = lean_apply_2(v_inst_689_, v_val_693_, v_val_695_);
v___x_697_ = lean_unbox(v___x_696_);
if (v___x_697_ == 0)
{
return v___x_692_;
}
else
{
return v___x_694_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_decidableLT___redArg___boxed(lean_object* v_inst_698_, lean_object* v_a_699_, lean_object* v_b_700_){
_start:
{
uint8_t v_res_701_; lean_object* v_r_702_; 
v_res_701_ = lp_mathlib_WithTop_decidableLT___redArg(v_inst_698_, v_a_699_, v_b_700_);
v_r_702_ = lean_box(v_res_701_);
return v_r_702_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithTop_decidableLT(lean_object* v_00_u03b1_703_, lean_object* v_inst_704_, lean_object* v_inst_705_, lean_object* v_a_706_, lean_object* v_b_707_){
_start:
{
uint8_t v___x_708_; 
v___x_708_ = lp_mathlib_WithTop_decidableLT___redArg(v_inst_705_, v_a_706_, v_b_707_);
return v___x_708_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_decidableLT___boxed(lean_object* v_00_u03b1_709_, lean_object* v_inst_710_, lean_object* v_inst_711_, lean_object* v_a_712_, lean_object* v_b_713_){
_start:
{
uint8_t v_res_714_; lean_object* v_r_715_; 
v_res_714_ = lp_mathlib_WithTop_decidableLT(v_00_u03b1_709_, v_inst_710_, v_inst_711_, v_a_712_, v_b_713_);
v_r_715_ = lean_box(v_res_714_);
return v_r_715_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_WithBot_0__WithBot_decidableLT_match__1_splitter___redArg(lean_object* v_x_716_, lean_object* v_x_717_, lean_object* v_h__1_718_, lean_object* v_h__2_719_, lean_object* v_h__3_720_){
_start:
{
if (lean_obj_tag(v_x_717_) == 0)
{
lean_object* v___x_721_; 
lean_dec(v_h__3_720_);
lean_dec(v_h__2_719_);
v___x_721_ = lean_apply_1(v_h__1_718_, v_x_716_);
return v___x_721_;
}
else
{
lean_dec(v_h__1_718_);
if (lean_obj_tag(v_x_716_) == 0)
{
lean_object* v_val_722_; lean_object* v___x_723_; 
lean_dec(v_h__3_720_);
v_val_722_ = lean_ctor_get(v_x_717_, 0);
lean_inc(v_val_722_);
lean_dec_ref_known(v_x_717_, 1);
v___x_723_ = lean_apply_1(v_h__2_719_, v_val_722_);
return v___x_723_;
}
else
{
lean_object* v_val_724_; lean_object* v_val_725_; lean_object* v___x_726_; 
lean_dec(v_h__2_719_);
v_val_724_ = lean_ctor_get(v_x_717_, 0);
lean_inc(v_val_724_);
lean_dec_ref_known(v_x_717_, 1);
v_val_725_ = lean_ctor_get(v_x_716_, 0);
lean_inc(v_val_725_);
lean_dec_ref_known(v_x_716_, 1);
v___x_726_ = lean_apply_2(v_h__3_720_, v_val_725_, v_val_724_);
return v___x_726_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_WithBot_0__WithBot_decidableLT_match__1_splitter(lean_object* v_00_u03b1_727_, lean_object* v_motive_728_, lean_object* v_x_729_, lean_object* v_x_730_, lean_object* v_h__1_731_, lean_object* v_h__2_732_, lean_object* v_h__3_733_){
_start:
{
if (lean_obj_tag(v_x_730_) == 0)
{
lean_object* v___x_734_; 
lean_dec(v_h__3_733_);
lean_dec(v_h__2_732_);
v___x_734_ = lean_apply_1(v_h__1_731_, v_x_729_);
return v___x_734_;
}
else
{
lean_dec(v_h__1_731_);
if (lean_obj_tag(v_x_729_) == 0)
{
lean_object* v_val_735_; lean_object* v___x_736_; 
lean_dec(v_h__3_733_);
v_val_735_ = lean_ctor_get(v_x_730_, 0);
lean_inc(v_val_735_);
lean_dec_ref_known(v_x_730_, 1);
v___x_736_ = lean_apply_1(v_h__2_732_, v_val_735_);
return v___x_736_;
}
else
{
lean_object* v_val_737_; lean_object* v_val_738_; lean_object* v___x_739_; 
lean_dec(v_h__2_732_);
v_val_737_ = lean_ctor_get(v_x_730_, 0);
lean_inc(v_val_737_);
lean_dec_ref_known(v_x_730_, 1);
v_val_738_ = lean_ctor_get(v_x_729_, 0);
lean_inc(v_val_738_);
lean_dec_ref_known(v_x_729_, 1);
v___x_739_ = lean_apply_2(v_h__3_733_, v_val_738_, v_val_737_);
return v___x_739_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_WithBot_0__WithTop_decidableLT_match__1_splitter___redArg(lean_object* v_x_740_, lean_object* v_x_741_, lean_object* v_h__1_742_, lean_object* v_h__2_743_, lean_object* v_h__3_744_){
_start:
{
if (lean_obj_tag(v_x_741_) == 0)
{
lean_object* v___x_745_; 
lean_dec(v_h__3_744_);
lean_dec(v_h__2_743_);
v___x_745_ = lean_apply_1(v_h__1_742_, v_x_740_);
return v___x_745_;
}
else
{
lean_dec(v_h__1_742_);
if (lean_obj_tag(v_x_740_) == 0)
{
lean_object* v_val_746_; lean_object* v___x_747_; 
lean_dec(v_h__3_744_);
v_val_746_ = lean_ctor_get(v_x_741_, 0);
lean_inc(v_val_746_);
lean_dec_ref_known(v_x_741_, 1);
v___x_747_ = lean_apply_1(v_h__2_743_, v_val_746_);
return v___x_747_;
}
else
{
lean_object* v_val_748_; lean_object* v_val_749_; lean_object* v___x_750_; 
lean_dec(v_h__2_743_);
v_val_748_ = lean_ctor_get(v_x_741_, 0);
lean_inc(v_val_748_);
lean_dec_ref_known(v_x_741_, 1);
v_val_749_ = lean_ctor_get(v_x_740_, 0);
lean_inc(v_val_749_);
lean_dec_ref_known(v_x_740_, 1);
v___x_750_ = lean_apply_2(v_h__3_744_, v_val_749_, v_val_748_);
return v___x_750_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_WithBot_0__WithTop_decidableLT_match__1_splitter(lean_object* v_00_u03b1_751_, lean_object* v_motive_752_, lean_object* v_x_753_, lean_object* v_x_754_, lean_object* v_h__1_755_, lean_object* v_h__2_756_, lean_object* v_h__3_757_){
_start:
{
if (lean_obj_tag(v_x_754_) == 0)
{
lean_object* v___x_758_; 
lean_dec(v_h__3_757_);
lean_dec(v_h__2_756_);
v___x_758_ = lean_apply_1(v_h__1_755_, v_x_753_);
return v___x_758_;
}
else
{
lean_dec(v_h__1_755_);
if (lean_obj_tag(v_x_753_) == 0)
{
lean_object* v_val_759_; lean_object* v___x_760_; 
lean_dec(v_h__3_757_);
v_val_759_ = lean_ctor_get(v_x_754_, 0);
lean_inc(v_val_759_);
lean_dec_ref_known(v_x_754_, 1);
v___x_760_ = lean_apply_1(v_h__2_756_, v_val_759_);
return v___x_760_;
}
else
{
lean_object* v_val_761_; lean_object* v_val_762_; lean_object* v___x_763_; 
lean_dec(v_h__2_756_);
v_val_761_ = lean_ctor_get(v_x_754_, 0);
lean_inc(v_val_761_);
lean_dec_ref_known(v_x_754_, 1);
v_val_762_ = lean_ctor_get(v_x_753_, 0);
lean_inc(v_val_762_);
lean_dec_ref_known(v_x_753_, 1);
v___x_763_ = lean_apply_2(v_h__3_757_, v_val_762_, v_val_761_);
return v___x_763_;
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithBot_linearOrder___redArg___lam__0(lean_object* v_inst_764_, lean_object* v_a_765_, lean_object* v_b_766_){
_start:
{
lean_object* v_toDecidableEq_767_; lean_object* v___x_768_; uint8_t v___x_769_; 
v_toDecidableEq_767_ = lean_ctor_get(v_inst_764_, 5);
lean_inc_ref(v_toDecidableEq_767_);
lean_dec_ref(v_inst_764_);
v___x_768_ = lean_apply_2(v_toDecidableEq_767_, v_a_765_, v_b_766_);
v___x_769_ = lean_unbox(v___x_768_);
return v___x_769_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_linearOrder___redArg___lam__0___boxed(lean_object* v_inst_770_, lean_object* v_a_771_, lean_object* v_b_772_){
_start:
{
uint8_t v_res_773_; lean_object* v_r_774_; 
v_res_773_ = lp_mathlib_WithBot_linearOrder___redArg___lam__0(v_inst_770_, v_a_771_, v_b_772_);
v_r_774_ = lean_box(v_res_773_);
return v_r_774_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithBot_linearOrder___redArg___lam__1(lean_object* v___f_775_, lean_object* v_a_776_, lean_object* v_b_777_){
_start:
{
uint8_t v___x_778_; 
v___x_778_ = l_Option_instDecidableEq___redArg(v___f_775_, v_a_776_, v_b_777_);
return v___x_778_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_linearOrder___redArg___lam__1___boxed(lean_object* v___f_779_, lean_object* v_a_780_, lean_object* v_b_781_){
_start:
{
uint8_t v_res_782_; lean_object* v_r_783_; 
v_res_782_ = lp_mathlib_WithBot_linearOrder___redArg___lam__1(v___f_779_, v_a_780_, v_b_781_);
v_r_783_ = lean_box(v_res_782_);
return v_r_783_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithBot_linearOrder___redArg___lam__2(lean_object* v_inst_784_, lean_object* v_a_785_, lean_object* v_b_786_){
_start:
{
lean_object* v_toDecidableLE_787_; uint8_t v___x_788_; 
v_toDecidableLE_787_ = lean_ctor_get(v_inst_784_, 4);
lean_inc_ref(v_toDecidableLE_787_);
lean_dec_ref(v_inst_784_);
v___x_788_ = lp_mathlib_WithBot_decidableLE___redArg(v_toDecidableLE_787_, v_a_785_, v_b_786_);
return v___x_788_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_linearOrder___redArg___lam__2___boxed(lean_object* v_inst_789_, lean_object* v_a_790_, lean_object* v_b_791_){
_start:
{
uint8_t v_res_792_; lean_object* v_r_793_; 
v_res_792_ = lp_mathlib_WithBot_linearOrder___redArg___lam__2(v_inst_789_, v_a_790_, v_b_791_);
v_r_793_ = lean_box(v_res_792_);
return v_r_793_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithBot_linearOrder___redArg___lam__3(lean_object* v_inst_794_, lean_object* v_a_795_, lean_object* v_b_796_){
_start:
{
lean_object* v_toDecidableLT_797_; uint8_t v___x_798_; 
v_toDecidableLT_797_ = lean_ctor_get(v_inst_794_, 6);
lean_inc_ref(v_toDecidableLT_797_);
lean_dec_ref(v_inst_794_);
v___x_798_ = lp_mathlib_WithBot_decidableLT___redArg(v_toDecidableLT_797_, v_a_795_, v_b_796_);
return v___x_798_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_linearOrder___redArg___lam__3___boxed(lean_object* v_inst_799_, lean_object* v_a_800_, lean_object* v_b_801_){
_start:
{
uint8_t v_res_802_; lean_object* v_r_803_; 
v_res_802_ = lp_mathlib_WithBot_linearOrder___redArg___lam__3(v_inst_799_, v_a_800_, v_b_801_);
v_r_803_ = lean_box(v_res_802_);
return v_r_803_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithBot_linearOrder___redArg___lam__4(lean_object* v___f_804_, lean_object* v___f_805_, lean_object* v_a_806_, lean_object* v_b_807_){
_start:
{
lean_object* v___x_808_; uint8_t v___x_809_; 
lean_inc(v_b_807_);
lean_inc(v_a_806_);
v___x_808_ = lean_apply_2(v___f_804_, v_a_806_, v_b_807_);
v___x_809_ = lean_unbox(v___x_808_);
if (v___x_809_ == 0)
{
uint8_t v___x_810_; 
v___x_810_ = l_Option_instDecidableEq___redArg(v___f_805_, v_a_806_, v_b_807_);
if (v___x_810_ == 0)
{
uint8_t v___x_811_; 
v___x_811_ = 2;
return v___x_811_;
}
else
{
uint8_t v___x_812_; 
v___x_812_ = 1;
return v___x_812_;
}
}
else
{
uint8_t v___x_813_; 
lean_dec(v_b_807_);
lean_dec(v_a_806_);
lean_dec_ref(v___f_805_);
v___x_813_ = 0;
return v___x_813_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_linearOrder___redArg___lam__4___boxed(lean_object* v___f_814_, lean_object* v___f_815_, lean_object* v_a_816_, lean_object* v_b_817_){
_start:
{
uint8_t v_res_818_; lean_object* v_r_819_; 
v_res_818_ = lp_mathlib_WithBot_linearOrder___redArg___lam__4(v___f_814_, v___f_815_, v_a_816_, v_b_817_);
v_r_819_ = lean_box(v_res_818_);
return v_r_819_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_linearOrder___redArg(lean_object* v_inst_820_){
_start:
{
lean_object* v___x_821_; lean_object* v___x_822_; lean_object* v___x_823_; lean_object* v_toPartialOrder_824_; lean_object* v_toSemilatticeSup_825_; lean_object* v___f_826_; lean_object* v___f_827_; lean_object* v___f_828_; lean_object* v___f_829_; lean_object* v___f_830_; lean_object* v___f_831_; lean_object* v___f_832_; lean_object* v___x_833_; 
v___x_821_ = lp_mathlib_LinearOrder_toLattice___redArg(v_inst_820_);
v___x_822_ = lp_mathlib_WithBot_lattice___redArg(v___x_821_);
lean_inc_ref(v___x_822_);
v___x_823_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_822_);
v_toPartialOrder_824_ = lean_ctor_get(v___x_823_, 0);
lean_inc_ref(v_toPartialOrder_824_);
v_toSemilatticeSup_825_ = lean_ctor_get(v___x_822_, 0);
lean_inc_ref(v_toSemilatticeSup_825_);
lean_dec_ref(v___x_822_);
lean_inc_ref_n(v_inst_820_, 2);
v___f_826_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_linearOrder___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_826_, 0, v_inst_820_);
lean_inc_ref(v___f_826_);
v___f_827_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_linearOrder___redArg___lam__1___boxed), 3, 1);
lean_closure_set(v___f_827_, 0, v___f_826_);
v___f_828_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_linearOrder___redArg___lam__2___boxed), 3, 1);
lean_closure_set(v___f_828_, 0, v_inst_820_);
v___f_829_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_linearOrder___redArg___lam__3___boxed), 3, 1);
lean_closure_set(v___f_829_, 0, v_inst_820_);
lean_inc_ref(v___f_829_);
v___f_830_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_linearOrder___redArg___lam__4___boxed), 4, 2);
lean_closure_set(v___f_830_, 0, v___f_829_);
lean_closure_set(v___f_830_, 1, v___f_826_);
v___f_831_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_831_, 0, v___x_823_);
v___f_832_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_832_, 0, v_toSemilatticeSup_825_);
v___x_833_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_833_, 0, v_toPartialOrder_824_);
lean_ctor_set(v___x_833_, 1, v___f_831_);
lean_ctor_set(v___x_833_, 2, v___f_832_);
lean_ctor_set(v___x_833_, 3, v___f_830_);
lean_ctor_set(v___x_833_, 4, v___f_828_);
lean_ctor_set(v___x_833_, 5, v___f_827_);
lean_ctor_set(v___x_833_, 6, v___f_829_);
return v___x_833_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_linearOrder(lean_object* v_00_u03b1_834_, lean_object* v_inst_835_){
_start:
{
lean_object* v___x_836_; 
v___x_836_ = lp_mathlib_WithBot_linearOrder___redArg(v_inst_835_);
return v___x_836_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithTop_linearOrder___redArg___lam__2(lean_object* v_inst_837_, lean_object* v_a_838_, lean_object* v_b_839_){
_start:
{
lean_object* v_toDecidableLE_840_; uint8_t v___x_841_; 
v_toDecidableLE_840_ = lean_ctor_get(v_inst_837_, 4);
lean_inc_ref(v_toDecidableLE_840_);
lean_dec_ref(v_inst_837_);
v___x_841_ = lp_mathlib_WithTop_decidableLE___redArg(v_toDecidableLE_840_, v_a_838_, v_b_839_);
return v___x_841_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_linearOrder___redArg___lam__2___boxed(lean_object* v_inst_842_, lean_object* v_a_843_, lean_object* v_b_844_){
_start:
{
uint8_t v_res_845_; lean_object* v_r_846_; 
v_res_845_ = lp_mathlib_WithTop_linearOrder___redArg___lam__2(v_inst_842_, v_a_843_, v_b_844_);
v_r_846_ = lean_box(v_res_845_);
return v_r_846_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_WithTop_linearOrder___redArg___lam__0(lean_object* v_inst_847_, lean_object* v_a_848_, lean_object* v_b_849_){
_start:
{
lean_object* v_toDecidableLT_850_; uint8_t v___x_851_; 
v_toDecidableLT_850_ = lean_ctor_get(v_inst_847_, 6);
lean_inc_ref(v_toDecidableLT_850_);
lean_dec_ref(v_inst_847_);
v___x_851_ = lp_mathlib_WithTop_decidableLT___redArg(v_toDecidableLT_850_, v_a_848_, v_b_849_);
return v___x_851_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_linearOrder___redArg___lam__0___boxed(lean_object* v_inst_852_, lean_object* v_a_853_, lean_object* v_b_854_){
_start:
{
uint8_t v_res_855_; lean_object* v_r_856_; 
v_res_855_ = lp_mathlib_WithTop_linearOrder___redArg___lam__0(v_inst_852_, v_a_853_, v_b_854_);
v_r_856_ = lean_box(v_res_855_);
return v_r_856_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_linearOrder___redArg(lean_object* v_inst_857_){
_start:
{
lean_object* v___x_858_; lean_object* v___x_859_; lean_object* v___x_860_; lean_object* v_toPartialOrder_861_; lean_object* v_toSemilatticeSup_862_; lean_object* v___f_863_; lean_object* v___f_864_; lean_object* v___f_865_; lean_object* v___f_866_; lean_object* v___f_867_; lean_object* v___f_868_; lean_object* v___f_869_; lean_object* v___x_870_; 
v___x_858_ = lp_mathlib_LinearOrder_toLattice___redArg(v_inst_857_);
v___x_859_ = lp_mathlib_WithTop_lattice___redArg(v___x_858_);
lean_inc_ref(v___x_859_);
v___x_860_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_859_);
v_toPartialOrder_861_ = lean_ctor_get(v___x_860_, 0);
lean_inc_ref(v_toPartialOrder_861_);
v_toSemilatticeSup_862_ = lean_ctor_get(v___x_859_, 0);
lean_inc_ref(v_toSemilatticeSup_862_);
lean_dec_ref(v___x_859_);
lean_inc_ref_n(v_inst_857_, 2);
v___f_863_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_linearOrder___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_863_, 0, v_inst_857_);
lean_inc_ref(v___f_863_);
v___f_864_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_linearOrder___redArg___lam__1___boxed), 3, 1);
lean_closure_set(v___f_864_, 0, v___f_863_);
v___f_865_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_linearOrder___redArg___lam__2___boxed), 3, 1);
lean_closure_set(v___f_865_, 0, v_inst_857_);
v___f_866_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_linearOrder___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_866_, 0, v_inst_857_);
lean_inc_ref(v___f_866_);
v___f_867_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_linearOrder___redArg___lam__4___boxed), 4, 2);
lean_closure_set(v___f_867_, 0, v___f_866_);
lean_closure_set(v___f_867_, 1, v___f_863_);
v___f_868_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_868_, 0, v___x_860_);
v___f_869_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_869_, 0, v_toSemilatticeSup_862_);
v___x_870_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_870_, 0, v_toPartialOrder_861_);
lean_ctor_set(v___x_870_, 1, v___f_868_);
lean_ctor_set(v___x_870_, 2, v___f_869_);
lean_ctor_set(v___x_870_, 3, v___f_867_);
lean_ctor_set(v___x_870_, 4, v___f_865_);
lean_ctor_set(v___x_870_, 5, v___f_864_);
lean_ctor_set(v___x_870_, 6, v___f_866_);
return v___x_870_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_linearOrder(lean_object* v_00_u03b1_871_, lean_object* v_inst_872_){
_start:
{
lean_object* v___x_873_; 
v___x_873_ = lp_mathlib_WithTop_linearOrder___redArg(v_inst_872_);
return v___x_873_;
}
}
static lean_object* _init_lp_mathlib_WithBot_toDual___closed__0(void){
_start:
{
lean_object* v___x_874_; 
v___x_874_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_874_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_toDual(lean_object* v_00_u03b1_875_){
_start:
{
lean_object* v___x_876_; 
v___x_876_ = lean_obj_once(&lp_mathlib_WithBot_toDual___closed__0, &lp_mathlib_WithBot_toDual___closed__0_once, _init_lp_mathlib_WithBot_toDual___closed__0);
return v___x_876_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_toDual(lean_object* v_00_u03b1_877_){
_start:
{
lean_object* v___x_878_; 
v___x_878_ = lean_obj_once(&lp_mathlib_WithBot_toDual___closed__0, &lp_mathlib_WithBot_toDual___closed__0_once, _init_lp_mathlib_WithBot_toDual___closed__0);
return v___x_878_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_ofDual(lean_object* v_00_u03b1_879_){
_start:
{
lean_object* v___x_880_; 
v___x_880_ = lean_obj_once(&lp_mathlib_WithBot_toDual___closed__0, &lp_mathlib_WithBot_toDual___closed__0_once, _init_lp_mathlib_WithBot_toDual___closed__0);
return v___x_880_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_ofDual(lean_object* v_00_u03b1_881_){
_start:
{
lean_object* v___x_882_; 
v___x_882_ = lean_obj_once(&lp_mathlib_WithBot_toDual___closed__0, &lp_mathlib_WithBot_toDual___closed__0_once, _init_lp_mathlib_WithBot_toDual___closed__0);
return v___x_882_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Nontrivial_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_TypeTags(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Option_NAry(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Contrapose(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Lift(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Option_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Lattice(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_BoundedOrder_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_WithBot(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Nontrivial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_TypeTags(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Option_NAry(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Contrapose(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Lift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Option_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_BoundedOrder_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_WithBot(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Basic_Nontrivial_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_TypeTags(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Option_NAry(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Contrapose(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Lift(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Option_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Lattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_BoundedOrder_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_WithBot(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Nontrivial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_TypeTags(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Option_NAry(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Contrapose(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Lift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Option_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_BoundedOrder_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_WithBot(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_WithBot(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_WithBot(builtin);
}
#ifdef __cplusplus
}
#endif
