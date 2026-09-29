// Lean compiler output
// Module: Mathlib.Order.Hom.Bounded
// Imports: public import Init public meta import Init public import Mathlib.Order.Hom.Basic
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
lean_object* lp_mathlib_Function_Injective_semilatticeInf___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OrderHom_dual(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_OrderHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHomClass_toTopHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHomClass_toTopHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHomClass_toTopHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHomClass_toBotHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHomClass_toBotHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHomClass_toBotHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCTopHomOfTopHomClass___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCTopHomOfTopHomClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCBotHomOfBotHomClass___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCBotHomOfBotHomClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHomClass_toBoundedOrderHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHomClass_toBoundedOrderHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHomClass_toBoundedOrderHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCBoundedOrderHomOfBoundedOrderHomClass___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCBoundedOrderHomOfBoundedOrderHomClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instFunLike___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_TopHom_instFunLike___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_TopHom_instFunLike___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_TopHom_instFunLike___closed__0 = (const lean_object*)&lp_mathlib_TopHom_instFunLike___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instFunLike(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instFunLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instFunLike(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instFunLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instInhabited___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instInhabited___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instInhabited(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instInhabited___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instInhabited(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instInhabited___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_TopHom_id___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_TopHom_id___closed__0 = (const lean_object*)&lp_mathlib_TopHom_id___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_TopHom_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_comp___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_comp___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instLE(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instLE(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_TopHom_instPreorder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_TopHom_instPreorder___closed__0 = (const lean_object*)&lp_mathlib_TopHom_instPreorder___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instPreorder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instPreorder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instPreorder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instPreorder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instPartialOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instPartialOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instPartialOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instPartialOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instOrderTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instOrderTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instOrderTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instOrderBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instOrderBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instOrderBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instMin___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instMin___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instMin(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instMin___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instMax___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instMax___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instMax(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instMax___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instSemilatticeInf___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instSemilatticeInf___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instSemilatticeInf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instSemilatticeInf___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instSemilatticeSup___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instSemilatticeSup___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instSemilatticeSup___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instSemilatticeSup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instSemilatticeSup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instMax___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instMax(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instMax___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instMin___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instMin(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instMin___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instSemilatticeSup___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instSemilatticeSup___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instSemilatticeSup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instSemilatticeSup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instSemilatticeInf___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instSemilatticeInf___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instSemilatticeInf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instSemilatticeInf___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instLattice___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instLattice___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instLattice___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instLattice___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instLattice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instLattice___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instLattice___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instLattice___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instLattice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instLattice___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instDistribLattice___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instDistribLattice___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instDistribLattice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instDistribLattice___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instDistribLattice___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instDistribLattice___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instDistribLattice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instDistribLattice___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_toTopHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_toTopHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_toTopHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_toTopHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_toBotHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_toBotHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_toBotHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_toBotHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_id(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_id___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_instInhabited___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_TopHom_dual___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_TopHom_comp___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_TopHom_dual___closed__0 = (const lean_object*)&lp_mathlib_TopHom_dual___closed__0_value;
static const lean_ctor_object lp_mathlib_TopHom_dual___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_TopHom_dual___closed__0_value),((lean_object*)&lp_mathlib_TopHom_dual___closed__0_value)}};
static const lean_object* lp_mathlib_TopHom_dual___closed__1 = (const lean_object*)&lp_mathlib_TopHom_dual___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_TopHom_dual(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHom_dual___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_dual(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BotHom_dual___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_dual___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_dual___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_dual___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_dual___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_dual___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_dual(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_dual___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_TopHomClass_toTopHom___redArg(lean_object* v_inst_1_, lean_object* v_f_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_apply_1(v_inst_1_, v_f_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHomClass_toTopHom(lean_object* v_F_4_, lean_object* v_00_u03b1_5_, lean_object* v_00_u03b2_6_, lean_object* v_inst_7_, lean_object* v_inst_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_f_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lean_apply_1(v_inst_7_, v_f_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHomClass_toTopHom___boxed(lean_object* v_F_13_, lean_object* v_00_u03b1_14_, lean_object* v_00_u03b2_15_, lean_object* v_inst_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_inst_19_, lean_object* v_f_20_){
_start:
{
lean_object* v_res_21_; 
v_res_21_ = lp_mathlib_TopHomClass_toTopHom(v_F_13_, v_00_u03b1_14_, v_00_u03b2_15_, v_inst_16_, v_inst_17_, v_inst_18_, v_inst_19_, v_f_20_);
lean_dec(v_inst_18_);
lean_dec(v_inst_17_);
return v_res_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHomClass_toBotHom___redArg(lean_object* v_inst_22_, lean_object* v_f_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lean_apply_1(v_inst_22_, v_f_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHomClass_toBotHom(lean_object* v_F_25_, lean_object* v_00_u03b1_26_, lean_object* v_00_u03b2_27_, lean_object* v_inst_28_, lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_inst_31_, lean_object* v_f_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lean_apply_1(v_inst_28_, v_f_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHomClass_toBotHom___boxed(lean_object* v_F_34_, lean_object* v_00_u03b1_35_, lean_object* v_00_u03b2_36_, lean_object* v_inst_37_, lean_object* v_inst_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_f_41_){
_start:
{
lean_object* v_res_42_; 
v_res_42_ = lp_mathlib_BotHomClass_toBotHom(v_F_34_, v_00_u03b1_35_, v_00_u03b2_36_, v_inst_37_, v_inst_38_, v_inst_39_, v_inst_40_, v_f_41_);
lean_dec(v_inst_39_);
lean_dec(v_inst_38_);
return v_res_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCTopHomOfTopHomClass___redArg(lean_object* v_inst_43_, lean_object* v_inst_44_, lean_object* v_inst_45_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lean_alloc_closure((void*)(lp_mathlib_TopHomClass_toTopHom___boxed), 8, 7);
lean_closure_set(v___x_46_, 0, lean_box(0));
lean_closure_set(v___x_46_, 1, lean_box(0));
lean_closure_set(v___x_46_, 2, lean_box(0));
lean_closure_set(v___x_46_, 3, v_inst_43_);
lean_closure_set(v___x_46_, 4, v_inst_44_);
lean_closure_set(v___x_46_, 5, v_inst_45_);
lean_closure_set(v___x_46_, 6, lean_box(0));
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCTopHomOfTopHomClass(lean_object* v_F_47_, lean_object* v_00_u03b1_48_, lean_object* v_00_u03b2_49_, lean_object* v_inst_50_, lean_object* v_inst_51_, lean_object* v_inst_52_, lean_object* v_inst_53_){
_start:
{
lean_object* v___x_54_; 
v___x_54_ = lean_alloc_closure((void*)(lp_mathlib_TopHomClass_toTopHom___boxed), 8, 7);
lean_closure_set(v___x_54_, 0, lean_box(0));
lean_closure_set(v___x_54_, 1, lean_box(0));
lean_closure_set(v___x_54_, 2, lean_box(0));
lean_closure_set(v___x_54_, 3, v_inst_50_);
lean_closure_set(v___x_54_, 4, v_inst_51_);
lean_closure_set(v___x_54_, 5, v_inst_52_);
lean_closure_set(v___x_54_, 6, lean_box(0));
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCBotHomOfBotHomClass___redArg(lean_object* v_inst_55_, lean_object* v_inst_56_, lean_object* v_inst_57_){
_start:
{
lean_object* v___x_58_; 
v___x_58_ = lean_alloc_closure((void*)(lp_mathlib_BotHomClass_toBotHom___boxed), 8, 7);
lean_closure_set(v___x_58_, 0, lean_box(0));
lean_closure_set(v___x_58_, 1, lean_box(0));
lean_closure_set(v___x_58_, 2, lean_box(0));
lean_closure_set(v___x_58_, 3, v_inst_55_);
lean_closure_set(v___x_58_, 4, v_inst_56_);
lean_closure_set(v___x_58_, 5, v_inst_57_);
lean_closure_set(v___x_58_, 6, lean_box(0));
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCBotHomOfBotHomClass(lean_object* v_F_59_, lean_object* v_00_u03b1_60_, lean_object* v_00_u03b2_61_, lean_object* v_inst_62_, lean_object* v_inst_63_, lean_object* v_inst_64_, lean_object* v_inst_65_){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = lean_alloc_closure((void*)(lp_mathlib_BotHomClass_toBotHom___boxed), 8, 7);
lean_closure_set(v___x_66_, 0, lean_box(0));
lean_closure_set(v___x_66_, 1, lean_box(0));
lean_closure_set(v___x_66_, 2, lean_box(0));
lean_closure_set(v___x_66_, 3, v_inst_62_);
lean_closure_set(v___x_66_, 4, v_inst_63_);
lean_closure_set(v___x_66_, 5, v_inst_64_);
lean_closure_set(v___x_66_, 6, lean_box(0));
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHomClass_toBoundedOrderHom___redArg(lean_object* v_inst_67_, lean_object* v_f_68_){
_start:
{
lean_object* v___x_69_; 
v___x_69_ = lean_apply_1(v_inst_67_, v_f_68_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHomClass_toBoundedOrderHom(lean_object* v_F_70_, lean_object* v_00_u03b1_71_, lean_object* v_00_u03b2_72_, lean_object* v_inst_73_, lean_object* v_inst_74_, lean_object* v_inst_75_, lean_object* v_inst_76_, lean_object* v_inst_77_, lean_object* v_inst_78_, lean_object* v_f_79_){
_start:
{
lean_object* v___x_80_; 
v___x_80_ = lean_apply_1(v_inst_73_, v_f_79_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHomClass_toBoundedOrderHom___boxed(lean_object* v_F_81_, lean_object* v_00_u03b1_82_, lean_object* v_00_u03b2_83_, lean_object* v_inst_84_, lean_object* v_inst_85_, lean_object* v_inst_86_, lean_object* v_inst_87_, lean_object* v_inst_88_, lean_object* v_inst_89_, lean_object* v_f_90_){
_start:
{
lean_object* v_res_91_; 
v_res_91_ = lp_mathlib_BoundedOrderHomClass_toBoundedOrderHom(v_F_81_, v_00_u03b1_82_, v_00_u03b2_83_, v_inst_84_, v_inst_85_, v_inst_86_, v_inst_87_, v_inst_88_, v_inst_89_, v_f_90_);
lean_dec_ref(v_inst_88_);
lean_dec_ref(v_inst_87_);
lean_dec_ref(v_inst_86_);
lean_dec_ref(v_inst_85_);
return v_res_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCBoundedOrderHomOfBoundedOrderHomClass___redArg(lean_object* v_inst_92_, lean_object* v_inst_93_, lean_object* v_inst_94_, lean_object* v_inst_95_, lean_object* v_inst_96_){
_start:
{
lean_object* v___x_97_; 
v___x_97_ = lean_alloc_closure((void*)(lp_mathlib_BoundedOrderHomClass_toBoundedOrderHom___boxed), 10, 9);
lean_closure_set(v___x_97_, 0, lean_box(0));
lean_closure_set(v___x_97_, 1, lean_box(0));
lean_closure_set(v___x_97_, 2, lean_box(0));
lean_closure_set(v___x_97_, 3, v_inst_92_);
lean_closure_set(v___x_97_, 4, v_inst_93_);
lean_closure_set(v___x_97_, 5, v_inst_94_);
lean_closure_set(v___x_97_, 6, v_inst_95_);
lean_closure_set(v___x_97_, 7, v_inst_96_);
lean_closure_set(v___x_97_, 8, lean_box(0));
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCBoundedOrderHomOfBoundedOrderHomClass(lean_object* v_F_98_, lean_object* v_00_u03b1_99_, lean_object* v_00_u03b2_100_, lean_object* v_inst_101_, lean_object* v_inst_102_, lean_object* v_inst_103_, lean_object* v_inst_104_, lean_object* v_inst_105_, lean_object* v_inst_106_){
_start:
{
lean_object* v___x_107_; 
v___x_107_ = lean_alloc_closure((void*)(lp_mathlib_BoundedOrderHomClass_toBoundedOrderHom___boxed), 10, 9);
lean_closure_set(v___x_107_, 0, lean_box(0));
lean_closure_set(v___x_107_, 1, lean_box(0));
lean_closure_set(v___x_107_, 2, lean_box(0));
lean_closure_set(v___x_107_, 3, v_inst_101_);
lean_closure_set(v___x_107_, 4, v_inst_102_);
lean_closure_set(v___x_107_, 5, v_inst_103_);
lean_closure_set(v___x_107_, 6, v_inst_104_);
lean_closure_set(v___x_107_, 7, v_inst_105_);
lean_closure_set(v___x_107_, 8, lean_box(0));
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instFunLike___lam__0(lean_object* v_self_108_, lean_object* v___y_109_){
_start:
{
lean_object* v___x_110_; 
v___x_110_ = lean_apply_1(v_self_108_, v___y_109_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instFunLike(lean_object* v_00_u03b1_112_, lean_object* v_00_u03b2_113_, lean_object* v_inst_114_, lean_object* v_inst_115_){
_start:
{
lean_object* v___f_116_; 
v___f_116_ = ((lean_object*)(lp_mathlib_TopHom_instFunLike___closed__0));
return v___f_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instFunLike___boxed(lean_object* v_00_u03b1_117_, lean_object* v_00_u03b2_118_, lean_object* v_inst_119_, lean_object* v_inst_120_){
_start:
{
lean_object* v_res_121_; 
v_res_121_ = lp_mathlib_TopHom_instFunLike(v_00_u03b1_117_, v_00_u03b2_118_, v_inst_119_, v_inst_120_);
lean_dec(v_inst_120_);
lean_dec(v_inst_119_);
return v_res_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instFunLike(lean_object* v_00_u03b1_122_, lean_object* v_00_u03b2_123_, lean_object* v_inst_124_, lean_object* v_inst_125_){
_start:
{
lean_object* v___f_126_; 
v___f_126_ = ((lean_object*)(lp_mathlib_TopHom_instFunLike___closed__0));
return v___f_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instFunLike___boxed(lean_object* v_00_u03b1_127_, lean_object* v_00_u03b2_128_, lean_object* v_inst_129_, lean_object* v_inst_130_){
_start:
{
lean_object* v_res_131_; 
v_res_131_ = lp_mathlib_BotHom_instFunLike(v_00_u03b1_127_, v_00_u03b2_128_, v_inst_129_, v_inst_130_);
lean_dec(v_inst_130_);
lean_dec(v_inst_129_);
return v_res_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_copy___redArg(lean_object* v_f_x27_132_){
_start:
{
lean_inc(v_f_x27_132_);
return v_f_x27_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_copy___redArg___boxed(lean_object* v_f_x27_133_){
_start:
{
lean_object* v_res_134_; 
v_res_134_ = lp_mathlib_TopHom_copy___redArg(v_f_x27_133_);
lean_dec(v_f_x27_133_);
return v_res_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_copy(lean_object* v_00_u03b1_135_, lean_object* v_00_u03b2_136_, lean_object* v_inst_137_, lean_object* v_inst_138_, lean_object* v_f_139_, lean_object* v_f_x27_140_, lean_object* v_h_141_){
_start:
{
lean_inc(v_f_x27_140_);
return v_f_x27_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_copy___boxed(lean_object* v_00_u03b1_142_, lean_object* v_00_u03b2_143_, lean_object* v_inst_144_, lean_object* v_inst_145_, lean_object* v_f_146_, lean_object* v_f_x27_147_, lean_object* v_h_148_){
_start:
{
lean_object* v_res_149_; 
v_res_149_ = lp_mathlib_TopHom_copy(v_00_u03b1_142_, v_00_u03b2_143_, v_inst_144_, v_inst_145_, v_f_146_, v_f_x27_147_, v_h_148_);
lean_dec(v_f_x27_147_);
lean_dec(v_f_146_);
lean_dec(v_inst_145_);
lean_dec(v_inst_144_);
return v_res_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_copy___redArg(lean_object* v_f_x27_150_){
_start:
{
lean_inc(v_f_x27_150_);
return v_f_x27_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_copy___redArg___boxed(lean_object* v_f_x27_151_){
_start:
{
lean_object* v_res_152_; 
v_res_152_ = lp_mathlib_BotHom_copy___redArg(v_f_x27_151_);
lean_dec(v_f_x27_151_);
return v_res_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_copy(lean_object* v_00_u03b1_153_, lean_object* v_00_u03b2_154_, lean_object* v_inst_155_, lean_object* v_inst_156_, lean_object* v_f_157_, lean_object* v_f_x27_158_, lean_object* v_h_159_){
_start:
{
lean_inc(v_f_x27_158_);
return v_f_x27_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_copy___boxed(lean_object* v_00_u03b1_160_, lean_object* v_00_u03b2_161_, lean_object* v_inst_162_, lean_object* v_inst_163_, lean_object* v_f_164_, lean_object* v_f_x27_165_, lean_object* v_h_166_){
_start:
{
lean_object* v_res_167_; 
v_res_167_ = lp_mathlib_BotHom_copy(v_00_u03b1_160_, v_00_u03b2_161_, v_inst_162_, v_inst_163_, v_f_164_, v_f_x27_165_, v_h_166_);
lean_dec(v_f_x27_165_);
lean_dec(v_f_164_);
lean_dec(v_inst_163_);
lean_dec(v_inst_162_);
return v_res_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instInhabited___redArg___lam__0(lean_object* v_inst_168_, lean_object* v_x_169_){
_start:
{
lean_inc(v_inst_168_);
return v_inst_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instInhabited___redArg___lam__0___boxed(lean_object* v_inst_170_, lean_object* v_x_171_){
_start:
{
lean_object* v_res_172_; 
v_res_172_ = lp_mathlib_TopHom_instInhabited___redArg___lam__0(v_inst_170_, v_x_171_);
lean_dec(v_x_171_);
lean_dec(v_inst_170_);
return v_res_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instInhabited___redArg(lean_object* v_inst_173_){
_start:
{
lean_object* v___f_174_; 
v___f_174_ = lean_alloc_closure((void*)(lp_mathlib_TopHom_instInhabited___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_174_, 0, v_inst_173_);
return v___f_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instInhabited(lean_object* v_00_u03b1_175_, lean_object* v_00_u03b2_176_, lean_object* v_inst_177_, lean_object* v_inst_178_){
_start:
{
lean_object* v___f_179_; 
v___f_179_ = lean_alloc_closure((void*)(lp_mathlib_TopHom_instInhabited___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_179_, 0, v_inst_178_);
return v___f_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instInhabited___boxed(lean_object* v_00_u03b1_180_, lean_object* v_00_u03b2_181_, lean_object* v_inst_182_, lean_object* v_inst_183_){
_start:
{
lean_object* v_res_184_; 
v_res_184_ = lp_mathlib_TopHom_instInhabited(v_00_u03b1_180_, v_00_u03b2_181_, v_inst_182_, v_inst_183_);
lean_dec(v_inst_182_);
return v_res_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instInhabited___redArg(lean_object* v_inst_185_){
_start:
{
lean_object* v___f_186_; 
v___f_186_ = lean_alloc_closure((void*)(lp_mathlib_TopHom_instInhabited___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_186_, 0, v_inst_185_);
return v___f_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instInhabited(lean_object* v_00_u03b1_187_, lean_object* v_00_u03b2_188_, lean_object* v_inst_189_, lean_object* v_inst_190_){
_start:
{
lean_object* v___f_191_; 
v___f_191_ = lean_alloc_closure((void*)(lp_mathlib_TopHom_instInhabited___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_191_, 0, v_inst_190_);
return v___f_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instInhabited___boxed(lean_object* v_00_u03b1_192_, lean_object* v_00_u03b2_193_, lean_object* v_inst_194_, lean_object* v_inst_195_){
_start:
{
lean_object* v_res_196_; 
v_res_196_ = lp_mathlib_BotHom_instInhabited(v_00_u03b1_192_, v_00_u03b2_193_, v_inst_194_, v_inst_195_);
lean_dec(v_inst_194_);
return v_res_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_id(lean_object* v_00_u03b1_198_, lean_object* v_inst_199_){
_start:
{
lean_object* v___x_200_; 
v___x_200_ = ((lean_object*)(lp_mathlib_TopHom_id___closed__0));
return v___x_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_id___boxed(lean_object* v_00_u03b1_201_, lean_object* v_inst_202_){
_start:
{
lean_object* v_res_203_; 
v_res_203_ = lp_mathlib_TopHom_id(v_00_u03b1_201_, v_inst_202_);
lean_dec(v_inst_202_);
return v_res_203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_id(lean_object* v_00_u03b1_204_, lean_object* v_inst_205_){
_start:
{
lean_object* v___x_206_; 
v___x_206_ = ((lean_object*)(lp_mathlib_TopHom_id___closed__0));
return v___x_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_id___boxed(lean_object* v_00_u03b1_207_, lean_object* v_inst_208_){
_start:
{
lean_object* v_res_209_; 
v_res_209_ = lp_mathlib_BotHom_id(v_00_u03b1_207_, v_inst_208_);
lean_dec(v_inst_208_);
return v_res_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_comp___redArg___lam__0(lean_object* v_f_210_, lean_object* v___y_211_){
_start:
{
lean_object* v___x_212_; 
v___x_212_ = lean_apply_1(v_f_210_, v___y_211_);
return v___x_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_comp___redArg___lam__1(lean_object* v_g_213_, lean_object* v___y_214_){
_start:
{
lean_object* v___x_215_; 
v___x_215_ = lean_apply_1(v_g_213_, v___y_214_);
return v___x_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_comp___redArg(lean_object* v_f_216_, lean_object* v_g_217_){
_start:
{
lean_object* v___f_218_; lean_object* v___f_219_; lean_object* v___x_220_; 
v___f_218_ = lean_alloc_closure((void*)(lp_mathlib_TopHom_comp___redArg___lam__0), 2, 1);
lean_closure_set(v___f_218_, 0, v_f_216_);
v___f_219_ = lean_alloc_closure((void*)(lp_mathlib_TopHom_comp___redArg___lam__1), 2, 1);
lean_closure_set(v___f_219_, 0, v_g_217_);
v___x_220_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_220_, 0, lean_box(0));
lean_closure_set(v___x_220_, 1, lean_box(0));
lean_closure_set(v___x_220_, 2, lean_box(0));
lean_closure_set(v___x_220_, 3, v___f_218_);
lean_closure_set(v___x_220_, 4, v___f_219_);
return v___x_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_comp(lean_object* v_00_u03b1_221_, lean_object* v_00_u03b2_222_, lean_object* v_00_u03b3_223_, lean_object* v_inst_224_, lean_object* v_inst_225_, lean_object* v_inst_226_, lean_object* v_f_227_, lean_object* v_g_228_){
_start:
{
lean_object* v___x_229_; 
v___x_229_ = lp_mathlib_TopHom_comp___redArg(v_f_227_, v_g_228_);
return v___x_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_comp___boxed(lean_object* v_00_u03b1_230_, lean_object* v_00_u03b2_231_, lean_object* v_00_u03b3_232_, lean_object* v_inst_233_, lean_object* v_inst_234_, lean_object* v_inst_235_, lean_object* v_f_236_, lean_object* v_g_237_){
_start:
{
lean_object* v_res_238_; 
v_res_238_ = lp_mathlib_TopHom_comp(v_00_u03b1_230_, v_00_u03b2_231_, v_00_u03b3_232_, v_inst_233_, v_inst_234_, v_inst_235_, v_f_236_, v_g_237_);
lean_dec(v_inst_235_);
lean_dec(v_inst_234_);
lean_dec(v_inst_233_);
return v_res_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_comp___redArg(lean_object* v_f_239_, lean_object* v_g_240_){
_start:
{
lean_object* v___f_241_; lean_object* v___f_242_; lean_object* v___x_243_; 
v___f_241_ = lean_alloc_closure((void*)(lp_mathlib_TopHom_comp___redArg___lam__0), 2, 1);
lean_closure_set(v___f_241_, 0, v_f_239_);
v___f_242_ = lean_alloc_closure((void*)(lp_mathlib_TopHom_comp___redArg___lam__1), 2, 1);
lean_closure_set(v___f_242_, 0, v_g_240_);
v___x_243_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_243_, 0, lean_box(0));
lean_closure_set(v___x_243_, 1, lean_box(0));
lean_closure_set(v___x_243_, 2, lean_box(0));
lean_closure_set(v___x_243_, 3, v___f_241_);
lean_closure_set(v___x_243_, 4, v___f_242_);
return v___x_243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_comp(lean_object* v_00_u03b1_244_, lean_object* v_00_u03b2_245_, lean_object* v_00_u03b3_246_, lean_object* v_inst_247_, lean_object* v_inst_248_, lean_object* v_inst_249_, lean_object* v_f_250_, lean_object* v_g_251_){
_start:
{
lean_object* v___x_252_; 
v___x_252_ = lp_mathlib_BotHom_comp___redArg(v_f_250_, v_g_251_);
return v___x_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_comp___boxed(lean_object* v_00_u03b1_253_, lean_object* v_00_u03b2_254_, lean_object* v_00_u03b3_255_, lean_object* v_inst_256_, lean_object* v_inst_257_, lean_object* v_inst_258_, lean_object* v_f_259_, lean_object* v_g_260_){
_start:
{
lean_object* v_res_261_; 
v_res_261_ = lp_mathlib_BotHom_comp(v_00_u03b1_253_, v_00_u03b2_254_, v_00_u03b3_255_, v_inst_256_, v_inst_257_, v_inst_258_, v_f_259_, v_g_260_);
lean_dec(v_inst_258_);
lean_dec(v_inst_257_);
lean_dec(v_inst_256_);
return v_res_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instLE(lean_object* v_00_u03b1_262_, lean_object* v_00_u03b2_263_, lean_object* v_inst_264_, lean_object* v_inst_265_, lean_object* v_inst_266_){
_start:
{
lean_object* v___x_267_; 
v___x_267_ = lean_box(0);
return v___x_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instLE___boxed(lean_object* v_00_u03b1_268_, lean_object* v_00_u03b2_269_, lean_object* v_inst_270_, lean_object* v_inst_271_, lean_object* v_inst_272_){
_start:
{
lean_object* v_res_273_; 
v_res_273_ = lp_mathlib_TopHom_instLE(v_00_u03b1_268_, v_00_u03b2_269_, v_inst_270_, v_inst_271_, v_inst_272_);
lean_dec(v_inst_272_);
lean_dec(v_inst_270_);
return v_res_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instLE(lean_object* v_00_u03b1_274_, lean_object* v_00_u03b2_275_, lean_object* v_inst_276_, lean_object* v_inst_277_, lean_object* v_inst_278_){
_start:
{
lean_object* v___x_279_; 
v___x_279_ = lean_box(0);
return v___x_279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instLE___boxed(lean_object* v_00_u03b1_280_, lean_object* v_00_u03b2_281_, lean_object* v_inst_282_, lean_object* v_inst_283_, lean_object* v_inst_284_){
_start:
{
lean_object* v_res_285_; 
v_res_285_ = lp_mathlib_BotHom_instLE(v_00_u03b1_280_, v_00_u03b2_281_, v_inst_282_, v_inst_283_, v_inst_284_);
lean_dec(v_inst_284_);
lean_dec(v_inst_282_);
return v_res_285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instPreorder(lean_object* v_00_u03b1_289_, lean_object* v_00_u03b2_290_, lean_object* v_inst_291_, lean_object* v_inst_292_, lean_object* v_inst_293_){
_start:
{
lean_object* v___x_294_; 
v___x_294_ = ((lean_object*)(lp_mathlib_TopHom_instPreorder___closed__0));
return v___x_294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instPreorder___boxed(lean_object* v_00_u03b1_295_, lean_object* v_00_u03b2_296_, lean_object* v_inst_297_, lean_object* v_inst_298_, lean_object* v_inst_299_){
_start:
{
lean_object* v_res_300_; 
v_res_300_ = lp_mathlib_TopHom_instPreorder(v_00_u03b1_295_, v_00_u03b2_296_, v_inst_297_, v_inst_298_, v_inst_299_);
lean_dec(v_inst_299_);
lean_dec_ref(v_inst_298_);
lean_dec(v_inst_297_);
return v_res_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instPreorder(lean_object* v_00_u03b1_301_, lean_object* v_00_u03b2_302_, lean_object* v_inst_303_, lean_object* v_inst_304_, lean_object* v_inst_305_){
_start:
{
lean_object* v___x_306_; 
v___x_306_ = ((lean_object*)(lp_mathlib_TopHom_instPreorder___closed__0));
return v___x_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instPreorder___boxed(lean_object* v_00_u03b1_307_, lean_object* v_00_u03b2_308_, lean_object* v_inst_309_, lean_object* v_inst_310_, lean_object* v_inst_311_){
_start:
{
lean_object* v_res_312_; 
v_res_312_ = lp_mathlib_BotHom_instPreorder(v_00_u03b1_307_, v_00_u03b2_308_, v_inst_309_, v_inst_310_, v_inst_311_);
lean_dec(v_inst_311_);
lean_dec_ref(v_inst_310_);
lean_dec(v_inst_309_);
return v_res_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instPartialOrder(lean_object* v_00_u03b1_313_, lean_object* v_00_u03b2_314_, lean_object* v_inst_315_, lean_object* v_inst_316_, lean_object* v_inst_317_){
_start:
{
lean_object* v___x_318_; 
v___x_318_ = ((lean_object*)(lp_mathlib_TopHom_instPreorder___closed__0));
return v___x_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instPartialOrder___boxed(lean_object* v_00_u03b1_319_, lean_object* v_00_u03b2_320_, lean_object* v_inst_321_, lean_object* v_inst_322_, lean_object* v_inst_323_){
_start:
{
lean_object* v_res_324_; 
v_res_324_ = lp_mathlib_TopHom_instPartialOrder(v_00_u03b1_319_, v_00_u03b2_320_, v_inst_321_, v_inst_322_, v_inst_323_);
lean_dec(v_inst_323_);
lean_dec_ref(v_inst_322_);
lean_dec(v_inst_321_);
return v_res_324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instPartialOrder(lean_object* v_00_u03b1_325_, lean_object* v_00_u03b2_326_, lean_object* v_inst_327_, lean_object* v_inst_328_, lean_object* v_inst_329_){
_start:
{
lean_object* v___x_330_; 
v___x_330_ = ((lean_object*)(lp_mathlib_TopHom_instPreorder___closed__0));
return v___x_330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instPartialOrder___boxed(lean_object* v_00_u03b1_331_, lean_object* v_00_u03b2_332_, lean_object* v_inst_333_, lean_object* v_inst_334_, lean_object* v_inst_335_){
_start:
{
lean_object* v_res_336_; 
v_res_336_ = lp_mathlib_BotHom_instPartialOrder(v_00_u03b1_331_, v_00_u03b2_332_, v_inst_333_, v_inst_334_, v_inst_335_);
lean_dec(v_inst_335_);
lean_dec_ref(v_inst_334_);
lean_dec(v_inst_333_);
return v_res_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instOrderTop___redArg(lean_object* v_inst_337_){
_start:
{
lean_object* v___f_338_; 
v___f_338_ = lean_alloc_closure((void*)(lp_mathlib_TopHom_instInhabited___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_338_, 0, v_inst_337_);
return v___f_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instOrderTop(lean_object* v_00_u03b1_339_, lean_object* v_00_u03b2_340_, lean_object* v_inst_341_, lean_object* v_inst_342_, lean_object* v_inst_343_){
_start:
{
lean_object* v___f_344_; 
v___f_344_ = lean_alloc_closure((void*)(lp_mathlib_TopHom_instInhabited___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_344_, 0, v_inst_343_);
return v___f_344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instOrderTop___boxed(lean_object* v_00_u03b1_345_, lean_object* v_00_u03b2_346_, lean_object* v_inst_347_, lean_object* v_inst_348_, lean_object* v_inst_349_){
_start:
{
lean_object* v_res_350_; 
v_res_350_ = lp_mathlib_TopHom_instOrderTop(v_00_u03b1_345_, v_00_u03b2_346_, v_inst_347_, v_inst_348_, v_inst_349_);
lean_dec(v_inst_347_);
return v_res_350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instOrderBot___redArg(lean_object* v_inst_351_){
_start:
{
lean_object* v___f_352_; 
v___f_352_ = lean_alloc_closure((void*)(lp_mathlib_TopHom_instInhabited___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_352_, 0, v_inst_351_);
return v___f_352_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instOrderBot(lean_object* v_00_u03b1_353_, lean_object* v_00_u03b2_354_, lean_object* v_inst_355_, lean_object* v_inst_356_, lean_object* v_inst_357_){
_start:
{
lean_object* v___f_358_; 
v___f_358_ = lean_alloc_closure((void*)(lp_mathlib_TopHom_instInhabited___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_358_, 0, v_inst_357_);
return v___f_358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instOrderBot___boxed(lean_object* v_00_u03b1_359_, lean_object* v_00_u03b2_360_, lean_object* v_inst_361_, lean_object* v_inst_362_, lean_object* v_inst_363_){
_start:
{
lean_object* v_res_364_; 
v_res_364_ = lp_mathlib_BotHom_instOrderBot(v_00_u03b1_359_, v_00_u03b2_360_, v_inst_361_, v_inst_362_, v_inst_363_);
lean_dec(v_inst_361_);
return v_res_364_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instMin___redArg___lam__0(lean_object* v_inst_365_, lean_object* v_f_366_, lean_object* v_g_367_, lean_object* v___y_368_){
_start:
{
lean_object* v_inf_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; 
v_inf_369_ = lean_ctor_get(v_inst_365_, 1);
lean_inc(v_inf_369_);
lean_dec_ref(v_inst_365_);
lean_inc(v___y_368_);
v___x_370_ = lean_apply_1(v_f_366_, v___y_368_);
v___x_371_ = lean_apply_1(v_g_367_, v___y_368_);
v___x_372_ = lean_apply_2(v_inf_369_, v___x_370_, v___x_371_);
return v___x_372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instMin___redArg(lean_object* v_inst_373_){
_start:
{
lean_object* v___f_374_; 
v___f_374_ = lean_alloc_closure((void*)(lp_mathlib_TopHom_instMin___redArg___lam__0), 4, 1);
lean_closure_set(v___f_374_, 0, v_inst_373_);
return v___f_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instMin(lean_object* v_00_u03b1_375_, lean_object* v_00_u03b2_376_, lean_object* v_inst_377_, lean_object* v_inst_378_, lean_object* v_inst_379_){
_start:
{
lean_object* v___f_380_; 
v___f_380_ = lean_alloc_closure((void*)(lp_mathlib_TopHom_instMin___redArg___lam__0), 4, 1);
lean_closure_set(v___f_380_, 0, v_inst_378_);
return v___f_380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instMin___boxed(lean_object* v_00_u03b1_381_, lean_object* v_00_u03b2_382_, lean_object* v_inst_383_, lean_object* v_inst_384_, lean_object* v_inst_385_){
_start:
{
lean_object* v_res_386_; 
v_res_386_ = lp_mathlib_TopHom_instMin(v_00_u03b1_381_, v_00_u03b2_382_, v_inst_383_, v_inst_384_, v_inst_385_);
lean_dec(v_inst_385_);
lean_dec(v_inst_383_);
return v_res_386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instMax___redArg___lam__0(lean_object* v_inst_387_, lean_object* v_f_388_, lean_object* v_g_389_, lean_object* v___y_390_){
_start:
{
lean_object* v_sup_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; 
v_sup_391_ = lean_ctor_get(v_inst_387_, 1);
lean_inc(v_sup_391_);
lean_dec_ref(v_inst_387_);
lean_inc(v___y_390_);
v___x_392_ = lean_apply_1(v_f_388_, v___y_390_);
v___x_393_ = lean_apply_1(v_g_389_, v___y_390_);
v___x_394_ = lean_apply_2(v_sup_391_, v___x_392_, v___x_393_);
return v___x_394_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instMax___redArg(lean_object* v_inst_395_){
_start:
{
lean_object* v___f_396_; 
v___f_396_ = lean_alloc_closure((void*)(lp_mathlib_BotHom_instMax___redArg___lam__0), 4, 1);
lean_closure_set(v___f_396_, 0, v_inst_395_);
return v___f_396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instMax(lean_object* v_00_u03b1_397_, lean_object* v_00_u03b2_398_, lean_object* v_inst_399_, lean_object* v_inst_400_, lean_object* v_inst_401_){
_start:
{
lean_object* v___f_402_; 
v___f_402_ = lean_alloc_closure((void*)(lp_mathlib_BotHom_instMax___redArg___lam__0), 4, 1);
lean_closure_set(v___f_402_, 0, v_inst_400_);
return v___f_402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instMax___boxed(lean_object* v_00_u03b1_403_, lean_object* v_00_u03b2_404_, lean_object* v_inst_405_, lean_object* v_inst_406_, lean_object* v_inst_407_){
_start:
{
lean_object* v_res_408_; 
v_res_408_ = lp_mathlib_BotHom_instMax(v_00_u03b1_403_, v_00_u03b2_404_, v_inst_405_, v_inst_406_, v_inst_407_);
lean_dec(v_inst_407_);
lean_dec(v_inst_405_);
return v_res_408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instSemilatticeInf___redArg(lean_object* v_inst_409_, lean_object* v_inst_410_, lean_object* v_inst_411_){
_start:
{
lean_object* v_toPartialOrder_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v_toLT_415_; lean_object* v___f_416_; lean_object* v___x_417_; 
v_toPartialOrder_412_ = lean_ctor_get(v_inst_410_, 0);
v___x_413_ = lean_box(0);
v___x_414_ = lp_mathlib_TopHom_instPreorder(lean_box(0), lean_box(0), v_inst_409_, v_toPartialOrder_412_, v_inst_411_);
v_toLT_415_ = lean_ctor_get(v___x_414_, 1);
lean_inc(v_toLT_415_);
lean_dec_ref(v___x_414_);
v___f_416_ = lean_alloc_closure((void*)(lp_mathlib_TopHom_instMin___redArg___lam__0), 4, 1);
lean_closure_set(v___f_416_, 0, v_inst_410_);
v___x_417_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v___f_416_, v___x_413_, v_toLT_415_);
return v___x_417_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instSemilatticeInf___redArg___boxed(lean_object* v_inst_418_, lean_object* v_inst_419_, lean_object* v_inst_420_){
_start:
{
lean_object* v_res_421_; 
v_res_421_ = lp_mathlib_TopHom_instSemilatticeInf___redArg(v_inst_418_, v_inst_419_, v_inst_420_);
lean_dec(v_inst_420_);
lean_dec(v_inst_418_);
return v_res_421_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instSemilatticeInf(lean_object* v_00_u03b1_422_, lean_object* v_00_u03b2_423_, lean_object* v_inst_424_, lean_object* v_inst_425_, lean_object* v_inst_426_){
_start:
{
lean_object* v___x_427_; 
v___x_427_ = lp_mathlib_TopHom_instSemilatticeInf___redArg(v_inst_424_, v_inst_425_, v_inst_426_);
return v___x_427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instSemilatticeInf___boxed(lean_object* v_00_u03b1_428_, lean_object* v_00_u03b2_429_, lean_object* v_inst_430_, lean_object* v_inst_431_, lean_object* v_inst_432_){
_start:
{
lean_object* v_res_433_; 
v_res_433_ = lp_mathlib_TopHom_instSemilatticeInf(v_00_u03b1_428_, v_00_u03b2_429_, v_inst_430_, v_inst_431_, v_inst_432_);
lean_dec(v_inst_432_);
lean_dec(v_inst_430_);
return v_res_433_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instSemilatticeSup___redArg___lam__0(lean_object* v_sup_434_, lean_object* v_a_435_, lean_object* v_b_436_, lean_object* v___y_437_){
_start:
{
lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; 
lean_inc(v___y_437_);
v___x_438_ = lean_apply_1(v_a_435_, v___y_437_);
v___x_439_ = lean_apply_1(v_b_436_, v___y_437_);
v___x_440_ = lean_apply_2(v_sup_434_, v___x_438_, v___x_439_);
return v___x_440_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instSemilatticeSup___redArg(lean_object* v_inst_441_, lean_object* v_inst_442_, lean_object* v_inst_443_){
_start:
{
lean_object* v_toPartialOrder_444_; lean_object* v_sup_445_; lean_object* v___x_447_; uint8_t v_isShared_448_; uint8_t v_isSharedCheck_464_; 
v_toPartialOrder_444_ = lean_ctor_get(v_inst_442_, 0);
v_sup_445_ = lean_ctor_get(v_inst_442_, 1);
v_isSharedCheck_464_ = !lean_is_exclusive(v_inst_442_);
if (v_isSharedCheck_464_ == 0)
{
v___x_447_ = v_inst_442_;
v_isShared_448_ = v_isSharedCheck_464_;
goto v_resetjp_446_;
}
else
{
lean_inc(v_sup_445_);
lean_inc(v_toPartialOrder_444_);
lean_dec(v_inst_442_);
v___x_447_ = lean_box(0);
v_isShared_448_ = v_isSharedCheck_464_;
goto v_resetjp_446_;
}
v_resetjp_446_:
{
lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v_toLT_451_; lean_object* v___x_453_; uint8_t v_isShared_454_; uint8_t v_isSharedCheck_462_; 
v___x_449_ = lean_box(0);
v___x_450_ = lp_mathlib_BotHom_instPreorder(lean_box(0), lean_box(0), v_inst_441_, v_toPartialOrder_444_, v_inst_443_);
lean_dec_ref(v_toPartialOrder_444_);
v_toLT_451_ = lean_ctor_get(v___x_450_, 1);
v_isSharedCheck_462_ = !lean_is_exclusive(v___x_450_);
if (v_isSharedCheck_462_ == 0)
{
lean_object* v_unused_463_; 
v_unused_463_ = lean_ctor_get(v___x_450_, 0);
lean_dec(v_unused_463_);
v___x_453_ = v___x_450_;
v_isShared_454_ = v_isSharedCheck_462_;
goto v_resetjp_452_;
}
else
{
lean_inc(v_toLT_451_);
lean_dec(v___x_450_);
v___x_453_ = lean_box(0);
v_isShared_454_ = v_isSharedCheck_462_;
goto v_resetjp_452_;
}
v_resetjp_452_:
{
lean_object* v___f_455_; lean_object* v___x_457_; 
v___f_455_ = lean_alloc_closure((void*)(lp_mathlib_BotHom_instSemilatticeSup___redArg___lam__0), 4, 1);
lean_closure_set(v___f_455_, 0, v_sup_445_);
if (v_isShared_454_ == 0)
{
lean_ctor_set(v___x_453_, 0, v___x_449_);
v___x_457_ = v___x_453_;
goto v_reusejp_456_;
}
else
{
lean_object* v_reuseFailAlloc_461_; 
v_reuseFailAlloc_461_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_461_, 0, v___x_449_);
lean_ctor_set(v_reuseFailAlloc_461_, 1, v_toLT_451_);
v___x_457_ = v_reuseFailAlloc_461_;
goto v_reusejp_456_;
}
v_reusejp_456_:
{
lean_object* v___x_459_; 
if (v_isShared_448_ == 0)
{
lean_ctor_set(v___x_447_, 1, v___f_455_);
lean_ctor_set(v___x_447_, 0, v___x_457_);
v___x_459_ = v___x_447_;
goto v_reusejp_458_;
}
else
{
lean_object* v_reuseFailAlloc_460_; 
v_reuseFailAlloc_460_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_460_, 0, v___x_457_);
lean_ctor_set(v_reuseFailAlloc_460_, 1, v___f_455_);
v___x_459_ = v_reuseFailAlloc_460_;
goto v_reusejp_458_;
}
v_reusejp_458_:
{
return v___x_459_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instSemilatticeSup___redArg___boxed(lean_object* v_inst_465_, lean_object* v_inst_466_, lean_object* v_inst_467_){
_start:
{
lean_object* v_res_468_; 
v_res_468_ = lp_mathlib_BotHom_instSemilatticeSup___redArg(v_inst_465_, v_inst_466_, v_inst_467_);
lean_dec(v_inst_467_);
lean_dec(v_inst_465_);
return v_res_468_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instSemilatticeSup(lean_object* v_00_u03b1_469_, lean_object* v_00_u03b2_470_, lean_object* v_inst_471_, lean_object* v_inst_472_, lean_object* v_inst_473_){
_start:
{
lean_object* v___x_474_; 
v___x_474_ = lp_mathlib_BotHom_instSemilatticeSup___redArg(v_inst_471_, v_inst_472_, v_inst_473_);
return v___x_474_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instSemilatticeSup___boxed(lean_object* v_00_u03b1_475_, lean_object* v_00_u03b2_476_, lean_object* v_inst_477_, lean_object* v_inst_478_, lean_object* v_inst_479_){
_start:
{
lean_object* v_res_480_; 
v_res_480_ = lp_mathlib_BotHom_instSemilatticeSup(v_00_u03b1_475_, v_00_u03b2_476_, v_inst_477_, v_inst_478_, v_inst_479_);
lean_dec(v_inst_479_);
lean_dec(v_inst_477_);
return v_res_480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instMax___redArg(lean_object* v_inst_481_){
_start:
{
lean_object* v___f_482_; 
v___f_482_ = lean_alloc_closure((void*)(lp_mathlib_BotHom_instMax___redArg___lam__0), 4, 1);
lean_closure_set(v___f_482_, 0, v_inst_481_);
return v___f_482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instMax(lean_object* v_00_u03b1_483_, lean_object* v_00_u03b2_484_, lean_object* v_inst_485_, lean_object* v_inst_486_, lean_object* v_inst_487_){
_start:
{
lean_object* v___f_488_; 
v___f_488_ = lean_alloc_closure((void*)(lp_mathlib_BotHom_instMax___redArg___lam__0), 4, 1);
lean_closure_set(v___f_488_, 0, v_inst_486_);
return v___f_488_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instMax___boxed(lean_object* v_00_u03b1_489_, lean_object* v_00_u03b2_490_, lean_object* v_inst_491_, lean_object* v_inst_492_, lean_object* v_inst_493_){
_start:
{
lean_object* v_res_494_; 
v_res_494_ = lp_mathlib_TopHom_instMax(v_00_u03b1_489_, v_00_u03b2_490_, v_inst_491_, v_inst_492_, v_inst_493_);
lean_dec(v_inst_493_);
lean_dec(v_inst_491_);
return v_res_494_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instMin___redArg(lean_object* v_inst_495_){
_start:
{
lean_object* v___f_496_; 
v___f_496_ = lean_alloc_closure((void*)(lp_mathlib_TopHom_instMin___redArg___lam__0), 4, 1);
lean_closure_set(v___f_496_, 0, v_inst_495_);
return v___f_496_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instMin(lean_object* v_00_u03b1_497_, lean_object* v_00_u03b2_498_, lean_object* v_inst_499_, lean_object* v_inst_500_, lean_object* v_inst_501_){
_start:
{
lean_object* v___f_502_; 
v___f_502_ = lean_alloc_closure((void*)(lp_mathlib_TopHom_instMin___redArg___lam__0), 4, 1);
lean_closure_set(v___f_502_, 0, v_inst_500_);
return v___f_502_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instMin___boxed(lean_object* v_00_u03b1_503_, lean_object* v_00_u03b2_504_, lean_object* v_inst_505_, lean_object* v_inst_506_, lean_object* v_inst_507_){
_start:
{
lean_object* v_res_508_; 
v_res_508_ = lp_mathlib_BotHom_instMin(v_00_u03b1_503_, v_00_u03b2_504_, v_inst_505_, v_inst_506_, v_inst_507_);
lean_dec(v_inst_507_);
lean_dec(v_inst_505_);
return v_res_508_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instSemilatticeSup___redArg(lean_object* v_inst_509_, lean_object* v_inst_510_, lean_object* v_inst_511_){
_start:
{
lean_object* v_toPartialOrder_512_; lean_object* v_sup_513_; lean_object* v___x_515_; uint8_t v_isShared_516_; uint8_t v_isSharedCheck_532_; 
v_toPartialOrder_512_ = lean_ctor_get(v_inst_510_, 0);
v_sup_513_ = lean_ctor_get(v_inst_510_, 1);
v_isSharedCheck_532_ = !lean_is_exclusive(v_inst_510_);
if (v_isSharedCheck_532_ == 0)
{
v___x_515_ = v_inst_510_;
v_isShared_516_ = v_isSharedCheck_532_;
goto v_resetjp_514_;
}
else
{
lean_inc(v_sup_513_);
lean_inc(v_toPartialOrder_512_);
lean_dec(v_inst_510_);
v___x_515_ = lean_box(0);
v_isShared_516_ = v_isSharedCheck_532_;
goto v_resetjp_514_;
}
v_resetjp_514_:
{
lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v_toLT_519_; lean_object* v___x_521_; uint8_t v_isShared_522_; uint8_t v_isSharedCheck_530_; 
v___x_517_ = lean_box(0);
v___x_518_ = lp_mathlib_TopHom_instPreorder(lean_box(0), lean_box(0), v_inst_509_, v_toPartialOrder_512_, v_inst_511_);
lean_dec_ref(v_toPartialOrder_512_);
v_toLT_519_ = lean_ctor_get(v___x_518_, 1);
v_isSharedCheck_530_ = !lean_is_exclusive(v___x_518_);
if (v_isSharedCheck_530_ == 0)
{
lean_object* v_unused_531_; 
v_unused_531_ = lean_ctor_get(v___x_518_, 0);
lean_dec(v_unused_531_);
v___x_521_ = v___x_518_;
v_isShared_522_ = v_isSharedCheck_530_;
goto v_resetjp_520_;
}
else
{
lean_inc(v_toLT_519_);
lean_dec(v___x_518_);
v___x_521_ = lean_box(0);
v_isShared_522_ = v_isSharedCheck_530_;
goto v_resetjp_520_;
}
v_resetjp_520_:
{
lean_object* v___f_523_; lean_object* v___x_525_; 
v___f_523_ = lean_alloc_closure((void*)(lp_mathlib_BotHom_instSemilatticeSup___redArg___lam__0), 4, 1);
lean_closure_set(v___f_523_, 0, v_sup_513_);
if (v_isShared_522_ == 0)
{
lean_ctor_set(v___x_521_, 0, v___x_517_);
v___x_525_ = v___x_521_;
goto v_reusejp_524_;
}
else
{
lean_object* v_reuseFailAlloc_529_; 
v_reuseFailAlloc_529_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_529_, 0, v___x_517_);
lean_ctor_set(v_reuseFailAlloc_529_, 1, v_toLT_519_);
v___x_525_ = v_reuseFailAlloc_529_;
goto v_reusejp_524_;
}
v_reusejp_524_:
{
lean_object* v___x_527_; 
if (v_isShared_516_ == 0)
{
lean_ctor_set(v___x_515_, 1, v___f_523_);
lean_ctor_set(v___x_515_, 0, v___x_525_);
v___x_527_ = v___x_515_;
goto v_reusejp_526_;
}
else
{
lean_object* v_reuseFailAlloc_528_; 
v_reuseFailAlloc_528_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_528_, 0, v___x_525_);
lean_ctor_set(v_reuseFailAlloc_528_, 1, v___f_523_);
v___x_527_ = v_reuseFailAlloc_528_;
goto v_reusejp_526_;
}
v_reusejp_526_:
{
return v___x_527_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instSemilatticeSup___redArg___boxed(lean_object* v_inst_533_, lean_object* v_inst_534_, lean_object* v_inst_535_){
_start:
{
lean_object* v_res_536_; 
v_res_536_ = lp_mathlib_TopHom_instSemilatticeSup___redArg(v_inst_533_, v_inst_534_, v_inst_535_);
lean_dec(v_inst_535_);
lean_dec(v_inst_533_);
return v_res_536_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instSemilatticeSup(lean_object* v_00_u03b1_537_, lean_object* v_00_u03b2_538_, lean_object* v_inst_539_, lean_object* v_inst_540_, lean_object* v_inst_541_){
_start:
{
lean_object* v___x_542_; 
v___x_542_ = lp_mathlib_TopHom_instSemilatticeSup___redArg(v_inst_539_, v_inst_540_, v_inst_541_);
return v___x_542_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instSemilatticeSup___boxed(lean_object* v_00_u03b1_543_, lean_object* v_00_u03b2_544_, lean_object* v_inst_545_, lean_object* v_inst_546_, lean_object* v_inst_547_){
_start:
{
lean_object* v_res_548_; 
v_res_548_ = lp_mathlib_TopHom_instSemilatticeSup(v_00_u03b1_543_, v_00_u03b2_544_, v_inst_545_, v_inst_546_, v_inst_547_);
lean_dec(v_inst_547_);
lean_dec(v_inst_545_);
return v_res_548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instSemilatticeInf___redArg(lean_object* v_inst_549_, lean_object* v_inst_550_, lean_object* v_inst_551_){
_start:
{
lean_object* v_toPartialOrder_552_; lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v_toLT_555_; lean_object* v___f_556_; lean_object* v___x_557_; 
v_toPartialOrder_552_ = lean_ctor_get(v_inst_550_, 0);
v___x_553_ = lean_box(0);
v___x_554_ = lp_mathlib_BotHom_instPreorder(lean_box(0), lean_box(0), v_inst_549_, v_toPartialOrder_552_, v_inst_551_);
v_toLT_555_ = lean_ctor_get(v___x_554_, 1);
lean_inc(v_toLT_555_);
lean_dec_ref(v___x_554_);
v___f_556_ = lean_alloc_closure((void*)(lp_mathlib_TopHom_instMin___redArg___lam__0), 4, 1);
lean_closure_set(v___f_556_, 0, v_inst_550_);
v___x_557_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v___f_556_, v___x_553_, v_toLT_555_);
return v___x_557_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instSemilatticeInf___redArg___boxed(lean_object* v_inst_558_, lean_object* v_inst_559_, lean_object* v_inst_560_){
_start:
{
lean_object* v_res_561_; 
v_res_561_ = lp_mathlib_BotHom_instSemilatticeInf___redArg(v_inst_558_, v_inst_559_, v_inst_560_);
lean_dec(v_inst_560_);
lean_dec(v_inst_558_);
return v_res_561_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instSemilatticeInf(lean_object* v_00_u03b1_562_, lean_object* v_00_u03b2_563_, lean_object* v_inst_564_, lean_object* v_inst_565_, lean_object* v_inst_566_){
_start:
{
lean_object* v___x_567_; 
v___x_567_ = lp_mathlib_BotHom_instSemilatticeInf___redArg(v_inst_564_, v_inst_565_, v_inst_566_);
return v___x_567_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instSemilatticeInf___boxed(lean_object* v_00_u03b1_568_, lean_object* v_00_u03b2_569_, lean_object* v_inst_570_, lean_object* v_inst_571_, lean_object* v_inst_572_){
_start:
{
lean_object* v_res_573_; 
v_res_573_ = lp_mathlib_BotHom_instSemilatticeInf(v_00_u03b1_568_, v_00_u03b2_569_, v_inst_570_, v_inst_571_, v_inst_572_);
lean_dec(v_inst_572_);
lean_dec(v_inst_570_);
return v_res_573_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instLattice___redArg___lam__0(lean_object* v_inf_574_, lean_object* v_a_575_, lean_object* v_b_576_, lean_object* v___y_577_){
_start:
{
lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; 
lean_inc(v___y_577_);
v___x_578_ = lean_apply_1(v_a_575_, v___y_577_);
v___x_579_ = lean_apply_1(v_b_576_, v___y_577_);
v___x_580_ = lean_apply_2(v_inf_574_, v___x_578_, v___x_579_);
return v___x_580_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instLattice___redArg___lam__1(lean_object* v_toSemilatticeSup_581_, lean_object* v_a_582_, lean_object* v_b_583_, lean_object* v___y_584_){
_start:
{
lean_object* v_sup_585_; lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v___x_588_; 
v_sup_585_ = lean_ctor_get(v_toSemilatticeSup_581_, 1);
lean_inc(v_sup_585_);
lean_dec_ref(v_toSemilatticeSup_581_);
lean_inc(v___y_584_);
v___x_586_ = lean_apply_1(v_a_582_, v___y_584_);
v___x_587_ = lean_apply_1(v_b_583_, v___y_584_);
v___x_588_ = lean_apply_2(v_sup_585_, v___x_586_, v___x_587_);
return v___x_588_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instLattice___redArg(lean_object* v_inst_589_, lean_object* v_inst_590_, lean_object* v_inst_591_){
_start:
{
lean_object* v_toSemilatticeSup_592_; lean_object* v_inf_593_; lean_object* v___x_594_; lean_object* v_toPartialOrder_595_; lean_object* v___x_597_; uint8_t v_isShared_598_; uint8_t v_isSharedCheck_616_; 
v_toSemilatticeSup_592_ = lean_ctor_get(v_inst_590_, 0);
lean_inc_ref(v_toSemilatticeSup_592_);
v_inf_593_ = lean_ctor_get(v_inst_590_, 1);
lean_inc(v_inf_593_);
v___x_594_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_inst_590_);
v_toPartialOrder_595_ = lean_ctor_get(v___x_594_, 0);
v_isSharedCheck_616_ = !lean_is_exclusive(v___x_594_);
if (v_isSharedCheck_616_ == 0)
{
lean_object* v_unused_617_; 
v_unused_617_ = lean_ctor_get(v___x_594_, 1);
lean_dec(v_unused_617_);
v___x_597_ = v___x_594_;
v_isShared_598_ = v_isSharedCheck_616_;
goto v_resetjp_596_;
}
else
{
lean_inc(v_toPartialOrder_595_);
lean_dec(v___x_594_);
v___x_597_ = lean_box(0);
v_isShared_598_ = v_isSharedCheck_616_;
goto v_resetjp_596_;
}
v_resetjp_596_:
{
lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v_toLT_601_; lean_object* v___x_603_; uint8_t v_isShared_604_; uint8_t v_isSharedCheck_614_; 
v___x_599_ = lean_box(0);
v___x_600_ = lp_mathlib_TopHom_instPreorder(lean_box(0), lean_box(0), v_inst_589_, v_toPartialOrder_595_, v_inst_591_);
lean_dec_ref(v_toPartialOrder_595_);
v_toLT_601_ = lean_ctor_get(v___x_600_, 1);
v_isSharedCheck_614_ = !lean_is_exclusive(v___x_600_);
if (v_isSharedCheck_614_ == 0)
{
lean_object* v_unused_615_; 
v_unused_615_ = lean_ctor_get(v___x_600_, 0);
lean_dec(v_unused_615_);
v___x_603_ = v___x_600_;
v_isShared_604_ = v_isSharedCheck_614_;
goto v_resetjp_602_;
}
else
{
lean_inc(v_toLT_601_);
lean_dec(v___x_600_);
v___x_603_ = lean_box(0);
v_isShared_604_ = v_isSharedCheck_614_;
goto v_resetjp_602_;
}
v_resetjp_602_:
{
lean_object* v___f_605_; lean_object* v___f_606_; lean_object* v___x_608_; 
v___f_605_ = lean_alloc_closure((void*)(lp_mathlib_TopHom_instLattice___redArg___lam__0), 4, 1);
lean_closure_set(v___f_605_, 0, v_inf_593_);
v___f_606_ = lean_alloc_closure((void*)(lp_mathlib_TopHom_instLattice___redArg___lam__1), 4, 1);
lean_closure_set(v___f_606_, 0, v_toSemilatticeSup_592_);
if (v_isShared_604_ == 0)
{
lean_ctor_set(v___x_603_, 0, v___x_599_);
v___x_608_ = v___x_603_;
goto v_reusejp_607_;
}
else
{
lean_object* v_reuseFailAlloc_613_; 
v_reuseFailAlloc_613_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_613_, 0, v___x_599_);
lean_ctor_set(v_reuseFailAlloc_613_, 1, v_toLT_601_);
v___x_608_ = v_reuseFailAlloc_613_;
goto v_reusejp_607_;
}
v_reusejp_607_:
{
lean_object* v___x_610_; 
if (v_isShared_598_ == 0)
{
lean_ctor_set(v___x_597_, 1, v___f_606_);
lean_ctor_set(v___x_597_, 0, v___x_608_);
v___x_610_ = v___x_597_;
goto v_reusejp_609_;
}
else
{
lean_object* v_reuseFailAlloc_612_; 
v_reuseFailAlloc_612_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_612_, 0, v___x_608_);
lean_ctor_set(v_reuseFailAlloc_612_, 1, v___f_606_);
v___x_610_ = v_reuseFailAlloc_612_;
goto v_reusejp_609_;
}
v_reusejp_609_:
{
lean_object* v___x_611_; 
v___x_611_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_611_, 0, v___x_610_);
lean_ctor_set(v___x_611_, 1, v___f_605_);
return v___x_611_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instLattice___redArg___boxed(lean_object* v_inst_618_, lean_object* v_inst_619_, lean_object* v_inst_620_){
_start:
{
lean_object* v_res_621_; 
v_res_621_ = lp_mathlib_TopHom_instLattice___redArg(v_inst_618_, v_inst_619_, v_inst_620_);
lean_dec(v_inst_620_);
lean_dec(v_inst_618_);
return v_res_621_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instLattice(lean_object* v_00_u03b1_622_, lean_object* v_00_u03b2_623_, lean_object* v_inst_624_, lean_object* v_inst_625_, lean_object* v_inst_626_){
_start:
{
lean_object* v___x_627_; 
v___x_627_ = lp_mathlib_TopHom_instLattice___redArg(v_inst_624_, v_inst_625_, v_inst_626_);
return v___x_627_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instLattice___boxed(lean_object* v_00_u03b1_628_, lean_object* v_00_u03b2_629_, lean_object* v_inst_630_, lean_object* v_inst_631_, lean_object* v_inst_632_){
_start:
{
lean_object* v_res_633_; 
v_res_633_ = lp_mathlib_TopHom_instLattice(v_00_u03b1_628_, v_00_u03b2_629_, v_inst_630_, v_inst_631_, v_inst_632_);
lean_dec(v_inst_632_);
lean_dec(v_inst_630_);
return v_res_633_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instLattice___redArg(lean_object* v_inst_634_, lean_object* v_inst_635_, lean_object* v_inst_636_){
_start:
{
lean_object* v_toSemilatticeSup_637_; lean_object* v_inf_638_; lean_object* v___x_640_; uint8_t v_isShared_641_; uint8_t v_isSharedCheck_667_; 
v_toSemilatticeSup_637_ = lean_ctor_get(v_inst_635_, 0);
v_inf_638_ = lean_ctor_get(v_inst_635_, 1);
v_isSharedCheck_667_ = !lean_is_exclusive(v_inst_635_);
if (v_isSharedCheck_667_ == 0)
{
v___x_640_ = v_inst_635_;
v_isShared_641_ = v_isSharedCheck_667_;
goto v_resetjp_639_;
}
else
{
lean_inc(v_inf_638_);
lean_inc(v_toSemilatticeSup_637_);
lean_dec(v_inst_635_);
v___x_640_ = lean_box(0);
v_isShared_641_ = v_isSharedCheck_667_;
goto v_resetjp_639_;
}
v_resetjp_639_:
{
lean_object* v_toPartialOrder_642_; lean_object* v_sup_643_; lean_object* v___x_645_; uint8_t v_isShared_646_; uint8_t v_isSharedCheck_666_; 
v_toPartialOrder_642_ = lean_ctor_get(v_toSemilatticeSup_637_, 0);
v_sup_643_ = lean_ctor_get(v_toSemilatticeSup_637_, 1);
v_isSharedCheck_666_ = !lean_is_exclusive(v_toSemilatticeSup_637_);
if (v_isSharedCheck_666_ == 0)
{
v___x_645_ = v_toSemilatticeSup_637_;
v_isShared_646_ = v_isSharedCheck_666_;
goto v_resetjp_644_;
}
else
{
lean_inc(v_sup_643_);
lean_inc(v_toPartialOrder_642_);
lean_dec(v_toSemilatticeSup_637_);
v___x_645_ = lean_box(0);
v_isShared_646_ = v_isSharedCheck_666_;
goto v_resetjp_644_;
}
v_resetjp_644_:
{
lean_object* v___x_647_; lean_object* v___x_648_; lean_object* v_toLT_649_; lean_object* v___x_651_; uint8_t v_isShared_652_; uint8_t v_isSharedCheck_664_; 
v___x_647_ = lean_box(0);
v___x_648_ = lp_mathlib_BotHom_instPreorder(lean_box(0), lean_box(0), v_inst_634_, v_toPartialOrder_642_, v_inst_636_);
lean_dec_ref(v_toPartialOrder_642_);
v_toLT_649_ = lean_ctor_get(v___x_648_, 1);
v_isSharedCheck_664_ = !lean_is_exclusive(v___x_648_);
if (v_isSharedCheck_664_ == 0)
{
lean_object* v_unused_665_; 
v_unused_665_ = lean_ctor_get(v___x_648_, 0);
lean_dec(v_unused_665_);
v___x_651_ = v___x_648_;
v_isShared_652_ = v_isSharedCheck_664_;
goto v_resetjp_650_;
}
else
{
lean_inc(v_toLT_649_);
lean_dec(v___x_648_);
v___x_651_ = lean_box(0);
v_isShared_652_ = v_isSharedCheck_664_;
goto v_resetjp_650_;
}
v_resetjp_650_:
{
lean_object* v___f_653_; lean_object* v___f_654_; lean_object* v___x_656_; 
v___f_653_ = lean_alloc_closure((void*)(lp_mathlib_TopHom_instLattice___redArg___lam__0), 4, 1);
lean_closure_set(v___f_653_, 0, v_inf_638_);
v___f_654_ = lean_alloc_closure((void*)(lp_mathlib_BotHom_instSemilatticeSup___redArg___lam__0), 4, 1);
lean_closure_set(v___f_654_, 0, v_sup_643_);
if (v_isShared_652_ == 0)
{
lean_ctor_set(v___x_651_, 0, v___x_647_);
v___x_656_ = v___x_651_;
goto v_reusejp_655_;
}
else
{
lean_object* v_reuseFailAlloc_663_; 
v_reuseFailAlloc_663_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_663_, 0, v___x_647_);
lean_ctor_set(v_reuseFailAlloc_663_, 1, v_toLT_649_);
v___x_656_ = v_reuseFailAlloc_663_;
goto v_reusejp_655_;
}
v_reusejp_655_:
{
lean_object* v___x_658_; 
if (v_isShared_646_ == 0)
{
lean_ctor_set(v___x_645_, 1, v___f_654_);
lean_ctor_set(v___x_645_, 0, v___x_656_);
v___x_658_ = v___x_645_;
goto v_reusejp_657_;
}
else
{
lean_object* v_reuseFailAlloc_662_; 
v_reuseFailAlloc_662_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_662_, 0, v___x_656_);
lean_ctor_set(v_reuseFailAlloc_662_, 1, v___f_654_);
v___x_658_ = v_reuseFailAlloc_662_;
goto v_reusejp_657_;
}
v_reusejp_657_:
{
lean_object* v___x_660_; 
if (v_isShared_641_ == 0)
{
lean_ctor_set(v___x_640_, 1, v___f_653_);
lean_ctor_set(v___x_640_, 0, v___x_658_);
v___x_660_ = v___x_640_;
goto v_reusejp_659_;
}
else
{
lean_object* v_reuseFailAlloc_661_; 
v_reuseFailAlloc_661_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_661_, 0, v___x_658_);
lean_ctor_set(v_reuseFailAlloc_661_, 1, v___f_653_);
v___x_660_ = v_reuseFailAlloc_661_;
goto v_reusejp_659_;
}
v_reusejp_659_:
{
return v___x_660_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instLattice___redArg___boxed(lean_object* v_inst_668_, lean_object* v_inst_669_, lean_object* v_inst_670_){
_start:
{
lean_object* v_res_671_; 
v_res_671_ = lp_mathlib_BotHom_instLattice___redArg(v_inst_668_, v_inst_669_, v_inst_670_);
lean_dec(v_inst_670_);
lean_dec(v_inst_668_);
return v_res_671_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instLattice(lean_object* v_00_u03b1_672_, lean_object* v_00_u03b2_673_, lean_object* v_inst_674_, lean_object* v_inst_675_, lean_object* v_inst_676_){
_start:
{
lean_object* v___x_677_; 
v___x_677_ = lp_mathlib_BotHom_instLattice___redArg(v_inst_674_, v_inst_675_, v_inst_676_);
return v___x_677_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instLattice___boxed(lean_object* v_00_u03b1_678_, lean_object* v_00_u03b2_679_, lean_object* v_inst_680_, lean_object* v_inst_681_, lean_object* v_inst_682_){
_start:
{
lean_object* v_res_683_; 
v_res_683_ = lp_mathlib_BotHom_instLattice(v_00_u03b1_678_, v_00_u03b2_679_, v_inst_680_, v_inst_681_, v_inst_682_);
lean_dec(v_inst_682_);
lean_dec(v_inst_680_);
return v_res_683_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instDistribLattice___redArg(lean_object* v_inst_684_, lean_object* v_inst_685_, lean_object* v_inst_686_){
_start:
{
lean_object* v_toSemilatticeSup_687_; lean_object* v_inf_688_; lean_object* v___x_689_; lean_object* v_toPartialOrder_690_; lean_object* v___x_692_; uint8_t v_isShared_693_; uint8_t v_isSharedCheck_711_; 
v_toSemilatticeSup_687_ = lean_ctor_get(v_inst_685_, 0);
lean_inc_ref(v_toSemilatticeSup_687_);
v_inf_688_ = lean_ctor_get(v_inst_685_, 1);
lean_inc(v_inf_688_);
v___x_689_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_inst_685_);
v_toPartialOrder_690_ = lean_ctor_get(v___x_689_, 0);
v_isSharedCheck_711_ = !lean_is_exclusive(v___x_689_);
if (v_isSharedCheck_711_ == 0)
{
lean_object* v_unused_712_; 
v_unused_712_ = lean_ctor_get(v___x_689_, 1);
lean_dec(v_unused_712_);
v___x_692_ = v___x_689_;
v_isShared_693_ = v_isSharedCheck_711_;
goto v_resetjp_691_;
}
else
{
lean_inc(v_toPartialOrder_690_);
lean_dec(v___x_689_);
v___x_692_ = lean_box(0);
v_isShared_693_ = v_isSharedCheck_711_;
goto v_resetjp_691_;
}
v_resetjp_691_:
{
lean_object* v___x_694_; lean_object* v___x_695_; lean_object* v_toLT_696_; lean_object* v___x_698_; uint8_t v_isShared_699_; uint8_t v_isSharedCheck_709_; 
v___x_694_ = lean_box(0);
v___x_695_ = lp_mathlib_TopHom_instPreorder(lean_box(0), lean_box(0), v_inst_684_, v_toPartialOrder_690_, v_inst_686_);
lean_dec_ref(v_toPartialOrder_690_);
v_toLT_696_ = lean_ctor_get(v___x_695_, 1);
v_isSharedCheck_709_ = !lean_is_exclusive(v___x_695_);
if (v_isSharedCheck_709_ == 0)
{
lean_object* v_unused_710_; 
v_unused_710_ = lean_ctor_get(v___x_695_, 0);
lean_dec(v_unused_710_);
v___x_698_ = v___x_695_;
v_isShared_699_ = v_isSharedCheck_709_;
goto v_resetjp_697_;
}
else
{
lean_inc(v_toLT_696_);
lean_dec(v___x_695_);
v___x_698_ = lean_box(0);
v_isShared_699_ = v_isSharedCheck_709_;
goto v_resetjp_697_;
}
v_resetjp_697_:
{
lean_object* v___f_700_; lean_object* v___f_701_; lean_object* v___x_703_; 
v___f_700_ = lean_alloc_closure((void*)(lp_mathlib_TopHom_instLattice___redArg___lam__0), 4, 1);
lean_closure_set(v___f_700_, 0, v_inf_688_);
v___f_701_ = lean_alloc_closure((void*)(lp_mathlib_TopHom_instLattice___redArg___lam__1), 4, 1);
lean_closure_set(v___f_701_, 0, v_toSemilatticeSup_687_);
if (v_isShared_699_ == 0)
{
lean_ctor_set(v___x_698_, 0, v___x_694_);
v___x_703_ = v___x_698_;
goto v_reusejp_702_;
}
else
{
lean_object* v_reuseFailAlloc_708_; 
v_reuseFailAlloc_708_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_708_, 0, v___x_694_);
lean_ctor_set(v_reuseFailAlloc_708_, 1, v_toLT_696_);
v___x_703_ = v_reuseFailAlloc_708_;
goto v_reusejp_702_;
}
v_reusejp_702_:
{
lean_object* v___x_705_; 
if (v_isShared_693_ == 0)
{
lean_ctor_set(v___x_692_, 1, v___f_701_);
lean_ctor_set(v___x_692_, 0, v___x_703_);
v___x_705_ = v___x_692_;
goto v_reusejp_704_;
}
else
{
lean_object* v_reuseFailAlloc_707_; 
v_reuseFailAlloc_707_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_707_, 0, v___x_703_);
lean_ctor_set(v_reuseFailAlloc_707_, 1, v___f_701_);
v___x_705_ = v_reuseFailAlloc_707_;
goto v_reusejp_704_;
}
v_reusejp_704_:
{
lean_object* v___x_706_; 
v___x_706_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_706_, 0, v___x_705_);
lean_ctor_set(v___x_706_, 1, v___f_700_);
return v___x_706_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instDistribLattice___redArg___boxed(lean_object* v_inst_713_, lean_object* v_inst_714_, lean_object* v_inst_715_){
_start:
{
lean_object* v_res_716_; 
v_res_716_ = lp_mathlib_TopHom_instDistribLattice___redArg(v_inst_713_, v_inst_714_, v_inst_715_);
lean_dec(v_inst_715_);
lean_dec(v_inst_713_);
return v_res_716_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instDistribLattice(lean_object* v_00_u03b1_717_, lean_object* v_00_u03b2_718_, lean_object* v_inst_719_, lean_object* v_inst_720_, lean_object* v_inst_721_){
_start:
{
lean_object* v___x_722_; 
v___x_722_ = lp_mathlib_TopHom_instDistribLattice___redArg(v_inst_719_, v_inst_720_, v_inst_721_);
return v___x_722_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_instDistribLattice___boxed(lean_object* v_00_u03b1_723_, lean_object* v_00_u03b2_724_, lean_object* v_inst_725_, lean_object* v_inst_726_, lean_object* v_inst_727_){
_start:
{
lean_object* v_res_728_; 
v_res_728_ = lp_mathlib_TopHom_instDistribLattice(v_00_u03b1_723_, v_00_u03b2_724_, v_inst_725_, v_inst_726_, v_inst_727_);
lean_dec(v_inst_727_);
lean_dec(v_inst_725_);
return v_res_728_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instDistribLattice___redArg(lean_object* v_inst_729_, lean_object* v_inst_730_, lean_object* v_inst_731_){
_start:
{
lean_object* v_toSemilatticeSup_732_; lean_object* v_inf_733_; lean_object* v___x_735_; uint8_t v_isShared_736_; uint8_t v_isSharedCheck_762_; 
v_toSemilatticeSup_732_ = lean_ctor_get(v_inst_730_, 0);
v_inf_733_ = lean_ctor_get(v_inst_730_, 1);
v_isSharedCheck_762_ = !lean_is_exclusive(v_inst_730_);
if (v_isSharedCheck_762_ == 0)
{
v___x_735_ = v_inst_730_;
v_isShared_736_ = v_isSharedCheck_762_;
goto v_resetjp_734_;
}
else
{
lean_inc(v_inf_733_);
lean_inc(v_toSemilatticeSup_732_);
lean_dec(v_inst_730_);
v___x_735_ = lean_box(0);
v_isShared_736_ = v_isSharedCheck_762_;
goto v_resetjp_734_;
}
v_resetjp_734_:
{
lean_object* v_toPartialOrder_737_; lean_object* v_sup_738_; lean_object* v___x_740_; uint8_t v_isShared_741_; uint8_t v_isSharedCheck_761_; 
v_toPartialOrder_737_ = lean_ctor_get(v_toSemilatticeSup_732_, 0);
v_sup_738_ = lean_ctor_get(v_toSemilatticeSup_732_, 1);
v_isSharedCheck_761_ = !lean_is_exclusive(v_toSemilatticeSup_732_);
if (v_isSharedCheck_761_ == 0)
{
v___x_740_ = v_toSemilatticeSup_732_;
v_isShared_741_ = v_isSharedCheck_761_;
goto v_resetjp_739_;
}
else
{
lean_inc(v_sup_738_);
lean_inc(v_toPartialOrder_737_);
lean_dec(v_toSemilatticeSup_732_);
v___x_740_ = lean_box(0);
v_isShared_741_ = v_isSharedCheck_761_;
goto v_resetjp_739_;
}
v_resetjp_739_:
{
lean_object* v___x_742_; lean_object* v___x_743_; lean_object* v_toLT_744_; lean_object* v___x_746_; uint8_t v_isShared_747_; uint8_t v_isSharedCheck_759_; 
v___x_742_ = lean_box(0);
v___x_743_ = lp_mathlib_BotHom_instPreorder(lean_box(0), lean_box(0), v_inst_729_, v_toPartialOrder_737_, v_inst_731_);
lean_dec_ref(v_toPartialOrder_737_);
v_toLT_744_ = lean_ctor_get(v___x_743_, 1);
v_isSharedCheck_759_ = !lean_is_exclusive(v___x_743_);
if (v_isSharedCheck_759_ == 0)
{
lean_object* v_unused_760_; 
v_unused_760_ = lean_ctor_get(v___x_743_, 0);
lean_dec(v_unused_760_);
v___x_746_ = v___x_743_;
v_isShared_747_ = v_isSharedCheck_759_;
goto v_resetjp_745_;
}
else
{
lean_inc(v_toLT_744_);
lean_dec(v___x_743_);
v___x_746_ = lean_box(0);
v_isShared_747_ = v_isSharedCheck_759_;
goto v_resetjp_745_;
}
v_resetjp_745_:
{
lean_object* v___f_748_; lean_object* v___f_749_; lean_object* v___x_751_; 
v___f_748_ = lean_alloc_closure((void*)(lp_mathlib_TopHom_instLattice___redArg___lam__0), 4, 1);
lean_closure_set(v___f_748_, 0, v_inf_733_);
v___f_749_ = lean_alloc_closure((void*)(lp_mathlib_BotHom_instSemilatticeSup___redArg___lam__0), 4, 1);
lean_closure_set(v___f_749_, 0, v_sup_738_);
if (v_isShared_747_ == 0)
{
lean_ctor_set(v___x_746_, 0, v___x_742_);
v___x_751_ = v___x_746_;
goto v_reusejp_750_;
}
else
{
lean_object* v_reuseFailAlloc_758_; 
v_reuseFailAlloc_758_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_758_, 0, v___x_742_);
lean_ctor_set(v_reuseFailAlloc_758_, 1, v_toLT_744_);
v___x_751_ = v_reuseFailAlloc_758_;
goto v_reusejp_750_;
}
v_reusejp_750_:
{
lean_object* v___x_753_; 
if (v_isShared_741_ == 0)
{
lean_ctor_set(v___x_740_, 1, v___f_749_);
lean_ctor_set(v___x_740_, 0, v___x_751_);
v___x_753_ = v___x_740_;
goto v_reusejp_752_;
}
else
{
lean_object* v_reuseFailAlloc_757_; 
v_reuseFailAlloc_757_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_757_, 0, v___x_751_);
lean_ctor_set(v_reuseFailAlloc_757_, 1, v___f_749_);
v___x_753_ = v_reuseFailAlloc_757_;
goto v_reusejp_752_;
}
v_reusejp_752_:
{
lean_object* v___x_755_; 
if (v_isShared_736_ == 0)
{
lean_ctor_set(v___x_735_, 1, v___f_748_);
lean_ctor_set(v___x_735_, 0, v___x_753_);
v___x_755_ = v___x_735_;
goto v_reusejp_754_;
}
else
{
lean_object* v_reuseFailAlloc_756_; 
v_reuseFailAlloc_756_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_756_, 0, v___x_753_);
lean_ctor_set(v_reuseFailAlloc_756_, 1, v___f_748_);
v___x_755_ = v_reuseFailAlloc_756_;
goto v_reusejp_754_;
}
v_reusejp_754_:
{
return v___x_755_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instDistribLattice___redArg___boxed(lean_object* v_inst_763_, lean_object* v_inst_764_, lean_object* v_inst_765_){
_start:
{
lean_object* v_res_766_; 
v_res_766_ = lp_mathlib_BotHom_instDistribLattice___redArg(v_inst_763_, v_inst_764_, v_inst_765_);
lean_dec(v_inst_765_);
lean_dec(v_inst_763_);
return v_res_766_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instDistribLattice(lean_object* v_00_u03b1_767_, lean_object* v_00_u03b2_768_, lean_object* v_inst_769_, lean_object* v_inst_770_, lean_object* v_inst_771_){
_start:
{
lean_object* v___x_772_; 
v___x_772_ = lp_mathlib_BotHom_instDistribLattice___redArg(v_inst_769_, v_inst_770_, v_inst_771_);
return v___x_772_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_instDistribLattice___boxed(lean_object* v_00_u03b1_773_, lean_object* v_00_u03b2_774_, lean_object* v_inst_775_, lean_object* v_inst_776_, lean_object* v_inst_777_){
_start:
{
lean_object* v_res_778_; 
v_res_778_ = lp_mathlib_BotHom_instDistribLattice(v_00_u03b1_773_, v_00_u03b2_774_, v_inst_775_, v_inst_776_, v_inst_777_);
lean_dec(v_inst_777_);
lean_dec(v_inst_775_);
return v_res_778_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_toTopHom___redArg(lean_object* v_f_779_){
_start:
{
lean_inc(v_f_779_);
return v_f_779_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_toTopHom___redArg___boxed(lean_object* v_f_780_){
_start:
{
lean_object* v_res_781_; 
v_res_781_ = lp_mathlib_BoundedOrderHom_toTopHom___redArg(v_f_780_);
lean_dec(v_f_780_);
return v_res_781_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_toTopHom(lean_object* v_00_u03b1_782_, lean_object* v_00_u03b2_783_, lean_object* v_inst_784_, lean_object* v_inst_785_, lean_object* v_inst_786_, lean_object* v_inst_787_, lean_object* v_f_788_){
_start:
{
lean_inc(v_f_788_);
return v_f_788_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_toTopHom___boxed(lean_object* v_00_u03b1_789_, lean_object* v_00_u03b2_790_, lean_object* v_inst_791_, lean_object* v_inst_792_, lean_object* v_inst_793_, lean_object* v_inst_794_, lean_object* v_f_795_){
_start:
{
lean_object* v_res_796_; 
v_res_796_ = lp_mathlib_BoundedOrderHom_toTopHom(v_00_u03b1_789_, v_00_u03b2_790_, v_inst_791_, v_inst_792_, v_inst_793_, v_inst_794_, v_f_795_);
lean_dec(v_f_795_);
lean_dec_ref(v_inst_794_);
lean_dec_ref(v_inst_793_);
lean_dec_ref(v_inst_792_);
lean_dec_ref(v_inst_791_);
return v_res_796_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_toBotHom___redArg(lean_object* v_f_797_){
_start:
{
lean_inc(v_f_797_);
return v_f_797_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_toBotHom___redArg___boxed(lean_object* v_f_798_){
_start:
{
lean_object* v_res_799_; 
v_res_799_ = lp_mathlib_BoundedOrderHom_toBotHom___redArg(v_f_798_);
lean_dec(v_f_798_);
return v_res_799_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_toBotHom(lean_object* v_00_u03b1_800_, lean_object* v_00_u03b2_801_, lean_object* v_inst_802_, lean_object* v_inst_803_, lean_object* v_inst_804_, lean_object* v_inst_805_, lean_object* v_f_806_){
_start:
{
lean_inc(v_f_806_);
return v_f_806_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_toBotHom___boxed(lean_object* v_00_u03b1_807_, lean_object* v_00_u03b2_808_, lean_object* v_inst_809_, lean_object* v_inst_810_, lean_object* v_inst_811_, lean_object* v_inst_812_, lean_object* v_f_813_){
_start:
{
lean_object* v_res_814_; 
v_res_814_ = lp_mathlib_BoundedOrderHom_toBotHom(v_00_u03b1_807_, v_00_u03b2_808_, v_inst_809_, v_inst_810_, v_inst_811_, v_inst_812_, v_f_813_);
lean_dec(v_f_813_);
lean_dec_ref(v_inst_812_);
lean_dec_ref(v_inst_811_);
lean_dec_ref(v_inst_810_);
lean_dec_ref(v_inst_809_);
return v_res_814_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_copy___redArg(lean_object* v_f_x27_815_){
_start:
{
lean_inc(v_f_x27_815_);
return v_f_x27_815_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_copy___redArg___boxed(lean_object* v_f_x27_816_){
_start:
{
lean_object* v_res_817_; 
v_res_817_ = lp_mathlib_BoundedOrderHom_copy___redArg(v_f_x27_816_);
lean_dec(v_f_x27_816_);
return v_res_817_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_copy(lean_object* v_00_u03b1_818_, lean_object* v_00_u03b2_819_, lean_object* v_inst_820_, lean_object* v_inst_821_, lean_object* v_inst_822_, lean_object* v_inst_823_, lean_object* v_f_824_, lean_object* v_f_x27_825_, lean_object* v_h_826_){
_start:
{
lean_inc(v_f_x27_825_);
return v_f_x27_825_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_copy___boxed(lean_object* v_00_u03b1_827_, lean_object* v_00_u03b2_828_, lean_object* v_inst_829_, lean_object* v_inst_830_, lean_object* v_inst_831_, lean_object* v_inst_832_, lean_object* v_f_833_, lean_object* v_f_x27_834_, lean_object* v_h_835_){
_start:
{
lean_object* v_res_836_; 
v_res_836_ = lp_mathlib_BoundedOrderHom_copy(v_00_u03b1_827_, v_00_u03b2_828_, v_inst_829_, v_inst_830_, v_inst_831_, v_inst_832_, v_f_833_, v_f_x27_834_, v_h_835_);
lean_dec(v_f_x27_834_);
lean_dec(v_f_833_);
lean_dec_ref(v_inst_832_);
lean_dec_ref(v_inst_831_);
lean_dec_ref(v_inst_830_);
lean_dec_ref(v_inst_829_);
return v_res_836_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_id(lean_object* v_00_u03b1_837_, lean_object* v_inst_838_, lean_object* v_inst_839_){
_start:
{
lean_object* v___x_840_; 
v___x_840_ = ((lean_object*)(lp_mathlib_TopHom_id___closed__0));
return v___x_840_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_id___boxed(lean_object* v_00_u03b1_841_, lean_object* v_inst_842_, lean_object* v_inst_843_){
_start:
{
lean_object* v_res_844_; 
v_res_844_ = lp_mathlib_BoundedOrderHom_id(v_00_u03b1_841_, v_inst_842_, v_inst_843_);
lean_dec_ref(v_inst_843_);
lean_dec_ref(v_inst_842_);
return v_res_844_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_instInhabited(lean_object* v_00_u03b1_845_, lean_object* v_inst_846_, lean_object* v_inst_847_){
_start:
{
lean_object* v___x_848_; 
v___x_848_ = ((lean_object*)(lp_mathlib_TopHom_id___closed__0));
return v___x_848_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_instInhabited___boxed(lean_object* v_00_u03b1_849_, lean_object* v_inst_850_, lean_object* v_inst_851_){
_start:
{
lean_object* v_res_852_; 
v_res_852_ = lp_mathlib_BoundedOrderHom_instInhabited(v_00_u03b1_849_, v_inst_850_, v_inst_851_);
lean_dec_ref(v_inst_851_);
lean_dec_ref(v_inst_850_);
return v_res_852_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_comp___redArg(lean_object* v_f_853_, lean_object* v_g_854_){
_start:
{
lean_object* v___x_855_; 
v___x_855_ = lp_mathlib_OrderHom_comp___redArg(v_f_853_, v_g_854_);
return v___x_855_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_comp(lean_object* v_00_u03b1_856_, lean_object* v_00_u03b2_857_, lean_object* v_00_u03b3_858_, lean_object* v_inst_859_, lean_object* v_inst_860_, lean_object* v_inst_861_, lean_object* v_inst_862_, lean_object* v_inst_863_, lean_object* v_inst_864_, lean_object* v_f_865_, lean_object* v_g_866_){
_start:
{
lean_object* v___x_867_; 
v___x_867_ = lp_mathlib_OrderHom_comp___redArg(v_f_865_, v_g_866_);
return v___x_867_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_comp___boxed(lean_object* v_00_u03b1_868_, lean_object* v_00_u03b2_869_, lean_object* v_00_u03b3_870_, lean_object* v_inst_871_, lean_object* v_inst_872_, lean_object* v_inst_873_, lean_object* v_inst_874_, lean_object* v_inst_875_, lean_object* v_inst_876_, lean_object* v_f_877_, lean_object* v_g_878_){
_start:
{
lean_object* v_res_879_; 
v_res_879_ = lp_mathlib_BoundedOrderHom_comp(v_00_u03b1_868_, v_00_u03b2_869_, v_00_u03b3_870_, v_inst_871_, v_inst_872_, v_inst_873_, v_inst_874_, v_inst_875_, v_inst_876_, v_f_877_, v_g_878_);
lean_dec_ref(v_inst_876_);
lean_dec_ref(v_inst_875_);
lean_dec_ref(v_inst_874_);
lean_dec_ref(v_inst_873_);
lean_dec_ref(v_inst_872_);
lean_dec_ref(v_inst_871_);
return v_res_879_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_dual(lean_object* v_00_u03b1_883_, lean_object* v_00_u03b2_884_, lean_object* v_inst_885_, lean_object* v_inst_886_, lean_object* v_inst_887_, lean_object* v_inst_888_){
_start:
{
lean_object* v___x_889_; 
v___x_889_ = ((lean_object*)(lp_mathlib_TopHom_dual___closed__1));
return v___x_889_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_TopHom_dual___boxed(lean_object* v_00_u03b1_890_, lean_object* v_00_u03b2_891_, lean_object* v_inst_892_, lean_object* v_inst_893_, lean_object* v_inst_894_, lean_object* v_inst_895_){
_start:
{
lean_object* v_res_896_; 
v_res_896_ = lp_mathlib_TopHom_dual(v_00_u03b1_890_, v_00_u03b2_891_, v_inst_892_, v_inst_893_, v_inst_894_, v_inst_895_);
lean_dec(v_inst_895_);
lean_dec(v_inst_893_);
return v_res_896_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_dual(lean_object* v_00_u03b1_897_, lean_object* v_00_u03b2_898_, lean_object* v_inst_899_, lean_object* v_inst_900_, lean_object* v_inst_901_, lean_object* v_inst_902_){
_start:
{
lean_object* v___x_903_; 
v___x_903_ = ((lean_object*)(lp_mathlib_TopHom_dual___closed__1));
return v___x_903_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BotHom_dual___boxed(lean_object* v_00_u03b1_904_, lean_object* v_00_u03b2_905_, lean_object* v_inst_906_, lean_object* v_inst_907_, lean_object* v_inst_908_, lean_object* v_inst_909_){
_start:
{
lean_object* v_res_910_; 
v_res_910_ = lp_mathlib_BotHom_dual(v_00_u03b1_904_, v_00_u03b2_905_, v_inst_906_, v_inst_907_, v_inst_908_, v_inst_909_);
lean_dec(v_inst_909_);
lean_dec(v_inst_907_);
return v_res_910_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_dual___redArg___lam__0(lean_object* v_inst_911_, lean_object* v_inst_912_, lean_object* v_f_913_, lean_object* v___y_914_){
_start:
{
lean_object* v___x_915_; lean_object* v___x_916_; lean_object* v_toFun_917_; lean_object* v___x_918_; 
v___x_915_ = lp_mathlib_OrderHom_dual(lean_box(0), lean_box(0), v_inst_911_, v_inst_912_);
v___x_916_ = lp_mathlib_Equiv_symm___redArg(v___x_915_);
v_toFun_917_ = lean_ctor_get(v___x_916_, 0);
lean_inc(v_toFun_917_);
lean_dec_ref(v___x_916_);
v___x_918_ = lean_apply_2(v_toFun_917_, v_f_913_, v___y_914_);
return v___x_918_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_dual___redArg___lam__0___boxed(lean_object* v_inst_919_, lean_object* v_inst_920_, lean_object* v_f_921_, lean_object* v___y_922_){
_start:
{
lean_object* v_res_923_; 
v_res_923_ = lp_mathlib_BoundedOrderHom_dual___redArg___lam__0(v_inst_919_, v_inst_920_, v_f_921_, v___y_922_);
lean_dec_ref(v_inst_920_);
lean_dec_ref(v_inst_919_);
return v_res_923_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_dual___redArg___lam__1(lean_object* v_inst_924_, lean_object* v_inst_925_, lean_object* v_f_926_, lean_object* v___y_927_){
_start:
{
lean_object* v___x_928_; lean_object* v_toFun_929_; lean_object* v___x_930_; 
v___x_928_ = lp_mathlib_OrderHom_dual(lean_box(0), lean_box(0), v_inst_924_, v_inst_925_);
v_toFun_929_ = lean_ctor_get(v___x_928_, 0);
lean_inc(v_toFun_929_);
lean_dec_ref(v___x_928_);
v___x_930_ = lean_apply_2(v_toFun_929_, v_f_926_, v___y_927_);
return v___x_930_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_dual___redArg___lam__1___boxed(lean_object* v_inst_931_, lean_object* v_inst_932_, lean_object* v_f_933_, lean_object* v___y_934_){
_start:
{
lean_object* v_res_935_; 
v_res_935_ = lp_mathlib_BoundedOrderHom_dual___redArg___lam__1(v_inst_931_, v_inst_932_, v_f_933_, v___y_934_);
lean_dec_ref(v_inst_932_);
lean_dec_ref(v_inst_931_);
return v_res_935_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_dual___redArg(lean_object* v_inst_936_, lean_object* v_inst_937_){
_start:
{
lean_object* v___f_938_; lean_object* v___f_939_; lean_object* v___x_940_; 
lean_inc_ref(v_inst_937_);
lean_inc_ref(v_inst_936_);
v___f_938_ = lean_alloc_closure((void*)(lp_mathlib_BoundedOrderHom_dual___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_938_, 0, v_inst_936_);
lean_closure_set(v___f_938_, 1, v_inst_937_);
v___f_939_ = lean_alloc_closure((void*)(lp_mathlib_BoundedOrderHom_dual___redArg___lam__1___boxed), 4, 2);
lean_closure_set(v___f_939_, 0, v_inst_936_);
lean_closure_set(v___f_939_, 1, v_inst_937_);
v___x_940_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_940_, 0, v___f_939_);
lean_ctor_set(v___x_940_, 1, v___f_938_);
return v___x_940_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_dual(lean_object* v_00_u03b1_941_, lean_object* v_00_u03b2_942_, lean_object* v_inst_943_, lean_object* v_inst_944_, lean_object* v_inst_945_, lean_object* v_inst_946_){
_start:
{
lean_object* v___x_947_; 
v___x_947_ = lp_mathlib_BoundedOrderHom_dual___redArg(v_inst_943_, v_inst_945_);
return v___x_947_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BoundedOrderHom_dual___boxed(lean_object* v_00_u03b1_948_, lean_object* v_00_u03b2_949_, lean_object* v_inst_950_, lean_object* v_inst_951_, lean_object* v_inst_952_, lean_object* v_inst_953_){
_start:
{
lean_object* v_res_954_; 
v_res_954_ = lp_mathlib_BoundedOrderHom_dual(v_00_u03b1_948_, v_00_u03b2_949_, v_inst_950_, v_inst_951_, v_inst_952_, v_inst_953_);
lean_dec_ref(v_inst_953_);
lean_dec_ref(v_inst_951_);
return v_res_954_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_Bounded(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Hom_Bounded(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Hom_Bounded(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_Bounded(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Hom_Bounded(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Hom_Bounded(builtin);
}
#ifdef __cplusplus
}
#endif
