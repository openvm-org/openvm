// Lean compiler output
// Module: Mathlib.Order.Hom.Lattice
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
lean_object* lp_mathlib_Function_eval(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
lean_object* lp_mathlib_SemilatticeInf_toMin___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SemilatticeSup_toMax___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_semilatticeInf___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_toInfHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_toInfHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_toInfHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_toInfHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_orderEmbeddingOfInjective___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_orderEmbeddingOfInjective(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_orderEmbeddingOfInjective___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSupHomOfSupHomClass___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSupHomOfSupHomClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSupHomOfSupHomClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSupHomOfSupHomClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCInfHomOfInfHomClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCInfHomOfInfHomClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCInfHomOfInfHomClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCLatticeHomOfLatticeHomClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCLatticeHomOfLatticeHomClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCLatticeHomOfLatticeHomClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instFunLike___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_SupHom_instFunLike___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SupHom_instFunLike___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_SupHom_instFunLike___closed__0 = (const lean_object*)&lp_mathlib_SupHom_instFunLike___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instFunLike(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instFunLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instFunLike(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instFunLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_SupHom_id___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_SupHom_id___closed__0 = (const lean_object*)&lp_mathlib_SupHom_id___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_SupHom_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_comp___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_comp___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_const___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_const___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_const___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_const(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_const___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_const___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_const(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_const___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instMax___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instMax___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instMax(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instMax___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instMin___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instMin___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instMin(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instMin___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_SupHom_instPartialOrder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_SupHom_instPartialOrder___closed__0 = (const lean_object*)&lp_mathlib_SupHom_instPartialOrder___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instPartialOrder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instPartialOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instPartialOrder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instPartialOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instSemilatticeSup___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instSemilatticeSup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instSemilatticeSup___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instSemilatticeSup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instSemilatticeSup___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instSemilatticeInf___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instSemilatticeInf___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instSemilatticeInf(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instSemilatticeInf___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instOrderBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instOrderBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instOrderBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instOrderTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instOrderTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instOrderTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instOrderTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instOrderTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instOrderTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instOrderBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instOrderBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instOrderBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instBoundedOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instBoundedOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instBoundedOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instBoundedOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instBoundedOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instBoundedOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_subtypeVal___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_subtypeVal___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_SupHom_subtypeVal___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SupHom_subtypeVal___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_SupHom_subtypeVal___closed__0 = (const lean_object*)&lp_mathlib_SupHom_subtypeVal___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_SupHom_subtypeVal(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_subtypeVal___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_subtypeVal(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_subtypeVal___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_subtypeVal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_subtypeVal___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHomClass_toLatticeHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHomClass_toLatticeHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHomClass_toLatticeHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_SupHom_dual___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SupHom_comp___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_SupHom_dual___closed__0 = (const lean_object*)&lp_mathlib_SupHom_dual___closed__0_value;
static const lean_ctor_object lp_mathlib_SupHom_dual___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_SupHom_dual___closed__0_value),((lean_object*)&lp_mathlib_SupHom_dual___closed__0_value)}};
static const lean_object* lp_mathlib_SupHom_dual___closed__1 = (const lean_object*)&lp_mathlib_SupHom_dual___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_SupHom_dual(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_dual___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_dual(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_dual___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_dual___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_dual___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_dual___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_dual___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_dual___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_dual(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_fst___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_fst___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_LatticeHom_fst___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LatticeHom_fst___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LatticeHom_fst___closed__0 = (const lean_object*)&lp_mathlib_LatticeHom_fst___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_fst(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_fst___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_snd___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_snd___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_LatticeHom_snd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LatticeHom_snd___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_LatticeHom_snd___closed__0 = (const lean_object*)&lp_mathlib_LatticeHom_snd___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_snd(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_snd___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalLatticeHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalLatticeHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalLatticeHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_toInfHom___redArg(lean_object* v_self_1_){
_start:
{
lean_inc(v_self_1_);
return v_self_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_toInfHom___redArg___boxed(lean_object* v_self_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_LatticeHom_toInfHom___redArg(v_self_2_);
lean_dec(v_self_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_toInfHom(lean_object* v_00_u03b1_4_, lean_object* v_00_u03b2_5_, lean_object* v_inst_6_, lean_object* v_inst_7_, lean_object* v_self_8_){
_start:
{
lean_inc(v_self_8_);
return v_self_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_toInfHom___boxed(lean_object* v_00_u03b1_9_, lean_object* v_00_u03b2_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_self_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_LatticeHom_toInfHom(v_00_u03b1_9_, v_00_u03b2_10_, v_inst_11_, v_inst_12_, v_self_13_);
lean_dec(v_self_13_);
lean_dec_ref(v_inst_12_);
lean_dec_ref(v_inst_11_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_orderEmbeddingOfInjective___redArg(lean_object* v_inst_15_, lean_object* v_f_16_){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = lean_apply_1(v_inst_15_, v_f_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_orderEmbeddingOfInjective(lean_object* v_F_18_, lean_object* v_00_u03b1_19_, lean_object* v_00_u03b2_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_inst_23_, lean_object* v_f_24_, lean_object* v_inst_25_, lean_object* v_hf_26_){
_start:
{
lean_object* v___x_27_; 
v___x_27_ = lean_apply_1(v_inst_21_, v_f_24_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_orderEmbeddingOfInjective___boxed(lean_object* v_F_28_, lean_object* v_00_u03b1_29_, lean_object* v_00_u03b2_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_f_34_, lean_object* v_inst_35_, lean_object* v_hf_36_){
_start:
{
lean_object* v_res_37_; 
v_res_37_ = lp_mathlib_orderEmbeddingOfInjective(v_F_28_, v_00_u03b1_29_, v_00_u03b2_30_, v_inst_31_, v_inst_32_, v_inst_33_, v_f_34_, v_inst_35_, v_hf_36_);
lean_dec_ref(v_inst_33_);
lean_dec_ref(v_inst_32_);
return v_res_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSupHomOfSupHomClass___redArg___lam__0(lean_object* v_inst_38_, lean_object* v_f_39_, lean_object* v___y_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lean_apply_2(v_inst_38_, v_f_39_, v___y_40_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSupHomOfSupHomClass___redArg(lean_object* v_inst_42_){
_start:
{
lean_object* v___f_43_; 
v___f_43_ = lean_alloc_closure((void*)(lp_mathlib_instCoeTCSupHomOfSupHomClass___redArg___lam__0), 3, 1);
lean_closure_set(v___f_43_, 0, v_inst_42_);
return v___f_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSupHomOfSupHomClass(lean_object* v_F_44_, lean_object* v_00_u03b1_45_, lean_object* v_00_u03b2_46_, lean_object* v_inst_47_, lean_object* v_inst_48_, lean_object* v_inst_49_, lean_object* v_inst_50_){
_start:
{
lean_object* v___f_51_; 
v___f_51_ = lean_alloc_closure((void*)(lp_mathlib_instCoeTCSupHomOfSupHomClass___redArg___lam__0), 3, 1);
lean_closure_set(v___f_51_, 0, v_inst_47_);
return v___f_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSupHomOfSupHomClass___boxed(lean_object* v_F_52_, lean_object* v_00_u03b1_53_, lean_object* v_00_u03b2_54_, lean_object* v_inst_55_, lean_object* v_inst_56_, lean_object* v_inst_57_, lean_object* v_inst_58_){
_start:
{
lean_object* v_res_59_; 
v_res_59_ = lp_mathlib_instCoeTCSupHomOfSupHomClass(v_F_52_, v_00_u03b1_53_, v_00_u03b2_54_, v_inst_55_, v_inst_56_, v_inst_57_, v_inst_58_);
lean_dec(v_inst_57_);
lean_dec(v_inst_56_);
return v_res_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCInfHomOfInfHomClass___redArg(lean_object* v_inst_60_){
_start:
{
lean_object* v___f_61_; 
v___f_61_ = lean_alloc_closure((void*)(lp_mathlib_instCoeTCSupHomOfSupHomClass___redArg___lam__0), 3, 1);
lean_closure_set(v___f_61_, 0, v_inst_60_);
return v___f_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCInfHomOfInfHomClass(lean_object* v_F_62_, lean_object* v_00_u03b1_63_, lean_object* v_00_u03b2_64_, lean_object* v_inst_65_, lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_inst_68_){
_start:
{
lean_object* v___f_69_; 
v___f_69_ = lean_alloc_closure((void*)(lp_mathlib_instCoeTCSupHomOfSupHomClass___redArg___lam__0), 3, 1);
lean_closure_set(v___f_69_, 0, v_inst_65_);
return v___f_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCInfHomOfInfHomClass___boxed(lean_object* v_F_70_, lean_object* v_00_u03b1_71_, lean_object* v_00_u03b2_72_, lean_object* v_inst_73_, lean_object* v_inst_74_, lean_object* v_inst_75_, lean_object* v_inst_76_){
_start:
{
lean_object* v_res_77_; 
v_res_77_ = lp_mathlib_instCoeTCInfHomOfInfHomClass(v_F_70_, v_00_u03b1_71_, v_00_u03b2_72_, v_inst_73_, v_inst_74_, v_inst_75_, v_inst_76_);
lean_dec(v_inst_75_);
lean_dec(v_inst_74_);
return v_res_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCLatticeHomOfLatticeHomClass___redArg(lean_object* v_inst_78_){
_start:
{
lean_object* v___f_79_; 
v___f_79_ = lean_alloc_closure((void*)(lp_mathlib_instCoeTCSupHomOfSupHomClass___redArg___lam__0), 3, 1);
lean_closure_set(v___f_79_, 0, v_inst_78_);
return v___f_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCLatticeHomOfLatticeHomClass(lean_object* v_F_80_, lean_object* v_00_u03b1_81_, lean_object* v_00_u03b2_82_, lean_object* v_inst_83_, lean_object* v_inst_84_, lean_object* v_inst_85_, lean_object* v_inst_86_){
_start:
{
lean_object* v___f_87_; 
v___f_87_ = lean_alloc_closure((void*)(lp_mathlib_instCoeTCSupHomOfSupHomClass___redArg___lam__0), 3, 1);
lean_closure_set(v___f_87_, 0, v_inst_83_);
return v___f_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCLatticeHomOfLatticeHomClass___boxed(lean_object* v_F_88_, lean_object* v_00_u03b1_89_, lean_object* v_00_u03b2_90_, lean_object* v_inst_91_, lean_object* v_inst_92_, lean_object* v_inst_93_, lean_object* v_inst_94_){
_start:
{
lean_object* v_res_95_; 
v_res_95_ = lp_mathlib_instCoeTCLatticeHomOfLatticeHomClass(v_F_88_, v_00_u03b1_89_, v_00_u03b2_90_, v_inst_91_, v_inst_92_, v_inst_93_, v_inst_94_);
lean_dec_ref(v_inst_93_);
lean_dec_ref(v_inst_92_);
return v_res_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instFunLike___lam__0(lean_object* v_self_96_, lean_object* v___y_97_){
_start:
{
lean_object* v___x_98_; 
v___x_98_ = lean_apply_1(v_self_96_, v___y_97_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instFunLike(lean_object* v_00_u03b1_100_, lean_object* v_00_u03b2_101_, lean_object* v_inst_102_, lean_object* v_inst_103_){
_start:
{
lean_object* v___f_104_; 
v___f_104_ = ((lean_object*)(lp_mathlib_SupHom_instFunLike___closed__0));
return v___f_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instFunLike___boxed(lean_object* v_00_u03b1_105_, lean_object* v_00_u03b2_106_, lean_object* v_inst_107_, lean_object* v_inst_108_){
_start:
{
lean_object* v_res_109_; 
v_res_109_ = lp_mathlib_SupHom_instFunLike(v_00_u03b1_105_, v_00_u03b2_106_, v_inst_107_, v_inst_108_);
lean_dec(v_inst_108_);
lean_dec(v_inst_107_);
return v_res_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instFunLike(lean_object* v_00_u03b1_110_, lean_object* v_00_u03b2_111_, lean_object* v_inst_112_, lean_object* v_inst_113_){
_start:
{
lean_object* v___f_114_; 
v___f_114_ = ((lean_object*)(lp_mathlib_SupHom_instFunLike___closed__0));
return v___f_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instFunLike___boxed(lean_object* v_00_u03b1_115_, lean_object* v_00_u03b2_116_, lean_object* v_inst_117_, lean_object* v_inst_118_){
_start:
{
lean_object* v_res_119_; 
v_res_119_ = lp_mathlib_InfHom_instFunLike(v_00_u03b1_115_, v_00_u03b2_116_, v_inst_117_, v_inst_118_);
lean_dec(v_inst_118_);
lean_dec(v_inst_117_);
return v_res_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_copy___redArg(lean_object* v_f_x27_120_){
_start:
{
lean_inc(v_f_x27_120_);
return v_f_x27_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_copy___redArg___boxed(lean_object* v_f_x27_121_){
_start:
{
lean_object* v_res_122_; 
v_res_122_ = lp_mathlib_SupHom_copy___redArg(v_f_x27_121_);
lean_dec(v_f_x27_121_);
return v_res_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_copy(lean_object* v_00_u03b1_123_, lean_object* v_00_u03b2_124_, lean_object* v_inst_125_, lean_object* v_inst_126_, lean_object* v_f_127_, lean_object* v_f_x27_128_, lean_object* v_h_129_){
_start:
{
lean_inc(v_f_x27_128_);
return v_f_x27_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_copy___boxed(lean_object* v_00_u03b1_130_, lean_object* v_00_u03b2_131_, lean_object* v_inst_132_, lean_object* v_inst_133_, lean_object* v_f_134_, lean_object* v_f_x27_135_, lean_object* v_h_136_){
_start:
{
lean_object* v_res_137_; 
v_res_137_ = lp_mathlib_SupHom_copy(v_00_u03b1_130_, v_00_u03b2_131_, v_inst_132_, v_inst_133_, v_f_134_, v_f_x27_135_, v_h_136_);
lean_dec(v_f_x27_135_);
lean_dec(v_f_134_);
lean_dec(v_inst_133_);
lean_dec(v_inst_132_);
return v_res_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_copy___redArg(lean_object* v_f_x27_138_){
_start:
{
lean_inc(v_f_x27_138_);
return v_f_x27_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_copy___redArg___boxed(lean_object* v_f_x27_139_){
_start:
{
lean_object* v_res_140_; 
v_res_140_ = lp_mathlib_InfHom_copy___redArg(v_f_x27_139_);
lean_dec(v_f_x27_139_);
return v_res_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_copy(lean_object* v_00_u03b1_141_, lean_object* v_00_u03b2_142_, lean_object* v_inst_143_, lean_object* v_inst_144_, lean_object* v_f_145_, lean_object* v_f_x27_146_, lean_object* v_h_147_){
_start:
{
lean_inc(v_f_x27_146_);
return v_f_x27_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_copy___boxed(lean_object* v_00_u03b1_148_, lean_object* v_00_u03b2_149_, lean_object* v_inst_150_, lean_object* v_inst_151_, lean_object* v_f_152_, lean_object* v_f_x27_153_, lean_object* v_h_154_){
_start:
{
lean_object* v_res_155_; 
v_res_155_ = lp_mathlib_InfHom_copy(v_00_u03b1_148_, v_00_u03b2_149_, v_inst_150_, v_inst_151_, v_f_152_, v_f_x27_153_, v_h_154_);
lean_dec(v_f_x27_153_);
lean_dec(v_f_152_);
lean_dec(v_inst_151_);
lean_dec(v_inst_150_);
return v_res_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_id(lean_object* v_00_u03b1_157_, lean_object* v_inst_158_){
_start:
{
lean_object* v___x_159_; 
v___x_159_ = ((lean_object*)(lp_mathlib_SupHom_id___closed__0));
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_id___boxed(lean_object* v_00_u03b1_160_, lean_object* v_inst_161_){
_start:
{
lean_object* v_res_162_; 
v_res_162_ = lp_mathlib_SupHom_id(v_00_u03b1_160_, v_inst_161_);
lean_dec(v_inst_161_);
return v_res_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_id(lean_object* v_00_u03b1_163_, lean_object* v_inst_164_){
_start:
{
lean_object* v___x_165_; 
v___x_165_ = ((lean_object*)(lp_mathlib_SupHom_id___closed__0));
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_id___boxed(lean_object* v_00_u03b1_166_, lean_object* v_inst_167_){
_start:
{
lean_object* v_res_168_; 
v_res_168_ = lp_mathlib_InfHom_id(v_00_u03b1_166_, v_inst_167_);
lean_dec(v_inst_167_);
return v_res_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instInhabited(lean_object* v_00_u03b1_169_, lean_object* v_inst_170_){
_start:
{
lean_object* v___x_171_; 
v___x_171_ = ((lean_object*)(lp_mathlib_SupHom_id___closed__0));
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instInhabited___boxed(lean_object* v_00_u03b1_172_, lean_object* v_inst_173_){
_start:
{
lean_object* v_res_174_; 
v_res_174_ = lp_mathlib_SupHom_instInhabited(v_00_u03b1_172_, v_inst_173_);
lean_dec(v_inst_173_);
return v_res_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instInhabited(lean_object* v_00_u03b1_175_, lean_object* v_inst_176_){
_start:
{
lean_object* v___x_177_; 
v___x_177_ = ((lean_object*)(lp_mathlib_SupHom_id___closed__0));
return v___x_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instInhabited___boxed(lean_object* v_00_u03b1_178_, lean_object* v_inst_179_){
_start:
{
lean_object* v_res_180_; 
v_res_180_ = lp_mathlib_InfHom_instInhabited(v_00_u03b1_178_, v_inst_179_);
lean_dec(v_inst_179_);
return v_res_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_comp___redArg___lam__0(lean_object* v_f_181_, lean_object* v___y_182_){
_start:
{
lean_object* v___x_183_; 
v___x_183_ = lean_apply_1(v_f_181_, v___y_182_);
return v___x_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_comp___redArg___lam__1(lean_object* v_g_184_, lean_object* v___y_185_){
_start:
{
lean_object* v___x_186_; 
v___x_186_ = lean_apply_1(v_g_184_, v___y_185_);
return v___x_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_comp___redArg(lean_object* v_f_187_, lean_object* v_g_188_){
_start:
{
lean_object* v___f_189_; lean_object* v___f_190_; lean_object* v___x_191_; 
v___f_189_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_comp___redArg___lam__0), 2, 1);
lean_closure_set(v___f_189_, 0, v_f_187_);
v___f_190_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_comp___redArg___lam__1), 2, 1);
lean_closure_set(v___f_190_, 0, v_g_188_);
v___x_191_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_191_, 0, lean_box(0));
lean_closure_set(v___x_191_, 1, lean_box(0));
lean_closure_set(v___x_191_, 2, lean_box(0));
lean_closure_set(v___x_191_, 3, v___f_189_);
lean_closure_set(v___x_191_, 4, v___f_190_);
return v___x_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_comp(lean_object* v_00_u03b1_192_, lean_object* v_00_u03b2_193_, lean_object* v_00_u03b3_194_, lean_object* v_inst_195_, lean_object* v_inst_196_, lean_object* v_inst_197_, lean_object* v_f_198_, lean_object* v_g_199_){
_start:
{
lean_object* v___x_200_; 
v___x_200_ = lp_mathlib_SupHom_comp___redArg(v_f_198_, v_g_199_);
return v___x_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_comp___boxed(lean_object* v_00_u03b1_201_, lean_object* v_00_u03b2_202_, lean_object* v_00_u03b3_203_, lean_object* v_inst_204_, lean_object* v_inst_205_, lean_object* v_inst_206_, lean_object* v_f_207_, lean_object* v_g_208_){
_start:
{
lean_object* v_res_209_; 
v_res_209_ = lp_mathlib_SupHom_comp(v_00_u03b1_201_, v_00_u03b2_202_, v_00_u03b3_203_, v_inst_204_, v_inst_205_, v_inst_206_, v_f_207_, v_g_208_);
lean_dec(v_inst_206_);
lean_dec(v_inst_205_);
lean_dec(v_inst_204_);
return v_res_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_comp___redArg(lean_object* v_f_210_, lean_object* v_g_211_){
_start:
{
lean_object* v___f_212_; lean_object* v___f_213_; lean_object* v___x_214_; 
v___f_212_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_comp___redArg___lam__0), 2, 1);
lean_closure_set(v___f_212_, 0, v_f_210_);
v___f_213_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_comp___redArg___lam__1), 2, 1);
lean_closure_set(v___f_213_, 0, v_g_211_);
v___x_214_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_214_, 0, lean_box(0));
lean_closure_set(v___x_214_, 1, lean_box(0));
lean_closure_set(v___x_214_, 2, lean_box(0));
lean_closure_set(v___x_214_, 3, v___f_212_);
lean_closure_set(v___x_214_, 4, v___f_213_);
return v___x_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_comp(lean_object* v_00_u03b1_215_, lean_object* v_00_u03b2_216_, lean_object* v_00_u03b3_217_, lean_object* v_inst_218_, lean_object* v_inst_219_, lean_object* v_inst_220_, lean_object* v_f_221_, lean_object* v_g_222_){
_start:
{
lean_object* v___x_223_; 
v___x_223_ = lp_mathlib_InfHom_comp___redArg(v_f_221_, v_g_222_);
return v___x_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_comp___boxed(lean_object* v_00_u03b1_224_, lean_object* v_00_u03b2_225_, lean_object* v_00_u03b3_226_, lean_object* v_inst_227_, lean_object* v_inst_228_, lean_object* v_inst_229_, lean_object* v_f_230_, lean_object* v_g_231_){
_start:
{
lean_object* v_res_232_; 
v_res_232_ = lp_mathlib_InfHom_comp(v_00_u03b1_224_, v_00_u03b2_225_, v_00_u03b3_226_, v_inst_227_, v_inst_228_, v_inst_229_, v_f_230_, v_g_231_);
lean_dec(v_inst_229_);
lean_dec(v_inst_228_);
lean_dec(v_inst_227_);
return v_res_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_const___redArg___lam__0(lean_object* v_b_233_, lean_object* v_x_234_){
_start:
{
lean_inc(v_b_233_);
return v_b_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_const___redArg___lam__0___boxed(lean_object* v_b_235_, lean_object* v_x_236_){
_start:
{
lean_object* v_res_237_; 
v_res_237_ = lp_mathlib_SupHom_const___redArg___lam__0(v_b_235_, v_x_236_);
lean_dec(v_x_236_);
lean_dec(v_b_235_);
return v_res_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_const___redArg(lean_object* v_b_238_){
_start:
{
lean_object* v___f_239_; 
v___f_239_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_239_, 0, v_b_238_);
return v___f_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_const(lean_object* v_00_u03b1_240_, lean_object* v_00_u03b2_241_, lean_object* v_inst_242_, lean_object* v_inst_243_, lean_object* v_b_244_){
_start:
{
lean_object* v___f_245_; 
v___f_245_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_245_, 0, v_b_244_);
return v___f_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_const___boxed(lean_object* v_00_u03b1_246_, lean_object* v_00_u03b2_247_, lean_object* v_inst_248_, lean_object* v_inst_249_, lean_object* v_b_250_){
_start:
{
lean_object* v_res_251_; 
v_res_251_ = lp_mathlib_SupHom_const(v_00_u03b1_246_, v_00_u03b2_247_, v_inst_248_, v_inst_249_, v_b_250_);
lean_dec_ref(v_inst_249_);
lean_dec(v_inst_248_);
return v_res_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_const___redArg(lean_object* v_b_252_){
_start:
{
lean_object* v___f_253_; 
v___f_253_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_253_, 0, v_b_252_);
return v___f_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_const(lean_object* v_00_u03b1_254_, lean_object* v_00_u03b2_255_, lean_object* v_inst_256_, lean_object* v_inst_257_, lean_object* v_b_258_){
_start:
{
lean_object* v___f_259_; 
v___f_259_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_259_, 0, v_b_258_);
return v___f_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_const___boxed(lean_object* v_00_u03b1_260_, lean_object* v_00_u03b2_261_, lean_object* v_inst_262_, lean_object* v_inst_263_, lean_object* v_b_264_){
_start:
{
lean_object* v_res_265_; 
v_res_265_ = lp_mathlib_InfHom_const(v_00_u03b1_260_, v_00_u03b2_261_, v_inst_262_, v_inst_263_, v_b_264_);
lean_dec_ref(v_inst_263_);
lean_dec(v_inst_262_);
return v_res_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instMax___redArg___lam__0(lean_object* v_inst_266_, lean_object* v_f_267_, lean_object* v_g_268_, lean_object* v___y_269_){
_start:
{
lean_object* v_sup_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; 
v_sup_270_ = lean_ctor_get(v_inst_266_, 1);
lean_inc(v_sup_270_);
lean_dec_ref(v_inst_266_);
lean_inc(v___y_269_);
v___x_271_ = lean_apply_1(v_f_267_, v___y_269_);
v___x_272_ = lean_apply_1(v_g_268_, v___y_269_);
v___x_273_ = lean_apply_2(v_sup_270_, v___x_271_, v___x_272_);
return v___x_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instMax___redArg(lean_object* v_inst_274_){
_start:
{
lean_object* v___f_275_; 
v___f_275_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_instMax___redArg___lam__0), 4, 1);
lean_closure_set(v___f_275_, 0, v_inst_274_);
return v___f_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instMax(lean_object* v_00_u03b1_276_, lean_object* v_00_u03b2_277_, lean_object* v_inst_278_, lean_object* v_inst_279_){
_start:
{
lean_object* v___f_280_; 
v___f_280_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_instMax___redArg___lam__0), 4, 1);
lean_closure_set(v___f_280_, 0, v_inst_279_);
return v___f_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instMax___boxed(lean_object* v_00_u03b1_281_, lean_object* v_00_u03b2_282_, lean_object* v_inst_283_, lean_object* v_inst_284_){
_start:
{
lean_object* v_res_285_; 
v_res_285_ = lp_mathlib_SupHom_instMax(v_00_u03b1_281_, v_00_u03b2_282_, v_inst_283_, v_inst_284_);
lean_dec(v_inst_283_);
return v_res_285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instMin___redArg___lam__0(lean_object* v_inst_286_, lean_object* v_f_287_, lean_object* v_g_288_, lean_object* v___y_289_){
_start:
{
lean_object* v_inf_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; 
v_inf_290_ = lean_ctor_get(v_inst_286_, 1);
lean_inc(v_inf_290_);
lean_dec_ref(v_inst_286_);
lean_inc(v___y_289_);
v___x_291_ = lean_apply_1(v_f_287_, v___y_289_);
v___x_292_ = lean_apply_1(v_g_288_, v___y_289_);
v___x_293_ = lean_apply_2(v_inf_290_, v___x_291_, v___x_292_);
return v___x_293_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instMin___redArg(lean_object* v_inst_294_){
_start:
{
lean_object* v___f_295_; 
v___f_295_ = lean_alloc_closure((void*)(lp_mathlib_InfHom_instMin___redArg___lam__0), 4, 1);
lean_closure_set(v___f_295_, 0, v_inst_294_);
return v___f_295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instMin(lean_object* v_00_u03b1_296_, lean_object* v_00_u03b2_297_, lean_object* v_inst_298_, lean_object* v_inst_299_){
_start:
{
lean_object* v___f_300_; 
v___f_300_ = lean_alloc_closure((void*)(lp_mathlib_InfHom_instMin___redArg___lam__0), 4, 1);
lean_closure_set(v___f_300_, 0, v_inst_299_);
return v___f_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instMin___boxed(lean_object* v_00_u03b1_301_, lean_object* v_00_u03b2_302_, lean_object* v_inst_303_, lean_object* v_inst_304_){
_start:
{
lean_object* v_res_305_; 
v_res_305_ = lp_mathlib_InfHom_instMin(v_00_u03b1_301_, v_00_u03b2_302_, v_inst_303_, v_inst_304_);
lean_dec(v_inst_303_);
return v_res_305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instPartialOrder(lean_object* v_00_u03b1_309_, lean_object* v_00_u03b2_310_, lean_object* v_inst_311_, lean_object* v_inst_312_){
_start:
{
lean_object* v___x_313_; 
v___x_313_ = ((lean_object*)(lp_mathlib_SupHom_instPartialOrder___closed__0));
return v___x_313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instPartialOrder___boxed(lean_object* v_00_u03b1_314_, lean_object* v_00_u03b2_315_, lean_object* v_inst_316_, lean_object* v_inst_317_){
_start:
{
lean_object* v_res_318_; 
v_res_318_ = lp_mathlib_SupHom_instPartialOrder(v_00_u03b1_314_, v_00_u03b2_315_, v_inst_316_, v_inst_317_);
lean_dec_ref(v_inst_317_);
lean_dec(v_inst_316_);
return v_res_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instPartialOrder(lean_object* v_00_u03b1_319_, lean_object* v_00_u03b2_320_, lean_object* v_inst_321_, lean_object* v_inst_322_){
_start:
{
lean_object* v___x_323_; 
v___x_323_ = ((lean_object*)(lp_mathlib_SupHom_instPartialOrder___closed__0));
return v___x_323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instPartialOrder___boxed(lean_object* v_00_u03b1_324_, lean_object* v_00_u03b2_325_, lean_object* v_inst_326_, lean_object* v_inst_327_){
_start:
{
lean_object* v_res_328_; 
v_res_328_ = lp_mathlib_InfHom_instPartialOrder(v_00_u03b1_324_, v_00_u03b2_325_, v_inst_326_, v_inst_327_);
lean_dec_ref(v_inst_327_);
lean_dec(v_inst_326_);
return v_res_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instSemilatticeSup___redArg___lam__0(lean_object* v_inst_329_, lean_object* v_a_330_, lean_object* v_b_331_, lean_object* v___y_332_){
_start:
{
lean_object* v_sup_333_; lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; 
v_sup_333_ = lean_ctor_get(v_inst_329_, 1);
lean_inc(v_sup_333_);
lean_dec_ref(v_inst_329_);
lean_inc(v___y_332_);
v___x_334_ = lean_apply_1(v_a_330_, v___y_332_);
v___x_335_ = lean_apply_1(v_b_331_, v___y_332_);
v___x_336_ = lean_apply_2(v_sup_333_, v___x_334_, v___x_335_);
return v___x_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instSemilatticeSup___redArg(lean_object* v_inst_337_, lean_object* v_inst_338_){
_start:
{
lean_object* v___x_339_; lean_object* v_toLE_340_; lean_object* v_toLT_341_; lean_object* v___x_343_; uint8_t v_isShared_344_; uint8_t v_isSharedCheck_350_; 
v___x_339_ = lp_mathlib_SupHom_instPartialOrder(lean_box(0), lean_box(0), v_inst_337_, v_inst_338_);
v_toLE_340_ = lean_ctor_get(v___x_339_, 0);
v_toLT_341_ = lean_ctor_get(v___x_339_, 1);
v_isSharedCheck_350_ = !lean_is_exclusive(v___x_339_);
if (v_isSharedCheck_350_ == 0)
{
v___x_343_ = v___x_339_;
v_isShared_344_ = v_isSharedCheck_350_;
goto v_resetjp_342_;
}
else
{
lean_inc(v_toLT_341_);
lean_inc(v_toLE_340_);
lean_dec(v___x_339_);
v___x_343_ = lean_box(0);
v_isShared_344_ = v_isSharedCheck_350_;
goto v_resetjp_342_;
}
v_resetjp_342_:
{
lean_object* v___f_345_; lean_object* v___x_347_; 
v___f_345_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_instSemilatticeSup___redArg___lam__0), 4, 1);
lean_closure_set(v___f_345_, 0, v_inst_338_);
if (v_isShared_344_ == 0)
{
v___x_347_ = v___x_343_;
goto v_reusejp_346_;
}
else
{
lean_object* v_reuseFailAlloc_349_; 
v_reuseFailAlloc_349_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_349_, 0, v_toLE_340_);
lean_ctor_set(v_reuseFailAlloc_349_, 1, v_toLT_341_);
v___x_347_ = v_reuseFailAlloc_349_;
goto v_reusejp_346_;
}
v_reusejp_346_:
{
lean_object* v___x_348_; 
v___x_348_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_348_, 0, v___x_347_);
lean_ctor_set(v___x_348_, 1, v___f_345_);
return v___x_348_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instSemilatticeSup___redArg___boxed(lean_object* v_inst_351_, lean_object* v_inst_352_){
_start:
{
lean_object* v_res_353_; 
v_res_353_ = lp_mathlib_SupHom_instSemilatticeSup___redArg(v_inst_351_, v_inst_352_);
lean_dec(v_inst_351_);
return v_res_353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instSemilatticeSup(lean_object* v_00_u03b1_354_, lean_object* v_00_u03b2_355_, lean_object* v_inst_356_, lean_object* v_inst_357_){
_start:
{
lean_object* v___x_358_; 
v___x_358_ = lp_mathlib_SupHom_instSemilatticeSup___redArg(v_inst_356_, v_inst_357_);
return v___x_358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instSemilatticeSup___boxed(lean_object* v_00_u03b1_359_, lean_object* v_00_u03b2_360_, lean_object* v_inst_361_, lean_object* v_inst_362_){
_start:
{
lean_object* v_res_363_; 
v_res_363_ = lp_mathlib_SupHom_instSemilatticeSup(v_00_u03b1_359_, v_00_u03b2_360_, v_inst_361_, v_inst_362_);
lean_dec(v_inst_361_);
return v_res_363_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instSemilatticeInf___redArg(lean_object* v_inst_364_, lean_object* v_inst_365_){
_start:
{
lean_object* v___x_366_; lean_object* v_toLE_367_; lean_object* v_toLT_368_; lean_object* v___f_369_; lean_object* v___x_370_; 
v___x_366_ = lp_mathlib_InfHom_instPartialOrder(lean_box(0), lean_box(0), v_inst_364_, v_inst_365_);
v_toLE_367_ = lean_ctor_get(v___x_366_, 0);
lean_inc(v_toLE_367_);
v_toLT_368_ = lean_ctor_get(v___x_366_, 1);
lean_inc(v_toLT_368_);
lean_dec_ref(v___x_366_);
v___f_369_ = lean_alloc_closure((void*)(lp_mathlib_InfHom_instMin___redArg___lam__0), 4, 1);
lean_closure_set(v___f_369_, 0, v_inst_365_);
v___x_370_ = lp_mathlib_Function_Injective_semilatticeInf___redArg(v___f_369_, v_toLE_367_, v_toLT_368_);
return v___x_370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instSemilatticeInf___redArg___boxed(lean_object* v_inst_371_, lean_object* v_inst_372_){
_start:
{
lean_object* v_res_373_; 
v_res_373_ = lp_mathlib_InfHom_instSemilatticeInf___redArg(v_inst_371_, v_inst_372_);
lean_dec(v_inst_371_);
return v_res_373_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instSemilatticeInf(lean_object* v_00_u03b1_374_, lean_object* v_00_u03b2_375_, lean_object* v_inst_376_, lean_object* v_inst_377_){
_start:
{
lean_object* v___x_378_; 
v___x_378_ = lp_mathlib_InfHom_instSemilatticeInf___redArg(v_inst_376_, v_inst_377_);
return v___x_378_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instSemilatticeInf___boxed(lean_object* v_00_u03b1_379_, lean_object* v_00_u03b2_380_, lean_object* v_inst_381_, lean_object* v_inst_382_){
_start:
{
lean_object* v_res_383_; 
v_res_383_ = lp_mathlib_InfHom_instSemilatticeInf(v_00_u03b1_379_, v_00_u03b2_380_, v_inst_381_, v_inst_382_);
lean_dec(v_inst_381_);
return v_res_383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instBot___redArg(lean_object* v_inst_384_){
_start:
{
lean_object* v___f_385_; 
v___f_385_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_385_, 0, v_inst_384_);
return v___f_385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instBot(lean_object* v_00_u03b1_386_, lean_object* v_00_u03b2_387_, lean_object* v_inst_388_, lean_object* v_inst_389_, lean_object* v_inst_390_){
_start:
{
lean_object* v___f_391_; 
v___f_391_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_391_, 0, v_inst_390_);
return v___f_391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instBot___boxed(lean_object* v_00_u03b1_392_, lean_object* v_00_u03b2_393_, lean_object* v_inst_394_, lean_object* v_inst_395_, lean_object* v_inst_396_){
_start:
{
lean_object* v_res_397_; 
v_res_397_ = lp_mathlib_SupHom_instBot(v_00_u03b1_392_, v_00_u03b2_393_, v_inst_394_, v_inst_395_, v_inst_396_);
lean_dec_ref(v_inst_395_);
lean_dec(v_inst_394_);
return v_res_397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instTop___redArg(lean_object* v_inst_398_){
_start:
{
lean_object* v___f_399_; 
v___f_399_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_399_, 0, v_inst_398_);
return v___f_399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instTop(lean_object* v_00_u03b1_400_, lean_object* v_00_u03b2_401_, lean_object* v_inst_402_, lean_object* v_inst_403_, lean_object* v_inst_404_){
_start:
{
lean_object* v___f_405_; 
v___f_405_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_405_, 0, v_inst_404_);
return v___f_405_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instTop___boxed(lean_object* v_00_u03b1_406_, lean_object* v_00_u03b2_407_, lean_object* v_inst_408_, lean_object* v_inst_409_, lean_object* v_inst_410_){
_start:
{
lean_object* v_res_411_; 
v_res_411_ = lp_mathlib_InfHom_instTop(v_00_u03b1_406_, v_00_u03b2_407_, v_inst_408_, v_inst_409_, v_inst_410_);
lean_dec_ref(v_inst_409_);
lean_dec(v_inst_408_);
return v_res_411_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instTop___redArg(lean_object* v_inst_412_){
_start:
{
lean_object* v___f_413_; 
v___f_413_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_413_, 0, v_inst_412_);
return v___f_413_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instTop(lean_object* v_00_u03b1_414_, lean_object* v_00_u03b2_415_, lean_object* v_inst_416_, lean_object* v_inst_417_, lean_object* v_inst_418_){
_start:
{
lean_object* v___f_419_; 
v___f_419_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_419_, 0, v_inst_418_);
return v___f_419_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instTop___boxed(lean_object* v_00_u03b1_420_, lean_object* v_00_u03b2_421_, lean_object* v_inst_422_, lean_object* v_inst_423_, lean_object* v_inst_424_){
_start:
{
lean_object* v_res_425_; 
v_res_425_ = lp_mathlib_SupHom_instTop(v_00_u03b1_420_, v_00_u03b2_421_, v_inst_422_, v_inst_423_, v_inst_424_);
lean_dec_ref(v_inst_423_);
lean_dec(v_inst_422_);
return v_res_425_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instBot___redArg(lean_object* v_inst_426_){
_start:
{
lean_object* v___f_427_; 
v___f_427_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_427_, 0, v_inst_426_);
return v___f_427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instBot(lean_object* v_00_u03b1_428_, lean_object* v_00_u03b2_429_, lean_object* v_inst_430_, lean_object* v_inst_431_, lean_object* v_inst_432_){
_start:
{
lean_object* v___f_433_; 
v___f_433_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_433_, 0, v_inst_432_);
return v___f_433_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instBot___boxed(lean_object* v_00_u03b1_434_, lean_object* v_00_u03b2_435_, lean_object* v_inst_436_, lean_object* v_inst_437_, lean_object* v_inst_438_){
_start:
{
lean_object* v_res_439_; 
v_res_439_ = lp_mathlib_InfHom_instBot(v_00_u03b1_434_, v_00_u03b2_435_, v_inst_436_, v_inst_437_, v_inst_438_);
lean_dec_ref(v_inst_437_);
lean_dec(v_inst_436_);
return v_res_439_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instOrderBot___redArg(lean_object* v_inst_440_){
_start:
{
lean_object* v___f_441_; 
v___f_441_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_441_, 0, v_inst_440_);
return v___f_441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instOrderBot(lean_object* v_00_u03b1_442_, lean_object* v_00_u03b2_443_, lean_object* v_inst_444_, lean_object* v_inst_445_, lean_object* v_inst_446_){
_start:
{
lean_object* v___f_447_; 
v___f_447_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_447_, 0, v_inst_446_);
return v___f_447_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instOrderBot___boxed(lean_object* v_00_u03b1_448_, lean_object* v_00_u03b2_449_, lean_object* v_inst_450_, lean_object* v_inst_451_, lean_object* v_inst_452_){
_start:
{
lean_object* v_res_453_; 
v_res_453_ = lp_mathlib_SupHom_instOrderBot(v_00_u03b1_448_, v_00_u03b2_449_, v_inst_450_, v_inst_451_, v_inst_452_);
lean_dec_ref(v_inst_451_);
lean_dec(v_inst_450_);
return v_res_453_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instOrderTop___redArg(lean_object* v_inst_454_){
_start:
{
lean_object* v___f_455_; 
v___f_455_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_455_, 0, v_inst_454_);
return v___f_455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instOrderTop(lean_object* v_00_u03b1_456_, lean_object* v_00_u03b2_457_, lean_object* v_inst_458_, lean_object* v_inst_459_, lean_object* v_inst_460_){
_start:
{
lean_object* v___f_461_; 
v___f_461_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_461_, 0, v_inst_460_);
return v___f_461_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instOrderTop___boxed(lean_object* v_00_u03b1_462_, lean_object* v_00_u03b2_463_, lean_object* v_inst_464_, lean_object* v_inst_465_, lean_object* v_inst_466_){
_start:
{
lean_object* v_res_467_; 
v_res_467_ = lp_mathlib_InfHom_instOrderTop(v_00_u03b1_462_, v_00_u03b2_463_, v_inst_464_, v_inst_465_, v_inst_466_);
lean_dec_ref(v_inst_465_);
lean_dec(v_inst_464_);
return v_res_467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instOrderTop___redArg(lean_object* v_inst_468_){
_start:
{
lean_object* v___f_469_; 
v___f_469_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_469_, 0, v_inst_468_);
return v___f_469_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instOrderTop(lean_object* v_00_u03b1_470_, lean_object* v_00_u03b2_471_, lean_object* v_inst_472_, lean_object* v_inst_473_, lean_object* v_inst_474_){
_start:
{
lean_object* v___f_475_; 
v___f_475_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_475_, 0, v_inst_474_);
return v___f_475_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instOrderTop___boxed(lean_object* v_00_u03b1_476_, lean_object* v_00_u03b2_477_, lean_object* v_inst_478_, lean_object* v_inst_479_, lean_object* v_inst_480_){
_start:
{
lean_object* v_res_481_; 
v_res_481_ = lp_mathlib_SupHom_instOrderTop(v_00_u03b1_476_, v_00_u03b2_477_, v_inst_478_, v_inst_479_, v_inst_480_);
lean_dec_ref(v_inst_479_);
lean_dec(v_inst_478_);
return v_res_481_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instOrderBot___redArg(lean_object* v_inst_482_){
_start:
{
lean_object* v___f_483_; 
v___f_483_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_483_, 0, v_inst_482_);
return v___f_483_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instOrderBot(lean_object* v_00_u03b1_484_, lean_object* v_00_u03b2_485_, lean_object* v_inst_486_, lean_object* v_inst_487_, lean_object* v_inst_488_){
_start:
{
lean_object* v___f_489_; 
v___f_489_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_489_, 0, v_inst_488_);
return v___f_489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instOrderBot___boxed(lean_object* v_00_u03b1_490_, lean_object* v_00_u03b2_491_, lean_object* v_inst_492_, lean_object* v_inst_493_, lean_object* v_inst_494_){
_start:
{
lean_object* v_res_495_; 
v_res_495_ = lp_mathlib_InfHom_instOrderBot(v_00_u03b1_490_, v_00_u03b2_491_, v_inst_492_, v_inst_493_, v_inst_494_);
lean_dec_ref(v_inst_493_);
lean_dec(v_inst_492_);
return v_res_495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instBoundedOrder___redArg(lean_object* v_inst_496_){
_start:
{
lean_object* v_toOrderTop_497_; lean_object* v_toOrderBot_498_; lean_object* v___x_500_; uint8_t v_isShared_501_; uint8_t v_isSharedCheck_507_; 
v_toOrderTop_497_ = lean_ctor_get(v_inst_496_, 0);
v_toOrderBot_498_ = lean_ctor_get(v_inst_496_, 1);
v_isSharedCheck_507_ = !lean_is_exclusive(v_inst_496_);
if (v_isSharedCheck_507_ == 0)
{
v___x_500_ = v_inst_496_;
v_isShared_501_ = v_isSharedCheck_507_;
goto v_resetjp_499_;
}
else
{
lean_inc(v_toOrderBot_498_);
lean_inc(v_toOrderTop_497_);
lean_dec(v_inst_496_);
v___x_500_ = lean_box(0);
v_isShared_501_ = v_isSharedCheck_507_;
goto v_resetjp_499_;
}
v_resetjp_499_:
{
lean_object* v___f_502_; lean_object* v___f_503_; lean_object* v___x_505_; 
v___f_502_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_502_, 0, v_toOrderTop_497_);
v___f_503_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_503_, 0, v_toOrderBot_498_);
if (v_isShared_501_ == 0)
{
lean_ctor_set(v___x_500_, 1, v___f_503_);
lean_ctor_set(v___x_500_, 0, v___f_502_);
v___x_505_ = v___x_500_;
goto v_reusejp_504_;
}
else
{
lean_object* v_reuseFailAlloc_506_; 
v_reuseFailAlloc_506_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_506_, 0, v___f_502_);
lean_ctor_set(v_reuseFailAlloc_506_, 1, v___f_503_);
v___x_505_ = v_reuseFailAlloc_506_;
goto v_reusejp_504_;
}
v_reusejp_504_:
{
return v___x_505_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instBoundedOrder(lean_object* v_00_u03b1_508_, lean_object* v_00_u03b2_509_, lean_object* v_inst_510_, lean_object* v_inst_511_, lean_object* v_inst_512_){
_start:
{
lean_object* v___x_513_; 
v___x_513_ = lp_mathlib_SupHom_instBoundedOrder___redArg(v_inst_512_);
return v___x_513_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_instBoundedOrder___boxed(lean_object* v_00_u03b1_514_, lean_object* v_00_u03b2_515_, lean_object* v_inst_516_, lean_object* v_inst_517_, lean_object* v_inst_518_){
_start:
{
lean_object* v_res_519_; 
v_res_519_ = lp_mathlib_SupHom_instBoundedOrder(v_00_u03b1_514_, v_00_u03b2_515_, v_inst_516_, v_inst_517_, v_inst_518_);
lean_dec_ref(v_inst_517_);
lean_dec(v_inst_516_);
return v_res_519_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instBoundedOrder___redArg(lean_object* v_inst_520_){
_start:
{
lean_object* v_toOrderTop_521_; lean_object* v_toOrderBot_522_; lean_object* v___x_524_; uint8_t v_isShared_525_; uint8_t v_isSharedCheck_531_; 
v_toOrderTop_521_ = lean_ctor_get(v_inst_520_, 0);
v_toOrderBot_522_ = lean_ctor_get(v_inst_520_, 1);
v_isSharedCheck_531_ = !lean_is_exclusive(v_inst_520_);
if (v_isSharedCheck_531_ == 0)
{
v___x_524_ = v_inst_520_;
v_isShared_525_ = v_isSharedCheck_531_;
goto v_resetjp_523_;
}
else
{
lean_inc(v_toOrderBot_522_);
lean_inc(v_toOrderTop_521_);
lean_dec(v_inst_520_);
v___x_524_ = lean_box(0);
v_isShared_525_ = v_isSharedCheck_531_;
goto v_resetjp_523_;
}
v_resetjp_523_:
{
lean_object* v___f_526_; lean_object* v___f_527_; lean_object* v___x_529_; 
v___f_526_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_526_, 0, v_toOrderTop_521_);
v___f_527_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_const___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_527_, 0, v_toOrderBot_522_);
if (v_isShared_525_ == 0)
{
lean_ctor_set(v___x_524_, 1, v___f_527_);
lean_ctor_set(v___x_524_, 0, v___f_526_);
v___x_529_ = v___x_524_;
goto v_reusejp_528_;
}
else
{
lean_object* v_reuseFailAlloc_530_; 
v_reuseFailAlloc_530_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_530_, 0, v___f_526_);
lean_ctor_set(v_reuseFailAlloc_530_, 1, v___f_527_);
v___x_529_ = v_reuseFailAlloc_530_;
goto v_reusejp_528_;
}
v_reusejp_528_:
{
return v___x_529_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instBoundedOrder(lean_object* v_00_u03b1_532_, lean_object* v_00_u03b2_533_, lean_object* v_inst_534_, lean_object* v_inst_535_, lean_object* v_inst_536_){
_start:
{
lean_object* v___x_537_; 
v___x_537_ = lp_mathlib_InfHom_instBoundedOrder___redArg(v_inst_536_);
return v___x_537_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_instBoundedOrder___boxed(lean_object* v_00_u03b1_538_, lean_object* v_00_u03b2_539_, lean_object* v_inst_540_, lean_object* v_inst_541_, lean_object* v_inst_542_){
_start:
{
lean_object* v_res_543_; 
v_res_543_ = lp_mathlib_InfHom_instBoundedOrder(v_00_u03b1_538_, v_00_u03b2_539_, v_inst_540_, v_inst_541_, v_inst_542_);
lean_dec_ref(v_inst_541_);
lean_dec(v_inst_540_);
return v_res_543_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_subtypeVal___lam__0(lean_object* v_self_544_){
_start:
{
lean_inc(v_self_544_);
return v_self_544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_subtypeVal___lam__0___boxed(lean_object* v_self_545_){
_start:
{
lean_object* v_res_546_; 
v_res_546_ = lp_mathlib_SupHom_subtypeVal___lam__0(v_self_545_);
lean_dec(v_self_545_);
return v_res_546_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_subtypeVal(lean_object* v_00_u03b2_548_, lean_object* v_inst_549_, lean_object* v_P_550_, lean_object* v_Psup_551_){
_start:
{
lean_object* v___f_552_; 
v___f_552_ = ((lean_object*)(lp_mathlib_SupHom_subtypeVal___closed__0));
return v___f_552_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_subtypeVal___boxed(lean_object* v_00_u03b2_553_, lean_object* v_inst_554_, lean_object* v_P_555_, lean_object* v_Psup_556_){
_start:
{
lean_object* v_res_557_; 
v_res_557_ = lp_mathlib_SupHom_subtypeVal(v_00_u03b2_553_, v_inst_554_, v_P_555_, v_Psup_556_);
lean_dec_ref(v_inst_554_);
return v_res_557_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_subtypeVal(lean_object* v_00_u03b2_558_, lean_object* v_inst_559_, lean_object* v_P_560_, lean_object* v_Psup_561_){
_start:
{
lean_object* v___f_562_; 
v___f_562_ = ((lean_object*)(lp_mathlib_SupHom_subtypeVal___closed__0));
return v___f_562_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_subtypeVal___boxed(lean_object* v_00_u03b2_563_, lean_object* v_inst_564_, lean_object* v_P_565_, lean_object* v_Psup_566_){
_start:
{
lean_object* v_res_567_; 
v_res_567_ = lp_mathlib_InfHom_subtypeVal(v_00_u03b2_563_, v_inst_564_, v_P_565_, v_Psup_566_);
lean_dec_ref(v_inst_564_);
return v_res_567_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_copy___redArg(lean_object* v_f_x27_568_){
_start:
{
lean_inc(v_f_x27_568_);
return v_f_x27_568_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_copy___redArg___boxed(lean_object* v_f_x27_569_){
_start:
{
lean_object* v_res_570_; 
v_res_570_ = lp_mathlib_LatticeHom_copy___redArg(v_f_x27_569_);
lean_dec(v_f_x27_569_);
return v_res_570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_copy(lean_object* v_00_u03b1_571_, lean_object* v_00_u03b2_572_, lean_object* v_inst_573_, lean_object* v_inst_574_, lean_object* v_f_575_, lean_object* v_f_x27_576_, lean_object* v_h_577_){
_start:
{
lean_inc(v_f_x27_576_);
return v_f_x27_576_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_copy___boxed(lean_object* v_00_u03b1_578_, lean_object* v_00_u03b2_579_, lean_object* v_inst_580_, lean_object* v_inst_581_, lean_object* v_f_582_, lean_object* v_f_x27_583_, lean_object* v_h_584_){
_start:
{
lean_object* v_res_585_; 
v_res_585_ = lp_mathlib_LatticeHom_copy(v_00_u03b1_578_, v_00_u03b2_579_, v_inst_580_, v_inst_581_, v_f_582_, v_f_x27_583_, v_h_584_);
lean_dec(v_f_x27_583_);
lean_dec(v_f_582_);
lean_dec_ref(v_inst_581_);
lean_dec_ref(v_inst_580_);
return v_res_585_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_id(lean_object* v_00_u03b1_586_, lean_object* v_inst_587_){
_start:
{
lean_object* v___x_588_; 
v___x_588_ = ((lean_object*)(lp_mathlib_SupHom_id___closed__0));
return v___x_588_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_id___boxed(lean_object* v_00_u03b1_589_, lean_object* v_inst_590_){
_start:
{
lean_object* v_res_591_; 
v_res_591_ = lp_mathlib_LatticeHom_id(v_00_u03b1_589_, v_inst_590_);
lean_dec_ref(v_inst_590_);
return v_res_591_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_instInhabited(lean_object* v_00_u03b1_592_, lean_object* v_inst_593_){
_start:
{
lean_object* v___x_594_; 
v___x_594_ = ((lean_object*)(lp_mathlib_SupHom_id___closed__0));
return v___x_594_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_instInhabited___boxed(lean_object* v_00_u03b1_595_, lean_object* v_inst_596_){
_start:
{
lean_object* v_res_597_; 
v_res_597_ = lp_mathlib_LatticeHom_instInhabited(v_00_u03b1_595_, v_inst_596_);
lean_dec_ref(v_inst_596_);
return v_res_597_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_comp___redArg(lean_object* v_f_598_, lean_object* v_g_599_){
_start:
{
lean_object* v___x_600_; 
v___x_600_ = lp_mathlib_SupHom_comp___redArg(v_f_598_, v_g_599_);
return v___x_600_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_comp(lean_object* v_00_u03b1_601_, lean_object* v_00_u03b2_602_, lean_object* v_00_u03b3_603_, lean_object* v_inst_604_, lean_object* v_inst_605_, lean_object* v_inst_606_, lean_object* v_f_607_, lean_object* v_g_608_){
_start:
{
lean_object* v___x_609_; 
v___x_609_ = lp_mathlib_SupHom_comp___redArg(v_f_607_, v_g_608_);
return v___x_609_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_comp___boxed(lean_object* v_00_u03b1_610_, lean_object* v_00_u03b2_611_, lean_object* v_00_u03b3_612_, lean_object* v_inst_613_, lean_object* v_inst_614_, lean_object* v_inst_615_, lean_object* v_f_616_, lean_object* v_g_617_){
_start:
{
lean_object* v_res_618_; 
v_res_618_ = lp_mathlib_LatticeHom_comp(v_00_u03b1_610_, v_00_u03b2_611_, v_00_u03b3_612_, v_inst_613_, v_inst_614_, v_inst_615_, v_f_616_, v_g_617_);
lean_dec_ref(v_inst_615_);
lean_dec_ref(v_inst_614_);
lean_dec_ref(v_inst_613_);
return v_res_618_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_subtypeVal(lean_object* v_00_u03b2_619_, lean_object* v_inst_620_, lean_object* v_P_621_, lean_object* v_Psup_622_, lean_object* v_Pinf_623_){
_start:
{
lean_object* v___f_624_; 
v___f_624_ = ((lean_object*)(lp_mathlib_SupHom_subtypeVal___closed__0));
return v___f_624_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_subtypeVal___boxed(lean_object* v_00_u03b2_625_, lean_object* v_inst_626_, lean_object* v_P_627_, lean_object* v_Psup_628_, lean_object* v_Pinf_629_){
_start:
{
lean_object* v_res_630_; 
v_res_630_ = lp_mathlib_LatticeHom_subtypeVal(v_00_u03b2_625_, v_inst_626_, v_P_627_, v_Psup_628_, v_Pinf_629_);
lean_dec_ref(v_inst_626_);
return v_res_630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHomClass_toLatticeHom___redArg(lean_object* v_inst_631_, lean_object* v_f_632_){
_start:
{
lean_object* v___x_633_; 
v___x_633_ = lean_apply_1(v_inst_631_, v_f_632_);
return v___x_633_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHomClass_toLatticeHom(lean_object* v_F_634_, lean_object* v_00_u03b1_635_, lean_object* v_00_u03b2_636_, lean_object* v_inst_637_, lean_object* v_inst_638_, lean_object* v_inst_639_, lean_object* v_inst_640_, lean_object* v_f_641_){
_start:
{
lean_object* v___x_642_; 
v___x_642_ = lean_apply_1(v_inst_637_, v_f_641_);
return v___x_642_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHomClass_toLatticeHom___boxed(lean_object* v_F_643_, lean_object* v_00_u03b1_644_, lean_object* v_00_u03b2_645_, lean_object* v_inst_646_, lean_object* v_inst_647_, lean_object* v_inst_648_, lean_object* v_inst_649_, lean_object* v_f_650_){
_start:
{
lean_object* v_res_651_; 
v_res_651_ = lp_mathlib_OrderHomClass_toLatticeHom(v_F_643_, v_00_u03b1_644_, v_00_u03b2_645_, v_inst_646_, v_inst_647_, v_inst_648_, v_inst_649_, v_f_650_);
lean_dec_ref(v_inst_648_);
lean_dec_ref(v_inst_647_);
return v_res_651_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_dual(lean_object* v_00_u03b1_655_, lean_object* v_00_u03b2_656_, lean_object* v_inst_657_, lean_object* v_inst_658_){
_start:
{
lean_object* v___x_659_; 
v___x_659_ = ((lean_object*)(lp_mathlib_SupHom_dual___closed__1));
return v___x_659_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_dual___boxed(lean_object* v_00_u03b1_660_, lean_object* v_00_u03b2_661_, lean_object* v_inst_662_, lean_object* v_inst_663_){
_start:
{
lean_object* v_res_664_; 
v_res_664_ = lp_mathlib_SupHom_dual(v_00_u03b1_660_, v_00_u03b2_661_, v_inst_662_, v_inst_663_);
lean_dec(v_inst_663_);
lean_dec(v_inst_662_);
return v_res_664_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_dual(lean_object* v_00_u03b1_665_, lean_object* v_00_u03b2_666_, lean_object* v_inst_667_, lean_object* v_inst_668_){
_start:
{
lean_object* v___x_669_; 
v___x_669_ = ((lean_object*)(lp_mathlib_SupHom_dual___closed__1));
return v___x_669_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_dual___boxed(lean_object* v_00_u03b1_670_, lean_object* v_00_u03b2_671_, lean_object* v_inst_672_, lean_object* v_inst_673_){
_start:
{
lean_object* v_res_674_; 
v_res_674_ = lp_mathlib_InfHom_dual(v_00_u03b1_670_, v_00_u03b2_671_, v_inst_672_, v_inst_673_);
lean_dec(v_inst_673_);
lean_dec(v_inst_672_);
return v_res_674_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_dual___redArg___lam__0(lean_object* v___f_675_, lean_object* v___f_676_, lean_object* v_f_677_, lean_object* v___y_678_){
_start:
{
lean_object* v___x_679_; lean_object* v_toFun_680_; lean_object* v___x_681_; 
v___x_679_ = lp_mathlib_InfHom_dual(lean_box(0), lean_box(0), v___f_675_, v___f_676_);
v_toFun_680_ = lean_ctor_get(v___x_679_, 0);
lean_inc(v_toFun_680_);
lean_dec_ref(v___x_679_);
v___x_681_ = lean_apply_2(v_toFun_680_, v_f_677_, v___y_678_);
return v___x_681_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_dual___redArg___lam__0___boxed(lean_object* v___f_682_, lean_object* v___f_683_, lean_object* v_f_684_, lean_object* v___y_685_){
_start:
{
lean_object* v_res_686_; 
v_res_686_ = lp_mathlib_LatticeHom_dual___redArg___lam__0(v___f_682_, v___f_683_, v_f_684_, v___y_685_);
lean_dec(v___f_683_);
lean_dec(v___f_682_);
return v_res_686_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_dual___redArg___lam__1(lean_object* v___f_687_, lean_object* v___f_688_, lean_object* v_f_689_, lean_object* v___y_690_){
_start:
{
lean_object* v___x_691_; lean_object* v___x_692_; lean_object* v_toFun_693_; lean_object* v___x_694_; 
v___x_691_ = lp_mathlib_SupHom_dual(lean_box(0), lean_box(0), v___f_687_, v___f_688_);
v___x_692_ = lp_mathlib_Equiv_symm___redArg(v___x_691_);
v_toFun_693_ = lean_ctor_get(v___x_692_, 0);
lean_inc(v_toFun_693_);
lean_dec_ref(v___x_692_);
v___x_694_ = lean_apply_2(v_toFun_693_, v_f_689_, v___y_690_);
return v___x_694_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_dual___redArg___lam__1___boxed(lean_object* v___f_695_, lean_object* v___f_696_, lean_object* v_f_697_, lean_object* v___y_698_){
_start:
{
lean_object* v_res_699_; 
v_res_699_ = lp_mathlib_LatticeHom_dual___redArg___lam__1(v___f_695_, v___f_696_, v_f_697_, v___y_698_);
lean_dec(v___f_696_);
lean_dec(v___f_695_);
return v_res_699_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_dual___redArg(lean_object* v_inst_700_, lean_object* v_inst_701_){
_start:
{
lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v_toSemilatticeSup_704_; lean_object* v_toSemilatticeSup_705_; lean_object* v___x_707_; uint8_t v_isShared_708_; uint8_t v_isSharedCheck_718_; 
lean_inc_ref(v_inst_700_);
v___x_702_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_inst_700_);
lean_inc_ref(v_inst_701_);
v___x_703_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v_inst_701_);
v_toSemilatticeSup_704_ = lean_ctor_get(v_inst_700_, 0);
lean_inc_ref(v_toSemilatticeSup_704_);
lean_dec_ref(v_inst_700_);
v_toSemilatticeSup_705_ = lean_ctor_get(v_inst_701_, 0);
v_isSharedCheck_718_ = !lean_is_exclusive(v_inst_701_);
if (v_isSharedCheck_718_ == 0)
{
lean_object* v_unused_719_; 
v_unused_719_ = lean_ctor_get(v_inst_701_, 1);
lean_dec(v_unused_719_);
v___x_707_ = v_inst_701_;
v_isShared_708_ = v_isSharedCheck_718_;
goto v_resetjp_706_;
}
else
{
lean_inc(v_toSemilatticeSup_705_);
lean_dec(v_inst_701_);
v___x_707_ = lean_box(0);
v_isShared_708_ = v_isSharedCheck_718_;
goto v_resetjp_706_;
}
v_resetjp_706_:
{
lean_object* v___f_709_; lean_object* v___f_710_; lean_object* v___f_711_; lean_object* v___f_712_; lean_object* v___f_713_; lean_object* v___f_714_; lean_object* v___x_716_; 
v___f_709_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_709_, 0, v___x_702_);
v___f_710_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toMin___redArg___lam__0), 3, 1);
lean_closure_set(v___f_710_, 0, v___x_703_);
v___f_711_ = lean_alloc_closure((void*)(lp_mathlib_LatticeHom_dual___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_711_, 0, v___f_709_);
lean_closure_set(v___f_711_, 1, v___f_710_);
v___f_712_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_712_, 0, v_toSemilatticeSup_704_);
v___f_713_ = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toMax___redArg___lam__0), 3, 1);
lean_closure_set(v___f_713_, 0, v_toSemilatticeSup_705_);
v___f_714_ = lean_alloc_closure((void*)(lp_mathlib_LatticeHom_dual___redArg___lam__1___boxed), 4, 2);
lean_closure_set(v___f_714_, 0, v___f_712_);
lean_closure_set(v___f_714_, 1, v___f_713_);
if (v_isShared_708_ == 0)
{
lean_ctor_set(v___x_707_, 1, v___f_714_);
lean_ctor_set(v___x_707_, 0, v___f_711_);
v___x_716_ = v___x_707_;
goto v_reusejp_715_;
}
else
{
lean_object* v_reuseFailAlloc_717_; 
v_reuseFailAlloc_717_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_717_, 0, v___f_711_);
lean_ctor_set(v_reuseFailAlloc_717_, 1, v___f_714_);
v___x_716_ = v_reuseFailAlloc_717_;
goto v_reusejp_715_;
}
v_reusejp_715_:
{
return v___x_716_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_dual(lean_object* v_00_u03b1_720_, lean_object* v_00_u03b2_721_, lean_object* v_inst_722_, lean_object* v_inst_723_){
_start:
{
lean_object* v___x_724_; 
v___x_724_ = lp_mathlib_LatticeHom_dual___redArg(v_inst_722_, v_inst_723_);
return v___x_724_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_fst___lam__0(lean_object* v_self_725_){
_start:
{
lean_object* v_fst_726_; 
v_fst_726_ = lean_ctor_get(v_self_725_, 0);
lean_inc(v_fst_726_);
return v_fst_726_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_fst___lam__0___boxed(lean_object* v_self_727_){
_start:
{
lean_object* v_res_728_; 
v_res_728_ = lp_mathlib_LatticeHom_fst___lam__0(v_self_727_);
lean_dec_ref(v_self_727_);
return v_res_728_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_fst(lean_object* v_00_u03b1_730_, lean_object* v_00_u03b2_731_, lean_object* v_inst_732_, lean_object* v_inst_733_){
_start:
{
lean_object* v___f_734_; 
v___f_734_ = ((lean_object*)(lp_mathlib_LatticeHom_fst___closed__0));
return v___f_734_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_fst___boxed(lean_object* v_00_u03b1_735_, lean_object* v_00_u03b2_736_, lean_object* v_inst_737_, lean_object* v_inst_738_){
_start:
{
lean_object* v_res_739_; 
v_res_739_ = lp_mathlib_LatticeHom_fst(v_00_u03b1_735_, v_00_u03b2_736_, v_inst_737_, v_inst_738_);
lean_dec_ref(v_inst_738_);
lean_dec_ref(v_inst_737_);
return v_res_739_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_snd___lam__0(lean_object* v_self_740_){
_start:
{
lean_object* v_snd_741_; 
v_snd_741_ = lean_ctor_get(v_self_740_, 1);
lean_inc(v_snd_741_);
return v_snd_741_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_snd___lam__0___boxed(lean_object* v_self_742_){
_start:
{
lean_object* v_res_743_; 
v_res_743_ = lp_mathlib_LatticeHom_snd___lam__0(v_self_742_);
lean_dec_ref(v_self_742_);
return v_res_743_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_snd(lean_object* v_00_u03b1_745_, lean_object* v_00_u03b2_746_, lean_object* v_inst_747_, lean_object* v_inst_748_){
_start:
{
lean_object* v___f_749_; 
v___f_749_ = ((lean_object*)(lp_mathlib_LatticeHom_snd___closed__0));
return v___f_749_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_snd___boxed(lean_object* v_00_u03b1_750_, lean_object* v_00_u03b2_751_, lean_object* v_inst_752_, lean_object* v_inst_753_){
_start:
{
lean_object* v_res_754_; 
v_res_754_ = lp_mathlib_LatticeHom_snd(v_00_u03b1_750_, v_00_u03b2_751_, v_inst_752_, v_inst_753_);
lean_dec_ref(v_inst_753_);
lean_dec_ref(v_inst_752_);
return v_res_754_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalLatticeHom___redArg(lean_object* v_i_755_){
_start:
{
lean_object* v___x_756_; 
v___x_756_ = lean_alloc_closure((void*)(lp_mathlib_Function_eval), 4, 3);
lean_closure_set(v___x_756_, 0, lean_box(0));
lean_closure_set(v___x_756_, 1, lean_box(0));
lean_closure_set(v___x_756_, 2, v_i_755_);
return v___x_756_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalLatticeHom(lean_object* v_00_u03b9_757_, lean_object* v_00_u03b1_758_, lean_object* v_inst_759_, lean_object* v_i_760_){
_start:
{
lean_object* v___x_761_; 
v___x_761_ = lean_alloc_closure((void*)(lp_mathlib_Function_eval), 4, 3);
lean_closure_set(v___x_761_, 0, lean_box(0));
lean_closure_set(v___x_761_, 1, lean_box(0));
lean_closure_set(v___x_761_, 2, v_i_760_);
return v___x_761_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_evalLatticeHom___boxed(lean_object* v_00_u03b9_762_, lean_object* v_00_u03b1_763_, lean_object* v_inst_764_, lean_object* v_i_765_){
_start:
{
lean_object* v_res_766_; 
v_res_766_ = lp_mathlib_Pi_evalLatticeHom(v_00_u03b9_762_, v_00_u03b1_763_, v_inst_764_, v_i_765_);
lean_dec_ref(v_inst_764_);
return v_res_766_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_Lattice(uint8_t builtin) {
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
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Hom_Lattice(uint8_t builtin) {
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
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Hom_Lattice(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Order_Hom_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Hom_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Hom_Lattice(builtin);
}
#ifdef __cplusplus
}
#endif
