// Lean compiler output
// Module: Mathlib.RingTheory.GradedAlgebra.Homogeneous.Ideal
// Imports: public import Init public meta import Init public import Mathlib.LinearAlgebra.Finsupp.SumProd public import Mathlib.RingTheory.GradedAlgebra.Basic public import Mathlib.RingTheory.Ideal.Basic public import Mathlib.RingTheory.Ideal.BigOperators public import Mathlib.RingTheory.Ideal.Maps public import Mathlib.RingTheory.GradedAlgebra.Homogeneous.Submodule
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
lean_object* lp_mathlib_PartialOrder_ofSetLike(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toModule___redArg(lean_object*);
lean_object* lp_mathlib_Submodule_completeLattice___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(lean_object*);
lean_object* lp_mathlib_ConditionallyCompleteLattice_toConditionallyCompletePartialOrder___redArg(lean_object*);
lean_object* lp_mathlib_Submodule_instInfSet___lam__0(lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddSubmonoid_instCompleteLattice___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_toIdeal___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_toIdeal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_toIdeal___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_setLike(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_setLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_instPartialOrderHomogeneousIdeal___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instPartialOrderHomogeneousIdeal___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_instPartialOrderHomogeneousIdeal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instPartialOrderHomogeneousIdeal___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_homogeneousCore_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_homogeneousCore_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_homogeneousCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_homogeneousCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instMax___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_HomogeneousIdeal_instMax___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_HomogeneousIdeal_instMax___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_HomogeneousIdeal_instMax___closed__0 = (const lean_object*)&lp_mathlib_HomogeneousIdeal_instMax___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instMax(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instMax___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instMin(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instMin___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instSupSet___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instSupSet___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instSupSet___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instSupSet(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instSupSet___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_HomogeneousIdeal_instInfSet___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_HomogeneousIdeal_instInfSet___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instInfSet___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_HomogeneousIdeal_instInfSet___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_HomogeneousIdeal_instInfSet___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_HomogeneousIdeal_instInfSet___closed__0 = (const lean_object*)&lp_mathlib_HomogeneousIdeal_instInfSet___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instInfSet(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instInfSet___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_completeLattice___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_HomogeneousIdeal_completeLattice___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_HomogeneousIdeal_completeLattice___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_HomogeneousIdeal_completeLattice___redArg___closed__0 = (const lean_object*)&lp_mathlib_HomogeneousIdeal_completeLattice___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_HomogeneousIdeal_completeLattice___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_HomogeneousIdeal_completeLattice___redArg___closed__1 = (const lean_object*)&lp_mathlib_HomogeneousIdeal_completeLattice___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_completeLattice___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_completeLattice___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_completeLattice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_completeLattice___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instAdd___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_HomogeneousIdeal_instAdd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_HomogeneousIdeal_instAdd___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_HomogeneousIdeal_instAdd___closed__0 = (const lean_object*)&lp_mathlib_HomogeneousIdeal_instAdd___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instAdd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instAdd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instInhabited(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instInhabited___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulHomogeneousIdeal___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulHomogeneousIdeal___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulHomogeneousIdeal___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulHomogeneousIdeal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instMulHomogeneousIdeal___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_homogeneousCore_gi___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Ideal_homogeneousCore_gi___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Ideal_homogeneousCore_gi___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Ideal_homogeneousCore_gi___closed__0 = (const lean_object*)&lp_mathlib_Ideal_homogeneousCore_gi___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Ideal_homogeneousCore_gi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_homogeneousCore_gi___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_homogeneousHull(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_homogeneousHull___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_homogeneousHull_gi___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Ideal_homogeneousHull_gi___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Ideal_homogeneousHull_gi___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Ideal_homogeneousHull_gi___closed__0 = (const lean_object*)&lp_mathlib_Ideal_homogeneousHull_gi___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Ideal_homogeneousHull_gi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_homogeneousHull_gi___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_irrelevant(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_irrelevant___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_HomogeneousIdeal_term___u208a___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "HomogeneousIdeal"};
static const lean_object* lp_mathlib_HomogeneousIdeal_term___u208a___closed__0 = (const lean_object*)&lp_mathlib_HomogeneousIdeal_term___u208a___closed__0_value;
static const lean_string_object lp_mathlib_HomogeneousIdeal_term___u208a___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 6, .m_data = "term_₊"};
static const lean_object* lp_mathlib_HomogeneousIdeal_term___u208a___closed__1 = (const lean_object*)&lp_mathlib_HomogeneousIdeal_term___u208a___closed__1_value;
static const lean_ctor_object lp_mathlib_HomogeneousIdeal_term___u208a___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_HomogeneousIdeal_term___u208a___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 183, 136, 92, 189, 172, 82, 199)}};
static const lean_ctor_object lp_mathlib_HomogeneousIdeal_term___u208a___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_HomogeneousIdeal_term___u208a___closed__2_value_aux_0),((lean_object*)&lp_mathlib_HomogeneousIdeal_term___u208a___closed__1_value),LEAN_SCALAR_PTR_LITERAL(1, 44, 151, 233, 191, 54, 83, 156)}};
static const lean_object* lp_mathlib_HomogeneousIdeal_term___u208a___closed__2 = (const lean_object*)&lp_mathlib_HomogeneousIdeal_term___u208a___closed__2_value;
static const lean_string_object lp_mathlib_HomogeneousIdeal_term___u208a___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "₊"};
static const lean_object* lp_mathlib_HomogeneousIdeal_term___u208a___closed__3 = (const lean_object*)&lp_mathlib_HomogeneousIdeal_term___u208a___closed__3_value;
static const lean_ctor_object lp_mathlib_HomogeneousIdeal_term___u208a___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_HomogeneousIdeal_term___u208a___closed__3_value)}};
static const lean_object* lp_mathlib_HomogeneousIdeal_term___u208a___closed__4 = (const lean_object*)&lp_mathlib_HomogeneousIdeal_term___u208a___closed__4_value;
static const lean_ctor_object lp_mathlib_HomogeneousIdeal_term___u208a___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_HomogeneousIdeal_term___u208a___closed__2_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_HomogeneousIdeal_term___u208a___closed__4_value)}};
static const lean_object* lp_mathlib_HomogeneousIdeal_term___u208a___closed__5 = (const lean_object*)&lp_mathlib_HomogeneousIdeal_term___u208a___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_HomogeneousIdeal_term___u208a = (const lean_object*)&lp_mathlib_HomogeneousIdeal_term___u208a___closed__5_value;
static const lean_string_object lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__0 = (const lean_object*)&lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__0_value;
static const lean_string_object lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__1 = (const lean_object*)&lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__1_value;
static const lean_string_object lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__2 = (const lean_object*)&lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__2_value;
static const lean_string_object lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__3 = (const lean_object*)&lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__3_value;
static const lean_ctor_object lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__4 = (const lean_object*)&lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__4_value;
static const lean_string_object lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "irrelevant"};
static const lean_object* lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__5 = (const lean_object*)&lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__5_value;
static lean_once_cell_t lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__6;
static const lean_ctor_object lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(243, 200, 210, 146, 146, 240, 169, 142)}};
static const lean_object* lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__7 = (const lean_object*)&lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__7_value;
static const lean_ctor_object lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_HomogeneousIdeal_term___u208a___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 183, 136, 92, 189, 172, 82, 199)}};
static const lean_ctor_object lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(162, 182, 201, 36, 241, 216, 212, 101)}};
static const lean_object* lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__8 = (const lean_object*)&lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__8_value;
static const lean_ctor_object lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__9 = (const lean_object*)&lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__9_value;
static const lean_ctor_object lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__10 = (const lean_object*)&lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__10_value;
static const lean_string_object lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__11 = (const lean_object*)&lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__11_value;
static const lean_ctor_object lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__12 = (const lean_object*)&lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______unexpand__HomogeneousIdeal__irrelevant__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______unexpand__HomogeneousIdeal__irrelevant__1___closed__0 = (const lean_object*)&lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______unexpand__HomogeneousIdeal__irrelevant__1___closed__0_value;
static const lean_ctor_object lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______unexpand__HomogeneousIdeal__irrelevant__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______unexpand__HomogeneousIdeal__irrelevant__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______unexpand__HomogeneousIdeal__irrelevant__1___closed__1 = (const lean_object*)&lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______unexpand__HomogeneousIdeal__irrelevant__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______unexpand__HomogeneousIdeal__irrelevant__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______unexpand__HomogeneousIdeal__irrelevant__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_toIdeal___redArg(lean_object* v_I_1_){
_start:
{
return v_I_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_toIdeal(lean_object* v_00_u03b9_2_, lean_object* v_00_u03c3_3_, lean_object* v_A_4_, lean_object* v_inst_5_, lean_object* v_inst_6_, lean_object* v_inst_7_, lean_object* v_00_U0001d49c_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_I_12_){
_start:
{
return v_I_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_toIdeal___boxed(lean_object* v_00_u03b9_13_, lean_object* v_00_u03c3_14_, lean_object* v_A_15_, lean_object* v_inst_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_00_U0001d49c_19_, lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_I_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_HomogeneousIdeal_toIdeal(v_00_u03b9_13_, v_00_u03c3_14_, v_A_15_, v_inst_16_, v_inst_17_, v_inst_18_, v_00_U0001d49c_19_, v_inst_20_, v_inst_21_, v_inst_22_, v_I_23_);
lean_dec_ref(v_inst_22_);
lean_dec_ref(v_inst_21_);
lean_dec_ref(v_inst_20_);
lean_dec(v_00_U0001d49c_19_);
lean_dec_ref(v_inst_16_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_setLike(lean_object* v_00_u03b9_25_, lean_object* v_00_u03c3_26_, lean_object* v_A_27_, lean_object* v_inst_28_, lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_00_U0001d49c_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_inst_34_){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = lean_box(0);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_setLike___boxed(lean_object* v_00_u03b9_36_, lean_object* v_00_u03c3_37_, lean_object* v_A_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_inst_41_, lean_object* v_00_U0001d49c_42_, lean_object* v_inst_43_, lean_object* v_inst_44_, lean_object* v_inst_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_HomogeneousIdeal_setLike(v_00_u03b9_36_, v_00_u03c3_37_, v_A_38_, v_inst_39_, v_inst_40_, v_inst_41_, v_00_U0001d49c_42_, v_inst_43_, v_inst_44_, v_inst_45_);
lean_dec_ref(v_inst_45_);
lean_dec_ref(v_inst_44_);
lean_dec_ref(v_inst_43_);
lean_dec(v_00_U0001d49c_42_);
lean_dec_ref(v_inst_39_);
return v_res_46_;
}
}
static lean_object* _init_lp_mathlib_instPartialOrderHomogeneousIdeal___closed__0(void){
_start:
{
lean_object* v___x_47_; lean_object* v___x_48_; 
v___x_47_ = lean_box(0);
v___x_48_ = lp_mathlib_PartialOrder_ofSetLike(lean_box(0), lean_box(0), v___x_47_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instPartialOrderHomogeneousIdeal(lean_object* v_00_u03b9_49_, lean_object* v_00_u03c3_50_, lean_object* v_A_51_, lean_object* v_inst_52_, lean_object* v_inst_53_, lean_object* v_inst_54_, lean_object* v_00_U0001d49c_55_, lean_object* v_inst_56_, lean_object* v_inst_57_, lean_object* v_inst_58_){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = lean_obj_once(&lp_mathlib_instPartialOrderHomogeneousIdeal___closed__0, &lp_mathlib_instPartialOrderHomogeneousIdeal___closed__0_once, _init_lp_mathlib_instPartialOrderHomogeneousIdeal___closed__0);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instPartialOrderHomogeneousIdeal___boxed(lean_object* v_00_u03b9_60_, lean_object* v_00_u03c3_61_, lean_object* v_A_62_, lean_object* v_inst_63_, lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_00_U0001d49c_66_, lean_object* v_inst_67_, lean_object* v_inst_68_, lean_object* v_inst_69_){
_start:
{
lean_object* v_res_70_; 
v_res_70_ = lp_mathlib_instPartialOrderHomogeneousIdeal(v_00_u03b9_60_, v_00_u03c3_61_, v_A_62_, v_inst_63_, v_inst_64_, v_inst_65_, v_00_U0001d49c_66_, v_inst_67_, v_inst_68_, v_inst_69_);
lean_dec_ref(v_inst_69_);
lean_dec_ref(v_inst_68_);
lean_dec_ref(v_inst_67_);
lean_dec(v_00_U0001d49c_66_);
lean_dec_ref(v_inst_63_);
return v_res_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_homogeneousCore_x27(lean_object* v_00_u03b9_71_, lean_object* v_00_u03c3_72_, lean_object* v_A_73_, lean_object* v_inst_74_, lean_object* v_inst_75_, lean_object* v_00_U0001d49c_76_, lean_object* v_I_77_){
_start:
{
lean_object* v___x_78_; 
v___x_78_ = lean_box(0);
return v___x_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_homogeneousCore_x27___boxed(lean_object* v_00_u03b9_79_, lean_object* v_00_u03c3_80_, lean_object* v_A_81_, lean_object* v_inst_82_, lean_object* v_inst_83_, lean_object* v_00_U0001d49c_84_, lean_object* v_I_85_){
_start:
{
lean_object* v_res_86_; 
v_res_86_ = lp_mathlib_Ideal_homogeneousCore_x27(v_00_u03b9_79_, v_00_u03c3_80_, v_A_81_, v_inst_82_, v_inst_83_, v_00_U0001d49c_84_, v_I_85_);
lean_dec(v_00_U0001d49c_84_);
lean_dec_ref(v_inst_82_);
return v_res_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_homogeneousCore(lean_object* v_00_u03b9_87_, lean_object* v_00_u03c3_88_, lean_object* v_A_89_, lean_object* v_inst_90_, lean_object* v_inst_91_, lean_object* v_inst_92_, lean_object* v_00_U0001d49c_93_, lean_object* v_inst_94_, lean_object* v_inst_95_, lean_object* v_inst_96_, lean_object* v_I_97_){
_start:
{
lean_object* v___x_98_; 
v___x_98_ = lean_box(0);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_homogeneousCore___boxed(lean_object* v_00_u03b9_99_, lean_object* v_00_u03c3_100_, lean_object* v_A_101_, lean_object* v_inst_102_, lean_object* v_inst_103_, lean_object* v_inst_104_, lean_object* v_00_U0001d49c_105_, lean_object* v_inst_106_, lean_object* v_inst_107_, lean_object* v_inst_108_, lean_object* v_I_109_){
_start:
{
lean_object* v_res_110_; 
v_res_110_ = lp_mathlib_Ideal_homogeneousCore(v_00_u03b9_99_, v_00_u03c3_100_, v_A_101_, v_inst_102_, v_inst_103_, v_inst_104_, v_00_U0001d49c_105_, v_inst_106_, v_inst_107_, v_inst_108_, v_I_109_);
lean_dec_ref(v_inst_108_);
lean_dec_ref(v_inst_107_);
lean_dec_ref(v_inst_106_);
lean_dec(v_00_U0001d49c_105_);
lean_dec_ref(v_inst_102_);
return v_res_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instTop(lean_object* v_00_u03b9_111_, lean_object* v_00_u03c3_112_, lean_object* v_A_113_, lean_object* v_inst_114_, lean_object* v_inst_115_, lean_object* v_inst_116_, lean_object* v_inst_117_, lean_object* v_inst_118_, lean_object* v_00_U0001d49c_119_, lean_object* v_inst_120_){
_start:
{
lean_object* v___x_121_; 
v___x_121_ = lean_box(0);
return v___x_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instTop___boxed(lean_object* v_00_u03b9_122_, lean_object* v_00_u03c3_123_, lean_object* v_A_124_, lean_object* v_inst_125_, lean_object* v_inst_126_, lean_object* v_inst_127_, lean_object* v_inst_128_, lean_object* v_inst_129_, lean_object* v_00_U0001d49c_130_, lean_object* v_inst_131_){
_start:
{
lean_object* v_res_132_; 
v_res_132_ = lp_mathlib_HomogeneousIdeal_instTop(v_00_u03b9_122_, v_00_u03c3_123_, v_A_124_, v_inst_125_, v_inst_126_, v_inst_127_, v_inst_128_, v_inst_129_, v_00_U0001d49c_130_, v_inst_131_);
lean_dec_ref(v_inst_131_);
lean_dec(v_00_U0001d49c_130_);
lean_dec_ref(v_inst_127_);
lean_dec_ref(v_inst_126_);
lean_dec_ref(v_inst_125_);
return v_res_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instBot(lean_object* v_00_u03b9_133_, lean_object* v_00_u03c3_134_, lean_object* v_A_135_, lean_object* v_inst_136_, lean_object* v_inst_137_, lean_object* v_inst_138_, lean_object* v_inst_139_, lean_object* v_inst_140_, lean_object* v_00_U0001d49c_141_, lean_object* v_inst_142_){
_start:
{
lean_object* v___x_143_; 
v___x_143_ = lean_box(0);
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instBot___boxed(lean_object* v_00_u03b9_144_, lean_object* v_00_u03c3_145_, lean_object* v_A_146_, lean_object* v_inst_147_, lean_object* v_inst_148_, lean_object* v_inst_149_, lean_object* v_inst_150_, lean_object* v_inst_151_, lean_object* v_00_U0001d49c_152_, lean_object* v_inst_153_){
_start:
{
lean_object* v_res_154_; 
v_res_154_ = lp_mathlib_HomogeneousIdeal_instBot(v_00_u03b9_144_, v_00_u03c3_145_, v_A_146_, v_inst_147_, v_inst_148_, v_inst_149_, v_inst_150_, v_inst_151_, v_00_U0001d49c_152_, v_inst_153_);
lean_dec_ref(v_inst_153_);
lean_dec(v_00_U0001d49c_152_);
lean_dec_ref(v_inst_149_);
lean_dec_ref(v_inst_148_);
lean_dec_ref(v_inst_147_);
return v_res_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instMax___lam__0(lean_object* v_I_155_, lean_object* v_J_156_){
_start:
{
lean_object* v___x_157_; 
v___x_157_ = lean_box(0);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instMax(lean_object* v_00_u03b9_159_, lean_object* v_00_u03c3_160_, lean_object* v_A_161_, lean_object* v_inst_162_, lean_object* v_inst_163_, lean_object* v_inst_164_, lean_object* v_inst_165_, lean_object* v_inst_166_, lean_object* v_00_U0001d49c_167_, lean_object* v_inst_168_){
_start:
{
lean_object* v___f_169_; 
v___f_169_ = ((lean_object*)(lp_mathlib_HomogeneousIdeal_instMax___closed__0));
return v___f_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instMax___boxed(lean_object* v_00_u03b9_170_, lean_object* v_00_u03c3_171_, lean_object* v_A_172_, lean_object* v_inst_173_, lean_object* v_inst_174_, lean_object* v_inst_175_, lean_object* v_inst_176_, lean_object* v_inst_177_, lean_object* v_00_U0001d49c_178_, lean_object* v_inst_179_){
_start:
{
lean_object* v_res_180_; 
v_res_180_ = lp_mathlib_HomogeneousIdeal_instMax(v_00_u03b9_170_, v_00_u03c3_171_, v_A_172_, v_inst_173_, v_inst_174_, v_inst_175_, v_inst_176_, v_inst_177_, v_00_U0001d49c_178_, v_inst_179_);
lean_dec_ref(v_inst_179_);
lean_dec(v_00_U0001d49c_178_);
lean_dec_ref(v_inst_175_);
lean_dec_ref(v_inst_174_);
lean_dec_ref(v_inst_173_);
return v_res_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instMin(lean_object* v_00_u03b9_181_, lean_object* v_00_u03c3_182_, lean_object* v_A_183_, lean_object* v_inst_184_, lean_object* v_inst_185_, lean_object* v_inst_186_, lean_object* v_inst_187_, lean_object* v_inst_188_, lean_object* v_00_U0001d49c_189_, lean_object* v_inst_190_){
_start:
{
lean_object* v___f_191_; 
v___f_191_ = ((lean_object*)(lp_mathlib_HomogeneousIdeal_instMax___closed__0));
return v___f_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instMin___boxed(lean_object* v_00_u03b9_192_, lean_object* v_00_u03c3_193_, lean_object* v_A_194_, lean_object* v_inst_195_, lean_object* v_inst_196_, lean_object* v_inst_197_, lean_object* v_inst_198_, lean_object* v_inst_199_, lean_object* v_00_U0001d49c_200_, lean_object* v_inst_201_){
_start:
{
lean_object* v_res_202_; 
v_res_202_ = lp_mathlib_HomogeneousIdeal_instMin(v_00_u03b9_192_, v_00_u03c3_193_, v_A_194_, v_inst_195_, v_inst_196_, v_inst_197_, v_inst_198_, v_inst_199_, v_00_U0001d49c_200_, v_inst_201_);
lean_dec_ref(v_inst_201_);
lean_dec(v_00_U0001d49c_200_);
lean_dec_ref(v_inst_197_);
lean_dec_ref(v_inst_196_);
lean_dec_ref(v_inst_195_);
return v_res_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instSupSet___redArg___lam__0(lean_object* v_toSupSet_203_, lean_object* v_S_204_){
_start:
{
lean_object* v___x_205_; 
v___x_205_ = lean_apply_1(v_toSupSet_203_, lean_box(0));
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instSupSet___redArg(lean_object* v_inst_206_){
_start:
{
lean_object* v_toAddCommMonoid_207_; lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v_toConditionallyCompletePartialOrderSup_212_; lean_object* v_toSupSet_213_; lean_object* v___f_214_; 
v_toAddCommMonoid_207_ = lean_ctor_get(v_inst_206_, 0);
v___x_208_ = lp_mathlib_Semiring_toModule___redArg(v_inst_206_);
v___x_209_ = lp_mathlib_Submodule_completeLattice___redArg(v_inst_206_, v_toAddCommMonoid_207_, v___x_208_);
lean_dec(v___x_208_);
v___x_210_ = lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(v___x_209_);
v___x_211_ = lp_mathlib_ConditionallyCompleteLattice_toConditionallyCompletePartialOrder___redArg(v___x_210_);
v_toConditionallyCompletePartialOrderSup_212_ = lean_ctor_get(v___x_211_, 0);
lean_inc_ref(v_toConditionallyCompletePartialOrderSup_212_);
lean_dec_ref(v___x_211_);
v_toSupSet_213_ = lean_ctor_get(v_toConditionallyCompletePartialOrderSup_212_, 1);
lean_inc(v_toSupSet_213_);
lean_dec_ref(v_toConditionallyCompletePartialOrderSup_212_);
v___f_214_ = lean_alloc_closure((void*)(lp_mathlib_HomogeneousIdeal_instSupSet___redArg___lam__0), 2, 1);
lean_closure_set(v___f_214_, 0, v_toSupSet_213_);
return v___f_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instSupSet___redArg___boxed(lean_object* v_inst_215_){
_start:
{
lean_object* v_res_216_; 
v_res_216_ = lp_mathlib_HomogeneousIdeal_instSupSet___redArg(v_inst_215_);
lean_dec_ref(v_inst_215_);
return v_res_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instSupSet(lean_object* v_00_u03b9_217_, lean_object* v_00_u03c3_218_, lean_object* v_A_219_, lean_object* v_inst_220_, lean_object* v_inst_221_, lean_object* v_inst_222_, lean_object* v_inst_223_, lean_object* v_inst_224_, lean_object* v_00_U0001d49c_225_, lean_object* v_inst_226_){
_start:
{
lean_object* v___x_227_; 
v___x_227_ = lp_mathlib_HomogeneousIdeal_instSupSet___redArg(v_inst_220_);
return v___x_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instSupSet___boxed(lean_object* v_00_u03b9_228_, lean_object* v_00_u03c3_229_, lean_object* v_A_230_, lean_object* v_inst_231_, lean_object* v_inst_232_, lean_object* v_inst_233_, lean_object* v_inst_234_, lean_object* v_inst_235_, lean_object* v_00_U0001d49c_236_, lean_object* v_inst_237_){
_start:
{
lean_object* v_res_238_; 
v_res_238_ = lp_mathlib_HomogeneousIdeal_instSupSet(v_00_u03b9_228_, v_00_u03c3_229_, v_A_230_, v_inst_231_, v_inst_232_, v_inst_233_, v_inst_234_, v_inst_235_, v_00_U0001d49c_236_, v_inst_237_);
lean_dec_ref(v_inst_237_);
lean_dec(v_00_U0001d49c_236_);
lean_dec_ref(v_inst_233_);
lean_dec_ref(v_inst_232_);
lean_dec_ref(v_inst_231_);
return v_res_238_;
}
}
static lean_object* _init_lp_mathlib_HomogeneousIdeal_instInfSet___lam__0___closed__0(void){
_start:
{
lean_object* v___x_239_; 
v___x_239_ = lp_mathlib_Submodule_instInfSet___lam__0(lean_box(0));
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instInfSet___lam__0(lean_object* v_S_240_){
_start:
{
lean_object* v___x_241_; 
v___x_241_ = lean_obj_once(&lp_mathlib_HomogeneousIdeal_instInfSet___lam__0___closed__0, &lp_mathlib_HomogeneousIdeal_instInfSet___lam__0___closed__0_once, _init_lp_mathlib_HomogeneousIdeal_instInfSet___lam__0___closed__0);
return v___x_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instInfSet(lean_object* v_00_u03b9_243_, lean_object* v_00_u03c3_244_, lean_object* v_A_245_, lean_object* v_inst_246_, lean_object* v_inst_247_, lean_object* v_inst_248_, lean_object* v_inst_249_, lean_object* v_inst_250_, lean_object* v_00_U0001d49c_251_, lean_object* v_inst_252_){
_start:
{
lean_object* v___f_253_; 
v___f_253_ = ((lean_object*)(lp_mathlib_HomogeneousIdeal_instInfSet___closed__0));
return v___f_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instInfSet___boxed(lean_object* v_00_u03b9_254_, lean_object* v_00_u03c3_255_, lean_object* v_A_256_, lean_object* v_inst_257_, lean_object* v_inst_258_, lean_object* v_inst_259_, lean_object* v_inst_260_, lean_object* v_inst_261_, lean_object* v_00_U0001d49c_262_, lean_object* v_inst_263_){
_start:
{
lean_object* v_res_264_; 
v_res_264_ = lp_mathlib_HomogeneousIdeal_instInfSet(v_00_u03b9_254_, v_00_u03c3_255_, v_A_256_, v_inst_257_, v_inst_258_, v_inst_259_, v_inst_260_, v_inst_261_, v_00_U0001d49c_262_, v_inst_263_);
lean_dec_ref(v_inst_263_);
lean_dec(v_00_U0001d49c_262_);
lean_dec_ref(v_inst_259_);
lean_dec_ref(v_inst_258_);
lean_dec_ref(v_inst_257_);
return v_res_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_completeLattice___redArg___lam__0(lean_object* v_a_265_, lean_object* v_b_266_){
_start:
{
lean_object* v___x_267_; 
v___x_267_ = lean_box(0);
return v___x_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_completeLattice___redArg(lean_object* v_inst_271_, lean_object* v_inst_272_, lean_object* v_inst_273_, lean_object* v_inst_274_, lean_object* v_00_U0001d49c_275_, lean_object* v_inst_276_){
_start:
{
lean_object* v___x_277_; lean_object* v_toLE_278_; lean_object* v_toLT_279_; lean_object* v___x_281_; uint8_t v_isShared_282_; uint8_t v_isSharedCheck_293_; 
v___x_277_ = lp_mathlib_instPartialOrderHomogeneousIdeal(lean_box(0), lean_box(0), lean_box(0), v_inst_271_, v_inst_274_, lean_box(0), v_00_U0001d49c_275_, v_inst_272_, v_inst_273_, v_inst_276_);
v_toLE_278_ = lean_ctor_get(v___x_277_, 0);
v_toLT_279_ = lean_ctor_get(v___x_277_, 1);
v_isSharedCheck_293_ = !lean_is_exclusive(v___x_277_);
if (v_isSharedCheck_293_ == 0)
{
v___x_281_ = v___x_277_;
v_isShared_282_ = v_isSharedCheck_293_;
goto v_resetjp_280_;
}
else
{
lean_inc(v_toLT_279_);
lean_inc(v_toLE_278_);
lean_dec(v___x_277_);
v___x_281_ = lean_box(0);
v_isShared_282_ = v_isSharedCheck_293_;
goto v_resetjp_280_;
}
v_resetjp_280_:
{
lean_object* v___f_283_; lean_object* v___x_284_; lean_object* v___f_285_; lean_object* v___x_287_; 
v___f_283_ = ((lean_object*)(lp_mathlib_HomogeneousIdeal_completeLattice___redArg___closed__0));
v___x_284_ = lp_mathlib_HomogeneousIdeal_instSupSet___redArg(v_inst_271_);
v___f_285_ = ((lean_object*)(lp_mathlib_HomogeneousIdeal_instInfSet___closed__0));
if (v_isShared_282_ == 0)
{
v___x_287_ = v___x_281_;
goto v_reusejp_286_;
}
else
{
lean_object* v_reuseFailAlloc_292_; 
v_reuseFailAlloc_292_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_292_, 0, v_toLE_278_);
lean_ctor_set(v_reuseFailAlloc_292_, 1, v_toLT_279_);
v___x_287_ = v_reuseFailAlloc_292_;
goto v_reusejp_286_;
}
v_reusejp_286_:
{
lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; 
v___x_288_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_288_, 0, v___x_287_);
lean_ctor_set(v___x_288_, 1, v___f_283_);
v___x_289_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_289_, 0, v___x_288_);
lean_ctor_set(v___x_289_, 1, v___f_283_);
v___x_290_ = ((lean_object*)(lp_mathlib_HomogeneousIdeal_completeLattice___redArg___closed__1));
v___x_291_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_291_, 0, v___x_289_);
lean_ctor_set(v___x_291_, 1, v___x_284_);
lean_ctor_set(v___x_291_, 2, v___f_285_);
lean_ctor_set(v___x_291_, 3, v___x_290_);
return v___x_291_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_completeLattice___redArg___boxed(lean_object* v_inst_294_, lean_object* v_inst_295_, lean_object* v_inst_296_, lean_object* v_inst_297_, lean_object* v_00_U0001d49c_298_, lean_object* v_inst_299_){
_start:
{
lean_object* v_res_300_; 
v_res_300_ = lp_mathlib_HomogeneousIdeal_completeLattice___redArg(v_inst_294_, v_inst_295_, v_inst_296_, v_inst_297_, v_00_U0001d49c_298_, v_inst_299_);
lean_dec_ref(v_inst_299_);
lean_dec(v_00_U0001d49c_298_);
lean_dec_ref(v_inst_296_);
lean_dec_ref(v_inst_295_);
lean_dec_ref(v_inst_294_);
return v_res_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_completeLattice(lean_object* v_00_u03b9_301_, lean_object* v_00_u03c3_302_, lean_object* v_A_303_, lean_object* v_inst_304_, lean_object* v_inst_305_, lean_object* v_inst_306_, lean_object* v_inst_307_, lean_object* v_inst_308_, lean_object* v_00_U0001d49c_309_, lean_object* v_inst_310_){
_start:
{
lean_object* v___x_311_; 
v___x_311_ = lp_mathlib_HomogeneousIdeal_completeLattice___redArg(v_inst_304_, v_inst_305_, v_inst_306_, v_inst_307_, v_00_U0001d49c_309_, v_inst_310_);
return v___x_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_completeLattice___boxed(lean_object* v_00_u03b9_312_, lean_object* v_00_u03c3_313_, lean_object* v_A_314_, lean_object* v_inst_315_, lean_object* v_inst_316_, lean_object* v_inst_317_, lean_object* v_inst_318_, lean_object* v_inst_319_, lean_object* v_00_U0001d49c_320_, lean_object* v_inst_321_){
_start:
{
lean_object* v_res_322_; 
v_res_322_ = lp_mathlib_HomogeneousIdeal_completeLattice(v_00_u03b9_312_, v_00_u03c3_313_, v_A_314_, v_inst_315_, v_inst_316_, v_inst_317_, v_inst_318_, v_inst_319_, v_00_U0001d49c_320_, v_inst_321_);
lean_dec_ref(v_inst_321_);
lean_dec(v_00_U0001d49c_320_);
lean_dec_ref(v_inst_317_);
lean_dec_ref(v_inst_316_);
lean_dec_ref(v_inst_315_);
return v_res_322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instAdd___lam__0(lean_object* v_x1_323_, lean_object* v_x2_324_){
_start:
{
lean_object* v___x_325_; 
v___x_325_ = lean_box(0);
return v___x_325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instAdd(lean_object* v_00_u03b9_327_, lean_object* v_00_u03c3_328_, lean_object* v_A_329_, lean_object* v_inst_330_, lean_object* v_inst_331_, lean_object* v_inst_332_, lean_object* v_inst_333_, lean_object* v_inst_334_, lean_object* v_00_U0001d49c_335_, lean_object* v_inst_336_){
_start:
{
lean_object* v___f_337_; 
v___f_337_ = ((lean_object*)(lp_mathlib_HomogeneousIdeal_instAdd___closed__0));
return v___f_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instAdd___boxed(lean_object* v_00_u03b9_338_, lean_object* v_00_u03c3_339_, lean_object* v_A_340_, lean_object* v_inst_341_, lean_object* v_inst_342_, lean_object* v_inst_343_, lean_object* v_inst_344_, lean_object* v_inst_345_, lean_object* v_00_U0001d49c_346_, lean_object* v_inst_347_){
_start:
{
lean_object* v_res_348_; 
v_res_348_ = lp_mathlib_HomogeneousIdeal_instAdd(v_00_u03b9_338_, v_00_u03c3_339_, v_A_340_, v_inst_341_, v_inst_342_, v_inst_343_, v_inst_344_, v_inst_345_, v_00_U0001d49c_346_, v_inst_347_);
lean_dec_ref(v_inst_347_);
lean_dec(v_00_U0001d49c_346_);
lean_dec_ref(v_inst_343_);
lean_dec_ref(v_inst_342_);
lean_dec_ref(v_inst_341_);
return v_res_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instInhabited(lean_object* v_00_u03b9_349_, lean_object* v_00_u03c3_350_, lean_object* v_A_351_, lean_object* v_inst_352_, lean_object* v_inst_353_, lean_object* v_inst_354_, lean_object* v_inst_355_, lean_object* v_inst_356_, lean_object* v_00_U0001d49c_357_, lean_object* v_inst_358_){
_start:
{
lean_object* v___x_359_; 
v___x_359_ = lean_box(0);
return v___x_359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_instInhabited___boxed(lean_object* v_00_u03b9_360_, lean_object* v_00_u03c3_361_, lean_object* v_A_362_, lean_object* v_inst_363_, lean_object* v_inst_364_, lean_object* v_inst_365_, lean_object* v_inst_366_, lean_object* v_inst_367_, lean_object* v_00_U0001d49c_368_, lean_object* v_inst_369_){
_start:
{
lean_object* v_res_370_; 
v_res_370_ = lp_mathlib_HomogeneousIdeal_instInhabited(v_00_u03b9_360_, v_00_u03c3_361_, v_A_362_, v_inst_363_, v_inst_364_, v_inst_365_, v_inst_366_, v_inst_367_, v_00_U0001d49c_368_, v_inst_369_);
lean_dec_ref(v_inst_369_);
lean_dec(v_00_U0001d49c_368_);
lean_dec_ref(v_inst_365_);
lean_dec_ref(v_inst_364_);
lean_dec_ref(v_inst_363_);
return v_res_370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulHomogeneousIdeal___redArg___lam__0(lean_object* v_toAddCommMonoid_371_, lean_object* v_I_372_, lean_object* v_J_373_){
_start:
{
lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v_toConditionallyCompletePartialOrderSup_378_; lean_object* v_toSupSet_379_; lean_object* v___x_380_; 
v___x_374_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_toAddCommMonoid_371_);
v___x_375_ = lp_mathlib_AddSubmonoid_instCompleteLattice___redArg(v___x_374_);
lean_dec_ref(v___x_374_);
v___x_376_ = lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(v___x_375_);
v___x_377_ = lp_mathlib_ConditionallyCompleteLattice_toConditionallyCompletePartialOrder___redArg(v___x_376_);
v_toConditionallyCompletePartialOrderSup_378_ = lean_ctor_get(v___x_377_, 0);
lean_inc_ref(v_toConditionallyCompletePartialOrderSup_378_);
lean_dec_ref(v___x_377_);
v_toSupSet_379_ = lean_ctor_get(v_toConditionallyCompletePartialOrderSup_378_, 1);
lean_inc(v_toSupSet_379_);
lean_dec_ref(v_toConditionallyCompletePartialOrderSup_378_);
v___x_380_ = lean_apply_1(v_toSupSet_379_, lean_box(0));
return v___x_380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulHomogeneousIdeal___redArg___lam__0___boxed(lean_object* v_toAddCommMonoid_381_, lean_object* v_I_382_, lean_object* v_J_383_){
_start:
{
lean_object* v_res_384_; 
v_res_384_ = lp_mathlib_instMulHomogeneousIdeal___redArg___lam__0(v_toAddCommMonoid_381_, v_I_382_, v_J_383_);
lean_dec_ref(v_toAddCommMonoid_381_);
return v_res_384_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulHomogeneousIdeal___redArg(lean_object* v_inst_385_){
_start:
{
lean_object* v_toAddCommMonoid_386_; lean_object* v___f_387_; 
v_toAddCommMonoid_386_ = lean_ctor_get(v_inst_385_, 0);
lean_inc_ref(v_toAddCommMonoid_386_);
lean_dec_ref(v_inst_385_);
v___f_387_ = lean_alloc_closure((void*)(lp_mathlib_instMulHomogeneousIdeal___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_387_, 0, v_toAddCommMonoid_386_);
return v___f_387_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulHomogeneousIdeal(lean_object* v_00_u03b9_388_, lean_object* v_00_u03c3_389_, lean_object* v_A_390_, lean_object* v_inst_391_, lean_object* v_inst_392_, lean_object* v_inst_393_, lean_object* v_inst_394_, lean_object* v_inst_395_, lean_object* v_00_U0001d49c_396_, lean_object* v_inst_397_){
_start:
{
lean_object* v___x_398_; 
v___x_398_ = lp_mathlib_instMulHomogeneousIdeal___redArg(v_inst_391_);
return v___x_398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instMulHomogeneousIdeal___boxed(lean_object* v_00_u03b9_399_, lean_object* v_00_u03c3_400_, lean_object* v_A_401_, lean_object* v_inst_402_, lean_object* v_inst_403_, lean_object* v_inst_404_, lean_object* v_inst_405_, lean_object* v_inst_406_, lean_object* v_00_U0001d49c_407_, lean_object* v_inst_408_){
_start:
{
lean_object* v_res_409_; 
v_res_409_ = lp_mathlib_instMulHomogeneousIdeal(v_00_u03b9_399_, v_00_u03c3_400_, v_A_401_, v_inst_402_, v_inst_403_, v_inst_404_, v_inst_405_, v_inst_406_, v_00_U0001d49c_407_, v_inst_408_);
lean_dec_ref(v_inst_408_);
lean_dec(v_00_U0001d49c_407_);
lean_dec_ref(v_inst_404_);
lean_dec_ref(v_inst_403_);
return v_res_409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_homogeneousCore_gi___lam__0(lean_object* v_I_410_, lean_object* v_HI_411_){
_start:
{
return v_I_410_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_homogeneousCore_gi(lean_object* v_00_u03b9_413_, lean_object* v_00_u03c3_414_, lean_object* v_A_415_, lean_object* v_inst_416_, lean_object* v_inst_417_, lean_object* v_inst_418_, lean_object* v_inst_419_, lean_object* v_inst_420_, lean_object* v_00_U0001d49c_421_, lean_object* v_inst_422_){
_start:
{
lean_object* v___f_423_; 
v___f_423_ = ((lean_object*)(lp_mathlib_Ideal_homogeneousCore_gi___closed__0));
return v___f_423_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_homogeneousCore_gi___boxed(lean_object* v_00_u03b9_424_, lean_object* v_00_u03c3_425_, lean_object* v_A_426_, lean_object* v_inst_427_, lean_object* v_inst_428_, lean_object* v_inst_429_, lean_object* v_inst_430_, lean_object* v_inst_431_, lean_object* v_00_U0001d49c_432_, lean_object* v_inst_433_){
_start:
{
lean_object* v_res_434_; 
v_res_434_ = lp_mathlib_Ideal_homogeneousCore_gi(v_00_u03b9_424_, v_00_u03c3_425_, v_A_426_, v_inst_427_, v_inst_428_, v_inst_429_, v_inst_430_, v_inst_431_, v_00_U0001d49c_432_, v_inst_433_);
lean_dec_ref(v_inst_433_);
lean_dec(v_00_U0001d49c_432_);
lean_dec_ref(v_inst_429_);
lean_dec_ref(v_inst_428_);
lean_dec_ref(v_inst_427_);
return v_res_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_homogeneousHull(lean_object* v_00_u03b9_435_, lean_object* v_00_u03c3_436_, lean_object* v_A_437_, lean_object* v_inst_438_, lean_object* v_inst_439_, lean_object* v_inst_440_, lean_object* v_inst_441_, lean_object* v_inst_442_, lean_object* v_00_U0001d49c_443_, lean_object* v_inst_444_, lean_object* v_I_445_){
_start:
{
lean_object* v___x_446_; 
v___x_446_ = lean_box(0);
return v___x_446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_homogeneousHull___boxed(lean_object* v_00_u03b9_447_, lean_object* v_00_u03c3_448_, lean_object* v_A_449_, lean_object* v_inst_450_, lean_object* v_inst_451_, lean_object* v_inst_452_, lean_object* v_inst_453_, lean_object* v_inst_454_, lean_object* v_00_U0001d49c_455_, lean_object* v_inst_456_, lean_object* v_I_457_){
_start:
{
lean_object* v_res_458_; 
v_res_458_ = lp_mathlib_Ideal_homogeneousHull(v_00_u03b9_447_, v_00_u03c3_448_, v_A_449_, v_inst_450_, v_inst_451_, v_inst_452_, v_inst_453_, v_inst_454_, v_00_U0001d49c_455_, v_inst_456_, v_I_457_);
lean_dec_ref(v_inst_456_);
lean_dec(v_00_U0001d49c_455_);
lean_dec_ref(v_inst_452_);
lean_dec_ref(v_inst_451_);
lean_dec_ref(v_inst_450_);
return v_res_458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_homogeneousHull_gi___lam__0(lean_object* v_I_459_, lean_object* v_H_460_){
_start:
{
return v_I_459_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_homogeneousHull_gi(lean_object* v_00_u03b9_462_, lean_object* v_00_u03c3_463_, lean_object* v_A_464_, lean_object* v_inst_465_, lean_object* v_inst_466_, lean_object* v_inst_467_, lean_object* v_inst_468_, lean_object* v_inst_469_, lean_object* v_00_U0001d49c_470_, lean_object* v_inst_471_){
_start:
{
lean_object* v___f_472_; 
v___f_472_ = ((lean_object*)(lp_mathlib_Ideal_homogeneousHull_gi___closed__0));
return v___f_472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_homogeneousHull_gi___boxed(lean_object* v_00_u03b9_473_, lean_object* v_00_u03c3_474_, lean_object* v_A_475_, lean_object* v_inst_476_, lean_object* v_inst_477_, lean_object* v_inst_478_, lean_object* v_inst_479_, lean_object* v_inst_480_, lean_object* v_00_U0001d49c_481_, lean_object* v_inst_482_){
_start:
{
lean_object* v_res_483_; 
v_res_483_ = lp_mathlib_Ideal_homogeneousHull_gi(v_00_u03b9_473_, v_00_u03c3_474_, v_A_475_, v_inst_476_, v_inst_477_, v_inst_478_, v_inst_479_, v_inst_480_, v_00_U0001d49c_481_, v_inst_482_);
lean_dec_ref(v_inst_482_);
lean_dec(v_00_U0001d49c_481_);
lean_dec_ref(v_inst_478_);
lean_dec_ref(v_inst_477_);
lean_dec_ref(v_inst_476_);
return v_res_483_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_irrelevant(lean_object* v_00_u03b9_484_, lean_object* v_00_u03c3_485_, lean_object* v_A_486_, lean_object* v_inst_487_, lean_object* v_inst_488_, lean_object* v_inst_489_, lean_object* v_inst_490_, lean_object* v_inst_491_, lean_object* v_inst_492_, lean_object* v_inst_493_, lean_object* v_00_U0001d49c_494_, lean_object* v_inst_495_){
_start:
{
lean_object* v___x_496_; 
v___x_496_ = lean_box(0);
return v___x_496_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal_irrelevant___boxed(lean_object* v_00_u03b9_497_, lean_object* v_00_u03c3_498_, lean_object* v_A_499_, lean_object* v_inst_500_, lean_object* v_inst_501_, lean_object* v_inst_502_, lean_object* v_inst_503_, lean_object* v_inst_504_, lean_object* v_inst_505_, lean_object* v_inst_506_, lean_object* v_00_U0001d49c_507_, lean_object* v_inst_508_){
_start:
{
lean_object* v_res_509_; 
v_res_509_ = lp_mathlib_HomogeneousIdeal_irrelevant(v_00_u03b9_497_, v_00_u03c3_498_, v_A_499_, v_inst_500_, v_inst_501_, v_inst_502_, v_inst_503_, v_inst_504_, v_inst_505_, v_inst_506_, v_00_U0001d49c_507_, v_inst_508_);
lean_dec_ref(v_inst_508_);
lean_dec(v_00_U0001d49c_507_);
lean_dec_ref(v_inst_503_);
lean_dec_ref(v_inst_502_);
lean_dec_ref(v_inst_501_);
lean_dec_ref(v_inst_500_);
return v_res_509_;
}
}
static lean_object* _init_lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__6(void){
_start:
{
lean_object* v___x_534_; lean_object* v___x_535_; 
v___x_534_ = ((lean_object*)(lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__5));
v___x_535_ = l_String_toRawSubstring_x27(v___x_534_);
return v___x_535_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1(lean_object* v_x_550_, lean_object* v_a_551_, lean_object* v_a_552_){
_start:
{
lean_object* v___x_553_; uint8_t v___x_554_; 
v___x_553_ = ((lean_object*)(lp_mathlib_HomogeneousIdeal_term___u208a___closed__2));
lean_inc(v_x_550_);
v___x_554_ = l_Lean_Syntax_isOfKind(v_x_550_, v___x_553_);
if (v___x_554_ == 0)
{
lean_object* v___x_555_; lean_object* v___x_556_; 
lean_dec(v_x_550_);
v___x_555_ = lean_box(1);
v___x_556_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_556_, 0, v___x_555_);
lean_ctor_set(v___x_556_, 1, v_a_552_);
return v___x_556_;
}
else
{
lean_object* v_quotContext_557_; lean_object* v_currMacroScope_558_; lean_object* v_ref_559_; lean_object* v___x_560_; lean_object* v___x_561_; uint8_t v___x_562_; lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_572_; lean_object* v___x_573_; 
v_quotContext_557_ = lean_ctor_get(v_a_551_, 1);
v_currMacroScope_558_ = lean_ctor_get(v_a_551_, 2);
v_ref_559_ = lean_ctor_get(v_a_551_, 5);
v___x_560_ = lean_unsigned_to_nat(0u);
v___x_561_ = l_Lean_Syntax_getArg(v_x_550_, v___x_560_);
lean_dec(v_x_550_);
v___x_562_ = 0;
v___x_563_ = l_Lean_SourceInfo_fromRef(v_ref_559_, v___x_562_);
v___x_564_ = ((lean_object*)(lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__4));
v___x_565_ = lean_obj_once(&lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__6, &lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__6_once, _init_lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__6);
v___x_566_ = ((lean_object*)(lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__7));
lean_inc(v_currMacroScope_558_);
lean_inc(v_quotContext_557_);
v___x_567_ = l_Lean_addMacroScope(v_quotContext_557_, v___x_566_, v_currMacroScope_558_);
v___x_568_ = ((lean_object*)(lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__10));
lean_inc_n(v___x_563_, 2);
v___x_569_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_569_, 0, v___x_563_);
lean_ctor_set(v___x_569_, 1, v___x_565_);
lean_ctor_set(v___x_569_, 2, v___x_567_);
lean_ctor_set(v___x_569_, 3, v___x_568_);
v___x_570_ = ((lean_object*)(lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__12));
v___x_571_ = l_Lean_Syntax_node1(v___x_563_, v___x_570_, v___x_561_);
v___x_572_ = l_Lean_Syntax_node2(v___x_563_, v___x_564_, v___x_569_, v___x_571_);
v___x_573_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_573_, 0, v___x_572_);
lean_ctor_set(v___x_573_, 1, v_a_552_);
return v___x_573_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___boxed(lean_object* v_x_574_, lean_object* v_a_575_, lean_object* v_a_576_){
_start:
{
lean_object* v_res_577_; 
v_res_577_ = lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1(v_x_574_, v_a_575_, v_a_576_);
lean_dec_ref(v_a_575_);
return v_res_577_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______unexpand__HomogeneousIdeal__irrelevant__1(lean_object* v_x_581_, lean_object* v_a_582_, lean_object* v_a_583_){
_start:
{
lean_object* v___x_584_; uint8_t v___x_585_; 
v___x_584_ = ((lean_object*)(lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______macroRules__HomogeneousIdeal__term___u208a__1___closed__4));
lean_inc(v_x_581_);
v___x_585_ = l_Lean_Syntax_isOfKind(v_x_581_, v___x_584_);
if (v___x_585_ == 0)
{
lean_object* v___x_586_; lean_object* v___x_587_; 
lean_dec(v_x_581_);
v___x_586_ = lean_box(0);
v___x_587_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_587_, 0, v___x_586_);
lean_ctor_set(v___x_587_, 1, v_a_583_);
return v___x_587_;
}
else
{
lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; uint8_t v___x_591_; 
v___x_588_ = lean_unsigned_to_nat(0u);
v___x_589_ = l_Lean_Syntax_getArg(v_x_581_, v___x_588_);
v___x_590_ = ((lean_object*)(lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______unexpand__HomogeneousIdeal__irrelevant__1___closed__1));
lean_inc(v___x_589_);
v___x_591_ = l_Lean_Syntax_isOfKind(v___x_589_, v___x_590_);
if (v___x_591_ == 0)
{
lean_object* v___x_592_; lean_object* v___x_593_; 
lean_dec(v___x_589_);
lean_dec(v_x_581_);
v___x_592_ = lean_box(0);
v___x_593_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_593_, 0, v___x_592_);
lean_ctor_set(v___x_593_, 1, v_a_583_);
return v___x_593_;
}
else
{
lean_object* v___x_594_; lean_object* v___x_595_; uint8_t v___x_596_; 
v___x_594_ = lean_unsigned_to_nat(1u);
v___x_595_ = l_Lean_Syntax_getArg(v_x_581_, v___x_594_);
lean_dec(v_x_581_);
lean_inc(v___x_595_);
v___x_596_ = l_Lean_Syntax_matchesNull(v___x_595_, v___x_594_);
if (v___x_596_ == 0)
{
lean_object* v___x_597_; lean_object* v___x_598_; 
lean_dec(v___x_595_);
lean_dec(v___x_589_);
v___x_597_ = lean_box(0);
v___x_598_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_598_, 0, v___x_597_);
lean_ctor_set(v___x_598_, 1, v_a_583_);
return v___x_598_;
}
else
{
lean_object* v___x_599_; lean_object* v_ref_600_; uint8_t v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_605_; lean_object* v___x_606_; lean_object* v___x_607_; 
v___x_599_ = l_Lean_Syntax_getArg(v___x_595_, v___x_588_);
lean_dec(v___x_595_);
v_ref_600_ = l_Lean_replaceRef(v___x_589_, v_a_582_);
lean_dec(v___x_589_);
v___x_601_ = 0;
v___x_602_ = l_Lean_SourceInfo_fromRef(v_ref_600_, v___x_601_);
lean_dec(v_ref_600_);
v___x_603_ = ((lean_object*)(lp_mathlib_HomogeneousIdeal_term___u208a___closed__2));
v___x_604_ = ((lean_object*)(lp_mathlib_HomogeneousIdeal_term___u208a___closed__3));
lean_inc(v___x_602_);
v___x_605_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_605_, 0, v___x_602_);
lean_ctor_set(v___x_605_, 1, v___x_604_);
v___x_606_ = l_Lean_Syntax_node2(v___x_602_, v___x_603_, v___x_599_, v___x_605_);
v___x_607_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_607_, 0, v___x_606_);
lean_ctor_set(v___x_607_, 1, v_a_583_);
return v___x_607_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______unexpand__HomogeneousIdeal__irrelevant__1___boxed(lean_object* v_x_608_, lean_object* v_a_609_, lean_object* v_a_610_){
_start:
{
lean_object* v_res_611_; 
v_res_611_ = lp_mathlib_HomogeneousIdeal___aux__Mathlib__RingTheory__GradedAlgebra__Homogeneous__Ideal______unexpand__HomogeneousIdeal__irrelevant__1(v_x_608_, v_a_609_, v_a_610_);
lean_dec(v_a_609_);
return v_res_611_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_SumProd(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_GradedAlgebra_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_Ideal_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_Ideal_BigOperators(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_Ideal_Maps(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_GradedAlgebra_Homogeneous_Submodule(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_GradedAlgebra_Homogeneous_Ideal(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_SumProd(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_GradedAlgebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_Ideal_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_Ideal_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_Ideal_Maps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_GradedAlgebra_Homogeneous_Submodule(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_RingTheory_GradedAlgebra_Homogeneous_Ideal(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_SumProd(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_GradedAlgebra_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Ideal_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Ideal_BigOperators(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Ideal_Maps(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_GradedAlgebra_Homogeneous_Submodule(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_GradedAlgebra_Homogeneous_Ideal(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_SumProd(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_GradedAlgebra_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Ideal_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Ideal_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Ideal_Maps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_GradedAlgebra_Homogeneous_Submodule(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_GradedAlgebra_Homogeneous_Ideal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_RingTheory_GradedAlgebra_Homogeneous_Ideal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_RingTheory_GradedAlgebra_Homogeneous_Ideal(builtin);
}
#ifdef __cplusplus
}
#endif
