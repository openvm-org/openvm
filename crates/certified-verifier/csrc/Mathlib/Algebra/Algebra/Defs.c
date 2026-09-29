// Lean compiler output
// Module: Mathlib.Algebra.Algebra.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Module.LinearMap.Defs
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
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_instDistribOfSemiring___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SMulWithZero_compHom___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_RingHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalRingHom_id___lam__0___boxed(lean_object*);
lean_object* l_instSMulOfMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_cast___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_cast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_cast___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_algebraMap_coeHTCT___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_algebraMap_coeHTCT(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toAlgebra_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toAlgebra_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toAlgebra_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toAlgebra_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toAlgebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toAlgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_ofModule_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_ofModule_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_ofModule_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_ofModule_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_ofModule___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_ofModule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_ofModule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_toModule___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_toModule___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_toModule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_toModule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_compHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_compHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_compHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_linearMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_linearMap___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_linearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_linearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "RingTheory"};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__0 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__0_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "LinearMap"};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__1 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__1_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 5, .m_data = "termη"};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__2 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__2_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__0_value),LEAN_SCALAR_PTR_LITERAL(204, 50, 210, 176, 233, 167, 74, 91)}};
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__3_value_aux_0),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 100, 27, 238, 183, 36, 185, 13)}};
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__3_value_aux_1),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__2_value),LEAN_SCALAR_PTR_LITERAL(219, 5, 126, 155, 143, 207, 67, 81)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__3 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__3_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 1, .m_data = "η"};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__4 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__4_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__4_value)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__5 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__5_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__3_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__5_value)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__6 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03b7 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__6_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__0 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__0_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__1 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__1_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__2 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__2_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__3 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__3_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__4 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__4_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "Algebra.linearMap"};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__5 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__5_value;
static lean_once_cell_t lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__6;
static const lean_string_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Algebra"};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__7 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__7_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "linearMap"};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__8 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__8_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(23, 115, 243, 139, 49, 165, 250, 62)}};
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__9_value_aux_0),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(235, 71, 215, 72, 138, 146, 28, 92)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__9 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__9_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__10 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__10_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__11 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__11_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__12 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__12_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__13 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__13_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hole"};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__14 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__14_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__15_value_aux_0),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__15_value_aux_1),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__15_value_aux_2),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(135, 134, 219, 115, 97, 130, 74, 55)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__15 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__15_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__16 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__16_value;
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______unexpand__Algebra__linearMap__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______unexpand__Algebra__linearMap__1___closed__0 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______unexpand__Algebra__linearMap__1___closed__0_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______unexpand__Algebra__linearMap__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______unexpand__Algebra__linearMap__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______unexpand__Algebra__linearMap__1___closed__1 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______unexpand__Algebra__linearMap__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______unexpand__Algebra__linearMap__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______unexpand__Algebra__linearMap__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 8, .m_data = "termη[_]"};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__0 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__0_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__0_value),LEAN_SCALAR_PTR_LITERAL(204, 50, 210, 176, 233, 167, 74, 91)}};
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__1_value_aux_0),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 100, 27, 238, 183, 36, 185, 13)}};
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__1_value_aux_1),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__0_value),LEAN_SCALAR_PTR_LITERAL(154, 239, 23, 182, 67, 206, 211, 135)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__1 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__1_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__2 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__2_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__3 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__3_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 2, .m_data = "η["};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__4 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__4_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__4_value)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__5 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__5_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__6 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__6_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__7 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__7_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__8 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__8_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__3_value),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__5_value),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__8_value)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__9 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__9_value;
static const lean_string_object lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__10 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__10_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__10_value)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__11 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__11_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__3_value),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__9_value),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__11_value)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__12 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__12_value;
static const lean_ctor_object lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__12_value)}};
static const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__13 = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__13_value;
LEAN_EXPORT const lean_object* lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d = (const lean_object*)&lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7_x5b___x5d__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7_x5b___x5d__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______unexpand__Algebra__linearMap__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______unexpand__Algebra__linearMap__2___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Algebra_id___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalRingHom_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Algebra_id___redArg___closed__0 = (const lean_object*)&lp_mathlib_Algebra_id___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Algebra_id___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Algebra_cast___redArg(lean_object* v_inst_1_, lean_object* v_a_2_){
_start:
{
lean_object* v_algebraMap_3_; lean_object* v___x_4_; 
v_algebraMap_3_ = lean_ctor_get(v_inst_1_, 1);
lean_inc(v_algebraMap_3_);
lean_dec_ref(v_inst_1_);
v___x_4_ = lean_apply_1(v_algebraMap_3_, v_a_2_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_cast(lean_object* v_R_5_, lean_object* v_A_6_, lean_object* v_inst_7_, lean_object* v_inst_8_, lean_object* v_inst_9_, lean_object* v_a_10_){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lp_mathlib_Algebra_cast___redArg(v_inst_9_, v_a_10_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_cast___boxed(lean_object* v_R_12_, lean_object* v_A_13_, lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_a_17_){
_start:
{
lean_object* v_res_18_; 
v_res_18_ = lp_mathlib_Algebra_cast(v_R_12_, v_A_13_, v_inst_14_, v_inst_15_, v_inst_16_, v_a_17_);
lean_dec_ref(v_inst_15_);
lean_dec_ref(v_inst_14_);
return v_res_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_algebraMap_coeHTCT___redArg(lean_object* v_inst_19_, lean_object* v_inst_20_, lean_object* v_inst_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lean_alloc_closure((void*)(lp_mathlib_Algebra_cast___boxed), 6, 5);
lean_closure_set(v___x_22_, 0, lean_box(0));
lean_closure_set(v___x_22_, 1, lean_box(0));
lean_closure_set(v___x_22_, 2, v_inst_19_);
lean_closure_set(v___x_22_, 3, v_inst_20_);
lean_closure_set(v___x_22_, 4, v_inst_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_algebraMap_coeHTCT(lean_object* v_R_23_, lean_object* v_A_24_, lean_object* v_inst_25_, lean_object* v_inst_26_, lean_object* v_inst_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lean_alloc_closure((void*)(lp_mathlib_Algebra_cast___boxed), 6, 5);
lean_closure_set(v___x_28_, 0, lean_box(0));
lean_closure_set(v___x_28_, 1, lean_box(0));
lean_closure_set(v___x_28_, 2, v_inst_25_);
lean_closure_set(v___x_28_, 3, v_inst_26_);
lean_closure_set(v___x_28_, 4, v_inst_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toAlgebra_x27___redArg___lam__0(lean_object* v_i_29_, lean_object* v_toMul_30_, lean_object* v_c_31_, lean_object* v_x_32_){
_start:
{
lean_object* v___x_33_; lean_object* v___x_34_; 
v___x_33_ = lean_apply_1(v_i_29_, v_c_31_);
v___x_34_ = lean_apply_2(v_toMul_30_, v___x_33_, v_x_32_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toAlgebra_x27___redArg(lean_object* v_inst_35_, lean_object* v_i_36_){
_start:
{
lean_object* v___x_37_; lean_object* v_toMul_38_; lean_object* v___x_40_; uint8_t v_isShared_41_; uint8_t v_isSharedCheck_46_; 
v___x_37_ = lp_mathlib_instDistribOfSemiring___redArg(v_inst_35_);
v_toMul_38_ = lean_ctor_get(v___x_37_, 0);
v_isSharedCheck_46_ = !lean_is_exclusive(v___x_37_);
if (v_isSharedCheck_46_ == 0)
{
lean_object* v_unused_47_; 
v_unused_47_ = lean_ctor_get(v___x_37_, 1);
lean_dec(v_unused_47_);
v___x_40_ = v___x_37_;
v_isShared_41_ = v_isSharedCheck_46_;
goto v_resetjp_39_;
}
else
{
lean_inc(v_toMul_38_);
lean_dec(v___x_37_);
v___x_40_ = lean_box(0);
v_isShared_41_ = v_isSharedCheck_46_;
goto v_resetjp_39_;
}
v_resetjp_39_:
{
lean_object* v___f_42_; lean_object* v___x_44_; 
lean_inc(v_i_36_);
v___f_42_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_toAlgebra_x27___redArg___lam__0), 4, 2);
lean_closure_set(v___f_42_, 0, v_i_36_);
lean_closure_set(v___f_42_, 1, v_toMul_38_);
if (v_isShared_41_ == 0)
{
lean_ctor_set(v___x_40_, 1, v_i_36_);
lean_ctor_set(v___x_40_, 0, v___f_42_);
v___x_44_ = v___x_40_;
goto v_reusejp_43_;
}
else
{
lean_object* v_reuseFailAlloc_45_; 
v_reuseFailAlloc_45_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_45_, 0, v___f_42_);
lean_ctor_set(v_reuseFailAlloc_45_, 1, v_i_36_);
v___x_44_ = v_reuseFailAlloc_45_;
goto v_reusejp_43_;
}
v_reusejp_43_:
{
return v___x_44_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toAlgebra_x27(lean_object* v_R_48_, lean_object* v_S_49_, lean_object* v_inst_50_, lean_object* v_inst_51_, lean_object* v_i_52_, lean_object* v_h_53_){
_start:
{
lean_object* v___x_54_; lean_object* v_toMul_55_; lean_object* v___x_57_; uint8_t v_isShared_58_; uint8_t v_isSharedCheck_63_; 
v___x_54_ = lp_mathlib_instDistribOfSemiring___redArg(v_inst_51_);
v_toMul_55_ = lean_ctor_get(v___x_54_, 0);
v_isSharedCheck_63_ = !lean_is_exclusive(v___x_54_);
if (v_isSharedCheck_63_ == 0)
{
lean_object* v_unused_64_; 
v_unused_64_ = lean_ctor_get(v___x_54_, 1);
lean_dec(v_unused_64_);
v___x_57_ = v___x_54_;
v_isShared_58_ = v_isSharedCheck_63_;
goto v_resetjp_56_;
}
else
{
lean_inc(v_toMul_55_);
lean_dec(v___x_54_);
v___x_57_ = lean_box(0);
v_isShared_58_ = v_isSharedCheck_63_;
goto v_resetjp_56_;
}
v_resetjp_56_:
{
lean_object* v___f_59_; lean_object* v___x_61_; 
lean_inc(v_i_52_);
v___f_59_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_toAlgebra_x27___redArg___lam__0), 4, 2);
lean_closure_set(v___f_59_, 0, v_i_52_);
lean_closure_set(v___f_59_, 1, v_toMul_55_);
if (v_isShared_58_ == 0)
{
lean_ctor_set(v___x_57_, 1, v_i_52_);
lean_ctor_set(v___x_57_, 0, v___f_59_);
v___x_61_ = v___x_57_;
goto v_reusejp_60_;
}
else
{
lean_object* v_reuseFailAlloc_62_; 
v_reuseFailAlloc_62_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_62_, 0, v___f_59_);
lean_ctor_set(v_reuseFailAlloc_62_, 1, v_i_52_);
v___x_61_ = v_reuseFailAlloc_62_;
goto v_reusejp_60_;
}
v_reusejp_60_:
{
return v___x_61_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toAlgebra_x27___boxed(lean_object* v_R_65_, lean_object* v_S_66_, lean_object* v_inst_67_, lean_object* v_inst_68_, lean_object* v_i_69_, lean_object* v_h_70_){
_start:
{
lean_object* v_res_71_; 
v_res_71_ = lp_mathlib_RingHom_toAlgebra_x27(v_R_65_, v_S_66_, v_inst_67_, v_inst_68_, v_i_69_, v_h_70_);
lean_dec_ref(v_inst_67_);
return v_res_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toAlgebra___redArg(lean_object* v_inst_72_, lean_object* v_i_73_){
_start:
{
lean_object* v___x_74_; lean_object* v_toMul_75_; lean_object* v___x_77_; uint8_t v_isShared_78_; uint8_t v_isSharedCheck_83_; 
v___x_74_ = lp_mathlib_instDistribOfSemiring___redArg(v_inst_72_);
v_toMul_75_ = lean_ctor_get(v___x_74_, 0);
v_isSharedCheck_83_ = !lean_is_exclusive(v___x_74_);
if (v_isSharedCheck_83_ == 0)
{
lean_object* v_unused_84_; 
v_unused_84_ = lean_ctor_get(v___x_74_, 1);
lean_dec(v_unused_84_);
v___x_77_ = v___x_74_;
v_isShared_78_ = v_isSharedCheck_83_;
goto v_resetjp_76_;
}
else
{
lean_inc(v_toMul_75_);
lean_dec(v___x_74_);
v___x_77_ = lean_box(0);
v_isShared_78_ = v_isSharedCheck_83_;
goto v_resetjp_76_;
}
v_resetjp_76_:
{
lean_object* v___f_79_; lean_object* v___x_81_; 
lean_inc(v_i_73_);
v___f_79_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_toAlgebra_x27___redArg___lam__0), 4, 2);
lean_closure_set(v___f_79_, 0, v_i_73_);
lean_closure_set(v___f_79_, 1, v_toMul_75_);
if (v_isShared_78_ == 0)
{
lean_ctor_set(v___x_77_, 1, v_i_73_);
lean_ctor_set(v___x_77_, 0, v___f_79_);
v___x_81_ = v___x_77_;
goto v_reusejp_80_;
}
else
{
lean_object* v_reuseFailAlloc_82_; 
v_reuseFailAlloc_82_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_82_, 0, v___f_79_);
lean_ctor_set(v_reuseFailAlloc_82_, 1, v_i_73_);
v___x_81_ = v_reuseFailAlloc_82_;
goto v_reusejp_80_;
}
v_reusejp_80_:
{
return v___x_81_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toAlgebra(lean_object* v_R_85_, lean_object* v_S_86_, lean_object* v_inst_87_, lean_object* v_inst_88_, lean_object* v_i_89_){
_start:
{
lean_object* v___x_90_; lean_object* v_toMul_91_; lean_object* v___x_93_; uint8_t v_isShared_94_; uint8_t v_isSharedCheck_99_; 
v___x_90_ = lp_mathlib_instDistribOfSemiring___redArg(v_inst_88_);
v_toMul_91_ = lean_ctor_get(v___x_90_, 0);
v_isSharedCheck_99_ = !lean_is_exclusive(v___x_90_);
if (v_isSharedCheck_99_ == 0)
{
lean_object* v_unused_100_; 
v_unused_100_ = lean_ctor_get(v___x_90_, 1);
lean_dec(v_unused_100_);
v___x_93_ = v___x_90_;
v_isShared_94_ = v_isSharedCheck_99_;
goto v_resetjp_92_;
}
else
{
lean_inc(v_toMul_91_);
lean_dec(v___x_90_);
v___x_93_ = lean_box(0);
v_isShared_94_ = v_isSharedCheck_99_;
goto v_resetjp_92_;
}
v_resetjp_92_:
{
lean_object* v___f_95_; lean_object* v___x_97_; 
lean_inc(v_i_89_);
v___f_95_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_toAlgebra_x27___redArg___lam__0), 4, 2);
lean_closure_set(v___f_95_, 0, v_i_89_);
lean_closure_set(v___f_95_, 1, v_toMul_91_);
if (v_isShared_94_ == 0)
{
lean_ctor_set(v___x_93_, 1, v_i_89_);
lean_ctor_set(v___x_93_, 0, v___f_95_);
v___x_97_ = v___x_93_;
goto v_reusejp_96_;
}
else
{
lean_object* v_reuseFailAlloc_98_; 
v_reuseFailAlloc_98_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_98_, 0, v___f_95_);
lean_ctor_set(v_reuseFailAlloc_98_, 1, v_i_89_);
v___x_97_ = v_reuseFailAlloc_98_;
goto v_reusejp_96_;
}
v_reusejp_96_:
{
return v___x_97_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_toAlgebra___boxed(lean_object* v_R_101_, lean_object* v_S_102_, lean_object* v_inst_103_, lean_object* v_inst_104_, lean_object* v_i_105_){
_start:
{
lean_object* v_res_106_; 
v_res_106_ = lp_mathlib_RingHom_toAlgebra(v_R_101_, v_S_102_, v_inst_103_, v_inst_104_, v_i_105_);
lean_dec_ref(v_inst_103_);
return v_res_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_ofModule_x27___redArg___lam__0(lean_object* v_inst_107_, lean_object* v_toOne_108_, lean_object* v_r_109_){
_start:
{
lean_object* v___x_110_; 
v___x_110_ = lean_apply_2(v_inst_107_, v_r_109_, v_toOne_108_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_ofModule_x27___redArg(lean_object* v_inst_111_, lean_object* v_inst_112_){
_start:
{
lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v_toOne_115_; lean_object* v___f_116_; lean_object* v___x_117_; 
v___x_113_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_111_);
v___x_114_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_113_);
v_toOne_115_ = lean_ctor_get(v___x_114_, 2);
lean_inc(v_toOne_115_);
lean_dec_ref(v___x_114_);
lean_inc(v_inst_112_);
v___f_116_ = lean_alloc_closure((void*)(lp_mathlib_Algebra_ofModule_x27___redArg___lam__0), 3, 2);
lean_closure_set(v___f_116_, 0, v_inst_112_);
lean_closure_set(v___f_116_, 1, v_toOne_115_);
v___x_117_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_117_, 0, v_inst_112_);
lean_ctor_set(v___x_117_, 1, v___f_116_);
return v___x_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_ofModule_x27(lean_object* v_R_118_, lean_object* v_A_119_, lean_object* v_inst_120_, lean_object* v_inst_121_, lean_object* v_inst_122_, lean_object* v_h_u2081_123_, lean_object* v_h_u2082_124_){
_start:
{
lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v_toOne_127_; lean_object* v___f_128_; lean_object* v___x_129_; 
v___x_125_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_121_);
v___x_126_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_125_);
v_toOne_127_ = lean_ctor_get(v___x_126_, 2);
lean_inc(v_toOne_127_);
lean_dec_ref(v___x_126_);
lean_inc(v_inst_122_);
v___f_128_ = lean_alloc_closure((void*)(lp_mathlib_Algebra_ofModule_x27___redArg___lam__0), 3, 2);
lean_closure_set(v___f_128_, 0, v_inst_122_);
lean_closure_set(v___f_128_, 1, v_toOne_127_);
v___x_129_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_129_, 0, v_inst_122_);
lean_ctor_set(v___x_129_, 1, v___f_128_);
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_ofModule_x27___boxed(lean_object* v_R_130_, lean_object* v_A_131_, lean_object* v_inst_132_, lean_object* v_inst_133_, lean_object* v_inst_134_, lean_object* v_h_u2081_135_, lean_object* v_h_u2082_136_){
_start:
{
lean_object* v_res_137_; 
v_res_137_ = lp_mathlib_Algebra_ofModule_x27(v_R_130_, v_A_131_, v_inst_132_, v_inst_133_, v_inst_134_, v_h_u2081_135_, v_h_u2082_136_);
lean_dec_ref(v_inst_132_);
return v_res_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_ofModule___redArg(lean_object* v_inst_138_, lean_object* v_inst_139_){
_start:
{
lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v_toOne_142_; lean_object* v___f_143_; lean_object* v___x_144_; 
v___x_140_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_138_);
v___x_141_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_140_);
v_toOne_142_ = lean_ctor_get(v___x_141_, 2);
lean_inc(v_toOne_142_);
lean_dec_ref(v___x_141_);
lean_inc(v_inst_139_);
v___f_143_ = lean_alloc_closure((void*)(lp_mathlib_Algebra_ofModule_x27___redArg___lam__0), 3, 2);
lean_closure_set(v___f_143_, 0, v_inst_139_);
lean_closure_set(v___f_143_, 1, v_toOne_142_);
v___x_144_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_144_, 0, v_inst_139_);
lean_ctor_set(v___x_144_, 1, v___f_143_);
return v___x_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_ofModule(lean_object* v_R_145_, lean_object* v_A_146_, lean_object* v_inst_147_, lean_object* v_inst_148_, lean_object* v_inst_149_, lean_object* v_h_u2081_150_, lean_object* v_h_u2082_151_){
_start:
{
lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v_toOne_154_; lean_object* v___f_155_; lean_object* v___x_156_; 
v___x_152_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_148_);
v___x_153_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_152_);
v_toOne_154_ = lean_ctor_get(v___x_153_, 2);
lean_inc(v_toOne_154_);
lean_dec_ref(v___x_153_);
lean_inc(v_inst_149_);
v___f_155_ = lean_alloc_closure((void*)(lp_mathlib_Algebra_ofModule_x27___redArg___lam__0), 3, 2);
lean_closure_set(v___f_155_, 0, v_inst_149_);
lean_closure_set(v___f_155_, 1, v_toOne_154_);
v___x_156_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_156_, 0, v_inst_149_);
lean_ctor_set(v___x_156_, 1, v___f_155_);
return v___x_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_ofModule___boxed(lean_object* v_R_157_, lean_object* v_A_158_, lean_object* v_inst_159_, lean_object* v_inst_160_, lean_object* v_inst_161_, lean_object* v_h_u2081_162_, lean_object* v_h_u2082_163_){
_start:
{
lean_object* v_res_164_; 
v_res_164_ = lp_mathlib_Algebra_ofModule(v_R_157_, v_A_158_, v_inst_159_, v_inst_160_, v_inst_161_, v_h_u2081_162_, v_h_u2082_163_);
lean_dec_ref(v_inst_159_);
return v_res_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_toModule___redArg(lean_object* v_inst_165_){
_start:
{
lean_object* v_toSMul_166_; 
v_toSMul_166_ = lean_ctor_get(v_inst_165_, 0);
lean_inc(v_toSMul_166_);
return v_toSMul_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_toModule___redArg___boxed(lean_object* v_inst_167_){
_start:
{
lean_object* v_res_168_; 
v_res_168_ = lp_mathlib_Algebra_toModule___redArg(v_inst_167_);
lean_dec_ref(v_inst_167_);
return v_res_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_toModule(lean_object* v_R_169_, lean_object* v_A_170_, lean_object* v_x_171_, lean_object* v_x_172_, lean_object* v_inst_173_){
_start:
{
lean_object* v_toSMul_174_; 
v_toSMul_174_ = lean_ctor_get(v_inst_173_, 0);
lean_inc(v_toSMul_174_);
return v_toSMul_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_toModule___boxed(lean_object* v_R_175_, lean_object* v_A_176_, lean_object* v_x_177_, lean_object* v_x_178_, lean_object* v_inst_179_){
_start:
{
lean_object* v_res_180_; 
v_res_180_ = lp_mathlib_Algebra_toModule(v_R_175_, v_A_176_, v_x_177_, v_x_178_, v_inst_179_);
lean_dec_ref(v_inst_179_);
lean_dec_ref(v_x_178_);
lean_dec_ref(v_x_177_);
return v_res_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_compHom___redArg(lean_object* v_inst_181_, lean_object* v_f_182_){
_start:
{
lean_object* v_toSMul_183_; lean_object* v_algebraMap_184_; lean_object* v___x_186_; uint8_t v_isShared_187_; uint8_t v_isSharedCheck_193_; 
v_toSMul_183_ = lean_ctor_get(v_inst_181_, 0);
v_algebraMap_184_ = lean_ctor_get(v_inst_181_, 1);
v_isSharedCheck_193_ = !lean_is_exclusive(v_inst_181_);
if (v_isSharedCheck_193_ == 0)
{
v___x_186_ = v_inst_181_;
v_isShared_187_ = v_isSharedCheck_193_;
goto v_resetjp_185_;
}
else
{
lean_inc(v_algebraMap_184_);
lean_inc(v_toSMul_183_);
lean_dec(v_inst_181_);
v___x_186_ = lean_box(0);
v_isShared_187_ = v_isSharedCheck_193_;
goto v_resetjp_185_;
}
v_resetjp_185_:
{
lean_object* v___f_188_; lean_object* v___f_189_; lean_object* v___x_191_; 
lean_inc(v_f_182_);
v___f_188_ = lean_alloc_closure((void*)(lp_mathlib_SMulWithZero_compHom___redArg___lam__0), 4, 2);
lean_closure_set(v___f_188_, 0, v_f_182_);
lean_closure_set(v___f_188_, 1, v_toSMul_183_);
v___f_189_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_189_, 0, v_f_182_);
lean_closure_set(v___f_189_, 1, v_algebraMap_184_);
if (v_isShared_187_ == 0)
{
lean_ctor_set(v___x_186_, 1, v___f_189_);
lean_ctor_set(v___x_186_, 0, v___f_188_);
v___x_191_ = v___x_186_;
goto v_reusejp_190_;
}
else
{
lean_object* v_reuseFailAlloc_192_; 
v_reuseFailAlloc_192_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_192_, 0, v___f_188_);
lean_ctor_set(v_reuseFailAlloc_192_, 1, v___f_189_);
v___x_191_ = v_reuseFailAlloc_192_;
goto v_reusejp_190_;
}
v_reusejp_190_:
{
return v___x_191_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_compHom(lean_object* v_R_194_, lean_object* v_S_195_, lean_object* v_A_196_, lean_object* v_inst_197_, lean_object* v_inst_198_, lean_object* v_inst_199_, lean_object* v_inst_200_, lean_object* v_f_201_){
_start:
{
lean_object* v_toSMul_202_; lean_object* v_algebraMap_203_; lean_object* v___x_205_; uint8_t v_isShared_206_; uint8_t v_isSharedCheck_212_; 
v_toSMul_202_ = lean_ctor_get(v_inst_200_, 0);
v_algebraMap_203_ = lean_ctor_get(v_inst_200_, 1);
v_isSharedCheck_212_ = !lean_is_exclusive(v_inst_200_);
if (v_isSharedCheck_212_ == 0)
{
v___x_205_ = v_inst_200_;
v_isShared_206_ = v_isSharedCheck_212_;
goto v_resetjp_204_;
}
else
{
lean_inc(v_algebraMap_203_);
lean_inc(v_toSMul_202_);
lean_dec(v_inst_200_);
v___x_205_ = lean_box(0);
v_isShared_206_ = v_isSharedCheck_212_;
goto v_resetjp_204_;
}
v_resetjp_204_:
{
lean_object* v___f_207_; lean_object* v___f_208_; lean_object* v___x_210_; 
lean_inc(v_f_201_);
v___f_207_ = lean_alloc_closure((void*)(lp_mathlib_SMulWithZero_compHom___redArg___lam__0), 4, 2);
lean_closure_set(v___f_207_, 0, v_f_201_);
lean_closure_set(v___f_207_, 1, v_toSMul_202_);
v___f_208_ = lean_alloc_closure((void*)(lp_mathlib_RingHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_208_, 0, v_f_201_);
lean_closure_set(v___f_208_, 1, v_algebraMap_203_);
if (v_isShared_206_ == 0)
{
lean_ctor_set(v___x_205_, 1, v___f_208_);
lean_ctor_set(v___x_205_, 0, v___f_207_);
v___x_210_ = v___x_205_;
goto v_reusejp_209_;
}
else
{
lean_object* v_reuseFailAlloc_211_; 
v_reuseFailAlloc_211_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_211_, 0, v___f_207_);
lean_ctor_set(v_reuseFailAlloc_211_, 1, v___f_208_);
v___x_210_ = v_reuseFailAlloc_211_;
goto v_reusejp_209_;
}
v_reusejp_209_:
{
return v___x_210_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_compHom___boxed(lean_object* v_R_213_, lean_object* v_S_214_, lean_object* v_A_215_, lean_object* v_inst_216_, lean_object* v_inst_217_, lean_object* v_inst_218_, lean_object* v_inst_219_, lean_object* v_f_220_){
_start:
{
lean_object* v_res_221_; 
v_res_221_ = lp_mathlib_Algebra_compHom(v_R_213_, v_S_214_, v_A_215_, v_inst_216_, v_inst_217_, v_inst_218_, v_inst_219_, v_f_220_);
lean_dec_ref(v_inst_218_);
lean_dec_ref(v_inst_217_);
lean_dec_ref(v_inst_216_);
return v_res_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_linearMap___redArg(lean_object* v_inst_222_){
_start:
{
lean_object* v_algebraMap_223_; 
v_algebraMap_223_ = lean_ctor_get(v_inst_222_, 1);
lean_inc(v_algebraMap_223_);
return v_algebraMap_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_linearMap___redArg___boxed(lean_object* v_inst_224_){
_start:
{
lean_object* v_res_225_; 
v_res_225_ = lp_mathlib_Algebra_linearMap___redArg(v_inst_224_);
lean_dec_ref(v_inst_224_);
return v_res_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_linearMap(lean_object* v_R_226_, lean_object* v_A_227_, lean_object* v_inst_228_, lean_object* v_inst_229_, lean_object* v_inst_230_){
_start:
{
lean_object* v_algebraMap_231_; 
v_algebraMap_231_ = lean_ctor_get(v_inst_230_, 1);
lean_inc(v_algebraMap_231_);
return v_algebraMap_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_linearMap___boxed(lean_object* v_R_232_, lean_object* v_A_233_, lean_object* v_inst_234_, lean_object* v_inst_235_, lean_object* v_inst_236_){
_start:
{
lean_object* v_res_237_; 
v_res_237_ = lp_mathlib_Algebra_linearMap(v_R_232_, v_A_233_, v_inst_234_, v_inst_235_, v_inst_236_);
lean_dec_ref(v_inst_236_);
lean_dec_ref(v_inst_235_);
lean_dec_ref(v_inst_234_);
return v_res_237_;
}
}
static lean_object* _init_lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__6(void){
_start:
{
lean_object* v___x_263_; lean_object* v___x_264_; 
v___x_263_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__5));
v___x_264_ = l_String_toRawSubstring_x27(v___x_263_);
return v___x_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1(lean_object* v_x_286_, lean_object* v_a_287_, lean_object* v_a_288_){
_start:
{
lean_object* v___x_289_; uint8_t v___x_290_; 
v___x_289_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__3));
v___x_290_ = l_Lean_Syntax_isOfKind(v_x_286_, v___x_289_);
if (v___x_290_ == 0)
{
lean_object* v___x_291_; lean_object* v___x_292_; 
v___x_291_ = lean_box(1);
v___x_292_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_292_, 0, v___x_291_);
lean_ctor_set(v___x_292_, 1, v_a_288_);
return v___x_292_;
}
else
{
lean_object* v_quotContext_293_; lean_object* v_currMacroScope_294_; lean_object* v_ref_295_; uint8_t v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; 
v_quotContext_293_ = lean_ctor_get(v_a_287_, 1);
v_currMacroScope_294_ = lean_ctor_get(v_a_287_, 2);
v_ref_295_ = lean_ctor_get(v_a_287_, 5);
v___x_296_ = 0;
v___x_297_ = l_Lean_SourceInfo_fromRef(v_ref_295_, v___x_296_);
v___x_298_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__4));
v___x_299_ = lean_obj_once(&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__6, &lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__6_once, _init_lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__6);
v___x_300_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__9));
lean_inc(v_currMacroScope_294_);
lean_inc(v_quotContext_293_);
v___x_301_ = l_Lean_addMacroScope(v_quotContext_293_, v___x_300_, v_currMacroScope_294_);
v___x_302_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__11));
lean_inc_n(v___x_297_, 4);
v___x_303_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_303_, 0, v___x_297_);
lean_ctor_set(v___x_303_, 1, v___x_299_);
lean_ctor_set(v___x_303_, 2, v___x_301_);
lean_ctor_set(v___x_303_, 3, v___x_302_);
v___x_304_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__13));
v___x_305_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__15));
v___x_306_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__16));
v___x_307_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_307_, 0, v___x_297_);
lean_ctor_set(v___x_307_, 1, v___x_306_);
v___x_308_ = l_Lean_Syntax_node1(v___x_297_, v___x_305_, v___x_307_);
lean_inc(v___x_308_);
v___x_309_ = l_Lean_Syntax_node2(v___x_297_, v___x_304_, v___x_308_, v___x_308_);
v___x_310_ = l_Lean_Syntax_node2(v___x_297_, v___x_298_, v___x_303_, v___x_309_);
v___x_311_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_311_, 0, v___x_310_);
lean_ctor_set(v___x_311_, 1, v_a_288_);
return v___x_311_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___boxed(lean_object* v_x_312_, lean_object* v_a_313_, lean_object* v_a_314_){
_start:
{
lean_object* v_res_315_; 
v_res_315_ = lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1(v_x_312_, v_a_313_, v_a_314_);
lean_dec_ref(v_a_313_);
return v_res_315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______unexpand__Algebra__linearMap__1(lean_object* v_x_319_, lean_object* v_a_320_, lean_object* v_a_321_){
_start:
{
lean_object* v___x_322_; uint8_t v___x_323_; 
v___x_322_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__4));
lean_inc(v_x_319_);
v___x_323_ = l_Lean_Syntax_isOfKind(v_x_319_, v___x_322_);
if (v___x_323_ == 0)
{
lean_object* v___x_324_; lean_object* v___x_325_; 
lean_dec(v_x_319_);
v___x_324_ = lean_box(0);
v___x_325_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_325_, 0, v___x_324_);
lean_ctor_set(v___x_325_, 1, v_a_321_);
return v___x_325_;
}
else
{
lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; uint8_t v___x_329_; 
v___x_326_ = lean_unsigned_to_nat(0u);
v___x_327_ = l_Lean_Syntax_getArg(v_x_319_, v___x_326_);
v___x_328_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______unexpand__Algebra__linearMap__1___closed__1));
lean_inc(v___x_327_);
v___x_329_ = l_Lean_Syntax_isOfKind(v___x_327_, v___x_328_);
if (v___x_329_ == 0)
{
lean_object* v___x_330_; lean_object* v___x_331_; 
lean_dec(v___x_327_);
lean_dec(v_x_319_);
v___x_330_ = lean_box(0);
v___x_331_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_331_, 0, v___x_330_);
lean_ctor_set(v___x_331_, 1, v_a_321_);
return v___x_331_;
}
else
{
lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; uint8_t v___x_335_; 
v___x_332_ = lean_unsigned_to_nat(1u);
v___x_333_ = l_Lean_Syntax_getArg(v_x_319_, v___x_332_);
lean_dec(v_x_319_);
v___x_334_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_333_);
v___x_335_ = l_Lean_Syntax_matchesNull(v___x_333_, v___x_334_);
if (v___x_335_ == 0)
{
lean_object* v___x_336_; lean_object* v___x_337_; 
lean_dec(v___x_333_);
lean_dec(v___x_327_);
v___x_336_ = lean_box(0);
v___x_337_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_337_, 0, v___x_336_);
lean_ctor_set(v___x_337_, 1, v_a_321_);
return v___x_337_;
}
else
{
lean_object* v___x_338_; lean_object* v___x_339_; uint8_t v___x_340_; 
v___x_338_ = l_Lean_Syntax_getArg(v___x_333_, v___x_326_);
v___x_339_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__15));
v___x_340_ = l_Lean_Syntax_isOfKind(v___x_338_, v___x_339_);
if (v___x_340_ == 0)
{
lean_object* v___x_341_; lean_object* v___x_342_; 
lean_dec(v___x_333_);
lean_dec(v___x_327_);
v___x_341_ = lean_box(0);
v___x_342_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_342_, 0, v___x_341_);
lean_ctor_set(v___x_342_, 1, v_a_321_);
return v___x_342_;
}
else
{
lean_object* v___x_343_; uint8_t v___x_344_; 
v___x_343_ = l_Lean_Syntax_getArg(v___x_333_, v___x_332_);
lean_dec(v___x_333_);
v___x_344_ = l_Lean_Syntax_isOfKind(v___x_343_, v___x_339_);
if (v___x_344_ == 0)
{
lean_object* v___x_345_; lean_object* v___x_346_; 
lean_dec(v___x_327_);
v___x_345_ = lean_box(0);
v___x_346_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_346_, 0, v___x_345_);
lean_ctor_set(v___x_346_, 1, v_a_321_);
return v___x_346_;
}
else
{
lean_object* v_ref_347_; uint8_t v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; 
v_ref_347_ = l_Lean_replaceRef(v___x_327_, v_a_320_);
lean_dec(v___x_327_);
v___x_348_ = 0;
v___x_349_ = l_Lean_SourceInfo_fromRef(v_ref_347_, v___x_348_);
lean_dec(v_ref_347_);
v___x_350_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__3));
v___x_351_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap_term_u03b7___closed__4));
lean_inc(v___x_349_);
v___x_352_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_352_, 0, v___x_349_);
lean_ctor_set(v___x_352_, 1, v___x_351_);
v___x_353_ = l_Lean_Syntax_node1(v___x_349_, v___x_350_, v___x_352_);
v___x_354_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_354_, 0, v___x_353_);
lean_ctor_set(v___x_354_, 1, v_a_321_);
return v___x_354_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______unexpand__Algebra__linearMap__1___boxed(lean_object* v_x_355_, lean_object* v_a_356_, lean_object* v_a_357_){
_start:
{
lean_object* v_res_358_; 
v_res_358_ = lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______unexpand__Algebra__linearMap__1(v_x_355_, v_a_356_, v_a_357_);
lean_dec(v_a_356_);
return v_res_358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7_x5b___x5d__1(lean_object* v_x_392_, lean_object* v_a_393_, lean_object* v_a_394_){
_start:
{
lean_object* v___x_395_; uint8_t v___x_396_; 
v___x_395_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__1));
lean_inc(v_x_392_);
v___x_396_ = l_Lean_Syntax_isOfKind(v_x_392_, v___x_395_);
if (v___x_396_ == 0)
{
lean_object* v___x_397_; lean_object* v___x_398_; 
lean_dec(v_x_392_);
v___x_397_ = lean_box(1);
v___x_398_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_398_, 0, v___x_397_);
lean_ctor_set(v___x_398_, 1, v_a_394_);
return v___x_398_;
}
else
{
lean_object* v_quotContext_399_; lean_object* v_currMacroScope_400_; lean_object* v_ref_401_; lean_object* v___x_402_; lean_object* v___x_403_; uint8_t v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___x_419_; 
v_quotContext_399_ = lean_ctor_get(v_a_393_, 1);
v_currMacroScope_400_ = lean_ctor_get(v_a_393_, 2);
v_ref_401_ = lean_ctor_get(v_a_393_, 5);
v___x_402_ = lean_unsigned_to_nat(1u);
v___x_403_ = l_Lean_Syntax_getArg(v_x_392_, v___x_402_);
lean_dec(v_x_392_);
v___x_404_ = 0;
v___x_405_ = l_Lean_SourceInfo_fromRef(v_ref_401_, v___x_404_);
v___x_406_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__4));
v___x_407_ = lean_obj_once(&lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__6, &lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__6_once, _init_lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__6);
v___x_408_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__9));
lean_inc(v_currMacroScope_400_);
lean_inc(v_quotContext_399_);
v___x_409_ = l_Lean_addMacroScope(v_quotContext_399_, v___x_408_, v_currMacroScope_400_);
v___x_410_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__11));
lean_inc_n(v___x_405_, 4);
v___x_411_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_411_, 0, v___x_405_);
lean_ctor_set(v___x_411_, 1, v___x_407_);
lean_ctor_set(v___x_411_, 2, v___x_409_);
lean_ctor_set(v___x_411_, 3, v___x_410_);
v___x_412_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__13));
v___x_413_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__15));
v___x_414_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__16));
v___x_415_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_415_, 0, v___x_405_);
lean_ctor_set(v___x_415_, 1, v___x_414_);
v___x_416_ = l_Lean_Syntax_node1(v___x_405_, v___x_413_, v___x_415_);
v___x_417_ = l_Lean_Syntax_node2(v___x_405_, v___x_412_, v___x_403_, v___x_416_);
v___x_418_ = l_Lean_Syntax_node2(v___x_405_, v___x_406_, v___x_411_, v___x_417_);
v___x_419_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_419_, 0, v___x_418_);
lean_ctor_set(v___x_419_, 1, v_a_394_);
return v___x_419_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7_x5b___x5d__1___boxed(lean_object* v_x_420_, lean_object* v_a_421_, lean_object* v_a_422_){
_start:
{
lean_object* v_res_423_; 
v_res_423_ = lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7_x5b___x5d__1(v_x_420_, v_a_421_, v_a_422_);
lean_dec_ref(v_a_421_);
return v_res_423_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______unexpand__Algebra__linearMap__2(lean_object* v_x_424_, lean_object* v_a_425_, lean_object* v_a_426_){
_start:
{
lean_object* v___x_427_; uint8_t v___x_428_; 
v___x_427_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__4));
lean_inc(v_x_424_);
v___x_428_ = l_Lean_Syntax_isOfKind(v_x_424_, v___x_427_);
if (v___x_428_ == 0)
{
lean_object* v___x_429_; lean_object* v___x_430_; 
lean_dec(v_x_424_);
v___x_429_ = lean_box(0);
v___x_430_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_430_, 0, v___x_429_);
lean_ctor_set(v___x_430_, 1, v_a_426_);
return v___x_430_;
}
else
{
lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; uint8_t v___x_434_; 
v___x_431_ = lean_unsigned_to_nat(0u);
v___x_432_ = l_Lean_Syntax_getArg(v_x_424_, v___x_431_);
v___x_433_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______unexpand__Algebra__linearMap__1___closed__1));
lean_inc(v___x_432_);
v___x_434_ = l_Lean_Syntax_isOfKind(v___x_432_, v___x_433_);
if (v___x_434_ == 0)
{
lean_object* v___x_435_; lean_object* v___x_436_; 
lean_dec(v___x_432_);
lean_dec(v_x_424_);
v___x_435_ = lean_box(0);
v___x_436_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_436_, 0, v___x_435_);
lean_ctor_set(v___x_436_, 1, v_a_426_);
return v___x_436_;
}
else
{
lean_object* v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; uint8_t v___x_440_; 
v___x_437_ = lean_unsigned_to_nat(1u);
v___x_438_ = l_Lean_Syntax_getArg(v_x_424_, v___x_437_);
lean_dec(v_x_424_);
v___x_439_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_438_);
v___x_440_ = l_Lean_Syntax_matchesNull(v___x_438_, v___x_439_);
if (v___x_440_ == 0)
{
lean_object* v___x_441_; lean_object* v___x_442_; 
lean_dec(v___x_438_);
lean_dec(v___x_432_);
v___x_441_ = lean_box(0);
v___x_442_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_442_, 0, v___x_441_);
lean_ctor_set(v___x_442_, 1, v_a_426_);
return v___x_442_;
}
else
{
lean_object* v___x_443_; lean_object* v___x_444_; uint8_t v___x_445_; 
v___x_443_ = l_Lean_Syntax_getArg(v___x_438_, v___x_437_);
v___x_444_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______macroRules__RingTheory__LinearMap__term_u03b7__1___closed__15));
v___x_445_ = l_Lean_Syntax_isOfKind(v___x_443_, v___x_444_);
if (v___x_445_ == 0)
{
lean_object* v___x_446_; lean_object* v___x_447_; 
lean_dec(v___x_438_);
lean_dec(v___x_432_);
v___x_446_ = lean_box(0);
v___x_447_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_447_, 0, v___x_446_);
lean_ctor_set(v___x_447_, 1, v_a_426_);
return v___x_447_;
}
else
{
lean_object* v___x_448_; lean_object* v_ref_449_; uint8_t v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; 
v___x_448_ = l_Lean_Syntax_getArg(v___x_438_, v___x_431_);
lean_dec(v___x_438_);
v_ref_449_ = l_Lean_replaceRef(v___x_432_, v_a_425_);
lean_dec(v___x_432_);
v___x_450_ = 0;
v___x_451_ = l_Lean_SourceInfo_fromRef(v_ref_449_, v___x_450_);
lean_dec(v_ref_449_);
v___x_452_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__1));
v___x_453_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__4));
lean_inc_n(v___x_451_, 2);
v___x_454_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_454_, 0, v___x_451_);
lean_ctor_set(v___x_454_, 1, v___x_453_);
v___x_455_ = ((lean_object*)(lp_mathlib_RingTheory_LinearMap_term_u03b7_x5b___x5d___closed__10));
v___x_456_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_456_, 0, v___x_451_);
lean_ctor_set(v___x_456_, 1, v___x_455_);
v___x_457_ = l_Lean_Syntax_node3(v___x_451_, v___x_452_, v___x_454_, v___x_448_, v___x_456_);
v___x_458_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_458_, 0, v___x_457_);
lean_ctor_set(v___x_458_, 1, v_a_426_);
return v___x_458_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______unexpand__Algebra__linearMap__2___boxed(lean_object* v_x_459_, lean_object* v_a_460_, lean_object* v_a_461_){
_start:
{
lean_object* v_res_462_; 
v_res_462_ = lp_mathlib_RingTheory_LinearMap___aux__Mathlib__Algebra__Algebra__Defs______unexpand__Algebra__linearMap__2(v_x_459_, v_a_460_, v_a_461_);
lean_dec(v_a_460_);
return v_res_462_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_id___redArg(lean_object* v_inst_464_){
_start:
{
lean_object* v___x_465_; lean_object* v_toMul_466_; lean_object* v___x_468_; uint8_t v_isShared_469_; uint8_t v_isSharedCheck_475_; 
v___x_465_ = lp_mathlib_instDistribOfSemiring___redArg(v_inst_464_);
v_toMul_466_ = lean_ctor_get(v___x_465_, 0);
v_isSharedCheck_475_ = !lean_is_exclusive(v___x_465_);
if (v_isSharedCheck_475_ == 0)
{
lean_object* v_unused_476_; 
v_unused_476_ = lean_ctor_get(v___x_465_, 1);
lean_dec(v_unused_476_);
v___x_468_ = v___x_465_;
v_isShared_469_ = v_isSharedCheck_475_;
goto v_resetjp_467_;
}
else
{
lean_inc(v_toMul_466_);
lean_dec(v___x_465_);
v___x_468_ = lean_box(0);
v_isShared_469_ = v_isSharedCheck_475_;
goto v_resetjp_467_;
}
v_resetjp_467_:
{
lean_object* v___f_470_; lean_object* v___f_471_; lean_object* v___x_473_; 
v___f_470_ = ((lean_object*)(lp_mathlib_Algebra_id___redArg___closed__0));
v___f_471_ = lean_alloc_closure((void*)(l_instSMulOfMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_471_, 0, v_toMul_466_);
if (v_isShared_469_ == 0)
{
lean_ctor_set(v___x_468_, 1, v___f_470_);
lean_ctor_set(v___x_468_, 0, v___f_471_);
v___x_473_ = v___x_468_;
goto v_reusejp_472_;
}
else
{
lean_object* v_reuseFailAlloc_474_; 
v_reuseFailAlloc_474_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_474_, 0, v___f_471_);
lean_ctor_set(v_reuseFailAlloc_474_, 1, v___f_470_);
v___x_473_ = v_reuseFailAlloc_474_;
goto v_reusejp_472_;
}
v_reusejp_472_:
{
return v___x_473_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Algebra_id(lean_object* v_R_477_, lean_object* v_inst_478_){
_start:
{
lean_object* v___x_479_; 
v___x_479_ = lp_mathlib_Algebra_id___redArg(v_inst_478_);
return v___x_479_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Algebra_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Algebra_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Algebra_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
