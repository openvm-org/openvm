// Lean compiler output
// Module: Mathlib.LinearAlgebra.Span.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Module.Submodule.Lattice public import Mathlib.Algebra.Group.Pointwise.Set.Basic
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
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesIdent(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_span(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_span___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_gi___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Submodule_gi___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Submodule_gi___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_gi___closed__0 = (const lean_object*)&lp_mathlib_Submodule_gi___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_gi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_gi___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Submodule_term___u2219___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Submodule"};
static const lean_object* lp_mathlib_Submodule_term___u2219___00__closed__0 = (const lean_object*)&lp_mathlib_Submodule_term___u2219___00__closed__0_value;
static const lean_string_object lp_mathlib_Submodule_term___u2219___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_∙_"};
static const lean_object* lp_mathlib_Submodule_term___u2219___00__closed__1 = (const lean_object*)&lp_mathlib_Submodule_term___u2219___00__closed__1_value;
static const lean_ctor_object lp_mathlib_Submodule_term___u2219___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Submodule_term___u2219___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(67, 59, 114, 255, 83, 15, 173, 8)}};
static const lean_ctor_object lp_mathlib_Submodule_term___u2219___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Submodule_term___u2219___00__closed__2_value_aux_0),((lean_object*)&lp_mathlib_Submodule_term___u2219___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(184, 241, 202, 91, 130, 118, 218, 166)}};
static const lean_object* lp_mathlib_Submodule_term___u2219___00__closed__2 = (const lean_object*)&lp_mathlib_Submodule_term___u2219___00__closed__2_value;
static const lean_string_object lp_mathlib_Submodule_term___u2219___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Submodule_term___u2219___00__closed__3 = (const lean_object*)&lp_mathlib_Submodule_term___u2219___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Submodule_term___u2219___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Submodule_term___u2219___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Submodule_term___u2219___00__closed__4 = (const lean_object*)&lp_mathlib_Submodule_term___u2219___00__closed__4_value;
static const lean_string_object lp_mathlib_Submodule_term___u2219___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ∙ "};
static const lean_object* lp_mathlib_Submodule_term___u2219___00__closed__5 = (const lean_object*)&lp_mathlib_Submodule_term___u2219___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Submodule_term___u2219___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Submodule_term___u2219___00__closed__5_value)}};
static const lean_object* lp_mathlib_Submodule_term___u2219___00__closed__6 = (const lean_object*)&lp_mathlib_Submodule_term___u2219___00__closed__6_value;
static const lean_string_object lp_mathlib_Submodule_term___u2219___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Submodule_term___u2219___00__closed__7 = (const lean_object*)&lp_mathlib_Submodule_term___u2219___00__closed__7_value;
static const lean_ctor_object lp_mathlib_Submodule_term___u2219___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Submodule_term___u2219___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Submodule_term___u2219___00__closed__8 = (const lean_object*)&lp_mathlib_Submodule_term___u2219___00__closed__8_value;
static const lean_ctor_object lp_mathlib_Submodule_term___u2219___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Submodule_term___u2219___00__closed__8_value),((lean_object*)(((size_t)(70) << 1) | 1))}};
static const lean_object* lp_mathlib_Submodule_term___u2219___00__closed__9 = (const lean_object*)&lp_mathlib_Submodule_term___u2219___00__closed__9_value;
static const lean_ctor_object lp_mathlib_Submodule_term___u2219___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Submodule_term___u2219___00__closed__4_value),((lean_object*)&lp_mathlib_Submodule_term___u2219___00__closed__6_value),((lean_object*)&lp_mathlib_Submodule_term___u2219___00__closed__9_value)}};
static const lean_object* lp_mathlib_Submodule_term___u2219___00__closed__10 = (const lean_object*)&lp_mathlib_Submodule_term___u2219___00__closed__10_value;
static const lean_ctor_object lp_mathlib_Submodule_term___u2219___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_Submodule_term___u2219___00__closed__2_value),((lean_object*)(((size_t)(70) << 1) | 1)),((lean_object*)(((size_t)(70) << 1) | 1)),((lean_object*)&lp_mathlib_Submodule_term___u2219___00__closed__10_value)}};
static const lean_object* lp_mathlib_Submodule_term___u2219___00__closed__11 = (const lean_object*)&lp_mathlib_Submodule_term___u2219___00__closed__11_value;
LEAN_EXPORT const lean_object* lp_mathlib_Submodule_term___u2219__ = (const lean_object*)&lp_mathlib_Submodule_term___u2219___00__closed__11_value;
static const lean_string_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__0 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__0_value;
static const lean_string_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__1 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__1_value;
static const lean_string_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__2 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__2_value;
static const lean_string_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__3 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__4 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__4_value;
static const lean_string_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "span"};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__5 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__5_value;
static lean_once_cell_t lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__6;
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(180, 108, 82, 22, 163, 40, 219, 84)}};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__7 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__7_value;
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Submodule_term___u2219___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(67, 59, 114, 255, 83, 15, 173, 8)}};
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(76, 41, 35, 225, 84, 44, 177, 172)}};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__8 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__8_value;
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__9 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__9_value;
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__10 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__10_value;
static const lean_string_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__11 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__11_value;
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__12 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__12_value;
static const lean_string_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__13 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__13_value;
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__14_value_aux_1),((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__14_value_aux_2),((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(124, 9, 161, 194, 227, 100, 20, 110)}};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__14 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__14_value;
static const lean_string_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__15 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__15_value;
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__16_value_aux_1),((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__16_value_aux_2),((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__16 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__16_value;
static const lean_string_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__17 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__17_value;
static const lean_string_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__18 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__18_value;
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__18_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__19 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__19_value;
static const lean_string_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__20 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__20_value;
static lean_once_cell_t lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__21;
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Submodule_term___u2219___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(67, 59, 114, 255, 83, 15, 173, 8)}};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__22 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__22_value;
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__22_value)}};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__23 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__23_value;
static const lean_string_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Set"};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__24 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__24_value;
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__24_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__25 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__25_value;
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__25_value)}};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__26 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__26_value;
static const lean_string_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Function"};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__27 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__27_value;
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__27_value),LEAN_SCALAR_PTR_LITERAL(225, 8, 186, 189, 152, 89, 197, 12)}};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__28 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__28_value;
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__28_value)}};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__29 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__29_value;
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__29_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__30 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__30_value;
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__26_value),((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__30_value)}};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__31 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__31_value;
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__23_value),((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__31_value)}};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__32 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__32_value;
static const lean_string_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "singleton"};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__33 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__33_value;
static lean_once_cell_t lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__34_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__34;
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__33_value),LEAN_SCALAR_PTR_LITERAL(208, 33, 246, 107, 223, 5, 156, 82)}};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__35 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__35_value;
static const lean_string_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Singleton"};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__36 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__36_value;
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__37_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__36_value),LEAN_SCALAR_PTR_LITERAL(190, 73, 36, 155, 228, 35, 161, 122)}};
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__37_value_aux_0),((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__33_value),LEAN_SCALAR_PTR_LITERAL(185, 48, 115, 60, 21, 14, 217, 215)}};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__37 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__37_value;
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__37_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__38 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__38_value;
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__38_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__39 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__39_value;
static const lean_string_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__40 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__40_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______unexpand__Submodule__span__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______unexpand__Submodule__span__1___closed__0 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______unexpand__Submodule__span__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______unexpand__Submodule__span__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______unexpand__Submodule__span__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______unexpand__Submodule__span__1___closed__1 = (const lean_object*)&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______unexpand__Submodule__span__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______unexpand__Submodule__span__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______unexpand__Submodule__span__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Submodule_span_unexpander___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "term{_}"};
static const lean_object* lp_mathlib_Submodule_span_unexpander___closed__0 = (const lean_object*)&lp_mathlib_Submodule_span_unexpander___closed__0_value;
static const lean_ctor_object lp_mathlib_Submodule_span_unexpander___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Submodule_span_unexpander___closed__0_value),LEAN_SCALAR_PTR_LITERAL(225, 26, 220, 95, 138, 254, 219, 101)}};
static const lean_object* lp_mathlib_Submodule_span_unexpander___closed__1 = (const lean_object*)&lp_mathlib_Submodule_span_unexpander___closed__1_value;
static const lean_string_object lp_mathlib_Submodule_span_unexpander___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "∙"};
static const lean_object* lp_mathlib_Submodule_span_unexpander___closed__2 = (const lean_object*)&lp_mathlib_Submodule_span_unexpander___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_span_unexpander(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_span_unexpander___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_span(lean_object* v_R_1_, lean_object* v_M_2_, lean_object* v_inst_3_, lean_object* v_inst_4_, lean_object* v_inst_5_, lean_object* v_s_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lean_box(0);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_span___boxed(lean_object* v_R_8_, lean_object* v_M_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_s_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_Submodule_span(v_R_8_, v_M_9_, v_inst_10_, v_inst_11_, v_inst_12_, v_s_13_);
lean_dec(v_inst_12_);
lean_dec_ref(v_inst_11_);
lean_dec_ref(v_inst_10_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_gi___lam__0(lean_object* v_s_15_, lean_object* v_x_16_){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = lean_box(0);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_gi(lean_object* v_R_19_, lean_object* v_M_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_inst_23_){
_start:
{
lean_object* v___f_24_; 
v___f_24_ = ((lean_object*)(lp_mathlib_Submodule_gi___closed__0));
return v___f_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_gi___boxed(lean_object* v_R_25_, lean_object* v_M_26_, lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_inst_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_mathlib_Submodule_gi(v_R_25_, v_M_26_, v_inst_27_, v_inst_28_, v_inst_29_);
lean_dec(v_inst_29_);
lean_dec_ref(v_inst_28_);
lean_dec_ref(v_inst_27_);
return v_res_30_;
}
}
static lean_object* _init_lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__6(void){
_start:
{
lean_object* v___x_67_; lean_object* v___x_68_; 
v___x_67_ = ((lean_object*)(lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__5));
v___x_68_ = l_String_toRawSubstring_x27(v___x_67_);
return v___x_68_;
}
}
static lean_object* _init_lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__21(void){
_start:
{
lean_object* v___x_100_; lean_object* v___x_101_; 
v___x_100_ = ((lean_object*)(lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__20));
v___x_101_ = l_String_toRawSubstring_x27(v___x_100_);
return v___x_101_;
}
}
static lean_object* _init_lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__34(void){
_start:
{
lean_object* v___x_126_; lean_object* v___x_127_; 
v___x_126_ = ((lean_object*)(lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__33));
v___x_127_ = l_String_toRawSubstring_x27(v___x_126_);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1(lean_object* v_x_141_, lean_object* v_a_142_, lean_object* v_a_143_){
_start:
{
lean_object* v___x_144_; uint8_t v___x_145_; 
v___x_144_ = ((lean_object*)(lp_mathlib_Submodule_term___u2219___00__closed__2));
lean_inc(v_x_141_);
v___x_145_ = l_Lean_Syntax_isOfKind(v_x_141_, v___x_144_);
if (v___x_145_ == 0)
{
lean_object* v___x_146_; lean_object* v___x_147_; 
lean_dec(v_x_141_);
v___x_146_ = lean_box(1);
v___x_147_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_147_, 0, v___x_146_);
lean_ctor_set(v___x_147_, 1, v_a_143_);
return v___x_147_;
}
else
{
lean_object* v_quotContext_148_; lean_object* v_currMacroScope_149_; lean_object* v_ref_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; uint8_t v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; 
v_quotContext_148_ = lean_ctor_get(v_a_142_, 1);
v_currMacroScope_149_ = lean_ctor_get(v_a_142_, 2);
v_ref_150_ = lean_ctor_get(v_a_142_, 5);
v___x_151_ = lean_unsigned_to_nat(0u);
v___x_152_ = l_Lean_Syntax_getArg(v_x_141_, v___x_151_);
v___x_153_ = lean_unsigned_to_nat(2u);
v___x_154_ = l_Lean_Syntax_getArg(v_x_141_, v___x_153_);
lean_dec(v_x_141_);
v___x_155_ = 0;
v___x_156_ = l_Lean_SourceInfo_fromRef(v_ref_150_, v___x_155_);
v___x_157_ = ((lean_object*)(lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__4));
v___x_158_ = lean_obj_once(&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__6, &lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__6_once, _init_lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__6);
v___x_159_ = ((lean_object*)(lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__7));
lean_inc_n(v_currMacroScope_149_, 3);
lean_inc_n(v_quotContext_148_, 3);
v___x_160_ = l_Lean_addMacroScope(v_quotContext_148_, v___x_159_, v_currMacroScope_149_);
v___x_161_ = ((lean_object*)(lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__10));
lean_inc_n(v___x_156_, 11);
v___x_162_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_162_, 0, v___x_156_);
lean_ctor_set(v___x_162_, 1, v___x_158_);
lean_ctor_set(v___x_162_, 2, v___x_160_);
lean_ctor_set(v___x_162_, 3, v___x_161_);
v___x_163_ = ((lean_object*)(lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__12));
v___x_164_ = ((lean_object*)(lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__14));
v___x_165_ = ((lean_object*)(lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__16));
v___x_166_ = ((lean_object*)(lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__17));
v___x_167_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_167_, 0, v___x_156_);
lean_ctor_set(v___x_167_, 1, v___x_166_);
v___x_168_ = ((lean_object*)(lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__19));
v___x_169_ = lean_obj_once(&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__21, &lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__21_once, _init_lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__21);
v___x_170_ = lean_box(0);
v___x_171_ = l_Lean_addMacroScope(v_quotContext_148_, v___x_170_, v_currMacroScope_149_);
v___x_172_ = ((lean_object*)(lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__32));
v___x_173_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_173_, 0, v___x_156_);
lean_ctor_set(v___x_173_, 1, v___x_169_);
lean_ctor_set(v___x_173_, 2, v___x_171_);
lean_ctor_set(v___x_173_, 3, v___x_172_);
v___x_174_ = l_Lean_Syntax_node1(v___x_156_, v___x_168_, v___x_173_);
v___x_175_ = l_Lean_Syntax_node2(v___x_156_, v___x_165_, v___x_167_, v___x_174_);
v___x_176_ = lean_obj_once(&lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__34, &lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__34_once, _init_lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__34);
v___x_177_ = ((lean_object*)(lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__35));
v___x_178_ = l_Lean_addMacroScope(v_quotContext_148_, v___x_177_, v_currMacroScope_149_);
v___x_179_ = ((lean_object*)(lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__39));
v___x_180_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_180_, 0, v___x_156_);
lean_ctor_set(v___x_180_, 1, v___x_176_);
lean_ctor_set(v___x_180_, 2, v___x_178_);
lean_ctor_set(v___x_180_, 3, v___x_179_);
v___x_181_ = l_Lean_Syntax_node1(v___x_156_, v___x_163_, v___x_154_);
v___x_182_ = l_Lean_Syntax_node2(v___x_156_, v___x_157_, v___x_180_, v___x_181_);
v___x_183_ = ((lean_object*)(lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__40));
v___x_184_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_184_, 0, v___x_156_);
lean_ctor_set(v___x_184_, 1, v___x_183_);
v___x_185_ = l_Lean_Syntax_node3(v___x_156_, v___x_164_, v___x_175_, v___x_182_, v___x_184_);
v___x_186_ = l_Lean_Syntax_node2(v___x_156_, v___x_163_, v___x_152_, v___x_185_);
v___x_187_ = l_Lean_Syntax_node2(v___x_156_, v___x_157_, v___x_162_, v___x_186_);
v___x_188_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_188_, 0, v___x_187_);
lean_ctor_set(v___x_188_, 1, v_a_143_);
return v___x_188_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___boxed(lean_object* v_x_189_, lean_object* v_a_190_, lean_object* v_a_191_){
_start:
{
lean_object* v_res_192_; 
v_res_192_ = lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1(v_x_189_, v_a_190_, v_a_191_);
lean_dec_ref(v_a_190_);
return v_res_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______unexpand__Submodule__span__1(lean_object* v_x_196_, lean_object* v_a_197_, lean_object* v_a_198_){
_start:
{
lean_object* v___x_199_; uint8_t v___x_200_; 
v___x_199_ = ((lean_object*)(lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__4));
lean_inc(v_x_196_);
v___x_200_ = l_Lean_Syntax_isOfKind(v_x_196_, v___x_199_);
if (v___x_200_ == 0)
{
lean_object* v___x_201_; lean_object* v___x_202_; 
lean_dec(v_x_196_);
v___x_201_ = lean_box(0);
v___x_202_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_202_, 0, v___x_201_);
lean_ctor_set(v___x_202_, 1, v_a_198_);
return v___x_202_;
}
else
{
lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; uint8_t v___x_206_; 
v___x_203_ = lean_unsigned_to_nat(0u);
v___x_204_ = l_Lean_Syntax_getArg(v_x_196_, v___x_203_);
v___x_205_ = ((lean_object*)(lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______unexpand__Submodule__span__1___closed__1));
lean_inc(v___x_204_);
v___x_206_ = l_Lean_Syntax_isOfKind(v___x_204_, v___x_205_);
if (v___x_206_ == 0)
{
lean_object* v___x_207_; lean_object* v___x_208_; 
lean_dec(v___x_204_);
lean_dec(v_x_196_);
v___x_207_ = lean_box(0);
v___x_208_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_208_, 0, v___x_207_);
lean_ctor_set(v___x_208_, 1, v_a_198_);
return v___x_208_;
}
else
{
lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; uint8_t v___x_212_; 
v___x_209_ = lean_unsigned_to_nat(1u);
v___x_210_ = l_Lean_Syntax_getArg(v_x_196_, v___x_209_);
lean_dec(v_x_196_);
v___x_211_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_210_);
v___x_212_ = l_Lean_Syntax_matchesNull(v___x_210_, v___x_211_);
if (v___x_212_ == 0)
{
lean_object* v___x_213_; lean_object* v___x_214_; 
lean_dec(v___x_210_);
lean_dec(v___x_204_);
v___x_213_ = lean_box(0);
v___x_214_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_214_, 0, v___x_213_);
lean_ctor_set(v___x_214_, 1, v_a_198_);
return v___x_214_;
}
else
{
lean_object* v___x_215_; uint8_t v___x_216_; 
v___x_215_ = l_Lean_Syntax_getArg(v___x_210_, v___x_209_);
lean_inc(v___x_215_);
v___x_216_ = l_Lean_Syntax_isOfKind(v___x_215_, v___x_199_);
if (v___x_216_ == 0)
{
lean_object* v___x_217_; lean_object* v___x_218_; 
lean_dec(v___x_215_);
lean_dec(v___x_210_);
lean_dec(v___x_204_);
v___x_217_ = lean_box(0);
v___x_218_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_218_, 0, v___x_217_);
lean_ctor_set(v___x_218_, 1, v_a_198_);
return v___x_218_;
}
else
{
lean_object* v___x_219_; lean_object* v___x_220_; uint8_t v___x_221_; 
v___x_219_ = l_Lean_Syntax_getArg(v___x_215_, v___x_203_);
v___x_220_ = ((lean_object*)(lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__35));
v___x_221_ = l_Lean_Syntax_matchesIdent(v___x_219_, v___x_220_);
lean_dec(v___x_219_);
if (v___x_221_ == 0)
{
lean_object* v___x_222_; lean_object* v___x_223_; 
lean_dec(v___x_215_);
lean_dec(v___x_210_);
lean_dec(v___x_204_);
v___x_222_ = lean_box(0);
v___x_223_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_223_, 0, v___x_222_);
lean_ctor_set(v___x_223_, 1, v_a_198_);
return v___x_223_;
}
else
{
lean_object* v___x_224_; uint8_t v___x_225_; 
v___x_224_ = l_Lean_Syntax_getArg(v___x_215_, v___x_209_);
lean_dec(v___x_215_);
lean_inc(v___x_224_);
v___x_225_ = l_Lean_Syntax_matchesNull(v___x_224_, v___x_209_);
if (v___x_225_ == 0)
{
lean_object* v___x_226_; lean_object* v___x_227_; 
lean_dec(v___x_224_);
lean_dec(v___x_210_);
lean_dec(v___x_204_);
v___x_226_ = lean_box(0);
v___x_227_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_227_, 0, v___x_226_);
lean_ctor_set(v___x_227_, 1, v_a_198_);
return v___x_227_;
}
else
{
lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v_ref_230_; uint8_t v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; 
v___x_228_ = l_Lean_Syntax_getArg(v___x_210_, v___x_203_);
lean_dec(v___x_210_);
v___x_229_ = l_Lean_Syntax_getArg(v___x_224_, v___x_203_);
lean_dec(v___x_224_);
v_ref_230_ = l_Lean_replaceRef(v___x_204_, v_a_197_);
lean_dec(v___x_204_);
v___x_231_ = 0;
v___x_232_ = l_Lean_SourceInfo_fromRef(v_ref_230_, v___x_231_);
lean_dec(v_ref_230_);
v___x_233_ = ((lean_object*)(lp_mathlib_Submodule_term___u2219___00__closed__2));
v___x_234_ = ((lean_object*)(lp_mathlib_Submodule_term___u2219___00__closed__5));
lean_inc(v___x_232_);
v___x_235_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_235_, 0, v___x_232_);
lean_ctor_set(v___x_235_, 1, v___x_234_);
v___x_236_ = l_Lean_Syntax_node3(v___x_232_, v___x_233_, v___x_228_, v___x_235_, v___x_229_);
v___x_237_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_237_, 0, v___x_236_);
lean_ctor_set(v___x_237_, 1, v_a_198_);
return v___x_237_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______unexpand__Submodule__span__1___boxed(lean_object* v_x_238_, lean_object* v_a_239_, lean_object* v_a_240_){
_start:
{
lean_object* v_res_241_; 
v_res_241_ = lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______unexpand__Submodule__span__1(v_x_238_, v_a_239_, v_a_240_);
lean_dec(v_a_239_);
return v_res_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_span_unexpander(lean_object* v_x_246_, lean_object* v_a_247_, lean_object* v_a_248_){
_start:
{
lean_object* v___x_249_; uint8_t v___x_250_; 
v___x_249_ = ((lean_object*)(lp_mathlib_Submodule___aux__Mathlib__LinearAlgebra__Span__Defs______macroRules__Submodule__term___u2219____1___closed__4));
lean_inc(v_x_246_);
v___x_250_ = l_Lean_Syntax_isOfKind(v_x_246_, v___x_249_);
if (v___x_250_ == 0)
{
lean_object* v___x_251_; lean_object* v___x_252_; 
lean_dec(v_x_246_);
v___x_251_ = lean_box(0);
v___x_252_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_252_, 0, v___x_251_);
lean_ctor_set(v___x_252_, 1, v_a_248_);
return v___x_252_;
}
else
{
lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; uint8_t v___x_256_; 
v___x_253_ = lean_unsigned_to_nat(1u);
v___x_254_ = l_Lean_Syntax_getArg(v_x_246_, v___x_253_);
lean_dec(v_x_246_);
v___x_255_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_254_);
v___x_256_ = l_Lean_Syntax_matchesNull(v___x_254_, v___x_255_);
if (v___x_256_ == 0)
{
lean_object* v___x_257_; lean_object* v___x_258_; 
lean_dec(v___x_254_);
v___x_257_ = lean_box(0);
v___x_258_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_258_, 0, v___x_257_);
lean_ctor_set(v___x_258_, 1, v_a_248_);
return v___x_258_;
}
else
{
lean_object* v___x_259_; lean_object* v___x_260_; uint8_t v___x_261_; 
v___x_259_ = l_Lean_Syntax_getArg(v___x_254_, v___x_253_);
v___x_260_ = ((lean_object*)(lp_mathlib_Submodule_span_unexpander___closed__1));
lean_inc(v___x_259_);
v___x_261_ = l_Lean_Syntax_isOfKind(v___x_259_, v___x_260_);
if (v___x_261_ == 0)
{
lean_object* v___x_262_; lean_object* v___x_263_; 
lean_dec(v___x_259_);
lean_dec(v___x_254_);
v___x_262_ = lean_box(0);
v___x_263_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_263_, 0, v___x_262_);
lean_ctor_set(v___x_263_, 1, v_a_248_);
return v___x_263_;
}
else
{
lean_object* v___x_264_; uint8_t v___x_265_; 
v___x_264_ = l_Lean_Syntax_getArg(v___x_259_, v___x_253_);
lean_dec(v___x_259_);
lean_inc(v___x_264_);
v___x_265_ = l_Lean_Syntax_matchesNull(v___x_264_, v___x_253_);
if (v___x_265_ == 0)
{
lean_object* v___x_266_; lean_object* v___x_267_; 
lean_dec(v___x_264_);
lean_dec(v___x_254_);
v___x_266_ = lean_box(0);
v___x_267_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_267_, 0, v___x_266_);
lean_ctor_set(v___x_267_, 1, v_a_248_);
return v___x_267_;
}
else
{
lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; uint8_t v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; 
v___x_268_ = lean_unsigned_to_nat(0u);
v___x_269_ = l_Lean_Syntax_getArg(v___x_254_, v___x_268_);
lean_dec(v___x_254_);
v___x_270_ = l_Lean_Syntax_getArg(v___x_264_, v___x_268_);
lean_dec(v___x_264_);
v___x_271_ = 0;
v___x_272_ = l_Lean_SourceInfo_fromRef(v_a_247_, v___x_271_);
v___x_273_ = ((lean_object*)(lp_mathlib_Submodule_term___u2219___00__closed__2));
v___x_274_ = ((lean_object*)(lp_mathlib_Submodule_span_unexpander___closed__2));
lean_inc(v___x_272_);
v___x_275_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_275_, 0, v___x_272_);
lean_ctor_set(v___x_275_, 1, v___x_274_);
v___x_276_ = l_Lean_Syntax_node3(v___x_272_, v___x_273_, v___x_269_, v___x_275_, v___x_270_);
v___x_277_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_277_, 0, v___x_276_);
lean_ctor_set(v___x_277_, 1, v_a_248_);
return v___x_277_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_span_unexpander___boxed(lean_object* v_x_278_, lean_object* v_a_279_, lean_object* v_a_280_){
_start:
{
lean_object* v_res_281_; 
v_res_281_ = lp_mathlib_Submodule_span_unexpander(v_x_278_, v_a_279_, v_a_280_);
lean_dec(v_a_279_);
return v_res_281_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Lattice(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Span_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Submodule_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_LinearAlgebra_Span_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_Lattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Span_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Submodule_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pointwise_Set_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Span_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_LinearAlgebra_Span_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_LinearAlgebra_Span_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
