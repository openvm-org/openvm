// Lean compiler output
// Module: Mathlib.Algebra.Notation
// Imports: public import Init public meta import Init public import Mathlib.Tactic.Translate.ToAdditive
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u207a_u1d50___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 7, .m_data = "term_⁺ᵐ"};
static const lean_object* lp_mathlib_term___u207a_u1d50___closed__0 = (const lean_object*)&lp_mathlib_term___u207a_u1d50___closed__0_value;
static const lean_ctor_object lp_mathlib_term___u207a_u1d50___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u207a_u1d50___closed__0_value),LEAN_SCALAR_PTR_LITERAL(236, 246, 37, 150, 121, 29, 208, 121)}};
static const lean_object* lp_mathlib_term___u207a_u1d50___closed__1 = (const lean_object*)&lp_mathlib_term___u207a_u1d50___closed__1_value;
static const lean_string_object lp_mathlib_term___u207a_u1d50___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 2, .m_data = "⁺ᵐ"};
static const lean_object* lp_mathlib_term___u207a_u1d50___closed__2 = (const lean_object*)&lp_mathlib_term___u207a_u1d50___closed__2_value;
static const lean_ctor_object lp_mathlib_term___u207a_u1d50___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u207a_u1d50___closed__2_value)}};
static const lean_object* lp_mathlib_term___u207a_u1d50___closed__3 = (const lean_object*)&lp_mathlib_term___u207a_u1d50___closed__3_value;
static const lean_ctor_object lp_mathlib_term___u207a_u1d50___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u207a_u1d50___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_term___u207a_u1d50___closed__3_value)}};
static const lean_object* lp_mathlib_term___u207a_u1d50___closed__4 = (const lean_object*)&lp_mathlib_term___u207a_u1d50___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u207a_u1d50 = (const lean_object*)&lp_mathlib_term___u207a_u1d50___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "OneLePart.oneLePart"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__6;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "OneLePart"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__7_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "oneLePart"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(213, 166, 223, 233, 149, 195, 112, 69)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__9_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(44, 2, 169, 99, 167, 45, 132, 148)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__9_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__11_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__12_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__OneLePart__oneLePart__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__OneLePart__oneLePart__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__OneLePart__oneLePart__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__OneLePart__oneLePart__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__OneLePart__oneLePart__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__OneLePart__oneLePart__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__OneLePart__oneLePart__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__OneLePart__oneLePart__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__OneLePart__oneLePart__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u207b_u1d50___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 7, .m_data = "term_⁻ᵐ"};
static const lean_object* lp_mathlib_term___u207b_u1d50___closed__0 = (const lean_object*)&lp_mathlib_term___u207b_u1d50___closed__0_value;
static const lean_ctor_object lp_mathlib_term___u207b_u1d50___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u207b_u1d50___closed__0_value),LEAN_SCALAR_PTR_LITERAL(14, 24, 92, 184, 141, 89, 56, 2)}};
static const lean_object* lp_mathlib_term___u207b_u1d50___closed__1 = (const lean_object*)&lp_mathlib_term___u207b_u1d50___closed__1_value;
static const lean_string_object lp_mathlib_term___u207b_u1d50___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 2, .m_data = "⁻ᵐ"};
static const lean_object* lp_mathlib_term___u207b_u1d50___closed__2 = (const lean_object*)&lp_mathlib_term___u207b_u1d50___closed__2_value;
static const lean_ctor_object lp_mathlib_term___u207b_u1d50___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u207b_u1d50___closed__2_value)}};
static const lean_object* lp_mathlib_term___u207b_u1d50___closed__3 = (const lean_object*)&lp_mathlib_term___u207b_u1d50___closed__3_value;
static const lean_ctor_object lp_mathlib_term___u207b_u1d50___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u207b_u1d50___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_term___u207b_u1d50___closed__3_value)}};
static const lean_object* lp_mathlib_term___u207b_u1d50___closed__4 = (const lean_object*)&lp_mathlib_term___u207b_u1d50___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u207b_u1d50 = (const lean_object*)&lp_mathlib_term___u207b_u1d50___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "LeOnePart.leOnePart"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__1;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "LeOnePart"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "leOnePart"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(133, 77, 234, 131, 64, 105, 109, 31)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(115, 116, 185, 47, 71, 146, 3, 115)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__LeOnePart__leOnePart__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__LeOnePart__leOnePart__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u207a___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 6, .m_data = "term_⁺"};
static const lean_object* lp_mathlib_term___u207a___closed__0 = (const lean_object*)&lp_mathlib_term___u207a___closed__0_value;
static const lean_ctor_object lp_mathlib_term___u207a___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u207a___closed__0_value),LEAN_SCALAR_PTR_LITERAL(80, 0, 65, 250, 215, 22, 207, 100)}};
static const lean_object* lp_mathlib_term___u207a___closed__1 = (const lean_object*)&lp_mathlib_term___u207a___closed__1_value;
static const lean_string_object lp_mathlib_term___u207a___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⁺"};
static const lean_object* lp_mathlib_term___u207a___closed__2 = (const lean_object*)&lp_mathlib_term___u207a___closed__2_value;
static const lean_ctor_object lp_mathlib_term___u207a___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u207a___closed__2_value)}};
static const lean_object* lp_mathlib_term___u207a___closed__3 = (const lean_object*)&lp_mathlib_term___u207a___closed__3_value;
static const lean_ctor_object lp_mathlib_term___u207a___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u207a___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_term___u207a___closed__3_value)}};
static const lean_object* lp_mathlib_term___u207a___closed__4 = (const lean_object*)&lp_mathlib_term___u207a___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u207a = (const lean_object*)&lp_mathlib_term___u207a___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "PosPart.posPart"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__1;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "PosPart"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "posPart"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(20, 1, 20, 98, 145, 139, 205, 125)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(34, 153, 124, 111, 50, 90, 248, 78)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__PosPart__posPart__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__PosPart__posPart__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u207b___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 6, .m_data = "term_⁻"};
static const lean_object* lp_mathlib_term___u207b___closed__0 = (const lean_object*)&lp_mathlib_term___u207b___closed__0_value;
static const lean_ctor_object lp_mathlib_term___u207b___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u207b___closed__0_value),LEAN_SCALAR_PTR_LITERAL(227, 18, 255, 196, 68, 205, 66, 58)}};
static const lean_object* lp_mathlib_term___u207b___closed__1 = (const lean_object*)&lp_mathlib_term___u207b___closed__1_value;
static const lean_string_object lp_mathlib_term___u207b___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⁻"};
static const lean_object* lp_mathlib_term___u207b___closed__2 = (const lean_object*)&lp_mathlib_term___u207b___closed__2_value;
static const lean_ctor_object lp_mathlib_term___u207b___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u207b___closed__2_value)}};
static const lean_object* lp_mathlib_term___u207b___closed__3 = (const lean_object*)&lp_mathlib_term___u207b___closed__3_value;
static const lean_ctor_object lp_mathlib_term___u207b___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u207b___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_term___u207b___closed__3_value)}};
static const lean_object* lp_mathlib_term___u207b___closed__4 = (const lean_object*)&lp_mathlib_term___u207b___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u207b = (const lean_object*)&lp_mathlib_term___u207b___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "NegPart.negPart"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__1;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "NegPart"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "negPart"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(194, 253, 249, 1, 193, 108, 17, 166)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(91, 185, 135, 97, 6, 181, 117, 233)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__NegPart__negPart__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__NegPart__negPart__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__6(void){
_start:
{
lean_object* v___x_22_; lean_object* v___x_23_; 
v___x_22_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__5));
v___x_23_ = l_String_toRawSubstring_x27(v___x_22_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1(lean_object* v_x_38_, lean_object* v_a_39_, lean_object* v_a_40_){
_start:
{
lean_object* v___x_41_; uint8_t v___x_42_; 
v___x_41_ = ((lean_object*)(lp_mathlib_term___u207a_u1d50___closed__1));
lean_inc(v_x_38_);
v___x_42_ = l_Lean_Syntax_isOfKind(v_x_38_, v___x_41_);
if (v___x_42_ == 0)
{
lean_object* v___x_43_; lean_object* v___x_44_; 
lean_dec(v_x_38_);
v___x_43_ = lean_box(1);
v___x_44_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_44_, 0, v___x_43_);
lean_ctor_set(v___x_44_, 1, v_a_40_);
return v___x_44_;
}
else
{
lean_object* v_quotContext_45_; lean_object* v_currMacroScope_46_; lean_object* v_ref_47_; lean_object* v___x_48_; lean_object* v___x_49_; uint8_t v___x_50_; lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; 
v_quotContext_45_ = lean_ctor_get(v_a_39_, 1);
v_currMacroScope_46_ = lean_ctor_get(v_a_39_, 2);
v_ref_47_ = lean_ctor_get(v_a_39_, 5);
v___x_48_ = lean_unsigned_to_nat(0u);
v___x_49_ = l_Lean_Syntax_getArg(v_x_38_, v___x_48_);
lean_dec(v_x_38_);
v___x_50_ = 0;
v___x_51_ = l_Lean_SourceInfo_fromRef(v_ref_47_, v___x_50_);
v___x_52_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__4));
v___x_53_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__6, &lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__6);
v___x_54_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__9));
lean_inc(v_currMacroScope_46_);
lean_inc(v_quotContext_45_);
v___x_55_ = l_Lean_addMacroScope(v_quotContext_45_, v___x_54_, v_currMacroScope_46_);
v___x_56_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__11));
lean_inc_n(v___x_51_, 2);
v___x_57_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_57_, 0, v___x_51_);
lean_ctor_set(v___x_57_, 1, v___x_53_);
lean_ctor_set(v___x_57_, 2, v___x_55_);
lean_ctor_set(v___x_57_, 3, v___x_56_);
v___x_58_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__13));
v___x_59_ = l_Lean_Syntax_node1(v___x_51_, v___x_58_, v___x_49_);
v___x_60_ = l_Lean_Syntax_node2(v___x_51_, v___x_52_, v___x_57_, v___x_59_);
v___x_61_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_61_, 0, v___x_60_);
lean_ctor_set(v___x_61_, 1, v_a_40_);
return v___x_61_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___boxed(lean_object* v_x_62_, lean_object* v_a_63_, lean_object* v_a_64_){
_start:
{
lean_object* v_res_65_; 
v_res_65_ = lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1(v_x_62_, v_a_63_, v_a_64_);
lean_dec_ref(v_a_63_);
return v_res_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__OneLePart__oneLePart__1(lean_object* v_x_69_, lean_object* v_a_70_, lean_object* v_a_71_){
_start:
{
lean_object* v___x_72_; uint8_t v___x_73_; 
v___x_72_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__4));
lean_inc(v_x_69_);
v___x_73_ = l_Lean_Syntax_isOfKind(v_x_69_, v___x_72_);
if (v___x_73_ == 0)
{
lean_object* v___x_74_; lean_object* v___x_75_; 
lean_dec(v_x_69_);
v___x_74_ = lean_box(0);
v___x_75_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_75_, 0, v___x_74_);
lean_ctor_set(v___x_75_, 1, v_a_71_);
return v___x_75_;
}
else
{
lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; uint8_t v___x_79_; 
v___x_76_ = lean_unsigned_to_nat(0u);
v___x_77_ = l_Lean_Syntax_getArg(v_x_69_, v___x_76_);
v___x_78_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__OneLePart__oneLePart__1___closed__1));
lean_inc(v___x_77_);
v___x_79_ = l_Lean_Syntax_isOfKind(v___x_77_, v___x_78_);
if (v___x_79_ == 0)
{
lean_object* v___x_80_; lean_object* v___x_81_; 
lean_dec(v___x_77_);
lean_dec(v_x_69_);
v___x_80_ = lean_box(0);
v___x_81_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_81_, 0, v___x_80_);
lean_ctor_set(v___x_81_, 1, v_a_71_);
return v___x_81_;
}
else
{
lean_object* v___x_82_; lean_object* v___x_83_; uint8_t v___x_84_; 
v___x_82_ = lean_unsigned_to_nat(1u);
v___x_83_ = l_Lean_Syntax_getArg(v_x_69_, v___x_82_);
lean_dec(v_x_69_);
lean_inc(v___x_83_);
v___x_84_ = l_Lean_Syntax_matchesNull(v___x_83_, v___x_82_);
if (v___x_84_ == 0)
{
lean_object* v___x_85_; lean_object* v___x_86_; 
lean_dec(v___x_83_);
lean_dec(v___x_77_);
v___x_85_ = lean_box(0);
v___x_86_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_86_, 0, v___x_85_);
lean_ctor_set(v___x_86_, 1, v_a_71_);
return v___x_86_;
}
else
{
lean_object* v___x_87_; lean_object* v_ref_88_; uint8_t v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; 
v___x_87_ = l_Lean_Syntax_getArg(v___x_83_, v___x_76_);
lean_dec(v___x_83_);
v_ref_88_ = l_Lean_replaceRef(v___x_77_, v_a_70_);
lean_dec(v___x_77_);
v___x_89_ = 0;
v___x_90_ = l_Lean_SourceInfo_fromRef(v_ref_88_, v___x_89_);
lean_dec(v_ref_88_);
v___x_91_ = ((lean_object*)(lp_mathlib_term___u207a_u1d50___closed__1));
v___x_92_ = ((lean_object*)(lp_mathlib_term___u207a_u1d50___closed__2));
lean_inc(v___x_90_);
v___x_93_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_93_, 0, v___x_90_);
lean_ctor_set(v___x_93_, 1, v___x_92_);
v___x_94_ = l_Lean_Syntax_node2(v___x_90_, v___x_91_, v___x_87_, v___x_93_);
v___x_95_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_95_, 0, v___x_94_);
lean_ctor_set(v___x_95_, 1, v_a_71_);
return v___x_95_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__OneLePart__oneLePart__1___boxed(lean_object* v_x_96_, lean_object* v_a_97_, lean_object* v_a_98_){
_start:
{
lean_object* v_res_99_; 
v_res_99_ = lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__OneLePart__oneLePart__1(v_x_96_, v_a_97_, v_a_98_);
lean_dec(v_a_97_);
return v_res_99_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__1(void){
_start:
{
lean_object* v___x_112_; lean_object* v___x_113_; 
v___x_112_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__0));
v___x_113_ = l_String_toRawSubstring_x27(v___x_112_);
return v___x_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1(lean_object* v_x_125_, lean_object* v_a_126_, lean_object* v_a_127_){
_start:
{
lean_object* v___x_128_; uint8_t v___x_129_; 
v___x_128_ = ((lean_object*)(lp_mathlib_term___u207b_u1d50___closed__1));
lean_inc(v_x_125_);
v___x_129_ = l_Lean_Syntax_isOfKind(v_x_125_, v___x_128_);
if (v___x_129_ == 0)
{
lean_object* v___x_130_; lean_object* v___x_131_; 
lean_dec(v_x_125_);
v___x_130_ = lean_box(1);
v___x_131_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_131_, 0, v___x_130_);
lean_ctor_set(v___x_131_, 1, v_a_127_);
return v___x_131_;
}
else
{
lean_object* v_quotContext_132_; lean_object* v_currMacroScope_133_; lean_object* v_ref_134_; lean_object* v___x_135_; lean_object* v___x_136_; uint8_t v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; 
v_quotContext_132_ = lean_ctor_get(v_a_126_, 1);
v_currMacroScope_133_ = lean_ctor_get(v_a_126_, 2);
v_ref_134_ = lean_ctor_get(v_a_126_, 5);
v___x_135_ = lean_unsigned_to_nat(0u);
v___x_136_ = l_Lean_Syntax_getArg(v_x_125_, v___x_135_);
lean_dec(v_x_125_);
v___x_137_ = 0;
v___x_138_ = l_Lean_SourceInfo_fromRef(v_ref_134_, v___x_137_);
v___x_139_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__4));
v___x_140_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__1, &lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__1);
v___x_141_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__4));
lean_inc(v_currMacroScope_133_);
lean_inc(v_quotContext_132_);
v___x_142_ = l_Lean_addMacroScope(v_quotContext_132_, v___x_141_, v_currMacroScope_133_);
v___x_143_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___closed__6));
lean_inc_n(v___x_138_, 2);
v___x_144_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_144_, 0, v___x_138_);
lean_ctor_set(v___x_144_, 1, v___x_140_);
lean_ctor_set(v___x_144_, 2, v___x_142_);
lean_ctor_set(v___x_144_, 3, v___x_143_);
v___x_145_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__13));
v___x_146_ = l_Lean_Syntax_node1(v___x_138_, v___x_145_, v___x_136_);
v___x_147_ = l_Lean_Syntax_node2(v___x_138_, v___x_139_, v___x_144_, v___x_146_);
v___x_148_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_148_, 0, v___x_147_);
lean_ctor_set(v___x_148_, 1, v_a_127_);
return v___x_148_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1___boxed(lean_object* v_x_149_, lean_object* v_a_150_, lean_object* v_a_151_){
_start:
{
lean_object* v_res_152_; 
v_res_152_ = lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b_u1d50__1(v_x_149_, v_a_150_, v_a_151_);
lean_dec_ref(v_a_150_);
return v_res_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__LeOnePart__leOnePart__1(lean_object* v_x_153_, lean_object* v_a_154_, lean_object* v_a_155_){
_start:
{
lean_object* v___x_156_; uint8_t v___x_157_; 
v___x_156_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__4));
lean_inc(v_x_153_);
v___x_157_ = l_Lean_Syntax_isOfKind(v_x_153_, v___x_156_);
if (v___x_157_ == 0)
{
lean_object* v___x_158_; lean_object* v___x_159_; 
lean_dec(v_x_153_);
v___x_158_ = lean_box(0);
v___x_159_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_159_, 0, v___x_158_);
lean_ctor_set(v___x_159_, 1, v_a_155_);
return v___x_159_;
}
else
{
lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; uint8_t v___x_163_; 
v___x_160_ = lean_unsigned_to_nat(0u);
v___x_161_ = l_Lean_Syntax_getArg(v_x_153_, v___x_160_);
v___x_162_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__OneLePart__oneLePart__1___closed__1));
lean_inc(v___x_161_);
v___x_163_ = l_Lean_Syntax_isOfKind(v___x_161_, v___x_162_);
if (v___x_163_ == 0)
{
lean_object* v___x_164_; lean_object* v___x_165_; 
lean_dec(v___x_161_);
lean_dec(v_x_153_);
v___x_164_ = lean_box(0);
v___x_165_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_165_, 0, v___x_164_);
lean_ctor_set(v___x_165_, 1, v_a_155_);
return v___x_165_;
}
else
{
lean_object* v___x_166_; lean_object* v___x_167_; uint8_t v___x_168_; 
v___x_166_ = lean_unsigned_to_nat(1u);
v___x_167_ = l_Lean_Syntax_getArg(v_x_153_, v___x_166_);
lean_dec(v_x_153_);
lean_inc(v___x_167_);
v___x_168_ = l_Lean_Syntax_matchesNull(v___x_167_, v___x_166_);
if (v___x_168_ == 0)
{
lean_object* v___x_169_; lean_object* v___x_170_; 
lean_dec(v___x_167_);
lean_dec(v___x_161_);
v___x_169_ = lean_box(0);
v___x_170_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_170_, 0, v___x_169_);
lean_ctor_set(v___x_170_, 1, v_a_155_);
return v___x_170_;
}
else
{
lean_object* v___x_171_; lean_object* v_ref_172_; uint8_t v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; 
v___x_171_ = l_Lean_Syntax_getArg(v___x_167_, v___x_160_);
lean_dec(v___x_167_);
v_ref_172_ = l_Lean_replaceRef(v___x_161_, v_a_154_);
lean_dec(v___x_161_);
v___x_173_ = 0;
v___x_174_ = l_Lean_SourceInfo_fromRef(v_ref_172_, v___x_173_);
lean_dec(v_ref_172_);
v___x_175_ = ((lean_object*)(lp_mathlib_term___u207b_u1d50___closed__1));
v___x_176_ = ((lean_object*)(lp_mathlib_term___u207b_u1d50___closed__2));
lean_inc(v___x_174_);
v___x_177_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_177_, 0, v___x_174_);
lean_ctor_set(v___x_177_, 1, v___x_176_);
v___x_178_ = l_Lean_Syntax_node2(v___x_174_, v___x_175_, v___x_171_, v___x_177_);
v___x_179_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_179_, 0, v___x_178_);
lean_ctor_set(v___x_179_, 1, v_a_155_);
return v___x_179_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__LeOnePart__leOnePart__1___boxed(lean_object* v_x_180_, lean_object* v_a_181_, lean_object* v_a_182_){
_start:
{
lean_object* v_res_183_; 
v_res_183_ = lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__LeOnePart__leOnePart__1(v_x_180_, v_a_181_, v_a_182_);
lean_dec(v_a_181_);
return v_res_183_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__1(void){
_start:
{
lean_object* v___x_196_; lean_object* v___x_197_; 
v___x_196_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__0));
v___x_197_ = l_String_toRawSubstring_x27(v___x_196_);
return v___x_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1(lean_object* v_x_209_, lean_object* v_a_210_, lean_object* v_a_211_){
_start:
{
lean_object* v___x_212_; uint8_t v___x_213_; 
v___x_212_ = ((lean_object*)(lp_mathlib_term___u207a___closed__1));
lean_inc(v_x_209_);
v___x_213_ = l_Lean_Syntax_isOfKind(v_x_209_, v___x_212_);
if (v___x_213_ == 0)
{
lean_object* v___x_214_; lean_object* v___x_215_; 
lean_dec(v_x_209_);
v___x_214_ = lean_box(1);
v___x_215_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_215_, 0, v___x_214_);
lean_ctor_set(v___x_215_, 1, v_a_211_);
return v___x_215_;
}
else
{
lean_object* v_quotContext_216_; lean_object* v_currMacroScope_217_; lean_object* v_ref_218_; lean_object* v___x_219_; lean_object* v___x_220_; uint8_t v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; 
v_quotContext_216_ = lean_ctor_get(v_a_210_, 1);
v_currMacroScope_217_ = lean_ctor_get(v_a_210_, 2);
v_ref_218_ = lean_ctor_get(v_a_210_, 5);
v___x_219_ = lean_unsigned_to_nat(0u);
v___x_220_ = l_Lean_Syntax_getArg(v_x_209_, v___x_219_);
lean_dec(v_x_209_);
v___x_221_ = 0;
v___x_222_ = l_Lean_SourceInfo_fromRef(v_ref_218_, v___x_221_);
v___x_223_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__4));
v___x_224_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__1, &lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__1);
v___x_225_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__4));
lean_inc(v_currMacroScope_217_);
lean_inc(v_quotContext_216_);
v___x_226_ = l_Lean_addMacroScope(v_quotContext_216_, v___x_225_, v_currMacroScope_217_);
v___x_227_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___closed__6));
lean_inc_n(v___x_222_, 2);
v___x_228_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_228_, 0, v___x_222_);
lean_ctor_set(v___x_228_, 1, v___x_224_);
lean_ctor_set(v___x_228_, 2, v___x_226_);
lean_ctor_set(v___x_228_, 3, v___x_227_);
v___x_229_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__13));
v___x_230_ = l_Lean_Syntax_node1(v___x_222_, v___x_229_, v___x_220_);
v___x_231_ = l_Lean_Syntax_node2(v___x_222_, v___x_223_, v___x_228_, v___x_230_);
v___x_232_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_232_, 0, v___x_231_);
lean_ctor_set(v___x_232_, 1, v_a_211_);
return v___x_232_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1___boxed(lean_object* v_x_233_, lean_object* v_a_234_, lean_object* v_a_235_){
_start:
{
lean_object* v_res_236_; 
v_res_236_ = lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a__1(v_x_233_, v_a_234_, v_a_235_);
lean_dec_ref(v_a_234_);
return v_res_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__PosPart__posPart__1(lean_object* v_x_237_, lean_object* v_a_238_, lean_object* v_a_239_){
_start:
{
lean_object* v___x_240_; uint8_t v___x_241_; 
v___x_240_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__4));
lean_inc(v_x_237_);
v___x_241_ = l_Lean_Syntax_isOfKind(v_x_237_, v___x_240_);
if (v___x_241_ == 0)
{
lean_object* v___x_242_; lean_object* v___x_243_; 
lean_dec(v_x_237_);
v___x_242_ = lean_box(0);
v___x_243_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_243_, 0, v___x_242_);
lean_ctor_set(v___x_243_, 1, v_a_239_);
return v___x_243_;
}
else
{
lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; uint8_t v___x_247_; 
v___x_244_ = lean_unsigned_to_nat(0u);
v___x_245_ = l_Lean_Syntax_getArg(v_x_237_, v___x_244_);
v___x_246_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__OneLePart__oneLePart__1___closed__1));
lean_inc(v___x_245_);
v___x_247_ = l_Lean_Syntax_isOfKind(v___x_245_, v___x_246_);
if (v___x_247_ == 0)
{
lean_object* v___x_248_; lean_object* v___x_249_; 
lean_dec(v___x_245_);
lean_dec(v_x_237_);
v___x_248_ = lean_box(0);
v___x_249_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_249_, 0, v___x_248_);
lean_ctor_set(v___x_249_, 1, v_a_239_);
return v___x_249_;
}
else
{
lean_object* v___x_250_; lean_object* v___x_251_; uint8_t v___x_252_; 
v___x_250_ = lean_unsigned_to_nat(1u);
v___x_251_ = l_Lean_Syntax_getArg(v_x_237_, v___x_250_);
lean_dec(v_x_237_);
lean_inc(v___x_251_);
v___x_252_ = l_Lean_Syntax_matchesNull(v___x_251_, v___x_250_);
if (v___x_252_ == 0)
{
lean_object* v___x_253_; lean_object* v___x_254_; 
lean_dec(v___x_251_);
lean_dec(v___x_245_);
v___x_253_ = lean_box(0);
v___x_254_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_254_, 0, v___x_253_);
lean_ctor_set(v___x_254_, 1, v_a_239_);
return v___x_254_;
}
else
{
lean_object* v___x_255_; lean_object* v_ref_256_; uint8_t v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; 
v___x_255_ = l_Lean_Syntax_getArg(v___x_251_, v___x_244_);
lean_dec(v___x_251_);
v_ref_256_ = l_Lean_replaceRef(v___x_245_, v_a_238_);
lean_dec(v___x_245_);
v___x_257_ = 0;
v___x_258_ = l_Lean_SourceInfo_fromRef(v_ref_256_, v___x_257_);
lean_dec(v_ref_256_);
v___x_259_ = ((lean_object*)(lp_mathlib_term___u207a___closed__1));
v___x_260_ = ((lean_object*)(lp_mathlib_term___u207a___closed__2));
lean_inc(v___x_258_);
v___x_261_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_261_, 0, v___x_258_);
lean_ctor_set(v___x_261_, 1, v___x_260_);
v___x_262_ = l_Lean_Syntax_node2(v___x_258_, v___x_259_, v___x_255_, v___x_261_);
v___x_263_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_263_, 0, v___x_262_);
lean_ctor_set(v___x_263_, 1, v_a_239_);
return v___x_263_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__PosPart__posPart__1___boxed(lean_object* v_x_264_, lean_object* v_a_265_, lean_object* v_a_266_){
_start:
{
lean_object* v_res_267_; 
v_res_267_ = lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__PosPart__posPart__1(v_x_264_, v_a_265_, v_a_266_);
lean_dec(v_a_265_);
return v_res_267_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__1(void){
_start:
{
lean_object* v___x_280_; lean_object* v___x_281_; 
v___x_280_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__0));
v___x_281_ = l_String_toRawSubstring_x27(v___x_280_);
return v___x_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1(lean_object* v_x_293_, lean_object* v_a_294_, lean_object* v_a_295_){
_start:
{
lean_object* v___x_296_; uint8_t v___x_297_; 
v___x_296_ = ((lean_object*)(lp_mathlib_term___u207b___closed__1));
lean_inc(v_x_293_);
v___x_297_ = l_Lean_Syntax_isOfKind(v_x_293_, v___x_296_);
if (v___x_297_ == 0)
{
lean_object* v___x_298_; lean_object* v___x_299_; 
lean_dec(v_x_293_);
v___x_298_ = lean_box(1);
v___x_299_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_299_, 0, v___x_298_);
lean_ctor_set(v___x_299_, 1, v_a_295_);
return v___x_299_;
}
else
{
lean_object* v_quotContext_300_; lean_object* v_currMacroScope_301_; lean_object* v_ref_302_; lean_object* v___x_303_; lean_object* v___x_304_; uint8_t v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; 
v_quotContext_300_ = lean_ctor_get(v_a_294_, 1);
v_currMacroScope_301_ = lean_ctor_get(v_a_294_, 2);
v_ref_302_ = lean_ctor_get(v_a_294_, 5);
v___x_303_ = lean_unsigned_to_nat(0u);
v___x_304_ = l_Lean_Syntax_getArg(v_x_293_, v___x_303_);
lean_dec(v_x_293_);
v___x_305_ = 0;
v___x_306_ = l_Lean_SourceInfo_fromRef(v_ref_302_, v___x_305_);
v___x_307_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__4));
v___x_308_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__1, &lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__1);
v___x_309_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__4));
lean_inc(v_currMacroScope_301_);
lean_inc(v_quotContext_300_);
v___x_310_ = l_Lean_addMacroScope(v_quotContext_300_, v___x_309_, v_currMacroScope_301_);
v___x_311_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___closed__6));
lean_inc_n(v___x_306_, 2);
v___x_312_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_312_, 0, v___x_306_);
lean_ctor_set(v___x_312_, 1, v___x_308_);
lean_ctor_set(v___x_312_, 2, v___x_310_);
lean_ctor_set(v___x_312_, 3, v___x_311_);
v___x_313_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__13));
v___x_314_ = l_Lean_Syntax_node1(v___x_306_, v___x_313_, v___x_304_);
v___x_315_ = l_Lean_Syntax_node2(v___x_306_, v___x_307_, v___x_312_, v___x_314_);
v___x_316_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_316_, 0, v___x_315_);
lean_ctor_set(v___x_316_, 1, v_a_295_);
return v___x_316_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1___boxed(lean_object* v_x_317_, lean_object* v_a_318_, lean_object* v_a_319_){
_start:
{
lean_object* v_res_320_; 
v_res_320_ = lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207b__1(v_x_317_, v_a_318_, v_a_319_);
lean_dec_ref(v_a_318_);
return v_res_320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__NegPart__negPart__1(lean_object* v_x_321_, lean_object* v_a_322_, lean_object* v_a_323_){
_start:
{
lean_object* v___x_324_; uint8_t v___x_325_; 
v___x_324_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation______macroRules__term___u207a_u1d50__1___closed__4));
lean_inc(v_x_321_);
v___x_325_ = l_Lean_Syntax_isOfKind(v_x_321_, v___x_324_);
if (v___x_325_ == 0)
{
lean_object* v___x_326_; lean_object* v___x_327_; 
lean_dec(v_x_321_);
v___x_326_ = lean_box(0);
v___x_327_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_327_, 0, v___x_326_);
lean_ctor_set(v___x_327_, 1, v_a_323_);
return v___x_327_;
}
else
{
lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; uint8_t v___x_331_; 
v___x_328_ = lean_unsigned_to_nat(0u);
v___x_329_ = l_Lean_Syntax_getArg(v_x_321_, v___x_328_);
v___x_330_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__OneLePart__oneLePart__1___closed__1));
lean_inc(v___x_329_);
v___x_331_ = l_Lean_Syntax_isOfKind(v___x_329_, v___x_330_);
if (v___x_331_ == 0)
{
lean_object* v___x_332_; lean_object* v___x_333_; 
lean_dec(v___x_329_);
lean_dec(v_x_321_);
v___x_332_ = lean_box(0);
v___x_333_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_333_, 0, v___x_332_);
lean_ctor_set(v___x_333_, 1, v_a_323_);
return v___x_333_;
}
else
{
lean_object* v___x_334_; lean_object* v___x_335_; uint8_t v___x_336_; 
v___x_334_ = lean_unsigned_to_nat(1u);
v___x_335_ = l_Lean_Syntax_getArg(v_x_321_, v___x_334_);
lean_dec(v_x_321_);
lean_inc(v___x_335_);
v___x_336_ = l_Lean_Syntax_matchesNull(v___x_335_, v___x_334_);
if (v___x_336_ == 0)
{
lean_object* v___x_337_; lean_object* v___x_338_; 
lean_dec(v___x_335_);
lean_dec(v___x_329_);
v___x_337_ = lean_box(0);
v___x_338_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_338_, 0, v___x_337_);
lean_ctor_set(v___x_338_, 1, v_a_323_);
return v___x_338_;
}
else
{
lean_object* v___x_339_; lean_object* v_ref_340_; uint8_t v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; 
v___x_339_ = l_Lean_Syntax_getArg(v___x_335_, v___x_328_);
lean_dec(v___x_335_);
v_ref_340_ = l_Lean_replaceRef(v___x_329_, v_a_322_);
lean_dec(v___x_329_);
v___x_341_ = 0;
v___x_342_ = l_Lean_SourceInfo_fromRef(v_ref_340_, v___x_341_);
lean_dec(v_ref_340_);
v___x_343_ = ((lean_object*)(lp_mathlib_term___u207b___closed__1));
v___x_344_ = ((lean_object*)(lp_mathlib_term___u207b___closed__2));
lean_inc(v___x_342_);
v___x_345_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_345_, 0, v___x_342_);
lean_ctor_set(v___x_345_, 1, v___x_344_);
v___x_346_ = l_Lean_Syntax_node2(v___x_342_, v___x_343_, v___x_339_, v___x_345_);
v___x_347_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_347_, 0, v___x_346_);
lean_ctor_set(v___x_347_, 1, v_a_323_);
return v___x_347_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__NegPart__negPart__1___boxed(lean_object* v_x_348_, lean_object* v_a_349_, lean_object* v_a_350_){
_start:
{
lean_object* v_res_351_; 
v_res_351_ = lp_mathlib___aux__Mathlib__Algebra__Notation______unexpand__NegPart__negPart__1(v_x_348_, v_a_349_, v_a_350_);
lean_dec(v_a_349_);
return v_res_351_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Translate_ToAdditive(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Notation(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Translate_ToAdditive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Notation(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Tactic_Translate_ToAdditive(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Notation(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Translate_ToAdditive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Notation(builtin);
}
#ifdef __cplusplus
}
#endif
