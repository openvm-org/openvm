// Lean compiler output
// Module: Mathlib.Algebra.Notation.Defs
// Imports: public import Init public meta import Init public import Mathlib.Tactic.Simps public import Mathlib.Tactic.ToAdditive
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instHVAdd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instHVAdd___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instHVAdd(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instHVAdd___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___x2b_u1d65___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_+ᵥ_"};
static const lean_object* lp_mathlib_term___x2b_u1d65___00__closed__0 = (const lean_object*)&lp_mathlib_term___x2b_u1d65___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___x2b_u1d65___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___x2b_u1d65___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(185, 242, 40, 27, 208, 97, 34, 167)}};
static const lean_object* lp_mathlib_term___x2b_u1d65___00__closed__1 = (const lean_object*)&lp_mathlib_term___x2b_u1d65___00__closed__1_value;
static const lean_string_object lp_mathlib_term___x2b_u1d65___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term___x2b_u1d65___00__closed__2 = (const lean_object*)&lp_mathlib_term___x2b_u1d65___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___x2b_u1d65___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___x2b_u1d65___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term___x2b_u1d65___00__closed__3 = (const lean_object*)&lp_mathlib_term___x2b_u1d65___00__closed__3_value;
static const lean_string_object lp_mathlib_term___x2b_u1d65___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " +ᵥ "};
static const lean_object* lp_mathlib_term___x2b_u1d65___00__closed__4 = (const lean_object*)&lp_mathlib_term___x2b_u1d65___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___x2b_u1d65___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___x2b_u1d65___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___x2b_u1d65___00__closed__5 = (const lean_object*)&lp_mathlib_term___x2b_u1d65___00__closed__5_value;
static const lean_string_object lp_mathlib_term___x2b_u1d65___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term___x2b_u1d65___00__closed__6 = (const lean_object*)&lp_mathlib_term___x2b_u1d65___00__closed__6_value;
static const lean_ctor_object lp_mathlib_term___x2b_u1d65___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___x2b_u1d65___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term___x2b_u1d65___00__closed__7 = (const lean_object*)&lp_mathlib_term___x2b_u1d65___00__closed__7_value;
static const lean_ctor_object lp_mathlib_term___x2b_u1d65___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___x2b_u1d65___00__closed__7_value),((lean_object*)(((size_t)(65) << 1) | 1))}};
static const lean_object* lp_mathlib_term___x2b_u1d65___00__closed__8 = (const lean_object*)&lp_mathlib_term___x2b_u1d65___00__closed__8_value;
static const lean_ctor_object lp_mathlib_term___x2b_u1d65___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___x2b_u1d65___00__closed__3_value),((lean_object*)&lp_mathlib_term___x2b_u1d65___00__closed__5_value),((lean_object*)&lp_mathlib_term___x2b_u1d65___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___x2b_u1d65___00__closed__9 = (const lean_object*)&lp_mathlib_term___x2b_u1d65___00__closed__9_value;
static const lean_ctor_object lp_mathlib_term___x2b_u1d65___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___x2b_u1d65___00__closed__1_value),((lean_object*)(((size_t)(65) << 1) | 1)),((lean_object*)(((size_t)(66) << 1) | 1)),((lean_object*)&lp_mathlib_term___x2b_u1d65___00__closed__9_value)}};
static const lean_object* lp_mathlib_term___x2b_u1d65___00__closed__10 = (const lean_object*)&lp_mathlib_term___x2b_u1d65___00__closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___x2b_u1d65__ = (const lean_object*)&lp_mathlib_term___x2b_u1d65___00__closed__10_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "HVAdd.hVAdd"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__6;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "HVAdd"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__7_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "hVAdd"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(239, 135, 107, 242, 117, 15, 176, 86)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__9_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(90, 24, 198, 227, 204, 199, 190, 118)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__9_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__11_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__12_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______unexpand__HVAdd__hVAdd__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______unexpand__HVAdd__hVAdd__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______unexpand__HVAdd__hVAdd__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______unexpand__HVAdd__hVAdd__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______unexpand__HVAdd__hVAdd__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______unexpand__HVAdd__hVAdd__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______unexpand__HVAdd__hVAdd__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______unexpand__HVAdd__hVAdd__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______unexpand__HVAdd__hVAdd__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___x2d_u1d65___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_-ᵥ_"};
static const lean_object* lp_mathlib_term___x2d_u1d65___00__closed__0 = (const lean_object*)&lp_mathlib_term___x2d_u1d65___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___x2d_u1d65___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___x2d_u1d65___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(243, 128, 142, 252, 89, 101, 92, 11)}};
static const lean_object* lp_mathlib_term___x2d_u1d65___00__closed__1 = (const lean_object*)&lp_mathlib_term___x2d_u1d65___00__closed__1_value;
static const lean_string_object lp_mathlib_term___x2d_u1d65___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " -ᵥ "};
static const lean_object* lp_mathlib_term___x2d_u1d65___00__closed__2 = (const lean_object*)&lp_mathlib_term___x2d_u1d65___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___x2d_u1d65___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___x2d_u1d65___00__closed__2_value)}};
static const lean_object* lp_mathlib_term___x2d_u1d65___00__closed__3 = (const lean_object*)&lp_mathlib_term___x2d_u1d65___00__closed__3_value;
static const lean_ctor_object lp_mathlib_term___x2d_u1d65___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___x2b_u1d65___00__closed__7_value),((lean_object*)(((size_t)(66) << 1) | 1))}};
static const lean_object* lp_mathlib_term___x2d_u1d65___00__closed__4 = (const lean_object*)&lp_mathlib_term___x2d_u1d65___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___x2d_u1d65___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___x2b_u1d65___00__closed__3_value),((lean_object*)&lp_mathlib_term___x2d_u1d65___00__closed__3_value),((lean_object*)&lp_mathlib_term___x2d_u1d65___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___x2d_u1d65___00__closed__5 = (const lean_object*)&lp_mathlib_term___x2d_u1d65___00__closed__5_value;
static const lean_ctor_object lp_mathlib_term___x2d_u1d65___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___x2d_u1d65___00__closed__1_value),((lean_object*)(((size_t)(65) << 1) | 1)),((lean_object*)(((size_t)(65) << 1) | 1)),((lean_object*)&lp_mathlib_term___x2d_u1d65___00__closed__5_value)}};
static const lean_object* lp_mathlib_term___x2d_u1d65___00__closed__6 = (const lean_object*)&lp_mathlib_term___x2d_u1d65___00__closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___x2d_u1d65__ = (const lean_object*)&lp_mathlib_term___x2d_u1d65___00__closed__6_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "VSub.vsub"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__1;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "VSub"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "vsub"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(153, 196, 74, 223, 242, 216, 40, 206)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(145, 86, 75, 123, 100, 102, 203, 116)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______unexpand__VSub__vsub__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______unexpand__VSub__vsub__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___x2f_u209b___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_/ₛ_"};
static const lean_object* lp_mathlib_term___x2f_u209b___00__closed__0 = (const lean_object*)&lp_mathlib_term___x2f_u209b___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___x2f_u209b___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___x2f_u209b___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(60, 126, 82, 68, 248, 57, 140, 189)}};
static const lean_object* lp_mathlib_term___x2f_u209b___00__closed__1 = (const lean_object*)&lp_mathlib_term___x2f_u209b___00__closed__1_value;
static const lean_string_object lp_mathlib_term___x2f_u209b___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " /ₛ "};
static const lean_object* lp_mathlib_term___x2f_u209b___00__closed__2 = (const lean_object*)&lp_mathlib_term___x2f_u209b___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___x2f_u209b___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___x2f_u209b___00__closed__2_value)}};
static const lean_object* lp_mathlib_term___x2f_u209b___00__closed__3 = (const lean_object*)&lp_mathlib_term___x2f_u209b___00__closed__3_value;
static const lean_ctor_object lp_mathlib_term___x2f_u209b___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___x2b_u1d65___00__closed__3_value),((lean_object*)&lp_mathlib_term___x2f_u209b___00__closed__3_value),((lean_object*)&lp_mathlib_term___x2d_u1d65___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___x2f_u209b___00__closed__4 = (const lean_object*)&lp_mathlib_term___x2f_u209b___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___x2f_u209b___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___x2f_u209b___00__closed__1_value),((lean_object*)(((size_t)(65) << 1) | 1)),((lean_object*)(((size_t)(65) << 1) | 1)),((lean_object*)&lp_mathlib_term___x2f_u209b___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___x2f_u209b___00__closed__5 = (const lean_object*)&lp_mathlib_term___x2f_u209b___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___x2f_u209b__ = (const lean_object*)&lp_mathlib_term___x2f_u209b___00__closed__5_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "SDiv.sdiv"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__1;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "SDiv"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "sdiv"};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(205, 198, 206, 111, 184, 198, 122, 249)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(107, 221, 142, 82, 10, 240, 188, 95)}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______unexpand__SDiv__sdiv__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______unexpand__SDiv__sdiv__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instHVAdd___redArg(lean_object* v_inst_1_){
_start:
{
lean_inc(v_inst_1_);
return v_inst_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instHVAdd___redArg___boxed(lean_object* v_inst_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_instHVAdd___redArg(v_inst_2_);
lean_dec(v_inst_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instHVAdd(lean_object* v_00_u03b1_4_, lean_object* v_00_u03b2_5_, lean_object* v_inst_6_){
_start:
{
lean_inc(v_inst_6_);
return v_inst_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instHVAdd___boxed(lean_object* v_00_u03b1_7_, lean_object* v_00_u03b2_8_, lean_object* v_inst_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_mathlib_instHVAdd(v_00_u03b1_7_, v_00_u03b2_8_, v_inst_9_);
lean_dec(v_inst_9_);
return v_res_10_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__6(void){
_start:
{
lean_object* v___x_46_; lean_object* v___x_47_; 
v___x_46_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__5));
v___x_47_ = l_String_toRawSubstring_x27(v___x_46_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1(lean_object* v_x_62_, lean_object* v_a_63_, lean_object* v_a_64_){
_start:
{
lean_object* v___x_65_; uint8_t v___x_66_; 
v___x_65_ = ((lean_object*)(lp_mathlib_term___x2b_u1d65___00__closed__1));
lean_inc(v_x_62_);
v___x_66_ = l_Lean_Syntax_isOfKind(v_x_62_, v___x_65_);
if (v___x_66_ == 0)
{
lean_object* v___x_67_; lean_object* v___x_68_; 
lean_dec(v_x_62_);
v___x_67_ = lean_box(1);
v___x_68_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_68_, 0, v___x_67_);
lean_ctor_set(v___x_68_, 1, v_a_64_);
return v___x_68_;
}
else
{
lean_object* v_quotContext_69_; lean_object* v_currMacroScope_70_; lean_object* v_ref_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; uint8_t v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; 
v_quotContext_69_ = lean_ctor_get(v_a_63_, 1);
v_currMacroScope_70_ = lean_ctor_get(v_a_63_, 2);
v_ref_71_ = lean_ctor_get(v_a_63_, 5);
v___x_72_ = lean_unsigned_to_nat(0u);
v___x_73_ = l_Lean_Syntax_getArg(v_x_62_, v___x_72_);
v___x_74_ = lean_unsigned_to_nat(2u);
v___x_75_ = l_Lean_Syntax_getArg(v_x_62_, v___x_74_);
lean_dec(v_x_62_);
v___x_76_ = 0;
v___x_77_ = l_Lean_SourceInfo_fromRef(v_ref_71_, v___x_76_);
v___x_78_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__4));
v___x_79_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__6, &lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__6);
v___x_80_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__9));
lean_inc(v_currMacroScope_70_);
lean_inc(v_quotContext_69_);
v___x_81_ = l_Lean_addMacroScope(v_quotContext_69_, v___x_80_, v_currMacroScope_70_);
v___x_82_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__11));
lean_inc_n(v___x_77_, 2);
v___x_83_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_83_, 0, v___x_77_);
lean_ctor_set(v___x_83_, 1, v___x_79_);
lean_ctor_set(v___x_83_, 2, v___x_81_);
lean_ctor_set(v___x_83_, 3, v___x_82_);
v___x_84_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__13));
v___x_85_ = l_Lean_Syntax_node2(v___x_77_, v___x_84_, v___x_73_, v___x_75_);
v___x_86_ = l_Lean_Syntax_node2(v___x_77_, v___x_78_, v___x_83_, v___x_85_);
v___x_87_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_87_, 0, v___x_86_);
lean_ctor_set(v___x_87_, 1, v_a_64_);
return v___x_87_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___boxed(lean_object* v_x_88_, lean_object* v_a_89_, lean_object* v_a_90_){
_start:
{
lean_object* v_res_91_; 
v_res_91_ = lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1(v_x_88_, v_a_89_, v_a_90_);
lean_dec_ref(v_a_89_);
return v_res_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______unexpand__HVAdd__hVAdd__1(lean_object* v_x_95_, lean_object* v_a_96_, lean_object* v_a_97_){
_start:
{
lean_object* v___x_98_; uint8_t v___x_99_; 
v___x_98_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__4));
lean_inc(v_x_95_);
v___x_99_ = l_Lean_Syntax_isOfKind(v_x_95_, v___x_98_);
if (v___x_99_ == 0)
{
lean_object* v___x_100_; lean_object* v___x_101_; 
lean_dec(v_x_95_);
v___x_100_ = lean_box(0);
v___x_101_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_101_, 0, v___x_100_);
lean_ctor_set(v___x_101_, 1, v_a_97_);
return v___x_101_;
}
else
{
lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; uint8_t v___x_105_; 
v___x_102_ = lean_unsigned_to_nat(0u);
v___x_103_ = l_Lean_Syntax_getArg(v_x_95_, v___x_102_);
v___x_104_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______unexpand__HVAdd__hVAdd__1___closed__1));
lean_inc(v___x_103_);
v___x_105_ = l_Lean_Syntax_isOfKind(v___x_103_, v___x_104_);
if (v___x_105_ == 0)
{
lean_object* v___x_106_; lean_object* v___x_107_; 
lean_dec(v___x_103_);
lean_dec(v_x_95_);
v___x_106_ = lean_box(0);
v___x_107_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_107_, 0, v___x_106_);
lean_ctor_set(v___x_107_, 1, v_a_97_);
return v___x_107_;
}
else
{
lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; uint8_t v___x_111_; 
v___x_108_ = lean_unsigned_to_nat(1u);
v___x_109_ = l_Lean_Syntax_getArg(v_x_95_, v___x_108_);
lean_dec(v_x_95_);
v___x_110_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_109_);
v___x_111_ = l_Lean_Syntax_matchesNull(v___x_109_, v___x_110_);
if (v___x_111_ == 0)
{
lean_object* v___x_112_; lean_object* v___x_113_; 
lean_dec(v___x_109_);
lean_dec(v___x_103_);
v___x_112_ = lean_box(0);
v___x_113_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_113_, 0, v___x_112_);
lean_ctor_set(v___x_113_, 1, v_a_97_);
return v___x_113_;
}
else
{
lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v_ref_116_; uint8_t v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; 
v___x_114_ = l_Lean_Syntax_getArg(v___x_109_, v___x_102_);
v___x_115_ = l_Lean_Syntax_getArg(v___x_109_, v___x_108_);
lean_dec(v___x_109_);
v_ref_116_ = l_Lean_replaceRef(v___x_103_, v_a_96_);
lean_dec(v___x_103_);
v___x_117_ = 0;
v___x_118_ = l_Lean_SourceInfo_fromRef(v_ref_116_, v___x_117_);
lean_dec(v_ref_116_);
v___x_119_ = ((lean_object*)(lp_mathlib_term___x2b_u1d65___00__closed__1));
v___x_120_ = ((lean_object*)(lp_mathlib_term___x2b_u1d65___00__closed__4));
lean_inc(v___x_118_);
v___x_121_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_121_, 0, v___x_118_);
lean_ctor_set(v___x_121_, 1, v___x_120_);
v___x_122_ = l_Lean_Syntax_node3(v___x_118_, v___x_119_, v___x_114_, v___x_121_, v___x_115_);
v___x_123_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_123_, 0, v___x_122_);
lean_ctor_set(v___x_123_, 1, v_a_97_);
return v___x_123_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______unexpand__HVAdd__hVAdd__1___boxed(lean_object* v_x_124_, lean_object* v_a_125_, lean_object* v_a_126_){
_start:
{
lean_object* v_res_127_; 
v_res_127_ = lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______unexpand__HVAdd__hVAdd__1(v_x_124_, v_a_125_, v_a_126_);
lean_dec(v_a_125_);
return v_res_127_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__1(void){
_start:
{
lean_object* v___x_147_; lean_object* v___x_148_; 
v___x_147_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__0));
v___x_148_ = l_String_toRawSubstring_x27(v___x_147_);
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1(lean_object* v_x_160_, lean_object* v_a_161_, lean_object* v_a_162_){
_start:
{
lean_object* v___x_163_; uint8_t v___x_164_; 
v___x_163_ = ((lean_object*)(lp_mathlib_term___x2d_u1d65___00__closed__1));
lean_inc(v_x_160_);
v___x_164_ = l_Lean_Syntax_isOfKind(v_x_160_, v___x_163_);
if (v___x_164_ == 0)
{
lean_object* v___x_165_; lean_object* v___x_166_; 
lean_dec(v_x_160_);
v___x_165_ = lean_box(1);
v___x_166_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_166_, 0, v___x_165_);
lean_ctor_set(v___x_166_, 1, v_a_162_);
return v___x_166_;
}
else
{
lean_object* v_quotContext_167_; lean_object* v_currMacroScope_168_; lean_object* v_ref_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; uint8_t v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; 
v_quotContext_167_ = lean_ctor_get(v_a_161_, 1);
v_currMacroScope_168_ = lean_ctor_get(v_a_161_, 2);
v_ref_169_ = lean_ctor_get(v_a_161_, 5);
v___x_170_ = lean_unsigned_to_nat(0u);
v___x_171_ = l_Lean_Syntax_getArg(v_x_160_, v___x_170_);
v___x_172_ = lean_unsigned_to_nat(2u);
v___x_173_ = l_Lean_Syntax_getArg(v_x_160_, v___x_172_);
lean_dec(v_x_160_);
v___x_174_ = 0;
v___x_175_ = l_Lean_SourceInfo_fromRef(v_ref_169_, v___x_174_);
v___x_176_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__4));
v___x_177_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__1, &lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__1);
v___x_178_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__4));
lean_inc(v_currMacroScope_168_);
lean_inc(v_quotContext_167_);
v___x_179_ = l_Lean_addMacroScope(v_quotContext_167_, v___x_178_, v_currMacroScope_168_);
v___x_180_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___closed__6));
lean_inc_n(v___x_175_, 2);
v___x_181_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_181_, 0, v___x_175_);
lean_ctor_set(v___x_181_, 1, v___x_177_);
lean_ctor_set(v___x_181_, 2, v___x_179_);
lean_ctor_set(v___x_181_, 3, v___x_180_);
v___x_182_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__13));
v___x_183_ = l_Lean_Syntax_node2(v___x_175_, v___x_182_, v___x_171_, v___x_173_);
v___x_184_ = l_Lean_Syntax_node2(v___x_175_, v___x_176_, v___x_181_, v___x_183_);
v___x_185_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_185_, 0, v___x_184_);
lean_ctor_set(v___x_185_, 1, v_a_162_);
return v___x_185_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1___boxed(lean_object* v_x_186_, lean_object* v_a_187_, lean_object* v_a_188_){
_start:
{
lean_object* v_res_189_; 
v_res_189_ = lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2d_u1d65____1(v_x_186_, v_a_187_, v_a_188_);
lean_dec_ref(v_a_187_);
return v_res_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______unexpand__VSub__vsub__1(lean_object* v_x_190_, lean_object* v_a_191_, lean_object* v_a_192_){
_start:
{
lean_object* v___x_193_; uint8_t v___x_194_; 
v___x_193_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__4));
lean_inc(v_x_190_);
v___x_194_ = l_Lean_Syntax_isOfKind(v_x_190_, v___x_193_);
if (v___x_194_ == 0)
{
lean_object* v___x_195_; lean_object* v___x_196_; 
lean_dec(v_x_190_);
v___x_195_ = lean_box(0);
v___x_196_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_196_, 0, v___x_195_);
lean_ctor_set(v___x_196_, 1, v_a_192_);
return v___x_196_;
}
else
{
lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; uint8_t v___x_200_; 
v___x_197_ = lean_unsigned_to_nat(0u);
v___x_198_ = l_Lean_Syntax_getArg(v_x_190_, v___x_197_);
v___x_199_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______unexpand__HVAdd__hVAdd__1___closed__1));
lean_inc(v___x_198_);
v___x_200_ = l_Lean_Syntax_isOfKind(v___x_198_, v___x_199_);
if (v___x_200_ == 0)
{
lean_object* v___x_201_; lean_object* v___x_202_; 
lean_dec(v___x_198_);
lean_dec(v_x_190_);
v___x_201_ = lean_box(0);
v___x_202_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_202_, 0, v___x_201_);
lean_ctor_set(v___x_202_, 1, v_a_192_);
return v___x_202_;
}
else
{
lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; uint8_t v___x_206_; 
v___x_203_ = lean_unsigned_to_nat(1u);
v___x_204_ = l_Lean_Syntax_getArg(v_x_190_, v___x_203_);
lean_dec(v_x_190_);
v___x_205_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_204_);
v___x_206_ = l_Lean_Syntax_matchesNull(v___x_204_, v___x_205_);
if (v___x_206_ == 0)
{
lean_object* v___x_207_; lean_object* v___x_208_; 
lean_dec(v___x_204_);
lean_dec(v___x_198_);
v___x_207_ = lean_box(0);
v___x_208_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_208_, 0, v___x_207_);
lean_ctor_set(v___x_208_, 1, v_a_192_);
return v___x_208_;
}
else
{
lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v_ref_211_; uint8_t v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; 
v___x_209_ = l_Lean_Syntax_getArg(v___x_204_, v___x_197_);
v___x_210_ = l_Lean_Syntax_getArg(v___x_204_, v___x_203_);
lean_dec(v___x_204_);
v_ref_211_ = l_Lean_replaceRef(v___x_198_, v_a_191_);
lean_dec(v___x_198_);
v___x_212_ = 0;
v___x_213_ = l_Lean_SourceInfo_fromRef(v_ref_211_, v___x_212_);
lean_dec(v_ref_211_);
v___x_214_ = ((lean_object*)(lp_mathlib_term___x2d_u1d65___00__closed__1));
v___x_215_ = ((lean_object*)(lp_mathlib_term___x2d_u1d65___00__closed__2));
lean_inc(v___x_213_);
v___x_216_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_216_, 0, v___x_213_);
lean_ctor_set(v___x_216_, 1, v___x_215_);
v___x_217_ = l_Lean_Syntax_node3(v___x_213_, v___x_214_, v___x_209_, v___x_216_, v___x_210_);
v___x_218_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_218_, 0, v___x_217_);
lean_ctor_set(v___x_218_, 1, v_a_192_);
return v___x_218_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______unexpand__VSub__vsub__1___boxed(lean_object* v_x_219_, lean_object* v_a_220_, lean_object* v_a_221_){
_start:
{
lean_object* v_res_222_; 
v_res_222_ = lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______unexpand__VSub__vsub__1(v_x_219_, v_a_220_, v_a_221_);
lean_dec(v_a_220_);
return v_res_222_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__1(void){
_start:
{
lean_object* v___x_239_; lean_object* v___x_240_; 
v___x_239_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__0));
v___x_240_ = l_String_toRawSubstring_x27(v___x_239_);
return v___x_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1(lean_object* v_x_252_, lean_object* v_a_253_, lean_object* v_a_254_){
_start:
{
lean_object* v___x_255_; uint8_t v___x_256_; 
v___x_255_ = ((lean_object*)(lp_mathlib_term___x2f_u209b___00__closed__1));
lean_inc(v_x_252_);
v___x_256_ = l_Lean_Syntax_isOfKind(v_x_252_, v___x_255_);
if (v___x_256_ == 0)
{
lean_object* v___x_257_; lean_object* v___x_258_; 
lean_dec(v_x_252_);
v___x_257_ = lean_box(1);
v___x_258_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_258_, 0, v___x_257_);
lean_ctor_set(v___x_258_, 1, v_a_254_);
return v___x_258_;
}
else
{
lean_object* v_quotContext_259_; lean_object* v_currMacroScope_260_; lean_object* v_ref_261_; lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; uint8_t v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; 
v_quotContext_259_ = lean_ctor_get(v_a_253_, 1);
v_currMacroScope_260_ = lean_ctor_get(v_a_253_, 2);
v_ref_261_ = lean_ctor_get(v_a_253_, 5);
v___x_262_ = lean_unsigned_to_nat(0u);
v___x_263_ = l_Lean_Syntax_getArg(v_x_252_, v___x_262_);
v___x_264_ = lean_unsigned_to_nat(2u);
v___x_265_ = l_Lean_Syntax_getArg(v_x_252_, v___x_264_);
lean_dec(v_x_252_);
v___x_266_ = 0;
v___x_267_ = l_Lean_SourceInfo_fromRef(v_ref_261_, v___x_266_);
v___x_268_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__4));
v___x_269_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__1, &lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__1);
v___x_270_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__4));
lean_inc(v_currMacroScope_260_);
lean_inc(v_quotContext_259_);
v___x_271_ = l_Lean_addMacroScope(v_quotContext_259_, v___x_270_, v_currMacroScope_260_);
v___x_272_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___closed__6));
lean_inc_n(v___x_267_, 2);
v___x_273_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_273_, 0, v___x_267_);
lean_ctor_set(v___x_273_, 1, v___x_269_);
lean_ctor_set(v___x_273_, 2, v___x_271_);
lean_ctor_set(v___x_273_, 3, v___x_272_);
v___x_274_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__13));
v___x_275_ = l_Lean_Syntax_node2(v___x_267_, v___x_274_, v___x_263_, v___x_265_);
v___x_276_ = l_Lean_Syntax_node2(v___x_267_, v___x_268_, v___x_273_, v___x_275_);
v___x_277_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_277_, 0, v___x_276_);
lean_ctor_set(v___x_277_, 1, v_a_254_);
return v___x_277_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1___boxed(lean_object* v_x_278_, lean_object* v_a_279_, lean_object* v_a_280_){
_start:
{
lean_object* v_res_281_; 
v_res_281_ = lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2f_u209b____1(v_x_278_, v_a_279_, v_a_280_);
lean_dec_ref(v_a_279_);
return v_res_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______unexpand__SDiv__sdiv__1(lean_object* v_x_282_, lean_object* v_a_283_, lean_object* v_a_284_){
_start:
{
lean_object* v___x_285_; uint8_t v___x_286_; 
v___x_285_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______macroRules__term___x2b_u1d65____1___closed__4));
lean_inc(v_x_282_);
v___x_286_ = l_Lean_Syntax_isOfKind(v_x_282_, v___x_285_);
if (v___x_286_ == 0)
{
lean_object* v___x_287_; lean_object* v___x_288_; 
lean_dec(v_x_282_);
v___x_287_ = lean_box(0);
v___x_288_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_288_, 0, v___x_287_);
lean_ctor_set(v___x_288_, 1, v_a_284_);
return v___x_288_;
}
else
{
lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; uint8_t v___x_292_; 
v___x_289_ = lean_unsigned_to_nat(0u);
v___x_290_ = l_Lean_Syntax_getArg(v_x_282_, v___x_289_);
v___x_291_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______unexpand__HVAdd__hVAdd__1___closed__1));
lean_inc(v___x_290_);
v___x_292_ = l_Lean_Syntax_isOfKind(v___x_290_, v___x_291_);
if (v___x_292_ == 0)
{
lean_object* v___x_293_; lean_object* v___x_294_; 
lean_dec(v___x_290_);
lean_dec(v_x_282_);
v___x_293_ = lean_box(0);
v___x_294_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_294_, 0, v___x_293_);
lean_ctor_set(v___x_294_, 1, v_a_284_);
return v___x_294_;
}
else
{
lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; uint8_t v___x_298_; 
v___x_295_ = lean_unsigned_to_nat(1u);
v___x_296_ = l_Lean_Syntax_getArg(v_x_282_, v___x_295_);
lean_dec(v_x_282_);
v___x_297_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_296_);
v___x_298_ = l_Lean_Syntax_matchesNull(v___x_296_, v___x_297_);
if (v___x_298_ == 0)
{
lean_object* v___x_299_; lean_object* v___x_300_; 
lean_dec(v___x_296_);
lean_dec(v___x_290_);
v___x_299_ = lean_box(0);
v___x_300_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_300_, 0, v___x_299_);
lean_ctor_set(v___x_300_, 1, v_a_284_);
return v___x_300_;
}
else
{
lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v_ref_303_; uint8_t v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; 
v___x_301_ = l_Lean_Syntax_getArg(v___x_296_, v___x_289_);
v___x_302_ = l_Lean_Syntax_getArg(v___x_296_, v___x_295_);
lean_dec(v___x_296_);
v_ref_303_ = l_Lean_replaceRef(v___x_290_, v_a_283_);
lean_dec(v___x_290_);
v___x_304_ = 0;
v___x_305_ = l_Lean_SourceInfo_fromRef(v_ref_303_, v___x_304_);
lean_dec(v_ref_303_);
v___x_306_ = ((lean_object*)(lp_mathlib_term___x2f_u209b___00__closed__1));
v___x_307_ = ((lean_object*)(lp_mathlib_term___x2f_u209b___00__closed__2));
lean_inc(v___x_305_);
v___x_308_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_308_, 0, v___x_305_);
lean_ctor_set(v___x_308_, 1, v___x_307_);
v___x_309_ = l_Lean_Syntax_node3(v___x_305_, v___x_306_, v___x_301_, v___x_308_, v___x_302_);
v___x_310_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_310_, 0, v___x_309_);
lean_ctor_set(v___x_310_, 1, v_a_284_);
return v___x_310_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______unexpand__SDiv__sdiv__1___boxed(lean_object* v_x_311_, lean_object* v_a_312_, lean_object* v_a_313_){
_start:
{
lean_object* v_res_314_; 
v_res_314_ = lp_mathlib___aux__Mathlib__Algebra__Notation__Defs______unexpand__SDiv__sdiv__1(v_x_311_, v_a_312_, v_a_313_);
lean_dec(v_a_312_);
return v_res_314_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Simps(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ToAdditive(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Notation_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Simps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ToAdditive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Notation_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Tactic_Simps(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToAdditive(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Notation_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Simps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToAdditive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Notation_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Notation_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Notation_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
