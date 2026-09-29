// Lean compiler output
// Module: Mathlib.Logic.Function.Defs
// Imports: public import Init public meta import Init public import Mathlib.Init import Mathlib.Tactic.Attr.Register
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
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_dcomp___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_dcomp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Function_term___u2218_x27___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Function"};
static const lean_object* lp_mathlib_Function_term___u2218_x27___00__closed__0 = (const lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__0_value;
static const lean_string_object lp_mathlib_Function_term___u2218_x27___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term_∘'_"};
static const lean_object* lp_mathlib_Function_term___u2218_x27___00__closed__1 = (const lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__1_value;
static const lean_ctor_object lp_mathlib_Function_term___u2218_x27___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(225, 8, 186, 189, 152, 89, 197, 12)}};
static const lean_ctor_object lp_mathlib_Function_term___u2218_x27___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__2_value_aux_0),((lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(222, 216, 171, 138, 148, 141, 52, 168)}};
static const lean_object* lp_mathlib_Function_term___u2218_x27___00__closed__2 = (const lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__2_value;
static const lean_string_object lp_mathlib_Function_term___u2218_x27___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Function_term___u2218_x27___00__closed__3 = (const lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Function_term___u2218_x27___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Function_term___u2218_x27___00__closed__4 = (const lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__4_value;
static const lean_string_object lp_mathlib_Function_term___u2218_x27___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 4, .m_data = " ∘' "};
static const lean_object* lp_mathlib_Function_term___u2218_x27___00__closed__5 = (const lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Function_term___u2218_x27___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__5_value)}};
static const lean_object* lp_mathlib_Function_term___u2218_x27___00__closed__6 = (const lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__6_value;
static const lean_string_object lp_mathlib_Function_term___u2218_x27___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Function_term___u2218_x27___00__closed__7 = (const lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__7_value;
static const lean_ctor_object lp_mathlib_Function_term___u2218_x27___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Function_term___u2218_x27___00__closed__8 = (const lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__8_value;
static const lean_ctor_object lp_mathlib_Function_term___u2218_x27___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__8_value),((lean_object*)(((size_t)(80) << 1) | 1))}};
static const lean_object* lp_mathlib_Function_term___u2218_x27___00__closed__9 = (const lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__9_value;
static const lean_ctor_object lp_mathlib_Function_term___u2218_x27___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__4_value),((lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__6_value),((lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__9_value)}};
static const lean_object* lp_mathlib_Function_term___u2218_x27___00__closed__10 = (const lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__10_value;
static const lean_ctor_object lp_mathlib_Function_term___u2218_x27___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__2_value),((lean_object*)(((size_t)(80) << 1) | 1)),((lean_object*)(((size_t)(81) << 1) | 1)),((lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__10_value)}};
static const lean_object* lp_mathlib_Function_term___u2218_x27___00__closed__11 = (const lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__11_value;
LEAN_EXPORT const lean_object* lp_mathlib_Function_term___u2218_x27__ = (const lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__11_value;
static const lean_string_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__0 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__0_value;
static const lean_string_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__1 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__1_value;
static const lean_string_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__2 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__2_value;
static const lean_string_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__3 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__4 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__4_value;
static const lean_string_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "Function.dcomp"};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__5 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__5_value;
static lean_once_cell_t lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__6;
static const lean_string_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "dcomp"};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__7 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__7_value;
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(225, 8, 186, 189, 152, 89, 197, 12)}};
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(75, 231, 86, 174, 229, 20, 90, 50)}};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__8 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__8_value;
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__9 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__9_value;
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__10 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__10_value;
static const lean_string_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__11 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__11_value;
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__12 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______unexpand__Function__dcomp__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______unexpand__Function__dcomp__1___closed__0 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______unexpand__Function__dcomp__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______unexpand__Function__dcomp__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______unexpand__Function__dcomp__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______unexpand__Function__dcomp__1___closed__1 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______unexpand__Function__dcomp__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______unexpand__Function__dcomp__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______unexpand__Function__dcomp__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_prod___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_diag___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_diag(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_onFun___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_onFun(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Function_term__On___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "term_On_"};
static const lean_object* lp_mathlib_Function_term__On___00__closed__0 = (const lean_object*)&lp_mathlib_Function_term__On___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Function_term__On___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(225, 8, 186, 189, 152, 89, 197, 12)}};
static const lean_ctor_object lp_mathlib_Function_term__On___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Function_term__On___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Function_term__On___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(13, 215, 234, 216, 218, 14, 218, 22)}};
static const lean_object* lp_mathlib_Function_term__On___00__closed__1 = (const lean_object*)&lp_mathlib_Function_term__On___00__closed__1_value;
static const lean_string_object lp_mathlib_Function_term__On___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " on "};
static const lean_object* lp_mathlib_Function_term__On___00__closed__2 = (const lean_object*)&lp_mathlib_Function_term__On___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Function_term__On___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Function_term__On___00__closed__2_value)}};
static const lean_object* lp_mathlib_Function_term__On___00__closed__3 = (const lean_object*)&lp_mathlib_Function_term__On___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Function_term__On___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__8_value),((lean_object*)(((size_t)(3) << 1) | 1))}};
static const lean_object* lp_mathlib_Function_term__On___00__closed__4 = (const lean_object*)&lp_mathlib_Function_term__On___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Function_term__On___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__4_value),((lean_object*)&lp_mathlib_Function_term__On___00__closed__3_value),((lean_object*)&lp_mathlib_Function_term__On___00__closed__4_value)}};
static const lean_object* lp_mathlib_Function_term__On___00__closed__5 = (const lean_object*)&lp_mathlib_Function_term__On___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Function_term__On___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_Function_term__On___00__closed__1_value),((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_mathlib_Function_term__On___00__closed__5_value)}};
static const lean_object* lp_mathlib_Function_term__On___00__closed__6 = (const lean_object*)&lp_mathlib_Function_term__On___00__closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_Function_term__On__ = (const lean_object*)&lp_mathlib_Function_term__On___00__closed__6_value;
static const lean_string_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "onFun"};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___closed__0 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___closed__0_value;
static lean_once_cell_t lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___closed__1;
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(109, 62, 90, 108, 120, 135, 177, 12)}};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___closed__2 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___closed__2_value;
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(225, 8, 186, 189, 152, 89, 197, 12)}};
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(95, 72, 106, 114, 23, 67, 43, 30)}};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___closed__3 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___closed__4 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___closed__4_value;
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___closed__5 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______unexpand__Function__onFun__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______unexpand__Function__onFun__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_swap___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_swap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_bicompl___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_bicompl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_bicompr___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_bicompr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Logic"};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__4_value),LEAN_SCALAR_PTR_LITERAL(222, 164, 153, 31, 25, 197, 191, 150)}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__5_value),((lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(216, 6, 48, 81, 151, 76, 239, 161)}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Defs"};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__6_value),((lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(32, 4, 199, 46, 21, 117, 115, 192)}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(89, 23, 119, 108, 250, 56, 206, 140)}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__9_value),((lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(107, 249, 114, 148, 157, 221, 247, 69)}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 8, .m_data = "term_∘₂_"};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__11_value),LEAN_SCALAR_PTR_LITERAL(51, 65, 14, 47, 208, 116, 134, 30)}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 4, .m_data = " ∘₂ "};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__13_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__4_value),((lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__14_value),((lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__15_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__12_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__16_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__17_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082__ = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__17_value;
static const lean_string_object lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "bicompr"};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___closed__1;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(146, 56, 169, 255, 28, 79, 55, 157)}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Function_term___u2218_x27___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(225, 8, 186, 189, 152, 89, 197, 12)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(208, 202, 145, 141, 13, 222, 51, 212)}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______unexpand__Function__bicompr__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______unexpand__Function__bicompr__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Function_IsFixedPt_decidable___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_IsFixedPt_decidable___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Function_IsFixedPt_decidable(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_IsFixedPt_decidable___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_map___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_dcomp___redArg(lean_object* v_f_1_, lean_object* v_g_2_, lean_object* v_x_3_){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; 
lean_inc(v_x_3_);
v___x_4_ = lean_apply_1(v_g_2_, v_x_3_);
v___x_5_ = lean_apply_2(v_f_1_, v_x_3_, v___x_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_dcomp(lean_object* v_00_u03b1_6_, lean_object* v_00_u03b2_7_, lean_object* v_00_u03c6_8_, lean_object* v_f_9_, lean_object* v_g_10_, lean_object* v_x_11_){
_start:
{
lean_object* v___x_12_; lean_object* v___x_13_; 
lean_inc(v_x_11_);
v___x_12_ = lean_apply_1(v_g_10_, v_x_11_);
v___x_13_ = lean_apply_2(v_f_9_, v_x_11_, v___x_12_);
return v___x_13_;
}
}
static lean_object* _init_lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__6(void){
_start:
{
lean_object* v___x_51_; lean_object* v___x_52_; 
v___x_51_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__5));
v___x_52_ = l_String_toRawSubstring_x27(v___x_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1(lean_object* v_x_66_, lean_object* v_a_67_, lean_object* v_a_68_){
_start:
{
lean_object* v___x_69_; uint8_t v___x_70_; 
v___x_69_ = ((lean_object*)(lp_mathlib_Function_term___u2218_x27___00__closed__2));
lean_inc(v_x_66_);
v___x_70_ = l_Lean_Syntax_isOfKind(v_x_66_, v___x_69_);
if (v___x_70_ == 0)
{
lean_object* v___x_71_; lean_object* v___x_72_; 
lean_dec(v_x_66_);
v___x_71_ = lean_box(1);
v___x_72_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_72_, 0, v___x_71_);
lean_ctor_set(v___x_72_, 1, v_a_68_);
return v___x_72_;
}
else
{
lean_object* v_quotContext_73_; lean_object* v_currMacroScope_74_; lean_object* v_ref_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; uint8_t v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; 
v_quotContext_73_ = lean_ctor_get(v_a_67_, 1);
v_currMacroScope_74_ = lean_ctor_get(v_a_67_, 2);
v_ref_75_ = lean_ctor_get(v_a_67_, 5);
v___x_76_ = lean_unsigned_to_nat(0u);
v___x_77_ = l_Lean_Syntax_getArg(v_x_66_, v___x_76_);
v___x_78_ = lean_unsigned_to_nat(2u);
v___x_79_ = l_Lean_Syntax_getArg(v_x_66_, v___x_78_);
lean_dec(v_x_66_);
v___x_80_ = 0;
v___x_81_ = l_Lean_SourceInfo_fromRef(v_ref_75_, v___x_80_);
v___x_82_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__4));
v___x_83_ = lean_obj_once(&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__6, &lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__6_once, _init_lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__6);
v___x_84_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__8));
lean_inc(v_currMacroScope_74_);
lean_inc(v_quotContext_73_);
v___x_85_ = l_Lean_addMacroScope(v_quotContext_73_, v___x_84_, v_currMacroScope_74_);
v___x_86_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__10));
lean_inc_n(v___x_81_, 2);
v___x_87_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_87_, 0, v___x_81_);
lean_ctor_set(v___x_87_, 1, v___x_83_);
lean_ctor_set(v___x_87_, 2, v___x_85_);
lean_ctor_set(v___x_87_, 3, v___x_86_);
v___x_88_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__12));
v___x_89_ = l_Lean_Syntax_node2(v___x_81_, v___x_88_, v___x_77_, v___x_79_);
v___x_90_ = l_Lean_Syntax_node2(v___x_81_, v___x_82_, v___x_87_, v___x_89_);
v___x_91_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_91_, 0, v___x_90_);
lean_ctor_set(v___x_91_, 1, v_a_68_);
return v___x_91_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___boxed(lean_object* v_x_92_, lean_object* v_a_93_, lean_object* v_a_94_){
_start:
{
lean_object* v_res_95_; 
v_res_95_ = lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1(v_x_92_, v_a_93_, v_a_94_);
lean_dec_ref(v_a_93_);
return v_res_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______unexpand__Function__dcomp__1(lean_object* v_x_99_, lean_object* v_a_100_, lean_object* v_a_101_){
_start:
{
lean_object* v___x_102_; uint8_t v___x_103_; 
v___x_102_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__4));
lean_inc(v_x_99_);
v___x_103_ = l_Lean_Syntax_isOfKind(v_x_99_, v___x_102_);
if (v___x_103_ == 0)
{
lean_object* v___x_104_; lean_object* v___x_105_; 
lean_dec(v_x_99_);
v___x_104_ = lean_box(0);
v___x_105_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_105_, 0, v___x_104_);
lean_ctor_set(v___x_105_, 1, v_a_101_);
return v___x_105_;
}
else
{
lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; uint8_t v___x_109_; 
v___x_106_ = lean_unsigned_to_nat(0u);
v___x_107_ = l_Lean_Syntax_getArg(v_x_99_, v___x_106_);
v___x_108_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______unexpand__Function__dcomp__1___closed__1));
lean_inc(v___x_107_);
v___x_109_ = l_Lean_Syntax_isOfKind(v___x_107_, v___x_108_);
if (v___x_109_ == 0)
{
lean_object* v___x_110_; lean_object* v___x_111_; 
lean_dec(v___x_107_);
lean_dec(v_x_99_);
v___x_110_ = lean_box(0);
v___x_111_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_111_, 0, v___x_110_);
lean_ctor_set(v___x_111_, 1, v_a_101_);
return v___x_111_;
}
else
{
lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; uint8_t v___x_115_; 
v___x_112_ = lean_unsigned_to_nat(1u);
v___x_113_ = l_Lean_Syntax_getArg(v_x_99_, v___x_112_);
lean_dec(v_x_99_);
v___x_114_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_113_);
v___x_115_ = l_Lean_Syntax_matchesNull(v___x_113_, v___x_114_);
if (v___x_115_ == 0)
{
lean_object* v___x_116_; lean_object* v___x_117_; 
lean_dec(v___x_113_);
lean_dec(v___x_107_);
v___x_116_ = lean_box(0);
v___x_117_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_117_, 0, v___x_116_);
lean_ctor_set(v___x_117_, 1, v_a_101_);
return v___x_117_;
}
else
{
lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v_ref_120_; uint8_t v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; 
v___x_118_ = l_Lean_Syntax_getArg(v___x_113_, v___x_106_);
v___x_119_ = l_Lean_Syntax_getArg(v___x_113_, v___x_112_);
lean_dec(v___x_113_);
v_ref_120_ = l_Lean_replaceRef(v___x_107_, v_a_100_);
lean_dec(v___x_107_);
v___x_121_ = 0;
v___x_122_ = l_Lean_SourceInfo_fromRef(v_ref_120_, v___x_121_);
lean_dec(v_ref_120_);
v___x_123_ = ((lean_object*)(lp_mathlib_Function_term___u2218_x27___00__closed__2));
v___x_124_ = ((lean_object*)(lp_mathlib_Function_term___u2218_x27___00__closed__5));
lean_inc(v___x_122_);
v___x_125_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_125_, 0, v___x_122_);
lean_ctor_set(v___x_125_, 1, v___x_124_);
v___x_126_ = l_Lean_Syntax_node3(v___x_122_, v___x_123_, v___x_118_, v___x_125_, v___x_119_);
v___x_127_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_127_, 0, v___x_126_);
lean_ctor_set(v___x_127_, 1, v_a_101_);
return v___x_127_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______unexpand__Function__dcomp__1___boxed(lean_object* v_x_128_, lean_object* v_a_129_, lean_object* v_a_130_){
_start:
{
lean_object* v_res_131_; 
v_res_131_ = lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______unexpand__Function__dcomp__1(v_x_128_, v_a_129_, v_a_130_);
lean_dec(v_a_129_);
return v_res_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_prod___redArg(lean_object* v_f_132_, lean_object* v_g_133_, lean_object* v_i_134_){
_start:
{
lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; 
lean_inc(v_i_134_);
v___x_135_ = lean_apply_1(v_f_132_, v_i_134_);
v___x_136_ = lean_apply_1(v_g_133_, v_i_134_);
v___x_137_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_137_, 0, v___x_135_);
lean_ctor_set(v___x_137_, 1, v___x_136_);
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_prod(lean_object* v_00_u03b9_138_, lean_object* v_00_u03b1_139_, lean_object* v_00_u03b2_140_, lean_object* v_f_141_, lean_object* v_g_142_, lean_object* v_i_143_){
_start:
{
lean_object* v___x_144_; 
v___x_144_ = lp_mathlib_Function_prod___redArg(v_f_141_, v_g_142_, v_i_143_);
return v___x_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_diag___redArg(lean_object* v_a_145_){
_start:
{
lean_object* v___x_146_; 
lean_inc(v_a_145_);
v___x_146_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_146_, 0, v_a_145_);
lean_ctor_set(v___x_146_, 1, v_a_145_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_diag(lean_object* v_00_u03b1_147_, lean_object* v_a_148_){
_start:
{
lean_object* v___x_149_; 
lean_inc(v_a_148_);
v___x_149_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_149_, 0, v_a_148_);
lean_ctor_set(v___x_149_, 1, v_a_148_);
return v___x_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_onFun___redArg(lean_object* v_f_150_, lean_object* v_g_151_, lean_object* v_x_152_, lean_object* v_y_153_){
_start:
{
lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; 
lean_inc(v_g_151_);
v___x_154_ = lean_apply_1(v_g_151_, v_x_152_);
v___x_155_ = lean_apply_1(v_g_151_, v_y_153_);
v___x_156_ = lean_apply_2(v_f_150_, v___x_154_, v___x_155_);
return v___x_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_onFun(lean_object* v_00_u03b1_157_, lean_object* v_00_u03b2_158_, lean_object* v_00_u03c6_159_, lean_object* v_f_160_, lean_object* v_g_161_, lean_object* v_x_162_, lean_object* v_y_163_){
_start:
{
lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; 
lean_inc(v_g_161_);
v___x_164_ = lean_apply_1(v_g_161_, v_x_162_);
v___x_165_ = lean_apply_1(v_g_161_, v_y_163_);
v___x_166_ = lean_apply_2(v_f_160_, v___x_164_, v___x_165_);
return v___x_166_;
}
}
static lean_object* _init_lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___closed__1(void){
_start:
{
lean_object* v___x_187_; lean_object* v___x_188_; 
v___x_187_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___closed__0));
v___x_188_ = l_String_toRawSubstring_x27(v___x_187_);
return v___x_188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1(lean_object* v_x_200_, lean_object* v_a_201_, lean_object* v_a_202_){
_start:
{
lean_object* v___x_203_; uint8_t v___x_204_; 
v___x_203_ = ((lean_object*)(lp_mathlib_Function_term__On___00__closed__1));
lean_inc(v_x_200_);
v___x_204_ = l_Lean_Syntax_isOfKind(v_x_200_, v___x_203_);
if (v___x_204_ == 0)
{
lean_object* v___x_205_; lean_object* v___x_206_; 
lean_dec(v_x_200_);
v___x_205_ = lean_box(1);
v___x_206_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_206_, 0, v___x_205_);
lean_ctor_set(v___x_206_, 1, v_a_202_);
return v___x_206_;
}
else
{
lean_object* v_quotContext_207_; lean_object* v_currMacroScope_208_; lean_object* v_ref_209_; lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___x_213_; uint8_t v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; 
v_quotContext_207_ = lean_ctor_get(v_a_201_, 1);
v_currMacroScope_208_ = lean_ctor_get(v_a_201_, 2);
v_ref_209_ = lean_ctor_get(v_a_201_, 5);
v___x_210_ = lean_unsigned_to_nat(0u);
v___x_211_ = l_Lean_Syntax_getArg(v_x_200_, v___x_210_);
v___x_212_ = lean_unsigned_to_nat(2u);
v___x_213_ = l_Lean_Syntax_getArg(v_x_200_, v___x_212_);
lean_dec(v_x_200_);
v___x_214_ = 0;
v___x_215_ = l_Lean_SourceInfo_fromRef(v_ref_209_, v___x_214_);
v___x_216_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__4));
v___x_217_ = lean_obj_once(&lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___closed__1, &lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___closed__1_once, _init_lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___closed__1);
v___x_218_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___closed__2));
lean_inc(v_currMacroScope_208_);
lean_inc(v_quotContext_207_);
v___x_219_ = l_Lean_addMacroScope(v_quotContext_207_, v___x_218_, v_currMacroScope_208_);
v___x_220_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___closed__5));
lean_inc_n(v___x_215_, 2);
v___x_221_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_221_, 0, v___x_215_);
lean_ctor_set(v___x_221_, 1, v___x_217_);
lean_ctor_set(v___x_221_, 2, v___x_219_);
lean_ctor_set(v___x_221_, 3, v___x_220_);
v___x_222_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__12));
v___x_223_ = l_Lean_Syntax_node2(v___x_215_, v___x_222_, v___x_211_, v___x_213_);
v___x_224_ = l_Lean_Syntax_node2(v___x_215_, v___x_216_, v___x_221_, v___x_223_);
v___x_225_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_225_, 0, v___x_224_);
lean_ctor_set(v___x_225_, 1, v_a_202_);
return v___x_225_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1___boxed(lean_object* v_x_226_, lean_object* v_a_227_, lean_object* v_a_228_){
_start:
{
lean_object* v_res_229_; 
v_res_229_ = lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term__On____1(v_x_226_, v_a_227_, v_a_228_);
lean_dec_ref(v_a_227_);
return v_res_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______unexpand__Function__onFun__1(lean_object* v_x_230_, lean_object* v_a_231_, lean_object* v_a_232_){
_start:
{
lean_object* v___x_233_; uint8_t v___x_234_; 
v___x_233_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__4));
lean_inc(v_x_230_);
v___x_234_ = l_Lean_Syntax_isOfKind(v_x_230_, v___x_233_);
if (v___x_234_ == 0)
{
lean_object* v___x_235_; lean_object* v___x_236_; 
lean_dec(v_x_230_);
v___x_235_ = lean_box(0);
v___x_236_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_236_, 0, v___x_235_);
lean_ctor_set(v___x_236_, 1, v_a_232_);
return v___x_236_;
}
else
{
lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; uint8_t v___x_240_; 
v___x_237_ = lean_unsigned_to_nat(0u);
v___x_238_ = l_Lean_Syntax_getArg(v_x_230_, v___x_237_);
v___x_239_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______unexpand__Function__dcomp__1___closed__1));
lean_inc(v___x_238_);
v___x_240_ = l_Lean_Syntax_isOfKind(v___x_238_, v___x_239_);
if (v___x_240_ == 0)
{
lean_object* v___x_241_; lean_object* v___x_242_; 
lean_dec(v___x_238_);
lean_dec(v_x_230_);
v___x_241_ = lean_box(0);
v___x_242_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_242_, 0, v___x_241_);
lean_ctor_set(v___x_242_, 1, v_a_232_);
return v___x_242_;
}
else
{
lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; uint8_t v___x_246_; 
v___x_243_ = lean_unsigned_to_nat(1u);
v___x_244_ = l_Lean_Syntax_getArg(v_x_230_, v___x_243_);
lean_dec(v_x_230_);
v___x_245_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_244_);
v___x_246_ = l_Lean_Syntax_matchesNull(v___x_244_, v___x_245_);
if (v___x_246_ == 0)
{
lean_object* v___x_247_; lean_object* v___x_248_; 
lean_dec(v___x_244_);
lean_dec(v___x_238_);
v___x_247_ = lean_box(0);
v___x_248_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_248_, 0, v___x_247_);
lean_ctor_set(v___x_248_, 1, v_a_232_);
return v___x_248_;
}
else
{
lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v_ref_251_; uint8_t v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; 
v___x_249_ = l_Lean_Syntax_getArg(v___x_244_, v___x_237_);
v___x_250_ = l_Lean_Syntax_getArg(v___x_244_, v___x_243_);
lean_dec(v___x_244_);
v_ref_251_ = l_Lean_replaceRef(v___x_238_, v_a_231_);
lean_dec(v___x_238_);
v___x_252_ = 0;
v___x_253_ = l_Lean_SourceInfo_fromRef(v_ref_251_, v___x_252_);
lean_dec(v_ref_251_);
v___x_254_ = ((lean_object*)(lp_mathlib_Function_term__On___00__closed__1));
v___x_255_ = ((lean_object*)(lp_mathlib_Function_term__On___00__closed__2));
lean_inc(v___x_253_);
v___x_256_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_256_, 0, v___x_253_);
lean_ctor_set(v___x_256_, 1, v___x_255_);
v___x_257_ = l_Lean_Syntax_node3(v___x_253_, v___x_254_, v___x_249_, v___x_256_, v___x_250_);
v___x_258_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_258_, 0, v___x_257_);
lean_ctor_set(v___x_258_, 1, v_a_232_);
return v___x_258_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______unexpand__Function__onFun__1___boxed(lean_object* v_x_259_, lean_object* v_a_260_, lean_object* v_a_261_){
_start:
{
lean_object* v_res_262_; 
v_res_262_ = lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______unexpand__Function__onFun__1(v_x_259_, v_a_260_, v_a_261_);
lean_dec(v_a_260_);
return v_res_262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_swap___redArg(lean_object* v_f_263_, lean_object* v_y_264_, lean_object* v_x_265_){
_start:
{
lean_object* v___x_266_; 
v___x_266_ = lean_apply_2(v_f_263_, v_x_265_, v_y_264_);
return v___x_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_swap(lean_object* v_00_u03b1_267_, lean_object* v_00_u03b2_268_, lean_object* v_00_u03c6_269_, lean_object* v_f_270_, lean_object* v_y_271_, lean_object* v_x_272_){
_start:
{
lean_object* v___x_273_; 
v___x_273_ = lean_apply_2(v_f_270_, v_x_272_, v_y_271_);
return v___x_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_bicompl___redArg(lean_object* v_f_274_, lean_object* v_g_275_, lean_object* v_h_276_, lean_object* v_a_277_, lean_object* v_b_278_){
_start:
{
lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; 
v___x_279_ = lean_apply_1(v_g_275_, v_a_277_);
v___x_280_ = lean_apply_1(v_h_276_, v_b_278_);
v___x_281_ = lean_apply_2(v_f_274_, v___x_279_, v___x_280_);
return v___x_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_bicompl(lean_object* v_00_u03b1_282_, lean_object* v_00_u03b2_283_, lean_object* v_00_u03b3_284_, lean_object* v_00_u03b4_285_, lean_object* v_00_u03b5_286_, lean_object* v_f_287_, lean_object* v_g_288_, lean_object* v_h_289_, lean_object* v_a_290_, lean_object* v_b_291_){
_start:
{
lean_object* v___x_292_; 
v___x_292_ = lp_mathlib_Function_bicompl___redArg(v_f_287_, v_g_288_, v_h_289_, v_a_290_, v_b_291_);
return v___x_292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_bicompr___redArg(lean_object* v_f_293_, lean_object* v_g_294_, lean_object* v_a_295_, lean_object* v_b_296_){
_start:
{
lean_object* v___x_297_; lean_object* v___x_298_; 
v___x_297_ = lean_apply_2(v_g_294_, v_a_295_, v_b_296_);
v___x_298_ = lean_apply_1(v_f_293_, v___x_297_);
return v___x_298_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_bicompr(lean_object* v_00_u03b1_299_, lean_object* v_00_u03b2_300_, lean_object* v_00_u03b3_301_, lean_object* v_00_u03b4_302_, lean_object* v_f_303_, lean_object* v_g_304_, lean_object* v_a_305_, lean_object* v_b_306_){
_start:
{
lean_object* v___x_307_; 
v___x_307_ = lp_mathlib_Function_bicompr___redArg(v_f_303_, v_g_304_, v_a_305_, v_b_306_);
return v___x_307_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___closed__1(void){
_start:
{
lean_object* v___x_354_; lean_object* v___x_355_; 
v___x_354_ = ((lean_object*)(lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___closed__0));
v___x_355_ = l_String_toRawSubstring_x27(v___x_354_);
return v___x_355_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1(lean_object* v_x_367_, lean_object* v_a_368_, lean_object* v_a_369_){
_start:
{
lean_object* v___x_370_; lean_object* v___x_371_; uint8_t v___x_372_; 
v___x_370_ = lean_unsigned_to_nat(0u);
v___x_371_ = ((lean_object*)(lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__12));
lean_inc(v_x_367_);
v___x_372_ = l_Lean_Syntax_isOfKind(v_x_367_, v___x_371_);
if (v___x_372_ == 0)
{
lean_object* v___x_373_; lean_object* v___x_374_; 
lean_dec(v_x_367_);
v___x_373_ = lean_box(1);
v___x_374_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_374_, 0, v___x_373_);
lean_ctor_set(v___x_374_, 1, v_a_369_);
return v___x_374_;
}
else
{
lean_object* v_quotContext_375_; lean_object* v_currMacroScope_376_; lean_object* v_ref_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; uint8_t v___x_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; 
v_quotContext_375_ = lean_ctor_get(v_a_368_, 1);
v_currMacroScope_376_ = lean_ctor_get(v_a_368_, 2);
v_ref_377_ = lean_ctor_get(v_a_368_, 5);
v___x_378_ = l_Lean_Syntax_getArg(v_x_367_, v___x_370_);
v___x_379_ = lean_unsigned_to_nat(2u);
v___x_380_ = l_Lean_Syntax_getArg(v_x_367_, v___x_379_);
lean_dec(v_x_367_);
v___x_381_ = 0;
v___x_382_ = l_Lean_SourceInfo_fromRef(v_ref_377_, v___x_381_);
v___x_383_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__4));
v___x_384_ = lean_obj_once(&lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___closed__1, &lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___closed__1_once, _init_lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___closed__1);
v___x_385_ = ((lean_object*)(lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___closed__2));
lean_inc(v_currMacroScope_376_);
lean_inc(v_quotContext_375_);
v___x_386_ = l_Lean_addMacroScope(v_quotContext_375_, v___x_385_, v_currMacroScope_376_);
v___x_387_ = ((lean_object*)(lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___closed__5));
lean_inc_n(v___x_382_, 2);
v___x_388_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_388_, 0, v___x_382_);
lean_ctor_set(v___x_388_, 1, v___x_384_);
lean_ctor_set(v___x_388_, 2, v___x_386_);
lean_ctor_set(v___x_388_, 3, v___x_387_);
v___x_389_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__12));
v___x_390_ = l_Lean_Syntax_node2(v___x_382_, v___x_389_, v___x_378_, v___x_380_);
v___x_391_ = l_Lean_Syntax_node2(v___x_382_, v___x_383_, v___x_388_, v___x_390_);
v___x_392_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_392_, 0, v___x_391_);
lean_ctor_set(v___x_392_, 1, v_a_369_);
return v___x_392_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1___boxed(lean_object* v_x_393_, lean_object* v_a_394_, lean_object* v_a_395_){
_start:
{
lean_object* v_res_396_; 
v_res_396_ = lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______macroRules____private__Mathlib__Logic__Function__Defs__0__Function__term___u2218_u2082____1(v_x_393_, v_a_394_, v_a_395_);
lean_dec_ref(v_a_394_);
return v_res_396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______unexpand__Function__bicompr__1(lean_object* v_x_397_, lean_object* v_a_398_, lean_object* v_a_399_){
_start:
{
lean_object* v___x_400_; uint8_t v___x_401_; 
v___x_400_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______macroRules__Function__term___u2218_x27____1___closed__4));
lean_inc(v_x_397_);
v___x_401_ = l_Lean_Syntax_isOfKind(v_x_397_, v___x_400_);
if (v___x_401_ == 0)
{
lean_object* v___x_402_; lean_object* v___x_403_; 
lean_dec(v_x_397_);
v___x_402_ = lean_box(0);
v___x_403_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_403_, 0, v___x_402_);
lean_ctor_set(v___x_403_, 1, v_a_399_);
return v___x_403_;
}
else
{
lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; uint8_t v___x_407_; 
v___x_404_ = lean_unsigned_to_nat(0u);
v___x_405_ = l_Lean_Syntax_getArg(v_x_397_, v___x_404_);
v___x_406_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Function__Defs______unexpand__Function__dcomp__1___closed__1));
lean_inc(v___x_405_);
v___x_407_ = l_Lean_Syntax_isOfKind(v___x_405_, v___x_406_);
if (v___x_407_ == 0)
{
lean_object* v___x_408_; lean_object* v___x_409_; 
lean_dec(v___x_405_);
lean_dec(v_x_397_);
v___x_408_ = lean_box(0);
v___x_409_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_409_, 0, v___x_408_);
lean_ctor_set(v___x_409_, 1, v_a_399_);
return v___x_409_;
}
else
{
lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; uint8_t v___x_413_; 
v___x_410_ = lean_unsigned_to_nat(1u);
v___x_411_ = l_Lean_Syntax_getArg(v_x_397_, v___x_410_);
lean_dec(v_x_397_);
v___x_412_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_411_);
v___x_413_ = l_Lean_Syntax_matchesNull(v___x_411_, v___x_412_);
if (v___x_413_ == 0)
{
lean_object* v___x_414_; lean_object* v___x_415_; 
lean_dec(v___x_411_);
lean_dec(v___x_405_);
v___x_414_ = lean_box(0);
v___x_415_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_415_, 0, v___x_414_);
lean_ctor_set(v___x_415_, 1, v_a_399_);
return v___x_415_;
}
else
{
lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v_ref_418_; uint8_t v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; 
v___x_416_ = l_Lean_Syntax_getArg(v___x_411_, v___x_404_);
v___x_417_ = l_Lean_Syntax_getArg(v___x_411_, v___x_410_);
lean_dec(v___x_411_);
v_ref_418_ = l_Lean_replaceRef(v___x_405_, v_a_398_);
lean_dec(v___x_405_);
v___x_419_ = 0;
v___x_420_ = l_Lean_SourceInfo_fromRef(v_ref_418_, v___x_419_);
lean_dec(v_ref_418_);
v___x_421_ = ((lean_object*)(lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__12));
v___x_422_ = ((lean_object*)(lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function_term___u2218_u2082___00__closed__13));
lean_inc(v___x_420_);
v___x_423_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_423_, 0, v___x_420_);
lean_ctor_set(v___x_423_, 1, v___x_422_);
v___x_424_ = l_Lean_Syntax_node3(v___x_420_, v___x_421_, v___x_416_, v___x_423_, v___x_417_);
v___x_425_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_425_, 0, v___x_424_);
lean_ctor_set(v___x_425_, 1, v_a_399_);
return v___x_425_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______unexpand__Function__bicompr__1___boxed(lean_object* v_x_426_, lean_object* v_a_427_, lean_object* v_a_428_){
_start:
{
lean_object* v_res_429_; 
v_res_429_ = lp_mathlib___private_Mathlib_Logic_Function_Defs_0__Function___aux__Mathlib__Logic__Function__Defs______unexpand__Function__bicompr__1(v_x_426_, v_a_427_, v_a_428_);
lean_dec(v_a_427_);
return v_res_429_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Function_IsFixedPt_decidable___redArg(lean_object* v_h_430_, lean_object* v_f_431_, lean_object* v_x_432_){
_start:
{
lean_object* v___x_433_; lean_object* v___x_434_; uint8_t v___x_435_; 
lean_inc(v_x_432_);
v___x_433_ = lean_apply_1(v_f_431_, v_x_432_);
v___x_434_ = lean_apply_2(v_h_430_, v___x_433_, v_x_432_);
v___x_435_ = lean_unbox(v___x_434_);
return v___x_435_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_IsFixedPt_decidable___redArg___boxed(lean_object* v_h_436_, lean_object* v_f_437_, lean_object* v_x_438_){
_start:
{
uint8_t v_res_439_; lean_object* v_r_440_; 
v_res_439_ = lp_mathlib_Function_IsFixedPt_decidable___redArg(v_h_436_, v_f_437_, v_x_438_);
v_r_440_ = lean_box(v_res_439_);
return v_r_440_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Function_IsFixedPt_decidable(lean_object* v_00_u03b1_441_, lean_object* v_h_442_, lean_object* v_f_443_, lean_object* v_x_444_){
_start:
{
uint8_t v___x_445_; 
v___x_445_ = lp_mathlib_Function_IsFixedPt_decidable___redArg(v_h_442_, v_f_443_, v_x_444_);
return v___x_445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_IsFixedPt_decidable___boxed(lean_object* v_00_u03b1_446_, lean_object* v_h_447_, lean_object* v_f_448_, lean_object* v_x_449_){
_start:
{
uint8_t v_res_450_; lean_object* v_r_451_; 
v_res_450_ = lp_mathlib_Function_IsFixedPt_decidable(v_00_u03b1_446_, v_h_447_, v_f_448_, v_x_449_);
v_r_451_ = lean_box(v_res_450_);
return v_r_451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_map___redArg(lean_object* v_f_452_, lean_object* v_a_453_, lean_object* v_i_454_){
_start:
{
lean_object* v___x_455_; lean_object* v___x_456_; 
lean_inc(v_i_454_);
v___x_455_ = lean_apply_1(v_a_453_, v_i_454_);
v___x_456_ = lean_apply_2(v_f_452_, v_i_454_, v___x_455_);
return v___x_456_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_map(lean_object* v_00_u03b9_457_, lean_object* v_00_u03b1_458_, lean_object* v_00_u03b2_459_, lean_object* v_f_460_, lean_object* v_a_461_, lean_object* v_i_462_){
_start:
{
lean_object* v___x_463_; 
v___x_463_ = lp_mathlib_Pi_map___redArg(v_f_460_, v_a_461_, v_i_462_);
return v___x_463_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Attr_Register(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Logic_Function_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Attr_Register(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Logic_Function_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Attr_Register(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Logic_Function_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Attr_Register(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Function_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Logic_Function_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Logic_Function_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
