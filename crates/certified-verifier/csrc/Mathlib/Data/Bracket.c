// Lean compiler output
// Module: Mathlib.Data.Bracket
// Imports: public import Init public meta import Init public import Mathlib.Init
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
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term_u2045___x2c___u2046___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 9, .m_data = "term⁅_,_⁆"};
static const lean_object* lp_mathlib_term_u2045___x2c___u2046___closed__0 = (const lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__0_value;
static const lean_ctor_object lp_mathlib_term_u2045___x2c___u2046___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__0_value),LEAN_SCALAR_PTR_LITERAL(56, 64, 144, 192, 237, 157, 40, 76)}};
static const lean_object* lp_mathlib_term_u2045___x2c___u2046___closed__1 = (const lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__1_value;
static const lean_string_object lp_mathlib_term_u2045___x2c___u2046___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term_u2045___x2c___u2046___closed__2 = (const lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__2_value;
static const lean_ctor_object lp_mathlib_term_u2045___x2c___u2046___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term_u2045___x2c___u2046___closed__3 = (const lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__3_value;
static const lean_string_object lp_mathlib_term_u2045___x2c___u2046___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⁅"};
static const lean_object* lp_mathlib_term_u2045___x2c___u2046___closed__4 = (const lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__4_value;
static const lean_ctor_object lp_mathlib_term_u2045___x2c___u2046___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__4_value)}};
static const lean_object* lp_mathlib_term_u2045___x2c___u2046___closed__5 = (const lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__5_value;
static const lean_string_object lp_mathlib_term_u2045___x2c___u2046___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term_u2045___x2c___u2046___closed__6 = (const lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__6_value;
static const lean_ctor_object lp_mathlib_term_u2045___x2c___u2046___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term_u2045___x2c___u2046___closed__7 = (const lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__7_value;
static const lean_ctor_object lp_mathlib_term_u2045___x2c___u2046___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_term_u2045___x2c___u2046___closed__8 = (const lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__8_value;
static const lean_ctor_object lp_mathlib_term_u2045___x2c___u2046___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__3_value),((lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__5_value),((lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__8_value)}};
static const lean_object* lp_mathlib_term_u2045___x2c___u2046___closed__9 = (const lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__9_value;
static const lean_string_object lp_mathlib_term_u2045___x2c___u2046___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_term_u2045___x2c___u2046___closed__10 = (const lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__10_value;
static const lean_ctor_object lp_mathlib_term_u2045___x2c___u2046___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__10_value)}};
static const lean_object* lp_mathlib_term_u2045___x2c___u2046___closed__11 = (const lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__11_value;
static const lean_ctor_object lp_mathlib_term_u2045___x2c___u2046___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__3_value),((lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__9_value),((lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__11_value)}};
static const lean_object* lp_mathlib_term_u2045___x2c___u2046___closed__12 = (const lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__12_value;
static const lean_ctor_object lp_mathlib_term_u2045___x2c___u2046___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__3_value),((lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__12_value),((lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__8_value)}};
static const lean_object* lp_mathlib_term_u2045___x2c___u2046___closed__13 = (const lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__13_value;
static const lean_string_object lp_mathlib_term_u2045___x2c___u2046___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⁆"};
static const lean_object* lp_mathlib_term_u2045___x2c___u2046___closed__14 = (const lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__14_value;
static const lean_ctor_object lp_mathlib_term_u2045___x2c___u2046___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__14_value)}};
static const lean_object* lp_mathlib_term_u2045___x2c___u2046___closed__15 = (const lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__15_value;
static const lean_ctor_object lp_mathlib_term_u2045___x2c___u2046___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__3_value),((lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__13_value),((lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__15_value)}};
static const lean_object* lp_mathlib_term_u2045___x2c___u2046___closed__16 = (const lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__16_value;
static const lean_ctor_object lp_mathlib_term_u2045___x2c___u2046___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__16_value)}};
static const lean_object* lp_mathlib_term_u2045___x2c___u2046___closed__17 = (const lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__17_value;
LEAN_EXPORT const lean_object* lp_mathlib_term_u2045___x2c___u2046 = (const lean_object*)&lp_mathlib_term_u2045___x2c___u2046___closed__17_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "Bracket.bracket"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__6;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Bracket"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__7_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "bracket"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(153, 239, 151, 54, 230, 145, 193, 80)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__9_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(93, 33, 16, 91, 219, 245, 216, 54)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__9_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__11_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__12_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Bracket______unexpand__Bracket__bracket__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Bracket______unexpand__Bracket__bracket__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Bracket______unexpand__Bracket__bracket__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Bracket______unexpand__Bracket__bracket__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Bracket______unexpand__Bracket__bracket__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Bracket______unexpand__Bracket__bracket__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Bracket______unexpand__Bracket__bracket__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Bracket______unexpand__Bracket__bracket__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Bracket______unexpand__Bracket__bracket__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__6(void){
_start:
{
lean_object* v___x_53_; lean_object* v___x_54_; 
v___x_53_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__5));
v___x_54_ = l_String_toRawSubstring_x27(v___x_53_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1(lean_object* v_x_69_, lean_object* v_a_70_, lean_object* v_a_71_){
_start:
{
lean_object* v___x_72_; uint8_t v___x_73_; 
v___x_72_ = ((lean_object*)(lp_mathlib_term_u2045___x2c___u2046___closed__1));
lean_inc(v_x_69_);
v___x_73_ = l_Lean_Syntax_isOfKind(v_x_69_, v___x_72_);
if (v___x_73_ == 0)
{
lean_object* v___x_74_; lean_object* v___x_75_; 
lean_dec(v_x_69_);
v___x_74_ = lean_box(1);
v___x_75_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_75_, 0, v___x_74_);
lean_ctor_set(v___x_75_, 1, v_a_71_);
return v___x_75_;
}
else
{
lean_object* v_quotContext_76_; lean_object* v_currMacroScope_77_; lean_object* v_ref_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; uint8_t v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; 
v_quotContext_76_ = lean_ctor_get(v_a_70_, 1);
v_currMacroScope_77_ = lean_ctor_get(v_a_70_, 2);
v_ref_78_ = lean_ctor_get(v_a_70_, 5);
v___x_79_ = lean_unsigned_to_nat(1u);
v___x_80_ = l_Lean_Syntax_getArg(v_x_69_, v___x_79_);
v___x_81_ = lean_unsigned_to_nat(3u);
v___x_82_ = l_Lean_Syntax_getArg(v_x_69_, v___x_81_);
lean_dec(v_x_69_);
v___x_83_ = 0;
v___x_84_ = l_Lean_SourceInfo_fromRef(v_ref_78_, v___x_83_);
v___x_85_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__4));
v___x_86_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__6, &lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__6_once, _init_lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__6);
v___x_87_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__9));
lean_inc(v_currMacroScope_77_);
lean_inc(v_quotContext_76_);
v___x_88_ = l_Lean_addMacroScope(v_quotContext_76_, v___x_87_, v_currMacroScope_77_);
v___x_89_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__11));
lean_inc_n(v___x_84_, 2);
v___x_90_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_90_, 0, v___x_84_);
lean_ctor_set(v___x_90_, 1, v___x_86_);
lean_ctor_set(v___x_90_, 2, v___x_88_);
lean_ctor_set(v___x_90_, 3, v___x_89_);
v___x_91_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__13));
v___x_92_ = l_Lean_Syntax_node2(v___x_84_, v___x_91_, v___x_80_, v___x_82_);
v___x_93_ = l_Lean_Syntax_node2(v___x_84_, v___x_85_, v___x_90_, v___x_92_);
v___x_94_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_94_, 0, v___x_93_);
lean_ctor_set(v___x_94_, 1, v_a_71_);
return v___x_94_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___boxed(lean_object* v_x_95_, lean_object* v_a_96_, lean_object* v_a_97_){
_start:
{
lean_object* v_res_98_; 
v_res_98_ = lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1(v_x_95_, v_a_96_, v_a_97_);
lean_dec_ref(v_a_96_);
return v_res_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Bracket______unexpand__Bracket__bracket__1(lean_object* v_x_102_, lean_object* v_a_103_, lean_object* v_a_104_){
_start:
{
lean_object* v___x_105_; uint8_t v___x_106_; 
v___x_105_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Bracket______macroRules__term_u2045___x2c___u2046__1___closed__4));
lean_inc(v_x_102_);
v___x_106_ = l_Lean_Syntax_isOfKind(v_x_102_, v___x_105_);
if (v___x_106_ == 0)
{
lean_object* v___x_107_; lean_object* v___x_108_; 
lean_dec(v_x_102_);
v___x_107_ = lean_box(0);
v___x_108_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_108_, 0, v___x_107_);
lean_ctor_set(v___x_108_, 1, v_a_104_);
return v___x_108_;
}
else
{
lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; uint8_t v___x_112_; 
v___x_109_ = lean_unsigned_to_nat(0u);
v___x_110_ = l_Lean_Syntax_getArg(v_x_102_, v___x_109_);
v___x_111_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Bracket______unexpand__Bracket__bracket__1___closed__1));
lean_inc(v___x_110_);
v___x_112_ = l_Lean_Syntax_isOfKind(v___x_110_, v___x_111_);
if (v___x_112_ == 0)
{
lean_object* v___x_113_; lean_object* v___x_114_; 
lean_dec(v___x_110_);
lean_dec(v_x_102_);
v___x_113_ = lean_box(0);
v___x_114_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_114_, 0, v___x_113_);
lean_ctor_set(v___x_114_, 1, v_a_104_);
return v___x_114_;
}
else
{
lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; uint8_t v___x_118_; 
v___x_115_ = lean_unsigned_to_nat(1u);
v___x_116_ = l_Lean_Syntax_getArg(v_x_102_, v___x_115_);
lean_dec(v_x_102_);
v___x_117_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_116_);
v___x_118_ = l_Lean_Syntax_matchesNull(v___x_116_, v___x_117_);
if (v___x_118_ == 0)
{
lean_object* v___x_119_; lean_object* v___x_120_; 
lean_dec(v___x_116_);
lean_dec(v___x_110_);
v___x_119_ = lean_box(0);
v___x_120_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_120_, 0, v___x_119_);
lean_ctor_set(v___x_120_, 1, v_a_104_);
return v___x_120_;
}
else
{
lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v_ref_123_; uint8_t v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; 
v___x_121_ = l_Lean_Syntax_getArg(v___x_116_, v___x_109_);
v___x_122_ = l_Lean_Syntax_getArg(v___x_116_, v___x_115_);
lean_dec(v___x_116_);
v_ref_123_ = l_Lean_replaceRef(v___x_110_, v_a_103_);
lean_dec(v___x_110_);
v___x_124_ = 0;
v___x_125_ = l_Lean_SourceInfo_fromRef(v_ref_123_, v___x_124_);
lean_dec(v_ref_123_);
v___x_126_ = ((lean_object*)(lp_mathlib_term_u2045___x2c___u2046___closed__1));
v___x_127_ = ((lean_object*)(lp_mathlib_term_u2045___x2c___u2046___closed__4));
lean_inc_n(v___x_125_, 3);
v___x_128_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_128_, 0, v___x_125_);
lean_ctor_set(v___x_128_, 1, v___x_127_);
v___x_129_ = ((lean_object*)(lp_mathlib_term_u2045___x2c___u2046___closed__10));
v___x_130_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_130_, 0, v___x_125_);
lean_ctor_set(v___x_130_, 1, v___x_129_);
v___x_131_ = ((lean_object*)(lp_mathlib_term_u2045___x2c___u2046___closed__14));
v___x_132_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_132_, 0, v___x_125_);
lean_ctor_set(v___x_132_, 1, v___x_131_);
v___x_133_ = l_Lean_Syntax_node5(v___x_125_, v___x_126_, v___x_128_, v___x_121_, v___x_130_, v___x_122_, v___x_132_);
v___x_134_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_134_, 0, v___x_133_);
lean_ctor_set(v___x_134_, 1, v_a_104_);
return v___x_134_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Bracket______unexpand__Bracket__bracket__1___boxed(lean_object* v_x_135_, lean_object* v_a_136_, lean_object* v_a_137_){
_start:
{
lean_object* v_res_138_; 
v_res_138_ = lp_mathlib___aux__Mathlib__Data__Bracket______unexpand__Bracket__bracket__1(v_x_135_, v_a_136_, v_a_137_);
lean_dec(v_a_136_);
return v_res_138_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Bracket(uint8_t builtin) {
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
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Bracket(uint8_t builtin) {
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
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Bracket(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Data_Bracket(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Bracket(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Bracket(builtin);
}
#ifdef __cplusplus
}
#endif
