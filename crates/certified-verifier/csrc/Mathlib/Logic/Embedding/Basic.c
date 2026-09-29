// Lean compiler output
// Module: Mathlib.Logic.Embedding.Basic
// Imports: public import Init public meta import Init public import Mathlib.Basic.IsEmpty.Basic public import Mathlib.Data.Option.Basic public import Mathlib.Data.Prod.Basic public import Mathlib.Data.Prod.PProd public import Mathlib.Data.Sum.Basic public import Mathlib.Logic.Equiv.Basic
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
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Sigma_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Subtype_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Prod_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Sum_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Function_term___u21aa___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Function"};
static const lean_object* lp_mathlib_Function_term___u21aa___00__closed__0 = (const lean_object*)&lp_mathlib_Function_term___u21aa___00__closed__0_value;
static const lean_string_object lp_mathlib_Function_term___u21aa___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_↪_"};
static const lean_object* lp_mathlib_Function_term___u21aa___00__closed__1 = (const lean_object*)&lp_mathlib_Function_term___u21aa___00__closed__1_value;
static const lean_ctor_object lp_mathlib_Function_term___u21aa___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Function_term___u21aa___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(225, 8, 186, 189, 152, 89, 197, 12)}};
static const lean_ctor_object lp_mathlib_Function_term___u21aa___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Function_term___u21aa___00__closed__2_value_aux_0),((lean_object*)&lp_mathlib_Function_term___u21aa___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(121, 91, 31, 135, 77, 235, 184, 192)}};
static const lean_object* lp_mathlib_Function_term___u21aa___00__closed__2 = (const lean_object*)&lp_mathlib_Function_term___u21aa___00__closed__2_value;
static const lean_string_object lp_mathlib_Function_term___u21aa___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Function_term___u21aa___00__closed__3 = (const lean_object*)&lp_mathlib_Function_term___u21aa___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Function_term___u21aa___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Function_term___u21aa___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Function_term___u21aa___00__closed__4 = (const lean_object*)&lp_mathlib_Function_term___u21aa___00__closed__4_value;
static const lean_string_object lp_mathlib_Function_term___u21aa___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ↪ "};
static const lean_object* lp_mathlib_Function_term___u21aa___00__closed__5 = (const lean_object*)&lp_mathlib_Function_term___u21aa___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Function_term___u21aa___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Function_term___u21aa___00__closed__5_value)}};
static const lean_object* lp_mathlib_Function_term___u21aa___00__closed__6 = (const lean_object*)&lp_mathlib_Function_term___u21aa___00__closed__6_value;
static const lean_string_object lp_mathlib_Function_term___u21aa___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Function_term___u21aa___00__closed__7 = (const lean_object*)&lp_mathlib_Function_term___u21aa___00__closed__7_value;
static const lean_ctor_object lp_mathlib_Function_term___u21aa___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Function_term___u21aa___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Function_term___u21aa___00__closed__8 = (const lean_object*)&lp_mathlib_Function_term___u21aa___00__closed__8_value;
static const lean_ctor_object lp_mathlib_Function_term___u21aa___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Function_term___u21aa___00__closed__8_value),((lean_object*)(((size_t)(25) << 1) | 1))}};
static const lean_object* lp_mathlib_Function_term___u21aa___00__closed__9 = (const lean_object*)&lp_mathlib_Function_term___u21aa___00__closed__9_value;
static const lean_ctor_object lp_mathlib_Function_term___u21aa___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Function_term___u21aa___00__closed__4_value),((lean_object*)&lp_mathlib_Function_term___u21aa___00__closed__6_value),((lean_object*)&lp_mathlib_Function_term___u21aa___00__closed__9_value)}};
static const lean_object* lp_mathlib_Function_term___u21aa___00__closed__10 = (const lean_object*)&lp_mathlib_Function_term___u21aa___00__closed__10_value;
static const lean_ctor_object lp_mathlib_Function_term___u21aa___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_Function_term___u21aa___00__closed__2_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(26) << 1) | 1)),((lean_object*)&lp_mathlib_Function_term___u21aa___00__closed__10_value)}};
static const lean_object* lp_mathlib_Function_term___u21aa___00__closed__11 = (const lean_object*)&lp_mathlib_Function_term___u21aa___00__closed__11_value;
LEAN_EXPORT const lean_object* lp_mathlib_Function_term___u21aa__ = (const lean_object*)&lp_mathlib_Function_term___u21aa___00__closed__11_value;
static const lean_string_object lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__0 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__0_value;
static const lean_string_object lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__1 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__1_value;
static const lean_string_object lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__2 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__2_value;
static const lean_string_object lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__3 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__4 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__4_value;
static const lean_string_object lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Embedding"};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__5 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__5_value;
static lean_once_cell_t lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__6;
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(91, 51, 192, 19, 17, 86, 160, 205)}};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__7 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__7_value;
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Function_term___u21aa___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(225, 8, 186, 189, 152, 89, 197, 12)}};
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(25, 64, 244, 242, 24, 63, 210, 118)}};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__8 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__8_value;
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__9 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__9_value;
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__8_value)}};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__10 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__10_value;
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__11 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__11_value;
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__9_value),((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__11_value)}};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__12 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__12_value;
static const lean_string_object lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__13 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__13_value;
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__14 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__14_value;
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______unexpand__Function__Embedding__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______unexpand__Function__Embedding__1___closed__0 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______unexpand__Function__Embedding__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______unexpand__Function__Embedding__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______unexpand__Function__Embedding__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______unexpand__Function__Embedding__1___closed__1 = (const lean_object*)&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______unexpand__Function__Embedding__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______unexpand__Function__Embedding__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______unexpand__Function__Embedding__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_toEmbedding___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_toEmbedding___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_toEmbedding(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_coeEmbedding___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_toEmbedding, .m_arity = 3, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Equiv_coeEmbedding___closed__0 = (const lean_object*)&lp_mathlib_Equiv_coeEmbedding___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_coeEmbedding(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_instUniqueOfIsEmpty___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_instUniqueOfIsEmpty___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Function_Embedding_instUniqueOfIsEmpty___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Function_Embedding_instUniqueOfIsEmpty___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Function_Embedding_instUniqueOfIsEmpty___closed__0 = (const lean_object*)&lp_mathlib_Function_Embedding_instUniqueOfIsEmpty___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_instUniqueOfIsEmpty(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_refl___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_refl___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Function_Embedding_refl___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Function_Embedding_refl___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Function_Embedding_refl___closed__0 = (const lean_object*)&lp_mathlib_Function_Embedding_refl___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_refl(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_trans___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_trans(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Function_Embedding_instTrans___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Function_Embedding_trans, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Function_Embedding_instTrans___closed__0 = (const lean_object*)&lp_mathlib_Function_Embedding_instTrans___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Function_Embedding_instTrans = (const lean_object*)&lp_mathlib_Function_Embedding_instTrans___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_congr___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_congr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_ofIsEmpty(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_setValue___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_setValue___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_setValue___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_setValue(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_some___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Function_Embedding_some___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Function_Embedding_some___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Function_Embedding_some___closed__0 = (const lean_object*)&lp_mathlib_Function_Embedding_some___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_some(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_optionMap___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_optionMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_optionMap(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_subtype___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_subtype___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Function_Embedding_subtype___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Function_Embedding_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Function_Embedding_subtype___closed__0 = (const lean_object*)&lp_mathlib_Function_Embedding_subtype___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_subtype(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_punit___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_punit___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_punit___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_punit(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_oneEmbeddingEquiv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_oneEmbeddingEquiv___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_oneEmbeddingEquiv___redArg___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Function_Embedding_oneEmbeddingEquiv___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Function_Embedding_oneEmbeddingEquiv___redArg___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Function_Embedding_oneEmbeddingEquiv___redArg___closed__0 = (const lean_object*)&lp_mathlib_Function_Embedding_oneEmbeddingEquiv___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_oneEmbeddingEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_oneEmbeddingEquiv(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_sectL___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_sectL___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_sectL(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_sectR___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_sectR___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_sectR(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_prodMap___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_prodMap___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_prodMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_prodMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_pprodMap___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_pprodMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_pprodMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_sumMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_sumMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_inl___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Function_Embedding_inl___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Function_Embedding_inl___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Function_Embedding_inl___closed__0 = (const lean_object*)&lp_mathlib_Function_Embedding_inl___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_inl(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_inr___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Function_Embedding_inr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Function_Embedding_inr___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Function_Embedding_inr___closed__0 = (const lean_object*)&lp_mathlib_Function_Embedding_inr___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_inr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_sigmaMk___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_sigmaMk___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_sigmaMk(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_sigmaMap___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_sigmaMap___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_sigmaMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_sigmaMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_piCongrRight___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_piCongrRight___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_piCongrRight(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_arrowCongrRight___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_arrowCongrRight___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_arrowCongrRight___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_arrowCongrRight(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_subtypeMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_subtypeMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_asEmbedding___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_asEmbedding(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_subtypeInjectiveEquivEmbedding___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Function_Embedding_sigmaMap___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_subtypeInjectiveEquivEmbedding___closed__0 = (const lean_object*)&lp_mathlib_Equiv_subtypeInjectiveEquivEmbedding___closed__0_value;
static const lean_ctor_object lp_mathlib_Equiv_subtypeInjectiveEquivEmbedding___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_subtypeInjectiveEquivEmbedding___closed__0_value),((lean_object*)&lp_mathlib_Equiv_subtypeInjectiveEquivEmbedding___closed__0_value)}};
static const lean_object* lp_mathlib_Equiv_subtypeInjectiveEquivEmbedding___closed__1 = (const lean_object*)&lp_mathlib_Equiv_subtypeInjectiveEquivEmbedding___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeInjectiveEquivEmbedding(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_embeddingCongr___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_embeddingCongr___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_embeddingCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_embeddingCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_subtypeOrLeftEmbedding___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_subtypeOrLeftEmbedding___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_subtypeOrLeftEmbedding(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_impEmbedding___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_impEmbedding___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Subtype_impEmbedding___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subtype_impEmbedding___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Subtype_impEmbedding___closed__0 = (const lean_object*)&lp_mathlib_Subtype_impEmbedding___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Subtype_impEmbedding(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__6(void){
_start:
{
lean_object* v___x_38_; lean_object* v___x_39_; 
v___x_38_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__5));
v___x_39_ = l_String_toRawSubstring_x27(v___x_38_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1(lean_object* v_x_59_, lean_object* v_a_60_, lean_object* v_a_61_){
_start:
{
lean_object* v___x_62_; uint8_t v___x_63_; 
v___x_62_ = ((lean_object*)(lp_mathlib_Function_term___u21aa___00__closed__2));
lean_inc(v_x_59_);
v___x_63_ = l_Lean_Syntax_isOfKind(v_x_59_, v___x_62_);
if (v___x_63_ == 0)
{
lean_object* v___x_64_; lean_object* v___x_65_; 
lean_dec(v_x_59_);
v___x_64_ = lean_box(1);
v___x_65_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_65_, 0, v___x_64_);
lean_ctor_set(v___x_65_, 1, v_a_61_);
return v___x_65_;
}
else
{
lean_object* v_quotContext_66_; lean_object* v_currMacroScope_67_; lean_object* v_ref_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; uint8_t v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; 
v_quotContext_66_ = lean_ctor_get(v_a_60_, 1);
v_currMacroScope_67_ = lean_ctor_get(v_a_60_, 2);
v_ref_68_ = lean_ctor_get(v_a_60_, 5);
v___x_69_ = lean_unsigned_to_nat(0u);
v___x_70_ = l_Lean_Syntax_getArg(v_x_59_, v___x_69_);
v___x_71_ = lean_unsigned_to_nat(2u);
v___x_72_ = l_Lean_Syntax_getArg(v_x_59_, v___x_71_);
lean_dec(v_x_59_);
v___x_73_ = 0;
v___x_74_ = l_Lean_SourceInfo_fromRef(v_ref_68_, v___x_73_);
v___x_75_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__4));
v___x_76_ = lean_obj_once(&lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__6, &lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__6_once, _init_lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__6);
v___x_77_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__7));
lean_inc(v_currMacroScope_67_);
lean_inc(v_quotContext_66_);
v___x_78_ = l_Lean_addMacroScope(v_quotContext_66_, v___x_77_, v_currMacroScope_67_);
v___x_79_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__12));
lean_inc_n(v___x_74_, 2);
v___x_80_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_80_, 0, v___x_74_);
lean_ctor_set(v___x_80_, 1, v___x_76_);
lean_ctor_set(v___x_80_, 2, v___x_78_);
lean_ctor_set(v___x_80_, 3, v___x_79_);
v___x_81_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__14));
v___x_82_ = l_Lean_Syntax_node2(v___x_74_, v___x_81_, v___x_70_, v___x_72_);
v___x_83_ = l_Lean_Syntax_node2(v___x_74_, v___x_75_, v___x_80_, v___x_82_);
v___x_84_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_84_, 0, v___x_83_);
lean_ctor_set(v___x_84_, 1, v_a_61_);
return v___x_84_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___boxed(lean_object* v_x_85_, lean_object* v_a_86_, lean_object* v_a_87_){
_start:
{
lean_object* v_res_88_; 
v_res_88_ = lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1(v_x_85_, v_a_86_, v_a_87_);
lean_dec_ref(v_a_86_);
return v_res_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______unexpand__Function__Embedding__1(lean_object* v_x_92_, lean_object* v_a_93_, lean_object* v_a_94_){
_start:
{
lean_object* v___x_95_; uint8_t v___x_96_; 
v___x_95_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______macroRules__Function__term___u21aa____1___closed__4));
lean_inc(v_x_92_);
v___x_96_ = l_Lean_Syntax_isOfKind(v_x_92_, v___x_95_);
if (v___x_96_ == 0)
{
lean_object* v___x_97_; lean_object* v___x_98_; 
lean_dec(v_x_92_);
v___x_97_ = lean_box(0);
v___x_98_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_98_, 0, v___x_97_);
lean_ctor_set(v___x_98_, 1, v_a_94_);
return v___x_98_;
}
else
{
lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; uint8_t v___x_102_; 
v___x_99_ = lean_unsigned_to_nat(0u);
v___x_100_ = l_Lean_Syntax_getArg(v_x_92_, v___x_99_);
v___x_101_ = ((lean_object*)(lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______unexpand__Function__Embedding__1___closed__1));
lean_inc(v___x_100_);
v___x_102_ = l_Lean_Syntax_isOfKind(v___x_100_, v___x_101_);
if (v___x_102_ == 0)
{
lean_object* v___x_103_; lean_object* v___x_104_; 
lean_dec(v___x_100_);
lean_dec(v_x_92_);
v___x_103_ = lean_box(0);
v___x_104_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_104_, 0, v___x_103_);
lean_ctor_set(v___x_104_, 1, v_a_94_);
return v___x_104_;
}
else
{
lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; uint8_t v___x_108_; 
v___x_105_ = lean_unsigned_to_nat(1u);
v___x_106_ = l_Lean_Syntax_getArg(v_x_92_, v___x_105_);
lean_dec(v_x_92_);
v___x_107_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_106_);
v___x_108_ = l_Lean_Syntax_matchesNull(v___x_106_, v___x_107_);
if (v___x_108_ == 0)
{
lean_object* v___x_109_; lean_object* v___x_110_; 
lean_dec(v___x_106_);
lean_dec(v___x_100_);
v___x_109_ = lean_box(0);
v___x_110_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_110_, 0, v___x_109_);
lean_ctor_set(v___x_110_, 1, v_a_94_);
return v___x_110_;
}
else
{
lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v_ref_113_; uint8_t v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; 
v___x_111_ = l_Lean_Syntax_getArg(v___x_106_, v___x_99_);
v___x_112_ = l_Lean_Syntax_getArg(v___x_106_, v___x_105_);
lean_dec(v___x_106_);
v_ref_113_ = l_Lean_replaceRef(v___x_100_, v_a_93_);
lean_dec(v___x_100_);
v___x_114_ = 0;
v___x_115_ = l_Lean_SourceInfo_fromRef(v_ref_113_, v___x_114_);
lean_dec(v_ref_113_);
v___x_116_ = ((lean_object*)(lp_mathlib_Function_term___u21aa___00__closed__2));
v___x_117_ = ((lean_object*)(lp_mathlib_Function_term___u21aa___00__closed__5));
lean_inc(v___x_115_);
v___x_118_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_118_, 0, v___x_115_);
lean_ctor_set(v___x_118_, 1, v___x_117_);
v___x_119_ = l_Lean_Syntax_node3(v___x_115_, v___x_116_, v___x_111_, v___x_118_, v___x_112_);
v___x_120_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_120_, 0, v___x_119_);
lean_ctor_set(v___x_120_, 1, v_a_94_);
return v___x_120_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______unexpand__Function__Embedding__1___boxed(lean_object* v_x_121_, lean_object* v_a_122_, lean_object* v_a_123_){
_start:
{
lean_object* v_res_124_; 
v_res_124_ = lp_mathlib_Function___aux__Mathlib__Logic__Embedding__Basic______unexpand__Function__Embedding__1(v_x_121_, v_a_122_, v_a_123_);
lean_dec(v_a_122_);
return v_res_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_toEmbedding___redArg___lam__0(lean_object* v_f_125_, lean_object* v___y_126_){
_start:
{
lean_object* v_toFun_127_; lean_object* v___x_128_; 
v_toFun_127_ = lean_ctor_get(v_f_125_, 0);
lean_inc(v_toFun_127_);
lean_dec_ref(v_f_125_);
v___x_128_ = lean_apply_1(v_toFun_127_, v___y_126_);
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_toEmbedding___redArg(lean_object* v_f_129_){
_start:
{
lean_object* v___f_130_; 
v___f_130_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_130_, 0, v_f_129_);
return v___f_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_toEmbedding(lean_object* v_00_u03b1_131_, lean_object* v_00_u03b2_132_, lean_object* v_f_133_){
_start:
{
lean_object* v___f_134_; 
v___f_134_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_134_, 0, v_f_133_);
return v___f_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_coeEmbedding(lean_object* v_00_u03b1_136_, lean_object* v_00_u03b2_137_){
_start:
{
lean_object* v___x_138_; 
v___x_138_ = ((lean_object*)(lp_mathlib_Equiv_coeEmbedding___closed__0));
return v___x_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_instUniqueOfIsEmpty___lam__0(lean_object* v_a_139_){
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_instUniqueOfIsEmpty___lam__0___boxed(lean_object* v_a_140_){
_start:
{
lean_object* v_res_141_; 
v_res_141_ = lp_mathlib_Function_Embedding_instUniqueOfIsEmpty___lam__0(v_a_140_);
lean_dec(v_a_140_);
return v_res_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_instUniqueOfIsEmpty(lean_object* v_00_u03b1_143_, lean_object* v_00_u03b2_144_, lean_object* v_inst_145_){
_start:
{
lean_object* v___f_146_; 
v___f_146_ = ((lean_object*)(lp_mathlib_Function_Embedding_instUniqueOfIsEmpty___closed__0));
return v___f_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_refl___lam__0(lean_object* v___y_147_){
_start:
{
lean_inc(v___y_147_);
return v___y_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_refl___lam__0___boxed(lean_object* v___y_148_){
_start:
{
lean_object* v_res_149_; 
v_res_149_ = lp_mathlib_Function_Embedding_refl___lam__0(v___y_148_);
lean_dec(v___y_148_);
return v_res_149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_refl(lean_object* v_00_u03b1_151_){
_start:
{
lean_object* v___f_152_; 
v___f_152_ = ((lean_object*)(lp_mathlib_Function_Embedding_refl___closed__0));
return v___f_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_trans___redArg___lam__0(lean_object* v_f_153_, lean_object* v_g_154_, lean_object* v___y_155_){
_start:
{
lean_object* v___x_156_; lean_object* v___x_157_; 
v___x_156_ = lean_apply_1(v_f_153_, v___y_155_);
v___x_157_ = lean_apply_1(v_g_154_, v___x_156_);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_trans___redArg(lean_object* v_f_158_, lean_object* v_g_159_){
_start:
{
lean_object* v___f_160_; 
v___f_160_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_trans___redArg___lam__0), 3, 2);
lean_closure_set(v___f_160_, 0, v_f_158_);
lean_closure_set(v___f_160_, 1, v_g_159_);
return v___f_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_trans(lean_object* v_00_u03b1_161_, lean_object* v_00_u03b2_162_, lean_object* v_00_u03b3_163_, lean_object* v_f_164_, lean_object* v_g_165_){
_start:
{
lean_object* v___f_166_; 
v___f_166_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_trans___redArg___lam__0), 3, 2);
lean_closure_set(v___f_166_, 0, v_f_164_);
lean_closure_set(v___f_166_, 1, v_g_165_);
return v___f_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_congr___redArg(lean_object* v_e_u2081_169_, lean_object* v_e_u2082_170_, lean_object* v_f_171_){
_start:
{
lean_object* v___x_172_; lean_object* v___f_173_; lean_object* v___f_174_; lean_object* v___f_175_; lean_object* v___f_176_; 
v___x_172_ = lp_mathlib_Equiv_symm___redArg(v_e_u2081_169_);
v___f_173_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_173_, 0, v___x_172_);
v___f_174_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_174_, 0, v_e_u2082_170_);
v___f_175_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_trans___redArg___lam__0), 3, 2);
lean_closure_set(v___f_175_, 0, v_f_171_);
lean_closure_set(v___f_175_, 1, v___f_174_);
v___f_176_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_trans___redArg___lam__0), 3, 2);
lean_closure_set(v___f_176_, 0, v___f_173_);
lean_closure_set(v___f_176_, 1, v___f_175_);
return v___f_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_congr(lean_object* v_00_u03b1_177_, lean_object* v_00_u03b2_178_, lean_object* v_00_u03b3_179_, lean_object* v_00_u03b4_180_, lean_object* v_e_u2081_181_, lean_object* v_e_u2082_182_, lean_object* v_f_183_){
_start:
{
lean_object* v___x_184_; 
v___x_184_ = lp_mathlib_Function_Embedding_congr___redArg(v_e_u2081_181_, v_e_u2082_182_, v_f_183_);
return v___x_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_ofIsEmpty(lean_object* v_00_u03b1_185_, lean_object* v_00_u03b2_186_, lean_object* v_inst_187_){
_start:
{
lean_object* v___f_188_; 
v___f_188_ = ((lean_object*)(lp_mathlib_Function_Embedding_instUniqueOfIsEmpty___closed__0));
return v___f_188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_setValue___redArg___lam__0(lean_object* v_inst_189_, lean_object* v_inst_190_, lean_object* v_f_191_, lean_object* v_a_192_, lean_object* v_b_193_, lean_object* v_a_x27_194_){
_start:
{
lean_object* v___x_195_; uint8_t v___x_196_; 
lean_inc(v_a_x27_194_);
v___x_195_ = lean_apply_1(v_inst_189_, v_a_x27_194_);
v___x_196_ = lean_unbox(v___x_195_);
if (v___x_196_ == 0)
{
lean_object* v___x_197_; uint8_t v___x_198_; 
lean_inc(v_a_x27_194_);
v___x_197_ = lean_apply_1(v_inst_190_, v_a_x27_194_);
v___x_198_ = lean_unbox(v___x_197_);
if (v___x_198_ == 0)
{
lean_object* v___x_199_; 
lean_dec(v_a_192_);
v___x_199_ = lean_apply_1(v_f_191_, v_a_x27_194_);
return v___x_199_;
}
else
{
lean_object* v___x_200_; 
lean_dec(v_a_x27_194_);
v___x_200_ = lean_apply_1(v_f_191_, v_a_192_);
return v___x_200_;
}
}
else
{
lean_dec(v_a_x27_194_);
lean_dec(v_a_192_);
lean_dec(v_f_191_);
lean_dec_ref(v_inst_190_);
lean_inc(v_b_193_);
return v_b_193_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_setValue___redArg___lam__0___boxed(lean_object* v_inst_201_, lean_object* v_inst_202_, lean_object* v_f_203_, lean_object* v_a_204_, lean_object* v_b_205_, lean_object* v_a_x27_206_){
_start:
{
lean_object* v_res_207_; 
v_res_207_ = lp_mathlib_Function_Embedding_setValue___redArg___lam__0(v_inst_201_, v_inst_202_, v_f_203_, v_a_204_, v_b_205_, v_a_x27_206_);
lean_dec(v_b_205_);
return v_res_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_setValue___redArg(lean_object* v_f_208_, lean_object* v_a_209_, lean_object* v_b_210_, lean_object* v_inst_211_, lean_object* v_inst_212_){
_start:
{
lean_object* v___f_213_; 
v___f_213_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_setValue___redArg___lam__0___boxed), 6, 5);
lean_closure_set(v___f_213_, 0, v_inst_211_);
lean_closure_set(v___f_213_, 1, v_inst_212_);
lean_closure_set(v___f_213_, 2, v_f_208_);
lean_closure_set(v___f_213_, 3, v_a_209_);
lean_closure_set(v___f_213_, 4, v_b_210_);
return v___f_213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_setValue(lean_object* v_00_u03b1_214_, lean_object* v_00_u03b2_215_, lean_object* v_f_216_, lean_object* v_a_217_, lean_object* v_b_218_, lean_object* v_inst_219_, lean_object* v_inst_220_){
_start:
{
lean_object* v___f_221_; 
v___f_221_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_setValue___redArg___lam__0___boxed), 6, 5);
lean_closure_set(v___f_221_, 0, v_inst_219_);
lean_closure_set(v___f_221_, 1, v_inst_220_);
lean_closure_set(v___f_221_, 2, v_f_216_);
lean_closure_set(v___f_221_, 3, v_a_217_);
lean_closure_set(v___f_221_, 4, v_b_218_);
return v___f_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_some___lam__0(lean_object* v_val_222_){
_start:
{
lean_object* v___x_223_; 
v___x_223_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_223_, 0, v_val_222_);
return v___x_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_some(lean_object* v_00_u03b1_225_){
_start:
{
lean_object* v___f_226_; 
v___f_226_ = ((lean_object*)(lp_mathlib_Function_Embedding_some___closed__0));
return v___f_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_optionMap___redArg___lam__0(lean_object* v_f_227_, lean_object* v___y_228_){
_start:
{
if (lean_obj_tag(v___y_228_) == 0)
{
lean_object* v___x_229_; 
lean_dec(v_f_227_);
v___x_229_ = lean_box(0);
return v___x_229_;
}
else
{
lean_object* v_val_230_; lean_object* v___x_232_; uint8_t v_isShared_233_; uint8_t v_isSharedCheck_238_; 
v_val_230_ = lean_ctor_get(v___y_228_, 0);
v_isSharedCheck_238_ = !lean_is_exclusive(v___y_228_);
if (v_isSharedCheck_238_ == 0)
{
v___x_232_ = v___y_228_;
v_isShared_233_ = v_isSharedCheck_238_;
goto v_resetjp_231_;
}
else
{
lean_inc(v_val_230_);
lean_dec(v___y_228_);
v___x_232_ = lean_box(0);
v_isShared_233_ = v_isSharedCheck_238_;
goto v_resetjp_231_;
}
v_resetjp_231_:
{
lean_object* v___x_234_; lean_object* v___x_236_; 
v___x_234_ = lean_apply_1(v_f_227_, v_val_230_);
if (v_isShared_233_ == 0)
{
lean_ctor_set(v___x_232_, 0, v___x_234_);
v___x_236_ = v___x_232_;
goto v_reusejp_235_;
}
else
{
lean_object* v_reuseFailAlloc_237_; 
v_reuseFailAlloc_237_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_237_, 0, v___x_234_);
v___x_236_ = v_reuseFailAlloc_237_;
goto v_reusejp_235_;
}
v_reusejp_235_:
{
return v___x_236_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_optionMap___redArg(lean_object* v_f_239_){
_start:
{
lean_object* v___f_240_; 
v___f_240_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_optionMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_240_, 0, v_f_239_);
return v___f_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_optionMap(lean_object* v_00_u03b1_241_, lean_object* v_00_u03b2_242_, lean_object* v_f_243_){
_start:
{
lean_object* v___f_244_; 
v___f_244_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_optionMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_244_, 0, v_f_243_);
return v___f_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_subtype___lam__0(lean_object* v_self_245_){
_start:
{
lean_inc(v_self_245_);
return v_self_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_subtype___lam__0___boxed(lean_object* v_self_246_){
_start:
{
lean_object* v_res_247_; 
v_res_247_ = lp_mathlib_Function_Embedding_subtype___lam__0(v_self_246_);
lean_dec(v_self_246_);
return v_res_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_subtype(lean_object* v_00_u03b1_249_, lean_object* v_p_250_){
_start:
{
lean_object* v___f_251_; 
v___f_251_ = ((lean_object*)(lp_mathlib_Function_Embedding_subtype___closed__0));
return v___f_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_punit___redArg___lam__0(lean_object* v_b_252_, lean_object* v_x_253_){
_start:
{
lean_inc(v_b_252_);
return v_b_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_punit___redArg___lam__0___boxed(lean_object* v_b_254_, lean_object* v_x_255_){
_start:
{
lean_object* v_res_256_; 
v_res_256_ = lp_mathlib_Function_Embedding_punit___redArg___lam__0(v_b_254_, v_x_255_);
lean_dec(v_b_254_);
return v_res_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_punit___redArg(lean_object* v_b_257_){
_start:
{
lean_object* v___f_258_; 
v___f_258_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_punit___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_258_, 0, v_b_257_);
return v___f_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_punit(lean_object* v_00_u03b2_259_, lean_object* v_b_260_){
_start:
{
lean_object* v___f_261_; 
v___f_261_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_punit___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_261_, 0, v_b_260_);
return v___f_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_oneEmbeddingEquiv___redArg___lam__0(lean_object* v_a_262_, lean_object* v___y_263_){
_start:
{
lean_inc(v_a_262_);
return v_a_262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_oneEmbeddingEquiv___redArg___lam__0___boxed(lean_object* v_a_264_, lean_object* v___y_265_){
_start:
{
lean_object* v_res_266_; 
v_res_266_ = lp_mathlib_Function_Embedding_oneEmbeddingEquiv___redArg___lam__0(v_a_264_, v___y_265_);
lean_dec(v___y_265_);
lean_dec(v_a_264_);
return v_res_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_oneEmbeddingEquiv___redArg___lam__1(lean_object* v_inst_267_, lean_object* v_f_268_){
_start:
{
lean_object* v___x_269_; 
v___x_269_ = lean_apply_1(v_f_268_, v_inst_267_);
return v___x_269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_oneEmbeddingEquiv___redArg(lean_object* v_inst_271_){
_start:
{
lean_object* v___f_272_; lean_object* v___f_273_; lean_object* v___x_274_; 
v___f_272_ = ((lean_object*)(lp_mathlib_Function_Embedding_oneEmbeddingEquiv___redArg___closed__0));
v___f_273_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_oneEmbeddingEquiv___redArg___lam__1), 2, 1);
lean_closure_set(v___f_273_, 0, v_inst_271_);
v___x_274_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_274_, 0, v___f_273_);
lean_ctor_set(v___x_274_, 1, v___f_272_);
return v___x_274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_oneEmbeddingEquiv(lean_object* v_one_275_, lean_object* v_00_u03b1_276_, lean_object* v_inst_277_){
_start:
{
lean_object* v___x_278_; 
v___x_278_ = lp_mathlib_Function_Embedding_oneEmbeddingEquiv___redArg(v_inst_277_);
return v___x_278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_sectL___redArg___lam__0(lean_object* v_b_279_, lean_object* v_a_280_){
_start:
{
lean_object* v___x_281_; 
v___x_281_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_281_, 0, v_a_280_);
lean_ctor_set(v___x_281_, 1, v_b_279_);
return v___x_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_sectL___redArg(lean_object* v_b_282_){
_start:
{
lean_object* v___f_283_; 
v___f_283_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_sectL___redArg___lam__0), 2, 1);
lean_closure_set(v___f_283_, 0, v_b_282_);
return v___f_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_sectL(lean_object* v_00_u03b1_284_, lean_object* v_00_u03b2_285_, lean_object* v_b_286_){
_start:
{
lean_object* v___f_287_; 
v___f_287_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_sectL___redArg___lam__0), 2, 1);
lean_closure_set(v___f_287_, 0, v_b_286_);
return v___f_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_sectR___redArg___lam__0(lean_object* v_a_288_, lean_object* v_b_289_){
_start:
{
lean_object* v___x_290_; 
v___x_290_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_290_, 0, v_a_288_);
lean_ctor_set(v___x_290_, 1, v_b_289_);
return v___x_290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_sectR___redArg(lean_object* v_a_291_){
_start:
{
lean_object* v___f_292_; 
v___f_292_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_sectR___redArg___lam__0), 2, 1);
lean_closure_set(v___f_292_, 0, v_a_291_);
return v___f_292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_sectR(lean_object* v_00_u03b1_293_, lean_object* v_a_294_, lean_object* v_00_u03b2_295_){
_start:
{
lean_object* v___f_296_; 
v___f_296_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_sectR___redArg___lam__0), 2, 1);
lean_closure_set(v___f_296_, 0, v_a_294_);
return v___f_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_prodMap___redArg___lam__0(lean_object* v_e_u2081_297_, lean_object* v___y_298_){
_start:
{
lean_object* v___x_299_; 
v___x_299_ = lean_apply_1(v_e_u2081_297_, v___y_298_);
return v___x_299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_prodMap___redArg___lam__1(lean_object* v_e_u2082_300_, lean_object* v___y_301_){
_start:
{
lean_object* v___x_302_; 
v___x_302_ = lean_apply_1(v_e_u2082_300_, v___y_301_);
return v___x_302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_prodMap___redArg(lean_object* v_e_u2081_303_, lean_object* v_e_u2082_304_){
_start:
{
lean_object* v___f_305_; lean_object* v___f_306_; lean_object* v___x_307_; 
v___f_305_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_prodMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_305_, 0, v_e_u2081_303_);
v___f_306_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_prodMap___redArg___lam__1), 2, 1);
lean_closure_set(v___f_306_, 0, v_e_u2082_304_);
v___x_307_ = lean_alloc_closure((void*)(l_Prod_map), 7, 6);
lean_closure_set(v___x_307_, 0, lean_box(0));
lean_closure_set(v___x_307_, 1, lean_box(0));
lean_closure_set(v___x_307_, 2, lean_box(0));
lean_closure_set(v___x_307_, 3, lean_box(0));
lean_closure_set(v___x_307_, 4, v___f_305_);
lean_closure_set(v___x_307_, 5, v___f_306_);
return v___x_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_prodMap(lean_object* v_00_u03b1_308_, lean_object* v_00_u03b2_309_, lean_object* v_00_u03b3_310_, lean_object* v_00_u03b4_311_, lean_object* v_e_u2081_312_, lean_object* v_e_u2082_313_){
_start:
{
lean_object* v___x_314_; 
v___x_314_ = lp_mathlib_Function_Embedding_prodMap___redArg(v_e_u2081_312_, v_e_u2082_313_);
return v___x_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_pprodMap___redArg___lam__0(lean_object* v_e_u2081_315_, lean_object* v_e_u2082_316_, lean_object* v_x_317_){
_start:
{
lean_object* v_fst_318_; lean_object* v_snd_319_; lean_object* v___x_321_; uint8_t v_isShared_322_; uint8_t v_isSharedCheck_328_; 
v_fst_318_ = lean_ctor_get(v_x_317_, 0);
v_snd_319_ = lean_ctor_get(v_x_317_, 1);
v_isSharedCheck_328_ = !lean_is_exclusive(v_x_317_);
if (v_isSharedCheck_328_ == 0)
{
v___x_321_ = v_x_317_;
v_isShared_322_ = v_isSharedCheck_328_;
goto v_resetjp_320_;
}
else
{
lean_inc(v_snd_319_);
lean_inc(v_fst_318_);
lean_dec(v_x_317_);
v___x_321_ = lean_box(0);
v_isShared_322_ = v_isSharedCheck_328_;
goto v_resetjp_320_;
}
v_resetjp_320_:
{
lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_326_; 
v___x_323_ = lean_apply_1(v_e_u2081_315_, v_fst_318_);
v___x_324_ = lean_apply_1(v_e_u2082_316_, v_snd_319_);
if (v_isShared_322_ == 0)
{
lean_ctor_set(v___x_321_, 1, v___x_324_);
lean_ctor_set(v___x_321_, 0, v___x_323_);
v___x_326_ = v___x_321_;
goto v_reusejp_325_;
}
else
{
lean_object* v_reuseFailAlloc_327_; 
v_reuseFailAlloc_327_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_327_, 0, v___x_323_);
lean_ctor_set(v_reuseFailAlloc_327_, 1, v___x_324_);
v___x_326_ = v_reuseFailAlloc_327_;
goto v_reusejp_325_;
}
v_reusejp_325_:
{
return v___x_326_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_pprodMap___redArg(lean_object* v_e_u2081_329_, lean_object* v_e_u2082_330_){
_start:
{
lean_object* v___f_331_; 
v___f_331_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_pprodMap___redArg___lam__0), 3, 2);
lean_closure_set(v___f_331_, 0, v_e_u2081_329_);
lean_closure_set(v___f_331_, 1, v_e_u2082_330_);
return v___f_331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_pprodMap(lean_object* v_00_u03b1_332_, lean_object* v_00_u03b2_333_, lean_object* v_00_u03b3_334_, lean_object* v_00_u03b4_335_, lean_object* v_e_u2081_336_, lean_object* v_e_u2082_337_){
_start:
{
lean_object* v___f_338_; 
v___f_338_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_pprodMap___redArg___lam__0), 3, 2);
lean_closure_set(v___f_338_, 0, v_e_u2081_336_);
lean_closure_set(v___f_338_, 1, v_e_u2082_337_);
return v___f_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_sumMap___redArg(lean_object* v_e_u2081_339_, lean_object* v_e_u2082_340_){
_start:
{
lean_object* v___f_341_; lean_object* v___f_342_; lean_object* v___x_343_; 
v___f_341_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_prodMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_341_, 0, v_e_u2081_339_);
v___f_342_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_prodMap___redArg___lam__1), 2, 1);
lean_closure_set(v___f_342_, 0, v_e_u2082_340_);
v___x_343_ = lean_alloc_closure((void*)(l_Sum_map), 7, 6);
lean_closure_set(v___x_343_, 0, lean_box(0));
lean_closure_set(v___x_343_, 1, lean_box(0));
lean_closure_set(v___x_343_, 2, lean_box(0));
lean_closure_set(v___x_343_, 3, lean_box(0));
lean_closure_set(v___x_343_, 4, v___f_341_);
lean_closure_set(v___x_343_, 5, v___f_342_);
return v___x_343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_sumMap(lean_object* v_00_u03b1_344_, lean_object* v_00_u03b2_345_, lean_object* v_00_u03b3_346_, lean_object* v_00_u03b4_347_, lean_object* v_e_u2081_348_, lean_object* v_e_u2082_349_){
_start:
{
lean_object* v___x_350_; 
v___x_350_ = lp_mathlib_Function_Embedding_sumMap___redArg(v_e_u2081_348_, v_e_u2082_349_);
return v___x_350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_inl___lam__0(lean_object* v_val_351_){
_start:
{
lean_object* v___x_352_; 
v___x_352_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_352_, 0, v_val_351_);
return v___x_352_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_inl(lean_object* v_00_u03b1_354_, lean_object* v_00_u03b2_355_){
_start:
{
lean_object* v___f_356_; 
v___f_356_ = ((lean_object*)(lp_mathlib_Function_Embedding_inl___closed__0));
return v___f_356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_inr___lam__0(lean_object* v_val_357_){
_start:
{
lean_object* v___x_358_; 
v___x_358_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_358_, 0, v_val_357_);
return v___x_358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_inr(lean_object* v_00_u03b1_360_, lean_object* v_00_u03b2_361_){
_start:
{
lean_object* v___f_362_; 
v___f_362_ = ((lean_object*)(lp_mathlib_Function_Embedding_inr___closed__0));
return v___f_362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_sigmaMk___redArg___lam__0(lean_object* v_a_363_, lean_object* v_snd_364_){
_start:
{
lean_object* v___x_365_; 
v___x_365_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_365_, 0, v_a_363_);
lean_ctor_set(v___x_365_, 1, v_snd_364_);
return v___x_365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_sigmaMk___redArg(lean_object* v_a_366_){
_start:
{
lean_object* v___f_367_; 
v___f_367_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_sigmaMk___redArg___lam__0), 2, 1);
lean_closure_set(v___f_367_, 0, v_a_366_);
return v___f_367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_sigmaMk(lean_object* v_00_u03b1_368_, lean_object* v_00_u03b2_369_, lean_object* v_a_370_){
_start:
{
lean_object* v___f_371_; 
v___f_371_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_sigmaMk___redArg___lam__0), 2, 1);
lean_closure_set(v___f_371_, 0, v_a_370_);
return v___f_371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_sigmaMap___redArg___lam__0(lean_object* v_f_372_, lean_object* v___y_373_){
_start:
{
lean_object* v___x_374_; 
v___x_374_ = lean_apply_1(v_f_372_, v___y_373_);
return v___x_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_sigmaMap___redArg___lam__1(lean_object* v_g_375_, lean_object* v_a_376_, lean_object* v___y_377_){
_start:
{
lean_object* v___x_378_; 
v___x_378_ = lean_apply_2(v_g_375_, v_a_376_, v___y_377_);
return v___x_378_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_sigmaMap___redArg(lean_object* v_f_379_, lean_object* v_g_380_){
_start:
{
lean_object* v___f_381_; lean_object* v___f_382_; lean_object* v___x_383_; 
v___f_381_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_sigmaMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_381_, 0, v_f_379_);
v___f_382_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_sigmaMap___redArg___lam__1), 3, 1);
lean_closure_set(v___f_382_, 0, v_g_380_);
v___x_383_ = lean_alloc_closure((void*)(lp_mathlib_Sigma_map), 7, 6);
lean_closure_set(v___x_383_, 0, lean_box(0));
lean_closure_set(v___x_383_, 1, lean_box(0));
lean_closure_set(v___x_383_, 2, lean_box(0));
lean_closure_set(v___x_383_, 3, lean_box(0));
lean_closure_set(v___x_383_, 4, v___f_381_);
lean_closure_set(v___x_383_, 5, v___f_382_);
return v___x_383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_sigmaMap(lean_object* v_00_u03b1_384_, lean_object* v_00_u03b1_x27_385_, lean_object* v_00_u03b2_386_, lean_object* v_00_u03b2_x27_387_, lean_object* v_f_388_, lean_object* v_g_389_){
_start:
{
lean_object* v___x_390_; 
v___x_390_ = lp_mathlib_Function_Embedding_sigmaMap___redArg(v_f_388_, v_g_389_);
return v___x_390_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_piCongrRight___redArg___lam__0(lean_object* v_e_391_, lean_object* v_f_392_, lean_object* v_a_393_){
_start:
{
lean_object* v___x_394_; lean_object* v___x_395_; 
lean_inc(v_a_393_);
v___x_394_ = lean_apply_1(v_f_392_, v_a_393_);
v___x_395_ = lean_apply_2(v_e_391_, v_a_393_, v___x_394_);
return v___x_395_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_piCongrRight___redArg(lean_object* v_e_396_){
_start:
{
lean_object* v___f_397_; 
v___f_397_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_piCongrRight___redArg___lam__0), 3, 1);
lean_closure_set(v___f_397_, 0, v_e_396_);
return v___f_397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_piCongrRight(lean_object* v_00_u03b1_398_, lean_object* v_00_u03b2_399_, lean_object* v_00_u03b3_400_, lean_object* v_e_401_){
_start:
{
lean_object* v___f_402_; 
v___f_402_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_piCongrRight___redArg___lam__0), 3, 1);
lean_closure_set(v___f_402_, 0, v_e_401_);
return v___f_402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_arrowCongrRight___redArg___lam__0(lean_object* v_e_403_, lean_object* v_x_404_, lean_object* v___y_405_){
_start:
{
lean_object* v___x_406_; 
v___x_406_ = lean_apply_1(v_e_403_, v___y_405_);
return v___x_406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_arrowCongrRight___redArg___lam__0___boxed(lean_object* v_e_407_, lean_object* v_x_408_, lean_object* v___y_409_){
_start:
{
lean_object* v_res_410_; 
v_res_410_ = lp_mathlib_Function_Embedding_arrowCongrRight___redArg___lam__0(v_e_407_, v_x_408_, v___y_409_);
lean_dec(v_x_408_);
return v_res_410_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_arrowCongrRight___redArg(lean_object* v_e_411_){
_start:
{
lean_object* v___f_412_; lean_object* v___f_413_; 
v___f_412_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_arrowCongrRight___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_412_, 0, v_e_411_);
v___f_413_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_piCongrRight___redArg___lam__0), 3, 1);
lean_closure_set(v___f_413_, 0, v___f_412_);
return v___f_413_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_arrowCongrRight(lean_object* v_00_u03b1_414_, lean_object* v_00_u03b2_415_, lean_object* v_00_u03b3_416_, lean_object* v_e_417_){
_start:
{
lean_object* v___x_418_; 
v___x_418_ = lp_mathlib_Function_Embedding_arrowCongrRight___redArg(v_e_417_);
return v___x_418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_subtypeMap___redArg(lean_object* v_f_419_){
_start:
{
lean_object* v___f_420_; lean_object* v___x_421_; 
v___f_420_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_sigmaMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_420_, 0, v_f_419_);
v___x_421_ = lean_alloc_closure((void*)(lp_mathlib_Subtype_map), 7, 6);
lean_closure_set(v___x_421_, 0, lean_box(0));
lean_closure_set(v___x_421_, 1, lean_box(0));
lean_closure_set(v___x_421_, 2, lean_box(0));
lean_closure_set(v___x_421_, 3, lean_box(0));
lean_closure_set(v___x_421_, 4, v___f_420_);
lean_closure_set(v___x_421_, 5, lean_box(0));
return v___x_421_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_subtypeMap(lean_object* v_00_u03b1_422_, lean_object* v_00_u03b2_423_, lean_object* v_p_424_, lean_object* v_q_425_, lean_object* v_f_426_, lean_object* v_h_427_){
_start:
{
lean_object* v___x_428_; 
v___x_428_ = lp_mathlib_Function_Embedding_subtypeMap___redArg(v_f_426_);
return v___x_428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_asEmbedding___redArg(lean_object* v_e_429_){
_start:
{
lean_object* v___f_430_; lean_object* v___f_431_; lean_object* v___f_432_; 
v___f_430_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_430_, 0, v_e_429_);
v___f_431_ = ((lean_object*)(lp_mathlib_Function_Embedding_subtype___closed__0));
v___f_432_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_trans___redArg___lam__0), 3, 2);
lean_closure_set(v___f_432_, 0, v___f_430_);
lean_closure_set(v___f_432_, 1, v___f_431_);
return v___f_432_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_asEmbedding(lean_object* v_00_u03b2_433_, lean_object* v_00_u03b1_434_, lean_object* v_p_435_, lean_object* v_e_436_){
_start:
{
lean_object* v___x_437_; 
v___x_437_ = lp_mathlib_Equiv_asEmbedding___redArg(v_e_436_);
return v___x_437_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeInjectiveEquivEmbedding(lean_object* v_00_u03b1_441_, lean_object* v_00_u03b2_442_){
_start:
{
lean_object* v___x_443_; 
v___x_443_ = ((lean_object*)(lp_mathlib_Equiv_subtypeInjectiveEquivEmbedding___closed__1));
return v___x_443_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_embeddingCongr___redArg___lam__0(lean_object* v_h_444_, lean_object* v_h_x27_445_, lean_object* v_f_446_, lean_object* v___y_447_){
_start:
{
lean_object* v___x_15__overap_448_; lean_object* v___x_449_; 
v___x_15__overap_448_ = lp_mathlib_Function_Embedding_congr___redArg(v_h_444_, v_h_x27_445_, v_f_446_);
v___x_449_ = lean_apply_1(v___x_15__overap_448_, v___y_447_);
return v___x_449_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_embeddingCongr___redArg___lam__1(lean_object* v_h_450_, lean_object* v_h_x27_451_, lean_object* v_f_452_, lean_object* v___y_453_){
_start:
{
lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_19__overap_456_; lean_object* v___x_457_; 
v___x_454_ = lp_mathlib_Equiv_symm___redArg(v_h_450_);
v___x_455_ = lp_mathlib_Equiv_symm___redArg(v_h_x27_451_);
v___x_19__overap_456_ = lp_mathlib_Function_Embedding_congr___redArg(v___x_454_, v___x_455_, v_f_452_);
v___x_457_ = lean_apply_1(v___x_19__overap_456_, v___y_453_);
return v___x_457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_embeddingCongr___redArg(lean_object* v_h_458_, lean_object* v_h_x27_459_){
_start:
{
lean_object* v___f_460_; lean_object* v___f_461_; lean_object* v___x_462_; 
lean_inc_ref(v_h_x27_459_);
lean_inc_ref(v_h_458_);
v___f_460_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_embeddingCongr___redArg___lam__0), 4, 2);
lean_closure_set(v___f_460_, 0, v_h_458_);
lean_closure_set(v___f_460_, 1, v_h_x27_459_);
v___f_461_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_embeddingCongr___redArg___lam__1), 4, 2);
lean_closure_set(v___f_461_, 0, v_h_458_);
lean_closure_set(v___f_461_, 1, v_h_x27_459_);
v___x_462_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_462_, 0, v___f_460_);
lean_ctor_set(v___x_462_, 1, v___f_461_);
return v___x_462_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_embeddingCongr(lean_object* v_00_u03b1_463_, lean_object* v_00_u03b2_464_, lean_object* v_00_u03b3_465_, lean_object* v_00_u03b4_466_, lean_object* v_h_467_, lean_object* v_h_x27_468_){
_start:
{
lean_object* v___x_469_; 
v___x_469_ = lp_mathlib_Equiv_embeddingCongr___redArg(v_h_467_, v_h_x27_468_);
return v___x_469_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_subtypeOrLeftEmbedding___redArg___lam__0(lean_object* v_inst_470_, lean_object* v_x_471_){
_start:
{
lean_object* v___x_472_; uint8_t v___x_473_; 
lean_inc(v_x_471_);
v___x_472_ = lean_apply_1(v_inst_470_, v_x_471_);
v___x_473_ = lean_unbox(v___x_472_);
if (v___x_473_ == 0)
{
lean_object* v___x_474_; 
v___x_474_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_474_, 0, v_x_471_);
return v___x_474_;
}
else
{
lean_object* v___x_475_; 
v___x_475_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_475_, 0, v_x_471_);
return v___x_475_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_subtypeOrLeftEmbedding___redArg(lean_object* v_inst_476_){
_start:
{
lean_object* v___f_477_; 
v___f_477_ = lean_alloc_closure((void*)(lp_mathlib_subtypeOrLeftEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_477_, 0, v_inst_476_);
return v___f_477_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_subtypeOrLeftEmbedding(lean_object* v_00_u03b1_478_, lean_object* v_p_479_, lean_object* v_q_480_, lean_object* v_inst_481_){
_start:
{
lean_object* v___f_482_; 
v___f_482_ = lean_alloc_closure((void*)(lp_mathlib_subtypeOrLeftEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_482_, 0, v_inst_481_);
return v___f_482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_impEmbedding___lam__0(lean_object* v_x_483_){
_start:
{
lean_inc(v_x_483_);
return v_x_483_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_impEmbedding___lam__0___boxed(lean_object* v_x_484_){
_start:
{
lean_object* v_res_485_; 
v_res_485_ = lp_mathlib_Subtype_impEmbedding___lam__0(v_x_484_);
lean_dec(v_x_484_);
return v_res_485_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_impEmbedding(lean_object* v_00_u03b1_487_, lean_object* v_p_488_, lean_object* v_q_489_, lean_object* v_h_490_){
_start:
{
lean_object* v___f_491_; 
v___f_491_ = ((lean_object*)(lp_mathlib_Subtype_impEmbedding___closed__0));
return v___f_491_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_IsEmpty_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Option_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Prod_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Prod_PProd(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Sum_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Logic_Embedding_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_IsEmpty_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Option_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Prod_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Prod_PProd(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Sum_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Logic_Embedding_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Basic_IsEmpty_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Option_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Prod_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Prod_PProd(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Sum_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Logic_Embedding_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_IsEmpty_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Option_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Prod_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Prod_PProd(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Sum_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Embedding_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Logic_Embedding_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Logic_Embedding_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
