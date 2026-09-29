// Lean compiler output
// Module: Mathlib.Logic.Relator
// Imports: public import Init public meta import Init public import Mathlib.Logic.Function.Defs
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Relator_term___u21d2___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Relator"};
static const lean_object* lp_mathlib_Relator_term___u21d2___00__closed__0 = (const lean_object*)&lp_mathlib_Relator_term___u21d2___00__closed__0_value;
static const lean_string_object lp_mathlib_Relator_term___u21d2___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_⇒_"};
static const lean_object* lp_mathlib_Relator_term___u21d2___00__closed__1 = (const lean_object*)&lp_mathlib_Relator_term___u21d2___00__closed__1_value;
static const lean_ctor_object lp_mathlib_Relator_term___u21d2___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Relator_term___u21d2___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(105, 23, 85, 121, 202, 52, 241, 30)}};
static const lean_ctor_object lp_mathlib_Relator_term___u21d2___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Relator_term___u21d2___00__closed__2_value_aux_0),((lean_object*)&lp_mathlib_Relator_term___u21d2___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(30, 221, 203, 243, 202, 201, 160, 54)}};
static const lean_object* lp_mathlib_Relator_term___u21d2___00__closed__2 = (const lean_object*)&lp_mathlib_Relator_term___u21d2___00__closed__2_value;
static const lean_string_object lp_mathlib_Relator_term___u21d2___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Relator_term___u21d2___00__closed__3 = (const lean_object*)&lp_mathlib_Relator_term___u21d2___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Relator_term___u21d2___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Relator_term___u21d2___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Relator_term___u21d2___00__closed__4 = (const lean_object*)&lp_mathlib_Relator_term___u21d2___00__closed__4_value;
static const lean_string_object lp_mathlib_Relator_term___u21d2___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ⇒ "};
static const lean_object* lp_mathlib_Relator_term___u21d2___00__closed__5 = (const lean_object*)&lp_mathlib_Relator_term___u21d2___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Relator_term___u21d2___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Relator_term___u21d2___00__closed__5_value)}};
static const lean_object* lp_mathlib_Relator_term___u21d2___00__closed__6 = (const lean_object*)&lp_mathlib_Relator_term___u21d2___00__closed__6_value;
static const lean_string_object lp_mathlib_Relator_term___u21d2___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Relator_term___u21d2___00__closed__7 = (const lean_object*)&lp_mathlib_Relator_term___u21d2___00__closed__7_value;
static const lean_ctor_object lp_mathlib_Relator_term___u21d2___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Relator_term___u21d2___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Relator_term___u21d2___00__closed__8 = (const lean_object*)&lp_mathlib_Relator_term___u21d2___00__closed__8_value;
static const lean_ctor_object lp_mathlib_Relator_term___u21d2___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Relator_term___u21d2___00__closed__8_value),((lean_object*)(((size_t)(40) << 1) | 1))}};
static const lean_object* lp_mathlib_Relator_term___u21d2___00__closed__9 = (const lean_object*)&lp_mathlib_Relator_term___u21d2___00__closed__9_value;
static const lean_ctor_object lp_mathlib_Relator_term___u21d2___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Relator_term___u21d2___00__closed__4_value),((lean_object*)&lp_mathlib_Relator_term___u21d2___00__closed__6_value),((lean_object*)&lp_mathlib_Relator_term___u21d2___00__closed__9_value)}};
static const lean_object* lp_mathlib_Relator_term___u21d2___00__closed__10 = (const lean_object*)&lp_mathlib_Relator_term___u21d2___00__closed__10_value;
static const lean_ctor_object lp_mathlib_Relator_term___u21d2___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_Relator_term___u21d2___00__closed__2_value),((lean_object*)(((size_t)(40) << 1) | 1)),((lean_object*)(((size_t)(41) << 1) | 1)),((lean_object*)&lp_mathlib_Relator_term___u21d2___00__closed__10_value)}};
static const lean_object* lp_mathlib_Relator_term___u21d2___00__closed__11 = (const lean_object*)&lp_mathlib_Relator_term___u21d2___00__closed__11_value;
LEAN_EXPORT const lean_object* lp_mathlib_Relator_term___u21d2__ = (const lean_object*)&lp_mathlib_Relator_term___u21d2___00__closed__11_value;
static const lean_string_object lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__0 = (const lean_object*)&lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__0_value;
static const lean_string_object lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__1 = (const lean_object*)&lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__1_value;
static const lean_string_object lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__2 = (const lean_object*)&lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__2_value;
static const lean_string_object lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__3 = (const lean_object*)&lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__4 = (const lean_object*)&lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__4_value;
static const lean_string_object lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "LiftFun"};
static const lean_object* lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__5 = (const lean_object*)&lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__5_value;
static lean_once_cell_t lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__6;
static const lean_ctor_object lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(81, 234, 28, 235, 70, 102, 92, 3)}};
static const lean_object* lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__7 = (const lean_object*)&lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__7_value;
static const lean_ctor_object lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Relator_term___u21d2___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(105, 23, 85, 121, 202, 52, 241, 30)}};
static const lean_ctor_object lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(11, 12, 26, 42, 55, 237, 128, 241)}};
static const lean_object* lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__8 = (const lean_object*)&lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__8_value;
static const lean_ctor_object lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__9 = (const lean_object*)&lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__9_value;
static const lean_ctor_object lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__10 = (const lean_object*)&lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__10_value;
static const lean_string_object lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__11 = (const lean_object*)&lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__11_value;
static const lean_ctor_object lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__12 = (const lean_object*)&lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Relator___aux__Mathlib__Logic__Relator______unexpand__Relator__LiftFun__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Relator___aux__Mathlib__Logic__Relator______unexpand__Relator__LiftFun__1___closed__0 = (const lean_object*)&lp_mathlib_Relator___aux__Mathlib__Logic__Relator______unexpand__Relator__LiftFun__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Relator___aux__Mathlib__Logic__Relator______unexpand__Relator__LiftFun__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Relator___aux__Mathlib__Logic__Relator______unexpand__Relator__LiftFun__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Relator___aux__Mathlib__Logic__Relator______unexpand__Relator__LiftFun__1___closed__1 = (const lean_object*)&lp_mathlib_Relator___aux__Mathlib__Logic__Relator______unexpand__Relator__LiftFun__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Relator___aux__Mathlib__Logic__Relator______unexpand__Relator__LiftFun__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Relator___aux__Mathlib__Logic__Relator______unexpand__Relator__LiftFun__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__6(void){
_start:
{
lean_object* v___x_38_; lean_object* v___x_39_; 
v___x_38_ = ((lean_object*)(lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__5));
v___x_39_ = l_String_toRawSubstring_x27(v___x_38_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1(lean_object* v_x_54_, lean_object* v_a_55_, lean_object* v_a_56_){
_start:
{
lean_object* v___x_57_; uint8_t v___x_58_; 
v___x_57_ = ((lean_object*)(lp_mathlib_Relator_term___u21d2___00__closed__2));
lean_inc(v_x_54_);
v___x_58_ = l_Lean_Syntax_isOfKind(v_x_54_, v___x_57_);
if (v___x_58_ == 0)
{
lean_object* v___x_59_; lean_object* v___x_60_; 
lean_dec(v_x_54_);
v___x_59_ = lean_box(1);
v___x_60_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_60_, 0, v___x_59_);
lean_ctor_set(v___x_60_, 1, v_a_56_);
return v___x_60_;
}
else
{
lean_object* v_quotContext_61_; lean_object* v_currMacroScope_62_; lean_object* v_ref_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; uint8_t v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; 
v_quotContext_61_ = lean_ctor_get(v_a_55_, 1);
v_currMacroScope_62_ = lean_ctor_get(v_a_55_, 2);
v_ref_63_ = lean_ctor_get(v_a_55_, 5);
v___x_64_ = lean_unsigned_to_nat(0u);
v___x_65_ = l_Lean_Syntax_getArg(v_x_54_, v___x_64_);
v___x_66_ = lean_unsigned_to_nat(2u);
v___x_67_ = l_Lean_Syntax_getArg(v_x_54_, v___x_66_);
lean_dec(v_x_54_);
v___x_68_ = 0;
v___x_69_ = l_Lean_SourceInfo_fromRef(v_ref_63_, v___x_68_);
v___x_70_ = ((lean_object*)(lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__4));
v___x_71_ = lean_obj_once(&lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__6, &lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__6_once, _init_lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__6);
v___x_72_ = ((lean_object*)(lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__7));
lean_inc(v_currMacroScope_62_);
lean_inc(v_quotContext_61_);
v___x_73_ = l_Lean_addMacroScope(v_quotContext_61_, v___x_72_, v_currMacroScope_62_);
v___x_74_ = ((lean_object*)(lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__10));
lean_inc_n(v___x_69_, 2);
v___x_75_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_75_, 0, v___x_69_);
lean_ctor_set(v___x_75_, 1, v___x_71_);
lean_ctor_set(v___x_75_, 2, v___x_73_);
lean_ctor_set(v___x_75_, 3, v___x_74_);
v___x_76_ = ((lean_object*)(lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__12));
v___x_77_ = l_Lean_Syntax_node2(v___x_69_, v___x_76_, v___x_65_, v___x_67_);
v___x_78_ = l_Lean_Syntax_node2(v___x_69_, v___x_70_, v___x_75_, v___x_77_);
v___x_79_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_79_, 0, v___x_78_);
lean_ctor_set(v___x_79_, 1, v_a_56_);
return v___x_79_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___boxed(lean_object* v_x_80_, lean_object* v_a_81_, lean_object* v_a_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1(v_x_80_, v_a_81_, v_a_82_);
lean_dec_ref(v_a_81_);
return v_res_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Relator___aux__Mathlib__Logic__Relator______unexpand__Relator__LiftFun__1(lean_object* v_x_87_, lean_object* v_a_88_, lean_object* v_a_89_){
_start:
{
lean_object* v___x_90_; uint8_t v___x_91_; 
v___x_90_ = ((lean_object*)(lp_mathlib_Relator___aux__Mathlib__Logic__Relator______macroRules__Relator__term___u21d2____1___closed__4));
lean_inc(v_x_87_);
v___x_91_ = l_Lean_Syntax_isOfKind(v_x_87_, v___x_90_);
if (v___x_91_ == 0)
{
lean_object* v___x_92_; lean_object* v___x_93_; 
lean_dec(v_x_87_);
v___x_92_ = lean_box(0);
v___x_93_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_93_, 0, v___x_92_);
lean_ctor_set(v___x_93_, 1, v_a_89_);
return v___x_93_;
}
else
{
lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; uint8_t v___x_97_; 
v___x_94_ = lean_unsigned_to_nat(0u);
v___x_95_ = l_Lean_Syntax_getArg(v_x_87_, v___x_94_);
v___x_96_ = ((lean_object*)(lp_mathlib_Relator___aux__Mathlib__Logic__Relator______unexpand__Relator__LiftFun__1___closed__1));
lean_inc(v___x_95_);
v___x_97_ = l_Lean_Syntax_isOfKind(v___x_95_, v___x_96_);
if (v___x_97_ == 0)
{
lean_object* v___x_98_; lean_object* v___x_99_; 
lean_dec(v___x_95_);
lean_dec(v_x_87_);
v___x_98_ = lean_box(0);
v___x_99_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_99_, 0, v___x_98_);
lean_ctor_set(v___x_99_, 1, v_a_89_);
return v___x_99_;
}
else
{
lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; uint8_t v___x_103_; 
v___x_100_ = lean_unsigned_to_nat(1u);
v___x_101_ = l_Lean_Syntax_getArg(v_x_87_, v___x_100_);
lean_dec(v_x_87_);
v___x_102_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_101_);
v___x_103_ = l_Lean_Syntax_matchesNull(v___x_101_, v___x_102_);
if (v___x_103_ == 0)
{
lean_object* v___x_104_; lean_object* v___x_105_; 
lean_dec(v___x_101_);
lean_dec(v___x_95_);
v___x_104_ = lean_box(0);
v___x_105_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_105_, 0, v___x_104_);
lean_ctor_set(v___x_105_, 1, v_a_89_);
return v___x_105_;
}
else
{
lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v_ref_108_; uint8_t v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; 
v___x_106_ = l_Lean_Syntax_getArg(v___x_101_, v___x_94_);
v___x_107_ = l_Lean_Syntax_getArg(v___x_101_, v___x_100_);
lean_dec(v___x_101_);
v_ref_108_ = l_Lean_replaceRef(v___x_95_, v_a_88_);
lean_dec(v___x_95_);
v___x_109_ = 0;
v___x_110_ = l_Lean_SourceInfo_fromRef(v_ref_108_, v___x_109_);
lean_dec(v_ref_108_);
v___x_111_ = ((lean_object*)(lp_mathlib_Relator_term___u21d2___00__closed__2));
v___x_112_ = ((lean_object*)(lp_mathlib_Relator_term___u21d2___00__closed__5));
lean_inc(v___x_110_);
v___x_113_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_113_, 0, v___x_110_);
lean_ctor_set(v___x_113_, 1, v___x_112_);
v___x_114_ = l_Lean_Syntax_node3(v___x_110_, v___x_111_, v___x_106_, v___x_113_, v___x_107_);
v___x_115_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_115_, 0, v___x_114_);
lean_ctor_set(v___x_115_, 1, v_a_89_);
return v___x_115_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Relator___aux__Mathlib__Logic__Relator______unexpand__Relator__LiftFun__1___boxed(lean_object* v_x_116_, lean_object* v_a_117_, lean_object* v_a_118_){
_start:
{
lean_object* v_res_119_; 
v_res_119_ = lp_mathlib_Relator___aux__Mathlib__Logic__Relator______unexpand__Relator__LiftFun__1(v_x_116_, v_a_117_, v_a_118_);
lean_dec(v_a_117_);
return v_res_119_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Function_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Logic_Relator(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Function_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Logic_Relator(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Logic_Function_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Logic_Relator(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Function_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Relator(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Logic_Relator(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Logic_Relator(builtin);
}
#ifdef __cplusplus
}
#endif
