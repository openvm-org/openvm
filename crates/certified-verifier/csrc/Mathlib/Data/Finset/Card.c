// Lean compiler output
// Module: Mathlib.Data.Finset.Card
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Basic public import Mathlib.Data.Finset.Image public import Mathlib.Data.Finset.Lattice.Lemmas
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
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_card___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_card___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_card(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_card___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Finset_term_x23___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Finset"};
static const lean_object* lp_mathlib_Finset_term_x23___00__closed__0 = (const lean_object*)&lp_mathlib_Finset_term_x23___00__closed__0_value;
static const lean_string_object lp_mathlib_Finset_term_x23___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "term#_"};
static const lean_object* lp_mathlib_Finset_term_x23___00__closed__1 = (const lean_object*)&lp_mathlib_Finset_term_x23___00__closed__1_value;
static const lean_ctor_object lp_mathlib_Finset_term_x23___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_term_x23___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_ctor_object lp_mathlib_Finset_term_x23___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_term_x23___00__closed__2_value_aux_0),((lean_object*)&lp_mathlib_Finset_term_x23___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(149, 184, 206, 38, 112, 127, 88, 44)}};
static const lean_object* lp_mathlib_Finset_term_x23___00__closed__2 = (const lean_object*)&lp_mathlib_Finset_term_x23___00__closed__2_value;
static const lean_string_object lp_mathlib_Finset_term_x23___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Finset_term_x23___00__closed__3 = (const lean_object*)&lp_mathlib_Finset_term_x23___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Finset_term_x23___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_term_x23___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Finset_term_x23___00__closed__4 = (const lean_object*)&lp_mathlib_Finset_term_x23___00__closed__4_value;
static const lean_string_object lp_mathlib_Finset_term_x23___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "#"};
static const lean_object* lp_mathlib_Finset_term_x23___00__closed__5 = (const lean_object*)&lp_mathlib_Finset_term_x23___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Finset_term_x23___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Finset_term_x23___00__closed__5_value)}};
static const lean_object* lp_mathlib_Finset_term_x23___00__closed__6 = (const lean_object*)&lp_mathlib_Finset_term_x23___00__closed__6_value;
static const lean_string_object lp_mathlib_Finset_term_x23___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Finset_term_x23___00__closed__7 = (const lean_object*)&lp_mathlib_Finset_term_x23___00__closed__7_value;
static const lean_ctor_object lp_mathlib_Finset_term_x23___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_term_x23___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Finset_term_x23___00__closed__8 = (const lean_object*)&lp_mathlib_Finset_term_x23___00__closed__8_value;
static const lean_ctor_object lp_mathlib_Finset_term_x23___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Finset_term_x23___00__closed__8_value),((lean_object*)(((size_t)(1023) << 1) | 1))}};
static const lean_object* lp_mathlib_Finset_term_x23___00__closed__9 = (const lean_object*)&lp_mathlib_Finset_term_x23___00__closed__9_value;
static const lean_ctor_object lp_mathlib_Finset_term_x23___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Finset_term_x23___00__closed__4_value),((lean_object*)&lp_mathlib_Finset_term_x23___00__closed__6_value),((lean_object*)&lp_mathlib_Finset_term_x23___00__closed__9_value)}};
static const lean_object* lp_mathlib_Finset_term_x23___00__closed__10 = (const lean_object*)&lp_mathlib_Finset_term_x23___00__closed__10_value;
static const lean_ctor_object lp_mathlib_Finset_term_x23___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Finset_term_x23___00__closed__2_value),((lean_object*)(((size_t)(1023) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_term_x23___00__closed__10_value)}};
static const lean_object* lp_mathlib_Finset_term_x23___00__closed__11 = (const lean_object*)&lp_mathlib_Finset_term_x23___00__closed__11_value;
LEAN_EXPORT const lean_object* lp_mathlib_Finset_term_x23__ = (const lean_object*)&lp_mathlib_Finset_term_x23___00__closed__11_value;
static const lean_string_object lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__0 = (const lean_object*)&lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__0_value;
static const lean_string_object lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__1 = (const lean_object*)&lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__1_value;
static const lean_string_object lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__2 = (const lean_object*)&lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__2_value;
static const lean_string_object lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__3 = (const lean_object*)&lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__4 = (const lean_object*)&lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__4_value;
static const lean_string_object lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Finset.card"};
static const lean_object* lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__5 = (const lean_object*)&lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__5_value;
static lean_once_cell_t lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__6;
static const lean_string_object lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "card"};
static const lean_object* lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__7 = (const lean_object*)&lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__7_value;
static const lean_ctor_object lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_term_x23___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_ctor_object lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(192, 121, 103, 146, 194, 87, 130, 177)}};
static const lean_object* lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__8 = (const lean_object*)&lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__8_value;
static const lean_ctor_object lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__9 = (const lean_object*)&lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__9_value;
static const lean_ctor_object lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__10 = (const lean_object*)&lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__10_value;
static const lean_string_object lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__11 = (const lean_object*)&lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__11_value;
static const lean_ctor_object lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__12 = (const lean_object*)&lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______unexpand__Finset__card__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______unexpand__Finset__card__1___closed__0 = (const lean_object*)&lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______unexpand__Finset__card__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______unexpand__Finset__card__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______unexpand__Finset__card__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______unexpand__Finset__card__1___closed__1 = (const lean_object*)&lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______unexpand__Finset__card__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______unexpand__Finset__card__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______unexpand__Finset__card__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_strongInduction___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_strongInduction___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_strongInduction(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_strongInductionOn___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_strongInductionOn(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_strongDownwardInduction___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_strongDownwardInduction___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_strongDownwardInduction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_strongDownwardInduction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_strongDownwardInductionOn___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_strongDownwardInductionOn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_strongDownwardInductionOn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_card___redArg(lean_object* v_s_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = l_List_lengthTR___redArg(v_s_1_);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_card___redArg___boxed(lean_object* v_s_3_){
_start:
{
lean_object* v_res_4_; 
v_res_4_ = lp_mathlib_Finset_card___redArg(v_s_3_);
lean_dec(v_s_3_);
return v_res_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_card(lean_object* v_00_u03b1_5_, lean_object* v_s_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = l_List_lengthTR___redArg(v_s_6_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_card___boxed(lean_object* v_00_u03b1_8_, lean_object* v_s_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_mathlib_Finset_card(v_00_u03b1_8_, v_s_9_);
lean_dec(v_s_9_);
return v_res_10_;
}
}
static lean_object* _init_lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__6(void){
_start:
{
lean_object* v___x_47_; lean_object* v___x_48_; 
v___x_47_ = ((lean_object*)(lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__5));
v___x_48_ = l_String_toRawSubstring_x27(v___x_47_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1(lean_object* v_x_62_, lean_object* v_a_63_, lean_object* v_a_64_){
_start:
{
lean_object* v___x_65_; uint8_t v___x_66_; 
v___x_65_ = ((lean_object*)(lp_mathlib_Finset_term_x23___00__closed__2));
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
lean_object* v_quotContext_69_; lean_object* v_currMacroScope_70_; lean_object* v_ref_71_; lean_object* v___x_72_; lean_object* v___x_73_; uint8_t v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; 
v_quotContext_69_ = lean_ctor_get(v_a_63_, 1);
v_currMacroScope_70_ = lean_ctor_get(v_a_63_, 2);
v_ref_71_ = lean_ctor_get(v_a_63_, 5);
v___x_72_ = lean_unsigned_to_nat(1u);
v___x_73_ = l_Lean_Syntax_getArg(v_x_62_, v___x_72_);
lean_dec(v_x_62_);
v___x_74_ = 0;
v___x_75_ = l_Lean_SourceInfo_fromRef(v_ref_71_, v___x_74_);
v___x_76_ = ((lean_object*)(lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__4));
v___x_77_ = lean_obj_once(&lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__6, &lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__6_once, _init_lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__6);
v___x_78_ = ((lean_object*)(lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__8));
lean_inc(v_currMacroScope_70_);
lean_inc(v_quotContext_69_);
v___x_79_ = l_Lean_addMacroScope(v_quotContext_69_, v___x_78_, v_currMacroScope_70_);
v___x_80_ = ((lean_object*)(lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__10));
lean_inc_n(v___x_75_, 2);
v___x_81_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_81_, 0, v___x_75_);
lean_ctor_set(v___x_81_, 1, v___x_77_);
lean_ctor_set(v___x_81_, 2, v___x_79_);
lean_ctor_set(v___x_81_, 3, v___x_80_);
v___x_82_ = ((lean_object*)(lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__12));
v___x_83_ = l_Lean_Syntax_node1(v___x_75_, v___x_82_, v___x_73_);
v___x_84_ = l_Lean_Syntax_node2(v___x_75_, v___x_76_, v___x_81_, v___x_83_);
v___x_85_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_85_, 0, v___x_84_);
lean_ctor_set(v___x_85_, 1, v_a_64_);
return v___x_85_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___boxed(lean_object* v_x_86_, lean_object* v_a_87_, lean_object* v_a_88_){
_start:
{
lean_object* v_res_89_; 
v_res_89_ = lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1(v_x_86_, v_a_87_, v_a_88_);
lean_dec_ref(v_a_87_);
return v_res_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______unexpand__Finset__card__1(lean_object* v_x_93_, lean_object* v_a_94_, lean_object* v_a_95_){
_start:
{
lean_object* v___x_96_; uint8_t v___x_97_; 
v___x_96_ = ((lean_object*)(lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______macroRules__Finset__term_x23____1___closed__4));
lean_inc(v_x_93_);
v___x_97_ = l_Lean_Syntax_isOfKind(v_x_93_, v___x_96_);
if (v___x_97_ == 0)
{
lean_object* v___x_98_; lean_object* v___x_99_; 
lean_dec(v_x_93_);
v___x_98_ = lean_box(0);
v___x_99_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_99_, 0, v___x_98_);
lean_ctor_set(v___x_99_, 1, v_a_95_);
return v___x_99_;
}
else
{
lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; uint8_t v___x_103_; 
v___x_100_ = lean_unsigned_to_nat(0u);
v___x_101_ = l_Lean_Syntax_getArg(v_x_93_, v___x_100_);
v___x_102_ = ((lean_object*)(lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______unexpand__Finset__card__1___closed__1));
lean_inc(v___x_101_);
v___x_103_ = l_Lean_Syntax_isOfKind(v___x_101_, v___x_102_);
if (v___x_103_ == 0)
{
lean_object* v___x_104_; lean_object* v___x_105_; 
lean_dec(v___x_101_);
lean_dec(v_x_93_);
v___x_104_ = lean_box(0);
v___x_105_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_105_, 0, v___x_104_);
lean_ctor_set(v___x_105_, 1, v_a_95_);
return v___x_105_;
}
else
{
lean_object* v___x_106_; lean_object* v___x_107_; uint8_t v___x_108_; 
v___x_106_ = lean_unsigned_to_nat(1u);
v___x_107_ = l_Lean_Syntax_getArg(v_x_93_, v___x_106_);
lean_dec(v_x_93_);
lean_inc(v___x_107_);
v___x_108_ = l_Lean_Syntax_matchesNull(v___x_107_, v___x_106_);
if (v___x_108_ == 0)
{
lean_object* v___x_109_; lean_object* v___x_110_; 
lean_dec(v___x_107_);
lean_dec(v___x_101_);
v___x_109_ = lean_box(0);
v___x_110_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_110_, 0, v___x_109_);
lean_ctor_set(v___x_110_, 1, v_a_95_);
return v___x_110_;
}
else
{
lean_object* v___x_111_; lean_object* v_ref_112_; uint8_t v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; 
v___x_111_ = l_Lean_Syntax_getArg(v___x_107_, v___x_100_);
lean_dec(v___x_107_);
v_ref_112_ = l_Lean_replaceRef(v___x_101_, v_a_94_);
lean_dec(v___x_101_);
v___x_113_ = 0;
v___x_114_ = l_Lean_SourceInfo_fromRef(v_ref_112_, v___x_113_);
lean_dec(v_ref_112_);
v___x_115_ = ((lean_object*)(lp_mathlib_Finset_term_x23___00__closed__2));
v___x_116_ = ((lean_object*)(lp_mathlib_Finset_term_x23___00__closed__5));
lean_inc(v___x_114_);
v___x_117_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_117_, 0, v___x_114_);
lean_ctor_set(v___x_117_, 1, v___x_116_);
v___x_118_ = l_Lean_Syntax_node2(v___x_114_, v___x_115_, v___x_117_, v___x_111_);
v___x_119_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_119_, 0, v___x_118_);
lean_ctor_set(v___x_119_, 1, v_a_95_);
return v___x_119_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______unexpand__Finset__card__1___boxed(lean_object* v_x_120_, lean_object* v_a_121_, lean_object* v_a_122_){
_start:
{
lean_object* v_res_123_; 
v_res_123_ = lp_mathlib_Finset___aux__Mathlib__Data__Finset__Card______unexpand__Finset__card__1(v_x_120_, v_a_121_, v_a_122_);
lean_dec(v_a_121_);
return v_res_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_strongInduction___redArg(lean_object* v_H_124_, lean_object* v_x_125_){
_start:
{
lean_object* v___f_126_; lean_object* v___x_127_; 
lean_inc(v_H_124_);
v___f_126_ = lean_alloc_closure((void*)(lp_mathlib_Finset_strongInduction___redArg___lam__0), 3, 1);
lean_closure_set(v___f_126_, 0, v_H_124_);
v___x_127_ = lean_apply_2(v_H_124_, v_x_125_, v___f_126_);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_strongInduction___redArg___lam__0(lean_object* v_H_128_, lean_object* v_t_129_, lean_object* v_h_130_){
_start:
{
lean_object* v___x_131_; 
v___x_131_ = lp_mathlib_Finset_strongInduction___redArg(v_H_128_, v_t_129_);
return v___x_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_strongInduction(lean_object* v_00_u03b1_132_, lean_object* v_p_133_, lean_object* v_H_134_, lean_object* v_x_135_){
_start:
{
lean_object* v___x_136_; 
v___x_136_ = lp_mathlib_Finset_strongInduction___redArg(v_H_134_, v_x_135_);
return v___x_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_strongInductionOn___redArg(lean_object* v_s_137_, lean_object* v_H_138_){
_start:
{
lean_object* v___x_139_; 
v___x_139_ = lp_mathlib_Finset_strongInduction___redArg(v_H_138_, v_s_137_);
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_strongInductionOn(lean_object* v_00_u03b1_140_, lean_object* v_p_141_, lean_object* v_s_142_, lean_object* v_H_143_){
_start:
{
lean_object* v___x_144_; 
v___x_144_ = lp_mathlib_Finset_strongInduction___redArg(v_H_143_, v_s_142_);
return v___x_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_strongDownwardInduction___redArg(lean_object* v_H_145_, lean_object* v_x_146_){
_start:
{
lean_object* v___f_147_; lean_object* v___x_148_; 
lean_inc(v_H_145_);
v___f_147_ = lean_alloc_closure((void*)(lp_mathlib_Finset_strongDownwardInduction___redArg___lam__0), 4, 1);
lean_closure_set(v___f_147_, 0, v_H_145_);
v___x_148_ = lean_apply_3(v_H_145_, v_x_146_, v___f_147_, lean_box(0));
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_strongDownwardInduction___redArg___lam__0(lean_object* v_H_149_, lean_object* v_t_150_, lean_object* v_ht_151_, lean_object* v_h_152_){
_start:
{
lean_object* v___x_153_; 
v___x_153_ = lp_mathlib_Finset_strongDownwardInduction___redArg(v_H_149_, v_t_150_);
return v___x_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_strongDownwardInduction(lean_object* v_00_u03b1_154_, lean_object* v_p_155_, lean_object* v_n_156_, lean_object* v_H_157_, lean_object* v_x_158_, lean_object* v_a_159_){
_start:
{
lean_object* v___x_160_; 
v___x_160_ = lp_mathlib_Finset_strongDownwardInduction___redArg(v_H_157_, v_x_158_);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_strongDownwardInduction___boxed(lean_object* v_00_u03b1_161_, lean_object* v_p_162_, lean_object* v_n_163_, lean_object* v_H_164_, lean_object* v_x_165_, lean_object* v_a_166_){
_start:
{
lean_object* v_res_167_; 
v_res_167_ = lp_mathlib_Finset_strongDownwardInduction(v_00_u03b1_161_, v_p_162_, v_n_163_, v_H_164_, v_x_165_, v_a_166_);
lean_dec(v_n_163_);
return v_res_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_strongDownwardInductionOn___redArg(lean_object* v_s_168_, lean_object* v_H_169_){
_start:
{
lean_object* v___x_170_; 
v___x_170_ = lp_mathlib_Finset_strongDownwardInduction___redArg(v_H_169_, v_s_168_);
return v___x_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_strongDownwardInductionOn(lean_object* v_00_u03b1_171_, lean_object* v_n_172_, lean_object* v_p_173_, lean_object* v_s_174_, lean_object* v_H_175_, lean_object* v_a_176_){
_start:
{
lean_object* v___x_177_; 
v___x_177_ = lp_mathlib_Finset_strongDownwardInduction___redArg(v_H_175_, v_s_174_);
return v___x_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_strongDownwardInductionOn___boxed(lean_object* v_00_u03b1_178_, lean_object* v_n_179_, lean_object* v_p_180_, lean_object* v_s_181_, lean_object* v_H_182_, lean_object* v_a_183_){
_start:
{
lean_object* v_res_184_; 
v_res_184_ = lp_mathlib_Finset_strongDownwardInductionOn(v_00_u03b1_178_, v_n_179_, v_p_180_, v_s_181_, v_H_182_, v_a_183_);
lean_dec(v_n_179_);
return v_res_184_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Image(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Lemmas(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Card(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finset_Card(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Image(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Lattice_Lemmas(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_Card(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Lattice_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finset_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finset_Card(builtin);
}
#ifdef __cplusplus
}
#endif
