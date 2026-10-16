// Lean compiler output
// Module: Mathlib.Data.Set.Operations
// Imports: public import Init public meta import Init public import Mathlib.Data.Set.CoeSort public import Mathlib.Data.SProd public import Mathlib.Data.Subtype public import Mathlib.Order.Notation public import Mathlib.Tactic.CrossRefAttribute public import Mathlib.Tactic.Push.Attr import Aesop.BuiltinRules import Aesop.Frontend.Tactic import Aesop.Main
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instCompl(lean_object*);
static const lean_string_object lp_mathlib_Set_term___u207b_xb9_x27___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Set"};
static const lean_object* lp_mathlib_Set_term___u207b_xb9_x27___00__closed__0 = (const lean_object*)&lp_mathlib_Set_term___u207b_xb9_x27___00__closed__0_value;
static const lean_string_object lp_mathlib_Set_term___u207b_xb9_x27___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 9, .m_data = "term_⁻¹'_"};
static const lean_object* lp_mathlib_Set_term___u207b_xb9_x27___00__closed__1 = (const lean_object*)&lp_mathlib_Set_term___u207b_xb9_x27___00__closed__1_value;
static const lean_ctor_object lp_mathlib_Set_term___u207b_xb9_x27___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_term___u207b_xb9_x27___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_ctor_object lp_mathlib_Set_term___u207b_xb9_x27___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_term___u207b_xb9_x27___00__closed__2_value_aux_0),((lean_object*)&lp_mathlib_Set_term___u207b_xb9_x27___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(78, 198, 251, 91, 103, 37, 163, 106)}};
static const lean_object* lp_mathlib_Set_term___u207b_xb9_x27___00__closed__2 = (const lean_object*)&lp_mathlib_Set_term___u207b_xb9_x27___00__closed__2_value;
static const lean_string_object lp_mathlib_Set_term___u207b_xb9_x27___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Set_term___u207b_xb9_x27___00__closed__3 = (const lean_object*)&lp_mathlib_Set_term___u207b_xb9_x27___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Set_term___u207b_xb9_x27___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_term___u207b_xb9_x27___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Set_term___u207b_xb9_x27___00__closed__4 = (const lean_object*)&lp_mathlib_Set_term___u207b_xb9_x27___00__closed__4_value;
static const lean_string_object lp_mathlib_Set_term___u207b_xb9_x27___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 5, .m_data = " ⁻¹' "};
static const lean_object* lp_mathlib_Set_term___u207b_xb9_x27___00__closed__5 = (const lean_object*)&lp_mathlib_Set_term___u207b_xb9_x27___00__closed__5_value;
static const lean_ctor_object lp_mathlib_Set_term___u207b_xb9_x27___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Set_term___u207b_xb9_x27___00__closed__5_value)}};
static const lean_object* lp_mathlib_Set_term___u207b_xb9_x27___00__closed__6 = (const lean_object*)&lp_mathlib_Set_term___u207b_xb9_x27___00__closed__6_value;
static const lean_string_object lp_mathlib_Set_term___u207b_xb9_x27___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Set_term___u207b_xb9_x27___00__closed__7 = (const lean_object*)&lp_mathlib_Set_term___u207b_xb9_x27___00__closed__7_value;
static const lean_ctor_object lp_mathlib_Set_term___u207b_xb9_x27___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_term___u207b_xb9_x27___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Set_term___u207b_xb9_x27___00__closed__8 = (const lean_object*)&lp_mathlib_Set_term___u207b_xb9_x27___00__closed__8_value;
static const lean_ctor_object lp_mathlib_Set_term___u207b_xb9_x27___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Set_term___u207b_xb9_x27___00__closed__8_value),((lean_object*)(((size_t)(80) << 1) | 1))}};
static const lean_object* lp_mathlib_Set_term___u207b_xb9_x27___00__closed__9 = (const lean_object*)&lp_mathlib_Set_term___u207b_xb9_x27___00__closed__9_value;
static const lean_ctor_object lp_mathlib_Set_term___u207b_xb9_x27___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Set_term___u207b_xb9_x27___00__closed__4_value),((lean_object*)&lp_mathlib_Set_term___u207b_xb9_x27___00__closed__6_value),((lean_object*)&lp_mathlib_Set_term___u207b_xb9_x27___00__closed__9_value)}};
static const lean_object* lp_mathlib_Set_term___u207b_xb9_x27___00__closed__10 = (const lean_object*)&lp_mathlib_Set_term___u207b_xb9_x27___00__closed__10_value;
static const lean_ctor_object lp_mathlib_Set_term___u207b_xb9_x27___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_Set_term___u207b_xb9_x27___00__closed__2_value),((lean_object*)(((size_t)(80) << 1) | 1)),((lean_object*)(((size_t)(81) << 1) | 1)),((lean_object*)&lp_mathlib_Set_term___u207b_xb9_x27___00__closed__10_value)}};
static const lean_object* lp_mathlib_Set_term___u207b_xb9_x27___00__closed__11 = (const lean_object*)&lp_mathlib_Set_term___u207b_xb9_x27___00__closed__11_value;
LEAN_EXPORT const lean_object* lp_mathlib_Set_term___u207b_xb9_x27__ = (const lean_object*)&lp_mathlib_Set_term___u207b_xb9_x27___00__closed__11_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__0 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__0_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__1 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__1_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__2 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__2_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__3 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__4 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__4_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "preimage"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__5 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__5_value;
static lean_once_cell_t lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__6;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(93, 106, 4, 23, 218, 187, 126, 198)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__7 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__7_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_term___u207b_xb9_x27___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(228, 75, 132, 28, 148, 34, 22, 147)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__8 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__8_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__9 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__9_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__10 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__10_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__11 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__11_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__12 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______unexpand__Set__preimage__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______unexpand__Set__preimage__1___closed__0 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______unexpand__Set__preimage__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______unexpand__Set__preimage__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______unexpand__Set__preimage__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______unexpand__Set__preimage__1___closed__1 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______unexpand__Set__preimage__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______unexpand__Set__preimage__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______unexpand__Set__preimage__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Set_term___x27_x27___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "term_''_"};
static const lean_object* lp_mathlib_Set_term___x27_x27___00__closed__0 = (const lean_object*)&lp_mathlib_Set_term___x27_x27___00__closed__0_value;
static const lean_ctor_object lp_mathlib_Set_term___x27_x27___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_term___u207b_xb9_x27___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_ctor_object lp_mathlib_Set_term___x27_x27___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_term___x27_x27___00__closed__1_value_aux_0),((lean_object*)&lp_mathlib_Set_term___x27_x27___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(20, 19, 7, 231, 238, 117, 253, 168)}};
static const lean_object* lp_mathlib_Set_term___x27_x27___00__closed__1 = (const lean_object*)&lp_mathlib_Set_term___x27_x27___00__closed__1_value;
static const lean_string_object lp_mathlib_Set_term___x27_x27___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " '' "};
static const lean_object* lp_mathlib_Set_term___x27_x27___00__closed__2 = (const lean_object*)&lp_mathlib_Set_term___x27_x27___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Set_term___x27_x27___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Set_term___x27_x27___00__closed__2_value)}};
static const lean_object* lp_mathlib_Set_term___x27_x27___00__closed__3 = (const lean_object*)&lp_mathlib_Set_term___x27_x27___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Set_term___x27_x27___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Set_term___u207b_xb9_x27___00__closed__4_value),((lean_object*)&lp_mathlib_Set_term___x27_x27___00__closed__3_value),((lean_object*)&lp_mathlib_Set_term___u207b_xb9_x27___00__closed__9_value)}};
static const lean_object* lp_mathlib_Set_term___x27_x27___00__closed__4 = (const lean_object*)&lp_mathlib_Set_term___x27_x27___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Set_term___x27_x27___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_Set_term___x27_x27___00__closed__1_value),((lean_object*)(((size_t)(80) << 1) | 1)),((lean_object*)(((size_t)(81) << 1) | 1)),((lean_object*)&lp_mathlib_Set_term___x27_x27___00__closed__4_value)}};
static const lean_object* lp_mathlib_Set_term___x27_x27___00__closed__5 = (const lean_object*)&lp_mathlib_Set_term___x27_x27___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Set_term___x27_x27__ = (const lean_object*)&lp_mathlib_Set_term___x27_x27___00__closed__5_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "image"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___closed__0 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___closed__0_value;
static lean_once_cell_t lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___closed__1;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(122, 26, 231, 33, 57, 209, 255, 167)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___closed__2 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___closed__2_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_term___u207b_xb9_x27___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(27, 176, 66, 9, 14, 111, 229, 187)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___closed__3 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___closed__3_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___closed__4 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___closed__4_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___closed__5 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______unexpand__Set__image__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______unexpand__Set__image__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_imageFactorization___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_imageFactorization(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_rangeFactorization___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_rangeFactorization(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instSProd(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_MapsTo_restrict___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_MapsTo_restrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_restrictPreimage___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_restrictPreimage(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instCompl(lean_object* v_00_u03b1_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_box(0);
return v___x_2_;
}
}
static lean_object* _init_lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__6(void){
_start:
{
lean_object* v___x_40_; lean_object* v___x_41_; 
v___x_40_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__5));
v___x_41_ = l_String_toRawSubstring_x27(v___x_40_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1(lean_object* v_x_56_, lean_object* v_a_57_, lean_object* v_a_58_){
_start:
{
lean_object* v___x_59_; uint8_t v___x_60_; 
v___x_59_ = ((lean_object*)(lp_mathlib_Set_term___u207b_xb9_x27___00__closed__2));
lean_inc(v_x_56_);
v___x_60_ = l_Lean_Syntax_isOfKind(v_x_56_, v___x_59_);
if (v___x_60_ == 0)
{
lean_object* v___x_61_; lean_object* v___x_62_; 
lean_dec(v_x_56_);
v___x_61_ = lean_box(1);
v___x_62_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_62_, 0, v___x_61_);
lean_ctor_set(v___x_62_, 1, v_a_58_);
return v___x_62_;
}
else
{
lean_object* v_quotContext_63_; lean_object* v_currMacroScope_64_; lean_object* v_ref_65_; lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; uint8_t v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; 
v_quotContext_63_ = lean_ctor_get(v_a_57_, 1);
v_currMacroScope_64_ = lean_ctor_get(v_a_57_, 2);
v_ref_65_ = lean_ctor_get(v_a_57_, 5);
v___x_66_ = lean_unsigned_to_nat(0u);
v___x_67_ = l_Lean_Syntax_getArg(v_x_56_, v___x_66_);
v___x_68_ = lean_unsigned_to_nat(2u);
v___x_69_ = l_Lean_Syntax_getArg(v_x_56_, v___x_68_);
lean_dec(v_x_56_);
v___x_70_ = 0;
v___x_71_ = l_Lean_SourceInfo_fromRef(v_ref_65_, v___x_70_);
v___x_72_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__4));
v___x_73_ = lean_obj_once(&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__6, &lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__6_once, _init_lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__6);
v___x_74_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__7));
lean_inc(v_currMacroScope_64_);
lean_inc(v_quotContext_63_);
v___x_75_ = l_Lean_addMacroScope(v_quotContext_63_, v___x_74_, v_currMacroScope_64_);
v___x_76_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__10));
lean_inc_n(v___x_71_, 2);
v___x_77_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_77_, 0, v___x_71_);
lean_ctor_set(v___x_77_, 1, v___x_73_);
lean_ctor_set(v___x_77_, 2, v___x_75_);
lean_ctor_set(v___x_77_, 3, v___x_76_);
v___x_78_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__12));
v___x_79_ = l_Lean_Syntax_node2(v___x_71_, v___x_78_, v___x_67_, v___x_69_);
v___x_80_ = l_Lean_Syntax_node2(v___x_71_, v___x_72_, v___x_77_, v___x_79_);
v___x_81_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_81_, 0, v___x_80_);
lean_ctor_set(v___x_81_, 1, v_a_58_);
return v___x_81_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___boxed(lean_object* v_x_82_, lean_object* v_a_83_, lean_object* v_a_84_){
_start:
{
lean_object* v_res_85_; 
v_res_85_ = lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1(v_x_82_, v_a_83_, v_a_84_);
lean_dec_ref(v_a_83_);
return v_res_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______unexpand__Set__preimage__1(lean_object* v_x_89_, lean_object* v_a_90_, lean_object* v_a_91_){
_start:
{
lean_object* v___x_92_; uint8_t v___x_93_; 
v___x_92_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__4));
lean_inc(v_x_89_);
v___x_93_ = l_Lean_Syntax_isOfKind(v_x_89_, v___x_92_);
if (v___x_93_ == 0)
{
lean_object* v___x_94_; lean_object* v___x_95_; 
lean_dec(v_x_89_);
v___x_94_ = lean_box(0);
v___x_95_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_95_, 0, v___x_94_);
lean_ctor_set(v___x_95_, 1, v_a_91_);
return v___x_95_;
}
else
{
lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; uint8_t v___x_99_; 
v___x_96_ = lean_unsigned_to_nat(0u);
v___x_97_ = l_Lean_Syntax_getArg(v_x_89_, v___x_96_);
v___x_98_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______unexpand__Set__preimage__1___closed__1));
lean_inc(v___x_97_);
v___x_99_ = l_Lean_Syntax_isOfKind(v___x_97_, v___x_98_);
if (v___x_99_ == 0)
{
lean_object* v___x_100_; lean_object* v___x_101_; 
lean_dec(v___x_97_);
lean_dec(v_x_89_);
v___x_100_ = lean_box(0);
v___x_101_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_101_, 0, v___x_100_);
lean_ctor_set(v___x_101_, 1, v_a_91_);
return v___x_101_;
}
else
{
lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; uint8_t v___x_105_; 
v___x_102_ = lean_unsigned_to_nat(1u);
v___x_103_ = l_Lean_Syntax_getArg(v_x_89_, v___x_102_);
lean_dec(v_x_89_);
v___x_104_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_103_);
v___x_105_ = l_Lean_Syntax_matchesNull(v___x_103_, v___x_104_);
if (v___x_105_ == 0)
{
lean_object* v___x_106_; lean_object* v___x_107_; 
lean_dec(v___x_103_);
lean_dec(v___x_97_);
v___x_106_ = lean_box(0);
v___x_107_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_107_, 0, v___x_106_);
lean_ctor_set(v___x_107_, 1, v_a_91_);
return v___x_107_;
}
else
{
lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v_ref_110_; uint8_t v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; 
v___x_108_ = l_Lean_Syntax_getArg(v___x_103_, v___x_96_);
v___x_109_ = l_Lean_Syntax_getArg(v___x_103_, v___x_102_);
lean_dec(v___x_103_);
v_ref_110_ = l_Lean_replaceRef(v___x_97_, v_a_90_);
lean_dec(v___x_97_);
v___x_111_ = 0;
v___x_112_ = l_Lean_SourceInfo_fromRef(v_ref_110_, v___x_111_);
lean_dec(v_ref_110_);
v___x_113_ = ((lean_object*)(lp_mathlib_Set_term___u207b_xb9_x27___00__closed__2));
v___x_114_ = ((lean_object*)(lp_mathlib_Set_term___u207b_xb9_x27___00__closed__5));
lean_inc(v___x_112_);
v___x_115_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_115_, 0, v___x_112_);
lean_ctor_set(v___x_115_, 1, v___x_114_);
v___x_116_ = l_Lean_Syntax_node3(v___x_112_, v___x_113_, v___x_108_, v___x_115_, v___x_109_);
v___x_117_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_117_, 0, v___x_116_);
lean_ctor_set(v___x_117_, 1, v_a_91_);
return v___x_117_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______unexpand__Set__preimage__1___boxed(lean_object* v_x_118_, lean_object* v_a_119_, lean_object* v_a_120_){
_start:
{
lean_object* v_res_121_; 
v_res_121_ = lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______unexpand__Set__preimage__1(v_x_118_, v_a_119_, v_a_120_);
lean_dec(v_a_119_);
return v_res_121_;
}
}
static lean_object* _init_lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___closed__1(void){
_start:
{
lean_object* v___x_140_; lean_object* v___x_141_; 
v___x_140_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___closed__0));
v___x_141_ = l_String_toRawSubstring_x27(v___x_140_);
return v___x_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1(lean_object* v_x_153_, lean_object* v_a_154_, lean_object* v_a_155_){
_start:
{
lean_object* v___x_156_; uint8_t v___x_157_; 
v___x_156_ = ((lean_object*)(lp_mathlib_Set_term___x27_x27___00__closed__1));
lean_inc(v_x_153_);
v___x_157_ = l_Lean_Syntax_isOfKind(v_x_153_, v___x_156_);
if (v___x_157_ == 0)
{
lean_object* v___x_158_; lean_object* v___x_159_; 
lean_dec(v_x_153_);
v___x_158_ = lean_box(1);
v___x_159_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_159_, 0, v___x_158_);
lean_ctor_set(v___x_159_, 1, v_a_155_);
return v___x_159_;
}
else
{
lean_object* v_quotContext_160_; lean_object* v_currMacroScope_161_; lean_object* v_ref_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; uint8_t v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; 
v_quotContext_160_ = lean_ctor_get(v_a_154_, 1);
v_currMacroScope_161_ = lean_ctor_get(v_a_154_, 2);
v_ref_162_ = lean_ctor_get(v_a_154_, 5);
v___x_163_ = lean_unsigned_to_nat(0u);
v___x_164_ = l_Lean_Syntax_getArg(v_x_153_, v___x_163_);
v___x_165_ = lean_unsigned_to_nat(2u);
v___x_166_ = l_Lean_Syntax_getArg(v_x_153_, v___x_165_);
lean_dec(v_x_153_);
v___x_167_ = 0;
v___x_168_ = l_Lean_SourceInfo_fromRef(v_ref_162_, v___x_167_);
v___x_169_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__4));
v___x_170_ = lean_obj_once(&lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___closed__1, &lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___closed__1_once, _init_lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___closed__1);
v___x_171_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___closed__2));
lean_inc(v_currMacroScope_161_);
lean_inc(v_quotContext_160_);
v___x_172_ = l_Lean_addMacroScope(v_quotContext_160_, v___x_171_, v_currMacroScope_161_);
v___x_173_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___closed__5));
lean_inc_n(v___x_168_, 2);
v___x_174_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_174_, 0, v___x_168_);
lean_ctor_set(v___x_174_, 1, v___x_170_);
lean_ctor_set(v___x_174_, 2, v___x_172_);
lean_ctor_set(v___x_174_, 3, v___x_173_);
v___x_175_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__12));
v___x_176_ = l_Lean_Syntax_node2(v___x_168_, v___x_175_, v___x_164_, v___x_166_);
v___x_177_ = l_Lean_Syntax_node2(v___x_168_, v___x_169_, v___x_174_, v___x_176_);
v___x_178_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_178_, 0, v___x_177_);
lean_ctor_set(v___x_178_, 1, v_a_155_);
return v___x_178_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1___boxed(lean_object* v_x_179_, lean_object* v_a_180_, lean_object* v_a_181_){
_start:
{
lean_object* v_res_182_; 
v_res_182_ = lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___x27_x27____1(v_x_179_, v_a_180_, v_a_181_);
lean_dec_ref(v_a_180_);
return v_res_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______unexpand__Set__image__1(lean_object* v_x_183_, lean_object* v_a_184_, lean_object* v_a_185_){
_start:
{
lean_object* v___x_186_; uint8_t v___x_187_; 
v___x_186_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______macroRules__Set__term___u207b_xb9_x27____1___closed__4));
lean_inc(v_x_183_);
v___x_187_ = l_Lean_Syntax_isOfKind(v_x_183_, v___x_186_);
if (v___x_187_ == 0)
{
lean_object* v___x_188_; lean_object* v___x_189_; 
lean_dec(v_x_183_);
v___x_188_ = lean_box(0);
v___x_189_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_189_, 0, v___x_188_);
lean_ctor_set(v___x_189_, 1, v_a_185_);
return v___x_189_;
}
else
{
lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; uint8_t v___x_193_; 
v___x_190_ = lean_unsigned_to_nat(0u);
v___x_191_ = l_Lean_Syntax_getArg(v_x_183_, v___x_190_);
v___x_192_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______unexpand__Set__preimage__1___closed__1));
lean_inc(v___x_191_);
v___x_193_ = l_Lean_Syntax_isOfKind(v___x_191_, v___x_192_);
if (v___x_193_ == 0)
{
lean_object* v___x_194_; lean_object* v___x_195_; 
lean_dec(v___x_191_);
lean_dec(v_x_183_);
v___x_194_ = lean_box(0);
v___x_195_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_195_, 0, v___x_194_);
lean_ctor_set(v___x_195_, 1, v_a_185_);
return v___x_195_;
}
else
{
lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; uint8_t v___x_199_; 
v___x_196_ = lean_unsigned_to_nat(1u);
v___x_197_ = l_Lean_Syntax_getArg(v_x_183_, v___x_196_);
lean_dec(v_x_183_);
v___x_198_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_197_);
v___x_199_ = l_Lean_Syntax_matchesNull(v___x_197_, v___x_198_);
if (v___x_199_ == 0)
{
lean_object* v___x_200_; lean_object* v___x_201_; 
lean_dec(v___x_197_);
lean_dec(v___x_191_);
v___x_200_ = lean_box(0);
v___x_201_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_201_, 0, v___x_200_);
lean_ctor_set(v___x_201_, 1, v_a_185_);
return v___x_201_;
}
else
{
lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v_ref_204_; uint8_t v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; 
v___x_202_ = l_Lean_Syntax_getArg(v___x_197_, v___x_190_);
v___x_203_ = l_Lean_Syntax_getArg(v___x_197_, v___x_196_);
lean_dec(v___x_197_);
v_ref_204_ = l_Lean_replaceRef(v___x_191_, v_a_184_);
lean_dec(v___x_191_);
v___x_205_ = 0;
v___x_206_ = l_Lean_SourceInfo_fromRef(v_ref_204_, v___x_205_);
lean_dec(v_ref_204_);
v___x_207_ = ((lean_object*)(lp_mathlib_Set_term___x27_x27___00__closed__1));
v___x_208_ = ((lean_object*)(lp_mathlib_Set_term___x27_x27___00__closed__2));
lean_inc(v___x_206_);
v___x_209_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_209_, 0, v___x_206_);
lean_ctor_set(v___x_209_, 1, v___x_208_);
v___x_210_ = l_Lean_Syntax_node3(v___x_206_, v___x_207_, v___x_202_, v___x_209_, v___x_203_);
v___x_211_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_211_, 0, v___x_210_);
lean_ctor_set(v___x_211_, 1, v_a_185_);
return v___x_211_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______unexpand__Set__image__1___boxed(lean_object* v_x_212_, lean_object* v_a_213_, lean_object* v_a_214_){
_start:
{
lean_object* v_res_215_; 
v_res_215_ = lp_mathlib_Set___aux__Mathlib__Data__Set__Operations______unexpand__Set__image__1(v_x_212_, v_a_213_, v_a_214_);
lean_dec(v_a_213_);
return v_res_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_imageFactorization___redArg(lean_object* v_f_216_, lean_object* v_p_217_){
_start:
{
lean_object* v___x_218_; 
v___x_218_ = lean_apply_1(v_f_216_, v_p_217_);
return v___x_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_imageFactorization(lean_object* v_00_u03b1_219_, lean_object* v_00_u03b2_220_, lean_object* v_f_221_, lean_object* v_s_222_, lean_object* v_p_223_){
_start:
{
lean_object* v___x_224_; 
v___x_224_ = lean_apply_1(v_f_221_, v_p_223_);
return v___x_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_rangeFactorization___redArg(lean_object* v_f_225_, lean_object* v_i_226_){
_start:
{
lean_object* v___x_227_; 
v___x_227_ = lean_apply_1(v_f_225_, v_i_226_);
return v___x_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_rangeFactorization(lean_object* v_00_u03b1_228_, lean_object* v_00_u03b9_229_, lean_object* v_f_230_, lean_object* v_i_231_){
_start:
{
lean_object* v___x_232_; 
v___x_232_ = lean_apply_1(v_f_230_, v_i_231_);
return v___x_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instSProd(lean_object* v_00_u03b1_233_, lean_object* v_00_u03b2_234_){
_start:
{
lean_object* v___x_235_; 
v___x_235_ = lean_box(0);
return v___x_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_MapsTo_restrict___redArg(lean_object* v_f_236_, lean_object* v_a_237_){
_start:
{
lean_object* v___x_238_; 
v___x_238_ = lean_apply_1(v_f_236_, v_a_237_);
return v___x_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_MapsTo_restrict(lean_object* v_00_u03b1_239_, lean_object* v_00_u03b2_240_, lean_object* v_f_241_, lean_object* v_s_242_, lean_object* v_t_243_, lean_object* v_h_244_, lean_object* v_a_245_){
_start:
{
lean_object* v___x_246_; 
v___x_246_ = lean_apply_1(v_f_241_, v_a_245_);
return v___x_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_restrictPreimage___redArg(lean_object* v_f_247_, lean_object* v_a_248_){
_start:
{
lean_object* v___x_249_; 
v___x_249_ = lean_apply_1(v_f_247_, v_a_248_);
return v___x_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_restrictPreimage(lean_object* v_00_u03b1_250_, lean_object* v_00_u03b2_251_, lean_object* v_t_252_, lean_object* v_f_253_, lean_object* v_a_254_){
_start:
{
lean_object* v___x_255_; 
v___x_255_ = lean_apply_1(v_f_253_, v_a_254_);
return v___x_255_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_CoeSort(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_SProd(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Subtype(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Notation(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Push_Attr(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_BuiltinRules(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Frontend_Tactic(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Main(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Operations(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_CoeSort(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_SProd(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Subtype(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Push_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_BuiltinRules(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Frontend_Tactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Set_Operations(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Set_CoeSort(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_SProd(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Subtype(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Notation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Push_Attr(uint8_t builtin);
lean_object* initialize_aesop_Aesop_BuiltinRules(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Frontend_Tactic(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Main(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Set_Operations(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_CoeSort(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_SProd(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Subtype(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Push_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_BuiltinRules(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Frontend_Tactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Set_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Set_Operations(builtin);
}
#ifdef __cplusplus
}
#endif
