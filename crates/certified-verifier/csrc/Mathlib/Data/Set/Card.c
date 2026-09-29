// Lean compiler output
// Module: Mathlib.Data.Set.Card
// Imports: public import Init public meta import Init public import Mathlib.SetTheory.Cardinal.Finite public import Mathlib.Data.Set.Finite.Powerset
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
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Set_tacticToFinite__tac___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Set"};
static const lean_object* lp_mathlib_Set_tacticToFinite__tac___closed__0 = (const lean_object*)&lp_mathlib_Set_tacticToFinite__tac___closed__0_value;
static const lean_string_object lp_mathlib_Set_tacticToFinite__tac___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticToFinite_tac"};
static const lean_object* lp_mathlib_Set_tacticToFinite__tac___closed__1 = (const lean_object*)&lp_mathlib_Set_tacticToFinite__tac___closed__1_value;
static const lean_ctor_object lp_mathlib_Set_tacticToFinite__tac___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_tacticToFinite__tac___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_ctor_object lp_mathlib_Set_tacticToFinite__tac___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_tacticToFinite__tac___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Set_tacticToFinite__tac___closed__1_value),LEAN_SCALAR_PTR_LITERAL(67, 12, 27, 51, 244, 1, 46, 176)}};
static const lean_object* lp_mathlib_Set_tacticToFinite__tac___closed__2 = (const lean_object*)&lp_mathlib_Set_tacticToFinite__tac___closed__2_value;
static const lean_string_object lp_mathlib_Set_tacticToFinite__tac___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "toFinite_tac"};
static const lean_object* lp_mathlib_Set_tacticToFinite__tac___closed__3 = (const lean_object*)&lp_mathlib_Set_tacticToFinite__tac___closed__3_value;
static const lean_ctor_object lp_mathlib_Set_tacticToFinite__tac___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Set_tacticToFinite__tac___closed__3_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Set_tacticToFinite__tac___closed__4 = (const lean_object*)&lp_mathlib_Set_tacticToFinite__tac___closed__4_value;
static const lean_ctor_object lp_mathlib_Set_tacticToFinite__tac___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Set_tacticToFinite__tac___closed__2_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Set_tacticToFinite__tac___closed__4_value)}};
static const lean_object* lp_mathlib_Set_tacticToFinite__tac___closed__5 = (const lean_object*)&lp_mathlib_Set_tacticToFinite__tac___closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Set_tacticToFinite__tac = (const lean_object*)&lp_mathlib_Set_tacticToFinite__tac___closed__5_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__0 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__0_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__1 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__1_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__2 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__2_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "apply"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__3 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(202, 125, 237, 78, 179, 140, 218, 80)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__4 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__4_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Set.toFinite"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__5 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__5_value;
static lean_once_cell_t lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__6;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "toFinite"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__7 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__7_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_tacticToFinite__tac___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(78, 221, 101, 14, 128, 211, 37, 134)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__8 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__9 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__9_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__10 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Set_tacticTo__encard__tac___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "tacticTo_encard_tac"};
static const lean_object* lp_mathlib_Set_tacticTo__encard__tac___closed__0 = (const lean_object*)&lp_mathlib_Set_tacticTo__encard__tac___closed__0_value;
static const lean_ctor_object lp_mathlib_Set_tacticTo__encard__tac___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set_tacticToFinite__tac___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_ctor_object lp_mathlib_Set_tacticTo__encard__tac___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_tacticTo__encard__tac___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Set_tacticTo__encard__tac___closed__0_value),LEAN_SCALAR_PTR_LITERAL(9, 169, 105, 214, 242, 221, 94, 140)}};
static const lean_object* lp_mathlib_Set_tacticTo__encard__tac___closed__1 = (const lean_object*)&lp_mathlib_Set_tacticTo__encard__tac___closed__1_value;
static const lean_string_object lp_mathlib_Set_tacticTo__encard__tac___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "to_encard_tac"};
static const lean_object* lp_mathlib_Set_tacticTo__encard__tac___closed__2 = (const lean_object*)&lp_mathlib_Set_tacticTo__encard__tac___closed__2_value;
static const lean_ctor_object lp_mathlib_Set_tacticTo__encard__tac___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Set_tacticTo__encard__tac___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Set_tacticTo__encard__tac___closed__3 = (const lean_object*)&lp_mathlib_Set_tacticTo__encard__tac___closed__3_value;
static const lean_ctor_object lp_mathlib_Set_tacticTo__encard__tac___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Set_tacticTo__encard__tac___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Set_tacticTo__encard__tac___closed__3_value)}};
static const lean_object* lp_mathlib_Set_tacticTo__encard__tac___closed__4 = (const lean_object*)&lp_mathlib_Set_tacticTo__encard__tac___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_Set_tacticTo__encard__tac = (const lean_object*)&lp_mathlib_Set_tacticTo__encard__tac___closed__4_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__0 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(50, 13, 241, 145, 67, 153, 105, 177)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__1 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__1_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__2 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__3 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__3_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__4 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__4_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__5 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__5_value;
static lean_once_cell_t lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__6;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "only"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__7 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__7_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__8 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__8_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "simpLemma"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__9 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__9_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__10_value_aux_1),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__10_value_aux_2),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(38, 215, 101, 250, 181, 108, 118, 102)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__10 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__10_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "←"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__11 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__11_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__12 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__12_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__13 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__13_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__14_value_aux_1),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__14_value_aux_2),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__14 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__14_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Nat.cast_le"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__15 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__15_value;
static lean_once_cell_t lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__16;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__17 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__17_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "cast_le"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__18 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__18_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__19_value_aux_0),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__18_value),LEAN_SCALAR_PTR_LITERAL(98, 206, 66, 71, 67, 150, 149, 202)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__19 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__19_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__19_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__20 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__20_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__20_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__21 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__21_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "namedArgument"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__22 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__22_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__23_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__23_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__23_value_aux_0),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__23_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__23_value_aux_1),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__23_value_aux_2),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__22_value),LEAN_SCALAR_PTR_LITERAL(226, 89, 129, 113, 173, 121, 169, 188)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__23 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__23_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__24 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__24_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 1, .m_data = "α"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__25 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__25_value;
static lean_once_cell_t lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__26;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__25_value),LEAN_SCALAR_PTR_LITERAL(102, 24, 27, 80, 217, 159, 184, 13)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__27 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__27_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__25_value),LEAN_SCALAR_PTR_LITERAL(102, 24, 27, 80, 217, 159, 184, 13)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__28 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__28_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__29 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__29_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__28_value),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__29_value),LEAN_SCALAR_PTR_LITERAL(151, 95, 206, 72, 126, 215, 111, 61)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__30 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__30_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__31 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__31_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__30_value),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__31_value),LEAN_SCALAR_PTR_LITERAL(26, 242, 46, 175, 203, 191, 82, 215)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__32 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__32_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Data"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__33 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__33_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__32_value),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__33_value),LEAN_SCALAR_PTR_LITERAL(69, 170, 154, 243, 136, 247, 115, 162)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__34 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__34_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__34_value),((lean_object*)&lp_mathlib_Set_tacticToFinite__tac___closed__0_value),LEAN_SCALAR_PTR_LITERAL(144, 133, 99, 207, 87, 58, 150, 136)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__35 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__35_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Card"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__36 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__36_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__35_value),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__36_value),LEAN_SCALAR_PTR_LITERAL(122, 195, 221, 251, 160, 74, 1, 232)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__37 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__37_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__37_value),((lean_object*)(((size_t)(1921071020) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(216, 33, 122, 151, 98, 251, 90, 25)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__38 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__38_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__39 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__39_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__38_value),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__39_value),LEAN_SCALAR_PTR_LITERAL(159, 79, 144, 147, 108, 251, 79, 209)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__40 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__40_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__41 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__41_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__40_value),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__41_value),LEAN_SCALAR_PTR_LITERAL(39, 31, 239, 134, 59, 32, 70, 46)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__42 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__42_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__42_value),((lean_object*)(((size_t)(9) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(76, 143, 218, 91, 175, 0, 53, 34)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__43 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__43_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__43_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__44 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__44_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__44_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__45 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__45_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ":="};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__46 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__46_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 6, .m_data = "termℕ∞"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__47 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__47_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__47_value),LEAN_SCALAR_PTR_LITERAL(249, 167, 24, 203, 93, 182, 115, 9)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__48 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__48_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 2, .m_data = "ℕ∞"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__49 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__49_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__50 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__50_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__51 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__51_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Nat.cast_inj"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__52 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__52_value;
static lean_once_cell_t lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__53_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__53;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "cast_inj"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__54 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__54_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__55_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__55_value_aux_0),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__54_value),LEAN_SCALAR_PTR_LITERAL(54, 74, 61, 132, 63, 194, 118, 101)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__55 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__55_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__55_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__56 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__56_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__56_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__57 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__57_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "R"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__58 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__58_value;
static lean_once_cell_t lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__59_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__59;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__58_value),LEAN_SCALAR_PTR_LITERAL(10, 150, 1, 122, 163, 250, 19, 99)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__60 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__60_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__61_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Nat.cast_add"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__61 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__61_value;
static lean_once_cell_t lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__62_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__62;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__63_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "cast_add"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__63 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__63_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__64_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__64_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__64_value_aux_0),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__63_value),LEAN_SCALAR_PTR_LITERAL(45, 11, 172, 54, 60, 245, 205, 126)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__64 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__64_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__65_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__64_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__65 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__65_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__66_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__65_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__66 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__66_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__67_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Nat.cast_one"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__67 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__67_value;
static lean_once_cell_t lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__68_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__68;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__69_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "cast_one"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__69 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__69_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__70_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__70_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__70_value_aux_0),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__69_value),LEAN_SCALAR_PTR_LITERAL(158, 72, 22, 158, 153, 136, 145, 225)}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__70 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__70_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__71_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__70_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__71 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__71_value;
static const lean_ctor_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__72_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__71_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__72 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__72_value;
static const lean_string_object lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__73_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__73 = (const lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__73_value;
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__0 = (const lean_object*)&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__1 = (const lean_object*)&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__1_value;
static const lean_array_object lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__2 = (const lean_object*)&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__2_value;
static const lean_string_object lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__3 = (const lean_object*)&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__4 = (const lean_object*)&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__4_value;
static lean_once_cell_t lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__5;
static lean_once_cell_t lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__6;
static lean_once_cell_t lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__7;
static lean_once_cell_t lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__8;
static lean_once_cell_t lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__9;
static lean_once_cell_t lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__10;
static lean_once_cell_t lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__11;
static lean_once_cell_t lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__12;
static lean_once_cell_t lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__eq__toFinset__card___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__le__ncard___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__eq__zero___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__pos___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__ne__zero__of__mem___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__powerset___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__insert__of__notMem___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_one__le__ncard__insert___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__insert__eq__ite___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__sdiff__singleton__add__one___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__sdiff__singleton__lt__of__mem___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_odd__card__insert__iff___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_even__card__insert__iff___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__image__le___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_injOn__of__ncard__image__eq___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__image__iff___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_fiber__ncard__ne__zero__iff__mem__image___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__inter__le__ncard__left___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__inter__le__ncard__right___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_eq__of__subset__of__ncard__le___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_subset__iff__eq__of__ncard__le___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_map__eq__of__subset___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_sep__of__ncard__eq___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__lt__ncard___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__le__ncard__of__injOn___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_exists__ne__map__eq__of__ncard__lt__of__maps__to___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_le__ncard__of__inj__on__range___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_surj__on__of__inj__on__of__ncard__le___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_inj__on__of__surj__on__of__ncard__le___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__union__add__ncard__inter___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__union__add__ncard__inter___auto__3;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__inter__add__ncard__union___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__inter__add__ncard__union___auto__3;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__union__eq___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__union__eq___auto__3;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__union__eq__iff___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__union__eq__iff___auto__3;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__union__lt___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__union__lt___auto__3;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__sdiff__add__ncard__of__subset___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__sdiff_x27___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__sdiff___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__le__ncard__sdiff__add__ncard___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_le__ncard__sdiff___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__sdiff__add__ncard___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__sdiff__add__ncard___auto__3;
LEAN_EXPORT lean_object* lp_mathlib_Set_sdiff__nonempty__of__ncard__lt__ncard___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_exists__mem__notMem__of__ncard__lt__ncard___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__inter__add__ncard__sdiff__eq__ncard___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__eq__ncard__iff__ncard__sdiff__eq__ncard__sdiff___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__eq__ncard__iff__ncard__sdiff__eq__ncard__sdiff___auto__3;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__le__ncard__iff__ncard__sdiff__le__ncard__sdiff___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__le__ncard__iff__ncard__sdiff__le__ncard__sdiff___auto__3;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__lt__ncard__iff__ncard__sdiff__lt__ncard__sdiff___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__lt__ncard__iff__ncard__sdiff__lt__ncard__sdiff___auto__3;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__add__ncard__compl___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__add__ncard__compl___auto__3;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__compl__add__ncard___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__compl__add__ncard___auto__3;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__compl___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__compl___auto__3;
LEAN_EXPORT lean_object* lp_mathlib_Set_exists__eq__insert__iff__ncard___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__le__one___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__le__one__iff___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__le__one__iff__eq___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__le__one__iff__subset__singleton___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_one__lt__ncard___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_one__lt__ncard__iff___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_one__lt__ncard__of__nonempty__of__even___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_two__lt__ncard__iff___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_two__lt__ncard___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_three__lt__ncard__iff___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_three__lt__ncard___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_ncard__eq__succ___auto__1;
static lean_object* _init_lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__6(void){
_start:
{
lean_object* v___x_25_; lean_object* v___x_26_; 
v___x_25_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__5));
v___x_26_ = l_String_toRawSubstring_x27(v___x_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1(lean_object* v_x_37_, lean_object* v_a_38_, lean_object* v_a_39_){
_start:
{
lean_object* v___x_40_; uint8_t v___x_41_; 
v___x_40_ = ((lean_object*)(lp_mathlib_Set_tacticToFinite__tac___closed__2));
v___x_41_ = l_Lean_Syntax_isOfKind(v_x_37_, v___x_40_);
if (v___x_41_ == 0)
{
lean_object* v___x_42_; lean_object* v___x_43_; 
v___x_42_ = lean_box(1);
v___x_43_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_43_, 0, v___x_42_);
lean_ctor_set(v___x_43_, 1, v_a_39_);
return v___x_43_;
}
else
{
lean_object* v_quotContext_44_; lean_object* v_currMacroScope_45_; lean_object* v_ref_46_; uint8_t v___x_47_; lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; 
v_quotContext_44_ = lean_ctor_get(v_a_38_, 1);
v_currMacroScope_45_ = lean_ctor_get(v_a_38_, 2);
v_ref_46_ = lean_ctor_get(v_a_38_, 5);
v___x_47_ = 0;
v___x_48_ = l_Lean_SourceInfo_fromRef(v_ref_46_, v___x_47_);
v___x_49_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__3));
v___x_50_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__4));
lean_inc_n(v___x_48_, 2);
v___x_51_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_51_, 0, v___x_48_);
lean_ctor_set(v___x_51_, 1, v___x_49_);
v___x_52_ = lean_obj_once(&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__6, &lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__6_once, _init_lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__6);
v___x_53_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__8));
lean_inc(v_currMacroScope_45_);
lean_inc(v_quotContext_44_);
v___x_54_ = l_Lean_addMacroScope(v_quotContext_44_, v___x_53_, v_currMacroScope_45_);
v___x_55_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___closed__10));
v___x_56_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_56_, 0, v___x_48_);
lean_ctor_set(v___x_56_, 1, v___x_52_);
lean_ctor_set(v___x_56_, 2, v___x_54_);
lean_ctor_set(v___x_56_, 3, v___x_55_);
v___x_57_ = l_Lean_Syntax_node2(v___x_48_, v___x_50_, v___x_51_, v___x_56_);
v___x_58_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_58_, 0, v___x_57_);
lean_ctor_set(v___x_58_, 1, v_a_39_);
return v___x_58_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1___boxed(lean_object* v_x_59_, lean_object* v_a_60_, lean_object* v_a_61_){
_start:
{
lean_object* v_res_62_; 
v_res_62_ = lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticToFinite__tac__1(v_x_59_, v_a_60_, v_a_61_);
lean_dec_ref(v_a_60_);
return v_res_62_;
}
}
static lean_object* _init_lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__6(void){
_start:
{
lean_object* v___x_91_; 
v___x_91_ = l_Array_mkArray0(lean_box(0));
return v___x_91_;
}
}
static lean_object* _init_lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__16(void){
_start:
{
lean_object* v___x_109_; lean_object* v___x_110_; 
v___x_109_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__15));
v___x_110_ = l_String_toRawSubstring_x27(v___x_109_);
return v___x_110_;
}
}
static lean_object* _init_lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__26(void){
_start:
{
lean_object* v___x_130_; lean_object* v___x_131_; 
v___x_130_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__25));
v___x_131_ = l_String_toRawSubstring_x27(v___x_130_);
return v___x_131_;
}
}
static lean_object* _init_lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__53(void){
_start:
{
lean_object* v___x_184_; lean_object* v___x_185_; 
v___x_184_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__52));
v___x_185_ = l_String_toRawSubstring_x27(v___x_184_);
return v___x_185_;
}
}
static lean_object* _init_lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__59(void){
_start:
{
lean_object* v___x_197_; lean_object* v___x_198_; 
v___x_197_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__58));
v___x_198_ = l_String_toRawSubstring_x27(v___x_197_);
return v___x_198_;
}
}
static lean_object* _init_lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__62(void){
_start:
{
lean_object* v___x_202_; lean_object* v___x_203_; 
v___x_202_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__61));
v___x_203_ = l_String_toRawSubstring_x27(v___x_202_);
return v___x_203_;
}
}
static lean_object* _init_lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__68(void){
_start:
{
lean_object* v___x_215_; lean_object* v___x_216_; 
v___x_215_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__67));
v___x_216_ = l_String_toRawSubstring_x27(v___x_215_);
return v___x_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1(lean_object* v_x_228_, lean_object* v_a_229_, lean_object* v_a_230_){
_start:
{
lean_object* v___x_231_; uint8_t v___x_232_; 
v___x_231_ = ((lean_object*)(lp_mathlib_Set_tacticTo__encard__tac___closed__1));
v___x_232_ = l_Lean_Syntax_isOfKind(v_x_228_, v___x_231_);
if (v___x_232_ == 0)
{
lean_object* v___x_233_; lean_object* v___x_234_; 
v___x_233_ = lean_box(1);
v___x_234_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_234_, 0, v___x_233_);
lean_ctor_set(v___x_234_, 1, v_a_230_);
return v___x_234_;
}
else
{
lean_object* v_quotContext_235_; lean_object* v_currMacroScope_236_; lean_object* v_ref_237_; uint8_t v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; 
v_quotContext_235_ = lean_ctor_get(v_a_229_, 1);
v_currMacroScope_236_ = lean_ctor_get(v_a_229_, 2);
v_ref_237_ = lean_ctor_get(v_a_229_, 5);
v___x_238_ = 0;
v___x_239_ = l_Lean_SourceInfo_fromRef(v_ref_237_, v___x_238_);
v___x_240_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__0));
v___x_241_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__1));
lean_inc_n(v___x_239_, 33);
v___x_242_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_242_, 0, v___x_239_);
lean_ctor_set(v___x_242_, 1, v___x_240_);
v___x_243_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__3));
v___x_244_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__5));
v___x_245_ = lean_obj_once(&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__6, &lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__6_once, _init_lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__6);
v___x_246_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_246_, 0, v___x_239_);
lean_ctor_set(v___x_246_, 1, v___x_244_);
lean_ctor_set(v___x_246_, 2, v___x_245_);
lean_inc_ref_n(v___x_246_, 8);
v___x_247_ = l_Lean_Syntax_node1(v___x_239_, v___x_243_, v___x_246_);
v___x_248_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__7));
v___x_249_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_249_, 0, v___x_239_);
lean_ctor_set(v___x_249_, 1, v___x_248_);
v___x_250_ = l_Lean_Syntax_node1(v___x_239_, v___x_244_, v___x_249_);
v___x_251_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__8));
v___x_252_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_252_, 0, v___x_239_);
lean_ctor_set(v___x_252_, 1, v___x_251_);
v___x_253_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__10));
v___x_254_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__11));
v___x_255_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_255_, 0, v___x_239_);
lean_ctor_set(v___x_255_, 1, v___x_254_);
v___x_256_ = l_Lean_Syntax_node1(v___x_239_, v___x_244_, v___x_255_);
v___x_257_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__14));
v___x_258_ = lean_obj_once(&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__16, &lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__16_once, _init_lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__16);
v___x_259_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__19));
lean_inc_n(v_currMacroScope_236_, 6);
lean_inc_n(v_quotContext_235_, 6);
v___x_260_ = l_Lean_addMacroScope(v_quotContext_235_, v___x_259_, v_currMacroScope_236_);
v___x_261_ = lean_box(0);
v___x_262_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__21));
v___x_263_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_263_, 0, v___x_239_);
lean_ctor_set(v___x_263_, 1, v___x_258_);
lean_ctor_set(v___x_263_, 2, v___x_260_);
lean_ctor_set(v___x_263_, 3, v___x_262_);
v___x_264_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__23));
v___x_265_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__24));
v___x_266_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_266_, 0, v___x_239_);
lean_ctor_set(v___x_266_, 1, v___x_265_);
v___x_267_ = lean_obj_once(&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__26, &lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__26_once, _init_lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__26);
v___x_268_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__27));
v___x_269_ = l_Lean_addMacroScope(v_quotContext_235_, v___x_268_, v_currMacroScope_236_);
v___x_270_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__45));
v___x_271_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_271_, 0, v___x_239_);
lean_ctor_set(v___x_271_, 1, v___x_267_);
lean_ctor_set(v___x_271_, 2, v___x_269_);
lean_ctor_set(v___x_271_, 3, v___x_270_);
v___x_272_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__46));
v___x_273_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_273_, 0, v___x_239_);
lean_ctor_set(v___x_273_, 1, v___x_272_);
v___x_274_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__48));
v___x_275_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__49));
v___x_276_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_276_, 0, v___x_239_);
lean_ctor_set(v___x_276_, 1, v___x_275_);
v___x_277_ = l_Lean_Syntax_node1(v___x_239_, v___x_274_, v___x_276_);
v___x_278_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__50));
v___x_279_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_279_, 0, v___x_239_);
lean_ctor_set(v___x_279_, 1, v___x_278_);
lean_inc_ref(v___x_279_);
lean_inc(v___x_277_);
lean_inc_ref(v___x_273_);
lean_inc_ref(v___x_266_);
v___x_280_ = l_Lean_Syntax_node5(v___x_239_, v___x_264_, v___x_266_, v___x_271_, v___x_273_, v___x_277_, v___x_279_);
v___x_281_ = l_Lean_Syntax_node1(v___x_239_, v___x_244_, v___x_280_);
v___x_282_ = l_Lean_Syntax_node2(v___x_239_, v___x_257_, v___x_263_, v___x_281_);
lean_inc(v___x_256_);
v___x_283_ = l_Lean_Syntax_node3(v___x_239_, v___x_253_, v___x_246_, v___x_256_, v___x_282_);
v___x_284_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__51));
v___x_285_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_285_, 0, v___x_239_);
lean_ctor_set(v___x_285_, 1, v___x_284_);
v___x_286_ = lean_obj_once(&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__53, &lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__53_once, _init_lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__53);
v___x_287_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__55));
v___x_288_ = l_Lean_addMacroScope(v_quotContext_235_, v___x_287_, v_currMacroScope_236_);
v___x_289_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__57));
v___x_290_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_290_, 0, v___x_239_);
lean_ctor_set(v___x_290_, 1, v___x_286_);
lean_ctor_set(v___x_290_, 2, v___x_288_);
lean_ctor_set(v___x_290_, 3, v___x_289_);
v___x_291_ = lean_obj_once(&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__59, &lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__59_once, _init_lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__59);
v___x_292_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__60));
v___x_293_ = l_Lean_addMacroScope(v_quotContext_235_, v___x_292_, v_currMacroScope_236_);
v___x_294_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_294_, 0, v___x_239_);
lean_ctor_set(v___x_294_, 1, v___x_291_);
lean_ctor_set(v___x_294_, 2, v___x_293_);
lean_ctor_set(v___x_294_, 3, v___x_261_);
v___x_295_ = l_Lean_Syntax_node5(v___x_239_, v___x_264_, v___x_266_, v___x_294_, v___x_273_, v___x_277_, v___x_279_);
v___x_296_ = l_Lean_Syntax_node1(v___x_239_, v___x_244_, v___x_295_);
v___x_297_ = l_Lean_Syntax_node2(v___x_239_, v___x_257_, v___x_290_, v___x_296_);
v___x_298_ = l_Lean_Syntax_node3(v___x_239_, v___x_253_, v___x_246_, v___x_256_, v___x_297_);
v___x_299_ = lean_obj_once(&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__62, &lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__62_once, _init_lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__62);
v___x_300_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__64));
v___x_301_ = l_Lean_addMacroScope(v_quotContext_235_, v___x_300_, v_currMacroScope_236_);
v___x_302_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__66));
v___x_303_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_303_, 0, v___x_239_);
lean_ctor_set(v___x_303_, 1, v___x_299_);
lean_ctor_set(v___x_303_, 2, v___x_301_);
lean_ctor_set(v___x_303_, 3, v___x_302_);
v___x_304_ = l_Lean_Syntax_node3(v___x_239_, v___x_253_, v___x_246_, v___x_246_, v___x_303_);
v___x_305_ = lean_obj_once(&lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__68, &lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__68_once, _init_lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__68);
v___x_306_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__70));
v___x_307_ = l_Lean_addMacroScope(v_quotContext_235_, v___x_306_, v_currMacroScope_236_);
v___x_308_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__72));
v___x_309_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_309_, 0, v___x_239_);
lean_ctor_set(v___x_309_, 1, v___x_305_);
lean_ctor_set(v___x_309_, 2, v___x_307_);
lean_ctor_set(v___x_309_, 3, v___x_308_);
v___x_310_ = l_Lean_Syntax_node3(v___x_239_, v___x_253_, v___x_246_, v___x_246_, v___x_309_);
lean_inc_ref_n(v___x_285_, 2);
v___x_311_ = l_Lean_Syntax_node7(v___x_239_, v___x_244_, v___x_283_, v___x_285_, v___x_298_, v___x_285_, v___x_304_, v___x_285_, v___x_310_);
v___x_312_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__73));
v___x_313_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_313_, 0, v___x_239_);
lean_ctor_set(v___x_313_, 1, v___x_312_);
v___x_314_ = l_Lean_Syntax_node3(v___x_239_, v___x_244_, v___x_252_, v___x_311_, v___x_313_);
v___x_315_ = l_Lean_Syntax_node6(v___x_239_, v___x_241_, v___x_242_, v___x_247_, v___x_246_, v___x_250_, v___x_314_, v___x_246_);
v___x_316_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_316_, 0, v___x_315_);
lean_ctor_set(v___x_316_, 1, v_a_230_);
return v___x_316_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___boxed(lean_object* v_x_317_, lean_object* v_a_318_, lean_object* v_a_319_){
_start:
{
lean_object* v_res_320_; 
v_res_320_ = lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1(v_x_317_, v_a_318_, v_a_319_);
lean_dec_ref(v_a_318_);
return v_res_320_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__5(void){
_start:
{
lean_object* v___x_335_; lean_object* v___x_336_; 
v___x_335_ = ((lean_object*)(lp_mathlib_Set_tacticToFinite__tac___closed__3));
v___x_336_ = l_Lean_mkAtom(v___x_335_);
return v___x_336_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__6(void){
_start:
{
lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; 
v___x_337_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__5, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__5_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__5);
v___x_338_ = ((lean_object*)(lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__2));
v___x_339_ = lean_array_push(v___x_338_, v___x_337_);
return v___x_339_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__7(void){
_start:
{
lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; 
v___x_340_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__6, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__6_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__6);
v___x_341_ = ((lean_object*)(lp_mathlib_Set_tacticToFinite__tac___closed__2));
v___x_342_ = lean_box(2);
v___x_343_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_343_, 0, v___x_342_);
lean_ctor_set(v___x_343_, 1, v___x_341_);
lean_ctor_set(v___x_343_, 2, v___x_340_);
return v___x_343_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__8(void){
_start:
{
lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; 
v___x_344_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__7, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__7_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__7);
v___x_345_ = ((lean_object*)(lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__2));
v___x_346_ = lean_array_push(v___x_345_, v___x_344_);
return v___x_346_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__9(void){
_start:
{
lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; 
v___x_347_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__8, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__8_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__8);
v___x_348_ = ((lean_object*)(lp_mathlib_Set___aux__Mathlib__Data__Set__Card______macroRules__Set__tacticTo__encard__tac__1___closed__5));
v___x_349_ = lean_box(2);
v___x_350_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_350_, 0, v___x_349_);
lean_ctor_set(v___x_350_, 1, v___x_348_);
lean_ctor_set(v___x_350_, 2, v___x_347_);
return v___x_350_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__10(void){
_start:
{
lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; 
v___x_351_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__9, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__9_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__9);
v___x_352_ = ((lean_object*)(lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__2));
v___x_353_ = lean_array_push(v___x_352_, v___x_351_);
return v___x_353_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__11(void){
_start:
{
lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; 
v___x_354_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__10, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__10_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__10);
v___x_355_ = ((lean_object*)(lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__4));
v___x_356_ = lean_box(2);
v___x_357_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_357_, 0, v___x_356_);
lean_ctor_set(v___x_357_, 1, v___x_355_);
lean_ctor_set(v___x_357_, 2, v___x_354_);
return v___x_357_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__12(void){
_start:
{
lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; 
v___x_358_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__11, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__11_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__11);
v___x_359_ = ((lean_object*)(lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__2));
v___x_360_ = lean_array_push(v___x_359_, v___x_358_);
return v___x_360_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13(void){
_start:
{
lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; lean_object* v___x_364_; 
v___x_361_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__12, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__12_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__12);
v___x_362_ = ((lean_object*)(lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__1));
v___x_363_ = lean_box(2);
v___x_364_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_364_, 0, v___x_363_);
lean_ctor_set(v___x_364_, 1, v___x_362_);
lean_ctor_set(v___x_364_, 2, v___x_361_);
return v___x_364_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1(void){
_start:
{
lean_object* v___x_365_; 
v___x_365_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_365_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__le__ncard___auto__1(void){
_start:
{
lean_object* v___x_366_; 
v___x_366_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_366_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__eq__zero___auto__1(void){
_start:
{
lean_object* v___x_367_; 
v___x_367_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_367_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__pos___auto__1(void){
_start:
{
lean_object* v___x_368_; 
v___x_368_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_368_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__ne__zero__of__mem___auto__1(void){
_start:
{
lean_object* v___x_369_; 
v___x_369_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_369_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__powerset___auto__1(void){
_start:
{
lean_object* v___x_370_; 
v___x_370_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_370_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__insert__of__notMem___auto__1(void){
_start:
{
lean_object* v___x_371_; 
v___x_371_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_371_;
}
}
static lean_object* _init_lp_mathlib_Set_one__le__ncard__insert___auto__1(void){
_start:
{
lean_object* v___x_372_; 
v___x_372_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_372_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__insert__eq__ite___auto__1(void){
_start:
{
lean_object* v___x_373_; 
v___x_373_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_373_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__sdiff__singleton__add__one___auto__1(void){
_start:
{
lean_object* v___x_374_; 
v___x_374_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_374_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__sdiff__singleton__lt__of__mem___auto__1(void){
_start:
{
lean_object* v___x_375_; 
v___x_375_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_375_;
}
}
static lean_object* _init_lp_mathlib_Set_odd__card__insert__iff___auto__1(void){
_start:
{
lean_object* v___x_376_; 
v___x_376_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_376_;
}
}
static lean_object* _init_lp_mathlib_Set_even__card__insert__iff___auto__1(void){
_start:
{
lean_object* v___x_377_; 
v___x_377_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_377_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__image__le___auto__1(void){
_start:
{
lean_object* v___x_378_; 
v___x_378_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_378_;
}
}
static lean_object* _init_lp_mathlib_Set_injOn__of__ncard__image__eq___auto__1(void){
_start:
{
lean_object* v___x_379_; 
v___x_379_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_379_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__image__iff___auto__1(void){
_start:
{
lean_object* v___x_380_; 
v___x_380_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_380_;
}
}
static lean_object* _init_lp_mathlib_Set_fiber__ncard__ne__zero__iff__mem__image___auto__1(void){
_start:
{
lean_object* v___x_381_; 
v___x_381_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_381_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__inter__le__ncard__left___auto__1(void){
_start:
{
lean_object* v___x_382_; 
v___x_382_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_382_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__inter__le__ncard__right___auto__1(void){
_start:
{
lean_object* v___x_383_; 
v___x_383_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_383_;
}
}
static lean_object* _init_lp_mathlib_Set_eq__of__subset__of__ncard__le___auto__1(void){
_start:
{
lean_object* v___x_384_; 
v___x_384_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_384_;
}
}
static lean_object* _init_lp_mathlib_Set_subset__iff__eq__of__ncard__le___auto__1(void){
_start:
{
lean_object* v___x_385_; 
v___x_385_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_385_;
}
}
static lean_object* _init_lp_mathlib_Set_map__eq__of__subset___auto__1(void){
_start:
{
lean_object* v___x_386_; 
v___x_386_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_386_;
}
}
static lean_object* _init_lp_mathlib_Set_sep__of__ncard__eq___auto__1(void){
_start:
{
lean_object* v___x_387_; 
v___x_387_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_387_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__lt__ncard___auto__1(void){
_start:
{
lean_object* v___x_388_; 
v___x_388_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_388_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__le__ncard__of__injOn___auto__1(void){
_start:
{
lean_object* v___x_389_; 
v___x_389_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_389_;
}
}
static lean_object* _init_lp_mathlib_Set_exists__ne__map__eq__of__ncard__lt__of__maps__to___auto__1(void){
_start:
{
lean_object* v___x_390_; 
v___x_390_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_390_;
}
}
static lean_object* _init_lp_mathlib_Set_le__ncard__of__inj__on__range___auto__1(void){
_start:
{
lean_object* v___x_391_; 
v___x_391_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_391_;
}
}
static lean_object* _init_lp_mathlib_Set_surj__on__of__inj__on__of__ncard__le___auto__1(void){
_start:
{
lean_object* v___x_392_; 
v___x_392_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_392_;
}
}
static lean_object* _init_lp_mathlib_Set_inj__on__of__surj__on__of__ncard__le___auto__1(void){
_start:
{
lean_object* v___x_393_; 
v___x_393_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_393_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__union__add__ncard__inter___auto__1(void){
_start:
{
lean_object* v___x_394_; 
v___x_394_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_394_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__union__add__ncard__inter___auto__3(void){
_start:
{
lean_object* v___x_395_; 
v___x_395_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_395_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__inter__add__ncard__union___auto__1(void){
_start:
{
lean_object* v___x_396_; 
v___x_396_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_396_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__inter__add__ncard__union___auto__3(void){
_start:
{
lean_object* v___x_397_; 
v___x_397_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_397_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__union__eq___auto__1(void){
_start:
{
lean_object* v___x_398_; 
v___x_398_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_398_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__union__eq___auto__3(void){
_start:
{
lean_object* v___x_399_; 
v___x_399_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_399_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__union__eq__iff___auto__1(void){
_start:
{
lean_object* v___x_400_; 
v___x_400_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_400_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__union__eq__iff___auto__3(void){
_start:
{
lean_object* v___x_401_; 
v___x_401_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_401_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__union__lt___auto__1(void){
_start:
{
lean_object* v___x_402_; 
v___x_402_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_402_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__union__lt___auto__3(void){
_start:
{
lean_object* v___x_403_; 
v___x_403_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_403_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__sdiff__add__ncard__of__subset___auto__1(void){
_start:
{
lean_object* v___x_404_; 
v___x_404_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_404_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__sdiff_x27___auto__1(void){
_start:
{
lean_object* v___x_405_; 
v___x_405_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_405_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__sdiff___auto__1(void){
_start:
{
lean_object* v___x_406_; 
v___x_406_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_406_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__le__ncard__sdiff__add__ncard___auto__1(void){
_start:
{
lean_object* v___x_407_; 
v___x_407_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_407_;
}
}
static lean_object* _init_lp_mathlib_Set_le__ncard__sdiff___auto__1(void){
_start:
{
lean_object* v___x_408_; 
v___x_408_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_408_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__sdiff__add__ncard___auto__1(void){
_start:
{
lean_object* v___x_409_; 
v___x_409_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_409_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__sdiff__add__ncard___auto__3(void){
_start:
{
lean_object* v___x_410_; 
v___x_410_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_410_;
}
}
static lean_object* _init_lp_mathlib_Set_sdiff__nonempty__of__ncard__lt__ncard___auto__1(void){
_start:
{
lean_object* v___x_411_; 
v___x_411_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_411_;
}
}
static lean_object* _init_lp_mathlib_Set_exists__mem__notMem__of__ncard__lt__ncard___auto__1(void){
_start:
{
lean_object* v___x_412_; 
v___x_412_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_412_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__inter__add__ncard__sdiff__eq__ncard___auto__1(void){
_start:
{
lean_object* v___x_413_; 
v___x_413_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_413_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__eq__ncard__iff__ncard__sdiff__eq__ncard__sdiff___auto__1(void){
_start:
{
lean_object* v___x_414_; 
v___x_414_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_414_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__eq__ncard__iff__ncard__sdiff__eq__ncard__sdiff___auto__3(void){
_start:
{
lean_object* v___x_415_; 
v___x_415_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_415_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__le__ncard__iff__ncard__sdiff__le__ncard__sdiff___auto__1(void){
_start:
{
lean_object* v___x_416_; 
v___x_416_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_416_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__le__ncard__iff__ncard__sdiff__le__ncard__sdiff___auto__3(void){
_start:
{
lean_object* v___x_417_; 
v___x_417_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_417_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__lt__ncard__iff__ncard__sdiff__lt__ncard__sdiff___auto__1(void){
_start:
{
lean_object* v___x_418_; 
v___x_418_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_418_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__lt__ncard__iff__ncard__sdiff__lt__ncard__sdiff___auto__3(void){
_start:
{
lean_object* v___x_419_; 
v___x_419_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_419_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__add__ncard__compl___auto__1(void){
_start:
{
lean_object* v___x_420_; 
v___x_420_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_420_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__add__ncard__compl___auto__3(void){
_start:
{
lean_object* v___x_421_; 
v___x_421_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_421_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__compl__add__ncard___auto__1(void){
_start:
{
lean_object* v___x_422_; 
v___x_422_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_422_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__compl__add__ncard___auto__3(void){
_start:
{
lean_object* v___x_423_; 
v___x_423_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_423_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__compl___auto__1(void){
_start:
{
lean_object* v___x_424_; 
v___x_424_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_424_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__compl___auto__3(void){
_start:
{
lean_object* v___x_425_; 
v___x_425_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_425_;
}
}
static lean_object* _init_lp_mathlib_Set_exists__eq__insert__iff__ncard___auto__1(void){
_start:
{
lean_object* v___x_426_; 
v___x_426_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_426_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__le__one___auto__1(void){
_start:
{
lean_object* v___x_427_; 
v___x_427_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_427_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__le__one__iff___auto__1(void){
_start:
{
lean_object* v___x_428_; 
v___x_428_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_428_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__le__one__iff__eq___auto__1(void){
_start:
{
lean_object* v___x_429_; 
v___x_429_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_429_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__le__one__iff__subset__singleton___auto__1(void){
_start:
{
lean_object* v___x_430_; 
v___x_430_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_430_;
}
}
static lean_object* _init_lp_mathlib_Set_one__lt__ncard___auto__1(void){
_start:
{
lean_object* v___x_431_; 
v___x_431_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_431_;
}
}
static lean_object* _init_lp_mathlib_Set_one__lt__ncard__iff___auto__1(void){
_start:
{
lean_object* v___x_432_; 
v___x_432_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_432_;
}
}
static lean_object* _init_lp_mathlib_Set_one__lt__ncard__of__nonempty__of__even___auto__1(void){
_start:
{
lean_object* v___x_433_; 
v___x_433_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_433_;
}
}
static lean_object* _init_lp_mathlib_Set_two__lt__ncard__iff___auto__1(void){
_start:
{
lean_object* v___x_434_; 
v___x_434_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_434_;
}
}
static lean_object* _init_lp_mathlib_Set_two__lt__ncard___auto__1(void){
_start:
{
lean_object* v___x_435_; 
v___x_435_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_435_;
}
}
static lean_object* _init_lp_mathlib_Set_three__lt__ncard__iff___auto__1(void){
_start:
{
lean_object* v___x_436_; 
v___x_436_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_436_;
}
}
static lean_object* _init_lp_mathlib_Set_three__lt__ncard___auto__1(void){
_start:
{
lean_object* v___x_437_; 
v___x_437_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_437_;
}
}
static lean_object* _init_lp_mathlib_Set_ncard__eq__succ___auto__1(void){
_start:
{
lean_object* v___x_438_; 
v___x_438_ = lean_obj_once(&lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13, &lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13_once, _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1___closed__13);
return v___x_438_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_Finite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Powerset(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Card(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Powerset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Set_Card(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Set_ncard__eq__toFinset__card___auto__1 = _init_lp_mathlib_Set_ncard__eq__toFinset__card___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__eq__toFinset__card___auto__1);
lp_mathlib_Set_ncard__le__ncard___auto__1 = _init_lp_mathlib_Set_ncard__le__ncard___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__le__ncard___auto__1);
lp_mathlib_Set_ncard__eq__zero___auto__1 = _init_lp_mathlib_Set_ncard__eq__zero___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__eq__zero___auto__1);
lp_mathlib_Set_ncard__pos___auto__1 = _init_lp_mathlib_Set_ncard__pos___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__pos___auto__1);
lp_mathlib_Set_ncard__ne__zero__of__mem___auto__1 = _init_lp_mathlib_Set_ncard__ne__zero__of__mem___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__ne__zero__of__mem___auto__1);
lp_mathlib_Set_ncard__powerset___auto__1 = _init_lp_mathlib_Set_ncard__powerset___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__powerset___auto__1);
lp_mathlib_Set_ncard__insert__of__notMem___auto__1 = _init_lp_mathlib_Set_ncard__insert__of__notMem___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__insert__of__notMem___auto__1);
lp_mathlib_Set_one__le__ncard__insert___auto__1 = _init_lp_mathlib_Set_one__le__ncard__insert___auto__1();
lean_mark_persistent(lp_mathlib_Set_one__le__ncard__insert___auto__1);
lp_mathlib_Set_ncard__insert__eq__ite___auto__1 = _init_lp_mathlib_Set_ncard__insert__eq__ite___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__insert__eq__ite___auto__1);
lp_mathlib_Set_ncard__sdiff__singleton__add__one___auto__1 = _init_lp_mathlib_Set_ncard__sdiff__singleton__add__one___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__sdiff__singleton__add__one___auto__1);
lp_mathlib_Set_ncard__sdiff__singleton__lt__of__mem___auto__1 = _init_lp_mathlib_Set_ncard__sdiff__singleton__lt__of__mem___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__sdiff__singleton__lt__of__mem___auto__1);
lp_mathlib_Set_odd__card__insert__iff___auto__1 = _init_lp_mathlib_Set_odd__card__insert__iff___auto__1();
lean_mark_persistent(lp_mathlib_Set_odd__card__insert__iff___auto__1);
lp_mathlib_Set_even__card__insert__iff___auto__1 = _init_lp_mathlib_Set_even__card__insert__iff___auto__1();
lean_mark_persistent(lp_mathlib_Set_even__card__insert__iff___auto__1);
lp_mathlib_Set_ncard__image__le___auto__1 = _init_lp_mathlib_Set_ncard__image__le___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__image__le___auto__1);
lp_mathlib_Set_injOn__of__ncard__image__eq___auto__1 = _init_lp_mathlib_Set_injOn__of__ncard__image__eq___auto__1();
lean_mark_persistent(lp_mathlib_Set_injOn__of__ncard__image__eq___auto__1);
lp_mathlib_Set_ncard__image__iff___auto__1 = _init_lp_mathlib_Set_ncard__image__iff___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__image__iff___auto__1);
lp_mathlib_Set_fiber__ncard__ne__zero__iff__mem__image___auto__1 = _init_lp_mathlib_Set_fiber__ncard__ne__zero__iff__mem__image___auto__1();
lean_mark_persistent(lp_mathlib_Set_fiber__ncard__ne__zero__iff__mem__image___auto__1);
lp_mathlib_Set_ncard__inter__le__ncard__left___auto__1 = _init_lp_mathlib_Set_ncard__inter__le__ncard__left___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__inter__le__ncard__left___auto__1);
lp_mathlib_Set_ncard__inter__le__ncard__right___auto__1 = _init_lp_mathlib_Set_ncard__inter__le__ncard__right___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__inter__le__ncard__right___auto__1);
lp_mathlib_Set_eq__of__subset__of__ncard__le___auto__1 = _init_lp_mathlib_Set_eq__of__subset__of__ncard__le___auto__1();
lean_mark_persistent(lp_mathlib_Set_eq__of__subset__of__ncard__le___auto__1);
lp_mathlib_Set_subset__iff__eq__of__ncard__le___auto__1 = _init_lp_mathlib_Set_subset__iff__eq__of__ncard__le___auto__1();
lean_mark_persistent(lp_mathlib_Set_subset__iff__eq__of__ncard__le___auto__1);
lp_mathlib_Set_map__eq__of__subset___auto__1 = _init_lp_mathlib_Set_map__eq__of__subset___auto__1();
lean_mark_persistent(lp_mathlib_Set_map__eq__of__subset___auto__1);
lp_mathlib_Set_sep__of__ncard__eq___auto__1 = _init_lp_mathlib_Set_sep__of__ncard__eq___auto__1();
lean_mark_persistent(lp_mathlib_Set_sep__of__ncard__eq___auto__1);
lp_mathlib_Set_ncard__lt__ncard___auto__1 = _init_lp_mathlib_Set_ncard__lt__ncard___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__lt__ncard___auto__1);
lp_mathlib_Set_ncard__le__ncard__of__injOn___auto__1 = _init_lp_mathlib_Set_ncard__le__ncard__of__injOn___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__le__ncard__of__injOn___auto__1);
lp_mathlib_Set_exists__ne__map__eq__of__ncard__lt__of__maps__to___auto__1 = _init_lp_mathlib_Set_exists__ne__map__eq__of__ncard__lt__of__maps__to___auto__1();
lean_mark_persistent(lp_mathlib_Set_exists__ne__map__eq__of__ncard__lt__of__maps__to___auto__1);
lp_mathlib_Set_le__ncard__of__inj__on__range___auto__1 = _init_lp_mathlib_Set_le__ncard__of__inj__on__range___auto__1();
lean_mark_persistent(lp_mathlib_Set_le__ncard__of__inj__on__range___auto__1);
lp_mathlib_Set_surj__on__of__inj__on__of__ncard__le___auto__1 = _init_lp_mathlib_Set_surj__on__of__inj__on__of__ncard__le___auto__1();
lean_mark_persistent(lp_mathlib_Set_surj__on__of__inj__on__of__ncard__le___auto__1);
lp_mathlib_Set_inj__on__of__surj__on__of__ncard__le___auto__1 = _init_lp_mathlib_Set_inj__on__of__surj__on__of__ncard__le___auto__1();
lean_mark_persistent(lp_mathlib_Set_inj__on__of__surj__on__of__ncard__le___auto__1);
lp_mathlib_Set_ncard__union__add__ncard__inter___auto__1 = _init_lp_mathlib_Set_ncard__union__add__ncard__inter___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__union__add__ncard__inter___auto__1);
lp_mathlib_Set_ncard__union__add__ncard__inter___auto__3 = _init_lp_mathlib_Set_ncard__union__add__ncard__inter___auto__3();
lean_mark_persistent(lp_mathlib_Set_ncard__union__add__ncard__inter___auto__3);
lp_mathlib_Set_ncard__inter__add__ncard__union___auto__1 = _init_lp_mathlib_Set_ncard__inter__add__ncard__union___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__inter__add__ncard__union___auto__1);
lp_mathlib_Set_ncard__inter__add__ncard__union___auto__3 = _init_lp_mathlib_Set_ncard__inter__add__ncard__union___auto__3();
lean_mark_persistent(lp_mathlib_Set_ncard__inter__add__ncard__union___auto__3);
lp_mathlib_Set_ncard__union__eq___auto__1 = _init_lp_mathlib_Set_ncard__union__eq___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__union__eq___auto__1);
lp_mathlib_Set_ncard__union__eq___auto__3 = _init_lp_mathlib_Set_ncard__union__eq___auto__3();
lean_mark_persistent(lp_mathlib_Set_ncard__union__eq___auto__3);
lp_mathlib_Set_ncard__union__eq__iff___auto__1 = _init_lp_mathlib_Set_ncard__union__eq__iff___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__union__eq__iff___auto__1);
lp_mathlib_Set_ncard__union__eq__iff___auto__3 = _init_lp_mathlib_Set_ncard__union__eq__iff___auto__3();
lean_mark_persistent(lp_mathlib_Set_ncard__union__eq__iff___auto__3);
lp_mathlib_Set_ncard__union__lt___auto__1 = _init_lp_mathlib_Set_ncard__union__lt___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__union__lt___auto__1);
lp_mathlib_Set_ncard__union__lt___auto__3 = _init_lp_mathlib_Set_ncard__union__lt___auto__3();
lean_mark_persistent(lp_mathlib_Set_ncard__union__lt___auto__3);
lp_mathlib_Set_ncard__sdiff__add__ncard__of__subset___auto__1 = _init_lp_mathlib_Set_ncard__sdiff__add__ncard__of__subset___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__sdiff__add__ncard__of__subset___auto__1);
lp_mathlib_Set_ncard__sdiff_x27___auto__1 = _init_lp_mathlib_Set_ncard__sdiff_x27___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__sdiff_x27___auto__1);
lp_mathlib_Set_ncard__sdiff___auto__1 = _init_lp_mathlib_Set_ncard__sdiff___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__sdiff___auto__1);
lp_mathlib_Set_ncard__le__ncard__sdiff__add__ncard___auto__1 = _init_lp_mathlib_Set_ncard__le__ncard__sdiff__add__ncard___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__le__ncard__sdiff__add__ncard___auto__1);
lp_mathlib_Set_le__ncard__sdiff___auto__1 = _init_lp_mathlib_Set_le__ncard__sdiff___auto__1();
lean_mark_persistent(lp_mathlib_Set_le__ncard__sdiff___auto__1);
lp_mathlib_Set_ncard__sdiff__add__ncard___auto__1 = _init_lp_mathlib_Set_ncard__sdiff__add__ncard___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__sdiff__add__ncard___auto__1);
lp_mathlib_Set_ncard__sdiff__add__ncard___auto__3 = _init_lp_mathlib_Set_ncard__sdiff__add__ncard___auto__3();
lean_mark_persistent(lp_mathlib_Set_ncard__sdiff__add__ncard___auto__3);
lp_mathlib_Set_sdiff__nonempty__of__ncard__lt__ncard___auto__1 = _init_lp_mathlib_Set_sdiff__nonempty__of__ncard__lt__ncard___auto__1();
lean_mark_persistent(lp_mathlib_Set_sdiff__nonempty__of__ncard__lt__ncard___auto__1);
lp_mathlib_Set_exists__mem__notMem__of__ncard__lt__ncard___auto__1 = _init_lp_mathlib_Set_exists__mem__notMem__of__ncard__lt__ncard___auto__1();
lean_mark_persistent(lp_mathlib_Set_exists__mem__notMem__of__ncard__lt__ncard___auto__1);
lp_mathlib_Set_ncard__inter__add__ncard__sdiff__eq__ncard___auto__1 = _init_lp_mathlib_Set_ncard__inter__add__ncard__sdiff__eq__ncard___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__inter__add__ncard__sdiff__eq__ncard___auto__1);
lp_mathlib_Set_ncard__eq__ncard__iff__ncard__sdiff__eq__ncard__sdiff___auto__1 = _init_lp_mathlib_Set_ncard__eq__ncard__iff__ncard__sdiff__eq__ncard__sdiff___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__eq__ncard__iff__ncard__sdiff__eq__ncard__sdiff___auto__1);
lp_mathlib_Set_ncard__eq__ncard__iff__ncard__sdiff__eq__ncard__sdiff___auto__3 = _init_lp_mathlib_Set_ncard__eq__ncard__iff__ncard__sdiff__eq__ncard__sdiff___auto__3();
lean_mark_persistent(lp_mathlib_Set_ncard__eq__ncard__iff__ncard__sdiff__eq__ncard__sdiff___auto__3);
lp_mathlib_Set_ncard__le__ncard__iff__ncard__sdiff__le__ncard__sdiff___auto__1 = _init_lp_mathlib_Set_ncard__le__ncard__iff__ncard__sdiff__le__ncard__sdiff___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__le__ncard__iff__ncard__sdiff__le__ncard__sdiff___auto__1);
lp_mathlib_Set_ncard__le__ncard__iff__ncard__sdiff__le__ncard__sdiff___auto__3 = _init_lp_mathlib_Set_ncard__le__ncard__iff__ncard__sdiff__le__ncard__sdiff___auto__3();
lean_mark_persistent(lp_mathlib_Set_ncard__le__ncard__iff__ncard__sdiff__le__ncard__sdiff___auto__3);
lp_mathlib_Set_ncard__lt__ncard__iff__ncard__sdiff__lt__ncard__sdiff___auto__1 = _init_lp_mathlib_Set_ncard__lt__ncard__iff__ncard__sdiff__lt__ncard__sdiff___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__lt__ncard__iff__ncard__sdiff__lt__ncard__sdiff___auto__1);
lp_mathlib_Set_ncard__lt__ncard__iff__ncard__sdiff__lt__ncard__sdiff___auto__3 = _init_lp_mathlib_Set_ncard__lt__ncard__iff__ncard__sdiff__lt__ncard__sdiff___auto__3();
lean_mark_persistent(lp_mathlib_Set_ncard__lt__ncard__iff__ncard__sdiff__lt__ncard__sdiff___auto__3);
lp_mathlib_Set_ncard__add__ncard__compl___auto__1 = _init_lp_mathlib_Set_ncard__add__ncard__compl___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__add__ncard__compl___auto__1);
lp_mathlib_Set_ncard__add__ncard__compl___auto__3 = _init_lp_mathlib_Set_ncard__add__ncard__compl___auto__3();
lean_mark_persistent(lp_mathlib_Set_ncard__add__ncard__compl___auto__3);
lp_mathlib_Set_ncard__compl__add__ncard___auto__1 = _init_lp_mathlib_Set_ncard__compl__add__ncard___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__compl__add__ncard___auto__1);
lp_mathlib_Set_ncard__compl__add__ncard___auto__3 = _init_lp_mathlib_Set_ncard__compl__add__ncard___auto__3();
lean_mark_persistent(lp_mathlib_Set_ncard__compl__add__ncard___auto__3);
lp_mathlib_Set_ncard__compl___auto__1 = _init_lp_mathlib_Set_ncard__compl___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__compl___auto__1);
lp_mathlib_Set_ncard__compl___auto__3 = _init_lp_mathlib_Set_ncard__compl___auto__3();
lean_mark_persistent(lp_mathlib_Set_ncard__compl___auto__3);
lp_mathlib_Set_exists__eq__insert__iff__ncard___auto__1 = _init_lp_mathlib_Set_exists__eq__insert__iff__ncard___auto__1();
lean_mark_persistent(lp_mathlib_Set_exists__eq__insert__iff__ncard___auto__1);
lp_mathlib_Set_ncard__le__one___auto__1 = _init_lp_mathlib_Set_ncard__le__one___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__le__one___auto__1);
lp_mathlib_Set_ncard__le__one__iff___auto__1 = _init_lp_mathlib_Set_ncard__le__one__iff___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__le__one__iff___auto__1);
lp_mathlib_Set_ncard__le__one__iff__eq___auto__1 = _init_lp_mathlib_Set_ncard__le__one__iff__eq___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__le__one__iff__eq___auto__1);
lp_mathlib_Set_ncard__le__one__iff__subset__singleton___auto__1 = _init_lp_mathlib_Set_ncard__le__one__iff__subset__singleton___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__le__one__iff__subset__singleton___auto__1);
lp_mathlib_Set_one__lt__ncard___auto__1 = _init_lp_mathlib_Set_one__lt__ncard___auto__1();
lean_mark_persistent(lp_mathlib_Set_one__lt__ncard___auto__1);
lp_mathlib_Set_one__lt__ncard__iff___auto__1 = _init_lp_mathlib_Set_one__lt__ncard__iff___auto__1();
lean_mark_persistent(lp_mathlib_Set_one__lt__ncard__iff___auto__1);
lp_mathlib_Set_one__lt__ncard__of__nonempty__of__even___auto__1 = _init_lp_mathlib_Set_one__lt__ncard__of__nonempty__of__even___auto__1();
lean_mark_persistent(lp_mathlib_Set_one__lt__ncard__of__nonempty__of__even___auto__1);
lp_mathlib_Set_two__lt__ncard__iff___auto__1 = _init_lp_mathlib_Set_two__lt__ncard__iff___auto__1();
lean_mark_persistent(lp_mathlib_Set_two__lt__ncard__iff___auto__1);
lp_mathlib_Set_two__lt__ncard___auto__1 = _init_lp_mathlib_Set_two__lt__ncard___auto__1();
lean_mark_persistent(lp_mathlib_Set_two__lt__ncard___auto__1);
lp_mathlib_Set_three__lt__ncard__iff___auto__1 = _init_lp_mathlib_Set_three__lt__ncard__iff___auto__1();
lean_mark_persistent(lp_mathlib_Set_three__lt__ncard__iff___auto__1);
lp_mathlib_Set_three__lt__ncard___auto__1 = _init_lp_mathlib_Set_three__lt__ncard___auto__1();
lean_mark_persistent(lp_mathlib_Set_three__lt__ncard___auto__1);
lp_mathlib_Set_ncard__eq__succ___auto__1 = _init_lp_mathlib_Set_ncard__eq__succ___auto__1();
lean_mark_persistent(lp_mathlib_Set_ncard__eq__succ___auto__1);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_Finite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Powerset(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Set_Card(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Cardinal_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Finite_Powerset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Set_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Set_Card(builtin);
}
#ifdef __cplusplus
}
#endif
