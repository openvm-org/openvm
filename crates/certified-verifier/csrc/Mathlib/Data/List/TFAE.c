// Lean compiler output
// Module: Mathlib.Data.List.TFAE
// Imports: public import Init public meta import Init public import Batteries.Tactic.Alias public import Mathlib.Init
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_TFAE_out___auto__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_List_TFAE_out___auto__1___closed__0 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__0_value;
static const lean_string_object lp_mathlib_List_TFAE_out___auto__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_List_TFAE_out___auto__1___closed__1 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__1_value;
static const lean_string_object lp_mathlib_List_TFAE_out___auto__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_List_TFAE_out___auto__1___closed__2 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__2_value;
static const lean_string_object lp_mathlib_List_TFAE_out___auto__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_List_TFAE_out___auto__1___closed__3 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__3_value;
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_List_TFAE_out___auto__1___closed__4 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__4_value;
static const lean_array_object lp_mathlib_List_TFAE_out___auto__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_List_TFAE_out___auto__1___closed__5 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__5_value;
static const lean_string_object lp_mathlib_List_TFAE_out___auto__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_List_TFAE_out___auto__1___closed__6 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__6_value;
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__7_value_aux_0),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__7_value_aux_1),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__7_value_aux_2),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_List_TFAE_out___auto__1___closed__7 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__7_value;
static const lean_string_object lp_mathlib_List_TFAE_out___auto__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_List_TFAE_out___auto__1___closed__8 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__8_value;
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_List_TFAE_out___auto__1___closed__9 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__9_value;
static const lean_string_object lp_mathlib_List_TFAE_out___auto__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticRfl"};
static const lean_object* lp_mathlib_List_TFAE_out___auto__1___closed__10 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__10_value;
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__11_value_aux_0),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__1___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__11_value_aux_1),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__11_value_aux_2),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(201, 188, 173, 198, 169, 252, 183, 45)}};
static const lean_object* lp_mathlib_List_TFAE_out___auto__1___closed__11 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__11_value;
static const lean_string_object lp_mathlib_List_TFAE_out___auto__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rfl"};
static const lean_object* lp_mathlib_List_TFAE_out___auto__1___closed__12 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__12_value;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__1___closed__13;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__1___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__1___closed__14;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__1___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__1___closed__15;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__1___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__1___closed__16;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__1___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__1___closed__17;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__1___closed__18;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__1___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__1___closed__19;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__1___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__1___closed__20;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__1___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__1___closed__21;
LEAN_EXPORT lean_object* lp_mathlib_List_TFAE_out___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_List_TFAE_out___auto__3;
static const lean_string_object lp_mathlib_List_TFAE_out___auto__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "first"};
static const lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__0 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__0_value;
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__5___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__5___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__1_value_aux_0),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__5___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__1_value_aux_1),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__1_value_aux_2),((lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__0_value),LEAN_SCALAR_PTR_LITERAL(59, 232, 35, 17, 172, 62, 48, 174)}};
static const lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__1 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__1_value;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__2;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__3;
static const lean_string_object lp_mathlib_List_TFAE_out___auto__5___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__4 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__4_value;
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__5___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__4_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__5 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__5_value;
static const lean_string_object lp_mathlib_List_TFAE_out___auto__5___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__6 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__6_value;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__7;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__8;
static const lean_string_object lp_mathlib_List_TFAE_out___auto__5___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "decide"};
static const lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__9 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__9_value;
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__5___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__5___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__10_value_aux_0),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__5___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__10_value_aux_1),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__5___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__10_value_aux_2),((lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__9_value),LEAN_SCALAR_PTR_LITERAL(53, 158, 1, 232, 101, 200, 191, 197)}};
static const lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__10 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__10_value;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__11;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__12;
static const lean_string_object lp_mathlib_List_TFAE_out___auto__5___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__13 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__13_value;
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__5___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__5___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__14_value_aux_0),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__5___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__14_value_aux_1),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__5___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__14_value_aux_2),((lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__13_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__14 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__14_value;
static const lean_string_object lp_mathlib_List_TFAE_out___auto__5___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "configItem"};
static const lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__15 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__15_value;
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__5___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__5___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__16_value_aux_0),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__5___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__16_value_aux_1),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__5___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__16_value_aux_2),((lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__15_value),LEAN_SCALAR_PTR_LITERAL(205, 9, 236, 192, 59, 252, 178, 140)}};
static const lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__16 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__16_value;
static const lean_string_object lp_mathlib_List_TFAE_out___auto__5___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "posConfigItem"};
static const lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__17 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__17_value;
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__5___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__5___closed__18_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__18_value_aux_0),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__5___closed__18_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__18_value_aux_1),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__5___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__18_value_aux_2),((lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__17_value),LEAN_SCALAR_PTR_LITERAL(232, 137, 50, 117, 152, 182, 155, 132)}};
static const lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__18 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__18_value;
static const lean_string_object lp_mathlib_List_TFAE_out___auto__5___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "+"};
static const lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__19 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__19_value;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__20;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__21;
static const lean_string_object lp_mathlib_List_TFAE_out___auto__5___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "kernel"};
static const lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__22 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__22_value;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__23;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__24;
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__5___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__22_value),LEAN_SCALAR_PTR_LITERAL(78, 244, 187, 126, 175, 157, 90, 69)}};
static const lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__25 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__25_value;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__26;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__27;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__28;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__29;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__30;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__31;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__32;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__33;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__34_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__34;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__35_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__35;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__36_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__36;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__37_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__37;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__38_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__38;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__39_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__39;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__40_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__40;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__41_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__41;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__42_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__42;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__43_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__43;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__44_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__44;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__45_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__45;
static const lean_string_object lp_mathlib_List_TFAE_out___auto__5___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "fail"};
static const lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__46 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__46_value;
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__5___closed__47_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__5___closed__47_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__47_value_aux_0),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__5___closed__47_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__47_value_aux_1),((lean_object*)&lp_mathlib_List_TFAE_out___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__5___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__47_value_aux_2),((lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__46_value),LEAN_SCALAR_PTR_LITERAL(251, 214, 242, 89, 226, 36, 213, 0)}};
static const lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__47 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__47_value;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__48_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__48;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__49_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__49;
static const lean_string_object lp_mathlib_List_TFAE_out___auto__5___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "str"};
static const lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__50 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__50_value;
static const lean_ctor_object lp_mathlib_List_TFAE_out___auto__5___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__50_value),LEAN_SCALAR_PTR_LITERAL(255, 188, 142, 1, 190, 33, 34, 128)}};
static const lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__51 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__51_value;
static const lean_string_object lp_mathlib_List_TFAE_out___auto__5___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "\"TFAE indices start at 1.\""};
static const lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__52 = (const lean_object*)&lp_mathlib_List_TFAE_out___auto__5___closed__52_value;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__53_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__53;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__54_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__54;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__55_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__55;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__56_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__56;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__57_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__57;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__58_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__58;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__59_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__59;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__60_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__60;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__61_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__61;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__62_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__62;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__63_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__63;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__64_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__64;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__65_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__65;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__66_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__66;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__67_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__67;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__68_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__68;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__69_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__69;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__70_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__70;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__71_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__71;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__72_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__72;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__73_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__73;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__74_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__74;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__75_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__75;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__76_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__76;
static lean_once_cell_t lp_mathlib_List_TFAE_out___auto__5___closed__77_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_TFAE_out___auto__5___closed__77;
LEAN_EXPORT lean_object* lp_mathlib_List_TFAE_out___auto__5;
LEAN_EXPORT lean_object* lp_mathlib_List_TFAE_out___auto__7;
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__1___closed__13(void){
_start:
{
lean_object* v___x_28_; lean_object* v___x_29_; 
v___x_28_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__12));
v___x_29_ = l_Lean_mkAtom(v___x_28_);
return v___x_29_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__1___closed__14(void){
_start:
{
lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v___x_32_; 
v___x_30_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__1___closed__13, &lp_mathlib_List_TFAE_out___auto__1___closed__13_once, _init_lp_mathlib_List_TFAE_out___auto__1___closed__13);
v___x_31_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__5));
v___x_32_ = lean_array_push(v___x_31_, v___x_30_);
return v___x_32_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__1___closed__15(void){
_start:
{
lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v___x_36_; 
v___x_33_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__1___closed__14, &lp_mathlib_List_TFAE_out___auto__1___closed__14_once, _init_lp_mathlib_List_TFAE_out___auto__1___closed__14);
v___x_34_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__11));
v___x_35_ = lean_box(2);
v___x_36_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_36_, 0, v___x_35_);
lean_ctor_set(v___x_36_, 1, v___x_34_);
lean_ctor_set(v___x_36_, 2, v___x_33_);
return v___x_36_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__1___closed__16(void){
_start:
{
lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; 
v___x_37_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__1___closed__15, &lp_mathlib_List_TFAE_out___auto__1___closed__15_once, _init_lp_mathlib_List_TFAE_out___auto__1___closed__15);
v___x_38_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__5));
v___x_39_ = lean_array_push(v___x_38_, v___x_37_);
return v___x_39_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__1___closed__17(void){
_start:
{
lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; 
v___x_40_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__1___closed__16, &lp_mathlib_List_TFAE_out___auto__1___closed__16_once, _init_lp_mathlib_List_TFAE_out___auto__1___closed__16);
v___x_41_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__9));
v___x_42_ = lean_box(2);
v___x_43_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_43_, 0, v___x_42_);
lean_ctor_set(v___x_43_, 1, v___x_41_);
lean_ctor_set(v___x_43_, 2, v___x_40_);
return v___x_43_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__1___closed__18(void){
_start:
{
lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v___x_46_; 
v___x_44_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__1___closed__17, &lp_mathlib_List_TFAE_out___auto__1___closed__17_once, _init_lp_mathlib_List_TFAE_out___auto__1___closed__17);
v___x_45_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__5));
v___x_46_ = lean_array_push(v___x_45_, v___x_44_);
return v___x_46_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__1___closed__19(void){
_start:
{
lean_object* v___x_47_; lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; 
v___x_47_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__1___closed__18, &lp_mathlib_List_TFAE_out___auto__1___closed__18_once, _init_lp_mathlib_List_TFAE_out___auto__1___closed__18);
v___x_48_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__7));
v___x_49_ = lean_box(2);
v___x_50_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_50_, 0, v___x_49_);
lean_ctor_set(v___x_50_, 1, v___x_48_);
lean_ctor_set(v___x_50_, 2, v___x_47_);
return v___x_50_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__1___closed__20(void){
_start:
{
lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; 
v___x_51_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__1___closed__19, &lp_mathlib_List_TFAE_out___auto__1___closed__19_once, _init_lp_mathlib_List_TFAE_out___auto__1___closed__19);
v___x_52_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__5));
v___x_53_ = lean_array_push(v___x_52_, v___x_51_);
return v___x_53_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__1___closed__21(void){
_start:
{
lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; 
v___x_54_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__1___closed__20, &lp_mathlib_List_TFAE_out___auto__1___closed__20_once, _init_lp_mathlib_List_TFAE_out___auto__1___closed__20);
v___x_55_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__4));
v___x_56_ = lean_box(2);
v___x_57_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_57_, 0, v___x_56_);
lean_ctor_set(v___x_57_, 1, v___x_55_);
lean_ctor_set(v___x_57_, 2, v___x_54_);
return v___x_57_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__1(void){
_start:
{
lean_object* v___x_58_; 
v___x_58_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__1___closed__21, &lp_mathlib_List_TFAE_out___auto__1___closed__21_once, _init_lp_mathlib_List_TFAE_out___auto__1___closed__21);
return v___x_58_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__3(void){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__1___closed__21, &lp_mathlib_List_TFAE_out___auto__1___closed__21_once, _init_lp_mathlib_List_TFAE_out___auto__1___closed__21);
return v___x_59_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__2(void){
_start:
{
lean_object* v___x_66_; lean_object* v___x_67_; 
v___x_66_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__5___closed__0));
v___x_67_ = l_Lean_mkAtom(v___x_66_);
return v___x_67_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__3(void){
_start:
{
lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; 
v___x_68_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__2, &lp_mathlib_List_TFAE_out___auto__5___closed__2_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__2);
v___x_69_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__5));
v___x_70_ = lean_array_push(v___x_69_, v___x_68_);
return v___x_70_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__7(void){
_start:
{
lean_object* v___x_75_; lean_object* v___x_76_; 
v___x_75_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__5___closed__6));
v___x_76_ = l_Lean_mkAtom(v___x_75_);
return v___x_76_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__8(void){
_start:
{
lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; 
v___x_77_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__7, &lp_mathlib_List_TFAE_out___auto__5___closed__7_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__7);
v___x_78_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__5));
v___x_79_ = lean_array_push(v___x_78_, v___x_77_);
return v___x_79_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__11(void){
_start:
{
lean_object* v___x_86_; lean_object* v___x_87_; 
v___x_86_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__5___closed__9));
v___x_87_ = l_Lean_mkAtom(v___x_86_);
return v___x_87_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__12(void){
_start:
{
lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; 
v___x_88_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__11, &lp_mathlib_List_TFAE_out___auto__5___closed__11_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__11);
v___x_89_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__5));
v___x_90_ = lean_array_push(v___x_89_, v___x_88_);
return v___x_90_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__20(void){
_start:
{
lean_object* v___x_110_; lean_object* v___x_111_; 
v___x_110_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__5___closed__19));
v___x_111_ = l_Lean_mkAtom(v___x_110_);
return v___x_111_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__21(void){
_start:
{
lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; 
v___x_112_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__20, &lp_mathlib_List_TFAE_out___auto__5___closed__20_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__20);
v___x_113_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__5));
v___x_114_ = lean_array_push(v___x_113_, v___x_112_);
return v___x_114_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__23(void){
_start:
{
lean_object* v___x_116_; lean_object* v___x_117_; 
v___x_116_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__5___closed__22));
v___x_117_ = lean_string_utf8_byte_size(v___x_116_);
return v___x_117_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__24(void){
_start:
{
lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; 
v___x_118_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__23, &lp_mathlib_List_TFAE_out___auto__5___closed__23_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__23);
v___x_119_ = lean_unsigned_to_nat(0u);
v___x_120_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__5___closed__22));
v___x_121_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_121_, 0, v___x_120_);
lean_ctor_set(v___x_121_, 1, v___x_119_);
lean_ctor_set(v___x_121_, 2, v___x_118_);
return v___x_121_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__26(void){
_start:
{
lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; 
v___x_124_ = lean_box(0);
v___x_125_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__5___closed__25));
v___x_126_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__24, &lp_mathlib_List_TFAE_out___auto__5___closed__24_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__24);
v___x_127_ = lean_box(2);
v___x_128_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_128_, 0, v___x_127_);
lean_ctor_set(v___x_128_, 1, v___x_126_);
lean_ctor_set(v___x_128_, 2, v___x_125_);
lean_ctor_set(v___x_128_, 3, v___x_124_);
return v___x_128_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__27(void){
_start:
{
lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; 
v___x_129_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__26, &lp_mathlib_List_TFAE_out___auto__5___closed__26_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__26);
v___x_130_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__21, &lp_mathlib_List_TFAE_out___auto__5___closed__21_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__21);
v___x_131_ = lean_array_push(v___x_130_, v___x_129_);
return v___x_131_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__28(void){
_start:
{
lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; 
v___x_132_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__27, &lp_mathlib_List_TFAE_out___auto__5___closed__27_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__27);
v___x_133_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__5___closed__18));
v___x_134_ = lean_box(2);
v___x_135_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_135_, 0, v___x_134_);
lean_ctor_set(v___x_135_, 1, v___x_133_);
lean_ctor_set(v___x_135_, 2, v___x_132_);
return v___x_135_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__29(void){
_start:
{
lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; 
v___x_136_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__28, &lp_mathlib_List_TFAE_out___auto__5___closed__28_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__28);
v___x_137_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__5));
v___x_138_ = lean_array_push(v___x_137_, v___x_136_);
return v___x_138_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__30(void){
_start:
{
lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; 
v___x_139_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__29, &lp_mathlib_List_TFAE_out___auto__5___closed__29_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__29);
v___x_140_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__5___closed__16));
v___x_141_ = lean_box(2);
v___x_142_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_142_, 0, v___x_141_);
lean_ctor_set(v___x_142_, 1, v___x_140_);
lean_ctor_set(v___x_142_, 2, v___x_139_);
return v___x_142_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__31(void){
_start:
{
lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; 
v___x_143_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__30, &lp_mathlib_List_TFAE_out___auto__5___closed__30_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__30);
v___x_144_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__5));
v___x_145_ = lean_array_push(v___x_144_, v___x_143_);
return v___x_145_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__32(void){
_start:
{
lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; 
v___x_146_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__31, &lp_mathlib_List_TFAE_out___auto__5___closed__31_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__31);
v___x_147_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__9));
v___x_148_ = lean_box(2);
v___x_149_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_149_, 0, v___x_148_);
lean_ctor_set(v___x_149_, 1, v___x_147_);
lean_ctor_set(v___x_149_, 2, v___x_146_);
return v___x_149_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__33(void){
_start:
{
lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; 
v___x_150_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__32, &lp_mathlib_List_TFAE_out___auto__5___closed__32_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__32);
v___x_151_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__5));
v___x_152_ = lean_array_push(v___x_151_, v___x_150_);
return v___x_152_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__34(void){
_start:
{
lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; 
v___x_153_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__33, &lp_mathlib_List_TFAE_out___auto__5___closed__33_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__33);
v___x_154_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__5___closed__14));
v___x_155_ = lean_box(2);
v___x_156_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_156_, 0, v___x_155_);
lean_ctor_set(v___x_156_, 1, v___x_154_);
lean_ctor_set(v___x_156_, 2, v___x_153_);
return v___x_156_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__35(void){
_start:
{
lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; 
v___x_157_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__34, &lp_mathlib_List_TFAE_out___auto__5___closed__34_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__34);
v___x_158_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__12, &lp_mathlib_List_TFAE_out___auto__5___closed__12_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__12);
v___x_159_ = lean_array_push(v___x_158_, v___x_157_);
return v___x_159_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__36(void){
_start:
{
lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; 
v___x_160_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__35, &lp_mathlib_List_TFAE_out___auto__5___closed__35_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__35);
v___x_161_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__5___closed__10));
v___x_162_ = lean_box(2);
v___x_163_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_163_, 0, v___x_162_);
lean_ctor_set(v___x_163_, 1, v___x_161_);
lean_ctor_set(v___x_163_, 2, v___x_160_);
return v___x_163_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__37(void){
_start:
{
lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; 
v___x_164_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__36, &lp_mathlib_List_TFAE_out___auto__5___closed__36_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__36);
v___x_165_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__5));
v___x_166_ = lean_array_push(v___x_165_, v___x_164_);
return v___x_166_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__38(void){
_start:
{
lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; 
v___x_167_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__37, &lp_mathlib_List_TFAE_out___auto__5___closed__37_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__37);
v___x_168_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__9));
v___x_169_ = lean_box(2);
v___x_170_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_170_, 0, v___x_169_);
lean_ctor_set(v___x_170_, 1, v___x_168_);
lean_ctor_set(v___x_170_, 2, v___x_167_);
return v___x_170_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__39(void){
_start:
{
lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; 
v___x_171_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__38, &lp_mathlib_List_TFAE_out___auto__5___closed__38_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__38);
v___x_172_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__5));
v___x_173_ = lean_array_push(v___x_172_, v___x_171_);
return v___x_173_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__40(void){
_start:
{
lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; 
v___x_174_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__39, &lp_mathlib_List_TFAE_out___auto__5___closed__39_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__39);
v___x_175_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__7));
v___x_176_ = lean_box(2);
v___x_177_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_177_, 0, v___x_176_);
lean_ctor_set(v___x_177_, 1, v___x_175_);
lean_ctor_set(v___x_177_, 2, v___x_174_);
return v___x_177_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__41(void){
_start:
{
lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; 
v___x_178_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__40, &lp_mathlib_List_TFAE_out___auto__5___closed__40_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__40);
v___x_179_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__5));
v___x_180_ = lean_array_push(v___x_179_, v___x_178_);
return v___x_180_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__42(void){
_start:
{
lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; 
v___x_181_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__41, &lp_mathlib_List_TFAE_out___auto__5___closed__41_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__41);
v___x_182_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__4));
v___x_183_ = lean_box(2);
v___x_184_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_184_, 0, v___x_183_);
lean_ctor_set(v___x_184_, 1, v___x_182_);
lean_ctor_set(v___x_184_, 2, v___x_181_);
return v___x_184_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__43(void){
_start:
{
lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; 
v___x_185_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__42, &lp_mathlib_List_TFAE_out___auto__5___closed__42_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__42);
v___x_186_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__8, &lp_mathlib_List_TFAE_out___auto__5___closed__8_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__8);
v___x_187_ = lean_array_push(v___x_186_, v___x_185_);
return v___x_187_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__44(void){
_start:
{
lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; 
v___x_188_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__43, &lp_mathlib_List_TFAE_out___auto__5___closed__43_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__43);
v___x_189_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__5___closed__5));
v___x_190_ = lean_box(2);
v___x_191_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_191_, 0, v___x_190_);
lean_ctor_set(v___x_191_, 1, v___x_189_);
lean_ctor_set(v___x_191_, 2, v___x_188_);
return v___x_191_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__45(void){
_start:
{
lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; 
v___x_192_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__44, &lp_mathlib_List_TFAE_out___auto__5___closed__44_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__44);
v___x_193_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__5));
v___x_194_ = lean_array_push(v___x_193_, v___x_192_);
return v___x_194_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__48(void){
_start:
{
lean_object* v___x_201_; lean_object* v___x_202_; 
v___x_201_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__5___closed__46));
v___x_202_ = l_Lean_mkAtom(v___x_201_);
return v___x_202_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__49(void){
_start:
{
lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; 
v___x_203_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__48, &lp_mathlib_List_TFAE_out___auto__5___closed__48_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__48);
v___x_204_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__5));
v___x_205_ = lean_array_push(v___x_204_, v___x_203_);
return v___x_205_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__53(void){
_start:
{
lean_object* v___x_210_; lean_object* v___x_211_; 
v___x_210_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__5___closed__52));
v___x_211_ = l_Lean_mkAtom(v___x_210_);
return v___x_211_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__54(void){
_start:
{
lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; 
v___x_212_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__53, &lp_mathlib_List_TFAE_out___auto__5___closed__53_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__53);
v___x_213_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__5));
v___x_214_ = lean_array_push(v___x_213_, v___x_212_);
return v___x_214_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__55(void){
_start:
{
lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; 
v___x_215_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__54, &lp_mathlib_List_TFAE_out___auto__5___closed__54_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__54);
v___x_216_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__5___closed__51));
v___x_217_ = lean_box(2);
v___x_218_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_218_, 0, v___x_217_);
lean_ctor_set(v___x_218_, 1, v___x_216_);
lean_ctor_set(v___x_218_, 2, v___x_215_);
return v___x_218_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__56(void){
_start:
{
lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; 
v___x_219_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__55, &lp_mathlib_List_TFAE_out___auto__5___closed__55_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__55);
v___x_220_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__5));
v___x_221_ = lean_array_push(v___x_220_, v___x_219_);
return v___x_221_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__57(void){
_start:
{
lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; 
v___x_222_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__56, &lp_mathlib_List_TFAE_out___auto__5___closed__56_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__56);
v___x_223_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__9));
v___x_224_ = lean_box(2);
v___x_225_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_225_, 0, v___x_224_);
lean_ctor_set(v___x_225_, 1, v___x_223_);
lean_ctor_set(v___x_225_, 2, v___x_222_);
return v___x_225_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__58(void){
_start:
{
lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; 
v___x_226_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__57, &lp_mathlib_List_TFAE_out___auto__5___closed__57_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__57);
v___x_227_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__49, &lp_mathlib_List_TFAE_out___auto__5___closed__49_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__49);
v___x_228_ = lean_array_push(v___x_227_, v___x_226_);
return v___x_228_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__59(void){
_start:
{
lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; 
v___x_229_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__58, &lp_mathlib_List_TFAE_out___auto__5___closed__58_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__58);
v___x_230_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__5___closed__47));
v___x_231_ = lean_box(2);
v___x_232_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_232_, 0, v___x_231_);
lean_ctor_set(v___x_232_, 1, v___x_230_);
lean_ctor_set(v___x_232_, 2, v___x_229_);
return v___x_232_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__60(void){
_start:
{
lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; 
v___x_233_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__59, &lp_mathlib_List_TFAE_out___auto__5___closed__59_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__59);
v___x_234_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__5));
v___x_235_ = lean_array_push(v___x_234_, v___x_233_);
return v___x_235_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__61(void){
_start:
{
lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; 
v___x_236_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__60, &lp_mathlib_List_TFAE_out___auto__5___closed__60_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__60);
v___x_237_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__9));
v___x_238_ = lean_box(2);
v___x_239_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_239_, 0, v___x_238_);
lean_ctor_set(v___x_239_, 1, v___x_237_);
lean_ctor_set(v___x_239_, 2, v___x_236_);
return v___x_239_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__62(void){
_start:
{
lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; 
v___x_240_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__61, &lp_mathlib_List_TFAE_out___auto__5___closed__61_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__61);
v___x_241_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__5));
v___x_242_ = lean_array_push(v___x_241_, v___x_240_);
return v___x_242_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__63(void){
_start:
{
lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; 
v___x_243_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__62, &lp_mathlib_List_TFAE_out___auto__5___closed__62_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__62);
v___x_244_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__7));
v___x_245_ = lean_box(2);
v___x_246_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_246_, 0, v___x_245_);
lean_ctor_set(v___x_246_, 1, v___x_244_);
lean_ctor_set(v___x_246_, 2, v___x_243_);
return v___x_246_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__64(void){
_start:
{
lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; 
v___x_247_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__63, &lp_mathlib_List_TFAE_out___auto__5___closed__63_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__63);
v___x_248_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__5));
v___x_249_ = lean_array_push(v___x_248_, v___x_247_);
return v___x_249_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__65(void){
_start:
{
lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; 
v___x_250_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__64, &lp_mathlib_List_TFAE_out___auto__5___closed__64_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__64);
v___x_251_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__4));
v___x_252_ = lean_box(2);
v___x_253_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_253_, 0, v___x_252_);
lean_ctor_set(v___x_253_, 1, v___x_251_);
lean_ctor_set(v___x_253_, 2, v___x_250_);
return v___x_253_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__66(void){
_start:
{
lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; 
v___x_254_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__65, &lp_mathlib_List_TFAE_out___auto__5___closed__65_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__65);
v___x_255_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__8, &lp_mathlib_List_TFAE_out___auto__5___closed__8_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__8);
v___x_256_ = lean_array_push(v___x_255_, v___x_254_);
return v___x_256_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__67(void){
_start:
{
lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; 
v___x_257_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__66, &lp_mathlib_List_TFAE_out___auto__5___closed__66_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__66);
v___x_258_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__5___closed__5));
v___x_259_ = lean_box(2);
v___x_260_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_260_, 0, v___x_259_);
lean_ctor_set(v___x_260_, 1, v___x_258_);
lean_ctor_set(v___x_260_, 2, v___x_257_);
return v___x_260_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__68(void){
_start:
{
lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; 
v___x_261_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__67, &lp_mathlib_List_TFAE_out___auto__5___closed__67_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__67);
v___x_262_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__45, &lp_mathlib_List_TFAE_out___auto__5___closed__45_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__45);
v___x_263_ = lean_array_push(v___x_262_, v___x_261_);
return v___x_263_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__69(void){
_start:
{
lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; 
v___x_264_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__68, &lp_mathlib_List_TFAE_out___auto__5___closed__68_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__68);
v___x_265_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__9));
v___x_266_ = lean_box(2);
v___x_267_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_267_, 0, v___x_266_);
lean_ctor_set(v___x_267_, 1, v___x_265_);
lean_ctor_set(v___x_267_, 2, v___x_264_);
return v___x_267_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__70(void){
_start:
{
lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; 
v___x_268_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__69, &lp_mathlib_List_TFAE_out___auto__5___closed__69_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__69);
v___x_269_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__3, &lp_mathlib_List_TFAE_out___auto__5___closed__3_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__3);
v___x_270_ = lean_array_push(v___x_269_, v___x_268_);
return v___x_270_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__71(void){
_start:
{
lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; 
v___x_271_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__70, &lp_mathlib_List_TFAE_out___auto__5___closed__70_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__70);
v___x_272_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__5___closed__1));
v___x_273_ = lean_box(2);
v___x_274_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_274_, 0, v___x_273_);
lean_ctor_set(v___x_274_, 1, v___x_272_);
lean_ctor_set(v___x_274_, 2, v___x_271_);
return v___x_274_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__72(void){
_start:
{
lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; 
v___x_275_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__71, &lp_mathlib_List_TFAE_out___auto__5___closed__71_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__71);
v___x_276_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__5));
v___x_277_ = lean_array_push(v___x_276_, v___x_275_);
return v___x_277_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__73(void){
_start:
{
lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; 
v___x_278_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__72, &lp_mathlib_List_TFAE_out___auto__5___closed__72_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__72);
v___x_279_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__9));
v___x_280_ = lean_box(2);
v___x_281_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_281_, 0, v___x_280_);
lean_ctor_set(v___x_281_, 1, v___x_279_);
lean_ctor_set(v___x_281_, 2, v___x_278_);
return v___x_281_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__74(void){
_start:
{
lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; 
v___x_282_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__73, &lp_mathlib_List_TFAE_out___auto__5___closed__73_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__73);
v___x_283_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__5));
v___x_284_ = lean_array_push(v___x_283_, v___x_282_);
return v___x_284_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__75(void){
_start:
{
lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; 
v___x_285_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__74, &lp_mathlib_List_TFAE_out___auto__5___closed__74_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__74);
v___x_286_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__7));
v___x_287_ = lean_box(2);
v___x_288_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_288_, 0, v___x_287_);
lean_ctor_set(v___x_288_, 1, v___x_286_);
lean_ctor_set(v___x_288_, 2, v___x_285_);
return v___x_288_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__76(void){
_start:
{
lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; 
v___x_289_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__75, &lp_mathlib_List_TFAE_out___auto__5___closed__75_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__75);
v___x_290_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__5));
v___x_291_ = lean_array_push(v___x_290_, v___x_289_);
return v___x_291_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5___closed__77(void){
_start:
{
lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; 
v___x_292_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__76, &lp_mathlib_List_TFAE_out___auto__5___closed__76_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__76);
v___x_293_ = ((lean_object*)(lp_mathlib_List_TFAE_out___auto__1___closed__4));
v___x_294_ = lean_box(2);
v___x_295_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_295_, 0, v___x_294_);
lean_ctor_set(v___x_295_, 1, v___x_293_);
lean_ctor_set(v___x_295_, 2, v___x_292_);
return v___x_295_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__5(void){
_start:
{
lean_object* v___x_296_; 
v___x_296_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__77, &lp_mathlib_List_TFAE_out___auto__5___closed__77_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__77);
return v___x_296_;
}
}
static lean_object* _init_lp_mathlib_List_TFAE_out___auto__7(void){
_start:
{
lean_object* v___x_297_; 
v___x_297_ = lean_obj_once(&lp_mathlib_List_TFAE_out___auto__5___closed__77, &lp_mathlib_List_TFAE_out___auto__5___closed__77_once, _init_lp_mathlib_List_TFAE_out___auto__5___closed__77);
return v___x_297_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Alias(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_List_TFAE(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Alias(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_List_TFAE(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_List_TFAE_out___auto__1 = _init_lp_mathlib_List_TFAE_out___auto__1();
lean_mark_persistent(lp_mathlib_List_TFAE_out___auto__1);
lp_mathlib_List_TFAE_out___auto__3 = _init_lp_mathlib_List_TFAE_out___auto__3();
lean_mark_persistent(lp_mathlib_List_TFAE_out___auto__3);
lp_mathlib_List_TFAE_out___auto__5 = _init_lp_mathlib_List_TFAE_out___auto__5();
lean_mark_persistent(lp_mathlib_List_TFAE_out___auto__5);
lp_mathlib_List_TFAE_out___auto__7 = _init_lp_mathlib_List_TFAE_out___auto__7();
lean_mark_persistent(lp_mathlib_List_TFAE_out___auto__7);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Alias(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_List_TFAE(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Alias(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_TFAE(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_List_TFAE(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_List_TFAE(builtin);
}
#ifdef __cplusplus
}
#endif
