// Lean compiler output
// Module: Mathlib.Order.ConditionallyCompleteLattice.Finset
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Max public import Mathlib.Data.Set.Finite.Lattice public import Mathlib.Order.ConditionallyCompleteLattice.Indexed
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
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__0 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__0_value;
static const lean_string_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__1 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__1_value;
static const lean_string_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__2 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__2_value;
static const lean_string_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__3 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__4 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__4_value;
static const lean_array_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5_value;
static const lean_string_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__6 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__7_value_aux_1),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__7_value_aux_2),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__7 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__7_value;
static const lean_string_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__8 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__9 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__9_value;
static const lean_string_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "exact"};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__10 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__10_value;
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__11_value_aux_1),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__11_value_aux_2),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(108, 106, 111, 83, 219, 207, 32, 208)}};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__11 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__11_value;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__12;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__13;
static const lean_string_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__14 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__14_value;
static const lean_string_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__15 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__15_value;
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__16_value_aux_1),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__16_value_aux_2),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__16 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__16_value;
static const lean_string_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "image_nonempty.mpr"};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__17 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__17_value;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__18;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__19;
static const lean_string_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "image_nonempty"};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__20 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__20_value;
static const lean_string_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "mpr"};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__21 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__21_value;
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__20_value),LEAN_SCALAR_PTR_LITERAL(42, 44, 209, 98, 103, 49, 241, 191)}};
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__22_value_aux_0),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(155, 162, 49, 202, 135, 181, 117, 230)}};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__22 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__22_value;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__23;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__24;
static const lean_string_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__25 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__25_value;
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__26_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__26_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__26_value_aux_0),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__26_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__26_value_aux_1),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__26_value_aux_2),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__25_value),LEAN_SCALAR_PTR_LITERAL(124, 9, 161, 194, 227, 100, 20, 110)}};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__26 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__26_value;
static const lean_string_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__27 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__27_value;
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__28_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__28_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__28_value_aux_0),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__28_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__28_value_aux_1),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__28_value_aux_2),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__27_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__28 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__28_value;
static const lean_string_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__29 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__29_value;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__30;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__31;
static const lean_string_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__32 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__32_value;
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__32_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__33 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__33_value;
static const lean_string_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "[anonymous]"};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__34 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__34_value;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__35_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__35;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__36_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__36;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__37_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__37;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__38_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__38;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__39_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__39;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__40_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__40;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__41_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__41;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__42_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__42;
static const lean_string_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "h.imp"};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__43 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__43_value;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__44_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__44;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__45_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__45;
static const lean_string_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "h"};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__46 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__46_value;
static const lean_string_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "imp"};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__47 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__47_value;
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__48_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__46_value),LEAN_SCALAR_PTR_LITERAL(176, 181, 207, 77, 197, 87, 68, 121)}};
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__48_value_aux_0),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__47_value),LEAN_SCALAR_PTR_LITERAL(36, 18, 112, 112, 16, 19, 143, 3)}};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__48 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__48_value;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__49_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__49;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__50_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__50;
static const lean_string_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "fun"};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__51 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__51_value;
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__52_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__52_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__52_value_aux_0),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__52_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__52_value_aux_1),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__52_value_aux_2),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__51_value),LEAN_SCALAR_PTR_LITERAL(249, 155, 133, 242, 71, 132, 191, 97)}};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__52 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__52_value;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__53_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__53;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__54_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__54;
static const lean_string_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "basicFun"};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__55 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__55_value;
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__56_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__56_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__56_value_aux_0),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__56_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__56_value_aux_1),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__56_value_aux_2),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__55_value),LEAN_SCALAR_PTR_LITERAL(209, 134, 40, 160, 122, 195, 31, 223)}};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__56 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__56_value;
static const lean_string_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hole"};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__57 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__57_value;
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__58_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__58_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__58_value_aux_0),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__58_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__58_value_aux_1),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__58_value_aux_2),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__57_value),LEAN_SCALAR_PTR_LITERAL(135, 134, 219, 115, 97, 130, 74, 55)}};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__58 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__58_value;
static const lean_string_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__59 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__59_value;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__60_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__60;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__61_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__61;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__62_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__62;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__63_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__63;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__64_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__64;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__65_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__65;
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__66_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__9_value),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5_value)}};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__66 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__66_value;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__67_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__67;
static const lean_string_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__68_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "↦"};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__68 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__68_value;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__69_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__69;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__70_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__70;
static const lean_string_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__71_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "And.left"};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__71 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__71_value;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__72_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__72;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__73_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__73;
static const lean_string_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__74_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "And"};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__74 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__74_value;
static const lean_string_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__75_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "left"};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__75 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__75_value;
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__76_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__74_value),LEAN_SCALAR_PTR_LITERAL(49, 220, 212, 156, 122, 214, 55, 135)}};
static const lean_ctor_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__76_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__76_value_aux_0),((lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__75_value),LEAN_SCALAR_PTR_LITERAL(12, 252, 227, 83, 88, 185, 40, 148)}};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__76 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__76_value;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__77_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__77;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__78_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__78;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__79_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__79;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__80_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__80;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__81_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__81;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__82_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__82;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__83_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__83;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__84_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__84;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__85_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__85;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__86_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__86;
static const lean_string_object lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__87_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__87 = (const lean_object*)&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__87_value;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__88_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__88;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__89_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__89;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__90_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__90;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__91_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__91;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__92_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__92;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__93_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__93;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__94_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__94;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__95_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__95;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__96_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__96;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__97_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__97;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__98_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__98;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__99_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__99;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__100_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__100;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__101_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__101;
static lean_once_cell_t lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__102_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__102;
LEAN_EXPORT lean_object* lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Finset_ciInf__eq__min_x27__image___auto__1;
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__12(void){
_start:
{
lean_object* v___x_27_; lean_object* v___x_28_; 
v___x_27_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__10));
v___x_28_ = l_Lean_mkAtom(v___x_27_);
return v___x_28_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__13(void){
_start:
{
lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v___x_31_; 
v___x_29_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__12, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__12_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__12);
v___x_30_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5));
v___x_31_ = lean_array_push(v___x_30_, v___x_29_);
return v___x_31_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__18(void){
_start:
{
lean_object* v___x_40_; lean_object* v___x_41_; 
v___x_40_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__17));
v___x_41_ = lean_string_utf8_byte_size(v___x_40_);
return v___x_41_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__19(void){
_start:
{
lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; 
v___x_42_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__18, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__18_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__18);
v___x_43_ = lean_unsigned_to_nat(0u);
v___x_44_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__17));
v___x_45_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_45_, 0, v___x_44_);
lean_ctor_set(v___x_45_, 1, v___x_43_);
lean_ctor_set(v___x_45_, 2, v___x_42_);
return v___x_45_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__23(void){
_start:
{
lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; 
v___x_51_ = lean_box(0);
v___x_52_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__22));
v___x_53_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__19, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__19_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__19);
v___x_54_ = lean_box(2);
v___x_55_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_55_, 0, v___x_54_);
lean_ctor_set(v___x_55_, 1, v___x_53_);
lean_ctor_set(v___x_55_, 2, v___x_52_);
lean_ctor_set(v___x_55_, 3, v___x_51_);
return v___x_55_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__24(void){
_start:
{
lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; 
v___x_56_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__23, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__23_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__23);
v___x_57_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5));
v___x_58_ = lean_array_push(v___x_57_, v___x_56_);
return v___x_58_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__30(void){
_start:
{
lean_object* v___x_72_; lean_object* v___x_73_; 
v___x_72_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__29));
v___x_73_ = l_Lean_mkAtom(v___x_72_);
return v___x_73_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__31(void){
_start:
{
lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; 
v___x_74_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__30, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__30_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__30);
v___x_75_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5));
v___x_76_ = lean_array_push(v___x_75_, v___x_74_);
return v___x_76_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__35(void){
_start:
{
lean_object* v___x_81_; lean_object* v___x_82_; 
v___x_81_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__34));
v___x_82_ = lean_string_utf8_byte_size(v___x_81_);
return v___x_82_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__36(void){
_start:
{
lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; 
v___x_83_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__35, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__35_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__35);
v___x_84_ = lean_unsigned_to_nat(0u);
v___x_85_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__34));
v___x_86_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_86_, 0, v___x_85_);
lean_ctor_set(v___x_86_, 1, v___x_84_);
lean_ctor_set(v___x_86_, 2, v___x_83_);
return v___x_86_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__37(void){
_start:
{
lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; 
v___x_87_ = lean_box(0);
v___x_88_ = lean_box(0);
v___x_89_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__36, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__36_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__36);
v___x_90_ = lean_box(2);
v___x_91_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_91_, 0, v___x_90_);
lean_ctor_set(v___x_91_, 1, v___x_89_);
lean_ctor_set(v___x_91_, 2, v___x_88_);
lean_ctor_set(v___x_91_, 3, v___x_87_);
return v___x_91_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__38(void){
_start:
{
lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; 
v___x_92_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__37, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__37_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__37);
v___x_93_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5));
v___x_94_ = lean_array_push(v___x_93_, v___x_92_);
return v___x_94_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__39(void){
_start:
{
lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; 
v___x_95_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__38, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__38_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__38);
v___x_96_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__33));
v___x_97_ = lean_box(2);
v___x_98_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_98_, 0, v___x_97_);
lean_ctor_set(v___x_98_, 1, v___x_96_);
lean_ctor_set(v___x_98_, 2, v___x_95_);
return v___x_98_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__40(void){
_start:
{
lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; 
v___x_99_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__39, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__39_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__39);
v___x_100_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__31, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__31_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__31);
v___x_101_ = lean_array_push(v___x_100_, v___x_99_);
return v___x_101_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__41(void){
_start:
{
lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; 
v___x_102_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__40, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__40_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__40);
v___x_103_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__28));
v___x_104_ = lean_box(2);
v___x_105_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_105_, 0, v___x_104_);
lean_ctor_set(v___x_105_, 1, v___x_103_);
lean_ctor_set(v___x_105_, 2, v___x_102_);
return v___x_105_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__42(void){
_start:
{
lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; 
v___x_106_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__41, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__41_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__41);
v___x_107_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5));
v___x_108_ = lean_array_push(v___x_107_, v___x_106_);
return v___x_108_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__44(void){
_start:
{
lean_object* v___x_110_; lean_object* v___x_111_; 
v___x_110_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__43));
v___x_111_ = lean_string_utf8_byte_size(v___x_110_);
return v___x_111_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__45(void){
_start:
{
lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; 
v___x_112_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__44, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__44_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__44);
v___x_113_ = lean_unsigned_to_nat(0u);
v___x_114_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__43));
v___x_115_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_115_, 0, v___x_114_);
lean_ctor_set(v___x_115_, 1, v___x_113_);
lean_ctor_set(v___x_115_, 2, v___x_112_);
return v___x_115_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__49(void){
_start:
{
lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; 
v___x_121_ = lean_box(0);
v___x_122_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__48));
v___x_123_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__45, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__45_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__45);
v___x_124_ = lean_box(2);
v___x_125_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_125_, 0, v___x_124_);
lean_ctor_set(v___x_125_, 1, v___x_123_);
lean_ctor_set(v___x_125_, 2, v___x_122_);
lean_ctor_set(v___x_125_, 3, v___x_121_);
return v___x_125_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__50(void){
_start:
{
lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; 
v___x_126_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__49, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__49_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__49);
v___x_127_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5));
v___x_128_ = lean_array_push(v___x_127_, v___x_126_);
return v___x_128_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__53(void){
_start:
{
lean_object* v___x_135_; lean_object* v___x_136_; 
v___x_135_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__51));
v___x_136_ = l_Lean_mkAtom(v___x_135_);
return v___x_136_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__54(void){
_start:
{
lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; 
v___x_137_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__53, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__53_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__53);
v___x_138_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5));
v___x_139_ = lean_array_push(v___x_138_, v___x_137_);
return v___x_139_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__60(void){
_start:
{
lean_object* v___x_153_; lean_object* v___x_154_; 
v___x_153_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__59));
v___x_154_ = l_Lean_mkAtom(v___x_153_);
return v___x_154_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__61(void){
_start:
{
lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; 
v___x_155_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__60, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__60_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__60);
v___x_156_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5));
v___x_157_ = lean_array_push(v___x_156_, v___x_155_);
return v___x_157_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__62(void){
_start:
{
lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; 
v___x_158_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__61, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__61_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__61);
v___x_159_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__58));
v___x_160_ = lean_box(2);
v___x_161_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_161_, 0, v___x_160_);
lean_ctor_set(v___x_161_, 1, v___x_159_);
lean_ctor_set(v___x_161_, 2, v___x_158_);
return v___x_161_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__63(void){
_start:
{
lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; 
v___x_162_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__62, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__62_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__62);
v___x_163_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5));
v___x_164_ = lean_array_push(v___x_163_, v___x_162_);
return v___x_164_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__64(void){
_start:
{
lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; 
v___x_165_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__63, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__63_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__63);
v___x_166_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__9));
v___x_167_ = lean_box(2);
v___x_168_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_168_, 0, v___x_167_);
lean_ctor_set(v___x_168_, 1, v___x_166_);
lean_ctor_set(v___x_168_, 2, v___x_165_);
return v___x_168_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__65(void){
_start:
{
lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; 
v___x_169_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__64, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__64_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__64);
v___x_170_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5));
v___x_171_ = lean_array_push(v___x_170_, v___x_169_);
return v___x_171_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__67(void){
_start:
{
lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; 
v___x_176_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__66));
v___x_177_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__65, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__65_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__65);
v___x_178_ = lean_array_push(v___x_177_, v___x_176_);
return v___x_178_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__69(void){
_start:
{
lean_object* v___x_180_; lean_object* v___x_181_; 
v___x_180_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__68));
v___x_181_ = l_Lean_mkAtom(v___x_180_);
return v___x_181_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__70(void){
_start:
{
lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; 
v___x_182_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__69, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__69_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__69);
v___x_183_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__67, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__67_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__67);
v___x_184_ = lean_array_push(v___x_183_, v___x_182_);
return v___x_184_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__72(void){
_start:
{
lean_object* v___x_186_; lean_object* v___x_187_; 
v___x_186_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__71));
v___x_187_ = lean_string_utf8_byte_size(v___x_186_);
return v___x_187_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__73(void){
_start:
{
lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; 
v___x_188_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__72, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__72_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__72);
v___x_189_ = lean_unsigned_to_nat(0u);
v___x_190_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__71));
v___x_191_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_191_, 0, v___x_190_);
lean_ctor_set(v___x_191_, 1, v___x_189_);
lean_ctor_set(v___x_191_, 2, v___x_188_);
return v___x_191_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__77(void){
_start:
{
lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; 
v___x_197_ = lean_box(0);
v___x_198_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__76));
v___x_199_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__73, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__73_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__73);
v___x_200_ = lean_box(2);
v___x_201_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_201_, 0, v___x_200_);
lean_ctor_set(v___x_201_, 1, v___x_199_);
lean_ctor_set(v___x_201_, 2, v___x_198_);
lean_ctor_set(v___x_201_, 3, v___x_197_);
return v___x_201_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__78(void){
_start:
{
lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; 
v___x_202_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__77, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__77_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__77);
v___x_203_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__70, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__70_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__70);
v___x_204_ = lean_array_push(v___x_203_, v___x_202_);
return v___x_204_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__79(void){
_start:
{
lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; 
v___x_205_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__78, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__78_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__78);
v___x_206_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__56));
v___x_207_ = lean_box(2);
v___x_208_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_208_, 0, v___x_207_);
lean_ctor_set(v___x_208_, 1, v___x_206_);
lean_ctor_set(v___x_208_, 2, v___x_205_);
return v___x_208_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__80(void){
_start:
{
lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; 
v___x_209_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__79, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__79_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__79);
v___x_210_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__54, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__54_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__54);
v___x_211_ = lean_array_push(v___x_210_, v___x_209_);
return v___x_211_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__81(void){
_start:
{
lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; 
v___x_212_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__80, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__80_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__80);
v___x_213_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__52));
v___x_214_ = lean_box(2);
v___x_215_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_215_, 0, v___x_214_);
lean_ctor_set(v___x_215_, 1, v___x_213_);
lean_ctor_set(v___x_215_, 2, v___x_212_);
return v___x_215_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__82(void){
_start:
{
lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; 
v___x_216_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__81, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__81_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__81);
v___x_217_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5));
v___x_218_ = lean_array_push(v___x_217_, v___x_216_);
return v___x_218_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__83(void){
_start:
{
lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; 
v___x_219_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__82, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__82_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__82);
v___x_220_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__9));
v___x_221_ = lean_box(2);
v___x_222_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_222_, 0, v___x_221_);
lean_ctor_set(v___x_222_, 1, v___x_220_);
lean_ctor_set(v___x_222_, 2, v___x_219_);
return v___x_222_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__84(void){
_start:
{
lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; 
v___x_223_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__83, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__83_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__83);
v___x_224_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__50, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__50_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__50);
v___x_225_ = lean_array_push(v___x_224_, v___x_223_);
return v___x_225_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__85(void){
_start:
{
lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; 
v___x_226_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__84, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__84_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__84);
v___x_227_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__16));
v___x_228_ = lean_box(2);
v___x_229_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_229_, 0, v___x_228_);
lean_ctor_set(v___x_229_, 1, v___x_227_);
lean_ctor_set(v___x_229_, 2, v___x_226_);
return v___x_229_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__86(void){
_start:
{
lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; 
v___x_230_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__85, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__85_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__85);
v___x_231_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__42, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__42_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__42);
v___x_232_ = lean_array_push(v___x_231_, v___x_230_);
return v___x_232_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__88(void){
_start:
{
lean_object* v___x_234_; lean_object* v___x_235_; 
v___x_234_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__87));
v___x_235_ = l_Lean_mkAtom(v___x_234_);
return v___x_235_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__89(void){
_start:
{
lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; 
v___x_236_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__88, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__88_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__88);
v___x_237_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__86, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__86_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__86);
v___x_238_ = lean_array_push(v___x_237_, v___x_236_);
return v___x_238_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__90(void){
_start:
{
lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; 
v___x_239_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__89, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__89_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__89);
v___x_240_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__26));
v___x_241_ = lean_box(2);
v___x_242_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_242_, 0, v___x_241_);
lean_ctor_set(v___x_242_, 1, v___x_240_);
lean_ctor_set(v___x_242_, 2, v___x_239_);
return v___x_242_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__91(void){
_start:
{
lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; 
v___x_243_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__90, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__90_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__90);
v___x_244_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5));
v___x_245_ = lean_array_push(v___x_244_, v___x_243_);
return v___x_245_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__92(void){
_start:
{
lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; 
v___x_246_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__91, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__91_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__91);
v___x_247_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__9));
v___x_248_ = lean_box(2);
v___x_249_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_249_, 0, v___x_248_);
lean_ctor_set(v___x_249_, 1, v___x_247_);
lean_ctor_set(v___x_249_, 2, v___x_246_);
return v___x_249_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__93(void){
_start:
{
lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; 
v___x_250_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__92, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__92_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__92);
v___x_251_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__24, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__24_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__24);
v___x_252_ = lean_array_push(v___x_251_, v___x_250_);
return v___x_252_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__94(void){
_start:
{
lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; 
v___x_253_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__93, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__93_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__93);
v___x_254_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__16));
v___x_255_ = lean_box(2);
v___x_256_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_256_, 0, v___x_255_);
lean_ctor_set(v___x_256_, 1, v___x_254_);
lean_ctor_set(v___x_256_, 2, v___x_253_);
return v___x_256_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__95(void){
_start:
{
lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; 
v___x_257_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__94, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__94_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__94);
v___x_258_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__13, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__13_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__13);
v___x_259_ = lean_array_push(v___x_258_, v___x_257_);
return v___x_259_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__96(void){
_start:
{
lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; 
v___x_260_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__95, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__95_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__95);
v___x_261_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__11));
v___x_262_ = lean_box(2);
v___x_263_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_263_, 0, v___x_262_);
lean_ctor_set(v___x_263_, 1, v___x_261_);
lean_ctor_set(v___x_263_, 2, v___x_260_);
return v___x_263_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__97(void){
_start:
{
lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; 
v___x_264_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__96, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__96_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__96);
v___x_265_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5));
v___x_266_ = lean_array_push(v___x_265_, v___x_264_);
return v___x_266_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__98(void){
_start:
{
lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; 
v___x_267_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__97, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__97_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__97);
v___x_268_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__9));
v___x_269_ = lean_box(2);
v___x_270_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_270_, 0, v___x_269_);
lean_ctor_set(v___x_270_, 1, v___x_268_);
lean_ctor_set(v___x_270_, 2, v___x_267_);
return v___x_270_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__99(void){
_start:
{
lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; 
v___x_271_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__98, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__98_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__98);
v___x_272_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5));
v___x_273_ = lean_array_push(v___x_272_, v___x_271_);
return v___x_273_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__100(void){
_start:
{
lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; 
v___x_274_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__99, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__99_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__99);
v___x_275_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__7));
v___x_276_ = lean_box(2);
v___x_277_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_277_, 0, v___x_276_);
lean_ctor_set(v___x_277_, 1, v___x_275_);
lean_ctor_set(v___x_277_, 2, v___x_274_);
return v___x_277_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__101(void){
_start:
{
lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; 
v___x_278_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__100, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__100_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__100);
v___x_279_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__5));
v___x_280_ = lean_array_push(v___x_279_, v___x_278_);
return v___x_280_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__102(void){
_start:
{
lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; 
v___x_281_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__101, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__101_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__101);
v___x_282_ = ((lean_object*)(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__4));
v___x_283_ = lean_box(2);
v___x_284_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_284_, 0, v___x_283_);
lean_ctor_set(v___x_284_, 1, v___x_282_);
lean_ctor_set(v___x_284_, 2, v___x_281_);
return v___x_284_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1(void){
_start:
{
lean_object* v___x_285_; 
v___x_285_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__102, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__102_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__102);
return v___x_285_;
}
}
static lean_object* _init_lp_mathlib_Finset_ciInf__eq__min_x27__image___auto__1(void){
_start:
{
lean_object* v___x_286_; 
v___x_286_ = lean_obj_once(&lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__102, &lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__102_once, _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1___closed__102);
return v___x_286_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Max(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Lattice(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Indexed(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Finset(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Max(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Indexed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Finset(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1 = _init_lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1();
lean_mark_persistent(lp_mathlib_Finset_ciSup__eq__max_x27__image___auto__1);
lp_mathlib_Finset_ciInf__eq__min_x27__image___auto__1 = _init_lp_mathlib_Finset_ciInf__eq__min_x27__image___auto__1();
lean_mark_persistent(lp_mathlib_Finset_ciInf__eq__min_x27__image___auto__1);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Max(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Lattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Indexed(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Finset(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Max(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Finite_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Indexed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Finset(builtin);
}
#ifdef __cplusplus
}
#endif
