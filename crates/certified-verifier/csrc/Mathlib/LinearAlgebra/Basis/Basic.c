// Lean compiler output
// Module: Mathlib.LinearAlgebra.Basis.Basic
// Imports: public import Init public meta import Init public import Mathlib.LinearAlgebra.Basis.Defs public import Mathlib.LinearAlgebra.LinearIndependent.Basic public import Mathlib.LinearAlgebra.Span.Basic
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
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__0 = (const lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__0_value;
static const lean_string_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__1 = (const lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__1_value;
static const lean_string_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__2 = (const lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__2_value;
static const lean_string_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__3 = (const lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__4 = (const lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__4_value;
static const lean_array_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__5 = (const lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__5_value;
static const lean_string_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__6 = (const lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__7_value_aux_1),((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__7_value_aux_2),((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__7 = (const lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__7_value;
static const lean_string_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__8 = (const lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__9 = (const lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__9_value;
static const lean_string_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__10 = (const lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__10_value;
static const lean_ctor_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__11_value_aux_1),((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__11_value_aux_2),((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(50, 13, 241, 145, 67, 153, 105, 177)}};
static const lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__11 = (const lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__11_value;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__12;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__13;
static const lean_string_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__14 = (const lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__14_value;
static const lean_ctor_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__15_value_aux_0),((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__15_value_aux_1),((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__15_value_aux_2),((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__15 = (const lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__15_value;
static const lean_ctor_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__9_value),((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__5_value)}};
static const lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__16 = (const lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__16_value;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__17;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__18;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__19;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__20;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__21;
static const lean_string_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__22 = (const lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__22_value;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__23;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__24;
static const lean_string_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "simpLemma"};
static const lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__25 = (const lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__25_value;
static const lean_ctor_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__26_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__26_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__26_value_aux_0),((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__26_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__26_value_aux_1),((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__26_value_aux_2),((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__25_value),LEAN_SCALAR_PTR_LITERAL(38, 215, 101, 250, 181, 108, 118, 102)}};
static const lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__26 = (const lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__26_value;
static const lean_string_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "←"};
static const lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__27 = (const lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__27_value;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__28;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__29;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__30;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__31;
static const lean_string_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "neg_range'"};
static const lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__32 = (const lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__32_value;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__33;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__34_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__34;
static const lean_ctor_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__32_value),LEAN_SCALAR_PTR_LITERAL(215, 158, 91, 97, 241, 18, 192, 79)}};
static const lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__35 = (const lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__35_value;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__36_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__36;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__37_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__37;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__38_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__38;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__39_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__39;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__40_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__40;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__41_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__41;
static const lean_string_object lp_mathlib_Module_Basis_span__neg___auto__1___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__42 = (const lean_object*)&lp_mathlib_Module_Basis_span__neg___auto__1___closed__42_value;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__43_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__43;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__44_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__44;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__45_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__45;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__46_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__46;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__47_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__47;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__48_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__48;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__49_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__49;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__50_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__50;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__51_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__51;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__52_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__52;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__53_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__53;
static lean_once_cell_t lp_mathlib_Module_Basis_span__neg___auto__1___closed__54_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Module_Basis_span__neg___auto__1___closed__54;
LEAN_EXPORT lean_object* lp_mathlib_Module_Basis_span__neg___auto__1;
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__12(void){
_start:
{
lean_object* v___x_27_; lean_object* v___x_28_; 
v___x_27_ = ((lean_object*)(lp_mathlib_Module_Basis_span__neg___auto__1___closed__10));
v___x_28_ = l_Lean_mkAtom(v___x_27_);
return v___x_28_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__13(void){
_start:
{
lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v___x_31_; 
v___x_29_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__12, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__12_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__12);
v___x_30_ = ((lean_object*)(lp_mathlib_Module_Basis_span__neg___auto__1___closed__5));
v___x_31_ = lean_array_push(v___x_30_, v___x_29_);
return v___x_31_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__17(void){
_start:
{
lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; 
v___x_42_ = ((lean_object*)(lp_mathlib_Module_Basis_span__neg___auto__1___closed__16));
v___x_43_ = ((lean_object*)(lp_mathlib_Module_Basis_span__neg___auto__1___closed__5));
v___x_44_ = lean_array_push(v___x_43_, v___x_42_);
return v___x_44_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__18(void){
_start:
{
lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; 
v___x_45_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__17, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__17_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__17);
v___x_46_ = ((lean_object*)(lp_mathlib_Module_Basis_span__neg___auto__1___closed__15));
v___x_47_ = lean_box(2);
v___x_48_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_48_, 0, v___x_47_);
lean_ctor_set(v___x_48_, 1, v___x_46_);
lean_ctor_set(v___x_48_, 2, v___x_45_);
return v___x_48_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__19(void){
_start:
{
lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; 
v___x_49_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__18, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__18_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__18);
v___x_50_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__13, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__13_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__13);
v___x_51_ = lean_array_push(v___x_50_, v___x_49_);
return v___x_51_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__20(void){
_start:
{
lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; 
v___x_52_ = ((lean_object*)(lp_mathlib_Module_Basis_span__neg___auto__1___closed__16));
v___x_53_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__19, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__19_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__19);
v___x_54_ = lean_array_push(v___x_53_, v___x_52_);
return v___x_54_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__21(void){
_start:
{
lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; 
v___x_55_ = ((lean_object*)(lp_mathlib_Module_Basis_span__neg___auto__1___closed__16));
v___x_56_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__20, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__20_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__20);
v___x_57_ = lean_array_push(v___x_56_, v___x_55_);
return v___x_57_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__23(void){
_start:
{
lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_59_ = ((lean_object*)(lp_mathlib_Module_Basis_span__neg___auto__1___closed__22));
v___x_60_ = l_Lean_mkAtom(v___x_59_);
return v___x_60_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__24(void){
_start:
{
lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_61_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__23, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__23_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__23);
v___x_62_ = ((lean_object*)(lp_mathlib_Module_Basis_span__neg___auto__1___closed__5));
v___x_63_ = lean_array_push(v___x_62_, v___x_61_);
return v___x_63_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__28(void){
_start:
{
lean_object* v___x_71_; lean_object* v___x_72_; 
v___x_71_ = ((lean_object*)(lp_mathlib_Module_Basis_span__neg___auto__1___closed__27));
v___x_72_ = l_Lean_mkAtom(v___x_71_);
return v___x_72_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__29(void){
_start:
{
lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; 
v___x_73_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__28, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__28_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__28);
v___x_74_ = ((lean_object*)(lp_mathlib_Module_Basis_span__neg___auto__1___closed__5));
v___x_75_ = lean_array_push(v___x_74_, v___x_73_);
return v___x_75_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__30(void){
_start:
{
lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; 
v___x_76_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__29, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__29_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__29);
v___x_77_ = ((lean_object*)(lp_mathlib_Module_Basis_span__neg___auto__1___closed__9));
v___x_78_ = lean_box(2);
v___x_79_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_79_, 0, v___x_78_);
lean_ctor_set(v___x_79_, 1, v___x_77_);
lean_ctor_set(v___x_79_, 2, v___x_76_);
return v___x_79_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__31(void){
_start:
{
lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; 
v___x_80_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__30, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__30_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__30);
v___x_81_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__17, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__17_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__17);
v___x_82_ = lean_array_push(v___x_81_, v___x_80_);
return v___x_82_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__33(void){
_start:
{
lean_object* v___x_84_; lean_object* v___x_85_; 
v___x_84_ = ((lean_object*)(lp_mathlib_Module_Basis_span__neg___auto__1___closed__32));
v___x_85_ = lean_string_utf8_byte_size(v___x_84_);
return v___x_85_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__34(void){
_start:
{
lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; 
v___x_86_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__33, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__33_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__33);
v___x_87_ = lean_unsigned_to_nat(0u);
v___x_88_ = ((lean_object*)(lp_mathlib_Module_Basis_span__neg___auto__1___closed__32));
v___x_89_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_89_, 0, v___x_88_);
lean_ctor_set(v___x_89_, 1, v___x_87_);
lean_ctor_set(v___x_89_, 2, v___x_86_);
return v___x_89_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__36(void){
_start:
{
lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; 
v___x_92_ = lean_box(0);
v___x_93_ = ((lean_object*)(lp_mathlib_Module_Basis_span__neg___auto__1___closed__35));
v___x_94_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__34, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__34_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__34);
v___x_95_ = lean_box(2);
v___x_96_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_96_, 0, v___x_95_);
lean_ctor_set(v___x_96_, 1, v___x_94_);
lean_ctor_set(v___x_96_, 2, v___x_93_);
lean_ctor_set(v___x_96_, 3, v___x_92_);
return v___x_96_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__37(void){
_start:
{
lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; 
v___x_97_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__36, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__36_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__36);
v___x_98_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__31, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__31_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__31);
v___x_99_ = lean_array_push(v___x_98_, v___x_97_);
return v___x_99_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__38(void){
_start:
{
lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; 
v___x_100_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__37, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__37_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__37);
v___x_101_ = ((lean_object*)(lp_mathlib_Module_Basis_span__neg___auto__1___closed__26));
v___x_102_ = lean_box(2);
v___x_103_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_103_, 0, v___x_102_);
lean_ctor_set(v___x_103_, 1, v___x_101_);
lean_ctor_set(v___x_103_, 2, v___x_100_);
return v___x_103_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__39(void){
_start:
{
lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; 
v___x_104_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__38, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__38_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__38);
v___x_105_ = ((lean_object*)(lp_mathlib_Module_Basis_span__neg___auto__1___closed__5));
v___x_106_ = lean_array_push(v___x_105_, v___x_104_);
return v___x_106_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__40(void){
_start:
{
lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; 
v___x_107_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__39, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__39_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__39);
v___x_108_ = ((lean_object*)(lp_mathlib_Module_Basis_span__neg___auto__1___closed__9));
v___x_109_ = lean_box(2);
v___x_110_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_110_, 0, v___x_109_);
lean_ctor_set(v___x_110_, 1, v___x_108_);
lean_ctor_set(v___x_110_, 2, v___x_107_);
return v___x_110_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__41(void){
_start:
{
lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; 
v___x_111_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__40, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__40_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__40);
v___x_112_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__24, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__24_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__24);
v___x_113_ = lean_array_push(v___x_112_, v___x_111_);
return v___x_113_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__43(void){
_start:
{
lean_object* v___x_115_; lean_object* v___x_116_; 
v___x_115_ = ((lean_object*)(lp_mathlib_Module_Basis_span__neg___auto__1___closed__42));
v___x_116_ = l_Lean_mkAtom(v___x_115_);
return v___x_116_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__44(void){
_start:
{
lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; 
v___x_117_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__43, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__43_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__43);
v___x_118_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__41, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__41_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__41);
v___x_119_ = lean_array_push(v___x_118_, v___x_117_);
return v___x_119_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__45(void){
_start:
{
lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; 
v___x_120_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__44, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__44_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__44);
v___x_121_ = ((lean_object*)(lp_mathlib_Module_Basis_span__neg___auto__1___closed__9));
v___x_122_ = lean_box(2);
v___x_123_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_123_, 0, v___x_122_);
lean_ctor_set(v___x_123_, 1, v___x_121_);
lean_ctor_set(v___x_123_, 2, v___x_120_);
return v___x_123_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__46(void){
_start:
{
lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; 
v___x_124_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__45, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__45_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__45);
v___x_125_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__21, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__21_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__21);
v___x_126_ = lean_array_push(v___x_125_, v___x_124_);
return v___x_126_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__47(void){
_start:
{
lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; 
v___x_127_ = ((lean_object*)(lp_mathlib_Module_Basis_span__neg___auto__1___closed__16));
v___x_128_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__46, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__46_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__46);
v___x_129_ = lean_array_push(v___x_128_, v___x_127_);
return v___x_129_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__48(void){
_start:
{
lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; 
v___x_130_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__47, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__47_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__47);
v___x_131_ = ((lean_object*)(lp_mathlib_Module_Basis_span__neg___auto__1___closed__11));
v___x_132_ = lean_box(2);
v___x_133_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_133_, 0, v___x_132_);
lean_ctor_set(v___x_133_, 1, v___x_131_);
lean_ctor_set(v___x_133_, 2, v___x_130_);
return v___x_133_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__49(void){
_start:
{
lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; 
v___x_134_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__48, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__48_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__48);
v___x_135_ = ((lean_object*)(lp_mathlib_Module_Basis_span__neg___auto__1___closed__5));
v___x_136_ = lean_array_push(v___x_135_, v___x_134_);
return v___x_136_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__50(void){
_start:
{
lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; 
v___x_137_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__49, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__49_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__49);
v___x_138_ = ((lean_object*)(lp_mathlib_Module_Basis_span__neg___auto__1___closed__9));
v___x_139_ = lean_box(2);
v___x_140_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_140_, 0, v___x_139_);
lean_ctor_set(v___x_140_, 1, v___x_138_);
lean_ctor_set(v___x_140_, 2, v___x_137_);
return v___x_140_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__51(void){
_start:
{
lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; 
v___x_141_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__50, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__50_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__50);
v___x_142_ = ((lean_object*)(lp_mathlib_Module_Basis_span__neg___auto__1___closed__5));
v___x_143_ = lean_array_push(v___x_142_, v___x_141_);
return v___x_143_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__52(void){
_start:
{
lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; 
v___x_144_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__51, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__51_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__51);
v___x_145_ = ((lean_object*)(lp_mathlib_Module_Basis_span__neg___auto__1___closed__7));
v___x_146_ = lean_box(2);
v___x_147_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_147_, 0, v___x_146_);
lean_ctor_set(v___x_147_, 1, v___x_145_);
lean_ctor_set(v___x_147_, 2, v___x_144_);
return v___x_147_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__53(void){
_start:
{
lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; 
v___x_148_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__52, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__52_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__52);
v___x_149_ = ((lean_object*)(lp_mathlib_Module_Basis_span__neg___auto__1___closed__5));
v___x_150_ = lean_array_push(v___x_149_, v___x_148_);
return v___x_150_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__54(void){
_start:
{
lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; 
v___x_151_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__53, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__53_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__53);
v___x_152_ = ((lean_object*)(lp_mathlib_Module_Basis_span__neg___auto__1___closed__4));
v___x_153_ = lean_box(2);
v___x_154_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_154_, 0, v___x_153_);
lean_ctor_set(v___x_154_, 1, v___x_152_);
lean_ctor_set(v___x_154_, 2, v___x_151_);
return v___x_154_;
}
}
static lean_object* _init_lp_mathlib_Module_Basis_span__neg___auto__1(void){
_start:
{
lean_object* v___x_155_; 
v___x_155_ = lean_obj_once(&lp_mathlib_Module_Basis_span__neg___auto__1___closed__54, &lp_mathlib_Module_Basis_span__neg___auto__1___closed__54_once, _init_lp_mathlib_Module_Basis_span__neg___auto__1___closed__54);
return v___x_155_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Basis_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_LinearIndependent_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Basis_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Basis_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_LinearIndependent_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_LinearAlgebra_Basis_Basic(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Module_Basis_span__neg___auto__1 = _init_lp_mathlib_Module_Basis_span__neg___auto__1();
lean_mark_persistent(lp_mathlib_Module_Basis_span__neg___auto__1);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Basis_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_LinearIndependent_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Basis_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Basis_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_LinearIndependent_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Span_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Basis_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_LinearAlgebra_Basis_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_LinearAlgebra_Basis_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
