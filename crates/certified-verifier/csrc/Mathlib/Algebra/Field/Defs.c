// Lean compiler output
// Module: Mathlib.Algebra.Field.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Ring.Defs public import Mathlib.Data.Rat.Init
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
lean_object* lp_mathlib_NNRat_num(lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_castRec___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_castRec(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_castRec___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_castRec(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__0 = (const lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__0_value;
static const lean_string_object lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__1 = (const lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__1_value;
static const lean_string_object lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__2 = (const lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__2_value;
static const lean_string_object lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__3 = (const lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__3_value;
static const lean_ctor_object lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__4_value_aux_0),((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__4_value_aux_1),((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__4_value_aux_2),((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__4 = (const lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__4_value;
static const lean_array_object lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__5 = (const lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__5_value;
static const lean_string_object lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__6 = (const lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__6_value;
static const lean_ctor_object lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__7_value_aux_0),((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__7_value_aux_1),((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__7_value_aux_2),((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__7 = (const lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__7_value;
static const lean_string_object lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__8 = (const lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__8_value;
static const lean_ctor_object lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__9 = (const lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__9_value;
static const lean_string_object lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "intros"};
static const lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__10 = (const lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__10_value;
static const lean_ctor_object lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__11_value_aux_0),((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__11_value_aux_1),((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__11_value_aux_2),((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__10_value),LEAN_SCALAR_PTR_LITERAL(26, 175, 18, 116, 252, 50, 128, 45)}};
static const lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__11 = (const lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__11_value;
static lean_once_cell_t lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__12;
static lean_once_cell_t lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__13;
static const lean_ctor_object lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__9_value),((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__5_value)}};
static const lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__14 = (const lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__14_value;
static lean_once_cell_t lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__15;
static lean_once_cell_t lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__16;
static lean_once_cell_t lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__17;
static const lean_string_object lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ";"};
static const lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__18 = (const lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__18_value;
static lean_once_cell_t lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__19;
static lean_once_cell_t lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__20;
static const lean_string_object lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticRfl"};
static const lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__21 = (const lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__21_value;
static const lean_ctor_object lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__22_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__22_value_aux_0),((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__22_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__22_value_aux_1),((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__22_value_aux_2),((lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__21_value),LEAN_SCALAR_PTR_LITERAL(201, 188, 173, 198, 169, 252, 183, 45)}};
static const lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__22 = (const lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__22_value;
static const lean_string_object lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rfl"};
static const lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__23 = (const lean_object*)&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__23_value;
static lean_once_cell_t lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__24;
static lean_once_cell_t lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__25;
static lean_once_cell_t lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__26;
static lean_once_cell_t lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__27;
static lean_once_cell_t lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__28;
static lean_once_cell_t lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__29;
static lean_once_cell_t lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__30;
static lean_once_cell_t lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__31;
static lean_once_cell_t lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__32;
LEAN_EXPORT lean_object* lp_mathlib_DivisionSemiring_nnratCast__def___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_DivisionSemiring_nnqsmul__def___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionSemiring_toGroupWithZero___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionSemiring_toGroupWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionSemiring_toGroupWithZero___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionRing_nnratCast__def___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_DivisionRing_nnqsmul__def___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_DivisionRing_ratCast__def___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_DivisionRing_qsmul__def___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_DivisionRing_toDivInvMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionRing_toDivInvMonoid___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionRing_toDivInvMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionRing_toDivInvMonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionRing_toDivisionSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionRing_toDivisionSemiring___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionRing_toDivisionSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DivisionRing_toDivisionSemiring___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semifield_toDivisionSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semifield_toDivisionSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semifield_toCommGroupWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semifield_toCommGroupWithZero___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semifield_toCommGroupWithZero(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Semifield_toCommGroupWithZero___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Field_toDivisionRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Field_toDivisionRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Field_toSemifield___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Field_toSemifield___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Field_toSemifield(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Field_toSemifield___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_smulDivisionSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_smulDivisionSemiring___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_smulDivisionSemiring(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_smulDivisionSemiring___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_smulDivisionRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_smulDivisionRing___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_smulDivisionRing(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_smulDivisionRing___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_castRec___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_, lean_object* v_q_3_){
_start:
{
lean_object* v_den_4_; lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v___x_8_; 
v_den_4_ = lean_ctor_get(v_q_3_, 1);
lean_inc(v_den_4_);
v___x_5_ = lp_mathlib_NNRat_num(v_q_3_);
lean_dec_ref(v_q_3_);
lean_inc(v_inst_1_);
v___x_6_ = lean_apply_1(v_inst_1_, v___x_5_);
v___x_7_ = lean_apply_1(v_inst_1_, v_den_4_);
v___x_8_ = lean_apply_2(v_inst_2_, v___x_6_, v___x_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_castRec(lean_object* v_K_9_, lean_object* v_inst_10_, lean_object* v_inst_11_, lean_object* v_q_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lp_mathlib_NNRat_castRec___redArg(v_inst_10_, v_inst_11_, v_q_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_castRec___redArg(lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_q_17_){
_start:
{
lean_object* v_num_18_; lean_object* v_den_19_; lean_object* v___x_20_; lean_object* v___x_21_; lean_object* v___x_22_; 
v_num_18_ = lean_ctor_get(v_q_17_, 0);
lean_inc(v_num_18_);
v_den_19_ = lean_ctor_get(v_q_17_, 1);
lean_inc(v_den_19_);
lean_dec_ref(v_q_17_);
v___x_20_ = lean_apply_1(v_inst_15_, v_num_18_);
v___x_21_ = lean_apply_1(v_inst_14_, v_den_19_);
v___x_22_ = lean_apply_2(v_inst_16_, v___x_20_, v___x_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_castRec(lean_object* v_K_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_inst_26_, lean_object* v_q_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lp_mathlib_Rat_castRec___redArg(v_inst_24_, v_inst_25_, v_inst_26_, v_q_27_);
return v___x_28_;
}
}
static lean_object* _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__12(void){
_start:
{
lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_55_ = ((lean_object*)(lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__10));
v___x_56_ = l_Lean_mkAtom(v___x_55_);
return v___x_56_;
}
}
static lean_object* _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__13(void){
_start:
{
lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; 
v___x_57_ = lean_obj_once(&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__12, &lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__12_once, _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__12);
v___x_58_ = ((lean_object*)(lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__5));
v___x_59_ = lean_array_push(v___x_58_, v___x_57_);
return v___x_59_;
}
}
static lean_object* _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__15(void){
_start:
{
lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; 
v___x_64_ = ((lean_object*)(lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__14));
v___x_65_ = lean_obj_once(&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__13, &lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__13_once, _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__13);
v___x_66_ = lean_array_push(v___x_65_, v___x_64_);
return v___x_66_;
}
}
static lean_object* _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__16(void){
_start:
{
lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; 
v___x_67_ = lean_obj_once(&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__15, &lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__15_once, _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__15);
v___x_68_ = ((lean_object*)(lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__11));
v___x_69_ = lean_box(2);
v___x_70_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_70_, 0, v___x_69_);
lean_ctor_set(v___x_70_, 1, v___x_68_);
lean_ctor_set(v___x_70_, 2, v___x_67_);
return v___x_70_;
}
}
static lean_object* _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__17(void){
_start:
{
lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; 
v___x_71_ = lean_obj_once(&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__16, &lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__16_once, _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__16);
v___x_72_ = ((lean_object*)(lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__5));
v___x_73_ = lean_array_push(v___x_72_, v___x_71_);
return v___x_73_;
}
}
static lean_object* _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__19(void){
_start:
{
lean_object* v___x_75_; lean_object* v___x_76_; 
v___x_75_ = ((lean_object*)(lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__18));
v___x_76_ = l_Lean_mkAtom(v___x_75_);
return v___x_76_;
}
}
static lean_object* _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__20(void){
_start:
{
lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; 
v___x_77_ = lean_obj_once(&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__19, &lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__19_once, _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__19);
v___x_78_ = lean_obj_once(&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__17, &lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__17_once, _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__17);
v___x_79_ = lean_array_push(v___x_78_, v___x_77_);
return v___x_79_;
}
}
static lean_object* _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__24(void){
_start:
{
lean_object* v___x_87_; lean_object* v___x_88_; 
v___x_87_ = ((lean_object*)(lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__23));
v___x_88_ = l_Lean_mkAtom(v___x_87_);
return v___x_88_;
}
}
static lean_object* _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__25(void){
_start:
{
lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; 
v___x_89_ = lean_obj_once(&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__24, &lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__24_once, _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__24);
v___x_90_ = ((lean_object*)(lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__5));
v___x_91_ = lean_array_push(v___x_90_, v___x_89_);
return v___x_91_;
}
}
static lean_object* _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__26(void){
_start:
{
lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; 
v___x_92_ = lean_obj_once(&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__25, &lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__25_once, _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__25);
v___x_93_ = ((lean_object*)(lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__22));
v___x_94_ = lean_box(2);
v___x_95_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_95_, 0, v___x_94_);
lean_ctor_set(v___x_95_, 1, v___x_93_);
lean_ctor_set(v___x_95_, 2, v___x_92_);
return v___x_95_;
}
}
static lean_object* _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__27(void){
_start:
{
lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; 
v___x_96_ = lean_obj_once(&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__26, &lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__26_once, _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__26);
v___x_97_ = lean_obj_once(&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__20, &lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__20_once, _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__20);
v___x_98_ = lean_array_push(v___x_97_, v___x_96_);
return v___x_98_;
}
}
static lean_object* _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__28(void){
_start:
{
lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; 
v___x_99_ = lean_obj_once(&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__27, &lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__27_once, _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__27);
v___x_100_ = ((lean_object*)(lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__9));
v___x_101_ = lean_box(2);
v___x_102_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_102_, 0, v___x_101_);
lean_ctor_set(v___x_102_, 1, v___x_100_);
lean_ctor_set(v___x_102_, 2, v___x_99_);
return v___x_102_;
}
}
static lean_object* _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__29(void){
_start:
{
lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; 
v___x_103_ = lean_obj_once(&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__28, &lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__28_once, _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__28);
v___x_104_ = ((lean_object*)(lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__5));
v___x_105_ = lean_array_push(v___x_104_, v___x_103_);
return v___x_105_;
}
}
static lean_object* _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__30(void){
_start:
{
lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; 
v___x_106_ = lean_obj_once(&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__29, &lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__29_once, _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__29);
v___x_107_ = ((lean_object*)(lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__7));
v___x_108_ = lean_box(2);
v___x_109_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_109_, 0, v___x_108_);
lean_ctor_set(v___x_109_, 1, v___x_107_);
lean_ctor_set(v___x_109_, 2, v___x_106_);
return v___x_109_;
}
}
static lean_object* _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__31(void){
_start:
{
lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; 
v___x_110_ = lean_obj_once(&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__30, &lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__30_once, _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__30);
v___x_111_ = ((lean_object*)(lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__5));
v___x_112_ = lean_array_push(v___x_111_, v___x_110_);
return v___x_112_;
}
}
static lean_object* _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__32(void){
_start:
{
lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; 
v___x_113_ = lean_obj_once(&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__31, &lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__31_once, _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__31);
v___x_114_ = ((lean_object*)(lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__4));
v___x_115_ = lean_box(2);
v___x_116_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_116_, 0, v___x_115_);
lean_ctor_set(v___x_116_, 1, v___x_114_);
lean_ctor_set(v___x_116_, 2, v___x_113_);
return v___x_116_;
}
}
static lean_object* _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam(void){
_start:
{
lean_object* v___x_117_; 
v___x_117_ = lean_obj_once(&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__32, &lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__32_once, _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__32);
return v___x_117_;
}
}
static lean_object* _init_lp_mathlib_DivisionSemiring_nnqsmul__def___autoParam(void){
_start:
{
lean_object* v___x_118_; 
v___x_118_ = lean_obj_once(&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__32, &lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__32_once, _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__32);
return v___x_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(lean_object* v_self_119_){
_start:
{
lean_object* v_toSemiring_120_; lean_object* v_toAddCommMonoid_121_; lean_object* v_toInv_122_; lean_object* v_toDiv_123_; lean_object* v_toZPow_124_; lean_object* v_toMonoid_125_; lean_object* v_toZero_126_; lean_object* v___x_127_; lean_object* v___x_128_; 
v_toSemiring_120_ = lean_ctor_get(v_self_119_, 0);
v_toAddCommMonoid_121_ = lean_ctor_get(v_toSemiring_120_, 0);
v_toInv_122_ = lean_ctor_get(v_self_119_, 1);
v_toDiv_123_ = lean_ctor_get(v_self_119_, 2);
v_toZPow_124_ = lean_ctor_get(v_self_119_, 3);
v_toMonoid_125_ = lean_ctor_get(v_toSemiring_120_, 1);
v_toZero_126_ = lean_ctor_get(v_toAddCommMonoid_121_, 0);
lean_inc(v_toZero_126_);
lean_inc_ref(v_toMonoid_125_);
v___x_127_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_127_, 0, v_toMonoid_125_);
lean_ctor_set(v___x_127_, 1, v_toZero_126_);
lean_inc(v_toZPow_124_);
lean_inc(v_toDiv_123_);
lean_inc(v_toInv_122_);
v___x_128_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_128_, 0, v___x_127_);
lean_ctor_set(v___x_128_, 1, v_toInv_122_);
lean_ctor_set(v___x_128_, 2, v_toDiv_123_);
lean_ctor_set(v___x_128_, 3, v_toZPow_124_);
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionSemiring_toGroupWithZero___redArg___boxed(lean_object* v_self_129_){
_start:
{
lean_object* v_res_130_; 
v_res_130_ = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(v_self_129_);
lean_dec_ref(v_self_129_);
return v_res_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionSemiring_toGroupWithZero(lean_object* v_K_131_, lean_object* v_self_132_){
_start:
{
lean_object* v___x_133_; 
v___x_133_ = lp_mathlib_DivisionSemiring_toGroupWithZero___redArg(v_self_132_);
return v___x_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionSemiring_toGroupWithZero___boxed(lean_object* v_K_134_, lean_object* v_self_135_){
_start:
{
lean_object* v_res_136_; 
v_res_136_ = lp_mathlib_DivisionSemiring_toGroupWithZero(v_K_134_, v_self_135_);
lean_dec_ref(v_self_135_);
return v_res_136_;
}
}
static lean_object* _init_lp_mathlib_DivisionRing_nnratCast__def___autoParam(void){
_start:
{
lean_object* v___x_137_; 
v___x_137_ = lean_obj_once(&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__32, &lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__32_once, _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__32);
return v___x_137_;
}
}
static lean_object* _init_lp_mathlib_DivisionRing_nnqsmul__def___autoParam(void){
_start:
{
lean_object* v___x_138_; 
v___x_138_ = lean_obj_once(&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__32, &lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__32_once, _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__32);
return v___x_138_;
}
}
static lean_object* _init_lp_mathlib_DivisionRing_ratCast__def___autoParam(void){
_start:
{
lean_object* v___x_139_; 
v___x_139_ = lean_obj_once(&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__32, &lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__32_once, _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__32);
return v___x_139_;
}
}
static lean_object* _init_lp_mathlib_DivisionRing_qsmul__def___autoParam(void){
_start:
{
lean_object* v___x_140_; 
v___x_140_ = lean_obj_once(&lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__32, &lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__32_once, _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam___closed__32);
return v___x_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionRing_toDivInvMonoid___redArg(lean_object* v_self_141_){
_start:
{
lean_object* v_toRing_142_; lean_object* v_toSemiring_143_; lean_object* v_toInv_144_; lean_object* v_toDiv_145_; lean_object* v_toZPow_146_; lean_object* v_toMonoid_147_; lean_object* v___x_148_; 
v_toRing_142_ = lean_ctor_get(v_self_141_, 0);
v_toSemiring_143_ = lean_ctor_get(v_toRing_142_, 0);
v_toInv_144_ = lean_ctor_get(v_self_141_, 1);
v_toDiv_145_ = lean_ctor_get(v_self_141_, 2);
v_toZPow_146_ = lean_ctor_get(v_self_141_, 3);
v_toMonoid_147_ = lean_ctor_get(v_toSemiring_143_, 1);
lean_inc(v_toZPow_146_);
lean_inc(v_toDiv_145_);
lean_inc(v_toInv_144_);
lean_inc_ref(v_toMonoid_147_);
v___x_148_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_148_, 0, v_toMonoid_147_);
lean_ctor_set(v___x_148_, 1, v_toInv_144_);
lean_ctor_set(v___x_148_, 2, v_toDiv_145_);
lean_ctor_set(v___x_148_, 3, v_toZPow_146_);
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionRing_toDivInvMonoid___redArg___boxed(lean_object* v_self_149_){
_start:
{
lean_object* v_res_150_; 
v_res_150_ = lp_mathlib_DivisionRing_toDivInvMonoid___redArg(v_self_149_);
lean_dec_ref(v_self_149_);
return v_res_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionRing_toDivInvMonoid(lean_object* v_K_151_, lean_object* v_self_152_){
_start:
{
lean_object* v___x_153_; 
v___x_153_ = lp_mathlib_DivisionRing_toDivInvMonoid___redArg(v_self_152_);
return v___x_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionRing_toDivInvMonoid___boxed(lean_object* v_K_154_, lean_object* v_self_155_){
_start:
{
lean_object* v_res_156_; 
v_res_156_ = lp_mathlib_DivisionRing_toDivInvMonoid(v_K_154_, v_self_155_);
lean_dec_ref(v_self_155_);
return v_res_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionRing_toDivisionSemiring___redArg(lean_object* v_inst_157_){
_start:
{
lean_object* v_toRing_158_; lean_object* v_toInv_159_; lean_object* v_toDiv_160_; lean_object* v_toZPow_161_; lean_object* v_toNNRatCast_162_; lean_object* v_nnqsmul_163_; lean_object* v_toSemiring_164_; lean_object* v___x_165_; 
v_toRing_158_ = lean_ctor_get(v_inst_157_, 0);
v_toInv_159_ = lean_ctor_get(v_inst_157_, 1);
v_toDiv_160_ = lean_ctor_get(v_inst_157_, 2);
v_toZPow_161_ = lean_ctor_get(v_inst_157_, 3);
v_toNNRatCast_162_ = lean_ctor_get(v_inst_157_, 4);
v_nnqsmul_163_ = lean_ctor_get(v_inst_157_, 6);
v_toSemiring_164_ = lean_ctor_get(v_toRing_158_, 0);
lean_inc(v_nnqsmul_163_);
lean_inc(v_toNNRatCast_162_);
lean_inc(v_toZPow_161_);
lean_inc(v_toDiv_160_);
lean_inc(v_toInv_159_);
lean_inc_ref(v_toSemiring_164_);
v___x_165_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_165_, 0, v_toSemiring_164_);
lean_ctor_set(v___x_165_, 1, v_toInv_159_);
lean_ctor_set(v___x_165_, 2, v_toDiv_160_);
lean_ctor_set(v___x_165_, 3, v_toZPow_161_);
lean_ctor_set(v___x_165_, 4, v_toNNRatCast_162_);
lean_ctor_set(v___x_165_, 5, v_nnqsmul_163_);
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionRing_toDivisionSemiring___redArg___boxed(lean_object* v_inst_166_){
_start:
{
lean_object* v_res_167_; 
v_res_167_ = lp_mathlib_DivisionRing_toDivisionSemiring___redArg(v_inst_166_);
lean_dec_ref(v_inst_166_);
return v_res_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionRing_toDivisionSemiring(lean_object* v_K_168_, lean_object* v_inst_169_){
_start:
{
lean_object* v___x_170_; 
v___x_170_ = lp_mathlib_DivisionRing_toDivisionSemiring___redArg(v_inst_169_);
return v___x_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DivisionRing_toDivisionSemiring___boxed(lean_object* v_K_171_, lean_object* v_inst_172_){
_start:
{
lean_object* v_res_173_; 
v_res_173_ = lp_mathlib_DivisionRing_toDivisionSemiring(v_K_171_, v_inst_172_);
lean_dec_ref(v_inst_172_);
return v_res_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semifield_toDivisionSemiring___redArg(lean_object* v_self_174_){
_start:
{
lean_object* v_toCommSemiring_175_; lean_object* v_toInv_176_; lean_object* v_toDiv_177_; lean_object* v_toZPow_178_; lean_object* v_toNNRatCast_179_; lean_object* v_nnqsmul_180_; lean_object* v___x_182_; uint8_t v_isShared_183_; uint8_t v_isSharedCheck_187_; 
v_toCommSemiring_175_ = lean_ctor_get(v_self_174_, 0);
v_toInv_176_ = lean_ctor_get(v_self_174_, 1);
v_toDiv_177_ = lean_ctor_get(v_self_174_, 2);
v_toZPow_178_ = lean_ctor_get(v_self_174_, 3);
v_toNNRatCast_179_ = lean_ctor_get(v_self_174_, 4);
v_nnqsmul_180_ = lean_ctor_get(v_self_174_, 5);
v_isSharedCheck_187_ = !lean_is_exclusive(v_self_174_);
if (v_isSharedCheck_187_ == 0)
{
v___x_182_ = v_self_174_;
v_isShared_183_ = v_isSharedCheck_187_;
goto v_resetjp_181_;
}
else
{
lean_inc(v_nnqsmul_180_);
lean_inc(v_toNNRatCast_179_);
lean_inc(v_toZPow_178_);
lean_inc(v_toDiv_177_);
lean_inc(v_toInv_176_);
lean_inc(v_toCommSemiring_175_);
lean_dec(v_self_174_);
v___x_182_ = lean_box(0);
v_isShared_183_ = v_isSharedCheck_187_;
goto v_resetjp_181_;
}
v_resetjp_181_:
{
lean_object* v___x_185_; 
if (v_isShared_183_ == 0)
{
v___x_185_ = v___x_182_;
goto v_reusejp_184_;
}
else
{
lean_object* v_reuseFailAlloc_186_; 
v_reuseFailAlloc_186_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_186_, 0, v_toCommSemiring_175_);
lean_ctor_set(v_reuseFailAlloc_186_, 1, v_toInv_176_);
lean_ctor_set(v_reuseFailAlloc_186_, 2, v_toDiv_177_);
lean_ctor_set(v_reuseFailAlloc_186_, 3, v_toZPow_178_);
lean_ctor_set(v_reuseFailAlloc_186_, 4, v_toNNRatCast_179_);
lean_ctor_set(v_reuseFailAlloc_186_, 5, v_nnqsmul_180_);
v___x_185_ = v_reuseFailAlloc_186_;
goto v_reusejp_184_;
}
v_reusejp_184_:
{
return v___x_185_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semifield_toDivisionSemiring(lean_object* v_K_188_, lean_object* v_self_189_){
_start:
{
lean_object* v___x_190_; 
v___x_190_ = lp_mathlib_Semifield_toDivisionSemiring___redArg(v_self_189_);
return v___x_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semifield_toCommGroupWithZero___redArg(lean_object* v_self_191_){
_start:
{
lean_object* v_toCommSemiring_192_; lean_object* v_toAddCommMonoid_193_; lean_object* v_toInv_194_; lean_object* v_toDiv_195_; lean_object* v_toZPow_196_; lean_object* v_toMonoid_197_; lean_object* v_toZero_198_; lean_object* v___x_199_; lean_object* v___x_200_; 
v_toCommSemiring_192_ = lean_ctor_get(v_self_191_, 0);
v_toAddCommMonoid_193_ = lean_ctor_get(v_toCommSemiring_192_, 0);
v_toInv_194_ = lean_ctor_get(v_self_191_, 1);
v_toDiv_195_ = lean_ctor_get(v_self_191_, 2);
v_toZPow_196_ = lean_ctor_get(v_self_191_, 3);
v_toMonoid_197_ = lean_ctor_get(v_toCommSemiring_192_, 1);
v_toZero_198_ = lean_ctor_get(v_toAddCommMonoid_193_, 0);
lean_inc(v_toZero_198_);
lean_inc_ref(v_toMonoid_197_);
v___x_199_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_199_, 0, v_toMonoid_197_);
lean_ctor_set(v___x_199_, 1, v_toZero_198_);
lean_inc(v_toZPow_196_);
lean_inc(v_toDiv_195_);
lean_inc(v_toInv_194_);
v___x_200_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_200_, 0, v___x_199_);
lean_ctor_set(v___x_200_, 1, v_toInv_194_);
lean_ctor_set(v___x_200_, 2, v_toDiv_195_);
lean_ctor_set(v___x_200_, 3, v_toZPow_196_);
return v___x_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semifield_toCommGroupWithZero___redArg___boxed(lean_object* v_self_201_){
_start:
{
lean_object* v_res_202_; 
v_res_202_ = lp_mathlib_Semifield_toCommGroupWithZero___redArg(v_self_201_);
lean_dec_ref(v_self_201_);
return v_res_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semifield_toCommGroupWithZero(lean_object* v_K_203_, lean_object* v_self_204_){
_start:
{
lean_object* v___x_205_; 
v___x_205_ = lp_mathlib_Semifield_toCommGroupWithZero___redArg(v_self_204_);
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Semifield_toCommGroupWithZero___boxed(lean_object* v_K_206_, lean_object* v_self_207_){
_start:
{
lean_object* v_res_208_; 
v_res_208_ = lp_mathlib_Semifield_toCommGroupWithZero(v_K_206_, v_self_207_);
lean_dec_ref(v_self_207_);
return v_res_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Field_toDivisionRing___redArg(lean_object* v_self_209_){
_start:
{
lean_object* v_toCommRing_210_; lean_object* v_toInv_211_; lean_object* v_toDiv_212_; lean_object* v_toZPow_213_; lean_object* v_toNNRatCast_214_; lean_object* v_toRatCast_215_; lean_object* v_nnqsmul_216_; lean_object* v_qsmul_217_; lean_object* v___x_219_; uint8_t v_isShared_220_; uint8_t v_isSharedCheck_224_; 
v_toCommRing_210_ = lean_ctor_get(v_self_209_, 0);
v_toInv_211_ = lean_ctor_get(v_self_209_, 1);
v_toDiv_212_ = lean_ctor_get(v_self_209_, 2);
v_toZPow_213_ = lean_ctor_get(v_self_209_, 3);
v_toNNRatCast_214_ = lean_ctor_get(v_self_209_, 4);
v_toRatCast_215_ = lean_ctor_get(v_self_209_, 5);
v_nnqsmul_216_ = lean_ctor_get(v_self_209_, 6);
v_qsmul_217_ = lean_ctor_get(v_self_209_, 7);
v_isSharedCheck_224_ = !lean_is_exclusive(v_self_209_);
if (v_isSharedCheck_224_ == 0)
{
v___x_219_ = v_self_209_;
v_isShared_220_ = v_isSharedCheck_224_;
goto v_resetjp_218_;
}
else
{
lean_inc(v_qsmul_217_);
lean_inc(v_nnqsmul_216_);
lean_inc(v_toRatCast_215_);
lean_inc(v_toNNRatCast_214_);
lean_inc(v_toZPow_213_);
lean_inc(v_toDiv_212_);
lean_inc(v_toInv_211_);
lean_inc(v_toCommRing_210_);
lean_dec(v_self_209_);
v___x_219_ = lean_box(0);
v_isShared_220_ = v_isSharedCheck_224_;
goto v_resetjp_218_;
}
v_resetjp_218_:
{
lean_object* v___x_222_; 
if (v_isShared_220_ == 0)
{
v___x_222_ = v___x_219_;
goto v_reusejp_221_;
}
else
{
lean_object* v_reuseFailAlloc_223_; 
v_reuseFailAlloc_223_ = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(v_reuseFailAlloc_223_, 0, v_toCommRing_210_);
lean_ctor_set(v_reuseFailAlloc_223_, 1, v_toInv_211_);
lean_ctor_set(v_reuseFailAlloc_223_, 2, v_toDiv_212_);
lean_ctor_set(v_reuseFailAlloc_223_, 3, v_toZPow_213_);
lean_ctor_set(v_reuseFailAlloc_223_, 4, v_toNNRatCast_214_);
lean_ctor_set(v_reuseFailAlloc_223_, 5, v_toRatCast_215_);
lean_ctor_set(v_reuseFailAlloc_223_, 6, v_nnqsmul_216_);
lean_ctor_set(v_reuseFailAlloc_223_, 7, v_qsmul_217_);
v___x_222_ = v_reuseFailAlloc_223_;
goto v_reusejp_221_;
}
v_reusejp_221_:
{
return v___x_222_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Field_toDivisionRing(lean_object* v_K_225_, lean_object* v_self_226_){
_start:
{
lean_object* v___x_227_; 
v___x_227_ = lp_mathlib_Field_toDivisionRing___redArg(v_self_226_);
return v___x_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Field_toSemifield___redArg(lean_object* v_inst_228_){
_start:
{
lean_object* v_toCommRing_229_; lean_object* v_toInv_230_; lean_object* v_toDiv_231_; lean_object* v_toZPow_232_; lean_object* v_toNNRatCast_233_; lean_object* v_nnqsmul_234_; lean_object* v_toSemiring_235_; lean_object* v___x_236_; 
v_toCommRing_229_ = lean_ctor_get(v_inst_228_, 0);
v_toInv_230_ = lean_ctor_get(v_inst_228_, 1);
v_toDiv_231_ = lean_ctor_get(v_inst_228_, 2);
v_toZPow_232_ = lean_ctor_get(v_inst_228_, 3);
v_toNNRatCast_233_ = lean_ctor_get(v_inst_228_, 4);
v_nnqsmul_234_ = lean_ctor_get(v_inst_228_, 6);
v_toSemiring_235_ = lean_ctor_get(v_toCommRing_229_, 0);
lean_inc(v_nnqsmul_234_);
lean_inc(v_toNNRatCast_233_);
lean_inc(v_toZPow_232_);
lean_inc(v_toDiv_231_);
lean_inc(v_toInv_230_);
lean_inc_ref(v_toSemiring_235_);
v___x_236_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_236_, 0, v_toSemiring_235_);
lean_ctor_set(v___x_236_, 1, v_toInv_230_);
lean_ctor_set(v___x_236_, 2, v_toDiv_231_);
lean_ctor_set(v___x_236_, 3, v_toZPow_232_);
lean_ctor_set(v___x_236_, 4, v_toNNRatCast_233_);
lean_ctor_set(v___x_236_, 5, v_nnqsmul_234_);
return v___x_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Field_toSemifield___redArg___boxed(lean_object* v_inst_237_){
_start:
{
lean_object* v_res_238_; 
v_res_238_ = lp_mathlib_Field_toSemifield___redArg(v_inst_237_);
lean_dec_ref(v_inst_237_);
return v_res_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Field_toSemifield(lean_object* v_K_239_, lean_object* v_inst_240_){
_start:
{
lean_object* v___x_241_; 
v___x_241_ = lp_mathlib_Field_toSemifield___redArg(v_inst_240_);
return v___x_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Field_toSemifield___boxed(lean_object* v_K_242_, lean_object* v_inst_243_){
_start:
{
lean_object* v_res_244_; 
v_res_244_ = lp_mathlib_Field_toSemifield(v_K_242_, v_inst_243_);
lean_dec_ref(v_inst_243_);
return v_res_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_smulDivisionSemiring___redArg(lean_object* v_inst_245_){
_start:
{
lean_object* v_nnqsmul_246_; 
v_nnqsmul_246_ = lean_ctor_get(v_inst_245_, 5);
lean_inc(v_nnqsmul_246_);
return v_nnqsmul_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_smulDivisionSemiring___redArg___boxed(lean_object* v_inst_247_){
_start:
{
lean_object* v_res_248_; 
v_res_248_ = lp_mathlib_NNRat_smulDivisionSemiring___redArg(v_inst_247_);
lean_dec_ref(v_inst_247_);
return v_res_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_smulDivisionSemiring(lean_object* v_K_249_, lean_object* v_inst_250_){
_start:
{
lean_object* v_nnqsmul_251_; 
v_nnqsmul_251_ = lean_ctor_get(v_inst_250_, 5);
lean_inc(v_nnqsmul_251_);
return v_nnqsmul_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_smulDivisionSemiring___boxed(lean_object* v_K_252_, lean_object* v_inst_253_){
_start:
{
lean_object* v_res_254_; 
v_res_254_ = lp_mathlib_NNRat_smulDivisionSemiring(v_K_252_, v_inst_253_);
lean_dec_ref(v_inst_253_);
return v_res_254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_smulDivisionRing___redArg(lean_object* v_inst_255_){
_start:
{
lean_object* v_qsmul_256_; 
v_qsmul_256_ = lean_ctor_get(v_inst_255_, 7);
lean_inc(v_qsmul_256_);
return v_qsmul_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_smulDivisionRing___redArg___boxed(lean_object* v_inst_257_){
_start:
{
lean_object* v_res_258_; 
v_res_258_ = lp_mathlib_Rat_smulDivisionRing___redArg(v_inst_257_);
lean_dec_ref(v_inst_257_);
return v_res_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_smulDivisionRing(lean_object* v_K_259_, lean_object* v_inst_260_){
_start:
{
lean_object* v_qsmul_261_; 
v_qsmul_261_ = lean_ctor_get(v_inst_260_, 7);
lean_inc(v_qsmul_261_);
return v_qsmul_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_smulDivisionRing___boxed(lean_object* v_K_262_, lean_object* v_inst_263_){
_start:
{
lean_object* v_res_264_; 
v_res_264_ = lp_mathlib_Rat_smulDivisionRing(v_K_262_, v_inst_263_);
lean_dec_ref(v_inst_263_);
return v_res_264_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Rat_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Rat_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_DivisionSemiring_nnratCast__def___autoParam = _init_lp_mathlib_DivisionSemiring_nnratCast__def___autoParam();
lean_mark_persistent(lp_mathlib_DivisionSemiring_nnratCast__def___autoParam);
lp_mathlib_DivisionSemiring_nnqsmul__def___autoParam = _init_lp_mathlib_DivisionSemiring_nnqsmul__def___autoParam();
lean_mark_persistent(lp_mathlib_DivisionSemiring_nnqsmul__def___autoParam);
lp_mathlib_DivisionRing_nnratCast__def___autoParam = _init_lp_mathlib_DivisionRing_nnratCast__def___autoParam();
lean_mark_persistent(lp_mathlib_DivisionRing_nnratCast__def___autoParam);
lp_mathlib_DivisionRing_nnqsmul__def___autoParam = _init_lp_mathlib_DivisionRing_nnqsmul__def___autoParam();
lean_mark_persistent(lp_mathlib_DivisionRing_nnqsmul__def___autoParam);
lp_mathlib_DivisionRing_ratCast__def___autoParam = _init_lp_mathlib_DivisionRing_ratCast__def___autoParam();
lean_mark_persistent(lp_mathlib_DivisionRing_ratCast__def___autoParam);
lp_mathlib_DivisionRing_qsmul__def___autoParam = _init_lp_mathlib_DivisionRing_qsmul__def___autoParam();
lean_mark_persistent(lp_mathlib_DivisionRing_qsmul__def___autoParam);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Rat_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Rat_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
