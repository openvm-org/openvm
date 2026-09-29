// Lean compiler output
// Module: Batteries.Data.Nat.Lemmas
// Imports: public import Init public meta import Init public import Batteries.Tactic.Alias public import Batteries.Data.Nat.Basic
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
static const lean_string_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__0 = (const lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__0_value;
static const lean_string_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__1 = (const lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__1_value;
static const lean_string_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__2 = (const lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__2_value;
static const lean_string_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__3 = (const lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__3_value;
static const lean_ctor_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__4_value_aux_0),((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__4_value_aux_1),((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__4_value_aux_2),((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__4 = (const lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__4_value;
static const lean_array_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__5 = (const lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__5_value;
static const lean_string_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__6 = (const lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__6_value;
static const lean_ctor_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__7_value_aux_0),((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__7_value_aux_1),((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__7_value_aux_2),((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__7 = (const lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__7_value;
static const lean_string_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__8 = (const lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__8_value;
static const lean_ctor_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__9 = (const lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__9_value;
static const lean_string_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "first"};
static const lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__10 = (const lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__10_value;
static const lean_ctor_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__11_value_aux_0),((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__11_value_aux_1),((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__11_value_aux_2),((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(59, 232, 35, 17, 172, 62, 48, 174)}};
static const lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__11 = (const lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__11_value;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__12;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__13;
static const lean_string_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__14 = (const lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__14_value;
static const lean_ctor_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__15 = (const lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__15_value;
static const lean_string_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__16 = (const lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__16_value;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__17;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__18;
static const lean_string_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "assumption"};
static const lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__19 = (const lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__19_value;
static const lean_ctor_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__20_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__20_value_aux_0),((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__20_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__20_value_aux_1),((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__20_value_aux_2),((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__19_value),LEAN_SCALAR_PTR_LITERAL(240, 50, 167, 190, 65, 82, 149, 231)}};
static const lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__20 = (const lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__20_value;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__21;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__22;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__23;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__24;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__25;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__26;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__27;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__28;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__29;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__30;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__31;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__32;
static const lean_string_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "tacticTrivial"};
static const lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__33 = (const lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__33_value;
static const lean_ctor_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__34_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__34_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__34_value_aux_0),((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__34_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__34_value_aux_1),((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__34_value_aux_2),((lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__33_value),LEAN_SCALAR_PTR_LITERAL(91, 113, 211, 1, 53, 106, 100, 38)}};
static const lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__34 = (const lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__34_value;
static const lean_string_object lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "trivial"};
static const lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__35 = (const lean_object*)&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__35_value;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__36_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__36;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__37_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__37;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__38_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__38;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__39_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__39;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__40_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__40;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__41_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__41;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__42_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__42;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__43_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__43;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__44_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__44;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__45_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__45;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__46_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__46;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__47_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__47;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__48_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__48;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__49_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__49;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__50_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__50;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__51_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__51;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__52_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__52;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__53_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__53;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__54_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__54;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__55_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__55;
static lean_once_cell_t lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__56_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__56;
LEAN_EXPORT lean_object* lp_batteries_Nat_recDiagAux__zero__right___auto__1;
LEAN_EXPORT lean_object* lp_batteries_Nat_lt__sum__ge(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_lt__sum__ge___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Nat_sum__trichotomy___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_sum__trichotomy___closed__0;
static lean_once_cell_t lp_batteries_Nat_sum__trichotomy___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Nat_sum__trichotomy___closed__1;
LEAN_EXPORT lean_object* lp_batteries_Nat_sum__trichotomy(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_sum__trichotomy___boxed(lean_object*, lean_object*);
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__12(void){
_start:
{
lean_object* v___x_27_; lean_object* v___x_28_; 
v___x_27_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__10));
v___x_28_ = l_Lean_mkAtom(v___x_27_);
return v___x_28_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__13(void){
_start:
{
lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v___x_31_; 
v___x_29_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__12, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__12_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__12);
v___x_30_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__5));
v___x_31_ = lean_array_push(v___x_30_, v___x_29_);
return v___x_31_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__17(void){
_start:
{
lean_object* v___x_36_; lean_object* v___x_37_; 
v___x_36_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__16));
v___x_37_ = l_Lean_mkAtom(v___x_36_);
return v___x_37_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__18(void){
_start:
{
lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; 
v___x_38_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__17, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__17_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__17);
v___x_39_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__5));
v___x_40_ = lean_array_push(v___x_39_, v___x_38_);
return v___x_40_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__21(void){
_start:
{
lean_object* v___x_47_; lean_object* v___x_48_; 
v___x_47_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__19));
v___x_48_ = l_Lean_mkAtom(v___x_47_);
return v___x_48_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__22(void){
_start:
{
lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; 
v___x_49_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__21, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__21_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__21);
v___x_50_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__5));
v___x_51_ = lean_array_push(v___x_50_, v___x_49_);
return v___x_51_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__23(void){
_start:
{
lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; 
v___x_52_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__22, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__22_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__22);
v___x_53_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__20));
v___x_54_ = lean_box(2);
v___x_55_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_55_, 0, v___x_54_);
lean_ctor_set(v___x_55_, 1, v___x_53_);
lean_ctor_set(v___x_55_, 2, v___x_52_);
return v___x_55_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__24(void){
_start:
{
lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; 
v___x_56_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__23, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__23_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__23);
v___x_57_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__5));
v___x_58_ = lean_array_push(v___x_57_, v___x_56_);
return v___x_58_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__25(void){
_start:
{
lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; 
v___x_59_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__24, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__24_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__24);
v___x_60_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__9));
v___x_61_ = lean_box(2);
v___x_62_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_62_, 0, v___x_61_);
lean_ctor_set(v___x_62_, 1, v___x_60_);
lean_ctor_set(v___x_62_, 2, v___x_59_);
return v___x_62_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__26(void){
_start:
{
lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; 
v___x_63_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__25, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__25_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__25);
v___x_64_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__5));
v___x_65_ = lean_array_push(v___x_64_, v___x_63_);
return v___x_65_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__27(void){
_start:
{
lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; 
v___x_66_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__26, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__26_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__26);
v___x_67_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__7));
v___x_68_ = lean_box(2);
v___x_69_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_69_, 0, v___x_68_);
lean_ctor_set(v___x_69_, 1, v___x_67_);
lean_ctor_set(v___x_69_, 2, v___x_66_);
return v___x_69_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__28(void){
_start:
{
lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; 
v___x_70_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__27, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__27_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__27);
v___x_71_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__5));
v___x_72_ = lean_array_push(v___x_71_, v___x_70_);
return v___x_72_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__29(void){
_start:
{
lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; 
v___x_73_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__28, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__28_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__28);
v___x_74_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__4));
v___x_75_ = lean_box(2);
v___x_76_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_76_, 0, v___x_75_);
lean_ctor_set(v___x_76_, 1, v___x_74_);
lean_ctor_set(v___x_76_, 2, v___x_73_);
return v___x_76_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__30(void){
_start:
{
lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; 
v___x_77_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__29, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__29_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__29);
v___x_78_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__18, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__18_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__18);
v___x_79_ = lean_array_push(v___x_78_, v___x_77_);
return v___x_79_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__31(void){
_start:
{
lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; 
v___x_80_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__30, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__30_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__30);
v___x_81_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__15));
v___x_82_ = lean_box(2);
v___x_83_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_83_, 0, v___x_82_);
lean_ctor_set(v___x_83_, 1, v___x_81_);
lean_ctor_set(v___x_83_, 2, v___x_80_);
return v___x_83_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__32(void){
_start:
{
lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; 
v___x_84_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__31, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__31_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__31);
v___x_85_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__5));
v___x_86_ = lean_array_push(v___x_85_, v___x_84_);
return v___x_86_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__36(void){
_start:
{
lean_object* v___x_94_; lean_object* v___x_95_; 
v___x_94_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__35));
v___x_95_ = l_Lean_mkAtom(v___x_94_);
return v___x_95_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__37(void){
_start:
{
lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; 
v___x_96_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__36, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__36_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__36);
v___x_97_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__5));
v___x_98_ = lean_array_push(v___x_97_, v___x_96_);
return v___x_98_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__38(void){
_start:
{
lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; 
v___x_99_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__37, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__37_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__37);
v___x_100_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__34));
v___x_101_ = lean_box(2);
v___x_102_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_102_, 0, v___x_101_);
lean_ctor_set(v___x_102_, 1, v___x_100_);
lean_ctor_set(v___x_102_, 2, v___x_99_);
return v___x_102_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__39(void){
_start:
{
lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; 
v___x_103_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__38, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__38_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__38);
v___x_104_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__5));
v___x_105_ = lean_array_push(v___x_104_, v___x_103_);
return v___x_105_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__40(void){
_start:
{
lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; 
v___x_106_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__39, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__39_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__39);
v___x_107_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__9));
v___x_108_ = lean_box(2);
v___x_109_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_109_, 0, v___x_108_);
lean_ctor_set(v___x_109_, 1, v___x_107_);
lean_ctor_set(v___x_109_, 2, v___x_106_);
return v___x_109_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__41(void){
_start:
{
lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; 
v___x_110_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__40, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__40_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__40);
v___x_111_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__5));
v___x_112_ = lean_array_push(v___x_111_, v___x_110_);
return v___x_112_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__42(void){
_start:
{
lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; 
v___x_113_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__41, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__41_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__41);
v___x_114_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__7));
v___x_115_ = lean_box(2);
v___x_116_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_116_, 0, v___x_115_);
lean_ctor_set(v___x_116_, 1, v___x_114_);
lean_ctor_set(v___x_116_, 2, v___x_113_);
return v___x_116_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__43(void){
_start:
{
lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; 
v___x_117_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__42, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__42_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__42);
v___x_118_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__5));
v___x_119_ = lean_array_push(v___x_118_, v___x_117_);
return v___x_119_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__44(void){
_start:
{
lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; 
v___x_120_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__43, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__43_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__43);
v___x_121_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__4));
v___x_122_ = lean_box(2);
v___x_123_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_123_, 0, v___x_122_);
lean_ctor_set(v___x_123_, 1, v___x_121_);
lean_ctor_set(v___x_123_, 2, v___x_120_);
return v___x_123_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__45(void){
_start:
{
lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; 
v___x_124_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__44, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__44_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__44);
v___x_125_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__18, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__18_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__18);
v___x_126_ = lean_array_push(v___x_125_, v___x_124_);
return v___x_126_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__46(void){
_start:
{
lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; 
v___x_127_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__45, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__45_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__45);
v___x_128_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__15));
v___x_129_ = lean_box(2);
v___x_130_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_130_, 0, v___x_129_);
lean_ctor_set(v___x_130_, 1, v___x_128_);
lean_ctor_set(v___x_130_, 2, v___x_127_);
return v___x_130_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__47(void){
_start:
{
lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; 
v___x_131_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__46, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__46_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__46);
v___x_132_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__32, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__32_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__32);
v___x_133_ = lean_array_push(v___x_132_, v___x_131_);
return v___x_133_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__48(void){
_start:
{
lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; 
v___x_134_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__47, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__47_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__47);
v___x_135_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__9));
v___x_136_ = lean_box(2);
v___x_137_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_137_, 0, v___x_136_);
lean_ctor_set(v___x_137_, 1, v___x_135_);
lean_ctor_set(v___x_137_, 2, v___x_134_);
return v___x_137_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__49(void){
_start:
{
lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; 
v___x_138_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__48, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__48_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__48);
v___x_139_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__13, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__13_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__13);
v___x_140_ = lean_array_push(v___x_139_, v___x_138_);
return v___x_140_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__50(void){
_start:
{
lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; 
v___x_141_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__49, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__49_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__49);
v___x_142_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__11));
v___x_143_ = lean_box(2);
v___x_144_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_144_, 0, v___x_143_);
lean_ctor_set(v___x_144_, 1, v___x_142_);
lean_ctor_set(v___x_144_, 2, v___x_141_);
return v___x_144_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__51(void){
_start:
{
lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; 
v___x_145_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__50, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__50_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__50);
v___x_146_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__5));
v___x_147_ = lean_array_push(v___x_146_, v___x_145_);
return v___x_147_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__52(void){
_start:
{
lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; 
v___x_148_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__51, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__51_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__51);
v___x_149_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__9));
v___x_150_ = lean_box(2);
v___x_151_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_151_, 0, v___x_150_);
lean_ctor_set(v___x_151_, 1, v___x_149_);
lean_ctor_set(v___x_151_, 2, v___x_148_);
return v___x_151_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__53(void){
_start:
{
lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; 
v___x_152_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__52, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__52_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__52);
v___x_153_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__5));
v___x_154_ = lean_array_push(v___x_153_, v___x_152_);
return v___x_154_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__54(void){
_start:
{
lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; 
v___x_155_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__53, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__53_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__53);
v___x_156_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__7));
v___x_157_ = lean_box(2);
v___x_158_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_158_, 0, v___x_157_);
lean_ctor_set(v___x_158_, 1, v___x_156_);
lean_ctor_set(v___x_158_, 2, v___x_155_);
return v___x_158_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__55(void){
_start:
{
lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; 
v___x_159_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__54, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__54_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__54);
v___x_160_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__5));
v___x_161_ = lean_array_push(v___x_160_, v___x_159_);
return v___x_161_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__56(void){
_start:
{
lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; 
v___x_162_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__55, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__55_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__55);
v___x_163_ = ((lean_object*)(lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__4));
v___x_164_ = lean_box(2);
v___x_165_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_165_, 0, v___x_164_);
lean_ctor_set(v___x_165_, 1, v___x_163_);
lean_ctor_set(v___x_165_, 2, v___x_162_);
return v___x_165_;
}
}
static lean_object* _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1(void){
_start:
{
lean_object* v___x_166_; 
v___x_166_ = lean_obj_once(&lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__56, &lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__56_once, _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1___closed__56);
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_lt__sum__ge(lean_object* v_a_167_, lean_object* v_b_168_){
_start:
{
uint8_t v___x_169_; 
v___x_169_ = lean_nat_dec_lt(v_a_167_, v_b_168_);
if (v___x_169_ == 0)
{
lean_object* v___x_170_; 
v___x_170_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_170_, 0, lean_box(0));
return v___x_170_;
}
else
{
lean_object* v___x_171_; 
v___x_171_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_171_, 0, lean_box(0));
return v___x_171_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_lt__sum__ge___boxed(lean_object* v_a_172_, lean_object* v_b_173_){
_start:
{
lean_object* v_res_174_; 
v_res_174_ = lp_batteries_Nat_lt__sum__ge(v_a_172_, v_b_173_);
lean_dec(v_b_173_);
lean_dec(v_a_172_);
return v_res_174_;
}
}
static lean_object* _init_lp_batteries_Nat_sum__trichotomy___closed__0(void){
_start:
{
lean_object* v___x_175_; lean_object* v___x_176_; 
v___x_175_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_175_, 0, lean_box(0));
v___x_176_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_176_, 0, v___x_175_);
return v___x_176_;
}
}
static lean_object* _init_lp_batteries_Nat_sum__trichotomy___closed__1(void){
_start:
{
lean_object* v___x_177_; lean_object* v___x_178_; 
v___x_177_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_177_, 0, lean_box(0));
v___x_178_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_178_, 0, v___x_177_);
return v___x_178_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_sum__trichotomy(lean_object* v_a_179_, lean_object* v_b_180_){
_start:
{
uint8_t v___x_181_; 
v___x_181_ = lean_nat_dec_lt(v_a_179_, v_b_180_);
if (v___x_181_ == 0)
{
uint8_t v___x_182_; 
v___x_182_ = lean_nat_dec_eq(v_a_179_, v_b_180_);
if (v___x_182_ == 0)
{
lean_object* v___x_183_; 
v___x_183_ = lean_obj_once(&lp_batteries_Nat_sum__trichotomy___closed__0, &lp_batteries_Nat_sum__trichotomy___closed__0_once, _init_lp_batteries_Nat_sum__trichotomy___closed__0);
return v___x_183_;
}
else
{
lean_object* v___x_184_; 
v___x_184_ = lean_obj_once(&lp_batteries_Nat_sum__trichotomy___closed__1, &lp_batteries_Nat_sum__trichotomy___closed__1_once, _init_lp_batteries_Nat_sum__trichotomy___closed__1);
return v___x_184_;
}
}
else
{
lean_object* v___x_185_; 
v___x_185_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_185_, 0, lean_box(0));
return v___x_185_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_sum__trichotomy___boxed(lean_object* v_a_186_, lean_object* v_b_187_){
_start:
{
lean_object* v_res_188_; 
v_res_188_ = lp_batteries_Nat_sum__trichotomy(v_a_186_, v_b_187_);
lean_dec(v_b_187_);
lean_dec(v_a_186_);
return v_res_188_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Alias(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Data_Nat_Basic(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Data_Nat_Lemmas(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Alias(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_Nat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Data_Nat_Lemmas(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_batteries_Nat_recDiagAux__zero__right___auto__1 = _init_lp_batteries_Nat_recDiagAux__zero__right___auto__1();
lean_mark_persistent(lp_batteries_Nat_recDiagAux__zero__right___auto__1);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Alias(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_Nat_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Data_Nat_Lemmas(uint8_t builtin) {
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
res = initialize_batteries_Batteries_Data_Nat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_Nat_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Data_Nat_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Data_Nat_Lemmas(builtin);
}
#ifdef __cplusplus
}
#endif
