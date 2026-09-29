// Lean compiler output
// Module: Mathlib.Data.Finset.Sort
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Max public import Mathlib.Data.Fintype.EquivFin public import Mathlib.Data.List.Pairwise public import Mathlib.Data.Multiset.Sort public import Mathlib.Order.RelIso.Set
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
lean_object* lean_string_utf8_byte_size(lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_sort___redArg(lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* lp_mathlib_Fin_castOrderIso(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_List_Nodup_getEquiv___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_subtypeEquivProp(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_toEmbedding___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Embedding_subtype___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Function_Embedding_trans___redArg___lam__0(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Std_instToFormatFormat___lam__0___boxed(lean_object*);
lean_object* l_repr(lean_object*, lean_object*, lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Std_Format_joinSep___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_string_length(lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* lp_mathlib_Equiv_Set_univ(lean_object*);
lean_object* lp_mathlib_Fin_castOrderIso___lam__0___boxed(lean_object*);
static const lean_string_object lp_mathlib_Finset_sort___auto__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Finset_sort___auto__1___closed__0 = (const lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__0_value;
static const lean_string_object lp_mathlib_Finset_sort___auto__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Finset_sort___auto__1___closed__1 = (const lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__1_value;
static const lean_string_object lp_mathlib_Finset_sort___auto__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Finset_sort___auto__1___closed__2 = (const lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__2_value;
static const lean_string_object lp_mathlib_Finset_sort___auto__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Finset_sort___auto__1___closed__3 = (const lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Finset_sort___auto__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Finset_sort___auto__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Finset_sort___auto__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Finset_sort___auto__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_Finset_sort___auto__1___closed__4 = (const lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__4_value;
static const lean_array_object lp_mathlib_Finset_sort___auto__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Finset_sort___auto__1___closed__5 = (const lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__5_value;
static const lean_string_object lp_mathlib_Finset_sort___auto__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Finset_sort___auto__1___closed__6 = (const lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Finset_sort___auto__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Finset_sort___auto__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Finset_sort___auto__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__7_value_aux_1),((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Finset_sort___auto__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__7_value_aux_2),((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_Finset_sort___auto__1___closed__7 = (const lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__7_value;
static const lean_string_object lp_mathlib_Finset_sort___auto__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Finset_sort___auto__1___closed__8 = (const lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Finset_sort___auto__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Finset_sort___auto__1___closed__9 = (const lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__9_value;
static const lean_string_object lp_mathlib_Finset_sort___auto__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "exact"};
static const lean_object* lp_mathlib_Finset_sort___auto__1___closed__10 = (const lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__10_value;
static const lean_ctor_object lp_mathlib_Finset_sort___auto__1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Finset_sort___auto__1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Finset_sort___auto__1___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__11_value_aux_1),((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Finset_sort___auto__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__11_value_aux_2),((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(108, 106, 111, 83, 219, 207, 32, 208)}};
static const lean_object* lp_mathlib_Finset_sort___auto__1___closed__11 = (const lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__11_value;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__12;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__13;
static const lean_string_object lp_mathlib_Finset_sort___auto__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Finset_sort___auto__1___closed__14 = (const lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__14_value;
static const lean_string_object lp_mathlib_Finset_sort___auto__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "fun"};
static const lean_object* lp_mathlib_Finset_sort___auto__1___closed__15 = (const lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__15_value;
static const lean_ctor_object lp_mathlib_Finset_sort___auto__1___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Finset_sort___auto__1___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Finset_sort___auto__1___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__16_value_aux_1),((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Finset_sort___auto__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__16_value_aux_2),((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(249, 155, 133, 242, 71, 132, 191, 97)}};
static const lean_object* lp_mathlib_Finset_sort___auto__1___closed__16 = (const lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__16_value;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__17;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__18;
static const lean_string_object lp_mathlib_Finset_sort___auto__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "basicFun"};
static const lean_object* lp_mathlib_Finset_sort___auto__1___closed__19 = (const lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__19_value;
static const lean_ctor_object lp_mathlib_Finset_sort___auto__1___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Finset_sort___auto__1___closed__20_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__20_value_aux_0),((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Finset_sort___auto__1___closed__20_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__20_value_aux_1),((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Finset_sort___auto__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__20_value_aux_2),((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__19_value),LEAN_SCALAR_PTR_LITERAL(209, 134, 40, 160, 122, 195, 31, 223)}};
static const lean_object* lp_mathlib_Finset_sort___auto__1___closed__20 = (const lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__20_value;
static const lean_string_object lp_mathlib_Finset_sort___auto__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "a"};
static const lean_object* lp_mathlib_Finset_sort___auto__1___closed__21 = (const lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__21_value;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__22;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__23;
static const lean_ctor_object lp_mathlib_Finset_sort___auto__1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(247, 80, 99, 121, 74, 33, 203, 108)}};
static const lean_object* lp_mathlib_Finset_sort___auto__1___closed__24 = (const lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__24_value;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__25;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__26;
static const lean_string_object lp_mathlib_Finset_sort___auto__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "b"};
static const lean_object* lp_mathlib_Finset_sort___auto__1___closed__27 = (const lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__27_value;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__28;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__29;
static const lean_ctor_object lp_mathlib_Finset_sort___auto__1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__27_value),LEAN_SCALAR_PTR_LITERAL(47, 22, 244, 233, 226, 169, 241, 142)}};
static const lean_object* lp_mathlib_Finset_sort___auto__1___closed__30 = (const lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__30_value;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__31;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__32;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__33;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__34_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__34;
static const lean_ctor_object lp_mathlib_Finset_sort___auto__1___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__9_value),((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__5_value)}};
static const lean_object* lp_mathlib_Finset_sort___auto__1___closed__35 = (const lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__35_value;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__36_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__36;
static const lean_string_object lp_mathlib_Finset_sort___auto__1___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=>"};
static const lean_object* lp_mathlib_Finset_sort___auto__1___closed__37 = (const lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__37_value;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__38_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__38;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__39_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__39;
static const lean_string_object lp_mathlib_Finset_sort___auto__1___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_≤_"};
static const lean_object* lp_mathlib_Finset_sort___auto__1___closed__40 = (const lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__40_value;
static const lean_ctor_object lp_mathlib_Finset_sort___auto__1___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__40_value),LEAN_SCALAR_PTR_LITERAL(111, 3, 61, 112, 38, 138, 106, 121)}};
static const lean_object* lp_mathlib_Finset_sort___auto__1___closed__41 = (const lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__41_value;
static const lean_string_object lp_mathlib_Finset_sort___auto__1___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "≤"};
static const lean_object* lp_mathlib_Finset_sort___auto__1___closed__42 = (const lean_object*)&lp_mathlib_Finset_sort___auto__1___closed__42_value;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__43_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__43;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__44_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__44;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__45_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__45;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__46_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__46;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__47_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__47;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__48_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__48;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__49_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__49;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__50_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__50;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__51_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__51;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__52_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__52;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__53_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__53;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__54_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__54;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__55_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__55;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__56_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__56;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__57_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__57;
static lean_once_cell_t lp_mathlib_Finset_sort___auto__1___closed__58_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_sort___auto__1___closed__58;
LEAN_EXPORT lean_object* lp_mathlib_Finset_sort___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Finset_sort___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sort(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finset_orderIsoOfFin___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_orderIsoOfFin___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Finset_orderIsoOfFin___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_orderIsoOfFin___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Finset_orderIsoOfFin___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_orderIsoOfFin___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_orderIsoOfFin(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_orderIsoOfFin___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Finset_orderEmbOfFin___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Function_Embedding_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Finset_orderEmbOfFin___redArg___closed__0 = (const lean_object*)&lp_mathlib_Finset_orderEmbOfFin___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Finset_orderEmbOfFin___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_orderEmbOfFin___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_orderEmbOfFin(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_orderEmbOfFin___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Finset_orderEmbOfCardLe___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Fin_castOrderIso___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Finset_orderEmbOfCardLe___redArg___closed__0 = (const lean_object*)&lp_mathlib_Finset_orderEmbOfCardLe___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Finset_orderEmbOfCardLe___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_orderEmbOfCardLe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_orderEmbOfCardLe___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Finset_instRepr___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "{"};
static const lean_object* lp_mathlib_Finset_instRepr___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Finset_instRepr___redArg___lam__0___closed__0_value;
static const lean_closure_object lp_mathlib_Finset_instRepr___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Std_instToFormatFormat___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Finset_instRepr___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Finset_instRepr___redArg___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Finset_instRepr___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Finset_instRepr___redArg___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Finset_instRepr___redArg___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib_Finset_instRepr___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Finset_instRepr___redArg___lam__0___closed__2_value)}};
static const lean_object* lp_mathlib_Finset_instRepr___redArg___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Finset_instRepr___redArg___lam__0___closed__3_value;
static const lean_ctor_object lp_mathlib_Finset_instRepr___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Finset_instRepr___redArg___lam__0___closed__3_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_Finset_instRepr___redArg___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Finset_instRepr___redArg___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Finset_instRepr___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "}"};
static const lean_object* lp_mathlib_Finset_instRepr___redArg___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Finset_instRepr___redArg___lam__0___closed__5_value;
static lean_once_cell_t lp_mathlib_Finset_instRepr___redArg___lam__0___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_instRepr___redArg___lam__0___closed__6;
static lean_once_cell_t lp_mathlib_Finset_instRepr___redArg___lam__0___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_instRepr___redArg___lam__0___closed__7;
static const lean_ctor_object lp_mathlib_Finset_instRepr___redArg___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Finset_instRepr___redArg___lam__0___closed__0_value)}};
static const lean_object* lp_mathlib_Finset_instRepr___redArg___lam__0___closed__8 = (const lean_object*)&lp_mathlib_Finset_instRepr___redArg___lam__0___closed__8_value;
static const lean_ctor_object lp_mathlib_Finset_instRepr___redArg___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Finset_instRepr___redArg___lam__0___closed__5_value)}};
static const lean_object* lp_mathlib_Finset_instRepr___redArg___lam__0___closed__9 = (const lean_object*)&lp_mathlib_Finset_instRepr___redArg___lam__0___closed__9_value;
static const lean_string_object lp_mathlib_Finset_instRepr___redArg___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "0"};
static const lean_object* lp_mathlib_Finset_instRepr___redArg___lam__0___closed__10 = (const lean_object*)&lp_mathlib_Finset_instRepr___redArg___lam__0___closed__10_value;
static const lean_ctor_object lp_mathlib_Finset_instRepr___redArg___lam__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Finset_instRepr___redArg___lam__0___closed__10_value)}};
static const lean_object* lp_mathlib_Finset_instRepr___redArg___lam__0___closed__11 = (const lean_object*)&lp_mathlib_Finset_instRepr___redArg___lam__0___closed__11_value;
static const lean_string_object lp_mathlib_Finset_instRepr___redArg___lam__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "∅"};
static const lean_object* lp_mathlib_Finset_instRepr___redArg___lam__0___closed__12 = (const lean_object*)&lp_mathlib_Finset_instRepr___redArg___lam__0___closed__12_value;
static const lean_ctor_object lp_mathlib_Finset_instRepr___redArg___lam__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Finset_instRepr___redArg___lam__0___closed__12_value)}};
static const lean_object* lp_mathlib_Finset_instRepr___redArg___lam__0___closed__13 = (const lean_object*)&lp_mathlib_Finset_instRepr___redArg___lam__0___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib_Finset_instRepr___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instRepr___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instRepr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_instRepr(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Fintype_orderIsoFinOfCardEq___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Fintype_orderIsoFinOfCardEq___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Fintype_orderIsoFinOfCardEq___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Fintype_orderIsoFinOfCardEq___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Fintype_orderIsoFinOfCardEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_orderIsoFinOfCardEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_orderIsoFinOfCardEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_orderIsoFinOfCardEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__12(void){
_start:
{
lean_object* v___x_27_; lean_object* v___x_28_; 
v___x_27_ = ((lean_object*)(lp_mathlib_Finset_sort___auto__1___closed__10));
v___x_28_ = l_Lean_mkAtom(v___x_27_);
return v___x_28_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__13(void){
_start:
{
lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v___x_31_; 
v___x_29_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__12, &lp_mathlib_Finset_sort___auto__1___closed__12_once, _init_lp_mathlib_Finset_sort___auto__1___closed__12);
v___x_30_ = ((lean_object*)(lp_mathlib_Finset_sort___auto__1___closed__5));
v___x_31_ = lean_array_push(v___x_30_, v___x_29_);
return v___x_31_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__17(void){
_start:
{
lean_object* v___x_39_; lean_object* v___x_40_; 
v___x_39_ = ((lean_object*)(lp_mathlib_Finset_sort___auto__1___closed__15));
v___x_40_ = l_Lean_mkAtom(v___x_39_);
return v___x_40_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__18(void){
_start:
{
lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; 
v___x_41_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__17, &lp_mathlib_Finset_sort___auto__1___closed__17_once, _init_lp_mathlib_Finset_sort___auto__1___closed__17);
v___x_42_ = ((lean_object*)(lp_mathlib_Finset_sort___auto__1___closed__5));
v___x_43_ = lean_array_push(v___x_42_, v___x_41_);
return v___x_43_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__22(void){
_start:
{
lean_object* v___x_51_; lean_object* v___x_52_; 
v___x_51_ = ((lean_object*)(lp_mathlib_Finset_sort___auto__1___closed__21));
v___x_52_ = lean_string_utf8_byte_size(v___x_51_);
return v___x_52_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__23(void){
_start:
{
lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_53_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__22, &lp_mathlib_Finset_sort___auto__1___closed__22_once, _init_lp_mathlib_Finset_sort___auto__1___closed__22);
v___x_54_ = lean_unsigned_to_nat(0u);
v___x_55_ = ((lean_object*)(lp_mathlib_Finset_sort___auto__1___closed__21));
v___x_56_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_56_, 0, v___x_55_);
lean_ctor_set(v___x_56_, 1, v___x_54_);
lean_ctor_set(v___x_56_, 2, v___x_53_);
return v___x_56_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__25(void){
_start:
{
lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_59_ = lean_box(0);
v___x_60_ = ((lean_object*)(lp_mathlib_Finset_sort___auto__1___closed__24));
v___x_61_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__23, &lp_mathlib_Finset_sort___auto__1___closed__23_once, _init_lp_mathlib_Finset_sort___auto__1___closed__23);
v___x_62_ = lean_box(2);
v___x_63_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_63_, 0, v___x_62_);
lean_ctor_set(v___x_63_, 1, v___x_61_);
lean_ctor_set(v___x_63_, 2, v___x_60_);
lean_ctor_set(v___x_63_, 3, v___x_59_);
return v___x_63_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__26(void){
_start:
{
lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; 
v___x_64_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__25, &lp_mathlib_Finset_sort___auto__1___closed__25_once, _init_lp_mathlib_Finset_sort___auto__1___closed__25);
v___x_65_ = ((lean_object*)(lp_mathlib_Finset_sort___auto__1___closed__5));
v___x_66_ = lean_array_push(v___x_65_, v___x_64_);
return v___x_66_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__28(void){
_start:
{
lean_object* v___x_68_; lean_object* v___x_69_; 
v___x_68_ = ((lean_object*)(lp_mathlib_Finset_sort___auto__1___closed__27));
v___x_69_ = lean_string_utf8_byte_size(v___x_68_);
return v___x_69_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__29(void){
_start:
{
lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; 
v___x_70_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__28, &lp_mathlib_Finset_sort___auto__1___closed__28_once, _init_lp_mathlib_Finset_sort___auto__1___closed__28);
v___x_71_ = lean_unsigned_to_nat(0u);
v___x_72_ = ((lean_object*)(lp_mathlib_Finset_sort___auto__1___closed__27));
v___x_73_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_73_, 0, v___x_72_);
lean_ctor_set(v___x_73_, 1, v___x_71_);
lean_ctor_set(v___x_73_, 2, v___x_70_);
return v___x_73_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__31(void){
_start:
{
lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; 
v___x_76_ = lean_box(0);
v___x_77_ = ((lean_object*)(lp_mathlib_Finset_sort___auto__1___closed__30));
v___x_78_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__29, &lp_mathlib_Finset_sort___auto__1___closed__29_once, _init_lp_mathlib_Finset_sort___auto__1___closed__29);
v___x_79_ = lean_box(2);
v___x_80_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_80_, 0, v___x_79_);
lean_ctor_set(v___x_80_, 1, v___x_78_);
lean_ctor_set(v___x_80_, 2, v___x_77_);
lean_ctor_set(v___x_80_, 3, v___x_76_);
return v___x_80_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__32(void){
_start:
{
lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; 
v___x_81_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__31, &lp_mathlib_Finset_sort___auto__1___closed__31_once, _init_lp_mathlib_Finset_sort___auto__1___closed__31);
v___x_82_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__26, &lp_mathlib_Finset_sort___auto__1___closed__26_once, _init_lp_mathlib_Finset_sort___auto__1___closed__26);
v___x_83_ = lean_array_push(v___x_82_, v___x_81_);
return v___x_83_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__33(void){
_start:
{
lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; 
v___x_84_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__32, &lp_mathlib_Finset_sort___auto__1___closed__32_once, _init_lp_mathlib_Finset_sort___auto__1___closed__32);
v___x_85_ = ((lean_object*)(lp_mathlib_Finset_sort___auto__1___closed__9));
v___x_86_ = lean_box(2);
v___x_87_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_87_, 0, v___x_86_);
lean_ctor_set(v___x_87_, 1, v___x_85_);
lean_ctor_set(v___x_87_, 2, v___x_84_);
return v___x_87_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__34(void){
_start:
{
lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; 
v___x_88_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__33, &lp_mathlib_Finset_sort___auto__1___closed__33_once, _init_lp_mathlib_Finset_sort___auto__1___closed__33);
v___x_89_ = ((lean_object*)(lp_mathlib_Finset_sort___auto__1___closed__5));
v___x_90_ = lean_array_push(v___x_89_, v___x_88_);
return v___x_90_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__36(void){
_start:
{
lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; 
v___x_95_ = ((lean_object*)(lp_mathlib_Finset_sort___auto__1___closed__35));
v___x_96_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__34, &lp_mathlib_Finset_sort___auto__1___closed__34_once, _init_lp_mathlib_Finset_sort___auto__1___closed__34);
v___x_97_ = lean_array_push(v___x_96_, v___x_95_);
return v___x_97_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__38(void){
_start:
{
lean_object* v___x_99_; lean_object* v___x_100_; 
v___x_99_ = ((lean_object*)(lp_mathlib_Finset_sort___auto__1___closed__37));
v___x_100_ = l_Lean_mkAtom(v___x_99_);
return v___x_100_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__39(void){
_start:
{
lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; 
v___x_101_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__38, &lp_mathlib_Finset_sort___auto__1___closed__38_once, _init_lp_mathlib_Finset_sort___auto__1___closed__38);
v___x_102_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__36, &lp_mathlib_Finset_sort___auto__1___closed__36_once, _init_lp_mathlib_Finset_sort___auto__1___closed__36);
v___x_103_ = lean_array_push(v___x_102_, v___x_101_);
return v___x_103_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__43(void){
_start:
{
lean_object* v___x_108_; lean_object* v___x_109_; 
v___x_108_ = ((lean_object*)(lp_mathlib_Finset_sort___auto__1___closed__42));
v___x_109_ = l_Lean_mkAtom(v___x_108_);
return v___x_109_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__44(void){
_start:
{
lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; 
v___x_110_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__43, &lp_mathlib_Finset_sort___auto__1___closed__43_once, _init_lp_mathlib_Finset_sort___auto__1___closed__43);
v___x_111_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__26, &lp_mathlib_Finset_sort___auto__1___closed__26_once, _init_lp_mathlib_Finset_sort___auto__1___closed__26);
v___x_112_ = lean_array_push(v___x_111_, v___x_110_);
return v___x_112_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__45(void){
_start:
{
lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; 
v___x_113_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__31, &lp_mathlib_Finset_sort___auto__1___closed__31_once, _init_lp_mathlib_Finset_sort___auto__1___closed__31);
v___x_114_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__44, &lp_mathlib_Finset_sort___auto__1___closed__44_once, _init_lp_mathlib_Finset_sort___auto__1___closed__44);
v___x_115_ = lean_array_push(v___x_114_, v___x_113_);
return v___x_115_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__46(void){
_start:
{
lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; 
v___x_116_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__45, &lp_mathlib_Finset_sort___auto__1___closed__45_once, _init_lp_mathlib_Finset_sort___auto__1___closed__45);
v___x_117_ = ((lean_object*)(lp_mathlib_Finset_sort___auto__1___closed__41));
v___x_118_ = lean_box(2);
v___x_119_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_119_, 0, v___x_118_);
lean_ctor_set(v___x_119_, 1, v___x_117_);
lean_ctor_set(v___x_119_, 2, v___x_116_);
return v___x_119_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__47(void){
_start:
{
lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; 
v___x_120_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__46, &lp_mathlib_Finset_sort___auto__1___closed__46_once, _init_lp_mathlib_Finset_sort___auto__1___closed__46);
v___x_121_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__39, &lp_mathlib_Finset_sort___auto__1___closed__39_once, _init_lp_mathlib_Finset_sort___auto__1___closed__39);
v___x_122_ = lean_array_push(v___x_121_, v___x_120_);
return v___x_122_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__48(void){
_start:
{
lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; 
v___x_123_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__47, &lp_mathlib_Finset_sort___auto__1___closed__47_once, _init_lp_mathlib_Finset_sort___auto__1___closed__47);
v___x_124_ = ((lean_object*)(lp_mathlib_Finset_sort___auto__1___closed__20));
v___x_125_ = lean_box(2);
v___x_126_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_126_, 0, v___x_125_);
lean_ctor_set(v___x_126_, 1, v___x_124_);
lean_ctor_set(v___x_126_, 2, v___x_123_);
return v___x_126_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__49(void){
_start:
{
lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; 
v___x_127_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__48, &lp_mathlib_Finset_sort___auto__1___closed__48_once, _init_lp_mathlib_Finset_sort___auto__1___closed__48);
v___x_128_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__18, &lp_mathlib_Finset_sort___auto__1___closed__18_once, _init_lp_mathlib_Finset_sort___auto__1___closed__18);
v___x_129_ = lean_array_push(v___x_128_, v___x_127_);
return v___x_129_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__50(void){
_start:
{
lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; 
v___x_130_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__49, &lp_mathlib_Finset_sort___auto__1___closed__49_once, _init_lp_mathlib_Finset_sort___auto__1___closed__49);
v___x_131_ = ((lean_object*)(lp_mathlib_Finset_sort___auto__1___closed__16));
v___x_132_ = lean_box(2);
v___x_133_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_133_, 0, v___x_132_);
lean_ctor_set(v___x_133_, 1, v___x_131_);
lean_ctor_set(v___x_133_, 2, v___x_130_);
return v___x_133_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__51(void){
_start:
{
lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; 
v___x_134_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__50, &lp_mathlib_Finset_sort___auto__1___closed__50_once, _init_lp_mathlib_Finset_sort___auto__1___closed__50);
v___x_135_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__13, &lp_mathlib_Finset_sort___auto__1___closed__13_once, _init_lp_mathlib_Finset_sort___auto__1___closed__13);
v___x_136_ = lean_array_push(v___x_135_, v___x_134_);
return v___x_136_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__52(void){
_start:
{
lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; 
v___x_137_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__51, &lp_mathlib_Finset_sort___auto__1___closed__51_once, _init_lp_mathlib_Finset_sort___auto__1___closed__51);
v___x_138_ = ((lean_object*)(lp_mathlib_Finset_sort___auto__1___closed__11));
v___x_139_ = lean_box(2);
v___x_140_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_140_, 0, v___x_139_);
lean_ctor_set(v___x_140_, 1, v___x_138_);
lean_ctor_set(v___x_140_, 2, v___x_137_);
return v___x_140_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__53(void){
_start:
{
lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; 
v___x_141_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__52, &lp_mathlib_Finset_sort___auto__1___closed__52_once, _init_lp_mathlib_Finset_sort___auto__1___closed__52);
v___x_142_ = ((lean_object*)(lp_mathlib_Finset_sort___auto__1___closed__5));
v___x_143_ = lean_array_push(v___x_142_, v___x_141_);
return v___x_143_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__54(void){
_start:
{
lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; 
v___x_144_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__53, &lp_mathlib_Finset_sort___auto__1___closed__53_once, _init_lp_mathlib_Finset_sort___auto__1___closed__53);
v___x_145_ = ((lean_object*)(lp_mathlib_Finset_sort___auto__1___closed__9));
v___x_146_ = lean_box(2);
v___x_147_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_147_, 0, v___x_146_);
lean_ctor_set(v___x_147_, 1, v___x_145_);
lean_ctor_set(v___x_147_, 2, v___x_144_);
return v___x_147_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__55(void){
_start:
{
lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; 
v___x_148_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__54, &lp_mathlib_Finset_sort___auto__1___closed__54_once, _init_lp_mathlib_Finset_sort___auto__1___closed__54);
v___x_149_ = ((lean_object*)(lp_mathlib_Finset_sort___auto__1___closed__5));
v___x_150_ = lean_array_push(v___x_149_, v___x_148_);
return v___x_150_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__56(void){
_start:
{
lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; 
v___x_151_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__55, &lp_mathlib_Finset_sort___auto__1___closed__55_once, _init_lp_mathlib_Finset_sort___auto__1___closed__55);
v___x_152_ = ((lean_object*)(lp_mathlib_Finset_sort___auto__1___closed__7));
v___x_153_ = lean_box(2);
v___x_154_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_154_, 0, v___x_153_);
lean_ctor_set(v___x_154_, 1, v___x_152_);
lean_ctor_set(v___x_154_, 2, v___x_151_);
return v___x_154_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__57(void){
_start:
{
lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; 
v___x_155_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__56, &lp_mathlib_Finset_sort___auto__1___closed__56_once, _init_lp_mathlib_Finset_sort___auto__1___closed__56);
v___x_156_ = ((lean_object*)(lp_mathlib_Finset_sort___auto__1___closed__5));
v___x_157_ = lean_array_push(v___x_156_, v___x_155_);
return v___x_157_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1___closed__58(void){
_start:
{
lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; 
v___x_158_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__57, &lp_mathlib_Finset_sort___auto__1___closed__57_once, _init_lp_mathlib_Finset_sort___auto__1___closed__57);
v___x_159_ = ((lean_object*)(lp_mathlib_Finset_sort___auto__1___closed__4));
v___x_160_ = lean_box(2);
v___x_161_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_161_, 0, v___x_160_);
lean_ctor_set(v___x_161_, 1, v___x_159_);
lean_ctor_set(v___x_161_, 2, v___x_158_);
return v___x_161_;
}
}
static lean_object* _init_lp_mathlib_Finset_sort___auto__1(void){
_start:
{
lean_object* v___x_162_; 
v___x_162_ = lean_obj_once(&lp_mathlib_Finset_sort___auto__1___closed__58, &lp_mathlib_Finset_sort___auto__1___closed__58_once, _init_lp_mathlib_Finset_sort___auto__1___closed__58);
return v___x_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sort___redArg(lean_object* v_s_163_, lean_object* v_inst_164_){
_start:
{
lean_object* v___x_165_; 
v___x_165_ = lp_mathlib_Multiset_sort___redArg(v_s_163_, v_inst_164_);
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sort(lean_object* v_00_u03b1_166_, lean_object* v_s_167_, lean_object* v_r_168_, lean_object* v_inst_169_, lean_object* v_inst_170_, lean_object* v_inst_171_, lean_object* v_inst_172_){
_start:
{
lean_object* v___x_173_; 
v___x_173_ = lp_mathlib_Multiset_sort___redArg(v_s_167_, v_inst_169_);
return v___x_173_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finset_orderIsoOfFin___redArg___lam__0(lean_object* v_toDecidableEq_174_, lean_object* v_a_175_, lean_object* v_b_176_){
_start:
{
lean_object* v___x_177_; uint8_t v___x_178_; 
v___x_177_ = lean_apply_2(v_toDecidableEq_174_, v_a_175_, v_b_176_);
v___x_178_ = lean_unbox(v___x_177_);
return v___x_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_orderIsoOfFin___redArg___lam__0___boxed(lean_object* v_toDecidableEq_179_, lean_object* v_a_180_, lean_object* v_b_181_){
_start:
{
uint8_t v_res_182_; lean_object* v_r_183_; 
v_res_182_ = lp_mathlib_Finset_orderIsoOfFin___redArg___lam__0(v_toDecidableEq_179_, v_a_180_, v_b_181_);
v_r_183_ = lean_box(v_res_182_);
return v_r_183_;
}
}
static lean_object* _init_lp_mathlib_Finset_orderIsoOfFin___redArg___closed__0(void){
_start:
{
lean_object* v___x_184_; 
v___x_184_ = lp_mathlib_Equiv_subtypeEquivProp(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_orderIsoOfFin___redArg(lean_object* v_inst_185_, lean_object* v_s_186_, lean_object* v_k_187_){
_start:
{
lean_object* v_toDecidableLE_188_; lean_object* v_toDecidableEq_189_; lean_object* v___f_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; 
v_toDecidableLE_188_ = lean_ctor_get(v_inst_185_, 4);
lean_inc_ref(v_toDecidableLE_188_);
v_toDecidableEq_189_ = lean_ctor_get(v_inst_185_, 5);
lean_inc_ref(v_toDecidableEq_189_);
lean_dec_ref(v_inst_185_);
v___f_190_ = lean_alloc_closure((void*)(lp_mathlib_Finset_orderIsoOfFin___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_190_, 0, v_toDecidableEq_189_);
v___x_191_ = lp_mathlib_Multiset_sort___redArg(v_s_186_, v_toDecidableLE_188_);
v___x_192_ = l_List_lengthTR___redArg(v___x_191_);
v___x_193_ = lp_mathlib_Fin_castOrderIso(v___x_192_, v_k_187_, lean_box(0));
lean_dec(v___x_192_);
v___x_194_ = lp_mathlib_List_Nodup_getEquiv___redArg(v___f_190_, v___x_191_);
v___x_195_ = lean_obj_once(&lp_mathlib_Finset_orderIsoOfFin___redArg___closed__0, &lp_mathlib_Finset_orderIsoOfFin___redArg___closed__0_once, _init_lp_mathlib_Finset_orderIsoOfFin___redArg___closed__0);
v___x_196_ = lp_mathlib_Equiv_trans___redArg(v___x_194_, v___x_195_);
v___x_197_ = lp_mathlib_Equiv_trans___redArg(v___x_193_, v___x_196_);
return v___x_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_orderIsoOfFin___redArg___boxed(lean_object* v_inst_198_, lean_object* v_s_199_, lean_object* v_k_200_){
_start:
{
lean_object* v_res_201_; 
v_res_201_ = lp_mathlib_Finset_orderIsoOfFin___redArg(v_inst_198_, v_s_199_, v_k_200_);
lean_dec(v_k_200_);
return v_res_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_orderIsoOfFin(lean_object* v_00_u03b1_202_, lean_object* v_inst_203_, lean_object* v_s_204_, lean_object* v_k_205_, lean_object* v_h_206_){
_start:
{
lean_object* v___x_207_; 
v___x_207_ = lp_mathlib_Finset_orderIsoOfFin___redArg(v_inst_203_, v_s_204_, v_k_205_);
return v___x_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_orderIsoOfFin___boxed(lean_object* v_00_u03b1_208_, lean_object* v_inst_209_, lean_object* v_s_210_, lean_object* v_k_211_, lean_object* v_h_212_){
_start:
{
lean_object* v_res_213_; 
v_res_213_ = lp_mathlib_Finset_orderIsoOfFin(v_00_u03b1_208_, v_inst_209_, v_s_210_, v_k_211_, v_h_212_);
lean_dec(v_k_211_);
return v_res_213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_orderEmbOfFin___redArg(lean_object* v_inst_215_, lean_object* v_s_216_, lean_object* v_k_217_){
_start:
{
lean_object* v___x_218_; lean_object* v___f_219_; lean_object* v___f_220_; lean_object* v___f_221_; 
v___x_218_ = lp_mathlib_Finset_orderIsoOfFin___redArg(v_inst_215_, v_s_216_, v_k_217_);
v___f_219_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_219_, 0, v___x_218_);
v___f_220_ = ((lean_object*)(lp_mathlib_Finset_orderEmbOfFin___redArg___closed__0));
v___f_221_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_trans___redArg___lam__0), 3, 2);
lean_closure_set(v___f_221_, 0, v___f_219_);
lean_closure_set(v___f_221_, 1, v___f_220_);
return v___f_221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_orderEmbOfFin___redArg___boxed(lean_object* v_inst_222_, lean_object* v_s_223_, lean_object* v_k_224_){
_start:
{
lean_object* v_res_225_; 
v_res_225_ = lp_mathlib_Finset_orderEmbOfFin___redArg(v_inst_222_, v_s_223_, v_k_224_);
lean_dec(v_k_224_);
return v_res_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_orderEmbOfFin(lean_object* v_00_u03b1_226_, lean_object* v_inst_227_, lean_object* v_s_228_, lean_object* v_k_229_, lean_object* v_h_230_){
_start:
{
lean_object* v___x_231_; 
v___x_231_ = lp_mathlib_Finset_orderEmbOfFin___redArg(v_inst_227_, v_s_228_, v_k_229_);
return v___x_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_orderEmbOfFin___boxed(lean_object* v_00_u03b1_232_, lean_object* v_inst_233_, lean_object* v_s_234_, lean_object* v_k_235_, lean_object* v_h_236_){
_start:
{
lean_object* v_res_237_; 
v_res_237_ = lp_mathlib_Finset_orderEmbOfFin(v_00_u03b1_232_, v_inst_233_, v_s_234_, v_k_235_, v_h_236_);
lean_dec(v_k_235_);
return v_res_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_orderEmbOfCardLe___redArg(lean_object* v_inst_239_, lean_object* v_s_240_){
_start:
{
lean_object* v___x_241_; lean_object* v___f_242_; lean_object* v___x_243_; lean_object* v___f_244_; 
v___x_241_ = l_List_lengthTR___redArg(v_s_240_);
v___f_242_ = ((lean_object*)(lp_mathlib_Finset_orderEmbOfCardLe___redArg___closed__0));
v___x_243_ = lp_mathlib_Finset_orderEmbOfFin___redArg(v_inst_239_, v_s_240_, v___x_241_);
lean_dec(v___x_241_);
v___f_244_ = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_trans___redArg___lam__0), 3, 2);
lean_closure_set(v___f_244_, 0, v___f_242_);
lean_closure_set(v___f_244_, 1, v___x_243_);
return v___f_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_orderEmbOfCardLe(lean_object* v_00_u03b1_245_, lean_object* v_inst_246_, lean_object* v_s_247_, lean_object* v_k_248_, lean_object* v_h_249_){
_start:
{
lean_object* v___x_250_; 
v___x_250_ = lp_mathlib_Finset_orderEmbOfCardLe___redArg(v_inst_246_, v_s_247_);
return v___x_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_orderEmbOfCardLe___boxed(lean_object* v_00_u03b1_251_, lean_object* v_inst_252_, lean_object* v_s_253_, lean_object* v_k_254_, lean_object* v_h_255_){
_start:
{
lean_object* v_res_256_; 
v_res_256_ = lp_mathlib_Finset_orderEmbOfCardLe(v_00_u03b1_251_, v_inst_252_, v_s_253_, v_k_254_, v_h_255_);
lean_dec(v_k_254_);
return v_res_256_;
}
}
static lean_object* _init_lp_mathlib_Finset_instRepr___redArg___lam__0___closed__6(void){
_start:
{
lean_object* v___x_266_; lean_object* v___x_267_; 
v___x_266_ = ((lean_object*)(lp_mathlib_Finset_instRepr___redArg___lam__0___closed__0));
v___x_267_ = lean_string_length(v___x_266_);
return v___x_267_;
}
}
static lean_object* _init_lp_mathlib_Finset_instRepr___redArg___lam__0___closed__7(void){
_start:
{
lean_object* v___x_268_; lean_object* v___x_269_; 
v___x_268_ = lean_obj_once(&lp_mathlib_Finset_instRepr___redArg___lam__0___closed__6, &lp_mathlib_Finset_instRepr___redArg___lam__0___closed__6_once, _init_lp_mathlib_Finset_instRepr___redArg___lam__0___closed__6);
v___x_269_ = lean_nat_to_int(v___x_268_);
return v___x_269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instRepr___redArg___lam__0(lean_object* v_inst_280_, lean_object* v_s_281_, lean_object* v_x_282_){
_start:
{
lean_object* v___x_283_; lean_object* v___x_284_; uint8_t v___x_285_; 
v___x_283_ = l_List_lengthTR___redArg(v_s_281_);
v___x_284_ = lean_unsigned_to_nat(0u);
v___x_285_ = lean_nat_dec_eq(v___x_283_, v___x_284_);
lean_dec(v___x_283_);
if (v___x_285_ == 0)
{
if (v___x_285_ == 0)
{
lean_object* v___f_286_; lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; uint8_t v___x_298_; lean_object* v___x_299_; 
v___f_286_ = ((lean_object*)(lp_mathlib_Finset_instRepr___redArg___lam__0___closed__1));
v___x_287_ = lean_alloc_closure((void*)(l_repr), 3, 2);
lean_closure_set(v___x_287_, 0, lean_box(0));
lean_closure_set(v___x_287_, 1, v_inst_280_);
v___x_288_ = lean_box(0);
v___x_289_ = l_List_mapTR_loop___redArg(v___x_287_, v_s_281_, v___x_288_);
v___x_290_ = ((lean_object*)(lp_mathlib_Finset_instRepr___redArg___lam__0___closed__4));
v___x_291_ = l_Std_Format_joinSep___redArg(v___f_286_, v___x_289_, v___x_290_);
v___x_292_ = lean_obj_once(&lp_mathlib_Finset_instRepr___redArg___lam__0___closed__7, &lp_mathlib_Finset_instRepr___redArg___lam__0___closed__7_once, _init_lp_mathlib_Finset_instRepr___redArg___lam__0___closed__7);
v___x_293_ = ((lean_object*)(lp_mathlib_Finset_instRepr___redArg___lam__0___closed__8));
v___x_294_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_294_, 0, v___x_293_);
lean_ctor_set(v___x_294_, 1, v___x_291_);
v___x_295_ = ((lean_object*)(lp_mathlib_Finset_instRepr___redArg___lam__0___closed__9));
v___x_296_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_296_, 0, v___x_294_);
lean_ctor_set(v___x_296_, 1, v___x_295_);
v___x_297_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_297_, 0, v___x_292_);
lean_ctor_set(v___x_297_, 1, v___x_296_);
v___x_298_ = 0;
v___x_299_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_299_, 0, v___x_297_);
lean_ctor_set_uint8(v___x_299_, sizeof(void*)*1, v___x_298_);
return v___x_299_;
}
else
{
lean_object* v___x_300_; 
lean_dec(v_s_281_);
lean_dec_ref(v_inst_280_);
v___x_300_ = ((lean_object*)(lp_mathlib_Finset_instRepr___redArg___lam__0___closed__11));
return v___x_300_;
}
}
else
{
lean_object* v___x_301_; 
lean_dec(v_s_281_);
lean_dec_ref(v_inst_280_);
v___x_301_ = ((lean_object*)(lp_mathlib_Finset_instRepr___redArg___lam__0___closed__13));
return v___x_301_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instRepr___redArg___lam__0___boxed(lean_object* v_inst_302_, lean_object* v_s_303_, lean_object* v_x_304_){
_start:
{
lean_object* v_res_305_; 
v_res_305_ = lp_mathlib_Finset_instRepr___redArg___lam__0(v_inst_302_, v_s_303_, v_x_304_);
lean_dec(v_x_304_);
return v_res_305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instRepr___redArg(lean_object* v_inst_306_){
_start:
{
lean_object* v___f_307_; 
v___f_307_ = lean_alloc_closure((void*)(lp_mathlib_Finset_instRepr___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_307_, 0, v_inst_306_);
return v___f_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_instRepr(lean_object* v_00_u03b1_308_, lean_object* v_inst_309_){
_start:
{
lean_object* v___f_310_; 
v___f_310_ = lean_alloc_closure((void*)(lp_mathlib_Finset_instRepr___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_310_, 0, v_inst_309_);
return v___f_310_;
}
}
static lean_object* _init_lp_mathlib_Fintype_orderIsoFinOfCardEq___redArg___closed__0(void){
_start:
{
lean_object* v___x_311_; 
v___x_311_ = lp_mathlib_Equiv_Set_univ(lean_box(0));
return v___x_311_;
}
}
static lean_object* _init_lp_mathlib_Fintype_orderIsoFinOfCardEq___redArg___closed__1(void){
_start:
{
lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; 
v___x_312_ = lean_obj_once(&lp_mathlib_Fintype_orderIsoFinOfCardEq___redArg___closed__0, &lp_mathlib_Fintype_orderIsoFinOfCardEq___redArg___closed__0_once, _init_lp_mathlib_Fintype_orderIsoFinOfCardEq___redArg___closed__0);
v___x_313_ = lean_obj_once(&lp_mathlib_Finset_orderIsoOfFin___redArg___closed__0, &lp_mathlib_Finset_orderIsoOfFin___redArg___closed__0_once, _init_lp_mathlib_Finset_orderIsoOfFin___redArg___closed__0);
v___x_314_ = lp_mathlib_Equiv_trans___redArg(v___x_313_, v___x_312_);
return v___x_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_orderIsoFinOfCardEq___redArg(lean_object* v_inst_315_, lean_object* v_inst_316_, lean_object* v_k_317_){
_start:
{
lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; 
v___x_318_ = lp_mathlib_Finset_orderIsoOfFin___redArg(v_inst_315_, v_inst_316_, v_k_317_);
v___x_319_ = lean_obj_once(&lp_mathlib_Fintype_orderIsoFinOfCardEq___redArg___closed__1, &lp_mathlib_Fintype_orderIsoFinOfCardEq___redArg___closed__1_once, _init_lp_mathlib_Fintype_orderIsoFinOfCardEq___redArg___closed__1);
v___x_320_ = lp_mathlib_Equiv_trans___redArg(v___x_318_, v___x_319_);
return v___x_320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_orderIsoFinOfCardEq___redArg___boxed(lean_object* v_inst_321_, lean_object* v_inst_322_, lean_object* v_k_323_){
_start:
{
lean_object* v_res_324_; 
v_res_324_ = lp_mathlib_Fintype_orderIsoFinOfCardEq___redArg(v_inst_321_, v_inst_322_, v_k_323_);
lean_dec(v_k_323_);
return v_res_324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_orderIsoFinOfCardEq(lean_object* v_00_u03b1_325_, lean_object* v_inst_326_, lean_object* v_inst_327_, lean_object* v_k_328_, lean_object* v_h_329_){
_start:
{
lean_object* v___x_330_; 
v___x_330_ = lp_mathlib_Fintype_orderIsoFinOfCardEq___redArg(v_inst_326_, v_inst_327_, v_k_328_);
return v___x_330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_orderIsoFinOfCardEq___boxed(lean_object* v_00_u03b1_331_, lean_object* v_inst_332_, lean_object* v_inst_333_, lean_object* v_k_334_, lean_object* v_h_335_){
_start:
{
lean_object* v_res_336_; 
v_res_336_ = lp_mathlib_Fintype_orderIsoFinOfCardEq(v_00_u03b1_331_, v_inst_332_, v_inst_333_, v_k_334_, v_h_335_);
lean_dec(v_k_334_);
return v_res_336_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Max(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_EquivFin(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Pairwise(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Sort(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_RelIso_Set(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Sort(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_EquivFin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Pairwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Sort(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_RelIso_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finset_Sort(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Finset_sort___auto__1 = _init_lp_mathlib_Finset_sort___auto__1();
lean_mark_persistent(lp_mathlib_Finset_sort___auto__1);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Max(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_EquivFin(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Pairwise(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Sort(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_RelIso_Set(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_Sort(uint8_t builtin) {
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
res = initialize_mathlib_Mathlib_Data_Fintype_EquivFin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Pairwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Sort(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_RelIso_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Sort(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finset_Sort(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finset_Sort(builtin);
}
#ifdef __cplusplus
}
#endif
