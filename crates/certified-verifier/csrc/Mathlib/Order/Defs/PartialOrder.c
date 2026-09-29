// Lean compiler output
// Module: Mathlib.Order.Defs.PartialOrder
// Imports: public import Init public meta import Init public import Batteries.Tactic.Alias public import Batteries.Tactic.Trans public import Mathlib.Tactic.ToDual
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__0 = (const lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__0_value;
static const lean_string_object lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__1 = (const lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__1_value;
static const lean_string_object lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__2 = (const lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__2_value;
static const lean_string_object lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__3 = (const lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__3_value;
static const lean_ctor_object lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__4 = (const lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__4_value;
static const lean_array_object lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__5 = (const lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__5_value;
static const lean_string_object lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__6 = (const lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__6_value;
static const lean_ctor_object lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__7_value_aux_1),((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__7_value_aux_2),((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__7 = (const lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__7_value;
static const lean_string_object lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__8 = (const lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__8_value;
static const lean_ctor_object lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__9 = (const lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__9_value;
static const lean_string_object lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "intros"};
static const lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__10 = (const lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__10_value;
static const lean_ctor_object lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__11_value_aux_1),((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__11_value_aux_2),((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__10_value),LEAN_SCALAR_PTR_LITERAL(26, 175, 18, 116, 252, 50, 128, 45)}};
static const lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__11 = (const lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__11_value;
static lean_once_cell_t lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__12;
static lean_once_cell_t lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__13;
static const lean_ctor_object lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__9_value),((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__5_value)}};
static const lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__14 = (const lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__14_value;
static lean_once_cell_t lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__15;
static lean_once_cell_t lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__16;
static lean_once_cell_t lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__17;
static const lean_string_object lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ";"};
static const lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__18 = (const lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__18_value;
static lean_once_cell_t lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__19;
static lean_once_cell_t lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__20;
static const lean_string_object lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticRfl"};
static const lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__21 = (const lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__21_value;
static const lean_ctor_object lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__22_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__22_value_aux_0),((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__22_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__22_value_aux_1),((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__22_value_aux_2),((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__21_value),LEAN_SCALAR_PTR_LITERAL(201, 188, 173, 198, 169, 252, 183, 45)}};
static const lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__22 = (const lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__22_value;
static const lean_string_object lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rfl"};
static const lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__23 = (const lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__23_value;
static lean_once_cell_t lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__24;
static lean_once_cell_t lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__25;
static lean_once_cell_t lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__26;
static lean_once_cell_t lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__27;
static lean_once_cell_t lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__28;
static lean_once_cell_t lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__29;
static lean_once_cell_t lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__30;
static lean_once_cell_t lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__31;
static lean_once_cell_t lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__32;
LEAN_EXPORT lean_object* lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_instTransLE(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTransLE___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTransLT(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTransLT___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTransLTLE(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTransLTLE___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTransLELT(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTransLELT___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTransGE(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTransGE___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTransGT(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTransGT___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTransGTGE(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTransGTGE___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTransGEGT(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTransGEGT___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_decidableLTOfDecidableLE___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_decidableLTOfDecidableLE___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_decidableLTOfDecidableLE(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_decidableLTOfDecidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u2a7f___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_⩿_"};
static const lean_object* lp_mathlib_term___u2a7f___00__closed__0 = (const lean_object*)&lp_mathlib_term___u2a7f___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u2a7f___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2a7f___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(167, 178, 115, 239, 180, 200, 69, 15)}};
static const lean_object* lp_mathlib_term___u2a7f___00__closed__1 = (const lean_object*)&lp_mathlib_term___u2a7f___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u2a7f___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term___u2a7f___00__closed__2 = (const lean_object*)&lp_mathlib_term___u2a7f___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u2a7f___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2a7f___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term___u2a7f___00__closed__3 = (const lean_object*)&lp_mathlib_term___u2a7f___00__closed__3_value;
static const lean_string_object lp_mathlib_term___u2a7f___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ⩿ "};
static const lean_object* lp_mathlib_term___u2a7f___00__closed__4 = (const lean_object*)&lp_mathlib_term___u2a7f___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u2a7f___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u2a7f___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u2a7f___00__closed__5 = (const lean_object*)&lp_mathlib_term___u2a7f___00__closed__5_value;
static const lean_string_object lp_mathlib_term___u2a7f___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term___u2a7f___00__closed__6 = (const lean_object*)&lp_mathlib_term___u2a7f___00__closed__6_value;
static const lean_ctor_object lp_mathlib_term___u2a7f___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2a7f___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term___u2a7f___00__closed__7 = (const lean_object*)&lp_mathlib_term___u2a7f___00__closed__7_value;
static const lean_ctor_object lp_mathlib_term___u2a7f___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term___u2a7f___00__closed__7_value),((lean_object*)(((size_t)(51) << 1) | 1))}};
static const lean_object* lp_mathlib_term___u2a7f___00__closed__8 = (const lean_object*)&lp_mathlib_term___u2a7f___00__closed__8_value;
static const lean_ctor_object lp_mathlib_term___u2a7f___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2a7f___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2a7f___00__closed__5_value),((lean_object*)&lp_mathlib_term___u2a7f___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u2a7f___00__closed__9 = (const lean_object*)&lp_mathlib_term___u2a7f___00__closed__9_value;
static const lean_ctor_object lp_mathlib_term___u2a7f___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u2a7f___00__closed__1_value),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)&lp_mathlib_term___u2a7f___00__closed__9_value)}};
static const lean_object* lp_mathlib_term___u2a7f___00__closed__10 = (const lean_object*)&lp_mathlib_term___u2a7f___00__closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u2a7f__ = (const lean_object*)&lp_mathlib_term___u2a7f___00__closed__10_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__1_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__2_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__2_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "WCovBy"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__3_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__4;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(179, 238, 125, 36, 69, 56, 223, 167)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__5_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__6_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__6_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______unexpand__WCovBy__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______unexpand__WCovBy__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______unexpand__WCovBy__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______unexpand__WCovBy__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______unexpand__WCovBy__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______unexpand__WCovBy__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______unexpand__WCovBy__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______unexpand__WCovBy__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______unexpand__WCovBy__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term___u22d6___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_⋖_"};
static const lean_object* lp_mathlib_term___u22d6___00__closed__0 = (const lean_object*)&lp_mathlib_term___u22d6___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term___u22d6___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term___u22d6___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(23, 78, 83, 12, 169, 91, 195, 146)}};
static const lean_object* lp_mathlib_term___u22d6___00__closed__1 = (const lean_object*)&lp_mathlib_term___u22d6___00__closed__1_value;
static const lean_string_object lp_mathlib_term___u22d6___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ⋖ "};
static const lean_object* lp_mathlib_term___u22d6___00__closed__2 = (const lean_object*)&lp_mathlib_term___u22d6___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term___u22d6___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term___u22d6___00__closed__2_value)}};
static const lean_object* lp_mathlib_term___u22d6___00__closed__3 = (const lean_object*)&lp_mathlib_term___u22d6___00__closed__3_value;
static const lean_ctor_object lp_mathlib_term___u22d6___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_term___u2a7f___00__closed__3_value),((lean_object*)&lp_mathlib_term___u22d6___00__closed__3_value),((lean_object*)&lp_mathlib_term___u2a7f___00__closed__8_value)}};
static const lean_object* lp_mathlib_term___u22d6___00__closed__4 = (const lean_object*)&lp_mathlib_term___u22d6___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term___u22d6___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_term___u22d6___00__closed__1_value),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)&lp_mathlib_term___u22d6___00__closed__4_value)}};
static const lean_object* lp_mathlib_term___u22d6___00__closed__5 = (const lean_object*)&lp_mathlib_term___u22d6___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_term___u22d6__ = (const lean_object*)&lp_mathlib_term___u22d6___00__closed__5_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u22d6____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "CovBy"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u22d6____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u22d6____1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u22d6____1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u22d6____1___closed__1;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u22d6____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u22d6____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(147, 209, 118, 95, 100, 140, 134, 45)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u22d6____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u22d6____1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u22d6____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u22d6____1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u22d6____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u22d6____1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u22d6____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u22d6____1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u22d6____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u22d6____1___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u22d6____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u22d6____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______unexpand__CovBy__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______unexpand__CovBy__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_decidableEqOfDecidableLE___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_decidableEqOfDecidableLE___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_decidableEqOfDecidableLE(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_decidableEqOfDecidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__12(void){
_start:
{
lean_object* v___x_27_; lean_object* v___x_28_; 
v___x_27_ = ((lean_object*)(lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__10));
v___x_28_ = l_Lean_mkAtom(v___x_27_);
return v___x_28_;
}
}
static lean_object* _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__13(void){
_start:
{
lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v___x_31_; 
v___x_29_ = lean_obj_once(&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__12, &lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__12_once, _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__12);
v___x_30_ = ((lean_object*)(lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__5));
v___x_31_ = lean_array_push(v___x_30_, v___x_29_);
return v___x_31_;
}
}
static lean_object* _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__15(void){
_start:
{
lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v___x_38_; 
v___x_36_ = ((lean_object*)(lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__14));
v___x_37_ = lean_obj_once(&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__13, &lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__13_once, _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__13);
v___x_38_ = lean_array_push(v___x_37_, v___x_36_);
return v___x_38_;
}
}
static lean_object* _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__16(void){
_start:
{
lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; 
v___x_39_ = lean_obj_once(&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__15, &lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__15_once, _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__15);
v___x_40_ = ((lean_object*)(lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__11));
v___x_41_ = lean_box(2);
v___x_42_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_42_, 0, v___x_41_);
lean_ctor_set(v___x_42_, 1, v___x_40_);
lean_ctor_set(v___x_42_, 2, v___x_39_);
return v___x_42_;
}
}
static lean_object* _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__17(void){
_start:
{
lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; 
v___x_43_ = lean_obj_once(&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__16, &lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__16_once, _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__16);
v___x_44_ = ((lean_object*)(lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__5));
v___x_45_ = lean_array_push(v___x_44_, v___x_43_);
return v___x_45_;
}
}
static lean_object* _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__19(void){
_start:
{
lean_object* v___x_47_; lean_object* v___x_48_; 
v___x_47_ = ((lean_object*)(lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__18));
v___x_48_ = l_Lean_mkAtom(v___x_47_);
return v___x_48_;
}
}
static lean_object* _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__20(void){
_start:
{
lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; 
v___x_49_ = lean_obj_once(&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__19, &lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__19_once, _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__19);
v___x_50_ = lean_obj_once(&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__17, &lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__17_once, _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__17);
v___x_51_ = lean_array_push(v___x_50_, v___x_49_);
return v___x_51_;
}
}
static lean_object* _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__24(void){
_start:
{
lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_59_ = ((lean_object*)(lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__23));
v___x_60_ = l_Lean_mkAtom(v___x_59_);
return v___x_60_;
}
}
static lean_object* _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__25(void){
_start:
{
lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_61_ = lean_obj_once(&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__24, &lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__24_once, _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__24);
v___x_62_ = ((lean_object*)(lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__5));
v___x_63_ = lean_array_push(v___x_62_, v___x_61_);
return v___x_63_;
}
}
static lean_object* _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__26(void){
_start:
{
lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; 
v___x_64_ = lean_obj_once(&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__25, &lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__25_once, _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__25);
v___x_65_ = ((lean_object*)(lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__22));
v___x_66_ = lean_box(2);
v___x_67_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_67_, 0, v___x_66_);
lean_ctor_set(v___x_67_, 1, v___x_65_);
lean_ctor_set(v___x_67_, 2, v___x_64_);
return v___x_67_;
}
}
static lean_object* _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__27(void){
_start:
{
lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; 
v___x_68_ = lean_obj_once(&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__26, &lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__26_once, _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__26);
v___x_69_ = lean_obj_once(&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__20, &lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__20_once, _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__20);
v___x_70_ = lean_array_push(v___x_69_, v___x_68_);
return v___x_70_;
}
}
static lean_object* _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__28(void){
_start:
{
lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; 
v___x_71_ = lean_obj_once(&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__27, &lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__27_once, _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__27);
v___x_72_ = ((lean_object*)(lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__9));
v___x_73_ = lean_box(2);
v___x_74_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_74_, 0, v___x_73_);
lean_ctor_set(v___x_74_, 1, v___x_72_);
lean_ctor_set(v___x_74_, 2, v___x_71_);
return v___x_74_;
}
}
static lean_object* _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__29(void){
_start:
{
lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; 
v___x_75_ = lean_obj_once(&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__28, &lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__28_once, _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__28);
v___x_76_ = ((lean_object*)(lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__5));
v___x_77_ = lean_array_push(v___x_76_, v___x_75_);
return v___x_77_;
}
}
static lean_object* _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__30(void){
_start:
{
lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; 
v___x_78_ = lean_obj_once(&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__29, &lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__29_once, _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__29);
v___x_79_ = ((lean_object*)(lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__7));
v___x_80_ = lean_box(2);
v___x_81_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_81_, 0, v___x_80_);
lean_ctor_set(v___x_81_, 1, v___x_79_);
lean_ctor_set(v___x_81_, 2, v___x_78_);
return v___x_81_;
}
}
static lean_object* _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__31(void){
_start:
{
lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; 
v___x_82_ = lean_obj_once(&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__30, &lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__30_once, _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__30);
v___x_83_ = ((lean_object*)(lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__5));
v___x_84_ = lean_array_push(v___x_83_, v___x_82_);
return v___x_84_;
}
}
static lean_object* _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__32(void){
_start:
{
lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; 
v___x_85_ = lean_obj_once(&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__31, &lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__31_once, _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__31);
v___x_86_ = ((lean_object*)(lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__4));
v___x_87_ = lean_box(2);
v___x_88_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_88_, 0, v___x_87_);
lean_ctor_set(v___x_88_, 1, v___x_86_);
lean_ctor_set(v___x_88_, 2, v___x_85_);
return v___x_88_;
}
}
static lean_object* _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam(void){
_start:
{
lean_object* v___x_89_; 
v___x_89_ = lean_obj_once(&lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__32, &lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__32_once, _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__32);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTransLE(lean_object* v_00_u03b1_90_, lean_object* v_inst_91_){
_start:
{
lean_object* v___x_92_; 
v___x_92_ = lean_box(0);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTransLE___boxed(lean_object* v_00_u03b1_93_, lean_object* v_inst_94_){
_start:
{
lean_object* v_res_95_; 
v_res_95_ = lp_mathlib_instTransLE(v_00_u03b1_93_, v_inst_94_);
lean_dec_ref(v_inst_94_);
return v_res_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTransLT(lean_object* v_00_u03b1_96_, lean_object* v_inst_97_){
_start:
{
lean_object* v___x_98_; 
v___x_98_ = lean_box(0);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTransLT___boxed(lean_object* v_00_u03b1_99_, lean_object* v_inst_100_){
_start:
{
lean_object* v_res_101_; 
v_res_101_ = lp_mathlib_instTransLT(v_00_u03b1_99_, v_inst_100_);
lean_dec_ref(v_inst_100_);
return v_res_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTransLTLE(lean_object* v_00_u03b1_102_, lean_object* v_inst_103_){
_start:
{
lean_object* v___x_104_; 
v___x_104_ = lean_box(0);
return v___x_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTransLTLE___boxed(lean_object* v_00_u03b1_105_, lean_object* v_inst_106_){
_start:
{
lean_object* v_res_107_; 
v_res_107_ = lp_mathlib_instTransLTLE(v_00_u03b1_105_, v_inst_106_);
lean_dec_ref(v_inst_106_);
return v_res_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTransLELT(lean_object* v_00_u03b1_108_, lean_object* v_inst_109_){
_start:
{
lean_object* v___x_110_; 
v___x_110_ = lean_box(0);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTransLELT___boxed(lean_object* v_00_u03b1_111_, lean_object* v_inst_112_){
_start:
{
lean_object* v_res_113_; 
v_res_113_ = lp_mathlib_instTransLELT(v_00_u03b1_111_, v_inst_112_);
lean_dec_ref(v_inst_112_);
return v_res_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTransGE(lean_object* v_00_u03b1_114_, lean_object* v_inst_115_){
_start:
{
lean_object* v___x_116_; 
v___x_116_ = lean_box(0);
return v___x_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTransGE___boxed(lean_object* v_00_u03b1_117_, lean_object* v_inst_118_){
_start:
{
lean_object* v_res_119_; 
v_res_119_ = lp_mathlib_instTransGE(v_00_u03b1_117_, v_inst_118_);
lean_dec_ref(v_inst_118_);
return v_res_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTransGT(lean_object* v_00_u03b1_120_, lean_object* v_inst_121_){
_start:
{
lean_object* v___x_122_; 
v___x_122_ = lean_box(0);
return v___x_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTransGT___boxed(lean_object* v_00_u03b1_123_, lean_object* v_inst_124_){
_start:
{
lean_object* v_res_125_; 
v_res_125_ = lp_mathlib_instTransGT(v_00_u03b1_123_, v_inst_124_);
lean_dec_ref(v_inst_124_);
return v_res_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTransGTGE(lean_object* v_00_u03b1_126_, lean_object* v_inst_127_){
_start:
{
lean_object* v___x_128_; 
v___x_128_ = lean_box(0);
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTransGTGE___boxed(lean_object* v_00_u03b1_129_, lean_object* v_inst_130_){
_start:
{
lean_object* v_res_131_; 
v_res_131_ = lp_mathlib_instTransGTGE(v_00_u03b1_129_, v_inst_130_);
lean_dec_ref(v_inst_130_);
return v_res_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTransGEGT(lean_object* v_00_u03b1_132_, lean_object* v_inst_133_){
_start:
{
lean_object* v___x_134_; 
v___x_134_ = lean_box(0);
return v___x_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTransGEGT___boxed(lean_object* v_00_u03b1_135_, lean_object* v_inst_136_){
_start:
{
lean_object* v_res_137_; 
v_res_137_ = lp_mathlib_instTransGEGT(v_00_u03b1_135_, v_inst_136_);
lean_dec_ref(v_inst_136_);
return v_res_137_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_decidableLTOfDecidableLE___redArg(lean_object* v_inst_138_, lean_object* v_x_139_, lean_object* v_x_140_){
_start:
{
lean_object* v___x_141_; lean_object* v___x_142_; uint8_t v___x_143_; 
lean_inc_ref(v_inst_138_);
lean_inc(v_x_139_);
lean_inc(v_x_140_);
v___x_141_ = lean_apply_2(v_inst_138_, v_x_140_, v_x_139_);
v___x_142_ = lean_apply_2(v_inst_138_, v_x_139_, v_x_140_);
v___x_143_ = lean_unbox(v___x_142_);
if (v___x_143_ == 0)
{
uint8_t v___x_144_; 
v___x_144_ = lean_unbox(v___x_142_);
return v___x_144_;
}
else
{
uint8_t v___x_145_; 
v___x_145_ = lean_unbox(v___x_141_);
if (v___x_145_ == 0)
{
uint8_t v___x_146_; 
v___x_146_ = lean_unbox(v___x_142_);
return v___x_146_;
}
else
{
uint8_t v___x_147_; 
v___x_147_ = 0;
return v___x_147_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_decidableLTOfDecidableLE___redArg___boxed(lean_object* v_inst_148_, lean_object* v_x_149_, lean_object* v_x_150_){
_start:
{
uint8_t v_res_151_; lean_object* v_r_152_; 
v_res_151_ = lp_mathlib_decidableLTOfDecidableLE___redArg(v_inst_148_, v_x_149_, v_x_150_);
v_r_152_ = lean_box(v_res_151_);
return v_r_152_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_decidableLTOfDecidableLE(lean_object* v_00_u03b1_153_, lean_object* v_inst_154_, lean_object* v_inst_155_, lean_object* v_x_156_, lean_object* v_x_157_){
_start:
{
uint8_t v___x_158_; 
v___x_158_ = lp_mathlib_decidableLTOfDecidableLE___redArg(v_inst_155_, v_x_156_, v_x_157_);
return v___x_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_decidableLTOfDecidableLE___boxed(lean_object* v_00_u03b1_159_, lean_object* v_inst_160_, lean_object* v_inst_161_, lean_object* v_x_162_, lean_object* v_x_163_){
_start:
{
uint8_t v_res_164_; lean_object* v_r_165_; 
v_res_164_ = lp_mathlib_decidableLTOfDecidableLE(v_00_u03b1_159_, v_inst_160_, v_inst_161_, v_x_162_, v_x_163_);
lean_dec_ref(v_inst_160_);
v_r_165_ = lean_box(v_res_164_);
return v_r_165_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__4(void){
_start:
{
lean_object* v___x_198_; lean_object* v___x_199_; 
v___x_198_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__3));
v___x_199_ = l_String_toRawSubstring_x27(v___x_198_);
return v___x_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1(lean_object* v_x_208_, lean_object* v_a_209_, lean_object* v_a_210_){
_start:
{
lean_object* v___x_211_; uint8_t v___x_212_; 
v___x_211_ = ((lean_object*)(lp_mathlib_term___u2a7f___00__closed__1));
lean_inc(v_x_208_);
v___x_212_ = l_Lean_Syntax_isOfKind(v_x_208_, v___x_211_);
if (v___x_212_ == 0)
{
lean_object* v___x_213_; lean_object* v___x_214_; 
lean_dec(v_x_208_);
v___x_213_ = lean_box(1);
v___x_214_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_214_, 0, v___x_213_);
lean_ctor_set(v___x_214_, 1, v_a_210_);
return v___x_214_;
}
else
{
lean_object* v_quotContext_215_; lean_object* v_currMacroScope_216_; lean_object* v_ref_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; uint8_t v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; 
v_quotContext_215_ = lean_ctor_get(v_a_209_, 1);
v_currMacroScope_216_ = lean_ctor_get(v_a_209_, 2);
v_ref_217_ = lean_ctor_get(v_a_209_, 5);
v___x_218_ = lean_unsigned_to_nat(0u);
v___x_219_ = l_Lean_Syntax_getArg(v_x_208_, v___x_218_);
v___x_220_ = lean_unsigned_to_nat(2u);
v___x_221_ = l_Lean_Syntax_getArg(v_x_208_, v___x_220_);
lean_dec(v_x_208_);
v___x_222_ = 0;
v___x_223_ = l_Lean_SourceInfo_fromRef(v_ref_217_, v___x_222_);
v___x_224_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__2));
v___x_225_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__4, &lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__4_once, _init_lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__4);
v___x_226_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__5));
lean_inc(v_currMacroScope_216_);
lean_inc(v_quotContext_215_);
v___x_227_ = l_Lean_addMacroScope(v_quotContext_215_, v___x_226_, v_currMacroScope_216_);
v___x_228_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__7));
lean_inc_n(v___x_223_, 2);
v___x_229_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_229_, 0, v___x_223_);
lean_ctor_set(v___x_229_, 1, v___x_225_);
lean_ctor_set(v___x_229_, 2, v___x_227_);
lean_ctor_set(v___x_229_, 3, v___x_228_);
v___x_230_ = ((lean_object*)(lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__9));
v___x_231_ = l_Lean_Syntax_node2(v___x_223_, v___x_230_, v___x_219_, v___x_221_);
v___x_232_ = l_Lean_Syntax_node2(v___x_223_, v___x_224_, v___x_229_, v___x_231_);
v___x_233_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_233_, 0, v___x_232_);
lean_ctor_set(v___x_233_, 1, v_a_210_);
return v___x_233_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___boxed(lean_object* v_x_234_, lean_object* v_a_235_, lean_object* v_a_236_){
_start:
{
lean_object* v_res_237_; 
v_res_237_ = lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1(v_x_234_, v_a_235_, v_a_236_);
lean_dec_ref(v_a_235_);
return v_res_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______unexpand__WCovBy__1(lean_object* v_x_241_, lean_object* v_a_242_, lean_object* v_a_243_){
_start:
{
lean_object* v___x_244_; uint8_t v___x_245_; 
v___x_244_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__2));
lean_inc(v_x_241_);
v___x_245_ = l_Lean_Syntax_isOfKind(v_x_241_, v___x_244_);
if (v___x_245_ == 0)
{
lean_object* v___x_246_; lean_object* v___x_247_; 
lean_dec(v_x_241_);
v___x_246_ = lean_box(0);
v___x_247_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_247_, 0, v___x_246_);
lean_ctor_set(v___x_247_, 1, v_a_243_);
return v___x_247_;
}
else
{
lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; uint8_t v___x_251_; 
v___x_248_ = lean_unsigned_to_nat(0u);
v___x_249_ = l_Lean_Syntax_getArg(v_x_241_, v___x_248_);
v___x_250_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______unexpand__WCovBy__1___closed__1));
lean_inc(v___x_249_);
v___x_251_ = l_Lean_Syntax_isOfKind(v___x_249_, v___x_250_);
if (v___x_251_ == 0)
{
lean_object* v___x_252_; lean_object* v___x_253_; 
lean_dec(v___x_249_);
lean_dec(v_x_241_);
v___x_252_ = lean_box(0);
v___x_253_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_253_, 0, v___x_252_);
lean_ctor_set(v___x_253_, 1, v_a_243_);
return v___x_253_;
}
else
{
lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; uint8_t v___x_257_; 
v___x_254_ = lean_unsigned_to_nat(1u);
v___x_255_ = l_Lean_Syntax_getArg(v_x_241_, v___x_254_);
lean_dec(v_x_241_);
v___x_256_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_255_);
v___x_257_ = l_Lean_Syntax_matchesNull(v___x_255_, v___x_256_);
if (v___x_257_ == 0)
{
lean_object* v___x_258_; lean_object* v___x_259_; 
lean_dec(v___x_255_);
lean_dec(v___x_249_);
v___x_258_ = lean_box(0);
v___x_259_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_259_, 0, v___x_258_);
lean_ctor_set(v___x_259_, 1, v_a_243_);
return v___x_259_;
}
else
{
lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v_ref_262_; uint8_t v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; 
v___x_260_ = l_Lean_Syntax_getArg(v___x_255_, v___x_248_);
v___x_261_ = l_Lean_Syntax_getArg(v___x_255_, v___x_254_);
lean_dec(v___x_255_);
v_ref_262_ = l_Lean_replaceRef(v___x_249_, v_a_242_);
lean_dec(v___x_249_);
v___x_263_ = 0;
v___x_264_ = l_Lean_SourceInfo_fromRef(v_ref_262_, v___x_263_);
lean_dec(v_ref_262_);
v___x_265_ = ((lean_object*)(lp_mathlib_term___u2a7f___00__closed__1));
v___x_266_ = ((lean_object*)(lp_mathlib_term___u2a7f___00__closed__4));
lean_inc(v___x_264_);
v___x_267_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_267_, 0, v___x_264_);
lean_ctor_set(v___x_267_, 1, v___x_266_);
v___x_268_ = l_Lean_Syntax_node3(v___x_264_, v___x_265_, v___x_260_, v___x_267_, v___x_261_);
v___x_269_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_269_, 0, v___x_268_);
lean_ctor_set(v___x_269_, 1, v_a_243_);
return v___x_269_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______unexpand__WCovBy__1___boxed(lean_object* v_x_270_, lean_object* v_a_271_, lean_object* v_a_272_){
_start:
{
lean_object* v_res_273_; 
v_res_273_ = lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______unexpand__WCovBy__1(v_x_270_, v_a_271_, v_a_272_);
lean_dec(v_a_271_);
return v_res_273_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u22d6____1___closed__1(void){
_start:
{
lean_object* v___x_290_; lean_object* v___x_291_; 
v___x_290_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u22d6____1___closed__0));
v___x_291_ = l_String_toRawSubstring_x27(v___x_290_);
return v___x_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u22d6____1(lean_object* v_x_300_, lean_object* v_a_301_, lean_object* v_a_302_){
_start:
{
lean_object* v___x_303_; uint8_t v___x_304_; 
v___x_303_ = ((lean_object*)(lp_mathlib_term___u22d6___00__closed__1));
lean_inc(v_x_300_);
v___x_304_ = l_Lean_Syntax_isOfKind(v_x_300_, v___x_303_);
if (v___x_304_ == 0)
{
lean_object* v___x_305_; lean_object* v___x_306_; 
lean_dec(v_x_300_);
v___x_305_ = lean_box(1);
v___x_306_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_306_, 0, v___x_305_);
lean_ctor_set(v___x_306_, 1, v_a_302_);
return v___x_306_;
}
else
{
lean_object* v_quotContext_307_; lean_object* v_currMacroScope_308_; lean_object* v_ref_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; uint8_t v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; 
v_quotContext_307_ = lean_ctor_get(v_a_301_, 1);
v_currMacroScope_308_ = lean_ctor_get(v_a_301_, 2);
v_ref_309_ = lean_ctor_get(v_a_301_, 5);
v___x_310_ = lean_unsigned_to_nat(0u);
v___x_311_ = l_Lean_Syntax_getArg(v_x_300_, v___x_310_);
v___x_312_ = lean_unsigned_to_nat(2u);
v___x_313_ = l_Lean_Syntax_getArg(v_x_300_, v___x_312_);
lean_dec(v_x_300_);
v___x_314_ = 0;
v___x_315_ = l_Lean_SourceInfo_fromRef(v_ref_309_, v___x_314_);
v___x_316_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__2));
v___x_317_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u22d6____1___closed__1, &lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u22d6____1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u22d6____1___closed__1);
v___x_318_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u22d6____1___closed__2));
lean_inc(v_currMacroScope_308_);
lean_inc(v_quotContext_307_);
v___x_319_ = l_Lean_addMacroScope(v_quotContext_307_, v___x_318_, v_currMacroScope_308_);
v___x_320_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u22d6____1___closed__4));
lean_inc_n(v___x_315_, 2);
v___x_321_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_321_, 0, v___x_315_);
lean_ctor_set(v___x_321_, 1, v___x_317_);
lean_ctor_set(v___x_321_, 2, v___x_319_);
lean_ctor_set(v___x_321_, 3, v___x_320_);
v___x_322_ = ((lean_object*)(lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam___closed__9));
v___x_323_ = l_Lean_Syntax_node2(v___x_315_, v___x_322_, v___x_311_, v___x_313_);
v___x_324_ = l_Lean_Syntax_node2(v___x_315_, v___x_316_, v___x_321_, v___x_323_);
v___x_325_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_325_, 0, v___x_324_);
lean_ctor_set(v___x_325_, 1, v_a_302_);
return v___x_325_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u22d6____1___boxed(lean_object* v_x_326_, lean_object* v_a_327_, lean_object* v_a_328_){
_start:
{
lean_object* v_res_329_; 
v_res_329_ = lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u22d6____1(v_x_326_, v_a_327_, v_a_328_);
lean_dec_ref(v_a_327_);
return v_res_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______unexpand__CovBy__1(lean_object* v_x_330_, lean_object* v_a_331_, lean_object* v_a_332_){
_start:
{
lean_object* v___x_333_; uint8_t v___x_334_; 
v___x_333_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______macroRules__term___u2a7f____1___closed__2));
lean_inc(v_x_330_);
v___x_334_ = l_Lean_Syntax_isOfKind(v_x_330_, v___x_333_);
if (v___x_334_ == 0)
{
lean_object* v___x_335_; lean_object* v___x_336_; 
lean_dec(v_x_330_);
v___x_335_ = lean_box(0);
v___x_336_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_336_, 0, v___x_335_);
lean_ctor_set(v___x_336_, 1, v_a_332_);
return v___x_336_;
}
else
{
lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; uint8_t v___x_340_; 
v___x_337_ = lean_unsigned_to_nat(0u);
v___x_338_ = l_Lean_Syntax_getArg(v_x_330_, v___x_337_);
v___x_339_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______unexpand__WCovBy__1___closed__1));
lean_inc(v___x_338_);
v___x_340_ = l_Lean_Syntax_isOfKind(v___x_338_, v___x_339_);
if (v___x_340_ == 0)
{
lean_object* v___x_341_; lean_object* v___x_342_; 
lean_dec(v___x_338_);
lean_dec(v_x_330_);
v___x_341_ = lean_box(0);
v___x_342_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_342_, 0, v___x_341_);
lean_ctor_set(v___x_342_, 1, v_a_332_);
return v___x_342_;
}
else
{
lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; uint8_t v___x_346_; 
v___x_343_ = lean_unsigned_to_nat(1u);
v___x_344_ = l_Lean_Syntax_getArg(v_x_330_, v___x_343_);
lean_dec(v_x_330_);
v___x_345_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_344_);
v___x_346_ = l_Lean_Syntax_matchesNull(v___x_344_, v___x_345_);
if (v___x_346_ == 0)
{
lean_object* v___x_347_; lean_object* v___x_348_; 
lean_dec(v___x_344_);
lean_dec(v___x_338_);
v___x_347_ = lean_box(0);
v___x_348_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_348_, 0, v___x_347_);
lean_ctor_set(v___x_348_, 1, v_a_332_);
return v___x_348_;
}
else
{
lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v_ref_351_; uint8_t v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; 
v___x_349_ = l_Lean_Syntax_getArg(v___x_344_, v___x_337_);
v___x_350_ = l_Lean_Syntax_getArg(v___x_344_, v___x_343_);
lean_dec(v___x_344_);
v_ref_351_ = l_Lean_replaceRef(v___x_338_, v_a_331_);
lean_dec(v___x_338_);
v___x_352_ = 0;
v___x_353_ = l_Lean_SourceInfo_fromRef(v_ref_351_, v___x_352_);
lean_dec(v_ref_351_);
v___x_354_ = ((lean_object*)(lp_mathlib_term___u22d6___00__closed__1));
v___x_355_ = ((lean_object*)(lp_mathlib_term___u22d6___00__closed__2));
lean_inc(v___x_353_);
v___x_356_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_356_, 0, v___x_353_);
lean_ctor_set(v___x_356_, 1, v___x_355_);
v___x_357_ = l_Lean_Syntax_node3(v___x_353_, v___x_354_, v___x_349_, v___x_356_, v___x_350_);
v___x_358_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_358_, 0, v___x_357_);
lean_ctor_set(v___x_358_, 1, v_a_332_);
return v___x_358_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______unexpand__CovBy__1___boxed(lean_object* v_x_359_, lean_object* v_a_360_, lean_object* v_a_361_){
_start:
{
lean_object* v_res_362_; 
v_res_362_ = lp_mathlib___aux__Mathlib__Order__Defs__PartialOrder______unexpand__CovBy__1(v_x_359_, v_a_360_, v_a_361_);
lean_dec(v_a_360_);
return v_res_362_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_decidableEqOfDecidableLE___redArg(lean_object* v_inst_363_, lean_object* v_x_364_, lean_object* v_x_365_){
_start:
{
lean_object* v___x_366_; uint8_t v___x_367_; 
lean_inc_ref(v_inst_363_);
lean_inc(v_x_365_);
lean_inc(v_x_364_);
v___x_366_ = lean_apply_2(v_inst_363_, v_x_364_, v_x_365_);
v___x_367_ = lean_unbox(v___x_366_);
if (v___x_367_ == 0)
{
uint8_t v___x_368_; 
lean_dec(v_x_365_);
lean_dec(v_x_364_);
lean_dec_ref(v_inst_363_);
v___x_368_ = lean_unbox(v___x_366_);
return v___x_368_;
}
else
{
lean_object* v___x_369_; uint8_t v___x_370_; 
v___x_369_ = lean_apply_2(v_inst_363_, v_x_365_, v_x_364_);
v___x_370_ = lean_unbox(v___x_369_);
return v___x_370_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_decidableEqOfDecidableLE___redArg___boxed(lean_object* v_inst_371_, lean_object* v_x_372_, lean_object* v_x_373_){
_start:
{
uint8_t v_res_374_; lean_object* v_r_375_; 
v_res_374_ = lp_mathlib_decidableEqOfDecidableLE___redArg(v_inst_371_, v_x_372_, v_x_373_);
v_r_375_ = lean_box(v_res_374_);
return v_r_375_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_decidableEqOfDecidableLE(lean_object* v_00_u03b1_376_, lean_object* v_inst_377_, lean_object* v_inst_378_, lean_object* v_x_379_, lean_object* v_x_380_){
_start:
{
uint8_t v___x_381_; 
v___x_381_ = lp_mathlib_decidableEqOfDecidableLE___redArg(v_inst_378_, v_x_379_, v_x_380_);
return v___x_381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_decidableEqOfDecidableLE___boxed(lean_object* v_00_u03b1_382_, lean_object* v_inst_383_, lean_object* v_inst_384_, lean_object* v_x_385_, lean_object* v_x_386_){
_start:
{
uint8_t v_res_387_; lean_object* v_r_388_; 
v_res_387_ = lp_mathlib_decidableEqOfDecidableLE(v_00_u03b1_382_, v_inst_383_, v_inst_384_, v_x_385_, v_x_386_);
lean_dec_ref(v_inst_383_);
v_r_388_ = lean_box(v_res_387_);
return v_r_388_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Alias(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Trans(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ToDual(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Defs_PartialOrder(uint8_t builtin) {
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
res = runtime_initialize_batteries_Batteries_Tactic_Trans(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ToDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Defs_PartialOrder(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam = _init_lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam();
lean_mark_persistent(lp_mathlib_Preorder_lt__iff__le__not__ge___autoParam);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Alias(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Trans(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToDual(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Defs_PartialOrder(uint8_t builtin) {
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
res = initialize_batteries_Batteries_Tactic_Trans(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Defs_PartialOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Defs_PartialOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Defs_PartialOrder(builtin);
}
#ifdef __cplusplus
}
#endif
