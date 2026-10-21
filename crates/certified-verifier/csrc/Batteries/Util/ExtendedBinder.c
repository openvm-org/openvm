// Lean compiler output
// Module: Batteries.Util.ExtendedBinder
// Imports: public import Init public meta import Init
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
extern lean_object* l_Lean_binderIdent;
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Syntax_getNumArgs(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* l_Array_extract___redArg(lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_ExtendedBinder_extBinder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "extBinder"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinder___closed__0 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__0_value;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder_extBinder___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinder___closed__1 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__1_value;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder_extBinder___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "ExtendedBinder"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinder___closed__2 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_extBinder___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_extBinder___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__3_value_aux_0),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__2_value),LEAN_SCALAR_PTR_LITERAL(56, 78, 248, 154, 49, 0, 91, 17)}};
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_extBinder___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__3_value_aux_1),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__0_value),LEAN_SCALAR_PTR_LITERAL(140, 4, 199, 115, 152, 1, 62, 3)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinder___closed__3 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__3_value;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder_extBinder___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinder___closed__4 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__4_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_extBinder___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinder___closed__5 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__5_value;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder_extBinder___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinder___closed__6 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_extBinder___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__6_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinder___closed__7 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__7_value;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder_extBinder___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "orelse"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinder___closed__8 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__8_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_extBinder___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__8_value),LEAN_SCALAR_PTR_LITERAL(78, 76, 4, 51, 251, 212, 116, 5)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinder___closed__9 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__9_value;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder_extBinder___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinder___closed__10 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__10_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_extBinder___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__10_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinder___closed__11 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__11_value;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder_extBinder___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " : "};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinder___closed__12 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__12_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_extBinder___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__12_value)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinder___closed__13 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__13_value;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder_extBinder___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinder___closed__14 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__14_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_extBinder___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__14_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinder___closed__15 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__15_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_extBinder___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__15_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinder___closed__16 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__16_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_extBinder___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__5_value),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__13_value),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__16_value)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinder___closed__17 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__17_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_extBinder___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__11_value),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__17_value)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinder___closed__18 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__18_value;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder_extBinder___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "binderPred"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinder___closed__19 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__19_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_extBinder___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__19_value),LEAN_SCALAR_PTR_LITERAL(218, 134, 142, 164, 134, 201, 62, 191)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinder___closed__20 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__20_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_extBinder___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__20_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinder___closed__21 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__21_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_extBinder___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__9_value),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__18_value),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__21_value)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinder___closed__22 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__22_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_extBinder___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__7_value),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__22_value)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinder___closed__23 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__23_value;
static lean_once_cell_t lp_batteries_Batteries_ExtendedBinder_extBinder___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_ExtendedBinder_extBinder___closed__24;
static lean_once_cell_t lp_batteries_Batteries_ExtendedBinder_extBinder___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_ExtendedBinder_extBinder___closed__25;
LEAN_EXPORT lean_object* lp_batteries_Batteries_ExtendedBinder_extBinder;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "extBinderParenthesized"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__0 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__2_value),LEAN_SCALAR_PTR_LITERAL(56, 78, 248, 154, 49, 0, 91, 17)}};
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__0_value),LEAN_SCALAR_PTR_LITERAL(207, 166, 79, 161, 194, 16, 7, 156)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__1 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__1_value;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " ("};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__2 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__2_value)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__3 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__3_value;
static lean_once_cell_t lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__4;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__5 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__5_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__5_value)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__6 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__6_value;
static lean_once_cell_t lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__7;
static lean_once_cell_t lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__8;
LEAN_EXPORT lean_object* lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "extBinderCollection"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__0 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__2_value),LEAN_SCALAR_PTR_LITERAL(56, 78, 248, 154, 49, 0, 91, 17)}};
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__0_value),LEAN_SCALAR_PTR_LITERAL(144, 58, 22, 199, 215, 82, 42, 232)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__1 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__1_value;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "many"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__2 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__2_value),LEAN_SCALAR_PTR_LITERAL(41, 35, 40, 86, 189, 97, 244, 31)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__3 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__3_value;
static lean_once_cell_t lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__4;
static lean_once_cell_t lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__5;
LEAN_EXPORT lean_object* lp_batteries_Batteries_ExtendedBinder_extBinderCollection;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder_extBinders___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "extBinders"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinders___closed__0 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinders___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_extBinders___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_extBinders___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinders___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__2_value),LEAN_SCALAR_PTR_LITERAL(56, 78, 248, 154, 49, 0, 91, 17)}};
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_extBinders___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinders___closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinders___closed__0_value),LEAN_SCALAR_PTR_LITERAL(142, 202, 111, 171, 129, 134, 17, 161)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinders___closed__1 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinders___closed__1_value;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder_extBinders___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinders___closed__2 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinders___closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_extBinders___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinders___closed__2_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinders___closed__3 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinders___closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_extBinders___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinders___closed__3_value)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_extBinders___closed__4 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinders___closed__4_value;
static lean_once_cell_t lp_batteries_Batteries_ExtendedBinder_extBinders___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_ExtendedBinder_extBinders___closed__5;
static lean_once_cell_t lp_batteries_Batteries_ExtendedBinder_extBinders___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_ExtendedBinder_extBinders___closed__6;
static lean_once_cell_t lp_batteries_Batteries_ExtendedBinder_extBinders___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_ExtendedBinder_extBinders___closed__7;
LEAN_EXPORT lean_object* lp_batteries_Batteries_ExtendedBinder_extBinders;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 9, .m_data = "term∃ᵉ_,_"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__0 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__2_value),LEAN_SCALAR_PTR_LITERAL(56, 78, 248, 154, 49, 0, 91, 17)}};
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(224, 183, 129, 16, 236, 95, 122, 189)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__1 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__1_value;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 2, .m_data = "∃ᵉ"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__2 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__2_value)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__3 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__3_value;
static lean_once_cell_t lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__4;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__5 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__5_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__5_value)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__6 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__6_value;
static lean_once_cell_t lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__7;
static lean_once_cell_t lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__8;
static lean_once_cell_t lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__9;
LEAN_EXPORT lean_object* lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c__;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1_spec__1___closed__0 = (const lean_object*)&lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1_spec__1___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__0 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__1 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__1_value;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__2 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__2_value;
static lean_once_cell_t lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__3;
LEAN_EXPORT lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__0 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__0_value;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "binderIdent"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__1 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__1_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__2_value_aux_0),((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__1_value),LEAN_SCALAR_PTR_LITERAL(37, 194, 68, 106, 254, 181, 31, 191)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__2 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__2_value;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 9, .m_data = "term∃__,_"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__3 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__3_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__4_value_aux_0),((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__3_value),LEAN_SCALAR_PTR_LITERAL(25, 165, 221, 98, 134, 231, 221, 237)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__4 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__4_value;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "∃"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__5 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__5_value;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 8, .m_data = "term∃_,_"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__6 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__6_value),LEAN_SCALAR_PTR_LITERAL(224, 105, 219, 112, 166, 139, 167, 161)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__7 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__7_value;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "explicitBinders"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__8 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__8_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__9_value_aux_0),((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__8_value),LEAN_SCALAR_PTR_LITERAL(167, 149, 127, 13, 202, 239, 226, 94)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__9 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__9_value;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "unbracketedExplicitBinders"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__10 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__10_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__11_value_aux_0),((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__10_value),LEAN_SCALAR_PTR_LITERAL(187, 220, 119, 82, 242, 112, 119, 200)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__11 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__11_value;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__12 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__12_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 9, .m_data = "term∀ᵉ_,_"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__0 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__1_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__2_value),LEAN_SCALAR_PTR_LITERAL(56, 78, 248, 154, 49, 0, 91, 17)}};
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__1_value_aux_1),((lean_object*)&lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(192, 195, 227, 168, 35, 160, 185, 75)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__1 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__1_value;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 2, .m_data = "∀ᵉ"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__2 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__2_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__2_value)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__3 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__3_value;
static lean_once_cell_t lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__4;
static lean_once_cell_t lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__5;
static lean_once_cell_t lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__6;
static lean_once_cell_t lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__7;
LEAN_EXPORT lean_object* lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c__;
LEAN_EXPORT lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 9, .m_data = "term∀__,_"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__0 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__0_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__1_value_aux_0),((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(124, 16, 227, 203, 159, 8, 82, 19)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__1 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__1_value;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "∀"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__2 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__2_value;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__3 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__3_value;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__4 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__4_value;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hole"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__5 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__5_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__6_value_aux_0),((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__6_value_aux_1),((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__4_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__6_value_aux_2),((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__5_value),LEAN_SCALAR_PTR_LITERAL(135, 134, 219, 115, 97, 130, 74, 55)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__6 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__6_value;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__7 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__7_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__7_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__8 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__8_value;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "forall"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__9 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__9_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__10_value_aux_0),((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__10_value_aux_1),((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__4_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__10_value_aux_2),((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__9_value),LEAN_SCALAR_PTR_LITERAL(195, 142, 115, 15, 55, 103, 31, 115)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__10 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__10_value;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "typeSpec"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__11 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__11_value;
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__12_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__12_value_aux_0),((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__12_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__12_value_aux_1),((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__4_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__12_value_aux_2),((lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__11_value),LEAN_SCALAR_PTR_LITERAL(77, 126, 241, 117, 174, 189, 108, 62)}};
static const lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__12 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__12_value;
static const lean_string_object lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__13 = (const lean_object*)&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__13_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_batteries_Batteries_ExtendedBinder_extBinder___closed__24(void){
_start:
{
lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; lean_object* v___x_52_; 
v___x_49_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinder___closed__23));
v___x_50_ = l_Lean_binderIdent;
v___x_51_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinder___closed__5));
v___x_52_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_52_, 0, v___x_51_);
lean_ctor_set(v___x_52_, 1, v___x_50_);
lean_ctor_set(v___x_52_, 2, v___x_49_);
return v___x_52_;
}
}
static lean_object* _init_lp_batteries_Batteries_ExtendedBinder_extBinder___closed__25(void){
_start:
{
lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_53_ = lean_obj_once(&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__24, &lp_batteries_Batteries_ExtendedBinder_extBinder___closed__24_once, _init_lp_batteries_Batteries_ExtendedBinder_extBinder___closed__24);
v___x_54_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinder___closed__3));
v___x_55_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinder___closed__0));
v___x_56_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_56_, 0, v___x_55_);
lean_ctor_set(v___x_56_, 1, v___x_54_);
lean_ctor_set(v___x_56_, 2, v___x_53_);
return v___x_56_;
}
}
static lean_object* _init_lp_batteries_Batteries_ExtendedBinder_extBinder(void){
_start:
{
lean_object* v___x_57_; 
v___x_57_ = lean_obj_once(&lp_batteries_Batteries_ExtendedBinder_extBinder___closed__25, &lp_batteries_Batteries_ExtendedBinder_extBinder___closed__25_once, _init_lp_batteries_Batteries_ExtendedBinder_extBinder___closed__25);
return v___x_57_;
}
}
static lean_object* _init_lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__4(void){
_start:
{
lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; 
v___x_66_ = lp_batteries_Batteries_ExtendedBinder_extBinder;
v___x_67_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__3));
v___x_68_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinder___closed__5));
v___x_69_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_69_, 0, v___x_68_);
lean_ctor_set(v___x_69_, 1, v___x_67_);
lean_ctor_set(v___x_69_, 2, v___x_66_);
return v___x_69_;
}
}
static lean_object* _init_lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__7(void){
_start:
{
lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; 
v___x_73_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__6));
v___x_74_ = lean_obj_once(&lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__4, &lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__4_once, _init_lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__4);
v___x_75_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinder___closed__5));
v___x_76_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_76_, 0, v___x_75_);
lean_ctor_set(v___x_76_, 1, v___x_74_);
lean_ctor_set(v___x_76_, 2, v___x_73_);
return v___x_76_;
}
}
static lean_object* _init_lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__8(void){
_start:
{
lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; 
v___x_77_ = lean_obj_once(&lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__7, &lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__7_once, _init_lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__7);
v___x_78_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__1));
v___x_79_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__0));
v___x_80_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_80_, 0, v___x_79_);
lean_ctor_set(v___x_80_, 1, v___x_78_);
lean_ctor_set(v___x_80_, 2, v___x_77_);
return v___x_80_;
}
}
static lean_object* _init_lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized(void){
_start:
{
lean_object* v___x_81_; 
v___x_81_ = lean_obj_once(&lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__8, &lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__8_once, _init_lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__8);
return v___x_81_;
}
}
static lean_object* _init_lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__4(void){
_start:
{
lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; 
v___x_90_ = lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized;
v___x_91_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__3));
v___x_92_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_92_, 0, v___x_91_);
lean_ctor_set(v___x_92_, 1, v___x_90_);
return v___x_92_;
}
}
static lean_object* _init_lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__5(void){
_start:
{
lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; 
v___x_93_ = lean_obj_once(&lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__4, &lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__4_once, _init_lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__4);
v___x_94_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__1));
v___x_95_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__0));
v___x_96_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_96_, 0, v___x_95_);
lean_ctor_set(v___x_96_, 1, v___x_94_);
lean_ctor_set(v___x_96_, 2, v___x_93_);
return v___x_96_;
}
}
static lean_object* _init_lp_batteries_Batteries_ExtendedBinder_extBinderCollection(void){
_start:
{
lean_object* v___x_97_; 
v___x_97_ = lean_obj_once(&lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__5, &lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__5_once, _init_lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__5);
return v___x_97_;
}
}
static lean_object* _init_lp_batteries_Batteries_ExtendedBinder_extBinders___closed__5(void){
_start:
{
lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; 
v___x_108_ = lp_batteries_Batteries_ExtendedBinder_extBinder;
v___x_109_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinders___closed__4));
v___x_110_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinder___closed__5));
v___x_111_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_111_, 0, v___x_110_);
lean_ctor_set(v___x_111_, 1, v___x_109_);
lean_ctor_set(v___x_111_, 2, v___x_108_);
return v___x_111_;
}
}
static lean_object* _init_lp_batteries_Batteries_ExtendedBinder_extBinders___closed__6(void){
_start:
{
lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; 
v___x_112_ = lp_batteries_Batteries_ExtendedBinder_extBinderCollection;
v___x_113_ = lean_obj_once(&lp_batteries_Batteries_ExtendedBinder_extBinders___closed__5, &lp_batteries_Batteries_ExtendedBinder_extBinders___closed__5_once, _init_lp_batteries_Batteries_ExtendedBinder_extBinders___closed__5);
v___x_114_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinder___closed__9));
v___x_115_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_115_, 0, v___x_114_);
lean_ctor_set(v___x_115_, 1, v___x_113_);
lean_ctor_set(v___x_115_, 2, v___x_112_);
return v___x_115_;
}
}
static lean_object* _init_lp_batteries_Batteries_ExtendedBinder_extBinders___closed__7(void){
_start:
{
lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; 
v___x_116_ = lean_obj_once(&lp_batteries_Batteries_ExtendedBinder_extBinders___closed__6, &lp_batteries_Batteries_ExtendedBinder_extBinders___closed__6_once, _init_lp_batteries_Batteries_ExtendedBinder_extBinders___closed__6);
v___x_117_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinders___closed__1));
v___x_118_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinders___closed__0));
v___x_119_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_119_, 0, v___x_118_);
lean_ctor_set(v___x_119_, 1, v___x_117_);
lean_ctor_set(v___x_119_, 2, v___x_116_);
return v___x_119_;
}
}
static lean_object* _init_lp_batteries_Batteries_ExtendedBinder_extBinders(void){
_start:
{
lean_object* v___x_120_; 
v___x_120_ = lean_obj_once(&lp_batteries_Batteries_ExtendedBinder_extBinders___closed__7, &lp_batteries_Batteries_ExtendedBinder_extBinders___closed__7_once, _init_lp_batteries_Batteries_ExtendedBinder_extBinders___closed__7);
return v___x_120_;
}
}
static lean_object* _init_lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__4(void){
_start:
{
lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; 
v___x_129_ = lp_batteries_Batteries_ExtendedBinder_extBinders;
v___x_130_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__3));
v___x_131_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinder___closed__5));
v___x_132_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_132_, 0, v___x_131_);
lean_ctor_set(v___x_132_, 1, v___x_130_);
lean_ctor_set(v___x_132_, 2, v___x_129_);
return v___x_132_;
}
}
static lean_object* _init_lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__7(void){
_start:
{
lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; 
v___x_136_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__6));
v___x_137_ = lean_obj_once(&lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__4, &lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__4_once, _init_lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__4);
v___x_138_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinder___closed__5));
v___x_139_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_139_, 0, v___x_138_);
lean_ctor_set(v___x_139_, 1, v___x_137_);
lean_ctor_set(v___x_139_, 2, v___x_136_);
return v___x_139_;
}
}
static lean_object* _init_lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__8(void){
_start:
{
lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; 
v___x_140_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinder___closed__16));
v___x_141_ = lean_obj_once(&lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__7, &lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__7_once, _init_lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__7);
v___x_142_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinder___closed__5));
v___x_143_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_143_, 0, v___x_142_);
lean_ctor_set(v___x_143_, 1, v___x_141_);
lean_ctor_set(v___x_143_, 2, v___x_140_);
return v___x_143_;
}
}
static lean_object* _init_lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__9(void){
_start:
{
lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; 
v___x_144_ = lean_obj_once(&lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__8, &lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__8_once, _init_lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__8);
v___x_145_ = lean_unsigned_to_nat(1022u);
v___x_146_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__1));
v___x_147_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_147_, 0, v___x_146_);
lean_ctor_set(v___x_147_, 1, v___x_145_);
lean_ctor_set(v___x_147_, 2, v___x_144_);
return v___x_147_;
}
}
static lean_object* _init_lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c__(void){
_start:
{
lean_object* v___x_148_; 
v___x_148_ = lean_obj_once(&lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__9, &lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__9_once, _init_lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__9);
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1_spec__0(size_t v_sz_149_, size_t v_i_150_, lean_object* v_bs_151_){
_start:
{
uint8_t v___x_152_; 
v___x_152_ = lean_usize_dec_lt(v_i_150_, v_sz_149_);
if (v___x_152_ == 0)
{
lean_object* v___x_153_; 
v___x_153_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_153_, 0, v_bs_151_);
return v___x_153_;
}
else
{
lean_object* v___x_154_; lean_object* v_v_155_; uint8_t v___x_156_; 
v___x_154_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__1));
v_v_155_ = lean_array_uget_borrowed(v_bs_151_, v_i_150_);
lean_inc(v_v_155_);
v___x_156_ = l_Lean_Syntax_isOfKind(v_v_155_, v___x_154_);
if (v___x_156_ == 0)
{
lean_object* v___x_157_; 
lean_dec_ref(v_bs_151_);
v___x_157_ = lean_box(0);
return v___x_157_;
}
else
{
lean_object* v___x_158_; lean_object* v_ps_159_; lean_object* v___x_160_; uint8_t v___x_161_; 
v___x_158_ = lean_unsigned_to_nat(1u);
v_ps_159_ = l_Lean_Syntax_getArg(v_v_155_, v___x_158_);
v___x_160_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinder___closed__3));
lean_inc(v_ps_159_);
v___x_161_ = l_Lean_Syntax_isOfKind(v_ps_159_, v___x_160_);
if (v___x_161_ == 0)
{
lean_object* v___x_162_; 
lean_dec(v_ps_159_);
lean_dec_ref(v_bs_151_);
v___x_162_ = lean_box(0);
return v___x_162_;
}
else
{
lean_object* v___x_163_; lean_object* v_bs_x27_164_; size_t v___x_165_; size_t v___x_166_; lean_object* v___x_167_; 
v___x_163_ = lean_unsigned_to_nat(0u);
v_bs_x27_164_ = lean_array_uset(v_bs_151_, v_i_150_, v___x_163_);
v___x_165_ = ((size_t)1ULL);
v___x_166_ = lean_usize_add(v_i_150_, v___x_165_);
v___x_167_ = lean_array_uset(v_bs_x27_164_, v_i_150_, v_ps_159_);
v_i_150_ = v___x_166_;
v_bs_151_ = v___x_167_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1_spec__0___boxed(lean_object* v_sz_169_, lean_object* v_i_170_, lean_object* v_bs_171_){
_start:
{
size_t v_sz_boxed_172_; size_t v_i_boxed_173_; lean_object* v_res_174_; 
v_sz_boxed_172_ = lean_unbox_usize(v_sz_169_);
lean_dec(v_sz_169_);
v_i_boxed_173_ = lean_unbox_usize(v_i_170_);
lean_dec(v_i_170_);
v_res_174_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1_spec__0(v_sz_boxed_172_, v_i_boxed_173_, v_bs_171_);
return v_res_174_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1_spec__1(lean_object* v___x_176_, size_t v_sz_177_, size_t v_i_178_, lean_object* v_bs_179_){
_start:
{
uint8_t v___x_180_; 
v___x_180_ = lean_usize_dec_lt(v_i_178_, v_sz_177_);
if (v___x_180_ == 0)
{
lean_dec(v___x_176_);
return v_bs_179_;
}
else
{
lean_object* v_v_181_; lean_object* v___x_182_; lean_object* v_bs_x27_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; size_t v___x_190_; size_t v___x_191_; lean_object* v___x_192_; 
v_v_181_ = lean_array_uget(v_bs_179_, v_i_178_);
v___x_182_ = lean_unsigned_to_nat(0u);
v_bs_x27_183_ = lean_array_uset(v_bs_179_, v_i_178_, v___x_182_);
v___x_184_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__1));
v___x_185_ = ((lean_object*)(lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1_spec__1___closed__0));
lean_inc_n(v___x_176_, 3);
v___x_186_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_186_, 0, v___x_176_);
lean_ctor_set(v___x_186_, 1, v___x_185_);
v___x_187_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__5));
v___x_188_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_188_, 0, v___x_176_);
lean_ctor_set(v___x_188_, 1, v___x_187_);
v___x_189_ = l_Lean_Syntax_node3(v___x_176_, v___x_184_, v___x_186_, v_v_181_, v___x_188_);
v___x_190_ = ((size_t)1ULL);
v___x_191_ = lean_usize_add(v_i_178_, v___x_190_);
v___x_192_ = lean_array_uset(v_bs_x27_183_, v_i_178_, v___x_189_);
v_i_178_ = v___x_191_;
v_bs_179_ = v___x_192_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1_spec__1___boxed(lean_object* v___x_194_, lean_object* v_sz_195_, lean_object* v_i_196_, lean_object* v_bs_197_){
_start:
{
size_t v_sz_boxed_198_; size_t v_i_boxed_199_; lean_object* v_res_200_; 
v_sz_boxed_198_ = lean_unbox_usize(v_sz_195_);
lean_dec(v_sz_195_);
v_i_boxed_199_ = lean_unbox_usize(v_i_196_);
lean_dec(v_i_196_);
v_res_200_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1_spec__1(v___x_194_, v_sz_boxed_198_, v_i_boxed_199_, v_bs_197_);
return v_res_200_;
}
}
static lean_object* _init_lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__3(void){
_start:
{
lean_object* v___x_205_; 
v___x_205_ = l_Array_mkArray0(lean_box(0));
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1(lean_object* v_x_206_, lean_object* v_a_207_, lean_object* v_a_208_){
_start:
{
lean_object* v___y_210_; lean_object* v___x_213_; uint8_t v___x_214_; 
v___x_213_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__1));
lean_inc(v_x_206_);
v___x_214_ = l_Lean_Syntax_isOfKind(v_x_206_, v___x_213_);
if (v___x_214_ == 0)
{
lean_object* v___x_215_; lean_object* v___x_216_; 
lean_dec(v_x_206_);
v___x_215_ = lean_box(1);
v___x_216_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_216_, 0, v___x_215_);
lean_ctor_set(v___x_216_, 1, v_a_208_);
return v___x_216_;
}
else
{
lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; uint8_t v___x_220_; 
v___x_217_ = lean_unsigned_to_nat(1u);
v___x_218_ = l_Lean_Syntax_getArg(v_x_206_, v___x_217_);
v___x_219_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinders___closed__1));
lean_inc(v___x_218_);
v___x_220_ = l_Lean_Syntax_isOfKind(v___x_218_, v___x_219_);
if (v___x_220_ == 0)
{
lean_object* v___x_221_; lean_object* v___x_222_; 
lean_dec(v___x_218_);
lean_dec(v_x_206_);
v___x_221_ = lean_box(1);
v___x_222_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_222_, 0, v___x_221_);
lean_ctor_set(v___x_222_, 1, v_a_208_);
return v___x_222_;
}
else
{
lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; uint8_t v___x_226_; 
v___x_223_ = lean_unsigned_to_nat(0u);
v___x_224_ = l_Lean_Syntax_getArg(v___x_218_, v___x_223_);
lean_dec(v___x_218_);
v___x_225_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__1));
lean_inc(v___x_224_);
v___x_226_ = l_Lean_Syntax_isOfKind(v___x_224_, v___x_225_);
if (v___x_226_ == 0)
{
lean_object* v___x_227_; lean_object* v___x_228_; 
lean_dec(v___x_224_);
lean_dec(v_x_206_);
v___x_227_ = lean_box(1);
v___x_228_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_228_, 0, v___x_227_);
lean_ctor_set(v___x_228_, 1, v_a_208_);
return v___x_228_;
}
else
{
lean_object* v___x_229_; uint8_t v___x_230_; 
v___x_229_ = l_Lean_Syntax_getArg(v___x_224_, v___x_223_);
lean_dec(v___x_224_);
lean_inc(v___x_229_);
v___x_230_ = l_Lean_Syntax_matchesNull(v___x_229_, v___x_223_);
if (v___x_230_ == 0)
{
lean_object* v___x_231_; uint8_t v___x_232_; 
v___x_231_ = l_Lean_Syntax_getNumArgs(v___x_229_);
v___x_232_ = lean_nat_dec_le(v___x_217_, v___x_231_);
if (v___x_232_ == 0)
{
lean_dec(v___x_231_);
lean_dec(v___x_229_);
lean_dec(v_x_206_);
v___y_210_ = v_a_208_;
goto v___jp_209_;
}
else
{
lean_object* v___x_233_; lean_object* v___x_234_; uint8_t v___x_235_; 
v___x_233_ = l_Lean_Syntax_getArg(v___x_229_, v___x_223_);
v___x_234_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__1));
lean_inc(v___x_233_);
v___x_235_ = l_Lean_Syntax_isOfKind(v___x_233_, v___x_234_);
if (v___x_235_ == 0)
{
lean_dec(v___x_233_);
lean_dec(v___x_231_);
lean_dec(v___x_229_);
lean_dec(v_x_206_);
v___y_210_ = v_a_208_;
goto v___jp_209_;
}
else
{
lean_object* v___x_236_; lean_object* v___x_237_; uint8_t v___x_238_; 
v___x_236_ = l_Lean_Syntax_getArg(v___x_233_, v___x_217_);
lean_dec(v___x_233_);
v___x_237_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinder___closed__3));
lean_inc(v___x_236_);
v___x_238_ = l_Lean_Syntax_isOfKind(v___x_236_, v___x_237_);
if (v___x_238_ == 0)
{
lean_dec(v___x_236_);
lean_dec(v___x_231_);
lean_dec(v___x_229_);
lean_dec(v_x_206_);
v___y_210_ = v_a_208_;
goto v___jp_209_;
}
else
{
lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; size_t v_sz_245_; size_t v___x_246_; lean_object* v___x_247_; 
v___x_239_ = l_Lean_Syntax_getArgs(v___x_229_);
lean_dec(v___x_229_);
v___x_240_ = l_Array_extract___redArg(v___x_239_, v___x_217_, v___x_231_);
lean_dec_ref(v___x_239_);
v___x_241_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__1));
v___x_242_ = lean_box(2);
v___x_243_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_243_, 0, v___x_242_);
lean_ctor_set(v___x_243_, 1, v___x_241_);
lean_ctor_set(v___x_243_, 2, v___x_240_);
v___x_244_ = l_Lean_Syntax_getArgs(v___x_243_);
lean_dec_ref_known(v___x_243_, 3);
v_sz_245_ = lean_array_size(v___x_244_);
v___x_246_ = ((size_t)0ULL);
v___x_247_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1_spec__0(v_sz_245_, v___x_246_, v___x_244_);
if (lean_obj_tag(v___x_247_) == 0)
{
lean_dec(v___x_236_);
lean_dec(v_x_206_);
v___y_210_ = v_a_208_;
goto v___jp_209_;
}
else
{
lean_object* v_val_248_; lean_object* v_ref_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; size_t v_sz_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; 
v_val_248_ = lean_ctor_get(v___x_247_, 0);
lean_inc(v_val_248_);
lean_dec_ref_known(v___x_247_, 1);
v_ref_249_ = lean_ctor_get(v_a_207_, 5);
v___x_250_ = lean_unsigned_to_nat(3u);
v___x_251_ = l_Lean_Syntax_getArg(v_x_206_, v___x_250_);
lean_dec(v_x_206_);
v___x_252_ = l_Lean_SourceInfo_fromRef(v_ref_249_, v___x_230_);
v___x_253_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__2));
lean_inc_n(v___x_252_, 8);
v___x_254_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_254_, 0, v___x_252_);
lean_ctor_set(v___x_254_, 1, v___x_253_);
v___x_255_ = l_Lean_Syntax_node1(v___x_252_, v___x_219_, v___x_236_);
v___x_256_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__2));
v___x_257_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_257_, 0, v___x_252_);
lean_ctor_set(v___x_257_, 1, v___x_256_);
v___x_258_ = lean_obj_once(&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__3, &lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__3_once, _init_lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__3);
v_sz_259_ = lean_array_size(v_val_248_);
v___x_260_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1_spec__1(v___x_252_, v_sz_259_, v___x_246_, v_val_248_);
v___x_261_ = l_Array_append___redArg(v___x_258_, v___x_260_);
lean_dec_ref(v___x_260_);
v___x_262_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_262_, 0, v___x_252_);
lean_ctor_set(v___x_262_, 1, v___x_241_);
lean_ctor_set(v___x_262_, 2, v___x_261_);
v___x_263_ = l_Lean_Syntax_node1(v___x_252_, v___x_225_, v___x_262_);
v___x_264_ = l_Lean_Syntax_node1(v___x_252_, v___x_219_, v___x_263_);
lean_inc_ref(v___x_257_);
lean_inc_ref(v___x_254_);
v___x_265_ = l_Lean_Syntax_node4(v___x_252_, v___x_213_, v___x_254_, v___x_264_, v___x_257_, v___x_251_);
v___x_266_ = l_Lean_Syntax_node4(v___x_252_, v___x_213_, v___x_254_, v___x_255_, v___x_257_, v___x_265_);
v___x_267_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_267_, 0, v___x_266_);
lean_ctor_set(v___x_267_, 1, v_a_208_);
return v___x_267_;
}
}
}
}
}
else
{
lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; 
lean_dec(v___x_229_);
v___x_268_ = lean_unsigned_to_nat(3u);
v___x_269_ = l_Lean_Syntax_getArg(v_x_206_, v___x_268_);
lean_dec(v_x_206_);
v___x_270_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_270_, 0, v___x_269_);
lean_ctor_set(v___x_270_, 1, v_a_208_);
return v___x_270_;
}
}
}
}
v___jp_209_:
{
lean_object* v___x_211_; lean_object* v___x_212_; 
v___x_211_ = lean_box(1);
v___x_212_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_212_, 0, v___x_211_);
lean_ctor_set(v___x_212_, 1, v___y_210_);
return v___x_212_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___boxed(lean_object* v_x_271_, lean_object* v_a_272_, lean_object* v_a_273_){
_start:
{
lean_object* v_res_274_; 
v_res_274_ = lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1(v_x_271_, v_a_272_, v_a_273_);
lean_dec_ref(v_a_272_);
return v_res_274_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2(lean_object* v_x_297_, lean_object* v_a_298_, lean_object* v_a_299_){
_start:
{
lean_object* v___x_300_; uint8_t v___x_301_; 
v___x_300_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__1));
lean_inc(v_x_297_);
v___x_301_ = l_Lean_Syntax_isOfKind(v_x_297_, v___x_300_);
if (v___x_301_ == 0)
{
lean_object* v___x_302_; lean_object* v___x_303_; 
lean_dec(v_x_297_);
v___x_302_ = lean_box(1);
v___x_303_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_303_, 0, v___x_302_);
lean_ctor_set(v___x_303_, 1, v_a_299_);
return v___x_303_;
}
else
{
lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; uint8_t v___x_307_; 
v___x_304_ = lean_unsigned_to_nat(1u);
v___x_305_ = l_Lean_Syntax_getArg(v_x_297_, v___x_304_);
v___x_306_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinders___closed__1));
lean_inc(v___x_305_);
v___x_307_ = l_Lean_Syntax_isOfKind(v___x_305_, v___x_306_);
if (v___x_307_ == 0)
{
lean_object* v___x_308_; lean_object* v___x_309_; 
lean_dec(v___x_305_);
lean_dec(v_x_297_);
v___x_308_ = lean_box(1);
v___x_309_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_309_, 0, v___x_308_);
lean_ctor_set(v___x_309_, 1, v_a_299_);
return v___x_309_;
}
else
{
lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; uint8_t v___x_313_; 
v___x_310_ = lean_unsigned_to_nat(0u);
v___x_311_ = l_Lean_Syntax_getArg(v___x_305_, v___x_310_);
lean_dec(v___x_305_);
v___x_312_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinder___closed__3));
lean_inc(v___x_311_);
v___x_313_ = l_Lean_Syntax_isOfKind(v___x_311_, v___x_312_);
if (v___x_313_ == 0)
{
lean_object* v___x_314_; lean_object* v___x_315_; 
lean_dec(v___x_311_);
lean_dec(v_x_297_);
v___x_314_ = lean_box(1);
v___x_315_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_315_, 0, v___x_314_);
lean_ctor_set(v___x_315_, 1, v_a_299_);
return v___x_315_;
}
else
{
lean_object* v___x_316_; lean_object* v___x_317_; uint8_t v___x_318_; 
v___x_316_ = l_Lean_Syntax_getArg(v___x_311_, v___x_310_);
v___x_317_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__2));
lean_inc(v___x_316_);
v___x_318_ = l_Lean_Syntax_isOfKind(v___x_316_, v___x_317_);
if (v___x_318_ == 0)
{
lean_object* v___x_319_; lean_object* v___x_320_; 
lean_dec(v___x_316_);
lean_dec(v___x_311_);
lean_dec(v_x_297_);
v___x_319_ = lean_box(1);
v___x_320_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_320_, 0, v___x_319_);
lean_ctor_set(v___x_320_, 1, v_a_299_);
return v___x_320_;
}
else
{
lean_object* v___x_321_; uint8_t v___x_322_; 
v___x_321_ = l_Lean_Syntax_getArg(v___x_311_, v___x_304_);
lean_dec(v___x_311_);
lean_inc(v___x_321_);
v___x_322_ = l_Lean_Syntax_matchesNull(v___x_321_, v___x_310_);
if (v___x_322_ == 0)
{
uint8_t v___x_323_; 
lean_inc(v___x_321_);
v___x_323_ = l_Lean_Syntax_matchesNull(v___x_321_, v___x_304_);
if (v___x_323_ == 0)
{
lean_object* v___x_324_; lean_object* v___x_325_; 
lean_dec(v___x_321_);
lean_dec(v___x_316_);
lean_dec(v_x_297_);
v___x_324_ = lean_box(1);
v___x_325_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_325_, 0, v___x_324_);
lean_ctor_set(v___x_325_, 1, v_a_299_);
return v___x_325_;
}
else
{
lean_object* v___x_326_; lean_object* v___x_327_; uint8_t v___x_328_; 
v___x_326_ = l_Lean_Syntax_getArg(v___x_321_, v___x_310_);
lean_dec(v___x_321_);
v___x_327_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinder___closed__11));
lean_inc(v___x_326_);
v___x_328_ = l_Lean_Syntax_isOfKind(v___x_326_, v___x_327_);
if (v___x_328_ == 0)
{
lean_object* v_ref_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; 
v_ref_329_ = lean_ctor_get(v_a_298_, 5);
v___x_330_ = lean_unsigned_to_nat(3u);
v___x_331_ = l_Lean_Syntax_getArg(v_x_297_, v___x_330_);
lean_dec(v_x_297_);
v___x_332_ = l_Lean_SourceInfo_fromRef(v_ref_329_, v___x_328_);
v___x_333_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__4));
v___x_334_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__5));
lean_inc_n(v___x_332_, 2);
v___x_335_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_335_, 0, v___x_332_);
lean_ctor_set(v___x_335_, 1, v___x_334_);
v___x_336_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__2));
v___x_337_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_337_, 0, v___x_332_);
lean_ctor_set(v___x_337_, 1, v___x_336_);
v___x_338_ = l_Lean_Syntax_node5(v___x_332_, v___x_333_, v___x_335_, v___x_316_, v___x_326_, v___x_337_, v___x_331_);
v___x_339_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_339_, 0, v___x_338_);
lean_ctor_set(v___x_339_, 1, v_a_299_);
return v___x_339_;
}
else
{
lean_object* v_ref_340_; lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; 
v_ref_340_ = lean_ctor_get(v_a_298_, 5);
v___x_341_ = l_Lean_Syntax_getArg(v___x_326_, v___x_304_);
lean_dec(v___x_326_);
v___x_342_ = lean_unsigned_to_nat(3u);
v___x_343_ = l_Lean_Syntax_getArg(v_x_297_, v___x_342_);
lean_dec(v_x_297_);
v___x_344_ = l_Lean_SourceInfo_fromRef(v_ref_340_, v___x_322_);
v___x_345_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__7));
v___x_346_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__5));
lean_inc_n(v___x_344_, 7);
v___x_347_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_347_, 0, v___x_344_);
lean_ctor_set(v___x_347_, 1, v___x_346_);
v___x_348_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__9));
v___x_349_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__11));
v___x_350_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__1));
v___x_351_ = l_Lean_Syntax_node1(v___x_344_, v___x_350_, v___x_316_);
v___x_352_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__12));
v___x_353_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_353_, 0, v___x_344_);
lean_ctor_set(v___x_353_, 1, v___x_352_);
v___x_354_ = l_Lean_Syntax_node2(v___x_344_, v___x_350_, v___x_353_, v___x_341_);
v___x_355_ = l_Lean_Syntax_node2(v___x_344_, v___x_349_, v___x_351_, v___x_354_);
v___x_356_ = l_Lean_Syntax_node1(v___x_344_, v___x_348_, v___x_355_);
v___x_357_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__2));
v___x_358_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_358_, 0, v___x_344_);
lean_ctor_set(v___x_358_, 1, v___x_357_);
v___x_359_ = l_Lean_Syntax_node4(v___x_344_, v___x_345_, v___x_347_, v___x_356_, v___x_358_, v___x_343_);
v___x_360_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_360_, 0, v___x_359_);
lean_ctor_set(v___x_360_, 1, v_a_299_);
return v___x_360_;
}
}
}
else
{
lean_object* v_ref_361_; lean_object* v___x_362_; lean_object* v___x_363_; uint8_t v___x_364_; lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; 
lean_dec(v___x_321_);
v_ref_361_ = lean_ctor_get(v_a_298_, 5);
v___x_362_ = lean_unsigned_to_nat(3u);
v___x_363_ = l_Lean_Syntax_getArg(v_x_297_, v___x_362_);
lean_dec(v_x_297_);
v___x_364_ = 0;
v___x_365_ = l_Lean_SourceInfo_fromRef(v_ref_361_, v___x_364_);
v___x_366_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__7));
v___x_367_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__5));
lean_inc_n(v___x_365_, 6);
v___x_368_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_368_, 0, v___x_365_);
lean_ctor_set(v___x_368_, 1, v___x_367_);
v___x_369_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__9));
v___x_370_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__11));
v___x_371_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__1));
v___x_372_ = l_Lean_Syntax_node1(v___x_365_, v___x_371_, v___x_316_);
v___x_373_ = lean_obj_once(&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__3, &lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__3_once, _init_lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__3);
v___x_374_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_374_, 0, v___x_365_);
lean_ctor_set(v___x_374_, 1, v___x_371_);
lean_ctor_set(v___x_374_, 2, v___x_373_);
v___x_375_ = l_Lean_Syntax_node2(v___x_365_, v___x_370_, v___x_372_, v___x_374_);
v___x_376_ = l_Lean_Syntax_node1(v___x_365_, v___x_369_, v___x_375_);
v___x_377_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__2));
v___x_378_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_378_, 0, v___x_365_);
lean_ctor_set(v___x_378_, 1, v___x_377_);
v___x_379_ = l_Lean_Syntax_node4(v___x_365_, v___x_366_, v___x_368_, v___x_376_, v___x_378_, v___x_363_);
v___x_380_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_380_, 0, v___x_379_);
lean_ctor_set(v___x_380_, 1, v_a_299_);
return v___x_380_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___boxed(lean_object* v_x_381_, lean_object* v_a_382_, lean_object* v_a_383_){
_start:
{
lean_object* v_res_384_; 
v_res_384_ = lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2(v_x_381_, v_a_382_, v_a_383_);
lean_dec_ref(v_a_382_);
return v_res_384_;
}
}
static lean_object* _init_lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__4(void){
_start:
{
lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; 
v___x_393_ = lp_batteries_Batteries_ExtendedBinder_extBinders;
v___x_394_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__3));
v___x_395_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinder___closed__5));
v___x_396_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_396_, 0, v___x_395_);
lean_ctor_set(v___x_396_, 1, v___x_394_);
lean_ctor_set(v___x_396_, 2, v___x_393_);
return v___x_396_;
}
}
static lean_object* _init_lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__5(void){
_start:
{
lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; 
v___x_397_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c___00__closed__6));
v___x_398_ = lean_obj_once(&lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__4, &lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__4_once, _init_lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__4);
v___x_399_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinder___closed__5));
v___x_400_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_400_, 0, v___x_399_);
lean_ctor_set(v___x_400_, 1, v___x_398_);
lean_ctor_set(v___x_400_, 2, v___x_397_);
return v___x_400_;
}
}
static lean_object* _init_lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__6(void){
_start:
{
lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; 
v___x_401_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinder___closed__16));
v___x_402_ = lean_obj_once(&lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__5, &lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__5_once, _init_lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__5);
v___x_403_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinder___closed__5));
v___x_404_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_404_, 0, v___x_403_);
lean_ctor_set(v___x_404_, 1, v___x_402_);
lean_ctor_set(v___x_404_, 2, v___x_401_);
return v___x_404_;
}
}
static lean_object* _init_lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__7(void){
_start:
{
lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; 
v___x_405_ = lean_obj_once(&lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__6, &lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__6_once, _init_lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__6);
v___x_406_ = lean_unsigned_to_nat(1022u);
v___x_407_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__1));
v___x_408_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_408_, 0, v___x_407_);
lean_ctor_set(v___x_408_, 1, v___x_406_);
lean_ctor_set(v___x_408_, 2, v___x_405_);
return v___x_408_;
}
}
static lean_object* _init_lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c__(void){
_start:
{
lean_object* v___x_409_; 
v___x_409_ = lean_obj_once(&lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__7, &lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__7_once, _init_lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__7);
return v___x_409_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____1(lean_object* v_x_410_, lean_object* v_a_411_, lean_object* v_a_412_){
_start:
{
lean_object* v___y_414_; lean_object* v___x_417_; uint8_t v___x_418_; 
v___x_417_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__1));
lean_inc(v_x_410_);
v___x_418_ = l_Lean_Syntax_isOfKind(v_x_410_, v___x_417_);
if (v___x_418_ == 0)
{
lean_object* v___x_419_; lean_object* v___x_420_; 
lean_dec(v_x_410_);
v___x_419_ = lean_box(1);
v___x_420_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_420_, 0, v___x_419_);
lean_ctor_set(v___x_420_, 1, v_a_412_);
return v___x_420_;
}
else
{
lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v___x_423_; uint8_t v___x_424_; 
v___x_421_ = lean_unsigned_to_nat(1u);
v___x_422_ = l_Lean_Syntax_getArg(v_x_410_, v___x_421_);
v___x_423_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinders___closed__1));
lean_inc(v___x_422_);
v___x_424_ = l_Lean_Syntax_isOfKind(v___x_422_, v___x_423_);
if (v___x_424_ == 0)
{
lean_object* v___x_425_; lean_object* v___x_426_; 
lean_dec(v___x_422_);
lean_dec(v_x_410_);
v___x_425_ = lean_box(1);
v___x_426_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_426_, 0, v___x_425_);
lean_ctor_set(v___x_426_, 1, v_a_412_);
return v___x_426_;
}
else
{
lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; uint8_t v___x_430_; 
v___x_427_ = lean_unsigned_to_nat(0u);
v___x_428_ = l_Lean_Syntax_getArg(v___x_422_, v___x_427_);
lean_dec(v___x_422_);
v___x_429_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinderCollection___closed__1));
lean_inc(v___x_428_);
v___x_430_ = l_Lean_Syntax_isOfKind(v___x_428_, v___x_429_);
if (v___x_430_ == 0)
{
lean_object* v___x_431_; lean_object* v___x_432_; 
lean_dec(v___x_428_);
lean_dec(v_x_410_);
v___x_431_ = lean_box(1);
v___x_432_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_432_, 0, v___x_431_);
lean_ctor_set(v___x_432_, 1, v_a_412_);
return v___x_432_;
}
else
{
lean_object* v___x_433_; uint8_t v___x_434_; 
v___x_433_ = l_Lean_Syntax_getArg(v___x_428_, v___x_427_);
lean_dec(v___x_428_);
lean_inc(v___x_433_);
v___x_434_ = l_Lean_Syntax_matchesNull(v___x_433_, v___x_427_);
if (v___x_434_ == 0)
{
lean_object* v___x_435_; uint8_t v___x_436_; 
v___x_435_ = l_Lean_Syntax_getNumArgs(v___x_433_);
v___x_436_ = lean_nat_dec_le(v___x_421_, v___x_435_);
if (v___x_436_ == 0)
{
lean_dec(v___x_435_);
lean_dec(v___x_433_);
lean_dec(v_x_410_);
v___y_414_ = v_a_412_;
goto v___jp_413_;
}
else
{
lean_object* v___x_437_; lean_object* v___x_438_; uint8_t v___x_439_; 
v___x_437_ = l_Lean_Syntax_getArg(v___x_433_, v___x_427_);
v___x_438_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized___closed__1));
lean_inc(v___x_437_);
v___x_439_ = l_Lean_Syntax_isOfKind(v___x_437_, v___x_438_);
if (v___x_439_ == 0)
{
lean_dec(v___x_437_);
lean_dec(v___x_435_);
lean_dec(v___x_433_);
lean_dec(v_x_410_);
v___y_414_ = v_a_412_;
goto v___jp_413_;
}
else
{
lean_object* v___x_440_; lean_object* v___x_441_; uint8_t v___x_442_; 
v___x_440_ = l_Lean_Syntax_getArg(v___x_437_, v___x_421_);
lean_dec(v___x_437_);
v___x_441_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinder___closed__3));
lean_inc(v___x_440_);
v___x_442_ = l_Lean_Syntax_isOfKind(v___x_440_, v___x_441_);
if (v___x_442_ == 0)
{
lean_dec(v___x_440_);
lean_dec(v___x_435_);
lean_dec(v___x_433_);
lean_dec(v_x_410_);
v___y_414_ = v_a_412_;
goto v___jp_413_;
}
else
{
lean_object* v___x_443_; lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; size_t v_sz_449_; size_t v___x_450_; lean_object* v___x_451_; 
v___x_443_ = l_Lean_Syntax_getArgs(v___x_433_);
lean_dec(v___x_433_);
v___x_444_ = l_Array_extract___redArg(v___x_443_, v___x_421_, v___x_435_);
lean_dec_ref(v___x_443_);
v___x_445_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__1));
v___x_446_ = lean_box(2);
v___x_447_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_447_, 0, v___x_446_);
lean_ctor_set(v___x_447_, 1, v___x_445_);
lean_ctor_set(v___x_447_, 2, v___x_444_);
v___x_448_ = l_Lean_Syntax_getArgs(v___x_447_);
lean_dec_ref_known(v___x_447_, 3);
v_sz_449_ = lean_array_size(v___x_448_);
v___x_450_ = ((size_t)0ULL);
v___x_451_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1_spec__0(v_sz_449_, v___x_450_, v___x_448_);
if (lean_obj_tag(v___x_451_) == 0)
{
lean_dec(v___x_440_);
lean_dec(v_x_410_);
v___y_414_ = v_a_412_;
goto v___jp_413_;
}
else
{
lean_object* v_val_452_; lean_object* v_ref_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; size_t v_sz_463_; lean_object* v___x_464_; lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; 
v_val_452_ = lean_ctor_get(v___x_451_, 0);
lean_inc(v_val_452_);
lean_dec_ref_known(v___x_451_, 1);
v_ref_453_ = lean_ctor_get(v_a_411_, 5);
v___x_454_ = lean_unsigned_to_nat(3u);
v___x_455_ = l_Lean_Syntax_getArg(v_x_410_, v___x_454_);
lean_dec(v_x_410_);
v___x_456_ = l_Lean_SourceInfo_fromRef(v_ref_453_, v___x_434_);
v___x_457_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__2));
lean_inc_n(v___x_456_, 8);
v___x_458_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_458_, 0, v___x_456_);
lean_ctor_set(v___x_458_, 1, v___x_457_);
v___x_459_ = l_Lean_Syntax_node1(v___x_456_, v___x_423_, v___x_440_);
v___x_460_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__2));
v___x_461_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_461_, 0, v___x_456_);
lean_ctor_set(v___x_461_, 1, v___x_460_);
v___x_462_ = lean_obj_once(&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__3, &lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__3_once, _init_lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__3);
v_sz_463_ = lean_array_size(v_val_452_);
v___x_464_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1_spec__1(v___x_456_, v_sz_463_, v___x_450_, v_val_452_);
v___x_465_ = l_Array_append___redArg(v___x_462_, v___x_464_);
lean_dec_ref(v___x_464_);
v___x_466_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_466_, 0, v___x_456_);
lean_ctor_set(v___x_466_, 1, v___x_445_);
lean_ctor_set(v___x_466_, 2, v___x_465_);
v___x_467_ = l_Lean_Syntax_node1(v___x_456_, v___x_429_, v___x_466_);
v___x_468_ = l_Lean_Syntax_node1(v___x_456_, v___x_423_, v___x_467_);
lean_inc_ref(v___x_461_);
lean_inc_ref(v___x_458_);
v___x_469_ = l_Lean_Syntax_node4(v___x_456_, v___x_417_, v___x_458_, v___x_468_, v___x_461_, v___x_455_);
v___x_470_ = l_Lean_Syntax_node4(v___x_456_, v___x_417_, v___x_458_, v___x_459_, v___x_461_, v___x_469_);
v___x_471_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_471_, 0, v___x_470_);
lean_ctor_set(v___x_471_, 1, v_a_412_);
return v___x_471_;
}
}
}
}
}
else
{
lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; 
lean_dec(v___x_433_);
v___x_472_ = lean_unsigned_to_nat(3u);
v___x_473_ = l_Lean_Syntax_getArg(v_x_410_, v___x_472_);
lean_dec(v_x_410_);
v___x_474_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_474_, 0, v___x_473_);
lean_ctor_set(v___x_474_, 1, v_a_412_);
return v___x_474_;
}
}
}
}
v___jp_413_:
{
lean_object* v___x_415_; lean_object* v___x_416_; 
v___x_415_ = lean_box(1);
v___x_416_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_416_, 0, v___x_415_);
lean_ctor_set(v___x_416_, 1, v___y_414_);
return v___x_416_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____1___boxed(lean_object* v_x_475_, lean_object* v_a_476_, lean_object* v_a_477_){
_start:
{
lean_object* v_res_478_; 
v_res_478_ = lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____1(v_x_475_, v_a_476_, v_a_477_);
lean_dec_ref(v_a_476_);
return v_res_478_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2(lean_object* v_x_508_, lean_object* v_a_509_, lean_object* v_a_510_){
_start:
{
lean_object* v___x_511_; uint8_t v___x_512_; 
v___x_511_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c___00__closed__1));
lean_inc(v_x_508_);
v___x_512_ = l_Lean_Syntax_isOfKind(v_x_508_, v___x_511_);
if (v___x_512_ == 0)
{
lean_object* v___x_513_; lean_object* v___x_514_; 
lean_dec(v_x_508_);
v___x_513_ = lean_box(1);
v___x_514_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_514_, 0, v___x_513_);
lean_ctor_set(v___x_514_, 1, v_a_510_);
return v___x_514_;
}
else
{
lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; uint8_t v___x_518_; 
v___x_515_ = lean_unsigned_to_nat(1u);
v___x_516_ = l_Lean_Syntax_getArg(v_x_508_, v___x_515_);
v___x_517_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinders___closed__1));
lean_inc(v___x_516_);
v___x_518_ = l_Lean_Syntax_isOfKind(v___x_516_, v___x_517_);
if (v___x_518_ == 0)
{
lean_object* v___x_519_; lean_object* v___x_520_; 
lean_dec(v___x_516_);
lean_dec(v_x_508_);
v___x_519_ = lean_box(1);
v___x_520_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_520_, 0, v___x_519_);
lean_ctor_set(v___x_520_, 1, v_a_510_);
return v___x_520_;
}
else
{
lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; uint8_t v___x_524_; 
v___x_521_ = lean_unsigned_to_nat(0u);
v___x_522_ = l_Lean_Syntax_getArg(v___x_516_, v___x_521_);
lean_dec(v___x_516_);
v___x_523_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinder___closed__3));
lean_inc(v___x_522_);
v___x_524_ = l_Lean_Syntax_isOfKind(v___x_522_, v___x_523_);
if (v___x_524_ == 0)
{
lean_object* v___x_525_; lean_object* v___x_526_; 
lean_dec(v___x_522_);
lean_dec(v_x_508_);
v___x_525_ = lean_box(1);
v___x_526_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_526_, 0, v___x_525_);
lean_ctor_set(v___x_526_, 1, v_a_510_);
return v___x_526_;
}
else
{
lean_object* v___x_527_; lean_object* v___x_528_; uint8_t v___x_529_; 
v___x_527_ = l_Lean_Syntax_getArg(v___x_522_, v___x_521_);
v___x_528_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__2));
lean_inc(v___x_527_);
v___x_529_ = l_Lean_Syntax_isOfKind(v___x_527_, v___x_528_);
if (v___x_529_ == 0)
{
if (v___x_529_ == 0)
{
lean_object* v___x_530_; lean_object* v___x_531_; 
lean_dec(v___x_527_);
lean_dec(v___x_522_);
lean_dec(v_x_508_);
v___x_530_ = lean_box(1);
v___x_531_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_531_, 0, v___x_530_);
lean_ctor_set(v___x_531_, 1, v_a_510_);
return v___x_531_;
}
else
{
lean_object* v___x_532_; uint8_t v___x_533_; 
v___x_532_ = l_Lean_Syntax_getArg(v___x_522_, v___x_515_);
lean_dec(v___x_522_);
lean_inc(v___x_532_);
v___x_533_ = l_Lean_Syntax_matchesNull(v___x_532_, v___x_515_);
if (v___x_533_ == 0)
{
lean_object* v___x_534_; lean_object* v___x_535_; 
lean_dec(v___x_532_);
lean_dec(v___x_527_);
lean_dec(v_x_508_);
v___x_534_ = lean_box(1);
v___x_535_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_535_, 0, v___x_534_);
lean_ctor_set(v___x_535_, 1, v_a_510_);
return v___x_535_;
}
else
{
lean_object* v_ref_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; 
v_ref_536_ = lean_ctor_get(v_a_509_, 5);
v___x_537_ = l_Lean_Syntax_getArg(v___x_532_, v___x_521_);
lean_dec(v___x_532_);
v___x_538_ = lean_unsigned_to_nat(3u);
v___x_539_ = l_Lean_Syntax_getArg(v_x_508_, v___x_538_);
lean_dec(v_x_508_);
v___x_540_ = l_Lean_SourceInfo_fromRef(v_ref_536_, v___x_529_);
v___x_541_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__1));
v___x_542_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__2));
lean_inc_n(v___x_540_, 2);
v___x_543_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_543_, 0, v___x_540_);
lean_ctor_set(v___x_543_, 1, v___x_542_);
v___x_544_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__2));
v___x_545_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_545_, 0, v___x_540_);
lean_ctor_set(v___x_545_, 1, v___x_544_);
v___x_546_ = l_Lean_Syntax_node5(v___x_540_, v___x_541_, v___x_543_, v___x_527_, v___x_537_, v___x_545_, v___x_539_);
v___x_547_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_547_, 0, v___x_546_);
lean_ctor_set(v___x_547_, 1, v_a_510_);
return v___x_547_;
}
}
}
else
{
lean_object* v___x_548_; lean_object* v___x_549_; uint8_t v___x_550_; 
v___x_548_ = l_Lean_Syntax_getArg(v___x_527_, v___x_521_);
v___x_549_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__6));
lean_inc(v___x_548_);
v___x_550_ = l_Lean_Syntax_isOfKind(v___x_548_, v___x_549_);
if (v___x_550_ == 0)
{
lean_object* v___x_551_; uint8_t v___x_552_; 
v___x_551_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__8));
lean_inc(v___x_548_);
v___x_552_ = l_Lean_Syntax_isOfKind(v___x_548_, v___x_551_);
if (v___x_552_ == 0)
{
lean_object* v___x_553_; uint8_t v___x_554_; 
lean_dec(v___x_548_);
v___x_553_ = l_Lean_Syntax_getArg(v___x_522_, v___x_515_);
lean_dec(v___x_522_);
lean_inc(v___x_553_);
v___x_554_ = l_Lean_Syntax_matchesNull(v___x_553_, v___x_515_);
if (v___x_554_ == 0)
{
lean_object* v___x_555_; lean_object* v___x_556_; 
lean_dec(v___x_553_);
lean_dec(v___x_527_);
lean_dec(v_x_508_);
v___x_555_ = lean_box(1);
v___x_556_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_556_, 0, v___x_555_);
lean_ctor_set(v___x_556_, 1, v_a_510_);
return v___x_556_;
}
else
{
lean_object* v_ref_557_; lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_561_; lean_object* v___x_562_; lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; 
v_ref_557_ = lean_ctor_get(v_a_509_, 5);
v___x_558_ = l_Lean_Syntax_getArg(v___x_553_, v___x_521_);
lean_dec(v___x_553_);
v___x_559_ = lean_unsigned_to_nat(3u);
v___x_560_ = l_Lean_Syntax_getArg(v_x_508_, v___x_559_);
lean_dec(v_x_508_);
v___x_561_ = l_Lean_SourceInfo_fromRef(v_ref_557_, v___x_552_);
v___x_562_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__1));
v___x_563_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__2));
lean_inc_n(v___x_561_, 2);
v___x_564_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_564_, 0, v___x_561_);
lean_ctor_set(v___x_564_, 1, v___x_563_);
v___x_565_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__2));
v___x_566_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_566_, 0, v___x_561_);
lean_ctor_set(v___x_566_, 1, v___x_565_);
v___x_567_ = l_Lean_Syntax_node5(v___x_561_, v___x_562_, v___x_564_, v___x_527_, v___x_558_, v___x_566_, v___x_560_);
v___x_568_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_568_, 0, v___x_567_);
lean_ctor_set(v___x_568_, 1, v_a_510_);
return v___x_568_;
}
}
else
{
lean_object* v___x_569_; uint8_t v___x_570_; 
v___x_569_ = l_Lean_Syntax_getArg(v___x_522_, v___x_515_);
lean_dec(v___x_522_);
lean_inc(v___x_569_);
v___x_570_ = l_Lean_Syntax_matchesNull(v___x_569_, v___x_521_);
if (v___x_570_ == 0)
{
uint8_t v___x_571_; 
lean_inc(v___x_569_);
v___x_571_ = l_Lean_Syntax_matchesNull(v___x_569_, v___x_515_);
if (v___x_571_ == 0)
{
lean_object* v___x_572_; lean_object* v___x_573_; 
lean_dec(v___x_569_);
lean_dec(v___x_548_);
lean_dec(v___x_527_);
lean_dec(v_x_508_);
v___x_572_ = lean_box(1);
v___x_573_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_573_, 0, v___x_572_);
lean_ctor_set(v___x_573_, 1, v_a_510_);
return v___x_573_;
}
else
{
lean_object* v___x_574_; lean_object* v___x_575_; uint8_t v___x_576_; 
v___x_574_ = l_Lean_Syntax_getArg(v___x_569_, v___x_521_);
lean_dec(v___x_569_);
v___x_575_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinder___closed__11));
lean_inc(v___x_574_);
v___x_576_ = l_Lean_Syntax_isOfKind(v___x_574_, v___x_575_);
if (v___x_576_ == 0)
{
lean_object* v_ref_577_; lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v___x_587_; 
lean_dec(v___x_548_);
v_ref_577_ = lean_ctor_get(v_a_509_, 5);
v___x_578_ = lean_unsigned_to_nat(3u);
v___x_579_ = l_Lean_Syntax_getArg(v_x_508_, v___x_578_);
lean_dec(v_x_508_);
v___x_580_ = l_Lean_SourceInfo_fromRef(v_ref_577_, v___x_550_);
v___x_581_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__1));
v___x_582_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__2));
lean_inc_n(v___x_580_, 2);
v___x_583_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_583_, 0, v___x_580_);
lean_ctor_set(v___x_583_, 1, v___x_582_);
v___x_584_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__2));
v___x_585_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_585_, 0, v___x_580_);
lean_ctor_set(v___x_585_, 1, v___x_584_);
v___x_586_ = l_Lean_Syntax_node5(v___x_580_, v___x_581_, v___x_583_, v___x_527_, v___x_574_, v___x_585_, v___x_579_);
v___x_587_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_587_, 0, v___x_586_);
lean_ctor_set(v___x_587_, 1, v_a_510_);
return v___x_587_;
}
else
{
lean_object* v_ref_588_; lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; lean_object* v___x_592_; lean_object* v___x_593_; lean_object* v___x_594_; lean_object* v___x_595_; lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_605_; lean_object* v___x_606_; 
lean_dec(v___x_527_);
v_ref_588_ = lean_ctor_get(v_a_509_, 5);
v___x_589_ = l_Lean_Syntax_getArg(v___x_574_, v___x_515_);
lean_dec(v___x_574_);
v___x_590_ = lean_unsigned_to_nat(3u);
v___x_591_ = l_Lean_Syntax_getArg(v_x_508_, v___x_590_);
lean_dec(v_x_508_);
v___x_592_ = l_Lean_SourceInfo_fromRef(v_ref_588_, v___x_550_);
v___x_593_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__10));
v___x_594_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__2));
lean_inc_n(v___x_592_, 6);
v___x_595_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_595_, 0, v___x_592_);
lean_ctor_set(v___x_595_, 1, v___x_594_);
v___x_596_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__1));
v___x_597_ = l_Lean_Syntax_node1(v___x_592_, v___x_596_, v___x_548_);
v___x_598_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__12));
v___x_599_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__12));
v___x_600_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_600_, 0, v___x_592_);
lean_ctor_set(v___x_600_, 1, v___x_599_);
v___x_601_ = l_Lean_Syntax_node2(v___x_592_, v___x_598_, v___x_600_, v___x_589_);
v___x_602_ = l_Lean_Syntax_node1(v___x_592_, v___x_596_, v___x_601_);
v___x_603_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__2));
v___x_604_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_604_, 0, v___x_592_);
lean_ctor_set(v___x_604_, 1, v___x_603_);
v___x_605_ = l_Lean_Syntax_node5(v___x_592_, v___x_593_, v___x_595_, v___x_597_, v___x_602_, v___x_604_, v___x_591_);
v___x_606_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_606_, 0, v___x_605_);
lean_ctor_set(v___x_606_, 1, v_a_510_);
return v___x_606_;
}
}
}
else
{
lean_object* v_ref_607_; lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; lean_object* v___x_621_; 
lean_dec(v___x_569_);
lean_dec(v___x_527_);
v_ref_607_ = lean_ctor_get(v_a_509_, 5);
v___x_608_ = lean_unsigned_to_nat(3u);
v___x_609_ = l_Lean_Syntax_getArg(v_x_508_, v___x_608_);
lean_dec(v_x_508_);
v___x_610_ = l_Lean_SourceInfo_fromRef(v_ref_607_, v___x_550_);
v___x_611_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__10));
v___x_612_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__2));
lean_inc_n(v___x_610_, 4);
v___x_613_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_613_, 0, v___x_610_);
lean_ctor_set(v___x_613_, 1, v___x_612_);
v___x_614_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__1));
v___x_615_ = l_Lean_Syntax_node1(v___x_610_, v___x_614_, v___x_548_);
v___x_616_ = lean_obj_once(&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__3, &lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__3_once, _init_lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__3);
v___x_617_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_617_, 0, v___x_610_);
lean_ctor_set(v___x_617_, 1, v___x_614_);
lean_ctor_set(v___x_617_, 2, v___x_616_);
v___x_618_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__2));
v___x_619_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_619_, 0, v___x_610_);
lean_ctor_set(v___x_619_, 1, v___x_618_);
v___x_620_ = l_Lean_Syntax_node5(v___x_610_, v___x_611_, v___x_613_, v___x_615_, v___x_617_, v___x_619_, v___x_609_);
v___x_621_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_621_, 0, v___x_620_);
lean_ctor_set(v___x_621_, 1, v_a_510_);
return v___x_621_;
}
}
}
else
{
lean_object* v___x_622_; uint8_t v___x_623_; 
lean_dec(v___x_548_);
v___x_622_ = l_Lean_Syntax_getArg(v___x_522_, v___x_515_);
lean_dec(v___x_522_);
lean_inc(v___x_622_);
v___x_623_ = l_Lean_Syntax_matchesNull(v___x_622_, v___x_521_);
if (v___x_623_ == 0)
{
uint8_t v___x_624_; 
lean_inc(v___x_622_);
v___x_624_ = l_Lean_Syntax_matchesNull(v___x_622_, v___x_515_);
if (v___x_624_ == 0)
{
lean_object* v___x_625_; lean_object* v___x_626_; 
lean_dec(v___x_622_);
lean_dec(v___x_527_);
lean_dec(v_x_508_);
v___x_625_ = lean_box(1);
v___x_626_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_626_, 0, v___x_625_);
lean_ctor_set(v___x_626_, 1, v_a_510_);
return v___x_626_;
}
else
{
lean_object* v___x_627_; lean_object* v___x_628_; uint8_t v___x_629_; 
v___x_627_ = l_Lean_Syntax_getArg(v___x_622_, v___x_521_);
lean_dec(v___x_622_);
v___x_628_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder_extBinder___closed__11));
lean_inc(v___x_627_);
v___x_629_ = l_Lean_Syntax_isOfKind(v___x_627_, v___x_628_);
if (v___x_629_ == 0)
{
lean_object* v_ref_630_; lean_object* v___x_631_; lean_object* v___x_632_; lean_object* v___x_633_; lean_object* v___x_634_; lean_object* v___x_635_; lean_object* v___x_636_; lean_object* v___x_637_; lean_object* v___x_638_; lean_object* v___x_639_; lean_object* v___x_640_; 
v_ref_630_ = lean_ctor_get(v_a_509_, 5);
v___x_631_ = lean_unsigned_to_nat(3u);
v___x_632_ = l_Lean_Syntax_getArg(v_x_508_, v___x_631_);
lean_dec(v_x_508_);
v___x_633_ = l_Lean_SourceInfo_fromRef(v_ref_630_, v___x_629_);
v___x_634_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__1));
v___x_635_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__2));
lean_inc_n(v___x_633_, 2);
v___x_636_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_636_, 0, v___x_633_);
lean_ctor_set(v___x_636_, 1, v___x_635_);
v___x_637_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__2));
v___x_638_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_638_, 0, v___x_633_);
lean_ctor_set(v___x_638_, 1, v___x_637_);
v___x_639_ = l_Lean_Syntax_node5(v___x_633_, v___x_634_, v___x_636_, v___x_527_, v___x_627_, v___x_638_, v___x_632_);
v___x_640_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_640_, 0, v___x_639_);
lean_ctor_set(v___x_640_, 1, v_a_510_);
return v___x_640_;
}
else
{
lean_object* v_ref_641_; lean_object* v___x_642_; lean_object* v___x_643_; lean_object* v___x_644_; lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v___x_647_; lean_object* v___x_648_; lean_object* v___x_649_; lean_object* v___x_650_; lean_object* v___x_651_; lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; lean_object* v___x_655_; lean_object* v___x_656_; lean_object* v___x_657_; lean_object* v___x_658_; lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v___x_661_; lean_object* v___x_662_; 
lean_dec(v___x_527_);
v_ref_641_ = lean_ctor_get(v_a_509_, 5);
v___x_642_ = l_Lean_Syntax_getArg(v___x_627_, v___x_515_);
lean_dec(v___x_627_);
v___x_643_ = lean_unsigned_to_nat(3u);
v___x_644_ = l_Lean_Syntax_getArg(v_x_508_, v___x_643_);
lean_dec(v_x_508_);
v___x_645_ = l_Lean_SourceInfo_fromRef(v_ref_641_, v___x_623_);
v___x_646_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__10));
v___x_647_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__2));
lean_inc_n(v___x_645_, 8);
v___x_648_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_648_, 0, v___x_645_);
lean_ctor_set(v___x_648_, 1, v___x_647_);
v___x_649_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__1));
v___x_650_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__13));
v___x_651_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_651_, 0, v___x_645_);
lean_ctor_set(v___x_651_, 1, v___x_650_);
v___x_652_ = l_Lean_Syntax_node1(v___x_645_, v___x_549_, v___x_651_);
v___x_653_ = l_Lean_Syntax_node1(v___x_645_, v___x_649_, v___x_652_);
v___x_654_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__12));
v___x_655_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____2___closed__12));
v___x_656_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_656_, 0, v___x_645_);
lean_ctor_set(v___x_656_, 1, v___x_655_);
v___x_657_ = l_Lean_Syntax_node2(v___x_645_, v___x_654_, v___x_656_, v___x_642_);
v___x_658_ = l_Lean_Syntax_node1(v___x_645_, v___x_649_, v___x_657_);
v___x_659_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__2));
v___x_660_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_660_, 0, v___x_645_);
lean_ctor_set(v___x_660_, 1, v___x_659_);
v___x_661_ = l_Lean_Syntax_node5(v___x_645_, v___x_646_, v___x_648_, v___x_653_, v___x_658_, v___x_660_, v___x_644_);
v___x_662_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_662_, 0, v___x_661_);
lean_ctor_set(v___x_662_, 1, v_a_510_);
return v___x_662_;
}
}
}
else
{
lean_object* v_ref_663_; lean_object* v___x_664_; lean_object* v___x_665_; uint8_t v___x_666_; lean_object* v___x_667_; lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_670_; lean_object* v___x_671_; lean_object* v___x_672_; lean_object* v___x_673_; lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v___x_678_; lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_681_; 
lean_dec(v___x_622_);
lean_dec(v___x_527_);
v_ref_663_ = lean_ctor_get(v_a_509_, 5);
v___x_664_ = lean_unsigned_to_nat(3u);
v___x_665_ = l_Lean_Syntax_getArg(v_x_508_, v___x_664_);
lean_dec(v_x_508_);
v___x_666_ = 0;
v___x_667_ = l_Lean_SourceInfo_fromRef(v_ref_663_, v___x_666_);
v___x_668_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__10));
v___x_669_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__2));
lean_inc_n(v___x_667_, 6);
v___x_670_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_670_, 0, v___x_667_);
lean_ctor_set(v___x_670_, 1, v___x_669_);
v___x_671_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__1));
v___x_672_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___closed__13));
v___x_673_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_673_, 0, v___x_667_);
lean_ctor_set(v___x_673_, 1, v___x_672_);
v___x_674_ = l_Lean_Syntax_node1(v___x_667_, v___x_549_, v___x_673_);
v___x_675_ = l_Lean_Syntax_node1(v___x_667_, v___x_671_, v___x_674_);
v___x_676_ = lean_obj_once(&lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__3, &lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__3_once, _init_lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__3);
v___x_677_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_677_, 0, v___x_667_);
lean_ctor_set(v___x_677_, 1, v___x_671_);
lean_ctor_set(v___x_677_, 2, v___x_676_);
v___x_678_ = ((lean_object*)(lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2203_u1d49___x2c____1___closed__2));
v___x_679_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_679_, 0, v___x_667_);
lean_ctor_set(v___x_679_, 1, v___x_678_);
v___x_680_ = l_Lean_Syntax_node5(v___x_667_, v___x_668_, v___x_670_, v___x_675_, v___x_677_, v___x_679_, v___x_665_);
v___x_681_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_681_, 0, v___x_680_);
lean_ctor_set(v___x_681_, 1, v_a_510_);
return v___x_681_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2___boxed(lean_object* v_x_682_, lean_object* v_a_683_, lean_object* v_a_684_){
_start:
{
lean_object* v_res_685_; 
v_res_685_ = lp_batteries_Batteries_ExtendedBinder___aux__Batteries__Util__ExtendedBinder______macroRules__Batteries__ExtendedBinder__term_u2200_u1d49___x2c____2(v_x_682_, v_a_683_, v_a_684_);
lean_dec_ref(v_a_683_);
return v_res_685_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Util_ExtendedBinder(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Util_ExtendedBinder(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_batteries_Batteries_ExtendedBinder_extBinder = _init_lp_batteries_Batteries_ExtendedBinder_extBinder();
lean_mark_persistent(lp_batteries_Batteries_ExtendedBinder_extBinder);
lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized = _init_lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized();
lean_mark_persistent(lp_batteries_Batteries_ExtendedBinder_extBinderParenthesized);
lp_batteries_Batteries_ExtendedBinder_extBinderCollection = _init_lp_batteries_Batteries_ExtendedBinder_extBinderCollection();
lean_mark_persistent(lp_batteries_Batteries_ExtendedBinder_extBinderCollection);
lp_batteries_Batteries_ExtendedBinder_extBinders = _init_lp_batteries_Batteries_ExtendedBinder_extBinders();
lean_mark_persistent(lp_batteries_Batteries_ExtendedBinder_extBinders);
lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c__ = _init_lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c__();
lean_mark_persistent(lp_batteries_Batteries_ExtendedBinder_term_u2203_u1d49___x2c__);
lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c__ = _init_lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c__();
lean_mark_persistent(lp_batteries_Batteries_ExtendedBinder_term_u2200_u1d49___x2c__);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Util_ExtendedBinder(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Util_ExtendedBinder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Util_ExtendedBinder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Util_ExtendedBinder(builtin);
}
#ifdef __cplusplus
}
#endif
