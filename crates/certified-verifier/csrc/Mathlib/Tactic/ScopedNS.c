// Lean compiler output
// Module: Mathlib.Tactic.ScopedNS
// Imports: public import Init public meta import Init public import Mathlib.Util.WithWeakNamespace
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
extern lean_object* l_Lean_rootNamespace;
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
lean_object* l_Lean_mkIdentFrom(lean_object*, lean_object*, uint8_t);
size_t lean_array_size(lean_object*);
lean_object* l_Lean_mkSepArray(lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Array_mkArray1___redArg(lean_object*);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "scopedNS"};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__2_value),LEAN_SCALAR_PTR_LITERAL(20, 10, 101, 225, 139, 111, 199, 9)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__6_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "docComment"};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__8_value),LEAN_SCALAR_PTR_LITERAL(229, 56, 215, 222, 243, 187, 251, 54)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "attributes"};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__12_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__13_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__16_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__14_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__16_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__15_value),LEAN_SCALAR_PTR_LITERAL(66, 184, 196, 169, 25, 125, 40, 35)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 8}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__11_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__18_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "scoped"};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__19_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__21_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__22_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__23_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__22_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__24_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__25_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__26_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__27_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__28_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__25_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__28_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__29_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "] "};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__30_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__30_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__31_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__29_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__31_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__32_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "command"};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__33_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__33_value),LEAN_SCALAR_PTR_LITERAL(29, 69, 134, 125, 237, 175, 69, 70)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__34_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__34_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__35_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__32_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__35_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__36_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_scopedNS___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__3_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__36_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__37_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_scopedNS = (const lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__37_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__1(lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__1_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "attrInstance"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__12_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__13_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__14_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__3_value_aux_2),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(241, 75, 242, 110, 47, 5, 20, 104)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__3_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "attrKind"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__4 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__12_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__13_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__14_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__5_value_aux_2),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(32, 164, 20, 104, 12, 221, 204, 110)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__5 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__12_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__13_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__6_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__14_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__20_value),LEAN_SCALAR_PTR_LITERAL(199, 36, 31, 135, 78, 131, 139, 152)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__6 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__2(uint8_t, uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__3(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__3___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "choice"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(59, 66, 148, 42, 181, 100, 85, 166)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "commandWith_weak_namespace__"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__12_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(177, 181, 244, 12, 1, 14, 170, 235)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__5_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(127, 136, 203, 157, 163, 78, 111, 224)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "with_weak_namespace"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "attribute"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__12_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__13_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__8_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__8_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(79, 30, 18, 84, 71, 173, 185, 159)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__9;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__10_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__11;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "withWeakNamespace"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__12_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__14_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__13_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__14_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__14_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(9, 218, 49, 83, 69, 150, 121, 239)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=>"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__15_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "notation"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__12_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__18_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__18_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__13_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__18_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__18_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__18_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(13, 34, 53, 7, 182, 20, 8, 182)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "mixfix"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__12_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__20_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__20_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__13_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__20_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__20_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__20_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__19_value),LEAN_SCALAR_PTR_LITERAL(1, 31, 80, 86, 44, 46, 155, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__20_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "prefix"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__12_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__22_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__22_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__13_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__22_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__22_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__22_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(223, 255, 86, 177, 195, 168, 212, 163)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__22_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "infix"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__24_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__12_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__24_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__24_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__13_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__24_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__24_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__24_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__23_value),LEAN_SCALAR_PTR_LITERAL(8, 202, 116, 85, 196, 237, 101, 141)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "infixl"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__26_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__12_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__26_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__26_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__13_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__26_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__26_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__26_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__25_value),LEAN_SCALAR_PTR_LITERAL(118, 176, 144, 146, 48, 231, 100, 173)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__26_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "infixr"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__28_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__12_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__28_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__28_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__13_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__28_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__28_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__28_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__27_value),LEAN_SCALAR_PTR_LITERAL(9, 7, 27, 92, 157, 7, 198, 225)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__28_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "postfix"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__30_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__12_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__30_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__30_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_scopedNS___closed__13_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__30_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__30_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__30_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__29_value),LEAN_SCALAR_PTR_LITERAL(97, 175, 134, 52, 144, 48, 141, 10)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__30_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__0(lean_object* v_00___87_){
_start:
{
lean_object* v___x_88_; 
v___x_88_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__0___closed__0));
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__1(lean_object* v_x_89_){
_start:
{
lean_object* v___x_90_; lean_object* v___x_91_; 
v___x_90_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__0___closed__0));
v___x_91_ = lean_array_push(v___x_90_, v_x_89_);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0(lean_object* v___x_112_, size_t v_sz_113_, size_t v_i_114_, lean_object* v_bs_115_){
_start:
{
uint8_t v___x_116_; 
v___x_116_ = lean_usize_dec_lt(v_i_114_, v_sz_113_);
if (v___x_116_ == 0)
{
lean_dec(v___x_112_);
return v_bs_115_;
}
else
{
lean_object* v___x_117_; lean_object* v_v_118_; lean_object* v___x_119_; lean_object* v_bs_x27_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; size_t v___x_130_; size_t v___x_131_; lean_object* v___x_132_; 
v___x_117_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__1));
v_v_118_ = lean_array_uget(v_bs_115_, v_i_114_);
v___x_119_ = lean_unsigned_to_nat(0u);
v_bs_x27_120_ = lean_array_uset(v_bs_115_, v_i_114_, v___x_119_);
v___x_121_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__3));
v___x_122_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__5));
v___x_123_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_scopedNS___closed__20));
v___x_124_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__6));
lean_inc_n(v___x_112_, 5);
v___x_125_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_125_, 0, v___x_112_);
lean_ctor_set(v___x_125_, 1, v___x_123_);
v___x_126_ = l_Lean_Syntax_node1(v___x_112_, v___x_124_, v___x_125_);
v___x_127_ = l_Lean_Syntax_node1(v___x_112_, v___x_117_, v___x_126_);
v___x_128_ = l_Lean_Syntax_node1(v___x_112_, v___x_122_, v___x_127_);
v___x_129_ = l_Lean_Syntax_node2(v___x_112_, v___x_121_, v___x_128_, v_v_118_);
v___x_130_ = ((size_t)1ULL);
v___x_131_ = lean_usize_add(v_i_114_, v___x_130_);
v___x_132_ = lean_array_uset(v_bs_x27_120_, v_i_114_, v___x_129_);
v_i_114_ = v___x_131_;
v_bs_115_ = v___x_132_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___boxed(lean_object* v___x_134_, lean_object* v_sz_135_, lean_object* v_i_136_, lean_object* v_bs_137_){
_start:
{
size_t v_sz_boxed_138_; size_t v_i_boxed_139_; lean_object* v_res_140_; 
v_sz_boxed_138_ = lean_unbox_usize(v_sz_135_);
lean_dec(v_sz_135_);
v_i_boxed_139_ = lean_unbox_usize(v_i_136_);
lean_dec(v_i_136_);
v_res_140_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0(v___x_134_, v_sz_boxed_138_, v_i_boxed_139_, v_bs_137_);
return v_res_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__2(uint8_t v___x_141_, uint8_t v___x_142_, lean_object* v_as_143_, size_t v_i_144_, size_t v_stop_145_, lean_object* v_b_146_){
_start:
{
lean_object* v___y_148_; uint8_t v___x_152_; 
v___x_152_ = lean_usize_dec_eq(v_i_144_, v_stop_145_);
if (v___x_152_ == 0)
{
lean_object* v_fst_153_; uint8_t v___x_154_; 
v_fst_153_ = lean_ctor_get(v_b_146_, 0);
v___x_154_ = lean_unbox(v_fst_153_);
if (v___x_154_ == 0)
{
lean_object* v_snd_155_; lean_object* v___x_157_; uint8_t v_isShared_158_; uint8_t v_isSharedCheck_163_; 
v_snd_155_ = lean_ctor_get(v_b_146_, 1);
v_isSharedCheck_163_ = !lean_is_exclusive(v_b_146_);
if (v_isSharedCheck_163_ == 0)
{
lean_object* v_unused_164_; 
v_unused_164_ = lean_ctor_get(v_b_146_, 0);
lean_dec(v_unused_164_);
v___x_157_ = v_b_146_;
v_isShared_158_ = v_isSharedCheck_163_;
goto v_resetjp_156_;
}
else
{
lean_inc(v_snd_155_);
lean_dec(v_b_146_);
v___x_157_ = lean_box(0);
v_isShared_158_ = v_isSharedCheck_163_;
goto v_resetjp_156_;
}
v_resetjp_156_:
{
lean_object* v___x_159_; lean_object* v___x_161_; 
v___x_159_ = lean_box(v___x_141_);
if (v_isShared_158_ == 0)
{
lean_ctor_set(v___x_157_, 0, v___x_159_);
v___x_161_ = v___x_157_;
goto v_reusejp_160_;
}
else
{
lean_object* v_reuseFailAlloc_162_; 
v_reuseFailAlloc_162_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_162_, 0, v___x_159_);
lean_ctor_set(v_reuseFailAlloc_162_, 1, v_snd_155_);
v___x_161_ = v_reuseFailAlloc_162_;
goto v_reusejp_160_;
}
v_reusejp_160_:
{
v___y_148_ = v___x_161_;
goto v___jp_147_;
}
}
}
else
{
lean_object* v_snd_165_; lean_object* v___x_167_; uint8_t v_isShared_168_; uint8_t v_isSharedCheck_175_; 
v_snd_165_ = lean_ctor_get(v_b_146_, 1);
v_isSharedCheck_175_ = !lean_is_exclusive(v_b_146_);
if (v_isSharedCheck_175_ == 0)
{
lean_object* v_unused_176_; 
v_unused_176_ = lean_ctor_get(v_b_146_, 0);
lean_dec(v_unused_176_);
v___x_167_ = v_b_146_;
v_isShared_168_ = v_isSharedCheck_175_;
goto v_resetjp_166_;
}
else
{
lean_inc(v_snd_165_);
lean_dec(v_b_146_);
v___x_167_ = lean_box(0);
v_isShared_168_ = v_isSharedCheck_175_;
goto v_resetjp_166_;
}
v_resetjp_166_:
{
lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_173_; 
v___x_169_ = lean_array_uget_borrowed(v_as_143_, v_i_144_);
lean_inc(v___x_169_);
v___x_170_ = lean_array_push(v_snd_165_, v___x_169_);
v___x_171_ = lean_box(v___x_142_);
if (v_isShared_168_ == 0)
{
lean_ctor_set(v___x_167_, 1, v___x_170_);
lean_ctor_set(v___x_167_, 0, v___x_171_);
v___x_173_ = v___x_167_;
goto v_reusejp_172_;
}
else
{
lean_object* v_reuseFailAlloc_174_; 
v_reuseFailAlloc_174_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_174_, 0, v___x_171_);
lean_ctor_set(v_reuseFailAlloc_174_, 1, v___x_170_);
v___x_173_ = v_reuseFailAlloc_174_;
goto v_reusejp_172_;
}
v_reusejp_172_:
{
v___y_148_ = v___x_173_;
goto v___jp_147_;
}
}
}
}
else
{
return v_b_146_;
}
v___jp_147_:
{
size_t v___x_149_; size_t v___x_150_; 
v___x_149_ = ((size_t)1ULL);
v___x_150_ = lean_usize_add(v_i_144_, v___x_149_);
v_i_144_ = v___x_150_;
v_b_146_ = v___y_148_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__2___boxed(lean_object* v___x_177_, lean_object* v___x_178_, lean_object* v_as_179_, lean_object* v_i_180_, lean_object* v_stop_181_, lean_object* v_b_182_){
_start:
{
uint8_t v___x_79100__boxed_183_; uint8_t v___x_79101__boxed_184_; size_t v_i_boxed_185_; size_t v_stop_boxed_186_; lean_object* v_res_187_; 
v___x_79100__boxed_183_ = lean_unbox(v___x_177_);
v___x_79101__boxed_184_ = lean_unbox(v___x_178_);
v_i_boxed_185_ = lean_unbox_usize(v_i_180_);
lean_dec(v_i_180_);
v_stop_boxed_186_ = lean_unbox_usize(v_stop_181_);
lean_dec(v_stop_181_);
v_res_187_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__2(v___x_79100__boxed_183_, v___x_79101__boxed_184_, v_as_179_, v_i_boxed_185_, v_stop_boxed_186_, v_b_182_);
lean_dec_ref(v_as_179_);
return v_res_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__1(size_t v_sz_188_, size_t v_i_189_, lean_object* v_bs_190_){
_start:
{
uint8_t v___x_191_; 
v___x_191_ = lean_usize_dec_lt(v_i_189_, v_sz_188_);
if (v___x_191_ == 0)
{
lean_object* v___x_192_; 
v___x_192_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_192_, 0, v_bs_190_);
return v___x_192_;
}
else
{
lean_object* v_v_193_; lean_object* v___x_194_; uint8_t v___x_195_; 
v_v_193_ = lean_array_uget(v_bs_190_, v_i_189_);
v___x_194_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__3));
lean_inc(v_v_193_);
v___x_195_ = l_Lean_Syntax_isOfKind(v_v_193_, v___x_194_);
if (v___x_195_ == 0)
{
lean_object* v___x_196_; 
lean_dec(v_v_193_);
lean_dec_ref(v_bs_190_);
v___x_196_ = lean_box(0);
return v___x_196_;
}
else
{
lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; uint8_t v___x_200_; 
v___x_197_ = lean_unsigned_to_nat(0u);
v___x_198_ = l_Lean_Syntax_getArg(v_v_193_, v___x_197_);
v___x_199_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__5));
lean_inc(v___x_198_);
v___x_200_ = l_Lean_Syntax_isOfKind(v___x_198_, v___x_199_);
if (v___x_200_ == 0)
{
lean_object* v___x_201_; 
lean_dec(v___x_198_);
lean_dec(v_v_193_);
lean_dec_ref(v_bs_190_);
v___x_201_ = lean_box(0);
return v___x_201_;
}
else
{
lean_object* v___x_202_; uint8_t v___x_203_; 
v___x_202_ = l_Lean_Syntax_getArg(v___x_198_, v___x_197_);
lean_dec(v___x_198_);
v___x_203_ = l_Lean_Syntax_matchesNull(v___x_202_, v___x_197_);
if (v___x_203_ == 0)
{
lean_object* v___x_204_; 
lean_dec(v_v_193_);
lean_dec_ref(v_bs_190_);
v___x_204_ = lean_box(0);
return v___x_204_;
}
else
{
lean_object* v___x_205_; lean_object* v_bs_x27_206_; lean_object* v_attr_207_; size_t v___x_208_; size_t v___x_209_; lean_object* v___x_210_; 
v___x_205_ = lean_unsigned_to_nat(1u);
v_bs_x27_206_ = lean_array_uset(v_bs_190_, v_i_189_, v___x_197_);
v_attr_207_ = l_Lean_Syntax_getArg(v_v_193_, v___x_205_);
lean_dec(v_v_193_);
v___x_208_ = ((size_t)1ULL);
v___x_209_ = lean_usize_add(v_i_189_, v___x_208_);
v___x_210_ = lean_array_uset(v_bs_x27_206_, v_i_189_, v_attr_207_);
v_i_189_ = v___x_209_;
v_bs_190_ = v___x_210_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__1___boxed(lean_object* v_sz_212_, lean_object* v_i_213_, lean_object* v_bs_214_){
_start:
{
size_t v_sz_boxed_215_; size_t v_i_boxed_216_; lean_object* v_res_217_; 
v_sz_boxed_215_ = lean_unbox_usize(v_sz_212_);
lean_dec(v_sz_212_);
v_i_boxed_216_ = lean_unbox_usize(v_i_213_);
lean_dec(v_i_213_);
v_res_217_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__1(v_sz_boxed_215_, v_i_boxed_216_, v_bs_214_);
return v_res_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__3(size_t v_sz_218_, size_t v_i_219_, lean_object* v_bs_220_){
_start:
{
uint8_t v___x_221_; 
v___x_221_ = lean_usize_dec_lt(v_i_219_, v_sz_218_);
if (v___x_221_ == 0)
{
lean_object* v___x_222_; 
v___x_222_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_222_, 0, v_bs_220_);
return v___x_222_;
}
else
{
lean_object* v___x_223_; lean_object* v_v_224_; lean_object* v___x_225_; uint8_t v___x_226_; 
v___x_223_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__5));
v_v_224_ = lean_array_uget(v_bs_220_, v_i_219_);
v___x_225_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__3));
lean_inc(v_v_224_);
v___x_226_ = l_Lean_Syntax_isOfKind(v_v_224_, v___x_225_);
if (v___x_226_ == 0)
{
lean_object* v___x_227_; 
lean_dec(v_v_224_);
lean_dec_ref(v_bs_220_);
v___x_227_ = lean_box(0);
return v___x_227_;
}
else
{
lean_object* v___x_228_; lean_object* v___x_229_; uint8_t v___x_230_; 
v___x_228_ = lean_unsigned_to_nat(0u);
v___x_229_ = l_Lean_Syntax_getArg(v_v_224_, v___x_228_);
lean_inc(v___x_229_);
v___x_230_ = l_Lean_Syntax_isOfKind(v___x_229_, v___x_223_);
if (v___x_230_ == 0)
{
lean_object* v___x_231_; 
lean_dec(v___x_229_);
lean_dec(v_v_224_);
lean_dec_ref(v_bs_220_);
v___x_231_ = lean_box(0);
return v___x_231_;
}
else
{
lean_object* v___x_232_; uint8_t v___x_233_; 
v___x_232_ = l_Lean_Syntax_getArg(v___x_229_, v___x_228_);
lean_dec(v___x_229_);
v___x_233_ = l_Lean_Syntax_matchesNull(v___x_232_, v___x_228_);
if (v___x_233_ == 0)
{
lean_object* v___x_234_; 
lean_dec(v_v_224_);
lean_dec_ref(v_bs_220_);
v___x_234_ = lean_box(0);
return v___x_234_;
}
else
{
lean_object* v___x_235_; lean_object* v_bs_x27_236_; lean_object* v_attr_237_; size_t v___x_238_; size_t v___x_239_; lean_object* v___x_240_; 
v___x_235_ = lean_unsigned_to_nat(1u);
v_bs_x27_236_ = lean_array_uset(v_bs_220_, v_i_219_, v___x_228_);
v_attr_237_ = l_Lean_Syntax_getArg(v_v_224_, v___x_235_);
lean_dec(v_v_224_);
v___x_238_ = ((size_t)1ULL);
v___x_239_ = lean_usize_add(v_i_219_, v___x_238_);
v___x_240_ = lean_array_uset(v_bs_x27_236_, v_i_219_, v_attr_237_);
v_i_219_ = v___x_239_;
v_bs_220_ = v___x_240_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__3___boxed(lean_object* v_sz_242_, lean_object* v_i_243_, lean_object* v_bs_244_){
_start:
{
size_t v_sz_boxed_245_; size_t v_i_boxed_246_; lean_object* v_res_247_; 
v_sz_boxed_245_ = lean_unbox_usize(v_sz_242_);
lean_dec(v_sz_242_);
v_i_boxed_246_ = lean_unbox_usize(v_i_243_);
lean_dec(v_i_243_);
v_res_247_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__3(v_sz_boxed_245_, v_i_boxed_246_, v_bs_244_);
return v_res_247_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__9(void){
_start:
{
lean_object* v___x_266_; 
v___x_266_ = l_Array_mkArray0(lean_box(0));
return v___x_266_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__11(void){
_start:
{
lean_object* v___x_268_; lean_object* v___x_269_; 
v___x_268_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__10));
v___x_269_ = l_Lean_mkAtom(v___x_268_);
return v___x_269_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16(void){
_start:
{
lean_object* v___x_278_; lean_object* v___x_279_; 
v___x_278_ = lean_box(0);
v___x_279_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__0(v___x_278_);
return v___x_279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1(lean_object* v_x_322_, lean_object* v_a_323_, lean_object* v_a_324_){
_start:
{
lean_object* v___y_326_; lean_object* v_ns_330_; lean_object* v_attr_331_; lean_object* v_ids_332_; lean_object* v___y_333_; lean_object* v___y_334_; lean_object* v___y_371_; lean_object* v___y_372_; lean_object* v___y_373_; lean_object* v___y_374_; lean_object* v___y_375_; lean_object* v___y_376_; lean_object* v___y_384_; lean_object* v___y_385_; lean_object* v___y_386_; lean_object* v___y_387_; lean_object* v___y_388_; lean_object* v___y_389_; lean_object* v___y_397_; lean_object* v___y_398_; lean_object* v___y_399_; lean_object* v___y_400_; lean_object* v___y_401_; lean_object* v___y_402_; lean_object* v___y_410_; lean_object* v___y_411_; lean_object* v___y_412_; lean_object* v___y_413_; lean_object* v___y_414_; lean_object* v___y_415_; lean_object* v___y_423_; lean_object* v___y_424_; lean_object* v___y_425_; lean_object* v___y_426_; lean_object* v___y_427_; lean_object* v___y_428_; lean_object* v___y_436_; lean_object* v___y_437_; lean_object* v___y_438_; lean_object* v___y_439_; lean_object* v___y_440_; lean_object* v___y_441_; lean_object* v___y_449_; lean_object* v___y_450_; lean_object* v___y_451_; lean_object* v___y_452_; lean_object* v___y_453_; lean_object* v___y_454_; lean_object* v___y_455_; lean_object* v___y_456_; lean_object* v___y_457_; lean_object* v___y_458_; lean_object* v___y_459_; lean_object* v___y_460_; lean_object* v___y_461_; lean_object* v___y_462_; lean_object* v___y_463_; lean_object* v___y_464_; lean_object* v___y_465_; lean_object* v___y_466_; lean_object* v___y_467_; lean_object* v___y_468_; lean_object* v___y_469_; lean_object* v___y_494_; lean_object* v___y_495_; lean_object* v___y_496_; lean_object* v___y_497_; lean_object* v___y_498_; lean_object* v___y_499_; lean_object* v___y_500_; lean_object* v___y_501_; lean_object* v___y_502_; lean_object* v___y_503_; lean_object* v___y_504_; lean_object* v___y_505_; lean_object* v___y_506_; lean_object* v___y_507_; lean_object* v___y_508_; lean_object* v___y_509_; lean_object* v___y_510_; lean_object* v___y_511_; lean_object* v___y_512_; lean_object* v___y_513_; lean_object* v___y_514_; lean_object* v___y_539_; lean_object* v___y_540_; lean_object* v___y_541_; lean_object* v___y_542_; lean_object* v___y_543_; lean_object* v___y_544_; lean_object* v___y_545_; lean_object* v___y_546_; lean_object* v___y_547_; lean_object* v___y_548_; lean_object* v___y_549_; lean_object* v___y_550_; lean_object* v___y_551_; lean_object* v___y_552_; lean_object* v___y_553_; lean_object* v___y_554_; lean_object* v___y_555_; lean_object* v___y_556_; lean_object* v___y_557_; lean_object* v___y_558_; lean_object* v___y_559_; lean_object* v___y_584_; lean_object* v___y_585_; lean_object* v___y_586_; lean_object* v___y_587_; lean_object* v___y_588_; lean_object* v___y_589_; lean_object* v___y_590_; lean_object* v___y_591_; lean_object* v___y_592_; lean_object* v___y_593_; lean_object* v___y_594_; lean_object* v___y_595_; lean_object* v___y_596_; lean_object* v___y_597_; lean_object* v___y_598_; lean_object* v___y_599_; lean_object* v___y_600_; lean_object* v___y_601_; lean_object* v___y_602_; lean_object* v___y_603_; lean_object* v___y_604_; lean_object* v___y_629_; lean_object* v___y_630_; lean_object* v___y_631_; lean_object* v___y_632_; lean_object* v___y_633_; lean_object* v___y_634_; lean_object* v___y_635_; lean_object* v___y_636_; lean_object* v___y_637_; lean_object* v___y_638_; lean_object* v___y_639_; lean_object* v___y_640_; lean_object* v___y_641_; lean_object* v___y_642_; lean_object* v___y_643_; lean_object* v___y_644_; lean_object* v___y_645_; lean_object* v___y_646_; lean_object* v___y_647_; lean_object* v___y_648_; lean_object* v___y_649_; lean_object* v___y_674_; lean_object* v___y_675_; lean_object* v___y_676_; lean_object* v___y_677_; lean_object* v___y_678_; lean_object* v___y_679_; lean_object* v___y_687_; lean_object* v___y_688_; lean_object* v___y_689_; lean_object* v___y_690_; lean_object* v___y_691_; lean_object* v___y_692_; lean_object* v___y_700_; lean_object* v___y_701_; lean_object* v___y_702_; lean_object* v___y_703_; lean_object* v___y_704_; lean_object* v___y_705_; lean_object* v___y_713_; lean_object* v___y_714_; lean_object* v___y_715_; lean_object* v___y_716_; lean_object* v___y_717_; lean_object* v___y_718_; lean_object* v___y_726_; lean_object* v___y_727_; lean_object* v___y_728_; lean_object* v___y_729_; lean_object* v___y_730_; lean_object* v___y_731_; lean_object* v___y_732_; lean_object* v___y_733_; lean_object* v___y_734_; lean_object* v___y_735_; lean_object* v___y_736_; lean_object* v___y_737_; lean_object* v___y_738_; lean_object* v___y_739_; lean_object* v___y_740_; lean_object* v___y_741_; lean_object* v___y_742_; lean_object* v___y_743_; lean_object* v___y_744_; lean_object* v___y_745_; lean_object* v___y_746_; lean_object* v___x_772_; uint8_t v___x_773_; 
v___x_772_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_scopedNS___closed__3));
lean_inc(v_x_322_);
v___x_773_ = l_Lean_Syntax_isOfKind(v_x_322_, v___x_772_);
if (v___x_773_ == 0)
{
lean_object* v___x_774_; lean_object* v___x_775_; 
lean_dec(v_x_322_);
v___x_774_ = lean_box(1);
v___x_775_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_775_, 0, v___x_774_);
lean_ctor_set(v___x_775_, 1, v_a_324_);
return v___x_775_;
}
else
{
lean_object* v___y_777_; lean_object* v___y_778_; lean_object* v___y_779_; lean_object* v___y_780_; lean_object* v___y_781_; lean_object* v___y_782_; lean_object* v___y_783_; lean_object* v___y_784_; lean_object* v___y_785_; lean_object* v___y_786_; lean_object* v___y_787_; lean_object* v___y_788_; lean_object* v___y_789_; lean_object* v___y_790_; lean_object* v___y_791_; lean_object* v___y_792_; lean_object* v___y_793_; lean_object* v___y_794_; lean_object* v___y_795_; lean_object* v___y_796_; lean_object* v___y_797_; lean_object* v___y_804_; lean_object* v___y_805_; lean_object* v___y_806_; lean_object* v___y_807_; lean_object* v___y_808_; lean_object* v___y_809_; lean_object* v___y_810_; lean_object* v___y_811_; lean_object* v___y_812_; lean_object* v___y_813_; lean_object* v___y_814_; lean_object* v___y_815_; lean_object* v___y_816_; lean_object* v___y_817_; lean_object* v___y_818_; lean_object* v___y_819_; lean_object* v___y_820_; lean_object* v___y_821_; lean_object* v___y_822_; lean_object* v___y_823_; lean_object* v___y_824_; lean_object* v___y_831_; lean_object* v___y_832_; lean_object* v___y_833_; lean_object* v___y_834_; lean_object* v___y_835_; lean_object* v___y_836_; lean_object* v___y_837_; lean_object* v___y_838_; lean_object* v___y_839_; lean_object* v___y_840_; lean_object* v___y_841_; lean_object* v___y_842_; lean_object* v___y_843_; lean_object* v___y_844_; lean_object* v___y_845_; lean_object* v___y_846_; lean_object* v___y_847_; lean_object* v___y_848_; lean_object* v___y_849_; lean_object* v___y_850_; lean_object* v___y_851_; lean_object* v___y_852_; lean_object* v___y_866_; lean_object* v___y_867_; lean_object* v___y_868_; lean_object* v___y_869_; lean_object* v___y_870_; lean_object* v___y_871_; lean_object* v___y_872_; lean_object* v___y_873_; lean_object* v___y_874_; lean_object* v___y_875_; lean_object* v___y_876_; lean_object* v___y_877_; lean_object* v___y_878_; lean_object* v___y_879_; lean_object* v___y_880_; lean_object* v___y_881_; lean_object* v___y_882_; lean_object* v___y_883_; lean_object* v___y_884_; lean_object* v___y_885_; lean_object* v___y_886_; lean_object* v___y_887_; lean_object* v___y_894_; lean_object* v___y_895_; lean_object* v___y_896_; lean_object* v___y_897_; lean_object* v___y_898_; lean_object* v___y_899_; lean_object* v___y_900_; lean_object* v___y_901_; lean_object* v___y_902_; lean_object* v___y_903_; lean_object* v___y_904_; lean_object* v___y_905_; lean_object* v___y_906_; lean_object* v___y_907_; lean_object* v___y_908_; lean_object* v___y_909_; lean_object* v___y_910_; lean_object* v___y_911_; lean_object* v___y_912_; lean_object* v___y_913_; lean_object* v___y_914_; lean_object* v___y_921_; lean_object* v___y_922_; lean_object* v___y_923_; lean_object* v___y_924_; lean_object* v___y_925_; lean_object* v___y_926_; lean_object* v___y_927_; lean_object* v___y_928_; lean_object* v___y_929_; lean_object* v___y_930_; lean_object* v___y_931_; lean_object* v___y_932_; lean_object* v___y_933_; lean_object* v___y_934_; lean_object* v___y_935_; lean_object* v___y_936_; lean_object* v___y_937_; lean_object* v___y_938_; lean_object* v___y_939_; lean_object* v___y_940_; lean_object* v___y_941_; lean_object* v___y_942_; lean_object* v___y_955_; lean_object* v___y_956_; lean_object* v___y_957_; lean_object* v___y_958_; lean_object* v___y_959_; lean_object* v___y_960_; lean_object* v___y_961_; lean_object* v___y_962_; lean_object* v___y_963_; lean_object* v___y_964_; lean_object* v___y_965_; lean_object* v___y_966_; lean_object* v___y_967_; lean_object* v___y_968_; lean_object* v___y_969_; lean_object* v___y_970_; lean_object* v___y_971_; lean_object* v___y_972_; lean_object* v___y_973_; lean_object* v___y_974_; lean_object* v___y_975_; lean_object* v___y_976_; lean_object* v___y_983_; lean_object* v___y_984_; lean_object* v___y_985_; lean_object* v___y_986_; lean_object* v___y_987_; lean_object* v___y_988_; lean_object* v___y_989_; lean_object* v___y_990_; lean_object* v___y_991_; lean_object* v___y_992_; lean_object* v___y_993_; lean_object* v___y_994_; lean_object* v___y_995_; lean_object* v___y_996_; lean_object* v___y_997_; lean_object* v___y_998_; lean_object* v___y_999_; lean_object* v___y_1000_; lean_object* v___y_1001_; lean_object* v___y_1002_; lean_object* v___y_1003_; lean_object* v___y_1010_; lean_object* v___y_1011_; lean_object* v___y_1012_; lean_object* v___y_1013_; lean_object* v___y_1014_; lean_object* v___y_1015_; lean_object* v___y_1016_; lean_object* v___y_1017_; lean_object* v___y_1018_; lean_object* v___y_1019_; lean_object* v___y_1020_; lean_object* v___y_1021_; lean_object* v___y_1022_; lean_object* v___y_1023_; lean_object* v___y_1024_; lean_object* v___y_1025_; lean_object* v___y_1026_; lean_object* v___y_1027_; lean_object* v___y_1028_; lean_object* v___y_1029_; lean_object* v___y_1030_; lean_object* v___y_1031_; lean_object* v___y_1044_; lean_object* v___y_1045_; lean_object* v___y_1046_; lean_object* v___y_1047_; lean_object* v___y_1048_; lean_object* v___y_1049_; lean_object* v___y_1050_; lean_object* v___y_1051_; lean_object* v___y_1052_; lean_object* v___y_1053_; lean_object* v___y_1054_; lean_object* v___y_1055_; lean_object* v___y_1056_; lean_object* v___y_1057_; lean_object* v___y_1058_; lean_object* v___y_1059_; lean_object* v___y_1060_; lean_object* v___y_1061_; lean_object* v___y_1062_; lean_object* v___y_1063_; lean_object* v___y_1064_; lean_object* v___y_1065_; lean_object* v___y_1072_; lean_object* v___y_1073_; lean_object* v___y_1074_; lean_object* v___y_1075_; lean_object* v___y_1076_; lean_object* v___y_1077_; lean_object* v___y_1078_; lean_object* v___y_1079_; lean_object* v___y_1080_; lean_object* v___y_1081_; lean_object* v___y_1082_; lean_object* v___y_1083_; lean_object* v___y_1084_; lean_object* v___y_1085_; lean_object* v___y_1086_; lean_object* v___y_1087_; lean_object* v___y_1088_; lean_object* v___y_1089_; lean_object* v___y_1090_; lean_object* v___y_1091_; lean_object* v___y_1092_; lean_object* v___y_1099_; lean_object* v___y_1100_; lean_object* v___y_1101_; lean_object* v___y_1102_; lean_object* v___y_1103_; lean_object* v___y_1104_; lean_object* v___y_1105_; lean_object* v___y_1106_; lean_object* v___y_1107_; lean_object* v___y_1108_; lean_object* v___y_1109_; lean_object* v___y_1110_; lean_object* v___y_1111_; lean_object* v___y_1112_; lean_object* v___y_1113_; lean_object* v___y_1114_; lean_object* v___y_1115_; lean_object* v___y_1116_; lean_object* v___y_1117_; lean_object* v___y_1118_; lean_object* v___y_1119_; lean_object* v___y_1120_; lean_object* v___y_1133_; lean_object* v___y_1134_; lean_object* v___y_1135_; lean_object* v___y_1136_; lean_object* v___y_1137_; lean_object* v___y_1138_; lean_object* v___y_1139_; lean_object* v___y_1140_; lean_object* v___y_1141_; lean_object* v___y_1142_; lean_object* v___y_1143_; lean_object* v___y_1144_; lean_object* v___y_1145_; lean_object* v___y_1146_; lean_object* v___y_1147_; lean_object* v___y_1148_; lean_object* v___y_1149_; lean_object* v___y_1150_; lean_object* v___y_1151_; lean_object* v___y_1152_; lean_object* v___y_1153_; lean_object* v___y_1154_; lean_object* v___y_1161_; lean_object* v___y_1162_; lean_object* v___y_1163_; lean_object* v___y_1164_; lean_object* v___y_1165_; lean_object* v___y_1166_; lean_object* v___y_1167_; lean_object* v___y_1168_; lean_object* v___y_1169_; lean_object* v___y_1170_; lean_object* v___y_1171_; lean_object* v___y_1172_; lean_object* v___y_1173_; lean_object* v___y_1174_; lean_object* v___y_1175_; lean_object* v___y_1176_; lean_object* v___y_1177_; lean_object* v___y_1178_; lean_object* v___y_1179_; lean_object* v___y_1180_; lean_object* v___y_1181_; lean_object* v___y_1188_; lean_object* v___y_1189_; lean_object* v___y_1190_; lean_object* v___y_1191_; lean_object* v___y_1192_; lean_object* v___y_1193_; lean_object* v___y_1194_; lean_object* v___y_1195_; lean_object* v___y_1196_; lean_object* v___y_1197_; lean_object* v___y_1198_; lean_object* v___y_1199_; lean_object* v___y_1200_; lean_object* v___y_1201_; lean_object* v___y_1202_; lean_object* v___y_1203_; lean_object* v___y_1204_; lean_object* v___y_1205_; lean_object* v___y_1206_; lean_object* v___y_1207_; lean_object* v___y_1208_; lean_object* v___y_1209_; lean_object* v___y_1222_; lean_object* v___y_1223_; lean_object* v___y_1224_; lean_object* v___y_1225_; lean_object* v___y_1226_; lean_object* v___y_1227_; lean_object* v___y_1228_; lean_object* v___y_1229_; lean_object* v___y_1230_; lean_object* v___y_1231_; lean_object* v___y_1232_; lean_object* v___y_1233_; lean_object* v___y_1234_; lean_object* v___y_1235_; lean_object* v___y_1236_; lean_object* v___y_1237_; lean_object* v___y_1238_; lean_object* v___y_1239_; lean_object* v___y_1240_; lean_object* v___y_1241_; lean_object* v___y_1242_; lean_object* v___y_1243_; lean_object* v___y_1250_; lean_object* v___y_1251_; lean_object* v___y_1252_; lean_object* v___y_1253_; lean_object* v___y_1254_; lean_object* v___y_1255_; lean_object* v___y_1256_; lean_object* v___y_1257_; lean_object* v___y_1258_; lean_object* v___y_1259_; lean_object* v___y_1260_; lean_object* v___y_1261_; lean_object* v___y_1262_; lean_object* v___y_1263_; lean_object* v___y_1264_; lean_object* v___y_1265_; lean_object* v___y_1266_; lean_object* v___y_1267_; lean_object* v___y_1268_; lean_object* v___y_1269_; lean_object* v___y_1270_; lean_object* v___y_1277_; lean_object* v___y_1278_; lean_object* v___y_1279_; lean_object* v___y_1280_; lean_object* v___y_1281_; lean_object* v___y_1282_; lean_object* v___y_1283_; lean_object* v___y_1284_; lean_object* v___y_1285_; lean_object* v___y_1286_; lean_object* v___y_1287_; lean_object* v___y_1288_; lean_object* v___y_1289_; lean_object* v___y_1290_; lean_object* v___y_1291_; lean_object* v___y_1292_; lean_object* v___y_1293_; lean_object* v___y_1294_; lean_object* v___y_1295_; lean_object* v___y_1296_; lean_object* v___y_1297_; lean_object* v___y_1298_; lean_object* v___y_1311_; lean_object* v___y_1312_; lean_object* v___y_1313_; lean_object* v___y_1314_; lean_object* v___y_1315_; lean_object* v___y_1316_; lean_object* v___y_1317_; lean_object* v___y_1318_; lean_object* v___y_1319_; lean_object* v___y_1320_; lean_object* v___y_1321_; lean_object* v___y_1322_; lean_object* v___y_1323_; lean_object* v___y_1324_; lean_object* v___y_1325_; lean_object* v___y_1326_; lean_object* v___y_1327_; lean_object* v___y_1328_; lean_object* v___y_1329_; lean_object* v___y_1330_; lean_object* v___y_1331_; lean_object* v___y_1332_; lean_object* v___x_1338_; lean_object* v___y_1340_; lean_object* v___y_1341_; lean_object* v___y_1342_; lean_object* v___y_1343_; lean_object* v___y_1344_; lean_object* v___y_1345_; lean_object* v___y_1346_; lean_object* v___y_1347_; lean_object* v___y_1348_; lean_object* v___y_1349_; lean_object* v___y_1350_; lean_object* v___y_1351_; uint8_t v___y_1352_; lean_object* v___y_1353_; lean_object* v___y_1354_; lean_object* v___y_1355_; lean_object* v___y_1356_; lean_object* v___y_1357_; lean_object* v___y_1376_; lean_object* v___y_1377_; lean_object* v___y_1378_; lean_object* v___y_1379_; lean_object* v___y_1380_; lean_object* v___y_1381_; lean_object* v___y_1382_; lean_object* v___y_1383_; lean_object* v___y_1384_; lean_object* v___y_1385_; lean_object* v___y_1386_; lean_object* v___y_1387_; uint8_t v___y_1388_; lean_object* v___y_1389_; lean_object* v___y_1390_; lean_object* v___y_1391_; lean_object* v___y_1392_; lean_object* v___y_1393_; lean_object* v___y_1405_; lean_object* v___y_1406_; lean_object* v___y_1407_; lean_object* v___y_1408_; lean_object* v___y_1409_; lean_object* v___y_1410_; lean_object* v___y_1411_; lean_object* v___y_1412_; lean_object* v___y_1413_; lean_object* v___y_1414_; lean_object* v___y_1415_; lean_object* v___y_1416_; uint8_t v___y_1417_; lean_object* v___y_1418_; lean_object* v___y_1419_; lean_object* v___y_1420_; lean_object* v___y_1421_; lean_object* v___y_1422_; lean_object* v___y_1434_; lean_object* v___y_1435_; lean_object* v___y_1436_; lean_object* v___y_1437_; lean_object* v___y_1438_; lean_object* v___y_1439_; lean_object* v___y_1440_; lean_object* v___y_1441_; lean_object* v___y_1442_; lean_object* v___y_1443_; lean_object* v___y_1444_; uint8_t v___y_1445_; lean_object* v___y_1446_; lean_object* v___y_1447_; lean_object* v___y_1448_; lean_object* v___y_1449_; lean_object* v___y_1450_; lean_object* v___y_1451_; lean_object* v___y_1470_; lean_object* v___y_1471_; lean_object* v___y_1472_; lean_object* v___y_1473_; lean_object* v___y_1474_; lean_object* v___y_1475_; lean_object* v___y_1476_; lean_object* v___y_1477_; lean_object* v___y_1478_; lean_object* v___y_1479_; lean_object* v___y_1480_; uint8_t v___y_1481_; lean_object* v___y_1482_; lean_object* v___y_1483_; lean_object* v___y_1484_; lean_object* v___y_1485_; lean_object* v___y_1486_; lean_object* v___y_1487_; lean_object* v___y_1499_; lean_object* v___y_1500_; lean_object* v___y_1501_; lean_object* v___y_1502_; lean_object* v___y_1503_; lean_object* v___y_1504_; lean_object* v___y_1505_; lean_object* v___y_1506_; lean_object* v___y_1507_; lean_object* v___y_1508_; lean_object* v___y_1509_; uint8_t v___y_1510_; lean_object* v___y_1511_; lean_object* v___y_1512_; lean_object* v___y_1513_; lean_object* v___y_1514_; lean_object* v___y_1515_; lean_object* v___y_1516_; lean_object* v___y_1528_; lean_object* v___y_1529_; lean_object* v___y_1530_; lean_object* v___y_1531_; lean_object* v___y_1532_; lean_object* v___y_1533_; lean_object* v___y_1534_; lean_object* v___y_1535_; lean_object* v___y_1536_; lean_object* v___y_1537_; uint8_t v___y_1538_; lean_object* v___y_1539_; lean_object* v___y_1540_; lean_object* v___y_1541_; lean_object* v___y_1542_; lean_object* v___y_1543_; lean_object* v___y_1544_; lean_object* v___y_1545_; lean_object* v___y_1564_; lean_object* v___y_1565_; lean_object* v___y_1566_; lean_object* v___y_1567_; lean_object* v___y_1568_; lean_object* v___y_1569_; lean_object* v___y_1570_; lean_object* v___y_1571_; lean_object* v___y_1572_; lean_object* v___y_1573_; lean_object* v___y_1574_; uint8_t v___y_1575_; lean_object* v___y_1576_; lean_object* v___y_1577_; lean_object* v___y_1578_; lean_object* v___y_1579_; lean_object* v___y_1580_; lean_object* v___y_1581_; lean_object* v___y_1593_; lean_object* v___y_1594_; lean_object* v___y_1595_; lean_object* v___y_1596_; lean_object* v___y_1597_; lean_object* v___y_1598_; lean_object* v___y_1599_; lean_object* v___y_1600_; lean_object* v___y_1601_; lean_object* v___y_1602_; lean_object* v___y_1603_; uint8_t v___y_1604_; lean_object* v___y_1605_; lean_object* v___y_1606_; lean_object* v___y_1607_; lean_object* v___y_1608_; lean_object* v___y_1609_; lean_object* v___y_1610_; lean_object* v___y_1622_; lean_object* v___y_1623_; lean_object* v___y_1624_; lean_object* v___y_1625_; lean_object* v___y_1626_; uint8_t v___y_1627_; lean_object* v___y_1628_; lean_object* v___y_1629_; lean_object* v___y_1630_; lean_object* v___y_1631_; lean_object* v___y_1632_; lean_object* v___y_1633_; lean_object* v___y_1634_; lean_object* v___y_1635_; lean_object* v___y_1636_; lean_object* v___y_1637_; lean_object* v___y_1638_; lean_object* v___y_1639_; lean_object* v___y_1658_; lean_object* v___y_1659_; lean_object* v___y_1660_; lean_object* v___y_1661_; lean_object* v___y_1662_; uint8_t v___y_1663_; lean_object* v___y_1664_; lean_object* v___y_1665_; lean_object* v___y_1666_; lean_object* v___y_1667_; lean_object* v___y_1668_; lean_object* v___y_1669_; lean_object* v___y_1670_; lean_object* v___y_1671_; lean_object* v___y_1672_; lean_object* v___y_1673_; lean_object* v___y_1674_; lean_object* v___y_1675_; lean_object* v___y_1687_; lean_object* v___y_1688_; lean_object* v___y_1689_; lean_object* v___y_1690_; lean_object* v___y_1691_; lean_object* v___y_1692_; uint8_t v___y_1693_; lean_object* v___y_1694_; lean_object* v___y_1695_; lean_object* v___y_1696_; lean_object* v___y_1697_; lean_object* v___y_1698_; lean_object* v___y_1699_; lean_object* v___y_1700_; lean_object* v___y_1701_; lean_object* v___y_1702_; lean_object* v___y_1703_; lean_object* v___y_1704_; lean_object* v___y_1716_; lean_object* v___y_1717_; lean_object* v___y_1718_; lean_object* v___y_1719_; lean_object* v___y_1720_; lean_object* v___y_1721_; lean_object* v___y_1722_; lean_object* v___y_1723_; lean_object* v___y_1724_; lean_object* v___y_1725_; lean_object* v___y_1726_; lean_object* v___y_1727_; lean_object* v___y_1728_; lean_object* v___y_1729_; uint8_t v___y_1730_; lean_object* v___y_1731_; lean_object* v___y_1732_; lean_object* v___y_1733_; lean_object* v___y_1752_; lean_object* v___y_1753_; lean_object* v___y_1754_; lean_object* v___y_1755_; lean_object* v___y_1756_; lean_object* v___y_1757_; lean_object* v___y_1758_; lean_object* v___y_1759_; lean_object* v___y_1760_; lean_object* v___y_1761_; lean_object* v___y_1762_; lean_object* v___y_1763_; lean_object* v___y_1764_; lean_object* v___y_1765_; uint8_t v___y_1766_; lean_object* v___y_1767_; lean_object* v___y_1768_; lean_object* v___y_1769_; lean_object* v___y_1781_; lean_object* v___y_1782_; lean_object* v___y_1783_; lean_object* v___y_1784_; lean_object* v___y_1785_; lean_object* v___y_1786_; lean_object* v___y_1787_; lean_object* v___y_1788_; lean_object* v___y_1789_; lean_object* v___y_1790_; lean_object* v___y_1791_; lean_object* v___y_1792_; lean_object* v___y_1793_; uint8_t v___y_1794_; lean_object* v___y_1795_; lean_object* v___y_1796_; lean_object* v___y_1797_; lean_object* v___y_1798_; lean_object* v___y_1810_; lean_object* v___y_1811_; lean_object* v___y_1812_; lean_object* v___y_1813_; lean_object* v___y_1814_; lean_object* v___y_1815_; lean_object* v___y_1816_; lean_object* v___y_1817_; lean_object* v___y_1818_; lean_object* v___y_1819_; lean_object* v___y_1820_; lean_object* v___y_1821_; lean_object* v___y_1822_; lean_object* v___y_1823_; lean_object* v___y_1824_; lean_object* v___y_1825_; lean_object* v___y_1826_; lean_object* v___y_1846_; lean_object* v___y_1847_; lean_object* v___y_1848_; lean_object* v___y_1849_; lean_object* v___y_1850_; lean_object* v___y_1851_; lean_object* v___y_1852_; lean_object* v___y_1853_; lean_object* v___y_1854_; lean_object* v___y_1855_; lean_object* v___y_1856_; lean_object* v___y_1857_; lean_object* v___y_1858_; lean_object* v___y_1859_; lean_object* v___y_1860_; lean_object* v___y_1861_; lean_object* v___y_1862_; lean_object* v___y_1874_; lean_object* v___y_1875_; lean_object* v___y_1876_; lean_object* v___y_1877_; lean_object* v___y_1878_; lean_object* v___y_1879_; lean_object* v___y_1880_; lean_object* v___y_1881_; lean_object* v___y_1882_; lean_object* v___y_1883_; lean_object* v___y_1884_; lean_object* v___y_1885_; lean_object* v___y_1886_; lean_object* v___y_1887_; lean_object* v___y_1888_; lean_object* v___y_1889_; lean_object* v___y_1890_; lean_object* v___y_1902_; lean_object* v___y_1903_; lean_object* v___y_1904_; lean_object* v___y_1905_; lean_object* v___y_1906_; lean_object* v___y_1907_; lean_object* v___y_1908_; lean_object* v___y_1909_; lean_object* v___y_1910_; lean_object* v___y_1911_; lean_object* v___y_1912_; lean_object* v___y_1913_; lean_object* v___y_1914_; lean_object* v___y_1915_; lean_object* v___y_1916_; lean_object* v___y_1917_; lean_object* v___y_1918_; lean_object* v___x_1929_; lean_object* v_doc_1931_; lean_object* v___y_1932_; lean_object* v___y_1933_; uint8_t v___x_2280_; 
v___x_1338_ = lean_unsigned_to_nat(0u);
v___x_1929_ = l_Lean_Syntax_getArg(v_x_322_, v___x_1338_);
v___x_2280_ = l_Lean_Syntax_isNone(v___x_1929_);
if (v___x_2280_ == 0)
{
lean_object* v___x_2281_; uint8_t v___x_2282_; 
v___x_2281_ = lean_unsigned_to_nat(1u);
lean_inc(v___x_1929_);
v___x_2282_ = l_Lean_Syntax_matchesNull(v___x_1929_, v___x_2281_);
if (v___x_2282_ == 0)
{
uint8_t v___x_2283_; 
v___x_2283_ = l_Lean_Syntax_matchesNull(v___x_1929_, v___x_1338_);
if (v___x_2283_ == 0)
{
lean_dec(v_x_322_);
v___y_326_ = v_a_324_;
goto v___jp_325_;
}
else
{
lean_object* v___x_2284_; uint8_t v___x_2285_; 
v___x_2284_ = l_Lean_Syntax_getArg(v_x_322_, v___x_2281_);
v___x_2285_ = l_Lean_Syntax_matchesNull(v___x_2284_, v___x_1338_);
if (v___x_2285_ == 0)
{
lean_dec(v_x_322_);
v___y_326_ = v_a_324_;
goto v___jp_325_;
}
else
{
lean_object* v___x_2286_; lean_object* v___x_2287_; lean_object* v___x_2288_; uint8_t v___x_2289_; 
v___x_2286_ = lean_unsigned_to_nat(6u);
v___x_2287_ = l_Lean_Syntax_getArg(v_x_322_, v___x_2286_);
v___x_2288_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__8));
lean_inc(v___x_2287_);
v___x_2289_ = l_Lean_Syntax_isOfKind(v___x_2287_, v___x_2288_);
if (v___x_2289_ == 0)
{
lean_dec(v___x_2287_);
lean_dec(v_x_322_);
v___y_326_ = v_a_324_;
goto v___jp_325_;
}
else
{
lean_object* v___x_2290_; lean_object* v___x_2291_; lean_object* v_ns_2292_; lean_object* v___y_2294_; lean_object* v___x_2301_; lean_object* v___x_2302_; lean_object* v___x_2303_; lean_object* v___x_2304_; uint8_t v___x_2305_; 
v___x_2290_ = lean_unsigned_to_nat(2u);
v___x_2291_ = lean_unsigned_to_nat(4u);
v_ns_2292_ = l_Lean_Syntax_getArg(v_x_322_, v___x_2291_);
lean_dec(v_x_322_);
v___x_2301_ = l_Lean_Syntax_getArg(v___x_2287_, v___x_2290_);
v___x_2302_ = l_Lean_Syntax_getArgs(v___x_2301_);
lean_dec(v___x_2301_);
v___x_2303_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__0___closed__0));
v___x_2304_ = lean_array_get_size(v___x_2302_);
v___x_2305_ = lean_nat_dec_lt(v___x_1338_, v___x_2304_);
if (v___x_2305_ == 0)
{
lean_dec_ref(v___x_2302_);
v___y_2294_ = v___x_2303_;
goto v___jp_2293_;
}
else
{
lean_object* v___x_2306_; lean_object* v___x_2307_; uint8_t v___x_2308_; 
v___x_2306_ = lean_box(v___x_2289_);
v___x_2307_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2307_, 0, v___x_2306_);
lean_ctor_set(v___x_2307_, 1, v___x_2303_);
v___x_2308_ = lean_nat_dec_le(v___x_2304_, v___x_2304_);
if (v___x_2308_ == 0)
{
if (v___x_2305_ == 0)
{
lean_dec_ref_known(v___x_2307_, 2);
lean_dec_ref(v___x_2302_);
v___y_2294_ = v___x_2303_;
goto v___jp_2293_;
}
else
{
size_t v___x_2309_; size_t v___x_2310_; lean_object* v___x_2311_; lean_object* v_snd_2312_; 
v___x_2309_ = ((size_t)0ULL);
v___x_2310_ = lean_usize_of_nat(v___x_2304_);
v___x_2311_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__2(v___x_2289_, v___x_2282_, v___x_2302_, v___x_2309_, v___x_2310_, v___x_2307_);
lean_dec_ref(v___x_2302_);
v_snd_2312_ = lean_ctor_get(v___x_2311_, 1);
lean_inc(v_snd_2312_);
lean_dec_ref(v___x_2311_);
v___y_2294_ = v_snd_2312_;
goto v___jp_2293_;
}
}
else
{
size_t v___x_2313_; size_t v___x_2314_; lean_object* v___x_2315_; lean_object* v_snd_2316_; 
v___x_2313_ = ((size_t)0ULL);
v___x_2314_ = lean_usize_of_nat(v___x_2304_);
v___x_2315_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__2(v___x_2289_, v___x_2282_, v___x_2302_, v___x_2313_, v___x_2314_, v___x_2307_);
lean_dec_ref(v___x_2302_);
v_snd_2316_ = lean_ctor_get(v___x_2315_, 1);
lean_inc(v_snd_2316_);
lean_dec_ref(v___x_2315_);
v___y_2294_ = v_snd_2316_;
goto v___jp_2293_;
}
}
v___jp_2293_:
{
size_t v_sz_2295_; size_t v___x_2296_; lean_object* v___x_2297_; 
v_sz_2295_ = lean_array_size(v___y_2294_);
v___x_2296_ = ((size_t)0ULL);
v___x_2297_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__1(v_sz_2295_, v___x_2296_, v___y_2294_);
if (lean_obj_tag(v___x_2297_) == 0)
{
lean_dec(v_ns_2292_);
lean_dec(v___x_2287_);
v___y_326_ = v_a_324_;
goto v___jp_325_;
}
else
{
lean_object* v_val_2298_; lean_object* v___x_2299_; lean_object* v_ids_2300_; 
v_val_2298_ = lean_ctor_get(v___x_2297_, 0);
lean_inc(v_val_2298_);
lean_dec_ref_known(v___x_2297_, 1);
v___x_2299_ = l_Lean_Syntax_getArg(v___x_2287_, v___x_2291_);
lean_dec(v___x_2287_);
v_ids_2300_ = l_Lean_Syntax_getArgs(v___x_2299_);
lean_dec(v___x_2299_);
v_ns_330_ = v_ns_2292_;
v_attr_331_ = v_val_2298_;
v_ids_332_ = v_ids_2300_;
v___y_333_ = v_a_323_;
v___y_334_ = v_a_324_;
goto v___jp_329_;
}
}
}
}
}
}
else
{
lean_object* v_doc_2317_; lean_object* v___x_2318_; 
v_doc_2317_ = l_Lean_Syntax_getArg(v___x_1929_, v___x_1338_);
v___x_2318_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2318_, 0, v_doc_2317_);
v_doc_1931_ = v___x_2318_;
v___y_1932_ = v_a_323_;
v___y_1933_ = v_a_324_;
goto v___jp_1930_;
}
}
else
{
lean_object* v___x_2319_; 
v___x_2319_ = lean_box(0);
v_doc_1931_ = v___x_2319_;
v___y_1932_ = v_a_323_;
v___y_1933_ = v_a_324_;
goto v___jp_1930_;
}
v___jp_776_:
{
lean_object* v___x_798_; lean_object* v___x_799_; 
lean_inc_ref(v___y_792_);
v___x_798_ = l_Array_append___redArg(v___y_792_, v___y_797_);
lean_dec_ref(v___y_797_);
lean_inc(v___y_783_);
lean_inc(v___y_785_);
v___x_799_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_799_, 0, v___y_785_);
lean_ctor_set(v___x_799_, 1, v___y_783_);
lean_ctor_set(v___x_799_, 2, v___x_798_);
if (lean_obj_tag(v___y_781_) == 0)
{
lean_object* v___x_800_; 
v___x_800_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16);
v___y_726_ = v___y_777_;
v___y_727_ = v___y_778_;
v___y_728_ = v___y_779_;
v___y_729_ = v___y_780_;
v___y_730_ = v___y_782_;
v___y_731_ = v___y_783_;
v___y_732_ = v___y_784_;
v___y_733_ = v___y_785_;
v___y_734_ = v___y_786_;
v___y_735_ = v___x_799_;
v___y_736_ = v___y_787_;
v___y_737_ = v___y_788_;
v___y_738_ = v___y_789_;
v___y_739_ = v___y_791_;
v___y_740_ = v___y_790_;
v___y_741_ = v___y_792_;
v___y_742_ = v___y_794_;
v___y_743_ = v___y_793_;
v___y_744_ = v___y_795_;
v___y_745_ = v___y_796_;
v___y_746_ = v___x_800_;
goto v___jp_725_;
}
else
{
lean_object* v_val_801_; lean_object* v___x_802_; 
v_val_801_ = lean_ctor_get(v___y_781_, 0);
lean_inc(v_val_801_);
lean_dec_ref_known(v___y_781_, 1);
v___x_802_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__1(v_val_801_);
v___y_726_ = v___y_777_;
v___y_727_ = v___y_778_;
v___y_728_ = v___y_779_;
v___y_729_ = v___y_780_;
v___y_730_ = v___y_782_;
v___y_731_ = v___y_783_;
v___y_732_ = v___y_784_;
v___y_733_ = v___y_785_;
v___y_734_ = v___y_786_;
v___y_735_ = v___x_799_;
v___y_736_ = v___y_787_;
v___y_737_ = v___y_788_;
v___y_738_ = v___y_789_;
v___y_739_ = v___y_791_;
v___y_740_ = v___y_790_;
v___y_741_ = v___y_792_;
v___y_742_ = v___y_794_;
v___y_743_ = v___y_793_;
v___y_744_ = v___y_795_;
v___y_745_ = v___y_796_;
v___y_746_ = v___x_802_;
goto v___jp_725_;
}
}
v___jp_803_:
{
lean_object* v___x_825_; lean_object* v___x_826_; 
lean_inc_ref(v___y_820_);
v___x_825_ = l_Array_append___redArg(v___y_820_, v___y_824_);
lean_dec_ref(v___y_824_);
lean_inc(v___y_810_);
lean_inc(v___y_813_);
v___x_826_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_826_, 0, v___y_813_);
lean_ctor_set(v___x_826_, 1, v___y_810_);
lean_ctor_set(v___x_826_, 2, v___x_825_);
if (lean_obj_tag(v___y_812_) == 0)
{
lean_object* v___x_827_; 
v___x_827_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16);
v___y_777_ = v___y_804_;
v___y_778_ = v___y_805_;
v___y_779_ = v___y_806_;
v___y_780_ = v___y_807_;
v___y_781_ = v___y_808_;
v___y_782_ = v___y_809_;
v___y_783_ = v___y_810_;
v___y_784_ = v___y_811_;
v___y_785_ = v___y_813_;
v___y_786_ = v___y_814_;
v___y_787_ = v___y_815_;
v___y_788_ = v___y_816_;
v___y_789_ = v___y_817_;
v___y_790_ = v___y_819_;
v___y_791_ = v___y_818_;
v___y_792_ = v___y_820_;
v___y_793_ = v___y_822_;
v___y_794_ = v___y_821_;
v___y_795_ = v___y_823_;
v___y_796_ = v___x_826_;
v___y_797_ = v___x_827_;
goto v___jp_776_;
}
else
{
lean_object* v_val_828_; lean_object* v___x_829_; 
v_val_828_ = lean_ctor_get(v___y_812_, 0);
lean_inc(v_val_828_);
lean_dec_ref_known(v___y_812_, 1);
v___x_829_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__1(v_val_828_);
v___y_777_ = v___y_804_;
v___y_778_ = v___y_805_;
v___y_779_ = v___y_806_;
v___y_780_ = v___y_807_;
v___y_781_ = v___y_808_;
v___y_782_ = v___y_809_;
v___y_783_ = v___y_810_;
v___y_784_ = v___y_811_;
v___y_785_ = v___y_813_;
v___y_786_ = v___y_814_;
v___y_787_ = v___y_815_;
v___y_788_ = v___y_816_;
v___y_789_ = v___y_817_;
v___y_790_ = v___y_819_;
v___y_791_ = v___y_818_;
v___y_792_ = v___y_820_;
v___y_793_ = v___y_822_;
v___y_794_ = v___y_821_;
v___y_795_ = v___y_823_;
v___y_796_ = v___x_826_;
v___y_797_ = v___x_829_;
goto v___jp_776_;
}
}
v___jp_830_:
{
lean_object* v___x_853_; lean_object* v___x_854_; lean_object* v___x_855_; lean_object* v___x_856_; lean_object* v___x_857_; lean_object* v___x_858_; lean_object* v___x_859_; lean_object* v___x_860_; lean_object* v___x_861_; 
lean_inc_ref(v___y_848_);
v___x_853_ = l_Array_append___redArg(v___y_848_, v___y_852_);
lean_dec_ref(v___y_852_);
lean_inc_n(v___y_839_, 2);
lean_inc_n(v___y_842_, 6);
v___x_854_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_854_, 0, v___y_842_);
lean_ctor_set(v___x_854_, 1, v___y_839_);
lean_ctor_set(v___x_854_, 2, v___x_853_);
v___x_855_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_scopedNS___closed__20));
lean_inc_ref(v___y_838_);
lean_inc_ref(v___y_833_);
lean_inc_ref(v___y_851_);
v___x_856_ = l_Lean_Name_mkStr4(v___y_851_, v___y_833_, v___y_838_, v___x_855_);
v___x_857_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_857_, 0, v___y_842_);
lean_ctor_set(v___x_857_, 1, v___x_855_);
v___x_858_ = l_Lean_Syntax_node1(v___y_842_, v___x_856_, v___x_857_);
v___x_859_ = l_Lean_Syntax_node1(v___y_842_, v___y_839_, v___x_858_);
lean_inc(v___y_849_);
v___x_860_ = l_Lean_Syntax_node1(v___y_842_, v___y_849_, v___x_859_);
lean_inc_ref(v___y_835_);
v___x_861_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_861_, 0, v___y_842_);
lean_ctor_set(v___x_861_, 1, v___y_835_);
if (lean_obj_tag(v___y_837_) == 0)
{
lean_object* v___x_862_; 
v___x_862_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16);
v___y_804_ = v___y_831_;
v___y_805_ = v___x_860_;
v___y_806_ = v___y_832_;
v___y_807_ = v___y_833_;
v___y_808_ = v___y_834_;
v___y_809_ = v___y_836_;
v___y_810_ = v___y_839_;
v___y_811_ = v___y_840_;
v___y_812_ = v___y_841_;
v___y_813_ = v___y_842_;
v___y_814_ = v___x_854_;
v___y_815_ = v___y_844_;
v___y_816_ = v___y_843_;
v___y_817_ = v___y_845_;
v___y_818_ = v___y_847_;
v___y_819_ = v___y_846_;
v___y_820_ = v___y_848_;
v___y_821_ = v___y_851_;
v___y_822_ = v___y_850_;
v___y_823_ = v___x_861_;
v___y_824_ = v___x_862_;
goto v___jp_803_;
}
else
{
lean_object* v_val_863_; lean_object* v___x_864_; 
v_val_863_ = lean_ctor_get(v___y_837_, 0);
lean_inc(v_val_863_);
lean_dec_ref_known(v___y_837_, 1);
v___x_864_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__1(v_val_863_);
v___y_804_ = v___y_831_;
v___y_805_ = v___x_860_;
v___y_806_ = v___y_832_;
v___y_807_ = v___y_833_;
v___y_808_ = v___y_834_;
v___y_809_ = v___y_836_;
v___y_810_ = v___y_839_;
v___y_811_ = v___y_840_;
v___y_812_ = v___y_841_;
v___y_813_ = v___y_842_;
v___y_814_ = v___x_854_;
v___y_815_ = v___y_844_;
v___y_816_ = v___y_843_;
v___y_817_ = v___y_845_;
v___y_818_ = v___y_847_;
v___y_819_ = v___y_846_;
v___y_820_ = v___y_848_;
v___y_821_ = v___y_851_;
v___y_822_ = v___y_850_;
v___y_823_ = v___x_861_;
v___y_824_ = v___x_864_;
goto v___jp_803_;
}
}
v___jp_865_:
{
lean_object* v___x_888_; lean_object* v___x_889_; 
lean_inc_ref(v___y_883_);
v___x_888_ = l_Array_append___redArg(v___y_883_, v___y_887_);
lean_dec_ref(v___y_887_);
lean_inc(v___y_873_);
lean_inc(v___y_875_);
v___x_889_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_889_, 0, v___y_875_);
lean_ctor_set(v___x_889_, 1, v___y_873_);
lean_ctor_set(v___x_889_, 2, v___x_888_);
if (lean_obj_tag(v___y_880_) == 0)
{
lean_object* v___x_890_; 
v___x_890_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16);
v___y_831_ = v___x_889_;
v___y_832_ = v___y_866_;
v___y_833_ = v___y_867_;
v___y_834_ = v___y_868_;
v___y_835_ = v___y_869_;
v___y_836_ = v___y_870_;
v___y_837_ = v___y_871_;
v___y_838_ = v___y_872_;
v___y_839_ = v___y_873_;
v___y_840_ = v___y_874_;
v___y_841_ = v___y_876_;
v___y_842_ = v___y_875_;
v___y_843_ = v___y_878_;
v___y_844_ = v___y_877_;
v___y_845_ = v___y_879_;
v___y_846_ = v___y_882_;
v___y_847_ = v___y_881_;
v___y_848_ = v___y_883_;
v___y_849_ = v___y_884_;
v___y_850_ = v___y_886_;
v___y_851_ = v___y_885_;
v___y_852_ = v___x_890_;
goto v___jp_830_;
}
else
{
lean_object* v_val_891_; lean_object* v___x_892_; 
v_val_891_ = lean_ctor_get(v___y_880_, 0);
lean_inc(v_val_891_);
lean_dec_ref_known(v___y_880_, 1);
v___x_892_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__1(v_val_891_);
v___y_831_ = v___x_889_;
v___y_832_ = v___y_866_;
v___y_833_ = v___y_867_;
v___y_834_ = v___y_868_;
v___y_835_ = v___y_869_;
v___y_836_ = v___y_870_;
v___y_837_ = v___y_871_;
v___y_838_ = v___y_872_;
v___y_839_ = v___y_873_;
v___y_840_ = v___y_874_;
v___y_841_ = v___y_876_;
v___y_842_ = v___y_875_;
v___y_843_ = v___y_878_;
v___y_844_ = v___y_877_;
v___y_845_ = v___y_879_;
v___y_846_ = v___y_882_;
v___y_847_ = v___y_881_;
v___y_848_ = v___y_883_;
v___y_849_ = v___y_884_;
v___y_850_ = v___y_886_;
v___y_851_ = v___y_885_;
v___y_852_ = v___x_892_;
goto v___jp_830_;
}
}
v___jp_893_:
{
lean_object* v___x_915_; lean_object* v___x_916_; 
lean_inc_ref(v___y_900_);
v___x_915_ = l_Array_append___redArg(v___y_900_, v___y_914_);
lean_dec_ref(v___y_914_);
lean_inc(v___y_897_);
lean_inc(v___y_906_);
v___x_916_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_916_, 0, v___y_906_);
lean_ctor_set(v___x_916_, 1, v___y_897_);
lean_ctor_set(v___x_916_, 2, v___x_915_);
if (lean_obj_tag(v___y_905_) == 0)
{
lean_object* v___x_917_; 
v___x_917_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16);
v___y_629_ = v___y_894_;
v___y_630_ = v___y_895_;
v___y_631_ = v___y_896_;
v___y_632_ = v___y_897_;
v___y_633_ = v___y_898_;
v___y_634_ = v___x_916_;
v___y_635_ = v___y_899_;
v___y_636_ = v___y_900_;
v___y_637_ = v___y_901_;
v___y_638_ = v___y_902_;
v___y_639_ = v___y_903_;
v___y_640_ = v___y_904_;
v___y_641_ = v___y_907_;
v___y_642_ = v___y_906_;
v___y_643_ = v___y_908_;
v___y_644_ = v___y_909_;
v___y_645_ = v___y_911_;
v___y_646_ = v___y_910_;
v___y_647_ = v___y_912_;
v___y_648_ = v___y_913_;
v___y_649_ = v___x_917_;
goto v___jp_628_;
}
else
{
lean_object* v_val_918_; lean_object* v___x_919_; 
v_val_918_ = lean_ctor_get(v___y_905_, 0);
lean_inc(v_val_918_);
lean_dec_ref_known(v___y_905_, 1);
v___x_919_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__1(v_val_918_);
v___y_629_ = v___y_894_;
v___y_630_ = v___y_895_;
v___y_631_ = v___y_896_;
v___y_632_ = v___y_897_;
v___y_633_ = v___y_898_;
v___y_634_ = v___x_916_;
v___y_635_ = v___y_899_;
v___y_636_ = v___y_900_;
v___y_637_ = v___y_901_;
v___y_638_ = v___y_902_;
v___y_639_ = v___y_903_;
v___y_640_ = v___y_904_;
v___y_641_ = v___y_907_;
v___y_642_ = v___y_906_;
v___y_643_ = v___y_908_;
v___y_644_ = v___y_909_;
v___y_645_ = v___y_911_;
v___y_646_ = v___y_910_;
v___y_647_ = v___y_912_;
v___y_648_ = v___y_913_;
v___y_649_ = v___x_919_;
goto v___jp_628_;
}
}
v___jp_920_:
{
lean_object* v___x_943_; lean_object* v___x_944_; lean_object* v___x_945_; lean_object* v___x_946_; lean_object* v___x_947_; lean_object* v___x_948_; lean_object* v___x_949_; lean_object* v___x_950_; 
lean_inc_ref(v___y_927_);
v___x_943_ = l_Array_append___redArg(v___y_927_, v___y_942_);
lean_dec_ref(v___y_942_);
lean_inc_n(v___y_923_, 2);
lean_inc_n(v___y_934_, 5);
v___x_944_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_944_, 0, v___y_934_);
lean_ctor_set(v___x_944_, 1, v___y_923_);
lean_ctor_set(v___x_944_, 2, v___x_943_);
v___x_945_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_scopedNS___closed__20));
lean_inc_ref(v___y_940_);
lean_inc_ref(v___y_922_);
lean_inc_ref(v___y_939_);
v___x_946_ = l_Lean_Name_mkStr4(v___y_939_, v___y_922_, v___y_940_, v___x_945_);
v___x_947_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_947_, 0, v___y_934_);
lean_ctor_set(v___x_947_, 1, v___x_945_);
v___x_948_ = l_Lean_Syntax_node1(v___y_934_, v___x_946_, v___x_947_);
v___x_949_ = l_Lean_Syntax_node1(v___y_934_, v___y_923_, v___x_948_);
lean_inc(v___y_926_);
v___x_950_ = l_Lean_Syntax_node1(v___y_934_, v___y_926_, v___x_949_);
if (lean_obj_tag(v___y_938_) == 0)
{
lean_object* v___x_951_; 
v___x_951_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16);
v___y_894_ = v___y_921_;
v___y_895_ = v___y_922_;
v___y_896_ = v___x_944_;
v___y_897_ = v___y_923_;
v___y_898_ = v___y_924_;
v___y_899_ = v___y_925_;
v___y_900_ = v___y_927_;
v___y_901_ = v___y_928_;
v___y_902_ = v___y_929_;
v___y_903_ = v___y_931_;
v___y_904_ = v___y_930_;
v___y_905_ = v___y_933_;
v___y_906_ = v___y_934_;
v___y_907_ = v___y_932_;
v___y_908_ = v___y_935_;
v___y_909_ = v___y_936_;
v___y_910_ = v___x_950_;
v___y_911_ = v___y_937_;
v___y_912_ = v___y_939_;
v___y_913_ = v___y_941_;
v___y_914_ = v___x_951_;
goto v___jp_893_;
}
else
{
lean_object* v_val_952_; lean_object* v___x_953_; 
v_val_952_ = lean_ctor_get(v___y_938_, 0);
lean_inc(v_val_952_);
lean_dec_ref_known(v___y_938_, 1);
v___x_953_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__1(v_val_952_);
v___y_894_ = v___y_921_;
v___y_895_ = v___y_922_;
v___y_896_ = v___x_944_;
v___y_897_ = v___y_923_;
v___y_898_ = v___y_924_;
v___y_899_ = v___y_925_;
v___y_900_ = v___y_927_;
v___y_901_ = v___y_928_;
v___y_902_ = v___y_929_;
v___y_903_ = v___y_931_;
v___y_904_ = v___y_930_;
v___y_905_ = v___y_933_;
v___y_906_ = v___y_934_;
v___y_907_ = v___y_932_;
v___y_908_ = v___y_935_;
v___y_909_ = v___y_936_;
v___y_910_ = v___x_950_;
v___y_911_ = v___y_937_;
v___y_912_ = v___y_939_;
v___y_913_ = v___y_941_;
v___y_914_ = v___x_953_;
goto v___jp_893_;
}
}
v___jp_954_:
{
lean_object* v___x_977_; lean_object* v___x_978_; 
lean_inc_ref(v___y_961_);
v___x_977_ = l_Array_append___redArg(v___y_961_, v___y_976_);
lean_dec_ref(v___y_976_);
lean_inc(v___y_957_);
lean_inc(v___y_966_);
v___x_978_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_978_, 0, v___y_966_);
lean_ctor_set(v___x_978_, 1, v___y_957_);
lean_ctor_set(v___x_978_, 2, v___x_977_);
if (lean_obj_tag(v___y_955_) == 0)
{
lean_object* v___x_979_; 
v___x_979_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16);
v___y_921_ = v___x_978_;
v___y_922_ = v___y_956_;
v___y_923_ = v___y_957_;
v___y_924_ = v___y_958_;
v___y_925_ = v___y_959_;
v___y_926_ = v___y_960_;
v___y_927_ = v___y_961_;
v___y_928_ = v___y_962_;
v___y_929_ = v___y_963_;
v___y_930_ = v___y_964_;
v___y_931_ = v___y_965_;
v___y_932_ = v___y_967_;
v___y_933_ = v___y_968_;
v___y_934_ = v___y_966_;
v___y_935_ = v___y_969_;
v___y_936_ = v___y_970_;
v___y_937_ = v___y_971_;
v___y_938_ = v___y_973_;
v___y_939_ = v___y_972_;
v___y_940_ = v___y_974_;
v___y_941_ = v___y_975_;
v___y_942_ = v___x_979_;
goto v___jp_920_;
}
else
{
lean_object* v_val_980_; lean_object* v___x_981_; 
v_val_980_ = lean_ctor_get(v___y_955_, 0);
lean_inc(v_val_980_);
lean_dec_ref_known(v___y_955_, 1);
v___x_981_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__1(v_val_980_);
v___y_921_ = v___x_978_;
v___y_922_ = v___y_956_;
v___y_923_ = v___y_957_;
v___y_924_ = v___y_958_;
v___y_925_ = v___y_959_;
v___y_926_ = v___y_960_;
v___y_927_ = v___y_961_;
v___y_928_ = v___y_962_;
v___y_929_ = v___y_963_;
v___y_930_ = v___y_964_;
v___y_931_ = v___y_965_;
v___y_932_ = v___y_967_;
v___y_933_ = v___y_968_;
v___y_934_ = v___y_966_;
v___y_935_ = v___y_969_;
v___y_936_ = v___y_970_;
v___y_937_ = v___y_971_;
v___y_938_ = v___y_973_;
v___y_939_ = v___y_972_;
v___y_940_ = v___y_974_;
v___y_941_ = v___y_975_;
v___y_942_ = v___x_981_;
goto v___jp_920_;
}
}
v___jp_982_:
{
lean_object* v___x_1004_; lean_object* v___x_1005_; 
lean_inc_ref(v___y_984_);
v___x_1004_ = l_Array_append___redArg(v___y_984_, v___y_1003_);
lean_dec_ref(v___y_1003_);
lean_inc(v___y_986_);
lean_inc(v___y_1000_);
v___x_1005_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1005_, 0, v___y_1000_);
lean_ctor_set(v___x_1005_, 1, v___y_986_);
lean_ctor_set(v___x_1005_, 2, v___x_1004_);
if (lean_obj_tag(v___y_992_) == 0)
{
lean_object* v___x_1006_; 
v___x_1006_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16);
v___y_539_ = v___y_983_;
v___y_540_ = v___y_984_;
v___y_541_ = v___y_985_;
v___y_542_ = v___y_987_;
v___y_543_ = v___y_986_;
v___y_544_ = v___y_988_;
v___y_545_ = v___y_989_;
v___y_546_ = v___y_990_;
v___y_547_ = v___y_991_;
v___y_548_ = v___y_993_;
v___y_549_ = v___y_994_;
v___y_550_ = v___y_995_;
v___y_551_ = v___x_1005_;
v___y_552_ = v___y_996_;
v___y_553_ = v___y_997_;
v___y_554_ = v___y_998_;
v___y_555_ = v___y_999_;
v___y_556_ = v___y_1000_;
v___y_557_ = v___y_1001_;
v___y_558_ = v___y_1002_;
v___y_559_ = v___x_1006_;
goto v___jp_538_;
}
else
{
lean_object* v_val_1007_; lean_object* v___x_1008_; 
v_val_1007_ = lean_ctor_get(v___y_992_, 0);
lean_inc(v_val_1007_);
lean_dec_ref_known(v___y_992_, 1);
v___x_1008_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__1(v_val_1007_);
v___y_539_ = v___y_983_;
v___y_540_ = v___y_984_;
v___y_541_ = v___y_985_;
v___y_542_ = v___y_987_;
v___y_543_ = v___y_986_;
v___y_544_ = v___y_988_;
v___y_545_ = v___y_989_;
v___y_546_ = v___y_990_;
v___y_547_ = v___y_991_;
v___y_548_ = v___y_993_;
v___y_549_ = v___y_994_;
v___y_550_ = v___y_995_;
v___y_551_ = v___x_1005_;
v___y_552_ = v___y_996_;
v___y_553_ = v___y_997_;
v___y_554_ = v___y_998_;
v___y_555_ = v___y_999_;
v___y_556_ = v___y_1000_;
v___y_557_ = v___y_1001_;
v___y_558_ = v___y_1002_;
v___y_559_ = v___x_1008_;
goto v___jp_538_;
}
}
v___jp_1009_:
{
lean_object* v___x_1032_; lean_object* v___x_1033_; lean_object* v___x_1034_; lean_object* v___x_1035_; lean_object* v___x_1036_; lean_object* v___x_1037_; lean_object* v___x_1038_; lean_object* v___x_1039_; 
lean_inc_ref(v___y_1011_);
v___x_1032_ = l_Array_append___redArg(v___y_1011_, v___y_1031_);
lean_dec_ref(v___y_1031_);
lean_inc_n(v___y_1013_, 2);
lean_inc_n(v___y_1027_, 5);
v___x_1033_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1033_, 0, v___y_1027_);
lean_ctor_set(v___x_1033_, 1, v___y_1013_);
lean_ctor_set(v___x_1033_, 2, v___x_1032_);
v___x_1034_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_scopedNS___closed__20));
lean_inc_ref(v___y_1029_);
lean_inc_ref(v___y_1014_);
lean_inc_ref(v___y_1028_);
v___x_1035_ = l_Lean_Name_mkStr4(v___y_1028_, v___y_1014_, v___y_1029_, v___x_1034_);
v___x_1036_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1036_, 0, v___y_1027_);
lean_ctor_set(v___x_1036_, 1, v___x_1034_);
v___x_1037_ = l_Lean_Syntax_node1(v___y_1027_, v___x_1035_, v___x_1036_);
v___x_1038_ = l_Lean_Syntax_node1(v___y_1027_, v___y_1013_, v___x_1037_);
lean_inc(v___y_1018_);
v___x_1039_ = l_Lean_Syntax_node1(v___y_1027_, v___y_1018_, v___x_1038_);
if (lean_obj_tag(v___y_1026_) == 0)
{
lean_object* v___x_1040_; 
v___x_1040_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16);
v___y_983_ = v___y_1010_;
v___y_984_ = v___y_1011_;
v___y_985_ = v___y_1012_;
v___y_986_ = v___y_1013_;
v___y_987_ = v___y_1014_;
v___y_988_ = v___x_1033_;
v___y_989_ = v___y_1015_;
v___y_990_ = v___y_1016_;
v___y_991_ = v___y_1017_;
v___y_992_ = v___y_1019_;
v___y_993_ = v___y_1021_;
v___y_994_ = v___y_1020_;
v___y_995_ = v___y_1022_;
v___y_996_ = v___y_1023_;
v___y_997_ = v___y_1024_;
v___y_998_ = v___y_1025_;
v___y_999_ = v___y_1028_;
v___y_1000_ = v___y_1027_;
v___y_1001_ = v___x_1039_;
v___y_1002_ = v___y_1030_;
v___y_1003_ = v___x_1040_;
goto v___jp_982_;
}
else
{
lean_object* v_val_1041_; lean_object* v___x_1042_; 
v_val_1041_ = lean_ctor_get(v___y_1026_, 0);
lean_inc(v_val_1041_);
lean_dec_ref_known(v___y_1026_, 1);
v___x_1042_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__1(v_val_1041_);
v___y_983_ = v___y_1010_;
v___y_984_ = v___y_1011_;
v___y_985_ = v___y_1012_;
v___y_986_ = v___y_1013_;
v___y_987_ = v___y_1014_;
v___y_988_ = v___x_1033_;
v___y_989_ = v___y_1015_;
v___y_990_ = v___y_1016_;
v___y_991_ = v___y_1017_;
v___y_992_ = v___y_1019_;
v___y_993_ = v___y_1021_;
v___y_994_ = v___y_1020_;
v___y_995_ = v___y_1022_;
v___y_996_ = v___y_1023_;
v___y_997_ = v___y_1024_;
v___y_998_ = v___y_1025_;
v___y_999_ = v___y_1028_;
v___y_1000_ = v___y_1027_;
v___y_1001_ = v___x_1039_;
v___y_1002_ = v___y_1030_;
v___y_1003_ = v___x_1042_;
goto v___jp_982_;
}
}
v___jp_1043_:
{
lean_object* v___x_1066_; lean_object* v___x_1067_; 
lean_inc_ref(v___y_1044_);
v___x_1066_ = l_Array_append___redArg(v___y_1044_, v___y_1065_);
lean_dec_ref(v___y_1065_);
lean_inc(v___y_1046_);
lean_inc(v___y_1062_);
v___x_1067_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1067_, 0, v___y_1062_);
lean_ctor_set(v___x_1067_, 1, v___y_1046_);
lean_ctor_set(v___x_1067_, 2, v___x_1066_);
if (lean_obj_tag(v___y_1050_) == 0)
{
lean_object* v___x_1068_; 
v___x_1068_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16);
v___y_1010_ = v___x_1067_;
v___y_1011_ = v___y_1044_;
v___y_1012_ = v___y_1045_;
v___y_1013_ = v___y_1046_;
v___y_1014_ = v___y_1047_;
v___y_1015_ = v___y_1048_;
v___y_1016_ = v___y_1049_;
v___y_1017_ = v___y_1051_;
v___y_1018_ = v___y_1052_;
v___y_1019_ = v___y_1053_;
v___y_1020_ = v___y_1054_;
v___y_1021_ = v___y_1055_;
v___y_1022_ = v___y_1056_;
v___y_1023_ = v___y_1057_;
v___y_1024_ = v___y_1058_;
v___y_1025_ = v___y_1059_;
v___y_1026_ = v___y_1060_;
v___y_1027_ = v___y_1062_;
v___y_1028_ = v___y_1061_;
v___y_1029_ = v___y_1063_;
v___y_1030_ = v___y_1064_;
v___y_1031_ = v___x_1068_;
goto v___jp_1009_;
}
else
{
lean_object* v_val_1069_; lean_object* v___x_1070_; 
v_val_1069_ = lean_ctor_get(v___y_1050_, 0);
lean_inc(v_val_1069_);
lean_dec_ref_known(v___y_1050_, 1);
v___x_1070_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__1(v_val_1069_);
v___y_1010_ = v___x_1067_;
v___y_1011_ = v___y_1044_;
v___y_1012_ = v___y_1045_;
v___y_1013_ = v___y_1046_;
v___y_1014_ = v___y_1047_;
v___y_1015_ = v___y_1048_;
v___y_1016_ = v___y_1049_;
v___y_1017_ = v___y_1051_;
v___y_1018_ = v___y_1052_;
v___y_1019_ = v___y_1053_;
v___y_1020_ = v___y_1054_;
v___y_1021_ = v___y_1055_;
v___y_1022_ = v___y_1056_;
v___y_1023_ = v___y_1057_;
v___y_1024_ = v___y_1058_;
v___y_1025_ = v___y_1059_;
v___y_1026_ = v___y_1060_;
v___y_1027_ = v___y_1062_;
v___y_1028_ = v___y_1061_;
v___y_1029_ = v___y_1063_;
v___y_1030_ = v___y_1064_;
v___y_1031_ = v___x_1070_;
goto v___jp_1009_;
}
}
v___jp_1071_:
{
lean_object* v___x_1093_; lean_object* v___x_1094_; 
lean_inc_ref(v___y_1078_);
v___x_1093_ = l_Array_append___redArg(v___y_1078_, v___y_1092_);
lean_dec_ref(v___y_1092_);
lean_inc(v___y_1080_);
lean_inc(v___y_1091_);
v___x_1094_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1094_, 0, v___y_1091_);
lean_ctor_set(v___x_1094_, 1, v___y_1080_);
lean_ctor_set(v___x_1094_, 2, v___x_1093_);
if (lean_obj_tag(v___y_1089_) == 0)
{
lean_object* v___x_1095_; 
v___x_1095_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16);
v___y_449_ = v___y_1072_;
v___y_450_ = v___y_1073_;
v___y_451_ = v___y_1074_;
v___y_452_ = v___y_1075_;
v___y_453_ = v___y_1076_;
v___y_454_ = v___y_1077_;
v___y_455_ = v___y_1078_;
v___y_456_ = v___y_1079_;
v___y_457_ = v___y_1080_;
v___y_458_ = v___y_1081_;
v___y_459_ = v___x_1094_;
v___y_460_ = v___y_1082_;
v___y_461_ = v___y_1084_;
v___y_462_ = v___y_1083_;
v___y_463_ = v___y_1085_;
v___y_464_ = v___y_1086_;
v___y_465_ = v___y_1087_;
v___y_466_ = v___y_1088_;
v___y_467_ = v___y_1091_;
v___y_468_ = v___y_1090_;
v___y_469_ = v___x_1095_;
goto v___jp_448_;
}
else
{
lean_object* v_val_1096_; lean_object* v___x_1097_; 
v_val_1096_ = lean_ctor_get(v___y_1089_, 0);
lean_inc(v_val_1096_);
lean_dec_ref_known(v___y_1089_, 1);
v___x_1097_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__1(v_val_1096_);
v___y_449_ = v___y_1072_;
v___y_450_ = v___y_1073_;
v___y_451_ = v___y_1074_;
v___y_452_ = v___y_1075_;
v___y_453_ = v___y_1076_;
v___y_454_ = v___y_1077_;
v___y_455_ = v___y_1078_;
v___y_456_ = v___y_1079_;
v___y_457_ = v___y_1080_;
v___y_458_ = v___y_1081_;
v___y_459_ = v___x_1094_;
v___y_460_ = v___y_1082_;
v___y_461_ = v___y_1084_;
v___y_462_ = v___y_1083_;
v___y_463_ = v___y_1085_;
v___y_464_ = v___y_1086_;
v___y_465_ = v___y_1087_;
v___y_466_ = v___y_1088_;
v___y_467_ = v___y_1091_;
v___y_468_ = v___y_1090_;
v___y_469_ = v___x_1097_;
goto v___jp_448_;
}
}
v___jp_1098_:
{
lean_object* v___x_1121_; lean_object* v___x_1122_; lean_object* v___x_1123_; lean_object* v___x_1124_; lean_object* v___x_1125_; lean_object* v___x_1126_; lean_object* v___x_1127_; lean_object* v___x_1128_; 
lean_inc_ref(v___y_1106_);
v___x_1121_ = l_Array_append___redArg(v___y_1106_, v___y_1120_);
lean_dec_ref(v___y_1120_);
lean_inc_n(v___y_1107_, 2);
lean_inc_n(v___y_1119_, 5);
v___x_1122_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1122_, 0, v___y_1119_);
lean_ctor_set(v___x_1122_, 1, v___y_1107_);
lean_ctor_set(v___x_1122_, 2, v___x_1121_);
v___x_1123_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_scopedNS___closed__20));
lean_inc_ref(v___y_1116_);
lean_inc_ref(v___y_1100_);
lean_inc_ref(v___y_1115_);
v___x_1124_ = l_Lean_Name_mkStr4(v___y_1115_, v___y_1100_, v___y_1116_, v___x_1123_);
v___x_1125_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1125_, 0, v___y_1119_);
lean_ctor_set(v___x_1125_, 1, v___x_1123_);
v___x_1126_ = l_Lean_Syntax_node1(v___y_1119_, v___x_1124_, v___x_1125_);
v___x_1127_ = l_Lean_Syntax_node1(v___y_1119_, v___y_1107_, v___x_1126_);
lean_inc(v___y_1103_);
v___x_1128_ = l_Lean_Syntax_node1(v___y_1119_, v___y_1103_, v___x_1127_);
if (lean_obj_tag(v___y_1109_) == 0)
{
lean_object* v___x_1129_; 
v___x_1129_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16);
v___y_1072_ = v___y_1099_;
v___y_1073_ = v___y_1100_;
v___y_1074_ = v___y_1101_;
v___y_1075_ = v___y_1102_;
v___y_1076_ = v___y_1104_;
v___y_1077_ = v___y_1105_;
v___y_1078_ = v___y_1106_;
v___y_1079_ = v___x_1128_;
v___y_1080_ = v___y_1107_;
v___y_1081_ = v___y_1108_;
v___y_1082_ = v___y_1110_;
v___y_1083_ = v___y_1112_;
v___y_1084_ = v___y_1111_;
v___y_1085_ = v___y_1113_;
v___y_1086_ = v___x_1122_;
v___y_1087_ = v___y_1114_;
v___y_1088_ = v___y_1115_;
v___y_1089_ = v___y_1118_;
v___y_1090_ = v___y_1117_;
v___y_1091_ = v___y_1119_;
v___y_1092_ = v___x_1129_;
goto v___jp_1071_;
}
else
{
lean_object* v_val_1130_; lean_object* v___x_1131_; 
v_val_1130_ = lean_ctor_get(v___y_1109_, 0);
lean_inc(v_val_1130_);
lean_dec_ref_known(v___y_1109_, 1);
v___x_1131_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__1(v_val_1130_);
v___y_1072_ = v___y_1099_;
v___y_1073_ = v___y_1100_;
v___y_1074_ = v___y_1101_;
v___y_1075_ = v___y_1102_;
v___y_1076_ = v___y_1104_;
v___y_1077_ = v___y_1105_;
v___y_1078_ = v___y_1106_;
v___y_1079_ = v___x_1128_;
v___y_1080_ = v___y_1107_;
v___y_1081_ = v___y_1108_;
v___y_1082_ = v___y_1110_;
v___y_1083_ = v___y_1112_;
v___y_1084_ = v___y_1111_;
v___y_1085_ = v___y_1113_;
v___y_1086_ = v___x_1122_;
v___y_1087_ = v___y_1114_;
v___y_1088_ = v___y_1115_;
v___y_1089_ = v___y_1118_;
v___y_1090_ = v___y_1117_;
v___y_1091_ = v___y_1119_;
v___y_1092_ = v___x_1131_;
goto v___jp_1071_;
}
}
v___jp_1132_:
{
lean_object* v___x_1155_; lean_object* v___x_1156_; 
lean_inc_ref(v___y_1139_);
v___x_1155_ = l_Array_append___redArg(v___y_1139_, v___y_1154_);
lean_dec_ref(v___y_1154_);
lean_inc(v___y_1140_);
lean_inc(v___y_1153_);
v___x_1156_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1156_, 0, v___y_1153_);
lean_ctor_set(v___x_1156_, 1, v___y_1140_);
lean_ctor_set(v___x_1156_, 2, v___x_1155_);
if (lean_obj_tag(v___y_1150_) == 0)
{
lean_object* v___x_1157_; 
v___x_1157_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16);
v___y_1099_ = v___y_1133_;
v___y_1100_ = v___y_1134_;
v___y_1101_ = v___y_1135_;
v___y_1102_ = v___y_1136_;
v___y_1103_ = v___y_1137_;
v___y_1104_ = v___y_1138_;
v___y_1105_ = v___x_1156_;
v___y_1106_ = v___y_1139_;
v___y_1107_ = v___y_1140_;
v___y_1108_ = v___y_1141_;
v___y_1109_ = v___y_1142_;
v___y_1110_ = v___y_1143_;
v___y_1111_ = v___y_1145_;
v___y_1112_ = v___y_1144_;
v___y_1113_ = v___y_1146_;
v___y_1114_ = v___y_1147_;
v___y_1115_ = v___y_1148_;
v___y_1116_ = v___y_1149_;
v___y_1117_ = v___y_1152_;
v___y_1118_ = v___y_1151_;
v___y_1119_ = v___y_1153_;
v___y_1120_ = v___x_1157_;
goto v___jp_1098_;
}
else
{
lean_object* v_val_1158_; lean_object* v___x_1159_; 
v_val_1158_ = lean_ctor_get(v___y_1150_, 0);
lean_inc(v_val_1158_);
lean_dec_ref_known(v___y_1150_, 1);
v___x_1159_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__1(v_val_1158_);
v___y_1099_ = v___y_1133_;
v___y_1100_ = v___y_1134_;
v___y_1101_ = v___y_1135_;
v___y_1102_ = v___y_1136_;
v___y_1103_ = v___y_1137_;
v___y_1104_ = v___y_1138_;
v___y_1105_ = v___x_1156_;
v___y_1106_ = v___y_1139_;
v___y_1107_ = v___y_1140_;
v___y_1108_ = v___y_1141_;
v___y_1109_ = v___y_1142_;
v___y_1110_ = v___y_1143_;
v___y_1111_ = v___y_1145_;
v___y_1112_ = v___y_1144_;
v___y_1113_ = v___y_1146_;
v___y_1114_ = v___y_1147_;
v___y_1115_ = v___y_1148_;
v___y_1116_ = v___y_1149_;
v___y_1117_ = v___y_1152_;
v___y_1118_ = v___y_1151_;
v___y_1119_ = v___y_1153_;
v___y_1120_ = v___x_1159_;
goto v___jp_1098_;
}
}
v___jp_1160_:
{
lean_object* v___x_1182_; lean_object* v___x_1183_; 
lean_inc_ref(v___y_1166_);
v___x_1182_ = l_Array_append___redArg(v___y_1166_, v___y_1181_);
lean_dec_ref(v___y_1181_);
lean_inc(v___y_1162_);
lean_inc(v___y_1164_);
v___x_1183_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1183_, 0, v___y_1164_);
lean_ctor_set(v___x_1183_, 1, v___y_1162_);
lean_ctor_set(v___x_1183_, 2, v___x_1182_);
if (lean_obj_tag(v___y_1168_) == 0)
{
lean_object* v___x_1184_; 
v___x_1184_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16);
v___y_494_ = v___y_1161_;
v___y_495_ = v___y_1162_;
v___y_496_ = v___y_1163_;
v___y_497_ = v___y_1165_;
v___y_498_ = v___y_1164_;
v___y_499_ = v___y_1166_;
v___y_500_ = v___y_1167_;
v___y_501_ = v___x_1183_;
v___y_502_ = v___y_1169_;
v___y_503_ = v___y_1170_;
v___y_504_ = v___y_1171_;
v___y_505_ = v___y_1172_;
v___y_506_ = v___y_1173_;
v___y_507_ = v___y_1174_;
v___y_508_ = v___y_1175_;
v___y_509_ = v___y_1176_;
v___y_510_ = v___y_1177_;
v___y_511_ = v___y_1178_;
v___y_512_ = v___y_1179_;
v___y_513_ = v___y_1180_;
v___y_514_ = v___x_1184_;
goto v___jp_493_;
}
else
{
lean_object* v_val_1185_; lean_object* v___x_1186_; 
v_val_1185_ = lean_ctor_get(v___y_1168_, 0);
lean_inc(v_val_1185_);
lean_dec_ref_known(v___y_1168_, 1);
v___x_1186_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__1(v_val_1185_);
v___y_494_ = v___y_1161_;
v___y_495_ = v___y_1162_;
v___y_496_ = v___y_1163_;
v___y_497_ = v___y_1165_;
v___y_498_ = v___y_1164_;
v___y_499_ = v___y_1166_;
v___y_500_ = v___y_1167_;
v___y_501_ = v___x_1183_;
v___y_502_ = v___y_1169_;
v___y_503_ = v___y_1170_;
v___y_504_ = v___y_1171_;
v___y_505_ = v___y_1172_;
v___y_506_ = v___y_1173_;
v___y_507_ = v___y_1174_;
v___y_508_ = v___y_1175_;
v___y_509_ = v___y_1176_;
v___y_510_ = v___y_1177_;
v___y_511_ = v___y_1178_;
v___y_512_ = v___y_1179_;
v___y_513_ = v___y_1180_;
v___y_514_ = v___x_1186_;
goto v___jp_493_;
}
}
v___jp_1187_:
{
lean_object* v___x_1210_; lean_object* v___x_1211_; lean_object* v___x_1212_; lean_object* v___x_1213_; lean_object* v___x_1214_; lean_object* v___x_1215_; lean_object* v___x_1216_; lean_object* v___x_1217_; 
lean_inc_ref(v___y_1193_);
v___x_1210_ = l_Array_append___redArg(v___y_1193_, v___y_1209_);
lean_dec_ref(v___y_1209_);
lean_inc_n(v___y_1189_, 2);
lean_inc_n(v___y_1191_, 5);
v___x_1211_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1211_, 0, v___y_1191_);
lean_ctor_set(v___x_1211_, 1, v___y_1189_);
lean_ctor_set(v___x_1211_, 2, v___x_1210_);
v___x_1212_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_scopedNS___closed__20));
lean_inc_ref(v___y_1207_);
lean_inc_ref(v___y_1190_);
lean_inc_ref(v___y_1206_);
v___x_1213_ = l_Lean_Name_mkStr4(v___y_1206_, v___y_1190_, v___y_1207_, v___x_1212_);
v___x_1214_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1214_, 0, v___y_1191_);
lean_ctor_set(v___x_1214_, 1, v___x_1212_);
v___x_1215_ = l_Lean_Syntax_node1(v___y_1191_, v___x_1213_, v___x_1214_);
v___x_1216_ = l_Lean_Syntax_node1(v___y_1191_, v___y_1189_, v___x_1215_);
lean_inc(v___y_1195_);
v___x_1217_ = l_Lean_Syntax_node1(v___y_1191_, v___y_1195_, v___x_1216_);
if (lean_obj_tag(v___y_1203_) == 0)
{
lean_object* v___x_1218_; 
v___x_1218_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16);
v___y_1161_ = v___y_1188_;
v___y_1162_ = v___y_1189_;
v___y_1163_ = v___y_1190_;
v___y_1164_ = v___y_1191_;
v___y_1165_ = v___y_1192_;
v___y_1166_ = v___y_1193_;
v___y_1167_ = v___y_1194_;
v___y_1168_ = v___y_1196_;
v___y_1169_ = v___x_1211_;
v___y_1170_ = v___y_1197_;
v___y_1171_ = v___y_1198_;
v___y_1172_ = v___y_1199_;
v___y_1173_ = v___y_1201_;
v___y_1174_ = v___y_1200_;
v___y_1175_ = v___y_1202_;
v___y_1176_ = v___y_1204_;
v___y_1177_ = v___x_1217_;
v___y_1178_ = v___y_1205_;
v___y_1179_ = v___y_1206_;
v___y_1180_ = v___y_1208_;
v___y_1181_ = v___x_1218_;
goto v___jp_1160_;
}
else
{
lean_object* v_val_1219_; lean_object* v___x_1220_; 
v_val_1219_ = lean_ctor_get(v___y_1203_, 0);
lean_inc(v_val_1219_);
lean_dec_ref_known(v___y_1203_, 1);
v___x_1220_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__1(v_val_1219_);
v___y_1161_ = v___y_1188_;
v___y_1162_ = v___y_1189_;
v___y_1163_ = v___y_1190_;
v___y_1164_ = v___y_1191_;
v___y_1165_ = v___y_1192_;
v___y_1166_ = v___y_1193_;
v___y_1167_ = v___y_1194_;
v___y_1168_ = v___y_1196_;
v___y_1169_ = v___x_1211_;
v___y_1170_ = v___y_1197_;
v___y_1171_ = v___y_1198_;
v___y_1172_ = v___y_1199_;
v___y_1173_ = v___y_1201_;
v___y_1174_ = v___y_1200_;
v___y_1175_ = v___y_1202_;
v___y_1176_ = v___y_1204_;
v___y_1177_ = v___x_1217_;
v___y_1178_ = v___y_1205_;
v___y_1179_ = v___y_1206_;
v___y_1180_ = v___y_1208_;
v___y_1181_ = v___x_1220_;
goto v___jp_1160_;
}
}
v___jp_1221_:
{
lean_object* v___x_1244_; lean_object* v___x_1245_; 
lean_inc_ref(v___y_1227_);
v___x_1244_ = l_Array_append___redArg(v___y_1227_, v___y_1243_);
lean_dec_ref(v___y_1243_);
lean_inc(v___y_1223_);
lean_inc(v___y_1225_);
v___x_1245_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1245_, 0, v___y_1225_);
lean_ctor_set(v___x_1245_, 1, v___y_1223_);
lean_ctor_set(v___x_1245_, 2, v___x_1244_);
if (lean_obj_tag(v___y_1238_) == 0)
{
lean_object* v___x_1246_; 
v___x_1246_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16);
v___y_1188_ = v___y_1222_;
v___y_1189_ = v___y_1223_;
v___y_1190_ = v___y_1224_;
v___y_1191_ = v___y_1225_;
v___y_1192_ = v___y_1226_;
v___y_1193_ = v___y_1227_;
v___y_1194_ = v___y_1228_;
v___y_1195_ = v___y_1229_;
v___y_1196_ = v___y_1230_;
v___y_1197_ = v___y_1231_;
v___y_1198_ = v___y_1232_;
v___y_1199_ = v___y_1233_;
v___y_1200_ = v___x_1245_;
v___y_1201_ = v___y_1234_;
v___y_1202_ = v___y_1235_;
v___y_1203_ = v___y_1237_;
v___y_1204_ = v___y_1236_;
v___y_1205_ = v___y_1239_;
v___y_1206_ = v___y_1240_;
v___y_1207_ = v___y_1241_;
v___y_1208_ = v___y_1242_;
v___y_1209_ = v___x_1246_;
goto v___jp_1187_;
}
else
{
lean_object* v_val_1247_; lean_object* v___x_1248_; 
v_val_1247_ = lean_ctor_get(v___y_1238_, 0);
lean_inc(v_val_1247_);
lean_dec_ref_known(v___y_1238_, 1);
v___x_1248_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__1(v_val_1247_);
v___y_1188_ = v___y_1222_;
v___y_1189_ = v___y_1223_;
v___y_1190_ = v___y_1224_;
v___y_1191_ = v___y_1225_;
v___y_1192_ = v___y_1226_;
v___y_1193_ = v___y_1227_;
v___y_1194_ = v___y_1228_;
v___y_1195_ = v___y_1229_;
v___y_1196_ = v___y_1230_;
v___y_1197_ = v___y_1231_;
v___y_1198_ = v___y_1232_;
v___y_1199_ = v___y_1233_;
v___y_1200_ = v___x_1245_;
v___y_1201_ = v___y_1234_;
v___y_1202_ = v___y_1235_;
v___y_1203_ = v___y_1237_;
v___y_1204_ = v___y_1236_;
v___y_1205_ = v___y_1239_;
v___y_1206_ = v___y_1240_;
v___y_1207_ = v___y_1241_;
v___y_1208_ = v___y_1242_;
v___y_1209_ = v___x_1248_;
goto v___jp_1187_;
}
}
v___jp_1249_:
{
lean_object* v___x_1271_; lean_object* v___x_1272_; 
lean_inc_ref(v___y_1254_);
v___x_1271_ = l_Array_append___redArg(v___y_1254_, v___y_1270_);
lean_dec_ref(v___y_1270_);
lean_inc(v___y_1267_);
lean_inc(v___y_1263_);
v___x_1272_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1272_, 0, v___y_1263_);
lean_ctor_set(v___x_1272_, 1, v___y_1267_);
lean_ctor_set(v___x_1272_, 2, v___x_1271_);
if (lean_obj_tag(v___y_1266_) == 0)
{
lean_object* v___x_1273_; 
v___x_1273_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16);
v___y_584_ = v___y_1250_;
v___y_585_ = v___x_1272_;
v___y_586_ = v___y_1251_;
v___y_587_ = v___y_1252_;
v___y_588_ = v___y_1253_;
v___y_589_ = v___y_1254_;
v___y_590_ = v___y_1255_;
v___y_591_ = v___y_1256_;
v___y_592_ = v___y_1257_;
v___y_593_ = v___y_1258_;
v___y_594_ = v___y_1259_;
v___y_595_ = v___y_1261_;
v___y_596_ = v___y_1260_;
v___y_597_ = v___y_1262_;
v___y_598_ = v___y_1263_;
v___y_599_ = v___y_1265_;
v___y_600_ = v___y_1264_;
v___y_601_ = v___y_1267_;
v___y_602_ = v___y_1269_;
v___y_603_ = v___y_1268_;
v___y_604_ = v___x_1273_;
goto v___jp_583_;
}
else
{
lean_object* v_val_1274_; lean_object* v___x_1275_; 
v_val_1274_ = lean_ctor_get(v___y_1266_, 0);
lean_inc(v_val_1274_);
lean_dec_ref_known(v___y_1266_, 1);
v___x_1275_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__1(v_val_1274_);
v___y_584_ = v___y_1250_;
v___y_585_ = v___x_1272_;
v___y_586_ = v___y_1251_;
v___y_587_ = v___y_1252_;
v___y_588_ = v___y_1253_;
v___y_589_ = v___y_1254_;
v___y_590_ = v___y_1255_;
v___y_591_ = v___y_1256_;
v___y_592_ = v___y_1257_;
v___y_593_ = v___y_1258_;
v___y_594_ = v___y_1259_;
v___y_595_ = v___y_1261_;
v___y_596_ = v___y_1260_;
v___y_597_ = v___y_1262_;
v___y_598_ = v___y_1263_;
v___y_599_ = v___y_1265_;
v___y_600_ = v___y_1264_;
v___y_601_ = v___y_1267_;
v___y_602_ = v___y_1269_;
v___y_603_ = v___y_1268_;
v___y_604_ = v___x_1275_;
goto v___jp_583_;
}
}
v___jp_1276_:
{
lean_object* v___x_1299_; lean_object* v___x_1300_; lean_object* v___x_1301_; lean_object* v___x_1302_; lean_object* v___x_1303_; lean_object* v___x_1304_; lean_object* v___x_1305_; lean_object* v___x_1306_; 
lean_inc_ref(v___y_1280_);
v___x_1299_ = l_Array_append___redArg(v___y_1280_, v___y_1298_);
lean_dec_ref(v___y_1298_);
lean_inc_n(v___y_1295_, 2);
lean_inc_n(v___y_1289_, 5);
v___x_1300_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1300_, 0, v___y_1289_);
lean_ctor_set(v___x_1300_, 1, v___y_1295_);
lean_ctor_set(v___x_1300_, 2, v___x_1299_);
v___x_1301_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_scopedNS___closed__20));
lean_inc_ref(v___y_1293_);
lean_inc_ref(v___y_1278_);
lean_inc_ref(v___y_1291_);
v___x_1302_ = l_Lean_Name_mkStr4(v___y_1291_, v___y_1278_, v___y_1293_, v___x_1301_);
v___x_1303_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1303_, 0, v___y_1289_);
lean_ctor_set(v___x_1303_, 1, v___x_1301_);
v___x_1304_ = l_Lean_Syntax_node1(v___y_1289_, v___x_1302_, v___x_1303_);
v___x_1305_ = l_Lean_Syntax_node1(v___y_1289_, v___y_1295_, v___x_1304_);
lean_inc(v___y_1282_);
v___x_1306_ = l_Lean_Syntax_node1(v___y_1289_, v___y_1282_, v___x_1305_);
if (lean_obj_tag(v___y_1292_) == 0)
{
lean_object* v___x_1307_; 
v___x_1307_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16);
v___y_1250_ = v___y_1277_;
v___y_1251_ = v___x_1300_;
v___y_1252_ = v___y_1278_;
v___y_1253_ = v___y_1279_;
v___y_1254_ = v___y_1280_;
v___y_1255_ = v___x_1306_;
v___y_1256_ = v___y_1281_;
v___y_1257_ = v___y_1283_;
v___y_1258_ = v___y_1284_;
v___y_1259_ = v___y_1285_;
v___y_1260_ = v___y_1287_;
v___y_1261_ = v___y_1286_;
v___y_1262_ = v___y_1288_;
v___y_1263_ = v___y_1289_;
v___y_1264_ = v___y_1290_;
v___y_1265_ = v___y_1291_;
v___y_1266_ = v___y_1294_;
v___y_1267_ = v___y_1295_;
v___y_1268_ = v___y_1297_;
v___y_1269_ = v___y_1296_;
v___y_1270_ = v___x_1307_;
goto v___jp_1249_;
}
else
{
lean_object* v_val_1308_; lean_object* v___x_1309_; 
v_val_1308_ = lean_ctor_get(v___y_1292_, 0);
lean_inc(v_val_1308_);
lean_dec_ref_known(v___y_1292_, 1);
v___x_1309_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__1(v_val_1308_);
v___y_1250_ = v___y_1277_;
v___y_1251_ = v___x_1300_;
v___y_1252_ = v___y_1278_;
v___y_1253_ = v___y_1279_;
v___y_1254_ = v___y_1280_;
v___y_1255_ = v___x_1306_;
v___y_1256_ = v___y_1281_;
v___y_1257_ = v___y_1283_;
v___y_1258_ = v___y_1284_;
v___y_1259_ = v___y_1285_;
v___y_1260_ = v___y_1287_;
v___y_1261_ = v___y_1286_;
v___y_1262_ = v___y_1288_;
v___y_1263_ = v___y_1289_;
v___y_1264_ = v___y_1290_;
v___y_1265_ = v___y_1291_;
v___y_1266_ = v___y_1294_;
v___y_1267_ = v___y_1295_;
v___y_1268_ = v___y_1297_;
v___y_1269_ = v___y_1296_;
v___y_1270_ = v___x_1309_;
goto v___jp_1249_;
}
}
v___jp_1310_:
{
lean_object* v___x_1333_; lean_object* v___x_1334_; 
lean_inc_ref(v___y_1313_);
v___x_1333_ = l_Array_append___redArg(v___y_1313_, v___y_1332_);
lean_dec_ref(v___y_1332_);
lean_inc(v___y_1329_);
lean_inc(v___y_1323_);
v___x_1334_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1334_, 0, v___y_1323_);
lean_ctor_set(v___x_1334_, 1, v___y_1329_);
lean_ctor_set(v___x_1334_, 2, v___x_1333_);
if (lean_obj_tag(v___y_1317_) == 0)
{
lean_object* v___x_1335_; 
v___x_1335_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__16);
v___y_1277_ = v___y_1311_;
v___y_1278_ = v___y_1312_;
v___y_1279_ = v___x_1334_;
v___y_1280_ = v___y_1313_;
v___y_1281_ = v___y_1314_;
v___y_1282_ = v___y_1315_;
v___y_1283_ = v___y_1316_;
v___y_1284_ = v___y_1318_;
v___y_1285_ = v___y_1319_;
v___y_1286_ = v___y_1320_;
v___y_1287_ = v___y_1321_;
v___y_1288_ = v___y_1322_;
v___y_1289_ = v___y_1323_;
v___y_1290_ = v___y_1325_;
v___y_1291_ = v___y_1324_;
v___y_1292_ = v___y_1326_;
v___y_1293_ = v___y_1327_;
v___y_1294_ = v___y_1328_;
v___y_1295_ = v___y_1329_;
v___y_1296_ = v___y_1331_;
v___y_1297_ = v___y_1330_;
v___y_1298_ = v___x_1335_;
goto v___jp_1276_;
}
else
{
lean_object* v_val_1336_; lean_object* v___x_1337_; 
v_val_1336_ = lean_ctor_get(v___y_1317_, 0);
lean_inc(v_val_1336_);
lean_dec_ref_known(v___y_1317_, 1);
v___x_1337_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__1(v_val_1336_);
v___y_1277_ = v___y_1311_;
v___y_1278_ = v___y_1312_;
v___y_1279_ = v___x_1334_;
v___y_1280_ = v___y_1313_;
v___y_1281_ = v___y_1314_;
v___y_1282_ = v___y_1315_;
v___y_1283_ = v___y_1316_;
v___y_1284_ = v___y_1318_;
v___y_1285_ = v___y_1319_;
v___y_1286_ = v___y_1320_;
v___y_1287_ = v___y_1321_;
v___y_1288_ = v___y_1322_;
v___y_1289_ = v___y_1323_;
v___y_1290_ = v___y_1325_;
v___y_1291_ = v___y_1324_;
v___y_1292_ = v___y_1326_;
v___y_1293_ = v___y_1327_;
v___y_1294_ = v___y_1328_;
v___y_1295_ = v___y_1329_;
v___y_1296_ = v___y_1331_;
v___y_1297_ = v___y_1330_;
v___y_1298_ = v___x_1337_;
goto v___jp_1276_;
}
}
v___jp_1339_:
{
lean_object* v_ref_1358_; lean_object* v___x_1359_; lean_object* v___x_1360_; lean_object* v___x_1361_; lean_object* v___x_1362_; lean_object* v___x_1363_; lean_object* v___x_1364_; lean_object* v___x_1365_; lean_object* v___x_1366_; lean_object* v___x_1367_; lean_object* v___x_1368_; lean_object* v___x_1369_; lean_object* v___x_1370_; lean_object* v___x_1371_; 
v_ref_1358_ = lean_ctor_get(v___y_1343_, 5);
v___x_1359_ = l_Lean_SourceInfo_fromRef(v_ref_1358_, v___y_1352_);
v___x_1360_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__1));
v___x_1361_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__2));
v___x_1362_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__4));
lean_inc_ref(v___y_1350_);
lean_inc_ref(v___y_1353_);
v___x_1363_ = l_Lean_Name_mkStr4(v___y_1353_, v___x_1361_, v___y_1350_, v___x_1362_);
v___x_1364_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__6));
lean_inc(v___x_1359_);
v___x_1365_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1365_, 0, v___x_1359_);
lean_ctor_set(v___x_1365_, 1, v___x_1364_);
v___x_1366_ = l_Lean_rootNamespace;
v___x_1367_ = l_Lean_TSyntax_getId(v___y_1341_);
v___x_1368_ = l_Lean_Name_append(v___x_1366_, v___x_1367_);
v___x_1369_ = l_Lean_mkIdentFrom(v___y_1341_, v___x_1368_, v___y_1352_);
lean_dec(v___y_1341_);
v___x_1370_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__1));
v___x_1371_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__9, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__9_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__9);
if (lean_obj_tag(v___y_1351_) == 1)
{
lean_object* v_val_1372_; lean_object* v___x_1373_; 
v_val_1372_ = lean_ctor_get(v___y_1351_, 0);
lean_inc(v_val_1372_);
lean_dec_ref_known(v___y_1351_, 1);
v___x_1373_ = l_Array_mkArray1___redArg(v_val_1372_);
v___y_1133_ = v___x_1369_;
v___y_1134_ = v___y_1340_;
v___y_1135_ = v___x_1360_;
v___y_1136_ = v___y_1342_;
v___y_1137_ = v___y_1344_;
v___y_1138_ = v___x_1365_;
v___y_1139_ = v___x_1371_;
v___y_1140_ = v___x_1370_;
v___y_1141_ = v___y_1345_;
v___y_1142_ = v___y_1346_;
v___y_1143_ = v___y_1347_;
v___y_1144_ = v___y_1348_;
v___y_1145_ = v___y_1349_;
v___y_1146_ = v___y_1350_;
v___y_1147_ = v___x_1363_;
v___y_1148_ = v___y_1353_;
v___y_1149_ = v___y_1354_;
v___y_1150_ = v___y_1357_;
v___y_1151_ = v___y_1356_;
v___y_1152_ = v___y_1355_;
v___y_1153_ = v___x_1359_;
v___y_1154_ = v___x_1373_;
goto v___jp_1132_;
}
else
{
lean_object* v___x_1374_; 
lean_dec(v___y_1351_);
v___x_1374_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__0___closed__0));
v___y_1133_ = v___x_1369_;
v___y_1134_ = v___y_1340_;
v___y_1135_ = v___x_1360_;
v___y_1136_ = v___y_1342_;
v___y_1137_ = v___y_1344_;
v___y_1138_ = v___x_1365_;
v___y_1139_ = v___x_1371_;
v___y_1140_ = v___x_1370_;
v___y_1141_ = v___y_1345_;
v___y_1142_ = v___y_1346_;
v___y_1143_ = v___y_1347_;
v___y_1144_ = v___y_1348_;
v___y_1145_ = v___y_1349_;
v___y_1146_ = v___y_1350_;
v___y_1147_ = v___x_1363_;
v___y_1148_ = v___y_1353_;
v___y_1149_ = v___y_1354_;
v___y_1150_ = v___y_1357_;
v___y_1151_ = v___y_1356_;
v___y_1152_ = v___y_1355_;
v___y_1153_ = v___x_1359_;
v___y_1154_ = v___x_1374_;
goto v___jp_1132_;
}
}
v___jp_1375_:
{
lean_object* v___x_1394_; 
v___x_1394_ = l_Lean_Syntax_getOptional_x3f(v___y_1382_);
lean_dec(v___y_1382_);
if (lean_obj_tag(v___x_1394_) == 0)
{
lean_object* v___x_1395_; 
v___x_1395_ = lean_box(0);
v___y_1340_ = v___y_1376_;
v___y_1341_ = v___y_1377_;
v___y_1342_ = v___y_1378_;
v___y_1343_ = v___y_1379_;
v___y_1344_ = v___y_1380_;
v___y_1345_ = v___y_1381_;
v___y_1346_ = v___y_1393_;
v___y_1347_ = v___y_1383_;
v___y_1348_ = v___y_1384_;
v___y_1349_ = v___y_1385_;
v___y_1350_ = v___y_1386_;
v___y_1351_ = v___y_1387_;
v___y_1352_ = v___y_1388_;
v___y_1353_ = v___y_1389_;
v___y_1354_ = v___y_1390_;
v___y_1355_ = v___y_1392_;
v___y_1356_ = v___y_1391_;
v___y_1357_ = v___x_1395_;
goto v___jp_1339_;
}
else
{
lean_object* v_val_1396_; lean_object* v___x_1398_; uint8_t v_isShared_1399_; uint8_t v_isSharedCheck_1403_; 
v_val_1396_ = lean_ctor_get(v___x_1394_, 0);
v_isSharedCheck_1403_ = !lean_is_exclusive(v___x_1394_);
if (v_isSharedCheck_1403_ == 0)
{
v___x_1398_ = v___x_1394_;
v_isShared_1399_ = v_isSharedCheck_1403_;
goto v_resetjp_1397_;
}
else
{
lean_inc(v_val_1396_);
lean_dec(v___x_1394_);
v___x_1398_ = lean_box(0);
v_isShared_1399_ = v_isSharedCheck_1403_;
goto v_resetjp_1397_;
}
v_resetjp_1397_:
{
lean_object* v___x_1401_; 
if (v_isShared_1399_ == 0)
{
v___x_1401_ = v___x_1398_;
goto v_reusejp_1400_;
}
else
{
lean_object* v_reuseFailAlloc_1402_; 
v_reuseFailAlloc_1402_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1402_, 0, v_val_1396_);
v___x_1401_ = v_reuseFailAlloc_1402_;
goto v_reusejp_1400_;
}
v_reusejp_1400_:
{
v___y_1340_ = v___y_1376_;
v___y_1341_ = v___y_1377_;
v___y_1342_ = v___y_1378_;
v___y_1343_ = v___y_1379_;
v___y_1344_ = v___y_1380_;
v___y_1345_ = v___y_1381_;
v___y_1346_ = v___y_1393_;
v___y_1347_ = v___y_1383_;
v___y_1348_ = v___y_1384_;
v___y_1349_ = v___y_1385_;
v___y_1350_ = v___y_1386_;
v___y_1351_ = v___y_1387_;
v___y_1352_ = v___y_1388_;
v___y_1353_ = v___y_1389_;
v___y_1354_ = v___y_1390_;
v___y_1355_ = v___y_1392_;
v___y_1356_ = v___y_1391_;
v___y_1357_ = v___x_1401_;
goto v___jp_1339_;
}
}
}
}
v___jp_1404_:
{
lean_object* v___x_1423_; 
v___x_1423_ = l_Lean_Syntax_getOptional_x3f(v___y_1420_);
lean_dec(v___y_1420_);
if (lean_obj_tag(v___x_1423_) == 0)
{
lean_object* v___x_1424_; 
v___x_1424_ = lean_box(0);
v___y_1376_ = v___y_1405_;
v___y_1377_ = v___y_1406_;
v___y_1378_ = v___y_1407_;
v___y_1379_ = v___y_1408_;
v___y_1380_ = v___y_1409_;
v___y_1381_ = v___y_1410_;
v___y_1382_ = v___y_1411_;
v___y_1383_ = v___y_1412_;
v___y_1384_ = v___y_1413_;
v___y_1385_ = v___y_1414_;
v___y_1386_ = v___y_1415_;
v___y_1387_ = v___y_1416_;
v___y_1388_ = v___y_1417_;
v___y_1389_ = v___y_1418_;
v___y_1390_ = v___y_1419_;
v___y_1391_ = v___y_1422_;
v___y_1392_ = v___y_1421_;
v___y_1393_ = v___x_1424_;
goto v___jp_1375_;
}
else
{
lean_object* v_val_1425_; lean_object* v___x_1427_; uint8_t v_isShared_1428_; uint8_t v_isSharedCheck_1432_; 
v_val_1425_ = lean_ctor_get(v___x_1423_, 0);
v_isSharedCheck_1432_ = !lean_is_exclusive(v___x_1423_);
if (v_isSharedCheck_1432_ == 0)
{
v___x_1427_ = v___x_1423_;
v_isShared_1428_ = v_isSharedCheck_1432_;
goto v_resetjp_1426_;
}
else
{
lean_inc(v_val_1425_);
lean_dec(v___x_1423_);
v___x_1427_ = lean_box(0);
v_isShared_1428_ = v_isSharedCheck_1432_;
goto v_resetjp_1426_;
}
v_resetjp_1426_:
{
lean_object* v___x_1430_; 
if (v_isShared_1428_ == 0)
{
v___x_1430_ = v___x_1427_;
goto v_reusejp_1429_;
}
else
{
lean_object* v_reuseFailAlloc_1431_; 
v_reuseFailAlloc_1431_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1431_, 0, v_val_1425_);
v___x_1430_ = v_reuseFailAlloc_1431_;
goto v_reusejp_1429_;
}
v_reusejp_1429_:
{
v___y_1376_ = v___y_1405_;
v___y_1377_ = v___y_1406_;
v___y_1378_ = v___y_1407_;
v___y_1379_ = v___y_1408_;
v___y_1380_ = v___y_1409_;
v___y_1381_ = v___y_1410_;
v___y_1382_ = v___y_1411_;
v___y_1383_ = v___y_1412_;
v___y_1384_ = v___y_1413_;
v___y_1385_ = v___y_1414_;
v___y_1386_ = v___y_1415_;
v___y_1387_ = v___y_1416_;
v___y_1388_ = v___y_1417_;
v___y_1389_ = v___y_1418_;
v___y_1390_ = v___y_1419_;
v___y_1391_ = v___y_1422_;
v___y_1392_ = v___y_1421_;
v___y_1393_ = v___x_1430_;
goto v___jp_1375_;
}
}
}
}
v___jp_1433_:
{
lean_object* v_ref_1452_; lean_object* v___x_1453_; lean_object* v___x_1454_; lean_object* v___x_1455_; lean_object* v___x_1456_; lean_object* v___x_1457_; lean_object* v___x_1458_; lean_object* v___x_1459_; lean_object* v___x_1460_; lean_object* v___x_1461_; lean_object* v___x_1462_; lean_object* v___x_1463_; lean_object* v___x_1464_; lean_object* v___x_1465_; 
v_ref_1452_ = lean_ctor_get(v___y_1436_, 5);
v___x_1453_ = l_Lean_SourceInfo_fromRef(v_ref_1452_, v___y_1445_);
v___x_1454_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__1));
v___x_1455_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__2));
v___x_1456_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__4));
lean_inc_ref(v___y_1443_);
lean_inc_ref(v___y_1448_);
v___x_1457_ = l_Lean_Name_mkStr4(v___y_1448_, v___x_1455_, v___y_1443_, v___x_1456_);
v___x_1458_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__6));
lean_inc(v___x_1453_);
v___x_1459_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1459_, 0, v___x_1453_);
lean_ctor_set(v___x_1459_, 1, v___x_1458_);
v___x_1460_ = l_Lean_rootNamespace;
v___x_1461_ = l_Lean_TSyntax_getId(v___y_1435_);
v___x_1462_ = l_Lean_Name_append(v___x_1460_, v___x_1461_);
v___x_1463_ = l_Lean_mkIdentFrom(v___y_1435_, v___x_1462_, v___y_1445_);
lean_dec(v___y_1435_);
v___x_1464_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__1));
v___x_1465_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__9, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__9_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__9);
if (lean_obj_tag(v___y_1446_) == 1)
{
lean_object* v_val_1466_; lean_object* v___x_1467_; 
v_val_1466_ = lean_ctor_get(v___y_1446_, 0);
lean_inc(v_val_1466_);
lean_dec_ref_known(v___y_1446_, 1);
v___x_1467_ = l_Array_mkArray1___redArg(v_val_1466_);
v___y_1222_ = v___x_1463_;
v___y_1223_ = v___x_1464_;
v___y_1224_ = v___y_1434_;
v___y_1225_ = v___x_1453_;
v___y_1226_ = v___x_1454_;
v___y_1227_ = v___x_1465_;
v___y_1228_ = v___x_1457_;
v___y_1229_ = v___y_1437_;
v___y_1230_ = v___y_1438_;
v___y_1231_ = v___y_1439_;
v___y_1232_ = v___y_1440_;
v___y_1233_ = v___y_1441_;
v___y_1234_ = v___x_1459_;
v___y_1235_ = v___y_1442_;
v___y_1236_ = v___y_1443_;
v___y_1237_ = v___y_1444_;
v___y_1238_ = v___y_1451_;
v___y_1239_ = v___y_1447_;
v___y_1240_ = v___y_1448_;
v___y_1241_ = v___y_1449_;
v___y_1242_ = v___y_1450_;
v___y_1243_ = v___x_1467_;
goto v___jp_1221_;
}
else
{
lean_object* v___x_1468_; 
lean_dec(v___y_1446_);
v___x_1468_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__0___closed__0));
v___y_1222_ = v___x_1463_;
v___y_1223_ = v___x_1464_;
v___y_1224_ = v___y_1434_;
v___y_1225_ = v___x_1453_;
v___y_1226_ = v___x_1454_;
v___y_1227_ = v___x_1465_;
v___y_1228_ = v___x_1457_;
v___y_1229_ = v___y_1437_;
v___y_1230_ = v___y_1438_;
v___y_1231_ = v___y_1439_;
v___y_1232_ = v___y_1440_;
v___y_1233_ = v___y_1441_;
v___y_1234_ = v___x_1459_;
v___y_1235_ = v___y_1442_;
v___y_1236_ = v___y_1443_;
v___y_1237_ = v___y_1444_;
v___y_1238_ = v___y_1451_;
v___y_1239_ = v___y_1447_;
v___y_1240_ = v___y_1448_;
v___y_1241_ = v___y_1449_;
v___y_1242_ = v___y_1450_;
v___y_1243_ = v___x_1468_;
goto v___jp_1221_;
}
}
v___jp_1469_:
{
lean_object* v___x_1488_; 
v___x_1488_ = l_Lean_Syntax_getOptional_x3f(v___y_1475_);
lean_dec(v___y_1475_);
if (lean_obj_tag(v___x_1488_) == 0)
{
lean_object* v___x_1489_; 
v___x_1489_ = lean_box(0);
v___y_1434_ = v___y_1470_;
v___y_1435_ = v___y_1471_;
v___y_1436_ = v___y_1472_;
v___y_1437_ = v___y_1473_;
v___y_1438_ = v___y_1474_;
v___y_1439_ = v___y_1476_;
v___y_1440_ = v___y_1477_;
v___y_1441_ = v___y_1478_;
v___y_1442_ = v___y_1479_;
v___y_1443_ = v___y_1480_;
v___y_1444_ = v___y_1487_;
v___y_1445_ = v___y_1481_;
v___y_1446_ = v___y_1482_;
v___y_1447_ = v___y_1483_;
v___y_1448_ = v___y_1484_;
v___y_1449_ = v___y_1485_;
v___y_1450_ = v___y_1486_;
v___y_1451_ = v___x_1489_;
goto v___jp_1433_;
}
else
{
lean_object* v_val_1490_; lean_object* v___x_1492_; uint8_t v_isShared_1493_; uint8_t v_isSharedCheck_1497_; 
v_val_1490_ = lean_ctor_get(v___x_1488_, 0);
v_isSharedCheck_1497_ = !lean_is_exclusive(v___x_1488_);
if (v_isSharedCheck_1497_ == 0)
{
v___x_1492_ = v___x_1488_;
v_isShared_1493_ = v_isSharedCheck_1497_;
goto v_resetjp_1491_;
}
else
{
lean_inc(v_val_1490_);
lean_dec(v___x_1488_);
v___x_1492_ = lean_box(0);
v_isShared_1493_ = v_isSharedCheck_1497_;
goto v_resetjp_1491_;
}
v_resetjp_1491_:
{
lean_object* v___x_1495_; 
if (v_isShared_1493_ == 0)
{
v___x_1495_ = v___x_1492_;
goto v_reusejp_1494_;
}
else
{
lean_object* v_reuseFailAlloc_1496_; 
v_reuseFailAlloc_1496_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1496_, 0, v_val_1490_);
v___x_1495_ = v_reuseFailAlloc_1496_;
goto v_reusejp_1494_;
}
v_reusejp_1494_:
{
v___y_1434_ = v___y_1470_;
v___y_1435_ = v___y_1471_;
v___y_1436_ = v___y_1472_;
v___y_1437_ = v___y_1473_;
v___y_1438_ = v___y_1474_;
v___y_1439_ = v___y_1476_;
v___y_1440_ = v___y_1477_;
v___y_1441_ = v___y_1478_;
v___y_1442_ = v___y_1479_;
v___y_1443_ = v___y_1480_;
v___y_1444_ = v___y_1487_;
v___y_1445_ = v___y_1481_;
v___y_1446_ = v___y_1482_;
v___y_1447_ = v___y_1483_;
v___y_1448_ = v___y_1484_;
v___y_1449_ = v___y_1485_;
v___y_1450_ = v___y_1486_;
v___y_1451_ = v___x_1495_;
goto v___jp_1433_;
}
}
}
}
v___jp_1498_:
{
lean_object* v___x_1517_; 
v___x_1517_ = l_Lean_Syntax_getOptional_x3f(v___y_1499_);
lean_dec(v___y_1499_);
if (lean_obj_tag(v___x_1517_) == 0)
{
lean_object* v___x_1518_; 
v___x_1518_ = lean_box(0);
v___y_1470_ = v___y_1500_;
v___y_1471_ = v___y_1501_;
v___y_1472_ = v___y_1502_;
v___y_1473_ = v___y_1503_;
v___y_1474_ = v___y_1516_;
v___y_1475_ = v___y_1504_;
v___y_1476_ = v___y_1505_;
v___y_1477_ = v___y_1506_;
v___y_1478_ = v___y_1507_;
v___y_1479_ = v___y_1508_;
v___y_1480_ = v___y_1509_;
v___y_1481_ = v___y_1510_;
v___y_1482_ = v___y_1511_;
v___y_1483_ = v___y_1512_;
v___y_1484_ = v___y_1513_;
v___y_1485_ = v___y_1514_;
v___y_1486_ = v___y_1515_;
v___y_1487_ = v___x_1518_;
goto v___jp_1469_;
}
else
{
lean_object* v_val_1519_; lean_object* v___x_1521_; uint8_t v_isShared_1522_; uint8_t v_isSharedCheck_1526_; 
v_val_1519_ = lean_ctor_get(v___x_1517_, 0);
v_isSharedCheck_1526_ = !lean_is_exclusive(v___x_1517_);
if (v_isSharedCheck_1526_ == 0)
{
v___x_1521_ = v___x_1517_;
v_isShared_1522_ = v_isSharedCheck_1526_;
goto v_resetjp_1520_;
}
else
{
lean_inc(v_val_1519_);
lean_dec(v___x_1517_);
v___x_1521_ = lean_box(0);
v_isShared_1522_ = v_isSharedCheck_1526_;
goto v_resetjp_1520_;
}
v_resetjp_1520_:
{
lean_object* v___x_1524_; 
if (v_isShared_1522_ == 0)
{
v___x_1524_ = v___x_1521_;
goto v_reusejp_1523_;
}
else
{
lean_object* v_reuseFailAlloc_1525_; 
v_reuseFailAlloc_1525_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1525_, 0, v_val_1519_);
v___x_1524_ = v_reuseFailAlloc_1525_;
goto v_reusejp_1523_;
}
v_reusejp_1523_:
{
v___y_1470_ = v___y_1500_;
v___y_1471_ = v___y_1501_;
v___y_1472_ = v___y_1502_;
v___y_1473_ = v___y_1503_;
v___y_1474_ = v___y_1516_;
v___y_1475_ = v___y_1504_;
v___y_1476_ = v___y_1505_;
v___y_1477_ = v___y_1506_;
v___y_1478_ = v___y_1507_;
v___y_1479_ = v___y_1508_;
v___y_1480_ = v___y_1509_;
v___y_1481_ = v___y_1510_;
v___y_1482_ = v___y_1511_;
v___y_1483_ = v___y_1512_;
v___y_1484_ = v___y_1513_;
v___y_1485_ = v___y_1514_;
v___y_1486_ = v___y_1515_;
v___y_1487_ = v___x_1524_;
goto v___jp_1469_;
}
}
}
}
v___jp_1527_:
{
lean_object* v_ref_1546_; lean_object* v___x_1547_; lean_object* v___x_1548_; lean_object* v___x_1549_; lean_object* v___x_1550_; lean_object* v___x_1551_; lean_object* v___x_1552_; lean_object* v___x_1553_; lean_object* v___x_1554_; lean_object* v___x_1555_; lean_object* v___x_1556_; lean_object* v___x_1557_; lean_object* v___x_1558_; lean_object* v___x_1559_; 
v_ref_1546_ = lean_ctor_get(v___y_1531_, 5);
v___x_1547_ = l_Lean_SourceInfo_fromRef(v_ref_1546_, v___y_1538_);
v___x_1548_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__1));
v___x_1549_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__2));
v___x_1550_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__4));
lean_inc_ref(v___y_1537_);
lean_inc_ref(v___y_1542_);
v___x_1551_ = l_Lean_Name_mkStr4(v___y_1542_, v___x_1549_, v___y_1537_, v___x_1550_);
v___x_1552_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__6));
lean_inc(v___x_1547_);
v___x_1553_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1553_, 0, v___x_1547_);
lean_ctor_set(v___x_1553_, 1, v___x_1552_);
v___x_1554_ = l_Lean_rootNamespace;
v___x_1555_ = l_Lean_TSyntax_getId(v___y_1530_);
v___x_1556_ = l_Lean_Name_append(v___x_1554_, v___x_1555_);
v___x_1557_ = l_Lean_mkIdentFrom(v___y_1530_, v___x_1556_, v___y_1538_);
lean_dec(v___y_1530_);
v___x_1558_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__1));
v___x_1559_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__9, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__9_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__9);
if (lean_obj_tag(v___y_1539_) == 1)
{
lean_object* v_val_1560_; lean_object* v___x_1561_; 
v_val_1560_ = lean_ctor_get(v___y_1539_, 0);
lean_inc(v_val_1560_);
lean_dec_ref_known(v___y_1539_, 1);
v___x_1561_ = l_Array_mkArray1___redArg(v_val_1560_);
v___y_1044_ = v___x_1559_;
v___y_1045_ = v___x_1551_;
v___y_1046_ = v___x_1558_;
v___y_1047_ = v___y_1528_;
v___y_1048_ = v___y_1529_;
v___y_1049_ = v___y_1532_;
v___y_1050_ = v___y_1545_;
v___y_1051_ = v___x_1557_;
v___y_1052_ = v___y_1533_;
v___y_1053_ = v___y_1534_;
v___y_1054_ = v___y_1535_;
v___y_1055_ = v___x_1548_;
v___y_1056_ = v___y_1536_;
v___y_1057_ = v___x_1553_;
v___y_1058_ = v___y_1537_;
v___y_1059_ = v___y_1540_;
v___y_1060_ = v___y_1541_;
v___y_1061_ = v___y_1542_;
v___y_1062_ = v___x_1547_;
v___y_1063_ = v___y_1543_;
v___y_1064_ = v___y_1544_;
v___y_1065_ = v___x_1561_;
goto v___jp_1043_;
}
else
{
lean_object* v___x_1562_; 
lean_dec(v___y_1539_);
v___x_1562_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__0___closed__0));
v___y_1044_ = v___x_1559_;
v___y_1045_ = v___x_1551_;
v___y_1046_ = v___x_1558_;
v___y_1047_ = v___y_1528_;
v___y_1048_ = v___y_1529_;
v___y_1049_ = v___y_1532_;
v___y_1050_ = v___y_1545_;
v___y_1051_ = v___x_1557_;
v___y_1052_ = v___y_1533_;
v___y_1053_ = v___y_1534_;
v___y_1054_ = v___y_1535_;
v___y_1055_ = v___x_1548_;
v___y_1056_ = v___y_1536_;
v___y_1057_ = v___x_1553_;
v___y_1058_ = v___y_1537_;
v___y_1059_ = v___y_1540_;
v___y_1060_ = v___y_1541_;
v___y_1061_ = v___y_1542_;
v___y_1062_ = v___x_1547_;
v___y_1063_ = v___y_1543_;
v___y_1064_ = v___y_1544_;
v___y_1065_ = v___x_1562_;
goto v___jp_1043_;
}
}
v___jp_1563_:
{
lean_object* v___x_1582_; 
v___x_1582_ = l_Lean_Syntax_getOptional_x3f(v___y_1571_);
lean_dec(v___y_1571_);
if (lean_obj_tag(v___x_1582_) == 0)
{
lean_object* v___x_1583_; 
v___x_1583_ = lean_box(0);
v___y_1528_ = v___y_1564_;
v___y_1529_ = v___y_1565_;
v___y_1530_ = v___y_1566_;
v___y_1531_ = v___y_1567_;
v___y_1532_ = v___y_1568_;
v___y_1533_ = v___y_1569_;
v___y_1534_ = v___y_1570_;
v___y_1535_ = v___y_1572_;
v___y_1536_ = v___y_1573_;
v___y_1537_ = v___y_1574_;
v___y_1538_ = v___y_1575_;
v___y_1539_ = v___y_1576_;
v___y_1540_ = v___y_1577_;
v___y_1541_ = v___y_1581_;
v___y_1542_ = v___y_1578_;
v___y_1543_ = v___y_1579_;
v___y_1544_ = v___y_1580_;
v___y_1545_ = v___x_1583_;
goto v___jp_1527_;
}
else
{
lean_object* v_val_1584_; lean_object* v___x_1586_; uint8_t v_isShared_1587_; uint8_t v_isSharedCheck_1591_; 
v_val_1584_ = lean_ctor_get(v___x_1582_, 0);
v_isSharedCheck_1591_ = !lean_is_exclusive(v___x_1582_);
if (v_isSharedCheck_1591_ == 0)
{
v___x_1586_ = v___x_1582_;
v_isShared_1587_ = v_isSharedCheck_1591_;
goto v_resetjp_1585_;
}
else
{
lean_inc(v_val_1584_);
lean_dec(v___x_1582_);
v___x_1586_ = lean_box(0);
v_isShared_1587_ = v_isSharedCheck_1591_;
goto v_resetjp_1585_;
}
v_resetjp_1585_:
{
lean_object* v___x_1589_; 
if (v_isShared_1587_ == 0)
{
v___x_1589_ = v___x_1586_;
goto v_reusejp_1588_;
}
else
{
lean_object* v_reuseFailAlloc_1590_; 
v_reuseFailAlloc_1590_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1590_, 0, v_val_1584_);
v___x_1589_ = v_reuseFailAlloc_1590_;
goto v_reusejp_1588_;
}
v_reusejp_1588_:
{
v___y_1528_ = v___y_1564_;
v___y_1529_ = v___y_1565_;
v___y_1530_ = v___y_1566_;
v___y_1531_ = v___y_1567_;
v___y_1532_ = v___y_1568_;
v___y_1533_ = v___y_1569_;
v___y_1534_ = v___y_1570_;
v___y_1535_ = v___y_1572_;
v___y_1536_ = v___y_1573_;
v___y_1537_ = v___y_1574_;
v___y_1538_ = v___y_1575_;
v___y_1539_ = v___y_1576_;
v___y_1540_ = v___y_1577_;
v___y_1541_ = v___y_1581_;
v___y_1542_ = v___y_1578_;
v___y_1543_ = v___y_1579_;
v___y_1544_ = v___y_1580_;
v___y_1545_ = v___x_1589_;
goto v___jp_1527_;
}
}
}
}
v___jp_1592_:
{
lean_object* v___x_1611_; 
v___x_1611_ = l_Lean_Syntax_getOptional_x3f(v___y_1603_);
lean_dec(v___y_1603_);
if (lean_obj_tag(v___x_1611_) == 0)
{
lean_object* v___x_1612_; 
v___x_1612_ = lean_box(0);
v___y_1564_ = v___y_1593_;
v___y_1565_ = v___y_1594_;
v___y_1566_ = v___y_1595_;
v___y_1567_ = v___y_1596_;
v___y_1568_ = v___y_1597_;
v___y_1569_ = v___y_1598_;
v___y_1570_ = v___y_1610_;
v___y_1571_ = v___y_1599_;
v___y_1572_ = v___y_1600_;
v___y_1573_ = v___y_1601_;
v___y_1574_ = v___y_1602_;
v___y_1575_ = v___y_1604_;
v___y_1576_ = v___y_1605_;
v___y_1577_ = v___y_1606_;
v___y_1578_ = v___y_1607_;
v___y_1579_ = v___y_1608_;
v___y_1580_ = v___y_1609_;
v___y_1581_ = v___x_1612_;
goto v___jp_1563_;
}
else
{
lean_object* v_val_1613_; lean_object* v___x_1615_; uint8_t v_isShared_1616_; uint8_t v_isSharedCheck_1620_; 
v_val_1613_ = lean_ctor_get(v___x_1611_, 0);
v_isSharedCheck_1620_ = !lean_is_exclusive(v___x_1611_);
if (v_isSharedCheck_1620_ == 0)
{
v___x_1615_ = v___x_1611_;
v_isShared_1616_ = v_isSharedCheck_1620_;
goto v_resetjp_1614_;
}
else
{
lean_inc(v_val_1613_);
lean_dec(v___x_1611_);
v___x_1615_ = lean_box(0);
v_isShared_1616_ = v_isSharedCheck_1620_;
goto v_resetjp_1614_;
}
v_resetjp_1614_:
{
lean_object* v___x_1618_; 
if (v_isShared_1616_ == 0)
{
v___x_1618_ = v___x_1615_;
goto v_reusejp_1617_;
}
else
{
lean_object* v_reuseFailAlloc_1619_; 
v_reuseFailAlloc_1619_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1619_, 0, v_val_1613_);
v___x_1618_ = v_reuseFailAlloc_1619_;
goto v_reusejp_1617_;
}
v_reusejp_1617_:
{
v___y_1564_ = v___y_1593_;
v___y_1565_ = v___y_1594_;
v___y_1566_ = v___y_1595_;
v___y_1567_ = v___y_1596_;
v___y_1568_ = v___y_1597_;
v___y_1569_ = v___y_1598_;
v___y_1570_ = v___y_1610_;
v___y_1571_ = v___y_1599_;
v___y_1572_ = v___y_1600_;
v___y_1573_ = v___y_1601_;
v___y_1574_ = v___y_1602_;
v___y_1575_ = v___y_1604_;
v___y_1576_ = v___y_1605_;
v___y_1577_ = v___y_1606_;
v___y_1578_ = v___y_1607_;
v___y_1579_ = v___y_1608_;
v___y_1580_ = v___y_1609_;
v___y_1581_ = v___x_1618_;
goto v___jp_1563_;
}
}
}
}
v___jp_1621_:
{
lean_object* v_ref_1640_; lean_object* v___x_1641_; lean_object* v___x_1642_; lean_object* v___x_1643_; lean_object* v___x_1644_; lean_object* v___x_1645_; lean_object* v___x_1646_; lean_object* v___x_1647_; lean_object* v___x_1648_; lean_object* v___x_1649_; lean_object* v___x_1650_; lean_object* v___x_1651_; lean_object* v___x_1652_; lean_object* v___x_1653_; 
v_ref_1640_ = lean_ctor_get(v___y_1624_, 5);
v___x_1641_ = l_Lean_SourceInfo_fromRef(v_ref_1640_, v___y_1627_);
v___x_1642_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__1));
v___x_1643_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__2));
v___x_1644_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__4));
lean_inc_ref(v___y_1630_);
lean_inc_ref(v___y_1633_);
v___x_1645_ = l_Lean_Name_mkStr4(v___y_1633_, v___x_1643_, v___y_1630_, v___x_1644_);
v___x_1646_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__6));
lean_inc(v___x_1641_);
v___x_1647_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1647_, 0, v___x_1641_);
lean_ctor_set(v___x_1647_, 1, v___x_1646_);
v___x_1648_ = l_Lean_rootNamespace;
v___x_1649_ = l_Lean_TSyntax_getId(v___y_1623_);
v___x_1650_ = l_Lean_Name_append(v___x_1648_, v___x_1649_);
v___x_1651_ = l_Lean_mkIdentFrom(v___y_1623_, v___x_1650_, v___y_1627_);
lean_dec(v___y_1623_);
v___x_1652_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__1));
v___x_1653_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__9, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__9_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__9);
if (lean_obj_tag(v___y_1632_) == 1)
{
lean_object* v_val_1654_; lean_object* v___x_1655_; 
v_val_1654_ = lean_ctor_get(v___y_1632_, 0);
lean_inc(v_val_1654_);
lean_dec_ref_known(v___y_1632_, 1);
v___x_1655_ = l_Array_mkArray1___redArg(v_val_1654_);
v___y_1311_ = v___x_1642_;
v___y_1312_ = v___y_1622_;
v___y_1313_ = v___x_1653_;
v___y_1314_ = v___y_1625_;
v___y_1315_ = v___y_1626_;
v___y_1316_ = v___x_1647_;
v___y_1317_ = v___y_1639_;
v___y_1318_ = v___y_1628_;
v___y_1319_ = v___y_1629_;
v___y_1320_ = v___y_1630_;
v___y_1321_ = v___x_1645_;
v___y_1322_ = v___y_1631_;
v___y_1323_ = v___x_1641_;
v___y_1324_ = v___y_1633_;
v___y_1325_ = v___x_1651_;
v___y_1326_ = v___y_1634_;
v___y_1327_ = v___y_1635_;
v___y_1328_ = v___y_1636_;
v___y_1329_ = v___x_1652_;
v___y_1330_ = v___y_1638_;
v___y_1331_ = v___y_1637_;
v___y_1332_ = v___x_1655_;
goto v___jp_1310_;
}
else
{
lean_object* v___x_1656_; 
lean_dec(v___y_1632_);
v___x_1656_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__0___closed__0));
v___y_1311_ = v___x_1642_;
v___y_1312_ = v___y_1622_;
v___y_1313_ = v___x_1653_;
v___y_1314_ = v___y_1625_;
v___y_1315_ = v___y_1626_;
v___y_1316_ = v___x_1647_;
v___y_1317_ = v___y_1639_;
v___y_1318_ = v___y_1628_;
v___y_1319_ = v___y_1629_;
v___y_1320_ = v___y_1630_;
v___y_1321_ = v___x_1645_;
v___y_1322_ = v___y_1631_;
v___y_1323_ = v___x_1641_;
v___y_1324_ = v___y_1633_;
v___y_1325_ = v___x_1651_;
v___y_1326_ = v___y_1634_;
v___y_1327_ = v___y_1635_;
v___y_1328_ = v___y_1636_;
v___y_1329_ = v___x_1652_;
v___y_1330_ = v___y_1638_;
v___y_1331_ = v___y_1637_;
v___y_1332_ = v___x_1656_;
goto v___jp_1310_;
}
}
v___jp_1657_:
{
lean_object* v___x_1676_; 
v___x_1676_ = l_Lean_Syntax_getOptional_x3f(v___y_1664_);
lean_dec(v___y_1664_);
if (lean_obj_tag(v___x_1676_) == 0)
{
lean_object* v___x_1677_; 
v___x_1677_ = lean_box(0);
v___y_1622_ = v___y_1658_;
v___y_1623_ = v___y_1659_;
v___y_1624_ = v___y_1660_;
v___y_1625_ = v___y_1661_;
v___y_1626_ = v___y_1662_;
v___y_1627_ = v___y_1663_;
v___y_1628_ = v___y_1665_;
v___y_1629_ = v___y_1666_;
v___y_1630_ = v___y_1667_;
v___y_1631_ = v___y_1668_;
v___y_1632_ = v___y_1669_;
v___y_1633_ = v___y_1670_;
v___y_1634_ = v___y_1675_;
v___y_1635_ = v___y_1671_;
v___y_1636_ = v___y_1672_;
v___y_1637_ = v___y_1674_;
v___y_1638_ = v___y_1673_;
v___y_1639_ = v___x_1677_;
goto v___jp_1621_;
}
else
{
lean_object* v_val_1678_; lean_object* v___x_1680_; uint8_t v_isShared_1681_; uint8_t v_isSharedCheck_1685_; 
v_val_1678_ = lean_ctor_get(v___x_1676_, 0);
v_isSharedCheck_1685_ = !lean_is_exclusive(v___x_1676_);
if (v_isSharedCheck_1685_ == 0)
{
v___x_1680_ = v___x_1676_;
v_isShared_1681_ = v_isSharedCheck_1685_;
goto v_resetjp_1679_;
}
else
{
lean_inc(v_val_1678_);
lean_dec(v___x_1676_);
v___x_1680_ = lean_box(0);
v_isShared_1681_ = v_isSharedCheck_1685_;
goto v_resetjp_1679_;
}
v_resetjp_1679_:
{
lean_object* v___x_1683_; 
if (v_isShared_1681_ == 0)
{
v___x_1683_ = v___x_1680_;
goto v_reusejp_1682_;
}
else
{
lean_object* v_reuseFailAlloc_1684_; 
v_reuseFailAlloc_1684_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1684_, 0, v_val_1678_);
v___x_1683_ = v_reuseFailAlloc_1684_;
goto v_reusejp_1682_;
}
v_reusejp_1682_:
{
v___y_1622_ = v___y_1658_;
v___y_1623_ = v___y_1659_;
v___y_1624_ = v___y_1660_;
v___y_1625_ = v___y_1661_;
v___y_1626_ = v___y_1662_;
v___y_1627_ = v___y_1663_;
v___y_1628_ = v___y_1665_;
v___y_1629_ = v___y_1666_;
v___y_1630_ = v___y_1667_;
v___y_1631_ = v___y_1668_;
v___y_1632_ = v___y_1669_;
v___y_1633_ = v___y_1670_;
v___y_1634_ = v___y_1675_;
v___y_1635_ = v___y_1671_;
v___y_1636_ = v___y_1672_;
v___y_1637_ = v___y_1674_;
v___y_1638_ = v___y_1673_;
v___y_1639_ = v___x_1683_;
goto v___jp_1621_;
}
}
}
}
v___jp_1686_:
{
lean_object* v___x_1705_; 
v___x_1705_ = l_Lean_Syntax_getOptional_x3f(v___y_1689_);
lean_dec(v___y_1689_);
if (lean_obj_tag(v___x_1705_) == 0)
{
lean_object* v___x_1706_; 
v___x_1706_ = lean_box(0);
v___y_1658_ = v___y_1687_;
v___y_1659_ = v___y_1688_;
v___y_1660_ = v___y_1690_;
v___y_1661_ = v___y_1691_;
v___y_1662_ = v___y_1692_;
v___y_1663_ = v___y_1693_;
v___y_1664_ = v___y_1694_;
v___y_1665_ = v___y_1695_;
v___y_1666_ = v___y_1696_;
v___y_1667_ = v___y_1697_;
v___y_1668_ = v___y_1698_;
v___y_1669_ = v___y_1699_;
v___y_1670_ = v___y_1700_;
v___y_1671_ = v___y_1701_;
v___y_1672_ = v___y_1704_;
v___y_1673_ = v___y_1703_;
v___y_1674_ = v___y_1702_;
v___y_1675_ = v___x_1706_;
goto v___jp_1657_;
}
else
{
lean_object* v_val_1707_; lean_object* v___x_1709_; uint8_t v_isShared_1710_; uint8_t v_isSharedCheck_1714_; 
v_val_1707_ = lean_ctor_get(v___x_1705_, 0);
v_isSharedCheck_1714_ = !lean_is_exclusive(v___x_1705_);
if (v_isSharedCheck_1714_ == 0)
{
v___x_1709_ = v___x_1705_;
v_isShared_1710_ = v_isSharedCheck_1714_;
goto v_resetjp_1708_;
}
else
{
lean_inc(v_val_1707_);
lean_dec(v___x_1705_);
v___x_1709_ = lean_box(0);
v_isShared_1710_ = v_isSharedCheck_1714_;
goto v_resetjp_1708_;
}
v_resetjp_1708_:
{
lean_object* v___x_1712_; 
if (v_isShared_1710_ == 0)
{
v___x_1712_ = v___x_1709_;
goto v_reusejp_1711_;
}
else
{
lean_object* v_reuseFailAlloc_1713_; 
v_reuseFailAlloc_1713_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1713_, 0, v_val_1707_);
v___x_1712_ = v_reuseFailAlloc_1713_;
goto v_reusejp_1711_;
}
v_reusejp_1711_:
{
v___y_1658_ = v___y_1687_;
v___y_1659_ = v___y_1688_;
v___y_1660_ = v___y_1690_;
v___y_1661_ = v___y_1691_;
v___y_1662_ = v___y_1692_;
v___y_1663_ = v___y_1693_;
v___y_1664_ = v___y_1694_;
v___y_1665_ = v___y_1695_;
v___y_1666_ = v___y_1696_;
v___y_1667_ = v___y_1697_;
v___y_1668_ = v___y_1698_;
v___y_1669_ = v___y_1699_;
v___y_1670_ = v___y_1700_;
v___y_1671_ = v___y_1701_;
v___y_1672_ = v___y_1704_;
v___y_1673_ = v___y_1703_;
v___y_1674_ = v___y_1702_;
v___y_1675_ = v___x_1712_;
goto v___jp_1657_;
}
}
}
}
v___jp_1715_:
{
lean_object* v_ref_1734_; lean_object* v___x_1735_; lean_object* v___x_1736_; lean_object* v___x_1737_; lean_object* v___x_1738_; lean_object* v___x_1739_; lean_object* v___x_1740_; lean_object* v___x_1741_; lean_object* v___x_1742_; lean_object* v___x_1743_; lean_object* v___x_1744_; lean_object* v___x_1745_; lean_object* v___x_1746_; lean_object* v___x_1747_; 
v_ref_1734_ = lean_ctor_get(v___y_1718_, 5);
v___x_1735_ = l_Lean_SourceInfo_fromRef(v_ref_1734_, v___y_1730_);
v___x_1736_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__1));
v___x_1737_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__2));
v___x_1738_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__4));
lean_inc_ref(v___y_1726_);
lean_inc_ref(v___y_1729_);
v___x_1739_ = l_Lean_Name_mkStr4(v___y_1729_, v___x_1737_, v___y_1726_, v___x_1738_);
v___x_1740_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__6));
lean_inc(v___x_1735_);
v___x_1741_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1741_, 0, v___x_1735_);
lean_ctor_set(v___x_1741_, 1, v___x_1740_);
v___x_1742_ = l_Lean_rootNamespace;
v___x_1743_ = l_Lean_TSyntax_getId(v___y_1717_);
v___x_1744_ = l_Lean_Name_append(v___x_1742_, v___x_1743_);
v___x_1745_ = l_Lean_mkIdentFrom(v___y_1717_, v___x_1744_, v___y_1730_);
lean_dec(v___y_1717_);
v___x_1746_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__1));
v___x_1747_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__9, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__9_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__9);
if (lean_obj_tag(v___y_1727_) == 1)
{
lean_object* v_val_1748_; lean_object* v___x_1749_; 
v_val_1748_ = lean_ctor_get(v___y_1727_, 0);
lean_inc(v_val_1748_);
lean_dec_ref_known(v___y_1727_, 1);
v___x_1749_ = l_Array_mkArray1___redArg(v_val_1748_);
v___y_955_ = v___y_1733_;
v___y_956_ = v___y_1716_;
v___y_957_ = v___x_1746_;
v___y_958_ = v___y_1719_;
v___y_959_ = v___y_1720_;
v___y_960_ = v___y_1721_;
v___y_961_ = v___x_1747_;
v___y_962_ = v___x_1745_;
v___y_963_ = v___y_1722_;
v___y_964_ = v___x_1739_;
v___y_965_ = v___y_1723_;
v___y_966_ = v___x_1735_;
v___y_967_ = v___x_1741_;
v___y_968_ = v___y_1724_;
v___y_969_ = v___y_1725_;
v___y_970_ = v___y_1726_;
v___y_971_ = v___x_1736_;
v___y_972_ = v___y_1729_;
v___y_973_ = v___y_1728_;
v___y_974_ = v___y_1731_;
v___y_975_ = v___y_1732_;
v___y_976_ = v___x_1749_;
goto v___jp_954_;
}
else
{
lean_object* v___x_1750_; 
lean_dec(v___y_1727_);
v___x_1750_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__0___closed__0));
v___y_955_ = v___y_1733_;
v___y_956_ = v___y_1716_;
v___y_957_ = v___x_1746_;
v___y_958_ = v___y_1719_;
v___y_959_ = v___y_1720_;
v___y_960_ = v___y_1721_;
v___y_961_ = v___x_1747_;
v___y_962_ = v___x_1745_;
v___y_963_ = v___y_1722_;
v___y_964_ = v___x_1739_;
v___y_965_ = v___y_1723_;
v___y_966_ = v___x_1735_;
v___y_967_ = v___x_1741_;
v___y_968_ = v___y_1724_;
v___y_969_ = v___y_1725_;
v___y_970_ = v___y_1726_;
v___y_971_ = v___x_1736_;
v___y_972_ = v___y_1729_;
v___y_973_ = v___y_1728_;
v___y_974_ = v___y_1731_;
v___y_975_ = v___y_1732_;
v___y_976_ = v___x_1750_;
goto v___jp_954_;
}
}
v___jp_1751_:
{
lean_object* v___x_1770_; 
v___x_1770_ = l_Lean_Syntax_getOptional_x3f(v___y_1758_);
lean_dec(v___y_1758_);
if (lean_obj_tag(v___x_1770_) == 0)
{
lean_object* v___x_1771_; 
v___x_1771_ = lean_box(0);
v___y_1716_ = v___y_1752_;
v___y_1717_ = v___y_1753_;
v___y_1718_ = v___y_1754_;
v___y_1719_ = v___y_1755_;
v___y_1720_ = v___y_1756_;
v___y_1721_ = v___y_1757_;
v___y_1722_ = v___y_1759_;
v___y_1723_ = v___y_1760_;
v___y_1724_ = v___y_1761_;
v___y_1725_ = v___y_1762_;
v___y_1726_ = v___y_1763_;
v___y_1727_ = v___y_1764_;
v___y_1728_ = v___y_1769_;
v___y_1729_ = v___y_1765_;
v___y_1730_ = v___y_1766_;
v___y_1731_ = v___y_1767_;
v___y_1732_ = v___y_1768_;
v___y_1733_ = v___x_1771_;
goto v___jp_1715_;
}
else
{
lean_object* v_val_1772_; lean_object* v___x_1774_; uint8_t v_isShared_1775_; uint8_t v_isSharedCheck_1779_; 
v_val_1772_ = lean_ctor_get(v___x_1770_, 0);
v_isSharedCheck_1779_ = !lean_is_exclusive(v___x_1770_);
if (v_isSharedCheck_1779_ == 0)
{
v___x_1774_ = v___x_1770_;
v_isShared_1775_ = v_isSharedCheck_1779_;
goto v_resetjp_1773_;
}
else
{
lean_inc(v_val_1772_);
lean_dec(v___x_1770_);
v___x_1774_ = lean_box(0);
v_isShared_1775_ = v_isSharedCheck_1779_;
goto v_resetjp_1773_;
}
v_resetjp_1773_:
{
lean_object* v___x_1777_; 
if (v_isShared_1775_ == 0)
{
v___x_1777_ = v___x_1774_;
goto v_reusejp_1776_;
}
else
{
lean_object* v_reuseFailAlloc_1778_; 
v_reuseFailAlloc_1778_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1778_, 0, v_val_1772_);
v___x_1777_ = v_reuseFailAlloc_1778_;
goto v_reusejp_1776_;
}
v_reusejp_1776_:
{
v___y_1716_ = v___y_1752_;
v___y_1717_ = v___y_1753_;
v___y_1718_ = v___y_1754_;
v___y_1719_ = v___y_1755_;
v___y_1720_ = v___y_1756_;
v___y_1721_ = v___y_1757_;
v___y_1722_ = v___y_1759_;
v___y_1723_ = v___y_1760_;
v___y_1724_ = v___y_1761_;
v___y_1725_ = v___y_1762_;
v___y_1726_ = v___y_1763_;
v___y_1727_ = v___y_1764_;
v___y_1728_ = v___y_1769_;
v___y_1729_ = v___y_1765_;
v___y_1730_ = v___y_1766_;
v___y_1731_ = v___y_1767_;
v___y_1732_ = v___y_1768_;
v___y_1733_ = v___x_1777_;
goto v___jp_1715_;
}
}
}
}
v___jp_1780_:
{
lean_object* v___x_1799_; 
v___x_1799_ = l_Lean_Syntax_getOptional_x3f(v___y_1796_);
lean_dec(v___y_1796_);
if (lean_obj_tag(v___x_1799_) == 0)
{
lean_object* v___x_1800_; 
v___x_1800_ = lean_box(0);
v___y_1752_ = v___y_1781_;
v___y_1753_ = v___y_1782_;
v___y_1754_ = v___y_1783_;
v___y_1755_ = v___y_1784_;
v___y_1756_ = v___y_1785_;
v___y_1757_ = v___y_1786_;
v___y_1758_ = v___y_1787_;
v___y_1759_ = v___y_1788_;
v___y_1760_ = v___y_1789_;
v___y_1761_ = v___y_1798_;
v___y_1762_ = v___y_1790_;
v___y_1763_ = v___y_1791_;
v___y_1764_ = v___y_1792_;
v___y_1765_ = v___y_1793_;
v___y_1766_ = v___y_1794_;
v___y_1767_ = v___y_1795_;
v___y_1768_ = v___y_1797_;
v___y_1769_ = v___x_1800_;
goto v___jp_1751_;
}
else
{
lean_object* v_val_1801_; lean_object* v___x_1803_; uint8_t v_isShared_1804_; uint8_t v_isSharedCheck_1808_; 
v_val_1801_ = lean_ctor_get(v___x_1799_, 0);
v_isSharedCheck_1808_ = !lean_is_exclusive(v___x_1799_);
if (v_isSharedCheck_1808_ == 0)
{
v___x_1803_ = v___x_1799_;
v_isShared_1804_ = v_isSharedCheck_1808_;
goto v_resetjp_1802_;
}
else
{
lean_inc(v_val_1801_);
lean_dec(v___x_1799_);
v___x_1803_ = lean_box(0);
v_isShared_1804_ = v_isSharedCheck_1808_;
goto v_resetjp_1802_;
}
v_resetjp_1802_:
{
lean_object* v___x_1806_; 
if (v_isShared_1804_ == 0)
{
v___x_1806_ = v___x_1803_;
goto v_reusejp_1805_;
}
else
{
lean_object* v_reuseFailAlloc_1807_; 
v_reuseFailAlloc_1807_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1807_, 0, v_val_1801_);
v___x_1806_ = v_reuseFailAlloc_1807_;
goto v_reusejp_1805_;
}
v_reusejp_1805_:
{
v___y_1752_ = v___y_1781_;
v___y_1753_ = v___y_1782_;
v___y_1754_ = v___y_1783_;
v___y_1755_ = v___y_1784_;
v___y_1756_ = v___y_1785_;
v___y_1757_ = v___y_1786_;
v___y_1758_ = v___y_1787_;
v___y_1759_ = v___y_1788_;
v___y_1760_ = v___y_1789_;
v___y_1761_ = v___y_1798_;
v___y_1762_ = v___y_1790_;
v___y_1763_ = v___y_1791_;
v___y_1764_ = v___y_1792_;
v___y_1765_ = v___y_1793_;
v___y_1766_ = v___y_1794_;
v___y_1767_ = v___y_1795_;
v___y_1768_ = v___y_1797_;
v___y_1769_ = v___x_1806_;
goto v___jp_1751_;
}
}
}
}
v___jp_1809_:
{
lean_object* v_ref_1827_; uint8_t v___x_1828_; lean_object* v___x_1829_; lean_object* v___x_1830_; lean_object* v___x_1831_; lean_object* v___x_1832_; lean_object* v___x_1833_; lean_object* v___x_1834_; lean_object* v___x_1835_; lean_object* v___x_1836_; lean_object* v___x_1837_; lean_object* v___x_1838_; lean_object* v___x_1839_; lean_object* v___x_1840_; lean_object* v___x_1841_; 
v_ref_1827_ = lean_ctor_get(v___y_1812_, 5);
v___x_1828_ = 0;
v___x_1829_ = l_Lean_SourceInfo_fromRef(v_ref_1827_, v___x_1828_);
v___x_1830_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__1));
v___x_1831_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__2));
v___x_1832_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__4));
lean_inc_ref(v___y_1820_);
lean_inc_ref(v___y_1825_);
v___x_1833_ = l_Lean_Name_mkStr4(v___y_1825_, v___x_1831_, v___y_1820_, v___x_1832_);
v___x_1834_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__6));
lean_inc(v___x_1829_);
v___x_1835_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1835_, 0, v___x_1829_);
lean_ctor_set(v___x_1835_, 1, v___x_1834_);
v___x_1836_ = l_Lean_rootNamespace;
v___x_1837_ = l_Lean_TSyntax_getId(v___y_1811_);
v___x_1838_ = l_Lean_Name_append(v___x_1836_, v___x_1837_);
v___x_1839_ = l_Lean_mkIdentFrom(v___y_1811_, v___x_1838_, v___x_1828_);
lean_dec(v___y_1811_);
v___x_1840_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__1));
v___x_1841_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__9, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__9_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__9);
if (lean_obj_tag(v___y_1822_) == 1)
{
lean_object* v_val_1842_; lean_object* v___x_1843_; 
v_val_1842_ = lean_ctor_get(v___y_1822_, 0);
lean_inc(v_val_1842_);
lean_dec_ref_known(v___y_1822_, 1);
v___x_1843_ = l_Array_mkArray1___redArg(v_val_1842_);
v___y_866_ = v___x_1833_;
v___y_867_ = v___y_1810_;
v___y_868_ = v___y_1813_;
v___y_869_ = v___y_1814_;
v___y_870_ = v___x_1839_;
v___y_871_ = v___y_1815_;
v___y_872_ = v___y_1816_;
v___y_873_ = v___x_1840_;
v___y_874_ = v___x_1835_;
v___y_875_ = v___x_1829_;
v___y_876_ = v___y_1817_;
v___y_877_ = v___y_1818_;
v___y_878_ = v___y_1819_;
v___y_879_ = v___x_1830_;
v___y_880_ = v___y_1826_;
v___y_881_ = v___y_1820_;
v___y_882_ = v___y_1821_;
v___y_883_ = v___x_1841_;
v___y_884_ = v___y_1823_;
v___y_885_ = v___y_1825_;
v___y_886_ = v___y_1824_;
v___y_887_ = v___x_1843_;
goto v___jp_865_;
}
else
{
lean_object* v___x_1844_; 
lean_dec(v___y_1822_);
v___x_1844_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__0___closed__0));
v___y_866_ = v___x_1833_;
v___y_867_ = v___y_1810_;
v___y_868_ = v___y_1813_;
v___y_869_ = v___y_1814_;
v___y_870_ = v___x_1839_;
v___y_871_ = v___y_1815_;
v___y_872_ = v___y_1816_;
v___y_873_ = v___x_1840_;
v___y_874_ = v___x_1835_;
v___y_875_ = v___x_1829_;
v___y_876_ = v___y_1817_;
v___y_877_ = v___y_1818_;
v___y_878_ = v___y_1819_;
v___y_879_ = v___x_1830_;
v___y_880_ = v___y_1826_;
v___y_881_ = v___y_1820_;
v___y_882_ = v___y_1821_;
v___y_883_ = v___x_1841_;
v___y_884_ = v___y_1823_;
v___y_885_ = v___y_1825_;
v___y_886_ = v___y_1824_;
v___y_887_ = v___x_1844_;
goto v___jp_865_;
}
}
v___jp_1845_:
{
lean_object* v___x_1863_; 
v___x_1863_ = l_Lean_Syntax_getOptional_x3f(v___y_1852_);
lean_dec(v___y_1852_);
if (lean_obj_tag(v___x_1863_) == 0)
{
lean_object* v___x_1864_; 
v___x_1864_ = lean_box(0);
v___y_1810_ = v___y_1846_;
v___y_1811_ = v___y_1847_;
v___y_1812_ = v___y_1848_;
v___y_1813_ = v___y_1849_;
v___y_1814_ = v___y_1850_;
v___y_1815_ = v___y_1862_;
v___y_1816_ = v___y_1851_;
v___y_1817_ = v___y_1853_;
v___y_1818_ = v___y_1854_;
v___y_1819_ = v___y_1855_;
v___y_1820_ = v___y_1856_;
v___y_1821_ = v___y_1857_;
v___y_1822_ = v___y_1858_;
v___y_1823_ = v___y_1859_;
v___y_1824_ = v___y_1861_;
v___y_1825_ = v___y_1860_;
v___y_1826_ = v___x_1864_;
goto v___jp_1809_;
}
else
{
lean_object* v_val_1865_; lean_object* v___x_1867_; uint8_t v_isShared_1868_; uint8_t v_isSharedCheck_1872_; 
v_val_1865_ = lean_ctor_get(v___x_1863_, 0);
v_isSharedCheck_1872_ = !lean_is_exclusive(v___x_1863_);
if (v_isSharedCheck_1872_ == 0)
{
v___x_1867_ = v___x_1863_;
v_isShared_1868_ = v_isSharedCheck_1872_;
goto v_resetjp_1866_;
}
else
{
lean_inc(v_val_1865_);
lean_dec(v___x_1863_);
v___x_1867_ = lean_box(0);
v_isShared_1868_ = v_isSharedCheck_1872_;
goto v_resetjp_1866_;
}
v_resetjp_1866_:
{
lean_object* v___x_1870_; 
if (v_isShared_1868_ == 0)
{
v___x_1870_ = v___x_1867_;
goto v_reusejp_1869_;
}
else
{
lean_object* v_reuseFailAlloc_1871_; 
v_reuseFailAlloc_1871_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1871_, 0, v_val_1865_);
v___x_1870_ = v_reuseFailAlloc_1871_;
goto v_reusejp_1869_;
}
v_reusejp_1869_:
{
v___y_1810_ = v___y_1846_;
v___y_1811_ = v___y_1847_;
v___y_1812_ = v___y_1848_;
v___y_1813_ = v___y_1849_;
v___y_1814_ = v___y_1850_;
v___y_1815_ = v___y_1862_;
v___y_1816_ = v___y_1851_;
v___y_1817_ = v___y_1853_;
v___y_1818_ = v___y_1854_;
v___y_1819_ = v___y_1855_;
v___y_1820_ = v___y_1856_;
v___y_1821_ = v___y_1857_;
v___y_1822_ = v___y_1858_;
v___y_1823_ = v___y_1859_;
v___y_1824_ = v___y_1861_;
v___y_1825_ = v___y_1860_;
v___y_1826_ = v___x_1870_;
goto v___jp_1809_;
}
}
}
}
v___jp_1873_:
{
lean_object* v___x_1891_; 
v___x_1891_ = l_Lean_Syntax_getOptional_x3f(v___y_1881_);
lean_dec(v___y_1881_);
if (lean_obj_tag(v___x_1891_) == 0)
{
lean_object* v___x_1892_; 
v___x_1892_ = lean_box(0);
v___y_1846_ = v___y_1874_;
v___y_1847_ = v___y_1875_;
v___y_1848_ = v___y_1876_;
v___y_1849_ = v___y_1877_;
v___y_1850_ = v___y_1878_;
v___y_1851_ = v___y_1879_;
v___y_1852_ = v___y_1880_;
v___y_1853_ = v___y_1890_;
v___y_1854_ = v___y_1882_;
v___y_1855_ = v___y_1883_;
v___y_1856_ = v___y_1884_;
v___y_1857_ = v___y_1885_;
v___y_1858_ = v___y_1886_;
v___y_1859_ = v___y_1887_;
v___y_1860_ = v___y_1889_;
v___y_1861_ = v___y_1888_;
v___y_1862_ = v___x_1892_;
goto v___jp_1845_;
}
else
{
lean_object* v_val_1893_; lean_object* v___x_1895_; uint8_t v_isShared_1896_; uint8_t v_isSharedCheck_1900_; 
v_val_1893_ = lean_ctor_get(v___x_1891_, 0);
v_isSharedCheck_1900_ = !lean_is_exclusive(v___x_1891_);
if (v_isSharedCheck_1900_ == 0)
{
v___x_1895_ = v___x_1891_;
v_isShared_1896_ = v_isSharedCheck_1900_;
goto v_resetjp_1894_;
}
else
{
lean_inc(v_val_1893_);
lean_dec(v___x_1891_);
v___x_1895_ = lean_box(0);
v_isShared_1896_ = v_isSharedCheck_1900_;
goto v_resetjp_1894_;
}
v_resetjp_1894_:
{
lean_object* v___x_1898_; 
if (v_isShared_1896_ == 0)
{
v___x_1898_ = v___x_1895_;
goto v_reusejp_1897_;
}
else
{
lean_object* v_reuseFailAlloc_1899_; 
v_reuseFailAlloc_1899_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1899_, 0, v_val_1893_);
v___x_1898_ = v_reuseFailAlloc_1899_;
goto v_reusejp_1897_;
}
v_reusejp_1897_:
{
v___y_1846_ = v___y_1874_;
v___y_1847_ = v___y_1875_;
v___y_1848_ = v___y_1876_;
v___y_1849_ = v___y_1877_;
v___y_1850_ = v___y_1878_;
v___y_1851_ = v___y_1879_;
v___y_1852_ = v___y_1880_;
v___y_1853_ = v___y_1890_;
v___y_1854_ = v___y_1882_;
v___y_1855_ = v___y_1883_;
v___y_1856_ = v___y_1884_;
v___y_1857_ = v___y_1885_;
v___y_1858_ = v___y_1886_;
v___y_1859_ = v___y_1887_;
v___y_1860_ = v___y_1889_;
v___y_1861_ = v___y_1888_;
v___y_1862_ = v___x_1898_;
goto v___jp_1845_;
}
}
}
}
v___jp_1901_:
{
lean_object* v___x_1919_; 
v___x_1919_ = l_Lean_Syntax_getOptional_x3f(v___y_1913_);
lean_dec(v___y_1913_);
if (lean_obj_tag(v___x_1919_) == 0)
{
lean_object* v___x_1920_; 
v___x_1920_ = lean_box(0);
v___y_1874_ = v___y_1902_;
v___y_1875_ = v___y_1903_;
v___y_1876_ = v___y_1904_;
v___y_1877_ = v___y_1918_;
v___y_1878_ = v___y_1905_;
v___y_1879_ = v___y_1906_;
v___y_1880_ = v___y_1907_;
v___y_1881_ = v___y_1908_;
v___y_1882_ = v___y_1909_;
v___y_1883_ = v___y_1910_;
v___y_1884_ = v___y_1911_;
v___y_1885_ = v___y_1912_;
v___y_1886_ = v___y_1914_;
v___y_1887_ = v___y_1915_;
v___y_1888_ = v___y_1917_;
v___y_1889_ = v___y_1916_;
v___y_1890_ = v___x_1920_;
goto v___jp_1873_;
}
else
{
lean_object* v_val_1921_; lean_object* v___x_1923_; uint8_t v_isShared_1924_; uint8_t v_isSharedCheck_1928_; 
v_val_1921_ = lean_ctor_get(v___x_1919_, 0);
v_isSharedCheck_1928_ = !lean_is_exclusive(v___x_1919_);
if (v_isSharedCheck_1928_ == 0)
{
v___x_1923_ = v___x_1919_;
v_isShared_1924_ = v_isSharedCheck_1928_;
goto v_resetjp_1922_;
}
else
{
lean_inc(v_val_1921_);
lean_dec(v___x_1919_);
v___x_1923_ = lean_box(0);
v_isShared_1924_ = v_isSharedCheck_1928_;
goto v_resetjp_1922_;
}
v_resetjp_1922_:
{
lean_object* v___x_1926_; 
if (v_isShared_1924_ == 0)
{
v___x_1926_ = v___x_1923_;
goto v_reusejp_1925_;
}
else
{
lean_object* v_reuseFailAlloc_1927_; 
v_reuseFailAlloc_1927_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1927_, 0, v_val_1921_);
v___x_1926_ = v_reuseFailAlloc_1927_;
goto v_reusejp_1925_;
}
v_reusejp_1925_:
{
v___y_1874_ = v___y_1902_;
v___y_1875_ = v___y_1903_;
v___y_1876_ = v___y_1904_;
v___y_1877_ = v___y_1918_;
v___y_1878_ = v___y_1905_;
v___y_1879_ = v___y_1906_;
v___y_1880_ = v___y_1907_;
v___y_1881_ = v___y_1908_;
v___y_1882_ = v___y_1909_;
v___y_1883_ = v___y_1910_;
v___y_1884_ = v___y_1911_;
v___y_1885_ = v___y_1912_;
v___y_1886_ = v___y_1914_;
v___y_1887_ = v___y_1915_;
v___y_1888_ = v___y_1917_;
v___y_1889_ = v___y_1916_;
v___y_1890_ = v___x_1926_;
goto v___jp_1873_;
}
}
}
}
v___jp_1930_:
{
lean_object* v___x_1934_; lean_object* v___x_1935_; lean_object* v___x_1936_; lean_object* v___x_1937_; lean_object* v_ns_1938_; lean_object* v___x_1939_; lean_object* v___x_1940_; lean_object* v___x_1941_; lean_object* v___x_1942_; lean_object* v___x_1943_; lean_object* v___x_1944_; lean_object* v___x_1945_; lean_object* v___x_1946_; uint8_t v___x_1947_; 
v___x_1934_ = lean_unsigned_to_nat(1u);
v___x_1935_ = l_Lean_Syntax_getArg(v_x_322_, v___x_1934_);
v___x_1936_ = lean_unsigned_to_nat(2u);
v___x_1937_ = lean_unsigned_to_nat(4u);
v_ns_1938_ = l_Lean_Syntax_getArg(v_x_322_, v___x_1937_);
v___x_1939_ = lean_unsigned_to_nat(5u);
v___x_1940_ = lean_unsigned_to_nat(6u);
v___x_1941_ = l_Lean_Syntax_getArg(v_x_322_, v___x_1940_);
lean_dec(v_x_322_);
v___x_1942_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_scopedNS___closed__12));
v___x_1943_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_scopedNS___closed__13));
v___x_1944_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__3));
v___x_1945_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__17));
v___x_1946_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__18));
lean_inc(v___x_1941_);
v___x_1947_ = l_Lean_Syntax_isOfKind(v___x_1941_, v___x_1946_);
if (v___x_1947_ == 0)
{
lean_object* v___x_1948_; uint8_t v___x_1949_; 
v___x_1948_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__20));
lean_inc(v___x_1941_);
v___x_1949_ = l_Lean_Syntax_isOfKind(v___x_1941_, v___x_1948_);
if (v___x_1949_ == 0)
{
uint8_t v___x_1950_; 
lean_dec(v_doc_1931_);
v___x_1950_ = l_Lean_Syntax_matchesNull(v___x_1929_, v___x_1338_);
if (v___x_1950_ == 0)
{
lean_dec(v___x_1941_);
lean_dec(v_ns_1938_);
lean_dec(v___x_1935_);
v___y_326_ = v___y_1933_;
goto v___jp_325_;
}
else
{
uint8_t v___x_1951_; 
v___x_1951_ = l_Lean_Syntax_matchesNull(v___x_1935_, v___x_1338_);
if (v___x_1951_ == 0)
{
lean_dec(v___x_1941_);
lean_dec(v_ns_1938_);
v___y_326_ = v___y_1933_;
goto v___jp_325_;
}
else
{
lean_object* v___x_1952_; uint8_t v___x_1953_; 
v___x_1952_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__8));
lean_inc(v___x_1941_);
v___x_1953_ = l_Lean_Syntax_isOfKind(v___x_1941_, v___x_1952_);
if (v___x_1953_ == 0)
{
lean_dec(v___x_1941_);
lean_dec(v_ns_1938_);
v___y_326_ = v___y_1933_;
goto v___jp_325_;
}
else
{
lean_object* v___x_1954_; lean_object* v___x_1955_; lean_object* v___x_1956_; lean_object* v___x_1957_; uint8_t v___x_1958_; 
v___x_1954_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_1936_);
v___x_1955_ = l_Lean_Syntax_getArgs(v___x_1954_);
lean_dec(v___x_1954_);
v___x_1956_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__0___closed__0));
v___x_1957_ = lean_array_get_size(v___x_1955_);
v___x_1958_ = lean_nat_dec_lt(v___x_1338_, v___x_1957_);
if (v___x_1958_ == 0)
{
lean_dec_ref(v___x_1955_);
v___y_371_ = v___y_1933_;
v___y_372_ = v___x_1937_;
v___y_373_ = v_ns_1938_;
v___y_374_ = v___y_1932_;
v___y_375_ = v___x_1941_;
v___y_376_ = v___x_1956_;
goto v___jp_370_;
}
else
{
lean_object* v___x_1959_; lean_object* v___x_1960_; uint8_t v___x_1961_; 
v___x_1959_ = lean_box(v___x_1953_);
v___x_1960_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1960_, 0, v___x_1959_);
lean_ctor_set(v___x_1960_, 1, v___x_1956_);
v___x_1961_ = lean_nat_dec_le(v___x_1957_, v___x_1957_);
if (v___x_1961_ == 0)
{
if (v___x_1958_ == 0)
{
lean_dec_ref_known(v___x_1960_, 2);
lean_dec_ref(v___x_1955_);
v___y_371_ = v___y_1933_;
v___y_372_ = v___x_1937_;
v___y_373_ = v_ns_1938_;
v___y_374_ = v___y_1932_;
v___y_375_ = v___x_1941_;
v___y_376_ = v___x_1956_;
goto v___jp_370_;
}
else
{
size_t v___x_1962_; size_t v___x_1963_; lean_object* v___x_1964_; lean_object* v_snd_1965_; 
v___x_1962_ = ((size_t)0ULL);
v___x_1963_ = lean_usize_of_nat(v___x_1957_);
v___x_1964_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__2(v___x_1953_, v___x_1949_, v___x_1955_, v___x_1962_, v___x_1963_, v___x_1960_);
lean_dec_ref(v___x_1955_);
v_snd_1965_ = lean_ctor_get(v___x_1964_, 1);
lean_inc(v_snd_1965_);
lean_dec_ref(v___x_1964_);
v___y_371_ = v___y_1933_;
v___y_372_ = v___x_1937_;
v___y_373_ = v_ns_1938_;
v___y_374_ = v___y_1932_;
v___y_375_ = v___x_1941_;
v___y_376_ = v_snd_1965_;
goto v___jp_370_;
}
}
else
{
size_t v___x_1966_; size_t v___x_1967_; lean_object* v___x_1968_; lean_object* v_snd_1969_; 
v___x_1966_ = ((size_t)0ULL);
v___x_1967_ = lean_usize_of_nat(v___x_1957_);
v___x_1968_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__2(v___x_1953_, v___x_1949_, v___x_1955_, v___x_1966_, v___x_1967_, v___x_1960_);
lean_dec_ref(v___x_1955_);
v_snd_1969_ = lean_ctor_get(v___x_1968_, 1);
lean_inc(v_snd_1969_);
lean_dec_ref(v___x_1968_);
v___y_371_ = v___y_1933_;
v___y_372_ = v___x_1937_;
v___y_373_ = v_ns_1938_;
v___y_374_ = v___y_1932_;
v___y_375_ = v___x_1941_;
v___y_376_ = v_snd_1969_;
goto v___jp_370_;
}
}
}
}
}
}
else
{
lean_object* v___x_1970_; uint8_t v___x_1971_; 
v___x_1970_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_1338_);
v___x_1971_ = l_Lean_Syntax_matchesNull(v___x_1970_, v___x_1338_);
if (v___x_1971_ == 0)
{
uint8_t v___x_1972_; 
lean_dec(v_doc_1931_);
v___x_1972_ = l_Lean_Syntax_matchesNull(v___x_1929_, v___x_1338_);
if (v___x_1972_ == 0)
{
lean_dec(v___x_1941_);
lean_dec(v_ns_1938_);
lean_dec(v___x_1935_);
v___y_326_ = v___y_1933_;
goto v___jp_325_;
}
else
{
uint8_t v___x_1973_; 
v___x_1973_ = l_Lean_Syntax_matchesNull(v___x_1935_, v___x_1338_);
if (v___x_1973_ == 0)
{
lean_dec(v___x_1941_);
lean_dec(v_ns_1938_);
v___y_326_ = v___y_1933_;
goto v___jp_325_;
}
else
{
lean_object* v___x_1974_; uint8_t v___x_1975_; 
v___x_1974_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__8));
lean_inc(v___x_1941_);
v___x_1975_ = l_Lean_Syntax_isOfKind(v___x_1941_, v___x_1974_);
if (v___x_1975_ == 0)
{
lean_dec(v___x_1941_);
lean_dec(v_ns_1938_);
v___y_326_ = v___y_1933_;
goto v___jp_325_;
}
else
{
lean_object* v___x_1976_; lean_object* v___x_1977_; lean_object* v___x_1978_; lean_object* v___x_1979_; uint8_t v___x_1980_; 
v___x_1976_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_1936_);
v___x_1977_ = l_Lean_Syntax_getArgs(v___x_1976_);
lean_dec(v___x_1976_);
v___x_1978_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__0___closed__0));
v___x_1979_ = lean_array_get_size(v___x_1977_);
v___x_1980_ = lean_nat_dec_lt(v___x_1338_, v___x_1979_);
if (v___x_1980_ == 0)
{
lean_dec_ref(v___x_1977_);
v___y_384_ = v___y_1933_;
v___y_385_ = v___x_1937_;
v___y_386_ = v_ns_1938_;
v___y_387_ = v___y_1932_;
v___y_388_ = v___x_1941_;
v___y_389_ = v___x_1978_;
goto v___jp_383_;
}
else
{
lean_object* v___x_1981_; lean_object* v___x_1982_; uint8_t v___x_1983_; 
v___x_1981_ = lean_box(v___x_1975_);
v___x_1982_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1982_, 0, v___x_1981_);
lean_ctor_set(v___x_1982_, 1, v___x_1978_);
v___x_1983_ = lean_nat_dec_le(v___x_1979_, v___x_1979_);
if (v___x_1983_ == 0)
{
if (v___x_1980_ == 0)
{
lean_dec_ref_known(v___x_1982_, 2);
lean_dec_ref(v___x_1977_);
v___y_384_ = v___y_1933_;
v___y_385_ = v___x_1937_;
v___y_386_ = v_ns_1938_;
v___y_387_ = v___y_1932_;
v___y_388_ = v___x_1941_;
v___y_389_ = v___x_1978_;
goto v___jp_383_;
}
else
{
size_t v___x_1984_; size_t v___x_1985_; lean_object* v___x_1986_; lean_object* v_snd_1987_; 
v___x_1984_ = ((size_t)0ULL);
v___x_1985_ = lean_usize_of_nat(v___x_1979_);
v___x_1986_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__2(v___x_1975_, v___x_1971_, v___x_1977_, v___x_1984_, v___x_1985_, v___x_1982_);
lean_dec_ref(v___x_1977_);
v_snd_1987_ = lean_ctor_get(v___x_1986_, 1);
lean_inc(v_snd_1987_);
lean_dec_ref(v___x_1986_);
v___y_384_ = v___y_1933_;
v___y_385_ = v___x_1937_;
v___y_386_ = v_ns_1938_;
v___y_387_ = v___y_1932_;
v___y_388_ = v___x_1941_;
v___y_389_ = v_snd_1987_;
goto v___jp_383_;
}
}
else
{
size_t v___x_1988_; size_t v___x_1989_; lean_object* v___x_1990_; lean_object* v_snd_1991_; 
v___x_1988_ = ((size_t)0ULL);
v___x_1989_ = lean_usize_of_nat(v___x_1979_);
v___x_1990_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__2(v___x_1975_, v___x_1971_, v___x_1977_, v___x_1988_, v___x_1989_, v___x_1982_);
lean_dec_ref(v___x_1977_);
v_snd_1991_ = lean_ctor_get(v___x_1990_, 1);
lean_inc(v_snd_1991_);
lean_dec_ref(v___x_1990_);
v___y_384_ = v___y_1933_;
v___y_385_ = v___x_1937_;
v___y_386_ = v_ns_1938_;
v___y_387_ = v___y_1932_;
v___y_388_ = v___x_1941_;
v___y_389_ = v_snd_1991_;
goto v___jp_383_;
}
}
}
}
}
}
else
{
lean_object* v___x_1992_; uint8_t v___x_1993_; 
v___x_1992_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_1934_);
v___x_1993_ = l_Lean_Syntax_matchesNull(v___x_1992_, v___x_1338_);
if (v___x_1993_ == 0)
{
uint8_t v___x_1994_; 
lean_dec(v_doc_1931_);
v___x_1994_ = l_Lean_Syntax_matchesNull(v___x_1929_, v___x_1338_);
if (v___x_1994_ == 0)
{
lean_dec(v___x_1941_);
lean_dec(v_ns_1938_);
lean_dec(v___x_1935_);
v___y_326_ = v___y_1933_;
goto v___jp_325_;
}
else
{
uint8_t v___x_1995_; 
v___x_1995_ = l_Lean_Syntax_matchesNull(v___x_1935_, v___x_1338_);
if (v___x_1995_ == 0)
{
lean_dec(v___x_1941_);
lean_dec(v_ns_1938_);
v___y_326_ = v___y_1933_;
goto v___jp_325_;
}
else
{
lean_object* v___x_1996_; uint8_t v___x_1997_; 
v___x_1996_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__8));
lean_inc(v___x_1941_);
v___x_1997_ = l_Lean_Syntax_isOfKind(v___x_1941_, v___x_1996_);
if (v___x_1997_ == 0)
{
lean_dec(v___x_1941_);
lean_dec(v_ns_1938_);
v___y_326_ = v___y_1933_;
goto v___jp_325_;
}
else
{
lean_object* v___x_1998_; lean_object* v___x_1999_; lean_object* v___x_2000_; lean_object* v___x_2001_; uint8_t v___x_2002_; 
v___x_1998_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_1936_);
v___x_1999_ = l_Lean_Syntax_getArgs(v___x_1998_);
lean_dec(v___x_1998_);
v___x_2000_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__0___closed__0));
v___x_2001_ = lean_array_get_size(v___x_1999_);
v___x_2002_ = lean_nat_dec_lt(v___x_1338_, v___x_2001_);
if (v___x_2002_ == 0)
{
lean_dec_ref(v___x_1999_);
v___y_397_ = v___y_1933_;
v___y_398_ = v___x_1937_;
v___y_399_ = v_ns_1938_;
v___y_400_ = v___y_1932_;
v___y_401_ = v___x_1941_;
v___y_402_ = v___x_2000_;
goto v___jp_396_;
}
else
{
lean_object* v___x_2003_; lean_object* v___x_2004_; uint8_t v___x_2005_; 
v___x_2003_ = lean_box(v___x_1997_);
v___x_2004_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2004_, 0, v___x_2003_);
lean_ctor_set(v___x_2004_, 1, v___x_2000_);
v___x_2005_ = lean_nat_dec_le(v___x_2001_, v___x_2001_);
if (v___x_2005_ == 0)
{
if (v___x_2002_ == 0)
{
lean_dec_ref_known(v___x_2004_, 2);
lean_dec_ref(v___x_1999_);
v___y_397_ = v___y_1933_;
v___y_398_ = v___x_1937_;
v___y_399_ = v_ns_1938_;
v___y_400_ = v___y_1932_;
v___y_401_ = v___x_1941_;
v___y_402_ = v___x_2000_;
goto v___jp_396_;
}
else
{
size_t v___x_2006_; size_t v___x_2007_; lean_object* v___x_2008_; lean_object* v_snd_2009_; 
v___x_2006_ = ((size_t)0ULL);
v___x_2007_ = lean_usize_of_nat(v___x_2001_);
v___x_2008_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__2(v___x_1997_, v___x_1993_, v___x_1999_, v___x_2006_, v___x_2007_, v___x_2004_);
lean_dec_ref(v___x_1999_);
v_snd_2009_ = lean_ctor_get(v___x_2008_, 1);
lean_inc(v_snd_2009_);
lean_dec_ref(v___x_2008_);
v___y_397_ = v___y_1933_;
v___y_398_ = v___x_1937_;
v___y_399_ = v_ns_1938_;
v___y_400_ = v___y_1932_;
v___y_401_ = v___x_1941_;
v___y_402_ = v_snd_2009_;
goto v___jp_396_;
}
}
else
{
size_t v___x_2010_; size_t v___x_2011_; lean_object* v___x_2012_; lean_object* v_snd_2013_; 
v___x_2010_ = ((size_t)0ULL);
v___x_2011_ = lean_usize_of_nat(v___x_2001_);
v___x_2012_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__2(v___x_1997_, v___x_1993_, v___x_1999_, v___x_2010_, v___x_2011_, v___x_2004_);
lean_dec_ref(v___x_1999_);
v_snd_2013_ = lean_ctor_get(v___x_2012_, 1);
lean_inc(v_snd_2013_);
lean_dec_ref(v___x_2012_);
v___y_397_ = v___y_1933_;
v___y_398_ = v___x_1937_;
v___y_399_ = v_ns_1938_;
v___y_400_ = v___y_1932_;
v___y_401_ = v___x_1941_;
v___y_402_ = v_snd_2013_;
goto v___jp_396_;
}
}
}
}
}
}
else
{
lean_object* v___x_2014_; lean_object* v___x_2015_; lean_object* v___x_2016_; uint8_t v___x_2017_; 
v___x_2014_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_1936_);
v___x_2015_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_scopedNS___closed__14));
v___x_2016_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__5));
lean_inc(v___x_2014_);
v___x_2017_ = l_Lean_Syntax_isOfKind(v___x_2014_, v___x_2016_);
if (v___x_2017_ == 0)
{
uint8_t v___x_2018_; 
lean_dec(v_doc_1931_);
v___x_2018_ = l_Lean_Syntax_matchesNull(v___x_1929_, v___x_1338_);
if (v___x_2018_ == 0)
{
lean_dec(v___x_2014_);
lean_dec(v___x_1941_);
lean_dec(v_ns_1938_);
lean_dec(v___x_1935_);
v___y_326_ = v___y_1933_;
goto v___jp_325_;
}
else
{
uint8_t v___x_2019_; 
v___x_2019_ = l_Lean_Syntax_matchesNull(v___x_1935_, v___x_1338_);
if (v___x_2019_ == 0)
{
lean_dec(v___x_2014_);
lean_dec(v___x_1941_);
lean_dec(v_ns_1938_);
v___y_326_ = v___y_1933_;
goto v___jp_325_;
}
else
{
lean_object* v___x_2020_; uint8_t v___x_2021_; 
v___x_2020_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__8));
lean_inc(v___x_1941_);
v___x_2021_ = l_Lean_Syntax_isOfKind(v___x_1941_, v___x_2020_);
if (v___x_2021_ == 0)
{
lean_dec(v___x_2014_);
lean_dec(v___x_1941_);
lean_dec(v_ns_1938_);
v___y_326_ = v___y_1933_;
goto v___jp_325_;
}
else
{
lean_object* v___x_2022_; lean_object* v___x_2023_; lean_object* v___x_2024_; uint8_t v___x_2025_; 
v___x_2022_ = l_Lean_Syntax_getArgs(v___x_2014_);
lean_dec(v___x_2014_);
v___x_2023_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__0___closed__0));
v___x_2024_ = lean_array_get_size(v___x_2022_);
v___x_2025_ = lean_nat_dec_lt(v___x_1338_, v___x_2024_);
if (v___x_2025_ == 0)
{
lean_dec_ref(v___x_2022_);
v___y_410_ = v___y_1933_;
v___y_411_ = v___x_1937_;
v___y_412_ = v_ns_1938_;
v___y_413_ = v___y_1932_;
v___y_414_ = v___x_1941_;
v___y_415_ = v___x_2023_;
goto v___jp_409_;
}
else
{
lean_object* v___x_2026_; lean_object* v___x_2027_; uint8_t v___x_2028_; 
v___x_2026_ = lean_box(v___x_2021_);
v___x_2027_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2027_, 0, v___x_2026_);
lean_ctor_set(v___x_2027_, 1, v___x_2023_);
v___x_2028_ = lean_nat_dec_le(v___x_2024_, v___x_2024_);
if (v___x_2028_ == 0)
{
if (v___x_2025_ == 0)
{
lean_dec_ref_known(v___x_2027_, 2);
lean_dec_ref(v___x_2022_);
v___y_410_ = v___y_1933_;
v___y_411_ = v___x_1937_;
v___y_412_ = v_ns_1938_;
v___y_413_ = v___y_1932_;
v___y_414_ = v___x_1941_;
v___y_415_ = v___x_2023_;
goto v___jp_409_;
}
else
{
size_t v___x_2029_; size_t v___x_2030_; lean_object* v___x_2031_; lean_object* v_snd_2032_; 
v___x_2029_ = ((size_t)0ULL);
v___x_2030_ = lean_usize_of_nat(v___x_2024_);
v___x_2031_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__2(v___x_2021_, v___x_2017_, v___x_2022_, v___x_2029_, v___x_2030_, v___x_2027_);
lean_dec_ref(v___x_2022_);
v_snd_2032_ = lean_ctor_get(v___x_2031_, 1);
lean_inc(v_snd_2032_);
lean_dec_ref(v___x_2031_);
v___y_410_ = v___y_1933_;
v___y_411_ = v___x_1937_;
v___y_412_ = v_ns_1938_;
v___y_413_ = v___y_1932_;
v___y_414_ = v___x_1941_;
v___y_415_ = v_snd_2032_;
goto v___jp_409_;
}
}
else
{
size_t v___x_2033_; size_t v___x_2034_; lean_object* v___x_2035_; lean_object* v_snd_2036_; 
v___x_2033_ = ((size_t)0ULL);
v___x_2034_ = lean_usize_of_nat(v___x_2024_);
v___x_2035_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__2(v___x_2021_, v___x_2017_, v___x_2022_, v___x_2033_, v___x_2034_, v___x_2027_);
lean_dec_ref(v___x_2022_);
v_snd_2036_ = lean_ctor_get(v___x_2035_, 1);
lean_inc(v_snd_2036_);
lean_dec_ref(v___x_2035_);
v___y_410_ = v___y_1933_;
v___y_411_ = v___x_1937_;
v___y_412_ = v_ns_1938_;
v___y_413_ = v___y_1932_;
v___y_414_ = v___x_1941_;
v___y_415_ = v_snd_2036_;
goto v___jp_409_;
}
}
}
}
}
}
else
{
lean_object* v___x_2037_; uint8_t v___x_2038_; 
v___x_2037_ = l_Lean_Syntax_getArg(v___x_2014_, v___x_1338_);
v___x_2038_ = l_Lean_Syntax_matchesNull(v___x_2037_, v___x_1338_);
if (v___x_2038_ == 0)
{
uint8_t v___x_2039_; 
lean_dec(v_doc_1931_);
v___x_2039_ = l_Lean_Syntax_matchesNull(v___x_1929_, v___x_1338_);
if (v___x_2039_ == 0)
{
lean_dec(v___x_2014_);
lean_dec(v___x_1941_);
lean_dec(v_ns_1938_);
lean_dec(v___x_1935_);
v___y_326_ = v___y_1933_;
goto v___jp_325_;
}
else
{
uint8_t v___x_2040_; 
v___x_2040_ = l_Lean_Syntax_matchesNull(v___x_1935_, v___x_1338_);
if (v___x_2040_ == 0)
{
lean_dec(v___x_2014_);
lean_dec(v___x_1941_);
lean_dec(v_ns_1938_);
v___y_326_ = v___y_1933_;
goto v___jp_325_;
}
else
{
lean_object* v___x_2041_; uint8_t v___x_2042_; 
v___x_2041_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__8));
lean_inc(v___x_1941_);
v___x_2042_ = l_Lean_Syntax_isOfKind(v___x_1941_, v___x_2041_);
if (v___x_2042_ == 0)
{
lean_dec(v___x_2014_);
lean_dec(v___x_1941_);
lean_dec(v_ns_1938_);
v___y_326_ = v___y_1933_;
goto v___jp_325_;
}
else
{
lean_object* v___x_2043_; lean_object* v___x_2044_; lean_object* v___x_2045_; uint8_t v___x_2046_; 
v___x_2043_ = l_Lean_Syntax_getArgs(v___x_2014_);
lean_dec(v___x_2014_);
v___x_2044_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__0___closed__0));
v___x_2045_ = lean_array_get_size(v___x_2043_);
v___x_2046_ = lean_nat_dec_lt(v___x_1338_, v___x_2045_);
if (v___x_2046_ == 0)
{
lean_dec_ref(v___x_2043_);
v___y_423_ = v___y_1933_;
v___y_424_ = v___x_1937_;
v___y_425_ = v_ns_1938_;
v___y_426_ = v___y_1932_;
v___y_427_ = v___x_1941_;
v___y_428_ = v___x_2044_;
goto v___jp_422_;
}
else
{
lean_object* v___x_2047_; lean_object* v___x_2048_; uint8_t v___x_2049_; 
v___x_2047_ = lean_box(v___x_2042_);
v___x_2048_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2048_, 0, v___x_2047_);
lean_ctor_set(v___x_2048_, 1, v___x_2044_);
v___x_2049_ = lean_nat_dec_le(v___x_2045_, v___x_2045_);
if (v___x_2049_ == 0)
{
if (v___x_2046_ == 0)
{
lean_dec_ref_known(v___x_2048_, 2);
lean_dec_ref(v___x_2043_);
v___y_423_ = v___y_1933_;
v___y_424_ = v___x_1937_;
v___y_425_ = v_ns_1938_;
v___y_426_ = v___y_1932_;
v___y_427_ = v___x_1941_;
v___y_428_ = v___x_2044_;
goto v___jp_422_;
}
else
{
size_t v___x_2050_; size_t v___x_2051_; lean_object* v___x_2052_; lean_object* v_snd_2053_; 
v___x_2050_ = ((size_t)0ULL);
v___x_2051_ = lean_usize_of_nat(v___x_2045_);
v___x_2052_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__2(v___x_2042_, v___x_2038_, v___x_2043_, v___x_2050_, v___x_2051_, v___x_2048_);
lean_dec_ref(v___x_2043_);
v_snd_2053_ = lean_ctor_get(v___x_2052_, 1);
lean_inc(v_snd_2053_);
lean_dec_ref(v___x_2052_);
v___y_423_ = v___y_1933_;
v___y_424_ = v___x_1937_;
v___y_425_ = v_ns_1938_;
v___y_426_ = v___y_1932_;
v___y_427_ = v___x_1941_;
v___y_428_ = v_snd_2053_;
goto v___jp_422_;
}
}
else
{
size_t v___x_2054_; size_t v___x_2055_; lean_object* v___x_2056_; lean_object* v_snd_2057_; 
v___x_2054_ = ((size_t)0ULL);
v___x_2055_ = lean_usize_of_nat(v___x_2045_);
v___x_2056_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__2(v___x_2042_, v___x_2038_, v___x_2043_, v___x_2054_, v___x_2055_, v___x_2048_);
lean_dec_ref(v___x_2043_);
v_snd_2057_ = lean_ctor_get(v___x_2056_, 1);
lean_inc(v_snd_2057_);
lean_dec_ref(v___x_2056_);
v___y_423_ = v___y_1933_;
v___y_424_ = v___x_1937_;
v___y_425_ = v_ns_1938_;
v___y_426_ = v___y_1932_;
v___y_427_ = v___x_1941_;
v___y_428_ = v_snd_2057_;
goto v___jp_422_;
}
}
}
}
}
}
else
{
lean_object* v___x_2058_; lean_object* v___x_2059_; lean_object* v___x_2060_; uint8_t v___x_2061_; 
v___x_2058_ = lean_unsigned_to_nat(3u);
v___x_2059_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_2058_);
v___x_2060_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__22));
lean_inc(v___x_2059_);
v___x_2061_ = l_Lean_Syntax_isOfKind(v___x_2059_, v___x_2060_);
if (v___x_2061_ == 0)
{
lean_object* v___x_2062_; uint8_t v___x_2063_; 
v___x_2062_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__24));
lean_inc(v___x_2059_);
v___x_2063_ = l_Lean_Syntax_isOfKind(v___x_2059_, v___x_2062_);
if (v___x_2063_ == 0)
{
lean_object* v___x_2064_; uint8_t v___x_2065_; 
v___x_2064_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__26));
lean_inc(v___x_2059_);
v___x_2065_ = l_Lean_Syntax_isOfKind(v___x_2059_, v___x_2064_);
if (v___x_2065_ == 0)
{
lean_object* v___x_2066_; uint8_t v___x_2067_; 
v___x_2066_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__28));
lean_inc(v___x_2059_);
v___x_2067_ = l_Lean_Syntax_isOfKind(v___x_2059_, v___x_2066_);
if (v___x_2067_ == 0)
{
lean_object* v___x_2068_; uint8_t v___x_2069_; 
v___x_2068_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__30));
lean_inc(v___x_2059_);
v___x_2069_ = l_Lean_Syntax_isOfKind(v___x_2059_, v___x_2068_);
if (v___x_2069_ == 0)
{
uint8_t v___x_2070_; 
lean_dec(v___x_2059_);
lean_dec(v_doc_1931_);
v___x_2070_ = l_Lean_Syntax_matchesNull(v___x_1929_, v___x_1338_);
if (v___x_2070_ == 0)
{
lean_dec(v___x_2014_);
lean_dec(v___x_1941_);
lean_dec(v_ns_1938_);
lean_dec(v___x_1935_);
v___y_326_ = v___y_1933_;
goto v___jp_325_;
}
else
{
uint8_t v___x_2071_; 
v___x_2071_ = l_Lean_Syntax_matchesNull(v___x_1935_, v___x_1338_);
if (v___x_2071_ == 0)
{
lean_dec(v___x_2014_);
lean_dec(v___x_1941_);
lean_dec(v_ns_1938_);
v___y_326_ = v___y_1933_;
goto v___jp_325_;
}
else
{
lean_object* v___x_2072_; uint8_t v___x_2073_; 
v___x_2072_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__8));
lean_inc(v___x_1941_);
v___x_2073_ = l_Lean_Syntax_isOfKind(v___x_1941_, v___x_2072_);
if (v___x_2073_ == 0)
{
lean_dec(v___x_2014_);
lean_dec(v___x_1941_);
lean_dec(v_ns_1938_);
v___y_326_ = v___y_1933_;
goto v___jp_325_;
}
else
{
lean_object* v___x_2074_; lean_object* v___x_2075_; lean_object* v___x_2076_; uint8_t v___x_2077_; 
v___x_2074_ = l_Lean_Syntax_getArgs(v___x_2014_);
lean_dec(v___x_2014_);
v___x_2075_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__0___closed__0));
v___x_2076_ = lean_array_get_size(v___x_2074_);
v___x_2077_ = lean_nat_dec_lt(v___x_1338_, v___x_2076_);
if (v___x_2077_ == 0)
{
lean_dec_ref(v___x_2074_);
v___y_436_ = v___y_1933_;
v___y_437_ = v___x_1937_;
v___y_438_ = v_ns_1938_;
v___y_439_ = v___y_1932_;
v___y_440_ = v___x_1941_;
v___y_441_ = v___x_2075_;
goto v___jp_435_;
}
else
{
lean_object* v___x_2078_; lean_object* v___x_2079_; uint8_t v___x_2080_; 
v___x_2078_ = lean_box(v___x_2038_);
v___x_2079_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2079_, 0, v___x_2078_);
lean_ctor_set(v___x_2079_, 1, v___x_2075_);
v___x_2080_ = lean_nat_dec_le(v___x_2076_, v___x_2076_);
if (v___x_2080_ == 0)
{
if (v___x_2077_ == 0)
{
lean_dec_ref_known(v___x_2079_, 2);
lean_dec_ref(v___x_2074_);
v___y_436_ = v___y_1933_;
v___y_437_ = v___x_1937_;
v___y_438_ = v_ns_1938_;
v___y_439_ = v___y_1932_;
v___y_440_ = v___x_1941_;
v___y_441_ = v___x_2075_;
goto v___jp_435_;
}
else
{
size_t v___x_2081_; size_t v___x_2082_; lean_object* v___x_2083_; lean_object* v_snd_2084_; 
v___x_2081_ = ((size_t)0ULL);
v___x_2082_ = lean_usize_of_nat(v___x_2076_);
v___x_2083_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__2(v___x_2038_, v___x_2069_, v___x_2074_, v___x_2081_, v___x_2082_, v___x_2079_);
lean_dec_ref(v___x_2074_);
v_snd_2084_ = lean_ctor_get(v___x_2083_, 1);
lean_inc(v_snd_2084_);
lean_dec_ref(v___x_2083_);
v___y_436_ = v___y_1933_;
v___y_437_ = v___x_1937_;
v___y_438_ = v_ns_1938_;
v___y_439_ = v___y_1932_;
v___y_440_ = v___x_1941_;
v___y_441_ = v_snd_2084_;
goto v___jp_435_;
}
}
else
{
size_t v___x_2085_; size_t v___x_2086_; lean_object* v___x_2087_; lean_object* v_snd_2088_; 
v___x_2085_ = ((size_t)0ULL);
v___x_2086_ = lean_usize_of_nat(v___x_2076_);
v___x_2087_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__2(v___x_2038_, v___x_2069_, v___x_2074_, v___x_2085_, v___x_2086_, v___x_2079_);
lean_dec_ref(v___x_2074_);
v_snd_2088_ = lean_ctor_get(v___x_2087_, 1);
lean_inc(v_snd_2088_);
lean_dec_ref(v___x_2087_);
v___y_436_ = v___y_1933_;
v___y_437_ = v___x_1937_;
v___y_438_ = v_ns_1938_;
v___y_439_ = v___y_1932_;
v___y_440_ = v___x_1941_;
v___y_441_ = v_snd_2088_;
goto v___jp_435_;
}
}
}
}
}
}
else
{
lean_object* v___x_2089_; lean_object* v___x_2090_; lean_object* v___x_2091_; lean_object* v___x_2092_; lean_object* v___x_2093_; lean_object* v___x_2094_; lean_object* v___x_2095_; lean_object* v___x_2096_; 
lean_dec(v___x_2014_);
lean_dec(v___x_1929_);
v___x_2089_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_1937_);
v___x_2090_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_1939_);
v___x_2091_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_1940_);
v___x_2092_ = lean_unsigned_to_nat(7u);
v___x_2093_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_2092_);
v___x_2094_ = lean_unsigned_to_nat(9u);
v___x_2095_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_2094_);
lean_dec(v___x_1941_);
v___x_2096_ = l_Lean_Syntax_getOptional_x3f(v___x_2091_);
lean_dec(v___x_2091_);
if (lean_obj_tag(v___x_2096_) == 0)
{
lean_object* v___x_2097_; 
v___x_2097_ = lean_box(0);
v___y_1405_ = v___x_1943_;
v___y_1406_ = v_ns_1938_;
v___y_1407_ = v___x_2089_;
v___y_1408_ = v___y_1932_;
v___y_1409_ = v___x_2016_;
v___y_1410_ = v___x_2095_;
v___y_1411_ = v___x_1935_;
v___y_1412_ = v___x_2059_;
v___y_1413_ = v___x_2093_;
v___y_1414_ = v___y_1933_;
v___y_1415_ = v___x_1944_;
v___y_1416_ = v_doc_1931_;
v___y_1417_ = v___x_2067_;
v___y_1418_ = v___x_1942_;
v___y_1419_ = v___x_2015_;
v___y_1420_ = v___x_2090_;
v___y_1421_ = v___x_1948_;
v___y_1422_ = v___x_2097_;
goto v___jp_1404_;
}
else
{
lean_object* v_val_2098_; lean_object* v___x_2100_; uint8_t v_isShared_2101_; uint8_t v_isSharedCheck_2105_; 
v_val_2098_ = lean_ctor_get(v___x_2096_, 0);
v_isSharedCheck_2105_ = !lean_is_exclusive(v___x_2096_);
if (v_isSharedCheck_2105_ == 0)
{
v___x_2100_ = v___x_2096_;
v_isShared_2101_ = v_isSharedCheck_2105_;
goto v_resetjp_2099_;
}
else
{
lean_inc(v_val_2098_);
lean_dec(v___x_2096_);
v___x_2100_ = lean_box(0);
v_isShared_2101_ = v_isSharedCheck_2105_;
goto v_resetjp_2099_;
}
v_resetjp_2099_:
{
lean_object* v___x_2103_; 
if (v_isShared_2101_ == 0)
{
v___x_2103_ = v___x_2100_;
goto v_reusejp_2102_;
}
else
{
lean_object* v_reuseFailAlloc_2104_; 
v_reuseFailAlloc_2104_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2104_, 0, v_val_2098_);
v___x_2103_ = v_reuseFailAlloc_2104_;
goto v_reusejp_2102_;
}
v_reusejp_2102_:
{
v___y_1405_ = v___x_1943_;
v___y_1406_ = v_ns_1938_;
v___y_1407_ = v___x_2089_;
v___y_1408_ = v___y_1932_;
v___y_1409_ = v___x_2016_;
v___y_1410_ = v___x_2095_;
v___y_1411_ = v___x_1935_;
v___y_1412_ = v___x_2059_;
v___y_1413_ = v___x_2093_;
v___y_1414_ = v___y_1933_;
v___y_1415_ = v___x_1944_;
v___y_1416_ = v_doc_1931_;
v___y_1417_ = v___x_2067_;
v___y_1418_ = v___x_1942_;
v___y_1419_ = v___x_2015_;
v___y_1420_ = v___x_2090_;
v___y_1421_ = v___x_1948_;
v___y_1422_ = v___x_2103_;
goto v___jp_1404_;
}
}
}
}
}
else
{
lean_object* v___x_2106_; lean_object* v___x_2107_; lean_object* v___x_2108_; lean_object* v___x_2109_; lean_object* v___x_2110_; lean_object* v___x_2111_; lean_object* v___x_2112_; lean_object* v___x_2113_; 
lean_dec(v___x_2014_);
lean_dec(v___x_1929_);
v___x_2106_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_1937_);
v___x_2107_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_1939_);
v___x_2108_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_1940_);
v___x_2109_ = lean_unsigned_to_nat(7u);
v___x_2110_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_2109_);
v___x_2111_ = lean_unsigned_to_nat(9u);
v___x_2112_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_2111_);
lean_dec(v___x_1941_);
v___x_2113_ = l_Lean_Syntax_getOptional_x3f(v___x_2108_);
lean_dec(v___x_2108_);
if (lean_obj_tag(v___x_2113_) == 0)
{
lean_object* v___x_2114_; 
v___x_2114_ = lean_box(0);
v___y_1499_ = v___x_2107_;
v___y_1500_ = v___x_1943_;
v___y_1501_ = v_ns_1938_;
v___y_1502_ = v___y_1932_;
v___y_1503_ = v___x_2016_;
v___y_1504_ = v___x_1935_;
v___y_1505_ = v___x_2059_;
v___y_1506_ = v___x_2110_;
v___y_1507_ = v___y_1933_;
v___y_1508_ = v___x_2112_;
v___y_1509_ = v___x_1944_;
v___y_1510_ = v___x_2065_;
v___y_1511_ = v_doc_1931_;
v___y_1512_ = v___x_2106_;
v___y_1513_ = v___x_1942_;
v___y_1514_ = v___x_2015_;
v___y_1515_ = v___x_1948_;
v___y_1516_ = v___x_2114_;
goto v___jp_1498_;
}
else
{
lean_object* v_val_2115_; lean_object* v___x_2117_; uint8_t v_isShared_2118_; uint8_t v_isSharedCheck_2122_; 
v_val_2115_ = lean_ctor_get(v___x_2113_, 0);
v_isSharedCheck_2122_ = !lean_is_exclusive(v___x_2113_);
if (v_isSharedCheck_2122_ == 0)
{
v___x_2117_ = v___x_2113_;
v_isShared_2118_ = v_isSharedCheck_2122_;
goto v_resetjp_2116_;
}
else
{
lean_inc(v_val_2115_);
lean_dec(v___x_2113_);
v___x_2117_ = lean_box(0);
v_isShared_2118_ = v_isSharedCheck_2122_;
goto v_resetjp_2116_;
}
v_resetjp_2116_:
{
lean_object* v___x_2120_; 
if (v_isShared_2118_ == 0)
{
v___x_2120_ = v___x_2117_;
goto v_reusejp_2119_;
}
else
{
lean_object* v_reuseFailAlloc_2121_; 
v_reuseFailAlloc_2121_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2121_, 0, v_val_2115_);
v___x_2120_ = v_reuseFailAlloc_2121_;
goto v_reusejp_2119_;
}
v_reusejp_2119_:
{
v___y_1499_ = v___x_2107_;
v___y_1500_ = v___x_1943_;
v___y_1501_ = v_ns_1938_;
v___y_1502_ = v___y_1932_;
v___y_1503_ = v___x_2016_;
v___y_1504_ = v___x_1935_;
v___y_1505_ = v___x_2059_;
v___y_1506_ = v___x_2110_;
v___y_1507_ = v___y_1933_;
v___y_1508_ = v___x_2112_;
v___y_1509_ = v___x_1944_;
v___y_1510_ = v___x_2065_;
v___y_1511_ = v_doc_1931_;
v___y_1512_ = v___x_2106_;
v___y_1513_ = v___x_1942_;
v___y_1514_ = v___x_2015_;
v___y_1515_ = v___x_1948_;
v___y_1516_ = v___x_2120_;
goto v___jp_1498_;
}
}
}
}
}
else
{
lean_object* v___x_2123_; lean_object* v___x_2124_; lean_object* v___x_2125_; lean_object* v___x_2126_; lean_object* v___x_2127_; lean_object* v___x_2128_; lean_object* v___x_2129_; lean_object* v___x_2130_; 
lean_dec(v___x_2014_);
lean_dec(v___x_1929_);
v___x_2123_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_1937_);
v___x_2124_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_1939_);
v___x_2125_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_1940_);
v___x_2126_ = lean_unsigned_to_nat(7u);
v___x_2127_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_2126_);
v___x_2128_ = lean_unsigned_to_nat(9u);
v___x_2129_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_2128_);
lean_dec(v___x_1941_);
v___x_2130_ = l_Lean_Syntax_getOptional_x3f(v___x_2125_);
lean_dec(v___x_2125_);
if (lean_obj_tag(v___x_2130_) == 0)
{
lean_object* v___x_2131_; 
v___x_2131_ = lean_box(0);
v___y_1593_ = v___x_1943_;
v___y_1594_ = v___x_2127_;
v___y_1595_ = v_ns_1938_;
v___y_1596_ = v___y_1932_;
v___y_1597_ = v___x_2129_;
v___y_1598_ = v___x_2016_;
v___y_1599_ = v___x_1935_;
v___y_1600_ = v___x_2059_;
v___y_1601_ = v___y_1933_;
v___y_1602_ = v___x_1944_;
v___y_1603_ = v___x_2124_;
v___y_1604_ = v___x_2063_;
v___y_1605_ = v_doc_1931_;
v___y_1606_ = v___x_2123_;
v___y_1607_ = v___x_1942_;
v___y_1608_ = v___x_2015_;
v___y_1609_ = v___x_1948_;
v___y_1610_ = v___x_2131_;
goto v___jp_1592_;
}
else
{
lean_object* v_val_2132_; lean_object* v___x_2134_; uint8_t v_isShared_2135_; uint8_t v_isSharedCheck_2139_; 
v_val_2132_ = lean_ctor_get(v___x_2130_, 0);
v_isSharedCheck_2139_ = !lean_is_exclusive(v___x_2130_);
if (v_isSharedCheck_2139_ == 0)
{
v___x_2134_ = v___x_2130_;
v_isShared_2135_ = v_isSharedCheck_2139_;
goto v_resetjp_2133_;
}
else
{
lean_inc(v_val_2132_);
lean_dec(v___x_2130_);
v___x_2134_ = lean_box(0);
v_isShared_2135_ = v_isSharedCheck_2139_;
goto v_resetjp_2133_;
}
v_resetjp_2133_:
{
lean_object* v___x_2137_; 
if (v_isShared_2135_ == 0)
{
v___x_2137_ = v___x_2134_;
goto v_reusejp_2136_;
}
else
{
lean_object* v_reuseFailAlloc_2138_; 
v_reuseFailAlloc_2138_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2138_, 0, v_val_2132_);
v___x_2137_ = v_reuseFailAlloc_2138_;
goto v_reusejp_2136_;
}
v_reusejp_2136_:
{
v___y_1593_ = v___x_1943_;
v___y_1594_ = v___x_2127_;
v___y_1595_ = v_ns_1938_;
v___y_1596_ = v___y_1932_;
v___y_1597_ = v___x_2129_;
v___y_1598_ = v___x_2016_;
v___y_1599_ = v___x_1935_;
v___y_1600_ = v___x_2059_;
v___y_1601_ = v___y_1933_;
v___y_1602_ = v___x_1944_;
v___y_1603_ = v___x_2124_;
v___y_1604_ = v___x_2063_;
v___y_1605_ = v_doc_1931_;
v___y_1606_ = v___x_2123_;
v___y_1607_ = v___x_1942_;
v___y_1608_ = v___x_2015_;
v___y_1609_ = v___x_1948_;
v___y_1610_ = v___x_2137_;
goto v___jp_1592_;
}
}
}
}
}
else
{
lean_object* v___x_2140_; lean_object* v___x_2141_; lean_object* v___x_2142_; lean_object* v___x_2143_; lean_object* v___x_2144_; lean_object* v___x_2145_; lean_object* v___x_2146_; lean_object* v___x_2147_; 
lean_dec(v___x_2014_);
lean_dec(v___x_1929_);
v___x_2140_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_1937_);
v___x_2141_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_1939_);
v___x_2142_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_1940_);
v___x_2143_ = lean_unsigned_to_nat(7u);
v___x_2144_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_2143_);
v___x_2145_ = lean_unsigned_to_nat(9u);
v___x_2146_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_2145_);
lean_dec(v___x_1941_);
v___x_2147_ = l_Lean_Syntax_getOptional_x3f(v___x_2142_);
lean_dec(v___x_2142_);
if (lean_obj_tag(v___x_2147_) == 0)
{
lean_object* v___x_2148_; 
v___x_2148_ = lean_box(0);
v___y_1687_ = v___x_1943_;
v___y_1688_ = v_ns_1938_;
v___y_1689_ = v___x_2141_;
v___y_1690_ = v___y_1932_;
v___y_1691_ = v___x_2140_;
v___y_1692_ = v___x_2016_;
v___y_1693_ = v___x_2061_;
v___y_1694_ = v___x_1935_;
v___y_1695_ = v___x_2059_;
v___y_1696_ = v___y_1933_;
v___y_1697_ = v___x_1944_;
v___y_1698_ = v___x_2144_;
v___y_1699_ = v_doc_1931_;
v___y_1700_ = v___x_1942_;
v___y_1701_ = v___x_2015_;
v___y_1702_ = v___x_2146_;
v___y_1703_ = v___x_1948_;
v___y_1704_ = v___x_2148_;
goto v___jp_1686_;
}
else
{
lean_object* v_val_2149_; lean_object* v___x_2151_; uint8_t v_isShared_2152_; uint8_t v_isSharedCheck_2156_; 
v_val_2149_ = lean_ctor_get(v___x_2147_, 0);
v_isSharedCheck_2156_ = !lean_is_exclusive(v___x_2147_);
if (v_isSharedCheck_2156_ == 0)
{
v___x_2151_ = v___x_2147_;
v_isShared_2152_ = v_isSharedCheck_2156_;
goto v_resetjp_2150_;
}
else
{
lean_inc(v_val_2149_);
lean_dec(v___x_2147_);
v___x_2151_ = lean_box(0);
v_isShared_2152_ = v_isSharedCheck_2156_;
goto v_resetjp_2150_;
}
v_resetjp_2150_:
{
lean_object* v___x_2154_; 
if (v_isShared_2152_ == 0)
{
v___x_2154_ = v___x_2151_;
goto v_reusejp_2153_;
}
else
{
lean_object* v_reuseFailAlloc_2155_; 
v_reuseFailAlloc_2155_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2155_, 0, v_val_2149_);
v___x_2154_ = v_reuseFailAlloc_2155_;
goto v_reusejp_2153_;
}
v_reusejp_2153_:
{
v___y_1687_ = v___x_1943_;
v___y_1688_ = v_ns_1938_;
v___y_1689_ = v___x_2141_;
v___y_1690_ = v___y_1932_;
v___y_1691_ = v___x_2140_;
v___y_1692_ = v___x_2016_;
v___y_1693_ = v___x_2061_;
v___y_1694_ = v___x_1935_;
v___y_1695_ = v___x_2059_;
v___y_1696_ = v___y_1933_;
v___y_1697_ = v___x_1944_;
v___y_1698_ = v___x_2144_;
v___y_1699_ = v_doc_1931_;
v___y_1700_ = v___x_1942_;
v___y_1701_ = v___x_2015_;
v___y_1702_ = v___x_2146_;
v___y_1703_ = v___x_1948_;
v___y_1704_ = v___x_2154_;
goto v___jp_1686_;
}
}
}
}
}
else
{
lean_object* v___x_2157_; lean_object* v___x_2158_; lean_object* v___x_2159_; lean_object* v___x_2160_; lean_object* v___x_2161_; lean_object* v___x_2162_; lean_object* v___x_2163_; lean_object* v___x_2164_; 
lean_dec(v___x_2014_);
lean_dec(v___x_1929_);
v___x_2157_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_1937_);
v___x_2158_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_1939_);
v___x_2159_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_1940_);
v___x_2160_ = lean_unsigned_to_nat(7u);
v___x_2161_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_2160_);
v___x_2162_ = lean_unsigned_to_nat(9u);
v___x_2163_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_2162_);
lean_dec(v___x_1941_);
v___x_2164_ = l_Lean_Syntax_getOptional_x3f(v___x_2159_);
lean_dec(v___x_2159_);
if (lean_obj_tag(v___x_2164_) == 0)
{
lean_object* v___x_2165_; 
v___x_2165_ = lean_box(0);
v___y_1781_ = v___x_1943_;
v___y_1782_ = v_ns_1938_;
v___y_1783_ = v___y_1932_;
v___y_1784_ = v___x_2157_;
v___y_1785_ = v___x_2161_;
v___y_1786_ = v___x_2016_;
v___y_1787_ = v___x_1935_;
v___y_1788_ = v___x_2059_;
v___y_1789_ = v___y_1933_;
v___y_1790_ = v___x_2163_;
v___y_1791_ = v___x_1944_;
v___y_1792_ = v_doc_1931_;
v___y_1793_ = v___x_1942_;
v___y_1794_ = v___x_1947_;
v___y_1795_ = v___x_2015_;
v___y_1796_ = v___x_2158_;
v___y_1797_ = v___x_1948_;
v___y_1798_ = v___x_2165_;
goto v___jp_1780_;
}
else
{
lean_object* v_val_2166_; lean_object* v___x_2168_; uint8_t v_isShared_2169_; uint8_t v_isSharedCheck_2173_; 
v_val_2166_ = lean_ctor_get(v___x_2164_, 0);
v_isSharedCheck_2173_ = !lean_is_exclusive(v___x_2164_);
if (v_isSharedCheck_2173_ == 0)
{
v___x_2168_ = v___x_2164_;
v_isShared_2169_ = v_isSharedCheck_2173_;
goto v_resetjp_2167_;
}
else
{
lean_inc(v_val_2166_);
lean_dec(v___x_2164_);
v___x_2168_ = lean_box(0);
v_isShared_2169_ = v_isSharedCheck_2173_;
goto v_resetjp_2167_;
}
v_resetjp_2167_:
{
lean_object* v___x_2171_; 
if (v_isShared_2169_ == 0)
{
v___x_2171_ = v___x_2168_;
goto v_reusejp_2170_;
}
else
{
lean_object* v_reuseFailAlloc_2172_; 
v_reuseFailAlloc_2172_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2172_, 0, v_val_2166_);
v___x_2171_ = v_reuseFailAlloc_2172_;
goto v_reusejp_2170_;
}
v_reusejp_2170_:
{
v___y_1781_ = v___x_1943_;
v___y_1782_ = v_ns_1938_;
v___y_1783_ = v___y_1932_;
v___y_1784_ = v___x_2157_;
v___y_1785_ = v___x_2161_;
v___y_1786_ = v___x_2016_;
v___y_1787_ = v___x_1935_;
v___y_1788_ = v___x_2059_;
v___y_1789_ = v___y_1933_;
v___y_1790_ = v___x_2163_;
v___y_1791_ = v___x_1944_;
v___y_1792_ = v_doc_1931_;
v___y_1793_ = v___x_1942_;
v___y_1794_ = v___x_1947_;
v___y_1795_ = v___x_2015_;
v___y_1796_ = v___x_2158_;
v___y_1797_ = v___x_1948_;
v___y_1798_ = v___x_2171_;
goto v___jp_1780_;
}
}
}
}
}
}
}
}
}
}
else
{
lean_object* v___x_2174_; uint8_t v___x_2175_; 
v___x_2174_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_1338_);
v___x_2175_ = l_Lean_Syntax_matchesNull(v___x_2174_, v___x_1338_);
if (v___x_2175_ == 0)
{
uint8_t v___x_2176_; 
lean_dec(v_doc_1931_);
v___x_2176_ = l_Lean_Syntax_matchesNull(v___x_1929_, v___x_1338_);
if (v___x_2176_ == 0)
{
lean_dec(v___x_1941_);
lean_dec(v_ns_1938_);
lean_dec(v___x_1935_);
v___y_326_ = v___y_1933_;
goto v___jp_325_;
}
else
{
uint8_t v___x_2177_; 
v___x_2177_ = l_Lean_Syntax_matchesNull(v___x_1935_, v___x_1338_);
if (v___x_2177_ == 0)
{
lean_dec(v___x_1941_);
lean_dec(v_ns_1938_);
v___y_326_ = v___y_1933_;
goto v___jp_325_;
}
else
{
lean_object* v___x_2178_; uint8_t v___x_2179_; 
v___x_2178_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__8));
lean_inc(v___x_1941_);
v___x_2179_ = l_Lean_Syntax_isOfKind(v___x_1941_, v___x_2178_);
if (v___x_2179_ == 0)
{
lean_dec(v___x_1941_);
lean_dec(v_ns_1938_);
v___y_326_ = v___y_1933_;
goto v___jp_325_;
}
else
{
lean_object* v___x_2180_; lean_object* v___x_2181_; lean_object* v___x_2182_; lean_object* v___x_2183_; uint8_t v___x_2184_; 
v___x_2180_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_1936_);
v___x_2181_ = l_Lean_Syntax_getArgs(v___x_2180_);
lean_dec(v___x_2180_);
v___x_2182_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__0___closed__0));
v___x_2183_ = lean_array_get_size(v___x_2181_);
v___x_2184_ = lean_nat_dec_lt(v___x_1338_, v___x_2183_);
if (v___x_2184_ == 0)
{
lean_dec_ref(v___x_2181_);
v___y_674_ = v___y_1933_;
v___y_675_ = v___x_1937_;
v___y_676_ = v_ns_1938_;
v___y_677_ = v___y_1932_;
v___y_678_ = v___x_1941_;
v___y_679_ = v___x_2182_;
goto v___jp_673_;
}
else
{
lean_object* v___x_2185_; lean_object* v___x_2186_; uint8_t v___x_2187_; 
v___x_2185_ = lean_box(v___x_2179_);
v___x_2186_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2186_, 0, v___x_2185_);
lean_ctor_set(v___x_2186_, 1, v___x_2182_);
v___x_2187_ = lean_nat_dec_le(v___x_2183_, v___x_2183_);
if (v___x_2187_ == 0)
{
if (v___x_2184_ == 0)
{
lean_dec_ref_known(v___x_2186_, 2);
lean_dec_ref(v___x_2181_);
v___y_674_ = v___y_1933_;
v___y_675_ = v___x_1937_;
v___y_676_ = v_ns_1938_;
v___y_677_ = v___y_1932_;
v___y_678_ = v___x_1941_;
v___y_679_ = v___x_2182_;
goto v___jp_673_;
}
else
{
size_t v___x_2188_; size_t v___x_2189_; lean_object* v___x_2190_; lean_object* v_snd_2191_; 
v___x_2188_ = ((size_t)0ULL);
v___x_2189_ = lean_usize_of_nat(v___x_2183_);
v___x_2190_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__2(v___x_2179_, v___x_2175_, v___x_2181_, v___x_2188_, v___x_2189_, v___x_2186_);
lean_dec_ref(v___x_2181_);
v_snd_2191_ = lean_ctor_get(v___x_2190_, 1);
lean_inc(v_snd_2191_);
lean_dec_ref(v___x_2190_);
v___y_674_ = v___y_1933_;
v___y_675_ = v___x_1937_;
v___y_676_ = v_ns_1938_;
v___y_677_ = v___y_1932_;
v___y_678_ = v___x_1941_;
v___y_679_ = v_snd_2191_;
goto v___jp_673_;
}
}
else
{
size_t v___x_2192_; size_t v___x_2193_; lean_object* v___x_2194_; lean_object* v_snd_2195_; 
v___x_2192_ = ((size_t)0ULL);
v___x_2193_ = lean_usize_of_nat(v___x_2183_);
v___x_2194_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__2(v___x_2179_, v___x_2175_, v___x_2181_, v___x_2192_, v___x_2193_, v___x_2186_);
lean_dec_ref(v___x_2181_);
v_snd_2195_ = lean_ctor_get(v___x_2194_, 1);
lean_inc(v_snd_2195_);
lean_dec_ref(v___x_2194_);
v___y_674_ = v___y_1933_;
v___y_675_ = v___x_1937_;
v___y_676_ = v_ns_1938_;
v___y_677_ = v___y_1932_;
v___y_678_ = v___x_1941_;
v___y_679_ = v_snd_2195_;
goto v___jp_673_;
}
}
}
}
}
}
else
{
lean_object* v___x_2196_; uint8_t v___x_2197_; 
v___x_2196_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_1934_);
v___x_2197_ = l_Lean_Syntax_matchesNull(v___x_2196_, v___x_1338_);
if (v___x_2197_ == 0)
{
uint8_t v___x_2198_; 
lean_dec(v_doc_1931_);
v___x_2198_ = l_Lean_Syntax_matchesNull(v___x_1929_, v___x_1338_);
if (v___x_2198_ == 0)
{
lean_dec(v___x_1941_);
lean_dec(v_ns_1938_);
lean_dec(v___x_1935_);
v___y_326_ = v___y_1933_;
goto v___jp_325_;
}
else
{
uint8_t v___x_2199_; 
v___x_2199_ = l_Lean_Syntax_matchesNull(v___x_1935_, v___x_1338_);
if (v___x_2199_ == 0)
{
lean_dec(v___x_1941_);
lean_dec(v_ns_1938_);
v___y_326_ = v___y_1933_;
goto v___jp_325_;
}
else
{
lean_object* v___x_2200_; uint8_t v___x_2201_; 
v___x_2200_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__8));
lean_inc(v___x_1941_);
v___x_2201_ = l_Lean_Syntax_isOfKind(v___x_1941_, v___x_2200_);
if (v___x_2201_ == 0)
{
lean_dec(v___x_1941_);
lean_dec(v_ns_1938_);
v___y_326_ = v___y_1933_;
goto v___jp_325_;
}
else
{
lean_object* v___x_2202_; lean_object* v___x_2203_; lean_object* v___x_2204_; lean_object* v___x_2205_; uint8_t v___x_2206_; 
v___x_2202_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_1936_);
v___x_2203_ = l_Lean_Syntax_getArgs(v___x_2202_);
lean_dec(v___x_2202_);
v___x_2204_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__0___closed__0));
v___x_2205_ = lean_array_get_size(v___x_2203_);
v___x_2206_ = lean_nat_dec_lt(v___x_1338_, v___x_2205_);
if (v___x_2206_ == 0)
{
lean_dec_ref(v___x_2203_);
v___y_687_ = v___y_1933_;
v___y_688_ = v___x_1937_;
v___y_689_ = v_ns_1938_;
v___y_690_ = v___y_1932_;
v___y_691_ = v___x_1941_;
v___y_692_ = v___x_2204_;
goto v___jp_686_;
}
else
{
lean_object* v___x_2207_; lean_object* v___x_2208_; uint8_t v___x_2209_; 
v___x_2207_ = lean_box(v___x_2201_);
v___x_2208_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2208_, 0, v___x_2207_);
lean_ctor_set(v___x_2208_, 1, v___x_2204_);
v___x_2209_ = lean_nat_dec_le(v___x_2205_, v___x_2205_);
if (v___x_2209_ == 0)
{
if (v___x_2206_ == 0)
{
lean_dec_ref_known(v___x_2208_, 2);
lean_dec_ref(v___x_2203_);
v___y_687_ = v___y_1933_;
v___y_688_ = v___x_1937_;
v___y_689_ = v_ns_1938_;
v___y_690_ = v___y_1932_;
v___y_691_ = v___x_1941_;
v___y_692_ = v___x_2204_;
goto v___jp_686_;
}
else
{
size_t v___x_2210_; size_t v___x_2211_; lean_object* v___x_2212_; lean_object* v_snd_2213_; 
v___x_2210_ = ((size_t)0ULL);
v___x_2211_ = lean_usize_of_nat(v___x_2205_);
v___x_2212_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__2(v___x_2201_, v___x_2197_, v___x_2203_, v___x_2210_, v___x_2211_, v___x_2208_);
lean_dec_ref(v___x_2203_);
v_snd_2213_ = lean_ctor_get(v___x_2212_, 1);
lean_inc(v_snd_2213_);
lean_dec_ref(v___x_2212_);
v___y_687_ = v___y_1933_;
v___y_688_ = v___x_1937_;
v___y_689_ = v_ns_1938_;
v___y_690_ = v___y_1932_;
v___y_691_ = v___x_1941_;
v___y_692_ = v_snd_2213_;
goto v___jp_686_;
}
}
else
{
size_t v___x_2214_; size_t v___x_2215_; lean_object* v___x_2216_; lean_object* v_snd_2217_; 
v___x_2214_ = ((size_t)0ULL);
v___x_2215_ = lean_usize_of_nat(v___x_2205_);
v___x_2216_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__2(v___x_2201_, v___x_2197_, v___x_2203_, v___x_2214_, v___x_2215_, v___x_2208_);
lean_dec_ref(v___x_2203_);
v_snd_2217_ = lean_ctor_get(v___x_2216_, 1);
lean_inc(v_snd_2217_);
lean_dec_ref(v___x_2216_);
v___y_687_ = v___y_1933_;
v___y_688_ = v___x_1937_;
v___y_689_ = v_ns_1938_;
v___y_690_ = v___y_1932_;
v___y_691_ = v___x_1941_;
v___y_692_ = v_snd_2217_;
goto v___jp_686_;
}
}
}
}
}
}
else
{
lean_object* v___x_2218_; lean_object* v___x_2219_; lean_object* v___x_2220_; uint8_t v___x_2221_; 
v___x_2218_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_1936_);
v___x_2219_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_scopedNS___closed__14));
v___x_2220_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__5));
lean_inc(v___x_2218_);
v___x_2221_ = l_Lean_Syntax_isOfKind(v___x_2218_, v___x_2220_);
if (v___x_2221_ == 0)
{
uint8_t v___x_2222_; 
lean_dec(v_doc_1931_);
v___x_2222_ = l_Lean_Syntax_matchesNull(v___x_1929_, v___x_1338_);
if (v___x_2222_ == 0)
{
lean_dec(v___x_2218_);
lean_dec(v___x_1941_);
lean_dec(v_ns_1938_);
lean_dec(v___x_1935_);
v___y_326_ = v___y_1933_;
goto v___jp_325_;
}
else
{
uint8_t v___x_2223_; 
v___x_2223_ = l_Lean_Syntax_matchesNull(v___x_1935_, v___x_1338_);
if (v___x_2223_ == 0)
{
lean_dec(v___x_2218_);
lean_dec(v___x_1941_);
lean_dec(v_ns_1938_);
v___y_326_ = v___y_1933_;
goto v___jp_325_;
}
else
{
lean_object* v___x_2224_; uint8_t v___x_2225_; 
v___x_2224_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__8));
lean_inc(v___x_1941_);
v___x_2225_ = l_Lean_Syntax_isOfKind(v___x_1941_, v___x_2224_);
if (v___x_2225_ == 0)
{
lean_dec(v___x_2218_);
lean_dec(v___x_1941_);
lean_dec(v_ns_1938_);
v___y_326_ = v___y_1933_;
goto v___jp_325_;
}
else
{
lean_object* v___x_2226_; lean_object* v___x_2227_; lean_object* v___x_2228_; uint8_t v___x_2229_; 
v___x_2226_ = l_Lean_Syntax_getArgs(v___x_2218_);
lean_dec(v___x_2218_);
v___x_2227_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__0___closed__0));
v___x_2228_ = lean_array_get_size(v___x_2226_);
v___x_2229_ = lean_nat_dec_lt(v___x_1338_, v___x_2228_);
if (v___x_2229_ == 0)
{
lean_dec_ref(v___x_2226_);
v___y_700_ = v___y_1933_;
v___y_701_ = v___x_1937_;
v___y_702_ = v_ns_1938_;
v___y_703_ = v___y_1932_;
v___y_704_ = v___x_1941_;
v___y_705_ = v___x_2227_;
goto v___jp_699_;
}
else
{
lean_object* v___x_2230_; lean_object* v___x_2231_; uint8_t v___x_2232_; 
v___x_2230_ = lean_box(v___x_2225_);
v___x_2231_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2231_, 0, v___x_2230_);
lean_ctor_set(v___x_2231_, 1, v___x_2227_);
v___x_2232_ = lean_nat_dec_le(v___x_2228_, v___x_2228_);
if (v___x_2232_ == 0)
{
if (v___x_2229_ == 0)
{
lean_dec_ref_known(v___x_2231_, 2);
lean_dec_ref(v___x_2226_);
v___y_700_ = v___y_1933_;
v___y_701_ = v___x_1937_;
v___y_702_ = v_ns_1938_;
v___y_703_ = v___y_1932_;
v___y_704_ = v___x_1941_;
v___y_705_ = v___x_2227_;
goto v___jp_699_;
}
else
{
size_t v___x_2233_; size_t v___x_2234_; lean_object* v___x_2235_; lean_object* v_snd_2236_; 
v___x_2233_ = ((size_t)0ULL);
v___x_2234_ = lean_usize_of_nat(v___x_2228_);
v___x_2235_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__2(v___x_2225_, v___x_2221_, v___x_2226_, v___x_2233_, v___x_2234_, v___x_2231_);
lean_dec_ref(v___x_2226_);
v_snd_2236_ = lean_ctor_get(v___x_2235_, 1);
lean_inc(v_snd_2236_);
lean_dec_ref(v___x_2235_);
v___y_700_ = v___y_1933_;
v___y_701_ = v___x_1937_;
v___y_702_ = v_ns_1938_;
v___y_703_ = v___y_1932_;
v___y_704_ = v___x_1941_;
v___y_705_ = v_snd_2236_;
goto v___jp_699_;
}
}
else
{
size_t v___x_2237_; size_t v___x_2238_; lean_object* v___x_2239_; lean_object* v_snd_2240_; 
v___x_2237_ = ((size_t)0ULL);
v___x_2238_ = lean_usize_of_nat(v___x_2228_);
v___x_2239_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__2(v___x_2225_, v___x_2221_, v___x_2226_, v___x_2237_, v___x_2238_, v___x_2231_);
lean_dec_ref(v___x_2226_);
v_snd_2240_ = lean_ctor_get(v___x_2239_, 1);
lean_inc(v_snd_2240_);
lean_dec_ref(v___x_2239_);
v___y_700_ = v___y_1933_;
v___y_701_ = v___x_1937_;
v___y_702_ = v_ns_1938_;
v___y_703_ = v___y_1932_;
v___y_704_ = v___x_1941_;
v___y_705_ = v_snd_2240_;
goto v___jp_699_;
}
}
}
}
}
}
else
{
lean_object* v___x_2241_; uint8_t v___x_2242_; 
v___x_2241_ = l_Lean_Syntax_getArg(v___x_2218_, v___x_1338_);
v___x_2242_ = l_Lean_Syntax_matchesNull(v___x_2241_, v___x_1338_);
if (v___x_2242_ == 0)
{
uint8_t v___x_2243_; 
lean_dec(v_doc_1931_);
v___x_2243_ = l_Lean_Syntax_matchesNull(v___x_1929_, v___x_1338_);
if (v___x_2243_ == 0)
{
lean_dec(v___x_2218_);
lean_dec(v___x_1941_);
lean_dec(v_ns_1938_);
lean_dec(v___x_1935_);
v___y_326_ = v___y_1933_;
goto v___jp_325_;
}
else
{
uint8_t v___x_2244_; 
v___x_2244_ = l_Lean_Syntax_matchesNull(v___x_1935_, v___x_1338_);
if (v___x_2244_ == 0)
{
lean_dec(v___x_2218_);
lean_dec(v___x_1941_);
lean_dec(v_ns_1938_);
v___y_326_ = v___y_1933_;
goto v___jp_325_;
}
else
{
lean_object* v___x_2245_; uint8_t v___x_2246_; 
v___x_2245_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__8));
lean_inc(v___x_1941_);
v___x_2246_ = l_Lean_Syntax_isOfKind(v___x_1941_, v___x_2245_);
if (v___x_2246_ == 0)
{
lean_dec(v___x_2218_);
lean_dec(v___x_1941_);
lean_dec(v_ns_1938_);
v___y_326_ = v___y_1933_;
goto v___jp_325_;
}
else
{
lean_object* v___x_2247_; lean_object* v___x_2248_; lean_object* v___x_2249_; uint8_t v___x_2250_; 
v___x_2247_ = l_Lean_Syntax_getArgs(v___x_2218_);
lean_dec(v___x_2218_);
v___x_2248_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___lam__0___closed__0));
v___x_2249_ = lean_array_get_size(v___x_2247_);
v___x_2250_ = lean_nat_dec_lt(v___x_1338_, v___x_2249_);
if (v___x_2250_ == 0)
{
lean_dec_ref(v___x_2247_);
v___y_713_ = v___y_1933_;
v___y_714_ = v___x_1937_;
v___y_715_ = v_ns_1938_;
v___y_716_ = v___y_1932_;
v___y_717_ = v___x_1941_;
v___y_718_ = v___x_2248_;
goto v___jp_712_;
}
else
{
lean_object* v___x_2251_; lean_object* v___x_2252_; uint8_t v___x_2253_; 
v___x_2251_ = lean_box(v___x_2246_);
v___x_2252_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2252_, 0, v___x_2251_);
lean_ctor_set(v___x_2252_, 1, v___x_2248_);
v___x_2253_ = lean_nat_dec_le(v___x_2249_, v___x_2249_);
if (v___x_2253_ == 0)
{
if (v___x_2250_ == 0)
{
lean_dec_ref_known(v___x_2252_, 2);
lean_dec_ref(v___x_2247_);
v___y_713_ = v___y_1933_;
v___y_714_ = v___x_1937_;
v___y_715_ = v_ns_1938_;
v___y_716_ = v___y_1932_;
v___y_717_ = v___x_1941_;
v___y_718_ = v___x_2248_;
goto v___jp_712_;
}
else
{
size_t v___x_2254_; size_t v___x_2255_; lean_object* v___x_2256_; lean_object* v_snd_2257_; 
v___x_2254_ = ((size_t)0ULL);
v___x_2255_ = lean_usize_of_nat(v___x_2249_);
v___x_2256_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__2(v___x_2246_, v___x_2242_, v___x_2247_, v___x_2254_, v___x_2255_, v___x_2252_);
lean_dec_ref(v___x_2247_);
v_snd_2257_ = lean_ctor_get(v___x_2256_, 1);
lean_inc(v_snd_2257_);
lean_dec_ref(v___x_2256_);
v___y_713_ = v___y_1933_;
v___y_714_ = v___x_1937_;
v___y_715_ = v_ns_1938_;
v___y_716_ = v___y_1932_;
v___y_717_ = v___x_1941_;
v___y_718_ = v_snd_2257_;
goto v___jp_712_;
}
}
else
{
size_t v___x_2258_; size_t v___x_2259_; lean_object* v___x_2260_; lean_object* v_snd_2261_; 
v___x_2258_ = ((size_t)0ULL);
v___x_2259_ = lean_usize_of_nat(v___x_2249_);
v___x_2260_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__2(v___x_2246_, v___x_2242_, v___x_2247_, v___x_2258_, v___x_2259_, v___x_2252_);
lean_dec_ref(v___x_2247_);
v_snd_2261_ = lean_ctor_get(v___x_2260_, 1);
lean_inc(v_snd_2261_);
lean_dec_ref(v___x_2260_);
v___y_713_ = v___y_1933_;
v___y_714_ = v___x_1937_;
v___y_715_ = v_ns_1938_;
v___y_716_ = v___y_1932_;
v___y_717_ = v___x_1941_;
v___y_718_ = v_snd_2261_;
goto v___jp_712_;
}
}
}
}
}
}
else
{
lean_object* v___x_2262_; lean_object* v___x_2263_; lean_object* v___x_2264_; lean_object* v___x_2265_; lean_object* v___x_2266_; lean_object* v___x_2267_; lean_object* v___x_2268_; lean_object* v_sym_2269_; lean_object* v___x_2270_; 
lean_dec(v___x_2218_);
lean_dec(v___x_1929_);
v___x_2262_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_1937_);
v___x_2263_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_1939_);
v___x_2264_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_1940_);
v___x_2265_ = lean_unsigned_to_nat(7u);
v___x_2266_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_2265_);
v___x_2267_ = lean_unsigned_to_nat(9u);
v___x_2268_ = l_Lean_Syntax_getArg(v___x_1941_, v___x_2267_);
lean_dec(v___x_1941_);
v_sym_2269_ = l_Lean_Syntax_getArgs(v___x_2266_);
lean_dec(v___x_2266_);
v___x_2270_ = l_Lean_Syntax_getOptional_x3f(v___x_2264_);
lean_dec(v___x_2264_);
if (lean_obj_tag(v___x_2270_) == 0)
{
lean_object* v___x_2271_; 
v___x_2271_ = lean_box(0);
v___y_1902_ = v___x_1943_;
v___y_1903_ = v_ns_1938_;
v___y_1904_ = v___y_1932_;
v___y_1905_ = v___x_1945_;
v___y_1906_ = v___x_2219_;
v___y_1907_ = v___x_1935_;
v___y_1908_ = v___x_2262_;
v___y_1909_ = v___y_1933_;
v___y_1910_ = v_sym_2269_;
v___y_1911_ = v___x_1944_;
v___y_1912_ = v___x_2268_;
v___y_1913_ = v___x_2263_;
v___y_1914_ = v_doc_1931_;
v___y_1915_ = v___x_2220_;
v___y_1916_ = v___x_1942_;
v___y_1917_ = v___x_1946_;
v___y_1918_ = v___x_2271_;
goto v___jp_1901_;
}
else
{
lean_object* v_val_2272_; lean_object* v___x_2274_; uint8_t v_isShared_2275_; uint8_t v_isSharedCheck_2279_; 
v_val_2272_ = lean_ctor_get(v___x_2270_, 0);
v_isSharedCheck_2279_ = !lean_is_exclusive(v___x_2270_);
if (v_isSharedCheck_2279_ == 0)
{
v___x_2274_ = v___x_2270_;
v_isShared_2275_ = v_isSharedCheck_2279_;
goto v_resetjp_2273_;
}
else
{
lean_inc(v_val_2272_);
lean_dec(v___x_2270_);
v___x_2274_ = lean_box(0);
v_isShared_2275_ = v_isSharedCheck_2279_;
goto v_resetjp_2273_;
}
v_resetjp_2273_:
{
lean_object* v___x_2277_; 
if (v_isShared_2275_ == 0)
{
v___x_2277_ = v___x_2274_;
goto v_reusejp_2276_;
}
else
{
lean_object* v_reuseFailAlloc_2278_; 
v_reuseFailAlloc_2278_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2278_, 0, v_val_2272_);
v___x_2277_ = v_reuseFailAlloc_2278_;
goto v_reusejp_2276_;
}
v_reusejp_2276_:
{
v___y_1902_ = v___x_1943_;
v___y_1903_ = v_ns_1938_;
v___y_1904_ = v___y_1932_;
v___y_1905_ = v___x_1945_;
v___y_1906_ = v___x_2219_;
v___y_1907_ = v___x_1935_;
v___y_1908_ = v___x_2262_;
v___y_1909_ = v___y_1933_;
v___y_1910_ = v_sym_2269_;
v___y_1911_ = v___x_1944_;
v___y_1912_ = v___x_2268_;
v___y_1913_ = v___x_2263_;
v___y_1914_ = v_doc_1931_;
v___y_1915_ = v___x_2220_;
v___y_1916_ = v___x_1942_;
v___y_1917_ = v___x_1946_;
v___y_1918_ = v___x_2277_;
goto v___jp_1901_;
}
}
}
}
}
}
}
}
}
}
v___jp_325_:
{
lean_object* v___x_327_; lean_object* v___x_328_; 
v___x_327_ = lean_box(1);
v___x_328_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_328_, 0, v___x_327_);
lean_ctor_set(v___x_328_, 1, v___y_326_);
return v___x_328_;
}
v___jp_329_:
{
lean_object* v_ref_335_; uint8_t v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; size_t v_sz_353_; size_t v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; 
v_ref_335_ = lean_ctor_get(v___y_333_, 5);
v___x_336_ = 0;
v___x_337_ = l_Lean_SourceInfo_fromRef(v_ref_335_, v___x_336_);
v___x_338_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__1));
v___x_339_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__5));
v___x_340_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__6));
lean_inc_n(v___x_337_, 10);
v___x_341_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_341_, 0, v___x_337_);
lean_ctor_set(v___x_341_, 1, v___x_340_);
v___x_342_ = l_Lean_rootNamespace;
v___x_343_ = l_Lean_TSyntax_getId(v_ns_330_);
v___x_344_ = l_Lean_Name_append(v___x_342_, v___x_343_);
v___x_345_ = l_Lean_mkIdentFrom(v_ns_330_, v___x_344_, v___x_336_);
lean_dec(v_ns_330_);
v___x_346_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__7));
v___x_347_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__8));
v___x_348_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_348_, 0, v___x_337_);
lean_ctor_set(v___x_348_, 1, v___x_346_);
v___x_349_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_scopedNS___closed__23));
v___x_350_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_350_, 0, v___x_337_);
lean_ctor_set(v___x_350_, 1, v___x_349_);
v___x_351_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0___closed__1));
v___x_352_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__9, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__9_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__9);
v_sz_353_ = lean_array_size(v_attr_331_);
v___x_354_ = ((size_t)0ULL);
v___x_355_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__0(v___x_337_, v_sz_353_, v___x_354_, v_attr_331_);
v___x_356_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__11, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__11_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__11);
v___x_357_ = l_Lean_mkSepArray(v___x_355_, v___x_356_);
lean_dec_ref(v___x_355_);
v___x_358_ = l_Array_append___redArg(v___x_352_, v___x_357_);
lean_dec_ref(v___x_357_);
v___x_359_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_359_, 0, v___x_337_);
lean_ctor_set(v___x_359_, 1, v___x_351_);
lean_ctor_set(v___x_359_, 2, v___x_358_);
v___x_360_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__12));
v___x_361_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_361_, 0, v___x_337_);
lean_ctor_set(v___x_361_, 1, v___x_360_);
v___x_362_ = l_Array_append___redArg(v___x_352_, v_ids_332_);
lean_dec_ref(v_ids_332_);
v___x_363_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_363_, 0, v___x_337_);
lean_ctor_set(v___x_363_, 1, v___x_351_);
lean_ctor_set(v___x_363_, 2, v___x_362_);
v___x_364_ = l_Lean_Syntax_node5(v___x_337_, v___x_347_, v___x_348_, v___x_350_, v___x_359_, v___x_361_, v___x_363_);
lean_inc(v___x_364_);
lean_inc(v___x_345_);
lean_inc_ref(v___x_341_);
v___x_365_ = l_Lean_Syntax_node3(v___x_337_, v___x_339_, v___x_341_, v___x_345_, v___x_364_);
v___x_366_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__14));
v___x_367_ = l_Lean_Syntax_node3(v___x_337_, v___x_366_, v___x_341_, v___x_345_, v___x_364_);
v___x_368_ = l_Lean_Syntax_node2(v___x_337_, v___x_338_, v___x_365_, v___x_367_);
v___x_369_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_369_, 0, v___x_368_);
lean_ctor_set(v___x_369_, 1, v___y_334_);
return v___x_369_;
}
v___jp_370_:
{
size_t v_sz_377_; size_t v___x_378_; lean_object* v___x_379_; 
v_sz_377_ = lean_array_size(v___y_376_);
v___x_378_ = ((size_t)0ULL);
v___x_379_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__1(v_sz_377_, v___x_378_, v___y_376_);
if (lean_obj_tag(v___x_379_) == 0)
{
lean_dec(v___y_375_);
lean_dec(v___y_373_);
v___y_326_ = v___y_371_;
goto v___jp_325_;
}
else
{
lean_object* v_val_380_; lean_object* v___x_381_; lean_object* v_ids_382_; 
v_val_380_ = lean_ctor_get(v___x_379_, 0);
lean_inc(v_val_380_);
lean_dec_ref_known(v___x_379_, 1);
v___x_381_ = l_Lean_Syntax_getArg(v___y_375_, v___y_372_);
lean_dec(v___y_375_);
v_ids_382_ = l_Lean_Syntax_getArgs(v___x_381_);
lean_dec(v___x_381_);
v_ns_330_ = v___y_373_;
v_attr_331_ = v_val_380_;
v_ids_332_ = v_ids_382_;
v___y_333_ = v___y_374_;
v___y_334_ = v___y_371_;
goto v___jp_329_;
}
}
v___jp_383_:
{
size_t v_sz_390_; size_t v___x_391_; lean_object* v___x_392_; 
v_sz_390_ = lean_array_size(v___y_389_);
v___x_391_ = ((size_t)0ULL);
v___x_392_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__1(v_sz_390_, v___x_391_, v___y_389_);
if (lean_obj_tag(v___x_392_) == 0)
{
lean_dec(v___y_388_);
lean_dec(v___y_386_);
v___y_326_ = v___y_384_;
goto v___jp_325_;
}
else
{
lean_object* v_val_393_; lean_object* v___x_394_; lean_object* v_ids_395_; 
v_val_393_ = lean_ctor_get(v___x_392_, 0);
lean_inc(v_val_393_);
lean_dec_ref_known(v___x_392_, 1);
v___x_394_ = l_Lean_Syntax_getArg(v___y_388_, v___y_385_);
lean_dec(v___y_388_);
v_ids_395_ = l_Lean_Syntax_getArgs(v___x_394_);
lean_dec(v___x_394_);
v_ns_330_ = v___y_386_;
v_attr_331_ = v_val_393_;
v_ids_332_ = v_ids_395_;
v___y_333_ = v___y_387_;
v___y_334_ = v___y_384_;
goto v___jp_329_;
}
}
v___jp_396_:
{
size_t v_sz_403_; size_t v___x_404_; lean_object* v___x_405_; 
v_sz_403_ = lean_array_size(v___y_402_);
v___x_404_ = ((size_t)0ULL);
v___x_405_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__1(v_sz_403_, v___x_404_, v___y_402_);
if (lean_obj_tag(v___x_405_) == 0)
{
lean_dec(v___y_401_);
lean_dec(v___y_399_);
v___y_326_ = v___y_397_;
goto v___jp_325_;
}
else
{
lean_object* v_val_406_; lean_object* v___x_407_; lean_object* v_ids_408_; 
v_val_406_ = lean_ctor_get(v___x_405_, 0);
lean_inc(v_val_406_);
lean_dec_ref_known(v___x_405_, 1);
v___x_407_ = l_Lean_Syntax_getArg(v___y_401_, v___y_398_);
lean_dec(v___y_401_);
v_ids_408_ = l_Lean_Syntax_getArgs(v___x_407_);
lean_dec(v___x_407_);
v_ns_330_ = v___y_399_;
v_attr_331_ = v_val_406_;
v_ids_332_ = v_ids_408_;
v___y_333_ = v___y_400_;
v___y_334_ = v___y_397_;
goto v___jp_329_;
}
}
v___jp_409_:
{
size_t v_sz_416_; size_t v___x_417_; lean_object* v___x_418_; 
v_sz_416_ = lean_array_size(v___y_415_);
v___x_417_ = ((size_t)0ULL);
v___x_418_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__3(v_sz_416_, v___x_417_, v___y_415_);
if (lean_obj_tag(v___x_418_) == 0)
{
lean_dec(v___y_414_);
lean_dec(v___y_412_);
v___y_326_ = v___y_410_;
goto v___jp_325_;
}
else
{
lean_object* v_val_419_; lean_object* v___x_420_; lean_object* v_ids_421_; 
v_val_419_ = lean_ctor_get(v___x_418_, 0);
lean_inc(v_val_419_);
lean_dec_ref_known(v___x_418_, 1);
v___x_420_ = l_Lean_Syntax_getArg(v___y_414_, v___y_411_);
lean_dec(v___y_414_);
v_ids_421_ = l_Lean_Syntax_getArgs(v___x_420_);
lean_dec(v___x_420_);
v_ns_330_ = v___y_412_;
v_attr_331_ = v_val_419_;
v_ids_332_ = v_ids_421_;
v___y_333_ = v___y_413_;
v___y_334_ = v___y_410_;
goto v___jp_329_;
}
}
v___jp_422_:
{
size_t v_sz_429_; size_t v___x_430_; lean_object* v___x_431_; 
v_sz_429_ = lean_array_size(v___y_428_);
v___x_430_ = ((size_t)0ULL);
v___x_431_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__3(v_sz_429_, v___x_430_, v___y_428_);
if (lean_obj_tag(v___x_431_) == 0)
{
lean_dec(v___y_427_);
lean_dec(v___y_425_);
v___y_326_ = v___y_423_;
goto v___jp_325_;
}
else
{
lean_object* v_val_432_; lean_object* v___x_433_; lean_object* v_ids_434_; 
v_val_432_ = lean_ctor_get(v___x_431_, 0);
lean_inc(v_val_432_);
lean_dec_ref_known(v___x_431_, 1);
v___x_433_ = l_Lean_Syntax_getArg(v___y_427_, v___y_424_);
lean_dec(v___y_427_);
v_ids_434_ = l_Lean_Syntax_getArgs(v___x_433_);
lean_dec(v___x_433_);
v_ns_330_ = v___y_425_;
v_attr_331_ = v_val_432_;
v_ids_332_ = v_ids_434_;
v___y_333_ = v___y_426_;
v___y_334_ = v___y_423_;
goto v___jp_329_;
}
}
v___jp_435_:
{
size_t v_sz_442_; size_t v___x_443_; lean_object* v___x_444_; 
v_sz_442_ = lean_array_size(v___y_441_);
v___x_443_ = ((size_t)0ULL);
v___x_444_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__3(v_sz_442_, v___x_443_, v___y_441_);
if (lean_obj_tag(v___x_444_) == 0)
{
lean_dec(v___y_440_);
lean_dec(v___y_438_);
v___y_326_ = v___y_436_;
goto v___jp_325_;
}
else
{
lean_object* v_val_445_; lean_object* v___x_446_; lean_object* v_ids_447_; 
v_val_445_ = lean_ctor_get(v___x_444_, 0);
lean_inc(v_val_445_);
lean_dec_ref_known(v___x_444_, 1);
v___x_446_ = l_Lean_Syntax_getArg(v___y_440_, v___y_437_);
lean_dec(v___y_440_);
v_ids_447_ = l_Lean_Syntax_getArgs(v___x_446_);
lean_dec(v___x_446_);
v_ns_330_ = v___y_438_;
v_attr_331_ = v_val_445_;
v_ids_332_ = v_ids_447_;
v___y_333_ = v___y_439_;
v___y_334_ = v___y_436_;
goto v___jp_329_;
}
}
v___jp_448_:
{
lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; 
lean_inc_ref(v___y_455_);
v___x_470_ = l_Array_append___redArg(v___y_455_, v___y_469_);
lean_dec_ref(v___y_469_);
lean_inc(v___y_457_);
lean_inc_n(v___y_467_, 5);
v___x_471_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_471_, 0, v___y_467_);
lean_ctor_set(v___x_471_, 1, v___y_457_);
lean_ctor_set(v___x_471_, 2, v___x_470_);
v___x_472_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__15));
v___x_473_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_473_, 0, v___y_467_);
lean_ctor_set(v___x_473_, 1, v___x_472_);
v___x_474_ = lean_unsigned_to_nat(10u);
v___x_475_ = lean_mk_empty_array_with_capacity(v___x_474_);
v___x_476_ = lean_array_push(v___x_475_, v___y_454_);
v___x_477_ = lean_array_push(v___x_476_, v___y_464_);
v___x_478_ = lean_array_push(v___x_477_, v___y_456_);
v___x_479_ = lean_array_push(v___x_478_, v___y_460_);
v___x_480_ = lean_array_push(v___x_479_, v___y_452_);
v___x_481_ = lean_array_push(v___x_480_, v___y_459_);
v___x_482_ = lean_array_push(v___x_481_, v___x_471_);
v___x_483_ = lean_array_push(v___x_482_, v___y_462_);
v___x_484_ = lean_array_push(v___x_483_, v___x_473_);
v___x_485_ = lean_array_push(v___x_484_, v___y_458_);
lean_inc(v___y_468_);
v___x_486_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_486_, 0, v___y_467_);
lean_ctor_set(v___x_486_, 1, v___y_468_);
lean_ctor_set(v___x_486_, 2, v___x_485_);
lean_inc_ref(v___x_486_);
lean_inc(v___y_449_);
lean_inc(v___y_453_);
v___x_487_ = l_Lean_Syntax_node3(v___y_467_, v___y_465_, v___y_453_, v___y_449_, v___x_486_);
v___x_488_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__13));
lean_inc_ref(v___y_463_);
lean_inc_ref(v___y_450_);
lean_inc_ref(v___y_466_);
v___x_489_ = l_Lean_Name_mkStr4(v___y_466_, v___y_450_, v___y_463_, v___x_488_);
v___x_490_ = l_Lean_Syntax_node3(v___y_467_, v___x_489_, v___y_453_, v___y_449_, v___x_486_);
lean_inc(v___y_451_);
v___x_491_ = l_Lean_Syntax_node2(v___y_467_, v___y_451_, v___x_487_, v___x_490_);
v___x_492_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_492_, 0, v___x_491_);
lean_ctor_set(v___x_492_, 1, v___y_461_);
return v___x_492_;
}
v___jp_493_:
{
lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; 
lean_inc_ref(v___y_499_);
v___x_515_ = l_Array_append___redArg(v___y_499_, v___y_514_);
lean_dec_ref(v___y_514_);
lean_inc(v___y_495_);
lean_inc_n(v___y_498_, 5);
v___x_516_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_516_, 0, v___y_498_);
lean_ctor_set(v___x_516_, 1, v___y_495_);
lean_ctor_set(v___x_516_, 2, v___x_515_);
v___x_517_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__15));
v___x_518_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_518_, 0, v___y_498_);
lean_ctor_set(v___x_518_, 1, v___x_517_);
v___x_519_ = lean_unsigned_to_nat(10u);
v___x_520_ = lean_mk_empty_array_with_capacity(v___x_519_);
v___x_521_ = lean_array_push(v___x_520_, v___y_507_);
v___x_522_ = lean_array_push(v___x_521_, v___y_502_);
v___x_523_ = lean_array_push(v___x_522_, v___y_510_);
v___x_524_ = lean_array_push(v___x_523_, v___y_503_);
v___x_525_ = lean_array_push(v___x_524_, v___y_511_);
v___x_526_ = lean_array_push(v___x_525_, v___y_501_);
v___x_527_ = lean_array_push(v___x_526_, v___x_516_);
v___x_528_ = lean_array_push(v___x_527_, v___y_504_);
v___x_529_ = lean_array_push(v___x_528_, v___x_518_);
v___x_530_ = lean_array_push(v___x_529_, v___y_508_);
lean_inc(v___y_513_);
v___x_531_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_531_, 0, v___y_498_);
lean_ctor_set(v___x_531_, 1, v___y_513_);
lean_ctor_set(v___x_531_, 2, v___x_530_);
lean_inc_ref(v___x_531_);
lean_inc(v___y_494_);
lean_inc(v___y_506_);
v___x_532_ = l_Lean_Syntax_node3(v___y_498_, v___y_500_, v___y_506_, v___y_494_, v___x_531_);
v___x_533_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__13));
lean_inc_ref(v___y_509_);
lean_inc_ref(v___y_496_);
lean_inc_ref(v___y_512_);
v___x_534_ = l_Lean_Name_mkStr4(v___y_512_, v___y_496_, v___y_509_, v___x_533_);
v___x_535_ = l_Lean_Syntax_node3(v___y_498_, v___x_534_, v___y_506_, v___y_494_, v___x_531_);
lean_inc(v___y_497_);
v___x_536_ = l_Lean_Syntax_node2(v___y_498_, v___y_497_, v___x_532_, v___x_535_);
v___x_537_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_537_, 0, v___x_536_);
lean_ctor_set(v___x_537_, 1, v___y_505_);
return v___x_537_;
}
v___jp_538_:
{
lean_object* v___x_560_; lean_object* v___x_561_; lean_object* v___x_562_; lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_572_; lean_object* v___x_573_; lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; 
lean_inc_ref(v___y_540_);
v___x_560_ = l_Array_append___redArg(v___y_540_, v___y_559_);
lean_dec_ref(v___y_559_);
lean_inc(v___y_543_);
lean_inc_n(v___y_556_, 5);
v___x_561_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_561_, 0, v___y_556_);
lean_ctor_set(v___x_561_, 1, v___y_543_);
lean_ctor_set(v___x_561_, 2, v___x_560_);
v___x_562_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__15));
v___x_563_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_563_, 0, v___y_556_);
lean_ctor_set(v___x_563_, 1, v___x_562_);
v___x_564_ = lean_unsigned_to_nat(10u);
v___x_565_ = lean_mk_empty_array_with_capacity(v___x_564_);
v___x_566_ = lean_array_push(v___x_565_, v___y_539_);
v___x_567_ = lean_array_push(v___x_566_, v___y_544_);
v___x_568_ = lean_array_push(v___x_567_, v___y_557_);
v___x_569_ = lean_array_push(v___x_568_, v___y_549_);
v___x_570_ = lean_array_push(v___x_569_, v___y_554_);
v___x_571_ = lean_array_push(v___x_570_, v___y_551_);
v___x_572_ = lean_array_push(v___x_571_, v___x_561_);
v___x_573_ = lean_array_push(v___x_572_, v___y_545_);
v___x_574_ = lean_array_push(v___x_573_, v___x_563_);
v___x_575_ = lean_array_push(v___x_574_, v___y_546_);
lean_inc(v___y_558_);
v___x_576_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_576_, 0, v___y_556_);
lean_ctor_set(v___x_576_, 1, v___y_558_);
lean_ctor_set(v___x_576_, 2, v___x_575_);
lean_inc_ref(v___x_576_);
lean_inc(v___y_547_);
lean_inc(v___y_552_);
v___x_577_ = l_Lean_Syntax_node3(v___y_556_, v___y_541_, v___y_552_, v___y_547_, v___x_576_);
v___x_578_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__13));
lean_inc_ref(v___y_553_);
lean_inc_ref(v___y_542_);
lean_inc_ref(v___y_555_);
v___x_579_ = l_Lean_Name_mkStr4(v___y_555_, v___y_542_, v___y_553_, v___x_578_);
v___x_580_ = l_Lean_Syntax_node3(v___y_556_, v___x_579_, v___y_552_, v___y_547_, v___x_576_);
lean_inc(v___y_548_);
v___x_581_ = l_Lean_Syntax_node2(v___y_556_, v___y_548_, v___x_577_, v___x_580_);
v___x_582_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_582_, 0, v___x_581_);
lean_ctor_set(v___x_582_, 1, v___y_550_);
return v___x_582_;
}
v___jp_583_:
{
lean_object* v___x_605_; lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_627_; 
lean_inc_ref(v___y_589_);
v___x_605_ = l_Array_append___redArg(v___y_589_, v___y_604_);
lean_dec_ref(v___y_604_);
lean_inc(v___y_601_);
lean_inc_n(v___y_598_, 5);
v___x_606_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_606_, 0, v___y_598_);
lean_ctor_set(v___x_606_, 1, v___y_601_);
lean_ctor_set(v___x_606_, 2, v___x_605_);
v___x_607_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__15));
v___x_608_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_608_, 0, v___y_598_);
lean_ctor_set(v___x_608_, 1, v___x_607_);
v___x_609_ = lean_unsigned_to_nat(10u);
v___x_610_ = lean_mk_empty_array_with_capacity(v___x_609_);
v___x_611_ = lean_array_push(v___x_610_, v___y_588_);
v___x_612_ = lean_array_push(v___x_611_, v___y_586_);
v___x_613_ = lean_array_push(v___x_612_, v___y_590_);
v___x_614_ = lean_array_push(v___x_613_, v___y_593_);
v___x_615_ = lean_array_push(v___x_614_, v___y_591_);
v___x_616_ = lean_array_push(v___x_615_, v___y_585_);
v___x_617_ = lean_array_push(v___x_616_, v___x_606_);
v___x_618_ = lean_array_push(v___x_617_, v___y_597_);
v___x_619_ = lean_array_push(v___x_618_, v___x_608_);
v___x_620_ = lean_array_push(v___x_619_, v___y_602_);
lean_inc(v___y_603_);
v___x_621_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_621_, 0, v___y_598_);
lean_ctor_set(v___x_621_, 1, v___y_603_);
lean_ctor_set(v___x_621_, 2, v___x_620_);
lean_inc_ref(v___x_621_);
lean_inc(v___y_600_);
lean_inc(v___y_592_);
v___x_622_ = l_Lean_Syntax_node3(v___y_598_, v___y_596_, v___y_592_, v___y_600_, v___x_621_);
v___x_623_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__13));
lean_inc_ref(v___y_595_);
lean_inc_ref(v___y_587_);
lean_inc_ref(v___y_599_);
v___x_624_ = l_Lean_Name_mkStr4(v___y_599_, v___y_587_, v___y_595_, v___x_623_);
v___x_625_ = l_Lean_Syntax_node3(v___y_598_, v___x_624_, v___y_592_, v___y_600_, v___x_621_);
lean_inc(v___y_584_);
v___x_626_ = l_Lean_Syntax_node2(v___y_598_, v___y_584_, v___x_622_, v___x_625_);
v___x_627_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_627_, 0, v___x_626_);
lean_ctor_set(v___x_627_, 1, v___y_594_);
return v___x_627_;
}
v___jp_628_:
{
lean_object* v___x_650_; lean_object* v___x_651_; lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; lean_object* v___x_655_; lean_object* v___x_656_; lean_object* v___x_657_; lean_object* v___x_658_; lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v___x_661_; lean_object* v___x_662_; lean_object* v___x_663_; lean_object* v___x_664_; lean_object* v___x_665_; lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_670_; lean_object* v___x_671_; lean_object* v___x_672_; 
lean_inc_ref(v___y_636_);
v___x_650_ = l_Array_append___redArg(v___y_636_, v___y_649_);
lean_dec_ref(v___y_649_);
lean_inc(v___y_632_);
lean_inc_n(v___y_642_, 5);
v___x_651_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_651_, 0, v___y_642_);
lean_ctor_set(v___x_651_, 1, v___y_632_);
lean_ctor_set(v___x_651_, 2, v___x_650_);
v___x_652_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__15));
v___x_653_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_653_, 0, v___y_642_);
lean_ctor_set(v___x_653_, 1, v___x_652_);
v___x_654_ = lean_unsigned_to_nat(10u);
v___x_655_ = lean_mk_empty_array_with_capacity(v___x_654_);
v___x_656_ = lean_array_push(v___x_655_, v___y_629_);
v___x_657_ = lean_array_push(v___x_656_, v___y_631_);
v___x_658_ = lean_array_push(v___x_657_, v___y_646_);
v___x_659_ = lean_array_push(v___x_658_, v___y_638_);
v___x_660_ = lean_array_push(v___x_659_, v___y_633_);
v___x_661_ = lean_array_push(v___x_660_, v___y_634_);
v___x_662_ = lean_array_push(v___x_661_, v___x_651_);
v___x_663_ = lean_array_push(v___x_662_, v___y_635_);
v___x_664_ = lean_array_push(v___x_663_, v___x_653_);
v___x_665_ = lean_array_push(v___x_664_, v___y_643_);
lean_inc(v___y_648_);
v___x_666_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_666_, 0, v___y_642_);
lean_ctor_set(v___x_666_, 1, v___y_648_);
lean_ctor_set(v___x_666_, 2, v___x_665_);
lean_inc_ref(v___x_666_);
lean_inc(v___y_637_);
lean_inc(v___y_641_);
v___x_667_ = l_Lean_Syntax_node3(v___y_642_, v___y_640_, v___y_641_, v___y_637_, v___x_666_);
v___x_668_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__13));
lean_inc_ref(v___y_644_);
lean_inc_ref(v___y_630_);
lean_inc_ref(v___y_647_);
v___x_669_ = l_Lean_Name_mkStr4(v___y_647_, v___y_630_, v___y_644_, v___x_668_);
v___x_670_ = l_Lean_Syntax_node3(v___y_642_, v___x_669_, v___y_641_, v___y_637_, v___x_666_);
lean_inc(v___y_645_);
v___x_671_ = l_Lean_Syntax_node2(v___y_642_, v___y_645_, v___x_667_, v___x_670_);
v___x_672_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_672_, 0, v___x_671_);
lean_ctor_set(v___x_672_, 1, v___y_639_);
return v___x_672_;
}
v___jp_673_:
{
size_t v_sz_680_; size_t v___x_681_; lean_object* v___x_682_; 
v_sz_680_ = lean_array_size(v___y_679_);
v___x_681_ = ((size_t)0ULL);
v___x_682_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__1(v_sz_680_, v___x_681_, v___y_679_);
if (lean_obj_tag(v___x_682_) == 0)
{
lean_dec(v___y_678_);
lean_dec(v___y_676_);
v___y_326_ = v___y_674_;
goto v___jp_325_;
}
else
{
lean_object* v_val_683_; lean_object* v___x_684_; lean_object* v_ids_685_; 
v_val_683_ = lean_ctor_get(v___x_682_, 0);
lean_inc(v_val_683_);
lean_dec_ref_known(v___x_682_, 1);
v___x_684_ = l_Lean_Syntax_getArg(v___y_678_, v___y_675_);
lean_dec(v___y_678_);
v_ids_685_ = l_Lean_Syntax_getArgs(v___x_684_);
lean_dec(v___x_684_);
v_ns_330_ = v___y_676_;
v_attr_331_ = v_val_683_;
v_ids_332_ = v_ids_685_;
v___y_333_ = v___y_677_;
v___y_334_ = v___y_674_;
goto v___jp_329_;
}
}
v___jp_686_:
{
size_t v_sz_693_; size_t v___x_694_; lean_object* v___x_695_; 
v_sz_693_ = lean_array_size(v___y_692_);
v___x_694_ = ((size_t)0ULL);
v___x_695_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__1(v_sz_693_, v___x_694_, v___y_692_);
if (lean_obj_tag(v___x_695_) == 0)
{
lean_dec(v___y_691_);
lean_dec(v___y_689_);
v___y_326_ = v___y_687_;
goto v___jp_325_;
}
else
{
lean_object* v_val_696_; lean_object* v___x_697_; lean_object* v_ids_698_; 
v_val_696_ = lean_ctor_get(v___x_695_, 0);
lean_inc(v_val_696_);
lean_dec_ref_known(v___x_695_, 1);
v___x_697_ = l_Lean_Syntax_getArg(v___y_691_, v___y_688_);
lean_dec(v___y_691_);
v_ids_698_ = l_Lean_Syntax_getArgs(v___x_697_);
lean_dec(v___x_697_);
v_ns_330_ = v___y_689_;
v_attr_331_ = v_val_696_;
v_ids_332_ = v_ids_698_;
v___y_333_ = v___y_690_;
v___y_334_ = v___y_687_;
goto v___jp_329_;
}
}
v___jp_699_:
{
size_t v_sz_706_; size_t v___x_707_; lean_object* v___x_708_; 
v_sz_706_ = lean_array_size(v___y_705_);
v___x_707_ = ((size_t)0ULL);
v___x_708_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__3(v_sz_706_, v___x_707_, v___y_705_);
if (lean_obj_tag(v___x_708_) == 0)
{
lean_dec(v___y_704_);
lean_dec(v___y_702_);
v___y_326_ = v___y_700_;
goto v___jp_325_;
}
else
{
lean_object* v_val_709_; lean_object* v___x_710_; lean_object* v_ids_711_; 
v_val_709_ = lean_ctor_get(v___x_708_, 0);
lean_inc(v_val_709_);
lean_dec_ref_known(v___x_708_, 1);
v___x_710_ = l_Lean_Syntax_getArg(v___y_704_, v___y_701_);
lean_dec(v___y_704_);
v_ids_711_ = l_Lean_Syntax_getArgs(v___x_710_);
lean_dec(v___x_710_);
v_ns_330_ = v___y_702_;
v_attr_331_ = v_val_709_;
v_ids_332_ = v_ids_711_;
v___y_333_ = v___y_703_;
v___y_334_ = v___y_700_;
goto v___jp_329_;
}
}
v___jp_712_:
{
size_t v_sz_719_; size_t v___x_720_; lean_object* v___x_721_; 
v_sz_719_ = lean_array_size(v___y_718_);
v___x_720_ = ((size_t)0ULL);
v___x_721_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1_spec__3(v_sz_719_, v___x_720_, v___y_718_);
if (lean_obj_tag(v___x_721_) == 0)
{
lean_dec(v___y_717_);
lean_dec(v___y_715_);
v___y_326_ = v___y_713_;
goto v___jp_325_;
}
else
{
lean_object* v_val_722_; lean_object* v___x_723_; lean_object* v_ids_724_; 
v_val_722_ = lean_ctor_get(v___x_721_, 0);
lean_inc(v_val_722_);
lean_dec_ref_known(v___x_721_, 1);
v___x_723_ = l_Lean_Syntax_getArg(v___y_717_, v___y_714_);
lean_dec(v___y_717_);
v_ids_724_ = l_Lean_Syntax_getArgs(v___x_723_);
lean_dec(v___x_723_);
v_ns_330_ = v___y_715_;
v_attr_331_ = v_val_722_;
v_ids_332_ = v_ids_724_;
v___y_333_ = v___y_716_;
v___y_334_ = v___y_713_;
goto v___jp_329_;
}
}
v___jp_725_:
{
lean_object* v___x_747_; lean_object* v___x_748_; lean_object* v___x_749_; lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v___x_752_; lean_object* v___x_753_; lean_object* v___x_754_; lean_object* v___x_755_; lean_object* v___x_756_; lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; lean_object* v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; lean_object* v___x_765_; lean_object* v___x_766_; lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v___x_769_; lean_object* v___x_770_; lean_object* v___x_771_; 
lean_inc_ref_n(v___y_741_, 2);
v___x_747_ = l_Array_append___redArg(v___y_741_, v___y_746_);
lean_dec_ref(v___y_746_);
lean_inc_n(v___y_731_, 2);
lean_inc_n(v___y_733_, 6);
v___x_748_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_748_, 0, v___y_733_);
lean_ctor_set(v___x_748_, 1, v___y_731_);
lean_ctor_set(v___x_748_, 2, v___x_747_);
v___x_749_ = l_Array_append___redArg(v___y_741_, v___y_737_);
lean_dec_ref(v___y_737_);
v___x_750_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_750_, 0, v___y_733_);
lean_ctor_set(v___x_750_, 1, v___y_731_);
lean_ctor_set(v___x_750_, 2, v___x_749_);
v___x_751_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__15));
v___x_752_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_752_, 0, v___y_733_);
lean_ctor_set(v___x_752_, 1, v___x_751_);
v___x_753_ = lean_unsigned_to_nat(10u);
v___x_754_ = lean_mk_empty_array_with_capacity(v___x_753_);
v___x_755_ = lean_array_push(v___x_754_, v___y_726_);
v___x_756_ = lean_array_push(v___x_755_, v___y_734_);
v___x_757_ = lean_array_push(v___x_756_, v___y_727_);
v___x_758_ = lean_array_push(v___x_757_, v___y_744_);
v___x_759_ = lean_array_push(v___x_758_, v___y_745_);
v___x_760_ = lean_array_push(v___x_759_, v___y_735_);
v___x_761_ = lean_array_push(v___x_760_, v___x_748_);
v___x_762_ = lean_array_push(v___x_761_, v___x_750_);
v___x_763_ = lean_array_push(v___x_762_, v___x_752_);
v___x_764_ = lean_array_push(v___x_763_, v___y_740_);
lean_inc(v___y_743_);
v___x_765_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_765_, 0, v___y_733_);
lean_ctor_set(v___x_765_, 1, v___y_743_);
lean_ctor_set(v___x_765_, 2, v___x_764_);
lean_inc_ref(v___x_765_);
lean_inc(v___y_730_);
lean_inc(v___y_732_);
v___x_766_ = l_Lean_Syntax_node3(v___y_733_, v___y_728_, v___y_732_, v___y_730_, v___x_765_);
v___x_767_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___closed__13));
lean_inc_ref(v___y_739_);
lean_inc_ref(v___y_729_);
lean_inc_ref(v___y_742_);
v___x_768_ = l_Lean_Name_mkStr4(v___y_742_, v___y_729_, v___y_739_, v___x_767_);
v___x_769_ = l_Lean_Syntax_node3(v___y_733_, v___x_768_, v___y_732_, v___y_730_, v___x_765_);
lean_inc(v___y_738_);
v___x_770_ = l_Lean_Syntax_node2(v___y_733_, v___y_738_, v___x_766_, v___x_769_);
v___x_771_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_771_, 0, v___x_770_);
lean_ctor_set(v___x_771_, 1, v___y_736_);
return v___x_771_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1___boxed(lean_object* v_x_2320_, lean_object* v_a_2321_, lean_object* v_a_2322_){
_start:
{
lean_object* v_res_2323_; 
v_res_2323_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__ScopedNS______macroRules__Mathlib__Tactic__scopedNS__1(v_x_2320_, v_a_2321_, v_a_2322_);
lean_dec_ref(v_a_2321_);
return v_res_2323_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_WithWeakNamespace(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ScopedNS(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_WithWeakNamespace(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_ScopedNS(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Util_WithWeakNamespace(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_ScopedNS(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_WithWeakNamespace(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ScopedNS(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_ScopedNS(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_ScopedNS(builtin);
}
#ifdef __cplusplus
}
#endif
