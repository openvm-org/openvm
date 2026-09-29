// Lean compiler output
// Module: Mathlib.Tactic.Zify
// Imports: public import Init public meta import Init public import Mathlib.Data.Int.Cast.Basic public import Mathlib.Order.Basic public meta import Mathlib.Tactic.ToAdditive public meta import Mathlib.Tactic.ToDual
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
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray4___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Syntax_SepArray_ofElems(lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_getSimpTheorems___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_mkSimpContext(lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Syntax_TSepArray_getElems___redArg(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_Meta_simp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkExpectedTypeHint(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkEqMP(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Tactic_location;
extern lean_object* l_Lean_Parser_Tactic_simpArgs;
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Array_mkArray1___redArg(lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Zify_zify___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Zify_zify___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Zify_zify___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Zify"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Zify_zify___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "zify"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify_zify___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify_zify___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify_zify___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__2_value),LEAN_SCALAR_PTR_LITERAL(241, 91, 15, 7, 102, 11, 138, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify_zify___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__3_value),LEAN_SCALAR_PTR_LITERAL(58, 223, 165, 52, 54, 205, 87, 186)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Zify_zify___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify_zify___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__5_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify_zify___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__3_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Zify_zify___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify_zify___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__8_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__9_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Zify_zify___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__10;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Zify_zify___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__11;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Zify_zify___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__12;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Zify_zify___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__13;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Zify_zify___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify___closed__14;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify_zify;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(50, 13, 241, 145, 67, 153, 105, 177)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__5_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "configItem"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__9_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__9_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(205, 9, 236, 192, 59, 252, 178, 140)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "negConfigItem"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__11_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__11_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(196, 29, 29, 161, 247, 206, 181, 221)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "-"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "decide"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__13_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__14;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(236, 252, 83, 10, 217, 228, 80, 149)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Decidable"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__16_value),LEAN_SCALAR_PTR_LITERAL(87, 187, 205, 215, 218, 218, 68, 60)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__17_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(16, 96, 65, 173, 152, 155, 4, 222)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__17_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__19_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__19_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__19_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__19_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__19_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(53, 158, 1, 232, 101, 200, 191, 197)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__19_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__20_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__18_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__21_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__22_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__23;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "only"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__25_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "simpLemma"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__27_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__27_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__27_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__27_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__27_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__27_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__26_value),LEAN_SCALAR_PTR_LITERAL(38, 215, 101, 250, 181, 108, 118, 102)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__27_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "zify_simps"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__28_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__29;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__28_value),LEAN_SCALAR_PTR_LITERAL(83, 133, 36, 222, 110, 71, 247, 20)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__30_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__31_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "push_cast"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__32_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__33;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__32_value),LEAN_SCALAR_PTR_LITERAL(141, 141, 166, 133, 48, 139, 32, 66)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__34_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__35_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "location"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__36_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__37_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__37_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__37_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__37_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__37_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__37_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__36_value),LEAN_SCALAR_PTR_LITERAL(124, 82, 43, 228, 241, 102, 135, 24)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__37_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "at"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__38_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__39_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__40_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "simpArgs"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__41_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__42_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__42_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__42_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__42_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__42_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__42_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__41_value),LEAN_SCALAR_PTR_LITERAL(158, 198, 190, 154, 66, 126, 242, 208)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__42_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Zify_mkZifyContext_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Zify_mkZifyContext_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Zify_mkZifyContext___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_getSimpTheorems___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify_mkZifyContext___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify_mkZifyContext___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify_mkZifyContext(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify_mkZifyContext___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify_applySimpResultToProp_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify_applySimpResultToProp_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__1;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__3;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify_zifyProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify_zifyProof___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__10(void){
_start:
{
lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v___x_21_; 
v___x_19_ = l_Lean_Parser_Tactic_simpArgs;
v___x_20_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify_zify___closed__9));
v___x_21_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_21_, 0, v___x_20_);
lean_ctor_set(v___x_21_, 1, v___x_19_);
return v___x_21_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__11(void){
_start:
{
lean_object* v___x_22_; lean_object* v___x_23_; lean_object* v___x_24_; lean_object* v___x_25_; 
v___x_22_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__10, &lp_mathlib_Mathlib_Tactic_Zify_zify___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__10);
v___x_23_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify_zify___closed__7));
v___x_24_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify_zify___closed__6));
v___x_25_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_25_, 0, v___x_24_);
lean_ctor_set(v___x_25_, 1, v___x_23_);
lean_ctor_set(v___x_25_, 2, v___x_22_);
return v___x_25_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__12(void){
_start:
{
lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v___x_28_; 
v___x_26_ = l_Lean_Parser_Tactic_location;
v___x_27_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify_zify___closed__9));
v___x_28_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_28_, 0, v___x_27_);
lean_ctor_set(v___x_28_, 1, v___x_26_);
return v___x_28_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__13(void){
_start:
{
lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v___x_32_; 
v___x_29_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__12, &lp_mathlib_Mathlib_Tactic_Zify_zify___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__12);
v___x_30_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__11, &lp_mathlib_Mathlib_Tactic_Zify_zify___closed__11_once, _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__11);
v___x_31_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify_zify___closed__6));
v___x_32_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_32_, 0, v___x_31_);
lean_ctor_set(v___x_32_, 1, v___x_30_);
lean_ctor_set(v___x_32_, 2, v___x_29_);
return v___x_32_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__14(void){
_start:
{
lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v___x_36_; 
v___x_33_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__13, &lp_mathlib_Mathlib_Tactic_Zify_zify___closed__13_once, _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__13);
v___x_34_ = lean_unsigned_to_nat(1022u);
v___x_35_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify_zify___closed__4));
v___x_36_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_36_, 0, v___x_35_);
lean_ctor_set(v___x_36_, 1, v___x_34_);
lean_ctor_set(v___x_36_, 2, v___x_33_);
return v___x_36_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zify(void){
_start:
{
lean_object* v___x_37_; 
v___x_37_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Zify_zify___closed__14, &lp_mathlib_Mathlib_Tactic_Zify_zify___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_Zify_zify___closed__14);
return v___x_37_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__14(void){
_start:
{
lean_object* v___x_69_; lean_object* v___x_70_; 
v___x_69_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__13));
v___x_70_ = l_String_toRawSubstring_x27(v___x_69_);
return v___x_70_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__23(void){
_start:
{
lean_object* v___x_94_; 
v___x_94_ = l_Array_mkArray0(lean_box(0));
return v___x_94_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__29(void){
_start:
{
lean_object* v___x_104_; lean_object* v___x_105_; 
v___x_104_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__28));
v___x_105_ = l_String_toRawSubstring_x27(v___x_104_);
return v___x_105_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__33(void){
_start:
{
lean_object* v___x_110_; lean_object* v___x_111_; 
v___x_110_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__32));
v___x_111_ = l_String_toRawSubstring_x27(v___x_110_);
return v___x_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1(lean_object* v_x_132_, lean_object* v_a_133_, lean_object* v_a_134_){
_start:
{
lean_object* v___y_136_; lean_object* v___y_137_; lean_object* v___y_138_; lean_object* v___y_139_; lean_object* v___y_140_; lean_object* v___y_141_; lean_object* v___y_142_; lean_object* v___y_143_; lean_object* v___y_144_; lean_object* v___y_145_; lean_object* v___y_146_; lean_object* v___x_151_; uint8_t v___x_152_; 
v___x_151_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify_zify___closed__4));
lean_inc(v_x_132_);
v___x_152_ = l_Lean_Syntax_isOfKind(v_x_132_, v___x_151_);
if (v___x_152_ == 0)
{
lean_object* v___x_153_; lean_object* v___x_154_; 
lean_dec(v_x_132_);
v___x_153_ = lean_box(1);
v___x_154_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_154_, 0, v___x_153_);
lean_ctor_set(v___x_154_, 1, v_a_134_);
return v___x_154_;
}
else
{
lean_object* v___x_155_; lean_object* v___y_157_; lean_object* v___y_158_; lean_object* v___y_159_; lean_object* v___y_160_; lean_object* v___y_220_; lean_object* v_location_221_; lean_object* v___y_222_; lean_object* v___y_223_; lean_object* v___x_227_; lean_object* v_simpArgs_229_; lean_object* v___y_230_; lean_object* v___y_231_; lean_object* v___x_246_; uint8_t v___x_247_; 
v___x_155_ = lean_unsigned_to_nat(0u);
v___x_227_ = lean_unsigned_to_nat(1u);
v___x_246_ = l_Lean_Syntax_getArg(v_x_132_, v___x_227_);
v___x_247_ = l_Lean_Syntax_isNone(v___x_246_);
if (v___x_247_ == 0)
{
uint8_t v___x_248_; 
lean_inc(v___x_246_);
v___x_248_ = l_Lean_Syntax_matchesNull(v___x_246_, v___x_227_);
if (v___x_248_ == 0)
{
lean_object* v___x_249_; lean_object* v___x_250_; 
lean_dec(v___x_246_);
lean_dec(v_x_132_);
v___x_249_ = lean_box(1);
v___x_250_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_250_, 0, v___x_249_);
lean_ctor_set(v___x_250_, 1, v_a_134_);
return v___x_250_;
}
else
{
lean_object* v___x_251_; lean_object* v___x_252_; uint8_t v___x_253_; 
v___x_251_ = l_Lean_Syntax_getArg(v___x_246_, v___x_155_);
lean_dec(v___x_246_);
v___x_252_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__42));
lean_inc(v___x_251_);
v___x_253_ = l_Lean_Syntax_isOfKind(v___x_251_, v___x_252_);
if (v___x_253_ == 0)
{
lean_object* v___x_254_; lean_object* v___x_255_; 
lean_dec(v___x_251_);
lean_dec(v_x_132_);
v___x_254_ = lean_box(1);
v___x_255_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_255_, 0, v___x_254_);
lean_ctor_set(v___x_255_, 1, v_a_134_);
return v___x_255_;
}
else
{
lean_object* v___x_256_; lean_object* v_simpArgs_257_; lean_object* v___x_258_; 
v___x_256_ = l_Lean_Syntax_getArg(v___x_251_, v___x_227_);
lean_dec(v___x_251_);
v_simpArgs_257_ = l_Lean_Syntax_getArgs(v___x_256_);
lean_dec(v___x_256_);
v___x_258_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_258_, 0, v_simpArgs_257_);
v_simpArgs_229_ = v___x_258_;
v___y_230_ = v_a_133_;
v___y_231_ = v_a_134_;
goto v___jp_228_;
}
}
}
else
{
lean_object* v___x_259_; 
lean_dec(v___x_246_);
v___x_259_ = lean_box(0);
v_simpArgs_229_ = v___x_259_;
v___y_230_ = v_a_133_;
v___y_231_ = v_a_134_;
goto v___jp_228_;
}
v___jp_156_:
{
lean_object* v_quotContext_161_; lean_object* v_currMacroScope_162_; lean_object* v_ref_163_; uint8_t v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; 
v_quotContext_161_ = lean_ctor_get(v___y_158_, 1);
v_currMacroScope_162_ = lean_ctor_get(v___y_158_, 2);
v_ref_163_ = lean_ctor_get(v___y_158_, 5);
v___x_164_ = 0;
v___x_165_ = l_Lean_SourceInfo_fromRef(v_ref_163_, v___x_164_);
v___x_166_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__2));
v___x_167_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__3));
lean_inc_n(v___x_165_, 19);
v___x_168_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_168_, 0, v___x_165_);
lean_ctor_set(v___x_168_, 1, v___x_166_);
v___x_169_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__5));
v___x_170_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__7));
v___x_171_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__9));
v___x_172_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__11));
v___x_173_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__12));
v___x_174_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_174_, 0, v___x_165_);
lean_ctor_set(v___x_174_, 1, v___x_173_);
v___x_175_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__14, &lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__14);
v___x_176_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__15));
lean_inc_n(v_currMacroScope_162_, 3);
lean_inc_n(v_quotContext_161_, 3);
v___x_177_ = l_Lean_addMacroScope(v_quotContext_161_, v___x_176_, v_currMacroScope_162_);
v___x_178_ = lean_box(0);
v___x_179_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__22));
v___x_180_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_180_, 0, v___x_165_);
lean_ctor_set(v___x_180_, 1, v___x_175_);
lean_ctor_set(v___x_180_, 2, v___x_177_);
lean_ctor_set(v___x_180_, 3, v___x_179_);
v___x_181_ = l_Lean_Syntax_node2(v___x_165_, v___x_172_, v___x_174_, v___x_180_);
v___x_182_ = l_Lean_Syntax_node1(v___x_165_, v___x_171_, v___x_181_);
v___x_183_ = l_Lean_Syntax_node1(v___x_165_, v___x_170_, v___x_182_);
v___x_184_ = l_Lean_Syntax_node1(v___x_165_, v___x_169_, v___x_183_);
v___x_185_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__23, &lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__23_once, _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__23);
v___x_186_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_186_, 0, v___x_165_);
lean_ctor_set(v___x_186_, 1, v___x_170_);
lean_ctor_set(v___x_186_, 2, v___x_185_);
v___x_187_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__24));
v___x_188_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_188_, 0, v___x_165_);
lean_ctor_set(v___x_188_, 1, v___x_187_);
v___x_189_ = l_Lean_Syntax_node1(v___x_165_, v___x_170_, v___x_188_);
v___x_190_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__25));
v___x_191_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_191_, 0, v___x_165_);
lean_ctor_set(v___x_191_, 1, v___x_190_);
v___x_192_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__27));
v___x_193_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__29, &lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__29_once, _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__29);
v___x_194_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__30));
v___x_195_ = l_Lean_addMacroScope(v_quotContext_161_, v___x_194_, v_currMacroScope_162_);
v___x_196_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_196_, 0, v___x_165_);
lean_ctor_set(v___x_196_, 1, v___x_193_);
lean_ctor_set(v___x_196_, 2, v___x_195_);
lean_ctor_set(v___x_196_, 3, v___x_178_);
lean_inc_ref_n(v___x_186_, 4);
v___x_197_ = l_Lean_Syntax_node3(v___x_165_, v___x_192_, v___x_186_, v___x_186_, v___x_196_);
v___x_198_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__31));
v___x_199_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_199_, 0, v___x_165_);
lean_ctor_set(v___x_199_, 1, v___x_198_);
v___x_200_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__33, &lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__33_once, _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__33);
v___x_201_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__34));
v___x_202_ = l_Lean_addMacroScope(v_quotContext_161_, v___x_201_, v_currMacroScope_162_);
v___x_203_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_203_, 0, v___x_165_);
lean_ctor_set(v___x_203_, 1, v___x_200_);
lean_ctor_set(v___x_203_, 2, v___x_202_);
lean_ctor_set(v___x_203_, 3, v___x_178_);
v___x_204_ = l_Lean_Syntax_node3(v___x_165_, v___x_192_, v___x_186_, v___x_186_, v___x_203_);
lean_inc_ref(v___x_199_);
v___x_205_ = l_Array_mkArray4___redArg(v___x_197_, v___x_199_, v___x_204_, v___x_199_);
v___x_206_ = l_Lean_Syntax_SepArray_ofElems(v___x_198_, v___y_160_);
lean_dec_ref(v___y_160_);
v___x_207_ = l_Array_append___redArg(v___x_205_, v___x_206_);
lean_dec_ref(v___x_206_);
v___x_208_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_208_, 0, v___x_165_);
lean_ctor_set(v___x_208_, 1, v___x_170_);
lean_ctor_set(v___x_208_, 2, v___x_207_);
v___x_209_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__35));
v___x_210_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_210_, 0, v___x_165_);
lean_ctor_set(v___x_210_, 1, v___x_209_);
v___x_211_ = l_Lean_Syntax_node3(v___x_165_, v___x_170_, v___x_191_, v___x_208_, v___x_210_);
if (lean_obj_tag(v___y_157_) == 1)
{
lean_object* v_val_212_; lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; 
v_val_212_ = lean_ctor_get(v___y_157_, 0);
lean_inc(v_val_212_);
lean_dec_ref_known(v___y_157_, 1);
v___x_213_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__37));
v___x_214_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__38));
lean_inc_n(v___x_165_, 2);
v___x_215_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_215_, 0, v___x_165_);
lean_ctor_set(v___x_215_, 1, v___x_214_);
v___x_216_ = l_Lean_Syntax_node2(v___x_165_, v___x_213_, v___x_215_, v_val_212_);
v___x_217_ = l_Array_mkArray1___redArg(v___x_216_);
v___y_136_ = v___x_185_;
v___y_137_ = v___x_165_;
v___y_138_ = v___x_168_;
v___y_139_ = v___x_167_;
v___y_140_ = v___x_184_;
v___y_141_ = v___x_170_;
v___y_142_ = v___y_159_;
v___y_143_ = v___x_186_;
v___y_144_ = v___x_189_;
v___y_145_ = v___x_211_;
v___y_146_ = v___x_217_;
goto v___jp_135_;
}
else
{
lean_object* v___x_218_; 
lean_dec(v___y_157_);
v___x_218_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__39));
v___y_136_ = v___x_185_;
v___y_137_ = v___x_165_;
v___y_138_ = v___x_168_;
v___y_139_ = v___x_167_;
v___y_140_ = v___x_184_;
v___y_141_ = v___x_170_;
v___y_142_ = v___y_159_;
v___y_143_ = v___x_186_;
v___y_144_ = v___x_189_;
v___y_145_ = v___x_211_;
v___y_146_ = v___x_218_;
goto v___jp_135_;
}
}
v___jp_219_:
{
if (lean_obj_tag(v___y_220_) == 0)
{
lean_object* v___x_224_; 
v___x_224_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__40));
v___y_157_ = v_location_221_;
v___y_158_ = v___y_222_;
v___y_159_ = v___y_223_;
v___y_160_ = v___x_224_;
goto v___jp_156_;
}
else
{
lean_object* v_val_225_; lean_object* v___x_226_; 
v_val_225_ = lean_ctor_get(v___y_220_, 0);
lean_inc(v_val_225_);
lean_dec_ref_known(v___y_220_, 1);
v___x_226_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_val_225_);
lean_dec(v_val_225_);
v___y_157_ = v_location_221_;
v___y_158_ = v___y_222_;
v___y_159_ = v___y_223_;
v___y_160_ = v___x_226_;
goto v___jp_156_;
}
}
v___jp_228_:
{
lean_object* v___x_232_; lean_object* v___x_233_; uint8_t v___x_234_; 
v___x_232_ = lean_unsigned_to_nat(2u);
v___x_233_ = l_Lean_Syntax_getArg(v_x_132_, v___x_232_);
lean_dec(v_x_132_);
v___x_234_ = l_Lean_Syntax_isNone(v___x_233_);
if (v___x_234_ == 0)
{
uint8_t v___x_235_; 
lean_inc(v___x_233_);
v___x_235_ = l_Lean_Syntax_matchesNull(v___x_233_, v___x_227_);
if (v___x_235_ == 0)
{
lean_object* v___x_236_; lean_object* v___x_237_; 
lean_dec(v___x_233_);
lean_dec(v_simpArgs_229_);
v___x_236_ = lean_box(1);
v___x_237_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_237_, 0, v___x_236_);
lean_ctor_set(v___x_237_, 1, v___y_231_);
return v___x_237_;
}
else
{
lean_object* v___x_238_; lean_object* v___x_239_; uint8_t v___x_240_; 
v___x_238_ = l_Lean_Syntax_getArg(v___x_233_, v___x_155_);
lean_dec(v___x_233_);
v___x_239_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__37));
lean_inc(v___x_238_);
v___x_240_ = l_Lean_Syntax_isOfKind(v___x_238_, v___x_239_);
if (v___x_240_ == 0)
{
lean_object* v___x_241_; lean_object* v___x_242_; 
lean_dec(v___x_238_);
lean_dec(v_simpArgs_229_);
v___x_241_ = lean_box(1);
v___x_242_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_242_, 0, v___x_241_);
lean_ctor_set(v___x_242_, 1, v___y_231_);
return v___x_242_;
}
else
{
lean_object* v_location_243_; lean_object* v___x_244_; 
v_location_243_ = l_Lean_Syntax_getArg(v___x_238_, v___x_227_);
lean_dec(v___x_238_);
v___x_244_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_244_, 0, v_location_243_);
v___y_220_ = v_simpArgs_229_;
v_location_221_ = v___x_244_;
v___y_222_ = v___y_230_;
v___y_223_ = v___y_231_;
goto v___jp_219_;
}
}
}
else
{
lean_object* v___x_245_; 
lean_dec(v___x_233_);
v___x_245_ = lean_box(0);
v___y_220_ = v_simpArgs_229_;
v_location_221_ = v___x_245_;
v___y_222_ = v___y_230_;
v___y_223_ = v___y_231_;
goto v___jp_219_;
}
}
}
v___jp_135_:
{
lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; 
lean_inc_ref(v___y_136_);
v___x_147_ = l_Array_append___redArg(v___y_136_, v___y_146_);
lean_dec_ref(v___y_146_);
lean_inc(v___y_137_);
v___x_148_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_148_, 0, v___y_137_);
lean_ctor_set(v___x_148_, 1, v___y_141_);
lean_ctor_set(v___x_148_, 2, v___x_147_);
lean_inc(v___y_139_);
v___x_149_ = l_Lean_Syntax_node6(v___y_137_, v___y_139_, v___y_138_, v___y_140_, v___y_143_, v___y_144_, v___y_145_, v___x_148_);
v___x_150_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_150_, 0, v___x_149_);
lean_ctor_set(v___x_150_, 1, v___y_142_);
return v___x_150_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___boxed(lean_object* v_x_260_, lean_object* v_a_261_, lean_object* v_a_262_){
_start:
{
lean_object* v_res_263_; 
v_res_263_ = lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1(v_x_260_, v_a_261_, v_a_262_);
lean_dec_ref(v_a_261_);
return v_res_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Zify_mkZifyContext_spec__0(size_t v_sz_264_, size_t v_i_265_, lean_object* v_bs_266_){
_start:
{
uint8_t v___x_267_; 
v___x_267_ = lean_usize_dec_lt(v_i_265_, v_sz_264_);
if (v___x_267_ == 0)
{
return v_bs_266_;
}
else
{
lean_object* v_v_268_; lean_object* v___x_269_; lean_object* v_bs_x27_270_; size_t v___x_271_; size_t v___x_272_; lean_object* v___x_273_; 
v_v_268_ = lean_array_uget(v_bs_266_, v_i_265_);
v___x_269_ = lean_unsigned_to_nat(0u);
v_bs_x27_270_ = lean_array_uset(v_bs_266_, v_i_265_, v___x_269_);
v___x_271_ = ((size_t)1ULL);
v___x_272_ = lean_usize_add(v_i_265_, v___x_271_);
v___x_273_ = lean_array_uset(v_bs_x27_270_, v_i_265_, v_v_268_);
v_i_265_ = v___x_272_;
v_bs_266_ = v___x_273_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Zify_mkZifyContext_spec__0___boxed(lean_object* v_sz_275_, lean_object* v_i_276_, lean_object* v_bs_277_){
_start:
{
size_t v_sz_boxed_278_; size_t v_i_boxed_279_; lean_object* v_res_280_; 
v_sz_boxed_278_ = lean_unbox_usize(v_sz_275_);
lean_dec(v_sz_275_);
v_i_boxed_279_ = lean_unbox_usize(v_i_276_);
lean_dec(v_i_276_);
v_res_280_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Zify_mkZifyContext_spec__0(v_sz_boxed_278_, v_i_boxed_279_, v_bs_277_);
return v_res_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify_mkZifyContext(lean_object* v_simpArgs_282_, lean_object* v_a_283_, lean_object* v_a_284_, lean_object* v_a_285_, lean_object* v_a_286_, lean_object* v_a_287_, lean_object* v_a_288_, lean_object* v_a_289_, lean_object* v_a_290_){
_start:
{
lean_object* v___y_293_; 
if (lean_obj_tag(v_simpArgs_282_) == 0)
{
lean_object* v___x_352_; 
v___x_352_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__40));
v___y_293_ = v___x_352_;
goto v___jp_292_;
}
else
{
lean_object* v_val_353_; lean_object* v___x_354_; 
v_val_353_ = lean_ctor_get(v_simpArgs_282_, 0);
v___x_354_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_val_353_);
v___y_293_ = v___x_354_;
goto v___jp_292_;
}
v___jp_292_:
{
lean_object* v_ref_294_; lean_object* v_quotContext_295_; lean_object* v_currMacroScope_296_; uint8_t v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; size_t v_sz_339_; size_t v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; uint8_t v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; 
v_ref_294_ = lean_ctor_get(v_a_289_, 5);
v_quotContext_295_ = lean_ctor_get(v_a_289_, 10);
v_currMacroScope_296_ = lean_ctor_get(v_a_289_, 11);
v___x_297_ = 0;
v___x_298_ = l_Lean_SourceInfo_fromRef(v_ref_294_, v___x_297_);
v___x_299_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__2));
v___x_300_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__3));
lean_inc_n(v___x_298_, 19);
v___x_301_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_301_, 0, v___x_298_);
lean_ctor_set(v___x_301_, 1, v___x_299_);
v___x_302_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__5));
v___x_303_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__7));
v___x_304_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__9));
v___x_305_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__11));
v___x_306_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__12));
v___x_307_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_307_, 0, v___x_298_);
lean_ctor_set(v___x_307_, 1, v___x_306_);
v___x_308_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__14, &lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__14);
v___x_309_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__15));
lean_inc_n(v_currMacroScope_296_, 3);
lean_inc_n(v_quotContext_295_, 3);
v___x_310_ = l_Lean_addMacroScope(v_quotContext_295_, v___x_309_, v_currMacroScope_296_);
v___x_311_ = lean_box(0);
v___x_312_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__22));
v___x_313_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_313_, 0, v___x_298_);
lean_ctor_set(v___x_313_, 1, v___x_308_);
lean_ctor_set(v___x_313_, 2, v___x_310_);
lean_ctor_set(v___x_313_, 3, v___x_312_);
v___x_314_ = l_Lean_Syntax_node2(v___x_298_, v___x_305_, v___x_307_, v___x_313_);
v___x_315_ = l_Lean_Syntax_node1(v___x_298_, v___x_304_, v___x_314_);
v___x_316_ = l_Lean_Syntax_node1(v___x_298_, v___x_303_, v___x_315_);
v___x_317_ = l_Lean_Syntax_node1(v___x_298_, v___x_302_, v___x_316_);
v___x_318_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__23, &lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__23_once, _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__23);
v___x_319_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_319_, 0, v___x_298_);
lean_ctor_set(v___x_319_, 1, v___x_303_);
lean_ctor_set(v___x_319_, 2, v___x_318_);
v___x_320_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__24));
v___x_321_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_321_, 0, v___x_298_);
lean_ctor_set(v___x_321_, 1, v___x_320_);
v___x_322_ = l_Lean_Syntax_node1(v___x_298_, v___x_303_, v___x_321_);
v___x_323_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__25));
v___x_324_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_324_, 0, v___x_298_);
lean_ctor_set(v___x_324_, 1, v___x_323_);
v___x_325_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__27));
v___x_326_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__29, &lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__29_once, _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__29);
v___x_327_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__30));
v___x_328_ = l_Lean_addMacroScope(v_quotContext_295_, v___x_327_, v_currMacroScope_296_);
v___x_329_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_329_, 0, v___x_298_);
lean_ctor_set(v___x_329_, 1, v___x_326_);
lean_ctor_set(v___x_329_, 2, v___x_328_);
lean_ctor_set(v___x_329_, 3, v___x_311_);
lean_inc_ref_n(v___x_319_, 5);
v___x_330_ = l_Lean_Syntax_node3(v___x_298_, v___x_325_, v___x_319_, v___x_319_, v___x_329_);
v___x_331_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__31));
v___x_332_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_332_, 0, v___x_298_);
lean_ctor_set(v___x_332_, 1, v___x_331_);
v___x_333_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__33, &lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__33_once, _init_lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__33);
v___x_334_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__34));
v___x_335_ = l_Lean_addMacroScope(v_quotContext_295_, v___x_334_, v_currMacroScope_296_);
v___x_336_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_336_, 0, v___x_298_);
lean_ctor_set(v___x_336_, 1, v___x_333_);
lean_ctor_set(v___x_336_, 2, v___x_335_);
lean_ctor_set(v___x_336_, 3, v___x_311_);
v___x_337_ = l_Lean_Syntax_node3(v___x_298_, v___x_325_, v___x_319_, v___x_319_, v___x_336_);
lean_inc_ref(v___x_332_);
v___x_338_ = l_Array_mkArray4___redArg(v___x_330_, v___x_332_, v___x_337_, v___x_332_);
v_sz_339_ = lean_array_size(v___y_293_);
v___x_340_ = ((size_t)0ULL);
v___x_341_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Zify_mkZifyContext_spec__0(v_sz_339_, v___x_340_, v___y_293_);
v___x_342_ = l_Lean_Syntax_SepArray_ofElems(v___x_331_, v___x_341_);
lean_dec_ref(v___x_341_);
v___x_343_ = l_Array_append___redArg(v___x_338_, v___x_342_);
lean_dec_ref(v___x_342_);
v___x_344_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_344_, 0, v___x_298_);
lean_ctor_set(v___x_344_, 1, v___x_303_);
lean_ctor_set(v___x_344_, 2, v___x_343_);
v___x_345_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify___aux__Mathlib__Tactic__Zify______macroRules__Mathlib__Tactic__Zify__zify__1___closed__35));
v___x_346_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_346_, 0, v___x_298_);
lean_ctor_set(v___x_346_, 1, v___x_345_);
v___x_347_ = l_Lean_Syntax_node3(v___x_298_, v___x_303_, v___x_324_, v___x_344_, v___x_346_);
v___x_348_ = l_Lean_Syntax_node6(v___x_298_, v___x_300_, v___x_301_, v___x_317_, v___x_319_, v___x_322_, v___x_347_, v___x_319_);
v___x_349_ = 0;
v___x_350_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify_mkZifyContext___closed__0));
v___x_351_ = l_Lean_Elab_Tactic_mkSimpContext(v___x_348_, v___x_297_, v___x_349_, v___x_297_, v___x_350_, v_a_283_, v_a_284_, v_a_285_, v_a_286_, v_a_287_, v_a_288_, v_a_289_, v_a_290_);
lean_dec(v___x_348_);
return v___x_351_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify_mkZifyContext___boxed(lean_object* v_simpArgs_355_, lean_object* v_a_356_, lean_object* v_a_357_, lean_object* v_a_358_, lean_object* v_a_359_, lean_object* v_a_360_, lean_object* v_a_361_, lean_object* v_a_362_, lean_object* v_a_363_, lean_object* v_a_364_){
_start:
{
lean_object* v_res_365_; 
v_res_365_ = lp_mathlib_Mathlib_Tactic_Zify_mkZifyContext(v_simpArgs_355_, v_a_356_, v_a_357_, v_a_358_, v_a_359_, v_a_360_, v_a_361_, v_a_362_, v_a_363_);
lean_dec(v_a_363_);
lean_dec_ref(v_a_362_);
lean_dec(v_a_361_);
lean_dec_ref(v_a_360_);
lean_dec(v_a_359_);
lean_dec_ref(v_a_358_);
lean_dec(v_a_357_);
lean_dec_ref(v_a_356_);
lean_dec(v_simpArgs_355_);
return v_res_365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify_applySimpResultToProp_x27(lean_object* v_proof_366_, lean_object* v_prop_367_, lean_object* v_r_368_, lean_object* v_a_369_, lean_object* v_a_370_, lean_object* v_a_371_, lean_object* v_a_372_){
_start:
{
lean_object* v_proof_x3f_374_; 
v_proof_x3f_374_ = lean_ctor_get(v_r_368_, 1);
if (lean_obj_tag(v_proof_x3f_374_) == 0)
{
lean_object* v_expr_375_; uint8_t v___x_376_; 
v_expr_375_ = lean_ctor_get(v_r_368_, 0);
lean_inc_ref(v_expr_375_);
lean_dec_ref(v_r_368_);
v___x_376_ = lean_expr_eqv(v_expr_375_, v_prop_367_);
if (v___x_376_ == 0)
{
lean_object* v___x_377_; 
lean_inc_ref(v_expr_375_);
v___x_377_ = l_Lean_Meta_mkExpectedTypeHint(v_proof_366_, v_expr_375_, v_a_369_, v_a_370_, v_a_371_, v_a_372_);
if (lean_obj_tag(v___x_377_) == 0)
{
lean_object* v_a_378_; lean_object* v___x_380_; uint8_t v_isShared_381_; uint8_t v_isSharedCheck_386_; 
v_a_378_ = lean_ctor_get(v___x_377_, 0);
v_isSharedCheck_386_ = !lean_is_exclusive(v___x_377_);
if (v_isSharedCheck_386_ == 0)
{
v___x_380_ = v___x_377_;
v_isShared_381_ = v_isSharedCheck_386_;
goto v_resetjp_379_;
}
else
{
lean_inc(v_a_378_);
lean_dec(v___x_377_);
v___x_380_ = lean_box(0);
v_isShared_381_ = v_isSharedCheck_386_;
goto v_resetjp_379_;
}
v_resetjp_379_:
{
lean_object* v___x_382_; lean_object* v___x_384_; 
v___x_382_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_382_, 0, v_a_378_);
lean_ctor_set(v___x_382_, 1, v_expr_375_);
if (v_isShared_381_ == 0)
{
lean_ctor_set(v___x_380_, 0, v___x_382_);
v___x_384_ = v___x_380_;
goto v_reusejp_383_;
}
else
{
lean_object* v_reuseFailAlloc_385_; 
v_reuseFailAlloc_385_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_385_, 0, v___x_382_);
v___x_384_ = v_reuseFailAlloc_385_;
goto v_reusejp_383_;
}
v_reusejp_383_:
{
return v___x_384_;
}
}
}
else
{
lean_object* v_a_387_; lean_object* v___x_389_; uint8_t v_isShared_390_; uint8_t v_isSharedCheck_394_; 
lean_dec_ref(v_expr_375_);
v_a_387_ = lean_ctor_get(v___x_377_, 0);
v_isSharedCheck_394_ = !lean_is_exclusive(v___x_377_);
if (v_isSharedCheck_394_ == 0)
{
v___x_389_ = v___x_377_;
v_isShared_390_ = v_isSharedCheck_394_;
goto v_resetjp_388_;
}
else
{
lean_inc(v_a_387_);
lean_dec(v___x_377_);
v___x_389_ = lean_box(0);
v_isShared_390_ = v_isSharedCheck_394_;
goto v_resetjp_388_;
}
v_resetjp_388_:
{
lean_object* v___x_392_; 
if (v_isShared_390_ == 0)
{
v___x_392_ = v___x_389_;
goto v_reusejp_391_;
}
else
{
lean_object* v_reuseFailAlloc_393_; 
v_reuseFailAlloc_393_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_393_, 0, v_a_387_);
v___x_392_ = v_reuseFailAlloc_393_;
goto v_reusejp_391_;
}
v_reusejp_391_:
{
return v___x_392_;
}
}
}
}
else
{
lean_object* v___x_395_; lean_object* v___x_396_; 
v___x_395_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_395_, 0, v_proof_366_);
lean_ctor_set(v___x_395_, 1, v_expr_375_);
v___x_396_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_396_, 0, v___x_395_);
return v___x_396_;
}
}
else
{
lean_object* v_expr_397_; lean_object* v_val_398_; lean_object* v___x_399_; 
lean_inc_ref(v_proof_x3f_374_);
v_expr_397_ = lean_ctor_get(v_r_368_, 0);
lean_inc_ref(v_expr_397_);
lean_dec_ref(v_r_368_);
v_val_398_ = lean_ctor_get(v_proof_x3f_374_, 0);
lean_inc(v_val_398_);
lean_dec_ref_known(v_proof_x3f_374_, 1);
v___x_399_ = l_Lean_Meta_mkEqMP(v_val_398_, v_proof_366_, v_a_369_, v_a_370_, v_a_371_, v_a_372_);
if (lean_obj_tag(v___x_399_) == 0)
{
lean_object* v_a_400_; lean_object* v___x_401_; 
v_a_400_ = lean_ctor_get(v___x_399_, 0);
lean_inc(v_a_400_);
lean_dec_ref_known(v___x_399_, 1);
lean_inc_ref(v_expr_397_);
v___x_401_ = l_Lean_Meta_mkExpectedTypeHint(v_a_400_, v_expr_397_, v_a_369_, v_a_370_, v_a_371_, v_a_372_);
if (lean_obj_tag(v___x_401_) == 0)
{
lean_object* v_a_402_; lean_object* v___x_404_; uint8_t v_isShared_405_; uint8_t v_isSharedCheck_410_; 
v_a_402_ = lean_ctor_get(v___x_401_, 0);
v_isSharedCheck_410_ = !lean_is_exclusive(v___x_401_);
if (v_isSharedCheck_410_ == 0)
{
v___x_404_ = v___x_401_;
v_isShared_405_ = v_isSharedCheck_410_;
goto v_resetjp_403_;
}
else
{
lean_inc(v_a_402_);
lean_dec(v___x_401_);
v___x_404_ = lean_box(0);
v_isShared_405_ = v_isSharedCheck_410_;
goto v_resetjp_403_;
}
v_resetjp_403_:
{
lean_object* v___x_406_; lean_object* v___x_408_; 
v___x_406_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_406_, 0, v_a_402_);
lean_ctor_set(v___x_406_, 1, v_expr_397_);
if (v_isShared_405_ == 0)
{
lean_ctor_set(v___x_404_, 0, v___x_406_);
v___x_408_ = v___x_404_;
goto v_reusejp_407_;
}
else
{
lean_object* v_reuseFailAlloc_409_; 
v_reuseFailAlloc_409_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_409_, 0, v___x_406_);
v___x_408_ = v_reuseFailAlloc_409_;
goto v_reusejp_407_;
}
v_reusejp_407_:
{
return v___x_408_;
}
}
}
else
{
lean_object* v_a_411_; lean_object* v___x_413_; uint8_t v_isShared_414_; uint8_t v_isSharedCheck_418_; 
lean_dec_ref(v_expr_397_);
v_a_411_ = lean_ctor_get(v___x_401_, 0);
v_isSharedCheck_418_ = !lean_is_exclusive(v___x_401_);
if (v_isSharedCheck_418_ == 0)
{
v___x_413_ = v___x_401_;
v_isShared_414_ = v_isSharedCheck_418_;
goto v_resetjp_412_;
}
else
{
lean_inc(v_a_411_);
lean_dec(v___x_401_);
v___x_413_ = lean_box(0);
v_isShared_414_ = v_isSharedCheck_418_;
goto v_resetjp_412_;
}
v_resetjp_412_:
{
lean_object* v___x_416_; 
if (v_isShared_414_ == 0)
{
v___x_416_ = v___x_413_;
goto v_reusejp_415_;
}
else
{
lean_object* v_reuseFailAlloc_417_; 
v_reuseFailAlloc_417_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_417_, 0, v_a_411_);
v___x_416_ = v_reuseFailAlloc_417_;
goto v_reusejp_415_;
}
v_reusejp_415_:
{
return v___x_416_;
}
}
}
}
else
{
lean_object* v_a_419_; lean_object* v___x_421_; uint8_t v_isShared_422_; uint8_t v_isSharedCheck_426_; 
lean_dec_ref(v_expr_397_);
v_a_419_ = lean_ctor_get(v___x_399_, 0);
v_isSharedCheck_426_ = !lean_is_exclusive(v___x_399_);
if (v_isSharedCheck_426_ == 0)
{
v___x_421_ = v___x_399_;
v_isShared_422_ = v_isSharedCheck_426_;
goto v_resetjp_420_;
}
else
{
lean_inc(v_a_419_);
lean_dec(v___x_399_);
v___x_421_ = lean_box(0);
v_isShared_422_ = v_isSharedCheck_426_;
goto v_resetjp_420_;
}
v_resetjp_420_:
{
lean_object* v___x_424_; 
if (v_isShared_422_ == 0)
{
v___x_424_ = v___x_421_;
goto v_reusejp_423_;
}
else
{
lean_object* v_reuseFailAlloc_425_; 
v_reuseFailAlloc_425_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_425_, 0, v_a_419_);
v___x_424_ = v_reuseFailAlloc_425_;
goto v_reusejp_423_;
}
v_reusejp_423_:
{
return v___x_424_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify_applySimpResultToProp_x27___boxed(lean_object* v_proof_427_, lean_object* v_prop_428_, lean_object* v_r_429_, lean_object* v_a_430_, lean_object* v_a_431_, lean_object* v_a_432_, lean_object* v_a_433_, lean_object* v_a_434_){
_start:
{
lean_object* v_res_435_; 
v_res_435_ = lp_mathlib_Mathlib_Tactic_Zify_applySimpResultToProp_x27(v_proof_427_, v_prop_428_, v_r_429_, v_a_430_, v_a_431_, v_a_432_, v_a_433_);
lean_dec(v_a_433_);
lean_dec_ref(v_a_432_);
lean_dec(v_a_431_);
lean_dec_ref(v_a_430_);
lean_dec_ref(v_prop_428_);
return v_res_435_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__1(void){
_start:
{
lean_object* v___x_438_; 
v___x_438_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_438_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__2(void){
_start:
{
lean_object* v___x_439_; lean_object* v___x_440_; 
v___x_439_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__1, &lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__1);
v___x_440_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_440_, 0, v___x_439_);
return v___x_440_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__3(void){
_start:
{
lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; 
v___x_441_ = lean_unsigned_to_nat(0u);
v___x_442_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__2, &lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__2);
v___x_443_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_443_, 0, v___x_442_);
lean_ctor_set(v___x_443_, 1, v___x_441_);
return v___x_443_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__4(void){
_start:
{
lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___x_446_; 
v___x_444_ = lean_unsigned_to_nat(32u);
v___x_445_ = lean_mk_empty_array_with_capacity(v___x_444_);
v___x_446_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_446_, 0, v___x_445_);
return v___x_446_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__5(void){
_start:
{
size_t v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; 
v___x_447_ = ((size_t)5ULL);
v___x_448_ = lean_unsigned_to_nat(0u);
v___x_449_ = lean_unsigned_to_nat(32u);
v___x_450_ = lean_mk_empty_array_with_capacity(v___x_449_);
v___x_451_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__4, &lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__4);
v___x_452_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_452_, 0, v___x_451_);
lean_ctor_set(v___x_452_, 1, v___x_450_);
lean_ctor_set(v___x_452_, 2, v___x_448_);
lean_ctor_set(v___x_452_, 3, v___x_448_);
lean_ctor_set_usize(v___x_452_, 4, v___x_447_);
return v___x_452_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__6(void){
_start:
{
lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; 
v___x_453_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__5, &lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__5);
v___x_454_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__2, &lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__2);
v___x_455_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_455_, 0, v___x_454_);
lean_ctor_set(v___x_455_, 1, v___x_454_);
lean_ctor_set(v___x_455_, 2, v___x_454_);
lean_ctor_set(v___x_455_, 3, v___x_453_);
return v___x_455_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__7(void){
_start:
{
lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; 
v___x_456_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__6, &lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__6);
v___x_457_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__3, &lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__3);
v___x_458_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_458_, 0, v___x_457_);
lean_ctor_set(v___x_458_, 1, v___x_456_);
return v___x_458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify_zifyProof(lean_object* v_simpArgs_459_, lean_object* v_proof_460_, lean_object* v_prop_461_, lean_object* v_a_462_, lean_object* v_a_463_, lean_object* v_a_464_, lean_object* v_a_465_, lean_object* v_a_466_, lean_object* v_a_467_, lean_object* v_a_468_, lean_object* v_a_469_){
_start:
{
lean_object* v___x_471_; 
v___x_471_ = lp_mathlib_Mathlib_Tactic_Zify_mkZifyContext(v_simpArgs_459_, v_a_462_, v_a_463_, v_a_464_, v_a_465_, v_a_466_, v_a_467_, v_a_468_, v_a_469_);
if (lean_obj_tag(v___x_471_) == 0)
{
lean_object* v_a_472_; lean_object* v_ctx_473_; lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; 
v_a_472_ = lean_ctor_get(v___x_471_, 0);
lean_inc(v_a_472_);
lean_dec_ref_known(v___x_471_, 1);
v_ctx_473_ = lean_ctor_get(v_a_472_, 0);
lean_inc_ref(v_ctx_473_);
lean_dec(v_a_472_);
v___x_474_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__0));
v___x_475_ = lean_box(0);
v___x_476_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__7, &lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_Zify_zifyProof___closed__7);
lean_inc_ref(v_prop_461_);
v___x_477_ = l_Lean_Meta_simp(v_prop_461_, v_ctx_473_, v___x_474_, v___x_475_, v___x_476_, v_a_466_, v_a_467_, v_a_468_, v_a_469_);
if (lean_obj_tag(v___x_477_) == 0)
{
lean_object* v_a_478_; lean_object* v_fst_479_; lean_object* v___x_480_; 
v_a_478_ = lean_ctor_get(v___x_477_, 0);
lean_inc(v_a_478_);
lean_dec_ref_known(v___x_477_, 1);
v_fst_479_ = lean_ctor_get(v_a_478_, 0);
lean_inc(v_fst_479_);
lean_dec(v_a_478_);
v___x_480_ = lp_mathlib_Mathlib_Tactic_Zify_applySimpResultToProp_x27(v_proof_460_, v_prop_461_, v_fst_479_, v_a_466_, v_a_467_, v_a_468_, v_a_469_);
lean_dec_ref(v_prop_461_);
return v___x_480_;
}
else
{
lean_object* v_a_481_; lean_object* v___x_483_; uint8_t v_isShared_484_; uint8_t v_isSharedCheck_488_; 
lean_dec_ref(v_prop_461_);
lean_dec_ref(v_proof_460_);
v_a_481_ = lean_ctor_get(v___x_477_, 0);
v_isSharedCheck_488_ = !lean_is_exclusive(v___x_477_);
if (v_isSharedCheck_488_ == 0)
{
v___x_483_ = v___x_477_;
v_isShared_484_ = v_isSharedCheck_488_;
goto v_resetjp_482_;
}
else
{
lean_inc(v_a_481_);
lean_dec(v___x_477_);
v___x_483_ = lean_box(0);
v_isShared_484_ = v_isSharedCheck_488_;
goto v_resetjp_482_;
}
v_resetjp_482_:
{
lean_object* v___x_486_; 
if (v_isShared_484_ == 0)
{
v___x_486_ = v___x_483_;
goto v_reusejp_485_;
}
else
{
lean_object* v_reuseFailAlloc_487_; 
v_reuseFailAlloc_487_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_487_, 0, v_a_481_);
v___x_486_ = v_reuseFailAlloc_487_;
goto v_reusejp_485_;
}
v_reusejp_485_:
{
return v___x_486_;
}
}
}
}
else
{
lean_object* v_a_489_; lean_object* v___x_491_; uint8_t v_isShared_492_; uint8_t v_isSharedCheck_496_; 
lean_dec_ref(v_prop_461_);
lean_dec_ref(v_proof_460_);
v_a_489_ = lean_ctor_get(v___x_471_, 0);
v_isSharedCheck_496_ = !lean_is_exclusive(v___x_471_);
if (v_isSharedCheck_496_ == 0)
{
v___x_491_ = v___x_471_;
v_isShared_492_ = v_isSharedCheck_496_;
goto v_resetjp_490_;
}
else
{
lean_inc(v_a_489_);
lean_dec(v___x_471_);
v___x_491_ = lean_box(0);
v_isShared_492_ = v_isSharedCheck_496_;
goto v_resetjp_490_;
}
v_resetjp_490_:
{
lean_object* v___x_494_; 
if (v_isShared_492_ == 0)
{
v___x_494_ = v___x_491_;
goto v_reusejp_493_;
}
else
{
lean_object* v_reuseFailAlloc_495_; 
v_reuseFailAlloc_495_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_495_, 0, v_a_489_);
v___x_494_ = v_reuseFailAlloc_495_;
goto v_reusejp_493_;
}
v_reusejp_493_:
{
return v___x_494_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Zify_zifyProof___boxed(lean_object* v_simpArgs_497_, lean_object* v_proof_498_, lean_object* v_prop_499_, lean_object* v_a_500_, lean_object* v_a_501_, lean_object* v_a_502_, lean_object* v_a_503_, lean_object* v_a_504_, lean_object* v_a_505_, lean_object* v_a_506_, lean_object* v_a_507_, lean_object* v_a_508_){
_start:
{
lean_object* v_res_509_; 
v_res_509_ = lp_mathlib_Mathlib_Tactic_Zify_zifyProof(v_simpArgs_497_, v_proof_498_, v_prop_499_, v_a_500_, v_a_501_, v_a_502_, v_a_503_, v_a_504_, v_a_505_, v_a_506_, v_a_507_);
lean_dec(v_a_507_);
lean_dec_ref(v_a_506_);
lean_dec(v_a_505_);
lean_dec_ref(v_a_504_);
lean_dec(v_a_503_);
lean_dec_ref(v_a_502_);
lean_dec(v_a_501_);
lean_dec_ref(v_a_500_);
lean_dec(v_simpArgs_497_);
return v_res_509_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Zify(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ToAdditive(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ToDual(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Zify(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ToAdditive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ToDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_Zify_zify = _init_lp_mathlib_Mathlib_Tactic_Zify_zify();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Zify_zify);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Cast_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToAdditive(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToDual(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Zify(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToAdditive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Zify(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Zify(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Zify(builtin);
}
#ifdef __cplusplus
}
#endif
