// Lean compiler output
// Module: Mathlib.Tactic.Group
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Commutator public import Mathlib.Algebra.Order.Sub.Basic public import Mathlib.Tactic.FailIfNoProgress public import Mathlib.Tactic.Ring
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Array_mkArray1___redArg(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Tactic_location;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group_group___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group_group___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group_group___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group_group___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group_group___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Group"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group_group___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group_group___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group_group___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group_group___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group_group___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group_group___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__2_value),LEAN_SCALAR_PTR_LITERAL(105, 28, 189, 37, 66, 9, 55, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group_group___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__3_value),LEAN_SCALAR_PTR_LITERAL(20, 3, 115, 244, 56, 190, 13, 255)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group_group___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group_group___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group_group___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group_group___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__5_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group_group___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group_group___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__3_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group_group___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group_group___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group_group___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group_group___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__8_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group_group___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__9_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group_group___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group_group___closed__10;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group_group___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group_group___closed__11;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group_group___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group_group___closed__12;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Group_group;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "<;>"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "RingNF"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "ringNF"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(132, 186, 29, 238, 234, 106, 109, 235)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(169, 58, 229, 242, 41, 102, 20, 168)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ring_nf"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "valConfigItem"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "ifUnchanged"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__7;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(60, 106, 111, 249, 87, 213, 100, 133)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ":="};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "dotIdent"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "."};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "silent"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__13_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__14;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(126, 4, 152, 130, 182, 68, 169, 162)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "fail"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "str"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__18_value),LEAN_SCALAR_PTR_LITERAL(255, 188, 142, 1, 190, 33, 34, 128)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "\"`group` made no progress\""};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__20_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__21_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__22_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "first"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__24_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__24_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__24_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__22_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__24_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__24_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__24_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__23_value),LEAN_SCALAR_PTR_LITERAL(59, 232, 35, 17, 172, 62, 48, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__25_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__3_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__27_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__28_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__30_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__30_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__30_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__22_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__30_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__30_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__30_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__29_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__30_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__31_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__32_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__32_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__32_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__22_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__32_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__32_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__32_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__31_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__32_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "tacticRepeat1_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__33_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__34_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__34_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__34_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__34_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__33_value),LEAN_SCALAR_PTR_LITERAL(57, 234, 167, 155, 111, 232, 12, 212)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__34_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "repeat1"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__35_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "failIfNoProgress"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__36_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__37_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__37_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__37_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__37_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__36_value),LEAN_SCALAR_PTR_LITERAL(238, 120, 52, 11, 174, 48, 92, 172)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__37_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "fail_if_no_progress"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__38_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__39_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__40_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__40_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__40_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__22_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__40_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__40_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__40_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__39_value),LEAN_SCALAR_PTR_LITERAL(117, 253, 122, 28, 77, 248, 149, 120)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__40_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__41_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "tactic_<;>_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__42_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__43_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__43_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__43_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__22_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__43_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__43_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__43_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__42_value),LEAN_SCALAR_PTR_LITERAL(31, 118, 44, 159, 195, 11, 47, 176)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__43 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__43_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__44 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__44_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__45_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__45_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__45_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__22_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__45_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__45_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__45_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__44_value),LEAN_SCALAR_PTR_LITERAL(50, 13, 241, 145, 67, 153, 105, 177)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__45 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__45_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__46 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__46_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__47_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__47_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__47_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__22_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__47_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__47_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__47_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__46_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__47 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__47_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "configItem"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__48 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__48_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__49_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__49_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__49_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__22_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__49_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__49_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__49_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__48_value),LEAN_SCALAR_PTR_LITERAL(205, 9, 236, 192, 59, 252, 178, 140)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__49 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__49_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "negConfigItem"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__50 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__50_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__51_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__51_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__51_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__22_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__51_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__51_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__51_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__50_value),LEAN_SCALAR_PTR_LITERAL(196, 29, 29, 161, 247, 206, 181, 221)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__51 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__51_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "-"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__52 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__52_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "decide"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__53 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__53_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__54_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__54;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__53_value),LEAN_SCALAR_PTR_LITERAL(236, 252, 83, 10, 217, 228, 80, 149)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__55 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__55_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Decidable"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__56 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__56_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__57_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__56_value),LEAN_SCALAR_PTR_LITERAL(87, 187, 205, 215, 218, 218, 68, 60)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__57_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__53_value),LEAN_SCALAR_PTR_LITERAL(16, 96, 65, 173, 152, 155, 4, 222)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__57 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__57_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__57_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__58 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__58_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__59_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__59_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__59_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__22_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__59_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__59_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__59_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__53_value),LEAN_SCALAR_PTR_LITERAL(53, 158, 1, 232, 101, 200, 191, 197)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__59 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__59_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__59_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__60 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__60_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__61_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__60_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__61 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__61_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__62_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__58_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__61_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__62 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__62_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__63_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "failIfUnchanged"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__63 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__63_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__64_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__64;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__65_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__63_value),LEAN_SCALAR_PTR_LITERAL(6, 104, 167, 161, 191, 186, 8, 81)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__65 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__65_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__66_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__66;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__67_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "only"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__67 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__67_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__68_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__68 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__68_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__69_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "simpLemma"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__69 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__69_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__70_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__70_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__70_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__22_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__70_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__70_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__70_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__70_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__69_value),LEAN_SCALAR_PTR_LITERAL(38, 215, 101, 250, 181, 108, 118, 102)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__70 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__70_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__71_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "commutatorElement_def"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__71 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__71_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__72_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__72;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__73_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__71_value),LEAN_SCALAR_PTR_LITERAL(252, 16, 93, 165, 84, 200, 168, 12)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__73 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__73_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__74_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__73_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__74 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__74_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__75_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__74_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__75 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__75_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__76_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__76 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__76_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__77_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "mul_one"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__77 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__77_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__78_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__78;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__79_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__77_value),LEAN_SCALAR_PTR_LITERAL(185, 178, 196, 247, 70, 46, 81, 207)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__79 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__79_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__80_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__79_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__80 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__80_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__81_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__80_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__81 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__81_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__82_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "one_mul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__82 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__82_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__83_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__83;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__84_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__82_value),LEAN_SCALAR_PTR_LITERAL(211, 55, 221, 12, 196, 98, 247, 238)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__84 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__84_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__85_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__84_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__85 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__85_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__86_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__85_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__86 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__86_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__87_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "←"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__87 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__87_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__88_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "zpow_neg_one"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__88 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__88_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__89_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__89;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__90_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__88_value),LEAN_SCALAR_PTR_LITERAL(14, 73, 250, 138, 87, 37, 165, 32)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__90 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__90_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__91_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__90_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__91 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__91_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__92_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__91_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__92 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__92_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__93_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "zpow_natCast"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__93 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__93_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__94_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__94;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__95_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__93_value),LEAN_SCALAR_PTR_LITERAL(108, 241, 70, 173, 235, 189, 29, 2)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__95 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__95_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__96_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__95_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__96 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__96_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__97_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__96_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__97 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__97_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__98_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "zpow_mul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__98 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__98_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__99_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__99;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__100_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__98_value),LEAN_SCALAR_PTR_LITERAL(179, 169, 230, 36, 36, 125, 61, 50)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__100 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__100_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__101_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__100_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__101 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__101_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__102_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__101_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__102 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__102_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__103_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "Int.natCast_add"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__103 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__103_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__104_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__104;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__105_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Int"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__105 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__105_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__106_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "natCast_add"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__106 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__106_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__107_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__105_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__107_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__107_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__106_value),LEAN_SCALAR_PTR_LITERAL(245, 136, 148, 230, 33, 83, 184, 22)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__107 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__107_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__108_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__107_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__108 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__108_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__109_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__108_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__109 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__109_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__110_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "Int.natCast_mul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__110 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__110_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__111_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__111;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__112_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "natCast_mul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__112 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__112_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__113_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__105_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__113_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__113_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__112_value),LEAN_SCALAR_PTR_LITERAL(133, 50, 217, 182, 79, 25, 170, 123)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__113 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__113_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__114_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__113_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__114 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__114_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__115_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__114_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__115 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__115_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__116_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Int.mul_neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__116 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__116_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__117_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__117;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__118_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "mul_neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__118 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__118_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__119_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__105_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__119_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__119_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__118_value),LEAN_SCALAR_PTR_LITERAL(5, 67, 79, 42, 55, 144, 96, 112)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__119 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__119_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__120_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__119_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__120 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__120_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__121_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__120_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__121 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__121_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__122_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Int.neg_mul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__122 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__122_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__123_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__123;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__124_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "neg_mul"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__124 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__124_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__125_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__105_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__125_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__125_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__124_value),LEAN_SCALAR_PTR_LITERAL(178, 134, 178, 152, 255, 117, 101, 193)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__125 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__125_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__126_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__125_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__126 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__126_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__127_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__126_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__127 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__127_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__128_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "neg_neg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__128 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__128_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__129_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__129;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__130_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__128_value),LEAN_SCALAR_PTR_LITERAL(74, 57, 79, 148, 171, 238, 100, 82)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__130 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__130_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__131_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__130_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__131 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__131_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__132_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__131_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__132 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__132_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__133_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "one_zpow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__133 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__133_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__134_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__134;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__135_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__133_value),LEAN_SCALAR_PTR_LITERAL(54, 181, 206, 14, 73, 75, 116, 63)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__135 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__135_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__136_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__135_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__136 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__136_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__137_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__136_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__137 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__137_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__138_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "zpow_zero"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__138 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__138_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__139_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__139;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__140_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__138_value),LEAN_SCALAR_PTR_LITERAL(120, 225, 61, 254, 88, 231, 71, 52)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__140 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__140_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__141_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__140_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__141 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__141_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__142_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__141_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__142 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__142_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__143_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "zpow_one"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__143 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__143_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__144_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__144;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__145_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__143_value),LEAN_SCALAR_PTR_LITERAL(102, 221, 55, 131, 56, 239, 55, 79)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__145 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__145_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__146_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__145_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__146 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__146_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__147_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__146_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__147 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__147_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__148_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "mul_zpow_neg_one"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__148 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__148_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__149_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__149;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__150_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__148_value),LEAN_SCALAR_PTR_LITERAL(246, 85, 11, 156, 215, 40, 174, 96)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__150 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__150_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__151_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__150_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__151 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__151_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__152_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__151_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__152 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__152_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__153_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "mul_assoc"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__153 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__153_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__154_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__154;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__155_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__153_value),LEAN_SCALAR_PTR_LITERAL(18, 21, 243, 68, 110, 93, 69, 52)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__155 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__155_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__156_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__155_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__156 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__156_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__157_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__156_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__157 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__157_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__158_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "zpow_add"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__158 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__158_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__159_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__159;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__160_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__158_value),LEAN_SCALAR_PTR_LITERAL(164, 94, 242, 204, 56, 189, 112, 18)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__160 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__160_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__161_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__160_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__161 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__161_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__162_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__161_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__162 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__162_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__163_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "zpow_add_one"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__163 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__163_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__164_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__164;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__165_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__163_value),LEAN_SCALAR_PTR_LITERAL(229, 131, 146, 75, 106, 213, 160, 75)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__165 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__165_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__166_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__165_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__166 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__166_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__167_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__166_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__167 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__167_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__168_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "zpow_one_add"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__168 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__168_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__169_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__169;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__170_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__168_value),LEAN_SCALAR_PTR_LITERAL(191, 163, 18, 134, 122, 60, 57, 77)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__170 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__170_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__171_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__170_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__171 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__171_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__172_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__171_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__172 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__172_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__173_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "_zpow_trick"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__173 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__173_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__174_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__174;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__175_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__173_value),LEAN_SCALAR_PTR_LITERAL(102, 120, 124, 103, 7, 114, 55, 51)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__175 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__175_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__176_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__176_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__176_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__176_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__176_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__2_value),LEAN_SCALAR_PTR_LITERAL(105, 28, 189, 37, 66, 9, 55, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__176_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__176_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__173_value),LEAN_SCALAR_PTR_LITERAL(204, 213, 207, 20, 49, 23, 231, 56)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__176 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__176_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__177_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__176_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__177 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__177_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__178_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__177_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__178 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__178_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__179_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "_zpow_trick_one"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__179 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__179_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__180_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__180;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__181_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__179_value),LEAN_SCALAR_PTR_LITERAL(12, 80, 186, 104, 71, 53, 121, 162)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__181 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__181_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__182_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__182_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__182_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__182_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__182_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__2_value),LEAN_SCALAR_PTR_LITERAL(105, 28, 189, 37, 66, 9, 55, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__182_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__182_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__179_value),LEAN_SCALAR_PTR_LITERAL(38, 124, 74, 241, 194, 73, 106, 176)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__182 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__182_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__183_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__182_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__183 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__183_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__184_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__183_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__184 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__184_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__185_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "_zpow_trick_one'"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__185 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__185_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__186_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__186;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__187_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__185_value),LEAN_SCALAR_PTR_LITERAL(5, 38, 113, 57, 181, 254, 46, 44)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__187 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__187_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__188_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__188_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__188_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__188_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__188_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group_group___closed__2_value),LEAN_SCALAR_PTR_LITERAL(105, 28, 189, 37, 66, 9, 55, 97)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__188_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__188_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__185_value),LEAN_SCALAR_PTR_LITERAL(159, 120, 78, 112, 0, 87, 108, 18)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__188 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__188_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__189_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__188_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__189 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__189_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__190_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__189_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__190 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__190_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__191_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tsub_self"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__191 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__191_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__192_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__192;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__193_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__191_value),LEAN_SCALAR_PTR_LITERAL(133, 169, 79, 239, 212, 233, 242, 52)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__193 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__193_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__194_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__193_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__194 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__194_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__195_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__194_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__195 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__195_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__196_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "sub_self"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__196 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__196_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__197_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__197;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__198_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__196_value),LEAN_SCALAR_PTR_LITERAL(3, 129, 97, 139, 162, 129, 227, 32)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__198 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__198_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__199_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__198_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__199 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__199_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__200_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__199_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__200 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__200_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__201_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "add_neg_cancel"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__201 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__201_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__202_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__202;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__203_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__201_value),LEAN_SCALAR_PTR_LITERAL(158, 173, 31, 168, 70, 71, 235, 7)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__203 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__203_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__204_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__203_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__204 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__204_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__205_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__204_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__205 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__205_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__206_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "neg_add_cancel"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__206 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__206_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__207_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__207;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__208_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__206_value),LEAN_SCALAR_PTR_LITERAL(103, 100, 227, 22, 242, 179, 118, 145)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__208 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__208_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__209_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__208_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__209 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__209_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__210_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__209_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__210 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__210_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__211_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__211 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__211_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__212_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__212 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__212_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group_group___closed__10(void){
_start:
{
lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v___x_21_; 
v___x_19_ = l_Lean_Parser_Tactic_location;
v___x_20_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group_group___closed__9));
v___x_21_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_21_, 0, v___x_20_);
lean_ctor_set(v___x_21_, 1, v___x_19_);
return v___x_21_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group_group___closed__11(void){
_start:
{
lean_object* v___x_22_; lean_object* v___x_23_; lean_object* v___x_24_; lean_object* v___x_25_; 
v___x_22_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group_group___closed__10, &lp_mathlib_Mathlib_Tactic_Group_group___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_Group_group___closed__10);
v___x_23_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group_group___closed__7));
v___x_24_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group_group___closed__6));
v___x_25_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_25_, 0, v___x_24_);
lean_ctor_set(v___x_25_, 1, v___x_23_);
lean_ctor_set(v___x_25_, 2, v___x_22_);
return v___x_25_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group_group___closed__12(void){
_start:
{
lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v___x_28_; lean_object* v___x_29_; 
v___x_26_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group_group___closed__11, &lp_mathlib_Mathlib_Tactic_Group_group___closed__11_once, _init_lp_mathlib_Mathlib_Tactic_Group_group___closed__11);
v___x_27_ = lean_unsigned_to_nat(1022u);
v___x_28_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group_group___closed__4));
v___x_29_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_29_, 0, v___x_28_);
lean_ctor_set(v___x_29_, 1, v___x_27_);
lean_ctor_set(v___x_29_, 2, v___x_26_);
return v___x_29_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group_group(void){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group_group___closed__12, &lp_mathlib_Mathlib_Tactic_Group_group___closed__12_once, _init_lp_mathlib_Mathlib_Tactic_Group_group___closed__12);
return v___x_30_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__7(void){
_start:
{
lean_object* v___x_42_; lean_object* v___x_43_; 
v___x_42_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__6));
v___x_43_ = l_String_toRawSubstring_x27(v___x_42_);
return v___x_43_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__14(void){
_start:
{
lean_object* v___x_51_; lean_object* v___x_52_; 
v___x_51_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__13));
v___x_52_ = l_String_toRawSubstring_x27(v___x_51_);
return v___x_52_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__54(void){
_start:
{
lean_object* v___x_138_; lean_object* v___x_139_; 
v___x_138_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__53));
v___x_139_ = l_String_toRawSubstring_x27(v___x_138_);
return v___x_139_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__64(void){
_start:
{
lean_object* v___x_164_; lean_object* v___x_165_; 
v___x_164_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__63));
v___x_165_ = l_String_toRawSubstring_x27(v___x_164_);
return v___x_165_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__66(void){
_start:
{
lean_object* v___x_168_; 
v___x_168_ = l_Array_mkArray0(lean_box(0));
return v___x_168_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__72(void){
_start:
{
lean_object* v___x_178_; lean_object* v___x_179_; 
v___x_178_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__71));
v___x_179_ = l_String_toRawSubstring_x27(v___x_178_);
return v___x_179_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__78(void){
_start:
{
lean_object* v___x_190_; lean_object* v___x_191_; 
v___x_190_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__77));
v___x_191_ = l_String_toRawSubstring_x27(v___x_190_);
return v___x_191_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__83(void){
_start:
{
lean_object* v___x_201_; lean_object* v___x_202_; 
v___x_201_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__82));
v___x_202_ = l_String_toRawSubstring_x27(v___x_201_);
return v___x_202_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__89(void){
_start:
{
lean_object* v___x_213_; lean_object* v___x_214_; 
v___x_213_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__88));
v___x_214_ = l_String_toRawSubstring_x27(v___x_213_);
return v___x_214_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__94(void){
_start:
{
lean_object* v___x_224_; lean_object* v___x_225_; 
v___x_224_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__93));
v___x_225_ = l_String_toRawSubstring_x27(v___x_224_);
return v___x_225_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__99(void){
_start:
{
lean_object* v___x_235_; lean_object* v___x_236_; 
v___x_235_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__98));
v___x_236_ = l_String_toRawSubstring_x27(v___x_235_);
return v___x_236_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__104(void){
_start:
{
lean_object* v___x_246_; lean_object* v___x_247_; 
v___x_246_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__103));
v___x_247_ = l_String_toRawSubstring_x27(v___x_246_);
return v___x_247_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__111(void){
_start:
{
lean_object* v___x_260_; lean_object* v___x_261_; 
v___x_260_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__110));
v___x_261_ = l_String_toRawSubstring_x27(v___x_260_);
return v___x_261_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__117(void){
_start:
{
lean_object* v___x_273_; lean_object* v___x_274_; 
v___x_273_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__116));
v___x_274_ = l_String_toRawSubstring_x27(v___x_273_);
return v___x_274_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__123(void){
_start:
{
lean_object* v___x_286_; lean_object* v___x_287_; 
v___x_286_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__122));
v___x_287_ = l_String_toRawSubstring_x27(v___x_286_);
return v___x_287_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__129(void){
_start:
{
lean_object* v___x_299_; lean_object* v___x_300_; 
v___x_299_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__128));
v___x_300_ = l_String_toRawSubstring_x27(v___x_299_);
return v___x_300_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__134(void){
_start:
{
lean_object* v___x_310_; lean_object* v___x_311_; 
v___x_310_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__133));
v___x_311_ = l_String_toRawSubstring_x27(v___x_310_);
return v___x_311_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__139(void){
_start:
{
lean_object* v___x_321_; lean_object* v___x_322_; 
v___x_321_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__138));
v___x_322_ = l_String_toRawSubstring_x27(v___x_321_);
return v___x_322_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__144(void){
_start:
{
lean_object* v___x_332_; lean_object* v___x_333_; 
v___x_332_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__143));
v___x_333_ = l_String_toRawSubstring_x27(v___x_332_);
return v___x_333_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__149(void){
_start:
{
lean_object* v___x_343_; lean_object* v___x_344_; 
v___x_343_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__148));
v___x_344_ = l_String_toRawSubstring_x27(v___x_343_);
return v___x_344_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__154(void){
_start:
{
lean_object* v___x_354_; lean_object* v___x_355_; 
v___x_354_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__153));
v___x_355_ = l_String_toRawSubstring_x27(v___x_354_);
return v___x_355_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__159(void){
_start:
{
lean_object* v___x_365_; lean_object* v___x_366_; 
v___x_365_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__158));
v___x_366_ = l_String_toRawSubstring_x27(v___x_365_);
return v___x_366_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__164(void){
_start:
{
lean_object* v___x_376_; lean_object* v___x_377_; 
v___x_376_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__163));
v___x_377_ = l_String_toRawSubstring_x27(v___x_376_);
return v___x_377_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__169(void){
_start:
{
lean_object* v___x_387_; lean_object* v___x_388_; 
v___x_387_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__168));
v___x_388_ = l_String_toRawSubstring_x27(v___x_387_);
return v___x_388_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__174(void){
_start:
{
lean_object* v___x_398_; lean_object* v___x_399_; 
v___x_398_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__173));
v___x_399_ = l_String_toRawSubstring_x27(v___x_398_);
return v___x_399_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__180(void){
_start:
{
lean_object* v___x_414_; lean_object* v___x_415_; 
v___x_414_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__179));
v___x_415_ = l_String_toRawSubstring_x27(v___x_414_);
return v___x_415_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__186(void){
_start:
{
lean_object* v___x_430_; lean_object* v___x_431_; 
v___x_430_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__185));
v___x_431_ = l_String_toRawSubstring_x27(v___x_430_);
return v___x_431_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__192(void){
_start:
{
lean_object* v___x_446_; lean_object* v___x_447_; 
v___x_446_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__191));
v___x_447_ = l_String_toRawSubstring_x27(v___x_446_);
return v___x_447_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__197(void){
_start:
{
lean_object* v___x_457_; lean_object* v___x_458_; 
v___x_457_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__196));
v___x_458_ = l_String_toRawSubstring_x27(v___x_457_);
return v___x_458_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__202(void){
_start:
{
lean_object* v___x_468_; lean_object* v___x_469_; 
v___x_468_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__201));
v___x_469_ = l_String_toRawSubstring_x27(v___x_468_);
return v___x_469_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__207(void){
_start:
{
lean_object* v___x_479_; lean_object* v___x_480_; 
v___x_479_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__206));
v___x_480_ = l_String_toRawSubstring_x27(v___x_479_);
return v___x_480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1(lean_object* v_x_492_, lean_object* v_a_493_, lean_object* v_a_494_){
_start:
{
lean_object* v___x_495_; lean_object* v___y_497_; lean_object* v___y_498_; lean_object* v___y_499_; lean_object* v___y_500_; lean_object* v___y_501_; lean_object* v___y_502_; lean_object* v___y_503_; lean_object* v___y_504_; lean_object* v___y_505_; lean_object* v___y_506_; lean_object* v___y_507_; lean_object* v___y_508_; lean_object* v___y_509_; lean_object* v___y_510_; lean_object* v___y_511_; lean_object* v___y_512_; lean_object* v___y_513_; lean_object* v___y_514_; lean_object* v___y_515_; lean_object* v___y_516_; lean_object* v___y_517_; lean_object* v___y_518_; lean_object* v___y_519_; lean_object* v___y_520_; lean_object* v___y_521_; lean_object* v___y_522_; lean_object* v___y_523_; lean_object* v___y_524_; lean_object* v___y_525_; lean_object* v___y_526_; lean_object* v___y_527_; lean_object* v_loc_595_; lean_object* v___y_596_; lean_object* v___y_597_; lean_object* v___x_877_; uint8_t v___x_878_; 
v___x_495_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group_group___closed__1));
v___x_877_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group_group___closed__4));
lean_inc(v_x_492_);
v___x_878_ = l_Lean_Syntax_isOfKind(v_x_492_, v___x_877_);
if (v___x_878_ == 0)
{
lean_object* v___x_879_; lean_object* v___x_880_; 
lean_dec(v_x_492_);
v___x_879_ = lean_box(1);
v___x_880_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_880_, 0, v___x_879_);
lean_ctor_set(v___x_880_, 1, v_a_494_);
return v___x_880_;
}
else
{
lean_object* v___x_881_; lean_object* v___x_882_; uint8_t v___x_883_; 
v___x_881_ = lean_unsigned_to_nat(1u);
v___x_882_ = l_Lean_Syntax_getArg(v_x_492_, v___x_881_);
lean_dec(v_x_492_);
v___x_883_ = l_Lean_Syntax_isNone(v___x_882_);
if (v___x_883_ == 0)
{
uint8_t v___x_884_; 
lean_inc(v___x_882_);
v___x_884_ = l_Lean_Syntax_matchesNull(v___x_882_, v___x_881_);
if (v___x_884_ == 0)
{
lean_object* v___x_885_; lean_object* v___x_886_; 
lean_dec(v___x_882_);
v___x_885_ = lean_box(1);
v___x_886_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_886_, 0, v___x_885_);
lean_ctor_set(v___x_886_, 1, v_a_494_);
return v___x_886_;
}
else
{
lean_object* v___x_887_; lean_object* v_loc_888_; lean_object* v___x_889_; 
v___x_887_ = lean_unsigned_to_nat(0u);
v_loc_888_ = l_Lean_Syntax_getArg(v___x_882_, v___x_887_);
lean_dec(v___x_882_);
v___x_889_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_889_, 0, v_loc_888_);
v_loc_595_ = v___x_889_;
v___y_596_ = v_a_493_;
v___y_597_ = v_a_494_;
goto v___jp_594_;
}
}
else
{
lean_object* v___x_890_; 
lean_dec(v___x_882_);
v___x_890_ = lean_box(0);
v_loc_595_ = v___x_890_;
v___y_596_ = v_a_493_;
v___y_597_ = v_a_494_;
goto v___jp_594_;
}
}
v___jp_496_:
{
lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; lean_object* v___x_549_; lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v___x_552_; lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_555_; lean_object* v___x_556_; lean_object* v___x_557_; lean_object* v___x_558_; lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_561_; lean_object* v___x_562_; lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_572_; lean_object* v___x_573_; lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; lean_object* v___x_592_; lean_object* v___x_593_; 
lean_inc_ref(v___y_523_);
v___x_528_ = l_Array_append___redArg(v___y_523_, v___y_527_);
lean_dec_ref(v___y_527_);
lean_inc_n(v___y_503_, 8);
lean_inc_n(v___y_510_, 42);
v___x_529_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_529_, 0, v___y_510_);
lean_ctor_set(v___x_529_, 1, v___y_503_);
lean_ctor_set(v___x_529_, 2, v___x_528_);
lean_inc_ref(v___x_529_);
lean_inc(v___y_517_);
lean_inc(v___y_509_);
v___x_530_ = l_Lean_Syntax_node6(v___y_510_, v___y_509_, v___y_514_, v___y_522_, v___y_517_, v___y_499_, v___y_497_, v___x_529_);
v___x_531_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__0));
v___x_532_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_532_, 0, v___y_510_);
lean_ctor_set(v___x_532_, 1, v___x_531_);
v___x_533_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__3));
v___x_534_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__4));
v___x_535_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_535_, 0, v___y_510_);
lean_ctor_set(v___x_535_, 1, v___x_534_);
v___x_536_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__5));
lean_inc_ref_n(v___y_526_, 3);
lean_inc_ref_n(v___y_508_, 3);
v___x_537_ = l_Lean_Name_mkStr4(v___y_508_, v___y_526_, v___x_495_, v___x_536_);
v___x_538_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__7, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__7);
v___x_539_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__8));
lean_inc(v___y_512_);
lean_inc(v___y_498_);
v___x_540_ = l_Lean_addMacroScope(v___y_498_, v___x_539_, v___y_512_);
lean_inc_n(v___y_519_, 2);
v___x_541_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_541_, 0, v___y_510_);
lean_ctor_set(v___x_541_, 1, v___x_538_);
lean_ctor_set(v___x_541_, 2, v___x_540_);
lean_ctor_set(v___x_541_, 3, v___y_519_);
v___x_542_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__9));
v___x_543_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_543_, 0, v___y_510_);
lean_ctor_set(v___x_543_, 1, v___x_542_);
v___x_544_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__10));
v___x_545_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__11));
v___x_546_ = l_Lean_Name_mkStr4(v___y_508_, v___y_526_, v___x_544_, v___x_545_);
v___x_547_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__12));
v___x_548_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_548_, 0, v___y_510_);
lean_ctor_set(v___x_548_, 1, v___x_547_);
v___x_549_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__14, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__14);
v___x_550_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__15));
v___x_551_ = l_Lean_addMacroScope(v___y_498_, v___x_550_, v___y_512_);
v___x_552_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_552_, 0, v___y_510_);
lean_ctor_set(v___x_552_, 1, v___x_549_);
lean_ctor_set(v___x_552_, 2, v___x_551_);
lean_ctor_set(v___x_552_, 3, v___y_519_);
v___x_553_ = l_Lean_Syntax_node2(v___y_510_, v___x_546_, v___x_548_, v___x_552_);
v___x_554_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__16));
v___x_555_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_555_, 0, v___y_510_);
lean_ctor_set(v___x_555_, 1, v___x_554_);
lean_inc_ref(v___x_555_);
lean_inc(v___y_511_);
v___x_556_ = l_Lean_Syntax_node5(v___y_510_, v___x_537_, v___y_511_, v___x_541_, v___x_543_, v___x_553_, v___x_555_);
v___x_557_ = l_Lean_Syntax_node1(v___y_510_, v___y_516_, v___x_556_);
v___x_558_ = l_Lean_Syntax_node1(v___y_510_, v___y_503_, v___x_557_);
v___x_559_ = l_Lean_Syntax_node1(v___y_510_, v___y_520_, v___x_558_);
v___x_560_ = l_Lean_Syntax_node4(v___y_510_, v___x_533_, v___x_535_, v___y_517_, v___x_559_, v___x_529_);
lean_inc(v___y_525_);
v___x_561_ = l_Lean_Syntax_node3(v___y_510_, v___y_525_, v___x_530_, v___x_532_, v___x_560_);
v___x_562_ = l_Lean_Syntax_node1(v___y_510_, v___y_503_, v___x_561_);
lean_inc_n(v___y_501_, 5);
v___x_563_ = l_Lean_Syntax_node1(v___y_510_, v___y_501_, v___x_562_);
lean_inc_n(v___y_507_, 5);
v___x_564_ = l_Lean_Syntax_node1(v___y_510_, v___y_507_, v___x_563_);
lean_inc(v___y_502_);
v___x_565_ = l_Lean_Syntax_node3(v___y_510_, v___y_502_, v___y_511_, v___x_564_, v___x_555_);
v___x_566_ = l_Lean_Syntax_node1(v___y_510_, v___y_503_, v___x_565_);
v___x_567_ = l_Lean_Syntax_node1(v___y_510_, v___y_501_, v___x_566_);
v___x_568_ = l_Lean_Syntax_node1(v___y_510_, v___y_507_, v___x_567_);
lean_inc(v___y_500_);
v___x_569_ = l_Lean_Syntax_node2(v___y_510_, v___y_500_, v___y_504_, v___x_568_);
v___x_570_ = l_Lean_Syntax_node1(v___y_510_, v___y_503_, v___x_569_);
v___x_571_ = l_Lean_Syntax_node1(v___y_510_, v___y_501_, v___x_570_);
v___x_572_ = l_Lean_Syntax_node1(v___y_510_, v___y_507_, v___x_571_);
lean_inc(v___y_505_);
v___x_573_ = l_Lean_Syntax_node2(v___y_510_, v___y_505_, v___y_513_, v___x_572_);
v___x_574_ = l_Lean_Syntax_node1(v___y_510_, v___y_503_, v___x_573_);
v___x_575_ = l_Lean_Syntax_node1(v___y_510_, v___y_501_, v___x_574_);
v___x_576_ = l_Lean_Syntax_node1(v___y_510_, v___y_507_, v___x_575_);
lean_inc(v___y_521_);
lean_inc_n(v___y_506_, 2);
v___x_577_ = l_Lean_Syntax_node2(v___y_510_, v___y_506_, v___y_521_, v___x_576_);
v___x_578_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__17));
v___x_579_ = l_Lean_Name_mkStr4(v___y_508_, v___y_526_, v___x_495_, v___x_578_);
v___x_580_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_580_, 0, v___y_510_);
lean_ctor_set(v___x_580_, 1, v___x_578_);
v___x_581_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__19));
v___x_582_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__20));
v___x_583_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_583_, 0, v___y_510_);
lean_ctor_set(v___x_583_, 1, v___x_582_);
v___x_584_ = l_Lean_Syntax_node1(v___y_510_, v___x_581_, v___x_583_);
v___x_585_ = l_Lean_Syntax_node1(v___y_510_, v___y_503_, v___x_584_);
v___x_586_ = l_Lean_Syntax_node2(v___y_510_, v___x_579_, v___x_580_, v___x_585_);
v___x_587_ = l_Lean_Syntax_node1(v___y_510_, v___y_503_, v___x_586_);
v___x_588_ = l_Lean_Syntax_node1(v___y_510_, v___y_501_, v___x_587_);
v___x_589_ = l_Lean_Syntax_node1(v___y_510_, v___y_507_, v___x_588_);
v___x_590_ = l_Lean_Syntax_node2(v___y_510_, v___y_506_, v___y_521_, v___x_589_);
v___x_591_ = l_Lean_Syntax_node2(v___y_510_, v___y_503_, v___x_577_, v___x_590_);
lean_inc(v___y_524_);
v___x_592_ = l_Lean_Syntax_node2(v___y_510_, v___y_524_, v___y_518_, v___x_591_);
v___x_593_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_593_, 0, v___x_592_);
lean_ctor_set(v___x_593_, 1, v___y_515_);
return v___x_593_;
}
v___jp_594_:
{
lean_object* v_quotContext_598_; lean_object* v_currMacroScope_599_; lean_object* v_ref_600_; uint8_t v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_605_; lean_object* v___x_606_; lean_object* v___x_607_; lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_631_; lean_object* v___x_632_; lean_object* v___x_633_; lean_object* v___x_634_; lean_object* v___x_635_; lean_object* v___x_636_; lean_object* v___x_637_; lean_object* v___x_638_; lean_object* v___x_639_; lean_object* v___x_640_; lean_object* v___x_641_; lean_object* v___x_642_; lean_object* v___x_643_; lean_object* v___x_644_; lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v___x_647_; lean_object* v___x_648_; lean_object* v___x_649_; lean_object* v___x_650_; lean_object* v___x_651_; lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; lean_object* v___x_655_; lean_object* v___x_656_; lean_object* v___x_657_; lean_object* v___x_658_; lean_object* v___x_659_; lean_object* v___x_660_; lean_object* v___x_661_; lean_object* v___x_662_; lean_object* v___x_663_; lean_object* v___x_664_; lean_object* v___x_665_; lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v___x_668_; lean_object* v___x_669_; lean_object* v___x_670_; lean_object* v___x_671_; lean_object* v___x_672_; lean_object* v___x_673_; lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v___x_678_; lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_681_; lean_object* v___x_682_; lean_object* v___x_683_; lean_object* v___x_684_; lean_object* v___x_685_; lean_object* v___x_686_; lean_object* v___x_687_; lean_object* v___x_688_; lean_object* v___x_689_; lean_object* v___x_690_; lean_object* v___x_691_; lean_object* v___x_692_; lean_object* v___x_693_; lean_object* v___x_694_; lean_object* v___x_695_; lean_object* v___x_696_; lean_object* v___x_697_; lean_object* v___x_698_; lean_object* v___x_699_; lean_object* v___x_700_; lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; lean_object* v___x_708_; lean_object* v___x_709_; lean_object* v___x_710_; lean_object* v___x_711_; lean_object* v___x_712_; lean_object* v___x_713_; lean_object* v___x_714_; lean_object* v___x_715_; lean_object* v___x_716_; lean_object* v___x_717_; lean_object* v___x_718_; lean_object* v___x_719_; lean_object* v___x_720_; lean_object* v___x_721_; lean_object* v___x_722_; lean_object* v___x_723_; lean_object* v___x_724_; lean_object* v___x_725_; lean_object* v___x_726_; lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; lean_object* v___x_730_; lean_object* v___x_731_; lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___x_734_; lean_object* v___x_735_; lean_object* v___x_736_; lean_object* v___x_737_; lean_object* v___x_738_; lean_object* v___x_739_; lean_object* v___x_740_; lean_object* v___x_741_; lean_object* v___x_742_; lean_object* v___x_743_; lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v___x_746_; lean_object* v___x_747_; lean_object* v___x_748_; lean_object* v___x_749_; lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v___x_752_; lean_object* v___x_753_; lean_object* v___x_754_; lean_object* v___x_755_; lean_object* v___x_756_; lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; lean_object* v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; lean_object* v___x_765_; lean_object* v___x_766_; lean_object* v___x_767_; lean_object* v___x_768_; lean_object* v___x_769_; lean_object* v___x_770_; lean_object* v___x_771_; lean_object* v___x_772_; lean_object* v___x_773_; lean_object* v___x_774_; lean_object* v___x_775_; lean_object* v___x_776_; lean_object* v___x_777_; lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v___x_780_; lean_object* v___x_781_; lean_object* v___x_782_; lean_object* v___x_783_; lean_object* v___x_784_; lean_object* v___x_785_; lean_object* v___x_786_; lean_object* v___x_787_; lean_object* v___x_788_; lean_object* v___x_789_; lean_object* v___x_790_; lean_object* v___x_791_; lean_object* v___x_792_; lean_object* v___x_793_; lean_object* v___x_794_; lean_object* v___x_795_; lean_object* v___x_796_; lean_object* v___x_797_; lean_object* v___x_798_; lean_object* v___x_799_; lean_object* v___x_800_; lean_object* v___x_801_; lean_object* v___x_802_; lean_object* v___x_803_; lean_object* v___x_804_; lean_object* v___x_805_; lean_object* v___x_806_; lean_object* v___x_807_; lean_object* v___x_808_; lean_object* v___x_809_; lean_object* v___x_810_; lean_object* v___x_811_; lean_object* v___x_812_; lean_object* v___x_813_; lean_object* v___x_814_; lean_object* v___x_815_; lean_object* v___x_816_; lean_object* v___x_817_; lean_object* v___x_818_; lean_object* v___x_819_; lean_object* v___x_820_; lean_object* v___x_821_; lean_object* v___x_822_; lean_object* v___x_823_; lean_object* v___x_824_; lean_object* v___x_825_; lean_object* v___x_826_; lean_object* v___x_827_; lean_object* v___x_828_; lean_object* v___x_829_; lean_object* v___x_830_; lean_object* v___x_831_; lean_object* v___x_832_; lean_object* v___x_833_; lean_object* v___x_834_; lean_object* v___x_835_; lean_object* v___x_836_; lean_object* v___x_837_; lean_object* v___x_838_; lean_object* v___x_839_; lean_object* v___x_840_; lean_object* v___x_841_; lean_object* v___x_842_; lean_object* v___x_843_; lean_object* v___x_844_; lean_object* v___x_845_; lean_object* v___x_846_; lean_object* v___x_847_; lean_object* v___x_848_; lean_object* v___x_849_; lean_object* v___x_850_; lean_object* v___x_851_; lean_object* v___x_852_; lean_object* v___x_853_; lean_object* v___x_854_; lean_object* v___x_855_; lean_object* v___x_856_; lean_object* v___x_857_; lean_object* v___x_858_; lean_object* v___x_859_; lean_object* v___x_860_; lean_object* v___x_861_; lean_object* v___x_862_; lean_object* v___x_863_; lean_object* v___x_864_; lean_object* v___x_865_; lean_object* v___x_866_; lean_object* v___x_867_; lean_object* v___x_868_; lean_object* v___x_869_; lean_object* v___x_870_; lean_object* v___x_871_; lean_object* v___x_872_; lean_object* v___x_873_; 
v_quotContext_598_ = lean_ctor_get(v___y_596_, 1);
v_currMacroScope_599_ = lean_ctor_get(v___y_596_, 2);
v_ref_600_ = lean_ctor_get(v___y_596_, 5);
v___x_601_ = 0;
v___x_602_ = l_Lean_SourceInfo_fromRef(v_ref_600_, v___x_601_);
v___x_603_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__21));
v___x_604_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__22));
v___x_605_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__23));
v___x_606_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__24));
lean_inc_n(v___x_602_, 77);
v___x_607_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_607_, 0, v___x_602_);
lean_ctor_set(v___x_607_, 1, v___x_605_);
v___x_608_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__26));
v___x_609_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__27));
v___x_610_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__28));
v___x_611_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_611_, 0, v___x_602_);
lean_ctor_set(v___x_611_, 1, v___x_610_);
v___x_612_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__30));
v___x_613_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__32));
v___x_614_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__34));
v___x_615_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__35));
v___x_616_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_616_, 0, v___x_602_);
lean_ctor_set(v___x_616_, 1, v___x_615_);
v___x_617_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__37));
v___x_618_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__38));
v___x_619_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_619_, 0, v___x_602_);
lean_ctor_set(v___x_619_, 1, v___x_618_);
v___x_620_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__40));
v___x_621_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__41));
v___x_622_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_622_, 0, v___x_602_);
lean_ctor_set(v___x_622_, 1, v___x_621_);
v___x_623_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__43));
v___x_624_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__44));
v___x_625_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__45));
v___x_626_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_626_, 0, v___x_602_);
lean_ctor_set(v___x_626_, 1, v___x_624_);
v___x_627_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__47));
v___x_628_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__49));
v___x_629_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__51));
v___x_630_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__52));
v___x_631_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_631_, 0, v___x_602_);
lean_ctor_set(v___x_631_, 1, v___x_630_);
v___x_632_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__54, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__54_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__54);
v___x_633_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__55));
lean_inc_n(v_currMacroScope_599_, 28);
lean_inc_n(v_quotContext_598_, 28);
v___x_634_ = l_Lean_addMacroScope(v_quotContext_598_, v___x_633_, v_currMacroScope_599_);
v___x_635_ = lean_box(0);
v___x_636_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__62));
v___x_637_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_637_, 0, v___x_602_);
lean_ctor_set(v___x_637_, 1, v___x_632_);
lean_ctor_set(v___x_637_, 2, v___x_634_);
lean_ctor_set(v___x_637_, 3, v___x_636_);
lean_inc_ref(v___x_631_);
v___x_638_ = l_Lean_Syntax_node2(v___x_602_, v___x_629_, v___x_631_, v___x_637_);
v___x_639_ = l_Lean_Syntax_node1(v___x_602_, v___x_628_, v___x_638_);
v___x_640_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__64, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__64_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__64);
v___x_641_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__65));
v___x_642_ = l_Lean_addMacroScope(v_quotContext_598_, v___x_641_, v_currMacroScope_599_);
v___x_643_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_643_, 0, v___x_602_);
lean_ctor_set(v___x_643_, 1, v___x_640_);
lean_ctor_set(v___x_643_, 2, v___x_642_);
lean_ctor_set(v___x_643_, 3, v___x_635_);
v___x_644_ = l_Lean_Syntax_node2(v___x_602_, v___x_629_, v___x_631_, v___x_643_);
v___x_645_ = l_Lean_Syntax_node1(v___x_602_, v___x_628_, v___x_644_);
v___x_646_ = l_Lean_Syntax_node2(v___x_602_, v___x_608_, v___x_639_, v___x_645_);
v___x_647_ = l_Lean_Syntax_node1(v___x_602_, v___x_627_, v___x_646_);
v___x_648_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__66, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__66_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__66);
v___x_649_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_649_, 0, v___x_602_);
lean_ctor_set(v___x_649_, 1, v___x_608_);
lean_ctor_set(v___x_649_, 2, v___x_648_);
v___x_650_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__67));
v___x_651_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_651_, 0, v___x_602_);
lean_ctor_set(v___x_651_, 1, v___x_650_);
v___x_652_ = l_Lean_Syntax_node1(v___x_602_, v___x_608_, v___x_651_);
v___x_653_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__68));
v___x_654_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_654_, 0, v___x_602_);
lean_ctor_set(v___x_654_, 1, v___x_653_);
v___x_655_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__70));
v___x_656_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__72, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__72_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__72);
v___x_657_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__73));
v___x_658_ = l_Lean_addMacroScope(v_quotContext_598_, v___x_657_, v_currMacroScope_599_);
v___x_659_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__75));
v___x_660_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_660_, 0, v___x_602_);
lean_ctor_set(v___x_660_, 1, v___x_656_);
lean_ctor_set(v___x_660_, 2, v___x_658_);
lean_ctor_set(v___x_660_, 3, v___x_659_);
lean_inc_ref_n(v___x_649_, 45);
v___x_661_ = l_Lean_Syntax_node3(v___x_602_, v___x_655_, v___x_649_, v___x_649_, v___x_660_);
v___x_662_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__76));
v___x_663_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_663_, 0, v___x_602_);
lean_ctor_set(v___x_663_, 1, v___x_662_);
v___x_664_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__78, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__78_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__78);
v___x_665_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__79));
v___x_666_ = l_Lean_addMacroScope(v_quotContext_598_, v___x_665_, v_currMacroScope_599_);
v___x_667_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__81));
v___x_668_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_668_, 0, v___x_602_);
lean_ctor_set(v___x_668_, 1, v___x_664_);
lean_ctor_set(v___x_668_, 2, v___x_666_);
lean_ctor_set(v___x_668_, 3, v___x_667_);
v___x_669_ = l_Lean_Syntax_node3(v___x_602_, v___x_655_, v___x_649_, v___x_649_, v___x_668_);
v___x_670_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__83, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__83_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__83);
v___x_671_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__84));
v___x_672_ = l_Lean_addMacroScope(v_quotContext_598_, v___x_671_, v_currMacroScope_599_);
v___x_673_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__86));
v___x_674_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_674_, 0, v___x_602_);
lean_ctor_set(v___x_674_, 1, v___x_670_);
lean_ctor_set(v___x_674_, 2, v___x_672_);
lean_ctor_set(v___x_674_, 3, v___x_673_);
v___x_675_ = l_Lean_Syntax_node3(v___x_602_, v___x_655_, v___x_649_, v___x_649_, v___x_674_);
v___x_676_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__87));
v___x_677_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_677_, 0, v___x_602_);
lean_ctor_set(v___x_677_, 1, v___x_676_);
v___x_678_ = l_Lean_Syntax_node1(v___x_602_, v___x_608_, v___x_677_);
v___x_679_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__89, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__89_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__89);
v___x_680_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__90));
v___x_681_ = l_Lean_addMacroScope(v_quotContext_598_, v___x_680_, v_currMacroScope_599_);
v___x_682_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__92));
v___x_683_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_683_, 0, v___x_602_);
lean_ctor_set(v___x_683_, 1, v___x_679_);
lean_ctor_set(v___x_683_, 2, v___x_681_);
lean_ctor_set(v___x_683_, 3, v___x_682_);
lean_inc_n(v___x_678_, 6);
v___x_684_ = l_Lean_Syntax_node3(v___x_602_, v___x_655_, v___x_649_, v___x_678_, v___x_683_);
v___x_685_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__94, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__94_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__94);
v___x_686_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__95));
v___x_687_ = l_Lean_addMacroScope(v_quotContext_598_, v___x_686_, v_currMacroScope_599_);
v___x_688_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__97));
v___x_689_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_689_, 0, v___x_602_);
lean_ctor_set(v___x_689_, 1, v___x_685_);
lean_ctor_set(v___x_689_, 2, v___x_687_);
lean_ctor_set(v___x_689_, 3, v___x_688_);
v___x_690_ = l_Lean_Syntax_node3(v___x_602_, v___x_655_, v___x_649_, v___x_678_, v___x_689_);
v___x_691_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__99, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__99_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__99);
v___x_692_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__100));
v___x_693_ = l_Lean_addMacroScope(v_quotContext_598_, v___x_692_, v_currMacroScope_599_);
v___x_694_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__102));
v___x_695_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_695_, 0, v___x_602_);
lean_ctor_set(v___x_695_, 1, v___x_691_);
lean_ctor_set(v___x_695_, 2, v___x_693_);
lean_ctor_set(v___x_695_, 3, v___x_694_);
v___x_696_ = l_Lean_Syntax_node3(v___x_602_, v___x_655_, v___x_649_, v___x_678_, v___x_695_);
v___x_697_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__104, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__104_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__104);
v___x_698_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__107));
v___x_699_ = l_Lean_addMacroScope(v_quotContext_598_, v___x_698_, v_currMacroScope_599_);
v___x_700_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__109));
v___x_701_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_701_, 0, v___x_602_);
lean_ctor_set(v___x_701_, 1, v___x_697_);
lean_ctor_set(v___x_701_, 2, v___x_699_);
lean_ctor_set(v___x_701_, 3, v___x_700_);
v___x_702_ = l_Lean_Syntax_node3(v___x_602_, v___x_655_, v___x_649_, v___x_649_, v___x_701_);
v___x_703_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__111, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__111_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__111);
v___x_704_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__113));
v___x_705_ = l_Lean_addMacroScope(v_quotContext_598_, v___x_704_, v_currMacroScope_599_);
v___x_706_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__115));
v___x_707_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_707_, 0, v___x_602_);
lean_ctor_set(v___x_707_, 1, v___x_703_);
lean_ctor_set(v___x_707_, 2, v___x_705_);
lean_ctor_set(v___x_707_, 3, v___x_706_);
v___x_708_ = l_Lean_Syntax_node3(v___x_602_, v___x_655_, v___x_649_, v___x_649_, v___x_707_);
v___x_709_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__117, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__117_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__117);
v___x_710_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__119));
v___x_711_ = l_Lean_addMacroScope(v_quotContext_598_, v___x_710_, v_currMacroScope_599_);
v___x_712_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__121));
v___x_713_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_713_, 0, v___x_602_);
lean_ctor_set(v___x_713_, 1, v___x_709_);
lean_ctor_set(v___x_713_, 2, v___x_711_);
lean_ctor_set(v___x_713_, 3, v___x_712_);
v___x_714_ = l_Lean_Syntax_node3(v___x_602_, v___x_655_, v___x_649_, v___x_649_, v___x_713_);
v___x_715_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__123, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__123_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__123);
v___x_716_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__125));
v___x_717_ = l_Lean_addMacroScope(v_quotContext_598_, v___x_716_, v_currMacroScope_599_);
v___x_718_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__127));
v___x_719_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_719_, 0, v___x_602_);
lean_ctor_set(v___x_719_, 1, v___x_715_);
lean_ctor_set(v___x_719_, 2, v___x_717_);
lean_ctor_set(v___x_719_, 3, v___x_718_);
v___x_720_ = l_Lean_Syntax_node3(v___x_602_, v___x_655_, v___x_649_, v___x_649_, v___x_719_);
v___x_721_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__129, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__129_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__129);
v___x_722_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__130));
v___x_723_ = l_Lean_addMacroScope(v_quotContext_598_, v___x_722_, v_currMacroScope_599_);
v___x_724_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__132));
v___x_725_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_725_, 0, v___x_602_);
lean_ctor_set(v___x_725_, 1, v___x_721_);
lean_ctor_set(v___x_725_, 2, v___x_723_);
lean_ctor_set(v___x_725_, 3, v___x_724_);
v___x_726_ = l_Lean_Syntax_node3(v___x_602_, v___x_655_, v___x_649_, v___x_649_, v___x_725_);
v___x_727_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__134, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__134_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__134);
v___x_728_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__135));
v___x_729_ = l_Lean_addMacroScope(v_quotContext_598_, v___x_728_, v_currMacroScope_599_);
v___x_730_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__137));
v___x_731_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_731_, 0, v___x_602_);
lean_ctor_set(v___x_731_, 1, v___x_727_);
lean_ctor_set(v___x_731_, 2, v___x_729_);
lean_ctor_set(v___x_731_, 3, v___x_730_);
v___x_732_ = l_Lean_Syntax_node3(v___x_602_, v___x_655_, v___x_649_, v___x_649_, v___x_731_);
v___x_733_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__139, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__139_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__139);
v___x_734_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__140));
v___x_735_ = l_Lean_addMacroScope(v_quotContext_598_, v___x_734_, v_currMacroScope_599_);
v___x_736_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__142));
v___x_737_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_737_, 0, v___x_602_);
lean_ctor_set(v___x_737_, 1, v___x_733_);
lean_ctor_set(v___x_737_, 2, v___x_735_);
lean_ctor_set(v___x_737_, 3, v___x_736_);
v___x_738_ = l_Lean_Syntax_node3(v___x_602_, v___x_655_, v___x_649_, v___x_649_, v___x_737_);
v___x_739_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__144, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__144_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__144);
v___x_740_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__145));
v___x_741_ = l_Lean_addMacroScope(v_quotContext_598_, v___x_740_, v_currMacroScope_599_);
v___x_742_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__147));
v___x_743_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_743_, 0, v___x_602_);
lean_ctor_set(v___x_743_, 1, v___x_739_);
lean_ctor_set(v___x_743_, 2, v___x_741_);
lean_ctor_set(v___x_743_, 3, v___x_742_);
v___x_744_ = l_Lean_Syntax_node3(v___x_602_, v___x_655_, v___x_649_, v___x_649_, v___x_743_);
v___x_745_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__149, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__149_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__149);
v___x_746_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__150));
v___x_747_ = l_Lean_addMacroScope(v_quotContext_598_, v___x_746_, v_currMacroScope_599_);
v___x_748_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__152));
v___x_749_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_749_, 0, v___x_602_);
lean_ctor_set(v___x_749_, 1, v___x_745_);
lean_ctor_set(v___x_749_, 2, v___x_747_);
lean_ctor_set(v___x_749_, 3, v___x_748_);
v___x_750_ = l_Lean_Syntax_node3(v___x_602_, v___x_655_, v___x_649_, v___x_649_, v___x_749_);
v___x_751_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__154, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__154_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__154);
v___x_752_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__155));
v___x_753_ = l_Lean_addMacroScope(v_quotContext_598_, v___x_752_, v_currMacroScope_599_);
v___x_754_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__157));
v___x_755_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_755_, 0, v___x_602_);
lean_ctor_set(v___x_755_, 1, v___x_751_);
lean_ctor_set(v___x_755_, 2, v___x_753_);
lean_ctor_set(v___x_755_, 3, v___x_754_);
v___x_756_ = l_Lean_Syntax_node3(v___x_602_, v___x_655_, v___x_649_, v___x_678_, v___x_755_);
v___x_757_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__159, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__159_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__159);
v___x_758_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__160));
v___x_759_ = l_Lean_addMacroScope(v_quotContext_598_, v___x_758_, v_currMacroScope_599_);
v___x_760_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__162));
v___x_761_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_761_, 0, v___x_602_);
lean_ctor_set(v___x_761_, 1, v___x_757_);
lean_ctor_set(v___x_761_, 2, v___x_759_);
lean_ctor_set(v___x_761_, 3, v___x_760_);
v___x_762_ = l_Lean_Syntax_node3(v___x_602_, v___x_655_, v___x_649_, v___x_678_, v___x_761_);
v___x_763_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__164, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__164_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__164);
v___x_764_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__165));
v___x_765_ = l_Lean_addMacroScope(v_quotContext_598_, v___x_764_, v_currMacroScope_599_);
v___x_766_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__167));
v___x_767_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_767_, 0, v___x_602_);
lean_ctor_set(v___x_767_, 1, v___x_763_);
lean_ctor_set(v___x_767_, 2, v___x_765_);
lean_ctor_set(v___x_767_, 3, v___x_766_);
v___x_768_ = l_Lean_Syntax_node3(v___x_602_, v___x_655_, v___x_649_, v___x_678_, v___x_767_);
v___x_769_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__169, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__169_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__169);
v___x_770_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__170));
v___x_771_ = l_Lean_addMacroScope(v_quotContext_598_, v___x_770_, v_currMacroScope_599_);
v___x_772_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__172));
v___x_773_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_773_, 0, v___x_602_);
lean_ctor_set(v___x_773_, 1, v___x_769_);
lean_ctor_set(v___x_773_, 2, v___x_771_);
lean_ctor_set(v___x_773_, 3, v___x_772_);
v___x_774_ = l_Lean_Syntax_node3(v___x_602_, v___x_655_, v___x_649_, v___x_678_, v___x_773_);
v___x_775_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__174, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__174_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__174);
v___x_776_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__175));
v___x_777_ = l_Lean_addMacroScope(v_quotContext_598_, v___x_776_, v_currMacroScope_599_);
v___x_778_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__178));
v___x_779_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_779_, 0, v___x_602_);
lean_ctor_set(v___x_779_, 1, v___x_775_);
lean_ctor_set(v___x_779_, 2, v___x_777_);
lean_ctor_set(v___x_779_, 3, v___x_778_);
v___x_780_ = l_Lean_Syntax_node3(v___x_602_, v___x_655_, v___x_649_, v___x_649_, v___x_779_);
v___x_781_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__180, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__180_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__180);
v___x_782_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__181));
v___x_783_ = l_Lean_addMacroScope(v_quotContext_598_, v___x_782_, v_currMacroScope_599_);
v___x_784_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__184));
v___x_785_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_785_, 0, v___x_602_);
lean_ctor_set(v___x_785_, 1, v___x_781_);
lean_ctor_set(v___x_785_, 2, v___x_783_);
lean_ctor_set(v___x_785_, 3, v___x_784_);
v___x_786_ = l_Lean_Syntax_node3(v___x_602_, v___x_655_, v___x_649_, v___x_649_, v___x_785_);
v___x_787_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__186, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__186_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__186);
v___x_788_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__187));
v___x_789_ = l_Lean_addMacroScope(v_quotContext_598_, v___x_788_, v_currMacroScope_599_);
v___x_790_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__190));
v___x_791_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_791_, 0, v___x_602_);
lean_ctor_set(v___x_791_, 1, v___x_787_);
lean_ctor_set(v___x_791_, 2, v___x_789_);
lean_ctor_set(v___x_791_, 3, v___x_790_);
v___x_792_ = l_Lean_Syntax_node3(v___x_602_, v___x_655_, v___x_649_, v___x_649_, v___x_791_);
v___x_793_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__192, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__192_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__192);
v___x_794_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__193));
v___x_795_ = l_Lean_addMacroScope(v_quotContext_598_, v___x_794_, v_currMacroScope_599_);
v___x_796_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__195));
v___x_797_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_797_, 0, v___x_602_);
lean_ctor_set(v___x_797_, 1, v___x_793_);
lean_ctor_set(v___x_797_, 2, v___x_795_);
lean_ctor_set(v___x_797_, 3, v___x_796_);
v___x_798_ = l_Lean_Syntax_node3(v___x_602_, v___x_655_, v___x_649_, v___x_649_, v___x_797_);
v___x_799_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__197, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__197_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__197);
v___x_800_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__198));
v___x_801_ = l_Lean_addMacroScope(v_quotContext_598_, v___x_800_, v_currMacroScope_599_);
v___x_802_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__200));
v___x_803_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_803_, 0, v___x_602_);
lean_ctor_set(v___x_803_, 1, v___x_799_);
lean_ctor_set(v___x_803_, 2, v___x_801_);
lean_ctor_set(v___x_803_, 3, v___x_802_);
v___x_804_ = l_Lean_Syntax_node3(v___x_602_, v___x_655_, v___x_649_, v___x_649_, v___x_803_);
v___x_805_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__202, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__202_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__202);
v___x_806_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__203));
v___x_807_ = l_Lean_addMacroScope(v_quotContext_598_, v___x_806_, v_currMacroScope_599_);
v___x_808_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__205));
v___x_809_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_809_, 0, v___x_602_);
lean_ctor_set(v___x_809_, 1, v___x_805_);
lean_ctor_set(v___x_809_, 2, v___x_807_);
lean_ctor_set(v___x_809_, 3, v___x_808_);
v___x_810_ = l_Lean_Syntax_node3(v___x_602_, v___x_655_, v___x_649_, v___x_649_, v___x_809_);
v___x_811_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__207, &lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__207_once, _init_lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__207);
v___x_812_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__208));
v___x_813_ = l_Lean_addMacroScope(v_quotContext_598_, v___x_812_, v_currMacroScope_599_);
v___x_814_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__210));
v___x_815_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_815_, 0, v___x_602_);
lean_ctor_set(v___x_815_, 1, v___x_811_);
lean_ctor_set(v___x_815_, 2, v___x_813_);
lean_ctor_set(v___x_815_, 3, v___x_814_);
v___x_816_ = l_Lean_Syntax_node3(v___x_602_, v___x_655_, v___x_649_, v___x_649_, v___x_815_);
v___x_817_ = lean_unsigned_to_nat(51u);
v___x_818_ = lean_mk_empty_array_with_capacity(v___x_817_);
v___x_819_ = lean_array_push(v___x_818_, v___x_661_);
lean_inc_ref_n(v___x_663_, 24);
v___x_820_ = lean_array_push(v___x_819_, v___x_663_);
v___x_821_ = lean_array_push(v___x_820_, v___x_669_);
v___x_822_ = lean_array_push(v___x_821_, v___x_663_);
v___x_823_ = lean_array_push(v___x_822_, v___x_675_);
v___x_824_ = lean_array_push(v___x_823_, v___x_663_);
v___x_825_ = lean_array_push(v___x_824_, v___x_684_);
v___x_826_ = lean_array_push(v___x_825_, v___x_663_);
v___x_827_ = lean_array_push(v___x_826_, v___x_690_);
v___x_828_ = lean_array_push(v___x_827_, v___x_663_);
v___x_829_ = lean_array_push(v___x_828_, v___x_696_);
v___x_830_ = lean_array_push(v___x_829_, v___x_663_);
v___x_831_ = lean_array_push(v___x_830_, v___x_702_);
v___x_832_ = lean_array_push(v___x_831_, v___x_663_);
v___x_833_ = lean_array_push(v___x_832_, v___x_708_);
v___x_834_ = lean_array_push(v___x_833_, v___x_663_);
v___x_835_ = lean_array_push(v___x_834_, v___x_714_);
v___x_836_ = lean_array_push(v___x_835_, v___x_663_);
v___x_837_ = lean_array_push(v___x_836_, v___x_720_);
v___x_838_ = lean_array_push(v___x_837_, v___x_663_);
v___x_839_ = lean_array_push(v___x_838_, v___x_726_);
v___x_840_ = lean_array_push(v___x_839_, v___x_663_);
v___x_841_ = lean_array_push(v___x_840_, v___x_732_);
v___x_842_ = lean_array_push(v___x_841_, v___x_663_);
v___x_843_ = lean_array_push(v___x_842_, v___x_738_);
v___x_844_ = lean_array_push(v___x_843_, v___x_663_);
v___x_845_ = lean_array_push(v___x_844_, v___x_744_);
v___x_846_ = lean_array_push(v___x_845_, v___x_663_);
v___x_847_ = lean_array_push(v___x_846_, v___x_750_);
v___x_848_ = lean_array_push(v___x_847_, v___x_663_);
v___x_849_ = lean_array_push(v___x_848_, v___x_756_);
v___x_850_ = lean_array_push(v___x_849_, v___x_663_);
v___x_851_ = lean_array_push(v___x_850_, v___x_762_);
v___x_852_ = lean_array_push(v___x_851_, v___x_663_);
v___x_853_ = lean_array_push(v___x_852_, v___x_768_);
v___x_854_ = lean_array_push(v___x_853_, v___x_663_);
v___x_855_ = lean_array_push(v___x_854_, v___x_774_);
v___x_856_ = lean_array_push(v___x_855_, v___x_663_);
v___x_857_ = lean_array_push(v___x_856_, v___x_780_);
v___x_858_ = lean_array_push(v___x_857_, v___x_663_);
v___x_859_ = lean_array_push(v___x_858_, v___x_786_);
v___x_860_ = lean_array_push(v___x_859_, v___x_663_);
v___x_861_ = lean_array_push(v___x_860_, v___x_792_);
v___x_862_ = lean_array_push(v___x_861_, v___x_663_);
v___x_863_ = lean_array_push(v___x_862_, v___x_798_);
v___x_864_ = lean_array_push(v___x_863_, v___x_663_);
v___x_865_ = lean_array_push(v___x_864_, v___x_804_);
v___x_866_ = lean_array_push(v___x_865_, v___x_663_);
v___x_867_ = lean_array_push(v___x_866_, v___x_810_);
v___x_868_ = lean_array_push(v___x_867_, v___x_663_);
v___x_869_ = lean_array_push(v___x_868_, v___x_816_);
v___x_870_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_870_, 0, v___x_602_);
lean_ctor_set(v___x_870_, 1, v___x_608_);
lean_ctor_set(v___x_870_, 2, v___x_869_);
v___x_871_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__211));
v___x_872_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_872_, 0, v___x_602_);
lean_ctor_set(v___x_872_, 1, v___x_871_);
v___x_873_ = l_Lean_Syntax_node3(v___x_602_, v___x_608_, v___x_654_, v___x_870_, v___x_872_);
if (lean_obj_tag(v_loc_595_) == 1)
{
lean_object* v_val_874_; lean_object* v___x_875_; 
v_val_874_ = lean_ctor_get(v_loc_595_, 0);
lean_inc(v_val_874_);
lean_dec_ref_known(v_loc_595_, 1);
v___x_875_ = l_Array_mkArray1___redArg(v_val_874_);
lean_inc(v_currMacroScope_599_);
lean_inc(v_quotContext_598_);
v___y_497_ = v___x_873_;
v___y_498_ = v_quotContext_598_;
v___y_499_ = v___x_652_;
v___y_500_ = v___x_617_;
v___y_501_ = v___x_613_;
v___y_502_ = v___x_620_;
v___y_503_ = v___x_608_;
v___y_504_ = v___x_619_;
v___y_505_ = v___x_614_;
v___y_506_ = v___x_609_;
v___y_507_ = v___x_612_;
v___y_508_ = v___x_603_;
v___y_509_ = v___x_625_;
v___y_510_ = v___x_602_;
v___y_511_ = v___x_622_;
v___y_512_ = v_currMacroScope_599_;
v___y_513_ = v___x_616_;
v___y_514_ = v___x_626_;
v___y_515_ = v___y_597_;
v___y_516_ = v___x_628_;
v___y_517_ = v___x_649_;
v___y_518_ = v___x_607_;
v___y_519_ = v___x_635_;
v___y_520_ = v___x_627_;
v___y_521_ = v___x_611_;
v___y_522_ = v___x_647_;
v___y_523_ = v___x_648_;
v___y_524_ = v___x_606_;
v___y_525_ = v___x_623_;
v___y_526_ = v___x_604_;
v___y_527_ = v___x_875_;
goto v___jp_496_;
}
else
{
lean_object* v___x_876_; 
lean_dec(v_loc_595_);
v___x_876_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___closed__212));
lean_inc(v_currMacroScope_599_);
lean_inc(v_quotContext_598_);
v___y_497_ = v___x_873_;
v___y_498_ = v_quotContext_598_;
v___y_499_ = v___x_652_;
v___y_500_ = v___x_617_;
v___y_501_ = v___x_613_;
v___y_502_ = v___x_620_;
v___y_503_ = v___x_608_;
v___y_504_ = v___x_619_;
v___y_505_ = v___x_614_;
v___y_506_ = v___x_609_;
v___y_507_ = v___x_612_;
v___y_508_ = v___x_603_;
v___y_509_ = v___x_625_;
v___y_510_ = v___x_602_;
v___y_511_ = v___x_622_;
v___y_512_ = v_currMacroScope_599_;
v___y_513_ = v___x_616_;
v___y_514_ = v___x_626_;
v___y_515_ = v___y_597_;
v___y_516_ = v___x_628_;
v___y_517_ = v___x_649_;
v___y_518_ = v___x_607_;
v___y_519_ = v___x_635_;
v___y_520_ = v___x_627_;
v___y_521_ = v___x_611_;
v___y_522_ = v___x_647_;
v___y_523_ = v___x_648_;
v___y_524_ = v___x_606_;
v___y_525_ = v___x_623_;
v___y_526_ = v___x_604_;
v___y_527_ = v___x_876_;
goto v___jp_496_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1___boxed(lean_object* v_x_891_, lean_object* v_a_892_, lean_object* v_a_893_){
_start:
{
lean_object* v_res_894_; 
v_res_894_ = lp_mathlib_Mathlib_Tactic_Group___aux__Mathlib__Tactic__Group______macroRules__Mathlib__Tactic__Group__group__1(v_x_891_, v_a_892_, v_a_893_);
lean_dec_ref(v_a_892_);
return v_res_894_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Commutator(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Sub_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FailIfNoProgress(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Ring(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Group(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Commutator(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Sub_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FailIfNoProgress(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Group(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_Group_group = _init_lp_mathlib_Mathlib_Tactic_Group_group();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Group_group);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Commutator(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Sub_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FailIfNoProgress(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Ring(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Group(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Commutator(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Sub_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FailIfNoProgress(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Group(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Group(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Group(builtin);
}
#ifdef __cplusplus
}
#endif
