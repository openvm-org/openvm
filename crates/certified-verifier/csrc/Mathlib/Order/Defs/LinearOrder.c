// Lean compiler output
// Module: Mathlib.Order.Defs.LinearOrder
// Imports: public import Init public meta import Init public import Batteries.Classes.Order public import Batteries.Tactic.Trans public import Mathlib.Data.Ordering.Basic public import Mathlib.Tactic.Push.Attr public import Mathlib.Tactic.Simps public import Mathlib.Tactic.SplitIfs public import Mathlib.Order.Defs.PartialOrder public import Batteries.Tactic.Init
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
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_maxDefault___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_maxDefault(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_minDefault___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_minDefault(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_tacticCompareOfLessAndEq__rfl___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "tacticCompareOfLessAndEq_rfl"};
static const lean_object* lp_mathlib_tacticCompareOfLessAndEq__rfl___closed__0 = (const lean_object*)&lp_mathlib_tacticCompareOfLessAndEq__rfl___closed__0_value;
static const lean_ctor_object lp_mathlib_tacticCompareOfLessAndEq__rfl___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_tacticCompareOfLessAndEq__rfl___closed__0_value),LEAN_SCALAR_PTR_LITERAL(181, 115, 139, 42, 76, 141, 255, 128)}};
static const lean_object* lp_mathlib_tacticCompareOfLessAndEq__rfl___closed__1 = (const lean_object*)&lp_mathlib_tacticCompareOfLessAndEq__rfl___closed__1_value;
static const lean_string_object lp_mathlib_tacticCompareOfLessAndEq__rfl___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "compareOfLessAndEq_rfl"};
static const lean_object* lp_mathlib_tacticCompareOfLessAndEq__rfl___closed__2 = (const lean_object*)&lp_mathlib_tacticCompareOfLessAndEq__rfl___closed__2_value;
static const lean_ctor_object lp_mathlib_tacticCompareOfLessAndEq__rfl___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_tacticCompareOfLessAndEq__rfl___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_tacticCompareOfLessAndEq__rfl___closed__3 = (const lean_object*)&lp_mathlib_tacticCompareOfLessAndEq__rfl___closed__3_value;
static const lean_ctor_object lp_mathlib_tacticCompareOfLessAndEq__rfl___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_tacticCompareOfLessAndEq__rfl___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_tacticCompareOfLessAndEq__rfl___closed__3_value)}};
static const lean_object* lp_mathlib_tacticCompareOfLessAndEq__rfl___closed__4 = (const lean_object*)&lp_mathlib_tacticCompareOfLessAndEq__rfl___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_tacticCompareOfLessAndEq__rfl = (const lean_object*)&lp_mathlib_tacticCompareOfLessAndEq__rfl___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(117, 253, 122, 28, 77, 248, 149, 120)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__5_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__6_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__7_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__7_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__7_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__7_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__8_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__9_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__9_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__9_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__9_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__10_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__11_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "intro"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__12_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__13_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__13_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__13_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__13_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(41, 145, 9, 18, 75, 146, 159, 78)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__13_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "a"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__14 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__14_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__15;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(247, 80, 99, 121, 74, 33, 203, 108)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__16 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__16_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "b"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__17 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__17_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__18;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__17_value),LEAN_SCALAR_PTR_LITERAL(47, 22, 244, 233, 226, 169, 241, 142)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__19 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__19_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ";"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__20 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__20_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "first"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__21 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__21_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__22_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__22_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__22_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__22_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__22_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(59, 232, 35, 17, 172, 62, 48, 174)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__22 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__22_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__23 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__23_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__23_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__24 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__24_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__25 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__25_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticRfl"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__26 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__26_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__27_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__27_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__27_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__27_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__27_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__27_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__26_value),LEAN_SCALAR_PTR_LITERAL(201, 188, 173, 198, 169, 252, 183, 45)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__27 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__27_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rfl"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__28 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__28_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__29 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__29_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__30_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__30_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__30_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__30_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__30_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__30_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__29_value),LEAN_SCALAR_PTR_LITERAL(50, 13, 241, 145, 67, 153, 105, 177)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__30 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__30_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__31 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__31_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__32_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__32_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__32_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__32_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__32_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__32_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__31_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__32 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__32_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__33;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "only"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__34 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__34_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__35 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__35_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "simpLemma"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__36 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__36_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__37_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__37_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__37_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__37_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__37_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__37_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__36_value),LEAN_SCALAR_PTR_LITERAL(38, 215, 101, 250, 181, 108, 118, 102)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__37 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__37_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "compare"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__38 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__38_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__39_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__39;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__38_value),LEAN_SCALAR_PTR_LITERAL(109, 41, 149, 169, 79, 76, 232, 231)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__40 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__40_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Ord"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__41 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__41_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__42_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__41_value),LEAN_SCALAR_PTR_LITERAL(47, 34, 14, 190, 177, 218, 16, 31)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__42_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__38_value),LEAN_SCALAR_PTR_LITERAL(241, 180, 168, 39, 68, 69, 153, 110)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__42 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__42_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__42_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__43 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__43_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__43_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__44 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__44_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__45 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__45_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "compareOfLessAndEq"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__46 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__46_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__47_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__47;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__46_value),LEAN_SCALAR_PTR_LITERAL(92, 211, 162, 133, 160, 250, 146, 99)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__48 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__48_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__48_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__49 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__49_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__49_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__50 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__50_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__51 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__51_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "tactic_<;>_"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__52 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__52_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__53_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__53_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__53_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__53_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__53_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__53_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__52_value),LEAN_SCALAR_PTR_LITERAL(31, 118, 44, 159, 195, 11, 47, 176)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__53 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__53_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__54 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__54_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "splitIfs"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__55 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__55_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__56_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__54_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__56_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__56_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__56_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__55_value),LEAN_SCALAR_PTR_LITERAL(109, 48, 181, 174, 145, 245, 228, 97)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__56 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__56_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "split_ifs"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__57 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__57_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "<;>"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__58 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__58_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__59 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__59_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "induction"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__60 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__60_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__61_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__61_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__61_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__61_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__61_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__61_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__61_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__60_value),LEAN_SCALAR_PTR_LITERAL(231, 196, 247, 144, 178, 6, 178, 16)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__61 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__61_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__62_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "elimTarget"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__62 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__62_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__63_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__63_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__63_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__63_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__63_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__63_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__63_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__62_value),LEAN_SCALAR_PTR_LITERAL(136, 63, 46, 91, 99, 29, 205, 171)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__63 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__63_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__64_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "configItem"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__64 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__64_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__65_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__65_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__65_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__65_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__65_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__65_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__65_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__64_value),LEAN_SCALAR_PTR_LITERAL(205, 9, 236, 192, 59, 252, 178, 140)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__65 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__65_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__66_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "posConfigItem"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__66 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__66_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__67_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__67_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__67_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__67_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__67_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__67_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__67_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__66_value),LEAN_SCALAR_PTR_LITERAL(232, 137, 50, 117, 152, 182, 155, 132)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__67 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__67_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__68_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "+"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__68 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__68_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__69_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "decide"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__69 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__69_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__70_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__70;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__71_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__69_value),LEAN_SCALAR_PTR_LITERAL(236, 252, 83, 10, 217, 228, 80, 149)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__71 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__71_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__72_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Decidable"};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__72 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__72_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__73_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__72_value),LEAN_SCALAR_PTR_LITERAL(87, 187, 205, 215, 218, 218, 68, 60)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__73_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__73_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__69_value),LEAN_SCALAR_PTR_LITERAL(16, 96, 65, 173, 152, 155, 4, 222)}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__73 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__73_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__74_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__73_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__74 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__74_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__75_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__74_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__75 = (const lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__75_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_LinearOrder_min__def___autoParam___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_LinearOrder_min__def___autoParam___closed__0 = (const lean_object*)&lp_mathlib_LinearOrder_min__def___autoParam___closed__0_value;
static const lean_string_object lp_mathlib_LinearOrder_min__def___autoParam___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "intros"};
static const lean_object* lp_mathlib_LinearOrder_min__def___autoParam___closed__1 = (const lean_object*)&lp_mathlib_LinearOrder_min__def___autoParam___closed__1_value;
static const lean_ctor_object lp_mathlib_LinearOrder_min__def___autoParam___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_LinearOrder_min__def___autoParam___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearOrder_min__def___autoParam___closed__2_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_LinearOrder_min__def___autoParam___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearOrder_min__def___autoParam___closed__2_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_LinearOrder_min__def___autoParam___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_LinearOrder_min__def___autoParam___closed__2_value_aux_2),((lean_object*)&lp_mathlib_LinearOrder_min__def___autoParam___closed__1_value),LEAN_SCALAR_PTR_LITERAL(26, 175, 18, 116, 252, 50, 128, 45)}};
static const lean_object* lp_mathlib_LinearOrder_min__def___autoParam___closed__2 = (const lean_object*)&lp_mathlib_LinearOrder_min__def___autoParam___closed__2_value;
static lean_once_cell_t lp_mathlib_LinearOrder_min__def___autoParam___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearOrder_min__def___autoParam___closed__3;
static lean_once_cell_t lp_mathlib_LinearOrder_min__def___autoParam___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearOrder_min__def___autoParam___closed__4;
static const lean_ctor_object lp_mathlib_LinearOrder_min__def___autoParam___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__11_value),((lean_object*)&lp_mathlib_LinearOrder_min__def___autoParam___closed__0_value)}};
static const lean_object* lp_mathlib_LinearOrder_min__def___autoParam___closed__5 = (const lean_object*)&lp_mathlib_LinearOrder_min__def___autoParam___closed__5_value;
static lean_once_cell_t lp_mathlib_LinearOrder_min__def___autoParam___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearOrder_min__def___autoParam___closed__6;
static lean_once_cell_t lp_mathlib_LinearOrder_min__def___autoParam___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearOrder_min__def___autoParam___closed__7;
static lean_once_cell_t lp_mathlib_LinearOrder_min__def___autoParam___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearOrder_min__def___autoParam___closed__8;
static lean_once_cell_t lp_mathlib_LinearOrder_min__def___autoParam___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearOrder_min__def___autoParam___closed__9;
static lean_once_cell_t lp_mathlib_LinearOrder_min__def___autoParam___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearOrder_min__def___autoParam___closed__10;
static lean_once_cell_t lp_mathlib_LinearOrder_min__def___autoParam___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearOrder_min__def___autoParam___closed__11;
static lean_once_cell_t lp_mathlib_LinearOrder_min__def___autoParam___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearOrder_min__def___autoParam___closed__12;
static lean_once_cell_t lp_mathlib_LinearOrder_min__def___autoParam___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearOrder_min__def___autoParam___closed__13;
static lean_once_cell_t lp_mathlib_LinearOrder_min__def___autoParam___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearOrder_min__def___autoParam___closed__14;
static lean_once_cell_t lp_mathlib_LinearOrder_min__def___autoParam___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearOrder_min__def___autoParam___closed__15;
static lean_once_cell_t lp_mathlib_LinearOrder_min__def___autoParam___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearOrder_min__def___autoParam___closed__16;
static lean_once_cell_t lp_mathlib_LinearOrder_min__def___autoParam___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearOrder_min__def___autoParam___closed__17;
static lean_once_cell_t lp_mathlib_LinearOrder_min__def___autoParam___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearOrder_min__def___autoParam___closed__18;
static lean_once_cell_t lp_mathlib_LinearOrder_min__def___autoParam___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearOrder_min__def___autoParam___closed__19;
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_min__def___autoParam;
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_max__def___autoParam;
static lean_once_cell_t lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__0;
static lean_once_cell_t lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__1;
static lean_once_cell_t lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__2;
static lean_once_cell_t lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__3;
static lean_once_cell_t lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__4;
static lean_once_cell_t lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__5;
static lean_once_cell_t lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__6;
static lean_once_cell_t lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__7;
static lean_once_cell_t lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Defs_LinearOrder_0__Ordering_Compares_match__1_splitter___redArg(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Defs_LinearOrder_0__Ordering_Compares_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Defs_LinearOrder_0__Ordering_Compares_match__1_splitter(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Defs_LinearOrder_0__Ordering_Compares_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinOrd_instCoeSortType;
LEAN_EXPORT lean_object* lp_mathlib_maxDefault___redArg(lean_object* v_inst_1_, lean_object* v_a_2_, lean_object* v_b_3_){
_start:
{
lean_object* v___x_4_; uint8_t v___x_5_; 
lean_inc(v_b_3_);
lean_inc(v_a_2_);
v___x_4_ = lean_apply_2(v_inst_1_, v_a_2_, v_b_3_);
v___x_5_ = lean_unbox(v___x_4_);
if (v___x_5_ == 0)
{
lean_dec(v_b_3_);
return v_a_2_;
}
else
{
lean_dec(v_a_2_);
return v_b_3_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_maxDefault(lean_object* v_00_u03b1_6_, lean_object* v_inst_7_, lean_object* v_inst_8_, lean_object* v_a_9_, lean_object* v_b_10_){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lp_mathlib_maxDefault___redArg(v_inst_8_, v_a_9_, v_b_10_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_minDefault___redArg(lean_object* v_inst_12_, lean_object* v_a_13_, lean_object* v_b_14_){
_start:
{
lean_object* v___x_15_; uint8_t v___x_16_; 
lean_inc(v_b_14_);
lean_inc(v_a_13_);
v___x_15_ = lean_apply_2(v_inst_12_, v_a_13_, v_b_14_);
v___x_16_ = lean_unbox(v___x_15_);
if (v___x_16_ == 0)
{
lean_dec(v_a_13_);
return v_b_14_;
}
else
{
lean_dec(v_b_14_);
return v_a_13_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_minDefault(lean_object* v_00_u03b1_17_, lean_object* v_inst_18_, lean_object* v_inst_19_, lean_object* v_a_20_, lean_object* v_b_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lp_mathlib_minDefault___redArg(v_inst_19_, v_a_20_, v_b_21_);
return v___x_22_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__15(void){
_start:
{
lean_object* v___x_67_; lean_object* v___x_68_; 
v___x_67_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__14));
v___x_68_ = l_String_toRawSubstring_x27(v___x_67_);
return v___x_68_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__18(void){
_start:
{
lean_object* v___x_72_; lean_object* v___x_73_; 
v___x_72_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__17));
v___x_73_ = l_String_toRawSubstring_x27(v___x_72_);
return v___x_73_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__33(void){
_start:
{
lean_object* v___x_106_; 
v___x_106_ = l_Array_mkArray0(lean_box(0));
return v___x_106_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__39(void){
_start:
{
lean_object* v___x_116_; lean_object* v___x_117_; 
v___x_116_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__38));
v___x_117_ = l_String_toRawSubstring_x27(v___x_116_);
return v___x_117_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__47(void){
_start:
{
lean_object* v___x_132_; lean_object* v___x_133_; 
v___x_132_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__46));
v___x_133_ = l_String_toRawSubstring_x27(v___x_132_);
return v___x_133_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__70(void){
_start:
{
lean_object* v___x_184_; lean_object* v___x_185_; 
v___x_184_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__69));
v___x_185_ = l_String_toRawSubstring_x27(v___x_184_);
return v___x_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1(lean_object* v_x_198_, lean_object* v_a_199_, lean_object* v_a_200_){
_start:
{
lean_object* v___x_201_; uint8_t v___x_202_; 
v___x_201_ = ((lean_object*)(lp_mathlib_tacticCompareOfLessAndEq__rfl___closed__1));
v___x_202_ = l_Lean_Syntax_isOfKind(v_x_198_, v___x_201_);
if (v___x_202_ == 0)
{
lean_object* v___x_203_; lean_object* v___x_204_; 
v___x_203_ = lean_box(1);
v___x_204_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_204_, 0, v___x_203_);
lean_ctor_set(v___x_204_, 1, v_a_200_);
return v___x_204_;
}
else
{
lean_object* v_quotContext_205_; lean_object* v_currMacroScope_206_; lean_object* v_ref_207_; uint8_t v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; 
v_quotContext_205_ = lean_ctor_get(v_a_199_, 1);
v_currMacroScope_206_ = lean_ctor_get(v_a_199_, 2);
v_ref_207_ = lean_ctor_get(v_a_199_, 5);
v___x_208_ = 0;
v___x_209_ = l_Lean_SourceInfo_fromRef(v_ref_207_, v___x_208_);
v___x_210_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__4));
v___x_211_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__5));
lean_inc_n(v___x_209_, 72);
v___x_212_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_212_, 0, v___x_209_);
lean_ctor_set(v___x_212_, 1, v___x_211_);
v___x_213_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__7));
v___x_214_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__9));
v___x_215_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__11));
v___x_216_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__12));
v___x_217_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__13));
v___x_218_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_218_, 0, v___x_209_);
lean_ctor_set(v___x_218_, 1, v___x_216_);
v___x_219_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__15, &lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__15_once, _init_lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__15);
v___x_220_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__16));
lean_inc_n(v_currMacroScope_206_, 5);
lean_inc_n(v_quotContext_205_, 5);
v___x_221_ = l_Lean_addMacroScope(v_quotContext_205_, v___x_220_, v_currMacroScope_206_);
v___x_222_ = lean_box(0);
v___x_223_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_223_, 0, v___x_209_);
lean_ctor_set(v___x_223_, 1, v___x_219_);
lean_ctor_set(v___x_223_, 2, v___x_221_);
lean_ctor_set(v___x_223_, 3, v___x_222_);
v___x_224_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__18, &lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__18_once, _init_lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__18);
v___x_225_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__19));
v___x_226_ = l_Lean_addMacroScope(v_quotContext_205_, v___x_225_, v_currMacroScope_206_);
v___x_227_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_227_, 0, v___x_209_);
lean_ctor_set(v___x_227_, 1, v___x_224_);
lean_ctor_set(v___x_227_, 2, v___x_226_);
lean_ctor_set(v___x_227_, 3, v___x_222_);
lean_inc_ref(v___x_227_);
lean_inc_ref(v___x_223_);
v___x_228_ = l_Lean_Syntax_node2(v___x_209_, v___x_215_, v___x_223_, v___x_227_);
v___x_229_ = l_Lean_Syntax_node2(v___x_209_, v___x_217_, v___x_218_, v___x_228_);
v___x_230_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__20));
v___x_231_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_231_, 0, v___x_209_);
lean_ctor_set(v___x_231_, 1, v___x_230_);
v___x_232_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__21));
v___x_233_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__22));
v___x_234_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_234_, 0, v___x_209_);
lean_ctor_set(v___x_234_, 1, v___x_232_);
v___x_235_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__24));
v___x_236_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__25));
v___x_237_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_237_, 0, v___x_209_);
lean_ctor_set(v___x_237_, 1, v___x_236_);
v___x_238_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__27));
v___x_239_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__28));
v___x_240_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_240_, 0, v___x_209_);
lean_ctor_set(v___x_240_, 1, v___x_239_);
v___x_241_ = l_Lean_Syntax_node1(v___x_209_, v___x_238_, v___x_240_);
lean_inc(v___x_241_);
v___x_242_ = l_Lean_Syntax_node1(v___x_209_, v___x_215_, v___x_241_);
v___x_243_ = l_Lean_Syntax_node1(v___x_209_, v___x_214_, v___x_242_);
v___x_244_ = l_Lean_Syntax_node1(v___x_209_, v___x_213_, v___x_243_);
lean_inc_ref_n(v___x_237_, 2);
v___x_245_ = l_Lean_Syntax_node2(v___x_209_, v___x_235_, v___x_237_, v___x_244_);
v___x_246_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__29));
v___x_247_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__30));
v___x_248_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_248_, 0, v___x_209_);
lean_ctor_set(v___x_248_, 1, v___x_246_);
v___x_249_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__32));
v___x_250_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__33, &lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__33_once, _init_lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__33);
v___x_251_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_251_, 0, v___x_209_);
lean_ctor_set(v___x_251_, 1, v___x_215_);
lean_ctor_set(v___x_251_, 2, v___x_250_);
lean_inc_ref_n(v___x_251_, 19);
v___x_252_ = l_Lean_Syntax_node1(v___x_209_, v___x_249_, v___x_251_);
v___x_253_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__34));
v___x_254_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_254_, 0, v___x_209_);
lean_ctor_set(v___x_254_, 1, v___x_253_);
v___x_255_ = l_Lean_Syntax_node1(v___x_209_, v___x_215_, v___x_254_);
v___x_256_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__35));
v___x_257_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_257_, 0, v___x_209_);
lean_ctor_set(v___x_257_, 1, v___x_256_);
v___x_258_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__37));
v___x_259_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__39, &lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__39_once, _init_lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__39);
v___x_260_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__40));
v___x_261_ = l_Lean_addMacroScope(v_quotContext_205_, v___x_260_, v_currMacroScope_206_);
v___x_262_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__44));
v___x_263_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_263_, 0, v___x_209_);
lean_ctor_set(v___x_263_, 1, v___x_259_);
lean_ctor_set(v___x_263_, 2, v___x_261_);
lean_ctor_set(v___x_263_, 3, v___x_262_);
v___x_264_ = l_Lean_Syntax_node3(v___x_209_, v___x_258_, v___x_251_, v___x_251_, v___x_263_);
v___x_265_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__45));
v___x_266_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_266_, 0, v___x_209_);
lean_ctor_set(v___x_266_, 1, v___x_265_);
v___x_267_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__47, &lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__47_once, _init_lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__47);
v___x_268_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__48));
v___x_269_ = l_Lean_addMacroScope(v_quotContext_205_, v___x_268_, v_currMacroScope_206_);
v___x_270_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__50));
v___x_271_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_271_, 0, v___x_209_);
lean_ctor_set(v___x_271_, 1, v___x_267_);
lean_ctor_set(v___x_271_, 2, v___x_269_);
lean_ctor_set(v___x_271_, 3, v___x_270_);
v___x_272_ = l_Lean_Syntax_node3(v___x_209_, v___x_258_, v___x_251_, v___x_251_, v___x_271_);
v___x_273_ = l_Lean_Syntax_node3(v___x_209_, v___x_215_, v___x_264_, v___x_266_, v___x_272_);
v___x_274_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__51));
v___x_275_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_275_, 0, v___x_209_);
lean_ctor_set(v___x_275_, 1, v___x_274_);
v___x_276_ = l_Lean_Syntax_node3(v___x_209_, v___x_215_, v___x_257_, v___x_273_, v___x_275_);
lean_inc(v___x_255_);
lean_inc_ref(v___x_248_);
v___x_277_ = l_Lean_Syntax_node6(v___x_209_, v___x_247_, v___x_248_, v___x_252_, v___x_251_, v___x_255_, v___x_276_, v___x_251_);
v___x_278_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__53));
v___x_279_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__56));
v___x_280_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__57));
v___x_281_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_281_, 0, v___x_209_);
lean_ctor_set(v___x_281_, 1, v___x_280_);
v___x_282_ = l_Lean_Syntax_node3(v___x_209_, v___x_279_, v___x_281_, v___x_251_, v___x_251_);
v___x_283_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__58));
v___x_284_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_284_, 0, v___x_209_);
lean_ctor_set(v___x_284_, 1, v___x_283_);
lean_inc_ref_n(v___x_284_, 2);
v___x_285_ = l_Lean_Syntax_node3(v___x_209_, v___x_278_, v___x_282_, v___x_284_, v___x_241_);
lean_inc_ref(v___x_231_);
v___x_286_ = l_Lean_Syntax_node3(v___x_209_, v___x_215_, v___x_277_, v___x_231_, v___x_285_);
v___x_287_ = l_Lean_Syntax_node1(v___x_209_, v___x_214_, v___x_286_);
v___x_288_ = l_Lean_Syntax_node1(v___x_209_, v___x_213_, v___x_287_);
v___x_289_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__59));
v___x_290_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_290_, 0, v___x_209_);
lean_ctor_set(v___x_290_, 1, v___x_289_);
lean_inc_ref_n(v___x_290_, 2);
lean_inc_ref_n(v___x_212_, 2);
v___x_291_ = l_Lean_Syntax_node3(v___x_209_, v___x_210_, v___x_212_, v___x_288_, v___x_290_);
v___x_292_ = l_Lean_Syntax_node1(v___x_209_, v___x_215_, v___x_291_);
v___x_293_ = l_Lean_Syntax_node1(v___x_209_, v___x_214_, v___x_292_);
v___x_294_ = l_Lean_Syntax_node1(v___x_209_, v___x_213_, v___x_293_);
v___x_295_ = l_Lean_Syntax_node2(v___x_209_, v___x_235_, v___x_237_, v___x_294_);
v___x_296_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__60));
v___x_297_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__61));
v___x_298_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_298_, 0, v___x_209_);
lean_ctor_set(v___x_298_, 1, v___x_296_);
v___x_299_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__63));
v___x_300_ = l_Lean_Syntax_node2(v___x_209_, v___x_299_, v___x_251_, v___x_223_);
v___x_301_ = l_Lean_Syntax_node1(v___x_209_, v___x_215_, v___x_300_);
lean_inc_ref(v___x_298_);
v___x_302_ = l_Lean_Syntax_node5(v___x_209_, v___x_297_, v___x_298_, v___x_301_, v___x_251_, v___x_251_, v___x_251_);
v___x_303_ = l_Lean_Syntax_node2(v___x_209_, v___x_299_, v___x_251_, v___x_227_);
v___x_304_ = l_Lean_Syntax_node1(v___x_209_, v___x_215_, v___x_303_);
v___x_305_ = l_Lean_Syntax_node5(v___x_209_, v___x_297_, v___x_298_, v___x_304_, v___x_251_, v___x_251_, v___x_251_);
v___x_306_ = l_Lean_Syntax_node3(v___x_209_, v___x_278_, v___x_302_, v___x_284_, v___x_305_);
v___x_307_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__65));
v___x_308_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__67));
v___x_309_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__68));
v___x_310_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_310_, 0, v___x_209_);
lean_ctor_set(v___x_310_, 1, v___x_309_);
v___x_311_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__70, &lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__70_once, _init_lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__70);
v___x_312_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__71));
v___x_313_ = l_Lean_addMacroScope(v_quotContext_205_, v___x_312_, v_currMacroScope_206_);
v___x_314_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__75));
v___x_315_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_315_, 0, v___x_209_);
lean_ctor_set(v___x_315_, 1, v___x_311_);
lean_ctor_set(v___x_315_, 2, v___x_313_);
lean_ctor_set(v___x_315_, 3, v___x_314_);
v___x_316_ = l_Lean_Syntax_node2(v___x_209_, v___x_308_, v___x_310_, v___x_315_);
v___x_317_ = l_Lean_Syntax_node1(v___x_209_, v___x_307_, v___x_316_);
v___x_318_ = l_Lean_Syntax_node1(v___x_209_, v___x_215_, v___x_317_);
v___x_319_ = l_Lean_Syntax_node1(v___x_209_, v___x_249_, v___x_318_);
v___x_320_ = l_Lean_Syntax_node6(v___x_209_, v___x_247_, v___x_248_, v___x_319_, v___x_251_, v___x_255_, v___x_251_, v___x_251_);
v___x_321_ = l_Lean_Syntax_node3(v___x_209_, v___x_278_, v___x_306_, v___x_284_, v___x_320_);
v___x_322_ = l_Lean_Syntax_node1(v___x_209_, v___x_215_, v___x_321_);
v___x_323_ = l_Lean_Syntax_node1(v___x_209_, v___x_214_, v___x_322_);
v___x_324_ = l_Lean_Syntax_node1(v___x_209_, v___x_213_, v___x_323_);
v___x_325_ = l_Lean_Syntax_node3(v___x_209_, v___x_210_, v___x_212_, v___x_324_, v___x_290_);
v___x_326_ = l_Lean_Syntax_node1(v___x_209_, v___x_215_, v___x_325_);
v___x_327_ = l_Lean_Syntax_node1(v___x_209_, v___x_214_, v___x_326_);
v___x_328_ = l_Lean_Syntax_node1(v___x_209_, v___x_213_, v___x_327_);
v___x_329_ = l_Lean_Syntax_node2(v___x_209_, v___x_235_, v___x_237_, v___x_328_);
v___x_330_ = l_Lean_Syntax_node3(v___x_209_, v___x_215_, v___x_245_, v___x_295_, v___x_329_);
v___x_331_ = l_Lean_Syntax_node2(v___x_209_, v___x_233_, v___x_234_, v___x_330_);
v___x_332_ = l_Lean_Syntax_node3(v___x_209_, v___x_215_, v___x_229_, v___x_231_, v___x_331_);
v___x_333_ = l_Lean_Syntax_node1(v___x_209_, v___x_214_, v___x_332_);
v___x_334_ = l_Lean_Syntax_node1(v___x_209_, v___x_213_, v___x_333_);
v___x_335_ = l_Lean_Syntax_node3(v___x_209_, v___x_210_, v___x_212_, v___x_334_, v___x_290_);
v___x_336_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_336_, 0, v___x_335_);
lean_ctor_set(v___x_336_, 1, v_a_200_);
return v___x_336_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___boxed(lean_object* v_x_337_, lean_object* v_a_338_, lean_object* v_a_339_){
_start:
{
lean_object* v_res_340_; 
v_res_340_ = lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1(v_x_337_, v_a_338_, v_a_339_);
lean_dec_ref(v_a_338_);
return v_res_340_;
}
}
static lean_object* _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__3(void){
_start:
{
lean_object* v___x_349_; lean_object* v___x_350_; 
v___x_349_ = ((lean_object*)(lp_mathlib_LinearOrder_min__def___autoParam___closed__1));
v___x_350_ = l_Lean_mkAtom(v___x_349_);
return v___x_350_;
}
}
static lean_object* _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__4(void){
_start:
{
lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; 
v___x_351_ = lean_obj_once(&lp_mathlib_LinearOrder_min__def___autoParam___closed__3, &lp_mathlib_LinearOrder_min__def___autoParam___closed__3_once, _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__3);
v___x_352_ = ((lean_object*)(lp_mathlib_LinearOrder_min__def___autoParam___closed__0));
v___x_353_ = lean_array_push(v___x_352_, v___x_351_);
return v___x_353_;
}
}
static lean_object* _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__6(void){
_start:
{
lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; 
v___x_358_ = ((lean_object*)(lp_mathlib_LinearOrder_min__def___autoParam___closed__5));
v___x_359_ = lean_obj_once(&lp_mathlib_LinearOrder_min__def___autoParam___closed__4, &lp_mathlib_LinearOrder_min__def___autoParam___closed__4_once, _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__4);
v___x_360_ = lean_array_push(v___x_359_, v___x_358_);
return v___x_360_;
}
}
static lean_object* _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__7(void){
_start:
{
lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; lean_object* v___x_364_; 
v___x_361_ = lean_obj_once(&lp_mathlib_LinearOrder_min__def___autoParam___closed__6, &lp_mathlib_LinearOrder_min__def___autoParam___closed__6_once, _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__6);
v___x_362_ = ((lean_object*)(lp_mathlib_LinearOrder_min__def___autoParam___closed__2));
v___x_363_ = lean_box(2);
v___x_364_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_364_, 0, v___x_363_);
lean_ctor_set(v___x_364_, 1, v___x_362_);
lean_ctor_set(v___x_364_, 2, v___x_361_);
return v___x_364_;
}
}
static lean_object* _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__8(void){
_start:
{
lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; 
v___x_365_ = lean_obj_once(&lp_mathlib_LinearOrder_min__def___autoParam___closed__7, &lp_mathlib_LinearOrder_min__def___autoParam___closed__7_once, _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__7);
v___x_366_ = ((lean_object*)(lp_mathlib_LinearOrder_min__def___autoParam___closed__0));
v___x_367_ = lean_array_push(v___x_366_, v___x_365_);
return v___x_367_;
}
}
static lean_object* _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__9(void){
_start:
{
lean_object* v___x_368_; lean_object* v___x_369_; 
v___x_368_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__20));
v___x_369_ = l_Lean_mkAtom(v___x_368_);
return v___x_369_;
}
}
static lean_object* _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__10(void){
_start:
{
lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; 
v___x_370_ = lean_obj_once(&lp_mathlib_LinearOrder_min__def___autoParam___closed__9, &lp_mathlib_LinearOrder_min__def___autoParam___closed__9_once, _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__9);
v___x_371_ = lean_obj_once(&lp_mathlib_LinearOrder_min__def___autoParam___closed__8, &lp_mathlib_LinearOrder_min__def___autoParam___closed__8_once, _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__8);
v___x_372_ = lean_array_push(v___x_371_, v___x_370_);
return v___x_372_;
}
}
static lean_object* _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__11(void){
_start:
{
lean_object* v___x_373_; lean_object* v___x_374_; 
v___x_373_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__28));
v___x_374_ = l_Lean_mkAtom(v___x_373_);
return v___x_374_;
}
}
static lean_object* _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__12(void){
_start:
{
lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; 
v___x_375_ = lean_obj_once(&lp_mathlib_LinearOrder_min__def___autoParam___closed__11, &lp_mathlib_LinearOrder_min__def___autoParam___closed__11_once, _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__11);
v___x_376_ = ((lean_object*)(lp_mathlib_LinearOrder_min__def___autoParam___closed__0));
v___x_377_ = lean_array_push(v___x_376_, v___x_375_);
return v___x_377_;
}
}
static lean_object* _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__13(void){
_start:
{
lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; 
v___x_378_ = lean_obj_once(&lp_mathlib_LinearOrder_min__def___autoParam___closed__12, &lp_mathlib_LinearOrder_min__def___autoParam___closed__12_once, _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__12);
v___x_379_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__27));
v___x_380_ = lean_box(2);
v___x_381_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_381_, 0, v___x_380_);
lean_ctor_set(v___x_381_, 1, v___x_379_);
lean_ctor_set(v___x_381_, 2, v___x_378_);
return v___x_381_;
}
}
static lean_object* _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__14(void){
_start:
{
lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; 
v___x_382_ = lean_obj_once(&lp_mathlib_LinearOrder_min__def___autoParam___closed__13, &lp_mathlib_LinearOrder_min__def___autoParam___closed__13_once, _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__13);
v___x_383_ = lean_obj_once(&lp_mathlib_LinearOrder_min__def___autoParam___closed__10, &lp_mathlib_LinearOrder_min__def___autoParam___closed__10_once, _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__10);
v___x_384_ = lean_array_push(v___x_383_, v___x_382_);
return v___x_384_;
}
}
static lean_object* _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__15(void){
_start:
{
lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; 
v___x_385_ = lean_obj_once(&lp_mathlib_LinearOrder_min__def___autoParam___closed__14, &lp_mathlib_LinearOrder_min__def___autoParam___closed__14_once, _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__14);
v___x_386_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__11));
v___x_387_ = lean_box(2);
v___x_388_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_388_, 0, v___x_387_);
lean_ctor_set(v___x_388_, 1, v___x_386_);
lean_ctor_set(v___x_388_, 2, v___x_385_);
return v___x_388_;
}
}
static lean_object* _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__16(void){
_start:
{
lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; 
v___x_389_ = lean_obj_once(&lp_mathlib_LinearOrder_min__def___autoParam___closed__15, &lp_mathlib_LinearOrder_min__def___autoParam___closed__15_once, _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__15);
v___x_390_ = ((lean_object*)(lp_mathlib_LinearOrder_min__def___autoParam___closed__0));
v___x_391_ = lean_array_push(v___x_390_, v___x_389_);
return v___x_391_;
}
}
static lean_object* _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__17(void){
_start:
{
lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; 
v___x_392_ = lean_obj_once(&lp_mathlib_LinearOrder_min__def___autoParam___closed__16, &lp_mathlib_LinearOrder_min__def___autoParam___closed__16_once, _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__16);
v___x_393_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__9));
v___x_394_ = lean_box(2);
v___x_395_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_395_, 0, v___x_394_);
lean_ctor_set(v___x_395_, 1, v___x_393_);
lean_ctor_set(v___x_395_, 2, v___x_392_);
return v___x_395_;
}
}
static lean_object* _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__18(void){
_start:
{
lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_398_; 
v___x_396_ = lean_obj_once(&lp_mathlib_LinearOrder_min__def___autoParam___closed__17, &lp_mathlib_LinearOrder_min__def___autoParam___closed__17_once, _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__17);
v___x_397_ = ((lean_object*)(lp_mathlib_LinearOrder_min__def___autoParam___closed__0));
v___x_398_ = lean_array_push(v___x_397_, v___x_396_);
return v___x_398_;
}
}
static lean_object* _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__19(void){
_start:
{
lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; 
v___x_399_ = lean_obj_once(&lp_mathlib_LinearOrder_min__def___autoParam___closed__18, &lp_mathlib_LinearOrder_min__def___autoParam___closed__18_once, _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__18);
v___x_400_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__7));
v___x_401_ = lean_box(2);
v___x_402_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_402_, 0, v___x_401_);
lean_ctor_set(v___x_402_, 1, v___x_400_);
lean_ctor_set(v___x_402_, 2, v___x_399_);
return v___x_402_;
}
}
static lean_object* _init_lp_mathlib_LinearOrder_min__def___autoParam(void){
_start:
{
lean_object* v___x_403_; 
v___x_403_ = lean_obj_once(&lp_mathlib_LinearOrder_min__def___autoParam___closed__19, &lp_mathlib_LinearOrder_min__def___autoParam___closed__19_once, _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__19);
return v___x_403_;
}
}
static lean_object* _init_lp_mathlib_LinearOrder_max__def___autoParam(void){
_start:
{
lean_object* v___x_404_; 
v___x_404_ = lean_obj_once(&lp_mathlib_LinearOrder_min__def___autoParam___closed__19, &lp_mathlib_LinearOrder_min__def___autoParam___closed__19_once, _init_lp_mathlib_LinearOrder_min__def___autoParam___closed__19);
return v___x_404_;
}
}
static lean_object* _init_lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__0(void){
_start:
{
lean_object* v___x_405_; lean_object* v___x_406_; 
v___x_405_ = ((lean_object*)(lp_mathlib_tacticCompareOfLessAndEq__rfl___closed__2));
v___x_406_ = l_Lean_mkAtom(v___x_405_);
return v___x_406_;
}
}
static lean_object* _init_lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__1(void){
_start:
{
lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; 
v___x_407_ = lean_obj_once(&lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__0, &lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__0_once, _init_lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__0);
v___x_408_ = ((lean_object*)(lp_mathlib_LinearOrder_min__def___autoParam___closed__0));
v___x_409_ = lean_array_push(v___x_408_, v___x_407_);
return v___x_409_;
}
}
static lean_object* _init_lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__2(void){
_start:
{
lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; 
v___x_410_ = lean_obj_once(&lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__1, &lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__1_once, _init_lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__1);
v___x_411_ = ((lean_object*)(lp_mathlib_tacticCompareOfLessAndEq__rfl___closed__1));
v___x_412_ = lean_box(2);
v___x_413_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_413_, 0, v___x_412_);
lean_ctor_set(v___x_413_, 1, v___x_411_);
lean_ctor_set(v___x_413_, 2, v___x_410_);
return v___x_413_;
}
}
static lean_object* _init_lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__3(void){
_start:
{
lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; 
v___x_414_ = lean_obj_once(&lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__2, &lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__2_once, _init_lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__2);
v___x_415_ = ((lean_object*)(lp_mathlib_LinearOrder_min__def___autoParam___closed__0));
v___x_416_ = lean_array_push(v___x_415_, v___x_414_);
return v___x_416_;
}
}
static lean_object* _init_lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__4(void){
_start:
{
lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; 
v___x_417_ = lean_obj_once(&lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__3, &lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__3_once, _init_lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__3);
v___x_418_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__11));
v___x_419_ = lean_box(2);
v___x_420_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_420_, 0, v___x_419_);
lean_ctor_set(v___x_420_, 1, v___x_418_);
lean_ctor_set(v___x_420_, 2, v___x_417_);
return v___x_420_;
}
}
static lean_object* _init_lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__5(void){
_start:
{
lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v___x_423_; 
v___x_421_ = lean_obj_once(&lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__4, &lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__4_once, _init_lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__4);
v___x_422_ = ((lean_object*)(lp_mathlib_LinearOrder_min__def___autoParam___closed__0));
v___x_423_ = lean_array_push(v___x_422_, v___x_421_);
return v___x_423_;
}
}
static lean_object* _init_lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__6(void){
_start:
{
lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; 
v___x_424_ = lean_obj_once(&lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__5, &lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__5_once, _init_lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__5);
v___x_425_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__9));
v___x_426_ = lean_box(2);
v___x_427_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_427_, 0, v___x_426_);
lean_ctor_set(v___x_427_, 1, v___x_425_);
lean_ctor_set(v___x_427_, 2, v___x_424_);
return v___x_427_;
}
}
static lean_object* _init_lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__7(void){
_start:
{
lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v___x_430_; 
v___x_428_ = lean_obj_once(&lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__6, &lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__6_once, _init_lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__6);
v___x_429_ = ((lean_object*)(lp_mathlib_LinearOrder_min__def___autoParam___closed__0));
v___x_430_ = lean_array_push(v___x_429_, v___x_428_);
return v___x_430_;
}
}
static lean_object* _init_lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__8(void){
_start:
{
lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; lean_object* v___x_434_; 
v___x_431_ = lean_obj_once(&lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__7, &lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__7_once, _init_lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__7);
v___x_432_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Order__Defs__LinearOrder______macroRules__tacticCompareOfLessAndEq__rfl__1___closed__7));
v___x_433_ = lean_box(2);
v___x_434_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_434_, 0, v___x_433_);
lean_ctor_set(v___x_434_, 1, v___x_432_);
lean_ctor_set(v___x_434_, 2, v___x_431_);
return v___x_434_;
}
}
static lean_object* _init_lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam(void){
_start:
{
lean_object* v___x_435_; 
v___x_435_ = lean_obj_once(&lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__8, &lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__8_once, _init_lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam___closed__8);
return v___x_435_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Defs_LinearOrder_0__Ordering_Compares_match__1_splitter___redArg(uint8_t v_x_436_, lean_object* v_x_437_, lean_object* v_x_438_, lean_object* v_h__1_439_, lean_object* v_h__2_440_, lean_object* v_h__3_441_){
_start:
{
switch(v_x_436_)
{
case 0:
{
lean_object* v___x_442_; 
lean_dec(v_h__3_441_);
lean_dec(v_h__2_440_);
v___x_442_ = lean_apply_2(v_h__1_439_, v_x_437_, v_x_438_);
return v___x_442_;
}
case 1:
{
lean_object* v___x_443_; 
lean_dec(v_h__3_441_);
lean_dec(v_h__1_439_);
v___x_443_ = lean_apply_2(v_h__2_440_, v_x_437_, v_x_438_);
return v___x_443_;
}
default: 
{
lean_object* v___x_444_; 
lean_dec(v_h__2_440_);
lean_dec(v_h__1_439_);
v___x_444_ = lean_apply_2(v_h__3_441_, v_x_437_, v_x_438_);
return v___x_444_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Defs_LinearOrder_0__Ordering_Compares_match__1_splitter___redArg___boxed(lean_object* v_x_445_, lean_object* v_x_446_, lean_object* v_x_447_, lean_object* v_h__1_448_, lean_object* v_h__2_449_, lean_object* v_h__3_450_){
_start:
{
uint8_t v_x_28__boxed_451_; lean_object* v_res_452_; 
v_x_28__boxed_451_ = lean_unbox(v_x_445_);
v_res_452_ = lp_mathlib___private_Mathlib_Order_Defs_LinearOrder_0__Ordering_Compares_match__1_splitter___redArg(v_x_28__boxed_451_, v_x_446_, v_x_447_, v_h__1_448_, v_h__2_449_, v_h__3_450_);
return v_res_452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Defs_LinearOrder_0__Ordering_Compares_match__1_splitter(lean_object* v_00_u03b1_453_, lean_object* v_motive_454_, uint8_t v_x_455_, lean_object* v_x_456_, lean_object* v_x_457_, lean_object* v_h__1_458_, lean_object* v_h__2_459_, lean_object* v_h__3_460_){
_start:
{
switch(v_x_455_)
{
case 0:
{
lean_object* v___x_461_; 
lean_dec(v_h__3_460_);
lean_dec(v_h__2_459_);
v___x_461_ = lean_apply_2(v_h__1_458_, v_x_456_, v_x_457_);
return v___x_461_;
}
case 1:
{
lean_object* v___x_462_; 
lean_dec(v_h__3_460_);
lean_dec(v_h__1_458_);
v___x_462_ = lean_apply_2(v_h__2_459_, v_x_456_, v_x_457_);
return v___x_462_;
}
default: 
{
lean_object* v___x_463_; 
lean_dec(v_h__2_459_);
lean_dec(v_h__1_458_);
v___x_463_ = lean_apply_2(v_h__3_460_, v_x_456_, v_x_457_);
return v___x_463_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_Defs_LinearOrder_0__Ordering_Compares_match__1_splitter___boxed(lean_object* v_00_u03b1_464_, lean_object* v_motive_465_, lean_object* v_x_466_, lean_object* v_x_467_, lean_object* v_x_468_, lean_object* v_h__1_469_, lean_object* v_h__2_470_, lean_object* v_h__3_471_){
_start:
{
uint8_t v_x_43__boxed_472_; lean_object* v_res_473_; 
v_x_43__boxed_472_ = lean_unbox(v_x_466_);
v_res_473_ = lp_mathlib___private_Mathlib_Order_Defs_LinearOrder_0__Ordering_Compares_match__1_splitter(v_00_u03b1_464_, v_motive_465_, v_x_43__boxed_472_, v_x_467_, v_x_468_, v_h__1_469_, v_h__2_470_, v_h__3_471_);
return v_res_473_;
}
}
static lean_object* _init_lp_mathlib_LinOrd_instCoeSortType(void){
_start:
{
lean_object* v___x_474_; 
v___x_474_ = lean_box(0);
return v___x_474_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Classes_Order(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Trans(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Ordering_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Push_Attr(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Simps(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_SplitIfs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Defs_PartialOrder(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Defs_LinearOrder(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Classes_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Trans(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Ordering_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Push_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Simps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_SplitIfs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Defs_PartialOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_LinOrd_instCoeSortType = _init_lp_mathlib_LinOrd_instCoeSortType();
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Defs_LinearOrder(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_LinearOrder_min__def___autoParam = _init_lp_mathlib_LinearOrder_min__def___autoParam();
lean_mark_persistent(lp_mathlib_LinearOrder_min__def___autoParam);
lp_mathlib_LinearOrder_max__def___autoParam = _init_lp_mathlib_LinearOrder_max__def___autoParam();
lean_mark_persistent(lp_mathlib_LinearOrder_max__def___autoParam);
lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam = _init_lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam();
lean_mark_persistent(lp_mathlib_LinearOrder_compare__eq__compareOfLessAndEq___autoParam);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Classes_Order(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Trans(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Ordering_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Push_Attr(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Simps(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_SplitIfs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Defs_PartialOrder(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Defs_LinearOrder(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Classes_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Trans(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Ordering_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Push_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Simps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_SplitIfs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Defs_PartialOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Defs_LinearOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Defs_LinearOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Defs_LinearOrder(builtin);
}
#ifdef __cplusplus
}
#endif
