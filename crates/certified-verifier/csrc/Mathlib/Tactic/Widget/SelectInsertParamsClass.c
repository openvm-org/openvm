// Lean compiler output
// Module: Mathlib.Tactic.Widget.SelectInsertParamsClass
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Widget.InteractiveGoal public meta import Lean.Elab.Deriving.Basic public import Lean.Widget.InteractiveGoal
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
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkCIdent(lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Elab_Command_liftTermElabM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_elabCommand(lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_st_ref_get(lean_object*);
uint8_t l_Lean_isInductiveCore(lean_object*, lean_object*);
lean_object* l_Lean_Elab_registerDerivingHandler(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "declaration"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__3_value),LEAN_SCALAR_PTR_LITERAL(157, 246, 223, 221, 242, 35, 238, 117)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "declModifiers"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__6_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__6_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__6_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(0, 165, 146, 53, 36, 89, 7, 202)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__7_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__8_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__9;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "instance"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__11_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__11_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__11_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__10_value),LEAN_SCALAR_PTR_LITERAL(37, 156, 84, 218, 244, 57, 142, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__11_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "attrKind"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__14_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__14_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__14_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__13_value),LEAN_SCALAR_PTR_LITERAL(32, 164, 20, 104, 12, 221, 204, 110)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "declSig"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__16_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__16_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__16_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__15_value),LEAN_SCALAR_PTR_LITERAL(22, 101, 130, 251, 183, 19, 113, 82)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__16_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "typeSpec"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__17_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__18_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__18_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__18_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__18_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__18_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__17_value),LEAN_SCALAR_PTR_LITERAL(77, 126, 241, 117, 174, 189, 108, 62)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__18_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__19_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__20_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__21_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__21_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__21_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__21_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__21_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__20_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__21_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "SelectInsertParamsClass"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__22_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__23;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__22_value),LEAN_SCALAR_PTR_LITERAL(76, 63, 44, 37, 200, 168, 206, 15)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__24 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__24_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__24_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__25 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__25_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__24_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__26 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__26_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__26_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__27 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__27_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__25_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__27_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__28 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__28_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__29 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__29_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__30_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__30_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__30_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__30_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__30_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__30_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__29_value),LEAN_SCALAR_PTR_LITERAL(124, 9, 161, 194, 227, 100, 20, 110)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__30 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__30_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__31 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__31_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__32_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__32_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__32_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__32_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__32_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__32_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__31_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__32 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__32_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__33 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__33_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__34 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__34_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__34_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__35 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__35_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__36 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__36_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__37_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__37;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__38 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__38_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__39_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__39_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__38_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__39 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__39_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__39_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__40 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__40_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__41_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__41_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__41 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__41_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__41_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__42 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__42_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__43 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__43_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__44_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__44_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__44_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__43_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__44_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(247, 100, 102, 194, 167, 62, 107, 182)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__44 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__44_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__44_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__45 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__45_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__46_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__46_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__46_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__38_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__46_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(177, 181, 244, 12, 1, 14, 170, 235)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__46 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__46_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__46_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__47 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__47_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Server"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__48 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__48_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__49_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__49_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__48_value),LEAN_SCALAR_PTR_LITERAL(251, 1, 140, 35, 91, 244, 83, 213)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__49 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__49_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__49_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__50 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__50_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__51_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__51_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__43_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__51 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__51_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__51_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__52 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__52_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__53 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__53_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__53_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__54 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__54_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__54_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__55 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__55_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__52_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__55_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__56 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__56_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__50_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__56_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__57 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__57_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__47_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__57_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__58 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__58_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__45_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__58_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__59 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__59_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__42_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__59_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__60 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__60_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__61_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__42_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__60_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__61 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__61_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__62_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__40_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__61_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__62 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__62_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__63_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "explicit"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__63 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__63_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__64_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__64_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__64_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__64_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__64_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__64_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__64_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__63_value),LEAN_SCALAR_PTR_LITERAL(141, 201, 75, 195, 250, 223, 114, 184)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__64 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__64_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__65_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "@"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__65 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__65_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__66_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__66 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__66_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__67_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "declValSimple"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__67 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__67_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__68_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__68_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__68_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__68_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__68_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__68_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__68_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__67_value),LEAN_SCALAR_PTR_LITERAL(228, 117, 47, 248, 145, 185, 135, 188)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__68 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__68_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__69_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ":="};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__69 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__69_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__70_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "anonymousCtor"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__70 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__70_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__71_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__71_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__71_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__71_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__71_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__71_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__71_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__70_value),LEAN_SCALAR_PTR_LITERAL(56, 53, 154, 97, 179, 232, 94, 186)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__71 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__71_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__72_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⟨"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__72 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__72_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__73_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "fun"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__73 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__73_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__74_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__74_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__74_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__74_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__74_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__74_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__74_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__73_value),LEAN_SCALAR_PTR_LITERAL(249, 155, 133, 242, 71, 132, 191, 97)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__74 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__74_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__75_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "basicFun"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__75 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__75_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__76_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__76_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__76_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__76_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__76_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__76_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__76_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__75_value),LEAN_SCALAR_PTR_LITERAL(209, 134, 40, 160, 122, 195, 31, 223)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__76 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__76_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__77_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "prop"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__77 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__77_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__78_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__78;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__79_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__77_value),LEAN_SCALAR_PTR_LITERAL(56, 247, 67, 203, 121, 106, 5, 21)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__79 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__79_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__80_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=>"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__80 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__80_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__81_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "prop.pos"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__81 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__81_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__82_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__82;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__83_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "pos"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__83 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__83_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__84_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__77_value),LEAN_SCALAR_PTR_LITERAL(56, 247, 67, 203, 121, 106, 5, 21)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__84_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__84_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__83_value),LEAN_SCALAR_PTR_LITERAL(112, 108, 206, 132, 213, 178, 231, 70)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__84 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__84_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__85_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__85 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__85_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__86_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "prop.goals"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__86 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__86_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__87_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__87;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__88_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "goals"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__88 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__88_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__89_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__77_value),LEAN_SCALAR_PTR_LITERAL(56, 247, 67, 203, 121, 106, 5, 21)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__89_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__89_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__88_value),LEAN_SCALAR_PTR_LITERAL(11, 54, 13, 25, 121, 218, 193, 204)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__89 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__89_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__90_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "prop.selectedLocations"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__90 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__90_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__91_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__91;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__92_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "selectedLocations"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__92 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__92_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__93_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__77_value),LEAN_SCALAR_PTR_LITERAL(56, 247, 67, 203, 121, 106, 5, 21)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__93_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__93_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__92_value),LEAN_SCALAR_PTR_LITERAL(34, 49, 215, 250, 107, 200, 92, 83)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__93 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__93_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__94_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "prop.replaceRange"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__94 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__94_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__95_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__95;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__96_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "replaceRange"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__96 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__96_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__97_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__77_value),LEAN_SCALAR_PTR_LITERAL(56, 247, 67, 203, 121, 106, 5, 21)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__97_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__97_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__96_value),LEAN_SCALAR_PTR_LITERAL(103, 253, 137, 64, 127, 106, 247, 175)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__97 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__97_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__98_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⟩"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__98 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__98_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__99_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Termination"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__99 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__99_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__100_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "suffix"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__100 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__100_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__101_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__101_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__101_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__101_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__101_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__99_value),LEAN_SCALAR_PTR_LITERAL(128, 225, 226, 49, 186, 161, 212, 105)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__101_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__101_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__100_value),LEAN_SCALAR_PTR_LITERAL(245, 187, 99, 45, 217, 244, 244, 120)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__101 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__101_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_isInductive___at___00Lean_Elab_mkSelectInsertParamsInstanceHandler_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_isInductive___at___00Lean_Elab_mkSelectInsertParamsInstanceHandler_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_isInductive___at___00Lean_Elab_mkSelectInsertParamsInstanceHandler_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_isInductive___at___00Lean_Elab_mkSelectInsertParamsInstanceHandler_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_mkSelectInsertParamsInstanceHandler___lam__0(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_mkSelectInsertParamsInstanceHandler___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_Elab_mkSelectInsertParamsInstanceHandler_spec__2(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_Elab_mkSelectInsertParamsInstanceHandler_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_mkSelectInsertParamsInstanceHandler_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_mkSelectInsertParamsInstanceHandler_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_mkSelectInsertParamsInstanceHandler(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_mkSelectInsertParamsInstanceHandler___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_initFn___closed__0_00___x40_Mathlib_Tactic_Widget_SelectInsertParamsClass_1629491121____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Elab_mkSelectInsertParamsInstanceHandler___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_initFn___closed__0_00___x40_Mathlib_Tactic_Widget_SelectInsertParamsClass_1629491121____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_initFn___closed__0_00___x40_Mathlib_Tactic_Widget_SelectInsertParamsClass_1629491121____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_initFn_00___x40_Mathlib_Tactic_Widget_SelectInsertParamsClass_1629491121____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_initFn_00___x40_Mathlib_Tactic_Widget_SelectInsertParamsClass_1629491121____hygCtx___hyg_2____boxed(lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__9(void){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = l_Array_mkArray0(lean_box(0));
return v___x_19_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__23(void){
_start:
{
lean_object* v___x_53_; lean_object* v___x_54_; 
v___x_53_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__22));
v___x_54_ = l_String_toRawSubstring_x27(v___x_53_);
return v___x_54_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__37(void){
_start:
{
lean_object* v___x_85_; lean_object* v___x_86_; 
v___x_85_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__36));
v___x_86_ = l_String_toRawSubstring_x27(v___x_85_);
return v___x_86_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__78(void){
_start:
{
lean_object* v___x_185_; lean_object* v___x_186_; 
v___x_185_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__77));
v___x_186_ = l_String_toRawSubstring_x27(v___x_185_);
return v___x_186_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__82(void){
_start:
{
lean_object* v___x_191_; lean_object* v___x_192_; 
v___x_191_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__81));
v___x_192_ = l_String_toRawSubstring_x27(v___x_191_);
return v___x_192_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__87(void){
_start:
{
lean_object* v___x_199_; lean_object* v___x_200_; 
v___x_199_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__86));
v___x_200_ = l_String_toRawSubstring_x27(v___x_199_);
return v___x_200_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__91(void){
_start:
{
lean_object* v___x_206_; lean_object* v___x_207_; 
v___x_206_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__90));
v___x_207_ = l_String_toRawSubstring_x27(v___x_206_);
return v___x_207_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__95(void){
_start:
{
lean_object* v___x_213_; lean_object* v___x_214_; 
v___x_213_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__94));
v___x_214_ = l_String_toRawSubstring_x27(v___x_213_);
return v___x_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg(lean_object* v_declName_227_, lean_object* v_a_228_){
_start:
{
lean_object* v_ref_230_; lean_object* v_quotContext_231_; lean_object* v_currMacroScope_232_; uint8_t v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; 
v_ref_230_ = lean_ctor_get(v_a_228_, 5);
v_quotContext_231_ = lean_ctor_get(v_a_228_, 10);
v_currMacroScope_232_ = lean_ctor_get(v_a_228_, 11);
v___x_233_ = 0;
v___x_234_ = l_Lean_SourceInfo_fromRef(v_ref_230_, v___x_233_);
v___x_235_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__4));
v___x_236_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__6));
v___x_237_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__8));
v___x_238_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__9, &lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__9);
lean_inc_n(v___x_234_, 43);
v___x_239_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_239_, 0, v___x_234_);
lean_ctor_set(v___x_239_, 1, v___x_237_);
lean_ctor_set(v___x_239_, 2, v___x_238_);
lean_inc_ref_n(v___x_239_, 17);
v___x_240_ = l_Lean_Syntax_node7(v___x_234_, v___x_236_, v___x_239_, v___x_239_, v___x_239_, v___x_239_, v___x_239_, v___x_239_, v___x_239_);
v___x_241_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__10));
v___x_242_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__11));
v___x_243_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__14));
v___x_244_ = l_Lean_Syntax_node1(v___x_234_, v___x_243_, v___x_239_);
v___x_245_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_245_, 0, v___x_234_);
lean_ctor_set(v___x_245_, 1, v___x_241_);
v___x_246_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__16));
v___x_247_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__18));
v___x_248_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__19));
v___x_249_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_249_, 0, v___x_234_);
lean_ctor_set(v___x_249_, 1, v___x_248_);
v___x_250_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__21));
v___x_251_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__23, &lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__23_once, _init_lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__23);
v___x_252_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__24));
lean_inc_n(v_currMacroScope_232_, 7);
lean_inc_n(v_quotContext_231_, 7);
v___x_253_ = l_Lean_addMacroScope(v_quotContext_231_, v___x_252_, v_currMacroScope_232_);
v___x_254_ = lean_box(0);
v___x_255_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__28));
v___x_256_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_256_, 0, v___x_234_);
lean_ctor_set(v___x_256_, 1, v___x_251_);
lean_ctor_set(v___x_256_, 2, v___x_253_);
lean_ctor_set(v___x_256_, 3, v___x_255_);
v___x_257_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__30));
v___x_258_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__32));
v___x_259_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__33));
v___x_260_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_260_, 0, v___x_234_);
lean_ctor_set(v___x_260_, 1, v___x_259_);
v___x_261_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__35));
v___x_262_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__37, &lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__37_once, _init_lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__37);
v___x_263_ = lean_box(0);
v___x_264_ = l_Lean_addMacroScope(v_quotContext_231_, v___x_263_, v_currMacroScope_232_);
v___x_265_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__62));
v___x_266_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_266_, 0, v___x_234_);
lean_ctor_set(v___x_266_, 1, v___x_262_);
lean_ctor_set(v___x_266_, 2, v___x_264_);
lean_ctor_set(v___x_266_, 3, v___x_265_);
v___x_267_ = l_Lean_Syntax_node1(v___x_234_, v___x_261_, v___x_266_);
v___x_268_ = l_Lean_Syntax_node2(v___x_234_, v___x_258_, v___x_260_, v___x_267_);
v___x_269_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__64));
v___x_270_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__65));
v___x_271_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_271_, 0, v___x_234_);
lean_ctor_set(v___x_271_, 1, v___x_270_);
v___x_272_ = l_Lean_mkCIdent(v_declName_227_);
v___x_273_ = l_Lean_Syntax_node2(v___x_234_, v___x_269_, v___x_271_, v___x_272_);
v___x_274_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__66));
v___x_275_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_275_, 0, v___x_234_);
lean_ctor_set(v___x_275_, 1, v___x_274_);
v___x_276_ = l_Lean_Syntax_node3(v___x_234_, v___x_257_, v___x_268_, v___x_273_, v___x_275_);
v___x_277_ = l_Lean_Syntax_node1(v___x_234_, v___x_237_, v___x_276_);
v___x_278_ = l_Lean_Syntax_node2(v___x_234_, v___x_250_, v___x_256_, v___x_277_);
v___x_279_ = l_Lean_Syntax_node2(v___x_234_, v___x_247_, v___x_249_, v___x_278_);
v___x_280_ = l_Lean_Syntax_node2(v___x_234_, v___x_246_, v___x_239_, v___x_279_);
v___x_281_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__68));
v___x_282_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__69));
v___x_283_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_283_, 0, v___x_234_);
lean_ctor_set(v___x_283_, 1, v___x_282_);
v___x_284_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__71));
v___x_285_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__72));
v___x_286_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_286_, 0, v___x_234_);
lean_ctor_set(v___x_286_, 1, v___x_285_);
v___x_287_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__73));
v___x_288_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__74));
v___x_289_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_289_, 0, v___x_234_);
lean_ctor_set(v___x_289_, 1, v___x_287_);
v___x_290_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__76));
v___x_291_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__78, &lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__78_once, _init_lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__78);
v___x_292_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__79));
v___x_293_ = l_Lean_addMacroScope(v_quotContext_231_, v___x_292_, v_currMacroScope_232_);
v___x_294_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_294_, 0, v___x_234_);
lean_ctor_set(v___x_294_, 1, v___x_291_);
lean_ctor_set(v___x_294_, 2, v___x_293_);
lean_ctor_set(v___x_294_, 3, v___x_254_);
v___x_295_ = l_Lean_Syntax_node1(v___x_234_, v___x_237_, v___x_294_);
v___x_296_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__80));
v___x_297_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_297_, 0, v___x_234_);
lean_ctor_set(v___x_297_, 1, v___x_296_);
v___x_298_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__82, &lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__82_once, _init_lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__82);
v___x_299_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__84));
v___x_300_ = l_Lean_addMacroScope(v_quotContext_231_, v___x_299_, v_currMacroScope_232_);
v___x_301_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_301_, 0, v___x_234_);
lean_ctor_set(v___x_301_, 1, v___x_298_);
lean_ctor_set(v___x_301_, 2, v___x_300_);
lean_ctor_set(v___x_301_, 3, v___x_254_);
lean_inc_ref_n(v___x_297_, 3);
lean_inc_n(v___x_295_, 3);
v___x_302_ = l_Lean_Syntax_node4(v___x_234_, v___x_290_, v___x_295_, v___x_239_, v___x_297_, v___x_301_);
lean_inc_ref_n(v___x_289_, 3);
v___x_303_ = l_Lean_Syntax_node2(v___x_234_, v___x_288_, v___x_289_, v___x_302_);
v___x_304_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__85));
v___x_305_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_305_, 0, v___x_234_);
lean_ctor_set(v___x_305_, 1, v___x_304_);
v___x_306_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__87, &lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__87_once, _init_lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__87);
v___x_307_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__89));
v___x_308_ = l_Lean_addMacroScope(v_quotContext_231_, v___x_307_, v_currMacroScope_232_);
v___x_309_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_309_, 0, v___x_234_);
lean_ctor_set(v___x_309_, 1, v___x_306_);
lean_ctor_set(v___x_309_, 2, v___x_308_);
lean_ctor_set(v___x_309_, 3, v___x_254_);
v___x_310_ = l_Lean_Syntax_node4(v___x_234_, v___x_290_, v___x_295_, v___x_239_, v___x_297_, v___x_309_);
v___x_311_ = l_Lean_Syntax_node2(v___x_234_, v___x_288_, v___x_289_, v___x_310_);
v___x_312_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__91, &lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__91_once, _init_lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__91);
v___x_313_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__93));
v___x_314_ = l_Lean_addMacroScope(v_quotContext_231_, v___x_313_, v_currMacroScope_232_);
v___x_315_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_315_, 0, v___x_234_);
lean_ctor_set(v___x_315_, 1, v___x_312_);
lean_ctor_set(v___x_315_, 2, v___x_314_);
lean_ctor_set(v___x_315_, 3, v___x_254_);
v___x_316_ = l_Lean_Syntax_node4(v___x_234_, v___x_290_, v___x_295_, v___x_239_, v___x_297_, v___x_315_);
v___x_317_ = l_Lean_Syntax_node2(v___x_234_, v___x_288_, v___x_289_, v___x_316_);
v___x_318_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__95, &lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__95_once, _init_lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__95);
v___x_319_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__97));
v___x_320_ = l_Lean_addMacroScope(v_quotContext_231_, v___x_319_, v_currMacroScope_232_);
v___x_321_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_321_, 0, v___x_234_);
lean_ctor_set(v___x_321_, 1, v___x_318_);
lean_ctor_set(v___x_321_, 2, v___x_320_);
lean_ctor_set(v___x_321_, 3, v___x_254_);
v___x_322_ = l_Lean_Syntax_node4(v___x_234_, v___x_290_, v___x_295_, v___x_239_, v___x_297_, v___x_321_);
v___x_323_ = l_Lean_Syntax_node2(v___x_234_, v___x_288_, v___x_289_, v___x_322_);
lean_inc_ref_n(v___x_305_, 2);
v___x_324_ = l_Lean_Syntax_node7(v___x_234_, v___x_237_, v___x_303_, v___x_305_, v___x_311_, v___x_305_, v___x_317_, v___x_305_, v___x_323_);
v___x_325_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__98));
v___x_326_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_326_, 0, v___x_234_);
lean_ctor_set(v___x_326_, 1, v___x_325_);
v___x_327_ = l_Lean_Syntax_node3(v___x_234_, v___x_284_, v___x_286_, v___x_324_, v___x_326_);
v___x_328_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__101));
v___x_329_ = l_Lean_Syntax_node2(v___x_234_, v___x_328_, v___x_239_, v___x_239_);
v___x_330_ = l_Lean_Syntax_node4(v___x_234_, v___x_281_, v___x_283_, v___x_327_, v___x_329_, v___x_239_);
v___x_331_ = l_Lean_Syntax_node6(v___x_234_, v___x_242_, v___x_244_, v___x_245_, v___x_239_, v___x_239_, v___x_280_, v___x_330_);
v___x_332_ = l_Lean_Syntax_node2(v___x_234_, v___x_235_, v___x_240_, v___x_331_);
v___x_333_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_333_, 0, v___x_332_);
return v___x_333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___boxed(lean_object* v_declName_334_, lean_object* v_a_335_, lean_object* v_a_336_){
_start:
{
lean_object* v_res_337_; 
v_res_337_ = lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg(v_declName_334_, v_a_335_);
lean_dec_ref(v_a_335_);
return v_res_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance(lean_object* v_declName_338_, lean_object* v_a_339_, lean_object* v_a_340_, lean_object* v_a_341_, lean_object* v_a_342_, lean_object* v_a_343_, lean_object* v_a_344_){
_start:
{
lean_object* v___x_346_; 
v___x_346_ = lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg(v_declName_338_, v_a_343_);
return v___x_346_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___boxed(lean_object* v_declName_347_, lean_object* v_a_348_, lean_object* v_a_349_, lean_object* v_a_350_, lean_object* v_a_351_, lean_object* v_a_352_, lean_object* v_a_353_, lean_object* v_a_354_){
_start:
{
lean_object* v_res_355_; 
v_res_355_ = lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance(v_declName_347_, v_a_348_, v_a_349_, v_a_350_, v_a_351_, v_a_352_, v_a_353_);
lean_dec(v_a_353_);
lean_dec_ref(v_a_352_);
lean_dec(v_a_351_);
lean_dec_ref(v_a_350_);
lean_dec(v_a_349_);
lean_dec_ref(v_a_348_);
return v_res_355_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_isInductive___at___00Lean_Elab_mkSelectInsertParamsInstanceHandler_spec__1___redArg(lean_object* v_declName_356_, lean_object* v___y_357_){
_start:
{
lean_object* v___x_359_; lean_object* v_env_360_; uint8_t v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; 
v___x_359_ = lean_st_ref_get(v___y_357_);
v_env_360_ = lean_ctor_get(v___x_359_, 0);
lean_inc_ref(v_env_360_);
lean_dec(v___x_359_);
v___x_361_ = l_Lean_isInductiveCore(v_env_360_, v_declName_356_);
v___x_362_ = lean_box(v___x_361_);
v___x_363_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_363_, 0, v___x_362_);
return v___x_363_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_isInductive___at___00Lean_Elab_mkSelectInsertParamsInstanceHandler_spec__1___redArg___boxed(lean_object* v_declName_364_, lean_object* v___y_365_, lean_object* v___y_366_){
_start:
{
lean_object* v_res_367_; 
v_res_367_ = lp_mathlib_Lean_isInductive___at___00Lean_Elab_mkSelectInsertParamsInstanceHandler_spec__1___redArg(v_declName_364_, v___y_365_);
lean_dec(v___y_365_);
return v_res_367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_isInductive___at___00Lean_Elab_mkSelectInsertParamsInstanceHandler_spec__1(lean_object* v_declName_368_, lean_object* v___y_369_, lean_object* v___y_370_){
_start:
{
lean_object* v___x_372_; 
v___x_372_ = lp_mathlib_Lean_isInductive___at___00Lean_Elab_mkSelectInsertParamsInstanceHandler_spec__1___redArg(v_declName_368_, v___y_370_);
return v___x_372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_isInductive___at___00Lean_Elab_mkSelectInsertParamsInstanceHandler_spec__1___boxed(lean_object* v_declName_373_, lean_object* v___y_374_, lean_object* v___y_375_, lean_object* v___y_376_){
_start:
{
lean_object* v_res_377_; 
v_res_377_ = lp_mathlib_Lean_isInductive___at___00Lean_Elab_mkSelectInsertParamsInstanceHandler_spec__1(v_declName_373_, v___y_374_, v___y_375_);
lean_dec(v___y_375_);
lean_dec_ref(v___y_374_);
return v_res_377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_mkSelectInsertParamsInstanceHandler___lam__0(uint8_t v_____do__lift_378_, lean_object* v___y_379_, lean_object* v___y_380_){
_start:
{
if (v_____do__lift_378_ == 0)
{
uint8_t v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; 
v___x_382_ = 1;
v___x_383_ = lean_box(v___x_382_);
v___x_384_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_384_, 0, v___x_383_);
return v___x_384_;
}
else
{
uint8_t v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; 
v___x_385_ = 0;
v___x_386_ = lean_box(v___x_385_);
v___x_387_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_387_, 0, v___x_386_);
return v___x_387_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_mkSelectInsertParamsInstanceHandler___lam__0___boxed(lean_object* v_____do__lift_388_, lean_object* v___y_389_, lean_object* v___y_390_, lean_object* v___y_391_){
_start:
{
uint8_t v_____do__lift_1860__boxed_392_; lean_object* v_res_393_; 
v_____do__lift_1860__boxed_392_ = lean_unbox(v_____do__lift_388_);
v_res_393_ = lp_mathlib_Lean_Elab_mkSelectInsertParamsInstanceHandler___lam__0(v_____do__lift_1860__boxed_392_, v___y_389_, v___y_390_);
lean_dec(v___y_390_);
lean_dec_ref(v___y_389_);
return v_res_393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_Elab_mkSelectInsertParamsInstanceHandler_spec__2(lean_object* v_as_394_, size_t v_i_395_, size_t v_stop_396_, lean_object* v___y_397_, lean_object* v___y_398_){
_start:
{
uint8_t v___x_400_; 
v___x_400_ = lean_usize_dec_eq(v_i_395_, v_stop_396_);
if (v___x_400_ == 0)
{
uint8_t v___x_401_; uint8_t v_a_403_; lean_object* v___x_409_; lean_object* v___x_410_; 
v___x_401_ = 1;
v___x_409_ = lean_array_uget_borrowed(v_as_394_, v_i_395_);
lean_inc(v___x_409_);
v___x_410_ = lp_mathlib_Lean_isInductive___at___00Lean_Elab_mkSelectInsertParamsInstanceHandler_spec__1___redArg(v___x_409_, v___y_398_);
if (lean_obj_tag(v___x_410_) == 0)
{
lean_object* v_a_411_; lean_object* v___x_413_; uint8_t v_isShared_414_; uint8_t v_isSharedCheck_420_; 
v_a_411_ = lean_ctor_get(v___x_410_, 0);
v_isSharedCheck_420_ = !lean_is_exclusive(v___x_410_);
if (v_isSharedCheck_420_ == 0)
{
v___x_413_ = v___x_410_;
v_isShared_414_ = v_isSharedCheck_420_;
goto v_resetjp_412_;
}
else
{
lean_inc(v_a_411_);
lean_dec(v___x_410_);
v___x_413_ = lean_box(0);
v_isShared_414_ = v_isSharedCheck_420_;
goto v_resetjp_412_;
}
v_resetjp_412_:
{
uint8_t v___x_415_; 
v___x_415_ = lean_unbox(v_a_411_);
lean_dec(v_a_411_);
if (v___x_415_ == 0)
{
lean_object* v___x_416_; lean_object* v___x_418_; 
v___x_416_ = lean_box(v___x_401_);
if (v_isShared_414_ == 0)
{
lean_ctor_set(v___x_413_, 0, v___x_416_);
v___x_418_ = v___x_413_;
goto v_reusejp_417_;
}
else
{
lean_object* v_reuseFailAlloc_419_; 
v_reuseFailAlloc_419_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_419_, 0, v___x_416_);
v___x_418_ = v_reuseFailAlloc_419_;
goto v_reusejp_417_;
}
v_reusejp_417_:
{
return v___x_418_;
}
}
else
{
lean_del_object(v___x_413_);
v_a_403_ = v___x_400_;
goto v___jp_402_;
}
}
}
else
{
if (lean_obj_tag(v___x_410_) == 0)
{
lean_object* v_a_421_; uint8_t v___x_422_; 
v_a_421_ = lean_ctor_get(v___x_410_, 0);
lean_inc(v_a_421_);
lean_dec_ref_known(v___x_410_, 1);
v___x_422_ = lean_unbox(v_a_421_);
lean_dec(v_a_421_);
v_a_403_ = v___x_422_;
goto v___jp_402_;
}
else
{
return v___x_410_;
}
}
v___jp_402_:
{
if (v_a_403_ == 0)
{
size_t v___x_404_; size_t v___x_405_; 
v___x_404_ = ((size_t)1ULL);
v___x_405_ = lean_usize_add(v_i_395_, v___x_404_);
v_i_395_ = v___x_405_;
goto _start;
}
else
{
lean_object* v___x_407_; lean_object* v___x_408_; 
v___x_407_ = lean_box(v___x_401_);
v___x_408_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_408_, 0, v___x_407_);
return v___x_408_;
}
}
}
else
{
uint8_t v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; 
v___x_423_ = 0;
v___x_424_ = lean_box(v___x_423_);
v___x_425_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_425_, 0, v___x_424_);
return v___x_425_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_Elab_mkSelectInsertParamsInstanceHandler_spec__2___boxed(lean_object* v_as_426_, lean_object* v_i_427_, lean_object* v_stop_428_, lean_object* v___y_429_, lean_object* v___y_430_, lean_object* v___y_431_){
_start:
{
size_t v_i_boxed_432_; size_t v_stop_boxed_433_; lean_object* v_res_434_; 
v_i_boxed_432_ = lean_unbox_usize(v_i_427_);
lean_dec(v_i_427_);
v_stop_boxed_433_ = lean_unbox_usize(v_stop_428_);
lean_dec(v_stop_428_);
v_res_434_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_Elab_mkSelectInsertParamsInstanceHandler_spec__2(v_as_426_, v_i_boxed_432_, v_stop_boxed_433_, v___y_429_, v___y_430_);
lean_dec(v___y_430_);
lean_dec_ref(v___y_429_);
lean_dec_ref(v_as_426_);
return v_res_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_mkSelectInsertParamsInstanceHandler_spec__0(lean_object* v_as_435_, size_t v_sz_436_, size_t v_i_437_, lean_object* v_b_438_, lean_object* v___y_439_, lean_object* v___y_440_){
_start:
{
uint8_t v___x_442_; 
v___x_442_ = lean_usize_dec_lt(v_i_437_, v_sz_436_);
if (v___x_442_ == 0)
{
lean_object* v___x_443_; 
v___x_443_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_443_, 0, v_b_438_);
return v___x_443_;
}
else
{
lean_object* v_a_444_; lean_object* v___x_445_; lean_object* v___x_446_; 
v_a_444_ = lean_array_uget_borrowed(v_as_435_, v_i_437_);
lean_inc(v_a_444_);
v___x_445_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___boxed), 8, 1);
lean_closure_set(v___x_445_, 0, v_a_444_);
v___x_446_ = l_Lean_Elab_Command_liftTermElabM___redArg(v___x_445_, v___y_439_, v___y_440_);
if (lean_obj_tag(v___x_446_) == 0)
{
lean_object* v_a_447_; lean_object* v___x_448_; 
v_a_447_ = lean_ctor_get(v___x_446_, 0);
lean_inc(v_a_447_);
lean_dec_ref_known(v___x_446_, 1);
v___x_448_ = l_Lean_Elab_Command_elabCommand(v_a_447_, v___y_439_, v___y_440_);
if (lean_obj_tag(v___x_448_) == 0)
{
lean_object* v___x_449_; size_t v___x_450_; size_t v___x_451_; 
lean_dec_ref_known(v___x_448_, 1);
v___x_449_ = lean_box(0);
v___x_450_ = ((size_t)1ULL);
v___x_451_ = lean_usize_add(v_i_437_, v___x_450_);
v_i_437_ = v___x_451_;
v_b_438_ = v___x_449_;
goto _start;
}
else
{
return v___x_448_;
}
}
else
{
lean_object* v_a_453_; lean_object* v___x_455_; uint8_t v_isShared_456_; uint8_t v_isSharedCheck_460_; 
v_a_453_ = lean_ctor_get(v___x_446_, 0);
v_isSharedCheck_460_ = !lean_is_exclusive(v___x_446_);
if (v_isSharedCheck_460_ == 0)
{
v___x_455_ = v___x_446_;
v_isShared_456_ = v_isSharedCheck_460_;
goto v_resetjp_454_;
}
else
{
lean_inc(v_a_453_);
lean_dec(v___x_446_);
v___x_455_ = lean_box(0);
v_isShared_456_ = v_isSharedCheck_460_;
goto v_resetjp_454_;
}
v_resetjp_454_:
{
lean_object* v___x_458_; 
if (v_isShared_456_ == 0)
{
v___x_458_ = v___x_455_;
goto v_reusejp_457_;
}
else
{
lean_object* v_reuseFailAlloc_459_; 
v_reuseFailAlloc_459_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_459_, 0, v_a_453_);
v___x_458_ = v_reuseFailAlloc_459_;
goto v_reusejp_457_;
}
v_reusejp_457_:
{
return v___x_458_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_mkSelectInsertParamsInstanceHandler_spec__0___boxed(lean_object* v_as_461_, lean_object* v_sz_462_, lean_object* v_i_463_, lean_object* v_b_464_, lean_object* v___y_465_, lean_object* v___y_466_, lean_object* v___y_467_){
_start:
{
size_t v_sz_boxed_468_; size_t v_i_boxed_469_; lean_object* v_res_470_; 
v_sz_boxed_468_ = lean_unbox_usize(v_sz_462_);
lean_dec(v_sz_462_);
v_i_boxed_469_ = lean_unbox_usize(v_i_463_);
lean_dec(v_i_463_);
v_res_470_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_mkSelectInsertParamsInstanceHandler_spec__0(v_as_461_, v_sz_boxed_468_, v_i_boxed_469_, v_b_464_, v___y_465_, v___y_466_);
lean_dec(v___y_466_);
lean_dec_ref(v___y_465_);
lean_dec_ref(v_as_461_);
return v_res_470_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_mkSelectInsertParamsInstanceHandler(lean_object* v_declNames_471_, lean_object* v_a_472_, lean_object* v_a_473_){
_start:
{
lean_object* v___y_499_; lean_object* v___x_502_; lean_object* v___x_503_; uint8_t v___x_504_; 
v___x_502_ = lean_unsigned_to_nat(0u);
v___x_503_ = lean_array_get_size(v_declNames_471_);
v___x_504_ = lean_nat_dec_lt(v___x_502_, v___x_503_);
if (v___x_504_ == 0)
{
lean_object* v___x_505_; 
v___x_505_ = lp_mathlib_Lean_Elab_mkSelectInsertParamsInstanceHandler___lam__0(v___x_504_, v_a_472_, v_a_473_);
v___y_499_ = v___x_505_;
goto v___jp_498_;
}
else
{
if (v___x_504_ == 0)
{
goto v___jp_475_;
}
else
{
size_t v___x_506_; size_t v___x_507_; lean_object* v___x_508_; 
v___x_506_ = ((size_t)0ULL);
v___x_507_ = lean_usize_of_nat(v___x_503_);
v___x_508_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_Elab_mkSelectInsertParamsInstanceHandler_spec__2(v_declNames_471_, v___x_506_, v___x_507_, v_a_472_, v_a_473_);
if (lean_obj_tag(v___x_508_) == 0)
{
lean_object* v_a_509_; uint8_t v___x_510_; lean_object* v___x_511_; 
v_a_509_ = lean_ctor_get(v___x_508_, 0);
lean_inc(v_a_509_);
lean_dec_ref_known(v___x_508_, 1);
v___x_510_ = lean_unbox(v_a_509_);
lean_dec(v_a_509_);
v___x_511_ = lp_mathlib_Lean_Elab_mkSelectInsertParamsInstanceHandler___lam__0(v___x_510_, v_a_472_, v_a_473_);
v___y_499_ = v___x_511_;
goto v___jp_498_;
}
else
{
v___y_499_ = v___x_508_;
goto v___jp_498_;
}
}
}
v___jp_475_:
{
lean_object* v___x_476_; size_t v_sz_477_; size_t v___x_478_; lean_object* v___x_479_; 
v___x_476_ = lean_box(0);
v_sz_477_ = lean_array_size(v_declNames_471_);
v___x_478_ = ((size_t)0ULL);
v___x_479_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Elab_mkSelectInsertParamsInstanceHandler_spec__0(v_declNames_471_, v_sz_477_, v___x_478_, v___x_476_, v_a_472_, v_a_473_);
if (lean_obj_tag(v___x_479_) == 0)
{
lean_object* v___x_481_; uint8_t v_isShared_482_; uint8_t v_isSharedCheck_488_; 
v_isSharedCheck_488_ = !lean_is_exclusive(v___x_479_);
if (v_isSharedCheck_488_ == 0)
{
lean_object* v_unused_489_; 
v_unused_489_ = lean_ctor_get(v___x_479_, 0);
lean_dec(v_unused_489_);
v___x_481_ = v___x_479_;
v_isShared_482_ = v_isSharedCheck_488_;
goto v_resetjp_480_;
}
else
{
lean_dec(v___x_479_);
v___x_481_ = lean_box(0);
v_isShared_482_ = v_isSharedCheck_488_;
goto v_resetjp_480_;
}
v_resetjp_480_:
{
uint8_t v___x_483_; lean_object* v___x_484_; lean_object* v___x_486_; 
v___x_483_ = 1;
v___x_484_ = lean_box(v___x_483_);
if (v_isShared_482_ == 0)
{
lean_ctor_set(v___x_481_, 0, v___x_484_);
v___x_486_ = v___x_481_;
goto v_reusejp_485_;
}
else
{
lean_object* v_reuseFailAlloc_487_; 
v_reuseFailAlloc_487_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_487_, 0, v___x_484_);
v___x_486_ = v_reuseFailAlloc_487_;
goto v_reusejp_485_;
}
v_reusejp_485_:
{
return v___x_486_;
}
}
}
else
{
lean_object* v_a_490_; lean_object* v___x_492_; uint8_t v_isShared_493_; uint8_t v_isSharedCheck_497_; 
v_a_490_ = lean_ctor_get(v___x_479_, 0);
v_isSharedCheck_497_ = !lean_is_exclusive(v___x_479_);
if (v_isSharedCheck_497_ == 0)
{
v___x_492_ = v___x_479_;
v_isShared_493_ = v_isSharedCheck_497_;
goto v_resetjp_491_;
}
else
{
lean_inc(v_a_490_);
lean_dec(v___x_479_);
v___x_492_ = lean_box(0);
v_isShared_493_ = v_isSharedCheck_497_;
goto v_resetjp_491_;
}
v_resetjp_491_:
{
lean_object* v___x_495_; 
if (v_isShared_493_ == 0)
{
v___x_495_ = v___x_492_;
goto v_reusejp_494_;
}
else
{
lean_object* v_reuseFailAlloc_496_; 
v_reuseFailAlloc_496_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_496_, 0, v_a_490_);
v___x_495_ = v_reuseFailAlloc_496_;
goto v_reusejp_494_;
}
v_reusejp_494_:
{
return v___x_495_;
}
}
}
}
v___jp_498_:
{
if (lean_obj_tag(v___y_499_) == 0)
{
lean_object* v_a_500_; uint8_t v___x_501_; 
v_a_500_ = lean_ctor_get(v___y_499_, 0);
v___x_501_ = lean_unbox(v_a_500_);
if (v___x_501_ == 0)
{
return v___y_499_;
}
else
{
lean_dec_ref_known(v___y_499_, 1);
goto v___jp_475_;
}
}
else
{
return v___y_499_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_mkSelectInsertParamsInstanceHandler___boxed(lean_object* v_declNames_512_, lean_object* v_a_513_, lean_object* v_a_514_, lean_object* v_a_515_){
_start:
{
lean_object* v_res_516_; 
v_res_516_ = lp_mathlib_Lean_Elab_mkSelectInsertParamsInstanceHandler(v_declNames_512_, v_a_513_, v_a_514_);
lean_dec(v_a_514_);
lean_dec_ref(v_a_513_);
lean_dec_ref(v_declNames_512_);
return v_res_516_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_initFn_00___x40_Mathlib_Tactic_Widget_SelectInsertParamsClass_1629491121____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; 
v___x_519_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_mkSelectInsertParamsInstance___redArg___closed__24));
v___x_520_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_initFn___closed__0_00___x40_Mathlib_Tactic_Widget_SelectInsertParamsClass_1629491121____hygCtx___hyg_2_));
v___x_521_ = l_Lean_Elab_registerDerivingHandler(v___x_519_, v___x_520_);
return v___x_521_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_initFn_00___x40_Mathlib_Tactic_Widget_SelectInsertParamsClass_1629491121____hygCtx___hyg_2____boxed(lean_object* v_a_522_){
_start:
{
lean_object* v_res_523_; 
v_res_523_ = lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_initFn_00___x40_Mathlib_Tactic_Widget_SelectInsertParamsClass_1629491121____hygCtx___hyg_2_();
return v_res_523_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Widget_InteractiveGoal(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Widget_SelectInsertParamsClass(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Widget_InteractiveGoal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Widget_InteractiveGoal(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Deriving_Basic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Widget_SelectInsertParamsClass(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Widget_InteractiveGoal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Deriving_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Widget_SelectInsertParamsClass_0__Lean_Elab_initFn_00___x40_Mathlib_Tactic_Widget_SelectInsertParamsClass_1629491121____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Widget_InteractiveGoal(uint8_t builtin);
lean_object* initialize_Lean_Elab_Deriving_Basic(uint8_t builtin);
lean_object* initialize_Lean_Widget_InteractiveGoal(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Widget_SelectInsertParamsClass(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Widget_InteractiveGoal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Deriving_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Widget_InteractiveGoal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Widget_SelectInsertParamsClass(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Widget_SelectInsertParamsClass(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Widget_SelectInsertParamsClass(builtin);
}
#ifdef __cplusplus
}
#endif
