// Lean compiler output
// Module: Mathlib.Tactic.Contrapose
// Imports: public import Init public meta import Init public import Mathlib.Tactic.Push
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
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_Meta_throwTacticEx___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Tactic_optConfig;
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_Push_push(uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_whnf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_mkAppB(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
lean_object* l_Lean_Meta_isProp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_MVarId_getType_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasLooseBVars(lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__0_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "contrapose"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__0_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__0_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__1_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "negate_iff"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__1_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__1_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__2_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__0_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(127, 226, 84, 186, 108, 171, 56, 102)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__2_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__2_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__1_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(187, 204, 175, 124, 6, 6, 85, 247)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__2_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__2_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__3_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 58, .m_capacity = 58, .m_length = 51, .m_data = "contrapose a goal `a ↔ b` into the goal `¬ a ↔ ¬ b`"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__3_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__3_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__4_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__3_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__4_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__4_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__5_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__5_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__5_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__6_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__6_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__6_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__7_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Contrapose"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__7_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__7_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__8_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__5_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__8_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__8_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__6_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__8_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__8_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__7_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(58, 14, 24, 246, 34, 32, 81, 240)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__8_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__8_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__0_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(106, 134, 233, 192, 83, 115, 23, 212)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__8_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__8_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__1_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(34, 74, 154, 64, 251, 93, 173, 98)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__8_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__8_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_negate__iff;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__5_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__6_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__0_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__7_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(58, 14, 24, 246, 34, 32, 81, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__0_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__0_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(106, 134, 233, 192, 83, 115, 23, 212)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__1_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__0_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__4_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__6_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__9_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__13_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__12_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = " with "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__18_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__16_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__21_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__22_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__0_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__23_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__24_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__24_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__3_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__6_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(117, 253, 122, 28, 77, 248, 149, 120)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__6_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__6_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__8_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__6_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__8_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "revert"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__12_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__12_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__12_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__12_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__6_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__12_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(45, 160, 70, 247, 76, 165, 126, 200)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ";"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__13_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__14;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "intro"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__16_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__6_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__16_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(41, 145, 9, 18, 75, 146, 159, 78)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__17_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "the goal `"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 44, .m_capacity = 44, .m_length = 39, .m_data = "` is not of the form `_ → _` or `_ ↔ _`"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Not"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(185, 11, 203, 55, 27, 192, 137, 230)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 11, .m_data = "contrapose₁"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 11, .m_data = "contrapose₃"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 11, .m_data = "contrapose₂"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 11, .m_data = "contrapose₄"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "conclusion `"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__7;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "` is not a proposition"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__9;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "hypothesis `"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__10_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__11;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Option_getM___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__1_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Option_getM___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__1_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_getM___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_getM___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Iff"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 107, .m_capacity = 107, .m_length = 104, .m_data = "contraposing `↔` relations has been disabled.\nTo enable it, use `set_option contrapose.negate_iff true`."};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__3;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__4;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 15, .m_data = "contrapose_iff₁"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 15, .m_data = "contrapose_iff₃"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 15, .m_data = "contrapose_iff₂"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 15, .m_data = "contrapose_iff₄"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "` is a dependent arrow"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__9_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__10;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_getM___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_getM___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "contrapose!"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__5_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__6_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__7_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(58, 14, 24, 246, 34, 32, 81, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(245, 102, 23, 42, 225, 48, 238, 13)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__3;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__5_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__2_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__6_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__7_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(222, 117, 67, 148, 254, 69, 245, 138)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(167, 58, 208, 130, 39, 252, 94, 206)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__5_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(42, 157, 226, 254, 52, 143, 129, 29)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__6_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__6_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(127, 207, 96, 173, 75, 55, 39, 22)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__7_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(30, 3, 188, 49, 78, 119, 160, 7)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "tacticTry_push_neg_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__9_value),LEAN_SCALAR_PTR_LITERAL(238, 155, 11, 230, 238, 160, 213, 173)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "try_push_neg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__11_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__12_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__13;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__14;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg__;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules____private__Mathlib__Tactic__Contrapose__0__Mathlib__Tactic__Contrapose__tacticTry__push__neg____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules____private__Mathlib__Tactic__Contrapose__0__Mathlib__Tactic__Contrapose__tacticTry__push__neg____1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules____private__Mathlib__Tactic__Contrapose__0__Mathlib__Tactic__Contrapose__tacticTry__push__neg____1___closed__0_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules____private__Mathlib__Tactic__Contrapose__0__Mathlib__Tactic__Contrapose__tacticTry__push__neg____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules____private__Mathlib__Tactic__Contrapose__0__Mathlib__Tactic__Contrapose__tacticTry__push__neg____1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules____private__Mathlib__Tactic__Contrapose__0__Mathlib__Tactic__Contrapose__tacticTry__push__neg____1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules____private__Mathlib__Tactic__Contrapose__0__Mathlib__Tactic__Contrapose__tacticTry__push__neg____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules____private__Mathlib__Tactic__Contrapose__0__Mathlib__Tactic__Contrapose__tacticTry__push__neg____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose_x21__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose_x21__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose_x21__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose_x21__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose_x21__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose_x21__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose_x21__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose_x21__1___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__6_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose_x21__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose_x21__1___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose_x21__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose_x21__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose_x21__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose_x21__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose_x21__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__spec__0(lean_object* v_name_1_, lean_object* v_decl_2_, lean_object* v_ref_3_){
_start:
{
lean_object* v_defValue_5_; lean_object* v_descr_6_; lean_object* v_deprecation_x3f_7_; lean_object* v___x_8_; uint8_t v___x_9_; lean_object* v___x_10_; lean_object* v___x_11_; 
v_defValue_5_ = lean_ctor_get(v_decl_2_, 0);
v_descr_6_ = lean_ctor_get(v_decl_2_, 1);
v_deprecation_x3f_7_ = lean_ctor_get(v_decl_2_, 2);
v___x_8_ = lean_alloc_ctor(1, 0, 1);
v___x_9_ = lean_unbox(v_defValue_5_);
lean_ctor_set_uint8(v___x_8_, 0, v___x_9_);
lean_inc(v_deprecation_x3f_7_);
lean_inc_ref(v_descr_6_);
lean_inc_n(v_name_1_, 2);
v___x_10_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_10_, 0, v_name_1_);
lean_ctor_set(v___x_10_, 1, v_ref_3_);
lean_ctor_set(v___x_10_, 2, v___x_8_);
lean_ctor_set(v___x_10_, 3, v_descr_6_);
lean_ctor_set(v___x_10_, 4, v_deprecation_x3f_7_);
v___x_11_ = lean_register_option(v_name_1_, v___x_10_);
if (lean_obj_tag(v___x_11_) == 0)
{
lean_object* v___x_13_; uint8_t v_isShared_14_; uint8_t v_isSharedCheck_19_; 
v_isSharedCheck_19_ = !lean_is_exclusive(v___x_11_);
if (v_isSharedCheck_19_ == 0)
{
lean_object* v_unused_20_; 
v_unused_20_ = lean_ctor_get(v___x_11_, 0);
lean_dec(v_unused_20_);
v___x_13_ = v___x_11_;
v_isShared_14_ = v_isSharedCheck_19_;
goto v_resetjp_12_;
}
else
{
lean_dec(v___x_11_);
v___x_13_ = lean_box(0);
v_isShared_14_ = v_isSharedCheck_19_;
goto v_resetjp_12_;
}
v_resetjp_12_:
{
lean_object* v___x_15_; lean_object* v___x_17_; 
lean_inc(v_defValue_5_);
v___x_15_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_15_, 0, v_name_1_);
lean_ctor_set(v___x_15_, 1, v_defValue_5_);
if (v_isShared_14_ == 0)
{
lean_ctor_set(v___x_13_, 0, v___x_15_);
v___x_17_ = v___x_13_;
goto v_reusejp_16_;
}
else
{
lean_object* v_reuseFailAlloc_18_; 
v_reuseFailAlloc_18_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_18_, 0, v___x_15_);
v___x_17_ = v_reuseFailAlloc_18_;
goto v_reusejp_16_;
}
v_reusejp_16_:
{
return v___x_17_;
}
}
}
else
{
lean_object* v_a_21_; lean_object* v___x_23_; uint8_t v_isShared_24_; uint8_t v_isSharedCheck_28_; 
lean_dec(v_name_1_);
v_a_21_ = lean_ctor_get(v___x_11_, 0);
v_isSharedCheck_28_ = !lean_is_exclusive(v___x_11_);
if (v_isSharedCheck_28_ == 0)
{
v___x_23_ = v___x_11_;
v_isShared_24_ = v_isSharedCheck_28_;
goto v_resetjp_22_;
}
else
{
lean_inc(v_a_21_);
lean_dec(v___x_11_);
v___x_23_ = lean_box(0);
v_isShared_24_ = v_isSharedCheck_28_;
goto v_resetjp_22_;
}
v_resetjp_22_:
{
lean_object* v___x_26_; 
if (v_isShared_24_ == 0)
{
v___x_26_ = v___x_23_;
goto v_reusejp_25_;
}
else
{
lean_object* v_reuseFailAlloc_27_; 
v_reuseFailAlloc_27_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_27_, 0, v_a_21_);
v___x_26_ = v_reuseFailAlloc_27_;
goto v_reusejp_25_;
}
v_reusejp_25_:
{
return v___x_26_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_29_, lean_object* v_decl_30_, lean_object* v_ref_31_, lean_object* v_a_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__spec__0(v_name_29_, v_decl_30_, v_ref_31_);
lean_dec_ref(v_decl_30_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; 
v___x_55_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__2_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4_));
v___x_56_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__4_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4_));
v___x_57_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__8_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4_));
v___x_58_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4__spec__0(v___x_55_, v___x_56_, v___x_57_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4____boxed(lean_object* v_a_59_){
_start:
{
lean_object* v_res_60_; 
v_res_60_ = lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4_();
return v_res_60_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__14(void){
_start:
{
lean_object* v___x_155_; 
v___x_155_ = l_Array_mkArray0(lean_box(0));
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1(lean_object* v_x_163_, lean_object* v_a_164_, lean_object* v_a_165_){
_start:
{
lean_object* v___x_166_; lean_object* v___x_167_; uint8_t v___x_168_; 
v___x_166_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__0_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4_));
v___x_167_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__0));
lean_inc(v_x_163_);
v___x_168_ = l_Lean_Syntax_isOfKind(v_x_163_, v___x_167_);
if (v___x_168_ == 0)
{
lean_object* v___x_169_; lean_object* v___x_170_; 
lean_dec(v_x_163_);
v___x_169_ = lean_box(1);
v___x_170_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_170_, 0, v___x_169_);
lean_ctor_set(v___x_170_, 1, v_a_165_);
return v___x_170_;
}
else
{
lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; uint8_t v___x_174_; 
v___x_171_ = lean_unsigned_to_nat(1u);
v___x_172_ = l_Lean_Syntax_getArg(v_x_163_, v___x_171_);
lean_dec(v_x_163_);
v___x_173_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_172_);
v___x_174_ = l_Lean_Syntax_matchesNull(v___x_172_, v___x_173_);
if (v___x_174_ == 0)
{
lean_object* v___x_175_; lean_object* v___x_176_; 
lean_dec(v___x_172_);
v___x_175_ = lean_box(1);
v___x_176_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_176_, 0, v___x_175_);
lean_ctor_set(v___x_176_, 1, v_a_165_);
return v___x_176_;
}
else
{
lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; uint8_t v___x_180_; 
v___x_177_ = lean_unsigned_to_nat(0u);
v___x_178_ = l_Lean_Syntax_getArg(v___x_172_, v___x_177_);
v___x_179_ = l_Lean_Syntax_getArg(v___x_172_, v___x_171_);
lean_dec(v___x_172_);
lean_inc(v___x_179_);
v___x_180_ = l_Lean_Syntax_matchesNull(v___x_179_, v___x_177_);
if (v___x_180_ == 0)
{
uint8_t v___x_181_; 
lean_inc(v___x_179_);
v___x_181_ = l_Lean_Syntax_matchesNull(v___x_179_, v___x_173_);
if (v___x_181_ == 0)
{
lean_object* v___x_182_; lean_object* v___x_183_; 
lean_dec(v___x_179_);
lean_dec(v___x_178_);
v___x_182_ = lean_box(1);
v___x_183_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_183_, 0, v___x_182_);
lean_ctor_set(v___x_183_, 1, v_a_165_);
return v___x_183_;
}
else
{
lean_object* v_ref_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; 
v_ref_184_ = lean_ctor_get(v_a_164_, 5);
v___x_185_ = l_Lean_Syntax_getArg(v___x_179_, v___x_171_);
lean_dec(v___x_179_);
v___x_186_ = l_Lean_SourceInfo_fromRef(v_ref_184_, v___x_180_);
v___x_187_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__3));
v___x_188_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__4));
lean_inc_n(v___x_186_, 15);
v___x_189_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_189_, 0, v___x_186_);
lean_ctor_set(v___x_189_, 1, v___x_188_);
v___x_190_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__6));
v___x_191_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__8));
v___x_192_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__10));
v___x_193_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__11));
v___x_194_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__12));
v___x_195_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_195_, 0, v___x_186_);
lean_ctor_set(v___x_195_, 1, v___x_193_);
v___x_196_ = l_Lean_Syntax_node1(v___x_186_, v___x_192_, v___x_178_);
v___x_197_ = l_Lean_Syntax_node2(v___x_186_, v___x_194_, v___x_195_, v___x_196_);
v___x_198_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__13));
v___x_199_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_199_, 0, v___x_186_);
lean_ctor_set(v___x_199_, 1, v___x_198_);
v___x_200_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_200_, 0, v___x_186_);
lean_ctor_set(v___x_200_, 1, v___x_166_);
v___x_201_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__14, &lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__14);
v___x_202_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_202_, 0, v___x_186_);
lean_ctor_set(v___x_202_, 1, v___x_192_);
lean_ctor_set(v___x_202_, 2, v___x_201_);
v___x_203_ = l_Lean_Syntax_node2(v___x_186_, v___x_167_, v___x_200_, v___x_202_);
v___x_204_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__15));
v___x_205_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__16));
v___x_206_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_206_, 0, v___x_186_);
lean_ctor_set(v___x_206_, 1, v___x_204_);
v___x_207_ = l_Lean_Syntax_node1(v___x_186_, v___x_192_, v___x_185_);
v___x_208_ = l_Lean_Syntax_node2(v___x_186_, v___x_205_, v___x_206_, v___x_207_);
lean_inc_ref(v___x_199_);
v___x_209_ = l_Lean_Syntax_node5(v___x_186_, v___x_192_, v___x_197_, v___x_199_, v___x_203_, v___x_199_, v___x_208_);
v___x_210_ = l_Lean_Syntax_node1(v___x_186_, v___x_191_, v___x_209_);
v___x_211_ = l_Lean_Syntax_node1(v___x_186_, v___x_190_, v___x_210_);
v___x_212_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__17));
v___x_213_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_213_, 0, v___x_186_);
lean_ctor_set(v___x_213_, 1, v___x_212_);
v___x_214_ = l_Lean_Syntax_node3(v___x_186_, v___x_187_, v___x_189_, v___x_211_, v___x_213_);
v___x_215_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_215_, 0, v___x_214_);
lean_ctor_set(v___x_215_, 1, v_a_165_);
return v___x_215_;
}
}
else
{
lean_object* v_ref_216_; uint8_t v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; 
lean_dec(v___x_179_);
v_ref_216_ = lean_ctor_get(v_a_164_, 5);
v___x_217_ = 0;
v___x_218_ = l_Lean_SourceInfo_fromRef(v_ref_216_, v___x_217_);
v___x_219_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__3));
v___x_220_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__4));
lean_inc_n(v___x_218_, 14);
v___x_221_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_221_, 0, v___x_218_);
lean_ctor_set(v___x_221_, 1, v___x_220_);
v___x_222_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__6));
v___x_223_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__8));
v___x_224_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__10));
v___x_225_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__11));
v___x_226_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__12));
v___x_227_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_227_, 0, v___x_218_);
lean_ctor_set(v___x_227_, 1, v___x_225_);
v___x_228_ = l_Lean_Syntax_node1(v___x_218_, v___x_224_, v___x_178_);
lean_inc(v___x_228_);
v___x_229_ = l_Lean_Syntax_node2(v___x_218_, v___x_226_, v___x_227_, v___x_228_);
v___x_230_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__13));
v___x_231_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_231_, 0, v___x_218_);
lean_ctor_set(v___x_231_, 1, v___x_230_);
v___x_232_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_232_, 0, v___x_218_);
lean_ctor_set(v___x_232_, 1, v___x_166_);
v___x_233_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__14, &lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__14);
v___x_234_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_234_, 0, v___x_218_);
lean_ctor_set(v___x_234_, 1, v___x_224_);
lean_ctor_set(v___x_234_, 2, v___x_233_);
v___x_235_ = l_Lean_Syntax_node2(v___x_218_, v___x_167_, v___x_232_, v___x_234_);
v___x_236_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__15));
v___x_237_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__16));
v___x_238_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_238_, 0, v___x_218_);
lean_ctor_set(v___x_238_, 1, v___x_236_);
v___x_239_ = l_Lean_Syntax_node2(v___x_218_, v___x_237_, v___x_238_, v___x_228_);
lean_inc_ref(v___x_231_);
v___x_240_ = l_Lean_Syntax_node5(v___x_218_, v___x_224_, v___x_229_, v___x_231_, v___x_235_, v___x_231_, v___x_239_);
v___x_241_ = l_Lean_Syntax_node1(v___x_218_, v___x_223_, v___x_240_);
v___x_242_ = l_Lean_Syntax_node1(v___x_218_, v___x_222_, v___x_241_);
v___x_243_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__17));
v___x_244_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_244_, 0, v___x_218_);
lean_ctor_set(v___x_244_, 1, v___x_243_);
v___x_245_ = l_Lean_Syntax_node3(v___x_218_, v___x_219_, v___x_221_, v___x_242_, v___x_244_);
v___x_246_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_246_, 0, v___x_245_);
lean_ctor_set(v___x_246_, 1, v_a_165_);
return v___x_246_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___boxed(lean_object* v_x_247_, lean_object* v_a_248_, lean_object* v_a_249_){
_start:
{
lean_object* v_res_250_; 
v_res_250_ = lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1(v_x_247_, v_a_248_, v_a_249_);
lean_dec_ref(v_a_248_);
return v_res_250_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; 
v___x_251_ = lean_box(0);
v___x_252_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_253_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_253_, 0, v___x_252_);
lean_ctor_set(v___x_253_, 1, v___x_251_);
return v___x_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__0___redArg(){
_start:
{
lean_object* v___x_255_; lean_object* v___x_256_; 
v___x_255_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__0___redArg___closed__0);
v___x_256_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_256_, 0, v___x_255_);
return v___x_256_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__0___redArg___boxed(lean_object* v___y_257_){
_start:
{
lean_object* v_res_258_; 
v_res_258_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__0___redArg();
return v_res_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__0(lean_object* v_00_u03b1_259_, lean_object* v___y_260_, lean_object* v___y_261_, lean_object* v___y_262_, lean_object* v___y_263_, lean_object* v___y_264_, lean_object* v___y_265_, lean_object* v___y_266_, lean_object* v___y_267_){
_start:
{
lean_object* v___x_269_; 
v___x_269_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__0___redArg();
return v___x_269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__0___boxed(lean_object* v_00_u03b1_270_, lean_object* v___y_271_, lean_object* v___y_272_, lean_object* v___y_273_, lean_object* v___y_274_, lean_object* v___y_275_, lean_object* v___y_276_, lean_object* v___y_277_, lean_object* v___y_278_, lean_object* v___y_279_){
_start:
{
lean_object* v_res_280_; 
v_res_280_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__0(v_00_u03b1_270_, v___y_271_, v___y_272_, v___y_273_, v___y_274_, v___y_275_, v___y_276_, v___y_277_, v___y_278_);
lean_dec(v___y_278_);
lean_dec_ref(v___y_277_);
lean_dec(v___y_276_);
lean_dec_ref(v___y_275_);
lean_dec(v___y_274_);
lean_dec_ref(v___y_273_);
lean_dec(v___y_272_);
lean_dec_ref(v___y_271_);
return v_res_280_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0___closed__1(void){
_start:
{
lean_object* v___x_282_; lean_object* v___x_283_; 
v___x_282_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0___closed__0));
v___x_283_ = l_Lean_stringToMessageData(v___x_282_);
return v___x_283_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0___closed__3(void){
_start:
{
lean_object* v___x_285_; lean_object* v___x_286_; 
v___x_285_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0___closed__2));
v___x_286_ = l_Lean_stringToMessageData(v___x_285_);
return v___x_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0(lean_object* v___x_287_, lean_object* v_a_288_, lean_object* v_a_289_, lean_object* v_x_290_, lean_object* v___y_291_, lean_object* v___y_292_, lean_object* v___y_293_, lean_object* v___y_294_){
_start:
{
lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; 
v___x_296_ = l_Lean_Name_mkStr1(v___x_287_);
v___x_297_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0___closed__1);
v___x_298_ = l_Lean_MessageData_ofExpr(v_a_288_);
v___x_299_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_299_, 0, v___x_297_);
lean_ctor_set(v___x_299_, 1, v___x_298_);
v___x_300_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0___closed__3, &lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0___closed__3);
v___x_301_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_301_, 0, v___x_299_);
lean_ctor_set(v___x_301_, 1, v___x_300_);
v___x_302_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_302_, 0, v___x_301_);
v___x_303_ = l_Lean_Meta_throwTacticEx___redArg(v___x_296_, v_a_289_, v___x_302_, v___y_291_, v___y_292_, v___y_293_, v___y_294_);
return v___x_303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0___boxed(lean_object* v___x_304_, lean_object* v_a_305_, lean_object* v_a_306_, lean_object* v_x_307_, lean_object* v___y_308_, lean_object* v___y_309_, lean_object* v___y_310_, lean_object* v___y_311_, lean_object* v___y_312_){
_start:
{
lean_object* v_res_313_; 
v_res_313_ = lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0(v___x_304_, v_a_305_, v_a_306_, v_x_307_, v___y_308_, v___y_309_, v___y_310_, v___y_311_);
lean_dec(v___y_311_);
lean_dec_ref(v___y_310_);
lean_dec(v___y_309_);
lean_dec_ref(v___y_308_);
lean_dec_ref(v_x_307_);
return v_res_313_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__7(void){
_start:
{
lean_object* v___x_322_; lean_object* v___x_323_; 
v___x_322_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__6));
v___x_323_ = l_Lean_stringToMessageData(v___x_322_);
return v___x_323_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__9(void){
_start:
{
lean_object* v___x_325_; lean_object* v___x_326_; 
v___x_325_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__8));
v___x_326_ = l_Lean_stringToMessageData(v___x_325_);
return v___x_326_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__11(void){
_start:
{
lean_object* v___x_328_; lean_object* v___x_329_; 
v___x_328_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__10));
v___x_329_ = l_Lean_stringToMessageData(v___x_328_);
return v___x_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1(lean_object* v_binderType_330_, lean_object* v_body_331_, lean_object* v___x_332_, lean_object* v___x_333_, lean_object* v___x_334_, lean_object* v___x_335_, uint8_t v___x_336_, lean_object* v_a_337_, lean_object* v___x_338_, lean_object* v_____r_339_, lean_object* v___y_340_, lean_object* v___y_341_, lean_object* v___y_342_, lean_object* v___y_343_){
_start:
{
lean_object* v___y_346_; lean_object* v___y_347_; lean_object* v___y_348_; lean_object* v___y_349_; lean_object* v___y_415_; lean_object* v___y_416_; lean_object* v___y_417_; lean_object* v___y_418_; lean_object* v___x_446_; 
lean_inc_ref(v_binderType_330_);
v___x_446_ = l_Lean_Meta_isProp(v_binderType_330_, v___y_340_, v___y_341_, v___y_342_, v___y_343_);
if (lean_obj_tag(v___x_446_) == 0)
{
lean_object* v_a_447_; uint8_t v___x_448_; 
v_a_447_ = lean_ctor_get(v___x_446_, 0);
lean_inc(v_a_447_);
lean_dec_ref_known(v___x_446_, 1);
v___x_448_ = lean_unbox(v_a_447_);
lean_dec(v_a_447_);
if (v___x_448_ == 0)
{
lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; 
lean_inc_ref(v___x_338_);
v___x_449_ = l_Lean_Name_mkStr1(v___x_338_);
v___x_450_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__11, &lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__11_once, _init_lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__11);
lean_inc_ref(v_binderType_330_);
v___x_451_ = l_Lean_MessageData_ofExpr(v_binderType_330_);
v___x_452_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_452_, 0, v___x_450_);
lean_ctor_set(v___x_452_, 1, v___x_451_);
v___x_453_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__9, &lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__9);
v___x_454_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_454_, 0, v___x_452_);
lean_ctor_set(v___x_454_, 1, v___x_453_);
v___x_455_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_455_, 0, v___x_454_);
lean_inc(v_a_337_);
v___x_456_ = l_Lean_Meta_throwTacticEx___redArg(v___x_449_, v_a_337_, v___x_455_, v___y_340_, v___y_341_, v___y_342_, v___y_343_);
if (lean_obj_tag(v___x_456_) == 0)
{
lean_dec_ref_known(v___x_456_, 1);
v___y_415_ = v___y_340_;
v___y_416_ = v___y_341_;
v___y_417_ = v___y_342_;
v___y_418_ = v___y_343_;
goto v___jp_414_;
}
else
{
lean_object* v_a_457_; lean_object* v___x_459_; uint8_t v_isShared_460_; uint8_t v_isSharedCheck_464_; 
lean_dec_ref(v___x_338_);
lean_dec(v_a_337_);
lean_dec_ref(v___x_335_);
lean_dec_ref(v___x_334_);
lean_dec_ref(v___x_333_);
lean_dec(v___x_332_);
lean_dec_ref(v_body_331_);
lean_dec_ref(v_binderType_330_);
v_a_457_ = lean_ctor_get(v___x_456_, 0);
v_isSharedCheck_464_ = !lean_is_exclusive(v___x_456_);
if (v_isSharedCheck_464_ == 0)
{
v___x_459_ = v___x_456_;
v_isShared_460_ = v_isSharedCheck_464_;
goto v_resetjp_458_;
}
else
{
lean_inc(v_a_457_);
lean_dec(v___x_456_);
v___x_459_ = lean_box(0);
v_isShared_460_ = v_isSharedCheck_464_;
goto v_resetjp_458_;
}
v_resetjp_458_:
{
lean_object* v___x_462_; 
if (v_isShared_460_ == 0)
{
v___x_462_ = v___x_459_;
goto v_reusejp_461_;
}
else
{
lean_object* v_reuseFailAlloc_463_; 
v_reuseFailAlloc_463_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_463_, 0, v_a_457_);
v___x_462_ = v_reuseFailAlloc_463_;
goto v_reusejp_461_;
}
v_reusejp_461_:
{
return v___x_462_;
}
}
}
}
else
{
v___y_415_ = v___y_340_;
v___y_416_ = v___y_341_;
v___y_417_ = v___y_342_;
v___y_418_ = v___y_343_;
goto v___jp_414_;
}
}
else
{
lean_object* v_a_465_; lean_object* v___x_467_; uint8_t v_isShared_468_; uint8_t v_isSharedCheck_472_; 
lean_dec_ref(v___x_338_);
lean_dec(v_a_337_);
lean_dec_ref(v___x_335_);
lean_dec_ref(v___x_334_);
lean_dec_ref(v___x_333_);
lean_dec(v___x_332_);
lean_dec_ref(v_body_331_);
lean_dec_ref(v_binderType_330_);
v_a_465_ = lean_ctor_get(v___x_446_, 0);
v_isSharedCheck_472_ = !lean_is_exclusive(v___x_446_);
if (v_isSharedCheck_472_ == 0)
{
v___x_467_ = v___x_446_;
v_isShared_468_ = v_isSharedCheck_472_;
goto v_resetjp_466_;
}
else
{
lean_inc(v_a_465_);
lean_dec(v___x_446_);
v___x_467_ = lean_box(0);
v_isShared_468_ = v_isSharedCheck_472_;
goto v_resetjp_466_;
}
v_resetjp_466_:
{
lean_object* v___x_470_; 
if (v_isShared_468_ == 0)
{
v___x_470_ = v___x_467_;
goto v_reusejp_469_;
}
else
{
lean_object* v_reuseFailAlloc_471_; 
v_reuseFailAlloc_471_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_471_, 0, v_a_465_);
v___x_470_ = v_reuseFailAlloc_471_;
goto v_reusejp_469_;
}
v_reusejp_469_:
{
return v___x_470_;
}
}
}
v___jp_345_:
{
lean_object* v___x_350_; 
lean_inc(v___y_349_);
lean_inc_ref(v___y_348_);
lean_inc(v___y_347_);
lean_inc_ref(v___y_346_);
lean_inc_ref(v_binderType_330_);
v___x_350_ = lean_whnf(v_binderType_330_, v___y_346_, v___y_347_, v___y_348_, v___y_349_);
if (lean_obj_tag(v___x_350_) == 0)
{
lean_object* v_a_351_; lean_object* v___x_352_; 
v_a_351_ = lean_ctor_get(v___x_350_, 0);
lean_inc(v_a_351_);
lean_dec_ref_known(v___x_350_, 1);
lean_inc(v___y_349_);
lean_inc_ref(v___y_348_);
lean_inc(v___y_347_);
lean_inc_ref(v___y_346_);
lean_inc_ref(v_body_331_);
v___x_352_ = lean_whnf(v_body_331_, v___y_346_, v___y_347_, v___y_348_, v___y_349_);
if (lean_obj_tag(v___x_352_) == 0)
{
lean_object* v_a_353_; lean_object* v___x_354_; uint8_t v___x_355_; 
v_a_353_ = lean_ctor_get(v___x_352_, 0);
lean_inc(v_a_353_);
lean_dec_ref_known(v___x_352_, 1);
v___x_354_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__1));
lean_inc(v___x_332_);
v___x_355_ = l_Lean_Expr_isAppOfArity(v_a_351_, v___x_354_, v___x_332_);
if (v___x_355_ == 0)
{
uint8_t v___x_356_; 
lean_dec(v_a_351_);
v___x_356_ = l_Lean_Expr_isAppOfArity(v_a_353_, v___x_354_, v___x_332_);
if (v___x_356_ == 0)
{
lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; uint8_t v___x_362_; lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; 
lean_dec(v_a_353_);
v___x_357_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__2));
v___x_358_ = l_Lean_Name_mkStr4(v___x_333_, v___x_334_, v___x_335_, v___x_357_);
v___x_359_ = lean_box(0);
v___x_360_ = l_Lean_Expr_const___override(v___x_358_, v___x_359_);
v___x_361_ = l_Lean_mkAppB(v___x_360_, v_binderType_330_, v_body_331_);
v___x_362_ = 0;
v___x_363_ = lean_alloc_ctor(0, 0, 4);
lean_ctor_set_uint8(v___x_363_, 0, v___x_362_);
lean_ctor_set_uint8(v___x_363_, 1, v___x_336_);
lean_ctor_set_uint8(v___x_363_, 2, v___x_356_);
lean_ctor_set_uint8(v___x_363_, 3, v___x_336_);
v___x_364_ = lean_box(0);
v___x_365_ = l_Lean_MVarId_apply(v_a_337_, v___x_361_, v___x_363_, v___x_364_, v___y_346_, v___y_347_, v___y_348_, v___y_349_);
return v___x_365_;
}
else
{
lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; uint8_t v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; 
lean_dec_ref(v_body_331_);
v___x_366_ = l_Lean_Expr_appArg_x21(v_a_353_);
lean_dec(v_a_353_);
v___x_367_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__3));
v___x_368_ = l_Lean_Name_mkStr4(v___x_333_, v___x_334_, v___x_335_, v___x_367_);
v___x_369_ = lean_box(0);
v___x_370_ = l_Lean_Expr_const___override(v___x_368_, v___x_369_);
v___x_371_ = l_Lean_mkAppB(v___x_370_, v_binderType_330_, v___x_366_);
v___x_372_ = 0;
v___x_373_ = lean_alloc_ctor(0, 0, 4);
lean_ctor_set_uint8(v___x_373_, 0, v___x_372_);
lean_ctor_set_uint8(v___x_373_, 1, v___x_336_);
lean_ctor_set_uint8(v___x_373_, 2, v___x_355_);
lean_ctor_set_uint8(v___x_373_, 3, v___x_336_);
v___x_374_ = lean_box(0);
v___x_375_ = l_Lean_MVarId_apply(v_a_337_, v___x_371_, v___x_373_, v___x_374_, v___y_346_, v___y_347_, v___y_348_, v___y_349_);
return v___x_375_;
}
}
else
{
lean_object* v___x_376_; uint8_t v___x_377_; 
lean_dec_ref(v_binderType_330_);
v___x_376_ = l_Lean_Expr_appArg_x21(v_a_351_);
lean_dec(v_a_351_);
v___x_377_ = l_Lean_Expr_isAppOfArity(v_a_353_, v___x_354_, v___x_332_);
if (v___x_377_ == 0)
{
lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; uint8_t v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; 
lean_dec(v_a_353_);
v___x_378_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__4));
v___x_379_ = l_Lean_Name_mkStr4(v___x_333_, v___x_334_, v___x_335_, v___x_378_);
v___x_380_ = lean_box(0);
v___x_381_ = l_Lean_Expr_const___override(v___x_379_, v___x_380_);
v___x_382_ = l_Lean_mkAppB(v___x_381_, v___x_376_, v_body_331_);
v___x_383_ = 0;
v___x_384_ = lean_alloc_ctor(0, 0, 4);
lean_ctor_set_uint8(v___x_384_, 0, v___x_383_);
lean_ctor_set_uint8(v___x_384_, 1, v___x_336_);
lean_ctor_set_uint8(v___x_384_, 2, v___x_377_);
lean_ctor_set_uint8(v___x_384_, 3, v___x_336_);
v___x_385_ = lean_box(0);
v___x_386_ = l_Lean_MVarId_apply(v_a_337_, v___x_382_, v___x_384_, v___x_385_, v___y_346_, v___y_347_, v___y_348_, v___y_349_);
return v___x_386_;
}
else
{
lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; uint8_t v___x_393_; uint8_t v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; 
lean_dec_ref(v_body_331_);
v___x_387_ = l_Lean_Expr_appArg_x21(v_a_353_);
lean_dec(v_a_353_);
v___x_388_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__5));
v___x_389_ = l_Lean_Name_mkStr4(v___x_333_, v___x_334_, v___x_335_, v___x_388_);
v___x_390_ = lean_box(0);
v___x_391_ = l_Lean_Expr_const___override(v___x_389_, v___x_390_);
v___x_392_ = l_Lean_mkAppB(v___x_391_, v___x_376_, v___x_387_);
v___x_393_ = 0;
v___x_394_ = 0;
v___x_395_ = lean_alloc_ctor(0, 0, 4);
lean_ctor_set_uint8(v___x_395_, 0, v___x_393_);
lean_ctor_set_uint8(v___x_395_, 1, v___x_336_);
lean_ctor_set_uint8(v___x_395_, 2, v___x_394_);
lean_ctor_set_uint8(v___x_395_, 3, v___x_336_);
v___x_396_ = lean_box(0);
v___x_397_ = l_Lean_MVarId_apply(v_a_337_, v___x_392_, v___x_395_, v___x_396_, v___y_346_, v___y_347_, v___y_348_, v___y_349_);
return v___x_397_;
}
}
}
else
{
lean_object* v_a_398_; lean_object* v___x_400_; uint8_t v_isShared_401_; uint8_t v_isSharedCheck_405_; 
lean_dec(v_a_351_);
lean_dec(v_a_337_);
lean_dec_ref(v___x_335_);
lean_dec_ref(v___x_334_);
lean_dec_ref(v___x_333_);
lean_dec(v___x_332_);
lean_dec_ref(v_body_331_);
lean_dec_ref(v_binderType_330_);
v_a_398_ = lean_ctor_get(v___x_352_, 0);
v_isSharedCheck_405_ = !lean_is_exclusive(v___x_352_);
if (v_isSharedCheck_405_ == 0)
{
v___x_400_ = v___x_352_;
v_isShared_401_ = v_isSharedCheck_405_;
goto v_resetjp_399_;
}
else
{
lean_inc(v_a_398_);
lean_dec(v___x_352_);
v___x_400_ = lean_box(0);
v_isShared_401_ = v_isSharedCheck_405_;
goto v_resetjp_399_;
}
v_resetjp_399_:
{
lean_object* v___x_403_; 
if (v_isShared_401_ == 0)
{
v___x_403_ = v___x_400_;
goto v_reusejp_402_;
}
else
{
lean_object* v_reuseFailAlloc_404_; 
v_reuseFailAlloc_404_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_404_, 0, v_a_398_);
v___x_403_ = v_reuseFailAlloc_404_;
goto v_reusejp_402_;
}
v_reusejp_402_:
{
return v___x_403_;
}
}
}
}
else
{
lean_object* v_a_406_; lean_object* v___x_408_; uint8_t v_isShared_409_; uint8_t v_isSharedCheck_413_; 
lean_dec(v_a_337_);
lean_dec_ref(v___x_335_);
lean_dec_ref(v___x_334_);
lean_dec_ref(v___x_333_);
lean_dec(v___x_332_);
lean_dec_ref(v_body_331_);
lean_dec_ref(v_binderType_330_);
v_a_406_ = lean_ctor_get(v___x_350_, 0);
v_isSharedCheck_413_ = !lean_is_exclusive(v___x_350_);
if (v_isSharedCheck_413_ == 0)
{
v___x_408_ = v___x_350_;
v_isShared_409_ = v_isSharedCheck_413_;
goto v_resetjp_407_;
}
else
{
lean_inc(v_a_406_);
lean_dec(v___x_350_);
v___x_408_ = lean_box(0);
v_isShared_409_ = v_isSharedCheck_413_;
goto v_resetjp_407_;
}
v_resetjp_407_:
{
lean_object* v___x_411_; 
if (v_isShared_409_ == 0)
{
v___x_411_ = v___x_408_;
goto v_reusejp_410_;
}
else
{
lean_object* v_reuseFailAlloc_412_; 
v_reuseFailAlloc_412_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_412_, 0, v_a_406_);
v___x_411_ = v_reuseFailAlloc_412_;
goto v_reusejp_410_;
}
v_reusejp_410_:
{
return v___x_411_;
}
}
}
}
v___jp_414_:
{
lean_object* v___x_419_; 
lean_inc_ref(v_body_331_);
v___x_419_ = l_Lean_Meta_isProp(v_body_331_, v___y_415_, v___y_416_, v___y_417_, v___y_418_);
if (lean_obj_tag(v___x_419_) == 0)
{
lean_object* v_a_420_; uint8_t v___x_421_; 
v_a_420_ = lean_ctor_get(v___x_419_, 0);
lean_inc(v_a_420_);
lean_dec_ref_known(v___x_419_, 1);
v___x_421_ = lean_unbox(v_a_420_);
lean_dec(v_a_420_);
if (v___x_421_ == 0)
{
lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; 
v___x_422_ = l_Lean_Name_mkStr1(v___x_338_);
v___x_423_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__7, &lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__7);
lean_inc_ref(v_body_331_);
v___x_424_ = l_Lean_MessageData_ofExpr(v_body_331_);
v___x_425_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_425_, 0, v___x_423_);
lean_ctor_set(v___x_425_, 1, v___x_424_);
v___x_426_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__9, &lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__9);
v___x_427_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_427_, 0, v___x_425_);
lean_ctor_set(v___x_427_, 1, v___x_426_);
v___x_428_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_428_, 0, v___x_427_);
lean_inc(v_a_337_);
v___x_429_ = l_Lean_Meta_throwTacticEx___redArg(v___x_422_, v_a_337_, v___x_428_, v___y_415_, v___y_416_, v___y_417_, v___y_418_);
if (lean_obj_tag(v___x_429_) == 0)
{
lean_dec_ref_known(v___x_429_, 1);
v___y_346_ = v___y_415_;
v___y_347_ = v___y_416_;
v___y_348_ = v___y_417_;
v___y_349_ = v___y_418_;
goto v___jp_345_;
}
else
{
lean_object* v_a_430_; lean_object* v___x_432_; uint8_t v_isShared_433_; uint8_t v_isSharedCheck_437_; 
lean_dec(v_a_337_);
lean_dec_ref(v___x_335_);
lean_dec_ref(v___x_334_);
lean_dec_ref(v___x_333_);
lean_dec(v___x_332_);
lean_dec_ref(v_body_331_);
lean_dec_ref(v_binderType_330_);
v_a_430_ = lean_ctor_get(v___x_429_, 0);
v_isSharedCheck_437_ = !lean_is_exclusive(v___x_429_);
if (v_isSharedCheck_437_ == 0)
{
v___x_432_ = v___x_429_;
v_isShared_433_ = v_isSharedCheck_437_;
goto v_resetjp_431_;
}
else
{
lean_inc(v_a_430_);
lean_dec(v___x_429_);
v___x_432_ = lean_box(0);
v_isShared_433_ = v_isSharedCheck_437_;
goto v_resetjp_431_;
}
v_resetjp_431_:
{
lean_object* v___x_435_; 
if (v_isShared_433_ == 0)
{
v___x_435_ = v___x_432_;
goto v_reusejp_434_;
}
else
{
lean_object* v_reuseFailAlloc_436_; 
v_reuseFailAlloc_436_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_436_, 0, v_a_430_);
v___x_435_ = v_reuseFailAlloc_436_;
goto v_reusejp_434_;
}
v_reusejp_434_:
{
return v___x_435_;
}
}
}
}
else
{
lean_dec_ref(v___x_338_);
v___y_346_ = v___y_415_;
v___y_347_ = v___y_416_;
v___y_348_ = v___y_417_;
v___y_349_ = v___y_418_;
goto v___jp_345_;
}
}
else
{
lean_object* v_a_438_; lean_object* v___x_440_; uint8_t v_isShared_441_; uint8_t v_isSharedCheck_445_; 
lean_dec_ref(v___x_338_);
lean_dec(v_a_337_);
lean_dec_ref(v___x_335_);
lean_dec_ref(v___x_334_);
lean_dec_ref(v___x_333_);
lean_dec(v___x_332_);
lean_dec_ref(v_body_331_);
lean_dec_ref(v_binderType_330_);
v_a_438_ = lean_ctor_get(v___x_419_, 0);
v_isSharedCheck_445_ = !lean_is_exclusive(v___x_419_);
if (v_isSharedCheck_445_ == 0)
{
v___x_440_ = v___x_419_;
v_isShared_441_ = v_isSharedCheck_445_;
goto v_resetjp_439_;
}
else
{
lean_inc(v_a_438_);
lean_dec(v___x_419_);
v___x_440_ = lean_box(0);
v_isShared_441_ = v_isSharedCheck_445_;
goto v_resetjp_439_;
}
v_resetjp_439_:
{
lean_object* v___x_443_; 
if (v_isShared_441_ == 0)
{
v___x_443_ = v___x_440_;
goto v_reusejp_442_;
}
else
{
lean_object* v_reuseFailAlloc_444_; 
v_reuseFailAlloc_444_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_444_, 0, v_a_438_);
v___x_443_ = v_reuseFailAlloc_444_;
goto v_reusejp_442_;
}
v_reusejp_442_:
{
return v___x_443_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___boxed(lean_object* v_binderType_473_, lean_object* v_body_474_, lean_object* v___x_475_, lean_object* v___x_476_, lean_object* v___x_477_, lean_object* v___x_478_, lean_object* v___x_479_, lean_object* v_a_480_, lean_object* v___x_481_, lean_object* v_____r_482_, lean_object* v___y_483_, lean_object* v___y_484_, lean_object* v___y_485_, lean_object* v___y_486_, lean_object* v___y_487_){
_start:
{
uint8_t v___x_7971__boxed_488_; lean_object* v_res_489_; 
v___x_7971__boxed_488_ = lean_unbox(v___x_479_);
v_res_489_ = lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1(v_binderType_473_, v_body_474_, v___x_475_, v___x_476_, v___x_477_, v___x_478_, v___x_7971__boxed_488_, v_a_480_, v___x_481_, v_____r_482_, v___y_483_, v___y_484_, v___y_485_, v___y_486_);
lean_dec(v___y_486_);
lean_dec_ref(v___y_485_);
lean_dec(v___y_484_);
lean_dec_ref(v___y_483_);
return v_res_489_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Option_getM___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__1_spec__1(lean_object* v_opts_490_, lean_object* v_opt_491_){
_start:
{
lean_object* v_name_492_; lean_object* v_defValue_493_; lean_object* v_map_494_; lean_object* v___x_495_; 
v_name_492_ = lean_ctor_get(v_opt_491_, 0);
v_defValue_493_ = lean_ctor_get(v_opt_491_, 1);
v_map_494_ = lean_ctor_get(v_opts_490_, 0);
v___x_495_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_494_, v_name_492_);
if (lean_obj_tag(v___x_495_) == 0)
{
uint8_t v___x_496_; 
v___x_496_ = lean_unbox(v_defValue_493_);
return v___x_496_;
}
else
{
lean_object* v_val_497_; 
v_val_497_ = lean_ctor_get(v___x_495_, 0);
lean_inc(v_val_497_);
lean_dec_ref_known(v___x_495_, 1);
if (lean_obj_tag(v_val_497_) == 1)
{
uint8_t v_v_498_; 
v_v_498_ = lean_ctor_get_uint8(v_val_497_, 0);
lean_dec_ref_known(v_val_497_, 0);
return v_v_498_;
}
else
{
uint8_t v___x_499_; 
lean_dec(v_val_497_);
v___x_499_ = lean_unbox(v_defValue_493_);
return v___x_499_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Option_getM___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__1_spec__1___boxed(lean_object* v_opts_500_, lean_object* v_opt_501_){
_start:
{
uint8_t v_res_502_; lean_object* v_r_503_; 
v_res_502_ = lp_mathlib_Lean_Option_get___at___00Lean_Option_getM___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__1_spec__1(v_opts_500_, v_opt_501_);
lean_dec_ref(v_opt_501_);
lean_dec_ref(v_opts_500_);
v_r_503_ = lean_box(v_res_502_);
return v_r_503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_getM___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__1___redArg(lean_object* v_opt_504_, lean_object* v___y_505_){
_start:
{
lean_object* v_options_507_; uint8_t v___x_508_; lean_object* v___x_509_; lean_object* v___x_510_; 
v_options_507_ = lean_ctor_get(v___y_505_, 2);
v___x_508_ = lp_mathlib_Lean_Option_get___at___00Lean_Option_getM___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__1_spec__1(v_options_507_, v_opt_504_);
v___x_509_ = lean_box(v___x_508_);
v___x_510_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_510_, 0, v___x_509_);
return v___x_510_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_getM___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__1___redArg___boxed(lean_object* v_opt_511_, lean_object* v___y_512_, lean_object* v___y_513_){
_start:
{
lean_object* v_res_514_; 
v_res_514_ = lp_mathlib_Lean_Option_getM___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__1___redArg(v_opt_511_, v___y_512_);
lean_dec_ref(v___y_512_);
lean_dec_ref(v_opt_511_);
return v_res_514_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__3(void){
_start:
{
lean_object* v___x_519_; lean_object* v___x_520_; 
v___x_519_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__2));
v___x_520_ = l_Lean_MessageData_ofFormat(v___x_519_);
return v___x_520_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__4(void){
_start:
{
lean_object* v___x_521_; lean_object* v___x_522_; 
v___x_521_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__3, &lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__3);
v___x_522_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_522_, 0, v___x_521_);
return v___x_522_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__10(void){
_start:
{
lean_object* v___x_528_; lean_object* v___x_529_; 
v___x_528_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__9));
v___x_529_ = l_Lean_stringToMessageData(v___x_528_);
return v___x_529_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2(lean_object* v___x_530_, lean_object* v___x_531_, lean_object* v___x_532_, lean_object* v___x_533_, lean_object* v___x_534_, uint8_t v___x_535_, lean_object* v___y_536_, lean_object* v___y_537_, lean_object* v___y_538_, lean_object* v___y_539_, lean_object* v___y_540_, lean_object* v___y_541_, lean_object* v___y_542_, lean_object* v___y_543_){
_start:
{
lean_object* v___y_546_; lean_object* v___x_566_; 
v___x_566_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_537_, v___y_540_, v___y_541_, v___y_542_, v___y_543_);
if (lean_obj_tag(v___x_566_) == 0)
{
lean_object* v_a_567_; lean_object* v_keyedConfig_568_; uint8_t v_trackZetaDelta_569_; lean_object* v_zetaDeltaSet_570_; lean_object* v_lctx_571_; lean_object* v_localInstances_572_; lean_object* v_defEqCtx_x3f_573_; lean_object* v_synthPendingDepth_574_; lean_object* v_customCanUnfoldPredicate_x3f_575_; uint8_t v_univApprox_576_; uint8_t v_inTypeClassResolution_577_; uint8_t v_cacheInferType_578_; uint8_t v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; 
v_a_567_ = lean_ctor_get(v___x_566_, 0);
lean_inc_n(v_a_567_, 2);
lean_dec_ref_known(v___x_566_, 1);
v_keyedConfig_568_ = lean_ctor_get(v___y_540_, 0);
v_trackZetaDelta_569_ = lean_ctor_get_uint8(v___y_540_, sizeof(void*)*7);
v_zetaDeltaSet_570_ = lean_ctor_get(v___y_540_, 1);
v_lctx_571_ = lean_ctor_get(v___y_540_, 2);
v_localInstances_572_ = lean_ctor_get(v___y_540_, 3);
v_defEqCtx_x3f_573_ = lean_ctor_get(v___y_540_, 4);
v_synthPendingDepth_574_ = lean_ctor_get(v___y_540_, 5);
v_customCanUnfoldPredicate_x3f_575_ = lean_ctor_get(v___y_540_, 6);
v_univApprox_576_ = lean_ctor_get_uint8(v___y_540_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_577_ = lean_ctor_get_uint8(v___y_540_, sizeof(void*)*7 + 2);
v_cacheInferType_578_ = lean_ctor_get_uint8(v___y_540_, sizeof(void*)*7 + 3);
v___x_579_ = 2;
lean_inc_ref(v_keyedConfig_568_);
v___x_580_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_579_, v_keyedConfig_568_);
lean_inc(v_customCanUnfoldPredicate_x3f_575_);
lean_inc(v_synthPendingDepth_574_);
lean_inc(v_defEqCtx_x3f_573_);
lean_inc_ref(v_localInstances_572_);
lean_inc_ref(v_lctx_571_);
lean_inc(v_zetaDeltaSet_570_);
v___x_581_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_581_, 0, v___x_580_);
lean_ctor_set(v___x_581_, 1, v_zetaDeltaSet_570_);
lean_ctor_set(v___x_581_, 2, v_lctx_571_);
lean_ctor_set(v___x_581_, 3, v_localInstances_572_);
lean_ctor_set(v___x_581_, 4, v_defEqCtx_x3f_573_);
lean_ctor_set(v___x_581_, 5, v_synthPendingDepth_574_);
lean_ctor_set(v___x_581_, 6, v_customCanUnfoldPredicate_x3f_575_);
lean_ctor_set_uint8(v___x_581_, sizeof(void*)*7, v_trackZetaDelta_569_);
lean_ctor_set_uint8(v___x_581_, sizeof(void*)*7 + 1, v_univApprox_576_);
lean_ctor_set_uint8(v___x_581_, sizeof(void*)*7 + 2, v_inTypeClassResolution_577_);
lean_ctor_set_uint8(v___x_581_, sizeof(void*)*7 + 3, v_cacheInferType_578_);
v___x_582_ = l_Lean_MVarId_getType_x27(v_a_567_, v___x_581_, v___y_541_, v___y_542_, v___y_543_);
if (lean_obj_tag(v___x_582_) == 0)
{
lean_object* v_a_583_; 
v_a_583_ = lean_ctor_get(v___x_582_, 0);
lean_inc(v_a_583_);
lean_dec_ref_known(v___x_582_, 1);
switch(lean_obj_tag(v_a_583_))
{
case 5:
{
lean_object* v_fn_584_; 
v_fn_584_ = lean_ctor_get(v_a_583_, 0);
if (lean_obj_tag(v_fn_584_) == 5)
{
lean_object* v_fn_585_; 
v_fn_585_ = lean_ctor_get(v_fn_584_, 0);
if (lean_obj_tag(v_fn_585_) == 4)
{
lean_object* v_declName_586_; 
v_declName_586_ = lean_ctor_get(v_fn_585_, 0);
if (lean_obj_tag(v_declName_586_) == 1)
{
lean_object* v_pre_587_; 
v_pre_587_ = lean_ctor_get(v_declName_586_, 0);
if (lean_obj_tag(v_pre_587_) == 0)
{
lean_object* v_arg_588_; lean_object* v_arg_589_; lean_object* v_str_590_; lean_object* v___x_591_; uint8_t v___x_592_; 
v_arg_588_ = lean_ctor_get(v_a_583_, 1);
v_arg_589_ = lean_ctor_get(v_fn_584_, 1);
v_str_590_ = lean_ctor_get(v_declName_586_, 1);
v___x_591_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__0));
v___x_592_ = lean_string_dec_eq(v_str_590_, v___x_591_);
if (v___x_592_ == 0)
{
lean_object* v___x_593_; 
lean_dec_ref(v___x_534_);
lean_dec_ref(v___x_533_);
lean_dec_ref(v___x_532_);
lean_dec(v___x_531_);
lean_inc_ref(v_a_583_);
v___x_593_ = lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0(v___x_530_, v_a_583_, v_a_567_, v_a_583_, v___x_581_, v___y_541_, v___y_542_, v___y_543_);
lean_dec_ref_known(v___x_581_, 7);
lean_dec_ref_known(v_a_583_, 2);
v___y_546_ = v___x_593_;
goto v___jp_545_;
}
else
{
lean_object* v___x_594_; lean_object* v___x_595_; lean_object* v_a_596_; uint8_t v___x_597_; 
lean_inc_ref(v_arg_589_);
lean_inc_ref(v_arg_588_);
lean_dec_ref_known(v_a_583_, 2);
v___x_594_ = lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_negate__iff;
v___x_595_ = lp_mathlib_Lean_Option_getM___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__1___redArg(v___x_594_, v___y_542_);
v_a_596_ = lean_ctor_get(v___x_595_, 0);
lean_inc(v_a_596_);
lean_dec_ref(v___x_595_);
v___x_597_ = lean_unbox(v_a_596_);
lean_dec(v_a_596_);
if (v___x_597_ == 0)
{
lean_object* v___x_598_; lean_object* v___x_599_; lean_object* v___x_600_; 
lean_dec_ref(v_arg_589_);
lean_dec_ref(v_arg_588_);
lean_dec_ref(v___x_534_);
lean_dec_ref(v___x_533_);
lean_dec_ref(v___x_532_);
lean_dec(v___x_531_);
v___x_598_ = l_Lean_Name_mkStr1(v___x_530_);
v___x_599_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__4, &lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__4);
v___x_600_ = l_Lean_Meta_throwTacticEx___redArg(v___x_598_, v_a_567_, v___x_599_, v___x_581_, v___y_541_, v___y_542_, v___y_543_);
lean_dec_ref_known(v___x_581_, 7);
v___y_546_ = v___x_600_;
goto v___jp_545_;
}
else
{
lean_object* v___x_601_; 
lean_dec_ref(v___x_530_);
lean_inc(v___y_543_);
lean_inc_ref(v___y_542_);
lean_inc(v___y_541_);
lean_inc_ref(v___x_581_);
lean_inc_ref(v_arg_589_);
v___x_601_ = lean_whnf(v_arg_589_, v___x_581_, v___y_541_, v___y_542_, v___y_543_);
if (lean_obj_tag(v___x_601_) == 0)
{
lean_object* v_a_602_; lean_object* v___x_603_; 
v_a_602_ = lean_ctor_get(v___x_601_, 0);
lean_inc(v_a_602_);
lean_dec_ref_known(v___x_601_, 1);
lean_inc(v___y_543_);
lean_inc_ref(v___y_542_);
lean_inc(v___y_541_);
lean_inc_ref(v___x_581_);
lean_inc_ref(v_arg_588_);
v___x_603_ = lean_whnf(v_arg_588_, v___x_581_, v___y_541_, v___y_542_, v___y_543_);
if (lean_obj_tag(v___x_603_) == 0)
{
lean_object* v_a_604_; lean_object* v___x_605_; uint8_t v___x_606_; 
v_a_604_ = lean_ctor_get(v___x_603_, 0);
lean_inc(v_a_604_);
lean_dec_ref_known(v___x_603_, 1);
v___x_605_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1___closed__1));
lean_inc(v___x_531_);
v___x_606_ = l_Lean_Expr_isAppOfArity(v_a_602_, v___x_605_, v___x_531_);
if (v___x_606_ == 0)
{
uint8_t v___x_607_; 
lean_dec(v_a_602_);
v___x_607_ = l_Lean_Expr_isAppOfArity(v_a_604_, v___x_605_, v___x_531_);
if (v___x_607_ == 0)
{
lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; lean_object* v___x_612_; uint8_t v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; 
lean_dec(v_a_604_);
v___x_608_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__5));
v___x_609_ = l_Lean_Name_mkStr4(v___x_532_, v___x_533_, v___x_534_, v___x_608_);
v___x_610_ = lean_box(0);
v___x_611_ = l_Lean_Expr_const___override(v___x_609_, v___x_610_);
v___x_612_ = l_Lean_mkAppB(v___x_611_, v_arg_589_, v_arg_588_);
v___x_613_ = 0;
v___x_614_ = lean_alloc_ctor(0, 0, 4);
lean_ctor_set_uint8(v___x_614_, 0, v___x_613_);
lean_ctor_set_uint8(v___x_614_, 1, v___x_535_);
lean_ctor_set_uint8(v___x_614_, 2, v___x_607_);
lean_ctor_set_uint8(v___x_614_, 3, v___x_535_);
v___x_615_ = lean_box(0);
v___x_616_ = l_Lean_MVarId_apply(v_a_567_, v___x_612_, v___x_614_, v___x_615_, v___x_581_, v___y_541_, v___y_542_, v___y_543_);
lean_dec_ref_known(v___x_581_, 7);
v___y_546_ = v___x_616_;
goto v___jp_545_;
}
else
{
lean_object* v___x_617_; lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; uint8_t v___x_623_; lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; 
lean_dec_ref(v_arg_588_);
v___x_617_ = l_Lean_Expr_appArg_x21(v_a_604_);
lean_dec(v_a_604_);
v___x_618_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__6));
v___x_619_ = l_Lean_Name_mkStr4(v___x_532_, v___x_533_, v___x_534_, v___x_618_);
v___x_620_ = lean_box(0);
v___x_621_ = l_Lean_Expr_const___override(v___x_619_, v___x_620_);
v___x_622_ = l_Lean_mkAppB(v___x_621_, v_arg_589_, v___x_617_);
v___x_623_ = 0;
v___x_624_ = lean_alloc_ctor(0, 0, 4);
lean_ctor_set_uint8(v___x_624_, 0, v___x_623_);
lean_ctor_set_uint8(v___x_624_, 1, v___x_535_);
lean_ctor_set_uint8(v___x_624_, 2, v___x_606_);
lean_ctor_set_uint8(v___x_624_, 3, v___x_535_);
v___x_625_ = lean_box(0);
v___x_626_ = l_Lean_MVarId_apply(v_a_567_, v___x_622_, v___x_624_, v___x_625_, v___x_581_, v___y_541_, v___y_542_, v___y_543_);
lean_dec_ref_known(v___x_581_, 7);
v___y_546_ = v___x_626_;
goto v___jp_545_;
}
}
else
{
lean_object* v___x_627_; uint8_t v___x_628_; 
lean_dec_ref(v_arg_589_);
v___x_627_ = l_Lean_Expr_appArg_x21(v_a_602_);
lean_dec(v_a_602_);
v___x_628_ = l_Lean_Expr_isAppOfArity(v_a_604_, v___x_605_, v___x_531_);
if (v___x_628_ == 0)
{
lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_631_; lean_object* v___x_632_; lean_object* v___x_633_; uint8_t v___x_634_; lean_object* v___x_635_; lean_object* v___x_636_; lean_object* v___x_637_; 
lean_dec(v_a_604_);
v___x_629_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__7));
v___x_630_ = l_Lean_Name_mkStr4(v___x_532_, v___x_533_, v___x_534_, v___x_629_);
v___x_631_ = lean_box(0);
v___x_632_ = l_Lean_Expr_const___override(v___x_630_, v___x_631_);
v___x_633_ = l_Lean_mkAppB(v___x_632_, v___x_627_, v_arg_588_);
v___x_634_ = 0;
v___x_635_ = lean_alloc_ctor(0, 0, 4);
lean_ctor_set_uint8(v___x_635_, 0, v___x_634_);
lean_ctor_set_uint8(v___x_635_, 1, v___x_535_);
lean_ctor_set_uint8(v___x_635_, 2, v___x_628_);
lean_ctor_set_uint8(v___x_635_, 3, v___x_535_);
v___x_636_ = lean_box(0);
v___x_637_ = l_Lean_MVarId_apply(v_a_567_, v___x_633_, v___x_635_, v___x_636_, v___x_581_, v___y_541_, v___y_542_, v___y_543_);
lean_dec_ref_known(v___x_581_, 7);
v___y_546_ = v___x_637_;
goto v___jp_545_;
}
else
{
lean_object* v___x_638_; lean_object* v___x_639_; lean_object* v___x_640_; lean_object* v___x_641_; lean_object* v___x_642_; lean_object* v___x_643_; uint8_t v___x_644_; uint8_t v___x_645_; lean_object* v___x_646_; lean_object* v___x_647_; lean_object* v___x_648_; 
lean_dec_ref(v_arg_588_);
v___x_638_ = l_Lean_Expr_appArg_x21(v_a_604_);
lean_dec(v_a_604_);
v___x_639_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__8));
v___x_640_ = l_Lean_Name_mkStr4(v___x_532_, v___x_533_, v___x_534_, v___x_639_);
v___x_641_ = lean_box(0);
v___x_642_ = l_Lean_Expr_const___override(v___x_640_, v___x_641_);
v___x_643_ = l_Lean_mkAppB(v___x_642_, v___x_627_, v___x_638_);
v___x_644_ = 0;
v___x_645_ = 0;
v___x_646_ = lean_alloc_ctor(0, 0, 4);
lean_ctor_set_uint8(v___x_646_, 0, v___x_644_);
lean_ctor_set_uint8(v___x_646_, 1, v___x_535_);
lean_ctor_set_uint8(v___x_646_, 2, v___x_645_);
lean_ctor_set_uint8(v___x_646_, 3, v___x_535_);
v___x_647_ = lean_box(0);
v___x_648_ = l_Lean_MVarId_apply(v_a_567_, v___x_643_, v___x_646_, v___x_647_, v___x_581_, v___y_541_, v___y_542_, v___y_543_);
lean_dec_ref_known(v___x_581_, 7);
v___y_546_ = v___x_648_;
goto v___jp_545_;
}
}
}
else
{
lean_object* v_a_649_; lean_object* v___x_651_; uint8_t v_isShared_652_; uint8_t v_isSharedCheck_656_; 
lean_dec(v_a_602_);
lean_dec_ref(v_arg_589_);
lean_dec_ref(v_arg_588_);
lean_dec_ref_known(v___x_581_, 7);
lean_dec(v_a_567_);
lean_dec(v___y_543_);
lean_dec_ref(v___y_542_);
lean_dec(v___y_541_);
lean_dec_ref(v___y_540_);
lean_dec_ref(v___x_534_);
lean_dec_ref(v___x_533_);
lean_dec_ref(v___x_532_);
lean_dec(v___x_531_);
v_a_649_ = lean_ctor_get(v___x_603_, 0);
v_isSharedCheck_656_ = !lean_is_exclusive(v___x_603_);
if (v_isSharedCheck_656_ == 0)
{
v___x_651_ = v___x_603_;
v_isShared_652_ = v_isSharedCheck_656_;
goto v_resetjp_650_;
}
else
{
lean_inc(v_a_649_);
lean_dec(v___x_603_);
v___x_651_ = lean_box(0);
v_isShared_652_ = v_isSharedCheck_656_;
goto v_resetjp_650_;
}
v_resetjp_650_:
{
lean_object* v___x_654_; 
if (v_isShared_652_ == 0)
{
v___x_654_ = v___x_651_;
goto v_reusejp_653_;
}
else
{
lean_object* v_reuseFailAlloc_655_; 
v_reuseFailAlloc_655_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_655_, 0, v_a_649_);
v___x_654_ = v_reuseFailAlloc_655_;
goto v_reusejp_653_;
}
v_reusejp_653_:
{
return v___x_654_;
}
}
}
}
else
{
lean_object* v_a_657_; lean_object* v___x_659_; uint8_t v_isShared_660_; uint8_t v_isSharedCheck_664_; 
lean_dec_ref(v_arg_589_);
lean_dec_ref(v_arg_588_);
lean_dec_ref_known(v___x_581_, 7);
lean_dec(v_a_567_);
lean_dec(v___y_543_);
lean_dec_ref(v___y_542_);
lean_dec(v___y_541_);
lean_dec_ref(v___y_540_);
lean_dec_ref(v___x_534_);
lean_dec_ref(v___x_533_);
lean_dec_ref(v___x_532_);
lean_dec(v___x_531_);
v_a_657_ = lean_ctor_get(v___x_601_, 0);
v_isSharedCheck_664_ = !lean_is_exclusive(v___x_601_);
if (v_isSharedCheck_664_ == 0)
{
v___x_659_ = v___x_601_;
v_isShared_660_ = v_isSharedCheck_664_;
goto v_resetjp_658_;
}
else
{
lean_inc(v_a_657_);
lean_dec(v___x_601_);
v___x_659_ = lean_box(0);
v_isShared_660_ = v_isSharedCheck_664_;
goto v_resetjp_658_;
}
v_resetjp_658_:
{
lean_object* v___x_662_; 
if (v_isShared_660_ == 0)
{
v___x_662_ = v___x_659_;
goto v_reusejp_661_;
}
else
{
lean_object* v_reuseFailAlloc_663_; 
v_reuseFailAlloc_663_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_663_, 0, v_a_657_);
v___x_662_ = v_reuseFailAlloc_663_;
goto v_reusejp_661_;
}
v_reusejp_661_:
{
return v___x_662_;
}
}
}
}
}
}
else
{
lean_object* v___x_665_; 
lean_dec_ref(v___x_534_);
lean_dec_ref(v___x_533_);
lean_dec_ref(v___x_532_);
lean_dec(v___x_531_);
lean_inc_ref(v_a_583_);
v___x_665_ = lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0(v___x_530_, v_a_583_, v_a_567_, v_a_583_, v___x_581_, v___y_541_, v___y_542_, v___y_543_);
lean_dec_ref_known(v___x_581_, 7);
lean_dec_ref_known(v_a_583_, 2);
v___y_546_ = v___x_665_;
goto v___jp_545_;
}
}
else
{
lean_object* v___x_666_; 
lean_dec_ref(v___x_534_);
lean_dec_ref(v___x_533_);
lean_dec_ref(v___x_532_);
lean_dec(v___x_531_);
lean_inc_ref(v_a_583_);
v___x_666_ = lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0(v___x_530_, v_a_583_, v_a_567_, v_a_583_, v___x_581_, v___y_541_, v___y_542_, v___y_543_);
lean_dec_ref_known(v___x_581_, 7);
lean_dec_ref_known(v_a_583_, 2);
v___y_546_ = v___x_666_;
goto v___jp_545_;
}
}
else
{
lean_object* v___x_667_; 
lean_dec_ref(v___x_534_);
lean_dec_ref(v___x_533_);
lean_dec_ref(v___x_532_);
lean_dec(v___x_531_);
lean_inc_ref(v_a_583_);
v___x_667_ = lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0(v___x_530_, v_a_583_, v_a_567_, v_a_583_, v___x_581_, v___y_541_, v___y_542_, v___y_543_);
lean_dec_ref_known(v___x_581_, 7);
lean_dec_ref_known(v_a_583_, 2);
v___y_546_ = v___x_667_;
goto v___jp_545_;
}
}
else
{
lean_object* v___x_668_; 
lean_dec_ref(v___x_534_);
lean_dec_ref(v___x_533_);
lean_dec_ref(v___x_532_);
lean_dec(v___x_531_);
lean_inc_ref(v_a_583_);
v___x_668_ = lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0(v___x_530_, v_a_583_, v_a_567_, v_a_583_, v___x_581_, v___y_541_, v___y_542_, v___y_543_);
lean_dec_ref_known(v___x_581_, 7);
lean_dec_ref_known(v_a_583_, 2);
v___y_546_ = v___x_668_;
goto v___jp_545_;
}
}
case 7:
{
lean_object* v_binderType_669_; lean_object* v_body_670_; uint8_t v___x_671_; 
v_binderType_669_ = lean_ctor_get(v_a_583_, 1);
lean_inc_ref(v_binderType_669_);
v_body_670_ = lean_ctor_get(v_a_583_, 2);
lean_inc_ref(v_body_670_);
v___x_671_ = l_Lean_Expr_hasLooseBVars(v_body_670_);
if (v___x_671_ == 0)
{
lean_object* v___x_672_; lean_object* v___x_673_; 
lean_dec_ref_known(v_a_583_, 3);
v___x_672_ = lean_box(0);
v___x_673_ = lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1(v_binderType_669_, v_body_670_, v___x_531_, v___x_532_, v___x_533_, v___x_534_, v___x_535_, v_a_567_, v___x_530_, v___x_672_, v___x_581_, v___y_541_, v___y_542_, v___y_543_);
lean_dec_ref_known(v___x_581_, 7);
v___y_546_ = v___x_673_;
goto v___jp_545_;
}
else
{
lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v___x_678_; lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_681_; 
lean_inc_ref(v___x_530_);
v___x_674_ = l_Lean_Name_mkStr1(v___x_530_);
v___x_675_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0___closed__1);
v___x_676_ = l_Lean_MessageData_ofExpr(v_a_583_);
v___x_677_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_677_, 0, v___x_675_);
lean_ctor_set(v___x_677_, 1, v___x_676_);
v___x_678_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__10, &lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___closed__10);
v___x_679_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_679_, 0, v___x_677_);
lean_ctor_set(v___x_679_, 1, v___x_678_);
v___x_680_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_680_, 0, v___x_679_);
lean_inc(v_a_567_);
v___x_681_ = l_Lean_Meta_throwTacticEx___redArg(v___x_674_, v_a_567_, v___x_680_, v___x_581_, v___y_541_, v___y_542_, v___y_543_);
if (lean_obj_tag(v___x_681_) == 0)
{
lean_object* v_a_682_; lean_object* v___x_683_; 
v_a_682_ = lean_ctor_get(v___x_681_, 0);
lean_inc(v_a_682_);
lean_dec_ref_known(v___x_681_, 1);
v___x_683_ = lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__1(v_binderType_669_, v_body_670_, v___x_531_, v___x_532_, v___x_533_, v___x_534_, v___x_535_, v_a_567_, v___x_530_, v_a_682_, v___x_581_, v___y_541_, v___y_542_, v___y_543_);
lean_dec_ref_known(v___x_581_, 7);
v___y_546_ = v___x_683_;
goto v___jp_545_;
}
else
{
lean_dec_ref(v_body_670_);
lean_dec_ref(v_binderType_669_);
lean_dec_ref_known(v___x_581_, 7);
lean_dec(v_a_567_);
lean_dec(v___y_543_);
lean_dec_ref(v___y_542_);
lean_dec(v___y_541_);
lean_dec_ref(v___y_540_);
lean_dec_ref(v___x_534_);
lean_dec_ref(v___x_533_);
lean_dec_ref(v___x_532_);
lean_dec(v___x_531_);
lean_dec_ref(v___x_530_);
return v___x_681_;
}
}
}
default: 
{
lean_object* v___x_684_; 
lean_dec_ref(v___x_534_);
lean_dec_ref(v___x_533_);
lean_dec_ref(v___x_532_);
lean_dec(v___x_531_);
lean_inc(v_a_583_);
v___x_684_ = lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__0(v___x_530_, v_a_583_, v_a_567_, v_a_583_, v___x_581_, v___y_541_, v___y_542_, v___y_543_);
lean_dec_ref_known(v___x_581_, 7);
lean_dec(v_a_583_);
v___y_546_ = v___x_684_;
goto v___jp_545_;
}
}
}
else
{
lean_object* v_a_685_; lean_object* v___x_687_; uint8_t v_isShared_688_; uint8_t v_isSharedCheck_692_; 
lean_dec_ref_known(v___x_581_, 7);
lean_dec(v_a_567_);
lean_dec(v___y_543_);
lean_dec_ref(v___y_542_);
lean_dec(v___y_541_);
lean_dec_ref(v___y_540_);
lean_dec_ref(v___x_534_);
lean_dec_ref(v___x_533_);
lean_dec_ref(v___x_532_);
lean_dec(v___x_531_);
lean_dec_ref(v___x_530_);
v_a_685_ = lean_ctor_get(v___x_582_, 0);
v_isSharedCheck_692_ = !lean_is_exclusive(v___x_582_);
if (v_isSharedCheck_692_ == 0)
{
v___x_687_ = v___x_582_;
v_isShared_688_ = v_isSharedCheck_692_;
goto v_resetjp_686_;
}
else
{
lean_inc(v_a_685_);
lean_dec(v___x_582_);
v___x_687_ = lean_box(0);
v_isShared_688_ = v_isSharedCheck_692_;
goto v_resetjp_686_;
}
v_resetjp_686_:
{
lean_object* v___x_690_; 
if (v_isShared_688_ == 0)
{
v___x_690_ = v___x_687_;
goto v_reusejp_689_;
}
else
{
lean_object* v_reuseFailAlloc_691_; 
v_reuseFailAlloc_691_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_691_, 0, v_a_685_);
v___x_690_ = v_reuseFailAlloc_691_;
goto v_reusejp_689_;
}
v_reusejp_689_:
{
return v___x_690_;
}
}
}
}
else
{
lean_object* v_a_693_; lean_object* v___x_695_; uint8_t v_isShared_696_; uint8_t v_isSharedCheck_700_; 
lean_dec(v___y_543_);
lean_dec_ref(v___y_542_);
lean_dec(v___y_541_);
lean_dec_ref(v___y_540_);
lean_dec_ref(v___x_534_);
lean_dec_ref(v___x_533_);
lean_dec_ref(v___x_532_);
lean_dec(v___x_531_);
lean_dec_ref(v___x_530_);
v_a_693_ = lean_ctor_get(v___x_566_, 0);
v_isSharedCheck_700_ = !lean_is_exclusive(v___x_566_);
if (v_isSharedCheck_700_ == 0)
{
v___x_695_ = v___x_566_;
v_isShared_696_ = v_isSharedCheck_700_;
goto v_resetjp_694_;
}
else
{
lean_inc(v_a_693_);
lean_dec(v___x_566_);
v___x_695_ = lean_box(0);
v_isShared_696_ = v_isSharedCheck_700_;
goto v_resetjp_694_;
}
v_resetjp_694_:
{
lean_object* v___x_698_; 
if (v_isShared_696_ == 0)
{
v___x_698_ = v___x_695_;
goto v_reusejp_697_;
}
else
{
lean_object* v_reuseFailAlloc_699_; 
v_reuseFailAlloc_699_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_699_, 0, v_a_693_);
v___x_698_ = v_reuseFailAlloc_699_;
goto v_reusejp_697_;
}
v_reusejp_697_:
{
return v___x_698_;
}
}
}
v___jp_545_:
{
if (lean_obj_tag(v___y_546_) == 0)
{
lean_object* v_a_547_; lean_object* v___x_548_; 
v_a_547_ = lean_ctor_get(v___y_546_, 0);
lean_inc(v_a_547_);
lean_dec_ref_known(v___y_546_, 1);
v___x_548_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v_a_547_, v___y_537_, v___y_540_, v___y_541_, v___y_542_, v___y_543_);
lean_dec(v___y_543_);
lean_dec_ref(v___y_542_);
lean_dec(v___y_541_);
lean_dec_ref(v___y_540_);
if (lean_obj_tag(v___x_548_) == 0)
{
lean_object* v___x_550_; uint8_t v_isShared_551_; uint8_t v_isSharedCheck_556_; 
v_isSharedCheck_556_ = !lean_is_exclusive(v___x_548_);
if (v_isSharedCheck_556_ == 0)
{
lean_object* v_unused_557_; 
v_unused_557_ = lean_ctor_get(v___x_548_, 0);
lean_dec(v_unused_557_);
v___x_550_ = v___x_548_;
v_isShared_551_ = v_isSharedCheck_556_;
goto v_resetjp_549_;
}
else
{
lean_dec(v___x_548_);
v___x_550_ = lean_box(0);
v_isShared_551_ = v_isSharedCheck_556_;
goto v_resetjp_549_;
}
v_resetjp_549_:
{
lean_object* v___x_552_; lean_object* v___x_554_; 
v___x_552_ = lean_box(0);
if (v_isShared_551_ == 0)
{
lean_ctor_set(v___x_550_, 0, v___x_552_);
v___x_554_ = v___x_550_;
goto v_reusejp_553_;
}
else
{
lean_object* v_reuseFailAlloc_555_; 
v_reuseFailAlloc_555_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_555_, 0, v___x_552_);
v___x_554_ = v_reuseFailAlloc_555_;
goto v_reusejp_553_;
}
v_reusejp_553_:
{
return v___x_554_;
}
}
}
else
{
return v___x_548_;
}
}
else
{
lean_object* v_a_558_; lean_object* v___x_560_; uint8_t v_isShared_561_; uint8_t v_isSharedCheck_565_; 
lean_dec(v___y_543_);
lean_dec_ref(v___y_542_);
lean_dec(v___y_541_);
lean_dec_ref(v___y_540_);
v_a_558_ = lean_ctor_get(v___y_546_, 0);
v_isSharedCheck_565_ = !lean_is_exclusive(v___y_546_);
if (v_isSharedCheck_565_ == 0)
{
v___x_560_ = v___y_546_;
v_isShared_561_ = v_isSharedCheck_565_;
goto v_resetjp_559_;
}
else
{
lean_inc(v_a_558_);
lean_dec(v___y_546_);
v___x_560_ = lean_box(0);
v_isShared_561_ = v_isSharedCheck_565_;
goto v_resetjp_559_;
}
v_resetjp_559_:
{
lean_object* v___x_563_; 
if (v_isShared_561_ == 0)
{
v___x_563_ = v___x_560_;
goto v_reusejp_562_;
}
else
{
lean_object* v_reuseFailAlloc_564_; 
v_reuseFailAlloc_564_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_564_, 0, v_a_558_);
v___x_563_ = v_reuseFailAlloc_564_;
goto v_reusejp_562_;
}
v_reusejp_562_:
{
return v___x_563_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___boxed(lean_object* v___x_701_, lean_object* v___x_702_, lean_object* v___x_703_, lean_object* v___x_704_, lean_object* v___x_705_, lean_object* v___x_706_, lean_object* v___y_707_, lean_object* v___y_708_, lean_object* v___y_709_, lean_object* v___y_710_, lean_object* v___y_711_, lean_object* v___y_712_, lean_object* v___y_713_, lean_object* v___y_714_, lean_object* v___y_715_){
_start:
{
uint8_t v___x_8347__boxed_716_; lean_object* v_res_717_; 
v___x_8347__boxed_716_ = lean_unbox(v___x_706_);
v_res_717_ = lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2(v___x_701_, v___x_702_, v___x_703_, v___x_704_, v___x_705_, v___x_8347__boxed_716_, v___y_707_, v___y_708_, v___y_709_, v___y_710_, v___y_711_, v___y_712_, v___y_713_, v___y_714_);
lean_dec(v___y_710_);
lean_dec_ref(v___y_709_);
lean_dec(v___y_708_);
lean_dec_ref(v___y_707_);
return v_res_717_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1(lean_object* v_x_718_, lean_object* v_a_719_, lean_object* v_a_720_, lean_object* v_a_721_, lean_object* v_a_722_, lean_object* v_a_723_, lean_object* v_a_724_, lean_object* v_a_725_, lean_object* v_a_726_){
_start:
{
lean_object* v___x_728_; lean_object* v___x_729_; lean_object* v___x_730_; lean_object* v___x_731_; lean_object* v___x_732_; uint8_t v___x_733_; 
v___x_728_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__5_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4_));
v___x_729_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__6_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4_));
v___x_730_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__7_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4_));
v___x_731_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__0_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4_));
v___x_732_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__0));
lean_inc(v_x_718_);
v___x_733_ = l_Lean_Syntax_isOfKind(v_x_718_, v___x_732_);
if (v___x_733_ == 0)
{
lean_object* v___x_734_; 
lean_dec(v_x_718_);
v___x_734_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__0___redArg();
return v___x_734_;
}
else
{
lean_object* v___x_735_; lean_object* v___x_736_; lean_object* v___x_737_; uint8_t v___x_738_; 
v___x_735_ = lean_unsigned_to_nat(0u);
v___x_736_ = lean_unsigned_to_nat(1u);
v___x_737_ = l_Lean_Syntax_getArg(v_x_718_, v___x_736_);
lean_dec(v_x_718_);
v___x_738_ = l_Lean_Syntax_matchesNull(v___x_737_, v___x_735_);
if (v___x_738_ == 0)
{
lean_object* v___x_739_; 
v___x_739_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__0___redArg();
return v___x_739_;
}
else
{
lean_object* v___x_740_; lean_object* v___f_741_; lean_object* v___x_742_; 
v___x_740_ = lean_box(v___x_738_);
v___f_741_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___lam__2___boxed), 15, 6);
lean_closure_set(v___f_741_, 0, v___x_731_);
lean_closure_set(v___f_741_, 1, v___x_736_);
lean_closure_set(v___f_741_, 2, v___x_728_);
lean_closure_set(v___f_741_, 3, v___x_729_);
lean_closure_set(v___f_741_, 4, v___x_730_);
lean_closure_set(v___f_741_, 5, v___x_740_);
v___x_742_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_741_, v_a_719_, v_a_720_, v_a_721_, v_a_722_, v_a_723_, v_a_724_, v_a_725_, v_a_726_);
return v___x_742_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1___boxed(lean_object* v_x_743_, lean_object* v_a_744_, lean_object* v_a_745_, lean_object* v_a_746_, lean_object* v_a_747_, lean_object* v_a_748_, lean_object* v_a_749_, lean_object* v_a_750_, lean_object* v_a_751_, lean_object* v_a_752_){
_start:
{
lean_object* v_res_753_; 
v_res_753_ = lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1(v_x_743_, v_a_744_, v_a_745_, v_a_746_, v_a_747_, v_a_748_, v_a_749_, v_a_750_, v_a_751_);
lean_dec(v_a_751_);
lean_dec_ref(v_a_750_);
lean_dec(v_a_749_);
lean_dec_ref(v_a_748_);
lean_dec(v_a_747_);
lean_dec_ref(v_a_746_);
lean_dec(v_a_745_);
lean_dec_ref(v_a_744_);
return v_res_753_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_getM___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__1(lean_object* v_opt_754_, lean_object* v___y_755_, lean_object* v___y_756_, lean_object* v___y_757_, lean_object* v___y_758_){
_start:
{
lean_object* v___x_760_; 
v___x_760_ = lp_mathlib_Lean_Option_getM___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__1___redArg(v_opt_754_, v___y_757_);
return v___x_760_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_getM___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__1___boxed(lean_object* v_opt_761_, lean_object* v___y_762_, lean_object* v___y_763_, lean_object* v___y_764_, lean_object* v___y_765_, lean_object* v___y_766_){
_start:
{
lean_object* v_res_767_; 
v_res_767_ = lp_mathlib_Lean_Option_getM___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__1(v_opt_761_, v___y_762_, v___y_763_, v___y_764_, v___y_765_);
lean_dec(v___y_765_);
lean_dec_ref(v___y_764_);
lean_dec(v___y_763_);
lean_dec_ref(v___y_762_);
lean_dec_ref(v_opt_761_);
return v_res_767_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__3(void){
_start:
{
lean_object* v___x_777_; lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v___x_780_; 
v___x_777_ = l_Lean_Parser_Tactic_optConfig;
v___x_778_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__2));
v___x_779_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__2));
v___x_780_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_780_, 0, v___x_779_);
lean_ctor_set(v___x_780_, 1, v___x_778_);
lean_ctor_set(v___x_780_, 2, v___x_777_);
return v___x_780_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__4(void){
_start:
{
lean_object* v___x_781_; lean_object* v___x_782_; lean_object* v___x_783_; lean_object* v___x_784_; 
v___x_781_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__22));
v___x_782_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__3, &lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__3);
v___x_783_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__2));
v___x_784_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_784_, 0, v___x_783_);
lean_ctor_set(v___x_784_, 1, v___x_782_);
lean_ctor_set(v___x_784_, 2, v___x_781_);
return v___x_784_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__5(void){
_start:
{
lean_object* v___x_785_; lean_object* v___x_786_; lean_object* v___x_787_; lean_object* v___x_788_; 
v___x_785_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__4, &lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__4);
v___x_786_ = lean_unsigned_to_nat(1022u);
v___x_787_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__1));
v___x_788_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_788_, 0, v___x_787_);
lean_ctor_set(v___x_788_, 1, v___x_786_);
lean_ctor_set(v___x_788_, 2, v___x_785_);
return v___x_788_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21(void){
_start:
{
lean_object* v___x_789_; 
v___x_789_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__5, &lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__5);
return v___x_789_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__13(void){
_start:
{
lean_object* v___x_823_; lean_object* v___x_824_; lean_object* v___x_825_; lean_object* v___x_826_; 
v___x_823_ = l_Lean_Parser_Tactic_optConfig;
v___x_824_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__12));
v___x_825_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__2));
v___x_826_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_826_, 0, v___x_825_);
lean_ctor_set(v___x_826_, 1, v___x_824_);
lean_ctor_set(v___x_826_, 2, v___x_823_);
return v___x_826_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__14(void){
_start:
{
lean_object* v___x_827_; lean_object* v___x_828_; lean_object* v___x_829_; lean_object* v___x_830_; 
v___x_827_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__13, &lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__13_once, _init_lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__13);
v___x_828_ = lean_unsigned_to_nat(1022u);
v___x_829_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__10));
v___x_830_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_830_, 0, v___x_829_);
lean_ctor_set(v___x_830_, 1, v___x_828_);
lean_ctor_set(v___x_830_, 2, v___x_827_);
return v___x_830_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg__(void){
_start:
{
lean_object* v___x_831_; 
v___x_831_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__14, &lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__14_once, _init_lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__14);
return v___x_831_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules____private__Mathlib__Tactic__Contrapose__0__Mathlib__Tactic__Contrapose__tacticTry__push__neg____1(lean_object* v_x_836_, lean_object* v_a_837_, lean_object* v_a_838_, lean_object* v_a_839_, lean_object* v_a_840_, lean_object* v_a_841_, lean_object* v_a_842_, lean_object* v_a_843_, lean_object* v_a_844_){
_start:
{
lean_object* v___x_846_; uint8_t v___x_847_; 
v___x_846_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__10));
lean_inc(v_x_836_);
v___x_847_ = l_Lean_Syntax_isOfKind(v_x_836_, v___x_846_);
if (v___x_847_ == 0)
{
lean_object* v___x_848_; 
lean_dec(v_x_836_);
v___x_848_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules__Mathlib__Tactic__Contrapose__contrapose__1_spec__0___redArg();
return v___x_848_;
}
else
{
lean_object* v___x_849_; lean_object* v___x_850_; uint8_t v___x_851_; lean_object* v___x_852_; 
v___x_849_ = lean_unsigned_to_nat(1u);
v___x_850_ = l_Lean_Syntax_getArg(v_x_836_, v___x_849_);
lean_dec(v_x_836_);
v___x_851_ = 0;
v___x_852_ = lp_mathlib_Mathlib_Tactic_Push_elabPushConfig___redArg(v___x_850_, v___x_851_, v___x_847_, v_a_837_, v_a_843_, v_a_844_);
if (lean_obj_tag(v___x_852_) == 0)
{
lean_object* v_a_853_; lean_object* v___x_854_; lean_object* v___x_855_; lean_object* v___x_856_; lean_object* v___x_857_; uint8_t v___x_858_; uint8_t v___x_859_; lean_object* v___x_860_; 
v_a_853_ = lean_ctor_get(v___x_852_, 0);
lean_inc(v_a_853_);
lean_dec_ref_known(v___x_852_, 1);
v___x_854_ = lean_box(0);
v___x_855_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules____private__Mathlib__Tactic__Contrapose__0__Mathlib__Tactic__Contrapose__tacticTry__push__neg____1___closed__0));
v___x_856_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules____private__Mathlib__Tactic__Contrapose__0__Mathlib__Tactic__Contrapose__tacticTry__push__neg____1___closed__1));
v___x_857_ = lean_alloc_ctor(1, 1, 1);
lean_ctor_set(v___x_857_, 0, v___x_856_);
lean_ctor_set_uint8(v___x_857_, sizeof(void*)*1, v___x_847_);
v___x_858_ = 0;
v___x_859_ = lean_unbox(v_a_853_);
lean_dec(v_a_853_);
v___x_860_ = lp_mathlib_Mathlib_Tactic_Push_push(v___x_859_, v___x_854_, v___x_855_, v___x_857_, v___x_858_, v_a_837_, v_a_838_, v_a_839_, v_a_840_, v_a_841_, v_a_842_, v_a_843_, v_a_844_);
lean_dec_ref_known(v___x_857_, 1);
return v___x_860_;
}
else
{
lean_object* v_a_861_; lean_object* v___x_863_; uint8_t v_isShared_864_; uint8_t v_isSharedCheck_868_; 
v_a_861_ = lean_ctor_get(v___x_852_, 0);
v_isSharedCheck_868_ = !lean_is_exclusive(v___x_852_);
if (v_isSharedCheck_868_ == 0)
{
v___x_863_ = v___x_852_;
v_isShared_864_ = v_isSharedCheck_868_;
goto v_resetjp_862_;
}
else
{
lean_inc(v_a_861_);
lean_dec(v___x_852_);
v___x_863_ = lean_box(0);
v_isShared_864_ = v_isSharedCheck_868_;
goto v_resetjp_862_;
}
v_resetjp_862_:
{
lean_object* v___x_866_; 
if (v_isShared_864_ == 0)
{
v___x_866_ = v___x_863_;
goto v_reusejp_865_;
}
else
{
lean_object* v_reuseFailAlloc_867_; 
v_reuseFailAlloc_867_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_867_, 0, v_a_861_);
v___x_866_ = v_reuseFailAlloc_867_;
goto v_reusejp_865_;
}
v_reusejp_865_:
{
return v___x_866_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules____private__Mathlib__Tactic__Contrapose__0__Mathlib__Tactic__Contrapose__tacticTry__push__neg____1___boxed(lean_object* v_x_869_, lean_object* v_a_870_, lean_object* v_a_871_, lean_object* v_a_872_, lean_object* v_a_873_, lean_object* v_a_874_, lean_object* v_a_875_, lean_object* v_a_876_, lean_object* v_a_877_, lean_object* v_a_878_){
_start:
{
lean_object* v_res_879_; 
v_res_879_ = lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______elabRules____private__Mathlib__Tactic__Contrapose__0__Mathlib__Tactic__Contrapose__tacticTry__push__neg____1(v_x_869_, v_a_870_, v_a_871_, v_a_872_, v_a_873_, v_a_874_, v_a_875_, v_a_876_, v_a_877_);
lean_dec(v_a_877_);
lean_dec_ref(v_a_876_);
lean_dec(v_a_875_);
lean_dec_ref(v_a_874_);
lean_dec(v_a_873_);
lean_dec_ref(v_a_872_);
lean_dec(v_a_871_);
lean_dec_ref(v_a_870_);
return v_res_879_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose_x21__1(lean_object* v_x_886_, lean_object* v_a_887_, lean_object* v_a_888_){
_start:
{
lean_object* v___y_890_; lean_object* v___x_893_; lean_object* v___x_894_; uint8_t v___x_895_; 
v___x_893_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__0));
v___x_894_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21___closed__1));
lean_inc(v_x_886_);
v___x_895_ = l_Lean_Syntax_isOfKind(v_x_886_, v___x_894_);
if (v___x_895_ == 0)
{
lean_object* v___x_896_; lean_object* v___x_897_; 
lean_dec(v_x_886_);
v___x_896_ = lean_box(1);
v___x_897_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_897_, 0, v___x_896_);
lean_ctor_set(v___x_897_, 1, v_a_888_);
return v___x_897_;
}
else
{
lean_object* v___x_898_; lean_object* v___x_899_; lean_object* v___x_900_; lean_object* v___x_901_; lean_object* v___x_902_; uint8_t v___x_903_; 
v___x_898_ = lean_unsigned_to_nat(0u);
v___x_899_ = lean_unsigned_to_nat(1u);
v___x_900_ = l_Lean_Syntax_getArg(v_x_886_, v___x_899_);
v___x_901_ = lean_unsigned_to_nat(2u);
v___x_902_ = l_Lean_Syntax_getArg(v_x_886_, v___x_901_);
lean_dec(v_x_886_);
lean_inc(v___x_902_);
v___x_903_ = l_Lean_Syntax_matchesNull(v___x_902_, v___x_898_);
if (v___x_903_ == 0)
{
lean_object* v___x_904_; uint8_t v___x_905_; 
v___x_904_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose_x21__1___closed__1));
lean_inc(v___x_900_);
v___x_905_ = l_Lean_Syntax_isOfKind(v___x_900_, v___x_904_);
if (v___x_905_ == 0)
{
lean_dec(v___x_902_);
lean_dec(v___x_900_);
v___y_890_ = v_a_888_;
goto v___jp_889_;
}
else
{
uint8_t v___x_906_; 
lean_inc(v___x_902_);
v___x_906_ = l_Lean_Syntax_matchesNull(v___x_902_, v___x_901_);
if (v___x_906_ == 0)
{
lean_dec(v___x_902_);
lean_dec(v___x_900_);
v___y_890_ = v_a_888_;
goto v___jp_889_;
}
else
{
lean_object* v___x_907_; lean_object* v___x_908_; uint8_t v___x_909_; 
v___x_907_ = l_Lean_Syntax_getArg(v___x_902_, v___x_898_);
v___x_908_ = l_Lean_Syntax_getArg(v___x_902_, v___x_899_);
lean_dec(v___x_902_);
lean_inc(v___x_908_);
v___x_909_ = l_Lean_Syntax_matchesNull(v___x_908_, v___x_898_);
if (v___x_909_ == 0)
{
uint8_t v___x_910_; 
lean_inc(v___x_908_);
v___x_910_ = l_Lean_Syntax_matchesNull(v___x_908_, v___x_901_);
if (v___x_910_ == 0)
{
lean_dec(v___x_908_);
lean_dec(v___x_907_);
lean_dec(v___x_900_);
v___y_890_ = v_a_888_;
goto v___jp_889_;
}
else
{
lean_object* v_ref_911_; lean_object* v___x_912_; lean_object* v___x_913_; lean_object* v___x_914_; lean_object* v___x_915_; lean_object* v___x_916_; lean_object* v___x_917_; lean_object* v___x_918_; lean_object* v___x_919_; lean_object* v___x_920_; lean_object* v___x_921_; lean_object* v___x_922_; lean_object* v___x_923_; lean_object* v___x_924_; lean_object* v___x_925_; lean_object* v___x_926_; lean_object* v___x_927_; lean_object* v___x_928_; lean_object* v___x_929_; lean_object* v___x_930_; lean_object* v___x_931_; lean_object* v___x_932_; lean_object* v___x_933_; lean_object* v___x_934_; lean_object* v___x_935_; lean_object* v___x_936_; lean_object* v___x_937_; lean_object* v___x_938_; lean_object* v___x_939_; lean_object* v___x_940_; lean_object* v___x_941_; lean_object* v___x_942_; 
v_ref_911_ = lean_ctor_get(v_a_887_, 5);
v___x_912_ = l_Lean_Syntax_getArg(v___x_908_, v___x_899_);
lean_dec(v___x_908_);
v___x_913_ = l_Lean_SourceInfo_fromRef(v_ref_911_, v___x_909_);
v___x_914_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__3));
v___x_915_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__4));
lean_inc_n(v___x_913_, 15);
v___x_916_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_916_, 0, v___x_913_);
lean_ctor_set(v___x_916_, 1, v___x_915_);
v___x_917_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__6));
v___x_918_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__8));
v___x_919_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__10));
v___x_920_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__11));
v___x_921_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__12));
v___x_922_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_922_, 0, v___x_913_);
lean_ctor_set(v___x_922_, 1, v___x_920_);
v___x_923_ = l_Lean_Syntax_node1(v___x_913_, v___x_919_, v___x_907_);
v___x_924_ = l_Lean_Syntax_node2(v___x_913_, v___x_921_, v___x_922_, v___x_923_);
v___x_925_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__13));
v___x_926_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_926_, 0, v___x_913_);
lean_ctor_set(v___x_926_, 1, v___x_925_);
v___x_927_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_927_, 0, v___x_913_);
lean_ctor_set(v___x_927_, 1, v___x_893_);
v___x_928_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__14, &lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__14);
v___x_929_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_929_, 0, v___x_913_);
lean_ctor_set(v___x_929_, 1, v___x_919_);
lean_ctor_set(v___x_929_, 2, v___x_928_);
v___x_930_ = l_Lean_Syntax_node3(v___x_913_, v___x_894_, v___x_927_, v___x_900_, v___x_929_);
v___x_931_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__15));
v___x_932_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__16));
v___x_933_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_933_, 0, v___x_913_);
lean_ctor_set(v___x_933_, 1, v___x_931_);
v___x_934_ = l_Lean_Syntax_node1(v___x_913_, v___x_919_, v___x_912_);
v___x_935_ = l_Lean_Syntax_node2(v___x_913_, v___x_932_, v___x_933_, v___x_934_);
lean_inc_ref(v___x_926_);
v___x_936_ = l_Lean_Syntax_node5(v___x_913_, v___x_919_, v___x_924_, v___x_926_, v___x_930_, v___x_926_, v___x_935_);
v___x_937_ = l_Lean_Syntax_node1(v___x_913_, v___x_918_, v___x_936_);
v___x_938_ = l_Lean_Syntax_node1(v___x_913_, v___x_917_, v___x_937_);
v___x_939_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__17));
v___x_940_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_940_, 0, v___x_913_);
lean_ctor_set(v___x_940_, 1, v___x_939_);
v___x_941_ = l_Lean_Syntax_node3(v___x_913_, v___x_914_, v___x_916_, v___x_938_, v___x_940_);
v___x_942_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_942_, 0, v___x_941_);
lean_ctor_set(v___x_942_, 1, v_a_888_);
return v___x_942_;
}
}
else
{
lean_object* v_ref_943_; lean_object* v___x_944_; lean_object* v___x_945_; lean_object* v___x_946_; lean_object* v___x_947_; lean_object* v___x_948_; lean_object* v___x_949_; lean_object* v___x_950_; lean_object* v___x_951_; lean_object* v___x_952_; lean_object* v___x_953_; lean_object* v___x_954_; lean_object* v___x_955_; lean_object* v___x_956_; lean_object* v___x_957_; lean_object* v___x_958_; lean_object* v___x_959_; lean_object* v___x_960_; lean_object* v___x_961_; lean_object* v___x_962_; lean_object* v___x_963_; lean_object* v___x_964_; lean_object* v___x_965_; lean_object* v___x_966_; lean_object* v___x_967_; lean_object* v___x_968_; lean_object* v___x_969_; lean_object* v___x_970_; lean_object* v___x_971_; lean_object* v___x_972_; 
lean_dec(v___x_908_);
v_ref_943_ = lean_ctor_get(v_a_887_, 5);
v___x_944_ = l_Lean_SourceInfo_fromRef(v_ref_943_, v___x_903_);
v___x_945_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__3));
v___x_946_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__4));
lean_inc_n(v___x_944_, 14);
v___x_947_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_947_, 0, v___x_944_);
lean_ctor_set(v___x_947_, 1, v___x_946_);
v___x_948_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__6));
v___x_949_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__8));
v___x_950_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__10));
v___x_951_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__11));
v___x_952_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__12));
v___x_953_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_953_, 0, v___x_944_);
lean_ctor_set(v___x_953_, 1, v___x_951_);
v___x_954_ = l_Lean_Syntax_node1(v___x_944_, v___x_950_, v___x_907_);
lean_inc(v___x_954_);
v___x_955_ = l_Lean_Syntax_node2(v___x_944_, v___x_952_, v___x_953_, v___x_954_);
v___x_956_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__13));
v___x_957_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_957_, 0, v___x_944_);
lean_ctor_set(v___x_957_, 1, v___x_956_);
v___x_958_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_958_, 0, v___x_944_);
lean_ctor_set(v___x_958_, 1, v___x_893_);
v___x_959_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__14, &lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__14);
v___x_960_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_960_, 0, v___x_944_);
lean_ctor_set(v___x_960_, 1, v___x_950_);
lean_ctor_set(v___x_960_, 2, v___x_959_);
v___x_961_ = l_Lean_Syntax_node3(v___x_944_, v___x_894_, v___x_958_, v___x_900_, v___x_960_);
v___x_962_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__15));
v___x_963_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__16));
v___x_964_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_964_, 0, v___x_944_);
lean_ctor_set(v___x_964_, 1, v___x_962_);
v___x_965_ = l_Lean_Syntax_node2(v___x_944_, v___x_963_, v___x_964_, v___x_954_);
lean_inc_ref(v___x_957_);
v___x_966_ = l_Lean_Syntax_node5(v___x_944_, v___x_950_, v___x_955_, v___x_957_, v___x_961_, v___x_957_, v___x_965_);
v___x_967_ = l_Lean_Syntax_node1(v___x_944_, v___x_949_, v___x_966_);
v___x_968_ = l_Lean_Syntax_node1(v___x_944_, v___x_948_, v___x_967_);
v___x_969_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__17));
v___x_970_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_970_, 0, v___x_944_);
lean_ctor_set(v___x_970_, 1, v___x_969_);
v___x_971_ = l_Lean_Syntax_node3(v___x_944_, v___x_945_, v___x_947_, v___x_968_, v___x_970_);
v___x_972_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_972_, 0, v___x_971_);
lean_ctor_set(v___x_972_, 1, v_a_888_);
return v___x_972_;
}
}
}
}
else
{
lean_object* v_ref_973_; uint8_t v___x_974_; lean_object* v___x_975_; lean_object* v___x_976_; lean_object* v___x_977_; lean_object* v___x_978_; lean_object* v___x_979_; lean_object* v___x_980_; lean_object* v___x_981_; lean_object* v___x_982_; lean_object* v___x_983_; lean_object* v___x_984_; lean_object* v___x_985_; lean_object* v___x_986_; lean_object* v___x_987_; lean_object* v___x_988_; lean_object* v___x_989_; lean_object* v___x_990_; lean_object* v___x_991_; lean_object* v___x_992_; lean_object* v___x_993_; lean_object* v___x_994_; lean_object* v___x_995_; lean_object* v___x_996_; lean_object* v___x_997_; lean_object* v___x_998_; lean_object* v___x_999_; lean_object* v___x_1000_; 
lean_dec(v___x_902_);
v_ref_973_ = lean_ctor_get(v_a_887_, 5);
v___x_974_ = 0;
v___x_975_ = l_Lean_SourceInfo_fromRef(v_ref_973_, v___x_974_);
v___x_976_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__3));
v___x_977_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__4));
lean_inc_n(v___x_975_, 11);
v___x_978_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_978_, 0, v___x_975_);
lean_ctor_set(v___x_978_, 1, v___x_977_);
v___x_979_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__6));
v___x_980_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__8));
v___x_981_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__10));
v___x_982_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn___closed__0_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4_));
v___x_983_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose_contrapose___closed__0));
v___x_984_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_984_, 0, v___x_975_);
lean_ctor_set(v___x_984_, 1, v___x_982_);
v___x_985_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__14, &lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__14);
v___x_986_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_986_, 0, v___x_975_);
lean_ctor_set(v___x_986_, 1, v___x_981_);
lean_ctor_set(v___x_986_, 2, v___x_985_);
v___x_987_ = l_Lean_Syntax_node2(v___x_975_, v___x_983_, v___x_984_, v___x_986_);
v___x_988_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__13));
v___x_989_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_989_, 0, v___x_975_);
lean_ctor_set(v___x_989_, 1, v___x_988_);
v___x_990_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__10));
v___x_991_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg___00__closed__11));
v___x_992_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_992_, 0, v___x_975_);
lean_ctor_set(v___x_992_, 1, v___x_991_);
v___x_993_ = l_Lean_Syntax_node2(v___x_975_, v___x_990_, v___x_992_, v___x_900_);
v___x_994_ = l_Lean_Syntax_node3(v___x_975_, v___x_981_, v___x_987_, v___x_989_, v___x_993_);
v___x_995_ = l_Lean_Syntax_node1(v___x_975_, v___x_980_, v___x_994_);
v___x_996_ = l_Lean_Syntax_node1(v___x_975_, v___x_979_, v___x_995_);
v___x_997_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose__1___closed__17));
v___x_998_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_998_, 0, v___x_975_);
lean_ctor_set(v___x_998_, 1, v___x_997_);
v___x_999_ = l_Lean_Syntax_node3(v___x_975_, v___x_976_, v___x_978_, v___x_996_, v___x_998_);
v___x_1000_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1000_, 0, v___x_999_);
lean_ctor_set(v___x_1000_, 1, v_a_888_);
return v___x_1000_;
}
}
v___jp_889_:
{
lean_object* v___x_891_; lean_object* v___x_892_; 
v___x_891_ = lean_box(1);
v___x_892_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_892_, 0, v___x_891_);
lean_ctor_set(v___x_892_, 1, v___y_890_);
return v___x_892_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose_x21__1___boxed(lean_object* v_x_1001_, lean_object* v_a_1002_, lean_object* v_a_1003_){
_start:
{
lean_object* v_res_1004_; 
v_res_1004_ = lp_mathlib_Mathlib_Tactic_Contrapose___aux__Mathlib__Tactic__Contrapose______macroRules__Mathlib__Tactic__Contrapose__contrapose_x21__1(v_x_1001_, v_a_1002_, v_a_1003_);
lean_dec_ref(v_a_1002_);
return v_res_1004_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Push(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Contrapose(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Push(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Contrapose(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_initFn_00___x40_Mathlib_Tactic_Contrapose_2719866683____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_negate__iff = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_negate__iff);
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21 = _init_lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Contrapose_contrapose_x21);
lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg__ = _init_lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg__();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Contrapose_0__Mathlib_Tactic_Contrapose_tacticTry__push__neg__);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Push(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Contrapose(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Push(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Contrapose(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Contrapose(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Contrapose(builtin);
}
#ifdef __cplusplus
}
#endif
