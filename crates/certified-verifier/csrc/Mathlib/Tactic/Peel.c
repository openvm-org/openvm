// Lean compiler output
// Module: Mathlib.Tactic.Peel
// Imports: public import Init public meta import Init public meta import Mathlib.Tactic.Basic public import Mathlib.Order.Filter.Basic public meta import Mathlib.Tactic.ToAdditive public meta import Mathlib.Tactic.ToDual public import Mathlib.Tactic.Basic
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
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instInhabitedMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_FVarId_getDecl___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_addZetaDeltaFVarId___redArg(lean_object*, lean_object*);
uint8_t l_Lean_LocalDecl_isImplementationDetail(lean_object*);
lean_object* l_Lean_Meta_Context_config(lean_object*);
uint8_t l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_MetavarContext_getExprAssignmentCore_x3f(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_WHNF_0__Lean_Meta_whnfCore_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFn(lean_object*);
lean_object* l_Lean_Name_beq___boxed(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_List_elem___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_unfoldDefinition_x3f(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_head_x3f___redArg(lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_whnfR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_introNCore(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* lp_batteries_Lean_MVarId_assignIfDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_MVarId_applyConst(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFnArgs(lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Core_mkFreshUserName(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
uint8_t l_Lean_Expr_isConst(lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
lean_object* l_Lean_Expr_bindingName_x21(lean_object*);
uint8_t l_Lean_Expr_isForall(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Mathlib_Tactic_usingArg;
lean_object* l_Lean_MVarId_clear(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_saveState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_SavedState_restore___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_elabTermForApply(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_Elab_Tactic_getNameOfIdent_x27(lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_to_list(lean_object*);
uint8_t l_List_isEmpty___redArg(lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainTarget(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getNat(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t l_Lean_Syntax_isIdent(lean_object*);
lean_object* l_Lean_Syntax_getId(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_MVarId_intro(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_replicateTR___redArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
lean_object* l_Lean_Syntax_getNumArgs(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* l_Array_extract___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray2___redArg(lean_object*, lean_object*);
lean_object* l_Array_mkArray1___redArg(lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Peel"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "peel"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__2_value),LEAN_SCALAR_PTR_LITERAL(90, 248, 111, 228, 249, 93, 69, 215)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__3_value),LEAN_SCALAR_PTR_LITERAL(159, 237, 214, 177, 63, 164, 177, 235)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__5_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__3_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__8_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "num"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__10_value),LEAN_SCALAR_PTR_LITERAL(227, 68, 22, 222, 47, 51, 204, 84)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__15_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "colGt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__18_value),LEAN_SCALAR_PTR_LITERAL(185, 236, 32, 153, 169, 213, 53, 244)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__17_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__21_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__22_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__23_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__23_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__21_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__24_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__25_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__14_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__26_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__27_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " with"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__28_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__28_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__29_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "many1"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__30_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__30_value),LEAN_SCALAR_PTR_LITERAL(55, 136, 52, 6, 12, 19, 78, 239)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__31_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "orelse"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__32_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__32_value),LEAN_SCALAR_PTR_LITERAL(78, 76, 4, 51, 251, 212, 116, 5)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__33_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__34_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__34_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__35_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__35_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__36_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hole"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__37_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__37_value),LEAN_SCALAR_PTR_LITERAL(247, 163, 83, 191, 48, 55, 64, 87)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__38_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__38_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__39_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__33_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__36_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__39_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__40_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__21_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__40_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__41_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__31_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__41_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__42_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__29_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__42_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__43 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__43_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__43_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__44 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__44_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peel___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__27_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__44_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__45 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__45_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Peel_peel___closed__46_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__46;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Peel_peel___closed__47_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__47;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Peel_peel___closed__48_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel___closed__48;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peel;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Exists"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__0_value),LEAN_SCALAR_PTR_LITERAL(65, 29, 48, 135, 199, 176, 149, 70)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "And"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__2_value),LEAN_SCALAR_PTR_LITERAL(49, 220, 212, 156, 122, 214, 55, 135)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Filter"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Eventually"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__4_value),LEAN_SCALAR_PTR_LITERAL(153, 224, 166, 105, 193, 53, 208, 235)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__5_value),LEAN_SCALAR_PTR_LITERAL(170, 201, 114, 242, 122, 67, 34, 48)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Frequently"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__4_value),LEAN_SCALAR_PTR_LITERAL(153, 224, 166, 105, 193, 53, 208, 235)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__7_value),LEAN_SCALAR_PTR_LITERAL(46, 70, 186, 141, 48, 174, 129, 118)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__1_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__12_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Peel_quantifiers = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__12_value;
static const lean_closure_object lp_mathlib_panic___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instInhabitedMetaM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__1___closed__0 = (const lean_object*)&lp_mathlib_panic___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__3___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "loose bvar in expression"};
static const lean_object* lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2___closed__2 = (const lean_object*)&lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2___closed__2_value;
static const lean_string_object lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "Lean.Meta.whnfEasyCases"};
static const lean_object* lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "Lean.Meta.WHNF"};
static const lean_object* lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2___closed__3;
static const lean_closure_object lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Name_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2___closed__4 = (const lean_object*)&lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_whnfQuantifier(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_whnfQuantifier___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Peel_throwPeelError_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Peel_throwPeelError_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Peel_throwPeelError_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Peel_throwPeelError_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 45, .m_capacity = 45, .m_length = 44, .m_data = "Tactic 'peel' could not match quantifiers in"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "\nand"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_throwPeelError(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Peel_throwPeelError_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Peel_throwPeelError_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_mkFreshBinderName___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "a"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_mkFreshBinderName___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_mkFreshBinderName___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_mkFreshBinderName___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_mkFreshBinderName___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(247, 80, 99, 121, 74, 33, 203, 108)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_mkFreshBinderName___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_mkFreshBinderName___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_mkFreshBinderName___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_mkFreshBinderName___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_mkFreshBinderName(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_mkFreshBinderName___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_applyPeelThm___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "peel: internal error"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_applyPeelThm___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_applyPeelThm___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Peel_applyPeelThm___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Peel_applyPeelThm___closed__1;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_applyPeelThm___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 8, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(0, 1, 0, 1, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_applyPeelThm___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_applyPeelThm___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_applyPeelThm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_applyPeelThm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Peel_peelCore_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Peel_peelCore_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Peel_peelCore_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Peel_peelCore_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "forall_imp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(85, 122, 221, 73, 222, 90, 200, 133)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "eventually_imp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__2_value),LEAN_SCALAR_PTR_LITERAL(90, 248, 111, 228, 249, 93, 69, 215)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(118, 90, 231, 210, 71, 65, 153, 182)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "frequently_imp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__2_value),LEAN_SCALAR_PTR_LITERAL(90, 248, 111, 228, 249, 93, 69, 215)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__5_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(239, 65, 35, 79, 159, 180, 72, 177)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "imp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__0_value),LEAN_SCALAR_PTR_LITERAL(65, 29, 48, 135, 199, 176, 149, 70)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__6_value),LEAN_SCALAR_PTR_LITERAL(201, 103, 59, 186, 19, 185, 194, 98)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "and_imp_left_of_imp_imp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__9_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__2_value),LEAN_SCALAR_PTR_LITERAL(90, 248, 111, 228, 249, 93, 69, 215)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__9_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__8_value),LEAN_SCALAR_PTR_LITERAL(214, 185, 177, 47, 104, 168, 156, 63)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "p"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__10_value),LEAN_SCALAR_PTR_LITERAL(34, 153, 146, 175, 179, 220, 230, 134)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelCore(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Peel_peelArgs_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Peel_peelArgs_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Peel_peelArgs_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Peel_peelArgs_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peelArgs___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "this"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelArgs___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelArgs___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelArgs___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelArgs___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(38, 116, 214, 236, 212, 160, 188, 150)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelArgs___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelArgs___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelArgs___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelArgs___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelArgs___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelArgs___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelArgs(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelArgs___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelUnbounded___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelUnbounded___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelUnbounded___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelUnbounded___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelUnbounded___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelUnbounded___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelUnbounded(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelUnbounded___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "focus"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__3_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__2_value),LEAN_SCALAR_PTR_LITERAL(198, 223, 207, 6, 131, 57, 182, 221)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__5_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__4_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__7_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__7_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "first"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__11_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__11_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__10_value),LEAN_SCALAR_PTR_LITERAL(59, 232, 35, 17, 172, 62, 48, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__12_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "apply"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__16_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__16_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__15_value),LEAN_SCALAR_PTR_LITERAL(202, 125, 237, 78, 179, 140, 218, 80)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "forall_congr'"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__17_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__18;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__17_value),LEAN_SCALAR_PTR_LITERAL(63, 255, 57, 163, 42, 11, 214, 176)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__19_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__20_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__21_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "exists_congr"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__22_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__23;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__22_value),LEAN_SCALAR_PTR_LITERAL(233, 95, 142, 123, 147, 63, 142, 134)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__24_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__25_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__26_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "eventually_congr"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__27_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__28;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__27_value),LEAN_SCALAR_PTR_LITERAL(182, 2, 154, 2, 193, 58, 167, 200)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__30_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__30_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__30_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__30_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__30_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__2_value),LEAN_SCALAR_PTR_LITERAL(90, 248, 111, 228, 249, 93, 69, 215)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__30_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__27_value),LEAN_SCALAR_PTR_LITERAL(43, 222, 115, 199, 245, 5, 50, 28)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__30_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__30_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__31_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__31_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__32_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "frequently_congr"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__33_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__34_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__34;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__33_value),LEAN_SCALAR_PTR_LITERAL(65, 151, 201, 112, 122, 2, 220, 93)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__35_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__36_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__36_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__36_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__36_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__36_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__2_value),LEAN_SCALAR_PTR_LITERAL(90, 248, 111, 228, 249, 93, 69, 215)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__36_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__33_value),LEAN_SCALAR_PTR_LITERAL(76, 205, 111, 127, 248, 85, 1, 40)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__36_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__36_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__37_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__37_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__38_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "and_congr_right"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__39_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__40_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__40;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__39_value),LEAN_SCALAR_PTR_LITERAL(164, 138, 87, 174, 211, 189, 52, 112)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__41_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__41_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__42_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__42_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__43 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__43_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "fail"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__44 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__44_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__45_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__45_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__45_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__45_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__45_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__45_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__44_value),LEAN_SCALAR_PTR_LITERAL(251, 214, 242, 89, 226, 36, 213, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__45 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__45_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "str"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__46 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__46_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__46_value),LEAN_SCALAR_PTR_LITERAL(255, 188, 142, 1, 190, 33, 34, 128)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__47 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__47_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 49, .m_capacity = 49, .m_length = 48, .m_data = "\"failed to apply a quantifier congruence lemma.\""};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__48 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__48_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelArgsIff___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelArgsIff___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelArgsIff(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelArgsIff___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1___lam__0(lean_object*, uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(168, 60, 211, 188, 58, 220, 100, 184)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ";"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "exact"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "with"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__2_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "usingArg"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(61, 85, 12, 81, 71, 87, 118, 61)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "seq1"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__7_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__1_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__7_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(242, 140, 137, 56, 141, 11, 143, 117)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Peel_peel___closed__46(void){
_start:
{
lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; 
v___x_103_ = lp_mathlib_Mathlib_Tactic_usingArg;
v___x_104_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peel___closed__9));
v___x_105_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_105_, 0, v___x_104_);
lean_ctor_set(v___x_105_, 1, v___x_103_);
return v___x_105_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Peel_peel___closed__47(void){
_start:
{
lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; 
v___x_106_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__46, &lp_mathlib_Mathlib_Tactic_Peel_peel___closed__46_once, _init_lp_mathlib_Mathlib_Tactic_Peel_peel___closed__46);
v___x_107_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peel___closed__45));
v___x_108_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peel___closed__6));
v___x_109_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_109_, 0, v___x_108_);
lean_ctor_set(v___x_109_, 1, v___x_107_);
lean_ctor_set(v___x_109_, 2, v___x_106_);
return v___x_109_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Peel_peel___closed__48(void){
_start:
{
lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; 
v___x_110_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__47, &lp_mathlib_Mathlib_Tactic_Peel_peel___closed__47_once, _init_lp_mathlib_Mathlib_Tactic_Peel_peel___closed__47);
v___x_111_ = lean_unsigned_to_nat(1022u);
v___x_112_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peel___closed__4));
v___x_113_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_113_, 0, v___x_112_);
lean_ctor_set(v___x_113_, 1, v___x_111_);
lean_ctor_set(v___x_113_, 2, v___x_110_);
return v___x_113_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Peel_peel(void){
_start:
{
lean_object* v___x_114_; 
v___x_114_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Peel_peel___closed__48, &lp_mathlib_Mathlib_Tactic_Peel_peel___closed__48_once, _init_lp_mathlib_Mathlib_Tactic_Peel_peel___closed__48);
return v___x_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__1(lean_object* v_msg_144_, lean_object* v___y_145_, lean_object* v___y_146_, lean_object* v___y_147_, lean_object* v___y_148_){
_start:
{
lean_object* v___f_150_; lean_object* v___x_803__overap_151_; lean_object* v___x_152_; 
v___f_150_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__1___closed__0));
v___x_803__overap_151_ = lean_panic_fn_borrowed(v___f_150_, v_msg_144_);
lean_inc(v___y_148_);
lean_inc_ref(v___y_147_);
lean_inc(v___y_146_);
lean_inc_ref(v___y_145_);
v___x_152_ = lean_apply_5(v___x_803__overap_151_, v___y_145_, v___y_146_, v___y_147_, v___y_148_, lean_box(0));
return v___x_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__1___boxed(lean_object* v_msg_153_, lean_object* v___y_154_, lean_object* v___y_155_, lean_object* v___y_156_, lean_object* v___y_157_, lean_object* v___y_158_){
_start:
{
lean_object* v_res_159_; 
v_res_159_ = lp_mathlib_panic___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__1(v_msg_153_, v___y_154_, v___y_155_, v___y_156_, v___y_157_);
lean_dec(v___y_157_);
lean_dec_ref(v___y_156_);
lean_dec(v___y_155_);
lean_dec_ref(v___y_154_);
return v_res_159_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__3___redArg(lean_object* v_k_160_, lean_object* v_t_161_){
_start:
{
if (lean_obj_tag(v_t_161_) == 0)
{
lean_object* v_k_162_; lean_object* v_l_163_; lean_object* v_r_164_; uint8_t v___x_165_; 
v_k_162_ = lean_ctor_get(v_t_161_, 1);
v_l_163_ = lean_ctor_get(v_t_161_, 3);
v_r_164_ = lean_ctor_get(v_t_161_, 4);
v___x_165_ = l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(v_k_160_, v_k_162_);
switch(v___x_165_)
{
case 0:
{
v_t_161_ = v_l_163_;
goto _start;
}
case 1:
{
uint8_t v___x_167_; 
v___x_167_ = 1;
return v___x_167_;
}
default: 
{
v_t_161_ = v_r_164_;
goto _start;
}
}
}
else
{
uint8_t v___x_169_; 
v___x_169_ = 0;
return v___x_169_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__3___redArg___boxed(lean_object* v_k_170_, lean_object* v_t_171_){
_start:
{
uint8_t v_res_172_; lean_object* v_r_173_; 
v_res_172_ = lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__3___redArg(v_k_170_, v_t_171_);
lean_dec(v_t_171_);
lean_dec(v_k_170_);
v_r_173_ = lean_box(v_res_172_);
return v_r_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__4___redArg(lean_object* v_mvarId_174_, lean_object* v___y_175_){
_start:
{
lean_object* v___x_177_; lean_object* v_mctx_178_; lean_object* v___x_179_; lean_object* v___x_180_; 
v___x_177_ = lean_st_ref_get(v___y_175_);
v_mctx_178_ = lean_ctor_get(v___x_177_, 0);
lean_inc_ref(v_mctx_178_);
lean_dec(v___x_177_);
v___x_179_ = l_Lean_MetavarContext_getExprAssignmentCore_x3f(v_mctx_178_, v_mvarId_174_);
lean_dec_ref(v_mctx_178_);
v___x_180_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_180_, 0, v___x_179_);
return v___x_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__4___redArg___boxed(lean_object* v_mvarId_181_, lean_object* v___y_182_, lean_object* v___y_183_){
_start:
{
lean_object* v_res_184_; 
v_res_184_ = lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__4___redArg(v_mvarId_181_, v___y_182_);
lean_dec(v___y_182_);
lean_dec(v_mvarId_181_);
return v_res_184_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2___closed__3(void){
_start:
{
lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; 
v___x_188_ = ((lean_object*)(lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2___closed__2));
v___x_189_ = lean_unsigned_to_nat(22u);
v___x_190_ = lean_unsigned_to_nat(391u);
v___x_191_ = ((lean_object*)(lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2___closed__1));
v___x_192_ = ((lean_object*)(lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2___closed__0));
v___x_193_ = l_mkPanicMessageWithDecl(v___x_192_, v___x_191_, v___x_190_, v___x_189_, v___x_188_);
return v___x_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2(uint8_t v_unfold_195_, lean_object* v_e_196_, lean_object* v_a_197_, lean_object* v_a_198_, lean_object* v_a_199_, lean_object* v_a_200_){
_start:
{
switch(lean_obj_tag(v_e_196_))
{
case 0:
{
lean_object* v___x_202_; lean_object* v___x_203_; 
lean_dec_ref_known(v_e_196_, 1);
v___x_202_ = lean_obj_once(&lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2___closed__3, &lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2___closed__3_once, _init_lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2___closed__3);
v___x_203_ = lp_mathlib_panic___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__1(v___x_202_, v_a_197_, v_a_198_, v_a_199_, v_a_200_);
return v___x_203_;
}
case 1:
{
lean_object* v_fvarId_204_; lean_object* v___x_205_; 
v_fvarId_204_ = lean_ctor_get(v_e_196_, 0);
lean_inc(v_fvarId_204_);
v___x_205_ = l_Lean_FVarId_getDecl___redArg(v_fvarId_204_, v_a_197_, v_a_199_, v_a_200_);
if (lean_obj_tag(v___x_205_) == 0)
{
lean_object* v_a_206_; lean_object* v___x_208_; uint8_t v_isShared_209_; uint8_t v_isSharedCheck_250_; 
v_a_206_ = lean_ctor_get(v___x_205_, 0);
v_isSharedCheck_250_ = !lean_is_exclusive(v___x_205_);
if (v_isSharedCheck_250_ == 0)
{
v___x_208_ = v___x_205_;
v_isShared_209_ = v_isSharedCheck_250_;
goto v_resetjp_207_;
}
else
{
lean_inc(v_a_206_);
lean_dec(v___x_205_);
v___x_208_ = lean_box(0);
v_isShared_209_ = v_isSharedCheck_250_;
goto v_resetjp_207_;
}
v_resetjp_207_:
{
if (lean_obj_tag(v_a_206_) == 1)
{
lean_object* v_value_210_; uint8_t v_nondep_211_; lean_object* v___y_213_; uint8_t v_trackZetaDelta_214_; lean_object* v___y_215_; lean_object* v___y_216_; lean_object* v___y_217_; lean_object* v___y_230_; lean_object* v___y_231_; lean_object* v___y_232_; lean_object* v___y_233_; 
v_value_210_ = lean_ctor_get(v_a_206_, 4);
lean_inc_ref(v_value_210_);
v_nondep_211_ = lean_ctor_get_uint8(v_a_206_, sizeof(void*)*5);
if (v_nondep_211_ == 0)
{
uint8_t v___x_235_; 
v___x_235_ = l_Lean_LocalDecl_isImplementationDetail(v_a_206_);
lean_dec_ref_known(v_a_206_, 5);
if (v___x_235_ == 0)
{
lean_object* v___x_236_; uint8_t v_zetaDelta_237_; 
v___x_236_ = l_Lean_Meta_Context_config(v_a_197_);
v_zetaDelta_237_ = lean_ctor_get_uint8(v___x_236_, 16);
lean_dec_ref(v___x_236_);
if (v_zetaDelta_237_ == 0)
{
uint8_t v_trackZetaDelta_238_; lean_object* v_zetaDeltaSet_239_; uint8_t v___x_240_; 
v_trackZetaDelta_238_ = lean_ctor_get_uint8(v_a_197_, sizeof(void*)*7);
v_zetaDeltaSet_239_ = lean_ctor_get(v_a_197_, 1);
v___x_240_ = lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__3___redArg(v_fvarId_204_, v_zetaDeltaSet_239_);
if (v___x_240_ == 0)
{
lean_object* v___x_242_; 
lean_dec_ref(v_value_210_);
if (v_isShared_209_ == 0)
{
lean_ctor_set(v___x_208_, 0, v_e_196_);
v___x_242_ = v___x_208_;
goto v_reusejp_241_;
}
else
{
lean_object* v_reuseFailAlloc_243_; 
v_reuseFailAlloc_243_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_243_, 0, v_e_196_);
v___x_242_ = v_reuseFailAlloc_243_;
goto v_reusejp_241_;
}
v_reusejp_241_:
{
return v___x_242_;
}
}
else
{
lean_inc(v_fvarId_204_);
lean_del_object(v___x_208_);
lean_dec_ref_known(v_e_196_, 1);
v___y_213_ = v_a_197_;
v_trackZetaDelta_214_ = v_trackZetaDelta_238_;
v___y_215_ = v_a_198_;
v___y_216_ = v_a_199_;
v___y_217_ = v_a_200_;
goto v___jp_212_;
}
}
else
{
lean_inc(v_fvarId_204_);
lean_del_object(v___x_208_);
lean_dec_ref_known(v_e_196_, 1);
v___y_230_ = v_a_197_;
v___y_231_ = v_a_198_;
v___y_232_ = v_a_199_;
v___y_233_ = v_a_200_;
goto v___jp_229_;
}
}
else
{
lean_inc(v_fvarId_204_);
lean_del_object(v___x_208_);
lean_dec_ref_known(v_e_196_, 1);
v___y_230_ = v_a_197_;
v___y_231_ = v_a_198_;
v___y_232_ = v_a_199_;
v___y_233_ = v_a_200_;
goto v___jp_229_;
}
}
else
{
lean_object* v___x_245_; 
lean_dec_ref(v_value_210_);
lean_dec_ref_known(v_a_206_, 5);
if (v_isShared_209_ == 0)
{
lean_ctor_set(v___x_208_, 0, v_e_196_);
v___x_245_ = v___x_208_;
goto v_reusejp_244_;
}
else
{
lean_object* v_reuseFailAlloc_246_; 
v_reuseFailAlloc_246_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_246_, 0, v_e_196_);
v___x_245_ = v_reuseFailAlloc_246_;
goto v_reusejp_244_;
}
v_reusejp_244_:
{
return v___x_245_;
}
}
v___jp_212_:
{
if (v_trackZetaDelta_214_ == 0)
{
lean_dec(v_fvarId_204_);
v_e_196_ = v_value_210_;
v_a_197_ = v___y_213_;
v_a_198_ = v___y_215_;
v_a_199_ = v___y_216_;
v_a_200_ = v___y_217_;
goto _start;
}
else
{
lean_object* v___x_219_; 
v___x_219_ = l_Lean_Meta_addZetaDeltaFVarId___redArg(v_fvarId_204_, v___y_215_);
if (lean_obj_tag(v___x_219_) == 0)
{
lean_dec_ref_known(v___x_219_, 1);
v_e_196_ = v_value_210_;
v_a_197_ = v___y_213_;
v_a_198_ = v___y_215_;
v_a_199_ = v___y_216_;
v_a_200_ = v___y_217_;
goto _start;
}
else
{
lean_object* v_a_221_; lean_object* v___x_223_; uint8_t v_isShared_224_; uint8_t v_isSharedCheck_228_; 
lean_dec_ref(v_value_210_);
v_a_221_ = lean_ctor_get(v___x_219_, 0);
v_isSharedCheck_228_ = !lean_is_exclusive(v___x_219_);
if (v_isSharedCheck_228_ == 0)
{
v___x_223_ = v___x_219_;
v_isShared_224_ = v_isSharedCheck_228_;
goto v_resetjp_222_;
}
else
{
lean_inc(v_a_221_);
lean_dec(v___x_219_);
v___x_223_ = lean_box(0);
v_isShared_224_ = v_isSharedCheck_228_;
goto v_resetjp_222_;
}
v_resetjp_222_:
{
lean_object* v___x_226_; 
if (v_isShared_224_ == 0)
{
v___x_226_ = v___x_223_;
goto v_reusejp_225_;
}
else
{
lean_object* v_reuseFailAlloc_227_; 
v_reuseFailAlloc_227_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_227_, 0, v_a_221_);
v___x_226_ = v_reuseFailAlloc_227_;
goto v_reusejp_225_;
}
v_reusejp_225_:
{
return v___x_226_;
}
}
}
}
}
v___jp_229_:
{
uint8_t v_trackZetaDelta_234_; 
v_trackZetaDelta_234_ = lean_ctor_get_uint8(v___y_230_, sizeof(void*)*7);
v___y_213_ = v___y_230_;
v_trackZetaDelta_214_ = v_trackZetaDelta_234_;
v___y_215_ = v___y_231_;
v___y_216_ = v___y_232_;
v___y_217_ = v___y_233_;
goto v___jp_212_;
}
}
else
{
lean_object* v___x_248_; 
lean_dec(v_a_206_);
if (v_isShared_209_ == 0)
{
lean_ctor_set(v___x_208_, 0, v_e_196_);
v___x_248_ = v___x_208_;
goto v_reusejp_247_;
}
else
{
lean_object* v_reuseFailAlloc_249_; 
v_reuseFailAlloc_249_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_249_, 0, v_e_196_);
v___x_248_ = v_reuseFailAlloc_249_;
goto v_reusejp_247_;
}
v_reusejp_247_:
{
return v___x_248_;
}
}
}
}
else
{
lean_object* v_a_251_; lean_object* v___x_253_; uint8_t v_isShared_254_; uint8_t v_isSharedCheck_258_; 
lean_dec_ref_known(v_e_196_, 1);
v_a_251_ = lean_ctor_get(v___x_205_, 0);
v_isSharedCheck_258_ = !lean_is_exclusive(v___x_205_);
if (v_isSharedCheck_258_ == 0)
{
v___x_253_ = v___x_205_;
v_isShared_254_ = v_isSharedCheck_258_;
goto v_resetjp_252_;
}
else
{
lean_inc(v_a_251_);
lean_dec(v___x_205_);
v___x_253_ = lean_box(0);
v_isShared_254_ = v_isSharedCheck_258_;
goto v_resetjp_252_;
}
v_resetjp_252_:
{
lean_object* v___x_256_; 
if (v_isShared_254_ == 0)
{
v___x_256_ = v___x_253_;
goto v_reusejp_255_;
}
else
{
lean_object* v_reuseFailAlloc_257_; 
v_reuseFailAlloc_257_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_257_, 0, v_a_251_);
v___x_256_ = v_reuseFailAlloc_257_;
goto v_reusejp_255_;
}
v_reusejp_255_:
{
return v___x_256_;
}
}
}
}
case 2:
{
lean_object* v_mvarId_259_; lean_object* v___x_260_; 
v_mvarId_259_ = lean_ctor_get(v_e_196_, 0);
v___x_260_ = lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__4___redArg(v_mvarId_259_, v_a_198_);
if (lean_obj_tag(v___x_260_) == 0)
{
lean_object* v_a_261_; lean_object* v___x_263_; uint8_t v_isShared_264_; uint8_t v_isSharedCheck_270_; 
v_a_261_ = lean_ctor_get(v___x_260_, 0);
v_isSharedCheck_270_ = !lean_is_exclusive(v___x_260_);
if (v_isSharedCheck_270_ == 0)
{
v___x_263_ = v___x_260_;
v_isShared_264_ = v_isSharedCheck_270_;
goto v_resetjp_262_;
}
else
{
lean_inc(v_a_261_);
lean_dec(v___x_260_);
v___x_263_ = lean_box(0);
v_isShared_264_ = v_isSharedCheck_270_;
goto v_resetjp_262_;
}
v_resetjp_262_:
{
if (lean_obj_tag(v_a_261_) == 0)
{
lean_object* v___x_266_; 
if (v_isShared_264_ == 0)
{
lean_ctor_set(v___x_263_, 0, v_e_196_);
v___x_266_ = v___x_263_;
goto v_reusejp_265_;
}
else
{
lean_object* v_reuseFailAlloc_267_; 
v_reuseFailAlloc_267_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_267_, 0, v_e_196_);
v___x_266_ = v_reuseFailAlloc_267_;
goto v_reusejp_265_;
}
v_reusejp_265_:
{
return v___x_266_;
}
}
else
{
lean_object* v_val_268_; 
lean_del_object(v___x_263_);
lean_dec_ref_known(v_e_196_, 1);
v_val_268_ = lean_ctor_get(v_a_261_, 0);
lean_inc(v_val_268_);
lean_dec_ref_known(v_a_261_, 1);
v_e_196_ = v_val_268_;
goto _start;
}
}
}
else
{
lean_object* v_a_271_; lean_object* v___x_273_; uint8_t v_isShared_274_; uint8_t v_isSharedCheck_278_; 
lean_dec_ref_known(v_e_196_, 1);
v_a_271_ = lean_ctor_get(v___x_260_, 0);
v_isSharedCheck_278_ = !lean_is_exclusive(v___x_260_);
if (v_isSharedCheck_278_ == 0)
{
v___x_273_ = v___x_260_;
v_isShared_274_ = v_isSharedCheck_278_;
goto v_resetjp_272_;
}
else
{
lean_inc(v_a_271_);
lean_dec(v___x_260_);
v___x_273_ = lean_box(0);
v_isShared_274_ = v_isSharedCheck_278_;
goto v_resetjp_272_;
}
v_resetjp_272_:
{
lean_object* v___x_276_; 
if (v_isShared_274_ == 0)
{
v___x_276_ = v___x_273_;
goto v_reusejp_275_;
}
else
{
lean_object* v_reuseFailAlloc_277_; 
v_reuseFailAlloc_277_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_277_, 0, v_a_271_);
v___x_276_ = v_reuseFailAlloc_277_;
goto v_reusejp_275_;
}
v_reusejp_275_:
{
return v___x_276_;
}
}
}
}
case 3:
{
lean_object* v___x_279_; 
v___x_279_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_279_, 0, v_e_196_);
return v___x_279_;
}
case 6:
{
lean_object* v___x_280_; 
v___x_280_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_280_, 0, v_e_196_);
return v___x_280_;
}
case 7:
{
lean_object* v___x_281_; 
v___x_281_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_281_, 0, v_e_196_);
return v___x_281_;
}
case 9:
{
lean_object* v___x_282_; 
v___x_282_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_282_, 0, v_e_196_);
return v___x_282_;
}
case 10:
{
lean_object* v_expr_283_; 
v_expr_283_ = lean_ctor_get(v_e_196_, 1);
lean_inc_ref(v_expr_283_);
lean_dec_ref_known(v_e_196_, 2);
v_e_196_ = v_expr_283_;
goto _start;
}
default: 
{
lean_object* v___x_285_; 
v___x_285_ = l___private_Lean_Meta_WHNF_0__Lean_Meta_whnfCore_go(v_e_196_, v_a_197_, v_a_198_, v_a_199_, v_a_200_);
if (lean_obj_tag(v___x_285_) == 0)
{
lean_object* v_a_286_; lean_object* v___x_287_; 
v_a_286_ = lean_ctor_get(v___x_285_, 0);
lean_inc(v_a_286_);
v___x_287_ = l_Lean_Expr_getAppFn(v_a_286_);
if (lean_obj_tag(v___x_287_) == 4)
{
lean_object* v_declName_288_; lean_object* v___x_289_; lean_object* v___x_290_; uint8_t v___x_291_; 
v_declName_288_ = lean_ctor_get(v___x_287_, 0);
lean_inc(v_declName_288_);
lean_dec_ref_known(v___x_287_, 2);
v___x_289_ = ((lean_object*)(lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2___closed__4));
v___x_290_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_quantifiers));
v___x_291_ = l_List_elem___redArg(v___x_289_, v_declName_288_, v___x_290_);
if (v___x_291_ == 0)
{
if (v_unfold_195_ == 0)
{
lean_dec(v_a_286_);
return v___x_285_;
}
else
{
lean_object* v___x_292_; 
lean_dec_ref_known(v___x_285_, 1);
lean_inc(v_a_286_);
v___x_292_ = l_Lean_Meta_unfoldDefinition_x3f(v_a_286_, v___x_291_, v_a_197_, v_a_198_, v_a_199_, v_a_200_);
if (lean_obj_tag(v___x_292_) == 0)
{
lean_object* v_a_293_; lean_object* v___x_295_; uint8_t v_isShared_296_; uint8_t v_isSharedCheck_302_; 
v_a_293_ = lean_ctor_get(v___x_292_, 0);
v_isSharedCheck_302_ = !lean_is_exclusive(v___x_292_);
if (v_isSharedCheck_302_ == 0)
{
v___x_295_ = v___x_292_;
v_isShared_296_ = v_isSharedCheck_302_;
goto v_resetjp_294_;
}
else
{
lean_inc(v_a_293_);
lean_dec(v___x_292_);
v___x_295_ = lean_box(0);
v_isShared_296_ = v_isSharedCheck_302_;
goto v_resetjp_294_;
}
v_resetjp_294_:
{
if (lean_obj_tag(v_a_293_) == 0)
{
lean_object* v___x_298_; 
if (v_isShared_296_ == 0)
{
lean_ctor_set(v___x_295_, 0, v_a_286_);
v___x_298_ = v___x_295_;
goto v_reusejp_297_;
}
else
{
lean_object* v_reuseFailAlloc_299_; 
v_reuseFailAlloc_299_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_299_, 0, v_a_286_);
v___x_298_ = v_reuseFailAlloc_299_;
goto v_reusejp_297_;
}
v_reusejp_297_:
{
return v___x_298_;
}
}
else
{
lean_object* v_val_300_; lean_object* v___x_301_; 
lean_del_object(v___x_295_);
lean_dec(v_a_286_);
v_val_300_ = lean_ctor_get(v_a_293_, 0);
lean_inc(v_val_300_);
lean_dec_ref_known(v_a_293_, 1);
v___x_301_ = lp_mathlib_Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0(v_unfold_195_, v_val_300_, v_a_197_, v_a_198_, v_a_199_, v_a_200_);
return v___x_301_;
}
}
}
else
{
lean_object* v_a_303_; lean_object* v___x_305_; uint8_t v_isShared_306_; uint8_t v_isSharedCheck_310_; 
lean_dec(v_a_286_);
v_a_303_ = lean_ctor_get(v___x_292_, 0);
v_isSharedCheck_310_ = !lean_is_exclusive(v___x_292_);
if (v_isSharedCheck_310_ == 0)
{
v___x_305_ = v___x_292_;
v_isShared_306_ = v_isSharedCheck_310_;
goto v_resetjp_304_;
}
else
{
lean_inc(v_a_303_);
lean_dec(v___x_292_);
v___x_305_ = lean_box(0);
v_isShared_306_ = v_isSharedCheck_310_;
goto v_resetjp_304_;
}
v_resetjp_304_:
{
lean_object* v___x_308_; 
if (v_isShared_306_ == 0)
{
v___x_308_ = v___x_305_;
goto v_reusejp_307_;
}
else
{
lean_object* v_reuseFailAlloc_309_; 
v_reuseFailAlloc_309_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_309_, 0, v_a_303_);
v___x_308_ = v_reuseFailAlloc_309_;
goto v_reusejp_307_;
}
v_reusejp_307_:
{
return v___x_308_;
}
}
}
}
}
else
{
lean_dec(v_a_286_);
return v___x_285_;
}
}
else
{
lean_dec_ref(v___x_287_);
lean_dec(v_a_286_);
return v___x_285_;
}
}
else
{
return v___x_285_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0(uint8_t v_unfold_311_, lean_object* v_e_312_, lean_object* v_a_313_, lean_object* v_a_314_, lean_object* v_a_315_, lean_object* v_a_316_){
_start:
{
switch(lean_obj_tag(v_e_312_))
{
case 0:
{
lean_object* v___x_318_; lean_object* v___x_319_; 
lean_dec_ref_known(v_e_312_, 1);
v___x_318_ = lean_obj_once(&lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2___closed__3, &lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2___closed__3_once, _init_lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2___closed__3);
v___x_319_ = lp_mathlib_panic___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__1(v___x_318_, v_a_313_, v_a_314_, v_a_315_, v_a_316_);
return v___x_319_;
}
case 1:
{
lean_object* v_fvarId_320_; lean_object* v___x_321_; 
v_fvarId_320_ = lean_ctor_get(v_e_312_, 0);
lean_inc(v_fvarId_320_);
v___x_321_ = l_Lean_FVarId_getDecl___redArg(v_fvarId_320_, v_a_313_, v_a_315_, v_a_316_);
if (lean_obj_tag(v___x_321_) == 0)
{
lean_object* v_a_322_; lean_object* v___x_324_; uint8_t v_isShared_325_; uint8_t v_isSharedCheck_366_; 
v_a_322_ = lean_ctor_get(v___x_321_, 0);
v_isSharedCheck_366_ = !lean_is_exclusive(v___x_321_);
if (v_isSharedCheck_366_ == 0)
{
v___x_324_ = v___x_321_;
v_isShared_325_ = v_isSharedCheck_366_;
goto v_resetjp_323_;
}
else
{
lean_inc(v_a_322_);
lean_dec(v___x_321_);
v___x_324_ = lean_box(0);
v_isShared_325_ = v_isSharedCheck_366_;
goto v_resetjp_323_;
}
v_resetjp_323_:
{
if (lean_obj_tag(v_a_322_) == 1)
{
lean_object* v_value_326_; uint8_t v_nondep_327_; lean_object* v___y_329_; uint8_t v_trackZetaDelta_330_; lean_object* v___y_331_; lean_object* v___y_332_; lean_object* v___y_333_; lean_object* v___y_346_; lean_object* v___y_347_; lean_object* v___y_348_; lean_object* v___y_349_; 
v_value_326_ = lean_ctor_get(v_a_322_, 4);
lean_inc_ref(v_value_326_);
v_nondep_327_ = lean_ctor_get_uint8(v_a_322_, sizeof(void*)*5);
if (v_nondep_327_ == 0)
{
uint8_t v___x_351_; 
v___x_351_ = l_Lean_LocalDecl_isImplementationDetail(v_a_322_);
lean_dec_ref_known(v_a_322_, 5);
if (v___x_351_ == 0)
{
lean_object* v___x_352_; uint8_t v_zetaDelta_353_; 
v___x_352_ = l_Lean_Meta_Context_config(v_a_313_);
v_zetaDelta_353_ = lean_ctor_get_uint8(v___x_352_, 16);
lean_dec_ref(v___x_352_);
if (v_zetaDelta_353_ == 0)
{
uint8_t v_trackZetaDelta_354_; lean_object* v_zetaDeltaSet_355_; uint8_t v___x_356_; 
v_trackZetaDelta_354_ = lean_ctor_get_uint8(v_a_313_, sizeof(void*)*7);
v_zetaDeltaSet_355_ = lean_ctor_get(v_a_313_, 1);
v___x_356_ = lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__3___redArg(v_fvarId_320_, v_zetaDeltaSet_355_);
if (v___x_356_ == 0)
{
lean_object* v___x_358_; 
lean_dec_ref(v_value_326_);
if (v_isShared_325_ == 0)
{
lean_ctor_set(v___x_324_, 0, v_e_312_);
v___x_358_ = v___x_324_;
goto v_reusejp_357_;
}
else
{
lean_object* v_reuseFailAlloc_359_; 
v_reuseFailAlloc_359_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_359_, 0, v_e_312_);
v___x_358_ = v_reuseFailAlloc_359_;
goto v_reusejp_357_;
}
v_reusejp_357_:
{
return v___x_358_;
}
}
else
{
lean_inc(v_fvarId_320_);
lean_del_object(v___x_324_);
lean_dec_ref_known(v_e_312_, 1);
v___y_329_ = v_a_313_;
v_trackZetaDelta_330_ = v_trackZetaDelta_354_;
v___y_331_ = v_a_314_;
v___y_332_ = v_a_315_;
v___y_333_ = v_a_316_;
goto v___jp_328_;
}
}
else
{
lean_inc(v_fvarId_320_);
lean_del_object(v___x_324_);
lean_dec_ref_known(v_e_312_, 1);
v___y_346_ = v_a_313_;
v___y_347_ = v_a_314_;
v___y_348_ = v_a_315_;
v___y_349_ = v_a_316_;
goto v___jp_345_;
}
}
else
{
lean_inc(v_fvarId_320_);
lean_del_object(v___x_324_);
lean_dec_ref_known(v_e_312_, 1);
v___y_346_ = v_a_313_;
v___y_347_ = v_a_314_;
v___y_348_ = v_a_315_;
v___y_349_ = v_a_316_;
goto v___jp_345_;
}
}
else
{
lean_object* v___x_361_; 
lean_dec_ref(v_value_326_);
lean_dec_ref_known(v_a_322_, 5);
if (v_isShared_325_ == 0)
{
lean_ctor_set(v___x_324_, 0, v_e_312_);
v___x_361_ = v___x_324_;
goto v_reusejp_360_;
}
else
{
lean_object* v_reuseFailAlloc_362_; 
v_reuseFailAlloc_362_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_362_, 0, v_e_312_);
v___x_361_ = v_reuseFailAlloc_362_;
goto v_reusejp_360_;
}
v_reusejp_360_:
{
return v___x_361_;
}
}
v___jp_328_:
{
if (v_trackZetaDelta_330_ == 0)
{
lean_object* v___x_334_; 
lean_dec(v_fvarId_320_);
v___x_334_ = lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2(v_unfold_311_, v_value_326_, v___y_329_, v___y_331_, v___y_332_, v___y_333_);
return v___x_334_;
}
else
{
lean_object* v___x_335_; 
v___x_335_ = l_Lean_Meta_addZetaDeltaFVarId___redArg(v_fvarId_320_, v___y_331_);
if (lean_obj_tag(v___x_335_) == 0)
{
lean_object* v___x_336_; 
lean_dec_ref_known(v___x_335_, 1);
v___x_336_ = lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2(v_unfold_311_, v_value_326_, v___y_329_, v___y_331_, v___y_332_, v___y_333_);
return v___x_336_;
}
else
{
lean_object* v_a_337_; lean_object* v___x_339_; uint8_t v_isShared_340_; uint8_t v_isSharedCheck_344_; 
lean_dec_ref(v_value_326_);
v_a_337_ = lean_ctor_get(v___x_335_, 0);
v_isSharedCheck_344_ = !lean_is_exclusive(v___x_335_);
if (v_isSharedCheck_344_ == 0)
{
v___x_339_ = v___x_335_;
v_isShared_340_ = v_isSharedCheck_344_;
goto v_resetjp_338_;
}
else
{
lean_inc(v_a_337_);
lean_dec(v___x_335_);
v___x_339_ = lean_box(0);
v_isShared_340_ = v_isSharedCheck_344_;
goto v_resetjp_338_;
}
v_resetjp_338_:
{
lean_object* v___x_342_; 
if (v_isShared_340_ == 0)
{
v___x_342_ = v___x_339_;
goto v_reusejp_341_;
}
else
{
lean_object* v_reuseFailAlloc_343_; 
v_reuseFailAlloc_343_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_343_, 0, v_a_337_);
v___x_342_ = v_reuseFailAlloc_343_;
goto v_reusejp_341_;
}
v_reusejp_341_:
{
return v___x_342_;
}
}
}
}
}
v___jp_345_:
{
uint8_t v_trackZetaDelta_350_; 
v_trackZetaDelta_350_ = lean_ctor_get_uint8(v___y_346_, sizeof(void*)*7);
v___y_329_ = v___y_346_;
v_trackZetaDelta_330_ = v_trackZetaDelta_350_;
v___y_331_ = v___y_347_;
v___y_332_ = v___y_348_;
v___y_333_ = v___y_349_;
goto v___jp_328_;
}
}
else
{
lean_object* v___x_364_; 
lean_dec(v_a_322_);
if (v_isShared_325_ == 0)
{
lean_ctor_set(v___x_324_, 0, v_e_312_);
v___x_364_ = v___x_324_;
goto v_reusejp_363_;
}
else
{
lean_object* v_reuseFailAlloc_365_; 
v_reuseFailAlloc_365_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_365_, 0, v_e_312_);
v___x_364_ = v_reuseFailAlloc_365_;
goto v_reusejp_363_;
}
v_reusejp_363_:
{
return v___x_364_;
}
}
}
}
else
{
lean_object* v_a_367_; lean_object* v___x_369_; uint8_t v_isShared_370_; uint8_t v_isSharedCheck_374_; 
lean_dec_ref_known(v_e_312_, 1);
v_a_367_ = lean_ctor_get(v___x_321_, 0);
v_isSharedCheck_374_ = !lean_is_exclusive(v___x_321_);
if (v_isSharedCheck_374_ == 0)
{
v___x_369_ = v___x_321_;
v_isShared_370_ = v_isSharedCheck_374_;
goto v_resetjp_368_;
}
else
{
lean_inc(v_a_367_);
lean_dec(v___x_321_);
v___x_369_ = lean_box(0);
v_isShared_370_ = v_isSharedCheck_374_;
goto v_resetjp_368_;
}
v_resetjp_368_:
{
lean_object* v___x_372_; 
if (v_isShared_370_ == 0)
{
v___x_372_ = v___x_369_;
goto v_reusejp_371_;
}
else
{
lean_object* v_reuseFailAlloc_373_; 
v_reuseFailAlloc_373_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_373_, 0, v_a_367_);
v___x_372_ = v_reuseFailAlloc_373_;
goto v_reusejp_371_;
}
v_reusejp_371_:
{
return v___x_372_;
}
}
}
}
case 2:
{
lean_object* v_mvarId_375_; lean_object* v___x_376_; 
v_mvarId_375_ = lean_ctor_get(v_e_312_, 0);
v___x_376_ = lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__4___redArg(v_mvarId_375_, v_a_314_);
if (lean_obj_tag(v___x_376_) == 0)
{
lean_object* v_a_377_; lean_object* v___x_379_; uint8_t v_isShared_380_; uint8_t v_isSharedCheck_386_; 
v_a_377_ = lean_ctor_get(v___x_376_, 0);
v_isSharedCheck_386_ = !lean_is_exclusive(v___x_376_);
if (v_isSharedCheck_386_ == 0)
{
v___x_379_ = v___x_376_;
v_isShared_380_ = v_isSharedCheck_386_;
goto v_resetjp_378_;
}
else
{
lean_inc(v_a_377_);
lean_dec(v___x_376_);
v___x_379_ = lean_box(0);
v_isShared_380_ = v_isSharedCheck_386_;
goto v_resetjp_378_;
}
v_resetjp_378_:
{
if (lean_obj_tag(v_a_377_) == 0)
{
lean_object* v___x_382_; 
if (v_isShared_380_ == 0)
{
lean_ctor_set(v___x_379_, 0, v_e_312_);
v___x_382_ = v___x_379_;
goto v_reusejp_381_;
}
else
{
lean_object* v_reuseFailAlloc_383_; 
v_reuseFailAlloc_383_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_383_, 0, v_e_312_);
v___x_382_ = v_reuseFailAlloc_383_;
goto v_reusejp_381_;
}
v_reusejp_381_:
{
return v___x_382_;
}
}
else
{
lean_object* v_val_384_; lean_object* v___x_385_; 
lean_del_object(v___x_379_);
lean_dec_ref_known(v_e_312_, 1);
v_val_384_ = lean_ctor_get(v_a_377_, 0);
lean_inc(v_val_384_);
lean_dec_ref_known(v_a_377_, 1);
v___x_385_ = lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2(v_unfold_311_, v_val_384_, v_a_313_, v_a_314_, v_a_315_, v_a_316_);
return v___x_385_;
}
}
}
else
{
lean_object* v_a_387_; lean_object* v___x_389_; uint8_t v_isShared_390_; uint8_t v_isSharedCheck_394_; 
lean_dec_ref_known(v_e_312_, 1);
v_a_387_ = lean_ctor_get(v___x_376_, 0);
v_isSharedCheck_394_ = !lean_is_exclusive(v___x_376_);
if (v_isSharedCheck_394_ == 0)
{
v___x_389_ = v___x_376_;
v_isShared_390_ = v_isSharedCheck_394_;
goto v_resetjp_388_;
}
else
{
lean_inc(v_a_387_);
lean_dec(v___x_376_);
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
case 3:
{
lean_object* v___x_395_; 
v___x_395_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_395_, 0, v_e_312_);
return v___x_395_;
}
case 6:
{
lean_object* v___x_396_; 
v___x_396_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_396_, 0, v_e_312_);
return v___x_396_;
}
case 7:
{
lean_object* v___x_397_; 
v___x_397_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_397_, 0, v_e_312_);
return v___x_397_;
}
case 9:
{
lean_object* v___x_398_; 
v___x_398_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_398_, 0, v_e_312_);
return v___x_398_;
}
case 10:
{
lean_object* v_expr_399_; lean_object* v___x_400_; 
v_expr_399_ = lean_ctor_get(v_e_312_, 1);
lean_inc_ref(v_expr_399_);
lean_dec_ref_known(v_e_312_, 2);
v___x_400_ = lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2(v_unfold_311_, v_expr_399_, v_a_313_, v_a_314_, v_a_315_, v_a_316_);
return v___x_400_;
}
default: 
{
lean_object* v___x_401_; 
v___x_401_ = l___private_Lean_Meta_WHNF_0__Lean_Meta_whnfCore_go(v_e_312_, v_a_313_, v_a_314_, v_a_315_, v_a_316_);
if (lean_obj_tag(v___x_401_) == 0)
{
lean_object* v_a_402_; lean_object* v___x_403_; 
v_a_402_ = lean_ctor_get(v___x_401_, 0);
lean_inc(v_a_402_);
v___x_403_ = l_Lean_Expr_getAppFn(v_a_402_);
if (lean_obj_tag(v___x_403_) == 4)
{
lean_object* v_declName_404_; lean_object* v___x_405_; lean_object* v___x_406_; uint8_t v___x_407_; 
v_declName_404_ = lean_ctor_get(v___x_403_, 0);
lean_inc(v_declName_404_);
lean_dec_ref_known(v___x_403_, 2);
v___x_405_ = ((lean_object*)(lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2___closed__4));
v___x_406_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_quantifiers));
v___x_407_ = l_List_elem___redArg(v___x_405_, v_declName_404_, v___x_406_);
if (v___x_407_ == 0)
{
if (v_unfold_311_ == 0)
{
lean_dec(v_a_402_);
return v___x_401_;
}
else
{
lean_object* v___x_408_; 
lean_dec_ref_known(v___x_401_, 1);
lean_inc(v_a_402_);
v___x_408_ = l_Lean_Meta_unfoldDefinition_x3f(v_a_402_, v___x_407_, v_a_313_, v_a_314_, v_a_315_, v_a_316_);
if (lean_obj_tag(v___x_408_) == 0)
{
lean_object* v_a_409_; lean_object* v___x_411_; uint8_t v_isShared_412_; uint8_t v_isSharedCheck_418_; 
v_a_409_ = lean_ctor_get(v___x_408_, 0);
v_isSharedCheck_418_ = !lean_is_exclusive(v___x_408_);
if (v_isSharedCheck_418_ == 0)
{
v___x_411_ = v___x_408_;
v_isShared_412_ = v_isSharedCheck_418_;
goto v_resetjp_410_;
}
else
{
lean_inc(v_a_409_);
lean_dec(v___x_408_);
v___x_411_ = lean_box(0);
v_isShared_412_ = v_isSharedCheck_418_;
goto v_resetjp_410_;
}
v_resetjp_410_:
{
if (lean_obj_tag(v_a_409_) == 0)
{
lean_object* v___x_414_; 
if (v_isShared_412_ == 0)
{
lean_ctor_set(v___x_411_, 0, v_a_402_);
v___x_414_ = v___x_411_;
goto v_reusejp_413_;
}
else
{
lean_object* v_reuseFailAlloc_415_; 
v_reuseFailAlloc_415_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_415_, 0, v_a_402_);
v___x_414_ = v_reuseFailAlloc_415_;
goto v_reusejp_413_;
}
v_reusejp_413_:
{
return v___x_414_;
}
}
else
{
lean_object* v_val_416_; lean_object* v___x_417_; 
lean_del_object(v___x_411_);
lean_dec(v_a_402_);
v_val_416_ = lean_ctor_get(v_a_409_, 0);
lean_inc(v_val_416_);
lean_dec_ref_known(v_a_409_, 1);
v___x_417_ = lp_mathlib_Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0(v_unfold_311_, v_val_416_, v_a_313_, v_a_314_, v_a_315_, v_a_316_);
return v___x_417_;
}
}
}
else
{
lean_object* v_a_419_; lean_object* v___x_421_; uint8_t v_isShared_422_; uint8_t v_isSharedCheck_426_; 
lean_dec(v_a_402_);
v_a_419_ = lean_ctor_get(v___x_408_, 0);
v_isSharedCheck_426_ = !lean_is_exclusive(v___x_408_);
if (v_isSharedCheck_426_ == 0)
{
v___x_421_ = v___x_408_;
v_isShared_422_ = v_isSharedCheck_426_;
goto v_resetjp_420_;
}
else
{
lean_inc(v_a_419_);
lean_dec(v___x_408_);
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
else
{
lean_dec(v_a_402_);
return v___x_401_;
}
}
else
{
lean_dec_ref(v___x_403_);
lean_dec(v_a_402_);
return v___x_401_;
}
}
else
{
return v___x_401_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0(uint8_t v_unfold_427_, lean_object* v_e_428_, lean_object* v_a_429_, lean_object* v_a_430_, lean_object* v_a_431_, lean_object* v_a_432_){
_start:
{
lean_object* v___x_434_; 
v___x_434_ = lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0(v_unfold_427_, v_e_428_, v_a_429_, v_a_430_, v_a_431_, v_a_432_);
return v___x_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0___boxed(lean_object* v_unfold_435_, lean_object* v_e_436_, lean_object* v_a_437_, lean_object* v_a_438_, lean_object* v_a_439_, lean_object* v_a_440_, lean_object* v_a_441_){
_start:
{
uint8_t v_unfold_boxed_442_; lean_object* v_res_443_; 
v_unfold_boxed_442_ = lean_unbox(v_unfold_435_);
v_res_443_ = lp_mathlib_Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0(v_unfold_boxed_442_, v_e_436_, v_a_437_, v_a_438_, v_a_439_, v_a_440_);
lean_dec(v_a_440_);
lean_dec_ref(v_a_439_);
lean_dec(v_a_438_);
lean_dec_ref(v_a_437_);
return v_res_443_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2___boxed(lean_object* v_unfold_444_, lean_object* v_e_445_, lean_object* v_a_446_, lean_object* v_a_447_, lean_object* v_a_448_, lean_object* v_a_449_, lean_object* v_a_450_){
_start:
{
uint8_t v_unfold_boxed_451_; lean_object* v_res_452_; 
v_unfold_boxed_451_ = lean_unbox(v_unfold_444_);
v_res_452_ = lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__2(v_unfold_boxed_451_, v_e_445_, v_a_446_, v_a_447_, v_a_448_, v_a_449_);
lean_dec(v_a_449_);
lean_dec_ref(v_a_448_);
lean_dec(v_a_447_);
lean_dec_ref(v_a_446_);
return v_res_452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0___boxed(lean_object* v_unfold_453_, lean_object* v_e_454_, lean_object* v_a_455_, lean_object* v_a_456_, lean_object* v_a_457_, lean_object* v_a_458_, lean_object* v_a_459_){
_start:
{
uint8_t v_unfold_boxed_460_; lean_object* v_res_461_; 
v_unfold_boxed_460_ = lean_unbox(v_unfold_453_);
v_res_461_ = lp_mathlib_Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0(v_unfold_boxed_460_, v_e_454_, v_a_455_, v_a_456_, v_a_457_, v_a_458_);
lean_dec(v_a_458_);
lean_dec_ref(v_a_457_);
lean_dec(v_a_456_);
lean_dec_ref(v_a_455_);
return v_res_461_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_whnfQuantifier(lean_object* v_p_462_, uint8_t v_unfold_463_, lean_object* v_a_464_, lean_object* v_a_465_, lean_object* v_a_466_, lean_object* v_a_467_){
_start:
{
if (v_unfold_463_ == 0)
{
lean_object* v___x_469_; 
v___x_469_ = l_Lean_Meta_whnfR(v_p_462_, v_a_464_, v_a_465_, v_a_466_, v_a_467_);
return v___x_469_;
}
else
{
lean_object* v___x_470_; 
v___x_470_ = lp_mathlib_Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0(v_unfold_463_, v_p_462_, v_a_464_, v_a_465_, v_a_466_, v_a_467_);
return v___x_470_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_whnfQuantifier___boxed(lean_object* v_p_471_, lean_object* v_unfold_472_, lean_object* v_a_473_, lean_object* v_a_474_, lean_object* v_a_475_, lean_object* v_a_476_, lean_object* v_a_477_){
_start:
{
uint8_t v_unfold_boxed_478_; lean_object* v_res_479_; 
v_unfold_boxed_478_ = lean_unbox(v_unfold_472_);
v_res_479_ = lp_mathlib_Mathlib_Tactic_Peel_whnfQuantifier(v_p_471_, v_unfold_boxed_478_, v_a_473_, v_a_474_, v_a_475_, v_a_476_);
lean_dec(v_a_476_);
lean_dec_ref(v_a_475_);
lean_dec(v_a_474_);
lean_dec_ref(v_a_473_);
return v_res_479_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__4(lean_object* v_mvarId_480_, lean_object* v___y_481_, lean_object* v___y_482_, lean_object* v___y_483_, lean_object* v___y_484_){
_start:
{
lean_object* v___x_486_; 
v___x_486_ = lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__4___redArg(v_mvarId_480_, v___y_482_);
return v___x_486_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__4___boxed(lean_object* v_mvarId_487_, lean_object* v___y_488_, lean_object* v___y_489_, lean_object* v___y_490_, lean_object* v___y_491_, lean_object* v___y_492_){
_start:
{
lean_object* v_res_493_; 
v_res_493_ = lp_mathlib_Lean_getExprMVarAssignment_x3f___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__4(v_mvarId_487_, v___y_488_, v___y_489_, v___y_490_, v___y_491_);
lean_dec(v___y_491_);
lean_dec_ref(v___y_490_);
lean_dec(v___y_489_);
lean_dec_ref(v___y_488_);
lean_dec(v_mvarId_487_);
return v_res_493_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__3(lean_object* v_00_u03b2_494_, lean_object* v_k_495_, lean_object* v_t_496_){
_start:
{
uint8_t v___x_497_; 
v___x_497_ = lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__3___redArg(v_k_495_, v_t_496_);
return v___x_497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__3___boxed(lean_object* v_00_u03b2_498_, lean_object* v_k_499_, lean_object* v_t_500_){
_start:
{
uint8_t v_res_501_; lean_object* v_r_502_; 
v_res_501_ = lp_mathlib_Std_DTreeMap_Internal_Impl_contains___at___00Lean_Meta_whnfEasyCases___at___00Lean_Meta_whnfHeadPred___at___00Mathlib_Tactic_Peel_whnfQuantifier_spec__0_spec__0_spec__3(v_00_u03b2_498_, v_k_499_, v_t_500_);
lean_dec(v_t_500_);
lean_dec(v_k_499_);
v_r_502_ = lean_box(v_res_501_);
return v_r_502_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Peel_throwPeelError_spec__0_spec__0(lean_object* v_msgData_503_, lean_object* v___y_504_, lean_object* v___y_505_, lean_object* v___y_506_, lean_object* v___y_507_){
_start:
{
lean_object* v___x_509_; lean_object* v_env_510_; lean_object* v___x_511_; lean_object* v_mctx_512_; lean_object* v_lctx_513_; lean_object* v_options_514_; lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___x_517_; 
v___x_509_ = lean_st_ref_get(v___y_507_);
v_env_510_ = lean_ctor_get(v___x_509_, 0);
lean_inc_ref(v_env_510_);
lean_dec(v___x_509_);
v___x_511_ = lean_st_ref_get(v___y_505_);
v_mctx_512_ = lean_ctor_get(v___x_511_, 0);
lean_inc_ref(v_mctx_512_);
lean_dec(v___x_511_);
v_lctx_513_ = lean_ctor_get(v___y_504_, 2);
v_options_514_ = lean_ctor_get(v___y_506_, 2);
lean_inc_ref(v_options_514_);
lean_inc_ref(v_lctx_513_);
v___x_515_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_515_, 0, v_env_510_);
lean_ctor_set(v___x_515_, 1, v_mctx_512_);
lean_ctor_set(v___x_515_, 2, v_lctx_513_);
lean_ctor_set(v___x_515_, 3, v_options_514_);
v___x_516_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_516_, 0, v___x_515_);
lean_ctor_set(v___x_516_, 1, v_msgData_503_);
v___x_517_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_517_, 0, v___x_516_);
return v___x_517_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Peel_throwPeelError_spec__0_spec__0___boxed(lean_object* v_msgData_518_, lean_object* v___y_519_, lean_object* v___y_520_, lean_object* v___y_521_, lean_object* v___y_522_, lean_object* v___y_523_){
_start:
{
lean_object* v_res_524_; 
v_res_524_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Peel_throwPeelError_spec__0_spec__0(v_msgData_518_, v___y_519_, v___y_520_, v___y_521_, v___y_522_);
lean_dec(v___y_522_);
lean_dec_ref(v___y_521_);
lean_dec(v___y_520_);
lean_dec_ref(v___y_519_);
return v_res_524_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Peel_throwPeelError_spec__0___redArg(lean_object* v_msg_525_, lean_object* v___y_526_, lean_object* v___y_527_, lean_object* v___y_528_, lean_object* v___y_529_){
_start:
{
lean_object* v_ref_531_; lean_object* v___x_532_; lean_object* v_a_533_; lean_object* v___x_535_; uint8_t v_isShared_536_; uint8_t v_isSharedCheck_541_; 
v_ref_531_ = lean_ctor_get(v___y_528_, 5);
v___x_532_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Peel_throwPeelError_spec__0_spec__0(v_msg_525_, v___y_526_, v___y_527_, v___y_528_, v___y_529_);
v_a_533_ = lean_ctor_get(v___x_532_, 0);
v_isSharedCheck_541_ = !lean_is_exclusive(v___x_532_);
if (v_isSharedCheck_541_ == 0)
{
v___x_535_ = v___x_532_;
v_isShared_536_ = v_isSharedCheck_541_;
goto v_resetjp_534_;
}
else
{
lean_inc(v_a_533_);
lean_dec(v___x_532_);
v___x_535_ = lean_box(0);
v_isShared_536_ = v_isSharedCheck_541_;
goto v_resetjp_534_;
}
v_resetjp_534_:
{
lean_object* v___x_537_; lean_object* v___x_539_; 
lean_inc(v_ref_531_);
v___x_537_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_537_, 0, v_ref_531_);
lean_ctor_set(v___x_537_, 1, v_a_533_);
if (v_isShared_536_ == 0)
{
lean_ctor_set_tag(v___x_535_, 1);
lean_ctor_set(v___x_535_, 0, v___x_537_);
v___x_539_ = v___x_535_;
goto v_reusejp_538_;
}
else
{
lean_object* v_reuseFailAlloc_540_; 
v_reuseFailAlloc_540_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_540_, 0, v___x_537_);
v___x_539_ = v_reuseFailAlloc_540_;
goto v_reusejp_538_;
}
v_reusejp_538_:
{
return v___x_539_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Peel_throwPeelError_spec__0___redArg___boxed(lean_object* v_msg_542_, lean_object* v___y_543_, lean_object* v___y_544_, lean_object* v___y_545_, lean_object* v___y_546_, lean_object* v___y_547_){
_start:
{
lean_object* v_res_548_; 
v_res_548_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Peel_throwPeelError_spec__0___redArg(v_msg_542_, v___y_543_, v___y_544_, v___y_545_, v___y_546_);
lean_dec(v___y_546_);
lean_dec_ref(v___y_545_);
lean_dec(v___y_544_);
lean_dec_ref(v___y_543_);
return v_res_548_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg___closed__1(void){
_start:
{
lean_object* v___x_550_; lean_object* v___x_551_; 
v___x_550_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg___closed__0));
v___x_551_ = l_Lean_stringToMessageData(v___x_550_);
return v___x_551_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg___closed__3(void){
_start:
{
lean_object* v___x_553_; lean_object* v___x_554_; 
v___x_553_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg___closed__2));
v___x_554_ = l_Lean_stringToMessageData(v___x_553_);
return v___x_554_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg(lean_object* v_ty_555_, lean_object* v_target_556_, lean_object* v_a_557_, lean_object* v_a_558_, lean_object* v_a_559_, lean_object* v_a_560_){
_start:
{
lean_object* v___x_562_; lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; 
v___x_562_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg___closed__1, &lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg___closed__1);
v___x_563_ = l_Lean_MessageData_ofExpr(v_ty_555_);
v___x_564_ = l_Lean_indentD(v___x_563_);
v___x_565_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_565_, 0, v___x_562_);
lean_ctor_set(v___x_565_, 1, v___x_564_);
v___x_566_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg___closed__3, &lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg___closed__3);
v___x_567_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_567_, 0, v___x_565_);
lean_ctor_set(v___x_567_, 1, v___x_566_);
v___x_568_ = l_Lean_MessageData_ofExpr(v_target_556_);
v___x_569_ = l_Lean_indentD(v___x_568_);
v___x_570_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_570_, 0, v___x_567_);
lean_ctor_set(v___x_570_, 1, v___x_569_);
v___x_571_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Peel_throwPeelError_spec__0___redArg(v___x_570_, v_a_557_, v_a_558_, v_a_559_, v_a_560_);
return v___x_571_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg___boxed(lean_object* v_ty_572_, lean_object* v_target_573_, lean_object* v_a_574_, lean_object* v_a_575_, lean_object* v_a_576_, lean_object* v_a_577_, lean_object* v_a_578_){
_start:
{
lean_object* v_res_579_; 
v_res_579_ = lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg(v_ty_572_, v_target_573_, v_a_574_, v_a_575_, v_a_576_, v_a_577_);
lean_dec(v_a_577_);
lean_dec_ref(v_a_576_);
lean_dec(v_a_575_);
lean_dec_ref(v_a_574_);
return v_res_579_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_throwPeelError(lean_object* v_00_u03b1_580_, lean_object* v_ty_581_, lean_object* v_target_582_, lean_object* v_a_583_, lean_object* v_a_584_, lean_object* v_a_585_, lean_object* v_a_586_){
_start:
{
lean_object* v___x_588_; 
v___x_588_ = lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg(v_ty_581_, v_target_582_, v_a_583_, v_a_584_, v_a_585_, v_a_586_);
return v___x_588_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___boxed(lean_object* v_00_u03b1_589_, lean_object* v_ty_590_, lean_object* v_target_591_, lean_object* v_a_592_, lean_object* v_a_593_, lean_object* v_a_594_, lean_object* v_a_595_, lean_object* v_a_596_){
_start:
{
lean_object* v_res_597_; 
v_res_597_ = lp_mathlib_Mathlib_Tactic_Peel_throwPeelError(v_00_u03b1_589_, v_ty_590_, v_target_591_, v_a_592_, v_a_593_, v_a_594_, v_a_595_);
lean_dec(v_a_595_);
lean_dec_ref(v_a_594_);
lean_dec(v_a_593_);
lean_dec_ref(v_a_592_);
return v_res_597_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Peel_throwPeelError_spec__0(lean_object* v_00_u03b1_598_, lean_object* v_msg_599_, lean_object* v___y_600_, lean_object* v___y_601_, lean_object* v___y_602_, lean_object* v___y_603_){
_start:
{
lean_object* v___x_605_; 
v___x_605_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Peel_throwPeelError_spec__0___redArg(v_msg_599_, v___y_600_, v___y_601_, v___y_602_, v___y_603_);
return v___x_605_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Peel_throwPeelError_spec__0___boxed(lean_object* v_00_u03b1_606_, lean_object* v_msg_607_, lean_object* v___y_608_, lean_object* v___y_609_, lean_object* v___y_610_, lean_object* v___y_611_, lean_object* v___y_612_){
_start:
{
lean_object* v_res_613_; 
v_res_613_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Peel_throwPeelError_spec__0(v_00_u03b1_606_, v_msg_607_, v___y_608_, v___y_609_, v___y_610_, v___y_611_);
lean_dec(v___y_611_);
lean_dec_ref(v___y_610_);
lean_dec(v___y_609_);
lean_dec_ref(v___y_608_);
return v_res_613_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_mkFreshBinderName___redArg(lean_object* v_f_617_, lean_object* v_a_618_, lean_object* v_a_619_){
_start:
{
if (lean_obj_tag(v_f_617_) == 6)
{
lean_object* v_binderName_621_; lean_object* v___x_622_; 
v_binderName_621_ = lean_ctor_get(v_f_617_, 0);
lean_inc(v_binderName_621_);
lean_dec_ref_known(v_f_617_, 3);
v___x_622_ = l_Lean_Core_mkFreshUserName(v_binderName_621_, v_a_618_, v_a_619_);
return v___x_622_;
}
else
{
lean_object* v___x_623_; lean_object* v___x_624_; 
lean_dec_ref(v_f_617_);
v___x_623_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_mkFreshBinderName___redArg___closed__1));
v___x_624_ = l_Lean_Core_mkFreshUserName(v___x_623_, v_a_618_, v_a_619_);
return v___x_624_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_mkFreshBinderName___redArg___boxed(lean_object* v_f_625_, lean_object* v_a_626_, lean_object* v_a_627_, lean_object* v_a_628_){
_start:
{
lean_object* v_res_629_; 
v_res_629_ = lp_mathlib_Mathlib_Tactic_Peel_mkFreshBinderName___redArg(v_f_625_, v_a_626_, v_a_627_);
lean_dec(v_a_627_);
lean_dec_ref(v_a_626_);
return v_res_629_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_mkFreshBinderName(lean_object* v_f_630_, lean_object* v_a_631_, lean_object* v_a_632_, lean_object* v_a_633_, lean_object* v_a_634_){
_start:
{
lean_object* v___x_636_; 
v___x_636_ = lp_mathlib_Mathlib_Tactic_Peel_mkFreshBinderName___redArg(v_f_630_, v_a_633_, v_a_634_);
return v___x_636_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_mkFreshBinderName___boxed(lean_object* v_f_637_, lean_object* v_a_638_, lean_object* v_a_639_, lean_object* v_a_640_, lean_object* v_a_641_, lean_object* v_a_642_){
_start:
{
lean_object* v_res_643_; 
v_res_643_ = lp_mathlib_Mathlib_Tactic_Peel_mkFreshBinderName(v_f_637_, v_a_638_, v_a_639_, v_a_640_, v_a_641_);
lean_dec(v_a_641_);
lean_dec_ref(v_a_640_);
lean_dec(v_a_639_);
lean_dec_ref(v_a_638_);
return v_res_643_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Peel_applyPeelThm___closed__1(void){
_start:
{
lean_object* v___x_645_; lean_object* v___x_646_; 
v___x_645_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_applyPeelThm___closed__0));
v___x_646_ = l_Lean_stringToMessageData(v___x_645_);
return v___x_646_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_applyPeelThm(lean_object* v_thm_651_, lean_object* v_goal_652_, lean_object* v_e_653_, lean_object* v_ty_654_, lean_object* v_target_655_, lean_object* v_n_656_, lean_object* v_n_x27_657_, lean_object* v_a_658_, lean_object* v_a_659_, lean_object* v_a_660_, lean_object* v_a_661_){
_start:
{
lean_object* v___y_664_; lean_object* v___y_665_; lean_object* v___y_666_; lean_object* v___y_667_; uint8_t v___x_670_; lean_object* v___x_671_; lean_object* v___x_672_; 
v___x_670_ = 0;
v___x_671_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_applyPeelThm___closed__2));
v___x_672_ = l_Lean_Meta_saveState___redArg(v_a_659_, v_a_661_);
if (lean_obj_tag(v___x_672_) == 0)
{
lean_object* v_a_673_; lean_object* v___x_674_; lean_object* v___y_676_; lean_object* v___y_677_; lean_object* v___y_720_; lean_object* v___y_721_; lean_object* v___y_722_; uint8_t v___y_723_; lean_object* v___y_727_; lean_object* v___x_754_; 
v_a_673_ = lean_ctor_get(v___x_672_, 0);
lean_inc(v_a_673_);
lean_dec_ref_known(v___x_672_, 1);
v___x_674_ = lean_box(0);
v___x_754_ = l_Lean_MVarId_applyConst(v_goal_652_, v_thm_651_, v___x_671_, v_a_658_, v_a_659_, v_a_660_, v_a_661_);
if (lean_obj_tag(v___x_754_) == 0)
{
lean_dec(v_a_673_);
v___y_727_ = v___x_754_;
goto v___jp_726_;
}
else
{
lean_object* v_a_755_; uint8_t v___y_757_; uint8_t v___x_768_; 
v_a_755_ = lean_ctor_get(v___x_754_, 0);
lean_inc(v_a_755_);
v___x_768_ = l_Lean_Exception_isInterrupt(v_a_755_);
if (v___x_768_ == 0)
{
uint8_t v___x_769_; 
v___x_769_ = l_Lean_Exception_isRuntime(v_a_755_);
v___y_757_ = v___x_769_;
goto v___jp_756_;
}
else
{
lean_dec(v_a_755_);
v___y_757_ = v___x_768_;
goto v___jp_756_;
}
v___jp_756_:
{
if (v___y_757_ == 0)
{
lean_object* v___x_758_; 
lean_dec_ref_known(v___x_754_, 1);
v___x_758_ = l_Lean_Meta_SavedState_restore___redArg(v_a_673_, v_a_659_, v_a_661_);
lean_dec(v_a_673_);
if (lean_obj_tag(v___x_758_) == 0)
{
lean_object* v___x_759_; 
lean_dec_ref_known(v___x_758_, 1);
lean_inc_ref(v_target_655_);
lean_inc_ref(v_ty_654_);
v___x_759_ = lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg(v_ty_654_, v_target_655_, v_a_658_, v_a_659_, v_a_660_, v_a_661_);
v___y_727_ = v___x_759_;
goto v___jp_726_;
}
else
{
lean_object* v_a_760_; lean_object* v___x_762_; uint8_t v_isShared_763_; uint8_t v_isSharedCheck_767_; 
lean_dec(v_n_x27_657_);
lean_dec(v_n_656_);
lean_dec_ref(v_target_655_);
lean_dec_ref(v_ty_654_);
lean_dec_ref(v_e_653_);
v_a_760_ = lean_ctor_get(v___x_758_, 0);
v_isSharedCheck_767_ = !lean_is_exclusive(v___x_758_);
if (v_isSharedCheck_767_ == 0)
{
v___x_762_ = v___x_758_;
v_isShared_763_ = v_isSharedCheck_767_;
goto v_resetjp_761_;
}
else
{
lean_inc(v_a_760_);
lean_dec(v___x_758_);
v___x_762_ = lean_box(0);
v_isShared_763_ = v_isSharedCheck_767_;
goto v_resetjp_761_;
}
v_resetjp_761_:
{
lean_object* v___x_765_; 
if (v_isShared_763_ == 0)
{
v___x_765_ = v___x_762_;
goto v_reusejp_764_;
}
else
{
lean_object* v_reuseFailAlloc_766_; 
v_reuseFailAlloc_766_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_766_, 0, v_a_760_);
v___x_765_ = v_reuseFailAlloc_766_;
goto v_reusejp_764_;
}
v_reusejp_764_:
{
return v___x_765_;
}
}
}
}
else
{
lean_dec(v_a_673_);
v___y_727_ = v___x_754_;
goto v___jp_726_;
}
}
}
v___jp_675_:
{
if (lean_obj_tag(v___y_677_) == 0)
{
lean_object* v___x_678_; lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_681_; lean_object* v___x_682_; 
lean_dec_ref_known(v___y_677_, 1);
v___x_678_ = lean_unsigned_to_nat(2u);
v___x_679_ = lean_box(0);
v___x_680_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_680_, 0, v_n_x27_657_);
lean_ctor_set(v___x_680_, 1, v___x_679_);
v___x_681_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_681_, 0, v_n_656_);
lean_ctor_set(v___x_681_, 1, v___x_680_);
v___x_682_ = l_Lean_Meta_introNCore(v___y_676_, v___x_678_, v___x_681_, v___x_670_, v___x_670_, v_a_658_, v_a_659_, v_a_660_, v_a_661_);
if (lean_obj_tag(v___x_682_) == 0)
{
lean_object* v_a_683_; lean_object* v___x_685_; uint8_t v_isShared_686_; uint8_t v_isSharedCheck_702_; 
v_a_683_ = lean_ctor_get(v___x_682_, 0);
v_isSharedCheck_702_ = !lean_is_exclusive(v___x_682_);
if (v_isSharedCheck_702_ == 0)
{
v___x_685_ = v___x_682_;
v_isShared_686_ = v_isSharedCheck_702_;
goto v_resetjp_684_;
}
else
{
lean_inc(v_a_683_);
lean_dec(v___x_682_);
v___x_685_ = lean_box(0);
v_isShared_686_ = v_isSharedCheck_702_;
goto v_resetjp_684_;
}
v_resetjp_684_:
{
lean_object* v_fst_687_; lean_object* v_snd_688_; lean_object* v___x_690_; uint8_t v_isShared_691_; uint8_t v_isSharedCheck_701_; 
v_fst_687_ = lean_ctor_get(v_a_683_, 0);
v_snd_688_ = lean_ctor_get(v_a_683_, 1);
v_isSharedCheck_701_ = !lean_is_exclusive(v_a_683_);
if (v_isSharedCheck_701_ == 0)
{
v___x_690_ = v_a_683_;
v_isShared_691_ = v_isSharedCheck_701_;
goto v_resetjp_689_;
}
else
{
lean_inc(v_snd_688_);
lean_inc(v_fst_687_);
lean_dec(v_a_683_);
v___x_690_ = lean_box(0);
v_isShared_691_ = v_isSharedCheck_701_;
goto v_resetjp_689_;
}
v_resetjp_689_:
{
lean_object* v___x_692_; lean_object* v___x_693_; lean_object* v___x_694_; lean_object* v___x_696_; 
v___x_692_ = lean_unsigned_to_nat(1u);
v___x_693_ = lean_array_get(v___x_674_, v_fst_687_, v___x_692_);
lean_dec(v_fst_687_);
v___x_694_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_694_, 0, v_snd_688_);
lean_ctor_set(v___x_694_, 1, v___x_679_);
if (v_isShared_691_ == 0)
{
lean_ctor_set(v___x_690_, 1, v___x_694_);
lean_ctor_set(v___x_690_, 0, v___x_693_);
v___x_696_ = v___x_690_;
goto v_reusejp_695_;
}
else
{
lean_object* v_reuseFailAlloc_700_; 
v_reuseFailAlloc_700_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_700_, 0, v___x_693_);
lean_ctor_set(v_reuseFailAlloc_700_, 1, v___x_694_);
v___x_696_ = v_reuseFailAlloc_700_;
goto v_reusejp_695_;
}
v_reusejp_695_:
{
lean_object* v___x_698_; 
if (v_isShared_686_ == 0)
{
lean_ctor_set(v___x_685_, 0, v___x_696_);
v___x_698_ = v___x_685_;
goto v_reusejp_697_;
}
else
{
lean_object* v_reuseFailAlloc_699_; 
v_reuseFailAlloc_699_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_699_, 0, v___x_696_);
v___x_698_ = v_reuseFailAlloc_699_;
goto v_reusejp_697_;
}
v_reusejp_697_:
{
return v___x_698_;
}
}
}
}
}
else
{
lean_object* v_a_703_; lean_object* v___x_705_; uint8_t v_isShared_706_; uint8_t v_isSharedCheck_710_; 
v_a_703_ = lean_ctor_get(v___x_682_, 0);
v_isSharedCheck_710_ = !lean_is_exclusive(v___x_682_);
if (v_isSharedCheck_710_ == 0)
{
v___x_705_ = v___x_682_;
v_isShared_706_ = v_isSharedCheck_710_;
goto v_resetjp_704_;
}
else
{
lean_inc(v_a_703_);
lean_dec(v___x_682_);
v___x_705_ = lean_box(0);
v_isShared_706_ = v_isSharedCheck_710_;
goto v_resetjp_704_;
}
v_resetjp_704_:
{
lean_object* v___x_708_; 
if (v_isShared_706_ == 0)
{
v___x_708_ = v___x_705_;
goto v_reusejp_707_;
}
else
{
lean_object* v_reuseFailAlloc_709_; 
v_reuseFailAlloc_709_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_709_, 0, v_a_703_);
v___x_708_ = v_reuseFailAlloc_709_;
goto v_reusejp_707_;
}
v_reusejp_707_:
{
return v___x_708_;
}
}
}
}
else
{
lean_object* v_a_711_; lean_object* v___x_713_; uint8_t v_isShared_714_; uint8_t v_isSharedCheck_718_; 
lean_dec(v___y_676_);
lean_dec(v_n_x27_657_);
lean_dec(v_n_656_);
v_a_711_ = lean_ctor_get(v___y_677_, 0);
v_isSharedCheck_718_ = !lean_is_exclusive(v___y_677_);
if (v_isSharedCheck_718_ == 0)
{
v___x_713_ = v___y_677_;
v_isShared_714_ = v_isSharedCheck_718_;
goto v_resetjp_712_;
}
else
{
lean_inc(v_a_711_);
lean_dec(v___y_677_);
v___x_713_ = lean_box(0);
v_isShared_714_ = v_isSharedCheck_718_;
goto v_resetjp_712_;
}
v_resetjp_712_:
{
lean_object* v___x_716_; 
if (v_isShared_714_ == 0)
{
v___x_716_ = v___x_713_;
goto v_reusejp_715_;
}
else
{
lean_object* v_reuseFailAlloc_717_; 
v_reuseFailAlloc_717_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_717_, 0, v_a_711_);
v___x_716_ = v_reuseFailAlloc_717_;
goto v_reusejp_715_;
}
v_reusejp_715_:
{
return v___x_716_;
}
}
}
}
v___jp_719_:
{
if (v___y_723_ == 0)
{
lean_object* v___x_724_; 
lean_dec_ref(v___y_721_);
v___x_724_ = l_Lean_Meta_SavedState_restore___redArg(v___y_720_, v_a_659_, v_a_661_);
lean_dec_ref(v___y_720_);
if (lean_obj_tag(v___x_724_) == 0)
{
lean_object* v___x_725_; 
lean_dec_ref_known(v___x_724_, 1);
v___x_725_ = lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg(v_ty_654_, v_target_655_, v_a_658_, v_a_659_, v_a_660_, v_a_661_);
v___y_676_ = v___y_722_;
v___y_677_ = v___x_725_;
goto v___jp_675_;
}
else
{
lean_dec_ref(v_target_655_);
lean_dec_ref(v_ty_654_);
v___y_676_ = v___y_722_;
v___y_677_ = v___x_724_;
goto v___jp_675_;
}
}
else
{
lean_dec_ref(v___y_720_);
lean_dec_ref(v_target_655_);
lean_dec_ref(v_ty_654_);
v___y_676_ = v___y_722_;
v___y_677_ = v___y_721_;
goto v___jp_675_;
}
}
v___jp_726_:
{
if (lean_obj_tag(v___y_727_) == 0)
{
lean_object* v_a_728_; 
v_a_728_ = lean_ctor_get(v___y_727_, 0);
lean_inc(v_a_728_);
lean_dec_ref_known(v___y_727_, 1);
if (lean_obj_tag(v_a_728_) == 1)
{
lean_object* v_tail_729_; 
v_tail_729_ = lean_ctor_get(v_a_728_, 1);
lean_inc(v_tail_729_);
if (lean_obj_tag(v_tail_729_) == 1)
{
lean_object* v_head_730_; lean_object* v_head_731_; lean_object* v___x_732_; 
v_head_730_ = lean_ctor_get(v_a_728_, 0);
lean_inc(v_head_730_);
lean_dec_ref_known(v_a_728_, 2);
v_head_731_ = lean_ctor_get(v_tail_729_, 0);
lean_inc(v_head_731_);
lean_dec_ref_known(v_tail_729_, 2);
v___x_732_ = l_Lean_Meta_saveState___redArg(v_a_659_, v_a_661_);
if (lean_obj_tag(v___x_732_) == 0)
{
lean_object* v_a_733_; lean_object* v___x_734_; 
v_a_733_ = lean_ctor_get(v___x_732_, 0);
lean_inc(v_a_733_);
lean_dec_ref_known(v___x_732_, 1);
v___x_734_ = lp_batteries_Lean_MVarId_assignIfDefEq(v_head_731_, v_e_653_, v_a_658_, v_a_659_, v_a_660_, v_a_661_);
if (lean_obj_tag(v___x_734_) == 0)
{
lean_dec(v_a_733_);
lean_dec_ref(v_target_655_);
lean_dec_ref(v_ty_654_);
v___y_676_ = v_head_730_;
v___y_677_ = v___x_734_;
goto v___jp_675_;
}
else
{
lean_object* v_a_735_; uint8_t v___x_736_; 
v_a_735_ = lean_ctor_get(v___x_734_, 0);
lean_inc(v_a_735_);
v___x_736_ = l_Lean_Exception_isInterrupt(v_a_735_);
if (v___x_736_ == 0)
{
uint8_t v___x_737_; 
v___x_737_ = l_Lean_Exception_isRuntime(v_a_735_);
v___y_720_ = v_a_733_;
v___y_721_ = v___x_734_;
v___y_722_ = v_head_730_;
v___y_723_ = v___x_737_;
goto v___jp_719_;
}
else
{
lean_dec(v_a_735_);
v___y_720_ = v_a_733_;
v___y_721_ = v___x_734_;
v___y_722_ = v_head_730_;
v___y_723_ = v___x_736_;
goto v___jp_719_;
}
}
}
else
{
lean_object* v_a_738_; lean_object* v___x_740_; uint8_t v_isShared_741_; uint8_t v_isSharedCheck_745_; 
lean_dec(v_head_731_);
lean_dec(v_head_730_);
lean_dec(v_n_x27_657_);
lean_dec(v_n_656_);
lean_dec_ref(v_target_655_);
lean_dec_ref(v_ty_654_);
lean_dec_ref(v_e_653_);
v_a_738_ = lean_ctor_get(v___x_732_, 0);
v_isSharedCheck_745_ = !lean_is_exclusive(v___x_732_);
if (v_isSharedCheck_745_ == 0)
{
v___x_740_ = v___x_732_;
v_isShared_741_ = v_isSharedCheck_745_;
goto v_resetjp_739_;
}
else
{
lean_inc(v_a_738_);
lean_dec(v___x_732_);
v___x_740_ = lean_box(0);
v_isShared_741_ = v_isSharedCheck_745_;
goto v_resetjp_739_;
}
v_resetjp_739_:
{
lean_object* v___x_743_; 
if (v_isShared_741_ == 0)
{
v___x_743_ = v___x_740_;
goto v_reusejp_742_;
}
else
{
lean_object* v_reuseFailAlloc_744_; 
v_reuseFailAlloc_744_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_744_, 0, v_a_738_);
v___x_743_ = v_reuseFailAlloc_744_;
goto v_reusejp_742_;
}
v_reusejp_742_:
{
return v___x_743_;
}
}
}
}
else
{
lean_dec_ref_known(v_a_728_, 2);
lean_dec(v_tail_729_);
lean_dec(v_n_x27_657_);
lean_dec(v_n_656_);
lean_dec_ref(v_target_655_);
lean_dec_ref(v_ty_654_);
lean_dec_ref(v_e_653_);
v___y_664_ = v_a_658_;
v___y_665_ = v_a_659_;
v___y_666_ = v_a_660_;
v___y_667_ = v_a_661_;
goto v___jp_663_;
}
}
else
{
lean_dec(v_a_728_);
lean_dec(v_n_x27_657_);
lean_dec(v_n_656_);
lean_dec_ref(v_target_655_);
lean_dec_ref(v_ty_654_);
lean_dec_ref(v_e_653_);
v___y_664_ = v_a_658_;
v___y_665_ = v_a_659_;
v___y_666_ = v_a_660_;
v___y_667_ = v_a_661_;
goto v___jp_663_;
}
}
else
{
lean_object* v_a_746_; lean_object* v___x_748_; uint8_t v_isShared_749_; uint8_t v_isSharedCheck_753_; 
lean_dec(v_n_x27_657_);
lean_dec(v_n_656_);
lean_dec_ref(v_target_655_);
lean_dec_ref(v_ty_654_);
lean_dec_ref(v_e_653_);
v_a_746_ = lean_ctor_get(v___y_727_, 0);
v_isSharedCheck_753_ = !lean_is_exclusive(v___y_727_);
if (v_isSharedCheck_753_ == 0)
{
v___x_748_ = v___y_727_;
v_isShared_749_ = v_isSharedCheck_753_;
goto v_resetjp_747_;
}
else
{
lean_inc(v_a_746_);
lean_dec(v___y_727_);
v___x_748_ = lean_box(0);
v_isShared_749_ = v_isSharedCheck_753_;
goto v_resetjp_747_;
}
v_resetjp_747_:
{
lean_object* v___x_751_; 
if (v_isShared_749_ == 0)
{
v___x_751_ = v___x_748_;
goto v_reusejp_750_;
}
else
{
lean_object* v_reuseFailAlloc_752_; 
v_reuseFailAlloc_752_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_752_, 0, v_a_746_);
v___x_751_ = v_reuseFailAlloc_752_;
goto v_reusejp_750_;
}
v_reusejp_750_:
{
return v___x_751_;
}
}
}
}
}
else
{
lean_object* v_a_770_; lean_object* v___x_772_; uint8_t v_isShared_773_; uint8_t v_isSharedCheck_777_; 
lean_dec(v_n_x27_657_);
lean_dec(v_n_656_);
lean_dec_ref(v_target_655_);
lean_dec_ref(v_ty_654_);
lean_dec_ref(v_e_653_);
lean_dec(v_goal_652_);
lean_dec(v_thm_651_);
v_a_770_ = lean_ctor_get(v___x_672_, 0);
v_isSharedCheck_777_ = !lean_is_exclusive(v___x_672_);
if (v_isSharedCheck_777_ == 0)
{
v___x_772_ = v___x_672_;
v_isShared_773_ = v_isSharedCheck_777_;
goto v_resetjp_771_;
}
else
{
lean_inc(v_a_770_);
lean_dec(v___x_672_);
v___x_772_ = lean_box(0);
v_isShared_773_ = v_isSharedCheck_777_;
goto v_resetjp_771_;
}
v_resetjp_771_:
{
lean_object* v___x_775_; 
if (v_isShared_773_ == 0)
{
v___x_775_ = v___x_772_;
goto v_reusejp_774_;
}
else
{
lean_object* v_reuseFailAlloc_776_; 
v_reuseFailAlloc_776_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_776_, 0, v_a_770_);
v___x_775_ = v_reuseFailAlloc_776_;
goto v_reusejp_774_;
}
v_reusejp_774_:
{
return v___x_775_;
}
}
}
v___jp_663_:
{
lean_object* v___x_668_; lean_object* v___x_669_; 
v___x_668_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Peel_applyPeelThm___closed__1, &lp_mathlib_Mathlib_Tactic_Peel_applyPeelThm___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Peel_applyPeelThm___closed__1);
v___x_669_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Peel_throwPeelError_spec__0___redArg(v___x_668_, v___y_664_, v___y_665_, v___y_666_, v___y_667_);
return v___x_669_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_applyPeelThm___boxed(lean_object* v_thm_778_, lean_object* v_goal_779_, lean_object* v_e_780_, lean_object* v_ty_781_, lean_object* v_target_782_, lean_object* v_n_783_, lean_object* v_n_x27_784_, lean_object* v_a_785_, lean_object* v_a_786_, lean_object* v_a_787_, lean_object* v_a_788_, lean_object* v_a_789_){
_start:
{
lean_object* v_res_790_; 
v_res_790_ = lp_mathlib_Mathlib_Tactic_Peel_applyPeelThm(v_thm_778_, v_goal_779_, v_e_780_, v_ty_781_, v_target_782_, v_n_783_, v_n_x27_784_, v_a_785_, v_a_786_, v_a_787_, v_a_788_);
lean_dec(v_a_788_);
lean_dec_ref(v_a_787_);
lean_dec(v_a_786_);
lean_dec_ref(v_a_785_);
return v_res_790_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Peel_peelCore_spec__0___redArg(lean_object* v_mvarId_791_, lean_object* v_x_792_, lean_object* v___y_793_, lean_object* v___y_794_, lean_object* v___y_795_, lean_object* v___y_796_){
_start:
{
lean_object* v___x_798_; 
v___x_798_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_791_, v_x_792_, v___y_793_, v___y_794_, v___y_795_, v___y_796_);
if (lean_obj_tag(v___x_798_) == 0)
{
lean_object* v_a_799_; lean_object* v___x_801_; uint8_t v_isShared_802_; uint8_t v_isSharedCheck_806_; 
v_a_799_ = lean_ctor_get(v___x_798_, 0);
v_isSharedCheck_806_ = !lean_is_exclusive(v___x_798_);
if (v_isSharedCheck_806_ == 0)
{
v___x_801_ = v___x_798_;
v_isShared_802_ = v_isSharedCheck_806_;
goto v_resetjp_800_;
}
else
{
lean_inc(v_a_799_);
lean_dec(v___x_798_);
v___x_801_ = lean_box(0);
v_isShared_802_ = v_isSharedCheck_806_;
goto v_resetjp_800_;
}
v_resetjp_800_:
{
lean_object* v___x_804_; 
if (v_isShared_802_ == 0)
{
v___x_804_ = v___x_801_;
goto v_reusejp_803_;
}
else
{
lean_object* v_reuseFailAlloc_805_; 
v_reuseFailAlloc_805_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_805_, 0, v_a_799_);
v___x_804_ = v_reuseFailAlloc_805_;
goto v_reusejp_803_;
}
v_reusejp_803_:
{
return v___x_804_;
}
}
}
else
{
lean_object* v_a_807_; lean_object* v___x_809_; uint8_t v_isShared_810_; uint8_t v_isSharedCheck_814_; 
v_a_807_ = lean_ctor_get(v___x_798_, 0);
v_isSharedCheck_814_ = !lean_is_exclusive(v___x_798_);
if (v_isSharedCheck_814_ == 0)
{
v___x_809_ = v___x_798_;
v_isShared_810_ = v_isSharedCheck_814_;
goto v_resetjp_808_;
}
else
{
lean_inc(v_a_807_);
lean_dec(v___x_798_);
v___x_809_ = lean_box(0);
v_isShared_810_ = v_isSharedCheck_814_;
goto v_resetjp_808_;
}
v_resetjp_808_:
{
lean_object* v___x_812_; 
if (v_isShared_810_ == 0)
{
v___x_812_ = v___x_809_;
goto v_reusejp_811_;
}
else
{
lean_object* v_reuseFailAlloc_813_; 
v_reuseFailAlloc_813_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_813_, 0, v_a_807_);
v___x_812_ = v_reuseFailAlloc_813_;
goto v_reusejp_811_;
}
v_reusejp_811_:
{
return v___x_812_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Peel_peelCore_spec__0___redArg___boxed(lean_object* v_mvarId_815_, lean_object* v_x_816_, lean_object* v___y_817_, lean_object* v___y_818_, lean_object* v___y_819_, lean_object* v___y_820_, lean_object* v___y_821_){
_start:
{
lean_object* v_res_822_; 
v_res_822_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Peel_peelCore_spec__0___redArg(v_mvarId_815_, v_x_816_, v___y_817_, v___y_818_, v___y_819_, v___y_820_);
lean_dec(v___y_820_);
lean_dec_ref(v___y_819_);
lean_dec(v___y_818_);
lean_dec_ref(v___y_817_);
return v_res_822_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Peel_peelCore_spec__0(lean_object* v_00_u03b1_823_, lean_object* v_mvarId_824_, lean_object* v_x_825_, lean_object* v___y_826_, lean_object* v___y_827_, lean_object* v___y_828_, lean_object* v___y_829_){
_start:
{
lean_object* v___x_831_; 
v___x_831_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Peel_peelCore_spec__0___redArg(v_mvarId_824_, v_x_825_, v___y_826_, v___y_827_, v___y_828_, v___y_829_);
return v___x_831_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Peel_peelCore_spec__0___boxed(lean_object* v_00_u03b1_832_, lean_object* v_mvarId_833_, lean_object* v_x_834_, lean_object* v___y_835_, lean_object* v___y_836_, lean_object* v___y_837_, lean_object* v___y_838_, lean_object* v___y_839_){
_start:
{
lean_object* v_res_840_; 
v_res_840_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Peel_peelCore_spec__0(v_00_u03b1_832_, v_mvarId_833_, v_x_834_, v___y_835_, v___y_836_, v___y_837_, v___y_838_);
lean_dec(v___y_838_);
lean_dec_ref(v___y_837_);
lean_dec(v___y_836_);
lean_dec_ref(v___y_835_);
return v_res_840_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0(lean_object* v_e_869_, uint8_t v_unfold_870_, lean_object* v_goal_871_, lean_object* v_n_x27_872_, lean_object* v_n_x3f_873_, lean_object* v___y_874_, lean_object* v___y_875_, lean_object* v___y_876_, lean_object* v___y_877_){
_start:
{
lean_object* v___x_879_; 
lean_inc(v___y_877_);
lean_inc_ref(v___y_876_);
lean_inc(v___y_875_);
lean_inc_ref(v___y_874_);
lean_inc_ref(v_e_869_);
v___x_879_ = lean_infer_type(v_e_869_, v___y_874_, v___y_875_, v___y_876_, v___y_877_);
if (lean_obj_tag(v___x_879_) == 0)
{
lean_object* v_a_880_; lean_object* v___x_881_; 
v_a_880_ = lean_ctor_get(v___x_879_, 0);
lean_inc(v_a_880_);
lean_dec_ref_known(v___x_879_, 1);
v___x_881_ = lp_mathlib_Mathlib_Tactic_Peel_whnfQuantifier(v_a_880_, v_unfold_870_, v___y_874_, v___y_875_, v___y_876_, v___y_877_);
if (lean_obj_tag(v___x_881_) == 0)
{
lean_object* v_a_882_; lean_object* v___x_883_; 
v_a_882_ = lean_ctor_get(v___x_881_, 0);
lean_inc(v_a_882_);
lean_dec_ref_known(v___x_881_, 1);
lean_inc(v_goal_871_);
v___x_883_ = l_Lean_MVarId_getType(v_goal_871_, v___y_874_, v___y_875_, v___y_876_, v___y_877_);
if (lean_obj_tag(v___x_883_) == 0)
{
lean_object* v_a_884_; lean_object* v___x_885_; 
v_a_884_ = lean_ctor_get(v___x_883_, 0);
lean_inc(v_a_884_);
lean_dec_ref_known(v___x_883_, 1);
v___x_885_ = lp_mathlib_Mathlib_Tactic_Peel_whnfQuantifier(v_a_884_, v_unfold_870_, v___y_874_, v___y_875_, v___y_876_, v___y_877_);
if (lean_obj_tag(v___x_885_) == 0)
{
lean_object* v_a_886_; lean_object* v_a_888_; lean_object* v_a_892_; lean_object* v_a_896_; lean_object* v_a_900_; lean_object* v_a_904_; uint8_t v___y_908_; uint8_t v___y_1007_; uint8_t v___x_1025_; 
v_a_886_ = lean_ctor_get(v___x_885_, 0);
lean_inc(v_a_886_);
lean_dec_ref_known(v___x_885_, 1);
v___x_1025_ = l_Lean_Expr_isForall(v_a_882_);
if (v___x_1025_ == 0)
{
v___y_1007_ = v___x_1025_;
goto v___jp_1006_;
}
else
{
uint8_t v___x_1026_; 
v___x_1026_ = l_Lean_Expr_isForall(v_a_886_);
v___y_1007_ = v___x_1026_;
goto v___jp_1006_;
}
v___jp_887_:
{
lean_object* v___x_889_; lean_object* v___x_890_; 
v___x_889_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__1));
v___x_890_ = lp_mathlib_Mathlib_Tactic_Peel_applyPeelThm(v___x_889_, v_goal_871_, v_e_869_, v_a_882_, v_a_886_, v_a_888_, v_n_x27_872_, v___y_874_, v___y_875_, v___y_876_, v___y_877_);
lean_dec(v___y_877_);
lean_dec_ref(v___y_876_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
return v___x_890_;
}
v___jp_891_:
{
lean_object* v___x_893_; lean_object* v___x_894_; 
v___x_893_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__3));
v___x_894_ = lp_mathlib_Mathlib_Tactic_Peel_applyPeelThm(v___x_893_, v_goal_871_, v_e_869_, v_a_882_, v_a_886_, v_a_892_, v_n_x27_872_, v___y_874_, v___y_875_, v___y_876_, v___y_877_);
lean_dec(v___y_877_);
lean_dec_ref(v___y_876_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
return v___x_894_;
}
v___jp_895_:
{
lean_object* v___x_897_; lean_object* v___x_898_; 
v___x_897_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__5));
v___x_898_ = lp_mathlib_Mathlib_Tactic_Peel_applyPeelThm(v___x_897_, v_goal_871_, v_e_869_, v_a_882_, v_a_886_, v_a_896_, v_n_x27_872_, v___y_874_, v___y_875_, v___y_876_, v___y_877_);
lean_dec(v___y_877_);
lean_dec_ref(v___y_876_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
return v___x_898_;
}
v___jp_899_:
{
lean_object* v___x_901_; lean_object* v___x_902_; 
v___x_901_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__7));
v___x_902_ = lp_mathlib_Mathlib_Tactic_Peel_applyPeelThm(v___x_901_, v_goal_871_, v_e_869_, v_a_882_, v_a_886_, v_a_900_, v_n_x27_872_, v___y_874_, v___y_875_, v___y_876_, v___y_877_);
lean_dec(v___y_877_);
lean_dec_ref(v___y_876_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
return v___x_902_;
}
v___jp_903_:
{
lean_object* v___x_905_; lean_object* v___x_906_; 
v___x_905_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__9));
v___x_906_ = lp_mathlib_Mathlib_Tactic_Peel_applyPeelThm(v___x_905_, v_goal_871_, v_e_869_, v_a_882_, v_a_886_, v_a_904_, v_n_x27_872_, v___y_874_, v___y_875_, v___y_876_, v___y_877_);
lean_dec(v___y_877_);
lean_dec_ref(v___y_876_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
return v___x_906_;
}
v___jp_907_:
{
if (v___y_908_ == 0)
{
lean_object* v___x_909_; 
lean_dec(v_n_x3f_873_);
lean_dec(v_n_x27_872_);
lean_dec(v_goal_871_);
lean_dec_ref(v_e_869_);
v___x_909_ = lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg(v_a_882_, v_a_886_, v___y_874_, v___y_875_, v___y_876_, v___y_877_);
lean_dec(v___y_877_);
lean_dec_ref(v___y_876_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
return v___x_909_;
}
else
{
lean_object* v___x_910_; lean_object* v___x_911_; uint8_t v___x_912_; 
v___x_910_ = l_Lean_Expr_getAppFn(v_a_882_);
v___x_911_ = l_Lean_Expr_getAppFn(v_a_886_);
v___x_912_ = lean_expr_eqv(v___x_910_, v___x_911_);
lean_dec_ref(v___x_911_);
lean_dec_ref(v___x_910_);
if (v___x_912_ == 0)
{
lean_object* v___x_913_; 
lean_dec(v_n_x3f_873_);
lean_dec(v_n_x27_872_);
lean_dec(v_goal_871_);
lean_dec_ref(v_e_869_);
v___x_913_ = lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg(v_a_882_, v_a_886_, v___y_874_, v___y_875_, v___y_876_, v___y_877_);
lean_dec(v___y_877_);
lean_dec_ref(v___y_876_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
return v___x_913_;
}
else
{
lean_object* v___x_914_; lean_object* v_fst_915_; 
lean_inc(v_a_886_);
v___x_914_ = l_Lean_Expr_getAppFnArgs(v_a_886_);
v_fst_915_ = lean_ctor_get(v___x_914_, 0);
lean_inc(v_fst_915_);
if (lean_obj_tag(v_fst_915_) == 1)
{
lean_object* v_pre_916_; 
v_pre_916_ = lean_ctor_get(v_fst_915_, 0);
switch(lean_obj_tag(v_pre_916_))
{
case 0:
{
lean_object* v_snd_917_; lean_object* v_str_918_; lean_object* v___x_919_; uint8_t v___x_920_; 
v_snd_917_ = lean_ctor_get(v___x_914_, 1);
lean_inc(v_snd_917_);
lean_dec_ref(v___x_914_);
v_str_918_ = lean_ctor_get(v_fst_915_, 1);
lean_inc_ref(v_str_918_);
lean_dec_ref_known(v_fst_915_, 2);
v___x_919_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__0));
v___x_920_ = lean_string_dec_eq(v_str_918_, v___x_919_);
if (v___x_920_ == 0)
{
lean_object* v___x_921_; uint8_t v___x_922_; 
v___x_921_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__2));
v___x_922_ = lean_string_dec_eq(v_str_918_, v___x_921_);
lean_dec_ref(v_str_918_);
if (v___x_922_ == 0)
{
lean_object* v___x_923_; 
lean_dec(v_snd_917_);
lean_dec(v_n_x3f_873_);
lean_dec(v_n_x27_872_);
lean_dec(v_goal_871_);
lean_dec_ref(v_e_869_);
v___x_923_ = lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg(v_a_882_, v_a_886_, v___y_874_, v___y_875_, v___y_876_, v___y_877_);
lean_dec(v___y_877_);
lean_dec_ref(v___y_876_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
return v___x_923_;
}
else
{
lean_object* v___x_924_; lean_object* v___x_925_; uint8_t v___x_926_; 
v___x_924_ = lean_array_get_size(v_snd_917_);
lean_dec(v_snd_917_);
v___x_925_ = lean_unsigned_to_nat(2u);
v___x_926_ = lean_nat_dec_eq(v___x_924_, v___x_925_);
if (v___x_926_ == 0)
{
lean_object* v___x_927_; 
lean_dec(v_n_x3f_873_);
lean_dec(v_n_x27_872_);
lean_dec(v_goal_871_);
lean_dec_ref(v_e_869_);
v___x_927_ = lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg(v_a_882_, v_a_886_, v___y_874_, v___y_875_, v___y_876_, v___y_877_);
lean_dec(v___y_877_);
lean_dec_ref(v___y_876_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
return v___x_927_;
}
else
{
if (lean_obj_tag(v_n_x3f_873_) == 0)
{
lean_object* v___x_928_; lean_object* v___x_929_; 
v___x_928_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___closed__11));
v___x_929_ = l_Lean_Core_mkFreshUserName(v___x_928_, v___y_876_, v___y_877_);
if (lean_obj_tag(v___x_929_) == 0)
{
lean_object* v_a_930_; 
v_a_930_ = lean_ctor_get(v___x_929_, 0);
lean_inc(v_a_930_);
lean_dec_ref_known(v___x_929_, 1);
v_a_904_ = v_a_930_;
goto v___jp_903_;
}
else
{
lean_object* v_a_931_; lean_object* v___x_933_; uint8_t v_isShared_934_; uint8_t v_isSharedCheck_938_; 
lean_dec(v_a_886_);
lean_dec(v_a_882_);
lean_dec(v___y_877_);
lean_dec_ref(v___y_876_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
lean_dec(v_n_x27_872_);
lean_dec(v_goal_871_);
lean_dec_ref(v_e_869_);
v_a_931_ = lean_ctor_get(v___x_929_, 0);
v_isSharedCheck_938_ = !lean_is_exclusive(v___x_929_);
if (v_isSharedCheck_938_ == 0)
{
v___x_933_ = v___x_929_;
v_isShared_934_ = v_isSharedCheck_938_;
goto v_resetjp_932_;
}
else
{
lean_inc(v_a_931_);
lean_dec(v___x_929_);
v___x_933_ = lean_box(0);
v_isShared_934_ = v_isSharedCheck_938_;
goto v_resetjp_932_;
}
v_resetjp_932_:
{
lean_object* v___x_936_; 
if (v_isShared_934_ == 0)
{
v___x_936_ = v___x_933_;
goto v_reusejp_935_;
}
else
{
lean_object* v_reuseFailAlloc_937_; 
v_reuseFailAlloc_937_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_937_, 0, v_a_931_);
v___x_936_ = v_reuseFailAlloc_937_;
goto v_reusejp_935_;
}
v_reusejp_935_:
{
return v___x_936_;
}
}
}
}
else
{
lean_object* v_val_939_; 
v_val_939_ = lean_ctor_get(v_n_x3f_873_, 0);
lean_inc(v_val_939_);
lean_dec_ref_known(v_n_x3f_873_, 1);
v_a_904_ = v_val_939_;
goto v___jp_903_;
}
}
}
}
else
{
lean_object* v___x_940_; lean_object* v___x_941_; uint8_t v___x_942_; 
lean_dec_ref(v_str_918_);
v___x_940_ = lean_array_get_size(v_snd_917_);
v___x_941_ = lean_unsigned_to_nat(2u);
v___x_942_ = lean_nat_dec_eq(v___x_940_, v___x_941_);
if (v___x_942_ == 0)
{
lean_object* v___x_943_; 
lean_dec(v_snd_917_);
lean_dec(v_n_x3f_873_);
lean_dec(v_n_x27_872_);
lean_dec(v_goal_871_);
lean_dec_ref(v_e_869_);
v___x_943_ = lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg(v_a_882_, v_a_886_, v___y_874_, v___y_875_, v___y_876_, v___y_877_);
lean_dec(v___y_877_);
lean_dec_ref(v___y_876_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
return v___x_943_;
}
else
{
if (lean_obj_tag(v_n_x3f_873_) == 0)
{
lean_object* v___x_944_; lean_object* v___x_945_; lean_object* v___x_946_; 
v___x_944_ = lean_unsigned_to_nat(1u);
v___x_945_ = lean_array_fget(v_snd_917_, v___x_944_);
lean_dec(v_snd_917_);
v___x_946_ = lp_mathlib_Mathlib_Tactic_Peel_mkFreshBinderName___redArg(v___x_945_, v___y_876_, v___y_877_);
if (lean_obj_tag(v___x_946_) == 0)
{
lean_object* v_a_947_; 
v_a_947_ = lean_ctor_get(v___x_946_, 0);
lean_inc(v_a_947_);
lean_dec_ref_known(v___x_946_, 1);
v_a_900_ = v_a_947_;
goto v___jp_899_;
}
else
{
lean_object* v_a_948_; lean_object* v___x_950_; uint8_t v_isShared_951_; uint8_t v_isSharedCheck_955_; 
lean_dec(v_a_886_);
lean_dec(v_a_882_);
lean_dec(v___y_877_);
lean_dec_ref(v___y_876_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
lean_dec(v_n_x27_872_);
lean_dec(v_goal_871_);
lean_dec_ref(v_e_869_);
v_a_948_ = lean_ctor_get(v___x_946_, 0);
v_isSharedCheck_955_ = !lean_is_exclusive(v___x_946_);
if (v_isSharedCheck_955_ == 0)
{
v___x_950_ = v___x_946_;
v_isShared_951_ = v_isSharedCheck_955_;
goto v_resetjp_949_;
}
else
{
lean_inc(v_a_948_);
lean_dec(v___x_946_);
v___x_950_ = lean_box(0);
v_isShared_951_ = v_isSharedCheck_955_;
goto v_resetjp_949_;
}
v_resetjp_949_:
{
lean_object* v___x_953_; 
if (v_isShared_951_ == 0)
{
v___x_953_ = v___x_950_;
goto v_reusejp_952_;
}
else
{
lean_object* v_reuseFailAlloc_954_; 
v_reuseFailAlloc_954_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_954_, 0, v_a_948_);
v___x_953_ = v_reuseFailAlloc_954_;
goto v_reusejp_952_;
}
v_reusejp_952_:
{
return v___x_953_;
}
}
}
}
else
{
lean_object* v_val_956_; 
lean_dec(v_snd_917_);
v_val_956_ = lean_ctor_get(v_n_x3f_873_, 0);
lean_inc(v_val_956_);
lean_dec_ref_known(v_n_x3f_873_, 1);
v_a_900_ = v_val_956_;
goto v___jp_899_;
}
}
}
}
case 1:
{
lean_object* v_pre_957_; 
lean_inc_ref(v_pre_916_);
v_pre_957_ = lean_ctor_get(v_pre_916_, 0);
if (lean_obj_tag(v_pre_957_) == 0)
{
lean_object* v_snd_958_; lean_object* v_str_959_; lean_object* v_str_960_; lean_object* v___x_961_; uint8_t v___x_962_; 
v_snd_958_ = lean_ctor_get(v___x_914_, 1);
lean_inc(v_snd_958_);
lean_dec_ref(v___x_914_);
v_str_959_ = lean_ctor_get(v_fst_915_, 1);
lean_inc_ref(v_str_959_);
lean_dec_ref_known(v_fst_915_, 2);
v_str_960_ = lean_ctor_get(v_pre_916_, 1);
lean_inc_ref(v_str_960_);
lean_dec_ref_known(v_pre_916_, 2);
v___x_961_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__4));
v___x_962_ = lean_string_dec_eq(v_str_960_, v___x_961_);
lean_dec_ref(v_str_960_);
if (v___x_962_ == 0)
{
lean_object* v___x_963_; 
lean_dec_ref(v_str_959_);
lean_dec(v_snd_958_);
lean_dec(v_n_x3f_873_);
lean_dec(v_n_x27_872_);
lean_dec(v_goal_871_);
lean_dec_ref(v_e_869_);
v___x_963_ = lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg(v_a_882_, v_a_886_, v___y_874_, v___y_875_, v___y_876_, v___y_877_);
lean_dec(v___y_877_);
lean_dec_ref(v___y_876_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
return v___x_963_;
}
else
{
lean_object* v___x_964_; uint8_t v___x_965_; 
v___x_964_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__5));
v___x_965_ = lean_string_dec_eq(v_str_959_, v___x_964_);
if (v___x_965_ == 0)
{
lean_object* v___x_966_; uint8_t v___x_967_; 
v___x_966_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_quantifiers___closed__7));
v___x_967_ = lean_string_dec_eq(v_str_959_, v___x_966_);
lean_dec_ref(v_str_959_);
if (v___x_967_ == 0)
{
lean_object* v___x_968_; 
lean_dec(v_snd_958_);
lean_dec(v_n_x3f_873_);
lean_dec(v_n_x27_872_);
lean_dec(v_goal_871_);
lean_dec_ref(v_e_869_);
v___x_968_ = lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg(v_a_882_, v_a_886_, v___y_874_, v___y_875_, v___y_876_, v___y_877_);
lean_dec(v___y_877_);
lean_dec_ref(v___y_876_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
return v___x_968_;
}
else
{
lean_object* v___x_969_; lean_object* v___x_970_; uint8_t v___x_971_; 
v___x_969_ = lean_array_get_size(v_snd_958_);
v___x_970_ = lean_unsigned_to_nat(3u);
v___x_971_ = lean_nat_dec_eq(v___x_969_, v___x_970_);
if (v___x_971_ == 0)
{
lean_object* v___x_972_; 
lean_dec(v_snd_958_);
lean_dec(v_n_x3f_873_);
lean_dec(v_n_x27_872_);
lean_dec(v_goal_871_);
lean_dec_ref(v_e_869_);
v___x_972_ = lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg(v_a_882_, v_a_886_, v___y_874_, v___y_875_, v___y_876_, v___y_877_);
lean_dec(v___y_877_);
lean_dec_ref(v___y_876_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
return v___x_972_;
}
else
{
if (lean_obj_tag(v_n_x3f_873_) == 0)
{
lean_object* v___x_973_; lean_object* v___x_974_; lean_object* v___x_975_; 
v___x_973_ = lean_unsigned_to_nat(1u);
v___x_974_ = lean_array_fget(v_snd_958_, v___x_973_);
lean_dec(v_snd_958_);
v___x_975_ = lp_mathlib_Mathlib_Tactic_Peel_mkFreshBinderName___redArg(v___x_974_, v___y_876_, v___y_877_);
if (lean_obj_tag(v___x_975_) == 0)
{
lean_object* v_a_976_; 
v_a_976_ = lean_ctor_get(v___x_975_, 0);
lean_inc(v_a_976_);
lean_dec_ref_known(v___x_975_, 1);
v_a_896_ = v_a_976_;
goto v___jp_895_;
}
else
{
lean_object* v_a_977_; lean_object* v___x_979_; uint8_t v_isShared_980_; uint8_t v_isSharedCheck_984_; 
lean_dec(v_a_886_);
lean_dec(v_a_882_);
lean_dec(v___y_877_);
lean_dec_ref(v___y_876_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
lean_dec(v_n_x27_872_);
lean_dec(v_goal_871_);
lean_dec_ref(v_e_869_);
v_a_977_ = lean_ctor_get(v___x_975_, 0);
v_isSharedCheck_984_ = !lean_is_exclusive(v___x_975_);
if (v_isSharedCheck_984_ == 0)
{
v___x_979_ = v___x_975_;
v_isShared_980_ = v_isSharedCheck_984_;
goto v_resetjp_978_;
}
else
{
lean_inc(v_a_977_);
lean_dec(v___x_975_);
v___x_979_ = lean_box(0);
v_isShared_980_ = v_isSharedCheck_984_;
goto v_resetjp_978_;
}
v_resetjp_978_:
{
lean_object* v___x_982_; 
if (v_isShared_980_ == 0)
{
v___x_982_ = v___x_979_;
goto v_reusejp_981_;
}
else
{
lean_object* v_reuseFailAlloc_983_; 
v_reuseFailAlloc_983_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_983_, 0, v_a_977_);
v___x_982_ = v_reuseFailAlloc_983_;
goto v_reusejp_981_;
}
v_reusejp_981_:
{
return v___x_982_;
}
}
}
}
else
{
lean_object* v_val_985_; 
lean_dec(v_snd_958_);
v_val_985_ = lean_ctor_get(v_n_x3f_873_, 0);
lean_inc(v_val_985_);
lean_dec_ref_known(v_n_x3f_873_, 1);
v_a_896_ = v_val_985_;
goto v___jp_895_;
}
}
}
}
else
{
lean_object* v___x_986_; lean_object* v___x_987_; uint8_t v___x_988_; 
lean_dec_ref(v_str_959_);
v___x_986_ = lean_array_get_size(v_snd_958_);
v___x_987_ = lean_unsigned_to_nat(3u);
v___x_988_ = lean_nat_dec_eq(v___x_986_, v___x_987_);
if (v___x_988_ == 0)
{
lean_object* v___x_989_; 
lean_dec(v_snd_958_);
lean_dec(v_n_x3f_873_);
lean_dec(v_n_x27_872_);
lean_dec(v_goal_871_);
lean_dec_ref(v_e_869_);
v___x_989_ = lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg(v_a_882_, v_a_886_, v___y_874_, v___y_875_, v___y_876_, v___y_877_);
lean_dec(v___y_877_);
lean_dec_ref(v___y_876_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
return v___x_989_;
}
else
{
if (lean_obj_tag(v_n_x3f_873_) == 0)
{
lean_object* v___x_990_; lean_object* v___x_991_; lean_object* v___x_992_; 
v___x_990_ = lean_unsigned_to_nat(1u);
v___x_991_ = lean_array_fget(v_snd_958_, v___x_990_);
lean_dec(v_snd_958_);
v___x_992_ = lp_mathlib_Mathlib_Tactic_Peel_mkFreshBinderName___redArg(v___x_991_, v___y_876_, v___y_877_);
if (lean_obj_tag(v___x_992_) == 0)
{
lean_object* v_a_993_; 
v_a_993_ = lean_ctor_get(v___x_992_, 0);
lean_inc(v_a_993_);
lean_dec_ref_known(v___x_992_, 1);
v_a_892_ = v_a_993_;
goto v___jp_891_;
}
else
{
lean_object* v_a_994_; lean_object* v___x_996_; uint8_t v_isShared_997_; uint8_t v_isSharedCheck_1001_; 
lean_dec(v_a_886_);
lean_dec(v_a_882_);
lean_dec(v___y_877_);
lean_dec_ref(v___y_876_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
lean_dec(v_n_x27_872_);
lean_dec(v_goal_871_);
lean_dec_ref(v_e_869_);
v_a_994_ = lean_ctor_get(v___x_992_, 0);
v_isSharedCheck_1001_ = !lean_is_exclusive(v___x_992_);
if (v_isSharedCheck_1001_ == 0)
{
v___x_996_ = v___x_992_;
v_isShared_997_ = v_isSharedCheck_1001_;
goto v_resetjp_995_;
}
else
{
lean_inc(v_a_994_);
lean_dec(v___x_992_);
v___x_996_ = lean_box(0);
v_isShared_997_ = v_isSharedCheck_1001_;
goto v_resetjp_995_;
}
v_resetjp_995_:
{
lean_object* v___x_999_; 
if (v_isShared_997_ == 0)
{
v___x_999_ = v___x_996_;
goto v_reusejp_998_;
}
else
{
lean_object* v_reuseFailAlloc_1000_; 
v_reuseFailAlloc_1000_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1000_, 0, v_a_994_);
v___x_999_ = v_reuseFailAlloc_1000_;
goto v_reusejp_998_;
}
v_reusejp_998_:
{
return v___x_999_;
}
}
}
}
else
{
lean_object* v_val_1002_; 
lean_dec(v_snd_958_);
v_val_1002_ = lean_ctor_get(v_n_x3f_873_, 0);
lean_inc(v_val_1002_);
lean_dec_ref_known(v_n_x3f_873_, 1);
v_a_892_ = v_val_1002_;
goto v___jp_891_;
}
}
}
}
}
else
{
lean_object* v___x_1003_; 
lean_dec_ref_known(v_pre_916_, 2);
lean_dec_ref_known(v_fst_915_, 2);
lean_dec_ref(v___x_914_);
lean_dec(v_n_x3f_873_);
lean_dec(v_n_x27_872_);
lean_dec(v_goal_871_);
lean_dec_ref(v_e_869_);
v___x_1003_ = lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg(v_a_882_, v_a_886_, v___y_874_, v___y_875_, v___y_876_, v___y_877_);
lean_dec(v___y_877_);
lean_dec_ref(v___y_876_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
return v___x_1003_;
}
}
default: 
{
lean_object* v___x_1004_; 
lean_dec_ref_known(v_fst_915_, 2);
lean_dec_ref(v___x_914_);
lean_dec(v_n_x3f_873_);
lean_dec(v_n_x27_872_);
lean_dec(v_goal_871_);
lean_dec_ref(v_e_869_);
v___x_1004_ = lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg(v_a_882_, v_a_886_, v___y_874_, v___y_875_, v___y_876_, v___y_877_);
lean_dec(v___y_877_);
lean_dec_ref(v___y_876_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
return v___x_1004_;
}
}
}
else
{
lean_object* v___x_1005_; 
lean_dec(v_fst_915_);
lean_dec_ref(v___x_914_);
lean_dec(v_n_x3f_873_);
lean_dec(v_n_x27_872_);
lean_dec(v_goal_871_);
lean_dec_ref(v_e_869_);
v___x_1005_ = lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg(v_a_882_, v_a_886_, v___y_874_, v___y_875_, v___y_876_, v___y_877_);
lean_dec(v___y_877_);
lean_dec_ref(v___y_876_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
return v___x_1005_;
}
}
}
}
v___jp_1006_:
{
if (v___y_1007_ == 0)
{
lean_object* v___x_1008_; uint8_t v___x_1009_; 
v___x_1008_ = l_Lean_Expr_getAppFn(v_a_882_);
v___x_1009_ = l_Lean_Expr_isConst(v___x_1008_);
lean_dec_ref(v___x_1008_);
if (v___x_1009_ == 0)
{
v___y_908_ = v___x_1009_;
goto v___jp_907_;
}
else
{
lean_object* v___x_1010_; lean_object* v___x_1011_; uint8_t v___x_1012_; 
v___x_1010_ = l_Lean_Expr_getAppNumArgs(v_a_882_);
v___x_1011_ = l_Lean_Expr_getAppNumArgs(v_a_886_);
v___x_1012_ = lean_nat_dec_eq(v___x_1010_, v___x_1011_);
lean_dec(v___x_1011_);
lean_dec(v___x_1010_);
v___y_908_ = v___x_1012_;
goto v___jp_907_;
}
}
else
{
if (lean_obj_tag(v_n_x3f_873_) == 0)
{
lean_object* v___x_1013_; lean_object* v___x_1014_; 
v___x_1013_ = l_Lean_Expr_bindingName_x21(v_a_886_);
v___x_1014_ = l_Lean_Core_mkFreshUserName(v___x_1013_, v___y_876_, v___y_877_);
if (lean_obj_tag(v___x_1014_) == 0)
{
lean_object* v_a_1015_; 
v_a_1015_ = lean_ctor_get(v___x_1014_, 0);
lean_inc(v_a_1015_);
lean_dec_ref_known(v___x_1014_, 1);
v_a_888_ = v_a_1015_;
goto v___jp_887_;
}
else
{
lean_object* v_a_1016_; lean_object* v___x_1018_; uint8_t v_isShared_1019_; uint8_t v_isSharedCheck_1023_; 
lean_dec(v_a_886_);
lean_dec(v_a_882_);
lean_dec(v___y_877_);
lean_dec_ref(v___y_876_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
lean_dec(v_n_x27_872_);
lean_dec(v_goal_871_);
lean_dec_ref(v_e_869_);
v_a_1016_ = lean_ctor_get(v___x_1014_, 0);
v_isSharedCheck_1023_ = !lean_is_exclusive(v___x_1014_);
if (v_isSharedCheck_1023_ == 0)
{
v___x_1018_ = v___x_1014_;
v_isShared_1019_ = v_isSharedCheck_1023_;
goto v_resetjp_1017_;
}
else
{
lean_inc(v_a_1016_);
lean_dec(v___x_1014_);
v___x_1018_ = lean_box(0);
v_isShared_1019_ = v_isSharedCheck_1023_;
goto v_resetjp_1017_;
}
v_resetjp_1017_:
{
lean_object* v___x_1021_; 
if (v_isShared_1019_ == 0)
{
v___x_1021_ = v___x_1018_;
goto v_reusejp_1020_;
}
else
{
lean_object* v_reuseFailAlloc_1022_; 
v_reuseFailAlloc_1022_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1022_, 0, v_a_1016_);
v___x_1021_ = v_reuseFailAlloc_1022_;
goto v_reusejp_1020_;
}
v_reusejp_1020_:
{
return v___x_1021_;
}
}
}
}
else
{
lean_object* v_val_1024_; 
v_val_1024_ = lean_ctor_get(v_n_x3f_873_, 0);
lean_inc(v_val_1024_);
lean_dec_ref_known(v_n_x3f_873_, 1);
v_a_888_ = v_val_1024_;
goto v___jp_887_;
}
}
}
}
else
{
lean_object* v_a_1027_; lean_object* v___x_1029_; uint8_t v_isShared_1030_; uint8_t v_isSharedCheck_1034_; 
lean_dec(v_a_882_);
lean_dec(v___y_877_);
lean_dec_ref(v___y_876_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
lean_dec(v_n_x3f_873_);
lean_dec(v_n_x27_872_);
lean_dec(v_goal_871_);
lean_dec_ref(v_e_869_);
v_a_1027_ = lean_ctor_get(v___x_885_, 0);
v_isSharedCheck_1034_ = !lean_is_exclusive(v___x_885_);
if (v_isSharedCheck_1034_ == 0)
{
v___x_1029_ = v___x_885_;
v_isShared_1030_ = v_isSharedCheck_1034_;
goto v_resetjp_1028_;
}
else
{
lean_inc(v_a_1027_);
lean_dec(v___x_885_);
v___x_1029_ = lean_box(0);
v_isShared_1030_ = v_isSharedCheck_1034_;
goto v_resetjp_1028_;
}
v_resetjp_1028_:
{
lean_object* v___x_1032_; 
if (v_isShared_1030_ == 0)
{
v___x_1032_ = v___x_1029_;
goto v_reusejp_1031_;
}
else
{
lean_object* v_reuseFailAlloc_1033_; 
v_reuseFailAlloc_1033_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1033_, 0, v_a_1027_);
v___x_1032_ = v_reuseFailAlloc_1033_;
goto v_reusejp_1031_;
}
v_reusejp_1031_:
{
return v___x_1032_;
}
}
}
}
else
{
lean_object* v_a_1035_; lean_object* v___x_1037_; uint8_t v_isShared_1038_; uint8_t v_isSharedCheck_1042_; 
lean_dec(v_a_882_);
lean_dec(v___y_877_);
lean_dec_ref(v___y_876_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
lean_dec(v_n_x3f_873_);
lean_dec(v_n_x27_872_);
lean_dec(v_goal_871_);
lean_dec_ref(v_e_869_);
v_a_1035_ = lean_ctor_get(v___x_883_, 0);
v_isSharedCheck_1042_ = !lean_is_exclusive(v___x_883_);
if (v_isSharedCheck_1042_ == 0)
{
v___x_1037_ = v___x_883_;
v_isShared_1038_ = v_isSharedCheck_1042_;
goto v_resetjp_1036_;
}
else
{
lean_inc(v_a_1035_);
lean_dec(v___x_883_);
v___x_1037_ = lean_box(0);
v_isShared_1038_ = v_isSharedCheck_1042_;
goto v_resetjp_1036_;
}
v_resetjp_1036_:
{
lean_object* v___x_1040_; 
if (v_isShared_1038_ == 0)
{
v___x_1040_ = v___x_1037_;
goto v_reusejp_1039_;
}
else
{
lean_object* v_reuseFailAlloc_1041_; 
v_reuseFailAlloc_1041_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1041_, 0, v_a_1035_);
v___x_1040_ = v_reuseFailAlloc_1041_;
goto v_reusejp_1039_;
}
v_reusejp_1039_:
{
return v___x_1040_;
}
}
}
}
else
{
lean_object* v_a_1043_; lean_object* v___x_1045_; uint8_t v_isShared_1046_; uint8_t v_isSharedCheck_1050_; 
lean_dec(v___y_877_);
lean_dec_ref(v___y_876_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
lean_dec(v_n_x3f_873_);
lean_dec(v_n_x27_872_);
lean_dec(v_goal_871_);
lean_dec_ref(v_e_869_);
v_a_1043_ = lean_ctor_get(v___x_881_, 0);
v_isSharedCheck_1050_ = !lean_is_exclusive(v___x_881_);
if (v_isSharedCheck_1050_ == 0)
{
v___x_1045_ = v___x_881_;
v_isShared_1046_ = v_isSharedCheck_1050_;
goto v_resetjp_1044_;
}
else
{
lean_inc(v_a_1043_);
lean_dec(v___x_881_);
v___x_1045_ = lean_box(0);
v_isShared_1046_ = v_isSharedCheck_1050_;
goto v_resetjp_1044_;
}
v_resetjp_1044_:
{
lean_object* v___x_1048_; 
if (v_isShared_1046_ == 0)
{
v___x_1048_ = v___x_1045_;
goto v_reusejp_1047_;
}
else
{
lean_object* v_reuseFailAlloc_1049_; 
v_reuseFailAlloc_1049_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1049_, 0, v_a_1043_);
v___x_1048_ = v_reuseFailAlloc_1049_;
goto v_reusejp_1047_;
}
v_reusejp_1047_:
{
return v___x_1048_;
}
}
}
}
else
{
lean_object* v_a_1051_; lean_object* v___x_1053_; uint8_t v_isShared_1054_; uint8_t v_isSharedCheck_1058_; 
lean_dec(v___y_877_);
lean_dec_ref(v___y_876_);
lean_dec(v___y_875_);
lean_dec_ref(v___y_874_);
lean_dec(v_n_x3f_873_);
lean_dec(v_n_x27_872_);
lean_dec(v_goal_871_);
lean_dec_ref(v_e_869_);
v_a_1051_ = lean_ctor_get(v___x_879_, 0);
v_isSharedCheck_1058_ = !lean_is_exclusive(v___x_879_);
if (v_isSharedCheck_1058_ == 0)
{
v___x_1053_ = v___x_879_;
v_isShared_1054_ = v_isSharedCheck_1058_;
goto v_resetjp_1052_;
}
else
{
lean_inc(v_a_1051_);
lean_dec(v___x_879_);
v___x_1053_ = lean_box(0);
v_isShared_1054_ = v_isSharedCheck_1058_;
goto v_resetjp_1052_;
}
v_resetjp_1052_:
{
lean_object* v___x_1056_; 
if (v_isShared_1054_ == 0)
{
v___x_1056_ = v___x_1053_;
goto v_reusejp_1055_;
}
else
{
lean_object* v_reuseFailAlloc_1057_; 
v_reuseFailAlloc_1057_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1057_, 0, v_a_1051_);
v___x_1056_ = v_reuseFailAlloc_1057_;
goto v_reusejp_1055_;
}
v_reusejp_1055_:
{
return v___x_1056_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___boxed(lean_object* v_e_1059_, lean_object* v_unfold_1060_, lean_object* v_goal_1061_, lean_object* v_n_x27_1062_, lean_object* v_n_x3f_1063_, lean_object* v___y_1064_, lean_object* v___y_1065_, lean_object* v___y_1066_, lean_object* v___y_1067_, lean_object* v___y_1068_){
_start:
{
uint8_t v_unfold_boxed_1069_; lean_object* v_res_1070_; 
v_unfold_boxed_1069_ = lean_unbox(v_unfold_1060_);
v_res_1070_ = lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0(v_e_1059_, v_unfold_boxed_1069_, v_goal_1061_, v_n_x27_1062_, v_n_x3f_1063_, v___y_1064_, v___y_1065_, v___y_1066_, v___y_1067_);
return v_res_1070_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelCore(lean_object* v_goal_1071_, lean_object* v_e_1072_, lean_object* v_n_x3f_1073_, lean_object* v_n_x27_1074_, uint8_t v_unfold_1075_, lean_object* v_a_1076_, lean_object* v_a_1077_, lean_object* v_a_1078_, lean_object* v_a_1079_){
_start:
{
lean_object* v___x_1081_; lean_object* v___f_1082_; lean_object* v___x_1083_; 
v___x_1081_ = lean_box(v_unfold_1075_);
lean_inc(v_goal_1071_);
v___f_1082_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Peel_peelCore___lam__0___boxed), 10, 5);
lean_closure_set(v___f_1082_, 0, v_e_1072_);
lean_closure_set(v___f_1082_, 1, v___x_1081_);
lean_closure_set(v___f_1082_, 2, v_goal_1071_);
lean_closure_set(v___f_1082_, 3, v_n_x27_1074_);
lean_closure_set(v___f_1082_, 4, v_n_x3f_1073_);
v___x_1083_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Peel_peelCore_spec__0___redArg(v_goal_1071_, v___f_1082_, v_a_1076_, v_a_1077_, v_a_1078_, v_a_1079_);
return v___x_1083_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelCore___boxed(lean_object* v_goal_1084_, lean_object* v_e_1085_, lean_object* v_n_x3f_1086_, lean_object* v_n_x27_1087_, lean_object* v_unfold_1088_, lean_object* v_a_1089_, lean_object* v_a_1090_, lean_object* v_a_1091_, lean_object* v_a_1092_, lean_object* v_a_1093_){
_start:
{
uint8_t v_unfold_boxed_1094_; lean_object* v_res_1095_; 
v_unfold_boxed_1094_ = lean_unbox(v_unfold_1088_);
v_res_1095_ = lp_mathlib_Mathlib_Tactic_Peel_peelCore(v_goal_1084_, v_e_1085_, v_n_x3f_1086_, v_n_x27_1087_, v_unfold_boxed_1094_, v_a_1089_, v_a_1090_, v_a_1091_, v_a_1092_);
lean_dec(v_a_1092_);
lean_dec_ref(v_a_1091_);
lean_dec(v_a_1090_);
lean_dec_ref(v_a_1089_);
return v_res_1095_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Peel_peelArgs_spec__0___redArg(lean_object* v_x_1096_, lean_object* v___y_1097_, lean_object* v___y_1098_, lean_object* v___y_1099_, lean_object* v___y_1100_, lean_object* v___y_1101_, lean_object* v___y_1102_, lean_object* v___y_1103_, lean_object* v___y_1104_){
_start:
{
lean_object* v___x_1106_; 
v___x_1106_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_1098_, v___y_1100_, v___y_1102_, v___y_1104_);
if (lean_obj_tag(v___x_1106_) == 0)
{
lean_object* v_a_1107_; lean_object* v___x_1108_; 
v_a_1107_ = lean_ctor_get(v___x_1106_, 0);
lean_inc(v_a_1107_);
lean_dec_ref_known(v___x_1106_, 1);
v___x_1108_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_1098_, v___y_1100_, v___y_1102_, v___y_1104_);
if (lean_obj_tag(v___x_1108_) == 0)
{
lean_object* v_a_1109_; lean_object* v___x_1110_; 
v_a_1109_ = lean_ctor_get(v___x_1108_, 0);
lean_inc(v_a_1109_);
lean_dec_ref_known(v___x_1108_, 1);
lean_inc(v___y_1104_);
lean_inc_ref(v___y_1103_);
lean_inc(v___y_1102_);
lean_inc_ref(v___y_1101_);
lean_inc(v___y_1100_);
lean_inc_ref(v___y_1099_);
lean_inc(v___y_1098_);
lean_inc_ref(v___y_1097_);
v___x_1110_ = lean_apply_9(v_x_1096_, v___y_1097_, v___y_1098_, v___y_1099_, v___y_1100_, v___y_1101_, v___y_1102_, v___y_1103_, v___y_1104_, lean_box(0));
if (lean_obj_tag(v___x_1110_) == 0)
{
lean_object* v_a_1111_; lean_object* v___x_1113_; uint8_t v_isShared_1114_; uint8_t v_isSharedCheck_1119_; 
lean_dec(v_a_1109_);
lean_dec(v_a_1107_);
v_a_1111_ = lean_ctor_get(v___x_1110_, 0);
v_isSharedCheck_1119_ = !lean_is_exclusive(v___x_1110_);
if (v_isSharedCheck_1119_ == 0)
{
v___x_1113_ = v___x_1110_;
v_isShared_1114_ = v_isSharedCheck_1119_;
goto v_resetjp_1112_;
}
else
{
lean_inc(v_a_1111_);
lean_dec(v___x_1110_);
v___x_1113_ = lean_box(0);
v_isShared_1114_ = v_isSharedCheck_1119_;
goto v_resetjp_1112_;
}
v_resetjp_1112_:
{
lean_object* v___x_1115_; lean_object* v___x_1117_; 
v___x_1115_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1115_, 0, v_a_1111_);
if (v_isShared_1114_ == 0)
{
lean_ctor_set(v___x_1113_, 0, v___x_1115_);
v___x_1117_ = v___x_1113_;
goto v_reusejp_1116_;
}
else
{
lean_object* v_reuseFailAlloc_1118_; 
v_reuseFailAlloc_1118_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1118_, 0, v___x_1115_);
v___x_1117_ = v_reuseFailAlloc_1118_;
goto v_reusejp_1116_;
}
v_reusejp_1116_:
{
return v___x_1117_;
}
}
}
else
{
lean_object* v_a_1120_; lean_object* v___x_1122_; uint8_t v_isShared_1123_; uint8_t v_isSharedCheck_1158_; 
v_a_1120_ = lean_ctor_get(v___x_1110_, 0);
v_isSharedCheck_1158_ = !lean_is_exclusive(v___x_1110_);
if (v_isSharedCheck_1158_ == 0)
{
v___x_1122_ = v___x_1110_;
v_isShared_1123_ = v_isSharedCheck_1158_;
goto v_resetjp_1121_;
}
else
{
lean_inc(v_a_1120_);
lean_dec(v___x_1110_);
v___x_1122_ = lean_box(0);
v_isShared_1123_ = v_isSharedCheck_1158_;
goto v_resetjp_1121_;
}
v_resetjp_1121_:
{
uint8_t v___y_1125_; uint8_t v___x_1156_; 
v___x_1156_ = l_Lean_Exception_isInterrupt(v_a_1120_);
if (v___x_1156_ == 0)
{
uint8_t v___x_1157_; 
lean_inc(v_a_1120_);
v___x_1157_ = l_Lean_Exception_isRuntime(v_a_1120_);
v___y_1125_ = v___x_1157_;
goto v___jp_1124_;
}
else
{
v___y_1125_ = v___x_1156_;
goto v___jp_1124_;
}
v___jp_1124_:
{
if (v___y_1125_ == 0)
{
lean_object* v___x_1126_; 
lean_del_object(v___x_1122_);
lean_dec(v_a_1120_);
v___x_1126_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_1109_, v___y_1125_, v___y_1098_, v___y_1099_, v___y_1100_, v___y_1101_, v___y_1102_, v___y_1103_, v___y_1104_);
if (lean_obj_tag(v___x_1126_) == 0)
{
lean_object* v___x_1127_; 
lean_dec_ref_known(v___x_1126_, 1);
v___x_1127_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_1107_, v___y_1125_, v___y_1098_, v___y_1099_, v___y_1100_, v___y_1101_, v___y_1102_, v___y_1103_, v___y_1104_);
if (lean_obj_tag(v___x_1127_) == 0)
{
lean_object* v___x_1129_; uint8_t v_isShared_1130_; uint8_t v_isSharedCheck_1135_; 
v_isSharedCheck_1135_ = !lean_is_exclusive(v___x_1127_);
if (v_isSharedCheck_1135_ == 0)
{
lean_object* v_unused_1136_; 
v_unused_1136_ = lean_ctor_get(v___x_1127_, 0);
lean_dec(v_unused_1136_);
v___x_1129_ = v___x_1127_;
v_isShared_1130_ = v_isSharedCheck_1135_;
goto v_resetjp_1128_;
}
else
{
lean_dec(v___x_1127_);
v___x_1129_ = lean_box(0);
v_isShared_1130_ = v_isSharedCheck_1135_;
goto v_resetjp_1128_;
}
v_resetjp_1128_:
{
lean_object* v___x_1131_; lean_object* v___x_1133_; 
v___x_1131_ = lean_box(0);
if (v_isShared_1130_ == 0)
{
lean_ctor_set(v___x_1129_, 0, v___x_1131_);
v___x_1133_ = v___x_1129_;
goto v_reusejp_1132_;
}
else
{
lean_object* v_reuseFailAlloc_1134_; 
v_reuseFailAlloc_1134_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1134_, 0, v___x_1131_);
v___x_1133_ = v_reuseFailAlloc_1134_;
goto v_reusejp_1132_;
}
v_reusejp_1132_:
{
return v___x_1133_;
}
}
}
else
{
lean_object* v_a_1137_; lean_object* v___x_1139_; uint8_t v_isShared_1140_; uint8_t v_isSharedCheck_1144_; 
v_a_1137_ = lean_ctor_get(v___x_1127_, 0);
v_isSharedCheck_1144_ = !lean_is_exclusive(v___x_1127_);
if (v_isSharedCheck_1144_ == 0)
{
v___x_1139_ = v___x_1127_;
v_isShared_1140_ = v_isSharedCheck_1144_;
goto v_resetjp_1138_;
}
else
{
lean_inc(v_a_1137_);
lean_dec(v___x_1127_);
v___x_1139_ = lean_box(0);
v_isShared_1140_ = v_isSharedCheck_1144_;
goto v_resetjp_1138_;
}
v_resetjp_1138_:
{
lean_object* v___x_1142_; 
if (v_isShared_1140_ == 0)
{
v___x_1142_ = v___x_1139_;
goto v_reusejp_1141_;
}
else
{
lean_object* v_reuseFailAlloc_1143_; 
v_reuseFailAlloc_1143_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1143_, 0, v_a_1137_);
v___x_1142_ = v_reuseFailAlloc_1143_;
goto v_reusejp_1141_;
}
v_reusejp_1141_:
{
return v___x_1142_;
}
}
}
}
else
{
lean_object* v_a_1145_; lean_object* v___x_1147_; uint8_t v_isShared_1148_; uint8_t v_isSharedCheck_1152_; 
lean_dec(v_a_1107_);
v_a_1145_ = lean_ctor_get(v___x_1126_, 0);
v_isSharedCheck_1152_ = !lean_is_exclusive(v___x_1126_);
if (v_isSharedCheck_1152_ == 0)
{
v___x_1147_ = v___x_1126_;
v_isShared_1148_ = v_isSharedCheck_1152_;
goto v_resetjp_1146_;
}
else
{
lean_inc(v_a_1145_);
lean_dec(v___x_1126_);
v___x_1147_ = lean_box(0);
v_isShared_1148_ = v_isSharedCheck_1152_;
goto v_resetjp_1146_;
}
v_resetjp_1146_:
{
lean_object* v___x_1150_; 
if (v_isShared_1148_ == 0)
{
v___x_1150_ = v___x_1147_;
goto v_reusejp_1149_;
}
else
{
lean_object* v_reuseFailAlloc_1151_; 
v_reuseFailAlloc_1151_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1151_, 0, v_a_1145_);
v___x_1150_ = v_reuseFailAlloc_1151_;
goto v_reusejp_1149_;
}
v_reusejp_1149_:
{
return v___x_1150_;
}
}
}
}
else
{
lean_object* v___x_1154_; 
lean_dec(v_a_1109_);
lean_dec(v_a_1107_);
if (v_isShared_1123_ == 0)
{
v___x_1154_ = v___x_1122_;
goto v_reusejp_1153_;
}
else
{
lean_object* v_reuseFailAlloc_1155_; 
v_reuseFailAlloc_1155_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1155_, 0, v_a_1120_);
v___x_1154_ = v_reuseFailAlloc_1155_;
goto v_reusejp_1153_;
}
v_reusejp_1153_:
{
return v___x_1154_;
}
}
}
}
}
}
else
{
lean_object* v_a_1159_; lean_object* v___x_1161_; uint8_t v_isShared_1162_; uint8_t v_isSharedCheck_1166_; 
lean_dec(v_a_1107_);
lean_dec_ref(v_x_1096_);
v_a_1159_ = lean_ctor_get(v___x_1108_, 0);
v_isSharedCheck_1166_ = !lean_is_exclusive(v___x_1108_);
if (v_isSharedCheck_1166_ == 0)
{
v___x_1161_ = v___x_1108_;
v_isShared_1162_ = v_isSharedCheck_1166_;
goto v_resetjp_1160_;
}
else
{
lean_inc(v_a_1159_);
lean_dec(v___x_1108_);
v___x_1161_ = lean_box(0);
v_isShared_1162_ = v_isSharedCheck_1166_;
goto v_resetjp_1160_;
}
v_resetjp_1160_:
{
lean_object* v___x_1164_; 
if (v_isShared_1162_ == 0)
{
v___x_1164_ = v___x_1161_;
goto v_reusejp_1163_;
}
else
{
lean_object* v_reuseFailAlloc_1165_; 
v_reuseFailAlloc_1165_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1165_, 0, v_a_1159_);
v___x_1164_ = v_reuseFailAlloc_1165_;
goto v_reusejp_1163_;
}
v_reusejp_1163_:
{
return v___x_1164_;
}
}
}
}
else
{
lean_object* v_a_1167_; lean_object* v___x_1169_; uint8_t v_isShared_1170_; uint8_t v_isSharedCheck_1174_; 
lean_dec_ref(v_x_1096_);
v_a_1167_ = lean_ctor_get(v___x_1106_, 0);
v_isSharedCheck_1174_ = !lean_is_exclusive(v___x_1106_);
if (v_isSharedCheck_1174_ == 0)
{
v___x_1169_ = v___x_1106_;
v_isShared_1170_ = v_isSharedCheck_1174_;
goto v_resetjp_1168_;
}
else
{
lean_inc(v_a_1167_);
lean_dec(v___x_1106_);
v___x_1169_ = lean_box(0);
v_isShared_1170_ = v_isSharedCheck_1174_;
goto v_resetjp_1168_;
}
v_resetjp_1168_:
{
lean_object* v___x_1172_; 
if (v_isShared_1170_ == 0)
{
v___x_1172_ = v___x_1169_;
goto v_reusejp_1171_;
}
else
{
lean_object* v_reuseFailAlloc_1173_; 
v_reuseFailAlloc_1173_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1173_, 0, v_a_1167_);
v___x_1172_ = v_reuseFailAlloc_1173_;
goto v_reusejp_1171_;
}
v_reusejp_1171_:
{
return v___x_1172_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Peel_peelArgs_spec__0___redArg___boxed(lean_object* v_x_1175_, lean_object* v___y_1176_, lean_object* v___y_1177_, lean_object* v___y_1178_, lean_object* v___y_1179_, lean_object* v___y_1180_, lean_object* v___y_1181_, lean_object* v___y_1182_, lean_object* v___y_1183_, lean_object* v___y_1184_){
_start:
{
lean_object* v_res_1185_; 
v_res_1185_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Peel_peelArgs_spec__0___redArg(v_x_1175_, v___y_1176_, v___y_1177_, v___y_1178_, v___y_1179_, v___y_1180_, v___y_1181_, v___y_1182_, v___y_1183_);
lean_dec(v___y_1183_);
lean_dec_ref(v___y_1182_);
lean_dec(v___y_1181_);
lean_dec_ref(v___y_1180_);
lean_dec(v___y_1179_);
lean_dec_ref(v___y_1178_);
lean_dec(v___y_1177_);
lean_dec_ref(v___y_1176_);
return v_res_1185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Peel_peelArgs_spec__0(lean_object* v_00_u03b1_1186_, lean_object* v_x_1187_, lean_object* v___y_1188_, lean_object* v___y_1189_, lean_object* v___y_1190_, lean_object* v___y_1191_, lean_object* v___y_1192_, lean_object* v___y_1193_, lean_object* v___y_1194_, lean_object* v___y_1195_){
_start:
{
lean_object* v___x_1197_; 
v___x_1197_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Peel_peelArgs_spec__0___redArg(v_x_1187_, v___y_1188_, v___y_1189_, v___y_1190_, v___y_1191_, v___y_1192_, v___y_1193_, v___y_1194_, v___y_1195_);
return v___x_1197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Peel_peelArgs_spec__0___boxed(lean_object* v_00_u03b1_1198_, lean_object* v_x_1199_, lean_object* v___y_1200_, lean_object* v___y_1201_, lean_object* v___y_1202_, lean_object* v___y_1203_, lean_object* v___y_1204_, lean_object* v___y_1205_, lean_object* v___y_1206_, lean_object* v___y_1207_, lean_object* v___y_1208_){
_start:
{
lean_object* v_res_1209_; 
v_res_1209_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Peel_peelArgs_spec__0(v_00_u03b1_1198_, v_x_1199_, v___y_1200_, v___y_1201_, v___y_1202_, v___y_1203_, v___y_1204_, v___y_1205_, v___y_1206_, v___y_1207_);
lean_dec(v___y_1207_);
lean_dec_ref(v___y_1206_);
lean_dec(v___y_1205_);
lean_dec_ref(v___y_1204_);
lean_dec(v___y_1203_);
lean_dec_ref(v___y_1202_);
lean_dec(v___y_1201_);
lean_dec_ref(v___y_1200_);
return v_res_1209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelArgs___lam__0(lean_object* v_l_1213_, lean_object* v_n_x3f_1214_, lean_object* v_e_1215_, uint8_t v_unfold_1216_, lean_object* v___y_1217_, lean_object* v___y_1218_, lean_object* v___y_1219_, lean_object* v___y_1220_, lean_object* v___y_1221_, lean_object* v___y_1222_, lean_object* v___y_1223_, lean_object* v___y_1224_){
_start:
{
lean_object* v___y_1227_; lean_object* v___x_1256_; 
v___x_1256_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_1218_, v___y_1221_, v___y_1222_, v___y_1223_, v___y_1224_);
if (lean_obj_tag(v___x_1256_) == 0)
{
lean_object* v_a_1257_; lean_object* v___x_1258_; 
v_a_1257_ = lean_ctor_get(v___x_1256_, 0);
lean_inc(v_a_1257_);
lean_dec_ref_known(v___x_1256_, 1);
v___x_1258_ = l_List_head_x3f___redArg(v_l_1213_);
if (lean_obj_tag(v_n_x3f_1214_) == 0)
{
lean_object* v___x_1259_; lean_object* v___x_1260_; 
v___x_1259_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelArgs___lam__0___closed__1));
v___x_1260_ = lp_mathlib_Mathlib_Tactic_Peel_peelCore(v_a_1257_, v_e_1215_, v___x_1258_, v___x_1259_, v_unfold_1216_, v___y_1221_, v___y_1222_, v___y_1223_, v___y_1224_);
v___y_1227_ = v___x_1260_;
goto v___jp_1226_;
}
else
{
lean_object* v_val_1261_; lean_object* v___x_1262_; 
v_val_1261_ = lean_ctor_get(v_n_x3f_1214_, 0);
lean_inc(v_val_1261_);
lean_dec_ref_known(v_n_x3f_1214_, 1);
v___x_1262_ = lp_mathlib_Mathlib_Tactic_Peel_peelCore(v_a_1257_, v_e_1215_, v___x_1258_, v_val_1261_, v_unfold_1216_, v___y_1221_, v___y_1222_, v___y_1223_, v___y_1224_);
v___y_1227_ = v___x_1262_;
goto v___jp_1226_;
}
}
else
{
lean_object* v_a_1263_; lean_object* v___x_1265_; uint8_t v_isShared_1266_; uint8_t v_isSharedCheck_1270_; 
lean_dec_ref(v_e_1215_);
lean_dec(v_n_x3f_1214_);
v_a_1263_ = lean_ctor_get(v___x_1256_, 0);
v_isSharedCheck_1270_ = !lean_is_exclusive(v___x_1256_);
if (v_isSharedCheck_1270_ == 0)
{
v___x_1265_ = v___x_1256_;
v_isShared_1266_ = v_isSharedCheck_1270_;
goto v_resetjp_1264_;
}
else
{
lean_inc(v_a_1263_);
lean_dec(v___x_1256_);
v___x_1265_ = lean_box(0);
v_isShared_1266_ = v_isSharedCheck_1270_;
goto v_resetjp_1264_;
}
v_resetjp_1264_:
{
lean_object* v___x_1268_; 
if (v_isShared_1266_ == 0)
{
v___x_1268_ = v___x_1265_;
goto v_reusejp_1267_;
}
else
{
lean_object* v_reuseFailAlloc_1269_; 
v_reuseFailAlloc_1269_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1269_, 0, v_a_1263_);
v___x_1268_ = v_reuseFailAlloc_1269_;
goto v_reusejp_1267_;
}
v_reusejp_1267_:
{
return v___x_1268_;
}
}
}
v___jp_1226_:
{
if (lean_obj_tag(v___y_1227_) == 0)
{
lean_object* v_a_1228_; lean_object* v_fst_1229_; lean_object* v_snd_1230_; lean_object* v___x_1231_; 
v_a_1228_ = lean_ctor_get(v___y_1227_, 0);
lean_inc(v_a_1228_);
lean_dec_ref_known(v___y_1227_, 1);
v_fst_1229_ = lean_ctor_get(v_a_1228_, 0);
lean_inc(v_fst_1229_);
v_snd_1230_ = lean_ctor_get(v_a_1228_, 1);
lean_inc(v_snd_1230_);
lean_dec(v_a_1228_);
v___x_1231_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v_snd_1230_, v___y_1218_, v___y_1221_, v___y_1222_, v___y_1223_, v___y_1224_);
if (lean_obj_tag(v___x_1231_) == 0)
{
lean_object* v___x_1233_; uint8_t v_isShared_1234_; uint8_t v_isSharedCheck_1238_; 
v_isSharedCheck_1238_ = !lean_is_exclusive(v___x_1231_);
if (v_isSharedCheck_1238_ == 0)
{
lean_object* v_unused_1239_; 
v_unused_1239_ = lean_ctor_get(v___x_1231_, 0);
lean_dec(v_unused_1239_);
v___x_1233_ = v___x_1231_;
v_isShared_1234_ = v_isSharedCheck_1238_;
goto v_resetjp_1232_;
}
else
{
lean_dec(v___x_1231_);
v___x_1233_ = lean_box(0);
v_isShared_1234_ = v_isSharedCheck_1238_;
goto v_resetjp_1232_;
}
v_resetjp_1232_:
{
lean_object* v___x_1236_; 
if (v_isShared_1234_ == 0)
{
lean_ctor_set(v___x_1233_, 0, v_fst_1229_);
v___x_1236_ = v___x_1233_;
goto v_reusejp_1235_;
}
else
{
lean_object* v_reuseFailAlloc_1237_; 
v_reuseFailAlloc_1237_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1237_, 0, v_fst_1229_);
v___x_1236_ = v_reuseFailAlloc_1237_;
goto v_reusejp_1235_;
}
v_reusejp_1235_:
{
return v___x_1236_;
}
}
}
else
{
lean_object* v_a_1240_; lean_object* v___x_1242_; uint8_t v_isShared_1243_; uint8_t v_isSharedCheck_1247_; 
lean_dec(v_fst_1229_);
v_a_1240_ = lean_ctor_get(v___x_1231_, 0);
v_isSharedCheck_1247_ = !lean_is_exclusive(v___x_1231_);
if (v_isSharedCheck_1247_ == 0)
{
v___x_1242_ = v___x_1231_;
v_isShared_1243_ = v_isSharedCheck_1247_;
goto v_resetjp_1241_;
}
else
{
lean_inc(v_a_1240_);
lean_dec(v___x_1231_);
v___x_1242_ = lean_box(0);
v_isShared_1243_ = v_isSharedCheck_1247_;
goto v_resetjp_1241_;
}
v_resetjp_1241_:
{
lean_object* v___x_1245_; 
if (v_isShared_1243_ == 0)
{
v___x_1245_ = v___x_1242_;
goto v_reusejp_1244_;
}
else
{
lean_object* v_reuseFailAlloc_1246_; 
v_reuseFailAlloc_1246_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1246_, 0, v_a_1240_);
v___x_1245_ = v_reuseFailAlloc_1246_;
goto v_reusejp_1244_;
}
v_reusejp_1244_:
{
return v___x_1245_;
}
}
}
}
else
{
lean_object* v_a_1248_; lean_object* v___x_1250_; uint8_t v_isShared_1251_; uint8_t v_isSharedCheck_1255_; 
v_a_1248_ = lean_ctor_get(v___y_1227_, 0);
v_isSharedCheck_1255_ = !lean_is_exclusive(v___y_1227_);
if (v_isSharedCheck_1255_ == 0)
{
v___x_1250_ = v___y_1227_;
v_isShared_1251_ = v_isSharedCheck_1255_;
goto v_resetjp_1249_;
}
else
{
lean_inc(v_a_1248_);
lean_dec(v___y_1227_);
v___x_1250_ = lean_box(0);
v_isShared_1251_ = v_isSharedCheck_1255_;
goto v_resetjp_1249_;
}
v_resetjp_1249_:
{
lean_object* v___x_1253_; 
if (v_isShared_1251_ == 0)
{
v___x_1253_ = v___x_1250_;
goto v_reusejp_1252_;
}
else
{
lean_object* v_reuseFailAlloc_1254_; 
v_reuseFailAlloc_1254_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1254_, 0, v_a_1248_);
v___x_1253_ = v_reuseFailAlloc_1254_;
goto v_reusejp_1252_;
}
v_reusejp_1252_:
{
return v___x_1253_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelArgs___lam__0___boxed(lean_object* v_l_1271_, lean_object* v_n_x3f_1272_, lean_object* v_e_1273_, lean_object* v_unfold_1274_, lean_object* v___y_1275_, lean_object* v___y_1276_, lean_object* v___y_1277_, lean_object* v___y_1278_, lean_object* v___y_1279_, lean_object* v___y_1280_, lean_object* v___y_1281_, lean_object* v___y_1282_, lean_object* v___y_1283_){
_start:
{
uint8_t v_unfold_boxed_1284_; lean_object* v_res_1285_; 
v_unfold_boxed_1284_ = lean_unbox(v_unfold_1274_);
v_res_1285_ = lp_mathlib_Mathlib_Tactic_Peel_peelArgs___lam__0(v_l_1271_, v_n_x3f_1272_, v_e_1273_, v_unfold_boxed_1284_, v___y_1275_, v___y_1276_, v___y_1277_, v___y_1278_, v___y_1279_, v___y_1280_, v___y_1281_, v___y_1282_);
lean_dec(v___y_1282_);
lean_dec_ref(v___y_1281_);
lean_dec(v___y_1280_);
lean_dec_ref(v___y_1279_);
lean_dec(v___y_1278_);
lean_dec_ref(v___y_1277_);
lean_dec(v___y_1276_);
lean_dec_ref(v___y_1275_);
lean_dec(v_l_1271_);
return v_res_1285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelArgs___lam__1(lean_object* v_a_1286_, lean_object* v___y_1287_, lean_object* v___y_1288_, lean_object* v___y_1289_, lean_object* v___y_1290_, lean_object* v___y_1291_, lean_object* v___y_1292_, lean_object* v___y_1293_, lean_object* v___y_1294_){
_start:
{
lean_object* v___x_1296_; 
v___x_1296_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_1288_, v___y_1291_, v___y_1292_, v___y_1293_, v___y_1294_);
if (lean_obj_tag(v___x_1296_) == 0)
{
lean_object* v_a_1297_; lean_object* v___x_1298_; 
v_a_1297_ = lean_ctor_get(v___x_1296_, 0);
lean_inc(v_a_1297_);
lean_dec_ref_known(v___x_1296_, 1);
v___x_1298_ = l_Lean_MVarId_clear(v_a_1297_, v_a_1286_, v___y_1291_, v___y_1292_, v___y_1293_, v___y_1294_);
return v___x_1298_;
}
else
{
lean_dec(v_a_1286_);
return v___x_1296_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelArgs___lam__1___boxed(lean_object* v_a_1299_, lean_object* v___y_1300_, lean_object* v___y_1301_, lean_object* v___y_1302_, lean_object* v___y_1303_, lean_object* v___y_1304_, lean_object* v___y_1305_, lean_object* v___y_1306_, lean_object* v___y_1307_, lean_object* v___y_1308_){
_start:
{
lean_object* v_res_1309_; 
v_res_1309_ = lp_mathlib_Mathlib_Tactic_Peel_peelArgs___lam__1(v_a_1299_, v___y_1300_, v___y_1301_, v___y_1302_, v___y_1303_, v___y_1304_, v___y_1305_, v___y_1306_, v___y_1307_);
lean_dec(v___y_1307_);
lean_dec_ref(v___y_1306_);
lean_dec(v___y_1305_);
lean_dec_ref(v___y_1304_);
lean_dec(v___y_1303_);
lean_dec_ref(v___y_1302_);
lean_dec(v___y_1301_);
lean_dec_ref(v___y_1300_);
return v_res_1309_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelArgs(lean_object* v_e_1310_, lean_object* v_num_1311_, lean_object* v_l_1312_, lean_object* v_n_x3f_1313_, uint8_t v_unfold_1314_, lean_object* v_a_1315_, lean_object* v_a_1316_, lean_object* v_a_1317_, lean_object* v_a_1318_, lean_object* v_a_1319_, lean_object* v_a_1320_, lean_object* v_a_1321_, lean_object* v_a_1322_){
_start:
{
lean_object* v_zero_1324_; uint8_t v_isZero_1325_; 
v_zero_1324_ = lean_unsigned_to_nat(0u);
v_isZero_1325_ = lean_nat_dec_eq(v_num_1311_, v_zero_1324_);
if (v_isZero_1325_ == 1)
{
lean_object* v___x_1326_; lean_object* v___x_1327_; 
lean_dec(v_n_x3f_1313_);
lean_dec(v_l_1312_);
lean_dec_ref(v_e_1310_);
v___x_1326_ = lean_box(0);
v___x_1327_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1327_, 0, v___x_1326_);
return v___x_1327_;
}
else
{
lean_object* v___x_1328_; lean_object* v___f_1329_; lean_object* v___x_1330_; 
v___x_1328_ = lean_box(v_unfold_1314_);
lean_inc(v_n_x3f_1313_);
lean_inc(v_l_1312_);
v___f_1329_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Peel_peelArgs___lam__0___boxed), 13, 4);
lean_closure_set(v___f_1329_, 0, v_l_1312_);
lean_closure_set(v___f_1329_, 1, v_n_x3f_1313_);
lean_closure_set(v___f_1329_, 2, v_e_1310_);
lean_closure_set(v___f_1329_, 3, v___x_1328_);
v___x_1330_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_1329_, v_a_1315_, v_a_1316_, v_a_1317_, v_a_1318_, v_a_1319_, v_a_1320_, v_a_1321_, v_a_1322_);
if (lean_obj_tag(v___x_1330_) == 0)
{
lean_object* v_a_1331_; lean_object* v_one_1332_; lean_object* v_n_1333_; lean_object* v___f_1334_; lean_object* v___x_1335_; lean_object* v___y_1337_; 
v_a_1331_ = lean_ctor_get(v___x_1330_, 0);
lean_inc_n(v_a_1331_, 2);
lean_dec_ref_known(v___x_1330_, 1);
v_one_1332_ = lean_unsigned_to_nat(1u);
v_n_1333_ = lean_nat_sub(v_num_1311_, v_one_1332_);
v___f_1334_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Peel_peelArgs___lam__1___boxed), 10, 1);
lean_closure_set(v___f_1334_, 0, v_a_1331_);
v___x_1335_ = l_Lean_Expr_fvar___override(v_a_1331_);
if (lean_obj_tag(v_l_1312_) == 0)
{
v___y_1337_ = v_l_1312_;
goto v___jp_1336_;
}
else
{
lean_object* v_tail_1372_; 
v_tail_1372_ = lean_ctor_get(v_l_1312_, 1);
lean_inc(v_tail_1372_);
lean_dec_ref_known(v_l_1312_, 2);
v___y_1337_ = v_tail_1372_;
goto v___jp_1336_;
}
v___jp_1336_:
{
uint8_t v___x_1338_; lean_object* v___x_1339_; 
v___x_1338_ = 1;
v___x_1339_ = lp_mathlib_Mathlib_Tactic_Peel_peelArgs(v___x_1335_, v_n_1333_, v___y_1337_, v_n_x3f_1313_, v___x_1338_, v_a_1315_, v_a_1316_, v_a_1317_, v_a_1318_, v_a_1319_, v_a_1320_, v_a_1321_, v_a_1322_);
if (lean_obj_tag(v___x_1339_) == 0)
{
lean_object* v___x_1341_; uint8_t v_isShared_1342_; uint8_t v_isSharedCheck_1370_; 
v_isSharedCheck_1370_ = !lean_is_exclusive(v___x_1339_);
if (v_isSharedCheck_1370_ == 0)
{
lean_object* v_unused_1371_; 
v_unused_1371_ = lean_ctor_get(v___x_1339_, 0);
lean_dec(v_unused_1371_);
v___x_1341_ = v___x_1339_;
v_isShared_1342_ = v_isSharedCheck_1370_;
goto v_resetjp_1340_;
}
else
{
lean_dec(v___x_1339_);
v___x_1341_ = lean_box(0);
v_isShared_1342_ = v_isSharedCheck_1370_;
goto v_resetjp_1340_;
}
v_resetjp_1340_:
{
uint8_t v___x_1343_; 
v___x_1343_ = lean_nat_dec_eq(v_n_1333_, v_zero_1324_);
lean_dec(v_n_1333_);
if (v___x_1343_ == 0)
{
lean_object* v___x_1344_; 
lean_del_object(v___x_1341_);
v___x_1344_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Peel_peelArgs_spec__0___redArg(v___f_1334_, v_a_1315_, v_a_1316_, v_a_1317_, v_a_1318_, v_a_1319_, v_a_1320_, v_a_1321_, v_a_1322_);
if (lean_obj_tag(v___x_1344_) == 0)
{
lean_object* v_a_1345_; lean_object* v___x_1347_; uint8_t v_isShared_1348_; uint8_t v_isSharedCheck_1357_; 
v_a_1345_ = lean_ctor_get(v___x_1344_, 0);
v_isSharedCheck_1357_ = !lean_is_exclusive(v___x_1344_);
if (v_isSharedCheck_1357_ == 0)
{
v___x_1347_ = v___x_1344_;
v_isShared_1348_ = v_isSharedCheck_1357_;
goto v_resetjp_1346_;
}
else
{
lean_inc(v_a_1345_);
lean_dec(v___x_1344_);
v___x_1347_ = lean_box(0);
v_isShared_1348_ = v_isSharedCheck_1357_;
goto v_resetjp_1346_;
}
v_resetjp_1346_:
{
if (lean_obj_tag(v_a_1345_) == 1)
{
lean_object* v_val_1349_; lean_object* v___x_1350_; lean_object* v___x_1351_; lean_object* v___x_1352_; 
lean_del_object(v___x_1347_);
v_val_1349_ = lean_ctor_get(v_a_1345_, 0);
lean_inc(v_val_1349_);
lean_dec_ref_known(v_a_1345_, 1);
v___x_1350_ = lean_box(0);
v___x_1351_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1351_, 0, v_val_1349_);
lean_ctor_set(v___x_1351_, 1, v___x_1350_);
v___x_1352_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_1351_, v_a_1316_, v_a_1319_, v_a_1320_, v_a_1321_, v_a_1322_);
return v___x_1352_;
}
else
{
lean_object* v___x_1353_; lean_object* v___x_1355_; 
lean_dec(v_a_1345_);
v___x_1353_ = lean_box(0);
if (v_isShared_1348_ == 0)
{
lean_ctor_set(v___x_1347_, 0, v___x_1353_);
v___x_1355_ = v___x_1347_;
goto v_reusejp_1354_;
}
else
{
lean_object* v_reuseFailAlloc_1356_; 
v_reuseFailAlloc_1356_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1356_, 0, v___x_1353_);
v___x_1355_ = v_reuseFailAlloc_1356_;
goto v_reusejp_1354_;
}
v_reusejp_1354_:
{
return v___x_1355_;
}
}
}
}
else
{
lean_object* v_a_1358_; lean_object* v___x_1360_; uint8_t v_isShared_1361_; uint8_t v_isSharedCheck_1365_; 
v_a_1358_ = lean_ctor_get(v___x_1344_, 0);
v_isSharedCheck_1365_ = !lean_is_exclusive(v___x_1344_);
if (v_isSharedCheck_1365_ == 0)
{
v___x_1360_ = v___x_1344_;
v_isShared_1361_ = v_isSharedCheck_1365_;
goto v_resetjp_1359_;
}
else
{
lean_inc(v_a_1358_);
lean_dec(v___x_1344_);
v___x_1360_ = lean_box(0);
v_isShared_1361_ = v_isSharedCheck_1365_;
goto v_resetjp_1359_;
}
v_resetjp_1359_:
{
lean_object* v___x_1363_; 
if (v_isShared_1361_ == 0)
{
v___x_1363_ = v___x_1360_;
goto v_reusejp_1362_;
}
else
{
lean_object* v_reuseFailAlloc_1364_; 
v_reuseFailAlloc_1364_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1364_, 0, v_a_1358_);
v___x_1363_ = v_reuseFailAlloc_1364_;
goto v_reusejp_1362_;
}
v_reusejp_1362_:
{
return v___x_1363_;
}
}
}
}
else
{
lean_object* v___x_1366_; lean_object* v___x_1368_; 
lean_dec_ref(v___f_1334_);
v___x_1366_ = lean_box(0);
if (v_isShared_1342_ == 0)
{
lean_ctor_set(v___x_1341_, 0, v___x_1366_);
v___x_1368_ = v___x_1341_;
goto v_reusejp_1367_;
}
else
{
lean_object* v_reuseFailAlloc_1369_; 
v_reuseFailAlloc_1369_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1369_, 0, v___x_1366_);
v___x_1368_ = v_reuseFailAlloc_1369_;
goto v_reusejp_1367_;
}
v_reusejp_1367_:
{
return v___x_1368_;
}
}
}
}
else
{
lean_dec_ref(v___f_1334_);
lean_dec(v_n_1333_);
return v___x_1339_;
}
}
}
else
{
lean_object* v_a_1373_; lean_object* v___x_1375_; uint8_t v_isShared_1376_; uint8_t v_isSharedCheck_1380_; 
lean_dec(v_n_x3f_1313_);
lean_dec(v_l_1312_);
v_a_1373_ = lean_ctor_get(v___x_1330_, 0);
v_isSharedCheck_1380_ = !lean_is_exclusive(v___x_1330_);
if (v_isSharedCheck_1380_ == 0)
{
v___x_1375_ = v___x_1330_;
v_isShared_1376_ = v_isSharedCheck_1380_;
goto v_resetjp_1374_;
}
else
{
lean_inc(v_a_1373_);
lean_dec(v___x_1330_);
v___x_1375_ = lean_box(0);
v_isShared_1376_ = v_isSharedCheck_1380_;
goto v_resetjp_1374_;
}
v_resetjp_1374_:
{
lean_object* v___x_1378_; 
if (v_isShared_1376_ == 0)
{
v___x_1378_ = v___x_1375_;
goto v_reusejp_1377_;
}
else
{
lean_object* v_reuseFailAlloc_1379_; 
v_reuseFailAlloc_1379_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1379_, 0, v_a_1373_);
v___x_1378_ = v_reuseFailAlloc_1379_;
goto v_reusejp_1377_;
}
v_reusejp_1377_:
{
return v___x_1378_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelArgs___boxed(lean_object* v_e_1381_, lean_object* v_num_1382_, lean_object* v_l_1383_, lean_object* v_n_x3f_1384_, lean_object* v_unfold_1385_, lean_object* v_a_1386_, lean_object* v_a_1387_, lean_object* v_a_1388_, lean_object* v_a_1389_, lean_object* v_a_1390_, lean_object* v_a_1391_, lean_object* v_a_1392_, lean_object* v_a_1393_, lean_object* v_a_1394_){
_start:
{
uint8_t v_unfold_boxed_1395_; lean_object* v_res_1396_; 
v_unfold_boxed_1395_ = lean_unbox(v_unfold_1385_);
v_res_1396_ = lp_mathlib_Mathlib_Tactic_Peel_peelArgs(v_e_1381_, v_num_1382_, v_l_1383_, v_n_x3f_1384_, v_unfold_boxed_1395_, v_a_1386_, v_a_1387_, v_a_1388_, v_a_1389_, v_a_1390_, v_a_1391_, v_a_1392_, v_a_1393_);
lean_dec(v_a_1393_);
lean_dec_ref(v_a_1392_);
lean_dec(v_a_1391_);
lean_dec_ref(v_a_1390_);
lean_dec(v_a_1389_);
lean_dec_ref(v_a_1388_);
lean_dec(v_a_1387_);
lean_dec_ref(v_a_1386_);
lean_dec(v_num_1382_);
return v_res_1396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelUnbounded___lam__0(lean_object* v_n_x3f_1397_, lean_object* v_e_1398_, uint8_t v_unfold_1399_, lean_object* v___y_1400_, lean_object* v___y_1401_, lean_object* v___y_1402_, lean_object* v___y_1403_, lean_object* v___y_1404_, lean_object* v___y_1405_, lean_object* v___y_1406_, lean_object* v___y_1407_){
_start:
{
lean_object* v___y_1410_; lean_object* v___x_1439_; 
v___x_1439_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_1401_, v___y_1404_, v___y_1405_, v___y_1406_, v___y_1407_);
if (lean_obj_tag(v___x_1439_) == 0)
{
lean_object* v_a_1440_; lean_object* v___x_1441_; 
v_a_1440_ = lean_ctor_get(v___x_1439_, 0);
lean_inc(v_a_1440_);
lean_dec_ref_known(v___x_1439_, 1);
v___x_1441_ = lean_box(0);
if (lean_obj_tag(v_n_x3f_1397_) == 0)
{
lean_object* v___x_1442_; lean_object* v___x_1443_; 
v___x_1442_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelArgs___lam__0___closed__1));
v___x_1443_ = lp_mathlib_Mathlib_Tactic_Peel_peelCore(v_a_1440_, v_e_1398_, v___x_1441_, v___x_1442_, v_unfold_1399_, v___y_1404_, v___y_1405_, v___y_1406_, v___y_1407_);
v___y_1410_ = v___x_1443_;
goto v___jp_1409_;
}
else
{
lean_object* v_val_1444_; lean_object* v___x_1445_; 
v_val_1444_ = lean_ctor_get(v_n_x3f_1397_, 0);
lean_inc(v_val_1444_);
lean_dec_ref_known(v_n_x3f_1397_, 1);
v___x_1445_ = lp_mathlib_Mathlib_Tactic_Peel_peelCore(v_a_1440_, v_e_1398_, v___x_1441_, v_val_1444_, v_unfold_1399_, v___y_1404_, v___y_1405_, v___y_1406_, v___y_1407_);
v___y_1410_ = v___x_1445_;
goto v___jp_1409_;
}
}
else
{
lean_object* v_a_1446_; lean_object* v___x_1448_; uint8_t v_isShared_1449_; uint8_t v_isSharedCheck_1453_; 
lean_dec_ref(v_e_1398_);
lean_dec(v_n_x3f_1397_);
v_a_1446_ = lean_ctor_get(v___x_1439_, 0);
v_isSharedCheck_1453_ = !lean_is_exclusive(v___x_1439_);
if (v_isSharedCheck_1453_ == 0)
{
v___x_1448_ = v___x_1439_;
v_isShared_1449_ = v_isSharedCheck_1453_;
goto v_resetjp_1447_;
}
else
{
lean_inc(v_a_1446_);
lean_dec(v___x_1439_);
v___x_1448_ = lean_box(0);
v_isShared_1449_ = v_isSharedCheck_1453_;
goto v_resetjp_1447_;
}
v_resetjp_1447_:
{
lean_object* v___x_1451_; 
if (v_isShared_1449_ == 0)
{
v___x_1451_ = v___x_1448_;
goto v_reusejp_1450_;
}
else
{
lean_object* v_reuseFailAlloc_1452_; 
v_reuseFailAlloc_1452_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1452_, 0, v_a_1446_);
v___x_1451_ = v_reuseFailAlloc_1452_;
goto v_reusejp_1450_;
}
v_reusejp_1450_:
{
return v___x_1451_;
}
}
}
v___jp_1409_:
{
if (lean_obj_tag(v___y_1410_) == 0)
{
lean_object* v_a_1411_; lean_object* v_fst_1412_; lean_object* v_snd_1413_; lean_object* v___x_1414_; 
v_a_1411_ = lean_ctor_get(v___y_1410_, 0);
lean_inc(v_a_1411_);
lean_dec_ref_known(v___y_1410_, 1);
v_fst_1412_ = lean_ctor_get(v_a_1411_, 0);
lean_inc(v_fst_1412_);
v_snd_1413_ = lean_ctor_get(v_a_1411_, 1);
lean_inc(v_snd_1413_);
lean_dec(v_a_1411_);
v___x_1414_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v_snd_1413_, v___y_1401_, v___y_1404_, v___y_1405_, v___y_1406_, v___y_1407_);
if (lean_obj_tag(v___x_1414_) == 0)
{
lean_object* v___x_1416_; uint8_t v_isShared_1417_; uint8_t v_isSharedCheck_1421_; 
v_isSharedCheck_1421_ = !lean_is_exclusive(v___x_1414_);
if (v_isSharedCheck_1421_ == 0)
{
lean_object* v_unused_1422_; 
v_unused_1422_ = lean_ctor_get(v___x_1414_, 0);
lean_dec(v_unused_1422_);
v___x_1416_ = v___x_1414_;
v_isShared_1417_ = v_isSharedCheck_1421_;
goto v_resetjp_1415_;
}
else
{
lean_dec(v___x_1414_);
v___x_1416_ = lean_box(0);
v_isShared_1417_ = v_isSharedCheck_1421_;
goto v_resetjp_1415_;
}
v_resetjp_1415_:
{
lean_object* v___x_1419_; 
if (v_isShared_1417_ == 0)
{
lean_ctor_set(v___x_1416_, 0, v_fst_1412_);
v___x_1419_ = v___x_1416_;
goto v_reusejp_1418_;
}
else
{
lean_object* v_reuseFailAlloc_1420_; 
v_reuseFailAlloc_1420_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1420_, 0, v_fst_1412_);
v___x_1419_ = v_reuseFailAlloc_1420_;
goto v_reusejp_1418_;
}
v_reusejp_1418_:
{
return v___x_1419_;
}
}
}
else
{
lean_object* v_a_1423_; lean_object* v___x_1425_; uint8_t v_isShared_1426_; uint8_t v_isSharedCheck_1430_; 
lean_dec(v_fst_1412_);
v_a_1423_ = lean_ctor_get(v___x_1414_, 0);
v_isSharedCheck_1430_ = !lean_is_exclusive(v___x_1414_);
if (v_isSharedCheck_1430_ == 0)
{
v___x_1425_ = v___x_1414_;
v_isShared_1426_ = v_isSharedCheck_1430_;
goto v_resetjp_1424_;
}
else
{
lean_inc(v_a_1423_);
lean_dec(v___x_1414_);
v___x_1425_ = lean_box(0);
v_isShared_1426_ = v_isSharedCheck_1430_;
goto v_resetjp_1424_;
}
v_resetjp_1424_:
{
lean_object* v___x_1428_; 
if (v_isShared_1426_ == 0)
{
v___x_1428_ = v___x_1425_;
goto v_reusejp_1427_;
}
else
{
lean_object* v_reuseFailAlloc_1429_; 
v_reuseFailAlloc_1429_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1429_, 0, v_a_1423_);
v___x_1428_ = v_reuseFailAlloc_1429_;
goto v_reusejp_1427_;
}
v_reusejp_1427_:
{
return v___x_1428_;
}
}
}
}
else
{
lean_object* v_a_1431_; lean_object* v___x_1433_; uint8_t v_isShared_1434_; uint8_t v_isSharedCheck_1438_; 
v_a_1431_ = lean_ctor_get(v___y_1410_, 0);
v_isSharedCheck_1438_ = !lean_is_exclusive(v___y_1410_);
if (v_isSharedCheck_1438_ == 0)
{
v___x_1433_ = v___y_1410_;
v_isShared_1434_ = v_isSharedCheck_1438_;
goto v_resetjp_1432_;
}
else
{
lean_inc(v_a_1431_);
lean_dec(v___y_1410_);
v___x_1433_ = lean_box(0);
v_isShared_1434_ = v_isSharedCheck_1438_;
goto v_resetjp_1432_;
}
v_resetjp_1432_:
{
lean_object* v___x_1436_; 
if (v_isShared_1434_ == 0)
{
v___x_1436_ = v___x_1433_;
goto v_reusejp_1435_;
}
else
{
lean_object* v_reuseFailAlloc_1437_; 
v_reuseFailAlloc_1437_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1437_, 0, v_a_1431_);
v___x_1436_ = v_reuseFailAlloc_1437_;
goto v_reusejp_1435_;
}
v_reusejp_1435_:
{
return v___x_1436_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelUnbounded___lam__0___boxed(lean_object* v_n_x3f_1454_, lean_object* v_e_1455_, lean_object* v_unfold_1456_, lean_object* v___y_1457_, lean_object* v___y_1458_, lean_object* v___y_1459_, lean_object* v___y_1460_, lean_object* v___y_1461_, lean_object* v___y_1462_, lean_object* v___y_1463_, lean_object* v___y_1464_, lean_object* v___y_1465_){
_start:
{
uint8_t v_unfold_boxed_1466_; lean_object* v_res_1467_; 
v_unfold_boxed_1466_ = lean_unbox(v_unfold_1456_);
v_res_1467_ = lp_mathlib_Mathlib_Tactic_Peel_peelUnbounded___lam__0(v_n_x3f_1454_, v_e_1455_, v_unfold_boxed_1466_, v___y_1457_, v___y_1458_, v___y_1459_, v___y_1460_, v___y_1461_, v___y_1462_, v___y_1463_, v___y_1464_);
lean_dec(v___y_1464_);
lean_dec_ref(v___y_1463_);
lean_dec(v___y_1462_);
lean_dec_ref(v___y_1461_);
lean_dec(v___y_1460_);
lean_dec_ref(v___y_1459_);
lean_dec(v___y_1458_);
lean_dec_ref(v___y_1457_);
return v_res_1467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelUnbounded___lam__1(lean_object* v___f_1468_, lean_object* v___y_1469_, lean_object* v___y_1470_, lean_object* v___y_1471_, lean_object* v___y_1472_, lean_object* v___y_1473_, lean_object* v___y_1474_, lean_object* v___y_1475_, lean_object* v___y_1476_){
_start:
{
lean_object* v___x_1478_; 
v___x_1478_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_1468_, v___y_1469_, v___y_1470_, v___y_1471_, v___y_1472_, v___y_1473_, v___y_1474_, v___y_1475_, v___y_1476_);
return v___x_1478_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelUnbounded___lam__1___boxed(lean_object* v___f_1479_, lean_object* v___y_1480_, lean_object* v___y_1481_, lean_object* v___y_1482_, lean_object* v___y_1483_, lean_object* v___y_1484_, lean_object* v___y_1485_, lean_object* v___y_1486_, lean_object* v___y_1487_, lean_object* v___y_1488_){
_start:
{
lean_object* v_res_1489_; 
v_res_1489_ = lp_mathlib_Mathlib_Tactic_Peel_peelUnbounded___lam__1(v___f_1479_, v___y_1480_, v___y_1481_, v___y_1482_, v___y_1483_, v___y_1484_, v___y_1485_, v___y_1486_, v___y_1487_);
lean_dec(v___y_1487_);
lean_dec_ref(v___y_1486_);
lean_dec(v___y_1485_);
lean_dec_ref(v___y_1484_);
lean_dec(v___y_1483_);
lean_dec_ref(v___y_1482_);
lean_dec(v___y_1481_);
lean_dec_ref(v___y_1480_);
return v_res_1489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelUnbounded___lam__2(lean_object* v_val_1490_, lean_object* v___y_1491_, lean_object* v___y_1492_, lean_object* v___y_1493_, lean_object* v___y_1494_, lean_object* v___y_1495_, lean_object* v___y_1496_, lean_object* v___y_1497_, lean_object* v___y_1498_){
_start:
{
lean_object* v___x_1500_; 
v___x_1500_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_1492_, v___y_1495_, v___y_1496_, v___y_1497_, v___y_1498_);
if (lean_obj_tag(v___x_1500_) == 0)
{
lean_object* v_a_1501_; lean_object* v___x_1502_; 
v_a_1501_ = lean_ctor_get(v___x_1500_, 0);
lean_inc(v_a_1501_);
lean_dec_ref_known(v___x_1500_, 1);
v___x_1502_ = l_Lean_MVarId_clear(v_a_1501_, v_val_1490_, v___y_1495_, v___y_1496_, v___y_1497_, v___y_1498_);
return v___x_1502_;
}
else
{
lean_dec(v_val_1490_);
return v___x_1500_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelUnbounded___lam__2___boxed(lean_object* v_val_1503_, lean_object* v___y_1504_, lean_object* v___y_1505_, lean_object* v___y_1506_, lean_object* v___y_1507_, lean_object* v___y_1508_, lean_object* v___y_1509_, lean_object* v___y_1510_, lean_object* v___y_1511_, lean_object* v___y_1512_){
_start:
{
lean_object* v_res_1513_; 
v_res_1513_ = lp_mathlib_Mathlib_Tactic_Peel_peelUnbounded___lam__2(v_val_1503_, v___y_1504_, v___y_1505_, v___y_1506_, v___y_1507_, v___y_1508_, v___y_1509_, v___y_1510_, v___y_1511_);
lean_dec(v___y_1511_);
lean_dec_ref(v___y_1510_);
lean_dec(v___y_1509_);
lean_dec_ref(v___y_1508_);
lean_dec(v___y_1507_);
lean_dec_ref(v___y_1506_);
lean_dec(v___y_1505_);
lean_dec_ref(v___y_1504_);
return v_res_1513_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelUnbounded(lean_object* v_e_1514_, lean_object* v_n_x3f_1515_, uint8_t v_unfold_1516_, lean_object* v_a_1517_, lean_object* v_a_1518_, lean_object* v_a_1519_, lean_object* v_a_1520_, lean_object* v_a_1521_, lean_object* v_a_1522_, lean_object* v_a_1523_, lean_object* v_a_1524_){
_start:
{
lean_object* v___x_1530_; lean_object* v___f_1531_; lean_object* v___f_1532_; lean_object* v___x_1533_; 
v___x_1530_ = lean_box(v_unfold_1516_);
lean_inc(v_n_x3f_1515_);
v___f_1531_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Peel_peelUnbounded___lam__0___boxed), 12, 3);
lean_closure_set(v___f_1531_, 0, v_n_x3f_1515_);
lean_closure_set(v___f_1531_, 1, v_e_1514_);
lean_closure_set(v___f_1531_, 2, v___x_1530_);
v___f_1532_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Peel_peelUnbounded___lam__1___boxed), 10, 1);
lean_closure_set(v___f_1532_, 0, v___f_1531_);
v___x_1533_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Peel_peelArgs_spec__0___redArg(v___f_1532_, v_a_1517_, v_a_1518_, v_a_1519_, v_a_1520_, v_a_1521_, v_a_1522_, v_a_1523_, v_a_1524_);
if (lean_obj_tag(v___x_1533_) == 0)
{
lean_object* v_a_1534_; lean_object* v___x_1536_; uint8_t v_isShared_1537_; uint8_t v_isSharedCheck_1572_; 
v_a_1534_ = lean_ctor_get(v___x_1533_, 0);
v_isSharedCheck_1572_ = !lean_is_exclusive(v___x_1533_);
if (v_isSharedCheck_1572_ == 0)
{
v___x_1536_ = v___x_1533_;
v_isShared_1537_ = v_isSharedCheck_1572_;
goto v_resetjp_1535_;
}
else
{
lean_inc(v_a_1534_);
lean_dec(v___x_1533_);
v___x_1536_ = lean_box(0);
v_isShared_1537_ = v_isSharedCheck_1572_;
goto v_resetjp_1535_;
}
v_resetjp_1535_:
{
if (lean_obj_tag(v_a_1534_) == 1)
{
lean_object* v_val_1538_; lean_object* v___x_1539_; uint8_t v___x_1540_; lean_object* v___x_1541_; 
lean_del_object(v___x_1536_);
v_val_1538_ = lean_ctor_get(v_a_1534_, 0);
lean_inc_n(v_val_1538_, 2);
lean_dec_ref_known(v_a_1534_, 1);
v___x_1539_ = l_Lean_Expr_fvar___override(v_val_1538_);
v___x_1540_ = 0;
v___x_1541_ = lp_mathlib_Mathlib_Tactic_Peel_peelUnbounded(v___x_1539_, v_n_x3f_1515_, v___x_1540_, v_a_1517_, v_a_1518_, v_a_1519_, v_a_1520_, v_a_1521_, v_a_1522_, v_a_1523_, v_a_1524_);
if (lean_obj_tag(v___x_1541_) == 0)
{
lean_object* v_a_1542_; uint8_t v___x_1543_; 
v_a_1542_ = lean_ctor_get(v___x_1541_, 0);
lean_inc(v_a_1542_);
lean_dec_ref_known(v___x_1541_, 1);
v___x_1543_ = lean_unbox(v_a_1542_);
lean_dec(v_a_1542_);
if (v___x_1543_ == 0)
{
lean_dec(v_val_1538_);
goto v___jp_1526_;
}
else
{
lean_object* v___f_1544_; lean_object* v___x_1545_; 
v___f_1544_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Peel_peelUnbounded___lam__2___boxed), 10, 1);
lean_closure_set(v___f_1544_, 0, v_val_1538_);
v___x_1545_ = lp_mathlib_Lean_observing_x3f___at___00Mathlib_Tactic_Peel_peelArgs_spec__0___redArg(v___f_1544_, v_a_1517_, v_a_1518_, v_a_1519_, v_a_1520_, v_a_1521_, v_a_1522_, v_a_1523_, v_a_1524_);
if (lean_obj_tag(v___x_1545_) == 0)
{
lean_object* v_a_1546_; 
v_a_1546_ = lean_ctor_get(v___x_1545_, 0);
lean_inc(v_a_1546_);
lean_dec_ref_known(v___x_1545_, 1);
if (lean_obj_tag(v_a_1546_) == 1)
{
lean_object* v_val_1547_; lean_object* v___x_1548_; lean_object* v___x_1549_; lean_object* v___x_1550_; 
v_val_1547_ = lean_ctor_get(v_a_1546_, 0);
lean_inc(v_val_1547_);
lean_dec_ref_known(v_a_1546_, 1);
v___x_1548_ = lean_box(0);
v___x_1549_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1549_, 0, v_val_1547_);
lean_ctor_set(v___x_1549_, 1, v___x_1548_);
v___x_1550_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_1549_, v_a_1518_, v_a_1521_, v_a_1522_, v_a_1523_, v_a_1524_);
if (lean_obj_tag(v___x_1550_) == 0)
{
lean_dec_ref_known(v___x_1550_, 1);
goto v___jp_1526_;
}
else
{
lean_object* v_a_1551_; lean_object* v___x_1553_; uint8_t v_isShared_1554_; uint8_t v_isSharedCheck_1558_; 
v_a_1551_ = lean_ctor_get(v___x_1550_, 0);
v_isSharedCheck_1558_ = !lean_is_exclusive(v___x_1550_);
if (v_isSharedCheck_1558_ == 0)
{
v___x_1553_ = v___x_1550_;
v_isShared_1554_ = v_isSharedCheck_1558_;
goto v_resetjp_1552_;
}
else
{
lean_inc(v_a_1551_);
lean_dec(v___x_1550_);
v___x_1553_ = lean_box(0);
v_isShared_1554_ = v_isSharedCheck_1558_;
goto v_resetjp_1552_;
}
v_resetjp_1552_:
{
lean_object* v___x_1556_; 
if (v_isShared_1554_ == 0)
{
v___x_1556_ = v___x_1553_;
goto v_reusejp_1555_;
}
else
{
lean_object* v_reuseFailAlloc_1557_; 
v_reuseFailAlloc_1557_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1557_, 0, v_a_1551_);
v___x_1556_ = v_reuseFailAlloc_1557_;
goto v_reusejp_1555_;
}
v_reusejp_1555_:
{
return v___x_1556_;
}
}
}
}
else
{
lean_dec(v_a_1546_);
goto v___jp_1526_;
}
}
else
{
lean_object* v_a_1559_; lean_object* v___x_1561_; uint8_t v_isShared_1562_; uint8_t v_isSharedCheck_1566_; 
v_a_1559_ = lean_ctor_get(v___x_1545_, 0);
v_isSharedCheck_1566_ = !lean_is_exclusive(v___x_1545_);
if (v_isSharedCheck_1566_ == 0)
{
v___x_1561_ = v___x_1545_;
v_isShared_1562_ = v_isSharedCheck_1566_;
goto v_resetjp_1560_;
}
else
{
lean_inc(v_a_1559_);
lean_dec(v___x_1545_);
v___x_1561_ = lean_box(0);
v_isShared_1562_ = v_isSharedCheck_1566_;
goto v_resetjp_1560_;
}
v_resetjp_1560_:
{
lean_object* v___x_1564_; 
if (v_isShared_1562_ == 0)
{
v___x_1564_ = v___x_1561_;
goto v_reusejp_1563_;
}
else
{
lean_object* v_reuseFailAlloc_1565_; 
v_reuseFailAlloc_1565_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1565_, 0, v_a_1559_);
v___x_1564_ = v_reuseFailAlloc_1565_;
goto v_reusejp_1563_;
}
v_reusejp_1563_:
{
return v___x_1564_;
}
}
}
}
}
else
{
lean_dec(v_val_1538_);
return v___x_1541_;
}
}
else
{
uint8_t v___x_1567_; lean_object* v___x_1568_; lean_object* v___x_1570_; 
lean_dec(v_a_1534_);
lean_dec(v_n_x3f_1515_);
v___x_1567_ = 0;
v___x_1568_ = lean_box(v___x_1567_);
if (v_isShared_1537_ == 0)
{
lean_ctor_set(v___x_1536_, 0, v___x_1568_);
v___x_1570_ = v___x_1536_;
goto v_reusejp_1569_;
}
else
{
lean_object* v_reuseFailAlloc_1571_; 
v_reuseFailAlloc_1571_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1571_, 0, v___x_1568_);
v___x_1570_ = v_reuseFailAlloc_1571_;
goto v_reusejp_1569_;
}
v_reusejp_1569_:
{
return v___x_1570_;
}
}
}
}
else
{
lean_object* v_a_1573_; lean_object* v___x_1575_; uint8_t v_isShared_1576_; uint8_t v_isSharedCheck_1580_; 
lean_dec(v_n_x3f_1515_);
v_a_1573_ = lean_ctor_get(v___x_1533_, 0);
v_isSharedCheck_1580_ = !lean_is_exclusive(v___x_1533_);
if (v_isSharedCheck_1580_ == 0)
{
v___x_1575_ = v___x_1533_;
v_isShared_1576_ = v_isSharedCheck_1580_;
goto v_resetjp_1574_;
}
else
{
lean_inc(v_a_1573_);
lean_dec(v___x_1533_);
v___x_1575_ = lean_box(0);
v_isShared_1576_ = v_isSharedCheck_1580_;
goto v_resetjp_1574_;
}
v_resetjp_1574_:
{
lean_object* v___x_1578_; 
if (v_isShared_1576_ == 0)
{
v___x_1578_ = v___x_1575_;
goto v_reusejp_1577_;
}
else
{
lean_object* v_reuseFailAlloc_1579_; 
v_reuseFailAlloc_1579_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1579_, 0, v_a_1573_);
v___x_1578_ = v_reuseFailAlloc_1579_;
goto v_reusejp_1577_;
}
v_reusejp_1577_:
{
return v___x_1578_;
}
}
}
v___jp_1526_:
{
uint8_t v___x_1527_; lean_object* v___x_1528_; lean_object* v___x_1529_; 
v___x_1527_ = 1;
v___x_1528_ = lean_box(v___x_1527_);
v___x_1529_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1529_, 0, v___x_1528_);
return v___x_1529_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelUnbounded___boxed(lean_object* v_e_1581_, lean_object* v_n_x3f_1582_, lean_object* v_unfold_1583_, lean_object* v_a_1584_, lean_object* v_a_1585_, lean_object* v_a_1586_, lean_object* v_a_1587_, lean_object* v_a_1588_, lean_object* v_a_1589_, lean_object* v_a_1590_, lean_object* v_a_1591_, lean_object* v_a_1592_){
_start:
{
uint8_t v_unfold_boxed_1593_; lean_object* v_res_1594_; 
v_unfold_boxed_1593_ = lean_unbox(v_unfold_1583_);
v_res_1594_ = lp_mathlib_Mathlib_Tactic_Peel_peelUnbounded(v_e_1581_, v_n_x3f_1582_, v_unfold_boxed_1593_, v_a_1584_, v_a_1585_, v_a_1586_, v_a_1587_, v_a_1588_, v_a_1589_, v_a_1590_, v_a_1591_);
lean_dec(v_a_1591_);
lean_dec_ref(v_a_1590_);
lean_dec(v_a_1589_);
lean_dec_ref(v_a_1588_);
lean_dec(v_a_1587_);
lean_dec_ref(v_a_1586_);
lean_dec(v_a_1585_);
lean_dec_ref(v_a_1584_);
return v_res_1594_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__18(void){
_start:
{
lean_object* v___x_1635_; lean_object* v___x_1636_; 
v___x_1635_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__17));
v___x_1636_ = l_String_toRawSubstring_x27(v___x_1635_);
return v___x_1636_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__23(void){
_start:
{
lean_object* v___x_1646_; lean_object* v___x_1647_; 
v___x_1646_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__22));
v___x_1647_ = l_String_toRawSubstring_x27(v___x_1646_);
return v___x_1647_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__28(void){
_start:
{
lean_object* v___x_1657_; lean_object* v___x_1658_; 
v___x_1657_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__27));
v___x_1658_ = l_String_toRawSubstring_x27(v___x_1657_);
return v___x_1658_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__34(void){
_start:
{
lean_object* v___x_1673_; lean_object* v___x_1674_; 
v___x_1673_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__33));
v___x_1674_ = l_String_toRawSubstring_x27(v___x_1673_);
return v___x_1674_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__40(void){
_start:
{
lean_object* v___x_1689_; lean_object* v___x_1690_; 
v___x_1689_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__39));
v___x_1690_ = l_String_toRawSubstring_x27(v___x_1689_);
return v___x_1690_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux(lean_object* v_a_1709_, lean_object* v_a_1710_, lean_object* v_a_1711_, lean_object* v_a_1712_, lean_object* v_a_1713_, lean_object* v_a_1714_, lean_object* v_a_1715_, lean_object* v_a_1716_){
_start:
{
lean_object* v_ref_1718_; lean_object* v_quotContext_1719_; lean_object* v_currMacroScope_1720_; uint8_t v___x_1721_; lean_object* v___x_1722_; lean_object* v___x_1723_; lean_object* v___x_1724_; lean_object* v___x_1725_; lean_object* v___x_1726_; lean_object* v___x_1727_; lean_object* v___x_1728_; lean_object* v___x_1729_; lean_object* v___x_1730_; lean_object* v___x_1731_; lean_object* v___x_1732_; lean_object* v___x_1733_; lean_object* v___x_1734_; lean_object* v___x_1735_; lean_object* v___x_1736_; lean_object* v___x_1737_; lean_object* v___x_1738_; lean_object* v___x_1739_; lean_object* v___x_1740_; lean_object* v___x_1741_; lean_object* v___x_1742_; lean_object* v___x_1743_; lean_object* v___x_1744_; lean_object* v___x_1745_; lean_object* v___x_1746_; lean_object* v___x_1747_; lean_object* v___x_1748_; lean_object* v___x_1749_; lean_object* v___x_1750_; lean_object* v___x_1751_; lean_object* v___x_1752_; lean_object* v___x_1753_; lean_object* v___x_1754_; lean_object* v___x_1755_; lean_object* v___x_1756_; lean_object* v___x_1757_; lean_object* v___x_1758_; lean_object* v___x_1759_; lean_object* v___x_1760_; lean_object* v___x_1761_; lean_object* v___x_1762_; lean_object* v___x_1763_; lean_object* v___x_1764_; lean_object* v___x_1765_; lean_object* v___x_1766_; lean_object* v___x_1767_; lean_object* v___x_1768_; lean_object* v___x_1769_; lean_object* v___x_1770_; lean_object* v___x_1771_; lean_object* v___x_1772_; lean_object* v___x_1773_; lean_object* v___x_1774_; lean_object* v___x_1775_; lean_object* v___x_1776_; lean_object* v___x_1777_; lean_object* v___x_1778_; lean_object* v___x_1779_; lean_object* v___x_1780_; lean_object* v___x_1781_; lean_object* v___x_1782_; lean_object* v___x_1783_; lean_object* v___x_1784_; lean_object* v___x_1785_; lean_object* v___x_1786_; lean_object* v___x_1787_; lean_object* v___x_1788_; lean_object* v___x_1789_; lean_object* v___x_1790_; lean_object* v___x_1791_; lean_object* v___x_1792_; lean_object* v___x_1793_; lean_object* v___x_1794_; lean_object* v___x_1795_; lean_object* v___x_1796_; lean_object* v___x_1797_; lean_object* v___x_1798_; lean_object* v___x_1799_; lean_object* v___x_1800_; lean_object* v___x_1801_; lean_object* v___x_1802_; lean_object* v___x_1803_; lean_object* v___x_1804_; lean_object* v___x_1805_; lean_object* v___x_1806_; lean_object* v___x_1807_; 
v_ref_1718_ = lean_ctor_get(v_a_1715_, 5);
v_quotContext_1719_ = lean_ctor_get(v_a_1715_, 10);
v_currMacroScope_1720_ = lean_ctor_get(v_a_1715_, 11);
v___x_1721_ = 0;
v___x_1722_ = l_Lean_SourceInfo_fromRef(v_ref_1718_, v___x_1721_);
v___x_1723_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__2));
v___x_1724_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__3));
lean_inc_n(v___x_1722_, 48);
v___x_1725_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1725_, 0, v___x_1722_);
lean_ctor_set(v___x_1725_, 1, v___x_1723_);
v___x_1726_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__5));
v___x_1727_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__7));
v___x_1728_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__9));
v___x_1729_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__10));
v___x_1730_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__11));
v___x_1731_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1731_, 0, v___x_1722_);
lean_ctor_set(v___x_1731_, 1, v___x_1729_);
v___x_1732_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__13));
v___x_1733_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__14));
v___x_1734_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1734_, 0, v___x_1722_);
lean_ctor_set(v___x_1734_, 1, v___x_1733_);
v___x_1735_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__15));
v___x_1736_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__16));
v___x_1737_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1737_, 0, v___x_1722_);
lean_ctor_set(v___x_1737_, 1, v___x_1735_);
v___x_1738_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__18, &lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__18_once, _init_lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__18);
v___x_1739_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__19));
lean_inc_n(v_currMacroScope_1720_, 5);
lean_inc_n(v_quotContext_1719_, 5);
v___x_1740_ = l_Lean_addMacroScope(v_quotContext_1719_, v___x_1739_, v_currMacroScope_1720_);
v___x_1741_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__21));
v___x_1742_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1742_, 0, v___x_1722_);
lean_ctor_set(v___x_1742_, 1, v___x_1738_);
lean_ctor_set(v___x_1742_, 2, v___x_1740_);
lean_ctor_set(v___x_1742_, 3, v___x_1741_);
lean_inc_ref_n(v___x_1737_, 4);
v___x_1743_ = l_Lean_Syntax_node2(v___x_1722_, v___x_1736_, v___x_1737_, v___x_1742_);
v___x_1744_ = l_Lean_Syntax_node1(v___x_1722_, v___x_1728_, v___x_1743_);
v___x_1745_ = l_Lean_Syntax_node1(v___x_1722_, v___x_1727_, v___x_1744_);
v___x_1746_ = l_Lean_Syntax_node1(v___x_1722_, v___x_1726_, v___x_1745_);
lean_inc_ref_n(v___x_1734_, 5);
v___x_1747_ = l_Lean_Syntax_node2(v___x_1722_, v___x_1732_, v___x_1734_, v___x_1746_);
v___x_1748_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__23, &lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__23_once, _init_lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__23);
v___x_1749_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__24));
v___x_1750_ = l_Lean_addMacroScope(v_quotContext_1719_, v___x_1749_, v_currMacroScope_1720_);
v___x_1751_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__26));
v___x_1752_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1752_, 0, v___x_1722_);
lean_ctor_set(v___x_1752_, 1, v___x_1748_);
lean_ctor_set(v___x_1752_, 2, v___x_1750_);
lean_ctor_set(v___x_1752_, 3, v___x_1751_);
v___x_1753_ = l_Lean_Syntax_node2(v___x_1722_, v___x_1736_, v___x_1737_, v___x_1752_);
v___x_1754_ = l_Lean_Syntax_node1(v___x_1722_, v___x_1728_, v___x_1753_);
v___x_1755_ = l_Lean_Syntax_node1(v___x_1722_, v___x_1727_, v___x_1754_);
v___x_1756_ = l_Lean_Syntax_node1(v___x_1722_, v___x_1726_, v___x_1755_);
v___x_1757_ = l_Lean_Syntax_node2(v___x_1722_, v___x_1732_, v___x_1734_, v___x_1756_);
v___x_1758_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__28, &lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__28_once, _init_lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__28);
v___x_1759_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__29));
v___x_1760_ = l_Lean_addMacroScope(v_quotContext_1719_, v___x_1759_, v_currMacroScope_1720_);
v___x_1761_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__32));
v___x_1762_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1762_, 0, v___x_1722_);
lean_ctor_set(v___x_1762_, 1, v___x_1758_);
lean_ctor_set(v___x_1762_, 2, v___x_1760_);
lean_ctor_set(v___x_1762_, 3, v___x_1761_);
v___x_1763_ = l_Lean_Syntax_node2(v___x_1722_, v___x_1736_, v___x_1737_, v___x_1762_);
v___x_1764_ = l_Lean_Syntax_node1(v___x_1722_, v___x_1728_, v___x_1763_);
v___x_1765_ = l_Lean_Syntax_node1(v___x_1722_, v___x_1727_, v___x_1764_);
v___x_1766_ = l_Lean_Syntax_node1(v___x_1722_, v___x_1726_, v___x_1765_);
v___x_1767_ = l_Lean_Syntax_node2(v___x_1722_, v___x_1732_, v___x_1734_, v___x_1766_);
v___x_1768_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__34, &lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__34_once, _init_lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__34);
v___x_1769_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__35));
v___x_1770_ = l_Lean_addMacroScope(v_quotContext_1719_, v___x_1769_, v_currMacroScope_1720_);
v___x_1771_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__38));
v___x_1772_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1772_, 0, v___x_1722_);
lean_ctor_set(v___x_1772_, 1, v___x_1768_);
lean_ctor_set(v___x_1772_, 2, v___x_1770_);
lean_ctor_set(v___x_1772_, 3, v___x_1771_);
v___x_1773_ = l_Lean_Syntax_node2(v___x_1722_, v___x_1736_, v___x_1737_, v___x_1772_);
v___x_1774_ = l_Lean_Syntax_node1(v___x_1722_, v___x_1728_, v___x_1773_);
v___x_1775_ = l_Lean_Syntax_node1(v___x_1722_, v___x_1727_, v___x_1774_);
v___x_1776_ = l_Lean_Syntax_node1(v___x_1722_, v___x_1726_, v___x_1775_);
v___x_1777_ = l_Lean_Syntax_node2(v___x_1722_, v___x_1732_, v___x_1734_, v___x_1776_);
v___x_1778_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__40, &lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__40_once, _init_lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__40);
v___x_1779_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__41));
v___x_1780_ = l_Lean_addMacroScope(v_quotContext_1719_, v___x_1779_, v_currMacroScope_1720_);
v___x_1781_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__43));
v___x_1782_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1782_, 0, v___x_1722_);
lean_ctor_set(v___x_1782_, 1, v___x_1778_);
lean_ctor_set(v___x_1782_, 2, v___x_1780_);
lean_ctor_set(v___x_1782_, 3, v___x_1781_);
v___x_1783_ = l_Lean_Syntax_node2(v___x_1722_, v___x_1736_, v___x_1737_, v___x_1782_);
v___x_1784_ = l_Lean_Syntax_node1(v___x_1722_, v___x_1728_, v___x_1783_);
v___x_1785_ = l_Lean_Syntax_node1(v___x_1722_, v___x_1727_, v___x_1784_);
v___x_1786_ = l_Lean_Syntax_node1(v___x_1722_, v___x_1726_, v___x_1785_);
v___x_1787_ = l_Lean_Syntax_node2(v___x_1722_, v___x_1732_, v___x_1734_, v___x_1786_);
v___x_1788_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__44));
v___x_1789_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__45));
v___x_1790_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1790_, 0, v___x_1722_);
lean_ctor_set(v___x_1790_, 1, v___x_1788_);
v___x_1791_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__47));
v___x_1792_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__48));
v___x_1793_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1793_, 0, v___x_1722_);
lean_ctor_set(v___x_1793_, 1, v___x_1792_);
v___x_1794_ = l_Lean_Syntax_node1(v___x_1722_, v___x_1791_, v___x_1793_);
v___x_1795_ = l_Lean_Syntax_node1(v___x_1722_, v___x_1728_, v___x_1794_);
v___x_1796_ = l_Lean_Syntax_node2(v___x_1722_, v___x_1789_, v___x_1790_, v___x_1795_);
v___x_1797_ = l_Lean_Syntax_node1(v___x_1722_, v___x_1728_, v___x_1796_);
v___x_1798_ = l_Lean_Syntax_node1(v___x_1722_, v___x_1727_, v___x_1797_);
v___x_1799_ = l_Lean_Syntax_node1(v___x_1722_, v___x_1726_, v___x_1798_);
v___x_1800_ = l_Lean_Syntax_node2(v___x_1722_, v___x_1732_, v___x_1734_, v___x_1799_);
v___x_1801_ = l_Lean_Syntax_node6(v___x_1722_, v___x_1728_, v___x_1747_, v___x_1757_, v___x_1767_, v___x_1777_, v___x_1787_, v___x_1800_);
v___x_1802_ = l_Lean_Syntax_node2(v___x_1722_, v___x_1730_, v___x_1731_, v___x_1801_);
v___x_1803_ = l_Lean_Syntax_node1(v___x_1722_, v___x_1728_, v___x_1802_);
v___x_1804_ = l_Lean_Syntax_node1(v___x_1722_, v___x_1727_, v___x_1803_);
v___x_1805_ = l_Lean_Syntax_node1(v___x_1722_, v___x_1726_, v___x_1804_);
v___x_1806_ = l_Lean_Syntax_node2(v___x_1722_, v___x_1724_, v___x_1725_, v___x_1805_);
v___x_1807_ = l_Lean_Elab_Tactic_evalTactic(v___x_1806_, v_a_1709_, v_a_1710_, v_a_1711_, v_a_1712_, v_a_1713_, v_a_1714_, v_a_1715_, v_a_1716_);
return v___x_1807_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___boxed(lean_object* v_a_1808_, lean_object* v_a_1809_, lean_object* v_a_1810_, lean_object* v_a_1811_, lean_object* v_a_1812_, lean_object* v_a_1813_, lean_object* v_a_1814_, lean_object* v_a_1815_, lean_object* v_a_1816_){
_start:
{
lean_object* v_res_1817_; 
v_res_1817_ = lp_mathlib_Mathlib_Tactic_Peel_peelIffAux(v_a_1808_, v_a_1809_, v_a_1810_, v_a_1811_, v_a_1812_, v_a_1813_, v_a_1814_, v_a_1815_);
lean_dec(v_a_1815_);
lean_dec_ref(v_a_1814_);
lean_dec(v_a_1813_);
lean_dec_ref(v_a_1812_);
lean_dec(v_a_1811_);
lean_dec_ref(v_a_1810_);
lean_dec(v_a_1809_);
lean_dec_ref(v_a_1808_);
return v_res_1817_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelArgsIff___lam__0(lean_object* v_l_1818_, lean_object* v___y_1819_, lean_object* v___y_1820_, lean_object* v___y_1821_, lean_object* v___y_1822_, lean_object* v___y_1823_, lean_object* v___y_1824_, lean_object* v___y_1825_, lean_object* v___y_1826_){
_start:
{
if (lean_obj_tag(v_l_1818_) == 0)
{
lean_object* v___x_1828_; lean_object* v___x_1829_; 
v___x_1828_ = lean_box(0);
v___x_1829_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1829_, 0, v___x_1828_);
return v___x_1829_;
}
else
{
lean_object* v_head_1830_; lean_object* v_tail_1831_; lean_object* v___x_1833_; uint8_t v_isShared_1834_; uint8_t v_isSharedCheck_1863_; 
v_head_1830_ = lean_ctor_get(v_l_1818_, 0);
v_tail_1831_ = lean_ctor_get(v_l_1818_, 1);
v_isSharedCheck_1863_ = !lean_is_exclusive(v_l_1818_);
if (v_isSharedCheck_1863_ == 0)
{
v___x_1833_ = v_l_1818_;
v_isShared_1834_ = v_isSharedCheck_1863_;
goto v_resetjp_1832_;
}
else
{
lean_inc(v_tail_1831_);
lean_inc(v_head_1830_);
lean_dec(v_l_1818_);
v___x_1833_ = lean_box(0);
v_isShared_1834_ = v_isSharedCheck_1863_;
goto v_resetjp_1832_;
}
v_resetjp_1832_:
{
lean_object* v___x_1835_; 
v___x_1835_ = lp_mathlib_Mathlib_Tactic_Peel_peelIffAux(v___y_1819_, v___y_1820_, v___y_1821_, v___y_1822_, v___y_1823_, v___y_1824_, v___y_1825_, v___y_1826_);
if (lean_obj_tag(v___x_1835_) == 0)
{
lean_object* v___x_1836_; 
lean_dec_ref_known(v___x_1835_, 1);
v___x_1836_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_1820_, v___y_1823_, v___y_1824_, v___y_1825_, v___y_1826_);
if (lean_obj_tag(v___x_1836_) == 0)
{
lean_object* v_a_1837_; lean_object* v___x_1838_; 
v_a_1837_ = lean_ctor_get(v___x_1836_, 0);
lean_inc(v_a_1837_);
lean_dec_ref_known(v___x_1836_, 1);
v___x_1838_ = l_Lean_MVarId_intro(v_a_1837_, v_head_1830_, v___y_1823_, v___y_1824_, v___y_1825_, v___y_1826_);
if (lean_obj_tag(v___x_1838_) == 0)
{
lean_object* v_a_1839_; lean_object* v_snd_1840_; lean_object* v___x_1841_; lean_object* v___x_1843_; 
v_a_1839_ = lean_ctor_get(v___x_1838_, 0);
lean_inc(v_a_1839_);
lean_dec_ref_known(v___x_1838_, 1);
v_snd_1840_ = lean_ctor_get(v_a_1839_, 1);
lean_inc(v_snd_1840_);
lean_dec(v_a_1839_);
v___x_1841_ = lean_box(0);
if (v_isShared_1834_ == 0)
{
lean_ctor_set(v___x_1833_, 1, v___x_1841_);
lean_ctor_set(v___x_1833_, 0, v_snd_1840_);
v___x_1843_ = v___x_1833_;
goto v_reusejp_1842_;
}
else
{
lean_object* v_reuseFailAlloc_1846_; 
v_reuseFailAlloc_1846_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1846_, 0, v_snd_1840_);
lean_ctor_set(v_reuseFailAlloc_1846_, 1, v___x_1841_);
v___x_1843_ = v_reuseFailAlloc_1846_;
goto v_reusejp_1842_;
}
v_reusejp_1842_:
{
lean_object* v___x_1844_; 
v___x_1844_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_1843_, v___y_1820_, v___y_1823_, v___y_1824_, v___y_1825_, v___y_1826_);
if (lean_obj_tag(v___x_1844_) == 0)
{
lean_object* v___x_1845_; 
lean_dec_ref_known(v___x_1844_, 1);
v___x_1845_ = lp_mathlib_Mathlib_Tactic_Peel_peelArgsIff(v_tail_1831_, v___y_1819_, v___y_1820_, v___y_1821_, v___y_1822_, v___y_1823_, v___y_1824_, v___y_1825_, v___y_1826_);
return v___x_1845_;
}
else
{
lean_dec(v_tail_1831_);
return v___x_1844_;
}
}
}
else
{
lean_object* v_a_1847_; lean_object* v___x_1849_; uint8_t v_isShared_1850_; uint8_t v_isSharedCheck_1854_; 
lean_del_object(v___x_1833_);
lean_dec(v_tail_1831_);
v_a_1847_ = lean_ctor_get(v___x_1838_, 0);
v_isSharedCheck_1854_ = !lean_is_exclusive(v___x_1838_);
if (v_isSharedCheck_1854_ == 0)
{
v___x_1849_ = v___x_1838_;
v_isShared_1850_ = v_isSharedCheck_1854_;
goto v_resetjp_1848_;
}
else
{
lean_inc(v_a_1847_);
lean_dec(v___x_1838_);
v___x_1849_ = lean_box(0);
v_isShared_1850_ = v_isSharedCheck_1854_;
goto v_resetjp_1848_;
}
v_resetjp_1848_:
{
lean_object* v___x_1852_; 
if (v_isShared_1850_ == 0)
{
v___x_1852_ = v___x_1849_;
goto v_reusejp_1851_;
}
else
{
lean_object* v_reuseFailAlloc_1853_; 
v_reuseFailAlloc_1853_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1853_, 0, v_a_1847_);
v___x_1852_ = v_reuseFailAlloc_1853_;
goto v_reusejp_1851_;
}
v_reusejp_1851_:
{
return v___x_1852_;
}
}
}
}
else
{
lean_object* v_a_1855_; lean_object* v___x_1857_; uint8_t v_isShared_1858_; uint8_t v_isSharedCheck_1862_; 
lean_del_object(v___x_1833_);
lean_dec(v_tail_1831_);
lean_dec(v_head_1830_);
v_a_1855_ = lean_ctor_get(v___x_1836_, 0);
v_isSharedCheck_1862_ = !lean_is_exclusive(v___x_1836_);
if (v_isSharedCheck_1862_ == 0)
{
v___x_1857_ = v___x_1836_;
v_isShared_1858_ = v_isSharedCheck_1862_;
goto v_resetjp_1856_;
}
else
{
lean_inc(v_a_1855_);
lean_dec(v___x_1836_);
v___x_1857_ = lean_box(0);
v_isShared_1858_ = v_isSharedCheck_1862_;
goto v_resetjp_1856_;
}
v_resetjp_1856_:
{
lean_object* v___x_1860_; 
if (v_isShared_1858_ == 0)
{
v___x_1860_ = v___x_1857_;
goto v_reusejp_1859_;
}
else
{
lean_object* v_reuseFailAlloc_1861_; 
v_reuseFailAlloc_1861_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1861_, 0, v_a_1855_);
v___x_1860_ = v_reuseFailAlloc_1861_;
goto v_reusejp_1859_;
}
v_reusejp_1859_:
{
return v___x_1860_;
}
}
}
}
else
{
lean_del_object(v___x_1833_);
lean_dec(v_tail_1831_);
lean_dec(v_head_1830_);
return v___x_1835_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelArgsIff___lam__0___boxed(lean_object* v_l_1864_, lean_object* v___y_1865_, lean_object* v___y_1866_, lean_object* v___y_1867_, lean_object* v___y_1868_, lean_object* v___y_1869_, lean_object* v___y_1870_, lean_object* v___y_1871_, lean_object* v___y_1872_, lean_object* v___y_1873_){
_start:
{
lean_object* v_res_1874_; 
v_res_1874_ = lp_mathlib_Mathlib_Tactic_Peel_peelArgsIff___lam__0(v_l_1864_, v___y_1865_, v___y_1866_, v___y_1867_, v___y_1868_, v___y_1869_, v___y_1870_, v___y_1871_, v___y_1872_);
lean_dec(v___y_1872_);
lean_dec_ref(v___y_1871_);
lean_dec(v___y_1870_);
lean_dec_ref(v___y_1869_);
lean_dec(v___y_1868_);
lean_dec_ref(v___y_1867_);
lean_dec(v___y_1866_);
lean_dec_ref(v___y_1865_);
return v_res_1874_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelArgsIff(lean_object* v_l_1875_, lean_object* v_a_1876_, lean_object* v_a_1877_, lean_object* v_a_1878_, lean_object* v_a_1879_, lean_object* v_a_1880_, lean_object* v_a_1881_, lean_object* v_a_1882_, lean_object* v_a_1883_){
_start:
{
lean_object* v___y_1885_; lean_object* v___x_1886_; 
v___y_1885_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Peel_peelArgsIff___lam__0___boxed), 10, 1);
lean_closure_set(v___y_1885_, 0, v_l_1875_);
v___x_1886_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___y_1885_, v_a_1876_, v_a_1877_, v_a_1878_, v_a_1879_, v_a_1880_, v_a_1881_, v_a_1882_, v_a_1883_);
return v___x_1886_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel_peelArgsIff___boxed(lean_object* v_l_1887_, lean_object* v_a_1888_, lean_object* v_a_1889_, lean_object* v_a_1890_, lean_object* v_a_1891_, lean_object* v_a_1892_, lean_object* v_a_1893_, lean_object* v_a_1894_, lean_object* v_a_1895_, lean_object* v_a_1896_){
_start:
{
lean_object* v_res_1897_; 
v_res_1897_ = lp_mathlib_Mathlib_Tactic_Peel_peelArgsIff(v_l_1887_, v_a_1888_, v_a_1889_, v_a_1890_, v_a_1891_, v_a_1892_, v_a_1893_, v_a_1894_, v_a_1895_);
lean_dec(v_a_1895_);
lean_dec_ref(v_a_1894_);
lean_dec(v_a_1893_);
lean_dec_ref(v_a_1892_);
lean_dec(v_a_1891_);
lean_dec_ref(v_a_1890_);
lean_dec(v_a_1889_);
lean_dec_ref(v_a_1888_);
return v_res_1897_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg___closed__0(void){
_start:
{
lean_object* v___x_1898_; lean_object* v___x_1899_; lean_object* v___x_1900_; 
v___x_1898_ = lean_box(0);
v___x_1899_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1900_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1900_, 0, v___x_1899_);
lean_ctor_set(v___x_1900_, 1, v___x_1898_);
return v___x_1900_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg(){
_start:
{
lean_object* v___x_1902_; lean_object* v___x_1903_; 
v___x_1902_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg___closed__0);
v___x_1903_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1903_, 0, v___x_1902_);
return v___x_1903_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg___boxed(lean_object* v___y_1904_){
_start:
{
lean_object* v_res_1905_; 
v_res_1905_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v_res_1905_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1(lean_object* v_00_u03b1_1906_, lean_object* v___y_1907_, lean_object* v___y_1908_, lean_object* v___y_1909_, lean_object* v___y_1910_, lean_object* v___y_1911_, lean_object* v___y_1912_, lean_object* v___y_1913_, lean_object* v___y_1914_){
_start:
{
lean_object* v___x_1916_; 
v___x_1916_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_1916_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___boxed(lean_object* v_00_u03b1_1917_, lean_object* v___y_1918_, lean_object* v___y_1919_, lean_object* v___y_1920_, lean_object* v___y_1921_, lean_object* v___y_1922_, lean_object* v___y_1923_, lean_object* v___y_1924_, lean_object* v___y_1925_, lean_object* v___y_1926_){
_start:
{
lean_object* v_res_1927_; 
v_res_1927_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1(v_00_u03b1_1917_, v___y_1918_, v___y_1919_, v___y_1920_, v___y_1921_, v___y_1922_, v___y_1923_, v___y_1924_, v___y_1925_);
lean_dec(v___y_1925_);
lean_dec_ref(v___y_1924_);
lean_dec(v___y_1923_);
lean_dec_ref(v___y_1922_);
lean_dec(v___y_1921_);
lean_dec_ref(v___y_1920_);
lean_dec(v___y_1919_);
lean_dec_ref(v___y_1918_);
return v_res_1927_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__0(size_t v_sz_1928_, size_t v_i_1929_, lean_object* v_bs_1930_){
_start:
{
uint8_t v___x_1931_; 
v___x_1931_ = lean_usize_dec_lt(v_i_1929_, v_sz_1928_);
if (v___x_1931_ == 0)
{
return v_bs_1930_;
}
else
{
lean_object* v_v_1932_; lean_object* v___x_1933_; lean_object* v_bs_x27_1934_; lean_object* v___x_1935_; size_t v___x_1936_; size_t v___x_1937_; lean_object* v___x_1938_; 
v_v_1932_ = lean_array_uget(v_bs_1930_, v_i_1929_);
v___x_1933_ = lean_unsigned_to_nat(0u);
v_bs_x27_1934_ = lean_array_uset(v_bs_1930_, v_i_1929_, v___x_1933_);
v___x_1935_ = l_Lean_Elab_Tactic_getNameOfIdent_x27(v_v_1932_);
lean_dec(v_v_1932_);
v___x_1936_ = ((size_t)1ULL);
v___x_1937_ = lean_usize_add(v_i_1929_, v___x_1936_);
v___x_1938_ = lean_array_uset(v_bs_x27_1934_, v_i_1929_, v___x_1935_);
v_i_1929_ = v___x_1937_;
v_bs_1930_ = v___x_1938_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__0___boxed(lean_object* v_sz_1940_, lean_object* v_i_1941_, lean_object* v_bs_1942_){
_start:
{
size_t v_sz_boxed_1943_; size_t v_i_boxed_1944_; lean_object* v_res_1945_; 
v_sz_boxed_1943_ = lean_unbox_usize(v_sz_1940_);
lean_dec(v_sz_1940_);
v_i_boxed_1944_ = lean_unbox_usize(v_i_1941_);
lean_dec(v_i_1941_);
v_res_1945_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__0(v_sz_boxed_1943_, v_i_boxed_1944_, v_bs_1942_);
return v_res_1945_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1___lam__0(lean_object* v___x_1946_, uint8_t v___x_1947_, lean_object* v_num_x3f_1948_, uint8_t v___x_1949_, lean_object* v_l_x3f_1950_, lean_object* v___x_1951_, lean_object* v_n_x3f_1952_, lean_object* v___y_1953_, lean_object* v___y_1954_, lean_object* v___y_1955_, lean_object* v___y_1956_, lean_object* v___y_1957_, lean_object* v___y_1958_, lean_object* v___y_1959_, lean_object* v___y_1960_){
_start:
{
lean_object* v___x_1962_; 
v___x_1962_ = l_Lean_Elab_Tactic_elabTermForApply(v___x_1946_, v___x_1947_, v___y_1953_, v___y_1954_, v___y_1955_, v___y_1956_, v___y_1957_, v___y_1958_, v___y_1959_, v___y_1960_);
if (lean_obj_tag(v___x_1962_) == 0)
{
lean_object* v_a_1963_; lean_object* v___y_1965_; lean_object* v___y_1966_; lean_object* v___y_2018_; 
v_a_1963_ = lean_ctor_get(v___x_1962_, 0);
lean_inc(v_a_1963_);
lean_dec_ref_known(v___x_1962_, 1);
if (lean_obj_tag(v_n_x3f_1952_) == 0)
{
lean_object* v___x_2021_; 
v___x_2021_ = lean_box(0);
v___y_2018_ = v___x_2021_;
goto v___jp_2017_;
}
else
{
lean_object* v_val_2022_; lean_object* v___x_2024_; uint8_t v_isShared_2025_; uint8_t v_isSharedCheck_2032_; 
v_val_2022_ = lean_ctor_get(v_n_x3f_1952_, 0);
v_isSharedCheck_2032_ = !lean_is_exclusive(v_n_x3f_1952_);
if (v_isSharedCheck_2032_ == 0)
{
v___x_2024_ = v_n_x3f_1952_;
v_isShared_2025_ = v_isSharedCheck_2032_;
goto v_resetjp_2023_;
}
else
{
lean_inc(v_val_2022_);
lean_dec(v_n_x3f_1952_);
v___x_2024_ = lean_box(0);
v_isShared_2025_ = v_isSharedCheck_2032_;
goto v_resetjp_2023_;
}
v_resetjp_2023_:
{
uint8_t v___x_2026_; 
v___x_2026_ = l_Lean_Syntax_isIdent(v_val_2022_);
if (v___x_2026_ == 0)
{
lean_object* v___x_2027_; 
lean_del_object(v___x_2024_);
lean_dec(v_val_2022_);
v___x_2027_ = lean_box(0);
v___y_2018_ = v___x_2027_;
goto v___jp_2017_;
}
else
{
lean_object* v___x_2028_; lean_object* v___x_2030_; 
v___x_2028_ = l_Lean_Syntax_getId(v_val_2022_);
lean_dec(v_val_2022_);
if (v_isShared_2025_ == 0)
{
lean_ctor_set(v___x_2024_, 0, v___x_2028_);
v___x_2030_ = v___x_2024_;
goto v_reusejp_2029_;
}
else
{
lean_object* v_reuseFailAlloc_2031_; 
v_reuseFailAlloc_2031_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2031_, 0, v___x_2028_);
v___x_2030_ = v_reuseFailAlloc_2031_;
goto v_reusejp_2029_;
}
v_reusejp_2029_:
{
v___y_2018_ = v___x_2030_;
goto v___jp_2017_;
}
}
}
}
v___jp_1964_:
{
size_t v_sz_1967_; size_t v___x_1968_; lean_object* v___x_1969_; lean_object* v___x_1970_; 
v_sz_1967_ = lean_array_size(v___y_1966_);
v___x_1968_ = ((size_t)0ULL);
v___x_1969_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__0(v_sz_1967_, v___x_1968_, v___y_1966_);
v___x_1970_ = lean_array_to_list(v___x_1969_);
if (lean_obj_tag(v_num_x3f_1948_) == 0)
{
uint8_t v___x_1971_; 
v___x_1971_ = l_List_isEmpty___redArg(v___x_1970_);
if (v___x_1971_ == 0)
{
lean_object* v___x_1972_; lean_object* v___x_1973_; 
v___x_1972_ = l_List_lengthTR___redArg(v___x_1970_);
v___x_1973_ = lp_mathlib_Mathlib_Tactic_Peel_peelArgs(v_a_1963_, v___x_1972_, v___x_1970_, v___y_1965_, v___x_1949_, v___y_1953_, v___y_1954_, v___y_1955_, v___y_1956_, v___y_1957_, v___y_1958_, v___y_1959_, v___y_1960_);
lean_dec(v___y_1960_);
lean_dec_ref(v___y_1959_);
lean_dec(v___y_1958_);
lean_dec_ref(v___y_1957_);
lean_dec(v___x_1972_);
return v___x_1973_;
}
else
{
lean_object* v___x_1974_; 
lean_dec(v___x_1970_);
lean_inc(v_a_1963_);
v___x_1974_ = lp_mathlib_Mathlib_Tactic_Peel_peelUnbounded(v_a_1963_, v___y_1965_, v___x_1947_, v___y_1953_, v___y_1954_, v___y_1955_, v___y_1956_, v___y_1957_, v___y_1958_, v___y_1959_, v___y_1960_);
if (lean_obj_tag(v___x_1974_) == 0)
{
lean_object* v_a_1975_; lean_object* v___x_1977_; uint8_t v_isShared_1978_; uint8_t v_isSharedCheck_2005_; 
v_a_1975_ = lean_ctor_get(v___x_1974_, 0);
v_isSharedCheck_2005_ = !lean_is_exclusive(v___x_1974_);
if (v_isSharedCheck_2005_ == 0)
{
v___x_1977_ = v___x_1974_;
v_isShared_1978_ = v_isSharedCheck_2005_;
goto v_resetjp_1976_;
}
else
{
lean_inc(v_a_1975_);
lean_dec(v___x_1974_);
v___x_1977_ = lean_box(0);
v_isShared_1978_ = v_isSharedCheck_2005_;
goto v_resetjp_1976_;
}
v_resetjp_1976_:
{
uint8_t v___x_1979_; 
v___x_1979_ = lean_unbox(v_a_1975_);
lean_dec(v_a_1975_);
if (v___x_1979_ == 0)
{
lean_object* v___x_1980_; 
lean_del_object(v___x_1977_);
lean_inc(v___y_1960_);
lean_inc_ref(v___y_1959_);
lean_inc(v___y_1958_);
lean_inc_ref(v___y_1957_);
v___x_1980_ = lean_infer_type(v_a_1963_, v___y_1957_, v___y_1958_, v___y_1959_, v___y_1960_);
if (lean_obj_tag(v___x_1980_) == 0)
{
lean_object* v_a_1981_; lean_object* v___x_1982_; 
v_a_1981_ = lean_ctor_get(v___x_1980_, 0);
lean_inc(v_a_1981_);
lean_dec_ref_known(v___x_1980_, 1);
v___x_1982_ = l_Lean_Elab_Tactic_getMainTarget(v___y_1953_, v___y_1954_, v___y_1955_, v___y_1956_, v___y_1957_, v___y_1958_, v___y_1959_, v___y_1960_);
if (lean_obj_tag(v___x_1982_) == 0)
{
lean_object* v_a_1983_; lean_object* v___x_1984_; 
v_a_1983_ = lean_ctor_get(v___x_1982_, 0);
lean_inc(v_a_1983_);
lean_dec_ref_known(v___x_1982_, 1);
v___x_1984_ = lp_mathlib_Mathlib_Tactic_Peel_throwPeelError___redArg(v_a_1981_, v_a_1983_, v___y_1957_, v___y_1958_, v___y_1959_, v___y_1960_);
lean_dec(v___y_1960_);
lean_dec_ref(v___y_1959_);
lean_dec(v___y_1958_);
lean_dec_ref(v___y_1957_);
return v___x_1984_;
}
else
{
lean_object* v_a_1985_; lean_object* v___x_1987_; uint8_t v_isShared_1988_; uint8_t v_isSharedCheck_1992_; 
lean_dec(v_a_1981_);
lean_dec(v___y_1960_);
lean_dec_ref(v___y_1959_);
lean_dec(v___y_1958_);
lean_dec_ref(v___y_1957_);
v_a_1985_ = lean_ctor_get(v___x_1982_, 0);
v_isSharedCheck_1992_ = !lean_is_exclusive(v___x_1982_);
if (v_isSharedCheck_1992_ == 0)
{
v___x_1987_ = v___x_1982_;
v_isShared_1988_ = v_isSharedCheck_1992_;
goto v_resetjp_1986_;
}
else
{
lean_inc(v_a_1985_);
lean_dec(v___x_1982_);
v___x_1987_ = lean_box(0);
v_isShared_1988_ = v_isSharedCheck_1992_;
goto v_resetjp_1986_;
}
v_resetjp_1986_:
{
lean_object* v___x_1990_; 
if (v_isShared_1988_ == 0)
{
v___x_1990_ = v___x_1987_;
goto v_reusejp_1989_;
}
else
{
lean_object* v_reuseFailAlloc_1991_; 
v_reuseFailAlloc_1991_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1991_, 0, v_a_1985_);
v___x_1990_ = v_reuseFailAlloc_1991_;
goto v_reusejp_1989_;
}
v_reusejp_1989_:
{
return v___x_1990_;
}
}
}
}
else
{
lean_object* v_a_1993_; lean_object* v___x_1995_; uint8_t v_isShared_1996_; uint8_t v_isSharedCheck_2000_; 
lean_dec(v___y_1960_);
lean_dec_ref(v___y_1959_);
lean_dec(v___y_1958_);
lean_dec_ref(v___y_1957_);
v_a_1993_ = lean_ctor_get(v___x_1980_, 0);
v_isSharedCheck_2000_ = !lean_is_exclusive(v___x_1980_);
if (v_isSharedCheck_2000_ == 0)
{
v___x_1995_ = v___x_1980_;
v_isShared_1996_ = v_isSharedCheck_2000_;
goto v_resetjp_1994_;
}
else
{
lean_inc(v_a_1993_);
lean_dec(v___x_1980_);
v___x_1995_ = lean_box(0);
v_isShared_1996_ = v_isSharedCheck_2000_;
goto v_resetjp_1994_;
}
v_resetjp_1994_:
{
lean_object* v___x_1998_; 
if (v_isShared_1996_ == 0)
{
v___x_1998_ = v___x_1995_;
goto v_reusejp_1997_;
}
else
{
lean_object* v_reuseFailAlloc_1999_; 
v_reuseFailAlloc_1999_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1999_, 0, v_a_1993_);
v___x_1998_ = v_reuseFailAlloc_1999_;
goto v_reusejp_1997_;
}
v_reusejp_1997_:
{
return v___x_1998_;
}
}
}
}
else
{
lean_object* v___x_2001_; lean_object* v___x_2003_; 
lean_dec(v_a_1963_);
lean_dec(v___y_1960_);
lean_dec_ref(v___y_1959_);
lean_dec(v___y_1958_);
lean_dec_ref(v___y_1957_);
v___x_2001_ = lean_box(0);
if (v_isShared_1978_ == 0)
{
lean_ctor_set(v___x_1977_, 0, v___x_2001_);
v___x_2003_ = v___x_1977_;
goto v_reusejp_2002_;
}
else
{
lean_object* v_reuseFailAlloc_2004_; 
v_reuseFailAlloc_2004_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2004_, 0, v___x_2001_);
v___x_2003_ = v_reuseFailAlloc_2004_;
goto v_reusejp_2002_;
}
v_reusejp_2002_:
{
return v___x_2003_;
}
}
}
}
else
{
lean_object* v_a_2006_; lean_object* v___x_2008_; uint8_t v_isShared_2009_; uint8_t v_isSharedCheck_2013_; 
lean_dec(v_a_1963_);
lean_dec(v___y_1960_);
lean_dec_ref(v___y_1959_);
lean_dec(v___y_1958_);
lean_dec_ref(v___y_1957_);
v_a_2006_ = lean_ctor_get(v___x_1974_, 0);
v_isSharedCheck_2013_ = !lean_is_exclusive(v___x_1974_);
if (v_isSharedCheck_2013_ == 0)
{
v___x_2008_ = v___x_1974_;
v_isShared_2009_ = v_isSharedCheck_2013_;
goto v_resetjp_2007_;
}
else
{
lean_inc(v_a_2006_);
lean_dec(v___x_1974_);
v___x_2008_ = lean_box(0);
v_isShared_2009_ = v_isSharedCheck_2013_;
goto v_resetjp_2007_;
}
v_resetjp_2007_:
{
lean_object* v___x_2011_; 
if (v_isShared_2009_ == 0)
{
v___x_2011_ = v___x_2008_;
goto v_reusejp_2010_;
}
else
{
lean_object* v_reuseFailAlloc_2012_; 
v_reuseFailAlloc_2012_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2012_, 0, v_a_2006_);
v___x_2011_ = v_reuseFailAlloc_2012_;
goto v_reusejp_2010_;
}
v_reusejp_2010_:
{
return v___x_2011_;
}
}
}
}
}
else
{
lean_object* v_val_2014_; lean_object* v___x_2015_; lean_object* v___x_2016_; 
v_val_2014_ = lean_ctor_get(v_num_x3f_1948_, 0);
v___x_2015_ = l_Lean_TSyntax_getNat(v_val_2014_);
v___x_2016_ = lp_mathlib_Mathlib_Tactic_Peel_peelArgs(v_a_1963_, v___x_2015_, v___x_1970_, v___y_1965_, v___x_1949_, v___y_1953_, v___y_1954_, v___y_1955_, v___y_1956_, v___y_1957_, v___y_1958_, v___y_1959_, v___y_1960_);
lean_dec(v___y_1960_);
lean_dec_ref(v___y_1959_);
lean_dec(v___y_1958_);
lean_dec_ref(v___y_1957_);
lean_dec(v___x_2015_);
return v___x_2016_;
}
}
v___jp_2017_:
{
if (lean_obj_tag(v_l_x3f_1950_) == 0)
{
lean_object* v___x_2019_; 
v___x_2019_ = lean_mk_empty_array_with_capacity(v___x_1951_);
v___y_1965_ = v___y_2018_;
v___y_1966_ = v___x_2019_;
goto v___jp_1964_;
}
else
{
lean_object* v_val_2020_; 
v_val_2020_ = lean_ctor_get(v_l_x3f_1950_, 0);
lean_inc(v_val_2020_);
lean_dec_ref_known(v_l_x3f_1950_, 1);
v___y_1965_ = v___y_2018_;
v___y_1966_ = v_val_2020_;
goto v___jp_1964_;
}
}
}
else
{
lean_object* v_a_2033_; lean_object* v___x_2035_; uint8_t v_isShared_2036_; uint8_t v_isSharedCheck_2040_; 
lean_dec(v___y_1960_);
lean_dec_ref(v___y_1959_);
lean_dec(v___y_1958_);
lean_dec_ref(v___y_1957_);
lean_dec(v_n_x3f_1952_);
lean_dec(v_l_x3f_1950_);
v_a_2033_ = lean_ctor_get(v___x_1962_, 0);
v_isSharedCheck_2040_ = !lean_is_exclusive(v___x_1962_);
if (v_isSharedCheck_2040_ == 0)
{
v___x_2035_ = v___x_1962_;
v_isShared_2036_ = v_isSharedCheck_2040_;
goto v_resetjp_2034_;
}
else
{
lean_inc(v_a_2033_);
lean_dec(v___x_1962_);
v___x_2035_ = lean_box(0);
v_isShared_2036_ = v_isSharedCheck_2040_;
goto v_resetjp_2034_;
}
v_resetjp_2034_:
{
lean_object* v___x_2038_; 
if (v_isShared_2036_ == 0)
{
v___x_2038_ = v___x_2035_;
goto v_reusejp_2037_;
}
else
{
lean_object* v_reuseFailAlloc_2039_; 
v_reuseFailAlloc_2039_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2039_, 0, v_a_2033_);
v___x_2038_ = v_reuseFailAlloc_2039_;
goto v_reusejp_2037_;
}
v_reusejp_2037_:
{
return v___x_2038_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1___lam__0___boxed(lean_object* v___x_2041_, lean_object* v___x_2042_, lean_object* v_num_x3f_2043_, lean_object* v___x_2044_, lean_object* v_l_x3f_2045_, lean_object* v___x_2046_, lean_object* v_n_x3f_2047_, lean_object* v___y_2048_, lean_object* v___y_2049_, lean_object* v___y_2050_, lean_object* v___y_2051_, lean_object* v___y_2052_, lean_object* v___y_2053_, lean_object* v___y_2054_, lean_object* v___y_2055_, lean_object* v___y_2056_){
_start:
{
uint8_t v___x_15862__boxed_2057_; uint8_t v___x_15863__boxed_2058_; lean_object* v_res_2059_; 
v___x_15862__boxed_2057_ = lean_unbox(v___x_2042_);
v___x_15863__boxed_2058_ = lean_unbox(v___x_2044_);
v_res_2059_ = lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1___lam__0(v___x_2041_, v___x_15862__boxed_2057_, v_num_x3f_2043_, v___x_15863__boxed_2058_, v_l_x3f_2045_, v___x_2046_, v_n_x3f_2047_, v___y_2048_, v___y_2049_, v___y_2050_, v___y_2051_, v___y_2052_, v___y_2053_, v___y_2054_, v___y_2055_);
lean_dec(v___y_2051_);
lean_dec_ref(v___y_2050_);
lean_dec(v___y_2049_);
lean_dec_ref(v___y_2048_);
lean_dec(v___x_2046_);
lean_dec(v_num_x3f_2043_);
return v_res_2059_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1(lean_object* v_x_2063_, lean_object* v_a_2064_, lean_object* v_a_2065_, lean_object* v_a_2066_, lean_object* v_a_2067_, lean_object* v_a_2068_, lean_object* v_a_2069_, lean_object* v_a_2070_, lean_object* v_a_2071_){
_start:
{
lean_object* v_n_2074_; lean_object* v___y_2075_; lean_object* v___y_2076_; lean_object* v___y_2077_; lean_object* v___y_2078_; lean_object* v___y_2079_; lean_object* v___y_2080_; lean_object* v___y_2081_; lean_object* v___y_2082_; lean_object* v_args_2088_; lean_object* v___y_2089_; lean_object* v___y_2090_; lean_object* v___y_2091_; lean_object* v___y_2092_; lean_object* v___y_2093_; lean_object* v___y_2094_; lean_object* v___y_2095_; lean_object* v___y_2096_; lean_object* v___x_2102_; uint8_t v___x_2103_; 
v___x_2102_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peel___closed__4));
lean_inc(v_x_2063_);
v___x_2103_ = l_Lean_Syntax_isOfKind(v_x_2063_, v___x_2102_);
if (v___x_2103_ == 0)
{
lean_object* v___x_2104_; 
lean_dec(v_x_2063_);
v___x_2104_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2104_;
}
else
{
lean_object* v___x_2105_; lean_object* v___x_2106_; lean_object* v___x_2107_; lean_object* v___y_2109_; lean_object* v___y_2110_; lean_object* v___y_2111_; lean_object* v___y_2112_; lean_object* v___y_2113_; lean_object* v_l_x3f_2114_; lean_object* v_n_x3f_2115_; lean_object* v___y_2116_; lean_object* v___y_2117_; lean_object* v___y_2118_; lean_object* v___y_2119_; lean_object* v___y_2120_; lean_object* v___y_2121_; lean_object* v___y_2122_; lean_object* v___y_2123_; lean_object* v_num_x3f_2152_; lean_object* v___y_2153_; lean_object* v___y_2154_; lean_object* v___y_2155_; lean_object* v___y_2156_; lean_object* v___y_2157_; lean_object* v___y_2158_; lean_object* v___y_2159_; lean_object* v___y_2160_; uint8_t v___x_2251_; 
v___x_2105_ = lean_unsigned_to_nat(0u);
v___x_2106_ = lean_unsigned_to_nat(1u);
v___x_2107_ = l_Lean_Syntax_getArg(v_x_2063_, v___x_2106_);
v___x_2251_ = l_Lean_Syntax_isNone(v___x_2107_);
if (v___x_2251_ == 0)
{
uint8_t v___x_2252_; 
lean_inc(v___x_2107_);
v___x_2252_ = l_Lean_Syntax_matchesNull(v___x_2107_, v___x_2106_);
if (v___x_2252_ == 0)
{
if (v___x_2252_ == 0)
{
uint8_t v___x_2253_; 
v___x_2253_ = l_Lean_Syntax_matchesNull(v___x_2107_, v___x_2105_);
if (v___x_2253_ == 0)
{
lean_object* v___x_2254_; 
lean_dec(v_x_2063_);
v___x_2254_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2254_;
}
else
{
lean_object* v___x_2255_; lean_object* v___x_2256_; uint8_t v___x_2257_; 
v___x_2255_ = lean_unsigned_to_nat(2u);
v___x_2256_ = l_Lean_Syntax_getArg(v_x_2063_, v___x_2255_);
v___x_2257_ = l_Lean_Syntax_matchesNull(v___x_2256_, v___x_2105_);
if (v___x_2257_ == 0)
{
lean_object* v___x_2258_; 
lean_dec(v_x_2063_);
v___x_2258_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2258_;
}
else
{
lean_object* v___x_2259_; lean_object* v___x_2260_; uint8_t v___x_2261_; 
v___x_2259_ = lean_unsigned_to_nat(3u);
v___x_2260_ = l_Lean_Syntax_getArg(v_x_2063_, v___x_2259_);
lean_inc(v___x_2260_);
v___x_2261_ = l_Lean_Syntax_matchesNull(v___x_2260_, v___x_2255_);
if (v___x_2261_ == 0)
{
lean_object* v___x_2262_; 
lean_dec(v___x_2260_);
lean_dec(v_x_2063_);
v___x_2262_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2262_;
}
else
{
lean_object* v___x_2263_; lean_object* v___x_2264_; uint8_t v___x_2265_; 
v___x_2263_ = lean_unsigned_to_nat(4u);
v___x_2264_ = l_Lean_Syntax_getArg(v_x_2063_, v___x_2263_);
lean_dec(v_x_2063_);
v___x_2265_ = l_Lean_Syntax_matchesNull(v___x_2264_, v___x_2105_);
if (v___x_2265_ == 0)
{
lean_object* v___x_2266_; 
lean_dec(v___x_2260_);
v___x_2266_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2266_;
}
else
{
lean_object* v___x_2267_; lean_object* v_args_2268_; 
v___x_2267_ = l_Lean_Syntax_getArg(v___x_2260_, v___x_2106_);
lean_dec(v___x_2260_);
v_args_2268_ = l_Lean_Syntax_getArgs(v___x_2267_);
lean_dec(v___x_2267_);
v_args_2088_ = v_args_2268_;
v___y_2089_ = v_a_2064_;
v___y_2090_ = v_a_2065_;
v___y_2091_ = v_a_2066_;
v___y_2092_ = v_a_2067_;
v___y_2093_ = v_a_2068_;
v___y_2094_ = v_a_2069_;
v___y_2095_ = v_a_2070_;
v___y_2096_ = v_a_2071_;
goto v___jp_2087_;
}
}
}
}
}
else
{
lean_object* v_n_2269_; lean_object* v___x_2270_; uint8_t v___x_2271_; 
v_n_2269_ = l_Lean_Syntax_getArg(v___x_2107_, v___x_2105_);
lean_dec(v___x_2107_);
v___x_2270_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peel___closed__11));
lean_inc(v_n_2269_);
v___x_2271_ = l_Lean_Syntax_isOfKind(v_n_2269_, v___x_2270_);
if (v___x_2271_ == 0)
{
lean_object* v___x_2272_; 
lean_dec(v_n_2269_);
lean_dec(v_x_2063_);
v___x_2272_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2272_;
}
else
{
lean_object* v___x_2273_; lean_object* v___x_2274_; uint8_t v___x_2275_; 
v___x_2273_ = lean_unsigned_to_nat(2u);
v___x_2274_ = l_Lean_Syntax_getArg(v_x_2063_, v___x_2273_);
v___x_2275_ = l_Lean_Syntax_matchesNull(v___x_2274_, v___x_2105_);
if (v___x_2275_ == 0)
{
lean_object* v___x_2276_; 
lean_dec(v_n_2269_);
lean_dec(v_x_2063_);
v___x_2276_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2276_;
}
else
{
lean_object* v___x_2277_; lean_object* v___x_2278_; uint8_t v___x_2279_; 
v___x_2277_ = lean_unsigned_to_nat(3u);
v___x_2278_ = l_Lean_Syntax_getArg(v_x_2063_, v___x_2277_);
v___x_2279_ = l_Lean_Syntax_matchesNull(v___x_2278_, v___x_2105_);
if (v___x_2279_ == 0)
{
lean_object* v___x_2280_; 
lean_dec(v_n_2269_);
lean_dec(v_x_2063_);
v___x_2280_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2280_;
}
else
{
lean_object* v___x_2281_; lean_object* v___x_2282_; uint8_t v___x_2283_; 
v___x_2281_ = lean_unsigned_to_nat(4u);
v___x_2282_ = l_Lean_Syntax_getArg(v_x_2063_, v___x_2281_);
lean_dec(v_x_2063_);
v___x_2283_ = l_Lean_Syntax_matchesNull(v___x_2282_, v___x_2105_);
if (v___x_2283_ == 0)
{
lean_object* v___x_2284_; 
lean_dec(v_n_2269_);
v___x_2284_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2284_;
}
else
{
v_n_2074_ = v_n_2269_;
v___y_2075_ = v_a_2064_;
v___y_2076_ = v_a_2065_;
v___y_2077_ = v_a_2066_;
v___y_2078_ = v_a_2067_;
v___y_2079_ = v_a_2068_;
v___y_2080_ = v_a_2069_;
v___y_2081_ = v_a_2070_;
v___y_2082_ = v_a_2071_;
goto v___jp_2073_;
}
}
}
}
}
}
else
{
lean_object* v_num_x3f_2285_; lean_object* v___x_2286_; uint8_t v___x_2287_; 
v_num_x3f_2285_ = l_Lean_Syntax_getArg(v___x_2107_, v___x_2105_);
v___x_2286_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peel___closed__11));
lean_inc(v_num_x3f_2285_);
v___x_2287_ = l_Lean_Syntax_isOfKind(v_num_x3f_2285_, v___x_2286_);
if (v___x_2287_ == 0)
{
if (v___x_2252_ == 0)
{
uint8_t v___x_2288_; 
lean_dec(v_num_x3f_2285_);
v___x_2288_ = l_Lean_Syntax_matchesNull(v___x_2107_, v___x_2105_);
if (v___x_2288_ == 0)
{
lean_object* v___x_2289_; 
lean_dec(v_x_2063_);
v___x_2289_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2289_;
}
else
{
lean_object* v___x_2290_; lean_object* v___x_2291_; uint8_t v___x_2292_; 
v___x_2290_ = lean_unsigned_to_nat(2u);
v___x_2291_ = l_Lean_Syntax_getArg(v_x_2063_, v___x_2290_);
v___x_2292_ = l_Lean_Syntax_matchesNull(v___x_2291_, v___x_2105_);
if (v___x_2292_ == 0)
{
lean_object* v___x_2293_; 
lean_dec(v_x_2063_);
v___x_2293_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2293_;
}
else
{
lean_object* v___x_2294_; lean_object* v___x_2295_; uint8_t v___x_2296_; 
v___x_2294_ = lean_unsigned_to_nat(3u);
v___x_2295_ = l_Lean_Syntax_getArg(v_x_2063_, v___x_2294_);
lean_inc(v___x_2295_);
v___x_2296_ = l_Lean_Syntax_matchesNull(v___x_2295_, v___x_2290_);
if (v___x_2296_ == 0)
{
lean_object* v___x_2297_; 
lean_dec(v___x_2295_);
lean_dec(v_x_2063_);
v___x_2297_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2297_;
}
else
{
lean_object* v___x_2298_; lean_object* v___x_2299_; uint8_t v___x_2300_; 
v___x_2298_ = lean_unsigned_to_nat(4u);
v___x_2299_ = l_Lean_Syntax_getArg(v_x_2063_, v___x_2298_);
lean_dec(v_x_2063_);
v___x_2300_ = l_Lean_Syntax_matchesNull(v___x_2299_, v___x_2105_);
if (v___x_2300_ == 0)
{
lean_object* v___x_2301_; 
lean_dec(v___x_2295_);
v___x_2301_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2301_;
}
else
{
lean_object* v___x_2302_; lean_object* v_args_2303_; 
v___x_2302_ = l_Lean_Syntax_getArg(v___x_2295_, v___x_2106_);
lean_dec(v___x_2295_);
v_args_2303_ = l_Lean_Syntax_getArgs(v___x_2302_);
lean_dec(v___x_2302_);
v_args_2088_ = v_args_2303_;
v___y_2089_ = v_a_2064_;
v___y_2090_ = v_a_2065_;
v___y_2091_ = v_a_2066_;
v___y_2092_ = v_a_2067_;
v___y_2093_ = v_a_2068_;
v___y_2094_ = v_a_2069_;
v___y_2095_ = v_a_2070_;
v___y_2096_ = v_a_2071_;
goto v___jp_2087_;
}
}
}
}
}
else
{
lean_dec(v___x_2107_);
if (v___x_2287_ == 0)
{
lean_object* v___x_2304_; 
lean_dec(v_num_x3f_2285_);
lean_dec(v_x_2063_);
v___x_2304_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2304_;
}
else
{
lean_object* v___x_2305_; lean_object* v___x_2306_; uint8_t v___x_2307_; 
v___x_2305_ = lean_unsigned_to_nat(2u);
v___x_2306_ = l_Lean_Syntax_getArg(v_x_2063_, v___x_2305_);
v___x_2307_ = l_Lean_Syntax_matchesNull(v___x_2306_, v___x_2105_);
if (v___x_2307_ == 0)
{
lean_object* v___x_2308_; 
lean_dec(v_num_x3f_2285_);
lean_dec(v_x_2063_);
v___x_2308_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2308_;
}
else
{
lean_object* v___x_2309_; lean_object* v___x_2310_; uint8_t v___x_2311_; 
v___x_2309_ = lean_unsigned_to_nat(3u);
v___x_2310_ = l_Lean_Syntax_getArg(v_x_2063_, v___x_2309_);
v___x_2311_ = l_Lean_Syntax_matchesNull(v___x_2310_, v___x_2105_);
if (v___x_2311_ == 0)
{
lean_object* v___x_2312_; 
lean_dec(v_num_x3f_2285_);
lean_dec(v_x_2063_);
v___x_2312_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2312_;
}
else
{
lean_object* v___x_2313_; lean_object* v___x_2314_; uint8_t v___x_2315_; 
v___x_2313_ = lean_unsigned_to_nat(4u);
v___x_2314_ = l_Lean_Syntax_getArg(v_x_2063_, v___x_2313_);
lean_dec(v_x_2063_);
v___x_2315_ = l_Lean_Syntax_matchesNull(v___x_2314_, v___x_2105_);
if (v___x_2315_ == 0)
{
lean_object* v___x_2316_; 
lean_dec(v_num_x3f_2285_);
v___x_2316_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2316_;
}
else
{
v_n_2074_ = v_num_x3f_2285_;
v___y_2075_ = v_a_2064_;
v___y_2076_ = v_a_2065_;
v___y_2077_ = v_a_2066_;
v___y_2078_ = v_a_2067_;
v___y_2079_ = v_a_2068_;
v___y_2080_ = v_a_2069_;
v___y_2081_ = v_a_2070_;
v___y_2082_ = v_a_2071_;
goto v___jp_2073_;
}
}
}
}
}
}
else
{
lean_object* v___x_2317_; 
v___x_2317_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2317_, 0, v_num_x3f_2285_);
v_num_x3f_2152_ = v___x_2317_;
v___y_2153_ = v_a_2064_;
v___y_2154_ = v_a_2065_;
v___y_2155_ = v_a_2066_;
v___y_2156_ = v_a_2067_;
v___y_2157_ = v_a_2068_;
v___y_2158_ = v_a_2069_;
v___y_2159_ = v_a_2070_;
v___y_2160_ = v_a_2071_;
goto v___jp_2151_;
}
}
}
else
{
lean_object* v___x_2318_; 
v___x_2318_ = lean_box(0);
v_num_x3f_2152_ = v___x_2318_;
v___y_2153_ = v_a_2064_;
v___y_2154_ = v_a_2065_;
v___y_2155_ = v_a_2066_;
v___y_2156_ = v_a_2067_;
v___y_2157_ = v_a_2068_;
v___y_2158_ = v_a_2069_;
v___y_2159_ = v_a_2070_;
v___y_2160_ = v_a_2071_;
goto v___jp_2151_;
}
v___jp_2108_:
{
lean_object* v___x_2124_; lean_object* v___x_2125_; uint8_t v___x_2126_; 
v___x_2124_ = lean_unsigned_to_nat(4u);
v___x_2125_ = l_Lean_Syntax_getArg(v_x_2063_, v___x_2124_);
lean_dec(v_x_2063_);
v___x_2126_ = l_Lean_Syntax_matchesNull(v___x_2125_, v___x_2105_);
if (v___x_2126_ == 0)
{
uint8_t v___x_2127_; 
lean_dec(v_n_x3f_2115_);
lean_dec(v_l_x3f_2114_);
lean_dec(v___y_2110_);
lean_dec(v___y_2109_);
lean_inc(v___x_2107_);
v___x_2127_ = l_Lean_Syntax_matchesNull(v___x_2107_, v___x_2106_);
if (v___x_2127_ == 0)
{
uint8_t v___x_2128_; 
v___x_2128_ = l_Lean_Syntax_matchesNull(v___x_2107_, v___x_2105_);
if (v___x_2128_ == 0)
{
lean_object* v___x_2129_; 
lean_dec(v___y_2113_);
lean_dec(v___y_2112_);
v___x_2129_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2129_;
}
else
{
uint8_t v___x_2130_; 
v___x_2130_ = l_Lean_Syntax_matchesNull(v___y_2112_, v___x_2105_);
if (v___x_2130_ == 0)
{
lean_object* v___x_2131_; 
lean_dec(v___y_2113_);
v___x_2131_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2131_;
}
else
{
uint8_t v___x_2132_; 
lean_inc(v___y_2113_);
v___x_2132_ = l_Lean_Syntax_matchesNull(v___y_2113_, v___y_2111_);
if (v___x_2132_ == 0)
{
lean_object* v___x_2133_; 
lean_dec(v___y_2113_);
v___x_2133_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2133_;
}
else
{
if (v___x_2126_ == 0)
{
lean_object* v___x_2134_; 
lean_dec(v___y_2113_);
v___x_2134_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2134_;
}
else
{
lean_object* v___x_2135_; lean_object* v_args_2136_; 
v___x_2135_ = l_Lean_Syntax_getArg(v___y_2113_, v___x_2106_);
lean_dec(v___y_2113_);
v_args_2136_ = l_Lean_Syntax_getArgs(v___x_2135_);
lean_dec(v___x_2135_);
v_args_2088_ = v_args_2136_;
v___y_2089_ = v___y_2116_;
v___y_2090_ = v___y_2117_;
v___y_2091_ = v___y_2118_;
v___y_2092_ = v___y_2119_;
v___y_2093_ = v___y_2120_;
v___y_2094_ = v___y_2121_;
v___y_2095_ = v___y_2122_;
v___y_2096_ = v___y_2123_;
goto v___jp_2087_;
}
}
}
}
}
else
{
lean_object* v_n_2137_; lean_object* v___x_2138_; uint8_t v___x_2139_; 
v_n_2137_ = l_Lean_Syntax_getArg(v___x_2107_, v___x_2105_);
lean_dec(v___x_2107_);
v___x_2138_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peel___closed__11));
lean_inc(v_n_2137_);
v___x_2139_ = l_Lean_Syntax_isOfKind(v_n_2137_, v___x_2138_);
if (v___x_2139_ == 0)
{
lean_object* v___x_2140_; 
lean_dec(v_n_2137_);
lean_dec(v___y_2113_);
lean_dec(v___y_2112_);
v___x_2140_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2140_;
}
else
{
uint8_t v___x_2141_; 
v___x_2141_ = l_Lean_Syntax_matchesNull(v___y_2112_, v___x_2105_);
if (v___x_2141_ == 0)
{
lean_object* v___x_2142_; 
lean_dec(v_n_2137_);
lean_dec(v___y_2113_);
v___x_2142_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2142_;
}
else
{
uint8_t v___x_2143_; 
v___x_2143_ = l_Lean_Syntax_matchesNull(v___y_2113_, v___x_2105_);
if (v___x_2143_ == 0)
{
lean_object* v___x_2144_; 
lean_dec(v_n_2137_);
v___x_2144_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2144_;
}
else
{
if (v___x_2126_ == 0)
{
lean_object* v___x_2145_; 
lean_dec(v_n_2137_);
v___x_2145_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2145_;
}
else
{
v_n_2074_ = v_n_2137_;
v___y_2075_ = v___y_2116_;
v___y_2076_ = v___y_2117_;
v___y_2077_ = v___y_2118_;
v___y_2078_ = v___y_2119_;
v___y_2079_ = v___y_2120_;
v___y_2080_ = v___y_2121_;
v___y_2081_ = v___y_2122_;
v___y_2082_ = v___y_2123_;
goto v___jp_2073_;
}
}
}
}
}
}
else
{
uint8_t v___x_2146_; lean_object* v___x_2147_; lean_object* v___x_2148_; lean_object* v___f_2149_; lean_object* v___x_2150_; 
lean_dec(v___y_2113_);
lean_dec(v___y_2112_);
lean_dec(v___x_2107_);
v___x_2146_ = 0;
v___x_2147_ = lean_box(v___x_2146_);
v___x_2148_ = lean_box(v___x_2103_);
v___f_2149_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1___lam__0___boxed), 16, 7);
lean_closure_set(v___f_2149_, 0, v___y_2110_);
lean_closure_set(v___f_2149_, 1, v___x_2147_);
lean_closure_set(v___f_2149_, 2, v___y_2109_);
lean_closure_set(v___f_2149_, 3, v___x_2148_);
lean_closure_set(v___f_2149_, 4, v_l_x3f_2114_);
lean_closure_set(v___f_2149_, 5, v___x_2105_);
lean_closure_set(v___f_2149_, 6, v_n_x3f_2115_);
v___x_2150_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_2149_, v___y_2116_, v___y_2117_, v___y_2118_, v___y_2119_, v___y_2120_, v___y_2121_, v___y_2122_, v___y_2123_);
return v___x_2150_;
}
}
v___jp_2151_:
{
lean_object* v___x_2161_; lean_object* v___x_2162_; uint8_t v___x_2163_; 
v___x_2161_ = lean_unsigned_to_nat(2u);
v___x_2162_ = l_Lean_Syntax_getArg(v_x_2063_, v___x_2161_);
lean_inc(v___x_2162_);
v___x_2163_ = l_Lean_Syntax_matchesNull(v___x_2162_, v___x_2106_);
if (v___x_2163_ == 0)
{
lean_object* v___x_2164_; lean_object* v___x_2165_; lean_object* v___x_2166_; lean_object* v___x_2167_; uint8_t v___x_2168_; 
lean_dec(v_num_x3f_2152_);
v___x_2164_ = lean_unsigned_to_nat(3u);
v___x_2165_ = l_Lean_Syntax_getArg(v_x_2063_, v___x_2164_);
v___x_2166_ = lean_unsigned_to_nat(4u);
v___x_2167_ = l_Lean_Syntax_getArg(v_x_2063_, v___x_2166_);
lean_dec(v_x_2063_);
lean_inc(v___x_2107_);
v___x_2168_ = l_Lean_Syntax_matchesNull(v___x_2107_, v___x_2106_);
if (v___x_2168_ == 0)
{
uint8_t v___x_2169_; 
v___x_2169_ = l_Lean_Syntax_matchesNull(v___x_2107_, v___x_2105_);
if (v___x_2169_ == 0)
{
lean_object* v___x_2170_; 
lean_dec(v___x_2167_);
lean_dec(v___x_2165_);
lean_dec(v___x_2162_);
v___x_2170_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2170_;
}
else
{
uint8_t v___x_2171_; 
v___x_2171_ = l_Lean_Syntax_matchesNull(v___x_2162_, v___x_2105_);
if (v___x_2171_ == 0)
{
lean_object* v___x_2172_; 
lean_dec(v___x_2167_);
lean_dec(v___x_2165_);
v___x_2172_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2172_;
}
else
{
uint8_t v___x_2173_; 
lean_inc(v___x_2165_);
v___x_2173_ = l_Lean_Syntax_matchesNull(v___x_2165_, v___x_2161_);
if (v___x_2173_ == 0)
{
lean_object* v___x_2174_; 
lean_dec(v___x_2167_);
lean_dec(v___x_2165_);
v___x_2174_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2174_;
}
else
{
uint8_t v___x_2175_; 
v___x_2175_ = l_Lean_Syntax_matchesNull(v___x_2167_, v___x_2105_);
if (v___x_2175_ == 0)
{
lean_object* v___x_2176_; 
lean_dec(v___x_2165_);
v___x_2176_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2176_;
}
else
{
lean_object* v___x_2177_; lean_object* v_args_2178_; 
v___x_2177_ = l_Lean_Syntax_getArg(v___x_2165_, v___x_2106_);
lean_dec(v___x_2165_);
v_args_2178_ = l_Lean_Syntax_getArgs(v___x_2177_);
lean_dec(v___x_2177_);
v_args_2088_ = v_args_2178_;
v___y_2089_ = v___y_2153_;
v___y_2090_ = v___y_2154_;
v___y_2091_ = v___y_2155_;
v___y_2092_ = v___y_2156_;
v___y_2093_ = v___y_2157_;
v___y_2094_ = v___y_2158_;
v___y_2095_ = v___y_2159_;
v___y_2096_ = v___y_2160_;
goto v___jp_2087_;
}
}
}
}
}
else
{
lean_object* v_n_2179_; lean_object* v___x_2180_; uint8_t v___x_2181_; 
v_n_2179_ = l_Lean_Syntax_getArg(v___x_2107_, v___x_2105_);
lean_dec(v___x_2107_);
v___x_2180_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peel___closed__11));
lean_inc(v_n_2179_);
v___x_2181_ = l_Lean_Syntax_isOfKind(v_n_2179_, v___x_2180_);
if (v___x_2181_ == 0)
{
lean_object* v___x_2182_; 
lean_dec(v_n_2179_);
lean_dec(v___x_2167_);
lean_dec(v___x_2165_);
lean_dec(v___x_2162_);
v___x_2182_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2182_;
}
else
{
uint8_t v___x_2183_; 
v___x_2183_ = l_Lean_Syntax_matchesNull(v___x_2162_, v___x_2105_);
if (v___x_2183_ == 0)
{
lean_object* v___x_2184_; 
lean_dec(v_n_2179_);
lean_dec(v___x_2167_);
lean_dec(v___x_2165_);
v___x_2184_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2184_;
}
else
{
uint8_t v___x_2185_; 
v___x_2185_ = l_Lean_Syntax_matchesNull(v___x_2165_, v___x_2105_);
if (v___x_2185_ == 0)
{
lean_object* v___x_2186_; 
lean_dec(v_n_2179_);
lean_dec(v___x_2167_);
v___x_2186_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2186_;
}
else
{
uint8_t v___x_2187_; 
v___x_2187_ = l_Lean_Syntax_matchesNull(v___x_2167_, v___x_2105_);
if (v___x_2187_ == 0)
{
lean_object* v___x_2188_; 
lean_dec(v_n_2179_);
v___x_2188_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2188_;
}
else
{
v_n_2074_ = v_n_2179_;
v___y_2075_ = v___y_2153_;
v___y_2076_ = v___y_2154_;
v___y_2077_ = v___y_2155_;
v___y_2078_ = v___y_2156_;
v___y_2079_ = v___y_2157_;
v___y_2080_ = v___y_2158_;
v___y_2081_ = v___y_2159_;
v___y_2082_ = v___y_2160_;
goto v___jp_2073_;
}
}
}
}
}
}
else
{
lean_object* v___x_2189_; lean_object* v___x_2190_; lean_object* v___x_2191_; uint8_t v___x_2192_; 
v___x_2189_ = l_Lean_Syntax_getArg(v___x_2162_, v___x_2105_);
v___x_2190_ = lean_unsigned_to_nat(3u);
v___x_2191_ = l_Lean_Syntax_getArg(v_x_2063_, v___x_2190_);
v___x_2192_ = l_Lean_Syntax_isNone(v___x_2191_);
if (v___x_2192_ == 0)
{
uint8_t v___x_2193_; 
lean_inc(v___x_2191_);
v___x_2193_ = l_Lean_Syntax_matchesNull(v___x_2191_, v___x_2161_);
if (v___x_2193_ == 0)
{
lean_object* v___x_2194_; lean_object* v___x_2195_; uint8_t v___x_2196_; 
lean_dec(v___x_2189_);
lean_dec(v_num_x3f_2152_);
v___x_2194_ = lean_unsigned_to_nat(4u);
v___x_2195_ = l_Lean_Syntax_getArg(v_x_2063_, v___x_2194_);
lean_dec(v_x_2063_);
lean_inc(v___x_2107_);
v___x_2196_ = l_Lean_Syntax_matchesNull(v___x_2107_, v___x_2106_);
if (v___x_2196_ == 0)
{
uint8_t v___x_2197_; 
v___x_2197_ = l_Lean_Syntax_matchesNull(v___x_2107_, v___x_2105_);
if (v___x_2197_ == 0)
{
lean_object* v___x_2198_; 
lean_dec(v___x_2195_);
lean_dec(v___x_2191_);
lean_dec(v___x_2162_);
v___x_2198_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2198_;
}
else
{
uint8_t v___x_2199_; 
v___x_2199_ = l_Lean_Syntax_matchesNull(v___x_2162_, v___x_2105_);
if (v___x_2199_ == 0)
{
lean_object* v___x_2200_; 
lean_dec(v___x_2195_);
lean_dec(v___x_2191_);
v___x_2200_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2200_;
}
else
{
if (v___x_2193_ == 0)
{
lean_object* v___x_2201_; 
lean_dec(v___x_2195_);
lean_dec(v___x_2191_);
v___x_2201_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2201_;
}
else
{
uint8_t v___x_2202_; 
v___x_2202_ = l_Lean_Syntax_matchesNull(v___x_2195_, v___x_2105_);
if (v___x_2202_ == 0)
{
lean_object* v___x_2203_; 
lean_dec(v___x_2191_);
v___x_2203_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2203_;
}
else
{
lean_object* v___x_2204_; lean_object* v_args_2205_; 
v___x_2204_ = l_Lean_Syntax_getArg(v___x_2191_, v___x_2106_);
lean_dec(v___x_2191_);
v_args_2205_ = l_Lean_Syntax_getArgs(v___x_2204_);
lean_dec(v___x_2204_);
v_args_2088_ = v_args_2205_;
v___y_2089_ = v___y_2153_;
v___y_2090_ = v___y_2154_;
v___y_2091_ = v___y_2155_;
v___y_2092_ = v___y_2156_;
v___y_2093_ = v___y_2157_;
v___y_2094_ = v___y_2158_;
v___y_2095_ = v___y_2159_;
v___y_2096_ = v___y_2160_;
goto v___jp_2087_;
}
}
}
}
}
else
{
lean_object* v_n_2206_; lean_object* v___x_2207_; uint8_t v___x_2208_; 
v_n_2206_ = l_Lean_Syntax_getArg(v___x_2107_, v___x_2105_);
lean_dec(v___x_2107_);
v___x_2207_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peel___closed__11));
lean_inc(v_n_2206_);
v___x_2208_ = l_Lean_Syntax_isOfKind(v_n_2206_, v___x_2207_);
if (v___x_2208_ == 0)
{
lean_object* v___x_2209_; 
lean_dec(v_n_2206_);
lean_dec(v___x_2195_);
lean_dec(v___x_2191_);
lean_dec(v___x_2162_);
v___x_2209_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2209_;
}
else
{
uint8_t v___x_2210_; 
v___x_2210_ = l_Lean_Syntax_matchesNull(v___x_2162_, v___x_2105_);
if (v___x_2210_ == 0)
{
lean_object* v___x_2211_; 
lean_dec(v_n_2206_);
lean_dec(v___x_2195_);
lean_dec(v___x_2191_);
v___x_2211_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2211_;
}
else
{
uint8_t v___x_2212_; 
v___x_2212_ = l_Lean_Syntax_matchesNull(v___x_2191_, v___x_2105_);
if (v___x_2212_ == 0)
{
lean_object* v___x_2213_; 
lean_dec(v_n_2206_);
lean_dec(v___x_2195_);
v___x_2213_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2213_;
}
else
{
uint8_t v___x_2214_; 
v___x_2214_ = l_Lean_Syntax_matchesNull(v___x_2195_, v___x_2105_);
if (v___x_2214_ == 0)
{
lean_object* v___x_2215_; 
lean_dec(v_n_2206_);
v___x_2215_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2215_;
}
else
{
v_n_2074_ = v_n_2206_;
v___y_2075_ = v___y_2153_;
v___y_2076_ = v___y_2154_;
v___y_2077_ = v___y_2155_;
v___y_2078_ = v___y_2156_;
v___y_2079_ = v___y_2157_;
v___y_2080_ = v___y_2158_;
v___y_2081_ = v___y_2159_;
v___y_2082_ = v___y_2160_;
goto v___jp_2073_;
}
}
}
}
}
}
else
{
lean_object* v___x_2216_; lean_object* v___x_2217_; uint8_t v___x_2218_; 
v___x_2216_ = l_Lean_Syntax_getArg(v___x_2191_, v___x_2106_);
v___x_2217_ = l_Lean_Syntax_getNumArgs(v___x_2216_);
v___x_2218_ = lean_nat_dec_le(v___x_2106_, v___x_2217_);
if (v___x_2218_ == 0)
{
lean_object* v___x_2219_; lean_object* v___x_2220_; uint8_t v___x_2221_; 
lean_dec(v___x_2217_);
lean_dec(v___x_2189_);
lean_dec(v_num_x3f_2152_);
v___x_2219_ = lean_unsigned_to_nat(4u);
v___x_2220_ = l_Lean_Syntax_getArg(v_x_2063_, v___x_2219_);
lean_dec(v_x_2063_);
lean_inc(v___x_2107_);
v___x_2221_ = l_Lean_Syntax_matchesNull(v___x_2107_, v___x_2106_);
if (v___x_2221_ == 0)
{
uint8_t v___x_2222_; 
lean_dec(v___x_2191_);
v___x_2222_ = l_Lean_Syntax_matchesNull(v___x_2107_, v___x_2105_);
if (v___x_2222_ == 0)
{
lean_object* v___x_2223_; 
lean_dec(v___x_2220_);
lean_dec(v___x_2216_);
lean_dec(v___x_2162_);
v___x_2223_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2223_;
}
else
{
uint8_t v___x_2224_; 
v___x_2224_ = l_Lean_Syntax_matchesNull(v___x_2162_, v___x_2105_);
if (v___x_2224_ == 0)
{
lean_object* v___x_2225_; 
lean_dec(v___x_2220_);
lean_dec(v___x_2216_);
v___x_2225_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2225_;
}
else
{
if (v___x_2193_ == 0)
{
lean_object* v___x_2226_; 
lean_dec(v___x_2220_);
lean_dec(v___x_2216_);
v___x_2226_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2226_;
}
else
{
uint8_t v___x_2227_; 
v___x_2227_ = l_Lean_Syntax_matchesNull(v___x_2220_, v___x_2105_);
if (v___x_2227_ == 0)
{
lean_object* v___x_2228_; 
lean_dec(v___x_2216_);
v___x_2228_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2228_;
}
else
{
lean_object* v_args_2229_; 
v_args_2229_ = l_Lean_Syntax_getArgs(v___x_2216_);
lean_dec(v___x_2216_);
v_args_2088_ = v_args_2229_;
v___y_2089_ = v___y_2153_;
v___y_2090_ = v___y_2154_;
v___y_2091_ = v___y_2155_;
v___y_2092_ = v___y_2156_;
v___y_2093_ = v___y_2157_;
v___y_2094_ = v___y_2158_;
v___y_2095_ = v___y_2159_;
v___y_2096_ = v___y_2160_;
goto v___jp_2087_;
}
}
}
}
}
else
{
lean_object* v_n_2230_; lean_object* v___x_2231_; uint8_t v___x_2232_; 
lean_dec(v___x_2216_);
v_n_2230_ = l_Lean_Syntax_getArg(v___x_2107_, v___x_2105_);
lean_dec(v___x_2107_);
v___x_2231_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peel___closed__11));
lean_inc(v_n_2230_);
v___x_2232_ = l_Lean_Syntax_isOfKind(v_n_2230_, v___x_2231_);
if (v___x_2232_ == 0)
{
lean_object* v___x_2233_; 
lean_dec(v_n_2230_);
lean_dec(v___x_2220_);
lean_dec(v___x_2191_);
lean_dec(v___x_2162_);
v___x_2233_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2233_;
}
else
{
uint8_t v___x_2234_; 
v___x_2234_ = l_Lean_Syntax_matchesNull(v___x_2162_, v___x_2105_);
if (v___x_2234_ == 0)
{
lean_object* v___x_2235_; 
lean_dec(v_n_2230_);
lean_dec(v___x_2220_);
lean_dec(v___x_2191_);
v___x_2235_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2235_;
}
else
{
uint8_t v___x_2236_; 
v___x_2236_ = l_Lean_Syntax_matchesNull(v___x_2191_, v___x_2105_);
if (v___x_2236_ == 0)
{
lean_object* v___x_2237_; 
lean_dec(v_n_2230_);
lean_dec(v___x_2220_);
v___x_2237_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2237_;
}
else
{
uint8_t v___x_2238_; 
v___x_2238_ = l_Lean_Syntax_matchesNull(v___x_2220_, v___x_2105_);
if (v___x_2238_ == 0)
{
lean_object* v___x_2239_; 
lean_dec(v_n_2230_);
v___x_2239_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__1___redArg();
return v___x_2239_;
}
else
{
v_n_2074_ = v_n_2230_;
v___y_2075_ = v___y_2153_;
v___y_2076_ = v___y_2154_;
v___y_2077_ = v___y_2155_;
v___y_2078_ = v___y_2156_;
v___y_2079_ = v___y_2157_;
v___y_2080_ = v___y_2158_;
v___y_2081_ = v___y_2159_;
v___y_2082_ = v___y_2160_;
goto v___jp_2073_;
}
}
}
}
}
}
else
{
lean_object* v___x_2240_; lean_object* v___x_2241_; lean_object* v___x_2242_; lean_object* v___x_2243_; lean_object* v___x_2244_; lean_object* v___x_2245_; lean_object* v_n_x3f_2246_; lean_object* v_l_x3f_2247_; lean_object* v___x_2248_; lean_object* v___x_2249_; 
v___x_2240_ = l_Lean_Syntax_getArgs(v___x_2216_);
v___x_2241_ = lean_nat_sub(v___x_2217_, v___x_2106_);
lean_dec(v___x_2217_);
lean_inc(v___x_2241_);
v___x_2242_ = l_Array_extract___redArg(v___x_2240_, v___x_2105_, v___x_2241_);
lean_dec_ref(v___x_2240_);
v___x_2243_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__9));
v___x_2244_ = lean_box(2);
v___x_2245_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2245_, 0, v___x_2244_);
lean_ctor_set(v___x_2245_, 1, v___x_2243_);
lean_ctor_set(v___x_2245_, 2, v___x_2242_);
v_n_x3f_2246_ = l_Lean_Syntax_getArg(v___x_2216_, v___x_2241_);
lean_dec(v___x_2241_);
lean_dec(v___x_2216_);
v_l_x3f_2247_ = l_Lean_Syntax_getArgs(v___x_2245_);
lean_dec_ref_known(v___x_2245_, 3);
v___x_2248_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2248_, 0, v_l_x3f_2247_);
v___x_2249_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2249_, 0, v_n_x3f_2246_);
v___y_2109_ = v_num_x3f_2152_;
v___y_2110_ = v___x_2189_;
v___y_2111_ = v___x_2161_;
v___y_2112_ = v___x_2162_;
v___y_2113_ = v___x_2191_;
v_l_x3f_2114_ = v___x_2248_;
v_n_x3f_2115_ = v___x_2249_;
v___y_2116_ = v___y_2153_;
v___y_2117_ = v___y_2154_;
v___y_2118_ = v___y_2155_;
v___y_2119_ = v___y_2156_;
v___y_2120_ = v___y_2157_;
v___y_2121_ = v___y_2158_;
v___y_2122_ = v___y_2159_;
v___y_2123_ = v___y_2160_;
goto v___jp_2108_;
}
}
}
else
{
lean_object* v___x_2250_; 
v___x_2250_ = lean_box(0);
v___y_2109_ = v_num_x3f_2152_;
v___y_2110_ = v___x_2189_;
v___y_2111_ = v___x_2161_;
v___y_2112_ = v___x_2162_;
v___y_2113_ = v___x_2191_;
v_l_x3f_2114_ = v___x_2250_;
v_n_x3f_2115_ = v___x_2250_;
v___y_2116_ = v___y_2153_;
v___y_2117_ = v___y_2154_;
v___y_2118_ = v___y_2155_;
v___y_2119_ = v___y_2156_;
v___y_2120_ = v___y_2157_;
v___y_2121_ = v___y_2158_;
v___y_2122_ = v___y_2159_;
v___y_2123_ = v___y_2160_;
goto v___jp_2108_;
}
}
}
}
v___jp_2073_:
{
lean_object* v___x_2083_; lean_object* v___x_2084_; lean_object* v___x_2085_; lean_object* v___x_2086_; 
v___x_2083_ = l_Lean_TSyntax_getNat(v_n_2074_);
lean_dec(v_n_2074_);
v___x_2084_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1___closed__1));
v___x_2085_ = l_List_replicateTR___redArg(v___x_2083_, v___x_2084_);
v___x_2086_ = lp_mathlib_Mathlib_Tactic_Peel_peelArgsIff(v___x_2085_, v___y_2075_, v___y_2076_, v___y_2077_, v___y_2078_, v___y_2079_, v___y_2080_, v___y_2081_, v___y_2082_);
return v___x_2086_;
}
v___jp_2087_:
{
size_t v_sz_2097_; size_t v___x_2098_; lean_object* v___x_2099_; lean_object* v___x_2100_; lean_object* v___x_2101_; 
v_sz_2097_ = lean_array_size(v_args_2088_);
v___x_2098_ = ((size_t)0ULL);
v___x_2099_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1_spec__0(v_sz_2097_, v___x_2098_, v_args_2088_);
v___x_2100_ = lean_array_to_list(v___x_2099_);
v___x_2101_ = lp_mathlib_Mathlib_Tactic_Peel_peelArgsIff(v___x_2100_, v___y_2089_, v___y_2090_, v___y_2091_, v___y_2092_, v___y_2093_, v___y_2094_, v___y_2095_, v___y_2096_);
return v___x_2101_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1___boxed(lean_object* v_x_2319_, lean_object* v_a_2320_, lean_object* v_a_2321_, lean_object* v_a_2322_, lean_object* v_a_2323_, lean_object* v_a_2324_, lean_object* v_a_2325_, lean_object* v_a_2326_, lean_object* v_a_2327_, lean_object* v_a_2328_){
_start:
{
lean_object* v_res_2329_; 
v_res_2329_ = lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______elabRules__Mathlib__Tactic__Peel__peel__1(v_x_2319_, v_a_2320_, v_a_2321_, v_a_2322_, v_a_2323_, v_a_2324_, v_a_2325_, v_a_2326_, v_a_2327_);
lean_dec(v_a_2327_);
lean_dec_ref(v_a_2326_);
lean_dec(v_a_2325_);
lean_dec_ref(v_a_2324_);
lean_dec(v_a_2323_);
lean_dec_ref(v_a_2322_);
lean_dec(v_a_2321_);
lean_dec_ref(v_a_2320_);
return v_res_2329_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__8(void){
_start:
{
lean_object* v___x_2346_; 
v___x_2346_ = l_Array_mkArray0(lean_box(0));
return v___x_2346_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1(lean_object* v_x_2347_, lean_object* v_a_2348_, lean_object* v_a_2349_){
_start:
{
lean_object* v___x_2350_; lean_object* v___x_2351_; lean_object* v___x_2352_; lean_object* v___y_2354_; lean_object* v___y_2355_; lean_object* v___y_2356_; lean_object* v___y_2357_; lean_object* v___y_2358_; lean_object* v___y_2359_; lean_object* v___y_2360_; lean_object* v___y_2361_; lean_object* v___y_2362_; lean_object* v___y_2363_; lean_object* v___y_2364_; lean_object* v___y_2365_; uint8_t v___x_2379_; 
v___x_2350_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peel___closed__1));
v___x_2351_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peel___closed__3));
v___x_2352_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peel___closed__4));
lean_inc(v_x_2347_);
v___x_2379_ = l_Lean_Syntax_isOfKind(v_x_2347_, v___x_2352_);
if (v___x_2379_ == 0)
{
lean_object* v___x_2380_; lean_object* v___x_2381_; 
lean_dec(v_x_2347_);
v___x_2380_ = lean_box(1);
v___x_2381_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2381_, 0, v___x_2380_);
lean_ctor_set(v___x_2381_, 1, v_a_2349_);
return v___x_2381_;
}
else
{
lean_object* v___x_2382_; lean_object* v___y_2384_; lean_object* v___y_2385_; lean_object* v___y_2386_; lean_object* v___y_2387_; lean_object* v___y_2388_; lean_object* v___y_2389_; lean_object* v___y_2390_; lean_object* v___y_2391_; lean_object* v___y_2392_; lean_object* v___y_2393_; lean_object* v___y_2394_; lean_object* v___y_2395_; lean_object* v___y_2406_; lean_object* v___y_2407_; lean_object* v___y_2408_; lean_object* v___y_2409_; lean_object* v___y_2410_; lean_object* v___y_2411_; lean_object* v___y_2412_; lean_object* v___y_2413_; lean_object* v___y_2414_; lean_object* v___y_2415_; lean_object* v___y_2416_; lean_object* v___y_2417_; lean_object* v___x_2423_; lean_object* v___y_2425_; lean_object* v___y_2426_; lean_object* v_h_2427_; lean_object* v___y_2428_; lean_object* v___y_2429_; lean_object* v___y_2454_; lean_object* v___y_2455_; lean_object* v_e_2456_; lean_object* v___y_2457_; lean_object* v___y_2458_; lean_object* v_n_2470_; lean_object* v___y_2471_; lean_object* v___y_2472_; lean_object* v___x_2482_; uint8_t v___x_2483_; 
v___x_2382_ = lean_unsigned_to_nat(0u);
v___x_2423_ = lean_unsigned_to_nat(1u);
v___x_2482_ = l_Lean_Syntax_getArg(v_x_2347_, v___x_2423_);
v___x_2483_ = l_Lean_Syntax_isNone(v___x_2482_);
if (v___x_2483_ == 0)
{
uint8_t v___x_2484_; 
lean_inc(v___x_2482_);
v___x_2484_ = l_Lean_Syntax_matchesNull(v___x_2482_, v___x_2423_);
if (v___x_2484_ == 0)
{
lean_object* v___x_2485_; lean_object* v___x_2486_; 
lean_dec(v___x_2482_);
lean_dec(v_x_2347_);
v___x_2485_ = lean_box(1);
v___x_2486_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2486_, 0, v___x_2485_);
lean_ctor_set(v___x_2486_, 1, v_a_2349_);
return v___x_2486_;
}
else
{
lean_object* v_n_2487_; lean_object* v___x_2488_; uint8_t v___x_2489_; 
v_n_2487_ = l_Lean_Syntax_getArg(v___x_2482_, v___x_2382_);
lean_dec(v___x_2482_);
v___x_2488_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peel___closed__11));
lean_inc(v_n_2487_);
v___x_2489_ = l_Lean_Syntax_isOfKind(v_n_2487_, v___x_2488_);
if (v___x_2489_ == 0)
{
lean_object* v___x_2490_; lean_object* v___x_2491_; 
lean_dec(v_n_2487_);
lean_dec(v_x_2347_);
v___x_2490_ = lean_box(1);
v___x_2491_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2491_, 0, v___x_2490_);
lean_ctor_set(v___x_2491_, 1, v_a_2349_);
return v___x_2491_;
}
else
{
lean_object* v___x_2492_; 
v___x_2492_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2492_, 0, v_n_2487_);
v_n_2470_ = v___x_2492_;
v___y_2471_ = v_a_2348_;
v___y_2472_ = v_a_2349_;
goto v___jp_2469_;
}
}
}
else
{
lean_object* v___x_2493_; 
lean_dec(v___x_2482_);
v___x_2493_ = lean_box(0);
v_n_2470_ = v___x_2493_;
v___y_2471_ = v_a_2348_;
v___y_2472_ = v_a_2349_;
goto v___jp_2469_;
}
v___jp_2383_:
{
lean_object* v___x_2396_; lean_object* v___x_2397_; 
lean_inc_ref(v___y_2384_);
v___x_2396_ = l_Array_append___redArg(v___y_2384_, v___y_2395_);
lean_dec_ref(v___y_2395_);
lean_inc(v___y_2385_);
lean_inc(v___y_2387_);
v___x_2397_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2397_, 0, v___y_2387_);
lean_ctor_set(v___x_2397_, 1, v___y_2385_);
lean_ctor_set(v___x_2397_, 2, v___x_2396_);
if (lean_obj_tag(v___y_2393_) == 1)
{
lean_object* v_val_2398_; lean_object* v___x_2399_; lean_object* v___x_2400_; lean_object* v___x_2401_; lean_object* v___x_2402_; lean_object* v___x_2403_; 
v_val_2398_ = lean_ctor_get(v___y_2393_, 0);
lean_inc(v_val_2398_);
lean_dec_ref_known(v___y_2393_, 1);
v___x_2399_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__2));
lean_inc_n(v___y_2387_, 2);
v___x_2400_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2400_, 0, v___y_2387_);
lean_ctor_set(v___x_2400_, 1, v___x_2399_);
lean_inc_ref(v___y_2384_);
v___x_2401_ = l_Array_append___redArg(v___y_2384_, v_val_2398_);
lean_dec(v_val_2398_);
lean_inc(v___y_2385_);
v___x_2402_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2402_, 0, v___y_2387_);
lean_ctor_set(v___x_2402_, 1, v___y_2385_);
lean_ctor_set(v___x_2402_, 2, v___x_2401_);
v___x_2403_ = l_Array_mkArray2___redArg(v___x_2400_, v___x_2402_);
v___y_2354_ = v___y_2384_;
v___y_2355_ = v___y_2385_;
v___y_2356_ = v___x_2397_;
v___y_2357_ = v___y_2386_;
v___y_2358_ = v___y_2387_;
v___y_2359_ = v___y_2389_;
v___y_2360_ = v___y_2388_;
v___y_2361_ = v___y_2390_;
v___y_2362_ = v___y_2391_;
v___y_2363_ = v___y_2392_;
v___y_2364_ = v___y_2394_;
v___y_2365_ = v___x_2403_;
goto v___jp_2353_;
}
else
{
lean_object* v___x_2404_; 
lean_dec(v___y_2393_);
v___x_2404_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__3));
v___y_2354_ = v___y_2384_;
v___y_2355_ = v___y_2385_;
v___y_2356_ = v___x_2397_;
v___y_2357_ = v___y_2386_;
v___y_2358_ = v___y_2387_;
v___y_2359_ = v___y_2389_;
v___y_2360_ = v___y_2388_;
v___y_2361_ = v___y_2390_;
v___y_2362_ = v___y_2391_;
v___y_2363_ = v___y_2392_;
v___y_2364_ = v___y_2394_;
v___y_2365_ = v___x_2404_;
goto v___jp_2353_;
}
}
v___jp_2405_:
{
lean_object* v___x_2418_; lean_object* v___x_2419_; 
lean_inc_ref(v___y_2406_);
v___x_2418_ = l_Array_append___redArg(v___y_2406_, v___y_2417_);
lean_dec_ref(v___y_2417_);
lean_inc(v___y_2407_);
lean_inc(v___y_2409_);
v___x_2419_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2419_, 0, v___y_2409_);
lean_ctor_set(v___x_2419_, 1, v___y_2407_);
lean_ctor_set(v___x_2419_, 2, v___x_2418_);
if (lean_obj_tag(v___y_2414_) == 1)
{
lean_object* v_val_2420_; lean_object* v___x_2421_; 
v_val_2420_ = lean_ctor_get(v___y_2414_, 0);
lean_inc(v_val_2420_);
lean_dec_ref_known(v___y_2414_, 1);
v___x_2421_ = l_Array_mkArray1___redArg(v_val_2420_);
v___y_2384_ = v___y_2406_;
v___y_2385_ = v___y_2407_;
v___y_2386_ = v___y_2408_;
v___y_2387_ = v___y_2409_;
v___y_2388_ = v___y_2411_;
v___y_2389_ = v___y_2410_;
v___y_2390_ = v___y_2412_;
v___y_2391_ = v___y_2413_;
v___y_2392_ = v___x_2419_;
v___y_2393_ = v___y_2416_;
v___y_2394_ = v___y_2415_;
v___y_2395_ = v___x_2421_;
goto v___jp_2383_;
}
else
{
lean_object* v___x_2422_; 
lean_dec(v___y_2414_);
v___x_2422_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__3));
v___y_2384_ = v___y_2406_;
v___y_2385_ = v___y_2407_;
v___y_2386_ = v___y_2408_;
v___y_2387_ = v___y_2409_;
v___y_2388_ = v___y_2411_;
v___y_2389_ = v___y_2410_;
v___y_2390_ = v___y_2412_;
v___y_2391_ = v___y_2413_;
v___y_2392_ = v___x_2419_;
v___y_2393_ = v___y_2416_;
v___y_2394_ = v___y_2415_;
v___y_2395_ = v___x_2422_;
goto v___jp_2383_;
}
}
v___jp_2424_:
{
lean_object* v___x_2430_; lean_object* v___x_2431_; uint8_t v___x_2432_; 
v___x_2430_ = lean_unsigned_to_nat(4u);
v___x_2431_ = l_Lean_Syntax_getArg(v_x_2347_, v___x_2430_);
lean_dec(v_x_2347_);
lean_inc(v___x_2431_);
v___x_2432_ = l_Lean_Syntax_matchesNull(v___x_2431_, v___x_2423_);
if (v___x_2432_ == 0)
{
lean_object* v___x_2433_; lean_object* v___x_2434_; 
lean_dec(v___x_2431_);
lean_dec(v_h_2427_);
lean_dec(v___y_2426_);
lean_dec(v___y_2425_);
v___x_2433_ = lean_box(1);
v___x_2434_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2434_, 0, v___x_2433_);
lean_ctor_set(v___x_2434_, 1, v___y_2429_);
return v___x_2434_;
}
else
{
lean_object* v___x_2435_; lean_object* v___x_2436_; uint8_t v___x_2437_; 
v___x_2435_ = l_Lean_Syntax_getArg(v___x_2431_, v___x_2382_);
lean_dec(v___x_2431_);
v___x_2436_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__5));
lean_inc(v___x_2435_);
v___x_2437_ = l_Lean_Syntax_isOfKind(v___x_2435_, v___x_2436_);
if (v___x_2437_ == 0)
{
lean_object* v___x_2438_; lean_object* v___x_2439_; 
lean_dec(v___x_2435_);
lean_dec(v_h_2427_);
lean_dec(v___y_2426_);
lean_dec(v___y_2425_);
v___x_2438_ = lean_box(1);
v___x_2439_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2439_, 0, v___x_2438_);
lean_ctor_set(v___x_2439_, 1, v___y_2429_);
return v___x_2439_;
}
else
{
lean_object* v_ref_2440_; lean_object* v___x_2441_; uint8_t v___x_2442_; lean_object* v___x_2443_; lean_object* v___x_2444_; lean_object* v___x_2445_; lean_object* v___x_2446_; lean_object* v___x_2447_; lean_object* v___x_2448_; lean_object* v___x_2449_; 
v_ref_2440_ = lean_ctor_get(v___y_2428_, 5);
v___x_2441_ = l_Lean_Syntax_getArg(v___x_2435_, v___x_2423_);
lean_dec(v___x_2435_);
v___x_2442_ = 0;
v___x_2443_ = l_Lean_SourceInfo_fromRef(v_ref_2440_, v___x_2442_);
v___x_2444_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__0));
v___x_2445_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__1));
v___x_2446_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__7));
v___x_2447_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel_peelIffAux___closed__9));
lean_inc(v___x_2443_);
v___x_2448_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2448_, 0, v___x_2443_);
lean_ctor_set(v___x_2448_, 1, v___x_2351_);
v___x_2449_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__8, &lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__8);
if (lean_obj_tag(v___y_2425_) == 1)
{
lean_object* v_val_2450_; lean_object* v___x_2451_; 
v_val_2450_ = lean_ctor_get(v___y_2425_, 0);
lean_inc(v_val_2450_);
lean_dec_ref_known(v___y_2425_, 1);
v___x_2451_ = l_Array_mkArray1___redArg(v_val_2450_);
v___y_2406_ = v___x_2449_;
v___y_2407_ = v___x_2447_;
v___y_2408_ = v___x_2448_;
v___y_2409_ = v___x_2443_;
v___y_2410_ = v___y_2429_;
v___y_2411_ = v___x_2444_;
v___y_2412_ = v___x_2445_;
v___y_2413_ = v___x_2441_;
v___y_2414_ = v___y_2426_;
v___y_2415_ = v___x_2446_;
v___y_2416_ = v_h_2427_;
v___y_2417_ = v___x_2451_;
goto v___jp_2405_;
}
else
{
lean_object* v___x_2452_; 
lean_dec(v___y_2425_);
v___x_2452_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__3));
v___y_2406_ = v___x_2449_;
v___y_2407_ = v___x_2447_;
v___y_2408_ = v___x_2448_;
v___y_2409_ = v___x_2443_;
v___y_2410_ = v___y_2429_;
v___y_2411_ = v___x_2444_;
v___y_2412_ = v___x_2445_;
v___y_2413_ = v___x_2441_;
v___y_2414_ = v___y_2426_;
v___y_2415_ = v___x_2446_;
v___y_2416_ = v_h_2427_;
v___y_2417_ = v___x_2452_;
goto v___jp_2405_;
}
}
}
}
v___jp_2453_:
{
lean_object* v___x_2459_; lean_object* v___x_2460_; uint8_t v___x_2461_; 
v___x_2459_ = lean_unsigned_to_nat(3u);
v___x_2460_ = l_Lean_Syntax_getArg(v_x_2347_, v___x_2459_);
v___x_2461_ = l_Lean_Syntax_isNone(v___x_2460_);
if (v___x_2461_ == 0)
{
uint8_t v___x_2462_; 
lean_inc(v___x_2460_);
v___x_2462_ = l_Lean_Syntax_matchesNull(v___x_2460_, v___y_2455_);
if (v___x_2462_ == 0)
{
lean_object* v___x_2463_; lean_object* v___x_2464_; 
lean_dec(v___x_2460_);
lean_dec(v_e_2456_);
lean_dec(v___y_2454_);
lean_dec(v_x_2347_);
v___x_2463_ = lean_box(1);
v___x_2464_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2464_, 0, v___x_2463_);
lean_ctor_set(v___x_2464_, 1, v___y_2458_);
return v___x_2464_;
}
else
{
lean_object* v___x_2465_; lean_object* v_h_2466_; lean_object* v___x_2467_; 
v___x_2465_ = l_Lean_Syntax_getArg(v___x_2460_, v___x_2423_);
lean_dec(v___x_2460_);
v_h_2466_ = l_Lean_Syntax_getArgs(v___x_2465_);
lean_dec(v___x_2465_);
v___x_2467_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2467_, 0, v_h_2466_);
v___y_2425_ = v___y_2454_;
v___y_2426_ = v_e_2456_;
v_h_2427_ = v___x_2467_;
v___y_2428_ = v___y_2457_;
v___y_2429_ = v___y_2458_;
goto v___jp_2424_;
}
}
else
{
lean_object* v___x_2468_; 
lean_dec(v___x_2460_);
v___x_2468_ = lean_box(0);
v___y_2425_ = v___y_2454_;
v___y_2426_ = v_e_2456_;
v_h_2427_ = v___x_2468_;
v___y_2428_ = v___y_2457_;
v___y_2429_ = v___y_2458_;
goto v___jp_2424_;
}
}
v___jp_2469_:
{
lean_object* v___x_2473_; lean_object* v___x_2474_; uint8_t v___x_2475_; 
v___x_2473_ = lean_unsigned_to_nat(2u);
v___x_2474_ = l_Lean_Syntax_getArg(v_x_2347_, v___x_2473_);
v___x_2475_ = l_Lean_Syntax_isNone(v___x_2474_);
if (v___x_2475_ == 0)
{
uint8_t v___x_2476_; 
lean_inc(v___x_2474_);
v___x_2476_ = l_Lean_Syntax_matchesNull(v___x_2474_, v___x_2423_);
if (v___x_2476_ == 0)
{
lean_object* v___x_2477_; lean_object* v___x_2478_; 
lean_dec(v___x_2474_);
lean_dec(v_n_2470_);
lean_dec(v_x_2347_);
v___x_2477_ = lean_box(1);
v___x_2478_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2478_, 0, v___x_2477_);
lean_ctor_set(v___x_2478_, 1, v___y_2472_);
return v___x_2478_;
}
else
{
lean_object* v_e_2479_; lean_object* v___x_2480_; 
v_e_2479_ = l_Lean_Syntax_getArg(v___x_2474_, v___x_2382_);
lean_dec(v___x_2474_);
v___x_2480_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2480_, 0, v_e_2479_);
v___y_2454_ = v_n_2470_;
v___y_2455_ = v___x_2473_;
v_e_2456_ = v___x_2480_;
v___y_2457_ = v___y_2471_;
v___y_2458_ = v___y_2472_;
goto v___jp_2453_;
}
}
else
{
lean_object* v___x_2481_; 
lean_dec(v___x_2474_);
v___x_2481_ = lean_box(0);
v___y_2454_ = v_n_2470_;
v___y_2455_ = v___x_2473_;
v_e_2456_ = v___x_2481_;
v___y_2457_ = v___y_2471_;
v___y_2458_ = v___y_2472_;
goto v___jp_2453_;
}
}
}
v___jp_2353_:
{
lean_object* v___x_2366_; lean_object* v___x_2367_; lean_object* v___x_2368_; lean_object* v___x_2369_; lean_object* v___x_2370_; lean_object* v___x_2371_; lean_object* v___x_2372_; lean_object* v___x_2373_; lean_object* v___x_2374_; lean_object* v___x_2375_; lean_object* v___x_2376_; lean_object* v___x_2377_; lean_object* v___x_2378_; 
lean_inc_ref_n(v___y_2354_, 2);
v___x_2366_ = l_Array_append___redArg(v___y_2354_, v___y_2365_);
lean_dec_ref(v___y_2365_);
lean_inc_n(v___y_2355_, 3);
lean_inc_n(v___y_2358_, 7);
v___x_2367_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2367_, 0, v___y_2358_);
lean_ctor_set(v___x_2367_, 1, v___y_2355_);
lean_ctor_set(v___x_2367_, 2, v___x_2366_);
v___x_2368_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2368_, 0, v___y_2358_);
lean_ctor_set(v___x_2368_, 1, v___y_2355_);
lean_ctor_set(v___x_2368_, 2, v___y_2354_);
v___x_2369_ = l_Lean_Syntax_node5(v___y_2358_, v___x_2352_, v___y_2357_, v___y_2363_, v___y_2356_, v___x_2367_, v___x_2368_);
v___x_2370_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__0));
v___x_2371_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2371_, 0, v___y_2358_);
lean_ctor_set(v___x_2371_, 1, v___x_2370_);
v___x_2372_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___closed__1));
lean_inc_ref(v___y_2361_);
lean_inc_ref(v___y_2360_);
v___x_2373_ = l_Lean_Name_mkStr4(v___y_2360_, v___y_2361_, v___x_2350_, v___x_2372_);
v___x_2374_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2374_, 0, v___y_2358_);
lean_ctor_set(v___x_2374_, 1, v___x_2372_);
v___x_2375_ = l_Lean_Syntax_node2(v___y_2358_, v___x_2373_, v___x_2374_, v___y_2362_);
v___x_2376_ = l_Lean_Syntax_node3(v___y_2358_, v___y_2355_, v___x_2369_, v___x_2371_, v___x_2375_);
lean_inc(v___y_2364_);
v___x_2377_ = l_Lean_Syntax_node1(v___y_2358_, v___y_2364_, v___x_2376_);
v___x_2378_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2378_, 0, v___x_2377_);
lean_ctor_set(v___x_2378_, 1, v___y_2359_);
return v___x_2378_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1___boxed(lean_object* v_x_2494_, lean_object* v_a_2495_, lean_object* v_a_2496_){
_start:
{
lean_object* v_res_2497_; 
v_res_2497_ = lp_mathlib_Mathlib_Tactic_Peel___aux__Mathlib__Tactic__Peel______macroRules__Mathlib__Tactic__Peel__peel__1(v_x_2494_, v_a_2495_, v_a_2496_);
lean_dec_ref(v_a_2495_);
return v_res_2497_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Filter_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Peel(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Filter_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ToAdditive(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ToDual(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Peel(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ToAdditive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ToDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_Peel_peel = _init_lp_mathlib_Mathlib_Tactic_Peel_peel();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Peel_peel);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Filter_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToAdditive(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ToDual(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Peel(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Filter_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToAdditive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ToDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Peel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Peel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Peel(builtin);
}
#ifdef __cplusplus
}
#endif
