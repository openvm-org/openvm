// Lean compiler output
// Module: Mathlib.Tactic.ErwQuestion
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Elab.Tactic.Rewrite
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
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_rewriteLocalDecl(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_FVarId_getDecl___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvar___override(lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Expr_headBeta(lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFnArgs(lean_object*);
lean_object* lean_array_to_list(lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_appFn_x21(lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
lean_object* l_Lean_LocalDecl_type(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
extern lean_object* l_Lean_checkEmoji;
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
extern lean_object* l_Lean_crossEmoji;
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Name_isPrefixOf(lean_object*, lean_object*);
extern lean_object* l_Lean_diagnostics;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_maxRecDepth;
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
lean_object* l_Lean_Kernel_enableDiag(lean_object*, uint8_t);
uint8_t l_Lean_Kernel_isDiagnosticsEnabled(lean_object*);
lean_object* l_Lean_LocalDecl_toExpr(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_joinSep(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_rewriteTarget(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withLocation(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkOptionalNode(lean_object*);
lean_object* l_Lean_Elab_Tactic_expandOptLocation(lean_object*);
lean_object* l_Lean_Elab_Tactic_withRWRulesSeq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Tactic_location;
extern lean_object* l_Lean_Parser_Tactic_rwRuleSeq;
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Meta_throwTacticEx___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesIdent(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__0_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__0_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__0_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__1_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "erw\?"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__1_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__1_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__2_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "verbose"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__2_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__2_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__3_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__0_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(99, 76, 33, 121, 85, 143, 17, 224)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__3_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__3_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__1_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(37, 195, 34, 156, 42, 195, 118, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__3_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__3_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__2_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(117, 104, 123, 213, 65, 107, 160, 88)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__3_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__3_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__4_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 114, .m_capacity = 114, .m_length = 113, .m_data = "`erw\?` logs more information as it attempts to identify subexpressions which would block the use of `rw` instead."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__4_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__4_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__5_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__4_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__5_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__5_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__6_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__6_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__6_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__7_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__7_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__7_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__8_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Erw\?"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__8_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__8_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__9_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__6_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__9_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__9_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__7_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__9_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__9_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__8_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(144, 208, 161, 3, 139, 125, 7, 82)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__9_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__9_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__0_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(236, 252, 252, 165, 2, 107, 153, 171)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__9_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value_aux_4 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__9_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__1_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(54, 154, 59, 17, 30, 193, 223, 50)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__9_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__9_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value_aux_4),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__2_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(90, 62, 193, 220, 156, 92, 214, 108)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__9_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__9_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_tactic_erw_x3f_verbose;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__6_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__7_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__0_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__8_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(144, 208, 161, 3, 139, 125, 7, 82)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__0_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__1_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(74, 127, 233, 100, 233, 106, 190, 98)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "erw\? "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__3_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__6_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__8;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__9;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__10;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__2_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(107, 17, 151, 162, 143, 207, 214, 14)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "modify"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__7;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(28, 15, 159, 80, 159, 14, 30, 42)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__14_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__14_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__14_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(124, 9, 161, 194, 227, 100, 20, 110)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__16_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__16_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__16_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__16_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__16_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__16_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__17_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__18_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__19_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__20_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__21;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__6_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__22_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__22_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__7_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__22_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__8_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(144, 208, 161, 3, 139, 125, 7, 82)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__22 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__22_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__22_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__23_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__24 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__24_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__25_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__25_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__24_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__25 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__25_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__25_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__26 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__26_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__27 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__27_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__28_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__28_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__28_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__27_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__28_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__7_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(161, 230, 229, 85, 182, 144, 182, 176)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__28 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__28_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__28_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__29 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__29_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__30_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__30_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__27_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__30 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__30_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__30_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__31 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__31_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__32_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__32_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__32_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__32_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__7_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__32 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__32_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__32_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__33 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__33_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__34 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__34_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__34_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__35 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__35_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__35_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__36 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__36_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__33_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__36_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__37 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__37_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__31_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__37_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__38 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__38_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__29_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__38_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__39 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__39_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__26_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__39_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__40 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__40_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__23_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__40_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__41 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__41_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "proj"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__42 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__42_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__43_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__43_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__43_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__43_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__43_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__43_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__42_value),LEAN_SCALAR_PTR_LITERAL(103, 149, 207, 196, 17, 4, 77, 74)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__43 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__43_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cdot"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__44 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__44_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__45_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__45_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__45_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__45_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__45_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__45_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__44_value),LEAN_SCALAR_PTR_LITERAL(215, 94, 65, 66, 49, 100, 151, 85)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__45 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__45_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 1, .m_data = "·"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__46 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__46_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__47 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__47_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "push"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__48 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__48_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__49_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__49;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__48_value),LEAN_SCALAR_PTR_LITERAL(234, 36, 132, 139, 128, 248, 8, 42)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__50 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__50_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "fun"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__51 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__51_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__52_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__52_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__52_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__52_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__52_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__52_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__51_value),LEAN_SCALAR_PTR_LITERAL(249, 155, 133, 242, 71, 132, 191, 97)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__52 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__52_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "basicFun"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__53 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__53_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__54_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__54_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__54_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__54_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__54_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__54_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__53_value),LEAN_SCALAR_PTR_LITERAL(209, 134, 40, 160, 122, 195, 31, 223)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__54 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__54_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hole"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__55 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__55_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__56_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__56_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__56_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__56_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__56_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__56_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__55_value),LEAN_SCALAR_PTR_LITERAL(135, 134, 219, 115, 97, 130, 74, 55)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__56 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__56_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__57 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__57_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__58_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__58;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=>"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__59 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__59_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__60 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__60_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Options_set___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_Options_set___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Options_set___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Options_set___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Options_set___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib_Lean_Options_set___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_Options_set___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__0(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__2___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__0;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = " at reducible transparency,"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "\nand"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "\nare defeq."};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__6_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__1___closed__0;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 41, .m_capacity = 41, .m_length = 40, .m_data = "\nare not both applications or constants."};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__1___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__1___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__1(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 40, .m_capacity = 40, .m_length = 39, .m_data = "\nare not defeq at default transparency."};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__2___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__2___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__2(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__3___closed__0;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "\nare not defeq."};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__3___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__3___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__3___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__3___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__3(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "unsolvedGoals"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "synthPlaceholder"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "lean"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "inductionWithNoAlts"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "_namedError"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3___lam__0___closed__4_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "pp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "analyze"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__0_value),LEAN_SCALAR_PTR_LITERAL(249, 51, 192, 169, 230, 180, 160, 93)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__1_value),LEAN_SCALAR_PTR_LITERAL(219, 114, 68, 229, 251, 70, 44, 204)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "\nare not defeq, but they are at default transparency."};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__5;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__6;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Erw_x3f_extractRewriteEq_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Erw_x3f_extractRewriteEq_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 60, .m_capacity = 60, .m_length = 59, .m_data = "Unexpected term produced by `erw`, head is not an `Eq.mpr`."};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 71, .m_capacity = 71, .m_length = 70, .m_data = "Unexpected term produced by `erw`, not of the form: `Eq.mpr (id _) _`."};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "mpr"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "id"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__4_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 65, .m_capacity = 65, .m_length = 64, .m_data = "Unexpected term produced by `erw`, type hint is not an equality."};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__9;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 69, .m_capacity = 69, .m_length = 68, .m_data = "Unexpected term produced by `erw`, inferred type is not an equality."};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__10_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__11;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Erw_x3f_extractRewriteEq_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Erw_x3f_extractRewriteEq_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 78, .m_capacity = 78, .m_length = 77, .m_data = "Unexpected term produced by `erw at`, head is not an mvar applied to a proof."};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 62, .m_capacity = 62, .m_length = 61, .m_data = "Unexpected term produced by `erw at`, head is not an `Eq.mp`."};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 72, .m_capacity = 72, .m_length = 71, .m_data = "Unexpected term produced by `erw at`, inferred type is not an equality."};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "rewrite"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(109, 67, 55, 19, 78, 216, 184, 166)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 57, .m_capacity = 57, .m_length = 56, .m_data = "did not find instance of the pattern in the current goal"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___closed__4;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__1_spec__1___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__3(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "Expression appearing in "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "Expression from `erw`: "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__4_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__5;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "\n\n"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "tacticTry_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "try"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "withReducible"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "with_reducible"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticRfl"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rfl"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "Expression appearing in target:"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__9;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__3(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__3___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__4(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__4___boxed(lean_object**);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___boxed, .m_arity = 10, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "Debugging `erw\?`"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___closed__1_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__1_spec__1(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__spec__0(lean_object* v_name_1_, lean_object* v_decl_2_, lean_object* v_ref_3_){
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_29_, lean_object* v_decl_30_, lean_object* v_ref_31_, lean_object* v_a_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__spec__0(v_name_29_, v_decl_30_, v_ref_31_);
lean_dec_ref(v_decl_30_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; 
v___x_58_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__3_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4_));
v___x_59_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__5_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4_));
v___x_60_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__9_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4_));
v___x_61_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4__spec__0(v___x_58_, v___x_59_, v___x_60_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4____boxed(lean_object* v_a_62_){
_start:
{
lean_object* v_res_63_; 
v_res_63_ = lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4_();
return v_res_63_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__5(void){
_start:
{
lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; 
v___x_76_ = l_Lean_Parser_Tactic_rwRuleSeq;
v___x_77_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__4));
v___x_78_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__2));
v___x_79_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_79_, 0, v___x_78_);
lean_ctor_set(v___x_79_, 1, v___x_77_);
lean_ctor_set(v___x_79_, 2, v___x_76_);
return v___x_79_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__8(void){
_start:
{
lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; 
v___x_83_ = l_Lean_Parser_Tactic_location;
v___x_84_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__7));
v___x_85_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_85_, 0, v___x_84_);
lean_ctor_set(v___x_85_, 1, v___x_83_);
return v___x_85_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__9(void){
_start:
{
lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; 
v___x_86_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__8, &lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__8);
v___x_87_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__5, &lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__5);
v___x_88_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__2));
v___x_89_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_89_, 0, v___x_88_);
lean_ctor_set(v___x_89_, 1, v___x_87_);
lean_ctor_set(v___x_89_, 2, v___x_86_);
return v___x_89_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__10(void){
_start:
{
lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; 
v___x_90_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__9, &lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__9);
v___x_91_ = lean_unsigned_to_nat(1022u);
v___x_92_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__0));
v___x_93_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_93_, 0, v___x_92_);
lean_ctor_set(v___x_93_, 1, v___x_91_);
lean_ctor_set(v___x_93_, 2, v___x_90_);
return v___x_93_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f(void){
_start:
{
lean_object* v___x_94_; 
v___x_94_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__10, &lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__10);
return v___x_94_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__7(void){
_start:
{
lean_object* v___x_107_; lean_object* v___x_108_; 
v___x_107_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__6));
v___x_108_ = l_String_toRawSubstring_x27(v___x_107_);
return v___x_108_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__21(void){
_start:
{
lean_object* v___x_137_; lean_object* v___x_138_; 
v___x_137_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__20));
v___x_138_ = l_String_toRawSubstring_x27(v___x_137_);
return v___x_138_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__49(void){
_start:
{
lean_object* v___x_206_; lean_object* v___x_207_; 
v___x_206_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__48));
v___x_207_ = l_String_toRawSubstring_x27(v___x_206_);
return v___x_207_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__58(void){
_start:
{
lean_object* v___x_229_; 
v___x_229_ = l_Array_mkArray0(lean_box(0));
return v___x_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1(lean_object* v_x_232_, lean_object* v_a_233_, lean_object* v_a_234_){
_start:
{
lean_object* v___x_235_; uint8_t v___x_236_; 
v___x_235_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__4));
lean_inc(v_x_232_);
v___x_236_ = l_Lean_Syntax_isOfKind(v_x_232_, v___x_235_);
if (v___x_236_ == 0)
{
lean_object* v___x_237_; lean_object* v___x_238_; 
lean_dec(v_x_232_);
v___x_237_ = lean_box(1);
v___x_238_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_238_, 0, v___x_237_);
lean_ctor_set(v___x_238_, 1, v_a_234_);
return v___x_238_;
}
else
{
lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; uint8_t v___x_242_; 
v___x_239_ = lean_unsigned_to_nat(0u);
v___x_240_ = l_Lean_Syntax_getArg(v_x_232_, v___x_239_);
v___x_241_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__5));
v___x_242_ = l_Lean_Syntax_matchesIdent(v___x_240_, v___x_241_);
lean_dec(v___x_240_);
if (v___x_242_ == 0)
{
lean_object* v___x_243_; lean_object* v___x_244_; 
lean_dec(v_x_232_);
v___x_243_ = lean_box(1);
v___x_244_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_244_, 0, v___x_243_);
lean_ctor_set(v___x_244_, 1, v_a_234_);
return v___x_244_;
}
else
{
lean_object* v___x_245_; lean_object* v___x_246_; uint8_t v___x_247_; 
v___x_245_ = lean_unsigned_to_nat(1u);
v___x_246_ = l_Lean_Syntax_getArg(v_x_232_, v___x_245_);
lean_dec(v_x_232_);
lean_inc(v___x_246_);
v___x_247_ = l_Lean_Syntax_matchesNull(v___x_246_, v___x_245_);
if (v___x_247_ == 0)
{
lean_object* v___x_248_; lean_object* v___x_249_; 
lean_dec(v___x_246_);
v___x_248_ = lean_box(1);
v___x_249_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_249_, 0, v___x_248_);
lean_ctor_set(v___x_249_, 1, v_a_234_);
return v___x_249_;
}
else
{
lean_object* v_quotContext_250_; lean_object* v_currMacroScope_251_; lean_object* v_ref_252_; lean_object* v___x_253_; uint8_t v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; 
v_quotContext_250_ = lean_ctor_get(v_a_233_, 1);
v_currMacroScope_251_ = lean_ctor_get(v_a_233_, 2);
v_ref_252_ = lean_ctor_get(v_a_233_, 5);
v___x_253_ = l_Lean_Syntax_getArg(v___x_246_, v___x_239_);
lean_dec(v___x_246_);
v___x_254_ = 0;
v___x_255_ = l_Lean_SourceInfo_fromRef(v_ref_252_, v___x_254_);
v___x_256_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__7, &lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__7);
v___x_257_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__8));
lean_inc_n(v_currMacroScope_251_, 3);
lean_inc_n(v_quotContext_250_, 3);
v___x_258_ = l_Lean_addMacroScope(v_quotContext_250_, v___x_257_, v_currMacroScope_251_);
v___x_259_ = lean_box(0);
v___x_260_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__10));
lean_inc_n(v___x_255_, 23);
v___x_261_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_261_, 0, v___x_255_);
lean_ctor_set(v___x_261_, 1, v___x_256_);
lean_ctor_set(v___x_261_, 2, v___x_258_);
lean_ctor_set(v___x_261_, 3, v___x_260_);
v___x_262_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__12));
v___x_263_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__14));
v___x_264_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__16));
v___x_265_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__17));
v___x_266_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_266_, 0, v___x_255_);
lean_ctor_set(v___x_266_, 1, v___x_265_);
v___x_267_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__19));
v___x_268_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__21, &lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__21_once, _init_lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__21);
v___x_269_ = lean_box(0);
v___x_270_ = l_Lean_addMacroScope(v_quotContext_250_, v___x_269_, v_currMacroScope_251_);
v___x_271_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__41));
v___x_272_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_272_, 0, v___x_255_);
lean_ctor_set(v___x_272_, 1, v___x_268_);
lean_ctor_set(v___x_272_, 2, v___x_270_);
lean_ctor_set(v___x_272_, 3, v___x_271_);
v___x_273_ = l_Lean_Syntax_node1(v___x_255_, v___x_267_, v___x_272_);
lean_inc(v___x_273_);
v___x_274_ = l_Lean_Syntax_node2(v___x_255_, v___x_264_, v___x_266_, v___x_273_);
v___x_275_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__43));
v___x_276_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__45));
v___x_277_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__46));
v___x_278_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_278_, 0, v___x_255_);
lean_ctor_set(v___x_278_, 1, v___x_277_);
v___x_279_ = l_Lean_Syntax_node2(v___x_255_, v___x_276_, v___x_278_, v___x_273_);
v___x_280_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__47));
v___x_281_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_281_, 0, v___x_255_);
lean_ctor_set(v___x_281_, 1, v___x_280_);
v___x_282_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__49, &lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__49_once, _init_lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__49);
v___x_283_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__50));
v___x_284_ = l_Lean_addMacroScope(v_quotContext_250_, v___x_283_, v_currMacroScope_251_);
v___x_285_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_285_, 0, v___x_255_);
lean_ctor_set(v___x_285_, 1, v___x_282_);
lean_ctor_set(v___x_285_, 2, v___x_284_);
lean_ctor_set(v___x_285_, 3, v___x_259_);
v___x_286_ = l_Lean_Syntax_node3(v___x_255_, v___x_275_, v___x_279_, v___x_281_, v___x_285_);
v___x_287_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__51));
v___x_288_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__52));
v___x_289_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_289_, 0, v___x_255_);
lean_ctor_set(v___x_289_, 1, v___x_287_);
v___x_290_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__54));
v___x_291_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__56));
v___x_292_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__57));
v___x_293_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_293_, 0, v___x_255_);
lean_ctor_set(v___x_293_, 1, v___x_292_);
v___x_294_ = l_Lean_Syntax_node1(v___x_255_, v___x_291_, v___x_293_);
v___x_295_ = l_Lean_Syntax_node1(v___x_255_, v___x_262_, v___x_294_);
v___x_296_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__58, &lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__58_once, _init_lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__58);
v___x_297_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_297_, 0, v___x_255_);
lean_ctor_set(v___x_297_, 1, v___x_262_);
lean_ctor_set(v___x_297_, 2, v___x_296_);
v___x_298_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__59));
v___x_299_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_299_, 0, v___x_255_);
lean_ctor_set(v___x_299_, 1, v___x_298_);
v___x_300_ = l_Lean_Syntax_node4(v___x_255_, v___x_290_, v___x_295_, v___x_297_, v___x_299_, v___x_253_);
v___x_301_ = l_Lean_Syntax_node2(v___x_255_, v___x_288_, v___x_289_, v___x_300_);
v___x_302_ = l_Lean_Syntax_node1(v___x_255_, v___x_262_, v___x_301_);
v___x_303_ = l_Lean_Syntax_node2(v___x_255_, v___x_235_, v___x_286_, v___x_302_);
v___x_304_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__60));
v___x_305_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_305_, 0, v___x_255_);
lean_ctor_set(v___x_305_, 1, v___x_304_);
v___x_306_ = l_Lean_Syntax_node3(v___x_255_, v___x_263_, v___x_274_, v___x_303_, v___x_305_);
v___x_307_ = l_Lean_Syntax_node1(v___x_255_, v___x_262_, v___x_306_);
v___x_308_ = l_Lean_Syntax_node2(v___x_255_, v___x_235_, v___x_261_, v___x_307_);
v___x_309_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_309_, 0, v___x_308_);
lean_ctor_set(v___x_309_, 1, v_a_234_);
return v___x_309_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___boxed(lean_object* v_x_310_, lean_object* v_a_311_, lean_object* v_a_312_){
_start:
{
lean_object* v_res_313_; 
v_res_313_ = lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1(v_x_310_, v_a_311_, v_a_312_);
lean_dec_ref(v_a_311_);
return v_res_313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__0(lean_object* v_o_317_, lean_object* v_k_318_, uint8_t v_v_319_){
_start:
{
lean_object* v_map_320_; uint8_t v_hasTrace_321_; lean_object* v___x_323_; uint8_t v_isShared_324_; uint8_t v_isSharedCheck_335_; 
v_map_320_ = lean_ctor_get(v_o_317_, 0);
v_hasTrace_321_ = lean_ctor_get_uint8(v_o_317_, sizeof(void*)*1);
v_isSharedCheck_335_ = !lean_is_exclusive(v_o_317_);
if (v_isSharedCheck_335_ == 0)
{
v___x_323_ = v_o_317_;
v_isShared_324_ = v_isSharedCheck_335_;
goto v_resetjp_322_;
}
else
{
lean_inc(v_map_320_);
lean_dec(v_o_317_);
v___x_323_ = lean_box(0);
v_isShared_324_ = v_isSharedCheck_335_;
goto v_resetjp_322_;
}
v_resetjp_322_:
{
lean_object* v___x_325_; lean_object* v___x_326_; 
v___x_325_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_325_, 0, v_v_319_);
lean_inc(v_k_318_);
v___x_326_ = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(v_k_318_, v___x_325_, v_map_320_);
if (v_hasTrace_321_ == 0)
{
lean_object* v___x_327_; uint8_t v___x_328_; lean_object* v___x_330_; 
v___x_327_ = ((lean_object*)(lp_mathlib_Lean_Options_set___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__0___closed__1));
v___x_328_ = l_Lean_Name_isPrefixOf(v___x_327_, v_k_318_);
lean_dec(v_k_318_);
if (v_isShared_324_ == 0)
{
lean_ctor_set(v___x_323_, 0, v___x_326_);
v___x_330_ = v___x_323_;
goto v_reusejp_329_;
}
else
{
lean_object* v_reuseFailAlloc_331_; 
v_reuseFailAlloc_331_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_331_, 0, v___x_326_);
v___x_330_ = v_reuseFailAlloc_331_;
goto v_reusejp_329_;
}
v_reusejp_329_:
{
lean_ctor_set_uint8(v___x_330_, sizeof(void*)*1, v___x_328_);
return v___x_330_;
}
}
else
{
lean_object* v___x_333_; 
lean_dec(v_k_318_);
if (v_isShared_324_ == 0)
{
lean_ctor_set(v___x_323_, 0, v___x_326_);
v___x_333_ = v___x_323_;
goto v_reusejp_332_;
}
else
{
lean_object* v_reuseFailAlloc_334_; 
v_reuseFailAlloc_334_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_334_, 0, v___x_326_);
lean_ctor_set_uint8(v_reuseFailAlloc_334_, sizeof(void*)*1, v_hasTrace_321_);
v___x_333_ = v_reuseFailAlloc_334_;
goto v_reusejp_332_;
}
v_reusejp_332_:
{
return v___x_333_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__0___boxed(lean_object* v_o_336_, lean_object* v_k_337_, lean_object* v_v_338_){
_start:
{
uint8_t v_v_boxed_339_; lean_object* v_res_340_; 
v_v_boxed_339_ = lean_unbox(v_v_338_);
v_res_340_ = lp_mathlib_Lean_Options_set___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__0(v_o_336_, v_k_337_, v_v_boxed_339_);
return v_res_340_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__1(lean_object* v_opts_341_, lean_object* v_opt_342_){
_start:
{
lean_object* v_name_343_; lean_object* v_defValue_344_; lean_object* v_map_345_; lean_object* v___x_346_; 
v_name_343_ = lean_ctor_get(v_opt_342_, 0);
v_defValue_344_ = lean_ctor_get(v_opt_342_, 1);
v_map_345_ = lean_ctor_get(v_opts_341_, 0);
v___x_346_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_345_, v_name_343_);
if (lean_obj_tag(v___x_346_) == 0)
{
uint8_t v___x_347_; 
v___x_347_ = lean_unbox(v_defValue_344_);
return v___x_347_;
}
else
{
lean_object* v_val_348_; 
v_val_348_ = lean_ctor_get(v___x_346_, 0);
lean_inc(v_val_348_);
lean_dec_ref_known(v___x_346_, 1);
if (lean_obj_tag(v_val_348_) == 1)
{
uint8_t v_v_349_; 
v_v_349_ = lean_ctor_get_uint8(v_val_348_, 0);
lean_dec_ref_known(v_val_348_, 0);
return v_v_349_;
}
else
{
uint8_t v___x_350_; 
lean_dec(v_val_348_);
v___x_350_ = lean_unbox(v_defValue_344_);
return v___x_350_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__1___boxed(lean_object* v_opts_351_, lean_object* v_opt_352_){
_start:
{
uint8_t v_res_353_; lean_object* v_r_354_; 
v_res_353_ = lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__1(v_opts_351_, v_opt_352_);
lean_dec_ref(v_opt_352_);
lean_dec_ref(v_opts_351_);
v_r_354_ = lean_box(v_res_353_);
return v_r_354_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__2(lean_object* v_opts_355_, lean_object* v_opt_356_){
_start:
{
lean_object* v_name_357_; lean_object* v_defValue_358_; lean_object* v_map_359_; lean_object* v___x_360_; 
v_name_357_ = lean_ctor_get(v_opt_356_, 0);
v_defValue_358_ = lean_ctor_get(v_opt_356_, 1);
v_map_359_ = lean_ctor_get(v_opts_355_, 0);
v___x_360_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_359_, v_name_357_);
if (lean_obj_tag(v___x_360_) == 0)
{
lean_inc(v_defValue_358_);
return v_defValue_358_;
}
else
{
lean_object* v_val_361_; 
v_val_361_ = lean_ctor_get(v___x_360_, 0);
lean_inc(v_val_361_);
lean_dec_ref_known(v___x_360_, 1);
if (lean_obj_tag(v_val_361_) == 3)
{
lean_object* v_v_362_; 
v_v_362_ = lean_ctor_get(v_val_361_, 0);
lean_inc(v_v_362_);
lean_dec_ref_known(v_val_361_, 1);
return v_v_362_;
}
else
{
lean_dec(v_val_361_);
lean_inc(v_defValue_358_);
return v_defValue_358_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__2___boxed(lean_object* v_opts_363_, lean_object* v_opt_364_){
_start:
{
lean_object* v_res_365_; 
v_res_365_ = lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__2(v_opts_363_, v_opt_364_);
lean_dec_ref(v_opt_364_);
lean_dec_ref(v_opts_363_);
return v_res_365_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__0(void){
_start:
{
lean_object* v___x_366_; lean_object* v___x_367_; 
v___x_366_ = l_Lean_checkEmoji;
v___x_367_ = l_Lean_stringToMessageData(v___x_366_);
return v___x_367_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__2(void){
_start:
{
lean_object* v___x_369_; lean_object* v___x_370_; 
v___x_369_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__1));
v___x_370_ = l_Lean_stringToMessageData(v___x_369_);
return v___x_370_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__3(void){
_start:
{
lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; 
v___x_371_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__2, &lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__2);
v___x_372_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__0, &lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__0);
v___x_373_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_373_, 0, v___x_372_);
lean_ctor_set(v___x_373_, 1, v___x_371_);
return v___x_373_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__5(void){
_start:
{
lean_object* v___x_375_; lean_object* v___x_376_; 
v___x_375_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__4));
v___x_376_ = l_Lean_stringToMessageData(v___x_375_);
return v___x_376_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__7(void){
_start:
{
lean_object* v___x_378_; lean_object* v___x_379_; 
v___x_378_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__6));
v___x_379_ = l_Lean_stringToMessageData(v___x_378_);
return v___x_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0(lean_object* v_e_u2081_380_, lean_object* v_e_u2082_381_, lean_object* v_x_382_){
_start:
{
lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; 
v___x_383_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__3, &lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__3);
v___x_384_ = l_Lean_MessageData_ofExpr(v_e_u2081_380_);
v___x_385_ = l_Lean_indentD(v___x_384_);
v___x_386_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_386_, 0, v___x_383_);
lean_ctor_set(v___x_386_, 1, v___x_385_);
v___x_387_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__5, &lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__5);
v___x_388_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_388_, 0, v___x_386_);
lean_ctor_set(v___x_388_, 1, v___x_387_);
v___x_389_ = l_Lean_MessageData_ofExpr(v_e_u2082_381_);
v___x_390_ = l_Lean_indentD(v___x_389_);
v___x_391_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_391_, 0, v___x_388_);
lean_ctor_set(v___x_391_, 1, v___x_390_);
v___x_392_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__7, &lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__7);
v___x_393_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_393_, 0, v___x_391_);
lean_ctor_set(v___x_393_, 1, v___x_392_);
return v___x_393_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__1___closed__0(void){
_start:
{
lean_object* v___x_394_; lean_object* v___x_395_; 
v___x_394_ = l_Lean_crossEmoji;
v___x_395_ = l_Lean_stringToMessageData(v___x_394_);
return v___x_395_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__1___closed__2(void){
_start:
{
lean_object* v___x_397_; lean_object* v___x_398_; 
v___x_397_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__1___closed__1));
v___x_398_ = l_Lean_stringToMessageData(v___x_397_);
return v___x_398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__1(lean_object* v_e_u2081_399_, lean_object* v_e_u2082_400_, lean_object* v_x_401_){
_start:
{
lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; 
v___x_402_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__1___closed__0, &lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__1___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__1___closed__0);
v___x_403_ = l_Lean_MessageData_ofExpr(v_e_u2081_399_);
v___x_404_ = l_Lean_indentD(v___x_403_);
v___x_405_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_405_, 0, v___x_402_);
lean_ctor_set(v___x_405_, 1, v___x_404_);
v___x_406_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__5, &lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__5);
v___x_407_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_407_, 0, v___x_405_);
lean_ctor_set(v___x_407_, 1, v___x_406_);
v___x_408_ = l_Lean_MessageData_ofExpr(v_e_u2082_400_);
v___x_409_ = l_Lean_indentD(v___x_408_);
v___x_410_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_410_, 0, v___x_407_);
lean_ctor_set(v___x_410_, 1, v___x_409_);
v___x_411_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__1___closed__2, &lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__1___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__1___closed__2);
v___x_412_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_412_, 0, v___x_410_);
lean_ctor_set(v___x_412_, 1, v___x_411_);
return v___x_412_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__2___closed__1(void){
_start:
{
lean_object* v___x_414_; lean_object* v___x_415_; 
v___x_414_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__2___closed__0));
v___x_415_ = l_Lean_stringToMessageData(v___x_414_);
return v___x_415_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__2(lean_object* v_e_u2081_416_, lean_object* v_e_u2082_417_, lean_object* v_x_418_){
_start:
{
lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; 
v___x_419_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__1___closed__0, &lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__1___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__1___closed__0);
v___x_420_ = l_Lean_MessageData_ofExpr(v_e_u2081_416_);
v___x_421_ = l_Lean_indentD(v___x_420_);
v___x_422_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_422_, 0, v___x_419_);
lean_ctor_set(v___x_422_, 1, v___x_421_);
v___x_423_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__5, &lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__5);
v___x_424_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_424_, 0, v___x_422_);
lean_ctor_set(v___x_424_, 1, v___x_423_);
v___x_425_ = l_Lean_MessageData_ofExpr(v_e_u2082_417_);
v___x_426_ = l_Lean_indentD(v___x_425_);
v___x_427_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_427_, 0, v___x_424_);
lean_ctor_set(v___x_427_, 1, v___x_426_);
v___x_428_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__2___closed__1, &lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__2___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__2___closed__1);
v___x_429_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_429_, 0, v___x_427_);
lean_ctor_set(v___x_429_, 1, v___x_428_);
return v___x_429_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__3___closed__0(void){
_start:
{
lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_432_; 
v___x_430_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__2, &lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__2);
v___x_431_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__1___closed__0, &lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__1___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__1___closed__0);
v___x_432_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_432_, 0, v___x_431_);
lean_ctor_set(v___x_432_, 1, v___x_430_);
return v___x_432_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__3___closed__2(void){
_start:
{
lean_object* v___x_434_; lean_object* v___x_435_; 
v___x_434_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__3___closed__1));
v___x_435_ = l_Lean_stringToMessageData(v___x_434_);
return v___x_435_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__3(lean_object* v_e_u2081_436_, lean_object* v_e_u2082_437_, lean_object* v_x_438_){
_start:
{
lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; 
v___x_439_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__3___closed__0, &lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__3___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__3___closed__0);
v___x_440_ = l_Lean_MessageData_ofExpr(v_e_u2081_436_);
v___x_441_ = l_Lean_indentD(v___x_440_);
v___x_442_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_442_, 0, v___x_439_);
lean_ctor_set(v___x_442_, 1, v___x_441_);
v___x_443_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__5, &lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__5);
v___x_444_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_444_, 0, v___x_442_);
lean_ctor_set(v___x_444_, 1, v___x_443_);
v___x_445_ = l_Lean_MessageData_ofExpr(v_e_u2082_437_);
v___x_446_ = l_Lean_indentD(v___x_445_);
v___x_447_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_447_, 0, v___x_444_);
lean_ctor_set(v___x_447_, 1, v___x_446_);
v___x_448_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__3___closed__2, &lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__3___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__3___closed__2);
v___x_449_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_449_, 0, v___x_447_);
lean_ctor_set(v___x_449_, 1, v___x_448_);
return v___x_449_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3___lam__0(uint8_t v___y_455_, uint8_t v_suppressElabErrors_456_, lean_object* v_x_457_){
_start:
{
if (lean_obj_tag(v_x_457_) == 1)
{
lean_object* v_pre_458_; 
v_pre_458_ = lean_ctor_get(v_x_457_, 0);
switch(lean_obj_tag(v_pre_458_))
{
case 1:
{
lean_object* v_pre_459_; 
v_pre_459_ = lean_ctor_get(v_pre_458_, 0);
switch(lean_obj_tag(v_pre_459_))
{
case 0:
{
lean_object* v_str_460_; lean_object* v_str_461_; lean_object* v___x_462_; uint8_t v___x_463_; 
v_str_460_ = lean_ctor_get(v_x_457_, 1);
v_str_461_ = lean_ctor_get(v_pre_458_, 1);
v___x_462_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__27));
v___x_463_ = lean_string_dec_eq(v_str_461_, v___x_462_);
if (v___x_463_ == 0)
{
lean_object* v___x_464_; uint8_t v___x_465_; 
v___x_464_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__7_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4_));
v___x_465_ = lean_string_dec_eq(v_str_461_, v___x_464_);
if (v___x_465_ == 0)
{
return v___y_455_;
}
else
{
lean_object* v___x_466_; uint8_t v___x_467_; 
v___x_466_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3___lam__0___closed__0));
v___x_467_ = lean_string_dec_eq(v_str_460_, v___x_466_);
if (v___x_467_ == 0)
{
return v___y_455_;
}
else
{
return v_suppressElabErrors_456_;
}
}
}
else
{
lean_object* v___x_468_; uint8_t v___x_469_; 
v___x_468_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3___lam__0___closed__1));
v___x_469_ = lean_string_dec_eq(v_str_460_, v___x_468_);
if (v___x_469_ == 0)
{
return v___y_455_;
}
else
{
return v_suppressElabErrors_456_;
}
}
}
case 1:
{
lean_object* v_pre_470_; 
v_pre_470_ = lean_ctor_get(v_pre_459_, 0);
if (lean_obj_tag(v_pre_470_) == 0)
{
lean_object* v_str_471_; lean_object* v_str_472_; lean_object* v_str_473_; lean_object* v___x_474_; uint8_t v___x_475_; 
v_str_471_ = lean_ctor_get(v_x_457_, 1);
v_str_472_ = lean_ctor_get(v_pre_458_, 1);
v_str_473_ = lean_ctor_get(v_pre_459_, 1);
v___x_474_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3___lam__0___closed__2));
v___x_475_ = lean_string_dec_eq(v_str_473_, v___x_474_);
if (v___x_475_ == 0)
{
return v___y_455_;
}
else
{
lean_object* v___x_476_; uint8_t v___x_477_; 
v___x_476_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3___lam__0___closed__3));
v___x_477_ = lean_string_dec_eq(v_str_472_, v___x_476_);
if (v___x_477_ == 0)
{
return v___y_455_;
}
else
{
lean_object* v___x_478_; uint8_t v___x_479_; 
v___x_478_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3___lam__0___closed__4));
v___x_479_ = lean_string_dec_eq(v_str_471_, v___x_478_);
if (v___x_479_ == 0)
{
return v___y_455_;
}
else
{
return v_suppressElabErrors_456_;
}
}
}
}
else
{
return v___y_455_;
}
}
default: 
{
return v___y_455_;
}
}
}
case 0:
{
lean_object* v_str_480_; lean_object* v___x_481_; uint8_t v___x_482_; 
v_str_480_ = lean_ctor_get(v_x_457_, 1);
v___x_481_ = ((lean_object*)(lp_mathlib_Lean_Options_set___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__0___closed__0));
v___x_482_ = lean_string_dec_eq(v_str_480_, v___x_481_);
if (v___x_482_ == 0)
{
return v___y_455_;
}
else
{
return v_suppressElabErrors_456_;
}
}
default: 
{
return v___y_455_;
}
}
}
else
{
return v___y_455_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3___lam__0___boxed(lean_object* v___y_483_, lean_object* v_suppressElabErrors_484_, lean_object* v_x_485_){
_start:
{
uint8_t v___y_19340__boxed_486_; uint8_t v_suppressElabErrors_boxed_487_; uint8_t v_res_488_; lean_object* v_r_489_; 
v___y_19340__boxed_486_ = lean_unbox(v___y_483_);
v_suppressElabErrors_boxed_487_ = lean_unbox(v_suppressElabErrors_484_);
v_res_488_ = lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3___lam__0(v___y_19340__boxed_486_, v_suppressElabErrors_boxed_487_, v_x_485_);
lean_dec(v_x_485_);
v_r_489_ = lean_box(v_res_488_);
return v_r_489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3_spec__4(lean_object* v_msgData_490_, lean_object* v___y_491_, lean_object* v___y_492_, lean_object* v___y_493_, lean_object* v___y_494_){
_start:
{
lean_object* v___x_496_; lean_object* v_env_497_; lean_object* v___x_498_; lean_object* v_mctx_499_; lean_object* v_lctx_500_; lean_object* v_options_501_; lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; 
v___x_496_ = lean_st_ref_get(v___y_494_);
v_env_497_ = lean_ctor_get(v___x_496_, 0);
lean_inc_ref(v_env_497_);
lean_dec(v___x_496_);
v___x_498_ = lean_st_ref_get(v___y_492_);
v_mctx_499_ = lean_ctor_get(v___x_498_, 0);
lean_inc_ref(v_mctx_499_);
lean_dec(v___x_498_);
v_lctx_500_ = lean_ctor_get(v___y_491_, 2);
v_options_501_ = lean_ctor_get(v___y_493_, 2);
lean_inc_ref(v_options_501_);
lean_inc_ref(v_lctx_500_);
v___x_502_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_502_, 0, v_env_497_);
lean_ctor_set(v___x_502_, 1, v_mctx_499_);
lean_ctor_set(v___x_502_, 2, v_lctx_500_);
lean_ctor_set(v___x_502_, 3, v_options_501_);
v___x_503_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_503_, 0, v___x_502_);
lean_ctor_set(v___x_503_, 1, v_msgData_490_);
v___x_504_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_504_, 0, v___x_503_);
return v___x_504_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3_spec__4___boxed(lean_object* v_msgData_505_, lean_object* v___y_506_, lean_object* v___y_507_, lean_object* v___y_508_, lean_object* v___y_509_, lean_object* v___y_510_){
_start:
{
lean_object* v_res_511_; 
v_res_511_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3_spec__4(v_msgData_505_, v___y_506_, v___y_507_, v___y_508_, v___y_509_);
lean_dec(v___y_509_);
lean_dec_ref(v___y_508_);
lean_dec(v___y_507_);
lean_dec_ref(v___y_506_);
return v_res_511_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3(lean_object* v_ref_512_, lean_object* v_msgData_513_, uint8_t v_severity_514_, uint8_t v_isSilent_515_, lean_object* v___y_516_, lean_object* v___y_517_, lean_object* v___y_518_, lean_object* v___y_519_, lean_object* v___y_520_){
_start:
{
lean_object* v_a_523_; lean_object* v___y_527_; lean_object* v___y_528_; lean_object* v___y_529_; lean_object* v___y_530_; uint8_t v___y_531_; uint8_t v___y_532_; lean_object* v___y_533_; lean_object* v___y_534_; lean_object* v___y_535_; lean_object* v___y_562_; lean_object* v___y_563_; lean_object* v___y_564_; uint8_t v___y_565_; uint8_t v___y_566_; uint8_t v___y_567_; lean_object* v___y_568_; lean_object* v___y_569_; lean_object* v___y_586_; lean_object* v___y_587_; lean_object* v___y_588_; lean_object* v___y_589_; uint8_t v___y_590_; uint8_t v___y_591_; uint8_t v___y_592_; lean_object* v___y_593_; lean_object* v___y_597_; lean_object* v___y_598_; lean_object* v___y_599_; uint8_t v___y_600_; uint8_t v___y_601_; lean_object* v___y_602_; uint8_t v___y_603_; uint8_t v___x_608_; lean_object* v___y_610_; lean_object* v___y_611_; lean_object* v___y_612_; lean_object* v___y_613_; uint8_t v___y_614_; uint8_t v___y_615_; uint8_t v___y_616_; uint8_t v___y_618_; uint8_t v___x_634_; 
v___x_608_ = 2;
v___x_634_ = l_Lean_instBEqMessageSeverity_beq(v_severity_514_, v___x_608_);
if (v___x_634_ == 0)
{
v___y_618_ = v___x_634_;
goto v___jp_617_;
}
else
{
uint8_t v___x_635_; 
lean_inc_ref(v_msgData_513_);
v___x_635_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_513_);
v___y_618_ = v___x_635_;
goto v___jp_617_;
}
v___jp_522_:
{
lean_object* v___x_524_; lean_object* v___x_525_; 
v___x_524_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_524_, 0, v_a_523_);
lean_ctor_set(v___x_524_, 1, v___y_516_);
v___x_525_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_525_, 0, v___x_524_);
return v___x_525_;
}
v___jp_526_:
{
lean_object* v___x_536_; lean_object* v_currNamespace_537_; lean_object* v_openDecls_538_; lean_object* v_env_539_; lean_object* v_nextMacroScope_540_; lean_object* v_ngen_541_; lean_object* v_auxDeclNGen_542_; lean_object* v_traceState_543_; lean_object* v_cache_544_; lean_object* v_messages_545_; lean_object* v_infoState_546_; lean_object* v_snapshotTasks_547_; lean_object* v___x_549_; uint8_t v_isShared_550_; uint8_t v_isSharedCheck_560_; 
v___x_536_ = lean_st_ref_take(v___y_535_);
v_currNamespace_537_ = lean_ctor_get(v___y_534_, 6);
v_openDecls_538_ = lean_ctor_get(v___y_534_, 7);
v_env_539_ = lean_ctor_get(v___x_536_, 0);
v_nextMacroScope_540_ = lean_ctor_get(v___x_536_, 1);
v_ngen_541_ = lean_ctor_get(v___x_536_, 2);
v_auxDeclNGen_542_ = lean_ctor_get(v___x_536_, 3);
v_traceState_543_ = lean_ctor_get(v___x_536_, 4);
v_cache_544_ = lean_ctor_get(v___x_536_, 5);
v_messages_545_ = lean_ctor_get(v___x_536_, 6);
v_infoState_546_ = lean_ctor_get(v___x_536_, 7);
v_snapshotTasks_547_ = lean_ctor_get(v___x_536_, 8);
v_isSharedCheck_560_ = !lean_is_exclusive(v___x_536_);
if (v_isSharedCheck_560_ == 0)
{
v___x_549_ = v___x_536_;
v_isShared_550_ = v_isSharedCheck_560_;
goto v_resetjp_548_;
}
else
{
lean_inc(v_snapshotTasks_547_);
lean_inc(v_infoState_546_);
lean_inc(v_messages_545_);
lean_inc(v_cache_544_);
lean_inc(v_traceState_543_);
lean_inc(v_auxDeclNGen_542_);
lean_inc(v_ngen_541_);
lean_inc(v_nextMacroScope_540_);
lean_inc(v_env_539_);
lean_dec(v___x_536_);
v___x_549_ = lean_box(0);
v_isShared_550_ = v_isSharedCheck_560_;
goto v_resetjp_548_;
}
v_resetjp_548_:
{
lean_object* v___x_551_; lean_object* v___x_552_; lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_556_; 
lean_inc(v_openDecls_538_);
lean_inc(v_currNamespace_537_);
v___x_551_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_551_, 0, v_currNamespace_537_);
lean_ctor_set(v___x_551_, 1, v_openDecls_538_);
v___x_552_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_552_, 0, v___x_551_);
lean_ctor_set(v___x_552_, 1, v___y_529_);
lean_inc_ref(v___y_528_);
lean_inc_ref(v___y_527_);
v___x_553_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_553_, 0, v___y_527_);
lean_ctor_set(v___x_553_, 1, v___y_533_);
lean_ctor_set(v___x_553_, 2, v___y_530_);
lean_ctor_set(v___x_553_, 3, v___y_528_);
lean_ctor_set(v___x_553_, 4, v___x_552_);
lean_ctor_set_uint8(v___x_553_, sizeof(void*)*5, v___y_531_);
lean_ctor_set_uint8(v___x_553_, sizeof(void*)*5 + 1, v___y_532_);
lean_ctor_set_uint8(v___x_553_, sizeof(void*)*5 + 2, v_isSilent_515_);
v___x_554_ = l_Lean_MessageLog_add(v___x_553_, v_messages_545_);
if (v_isShared_550_ == 0)
{
lean_ctor_set(v___x_549_, 6, v___x_554_);
v___x_556_ = v___x_549_;
goto v_reusejp_555_;
}
else
{
lean_object* v_reuseFailAlloc_559_; 
v_reuseFailAlloc_559_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_559_, 0, v_env_539_);
lean_ctor_set(v_reuseFailAlloc_559_, 1, v_nextMacroScope_540_);
lean_ctor_set(v_reuseFailAlloc_559_, 2, v_ngen_541_);
lean_ctor_set(v_reuseFailAlloc_559_, 3, v_auxDeclNGen_542_);
lean_ctor_set(v_reuseFailAlloc_559_, 4, v_traceState_543_);
lean_ctor_set(v_reuseFailAlloc_559_, 5, v_cache_544_);
lean_ctor_set(v_reuseFailAlloc_559_, 6, v___x_554_);
lean_ctor_set(v_reuseFailAlloc_559_, 7, v_infoState_546_);
lean_ctor_set(v_reuseFailAlloc_559_, 8, v_snapshotTasks_547_);
v___x_556_ = v_reuseFailAlloc_559_;
goto v_reusejp_555_;
}
v_reusejp_555_:
{
lean_object* v___x_557_; lean_object* v___x_558_; 
v___x_557_ = lean_st_ref_set(v___y_535_, v___x_556_);
v___x_558_ = lean_box(0);
v_a_523_ = v___x_558_;
goto v___jp_522_;
}
}
}
v___jp_561_:
{
lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v_a_572_; lean_object* v___x_574_; uint8_t v_isShared_575_; uint8_t v_isSharedCheck_584_; 
v___x_570_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_513_);
v___x_571_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3_spec__4(v___x_570_, v___y_517_, v___y_518_, v___y_519_, v___y_520_);
v_a_572_ = lean_ctor_get(v___x_571_, 0);
v_isSharedCheck_584_ = !lean_is_exclusive(v___x_571_);
if (v_isSharedCheck_584_ == 0)
{
v___x_574_ = v___x_571_;
v_isShared_575_ = v_isSharedCheck_584_;
goto v_resetjp_573_;
}
else
{
lean_inc(v_a_572_);
lean_dec(v___x_571_);
v___x_574_ = lean_box(0);
v_isShared_575_ = v_isSharedCheck_584_;
goto v_resetjp_573_;
}
v_resetjp_573_:
{
lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_579_; 
lean_inc_ref_n(v___y_564_, 2);
v___x_576_ = l_Lean_FileMap_toPosition(v___y_564_, v___y_568_);
lean_dec(v___y_568_);
v___x_577_ = l_Lean_FileMap_toPosition(v___y_564_, v___y_569_);
lean_dec(v___y_569_);
if (v_isShared_575_ == 0)
{
lean_ctor_set_tag(v___x_574_, 1);
lean_ctor_set(v___x_574_, 0, v___x_577_);
v___x_579_ = v___x_574_;
goto v_reusejp_578_;
}
else
{
lean_object* v_reuseFailAlloc_583_; 
v_reuseFailAlloc_583_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_583_, 0, v___x_577_);
v___x_579_ = v_reuseFailAlloc_583_;
goto v_reusejp_578_;
}
v_reusejp_578_:
{
lean_object* v___x_580_; 
v___x_580_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__20));
if (v___y_567_ == 0)
{
lean_dec_ref(v___y_562_);
v___y_527_ = v___y_563_;
v___y_528_ = v___x_580_;
v___y_529_ = v_a_572_;
v___y_530_ = v___x_579_;
v___y_531_ = v___y_565_;
v___y_532_ = v___y_566_;
v___y_533_ = v___x_576_;
v___y_534_ = v___y_519_;
v___y_535_ = v___y_520_;
goto v___jp_526_;
}
else
{
uint8_t v___x_581_; 
lean_inc(v_a_572_);
v___x_581_ = l_Lean_MessageData_hasTag(v___y_562_, v_a_572_);
if (v___x_581_ == 0)
{
lean_object* v___x_582_; 
lean_dec_ref(v___x_579_);
lean_dec_ref(v___x_576_);
lean_dec(v_a_572_);
v___x_582_ = lean_box(0);
v_a_523_ = v___x_582_;
goto v___jp_522_;
}
else
{
v___y_527_ = v___y_563_;
v___y_528_ = v___x_580_;
v___y_529_ = v_a_572_;
v___y_530_ = v___x_579_;
v___y_531_ = v___y_565_;
v___y_532_ = v___y_566_;
v___y_533_ = v___x_576_;
v___y_534_ = v___y_519_;
v___y_535_ = v___y_520_;
goto v___jp_526_;
}
}
}
}
}
v___jp_585_:
{
lean_object* v___x_594_; 
v___x_594_ = l_Lean_Syntax_getTailPos_x3f(v___y_588_, v___y_590_);
lean_dec(v___y_588_);
if (lean_obj_tag(v___x_594_) == 0)
{
lean_inc(v___y_593_);
v___y_562_ = v___y_586_;
v___y_563_ = v___y_587_;
v___y_564_ = v___y_589_;
v___y_565_ = v___y_590_;
v___y_566_ = v___y_591_;
v___y_567_ = v___y_592_;
v___y_568_ = v___y_593_;
v___y_569_ = v___y_593_;
goto v___jp_561_;
}
else
{
lean_object* v_val_595_; 
v_val_595_ = lean_ctor_get(v___x_594_, 0);
lean_inc(v_val_595_);
lean_dec_ref_known(v___x_594_, 1);
v___y_562_ = v___y_586_;
v___y_563_ = v___y_587_;
v___y_564_ = v___y_589_;
v___y_565_ = v___y_590_;
v___y_566_ = v___y_591_;
v___y_567_ = v___y_592_;
v___y_568_ = v___y_593_;
v___y_569_ = v_val_595_;
goto v___jp_561_;
}
}
v___jp_596_:
{
lean_object* v_ref_604_; lean_object* v___x_605_; 
v_ref_604_ = l_Lean_replaceRef(v_ref_512_, v___y_602_);
v___x_605_ = l_Lean_Syntax_getPos_x3f(v_ref_604_, v___y_600_);
if (lean_obj_tag(v___x_605_) == 0)
{
lean_object* v___x_606_; 
v___x_606_ = lean_unsigned_to_nat(0u);
v___y_586_ = v___y_597_;
v___y_587_ = v___y_598_;
v___y_588_ = v_ref_604_;
v___y_589_ = v___y_599_;
v___y_590_ = v___y_600_;
v___y_591_ = v___y_603_;
v___y_592_ = v___y_601_;
v___y_593_ = v___x_606_;
goto v___jp_585_;
}
else
{
lean_object* v_val_607_; 
v_val_607_ = lean_ctor_get(v___x_605_, 0);
lean_inc(v_val_607_);
lean_dec_ref_known(v___x_605_, 1);
v___y_586_ = v___y_597_;
v___y_587_ = v___y_598_;
v___y_588_ = v_ref_604_;
v___y_589_ = v___y_599_;
v___y_590_ = v___y_600_;
v___y_591_ = v___y_603_;
v___y_592_ = v___y_601_;
v___y_593_ = v_val_607_;
goto v___jp_585_;
}
}
v___jp_609_:
{
if (v___y_616_ == 0)
{
v___y_597_ = v___y_610_;
v___y_598_ = v___y_611_;
v___y_599_ = v___y_612_;
v___y_600_ = v___y_615_;
v___y_601_ = v___y_614_;
v___y_602_ = v___y_613_;
v___y_603_ = v_severity_514_;
goto v___jp_596_;
}
else
{
v___y_597_ = v___y_610_;
v___y_598_ = v___y_611_;
v___y_599_ = v___y_612_;
v___y_600_ = v___y_615_;
v___y_601_ = v___y_614_;
v___y_602_ = v___y_613_;
v___y_603_ = v___x_608_;
goto v___jp_596_;
}
}
v___jp_617_:
{
if (v___y_618_ == 0)
{
lean_object* v_fileName_619_; lean_object* v_fileMap_620_; lean_object* v_options_621_; lean_object* v_ref_622_; uint8_t v_suppressElabErrors_623_; lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___f_626_; uint8_t v___x_627_; uint8_t v___x_628_; 
v_fileName_619_ = lean_ctor_get(v___y_519_, 0);
v_fileMap_620_ = lean_ctor_get(v___y_519_, 1);
v_options_621_ = lean_ctor_get(v___y_519_, 2);
v_ref_622_ = lean_ctor_get(v___y_519_, 5);
v_suppressElabErrors_623_ = lean_ctor_get_uint8(v___y_519_, sizeof(void*)*14 + 1);
v___x_624_ = lean_box(v___y_618_);
v___x_625_ = lean_box(v_suppressElabErrors_623_);
v___f_626_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3___lam__0___boxed), 3, 2);
lean_closure_set(v___f_626_, 0, v___x_624_);
lean_closure_set(v___f_626_, 1, v___x_625_);
v___x_627_ = 1;
v___x_628_ = l_Lean_instBEqMessageSeverity_beq(v_severity_514_, v___x_627_);
if (v___x_628_ == 0)
{
v___y_610_ = v___f_626_;
v___y_611_ = v_fileName_619_;
v___y_612_ = v_fileMap_620_;
v___y_613_ = v_ref_622_;
v___y_614_ = v_suppressElabErrors_623_;
v___y_615_ = v___y_618_;
v___y_616_ = v___x_628_;
goto v___jp_609_;
}
else
{
lean_object* v___x_629_; uint8_t v___x_630_; 
v___x_629_ = l_Lean_warningAsError;
v___x_630_ = lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__1(v_options_621_, v___x_629_);
v___y_610_ = v___f_626_;
v___y_611_ = v_fileName_619_;
v___y_612_ = v_fileMap_620_;
v___y_613_ = v_ref_622_;
v___y_614_ = v_suppressElabErrors_623_;
v___y_615_ = v___y_618_;
v___y_616_ = v___x_630_;
goto v___jp_609_;
}
}
else
{
lean_object* v___x_631_; lean_object* v___x_632_; lean_object* v___x_633_; 
lean_dec_ref(v_msgData_513_);
v___x_631_ = lean_box(0);
v___x_632_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_632_, 0, v___x_631_);
lean_ctor_set(v___x_632_, 1, v___y_516_);
v___x_633_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_633_, 0, v___x_632_);
return v___x_633_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3___boxed(lean_object* v_ref_636_, lean_object* v_msgData_637_, lean_object* v_severity_638_, lean_object* v_isSilent_639_, lean_object* v___y_640_, lean_object* v___y_641_, lean_object* v___y_642_, lean_object* v___y_643_, lean_object* v___y_644_, lean_object* v___y_645_){
_start:
{
uint8_t v_severity_boxed_646_; uint8_t v_isSilent_boxed_647_; lean_object* v_res_648_; 
v_severity_boxed_646_ = lean_unbox(v_severity_638_);
v_isSilent_boxed_647_ = lean_unbox(v_isSilent_639_);
v_res_648_ = lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3(v_ref_636_, v_msgData_637_, v_severity_boxed_646_, v_isSilent_boxed_647_, v___y_640_, v___y_641_, v___y_642_, v___y_643_, v___y_644_);
lean_dec(v___y_644_);
lean_dec_ref(v___y_643_);
lean_dec(v___y_642_);
lean_dec_ref(v___y_641_);
lean_dec(v_ref_636_);
return v_res_648_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3(lean_object* v_ref_649_, lean_object* v_msgData_650_, lean_object* v___y_651_, lean_object* v___y_652_, lean_object* v___y_653_, lean_object* v___y_654_, lean_object* v___y_655_){
_start:
{
uint8_t v___x_657_; uint8_t v___x_658_; lean_object* v___x_659_; 
v___x_657_ = 0;
v___x_658_ = 0;
v___x_659_ = lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3(v_ref_649_, v_msgData_650_, v___x_657_, v___x_658_, v___y_651_, v___y_652_, v___y_653_, v___y_654_, v___y_655_);
return v___x_659_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3___boxed(lean_object* v_ref_660_, lean_object* v_msgData_661_, lean_object* v___y_662_, lean_object* v___y_663_, lean_object* v___y_664_, lean_object* v___y_665_, lean_object* v___y_666_, lean_object* v___y_667_){
_start:
{
lean_object* v_res_668_; 
v_res_668_ = lp_mathlib_Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3(v_ref_660_, v_msgData_661_, v___y_662_, v___y_663_, v___y_664_, v___y_665_, v___y_666_);
lean_dec(v___y_666_);
lean_dec_ref(v___y_665_);
lean_dec(v___y_664_);
lean_dec_ref(v___y_663_);
lean_dec(v_ref_660_);
return v_res_668_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__4(void){
_start:
{
lean_object* v___x_675_; lean_object* v___x_676_; 
v___x_675_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__3));
v___x_676_ = l_Lean_stringToMessageData(v___x_675_);
return v___x_676_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__5(void){
_start:
{
lean_object* v___x_677_; 
v___x_677_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_677_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__6(void){
_start:
{
lean_object* v___x_678_; lean_object* v___x_679_; 
v___x_678_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__5, &lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__5);
v___x_679_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_679_, 0, v___x_678_);
return v___x_679_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__7(void){
_start:
{
lean_object* v___x_680_; lean_object* v___x_681_; 
v___x_680_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__6, &lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__6);
v___x_681_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_681_, 0, v___x_680_);
lean_ctor_set(v___x_681_, 1, v___x_680_);
return v___x_681_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs(lean_object* v_tk_682_, lean_object* v_e_u2081_683_, lean_object* v_e_u2082_684_, lean_object* v_a_685_, lean_object* v_a_686_, lean_object* v_a_687_, lean_object* v_a_688_, lean_object* v_a_689_){
_start:
{
lean_object* v___x_691_; lean_object* v_fileName_692_; lean_object* v_fileMap_693_; lean_object* v_options_694_; lean_object* v_currRecDepth_695_; lean_object* v_ref_696_; lean_object* v_currNamespace_697_; lean_object* v_openDecls_698_; lean_object* v_initHeartbeats_699_; lean_object* v_maxHeartbeats_700_; lean_object* v_quotContext_701_; lean_object* v_currMacroScope_702_; lean_object* v_cancelTk_x3f_703_; uint8_t v_suppressElabErrors_704_; lean_object* v_inheritedTraceOptions_705_; lean_object* v_env_706_; lean_object* v___f_707_; lean_object* v___f_708_; uint8_t v___y_710_; lean_object* v___y_711_; lean_object* v___f_716_; lean_object* v___f_717_; uint8_t v___x_718_; lean_object* v___x_719_; uint8_t v___x_720_; lean_object* v___x_721_; lean_object* v___x_722_; uint8_t v___x_723_; lean_object* v_fileName_725_; lean_object* v_fileMap_726_; lean_object* v_currRecDepth_727_; lean_object* v_ref_728_; lean_object* v_currNamespace_729_; lean_object* v_openDecls_730_; lean_object* v_initHeartbeats_731_; lean_object* v_maxHeartbeats_732_; lean_object* v_quotContext_733_; lean_object* v_currMacroScope_734_; lean_object* v_cancelTk_x3f_735_; uint8_t v_suppressElabErrors_736_; lean_object* v_inheritedTraceOptions_737_; lean_object* v___y_738_; uint8_t v___y_922_; uint8_t v___x_943_; 
v___x_691_ = lean_st_ref_get(v_a_689_);
v_fileName_692_ = lean_ctor_get(v_a_688_, 0);
v_fileMap_693_ = lean_ctor_get(v_a_688_, 1);
v_options_694_ = lean_ctor_get(v_a_688_, 2);
v_currRecDepth_695_ = lean_ctor_get(v_a_688_, 3);
v_ref_696_ = lean_ctor_get(v_a_688_, 5);
v_currNamespace_697_ = lean_ctor_get(v_a_688_, 6);
v_openDecls_698_ = lean_ctor_get(v_a_688_, 7);
v_initHeartbeats_699_ = lean_ctor_get(v_a_688_, 8);
v_maxHeartbeats_700_ = lean_ctor_get(v_a_688_, 9);
v_quotContext_701_ = lean_ctor_get(v_a_688_, 10);
v_currMacroScope_702_ = lean_ctor_get(v_a_688_, 11);
v_cancelTk_x3f_703_ = lean_ctor_get(v_a_688_, 12);
v_suppressElabErrors_704_ = lean_ctor_get_uint8(v_a_688_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_705_ = lean_ctor_get(v_a_688_, 13);
v_env_706_ = lean_ctor_get(v___x_691_, 0);
lean_inc_ref(v_env_706_);
lean_dec(v___x_691_);
lean_inc_ref_n(v_e_u2082_684_, 4);
lean_inc_ref_n(v_e_u2081_683_, 4);
v___f_707_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0), 3, 2);
lean_closure_set(v___f_707_, 0, v_e_u2081_683_);
lean_closure_set(v___f_707_, 1, v_e_u2082_684_);
v___f_708_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__1), 3, 2);
lean_closure_set(v___f_708_, 0, v_e_u2081_683_);
lean_closure_set(v___f_708_, 1, v_e_u2082_684_);
v___f_716_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__2), 3, 2);
lean_closure_set(v___f_716_, 0, v_e_u2081_683_);
lean_closure_set(v___f_716_, 1, v_e_u2082_684_);
v___f_717_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__3), 3, 2);
lean_closure_set(v___f_717_, 0, v_e_u2081_683_);
lean_closure_set(v___f_717_, 1, v_e_u2082_684_);
v___x_718_ = 2;
v___x_719_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__2));
v___x_720_ = 1;
lean_inc_ref(v_options_694_);
v___x_721_ = lp_mathlib_Lean_Options_set___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__0(v_options_694_, v___x_719_, v___x_720_);
v___x_722_ = l_Lean_diagnostics;
v___x_723_ = lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__1(v___x_721_, v___x_722_);
v___x_943_ = l_Lean_Kernel_isDiagnosticsEnabled(v_env_706_);
lean_dec_ref(v_env_706_);
if (v___x_943_ == 0)
{
if (v___x_723_ == 0)
{
v_fileName_725_ = v_fileName_692_;
v_fileMap_726_ = v_fileMap_693_;
v_currRecDepth_727_ = v_currRecDepth_695_;
v_ref_728_ = v_ref_696_;
v_currNamespace_729_ = v_currNamespace_697_;
v_openDecls_730_ = v_openDecls_698_;
v_initHeartbeats_731_ = v_initHeartbeats_699_;
v_maxHeartbeats_732_ = v_maxHeartbeats_700_;
v_quotContext_733_ = v_quotContext_701_;
v_currMacroScope_734_ = v_currMacroScope_702_;
v_cancelTk_x3f_735_ = v_cancelTk_x3f_703_;
v_suppressElabErrors_736_ = v_suppressElabErrors_704_;
v_inheritedTraceOptions_737_ = v_inheritedTraceOptions_705_;
v___y_738_ = v_a_689_;
goto v___jp_724_;
}
else
{
v___y_922_ = v___x_943_;
goto v___jp_921_;
}
}
else
{
v___y_922_ = v___x_723_;
goto v___jp_921_;
}
v___jp_709_:
{
lean_object* v___x_712_; lean_object* v___x_713_; lean_object* v___x_714_; lean_object* v___x_715_; 
v___x_712_ = lean_array_push(v___y_711_, v___f_708_);
v___x_713_ = lean_box(v___y_710_);
v___x_714_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_714_, 0, v___x_713_);
lean_ctor_set(v___x_714_, 1, v___x_712_);
v___x_715_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_715_, 0, v___x_714_);
return v___x_715_;
}
v___jp_724_:
{
lean_object* v_keyedConfig_739_; uint8_t v_trackZetaDelta_740_; lean_object* v_zetaDeltaSet_741_; lean_object* v_lctx_742_; lean_object* v_localInstances_743_; lean_object* v_defEqCtx_x3f_744_; lean_object* v_synthPendingDepth_745_; lean_object* v_customCanUnfoldPredicate_x3f_746_; uint8_t v_univApprox_747_; uint8_t v_inTypeClassResolution_748_; uint8_t v_cacheInferType_749_; lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v___x_752_; lean_object* v___x_753_; lean_object* v___x_754_; lean_object* v___x_755_; 
v_keyedConfig_739_ = lean_ctor_get(v_a_686_, 0);
v_trackZetaDelta_740_ = lean_ctor_get_uint8(v_a_686_, sizeof(void*)*7);
v_zetaDeltaSet_741_ = lean_ctor_get(v_a_686_, 1);
v_lctx_742_ = lean_ctor_get(v_a_686_, 2);
v_localInstances_743_ = lean_ctor_get(v_a_686_, 3);
v_defEqCtx_x3f_744_ = lean_ctor_get(v_a_686_, 4);
v_synthPendingDepth_745_ = lean_ctor_get(v_a_686_, 5);
v_customCanUnfoldPredicate_x3f_746_ = lean_ctor_get(v_a_686_, 6);
v_univApprox_747_ = lean_ctor_get_uint8(v_a_686_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_748_ = lean_ctor_get_uint8(v_a_686_, sizeof(void*)*7 + 2);
v_cacheInferType_749_ = lean_ctor_get_uint8(v_a_686_, sizeof(void*)*7 + 3);
v___x_750_ = l_Lean_maxRecDepth;
v___x_751_ = lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__2(v___x_721_, v___x_750_);
lean_inc_ref(v_inheritedTraceOptions_737_);
lean_inc(v_cancelTk_x3f_735_);
lean_inc(v_currMacroScope_734_);
lean_inc(v_quotContext_733_);
lean_inc(v_maxHeartbeats_732_);
lean_inc(v_initHeartbeats_731_);
lean_inc(v_openDecls_730_);
lean_inc(v_currNamespace_729_);
lean_inc(v_ref_728_);
lean_inc(v_currRecDepth_727_);
lean_inc_ref(v_fileMap_726_);
lean_inc_ref(v_fileName_725_);
v___x_752_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_752_, 0, v_fileName_725_);
lean_ctor_set(v___x_752_, 1, v_fileMap_726_);
lean_ctor_set(v___x_752_, 2, v___x_721_);
lean_ctor_set(v___x_752_, 3, v_currRecDepth_727_);
lean_ctor_set(v___x_752_, 4, v___x_751_);
lean_ctor_set(v___x_752_, 5, v_ref_728_);
lean_ctor_set(v___x_752_, 6, v_currNamespace_729_);
lean_ctor_set(v___x_752_, 7, v_openDecls_730_);
lean_ctor_set(v___x_752_, 8, v_initHeartbeats_731_);
lean_ctor_set(v___x_752_, 9, v_maxHeartbeats_732_);
lean_ctor_set(v___x_752_, 10, v_quotContext_733_);
lean_ctor_set(v___x_752_, 11, v_currMacroScope_734_);
lean_ctor_set(v___x_752_, 12, v_cancelTk_x3f_735_);
lean_ctor_set(v___x_752_, 13, v_inheritedTraceOptions_737_);
lean_ctor_set_uint8(v___x_752_, sizeof(void*)*14, v___x_723_);
lean_ctor_set_uint8(v___x_752_, sizeof(void*)*14 + 1, v_suppressElabErrors_736_);
lean_inc_ref(v_keyedConfig_739_);
v___x_753_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_718_, v_keyedConfig_739_);
lean_inc(v_customCanUnfoldPredicate_x3f_746_);
lean_inc(v_synthPendingDepth_745_);
lean_inc(v_defEqCtx_x3f_744_);
lean_inc_ref(v_localInstances_743_);
lean_inc_ref(v_lctx_742_);
lean_inc(v_zetaDeltaSet_741_);
v___x_754_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_754_, 0, v___x_753_);
lean_ctor_set(v___x_754_, 1, v_zetaDeltaSet_741_);
lean_ctor_set(v___x_754_, 2, v_lctx_742_);
lean_ctor_set(v___x_754_, 3, v_localInstances_743_);
lean_ctor_set(v___x_754_, 4, v_defEqCtx_x3f_744_);
lean_ctor_set(v___x_754_, 5, v_synthPendingDepth_745_);
lean_ctor_set(v___x_754_, 6, v_customCanUnfoldPredicate_x3f_746_);
lean_ctor_set_uint8(v___x_754_, sizeof(void*)*7, v_trackZetaDelta_740_);
lean_ctor_set_uint8(v___x_754_, sizeof(void*)*7 + 1, v_univApprox_747_);
lean_ctor_set_uint8(v___x_754_, sizeof(void*)*7 + 2, v_inTypeClassResolution_748_);
lean_ctor_set_uint8(v___x_754_, sizeof(void*)*7 + 3, v_cacheInferType_749_);
lean_inc_ref(v_e_u2082_684_);
lean_inc_ref(v_e_u2081_683_);
v___x_755_ = l_Lean_Meta_isExprDefEq(v_e_u2081_683_, v_e_u2082_684_, v___x_754_, v_a_687_, v___x_752_, v___y_738_);
lean_dec_ref_known(v___x_754_, 7);
if (lean_obj_tag(v___x_755_) == 0)
{
lean_object* v_a_756_; lean_object* v___x_758_; uint8_t v_isShared_759_; uint8_t v_isSharedCheck_912_; 
v_a_756_ = lean_ctor_get(v___x_755_, 0);
v_isSharedCheck_912_ = !lean_is_exclusive(v___x_755_);
if (v_isSharedCheck_912_ == 0)
{
v___x_758_ = v___x_755_;
v_isShared_759_ = v_isSharedCheck_912_;
goto v_resetjp_757_;
}
else
{
lean_inc(v_a_756_);
lean_dec(v___x_755_);
v___x_758_ = lean_box(0);
v_isShared_759_ = v_isSharedCheck_912_;
goto v_resetjp_757_;
}
v_resetjp_757_:
{
uint8_t v___x_760_; 
v___x_760_ = lean_unbox(v_a_756_);
if (v___x_760_ == 0)
{
lean_object* v___x_761_; 
lean_del_object(v___x_758_);
lean_dec_ref(v___f_707_);
lean_inc_ref(v_e_u2082_684_);
lean_inc_ref(v_e_u2081_683_);
v___x_761_ = l_Lean_Meta_isExprDefEq(v_e_u2081_683_, v_e_u2082_684_, v_a_686_, v_a_687_, v___x_752_, v___y_738_);
if (lean_obj_tag(v___x_761_) == 0)
{
lean_object* v_a_762_; lean_object* v___x_764_; uint8_t v_isShared_765_; uint8_t v_isSharedCheck_896_; 
v_a_762_ = lean_ctor_get(v___x_761_, 0);
v_isSharedCheck_896_ = !lean_is_exclusive(v___x_761_);
if (v_isSharedCheck_896_ == 0)
{
v___x_764_ = v___x_761_;
v_isShared_765_ = v_isSharedCheck_896_;
goto v_resetjp_763_;
}
else
{
lean_inc(v_a_762_);
lean_dec(v___x_761_);
v___x_764_ = lean_box(0);
v_isShared_765_ = v_isSharedCheck_896_;
goto v_resetjp_763_;
}
v_resetjp_763_:
{
lean_object* v___x_766_; uint8_t v___x_767_; 
v___x_766_ = lean_array_push(v_a_685_, v___f_717_);
v___x_767_ = lean_unbox(v_a_762_);
if (v___x_767_ == 0)
{
lean_object* v___x_768_; lean_object* v___x_769_; lean_object* v___x_771_; 
lean_dec(v_a_756_);
lean_dec_ref_known(v___x_752_, 14);
lean_dec_ref(v___f_708_);
lean_dec_ref(v_e_u2082_684_);
lean_dec_ref(v_e_u2081_683_);
v___x_768_ = lean_array_push(v___x_766_, v___f_716_);
v___x_769_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_769_, 0, v_a_762_);
lean_ctor_set(v___x_769_, 1, v___x_768_);
if (v_isShared_765_ == 0)
{
lean_ctor_set(v___x_764_, 0, v___x_769_);
v___x_771_ = v___x_764_;
goto v_reusejp_770_;
}
else
{
lean_object* v_reuseFailAlloc_772_; 
v_reuseFailAlloc_772_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_772_, 0, v___x_769_);
v___x_771_ = v_reuseFailAlloc_772_;
goto v_reusejp_770_;
}
v_reusejp_770_:
{
return v___x_771_;
}
}
else
{
lean_del_object(v___x_764_);
lean_dec_ref(v___f_716_);
switch(lean_obj_tag(v_e_u2081_683_))
{
case 5:
{
if (lean_obj_tag(v_e_u2082_684_) == 5)
{
lean_object* v_fn_773_; lean_object* v_arg_774_; lean_object* v_fn_775_; lean_object* v_arg_776_; lean_object* v___x_777_; 
lean_dec(v_a_756_);
lean_dec_ref(v___f_708_);
v_fn_773_ = lean_ctor_get(v_e_u2081_683_, 0);
v_arg_774_ = lean_ctor_get(v_e_u2081_683_, 1);
v_fn_775_ = lean_ctor_get(v_e_u2082_684_, 0);
v_arg_776_ = lean_ctor_get(v_e_u2082_684_, 1);
lean_inc_ref(v_arg_776_);
lean_inc_ref(v_arg_774_);
v___x_777_ = lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs(v_tk_682_, v_arg_774_, v_arg_776_, v___x_766_, v_a_686_, v_a_687_, v___x_752_, v___y_738_);
if (lean_obj_tag(v___x_777_) == 0)
{
lean_object* v_a_778_; lean_object* v___x_780_; uint8_t v_isShared_781_; uint8_t v_isSharedCheck_855_; 
v_a_778_ = lean_ctor_get(v___x_777_, 0);
v_isSharedCheck_855_ = !lean_is_exclusive(v___x_777_);
if (v_isSharedCheck_855_ == 0)
{
v___x_780_ = v___x_777_;
v_isShared_781_ = v_isSharedCheck_855_;
goto v_resetjp_779_;
}
else
{
lean_inc(v_a_778_);
lean_dec(v___x_777_);
v___x_780_ = lean_box(0);
v_isShared_781_ = v_isSharedCheck_855_;
goto v_resetjp_779_;
}
v_resetjp_779_:
{
lean_object* v_fst_782_; uint8_t v___x_783_; 
v_fst_782_ = lean_ctor_get(v_a_778_, 0);
v___x_783_ = lean_unbox(v_fst_782_);
if (v___x_783_ == 0)
{
lean_object* v_snd_784_; lean_object* v___x_785_; 
lean_del_object(v___x_780_);
v_snd_784_ = lean_ctor_get(v_a_778_, 1);
lean_inc(v_snd_784_);
lean_dec(v_a_778_);
lean_inc_ref(v_fn_775_);
lean_inc_ref(v_fn_773_);
v___x_785_ = lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs(v_tk_682_, v_fn_773_, v_fn_775_, v_snd_784_, v_a_686_, v_a_687_, v___x_752_, v___y_738_);
if (lean_obj_tag(v___x_785_) == 0)
{
lean_object* v_a_786_; lean_object* v___x_788_; uint8_t v_isShared_789_; uint8_t v_isSharedCheck_842_; 
v_a_786_ = lean_ctor_get(v___x_785_, 0);
v_isSharedCheck_842_ = !lean_is_exclusive(v___x_785_);
if (v_isSharedCheck_842_ == 0)
{
v___x_788_ = v___x_785_;
v_isShared_789_ = v_isSharedCheck_842_;
goto v_resetjp_787_;
}
else
{
lean_inc(v_a_786_);
lean_dec(v___x_785_);
v___x_788_ = lean_box(0);
v_isShared_789_ = v_isSharedCheck_842_;
goto v_resetjp_787_;
}
v_resetjp_787_:
{
lean_object* v_fst_790_; uint8_t v___x_791_; 
v_fst_790_ = lean_ctor_get(v_a_786_, 0);
v___x_791_ = lean_unbox(v_fst_790_);
if (v___x_791_ == 0)
{
lean_object* v_snd_792_; lean_object* v___x_793_; lean_object* v___x_794_; lean_object* v___x_795_; lean_object* v___x_796_; lean_object* v___x_797_; lean_object* v___x_798_; lean_object* v___x_799_; lean_object* v___x_800_; lean_object* v___x_801_; lean_object* v___x_802_; lean_object* v___x_803_; lean_object* v___x_804_; 
lean_del_object(v___x_788_);
v_snd_792_ = lean_ctor_get(v_a_786_, 1);
lean_inc(v_snd_792_);
lean_dec(v_a_786_);
v___x_793_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__3___closed__0, &lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__3___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__3___closed__0);
v___x_794_ = l_Lean_MessageData_ofExpr(v_e_u2081_683_);
v___x_795_ = l_Lean_indentD(v___x_794_);
v___x_796_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_796_, 0, v___x_793_);
lean_ctor_set(v___x_796_, 1, v___x_795_);
v___x_797_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__5, &lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__5);
v___x_798_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_798_, 0, v___x_796_);
lean_ctor_set(v___x_798_, 1, v___x_797_);
v___x_799_ = l_Lean_MessageData_ofExpr(v_e_u2082_684_);
v___x_800_ = l_Lean_indentD(v___x_799_);
v___x_801_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_801_, 0, v___x_798_);
lean_ctor_set(v___x_801_, 1, v___x_800_);
v___x_802_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__4, &lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__4);
v___x_803_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_803_, 0, v___x_801_);
lean_ctor_set(v___x_803_, 1, v___x_802_);
v___x_804_ = lp_mathlib_Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3(v_tk_682_, v___x_803_, v_snd_792_, v_a_686_, v_a_687_, v___x_752_, v___y_738_);
lean_dec_ref_known(v___x_752_, 14);
if (lean_obj_tag(v___x_804_) == 0)
{
lean_object* v_a_805_; lean_object* v___x_807_; uint8_t v_isShared_808_; uint8_t v_isSharedCheck_821_; 
v_a_805_ = lean_ctor_get(v___x_804_, 0);
v_isSharedCheck_821_ = !lean_is_exclusive(v___x_804_);
if (v_isSharedCheck_821_ == 0)
{
v___x_807_ = v___x_804_;
v_isShared_808_ = v_isSharedCheck_821_;
goto v_resetjp_806_;
}
else
{
lean_inc(v_a_805_);
lean_dec(v___x_804_);
v___x_807_ = lean_box(0);
v_isShared_808_ = v_isSharedCheck_821_;
goto v_resetjp_806_;
}
v_resetjp_806_:
{
lean_object* v_snd_809_; lean_object* v___x_811_; uint8_t v_isShared_812_; uint8_t v_isSharedCheck_819_; 
v_snd_809_ = lean_ctor_get(v_a_805_, 1);
v_isSharedCheck_819_ = !lean_is_exclusive(v_a_805_);
if (v_isSharedCheck_819_ == 0)
{
lean_object* v_unused_820_; 
v_unused_820_ = lean_ctor_get(v_a_805_, 0);
lean_dec(v_unused_820_);
v___x_811_ = v_a_805_;
v_isShared_812_ = v_isSharedCheck_819_;
goto v_resetjp_810_;
}
else
{
lean_inc(v_snd_809_);
lean_dec(v_a_805_);
v___x_811_ = lean_box(0);
v_isShared_812_ = v_isSharedCheck_819_;
goto v_resetjp_810_;
}
v_resetjp_810_:
{
lean_object* v___x_814_; 
if (v_isShared_812_ == 0)
{
lean_ctor_set(v___x_811_, 0, v_a_762_);
v___x_814_ = v___x_811_;
goto v_reusejp_813_;
}
else
{
lean_object* v_reuseFailAlloc_818_; 
v_reuseFailAlloc_818_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_818_, 0, v_a_762_);
lean_ctor_set(v_reuseFailAlloc_818_, 1, v_snd_809_);
v___x_814_ = v_reuseFailAlloc_818_;
goto v_reusejp_813_;
}
v_reusejp_813_:
{
lean_object* v___x_816_; 
if (v_isShared_808_ == 0)
{
lean_ctor_set(v___x_807_, 0, v___x_814_);
v___x_816_ = v___x_807_;
goto v_reusejp_815_;
}
else
{
lean_object* v_reuseFailAlloc_817_; 
v_reuseFailAlloc_817_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_817_, 0, v___x_814_);
v___x_816_ = v_reuseFailAlloc_817_;
goto v_reusejp_815_;
}
v_reusejp_815_:
{
return v___x_816_;
}
}
}
}
}
else
{
lean_object* v_a_822_; lean_object* v___x_824_; uint8_t v_isShared_825_; uint8_t v_isSharedCheck_829_; 
lean_dec(v_a_762_);
v_a_822_ = lean_ctor_get(v___x_804_, 0);
v_isSharedCheck_829_ = !lean_is_exclusive(v___x_804_);
if (v_isSharedCheck_829_ == 0)
{
v___x_824_ = v___x_804_;
v_isShared_825_ = v_isSharedCheck_829_;
goto v_resetjp_823_;
}
else
{
lean_inc(v_a_822_);
lean_dec(v___x_804_);
v___x_824_ = lean_box(0);
v_isShared_825_ = v_isSharedCheck_829_;
goto v_resetjp_823_;
}
v_resetjp_823_:
{
lean_object* v___x_827_; 
if (v_isShared_825_ == 0)
{
v___x_827_ = v___x_824_;
goto v_reusejp_826_;
}
else
{
lean_object* v_reuseFailAlloc_828_; 
v_reuseFailAlloc_828_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_828_, 0, v_a_822_);
v___x_827_ = v_reuseFailAlloc_828_;
goto v_reusejp_826_;
}
v_reusejp_826_:
{
return v___x_827_;
}
}
}
}
else
{
lean_object* v_snd_830_; lean_object* v___x_832_; uint8_t v_isShared_833_; uint8_t v_isSharedCheck_840_; 
lean_dec_ref_known(v_e_u2082_684_, 2);
lean_dec_ref_known(v_e_u2081_683_, 2);
lean_dec_ref_known(v___x_752_, 14);
v_snd_830_ = lean_ctor_get(v_a_786_, 1);
v_isSharedCheck_840_ = !lean_is_exclusive(v_a_786_);
if (v_isSharedCheck_840_ == 0)
{
lean_object* v_unused_841_; 
v_unused_841_ = lean_ctor_get(v_a_786_, 0);
lean_dec(v_unused_841_);
v___x_832_ = v_a_786_;
v_isShared_833_ = v_isSharedCheck_840_;
goto v_resetjp_831_;
}
else
{
lean_inc(v_snd_830_);
lean_dec(v_a_786_);
v___x_832_ = lean_box(0);
v_isShared_833_ = v_isSharedCheck_840_;
goto v_resetjp_831_;
}
v_resetjp_831_:
{
lean_object* v___x_835_; 
if (v_isShared_833_ == 0)
{
lean_ctor_set(v___x_832_, 0, v_a_762_);
v___x_835_ = v___x_832_;
goto v_reusejp_834_;
}
else
{
lean_object* v_reuseFailAlloc_839_; 
v_reuseFailAlloc_839_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_839_, 0, v_a_762_);
lean_ctor_set(v_reuseFailAlloc_839_, 1, v_snd_830_);
v___x_835_ = v_reuseFailAlloc_839_;
goto v_reusejp_834_;
}
v_reusejp_834_:
{
lean_object* v___x_837_; 
if (v_isShared_789_ == 0)
{
lean_ctor_set(v___x_788_, 0, v___x_835_);
v___x_837_ = v___x_788_;
goto v_reusejp_836_;
}
else
{
lean_object* v_reuseFailAlloc_838_; 
v_reuseFailAlloc_838_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_838_, 0, v___x_835_);
v___x_837_ = v_reuseFailAlloc_838_;
goto v_reusejp_836_;
}
v_reusejp_836_:
{
return v___x_837_;
}
}
}
}
}
}
else
{
lean_dec_ref_known(v_e_u2082_684_, 2);
lean_dec_ref_known(v_e_u2081_683_, 2);
lean_dec(v_a_762_);
lean_dec_ref_known(v___x_752_, 14);
return v___x_785_;
}
}
else
{
lean_object* v_snd_843_; lean_object* v___x_845_; uint8_t v_isShared_846_; uint8_t v_isSharedCheck_853_; 
lean_dec_ref_known(v_e_u2082_684_, 2);
lean_dec_ref_known(v_e_u2081_683_, 2);
lean_dec_ref_known(v___x_752_, 14);
v_snd_843_ = lean_ctor_get(v_a_778_, 1);
v_isSharedCheck_853_ = !lean_is_exclusive(v_a_778_);
if (v_isSharedCheck_853_ == 0)
{
lean_object* v_unused_854_; 
v_unused_854_ = lean_ctor_get(v_a_778_, 0);
lean_dec(v_unused_854_);
v___x_845_ = v_a_778_;
v_isShared_846_ = v_isSharedCheck_853_;
goto v_resetjp_844_;
}
else
{
lean_inc(v_snd_843_);
lean_dec(v_a_778_);
v___x_845_ = lean_box(0);
v_isShared_846_ = v_isSharedCheck_853_;
goto v_resetjp_844_;
}
v_resetjp_844_:
{
lean_object* v___x_848_; 
if (v_isShared_846_ == 0)
{
lean_ctor_set(v___x_845_, 0, v_a_762_);
v___x_848_ = v___x_845_;
goto v_reusejp_847_;
}
else
{
lean_object* v_reuseFailAlloc_852_; 
v_reuseFailAlloc_852_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_852_, 0, v_a_762_);
lean_ctor_set(v_reuseFailAlloc_852_, 1, v_snd_843_);
v___x_848_ = v_reuseFailAlloc_852_;
goto v_reusejp_847_;
}
v_reusejp_847_:
{
lean_object* v___x_850_; 
if (v_isShared_781_ == 0)
{
lean_ctor_set(v___x_780_, 0, v___x_848_);
v___x_850_ = v___x_780_;
goto v_reusejp_849_;
}
else
{
lean_object* v_reuseFailAlloc_851_; 
v_reuseFailAlloc_851_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_851_, 0, v___x_848_);
v___x_850_ = v_reuseFailAlloc_851_;
goto v_reusejp_849_;
}
v_reusejp_849_:
{
return v___x_850_;
}
}
}
}
}
}
else
{
lean_dec_ref_known(v_e_u2082_684_, 2);
lean_dec_ref_known(v_e_u2081_683_, 2);
lean_dec(v_a_762_);
lean_dec_ref_known(v___x_752_, 14);
return v___x_777_;
}
}
else
{
uint8_t v___x_856_; 
lean_dec_ref_known(v_e_u2081_683_, 2);
lean_dec(v_a_762_);
lean_dec_ref_known(v___x_752_, 14);
lean_dec_ref(v_e_u2082_684_);
v___x_856_ = lean_unbox(v_a_756_);
lean_dec(v_a_756_);
v___y_710_ = v___x_856_;
v___y_711_ = v___x_766_;
goto v___jp_709_;
}
}
case 4:
{
if (lean_obj_tag(v_e_u2082_684_) == 4)
{
lean_object* v___x_857_; lean_object* v___x_858_; lean_object* v___x_859_; lean_object* v___x_860_; lean_object* v___x_861_; lean_object* v___x_862_; lean_object* v___x_863_; lean_object* v___x_864_; lean_object* v___x_865_; lean_object* v___x_866_; lean_object* v___x_867_; lean_object* v___x_868_; 
lean_dec(v_a_756_);
lean_dec_ref(v___f_708_);
v___x_857_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__3___closed__0, &lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__3___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__3___closed__0);
v___x_858_ = l_Lean_MessageData_ofExpr(v_e_u2081_683_);
v___x_859_ = l_Lean_indentD(v___x_858_);
v___x_860_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_860_, 0, v___x_857_);
lean_ctor_set(v___x_860_, 1, v___x_859_);
v___x_861_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__5, &lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___lam__0___closed__5);
v___x_862_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_862_, 0, v___x_860_);
lean_ctor_set(v___x_862_, 1, v___x_861_);
v___x_863_ = l_Lean_MessageData_ofExpr(v_e_u2082_684_);
v___x_864_ = l_Lean_indentD(v___x_863_);
v___x_865_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_865_, 0, v___x_862_);
lean_ctor_set(v___x_865_, 1, v___x_864_);
v___x_866_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__4, &lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__4);
v___x_867_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_867_, 0, v___x_865_);
lean_ctor_set(v___x_867_, 1, v___x_866_);
v___x_868_ = lp_mathlib_Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3(v_tk_682_, v___x_867_, v___x_766_, v_a_686_, v_a_687_, v___x_752_, v___y_738_);
lean_dec_ref_known(v___x_752_, 14);
if (lean_obj_tag(v___x_868_) == 0)
{
lean_object* v_a_869_; lean_object* v___x_871_; uint8_t v_isShared_872_; uint8_t v_isSharedCheck_885_; 
v_a_869_ = lean_ctor_get(v___x_868_, 0);
v_isSharedCheck_885_ = !lean_is_exclusive(v___x_868_);
if (v_isSharedCheck_885_ == 0)
{
v___x_871_ = v___x_868_;
v_isShared_872_ = v_isSharedCheck_885_;
goto v_resetjp_870_;
}
else
{
lean_inc(v_a_869_);
lean_dec(v___x_868_);
v___x_871_ = lean_box(0);
v_isShared_872_ = v_isSharedCheck_885_;
goto v_resetjp_870_;
}
v_resetjp_870_:
{
lean_object* v_snd_873_; lean_object* v___x_875_; uint8_t v_isShared_876_; uint8_t v_isSharedCheck_883_; 
v_snd_873_ = lean_ctor_get(v_a_869_, 1);
v_isSharedCheck_883_ = !lean_is_exclusive(v_a_869_);
if (v_isSharedCheck_883_ == 0)
{
lean_object* v_unused_884_; 
v_unused_884_ = lean_ctor_get(v_a_869_, 0);
lean_dec(v_unused_884_);
v___x_875_ = v_a_869_;
v_isShared_876_ = v_isSharedCheck_883_;
goto v_resetjp_874_;
}
else
{
lean_inc(v_snd_873_);
lean_dec(v_a_869_);
v___x_875_ = lean_box(0);
v_isShared_876_ = v_isSharedCheck_883_;
goto v_resetjp_874_;
}
v_resetjp_874_:
{
lean_object* v___x_878_; 
if (v_isShared_876_ == 0)
{
lean_ctor_set(v___x_875_, 0, v_a_762_);
v___x_878_ = v___x_875_;
goto v_reusejp_877_;
}
else
{
lean_object* v_reuseFailAlloc_882_; 
v_reuseFailAlloc_882_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_882_, 0, v_a_762_);
lean_ctor_set(v_reuseFailAlloc_882_, 1, v_snd_873_);
v___x_878_ = v_reuseFailAlloc_882_;
goto v_reusejp_877_;
}
v_reusejp_877_:
{
lean_object* v___x_880_; 
if (v_isShared_872_ == 0)
{
lean_ctor_set(v___x_871_, 0, v___x_878_);
v___x_880_ = v___x_871_;
goto v_reusejp_879_;
}
else
{
lean_object* v_reuseFailAlloc_881_; 
v_reuseFailAlloc_881_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_881_, 0, v___x_878_);
v___x_880_ = v_reuseFailAlloc_881_;
goto v_reusejp_879_;
}
v_reusejp_879_:
{
return v___x_880_;
}
}
}
}
}
else
{
lean_object* v_a_886_; lean_object* v___x_888_; uint8_t v_isShared_889_; uint8_t v_isSharedCheck_893_; 
lean_dec(v_a_762_);
v_a_886_ = lean_ctor_get(v___x_868_, 0);
v_isSharedCheck_893_ = !lean_is_exclusive(v___x_868_);
if (v_isSharedCheck_893_ == 0)
{
v___x_888_ = v___x_868_;
v_isShared_889_ = v_isSharedCheck_893_;
goto v_resetjp_887_;
}
else
{
lean_inc(v_a_886_);
lean_dec(v___x_868_);
v___x_888_ = lean_box(0);
v_isShared_889_ = v_isSharedCheck_893_;
goto v_resetjp_887_;
}
v_resetjp_887_:
{
lean_object* v___x_891_; 
if (v_isShared_889_ == 0)
{
v___x_891_ = v___x_888_;
goto v_reusejp_890_;
}
else
{
lean_object* v_reuseFailAlloc_892_; 
v_reuseFailAlloc_892_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_892_, 0, v_a_886_);
v___x_891_ = v_reuseFailAlloc_892_;
goto v_reusejp_890_;
}
v_reusejp_890_:
{
return v___x_891_;
}
}
}
}
else
{
uint8_t v___x_894_; 
lean_dec_ref_known(v_e_u2081_683_, 2);
lean_dec(v_a_762_);
lean_dec_ref_known(v___x_752_, 14);
lean_dec_ref(v_e_u2082_684_);
v___x_894_ = lean_unbox(v_a_756_);
lean_dec(v_a_756_);
v___y_710_ = v___x_894_;
v___y_711_ = v___x_766_;
goto v___jp_709_;
}
}
default: 
{
uint8_t v___x_895_; 
lean_dec(v_a_762_);
lean_dec_ref_known(v___x_752_, 14);
lean_dec_ref(v_e_u2082_684_);
lean_dec_ref(v_e_u2081_683_);
v___x_895_ = lean_unbox(v_a_756_);
lean_dec(v_a_756_);
v___y_710_ = v___x_895_;
v___y_711_ = v___x_766_;
goto v___jp_709_;
}
}
}
}
}
else
{
lean_object* v_a_897_; lean_object* v___x_899_; uint8_t v_isShared_900_; uint8_t v_isSharedCheck_904_; 
lean_dec(v_a_756_);
lean_dec_ref_known(v___x_752_, 14);
lean_dec_ref(v___f_717_);
lean_dec_ref(v___f_716_);
lean_dec_ref(v___f_708_);
lean_dec_ref(v_a_685_);
lean_dec_ref(v_e_u2082_684_);
lean_dec_ref(v_e_u2081_683_);
v_a_897_ = lean_ctor_get(v___x_761_, 0);
v_isSharedCheck_904_ = !lean_is_exclusive(v___x_761_);
if (v_isSharedCheck_904_ == 0)
{
v___x_899_ = v___x_761_;
v_isShared_900_ = v_isSharedCheck_904_;
goto v_resetjp_898_;
}
else
{
lean_inc(v_a_897_);
lean_dec(v___x_761_);
v___x_899_ = lean_box(0);
v_isShared_900_ = v_isSharedCheck_904_;
goto v_resetjp_898_;
}
v_resetjp_898_:
{
lean_object* v___x_902_; 
if (v_isShared_900_ == 0)
{
v___x_902_ = v___x_899_;
goto v_reusejp_901_;
}
else
{
lean_object* v_reuseFailAlloc_903_; 
v_reuseFailAlloc_903_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_903_, 0, v_a_897_);
v___x_902_ = v_reuseFailAlloc_903_;
goto v_reusejp_901_;
}
v_reusejp_901_:
{
return v___x_902_;
}
}
}
}
else
{
lean_object* v___x_905_; uint8_t v___x_906_; lean_object* v___x_907_; lean_object* v___x_908_; lean_object* v___x_910_; 
lean_dec(v_a_756_);
lean_dec_ref_known(v___x_752_, 14);
lean_dec_ref(v___f_717_);
lean_dec_ref(v___f_716_);
lean_dec_ref(v___f_708_);
lean_dec_ref(v_e_u2082_684_);
lean_dec_ref(v_e_u2081_683_);
v___x_905_ = lean_array_push(v_a_685_, v___f_707_);
v___x_906_ = 0;
v___x_907_ = lean_box(v___x_906_);
v___x_908_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_908_, 0, v___x_907_);
lean_ctor_set(v___x_908_, 1, v___x_905_);
if (v_isShared_759_ == 0)
{
lean_ctor_set(v___x_758_, 0, v___x_908_);
v___x_910_ = v___x_758_;
goto v_reusejp_909_;
}
else
{
lean_object* v_reuseFailAlloc_911_; 
v_reuseFailAlloc_911_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_911_, 0, v___x_908_);
v___x_910_ = v_reuseFailAlloc_911_;
goto v_reusejp_909_;
}
v_reusejp_909_:
{
return v___x_910_;
}
}
}
}
else
{
lean_object* v_a_913_; lean_object* v___x_915_; uint8_t v_isShared_916_; uint8_t v_isSharedCheck_920_; 
lean_dec_ref_known(v___x_752_, 14);
lean_dec_ref(v___f_717_);
lean_dec_ref(v___f_716_);
lean_dec_ref(v___f_708_);
lean_dec_ref(v___f_707_);
lean_dec_ref(v_a_685_);
lean_dec_ref(v_e_u2082_684_);
lean_dec_ref(v_e_u2081_683_);
v_a_913_ = lean_ctor_get(v___x_755_, 0);
v_isSharedCheck_920_ = !lean_is_exclusive(v___x_755_);
if (v_isSharedCheck_920_ == 0)
{
v___x_915_ = v___x_755_;
v_isShared_916_ = v_isSharedCheck_920_;
goto v_resetjp_914_;
}
else
{
lean_inc(v_a_913_);
lean_dec(v___x_755_);
v___x_915_ = lean_box(0);
v_isShared_916_ = v_isSharedCheck_920_;
goto v_resetjp_914_;
}
v_resetjp_914_:
{
lean_object* v___x_918_; 
if (v_isShared_916_ == 0)
{
v___x_918_ = v___x_915_;
goto v_reusejp_917_;
}
else
{
lean_object* v_reuseFailAlloc_919_; 
v_reuseFailAlloc_919_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_919_, 0, v_a_913_);
v___x_918_ = v_reuseFailAlloc_919_;
goto v_reusejp_917_;
}
v_reusejp_917_:
{
return v___x_918_;
}
}
}
}
v___jp_921_:
{
if (v___y_922_ == 0)
{
lean_object* v___x_923_; lean_object* v_env_924_; lean_object* v_nextMacroScope_925_; lean_object* v_ngen_926_; lean_object* v_auxDeclNGen_927_; lean_object* v_traceState_928_; lean_object* v_messages_929_; lean_object* v_infoState_930_; lean_object* v_snapshotTasks_931_; lean_object* v___x_933_; uint8_t v_isShared_934_; uint8_t v_isSharedCheck_941_; 
v___x_923_ = lean_st_ref_take(v_a_689_);
v_env_924_ = lean_ctor_get(v___x_923_, 0);
v_nextMacroScope_925_ = lean_ctor_get(v___x_923_, 1);
v_ngen_926_ = lean_ctor_get(v___x_923_, 2);
v_auxDeclNGen_927_ = lean_ctor_get(v___x_923_, 3);
v_traceState_928_ = lean_ctor_get(v___x_923_, 4);
v_messages_929_ = lean_ctor_get(v___x_923_, 6);
v_infoState_930_ = lean_ctor_get(v___x_923_, 7);
v_snapshotTasks_931_ = lean_ctor_get(v___x_923_, 8);
v_isSharedCheck_941_ = !lean_is_exclusive(v___x_923_);
if (v_isSharedCheck_941_ == 0)
{
lean_object* v_unused_942_; 
v_unused_942_ = lean_ctor_get(v___x_923_, 5);
lean_dec(v_unused_942_);
v___x_933_ = v___x_923_;
v_isShared_934_ = v_isSharedCheck_941_;
goto v_resetjp_932_;
}
else
{
lean_inc(v_snapshotTasks_931_);
lean_inc(v_infoState_930_);
lean_inc(v_messages_929_);
lean_inc(v_traceState_928_);
lean_inc(v_auxDeclNGen_927_);
lean_inc(v_ngen_926_);
lean_inc(v_nextMacroScope_925_);
lean_inc(v_env_924_);
lean_dec(v___x_923_);
v___x_933_ = lean_box(0);
v_isShared_934_ = v_isSharedCheck_941_;
goto v_resetjp_932_;
}
v_resetjp_932_:
{
lean_object* v___x_935_; lean_object* v___x_936_; lean_object* v___x_938_; 
v___x_935_ = l_Lean_Kernel_enableDiag(v_env_924_, v___x_723_);
v___x_936_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__7, &lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__7_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___closed__7);
if (v_isShared_934_ == 0)
{
lean_ctor_set(v___x_933_, 5, v___x_936_);
lean_ctor_set(v___x_933_, 0, v___x_935_);
v___x_938_ = v___x_933_;
goto v_reusejp_937_;
}
else
{
lean_object* v_reuseFailAlloc_940_; 
v_reuseFailAlloc_940_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_940_, 0, v___x_935_);
lean_ctor_set(v_reuseFailAlloc_940_, 1, v_nextMacroScope_925_);
lean_ctor_set(v_reuseFailAlloc_940_, 2, v_ngen_926_);
lean_ctor_set(v_reuseFailAlloc_940_, 3, v_auxDeclNGen_927_);
lean_ctor_set(v_reuseFailAlloc_940_, 4, v_traceState_928_);
lean_ctor_set(v_reuseFailAlloc_940_, 5, v___x_936_);
lean_ctor_set(v_reuseFailAlloc_940_, 6, v_messages_929_);
lean_ctor_set(v_reuseFailAlloc_940_, 7, v_infoState_930_);
lean_ctor_set(v_reuseFailAlloc_940_, 8, v_snapshotTasks_931_);
v___x_938_ = v_reuseFailAlloc_940_;
goto v_reusejp_937_;
}
v_reusejp_937_:
{
lean_object* v___x_939_; 
v___x_939_ = lean_st_ref_set(v_a_689_, v___x_938_);
v_fileName_725_ = v_fileName_692_;
v_fileMap_726_ = v_fileMap_693_;
v_currRecDepth_727_ = v_currRecDepth_695_;
v_ref_728_ = v_ref_696_;
v_currNamespace_729_ = v_currNamespace_697_;
v_openDecls_730_ = v_openDecls_698_;
v_initHeartbeats_731_ = v_initHeartbeats_699_;
v_maxHeartbeats_732_ = v_maxHeartbeats_700_;
v_quotContext_733_ = v_quotContext_701_;
v_currMacroScope_734_ = v_currMacroScope_702_;
v_cancelTk_x3f_735_ = v_cancelTk_x3f_703_;
v_suppressElabErrors_736_ = v_suppressElabErrors_704_;
v_inheritedTraceOptions_737_ = v_inheritedTraceOptions_705_;
v___y_738_ = v_a_689_;
goto v___jp_724_;
}
}
}
else
{
v_fileName_725_ = v_fileName_692_;
v_fileMap_726_ = v_fileMap_693_;
v_currRecDepth_727_ = v_currRecDepth_695_;
v_ref_728_ = v_ref_696_;
v_currNamespace_729_ = v_currNamespace_697_;
v_openDecls_730_ = v_openDecls_698_;
v_initHeartbeats_731_ = v_initHeartbeats_699_;
v_maxHeartbeats_732_ = v_maxHeartbeats_700_;
v_quotContext_733_ = v_quotContext_701_;
v_currMacroScope_734_ = v_currMacroScope_702_;
v_cancelTk_x3f_735_ = v_cancelTk_x3f_703_;
v_suppressElabErrors_736_ = v_suppressElabErrors_704_;
v_inheritedTraceOptions_737_ = v_inheritedTraceOptions_705_;
v___y_738_ = v_a_689_;
goto v___jp_724_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs___boxed(lean_object* v_tk_944_, lean_object* v_e_u2081_945_, lean_object* v_e_u2082_946_, lean_object* v_a_947_, lean_object* v_a_948_, lean_object* v_a_949_, lean_object* v_a_950_, lean_object* v_a_951_, lean_object* v_a_952_){
_start:
{
lean_object* v_res_953_; 
v_res_953_ = lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs(v_tk_944_, v_e_u2081_945_, v_e_u2082_946_, v_a_947_, v_a_948_, v_a_949_, v_a_950_, v_a_951_);
lean_dec(v_a_951_);
lean_dec_ref(v_a_950_);
lean_dec(v_a_949_);
lean_dec_ref(v_a_948_);
lean_dec(v_tk_944_);
return v_res_953_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Erw_x3f_extractRewriteEq_spec__0___redArg(lean_object* v_msg_954_, lean_object* v___y_955_, lean_object* v___y_956_, lean_object* v___y_957_, lean_object* v___y_958_){
_start:
{
lean_object* v_ref_960_; lean_object* v___x_961_; lean_object* v_a_962_; lean_object* v___x_964_; uint8_t v_isShared_965_; uint8_t v_isSharedCheck_970_; 
v_ref_960_ = lean_ctor_get(v___y_957_, 5);
v___x_961_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3_spec__4(v_msg_954_, v___y_955_, v___y_956_, v___y_957_, v___y_958_);
v_a_962_ = lean_ctor_get(v___x_961_, 0);
v_isSharedCheck_970_ = !lean_is_exclusive(v___x_961_);
if (v_isSharedCheck_970_ == 0)
{
v___x_964_ = v___x_961_;
v_isShared_965_ = v_isSharedCheck_970_;
goto v_resetjp_963_;
}
else
{
lean_inc(v_a_962_);
lean_dec(v___x_961_);
v___x_964_ = lean_box(0);
v_isShared_965_ = v_isSharedCheck_970_;
goto v_resetjp_963_;
}
v_resetjp_963_:
{
lean_object* v___x_966_; lean_object* v___x_968_; 
lean_inc(v_ref_960_);
v___x_966_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_966_, 0, v_ref_960_);
lean_ctor_set(v___x_966_, 1, v_a_962_);
if (v_isShared_965_ == 0)
{
lean_ctor_set_tag(v___x_964_, 1);
lean_ctor_set(v___x_964_, 0, v___x_966_);
v___x_968_ = v___x_964_;
goto v_reusejp_967_;
}
else
{
lean_object* v_reuseFailAlloc_969_; 
v_reuseFailAlloc_969_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_969_, 0, v___x_966_);
v___x_968_ = v_reuseFailAlloc_969_;
goto v_reusejp_967_;
}
v_reusejp_967_:
{
return v___x_968_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Erw_x3f_extractRewriteEq_spec__0___redArg___boxed(lean_object* v_msg_971_, lean_object* v___y_972_, lean_object* v___y_973_, lean_object* v___y_974_, lean_object* v___y_975_, lean_object* v___y_976_){
_start:
{
lean_object* v_res_977_; 
v_res_977_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Erw_x3f_extractRewriteEq_spec__0___redArg(v_msg_971_, v___y_972_, v___y_973_, v___y_974_, v___y_975_);
lean_dec(v___y_975_);
lean_dec_ref(v___y_974_);
lean_dec(v___y_973_);
lean_dec_ref(v___y_972_);
return v_res_977_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__1(void){
_start:
{
lean_object* v___x_979_; lean_object* v___x_980_; 
v___x_979_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__0));
v___x_980_ = l_Lean_stringToMessageData(v___x_979_);
return v___x_980_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__3(void){
_start:
{
lean_object* v___x_982_; lean_object* v___x_983_; 
v___x_982_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__2));
v___x_983_ = l_Lean_stringToMessageData(v___x_982_);
return v___x_983_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__9(void){
_start:
{
lean_object* v___x_990_; lean_object* v___x_991_; 
v___x_990_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__8));
v___x_991_ = l_Lean_stringToMessageData(v___x_990_);
return v___x_991_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__11(void){
_start:
{
lean_object* v___x_993_; lean_object* v___x_994_; 
v___x_993_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__10));
v___x_994_ = l_Lean_stringToMessageData(v___x_993_);
return v___x_994_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq(lean_object* v_e_995_, lean_object* v_a_996_, lean_object* v_a_997_, lean_object* v_a_998_, lean_object* v_a_999_){
_start:
{
lean_object* v___y_1002_; lean_object* v___y_1003_; lean_object* v___y_1004_; lean_object* v___y_1005_; lean_object* v___y_1009_; lean_object* v___y_1010_; lean_object* v___y_1011_; lean_object* v___y_1012_; lean_object* v___x_1015_; lean_object* v_fst_1016_; 
v___x_1015_ = l_Lean_Expr_getAppFnArgs(v_e_995_);
v_fst_1016_ = lean_ctor_get(v___x_1015_, 0);
lean_inc(v_fst_1016_);
if (lean_obj_tag(v_fst_1016_) == 1)
{
lean_object* v_pre_1017_; 
v_pre_1017_ = lean_ctor_get(v_fst_1016_, 0);
lean_inc(v_pre_1017_);
if (lean_obj_tag(v_pre_1017_) == 1)
{
lean_object* v_pre_1018_; 
v_pre_1018_ = lean_ctor_get(v_pre_1017_, 0);
if (lean_obj_tag(v_pre_1018_) == 0)
{
lean_object* v_snd_1019_; lean_object* v_str_1020_; lean_object* v_str_1021_; lean_object* v___x_1022_; uint8_t v___x_1023_; 
v_snd_1019_ = lean_ctor_get(v___x_1015_, 1);
lean_inc(v_snd_1019_);
lean_dec_ref(v___x_1015_);
v_str_1020_ = lean_ctor_get(v_fst_1016_, 1);
lean_inc_ref(v_str_1020_);
lean_dec_ref_known(v_fst_1016_, 2);
v_str_1021_ = lean_ctor_get(v_pre_1017_, 1);
lean_inc_ref(v_str_1021_);
lean_dec_ref_known(v_pre_1017_, 2);
v___x_1022_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__4));
v___x_1023_ = lean_string_dec_eq(v_str_1021_, v___x_1022_);
lean_dec_ref(v_str_1021_);
if (v___x_1023_ == 0)
{
lean_dec_ref(v_str_1020_);
lean_dec(v_snd_1019_);
v___y_1002_ = v_a_996_;
v___y_1003_ = v_a_997_;
v___y_1004_ = v_a_998_;
v___y_1005_ = v_a_999_;
goto v___jp_1001_;
}
else
{
lean_object* v___x_1024_; uint8_t v___x_1025_; 
v___x_1024_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__5));
v___x_1025_ = lean_string_dec_eq(v_str_1020_, v___x_1024_);
lean_dec_ref(v_str_1020_);
if (v___x_1025_ == 0)
{
lean_dec(v_snd_1019_);
v___y_1002_ = v_a_996_;
v___y_1003_ = v_a_997_;
v___y_1004_ = v_a_998_;
v___y_1005_ = v_a_999_;
goto v___jp_1001_;
}
else
{
lean_object* v___x_1026_; lean_object* v___x_1027_; uint8_t v___x_1028_; 
v___x_1026_ = lean_array_get_size(v_snd_1019_);
v___x_1027_ = lean_unsigned_to_nat(4u);
v___x_1028_ = lean_nat_dec_eq(v___x_1026_, v___x_1027_);
if (v___x_1028_ == 0)
{
lean_dec(v_snd_1019_);
v___y_1002_ = v_a_996_;
v___y_1003_ = v_a_997_;
v___y_1004_ = v_a_998_;
v___y_1005_ = v_a_999_;
goto v___jp_1001_;
}
else
{
lean_object* v___x_1029_; lean_object* v___x_1030_; lean_object* v___x_1031_; lean_object* v_fst_1032_; 
v___x_1029_ = lean_unsigned_to_nat(2u);
v___x_1030_ = lean_array_fget(v_snd_1019_, v___x_1029_);
lean_dec(v_snd_1019_);
v___x_1031_ = l_Lean_Expr_getAppFnArgs(v___x_1030_);
v_fst_1032_ = lean_ctor_get(v___x_1031_, 0);
lean_inc(v_fst_1032_);
if (lean_obj_tag(v_fst_1032_) == 1)
{
lean_object* v_pre_1033_; 
v_pre_1033_ = lean_ctor_get(v_fst_1032_, 0);
if (lean_obj_tag(v_pre_1033_) == 0)
{
lean_object* v_snd_1034_; lean_object* v___x_1036_; uint8_t v_isShared_1037_; uint8_t v_isSharedCheck_1079_; 
v_snd_1034_ = lean_ctor_get(v___x_1031_, 1);
v_isSharedCheck_1079_ = !lean_is_exclusive(v___x_1031_);
if (v_isSharedCheck_1079_ == 0)
{
lean_object* v_unused_1080_; 
v_unused_1080_ = lean_ctor_get(v___x_1031_, 0);
lean_dec(v_unused_1080_);
v___x_1036_ = v___x_1031_;
v_isShared_1037_ = v_isSharedCheck_1079_;
goto v_resetjp_1035_;
}
else
{
lean_inc(v_snd_1034_);
lean_dec(v___x_1031_);
v___x_1036_ = lean_box(0);
v_isShared_1037_ = v_isSharedCheck_1079_;
goto v_resetjp_1035_;
}
v_resetjp_1035_:
{
lean_object* v_str_1038_; lean_object* v___x_1039_; uint8_t v___x_1040_; 
v_str_1038_ = lean_ctor_get(v_fst_1032_, 1);
lean_inc_ref(v_str_1038_);
lean_dec_ref_known(v_fst_1032_, 2);
v___x_1039_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__6));
v___x_1040_ = lean_string_dec_eq(v_str_1038_, v___x_1039_);
lean_dec_ref(v_str_1038_);
if (v___x_1040_ == 0)
{
lean_del_object(v___x_1036_);
lean_dec(v_snd_1034_);
v___y_1009_ = v_a_996_;
v___y_1010_ = v_a_997_;
v___y_1011_ = v_a_998_;
v___y_1012_ = v_a_999_;
goto v___jp_1008_;
}
else
{
lean_object* v___x_1041_; uint8_t v___x_1042_; 
v___x_1041_ = lean_array_get_size(v_snd_1034_);
v___x_1042_ = lean_nat_dec_eq(v___x_1041_, v___x_1029_);
if (v___x_1042_ == 0)
{
lean_del_object(v___x_1036_);
lean_dec(v_snd_1034_);
v___y_1009_ = v_a_996_;
v___y_1010_ = v_a_997_;
v___y_1011_ = v_a_998_;
v___y_1012_ = v_a_999_;
goto v___jp_1008_;
}
else
{
lean_object* v___x_1043_; lean_object* v___x_1044_; lean_object* v___x_1045_; lean_object* v___x_1046_; uint8_t v___x_1047_; 
v___x_1043_ = lean_unsigned_to_nat(0u);
v___x_1044_ = lean_array_fget(v_snd_1034_, v___x_1043_);
v___x_1045_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__7));
v___x_1046_ = lean_unsigned_to_nat(3u);
v___x_1047_ = l_Lean_Expr_isAppOfArity(v___x_1044_, v___x_1045_, v___x_1046_);
if (v___x_1047_ == 0)
{
lean_object* v___x_1048_; lean_object* v___x_1049_; 
lean_dec(v___x_1044_);
lean_del_object(v___x_1036_);
lean_dec(v_snd_1034_);
v___x_1048_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__9, &lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__9);
v___x_1049_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Erw_x3f_extractRewriteEq_spec__0___redArg(v___x_1048_, v_a_996_, v_a_997_, v_a_998_, v_a_999_);
return v___x_1049_;
}
else
{
lean_object* v___x_1050_; lean_object* v___x_1051_; lean_object* v___x_1052_; 
v___x_1050_ = lean_unsigned_to_nat(1u);
v___x_1051_ = lean_array_fget(v_snd_1034_, v___x_1050_);
lean_dec(v_snd_1034_);
lean_inc(v_a_999_);
lean_inc_ref(v_a_998_);
lean_inc(v_a_997_);
lean_inc_ref(v_a_996_);
v___x_1052_ = lean_infer_type(v___x_1051_, v_a_996_, v_a_997_, v_a_998_, v_a_999_);
if (lean_obj_tag(v___x_1052_) == 0)
{
lean_object* v_a_1053_; lean_object* v___x_1055_; uint8_t v_isShared_1056_; uint8_t v_isSharedCheck_1070_; 
v_a_1053_ = lean_ctor_get(v___x_1052_, 0);
v_isSharedCheck_1070_ = !lean_is_exclusive(v___x_1052_);
if (v_isSharedCheck_1070_ == 0)
{
v___x_1055_ = v___x_1052_;
v_isShared_1056_ = v_isSharedCheck_1070_;
goto v_resetjp_1054_;
}
else
{
lean_inc(v_a_1053_);
lean_dec(v___x_1052_);
v___x_1055_ = lean_box(0);
v_isShared_1056_ = v_isSharedCheck_1070_;
goto v_resetjp_1054_;
}
v_resetjp_1054_:
{
uint8_t v___x_1057_; 
v___x_1057_ = l_Lean_Expr_isAppOfArity(v_a_1053_, v___x_1045_, v___x_1046_);
if (v___x_1057_ == 0)
{
lean_object* v___x_1058_; lean_object* v___x_1059_; 
lean_del_object(v___x_1055_);
lean_dec(v_a_1053_);
lean_dec(v___x_1044_);
lean_del_object(v___x_1036_);
v___x_1058_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__11, &lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__11_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__11);
v___x_1059_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Erw_x3f_extractRewriteEq_spec__0___redArg(v___x_1058_, v_a_996_, v_a_997_, v_a_998_, v_a_999_);
return v___x_1059_;
}
else
{
lean_object* v___x_1060_; lean_object* v___x_1061_; lean_object* v___x_1062_; lean_object* v___x_1063_; lean_object* v___x_1065_; 
v___x_1060_ = l_Lean_Expr_appFn_x21(v___x_1044_);
lean_dec(v___x_1044_);
v___x_1061_ = l_Lean_Expr_appArg_x21(v___x_1060_);
lean_dec_ref(v___x_1060_);
v___x_1062_ = l_Lean_Expr_appFn_x21(v_a_1053_);
lean_dec(v_a_1053_);
v___x_1063_ = l_Lean_Expr_appArg_x21(v___x_1062_);
lean_dec_ref(v___x_1062_);
if (v_isShared_1037_ == 0)
{
lean_ctor_set(v___x_1036_, 1, v___x_1063_);
lean_ctor_set(v___x_1036_, 0, v___x_1061_);
v___x_1065_ = v___x_1036_;
goto v_reusejp_1064_;
}
else
{
lean_object* v_reuseFailAlloc_1069_; 
v_reuseFailAlloc_1069_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1069_, 0, v___x_1061_);
lean_ctor_set(v_reuseFailAlloc_1069_, 1, v___x_1063_);
v___x_1065_ = v_reuseFailAlloc_1069_;
goto v_reusejp_1064_;
}
v_reusejp_1064_:
{
lean_object* v___x_1067_; 
if (v_isShared_1056_ == 0)
{
lean_ctor_set(v___x_1055_, 0, v___x_1065_);
v___x_1067_ = v___x_1055_;
goto v_reusejp_1066_;
}
else
{
lean_object* v_reuseFailAlloc_1068_; 
v_reuseFailAlloc_1068_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1068_, 0, v___x_1065_);
v___x_1067_ = v_reuseFailAlloc_1068_;
goto v_reusejp_1066_;
}
v_reusejp_1066_:
{
return v___x_1067_;
}
}
}
}
}
else
{
lean_object* v_a_1071_; lean_object* v___x_1073_; uint8_t v_isShared_1074_; uint8_t v_isSharedCheck_1078_; 
lean_dec(v___x_1044_);
lean_del_object(v___x_1036_);
v_a_1071_ = lean_ctor_get(v___x_1052_, 0);
v_isSharedCheck_1078_ = !lean_is_exclusive(v___x_1052_);
if (v_isSharedCheck_1078_ == 0)
{
v___x_1073_ = v___x_1052_;
v_isShared_1074_ = v_isSharedCheck_1078_;
goto v_resetjp_1072_;
}
else
{
lean_inc(v_a_1071_);
lean_dec(v___x_1052_);
v___x_1073_ = lean_box(0);
v_isShared_1074_ = v_isSharedCheck_1078_;
goto v_resetjp_1072_;
}
v_resetjp_1072_:
{
lean_object* v___x_1076_; 
if (v_isShared_1074_ == 0)
{
v___x_1076_ = v___x_1073_;
goto v_reusejp_1075_;
}
else
{
lean_object* v_reuseFailAlloc_1077_; 
v_reuseFailAlloc_1077_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1077_, 0, v_a_1071_);
v___x_1076_ = v_reuseFailAlloc_1077_;
goto v_reusejp_1075_;
}
v_reusejp_1075_:
{
return v___x_1076_;
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
lean_dec_ref_known(v_fst_1032_, 2);
lean_dec_ref(v___x_1031_);
v___y_1009_ = v_a_996_;
v___y_1010_ = v_a_997_;
v___y_1011_ = v_a_998_;
v___y_1012_ = v_a_999_;
goto v___jp_1008_;
}
}
else
{
lean_dec(v_fst_1032_);
lean_dec_ref(v___x_1031_);
v___y_1009_ = v_a_996_;
v___y_1010_ = v_a_997_;
v___y_1011_ = v_a_998_;
v___y_1012_ = v_a_999_;
goto v___jp_1008_;
}
}
}
}
}
else
{
lean_dec_ref_known(v_pre_1017_, 2);
lean_dec_ref_known(v_fst_1016_, 2);
lean_dec_ref(v___x_1015_);
v___y_1002_ = v_a_996_;
v___y_1003_ = v_a_997_;
v___y_1004_ = v_a_998_;
v___y_1005_ = v_a_999_;
goto v___jp_1001_;
}
}
else
{
lean_dec_ref_known(v_fst_1016_, 2);
lean_dec(v_pre_1017_);
lean_dec_ref(v___x_1015_);
v___y_1002_ = v_a_996_;
v___y_1003_ = v_a_997_;
v___y_1004_ = v_a_998_;
v___y_1005_ = v_a_999_;
goto v___jp_1001_;
}
}
else
{
lean_dec(v_fst_1016_);
lean_dec_ref(v___x_1015_);
v___y_1002_ = v_a_996_;
v___y_1003_ = v_a_997_;
v___y_1004_ = v_a_998_;
v___y_1005_ = v_a_999_;
goto v___jp_1001_;
}
v___jp_1001_:
{
lean_object* v___x_1006_; lean_object* v___x_1007_; 
v___x_1006_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__1, &lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__1);
v___x_1007_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Erw_x3f_extractRewriteEq_spec__0___redArg(v___x_1006_, v___y_1002_, v___y_1003_, v___y_1004_, v___y_1005_);
return v___x_1007_;
}
v___jp_1008_:
{
lean_object* v___x_1013_; lean_object* v___x_1014_; 
v___x_1013_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__3, &lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__3);
v___x_1014_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Erw_x3f_extractRewriteEq_spec__0___redArg(v___x_1013_, v___y_1009_, v___y_1010_, v___y_1011_, v___y_1012_);
return v___x_1014_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___boxed(lean_object* v_e_1081_, lean_object* v_a_1082_, lean_object* v_a_1083_, lean_object* v_a_1084_, lean_object* v_a_1085_, lean_object* v_a_1086_){
_start:
{
lean_object* v_res_1087_; 
v_res_1087_ = lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq(v_e_1081_, v_a_1082_, v_a_1083_, v_a_1084_, v_a_1085_);
lean_dec(v_a_1085_);
lean_dec_ref(v_a_1084_);
lean_dec(v_a_1083_);
lean_dec_ref(v_a_1082_);
return v_res_1087_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Erw_x3f_extractRewriteEq_spec__0(lean_object* v_00_u03b1_1088_, lean_object* v_msg_1089_, lean_object* v___y_1090_, lean_object* v___y_1091_, lean_object* v___y_1092_, lean_object* v___y_1093_){
_start:
{
lean_object* v___x_1095_; 
v___x_1095_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Erw_x3f_extractRewriteEq_spec__0___redArg(v_msg_1089_, v___y_1090_, v___y_1091_, v___y_1092_, v___y_1093_);
return v___x_1095_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Erw_x3f_extractRewriteEq_spec__0___boxed(lean_object* v_00_u03b1_1096_, lean_object* v_msg_1097_, lean_object* v___y_1098_, lean_object* v___y_1099_, lean_object* v___y_1100_, lean_object* v___y_1101_, lean_object* v___y_1102_){
_start:
{
lean_object* v_res_1103_; 
v_res_1103_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Erw_x3f_extractRewriteEq_spec__0(v_00_u03b1_1096_, v_msg_1097_, v___y_1098_, v___y_1099_, v___y_1100_, v___y_1101_);
lean_dec(v___y_1101_);
lean_dec_ref(v___y_1100_);
lean_dec(v___y_1099_);
lean_dec_ref(v___y_1098_);
return v_res_1103_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__1(void){
_start:
{
lean_object* v___x_1105_; lean_object* v___x_1106_; 
v___x_1105_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__0));
v___x_1106_ = l_Lean_stringToMessageData(v___x_1105_);
return v___x_1106_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__3(void){
_start:
{
lean_object* v___x_1108_; lean_object* v___x_1109_; 
v___x_1108_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__2));
v___x_1109_ = l_Lean_stringToMessageData(v___x_1108_);
return v___x_1109_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__6(void){
_start:
{
lean_object* v___x_1112_; lean_object* v___x_1113_; 
v___x_1112_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__5));
v___x_1113_ = l_Lean_stringToMessageData(v___x_1112_);
return v___x_1113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq(lean_object* v_e_1114_, lean_object* v_a_1115_, lean_object* v_a_1116_, lean_object* v_a_1117_, lean_object* v_a_1118_){
_start:
{
lean_object* v___y_1121_; lean_object* v___y_1122_; lean_object* v___y_1123_; lean_object* v___y_1124_; lean_object* v___y_1128_; lean_object* v___y_1129_; lean_object* v___y_1130_; lean_object* v___y_1131_; lean_object* v___x_1134_; lean_object* v_fst_1135_; 
v___x_1134_ = l_Lean_Expr_getAppFnArgs(v_e_1114_);
v_fst_1135_ = lean_ctor_get(v___x_1134_, 0);
lean_inc(v_fst_1135_);
if (lean_obj_tag(v_fst_1135_) == 0)
{
lean_object* v_snd_1136_; lean_object* v_toList_1137_; 
v_snd_1136_ = lean_ctor_get(v___x_1134_, 1);
lean_inc(v_snd_1136_);
lean_dec_ref(v___x_1134_);
v_toList_1137_ = lean_array_to_list(v_snd_1136_);
if (lean_obj_tag(v_toList_1137_) == 1)
{
lean_object* v_head_1138_; lean_object* v___x_1139_; lean_object* v_fst_1140_; 
v_head_1138_ = lean_ctor_get(v_toList_1137_, 0);
lean_inc(v_head_1138_);
lean_dec_ref_known(v_toList_1137_, 2);
v___x_1139_ = l_Lean_Expr_getAppFnArgs(v_head_1138_);
v_fst_1140_ = lean_ctor_get(v___x_1139_, 0);
lean_inc(v_fst_1140_);
if (lean_obj_tag(v_fst_1140_) == 1)
{
lean_object* v_pre_1141_; 
v_pre_1141_ = lean_ctor_get(v_fst_1140_, 0);
lean_inc(v_pre_1141_);
if (lean_obj_tag(v_pre_1141_) == 1)
{
lean_object* v_pre_1142_; 
v_pre_1142_ = lean_ctor_get(v_pre_1141_, 0);
if (lean_obj_tag(v_pre_1142_) == 0)
{
lean_object* v_snd_1143_; lean_object* v_str_1144_; lean_object* v_str_1145_; lean_object* v___x_1146_; uint8_t v___x_1147_; 
v_snd_1143_ = lean_ctor_get(v___x_1139_, 1);
lean_inc(v_snd_1143_);
lean_dec_ref(v___x_1139_);
v_str_1144_ = lean_ctor_get(v_fst_1140_, 1);
lean_inc_ref(v_str_1144_);
lean_dec_ref_known(v_fst_1140_, 2);
v_str_1145_ = lean_ctor_get(v_pre_1141_, 1);
lean_inc_ref(v_str_1145_);
lean_dec_ref_known(v_pre_1141_, 2);
v___x_1146_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__4));
v___x_1147_ = lean_string_dec_eq(v_str_1145_, v___x_1146_);
lean_dec_ref(v_str_1145_);
if (v___x_1147_ == 0)
{
lean_dec_ref(v_str_1144_);
lean_dec(v_snd_1143_);
v___y_1128_ = v_a_1115_;
v___y_1129_ = v_a_1116_;
v___y_1130_ = v_a_1117_;
v___y_1131_ = v_a_1118_;
goto v___jp_1127_;
}
else
{
lean_object* v___x_1148_; uint8_t v___x_1149_; 
v___x_1148_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__4));
v___x_1149_ = lean_string_dec_eq(v_str_1144_, v___x_1148_);
lean_dec_ref(v_str_1144_);
if (v___x_1149_ == 0)
{
lean_dec(v_snd_1143_);
v___y_1128_ = v_a_1115_;
v___y_1129_ = v_a_1116_;
v___y_1130_ = v_a_1117_;
v___y_1131_ = v_a_1118_;
goto v___jp_1127_;
}
else
{
lean_object* v___x_1150_; lean_object* v___x_1151_; uint8_t v___x_1152_; 
v___x_1150_ = lean_array_get_size(v_snd_1143_);
v___x_1151_ = lean_unsigned_to_nat(4u);
v___x_1152_ = lean_nat_dec_eq(v___x_1150_, v___x_1151_);
if (v___x_1152_ == 0)
{
lean_dec(v_snd_1143_);
v___y_1128_ = v_a_1115_;
v___y_1129_ = v_a_1116_;
v___y_1130_ = v_a_1117_;
v___y_1131_ = v_a_1118_;
goto v___jp_1127_;
}
else
{
lean_object* v___x_1153_; lean_object* v___x_1154_; lean_object* v___x_1155_; 
v___x_1153_ = lean_unsigned_to_nat(2u);
v___x_1154_ = lean_array_fget(v_snd_1143_, v___x_1153_);
lean_dec(v_snd_1143_);
lean_inc(v_a_1118_);
lean_inc_ref(v_a_1117_);
lean_inc(v_a_1116_);
lean_inc_ref(v_a_1115_);
v___x_1155_ = lean_infer_type(v___x_1154_, v_a_1115_, v_a_1116_, v_a_1117_, v_a_1118_);
if (lean_obj_tag(v___x_1155_) == 0)
{
lean_object* v_a_1156_; lean_object* v___x_1158_; uint8_t v_isShared_1159_; uint8_t v_isSharedCheck_1170_; 
v_a_1156_ = lean_ctor_get(v___x_1155_, 0);
v_isSharedCheck_1170_ = !lean_is_exclusive(v___x_1155_);
if (v_isSharedCheck_1170_ == 0)
{
v___x_1158_ = v___x_1155_;
v_isShared_1159_ = v_isSharedCheck_1170_;
goto v_resetjp_1157_;
}
else
{
lean_inc(v_a_1156_);
lean_dec(v___x_1155_);
v___x_1158_ = lean_box(0);
v_isShared_1159_ = v_isSharedCheck_1170_;
goto v_resetjp_1157_;
}
v_resetjp_1157_:
{
lean_object* v___x_1160_; lean_object* v___x_1161_; uint8_t v___x_1162_; 
v___x_1160_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq___closed__7));
v___x_1161_ = lean_unsigned_to_nat(3u);
v___x_1162_ = l_Lean_Expr_isAppOfArity(v_a_1156_, v___x_1160_, v___x_1161_);
if (v___x_1162_ == 0)
{
lean_object* v___x_1163_; lean_object* v___x_1164_; 
lean_del_object(v___x_1158_);
lean_dec(v_a_1156_);
v___x_1163_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__6, &lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__6);
v___x_1164_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Erw_x3f_extractRewriteEq_spec__0___redArg(v___x_1163_, v_a_1115_, v_a_1116_, v_a_1117_, v_a_1118_);
return v___x_1164_;
}
else
{
lean_object* v___x_1165_; lean_object* v___x_1166_; lean_object* v___x_1168_; 
v___x_1165_ = l_Lean_Expr_appFn_x21(v_a_1156_);
lean_dec(v_a_1156_);
v___x_1166_ = l_Lean_Expr_appArg_x21(v___x_1165_);
lean_dec_ref(v___x_1165_);
if (v_isShared_1159_ == 0)
{
lean_ctor_set(v___x_1158_, 0, v___x_1166_);
v___x_1168_ = v___x_1158_;
goto v_reusejp_1167_;
}
else
{
lean_object* v_reuseFailAlloc_1169_; 
v_reuseFailAlloc_1169_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1169_, 0, v___x_1166_);
v___x_1168_ = v_reuseFailAlloc_1169_;
goto v_reusejp_1167_;
}
v_reusejp_1167_:
{
return v___x_1168_;
}
}
}
}
else
{
return v___x_1155_;
}
}
}
}
}
else
{
lean_dec_ref_known(v_pre_1141_, 2);
lean_dec_ref_known(v_fst_1140_, 2);
lean_dec_ref(v___x_1139_);
v___y_1128_ = v_a_1115_;
v___y_1129_ = v_a_1116_;
v___y_1130_ = v_a_1117_;
v___y_1131_ = v_a_1118_;
goto v___jp_1127_;
}
}
else
{
lean_dec_ref_known(v_fst_1140_, 2);
lean_dec(v_pre_1141_);
lean_dec_ref(v___x_1139_);
v___y_1128_ = v_a_1115_;
v___y_1129_ = v_a_1116_;
v___y_1130_ = v_a_1117_;
v___y_1131_ = v_a_1118_;
goto v___jp_1127_;
}
}
else
{
lean_dec(v_fst_1140_);
lean_dec_ref(v___x_1139_);
v___y_1128_ = v_a_1115_;
v___y_1129_ = v_a_1116_;
v___y_1130_ = v_a_1117_;
v___y_1131_ = v_a_1118_;
goto v___jp_1127_;
}
}
else
{
lean_dec(v_toList_1137_);
v___y_1121_ = v_a_1115_;
v___y_1122_ = v_a_1116_;
v___y_1123_ = v_a_1117_;
v___y_1124_ = v_a_1118_;
goto v___jp_1120_;
}
}
else
{
lean_dec(v_fst_1135_);
lean_dec_ref(v___x_1134_);
v___y_1121_ = v_a_1115_;
v___y_1122_ = v_a_1116_;
v___y_1123_ = v_a_1117_;
v___y_1124_ = v_a_1118_;
goto v___jp_1120_;
}
v___jp_1120_:
{
lean_object* v___x_1125_; lean_object* v___x_1126_; 
v___x_1125_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__1, &lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__1);
v___x_1126_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Erw_x3f_extractRewriteEq_spec__0___redArg(v___x_1125_, v___y_1121_, v___y_1122_, v___y_1123_, v___y_1124_);
return v___x_1126_;
}
v___jp_1127_:
{
lean_object* v___x_1132_; lean_object* v___x_1133_; 
v___x_1132_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__3, &lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___closed__3);
v___x_1133_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Erw_x3f_extractRewriteEq_spec__0___redArg(v___x_1132_, v___y_1128_, v___y_1129_, v___y_1130_, v___y_1131_);
return v___x_1133_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq___boxed(lean_object* v_e_1171_, lean_object* v_a_1172_, lean_object* v_a_1173_, lean_object* v_a_1174_, lean_object* v_a_1175_, lean_object* v_a_1176_){
_start:
{
lean_object* v_res_1177_; 
v_res_1177_ = lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq(v_e_1171_, v_a_1172_, v_a_1173_, v_a_1174_, v_a_1175_);
lean_dec(v_a_1175_);
lean_dec_ref(v_a_1174_);
lean_dec(v_a_1173_);
lean_dec_ref(v_a_1172_);
return v_res_1177_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_1178_; lean_object* v___x_1179_; lean_object* v___x_1180_; 
v___x_1178_ = lean_box(0);
v___x_1179_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1180_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1180_, 0, v___x_1179_);
lean_ctor_set(v___x_1180_, 1, v___x_1178_);
return v___x_1180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__0___redArg(){
_start:
{
lean_object* v___x_1182_; lean_object* v___x_1183_; 
v___x_1182_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__0___redArg___closed__0);
v___x_1183_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1183_, 0, v___x_1182_);
return v___x_1183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__0___redArg___boxed(lean_object* v___y_1184_){
_start:
{
lean_object* v_res_1185_; 
v_res_1185_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__0___redArg();
return v_res_1185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__0(lean_object* v_00_u03b1_1186_, lean_object* v___y_1187_, lean_object* v___y_1188_, lean_object* v___y_1189_, lean_object* v___y_1190_, lean_object* v___y_1191_, lean_object* v___y_1192_, lean_object* v___y_1193_, lean_object* v___y_1194_){
_start:
{
lean_object* v___x_1196_; 
v___x_1196_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__0___redArg();
return v___x_1196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__0___boxed(lean_object* v_00_u03b1_1197_, lean_object* v___y_1198_, lean_object* v___y_1199_, lean_object* v___y_1200_, lean_object* v___y_1201_, lean_object* v___y_1202_, lean_object* v___y_1203_, lean_object* v___y_1204_, lean_object* v___y_1205_, lean_object* v___y_1206_){
_start:
{
lean_object* v_res_1207_; 
v_res_1207_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__0(v_00_u03b1_1197_, v___y_1198_, v___y_1199_, v___y_1200_, v___y_1201_, v___y_1202_, v___y_1203_, v___y_1204_, v___y_1205_);
lean_dec(v___y_1205_);
lean_dec_ref(v___y_1204_);
lean_dec(v___y_1203_);
lean_dec_ref(v___y_1202_);
lean_dec(v___y_1201_);
lean_dec_ref(v___y_1200_);
lean_dec(v___y_1199_);
lean_dec_ref(v___y_1198_);
return v_res_1207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__2___redArg(lean_object* v_e_1208_, lean_object* v___y_1209_){
_start:
{
uint8_t v___x_1211_; 
v___x_1211_ = l_Lean_Expr_hasMVar(v_e_1208_);
if (v___x_1211_ == 0)
{
lean_object* v___x_1212_; 
v___x_1212_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1212_, 0, v_e_1208_);
return v___x_1212_;
}
else
{
lean_object* v___x_1213_; lean_object* v_mctx_1214_; lean_object* v___x_1215_; lean_object* v_fst_1216_; lean_object* v_snd_1217_; lean_object* v___x_1218_; lean_object* v_cache_1219_; lean_object* v_zetaDeltaFVarIds_1220_; lean_object* v_postponed_1221_; lean_object* v_diag_1222_; lean_object* v___x_1224_; uint8_t v_isShared_1225_; uint8_t v_isSharedCheck_1231_; 
v___x_1213_ = lean_st_ref_get(v___y_1209_);
v_mctx_1214_ = lean_ctor_get(v___x_1213_, 0);
lean_inc_ref(v_mctx_1214_);
lean_dec(v___x_1213_);
v___x_1215_ = l_Lean_instantiateMVarsCore(v_mctx_1214_, v_e_1208_);
v_fst_1216_ = lean_ctor_get(v___x_1215_, 0);
lean_inc(v_fst_1216_);
v_snd_1217_ = lean_ctor_get(v___x_1215_, 1);
lean_inc(v_snd_1217_);
lean_dec_ref(v___x_1215_);
v___x_1218_ = lean_st_ref_take(v___y_1209_);
v_cache_1219_ = lean_ctor_get(v___x_1218_, 1);
v_zetaDeltaFVarIds_1220_ = lean_ctor_get(v___x_1218_, 2);
v_postponed_1221_ = lean_ctor_get(v___x_1218_, 3);
v_diag_1222_ = lean_ctor_get(v___x_1218_, 4);
v_isSharedCheck_1231_ = !lean_is_exclusive(v___x_1218_);
if (v_isSharedCheck_1231_ == 0)
{
lean_object* v_unused_1232_; 
v_unused_1232_ = lean_ctor_get(v___x_1218_, 0);
lean_dec(v_unused_1232_);
v___x_1224_ = v___x_1218_;
v_isShared_1225_ = v_isSharedCheck_1231_;
goto v_resetjp_1223_;
}
else
{
lean_inc(v_diag_1222_);
lean_inc(v_postponed_1221_);
lean_inc(v_zetaDeltaFVarIds_1220_);
lean_inc(v_cache_1219_);
lean_dec(v___x_1218_);
v___x_1224_ = lean_box(0);
v_isShared_1225_ = v_isSharedCheck_1231_;
goto v_resetjp_1223_;
}
v_resetjp_1223_:
{
lean_object* v___x_1227_; 
if (v_isShared_1225_ == 0)
{
lean_ctor_set(v___x_1224_, 0, v_snd_1217_);
v___x_1227_ = v___x_1224_;
goto v_reusejp_1226_;
}
else
{
lean_object* v_reuseFailAlloc_1230_; 
v_reuseFailAlloc_1230_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1230_, 0, v_snd_1217_);
lean_ctor_set(v_reuseFailAlloc_1230_, 1, v_cache_1219_);
lean_ctor_set(v_reuseFailAlloc_1230_, 2, v_zetaDeltaFVarIds_1220_);
lean_ctor_set(v_reuseFailAlloc_1230_, 3, v_postponed_1221_);
lean_ctor_set(v_reuseFailAlloc_1230_, 4, v_diag_1222_);
v___x_1227_ = v_reuseFailAlloc_1230_;
goto v_reusejp_1226_;
}
v_reusejp_1226_:
{
lean_object* v___x_1228_; lean_object* v___x_1229_; 
v___x_1228_ = lean_st_ref_set(v___y_1209_, v___x_1227_);
v___x_1229_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1229_, 0, v_fst_1216_);
return v___x_1229_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__2___redArg___boxed(lean_object* v_e_1233_, lean_object* v___y_1234_, lean_object* v___y_1235_){
_start:
{
lean_object* v_res_1236_; 
v_res_1236_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__2___redArg(v_e_1233_, v___y_1234_);
lean_dec(v___y_1234_);
return v_res_1236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__2(lean_object* v_e_1237_, lean_object* v___y_1238_, lean_object* v___y_1239_, lean_object* v___y_1240_, lean_object* v___y_1241_, lean_object* v___y_1242_, lean_object* v___y_1243_, lean_object* v___y_1244_, lean_object* v___y_1245_){
_start:
{
lean_object* v___x_1247_; 
v___x_1247_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__2___redArg(v_e_1237_, v___y_1243_);
return v___x_1247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__2___boxed(lean_object* v_e_1248_, lean_object* v___y_1249_, lean_object* v___y_1250_, lean_object* v___y_1251_, lean_object* v___y_1252_, lean_object* v___y_1253_, lean_object* v___y_1254_, lean_object* v___y_1255_, lean_object* v___y_1256_, lean_object* v___y_1257_){
_start:
{
lean_object* v_res_1258_; 
v_res_1258_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__2(v_e_1248_, v___y_1249_, v___y_1250_, v___y_1251_, v___y_1252_, v___y_1253_, v___y_1254_, v___y_1255_, v___y_1256_);
lean_dec(v___y_1256_);
lean_dec_ref(v___y_1255_);
lean_dec(v___y_1254_);
lean_dec_ref(v___y_1253_);
lean_dec(v___y_1252_);
lean_dec_ref(v___y_1251_);
lean_dec(v___y_1250_);
lean_dec_ref(v___y_1249_);
return v_res_1258_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___closed__4(void){
_start:
{
lean_object* v___x_1265_; lean_object* v___x_1266_; 
v___x_1265_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___closed__3));
v___x_1266_ = l_Lean_MessageData_ofFormat(v___x_1265_);
return v___x_1266_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___closed__5(void){
_start:
{
lean_object* v___x_1267_; lean_object* v___x_1268_; 
v___x_1267_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___closed__4, &lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___closed__4);
v___x_1268_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1268_, 0, v___x_1267_);
return v___x_1268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0(lean_object* v_x_1269_, lean_object* v___y_1270_, lean_object* v___y_1271_, lean_object* v___y_1272_, lean_object* v___y_1273_, lean_object* v___y_1274_, lean_object* v___y_1275_, lean_object* v___y_1276_, lean_object* v___y_1277_){
_start:
{
lean_object* v___x_1279_; lean_object* v___x_1280_; lean_object* v___x_1281_; 
v___x_1279_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___closed__1));
v___x_1280_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___closed__5, &lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___closed__5);
v___x_1281_ = l_Lean_Meta_throwTacticEx___redArg(v___x_1279_, v_x_1269_, v___x_1280_, v___y_1274_, v___y_1275_, v___y_1276_, v___y_1277_);
return v___x_1281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0___boxed(lean_object* v_x_1282_, lean_object* v___y_1283_, lean_object* v___y_1284_, lean_object* v___y_1285_, lean_object* v___y_1286_, lean_object* v___y_1287_, lean_object* v___y_1288_, lean_object* v___y_1289_, lean_object* v___y_1290_, lean_object* v___y_1291_){
_start:
{
lean_object* v_res_1292_; 
v_res_1292_ = lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__0(v_x_1282_, v___y_1283_, v___y_1284_, v___y_1285_, v___y_1286_, v___y_1287_, v___y_1288_, v___y_1289_, v___y_1290_);
lean_dec(v___y_1290_);
lean_dec_ref(v___y_1289_);
lean_dec(v___y_1288_);
lean_dec_ref(v___y_1287_);
lean_dec(v___y_1286_);
lean_dec_ref(v___y_1285_);
lean_dec(v___y_1284_);
lean_dec_ref(v___y_1283_);
return v_res_1292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__1_spec__1___redArg(lean_object* v_ref_1293_, lean_object* v_msgData_1294_, uint8_t v_severity_1295_, uint8_t v_isSilent_1296_, lean_object* v___y_1297_, lean_object* v___y_1298_, lean_object* v___y_1299_, lean_object* v___y_1300_){
_start:
{
lean_object* v___y_1303_; uint8_t v___y_1304_; lean_object* v___y_1305_; lean_object* v___y_1306_; lean_object* v___y_1307_; uint8_t v___y_1308_; lean_object* v___y_1309_; lean_object* v___y_1310_; lean_object* v___y_1311_; lean_object* v___y_1339_; uint8_t v___y_1340_; lean_object* v___y_1341_; lean_object* v___y_1342_; uint8_t v___y_1343_; uint8_t v___y_1344_; lean_object* v___y_1345_; lean_object* v___y_1346_; lean_object* v___y_1364_; lean_object* v___y_1365_; uint8_t v___y_1366_; lean_object* v___y_1367_; lean_object* v___y_1368_; uint8_t v___y_1369_; uint8_t v___y_1370_; lean_object* v___y_1371_; lean_object* v___y_1375_; lean_object* v___y_1376_; lean_object* v___y_1377_; lean_object* v___y_1378_; uint8_t v___y_1379_; uint8_t v___y_1380_; uint8_t v___y_1381_; uint8_t v___x_1386_; lean_object* v___y_1388_; lean_object* v___y_1389_; lean_object* v___y_1390_; lean_object* v___y_1391_; uint8_t v___y_1392_; uint8_t v___y_1393_; uint8_t v___y_1394_; uint8_t v___y_1396_; uint8_t v___x_1411_; 
v___x_1386_ = 2;
v___x_1411_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1295_, v___x_1386_);
if (v___x_1411_ == 0)
{
v___y_1396_ = v___x_1411_;
goto v___jp_1395_;
}
else
{
uint8_t v___x_1412_; 
lean_inc_ref(v_msgData_1294_);
v___x_1412_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_1294_);
v___y_1396_ = v___x_1412_;
goto v___jp_1395_;
}
v___jp_1302_:
{
lean_object* v___x_1312_; lean_object* v_currNamespace_1313_; lean_object* v_openDecls_1314_; lean_object* v_env_1315_; lean_object* v_nextMacroScope_1316_; lean_object* v_ngen_1317_; lean_object* v_auxDeclNGen_1318_; lean_object* v_traceState_1319_; lean_object* v_cache_1320_; lean_object* v_messages_1321_; lean_object* v_infoState_1322_; lean_object* v_snapshotTasks_1323_; lean_object* v___x_1325_; uint8_t v_isShared_1326_; uint8_t v_isSharedCheck_1337_; 
v___x_1312_ = lean_st_ref_take(v___y_1311_);
v_currNamespace_1313_ = lean_ctor_get(v___y_1310_, 6);
v_openDecls_1314_ = lean_ctor_get(v___y_1310_, 7);
v_env_1315_ = lean_ctor_get(v___x_1312_, 0);
v_nextMacroScope_1316_ = lean_ctor_get(v___x_1312_, 1);
v_ngen_1317_ = lean_ctor_get(v___x_1312_, 2);
v_auxDeclNGen_1318_ = lean_ctor_get(v___x_1312_, 3);
v_traceState_1319_ = lean_ctor_get(v___x_1312_, 4);
v_cache_1320_ = lean_ctor_get(v___x_1312_, 5);
v_messages_1321_ = lean_ctor_get(v___x_1312_, 6);
v_infoState_1322_ = lean_ctor_get(v___x_1312_, 7);
v_snapshotTasks_1323_ = lean_ctor_get(v___x_1312_, 8);
v_isSharedCheck_1337_ = !lean_is_exclusive(v___x_1312_);
if (v_isSharedCheck_1337_ == 0)
{
v___x_1325_ = v___x_1312_;
v_isShared_1326_ = v_isSharedCheck_1337_;
goto v_resetjp_1324_;
}
else
{
lean_inc(v_snapshotTasks_1323_);
lean_inc(v_infoState_1322_);
lean_inc(v_messages_1321_);
lean_inc(v_cache_1320_);
lean_inc(v_traceState_1319_);
lean_inc(v_auxDeclNGen_1318_);
lean_inc(v_ngen_1317_);
lean_inc(v_nextMacroScope_1316_);
lean_inc(v_env_1315_);
lean_dec(v___x_1312_);
v___x_1325_ = lean_box(0);
v_isShared_1326_ = v_isSharedCheck_1337_;
goto v_resetjp_1324_;
}
v_resetjp_1324_:
{
lean_object* v___x_1327_; lean_object* v___x_1328_; lean_object* v___x_1329_; lean_object* v___x_1330_; lean_object* v___x_1332_; 
lean_inc(v_openDecls_1314_);
lean_inc(v_currNamespace_1313_);
v___x_1327_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1327_, 0, v_currNamespace_1313_);
lean_ctor_set(v___x_1327_, 1, v_openDecls_1314_);
v___x_1328_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1328_, 0, v___x_1327_);
lean_ctor_set(v___x_1328_, 1, v___y_1307_);
lean_inc_ref(v___y_1309_);
lean_inc_ref(v___y_1305_);
v___x_1329_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_1329_, 0, v___y_1305_);
lean_ctor_set(v___x_1329_, 1, v___y_1306_);
lean_ctor_set(v___x_1329_, 2, v___y_1303_);
lean_ctor_set(v___x_1329_, 3, v___y_1309_);
lean_ctor_set(v___x_1329_, 4, v___x_1328_);
lean_ctor_set_uint8(v___x_1329_, sizeof(void*)*5, v___y_1308_);
lean_ctor_set_uint8(v___x_1329_, sizeof(void*)*5 + 1, v___y_1304_);
lean_ctor_set_uint8(v___x_1329_, sizeof(void*)*5 + 2, v_isSilent_1296_);
v___x_1330_ = l_Lean_MessageLog_add(v___x_1329_, v_messages_1321_);
if (v_isShared_1326_ == 0)
{
lean_ctor_set(v___x_1325_, 6, v___x_1330_);
v___x_1332_ = v___x_1325_;
goto v_reusejp_1331_;
}
else
{
lean_object* v_reuseFailAlloc_1336_; 
v_reuseFailAlloc_1336_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1336_, 0, v_env_1315_);
lean_ctor_set(v_reuseFailAlloc_1336_, 1, v_nextMacroScope_1316_);
lean_ctor_set(v_reuseFailAlloc_1336_, 2, v_ngen_1317_);
lean_ctor_set(v_reuseFailAlloc_1336_, 3, v_auxDeclNGen_1318_);
lean_ctor_set(v_reuseFailAlloc_1336_, 4, v_traceState_1319_);
lean_ctor_set(v_reuseFailAlloc_1336_, 5, v_cache_1320_);
lean_ctor_set(v_reuseFailAlloc_1336_, 6, v___x_1330_);
lean_ctor_set(v_reuseFailAlloc_1336_, 7, v_infoState_1322_);
lean_ctor_set(v_reuseFailAlloc_1336_, 8, v_snapshotTasks_1323_);
v___x_1332_ = v_reuseFailAlloc_1336_;
goto v_reusejp_1331_;
}
v_reusejp_1331_:
{
lean_object* v___x_1333_; lean_object* v___x_1334_; lean_object* v___x_1335_; 
v___x_1333_ = lean_st_ref_set(v___y_1311_, v___x_1332_);
v___x_1334_ = lean_box(0);
v___x_1335_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1335_, 0, v___x_1334_);
return v___x_1335_;
}
}
}
v___jp_1338_:
{
lean_object* v___x_1347_; lean_object* v___x_1348_; lean_object* v_a_1349_; lean_object* v___x_1351_; uint8_t v_isShared_1352_; uint8_t v_isSharedCheck_1362_; 
v___x_1347_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_1294_);
v___x_1348_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3_spec__4(v___x_1347_, v___y_1297_, v___y_1298_, v___y_1299_, v___y_1300_);
v_a_1349_ = lean_ctor_get(v___x_1348_, 0);
v_isSharedCheck_1362_ = !lean_is_exclusive(v___x_1348_);
if (v_isSharedCheck_1362_ == 0)
{
v___x_1351_ = v___x_1348_;
v_isShared_1352_ = v_isSharedCheck_1362_;
goto v_resetjp_1350_;
}
else
{
lean_inc(v_a_1349_);
lean_dec(v___x_1348_);
v___x_1351_ = lean_box(0);
v_isShared_1352_ = v_isSharedCheck_1362_;
goto v_resetjp_1350_;
}
v_resetjp_1350_:
{
lean_object* v___x_1353_; lean_object* v___x_1354_; lean_object* v___x_1355_; lean_object* v___x_1356_; 
lean_inc_ref_n(v___y_1341_, 2);
v___x_1353_ = l_Lean_FileMap_toPosition(v___y_1341_, v___y_1345_);
lean_dec(v___y_1345_);
v___x_1354_ = l_Lean_FileMap_toPosition(v___y_1341_, v___y_1346_);
lean_dec(v___y_1346_);
v___x_1355_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1355_, 0, v___x_1354_);
v___x_1356_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__20));
if (v___y_1343_ == 0)
{
lean_del_object(v___x_1351_);
lean_dec_ref(v___y_1339_);
v___y_1303_ = v___x_1355_;
v___y_1304_ = v___y_1340_;
v___y_1305_ = v___y_1342_;
v___y_1306_ = v___x_1353_;
v___y_1307_ = v_a_1349_;
v___y_1308_ = v___y_1344_;
v___y_1309_ = v___x_1356_;
v___y_1310_ = v___y_1299_;
v___y_1311_ = v___y_1300_;
goto v___jp_1302_;
}
else
{
uint8_t v___x_1357_; 
lean_inc(v_a_1349_);
v___x_1357_ = l_Lean_MessageData_hasTag(v___y_1339_, v_a_1349_);
if (v___x_1357_ == 0)
{
lean_object* v___x_1358_; lean_object* v___x_1360_; 
lean_dec_ref_known(v___x_1355_, 1);
lean_dec_ref(v___x_1353_);
lean_dec(v_a_1349_);
v___x_1358_ = lean_box(0);
if (v_isShared_1352_ == 0)
{
lean_ctor_set(v___x_1351_, 0, v___x_1358_);
v___x_1360_ = v___x_1351_;
goto v_reusejp_1359_;
}
else
{
lean_object* v_reuseFailAlloc_1361_; 
v_reuseFailAlloc_1361_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1361_, 0, v___x_1358_);
v___x_1360_ = v_reuseFailAlloc_1361_;
goto v_reusejp_1359_;
}
v_reusejp_1359_:
{
return v___x_1360_;
}
}
else
{
lean_del_object(v___x_1351_);
v___y_1303_ = v___x_1355_;
v___y_1304_ = v___y_1340_;
v___y_1305_ = v___y_1342_;
v___y_1306_ = v___x_1353_;
v___y_1307_ = v_a_1349_;
v___y_1308_ = v___y_1344_;
v___y_1309_ = v___x_1356_;
v___y_1310_ = v___y_1299_;
v___y_1311_ = v___y_1300_;
goto v___jp_1302_;
}
}
}
}
v___jp_1363_:
{
lean_object* v___x_1372_; 
v___x_1372_ = l_Lean_Syntax_getTailPos_x3f(v___y_1368_, v___y_1370_);
lean_dec(v___y_1368_);
if (lean_obj_tag(v___x_1372_) == 0)
{
lean_inc(v___y_1371_);
v___y_1339_ = v___y_1364_;
v___y_1340_ = v___y_1366_;
v___y_1341_ = v___y_1365_;
v___y_1342_ = v___y_1367_;
v___y_1343_ = v___y_1369_;
v___y_1344_ = v___y_1370_;
v___y_1345_ = v___y_1371_;
v___y_1346_ = v___y_1371_;
goto v___jp_1338_;
}
else
{
lean_object* v_val_1373_; 
v_val_1373_ = lean_ctor_get(v___x_1372_, 0);
lean_inc(v_val_1373_);
lean_dec_ref_known(v___x_1372_, 1);
v___y_1339_ = v___y_1364_;
v___y_1340_ = v___y_1366_;
v___y_1341_ = v___y_1365_;
v___y_1342_ = v___y_1367_;
v___y_1343_ = v___y_1369_;
v___y_1344_ = v___y_1370_;
v___y_1345_ = v___y_1371_;
v___y_1346_ = v_val_1373_;
goto v___jp_1338_;
}
}
v___jp_1374_:
{
lean_object* v_ref_1382_; lean_object* v___x_1383_; 
v_ref_1382_ = l_Lean_replaceRef(v_ref_1293_, v___y_1376_);
v___x_1383_ = l_Lean_Syntax_getPos_x3f(v_ref_1382_, v___y_1380_);
if (lean_obj_tag(v___x_1383_) == 0)
{
lean_object* v___x_1384_; 
v___x_1384_ = lean_unsigned_to_nat(0u);
v___y_1364_ = v___y_1375_;
v___y_1365_ = v___y_1377_;
v___y_1366_ = v___y_1381_;
v___y_1367_ = v___y_1378_;
v___y_1368_ = v_ref_1382_;
v___y_1369_ = v___y_1379_;
v___y_1370_ = v___y_1380_;
v___y_1371_ = v___x_1384_;
goto v___jp_1363_;
}
else
{
lean_object* v_val_1385_; 
v_val_1385_ = lean_ctor_get(v___x_1383_, 0);
lean_inc(v_val_1385_);
lean_dec_ref_known(v___x_1383_, 1);
v___y_1364_ = v___y_1375_;
v___y_1365_ = v___y_1377_;
v___y_1366_ = v___y_1381_;
v___y_1367_ = v___y_1378_;
v___y_1368_ = v_ref_1382_;
v___y_1369_ = v___y_1379_;
v___y_1370_ = v___y_1380_;
v___y_1371_ = v_val_1385_;
goto v___jp_1363_;
}
}
v___jp_1387_:
{
if (v___y_1394_ == 0)
{
v___y_1375_ = v___y_1389_;
v___y_1376_ = v___y_1388_;
v___y_1377_ = v___y_1390_;
v___y_1378_ = v___y_1391_;
v___y_1379_ = v___y_1392_;
v___y_1380_ = v___y_1393_;
v___y_1381_ = v_severity_1295_;
goto v___jp_1374_;
}
else
{
v___y_1375_ = v___y_1389_;
v___y_1376_ = v___y_1388_;
v___y_1377_ = v___y_1390_;
v___y_1378_ = v___y_1391_;
v___y_1379_ = v___y_1392_;
v___y_1380_ = v___y_1393_;
v___y_1381_ = v___x_1386_;
goto v___jp_1374_;
}
}
v___jp_1395_:
{
if (v___y_1396_ == 0)
{
lean_object* v_fileName_1397_; lean_object* v_fileMap_1398_; lean_object* v_options_1399_; lean_object* v_ref_1400_; uint8_t v_suppressElabErrors_1401_; lean_object* v___x_1402_; lean_object* v___x_1403_; lean_object* v___f_1404_; uint8_t v___x_1405_; uint8_t v___x_1406_; 
v_fileName_1397_ = lean_ctor_get(v___y_1299_, 0);
v_fileMap_1398_ = lean_ctor_get(v___y_1299_, 1);
v_options_1399_ = lean_ctor_get(v___y_1299_, 2);
v_ref_1400_ = lean_ctor_get(v___y_1299_, 5);
v_suppressElabErrors_1401_ = lean_ctor_get_uint8(v___y_1299_, sizeof(void*)*14 + 1);
v___x_1402_ = lean_box(v___y_1396_);
v___x_1403_ = lean_box(v_suppressElabErrors_1401_);
v___f_1404_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__3_spec__3___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1404_, 0, v___x_1402_);
lean_closure_set(v___f_1404_, 1, v___x_1403_);
v___x_1405_ = 1;
v___x_1406_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1295_, v___x_1405_);
if (v___x_1406_ == 0)
{
v___y_1388_ = v_ref_1400_;
v___y_1389_ = v___f_1404_;
v___y_1390_ = v_fileMap_1398_;
v___y_1391_ = v_fileName_1397_;
v___y_1392_ = v_suppressElabErrors_1401_;
v___y_1393_ = v___y_1396_;
v___y_1394_ = v___x_1406_;
goto v___jp_1387_;
}
else
{
lean_object* v___x_1407_; uint8_t v___x_1408_; 
v___x_1407_ = l_Lean_warningAsError;
v___x_1408_ = lp_mathlib_Lean_Option_get___at___00Mathlib_Tactic_Erw_x3f_logDiffs_spec__1(v_options_1399_, v___x_1407_);
v___y_1388_ = v_ref_1400_;
v___y_1389_ = v___f_1404_;
v___y_1390_ = v_fileMap_1398_;
v___y_1391_ = v_fileName_1397_;
v___y_1392_ = v_suppressElabErrors_1401_;
v___y_1393_ = v___y_1396_;
v___y_1394_ = v___x_1408_;
goto v___jp_1387_;
}
}
else
{
lean_object* v___x_1409_; lean_object* v___x_1410_; 
lean_dec_ref(v_msgData_1294_);
v___x_1409_ = lean_box(0);
v___x_1410_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1410_, 0, v___x_1409_);
return v___x_1410_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__1_spec__1___redArg___boxed(lean_object* v_ref_1413_, lean_object* v_msgData_1414_, lean_object* v_severity_1415_, lean_object* v_isSilent_1416_, lean_object* v___y_1417_, lean_object* v___y_1418_, lean_object* v___y_1419_, lean_object* v___y_1420_, lean_object* v___y_1421_){
_start:
{
uint8_t v_severity_boxed_1422_; uint8_t v_isSilent_boxed_1423_; lean_object* v_res_1424_; 
v_severity_boxed_1422_ = lean_unbox(v_severity_1415_);
v_isSilent_boxed_1423_ = lean_unbox(v_isSilent_1416_);
v_res_1424_ = lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__1_spec__1___redArg(v_ref_1413_, v_msgData_1414_, v_severity_boxed_1422_, v_isSilent_boxed_1423_, v___y_1417_, v___y_1418_, v___y_1419_, v___y_1420_);
lean_dec(v___y_1420_);
lean_dec_ref(v___y_1419_);
lean_dec(v___y_1418_);
lean_dec_ref(v___y_1417_);
lean_dec(v_ref_1413_);
return v_res_1424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__1(lean_object* v_ref_1425_, lean_object* v_msgData_1426_, lean_object* v___y_1427_, lean_object* v___y_1428_, lean_object* v___y_1429_, lean_object* v___y_1430_, lean_object* v___y_1431_, lean_object* v___y_1432_, lean_object* v___y_1433_, lean_object* v___y_1434_){
_start:
{
uint8_t v___x_1436_; uint8_t v___x_1437_; lean_object* v___x_1438_; 
v___x_1436_ = 0;
v___x_1437_ = 0;
v___x_1438_ = lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__1_spec__1___redArg(v_ref_1425_, v_msgData_1426_, v___x_1436_, v___x_1437_, v___y_1431_, v___y_1432_, v___y_1433_, v___y_1434_);
return v___x_1438_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__1___boxed(lean_object* v_ref_1439_, lean_object* v_msgData_1440_, lean_object* v___y_1441_, lean_object* v___y_1442_, lean_object* v___y_1443_, lean_object* v___y_1444_, lean_object* v___y_1445_, lean_object* v___y_1446_, lean_object* v___y_1447_, lean_object* v___y_1448_, lean_object* v___y_1449_){
_start:
{
lean_object* v_res_1450_; 
v_res_1450_ = lp_mathlib_Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__1(v_ref_1439_, v_msgData_1440_, v___y_1441_, v___y_1442_, v___y_1443_, v___y_1444_, v___y_1445_, v___y_1446_, v___y_1447_, v___y_1448_);
lean_dec(v___y_1448_);
lean_dec_ref(v___y_1447_);
lean_dec(v___y_1446_);
lean_dec_ref(v___y_1445_);
lean_dec(v___y_1444_);
lean_dec_ref(v___y_1443_);
lean_dec(v___y_1442_);
lean_dec_ref(v___y_1441_);
lean_dec(v_ref_1439_);
return v_res_1450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__3(lean_object* v_a_1451_, lean_object* v_a_1452_){
_start:
{
if (lean_obj_tag(v_a_1451_) == 0)
{
lean_object* v___x_1453_; 
v___x_1453_ = l_List_reverse___redArg(v_a_1452_);
return v___x_1453_;
}
else
{
lean_object* v_head_1454_; lean_object* v_tail_1455_; lean_object* v___x_1457_; uint8_t v_isShared_1458_; uint8_t v_isSharedCheck_1465_; 
v_head_1454_ = lean_ctor_get(v_a_1451_, 0);
v_tail_1455_ = lean_ctor_get(v_a_1451_, 1);
v_isSharedCheck_1465_ = !lean_is_exclusive(v_a_1451_);
if (v_isSharedCheck_1465_ == 0)
{
v___x_1457_ = v_a_1451_;
v_isShared_1458_ = v_isSharedCheck_1465_;
goto v_resetjp_1456_;
}
else
{
lean_inc(v_tail_1455_);
lean_inc(v_head_1454_);
lean_dec(v_a_1451_);
v___x_1457_ = lean_box(0);
v_isShared_1458_ = v_isSharedCheck_1465_;
goto v_resetjp_1456_;
}
v_resetjp_1456_:
{
lean_object* v___x_1459_; lean_object* v___x_1460_; lean_object* v___x_1462_; 
v___x_1459_ = lean_box(0);
v___x_1460_ = lean_apply_1(v_head_1454_, v___x_1459_);
if (v_isShared_1458_ == 0)
{
lean_ctor_set(v___x_1457_, 1, v_a_1452_);
lean_ctor_set(v___x_1457_, 0, v___x_1460_);
v___x_1462_ = v___x_1457_;
goto v_reusejp_1461_;
}
else
{
lean_object* v_reuseFailAlloc_1464_; 
v_reuseFailAlloc_1464_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1464_, 0, v___x_1460_);
lean_ctor_set(v_reuseFailAlloc_1464_, 1, v_a_1452_);
v___x_1462_ = v_reuseFailAlloc_1464_;
goto v_reusejp_1461_;
}
v_reusejp_1461_:
{
v_a_1451_ = v_tail_1455_;
v_a_1452_ = v___x_1462_;
goto _start;
}
}
}
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__1(void){
_start:
{
lean_object* v___x_1467_; lean_object* v___x_1468_; 
v___x_1467_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__0));
v___x_1468_ = l_Lean_stringToMessageData(v___x_1467_);
return v___x_1468_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__3(void){
_start:
{
lean_object* v___x_1470_; lean_object* v___x_1471_; 
v___x_1470_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__2));
v___x_1471_ = l_Lean_stringToMessageData(v___x_1470_);
return v___x_1471_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__5(void){
_start:
{
lean_object* v___x_1473_; lean_object* v___x_1474_; 
v___x_1473_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__4));
v___x_1474_ = l_Lean_stringToMessageData(v___x_1473_);
return v___x_1474_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__8(void){
_start:
{
lean_object* v___x_1478_; lean_object* v___x_1479_; 
v___x_1478_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__7));
v___x_1479_ = l_Lean_MessageData_ofFormat(v___x_1478_);
return v___x_1479_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1(lean_object* v_term_1480_, uint8_t v_symm_1481_, lean_object* v___x_1482_, lean_object* v_tk_1483_, lean_object* v___x_1484_, uint8_t v___y_1485_, lean_object* v_loc_1486_, lean_object* v___y_1487_, lean_object* v___y_1488_, lean_object* v___y_1489_, lean_object* v___y_1490_, lean_object* v___y_1491_, lean_object* v___y_1492_, lean_object* v___y_1493_, lean_object* v___y_1494_){
_start:
{
lean_object* v___x_1496_; 
v___x_1496_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_1488_, v___y_1491_, v___y_1492_, v___y_1493_, v___y_1494_);
if (lean_obj_tag(v___x_1496_) == 0)
{
lean_object* v_a_1497_; lean_object* v___x_1498_; 
v_a_1497_ = lean_ctor_get(v___x_1496_, 0);
lean_inc(v_a_1497_);
lean_dec_ref_known(v___x_1496_, 1);
lean_inc(v_loc_1486_);
v___x_1498_ = l_Lean_Elab_Tactic_rewriteLocalDecl(v_term_1480_, v_symm_1481_, v_loc_1486_, v___x_1482_, v___y_1487_, v___y_1488_, v___y_1489_, v___y_1490_, v___y_1491_, v___y_1492_, v___y_1493_, v___y_1494_);
if (lean_obj_tag(v___x_1498_) == 0)
{
lean_object* v___x_1499_; 
lean_dec_ref_known(v___x_1498_, 1);
v___x_1499_ = l_Lean_FVarId_getDecl___redArg(v_loc_1486_, v___y_1491_, v___y_1493_, v___y_1494_);
if (lean_obj_tag(v___x_1499_) == 0)
{
lean_object* v_a_1500_; lean_object* v___x_1501_; lean_object* v___x_1502_; lean_object* v_a_1503_; lean_object* v_fileName_1504_; lean_object* v_fileMap_1505_; lean_object* v_options_1506_; lean_object* v_currRecDepth_1507_; lean_object* v_maxRecDepth_1508_; lean_object* v_ref_1509_; lean_object* v_currNamespace_1510_; lean_object* v_openDecls_1511_; lean_object* v_initHeartbeats_1512_; lean_object* v_maxHeartbeats_1513_; lean_object* v_quotContext_1514_; lean_object* v_currMacroScope_1515_; uint8_t v_diag_1516_; lean_object* v_cancelTk_x3f_1517_; uint8_t v_suppressElabErrors_1518_; lean_object* v_inheritedTraceOptions_1519_; lean_object* v___x_1520_; lean_object* v_ref_1521_; lean_object* v___x_1522_; lean_object* v___x_1523_; 
v_a_1500_ = lean_ctor_get(v___x_1499_, 0);
lean_inc(v_a_1500_);
lean_dec_ref_known(v___x_1499_, 1);
v___x_1501_ = l_Lean_Expr_mvar___override(v_a_1497_);
v___x_1502_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__2___redArg(v___x_1501_, v___y_1492_);
v_a_1503_ = lean_ctor_get(v___x_1502_, 0);
lean_inc(v_a_1503_);
lean_dec_ref(v___x_1502_);
v_fileName_1504_ = lean_ctor_get(v___y_1493_, 0);
v_fileMap_1505_ = lean_ctor_get(v___y_1493_, 1);
v_options_1506_ = lean_ctor_get(v___y_1493_, 2);
v_currRecDepth_1507_ = lean_ctor_get(v___y_1493_, 3);
v_maxRecDepth_1508_ = lean_ctor_get(v___y_1493_, 4);
v_ref_1509_ = lean_ctor_get(v___y_1493_, 5);
v_currNamespace_1510_ = lean_ctor_get(v___y_1493_, 6);
v_openDecls_1511_ = lean_ctor_get(v___y_1493_, 7);
v_initHeartbeats_1512_ = lean_ctor_get(v___y_1493_, 8);
v_maxHeartbeats_1513_ = lean_ctor_get(v___y_1493_, 9);
v_quotContext_1514_ = lean_ctor_get(v___y_1493_, 10);
v_currMacroScope_1515_ = lean_ctor_get(v___y_1493_, 11);
v_diag_1516_ = lean_ctor_get_uint8(v___y_1493_, sizeof(void*)*14);
v_cancelTk_x3f_1517_ = lean_ctor_get(v___y_1493_, 12);
v_suppressElabErrors_1518_ = lean_ctor_get_uint8(v___y_1493_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1519_ = lean_ctor_get(v___y_1493_, 13);
v___x_1520_ = l_Lean_Expr_headBeta(v_a_1503_);
v_ref_1521_ = l_Lean_replaceRef(v_tk_1483_, v_ref_1509_);
lean_inc_ref(v_inheritedTraceOptions_1519_);
lean_inc(v_cancelTk_x3f_1517_);
lean_inc(v_currMacroScope_1515_);
lean_inc(v_quotContext_1514_);
lean_inc(v_maxHeartbeats_1513_);
lean_inc(v_initHeartbeats_1512_);
lean_inc(v_openDecls_1511_);
lean_inc(v_currNamespace_1510_);
lean_inc(v_maxRecDepth_1508_);
lean_inc(v_currRecDepth_1507_);
lean_inc_ref(v_options_1506_);
lean_inc_ref(v_fileMap_1505_);
lean_inc_ref(v_fileName_1504_);
v___x_1522_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1522_, 0, v_fileName_1504_);
lean_ctor_set(v___x_1522_, 1, v_fileMap_1505_);
lean_ctor_set(v___x_1522_, 2, v_options_1506_);
lean_ctor_set(v___x_1522_, 3, v_currRecDepth_1507_);
lean_ctor_set(v___x_1522_, 4, v_maxRecDepth_1508_);
lean_ctor_set(v___x_1522_, 5, v_ref_1521_);
lean_ctor_set(v___x_1522_, 6, v_currNamespace_1510_);
lean_ctor_set(v___x_1522_, 7, v_openDecls_1511_);
lean_ctor_set(v___x_1522_, 8, v_initHeartbeats_1512_);
lean_ctor_set(v___x_1522_, 9, v_maxHeartbeats_1513_);
lean_ctor_set(v___x_1522_, 10, v_quotContext_1514_);
lean_ctor_set(v___x_1522_, 11, v_currMacroScope_1515_);
lean_ctor_set(v___x_1522_, 12, v_cancelTk_x3f_1517_);
lean_ctor_set(v___x_1522_, 13, v_inheritedTraceOptions_1519_);
lean_ctor_set_uint8(v___x_1522_, sizeof(void*)*14, v_diag_1516_);
lean_ctor_set_uint8(v___x_1522_, sizeof(void*)*14 + 1, v_suppressElabErrors_1518_);
v___x_1523_ = lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteHypEq(v___x_1520_, v___y_1491_, v___y_1492_, v___x_1522_, v___y_1494_);
lean_dec_ref_known(v___x_1522_, 14);
if (lean_obj_tag(v___x_1523_) == 0)
{
lean_object* v_a_1524_; lean_object* v___x_1525_; lean_object* v___x_1526_; lean_object* v___x_1527_; 
v_a_1524_ = lean_ctor_get(v___x_1523_, 0);
lean_inc_n(v_a_1524_, 2);
lean_dec_ref_known(v___x_1523_, 1);
v___x_1525_ = l_Lean_LocalDecl_type(v_a_1500_);
v___x_1526_ = lean_mk_empty_array_with_capacity(v___x_1484_);
lean_inc_ref(v___x_1525_);
v___x_1527_ = lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs(v_tk_1483_, v___x_1525_, v_a_1524_, v___x_1526_, v___y_1491_, v___y_1492_, v___y_1493_, v___y_1494_);
if (lean_obj_tag(v___x_1527_) == 0)
{
lean_object* v_a_1528_; lean_object* v___x_1530_; uint8_t v_isShared_1531_; uint8_t v_isSharedCheck_1564_; 
v_a_1528_ = lean_ctor_get(v___x_1527_, 0);
v_isSharedCheck_1564_ = !lean_is_exclusive(v___x_1527_);
if (v_isSharedCheck_1564_ == 0)
{
v___x_1530_ = v___x_1527_;
v_isShared_1531_ = v_isSharedCheck_1564_;
goto v_resetjp_1529_;
}
else
{
lean_inc(v_a_1528_);
lean_dec(v___x_1527_);
v___x_1530_ = lean_box(0);
v_isShared_1531_ = v_isSharedCheck_1564_;
goto v_resetjp_1529_;
}
v_resetjp_1529_:
{
if (v___y_1485_ == 0)
{
lean_object* v___x_1532_; lean_object* v___x_1534_; 
lean_dec(v_a_1528_);
lean_dec_ref(v___x_1525_);
lean_dec(v_a_1524_);
lean_dec(v_a_1500_);
v___x_1532_ = lean_box(0);
if (v_isShared_1531_ == 0)
{
lean_ctor_set(v___x_1530_, 0, v___x_1532_);
v___x_1534_ = v___x_1530_;
goto v_reusejp_1533_;
}
else
{
lean_object* v_reuseFailAlloc_1535_; 
v_reuseFailAlloc_1535_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1535_, 0, v___x_1532_);
v___x_1534_ = v_reuseFailAlloc_1535_;
goto v_reusejp_1533_;
}
v_reusejp_1533_:
{
return v___x_1534_;
}
}
else
{
lean_object* v_snd_1536_; lean_object* v___x_1538_; uint8_t v_isShared_1539_; uint8_t v_isSharedCheck_1562_; 
lean_del_object(v___x_1530_);
v_snd_1536_ = lean_ctor_get(v_a_1528_, 1);
v_isSharedCheck_1562_ = !lean_is_exclusive(v_a_1528_);
if (v_isSharedCheck_1562_ == 0)
{
lean_object* v_unused_1563_; 
v_unused_1563_ = lean_ctor_get(v_a_1528_, 0);
lean_dec(v_unused_1563_);
v___x_1538_ = v_a_1528_;
v_isShared_1539_ = v_isSharedCheck_1562_;
goto v_resetjp_1537_;
}
else
{
lean_inc(v_snd_1536_);
lean_dec(v_a_1528_);
v___x_1538_ = lean_box(0);
v_isShared_1539_ = v_isSharedCheck_1562_;
goto v_resetjp_1537_;
}
v_resetjp_1537_:
{
lean_object* v___x_1540_; lean_object* v___x_1541_; lean_object* v___x_1542_; lean_object* v___x_1544_; 
v___x_1540_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__1, &lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__1);
v___x_1541_ = l_Lean_LocalDecl_toExpr(v_a_1500_);
v___x_1542_ = l_Lean_MessageData_ofExpr(v___x_1541_);
if (v_isShared_1539_ == 0)
{
lean_ctor_set_tag(v___x_1538_, 7);
lean_ctor_set(v___x_1538_, 1, v___x_1542_);
lean_ctor_set(v___x_1538_, 0, v___x_1540_);
v___x_1544_ = v___x_1538_;
goto v_reusejp_1543_;
}
else
{
lean_object* v_reuseFailAlloc_1561_; 
v_reuseFailAlloc_1561_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1561_, 0, v___x_1540_);
lean_ctor_set(v_reuseFailAlloc_1561_, 1, v___x_1542_);
v___x_1544_ = v_reuseFailAlloc_1561_;
goto v_reusejp_1543_;
}
v_reusejp_1543_:
{
lean_object* v___x_1545_; lean_object* v___x_1546_; lean_object* v___x_1547_; lean_object* v___x_1548_; lean_object* v___x_1549_; lean_object* v___x_1550_; lean_object* v___x_1551_; lean_object* v___x_1552_; lean_object* v___x_1553_; lean_object* v___x_1554_; lean_object* v___x_1555_; lean_object* v___x_1556_; lean_object* v___x_1557_; lean_object* v___x_1558_; lean_object* v___x_1559_; lean_object* v___x_1560_; 
v___x_1545_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__3, &lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__3);
v___x_1546_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1546_, 0, v___x_1544_);
lean_ctor_set(v___x_1546_, 1, v___x_1545_);
v___x_1547_ = l_Lean_MessageData_ofExpr(v___x_1525_);
v___x_1548_ = l_Lean_indentD(v___x_1547_);
v___x_1549_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1549_, 0, v___x_1546_);
lean_ctor_set(v___x_1549_, 1, v___x_1548_);
v___x_1550_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__5, &lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__5);
v___x_1551_ = l_Lean_MessageData_ofExpr(v_a_1524_);
v___x_1552_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1552_, 0, v___x_1550_);
lean_ctor_set(v___x_1552_, 1, v___x_1551_);
v___x_1553_ = lean_array_to_list(v_snd_1536_);
v___x_1554_ = lean_box(0);
v___x_1555_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__3(v___x_1553_, v___x_1554_);
v___x_1556_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1556_, 0, v___x_1552_);
lean_ctor_set(v___x_1556_, 1, v___x_1555_);
v___x_1557_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1557_, 0, v___x_1549_);
lean_ctor_set(v___x_1557_, 1, v___x_1556_);
v___x_1558_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__8, &lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__8);
v___x_1559_ = l_Lean_MessageData_joinSep(v___x_1557_, v___x_1558_);
v___x_1560_ = lp_mathlib_Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__1(v_tk_1483_, v___x_1559_, v___y_1487_, v___y_1488_, v___y_1489_, v___y_1490_, v___y_1491_, v___y_1492_, v___y_1493_, v___y_1494_);
return v___x_1560_;
}
}
}
}
}
else
{
lean_object* v_a_1565_; lean_object* v___x_1567_; uint8_t v_isShared_1568_; uint8_t v_isSharedCheck_1572_; 
lean_dec_ref(v___x_1525_);
lean_dec(v_a_1524_);
lean_dec(v_a_1500_);
v_a_1565_ = lean_ctor_get(v___x_1527_, 0);
v_isSharedCheck_1572_ = !lean_is_exclusive(v___x_1527_);
if (v_isSharedCheck_1572_ == 0)
{
v___x_1567_ = v___x_1527_;
v_isShared_1568_ = v_isSharedCheck_1572_;
goto v_resetjp_1566_;
}
else
{
lean_inc(v_a_1565_);
lean_dec(v___x_1527_);
v___x_1567_ = lean_box(0);
v_isShared_1568_ = v_isSharedCheck_1572_;
goto v_resetjp_1566_;
}
v_resetjp_1566_:
{
lean_object* v___x_1570_; 
if (v_isShared_1568_ == 0)
{
v___x_1570_ = v___x_1567_;
goto v_reusejp_1569_;
}
else
{
lean_object* v_reuseFailAlloc_1571_; 
v_reuseFailAlloc_1571_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1571_, 0, v_a_1565_);
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
lean_dec(v_a_1500_);
v_a_1573_ = lean_ctor_get(v___x_1523_, 0);
v_isSharedCheck_1580_ = !lean_is_exclusive(v___x_1523_);
if (v_isSharedCheck_1580_ == 0)
{
v___x_1575_ = v___x_1523_;
v_isShared_1576_ = v_isSharedCheck_1580_;
goto v_resetjp_1574_;
}
else
{
lean_inc(v_a_1573_);
lean_dec(v___x_1523_);
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
}
else
{
lean_object* v_a_1581_; lean_object* v___x_1583_; uint8_t v_isShared_1584_; uint8_t v_isSharedCheck_1588_; 
lean_dec(v_a_1497_);
v_a_1581_ = lean_ctor_get(v___x_1499_, 0);
v_isSharedCheck_1588_ = !lean_is_exclusive(v___x_1499_);
if (v_isSharedCheck_1588_ == 0)
{
v___x_1583_ = v___x_1499_;
v_isShared_1584_ = v_isSharedCheck_1588_;
goto v_resetjp_1582_;
}
else
{
lean_inc(v_a_1581_);
lean_dec(v___x_1499_);
v___x_1583_ = lean_box(0);
v_isShared_1584_ = v_isSharedCheck_1588_;
goto v_resetjp_1582_;
}
v_resetjp_1582_:
{
lean_object* v___x_1586_; 
if (v_isShared_1584_ == 0)
{
v___x_1586_ = v___x_1583_;
goto v_reusejp_1585_;
}
else
{
lean_object* v_reuseFailAlloc_1587_; 
v_reuseFailAlloc_1587_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1587_, 0, v_a_1581_);
v___x_1586_ = v_reuseFailAlloc_1587_;
goto v_reusejp_1585_;
}
v_reusejp_1585_:
{
return v___x_1586_;
}
}
}
}
else
{
lean_dec(v_a_1497_);
lean_dec(v_loc_1486_);
return v___x_1498_;
}
}
else
{
lean_object* v_a_1589_; lean_object* v___x_1591_; uint8_t v_isShared_1592_; uint8_t v_isSharedCheck_1596_; 
lean_dec(v_loc_1486_);
lean_dec_ref(v___x_1482_);
lean_dec(v_term_1480_);
v_a_1589_ = lean_ctor_get(v___x_1496_, 0);
v_isSharedCheck_1596_ = !lean_is_exclusive(v___x_1496_);
if (v_isSharedCheck_1596_ == 0)
{
v___x_1591_ = v___x_1496_;
v_isShared_1592_ = v_isSharedCheck_1596_;
goto v_resetjp_1590_;
}
else
{
lean_inc(v_a_1589_);
lean_dec(v___x_1496_);
v___x_1591_ = lean_box(0);
v_isShared_1592_ = v_isSharedCheck_1596_;
goto v_resetjp_1590_;
}
v_resetjp_1590_:
{
lean_object* v___x_1594_; 
if (v_isShared_1592_ == 0)
{
v___x_1594_ = v___x_1591_;
goto v_reusejp_1593_;
}
else
{
lean_object* v_reuseFailAlloc_1595_; 
v_reuseFailAlloc_1595_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1595_, 0, v_a_1589_);
v___x_1594_ = v_reuseFailAlloc_1595_;
goto v_reusejp_1593_;
}
v_reusejp_1593_:
{
return v___x_1594_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___boxed(lean_object* v_term_1597_, lean_object* v_symm_1598_, lean_object* v___x_1599_, lean_object* v_tk_1600_, lean_object* v___x_1601_, lean_object* v___y_1602_, lean_object* v_loc_1603_, lean_object* v___y_1604_, lean_object* v___y_1605_, lean_object* v___y_1606_, lean_object* v___y_1607_, lean_object* v___y_1608_, lean_object* v___y_1609_, lean_object* v___y_1610_, lean_object* v___y_1611_, lean_object* v___y_1612_){
_start:
{
uint8_t v_symm_boxed_1613_; uint8_t v___y_17420__boxed_1614_; lean_object* v_res_1615_; 
v_symm_boxed_1613_ = lean_unbox(v_symm_1598_);
v___y_17420__boxed_1614_ = lean_unbox(v___y_1602_);
v_res_1615_ = lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1(v_term_1597_, v_symm_boxed_1613_, v___x_1599_, v_tk_1600_, v___x_1601_, v___y_17420__boxed_1614_, v_loc_1603_, v___y_1604_, v___y_1605_, v___y_1606_, v___y_1607_, v___y_1608_, v___y_1609_, v___y_1610_, v___y_1611_);
lean_dec(v___y_1611_);
lean_dec_ref(v___y_1610_);
lean_dec(v___y_1609_);
lean_dec_ref(v___y_1608_);
lean_dec(v___y_1607_);
lean_dec_ref(v___y_1606_);
lean_dec(v___y_1605_);
lean_dec_ref(v___y_1604_);
lean_dec(v___x_1601_);
lean_dec(v_tk_1600_);
return v_res_1615_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__9(void){
_start:
{
lean_object* v___x_1625_; lean_object* v___x_1626_; 
v___x_1625_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__8));
v___x_1626_ = l_Lean_stringToMessageData(v___x_1625_);
return v___x_1626_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2(lean_object* v_term_1627_, uint8_t v_symm_1628_, lean_object* v___x_1629_, lean_object* v___x_1630_, lean_object* v___x_1631_, lean_object* v___x_1632_, lean_object* v_tk_1633_, lean_object* v___x_1634_, uint8_t v___y_1635_, lean_object* v___y_1636_, lean_object* v___y_1637_, lean_object* v___y_1638_, lean_object* v___y_1639_, lean_object* v___y_1640_, lean_object* v___y_1641_, lean_object* v___y_1642_, lean_object* v___y_1643_){
_start:
{
lean_object* v___x_1645_; 
v___x_1645_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_1637_, v___y_1640_, v___y_1641_, v___y_1642_, v___y_1643_);
if (lean_obj_tag(v___x_1645_) == 0)
{
lean_object* v_a_1646_; lean_object* v___x_1647_; 
v_a_1646_ = lean_ctor_get(v___x_1645_, 0);
lean_inc(v_a_1646_);
lean_dec_ref_known(v___x_1645_, 1);
v___x_1647_ = l_Lean_Elab_Tactic_rewriteTarget(v_term_1627_, v_symm_1628_, v___x_1629_, v___y_1636_, v___y_1637_, v___y_1638_, v___y_1639_, v___y_1640_, v___y_1641_, v___y_1642_, v___y_1643_);
if (lean_obj_tag(v___x_1647_) == 0)
{
lean_object* v_fileName_1648_; lean_object* v_fileMap_1649_; lean_object* v_options_1650_; lean_object* v_currRecDepth_1651_; lean_object* v_maxRecDepth_1652_; lean_object* v_ref_1653_; lean_object* v_currNamespace_1654_; lean_object* v_openDecls_1655_; lean_object* v_initHeartbeats_1656_; lean_object* v_maxHeartbeats_1657_; lean_object* v_quotContext_1658_; lean_object* v_currMacroScope_1659_; uint8_t v_diag_1660_; lean_object* v_cancelTk_x3f_1661_; uint8_t v_suppressElabErrors_1662_; lean_object* v_inheritedTraceOptions_1663_; uint8_t v___x_1664_; lean_object* v___x_1665_; lean_object* v___x_1666_; lean_object* v___x_1667_; lean_object* v___x_1668_; lean_object* v___x_1669_; lean_object* v___x_1670_; lean_object* v___x_1671_; lean_object* v___x_1672_; lean_object* v___x_1673_; lean_object* v___x_1674_; lean_object* v___x_1675_; lean_object* v___x_1676_; lean_object* v___x_1677_; lean_object* v___x_1678_; lean_object* v___x_1679_; lean_object* v___x_1680_; lean_object* v___x_1681_; lean_object* v___x_1682_; lean_object* v___x_1683_; lean_object* v___x_1684_; lean_object* v___x_1685_; lean_object* v___x_1686_; lean_object* v___x_1687_; lean_object* v___x_1688_; lean_object* v___x_1689_; lean_object* v___x_1690_; lean_object* v___x_1691_; lean_object* v___x_1692_; 
lean_dec_ref_known(v___x_1647_, 1);
v_fileName_1648_ = lean_ctor_get(v___y_1642_, 0);
v_fileMap_1649_ = lean_ctor_get(v___y_1642_, 1);
v_options_1650_ = lean_ctor_get(v___y_1642_, 2);
v_currRecDepth_1651_ = lean_ctor_get(v___y_1642_, 3);
v_maxRecDepth_1652_ = lean_ctor_get(v___y_1642_, 4);
v_ref_1653_ = lean_ctor_get(v___y_1642_, 5);
v_currNamespace_1654_ = lean_ctor_get(v___y_1642_, 6);
v_openDecls_1655_ = lean_ctor_get(v___y_1642_, 7);
v_initHeartbeats_1656_ = lean_ctor_get(v___y_1642_, 8);
v_maxHeartbeats_1657_ = lean_ctor_get(v___y_1642_, 9);
v_quotContext_1658_ = lean_ctor_get(v___y_1642_, 10);
v_currMacroScope_1659_ = lean_ctor_get(v___y_1642_, 11);
v_diag_1660_ = lean_ctor_get_uint8(v___y_1642_, sizeof(void*)*14);
v_cancelTk_x3f_1661_ = lean_ctor_get(v___y_1642_, 12);
v_suppressElabErrors_1662_ = lean_ctor_get_uint8(v___y_1642_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1663_ = lean_ctor_get(v___y_1642_, 13);
v___x_1664_ = 0;
v___x_1665_ = l_Lean_SourceInfo_fromRef(v_ref_1653_, v___x_1664_);
v___x_1666_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__0));
lean_inc_ref_n(v___x_1632_, 4);
lean_inc_ref_n(v___x_1631_, 4);
lean_inc_ref_n(v___x_1630_, 4);
v___x_1667_ = l_Lean_Name_mkStr4(v___x_1630_, v___x_1631_, v___x_1632_, v___x_1666_);
v___x_1668_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__1));
lean_inc_n(v___x_1665_, 11);
v___x_1669_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1669_, 0, v___x_1665_);
lean_ctor_set(v___x_1669_, 1, v___x_1668_);
v___x_1670_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__2));
v___x_1671_ = l_Lean_Name_mkStr4(v___x_1630_, v___x_1631_, v___x_1632_, v___x_1670_);
v___x_1672_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__3));
v___x_1673_ = l_Lean_Name_mkStr4(v___x_1630_, v___x_1631_, v___x_1632_, v___x_1672_);
v___x_1674_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__12));
v___x_1675_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__4));
v___x_1676_ = l_Lean_Name_mkStr4(v___x_1630_, v___x_1631_, v___x_1632_, v___x_1675_);
v___x_1677_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__5));
v___x_1678_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1678_, 0, v___x_1665_);
lean_ctor_set(v___x_1678_, 1, v___x_1677_);
v___x_1679_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__6));
v___x_1680_ = l_Lean_Name_mkStr4(v___x_1630_, v___x_1631_, v___x_1632_, v___x_1679_);
v___x_1681_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__7));
v___x_1682_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1682_, 0, v___x_1665_);
lean_ctor_set(v___x_1682_, 1, v___x_1681_);
v___x_1683_ = l_Lean_Syntax_node1(v___x_1665_, v___x_1680_, v___x_1682_);
v___x_1684_ = l_Lean_Syntax_node1(v___x_1665_, v___x_1674_, v___x_1683_);
lean_inc(v___x_1673_);
v___x_1685_ = l_Lean_Syntax_node1(v___x_1665_, v___x_1673_, v___x_1684_);
lean_inc(v___x_1671_);
v___x_1686_ = l_Lean_Syntax_node1(v___x_1665_, v___x_1671_, v___x_1685_);
v___x_1687_ = l_Lean_Syntax_node2(v___x_1665_, v___x_1676_, v___x_1678_, v___x_1686_);
v___x_1688_ = l_Lean_Syntax_node1(v___x_1665_, v___x_1674_, v___x_1687_);
v___x_1689_ = l_Lean_Syntax_node1(v___x_1665_, v___x_1673_, v___x_1688_);
v___x_1690_ = l_Lean_Syntax_node1(v___x_1665_, v___x_1671_, v___x_1689_);
v___x_1691_ = l_Lean_Syntax_node2(v___x_1665_, v___x_1667_, v___x_1669_, v___x_1690_);
v___x_1692_ = l_Lean_Elab_Tactic_evalTactic(v___x_1691_, v___y_1636_, v___y_1637_, v___y_1638_, v___y_1639_, v___y_1640_, v___y_1641_, v___y_1642_, v___y_1643_);
if (lean_obj_tag(v___x_1692_) == 0)
{
lean_object* v___x_1693_; lean_object* v___x_1694_; lean_object* v_a_1695_; lean_object* v___x_1696_; lean_object* v_ref_1697_; lean_object* v___x_1698_; lean_object* v___x_1699_; 
lean_dec_ref_known(v___x_1692_, 1);
v___x_1693_ = l_Lean_Expr_mvar___override(v_a_1646_);
v___x_1694_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__2___redArg(v___x_1693_, v___y_1641_);
v_a_1695_ = lean_ctor_get(v___x_1694_, 0);
lean_inc(v_a_1695_);
lean_dec_ref(v___x_1694_);
v___x_1696_ = l_Lean_Expr_headBeta(v_a_1695_);
v_ref_1697_ = l_Lean_replaceRef(v_tk_1633_, v_ref_1653_);
lean_inc_ref(v_inheritedTraceOptions_1663_);
lean_inc(v_cancelTk_x3f_1661_);
lean_inc(v_currMacroScope_1659_);
lean_inc(v_quotContext_1658_);
lean_inc(v_maxHeartbeats_1657_);
lean_inc(v_initHeartbeats_1656_);
lean_inc(v_openDecls_1655_);
lean_inc(v_currNamespace_1654_);
lean_inc(v_maxRecDepth_1652_);
lean_inc(v_currRecDepth_1651_);
lean_inc_ref(v_options_1650_);
lean_inc_ref(v_fileMap_1649_);
lean_inc_ref(v_fileName_1648_);
v___x_1698_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1698_, 0, v_fileName_1648_);
lean_ctor_set(v___x_1698_, 1, v_fileMap_1649_);
lean_ctor_set(v___x_1698_, 2, v_options_1650_);
lean_ctor_set(v___x_1698_, 3, v_currRecDepth_1651_);
lean_ctor_set(v___x_1698_, 4, v_maxRecDepth_1652_);
lean_ctor_set(v___x_1698_, 5, v_ref_1697_);
lean_ctor_set(v___x_1698_, 6, v_currNamespace_1654_);
lean_ctor_set(v___x_1698_, 7, v_openDecls_1655_);
lean_ctor_set(v___x_1698_, 8, v_initHeartbeats_1656_);
lean_ctor_set(v___x_1698_, 9, v_maxHeartbeats_1657_);
lean_ctor_set(v___x_1698_, 10, v_quotContext_1658_);
lean_ctor_set(v___x_1698_, 11, v_currMacroScope_1659_);
lean_ctor_set(v___x_1698_, 12, v_cancelTk_x3f_1661_);
lean_ctor_set(v___x_1698_, 13, v_inheritedTraceOptions_1663_);
lean_ctor_set_uint8(v___x_1698_, sizeof(void*)*14, v_diag_1660_);
lean_ctor_set_uint8(v___x_1698_, sizeof(void*)*14 + 1, v_suppressElabErrors_1662_);
v___x_1699_ = lp_mathlib_Mathlib_Tactic_Erw_x3f_extractRewriteEq(v___x_1696_, v___y_1640_, v___y_1641_, v___x_1698_, v___y_1643_);
lean_dec_ref_known(v___x_1698_, 14);
if (lean_obj_tag(v___x_1699_) == 0)
{
lean_object* v_a_1700_; lean_object* v_fst_1701_; lean_object* v_snd_1702_; lean_object* v___x_1704_; uint8_t v_isShared_1705_; uint8_t v_isSharedCheck_1750_; 
v_a_1700_ = lean_ctor_get(v___x_1699_, 0);
lean_inc(v_a_1700_);
lean_dec_ref_known(v___x_1699_, 1);
v_fst_1701_ = lean_ctor_get(v_a_1700_, 0);
v_snd_1702_ = lean_ctor_get(v_a_1700_, 1);
v_isSharedCheck_1750_ = !lean_is_exclusive(v_a_1700_);
if (v_isSharedCheck_1750_ == 0)
{
v___x_1704_ = v_a_1700_;
v_isShared_1705_ = v_isSharedCheck_1750_;
goto v_resetjp_1703_;
}
else
{
lean_inc(v_snd_1702_);
lean_inc(v_fst_1701_);
lean_dec(v_a_1700_);
v___x_1704_ = lean_box(0);
v_isShared_1705_ = v_isSharedCheck_1750_;
goto v_resetjp_1703_;
}
v_resetjp_1703_:
{
lean_object* v___x_1706_; lean_object* v___x_1707_; 
v___x_1706_ = lean_mk_empty_array_with_capacity(v___x_1634_);
lean_inc(v_snd_1702_);
lean_inc(v_fst_1701_);
v___x_1707_ = lp_mathlib_Mathlib_Tactic_Erw_x3f_logDiffs(v_tk_1633_, v_fst_1701_, v_snd_1702_, v___x_1706_, v___y_1640_, v___y_1641_, v___y_1642_, v___y_1643_);
if (lean_obj_tag(v___x_1707_) == 0)
{
lean_object* v_a_1708_; lean_object* v___x_1710_; uint8_t v_isShared_1711_; uint8_t v_isSharedCheck_1741_; 
v_a_1708_ = lean_ctor_get(v___x_1707_, 0);
v_isSharedCheck_1741_ = !lean_is_exclusive(v___x_1707_);
if (v_isSharedCheck_1741_ == 0)
{
v___x_1710_ = v___x_1707_;
v_isShared_1711_ = v_isSharedCheck_1741_;
goto v_resetjp_1709_;
}
else
{
lean_inc(v_a_1708_);
lean_dec(v___x_1707_);
v___x_1710_ = lean_box(0);
v_isShared_1711_ = v_isSharedCheck_1741_;
goto v_resetjp_1709_;
}
v_resetjp_1709_:
{
if (v___y_1635_ == 0)
{
lean_object* v___x_1712_; lean_object* v___x_1714_; 
lean_dec(v_a_1708_);
lean_del_object(v___x_1704_);
lean_dec(v_snd_1702_);
lean_dec(v_fst_1701_);
v___x_1712_ = lean_box(0);
if (v_isShared_1711_ == 0)
{
lean_ctor_set(v___x_1710_, 0, v___x_1712_);
v___x_1714_ = v___x_1710_;
goto v_reusejp_1713_;
}
else
{
lean_object* v_reuseFailAlloc_1715_; 
v_reuseFailAlloc_1715_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1715_, 0, v___x_1712_);
v___x_1714_ = v_reuseFailAlloc_1715_;
goto v_reusejp_1713_;
}
v_reusejp_1713_:
{
return v___x_1714_;
}
}
else
{
lean_object* v_snd_1716_; lean_object* v___x_1718_; uint8_t v_isShared_1719_; uint8_t v_isSharedCheck_1739_; 
lean_del_object(v___x_1710_);
v_snd_1716_ = lean_ctor_get(v_a_1708_, 1);
v_isSharedCheck_1739_ = !lean_is_exclusive(v_a_1708_);
if (v_isSharedCheck_1739_ == 0)
{
lean_object* v_unused_1740_; 
v_unused_1740_ = lean_ctor_get(v_a_1708_, 0);
lean_dec(v_unused_1740_);
v___x_1718_ = v_a_1708_;
v_isShared_1719_ = v_isSharedCheck_1739_;
goto v_resetjp_1717_;
}
else
{
lean_inc(v_snd_1716_);
lean_dec(v_a_1708_);
v___x_1718_ = lean_box(0);
v_isShared_1719_ = v_isSharedCheck_1739_;
goto v_resetjp_1717_;
}
v_resetjp_1717_:
{
lean_object* v___x_1720_; lean_object* v___x_1721_; lean_object* v___x_1722_; lean_object* v___x_1724_; 
v___x_1720_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__9, &lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___closed__9);
v___x_1721_ = l_Lean_MessageData_ofExpr(v_fst_1701_);
v___x_1722_ = l_Lean_indentD(v___x_1721_);
if (v_isShared_1719_ == 0)
{
lean_ctor_set_tag(v___x_1718_, 7);
lean_ctor_set(v___x_1718_, 1, v___x_1722_);
lean_ctor_set(v___x_1718_, 0, v___x_1720_);
v___x_1724_ = v___x_1718_;
goto v_reusejp_1723_;
}
else
{
lean_object* v_reuseFailAlloc_1738_; 
v_reuseFailAlloc_1738_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1738_, 0, v___x_1720_);
lean_ctor_set(v_reuseFailAlloc_1738_, 1, v___x_1722_);
v___x_1724_ = v_reuseFailAlloc_1738_;
goto v_reusejp_1723_;
}
v_reusejp_1723_:
{
lean_object* v___x_1725_; lean_object* v___x_1726_; lean_object* v___x_1728_; 
v___x_1725_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__5, &lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__5_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__5);
v___x_1726_ = l_Lean_MessageData_ofExpr(v_snd_1702_);
if (v_isShared_1705_ == 0)
{
lean_ctor_set_tag(v___x_1704_, 7);
lean_ctor_set(v___x_1704_, 1, v___x_1726_);
lean_ctor_set(v___x_1704_, 0, v___x_1725_);
v___x_1728_ = v___x_1704_;
goto v_reusejp_1727_;
}
else
{
lean_object* v_reuseFailAlloc_1737_; 
v_reuseFailAlloc_1737_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1737_, 0, v___x_1725_);
lean_ctor_set(v_reuseFailAlloc_1737_, 1, v___x_1726_);
v___x_1728_ = v_reuseFailAlloc_1737_;
goto v_reusejp_1727_;
}
v_reusejp_1727_:
{
lean_object* v___x_1729_; lean_object* v___x_1730_; lean_object* v___x_1731_; lean_object* v___x_1732_; lean_object* v___x_1733_; lean_object* v___x_1734_; lean_object* v___x_1735_; lean_object* v___x_1736_; 
v___x_1729_ = lean_array_to_list(v_snd_1716_);
v___x_1730_ = lean_box(0);
v___x_1731_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__3(v___x_1729_, v___x_1730_);
v___x_1732_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1732_, 0, v___x_1728_);
lean_ctor_set(v___x_1732_, 1, v___x_1731_);
v___x_1733_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1733_, 0, v___x_1724_);
lean_ctor_set(v___x_1733_, 1, v___x_1732_);
v___x_1734_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__8, &lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__8_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___closed__8);
v___x_1735_ = l_Lean_MessageData_joinSep(v___x_1733_, v___x_1734_);
v___x_1736_ = lp_mathlib_Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__1(v_tk_1633_, v___x_1735_, v___y_1636_, v___y_1637_, v___y_1638_, v___y_1639_, v___y_1640_, v___y_1641_, v___y_1642_, v___y_1643_);
return v___x_1736_;
}
}
}
}
}
}
else
{
lean_object* v_a_1742_; lean_object* v___x_1744_; uint8_t v_isShared_1745_; uint8_t v_isSharedCheck_1749_; 
lean_del_object(v___x_1704_);
lean_dec(v_snd_1702_);
lean_dec(v_fst_1701_);
v_a_1742_ = lean_ctor_get(v___x_1707_, 0);
v_isSharedCheck_1749_ = !lean_is_exclusive(v___x_1707_);
if (v_isSharedCheck_1749_ == 0)
{
v___x_1744_ = v___x_1707_;
v_isShared_1745_ = v_isSharedCheck_1749_;
goto v_resetjp_1743_;
}
else
{
lean_inc(v_a_1742_);
lean_dec(v___x_1707_);
v___x_1744_ = lean_box(0);
v_isShared_1745_ = v_isSharedCheck_1749_;
goto v_resetjp_1743_;
}
v_resetjp_1743_:
{
lean_object* v___x_1747_; 
if (v_isShared_1745_ == 0)
{
v___x_1747_ = v___x_1744_;
goto v_reusejp_1746_;
}
else
{
lean_object* v_reuseFailAlloc_1748_; 
v_reuseFailAlloc_1748_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1748_, 0, v_a_1742_);
v___x_1747_ = v_reuseFailAlloc_1748_;
goto v_reusejp_1746_;
}
v_reusejp_1746_:
{
return v___x_1747_;
}
}
}
}
}
else
{
lean_object* v_a_1751_; lean_object* v___x_1753_; uint8_t v_isShared_1754_; uint8_t v_isSharedCheck_1758_; 
v_a_1751_ = lean_ctor_get(v___x_1699_, 0);
v_isSharedCheck_1758_ = !lean_is_exclusive(v___x_1699_);
if (v_isSharedCheck_1758_ == 0)
{
v___x_1753_ = v___x_1699_;
v_isShared_1754_ = v_isSharedCheck_1758_;
goto v_resetjp_1752_;
}
else
{
lean_inc(v_a_1751_);
lean_dec(v___x_1699_);
v___x_1753_ = lean_box(0);
v_isShared_1754_ = v_isSharedCheck_1758_;
goto v_resetjp_1752_;
}
v_resetjp_1752_:
{
lean_object* v___x_1756_; 
if (v_isShared_1754_ == 0)
{
v___x_1756_ = v___x_1753_;
goto v_reusejp_1755_;
}
else
{
lean_object* v_reuseFailAlloc_1757_; 
v_reuseFailAlloc_1757_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1757_, 0, v_a_1751_);
v___x_1756_ = v_reuseFailAlloc_1757_;
goto v_reusejp_1755_;
}
v_reusejp_1755_:
{
return v___x_1756_;
}
}
}
}
else
{
lean_dec(v_a_1646_);
return v___x_1692_;
}
}
else
{
lean_dec(v_a_1646_);
lean_dec_ref(v___x_1632_);
lean_dec_ref(v___x_1631_);
lean_dec_ref(v___x_1630_);
return v___x_1647_;
}
}
else
{
lean_object* v_a_1759_; lean_object* v___x_1761_; uint8_t v_isShared_1762_; uint8_t v_isSharedCheck_1766_; 
lean_dec_ref(v___x_1632_);
lean_dec_ref(v___x_1631_);
lean_dec_ref(v___x_1630_);
lean_dec_ref(v___x_1629_);
lean_dec(v_term_1627_);
v_a_1759_ = lean_ctor_get(v___x_1645_, 0);
v_isSharedCheck_1766_ = !lean_is_exclusive(v___x_1645_);
if (v_isSharedCheck_1766_ == 0)
{
v___x_1761_ = v___x_1645_;
v_isShared_1762_ = v_isSharedCheck_1766_;
goto v_resetjp_1760_;
}
else
{
lean_inc(v_a_1759_);
lean_dec(v___x_1645_);
v___x_1761_ = lean_box(0);
v_isShared_1762_ = v_isSharedCheck_1766_;
goto v_resetjp_1760_;
}
v_resetjp_1760_:
{
lean_object* v___x_1764_; 
if (v_isShared_1762_ == 0)
{
v___x_1764_ = v___x_1761_;
goto v_reusejp_1763_;
}
else
{
lean_object* v_reuseFailAlloc_1765_; 
v_reuseFailAlloc_1765_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1765_, 0, v_a_1759_);
v___x_1764_ = v_reuseFailAlloc_1765_;
goto v_reusejp_1763_;
}
v_reusejp_1763_:
{
return v___x_1764_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___boxed(lean_object** _args){
lean_object* v_term_1767_ = _args[0];
lean_object* v_symm_1768_ = _args[1];
lean_object* v___x_1769_ = _args[2];
lean_object* v___x_1770_ = _args[3];
lean_object* v___x_1771_ = _args[4];
lean_object* v___x_1772_ = _args[5];
lean_object* v_tk_1773_ = _args[6];
lean_object* v___x_1774_ = _args[7];
lean_object* v___y_1775_ = _args[8];
lean_object* v___y_1776_ = _args[9];
lean_object* v___y_1777_ = _args[10];
lean_object* v___y_1778_ = _args[11];
lean_object* v___y_1779_ = _args[12];
lean_object* v___y_1780_ = _args[13];
lean_object* v___y_1781_ = _args[14];
lean_object* v___y_1782_ = _args[15];
lean_object* v___y_1783_ = _args[16];
lean_object* v___y_1784_ = _args[17];
_start:
{
uint8_t v_symm_boxed_1785_; uint8_t v___y_17677__boxed_1786_; lean_object* v_res_1787_; 
v_symm_boxed_1785_ = lean_unbox(v_symm_1768_);
v___y_17677__boxed_1786_ = lean_unbox(v___y_1775_);
v_res_1787_ = lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2(v_term_1767_, v_symm_boxed_1785_, v___x_1769_, v___x_1770_, v___x_1771_, v___x_1772_, v_tk_1773_, v___x_1774_, v___y_17677__boxed_1786_, v___y_1776_, v___y_1777_, v___y_1778_, v___y_1779_, v___y_1780_, v___y_1781_, v___y_1782_, v___y_1783_);
lean_dec(v___y_1783_);
lean_dec_ref(v___y_1782_);
lean_dec(v___y_1781_);
lean_dec_ref(v___y_1780_);
lean_dec(v___y_1779_);
lean_dec_ref(v___y_1778_);
lean_dec(v___y_1777_);
lean_dec_ref(v___y_1776_);
lean_dec(v___x_1774_);
lean_dec(v_tk_1773_);
return v_res_1787_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__3(lean_object* v___x_1788_, lean_object* v_tk_1789_, lean_object* v___x_1790_, uint8_t v___y_1791_, lean_object* v___x_1792_, lean_object* v___x_1793_, lean_object* v___x_1794_, lean_object* v___x_1795_, lean_object* v___f_1796_, uint8_t v_symm_1797_, lean_object* v_term_1798_, lean_object* v___y_1799_, lean_object* v___y_1800_, lean_object* v___y_1801_, lean_object* v___y_1802_, lean_object* v___y_1803_, lean_object* v___y_1804_, lean_object* v___y_1805_, lean_object* v___y_1806_){
_start:
{
lean_object* v___x_1808_; lean_object* v___x_1809_; lean_object* v___f_1810_; lean_object* v___x_1811_; lean_object* v___x_1812_; lean_object* v___f_1813_; lean_object* v___x_1814_; 
v___x_1808_ = lean_box(v_symm_1797_);
v___x_1809_ = lean_box(v___y_1791_);
lean_inc(v___x_1790_);
lean_inc(v_tk_1789_);
lean_inc_ref(v___x_1788_);
lean_inc(v_term_1798_);
v___f_1810_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__1___boxed), 16, 6);
lean_closure_set(v___f_1810_, 0, v_term_1798_);
lean_closure_set(v___f_1810_, 1, v___x_1808_);
lean_closure_set(v___f_1810_, 2, v___x_1788_);
lean_closure_set(v___f_1810_, 3, v_tk_1789_);
lean_closure_set(v___f_1810_, 4, v___x_1790_);
lean_closure_set(v___f_1810_, 5, v___x_1809_);
v___x_1811_ = lean_box(v_symm_1797_);
v___x_1812_ = lean_box(v___y_1791_);
v___f_1813_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__2___boxed), 18, 9);
lean_closure_set(v___f_1813_, 0, v_term_1798_);
lean_closure_set(v___f_1813_, 1, v___x_1811_);
lean_closure_set(v___f_1813_, 2, v___x_1788_);
lean_closure_set(v___f_1813_, 3, v___x_1792_);
lean_closure_set(v___f_1813_, 4, v___x_1793_);
lean_closure_set(v___f_1813_, 5, v___x_1794_);
lean_closure_set(v___f_1813_, 6, v_tk_1789_);
lean_closure_set(v___f_1813_, 7, v___x_1790_);
lean_closure_set(v___f_1813_, 8, v___x_1812_);
v___x_1814_ = l_Lean_Elab_Tactic_withLocation(v___x_1795_, v___f_1810_, v___f_1813_, v___f_1796_, v___y_1799_, v___y_1800_, v___y_1801_, v___y_1802_, v___y_1803_, v___y_1804_, v___y_1805_, v___y_1806_);
return v___x_1814_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__3___boxed(lean_object** _args){
lean_object* v___x_1815_ = _args[0];
lean_object* v_tk_1816_ = _args[1];
lean_object* v___x_1817_ = _args[2];
lean_object* v___y_1818_ = _args[3];
lean_object* v___x_1819_ = _args[4];
lean_object* v___x_1820_ = _args[5];
lean_object* v___x_1821_ = _args[6];
lean_object* v___x_1822_ = _args[7];
lean_object* v___f_1823_ = _args[8];
lean_object* v_symm_1824_ = _args[9];
lean_object* v_term_1825_ = _args[10];
lean_object* v___y_1826_ = _args[11];
lean_object* v___y_1827_ = _args[12];
lean_object* v___y_1828_ = _args[13];
lean_object* v___y_1829_ = _args[14];
lean_object* v___y_1830_ = _args[15];
lean_object* v___y_1831_ = _args[16];
lean_object* v___y_1832_ = _args[17];
lean_object* v___y_1833_ = _args[18];
lean_object* v___y_1834_ = _args[19];
_start:
{
uint8_t v___y_17944__boxed_1835_; uint8_t v_symm_boxed_1836_; lean_object* v_res_1837_; 
v___y_17944__boxed_1835_ = lean_unbox(v___y_1818_);
v_symm_boxed_1836_ = lean_unbox(v_symm_1824_);
v_res_1837_ = lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__3(v___x_1815_, v_tk_1816_, v___x_1817_, v___y_17944__boxed_1835_, v___x_1819_, v___x_1820_, v___x_1821_, v___x_1822_, v___f_1823_, v_symm_boxed_1836_, v_term_1825_, v___y_1826_, v___y_1827_, v___y_1828_, v___y_1829_, v___y_1830_, v___y_1831_, v___y_1832_, v___y_1833_);
lean_dec(v___y_1833_);
lean_dec_ref(v___y_1832_);
lean_dec(v___y_1831_);
lean_dec_ref(v___y_1830_);
lean_dec(v___y_1829_);
lean_dec_ref(v___y_1828_);
lean_dec(v___y_1827_);
lean_dec_ref(v___y_1826_);
lean_dec(v___x_1822_);
return v_res_1837_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__4(lean_object* v___x_1838_, lean_object* v___x_1839_, uint8_t v___x_1840_, lean_object* v_tk_1841_, lean_object* v___x_1842_, lean_object* v___x_1843_, lean_object* v___x_1844_, lean_object* v___x_1845_, lean_object* v___f_1846_, lean_object* v___y_1847_, lean_object* v___x_1848_, lean_object* v___y_1849_, lean_object* v___y_1850_, lean_object* v___y_1851_, lean_object* v___y_1852_, lean_object* v___y_1853_, lean_object* v___y_1854_, lean_object* v___y_1855_, lean_object* v___y_1856_){
_start:
{
lean_object* v___x_1858_; 
v___x_1858_ = lp_mathlib_Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__1(v___x_1838_, v___x_1839_, v___y_1849_, v___y_1850_, v___y_1851_, v___y_1852_, v___y_1853_, v___y_1854_, v___y_1855_, v___y_1856_);
if (lean_obj_tag(v___x_1858_) == 0)
{
uint8_t v___y_1860_; lean_object* v___y_1861_; lean_object* v___y_1862_; uint8_t v___y_1869_; lean_object* v_options_1883_; lean_object* v_map_1884_; lean_object* v___x_1885_; lean_object* v___x_1886_; lean_object* v___x_1887_; uint8_t v___x_1888_; lean_object* v___x_1889_; 
lean_dec_ref_known(v___x_1858_, 1);
v_options_1883_ = lean_ctor_get(v___y_1855_, 2);
v_map_1884_ = lean_ctor_get(v_options_1883_, 0);
v___x_1885_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__0_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4_));
v___x_1886_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__2_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4_));
v___x_1887_ = l_Lean_Name_mkStr3(v___x_1885_, v___x_1848_, v___x_1886_);
v___x_1888_ = 0;
v___x_1889_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1884_, v___x_1887_);
lean_dec(v___x_1887_);
if (lean_obj_tag(v___x_1889_) == 0)
{
v___y_1869_ = v___x_1888_;
goto v___jp_1868_;
}
else
{
lean_object* v_val_1890_; 
v_val_1890_ = lean_ctor_get(v___x_1889_, 0);
lean_inc(v_val_1890_);
lean_dec_ref_known(v___x_1889_, 1);
if (lean_obj_tag(v_val_1890_) == 1)
{
uint8_t v_v_1891_; 
v_v_1891_ = lean_ctor_get_uint8(v_val_1890_, 0);
lean_dec_ref_known(v_val_1890_, 0);
v___y_1869_ = v_v_1891_;
goto v___jp_1868_;
}
else
{
lean_dec(v_val_1890_);
v___y_1869_ = v___x_1888_;
goto v___jp_1868_;
}
}
v___jp_1859_:
{
lean_object* v___x_1863_; lean_object* v___x_1864_; lean_object* v___x_1865_; lean_object* v___f_1866_; lean_object* v___x_1867_; 
v___x_1863_ = l_Lean_mkOptionalNode(v___y_1862_);
v___x_1864_ = l_Lean_Elab_Tactic_expandOptLocation(v___x_1863_);
lean_dec(v___x_1863_);
v___x_1865_ = lean_box(v___y_1860_);
lean_inc(v_tk_1841_);
v___f_1866_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__3___boxed), 20, 9);
lean_closure_set(v___f_1866_, 0, v___y_1861_);
lean_closure_set(v___f_1866_, 1, v_tk_1841_);
lean_closure_set(v___f_1866_, 2, v___x_1842_);
lean_closure_set(v___f_1866_, 3, v___x_1865_);
lean_closure_set(v___f_1866_, 4, v___x_1843_);
lean_closure_set(v___f_1866_, 5, v___x_1844_);
lean_closure_set(v___f_1866_, 6, v___x_1845_);
lean_closure_set(v___f_1866_, 7, v___x_1864_);
lean_closure_set(v___f_1866_, 8, v___f_1846_);
v___x_1867_ = l_Lean_Elab_Tactic_withRWRulesSeq(v_tk_1841_, v___x_1838_, v___f_1866_, v___y_1849_, v___y_1850_, v___y_1851_, v___y_1852_, v___y_1853_, v___y_1854_, v___y_1855_, v___y_1856_);
return v___x_1867_;
}
v___jp_1868_:
{
uint8_t v___x_1870_; lean_object* v___x_1871_; uint8_t v___x_1872_; lean_object* v___x_1873_; 
v___x_1870_ = 1;
v___x_1871_ = lean_box(0);
v___x_1872_ = 0;
v___x_1873_ = lean_alloc_ctor(0, 1, 3);
lean_ctor_set(v___x_1873_, 0, v___x_1871_);
lean_ctor_set_uint8(v___x_1873_, sizeof(void*)*1, v___x_1870_);
lean_ctor_set_uint8(v___x_1873_, sizeof(void*)*1 + 1, v___x_1840_);
lean_ctor_set_uint8(v___x_1873_, sizeof(void*)*1 + 2, v___x_1872_);
if (lean_obj_tag(v___y_1847_) == 0)
{
lean_object* v___x_1874_; 
v___x_1874_ = lean_box(0);
v___y_1860_ = v___y_1869_;
v___y_1861_ = v___x_1873_;
v___y_1862_ = v___x_1874_;
goto v___jp_1859_;
}
else
{
lean_object* v_val_1875_; lean_object* v___x_1877_; uint8_t v_isShared_1878_; uint8_t v_isSharedCheck_1882_; 
v_val_1875_ = lean_ctor_get(v___y_1847_, 0);
v_isSharedCheck_1882_ = !lean_is_exclusive(v___y_1847_);
if (v_isSharedCheck_1882_ == 0)
{
v___x_1877_ = v___y_1847_;
v_isShared_1878_ = v_isSharedCheck_1882_;
goto v_resetjp_1876_;
}
else
{
lean_inc(v_val_1875_);
lean_dec(v___y_1847_);
v___x_1877_ = lean_box(0);
v_isShared_1878_ = v_isSharedCheck_1882_;
goto v_resetjp_1876_;
}
v_resetjp_1876_:
{
lean_object* v___x_1880_; 
if (v_isShared_1878_ == 0)
{
v___x_1880_ = v___x_1877_;
goto v_reusejp_1879_;
}
else
{
lean_object* v_reuseFailAlloc_1881_; 
v_reuseFailAlloc_1881_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1881_, 0, v_val_1875_);
v___x_1880_ = v_reuseFailAlloc_1881_;
goto v_reusejp_1879_;
}
v_reusejp_1879_:
{
v___y_1860_ = v___y_1869_;
v___y_1861_ = v___x_1873_;
v___y_1862_ = v___x_1880_;
goto v___jp_1859_;
}
}
}
}
}
else
{
lean_dec_ref(v___x_1848_);
lean_dec(v___y_1847_);
lean_dec_ref(v___f_1846_);
lean_dec_ref(v___x_1845_);
lean_dec_ref(v___x_1844_);
lean_dec_ref(v___x_1843_);
lean_dec(v___x_1842_);
lean_dec(v_tk_1841_);
return v___x_1858_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__4___boxed(lean_object** _args){
lean_object* v___x_1892_ = _args[0];
lean_object* v___x_1893_ = _args[1];
lean_object* v___x_1894_ = _args[2];
lean_object* v_tk_1895_ = _args[3];
lean_object* v___x_1896_ = _args[4];
lean_object* v___x_1897_ = _args[5];
lean_object* v___x_1898_ = _args[6];
lean_object* v___x_1899_ = _args[7];
lean_object* v___f_1900_ = _args[8];
lean_object* v___y_1901_ = _args[9];
lean_object* v___x_1902_ = _args[10];
lean_object* v___y_1903_ = _args[11];
lean_object* v___y_1904_ = _args[12];
lean_object* v___y_1905_ = _args[13];
lean_object* v___y_1906_ = _args[14];
lean_object* v___y_1907_ = _args[15];
lean_object* v___y_1908_ = _args[16];
lean_object* v___y_1909_ = _args[17];
lean_object* v___y_1910_ = _args[18];
lean_object* v___y_1911_ = _args[19];
_start:
{
uint8_t v___x_18013__boxed_1912_; lean_object* v_res_1913_; 
v___x_18013__boxed_1912_ = lean_unbox(v___x_1894_);
v_res_1913_ = lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__4(v___x_1892_, v___x_1893_, v___x_18013__boxed_1912_, v_tk_1895_, v___x_1896_, v___x_1897_, v___x_1898_, v___x_1899_, v___f_1900_, v___y_1901_, v___x_1902_, v___y_1903_, v___y_1904_, v___y_1905_, v___y_1906_, v___y_1907_, v___y_1908_, v___y_1909_, v___y_1910_);
lean_dec(v___y_1910_);
lean_dec_ref(v___y_1909_);
lean_dec(v___y_1908_);
lean_dec_ref(v___y_1907_);
lean_dec(v___y_1906_);
lean_dec_ref(v___y_1905_);
lean_dec(v___y_1904_);
lean_dec_ref(v___y_1903_);
lean_dec(v___x_1892_);
return v_res_1913_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___closed__3(void){
_start:
{
lean_object* v___x_1918_; lean_object* v___x_1919_; 
v___x_1918_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___closed__2));
v___x_1919_ = l_Lean_MessageData_ofFormat(v___x_1918_);
return v___x_1919_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1(lean_object* v_x_1920_, lean_object* v_a_1921_, lean_object* v_a_1922_, lean_object* v_a_1923_, lean_object* v_a_1924_, lean_object* v_a_1925_, lean_object* v_a_1926_, lean_object* v_a_1927_, lean_object* v_a_1928_){
_start:
{
lean_object* v___x_1930_; lean_object* v___x_1931_; lean_object* v___x_1932_; uint8_t v___x_1933_; 
v___x_1930_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__7_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4_));
v___x_1931_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn___closed__1_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4_));
v___x_1932_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f___closed__0));
lean_inc(v_x_1920_);
v___x_1933_ = l_Lean_Syntax_isOfKind(v_x_1920_, v___x_1932_);
if (v___x_1933_ == 0)
{
lean_object* v___x_1934_; 
lean_dec(v_x_1920_);
v___x_1934_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__0___redArg();
return v___x_1934_;
}
else
{
lean_object* v___f_1935_; lean_object* v___x_1936_; lean_object* v_tk_1937_; lean_object* v___x_1938_; lean_object* v___x_1939_; lean_object* v___y_1941_; lean_object* v___x_1948_; lean_object* v___x_1949_; lean_object* v___x_1950_; 
v___f_1935_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___closed__0));
v___x_1936_ = lean_unsigned_to_nat(0u);
v_tk_1937_ = l_Lean_Syntax_getArg(v_x_1920_, v___x_1936_);
v___x_1938_ = lean_unsigned_to_nat(1u);
v___x_1939_ = l_Lean_Syntax_getArg(v_x_1920_, v___x_1938_);
v___x_1948_ = lean_unsigned_to_nat(2u);
v___x_1949_ = l_Lean_Syntax_getArg(v_x_1920_, v___x_1948_);
lean_dec(v_x_1920_);
v___x_1950_ = l_Lean_Syntax_getOptional_x3f(v___x_1949_);
lean_dec(v___x_1949_);
if (lean_obj_tag(v___x_1950_) == 0)
{
lean_object* v___x_1951_; 
v___x_1951_ = lean_box(0);
v___y_1941_ = v___x_1951_;
goto v___jp_1940_;
}
else
{
lean_object* v_val_1952_; lean_object* v___x_1954_; uint8_t v_isShared_1955_; uint8_t v_isSharedCheck_1959_; 
v_val_1952_ = lean_ctor_get(v___x_1950_, 0);
v_isSharedCheck_1959_ = !lean_is_exclusive(v___x_1950_);
if (v_isSharedCheck_1959_ == 0)
{
v___x_1954_ = v___x_1950_;
v_isShared_1955_ = v_isSharedCheck_1959_;
goto v_resetjp_1953_;
}
else
{
lean_inc(v_val_1952_);
lean_dec(v___x_1950_);
v___x_1954_ = lean_box(0);
v_isShared_1955_ = v_isSharedCheck_1959_;
goto v_resetjp_1953_;
}
v_resetjp_1953_:
{
lean_object* v___x_1957_; 
if (v_isShared_1955_ == 0)
{
v___x_1957_ = v___x_1954_;
goto v_reusejp_1956_;
}
else
{
lean_object* v_reuseFailAlloc_1958_; 
v_reuseFailAlloc_1958_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1958_, 0, v_val_1952_);
v___x_1957_ = v_reuseFailAlloc_1958_;
goto v_reusejp_1956_;
}
v_reusejp_1956_:
{
v___y_1941_ = v___x_1957_;
goto v___jp_1940_;
}
}
}
v___jp_1940_:
{
lean_object* v___x_1942_; lean_object* v___x_1943_; lean_object* v___x_1944_; lean_object* v___x_1945_; lean_object* v___f_1946_; lean_object* v___x_1947_; 
v___x_1942_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__0));
v___x_1943_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______macroRules__Lean__Parser__Term__app__1___closed__1));
v___x_1944_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___closed__3, &lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___closed__3);
v___x_1945_ = lean_box(v___x_1933_);
v___f_1946_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___lam__4___boxed), 20, 11);
lean_closure_set(v___f_1946_, 0, v___x_1939_);
lean_closure_set(v___f_1946_, 1, v___x_1944_);
lean_closure_set(v___f_1946_, 2, v___x_1945_);
lean_closure_set(v___f_1946_, 3, v_tk_1937_);
lean_closure_set(v___f_1946_, 4, v___x_1936_);
lean_closure_set(v___f_1946_, 5, v___x_1942_);
lean_closure_set(v___f_1946_, 6, v___x_1943_);
lean_closure_set(v___f_1946_, 7, v___x_1930_);
lean_closure_set(v___f_1946_, 8, v___f_1935_);
lean_closure_set(v___f_1946_, 9, v___y_1941_);
lean_closure_set(v___f_1946_, 10, v___x_1931_);
v___x_1947_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_1946_, v_a_1921_, v_a_1922_, v_a_1923_, v_a_1924_, v_a_1925_, v_a_1926_, v_a_1927_, v_a_1928_);
return v___x_1947_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1___boxed(lean_object* v_x_1960_, lean_object* v_a_1961_, lean_object* v_a_1962_, lean_object* v_a_1963_, lean_object* v_a_1964_, lean_object* v_a_1965_, lean_object* v_a_1966_, lean_object* v_a_1967_, lean_object* v_a_1968_, lean_object* v_a_1969_){
_start:
{
lean_object* v_res_1970_; 
v_res_1970_ = lp_mathlib_Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1(v_x_1960_, v_a_1961_, v_a_1962_, v_a_1963_, v_a_1964_, v_a_1965_, v_a_1966_, v_a_1967_, v_a_1968_);
lean_dec(v_a_1968_);
lean_dec_ref(v_a_1967_);
lean_dec(v_a_1966_);
lean_dec_ref(v_a_1965_);
lean_dec(v_a_1964_);
lean_dec_ref(v_a_1963_);
lean_dec(v_a_1962_);
lean_dec_ref(v_a_1961_);
return v_res_1970_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__1_spec__1(lean_object* v_ref_1971_, lean_object* v_msgData_1972_, uint8_t v_severity_1973_, uint8_t v_isSilent_1974_, lean_object* v___y_1975_, lean_object* v___y_1976_, lean_object* v___y_1977_, lean_object* v___y_1978_, lean_object* v___y_1979_, lean_object* v___y_1980_, lean_object* v___y_1981_, lean_object* v___y_1982_){
_start:
{
lean_object* v___x_1984_; 
v___x_1984_ = lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__1_spec__1___redArg(v_ref_1971_, v_msgData_1972_, v_severity_1973_, v_isSilent_1974_, v___y_1979_, v___y_1980_, v___y_1981_, v___y_1982_);
return v___x_1984_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__1_spec__1___boxed(lean_object* v_ref_1985_, lean_object* v_msgData_1986_, lean_object* v_severity_1987_, lean_object* v_isSilent_1988_, lean_object* v___y_1989_, lean_object* v___y_1990_, lean_object* v___y_1991_, lean_object* v___y_1992_, lean_object* v___y_1993_, lean_object* v___y_1994_, lean_object* v___y_1995_, lean_object* v___y_1996_, lean_object* v___y_1997_){
_start:
{
uint8_t v_severity_boxed_1998_; uint8_t v_isSilent_boxed_1999_; lean_object* v_res_2000_; 
v_severity_boxed_1998_ = lean_unbox(v_severity_1987_);
v_isSilent_boxed_1999_ = lean_unbox(v_isSilent_1988_);
v_res_2000_ = lp_mathlib_Lean_logAt___at___00Lean_logInfoAt___at___00Mathlib_Tactic_Erw_x3f___aux__Mathlib__Tactic__ErwQuestion______elabRules__Mathlib__Tactic__Erw_x3f__erw_x3f__1_spec__1_spec__1(v_ref_1985_, v_msgData_1986_, v_severity_boxed_1998_, v_isSilent_boxed_1999_, v___y_1989_, v___y_1990_, v___y_1991_, v___y_1992_, v___y_1993_, v___y_1994_, v___y_1995_, v___y_1996_);
lean_dec(v___y_1996_);
lean_dec_ref(v___y_1995_);
lean_dec(v___y_1994_);
lean_dec_ref(v___y_1993_);
lean_dec(v___y_1992_);
lean_dec_ref(v___y_1991_);
lean_dec(v___y_1990_);
lean_dec_ref(v___y_1989_);
lean_dec(v_ref_1985_);
return v_res_2000_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ErwQuestion(uint8_t builtin) {
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
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Rewrite(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_ErwQuestion(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Rewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_ErwQuestion_0__Mathlib_Tactic_Erw_x3f_initFn_00___x40_Mathlib_Tactic_ErwQuestion_844478417____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Tactic_Erw_x3f_tactic_erw_x3f_verbose = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Erw_x3f_tactic_erw_x3f_verbose);
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f = _init_lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Erw_x3f_erw_x3f);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Rewrite(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_ErwQuestion(uint8_t builtin) {
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
res = initialize_Lean_Elab_Tactic_Rewrite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ErwQuestion(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_ErwQuestion(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_ErwQuestion(builtin);
}
#ifdef __cplusplus
}
#endif
