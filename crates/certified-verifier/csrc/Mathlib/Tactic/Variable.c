// Lean compiler output
// Module: Mathlib.Tactic.Variable
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Meta.Tactic.TryThis public meta import Lean.Linter.UnusedVariables
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
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
lean_object* l_Lean_Expr_getAppFn(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_registerTagAttribute(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
uint8_t l_Lean_TagAttribute_hasTag(lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAuxAux(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* l_Lean_Elab_Term_elabType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_synthesizeSyntheticMVars(uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_List_isEmpty___redArg(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_Elab_Term_getSyntheticMVarDecl_x3f___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasExprMVar(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
size_t lean_array_size(lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
lean_object* l_Lean_Meta_mkForallFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_delab(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalContext_getFVarIds(lean_object*);
lean_object* l_Lean_MVarId_getDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
uint8_t l_Lean_instBEqFVarId_beq(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_unsetTrailing(lean_object*);
extern lean_object* l_Lean_maxRecDepth;
lean_object* l_Lean_Elab_Term_withAutoBoundImplicit___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Kernel_enableDiag(lean_object*, uint8_t);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Array_extract___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_insertIdx_loop(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_structEq(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_observing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_applyResult___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_synthesizeSyntheticMVarsNoPostponing(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVarAt(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* l_Lean_Meta_trySynthInstance(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_set(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_withoutAutoBoundImplicit___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
lean_object* l_Lean_Elab_Term_elabBinders___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_Term_checkBinderAnnotations;
lean_object* l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Name_isPrefixOf(lean_object*, lean_object*);
extern lean_object* l_Lean_diagnostics;
uint8_t l_Lean_Kernel_isDiagnosticsEnabled(lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_Stack_matches(lean_object*, lean_object*);
lean_object* l_Array_zip___redArg(lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_Lean_Meta_AbstractMVarsResult_numMVars(lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_Elab_Command_getBracketedBinderIds___redArg(lean_object*);
lean_object* l_Lean_Elab_Command_getCurrMacroScope___redArg(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_withFreshMacroScope___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_modifyScope___redArg(lean_object*, lean_object*);
lean_object* l_Lean_registerTraceClass(lean_object*, uint8_t, lean_object*);
extern lean_object* l_Lean_MessageData_nil;
lean_object* l_Lean_Meta_Tactic_TryThis_addSuggestion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_liftTermElabM___redArg(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_inheritedTraceOptions;
lean_object* l_Lean_LocalContext_getFVars(lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l_Lean_Meta_abstractMVars(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_mk(lean_object*);
lean_object* l_Lean_Elab_Term_withAutoBoundImplicit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Elab_Term_TermElabM_0__Lean_Elab_Term_withoutModifyingStateWithInfoAndMessagesImpl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_runTermElabM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__0_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "variable\?"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__0_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__0_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__1_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__0_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(54, 67, 10, 217, 96, 120, 114, 192)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__1_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__1_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__2_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__2_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__2_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__3_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__2_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__3_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__3_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__5_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__3_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__5_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__5_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__6_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__6_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__6_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__7_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__5_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__6_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__7_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__7_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__8_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Variable"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__8_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__8_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__9_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__7_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__8_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(255, 8, 145, 1, 193, 251, 37, 161)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__9_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__9_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__10_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__9_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(178, 172, 118, 74, 10, 5, 216, 42)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__10_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__10_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__11_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__10_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(99, 192, 71, 128, 99, 132, 12, 180)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__11_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__11_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__12_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__12_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__12_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__13_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__11_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__12_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(130, 18, 162, 66, 161, 171, 48, 85)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__13_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__13_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__14_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__13_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__8_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(86, 194, 188, 48, 251, 36, 190, 165)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__14_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__14_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__15_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__15_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__15_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__16_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__14_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__15_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(187, 143, 203, 28, 205, 79, 66, 127)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__16_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__16_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__17_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__17_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__17_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__18_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__16_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__17_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(158, 180, 65, 167, 118, 23, 244, 34)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__18_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__18_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__19_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__18_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(87, 241, 36, 63, 252, 79, 193, 247)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__19_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__19_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__20_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__19_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__6_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(150, 115, 195, 32, 83, 68, 194, 225)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__20_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__20_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__21_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__20_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__8_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(178, 77, 197, 41, 87, 209, 9, 167)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__21_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__21_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__22_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__21_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),((lean_object*)(((size_t)(1422233303) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(170, 170, 2, 227, 34, 44, 131, 37)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__22_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__22_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__23_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__23_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__23_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__24_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__22_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__23_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(229, 150, 76, 137, 141, 104, 222, 233)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__24_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__24_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__25_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__25_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__25_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__26_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__24_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__25_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(5, 227, 15, 225, 106, 74, 154, 53)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__26_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__26_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__27_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__26_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),((lean_object*)(((size_t)(2) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(72, 254, 228, 242, 75, 83, 87, 216)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__27_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__27_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__0_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "maxSteps"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__0_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__0_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__1_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__0_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(54, 67, 10, 217, 96, 120, 114, 192)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__1_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__1_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__0_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(62, 228, 154, 30, 117, 13, 126, 197)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__1_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__1_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__2_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 89, .m_capacity = 89, .m_length = 88, .m_data = "The maximum number of instance arguments `variable\?` will try to insert before giving up"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__2_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__2_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__3_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(15) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__2_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__3_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__3_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__12_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(27, 120, 182, 123, 91, 116, 151, 218)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__8_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(115, 30, 127, 137, 86, 116, 144, 69)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__0_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(206, 190, 234, 251, 59, 212, 130, 203)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__0_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(22, 236, 192, 22, 203, 206, 69, 171)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_variable_x3f_maxSteps;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__0_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "checkRedundant"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__0_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__0_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__1_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__0_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(54, 67, 10, 217, 96, 120, 114, 192)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__1_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__1_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__0_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(28, 9, 194, 54, 25, 125, 222, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__1_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__1_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__2_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 63, .m_capacity = 63, .m_length = 62, .m_data = "Warn if instance arguments can be inferred from preceding ones"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__2_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__2_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__3_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__2_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__3_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__3_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__12_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(27, 120, 182, 123, 91, 116, 151, 218)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__8_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(115, 30, 127, 137, 86, 116, 144, 69)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__0_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(206, 190, 234, 251, 59, 212, 130, 203)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__0_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(228, 204, 79, 39, 152, 252, 87, 254)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_variable_x3f_checkRedundant;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "explicitBinder"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__3_value),LEAN_SCALAR_PTR_LITERAL(49, 119, 193, 23, 170, 93, 183, 238)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "implicitBinder"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__6_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__6_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__5_value),LEAN_SCALAR_PTR_LITERAL(39, 181, 62, 102, 86, 14, 161, 96)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "strictImplicitBinder"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__8_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__8_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__7_value),LEAN_SCALAR_PTR_LITERAL(125, 223, 215, 186, 222, 17, 242, 189)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "instBinder"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__10_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__10_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__10_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__9_value),LEAN_SCALAR_PTR_LITERAL(198, 219, 89, 171, 221, 95, 22, 227)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__10_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_bracketedBinderType(lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__12_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(27, 120, 182, 123, 91, 116, 151, 218)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__0_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__8_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(115, 30, 127, 137, 86, 116, 144, 69)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__0_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__0_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(206, 190, 234, 251, 59, 212, 130, 203)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__1_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__0_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "many"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__4_value),LEAN_SCALAR_PTR_LITERAL(41, 35, 40, 86, 189, 97, 244, 31)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__6_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "bracketedBinder"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__9_value),LEAN_SCALAR_PTR_LITERAL(126, 188, 9, 177, 18, 110, 216, 30)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__15_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__16_value;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " =>"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__18_value),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__16_value),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__14_value),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__0_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__21_value)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__22_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Command_Variable_variable_x3f = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__22_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___lam__0_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___lam__0_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__0_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___lam__0_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2____boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__0_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__0_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__1_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "variable_alias"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__1_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__1_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__2_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__1_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(219, 205, 4, 220, 160, 44, 212, 40)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__2_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__2_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__3_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 57, .m_capacity = 57, .m_length = 56, .m_data = "Attribute to record aliases for the `variable\?` command."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__3_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__3_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "variableAliasAttr"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__5_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__5_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__5_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__12_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(27, 120, 182, 123, 91, 116, 151, 218)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__5_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__5_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__8_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(115, 30, 127, 137, 86, 116, 144, 69)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__5_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__5_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(39, 22, 19, 204, 112, 254, 170, 218)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__5_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__5_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_variableAliasAttr;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__6___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__6___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__6___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__6___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__6___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__6___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__6___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__6___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__6___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__6___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__6(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__5___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__1___redArg___closed__0 = (const lean_object*)&lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__1___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__1___redArg(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_pendingActionableSynthMVar___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "Cannot satisfy requirements for "};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_pendingActionableSynthMVar___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_pendingActionableSynthMVar___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Command_Variable_pendingActionableSynthMVar___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Command_Variable_pendingActionableSynthMVar___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_pendingActionableSynthMVar___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = " due to metavariables."};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_pendingActionableSynthMVar___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_pendingActionableSynthMVar___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Command_Variable_pendingActionableSynthMVar___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Command_Variable_pendingActionableSynthMVar___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_pendingActionableSynthMVar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_pendingActionableSynthMVar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__1(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Command_Variable_getSubproblem_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Command_Variable_getSubproblem_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Command_Variable_getSubproblem_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Command_Variable_getSubproblem_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Command_Variable_getSubproblem_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Command_Variable_getSubproblem_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_Variable_getSubproblem_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_Variable_getSubproblem_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Mathlib_Command_Variable_getSubproblem_spec__0_spec__0(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Mathlib_Command_Variable_getSubproblem_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_contains___at___00Mathlib_Command_Variable_getSubproblem_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_contains___at___00Mathlib_Command_Variable_getSubproblem_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Command_Variable_getSubproblem_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Command_Variable_getSubproblem_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg___closed__0;
static const lean_string_object lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg___closed__1_value;
static const lean_array_object lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__4_value;
static const lean_array_object lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__8;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "Actionable mvar:"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__9_value;
static lean_once_cell_t lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__10;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_getSubproblem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_getSubproblem___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias_spec__0___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias_spec__0(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Command_Variable_completeBinders_x27_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Command_Variable_completeBinders_x27_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Command_Variable_completeBinders_x27_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Command_Variable_completeBinders_x27_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Mathlib_Command_Variable_completeBinders_x27_spec__7(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Mathlib_Command_Variable_completeBinders_x27_spec__7___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Command_Variable_completeBinders_x27_spec__8(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "unsolvedGoals"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "synthPlaceholder"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "lean"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "inductionWithNoAlts"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "_namedError"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___lam__0___closed__5_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 59, .m_capacity = 59, .m_length = 58, .m_data = "Instance argument can be inferred from earlier arguments.\n"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "{"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hole"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "}"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 53, .m_capacity = 53, .m_length = 52, .m_data = "`variable_alias` binders can't have an explicit name"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__8;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2(lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_Variable_completeBinders_x27_spec__4___redArg(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_Variable_completeBinders_x27_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Command_Variable_completeBinders_x27_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00Mathlib_Command_Variable_completeBinders_x27_spec__6_spec__7(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00Mathlib_Command_Variable_completeBinders_x27_spec__6_spec__7___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_set___at___00Mathlib_Command_Variable_completeBinders_x27_spec__6(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_set___at___00Mathlib_Command_Variable_completeBinders_x27_spec__6___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Command_Variable_completeBinders_x27_spec__1(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Command_Variable_completeBinders_x27_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__1;
static lean_once_cell_t lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "variable"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__12_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__3_value),LEAN_SCALAR_PTR_LITERAL(250, 93, 226, 106, 76, 14, 69, 165)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 126, .m_capacity = 126, .m_length = 125, .m_data = "Maximum recursion depth for variables! reached. This might be a bug, or you can try adjusting `set_option variable\?.maxSteps "};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__6;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "`\n\nCurrent variable command:"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__7_value;
static lean_once_cell_t lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__8;
static const lean_closure_object lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__0___boxed, .m_arity = 8, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__1_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value)} };
static const lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Binder"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__10_value;
static lean_once_cell_t lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__11;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 79, .m_capacity = 79, .m_length = 78, .m_data = "\nwas not able to satisfy one of its dependencies using the pre-existing binder"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__12_value;
static lean_once_cell_t lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__13;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 192, .m_capacity = 192, .m_length = 191, .m_data = "\n\nThis might be due to differences in implicit arguments, which are not represented in binders since they are generated by pretty printing unsatisfied dependencies.\n\nCurrent variable command:"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__14_value;
static lean_once_cell_t lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__15;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 48, .m_capacity = 48, .m_length = 47, .m_data = "\n\nLocal context for the unsatisfied dependency:"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__16_value;
static lean_once_cell_t lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__17;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "new subproblem:"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__18_value;
static lean_once_cell_t lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__19;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__3___boxed(lean_object**);
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "elaborated binder types array = "};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__4___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__4___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__4___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__4___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__4(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__4___boxed(lean_object**);
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "value is none"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__22_value;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Option.get!"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__21_value;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Init.Data.Option.BasicAux"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__20_value;
static lean_once_cell_t lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__23;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Have "};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__24_value;
static lean_once_cell_t lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__25;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = " fvars and "};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__26_value;
static lean_once_cell_t lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__27;
static const lean_string_object lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = " local instances. Looking at"};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__28_value;
static lean_once_cell_t lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__29;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__3(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_Variable_completeBinders_x27_spec__4(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_Variable_completeBinders_x27_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Mathlib_Command_Variable_completeBinders___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_completeBinders___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Command_Variable_cleanBinders_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Command_Variable_cleanBinders_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Mathlib_Command_Variable_cleanBinders___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_cleanBinders___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_cleanBinders___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_cleanBinders(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_cleanBinders___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__1___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__1(size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__2___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__2___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__2(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__1___closed__0;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__2_spec__3___redArg(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__1_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__4(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 71, .m_capacity = 71, .m_length = 70, .m_data = "Calculated binders do not match the expected binders given after `=>`."};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__0_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__2;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 32, .m_capacity = 32, .m_length = 31, .m_data = "expected context: paramNames = "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__4;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__6;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__7;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "numMVars = "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__8_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__9;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__10_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__11;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "new context: paramNames = "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__12_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__13;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "checking expected binders"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__14_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__15;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5_spec__7_spec__8___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5_spec__7_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5_spec__7___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 47, .m_capacity = 47, .m_length = 46, .m_data = "variable\? cannot update pre-existing variables"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__6___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__6___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__6___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__6___closed__1;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__6(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Try this:"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__0_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__0___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__1_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__1___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__2_value;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__2___boxed, .m_arity = 9, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__2_value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=>"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "command"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__5_value),LEAN_SCALAR_PTR_LITERAL(29, 69, 134, 125, 237, 175, 69, 70)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "derived"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__7_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__8;
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__4___boxed, .m_arity = 8, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__1_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2__value)} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__9_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "variable\?.maxSteps = "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__10_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__11;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__2_spec__3(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5_spec__7(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5_spec__7_spec__8(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5_spec__7_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Command_Variable_elabVariables_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Command_Variable_elabVariables_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Command_Variable_elabVariables_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Command_Variable_elabVariables_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Command_Variable_elabVariables_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Command_Variable_elabVariables_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_elabVariables(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_elabVariables___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__8_value;
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_65_; uint8_t v___x_66_; lean_object* v___x_67_; lean_object* v___x_68_; 
v___x_65_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__1_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_));
v___x_66_ = 0;
v___x_67_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__27_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_));
v___x_68_ = l_Lean_registerTraceClass(v___x_65_, v___x_66_, v___x_67_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2____boxed(lean_object* v_a_69_){
_start:
{
lean_object* v_res_70_; 
v_res_70_ = lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_();
return v_res_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4__spec__0(lean_object* v_name_71_, lean_object* v_decl_72_, lean_object* v_ref_73_){
_start:
{
lean_object* v_defValue_75_; lean_object* v_descr_76_; lean_object* v_deprecation_x3f_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; 
v_defValue_75_ = lean_ctor_get(v_decl_72_, 0);
v_descr_76_ = lean_ctor_get(v_decl_72_, 1);
v_deprecation_x3f_77_ = lean_ctor_get(v_decl_72_, 2);
lean_inc(v_defValue_75_);
v___x_78_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_78_, 0, v_defValue_75_);
lean_inc(v_deprecation_x3f_77_);
lean_inc_ref(v_descr_76_);
lean_inc_n(v_name_71_, 2);
v___x_79_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_79_, 0, v_name_71_);
lean_ctor_set(v___x_79_, 1, v_ref_73_);
lean_ctor_set(v___x_79_, 2, v___x_78_);
lean_ctor_set(v___x_79_, 3, v_descr_76_);
lean_ctor_set(v___x_79_, 4, v_deprecation_x3f_77_);
v___x_80_ = lean_register_option(v_name_71_, v___x_79_);
if (lean_obj_tag(v___x_80_) == 0)
{
lean_object* v___x_82_; uint8_t v_isShared_83_; uint8_t v_isSharedCheck_88_; 
v_isSharedCheck_88_ = !lean_is_exclusive(v___x_80_);
if (v_isSharedCheck_88_ == 0)
{
lean_object* v_unused_89_; 
v_unused_89_ = lean_ctor_get(v___x_80_, 0);
lean_dec(v_unused_89_);
v___x_82_ = v___x_80_;
v_isShared_83_ = v_isSharedCheck_88_;
goto v_resetjp_81_;
}
else
{
lean_dec(v___x_80_);
v___x_82_ = lean_box(0);
v_isShared_83_ = v_isSharedCheck_88_;
goto v_resetjp_81_;
}
v_resetjp_81_:
{
lean_object* v___x_84_; lean_object* v___x_86_; 
lean_inc(v_defValue_75_);
v___x_84_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_84_, 0, v_name_71_);
lean_ctor_set(v___x_84_, 1, v_defValue_75_);
if (v_isShared_83_ == 0)
{
lean_ctor_set(v___x_82_, 0, v___x_84_);
v___x_86_ = v___x_82_;
goto v_reusejp_85_;
}
else
{
lean_object* v_reuseFailAlloc_87_; 
v_reuseFailAlloc_87_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_87_, 0, v___x_84_);
v___x_86_ = v_reuseFailAlloc_87_;
goto v_reusejp_85_;
}
v_reusejp_85_:
{
return v___x_86_;
}
}
}
else
{
lean_object* v_a_90_; lean_object* v___x_92_; uint8_t v_isShared_93_; uint8_t v_isSharedCheck_97_; 
lean_dec(v_name_71_);
v_a_90_ = lean_ctor_get(v___x_80_, 0);
v_isSharedCheck_97_ = !lean_is_exclusive(v___x_80_);
if (v_isSharedCheck_97_ == 0)
{
v___x_92_ = v___x_80_;
v_isShared_93_ = v_isSharedCheck_97_;
goto v_resetjp_91_;
}
else
{
lean_inc(v_a_90_);
lean_dec(v___x_80_);
v___x_92_ = lean_box(0);
v_isShared_93_ = v_isSharedCheck_97_;
goto v_resetjp_91_;
}
v_resetjp_91_:
{
lean_object* v___x_95_; 
if (v_isShared_93_ == 0)
{
v___x_95_ = v___x_92_;
goto v_reusejp_94_;
}
else
{
lean_object* v_reuseFailAlloc_96_; 
v_reuseFailAlloc_96_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_96_, 0, v_a_90_);
v___x_95_ = v_reuseFailAlloc_96_;
goto v_reusejp_94_;
}
v_reusejp_94_:
{
return v___x_95_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_98_, lean_object* v_decl_99_, lean_object* v_ref_100_, lean_object* v_a_101_){
_start:
{
lean_object* v_res_102_; 
v_res_102_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4__spec__0(v_name_98_, v_decl_99_, v_ref_100_);
lean_dec_ref(v_decl_99_);
return v_res_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; 
v___x_119_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__1_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4_));
v___x_120_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__3_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4_));
v___x_121_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4_));
v___x_122_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4__spec__0(v___x_119_, v___x_120_, v___x_121_);
return v___x_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4____boxed(lean_object* v_a_123_){
_start:
{
lean_object* v_res_124_; 
v_res_124_ = lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4_();
return v_res_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4__spec__0(lean_object* v_name_125_, lean_object* v_decl_126_, lean_object* v_ref_127_){
_start:
{
lean_object* v_defValue_129_; lean_object* v_descr_130_; lean_object* v_deprecation_x3f_131_; lean_object* v___x_132_; uint8_t v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; 
v_defValue_129_ = lean_ctor_get(v_decl_126_, 0);
v_descr_130_ = lean_ctor_get(v_decl_126_, 1);
v_deprecation_x3f_131_ = lean_ctor_get(v_decl_126_, 2);
v___x_132_ = lean_alloc_ctor(1, 0, 1);
v___x_133_ = lean_unbox(v_defValue_129_);
lean_ctor_set_uint8(v___x_132_, 0, v___x_133_);
lean_inc(v_deprecation_x3f_131_);
lean_inc_ref(v_descr_130_);
lean_inc_n(v_name_125_, 2);
v___x_134_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_134_, 0, v_name_125_);
lean_ctor_set(v___x_134_, 1, v_ref_127_);
lean_ctor_set(v___x_134_, 2, v___x_132_);
lean_ctor_set(v___x_134_, 3, v_descr_130_);
lean_ctor_set(v___x_134_, 4, v_deprecation_x3f_131_);
v___x_135_ = lean_register_option(v_name_125_, v___x_134_);
if (lean_obj_tag(v___x_135_) == 0)
{
lean_object* v___x_137_; uint8_t v_isShared_138_; uint8_t v_isSharedCheck_143_; 
v_isSharedCheck_143_ = !lean_is_exclusive(v___x_135_);
if (v_isSharedCheck_143_ == 0)
{
lean_object* v_unused_144_; 
v_unused_144_ = lean_ctor_get(v___x_135_, 0);
lean_dec(v_unused_144_);
v___x_137_ = v___x_135_;
v_isShared_138_ = v_isSharedCheck_143_;
goto v_resetjp_136_;
}
else
{
lean_dec(v___x_135_);
v___x_137_ = lean_box(0);
v_isShared_138_ = v_isSharedCheck_143_;
goto v_resetjp_136_;
}
v_resetjp_136_:
{
lean_object* v___x_139_; lean_object* v___x_141_; 
lean_inc(v_defValue_129_);
v___x_139_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_139_, 0, v_name_125_);
lean_ctor_set(v___x_139_, 1, v_defValue_129_);
if (v_isShared_138_ == 0)
{
lean_ctor_set(v___x_137_, 0, v___x_139_);
v___x_141_ = v___x_137_;
goto v_reusejp_140_;
}
else
{
lean_object* v_reuseFailAlloc_142_; 
v_reuseFailAlloc_142_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_142_, 0, v___x_139_);
v___x_141_ = v_reuseFailAlloc_142_;
goto v_reusejp_140_;
}
v_reusejp_140_:
{
return v___x_141_;
}
}
}
else
{
lean_object* v_a_145_; lean_object* v___x_147_; uint8_t v_isShared_148_; uint8_t v_isSharedCheck_152_; 
lean_dec(v_name_125_);
v_a_145_ = lean_ctor_get(v___x_135_, 0);
v_isSharedCheck_152_ = !lean_is_exclusive(v___x_135_);
if (v_isSharedCheck_152_ == 0)
{
v___x_147_ = v___x_135_;
v_isShared_148_ = v_isSharedCheck_152_;
goto v_resetjp_146_;
}
else
{
lean_inc(v_a_145_);
lean_dec(v___x_135_);
v___x_147_ = lean_box(0);
v_isShared_148_ = v_isSharedCheck_152_;
goto v_resetjp_146_;
}
v_resetjp_146_:
{
lean_object* v___x_150_; 
if (v_isShared_148_ == 0)
{
v___x_150_ = v___x_147_;
goto v_reusejp_149_;
}
else
{
lean_object* v_reuseFailAlloc_151_; 
v_reuseFailAlloc_151_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_151_, 0, v_a_145_);
v___x_150_ = v_reuseFailAlloc_151_;
goto v_reusejp_149_;
}
v_reusejp_149_:
{
return v___x_150_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_153_, lean_object* v_decl_154_, lean_object* v_ref_155_, lean_object* v_a_156_){
_start:
{
lean_object* v_res_157_; 
v_res_157_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4__spec__0(v_name_153_, v_decl_154_, v_ref_155_);
lean_dec_ref(v_decl_154_);
return v_res_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; 
v___x_175_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__1_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4_));
v___x_176_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__3_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4_));
v___x_177_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__4_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4_));
v___x_178_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4__spec__0(v___x_175_, v___x_176_, v___x_177_);
return v___x_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4____boxed(lean_object* v_a_179_){
_start:
{
lean_object* v_res_180_; 
v_res_180_ = lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4_();
return v_res_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_bracketedBinderType(lean_object* v_x_208_){
_start:
{
lean_object* v___x_213_; uint8_t v___x_214_; 
v___x_213_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__4));
lean_inc(v_x_208_);
v___x_214_ = l_Lean_Syntax_isOfKind(v_x_208_, v___x_213_);
if (v___x_214_ == 0)
{
lean_object* v___x_215_; uint8_t v___x_216_; 
v___x_215_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__6));
lean_inc(v_x_208_);
v___x_216_ = l_Lean_Syntax_isOfKind(v_x_208_, v___x_215_);
if (v___x_216_ == 0)
{
lean_object* v___x_217_; uint8_t v___x_218_; 
v___x_217_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__8));
lean_inc(v_x_208_);
v___x_218_ = l_Lean_Syntax_isOfKind(v_x_208_, v___x_217_);
if (v___x_218_ == 0)
{
lean_object* v___x_219_; uint8_t v___x_220_; 
v___x_219_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__10));
lean_inc(v_x_208_);
v___x_220_ = l_Lean_Syntax_isOfKind(v_x_208_, v___x_219_);
if (v___x_220_ == 0)
{
lean_object* v___x_221_; 
lean_dec(v_x_208_);
v___x_221_ = lean_box(0);
return v___x_221_;
}
else
{
lean_object* v___x_222_; lean_object* v___x_223_; uint8_t v___x_224_; 
v___x_222_ = lean_unsigned_to_nat(1u);
v___x_223_ = l_Lean_Syntax_getArg(v_x_208_, v___x_222_);
v___x_224_ = l_Lean_Syntax_isNone(v___x_223_);
if (v___x_224_ == 0)
{
lean_object* v___x_225_; uint8_t v___x_226_; 
v___x_225_ = lean_unsigned_to_nat(2u);
v___x_226_ = l_Lean_Syntax_matchesNull(v___x_223_, v___x_225_);
if (v___x_226_ == 0)
{
lean_object* v___x_227_; 
lean_dec(v_x_208_);
v___x_227_ = lean_box(0);
return v___x_227_;
}
else
{
goto v___jp_209_;
}
}
else
{
lean_dec(v___x_223_);
goto v___jp_209_;
}
}
}
else
{
lean_object* v___x_228_; lean_object* v___x_229_; uint8_t v___x_230_; 
v___x_228_ = lean_unsigned_to_nat(2u);
v___x_229_ = l_Lean_Syntax_getArg(v_x_208_, v___x_228_);
lean_dec(v_x_208_);
v___x_230_ = l_Lean_Syntax_isNone(v___x_229_);
if (v___x_230_ == 0)
{
uint8_t v___x_231_; 
lean_inc(v___x_229_);
v___x_231_ = l_Lean_Syntax_matchesNull(v___x_229_, v___x_228_);
if (v___x_231_ == 0)
{
lean_object* v___x_232_; 
lean_dec(v___x_229_);
v___x_232_ = lean_box(0);
return v___x_232_;
}
else
{
lean_object* v___x_233_; lean_object* v_ty_x3f_234_; lean_object* v___x_235_; 
v___x_233_ = lean_unsigned_to_nat(1u);
v_ty_x3f_234_ = l_Lean_Syntax_getArg(v___x_229_, v___x_233_);
lean_dec(v___x_229_);
v___x_235_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_235_, 0, v_ty_x3f_234_);
return v___x_235_;
}
}
else
{
lean_object* v___x_236_; 
lean_dec(v___x_229_);
v___x_236_ = lean_box(0);
return v___x_236_;
}
}
}
else
{
lean_object* v___x_237_; lean_object* v___x_238_; uint8_t v___x_239_; 
v___x_237_ = lean_unsigned_to_nat(2u);
v___x_238_ = l_Lean_Syntax_getArg(v_x_208_, v___x_237_);
lean_dec(v_x_208_);
v___x_239_ = l_Lean_Syntax_isNone(v___x_238_);
if (v___x_239_ == 0)
{
uint8_t v___x_240_; 
lean_inc(v___x_238_);
v___x_240_ = l_Lean_Syntax_matchesNull(v___x_238_, v___x_237_);
if (v___x_240_ == 0)
{
lean_object* v___x_241_; 
lean_dec(v___x_238_);
v___x_241_ = lean_box(0);
return v___x_241_;
}
else
{
lean_object* v___x_242_; lean_object* v_ty_x3f_243_; lean_object* v___x_244_; 
v___x_242_ = lean_unsigned_to_nat(1u);
v_ty_x3f_243_ = l_Lean_Syntax_getArg(v___x_238_, v___x_242_);
lean_dec(v___x_238_);
v___x_244_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_244_, 0, v_ty_x3f_243_);
return v___x_244_;
}
}
else
{
lean_object* v___x_245_; 
lean_dec(v___x_238_);
v___x_245_ = lean_box(0);
return v___x_245_;
}
}
}
else
{
lean_object* v___x_246_; lean_object* v___x_247_; uint8_t v___x_248_; 
v___x_246_ = lean_unsigned_to_nat(2u);
v___x_247_ = l_Lean_Syntax_getArg(v_x_208_, v___x_246_);
lean_dec(v_x_208_);
v___x_248_ = l_Lean_Syntax_isNone(v___x_247_);
if (v___x_248_ == 0)
{
uint8_t v___x_249_; 
lean_inc(v___x_247_);
v___x_249_ = l_Lean_Syntax_matchesNull(v___x_247_, v___x_246_);
if (v___x_249_ == 0)
{
lean_object* v___x_250_; 
lean_dec(v___x_247_);
v___x_250_ = lean_box(0);
return v___x_250_;
}
else
{
lean_object* v___x_251_; lean_object* v_ty_x3f_252_; lean_object* v___x_253_; 
v___x_251_ = lean_unsigned_to_nat(1u);
v_ty_x3f_252_ = l_Lean_Syntax_getArg(v___x_247_, v___x_251_);
lean_dec(v___x_247_);
v___x_253_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_253_, 0, v_ty_x3f_252_);
return v___x_253_;
}
}
else
{
lean_object* v___x_254_; 
lean_dec(v___x_247_);
v___x_254_ = lean_box(0);
return v___x_254_;
}
}
v___jp_209_:
{
lean_object* v___x_210_; lean_object* v_ty_211_; lean_object* v___x_212_; 
v___x_210_ = lean_unsigned_to_nat(2u);
v_ty_211_ = l_Lean_Syntax_getArg(v_x_208_, v___x_210_);
lean_dec(v_x_208_);
v___x_212_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_212_, 0, v_ty_211_);
return v___x_212_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___lam__0_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2_(lean_object* v_x_311_, lean_object* v___y_312_, lean_object* v___y_313_){
_start:
{
lean_object* v___x_315_; lean_object* v___x_316_; 
v___x_315_ = lean_box(0);
v___x_316_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_316_, 0, v___x_315_);
return v___x_316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___lam__0_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2____boxed(lean_object* v_x_317_, lean_object* v___y_318_, lean_object* v___y_319_, lean_object* v___y_320_){
_start:
{
lean_object* v_res_321_; 
v_res_321_ = lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___lam__0_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2_(v_x_317_, v___y_318_, v___y_319_);
lean_dec(v___y_319_);
lean_dec_ref(v___y_318_);
lean_dec(v_x_317_);
return v_res_321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2_(){
_start:
{
lean_object* v___f_334_; lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; uint8_t v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; 
v___f_334_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__0_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2_));
v___x_335_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__2_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2_));
v___x_336_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__3_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2_));
v___x_337_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__5_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2_));
v___x_338_ = 0;
v___x_339_ = lean_box(2);
v___x_340_ = l_Lean_registerTagAttribute(v___x_335_, v___x_336_, v___f_334_, v___x_337_, v___x_338_, v___x_339_);
return v___x_340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2____boxed(lean_object* v_a_341_){
_start:
{
lean_object* v_res_342_; 
v_res_342_ = lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2_();
return v_res_342_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__0___redArg(lean_object* v_e_343_, lean_object* v___y_344_){
_start:
{
uint8_t v___x_346_; 
v___x_346_ = l_Lean_Expr_hasMVar(v_e_343_);
if (v___x_346_ == 0)
{
lean_object* v___x_347_; 
v___x_347_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_347_, 0, v_e_343_);
return v___x_347_;
}
else
{
lean_object* v___x_348_; lean_object* v_mctx_349_; lean_object* v___x_350_; lean_object* v_fst_351_; lean_object* v_snd_352_; lean_object* v___x_353_; lean_object* v_cache_354_; lean_object* v_zetaDeltaFVarIds_355_; lean_object* v_postponed_356_; lean_object* v_diag_357_; lean_object* v___x_359_; uint8_t v_isShared_360_; uint8_t v_isSharedCheck_366_; 
v___x_348_ = lean_st_ref_get(v___y_344_);
v_mctx_349_ = lean_ctor_get(v___x_348_, 0);
lean_inc_ref(v_mctx_349_);
lean_dec(v___x_348_);
v___x_350_ = l_Lean_instantiateMVarsCore(v_mctx_349_, v_e_343_);
v_fst_351_ = lean_ctor_get(v___x_350_, 0);
lean_inc(v_fst_351_);
v_snd_352_ = lean_ctor_get(v___x_350_, 1);
lean_inc(v_snd_352_);
lean_dec_ref(v___x_350_);
v___x_353_ = lean_st_ref_take(v___y_344_);
v_cache_354_ = lean_ctor_get(v___x_353_, 1);
v_zetaDeltaFVarIds_355_ = lean_ctor_get(v___x_353_, 2);
v_postponed_356_ = lean_ctor_get(v___x_353_, 3);
v_diag_357_ = lean_ctor_get(v___x_353_, 4);
v_isSharedCheck_366_ = !lean_is_exclusive(v___x_353_);
if (v_isSharedCheck_366_ == 0)
{
lean_object* v_unused_367_; 
v_unused_367_ = lean_ctor_get(v___x_353_, 0);
lean_dec(v_unused_367_);
v___x_359_ = v___x_353_;
v_isShared_360_ = v_isSharedCheck_366_;
goto v_resetjp_358_;
}
else
{
lean_inc(v_diag_357_);
lean_inc(v_postponed_356_);
lean_inc(v_zetaDeltaFVarIds_355_);
lean_inc(v_cache_354_);
lean_dec(v___x_353_);
v___x_359_ = lean_box(0);
v_isShared_360_ = v_isSharedCheck_366_;
goto v_resetjp_358_;
}
v_resetjp_358_:
{
lean_object* v___x_362_; 
if (v_isShared_360_ == 0)
{
lean_ctor_set(v___x_359_, 0, v_snd_352_);
v___x_362_ = v___x_359_;
goto v_reusejp_361_;
}
else
{
lean_object* v_reuseFailAlloc_365_; 
v_reuseFailAlloc_365_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_365_, 0, v_snd_352_);
lean_ctor_set(v_reuseFailAlloc_365_, 1, v_cache_354_);
lean_ctor_set(v_reuseFailAlloc_365_, 2, v_zetaDeltaFVarIds_355_);
lean_ctor_set(v_reuseFailAlloc_365_, 3, v_postponed_356_);
lean_ctor_set(v_reuseFailAlloc_365_, 4, v_diag_357_);
v___x_362_ = v_reuseFailAlloc_365_;
goto v_reusejp_361_;
}
v_reusejp_361_:
{
lean_object* v___x_363_; lean_object* v___x_364_; 
v___x_363_ = lean_st_ref_set(v___y_344_, v___x_362_);
v___x_364_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_364_, 0, v_fst_351_);
return v___x_364_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__0___redArg___boxed(lean_object* v_e_368_, lean_object* v___y_369_, lean_object* v___y_370_){
_start:
{
lean_object* v_res_371_; 
v_res_371_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__0___redArg(v_e_368_, v___y_369_);
lean_dec(v___y_369_);
return v_res_371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__0(lean_object* v_e_372_, lean_object* v___y_373_, lean_object* v___y_374_, lean_object* v___y_375_, lean_object* v___y_376_, lean_object* v___y_377_, lean_object* v___y_378_){
_start:
{
lean_object* v___x_380_; 
v___x_380_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__0___redArg(v_e_372_, v___y_376_);
return v___x_380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__0___boxed(lean_object* v_e_381_, lean_object* v___y_382_, lean_object* v___y_383_, lean_object* v___y_384_, lean_object* v___y_385_, lean_object* v___y_386_, lean_object* v___y_387_, lean_object* v___y_388_){
_start:
{
lean_object* v_res_389_; 
v_res_389_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__0(v_e_381_, v___y_382_, v___y_383_, v___y_384_, v___y_385_, v___y_386_, v___y_387_);
lean_dec(v___y_387_);
lean_dec_ref(v___y_386_);
lean_dec(v___y_385_);
lean_dec_ref(v___y_384_);
lean_dec(v___y_383_);
lean_dec_ref(v___y_382_);
return v_res_389_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__6___closed__0(void){
_start:
{
lean_object* v___x_390_; lean_object* v___x_391_; 
v___x_390_ = lean_box(1);
v___x_391_ = l_Lean_MessageData_ofFormat(v___x_390_);
return v___x_391_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__6___closed__3(void){
_start:
{
lean_object* v___x_395_; lean_object* v___x_396_; 
v___x_395_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__6___closed__2));
v___x_396_ = l_Lean_MessageData_ofFormat(v___x_395_);
return v___x_396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__6(lean_object* v_x_397_, lean_object* v_x_398_){
_start:
{
if (lean_obj_tag(v_x_398_) == 0)
{
return v_x_397_;
}
else
{
lean_object* v_head_399_; lean_object* v_tail_400_; lean_object* v___x_402_; uint8_t v_isShared_403_; uint8_t v_isSharedCheck_422_; 
v_head_399_ = lean_ctor_get(v_x_398_, 0);
v_tail_400_ = lean_ctor_get(v_x_398_, 1);
v_isSharedCheck_422_ = !lean_is_exclusive(v_x_398_);
if (v_isSharedCheck_422_ == 0)
{
v___x_402_ = v_x_398_;
v_isShared_403_ = v_isSharedCheck_422_;
goto v_resetjp_401_;
}
else
{
lean_inc(v_tail_400_);
lean_inc(v_head_399_);
lean_dec(v_x_398_);
v___x_402_ = lean_box(0);
v_isShared_403_ = v_isSharedCheck_422_;
goto v_resetjp_401_;
}
v_resetjp_401_:
{
lean_object* v_before_404_; lean_object* v___x_406_; uint8_t v_isShared_407_; uint8_t v_isSharedCheck_420_; 
v_before_404_ = lean_ctor_get(v_head_399_, 0);
v_isSharedCheck_420_ = !lean_is_exclusive(v_head_399_);
if (v_isSharedCheck_420_ == 0)
{
lean_object* v_unused_421_; 
v_unused_421_ = lean_ctor_get(v_head_399_, 1);
lean_dec(v_unused_421_);
v___x_406_ = v_head_399_;
v_isShared_407_ = v_isSharedCheck_420_;
goto v_resetjp_405_;
}
else
{
lean_inc(v_before_404_);
lean_dec(v_head_399_);
v___x_406_ = lean_box(0);
v_isShared_407_ = v_isSharedCheck_420_;
goto v_resetjp_405_;
}
v_resetjp_405_:
{
lean_object* v___x_408_; lean_object* v___x_410_; 
v___x_408_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__6___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__6___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__6___closed__0);
if (v_isShared_407_ == 0)
{
lean_ctor_set_tag(v___x_406_, 7);
lean_ctor_set(v___x_406_, 1, v___x_408_);
lean_ctor_set(v___x_406_, 0, v_x_397_);
v___x_410_ = v___x_406_;
goto v_reusejp_409_;
}
else
{
lean_object* v_reuseFailAlloc_419_; 
v_reuseFailAlloc_419_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_419_, 0, v_x_397_);
lean_ctor_set(v_reuseFailAlloc_419_, 1, v___x_408_);
v___x_410_ = v_reuseFailAlloc_419_;
goto v_reusejp_409_;
}
v_reusejp_409_:
{
lean_object* v___x_411_; lean_object* v___x_413_; 
v___x_411_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__6___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__6___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__6___closed__3);
if (v_isShared_403_ == 0)
{
lean_ctor_set_tag(v___x_402_, 7);
lean_ctor_set(v___x_402_, 1, v___x_411_);
lean_ctor_set(v___x_402_, 0, v___x_410_);
v___x_413_ = v___x_402_;
goto v_reusejp_412_;
}
else
{
lean_object* v_reuseFailAlloc_418_; 
v_reuseFailAlloc_418_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_418_, 0, v___x_410_);
lean_ctor_set(v_reuseFailAlloc_418_, 1, v___x_411_);
v___x_413_ = v_reuseFailAlloc_418_;
goto v_reusejp_412_;
}
v_reusejp_412_:
{
lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; 
v___x_414_ = l_Lean_MessageData_ofSyntax(v_before_404_);
v___x_415_ = l_Lean_indentD(v___x_414_);
v___x_416_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_416_, 0, v___x_413_);
lean_ctor_set(v___x_416_, 1, v___x_415_);
v_x_397_ = v___x_416_;
v_x_398_ = v_tail_400_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__5(lean_object* v_opts_423_, lean_object* v_opt_424_){
_start:
{
lean_object* v_name_425_; lean_object* v_defValue_426_; lean_object* v_map_427_; lean_object* v___x_428_; 
v_name_425_ = lean_ctor_get(v_opt_424_, 0);
v_defValue_426_ = lean_ctor_get(v_opt_424_, 1);
v_map_427_ = lean_ctor_get(v_opts_423_, 0);
v___x_428_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_427_, v_name_425_);
if (lean_obj_tag(v___x_428_) == 0)
{
uint8_t v___x_429_; 
v___x_429_ = lean_unbox(v_defValue_426_);
return v___x_429_;
}
else
{
lean_object* v_val_430_; 
v_val_430_ = lean_ctor_get(v___x_428_, 0);
lean_inc(v_val_430_);
lean_dec_ref_known(v___x_428_, 1);
if (lean_obj_tag(v_val_430_) == 1)
{
uint8_t v_v_431_; 
v_v_431_ = lean_ctor_get_uint8(v_val_430_, 0);
lean_dec_ref_known(v_val_430_, 0);
return v_v_431_;
}
else
{
uint8_t v___x_432_; 
lean_dec(v_val_430_);
v___x_432_ = lean_unbox(v_defValue_426_);
return v___x_432_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__5___boxed(lean_object* v_opts_433_, lean_object* v_opt_434_){
_start:
{
uint8_t v_res_435_; lean_object* v_r_436_; 
v_res_435_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__5(v_opts_433_, v_opt_434_);
lean_dec_ref(v_opt_434_);
lean_dec_ref(v_opts_433_);
v_r_436_ = lean_box(v_res_435_);
return v_r_436_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4___redArg___closed__2(void){
_start:
{
lean_object* v___x_440_; lean_object* v___x_441_; 
v___x_440_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4___redArg___closed__1));
v___x_441_ = l_Lean_MessageData_ofFormat(v___x_440_);
return v___x_441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4___redArg(lean_object* v_msgData_442_, lean_object* v_macroStack_443_, lean_object* v___y_444_){
_start:
{
lean_object* v_options_446_; lean_object* v___x_447_; uint8_t v___x_448_; 
v_options_446_ = lean_ctor_get(v___y_444_, 2);
v___x_447_ = l_Lean_Elab_pp_macroStack;
v___x_448_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__5(v_options_446_, v___x_447_);
if (v___x_448_ == 0)
{
lean_object* v___x_449_; 
lean_dec(v_macroStack_443_);
v___x_449_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_449_, 0, v_msgData_442_);
return v___x_449_;
}
else
{
if (lean_obj_tag(v_macroStack_443_) == 0)
{
lean_object* v___x_450_; 
v___x_450_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_450_, 0, v_msgData_442_);
return v___x_450_;
}
else
{
lean_object* v_head_451_; lean_object* v_after_452_; lean_object* v___x_454_; uint8_t v_isShared_455_; uint8_t v_isSharedCheck_467_; 
v_head_451_ = lean_ctor_get(v_macroStack_443_, 0);
lean_inc(v_head_451_);
v_after_452_ = lean_ctor_get(v_head_451_, 1);
v_isSharedCheck_467_ = !lean_is_exclusive(v_head_451_);
if (v_isSharedCheck_467_ == 0)
{
lean_object* v_unused_468_; 
v_unused_468_ = lean_ctor_get(v_head_451_, 0);
lean_dec(v_unused_468_);
v___x_454_ = v_head_451_;
v_isShared_455_ = v_isSharedCheck_467_;
goto v_resetjp_453_;
}
else
{
lean_inc(v_after_452_);
lean_dec(v_head_451_);
v___x_454_ = lean_box(0);
v_isShared_455_ = v_isSharedCheck_467_;
goto v_resetjp_453_;
}
v_resetjp_453_:
{
lean_object* v___x_456_; lean_object* v___x_458_; 
v___x_456_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__6___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__6___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__6___closed__0);
if (v_isShared_455_ == 0)
{
lean_ctor_set_tag(v___x_454_, 7);
lean_ctor_set(v___x_454_, 1, v___x_456_);
lean_ctor_set(v___x_454_, 0, v_msgData_442_);
v___x_458_ = v___x_454_;
goto v_reusejp_457_;
}
else
{
lean_object* v_reuseFailAlloc_466_; 
v_reuseFailAlloc_466_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_466_, 0, v_msgData_442_);
lean_ctor_set(v_reuseFailAlloc_466_, 1, v___x_456_);
v___x_458_ = v_reuseFailAlloc_466_;
goto v_reusejp_457_;
}
v_reusejp_457_:
{
lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v_msgData_463_; lean_object* v___x_464_; lean_object* v___x_465_; 
v___x_459_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4___redArg___closed__2);
v___x_460_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_460_, 0, v___x_458_);
lean_ctor_set(v___x_460_, 1, v___x_459_);
v___x_461_ = l_Lean_MessageData_ofSyntax(v_after_452_);
v___x_462_ = l_Lean_indentD(v___x_461_);
v_msgData_463_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_463_, 0, v___x_460_);
lean_ctor_set(v_msgData_463_, 1, v___x_462_);
v___x_464_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__6(v_msgData_463_, v_macroStack_443_);
v___x_465_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_465_, 0, v___x_464_);
return v___x_465_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4___redArg___boxed(lean_object* v_msgData_469_, lean_object* v_macroStack_470_, lean_object* v___y_471_, lean_object* v___y_472_){
_start:
{
lean_object* v_res_473_; 
v_res_473_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4___redArg(v_msgData_469_, v_macroStack_470_, v___y_471_);
lean_dec_ref(v___y_471_);
return v_res_473_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__3(lean_object* v_msgData_474_, lean_object* v___y_475_, lean_object* v___y_476_, lean_object* v___y_477_, lean_object* v___y_478_){
_start:
{
lean_object* v___x_480_; lean_object* v_env_481_; lean_object* v___x_482_; lean_object* v_mctx_483_; lean_object* v_lctx_484_; lean_object* v_options_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; 
v___x_480_ = lean_st_ref_get(v___y_478_);
v_env_481_ = lean_ctor_get(v___x_480_, 0);
lean_inc_ref(v_env_481_);
lean_dec(v___x_480_);
v___x_482_ = lean_st_ref_get(v___y_476_);
v_mctx_483_ = lean_ctor_get(v___x_482_, 0);
lean_inc_ref(v_mctx_483_);
lean_dec(v___x_482_);
v_lctx_484_ = lean_ctor_get(v___y_475_, 2);
v_options_485_ = lean_ctor_get(v___y_477_, 2);
lean_inc_ref(v_options_485_);
lean_inc_ref(v_lctx_484_);
v___x_486_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_486_, 0, v_env_481_);
lean_ctor_set(v___x_486_, 1, v_mctx_483_);
lean_ctor_set(v___x_486_, 2, v_lctx_484_);
lean_ctor_set(v___x_486_, 3, v_options_485_);
v___x_487_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_487_, 0, v___x_486_);
lean_ctor_set(v___x_487_, 1, v_msgData_474_);
v___x_488_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_488_, 0, v___x_487_);
return v___x_488_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__3___boxed(lean_object* v_msgData_489_, lean_object* v___y_490_, lean_object* v___y_491_, lean_object* v___y_492_, lean_object* v___y_493_, lean_object* v___y_494_){
_start:
{
lean_object* v_res_495_; 
v_res_495_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__3(v_msgData_489_, v___y_490_, v___y_491_, v___y_492_, v___y_493_);
lean_dec(v___y_493_);
lean_dec_ref(v___y_492_);
lean_dec(v___y_491_);
lean_dec_ref(v___y_490_);
return v_res_495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2___redArg(lean_object* v_msg_496_, lean_object* v___y_497_, lean_object* v___y_498_, lean_object* v___y_499_, lean_object* v___y_500_, lean_object* v___y_501_, lean_object* v___y_502_){
_start:
{
lean_object* v_ref_504_; lean_object* v___x_505_; lean_object* v_a_506_; lean_object* v_macroStack_507_; lean_object* v___x_508_; lean_object* v___x_509_; lean_object* v_a_510_; lean_object* v___x_512_; uint8_t v_isShared_513_; uint8_t v_isSharedCheck_518_; 
v_ref_504_ = lean_ctor_get(v___y_501_, 5);
v___x_505_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__3(v_msg_496_, v___y_499_, v___y_500_, v___y_501_, v___y_502_);
v_a_506_ = lean_ctor_get(v___x_505_, 0);
lean_inc(v_a_506_);
lean_dec_ref(v___x_505_);
v_macroStack_507_ = lean_ctor_get(v___y_497_, 1);
v___x_508_ = l_Lean_Elab_getBetterRef(v_ref_504_, v_macroStack_507_);
lean_inc(v_macroStack_507_);
v___x_509_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4___redArg(v_a_506_, v_macroStack_507_, v___y_501_);
v_a_510_ = lean_ctor_get(v___x_509_, 0);
v_isSharedCheck_518_ = !lean_is_exclusive(v___x_509_);
if (v_isSharedCheck_518_ == 0)
{
v___x_512_ = v___x_509_;
v_isShared_513_ = v_isSharedCheck_518_;
goto v_resetjp_511_;
}
else
{
lean_inc(v_a_510_);
lean_dec(v___x_509_);
v___x_512_ = lean_box(0);
v_isShared_513_ = v_isSharedCheck_518_;
goto v_resetjp_511_;
}
v_resetjp_511_:
{
lean_object* v___x_514_; lean_object* v___x_516_; 
v___x_514_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_514_, 0, v___x_508_);
lean_ctor_set(v___x_514_, 1, v_a_510_);
if (v_isShared_513_ == 0)
{
lean_ctor_set_tag(v___x_512_, 1);
lean_ctor_set(v___x_512_, 0, v___x_514_);
v___x_516_ = v___x_512_;
goto v_reusejp_515_;
}
else
{
lean_object* v_reuseFailAlloc_517_; 
v_reuseFailAlloc_517_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_517_, 0, v___x_514_);
v___x_516_ = v_reuseFailAlloc_517_;
goto v_reusejp_515_;
}
v_reusejp_515_:
{
return v___x_516_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2___redArg___boxed(lean_object* v_msg_519_, lean_object* v___y_520_, lean_object* v___y_521_, lean_object* v___y_522_, lean_object* v___y_523_, lean_object* v___y_524_, lean_object* v___y_525_, lean_object* v___y_526_){
_start:
{
lean_object* v_res_527_; 
v_res_527_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2___redArg(v_msg_519_, v___y_520_, v___y_521_, v___y_522_, v___y_523_, v___y_524_, v___y_525_);
lean_dec(v___y_525_);
lean_dec_ref(v___y_524_);
lean_dec(v___y_523_);
lean_dec_ref(v___y_522_);
lean_dec(v___y_521_);
lean_dec_ref(v___y_520_);
return v_res_527_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2___redArg(lean_object* v_ref_528_, lean_object* v_msg_529_, lean_object* v___y_530_, lean_object* v___y_531_, lean_object* v___y_532_, lean_object* v___y_533_, lean_object* v___y_534_, lean_object* v___y_535_){
_start:
{
lean_object* v_fileName_537_; lean_object* v_fileMap_538_; lean_object* v_options_539_; lean_object* v_currRecDepth_540_; lean_object* v_maxRecDepth_541_; lean_object* v_ref_542_; lean_object* v_currNamespace_543_; lean_object* v_openDecls_544_; lean_object* v_initHeartbeats_545_; lean_object* v_maxHeartbeats_546_; lean_object* v_quotContext_547_; lean_object* v_currMacroScope_548_; uint8_t v_diag_549_; lean_object* v_cancelTk_x3f_550_; uint8_t v_suppressElabErrors_551_; lean_object* v_inheritedTraceOptions_552_; lean_object* v_ref_553_; lean_object* v___x_554_; lean_object* v___x_555_; 
v_fileName_537_ = lean_ctor_get(v___y_534_, 0);
v_fileMap_538_ = lean_ctor_get(v___y_534_, 1);
v_options_539_ = lean_ctor_get(v___y_534_, 2);
v_currRecDepth_540_ = lean_ctor_get(v___y_534_, 3);
v_maxRecDepth_541_ = lean_ctor_get(v___y_534_, 4);
v_ref_542_ = lean_ctor_get(v___y_534_, 5);
v_currNamespace_543_ = lean_ctor_get(v___y_534_, 6);
v_openDecls_544_ = lean_ctor_get(v___y_534_, 7);
v_initHeartbeats_545_ = lean_ctor_get(v___y_534_, 8);
v_maxHeartbeats_546_ = lean_ctor_get(v___y_534_, 9);
v_quotContext_547_ = lean_ctor_get(v___y_534_, 10);
v_currMacroScope_548_ = lean_ctor_get(v___y_534_, 11);
v_diag_549_ = lean_ctor_get_uint8(v___y_534_, sizeof(void*)*14);
v_cancelTk_x3f_550_ = lean_ctor_get(v___y_534_, 12);
v_suppressElabErrors_551_ = lean_ctor_get_uint8(v___y_534_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_552_ = lean_ctor_get(v___y_534_, 13);
v_ref_553_ = l_Lean_replaceRef(v_ref_528_, v_ref_542_);
lean_inc_ref(v_inheritedTraceOptions_552_);
lean_inc(v_cancelTk_x3f_550_);
lean_inc(v_currMacroScope_548_);
lean_inc(v_quotContext_547_);
lean_inc(v_maxHeartbeats_546_);
lean_inc(v_initHeartbeats_545_);
lean_inc(v_openDecls_544_);
lean_inc(v_currNamespace_543_);
lean_inc(v_maxRecDepth_541_);
lean_inc(v_currRecDepth_540_);
lean_inc_ref(v_options_539_);
lean_inc_ref(v_fileMap_538_);
lean_inc_ref(v_fileName_537_);
v___x_554_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_554_, 0, v_fileName_537_);
lean_ctor_set(v___x_554_, 1, v_fileMap_538_);
lean_ctor_set(v___x_554_, 2, v_options_539_);
lean_ctor_set(v___x_554_, 3, v_currRecDepth_540_);
lean_ctor_set(v___x_554_, 4, v_maxRecDepth_541_);
lean_ctor_set(v___x_554_, 5, v_ref_553_);
lean_ctor_set(v___x_554_, 6, v_currNamespace_543_);
lean_ctor_set(v___x_554_, 7, v_openDecls_544_);
lean_ctor_set(v___x_554_, 8, v_initHeartbeats_545_);
lean_ctor_set(v___x_554_, 9, v_maxHeartbeats_546_);
lean_ctor_set(v___x_554_, 10, v_quotContext_547_);
lean_ctor_set(v___x_554_, 11, v_currMacroScope_548_);
lean_ctor_set(v___x_554_, 12, v_cancelTk_x3f_550_);
lean_ctor_set(v___x_554_, 13, v_inheritedTraceOptions_552_);
lean_ctor_set_uint8(v___x_554_, sizeof(void*)*14, v_diag_549_);
lean_ctor_set_uint8(v___x_554_, sizeof(void*)*14 + 1, v_suppressElabErrors_551_);
v___x_555_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2___redArg(v_msg_529_, v___y_530_, v___y_531_, v___y_532_, v___y_533_, v___x_554_, v___y_535_);
lean_dec_ref_known(v___x_554_, 14);
return v___x_555_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2___redArg___boxed(lean_object* v_ref_556_, lean_object* v_msg_557_, lean_object* v___y_558_, lean_object* v___y_559_, lean_object* v___y_560_, lean_object* v___y_561_, lean_object* v___y_562_, lean_object* v___y_563_, lean_object* v___y_564_){
_start:
{
lean_object* v_res_565_; 
v_res_565_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2___redArg(v_ref_556_, v_msg_557_, v___y_558_, v___y_559_, v___y_560_, v___y_561_, v___y_562_, v___y_563_);
lean_dec(v___y_563_);
lean_dec_ref(v___y_562_);
lean_dec(v___y_561_);
lean_dec_ref(v___y_560_);
lean_dec(v___y_559_);
lean_dec_ref(v___y_558_);
lean_dec(v_ref_556_);
return v_res_565_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__1___redArg(uint8_t v___x_569_, lean_object* v_as_x27_570_, lean_object* v_b_571_, lean_object* v___y_572_, lean_object* v___y_573_, lean_object* v___y_574_, lean_object* v___y_575_, lean_object* v___y_576_, lean_object* v___y_577_){
_start:
{
if (lean_obj_tag(v_as_x27_570_) == 0)
{
lean_object* v___x_579_; 
v___x_579_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_579_, 0, v_b_571_);
return v___x_579_;
}
else
{
lean_object* v_head_580_; lean_object* v_tail_581_; lean_object* v___x_582_; 
lean_dec_ref(v_b_571_);
v_head_580_ = lean_ctor_get(v_as_x27_570_, 0);
v_tail_581_ = lean_ctor_get(v_as_x27_570_, 1);
v___x_582_ = l_Lean_Elab_Term_getSyntheticMVarDecl_x3f___redArg(v_head_580_, v___y_573_);
if (lean_obj_tag(v___x_582_) == 0)
{
lean_object* v_a_583_; lean_object* v___x_584_; lean_object* v___x_585_; 
v_a_583_ = lean_ctor_get(v___x_582_, 0);
lean_inc(v_a_583_);
lean_dec_ref_known(v___x_582_, 1);
v___x_584_ = lean_box(0);
v___x_585_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__1___redArg___closed__0));
if (lean_obj_tag(v_a_583_) == 1)
{
lean_object* v_val_586_; lean_object* v___x_588_; uint8_t v_isShared_589_; uint8_t v_isSharedCheck_633_; 
v_val_586_ = lean_ctor_get(v_a_583_, 0);
v_isSharedCheck_633_ = !lean_is_exclusive(v_a_583_);
if (v_isSharedCheck_633_ == 0)
{
v___x_588_ = v_a_583_;
v_isShared_589_ = v_isSharedCheck_633_;
goto v_resetjp_587_;
}
else
{
lean_inc(v_val_586_);
lean_dec(v_a_583_);
v___x_588_ = lean_box(0);
v_isShared_589_ = v_isSharedCheck_633_;
goto v_resetjp_587_;
}
v_resetjp_587_:
{
lean_object* v_kind_590_; lean_object* v___x_592_; uint8_t v_isShared_593_; uint8_t v_isSharedCheck_631_; 
v_kind_590_ = lean_ctor_get(v_val_586_, 1);
v_isSharedCheck_631_ = !lean_is_exclusive(v_val_586_);
if (v_isSharedCheck_631_ == 0)
{
lean_object* v_unused_632_; 
v_unused_632_ = lean_ctor_get(v_val_586_, 0);
lean_dec(v_unused_632_);
v___x_592_ = v_val_586_;
v_isShared_593_ = v_isSharedCheck_631_;
goto v_resetjp_591_;
}
else
{
lean_inc(v_kind_590_);
lean_dec(v_val_586_);
v___x_592_ = lean_box(0);
v_isShared_593_ = v_isSharedCheck_631_;
goto v_resetjp_591_;
}
v_resetjp_591_:
{
if (lean_obj_tag(v_kind_590_) == 0)
{
lean_object* v___x_595_; uint8_t v_isShared_596_; uint8_t v_isSharedCheck_628_; 
v_isSharedCheck_628_ = !lean_is_exclusive(v_kind_590_);
if (v_isSharedCheck_628_ == 0)
{
lean_object* v_unused_629_; 
v_unused_629_ = lean_ctor_get(v_kind_590_, 0);
lean_dec(v_unused_629_);
v___x_595_ = v_kind_590_;
v_isShared_596_ = v_isSharedCheck_628_;
goto v_resetjp_594_;
}
else
{
lean_dec(v_kind_590_);
v___x_595_ = lean_box(0);
v_isShared_596_ = v_isSharedCheck_628_;
goto v_resetjp_594_;
}
v_resetjp_594_:
{
lean_object* v___x_597_; 
lean_inc(v_head_580_);
v___x_597_ = l_Lean_MVarId_getType(v_head_580_, v___y_574_, v___y_575_, v___y_576_, v___y_577_);
if (lean_obj_tag(v___x_597_) == 0)
{
lean_object* v_a_598_; lean_object* v___x_599_; lean_object* v_a_600_; lean_object* v___x_602_; uint8_t v_isShared_603_; uint8_t v_isSharedCheck_619_; 
v_a_598_ = lean_ctor_get(v___x_597_, 0);
lean_inc(v_a_598_);
lean_dec_ref_known(v___x_597_, 1);
v___x_599_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__0___redArg(v_a_598_, v___y_575_);
v_a_600_ = lean_ctor_get(v___x_599_, 0);
v_isSharedCheck_619_ = !lean_is_exclusive(v___x_599_);
if (v_isSharedCheck_619_ == 0)
{
v___x_602_ = v___x_599_;
v_isShared_603_ = v_isSharedCheck_619_;
goto v_resetjp_601_;
}
else
{
lean_inc(v_a_600_);
lean_dec(v___x_599_);
v___x_602_ = lean_box(0);
v_isShared_603_ = v_isSharedCheck_619_;
goto v_resetjp_601_;
}
v_resetjp_601_:
{
uint8_t v___x_617_; 
v___x_617_ = l_Lean_Expr_hasExprMVar(v_a_600_);
lean_dec(v_a_600_);
if (v___x_617_ == 0)
{
goto v___jp_604_;
}
else
{
if (v___x_569_ == 0)
{
lean_del_object(v___x_602_);
lean_del_object(v___x_595_);
lean_del_object(v___x_592_);
lean_del_object(v___x_588_);
v_as_x27_570_ = v_tail_581_;
v_b_571_ = v___x_585_;
goto _start;
}
else
{
goto v___jp_604_;
}
}
v___jp_604_:
{
lean_object* v___x_606_; 
lean_inc(v_head_580_);
if (v_isShared_589_ == 0)
{
lean_ctor_set(v___x_588_, 0, v_head_580_);
v___x_606_ = v___x_588_;
goto v_reusejp_605_;
}
else
{
lean_object* v_reuseFailAlloc_616_; 
v_reuseFailAlloc_616_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_616_, 0, v_head_580_);
v___x_606_ = v_reuseFailAlloc_616_;
goto v_reusejp_605_;
}
v_reusejp_605_:
{
lean_object* v___x_608_; 
if (v_isShared_596_ == 0)
{
lean_ctor_set_tag(v___x_595_, 1);
lean_ctor_set(v___x_595_, 0, v___x_606_);
v___x_608_ = v___x_595_;
goto v_reusejp_607_;
}
else
{
lean_object* v_reuseFailAlloc_615_; 
v_reuseFailAlloc_615_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_615_, 0, v___x_606_);
v___x_608_ = v_reuseFailAlloc_615_;
goto v_reusejp_607_;
}
v_reusejp_607_:
{
lean_object* v___x_610_; 
if (v_isShared_593_ == 0)
{
lean_ctor_set(v___x_592_, 1, v___x_584_);
lean_ctor_set(v___x_592_, 0, v___x_608_);
v___x_610_ = v___x_592_;
goto v_reusejp_609_;
}
else
{
lean_object* v_reuseFailAlloc_614_; 
v_reuseFailAlloc_614_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_614_, 0, v___x_608_);
lean_ctor_set(v_reuseFailAlloc_614_, 1, v___x_584_);
v___x_610_ = v_reuseFailAlloc_614_;
goto v_reusejp_609_;
}
v_reusejp_609_:
{
lean_object* v___x_612_; 
if (v_isShared_603_ == 0)
{
lean_ctor_set(v___x_602_, 0, v___x_610_);
v___x_612_ = v___x_602_;
goto v_reusejp_611_;
}
else
{
lean_object* v_reuseFailAlloc_613_; 
v_reuseFailAlloc_613_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_613_, 0, v___x_610_);
v___x_612_ = v_reuseFailAlloc_613_;
goto v_reusejp_611_;
}
v_reusejp_611_:
{
return v___x_612_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_620_; lean_object* v___x_622_; uint8_t v_isShared_623_; uint8_t v_isSharedCheck_627_; 
lean_del_object(v___x_595_);
lean_del_object(v___x_592_);
lean_del_object(v___x_588_);
v_a_620_ = lean_ctor_get(v___x_597_, 0);
v_isSharedCheck_627_ = !lean_is_exclusive(v___x_597_);
if (v_isSharedCheck_627_ == 0)
{
v___x_622_ = v___x_597_;
v_isShared_623_ = v_isSharedCheck_627_;
goto v_resetjp_621_;
}
else
{
lean_inc(v_a_620_);
lean_dec(v___x_597_);
v___x_622_ = lean_box(0);
v_isShared_623_ = v_isSharedCheck_627_;
goto v_resetjp_621_;
}
v_resetjp_621_:
{
lean_object* v___x_625_; 
if (v_isShared_623_ == 0)
{
v___x_625_ = v___x_622_;
goto v_reusejp_624_;
}
else
{
lean_object* v_reuseFailAlloc_626_; 
v_reuseFailAlloc_626_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_626_, 0, v_a_620_);
v___x_625_ = v_reuseFailAlloc_626_;
goto v_reusejp_624_;
}
v_reusejp_624_:
{
return v___x_625_;
}
}
}
}
}
else
{
lean_del_object(v___x_592_);
lean_dec_ref(v_kind_590_);
lean_del_object(v___x_588_);
v_as_x27_570_ = v_tail_581_;
v_b_571_ = v___x_585_;
goto _start;
}
}
}
}
else
{
lean_dec(v_a_583_);
v_as_x27_570_ = v_tail_581_;
v_b_571_ = v___x_585_;
goto _start;
}
}
else
{
lean_object* v_a_635_; lean_object* v___x_637_; uint8_t v_isShared_638_; uint8_t v_isSharedCheck_642_; 
v_a_635_ = lean_ctor_get(v___x_582_, 0);
v_isSharedCheck_642_ = !lean_is_exclusive(v___x_582_);
if (v_isSharedCheck_642_ == 0)
{
v___x_637_ = v___x_582_;
v_isShared_638_ = v_isSharedCheck_642_;
goto v_resetjp_636_;
}
else
{
lean_inc(v_a_635_);
lean_dec(v___x_582_);
v___x_637_ = lean_box(0);
v_isShared_638_ = v_isSharedCheck_642_;
goto v_resetjp_636_;
}
v_resetjp_636_:
{
lean_object* v___x_640_; 
if (v_isShared_638_ == 0)
{
v___x_640_ = v___x_637_;
goto v_reusejp_639_;
}
else
{
lean_object* v_reuseFailAlloc_641_; 
v_reuseFailAlloc_641_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_641_, 0, v_a_635_);
v___x_640_ = v_reuseFailAlloc_641_;
goto v_reusejp_639_;
}
v_reusejp_639_:
{
return v___x_640_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__1___redArg___boxed(lean_object* v___x_643_, lean_object* v_as_x27_644_, lean_object* v_b_645_, lean_object* v___y_646_, lean_object* v___y_647_, lean_object* v___y_648_, lean_object* v___y_649_, lean_object* v___y_650_, lean_object* v___y_651_, lean_object* v___y_652_){
_start:
{
uint8_t v___x_6494__boxed_653_; lean_object* v_res_654_; 
v___x_6494__boxed_653_ = lean_unbox(v___x_643_);
v_res_654_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__1___redArg(v___x_6494__boxed_653_, v_as_x27_644_, v_b_645_, v___y_646_, v___y_647_, v___y_648_, v___y_649_, v___y_650_, v___y_651_);
lean_dec(v___y_651_);
lean_dec_ref(v___y_650_);
lean_dec(v___y_649_);
lean_dec_ref(v___y_648_);
lean_dec(v___y_647_);
lean_dec_ref(v___y_646_);
lean_dec(v_as_x27_644_);
return v_res_654_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Command_Variable_pendingActionableSynthMVar___closed__1(void){
_start:
{
lean_object* v___x_656_; lean_object* v___x_657_; 
v___x_656_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_pendingActionableSynthMVar___closed__0));
v___x_657_ = l_Lean_stringToMessageData(v___x_656_);
return v___x_657_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Command_Variable_pendingActionableSynthMVar___closed__3(void){
_start:
{
lean_object* v___x_659_; lean_object* v___x_660_; 
v___x_659_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_pendingActionableSynthMVar___closed__2));
v___x_660_ = l_Lean_stringToMessageData(v___x_659_);
return v___x_660_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_pendingActionableSynthMVar(lean_object* v_binder_661_, lean_object* v_a_662_, lean_object* v_a_663_, lean_object* v_a_664_, lean_object* v_a_665_, lean_object* v_a_666_, lean_object* v_a_667_){
_start:
{
lean_object* v___x_669_; lean_object* v_pendingMVars_670_; uint8_t v___x_671_; 
v___x_669_ = lean_st_ref_get(v_a_663_);
v_pendingMVars_670_ = lean_ctor_get(v___x_669_, 2);
lean_inc(v_pendingMVars_670_);
lean_dec(v___x_669_);
v___x_671_ = l_List_isEmpty___redArg(v_pendingMVars_670_);
if (v___x_671_ == 0)
{
lean_object* v___x_672_; lean_object* v___x_673_; lean_object* v___x_674_; 
v___x_672_ = l_List_reverse___redArg(v_pendingMVars_670_);
v___x_673_ = ((lean_object*)(lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__1___redArg___closed__0));
v___x_674_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__1___redArg(v___x_671_, v___x_672_, v___x_673_, v_a_662_, v_a_663_, v_a_664_, v_a_665_, v_a_666_, v_a_667_);
lean_dec(v___x_672_);
if (lean_obj_tag(v___x_674_) == 0)
{
lean_object* v_a_675_; lean_object* v___x_677_; uint8_t v_isShared_678_; uint8_t v_isSharedCheck_697_; 
v_a_675_ = lean_ctor_get(v___x_674_, 0);
v_isSharedCheck_697_ = !lean_is_exclusive(v___x_674_);
if (v_isSharedCheck_697_ == 0)
{
v___x_677_ = v___x_674_;
v_isShared_678_ = v_isSharedCheck_697_;
goto v_resetjp_676_;
}
else
{
lean_inc(v_a_675_);
lean_dec(v___x_674_);
v___x_677_ = lean_box(0);
v_isShared_678_ = v_isSharedCheck_697_;
goto v_resetjp_676_;
}
v_resetjp_676_:
{
lean_object* v_fst_679_; lean_object* v___x_681_; uint8_t v_isShared_682_; uint8_t v_isSharedCheck_695_; 
v_fst_679_ = lean_ctor_get(v_a_675_, 0);
v_isSharedCheck_695_ = !lean_is_exclusive(v_a_675_);
if (v_isSharedCheck_695_ == 0)
{
lean_object* v_unused_696_; 
v_unused_696_ = lean_ctor_get(v_a_675_, 1);
lean_dec(v_unused_696_);
v___x_681_ = v_a_675_;
v_isShared_682_ = v_isSharedCheck_695_;
goto v_resetjp_680_;
}
else
{
lean_inc(v_fst_679_);
lean_dec(v_a_675_);
v___x_681_ = lean_box(0);
v_isShared_682_ = v_isSharedCheck_695_;
goto v_resetjp_680_;
}
v_resetjp_680_:
{
if (lean_obj_tag(v_fst_679_) == 0)
{
lean_object* v___x_683_; lean_object* v___x_684_; lean_object* v___x_686_; 
lean_del_object(v___x_677_);
v___x_683_ = lean_obj_once(&lp_mathlib_Mathlib_Command_Variable_pendingActionableSynthMVar___closed__1, &lp_mathlib_Mathlib_Command_Variable_pendingActionableSynthMVar___closed__1_once, _init_lp_mathlib_Mathlib_Command_Variable_pendingActionableSynthMVar___closed__1);
lean_inc(v_binder_661_);
v___x_684_ = l_Lean_MessageData_ofSyntax(v_binder_661_);
if (v_isShared_682_ == 0)
{
lean_ctor_set_tag(v___x_681_, 7);
lean_ctor_set(v___x_681_, 1, v___x_684_);
lean_ctor_set(v___x_681_, 0, v___x_683_);
v___x_686_ = v___x_681_;
goto v_reusejp_685_;
}
else
{
lean_object* v_reuseFailAlloc_690_; 
v_reuseFailAlloc_690_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_690_, 0, v___x_683_);
lean_ctor_set(v_reuseFailAlloc_690_, 1, v___x_684_);
v___x_686_ = v_reuseFailAlloc_690_;
goto v_reusejp_685_;
}
v_reusejp_685_:
{
lean_object* v___x_687_; lean_object* v___x_688_; lean_object* v___x_689_; 
v___x_687_ = lean_obj_once(&lp_mathlib_Mathlib_Command_Variable_pendingActionableSynthMVar___closed__3, &lp_mathlib_Mathlib_Command_Variable_pendingActionableSynthMVar___closed__3_once, _init_lp_mathlib_Mathlib_Command_Variable_pendingActionableSynthMVar___closed__3);
v___x_688_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_688_, 0, v___x_686_);
lean_ctor_set(v___x_688_, 1, v___x_687_);
v___x_689_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2___redArg(v_binder_661_, v___x_688_, v_a_662_, v_a_663_, v_a_664_, v_a_665_, v_a_666_, v_a_667_);
lean_dec(v_binder_661_);
return v___x_689_;
}
}
else
{
lean_object* v_val_691_; lean_object* v___x_693_; 
lean_del_object(v___x_681_);
lean_dec(v_binder_661_);
v_val_691_ = lean_ctor_get(v_fst_679_, 0);
lean_inc(v_val_691_);
lean_dec_ref_known(v_fst_679_, 1);
if (v_isShared_678_ == 0)
{
lean_ctor_set(v___x_677_, 0, v_val_691_);
v___x_693_ = v___x_677_;
goto v_reusejp_692_;
}
else
{
lean_object* v_reuseFailAlloc_694_; 
v_reuseFailAlloc_694_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_694_, 0, v_val_691_);
v___x_693_ = v_reuseFailAlloc_694_;
goto v_reusejp_692_;
}
v_reusejp_692_:
{
return v___x_693_;
}
}
}
}
}
else
{
lean_object* v_a_698_; lean_object* v___x_700_; uint8_t v_isShared_701_; uint8_t v_isSharedCheck_705_; 
lean_dec(v_binder_661_);
v_a_698_ = lean_ctor_get(v___x_674_, 0);
v_isSharedCheck_705_ = !lean_is_exclusive(v___x_674_);
if (v_isSharedCheck_705_ == 0)
{
v___x_700_ = v___x_674_;
v_isShared_701_ = v_isSharedCheck_705_;
goto v_resetjp_699_;
}
else
{
lean_inc(v_a_698_);
lean_dec(v___x_674_);
v___x_700_ = lean_box(0);
v_isShared_701_ = v_isSharedCheck_705_;
goto v_resetjp_699_;
}
v_resetjp_699_:
{
lean_object* v___x_703_; 
if (v_isShared_701_ == 0)
{
v___x_703_ = v___x_700_;
goto v_reusejp_702_;
}
else
{
lean_object* v_reuseFailAlloc_704_; 
v_reuseFailAlloc_704_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_704_, 0, v_a_698_);
v___x_703_ = v_reuseFailAlloc_704_;
goto v_reusejp_702_;
}
v_reusejp_702_:
{
return v___x_703_;
}
}
}
}
else
{
lean_object* v___x_706_; lean_object* v___x_707_; 
lean_dec(v_pendingMVars_670_);
lean_dec(v_binder_661_);
v___x_706_ = lean_box(0);
v___x_707_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_707_, 0, v___x_706_);
return v___x_707_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_pendingActionableSynthMVar___boxed(lean_object* v_binder_708_, lean_object* v_a_709_, lean_object* v_a_710_, lean_object* v_a_711_, lean_object* v_a_712_, lean_object* v_a_713_, lean_object* v_a_714_, lean_object* v_a_715_){
_start:
{
lean_object* v_res_716_; 
v_res_716_ = lp_mathlib_Mathlib_Command_Variable_pendingActionableSynthMVar(v_binder_708_, v_a_709_, v_a_710_, v_a_711_, v_a_712_, v_a_713_, v_a_714_);
lean_dec(v_a_714_);
lean_dec_ref(v_a_713_);
lean_dec(v_a_712_);
lean_dec_ref(v_a_711_);
lean_dec(v_a_710_);
lean_dec_ref(v_a_709_);
return v_res_716_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__1(uint8_t v___x_717_, lean_object* v_as_718_, lean_object* v_as_x27_719_, lean_object* v_b_720_, lean_object* v_a_721_, lean_object* v___y_722_, lean_object* v___y_723_, lean_object* v___y_724_, lean_object* v___y_725_, lean_object* v___y_726_, lean_object* v___y_727_){
_start:
{
lean_object* v___x_729_; 
v___x_729_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__1___redArg(v___x_717_, v_as_x27_719_, v_b_720_, v___y_722_, v___y_723_, v___y_724_, v___y_725_, v___y_726_, v___y_727_);
return v___x_729_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__1___boxed(lean_object* v___x_730_, lean_object* v_as_731_, lean_object* v_as_x27_732_, lean_object* v_b_733_, lean_object* v_a_734_, lean_object* v___y_735_, lean_object* v___y_736_, lean_object* v___y_737_, lean_object* v___y_738_, lean_object* v___y_739_, lean_object* v___y_740_, lean_object* v___y_741_){
_start:
{
uint8_t v___x_6747__boxed_742_; lean_object* v_res_743_; 
v___x_6747__boxed_742_ = lean_unbox(v___x_730_);
v_res_743_ = lp_mathlib_List_forIn_x27_loop___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__1(v___x_6747__boxed_742_, v_as_731_, v_as_x27_732_, v_b_733_, v_a_734_, v___y_735_, v___y_736_, v___y_737_, v___y_738_, v___y_739_, v___y_740_);
lean_dec(v___y_740_);
lean_dec_ref(v___y_739_);
lean_dec(v___y_738_);
lean_dec_ref(v___y_737_);
lean_dec(v___y_736_);
lean_dec_ref(v___y_735_);
lean_dec(v_as_x27_732_);
lean_dec(v_as_731_);
return v_res_743_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2(lean_object* v_00_u03b1_744_, lean_object* v_ref_745_, lean_object* v_msg_746_, lean_object* v___y_747_, lean_object* v___y_748_, lean_object* v___y_749_, lean_object* v___y_750_, lean_object* v___y_751_, lean_object* v___y_752_){
_start:
{
lean_object* v___x_754_; 
v___x_754_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2___redArg(v_ref_745_, v_msg_746_, v___y_747_, v___y_748_, v___y_749_, v___y_750_, v___y_751_, v___y_752_);
return v___x_754_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2___boxed(lean_object* v_00_u03b1_755_, lean_object* v_ref_756_, lean_object* v_msg_757_, lean_object* v___y_758_, lean_object* v___y_759_, lean_object* v___y_760_, lean_object* v___y_761_, lean_object* v___y_762_, lean_object* v___y_763_, lean_object* v___y_764_){
_start:
{
lean_object* v_res_765_; 
v_res_765_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2(v_00_u03b1_755_, v_ref_756_, v_msg_757_, v___y_758_, v___y_759_, v___y_760_, v___y_761_, v___y_762_, v___y_763_);
lean_dec(v___y_763_);
lean_dec_ref(v___y_762_);
lean_dec(v___y_761_);
lean_dec_ref(v___y_760_);
lean_dec(v___y_759_);
lean_dec_ref(v___y_758_);
lean_dec(v_ref_756_);
return v_res_765_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2(lean_object* v_00_u03b1_766_, lean_object* v_msg_767_, lean_object* v___y_768_, lean_object* v___y_769_, lean_object* v___y_770_, lean_object* v___y_771_, lean_object* v___y_772_, lean_object* v___y_773_){
_start:
{
lean_object* v___x_775_; 
v___x_775_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2___redArg(v_msg_767_, v___y_768_, v___y_769_, v___y_770_, v___y_771_, v___y_772_, v___y_773_);
return v___x_775_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2___boxed(lean_object* v_00_u03b1_776_, lean_object* v_msg_777_, lean_object* v___y_778_, lean_object* v___y_779_, lean_object* v___y_780_, lean_object* v___y_781_, lean_object* v___y_782_, lean_object* v___y_783_, lean_object* v___y_784_){
_start:
{
lean_object* v_res_785_; 
v_res_785_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2(v_00_u03b1_776_, v_msg_777_, v___y_778_, v___y_779_, v___y_780_, v___y_781_, v___y_782_, v___y_783_);
lean_dec(v___y_783_);
lean_dec_ref(v___y_782_);
lean_dec(v___y_781_);
lean_dec_ref(v___y_780_);
lean_dec(v___y_779_);
lean_dec_ref(v___y_778_);
return v_res_785_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4(lean_object* v_msgData_786_, lean_object* v_macroStack_787_, lean_object* v___y_788_, lean_object* v___y_789_, lean_object* v___y_790_, lean_object* v___y_791_, lean_object* v___y_792_, lean_object* v___y_793_){
_start:
{
lean_object* v___x_795_; 
v___x_795_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4___redArg(v_msgData_786_, v_macroStack_787_, v___y_792_);
return v___x_795_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4___boxed(lean_object* v_msgData_796_, lean_object* v_macroStack_797_, lean_object* v___y_798_, lean_object* v___y_799_, lean_object* v___y_800_, lean_object* v___y_801_, lean_object* v___y_802_, lean_object* v___y_803_, lean_object* v___y_804_){
_start:
{
lean_object* v_res_805_; 
v_res_805_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4(v_msgData_796_, v_macroStack_797_, v___y_798_, v___y_799_, v___y_800_, v___y_801_, v___y_802_, v___y_803_);
lean_dec(v___y_803_);
lean_dec_ref(v___y_802_);
lean_dec(v___y_801_);
lean_dec_ref(v___y_800_);
lean_dec(v___y_799_);
lean_dec_ref(v___y_798_);
return v_res_805_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Command_Variable_getSubproblem_spec__2___redArg___lam__0(lean_object* v_x_806_, lean_object* v___y_807_, lean_object* v___y_808_, lean_object* v___y_809_, lean_object* v___y_810_, lean_object* v___y_811_, lean_object* v___y_812_){
_start:
{
lean_object* v___x_814_; 
lean_inc(v___y_808_);
lean_inc_ref(v___y_807_);
v___x_814_ = lean_apply_7(v_x_806_, v___y_807_, v___y_808_, v___y_809_, v___y_810_, v___y_811_, v___y_812_, lean_box(0));
return v___x_814_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Command_Variable_getSubproblem_spec__2___redArg___lam__0___boxed(lean_object* v_x_815_, lean_object* v___y_816_, lean_object* v___y_817_, lean_object* v___y_818_, lean_object* v___y_819_, lean_object* v___y_820_, lean_object* v___y_821_, lean_object* v___y_822_){
_start:
{
lean_object* v_res_823_; 
v_res_823_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Command_Variable_getSubproblem_spec__2___redArg___lam__0(v_x_815_, v___y_816_, v___y_817_, v___y_818_, v___y_819_, v___y_820_, v___y_821_);
lean_dec(v___y_817_);
lean_dec_ref(v___y_816_);
return v_res_823_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Command_Variable_getSubproblem_spec__2___redArg(lean_object* v_mvarId_824_, lean_object* v_x_825_, lean_object* v___y_826_, lean_object* v___y_827_, lean_object* v___y_828_, lean_object* v___y_829_, lean_object* v___y_830_, lean_object* v___y_831_){
_start:
{
lean_object* v___f_833_; lean_object* v___x_834_; 
lean_inc(v___y_827_);
lean_inc_ref(v___y_826_);
v___f_833_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Command_Variable_getSubproblem_spec__2___redArg___lam__0___boxed), 8, 3);
lean_closure_set(v___f_833_, 0, v_x_825_);
lean_closure_set(v___f_833_, 1, v___y_826_);
lean_closure_set(v___f_833_, 2, v___y_827_);
v___x_834_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_824_, v___f_833_, v___y_828_, v___y_829_, v___y_830_, v___y_831_);
if (lean_obj_tag(v___x_834_) == 0)
{
return v___x_834_;
}
else
{
lean_object* v_a_835_; lean_object* v___x_837_; uint8_t v_isShared_838_; uint8_t v_isSharedCheck_842_; 
v_a_835_ = lean_ctor_get(v___x_834_, 0);
v_isSharedCheck_842_ = !lean_is_exclusive(v___x_834_);
if (v_isSharedCheck_842_ == 0)
{
v___x_837_ = v___x_834_;
v_isShared_838_ = v_isSharedCheck_842_;
goto v_resetjp_836_;
}
else
{
lean_inc(v_a_835_);
lean_dec(v___x_834_);
v___x_837_ = lean_box(0);
v_isShared_838_ = v_isSharedCheck_842_;
goto v_resetjp_836_;
}
v_resetjp_836_:
{
lean_object* v___x_840_; 
if (v_isShared_838_ == 0)
{
v___x_840_ = v___x_837_;
goto v_reusejp_839_;
}
else
{
lean_object* v_reuseFailAlloc_841_; 
v_reuseFailAlloc_841_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_841_, 0, v_a_835_);
v___x_840_ = v_reuseFailAlloc_841_;
goto v_reusejp_839_;
}
v_reusejp_839_:
{
return v___x_840_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Command_Variable_getSubproblem_spec__2___redArg___boxed(lean_object* v_mvarId_843_, lean_object* v_x_844_, lean_object* v___y_845_, lean_object* v___y_846_, lean_object* v___y_847_, lean_object* v___y_848_, lean_object* v___y_849_, lean_object* v___y_850_, lean_object* v___y_851_){
_start:
{
lean_object* v_res_852_; 
v_res_852_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Command_Variable_getSubproblem_spec__2___redArg(v_mvarId_843_, v_x_844_, v___y_845_, v___y_846_, v___y_847_, v___y_848_, v___y_849_, v___y_850_);
lean_dec(v___y_850_);
lean_dec_ref(v___y_849_);
lean_dec(v___y_848_);
lean_dec_ref(v___y_847_);
lean_dec(v___y_846_);
lean_dec_ref(v___y_845_);
return v_res_852_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Command_Variable_getSubproblem_spec__2(lean_object* v_00_u03b1_853_, lean_object* v_mvarId_854_, lean_object* v_x_855_, lean_object* v___y_856_, lean_object* v___y_857_, lean_object* v___y_858_, lean_object* v___y_859_, lean_object* v___y_860_, lean_object* v___y_861_){
_start:
{
lean_object* v___x_863_; 
v___x_863_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Command_Variable_getSubproblem_spec__2___redArg(v_mvarId_854_, v_x_855_, v___y_856_, v___y_857_, v___y_858_, v___y_859_, v___y_860_, v___y_861_);
return v___x_863_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Command_Variable_getSubproblem_spec__2___boxed(lean_object* v_00_u03b1_864_, lean_object* v_mvarId_865_, lean_object* v_x_866_, lean_object* v___y_867_, lean_object* v___y_868_, lean_object* v___y_869_, lean_object* v___y_870_, lean_object* v___y_871_, lean_object* v___y_872_, lean_object* v___y_873_){
_start:
{
lean_object* v_res_874_; 
v_res_874_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Command_Variable_getSubproblem_spec__2(v_00_u03b1_864_, v_mvarId_865_, v_x_866_, v___y_867_, v___y_868_, v___y_869_, v___y_870_, v___y_871_, v___y_872_);
lean_dec(v___y_872_);
lean_dec_ref(v___y_871_);
lean_dec(v___y_870_);
lean_dec_ref(v___y_869_);
lean_dec(v___y_868_);
lean_dec_ref(v___y_867_);
return v_res_874_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_Variable_getSubproblem_spec__1(size_t v_sz_875_, size_t v_i_876_, lean_object* v_bs_877_){
_start:
{
uint8_t v___x_878_; 
v___x_878_ = lean_usize_dec_lt(v_i_876_, v_sz_875_);
if (v___x_878_ == 0)
{
return v_bs_877_;
}
else
{
lean_object* v_v_879_; lean_object* v___x_880_; lean_object* v_bs_x27_881_; lean_object* v___x_882_; size_t v___x_883_; size_t v___x_884_; lean_object* v___x_885_; 
v_v_879_ = lean_array_uget(v_bs_877_, v_i_876_);
v___x_880_ = lean_unsigned_to_nat(0u);
v_bs_x27_881_ = lean_array_uset(v_bs_877_, v_i_876_, v___x_880_);
v___x_882_ = l_Lean_Expr_fvar___override(v_v_879_);
v___x_883_ = ((size_t)1ULL);
v___x_884_ = lean_usize_add(v_i_876_, v___x_883_);
v___x_885_ = lean_array_uset(v_bs_x27_881_, v_i_876_, v___x_882_);
v_i_876_ = v___x_884_;
v_bs_877_ = v___x_885_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_Variable_getSubproblem_spec__1___boxed(lean_object* v_sz_887_, lean_object* v_i_888_, lean_object* v_bs_889_){
_start:
{
size_t v_sz_boxed_890_; size_t v_i_boxed_891_; lean_object* v_res_892_; 
v_sz_boxed_890_ = lean_unbox_usize(v_sz_887_);
lean_dec(v_sz_887_);
v_i_boxed_891_ = lean_unbox_usize(v_i_888_);
lean_dec(v_i_888_);
v_res_892_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_Variable_getSubproblem_spec__1(v_sz_boxed_890_, v_i_boxed_891_, v_bs_889_);
return v_res_892_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__0(lean_object* v_val_893_, lean_object* v___y_894_, uint8_t v___x_895_, lean_object* v___y_896_, lean_object* v___y_897_, lean_object* v___y_898_, lean_object* v___y_899_, lean_object* v___y_900_, lean_object* v___y_901_){
_start:
{
lean_object* v___x_903_; 
v___x_903_ = l_Lean_MVarId_getType(v_val_893_, v___y_898_, v___y_899_, v___y_900_, v___y_901_);
if (lean_obj_tag(v___x_903_) == 0)
{
lean_object* v_a_904_; size_t v_sz_905_; size_t v___x_906_; lean_object* v___x_907_; uint8_t v___x_908_; lean_object* v___x_909_; 
v_a_904_ = lean_ctor_get(v___x_903_, 0);
lean_inc(v_a_904_);
lean_dec_ref_known(v___x_903_, 1);
v_sz_905_ = lean_array_size(v___y_894_);
v___x_906_ = ((size_t)0ULL);
v___x_907_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_Variable_getSubproblem_spec__1(v_sz_905_, v___x_906_, v___y_894_);
v___x_908_ = 1;
v___x_909_ = l_Lean_Meta_mkForallFVars(v___x_907_, v_a_904_, v___x_895_, v___x_895_, v___x_895_, v___x_908_, v___y_898_, v___y_899_, v___y_900_, v___y_901_);
lean_dec_ref(v___x_907_);
if (lean_obj_tag(v___x_909_) == 0)
{
lean_object* v_a_910_; lean_object* v___x_911_; 
v_a_910_ = lean_ctor_get(v___x_909_, 0);
lean_inc(v_a_910_);
lean_dec_ref_known(v___x_909_, 1);
v___x_911_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__0___redArg(v_a_910_, v___y_899_);
return v___x_911_;
}
else
{
return v___x_909_;
}
}
else
{
lean_dec_ref(v___y_894_);
return v___x_903_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__0___boxed(lean_object* v_val_912_, lean_object* v___y_913_, lean_object* v___x_914_, lean_object* v___y_915_, lean_object* v___y_916_, lean_object* v___y_917_, lean_object* v___y_918_, lean_object* v___y_919_, lean_object* v___y_920_, lean_object* v___y_921_){
_start:
{
uint8_t v___x_11099__boxed_922_; lean_object* v_res_923_; 
v___x_11099__boxed_922_ = lean_unbox(v___x_914_);
v_res_923_ = lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__0(v_val_912_, v___y_913_, v___x_11099__boxed_922_, v___y_915_, v___y_916_, v___y_917_, v___y_918_, v___y_919_, v___y_920_);
lean_dec(v___y_920_);
lean_dec_ref(v___y_919_);
lean_dec(v___y_918_);
lean_dec_ref(v___y_917_);
lean_dec(v___y_916_);
lean_dec_ref(v___y_915_);
return v_res_923_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Mathlib_Command_Variable_getSubproblem_spec__0_spec__0(lean_object* v_a_924_, lean_object* v_as_925_, size_t v_i_926_, size_t v_stop_927_){
_start:
{
uint8_t v___x_928_; 
v___x_928_ = lean_usize_dec_eq(v_i_926_, v_stop_927_);
if (v___x_928_ == 0)
{
lean_object* v___x_929_; uint8_t v___x_930_; 
v___x_929_ = lean_array_uget_borrowed(v_as_925_, v_i_926_);
v___x_930_ = l_Lean_instBEqFVarId_beq(v_a_924_, v___x_929_);
if (v___x_930_ == 0)
{
size_t v___x_931_; size_t v___x_932_; 
v___x_931_ = ((size_t)1ULL);
v___x_932_ = lean_usize_add(v_i_926_, v___x_931_);
v_i_926_ = v___x_932_;
goto _start;
}
else
{
return v___x_930_;
}
}
else
{
uint8_t v___x_934_; 
v___x_934_ = 0;
return v___x_934_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Mathlib_Command_Variable_getSubproblem_spec__0_spec__0___boxed(lean_object* v_a_935_, lean_object* v_as_936_, lean_object* v_i_937_, lean_object* v_stop_938_){
_start:
{
size_t v_i_boxed_939_; size_t v_stop_boxed_940_; uint8_t v_res_941_; lean_object* v_r_942_; 
v_i_boxed_939_ = lean_unbox_usize(v_i_937_);
lean_dec(v_i_937_);
v_stop_boxed_940_ = lean_unbox_usize(v_stop_938_);
lean_dec(v_stop_938_);
v_res_941_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Mathlib_Command_Variable_getSubproblem_spec__0_spec__0(v_a_935_, v_as_936_, v_i_boxed_939_, v_stop_boxed_940_);
lean_dec_ref(v_as_936_);
lean_dec(v_a_935_);
v_r_942_ = lean_box(v_res_941_);
return v_r_942_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_contains___at___00Mathlib_Command_Variable_getSubproblem_spec__0(lean_object* v_as_943_, lean_object* v_a_944_){
_start:
{
lean_object* v___x_945_; lean_object* v___x_946_; uint8_t v___x_947_; 
v___x_945_ = lean_unsigned_to_nat(0u);
v___x_946_ = lean_array_get_size(v_as_943_);
v___x_947_ = lean_nat_dec_lt(v___x_945_, v___x_946_);
if (v___x_947_ == 0)
{
return v___x_947_;
}
else
{
if (v___x_947_ == 0)
{
return v___x_947_;
}
else
{
size_t v___x_948_; size_t v___x_949_; uint8_t v___x_950_; 
v___x_948_ = ((size_t)0ULL);
v___x_949_ = lean_usize_of_nat(v___x_946_);
v___x_950_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Mathlib_Command_Variable_getSubproblem_spec__0_spec__0(v_a_944_, v_as_943_, v___x_948_, v___x_949_);
return v___x_950_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_contains___at___00Mathlib_Command_Variable_getSubproblem_spec__0___boxed(lean_object* v_as_951_, lean_object* v_a_952_){
_start:
{
uint8_t v_res_953_; lean_object* v_r_954_; 
v_res_953_ = lp_mathlib_Array_contains___at___00Mathlib_Command_Variable_getSubproblem_spec__0(v_as_951_, v_a_952_);
lean_dec(v_a_952_);
lean_dec_ref(v_as_951_);
v_r_954_ = lean_box(v_res_953_);
return v_r_954_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Command_Variable_getSubproblem_spec__3(lean_object* v___x_955_, lean_object* v_as_956_, size_t v_i_957_, size_t v_stop_958_, lean_object* v_b_959_){
_start:
{
lean_object* v___y_961_; uint8_t v___x_965_; 
v___x_965_ = lean_usize_dec_eq(v_i_957_, v_stop_958_);
if (v___x_965_ == 0)
{
lean_object* v___x_966_; uint8_t v___x_967_; 
v___x_966_ = lean_array_uget_borrowed(v_as_956_, v_i_957_);
v___x_967_ = lp_mathlib_Array_contains___at___00Mathlib_Command_Variable_getSubproblem_spec__0(v___x_955_, v___x_966_);
if (v___x_967_ == 0)
{
lean_object* v___x_968_; 
lean_inc(v___x_966_);
v___x_968_ = lean_array_push(v_b_959_, v___x_966_);
v___y_961_ = v___x_968_;
goto v___jp_960_;
}
else
{
v___y_961_ = v_b_959_;
goto v___jp_960_;
}
}
else
{
return v_b_959_;
}
v___jp_960_:
{
size_t v___x_962_; size_t v___x_963_; 
v___x_962_ = ((size_t)1ULL);
v___x_963_ = lean_usize_add(v_i_957_, v___x_962_);
v_i_957_ = v___x_963_;
v_b_959_ = v___y_961_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Command_Variable_getSubproblem_spec__3___boxed(lean_object* v___x_969_, lean_object* v_as_970_, lean_object* v_i_971_, lean_object* v_stop_972_, lean_object* v_b_973_){
_start:
{
size_t v_i_boxed_974_; size_t v_stop_boxed_975_; lean_object* v_res_976_; 
v_i_boxed_974_ = lean_unbox_usize(v_i_971_);
lean_dec(v_i_971_);
v_stop_boxed_975_ = lean_unbox_usize(v_stop_972_);
lean_dec(v_stop_972_);
v_res_976_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Command_Variable_getSubproblem_spec__3(v___x_969_, v_as_970_, v_i_boxed_974_, v_stop_boxed_975_, v_b_973_);
lean_dec_ref(v_as_970_);
lean_dec_ref(v___x_969_);
return v_res_976_;
}
}
static double _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg___closed__0(void){
_start:
{
lean_object* v___x_977_; double v___x_978_; 
v___x_977_ = lean_unsigned_to_nat(0u);
v___x_978_ = lean_float_of_nat(v___x_977_);
return v___x_978_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg(lean_object* v_cls_982_, lean_object* v_msg_983_, lean_object* v___y_984_, lean_object* v___y_985_, lean_object* v___y_986_, lean_object* v___y_987_){
_start:
{
lean_object* v_ref_989_; lean_object* v___x_990_; lean_object* v_a_991_; lean_object* v___x_993_; uint8_t v_isShared_994_; uint8_t v_isSharedCheck_1035_; 
v_ref_989_ = lean_ctor_get(v___y_986_, 5);
v___x_990_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__3(v_msg_983_, v___y_984_, v___y_985_, v___y_986_, v___y_987_);
v_a_991_ = lean_ctor_get(v___x_990_, 0);
v_isSharedCheck_1035_ = !lean_is_exclusive(v___x_990_);
if (v_isSharedCheck_1035_ == 0)
{
v___x_993_ = v___x_990_;
v_isShared_994_ = v_isSharedCheck_1035_;
goto v_resetjp_992_;
}
else
{
lean_inc(v_a_991_);
lean_dec(v___x_990_);
v___x_993_ = lean_box(0);
v_isShared_994_ = v_isSharedCheck_1035_;
goto v_resetjp_992_;
}
v_resetjp_992_:
{
lean_object* v___x_995_; lean_object* v_traceState_996_; lean_object* v_env_997_; lean_object* v_nextMacroScope_998_; lean_object* v_ngen_999_; lean_object* v_auxDeclNGen_1000_; lean_object* v_cache_1001_; lean_object* v_messages_1002_; lean_object* v_infoState_1003_; lean_object* v_snapshotTasks_1004_; lean_object* v___x_1006_; uint8_t v_isShared_1007_; uint8_t v_isSharedCheck_1034_; 
v___x_995_ = lean_st_ref_take(v___y_987_);
v_traceState_996_ = lean_ctor_get(v___x_995_, 4);
v_env_997_ = lean_ctor_get(v___x_995_, 0);
v_nextMacroScope_998_ = lean_ctor_get(v___x_995_, 1);
v_ngen_999_ = lean_ctor_get(v___x_995_, 2);
v_auxDeclNGen_1000_ = lean_ctor_get(v___x_995_, 3);
v_cache_1001_ = lean_ctor_get(v___x_995_, 5);
v_messages_1002_ = lean_ctor_get(v___x_995_, 6);
v_infoState_1003_ = lean_ctor_get(v___x_995_, 7);
v_snapshotTasks_1004_ = lean_ctor_get(v___x_995_, 8);
v_isSharedCheck_1034_ = !lean_is_exclusive(v___x_995_);
if (v_isSharedCheck_1034_ == 0)
{
v___x_1006_ = v___x_995_;
v_isShared_1007_ = v_isSharedCheck_1034_;
goto v_resetjp_1005_;
}
else
{
lean_inc(v_snapshotTasks_1004_);
lean_inc(v_infoState_1003_);
lean_inc(v_messages_1002_);
lean_inc(v_cache_1001_);
lean_inc(v_traceState_996_);
lean_inc(v_auxDeclNGen_1000_);
lean_inc(v_ngen_999_);
lean_inc(v_nextMacroScope_998_);
lean_inc(v_env_997_);
lean_dec(v___x_995_);
v___x_1006_ = lean_box(0);
v_isShared_1007_ = v_isSharedCheck_1034_;
goto v_resetjp_1005_;
}
v_resetjp_1005_:
{
uint64_t v_tid_1008_; lean_object* v_traces_1009_; lean_object* v___x_1011_; uint8_t v_isShared_1012_; uint8_t v_isSharedCheck_1033_; 
v_tid_1008_ = lean_ctor_get_uint64(v_traceState_996_, sizeof(void*)*1);
v_traces_1009_ = lean_ctor_get(v_traceState_996_, 0);
v_isSharedCheck_1033_ = !lean_is_exclusive(v_traceState_996_);
if (v_isSharedCheck_1033_ == 0)
{
v___x_1011_ = v_traceState_996_;
v_isShared_1012_ = v_isSharedCheck_1033_;
goto v_resetjp_1010_;
}
else
{
lean_inc(v_traces_1009_);
lean_dec(v_traceState_996_);
v___x_1011_ = lean_box(0);
v_isShared_1012_ = v_isSharedCheck_1033_;
goto v_resetjp_1010_;
}
v_resetjp_1010_:
{
lean_object* v___x_1013_; double v___x_1014_; uint8_t v___x_1015_; lean_object* v___x_1016_; lean_object* v___x_1017_; lean_object* v___x_1018_; lean_object* v___x_1019_; lean_object* v___x_1020_; lean_object* v___x_1021_; lean_object* v___x_1023_; 
v___x_1013_ = lean_box(0);
v___x_1014_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg___closed__0, &lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg___closed__0);
v___x_1015_ = 0;
v___x_1016_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg___closed__1));
v___x_1017_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_1017_, 0, v_cls_982_);
lean_ctor_set(v___x_1017_, 1, v___x_1013_);
lean_ctor_set(v___x_1017_, 2, v___x_1016_);
lean_ctor_set_float(v___x_1017_, sizeof(void*)*3, v___x_1014_);
lean_ctor_set_float(v___x_1017_, sizeof(void*)*3 + 8, v___x_1014_);
lean_ctor_set_uint8(v___x_1017_, sizeof(void*)*3 + 16, v___x_1015_);
v___x_1018_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg___closed__2));
v___x_1019_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_1019_, 0, v___x_1017_);
lean_ctor_set(v___x_1019_, 1, v_a_991_);
lean_ctor_set(v___x_1019_, 2, v___x_1018_);
lean_inc(v_ref_989_);
v___x_1020_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1020_, 0, v_ref_989_);
lean_ctor_set(v___x_1020_, 1, v___x_1019_);
v___x_1021_ = l_Lean_PersistentArray_push___redArg(v_traces_1009_, v___x_1020_);
if (v_isShared_1012_ == 0)
{
lean_ctor_set(v___x_1011_, 0, v___x_1021_);
v___x_1023_ = v___x_1011_;
goto v_reusejp_1022_;
}
else
{
lean_object* v_reuseFailAlloc_1032_; 
v_reuseFailAlloc_1032_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1032_, 0, v___x_1021_);
lean_ctor_set_uint64(v_reuseFailAlloc_1032_, sizeof(void*)*1, v_tid_1008_);
v___x_1023_ = v_reuseFailAlloc_1032_;
goto v_reusejp_1022_;
}
v_reusejp_1022_:
{
lean_object* v___x_1025_; 
if (v_isShared_1007_ == 0)
{
lean_ctor_set(v___x_1006_, 4, v___x_1023_);
v___x_1025_ = v___x_1006_;
goto v_reusejp_1024_;
}
else
{
lean_object* v_reuseFailAlloc_1031_; 
v_reuseFailAlloc_1031_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1031_, 0, v_env_997_);
lean_ctor_set(v_reuseFailAlloc_1031_, 1, v_nextMacroScope_998_);
lean_ctor_set(v_reuseFailAlloc_1031_, 2, v_ngen_999_);
lean_ctor_set(v_reuseFailAlloc_1031_, 3, v_auxDeclNGen_1000_);
lean_ctor_set(v_reuseFailAlloc_1031_, 4, v___x_1023_);
lean_ctor_set(v_reuseFailAlloc_1031_, 5, v_cache_1001_);
lean_ctor_set(v_reuseFailAlloc_1031_, 6, v_messages_1002_);
lean_ctor_set(v_reuseFailAlloc_1031_, 7, v_infoState_1003_);
lean_ctor_set(v_reuseFailAlloc_1031_, 8, v_snapshotTasks_1004_);
v___x_1025_ = v_reuseFailAlloc_1031_;
goto v_reusejp_1024_;
}
v_reusejp_1024_:
{
lean_object* v___x_1026_; lean_object* v___x_1027_; lean_object* v___x_1029_; 
v___x_1026_ = lean_st_ref_set(v___y_987_, v___x_1025_);
v___x_1027_ = lean_box(0);
if (v_isShared_994_ == 0)
{
lean_ctor_set(v___x_993_, 0, v___x_1027_);
v___x_1029_ = v___x_993_;
goto v_reusejp_1028_;
}
else
{
lean_object* v_reuseFailAlloc_1030_; 
v_reuseFailAlloc_1030_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1030_, 0, v___x_1027_);
v___x_1029_ = v_reuseFailAlloc_1030_;
goto v_reusejp_1028_;
}
v_reusejp_1028_:
{
return v___x_1029_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg___boxed(lean_object* v_cls_1036_, lean_object* v_msg_1037_, lean_object* v___y_1038_, lean_object* v___y_1039_, lean_object* v___y_1040_, lean_object* v___y_1041_, lean_object* v___y_1042_){
_start:
{
lean_object* v_res_1043_; 
v_res_1043_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg(v_cls_1036_, v_msg_1037_, v___y_1038_, v___y_1039_, v___y_1040_, v___y_1041_);
lean_dec(v___y_1041_);
lean_dec_ref(v___y_1040_);
lean_dec(v___y_1039_);
lean_dec_ref(v___y_1038_);
return v_res_1043_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__3(void){
_start:
{
lean_object* v___x_1048_; 
v___x_1048_ = l_Array_mkArray0(lean_box(0));
return v___x_1048_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__8(void){
_start:
{
lean_object* v___x_1055_; lean_object* v___x_1056_; lean_object* v___x_1057_; 
v___x_1055_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__1_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_));
v___x_1056_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__7));
v___x_1057_ = l_Lean_Name_append(v___x_1056_, v___x_1055_);
return v___x_1057_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__10(void){
_start:
{
lean_object* v___x_1059_; lean_object* v___x_1060_; 
v___x_1059_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__9));
v___x_1060_ = l_Lean_stringToMessageData(v___x_1059_);
return v___x_1060_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1(lean_object* v_ty_1061_, lean_object* v_binder_1062_, lean_object* v___y_1063_, lean_object* v___y_1064_, lean_object* v___y_1065_, lean_object* v___y_1066_, lean_object* v___y_1067_, lean_object* v___y_1068_){
_start:
{
lean_object* v___x_1070_; 
v___x_1070_ = l_Lean_Elab_Term_elabType(v_ty_1061_, v___y_1063_, v___y_1064_, v___y_1065_, v___y_1066_, v___y_1067_, v___y_1068_);
if (lean_obj_tag(v___x_1070_) == 0)
{
uint8_t v___x_1071_; uint8_t v___x_1072_; lean_object* v___x_1073_; 
lean_dec_ref_known(v___x_1070_, 1);
v___x_1071_ = 0;
v___x_1072_ = 1;
v___x_1073_ = l_Lean_Elab_Term_synthesizeSyntheticMVars(v___x_1071_, v___x_1072_, v___y_1063_, v___y_1064_, v___y_1065_, v___y_1066_, v___y_1067_, v___y_1068_);
if (lean_obj_tag(v___x_1073_) == 0)
{
lean_object* v___x_1074_; 
lean_dec_ref_known(v___x_1073_, 1);
lean_inc(v_binder_1062_);
v___x_1074_ = lp_mathlib_Mathlib_Command_Variable_pendingActionableSynthMVar(v_binder_1062_, v___y_1063_, v___y_1064_, v___y_1065_, v___y_1066_, v___y_1067_, v___y_1068_);
if (lean_obj_tag(v___x_1074_) == 0)
{
lean_object* v_a_1075_; lean_object* v___x_1077_; uint8_t v_isShared_1078_; uint8_t v_isSharedCheck_1196_; 
v_a_1075_ = lean_ctor_get(v___x_1074_, 0);
v_isSharedCheck_1196_ = !lean_is_exclusive(v___x_1074_);
if (v_isSharedCheck_1196_ == 0)
{
v___x_1077_ = v___x_1074_;
v_isShared_1078_ = v_isSharedCheck_1196_;
goto v_resetjp_1076_;
}
else
{
lean_inc(v_a_1075_);
lean_dec(v___x_1074_);
v___x_1077_ = lean_box(0);
v_isShared_1078_ = v_isSharedCheck_1196_;
goto v_resetjp_1076_;
}
v_resetjp_1076_:
{
if (lean_obj_tag(v_a_1075_) == 1)
{
lean_object* v_val_1079_; lean_object* v___x_1081_; uint8_t v_isShared_1082_; uint8_t v_isSharedCheck_1191_; 
lean_del_object(v___x_1077_);
v_val_1079_ = lean_ctor_get(v_a_1075_, 0);
v_isSharedCheck_1191_ = !lean_is_exclusive(v_a_1075_);
if (v_isSharedCheck_1191_ == 0)
{
v___x_1081_ = v_a_1075_;
v_isShared_1082_ = v_isSharedCheck_1191_;
goto v_resetjp_1080_;
}
else
{
lean_inc(v_val_1079_);
lean_dec(v_a_1075_);
v___x_1081_ = lean_box(0);
v_isShared_1082_ = v_isSharedCheck_1191_;
goto v_resetjp_1080_;
}
v_resetjp_1080_:
{
lean_object* v___y_1084_; lean_object* v___y_1085_; lean_object* v___y_1086_; lean_object* v___y_1087_; lean_object* v___y_1088_; lean_object* v___y_1089_; lean_object* v___y_1090_; lean_object* v_options_1141_; lean_object* v_lctx_1142_; lean_object* v_inheritedTraceOptions_1143_; uint8_t v_hasTrace_1144_; lean_object* v___x_1145_; lean_object* v___y_1147_; lean_object* v___y_1148_; lean_object* v___y_1149_; lean_object* v___y_1150_; lean_object* v___y_1151_; lean_object* v___y_1152_; 
v_options_1141_ = lean_ctor_get(v___y_1067_, 2);
v_lctx_1142_ = lean_ctor_get(v___y_1065_, 2);
v_inheritedTraceOptions_1143_ = lean_ctor_get(v___y_1067_, 13);
v_hasTrace_1144_ = lean_ctor_get_uint8(v_options_1141_, sizeof(void*)*1);
v___x_1145_ = l_Lean_LocalContext_getFVarIds(v_lctx_1142_);
if (v_hasTrace_1144_ == 0)
{
v___y_1147_ = v___y_1063_;
v___y_1148_ = v___y_1064_;
v___y_1149_ = v___y_1065_;
v___y_1150_ = v___y_1066_;
v___y_1151_ = v___y_1067_;
v___y_1152_ = v___y_1068_;
goto v___jp_1146_;
}
else
{
lean_object* v___x_1176_; lean_object* v___x_1177_; uint8_t v___x_1178_; 
v___x_1176_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__1_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_));
v___x_1177_ = lean_obj_once(&lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__8, &lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__8_once, _init_lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__8);
v___x_1178_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1143_, v_options_1141_, v___x_1177_);
if (v___x_1178_ == 0)
{
v___y_1147_ = v___y_1063_;
v___y_1148_ = v___y_1064_;
v___y_1149_ = v___y_1065_;
v___y_1150_ = v___y_1066_;
v___y_1151_ = v___y_1067_;
v___y_1152_ = v___y_1068_;
goto v___jp_1146_;
}
else
{
lean_object* v___x_1179_; lean_object* v___x_1180_; lean_object* v___x_1181_; lean_object* v___x_1182_; 
v___x_1179_ = lean_obj_once(&lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__10, &lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__10_once, _init_lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__10);
lean_inc(v_val_1079_);
v___x_1180_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1180_, 0, v_val_1079_);
v___x_1181_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1181_, 0, v___x_1179_);
lean_ctor_set(v___x_1181_, 1, v___x_1180_);
v___x_1182_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg(v___x_1176_, v___x_1181_, v___y_1065_, v___y_1066_, v___y_1067_, v___y_1068_);
if (lean_obj_tag(v___x_1182_) == 0)
{
lean_dec_ref_known(v___x_1182_, 1);
v___y_1147_ = v___y_1063_;
v___y_1148_ = v___y_1064_;
v___y_1149_ = v___y_1065_;
v___y_1150_ = v___y_1066_;
v___y_1151_ = v___y_1067_;
v___y_1152_ = v___y_1068_;
goto v___jp_1146_;
}
else
{
lean_object* v_a_1183_; lean_object* v___x_1185_; uint8_t v_isShared_1186_; uint8_t v_isSharedCheck_1190_; 
lean_dec_ref(v___x_1145_);
lean_del_object(v___x_1081_);
lean_dec(v_val_1079_);
lean_dec(v_binder_1062_);
v_a_1183_ = lean_ctor_get(v___x_1182_, 0);
v_isSharedCheck_1190_ = !lean_is_exclusive(v___x_1182_);
if (v_isSharedCheck_1190_ == 0)
{
v___x_1185_ = v___x_1182_;
v_isShared_1186_ = v_isSharedCheck_1190_;
goto v_resetjp_1184_;
}
else
{
lean_inc(v_a_1183_);
lean_dec(v___x_1182_);
v___x_1185_ = lean_box(0);
v_isShared_1186_ = v_isSharedCheck_1190_;
goto v_resetjp_1184_;
}
v_resetjp_1184_:
{
lean_object* v___x_1188_; 
if (v_isShared_1186_ == 0)
{
v___x_1188_ = v___x_1185_;
goto v_reusejp_1187_;
}
else
{
lean_object* v_reuseFailAlloc_1189_; 
v_reuseFailAlloc_1189_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1189_, 0, v_a_1183_);
v___x_1188_ = v_reuseFailAlloc_1189_;
goto v_reusejp_1187_;
}
v_reusejp_1187_:
{
return v___x_1188_;
}
}
}
}
}
v___jp_1083_:
{
lean_object* v___x_1091_; lean_object* v___f_1092_; lean_object* v___x_1093_; 
v___x_1091_ = lean_box(v___x_1072_);
lean_inc_n(v_val_1079_, 2);
v___f_1092_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__0___boxed), 10, 3);
lean_closure_set(v___f_1092_, 0, v_val_1079_);
lean_closure_set(v___f_1092_, 1, v___y_1090_);
lean_closure_set(v___f_1092_, 2, v___x_1091_);
v___x_1093_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Command_Variable_getSubproblem_spec__2___redArg(v_val_1079_, v___f_1092_, v___y_1088_, v___y_1085_, v___y_1086_, v___y_1087_, v___y_1084_, v___y_1089_);
if (lean_obj_tag(v___x_1093_) == 0)
{
lean_object* v_a_1094_; lean_object* v___x_1095_; lean_object* v___x_1096_; 
v_a_1094_ = lean_ctor_get(v___x_1093_, 0);
lean_inc(v_a_1094_);
lean_dec_ref_known(v___x_1093_, 1);
v___x_1095_ = lean_box(1);
v___x_1096_ = l_Lean_PrettyPrinter_delab(v_a_1094_, v___x_1095_, v___y_1086_, v___y_1087_, v___y_1084_, v___y_1089_);
if (lean_obj_tag(v___x_1096_) == 0)
{
lean_object* v_a_1097_; lean_object* v_ref_1098_; lean_object* v___x_1099_; lean_object* v___x_1100_; lean_object* v_a_1101_; lean_object* v___x_1103_; uint8_t v_isShared_1104_; uint8_t v_isSharedCheck_1124_; 
v_a_1097_ = lean_ctor_get(v___x_1096_, 0);
lean_inc(v_a_1097_);
lean_dec_ref_known(v___x_1096_, 1);
v_ref_1098_ = lean_ctor_get(v___y_1084_, 5);
v___x_1099_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1099_, 0, v_val_1079_);
v___x_1100_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__3(v___x_1099_, v___y_1086_, v___y_1087_, v___y_1084_, v___y_1089_);
v_a_1101_ = lean_ctor_get(v___x_1100_, 0);
v_isSharedCheck_1124_ = !lean_is_exclusive(v___x_1100_);
if (v_isSharedCheck_1124_ == 0)
{
v___x_1103_ = v___x_1100_;
v_isShared_1104_ = v_isSharedCheck_1124_;
goto v_resetjp_1102_;
}
else
{
lean_inc(v_a_1101_);
lean_dec(v___x_1100_);
v___x_1103_ = lean_box(0);
v_isShared_1104_ = v_isSharedCheck_1124_;
goto v_resetjp_1102_;
}
v_resetjp_1102_:
{
lean_object* v_ref_1105_; uint8_t v___x_1106_; lean_object* v___x_1107_; lean_object* v___x_1108_; lean_object* v___x_1109_; lean_object* v___x_1110_; lean_object* v___x_1111_; lean_object* v___x_1112_; lean_object* v___x_1113_; lean_object* v___x_1114_; lean_object* v___x_1115_; lean_object* v___x_1116_; lean_object* v___x_1117_; lean_object* v___x_1119_; 
v_ref_1105_ = l_Lean_replaceRef(v_binder_1062_, v_ref_1098_);
lean_dec(v_binder_1062_);
v___x_1106_ = 0;
v___x_1107_ = l_Lean_SourceInfo_fromRef(v_ref_1105_, v___x_1106_);
lean_dec(v_ref_1105_);
v___x_1108_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__0));
lean_inc_n(v___x_1107_, 3);
v___x_1109_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1109_, 0, v___x_1107_);
lean_ctor_set(v___x_1109_, 1, v___x_1108_);
v___x_1110_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__2));
v___x_1111_ = lean_obj_once(&lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__3, &lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__3_once, _init_lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__3);
v___x_1112_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1112_, 0, v___x_1107_);
lean_ctor_set(v___x_1112_, 1, v___x_1110_);
lean_ctor_set(v___x_1112_, 2, v___x_1111_);
v___x_1113_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__4));
v___x_1114_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1114_, 0, v___x_1107_);
lean_ctor_set(v___x_1114_, 1, v___x_1113_);
v___x_1115_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__10));
v___x_1116_ = l_Lean_Syntax_node4(v___x_1107_, v___x_1115_, v___x_1109_, v___x_1112_, v_a_1097_, v___x_1114_);
v___x_1117_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1117_, 0, v_a_1101_);
lean_ctor_set(v___x_1117_, 1, v___x_1116_);
if (v_isShared_1082_ == 0)
{
lean_ctor_set(v___x_1081_, 0, v___x_1117_);
v___x_1119_ = v___x_1081_;
goto v_reusejp_1118_;
}
else
{
lean_object* v_reuseFailAlloc_1123_; 
v_reuseFailAlloc_1123_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1123_, 0, v___x_1117_);
v___x_1119_ = v_reuseFailAlloc_1123_;
goto v_reusejp_1118_;
}
v_reusejp_1118_:
{
lean_object* v___x_1121_; 
if (v_isShared_1104_ == 0)
{
lean_ctor_set(v___x_1103_, 0, v___x_1119_);
v___x_1121_ = v___x_1103_;
goto v_reusejp_1120_;
}
else
{
lean_object* v_reuseFailAlloc_1122_; 
v_reuseFailAlloc_1122_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1122_, 0, v___x_1119_);
v___x_1121_ = v_reuseFailAlloc_1122_;
goto v_reusejp_1120_;
}
v_reusejp_1120_:
{
return v___x_1121_;
}
}
}
}
else
{
lean_object* v_a_1125_; lean_object* v___x_1127_; uint8_t v_isShared_1128_; uint8_t v_isSharedCheck_1132_; 
lean_del_object(v___x_1081_);
lean_dec(v_val_1079_);
lean_dec(v_binder_1062_);
v_a_1125_ = lean_ctor_get(v___x_1096_, 0);
v_isSharedCheck_1132_ = !lean_is_exclusive(v___x_1096_);
if (v_isSharedCheck_1132_ == 0)
{
v___x_1127_ = v___x_1096_;
v_isShared_1128_ = v_isSharedCheck_1132_;
goto v_resetjp_1126_;
}
else
{
lean_inc(v_a_1125_);
lean_dec(v___x_1096_);
v___x_1127_ = lean_box(0);
v_isShared_1128_ = v_isSharedCheck_1132_;
goto v_resetjp_1126_;
}
v_resetjp_1126_:
{
lean_object* v___x_1130_; 
if (v_isShared_1128_ == 0)
{
v___x_1130_ = v___x_1127_;
goto v_reusejp_1129_;
}
else
{
lean_object* v_reuseFailAlloc_1131_; 
v_reuseFailAlloc_1131_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1131_, 0, v_a_1125_);
v___x_1130_ = v_reuseFailAlloc_1131_;
goto v_reusejp_1129_;
}
v_reusejp_1129_:
{
return v___x_1130_;
}
}
}
}
else
{
lean_object* v_a_1133_; lean_object* v___x_1135_; uint8_t v_isShared_1136_; uint8_t v_isSharedCheck_1140_; 
lean_del_object(v___x_1081_);
lean_dec(v_val_1079_);
lean_dec(v_binder_1062_);
v_a_1133_ = lean_ctor_get(v___x_1093_, 0);
v_isSharedCheck_1140_ = !lean_is_exclusive(v___x_1093_);
if (v_isSharedCheck_1140_ == 0)
{
v___x_1135_ = v___x_1093_;
v_isShared_1136_ = v_isSharedCheck_1140_;
goto v_resetjp_1134_;
}
else
{
lean_inc(v_a_1133_);
lean_dec(v___x_1093_);
v___x_1135_ = lean_box(0);
v_isShared_1136_ = v_isSharedCheck_1140_;
goto v_resetjp_1134_;
}
v_resetjp_1134_:
{
lean_object* v___x_1138_; 
if (v_isShared_1136_ == 0)
{
v___x_1138_ = v___x_1135_;
goto v_reusejp_1137_;
}
else
{
lean_object* v_reuseFailAlloc_1139_; 
v_reuseFailAlloc_1139_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1139_, 0, v_a_1133_);
v___x_1138_ = v_reuseFailAlloc_1139_;
goto v_reusejp_1137_;
}
v_reusejp_1137_:
{
return v___x_1138_;
}
}
}
}
v___jp_1146_:
{
lean_object* v___x_1153_; 
lean_inc(v_val_1079_);
v___x_1153_ = l_Lean_MVarId_getDecl(v_val_1079_, v___y_1149_, v___y_1150_, v___y_1151_, v___y_1152_);
if (lean_obj_tag(v___x_1153_) == 0)
{
lean_object* v_a_1154_; lean_object* v_lctx_1155_; lean_object* v___x_1156_; lean_object* v___x_1157_; lean_object* v___x_1158_; lean_object* v___x_1159_; uint8_t v___x_1160_; 
v_a_1154_ = lean_ctor_get(v___x_1153_, 0);
lean_inc(v_a_1154_);
lean_dec_ref_known(v___x_1153_, 1);
v_lctx_1155_ = lean_ctor_get(v_a_1154_, 1);
lean_inc_ref(v_lctx_1155_);
lean_dec(v_a_1154_);
v___x_1156_ = l_Lean_LocalContext_getFVarIds(v_lctx_1155_);
lean_dec_ref(v_lctx_1155_);
v___x_1157_ = lean_unsigned_to_nat(0u);
v___x_1158_ = lean_array_get_size(v___x_1156_);
v___x_1159_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__5));
v___x_1160_ = lean_nat_dec_lt(v___x_1157_, v___x_1158_);
if (v___x_1160_ == 0)
{
lean_dec_ref(v___x_1156_);
lean_dec_ref(v___x_1145_);
v___y_1084_ = v___y_1151_;
v___y_1085_ = v___y_1148_;
v___y_1086_ = v___y_1149_;
v___y_1087_ = v___y_1150_;
v___y_1088_ = v___y_1147_;
v___y_1089_ = v___y_1152_;
v___y_1090_ = v___x_1159_;
goto v___jp_1083_;
}
else
{
uint8_t v___x_1161_; 
v___x_1161_ = lean_nat_dec_le(v___x_1158_, v___x_1158_);
if (v___x_1161_ == 0)
{
if (v___x_1160_ == 0)
{
lean_dec_ref(v___x_1156_);
lean_dec_ref(v___x_1145_);
v___y_1084_ = v___y_1151_;
v___y_1085_ = v___y_1148_;
v___y_1086_ = v___y_1149_;
v___y_1087_ = v___y_1150_;
v___y_1088_ = v___y_1147_;
v___y_1089_ = v___y_1152_;
v___y_1090_ = v___x_1159_;
goto v___jp_1083_;
}
else
{
size_t v___x_1162_; size_t v___x_1163_; lean_object* v___x_1164_; 
v___x_1162_ = ((size_t)0ULL);
v___x_1163_ = lean_usize_of_nat(v___x_1158_);
v___x_1164_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Command_Variable_getSubproblem_spec__3(v___x_1145_, v___x_1156_, v___x_1162_, v___x_1163_, v___x_1159_);
lean_dec_ref(v___x_1156_);
lean_dec_ref(v___x_1145_);
v___y_1084_ = v___y_1151_;
v___y_1085_ = v___y_1148_;
v___y_1086_ = v___y_1149_;
v___y_1087_ = v___y_1150_;
v___y_1088_ = v___y_1147_;
v___y_1089_ = v___y_1152_;
v___y_1090_ = v___x_1164_;
goto v___jp_1083_;
}
}
else
{
size_t v___x_1165_; size_t v___x_1166_; lean_object* v___x_1167_; 
v___x_1165_ = ((size_t)0ULL);
v___x_1166_ = lean_usize_of_nat(v___x_1158_);
v___x_1167_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Command_Variable_getSubproblem_spec__3(v___x_1145_, v___x_1156_, v___x_1165_, v___x_1166_, v___x_1159_);
lean_dec_ref(v___x_1156_);
lean_dec_ref(v___x_1145_);
v___y_1084_ = v___y_1151_;
v___y_1085_ = v___y_1148_;
v___y_1086_ = v___y_1149_;
v___y_1087_ = v___y_1150_;
v___y_1088_ = v___y_1147_;
v___y_1089_ = v___y_1152_;
v___y_1090_ = v___x_1167_;
goto v___jp_1083_;
}
}
}
else
{
lean_object* v_a_1168_; lean_object* v___x_1170_; uint8_t v_isShared_1171_; uint8_t v_isSharedCheck_1175_; 
lean_dec_ref(v___x_1145_);
lean_del_object(v___x_1081_);
lean_dec(v_val_1079_);
lean_dec(v_binder_1062_);
v_a_1168_ = lean_ctor_get(v___x_1153_, 0);
v_isSharedCheck_1175_ = !lean_is_exclusive(v___x_1153_);
if (v_isSharedCheck_1175_ == 0)
{
v___x_1170_ = v___x_1153_;
v_isShared_1171_ = v_isSharedCheck_1175_;
goto v_resetjp_1169_;
}
else
{
lean_inc(v_a_1168_);
lean_dec(v___x_1153_);
v___x_1170_ = lean_box(0);
v_isShared_1171_ = v_isSharedCheck_1175_;
goto v_resetjp_1169_;
}
v_resetjp_1169_:
{
lean_object* v___x_1173_; 
if (v_isShared_1171_ == 0)
{
v___x_1173_ = v___x_1170_;
goto v_reusejp_1172_;
}
else
{
lean_object* v_reuseFailAlloc_1174_; 
v_reuseFailAlloc_1174_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1174_, 0, v_a_1168_);
v___x_1173_ = v_reuseFailAlloc_1174_;
goto v_reusejp_1172_;
}
v_reusejp_1172_:
{
return v___x_1173_;
}
}
}
}
}
}
else
{
lean_object* v___x_1192_; lean_object* v___x_1194_; 
lean_dec(v_a_1075_);
lean_dec(v_binder_1062_);
v___x_1192_ = lean_box(0);
if (v_isShared_1078_ == 0)
{
lean_ctor_set(v___x_1077_, 0, v___x_1192_);
v___x_1194_ = v___x_1077_;
goto v_reusejp_1193_;
}
else
{
lean_object* v_reuseFailAlloc_1195_; 
v_reuseFailAlloc_1195_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1195_, 0, v___x_1192_);
v___x_1194_ = v_reuseFailAlloc_1195_;
goto v_reusejp_1193_;
}
v_reusejp_1193_:
{
return v___x_1194_;
}
}
}
}
else
{
lean_object* v_a_1197_; lean_object* v___x_1199_; uint8_t v_isShared_1200_; uint8_t v_isSharedCheck_1204_; 
lean_dec(v_binder_1062_);
v_a_1197_ = lean_ctor_get(v___x_1074_, 0);
v_isSharedCheck_1204_ = !lean_is_exclusive(v___x_1074_);
if (v_isSharedCheck_1204_ == 0)
{
v___x_1199_ = v___x_1074_;
v_isShared_1200_ = v_isSharedCheck_1204_;
goto v_resetjp_1198_;
}
else
{
lean_inc(v_a_1197_);
lean_dec(v___x_1074_);
v___x_1199_ = lean_box(0);
v_isShared_1200_ = v_isSharedCheck_1204_;
goto v_resetjp_1198_;
}
v_resetjp_1198_:
{
lean_object* v___x_1202_; 
if (v_isShared_1200_ == 0)
{
v___x_1202_ = v___x_1199_;
goto v_reusejp_1201_;
}
else
{
lean_object* v_reuseFailAlloc_1203_; 
v_reuseFailAlloc_1203_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1203_, 0, v_a_1197_);
v___x_1202_ = v_reuseFailAlloc_1203_;
goto v_reusejp_1201_;
}
v_reusejp_1201_:
{
return v___x_1202_;
}
}
}
}
else
{
lean_object* v_a_1205_; lean_object* v___x_1207_; uint8_t v_isShared_1208_; uint8_t v_isSharedCheck_1212_; 
lean_dec(v_binder_1062_);
v_a_1205_ = lean_ctor_get(v___x_1073_, 0);
v_isSharedCheck_1212_ = !lean_is_exclusive(v___x_1073_);
if (v_isSharedCheck_1212_ == 0)
{
v___x_1207_ = v___x_1073_;
v_isShared_1208_ = v_isSharedCheck_1212_;
goto v_resetjp_1206_;
}
else
{
lean_inc(v_a_1205_);
lean_dec(v___x_1073_);
v___x_1207_ = lean_box(0);
v_isShared_1208_ = v_isSharedCheck_1212_;
goto v_resetjp_1206_;
}
v_resetjp_1206_:
{
lean_object* v___x_1210_; 
if (v_isShared_1208_ == 0)
{
v___x_1210_ = v___x_1207_;
goto v_reusejp_1209_;
}
else
{
lean_object* v_reuseFailAlloc_1211_; 
v_reuseFailAlloc_1211_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1211_, 0, v_a_1205_);
v___x_1210_ = v_reuseFailAlloc_1211_;
goto v_reusejp_1209_;
}
v_reusejp_1209_:
{
return v___x_1210_;
}
}
}
}
else
{
lean_object* v_a_1213_; lean_object* v___x_1215_; uint8_t v_isShared_1216_; uint8_t v_isSharedCheck_1220_; 
lean_dec(v_binder_1062_);
v_a_1213_ = lean_ctor_get(v___x_1070_, 0);
v_isSharedCheck_1220_ = !lean_is_exclusive(v___x_1070_);
if (v_isSharedCheck_1220_ == 0)
{
v___x_1215_ = v___x_1070_;
v_isShared_1216_ = v_isSharedCheck_1220_;
goto v_resetjp_1214_;
}
else
{
lean_inc(v_a_1213_);
lean_dec(v___x_1070_);
v___x_1215_ = lean_box(0);
v_isShared_1216_ = v_isSharedCheck_1220_;
goto v_resetjp_1214_;
}
v_resetjp_1214_:
{
lean_object* v___x_1218_; 
if (v_isShared_1216_ == 0)
{
v___x_1218_ = v___x_1215_;
goto v_reusejp_1217_;
}
else
{
lean_object* v_reuseFailAlloc_1219_; 
v_reuseFailAlloc_1219_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1219_, 0, v_a_1213_);
v___x_1218_ = v_reuseFailAlloc_1219_;
goto v_reusejp_1217_;
}
v_reusejp_1217_:
{
return v___x_1218_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___boxed(lean_object* v_ty_1221_, lean_object* v_binder_1222_, lean_object* v___y_1223_, lean_object* v___y_1224_, lean_object* v___y_1225_, lean_object* v___y_1226_, lean_object* v___y_1227_, lean_object* v___y_1228_, lean_object* v___y_1229_){
_start:
{
lean_object* v_res_1230_; 
v_res_1230_ = lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1(v_ty_1221_, v_binder_1222_, v___y_1223_, v___y_1224_, v___y_1225_, v___y_1226_, v___y_1227_, v___y_1228_);
lean_dec(v___y_1228_);
lean_dec_ref(v___y_1227_);
lean_dec(v___y_1226_);
lean_dec_ref(v___y_1225_);
lean_dec(v___y_1224_);
lean_dec_ref(v___y_1223_);
return v_res_1230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__2(lean_object* v___f_1231_, lean_object* v___y_1232_, lean_object* v___y_1233_, lean_object* v___y_1234_, lean_object* v___y_1235_, lean_object* v___y_1236_, lean_object* v___y_1237_){
_start:
{
lean_object* v_declName_x3f_1239_; lean_object* v_macroStack_1240_; uint8_t v_mayPostpone_1241_; uint8_t v_errToSorry_1242_; lean_object* v_autoBoundImplicitContext_1243_; lean_object* v_autoBoundImplicitForbidden_1244_; lean_object* v_sectionVars_1245_; lean_object* v_sectionFVars_1246_; uint8_t v_implicitLambda_1247_; uint8_t v_heedElabAsElim_1248_; uint8_t v_isNoncomputableSection_1249_; uint8_t v_isMetaSection_1250_; uint8_t v_inPattern_1251_; lean_object* v_tacSnap_x3f_1252_; uint8_t v_saveRecAppSyntax_1253_; uint8_t v_holesAsSyntheticOpaque_1254_; uint8_t v_checkDeprecated_1255_; lean_object* v_fixedTermElabs_1256_; lean_object* v___x_1258_; uint8_t v_isShared_1259_; uint8_t v_isSharedCheck_1265_; 
v_declName_x3f_1239_ = lean_ctor_get(v___y_1232_, 0);
v_macroStack_1240_ = lean_ctor_get(v___y_1232_, 1);
v_mayPostpone_1241_ = lean_ctor_get_uint8(v___y_1232_, sizeof(void*)*8);
v_errToSorry_1242_ = lean_ctor_get_uint8(v___y_1232_, sizeof(void*)*8 + 1);
v_autoBoundImplicitContext_1243_ = lean_ctor_get(v___y_1232_, 2);
v_autoBoundImplicitForbidden_1244_ = lean_ctor_get(v___y_1232_, 3);
v_sectionVars_1245_ = lean_ctor_get(v___y_1232_, 4);
v_sectionFVars_1246_ = lean_ctor_get(v___y_1232_, 5);
v_implicitLambda_1247_ = lean_ctor_get_uint8(v___y_1232_, sizeof(void*)*8 + 2);
v_heedElabAsElim_1248_ = lean_ctor_get_uint8(v___y_1232_, sizeof(void*)*8 + 3);
v_isNoncomputableSection_1249_ = lean_ctor_get_uint8(v___y_1232_, sizeof(void*)*8 + 4);
v_isMetaSection_1250_ = lean_ctor_get_uint8(v___y_1232_, sizeof(void*)*8 + 5);
v_inPattern_1251_ = lean_ctor_get_uint8(v___y_1232_, sizeof(void*)*8 + 7);
v_tacSnap_x3f_1252_ = lean_ctor_get(v___y_1232_, 6);
v_saveRecAppSyntax_1253_ = lean_ctor_get_uint8(v___y_1232_, sizeof(void*)*8 + 8);
v_holesAsSyntheticOpaque_1254_ = lean_ctor_get_uint8(v___y_1232_, sizeof(void*)*8 + 9);
v_checkDeprecated_1255_ = lean_ctor_get_uint8(v___y_1232_, sizeof(void*)*8 + 10);
v_fixedTermElabs_1256_ = lean_ctor_get(v___y_1232_, 7);
v_isSharedCheck_1265_ = !lean_is_exclusive(v___y_1232_);
if (v_isSharedCheck_1265_ == 0)
{
v___x_1258_ = v___y_1232_;
v_isShared_1259_ = v_isSharedCheck_1265_;
goto v_resetjp_1257_;
}
else
{
lean_inc(v_fixedTermElabs_1256_);
lean_inc(v_tacSnap_x3f_1252_);
lean_inc(v_sectionFVars_1246_);
lean_inc(v_sectionVars_1245_);
lean_inc(v_autoBoundImplicitForbidden_1244_);
lean_inc(v_autoBoundImplicitContext_1243_);
lean_inc(v_macroStack_1240_);
lean_inc(v_declName_x3f_1239_);
lean_dec(v___y_1232_);
v___x_1258_ = lean_box(0);
v_isShared_1259_ = v_isSharedCheck_1265_;
goto v_resetjp_1257_;
}
v_resetjp_1257_:
{
uint8_t v___x_1260_; lean_object* v___x_1262_; 
v___x_1260_ = 1;
if (v_isShared_1259_ == 0)
{
v___x_1262_ = v___x_1258_;
goto v_reusejp_1261_;
}
else
{
lean_object* v_reuseFailAlloc_1264_; 
v_reuseFailAlloc_1264_ = lean_alloc_ctor(0, 8, 11);
lean_ctor_set(v_reuseFailAlloc_1264_, 0, v_declName_x3f_1239_);
lean_ctor_set(v_reuseFailAlloc_1264_, 1, v_macroStack_1240_);
lean_ctor_set(v_reuseFailAlloc_1264_, 2, v_autoBoundImplicitContext_1243_);
lean_ctor_set(v_reuseFailAlloc_1264_, 3, v_autoBoundImplicitForbidden_1244_);
lean_ctor_set(v_reuseFailAlloc_1264_, 4, v_sectionVars_1245_);
lean_ctor_set(v_reuseFailAlloc_1264_, 5, v_sectionFVars_1246_);
lean_ctor_set(v_reuseFailAlloc_1264_, 6, v_tacSnap_x3f_1252_);
lean_ctor_set(v_reuseFailAlloc_1264_, 7, v_fixedTermElabs_1256_);
lean_ctor_set_uint8(v_reuseFailAlloc_1264_, sizeof(void*)*8, v_mayPostpone_1241_);
lean_ctor_set_uint8(v_reuseFailAlloc_1264_, sizeof(void*)*8 + 1, v_errToSorry_1242_);
lean_ctor_set_uint8(v_reuseFailAlloc_1264_, sizeof(void*)*8 + 2, v_implicitLambda_1247_);
lean_ctor_set_uint8(v_reuseFailAlloc_1264_, sizeof(void*)*8 + 3, v_heedElabAsElim_1248_);
lean_ctor_set_uint8(v_reuseFailAlloc_1264_, sizeof(void*)*8 + 4, v_isNoncomputableSection_1249_);
lean_ctor_set_uint8(v_reuseFailAlloc_1264_, sizeof(void*)*8 + 5, v_isMetaSection_1250_);
lean_ctor_set_uint8(v_reuseFailAlloc_1264_, sizeof(void*)*8 + 7, v_inPattern_1251_);
lean_ctor_set_uint8(v_reuseFailAlloc_1264_, sizeof(void*)*8 + 8, v_saveRecAppSyntax_1253_);
lean_ctor_set_uint8(v_reuseFailAlloc_1264_, sizeof(void*)*8 + 9, v_holesAsSyntheticOpaque_1254_);
lean_ctor_set_uint8(v_reuseFailAlloc_1264_, sizeof(void*)*8 + 10, v_checkDeprecated_1255_);
v___x_1262_ = v_reuseFailAlloc_1264_;
goto v_reusejp_1261_;
}
v_reusejp_1261_:
{
lean_object* v___x_1263_; 
lean_ctor_set_uint8(v___x_1262_, sizeof(void*)*8 + 6, v___x_1260_);
v___x_1263_ = l_Lean_Elab_Term_withAutoBoundImplicit___redArg(v___f_1231_, v___x_1262_, v___y_1233_, v___y_1234_, v___y_1235_, v___y_1236_, v___y_1237_);
lean_dec_ref(v___x_1262_);
return v___x_1263_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__2___boxed(lean_object* v___f_1266_, lean_object* v___y_1267_, lean_object* v___y_1268_, lean_object* v___y_1269_, lean_object* v___y_1270_, lean_object* v___y_1271_, lean_object* v___y_1272_, lean_object* v___y_1273_){
_start:
{
lean_object* v_res_1274_; 
v_res_1274_ = lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__2(v___f_1266_, v___y_1267_, v___y_1268_, v___y_1269_, v___y_1270_, v___y_1271_, v___y_1272_);
lean_dec(v___y_1272_);
lean_dec_ref(v___y_1271_);
lean_dec(v___y_1270_);
lean_dec_ref(v___y_1269_);
lean_dec(v___y_1268_);
return v_res_1274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_getSubproblem(lean_object* v_binder_1275_, lean_object* v_ty_1276_, lean_object* v_a_1277_, lean_object* v_a_1278_, lean_object* v_a_1279_, lean_object* v_a_1280_, lean_object* v_a_1281_, lean_object* v_a_1282_){
_start:
{
lean_object* v___f_1284_; lean_object* v___f_1285_; lean_object* v___x_1286_; 
v___f_1284_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___boxed), 9, 2);
lean_closure_set(v___f_1284_, 0, v_ty_1276_);
lean_closure_set(v___f_1284_, 1, v_binder_1275_);
v___f_1285_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__2___boxed), 8, 1);
lean_closure_set(v___f_1285_, 0, v___f_1284_);
v___x_1286_ = l_Lean_Elab_Term_observing___redArg(v___f_1285_, v_a_1277_, v_a_1278_, v_a_1279_, v_a_1280_, v_a_1281_, v_a_1282_);
if (lean_obj_tag(v___x_1286_) == 0)
{
lean_object* v_a_1287_; lean_object* v___x_1289_; uint8_t v_isShared_1290_; uint8_t v_isSharedCheck_1296_; 
v_a_1287_ = lean_ctor_get(v___x_1286_, 0);
v_isSharedCheck_1296_ = !lean_is_exclusive(v___x_1286_);
if (v_isSharedCheck_1296_ == 0)
{
v___x_1289_ = v___x_1286_;
v_isShared_1290_ = v_isSharedCheck_1296_;
goto v_resetjp_1288_;
}
else
{
lean_inc(v_a_1287_);
lean_dec(v___x_1286_);
v___x_1289_ = lean_box(0);
v_isShared_1290_ = v_isSharedCheck_1296_;
goto v_resetjp_1288_;
}
v_resetjp_1288_:
{
if (lean_obj_tag(v_a_1287_) == 0)
{
lean_object* v_a_1291_; lean_object* v___x_1293_; 
v_a_1291_ = lean_ctor_get(v_a_1287_, 0);
lean_inc(v_a_1291_);
lean_dec_ref_known(v_a_1287_, 2);
if (v_isShared_1290_ == 0)
{
lean_ctor_set(v___x_1289_, 0, v_a_1291_);
v___x_1293_ = v___x_1289_;
goto v_reusejp_1292_;
}
else
{
lean_object* v_reuseFailAlloc_1294_; 
v_reuseFailAlloc_1294_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1294_, 0, v_a_1291_);
v___x_1293_ = v_reuseFailAlloc_1294_;
goto v_reusejp_1292_;
}
v_reusejp_1292_:
{
return v___x_1293_;
}
}
else
{
lean_object* v___x_1295_; 
lean_del_object(v___x_1289_);
v___x_1295_ = l_Lean_Elab_Term_applyResult___redArg(v_a_1287_, v_a_1277_, v_a_1278_, v_a_1279_, v_a_1280_, v_a_1281_, v_a_1282_);
return v___x_1295_;
}
}
}
else
{
lean_object* v_a_1297_; lean_object* v___x_1299_; uint8_t v_isShared_1300_; uint8_t v_isSharedCheck_1304_; 
v_a_1297_ = lean_ctor_get(v___x_1286_, 0);
v_isSharedCheck_1304_ = !lean_is_exclusive(v___x_1286_);
if (v_isSharedCheck_1304_ == 0)
{
v___x_1299_ = v___x_1286_;
v_isShared_1300_ = v_isSharedCheck_1304_;
goto v_resetjp_1298_;
}
else
{
lean_inc(v_a_1297_);
lean_dec(v___x_1286_);
v___x_1299_ = lean_box(0);
v_isShared_1300_ = v_isSharedCheck_1304_;
goto v_resetjp_1298_;
}
v_resetjp_1298_:
{
lean_object* v___x_1302_; 
if (v_isShared_1300_ == 0)
{
v___x_1302_ = v___x_1299_;
goto v_reusejp_1301_;
}
else
{
lean_object* v_reuseFailAlloc_1303_; 
v_reuseFailAlloc_1303_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1303_, 0, v_a_1297_);
v___x_1302_ = v_reuseFailAlloc_1303_;
goto v_reusejp_1301_;
}
v_reusejp_1301_:
{
return v___x_1302_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_getSubproblem___boxed(lean_object* v_binder_1305_, lean_object* v_ty_1306_, lean_object* v_a_1307_, lean_object* v_a_1308_, lean_object* v_a_1309_, lean_object* v_a_1310_, lean_object* v_a_1311_, lean_object* v_a_1312_, lean_object* v_a_1313_){
_start:
{
lean_object* v_res_1314_; 
v_res_1314_ = lp_mathlib_Mathlib_Command_Variable_getSubproblem(v_binder_1305_, v_ty_1306_, v_a_1307_, v_a_1308_, v_a_1309_, v_a_1310_, v_a_1311_, v_a_1312_);
lean_dec(v_a_1312_);
lean_dec_ref(v_a_1311_);
lean_dec(v_a_1310_);
lean_dec_ref(v_a_1309_);
lean_dec(v_a_1308_);
lean_dec_ref(v_a_1307_);
return v_res_1314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4(lean_object* v_cls_1315_, lean_object* v_msg_1316_, lean_object* v___y_1317_, lean_object* v___y_1318_, lean_object* v___y_1319_, lean_object* v___y_1320_, lean_object* v___y_1321_, lean_object* v___y_1322_){
_start:
{
lean_object* v___x_1324_; 
v___x_1324_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg(v_cls_1315_, v_msg_1316_, v___y_1319_, v___y_1320_, v___y_1321_, v___y_1322_);
return v___x_1324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___boxed(lean_object* v_cls_1325_, lean_object* v_msg_1326_, lean_object* v___y_1327_, lean_object* v___y_1328_, lean_object* v___y_1329_, lean_object* v___y_1330_, lean_object* v___y_1331_, lean_object* v___y_1332_, lean_object* v___y_1333_){
_start:
{
lean_object* v_res_1334_; 
v_res_1334_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4(v_cls_1325_, v_msg_1326_, v___y_1327_, v___y_1328_, v___y_1329_, v___y_1330_, v___y_1331_, v___y_1332_);
lean_dec(v___y_1332_);
lean_dec_ref(v___y_1331_);
lean_dec(v___y_1330_);
lean_dec_ref(v___y_1329_);
lean_dec(v___y_1328_);
lean_dec_ref(v___y_1327_);
return v_res_1334_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias_spec__0___redArg___lam__0(lean_object* v_k_1335_, lean_object* v_b_1336_, lean_object* v_c_1337_, lean_object* v___y_1338_, lean_object* v___y_1339_, lean_object* v___y_1340_, lean_object* v___y_1341_){
_start:
{
lean_object* v___x_1343_; 
lean_inc(v___y_1341_);
lean_inc_ref(v___y_1340_);
lean_inc(v___y_1339_);
lean_inc_ref(v___y_1338_);
v___x_1343_ = lean_apply_7(v_k_1335_, v_b_1336_, v_c_1337_, v___y_1338_, v___y_1339_, v___y_1340_, v___y_1341_, lean_box(0));
return v___x_1343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias_spec__0___redArg___lam__0___boxed(lean_object* v_k_1344_, lean_object* v_b_1345_, lean_object* v_c_1346_, lean_object* v___y_1347_, lean_object* v___y_1348_, lean_object* v___y_1349_, lean_object* v___y_1350_, lean_object* v___y_1351_){
_start:
{
lean_object* v_res_1352_; 
v_res_1352_ = lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias_spec__0___redArg___lam__0(v_k_1344_, v_b_1345_, v_c_1346_, v___y_1347_, v___y_1348_, v___y_1349_, v___y_1350_);
lean_dec(v___y_1350_);
lean_dec_ref(v___y_1349_);
lean_dec(v___y_1348_);
lean_dec_ref(v___y_1347_);
return v_res_1352_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias_spec__0___redArg(lean_object* v_type_1353_, lean_object* v_k_1354_, uint8_t v_cleanupAnnotations_1355_, lean_object* v___y_1356_, lean_object* v___y_1357_, lean_object* v___y_1358_, lean_object* v___y_1359_){
_start:
{
lean_object* v___f_1361_; uint8_t v___x_1362_; lean_object* v___x_1363_; lean_object* v___x_1364_; 
v___f_1361_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias_spec__0___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_1361_, 0, v_k_1354_);
v___x_1362_ = 0;
v___x_1363_ = lean_box(0);
v___x_1364_ = l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAuxAux(lean_box(0), v___x_1362_, v___x_1363_, v_type_1353_, v___f_1361_, v_cleanupAnnotations_1355_, v___x_1362_, v___y_1356_, v___y_1357_, v___y_1358_, v___y_1359_);
if (lean_obj_tag(v___x_1364_) == 0)
{
lean_object* v_a_1365_; lean_object* v___x_1367_; uint8_t v_isShared_1368_; uint8_t v_isSharedCheck_1372_; 
v_a_1365_ = lean_ctor_get(v___x_1364_, 0);
v_isSharedCheck_1372_ = !lean_is_exclusive(v___x_1364_);
if (v_isSharedCheck_1372_ == 0)
{
v___x_1367_ = v___x_1364_;
v_isShared_1368_ = v_isSharedCheck_1372_;
goto v_resetjp_1366_;
}
else
{
lean_inc(v_a_1365_);
lean_dec(v___x_1364_);
v___x_1367_ = lean_box(0);
v_isShared_1368_ = v_isSharedCheck_1372_;
goto v_resetjp_1366_;
}
v_resetjp_1366_:
{
lean_object* v___x_1370_; 
if (v_isShared_1368_ == 0)
{
v___x_1370_ = v___x_1367_;
goto v_reusejp_1369_;
}
else
{
lean_object* v_reuseFailAlloc_1371_; 
v_reuseFailAlloc_1371_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1371_, 0, v_a_1365_);
v___x_1370_ = v_reuseFailAlloc_1371_;
goto v_reusejp_1369_;
}
v_reusejp_1369_:
{
return v___x_1370_;
}
}
}
else
{
lean_object* v_a_1373_; lean_object* v___x_1375_; uint8_t v_isShared_1376_; uint8_t v_isSharedCheck_1380_; 
v_a_1373_ = lean_ctor_get(v___x_1364_, 0);
v_isSharedCheck_1380_ = !lean_is_exclusive(v___x_1364_);
if (v_isSharedCheck_1380_ == 0)
{
v___x_1375_ = v___x_1364_;
v_isShared_1376_ = v_isSharedCheck_1380_;
goto v_resetjp_1374_;
}
else
{
lean_inc(v_a_1373_);
lean_dec(v___x_1364_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias_spec__0___redArg___boxed(lean_object* v_type_1381_, lean_object* v_k_1382_, lean_object* v_cleanupAnnotations_1383_, lean_object* v___y_1384_, lean_object* v___y_1385_, lean_object* v___y_1386_, lean_object* v___y_1387_, lean_object* v___y_1388_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_1389_; lean_object* v_res_1390_; 
v_cleanupAnnotations_boxed_1389_ = lean_unbox(v_cleanupAnnotations_1383_);
v_res_1390_ = lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias_spec__0___redArg(v_type_1381_, v_k_1382_, v_cleanupAnnotations_boxed_1389_, v___y_1384_, v___y_1385_, v___y_1386_, v___y_1387_);
lean_dec(v___y_1387_);
lean_dec_ref(v___y_1386_);
lean_dec(v___y_1385_);
lean_dec_ref(v___y_1384_);
return v_res_1390_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias_spec__0(lean_object* v_00_u03b1_1391_, lean_object* v_type_1392_, lean_object* v_k_1393_, uint8_t v_cleanupAnnotations_1394_, lean_object* v___y_1395_, lean_object* v___y_1396_, lean_object* v___y_1397_, lean_object* v___y_1398_){
_start:
{
lean_object* v___x_1400_; 
v___x_1400_ = lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias_spec__0___redArg(v_type_1392_, v_k_1393_, v_cleanupAnnotations_1394_, v___y_1395_, v___y_1396_, v___y_1397_, v___y_1398_);
return v___x_1400_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias_spec__0___boxed(lean_object* v_00_u03b1_1401_, lean_object* v_type_1402_, lean_object* v_k_1403_, lean_object* v_cleanupAnnotations_1404_, lean_object* v___y_1405_, lean_object* v___y_1406_, lean_object* v___y_1407_, lean_object* v___y_1408_, lean_object* v___y_1409_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_1410_; lean_object* v_res_1411_; 
v_cleanupAnnotations_boxed_1410_ = lean_unbox(v_cleanupAnnotations_1404_);
v_res_1411_ = lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias_spec__0(v_00_u03b1_1401_, v_type_1402_, v_k_1403_, v_cleanupAnnotations_boxed_1410_, v___y_1405_, v___y_1406_, v___y_1407_, v___y_1408_);
lean_dec(v___y_1408_);
lean_dec_ref(v___y_1407_);
lean_dec(v___y_1406_);
lean_dec_ref(v___y_1405_);
return v_res_1411_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias___lam__0(lean_object* v_x_1412_, lean_object* v_type_1413_, lean_object* v___y_1414_, lean_object* v___y_1415_, lean_object* v___y_1416_, lean_object* v___y_1417_){
_start:
{
lean_object* v___x_1423_; 
v___x_1423_ = l_Lean_Expr_getAppFn(v_type_1413_);
if (lean_obj_tag(v___x_1423_) == 4)
{
lean_object* v_declName_1424_; lean_object* v___x_1425_; lean_object* v_env_1426_; lean_object* v___x_1427_; uint8_t v___x_1428_; 
v_declName_1424_ = lean_ctor_get(v___x_1423_, 0);
lean_inc(v_declName_1424_);
lean_dec_ref_known(v___x_1423_, 2);
v___x_1425_ = lean_st_ref_get(v___y_1417_);
v_env_1426_ = lean_ctor_get(v___x_1425_, 0);
lean_inc_ref(v_env_1426_);
lean_dec(v___x_1425_);
v___x_1427_ = lp_mathlib_Mathlib_Command_Variable_variableAliasAttr;
v___x_1428_ = l_Lean_TagAttribute_hasTag(v___x_1427_, v_env_1426_, v_declName_1424_);
if (v___x_1428_ == 0)
{
goto v___jp_1419_;
}
else
{
lean_object* v___x_1429_; lean_object* v___x_1430_; 
v___x_1429_ = lean_box(v___x_1428_);
v___x_1430_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1430_, 0, v___x_1429_);
return v___x_1430_;
}
}
else
{
lean_dec_ref(v___x_1423_);
goto v___jp_1419_;
}
v___jp_1419_:
{
uint8_t v___x_1420_; lean_object* v___x_1421_; lean_object* v___x_1422_; 
v___x_1420_ = 0;
v___x_1421_ = lean_box(v___x_1420_);
v___x_1422_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1422_, 0, v___x_1421_);
return v___x_1422_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias___lam__0___boxed(lean_object* v_x_1431_, lean_object* v_type_1432_, lean_object* v___y_1433_, lean_object* v___y_1434_, lean_object* v___y_1435_, lean_object* v___y_1436_, lean_object* v___y_1437_){
_start:
{
lean_object* v_res_1438_; 
v_res_1438_ = lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias___lam__0(v_x_1431_, v_type_1432_, v___y_1433_, v___y_1434_, v___y_1435_, v___y_1436_);
lean_dec(v___y_1436_);
lean_dec_ref(v___y_1435_);
lean_dec(v___y_1434_);
lean_dec_ref(v___y_1433_);
lean_dec_ref(v_type_1432_);
lean_dec_ref(v_x_1431_);
return v_res_1438_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias(lean_object* v_type_1440_, lean_object* v_a_1441_, lean_object* v_a_1442_, lean_object* v_a_1443_, lean_object* v_a_1444_){
_start:
{
lean_object* v___f_1446_; uint8_t v___x_1447_; lean_object* v___x_1448_; 
v___f_1446_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias___closed__0));
v___x_1447_ = 0;
v___x_1448_ = lp_mathlib_Lean_Meta_forallTelescope___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias_spec__0___redArg(v_type_1440_, v___f_1446_, v___x_1447_, v_a_1441_, v_a_1442_, v_a_1443_, v_a_1444_);
return v___x_1448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias___boxed(lean_object* v_type_1449_, lean_object* v_a_1450_, lean_object* v_a_1451_, lean_object* v_a_1452_, lean_object* v_a_1453_, lean_object* v_a_1454_){
_start:
{
lean_object* v_res_1455_; 
v_res_1455_ = lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias(v_type_1449_, v_a_1450_, v_a_1451_, v_a_1452_, v_a_1453_);
lean_dec(v_a_1453_);
lean_dec_ref(v_a_1452_);
lean_dec(v_a_1451_);
lean_dec_ref(v_a_1450_);
return v_res_1455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Command_Variable_completeBinders_x27_spec__3___redArg(lean_object* v_lctx_1456_, lean_object* v_localInsts_1457_, lean_object* v_x_1458_, lean_object* v___y_1459_, lean_object* v___y_1460_, lean_object* v___y_1461_, lean_object* v___y_1462_, lean_object* v___y_1463_, lean_object* v___y_1464_){
_start:
{
lean_object* v___f_1466_; lean_object* v___x_1467_; 
lean_inc(v___y_1460_);
lean_inc_ref(v___y_1459_);
v___f_1466_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Command_Variable_getSubproblem_spec__2___redArg___lam__0___boxed), 8, 3);
lean_closure_set(v___f_1466_, 0, v_x_1458_);
lean_closure_set(v___f_1466_, 1, v___y_1459_);
lean_closure_set(v___f_1466_, 2, v___y_1460_);
v___x_1467_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_box(0), v_lctx_1456_, v_localInsts_1457_, v___f_1466_, v___y_1461_, v___y_1462_, v___y_1463_, v___y_1464_);
if (lean_obj_tag(v___x_1467_) == 0)
{
return v___x_1467_;
}
else
{
lean_object* v_a_1468_; lean_object* v___x_1470_; uint8_t v_isShared_1471_; uint8_t v_isSharedCheck_1475_; 
v_a_1468_ = lean_ctor_get(v___x_1467_, 0);
v_isSharedCheck_1475_ = !lean_is_exclusive(v___x_1467_);
if (v_isSharedCheck_1475_ == 0)
{
v___x_1470_ = v___x_1467_;
v_isShared_1471_ = v_isSharedCheck_1475_;
goto v_resetjp_1469_;
}
else
{
lean_inc(v_a_1468_);
lean_dec(v___x_1467_);
v___x_1470_ = lean_box(0);
v_isShared_1471_ = v_isSharedCheck_1475_;
goto v_resetjp_1469_;
}
v_resetjp_1469_:
{
lean_object* v___x_1473_; 
if (v_isShared_1471_ == 0)
{
v___x_1473_ = v___x_1470_;
goto v_reusejp_1472_;
}
else
{
lean_object* v_reuseFailAlloc_1474_; 
v_reuseFailAlloc_1474_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1474_, 0, v_a_1468_);
v___x_1473_ = v_reuseFailAlloc_1474_;
goto v_reusejp_1472_;
}
v_reusejp_1472_:
{
return v___x_1473_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Command_Variable_completeBinders_x27_spec__3___redArg___boxed(lean_object* v_lctx_1476_, lean_object* v_localInsts_1477_, lean_object* v_x_1478_, lean_object* v___y_1479_, lean_object* v___y_1480_, lean_object* v___y_1481_, lean_object* v___y_1482_, lean_object* v___y_1483_, lean_object* v___y_1484_, lean_object* v___y_1485_){
_start:
{
lean_object* v_res_1486_; 
v_res_1486_ = lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Command_Variable_completeBinders_x27_spec__3___redArg(v_lctx_1476_, v_localInsts_1477_, v_x_1478_, v___y_1479_, v___y_1480_, v___y_1481_, v___y_1482_, v___y_1483_, v___y_1484_);
lean_dec(v___y_1484_);
lean_dec_ref(v___y_1483_);
lean_dec(v___y_1482_);
lean_dec_ref(v___y_1481_);
lean_dec(v___y_1480_);
lean_dec_ref(v___y_1479_);
return v_res_1486_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Command_Variable_completeBinders_x27_spec__3(lean_object* v_00_u03b1_1487_, lean_object* v_lctx_1488_, lean_object* v_localInsts_1489_, lean_object* v_x_1490_, lean_object* v___y_1491_, lean_object* v___y_1492_, lean_object* v___y_1493_, lean_object* v___y_1494_, lean_object* v___y_1495_, lean_object* v___y_1496_){
_start:
{
lean_object* v___x_1498_; 
v___x_1498_ = lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Command_Variable_completeBinders_x27_spec__3___redArg(v_lctx_1488_, v_localInsts_1489_, v_x_1490_, v___y_1491_, v___y_1492_, v___y_1493_, v___y_1494_, v___y_1495_, v___y_1496_);
return v___x_1498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Command_Variable_completeBinders_x27_spec__3___boxed(lean_object* v_00_u03b1_1499_, lean_object* v_lctx_1500_, lean_object* v_localInsts_1501_, lean_object* v_x_1502_, lean_object* v___y_1503_, lean_object* v___y_1504_, lean_object* v___y_1505_, lean_object* v___y_1506_, lean_object* v___y_1507_, lean_object* v___y_1508_, lean_object* v___y_1509_){
_start:
{
lean_object* v_res_1510_; 
v_res_1510_ = lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Command_Variable_completeBinders_x27_spec__3(v_00_u03b1_1499_, v_lctx_1500_, v_localInsts_1501_, v_x_1502_, v___y_1503_, v___y_1504_, v___y_1505_, v___y_1506_, v___y_1507_, v___y_1508_);
lean_dec(v___y_1508_);
lean_dec_ref(v___y_1507_);
lean_dec(v___y_1506_);
lean_dec_ref(v___y_1505_);
lean_dec(v___y_1504_);
lean_dec_ref(v___y_1503_);
return v_res_1510_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Mathlib_Command_Variable_completeBinders_x27_spec__7(lean_object* v_opts_1511_, lean_object* v_opt_1512_){
_start:
{
lean_object* v_name_1513_; lean_object* v_defValue_1514_; lean_object* v_map_1515_; lean_object* v___x_1516_; 
v_name_1513_ = lean_ctor_get(v_opt_1512_, 0);
v_defValue_1514_ = lean_ctor_get(v_opt_1512_, 1);
v_map_1515_ = lean_ctor_get(v_opts_1511_, 0);
v___x_1516_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1515_, v_name_1513_);
if (lean_obj_tag(v___x_1516_) == 0)
{
lean_inc(v_defValue_1514_);
return v_defValue_1514_;
}
else
{
lean_object* v_val_1517_; 
v_val_1517_ = lean_ctor_get(v___x_1516_, 0);
lean_inc(v_val_1517_);
lean_dec_ref_known(v___x_1516_, 1);
if (lean_obj_tag(v_val_1517_) == 3)
{
lean_object* v_v_1518_; 
v_v_1518_ = lean_ctor_get(v_val_1517_, 0);
lean_inc(v_v_1518_);
lean_dec_ref_known(v_val_1517_, 1);
return v_v_1518_;
}
else
{
lean_dec(v_val_1517_);
lean_inc(v_defValue_1514_);
return v_defValue_1514_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Mathlib_Command_Variable_completeBinders_x27_spec__7___boxed(lean_object* v_opts_1519_, lean_object* v_opt_1520_){
_start:
{
lean_object* v_res_1521_; 
v_res_1521_ = lp_mathlib_Lean_Option_get___at___00Mathlib_Command_Variable_completeBinders_x27_spec__7(v_opts_1519_, v_opt_1520_);
lean_dec_ref(v_opt_1520_);
lean_dec_ref(v_opts_1519_);
return v_res_1521_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Mathlib_Command_Variable_completeBinders_x27_spec__8(lean_object* v_msg_1522_){
_start:
{
lean_object* v___x_1523_; lean_object* v___x_1524_; 
v___x_1523_ = lean_box(0);
v___x_1524_ = lean_panic_fn_borrowed(v___x_1523_, v_msg_1522_);
return v___x_1524_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__0(lean_object* v_cls_1525_, lean_object* v___y_1526_, lean_object* v___y_1527_, lean_object* v___y_1528_, lean_object* v___y_1529_, lean_object* v___y_1530_, lean_object* v___y_1531_){
_start:
{
lean_object* v_options_1533_; uint8_t v_hasTrace_1534_; 
v_options_1533_ = lean_ctor_get(v___y_1530_, 2);
v_hasTrace_1534_ = lean_ctor_get_uint8(v_options_1533_, sizeof(void*)*1);
if (v_hasTrace_1534_ == 0)
{
lean_object* v___x_1535_; lean_object* v___x_1536_; 
lean_dec(v_cls_1525_);
v___x_1535_ = lean_box(v_hasTrace_1534_);
v___x_1536_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1536_, 0, v___x_1535_);
return v___x_1536_;
}
else
{
lean_object* v_inheritedTraceOptions_1537_; lean_object* v___x_1538_; lean_object* v___x_1539_; uint8_t v___x_1540_; lean_object* v___x_1541_; lean_object* v___x_1542_; 
v_inheritedTraceOptions_1537_ = lean_ctor_get(v___y_1530_, 13);
v___x_1538_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__7));
v___x_1539_ = l_Lean_Name_append(v___x_1538_, v_cls_1525_);
v___x_1540_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_1537_, v_options_1533_, v___x_1539_);
lean_dec(v___x_1539_);
v___x_1541_ = lean_box(v___x_1540_);
v___x_1542_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1542_, 0, v___x_1541_);
return v___x_1542_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__0___boxed(lean_object* v_cls_1543_, lean_object* v___y_1544_, lean_object* v___y_1545_, lean_object* v___y_1546_, lean_object* v___y_1547_, lean_object* v___y_1548_, lean_object* v___y_1549_, lean_object* v___y_1550_){
_start:
{
lean_object* v_res_1551_; 
v_res_1551_ = lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__0(v_cls_1543_, v___y_1544_, v___y_1545_, v___y_1546_, v___y_1547_, v___y_1548_, v___y_1549_);
lean_dec(v___y_1549_);
lean_dec_ref(v___y_1548_);
lean_dec(v___y_1547_);
lean_dec_ref(v___y_1546_);
lean_dec(v___y_1545_);
lean_dec_ref(v___y_1544_);
return v_res_1551_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__1(lean_object* v_a_1552_, lean_object* v___x_1553_, lean_object* v___y_1554_, lean_object* v___y_1555_, lean_object* v___y_1556_, lean_object* v___y_1557_, lean_object* v___y_1558_, lean_object* v___y_1559_){
_start:
{
lean_object* v___x_1561_; 
v___x_1561_ = l_Lean_Meta_trySynthInstance(v_a_1552_, v___x_1553_, v___y_1556_, v___y_1557_, v___y_1558_, v___y_1559_);
return v___x_1561_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__1___boxed(lean_object* v_a_1562_, lean_object* v___x_1563_, lean_object* v___y_1564_, lean_object* v___y_1565_, lean_object* v___y_1566_, lean_object* v___y_1567_, lean_object* v___y_1568_, lean_object* v___y_1569_, lean_object* v___y_1570_){
_start:
{
lean_object* v_res_1571_; 
v_res_1571_ = lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__1(v_a_1562_, v___x_1563_, v___y_1564_, v___y_1565_, v___y_1566_, v___y_1567_, v___y_1568_, v___y_1569_);
lean_dec(v___y_1569_);
lean_dec_ref(v___y_1568_);
lean_dec(v___y_1567_);
lean_dec_ref(v___y_1566_);
lean_dec(v___y_1565_);
lean_dec_ref(v___y_1564_);
return v_res_1571_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___lam__0(uint8_t v___y_1578_, uint8_t v_suppressElabErrors_1579_, lean_object* v_x_1580_){
_start:
{
if (lean_obj_tag(v_x_1580_) == 1)
{
lean_object* v_pre_1581_; 
v_pre_1581_ = lean_ctor_get(v_x_1580_, 0);
switch(lean_obj_tag(v_pre_1581_))
{
case 1:
{
lean_object* v_pre_1582_; 
v_pre_1582_ = lean_ctor_get(v_pre_1581_, 0);
switch(lean_obj_tag(v_pre_1582_))
{
case 0:
{
lean_object* v_str_1583_; lean_object* v_str_1584_; lean_object* v___x_1585_; uint8_t v___x_1586_; 
v_str_1583_ = lean_ctor_get(v_x_1580_, 1);
v_str_1584_ = lean_ctor_get(v_pre_1581_, 1);
v___x_1585_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___lam__0___closed__0));
v___x_1586_ = lean_string_dec_eq(v_str_1584_, v___x_1585_);
if (v___x_1586_ == 0)
{
lean_object* v___x_1587_; uint8_t v___x_1588_; 
v___x_1587_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__6_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_));
v___x_1588_ = lean_string_dec_eq(v_str_1584_, v___x_1587_);
if (v___x_1588_ == 0)
{
return v___y_1578_;
}
else
{
lean_object* v___x_1589_; uint8_t v___x_1590_; 
v___x_1589_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___lam__0___closed__1));
v___x_1590_ = lean_string_dec_eq(v_str_1583_, v___x_1589_);
if (v___x_1590_ == 0)
{
return v___y_1578_;
}
else
{
return v_suppressElabErrors_1579_;
}
}
}
else
{
lean_object* v___x_1591_; uint8_t v___x_1592_; 
v___x_1591_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___lam__0___closed__2));
v___x_1592_ = lean_string_dec_eq(v_str_1583_, v___x_1591_);
if (v___x_1592_ == 0)
{
return v___y_1578_;
}
else
{
return v_suppressElabErrors_1579_;
}
}
}
case 1:
{
lean_object* v_pre_1593_; 
v_pre_1593_ = lean_ctor_get(v_pre_1582_, 0);
if (lean_obj_tag(v_pre_1593_) == 0)
{
lean_object* v_str_1594_; lean_object* v_str_1595_; lean_object* v_str_1596_; lean_object* v___x_1597_; uint8_t v___x_1598_; 
v_str_1594_ = lean_ctor_get(v_x_1580_, 1);
v_str_1595_ = lean_ctor_get(v_pre_1581_, 1);
v_str_1596_ = lean_ctor_get(v_pre_1582_, 1);
v___x_1597_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___lam__0___closed__3));
v___x_1598_ = lean_string_dec_eq(v_str_1596_, v___x_1597_);
if (v___x_1598_ == 0)
{
return v___y_1578_;
}
else
{
lean_object* v___x_1599_; uint8_t v___x_1600_; 
v___x_1599_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___lam__0___closed__4));
v___x_1600_ = lean_string_dec_eq(v_str_1595_, v___x_1599_);
if (v___x_1600_ == 0)
{
return v___y_1578_;
}
else
{
lean_object* v___x_1601_; uint8_t v___x_1602_; 
v___x_1601_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___lam__0___closed__5));
v___x_1602_ = lean_string_dec_eq(v_str_1594_, v___x_1601_);
if (v___x_1602_ == 0)
{
return v___y_1578_;
}
else
{
return v_suppressElabErrors_1579_;
}
}
}
}
else
{
return v___y_1578_;
}
}
default: 
{
return v___y_1578_;
}
}
}
case 0:
{
lean_object* v_str_1603_; lean_object* v___x_1604_; uint8_t v___x_1605_; 
v_str_1603_ = lean_ctor_get(v_x_1580_, 1);
v___x_1604_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__6));
v___x_1605_ = lean_string_dec_eq(v_str_1603_, v___x_1604_);
if (v___x_1605_ == 0)
{
return v___y_1578_;
}
else
{
return v_suppressElabErrors_1579_;
}
}
default: 
{
return v___y_1578_;
}
}
}
else
{
return v___y_1578_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___lam__0___boxed(lean_object* v___y_1606_, lean_object* v_suppressElabErrors_1607_, lean_object* v_x_1608_){
_start:
{
uint8_t v___y_31199__boxed_1609_; uint8_t v_suppressElabErrors_boxed_1610_; uint8_t v_res_1611_; lean_object* v_r_1612_; 
v___y_31199__boxed_1609_ = lean_unbox(v___y_1606_);
v_suppressElabErrors_boxed_1610_ = lean_unbox(v_suppressElabErrors_1607_);
v_res_1611_ = lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___lam__0(v___y_31199__boxed_1609_, v_suppressElabErrors_boxed_1610_, v_x_1608_);
lean_dec(v_x_1608_);
v_r_1612_ = lean_box(v_res_1611_);
return v_r_1612_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg(lean_object* v_ref_1613_, lean_object* v_msgData_1614_, uint8_t v_severity_1615_, uint8_t v_isSilent_1616_, lean_object* v___y_1617_, lean_object* v___y_1618_, lean_object* v___y_1619_, lean_object* v___y_1620_){
_start:
{
lean_object* v___y_1623_; uint8_t v___y_1624_; lean_object* v___y_1625_; uint8_t v___y_1626_; lean_object* v___y_1627_; lean_object* v___y_1628_; lean_object* v___y_1629_; lean_object* v___y_1630_; lean_object* v___y_1631_; lean_object* v___y_1659_; lean_object* v___y_1660_; lean_object* v___y_1661_; uint8_t v___y_1662_; uint8_t v___y_1663_; lean_object* v___y_1664_; uint8_t v___y_1665_; lean_object* v___y_1666_; lean_object* v___y_1684_; lean_object* v___y_1685_; lean_object* v___y_1686_; uint8_t v___y_1687_; uint8_t v___y_1688_; lean_object* v___y_1689_; uint8_t v___y_1690_; lean_object* v___y_1691_; lean_object* v___y_1695_; lean_object* v___y_1696_; lean_object* v___y_1697_; uint8_t v___y_1698_; lean_object* v___y_1699_; uint8_t v___y_1700_; uint8_t v___y_1701_; uint8_t v___x_1706_; lean_object* v___y_1708_; lean_object* v___y_1709_; lean_object* v___y_1710_; uint8_t v___y_1711_; lean_object* v___y_1712_; uint8_t v___y_1713_; uint8_t v___y_1714_; uint8_t v___y_1716_; uint8_t v___x_1731_; 
v___x_1706_ = 2;
v___x_1731_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1615_, v___x_1706_);
if (v___x_1731_ == 0)
{
v___y_1716_ = v___x_1731_;
goto v___jp_1715_;
}
else
{
uint8_t v___x_1732_; 
lean_inc_ref(v_msgData_1614_);
v___x_1732_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_1614_);
v___y_1716_ = v___x_1732_;
goto v___jp_1715_;
}
v___jp_1622_:
{
lean_object* v___x_1632_; lean_object* v_currNamespace_1633_; lean_object* v_openDecls_1634_; lean_object* v_env_1635_; lean_object* v_nextMacroScope_1636_; lean_object* v_ngen_1637_; lean_object* v_auxDeclNGen_1638_; lean_object* v_traceState_1639_; lean_object* v_cache_1640_; lean_object* v_messages_1641_; lean_object* v_infoState_1642_; lean_object* v_snapshotTasks_1643_; lean_object* v___x_1645_; uint8_t v_isShared_1646_; uint8_t v_isSharedCheck_1657_; 
v___x_1632_ = lean_st_ref_take(v___y_1631_);
v_currNamespace_1633_ = lean_ctor_get(v___y_1630_, 6);
v_openDecls_1634_ = lean_ctor_get(v___y_1630_, 7);
v_env_1635_ = lean_ctor_get(v___x_1632_, 0);
v_nextMacroScope_1636_ = lean_ctor_get(v___x_1632_, 1);
v_ngen_1637_ = lean_ctor_get(v___x_1632_, 2);
v_auxDeclNGen_1638_ = lean_ctor_get(v___x_1632_, 3);
v_traceState_1639_ = lean_ctor_get(v___x_1632_, 4);
v_cache_1640_ = lean_ctor_get(v___x_1632_, 5);
v_messages_1641_ = lean_ctor_get(v___x_1632_, 6);
v_infoState_1642_ = lean_ctor_get(v___x_1632_, 7);
v_snapshotTasks_1643_ = lean_ctor_get(v___x_1632_, 8);
v_isSharedCheck_1657_ = !lean_is_exclusive(v___x_1632_);
if (v_isSharedCheck_1657_ == 0)
{
v___x_1645_ = v___x_1632_;
v_isShared_1646_ = v_isSharedCheck_1657_;
goto v_resetjp_1644_;
}
else
{
lean_inc(v_snapshotTasks_1643_);
lean_inc(v_infoState_1642_);
lean_inc(v_messages_1641_);
lean_inc(v_cache_1640_);
lean_inc(v_traceState_1639_);
lean_inc(v_auxDeclNGen_1638_);
lean_inc(v_ngen_1637_);
lean_inc(v_nextMacroScope_1636_);
lean_inc(v_env_1635_);
lean_dec(v___x_1632_);
v___x_1645_ = lean_box(0);
v_isShared_1646_ = v_isSharedCheck_1657_;
goto v_resetjp_1644_;
}
v_resetjp_1644_:
{
lean_object* v___x_1647_; lean_object* v___x_1648_; lean_object* v___x_1649_; lean_object* v___x_1650_; lean_object* v___x_1652_; 
lean_inc(v_openDecls_1634_);
lean_inc(v_currNamespace_1633_);
v___x_1647_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1647_, 0, v_currNamespace_1633_);
lean_ctor_set(v___x_1647_, 1, v_openDecls_1634_);
v___x_1648_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1648_, 0, v___x_1647_);
lean_ctor_set(v___x_1648_, 1, v___y_1623_);
lean_inc_ref(v___y_1625_);
lean_inc_ref(v___y_1628_);
v___x_1649_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_1649_, 0, v___y_1628_);
lean_ctor_set(v___x_1649_, 1, v___y_1627_);
lean_ctor_set(v___x_1649_, 2, v___y_1629_);
lean_ctor_set(v___x_1649_, 3, v___y_1625_);
lean_ctor_set(v___x_1649_, 4, v___x_1648_);
lean_ctor_set_uint8(v___x_1649_, sizeof(void*)*5, v___y_1626_);
lean_ctor_set_uint8(v___x_1649_, sizeof(void*)*5 + 1, v___y_1624_);
lean_ctor_set_uint8(v___x_1649_, sizeof(void*)*5 + 2, v_isSilent_1616_);
v___x_1650_ = l_Lean_MessageLog_add(v___x_1649_, v_messages_1641_);
if (v_isShared_1646_ == 0)
{
lean_ctor_set(v___x_1645_, 6, v___x_1650_);
v___x_1652_ = v___x_1645_;
goto v_reusejp_1651_;
}
else
{
lean_object* v_reuseFailAlloc_1656_; 
v_reuseFailAlloc_1656_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1656_, 0, v_env_1635_);
lean_ctor_set(v_reuseFailAlloc_1656_, 1, v_nextMacroScope_1636_);
lean_ctor_set(v_reuseFailAlloc_1656_, 2, v_ngen_1637_);
lean_ctor_set(v_reuseFailAlloc_1656_, 3, v_auxDeclNGen_1638_);
lean_ctor_set(v_reuseFailAlloc_1656_, 4, v_traceState_1639_);
lean_ctor_set(v_reuseFailAlloc_1656_, 5, v_cache_1640_);
lean_ctor_set(v_reuseFailAlloc_1656_, 6, v___x_1650_);
lean_ctor_set(v_reuseFailAlloc_1656_, 7, v_infoState_1642_);
lean_ctor_set(v_reuseFailAlloc_1656_, 8, v_snapshotTasks_1643_);
v___x_1652_ = v_reuseFailAlloc_1656_;
goto v_reusejp_1651_;
}
v_reusejp_1651_:
{
lean_object* v___x_1653_; lean_object* v___x_1654_; lean_object* v___x_1655_; 
v___x_1653_ = lean_st_ref_set(v___y_1631_, v___x_1652_);
v___x_1654_ = lean_box(0);
v___x_1655_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1655_, 0, v___x_1654_);
return v___x_1655_;
}
}
}
v___jp_1658_:
{
lean_object* v___x_1667_; lean_object* v___x_1668_; lean_object* v_a_1669_; lean_object* v___x_1671_; uint8_t v_isShared_1672_; uint8_t v_isSharedCheck_1682_; 
v___x_1667_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_1614_);
v___x_1668_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__3(v___x_1667_, v___y_1617_, v___y_1618_, v___y_1619_, v___y_1620_);
v_a_1669_ = lean_ctor_get(v___x_1668_, 0);
v_isSharedCheck_1682_ = !lean_is_exclusive(v___x_1668_);
if (v_isSharedCheck_1682_ == 0)
{
v___x_1671_ = v___x_1668_;
v_isShared_1672_ = v_isSharedCheck_1682_;
goto v_resetjp_1670_;
}
else
{
lean_inc(v_a_1669_);
lean_dec(v___x_1668_);
v___x_1671_ = lean_box(0);
v_isShared_1672_ = v_isSharedCheck_1682_;
goto v_resetjp_1670_;
}
v_resetjp_1670_:
{
lean_object* v___x_1673_; lean_object* v___x_1674_; lean_object* v___x_1675_; lean_object* v___x_1676_; 
lean_inc_ref_n(v___y_1661_, 2);
v___x_1673_ = l_Lean_FileMap_toPosition(v___y_1661_, v___y_1660_);
lean_dec(v___y_1660_);
v___x_1674_ = l_Lean_FileMap_toPosition(v___y_1661_, v___y_1666_);
lean_dec(v___y_1666_);
v___x_1675_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1675_, 0, v___x_1674_);
v___x_1676_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg___closed__1));
if (v___y_1665_ == 0)
{
lean_del_object(v___x_1671_);
lean_dec_ref(v___y_1659_);
v___y_1623_ = v_a_1669_;
v___y_1624_ = v___y_1662_;
v___y_1625_ = v___x_1676_;
v___y_1626_ = v___y_1663_;
v___y_1627_ = v___x_1673_;
v___y_1628_ = v___y_1664_;
v___y_1629_ = v___x_1675_;
v___y_1630_ = v___y_1619_;
v___y_1631_ = v___y_1620_;
goto v___jp_1622_;
}
else
{
uint8_t v___x_1677_; 
lean_inc(v_a_1669_);
v___x_1677_ = l_Lean_MessageData_hasTag(v___y_1659_, v_a_1669_);
if (v___x_1677_ == 0)
{
lean_object* v___x_1678_; lean_object* v___x_1680_; 
lean_dec_ref_known(v___x_1675_, 1);
lean_dec_ref(v___x_1673_);
lean_dec(v_a_1669_);
v___x_1678_ = lean_box(0);
if (v_isShared_1672_ == 0)
{
lean_ctor_set(v___x_1671_, 0, v___x_1678_);
v___x_1680_ = v___x_1671_;
goto v_reusejp_1679_;
}
else
{
lean_object* v_reuseFailAlloc_1681_; 
v_reuseFailAlloc_1681_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1681_, 0, v___x_1678_);
v___x_1680_ = v_reuseFailAlloc_1681_;
goto v_reusejp_1679_;
}
v_reusejp_1679_:
{
return v___x_1680_;
}
}
else
{
lean_del_object(v___x_1671_);
v___y_1623_ = v_a_1669_;
v___y_1624_ = v___y_1662_;
v___y_1625_ = v___x_1676_;
v___y_1626_ = v___y_1663_;
v___y_1627_ = v___x_1673_;
v___y_1628_ = v___y_1664_;
v___y_1629_ = v___x_1675_;
v___y_1630_ = v___y_1619_;
v___y_1631_ = v___y_1620_;
goto v___jp_1622_;
}
}
}
}
v___jp_1683_:
{
lean_object* v___x_1692_; 
v___x_1692_ = l_Lean_Syntax_getTailPos_x3f(v___y_1685_, v___y_1688_);
lean_dec(v___y_1685_);
if (lean_obj_tag(v___x_1692_) == 0)
{
lean_inc(v___y_1691_);
v___y_1659_ = v___y_1684_;
v___y_1660_ = v___y_1691_;
v___y_1661_ = v___y_1686_;
v___y_1662_ = v___y_1687_;
v___y_1663_ = v___y_1688_;
v___y_1664_ = v___y_1689_;
v___y_1665_ = v___y_1690_;
v___y_1666_ = v___y_1691_;
goto v___jp_1658_;
}
else
{
lean_object* v_val_1693_; 
v_val_1693_ = lean_ctor_get(v___x_1692_, 0);
lean_inc(v_val_1693_);
lean_dec_ref_known(v___x_1692_, 1);
v___y_1659_ = v___y_1684_;
v___y_1660_ = v___y_1691_;
v___y_1661_ = v___y_1686_;
v___y_1662_ = v___y_1687_;
v___y_1663_ = v___y_1688_;
v___y_1664_ = v___y_1689_;
v___y_1665_ = v___y_1690_;
v___y_1666_ = v_val_1693_;
goto v___jp_1658_;
}
}
v___jp_1694_:
{
lean_object* v_ref_1702_; lean_object* v___x_1703_; 
v_ref_1702_ = l_Lean_replaceRef(v_ref_1613_, v___y_1696_);
v___x_1703_ = l_Lean_Syntax_getPos_x3f(v_ref_1702_, v___y_1698_);
if (lean_obj_tag(v___x_1703_) == 0)
{
lean_object* v___x_1704_; 
v___x_1704_ = lean_unsigned_to_nat(0u);
v___y_1684_ = v___y_1695_;
v___y_1685_ = v_ref_1702_;
v___y_1686_ = v___y_1697_;
v___y_1687_ = v___y_1701_;
v___y_1688_ = v___y_1698_;
v___y_1689_ = v___y_1699_;
v___y_1690_ = v___y_1700_;
v___y_1691_ = v___x_1704_;
goto v___jp_1683_;
}
else
{
lean_object* v_val_1705_; 
v_val_1705_ = lean_ctor_get(v___x_1703_, 0);
lean_inc(v_val_1705_);
lean_dec_ref_known(v___x_1703_, 1);
v___y_1684_ = v___y_1695_;
v___y_1685_ = v_ref_1702_;
v___y_1686_ = v___y_1697_;
v___y_1687_ = v___y_1701_;
v___y_1688_ = v___y_1698_;
v___y_1689_ = v___y_1699_;
v___y_1690_ = v___y_1700_;
v___y_1691_ = v_val_1705_;
goto v___jp_1683_;
}
}
v___jp_1707_:
{
if (v___y_1714_ == 0)
{
v___y_1695_ = v___y_1712_;
v___y_1696_ = v___y_1708_;
v___y_1697_ = v___y_1709_;
v___y_1698_ = v___y_1713_;
v___y_1699_ = v___y_1710_;
v___y_1700_ = v___y_1711_;
v___y_1701_ = v_severity_1615_;
goto v___jp_1694_;
}
else
{
v___y_1695_ = v___y_1712_;
v___y_1696_ = v___y_1708_;
v___y_1697_ = v___y_1709_;
v___y_1698_ = v___y_1713_;
v___y_1699_ = v___y_1710_;
v___y_1700_ = v___y_1711_;
v___y_1701_ = v___x_1706_;
goto v___jp_1694_;
}
}
v___jp_1715_:
{
if (v___y_1716_ == 0)
{
lean_object* v_fileName_1717_; lean_object* v_fileMap_1718_; lean_object* v_options_1719_; lean_object* v_ref_1720_; uint8_t v_suppressElabErrors_1721_; lean_object* v___x_1722_; lean_object* v___x_1723_; lean_object* v___f_1724_; uint8_t v___x_1725_; uint8_t v___x_1726_; 
v_fileName_1717_ = lean_ctor_get(v___y_1619_, 0);
v_fileMap_1718_ = lean_ctor_get(v___y_1619_, 1);
v_options_1719_ = lean_ctor_get(v___y_1619_, 2);
v_ref_1720_ = lean_ctor_get(v___y_1619_, 5);
v_suppressElabErrors_1721_ = lean_ctor_get_uint8(v___y_1619_, sizeof(void*)*14 + 1);
v___x_1722_ = lean_box(v___y_1716_);
v___x_1723_ = lean_box(v_suppressElabErrors_1721_);
v___f_1724_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1724_, 0, v___x_1722_);
lean_closure_set(v___f_1724_, 1, v___x_1723_);
v___x_1725_ = 1;
v___x_1726_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1615_, v___x_1725_);
if (v___x_1726_ == 0)
{
v___y_1708_ = v_ref_1720_;
v___y_1709_ = v_fileMap_1718_;
v___y_1710_ = v_fileName_1717_;
v___y_1711_ = v_suppressElabErrors_1721_;
v___y_1712_ = v___f_1724_;
v___y_1713_ = v___y_1716_;
v___y_1714_ = v___x_1726_;
goto v___jp_1707_;
}
else
{
lean_object* v___x_1727_; uint8_t v___x_1728_; 
v___x_1727_ = l_Lean_warningAsError;
v___x_1728_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__5(v_options_1719_, v___x_1727_);
v___y_1708_ = v_ref_1720_;
v___y_1709_ = v_fileMap_1718_;
v___y_1710_ = v_fileName_1717_;
v___y_1711_ = v_suppressElabErrors_1721_;
v___y_1712_ = v___f_1724_;
v___y_1713_ = v___y_1716_;
v___y_1714_ = v___x_1728_;
goto v___jp_1707_;
}
}
else
{
lean_object* v___x_1729_; lean_object* v___x_1730_; 
lean_dec_ref(v_msgData_1614_);
v___x_1729_ = lean_box(0);
v___x_1730_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1730_, 0, v___x_1729_);
return v___x_1730_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg___boxed(lean_object* v_ref_1733_, lean_object* v_msgData_1734_, lean_object* v_severity_1735_, lean_object* v_isSilent_1736_, lean_object* v___y_1737_, lean_object* v___y_1738_, lean_object* v___y_1739_, lean_object* v___y_1740_, lean_object* v___y_1741_){
_start:
{
uint8_t v_severity_boxed_1742_; uint8_t v_isSilent_boxed_1743_; lean_object* v_res_1744_; 
v_severity_boxed_1742_ = lean_unbox(v_severity_1735_);
v_isSilent_boxed_1743_ = lean_unbox(v_isSilent_1736_);
v_res_1744_ = lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg(v_ref_1733_, v_msgData_1734_, v_severity_boxed_1742_, v_isSilent_boxed_1743_, v___y_1737_, v___y_1738_, v___y_1739_, v___y_1740_);
lean_dec(v___y_1740_);
lean_dec_ref(v___y_1739_);
lean_dec(v___y_1738_);
lean_dec_ref(v___y_1737_);
lean_dec(v_ref_1733_);
return v_res_1744_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__2(lean_object* v_ref_1745_, lean_object* v_msgData_1746_, lean_object* v___y_1747_, lean_object* v___y_1748_, lean_object* v___y_1749_, lean_object* v___y_1750_, lean_object* v___y_1751_, lean_object* v___y_1752_){
_start:
{
uint8_t v___x_1754_; uint8_t v___x_1755_; lean_object* v___x_1756_; 
v___x_1754_ = 1;
v___x_1755_ = 0;
v___x_1756_ = lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg(v_ref_1745_, v_msgData_1746_, v___x_1754_, v___x_1755_, v___y_1749_, v___y_1750_, v___y_1751_, v___y_1752_);
return v___x_1756_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarningAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__2___boxed(lean_object* v_ref_1757_, lean_object* v_msgData_1758_, lean_object* v___y_1759_, lean_object* v___y_1760_, lean_object* v___y_1761_, lean_object* v___y_1762_, lean_object* v___y_1763_, lean_object* v___y_1764_, lean_object* v___y_1765_){
_start:
{
lean_object* v_res_1766_; 
v_res_1766_ = lp_mathlib_Lean_logWarningAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__2(v_ref_1757_, v_msgData_1758_, v___y_1759_, v___y_1760_, v___y_1761_, v___y_1762_, v___y_1763_, v___y_1764_);
lean_dec(v___y_1764_);
lean_dec_ref(v___y_1763_);
lean_dec(v___y_1762_);
lean_dec_ref(v___y_1761_);
lean_dec(v___y_1760_);
lean_dec_ref(v___y_1759_);
lean_dec(v_ref_1757_);
return v_res_1766_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__1(void){
_start:
{
lean_object* v___x_1768_; lean_object* v___x_1769_; 
v___x_1768_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__0));
v___x_1769_ = l_Lean_stringToMessageData(v___x_1768_);
return v___x_1769_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__8(void){
_start:
{
lean_object* v___x_1776_; lean_object* v___x_1777_; 
v___x_1776_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__7));
v___x_1777_ = l_Lean_stringToMessageData(v___x_1776_);
return v___x_1777_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2(lean_object* v_bindersElab_1778_, lean_object* v_toOmit_1779_, uint8_t v___x_1780_, lean_object* v_binders_1781_, uint8_t v___x_1782_, uint8_t v_checkRedundant_1783_, lean_object* v_lctx_1784_, lean_object* v_localInstances_1785_, lean_object* v___x_1786_, lean_object* v_binder_1787_, lean_object* v___x_1788_, lean_object* v___x_1789_, lean_object* v___x_1790_, lean_object* v_i_1791_, lean_object* v_x_1792_, lean_object* v_ident_x3f_1793_, lean_object* v___y_1794_, lean_object* v___y_1795_, lean_object* v___y_1796_, lean_object* v___y_1797_, lean_object* v___y_1798_, lean_object* v___y_1799_){
_start:
{
lean_object* v___x_1811_; lean_object* v___x_1812_; lean_object* v___x_1813_; lean_object* v___x_1814_; lean_object* v___x_1815_; lean_object* v___x_1816_; 
v___x_1811_ = l_Lean_instInhabitedExpr;
v___x_1812_ = lean_array_get_size(v_bindersElab_1778_);
v___x_1813_ = lean_unsigned_to_nat(1u);
v___x_1814_ = lean_nat_sub(v___x_1812_, v___x_1813_);
v___x_1815_ = lean_array_get_borrowed(v___x_1811_, v_bindersElab_1778_, v___x_1814_);
lean_dec(v___x_1814_);
lean_inc(v___y_1799_);
lean_inc_ref(v___y_1798_);
lean_inc(v___y_1797_);
lean_inc_ref(v___y_1796_);
lean_inc(v___x_1815_);
v___x_1816_ = lean_infer_type(v___x_1815_, v___y_1796_, v___y_1797_, v___y_1798_, v___y_1799_);
if (lean_obj_tag(v___x_1816_) == 0)
{
lean_object* v_a_1817_; lean_object* v___x_1818_; 
v_a_1817_ = lean_ctor_get(v___x_1816_, 0);
lean_inc(v_a_1817_);
lean_dec_ref_known(v___x_1816_, 1);
v___x_1818_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__0___redArg(v_a_1817_, v___y_1797_);
if (lean_obj_tag(v___x_1818_) == 0)
{
lean_object* v_a_1819_; lean_object* v___y_1821_; lean_object* v___x_1863_; 
v_a_1819_ = lean_ctor_get(v___x_1818_, 0);
lean_inc_n(v_a_1819_, 2);
lean_dec_ref_known(v___x_1818_, 1);
v___x_1863_ = lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_completeBinders_x27_isVariableAlias(v_a_1819_, v___y_1796_, v___y_1797_, v___y_1798_, v___y_1799_);
if (lean_obj_tag(v___x_1863_) == 0)
{
lean_object* v_a_1864_; lean_object* v___x_1866_; uint8_t v_isShared_1867_; uint8_t v_isSharedCheck_1918_; 
v_a_1864_ = lean_ctor_get(v___x_1863_, 0);
v_isSharedCheck_1918_ = !lean_is_exclusive(v___x_1863_);
if (v_isSharedCheck_1918_ == 0)
{
v___x_1866_ = v___x_1863_;
v_isShared_1867_ = v_isSharedCheck_1918_;
goto v_resetjp_1865_;
}
else
{
lean_inc(v_a_1864_);
lean_dec(v___x_1863_);
v___x_1866_ = lean_box(0);
v_isShared_1867_ = v_isSharedCheck_1918_;
goto v_resetjp_1865_;
}
v_resetjp_1865_:
{
uint8_t v___x_1868_; 
v___x_1868_ = lean_unbox(v_a_1864_);
lean_dec(v_a_1864_);
if (v___x_1868_ == 0)
{
lean_object* v___x_1869_; lean_object* v___f_1870_; lean_object* v___x_1871_; 
lean_del_object(v___x_1866_);
lean_dec_ref(v___x_1790_);
lean_dec_ref(v___x_1789_);
lean_dec_ref(v___x_1788_);
v___x_1869_ = lean_box(0);
lean_inc(v_a_1819_);
v___f_1870_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__1___boxed), 9, 2);
lean_closure_set(v___f_1870_, 0, v_a_1819_);
lean_closure_set(v___f_1870_, 1, v___x_1869_);
lean_inc_ref(v_localInstances_1785_);
lean_inc_ref(v_lctx_1784_);
v___x_1871_ = lp_mathlib_Lean_Meta_withLCtx___at___00Mathlib_Command_Variable_completeBinders_x27_spec__3___redArg(v_lctx_1784_, v_localInstances_1785_, v___f_1870_, v___y_1794_, v___y_1795_, v___y_1796_, v___y_1797_, v___y_1798_, v___y_1799_);
if (lean_obj_tag(v___x_1871_) == 0)
{
v___y_1821_ = v___x_1871_;
goto v___jp_1820_;
}
else
{
lean_object* v_a_1872_; uint8_t v___y_1874_; uint8_t v___x_1875_; 
v_a_1872_ = lean_ctor_get(v___x_1871_, 0);
lean_inc(v_a_1872_);
v___x_1875_ = l_Lean_Exception_isInterrupt(v_a_1872_);
if (v___x_1875_ == 0)
{
uint8_t v___x_1876_; 
v___x_1876_ = l_Lean_Exception_isRuntime(v_a_1872_);
v___y_1874_ = v___x_1876_;
goto v___jp_1873_;
}
else
{
lean_dec(v_a_1872_);
v___y_1874_ = v___x_1875_;
goto v___jp_1873_;
}
v___jp_1873_:
{
if (v___y_1874_ == 0)
{
lean_dec_ref_known(v___x_1871_, 1);
lean_dec(v_a_1819_);
lean_dec(v___x_1786_);
lean_dec_ref(v_localInstances_1785_);
lean_dec_ref(v_lctx_1784_);
goto v___jp_1806_;
}
else
{
v___y_1821_ = v___x_1871_;
goto v___jp_1820_;
}
}
}
}
else
{
lean_object* v___x_1877_; lean_object* v___x_1878_; lean_object* v___y_1880_; 
lean_dec(v_a_1819_);
lean_dec(v___x_1786_);
lean_dec_ref(v_localInstances_1785_);
lean_dec_ref(v_lctx_1784_);
v___x_1877_ = lean_unsigned_to_nat(2u);
v___x_1878_ = l_Lean_Syntax_getArg(v_binder_1787_, v___x_1877_);
if (lean_obj_tag(v_ident_x3f_1793_) == 0)
{
v___y_1880_ = v___y_1798_;
goto v___jp_1879_;
}
else
{
lean_object* v___x_1908_; lean_object* v___x_1909_; 
v___x_1908_ = lean_obj_once(&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__8, &lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__8_once, _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__8);
v___x_1909_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2___redArg(v_binder_1787_, v___x_1908_, v___y_1794_, v___y_1795_, v___y_1796_, v___y_1797_, v___y_1798_, v___y_1799_);
if (lean_obj_tag(v___x_1909_) == 0)
{
lean_dec_ref_known(v___x_1909_, 1);
v___y_1880_ = v___y_1798_;
goto v___jp_1879_;
}
else
{
lean_object* v_a_1910_; lean_object* v___x_1912_; uint8_t v_isShared_1913_; uint8_t v_isSharedCheck_1917_; 
lean_dec(v___x_1878_);
lean_del_object(v___x_1866_);
lean_dec_ref(v___x_1790_);
lean_dec_ref(v___x_1789_);
lean_dec_ref(v___x_1788_);
lean_dec_ref(v_binders_1781_);
lean_dec_ref(v_toOmit_1779_);
v_a_1910_ = lean_ctor_get(v___x_1909_, 0);
v_isSharedCheck_1917_ = !lean_is_exclusive(v___x_1909_);
if (v_isSharedCheck_1917_ == 0)
{
v___x_1912_ = v___x_1909_;
v_isShared_1913_ = v_isSharedCheck_1917_;
goto v_resetjp_1911_;
}
else
{
lean_inc(v_a_1910_);
lean_dec(v___x_1909_);
v___x_1912_ = lean_box(0);
v_isShared_1913_ = v_isSharedCheck_1917_;
goto v_resetjp_1911_;
}
v_resetjp_1911_:
{
lean_object* v___x_1915_; 
if (v_isShared_1913_ == 0)
{
v___x_1915_ = v___x_1912_;
goto v_reusejp_1914_;
}
else
{
lean_object* v_reuseFailAlloc_1916_; 
v_reuseFailAlloc_1916_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1916_, 0, v_a_1910_);
v___x_1915_ = v_reuseFailAlloc_1916_;
goto v_reusejp_1914_;
}
v_reusejp_1914_:
{
return v___x_1915_;
}
}
}
}
v___jp_1879_:
{
lean_object* v_ref_1881_; lean_object* v_ref_1882_; lean_object* v___x_1883_; lean_object* v___x_1884_; lean_object* v___x_1885_; lean_object* v___x_1886_; lean_object* v___x_1887_; lean_object* v___x_1888_; lean_object* v___x_1889_; lean_object* v___x_1890_; lean_object* v___x_1891_; lean_object* v___x_1892_; lean_object* v___x_1893_; lean_object* v___x_1894_; lean_object* v___x_1895_; lean_object* v___x_1896_; lean_object* v___x_1897_; lean_object* v___x_1898_; lean_object* v___x_1899_; lean_object* v___x_1900_; lean_object* v___x_1901_; lean_object* v___x_1902_; lean_object* v___x_1903_; lean_object* v___x_1904_; lean_object* v___x_1906_; 
v_ref_1881_ = lean_ctor_get(v___y_1880_, 5);
v_ref_1882_ = l_Lean_replaceRef(v_binder_1787_, v_ref_1881_);
v___x_1883_ = l_Lean_SourceInfo_fromRef(v_ref_1882_, v___x_1780_);
lean_dec(v_ref_1882_);
v___x_1884_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__5));
lean_inc_ref(v___x_1790_);
lean_inc_ref(v___x_1789_);
lean_inc_ref(v___x_1788_);
v___x_1885_ = l_Lean_Name_mkStr4(v___x_1788_, v___x_1789_, v___x_1790_, v___x_1884_);
v___x_1886_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__2));
lean_inc_n(v___x_1883_, 7);
v___x_1887_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1887_, 0, v___x_1883_);
lean_ctor_set(v___x_1887_, 1, v___x_1886_);
v___x_1888_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__2));
v___x_1889_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__3));
v___x_1890_ = l_Lean_Name_mkStr4(v___x_1788_, v___x_1789_, v___x_1790_, v___x_1889_);
v___x_1891_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__4));
v___x_1892_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1892_, 0, v___x_1883_);
lean_ctor_set(v___x_1892_, 1, v___x_1891_);
v___x_1893_ = l_Lean_Syntax_node1(v___x_1883_, v___x_1890_, v___x_1892_);
v___x_1894_ = l_Lean_Syntax_node1(v___x_1883_, v___x_1888_, v___x_1893_);
v___x_1895_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__5));
v___x_1896_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1896_, 0, v___x_1883_);
lean_ctor_set(v___x_1896_, 1, v___x_1895_);
v___x_1897_ = l_Lean_Syntax_node2(v___x_1883_, v___x_1888_, v___x_1896_, v___x_1878_);
v___x_1898_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__6));
v___x_1899_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1899_, 0, v___x_1883_);
lean_ctor_set(v___x_1899_, 1, v___x_1898_);
v___x_1900_ = l_Lean_Syntax_node4(v___x_1883_, v___x_1885_, v___x_1887_, v___x_1894_, v___x_1897_, v___x_1899_);
v___x_1901_ = lean_array_set(v_binders_1781_, v_i_1791_, v___x_1900_);
v___x_1902_ = lean_box(v___x_1782_);
v___x_1903_ = lean_array_push(v_toOmit_1779_, v___x_1902_);
v___x_1904_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1904_, 0, v___x_1901_);
lean_ctor_set(v___x_1904_, 1, v___x_1903_);
if (v_isShared_1867_ == 0)
{
lean_ctor_set(v___x_1866_, 0, v___x_1904_);
v___x_1906_ = v___x_1866_;
goto v_reusejp_1905_;
}
else
{
lean_object* v_reuseFailAlloc_1907_; 
v_reuseFailAlloc_1907_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1907_, 0, v___x_1904_);
v___x_1906_ = v_reuseFailAlloc_1907_;
goto v_reusejp_1905_;
}
v_reusejp_1905_:
{
return v___x_1906_;
}
}
}
}
}
else
{
lean_object* v_a_1919_; lean_object* v___x_1921_; uint8_t v_isShared_1922_; uint8_t v_isSharedCheck_1926_; 
lean_dec(v_a_1819_);
lean_dec_ref(v___x_1790_);
lean_dec_ref(v___x_1789_);
lean_dec_ref(v___x_1788_);
lean_dec(v___x_1786_);
lean_dec_ref(v_localInstances_1785_);
lean_dec_ref(v_lctx_1784_);
lean_dec_ref(v_binders_1781_);
lean_dec_ref(v_toOmit_1779_);
v_a_1919_ = lean_ctor_get(v___x_1863_, 0);
v_isSharedCheck_1926_ = !lean_is_exclusive(v___x_1863_);
if (v_isSharedCheck_1926_ == 0)
{
v___x_1921_ = v___x_1863_;
v_isShared_1922_ = v_isSharedCheck_1926_;
goto v_resetjp_1920_;
}
else
{
lean_inc(v_a_1919_);
lean_dec(v___x_1863_);
v___x_1921_ = lean_box(0);
v_isShared_1922_ = v_isSharedCheck_1926_;
goto v_resetjp_1920_;
}
v_resetjp_1920_:
{
lean_object* v___x_1924_; 
if (v_isShared_1922_ == 0)
{
v___x_1924_ = v___x_1921_;
goto v_reusejp_1923_;
}
else
{
lean_object* v_reuseFailAlloc_1925_; 
v_reuseFailAlloc_1925_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1925_, 0, v_a_1919_);
v___x_1924_ = v_reuseFailAlloc_1925_;
goto v_reusejp_1923_;
}
v_reusejp_1923_:
{
return v___x_1924_;
}
}
}
v___jp_1820_:
{
if (lean_obj_tag(v___y_1821_) == 0)
{
lean_object* v_a_1822_; 
v_a_1822_ = lean_ctor_get(v___y_1821_, 0);
lean_inc(v_a_1822_);
lean_dec_ref_known(v___y_1821_, 1);
if (lean_obj_tag(v_a_1822_) == 1)
{
lean_object* v___x_1824_; uint8_t v_isShared_1825_; uint8_t v_isSharedCheck_1853_; 
v_isSharedCheck_1853_ = !lean_is_exclusive(v_a_1822_);
if (v_isSharedCheck_1853_ == 0)
{
lean_object* v_unused_1854_; 
v_unused_1854_ = lean_ctor_get(v_a_1822_, 0);
lean_dec(v_unused_1854_);
v___x_1824_ = v_a_1822_;
v_isShared_1825_ = v_isSharedCheck_1853_;
goto v_resetjp_1823_;
}
else
{
lean_dec(v_a_1822_);
v___x_1824_ = lean_box(0);
v_isShared_1825_ = v_isSharedCheck_1853_;
goto v_resetjp_1823_;
}
v_resetjp_1823_:
{
if (v_checkRedundant_1783_ == 0)
{
lean_del_object(v___x_1824_);
lean_dec(v_a_1819_);
lean_dec(v___x_1786_);
lean_dec_ref(v_localInstances_1785_);
lean_dec_ref(v_lctx_1784_);
goto v___jp_1801_;
}
else
{
uint8_t v___x_1826_; lean_object* v___x_1827_; lean_object* v___x_1828_; 
v___x_1826_ = 0;
v___x_1827_ = lean_box(0);
v___x_1828_ = l_Lean_Meta_mkFreshExprMVarAt(v_lctx_1784_, v_localInstances_1785_, v_a_1819_, v___x_1826_, v___x_1827_, v___x_1786_, v___y_1796_, v___y_1797_, v___y_1798_, v___y_1799_);
if (lean_obj_tag(v___x_1828_) == 0)
{
lean_object* v_a_1829_; lean_object* v___x_1830_; lean_object* v___x_1831_; lean_object* v___x_1833_; 
v_a_1829_ = lean_ctor_get(v___x_1828_, 0);
lean_inc(v_a_1829_);
lean_dec_ref_known(v___x_1828_, 1);
v___x_1830_ = lean_obj_once(&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__1, &lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__1_once, _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___closed__1);
v___x_1831_ = l_Lean_Expr_mvarId_x21(v_a_1829_);
lean_dec(v_a_1829_);
if (v_isShared_1825_ == 0)
{
lean_ctor_set(v___x_1824_, 0, v___x_1831_);
v___x_1833_ = v___x_1824_;
goto v_reusejp_1832_;
}
else
{
lean_object* v_reuseFailAlloc_1844_; 
v_reuseFailAlloc_1844_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1844_, 0, v___x_1831_);
v___x_1833_ = v_reuseFailAlloc_1844_;
goto v_reusejp_1832_;
}
v_reusejp_1832_:
{
lean_object* v___x_1834_; lean_object* v___x_1835_; 
v___x_1834_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1834_, 0, v___x_1830_);
lean_ctor_set(v___x_1834_, 1, v___x_1833_);
v___x_1835_ = lp_mathlib_Lean_logWarningAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__2(v_binder_1787_, v___x_1834_, v___y_1794_, v___y_1795_, v___y_1796_, v___y_1797_, v___y_1798_, v___y_1799_);
if (lean_obj_tag(v___x_1835_) == 0)
{
lean_dec_ref_known(v___x_1835_, 1);
goto v___jp_1801_;
}
else
{
lean_object* v_a_1836_; lean_object* v___x_1838_; uint8_t v_isShared_1839_; uint8_t v_isSharedCheck_1843_; 
lean_dec_ref(v_binders_1781_);
lean_dec_ref(v_toOmit_1779_);
v_a_1836_ = lean_ctor_get(v___x_1835_, 0);
v_isSharedCheck_1843_ = !lean_is_exclusive(v___x_1835_);
if (v_isSharedCheck_1843_ == 0)
{
v___x_1838_ = v___x_1835_;
v_isShared_1839_ = v_isSharedCheck_1843_;
goto v_resetjp_1837_;
}
else
{
lean_inc(v_a_1836_);
lean_dec(v___x_1835_);
v___x_1838_ = lean_box(0);
v_isShared_1839_ = v_isSharedCheck_1843_;
goto v_resetjp_1837_;
}
v_resetjp_1837_:
{
lean_object* v___x_1841_; 
if (v_isShared_1839_ == 0)
{
v___x_1841_ = v___x_1838_;
goto v_reusejp_1840_;
}
else
{
lean_object* v_reuseFailAlloc_1842_; 
v_reuseFailAlloc_1842_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1842_, 0, v_a_1836_);
v___x_1841_ = v_reuseFailAlloc_1842_;
goto v_reusejp_1840_;
}
v_reusejp_1840_:
{
return v___x_1841_;
}
}
}
}
}
else
{
lean_object* v_a_1845_; lean_object* v___x_1847_; uint8_t v_isShared_1848_; uint8_t v_isSharedCheck_1852_; 
lean_del_object(v___x_1824_);
lean_dec_ref(v_binders_1781_);
lean_dec_ref(v_toOmit_1779_);
v_a_1845_ = lean_ctor_get(v___x_1828_, 0);
v_isSharedCheck_1852_ = !lean_is_exclusive(v___x_1828_);
if (v_isSharedCheck_1852_ == 0)
{
v___x_1847_ = v___x_1828_;
v_isShared_1848_ = v_isSharedCheck_1852_;
goto v_resetjp_1846_;
}
else
{
lean_inc(v_a_1845_);
lean_dec(v___x_1828_);
v___x_1847_ = lean_box(0);
v_isShared_1848_ = v_isSharedCheck_1852_;
goto v_resetjp_1846_;
}
v_resetjp_1846_:
{
lean_object* v___x_1850_; 
if (v_isShared_1848_ == 0)
{
v___x_1850_ = v___x_1847_;
goto v_reusejp_1849_;
}
else
{
lean_object* v_reuseFailAlloc_1851_; 
v_reuseFailAlloc_1851_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1851_, 0, v_a_1845_);
v___x_1850_ = v_reuseFailAlloc_1851_;
goto v_reusejp_1849_;
}
v_reusejp_1849_:
{
return v___x_1850_;
}
}
}
}
}
}
else
{
lean_dec(v_a_1822_);
lean_dec(v_a_1819_);
lean_dec(v___x_1786_);
lean_dec_ref(v_localInstances_1785_);
lean_dec_ref(v_lctx_1784_);
goto v___jp_1806_;
}
}
else
{
lean_object* v_a_1855_; lean_object* v___x_1857_; uint8_t v_isShared_1858_; uint8_t v_isSharedCheck_1862_; 
lean_dec(v_a_1819_);
lean_dec(v___x_1786_);
lean_dec_ref(v_localInstances_1785_);
lean_dec_ref(v_lctx_1784_);
lean_dec_ref(v_binders_1781_);
lean_dec_ref(v_toOmit_1779_);
v_a_1855_ = lean_ctor_get(v___y_1821_, 0);
v_isSharedCheck_1862_ = !lean_is_exclusive(v___y_1821_);
if (v_isSharedCheck_1862_ == 0)
{
v___x_1857_ = v___y_1821_;
v_isShared_1858_ = v_isSharedCheck_1862_;
goto v_resetjp_1856_;
}
else
{
lean_inc(v_a_1855_);
lean_dec(v___y_1821_);
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
}
else
{
lean_object* v_a_1927_; lean_object* v___x_1929_; uint8_t v_isShared_1930_; uint8_t v_isSharedCheck_1934_; 
lean_dec_ref(v___x_1790_);
lean_dec_ref(v___x_1789_);
lean_dec_ref(v___x_1788_);
lean_dec(v___x_1786_);
lean_dec_ref(v_localInstances_1785_);
lean_dec_ref(v_lctx_1784_);
lean_dec_ref(v_binders_1781_);
lean_dec_ref(v_toOmit_1779_);
v_a_1927_ = lean_ctor_get(v___x_1818_, 0);
v_isSharedCheck_1934_ = !lean_is_exclusive(v___x_1818_);
if (v_isSharedCheck_1934_ == 0)
{
v___x_1929_ = v___x_1818_;
v_isShared_1930_ = v_isSharedCheck_1934_;
goto v_resetjp_1928_;
}
else
{
lean_inc(v_a_1927_);
lean_dec(v___x_1818_);
v___x_1929_ = lean_box(0);
v_isShared_1930_ = v_isSharedCheck_1934_;
goto v_resetjp_1928_;
}
v_resetjp_1928_:
{
lean_object* v___x_1932_; 
if (v_isShared_1930_ == 0)
{
v___x_1932_ = v___x_1929_;
goto v_reusejp_1931_;
}
else
{
lean_object* v_reuseFailAlloc_1933_; 
v_reuseFailAlloc_1933_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1933_, 0, v_a_1927_);
v___x_1932_ = v_reuseFailAlloc_1933_;
goto v_reusejp_1931_;
}
v_reusejp_1931_:
{
return v___x_1932_;
}
}
}
}
else
{
lean_object* v_a_1935_; lean_object* v___x_1937_; uint8_t v_isShared_1938_; uint8_t v_isSharedCheck_1942_; 
lean_dec_ref(v___x_1790_);
lean_dec_ref(v___x_1789_);
lean_dec_ref(v___x_1788_);
lean_dec(v___x_1786_);
lean_dec_ref(v_localInstances_1785_);
lean_dec_ref(v_lctx_1784_);
lean_dec_ref(v_binders_1781_);
lean_dec_ref(v_toOmit_1779_);
v_a_1935_ = lean_ctor_get(v___x_1816_, 0);
v_isSharedCheck_1942_ = !lean_is_exclusive(v___x_1816_);
if (v_isSharedCheck_1942_ == 0)
{
v___x_1937_ = v___x_1816_;
v_isShared_1938_ = v_isSharedCheck_1942_;
goto v_resetjp_1936_;
}
else
{
lean_inc(v_a_1935_);
lean_dec(v___x_1816_);
v___x_1937_ = lean_box(0);
v_isShared_1938_ = v_isSharedCheck_1942_;
goto v_resetjp_1936_;
}
v_resetjp_1936_:
{
lean_object* v___x_1940_; 
if (v_isShared_1938_ == 0)
{
v___x_1940_ = v___x_1937_;
goto v_reusejp_1939_;
}
else
{
lean_object* v_reuseFailAlloc_1941_; 
v_reuseFailAlloc_1941_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1941_, 0, v_a_1935_);
v___x_1940_ = v_reuseFailAlloc_1941_;
goto v_reusejp_1939_;
}
v_reusejp_1939_:
{
return v___x_1940_;
}
}
}
v___jp_1801_:
{
lean_object* v___x_1802_; lean_object* v___x_1803_; lean_object* v___x_1804_; lean_object* v___x_1805_; 
v___x_1802_ = lean_box(v___x_1782_);
v___x_1803_ = lean_array_push(v_toOmit_1779_, v___x_1802_);
v___x_1804_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1804_, 0, v_binders_1781_);
lean_ctor_set(v___x_1804_, 1, v___x_1803_);
v___x_1805_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1805_, 0, v___x_1804_);
return v___x_1805_;
}
v___jp_1806_:
{
lean_object* v___x_1807_; lean_object* v___x_1808_; lean_object* v___x_1809_; lean_object* v___x_1810_; 
v___x_1807_ = lean_box(v___x_1780_);
v___x_1808_ = lean_array_push(v_toOmit_1779_, v___x_1807_);
v___x_1809_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1809_, 0, v_binders_1781_);
lean_ctor_set(v___x_1809_, 1, v___x_1808_);
v___x_1810_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1810_, 0, v___x_1809_);
return v___x_1810_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___boxed(lean_object** _args){
lean_object* v_bindersElab_1943_ = _args[0];
lean_object* v_toOmit_1944_ = _args[1];
lean_object* v___x_1945_ = _args[2];
lean_object* v_binders_1946_ = _args[3];
lean_object* v___x_1947_ = _args[4];
lean_object* v_checkRedundant_1948_ = _args[5];
lean_object* v_lctx_1949_ = _args[6];
lean_object* v_localInstances_1950_ = _args[7];
lean_object* v___x_1951_ = _args[8];
lean_object* v_binder_1952_ = _args[9];
lean_object* v___x_1953_ = _args[10];
lean_object* v___x_1954_ = _args[11];
lean_object* v___x_1955_ = _args[12];
lean_object* v_i_1956_ = _args[13];
lean_object* v_x_1957_ = _args[14];
lean_object* v_ident_x3f_1958_ = _args[15];
lean_object* v___y_1959_ = _args[16];
lean_object* v___y_1960_ = _args[17];
lean_object* v___y_1961_ = _args[18];
lean_object* v___y_1962_ = _args[19];
lean_object* v___y_1963_ = _args[20];
lean_object* v___y_1964_ = _args[21];
lean_object* v___y_1965_ = _args[22];
_start:
{
uint8_t v___x_31527__boxed_1966_; uint8_t v___x_31528__boxed_1967_; uint8_t v_checkRedundant_boxed_1968_; lean_object* v_res_1969_; 
v___x_31527__boxed_1966_ = lean_unbox(v___x_1945_);
v___x_31528__boxed_1967_ = lean_unbox(v___x_1947_);
v_checkRedundant_boxed_1968_ = lean_unbox(v_checkRedundant_1948_);
v_res_1969_ = lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2(v_bindersElab_1943_, v_toOmit_1944_, v___x_31527__boxed_1966_, v_binders_1946_, v___x_31528__boxed_1967_, v_checkRedundant_boxed_1968_, v_lctx_1949_, v_localInstances_1950_, v___x_1951_, v_binder_1952_, v___x_1953_, v___x_1954_, v___x_1955_, v_i_1956_, v_x_1957_, v_ident_x3f_1958_, v___y_1959_, v___y_1960_, v___y_1961_, v___y_1962_, v___y_1963_, v___y_1964_);
lean_dec(v___y_1964_);
lean_dec_ref(v___y_1963_);
lean_dec(v___y_1962_);
lean_dec_ref(v___y_1961_);
lean_dec(v___y_1960_);
lean_dec_ref(v___y_1959_);
lean_dec(v_ident_x3f_1958_);
lean_dec(v_i_1956_);
lean_dec(v_binder_1952_);
lean_dec_ref(v_bindersElab_1943_);
return v_res_1969_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_Variable_completeBinders_x27_spec__4___redArg(size_t v_sz_1970_, size_t v_i_1971_, lean_object* v_bs_1972_, lean_object* v___y_1973_, lean_object* v___y_1974_, lean_object* v___y_1975_, lean_object* v___y_1976_){
_start:
{
uint8_t v___x_1978_; 
v___x_1978_ = lean_usize_dec_lt(v_i_1971_, v_sz_1970_);
if (v___x_1978_ == 0)
{
lean_object* v___x_1979_; 
v___x_1979_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1979_, 0, v_bs_1972_);
return v___x_1979_;
}
else
{
lean_object* v_v_1980_; lean_object* v___x_1981_; 
v_v_1980_ = lean_array_uget_borrowed(v_bs_1972_, v_i_1971_);
lean_inc(v___y_1976_);
lean_inc_ref(v___y_1975_);
lean_inc(v___y_1974_);
lean_inc_ref(v___y_1973_);
lean_inc(v_v_1980_);
v___x_1981_ = lean_infer_type(v_v_1980_, v___y_1973_, v___y_1974_, v___y_1975_, v___y_1976_);
if (lean_obj_tag(v___x_1981_) == 0)
{
lean_object* v_a_1982_; lean_object* v___x_1983_; lean_object* v_bs_x27_1984_; size_t v___x_1985_; size_t v___x_1986_; lean_object* v___x_1987_; 
v_a_1982_ = lean_ctor_get(v___x_1981_, 0);
lean_inc(v_a_1982_);
lean_dec_ref_known(v___x_1981_, 1);
v___x_1983_ = lean_unsigned_to_nat(0u);
v_bs_x27_1984_ = lean_array_uset(v_bs_1972_, v_i_1971_, v___x_1983_);
v___x_1985_ = ((size_t)1ULL);
v___x_1986_ = lean_usize_add(v_i_1971_, v___x_1985_);
v___x_1987_ = lean_array_uset(v_bs_x27_1984_, v_i_1971_, v_a_1982_);
v_i_1971_ = v___x_1986_;
v_bs_1972_ = v___x_1987_;
goto _start;
}
else
{
lean_object* v_a_1989_; lean_object* v___x_1991_; uint8_t v_isShared_1992_; uint8_t v_isSharedCheck_1996_; 
lean_dec_ref(v_bs_1972_);
v_a_1989_ = lean_ctor_get(v___x_1981_, 0);
v_isSharedCheck_1996_ = !lean_is_exclusive(v___x_1981_);
if (v_isSharedCheck_1996_ == 0)
{
v___x_1991_ = v___x_1981_;
v_isShared_1992_ = v_isSharedCheck_1996_;
goto v_resetjp_1990_;
}
else
{
lean_inc(v_a_1989_);
lean_dec(v___x_1981_);
v___x_1991_ = lean_box(0);
v_isShared_1992_ = v_isSharedCheck_1996_;
goto v_resetjp_1990_;
}
v_resetjp_1990_:
{
lean_object* v___x_1994_; 
if (v_isShared_1992_ == 0)
{
v___x_1994_ = v___x_1991_;
goto v_reusejp_1993_;
}
else
{
lean_object* v_reuseFailAlloc_1995_; 
v_reuseFailAlloc_1995_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1995_, 0, v_a_1989_);
v___x_1994_ = v_reuseFailAlloc_1995_;
goto v_reusejp_1993_;
}
v_reusejp_1993_:
{
return v___x_1994_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_Variable_completeBinders_x27_spec__4___redArg___boxed(lean_object* v_sz_1997_, lean_object* v_i_1998_, lean_object* v_bs_1999_, lean_object* v___y_2000_, lean_object* v___y_2001_, lean_object* v___y_2002_, lean_object* v___y_2003_, lean_object* v___y_2004_){
_start:
{
size_t v_sz_boxed_2005_; size_t v_i_boxed_2006_; lean_object* v_res_2007_; 
v_sz_boxed_2005_ = lean_unbox_usize(v_sz_1997_);
lean_dec(v_sz_1997_);
v_i_boxed_2006_ = lean_unbox_usize(v_i_1998_);
lean_dec(v_i_1998_);
v_res_2007_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_Variable_completeBinders_x27_spec__4___redArg(v_sz_boxed_2005_, v_i_boxed_2006_, v_bs_1999_, v___y_2000_, v___y_2001_, v___y_2002_, v___y_2003_);
lean_dec(v___y_2003_);
lean_dec_ref(v___y_2002_);
lean_dec(v___y_2001_);
lean_dec_ref(v___y_2000_);
return v_res_2007_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Command_Variable_completeBinders_x27_spec__5(lean_object* v_a_2008_, lean_object* v_a_2009_){
_start:
{
if (lean_obj_tag(v_a_2008_) == 0)
{
lean_object* v___x_2010_; 
v___x_2010_ = l_List_reverse___redArg(v_a_2009_);
return v___x_2010_;
}
else
{
lean_object* v_head_2011_; lean_object* v_tail_2012_; lean_object* v___x_2014_; uint8_t v_isShared_2015_; uint8_t v_isSharedCheck_2021_; 
v_head_2011_ = lean_ctor_get(v_a_2008_, 0);
v_tail_2012_ = lean_ctor_get(v_a_2008_, 1);
v_isSharedCheck_2021_ = !lean_is_exclusive(v_a_2008_);
if (v_isSharedCheck_2021_ == 0)
{
v___x_2014_ = v_a_2008_;
v_isShared_2015_ = v_isSharedCheck_2021_;
goto v_resetjp_2013_;
}
else
{
lean_inc(v_tail_2012_);
lean_inc(v_head_2011_);
lean_dec(v_a_2008_);
v___x_2014_ = lean_box(0);
v_isShared_2015_ = v_isSharedCheck_2021_;
goto v_resetjp_2013_;
}
v_resetjp_2013_:
{
lean_object* v___x_2016_; lean_object* v___x_2018_; 
v___x_2016_ = l_Lean_MessageData_ofExpr(v_head_2011_);
if (v_isShared_2015_ == 0)
{
lean_ctor_set(v___x_2014_, 1, v_a_2009_);
lean_ctor_set(v___x_2014_, 0, v___x_2016_);
v___x_2018_ = v___x_2014_;
goto v_reusejp_2017_;
}
else
{
lean_object* v_reuseFailAlloc_2020_; 
v_reuseFailAlloc_2020_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2020_, 0, v___x_2016_);
lean_ctor_set(v_reuseFailAlloc_2020_, 1, v_a_2009_);
v___x_2018_ = v_reuseFailAlloc_2020_;
goto v_reusejp_2017_;
}
v_reusejp_2017_:
{
v_a_2008_ = v_tail_2012_;
v_a_2009_ = v___x_2018_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00Mathlib_Command_Variable_completeBinders_x27_spec__6_spec__7(lean_object* v_o_2022_, lean_object* v_k_2023_, uint8_t v_v_2024_){
_start:
{
lean_object* v_map_2025_; uint8_t v_hasTrace_2026_; lean_object* v___x_2028_; uint8_t v_isShared_2029_; uint8_t v_isSharedCheck_2040_; 
v_map_2025_ = lean_ctor_get(v_o_2022_, 0);
v_hasTrace_2026_ = lean_ctor_get_uint8(v_o_2022_, sizeof(void*)*1);
v_isSharedCheck_2040_ = !lean_is_exclusive(v_o_2022_);
if (v_isSharedCheck_2040_ == 0)
{
v___x_2028_ = v_o_2022_;
v_isShared_2029_ = v_isSharedCheck_2040_;
goto v_resetjp_2027_;
}
else
{
lean_inc(v_map_2025_);
lean_dec(v_o_2022_);
v___x_2028_ = lean_box(0);
v_isShared_2029_ = v_isSharedCheck_2040_;
goto v_resetjp_2027_;
}
v_resetjp_2027_:
{
lean_object* v___x_2030_; lean_object* v___x_2031_; 
v___x_2030_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_2030_, 0, v_v_2024_);
lean_inc(v_k_2023_);
v___x_2031_ = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(v_k_2023_, v___x_2030_, v_map_2025_);
if (v_hasTrace_2026_ == 0)
{
lean_object* v___x_2032_; uint8_t v___x_2033_; lean_object* v___x_2035_; 
v___x_2032_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__7));
v___x_2033_ = l_Lean_Name_isPrefixOf(v___x_2032_, v_k_2023_);
lean_dec(v_k_2023_);
if (v_isShared_2029_ == 0)
{
lean_ctor_set(v___x_2028_, 0, v___x_2031_);
v___x_2035_ = v___x_2028_;
goto v_reusejp_2034_;
}
else
{
lean_object* v_reuseFailAlloc_2036_; 
v_reuseFailAlloc_2036_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_2036_, 0, v___x_2031_);
v___x_2035_ = v_reuseFailAlloc_2036_;
goto v_reusejp_2034_;
}
v_reusejp_2034_:
{
lean_ctor_set_uint8(v___x_2035_, sizeof(void*)*1, v___x_2033_);
return v___x_2035_;
}
}
else
{
lean_object* v___x_2038_; 
lean_dec(v_k_2023_);
if (v_isShared_2029_ == 0)
{
lean_ctor_set(v___x_2028_, 0, v___x_2031_);
v___x_2038_ = v___x_2028_;
goto v_reusejp_2037_;
}
else
{
lean_object* v_reuseFailAlloc_2039_; 
v_reuseFailAlloc_2039_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v_reuseFailAlloc_2039_, 0, v___x_2031_);
lean_ctor_set_uint8(v_reuseFailAlloc_2039_, sizeof(void*)*1, v_hasTrace_2026_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00Mathlib_Command_Variable_completeBinders_x27_spec__6_spec__7___boxed(lean_object* v_o_2041_, lean_object* v_k_2042_, lean_object* v_v_2043_){
_start:
{
uint8_t v_v_boxed_2044_; lean_object* v_res_2045_; 
v_v_boxed_2044_ = lean_unbox(v_v_2043_);
v_res_2045_ = lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00Mathlib_Command_Variable_completeBinders_x27_spec__6_spec__7(v_o_2041_, v_k_2042_, v_v_boxed_2044_);
return v_res_2045_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_set___at___00Mathlib_Command_Variable_completeBinders_x27_spec__6(lean_object* v_opts_2046_, lean_object* v_opt_2047_, uint8_t v_val_2048_){
_start:
{
lean_object* v_name_2049_; lean_object* v___x_2050_; 
v_name_2049_ = lean_ctor_get(v_opt_2047_, 0);
lean_inc(v_name_2049_);
lean_dec_ref(v_opt_2047_);
v___x_2050_ = lp_mathlib_Lean_Options_set___at___00Lean_Option_set___at___00Mathlib_Command_Variable_completeBinders_x27_spec__6_spec__7(v_opts_2046_, v_name_2049_, v_val_2048_);
return v___x_2050_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_set___at___00Mathlib_Command_Variable_completeBinders_x27_spec__6___boxed(lean_object* v_opts_2051_, lean_object* v_opt_2052_, lean_object* v_val_2053_){
_start:
{
uint8_t v_val_boxed_2054_; lean_object* v_res_2055_; 
v_val_boxed_2054_ = lean_unbox(v_val_2053_);
v_res_2055_ = lp_mathlib_Lean_Option_set___at___00Mathlib_Command_Variable_completeBinders_x27_spec__6(v_opts_2051_, v_opt_2052_, v_val_boxed_2054_);
return v_res_2055_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Command_Variable_completeBinders_x27_spec__1(lean_object* v_snd_2056_, lean_object* v_as_2057_, size_t v_i_2058_, size_t v_stop_2059_){
_start:
{
uint8_t v___x_2060_; 
v___x_2060_ = lean_usize_dec_eq(v_i_2058_, v_stop_2059_);
if (v___x_2060_ == 0)
{
lean_object* v___x_2061_; uint8_t v___x_2062_; 
v___x_2061_ = lean_array_uget_borrowed(v_as_2057_, v_i_2058_);
v___x_2062_ = l_Lean_Syntax_structEq(v___x_2061_, v_snd_2056_);
if (v___x_2062_ == 0)
{
size_t v___x_2063_; size_t v___x_2064_; 
v___x_2063_ = ((size_t)1ULL);
v___x_2064_ = lean_usize_add(v_i_2058_, v___x_2063_);
v_i_2058_ = v___x_2064_;
goto _start;
}
else
{
return v___x_2062_;
}
}
else
{
uint8_t v___x_2066_; 
v___x_2066_ = 0;
return v___x_2066_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Command_Variable_completeBinders_x27_spec__1___boxed(lean_object* v_snd_2067_, lean_object* v_as_2068_, lean_object* v_i_2069_, lean_object* v_stop_2070_){
_start:
{
size_t v_i_boxed_2071_; size_t v_stop_boxed_2072_; uint8_t v_res_2073_; lean_object* v_r_2074_; 
v_i_boxed_2071_ = lean_unbox_usize(v_i_2069_);
lean_dec(v_i_2069_);
v_stop_boxed_2072_ = lean_unbox_usize(v_stop_2070_);
lean_dec(v_stop_2070_);
v_res_2073_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Command_Variable_completeBinders_x27_spec__1(v_snd_2067_, v_as_2068_, v_i_boxed_2071_, v_stop_boxed_2072_);
lean_dec_ref(v_as_2068_);
lean_dec(v_snd_2067_);
v_r_2074_ = lean_box(v_res_2073_);
return v_r_2074_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0(lean_object* v_ref_2075_, lean_object* v_msgData_2076_, lean_object* v___y_2077_, lean_object* v___y_2078_, lean_object* v___y_2079_, lean_object* v___y_2080_, lean_object* v___y_2081_, lean_object* v___y_2082_){
_start:
{
uint8_t v___x_2084_; uint8_t v___x_2085_; lean_object* v___x_2086_; 
v___x_2084_ = 2;
v___x_2085_ = 0;
v___x_2086_ = lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg(v_ref_2075_, v_msgData_2076_, v___x_2084_, v___x_2085_, v___y_2079_, v___y_2080_, v___y_2081_, v___y_2082_);
return v___x_2086_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0___boxed(lean_object* v_ref_2087_, lean_object* v_msgData_2088_, lean_object* v___y_2089_, lean_object* v___y_2090_, lean_object* v___y_2091_, lean_object* v___y_2092_, lean_object* v___y_2093_, lean_object* v___y_2094_, lean_object* v___y_2095_){
_start:
{
lean_object* v_res_2096_; 
v_res_2096_ = lp_mathlib_Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0(v_ref_2087_, v_msgData_2088_, v___y_2089_, v___y_2090_, v___y_2091_, v___y_2092_, v___y_2093_, v___y_2094_);
lean_dec(v___y_2094_);
lean_dec_ref(v___y_2093_);
lean_dec(v___y_2092_);
lean_dec_ref(v___y_2091_);
lean_dec(v___y_2090_);
lean_dec_ref(v___y_2089_);
lean_dec(v_ref_2087_);
return v_res_2096_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__0(void){
_start:
{
lean_object* v___x_2097_; 
v___x_2097_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_2097_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__1(void){
_start:
{
lean_object* v___x_2098_; lean_object* v___x_2099_; 
v___x_2098_ = lean_obj_once(&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__0, &lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__0_once, _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__0);
v___x_2099_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2099_, 0, v___x_2098_);
return v___x_2099_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__2(void){
_start:
{
lean_object* v___x_2100_; lean_object* v___x_2101_; 
v___x_2100_ = lean_obj_once(&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__1, &lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__1_once, _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__1);
v___x_2101_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2101_, 0, v___x_2100_);
lean_ctor_set(v___x_2101_, 1, v___x_2100_);
return v___x_2101_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__6(void){
_start:
{
lean_object* v___x_2109_; lean_object* v___x_2110_; 
v___x_2109_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__5));
v___x_2110_ = l_Lean_stringToMessageData(v___x_2109_);
return v___x_2110_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__8(void){
_start:
{
lean_object* v___x_2112_; lean_object* v___x_2113_; 
v___x_2112_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__7));
v___x_2113_ = l_Lean_stringToMessageData(v___x_2112_);
return v___x_2113_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__11(void){
_start:
{
lean_object* v___x_2117_; lean_object* v___x_2118_; 
v___x_2117_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__10));
v___x_2118_ = l_Lean_stringToMessageData(v___x_2117_);
return v___x_2118_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__13(void){
_start:
{
lean_object* v___x_2120_; lean_object* v___x_2121_; 
v___x_2120_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__12));
v___x_2121_ = l_Lean_stringToMessageData(v___x_2120_);
return v___x_2121_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__15(void){
_start:
{
lean_object* v___x_2123_; lean_object* v___x_2124_; 
v___x_2123_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__14));
v___x_2124_ = l_Lean_stringToMessageData(v___x_2123_);
return v___x_2124_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__17(void){
_start:
{
lean_object* v___x_2126_; lean_object* v___x_2127_; 
v___x_2126_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__16));
v___x_2127_ = l_Lean_stringToMessageData(v___x_2126_);
return v___x_2127_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__19(void){
_start:
{
lean_object* v___x_2129_; lean_object* v___x_2130_; 
v___x_2129_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__18));
v___x_2130_ = l_Lean_stringToMessageData(v___x_2129_);
return v___x_2130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__3___boxed(lean_object** _args){
lean_object* v_i_2131_ = _args[0];
lean_object* v_maxSteps_2132_ = _args[1];
lean_object* v_gas_2133_ = _args[2];
lean_object* v_checkRedundant_2134_ = _args[3];
lean_object* v___x_2135_ = _args[4];
lean_object* v_toOmit_2136_ = _args[5];
lean_object* v___x_2137_ = _args[6];
lean_object* v_binders_2138_ = _args[7];
lean_object* v_binder_2139_ = _args[8];
lean_object* v___x_2140_ = _args[9];
lean_object* v___f_2141_ = _args[10];
lean_object* v___y_2142_ = _args[11];
lean_object* v___y_2143_ = _args[12];
lean_object* v___y_2144_ = _args[13];
lean_object* v___y_2145_ = _args[14];
lean_object* v___y_2146_ = _args[15];
lean_object* v___y_2147_ = _args[16];
lean_object* v___y_2148_ = _args[17];
_start:
{
uint8_t v_checkRedundant_boxed_2149_; uint8_t v___x_32113__boxed_2150_; uint8_t v___x_32114__boxed_2151_; lean_object* v_res_2152_; 
v_checkRedundant_boxed_2149_ = lean_unbox(v_checkRedundant_2134_);
v___x_32113__boxed_2150_ = lean_unbox(v___x_2135_);
v___x_32114__boxed_2151_ = lean_unbox(v___x_2137_);
v_res_2152_ = lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__3(v_i_2131_, v_maxSteps_2132_, v_gas_2133_, v_checkRedundant_boxed_2149_, v___x_32113__boxed_2150_, v_toOmit_2136_, v___x_32114__boxed_2151_, v_binders_2138_, v_binder_2139_, v___x_2140_, v___f_2141_, v___y_2142_, v___y_2143_, v___y_2144_, v___y_2145_, v___y_2146_, v___y_2147_);
lean_dec(v___x_2140_);
lean_dec(v_binder_2139_);
lean_dec(v_i_2131_);
return v_res_2152_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__4___closed__1(void){
_start:
{
lean_object* v___x_2154_; lean_object* v___x_2155_; 
v___x_2154_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__4___closed__0));
v___x_2155_ = l_Lean_stringToMessageData(v___x_2154_);
return v___x_2155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__4(lean_object* v___f_2156_, lean_object* v_toOmit_2157_, lean_object* v_binders_2158_, uint8_t v___x_2159_, uint8_t v_checkRedundant_2160_, lean_object* v_lctx_2161_, lean_object* v_localInstances_2162_, lean_object* v___x_2163_, lean_object* v_binder_2164_, lean_object* v_i_2165_, lean_object* v_maxSteps_2166_, lean_object* v_gas_2167_, lean_object* v_cls_2168_, lean_object* v_bindersElab_2169_, lean_object* v___y_2170_, lean_object* v___y_2171_, lean_object* v___y_2172_, lean_object* v___y_2173_, lean_object* v___y_2174_, lean_object* v___y_2175_){
_start:
{
lean_object* v___y_2178_; lean_object* v___y_2179_; lean_object* v___y_2180_; lean_object* v___y_2181_; lean_object* v___y_2182_; lean_object* v___y_2183_; size_t v_sz_2208_; size_t v___x_2209_; lean_object* v___x_2210_; 
v_sz_2208_ = lean_array_size(v_bindersElab_2169_);
v___x_2209_ = ((size_t)0ULL);
lean_inc_ref(v_bindersElab_2169_);
v___x_2210_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_Variable_completeBinders_x27_spec__4___redArg(v_sz_2208_, v___x_2209_, v_bindersElab_2169_, v___y_2172_, v___y_2173_, v___y_2174_, v___y_2175_);
if (lean_obj_tag(v___x_2210_) == 0)
{
lean_object* v_a_2211_; lean_object* v___x_2212_; 
v_a_2211_ = lean_ctor_get(v___x_2210_, 0);
lean_inc(v_a_2211_);
lean_dec_ref_known(v___x_2210_, 1);
lean_inc(v___y_2175_);
lean_inc_ref(v___y_2174_);
lean_inc(v___y_2173_);
lean_inc_ref(v___y_2172_);
lean_inc(v___y_2171_);
lean_inc_ref(v___y_2170_);
v___x_2212_ = lean_apply_7(v___f_2156_, v___y_2170_, v___y_2171_, v___y_2172_, v___y_2173_, v___y_2174_, v___y_2175_, lean_box(0));
if (lean_obj_tag(v___x_2212_) == 0)
{
lean_object* v_a_2213_; uint8_t v___x_2214_; 
v_a_2213_ = lean_ctor_get(v___x_2212_, 0);
lean_inc(v_a_2213_);
lean_dec_ref_known(v___x_2212_, 1);
v___x_2214_ = lean_unbox(v_a_2213_);
lean_dec(v_a_2213_);
if (v___x_2214_ == 0)
{
lean_dec(v_a_2211_);
lean_dec(v_cls_2168_);
v___y_2178_ = v___y_2170_;
v___y_2179_ = v___y_2171_;
v___y_2180_ = v___y_2172_;
v___y_2181_ = v___y_2173_;
v___y_2182_ = v___y_2174_;
v___y_2183_ = v___y_2175_;
goto v___jp_2177_;
}
else
{
lean_object* v___x_2215_; lean_object* v___x_2216_; lean_object* v___x_2217_; lean_object* v___x_2218_; lean_object* v___x_2219_; lean_object* v___x_2220_; lean_object* v___x_2221_; 
v___x_2215_ = lean_obj_once(&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__4___closed__1, &lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__4___closed__1_once, _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__4___closed__1);
v___x_2216_ = lean_array_to_list(v_a_2211_);
v___x_2217_ = lean_box(0);
v___x_2218_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Command_Variable_completeBinders_x27_spec__5(v___x_2216_, v___x_2217_);
v___x_2219_ = l_Lean_MessageData_ofList(v___x_2218_);
v___x_2220_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2220_, 0, v___x_2215_);
lean_ctor_set(v___x_2220_, 1, v___x_2219_);
v___x_2221_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg(v_cls_2168_, v___x_2220_, v___y_2172_, v___y_2173_, v___y_2174_, v___y_2175_);
if (lean_obj_tag(v___x_2221_) == 0)
{
lean_dec_ref_known(v___x_2221_, 1);
v___y_2178_ = v___y_2170_;
v___y_2179_ = v___y_2171_;
v___y_2180_ = v___y_2172_;
v___y_2181_ = v___y_2173_;
v___y_2182_ = v___y_2174_;
v___y_2183_ = v___y_2175_;
goto v___jp_2177_;
}
else
{
lean_object* v_a_2222_; lean_object* v___x_2224_; uint8_t v_isShared_2225_; uint8_t v_isSharedCheck_2229_; 
lean_dec_ref(v_bindersElab_2169_);
lean_dec(v_gas_2167_);
lean_dec(v_maxSteps_2166_);
lean_dec(v_i_2165_);
lean_dec(v_binder_2164_);
lean_dec(v___x_2163_);
lean_dec_ref(v_localInstances_2162_);
lean_dec_ref(v_lctx_2161_);
lean_dec_ref(v_binders_2158_);
lean_dec_ref(v_toOmit_2157_);
v_a_2222_ = lean_ctor_get(v___x_2221_, 0);
v_isSharedCheck_2229_ = !lean_is_exclusive(v___x_2221_);
if (v_isSharedCheck_2229_ == 0)
{
v___x_2224_ = v___x_2221_;
v_isShared_2225_ = v_isSharedCheck_2229_;
goto v_resetjp_2223_;
}
else
{
lean_inc(v_a_2222_);
lean_dec(v___x_2221_);
v___x_2224_ = lean_box(0);
v_isShared_2225_ = v_isSharedCheck_2229_;
goto v_resetjp_2223_;
}
v_resetjp_2223_:
{
lean_object* v___x_2227_; 
if (v_isShared_2225_ == 0)
{
v___x_2227_ = v___x_2224_;
goto v_reusejp_2226_;
}
else
{
lean_object* v_reuseFailAlloc_2228_; 
v_reuseFailAlloc_2228_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2228_, 0, v_a_2222_);
v___x_2227_ = v_reuseFailAlloc_2228_;
goto v_reusejp_2226_;
}
v_reusejp_2226_:
{
return v___x_2227_;
}
}
}
}
}
else
{
lean_object* v_a_2230_; lean_object* v___x_2232_; uint8_t v_isShared_2233_; uint8_t v_isSharedCheck_2237_; 
lean_dec(v_a_2211_);
lean_dec_ref(v_bindersElab_2169_);
lean_dec(v_cls_2168_);
lean_dec(v_gas_2167_);
lean_dec(v_maxSteps_2166_);
lean_dec(v_i_2165_);
lean_dec(v_binder_2164_);
lean_dec(v___x_2163_);
lean_dec_ref(v_localInstances_2162_);
lean_dec_ref(v_lctx_2161_);
lean_dec_ref(v_binders_2158_);
lean_dec_ref(v_toOmit_2157_);
v_a_2230_ = lean_ctor_get(v___x_2212_, 0);
v_isSharedCheck_2237_ = !lean_is_exclusive(v___x_2212_);
if (v_isSharedCheck_2237_ == 0)
{
v___x_2232_ = v___x_2212_;
v_isShared_2233_ = v_isSharedCheck_2237_;
goto v_resetjp_2231_;
}
else
{
lean_inc(v_a_2230_);
lean_dec(v___x_2212_);
v___x_2232_ = lean_box(0);
v_isShared_2233_ = v_isSharedCheck_2237_;
goto v_resetjp_2231_;
}
v_resetjp_2231_:
{
lean_object* v___x_2235_; 
if (v_isShared_2233_ == 0)
{
v___x_2235_ = v___x_2232_;
goto v_reusejp_2234_;
}
else
{
lean_object* v_reuseFailAlloc_2236_; 
v_reuseFailAlloc_2236_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2236_, 0, v_a_2230_);
v___x_2235_ = v_reuseFailAlloc_2236_;
goto v_reusejp_2234_;
}
v_reusejp_2234_:
{
return v___x_2235_;
}
}
}
}
else
{
lean_object* v_a_2238_; lean_object* v___x_2240_; uint8_t v_isShared_2241_; uint8_t v_isSharedCheck_2245_; 
lean_dec_ref(v_bindersElab_2169_);
lean_dec(v_cls_2168_);
lean_dec(v_gas_2167_);
lean_dec(v_maxSteps_2166_);
lean_dec(v_i_2165_);
lean_dec(v_binder_2164_);
lean_dec(v___x_2163_);
lean_dec_ref(v_localInstances_2162_);
lean_dec_ref(v_lctx_2161_);
lean_dec_ref(v_binders_2158_);
lean_dec_ref(v_toOmit_2157_);
lean_dec_ref(v___f_2156_);
v_a_2238_ = lean_ctor_get(v___x_2210_, 0);
v_isSharedCheck_2245_ = !lean_is_exclusive(v___x_2210_);
if (v_isSharedCheck_2245_ == 0)
{
v___x_2240_ = v___x_2210_;
v_isShared_2241_ = v_isSharedCheck_2245_;
goto v_resetjp_2239_;
}
else
{
lean_inc(v_a_2238_);
lean_dec(v___x_2210_);
v___x_2240_ = lean_box(0);
v_isShared_2241_ = v_isSharedCheck_2245_;
goto v_resetjp_2239_;
}
v_resetjp_2239_:
{
lean_object* v___x_2243_; 
if (v_isShared_2241_ == 0)
{
v___x_2243_ = v___x_2240_;
goto v_reusejp_2242_;
}
else
{
lean_object* v_reuseFailAlloc_2244_; 
v_reuseFailAlloc_2244_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2244_, 0, v_a_2238_);
v___x_2243_ = v_reuseFailAlloc_2244_;
goto v_reusejp_2242_;
}
v_reusejp_2242_:
{
return v___x_2243_;
}
}
}
v___jp_2177_:
{
uint8_t v___x_2184_; lean_object* v___x_2185_; 
v___x_2184_ = 0;
v___x_2185_ = l_Lean_Elab_Term_synthesizeSyntheticMVarsNoPostponing(v___x_2184_, v___y_2178_, v___y_2179_, v___y_2180_, v___y_2181_, v___y_2182_, v___y_2183_);
if (lean_obj_tag(v___x_2185_) == 0)
{
lean_object* v___x_2186_; lean_object* v___x_2187_; lean_object* v___x_2188_; lean_object* v___x_2189_; lean_object* v___x_2190_; lean_object* v___x_2191_; lean_object* v___f_2192_; lean_object* v___x_2193_; uint8_t v___x_2194_; lean_object* v___x_2195_; lean_object* v___x_2196_; lean_object* v___x_2197_; lean_object* v___f_2198_; lean_object* v___x_2199_; 
lean_dec_ref_known(v___x_2185_, 1);
v___x_2186_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__0));
v___x_2187_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__1));
v___x_2188_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__2));
v___x_2189_ = lean_box(v___x_2184_);
v___x_2190_ = lean_box(v___x_2159_);
v___x_2191_ = lean_box(v_checkRedundant_2160_);
lean_inc(v_i_2165_);
lean_inc_n(v_binder_2164_, 2);
lean_inc(v___x_2163_);
lean_inc_ref(v_binders_2158_);
lean_inc_ref(v_toOmit_2157_);
v___f_2192_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__2___boxed), 23, 14);
lean_closure_set(v___f_2192_, 0, v_bindersElab_2169_);
lean_closure_set(v___f_2192_, 1, v_toOmit_2157_);
lean_closure_set(v___f_2192_, 2, v___x_2189_);
lean_closure_set(v___f_2192_, 3, v_binders_2158_);
lean_closure_set(v___f_2192_, 4, v___x_2190_);
lean_closure_set(v___f_2192_, 5, v___x_2191_);
lean_closure_set(v___f_2192_, 6, v_lctx_2161_);
lean_closure_set(v___f_2192_, 7, v_localInstances_2162_);
lean_closure_set(v___f_2192_, 8, v___x_2163_);
lean_closure_set(v___f_2192_, 9, v_binder_2164_);
lean_closure_set(v___f_2192_, 10, v___x_2186_);
lean_closure_set(v___f_2192_, 11, v___x_2187_);
lean_closure_set(v___f_2192_, 12, v___x_2188_);
lean_closure_set(v___f_2192_, 13, v_i_2165_);
v___x_2193_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_bracketedBinderType___closed__10));
v___x_2194_ = l_Lean_Syntax_isOfKind(v_binder_2164_, v___x_2193_);
v___x_2195_ = lean_box(v_checkRedundant_2160_);
v___x_2196_ = lean_box(v___x_2194_);
v___x_2197_ = lean_box(v___x_2184_);
v___f_2198_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__3___boxed), 18, 11);
lean_closure_set(v___f_2198_, 0, v_i_2165_);
lean_closure_set(v___f_2198_, 1, v_maxSteps_2166_);
lean_closure_set(v___f_2198_, 2, v_gas_2167_);
lean_closure_set(v___f_2198_, 3, v___x_2195_);
lean_closure_set(v___f_2198_, 4, v___x_2196_);
lean_closure_set(v___f_2198_, 5, v_toOmit_2157_);
lean_closure_set(v___f_2198_, 6, v___x_2197_);
lean_closure_set(v___f_2198_, 7, v_binders_2158_);
lean_closure_set(v___f_2198_, 8, v_binder_2164_);
lean_closure_set(v___f_2198_, 9, v___x_2163_);
lean_closure_set(v___f_2198_, 10, v___f_2192_);
v___x_2199_ = l_Lean_Elab_Term_withoutAutoBoundImplicit___redArg(v___f_2198_, v___y_2178_, v___y_2179_, v___y_2180_, v___y_2181_, v___y_2182_, v___y_2183_);
return v___x_2199_;
}
else
{
lean_object* v_a_2200_; lean_object* v___x_2202_; uint8_t v_isShared_2203_; uint8_t v_isSharedCheck_2207_; 
lean_dec_ref(v_bindersElab_2169_);
lean_dec(v_gas_2167_);
lean_dec(v_maxSteps_2166_);
lean_dec(v_i_2165_);
lean_dec(v_binder_2164_);
lean_dec(v___x_2163_);
lean_dec_ref(v_localInstances_2162_);
lean_dec_ref(v_lctx_2161_);
lean_dec_ref(v_binders_2158_);
lean_dec_ref(v_toOmit_2157_);
v_a_2200_ = lean_ctor_get(v___x_2185_, 0);
v_isSharedCheck_2207_ = !lean_is_exclusive(v___x_2185_);
if (v_isSharedCheck_2207_ == 0)
{
v___x_2202_ = v___x_2185_;
v_isShared_2203_ = v_isSharedCheck_2207_;
goto v_resetjp_2201_;
}
else
{
lean_inc(v_a_2200_);
lean_dec(v___x_2185_);
v___x_2202_ = lean_box(0);
v_isShared_2203_ = v_isSharedCheck_2207_;
goto v_resetjp_2201_;
}
v_resetjp_2201_:
{
lean_object* v___x_2205_; 
if (v_isShared_2203_ == 0)
{
v___x_2205_ = v___x_2202_;
goto v_reusejp_2204_;
}
else
{
lean_object* v_reuseFailAlloc_2206_; 
v_reuseFailAlloc_2206_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2206_, 0, v_a_2200_);
v___x_2205_ = v_reuseFailAlloc_2206_;
goto v_reusejp_2204_;
}
v_reusejp_2204_:
{
return v___x_2205_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__4___boxed(lean_object** _args){
lean_object* v___f_2246_ = _args[0];
lean_object* v_toOmit_2247_ = _args[1];
lean_object* v_binders_2248_ = _args[2];
lean_object* v___x_2249_ = _args[3];
lean_object* v_checkRedundant_2250_ = _args[4];
lean_object* v_lctx_2251_ = _args[5];
lean_object* v_localInstances_2252_ = _args[6];
lean_object* v___x_2253_ = _args[7];
lean_object* v_binder_2254_ = _args[8];
lean_object* v_i_2255_ = _args[9];
lean_object* v_maxSteps_2256_ = _args[10];
lean_object* v_gas_2257_ = _args[11];
lean_object* v_cls_2258_ = _args[12];
lean_object* v_bindersElab_2259_ = _args[13];
lean_object* v___y_2260_ = _args[14];
lean_object* v___y_2261_ = _args[15];
lean_object* v___y_2262_ = _args[16];
lean_object* v___y_2263_ = _args[17];
lean_object* v___y_2264_ = _args[18];
lean_object* v___y_2265_ = _args[19];
lean_object* v___y_2266_ = _args[20];
_start:
{
uint8_t v___x_32154__boxed_2267_; uint8_t v_checkRedundant_boxed_2268_; lean_object* v_res_2269_; 
v___x_32154__boxed_2267_ = lean_unbox(v___x_2249_);
v_checkRedundant_boxed_2268_ = lean_unbox(v_checkRedundant_2250_);
v_res_2269_ = lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__4(v___f_2246_, v_toOmit_2247_, v_binders_2248_, v___x_32154__boxed_2267_, v_checkRedundant_boxed_2268_, v_lctx_2251_, v_localInstances_2252_, v___x_2253_, v_binder_2254_, v_i_2255_, v_maxSteps_2256_, v_gas_2257_, v_cls_2258_, v_bindersElab_2259_, v___y_2260_, v___y_2261_, v___y_2262_, v___y_2263_, v___y_2264_, v___y_2265_);
lean_dec(v___y_2265_);
lean_dec_ref(v___y_2264_);
lean_dec(v___y_2263_);
lean_dec_ref(v___y_2262_);
lean_dec(v___y_2261_);
lean_dec_ref(v___y_2260_);
return v_res_2269_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__23(void){
_start:
{
lean_object* v___x_2273_; lean_object* v___x_2274_; lean_object* v___x_2275_; lean_object* v___x_2276_; lean_object* v___x_2277_; lean_object* v___x_2278_; 
v___x_2273_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__22));
v___x_2274_ = lean_unsigned_to_nat(14u);
v___x_2275_ = lean_unsigned_to_nat(22u);
v___x_2276_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__21));
v___x_2277_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__20));
v___x_2278_ = l_mkPanicMessageWithDecl(v___x_2277_, v___x_2276_, v___x_2275_, v___x_2274_, v___x_2273_);
return v___x_2278_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__25(void){
_start:
{
lean_object* v___x_2280_; lean_object* v___x_2281_; 
v___x_2280_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__24));
v___x_2281_ = l_Lean_stringToMessageData(v___x_2280_);
return v___x_2281_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__27(void){
_start:
{
lean_object* v___x_2283_; lean_object* v___x_2284_; 
v___x_2283_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__26));
v___x_2284_ = l_Lean_stringToMessageData(v___x_2283_);
return v___x_2284_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__29(void){
_start:
{
lean_object* v___x_2286_; lean_object* v___x_2287_; 
v___x_2286_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__28));
v___x_2287_ = l_Lean_stringToMessageData(v___x_2286_);
return v___x_2287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27(lean_object* v_maxSteps_2288_, lean_object* v_gas_2289_, uint8_t v_checkRedundant_2290_, lean_object* v_binders_2291_, lean_object* v_toOmit_2292_, lean_object* v_i_2293_, lean_object* v_a_2294_, lean_object* v_a_2295_, lean_object* v_a_2296_, lean_object* v_a_2297_, lean_object* v_a_2298_, lean_object* v_a_2299_){
_start:
{
lean_object* v___y_2305_; lean_object* v___y_2306_; lean_object* v___y_2307_; lean_object* v___y_2308_; lean_object* v___y_2309_; lean_object* v___y_2310_; uint8_t v___y_2311_; lean_object* v___y_2312_; lean_object* v___y_2313_; lean_object* v___y_2332_; lean_object* v___y_2333_; lean_object* v___y_2334_; lean_object* v___y_2335_; lean_object* v___y_2336_; lean_object* v___y_2337_; lean_object* v___y_2338_; lean_object* v___y_2339_; uint8_t v___y_2340_; uint8_t v___y_2341_; lean_object* v___x_2362_; uint8_t v___x_2399_; 
v___x_2362_ = lean_unsigned_to_nat(0u);
v___x_2399_ = lean_nat_dec_lt(v___x_2362_, v_gas_2289_);
if (v___x_2399_ == 0)
{
goto v___jp_2363_;
}
else
{
lean_object* v___x_2400_; lean_object* v___y_2402_; lean_object* v___y_2403_; lean_object* v___y_2404_; lean_object* v___y_2405_; lean_object* v___y_2406_; lean_object* v___y_2407_; lean_object* v___y_2408_; uint8_t v___x_2414_; 
v___x_2400_ = lean_array_get_size(v_binders_2291_);
v___x_2414_ = lean_nat_dec_lt(v_i_2293_, v___x_2400_);
if (v___x_2414_ == 0)
{
goto v___jp_2363_;
}
else
{
lean_object* v_cls_2415_; lean_object* v___f_2416_; lean_object* v___x_2417_; 
v_cls_2415_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__1_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_));
v___f_2416_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__9));
v___x_2417_ = lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__0(v_cls_2415_, v_a_2294_, v_a_2295_, v_a_2296_, v_a_2297_, v_a_2298_, v_a_2299_);
if (lean_obj_tag(v___x_2417_) == 0)
{
lean_object* v_a_2418_; lean_object* v_binder_2419_; lean_object* v___y_2421_; lean_object* v___y_2422_; lean_object* v___y_2423_; lean_object* v___y_2424_; lean_object* v___y_2425_; lean_object* v___y_2426_; lean_object* v___y_2427_; lean_object* v___y_2428_; lean_object* v___y_2429_; lean_object* v___y_2473_; lean_object* v___y_2474_; lean_object* v___y_2475_; lean_object* v___y_2476_; lean_object* v___y_2477_; lean_object* v___y_2478_; lean_object* v___y_2479_; lean_object* v___y_2480_; lean_object* v___y_2484_; lean_object* v___y_2485_; lean_object* v___y_2486_; lean_object* v___y_2487_; lean_object* v___y_2488_; lean_object* v___y_2489_; lean_object* v___y_2490_; lean_object* v___y_2549_; lean_object* v___y_2550_; lean_object* v___y_2551_; lean_object* v___y_2552_; lean_object* v___y_2553_; lean_object* v___y_2554_; uint8_t v___x_2559_; 
v_a_2418_ = lean_ctor_get(v___x_2417_, 0);
lean_inc(v_a_2418_);
lean_dec_ref_known(v___x_2417_, 1);
v_binder_2419_ = lean_array_fget_borrowed(v_binders_2291_, v_i_2293_);
v___x_2559_ = lean_unbox(v_a_2418_);
lean_dec(v_a_2418_);
if (v___x_2559_ == 0)
{
v___y_2549_ = v_a_2294_;
v___y_2550_ = v_a_2295_;
v___y_2551_ = v_a_2296_;
v___y_2552_ = v_a_2297_;
v___y_2553_ = v_a_2298_;
v___y_2554_ = v_a_2299_;
goto v___jp_2548_;
}
else
{
lean_object* v_lctx_2560_; lean_object* v_localInstances_2561_; lean_object* v___x_2562_; lean_object* v___x_2563_; lean_object* v___x_2564_; lean_object* v___x_2565_; lean_object* v___x_2566_; lean_object* v___x_2567_; lean_object* v___x_2568_; lean_object* v___x_2569_; lean_object* v___x_2570_; lean_object* v___x_2571_; lean_object* v___x_2572_; lean_object* v___x_2573_; lean_object* v___x_2574_; lean_object* v___x_2575_; lean_object* v___x_2576_; lean_object* v___x_2577_; lean_object* v___x_2578_; lean_object* v___x_2579_; lean_object* v___x_2580_; lean_object* v___x_2581_; 
v_lctx_2560_ = lean_ctor_get(v_a_2296_, 2);
v_localInstances_2561_ = lean_ctor_get(v_a_2296_, 3);
v___x_2562_ = lean_obj_once(&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__25, &lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__25_once, _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__25);
v___x_2563_ = l_Lean_LocalContext_getFVarIds(v_lctx_2560_);
v___x_2564_ = lean_array_get_size(v___x_2563_);
lean_dec_ref(v___x_2563_);
v___x_2565_ = l_Nat_reprFast(v___x_2564_);
v___x_2566_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2566_, 0, v___x_2565_);
v___x_2567_ = l_Lean_MessageData_ofFormat(v___x_2566_);
v___x_2568_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2568_, 0, v___x_2562_);
lean_ctor_set(v___x_2568_, 1, v___x_2567_);
v___x_2569_ = lean_obj_once(&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__27, &lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__27_once, _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__27);
v___x_2570_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2570_, 0, v___x_2568_);
lean_ctor_set(v___x_2570_, 1, v___x_2569_);
v___x_2571_ = lean_array_get_size(v_localInstances_2561_);
v___x_2572_ = l_Nat_reprFast(v___x_2571_);
v___x_2573_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2573_, 0, v___x_2572_);
v___x_2574_ = l_Lean_MessageData_ofFormat(v___x_2573_);
v___x_2575_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2575_, 0, v___x_2570_);
lean_ctor_set(v___x_2575_, 1, v___x_2574_);
v___x_2576_ = lean_obj_once(&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__29, &lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__29_once, _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__29);
v___x_2577_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2577_, 0, v___x_2575_);
lean_ctor_set(v___x_2577_, 1, v___x_2576_);
lean_inc(v_binder_2419_);
v___x_2578_ = l_Lean_MessageData_ofSyntax(v_binder_2419_);
v___x_2579_ = l_Lean_indentD(v___x_2578_);
v___x_2580_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2580_, 0, v___x_2577_);
lean_ctor_set(v___x_2580_, 1, v___x_2579_);
v___x_2581_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg(v_cls_2415_, v___x_2580_, v_a_2296_, v_a_2297_, v_a_2298_, v_a_2299_);
if (lean_obj_tag(v___x_2581_) == 0)
{
lean_dec_ref_known(v___x_2581_, 1);
v___y_2549_ = v_a_2294_;
v___y_2550_ = v_a_2295_;
v___y_2551_ = v_a_2296_;
v___y_2552_ = v_a_2297_;
v___y_2553_ = v_a_2298_;
v___y_2554_ = v_a_2299_;
goto v___jp_2548_;
}
else
{
lean_object* v_a_2582_; lean_object* v___x_2584_; uint8_t v_isShared_2585_; uint8_t v_isSharedCheck_2589_; 
lean_dec(v_i_2293_);
lean_dec_ref(v_toOmit_2292_);
lean_dec_ref(v_binders_2291_);
lean_dec(v_gas_2289_);
lean_dec(v_maxSteps_2288_);
v_a_2582_ = lean_ctor_get(v___x_2581_, 0);
v_isSharedCheck_2589_ = !lean_is_exclusive(v___x_2581_);
if (v_isSharedCheck_2589_ == 0)
{
v___x_2584_ = v___x_2581_;
v_isShared_2585_ = v_isSharedCheck_2589_;
goto v_resetjp_2583_;
}
else
{
lean_inc(v_a_2582_);
lean_dec(v___x_2581_);
v___x_2584_ = lean_box(0);
v_isShared_2585_ = v_isSharedCheck_2589_;
goto v_resetjp_2583_;
}
v_resetjp_2583_:
{
lean_object* v___x_2587_; 
if (v_isShared_2585_ == 0)
{
v___x_2587_ = v___x_2584_;
goto v_reusejp_2586_;
}
else
{
lean_object* v_reuseFailAlloc_2588_; 
v_reuseFailAlloc_2588_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2588_, 0, v_a_2582_);
v___x_2587_ = v_reuseFailAlloc_2588_;
goto v_reusejp_2586_;
}
v_reusejp_2586_:
{
return v___x_2587_;
}
}
}
}
v___jp_2420_:
{
uint8_t v___x_2430_; 
v___x_2430_ = lean_nat_dec_lt(v___x_2362_, v___y_2429_);
if (v___x_2430_ == 0)
{
lean_dec(v___y_2429_);
lean_dec_ref(v___y_2424_);
v___y_2402_ = v___y_2426_;
v___y_2403_ = v___y_2428_;
v___y_2404_ = v___y_2427_;
v___y_2405_ = v___y_2422_;
v___y_2406_ = v___y_2425_;
v___y_2407_ = v___y_2423_;
v___y_2408_ = v___y_2421_;
goto v___jp_2401_;
}
else
{
size_t v___x_2431_; size_t v___x_2432_; uint8_t v___x_2433_; 
v___x_2431_ = ((size_t)0ULL);
v___x_2432_ = lean_usize_of_nat(v___y_2429_);
lean_dec(v___y_2429_);
v___x_2433_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Mathlib_Command_Variable_completeBinders_x27_spec__1(v___y_2426_, v_binders_2291_, v___x_2431_, v___x_2432_);
if (v___x_2433_ == 0)
{
lean_dec_ref(v___y_2424_);
v___y_2402_ = v___y_2426_;
v___y_2403_ = v___y_2428_;
v___y_2404_ = v___y_2427_;
v___y_2405_ = v___y_2422_;
v___y_2406_ = v___y_2425_;
v___y_2407_ = v___y_2423_;
v___y_2408_ = v___y_2421_;
goto v___jp_2401_;
}
else
{
lean_object* v_ref_2434_; lean_object* v___x_2435_; uint8_t v___x_2436_; lean_object* v___x_2437_; lean_object* v___x_2438_; lean_object* v___x_2439_; lean_object* v___x_2440_; lean_object* v___x_2441_; lean_object* v___x_2442_; lean_object* v___x_2443_; lean_object* v___x_2444_; lean_object* v___x_2445_; lean_object* v___x_2446_; lean_object* v___x_2447_; lean_object* v___x_2448_; lean_object* v___x_2449_; lean_object* v___x_2450_; lean_object* v___x_2451_; lean_object* v___x_2452_; lean_object* v___x_2453_; lean_object* v___x_2454_; lean_object* v___x_2455_; lean_object* v___x_2456_; lean_object* v___x_2457_; lean_object* v___x_2458_; lean_object* v___x_2459_; lean_object* v___x_2460_; lean_object* v___x_2461_; lean_object* v___x_2462_; lean_object* v___x_2463_; 
v_ref_2434_ = lean_ctor_get(v___y_2423_, 5);
lean_inc(v_i_2293_);
v___x_2435_ = l_Array_extract___redArg(v_binders_2291_, v___x_2362_, v_i_2293_);
v___x_2436_ = 0;
v___x_2437_ = l_Lean_SourceInfo_fromRef(v_ref_2434_, v___x_2436_);
v___x_2438_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__3));
v___x_2439_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__4));
lean_inc_n(v___x_2437_, 2);
v___x_2440_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2440_, 0, v___x_2437_);
lean_ctor_set(v___x_2440_, 1, v___x_2438_);
v___x_2441_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__2));
v___x_2442_ = lean_obj_once(&lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__3, &lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__3_once, _init_lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__3);
v___x_2443_ = l_Array_append___redArg(v___x_2442_, v___x_2435_);
lean_dec_ref(v___x_2435_);
v___x_2444_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2444_, 0, v___x_2437_);
lean_ctor_set(v___x_2444_, 1, v___x_2441_);
lean_ctor_set(v___x_2444_, 2, v___x_2443_);
v___x_2445_ = l_Lean_Syntax_node2(v___x_2437_, v___x_2439_, v___x_2440_, v___x_2444_);
v___x_2446_ = lean_obj_once(&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__11, &lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__11_once, _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__11);
lean_inc(v_binder_2419_);
v___x_2447_ = l_Lean_MessageData_ofSyntax(v_binder_2419_);
v___x_2448_ = l_Lean_indentD(v___x_2447_);
v___x_2449_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2449_, 0, v___x_2446_);
lean_ctor_set(v___x_2449_, 1, v___x_2448_);
v___x_2450_ = lean_obj_once(&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__13, &lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__13_once, _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__13);
v___x_2451_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2451_, 0, v___x_2449_);
lean_ctor_set(v___x_2451_, 1, v___x_2450_);
lean_inc(v___y_2426_);
v___x_2452_ = l_Lean_MessageData_ofSyntax(v___y_2426_);
v___x_2453_ = l_Lean_indentD(v___x_2452_);
v___x_2454_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2454_, 0, v___x_2451_);
lean_ctor_set(v___x_2454_, 1, v___x_2453_);
v___x_2455_ = lean_obj_once(&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__15, &lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__15_once, _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__15);
v___x_2456_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2456_, 0, v___x_2454_);
lean_ctor_set(v___x_2456_, 1, v___x_2455_);
v___x_2457_ = l_Lean_MessageData_ofSyntax(v___x_2445_);
v___x_2458_ = l_Lean_indentD(v___x_2457_);
v___x_2459_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2459_, 0, v___x_2456_);
lean_ctor_set(v___x_2459_, 1, v___x_2458_);
v___x_2460_ = lean_obj_once(&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__17, &lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__17_once, _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__17);
v___x_2461_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2461_, 0, v___x_2459_);
lean_ctor_set(v___x_2461_, 1, v___x_2460_);
v___x_2462_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2462_, 0, v___x_2461_);
lean_ctor_set(v___x_2462_, 1, v___y_2424_);
v___x_2463_ = lp_mathlib_Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2___redArg(v_binder_2419_, v___x_2462_, v___y_2428_, v___y_2427_, v___y_2422_, v___y_2425_, v___y_2423_, v___y_2421_);
if (lean_obj_tag(v___x_2463_) == 0)
{
lean_dec_ref_known(v___x_2463_, 1);
v___y_2402_ = v___y_2426_;
v___y_2403_ = v___y_2428_;
v___y_2404_ = v___y_2427_;
v___y_2405_ = v___y_2422_;
v___y_2406_ = v___y_2425_;
v___y_2407_ = v___y_2423_;
v___y_2408_ = v___y_2421_;
goto v___jp_2401_;
}
else
{
lean_object* v_a_2464_; lean_object* v___x_2466_; uint8_t v_isShared_2467_; uint8_t v_isSharedCheck_2471_; 
lean_dec(v___y_2426_);
lean_dec(v_i_2293_);
lean_dec_ref(v_toOmit_2292_);
lean_dec_ref(v_binders_2291_);
lean_dec(v_gas_2289_);
lean_dec(v_maxSteps_2288_);
v_a_2464_ = lean_ctor_get(v___x_2463_, 0);
v_isSharedCheck_2471_ = !lean_is_exclusive(v___x_2463_);
if (v_isSharedCheck_2471_ == 0)
{
v___x_2466_ = v___x_2463_;
v_isShared_2467_ = v_isSharedCheck_2471_;
goto v_resetjp_2465_;
}
else
{
lean_inc(v_a_2464_);
lean_dec(v___x_2463_);
v___x_2466_ = lean_box(0);
v_isShared_2467_ = v_isSharedCheck_2471_;
goto v_resetjp_2465_;
}
v_resetjp_2465_:
{
lean_object* v___x_2469_; 
if (v_isShared_2467_ == 0)
{
v___x_2469_ = v___x_2466_;
goto v_reusejp_2468_;
}
else
{
lean_object* v_reuseFailAlloc_2470_; 
v_reuseFailAlloc_2470_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2470_, 0, v_a_2464_);
v___x_2469_ = v_reuseFailAlloc_2470_;
goto v_reusejp_2468_;
}
v_reusejp_2468_:
{
return v___x_2469_;
}
}
}
}
}
}
v___jp_2472_:
{
uint8_t v___x_2481_; 
v___x_2481_ = lean_nat_dec_lt(v___x_2362_, v_i_2293_);
if (v___x_2481_ == 0)
{
lean_dec_ref(v___y_2473_);
v___y_2402_ = v___y_2474_;
v___y_2403_ = v___y_2475_;
v___y_2404_ = v___y_2476_;
v___y_2405_ = v___y_2477_;
v___y_2406_ = v___y_2478_;
v___y_2407_ = v___y_2479_;
v___y_2408_ = v___y_2480_;
goto v___jp_2401_;
}
else
{
uint8_t v___x_2482_; 
v___x_2482_ = lean_nat_dec_le(v_i_2293_, v___x_2400_);
if (v___x_2482_ == 0)
{
v___y_2421_ = v___y_2480_;
v___y_2422_ = v___y_2477_;
v___y_2423_ = v___y_2479_;
v___y_2424_ = v___y_2473_;
v___y_2425_ = v___y_2478_;
v___y_2426_ = v___y_2474_;
v___y_2427_ = v___y_2476_;
v___y_2428_ = v___y_2475_;
v___y_2429_ = v___x_2400_;
goto v___jp_2420_;
}
else
{
lean_inc(v_i_2293_);
v___y_2421_ = v___y_2480_;
v___y_2422_ = v___y_2477_;
v___y_2423_ = v___y_2479_;
v___y_2424_ = v___y_2473_;
v___y_2425_ = v___y_2478_;
v___y_2426_ = v___y_2474_;
v___y_2427_ = v___y_2476_;
v___y_2428_ = v___y_2475_;
v___y_2429_ = v_i_2293_;
goto v___jp_2420_;
}
}
}
v___jp_2483_:
{
lean_object* v___x_2491_; 
lean_inc(v_binder_2419_);
v___x_2491_ = lp_mathlib_Mathlib_Command_Variable_getSubproblem(v_binder_2419_, v___y_2490_, v___y_2486_, v___y_2485_, v___y_2484_, v___y_2488_, v___y_2487_, v___y_2489_);
if (lean_obj_tag(v___x_2491_) == 0)
{
lean_object* v_a_2492_; 
v_a_2492_ = lean_ctor_get(v___x_2491_, 0);
lean_inc(v_a_2492_);
lean_dec_ref_known(v___x_2491_, 1);
if (lean_obj_tag(v_a_2492_) == 1)
{
lean_object* v_val_2493_; lean_object* v_options_2494_; uint8_t v_hasTrace_2495_; 
v_val_2493_ = lean_ctor_get(v_a_2492_, 0);
lean_inc(v_val_2493_);
lean_dec_ref_known(v_a_2492_, 1);
v_options_2494_ = lean_ctor_get(v___y_2487_, 2);
v_hasTrace_2495_ = lean_ctor_get_uint8(v_options_2494_, sizeof(void*)*1);
if (v_hasTrace_2495_ == 0)
{
lean_object* v_fst_2496_; lean_object* v_snd_2497_; 
v_fst_2496_ = lean_ctor_get(v_val_2493_, 0);
lean_inc(v_fst_2496_);
v_snd_2497_ = lean_ctor_get(v_val_2493_, 1);
lean_inc(v_snd_2497_);
lean_dec(v_val_2493_);
v___y_2473_ = v_fst_2496_;
v___y_2474_ = v_snd_2497_;
v___y_2475_ = v___y_2486_;
v___y_2476_ = v___y_2485_;
v___y_2477_ = v___y_2484_;
v___y_2478_ = v___y_2488_;
v___y_2479_ = v___y_2487_;
v___y_2480_ = v___y_2489_;
goto v___jp_2472_;
}
else
{
lean_object* v_fst_2498_; lean_object* v_snd_2499_; lean_object* v___x_2501_; uint8_t v_isShared_2502_; uint8_t v_isSharedCheck_2521_; 
v_fst_2498_ = lean_ctor_get(v_val_2493_, 0);
v_snd_2499_ = lean_ctor_get(v_val_2493_, 1);
v_isSharedCheck_2521_ = !lean_is_exclusive(v_val_2493_);
if (v_isSharedCheck_2521_ == 0)
{
v___x_2501_ = v_val_2493_;
v_isShared_2502_ = v_isSharedCheck_2521_;
goto v_resetjp_2500_;
}
else
{
lean_inc(v_snd_2499_);
lean_inc(v_fst_2498_);
lean_dec(v_val_2493_);
v___x_2501_ = lean_box(0);
v_isShared_2502_ = v_isSharedCheck_2521_;
goto v_resetjp_2500_;
}
v_resetjp_2500_:
{
lean_object* v_inheritedTraceOptions_2503_; lean_object* v___x_2504_; uint8_t v___x_2505_; 
v_inheritedTraceOptions_2503_ = lean_ctor_get(v___y_2487_, 13);
v___x_2504_ = lean_obj_once(&lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__8, &lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__8_once, _init_lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__8);
v___x_2505_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2503_, v_options_2494_, v___x_2504_);
if (v___x_2505_ == 0)
{
lean_del_object(v___x_2501_);
v___y_2473_ = v_fst_2498_;
v___y_2474_ = v_snd_2499_;
v___y_2475_ = v___y_2486_;
v___y_2476_ = v___y_2485_;
v___y_2477_ = v___y_2484_;
v___y_2478_ = v___y_2488_;
v___y_2479_ = v___y_2487_;
v___y_2480_ = v___y_2489_;
goto v___jp_2472_;
}
else
{
lean_object* v___x_2506_; lean_object* v___x_2507_; lean_object* v___x_2508_; lean_object* v___x_2510_; 
v___x_2506_ = lean_obj_once(&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__19, &lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__19_once, _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__19);
lean_inc(v_snd_2499_);
v___x_2507_ = l_Lean_MessageData_ofSyntax(v_snd_2499_);
v___x_2508_ = l_Lean_indentD(v___x_2507_);
if (v_isShared_2502_ == 0)
{
lean_ctor_set_tag(v___x_2501_, 7);
lean_ctor_set(v___x_2501_, 1, v___x_2508_);
lean_ctor_set(v___x_2501_, 0, v___x_2506_);
v___x_2510_ = v___x_2501_;
goto v_reusejp_2509_;
}
else
{
lean_object* v_reuseFailAlloc_2520_; 
v_reuseFailAlloc_2520_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2520_, 0, v___x_2506_);
lean_ctor_set(v_reuseFailAlloc_2520_, 1, v___x_2508_);
v___x_2510_ = v_reuseFailAlloc_2520_;
goto v_reusejp_2509_;
}
v_reusejp_2509_:
{
lean_object* v___x_2511_; 
v___x_2511_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg(v_cls_2415_, v___x_2510_, v___y_2484_, v___y_2488_, v___y_2487_, v___y_2489_);
if (lean_obj_tag(v___x_2511_) == 0)
{
lean_dec_ref_known(v___x_2511_, 1);
v___y_2473_ = v_fst_2498_;
v___y_2474_ = v_snd_2499_;
v___y_2475_ = v___y_2486_;
v___y_2476_ = v___y_2485_;
v___y_2477_ = v___y_2484_;
v___y_2478_ = v___y_2488_;
v___y_2479_ = v___y_2487_;
v___y_2480_ = v___y_2489_;
goto v___jp_2472_;
}
else
{
lean_object* v_a_2512_; lean_object* v___x_2514_; uint8_t v_isShared_2515_; uint8_t v_isSharedCheck_2519_; 
lean_dec(v_snd_2499_);
lean_dec(v_fst_2498_);
lean_dec(v_i_2293_);
lean_dec_ref(v_toOmit_2292_);
lean_dec_ref(v_binders_2291_);
lean_dec(v_gas_2289_);
lean_dec(v_maxSteps_2288_);
v_a_2512_ = lean_ctor_get(v___x_2511_, 0);
v_isSharedCheck_2519_ = !lean_is_exclusive(v___x_2511_);
if (v_isSharedCheck_2519_ == 0)
{
v___x_2514_ = v___x_2511_;
v_isShared_2515_ = v_isSharedCheck_2519_;
goto v_resetjp_2513_;
}
else
{
lean_inc(v_a_2512_);
lean_dec(v___x_2511_);
v___x_2514_ = lean_box(0);
v_isShared_2515_ = v_isSharedCheck_2519_;
goto v_resetjp_2513_;
}
v_resetjp_2513_:
{
lean_object* v___x_2517_; 
if (v_isShared_2515_ == 0)
{
v___x_2517_ = v___x_2514_;
goto v_reusejp_2516_;
}
else
{
lean_object* v_reuseFailAlloc_2518_; 
v_reuseFailAlloc_2518_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2518_, 0, v_a_2512_);
v___x_2517_ = v_reuseFailAlloc_2518_;
goto v_reusejp_2516_;
}
v_reusejp_2516_:
{
return v___x_2517_;
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
lean_object* v___x_2522_; lean_object* v_lctx_2523_; lean_object* v_localInstances_2524_; lean_object* v_options_2525_; lean_object* v_env_2526_; lean_object* v___x_2527_; lean_object* v___x_2528_; lean_object* v___f_2529_; lean_object* v___x_2530_; lean_object* v___x_2531_; lean_object* v___x_2532_; lean_object* v___x_2533_; lean_object* v___x_2534_; uint8_t v___x_2535_; lean_object* v___x_2536_; lean_object* v___x_2537_; uint8_t v___x_2538_; uint8_t v___x_2539_; 
lean_inc_n(v_binder_2419_, 2);
lean_dec(v_a_2492_);
v___x_2522_ = lean_st_ref_get(v___y_2489_);
v_lctx_2523_ = lean_ctor_get(v___y_2484_, 2);
v_localInstances_2524_ = lean_ctor_get(v___y_2484_, 3);
v_options_2525_ = lean_ctor_get(v___y_2487_, 2);
v_env_2526_ = lean_ctor_get(v___x_2522_, 0);
lean_inc_ref(v_env_2526_);
lean_dec(v___x_2522_);
v___x_2527_ = lean_box(v___x_2414_);
v___x_2528_ = lean_box(v_checkRedundant_2290_);
lean_inc_ref(v_localInstances_2524_);
lean_inc_ref(v_lctx_2523_);
v___f_2529_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__4___boxed), 21, 13);
lean_closure_set(v___f_2529_, 0, v___f_2416_);
lean_closure_set(v___f_2529_, 1, v_toOmit_2292_);
lean_closure_set(v___f_2529_, 2, v_binders_2291_);
lean_closure_set(v___f_2529_, 3, v___x_2527_);
lean_closure_set(v___f_2529_, 4, v___x_2528_);
lean_closure_set(v___f_2529_, 5, v_lctx_2523_);
lean_closure_set(v___f_2529_, 6, v_localInstances_2524_);
lean_closure_set(v___f_2529_, 7, v___x_2362_);
lean_closure_set(v___f_2529_, 8, v_binder_2419_);
lean_closure_set(v___f_2529_, 9, v_i_2293_);
lean_closure_set(v___f_2529_, 10, v_maxSteps_2288_);
lean_closure_set(v___f_2529_, 11, v_gas_2289_);
lean_closure_set(v___f_2529_, 12, v_cls_2415_);
v___x_2530_ = lean_unsigned_to_nat(1u);
v___x_2531_ = lean_mk_empty_array_with_capacity(v___x_2530_);
v___x_2532_ = lean_array_push(v___x_2531_, v_binder_2419_);
v___x_2533_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_elabBinders___boxed), 10, 3);
lean_closure_set(v___x_2533_, 0, lean_box(0));
lean_closure_set(v___x_2533_, 1, v___x_2532_);
lean_closure_set(v___x_2533_, 2, v___f_2529_);
v___x_2534_ = l_Lean_Elab_Term_checkBinderAnnotations;
v___x_2535_ = 0;
lean_inc_ref(v_options_2525_);
v___x_2536_ = lp_mathlib_Lean_Option_set___at___00Mathlib_Command_Variable_completeBinders_x27_spec__6(v_options_2525_, v___x_2534_, v___x_2535_);
v___x_2537_ = l_Lean_diagnostics;
v___x_2538_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__5(v___x_2536_, v___x_2537_);
v___x_2539_ = l_Lean_Kernel_isDiagnosticsEnabled(v_env_2526_);
lean_dec_ref(v_env_2526_);
if (v___x_2539_ == 0)
{
if (v___x_2538_ == 0)
{
v___y_2332_ = v___y_2485_;
v___y_2333_ = v___y_2484_;
v___y_2334_ = v___x_2536_;
v___y_2335_ = v___y_2486_;
v___y_2336_ = v___y_2487_;
v___y_2337_ = v___y_2488_;
v___y_2338_ = v___y_2489_;
v___y_2339_ = v___x_2533_;
v___y_2340_ = v___x_2538_;
v___y_2341_ = v___x_2414_;
goto v___jp_2331_;
}
else
{
v___y_2332_ = v___y_2485_;
v___y_2333_ = v___y_2484_;
v___y_2334_ = v___x_2536_;
v___y_2335_ = v___y_2486_;
v___y_2336_ = v___y_2487_;
v___y_2337_ = v___y_2488_;
v___y_2338_ = v___y_2489_;
v___y_2339_ = v___x_2533_;
v___y_2340_ = v___x_2538_;
v___y_2341_ = v___x_2539_;
goto v___jp_2331_;
}
}
else
{
v___y_2332_ = v___y_2485_;
v___y_2333_ = v___y_2484_;
v___y_2334_ = v___x_2536_;
v___y_2335_ = v___y_2486_;
v___y_2336_ = v___y_2487_;
v___y_2337_ = v___y_2488_;
v___y_2338_ = v___y_2489_;
v___y_2339_ = v___x_2533_;
v___y_2340_ = v___x_2538_;
v___y_2341_ = v___x_2538_;
goto v___jp_2331_;
}
}
}
else
{
lean_object* v_a_2540_; lean_object* v___x_2542_; uint8_t v_isShared_2543_; uint8_t v_isSharedCheck_2547_; 
lean_dec(v_i_2293_);
lean_dec_ref(v_toOmit_2292_);
lean_dec_ref(v_binders_2291_);
lean_dec(v_gas_2289_);
lean_dec(v_maxSteps_2288_);
v_a_2540_ = lean_ctor_get(v___x_2491_, 0);
v_isSharedCheck_2547_ = !lean_is_exclusive(v___x_2491_);
if (v_isSharedCheck_2547_ == 0)
{
v___x_2542_ = v___x_2491_;
v_isShared_2543_ = v_isSharedCheck_2547_;
goto v_resetjp_2541_;
}
else
{
lean_inc(v_a_2540_);
lean_dec(v___x_2491_);
v___x_2542_ = lean_box(0);
v_isShared_2543_ = v_isSharedCheck_2547_;
goto v_resetjp_2541_;
}
v_resetjp_2541_:
{
lean_object* v___x_2545_; 
if (v_isShared_2543_ == 0)
{
v___x_2545_ = v___x_2542_;
goto v_reusejp_2544_;
}
else
{
lean_object* v_reuseFailAlloc_2546_; 
v_reuseFailAlloc_2546_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2546_, 0, v_a_2540_);
v___x_2545_ = v_reuseFailAlloc_2546_;
goto v_reusejp_2544_;
}
v_reusejp_2544_:
{
return v___x_2545_;
}
}
}
}
v___jp_2548_:
{
lean_object* v___x_2555_; 
lean_inc(v_binder_2419_);
v___x_2555_ = lp_mathlib_Mathlib_Command_Variable_bracketedBinderType(v_binder_2419_);
if (lean_obj_tag(v___x_2555_) == 0)
{
lean_object* v___x_2556_; lean_object* v___x_2557_; 
v___x_2556_ = lean_obj_once(&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__23, &lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__23_once, _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__23);
v___x_2557_ = lp_mathlib_panic___at___00Mathlib_Command_Variable_completeBinders_x27_spec__8(v___x_2556_);
v___y_2484_ = v___y_2551_;
v___y_2485_ = v___y_2550_;
v___y_2486_ = v___y_2549_;
v___y_2487_ = v___y_2553_;
v___y_2488_ = v___y_2552_;
v___y_2489_ = v___y_2554_;
v___y_2490_ = v___x_2557_;
goto v___jp_2483_;
}
else
{
lean_object* v_val_2558_; 
v_val_2558_ = lean_ctor_get(v___x_2555_, 0);
lean_inc(v_val_2558_);
lean_dec_ref_known(v___x_2555_, 1);
v___y_2484_ = v___y_2551_;
v___y_2485_ = v___y_2550_;
v___y_2486_ = v___y_2549_;
v___y_2487_ = v___y_2553_;
v___y_2488_ = v___y_2552_;
v___y_2489_ = v___y_2554_;
v___y_2490_ = v_val_2558_;
goto v___jp_2483_;
}
}
}
else
{
lean_object* v_a_2590_; lean_object* v___x_2592_; uint8_t v_isShared_2593_; uint8_t v_isSharedCheck_2597_; 
lean_dec(v_i_2293_);
lean_dec_ref(v_toOmit_2292_);
lean_dec_ref(v_binders_2291_);
lean_dec(v_gas_2289_);
lean_dec(v_maxSteps_2288_);
v_a_2590_ = lean_ctor_get(v___x_2417_, 0);
v_isSharedCheck_2597_ = !lean_is_exclusive(v___x_2417_);
if (v_isSharedCheck_2597_ == 0)
{
v___x_2592_ = v___x_2417_;
v_isShared_2593_ = v_isSharedCheck_2597_;
goto v_resetjp_2591_;
}
else
{
lean_inc(v_a_2590_);
lean_dec(v___x_2417_);
v___x_2592_ = lean_box(0);
v_isShared_2593_ = v_isSharedCheck_2597_;
goto v_resetjp_2591_;
}
v_resetjp_2591_:
{
lean_object* v___x_2595_; 
if (v_isShared_2593_ == 0)
{
v___x_2595_ = v___x_2592_;
goto v_reusejp_2594_;
}
else
{
lean_object* v_reuseFailAlloc_2596_; 
v_reuseFailAlloc_2596_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2596_, 0, v_a_2590_);
v___x_2595_ = v_reuseFailAlloc_2596_;
goto v_reusejp_2594_;
}
v_reusejp_2594_:
{
return v___x_2595_;
}
}
}
}
v___jp_2401_:
{
lean_object* v_as_2409_; lean_object* v___x_2410_; lean_object* v___x_2411_; lean_object* v___x_2412_; 
v_as_2409_ = lean_array_push(v_binders_2291_, v___y_2402_);
v___x_2410_ = l___private_Init_Data_Array_Basic_0__Array_insertIdx_loop(lean_box(0), v_i_2293_, v_as_2409_, v___x_2400_);
v___x_2411_ = lean_unsigned_to_nat(1u);
v___x_2412_ = lean_nat_sub(v_gas_2289_, v___x_2411_);
lean_dec(v_gas_2289_);
v_gas_2289_ = v___x_2412_;
v_binders_2291_ = v___x_2410_;
v_a_2294_ = v___y_2403_;
v_a_2295_ = v___y_2404_;
v_a_2296_ = v___y_2405_;
v_a_2297_ = v___y_2406_;
v_a_2298_ = v___y_2407_;
v_a_2299_ = v___y_2408_;
goto _start;
}
}
v___jp_2301_:
{
lean_object* v___x_2302_; lean_object* v___x_2303_; 
v___x_2302_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2302_, 0, v_binders_2291_);
lean_ctor_set(v___x_2302_, 1, v_toOmit_2292_);
v___x_2303_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2303_, 0, v___x_2302_);
return v___x_2303_;
}
v___jp_2304_:
{
lean_object* v_fileName_2314_; lean_object* v_fileMap_2315_; lean_object* v_currRecDepth_2316_; lean_object* v_ref_2317_; lean_object* v_currNamespace_2318_; lean_object* v_openDecls_2319_; lean_object* v_initHeartbeats_2320_; lean_object* v_maxHeartbeats_2321_; lean_object* v_quotContext_2322_; lean_object* v_currMacroScope_2323_; lean_object* v_cancelTk_x3f_2324_; uint8_t v_suppressElabErrors_2325_; lean_object* v_inheritedTraceOptions_2326_; lean_object* v___x_2327_; lean_object* v___x_2328_; lean_object* v___x_2329_; lean_object* v___x_2330_; 
v_fileName_2314_ = lean_ctor_get(v___y_2312_, 0);
v_fileMap_2315_ = lean_ctor_get(v___y_2312_, 1);
v_currRecDepth_2316_ = lean_ctor_get(v___y_2312_, 3);
v_ref_2317_ = lean_ctor_get(v___y_2312_, 5);
v_currNamespace_2318_ = lean_ctor_get(v___y_2312_, 6);
v_openDecls_2319_ = lean_ctor_get(v___y_2312_, 7);
v_initHeartbeats_2320_ = lean_ctor_get(v___y_2312_, 8);
v_maxHeartbeats_2321_ = lean_ctor_get(v___y_2312_, 9);
v_quotContext_2322_ = lean_ctor_get(v___y_2312_, 10);
v_currMacroScope_2323_ = lean_ctor_get(v___y_2312_, 11);
v_cancelTk_x3f_2324_ = lean_ctor_get(v___y_2312_, 12);
v_suppressElabErrors_2325_ = lean_ctor_get_uint8(v___y_2312_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_2326_ = lean_ctor_get(v___y_2312_, 13);
v___x_2327_ = l_Lean_maxRecDepth;
v___x_2328_ = lp_mathlib_Lean_Option_get___at___00Mathlib_Command_Variable_completeBinders_x27_spec__7(v___y_2308_, v___x_2327_);
lean_inc_ref(v_inheritedTraceOptions_2326_);
lean_inc(v_cancelTk_x3f_2324_);
lean_inc(v_currMacroScope_2323_);
lean_inc(v_quotContext_2322_);
lean_inc(v_maxHeartbeats_2321_);
lean_inc(v_initHeartbeats_2320_);
lean_inc(v_openDecls_2319_);
lean_inc(v_currNamespace_2318_);
lean_inc(v_ref_2317_);
lean_inc(v_currRecDepth_2316_);
lean_inc_ref(v_fileMap_2315_);
lean_inc_ref(v_fileName_2314_);
v___x_2329_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_2329_, 0, v_fileName_2314_);
lean_ctor_set(v___x_2329_, 1, v_fileMap_2315_);
lean_ctor_set(v___x_2329_, 2, v___y_2308_);
lean_ctor_set(v___x_2329_, 3, v_currRecDepth_2316_);
lean_ctor_set(v___x_2329_, 4, v___x_2328_);
lean_ctor_set(v___x_2329_, 5, v_ref_2317_);
lean_ctor_set(v___x_2329_, 6, v_currNamespace_2318_);
lean_ctor_set(v___x_2329_, 7, v_openDecls_2319_);
lean_ctor_set(v___x_2329_, 8, v_initHeartbeats_2320_);
lean_ctor_set(v___x_2329_, 9, v_maxHeartbeats_2321_);
lean_ctor_set(v___x_2329_, 10, v_quotContext_2322_);
lean_ctor_set(v___x_2329_, 11, v_currMacroScope_2323_);
lean_ctor_set(v___x_2329_, 12, v_cancelTk_x3f_2324_);
lean_ctor_set(v___x_2329_, 13, v_inheritedTraceOptions_2326_);
lean_ctor_set_uint8(v___x_2329_, sizeof(void*)*14, v___y_2311_);
lean_ctor_set_uint8(v___x_2329_, sizeof(void*)*14 + 1, v_suppressElabErrors_2325_);
v___x_2330_ = l_Lean_Elab_Term_withAutoBoundImplicit___redArg(v___y_2310_, v___y_2307_, v___y_2306_, v___y_2305_, v___y_2309_, v___x_2329_, v___y_2313_);
lean_dec_ref_known(v___x_2329_, 14);
return v___x_2330_;
}
v___jp_2331_:
{
if (v___y_2341_ == 0)
{
lean_object* v___x_2342_; lean_object* v_env_2343_; lean_object* v_nextMacroScope_2344_; lean_object* v_ngen_2345_; lean_object* v_auxDeclNGen_2346_; lean_object* v_traceState_2347_; lean_object* v_messages_2348_; lean_object* v_infoState_2349_; lean_object* v_snapshotTasks_2350_; lean_object* v___x_2352_; uint8_t v_isShared_2353_; uint8_t v_isSharedCheck_2360_; 
v___x_2342_ = lean_st_ref_take(v___y_2338_);
v_env_2343_ = lean_ctor_get(v___x_2342_, 0);
v_nextMacroScope_2344_ = lean_ctor_get(v___x_2342_, 1);
v_ngen_2345_ = lean_ctor_get(v___x_2342_, 2);
v_auxDeclNGen_2346_ = lean_ctor_get(v___x_2342_, 3);
v_traceState_2347_ = lean_ctor_get(v___x_2342_, 4);
v_messages_2348_ = lean_ctor_get(v___x_2342_, 6);
v_infoState_2349_ = lean_ctor_get(v___x_2342_, 7);
v_snapshotTasks_2350_ = lean_ctor_get(v___x_2342_, 8);
v_isSharedCheck_2360_ = !lean_is_exclusive(v___x_2342_);
if (v_isSharedCheck_2360_ == 0)
{
lean_object* v_unused_2361_; 
v_unused_2361_ = lean_ctor_get(v___x_2342_, 5);
lean_dec(v_unused_2361_);
v___x_2352_ = v___x_2342_;
v_isShared_2353_ = v_isSharedCheck_2360_;
goto v_resetjp_2351_;
}
else
{
lean_inc(v_snapshotTasks_2350_);
lean_inc(v_infoState_2349_);
lean_inc(v_messages_2348_);
lean_inc(v_traceState_2347_);
lean_inc(v_auxDeclNGen_2346_);
lean_inc(v_ngen_2345_);
lean_inc(v_nextMacroScope_2344_);
lean_inc(v_env_2343_);
lean_dec(v___x_2342_);
v___x_2352_ = lean_box(0);
v_isShared_2353_ = v_isSharedCheck_2360_;
goto v_resetjp_2351_;
}
v_resetjp_2351_:
{
lean_object* v___x_2354_; lean_object* v___x_2355_; lean_object* v___x_2357_; 
v___x_2354_ = l_Lean_Kernel_enableDiag(v_env_2343_, v___y_2340_);
v___x_2355_ = lean_obj_once(&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__2, &lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__2_once, _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__2);
if (v_isShared_2353_ == 0)
{
lean_ctor_set(v___x_2352_, 5, v___x_2355_);
lean_ctor_set(v___x_2352_, 0, v___x_2354_);
v___x_2357_ = v___x_2352_;
goto v_reusejp_2356_;
}
else
{
lean_object* v_reuseFailAlloc_2359_; 
v_reuseFailAlloc_2359_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2359_, 0, v___x_2354_);
lean_ctor_set(v_reuseFailAlloc_2359_, 1, v_nextMacroScope_2344_);
lean_ctor_set(v_reuseFailAlloc_2359_, 2, v_ngen_2345_);
lean_ctor_set(v_reuseFailAlloc_2359_, 3, v_auxDeclNGen_2346_);
lean_ctor_set(v_reuseFailAlloc_2359_, 4, v_traceState_2347_);
lean_ctor_set(v_reuseFailAlloc_2359_, 5, v___x_2355_);
lean_ctor_set(v_reuseFailAlloc_2359_, 6, v_messages_2348_);
lean_ctor_set(v_reuseFailAlloc_2359_, 7, v_infoState_2349_);
lean_ctor_set(v_reuseFailAlloc_2359_, 8, v_snapshotTasks_2350_);
v___x_2357_ = v_reuseFailAlloc_2359_;
goto v_reusejp_2356_;
}
v_reusejp_2356_:
{
lean_object* v___x_2358_; 
v___x_2358_ = lean_st_ref_set(v___y_2338_, v___x_2357_);
v___y_2305_ = v___y_2333_;
v___y_2306_ = v___y_2332_;
v___y_2307_ = v___y_2335_;
v___y_2308_ = v___y_2334_;
v___y_2309_ = v___y_2337_;
v___y_2310_ = v___y_2339_;
v___y_2311_ = v___y_2340_;
v___y_2312_ = v___y_2336_;
v___y_2313_ = v___y_2338_;
goto v___jp_2304_;
}
}
}
else
{
v___y_2305_ = v___y_2333_;
v___y_2306_ = v___y_2332_;
v___y_2307_ = v___y_2335_;
v___y_2308_ = v___y_2334_;
v___y_2309_ = v___y_2337_;
v___y_2310_ = v___y_2339_;
v___y_2311_ = v___y_2340_;
v___y_2312_ = v___y_2336_;
v___y_2313_ = v___y_2338_;
goto v___jp_2304_;
}
}
v___jp_2363_:
{
uint8_t v___x_2364_; 
v___x_2364_ = lean_nat_dec_eq(v_gas_2289_, v___x_2362_);
lean_dec(v_gas_2289_);
if (v___x_2364_ == 0)
{
lean_dec(v_i_2293_);
lean_dec(v_maxSteps_2288_);
goto v___jp_2301_;
}
else
{
lean_object* v___x_2365_; uint8_t v___x_2366_; 
v___x_2365_ = lean_array_get_size(v_binders_2291_);
v___x_2366_ = lean_nat_dec_lt(v_i_2293_, v___x_2365_);
if (v___x_2366_ == 0)
{
lean_dec(v_i_2293_);
lean_dec(v_maxSteps_2288_);
goto v___jp_2301_;
}
else
{
lean_object* v_ref_2367_; lean_object* v_binders_x27_2368_; uint8_t v___x_2369_; lean_object* v___x_2370_; lean_object* v___x_2371_; lean_object* v___x_2372_; lean_object* v___x_2373_; lean_object* v___x_2374_; lean_object* v___x_2375_; lean_object* v___x_2376_; lean_object* v___x_2377_; lean_object* v___x_2378_; lean_object* v___x_2379_; lean_object* v___x_2380_; lean_object* v___x_2381_; lean_object* v___x_2382_; lean_object* v___x_2383_; lean_object* v___x_2384_; lean_object* v___x_2385_; lean_object* v___x_2386_; lean_object* v___x_2387_; lean_object* v___x_2388_; lean_object* v___x_2389_; lean_object* v___x_2390_; 
v_ref_2367_ = lean_ctor_get(v_a_2298_, 5);
lean_inc(v_i_2293_);
v_binders_x27_2368_ = l_Array_extract___redArg(v_binders_2291_, v___x_2362_, v_i_2293_);
v___x_2369_ = 0;
v___x_2370_ = l_Lean_SourceInfo_fromRef(v_ref_2367_, v___x_2369_);
v___x_2371_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__3));
v___x_2372_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__4));
lean_inc_n(v___x_2370_, 2);
v___x_2373_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_2373_, 0, v___x_2370_);
lean_ctor_set(v___x_2373_, 1, v___x_2371_);
v___x_2374_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__2));
v___x_2375_ = lean_obj_once(&lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__3, &lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__3_once, _init_lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__3);
v___x_2376_ = l_Array_append___redArg(v___x_2375_, v_binders_x27_2368_);
lean_dec_ref(v_binders_x27_2368_);
v___x_2377_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2377_, 0, v___x_2370_);
lean_ctor_set(v___x_2377_, 1, v___x_2374_);
lean_ctor_set(v___x_2377_, 2, v___x_2376_);
v___x_2378_ = l_Lean_Syntax_node2(v___x_2370_, v___x_2372_, v___x_2373_, v___x_2377_);
v___x_2379_ = lean_array_fget_borrowed(v_binders_2291_, v_i_2293_);
lean_dec(v_i_2293_);
v___x_2380_ = lean_obj_once(&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__6, &lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__6_once, _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__6);
v___x_2381_ = l_Nat_reprFast(v_maxSteps_2288_);
v___x_2382_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2382_, 0, v___x_2381_);
v___x_2383_ = l_Lean_MessageData_ofFormat(v___x_2382_);
v___x_2384_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2384_, 0, v___x_2380_);
lean_ctor_set(v___x_2384_, 1, v___x_2383_);
v___x_2385_ = lean_obj_once(&lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__8, &lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__8_once, _init_lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___closed__8);
v___x_2386_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2386_, 0, v___x_2384_);
lean_ctor_set(v___x_2386_, 1, v___x_2385_);
v___x_2387_ = l_Lean_MessageData_ofSyntax(v___x_2378_);
v___x_2388_ = l_Lean_indentD(v___x_2387_);
v___x_2389_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2389_, 0, v___x_2386_);
lean_ctor_set(v___x_2389_, 1, v___x_2388_);
v___x_2390_ = lp_mathlib_Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0(v___x_2379_, v___x_2389_, v_a_2294_, v_a_2295_, v_a_2296_, v_a_2297_, v_a_2298_, v_a_2299_);
if (lean_obj_tag(v___x_2390_) == 0)
{
lean_dec_ref_known(v___x_2390_, 1);
goto v___jp_2301_;
}
else
{
lean_object* v_a_2391_; lean_object* v___x_2393_; uint8_t v_isShared_2394_; uint8_t v_isSharedCheck_2398_; 
lean_dec_ref(v_toOmit_2292_);
lean_dec_ref(v_binders_2291_);
v_a_2391_ = lean_ctor_get(v___x_2390_, 0);
v_isSharedCheck_2398_ = !lean_is_exclusive(v___x_2390_);
if (v_isSharedCheck_2398_ == 0)
{
v___x_2393_ = v___x_2390_;
v_isShared_2394_ = v_isSharedCheck_2398_;
goto v_resetjp_2392_;
}
else
{
lean_inc(v_a_2391_);
lean_dec(v___x_2390_);
v___x_2393_ = lean_box(0);
v_isShared_2394_ = v_isSharedCheck_2398_;
goto v_resetjp_2392_;
}
v_resetjp_2392_:
{
lean_object* v___x_2396_; 
if (v_isShared_2394_ == 0)
{
v___x_2396_ = v___x_2393_;
goto v_reusejp_2395_;
}
else
{
lean_object* v_reuseFailAlloc_2397_; 
v_reuseFailAlloc_2397_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2397_, 0, v_a_2391_);
v___x_2396_ = v_reuseFailAlloc_2397_;
goto v_reusejp_2395_;
}
v_reusejp_2395_:
{
return v___x_2396_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___lam__3(lean_object* v_i_2598_, lean_object* v_maxSteps_2599_, lean_object* v_gas_2600_, uint8_t v_checkRedundant_2601_, uint8_t v___x_2602_, lean_object* v_toOmit_2603_, uint8_t v___x_2604_, lean_object* v_binders_2605_, lean_object* v_binder_2606_, lean_object* v___x_2607_, lean_object* v___f_2608_, lean_object* v___y_2609_, lean_object* v___y_2610_, lean_object* v___y_2611_, lean_object* v___y_2612_, lean_object* v___y_2613_, lean_object* v___y_2614_){
_start:
{
lean_object* v_fst_2617_; lean_object* v_snd_2618_; lean_object* v___y_2623_; 
if (v___x_2602_ == 0)
{
lean_object* v___x_2627_; lean_object* v___x_2628_; 
lean_dec_ref(v___f_2608_);
v___x_2627_ = lean_box(v___x_2604_);
v___x_2628_ = lean_array_push(v_toOmit_2603_, v___x_2627_);
v_fst_2617_ = v_binders_2605_;
v_snd_2618_ = v___x_2628_;
goto v___jp_2616_;
}
else
{
lean_object* v___x_2629_; lean_object* v___x_2630_; uint8_t v___x_2631_; 
v___x_2629_ = lean_unsigned_to_nat(1u);
v___x_2630_ = l_Lean_Syntax_getArg(v_binder_2606_, v___x_2629_);
v___x_2631_ = l_Lean_Syntax_isNone(v___x_2630_);
if (v___x_2631_ == 0)
{
lean_object* v___x_2632_; uint8_t v___x_2633_; 
v___x_2632_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_2630_);
v___x_2633_ = l_Lean_Syntax_matchesNull(v___x_2630_, v___x_2632_);
if (v___x_2633_ == 0)
{
lean_object* v___x_2634_; lean_object* v___x_2635_; 
lean_dec(v___x_2630_);
lean_dec_ref(v___f_2608_);
v___x_2634_ = lean_box(v___x_2604_);
v___x_2635_ = lean_array_push(v_toOmit_2603_, v___x_2634_);
v_fst_2617_ = v_binders_2605_;
v_snd_2618_ = v___x_2635_;
goto v___jp_2616_;
}
else
{
lean_object* v___x_2636_; lean_object* v___x_2637_; lean_object* v___x_2638_; lean_object* v___x_2639_; 
lean_dec_ref(v_binders_2605_);
lean_dec_ref(v_toOmit_2603_);
v___x_2636_ = l_Lean_Syntax_getArg(v___x_2630_, v___x_2607_);
lean_dec(v___x_2630_);
v___x_2637_ = lean_box(0);
v___x_2638_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2638_, 0, v___x_2636_);
lean_inc(v___y_2614_);
lean_inc_ref(v___y_2613_);
lean_inc(v___y_2612_);
lean_inc_ref(v___y_2611_);
lean_inc(v___y_2610_);
lean_inc_ref(v___y_2609_);
v___x_2639_ = lean_apply_9(v___f_2608_, v___x_2637_, v___x_2638_, v___y_2609_, v___y_2610_, v___y_2611_, v___y_2612_, v___y_2613_, v___y_2614_, lean_box(0));
v___y_2623_ = v___x_2639_;
goto v___jp_2622_;
}
}
else
{
lean_object* v___x_2640_; lean_object* v___x_2641_; lean_object* v___x_2642_; 
lean_dec(v___x_2630_);
lean_dec_ref(v_binders_2605_);
lean_dec_ref(v_toOmit_2603_);
v___x_2640_ = lean_box(0);
v___x_2641_ = lean_box(0);
lean_inc(v___y_2614_);
lean_inc_ref(v___y_2613_);
lean_inc(v___y_2612_);
lean_inc_ref(v___y_2611_);
lean_inc(v___y_2610_);
lean_inc_ref(v___y_2609_);
v___x_2642_ = lean_apply_9(v___f_2608_, v___x_2640_, v___x_2641_, v___y_2609_, v___y_2610_, v___y_2611_, v___y_2612_, v___y_2613_, v___y_2614_, lean_box(0));
v___y_2623_ = v___x_2642_;
goto v___jp_2622_;
}
}
v___jp_2616_:
{
lean_object* v___x_2619_; lean_object* v___x_2620_; lean_object* v___x_2621_; 
v___x_2619_ = lean_unsigned_to_nat(1u);
v___x_2620_ = lean_nat_add(v_i_2598_, v___x_2619_);
v___x_2621_ = lp_mathlib_Mathlib_Command_Variable_completeBinders_x27(v_maxSteps_2599_, v_gas_2600_, v_checkRedundant_2601_, v_fst_2617_, v_snd_2618_, v___x_2620_, v___y_2609_, v___y_2610_, v___y_2611_, v___y_2612_, v___y_2613_, v___y_2614_);
lean_dec(v___y_2614_);
lean_dec_ref(v___y_2613_);
lean_dec(v___y_2612_);
lean_dec_ref(v___y_2611_);
lean_dec(v___y_2610_);
lean_dec_ref(v___y_2609_);
return v___x_2621_;
}
v___jp_2622_:
{
if (lean_obj_tag(v___y_2623_) == 0)
{
lean_object* v_a_2624_; lean_object* v_fst_2625_; lean_object* v_snd_2626_; 
v_a_2624_ = lean_ctor_get(v___y_2623_, 0);
lean_inc(v_a_2624_);
lean_dec_ref_known(v___y_2623_, 1);
v_fst_2625_ = lean_ctor_get(v_a_2624_, 0);
lean_inc(v_fst_2625_);
v_snd_2626_ = lean_ctor_get(v_a_2624_, 1);
lean_inc(v_snd_2626_);
lean_dec(v_a_2624_);
v_fst_2617_ = v_fst_2625_;
v_snd_2618_ = v_snd_2626_;
goto v___jp_2616_;
}
else
{
lean_dec(v___y_2614_);
lean_dec_ref(v___y_2613_);
lean_dec(v___y_2612_);
lean_dec_ref(v___y_2611_);
lean_dec(v___y_2610_);
lean_dec_ref(v___y_2609_);
lean_dec(v_gas_2600_);
lean_dec(v_maxSteps_2599_);
return v___y_2623_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders_x27___boxed(lean_object* v_maxSteps_2643_, lean_object* v_gas_2644_, lean_object* v_checkRedundant_2645_, lean_object* v_binders_2646_, lean_object* v_toOmit_2647_, lean_object* v_i_2648_, lean_object* v_a_2649_, lean_object* v_a_2650_, lean_object* v_a_2651_, lean_object* v_a_2652_, lean_object* v_a_2653_, lean_object* v_a_2654_, lean_object* v_a_2655_){
_start:
{
uint8_t v_checkRedundant_boxed_2656_; lean_object* v_res_2657_; 
v_checkRedundant_boxed_2656_ = lean_unbox(v_checkRedundant_2645_);
v_res_2657_ = lp_mathlib_Mathlib_Command_Variable_completeBinders_x27(v_maxSteps_2643_, v_gas_2644_, v_checkRedundant_boxed_2656_, v_binders_2646_, v_toOmit_2647_, v_i_2648_, v_a_2649_, v_a_2650_, v_a_2651_, v_a_2652_, v_a_2653_, v_a_2654_);
lean_dec(v_a_2654_);
lean_dec_ref(v_a_2653_);
lean_dec(v_a_2652_);
lean_dec_ref(v_a_2651_);
lean_dec(v_a_2650_);
lean_dec_ref(v_a_2649_);
return v_res_2657_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_Variable_completeBinders_x27_spec__4(size_t v_sz_2658_, size_t v_i_2659_, lean_object* v_bs_2660_, lean_object* v___y_2661_, lean_object* v___y_2662_, lean_object* v___y_2663_, lean_object* v___y_2664_, lean_object* v___y_2665_, lean_object* v___y_2666_){
_start:
{
lean_object* v___x_2668_; 
v___x_2668_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_Variable_completeBinders_x27_spec__4___redArg(v_sz_2658_, v_i_2659_, v_bs_2660_, v___y_2663_, v___y_2664_, v___y_2665_, v___y_2666_);
return v___x_2668_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_Variable_completeBinders_x27_spec__4___boxed(lean_object* v_sz_2669_, lean_object* v_i_2670_, lean_object* v_bs_2671_, lean_object* v___y_2672_, lean_object* v___y_2673_, lean_object* v___y_2674_, lean_object* v___y_2675_, lean_object* v___y_2676_, lean_object* v___y_2677_, lean_object* v___y_2678_){
_start:
{
size_t v_sz_boxed_2679_; size_t v_i_boxed_2680_; lean_object* v_res_2681_; 
v_sz_boxed_2679_ = lean_unbox_usize(v_sz_2669_);
lean_dec(v_sz_2669_);
v_i_boxed_2680_ = lean_unbox_usize(v_i_2670_);
lean_dec(v_i_2670_);
v_res_2681_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Command_Variable_completeBinders_x27_spec__4(v_sz_boxed_2679_, v_i_boxed_2680_, v_bs_2671_, v___y_2672_, v___y_2673_, v___y_2674_, v___y_2675_, v___y_2676_, v___y_2677_);
lean_dec(v___y_2677_);
lean_dec_ref(v___y_2676_);
lean_dec(v___y_2675_);
lean_dec_ref(v___y_2674_);
lean_dec(v___y_2673_);
lean_dec_ref(v___y_2672_);
return v_res_2681_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0(lean_object* v_ref_2682_, lean_object* v_msgData_2683_, uint8_t v_severity_2684_, uint8_t v_isSilent_2685_, lean_object* v___y_2686_, lean_object* v___y_2687_, lean_object* v___y_2688_, lean_object* v___y_2689_, lean_object* v___y_2690_, lean_object* v___y_2691_){
_start:
{
lean_object* v___x_2693_; 
v___x_2693_ = lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg(v_ref_2682_, v_msgData_2683_, v_severity_2684_, v_isSilent_2685_, v___y_2688_, v___y_2689_, v___y_2690_, v___y_2691_);
return v___x_2693_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___boxed(lean_object* v_ref_2694_, lean_object* v_msgData_2695_, lean_object* v_severity_2696_, lean_object* v_isSilent_2697_, lean_object* v___y_2698_, lean_object* v___y_2699_, lean_object* v___y_2700_, lean_object* v___y_2701_, lean_object* v___y_2702_, lean_object* v___y_2703_, lean_object* v___y_2704_){
_start:
{
uint8_t v_severity_boxed_2705_; uint8_t v_isSilent_boxed_2706_; lean_object* v_res_2707_; 
v_severity_boxed_2705_ = lean_unbox(v_severity_2696_);
v_isSilent_boxed_2706_ = lean_unbox(v_isSilent_2697_);
v_res_2707_ = lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0(v_ref_2694_, v_msgData_2695_, v_severity_boxed_2705_, v_isSilent_boxed_2706_, v___y_2698_, v___y_2699_, v___y_2700_, v___y_2701_, v___y_2702_, v___y_2703_);
lean_dec(v___y_2703_);
lean_dec_ref(v___y_2702_);
lean_dec(v___y_2701_);
lean_dec_ref(v___y_2700_);
lean_dec(v___y_2699_);
lean_dec_ref(v___y_2698_);
lean_dec(v_ref_2694_);
return v_res_2707_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders(lean_object* v_maxSteps_2710_, uint8_t v_checkRedundant_2711_, lean_object* v_binders_2712_, lean_object* v_a_2713_, lean_object* v_a_2714_, lean_object* v_a_2715_, lean_object* v_a_2716_, lean_object* v_a_2717_, lean_object* v_a_2718_){
_start:
{
lean_object* v___x_2720_; lean_object* v___x_2721_; lean_object* v___x_2722_; 
v___x_2720_ = lean_unsigned_to_nat(0u);
v___x_2721_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_completeBinders___closed__0));
lean_inc(v_maxSteps_2710_);
v___x_2722_ = lp_mathlib_Mathlib_Command_Variable_completeBinders_x27(v_maxSteps_2710_, v_maxSteps_2710_, v_checkRedundant_2711_, v_binders_2712_, v___x_2721_, v___x_2720_, v_a_2713_, v_a_2714_, v_a_2715_, v_a_2716_, v_a_2717_, v_a_2718_);
return v___x_2722_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_completeBinders___boxed(lean_object* v_maxSteps_2723_, lean_object* v_checkRedundant_2724_, lean_object* v_binders_2725_, lean_object* v_a_2726_, lean_object* v_a_2727_, lean_object* v_a_2728_, lean_object* v_a_2729_, lean_object* v_a_2730_, lean_object* v_a_2731_, lean_object* v_a_2732_){
_start:
{
uint8_t v_checkRedundant_boxed_2733_; lean_object* v_res_2734_; 
v_checkRedundant_boxed_2733_ = lean_unbox(v_checkRedundant_2724_);
v_res_2734_ = lp_mathlib_Mathlib_Command_Variable_completeBinders(v_maxSteps_2723_, v_checkRedundant_boxed_2733_, v_binders_2725_, v_a_2726_, v_a_2727_, v_a_2728_, v_a_2729_, v_a_2730_, v_a_2731_);
lean_dec(v_a_2731_);
lean_dec_ref(v_a_2730_);
lean_dec(v_a_2729_);
lean_dec_ref(v_a_2728_);
lean_dec(v_a_2727_);
lean_dec_ref(v_a_2726_);
return v_res_2734_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Command_Variable_cleanBinders_spec__0(lean_object* v_as_2735_, size_t v_sz_2736_, size_t v_i_2737_, lean_object* v_b_2738_){
_start:
{
uint8_t v___x_2739_; 
v___x_2739_ = lean_usize_dec_lt(v_i_2737_, v_sz_2736_);
if (v___x_2739_ == 0)
{
return v_b_2738_;
}
else
{
lean_object* v_a_2740_; lean_object* v___x_2741_; lean_object* v___x_2742_; size_t v___x_2743_; size_t v___x_2744_; 
v_a_2740_ = lean_array_uget_borrowed(v_as_2735_, v_i_2737_);
lean_inc(v_a_2740_);
v___x_2741_ = l_Lean_Syntax_unsetTrailing(v_a_2740_);
v___x_2742_ = lean_array_push(v_b_2738_, v___x_2741_);
v___x_2743_ = ((size_t)1ULL);
v___x_2744_ = lean_usize_add(v_i_2737_, v___x_2743_);
v_i_2737_ = v___x_2744_;
v_b_2738_ = v___x_2742_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Command_Variable_cleanBinders_spec__0___boxed(lean_object* v_as_2746_, lean_object* v_sz_2747_, lean_object* v_i_2748_, lean_object* v_b_2749_){
_start:
{
size_t v_sz_boxed_2750_; size_t v_i_boxed_2751_; lean_object* v_res_2752_; 
v_sz_boxed_2750_ = lean_unbox_usize(v_sz_2747_);
lean_dec(v_sz_2747_);
v_i_boxed_2751_ = lean_unbox_usize(v_i_2748_);
lean_dec(v_i_2748_);
v_res_2752_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Command_Variable_cleanBinders_spec__0(v_as_2746_, v_sz_boxed_2750_, v_i_boxed_2751_, v_b_2749_);
lean_dec_ref(v_as_2746_);
return v_res_2752_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_cleanBinders(lean_object* v_binders_2755_){
_start:
{
lean_object* v_binders_x27_2756_; size_t v_sz_2757_; size_t v___x_2758_; lean_object* v___x_2759_; 
v_binders_x27_2756_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_cleanBinders___closed__0));
v_sz_2757_ = lean_array_size(v_binders_2755_);
v___x_2758_ = ((size_t)0ULL);
v___x_2759_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Command_Variable_cleanBinders_spec__0(v_binders_2755_, v_sz_2757_, v___x_2758_, v_binders_x27_2756_);
return v___x_2759_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_cleanBinders___boxed(lean_object* v_binders_2760_){
_start:
{
lean_object* v_res_2761_; 
v_res_2761_ = lp_mathlib_Mathlib_Command_Variable_cleanBinders(v_binders_2760_);
lean_dec_ref(v_binders_2760_);
return v_res_2761_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__0___redArg(lean_object* v___y_2762_){
_start:
{
lean_object* v___x_2764_; lean_object* v_env_2765_; lean_object* v___x_2766_; lean_object* v_mainModule_2767_; lean_object* v___x_2768_; 
v___x_2764_ = lean_st_ref_get(v___y_2762_);
v_env_2765_ = lean_ctor_get(v___x_2764_, 0);
lean_inc_ref(v_env_2765_);
lean_dec(v___x_2764_);
v___x_2766_ = l_Lean_Environment_header(v_env_2765_);
lean_dec_ref(v_env_2765_);
v_mainModule_2767_ = lean_ctor_get(v___x_2766_, 0);
lean_inc(v_mainModule_2767_);
lean_dec_ref(v___x_2766_);
v___x_2768_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2768_, 0, v_mainModule_2767_);
return v___x_2768_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__0___redArg___boxed(lean_object* v___y_2769_, lean_object* v___y_2770_){
_start:
{
lean_object* v_res_2771_; 
v_res_2771_ = lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__0___redArg(v___y_2769_);
lean_dec(v___y_2769_);
return v_res_2771_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__0(lean_object* v___y_2772_, lean_object* v___y_2773_){
_start:
{
lean_object* v___x_2775_; 
v___x_2775_ = lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__0___redArg(v___y_2773_);
return v___x_2775_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__0___boxed(lean_object* v___y_2776_, lean_object* v___y_2777_, lean_object* v___y_2778_){
_start:
{
lean_object* v_res_2779_; 
v_res_2779_ = lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__0(v___y_2776_, v___y_2777_);
lean_dec(v___y_2777_);
lean_dec_ref(v___y_2776_);
return v_res_2779_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__1___lam__0(lean_object* v_v_2780_, lean_object* v___y_2781_, lean_object* v___y_2782_){
_start:
{
lean_object* v_a_2785_; lean_object* v_quotContext_x3f_2804_; 
v_quotContext_x3f_2804_ = lean_ctor_get(v___y_2781_, 5);
if (lean_obj_tag(v_quotContext_x3f_2804_) == 0)
{
lean_object* v___x_2805_; 
v___x_2805_ = lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__0___redArg(v___y_2782_);
if (lean_obj_tag(v___x_2805_) == 0)
{
lean_object* v_a_2806_; 
v_a_2806_ = lean_ctor_get(v___x_2805_, 0);
lean_inc(v_a_2806_);
lean_dec_ref_known(v___x_2805_, 1);
v_a_2785_ = v_a_2806_;
goto v___jp_2784_;
}
else
{
lean_dec_ref(v___y_2781_);
lean_dec(v_v_2780_);
return v___x_2805_;
}
}
else
{
lean_object* v_val_2807_; 
v_val_2807_ = lean_ctor_get(v_quotContext_x3f_2804_, 0);
lean_inc(v_val_2807_);
v_a_2785_ = v_val_2807_;
goto v___jp_2784_;
}
v___jp_2784_:
{
lean_object* v___x_2786_; 
v___x_2786_ = l_Lean_Elab_Command_getCurrMacroScope___redArg(v___y_2781_);
lean_dec_ref(v___y_2781_);
if (lean_obj_tag(v___x_2786_) == 0)
{
lean_object* v_a_2787_; lean_object* v___x_2789_; uint8_t v_isShared_2790_; uint8_t v_isSharedCheck_2795_; 
v_a_2787_ = lean_ctor_get(v___x_2786_, 0);
v_isSharedCheck_2795_ = !lean_is_exclusive(v___x_2786_);
if (v_isSharedCheck_2795_ == 0)
{
v___x_2789_ = v___x_2786_;
v_isShared_2790_ = v_isSharedCheck_2795_;
goto v_resetjp_2788_;
}
else
{
lean_inc(v_a_2787_);
lean_dec(v___x_2786_);
v___x_2789_ = lean_box(0);
v_isShared_2790_ = v_isSharedCheck_2795_;
goto v_resetjp_2788_;
}
v_resetjp_2788_:
{
lean_object* v___x_2791_; lean_object* v___x_2793_; 
v___x_2791_ = l_Lean_addMacroScope(v_a_2785_, v_v_2780_, v_a_2787_);
if (v_isShared_2790_ == 0)
{
lean_ctor_set(v___x_2789_, 0, v___x_2791_);
v___x_2793_ = v___x_2789_;
goto v_reusejp_2792_;
}
else
{
lean_object* v_reuseFailAlloc_2794_; 
v_reuseFailAlloc_2794_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2794_, 0, v___x_2791_);
v___x_2793_ = v_reuseFailAlloc_2794_;
goto v_reusejp_2792_;
}
v_reusejp_2792_:
{
return v___x_2793_;
}
}
}
else
{
lean_object* v_a_2796_; lean_object* v___x_2798_; uint8_t v_isShared_2799_; uint8_t v_isSharedCheck_2803_; 
lean_dec(v_a_2785_);
lean_dec(v_v_2780_);
v_a_2796_ = lean_ctor_get(v___x_2786_, 0);
v_isSharedCheck_2803_ = !lean_is_exclusive(v___x_2786_);
if (v_isSharedCheck_2803_ == 0)
{
v___x_2798_ = v___x_2786_;
v_isShared_2799_ = v_isSharedCheck_2803_;
goto v_resetjp_2797_;
}
else
{
lean_inc(v_a_2796_);
lean_dec(v___x_2786_);
v___x_2798_ = lean_box(0);
v_isShared_2799_ = v_isSharedCheck_2803_;
goto v_resetjp_2797_;
}
v_resetjp_2797_:
{
lean_object* v___x_2801_; 
if (v_isShared_2799_ == 0)
{
v___x_2801_ = v___x_2798_;
goto v_reusejp_2800_;
}
else
{
lean_object* v_reuseFailAlloc_2802_; 
v_reuseFailAlloc_2802_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2802_, 0, v_a_2796_);
v___x_2801_ = v_reuseFailAlloc_2802_;
goto v_reusejp_2800_;
}
v_reusejp_2800_:
{
return v___x_2801_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__1___lam__0___boxed(lean_object* v_v_2808_, lean_object* v___y_2809_, lean_object* v___y_2810_, lean_object* v___y_2811_){
_start:
{
lean_object* v_res_2812_; 
v_res_2812_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__1___lam__0(v_v_2808_, v___y_2809_, v___y_2810_);
lean_dec(v___y_2810_);
return v_res_2812_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__1(size_t v_sz_2813_, size_t v_i_2814_, lean_object* v_bs_2815_, lean_object* v___y_2816_, lean_object* v___y_2817_){
_start:
{
uint8_t v___x_2819_; 
v___x_2819_ = lean_usize_dec_lt(v_i_2814_, v_sz_2813_);
if (v___x_2819_ == 0)
{
lean_object* v___x_2820_; 
v___x_2820_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2820_, 0, v_bs_2815_);
return v___x_2820_;
}
else
{
lean_object* v_v_2821_; lean_object* v___f_2822_; lean_object* v___x_2823_; 
v_v_2821_ = lean_array_uget_borrowed(v_bs_2815_, v_i_2814_);
lean_inc(v_v_2821_);
v___f_2822_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__1___lam__0___boxed), 4, 1);
lean_closure_set(v___f_2822_, 0, v_v_2821_);
v___x_2823_ = l_Lean_Elab_Command_withFreshMacroScope___redArg(v___f_2822_, v___y_2816_, v___y_2817_);
if (lean_obj_tag(v___x_2823_) == 0)
{
lean_object* v_a_2824_; lean_object* v___x_2825_; lean_object* v_bs_x27_2826_; size_t v___x_2827_; size_t v___x_2828_; lean_object* v___x_2829_; 
v_a_2824_ = lean_ctor_get(v___x_2823_, 0);
lean_inc(v_a_2824_);
lean_dec_ref_known(v___x_2823_, 1);
v___x_2825_ = lean_unsigned_to_nat(0u);
v_bs_x27_2826_ = lean_array_uset(v_bs_2815_, v_i_2814_, v___x_2825_);
v___x_2827_ = ((size_t)1ULL);
v___x_2828_ = lean_usize_add(v_i_2814_, v___x_2827_);
v___x_2829_ = lean_array_uset(v_bs_x27_2826_, v_i_2814_, v_a_2824_);
v_i_2814_ = v___x_2828_;
v_bs_2815_ = v___x_2829_;
goto _start;
}
else
{
lean_object* v_a_2831_; lean_object* v___x_2833_; uint8_t v_isShared_2834_; uint8_t v_isSharedCheck_2838_; 
lean_dec_ref(v_bs_2815_);
v_a_2831_ = lean_ctor_get(v___x_2823_, 0);
v_isSharedCheck_2838_ = !lean_is_exclusive(v___x_2823_);
if (v_isSharedCheck_2838_ == 0)
{
v___x_2833_ = v___x_2823_;
v_isShared_2834_ = v_isSharedCheck_2838_;
goto v_resetjp_2832_;
}
else
{
lean_inc(v_a_2831_);
lean_dec(v___x_2823_);
v___x_2833_ = lean_box(0);
v_isShared_2834_ = v_isSharedCheck_2838_;
goto v_resetjp_2832_;
}
v_resetjp_2832_:
{
lean_object* v___x_2836_; 
if (v_isShared_2834_ == 0)
{
v___x_2836_ = v___x_2833_;
goto v_reusejp_2835_;
}
else
{
lean_object* v_reuseFailAlloc_2837_; 
v_reuseFailAlloc_2837_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2837_, 0, v_a_2831_);
v___x_2836_ = v_reuseFailAlloc_2837_;
goto v_reusejp_2835_;
}
v_reusejp_2835_:
{
return v___x_2836_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__1___boxed(lean_object* v_sz_2839_, lean_object* v_i_2840_, lean_object* v_bs_2841_, lean_object* v___y_2842_, lean_object* v___y_2843_, lean_object* v___y_2844_){
_start:
{
size_t v_sz_boxed_2845_; size_t v_i_boxed_2846_; lean_object* v_res_2847_; 
v_sz_boxed_2845_ = lean_unbox_usize(v_sz_2839_);
lean_dec(v_sz_2839_);
v_i_boxed_2846_ = lean_unbox_usize(v_i_2840_);
lean_dec(v_i_2840_);
v_res_2847_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__1(v_sz_boxed_2845_, v_i_boxed_2846_, v_bs_2841_, v___y_2842_, v___y_2843_);
lean_dec(v___y_2843_);
lean_dec_ref(v___y_2842_);
return v_res_2847_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__2___lam__0(lean_object* v_a_2848_, lean_object* v_a_2849_, lean_object* v_scope_2850_){
_start:
{
lean_object* v_header_2851_; lean_object* v_opts_2852_; lean_object* v_currNamespace_2853_; lean_object* v_openDecls_2854_; lean_object* v_levelNames_2855_; lean_object* v_varDecls_2856_; lean_object* v_varUIds_2857_; lean_object* v_includedVars_2858_; lean_object* v_omittedVars_2859_; uint8_t v_isNoncomputable_2860_; uint8_t v_isPublic_2861_; uint8_t v_isMeta_2862_; lean_object* v_attrs_2863_; lean_object* v___x_2865_; uint8_t v_isShared_2866_; uint8_t v_isSharedCheck_2872_; 
v_header_2851_ = lean_ctor_get(v_scope_2850_, 0);
v_opts_2852_ = lean_ctor_get(v_scope_2850_, 1);
v_currNamespace_2853_ = lean_ctor_get(v_scope_2850_, 2);
v_openDecls_2854_ = lean_ctor_get(v_scope_2850_, 3);
v_levelNames_2855_ = lean_ctor_get(v_scope_2850_, 4);
v_varDecls_2856_ = lean_ctor_get(v_scope_2850_, 5);
v_varUIds_2857_ = lean_ctor_get(v_scope_2850_, 6);
v_includedVars_2858_ = lean_ctor_get(v_scope_2850_, 7);
v_omittedVars_2859_ = lean_ctor_get(v_scope_2850_, 8);
v_isNoncomputable_2860_ = lean_ctor_get_uint8(v_scope_2850_, sizeof(void*)*10);
v_isPublic_2861_ = lean_ctor_get_uint8(v_scope_2850_, sizeof(void*)*10 + 1);
v_isMeta_2862_ = lean_ctor_get_uint8(v_scope_2850_, sizeof(void*)*10 + 2);
v_attrs_2863_ = lean_ctor_get(v_scope_2850_, 9);
v_isSharedCheck_2872_ = !lean_is_exclusive(v_scope_2850_);
if (v_isSharedCheck_2872_ == 0)
{
v___x_2865_ = v_scope_2850_;
v_isShared_2866_ = v_isSharedCheck_2872_;
goto v_resetjp_2864_;
}
else
{
lean_inc(v_attrs_2863_);
lean_inc(v_omittedVars_2859_);
lean_inc(v_includedVars_2858_);
lean_inc(v_varUIds_2857_);
lean_inc(v_varDecls_2856_);
lean_inc(v_levelNames_2855_);
lean_inc(v_openDecls_2854_);
lean_inc(v_currNamespace_2853_);
lean_inc(v_opts_2852_);
lean_inc(v_header_2851_);
lean_dec(v_scope_2850_);
v___x_2865_ = lean_box(0);
v_isShared_2866_ = v_isSharedCheck_2872_;
goto v_resetjp_2864_;
}
v_resetjp_2864_:
{
lean_object* v___x_2867_; lean_object* v___x_2868_; lean_object* v___x_2870_; 
v___x_2867_ = lean_array_push(v_varDecls_2856_, v_a_2848_);
v___x_2868_ = l_Array_append___redArg(v_varUIds_2857_, v_a_2849_);
if (v_isShared_2866_ == 0)
{
lean_ctor_set(v___x_2865_, 6, v___x_2868_);
lean_ctor_set(v___x_2865_, 5, v___x_2867_);
v___x_2870_ = v___x_2865_;
goto v_reusejp_2869_;
}
else
{
lean_object* v_reuseFailAlloc_2871_; 
v_reuseFailAlloc_2871_ = lean_alloc_ctor(0, 10, 3);
lean_ctor_set(v_reuseFailAlloc_2871_, 0, v_header_2851_);
lean_ctor_set(v_reuseFailAlloc_2871_, 1, v_opts_2852_);
lean_ctor_set(v_reuseFailAlloc_2871_, 2, v_currNamespace_2853_);
lean_ctor_set(v_reuseFailAlloc_2871_, 3, v_openDecls_2854_);
lean_ctor_set(v_reuseFailAlloc_2871_, 4, v_levelNames_2855_);
lean_ctor_set(v_reuseFailAlloc_2871_, 5, v___x_2867_);
lean_ctor_set(v_reuseFailAlloc_2871_, 6, v___x_2868_);
lean_ctor_set(v_reuseFailAlloc_2871_, 7, v_includedVars_2858_);
lean_ctor_set(v_reuseFailAlloc_2871_, 8, v_omittedVars_2859_);
lean_ctor_set(v_reuseFailAlloc_2871_, 9, v_attrs_2863_);
lean_ctor_set_uint8(v_reuseFailAlloc_2871_, sizeof(void*)*10, v_isNoncomputable_2860_);
lean_ctor_set_uint8(v_reuseFailAlloc_2871_, sizeof(void*)*10 + 1, v_isPublic_2861_);
lean_ctor_set_uint8(v_reuseFailAlloc_2871_, sizeof(void*)*10 + 2, v_isMeta_2862_);
v___x_2870_ = v_reuseFailAlloc_2871_;
goto v_reusejp_2869_;
}
v_reusejp_2869_:
{
return v___x_2870_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__2___lam__0___boxed(lean_object* v_a_2873_, lean_object* v_a_2874_, lean_object* v_scope_2875_){
_start:
{
lean_object* v_res_2876_; 
v_res_2876_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__2___lam__0(v_a_2873_, v_a_2874_, v_scope_2875_);
lean_dec_ref(v_a_2874_);
return v_res_2876_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__2(lean_object* v_as_2877_, size_t v_sz_2878_, size_t v_i_2879_, lean_object* v_b_2880_, lean_object* v___y_2881_, lean_object* v___y_2882_){
_start:
{
uint8_t v___x_2884_; 
v___x_2884_ = lean_usize_dec_lt(v_i_2879_, v_sz_2878_);
if (v___x_2884_ == 0)
{
lean_object* v___x_2885_; 
v___x_2885_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2885_, 0, v_b_2880_);
return v___x_2885_;
}
else
{
lean_object* v_a_2886_; lean_object* v___x_2887_; 
v_a_2886_ = lean_array_uget_borrowed(v_as_2877_, v_i_2879_);
lean_inc(v_a_2886_);
v___x_2887_ = l_Lean_Elab_Command_getBracketedBinderIds___redArg(v_a_2886_);
if (lean_obj_tag(v___x_2887_) == 0)
{
lean_object* v_a_2888_; size_t v_sz_2889_; size_t v___x_2890_; lean_object* v___x_2891_; 
v_a_2888_ = lean_ctor_get(v___x_2887_, 0);
lean_inc(v_a_2888_);
lean_dec_ref_known(v___x_2887_, 1);
v_sz_2889_ = lean_array_size(v_a_2888_);
v___x_2890_ = ((size_t)0ULL);
v___x_2891_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__1(v_sz_2889_, v___x_2890_, v_a_2888_, v___y_2881_, v___y_2882_);
if (lean_obj_tag(v___x_2891_) == 0)
{
lean_object* v_a_2892_; lean_object* v___f_2893_; lean_object* v___x_2894_; 
v_a_2892_ = lean_ctor_get(v___x_2891_, 0);
lean_inc(v_a_2892_);
lean_dec_ref_known(v___x_2891_, 1);
lean_inc(v_a_2886_);
v___f_2893_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__2___lam__0___boxed), 3, 2);
lean_closure_set(v___f_2893_, 0, v_a_2886_);
lean_closure_set(v___f_2893_, 1, v_a_2892_);
v___x_2894_ = l_Lean_Elab_Command_modifyScope___redArg(v___f_2893_, v___y_2882_);
if (lean_obj_tag(v___x_2894_) == 0)
{
lean_object* v___x_2895_; size_t v___x_2896_; size_t v___x_2897_; 
lean_dec_ref_known(v___x_2894_, 1);
v___x_2895_ = lean_box(0);
v___x_2896_ = ((size_t)1ULL);
v___x_2897_ = lean_usize_add(v_i_2879_, v___x_2896_);
v_i_2879_ = v___x_2897_;
v_b_2880_ = v___x_2895_;
goto _start;
}
else
{
return v___x_2894_;
}
}
else
{
lean_object* v_a_2899_; lean_object* v___x_2901_; uint8_t v_isShared_2902_; uint8_t v_isSharedCheck_2906_; 
v_a_2899_ = lean_ctor_get(v___x_2891_, 0);
v_isSharedCheck_2906_ = !lean_is_exclusive(v___x_2891_);
if (v_isSharedCheck_2906_ == 0)
{
v___x_2901_ = v___x_2891_;
v_isShared_2902_ = v_isSharedCheck_2906_;
goto v_resetjp_2900_;
}
else
{
lean_inc(v_a_2899_);
lean_dec(v___x_2891_);
v___x_2901_ = lean_box(0);
v_isShared_2902_ = v_isSharedCheck_2906_;
goto v_resetjp_2900_;
}
v_resetjp_2900_:
{
lean_object* v___x_2904_; 
if (v_isShared_2902_ == 0)
{
v___x_2904_ = v___x_2901_;
goto v_reusejp_2903_;
}
else
{
lean_object* v_reuseFailAlloc_2905_; 
v_reuseFailAlloc_2905_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2905_, 0, v_a_2899_);
v___x_2904_ = v_reuseFailAlloc_2905_;
goto v_reusejp_2903_;
}
v_reusejp_2903_:
{
return v___x_2904_;
}
}
}
}
else
{
lean_object* v_a_2907_; lean_object* v___x_2909_; uint8_t v_isShared_2910_; uint8_t v_isSharedCheck_2914_; 
v_a_2907_ = lean_ctor_get(v___x_2887_, 0);
v_isSharedCheck_2914_ = !lean_is_exclusive(v___x_2887_);
if (v_isSharedCheck_2914_ == 0)
{
v___x_2909_ = v___x_2887_;
v_isShared_2910_ = v_isSharedCheck_2914_;
goto v_resetjp_2908_;
}
else
{
lean_inc(v_a_2907_);
lean_dec(v___x_2887_);
v___x_2909_ = lean_box(0);
v_isShared_2910_ = v_isSharedCheck_2914_;
goto v_resetjp_2908_;
}
v_resetjp_2908_:
{
lean_object* v___x_2912_; 
if (v_isShared_2910_ == 0)
{
v___x_2912_ = v___x_2909_;
goto v_reusejp_2911_;
}
else
{
lean_object* v_reuseFailAlloc_2913_; 
v_reuseFailAlloc_2913_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2913_, 0, v_a_2907_);
v___x_2912_ = v_reuseFailAlloc_2913_;
goto v_reusejp_2911_;
}
v_reusejp_2911_:
{
return v___x_2912_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__2___boxed(lean_object* v_as_2915_, lean_object* v_sz_2916_, lean_object* v_i_2917_, lean_object* v_b_2918_, lean_object* v___y_2919_, lean_object* v___y_2920_, lean_object* v___y_2921_){
_start:
{
size_t v_sz_boxed_2922_; size_t v_i_boxed_2923_; lean_object* v_res_2924_; 
v_sz_boxed_2922_ = lean_unbox_usize(v_sz_2916_);
lean_dec(v_sz_2916_);
v_i_boxed_2923_ = lean_unbox_usize(v_i_2917_);
lean_dec(v_i_2917_);
v_res_2924_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__2(v_as_2915_, v_sz_boxed_2922_, v_i_boxed_2923_, v_b_2918_, v___y_2919_, v___y_2920_);
lean_dec(v___y_2920_);
lean_dec_ref(v___y_2919_);
lean_dec_ref(v_as_2915_);
return v_res_2924_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope(lean_object* v_binders_2925_, lean_object* v_a_2926_, lean_object* v_a_2927_){
_start:
{
lean_object* v___x_2929_; size_t v_sz_2930_; size_t v___x_2931_; lean_object* v___x_2932_; 
v___x_2929_ = lean_box(0);
v_sz_2930_ = lean_array_size(v_binders_2925_);
v___x_2931_ = ((size_t)0ULL);
v___x_2932_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__2(v_binders_2925_, v_sz_2930_, v___x_2931_, v___x_2929_, v_a_2926_, v_a_2927_);
if (lean_obj_tag(v___x_2932_) == 0)
{
lean_object* v___x_2934_; uint8_t v_isShared_2935_; uint8_t v_isSharedCheck_2939_; 
v_isSharedCheck_2939_ = !lean_is_exclusive(v___x_2932_);
if (v_isSharedCheck_2939_ == 0)
{
lean_object* v_unused_2940_; 
v_unused_2940_ = lean_ctor_get(v___x_2932_, 0);
lean_dec(v_unused_2940_);
v___x_2934_ = v___x_2932_;
v_isShared_2935_ = v_isSharedCheck_2939_;
goto v_resetjp_2933_;
}
else
{
lean_dec(v___x_2932_);
v___x_2934_ = lean_box(0);
v_isShared_2935_ = v_isSharedCheck_2939_;
goto v_resetjp_2933_;
}
v_resetjp_2933_:
{
lean_object* v___x_2937_; 
if (v_isShared_2935_ == 0)
{
lean_ctor_set(v___x_2934_, 0, v___x_2929_);
v___x_2937_ = v___x_2934_;
goto v_reusejp_2936_;
}
else
{
lean_object* v_reuseFailAlloc_2938_; 
v_reuseFailAlloc_2938_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2938_, 0, v___x_2929_);
v___x_2937_ = v_reuseFailAlloc_2938_;
goto v_reusejp_2936_;
}
v_reusejp_2936_:
{
return v___x_2937_;
}
}
}
else
{
return v___x_2932_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope___boxed(lean_object* v_binders_2941_, lean_object* v_a_2942_, lean_object* v_a_2943_, lean_object* v_a_2944_){
_start:
{
lean_object* v_res_2945_; 
v_res_2945_ = lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope(v_binders_2941_, v_a_2942_, v_a_2943_);
lean_dec(v_a_2943_);
lean_dec_ref(v_a_2942_);
lean_dec_ref(v_binders_2941_);
return v_res_2945_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__0___redArg(lean_object* v_x_2946_, lean_object* v___y_2947_, lean_object* v___y_2948_, lean_object* v___y_2949_, lean_object* v___y_2950_, lean_object* v___y_2951_, lean_object* v___y_2952_){
_start:
{
lean_object* v___x_2954_; 
v___x_2954_ = l___private_Lean_Elab_Term_TermElabM_0__Lean_Elab_Term_withoutModifyingStateWithInfoAndMessagesImpl(lean_box(0), v_x_2946_, v___y_2947_, v___y_2948_, v___y_2949_, v___y_2950_, v___y_2951_, v___y_2952_);
if (lean_obj_tag(v___x_2954_) == 0)
{
lean_object* v_a_2955_; lean_object* v___x_2957_; uint8_t v_isShared_2958_; uint8_t v_isSharedCheck_2962_; 
v_a_2955_ = lean_ctor_get(v___x_2954_, 0);
v_isSharedCheck_2962_ = !lean_is_exclusive(v___x_2954_);
if (v_isSharedCheck_2962_ == 0)
{
v___x_2957_ = v___x_2954_;
v_isShared_2958_ = v_isSharedCheck_2962_;
goto v_resetjp_2956_;
}
else
{
lean_inc(v_a_2955_);
lean_dec(v___x_2954_);
v___x_2957_ = lean_box(0);
v_isShared_2958_ = v_isSharedCheck_2962_;
goto v_resetjp_2956_;
}
v_resetjp_2956_:
{
lean_object* v___x_2960_; 
if (v_isShared_2958_ == 0)
{
v___x_2960_ = v___x_2957_;
goto v_reusejp_2959_;
}
else
{
lean_object* v_reuseFailAlloc_2961_; 
v_reuseFailAlloc_2961_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2961_, 0, v_a_2955_);
v___x_2960_ = v_reuseFailAlloc_2961_;
goto v_reusejp_2959_;
}
v_reusejp_2959_:
{
return v___x_2960_;
}
}
}
else
{
lean_object* v_a_2963_; lean_object* v___x_2965_; uint8_t v_isShared_2966_; uint8_t v_isSharedCheck_2970_; 
v_a_2963_ = lean_ctor_get(v___x_2954_, 0);
v_isSharedCheck_2970_ = !lean_is_exclusive(v___x_2954_);
if (v_isSharedCheck_2970_ == 0)
{
v___x_2965_ = v___x_2954_;
v_isShared_2966_ = v_isSharedCheck_2970_;
goto v_resetjp_2964_;
}
else
{
lean_inc(v_a_2963_);
lean_dec(v___x_2954_);
v___x_2965_ = lean_box(0);
v_isShared_2966_ = v_isSharedCheck_2970_;
goto v_resetjp_2964_;
}
v_resetjp_2964_:
{
lean_object* v___x_2968_; 
if (v_isShared_2966_ == 0)
{
v___x_2968_ = v___x_2965_;
goto v_reusejp_2967_;
}
else
{
lean_object* v_reuseFailAlloc_2969_; 
v_reuseFailAlloc_2969_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2969_, 0, v_a_2963_);
v___x_2968_ = v_reuseFailAlloc_2969_;
goto v_reusejp_2967_;
}
v_reusejp_2967_:
{
return v___x_2968_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__0___redArg___boxed(lean_object* v_x_2971_, lean_object* v___y_2972_, lean_object* v___y_2973_, lean_object* v___y_2974_, lean_object* v___y_2975_, lean_object* v___y_2976_, lean_object* v___y_2977_, lean_object* v___y_2978_){
_start:
{
lean_object* v_res_2979_; 
v_res_2979_ = lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__0___redArg(v_x_2971_, v___y_2972_, v___y_2973_, v___y_2974_, v___y_2975_, v___y_2976_, v___y_2977_);
lean_dec(v___y_2977_);
lean_dec_ref(v___y_2976_);
lean_dec(v___y_2975_);
lean_dec_ref(v___y_2974_);
lean_dec(v___y_2973_);
lean_dec_ref(v___y_2972_);
return v_res_2979_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__0(lean_object* v_00_u03b1_2980_, lean_object* v_x_2981_, lean_object* v___y_2982_, lean_object* v___y_2983_, lean_object* v___y_2984_, lean_object* v___y_2985_, lean_object* v___y_2986_, lean_object* v___y_2987_){
_start:
{
lean_object* v___x_2989_; 
v___x_2989_ = lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__0___redArg(v_x_2981_, v___y_2982_, v___y_2983_, v___y_2984_, v___y_2985_, v___y_2986_, v___y_2987_);
return v___x_2989_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__0___boxed(lean_object* v_00_u03b1_2990_, lean_object* v_x_2991_, lean_object* v___y_2992_, lean_object* v___y_2993_, lean_object* v___y_2994_, lean_object* v___y_2995_, lean_object* v___y_2996_, lean_object* v___y_2997_, lean_object* v___y_2998_){
_start:
{
lean_object* v_res_2999_; 
v_res_2999_ = lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__0(v_00_u03b1_2990_, v_x_2991_, v___y_2992_, v___y_2993_, v___y_2994_, v___y_2995_, v___y_2996_, v___y_2997_);
lean_dec(v___y_2997_);
lean_dec_ref(v___y_2996_);
lean_dec(v___y_2995_);
lean_dec_ref(v___y_2994_);
lean_dec(v___y_2993_);
lean_dec_ref(v___y_2992_);
return v_res_2999_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__0(lean_object* v_x_3000_, lean_object* v___y_3001_, lean_object* v___y_3002_, lean_object* v___y_3003_, lean_object* v___y_3004_, lean_object* v___y_3005_, lean_object* v___y_3006_){
_start:
{
lean_object* v___x_3008_; lean_object* v___x_3009_; 
v___x_3008_ = lean_box(0);
v___x_3009_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3009_, 0, v___x_3008_);
return v___x_3009_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__0___boxed(lean_object* v_x_3010_, lean_object* v___y_3011_, lean_object* v___y_3012_, lean_object* v___y_3013_, lean_object* v___y_3014_, lean_object* v___y_3015_, lean_object* v___y_3016_, lean_object* v___y_3017_){
_start:
{
lean_object* v_res_3018_; 
v_res_3018_ = lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__0(v_x_3010_, v___y_3011_, v___y_3012_, v___y_3013_, v___y_3014_, v___y_3015_, v___y_3016_);
lean_dec(v___y_3016_);
lean_dec_ref(v___y_3015_);
lean_dec(v___y_3014_);
lean_dec_ref(v___y_3013_);
lean_dec(v___y_3012_);
lean_dec_ref(v___y_3011_);
lean_dec_ref(v_x_3010_);
return v_res_3018_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__1___closed__0(void){
_start:
{
lean_object* v___x_3019_; lean_object* v___x_3020_; 
v___x_3019_ = lean_box(0);
v___x_3020_ = l_Lean_Expr_sort___override(v___x_3019_);
return v___x_3020_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__1(lean_object* v_x_3021_, lean_object* v___y_3022_, lean_object* v___y_3023_, lean_object* v___y_3024_, lean_object* v___y_3025_, lean_object* v___y_3026_, lean_object* v___y_3027_){
_start:
{
lean_object* v_lctx_3029_; lean_object* v___x_3030_; lean_object* v___x_3031_; uint8_t v___x_3032_; uint8_t v___x_3033_; uint8_t v___x_3034_; lean_object* v___x_3035_; 
v_lctx_3029_ = lean_ctor_get(v___y_3024_, 2);
v___x_3030_ = l_Lean_LocalContext_getFVars(v_lctx_3029_);
v___x_3031_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__1___closed__0, &lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__1___closed__0_once, _init_lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__1___closed__0);
v___x_3032_ = 0;
v___x_3033_ = 1;
v___x_3034_ = 1;
v___x_3035_ = l_Lean_Meta_mkForallFVars(v___x_3030_, v___x_3031_, v___x_3032_, v___x_3033_, v___x_3033_, v___x_3034_, v___y_3024_, v___y_3025_, v___y_3026_, v___y_3027_);
lean_dec_ref(v___x_3030_);
if (lean_obj_tag(v___x_3035_) == 0)
{
lean_object* v_a_3036_; lean_object* v___x_3037_; 
v_a_3036_ = lean_ctor_get(v___x_3035_, 0);
lean_inc(v_a_3036_);
lean_dec_ref_known(v___x_3035_, 1);
v___x_3037_ = l_Lean_Meta_abstractMVars(v_a_3036_, v___x_3033_, v___y_3024_, v___y_3025_, v___y_3026_, v___y_3027_);
if (lean_obj_tag(v___x_3037_) == 0)
{
lean_object* v_a_3038_; lean_object* v___x_3040_; uint8_t v_isShared_3041_; uint8_t v_isSharedCheck_3059_; 
v_a_3038_ = lean_ctor_get(v___x_3037_, 0);
v_isSharedCheck_3059_ = !lean_is_exclusive(v___x_3037_);
if (v_isSharedCheck_3059_ == 0)
{
v___x_3040_ = v___x_3037_;
v_isShared_3041_ = v_isSharedCheck_3059_;
goto v_resetjp_3039_;
}
else
{
lean_inc(v_a_3038_);
lean_dec(v___x_3037_);
v___x_3040_ = lean_box(0);
v_isShared_3041_ = v_isSharedCheck_3059_;
goto v_resetjp_3039_;
}
v_resetjp_3039_:
{
lean_object* v___x_3042_; lean_object* v_levelNames_3043_; lean_object* v_paramNames_3044_; lean_object* v_mvars_3045_; lean_object* v_expr_3046_; lean_object* v___x_3048_; uint8_t v_isShared_3049_; uint8_t v_isSharedCheck_3058_; 
v___x_3042_ = lean_st_ref_get(v___y_3023_);
v_levelNames_3043_ = lean_ctor_get(v___x_3042_, 0);
lean_inc(v_levelNames_3043_);
lean_dec(v___x_3042_);
v_paramNames_3044_ = lean_ctor_get(v_a_3038_, 0);
v_mvars_3045_ = lean_ctor_get(v_a_3038_, 1);
v_expr_3046_ = lean_ctor_get(v_a_3038_, 2);
v_isSharedCheck_3058_ = !lean_is_exclusive(v_a_3038_);
if (v_isSharedCheck_3058_ == 0)
{
v___x_3048_ = v_a_3038_;
v_isShared_3049_ = v_isSharedCheck_3058_;
goto v_resetjp_3047_;
}
else
{
lean_inc(v_expr_3046_);
lean_inc(v_mvars_3045_);
lean_inc(v_paramNames_3044_);
lean_dec(v_a_3038_);
v___x_3048_ = lean_box(0);
v_isShared_3049_ = v_isSharedCheck_3058_;
goto v_resetjp_3047_;
}
v_resetjp_3047_:
{
lean_object* v___x_3050_; lean_object* v___x_3051_; lean_object* v___x_3053_; 
v___x_3050_ = lean_array_mk(v_levelNames_3043_);
v___x_3051_ = l_Array_append___redArg(v___x_3050_, v_paramNames_3044_);
lean_dec_ref(v_paramNames_3044_);
if (v_isShared_3049_ == 0)
{
lean_ctor_set(v___x_3048_, 0, v___x_3051_);
v___x_3053_ = v___x_3048_;
goto v_reusejp_3052_;
}
else
{
lean_object* v_reuseFailAlloc_3057_; 
v_reuseFailAlloc_3057_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_3057_, 0, v___x_3051_);
lean_ctor_set(v_reuseFailAlloc_3057_, 1, v_mvars_3045_);
lean_ctor_set(v_reuseFailAlloc_3057_, 2, v_expr_3046_);
v___x_3053_ = v_reuseFailAlloc_3057_;
goto v_reusejp_3052_;
}
v_reusejp_3052_:
{
lean_object* v___x_3055_; 
if (v_isShared_3041_ == 0)
{
lean_ctor_set(v___x_3040_, 0, v___x_3053_);
v___x_3055_ = v___x_3040_;
goto v_reusejp_3054_;
}
else
{
lean_object* v_reuseFailAlloc_3056_; 
v_reuseFailAlloc_3056_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3056_, 0, v___x_3053_);
v___x_3055_ = v_reuseFailAlloc_3056_;
goto v_reusejp_3054_;
}
v_reusejp_3054_:
{
return v___x_3055_;
}
}
}
}
}
else
{
return v___x_3037_;
}
}
else
{
lean_object* v_a_3060_; lean_object* v___x_3062_; uint8_t v_isShared_3063_; uint8_t v_isSharedCheck_3067_; 
v_a_3060_ = lean_ctor_get(v___x_3035_, 0);
v_isSharedCheck_3067_ = !lean_is_exclusive(v___x_3035_);
if (v_isSharedCheck_3067_ == 0)
{
v___x_3062_ = v___x_3035_;
v_isShared_3063_ = v_isSharedCheck_3067_;
goto v_resetjp_3061_;
}
else
{
lean_inc(v_a_3060_);
lean_dec(v___x_3035_);
v___x_3062_ = lean_box(0);
v_isShared_3063_ = v_isSharedCheck_3067_;
goto v_resetjp_3061_;
}
v_resetjp_3061_:
{
lean_object* v___x_3065_; 
if (v_isShared_3063_ == 0)
{
v___x_3065_ = v___x_3062_;
goto v_reusejp_3064_;
}
else
{
lean_object* v_reuseFailAlloc_3066_; 
v_reuseFailAlloc_3066_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3066_, 0, v_a_3060_);
v___x_3065_ = v_reuseFailAlloc_3066_;
goto v_reusejp_3064_;
}
v_reusejp_3064_:
{
return v___x_3065_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__1___boxed(lean_object* v_x_3068_, lean_object* v___y_3069_, lean_object* v___y_3070_, lean_object* v___y_3071_, lean_object* v___y_3072_, lean_object* v___y_3073_, lean_object* v___y_3074_, lean_object* v___y_3075_){
_start:
{
lean_object* v_res_3076_; 
v_res_3076_ = lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__1(v_x_3068_, v___y_3069_, v___y_3070_, v___y_3071_, v___y_3072_, v___y_3073_, v___y_3074_);
lean_dec(v___y_3074_);
lean_dec_ref(v___y_3073_);
lean_dec(v___y_3072_);
lean_dec_ref(v___y_3071_);
lean_dec(v___y_3070_);
lean_dec_ref(v___y_3069_);
lean_dec_ref(v_x_3068_);
return v_res_3076_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__2(lean_object* v___f_3077_, lean_object* v_binders_3078_, lean_object* v___y_3079_, lean_object* v___y_3080_, lean_object* v___y_3081_, lean_object* v___y_3082_, lean_object* v___y_3083_, lean_object* v___y_3084_){
_start:
{
lean_object* v___x_3086_; lean_object* v___x_3087_; lean_object* v___x_3088_; 
v___x_3086_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_elabBinders___boxed), 10, 3);
lean_closure_set(v___x_3086_, 0, lean_box(0));
lean_closure_set(v___x_3086_, 1, v_binders_3078_);
lean_closure_set(v___x_3086_, 2, v___f_3077_);
v___x_3087_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_withAutoBoundImplicit___boxed), 9, 2);
lean_closure_set(v___x_3087_, 0, lean_box(0));
lean_closure_set(v___x_3087_, 1, v___x_3086_);
v___x_3088_ = lp_mathlib_Lean_Elab_withoutModifyingStateWithInfoAndMessages___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__0___redArg(v___x_3087_, v___y_3079_, v___y_3080_, v___y_3081_, v___y_3082_, v___y_3083_, v___y_3084_);
return v___x_3088_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__2___boxed(lean_object* v___f_3089_, lean_object* v_binders_3090_, lean_object* v___y_3091_, lean_object* v___y_3092_, lean_object* v___y_3093_, lean_object* v___y_3094_, lean_object* v___y_3095_, lean_object* v___y_3096_, lean_object* v___y_3097_){
_start:
{
lean_object* v_res_3098_; 
v_res_3098_ = lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__2(v___f_3089_, v_binders_3090_, v___y_3091_, v___y_3092_, v___y_3093_, v___y_3094_, v___y_3095_, v___y_3096_);
lean_dec(v___y_3096_);
lean_dec_ref(v___y_3095_);
lean_dec(v___y_3094_);
lean_dec_ref(v___y_3093_);
lean_dec(v___y_3092_);
lean_dec_ref(v___y_3091_);
return v_res_3098_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__3(lean_object* v_stx_3099_, lean_object* v___x_3100_, lean_object* v___x_3101_, lean_object* v___x_3102_, lean_object* v___x_3103_, uint8_t v___x_3104_, lean_object* v___x_3105_, lean_object* v___y_3106_, lean_object* v___y_3107_, lean_object* v___y_3108_, lean_object* v___y_3109_, lean_object* v___y_3110_, lean_object* v___y_3111_){
_start:
{
lean_object* v___x_3113_; 
v___x_3113_ = l_Lean_Meta_Tactic_TryThis_addSuggestion(v_stx_3099_, v___x_3100_, v___x_3101_, v___x_3102_, v___x_3103_, v___x_3104_, v___x_3105_, v___y_3110_, v___y_3111_);
return v___x_3113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__3___boxed(lean_object* v_stx_3114_, lean_object* v___x_3115_, lean_object* v___x_3116_, lean_object* v___x_3117_, lean_object* v___x_3118_, lean_object* v___x_3119_, lean_object* v___x_3120_, lean_object* v___y_3121_, lean_object* v___y_3122_, lean_object* v___y_3123_, lean_object* v___y_3124_, lean_object* v___y_3125_, lean_object* v___y_3126_, lean_object* v___y_3127_){
_start:
{
uint8_t v___x_19723__boxed_3128_; lean_object* v_res_3129_; 
v___x_19723__boxed_3128_ = lean_unbox(v___x_3119_);
v_res_3129_ = lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__3(v_stx_3114_, v___x_3115_, v___x_3116_, v___x_3117_, v___x_3118_, v___x_19723__boxed_3128_, v___x_3120_, v___y_3121_, v___y_3122_, v___y_3123_, v___y_3124_, v___y_3125_, v___y_3126_);
lean_dec(v___y_3126_);
lean_dec_ref(v___y_3125_);
lean_dec(v___y_3124_);
lean_dec_ref(v___y_3123_);
lean_dec(v___y_3122_);
lean_dec_ref(v___y_3121_);
return v_res_3129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__4(lean_object* v___x_3130_, lean_object* v___y_3131_, lean_object* v___y_3132_, lean_object* v___y_3133_, lean_object* v___y_3134_, lean_object* v___y_3135_, lean_object* v___y_3136_){
_start:
{
lean_object* v_options_3138_; uint8_t v_hasTrace_3139_; 
v_options_3138_ = lean_ctor_get(v___y_3135_, 2);
v_hasTrace_3139_ = lean_ctor_get_uint8(v_options_3138_, sizeof(void*)*1);
if (v_hasTrace_3139_ == 0)
{
lean_object* v___x_3140_; lean_object* v___x_3141_; 
lean_dec(v___x_3130_);
v___x_3140_ = lean_box(v_hasTrace_3139_);
v___x_3141_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3141_, 0, v___x_3140_);
return v___x_3141_;
}
else
{
lean_object* v_inheritedTraceOptions_3142_; lean_object* v___x_3143_; lean_object* v___x_3144_; uint8_t v___x_3145_; lean_object* v___x_3146_; lean_object* v___x_3147_; 
v_inheritedTraceOptions_3142_ = lean_ctor_get(v___y_3135_, 13);
v___x_3143_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__7));
v___x_3144_ = l_Lean_Name_append(v___x_3143_, v___x_3130_);
v___x_3145_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3142_, v_options_3138_, v___x_3144_);
lean_dec(v___x_3144_);
v___x_3146_ = lean_box(v___x_3145_);
v___x_3147_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3147_, 0, v___x_3146_);
return v___x_3147_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__4___boxed(lean_object* v___x_3148_, lean_object* v___y_3149_, lean_object* v___y_3150_, lean_object* v___y_3151_, lean_object* v___y_3152_, lean_object* v___y_3153_, lean_object* v___y_3154_, lean_object* v___y_3155_){
_start:
{
lean_object* v_res_3156_; 
v_res_3156_ = lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__4(v___x_3148_, v___y_3149_, v___y_3150_, v___y_3151_, v___y_3152_, v___y_3153_, v___y_3154_);
lean_dec(v___y_3154_);
lean_dec_ref(v___y_3153_);
lean_dec(v___y_3152_);
lean_dec_ref(v___y_3151_);
lean_dec(v___y_3150_);
lean_dec_ref(v___y_3149_);
return v_res_3156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__2_spec__3___redArg(lean_object* v_msgData_3157_, uint8_t v_severity_3158_, uint8_t v_isSilent_3159_, lean_object* v___y_3160_, lean_object* v___y_3161_, lean_object* v___y_3162_, lean_object* v___y_3163_){
_start:
{
lean_object* v_ref_3165_; lean_object* v___x_3166_; 
v_ref_3165_ = lean_ctor_get(v___y_3162_, 5);
v___x_3166_ = lp_mathlib_Lean_logAt___at___00Lean_logErrorAt___at___00Mathlib_Command_Variable_completeBinders_x27_spec__0_spec__0___redArg(v_ref_3165_, v_msgData_3157_, v_severity_3158_, v_isSilent_3159_, v___y_3160_, v___y_3161_, v___y_3162_, v___y_3163_);
return v___x_3166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__2_spec__3___redArg___boxed(lean_object* v_msgData_3167_, lean_object* v_severity_3168_, lean_object* v_isSilent_3169_, lean_object* v___y_3170_, lean_object* v___y_3171_, lean_object* v___y_3172_, lean_object* v___y_3173_, lean_object* v___y_3174_){
_start:
{
uint8_t v_severity_boxed_3175_; uint8_t v_isSilent_boxed_3176_; lean_object* v_res_3177_; 
v_severity_boxed_3175_ = lean_unbox(v_severity_3168_);
v_isSilent_boxed_3176_ = lean_unbox(v_isSilent_3169_);
v_res_3177_ = lp_mathlib_Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__2_spec__3___redArg(v_msgData_3167_, v_severity_boxed_3175_, v_isSilent_boxed_3176_, v___y_3170_, v___y_3171_, v___y_3172_, v___y_3173_);
lean_dec(v___y_3173_);
lean_dec_ref(v___y_3172_);
lean_dec(v___y_3171_);
lean_dec_ref(v___y_3170_);
return v_res_3177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__2(lean_object* v_msgData_3178_, lean_object* v___y_3179_, lean_object* v___y_3180_, lean_object* v___y_3181_, lean_object* v___y_3182_, lean_object* v___y_3183_, lean_object* v___y_3184_){
_start:
{
uint8_t v___x_3186_; uint8_t v___x_3187_; lean_object* v___x_3188_; 
v___x_3186_ = 1;
v___x_3187_ = 0;
v___x_3188_ = lp_mathlib_Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__2_spec__3___redArg(v_msgData_3178_, v___x_3186_, v___x_3187_, v___y_3181_, v___y_3182_, v___y_3183_, v___y_3184_);
return v___x_3188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__2___boxed(lean_object* v_msgData_3189_, lean_object* v___y_3190_, lean_object* v___y_3191_, lean_object* v___y_3192_, lean_object* v___y_3193_, lean_object* v___y_3194_, lean_object* v___y_3195_, lean_object* v___y_3196_){
_start:
{
lean_object* v_res_3197_; 
v_res_3197_ = lp_mathlib_Lean_logWarning___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__2(v_msgData_3189_, v___y_3190_, v___y_3191_, v___y_3192_, v___y_3193_, v___y_3194_, v___y_3195_);
lean_dec(v___y_3195_);
lean_dec_ref(v___y_3194_);
lean_dec(v___y_3193_);
lean_dec_ref(v___y_3192_);
lean_dec(v___y_3191_);
lean_dec_ref(v___y_3190_);
return v_res_3197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__1_spec__1(lean_object* v_as_3198_, size_t v_i_3199_, size_t v_stop_3200_, lean_object* v_b_3201_){
_start:
{
lean_object* v___y_3203_; uint8_t v___x_3207_; 
v___x_3207_ = lean_usize_dec_eq(v_i_3199_, v_stop_3200_);
if (v___x_3207_ == 0)
{
lean_object* v___x_3208_; lean_object* v_snd_3209_; uint8_t v___x_3210_; 
v___x_3208_ = lean_array_uget_borrowed(v_as_3198_, v_i_3199_);
v_snd_3209_ = lean_ctor_get(v___x_3208_, 1);
v___x_3210_ = lean_unbox(v_snd_3209_);
if (v___x_3210_ == 0)
{
lean_object* v_fst_3211_; lean_object* v___x_3212_; 
v_fst_3211_ = lean_ctor_get(v___x_3208_, 0);
lean_inc(v_fst_3211_);
v___x_3212_ = lean_array_push(v_b_3201_, v_fst_3211_);
v___y_3203_ = v___x_3212_;
goto v___jp_3202_;
}
else
{
v___y_3203_ = v_b_3201_;
goto v___jp_3202_;
}
}
else
{
return v_b_3201_;
}
v___jp_3202_:
{
size_t v___x_3204_; size_t v___x_3205_; 
v___x_3204_ = ((size_t)1ULL);
v___x_3205_ = lean_usize_add(v_i_3199_, v___x_3204_);
v_i_3199_ = v___x_3205_;
v_b_3201_ = v___y_3203_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__1_spec__1___boxed(lean_object* v_as_3213_, lean_object* v_i_3214_, lean_object* v_stop_3215_, lean_object* v_b_3216_){
_start:
{
size_t v_i_boxed_3217_; size_t v_stop_boxed_3218_; lean_object* v_res_3219_; 
v_i_boxed_3217_ = lean_unbox_usize(v_i_3214_);
lean_dec(v_i_3214_);
v_stop_boxed_3218_ = lean_unbox_usize(v_stop_3215_);
lean_dec(v_stop_3215_);
v_res_3219_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__1_spec__1(v_as_3213_, v_i_boxed_3217_, v_stop_boxed_3218_, v_b_3216_);
lean_dec_ref(v_as_3213_);
return v_res_3219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__1(lean_object* v_as_3220_, lean_object* v_start_3221_, lean_object* v_stop_3222_){
_start:
{
lean_object* v___x_3223_; uint8_t v___x_3224_; 
v___x_3223_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_cleanBinders___closed__0));
v___x_3224_ = lean_nat_dec_lt(v_start_3221_, v_stop_3222_);
if (v___x_3224_ == 0)
{
return v___x_3223_;
}
else
{
lean_object* v___x_3225_; uint8_t v___x_3226_; 
v___x_3225_ = lean_array_get_size(v_as_3220_);
v___x_3226_ = lean_nat_dec_le(v_stop_3222_, v___x_3225_);
if (v___x_3226_ == 0)
{
uint8_t v___x_3227_; 
v___x_3227_ = lean_nat_dec_lt(v_start_3221_, v___x_3225_);
if (v___x_3227_ == 0)
{
return v___x_3223_;
}
else
{
size_t v___x_3228_; size_t v___x_3229_; lean_object* v___x_3230_; 
v___x_3228_ = lean_usize_of_nat(v_start_3221_);
v___x_3229_ = lean_usize_of_nat(v___x_3225_);
v___x_3230_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__1_spec__1(v_as_3220_, v___x_3228_, v___x_3229_, v___x_3223_);
return v___x_3230_;
}
}
else
{
size_t v___x_3231_; size_t v___x_3232_; lean_object* v___x_3233_; 
v___x_3231_ = lean_usize_of_nat(v_start_3221_);
v___x_3232_ = lean_usize_of_nat(v_stop_3222_);
v___x_3233_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__1_spec__1(v_as_3220_, v___x_3231_, v___x_3232_, v___x_3223_);
return v___x_3233_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__1___boxed(lean_object* v_as_3234_, lean_object* v_start_3235_, lean_object* v_stop_3236_){
_start:
{
lean_object* v_res_3237_; 
v_res_3237_ = lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__1(v_as_3234_, v_start_3235_, v_stop_3236_);
lean_dec(v_stop_3236_);
lean_dec(v_start_3235_);
lean_dec_ref(v_as_3234_);
return v_res_3237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__4(lean_object* v_a_3238_, lean_object* v_a_3239_){
_start:
{
if (lean_obj_tag(v_a_3238_) == 0)
{
lean_object* v___x_3240_; 
v___x_3240_ = l_List_reverse___redArg(v_a_3239_);
return v___x_3240_;
}
else
{
lean_object* v_head_3241_; lean_object* v_tail_3242_; lean_object* v___x_3244_; uint8_t v_isShared_3245_; uint8_t v_isSharedCheck_3251_; 
v_head_3241_ = lean_ctor_get(v_a_3238_, 0);
v_tail_3242_ = lean_ctor_get(v_a_3238_, 1);
v_isSharedCheck_3251_ = !lean_is_exclusive(v_a_3238_);
if (v_isSharedCheck_3251_ == 0)
{
v___x_3244_ = v_a_3238_;
v_isShared_3245_ = v_isSharedCheck_3251_;
goto v_resetjp_3243_;
}
else
{
lean_inc(v_tail_3242_);
lean_inc(v_head_3241_);
lean_dec(v_a_3238_);
v___x_3244_ = lean_box(0);
v_isShared_3245_ = v_isSharedCheck_3251_;
goto v_resetjp_3243_;
}
v_resetjp_3243_:
{
lean_object* v___x_3246_; lean_object* v___x_3248_; 
v___x_3246_ = l_Lean_MessageData_ofName(v_head_3241_);
if (v_isShared_3245_ == 0)
{
lean_ctor_set(v___x_3244_, 1, v_a_3239_);
lean_ctor_set(v___x_3244_, 0, v___x_3246_);
v___x_3248_ = v___x_3244_;
goto v_reusejp_3247_;
}
else
{
lean_object* v_reuseFailAlloc_3250_; 
v_reuseFailAlloc_3250_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3250_, 0, v___x_3246_);
lean_ctor_set(v_reuseFailAlloc_3250_, 1, v_a_3239_);
v___x_3248_ = v_reuseFailAlloc_3250_;
goto v_reusejp_3247_;
}
v_reusejp_3247_:
{
v_a_3238_ = v_tail_3242_;
v_a_3239_ = v___x_3248_;
goto _start;
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__3___redArg(lean_object* v_xs_3252_, lean_object* v_ys_3253_, lean_object* v_x_3254_){
_start:
{
lean_object* v_zero_3255_; uint8_t v_isZero_3256_; 
v_zero_3255_ = lean_unsigned_to_nat(0u);
v_isZero_3256_ = lean_nat_dec_eq(v_x_3254_, v_zero_3255_);
if (v_isZero_3256_ == 1)
{
lean_dec(v_x_3254_);
return v_isZero_3256_;
}
else
{
lean_object* v_one_3257_; lean_object* v_n_3258_; lean_object* v___x_3259_; lean_object* v___x_3260_; uint8_t v___x_3261_; 
v_one_3257_ = lean_unsigned_to_nat(1u);
v_n_3258_ = lean_nat_sub(v_x_3254_, v_one_3257_);
lean_dec(v_x_3254_);
v___x_3259_ = lean_array_fget_borrowed(v_xs_3252_, v_n_3258_);
v___x_3260_ = lean_array_fget_borrowed(v_ys_3253_, v_n_3258_);
v___x_3261_ = lean_name_eq(v___x_3259_, v___x_3260_);
if (v___x_3261_ == 0)
{
lean_dec(v_n_3258_);
return v___x_3261_;
}
else
{
v_x_3254_ = v_n_3258_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__3___redArg___boxed(lean_object* v_xs_3263_, lean_object* v_ys_3264_, lean_object* v_x_3265_){
_start:
{
uint8_t v_res_3266_; lean_object* v_r_3267_; 
v_res_3266_ = lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__3___redArg(v_xs_3263_, v_ys_3264_, v_x_3265_);
lean_dec_ref(v_ys_3264_);
lean_dec_ref(v_xs_3263_);
v_r_3267_ = lean_box(v_res_3266_);
return v_r_3267_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__2(void){
_start:
{
lean_object* v___x_3271_; lean_object* v___x_3272_; 
v___x_3271_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__1));
v___x_3272_ = l_Lean_MessageData_ofFormat(v___x_3271_);
return v___x_3272_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__4(void){
_start:
{
lean_object* v___x_3274_; lean_object* v___x_3275_; 
v___x_3274_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__3));
v___x_3275_ = l_Lean_stringToMessageData(v___x_3274_);
return v___x_3275_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__6(void){
_start:
{
lean_object* v___x_3277_; lean_object* v___x_3278_; 
v___x_3277_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__5));
v___x_3278_ = l_Lean_stringToMessageData(v___x_3277_);
return v___x_3278_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__7(void){
_start:
{
lean_object* v___x_3279_; lean_object* v___x_3280_; 
v___x_3279_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg___closed__1));
v___x_3280_ = l_Lean_stringToMessageData(v___x_3279_);
return v___x_3280_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__9(void){
_start:
{
lean_object* v___x_3282_; lean_object* v___x_3283_; 
v___x_3282_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__8));
v___x_3283_ = l_Lean_stringToMessageData(v___x_3282_);
return v___x_3283_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__11(void){
_start:
{
lean_object* v___x_3285_; lean_object* v___x_3286_; 
v___x_3285_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__10));
v___x_3286_ = l_Lean_stringToMessageData(v___x_3285_);
return v___x_3286_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__13(void){
_start:
{
lean_object* v___x_3288_; lean_object* v___x_3289_; 
v___x_3288_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__12));
v___x_3289_ = l_Lean_stringToMessageData(v___x_3288_);
return v___x_3289_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__15(void){
_start:
{
lean_object* v___x_3291_; lean_object* v___x_3292_; 
v___x_3291_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__14));
v___x_3292_ = l_Lean_stringToMessageData(v___x_3291_);
return v___x_3292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5(lean_object* v___x_3293_, uint8_t v_checkRedundant_3294_, lean_object* v_binders_3295_, lean_object* v___f_3296_, lean_object* v_expectedBinders_x3f_3297_, lean_object* v___f_3298_, lean_object* v___x_3299_, lean_object* v___f_3300_, lean_object* v_x_3301_, lean_object* v___y_3302_, lean_object* v___y_3303_, lean_object* v___y_3304_, lean_object* v___y_3305_, lean_object* v___y_3306_, lean_object* v___y_3307_){
_start:
{
lean_object* v___x_3309_; 
v___x_3309_ = lp_mathlib_Mathlib_Command_Variable_completeBinders(v___x_3293_, v_checkRedundant_3294_, v_binders_3295_, v___y_3302_, v___y_3303_, v___y_3304_, v___y_3305_, v___y_3306_, v___y_3307_);
if (lean_obj_tag(v___x_3309_) == 0)
{
lean_object* v_a_3310_; lean_object* v_fst_3311_; lean_object* v_snd_3312_; lean_object* v___x_3314_; uint8_t v_isShared_3315_; uint8_t v_isSharedCheck_3559_; 
v_a_3310_ = lean_ctor_get(v___x_3309_, 0);
lean_inc(v_a_3310_);
lean_dec_ref_known(v___x_3309_, 1);
v_fst_3311_ = lean_ctor_get(v_a_3310_, 0);
v_snd_3312_ = lean_ctor_get(v_a_3310_, 1);
v_isSharedCheck_3559_ = !lean_is_exclusive(v_a_3310_);
if (v_isSharedCheck_3559_ == 0)
{
v___x_3314_ = v_a_3310_;
v_isShared_3315_ = v_isSharedCheck_3559_;
goto v_resetjp_3313_;
}
else
{
lean_inc(v_snd_3312_);
lean_inc(v_fst_3311_);
lean_dec(v_a_3310_);
v___x_3314_ = lean_box(0);
v_isShared_3315_ = v_isSharedCheck_3559_;
goto v_resetjp_3313_;
}
v_resetjp_3313_:
{
lean_object* v___x_3316_; lean_object* v___x_3317_; 
lean_inc(v_fst_3311_);
v___x_3316_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_elabBinders___boxed), 10, 3);
lean_closure_set(v___x_3316_, 0, lean_box(0));
lean_closure_set(v___x_3316_, 1, v_fst_3311_);
lean_closure_set(v___x_3316_, 2, v___f_3296_);
v___x_3317_ = l_Lean_Elab_Term_withAutoBoundImplicit___redArg(v___x_3316_, v___y_3302_, v___y_3303_, v___y_3304_, v___y_3305_, v___y_3306_, v___y_3307_);
if (lean_obj_tag(v___x_3317_) == 0)
{
lean_object* v___x_3319_; uint8_t v_isShared_3320_; uint8_t v_isSharedCheck_3549_; 
v_isSharedCheck_3549_ = !lean_is_exclusive(v___x_3317_);
if (v_isSharedCheck_3549_ == 0)
{
lean_object* v_unused_3550_; 
v_unused_3550_ = lean_ctor_get(v___x_3317_, 0);
lean_dec(v_unused_3550_);
v___x_3319_ = v___x_3317_;
v_isShared_3320_ = v_isSharedCheck_3549_;
goto v_resetjp_3318_;
}
else
{
lean_dec(v___x_3317_);
v___x_3319_ = lean_box(0);
v_isShared_3320_ = v_isSharedCheck_3549_;
goto v_resetjp_3318_;
}
v_resetjp_3318_:
{
lean_object* v___x_3321_; lean_object* v___x_3322_; lean_object* v___x_3323_; lean_object* v___x_3324_; lean_object* v___y_3326_; lean_object* v___y_3327_; lean_object* v___y_3328_; lean_object* v___y_3329_; lean_object* v___y_3330_; lean_object* v___y_3331_; lean_object* v___y_3356_; lean_object* v___y_3357_; lean_object* v___y_3358_; lean_object* v___y_3359_; lean_object* v___y_3360_; lean_object* v___y_3361_; lean_object* v___y_3362_; lean_object* v___y_3363_; lean_object* v___y_3397_; lean_object* v___y_3398_; lean_object* v___y_3399_; lean_object* v___y_3400_; lean_object* v___y_3401_; lean_object* v___y_3402_; lean_object* v___y_3403_; lean_object* v___y_3404_; 
v___x_3321_ = l_Array_zip___redArg(v_fst_3311_, v_snd_3312_);
lean_dec(v_snd_3312_);
lean_dec(v_fst_3311_);
v___x_3322_ = lean_unsigned_to_nat(0u);
v___x_3323_ = lean_array_get_size(v___x_3321_);
v___x_3324_ = lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__1(v___x_3321_, v___x_3322_, v___x_3323_);
lean_dec_ref(v___x_3321_);
if (lean_obj_tag(v_expectedBinders_x3f_3297_) == 1)
{
lean_object* v_val_3444_; lean_object* v___x_3446_; uint8_t v_isShared_3447_; uint8_t v_isSharedCheck_3542_; 
lean_del_object(v___x_3319_);
v_val_3444_ = lean_ctor_get(v_expectedBinders_x3f_3297_, 0);
v_isSharedCheck_3542_ = !lean_is_exclusive(v_expectedBinders_x3f_3297_);
if (v_isSharedCheck_3542_ == 0)
{
v___x_3446_ = v_expectedBinders_x3f_3297_;
v_isShared_3447_ = v_isSharedCheck_3542_;
goto v_resetjp_3445_;
}
else
{
lean_inc(v_val_3444_);
lean_dec(v_expectedBinders_x3f_3297_);
v___x_3446_ = lean_box(0);
v_isShared_3447_ = v_isSharedCheck_3542_;
goto v_resetjp_3445_;
}
v_resetjp_3445_:
{
lean_object* v___y_3449_; lean_object* v___y_3450_; lean_object* v___y_3451_; lean_object* v___y_3452_; lean_object* v___y_3453_; lean_object* v___y_3454_; lean_object* v___x_3521_; 
lean_inc_ref(v___f_3298_);
lean_inc(v___y_3307_);
lean_inc_ref(v___y_3306_);
lean_inc(v___y_3305_);
lean_inc_ref(v___y_3304_);
lean_inc(v___y_3303_);
lean_inc_ref(v___y_3302_);
v___x_3521_ = lean_apply_7(v___f_3298_, v___y_3302_, v___y_3303_, v___y_3304_, v___y_3305_, v___y_3306_, v___y_3307_, lean_box(0));
if (lean_obj_tag(v___x_3521_) == 0)
{
lean_object* v_a_3522_; uint8_t v___x_3523_; 
v_a_3522_ = lean_ctor_get(v___x_3521_, 0);
lean_inc(v_a_3522_);
lean_dec_ref_known(v___x_3521_, 1);
v___x_3523_ = lean_unbox(v_a_3522_);
lean_dec(v_a_3522_);
if (v___x_3523_ == 0)
{
v___y_3449_ = v___y_3302_;
v___y_3450_ = v___y_3303_;
v___y_3451_ = v___y_3304_;
v___y_3452_ = v___y_3305_;
v___y_3453_ = v___y_3306_;
v___y_3454_ = v___y_3307_;
goto v___jp_3448_;
}
else
{
lean_object* v___x_3524_; lean_object* v___x_3525_; 
v___x_3524_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__15, &lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__15_once, _init_lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__15);
lean_inc(v___x_3299_);
v___x_3525_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg(v___x_3299_, v___x_3524_, v___y_3304_, v___y_3305_, v___y_3306_, v___y_3307_);
if (lean_obj_tag(v___x_3525_) == 0)
{
lean_dec_ref_known(v___x_3525_, 1);
v___y_3449_ = v___y_3302_;
v___y_3450_ = v___y_3303_;
v___y_3451_ = v___y_3304_;
v___y_3452_ = v___y_3305_;
v___y_3453_ = v___y_3306_;
v___y_3454_ = v___y_3307_;
goto v___jp_3448_;
}
else
{
lean_object* v_a_3526_; lean_object* v___x_3528_; uint8_t v_isShared_3529_; uint8_t v_isSharedCheck_3533_; 
lean_del_object(v___x_3446_);
lean_dec(v_val_3444_);
lean_dec_ref(v___x_3324_);
lean_del_object(v___x_3314_);
lean_dec_ref(v___f_3300_);
lean_dec(v___x_3299_);
lean_dec_ref(v___f_3298_);
v_a_3526_ = lean_ctor_get(v___x_3525_, 0);
v_isSharedCheck_3533_ = !lean_is_exclusive(v___x_3525_);
if (v_isSharedCheck_3533_ == 0)
{
v___x_3528_ = v___x_3525_;
v_isShared_3529_ = v_isSharedCheck_3533_;
goto v_resetjp_3527_;
}
else
{
lean_inc(v_a_3526_);
lean_dec(v___x_3525_);
v___x_3528_ = lean_box(0);
v_isShared_3529_ = v_isSharedCheck_3533_;
goto v_resetjp_3527_;
}
v_resetjp_3527_:
{
lean_object* v___x_3531_; 
if (v_isShared_3529_ == 0)
{
v___x_3531_ = v___x_3528_;
goto v_reusejp_3530_;
}
else
{
lean_object* v_reuseFailAlloc_3532_; 
v_reuseFailAlloc_3532_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3532_, 0, v_a_3526_);
v___x_3531_ = v_reuseFailAlloc_3532_;
goto v_reusejp_3530_;
}
v_reusejp_3530_:
{
return v___x_3531_;
}
}
}
}
}
else
{
lean_object* v_a_3534_; lean_object* v___x_3536_; uint8_t v_isShared_3537_; uint8_t v_isSharedCheck_3541_; 
lean_del_object(v___x_3446_);
lean_dec(v_val_3444_);
lean_dec_ref(v___x_3324_);
lean_del_object(v___x_3314_);
lean_dec_ref(v___f_3300_);
lean_dec(v___x_3299_);
lean_dec_ref(v___f_3298_);
v_a_3534_ = lean_ctor_get(v___x_3521_, 0);
v_isSharedCheck_3541_ = !lean_is_exclusive(v___x_3521_);
if (v_isSharedCheck_3541_ == 0)
{
v___x_3536_ = v___x_3521_;
v_isShared_3537_ = v_isSharedCheck_3541_;
goto v_resetjp_3535_;
}
else
{
lean_inc(v_a_3534_);
lean_dec(v___x_3521_);
v___x_3536_ = lean_box(0);
v_isShared_3537_ = v_isSharedCheck_3541_;
goto v_resetjp_3535_;
}
v_resetjp_3535_:
{
lean_object* v___x_3539_; 
if (v_isShared_3537_ == 0)
{
v___x_3539_ = v___x_3536_;
goto v_reusejp_3538_;
}
else
{
lean_object* v_reuseFailAlloc_3540_; 
v_reuseFailAlloc_3540_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3540_, 0, v_a_3534_);
v___x_3539_ = v_reuseFailAlloc_3540_;
goto v_reusejp_3538_;
}
v_reusejp_3538_:
{
return v___x_3539_;
}
}
}
v___jp_3448_:
{
lean_object* v___x_3455_; 
lean_inc_ref(v___f_3300_);
lean_inc(v___y_3454_);
lean_inc_ref(v___y_3453_);
lean_inc(v___y_3452_);
lean_inc_ref(v___y_3451_);
lean_inc(v___y_3450_);
lean_inc_ref(v___y_3449_);
lean_inc_ref(v___x_3324_);
v___x_3455_ = lean_apply_8(v___f_3300_, v___x_3324_, v___y_3449_, v___y_3450_, v___y_3451_, v___y_3452_, v___y_3453_, v___y_3454_, lean_box(0));
if (lean_obj_tag(v___x_3455_) == 0)
{
lean_object* v_a_3456_; lean_object* v___x_3457_; 
v_a_3456_ = lean_ctor_get(v___x_3455_, 0);
lean_inc(v_a_3456_);
lean_dec_ref_known(v___x_3455_, 1);
lean_inc(v___y_3454_);
lean_inc_ref(v___y_3453_);
lean_inc(v___y_3452_);
lean_inc_ref(v___y_3451_);
lean_inc(v___y_3450_);
lean_inc_ref(v___y_3449_);
v___x_3457_ = lean_apply_8(v___f_3300_, v_val_3444_, v___y_3449_, v___y_3450_, v___y_3451_, v___y_3452_, v___y_3453_, v___y_3454_, lean_box(0));
if (lean_obj_tag(v___x_3457_) == 0)
{
lean_object* v_a_3458_; lean_object* v___x_3459_; 
v_a_3458_ = lean_ctor_get(v___x_3457_, 0);
lean_inc(v_a_3458_);
lean_dec_ref_known(v___x_3457_, 1);
lean_inc(v___y_3454_);
lean_inc_ref(v___y_3453_);
lean_inc(v___y_3452_);
lean_inc_ref(v___y_3451_);
lean_inc(v___y_3450_);
lean_inc_ref(v___y_3449_);
v___x_3459_ = lean_apply_7(v___f_3298_, v___y_3449_, v___y_3450_, v___y_3451_, v___y_3452_, v___y_3453_, v___y_3454_, lean_box(0));
if (lean_obj_tag(v___x_3459_) == 0)
{
lean_object* v_a_3460_; uint8_t v___x_3461_; 
v_a_3460_ = lean_ctor_get(v___x_3459_, 0);
lean_inc(v_a_3460_);
lean_dec_ref_known(v___x_3459_, 1);
v___x_3461_ = lean_unbox(v_a_3460_);
lean_dec(v_a_3460_);
if (v___x_3461_ == 0)
{
lean_del_object(v___x_3446_);
v___y_3397_ = v_a_3456_;
v___y_3398_ = v_a_3458_;
v___y_3399_ = v___y_3449_;
v___y_3400_ = v___y_3450_;
v___y_3401_ = v___y_3451_;
v___y_3402_ = v___y_3452_;
v___y_3403_ = v___y_3453_;
v___y_3404_ = v___y_3454_;
goto v___jp_3396_;
}
else
{
lean_object* v_paramNames_3462_; lean_object* v_expr_3463_; lean_object* v___x_3464_; lean_object* v___x_3465_; lean_object* v___x_3466_; lean_object* v___x_3467_; lean_object* v___x_3468_; lean_object* v___x_3469_; lean_object* v___x_3470_; lean_object* v___x_3471_; lean_object* v___x_3472_; lean_object* v___x_3473_; lean_object* v___x_3474_; lean_object* v___x_3475_; lean_object* v___x_3476_; lean_object* v___x_3477_; lean_object* v___x_3479_; 
v_paramNames_3462_ = lean_ctor_get(v_a_3456_, 0);
v_expr_3463_ = lean_ctor_get(v_a_3456_, 2);
v___x_3464_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__13, &lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__13_once, _init_lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__13);
lean_inc_ref(v_paramNames_3462_);
v___x_3465_ = lean_array_to_list(v_paramNames_3462_);
v___x_3466_ = lean_box(0);
v___x_3467_ = lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__4(v___x_3465_, v___x_3466_);
v___x_3468_ = l_Lean_MessageData_ofList(v___x_3467_);
v___x_3469_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3469_, 0, v___x_3464_);
lean_ctor_set(v___x_3469_, 1, v___x_3468_);
v___x_3470_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__6, &lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__6_once, _init_lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__6);
v___x_3471_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3471_, 0, v___x_3469_);
lean_ctor_set(v___x_3471_, 1, v___x_3470_);
v___x_3472_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__7, &lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__7);
v___x_3473_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3473_, 0, v___x_3471_);
lean_ctor_set(v___x_3473_, 1, v___x_3472_);
v___x_3474_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__9, &lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__9);
v___x_3475_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3475_, 0, v___x_3473_);
lean_ctor_set(v___x_3475_, 1, v___x_3474_);
v___x_3476_ = l_Lean_Meta_AbstractMVarsResult_numMVars(v_a_3456_);
v___x_3477_ = l_Nat_reprFast(v___x_3476_);
if (v_isShared_3447_ == 0)
{
lean_ctor_set_tag(v___x_3446_, 3);
lean_ctor_set(v___x_3446_, 0, v___x_3477_);
v___x_3479_ = v___x_3446_;
goto v_reusejp_3478_;
}
else
{
lean_object* v_reuseFailAlloc_3496_; 
v_reuseFailAlloc_3496_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3496_, 0, v___x_3477_);
v___x_3479_ = v_reuseFailAlloc_3496_;
goto v_reusejp_3478_;
}
v_reusejp_3478_:
{
lean_object* v___x_3480_; lean_object* v___x_3481_; lean_object* v___x_3482_; lean_object* v___x_3483_; lean_object* v___x_3484_; lean_object* v___x_3485_; lean_object* v___x_3486_; lean_object* v___x_3487_; 
v___x_3480_ = l_Lean_MessageData_ofFormat(v___x_3479_);
v___x_3481_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3481_, 0, v___x_3475_);
lean_ctor_set(v___x_3481_, 1, v___x_3480_);
v___x_3482_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__11, &lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__11_once, _init_lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__11);
v___x_3483_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3483_, 0, v___x_3481_);
lean_ctor_set(v___x_3483_, 1, v___x_3482_);
lean_inc_ref(v_expr_3463_);
v___x_3484_ = l_Lean_MessageData_ofExpr(v_expr_3463_);
v___x_3485_ = l_Lean_indentD(v___x_3484_);
v___x_3486_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3486_, 0, v___x_3483_);
lean_ctor_set(v___x_3486_, 1, v___x_3485_);
lean_inc(v___x_3299_);
v___x_3487_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg(v___x_3299_, v___x_3486_, v___y_3451_, v___y_3452_, v___y_3453_, v___y_3454_);
if (lean_obj_tag(v___x_3487_) == 0)
{
lean_dec_ref_known(v___x_3487_, 1);
v___y_3397_ = v_a_3456_;
v___y_3398_ = v_a_3458_;
v___y_3399_ = v___y_3449_;
v___y_3400_ = v___y_3450_;
v___y_3401_ = v___y_3451_;
v___y_3402_ = v___y_3452_;
v___y_3403_ = v___y_3453_;
v___y_3404_ = v___y_3454_;
goto v___jp_3396_;
}
else
{
lean_object* v_a_3488_; lean_object* v___x_3490_; uint8_t v_isShared_3491_; uint8_t v_isSharedCheck_3495_; 
lean_dec(v_a_3458_);
lean_dec(v_a_3456_);
lean_dec_ref(v___x_3324_);
lean_del_object(v___x_3314_);
lean_dec(v___x_3299_);
v_a_3488_ = lean_ctor_get(v___x_3487_, 0);
v_isSharedCheck_3495_ = !lean_is_exclusive(v___x_3487_);
if (v_isSharedCheck_3495_ == 0)
{
v___x_3490_ = v___x_3487_;
v_isShared_3491_ = v_isSharedCheck_3495_;
goto v_resetjp_3489_;
}
else
{
lean_inc(v_a_3488_);
lean_dec(v___x_3487_);
v___x_3490_ = lean_box(0);
v_isShared_3491_ = v_isSharedCheck_3495_;
goto v_resetjp_3489_;
}
v_resetjp_3489_:
{
lean_object* v___x_3493_; 
if (v_isShared_3491_ == 0)
{
v___x_3493_ = v___x_3490_;
goto v_reusejp_3492_;
}
else
{
lean_object* v_reuseFailAlloc_3494_; 
v_reuseFailAlloc_3494_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3494_, 0, v_a_3488_);
v___x_3493_ = v_reuseFailAlloc_3494_;
goto v_reusejp_3492_;
}
v_reusejp_3492_:
{
return v___x_3493_;
}
}
}
}
}
}
else
{
lean_object* v_a_3497_; lean_object* v___x_3499_; uint8_t v_isShared_3500_; uint8_t v_isSharedCheck_3504_; 
lean_dec(v_a_3458_);
lean_dec(v_a_3456_);
lean_del_object(v___x_3446_);
lean_dec_ref(v___x_3324_);
lean_del_object(v___x_3314_);
lean_dec(v___x_3299_);
v_a_3497_ = lean_ctor_get(v___x_3459_, 0);
v_isSharedCheck_3504_ = !lean_is_exclusive(v___x_3459_);
if (v_isSharedCheck_3504_ == 0)
{
v___x_3499_ = v___x_3459_;
v_isShared_3500_ = v_isSharedCheck_3504_;
goto v_resetjp_3498_;
}
else
{
lean_inc(v_a_3497_);
lean_dec(v___x_3459_);
v___x_3499_ = lean_box(0);
v_isShared_3500_ = v_isSharedCheck_3504_;
goto v_resetjp_3498_;
}
v_resetjp_3498_:
{
lean_object* v___x_3502_; 
if (v_isShared_3500_ == 0)
{
v___x_3502_ = v___x_3499_;
goto v_reusejp_3501_;
}
else
{
lean_object* v_reuseFailAlloc_3503_; 
v_reuseFailAlloc_3503_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3503_, 0, v_a_3497_);
v___x_3502_ = v_reuseFailAlloc_3503_;
goto v_reusejp_3501_;
}
v_reusejp_3501_:
{
return v___x_3502_;
}
}
}
}
else
{
lean_object* v_a_3505_; lean_object* v___x_3507_; uint8_t v_isShared_3508_; uint8_t v_isSharedCheck_3512_; 
lean_dec(v_a_3456_);
lean_del_object(v___x_3446_);
lean_dec_ref(v___x_3324_);
lean_del_object(v___x_3314_);
lean_dec(v___x_3299_);
lean_dec_ref(v___f_3298_);
v_a_3505_ = lean_ctor_get(v___x_3457_, 0);
v_isSharedCheck_3512_ = !lean_is_exclusive(v___x_3457_);
if (v_isSharedCheck_3512_ == 0)
{
v___x_3507_ = v___x_3457_;
v_isShared_3508_ = v_isSharedCheck_3512_;
goto v_resetjp_3506_;
}
else
{
lean_inc(v_a_3505_);
lean_dec(v___x_3457_);
v___x_3507_ = lean_box(0);
v_isShared_3508_ = v_isSharedCheck_3512_;
goto v_resetjp_3506_;
}
v_resetjp_3506_:
{
lean_object* v___x_3510_; 
if (v_isShared_3508_ == 0)
{
v___x_3510_ = v___x_3507_;
goto v_reusejp_3509_;
}
else
{
lean_object* v_reuseFailAlloc_3511_; 
v_reuseFailAlloc_3511_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3511_, 0, v_a_3505_);
v___x_3510_ = v_reuseFailAlloc_3511_;
goto v_reusejp_3509_;
}
v_reusejp_3509_:
{
return v___x_3510_;
}
}
}
}
else
{
lean_object* v_a_3513_; lean_object* v___x_3515_; uint8_t v_isShared_3516_; uint8_t v_isSharedCheck_3520_; 
lean_del_object(v___x_3446_);
lean_dec(v_val_3444_);
lean_dec_ref(v___x_3324_);
lean_del_object(v___x_3314_);
lean_dec_ref(v___f_3300_);
lean_dec(v___x_3299_);
lean_dec_ref(v___f_3298_);
v_a_3513_ = lean_ctor_get(v___x_3455_, 0);
v_isSharedCheck_3520_ = !lean_is_exclusive(v___x_3455_);
if (v_isSharedCheck_3520_ == 0)
{
v___x_3515_ = v___x_3455_;
v_isShared_3516_ = v_isSharedCheck_3520_;
goto v_resetjp_3514_;
}
else
{
lean_inc(v_a_3513_);
lean_dec(v___x_3455_);
v___x_3515_ = lean_box(0);
v_isShared_3516_ = v_isSharedCheck_3520_;
goto v_resetjp_3514_;
}
v_resetjp_3514_:
{
lean_object* v___x_3518_; 
if (v_isShared_3516_ == 0)
{
v___x_3518_ = v___x_3515_;
goto v_reusejp_3517_;
}
else
{
lean_object* v_reuseFailAlloc_3519_; 
v_reuseFailAlloc_3519_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3519_, 0, v_a_3513_);
v___x_3518_ = v_reuseFailAlloc_3519_;
goto v_reusejp_3517_;
}
v_reusejp_3517_:
{
return v___x_3518_;
}
}
}
}
}
}
else
{
uint8_t v___x_3543_; lean_object* v___x_3544_; lean_object* v___x_3545_; lean_object* v___x_3547_; 
lean_del_object(v___x_3314_);
lean_dec_ref(v___f_3300_);
lean_dec(v___x_3299_);
lean_dec_ref(v___f_3298_);
lean_dec(v_expectedBinders_x3f_3297_);
v___x_3543_ = 1;
v___x_3544_ = lean_box(v___x_3543_);
v___x_3545_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3545_, 0, v___x_3324_);
lean_ctor_set(v___x_3545_, 1, v___x_3544_);
if (v_isShared_3320_ == 0)
{
lean_ctor_set(v___x_3319_, 0, v___x_3545_);
v___x_3547_ = v___x_3319_;
goto v_reusejp_3546_;
}
else
{
lean_object* v_reuseFailAlloc_3548_; 
v_reuseFailAlloc_3548_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3548_, 0, v___x_3545_);
v___x_3547_ = v_reuseFailAlloc_3548_;
goto v_reusejp_3546_;
}
v_reusejp_3546_:
{
return v___x_3547_;
}
}
v___jp_3325_:
{
lean_object* v___x_3332_; lean_object* v___x_3333_; 
v___x_3332_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__2, &lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__2);
v___x_3333_ = lp_mathlib_Lean_logWarning___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__2(v___x_3332_, v___y_3326_, v___y_3327_, v___y_3328_, v___y_3329_, v___y_3330_, v___y_3331_);
if (lean_obj_tag(v___x_3333_) == 0)
{
lean_object* v___x_3335_; uint8_t v_isShared_3336_; uint8_t v_isSharedCheck_3345_; 
v_isSharedCheck_3345_ = !lean_is_exclusive(v___x_3333_);
if (v_isSharedCheck_3345_ == 0)
{
lean_object* v_unused_3346_; 
v_unused_3346_ = lean_ctor_get(v___x_3333_, 0);
lean_dec(v_unused_3346_);
v___x_3335_ = v___x_3333_;
v_isShared_3336_ = v_isSharedCheck_3345_;
goto v_resetjp_3334_;
}
else
{
lean_dec(v___x_3333_);
v___x_3335_ = lean_box(0);
v_isShared_3336_ = v_isSharedCheck_3345_;
goto v_resetjp_3334_;
}
v_resetjp_3334_:
{
uint8_t v___x_3337_; lean_object* v___x_3338_; lean_object* v___x_3340_; 
v___x_3337_ = 1;
v___x_3338_ = lean_box(v___x_3337_);
if (v_isShared_3315_ == 0)
{
lean_ctor_set(v___x_3314_, 1, v___x_3338_);
lean_ctor_set(v___x_3314_, 0, v___x_3324_);
v___x_3340_ = v___x_3314_;
goto v_reusejp_3339_;
}
else
{
lean_object* v_reuseFailAlloc_3344_; 
v_reuseFailAlloc_3344_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3344_, 0, v___x_3324_);
lean_ctor_set(v_reuseFailAlloc_3344_, 1, v___x_3338_);
v___x_3340_ = v_reuseFailAlloc_3344_;
goto v_reusejp_3339_;
}
v_reusejp_3339_:
{
lean_object* v___x_3342_; 
if (v_isShared_3336_ == 0)
{
lean_ctor_set(v___x_3335_, 0, v___x_3340_);
v___x_3342_ = v___x_3335_;
goto v_reusejp_3341_;
}
else
{
lean_object* v_reuseFailAlloc_3343_; 
v_reuseFailAlloc_3343_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3343_, 0, v___x_3340_);
v___x_3342_ = v_reuseFailAlloc_3343_;
goto v_reusejp_3341_;
}
v_reusejp_3341_:
{
return v___x_3342_;
}
}
}
}
else
{
lean_object* v_a_3347_; lean_object* v___x_3349_; uint8_t v_isShared_3350_; uint8_t v_isSharedCheck_3354_; 
lean_dec_ref(v___x_3324_);
lean_del_object(v___x_3314_);
v_a_3347_ = lean_ctor_get(v___x_3333_, 0);
v_isSharedCheck_3354_ = !lean_is_exclusive(v___x_3333_);
if (v_isSharedCheck_3354_ == 0)
{
v___x_3349_ = v___x_3333_;
v_isShared_3350_ = v_isSharedCheck_3354_;
goto v_resetjp_3348_;
}
else
{
lean_inc(v_a_3347_);
lean_dec(v___x_3333_);
v___x_3349_ = lean_box(0);
v_isShared_3350_ = v_isSharedCheck_3354_;
goto v_resetjp_3348_;
}
v_resetjp_3348_:
{
lean_object* v___x_3352_; 
if (v_isShared_3350_ == 0)
{
v___x_3352_ = v___x_3349_;
goto v_reusejp_3351_;
}
else
{
lean_object* v_reuseFailAlloc_3353_; 
v_reuseFailAlloc_3353_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3353_, 0, v_a_3347_);
v___x_3352_ = v_reuseFailAlloc_3353_;
goto v_reusejp_3351_;
}
v_reusejp_3351_:
{
return v___x_3352_;
}
}
}
}
v___jp_3355_:
{
lean_object* v_paramNames_3364_; lean_object* v_expr_3365_; lean_object* v_paramNames_3366_; lean_object* v_expr_3367_; lean_object* v___x_3368_; lean_object* v___x_3369_; uint8_t v___x_3370_; 
v_paramNames_3364_ = lean_ctor_get(v___y_3357_, 0);
v_expr_3365_ = lean_ctor_get(v___y_3357_, 2);
lean_inc_ref(v_expr_3365_);
v_paramNames_3366_ = lean_ctor_get(v___y_3356_, 0);
v_expr_3367_ = lean_ctor_get(v___y_3356_, 2);
lean_inc_ref(v_expr_3367_);
v___x_3368_ = lean_array_get_size(v_paramNames_3364_);
v___x_3369_ = lean_array_get_size(v_paramNames_3366_);
v___x_3370_ = lean_nat_dec_eq(v___x_3368_, v___x_3369_);
if (v___x_3370_ == 0)
{
lean_dec_ref(v_expr_3367_);
lean_dec_ref(v_expr_3365_);
lean_dec_ref(v___y_3357_);
lean_dec_ref(v___y_3356_);
v___y_3326_ = v___y_3358_;
v___y_3327_ = v___y_3359_;
v___y_3328_ = v___y_3360_;
v___y_3329_ = v___y_3361_;
v___y_3330_ = v___y_3362_;
v___y_3331_ = v___y_3363_;
goto v___jp_3325_;
}
else
{
uint8_t v___x_3371_; 
v___x_3371_ = lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__3___redArg(v_paramNames_3364_, v_paramNames_3366_, v___x_3368_);
if (v___x_3371_ == 0)
{
lean_dec_ref(v_expr_3367_);
lean_dec_ref(v_expr_3365_);
lean_dec_ref(v___y_3357_);
lean_dec_ref(v___y_3356_);
v___y_3326_ = v___y_3358_;
v___y_3327_ = v___y_3359_;
v___y_3328_ = v___y_3360_;
v___y_3329_ = v___y_3361_;
v___y_3330_ = v___y_3362_;
v___y_3331_ = v___y_3363_;
goto v___jp_3325_;
}
else
{
lean_object* v___x_3372_; lean_object* v___x_3373_; uint8_t v___x_3374_; 
v___x_3372_ = l_Lean_Meta_AbstractMVarsResult_numMVars(v___y_3357_);
lean_dec_ref(v___y_3357_);
v___x_3373_ = l_Lean_Meta_AbstractMVarsResult_numMVars(v___y_3356_);
lean_dec_ref(v___y_3356_);
v___x_3374_ = lean_nat_dec_eq(v___x_3372_, v___x_3373_);
lean_dec(v___x_3373_);
lean_dec(v___x_3372_);
if (v___x_3374_ == 0)
{
lean_dec_ref(v_expr_3367_);
lean_dec_ref(v_expr_3365_);
v___y_3326_ = v___y_3358_;
v___y_3327_ = v___y_3359_;
v___y_3328_ = v___y_3360_;
v___y_3329_ = v___y_3361_;
v___y_3330_ = v___y_3362_;
v___y_3331_ = v___y_3363_;
goto v___jp_3325_;
}
else
{
lean_object* v___x_3375_; 
v___x_3375_ = l_Lean_Meta_isExprDefEq(v_expr_3365_, v_expr_3367_, v___y_3360_, v___y_3361_, v___y_3362_, v___y_3363_);
if (lean_obj_tag(v___x_3375_) == 0)
{
lean_object* v_a_3376_; lean_object* v___x_3378_; uint8_t v_isShared_3379_; uint8_t v_isSharedCheck_3387_; 
v_a_3376_ = lean_ctor_get(v___x_3375_, 0);
v_isSharedCheck_3387_ = !lean_is_exclusive(v___x_3375_);
if (v_isSharedCheck_3387_ == 0)
{
v___x_3378_ = v___x_3375_;
v_isShared_3379_ = v_isSharedCheck_3387_;
goto v_resetjp_3377_;
}
else
{
lean_inc(v_a_3376_);
lean_dec(v___x_3375_);
v___x_3378_ = lean_box(0);
v_isShared_3379_ = v_isSharedCheck_3387_;
goto v_resetjp_3377_;
}
v_resetjp_3377_:
{
uint8_t v___x_3380_; 
v___x_3380_ = lean_unbox(v_a_3376_);
lean_dec(v_a_3376_);
if (v___x_3380_ == 0)
{
lean_del_object(v___x_3378_);
v___y_3326_ = v___y_3358_;
v___y_3327_ = v___y_3359_;
v___y_3328_ = v___y_3360_;
v___y_3329_ = v___y_3361_;
v___y_3330_ = v___y_3362_;
v___y_3331_ = v___y_3363_;
goto v___jp_3325_;
}
else
{
uint8_t v___x_3381_; lean_object* v___x_3382_; lean_object* v___x_3383_; lean_object* v___x_3385_; 
lean_del_object(v___x_3314_);
v___x_3381_ = 0;
v___x_3382_ = lean_box(v___x_3381_);
v___x_3383_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3383_, 0, v___x_3324_);
lean_ctor_set(v___x_3383_, 1, v___x_3382_);
if (v_isShared_3379_ == 0)
{
lean_ctor_set(v___x_3378_, 0, v___x_3383_);
v___x_3385_ = v___x_3378_;
goto v_reusejp_3384_;
}
else
{
lean_object* v_reuseFailAlloc_3386_; 
v_reuseFailAlloc_3386_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3386_, 0, v___x_3383_);
v___x_3385_ = v_reuseFailAlloc_3386_;
goto v_reusejp_3384_;
}
v_reusejp_3384_:
{
return v___x_3385_;
}
}
}
}
else
{
lean_object* v_a_3388_; lean_object* v___x_3390_; uint8_t v_isShared_3391_; uint8_t v_isSharedCheck_3395_; 
lean_dec_ref(v___x_3324_);
lean_del_object(v___x_3314_);
v_a_3388_ = lean_ctor_get(v___x_3375_, 0);
v_isSharedCheck_3395_ = !lean_is_exclusive(v___x_3375_);
if (v_isSharedCheck_3395_ == 0)
{
v___x_3390_ = v___x_3375_;
v_isShared_3391_ = v_isSharedCheck_3395_;
goto v_resetjp_3389_;
}
else
{
lean_inc(v_a_3388_);
lean_dec(v___x_3375_);
v___x_3390_ = lean_box(0);
v_isShared_3391_ = v_isSharedCheck_3395_;
goto v_resetjp_3389_;
}
v_resetjp_3389_:
{
lean_object* v___x_3393_; 
if (v_isShared_3391_ == 0)
{
v___x_3393_ = v___x_3390_;
goto v_reusejp_3392_;
}
else
{
lean_object* v_reuseFailAlloc_3394_; 
v_reuseFailAlloc_3394_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3394_, 0, v_a_3388_);
v___x_3393_ = v_reuseFailAlloc_3394_;
goto v_reusejp_3392_;
}
v_reusejp_3392_:
{
return v___x_3393_;
}
}
}
}
}
}
}
v___jp_3396_:
{
lean_object* v_options_3405_; uint8_t v_hasTrace_3406_; 
v_options_3405_ = lean_ctor_get(v___y_3403_, 2);
v_hasTrace_3406_ = lean_ctor_get_uint8(v_options_3405_, sizeof(void*)*1);
if (v_hasTrace_3406_ == 0)
{
lean_dec(v___x_3299_);
v___y_3356_ = v___y_3398_;
v___y_3357_ = v___y_3397_;
v___y_3358_ = v___y_3399_;
v___y_3359_ = v___y_3400_;
v___y_3360_ = v___y_3401_;
v___y_3361_ = v___y_3402_;
v___y_3362_ = v___y_3403_;
v___y_3363_ = v___y_3404_;
goto v___jp_3355_;
}
else
{
lean_object* v_inheritedTraceOptions_3407_; lean_object* v___x_3408_; lean_object* v___x_3409_; uint8_t v___x_3410_; 
v_inheritedTraceOptions_3407_ = lean_ctor_get(v___y_3403_, 13);
v___x_3408_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__7));
lean_inc(v___x_3299_);
v___x_3409_ = l_Lean_Name_append(v___x_3408_, v___x_3299_);
v___x_3410_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3407_, v_options_3405_, v___x_3409_);
lean_dec(v___x_3409_);
if (v___x_3410_ == 0)
{
lean_dec(v___x_3299_);
v___y_3356_ = v___y_3398_;
v___y_3357_ = v___y_3397_;
v___y_3358_ = v___y_3399_;
v___y_3359_ = v___y_3400_;
v___y_3360_ = v___y_3401_;
v___y_3361_ = v___y_3402_;
v___y_3362_ = v___y_3403_;
v___y_3363_ = v___y_3404_;
goto v___jp_3355_;
}
else
{
lean_object* v_paramNames_3411_; lean_object* v_expr_3412_; lean_object* v___x_3413_; lean_object* v___x_3414_; lean_object* v___x_3415_; lean_object* v___x_3416_; lean_object* v___x_3417_; lean_object* v___x_3418_; lean_object* v___x_3419_; lean_object* v___x_3420_; lean_object* v___x_3421_; lean_object* v___x_3422_; lean_object* v___x_3423_; lean_object* v___x_3424_; lean_object* v___x_3425_; lean_object* v___x_3426_; lean_object* v___x_3427_; lean_object* v___x_3428_; lean_object* v___x_3429_; lean_object* v___x_3430_; lean_object* v___x_3431_; lean_object* v___x_3432_; lean_object* v___x_3433_; lean_object* v___x_3434_; lean_object* v___x_3435_; 
v_paramNames_3411_ = lean_ctor_get(v___y_3398_, 0);
v_expr_3412_ = lean_ctor_get(v___y_3398_, 2);
v___x_3413_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__4, &lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__4_once, _init_lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__4);
lean_inc_ref(v_paramNames_3411_);
v___x_3414_ = lean_array_to_list(v_paramNames_3411_);
v___x_3415_ = lean_box(0);
v___x_3416_ = lp_mathlib_List_mapTR_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__4(v___x_3414_, v___x_3415_);
v___x_3417_ = l_Lean_MessageData_ofList(v___x_3416_);
v___x_3418_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3418_, 0, v___x_3413_);
lean_ctor_set(v___x_3418_, 1, v___x_3417_);
v___x_3419_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__6, &lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__6_once, _init_lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__6);
v___x_3420_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3420_, 0, v___x_3418_);
lean_ctor_set(v___x_3420_, 1, v___x_3419_);
v___x_3421_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__7, &lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__7_once, _init_lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__7);
v___x_3422_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3422_, 0, v___x_3420_);
lean_ctor_set(v___x_3422_, 1, v___x_3421_);
v___x_3423_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__9, &lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__9_once, _init_lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__9);
v___x_3424_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3424_, 0, v___x_3422_);
lean_ctor_set(v___x_3424_, 1, v___x_3423_);
v___x_3425_ = l_Lean_Meta_AbstractMVarsResult_numMVars(v___y_3398_);
v___x_3426_ = l_Nat_reprFast(v___x_3425_);
v___x_3427_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3427_, 0, v___x_3426_);
v___x_3428_ = l_Lean_MessageData_ofFormat(v___x_3427_);
v___x_3429_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3429_, 0, v___x_3424_);
lean_ctor_set(v___x_3429_, 1, v___x_3428_);
v___x_3430_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__11, &lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__11_once, _init_lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___closed__11);
v___x_3431_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3431_, 0, v___x_3429_);
lean_ctor_set(v___x_3431_, 1, v___x_3430_);
lean_inc_ref(v_expr_3412_);
v___x_3432_ = l_Lean_MessageData_ofExpr(v_expr_3412_);
v___x_3433_ = l_Lean_indentD(v___x_3432_);
v___x_3434_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3434_, 0, v___x_3431_);
lean_ctor_set(v___x_3434_, 1, v___x_3433_);
v___x_3435_ = lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg(v___x_3299_, v___x_3434_, v___y_3401_, v___y_3402_, v___y_3403_, v___y_3404_);
if (lean_obj_tag(v___x_3435_) == 0)
{
lean_dec_ref_known(v___x_3435_, 1);
v___y_3356_ = v___y_3398_;
v___y_3357_ = v___y_3397_;
v___y_3358_ = v___y_3399_;
v___y_3359_ = v___y_3400_;
v___y_3360_ = v___y_3401_;
v___y_3361_ = v___y_3402_;
v___y_3362_ = v___y_3403_;
v___y_3363_ = v___y_3404_;
goto v___jp_3355_;
}
else
{
lean_object* v_a_3436_; lean_object* v___x_3438_; uint8_t v_isShared_3439_; uint8_t v_isSharedCheck_3443_; 
lean_dec_ref(v___y_3398_);
lean_dec_ref(v___y_3397_);
lean_dec_ref(v___x_3324_);
lean_del_object(v___x_3314_);
v_a_3436_ = lean_ctor_get(v___x_3435_, 0);
v_isSharedCheck_3443_ = !lean_is_exclusive(v___x_3435_);
if (v_isSharedCheck_3443_ == 0)
{
v___x_3438_ = v___x_3435_;
v_isShared_3439_ = v_isSharedCheck_3443_;
goto v_resetjp_3437_;
}
else
{
lean_inc(v_a_3436_);
lean_dec(v___x_3435_);
v___x_3438_ = lean_box(0);
v_isShared_3439_ = v_isSharedCheck_3443_;
goto v_resetjp_3437_;
}
v_resetjp_3437_:
{
lean_object* v___x_3441_; 
if (v_isShared_3439_ == 0)
{
v___x_3441_ = v___x_3438_;
goto v_reusejp_3440_;
}
else
{
lean_object* v_reuseFailAlloc_3442_; 
v_reuseFailAlloc_3442_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3442_, 0, v_a_3436_);
v___x_3441_ = v_reuseFailAlloc_3442_;
goto v_reusejp_3440_;
}
v_reusejp_3440_:
{
return v___x_3441_;
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
lean_object* v_a_3551_; lean_object* v___x_3553_; uint8_t v_isShared_3554_; uint8_t v_isSharedCheck_3558_; 
lean_del_object(v___x_3314_);
lean_dec(v_snd_3312_);
lean_dec(v_fst_3311_);
lean_dec_ref(v___f_3300_);
lean_dec(v___x_3299_);
lean_dec_ref(v___f_3298_);
lean_dec(v_expectedBinders_x3f_3297_);
v_a_3551_ = lean_ctor_get(v___x_3317_, 0);
v_isSharedCheck_3558_ = !lean_is_exclusive(v___x_3317_);
if (v_isSharedCheck_3558_ == 0)
{
v___x_3553_ = v___x_3317_;
v_isShared_3554_ = v_isSharedCheck_3558_;
goto v_resetjp_3552_;
}
else
{
lean_inc(v_a_3551_);
lean_dec(v___x_3317_);
v___x_3553_ = lean_box(0);
v_isShared_3554_ = v_isSharedCheck_3558_;
goto v_resetjp_3552_;
}
v_resetjp_3552_:
{
lean_object* v___x_3556_; 
if (v_isShared_3554_ == 0)
{
v___x_3556_ = v___x_3553_;
goto v_reusejp_3555_;
}
else
{
lean_object* v_reuseFailAlloc_3557_; 
v_reuseFailAlloc_3557_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3557_, 0, v_a_3551_);
v___x_3556_ = v_reuseFailAlloc_3557_;
goto v_reusejp_3555_;
}
v_reusejp_3555_:
{
return v___x_3556_;
}
}
}
}
}
else
{
lean_object* v_a_3560_; lean_object* v___x_3562_; uint8_t v_isShared_3563_; uint8_t v_isSharedCheck_3567_; 
lean_dec_ref(v___f_3300_);
lean_dec(v___x_3299_);
lean_dec_ref(v___f_3298_);
lean_dec(v_expectedBinders_x3f_3297_);
lean_dec_ref(v___f_3296_);
v_a_3560_ = lean_ctor_get(v___x_3309_, 0);
v_isSharedCheck_3567_ = !lean_is_exclusive(v___x_3309_);
if (v_isSharedCheck_3567_ == 0)
{
v___x_3562_ = v___x_3309_;
v_isShared_3563_ = v_isSharedCheck_3567_;
goto v_resetjp_3561_;
}
else
{
lean_inc(v_a_3560_);
lean_dec(v___x_3309_);
v___x_3562_ = lean_box(0);
v_isShared_3563_ = v_isSharedCheck_3567_;
goto v_resetjp_3561_;
}
v_resetjp_3561_:
{
lean_object* v___x_3565_; 
if (v_isShared_3563_ == 0)
{
v___x_3565_ = v___x_3562_;
goto v_reusejp_3564_;
}
else
{
lean_object* v_reuseFailAlloc_3566_; 
v_reuseFailAlloc_3566_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3566_, 0, v_a_3560_);
v___x_3565_ = v_reuseFailAlloc_3566_;
goto v_reusejp_3564_;
}
v_reusejp_3564_:
{
return v___x_3565_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___boxed(lean_object* v___x_3568_, lean_object* v_checkRedundant_3569_, lean_object* v_binders_3570_, lean_object* v___f_3571_, lean_object* v_expectedBinders_x3f_3572_, lean_object* v___f_3573_, lean_object* v___x_3574_, lean_object* v___f_3575_, lean_object* v_x_3576_, lean_object* v___y_3577_, lean_object* v___y_3578_, lean_object* v___y_3579_, lean_object* v___y_3580_, lean_object* v___y_3581_, lean_object* v___y_3582_, lean_object* v___y_3583_){
_start:
{
uint8_t v_checkRedundant_boxed_3584_; lean_object* v_res_3585_; 
v_checkRedundant_boxed_3584_ = lean_unbox(v_checkRedundant_3569_);
v_res_3585_ = lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5(v___x_3568_, v_checkRedundant_boxed_3584_, v_binders_3570_, v___f_3571_, v_expectedBinders_x3f_3572_, v___f_3573_, v___x_3574_, v___f_3575_, v_x_3576_, v___y_3577_, v___y_3578_, v___y_3579_, v___y_3580_, v___y_3581_, v___y_3582_);
lean_dec(v___y_3582_);
lean_dec_ref(v___y_3581_);
lean_dec(v___y_3580_);
lean_dec_ref(v___y_3579_);
lean_dec(v___y_3578_);
lean_dec_ref(v___y_3577_);
lean_dec_ref(v_x_3576_);
return v_res_3585_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5_spec__7_spec__8___redArg(lean_object* v_msgData_3586_, lean_object* v_macroStack_3587_, lean_object* v___y_3588_){
_start:
{
lean_object* v___x_3590_; lean_object* v_scopes_3591_; lean_object* v___x_3592_; lean_object* v___x_3593_; lean_object* v_opts_3594_; lean_object* v___x_3595_; uint8_t v___x_3596_; 
v___x_3590_ = lean_st_ref_get(v___y_3588_);
v_scopes_3591_ = lean_ctor_get(v___x_3590_, 2);
lean_inc(v_scopes_3591_);
lean_dec(v___x_3590_);
v___x_3592_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_3593_ = l_List_head_x21___redArg(v___x_3592_, v_scopes_3591_);
lean_dec(v_scopes_3591_);
v_opts_3594_ = lean_ctor_get(v___x_3593_, 1);
lean_inc_ref(v_opts_3594_);
lean_dec(v___x_3593_);
v___x_3595_ = l_Lean_Elab_pp_macroStack;
v___x_3596_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__5(v_opts_3594_, v___x_3595_);
lean_dec_ref(v_opts_3594_);
if (v___x_3596_ == 0)
{
lean_object* v___x_3597_; 
lean_dec(v_macroStack_3587_);
v___x_3597_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3597_, 0, v_msgData_3586_);
return v___x_3597_;
}
else
{
if (lean_obj_tag(v_macroStack_3587_) == 0)
{
lean_object* v___x_3598_; 
v___x_3598_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3598_, 0, v_msgData_3586_);
return v___x_3598_;
}
else
{
lean_object* v_head_3599_; lean_object* v_after_3600_; lean_object* v___x_3602_; uint8_t v_isShared_3603_; uint8_t v_isSharedCheck_3615_; 
v_head_3599_ = lean_ctor_get(v_macroStack_3587_, 0);
lean_inc(v_head_3599_);
v_after_3600_ = lean_ctor_get(v_head_3599_, 1);
v_isSharedCheck_3615_ = !lean_is_exclusive(v_head_3599_);
if (v_isSharedCheck_3615_ == 0)
{
lean_object* v_unused_3616_; 
v_unused_3616_ = lean_ctor_get(v_head_3599_, 0);
lean_dec(v_unused_3616_);
v___x_3602_ = v_head_3599_;
v_isShared_3603_ = v_isSharedCheck_3615_;
goto v_resetjp_3601_;
}
else
{
lean_inc(v_after_3600_);
lean_dec(v_head_3599_);
v___x_3602_ = lean_box(0);
v_isShared_3603_ = v_isSharedCheck_3615_;
goto v_resetjp_3601_;
}
v_resetjp_3601_:
{
lean_object* v___x_3604_; lean_object* v___x_3606_; 
v___x_3604_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__6___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__6___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__6___closed__0);
if (v_isShared_3603_ == 0)
{
lean_ctor_set_tag(v___x_3602_, 7);
lean_ctor_set(v___x_3602_, 1, v___x_3604_);
lean_ctor_set(v___x_3602_, 0, v_msgData_3586_);
v___x_3606_ = v___x_3602_;
goto v_reusejp_3605_;
}
else
{
lean_object* v_reuseFailAlloc_3614_; 
v_reuseFailAlloc_3614_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3614_, 0, v_msgData_3586_);
lean_ctor_set(v_reuseFailAlloc_3614_, 1, v___x_3604_);
v___x_3606_ = v_reuseFailAlloc_3614_;
goto v_reusejp_3605_;
}
v_reusejp_3605_:
{
lean_object* v___x_3607_; lean_object* v___x_3608_; lean_object* v___x_3609_; lean_object* v___x_3610_; lean_object* v_msgData_3611_; lean_object* v___x_3612_; lean_object* v___x_3613_; 
v___x_3607_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4___redArg___closed__2);
v___x_3608_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3608_, 0, v___x_3606_);
lean_ctor_set(v___x_3608_, 1, v___x_3607_);
v___x_3609_ = l_Lean_MessageData_ofSyntax(v_after_3600_);
v___x_3610_ = l_Lean_indentD(v___x_3609_);
v_msgData_3611_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_3611_, 0, v___x_3608_);
lean_ctor_set(v_msgData_3611_, 1, v___x_3610_);
v___x_3612_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__6(v_msgData_3611_, v_macroStack_3587_);
v___x_3613_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3613_, 0, v___x_3612_);
return v___x_3613_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5_spec__7_spec__8___redArg___boxed(lean_object* v_msgData_3617_, lean_object* v_macroStack_3618_, lean_object* v___y_3619_, lean_object* v___y_3620_){
_start:
{
lean_object* v_res_3621_; 
v_res_3621_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5_spec__7_spec__8___redArg(v_msgData_3617_, v_macroStack_3618_, v___y_3619_);
lean_dec(v___y_3619_);
return v_res_3621_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__0(void){
_start:
{
lean_object* v___x_3622_; 
v___x_3622_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_3622_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__1(void){
_start:
{
lean_object* v___x_3623_; lean_object* v___x_3624_; 
v___x_3623_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__0);
v___x_3624_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3624_, 0, v___x_3623_);
return v___x_3624_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__2(void){
_start:
{
lean_object* v___x_3625_; lean_object* v___x_3626_; lean_object* v___x_3627_; 
v___x_3625_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__1);
v___x_3626_ = lean_unsigned_to_nat(0u);
v___x_3627_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_3627_, 0, v___x_3626_);
lean_ctor_set(v___x_3627_, 1, v___x_3626_);
lean_ctor_set(v___x_3627_, 2, v___x_3626_);
lean_ctor_set(v___x_3627_, 3, v___x_3626_);
lean_ctor_set(v___x_3627_, 4, v___x_3625_);
lean_ctor_set(v___x_3627_, 5, v___x_3625_);
lean_ctor_set(v___x_3627_, 6, v___x_3625_);
lean_ctor_set(v___x_3627_, 7, v___x_3625_);
lean_ctor_set(v___x_3627_, 8, v___x_3625_);
lean_ctor_set(v___x_3627_, 9, v___x_3625_);
return v___x_3627_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__3(void){
_start:
{
lean_object* v___x_3628_; lean_object* v___x_3629_; lean_object* v___x_3630_; 
v___x_3628_ = lean_unsigned_to_nat(32u);
v___x_3629_ = lean_mk_empty_array_with_capacity(v___x_3628_);
v___x_3630_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3630_, 0, v___x_3629_);
return v___x_3630_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__4(void){
_start:
{
size_t v___x_3631_; lean_object* v___x_3632_; lean_object* v___x_3633_; lean_object* v___x_3634_; lean_object* v___x_3635_; lean_object* v___x_3636_; 
v___x_3631_ = ((size_t)5ULL);
v___x_3632_ = lean_unsigned_to_nat(0u);
v___x_3633_ = lean_unsigned_to_nat(32u);
v___x_3634_ = lean_mk_empty_array_with_capacity(v___x_3633_);
v___x_3635_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__3);
v___x_3636_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_3636_, 0, v___x_3635_);
lean_ctor_set(v___x_3636_, 1, v___x_3634_);
lean_ctor_set(v___x_3636_, 2, v___x_3632_);
lean_ctor_set(v___x_3636_, 3, v___x_3632_);
lean_ctor_set_usize(v___x_3636_, 4, v___x_3631_);
return v___x_3636_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__5(void){
_start:
{
lean_object* v___x_3637_; lean_object* v___x_3638_; lean_object* v___x_3639_; lean_object* v___x_3640_; 
v___x_3637_ = lean_box(1);
v___x_3638_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__4);
v___x_3639_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__1);
v___x_3640_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3640_, 0, v___x_3639_);
lean_ctor_set(v___x_3640_, 1, v___x_3638_);
lean_ctor_set(v___x_3640_, 2, v___x_3637_);
return v___x_3640_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg(lean_object* v_msgData_3641_, lean_object* v___y_3642_){
_start:
{
lean_object* v___x_3644_; lean_object* v_env_3645_; lean_object* v___x_3646_; lean_object* v_scopes_3647_; lean_object* v___x_3648_; lean_object* v___x_3649_; lean_object* v_opts_3650_; lean_object* v___x_3651_; lean_object* v___x_3652_; lean_object* v___x_3653_; lean_object* v___x_3654_; lean_object* v___x_3655_; 
v___x_3644_ = lean_st_ref_get(v___y_3642_);
v_env_3645_ = lean_ctor_get(v___x_3644_, 0);
lean_inc_ref(v_env_3645_);
lean_dec(v___x_3644_);
v___x_3646_ = lean_st_ref_get(v___y_3642_);
v_scopes_3647_ = lean_ctor_get(v___x_3646_, 2);
lean_inc(v_scopes_3647_);
lean_dec(v___x_3646_);
v___x_3648_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_3649_ = l_List_head_x21___redArg(v___x_3648_, v_scopes_3647_);
lean_dec(v_scopes_3647_);
v_opts_3650_ = lean_ctor_get(v___x_3649_, 1);
lean_inc_ref(v_opts_3650_);
lean_dec(v___x_3649_);
v___x_3651_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__2);
v___x_3652_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___closed__5);
v___x_3653_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_3653_, 0, v_env_3645_);
lean_ctor_set(v___x_3653_, 1, v___x_3651_);
lean_ctor_set(v___x_3653_, 2, v___x_3652_);
lean_ctor_set(v___x_3653_, 3, v_opts_3650_);
v___x_3654_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_3654_, 0, v___x_3653_);
lean_ctor_set(v___x_3654_, 1, v_msgData_3641_);
v___x_3655_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3655_, 0, v___x_3654_);
return v___x_3655_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg___boxed(lean_object* v_msgData_3656_, lean_object* v___y_3657_, lean_object* v___y_3658_){
_start:
{
lean_object* v_res_3659_; 
v_res_3659_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg(v_msgData_3656_, v___y_3657_);
lean_dec(v___y_3657_);
return v_res_3659_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5_spec__7___redArg(lean_object* v_msg_3660_, lean_object* v___y_3661_, lean_object* v___y_3662_){
_start:
{
lean_object* v___x_3664_; 
v___x_3664_ = l_Lean_Elab_Command_getRef___redArg(v___y_3661_);
if (lean_obj_tag(v___x_3664_) == 0)
{
lean_object* v_a_3665_; lean_object* v_macroStack_3666_; lean_object* v___x_3667_; lean_object* v_a_3668_; lean_object* v___x_3669_; lean_object* v___x_3670_; lean_object* v_a_3671_; lean_object* v___x_3673_; uint8_t v_isShared_3674_; uint8_t v_isSharedCheck_3679_; 
v_a_3665_ = lean_ctor_get(v___x_3664_, 0);
lean_inc(v_a_3665_);
lean_dec_ref_known(v___x_3664_, 1);
v_macroStack_3666_ = lean_ctor_get(v___y_3661_, 4);
v___x_3667_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg(v_msg_3660_, v___y_3662_);
v_a_3668_ = lean_ctor_get(v___x_3667_, 0);
lean_inc(v_a_3668_);
lean_dec_ref(v___x_3667_);
v___x_3669_ = l_Lean_Elab_getBetterRef(v_a_3665_, v_macroStack_3666_);
lean_dec(v_a_3665_);
lean_inc(v_macroStack_3666_);
v___x_3670_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5_spec__7_spec__8___redArg(v_a_3668_, v_macroStack_3666_, v___y_3662_);
v_a_3671_ = lean_ctor_get(v___x_3670_, 0);
v_isSharedCheck_3679_ = !lean_is_exclusive(v___x_3670_);
if (v_isSharedCheck_3679_ == 0)
{
v___x_3673_ = v___x_3670_;
v_isShared_3674_ = v_isSharedCheck_3679_;
goto v_resetjp_3672_;
}
else
{
lean_inc(v_a_3671_);
lean_dec(v___x_3670_);
v___x_3673_ = lean_box(0);
v_isShared_3674_ = v_isSharedCheck_3679_;
goto v_resetjp_3672_;
}
v_resetjp_3672_:
{
lean_object* v___x_3675_; lean_object* v___x_3677_; 
v___x_3675_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3675_, 0, v___x_3669_);
lean_ctor_set(v___x_3675_, 1, v_a_3671_);
if (v_isShared_3674_ == 0)
{
lean_ctor_set_tag(v___x_3673_, 1);
lean_ctor_set(v___x_3673_, 0, v___x_3675_);
v___x_3677_ = v___x_3673_;
goto v_reusejp_3676_;
}
else
{
lean_object* v_reuseFailAlloc_3678_; 
v_reuseFailAlloc_3678_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3678_, 0, v___x_3675_);
v___x_3677_ = v_reuseFailAlloc_3678_;
goto v_reusejp_3676_;
}
v_reusejp_3676_:
{
return v___x_3677_;
}
}
}
else
{
lean_object* v_a_3680_; lean_object* v___x_3682_; uint8_t v_isShared_3683_; uint8_t v_isSharedCheck_3687_; 
lean_dec_ref(v_msg_3660_);
v_a_3680_ = lean_ctor_get(v___x_3664_, 0);
v_isSharedCheck_3687_ = !lean_is_exclusive(v___x_3664_);
if (v_isSharedCheck_3687_ == 0)
{
v___x_3682_ = v___x_3664_;
v_isShared_3683_ = v_isSharedCheck_3687_;
goto v_resetjp_3681_;
}
else
{
lean_inc(v_a_3680_);
lean_dec(v___x_3664_);
v___x_3682_ = lean_box(0);
v_isShared_3683_ = v_isSharedCheck_3687_;
goto v_resetjp_3681_;
}
v_resetjp_3681_:
{
lean_object* v___x_3685_; 
if (v_isShared_3683_ == 0)
{
v___x_3685_ = v___x_3682_;
goto v_reusejp_3684_;
}
else
{
lean_object* v_reuseFailAlloc_3686_; 
v_reuseFailAlloc_3686_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3686_, 0, v_a_3680_);
v___x_3685_ = v_reuseFailAlloc_3686_;
goto v_reusejp_3684_;
}
v_reusejp_3684_:
{
return v___x_3685_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5_spec__7___redArg___boxed(lean_object* v_msg_3688_, lean_object* v___y_3689_, lean_object* v___y_3690_, lean_object* v___y_3691_){
_start:
{
lean_object* v_res_3692_; 
v_res_3692_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5_spec__7___redArg(v_msg_3688_, v___y_3689_, v___y_3690_);
lean_dec(v___y_3690_);
lean_dec_ref(v___y_3689_);
return v_res_3692_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5___redArg(lean_object* v_ref_3693_, lean_object* v_msg_3694_, lean_object* v___y_3695_, lean_object* v___y_3696_){
_start:
{
lean_object* v___x_3698_; 
v___x_3698_ = l_Lean_Elab_Command_getRef___redArg(v___y_3695_);
if (lean_obj_tag(v___x_3698_) == 0)
{
lean_object* v_a_3699_; lean_object* v_fileName_3700_; lean_object* v_fileMap_3701_; lean_object* v_currRecDepth_3702_; lean_object* v_cmdPos_3703_; lean_object* v_macroStack_3704_; lean_object* v_quotContext_x3f_3705_; lean_object* v_currMacroScope_3706_; lean_object* v_snap_x3f_3707_; lean_object* v_cancelTk_x3f_3708_; uint8_t v_suppressElabErrors_3709_; lean_object* v_ref_3710_; lean_object* v___x_3711_; lean_object* v___x_3712_; 
v_a_3699_ = lean_ctor_get(v___x_3698_, 0);
lean_inc(v_a_3699_);
lean_dec_ref_known(v___x_3698_, 1);
v_fileName_3700_ = lean_ctor_get(v___y_3695_, 0);
v_fileMap_3701_ = lean_ctor_get(v___y_3695_, 1);
v_currRecDepth_3702_ = lean_ctor_get(v___y_3695_, 2);
v_cmdPos_3703_ = lean_ctor_get(v___y_3695_, 3);
v_macroStack_3704_ = lean_ctor_get(v___y_3695_, 4);
v_quotContext_x3f_3705_ = lean_ctor_get(v___y_3695_, 5);
v_currMacroScope_3706_ = lean_ctor_get(v___y_3695_, 6);
v_snap_x3f_3707_ = lean_ctor_get(v___y_3695_, 8);
v_cancelTk_x3f_3708_ = lean_ctor_get(v___y_3695_, 9);
v_suppressElabErrors_3709_ = lean_ctor_get_uint8(v___y_3695_, sizeof(void*)*10);
v_ref_3710_ = l_Lean_replaceRef(v_ref_3693_, v_a_3699_);
lean_dec(v_a_3699_);
lean_inc(v_cancelTk_x3f_3708_);
lean_inc(v_snap_x3f_3707_);
lean_inc(v_currMacroScope_3706_);
lean_inc(v_quotContext_x3f_3705_);
lean_inc(v_macroStack_3704_);
lean_inc(v_cmdPos_3703_);
lean_inc(v_currRecDepth_3702_);
lean_inc_ref(v_fileMap_3701_);
lean_inc_ref(v_fileName_3700_);
v___x_3711_ = lean_alloc_ctor(0, 10, 1);
lean_ctor_set(v___x_3711_, 0, v_fileName_3700_);
lean_ctor_set(v___x_3711_, 1, v_fileMap_3701_);
lean_ctor_set(v___x_3711_, 2, v_currRecDepth_3702_);
lean_ctor_set(v___x_3711_, 3, v_cmdPos_3703_);
lean_ctor_set(v___x_3711_, 4, v_macroStack_3704_);
lean_ctor_set(v___x_3711_, 5, v_quotContext_x3f_3705_);
lean_ctor_set(v___x_3711_, 6, v_currMacroScope_3706_);
lean_ctor_set(v___x_3711_, 7, v_ref_3710_);
lean_ctor_set(v___x_3711_, 8, v_snap_x3f_3707_);
lean_ctor_set(v___x_3711_, 9, v_cancelTk_x3f_3708_);
lean_ctor_set_uint8(v___x_3711_, sizeof(void*)*10, v_suppressElabErrors_3709_);
v___x_3712_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5_spec__7___redArg(v_msg_3694_, v___x_3711_, v___y_3696_);
lean_dec_ref_known(v___x_3711_, 10);
return v___x_3712_;
}
else
{
lean_object* v_a_3713_; lean_object* v___x_3715_; uint8_t v_isShared_3716_; uint8_t v_isSharedCheck_3720_; 
lean_dec_ref(v_msg_3694_);
v_a_3713_ = lean_ctor_get(v___x_3698_, 0);
v_isSharedCheck_3720_ = !lean_is_exclusive(v___x_3698_);
if (v_isSharedCheck_3720_ == 0)
{
v___x_3715_ = v___x_3698_;
v_isShared_3716_ = v_isSharedCheck_3720_;
goto v_resetjp_3714_;
}
else
{
lean_inc(v_a_3713_);
lean_dec(v___x_3698_);
v___x_3715_ = lean_box(0);
v_isShared_3716_ = v_isSharedCheck_3720_;
goto v_resetjp_3714_;
}
v_resetjp_3714_:
{
lean_object* v___x_3718_; 
if (v_isShared_3716_ == 0)
{
v___x_3718_ = v___x_3715_;
goto v_reusejp_3717_;
}
else
{
lean_object* v_reuseFailAlloc_3719_; 
v_reuseFailAlloc_3719_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3719_, 0, v_a_3713_);
v___x_3718_ = v_reuseFailAlloc_3719_;
goto v_reusejp_3717_;
}
v_reusejp_3717_:
{
return v___x_3718_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5___redArg___boxed(lean_object* v_ref_3721_, lean_object* v_msg_3722_, lean_object* v___y_3723_, lean_object* v___y_3724_, lean_object* v___y_3725_){
_start:
{
lean_object* v_res_3726_; 
v_res_3726_ = lp_mathlib_Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5___redArg(v_ref_3721_, v_msg_3722_, v___y_3723_, v___y_3724_);
lean_dec(v___y_3724_);
lean_dec_ref(v___y_3723_);
lean_dec(v_ref_3721_);
return v_res_3726_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__6___closed__1(void){
_start:
{
lean_object* v___x_3728_; lean_object* v___x_3729_; 
v___x_3728_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__6___closed__0));
v___x_3729_ = l_Lean_stringToMessageData(v___x_3728_);
return v___x_3729_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__6(lean_object* v_as_3730_, size_t v_sz_3731_, size_t v_i_3732_, lean_object* v_b_3733_, lean_object* v___y_3734_, lean_object* v___y_3735_){
_start:
{
lean_object* v_a_3738_; uint8_t v___x_3742_; 
v___x_3742_ = lean_usize_dec_lt(v_i_3732_, v_sz_3731_);
if (v___x_3742_ == 0)
{
lean_object* v___x_3743_; 
v___x_3743_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3743_, 0, v_b_3733_);
return v___x_3743_;
}
else
{
lean_object* v___x_3744_; lean_object* v_a_3745_; lean_object* v___x_3746_; 
v___x_3744_ = lean_box(0);
v_a_3745_ = lean_array_uget_borrowed(v_as_3730_, v_i_3732_);
lean_inc(v_a_3745_);
v___x_3746_ = lp_mathlib_Mathlib_Command_Variable_bracketedBinderType(v_a_3745_);
if (lean_obj_tag(v___x_3746_) == 0)
{
lean_object* v___x_3747_; lean_object* v___x_3748_; 
v___x_3747_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__6___closed__1, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__6___closed__1_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__6___closed__1);
v___x_3748_ = lp_mathlib_Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5___redArg(v_a_3745_, v___x_3747_, v___y_3734_, v___y_3735_);
if (lean_obj_tag(v___x_3748_) == 0)
{
lean_dec_ref_known(v___x_3748_, 1);
v_a_3738_ = v___x_3744_;
goto v___jp_3737_;
}
else
{
return v___x_3748_;
}
}
else
{
lean_dec_ref_known(v___x_3746_, 1);
v_a_3738_ = v___x_3744_;
goto v___jp_3737_;
}
}
v___jp_3737_:
{
size_t v___x_3739_; size_t v___x_3740_; 
v___x_3739_ = ((size_t)1ULL);
v___x_3740_ = lean_usize_add(v_i_3732_, v___x_3739_);
v_i_3732_ = v___x_3740_;
v_b_3733_ = v_a_3738_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__6___boxed(lean_object* v_as_3749_, lean_object* v_sz_3750_, lean_object* v_i_3751_, lean_object* v_b_3752_, lean_object* v___y_3753_, lean_object* v___y_3754_, lean_object* v___y_3755_){
_start:
{
size_t v_sz_boxed_3756_; size_t v_i_boxed_3757_; lean_object* v_res_3758_; 
v_sz_boxed_3756_ = lean_unbox_usize(v_sz_3750_);
lean_dec(v_sz_3750_);
v_i_boxed_3757_ = lean_unbox_usize(v_i_3751_);
lean_dec(v_i_3751_);
v_res_3758_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__6(v_as_3749_, v_sz_boxed_3756_, v_i_boxed_3757_, v_b_3752_, v___y_3753_, v___y_3754_);
lean_dec(v___y_3754_);
lean_dec_ref(v___y_3753_);
lean_dec_ref(v_as_3749_);
return v_res_3758_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7(lean_object* v_cls_3759_, lean_object* v_msg_3760_, lean_object* v___y_3761_, lean_object* v___y_3762_){
_start:
{
lean_object* v___x_3764_; 
v___x_3764_ = l_Lean_Elab_Command_getRef___redArg(v___y_3761_);
if (lean_obj_tag(v___x_3764_) == 0)
{
lean_object* v_a_3765_; lean_object* v___x_3766_; lean_object* v_a_3767_; lean_object* v___x_3769_; uint8_t v_isShared_3770_; uint8_t v_isSharedCheck_3814_; 
v_a_3765_ = lean_ctor_get(v___x_3764_, 0);
lean_inc(v_a_3765_);
lean_dec_ref_known(v___x_3764_, 1);
v___x_3766_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg(v_msg_3760_, v___y_3762_);
v_a_3767_ = lean_ctor_get(v___x_3766_, 0);
v_isSharedCheck_3814_ = !lean_is_exclusive(v___x_3766_);
if (v_isSharedCheck_3814_ == 0)
{
v___x_3769_ = v___x_3766_;
v_isShared_3770_ = v_isSharedCheck_3814_;
goto v_resetjp_3768_;
}
else
{
lean_inc(v_a_3767_);
lean_dec(v___x_3766_);
v___x_3769_ = lean_box(0);
v_isShared_3770_ = v_isSharedCheck_3814_;
goto v_resetjp_3768_;
}
v_resetjp_3768_:
{
lean_object* v___x_3771_; lean_object* v_traceState_3772_; lean_object* v_env_3773_; lean_object* v_messages_3774_; lean_object* v_scopes_3775_; lean_object* v_usedQuotCtxts_3776_; lean_object* v_nextMacroScope_3777_; lean_object* v_maxRecDepth_3778_; lean_object* v_ngen_3779_; lean_object* v_auxDeclNGen_3780_; lean_object* v_infoState_3781_; lean_object* v_snapshotTasks_3782_; lean_object* v_prevLinterStates_3783_; lean_object* v___x_3785_; uint8_t v_isShared_3786_; uint8_t v_isSharedCheck_3813_; 
v___x_3771_ = lean_st_ref_take(v___y_3762_);
v_traceState_3772_ = lean_ctor_get(v___x_3771_, 9);
v_env_3773_ = lean_ctor_get(v___x_3771_, 0);
v_messages_3774_ = lean_ctor_get(v___x_3771_, 1);
v_scopes_3775_ = lean_ctor_get(v___x_3771_, 2);
v_usedQuotCtxts_3776_ = lean_ctor_get(v___x_3771_, 3);
v_nextMacroScope_3777_ = lean_ctor_get(v___x_3771_, 4);
v_maxRecDepth_3778_ = lean_ctor_get(v___x_3771_, 5);
v_ngen_3779_ = lean_ctor_get(v___x_3771_, 6);
v_auxDeclNGen_3780_ = lean_ctor_get(v___x_3771_, 7);
v_infoState_3781_ = lean_ctor_get(v___x_3771_, 8);
v_snapshotTasks_3782_ = lean_ctor_get(v___x_3771_, 10);
v_prevLinterStates_3783_ = lean_ctor_get(v___x_3771_, 11);
v_isSharedCheck_3813_ = !lean_is_exclusive(v___x_3771_);
if (v_isSharedCheck_3813_ == 0)
{
v___x_3785_ = v___x_3771_;
v_isShared_3786_ = v_isSharedCheck_3813_;
goto v_resetjp_3784_;
}
else
{
lean_inc(v_prevLinterStates_3783_);
lean_inc(v_snapshotTasks_3782_);
lean_inc(v_traceState_3772_);
lean_inc(v_infoState_3781_);
lean_inc(v_auxDeclNGen_3780_);
lean_inc(v_ngen_3779_);
lean_inc(v_maxRecDepth_3778_);
lean_inc(v_nextMacroScope_3777_);
lean_inc(v_usedQuotCtxts_3776_);
lean_inc(v_scopes_3775_);
lean_inc(v_messages_3774_);
lean_inc(v_env_3773_);
lean_dec(v___x_3771_);
v___x_3785_ = lean_box(0);
v_isShared_3786_ = v_isSharedCheck_3813_;
goto v_resetjp_3784_;
}
v_resetjp_3784_:
{
uint64_t v_tid_3787_; lean_object* v_traces_3788_; lean_object* v___x_3790_; uint8_t v_isShared_3791_; uint8_t v_isSharedCheck_3812_; 
v_tid_3787_ = lean_ctor_get_uint64(v_traceState_3772_, sizeof(void*)*1);
v_traces_3788_ = lean_ctor_get(v_traceState_3772_, 0);
v_isSharedCheck_3812_ = !lean_is_exclusive(v_traceState_3772_);
if (v_isSharedCheck_3812_ == 0)
{
v___x_3790_ = v_traceState_3772_;
v_isShared_3791_ = v_isSharedCheck_3812_;
goto v_resetjp_3789_;
}
else
{
lean_inc(v_traces_3788_);
lean_dec(v_traceState_3772_);
v___x_3790_ = lean_box(0);
v_isShared_3791_ = v_isSharedCheck_3812_;
goto v_resetjp_3789_;
}
v_resetjp_3789_:
{
lean_object* v___x_3792_; double v___x_3793_; uint8_t v___x_3794_; lean_object* v___x_3795_; lean_object* v___x_3796_; lean_object* v___x_3797_; lean_object* v___x_3798_; lean_object* v___x_3799_; lean_object* v___x_3800_; lean_object* v___x_3802_; 
v___x_3792_ = lean_box(0);
v___x_3793_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg___closed__0, &lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg___closed__0);
v___x_3794_ = 0;
v___x_3795_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg___closed__1));
v___x_3796_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_3796_, 0, v_cls_3759_);
lean_ctor_set(v___x_3796_, 1, v___x_3792_);
lean_ctor_set(v___x_3796_, 2, v___x_3795_);
lean_ctor_set_float(v___x_3796_, sizeof(void*)*3, v___x_3793_);
lean_ctor_set_float(v___x_3796_, sizeof(void*)*3 + 8, v___x_3793_);
lean_ctor_set_uint8(v___x_3796_, sizeof(void*)*3 + 16, v___x_3794_);
v___x_3797_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00Mathlib_Command_Variable_getSubproblem_spec__4___redArg___closed__2));
v___x_3798_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_3798_, 0, v___x_3796_);
lean_ctor_set(v___x_3798_, 1, v_a_3767_);
lean_ctor_set(v___x_3798_, 2, v___x_3797_);
v___x_3799_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3799_, 0, v_a_3765_);
lean_ctor_set(v___x_3799_, 1, v___x_3798_);
v___x_3800_ = l_Lean_PersistentArray_push___redArg(v_traces_3788_, v___x_3799_);
if (v_isShared_3791_ == 0)
{
lean_ctor_set(v___x_3790_, 0, v___x_3800_);
v___x_3802_ = v___x_3790_;
goto v_reusejp_3801_;
}
else
{
lean_object* v_reuseFailAlloc_3811_; 
v_reuseFailAlloc_3811_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_3811_, 0, v___x_3800_);
lean_ctor_set_uint64(v_reuseFailAlloc_3811_, sizeof(void*)*1, v_tid_3787_);
v___x_3802_ = v_reuseFailAlloc_3811_;
goto v_reusejp_3801_;
}
v_reusejp_3801_:
{
lean_object* v___x_3804_; 
if (v_isShared_3786_ == 0)
{
lean_ctor_set(v___x_3785_, 9, v___x_3802_);
v___x_3804_ = v___x_3785_;
goto v_reusejp_3803_;
}
else
{
lean_object* v_reuseFailAlloc_3810_; 
v_reuseFailAlloc_3810_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_3810_, 0, v_env_3773_);
lean_ctor_set(v_reuseFailAlloc_3810_, 1, v_messages_3774_);
lean_ctor_set(v_reuseFailAlloc_3810_, 2, v_scopes_3775_);
lean_ctor_set(v_reuseFailAlloc_3810_, 3, v_usedQuotCtxts_3776_);
lean_ctor_set(v_reuseFailAlloc_3810_, 4, v_nextMacroScope_3777_);
lean_ctor_set(v_reuseFailAlloc_3810_, 5, v_maxRecDepth_3778_);
lean_ctor_set(v_reuseFailAlloc_3810_, 6, v_ngen_3779_);
lean_ctor_set(v_reuseFailAlloc_3810_, 7, v_auxDeclNGen_3780_);
lean_ctor_set(v_reuseFailAlloc_3810_, 8, v_infoState_3781_);
lean_ctor_set(v_reuseFailAlloc_3810_, 9, v___x_3802_);
lean_ctor_set(v_reuseFailAlloc_3810_, 10, v_snapshotTasks_3782_);
lean_ctor_set(v_reuseFailAlloc_3810_, 11, v_prevLinterStates_3783_);
v___x_3804_ = v_reuseFailAlloc_3810_;
goto v_reusejp_3803_;
}
v_reusejp_3803_:
{
lean_object* v___x_3805_; lean_object* v___x_3806_; lean_object* v___x_3808_; 
v___x_3805_ = lean_st_ref_set(v___y_3762_, v___x_3804_);
v___x_3806_ = lean_box(0);
if (v_isShared_3770_ == 0)
{
lean_ctor_set(v___x_3769_, 0, v___x_3806_);
v___x_3808_ = v___x_3769_;
goto v_reusejp_3807_;
}
else
{
lean_object* v_reuseFailAlloc_3809_; 
v_reuseFailAlloc_3809_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3809_, 0, v___x_3806_);
v___x_3808_ = v_reuseFailAlloc_3809_;
goto v_reusejp_3807_;
}
v_reusejp_3807_:
{
return v___x_3808_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_3815_; lean_object* v___x_3817_; uint8_t v_isShared_3818_; uint8_t v_isSharedCheck_3822_; 
lean_dec_ref(v_msg_3760_);
lean_dec(v_cls_3759_);
v_a_3815_ = lean_ctor_get(v___x_3764_, 0);
v_isSharedCheck_3822_ = !lean_is_exclusive(v___x_3764_);
if (v_isSharedCheck_3822_ == 0)
{
v___x_3817_ = v___x_3764_;
v_isShared_3818_ = v_isSharedCheck_3822_;
goto v_resetjp_3816_;
}
else
{
lean_inc(v_a_3815_);
lean_dec(v___x_3764_);
v___x_3817_ = lean_box(0);
v_isShared_3818_ = v_isSharedCheck_3822_;
goto v_resetjp_3816_;
}
v_resetjp_3816_:
{
lean_object* v___x_3820_; 
if (v_isShared_3818_ == 0)
{
v___x_3820_ = v___x_3817_;
goto v_reusejp_3819_;
}
else
{
lean_object* v_reuseFailAlloc_3821_; 
v_reuseFailAlloc_3821_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3821_, 0, v_a_3815_);
v___x_3820_ = v_reuseFailAlloc_3821_;
goto v_reusejp_3819_;
}
v_reusejp_3819_:
{
return v___x_3820_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7___boxed(lean_object* v_cls_3823_, lean_object* v_msg_3824_, lean_object* v___y_3825_, lean_object* v___y_3826_, lean_object* v___y_3827_){
_start:
{
lean_object* v_res_3828_; 
v_res_3828_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7(v_cls_3823_, v_msg_3824_, v___y_3825_, v___y_3826_);
lean_dec(v___y_3826_);
lean_dec_ref(v___y_3825_);
return v_res_3828_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__8(void){
_start:
{
lean_object* v___x_3839_; lean_object* v___x_3840_; 
v___x_3839_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__7));
v___x_3840_ = l_Lean_stringToMessageData(v___x_3839_);
return v___x_3840_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__11(void){
_start:
{
lean_object* v___x_3844_; lean_object* v___x_3845_; 
v___x_3844_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__10));
v___x_3845_ = l_Lean_stringToMessageData(v___x_3844_);
return v___x_3845_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process(lean_object* v_stx_3846_, uint8_t v_checkRedundant_3847_, lean_object* v_binders_3848_, lean_object* v_expectedBinders_x3f_3849_, lean_object* v_a_3850_, lean_object* v_a_3851_){
_start:
{
lean_object* v___y_3854_; lean_object* v___y_3855_; uint8_t v___y_3856_; lean_object* v___y_3857_; lean_object* v___y_3858_; lean_object* v___y_3859_; lean_object* v___x_3871_; lean_object* v___x_3872_; lean_object* v___x_3873_; lean_object* v___x_3874_; lean_object* v_scopes_3875_; lean_object* v___x_3876_; lean_object* v___x_3877_; lean_object* v_opts_3878_; lean_object* v_scopes_3879_; lean_object* v___x_3880_; lean_object* v_opts_3881_; uint8_t v_hasTrace_3882_; lean_object* v___f_3883_; lean_object* v___f_3884_; lean_object* v_binders_3885_; lean_object* v___x_3886_; lean_object* v___x_3887_; lean_object* v___x_3888_; lean_object* v___x_3889_; lean_object* v___y_3891_; lean_object* v___y_3892_; lean_object* v___y_3893_; lean_object* v___y_3894_; uint8_t v___y_3895_; lean_object* v___y_3896_; lean_object* v___f_3923_; lean_object* v___x_3924_; lean_object* v___f_3925_; lean_object* v___y_3927_; lean_object* v___y_3928_; 
v___x_3871_ = lean_st_ref_get(v_a_3851_);
v___x_3872_ = l_Lean_inheritedTraceOptions;
v___x_3873_ = lean_st_ref_get(v___x_3872_);
v___x_3874_ = lean_st_ref_get(v_a_3851_);
v_scopes_3875_ = lean_ctor_get(v___x_3871_, 2);
lean_inc(v_scopes_3875_);
lean_dec(v___x_3871_);
v___x_3876_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_3877_ = l_List_head_x21___redArg(v___x_3876_, v_scopes_3875_);
lean_dec(v_scopes_3875_);
v_opts_3878_ = lean_ctor_get(v___x_3877_, 1);
lean_inc_ref(v_opts_3878_);
lean_dec(v___x_3877_);
v_scopes_3879_ = lean_ctor_get(v___x_3874_, 2);
lean_inc(v_scopes_3879_);
lean_dec(v___x_3874_);
v___x_3880_ = l_List_head_x21___redArg(v___x_3876_, v_scopes_3879_);
lean_dec(v_scopes_3879_);
v_opts_3881_ = lean_ctor_get(v___x_3880_, 1);
lean_inc_ref(v_opts_3881_);
lean_dec(v___x_3880_);
v_hasTrace_3882_ = lean_ctor_get_uint8(v_opts_3881_, sizeof(void*)*1);
v___f_3883_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__1));
v___f_3884_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__3));
v_binders_3885_ = lp_mathlib_Mathlib_Command_Variable_cleanBinders(v_binders_3848_);
v___x_3886_ = lp_mathlib_Mathlib_Command_Variable_variable_x3f_maxSteps;
v___x_3887_ = lp_mathlib_Lean_Option_get___at___00Mathlib_Command_Variable_completeBinders_x27_spec__7(v_opts_3878_, v___x_3886_);
lean_dec_ref(v_opts_3878_);
v___x_3888_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__0_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_));
v___x_3889_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn___closed__1_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_));
v___f_3923_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__9));
v___x_3924_ = lean_box(v_checkRedundant_3847_);
lean_inc_ref(v_binders_3885_);
lean_inc(v___x_3887_);
v___f_3925_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__5___boxed), 16, 8);
lean_closure_set(v___f_3925_, 0, v___x_3887_);
lean_closure_set(v___f_3925_, 1, v___x_3924_);
lean_closure_set(v___f_3925_, 2, v_binders_3885_);
lean_closure_set(v___f_3925_, 3, v___f_3883_);
lean_closure_set(v___f_3925_, 4, v_expectedBinders_x3f_3849_);
lean_closure_set(v___f_3925_, 5, v___f_3923_);
lean_closure_set(v___f_3925_, 6, v___x_3889_);
lean_closure_set(v___f_3925_, 7, v___f_3884_);
if (v_hasTrace_3882_ == 0)
{
lean_dec(v___x_3887_);
lean_dec_ref(v_opts_3881_);
lean_dec(v___x_3873_);
v___y_3927_ = v_a_3850_;
v___y_3928_ = v_a_3851_;
goto v___jp_3926_;
}
else
{
lean_object* v___x_3971_; uint8_t v___x_3972_; 
v___x_3971_ = lean_obj_once(&lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__8, &lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__8_once, _init_lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__8);
v___x_3972_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v___x_3873_, v_opts_3881_, v___x_3971_);
lean_dec_ref(v_opts_3881_);
lean_dec(v___x_3873_);
if (v___x_3972_ == 0)
{
lean_dec(v___x_3887_);
v___y_3927_ = v_a_3850_;
v___y_3928_ = v_a_3851_;
goto v___jp_3926_;
}
else
{
lean_object* v___x_3973_; lean_object* v___x_3974_; lean_object* v___x_3975_; lean_object* v___x_3976_; lean_object* v___x_3977_; lean_object* v___x_3978_; 
v___x_3973_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__11, &lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__11_once, _init_lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__11);
v___x_3974_ = l_Nat_reprFast(v___x_3887_);
v___x_3975_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3975_, 0, v___x_3974_);
v___x_3976_ = l_Lean_MessageData_ofFormat(v___x_3975_);
v___x_3977_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3977_, 0, v___x_3973_);
lean_ctor_set(v___x_3977_, 1, v___x_3976_);
v___x_3978_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7(v___x_3889_, v___x_3977_, v_a_3850_, v_a_3851_);
if (lean_obj_tag(v___x_3978_) == 0)
{
lean_dec_ref_known(v___x_3978_, 1);
v___y_3927_ = v_a_3850_;
v___y_3928_ = v_a_3851_;
goto v___jp_3926_;
}
else
{
lean_dec_ref(v___f_3925_);
lean_dec_ref(v_binders_3885_);
lean_dec(v_stx_3846_);
return v___x_3978_;
}
}
}
v___jp_3853_:
{
if (v___y_3856_ == 0)
{
lean_object* v___x_3860_; 
lean_dec(v___y_3855_);
lean_dec(v_stx_3846_);
v___x_3860_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3860_, 0, v___y_3857_);
return v___x_3860_;
}
else
{
lean_object* v___x_3861_; lean_object* v___x_3862_; lean_object* v___x_3863_; lean_object* v___x_3864_; lean_object* v___x_3865_; uint8_t v___x_3866_; lean_object* v___x_3867_; lean_object* v___x_3868_; lean_object* v___f_3869_; lean_object* v___x_3870_; 
lean_inc(v___y_3854_);
v___x_3861_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3861_, 0, v___y_3854_);
lean_ctor_set(v___x_3861_, 1, v___y_3855_);
v___x_3862_ = lean_box(0);
v___x_3863_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_3863_, 0, v___x_3861_);
lean_ctor_set(v___x_3863_, 1, v___x_3862_);
lean_ctor_set(v___x_3863_, 2, v___x_3862_);
lean_ctor_set(v___x_3863_, 3, v___x_3862_);
lean_ctor_set(v___x_3863_, 4, v___x_3862_);
lean_ctor_set(v___x_3863_, 5, v___x_3862_);
lean_inc(v_stx_3846_);
v___x_3864_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3864_, 0, v_stx_3846_);
v___x_3865_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__0));
v___x_3866_ = 4;
v___x_3867_ = l_Lean_MessageData_nil;
v___x_3868_ = lean_box(v___x_3866_);
v___f_3869_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___lam__3___boxed), 14, 7);
lean_closure_set(v___f_3869_, 0, v_stx_3846_);
lean_closure_set(v___f_3869_, 1, v___x_3863_);
lean_closure_set(v___f_3869_, 2, v___x_3864_);
lean_closure_set(v___f_3869_, 3, v___x_3865_);
lean_closure_set(v___f_3869_, 4, v___x_3862_);
lean_closure_set(v___f_3869_, 5, v___x_3868_);
lean_closure_set(v___f_3869_, 6, v___x_3867_);
v___x_3870_ = l_Lean_Elab_Command_liftTermElabM___redArg(v___f_3869_, v___y_3858_, v___y_3859_);
return v___x_3870_;
}
}
v___jp_3890_:
{
lean_object* v___x_3897_; lean_object* v___x_3898_; lean_object* v___x_3899_; lean_object* v___x_3900_; lean_object* v_scopes_3901_; lean_object* v___x_3902_; lean_object* v_opts_3903_; uint8_t v_hasTrace_3904_; lean_object* v___x_3905_; lean_object* v___x_3906_; lean_object* v___x_3907_; lean_object* v___x_3908_; lean_object* v___x_3909_; lean_object* v___x_3910_; lean_object* v___x_3911_; lean_object* v___x_3912_; lean_object* v___x_3913_; lean_object* v___x_3914_; lean_object* v___x_3915_; 
v___x_3897_ = lean_st_ref_get(v___x_3872_);
v___x_3898_ = lean_st_ref_get(v___y_3891_);
v___x_3899_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__0));
lean_inc_n(v___y_3893_, 5);
v___x_3900_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3900_, 0, v___y_3893_);
lean_ctor_set(v___x_3900_, 1, v___x_3888_);
v_scopes_3901_ = lean_ctor_get(v___x_3898_, 2);
lean_inc(v_scopes_3901_);
lean_dec(v___x_3898_);
v___x_3902_ = l_List_head_x21___redArg(v___x_3876_, v_scopes_3901_);
lean_dec(v_scopes_3901_);
v_opts_3903_ = lean_ctor_get(v___x_3902_, 1);
lean_inc_ref(v_opts_3903_);
lean_dec(v___x_3902_);
v_hasTrace_3904_ = lean_ctor_get_uint8(v_opts_3903_, sizeof(void*)*1);
v___x_3905_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__2));
v___x_3906_ = lean_obj_once(&lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__3, &lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__3_once, _init_lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__3);
v___x_3907_ = l_Array_append___redArg(v___x_3906_, v_binders_3885_);
lean_dec_ref(v_binders_3885_);
v___x_3908_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_3908_, 0, v___y_3893_);
lean_ctor_set(v___x_3908_, 1, v___x_3905_);
lean_ctor_set(v___x_3908_, 2, v___x_3907_);
v___x_3909_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__4));
v___x_3910_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3910_, 0, v___y_3893_);
lean_ctor_set(v___x_3910_, 1, v___x_3909_);
v___x_3911_ = l_Array_append___redArg(v___x_3906_, v___y_3892_);
lean_dec_ref(v___y_3892_);
v___x_3912_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_3912_, 0, v___y_3893_);
lean_ctor_set(v___x_3912_, 1, v___x_3905_);
lean_ctor_set(v___x_3912_, 2, v___x_3911_);
v___x_3913_ = l_Lean_Syntax_node2(v___y_3893_, v___x_3905_, v___x_3910_, v___x_3912_);
v___x_3914_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__6));
v___x_3915_ = l_Lean_Syntax_node3(v___y_3893_, v___x_3899_, v___x_3900_, v___x_3908_, v___x_3913_);
if (v_hasTrace_3904_ == 0)
{
lean_dec_ref(v_opts_3903_);
lean_dec(v___x_3897_);
v___y_3854_ = v___x_3914_;
v___y_3855_ = v___x_3915_;
v___y_3856_ = v___y_3895_;
v___y_3857_ = v___y_3896_;
v___y_3858_ = v___y_3894_;
v___y_3859_ = v___y_3891_;
goto v___jp_3853_;
}
else
{
lean_object* v___x_3916_; uint8_t v___x_3917_; 
v___x_3916_ = lean_obj_once(&lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__8, &lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__8_once, _init_lp_mathlib_Mathlib_Command_Variable_getSubproblem___lam__1___closed__8);
v___x_3917_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v___x_3897_, v_opts_3903_, v___x_3916_);
lean_dec_ref(v_opts_3903_);
lean_dec(v___x_3897_);
if (v___x_3917_ == 0)
{
v___y_3854_ = v___x_3914_;
v___y_3855_ = v___x_3915_;
v___y_3856_ = v___y_3895_;
v___y_3857_ = v___y_3896_;
v___y_3858_ = v___y_3894_;
v___y_3859_ = v___y_3891_;
goto v___jp_3853_;
}
else
{
lean_object* v___x_3918_; lean_object* v___x_3919_; lean_object* v___x_3920_; lean_object* v___x_3921_; lean_object* v___x_3922_; 
v___x_3918_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__8, &lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__8_once, _init_lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___closed__8);
lean_inc(v___x_3915_);
v___x_3919_ = l_Lean_MessageData_ofSyntax(v___x_3915_);
v___x_3920_ = l_Lean_indentD(v___x_3919_);
v___x_3921_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3921_, 0, v___x_3918_);
lean_ctor_set(v___x_3921_, 1, v___x_3920_);
v___x_3922_ = lp_mathlib_Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7(v___x_3889_, v___x_3921_, v___y_3894_, v___y_3891_);
if (lean_obj_tag(v___x_3922_) == 0)
{
lean_dec_ref_known(v___x_3922_, 1);
v___y_3854_ = v___x_3914_;
v___y_3855_ = v___x_3915_;
v___y_3856_ = v___y_3895_;
v___y_3857_ = v___y_3896_;
v___y_3858_ = v___y_3894_;
v___y_3859_ = v___y_3891_;
goto v___jp_3853_;
}
else
{
lean_dec(v___x_3915_);
lean_dec(v_stx_3846_);
return v___x_3922_;
}
}
}
}
v___jp_3926_:
{
lean_object* v___x_3929_; size_t v_sz_3930_; size_t v___x_3931_; lean_object* v___x_3932_; 
v___x_3929_ = lean_box(0);
v_sz_3930_ = lean_array_size(v_binders_3885_);
v___x_3931_ = ((size_t)0ULL);
v___x_3932_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__6(v_binders_3885_, v_sz_3930_, v___x_3931_, v___x_3929_, v___y_3927_, v___y_3928_);
if (lean_obj_tag(v___x_3932_) == 0)
{
lean_object* v___x_3933_; 
lean_dec_ref_known(v___x_3932_, 1);
v___x_3933_ = l_Lean_Elab_Command_runTermElabM___redArg(v___f_3925_, v___y_3927_, v___y_3928_);
if (lean_obj_tag(v___x_3933_) == 0)
{
lean_object* v_a_3934_; lean_object* v_fst_3935_; lean_object* v_snd_3936_; lean_object* v___x_3937_; 
v_a_3934_ = lean_ctor_get(v___x_3933_, 0);
lean_inc(v_a_3934_);
lean_dec_ref_known(v___x_3933_, 1);
v_fst_3935_ = lean_ctor_get(v_a_3934_, 0);
lean_inc(v_fst_3935_);
v_snd_3936_ = lean_ctor_get(v_a_3934_, 1);
lean_inc(v_snd_3936_);
lean_dec(v_a_3934_);
v___x_3937_ = lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope(v_fst_3935_, v___y_3927_, v___y_3928_);
if (lean_obj_tag(v___x_3937_) == 0)
{
lean_object* v___x_3938_; 
lean_dec_ref_known(v___x_3937_, 1);
v___x_3938_ = l_Lean_Elab_Command_getRef___redArg(v___y_3927_);
if (lean_obj_tag(v___x_3938_) == 0)
{
lean_object* v_a_3939_; lean_object* v___x_3940_; 
v_a_3939_ = lean_ctor_get(v___x_3938_, 0);
lean_inc(v_a_3939_);
lean_dec_ref_known(v___x_3938_, 1);
v___x_3940_ = l_Lean_Elab_Command_getCurrMacroScope___redArg(v___y_3927_);
if (lean_obj_tag(v___x_3940_) == 0)
{
lean_object* v_quotContext_x3f_3941_; uint8_t v___x_3942_; lean_object* v___x_3943_; 
lean_dec_ref_known(v___x_3940_, 1);
v_quotContext_x3f_3941_ = lean_ctor_get(v___y_3927_, 5);
v___x_3942_ = 0;
v___x_3943_ = l_Lean_SourceInfo_fromRef(v_a_3939_, v___x_3942_);
lean_dec(v_a_3939_);
if (lean_obj_tag(v_quotContext_x3f_3941_) == 0)
{
lean_object* v___x_3944_; uint8_t v___x_3945_; 
v___x_3944_ = lp_mathlib_Lean_getMainModule___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_extendScope_spec__0___redArg(v___y_3928_);
lean_dec_ref(v___x_3944_);
v___x_3945_ = lean_unbox(v_snd_3936_);
lean_dec(v_snd_3936_);
v___y_3891_ = v___y_3928_;
v___y_3892_ = v_fst_3935_;
v___y_3893_ = v___x_3943_;
v___y_3894_ = v___y_3927_;
v___y_3895_ = v___x_3945_;
v___y_3896_ = v___x_3929_;
goto v___jp_3890_;
}
else
{
uint8_t v___x_3946_; 
v___x_3946_ = lean_unbox(v_snd_3936_);
lean_dec(v_snd_3936_);
v___y_3891_ = v___y_3928_;
v___y_3892_ = v_fst_3935_;
v___y_3893_ = v___x_3943_;
v___y_3894_ = v___y_3927_;
v___y_3895_ = v___x_3946_;
v___y_3896_ = v___x_3929_;
goto v___jp_3890_;
}
}
else
{
lean_object* v_a_3947_; lean_object* v___x_3949_; uint8_t v_isShared_3950_; uint8_t v_isSharedCheck_3954_; 
lean_dec(v_a_3939_);
lean_dec(v_snd_3936_);
lean_dec(v_fst_3935_);
lean_dec_ref(v_binders_3885_);
lean_dec(v_stx_3846_);
v_a_3947_ = lean_ctor_get(v___x_3940_, 0);
v_isSharedCheck_3954_ = !lean_is_exclusive(v___x_3940_);
if (v_isSharedCheck_3954_ == 0)
{
v___x_3949_ = v___x_3940_;
v_isShared_3950_ = v_isSharedCheck_3954_;
goto v_resetjp_3948_;
}
else
{
lean_inc(v_a_3947_);
lean_dec(v___x_3940_);
v___x_3949_ = lean_box(0);
v_isShared_3950_ = v_isSharedCheck_3954_;
goto v_resetjp_3948_;
}
v_resetjp_3948_:
{
lean_object* v___x_3952_; 
if (v_isShared_3950_ == 0)
{
v___x_3952_ = v___x_3949_;
goto v_reusejp_3951_;
}
else
{
lean_object* v_reuseFailAlloc_3953_; 
v_reuseFailAlloc_3953_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3953_, 0, v_a_3947_);
v___x_3952_ = v_reuseFailAlloc_3953_;
goto v_reusejp_3951_;
}
v_reusejp_3951_:
{
return v___x_3952_;
}
}
}
}
else
{
lean_object* v_a_3955_; lean_object* v___x_3957_; uint8_t v_isShared_3958_; uint8_t v_isSharedCheck_3962_; 
lean_dec(v_snd_3936_);
lean_dec(v_fst_3935_);
lean_dec_ref(v_binders_3885_);
lean_dec(v_stx_3846_);
v_a_3955_ = lean_ctor_get(v___x_3938_, 0);
v_isSharedCheck_3962_ = !lean_is_exclusive(v___x_3938_);
if (v_isSharedCheck_3962_ == 0)
{
v___x_3957_ = v___x_3938_;
v_isShared_3958_ = v_isSharedCheck_3962_;
goto v_resetjp_3956_;
}
else
{
lean_inc(v_a_3955_);
lean_dec(v___x_3938_);
v___x_3957_ = lean_box(0);
v_isShared_3958_ = v_isSharedCheck_3962_;
goto v_resetjp_3956_;
}
v_resetjp_3956_:
{
lean_object* v___x_3960_; 
if (v_isShared_3958_ == 0)
{
v___x_3960_ = v___x_3957_;
goto v_reusejp_3959_;
}
else
{
lean_object* v_reuseFailAlloc_3961_; 
v_reuseFailAlloc_3961_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3961_, 0, v_a_3955_);
v___x_3960_ = v_reuseFailAlloc_3961_;
goto v_reusejp_3959_;
}
v_reusejp_3959_:
{
return v___x_3960_;
}
}
}
}
else
{
lean_dec(v_snd_3936_);
lean_dec(v_fst_3935_);
lean_dec_ref(v_binders_3885_);
lean_dec(v_stx_3846_);
return v___x_3937_;
}
}
else
{
lean_object* v_a_3963_; lean_object* v___x_3965_; uint8_t v_isShared_3966_; uint8_t v_isSharedCheck_3970_; 
lean_dec_ref(v_binders_3885_);
lean_dec(v_stx_3846_);
v_a_3963_ = lean_ctor_get(v___x_3933_, 0);
v_isSharedCheck_3970_ = !lean_is_exclusive(v___x_3933_);
if (v_isSharedCheck_3970_ == 0)
{
v___x_3965_ = v___x_3933_;
v_isShared_3966_ = v_isSharedCheck_3970_;
goto v_resetjp_3964_;
}
else
{
lean_inc(v_a_3963_);
lean_dec(v___x_3933_);
v___x_3965_ = lean_box(0);
v_isShared_3966_ = v_isSharedCheck_3970_;
goto v_resetjp_3964_;
}
v_resetjp_3964_:
{
lean_object* v___x_3968_; 
if (v_isShared_3966_ == 0)
{
v___x_3968_ = v___x_3965_;
goto v_reusejp_3967_;
}
else
{
lean_object* v_reuseFailAlloc_3969_; 
v_reuseFailAlloc_3969_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3969_, 0, v_a_3963_);
v___x_3968_ = v_reuseFailAlloc_3969_;
goto v_reusejp_3967_;
}
v_reusejp_3967_:
{
return v___x_3968_;
}
}
}
}
else
{
lean_dec_ref(v___f_3925_);
lean_dec_ref(v_binders_3885_);
lean_dec(v_stx_3846_);
return v___x_3932_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process___boxed(lean_object* v_stx_3979_, lean_object* v_checkRedundant_3980_, lean_object* v_binders_3981_, lean_object* v_expectedBinders_x3f_3982_, lean_object* v_a_3983_, lean_object* v_a_3984_, lean_object* v_a_3985_){
_start:
{
uint8_t v_checkRedundant_boxed_3986_; lean_object* v_res_3987_; 
v_checkRedundant_boxed_3986_ = lean_unbox(v_checkRedundant_3980_);
v_res_3987_ = lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process(v_stx_3979_, v_checkRedundant_boxed_3986_, v_binders_3981_, v_expectedBinders_x3f_3982_, v_a_3983_, v_a_3984_);
lean_dec(v_a_3984_);
lean_dec_ref(v_a_3983_);
lean_dec_ref(v_binders_3981_);
return v_res_3987_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__3(lean_object* v_xs_3988_, lean_object* v_ys_3989_, lean_object* v_hsz_3990_, lean_object* v_x_3991_, lean_object* v_x_3992_){
_start:
{
uint8_t v___x_3993_; 
v___x_3993_ = lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__3___redArg(v_xs_3988_, v_ys_3989_, v_x_3991_);
return v___x_3993_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__3___boxed(lean_object* v_xs_3994_, lean_object* v_ys_3995_, lean_object* v_hsz_3996_, lean_object* v_x_3997_, lean_object* v_x_3998_){
_start:
{
uint8_t v_res_3999_; lean_object* v_r_4000_; 
v_res_3999_ = lp_mathlib_Array_isEqvAux___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__3(v_xs_3994_, v_ys_3995_, v_hsz_3996_, v_x_3997_, v_x_3998_);
lean_dec_ref(v_ys_3995_);
lean_dec_ref(v_xs_3994_);
v_r_4000_ = lean_box(v_res_3999_);
return v_r_4000_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5(lean_object* v_00_u03b1_4001_, lean_object* v_ref_4002_, lean_object* v_msg_4003_, lean_object* v___y_4004_, lean_object* v___y_4005_){
_start:
{
lean_object* v___x_4007_; 
v___x_4007_ = lp_mathlib_Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5___redArg(v_ref_4002_, v_msg_4003_, v___y_4004_, v___y_4005_);
return v___x_4007_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5___boxed(lean_object* v_00_u03b1_4008_, lean_object* v_ref_4009_, lean_object* v_msg_4010_, lean_object* v___y_4011_, lean_object* v___y_4012_, lean_object* v___y_4013_){
_start:
{
lean_object* v_res_4014_; 
v_res_4014_ = lp_mathlib_Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5(v_00_u03b1_4008_, v_ref_4009_, v_msg_4010_, v___y_4011_, v___y_4012_);
lean_dec(v___y_4012_);
lean_dec_ref(v___y_4011_);
lean_dec(v_ref_4009_);
return v_res_4014_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10(lean_object* v_msgData_4015_, lean_object* v___y_4016_, lean_object* v___y_4017_){
_start:
{
lean_object* v___x_4019_; 
v___x_4019_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___redArg(v_msgData_4015_, v___y_4017_);
return v___x_4019_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10___boxed(lean_object* v_msgData_4020_, lean_object* v___y_4021_, lean_object* v___y_4022_, lean_object* v___y_4023_){
_start:
{
lean_object* v_res_4024_; 
v_res_4024_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__7_spec__10(v_msgData_4020_, v___y_4021_, v___y_4022_);
lean_dec(v___y_4022_);
lean_dec_ref(v___y_4021_);
return v_res_4024_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__2_spec__3(lean_object* v_msgData_4025_, uint8_t v_severity_4026_, uint8_t v_isSilent_4027_, lean_object* v___y_4028_, lean_object* v___y_4029_, lean_object* v___y_4030_, lean_object* v___y_4031_, lean_object* v___y_4032_, lean_object* v___y_4033_){
_start:
{
lean_object* v___x_4035_; 
v___x_4035_ = lp_mathlib_Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__2_spec__3___redArg(v_msgData_4025_, v_severity_4026_, v_isSilent_4027_, v___y_4030_, v___y_4031_, v___y_4032_, v___y_4033_);
return v___x_4035_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__2_spec__3___boxed(lean_object* v_msgData_4036_, lean_object* v_severity_4037_, lean_object* v_isSilent_4038_, lean_object* v___y_4039_, lean_object* v___y_4040_, lean_object* v___y_4041_, lean_object* v___y_4042_, lean_object* v___y_4043_, lean_object* v___y_4044_, lean_object* v___y_4045_){
_start:
{
uint8_t v_severity_boxed_4046_; uint8_t v_isSilent_boxed_4047_; lean_object* v_res_4048_; 
v_severity_boxed_4046_ = lean_unbox(v_severity_4037_);
v_isSilent_boxed_4047_ = lean_unbox(v_isSilent_4038_);
v_res_4048_ = lp_mathlib_Lean_log___at___00Lean_logWarning___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__2_spec__3(v_msgData_4036_, v_severity_boxed_4046_, v_isSilent_boxed_4047_, v___y_4039_, v___y_4040_, v___y_4041_, v___y_4042_, v___y_4043_, v___y_4044_);
lean_dec(v___y_4044_);
lean_dec_ref(v___y_4043_);
lean_dec(v___y_4042_);
lean_dec_ref(v___y_4041_);
lean_dec(v___y_4040_);
lean_dec_ref(v___y_4039_);
return v_res_4048_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5_spec__7(lean_object* v_00_u03b1_4049_, lean_object* v_msg_4050_, lean_object* v___y_4051_, lean_object* v___y_4052_){
_start:
{
lean_object* v___x_4054_; 
v___x_4054_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5_spec__7___redArg(v_msg_4050_, v___y_4051_, v___y_4052_);
return v___x_4054_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5_spec__7___boxed(lean_object* v_00_u03b1_4055_, lean_object* v_msg_4056_, lean_object* v___y_4057_, lean_object* v___y_4058_, lean_object* v___y_4059_){
_start:
{
lean_object* v_res_4060_; 
v_res_4060_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5_spec__7(v_00_u03b1_4055_, v_msg_4056_, v___y_4057_, v___y_4058_);
lean_dec(v___y_4058_);
lean_dec_ref(v___y_4057_);
return v_res_4060_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5_spec__7_spec__8(lean_object* v_msgData_4061_, lean_object* v_macroStack_4062_, lean_object* v___y_4063_, lean_object* v___y_4064_){
_start:
{
lean_object* v___x_4066_; 
v___x_4066_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5_spec__7_spec__8___redArg(v_msgData_4061_, v_macroStack_4062_, v___y_4064_);
return v___x_4066_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5_spec__7_spec__8___boxed(lean_object* v_msgData_4067_, lean_object* v_macroStack_4068_, lean_object* v___y_4069_, lean_object* v___y_4070_, lean_object* v___y_4071_){
_start:
{
lean_object* v_res_4072_; 
v_res_4072_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00__private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process_spec__5_spec__7_spec__8(v_msgData_4067_, v_macroStack_4068_, v___y_4069_, v___y_4070_);
lean_dec(v___y_4070_);
lean_dec_ref(v___y_4069_);
return v_res_4072_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Command_Variable_elabVariables_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_4073_; lean_object* v___x_4074_; lean_object* v___x_4075_; 
v___x_4073_ = lean_box(0);
v___x_4074_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_4075_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4075_, 0, v___x_4074_);
lean_ctor_set(v___x_4075_, 1, v___x_4073_);
return v___x_4075_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Command_Variable_elabVariables_spec__0___redArg(){
_start:
{
lean_object* v___x_4077_; lean_object* v___x_4078_; 
v___x_4077_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Command_Variable_elabVariables_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Command_Variable_elabVariables_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Command_Variable_elabVariables_spec__0___redArg___closed__0);
v___x_4078_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4078_, 0, v___x_4077_);
return v___x_4078_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Command_Variable_elabVariables_spec__0___redArg___boxed(lean_object* v___y_4079_){
_start:
{
lean_object* v_res_4080_; 
v_res_4080_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Command_Variable_elabVariables_spec__0___redArg();
return v_res_4080_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Command_Variable_elabVariables_spec__0(lean_object* v_00_u03b1_4081_, lean_object* v___y_4082_, lean_object* v___y_4083_){
_start:
{
lean_object* v___x_4085_; 
v___x_4085_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Command_Variable_elabVariables_spec__0___redArg();
return v___x_4085_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Command_Variable_elabVariables_spec__0___boxed(lean_object* v_00_u03b1_4086_, lean_object* v___y_4087_, lean_object* v___y_4088_, lean_object* v___y_4089_){
_start:
{
lean_object* v_res_4090_; 
v_res_4090_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Command_Variable_elabVariables_spec__0(v_00_u03b1_4086_, v___y_4087_, v___y_4088_);
lean_dec(v___y_4088_);
lean_dec_ref(v___y_4087_);
return v_res_4090_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_elabVariables(lean_object* v_stx_4091_, lean_object* v_a_4092_, lean_object* v_a_4093_){
_start:
{
lean_object* v___x_4095_; uint8_t v___x_4096_; 
v___x_4095_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_variable_x3f___closed__0));
lean_inc(v_stx_4091_);
v___x_4096_ = l_Lean_Syntax_isOfKind(v_stx_4091_, v___x_4095_);
if (v___x_4096_ == 0)
{
lean_object* v___x_4097_; 
lean_dec(v_stx_4091_);
v___x_4097_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Command_Variable_elabVariables_spec__0___redArg();
return v___x_4097_;
}
else
{
lean_object* v___x_4098_; lean_object* v___x_4099_; lean_object* v_expectedBinders_x3f_4101_; lean_object* v___y_4102_; lean_object* v___y_4103_; lean_object* v___x_4113_; lean_object* v___x_4114_; uint8_t v___x_4115_; 
v___x_4098_ = lean_unsigned_to_nat(1u);
v___x_4099_ = l_Lean_Syntax_getArg(v_stx_4091_, v___x_4098_);
v___x_4113_ = lean_unsigned_to_nat(2u);
v___x_4114_ = l_Lean_Syntax_getArg(v_stx_4091_, v___x_4113_);
v___x_4115_ = l_Lean_Syntax_isNone(v___x_4114_);
if (v___x_4115_ == 0)
{
uint8_t v___x_4116_; 
lean_inc(v___x_4114_);
v___x_4116_ = l_Lean_Syntax_matchesNull(v___x_4114_, v___x_4113_);
if (v___x_4116_ == 0)
{
lean_object* v___x_4117_; 
lean_dec(v___x_4114_);
lean_dec(v___x_4099_);
lean_dec(v_stx_4091_);
v___x_4117_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Command_Variable_elabVariables_spec__0___redArg();
return v___x_4117_;
}
else
{
lean_object* v___x_4118_; lean_object* v_expectedBinders_x3f_4119_; lean_object* v___x_4120_; 
v___x_4118_ = l_Lean_Syntax_getArg(v___x_4114_, v___x_4098_);
lean_dec(v___x_4114_);
v_expectedBinders_x3f_4119_ = l_Lean_Syntax_getArgs(v___x_4118_);
lean_dec(v___x_4118_);
v___x_4120_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4120_, 0, v_expectedBinders_x3f_4119_);
v_expectedBinders_x3f_4101_ = v___x_4120_;
v___y_4102_ = v_a_4092_;
v___y_4103_ = v_a_4093_;
goto v___jp_4100_;
}
}
else
{
lean_object* v___x_4121_; 
lean_dec(v___x_4114_);
v___x_4121_ = lean_box(0);
v_expectedBinders_x3f_4101_ = v___x_4121_;
v___y_4102_ = v_a_4092_;
v___y_4103_ = v_a_4093_;
goto v___jp_4100_;
}
v___jp_4100_:
{
lean_object* v___x_4104_; lean_object* v_scopes_4105_; lean_object* v___x_4106_; lean_object* v___x_4107_; lean_object* v_opts_4108_; lean_object* v_binders_4109_; lean_object* v___x_4110_; uint8_t v___x_4111_; lean_object* v___x_4112_; 
v___x_4104_ = lean_st_ref_get(v___y_4103_);
v_scopes_4105_ = lean_ctor_get(v___x_4104_, 2);
lean_inc(v_scopes_4105_);
lean_dec(v___x_4104_);
v___x_4106_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_4107_ = l_List_head_x21___redArg(v___x_4106_, v_scopes_4105_);
lean_dec(v_scopes_4105_);
v_opts_4108_ = lean_ctor_get(v___x_4107_, 1);
lean_inc_ref(v_opts_4108_);
lean_dec(v___x_4107_);
v_binders_4109_ = l_Lean_Syntax_getArgs(v___x_4099_);
lean_dec(v___x_4099_);
v___x_4110_ = lp_mathlib_Mathlib_Command_Variable_variable_x3f_checkRedundant;
v___x_4111_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Mathlib_Command_Variable_pendingActionableSynthMVar_spec__2_spec__2_spec__4_spec__5(v_opts_4108_, v___x_4110_);
lean_dec_ref(v_opts_4108_);
v___x_4112_ = lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_elabVariables_process(v_stx_4091_, v___x_4111_, v_binders_4109_, v_expectedBinders_x3f_4101_, v___y_4102_, v___y_4103_);
lean_dec_ref(v_binders_4109_);
return v___x_4112_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_elabVariables___boxed(lean_object* v_stx_4122_, lean_object* v_a_4123_, lean_object* v_a_4124_, lean_object* v_a_4125_){
_start:
{
lean_object* v_res_4126_; 
v_res_4126_ = lp_mathlib_Mathlib_Command_Variable_elabVariables(v_stx_4122_, v_a_4123_, v_a_4124_);
lean_dec(v_a_4124_);
lean_dec_ref(v_a_4123_);
return v_res_4126_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg(lean_object* v_stack_4152_){
_start:
{
lean_object* v___x_4153_; uint8_t v___x_4154_; 
v___x_4153_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__5));
lean_inc(v_stack_4152_);
v___x_4154_ = l_Lean_Syntax_Stack_matches(v_stack_4152_, v___x_4153_);
if (v___x_4154_ == 0)
{
lean_object* v___x_4155_; uint8_t v___x_4156_; 
v___x_4155_ = ((lean_object*)(lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___closed__8));
v___x_4156_ = l_Lean_Syntax_Stack_matches(v_stack_4152_, v___x_4155_);
return v___x_4156_;
}
else
{
lean_dec(v_stack_4152_);
return v___x_4154_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg___boxed(lean_object* v_stack_4157_){
_start:
{
uint8_t v_res_4158_; lean_object* v_r_4159_; 
v_res_4158_ = lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg(v_stack_4157_);
v_r_4159_ = lean_box(v_res_4158_);
return v_r_4159_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f(lean_object* v_x_4160_, lean_object* v_stack_4161_, lean_object* v_x_4162_){
_start:
{
uint8_t v___x_4163_; 
v___x_4163_ = lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___redArg(v_stack_4161_);
return v___x_4163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f___boxed(lean_object* v_x_4164_, lean_object* v_stack_4165_, lean_object* v_x_4166_){
_start:
{
uint8_t v_res_4167_; lean_object* v_r_4168_; 
v_res_4167_ = lp_mathlib_Mathlib_Command_Variable_ignorevariable_x3f(v_x_4164_, v_stack_4165_, v_x_4166_);
lean_dec_ref(v_x_4166_);
lean_dec(v_x_4164_);
v_r_4168_ = lean_box(v_res_4167_);
return v_r_4168_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Variable(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Meta_Tactic_TryThis(uint8_t builtin);
lean_object* runtime_initialize_Lean_Linter_UnusedVariables(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Variable(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Linter_UnusedVariables(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_1422233303____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_3395786082____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Command_Variable_variable_x3f_maxSteps = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Command_Variable_variable_x3f_maxSteps);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_1155143173____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Command_Variable_variable_x3f_checkRedundant = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Command_Variable_variable_x3f_checkRedundant);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Variable_0__Mathlib_Command_Variable_initFn_00___x40_Mathlib_Tactic_Variable_1020148838____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Mathlib_Command_Variable_variableAliasAttr = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Mathlib_Command_Variable_variableAliasAttr);
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_TryThis(uint8_t builtin);
lean_object* initialize_Lean_Linter_UnusedVariables(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Variable(uint8_t builtin) {
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
res = initialize_Lean_Meta_Tactic_TryThis(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Linter_UnusedVariables(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Variable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Variable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Variable(builtin);
}
#ifdef __cplusplus
}
#endif
